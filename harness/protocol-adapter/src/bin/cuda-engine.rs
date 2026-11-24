//! `cuda-engine` — the Engine Protocol v1 stdio binary.
//!
//! Reads NDJSON requests on stdin, writes NDJSON responses on stdout.
//!
//! WHICH BACKEND IT SERVES, in the order it decides:
//!
//! 1. THE RESIDENT ENGINE, whenever benchd hands the worker a resident socket
//!    (`BENCH_WORKER_RESIDENT_SOCKET`, or the legacy `DS4_RESIDENT_SOCKET`).
//!    This is the SCORED path. benchd spawns one of these per phase, and every
//!    one of them attaches to the single vLLM resident `tools/serve-up.sh`
//!    booted for the leg: THE MODEL IS LOADED ONCE PER LEG, and a phase costs a
//!    connect. Nothing is loaded in this process.
//! 2. the deterministic MOCK backend when no socket is named, so the protocol
//!    runs anywhere with no GPU. THE MOCK IS NOT INFERENCE, and this binary
//!    says so on stderr rather than leaving a reader to infer it from a
//!    suspiciously fast run.
//!
//! BOTH ANNOUNCE THEMSELVES. The hello's `backend`/`device` strings are what
//! benchd seals as `engine_backend`/`engine_device`, so they are the only place
//! a sealed artifact says which of the two produced the number. The mock says
//! `mock`/`none`; the resident-attached worker says `vllm-resident` with the
//! resident's `load_epoch` and identity.

use std::collections::BTreeMap;
use std::io::{self, BufWriter};
use std::path::Path;
use std::process::ExitCode;

use protocol_adapter::Adapter;

const L3_16_MIB: u64 = 16 * 1024 * 1024;

fn cuda_device() -> String {
    // setup.sh exports ENGINE_CUDA_ARCH for the box build. Keep the GB10
    // default for a direct Cargo build, but never label an explicitly targeted
    // build as sm_121 in the artifact benchd seals.
    format!(
        "cuda {}",
        option_env!("ENGINE_CUDA_ARCH").unwrap_or("sm_121")
    )
}

#[cfg(target_os = "linux")]
fn disable_coredump() {
    extern "C" {
        fn prctl(option: i32, arg2: usize, arg3: usize, arg4: usize, arg5: usize) -> i32;
    }
    // PR_SET_DUMPABLE = 4 -> 0 so SIGKILL/abort never invokes host apport (/var/crash).
    unsafe {
        let _ = prctl(4, 0, 0, 0, 0);
    }
}

#[cfg(not(target_os = "linux"))]
fn disable_coredump() {}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse::<u64>().ok()
}

fn parse_cache_size_bytes(raw: &str) -> u64 {
    let s = raw.trim();
    if s.is_empty() {
        return 0;
    }
    let (digits, mult) = match s.as_bytes().last().copied() {
        Some(b'K' | b'k') => (&s[..s.len() - 1], 1024_u64),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1024_u64 * 1024),
        Some(b'G' | b'g') => (&s[..s.len() - 1], 1024_u64 * 1024 * 1024),
        _ => (s, 1_u64),
    };
    digits
        .trim()
        .parse::<u64>()
        .ok()
        .and_then(|v| v.checked_mul(mult))
        .unwrap_or(0)
}

/// Select the 16 MiB-L3 fast X925 CPU cluster from `allowed` (falling back to
/// the highest-capacity cluster when `cache/index3/size` is absent, or empty
/// when CPUs are uniform/unreadable so x86 and sandboxes fail soft).
fn select_fast_cluster(allowed: &[usize], sysfs: &Path) -> Vec<usize> {
    if allowed.len() < 2 {
        return Vec::new();
    }
    for leaf in ["cpu_capacity", "cpufreq/cpuinfo_max_freq"] {
        let mut caps = BTreeMap::new();
        let mut all_ok = true;
        for &cpu in allowed {
            let p = sysfs.join(format!("cpu{cpu}")).join(leaf);
            match read_u64(&p) {
                Some(v) if v > 0 => {
                    caps.insert(cpu, v);
                }
                _ => {
                    all_ok = false;
                    break;
                }
            }
        }
        if !all_ok || caps.is_empty() {
            continue;
        }
        let top = *caps.values().max().unwrap_or(&0);
        let fast: Vec<usize> = allowed
            .iter()
            .copied()
            .filter(|c| (caps[c] as f64) >= 0.85 * (top as f64))
            .collect();
        if fast.is_empty() {
            continue;
        }

        let mut by_cluster: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut clusters_complete = true;
        for &cpu in &fast {
            let cpu_dir = sysfs.join(format!("cpu{cpu}"));
            let cl = read_trimmed(&cpu_dir.join("topology/cluster_cpus_list"))
                .or_else(|| read_trimmed(&cpu_dir.join("cache/index3/shared_cpu_list")));
            match cl {
                Some(key) => by_cluster.entry(key).or_default().push(cpu),
                None => {
                    clusters_complete = false;
                    break;
                }
            }
        }

        if clusters_complete && by_cluster.len() > 1 {
            let mut l3_sizes = Vec::with_capacity(by_cluster.len());
            for grp in by_cluster.values() {
                let l3 = grp
                    .iter()
                    .map(|&c| {
                        read_trimmed(&sysfs.join(format!("cpu{c}/cache/index3/size")))
                            .map(|s| parse_cache_size_bytes(&s))
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0);
                l3_sizes.push(l3);
            }
            let asymmetric_l3 = l3_sizes.windows(2).any(|w| w[0] != w[1]);
            if fast.len() != allowed.len() || asymmetric_l3 {
                if let Some(best) = by_cluster.into_values().max_by_key(|grp| {
                    let l3 = grp
                        .iter()
                        .map(|&c| {
                            read_trimmed(&sysfs.join(format!("cpu{c}/cache/index3/size")))
                                .map(|s| parse_cache_size_bytes(&s))
                                .unwrap_or(0)
                        })
                        .max()
                        .unwrap_or(0);
                    let cap_sum: u64 = grp.iter().map(|c| caps[c]).sum();
                    let max_id = grp.iter().copied().max().unwrap_or(0);
                    (l3 == L3_16_MIB, l3, cap_sum, grp.len(), max_id)
                }) {
                    if best.len() != allowed.len() {
                        return best;
                    }
                }
            }
        } else if fast.len() != allowed.len() {
            return fast;
        }
    }
    Vec::new()
}

#[cfg(target_os = "linux")]
fn pin_to_fast_cluster() {
    if let Ok(v) = std::env::var("CUDA_ENGINE_PIN") {
        let s = v.trim();
        if s == "0"
            || s.is_empty()
            || s.eq_ignore_ascii_case("false")
            || s.eq_ignore_ascii_case("no")
        {
            return;
        }
    }
    extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    const WORDS: usize = 16; // 1024 bits (glibc cpu_set_t)
    let mut mask = [0u64; WORDS];
    // SAFETY: `mask` is a valid 128-byte cpu_set_t buffer for pid 0 (self).
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
        return;
    }
    let mut allowed = Vec::new();
    for (w, &bits) in mask.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        for b in 0..64 {
            if (bits & (1u64 << b)) != 0 {
                allowed.push(w * 64 + b);
            }
        }
    }
    let selected = select_fast_cluster(&allowed, Path::new("/sys/devices/system/cpu"));
    if selected.is_empty() {
        return;
    }
    let mut next = [0u64; WORDS];
    for cpu in selected {
        if cpu < WORDS * 64 {
            next[cpu / 64] |= 1u64 << (cpu % 64);
        }
    }
    // SAFETY: `next` is a valid 128-byte cpu_set_t buffer; fails soft if denied.
    unsafe {
        let _ = sched_setaffinity(0, std::mem::size_of_val(&next), next.as_ptr());
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_fast_cluster() {}

fn main() -> ExitCode {
    disable_coredump();
    pin_to_fast_cluster();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let out = BufWriter::new(stdout.lock());

    let result = if let Some(socket) = protocol_adapter::resident::socket_from_env() {
        let (factory, hello, connect) =
            match protocol_adapter::session_backend::SessionFactory::from_resident(&socket) {
                Ok(attached) => attached,
                Err(err) => {
                    eprintln!("cuda-engine: cannot attach to the resident engine: {err}");
                    return ExitCode::FAILURE;
                }
            };
        // The engine identity is the RESIDENT's, because the resident is the
        // process that holds the weights. benchd records it from the hello, and
        // the string names the topology and carries the resident's load_epoch,
        // so the one-load claim reaches the sealed artifact.
        eprintln!(
            "cuda-engine: attached to the resident engine at {socket} in {:.3} ms (load epoch \
             {}, nothing loaded in this process); serving {}",
            connect.as_secs_f64() * 1e3,
            hello.load_epoch,
            hello.ident
        );
        let backend = hello.backend_string();
        #[cfg(unix)]
        {
            // benchd's next request is read the moment it lands (hotwait.rs);
            // an owned duplicate of fd 0 so no other buffer sits in front of
            // the poll, failing soft to stdin.lock() if dup is refused.
            match std::os::fd::AsFd::as_fd(&stdin).try_clone_to_owned() {
                Ok(fd) => {
                    let input = protocol_adapter::hotwait::HotReader::new(
                        std::fs::File::from(fd),
                        std::time::Duration::from_secs(2),
                    );
                    Adapter::with_backend(factory, backend, cuda_device()).run(input, out)
                }
                Err(_) => {
                    Adapter::with_backend(factory, backend, cuda_device()).run(stdin.lock(), out)
                }
            }
        }
        #[cfg(not(unix))]
        {
            Adapter::with_backend(factory, backend, cuda_device()).run(stdin.lock(), out)
        }
    } else {
        use protocol_adapter::mock::{MOCK_BACKEND, MOCK_DEVICE};
        eprintln!(
            "cuda-engine: serving the MOCK backend, announced as {MOCK_BACKEND}/{MOCK_DEVICE} in \
             the hello. Its tokens are a fixed function of the input and its timings mean \
             nothing. This binary exists to exercise the protocol. Boot a resident with \
             tools/serve-up.sh and export BENCH_WORKER_RESIDENT_SOCKET to attach to it."
        );
        let (factory, _log) = protocol_adapter::mock::MockFactory::new();
        Adapter::with_backend(factory, MOCK_BACKEND, MOCK_DEVICE).run(stdin.lock(), out)
    };

    // Protocol and engine errors are reported inline as `ok:false` responses;
    // run() returns Err only on a hard I/O failure.
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cuda-engine: I/O error on the protocol stream: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_cpu(
        root: &Path,
        cpu: usize,
        cap: Option<u64>,
        cluster: Option<&str>,
        l3_size: Option<&str>,
    ) {
        let dir = root.join(format!("cpu{cpu}"));
        std::fs::create_dir_all(dir.join("topology")).unwrap();
        std::fs::create_dir_all(dir.join("cache/index3")).unwrap();
        if let Some(c) = cap {
            std::fs::write(dir.join("cpu_capacity"), format!("{c}\n")).unwrap();
        }
        if let Some(cl) = cluster {
            std::fs::write(dir.join("topology/cluster_cpus_list"), format!("{cl}\n")).unwrap();
        }
        if let Some(sz) = l3_size {
            std::fs::write(dir.join("cache/index3/size"), format!("{sz}\n")).unwrap();
        }
    }

    #[test]
    fn picks_16mib_l3_x925_cluster_regardless_of_core_id_order() {
        let dir = std::env::temp_dir().join(format!("ce-cpu-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for c in 0..5 {
            write_cpu(&dir, c, Some(512), Some("0-9"), Some("16384K"));
        }
        for c in 5..10 {
            write_cpu(&dir, c, Some(1024), Some("0-9"), Some("16384K"));
        }
        for c in 10..15 {
            write_cpu(&dir, c, Some(512), Some("10-19"), Some("8192K"));
        }
        for c in 15..20 {
            write_cpu(&dir, c, Some(1024), Some("10-19"), Some("8192K"));
        }
        let all: Vec<usize> = (0..20).collect();
        assert_eq!(select_fast_cluster(&all, &dir), vec![5, 6, 7, 8, 9]);

        // Even if cgroup already restricted to the 10 X925 cores, narrows to 16 MiB L3.
        let x925_only: Vec<usize> = (5..10).chain(15..20).collect();
        assert_eq!(select_fast_cluster(&x925_only, &dir), vec![5, 6, 7, 8, 9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn falls_back_softly_on_uniform_x86_or_missing_index3() {
        let dir = std::env::temp_dir().join(format!("ce-cpu-x86-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for c in 0..8 {
            let cl = if c < 4 { "0-3" } else { "4-7" };
            write_cpu(&dir, c, Some(1024), Some(cl), Some("32768K"));
        }
        let all: Vec<usize> = (0..8).collect();
        assert!(select_fast_cluster(&all, &dir).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
        for c in 0..2 {
            write_cpu(&dir, c, Some(512), Some("0-1"), None);
        }
        for c in 2..4 {
            write_cpu(&dir, c, Some(1000), Some("2-3"), None);
        }
        for c in 4..6 {
            write_cpu(&dir, c, Some(1024), Some("4-5"), None);
        }
        assert_eq!(
            select_fast_cluster(&(0..6).collect::<Vec<_>>(), &dir),
            vec![4, 5]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
