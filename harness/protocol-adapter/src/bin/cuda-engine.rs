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

use std::io::{self, BufWriter};
use std::os::fd::AsFd;
use std::process::ExitCode;

use protocol_adapter::Adapter;

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

#[cfg(target_os = "linux")]
fn pin_fast_cpus() {
    if std::env::var("CUDA_ENGINE_PIN_FAST").is_ok_and(|v| v.trim() == "0") {
        return;
    }
    extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let mut allowed = [0u64; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&allowed), allowed.as_mut_ptr()) } != 0 {
        return;
    }
    let mut caps: Vec<(usize, u64)> = Vec::new();
    for cpu in 0..128usize {
        if (allowed[cpu / 64] & (1u64 << (cpu % 64))) == 0 {
            continue;
        }
        let cap = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity"))
            .or_else(|_| {
                std::fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
                ))
            })
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0);
        if cap > 0 {
            caps.push((cpu, cap));
        }
    }
    let Some(top) = caps.iter().map(|&(_, c)| c).max() else {
        return;
    };
    let fast: Vec<usize> = caps
        .iter()
        .filter(|&&(_, c)| c * 100 >= top * 85)
        .map(|&(cpu, _)| cpu)
        .collect();
    if fast.is_empty() || fast.len() == caps.len() {
        return;
    }
    let lower: Vec<usize> = fast.iter().copied().filter(|&c| c < 10).collect();
    let chosen = if lower.is_empty() { &fast } else { &lower };
    let mut target = [0u64; 16];
    for &cpu in chosen {
        target[cpu / 64] |= 1u64 << (cpu % 64);
    }
    unsafe {
        let _ = sched_setaffinity(0, std::mem::size_of_val(&target), target.as_ptr());
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_fast_cpus() {}

fn main() -> ExitCode {
    disable_coredump();
    pin_fast_cpus();
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
        let input = match stdin.as_fd().try_clone_to_owned() {
            Ok(fd) => protocol_adapter::hotwait::HotReader::new(
                std::fs::File::from(fd),
                std::time::Duration::from_millis(500),
            ),
            Err(err) => {
                eprintln!("cuda-engine: cannot duplicate stdin: {err}");
                return ExitCode::FAILURE;
            }
        };
        Adapter::with_backend(factory, backend, cuda_device()).run(input, out)
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
