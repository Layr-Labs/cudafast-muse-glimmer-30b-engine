
use std::os::fd::RawFd;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[repr(C)]
struct PollFd {
    fd: std::ffi::c_int,
    events: std::ffi::c_short,
    revents: std::ffi::c_short,
}

const POLLIN: std::ffi::c_short = 0x001;
const TICK_MS: std::ffi::c_int = 1;

extern "C" {
    fn poll(fds: *mut PollFd, nfds: std::ffi::c_ulong, timeout: std::ffi::c_int)
        -> std::ffi::c_int;
}

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUDA_ENGINE_HOT_WAIT").map_or(true, |v| v.trim() != "0"))
}

pub fn wait_readable(fd: RawFd, hot: Duration) {
    if !enabled() {
        return;
    }
    let started = Instant::now();
    loop {
        let mut pfd = PollFd {
            fd,
            events: POLLIN,
            revents: 0,
        };
        let ready = unsafe { poll(&mut pfd, 1, TICK_MS) };
        if ready != 0 || started.elapsed() >= hot {
            return;
        }
    }
}

pub struct HotReader<R: std::io::Read + std::os::fd::AsRawFd> {
    inner: std::io::BufReader<R>,
    hot: Duration,
}

impl<R: std::io::Read + std::os::fd::AsRawFd> HotReader<R> {
    pub fn new(inner: R, hot: Duration) -> Self {
        Self {
            inner: std::io::BufReader::new(inner),
            hot,
        }
    }
}

impl<R: std::io::Read + std::os::fd::AsRawFd> std::io::Read for HotReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use std::io::BufRead;
        let available = self.fill_buf()?;
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: std::io::Read + std::os::fd::AsRawFd> std::io::BufRead for HotReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.inner.buffer().is_empty() {
            wait_readable(self.inner.get_ref().as_raw_fd(), self.hot);
        }
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, Write};
    use std::os::unix::net::UnixStream;

    #[test]
    fn hot_reader_reads_the_same_lines() {
        let (mut a, b) = UnixStream::pair().unwrap();
        let writer = std::thread::spawn(move || {
            a.write_all(b"one\ntw").unwrap();
            std::thread::sleep(Duration::from_millis(5));
            a.write_all(b"o\nthree\n").unwrap();
        });
        let reader = HotReader::new(b, Duration::from_secs(2));
        let lines: Vec<String> = reader.lines().map(Result::unwrap).collect();
        writer.join().unwrap();
        assert_eq!(lines, ["one", "two", "three"]);
    }

    #[test]
    fn wait_gives_up_after_hot() {
        let (_a, b) = UnixStream::pair().unwrap();
        let started = Instant::now();
        wait_readable(
            std::os::fd::AsRawFd::as_raw_fd(&b),
            Duration::from_millis(20),
        );
        assert!(started.elapsed() >= Duration::from_millis(20));
    }
}
