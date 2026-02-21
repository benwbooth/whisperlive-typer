use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::debug;

/// Tracks hardware mute detection state via zero-sample counting.
pub struct HardwareMuteDetector {
    zero_count: u32,
    threshold: u32,
    pub muted: Arc<AtomicBool>,
}

impl HardwareMuteDetector {
    pub fn new() -> Self {
        Self {
            zero_count: 0,
            threshold: 10,
            muted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Call from the audio callback with the max absolute sample value.
    pub fn update(&mut self, max_abs: f32) {
        if max_abs < 1e-6 {
            self.zero_count += 1;
            if self.zero_count >= self.threshold && !self.muted.load(Ordering::Relaxed) {
                self.muted.store(true, Ordering::Relaxed);
                debug!("Hardware mute detected (audio is zeros)");
            }
        } else {
            if self.muted.load(Ordering::Relaxed) {
                debug!("Hardware unmute detected (audio resumed)");
            }
            self.zero_count = 0;
            self.muted.store(false, Ordering::Relaxed);
        }
    }
}

/// Polls software mute status via subprocess (pactl on Linux, osascript on macOS).
pub struct SoftwareMuteDetector {
    cache: Mutex<MuteCache>,
}

struct MuteCache {
    value: bool,
    last_check: Instant,
    ttl: Duration,
}

impl SoftwareMuteDetector {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(MuteCache {
                value: false,
                last_check: Instant::now() - Duration::from_secs(10), // force first check
                ttl: Duration::from_secs(1),
            }),
        }
    }

    pub async fn is_muted(&self) -> bool {
        let mut cache = self.cache.lock().await;
        if cache.last_check.elapsed() < cache.ttl {
            return cache.value;
        }

        let muted = tokio::task::spawn_blocking(check_software_mute)
            .await
            .unwrap_or(false);

        cache.value = muted;
        cache.last_check = Instant::now();
        muted
    }
}

fn check_software_mute() -> bool {
    if cfg!(target_os = "macos") {
        check_mute_macos()
    } else {
        check_mute_linux()
    }
}

fn check_mute_linux() -> bool {
    let result = Command::new("pactl")
        .args(["get-source-mute", "@DEFAULT_SOURCE@"])
        .output();
    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
            stdout.contains("yes")
        }
        _ => false,
    }
}

fn check_mute_macos() -> bool {
    let result = Command::new("osascript")
        .args(["-e", "input volume of (get volume settings)"])
        .output();
    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.trim().parse::<i32>().ok() == Some(0)
        }
        _ => false,
    }
}
