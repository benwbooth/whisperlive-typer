use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::mute::HardwareMuteDetector;

const TARGET_SAMPLE_RATE: u32 = 16000;
// Frames per read. At 16kHz mono f32, 1024 frames = ~64ms of audio per chunk.
const CHUNK_FRAMES: usize = 1024;

/// Opaque handle for a selected audio input device.
/// PulseAudio uses string names; we just thread the user's config value
/// through to `Simple::new` and let the daemon do the matching.
#[derive(Clone, Debug)]
pub struct AudioDevice(pub String);

/// Info about an available audio input device.
#[derive(Debug)]
pub struct AudioDeviceInfo {
    pub index: usize,
    pub name: String,
    pub sample_rate: u32,
}

/// List available audio input devices.
///
/// libpulse's async introspection API is overkill for this single use case
/// (one-shot listing for `--list-devices`), so we shell out to `pactl`.
pub fn list_devices() -> Result<Vec<AudioDeviceInfo>> {
    let output = Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
        .context("failed to run `pactl list short sources` (is PulseAudio/PipeWire running?)")?;
    if !output.status.success() {
        bail!(
            "pactl list short sources failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();
    for (i, line) in stdout.lines().enumerate() {
        // Format: "<index>\t<name>\t<driver>\t<sample_spec>\t<state>"
        // sample_spec example: "s16le 2ch 44100Hz"
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            continue;
        }
        let name = parts[1].to_string();
        let sample_rate = parts[3]
            .split_whitespace()
            .find_map(|tok| tok.strip_suffix("Hz").and_then(|s| s.parse::<u32>().ok()))
            .unwrap_or(0);
        devices.push(AudioDeviceInfo { index: i, name, sample_rate });
    }
    Ok(devices)
}

/// Resolve a user-supplied device query string to an AudioDevice.
///
/// PulseAudio accepts source names directly, so we only validate that
/// SOMETHING with that name (or substring) exists, then pass the canonical
/// name through. Numeric input is treated as an index into `list_devices`.
pub fn find_device(query: &str) -> Result<AudioDevice> {
    let devices = list_devices()?;

    if let Ok(index) = query.parse::<usize>() {
        let dev = devices
            .into_iter()
            .find(|d| d.index == index)
            .with_context(|| format!("No input device with index {index}"))?;
        return Ok(AudioDevice(dev.name));
    }

    let q = query.to_lowercase();
    let matches: Vec<AudioDeviceInfo> = devices
        .into_iter()
        .filter(|d| d.name.to_lowercase().contains(&q))
        .collect();

    match matches.len() {
        0 => bail!("No device found matching '{query}'"),
        1 => Ok(AudioDevice(matches.into_iter().next().unwrap().name)),
        _ => {
            let listing: Vec<String> = matches
                .iter()
                .map(|d| format!("  [{}] {}", d.index, d.name))
                .collect();
            bail!(
                "Multiple devices match '{query}':\n{}\nPlease be more specific or use the index number.",
                listing.join("\n")
            );
        }
    }
}

/// Running audio capture stream.
pub struct AudioCapture {
    pub rx: mpsc::Receiver<Vec<u8>>,
    pub hw_muted: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    _thread: std::thread::JoinHandle<()>,
}

impl AudioCapture {
    /// Open a PulseAudio capture stream and spawn a reader thread.
    ///
    /// PulseAudio handles resampling and channel downmix server-side when
    /// we request mono f32 at TARGET_SAMPLE_RATE, regardless of the source's
    /// native config.
    pub fn start(device: Option<AudioDevice>) -> Result<Self> {
        let spec = Spec {
            format: Format::FLOAT32NE,
            channels: 1,
            rate: TARGET_SAMPLE_RATE,
        };
        if !spec.is_valid() {
            bail!("Invalid pulse sample spec: {:?}", spec);
        }

        let device_name = device.as_ref().map(|d| d.0.as_str());
        // Low-latency buffer config. fragsize is the chunk pulse delivers per
        // read; tying it to CHUNK_FRAMES keeps per-read latency at ~64ms
        // instead of pulse's default of several hundred ms.
        let chunk_bytes = (CHUNK_FRAMES * std::mem::size_of::<f32>()) as u32;
        let attr = BufferAttr {
            maxlength: chunk_bytes * 4,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: chunk_bytes,
        };
        let simple = Simple::new(
            None,                  // default server
            "whisper-typer",       // application name
            Direction::Record,
            device_name,
            "Microphone",          // stream description
            &spec,
            None,                  // default channel map
            Some(&attr),
        )
        .with_context(|| format!("failed to open PulseAudio capture (device={:?})", device_name))?;

        info!(
            "Audio device: {} ({} Hz, mono, backend=pulse)",
            device_name.unwrap_or("<default>"),
            TARGET_SAMPLE_RATE
        );

        // Small backpressure window: 8 chunks × ~64ms ≈ 500ms total before
        // a slow consumer starts losing audio. Keeps end-to-end latency
        // bounded if the websocket/network stalls briefly.
        let (tx, rx) = mpsc::channel::<Vec<u8>>(8);
        let running = Arc::new(AtomicBool::new(true));

        let mut hw_mute = HardwareMuteDetector::new();
        let hw_muted = hw_mute.muted.clone();

        let running_thread = running.clone();
        let thread = std::thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                // f32 samples; CHUNK_FRAMES * 4 bytes/sample.
                let mut byte_buf = vec![0u8; CHUNK_FRAMES * std::mem::size_of::<f32>()];
                while running_thread.load(Ordering::Relaxed) {
                    if let Err(e) = simple.read(&mut byte_buf) {
                        warn!("Audio read error: {e}");
                        break;
                    }
                    // Compute the running max for the hardware-mute detector
                    // without an extra allocation.
                    let max_abs = byte_buf
                        .chunks_exact(4)
                        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]).abs())
                        .fold(0.0f32, f32::max);
                    hw_mute.update(max_abs);
                    // Send a fresh Vec so the consumer owns its buffer.
                    if tx.try_send(byte_buf.clone()).is_err() {
                        // Receiver gone or backpressured; drop and continue.
                    }
                }
            })
            .context("failed to spawn audio capture thread")?;

        info!("Audio capture started");
        Ok(Self {
            rx,
            hw_muted,
            running,
            _thread: thread,
        })
    }

    /// Drain all pending audio chunks from the channel.
    pub fn drain(&mut self) {
        let mut count = 0;
        while self.rx.try_recv().is_ok() {
            count += 1;
        }
        if count > 0 {
            debug!("Drained {count} audio chunks");
        }
    }

    /// Stop the audio capture.
    #[allow(dead_code)]
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        info!("Audio capture stopped");
    }
}
