use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::mute::HardwareMuteDetector;

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_SIZE: usize = 1024; // samples at target rate

/// Info about an available audio input device.
#[derive(Debug)]
pub struct AudioDeviceInfo {
    pub index: usize,
    pub name: String,
    pub sample_rate: u32,
}

/// List available audio input devices.
pub fn list_devices() -> Result<Vec<AudioDeviceInfo>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    for (i, dev) in host.input_devices()?.enumerate() {
        let name = dev.description().map(|d| d.to_string()).unwrap_or_else(|_| format!("Device {i}"));
        let config = dev.default_input_config();
        let sample_rate = config.map(|c| c.sample_rate()).unwrap_or(0);
        devices.push(AudioDeviceInfo {
            index: i,
            name,
            sample_rate,
        });
    }
    Ok(devices)
}

/// Find a device by index or name substring.
pub fn find_device(query: &str) -> Result<Device> {
    let host = cpal::default_host();
    let devices: Vec<(usize, Device)> = host.input_devices()?.enumerate().collect();

    // Try as numeric index first
    if let Ok(index) = query.parse::<usize>() {
        if let Some((_, dev)) = devices.into_iter().find(|(i, _)| *i == index) {
            return Ok(dev);
        }
        bail!("No input device with index {index}");
    }

    // Search by name substring
    let query_lower = query.to_lowercase();
    let mut matches: Vec<(usize, Device)> = devices
        .into_iter()
        .filter(|(_, dev)| {
            dev.description()
                .map(|d| d.to_string())
                .unwrap_or_default()
                .to_lowercase()
                .contains(&query_lower)
        })
        .collect();

    match matches.len() {
        0 => bail!("No device found matching '{query}'"),
        1 => Ok(matches.remove(0).1),
        _ => {
            let names: Vec<String> = matches
                .iter()
                .map(|(i, d)| format!("  [{i}] {}", d.description().map(|d| d.to_string()).unwrap_or_default()))
                .collect();
            bail!(
                "Multiple devices match '{query}':\n{}\nPlease be more specific or use the index number.",
                names.join("\n")
            );
        }
    }
}

/// Linear interpolation resampling.
fn resample(audio: &[f32], orig_sr: u32, target_sr: u32) -> Vec<f32> {
    if orig_sr == target_sr || audio.is_empty() {
        return audio.to_vec();
    }
    let duration = audio.len() as f64 / orig_sr as f64;
    let target_len = (duration * target_sr as f64) as usize;
    if target_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(target_len);
    let ratio = (audio.len() - 1) as f64 / (target_len - 1).max(1) as f64;
    for i in 0..target_len {
        let src = i as f64 * ratio;
        let idx = src as usize;
        let frac = src - idx as f64;
        if idx + 1 < audio.len() {
            out.push(audio[idx] * (1.0 - frac as f32) + audio[idx + 1] * frac as f32);
        } else {
            out.push(audio[audio.len() - 1]);
        }
    }
    out
}

/// Running audio capture stream.
pub struct AudioCapture {
    _stream: Stream,
    pub rx: mpsc::Receiver<Vec<u8>>,
    pub hw_muted: Arc<AtomicBool>,
    #[allow(dead_code)]
    running: Arc<AtomicBool>,
}

impl AudioCapture {
    /// Start capturing audio from the given device (or default).
    pub fn start(device: Option<Device>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device {
            Some(d) => d,
            None => host
                .default_input_device()
                .context("No default input device")?,
        };

        let dev_name = device.description().map(|d| d.to_string()).unwrap_or_else(|_| "unknown".into());
        let supported = device.default_input_config()?;
        let device_sr = supported.sample_rate();
        let channels = supported.channels() as usize;

        // Compute device-side block size to yield ~CHUNK_SIZE samples at target rate
        let device_chunk = if device_sr != TARGET_SAMPLE_RATE {
            (CHUNK_SIZE as f64 * device_sr as f64 / TARGET_SAMPLE_RATE as f64) as u32
        } else {
            CHUNK_SIZE as u32
        };

        info!(
            "Audio device: {dev_name} ({device_sr} Hz, {channels}ch, chunk={device_chunk})"
        );
        if device_sr != TARGET_SAMPLE_RATE {
            info!("Will resample {device_sr} Hz -> {TARGET_SAMPLE_RATE} Hz");
        }

        let config = StreamConfig {
            channels: channels as u16,
            sample_rate: device_sr,
            buffer_size: cpal::BufferSize::Fixed(device_chunk),
        };

        let (tx, rx) = mpsc::channel::<Vec<u8>>(64);
        let running = Arc::new(AtomicBool::new(true));

        let mut hw_mute = HardwareMuteDetector::new();
        let hw_muted = hw_mute.muted.clone();

        // Shared callback logic: convert mono f32 audio → bytes and send
        let process_mono = {
            let tx = tx.clone();
            let running = running.clone();
            move |mono: Vec<f32>| {
                if !running.load(Ordering::Relaxed) {
                    return;
                }
                let max_abs = mono.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                hw_mute.update(max_abs);
                let resampled = resample(&mono, device_sr, TARGET_SAMPLE_RATE);
                let bytes: Vec<u8> = resampled
                    .iter()
                    .flat_map(|&s| s.to_ne_bytes())
                    .collect();
                let _ = tx.try_send(bytes);
            }
        };

        let sample_format = supported.sample_format();
        let stream = match sample_format {
            SampleFormat::F32 => {
                let mut process = process_mono;
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let mono = if channels > 1 {
                            data.chunks(channels)
                                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                                .collect()
                        } else {
                            data.to_vec()
                        };
                        process(mono);
                    },
                    move |err| warn!("Audio stream error: {err}"),
                    None,
                )?
            }
            SampleFormat::I16 => {
                let mut process = process_mono;
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let mono = if channels > 1 {
                            data.chunks(channels)
                                .map(|frame| {
                                    frame.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                        / channels as f32
                                })
                                .collect()
                        } else {
                            data.iter().map(|&s| s as f32 / 32768.0).collect()
                        };
                        process(mono);
                    },
                    move |err| warn!("Audio stream error: {err}"),
                    None,
                )?
            }
            other => bail!("Unsupported sample format: {other:?}"),
        };

        stream.play()?;
        info!("Audio capture started");

        Ok(Self {
            _stream: stream,
            rx,
            hw_muted,
            running,
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
