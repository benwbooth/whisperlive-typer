//! Short start/stop chimes: three quick high beeps when dictation starts
//! listening, three low beeps when it stops. Synthesized at runtime and
//! played through PulseAudio (pipewire-pulse), so no sound files ship.

use std::f32::consts::TAU;

use anyhow::{Result, anyhow};
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use tracing::warn;

const RATE: u32 = 48_000;
// Very staccato: short pips with a fast decay and a longer gap.
const BEEP_SECS: f32 = 0.030;
const GAP_SECS: f32 = 0.055;
const HIGH_HZ: f32 = 879.0; // A5
const LOW_HZ: f32 = HIGH_HZ / 1.5; // a perfect fifth below (D5)

#[derive(Clone, Copy, Debug)]
pub enum Chime {
    /// Three high pips.
    On,
    /// Three pips a fifth lower.
    Off,
}

impl Chime {
    fn notes(self) -> [f32; 3] {
        match self {
            Chime::On => [HIGH_HZ; 3],
            Chime::Off => [LOW_HZ; 3],
        }
    }
}

fn synthesize(chime: Chime, volume: f32) -> Vec<f32> {
    let beep = (BEEP_SECS * RATE as f32) as usize;
    let gap = (GAP_SECS * RATE as f32) as usize;
    let attack = (0.002 * RATE as f32) as usize;
    let mut out = Vec::with_capacity(3 * (beep + gap));
    for freq in chime.notes() {
        for i in 0..beep {
            let t = i as f32 / RATE as f32;
            // Fast attack, exponential decay, short linear fade at the end
            // so the beep never clicks.
            let env = (i as f32 / attack as f32).min(1.0)
                * (-t * 90.0).exp()
                * ((beep - i) as f32 / attack as f32).min(1.0);
            // A little second harmonic softens the pure sine.
            let s = (TAU * freq * t).sin() + 0.25 * (TAU * 2.0 * freq * t).sin();
            out.push(volume * env * s / 1.25);
        }
        out.extend(std::iter::repeat_n(0.0, gap));
    }
    out
}

/// Play a chime and block until it has finished.
pub fn play_blocking(chime: Chime, volume: f32) -> Result<()> {
    let spec = Spec { format: Format::F32le, channels: 1, rate: RATE };
    let pa = Simple::new(
        None,
        "whisper-typer",
        Direction::Playback,
        None,
        "chime",
        &spec,
        None,
        None,
    )
    .map_err(|e| anyhow!("open playback stream: {e}"))?;
    let samples = synthesize(chime, volume.clamp(0.0, 1.0));
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    pa.write(&bytes).map_err(|e| anyhow!("write chime: {e}"))?;
    pa.drain().map_err(|e| anyhow!("drain chime: {e}"))?;
    Ok(())
}

/// Play a chime on a background thread; errors are only logged.
pub fn play(chime: Chime, volume: f32) {
    std::thread::spawn(move || {
        if let Err(e) = play_blocking(chime, volume) {
            warn!("Could not play chime: {e}");
        }
    });
}
