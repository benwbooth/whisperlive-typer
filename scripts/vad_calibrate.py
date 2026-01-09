#!/usr/bin/env -S uv run
"""
VAD Calibrator - helps tune Voice Activity Detection thresholds.

Shows real-time VAD probabilities so you can see:
- What triggers speech detection
- Background noise levels
- Optimal threshold settings

Silero VAD is neural-network based, not purely volume dependent.
It looks at spectral patterns characteristic of human speech.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import pyaudio
import torch
import yaml

CONFIG_PATH = Path.home() / ".config" / "whisper-typer" / "config.yaml"


class VADCalibrator:
    def __init__(self, device_index: int = None):
        self.sample_rate = 16000
        self.chunk_size = 512  # ~32ms at 16kHz
        self.device_index = device_index

        # Load Silero VAD
        print("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()

        # PyAudio setup
        self.pa = pyaudio.PyAudio()

    def list_devices(self):
        """List available input devices."""
        print("\nAvailable microphones:")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
        print()

    def get_vad_probability(self, audio_chunk: np.ndarray) -> float:
        """Get VAD probability for an audio chunk."""
        # Ensure float32 in range [-1, 1]
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk)

        # Get probability
        with torch.no_grad():
            prob = self.model(audio_tensor, self.sample_rate).item()

        return prob

    def get_volume_db(self, audio_chunk: np.ndarray) -> float:
        """Get volume in dB."""
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -100

    def get_device_name(self) -> str:
        """Get the name of the selected audio device."""
        if self.device_index is not None:
            info = self.pa.get_device_info_by_index(self.device_index)
            return f"[{self.device_index}] {info['name']}"
        else:
            default = self.pa.get_default_input_device_info()
            return f"[{default['index']}] {default['name']} (default)"

    def calibrate(self, duration: int = 15):
        """Run calibration, showing real-time VAD probabilities."""
        print(f"\n{'='*60}")
        print("VAD CALIBRATION")
        print(f"{'='*60}")
        print(f"\nMicrophone: {self.get_device_name()}")
        print("\nInstructions:")
        print("1. First, stay SILENT for 5 seconds (measuring noise floor)")
        print("2. Then SPEAK normally for remaining time")
        print("3. Watch the probability values")
        print()
        print("Legend:")
        print("  [VAD] Speech probability (0.0 = silence, 1.0 = speech)")
        print("  [VOL] Volume in dB (higher = louder)")
        print("  [BAR] Visual representation of VAD probability")
        print()
        input("Press Enter to start...")
        print()

        # Open audio stream
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )

        # Reset VAD state
        self.model.reset_states()

        # Collect statistics
        silence_probs = []
        speech_probs = []
        start_time = time.time()
        phase = "silence"

        try:
            print("=== SILENCE PHASE (stay quiet) ===")
            while True:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    break

                # Switch phase at 5 seconds
                if elapsed > 5 and phase == "silence":
                    phase = "speech"
                    print("\n=== SPEECH PHASE (speak normally) ===")

                # Read audio
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)

                # Get metrics
                prob = self.get_vad_probability(audio)
                vol_db = self.get_volume_db(audio)

                # Collect stats
                if phase == "silence":
                    silence_probs.append(prob)
                else:
                    speech_probs.append(prob)

                # Display
                bar_len = int(prob * 40)
                bar = '█' * bar_len + '░' * (40 - bar_len)

                # Color coding (ANSI)
                if prob < 0.1:
                    color = '\033[90m'  # Gray - silence
                elif prob < 0.5:
                    color = '\033[93m'  # Yellow - maybe speech
                else:
                    color = '\033[92m'  # Green - speech
                reset = '\033[0m'

                print(f"\r{color}[VAD {prob:.3f}] [VOL {vol_db:+6.1f}dB] [{bar}]{reset}", end='', flush=True)

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()

        print("\n")

        # Analyze results and return thresholds
        return self._analyze_results(silence_probs, speech_probs)

    def _analyze_results(self, silence_probs: list, speech_probs: list) -> tuple[float, float] | None:
        """Analyze collected data and suggest thresholds. Returns (onset, offset) or None."""
        print(f"{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")

        if not silence_probs or not speech_probs:
            print("\nInsufficient data for recommendations.")
            print("Make sure to stay silent for 5 seconds, then speak for the rest.")
            return None

        silence_max = max(silence_probs)
        silence_avg = sum(silence_probs) / len(silence_probs)
        silence_p95 = sorted(silence_probs)[int(len(silence_probs) * 0.95)]
        print(f"\nSilence phase ({len(silence_probs)} samples):")
        print(f"  Average: {silence_avg:.3f}")
        print(f"  Max:     {silence_max:.3f}")
        print(f"  95th %:  {silence_p95:.3f}")

        speech_min = min(speech_probs)
        speech_avg = sum(speech_probs) / len(speech_probs)
        speech_p10 = sorted(speech_probs)[int(len(speech_probs) * 0.1)]
        print(f"\nSpeech phase ({len(speech_probs)} samples):")
        print(f"  Average: {speech_avg:.3f}")
        print(f"  Min:     {speech_min:.3f}")
        print(f"  10th %:  {speech_p10:.3f}")

        # Calculate thresholds
        print(f"\n{'='*60}")
        print("RECOMMENDED THRESHOLDS")
        print(f"{'='*60}")

        # Onset should be:
        # - Above silence noise (silence_p95) to avoid false triggers
        # - Below speech level (speech_p10) to catch speech
        # Pick the midpoint between them, with some margin

        gap = speech_p10 - silence_p95

        if gap > 0.1:
            # Good separation - pick midpoint
            suggested_onset = silence_p95 + (gap * 0.4)
        elif gap > 0:
            # Small gap - stay closer to silence threshold
            suggested_onset = silence_p95 + (gap * 0.3)
        else:
            # Overlap - silence and speech overlap, pick a compromise
            # Use the lower of the two to at least catch speech
            suggested_onset = min(silence_p95, speech_p10) * 0.8
            print(f"\n⚠ WARNING: Silence and speech levels overlap!")
            print("  This may cause false triggers or missed speech.")
            print("  Try speaking louder or reducing background noise.")

        # Floor at 0.05 (very sensitive)
        suggested_onset = max(0.05, suggested_onset)

        # Offset: lower than onset to avoid cutting off speech
        suggested_offset = suggested_onset * 0.6

        # Round to 2 decimal places
        suggested_onset = round(suggested_onset, 2)
        suggested_offset = round(suggested_offset, 2)

        print(f"\n  Onset:  {suggested_onset}  (triggers speech detection)")
        print(f"  Offset: {suggested_offset}  (ends speech detection)")

        return suggested_onset, suggested_offset

    def write_config(self, onset: float, offset: float) -> bool:
        """Write VAD thresholds to config file. Returns True on success."""
        # Load existing config or create new
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Update VAD section
        if "vad" not in data:
            data["vad"] = {}
        data["vad"]["onset"] = onset
        data["vad"]["offset"] = offset

        # Write back
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Updated {CONFIG_PATH}")
        print(f"  vad.onset: {onset}")
        print(f"  vad.offset: {offset}")
        return True

    def close(self):
        self.pa.terminate()


def main():
    parser = argparse.ArgumentParser(
        description='VAD Calibrator - find optimal voice detection thresholds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Interactive calibration, asks before saving
  %(prog)s --auto           # Auto-calibrate and save to config
  %(prog)s --auto -t 10     # Auto-calibrate with 10 second duration
  %(prog)s --list-devices   # List available microphones
"""
    )
    parser.add_argument('--list-devices', action='store_true', help='List audio devices')
    parser.add_argument('--device', '-d', type=int, help='Audio device index')
    parser.add_argument('--duration', '-t', type=int, default=15, help='Calibration duration in seconds (default: 15)')
    parser.add_argument('--auto', '-a', action='store_true',
                        help='Automatically save thresholds to config without asking')
    args = parser.parse_args()

    calibrator = VADCalibrator(device_index=args.device)

    try:
        if args.list_devices:
            calibrator.list_devices()
            return

        result = calibrator.calibrate(duration=args.duration)

        if result:
            onset, offset = result

            if args.auto:
                # Auto-save without asking
                calibrator.write_config(onset, offset)
            else:
                # Ask user
                print()
                response = input(f"Save these thresholds to {CONFIG_PATH}? [Y/n] ").strip().lower()
                if response != 'n':
                    calibrator.write_config(onset, offset)
                else:
                    print("\nTo manually set thresholds, add to your config:")
                    print(f"  VAD_ONSET={onset}")
                    print(f"  VAD_OFFSET={offset}")
    finally:
        calibrator.close()


if __name__ == '__main__':
    main()
