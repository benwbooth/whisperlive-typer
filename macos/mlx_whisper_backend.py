"""
MLX Whisper backend for WhisperLive on macOS.

Uses Apple's MLX framework for GPU-accelerated inference on Apple Silicon.
"""

import json
import logging
import threading
from types import SimpleNamespace

import numpy as np
import sys
import os

# Add project root to path for patches import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patches.base import ServeClientBase

# MLX imports - only available on macOS with Apple Silicon
try:
    import mlx_whisper
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("mlx-whisper not available. Install with: pip install mlx mlx-whisper")


class ServeClientMLXWhisper(ServeClientBase):
    """
    WhisperLive client using MLX Whisper for macOS Apple Silicon.
    """

    def __init__(
        self,
        websocket,
        client_uid=None,
        model="mlx-community/whisper-large-v3-turbo",
        language="en",
        task="transcribe",
        initial_prompt=None,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        min_avg_logprob=-0.8,
        clip_audio=False,
        same_output_threshold=10,
    ):
        super().__init__(
            client_uid=client_uid,
            websocket=websocket,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            min_avg_logprob=min_avg_logprob,
        )

        if not MLX_AVAILABLE:
            logging.error("mlx-whisper not available. Install with: pip install mlx mlx-whisper")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": "mlx-whisper not available on this system"
            }))
            return

        self.model_path = model
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt

        logging.info(f"MLX Whisper backend initialized with model: {self.model_path}")

        # Start transcription thread
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "mlx_whisper"
        }))

    def transcribe_audio(self, input_sample):
        """
        Transcribe audio using MLX Whisper.

        Args:
            input_sample: NumPy array of audio samples (16kHz, float32)

        Returns:
            List of segment objects or None if no speech detected
        """
        if input_sample is None or len(input_sample) == 0:
            return None

        # Ensure correct dtype for MLX
        if input_sample.dtype != np.float32:
            input_sample = input_sample.astype(np.float32)

        try:
            # MLX Whisper accepts numpy arrays directly
            result = mlx_whisper.transcribe(
                input_sample,
                path_or_hf_repo=self.model_path,
                language=self.language,
                task=self.task,
                initial_prompt=self.initial_prompt,
                word_timestamps=False,
                condition_on_previous_text=True,
            )

            # Update language if auto-detected
            if self.language is None and "language" in result:
                self.language = result["language"]
                logging.info(f"Detected language: {self.language}")
                self.websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "language": self.language
                }))

            # Convert dict segments to objects for base class compatibility
            segments = result.get("segments", [])
            if not segments:
                return None

            return [self._wrap_segment(s) for s in segments]

        except Exception as e:
            logging.error(f"MLX Whisper transcription error: {e}")
            return None

    def _wrap_segment(self, seg_dict):
        """
        Convert mlx-whisper dict segment to object with attributes.

        The base class uses getattr() to access segment fields.
        """
        return SimpleNamespace(
            text=seg_dict.get("text", ""),
            start=seg_dict.get("start", 0.0),
            end=seg_dict.get("end", 0.0),
            no_speech_prob=seg_dict.get("no_speech_prob", 0.0),
            avg_logprob=seg_dict.get("avg_logprob", 0.0),
        )

    def handle_transcription_output(self, result, duration):
        """
        Process transcription output and send to client.

        Args:
            result: List of segment objects from transcribe_audio
            duration: Duration of the transcribed audio chunk
        """
        if result is None or len(result) == 0:
            return

        # Use base class's update_segments to process results
        last_segment = self.update_segments(result, duration)
        segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
