#!/usr/bin/env python3
"""
WhisperLive server for macOS using MLX backend.

Subclasses WhisperLive's TranscriptionServer to add MLX support.
"""

import argparse
import logging
import os
import sys

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_live.server import TranscriptionServer

from macos.mlx_whisper_backend import ServeClientMLXWhisper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Map shorthand model names to MLX HuggingFace repos
# MLX models use -mlx suffix: https://huggingface.co/collections/mlx-community/whisper
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_mlx_model(model_name: str) -> str:
    """Convert shorthand model name to MLX HuggingFace repo path."""
    if model_name in MLX_MODEL_MAP:
        resolved = MLX_MODEL_MAP[model_name]
        logger.info(f"Resolved model '{model_name}' -> '{resolved}'")
        return resolved
    # Already a full path or unknown - use as-is
    return model_name


class MLXTranscriptionServer(TranscriptionServer):
    """
    WhisperLive server with MLX backend support for macOS.
    """

    def initialize_client(
        self,
        websocket,
        options,
        faster_whisper_custom_model_path,
        whisper_tensorrt_path,
        trt_multilingual,
        trt_py_session=False,
    ):
        """
        Initialize an MLX Whisper client for the connection.

        Overrides parent to use MLX backend instead of faster_whisper/tensorrt.
        """
        # Resolve shorthand model names to MLX HuggingFace repos
        model = resolve_mlx_model(options.get("model", self.model))

        client = ServeClientMLXWhisper(
            websocket=websocket,
            model=model,
            language=options.get("language"),
            task=options.get("task", "transcribe"),
            client_uid=options.get("uid"),
            initial_prompt=options.get("initial_prompt"),
            no_speech_thresh=options.get("no_speech_thresh", 0.45),
            send_last_n_segments=options.get("send_last_n_segments", 10),
        )
        self.client_manager.add_client(websocket, client)

    def run(
        self,
        host,
        port=9090,
        model="mlx-community/whisper-large-v3-turbo",
        single_model=False,
        **kwargs,  # Ignore extra args
    ):
        """
        Run the MLX transcription server.

        Wraps parent run() but stores model for initialize_client.
        """
        self.model = model
        # Call parent with dummy backend (we override initialize_client anyway)
        super().run(
            host=host,
            port=port,
            backend="faster_whisper",  # Ignored - we override initialize_client
            single_model=single_model,
        )


def main():
    parser = argparse.ArgumentParser(
        description="WhisperLive server with MLX backend for macOS"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WHISPER_PORT", 9090)),
        help="Port to listen on (default: 9090)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"),
        help="MLX Whisper model to use"
    )
    args = parser.parse_args()

    # Check MLX availability
    try:
        import mlx.core as mx
        logger.info(f"MLX available: {mx.default_device()}")
    except ImportError as e:
        logger.error(f"MLX not available: {e}")
        logger.error("Install with: pip install mlx mlx-whisper")
        sys.exit(1)

    logger.info(f"Starting WhisperLive MLX server")
    logger.info(f"Model: {args.model}")

    server = MLXTranscriptionServer()
    server.run(
        host=args.host,
        port=args.port,
        model=args.model,
    )


if __name__ == "__main__":
    main()
