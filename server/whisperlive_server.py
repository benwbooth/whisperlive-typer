"""WhisperLive server launcher: caches/preloads the faster-whisper model so
client reconnects (e.g. toggling dictation) don't reload it."""
import os
from whisper_live.server import TranscriptionServer

port = int(os.environ.get("WHISPER_PORT", "9090"))
backend = os.environ.get("WHISPER_BACKEND", "faster_whisper")
model = os.environ.get("WHISPER_MODEL", None)

if backend == "faster_whisper":
    # Importing torch/CTranslate2 can take tens of seconds on a cold process.
    # Pay that cost when the persistent system service starts, not when the
    # user begins dictating.
    print("Loading faster-whisper runtime...")
    import logging
    import threading
    import time

    import numpy as np
    import torch
    from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

    # Upstream WhisperLive builds a fresh WhisperModel for every WebSocket
    # connection, so each client toggle reloaded the model (sometimes 30s+).
    # Keep loaded models in-process, keyed by what determines the weights,
    # and hand the same instance to every connection. CTranslate2 models are
    # safe to call from multiple threads.
    _orig_create_model = ServeClientFasterWhisper.create_model
    _model_cache = {}
    _model_cache_lock = threading.Lock()

    def _cached_create_model(self, device):
        key = (self.model_size_or_path, device, self.compute_type)
        with _model_cache_lock:
            cached = _model_cache.get(key)
            if cached is not None:
                logging.info(f"Reusing loaded model {key}")
                self.transcriber = cached
                return
            started = time.monotonic()
            _orig_create_model(self, device)
            _model_cache[key] = self.transcriber
            logging.info(f"Loaded model {key} in {time.monotonic() - started:.1f}s")

    ServeClientFasterWhisper.create_model = _cached_create_model

    preload = os.environ.get("WHISPER_PRELOAD_MODEL", "large-v3-turbo")
    if preload:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            compute_type = "float16" if major >= 7 else "float32"
        else:
            compute_type = "int8"
        stub = ServeClientFasterWhisper.__new__(ServeClientFasterWhisper)
        stub.model_size_or_path = model or preload
        stub.compute_type = compute_type
        stub.cache_path = "~/.cache/whisper-live/"
        # Upstream's built-in model names (normally set in __init__); names
        # outside this list are treated as paths / HuggingFace repo ids.
        stub.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo",
        ]
        print(f"Preloading model {stub.model_size_or_path} ({device}, {compute_type})...")
        _cached_create_model(stub, device)
        # Run one tiny inference so GPU kernels are initialized before the
        # first real utterance.
        segments, _ = stub.transcriber.transcribe(
            np.zeros(16000, dtype=np.float32), language="en", vad_filter=False
        )
        list(segments)
        print("Model ready")

print(f"Starting WhisperLive server on port {port} with backend {backend}")
server = TranscriptionServer()
server.run(
    "0.0.0.0",
    port=port,
    backend=backend,
    faster_whisper_custom_model_path=model,
)
