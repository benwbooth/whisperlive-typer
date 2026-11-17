"""WhisperLive server launcher: caches/preloads the faster-whisper model so
client reconnects (e.g. toggling dictation) don't reload it."""
import os

# WHISPER_DEVICE=cpu keeps Whisper off the GPU entirely. On the 7900 XTX that
# also drives the desktop, ROCm compute from this server coincided with amdgpu
# shader errors (sq_intr) and corrupted squares on screen. Hide the GPU before
# anything loads the HIP runtime.
device_pref = os.environ.get("WHISPER_DEVICE", "auto")
if device_pref == "cpu":
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", "8"))

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

    # faster-whisper uses 4 CPU threads by default; with the model on the CPU
    # each pass re-decodes several seconds of audio, so use more.
    import whisper_live.backend.faster_whisper_backend as _fwb
    _OrigWhisperModel = _fwb.WhisperModel

    def _whisper_model(*args, **kwargs):
        if kwargs.get("device", args[1] if len(args) > 1 else None) == "cpu":
            kwargs.setdefault("cpu_threads", cpu_threads)
        return _OrigWhisperModel(*args, **kwargs)

    _fwb.WhisperModel = _whisper_model

    # After a sentence is committed, the buffer often keeps a ~0.2 s sliver
    # of VAD padding before the silence. Whisper turns that sliver into a
    # whole invented sentence ("We'll be right back.", "Thank you."), which
    # then settles and gets typed. Don't transcribe so little speech: wait
    # for more audio, or drop the sliver once silence follows it.
    MIN_SPEECH_S = 0.35
    DROP_SLIVER_AFTER_S = 2.0

    import dataclasses
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    def _measure_trailing_silence(self, input_sample):
        """Seconds of silence at the end of the buffer, from Silero VAD.

        Whisper's own segment end times often run into the pause after the
        last word, so they can't tell when the speaker has stopped.
        """
        tail = input_sample[-3 * 16000:]
        params = self.vad_parameters or {}
        opts = params if isinstance(params, VadOptions) else VadOptions(**params)
        opts = dataclasses.replace(opts, speech_pad_ms=0, min_silence_duration_ms=100)
        speech = get_speech_timestamps(tail, opts)
        if not speech:
            return len(tail) / 16000
        return (len(tail) - speech[-1]["end"]) / 16000

    def _transcribe_audio(self, input_sample):
        try:
            self.trailing_silence_s = _measure_trailing_silence(self, input_sample)
        except Exception as e:
            logging.warning(f"Trailing-silence VAD failed: {e}")
            self.trailing_silence_s = None
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        try:
            result, info = self.transcriber.transcribe(
                input_sample,
                initial_prompt=self.initial_prompt,
                language=self.language,
                task=self.task,
                vad_filter=self.use_vad,
                vad_parameters=self.vad_parameters if self.use_vad else None)
            speech = getattr(info, "duration_after_vad", None)
            if self.use_vad and speech is not None and 0 < speech < MIN_SPEECH_S:
                duration = len(input_sample) / 16000
                logging.info(f"  Ignoring {speech:.2f}s of speech in {duration:.2f}s buffer")
                # None makes the caller skip past this buffer as silence;
                # [] keeps it so a word that is just starting can grow.
                return None if duration >= DROP_SLIVER_AFTER_S else []
            # faster-whisper returns None when VAD removed all the audio;
            # the caller treats None as silence and moves past the buffer.
            result = None if result is None else list(result)
        finally:
            if ServeClientFasterWhisper.SINGLE_MODEL:
                ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()
        if self.language is None and info is not None:
            self.set_language(info)
        return result

    ServeClientFasterWhisper.transcribe_audio = _transcribe_audio

    preload = os.environ.get("WHISPER_PRELOAD_MODEL", "small")
    if preload:
        # torch is CPU-only; CTranslate2 is the ROCm build.
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if device == "cuda":
            compute_type = "float16"
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
