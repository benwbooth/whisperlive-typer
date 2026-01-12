# WhisperLive server with ROCm support for AMD GPUs
# Uses CTranslate2-ROCm fork for faster-whisper GPU acceleration
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Build arguments for GPU architecture
# RX 7900 XTX/XT = gfx1100, RX 7800/7700 XT = gfx1101, RX 7600 = gfx1102
ARG PYTORCH_ROCM_ARCH=gfx1100
ARG HSA_OVERRIDE_GFX_VERSION=11.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV ROCM_PATH=/opt/rocm
ENV HIP_VISIBLE_DEVICES=0
ENV CXX=clang++

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libomp-dev \
    cmake \
    git \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Copy MIOpen patches for CTranslate2
COPY patches/ /build/patches/

# Clone and build CTranslate2 with ROCm/HIP + MIOpen support
RUN git clone --recursive https://github.com/arlo-phoenix/CTranslate2-rocm.git && \
    cd CTranslate2-rocm && \
    # Apply MIOpen patches
    bash /build/patches/add_miopen_cmake.sh CMakeLists.txt && \
    cp /build/patches/conv1d_gpu_miopen.cu src/ops/conv1d_gpu.cu && \
    # Build with MIOpen enabled
    cmake -S . -B build \
        -DWITH_MKL=OFF \
        -DWITH_HIP=ON \
        -DWITH_MIOPEN=ON \
        -DCMAKE_HIP_ARCHITECTURES=${PYTORCH_ROCM_ARCH} \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTS=OFF && \
    cmake --build build -- -j$(nproc) && \
    cd build && make install && \
    ldconfig

# Set library paths for Python bindings build
ENV LIBRARY_PATH=/usr/local/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install CTranslate2 Python bindings
# Build wheel from source
RUN cd /build/CTranslate2-rocm/python && \
    pip install -r install_requirements.txt && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl

# Install faster-whisper (--no-deps to prevent overwriting ROCm PyTorch)
RUN pip install --no-cache-dir --no-deps faster-whisper==1.1.0 && \
    pip install --no-cache-dir \
    av \
    huggingface_hub \
    tokenizers \
    onnxruntime \
    websockets \
    numpy \
    soundfile

# Clone and install WhisperLive (--no-deps to preserve ROCm PyTorch)
RUN git clone https://github.com/collabora/WhisperLive.git /app/WhisperLive && \
    cd /app/WhisperLive && \
    pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir scipy ffmpeg-python onnxruntime

# Apply patch to expose confidence fields (no_speech_prob, avg_logprob) in segment JSON
COPY patches/base.py /app/WhisperLive/whisper_live/backend/base.py

# Cleanup build directory to reduce image size
RUN rm -rf /build

WORKDIR /app/WhisperLive

# Copy custom run script
COPY run_server.sh /app/

# Expose WebSocket port
EXPOSE 9090

# Health check - verify GPU is accessible
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" || exit 1

# Run WhisperLive server
CMD ["/bin/bash", "/app/run_server.sh"]
