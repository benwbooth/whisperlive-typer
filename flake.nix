{
  description = "WhisperLive Typer - Speech-to-text typing with ROCm backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # No global rocmSupport: only CTranslate2 is built against ROCm
          # (explicitly, below). With rocmSupport on, torch was also built
          # against ROCm and brought up a second HIP runtime in the same
          # process, even though WhisperLive only runs CPU work (Silero
          # VAD) through it. Plain torch also comes from the binary cache.
          config.allowUnfree = true;
        };

        python = pkgs.python313;
        # Track the default ROCm (7.x), the same release as the system's
        # amdgpu/KFD stack and other ROCm users. ROCm 6.4 on a 7.x kernel
        # was the backdrop for a MES REMOVE_QUEUE hang + GPU reset on
        # 2026-11-14 that took the whole desktop down.
        rocm = pkgs.rocmPackages;

        # Target GPU architecture (RX 7900 XTX = gfx1100)
        gpuArch = "gfx1100";

        # CTranslate2 source (shared between lib and python bindings)
        ctranslate2-src = pkgs.fetchgit {
          name = "ctranslate2-rocm-source";
          url = "https://github.com/arlo-phoenix/CTranslate2-rocm.git";
          rev = "81c77087ec264299dbfe32a202c01f2b7e798a91";
          hash = "sha256-ALS6Bl/U4AY7j4IWLcKeUR8coXnxSNKn5z3yHjqwd1M=";
          # No deepClone: it keeps .git, whose contents vary with the git
          # version, so the hash broke on a nixpkgs bump.
          fetchSubmodules = true;
        };

        # Use ROCm's clang stdenv for proper HIP support
        rocmStdenv = rocm.llvm.rocmClangStdenv;

        # CTranslate2 with ROCm/HIP and MIOpen support
        ctranslate2-rocm = rocmStdenv.mkDerivation rec {
          pname = "ctranslate2-rocm";
          version = "unstable-2024-01-18";

          src = ctranslate2-src;

          nativeBuildInputs = with pkgs; [
            cmake
            rocm.hipcc
            git
            pkg-config
          ];

          buildInputs = with pkgs; [
            rocm.clr
            rocm.rocblas
            rocm.miopen-hip
            rocm.hipblas
            rocm.hipsparse
            rocm.hiprand
            rocm.rocrand
            rocm.rocprim
            rocm.rocthrust
            rocm.hipcub
            rocm.rocm-runtime
            rocm.rocm-device-libs
            rocm.rocm-cmake
            mkl  # Intel MKL includes libiomp5 (Intel OpenMP)
            oneDNN  # Intel DNNL
            openblas
          ];

          cmakeFlags = [
            "-DWITH_MKL=ON"
            "-DWITH_DNNL=ON"
            "-DWITH_HIP=ON"
            "-DWITH_MIOPEN=ON"
            "-DCMAKE_HIP_ARCHITECTURES=${gpuArch}"
            "-DCMAKE_HIP_COMPILER=${rocm.llvm.clang}/bin/clang++"
            "-DBUILD_TESTS=OFF"
            "-DINTEL_ROOT=${pkgs.mkl}"
            "-DDNNL_DIR=${pkgs.oneDNN}"
            "-DOPENMP_RUNTIME=INTEL"
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DBLA_STATIC=OFF"
          ];

          preConfigure = ''
            # Apply MIOpen CMake patch
            sed -i 's/option(WITH_CUDNN "Compile with cuDNN backend" OFF)/option(WITH_CUDNN "Compile with cuDNN backend" OFF)\noption(WITH_MIOPEN "Compile with MIOpen backend for AMD GPUs" OFF)/' CMakeLists.txt

            # Add hiprand include path near the top of CMakeLists.txt (after project() call)
            sed -i '/^project(ctranslate2/a include_directories("${rocm.hiprand}/include" "${rocm.rocrand}/include")' CMakeLists.txt

            # Force MKL to use shared libraries instead of static (.a -> .so)
            sed -i 's/libmkl_core\.a/libmkl_core.so/g' CMakeLists.txt
            sed -i 's/libmkl_intel_lp64\.a/libmkl_intel_lp64.so/g' CMakeLists.txt
            sed -i 's/libmkl_intel_ilp64\.a/libmkl_intel_ilp64.so/g' CMakeLists.txt
            sed -i 's/libmkl_sequential\.a/libmkl_sequential.so/g' CMakeLists.txt
            sed -i 's/libmkl_intel_thread\.a/libmkl_intel_thread.so/g' CMakeLists.txt
            sed -i 's/libmkl_gnu_thread\.a/libmkl_gnu_thread.so/g' CMakeLists.txt

            # ROCm 7: hipBLAS's _v2 GEMM entry points (hipDataType +
            # hipblasComputeType_t) became the plain names, and the _v2
            # aliases were removed.
            sed -i 's/hipblasGemmEx_v2/hipblasGemmEx/; s/hipblasGemmStridedBatchedEx_v2/hipblasGemmStridedBatchedEx/' src/cuda/primitives.cu

            # ROCm 7's rocThrust no longer pulls these in transitively.
            sed -i 's|#include <hip/hip_bf16.h>|&\n#include <thrust/iterator/counting_iterator.h>\n#include <thrust/reduce.h>\n#include <thrust/extrema.h>\n#include <thrust/execution_policy.h>|' src/cuda/helpers.h
            grep -q 'thrust/iterator/counting_iterator.h' src/cuda/helpers.h

            # Fix missing cstdint include in cxxopts (needed for uint8_t)
            sed -i '1i #include <cstdint>' third_party/cxxopts/include/cxxopts.hpp

            cat >> CMakeLists.txt << 'EOF'

# MIOpen support for AMD ROCm
if (WITH_MIOPEN)
  if (WITH_HIP)
    find_path(MIOPEN_INCLUDE_PATH NAMES miopen/miopen.h PATHS $ENV{ROCM_PATH}/include)
    find_library(MIOPEN_LIBRARY NAMES MIOpen PATHS $ENV{ROCM_PATH}/lib)
    if (MIOPEN_INCLUDE_PATH AND MIOPEN_LIBRARY)
      message(STATUS "Found MIOpen: ''${MIOPEN_LIBRARY}")
      target_include_directories(''${PROJECT_NAME} PRIVATE ''${MIOPEN_INCLUDE_PATH})
      target_link_libraries(''${PROJECT_NAME} PRIVATE ''${MIOPEN_LIBRARY})
      target_compile_definitions(''${PROJECT_NAME} PRIVATE CT2_WITH_MIOPEN)
    else()
      message(WARNING "MIOpen not found.")
    endif()
  endif()
endif()
EOF

            # Copy MIOpen conv1d implementation
            cp ${./patches/conv1d_gpu_miopen.cu} src/ops/conv1d_gpu.cu
          '';

          env = {
            ROCM_PATH = "${rocm.clr}";
            HIP_PATH = "${rocm.clr}";
            HIP_DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
            DEVICE_LIB_PATH = "${rocm.rocm-device-libs}/amdgcn/bitcode";
            # Add hiprand include paths
            NIX_CFLAGS_COMPILE = "-I${rocm.hiprand}/include -I${rocm.rocrand}/include";
          };

          meta = with pkgs.lib; {
            description = "Fast inference engine for Transformer models with ROCm support";
            homepage = "https://github.com/arlo-phoenix/CTranslate2-rocm";
            license = licenses.mit;
            platforms = [ "x86_64-linux" ];
          };
        };

        # Python bindings for CTranslate2
        ctranslate2-python = python.pkgs.buildPythonPackage rec {
          pname = "ctranslate2";
          version = "unstable-2024-01-18";
          format = "setuptools";

          src = ctranslate2-src;

          sourceRoot = "ctranslate2-rocm-source/python";

          nativeBuildInputs = with python.pkgs; [
            setuptools
            pybind11
          ];

          buildInputs = [
            ctranslate2-rocm
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            pyyaml
          ];

          env = {
            CTRANSLATE2_ROOT = "${ctranslate2-rocm}";
          };

          doCheck = false;
        };

        # faster-whisper with ROCm CTranslate2
        faster-whisper = python.pkgs.buildPythonPackage rec {
          pname = "faster-whisper";
          version = "1.1.0";
          format = "setuptools";

          src = pkgs.fetchFromGitHub {
            owner = "SYSTRAN";
            repo = "faster-whisper";
            rev = "v${version}";
            hash = "sha256-oJBCEwTfon80XQ9XIgnRw0SLvpwX0L5jnezwG0jv3Eg=";
          };

          propagatedBuildInputs = with python.pkgs; [
            ctranslate2-python
            huggingface-hub
            tokenizers
            onnxruntime
            av
          ];

          doCheck = false;
        };

        # WhisperLive server
        whisper-live = python.pkgs.buildPythonPackage rec {
          pname = "whisper-live";
          version = "unstable-2024-01-18";
          format = "setuptools";

          src = pkgs.fetchFromGitHub {
            owner = "collabora";
            repo = "WhisperLive";
            rev = "6c8142a9d2ca7accb0a686fd341191841427c43a";
            hash = "sha256-Q9UDqoMQe+Y6N/E2IMlxBH0rgLOLCniJp6iS7iWc/CM=";
          };

          propagatedBuildInputs = with python.pkgs; [
            faster-whisper
            websockets
            numpy
            scipy
            torch  # Required for VAD (Voice Activity Detection)
            ffmpeg-python
            soundfile
            fastapi
            uvicorn
          ];

          # Apply our patches
          postPatch = ''
            cp ${./patches/base.py} whisper_live/backend/base.py

            # torch is CPU-only here, so ask CTranslate2 (the ROCm build)
            # whether a GPU exists. ROCm GPUs all support float16.
            for f in whisper_live/backend/faster_whisper_backend.py whisper_live/server.py; do
              substituteInPlace "$f" \
                --replace-fail 'torch.cuda.is_available()' '__import__("ctranslate2").get_cuda_device_count() > 0'
            done
            substituteInPlace whisper_live/backend/faster_whisper_backend.py \
              --replace-fail 'major, _ = torch.cuda.get_device_capability(device)' 'major = 7'
          '';

          doCheck = false;
        };

        # Backend server package
        whisperlive-server = pkgs.writeShellApplication {
          name = "whisperlive-server";
          runtimeInputs = [
            (python.withPackages (ps: [
              whisper-live
              faster-whisper
              ctranslate2-python
            ]))
            pkgs.ffmpeg
          ];
          text = ''
            export HSA_OVERRIDE_GFX_VERSION=11.0.0
            export HIP_VISIBLE_DEVICES=0
            exec python -u ${./server/whisperlive_server.py}
          '';
        };

        # Toggle script for the client user service
        whisper-toggle = pkgs.writeShellScriptBin "whisper-toggle" ''
          SERVICE="whisper-typer.service"

          if systemctl --user is-active --quiet "$SERVICE"; then
            echo "Stopping $SERVICE..."
            systemctl --user stop "$SERVICE"
          else
            echo "Starting $SERVICE..."
            systemctl --user start "$SERVICE"
          fi
        '';

        # Rust frontend client
        whisper-typer-unwrapped = pkgs.rustPlatform.buildRustPackage {
          pname = "whisper-typer";
          version = "0.1.0";

          src = ./whisper-typer-rs;
          cargoLock.lockFile = ./whisper-typer-rs/Cargo.lock;

          nativeBuildInputs = [ pkgs.pkg-config ];
          # libpulse-sys links libpulse + libpulse-simple at build time
          # (pkg-config-driven). dlopen happens for the actual sound server
          # backend (pipewire-pulse / pulseaudio) at runtime.
          buildInputs = [ pkgs.libpulseaudio ];

          meta = with pkgs.lib; {
            description = "Speech-to-text typing client using WhisperLive";
            homepage = "https://github.com/benwbooth/whisperlive-typer";
            license = licenses.mit;
            platforms = platforms.linux;
            mainProgram = "whisper-typer";
          };
        };

        whisper-typer = pkgs.symlinkJoin {
          name = "whisper-typer";
          paths = [ whisper-typer-unwrapped whisper-toggle ];
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/whisper-typer \
              --set YDOTOOL_SOCKET /run/ydotoold/socket \
              --prefix PATH : ${pkgs.lib.makeBinPath [
                pkgs.ydotool
                pkgs.libnotify
                pkgs.pulseaudio  # provides pactl for --list-devices
                pkgs.glib        # provides gdbus for closing notifications
              ]}
          '';
          meta = whisper-typer-unwrapped.meta;
        };

      in
      {
        packages = {
          default = whisper-typer;
          inherit whisper-typer;
          inherit whisperlive-server;
          inherit ctranslate2-rocm;
          inherit faster-whisper;
          inherit whisper-live;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.portaudio
            pkgs.ydotool
            pkgs.libnotify
            python
            rocm.rocminfo
            # Rust toolchain
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config
            pkgs.alsa-lib
            pkgs.libpulseaudio  # linked by the Rust client (libpulse-simple)
          ];

          # Keep flake inputs in closure so GC doesn't collect them
          FLAKE_INPUTS = builtins.concatStringsSep ":" [ "${nixpkgs}" "${flake-utils}" ];

          LD_LIBRARY_PATH = "${pkgs.portaudio}/lib";

          shellHook = ''
            export YDOTOOL_SOCKET="/run/ydotoold/socket"
            export ROCM_PATH="${rocm.clr}"
          '';
        };

        # App runners
        apps = {
          default = {
            type = "app";
            program = "${whisper-typer}/bin/whisper-typer";
          };
          server = {
            type = "app";
            program = "${whisperlive-server}/bin/whisperlive-server";
          };
        };
      }
    ) // {
      # NixOS module
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.services.whisper-typer;
          system = pkgs.stdenv.hostPlatform.system;
        in
        {
          options.services.whisper-typer = {
            enable = mkEnableOption "WhisperLive Typer";

            server = {
              enable = mkEnableOption "WhisperLive server";

              port = mkOption {
                type = types.port;
                default = 9090;
                description = "Port for the WhisperLive server";
              };

              model = mkOption {
                type = types.str;
                default = "large-v3-turbo";
                description = "Whisper model to preload at startup (should match the client's whisper.model)";
              };

              gpuArch = mkOption {
                type = types.str;
                default = "gfx1100";
                description = "GPU architecture (gfx1100 for RX 7900 XTX)";
              };
            };

            client = {
              enable = mkEnableOption "WhisperLive client";
            };
          };

          config = mkIf cfg.enable (let
            clientPkg = self.packages.${system}.whisper-typer;
          in {
            programs.ydotool.enable = mkIf cfg.client.enable true;

            environment.systemPackages = mkMerge [
              (mkIf cfg.client.enable [ clientPkg ])
              (mkIf cfg.server.enable [ self.packages.${system}.whisperlive-server ])
            ];

            systemd.services.whisperlive-server = mkIf cfg.server.enable {
              description = "WhisperLive Server";
              after = [ "network.target" ];
              wantedBy = [ "multi-user.target" ];

              environment = {
                WHISPER_PORT = toString cfg.server.port;
                WHISPER_PRELOAD_MODEL = cfg.server.model;
                HSA_OVERRIDE_GFX_VERSION = "11.0.0";
              };

              serviceConfig = {
                Type = "simple";
                ExecStart = "${self.packages.${system}.whisperlive-server}/bin/whisperlive-server";
                Restart = "on-failure";
                # GPU access
                SupplementaryGroups = [ "video" "render" ];
                # The GPU reads host buffers mapped straight from this
                # process (KFD userptr). When the kernel swaps or reclaims
                # those pages, amdgpu has to evict and restore the compute
                # queues; on Navi 31 that path hung MES (full GPU reset,
                # 2026-11-14 14:28) and faulted the server (20:53) while the
                # host was deep in swap. Keep the server resident.
                MemorySwapMax = 0;
                MemoryMin = "4G";
              };
            };

            # User service for the client (toggle with whisper-toggle)
            systemd.user.services.whisper-typer = mkIf cfg.client.enable {
              description = "WhisperLive Typer Client";
              after = [ "graphical-session.target" "pipewire.service" ];
              requisite = [ "graphical-session.target" ];

              environment = {
                YDOTOOL_SOCKET = "/run/ydotoold/socket";
              };

              serviceConfig = {
                Type = "simple";
                ExecStart = "${clientPkg}/bin/whisper-typer";
                Restart = "on-failure";
                RestartSec = 3;
              };
            };
          });
        };

      overlays.default = final: prev: {
        whisper-typer = self.packages.${prev.system}.whisper-typer;
        whisper-typer-rs = self.packages.${prev.system}.whisper-typer-rs;
        whisperlive-server = self.packages.${prev.system}.whisperlive-server;
      };
    };
}
