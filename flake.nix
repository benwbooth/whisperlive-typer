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
          config = {
            allowUnfree = true;
            rocmSupport = true;
          };
        };

        python = pkgs.python311;
        rocm = pkgs.rocmPackages_6;

        # Target GPU architecture (RX 7900 XTX = gfx1100)
        gpuArch = "gfx1100";

        # CTranslate2 source (shared between lib and python bindings)
        ctranslate2-src = pkgs.fetchgit {
          name = "ctranslate2-rocm-source";
          url = "https://github.com/arlo-phoenix/CTranslate2-rocm.git";
          rev = "81c77087ec264299dbfe32a202c01f2b7e798a91";
          hash = "sha256-bQDaBfpPLIjMV4FzTjwudk6ZktUG9+UnEgf+aqi/0zQ=";
          fetchSubmodules = true;
          deepClone = true;
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
            rev = "main";
            hash = "sha256-ISSu25wlOqF/KvzRIU3hM9sd+0zmG6jRsId6xMzbue4=";
          };

          propagatedBuildInputs = with python.pkgs; [
            faster-whisper
            websockets
            numpy
            scipy
            torch  # Required for VAD (Voice Activity Detection)
            ffmpeg-python
            soundfile
          ];

          # Apply our patches
          postPatch = ''
            cp ${./patches/base.py} whisper_live/backend/base.py
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
            python << EOF
import os
from whisper_live.server import TranscriptionServer

port = int(os.environ.get("WHISPER_PORT", "9090"))
backend = os.environ.get("WHISPER_BACKEND", "faster_whisper")
model = os.environ.get("WHISPER_MODEL", None)

print(f"Starting WhisperLive server on port {port} with backend {backend}")
server = TranscriptionServer()
server.run(
    "0.0.0.0",
    port=port,
    backend=backend,
    faster_whisper_custom_model_path=model,
)
EOF
          '';
        };

        # Frontend: whisper-typer client (base package)
        whisper-typer-base = python.pkgs.buildPythonApplication {
          pname = "whisper-typer";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with python.pkgs; [
            setuptools
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            sounddevice
            websockets
            pyyaml
            pyaudio
            packaging
            grapheme
          ];

          doCheck = false;

          makeWrapperArgs = [
            "--set" "YDOTOOL_SOCKET" "/run/ydotoold/socket"
            "--prefix" "LD_LIBRARY_PATH" ":" "${pkgs.portaudio}/lib"
          ];

          meta = with pkgs.lib; {
            description = "Speech-to-text typing client using WhisperLive and ydotool";
            homepage = "https://github.com/benwbooth/whisperlive-typer";
            license = licenses.mit;
            platforms = platforms.linux;
            mainProgram = "whisper-typer";
          };
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

        # Combined package with client and toggle script
        whisper-typer = pkgs.symlinkJoin {
          name = "whisper-typer";
          paths = [ whisper-typer-base whisper-toggle ];
          meta = whisper-typer-base.meta;
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
          ];

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

          config = mkIf cfg.enable {
            programs.ydotool.enable = mkIf cfg.client.enable true;

            environment.systemPackages = mkMerge [
              (mkIf cfg.client.enable [ self.packages.${system}.whisper-typer ])
              (mkIf cfg.server.enable [ self.packages.${system}.whisperlive-server ])
            ];

            systemd.services.whisperlive-server = mkIf cfg.server.enable {
              description = "WhisperLive Server";
              after = [ "network.target" ];
              wantedBy = [ "multi-user.target" ];

              environment = {
                WHISPER_PORT = toString cfg.server.port;
                HSA_OVERRIDE_GFX_VERSION = "11.0.0";
              };

              serviceConfig = {
                Type = "simple";
                ExecStart = "${self.packages.${system}.whisperlive-server}/bin/whisperlive-server";
                Restart = "on-failure";
                # GPU access
                SupplementaryGroups = [ "video" "render" ];
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
                ExecStart = "${self.packages.${system}.whisper-typer}/bin/whisper-typer";
                Restart = "on-failure";
                RestartSec = 3;
              };
            };
          };
        };

      overlays.default = final: prev: {
        whisper-typer = self.packages.${prev.system}.whisper-typer;
        whisperlive-server = self.packages.${prev.system}.whisperlive-server;
      };
    };
}
