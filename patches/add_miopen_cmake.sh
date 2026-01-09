#!/bin/bash
# Patch CMakeLists.txt to add MIOpen support

set -e

CMAKEFILE="$1"

if [[ ! -f "$CMAKEFILE" ]]; then
    echo "Usage: $0 <CMakeLists.txt>"
    exit 1
fi

# Add WITH_MIOPEN option after WITH_CUDNN
sed -i 's/option(WITH_CUDNN "Compile with cuDNN backend" OFF)/option(WITH_CUDNN "Compile with cuDNN backend" OFF)\noption(WITH_MIOPEN "Compile with MIOpen backend for AMD GPUs" OFF)/' "$CMAKEFILE"

# Add MIOpen linking after the cuDNN block (find the endif after WITH_CUDNN block)
# We'll add a new block for MIOpen
cat >> "$CMAKEFILE" << 'EOF'

# MIOpen support for AMD ROCm
if (WITH_MIOPEN)
  if (WITH_HIP)
    find_path(MIOPEN_INCLUDE_PATH NAMES miopen/miopen.h PATHS $ENV{ROCM_PATH}/include /opt/rocm/include)
    find_library(MIOPEN_LIBRARY NAMES MIOpen PATHS $ENV{ROCM_PATH}/lib /opt/rocm/lib)
    if (MIOPEN_INCLUDE_PATH AND MIOPEN_LIBRARY)
      message(STATUS "Found MIOpen: ${MIOPEN_LIBRARY}")
      target_include_directories(${PROJECT_NAME} PRIVATE ${MIOPEN_INCLUDE_PATH})
      target_link_libraries(${PROJECT_NAME} PRIVATE ${MIOPEN_LIBRARY})
      target_compile_definitions(${PROJECT_NAME} PRIVATE CT2_WITH_MIOPEN)
    else()
      message(WARNING "MIOpen not found. Conv1D will not be supported on GPU.")
    endif()
  else()
    message(WARNING "MIOpen requires HIP. Set WITH_HIP=ON to enable MIOpen support.")
  endif()
endif()
EOF

echo "CMakeLists.txt patched for MIOpen support"
