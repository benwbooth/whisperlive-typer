// Conv1D GPU implementation with MIOpen support for AMD ROCm
#include "ctranslate2/ops/conv1d.h"
#include "cuda/helpers.h"

#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

namespace ctranslate2 {
  namespace ops {

    // Thread-local MIOpen handle
    static miopenHandle_t get_miopen_handle() {
      static thread_local miopenHandle_t handle = nullptr;
      if (!handle) {
        miopenStatus_t status = miopenCreate(&handle);
        if (status != miopenStatusSuccess) {
          throw std::runtime_error("Failed to create MIOpen handle");
        }
      }
      return handle;
    }

    // Helper to check MIOpen status
    static void check_miopen(miopenStatus_t status, const char* msg) {
      if (status != miopenStatusSuccess) {
        throw std::runtime_error(std::string("MIOpen error in ") + msg + ": " +
                                 std::to_string(static_cast<int>(status)));
      }
    }

    // Helper to check HIP status
    static void check_hip(hipError_t status, const char* msg) {
      if (status != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in ") + msg + ": " +
                                 hipGetErrorString(status));
      }
    }

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output,
                         const StorageView* qscale) const {
      (void)qscale;

      if (weight.dtype() == DataType::INT8 || weight.dtype() == DataType::INT16) {
        throw std::runtime_error("Quantization not supported in MIOpen Conv1D");
      }

      // Get dimensions: input is [batch, in_channels, length]
      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t in_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);

      // Determine MIOpen data type
      miopenDataType_t data_type;
      if constexpr (std::is_same<T, float>::value) {
        data_type = miopenFloat;
      } else if constexpr (std::is_same<T, float16_t>::value) {
        data_type = miopenHalf;
      } else {
        throw std::runtime_error("Unsupported data type for MIOpen Conv1D");
      }

      // Create descriptors
      miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
      miopenConvolutionDescriptor_t conv_desc;

      check_miopen(miopenCreateTensorDescriptor(&input_desc), "create input desc");
      check_miopen(miopenCreateTensorDescriptor(&weight_desc), "create weight desc");
      check_miopen(miopenCreateTensorDescriptor(&output_desc), "create output desc");
      check_miopen(miopenCreateConvolutionDescriptor(&conv_desc), "create conv desc");

      // Set tensor descriptors - treat 1D as 2D with H=1
      // Format: NCHW where H=1, W=length
      check_miopen(miopenSet4dTensorDescriptor(input_desc, data_type,
                   batch_size, in_channels, 1, in_length), "set input desc");

      check_miopen(miopenSet4dTensorDescriptor(weight_desc, data_type,
                   out_channels, in_channels, 1, kernel_size), "set weight desc");

      // Set convolution descriptor: pad_h=0, pad_w=padding, stride_h=1, stride_w=stride
      check_miopen(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                   0, _padding, 1, _stride, 1, _dilation), "init conv desc");

      // Get output dimensions
      int n, c, h, w;
      check_miopen(miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, weight_desc,
                   &n, &c, &h, &w), "get output dim");

      check_miopen(miopenSet4dTensorDescriptor(output_desc, data_type, n, c, h, w),
                   "set output desc");

      // Sync GPU before MIOpen operations
      check_hip(hipDeviceSynchronize(), "sync before find");

      // Get workspace size
      size_t workspace_size = 0;
      check_miopen(miopenConvolutionForwardGetWorkSpaceSize(get_miopen_handle(),
                   weight_desc, input_desc, conv_desc, output_desc, &workspace_size),
                   "get workspace size");

      // Allocate workspace
      void* workspace_ptr = nullptr;
      if (workspace_size > 0) {
        check_hip(hipMalloc(&workspace_ptr, workspace_size), "alloc workspace");
      }

      // Find algorithm
      const int request_algo_count = 1;
      int returned_algo_count = 0;
      miopenConvAlgoPerf_t perf_result;

      miopenStatus_t find_status = miopenFindConvolutionForwardAlgorithm(
          get_miopen_handle(),
          input_desc, input.data<T>(),
          weight_desc, weight.data<T>(),
          conv_desc,
          output_desc, output.data<T>(),
          request_algo_count, &returned_algo_count, &perf_result,
          workspace_ptr, workspace_size,
          false);  // exhaustiveSearch = false

      if (find_status != miopenStatusSuccess || returned_algo_count == 0) {
        if (workspace_ptr) hipFree(workspace_ptr);
        miopenDestroyTensorDescriptor(input_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(output_desc);
        miopenDestroyConvolutionDescriptor(conv_desc);
        throw std::runtime_error("MIOpen: Failed to find convolution algorithm");
      }

      // Execute convolution
      float alpha = 1.0f, beta = 0.0f;
      miopenStatus_t conv_status = miopenConvolutionForward(
          get_miopen_handle(),
          &alpha,
          input_desc, input.data<T>(),
          weight_desc, weight.data<T>(),
          conv_desc, perf_result.fwd_algo,
          &beta,
          output_desc, output.data<T>(),
          workspace_ptr, workspace_size);

      // Cleanup workspace
      if (workspace_ptr) {
        hipFree(workspace_ptr);
      }

      // Check convolution result
      if (conv_status != miopenStatusSuccess) {
        miopenDestroyTensorDescriptor(input_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(output_desc);
        miopenDestroyConvolutionDescriptor(conv_desc);
        throw std::runtime_error("MIOpen: Convolution forward failed");
      }

      // Add bias if present
      if (bias) {
        miopenTensorDescriptor_t bias_desc;
        check_miopen(miopenCreateTensorDescriptor(&bias_desc), "create bias desc");
        check_miopen(miopenSet4dTensorDescriptor(bias_desc, data_type,
                     1, out_channels, 1, 1), "set bias desc");

        float one = 1.0f;
        miopenStatus_t bias_status = miopenOpTensor(get_miopen_handle(),
            miopenTensorOpAdd,
            &one, output_desc, output.data<T>(),
            &one, bias_desc, bias->data<T>(),
            &beta, output_desc, output.data<T>());

        miopenDestroyTensorDescriptor(bias_desc);

        if (bias_status != miopenStatusSuccess) {
          miopenDestroyTensorDescriptor(input_desc);
          miopenDestroyTensorDescriptor(weight_desc);
          miopenDestroyTensorDescriptor(output_desc);
          miopenDestroyConvolutionDescriptor(conv_desc);
          throw std::runtime_error("MIOpen: Bias addition failed");
        }
      }

      // Cleanup descriptors
      miopenDestroyTensorDescriptor(input_desc);
      miopenDestroyTensorDescriptor(weight_desc);
      miopenDestroyTensorDescriptor(output_desc);
      miopenDestroyConvolutionDescriptor(conv_desc);
    }

    // Explicit template instantiations
    template void Conv1D::compute<Device::CUDA, float>(
        const StorageView&, const StorageView&, const StorageView*,
        StorageView&, const StorageView*) const;

    template void Conv1D::compute<Device::CUDA, float16_t>(
        const StorageView&, const StorageView&, const StorageView*,
        StorageView&, const StorageView*) const;

    template void Conv1D::compute<Device::CUDA, bfloat16_t>(
        const StorageView&, const StorageView&, const StorageView*,
        StorageView&, const StorageView*) const;

  }
}
