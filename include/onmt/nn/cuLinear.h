#pragma once

#include "onmt/nn/Linear.h"
#include "onmt/cuda/Utils.h"
#include "onmt/cuda/Kernels.cuh"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class cuLinear: public Linear<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      cuLinear(th::Table* data, cublasHandle_t* handle)
        : Linear<MatFwd, MatIn, MatEmb, ModelT>(data)
        , _handle(handle)
        // cuBLAS works with col-major matrices.
        , _bias_device((this->_bias->rows() > 0) ? cuda::to_device<float>(this->_bias->data(), this->_bias->rows()) : nullptr,
                [](float* p) { CUDA_CHECK(cudaFree(p)); })
        , _weight_device(cuda::to_device<float>(this->_weight->data(), this->_weight->cols(), this->_weight->rows()),
                [](float* p) { CUDA_CHECK(cudaFree(p)); })
        , _input_device(nullptr)
        , _output_device(nullptr)
        , _expanded_bias_device(nullptr)
        , _allocated_batches(0)
      {
      }

      cuLinear(const cuLinear& other)
        : Linear<MatFwd, MatIn, MatEmb, ModelT>(other)
        , _handle(other._handle)
        , _bias_device(other._bias_device)
        , _weight_device(other._weight_device)
        , _input_device(nullptr)
        , _output_device(nullptr)
        , _expanded_bias_device(nullptr)
        , _allocated_batches(0)
      {
      }

      ~cuLinear()
      {
        CUDA_CHECK(cudaFree(_input_device));
        CUDA_CHECK(cudaFree(_output_device));
        CUDA_CHECK(cudaFree(_expanded_bias_device));
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* clone(const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>*) const override
      {
        return new cuLinear(*this);
      }

      void set_handle(cublasHandle_t* handle)
      {
        _handle = handle;
      }

      void forward_impl(const MatFwd& input) override
      {
        // See http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm

        const size_t batch_size = input.rows();
        const int input_size = input.cols();
        const int output_size = this->_weight->rows();

        if (batch_size > _allocated_batches)
          this->realloc_device_buffers(batch_size);

        cuda::to_device<float>(_input_device, input.data(), input_size, batch_size);

        float alpha = 1;
        float beta = 0;

        CUBLAS_CHECK(cublasSgemm(*_handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 output_size, batch_size, input_size,
                                 &alpha,
                                 _weight_device.get(), input_size,
                                 _input_device, input_size,
                                 &beta,
                                 _output_device, output_size));

        if (_expanded_bias_device)
          cuda::kernels::add(_output_device, _expanded_bias_device, batch_size * output_size);

        this->_output.resize(batch_size, this->_weight->rows());
        cuda::to_host<float>(_output_device, this->_output.data(), output_size, batch_size);
      }

      virtual void apply_subdictionary(const std::vector<size_t>&)
      {
        throw std::runtime_error("subdictionary not implemented for cuLinear");
      }

    private:
      void realloc_device_buffers(int num_batches)
      {
        CUDA_CHECK(cudaFree(_output_device));
        CUDA_CHECK(cudaFree(_input_device));
        CUDA_CHECK(cudaFree(_expanded_bias_device));

        _output_device = cuda::to_device<float>(this->_weight->rows(), num_batches);
        _input_device = cuda::to_device<float>(this->_weight->cols(), num_batches);

        if (_bias_device)
        {
          _expanded_bias_device = cuda::to_device<float>(this->_weight->rows(), num_batches);
          for (int i = 0; i < num_batches; ++i)
            CUDA_CHECK(cudaMemcpy(_expanded_bias_device + i * this->_weight->rows(),
                                  _bias_device.get(),
                                  this->_weight->rows() * sizeof (float),
                                  cudaMemcpyDeviceToDevice));

        }

        _allocated_batches = num_batches;
      }

      cublasHandle_t* _handle;

      std::shared_ptr<float> _bias_device;
      std::shared_ptr<float> _weight_device;

      // Preallocate device buffers.
      float* _input_device;
      float* _output_device;
      float* _expanded_bias_device;
      size_t _allocated_batches;
    };

  }
}
