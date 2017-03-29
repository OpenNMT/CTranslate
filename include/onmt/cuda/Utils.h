#pragma once

#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(ans) { onmt::cuda::cudaAssert((ans), __FILE__, __LINE__); }
#define CUBLAS_CHECK(ans) { onmt::cuda::cublasAssert((ans), __FILE__, __LINE__); }

namespace onmt
{
  namespace cuda
  {

    std::string cublasGetStatusString(cublasStatus_t status);

    inline
    void cudaAssert(cudaError_t code, const std::string& file, int line)
    {
      if (code != cudaSuccess)
        throw std::runtime_error("CUDA failed with error " + std::string(cudaGetErrorString(code)) + " at " + file + ":" + std::to_string(line));
    }

    inline
    void cublasAssert(cublasStatus_t status, const std::string& file, int line)
    {
      if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS failed with status " + cublasGetStatusString(status) + " at " + file + ":" + std::to_string(line));
    }

    template <typename T>
    void to_device(T* device, const T* host, int rows, int cols)
    {
      CUBLAS_CHECK(cublasSetMatrix(rows, cols, sizeof (T), host, rows, device, rows));
    }

    template <typename T>
    T* to_device(const T* host, int rows, int cols)
    {
      T* device = nullptr;

      CUDA_CHECK(cudaMalloc(&device, rows * cols * sizeof (T)));
      CUBLAS_CHECK(cublasSetMatrix(rows, cols, sizeof (T), host, rows, device, rows));

      return device;
    }

    template <typename T>
    T* to_device(int rows, int cols)
    {
      T* device = nullptr;

      CUDA_CHECK(cudaMalloc(&device, rows * cols * sizeof (T)));

      return device;
    }

    template <typename T>
    T* to_device(const T* host, int n)
    {
      T* device = nullptr;

      CUDA_CHECK(cudaMalloc(&device, n * sizeof (T)));

      if (host)
        CUBLAS_CHECK(cublasSetVector(n, sizeof (T), host, 1, device, 1));

      return device;
    }

    template <typename T>
    T* to_host(const T* device, T* host, int rows, int cols)
    {
      if (!host)
        host = (T*) malloc(rows * cols * sizeof (T));

      CUBLAS_CHECK(cublasGetMatrix(rows, cols, sizeof (T), device, rows, host, rows));

      return host;
    }

  }
}
