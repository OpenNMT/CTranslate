#pragma once

#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace onmt
{
  namespace cuda
  {

    cublasHandle_t* get_handle();

    template <typename T>
    T* to_device(const T* host, int rows, int cols)
    {
      T* device = nullptr;

      cudaError_t cudaStatus = cudaMalloc(&device, rows * cols * sizeof (T));

      if (cudaStatus != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");

      if (host)
      {
        cublasStatus_t cublasStatus = cublasSetMatrix(rows, cols, sizeof (T), host, rows, device, rows);

        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
          cudaFree(device);
          throw std::runtime_error("cublasSetMatrix failed");
        }
      }

      return device;
    }

    template <typename T>
    T* to_device(const T* host, int n)
    {
      T* device = nullptr;

      cudaError_t cudaStatus = cudaMalloc(&device, n * sizeof (T));

      if (cudaStatus != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");

      if (host)
      {
        cublasStatus_t cublasStatus = cublasSetVector(n, sizeof (T), host, 1, device, 1);

        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
          cudaFree(device);
          throw std::runtime_error("cublasSetVector failed");
        }
      }

      return device;
    }

    template <typename T>
    T* to_host(const T* device, T* host, int rows, int cols)
    {
      bool allocate = !host;

      if (allocate)
      {
        host = (T*) malloc(rows * cols * sizeof (T));
      }

      cublasStatus_t cublasStatus = cublasGetMatrix(rows, cols, sizeof (T), device, rows, host, rows);

      if (cublasStatus != CUBLAS_STATUS_SUCCESS)
      {
        if (allocate)
          free(host);
        throw std::runtime_error("cublasGetMatrix failed");
      }

      return host;
    }

  }
}
