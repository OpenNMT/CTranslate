#include "onmt/cuda/Utils.h"

namespace onmt
{
  namespace cuda
  {

    cublasHandle_t* get_handle()
    {
      static cublasHandle_t handle;
      static cublasStatus_t cublasStatus = cublasCreate(&handle);

      if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS initialization failed");

      return &handle;
    }

  }
}
