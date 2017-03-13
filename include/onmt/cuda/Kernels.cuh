#pragma once

namespace onmt
{
  namespace cuda
  {

    template <typename T>
    void replicate(const T* vector, int len, T* matrix, int dim);

  }
}