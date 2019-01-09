#pragma once

#include "Eigen/MatrixBatch.h"

#include "onmt/th/Utils.h"
#include "onmt/th/Obj.h"

namespace onmt
{

  // This class can be specialized to implement different loading behaviours
  // (including conversions) according to the source and/or target types.
  template <typename ToT, typename FromT>
  class StorageLoader;

  // This default specialization maps the storage to a Eigen structure without
  // any changes (same precision and same storage order).
  template <typename T>
  class StorageLoader<Eigen::RowMajorMatMap<T>, T>
  {
  public:
    static Eigen::RowMajorMatMap<T> get_matrix(th::Table* module_data,
                                                               const std::string& name)
    {
      th::Tensor<T>* tensor = th::get_field<th::Tensor<T>*>(module_data, name);

      if (!tensor)
        return Eigen::RowMajorMatMap<T>(nullptr, 0, 0);

      size_t rows = tensor->get_size()[0];
      size_t cols = tensor->get_dimension() == 1 ? 1 : tensor->get_size()[1];

      const T* storage_data = get_tensor_data(tensor);

      return Eigen::RowMajorMatMap<T>(storage_data, rows, cols);
    }
  };

}
