#pragma once

#include "Eigen/MatrixBatch.h"

#include "onmt/th/Utils.h"
#include "onmt/th/Obj.h"

namespace onmt
{

  // This class can be specialized to implement different loading behaviours
  // (including conversions) according to the source and/or target types.
  template <typename FromT, typename ToT>
  class StorageLoader
  {
  public:
    static ToT get_matrix(th::Table* module_data, const std::string& name);
  };

  // This default specialization maps the storage to a Eigen structure without
  // any changes (same precision and same storage order).
  template <typename T>
  class StorageLoader<Eigen::Map<const Eigen::RowMajorMat<T> >, T>
  {
  public:
    static Eigen::Map<const Eigen::RowMajorMat<T> > get_matrix(th::Table* module_data,
                                                               const std::string& name)
    {
      th::Tensor<T>* tensor = th::get_field<th::Tensor<T>*>(module_data, name);

      if (!tensor)
        return Eigen::Map<const Eigen::RowMajorMat<T> >(nullptr, 0, 0);

      size_t rows = tensor->get_size()[0];
      size_t cols = tensor->get_dimension() == 1 ? 1 : tensor->get_size()[1];

      const T* storage_data = get_tensor_data(tensor);

      return Eigen::Map<const Eigen::RowMajorMat<T> >(storage_data, rows, cols);
    }
  };

  // This specialization creates a sparse matrix from a CSR representation.
  template <typename T>
  class StorageLoader<const Eigen::RowMajorSparseMat<T>, T>
  {
  public:
    static const Eigen::RowMajorSparseMat<T> get_matrix(th::Table* module_data,
                                                        const std::string& name)
    {
      auto values_tensor = th::get_field<th::Tensor<T>*>(module_data, name);

      if (!values_tensor)
        return Eigen::RowMajorSparseMat<T>();

      auto sizes_tensor = th::get_field<th::Tensor<int>*>(module_data, name + "_size");
      const int* sizes = get_tensor_data(sizes_tensor);

      size_t num_rows = sizes[0];
      size_t num_cols = sizes_tensor->get_size()[0] == 2 ? sizes[1] : 1;

      Eigen::RowMajorSparseMat<T> sparse(num_rows, num_cols);
      sparse.reserve(values_tensor->get_size()[0]);

      const T* values = get_tensor_data(values_tensor);

      if (sizes_tensor->get_size()[0] == 1)
      {
        auto rows_tensor = th::get_field<th::Tensor<int>*>(module_data, name + "_rows");
        const int* rows = get_tensor_data(rows_tensor);

        for (int i = 0; i < values_tensor->get_size()[0]; ++i)
          sparse.insert(rows[i] - 1, 0) = values[i];

        rows_tensor->release_storage();
      }
      else
      {
        auto row_offsets_tensor = th::get_field<th::Tensor<int>*>(module_data, name + "_row_offsets");
        auto cols_tensor = th::get_field<th::Tensor<int>*>(module_data, name + "_cols");

        const int* row_offsets = get_tensor_data(row_offsets_tensor);
        const int* cols = get_tensor_data(cols_tensor);

        size_t cur_row = 0;
        size_t cur_row_size = row_offsets[cur_row + 1] - row_offsets[cur_row];

        for (int i = 0; i < values_tensor->get_size()[0]; ++i)
        {
          while (cur_row_size == 0)
          {
            cur_row++;
            cur_row_size = row_offsets[cur_row + 1] - row_offsets[cur_row];
          }

          sparse.insert(cur_row, cols[i] - 1) = values[i];
          cur_row_size--;
        }

        row_offsets_tensor->release_storage();
        cols_tensor->release_storage();
      }

      values_tensor->release_storage();
      sizes_tensor->release_storage();

      return sparse;
    }
  };

}
