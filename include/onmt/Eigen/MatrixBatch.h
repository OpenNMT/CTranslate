#pragma once

#include <Eigen/Dense>

namespace onmt
{
  namespace Eigen
  {

    template <typename T>
    using RowMajorMat = ::Eigen::Matrix<T, ::Eigen::Dynamic, ::Eigen::Dynamic, ::Eigen::RowMajor>;

    template <typename T>
    using MatrixBatchBase = RowMajorMat<T>;

    template <typename T>
    using Map = ::Eigen::Map<T>;

    // This class inherits from Eigen::Matrix to simulate a batch of Matrix
    // (a.k.a. a 3D Tensor). The object stores a hidden dimension, for example:
    //
    // Eigen::MatrixBatch mat(2, 4000); // 2x4000
    // mat.setHiddenDim(4); // virtually 2x4x1000 but still 2x4000
    //
    // When setHiddenDim is not called, the class behaves like a standard Matrix.
    template <typename T>
    class MatrixBatch: public MatrixBatchBase<T>
    {
    public:
      using MatrixBatchBase<T>::MatrixBatchBase;

      MatrixBatch()
        : MatrixBatchBase<T>()
      {
      }

      template<typename OtherDerived>
      MatrixBatch(const ::Eigen::MatrixBase<OtherDerived>& other)
        : MatrixBatchBase<T>(other)
      {
      }

      template<typename OtherDerived>
      MatrixBatch& operator=(const ::Eigen::MatrixBase<OtherDerived>& other)
      {
        this->MatrixBatchBase<T>::operator=(other);
        return *this;
      }

      void setHiddenDim(size_t size)
      {
        _rows = size;
        _cols = this->cols() / size;
        _ndim = 3;
      }

      void resetHiddenDim()
      {
        _ndim = 2;
      }

      MatrixBatchBase<T> batch(size_t b) const
      {
        if (_ndim == 3)
          return Map<const MatrixBatchBase<T> >(this->row(b).data(), _rows, _cols);
        else
          return this->row(b);
      }

      MatrixBatchBase<T> sum(int dimension) const
      {
        size_t new_cols = 0;

        if (dimension == 2)
        {
          new_cols = _cols;
        }
        else if (dimension == 3)
        {
          new_cols = _rows;
        }

        MatrixBatchBase<T> out(this->batches(), new_cols);

        for (size_t b = 0; b < this->batches(); ++b)
        {
          Map<const MatrixBatchBase<T> > mat(this->row(b).data(), _rows, _cols);
          if (dimension == 2)
            out.row(b).noalias() = mat.colwise().sum();
          else if (dimension == 3)
            out.row(b).noalias() = mat.rowwise().sum();
        }

        return out;
      }

      void squeeze(int dimension)
      {
        if (dimension == 2 && _rows == 1)
          resetHiddenDim();
        else if (dimension == 3 && _cols == 1)
          resetHiddenDim();
      }

      void assign(size_t b, MatrixBatch& mat)
      {
        this->row(b).noalias() = Map<MatrixBatchBase<T> >(mat.data(), 1, mat.cols() * mat.rows());
      }

      size_t batches() const
      {
        return this->rows();
      }

      size_t virtualRows() const
      {
        if (_ndim == 3)
          return _rows;
        else
          return 1;
      }

      size_t virtualCols() const
      {
        if (_ndim == 3)
          return _cols;
        else
          return this->cols();
      }

      std::ostream& printSizes(std::ostream& os) const
      {
        os << this->rows() << "x";

        if (_ndim == 3)
          os << _rows << "x" << _cols;
        else
          os << this->cols();

        os << std::endl;

        return os;
      }

    private:
      size_t _ndim;
      size_t _rows;
      size_t _cols;
    };

  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, std::vector<Eigen::MatrixBatch<T> > table)
  {
    os << "{" << std::endl;

    for (const auto& mat: table)
    {
      os << "  MatrixBatch<" << typeid(T).name() << ">: ";
      mat.printSizes(os);
    }

    os << "}";

    return os;
  }

}
