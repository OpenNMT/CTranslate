#pragma once

#include <sstream>

namespace onmt
{
  namespace th
  {

    // Storage
    template <typename T>
    Storage<T>::Storage(const std::string& classname, int version)
      : TorchObj(classname, version)
      , _data(nullptr)
    {
    }

    template <typename T>
    Storage<T>::~Storage()
    {
      if (_data != nullptr)
        release();
    }

    template <typename T>
    const T* Storage<T>::get_data() const
    {
      return _data;
    }

    template <typename T>
    long Storage<T>::get_size() const
    {
      return _size;
    }

    template <typename T>
    void Storage<T>::release()
    {
      THFree(_data);
      _data = nullptr;
    }


    // Tensor
    template <typename T>
    Tensor<T>::Tensor(const std::string& classname, int version)
      : TorchObj(classname, version)
      , _n_dimension(0)
      , _size(nullptr)
      , _stride(nullptr)
      , _thstorage(nullptr)
    {
    }

    template <typename T>
    Tensor<T>::~Tensor()
    {
      THFree(_size);
      THFree(_stride);
    }

    template <typename T>
    Obj* Tensor<T>::get_storage() const
    {
      return _thstorage;
    }

    template <typename T>
    const long* Tensor<T>::get_size() const
    {
      return _size;
    }

    template <typename T>
    int Tensor<T>::get_dimension() const
    {
      return _n_dimension;
    }

    template <typename T>
    long Tensor<T>::get_storage_offset() const
    {
      return _storage_offset;
    }

    template <typename T>
    void Tensor<T>::release_storage() const
    {
      dynamic_cast<Storage<T>*>(_thstorage)->release();
    }

    template <typename T>
    void Tensor<T>::read(THFile* tf, Env& env)
    {
      if (!tf)
        return;

      _n_dimension = THFile_readIntScalar(tf);
      _size = reinterpret_cast<long*>(THAlloc(dfLongSize * _n_dimension));
      _stride = reinterpret_cast<long*>(THAlloc(dfLongSize * _n_dimension));
      THFile_readLongRaw(tf, _size, _n_dimension);
      THFile_readLongRaw(tf, _stride, _n_dimension);
      _storage_offset = THFile_readLongScalar(tf)-1;
      _thstorage = read_obj(tf, env);
    }

  }
}
