#pragma once

#include "onmt/th/Obj.h"

namespace onmt
{
  namespace th
  {

    int get_number(Table* module_data, const std::string& name);
    bool get_boolean(Table* module_data, const std::string& name);

    template <typename T>
    T get_field(Obj* obj, const std::string& name)
    {
      auto table = dynamic_cast<Table*>(obj);

      if (!table)
        return nullptr;

      auto it = table->get_object().find(name);

      if (it == table->get_object().end())
        return nullptr;

      return dynamic_cast<T>(it->second);
    }

    template <typename T>
    T get_scalar(Table* module_data, const std::string& name)
    {
      Number* dim = get_field<Number*>(module_data, name);
      return dim ? static_cast<T>(dim->get_value()) : -1;
    }

    template <typename T>
    std::vector<T> get_storage_as_vector(Obj* obj, const std::string& name)
    {
      Storage<T>* storage = get_field<Storage<T>*>(obj, name);

      const T* data = storage->get_data();
      auto size = storage->get_size();

      std::vector<T> vec;
      vec.reserve(size);

      for (int i = 0; i < size; ++i)
        vec.push_back(data[i]);

      return vec;
    }

    template <typename T>
    const T* get_tensor_data(Tensor<T>* tensor)
    {
      auto storage = dynamic_cast<Storage<T>*>(tensor->get_storage());
      return storage->get_data() + tensor->get_storage_offset();
    }

  }
}
