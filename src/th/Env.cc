#include "onmt/th/Env.h"

namespace onmt
{
  namespace th
  {

    Env::~Env()
    {
      for (const auto& pair: _idx_obj)
        delete pair.second;

      for (const auto& obj: _list_obj)
        delete obj;
    }

    Obj* Env::get_object(int index)
    {
      auto it = _idx_obj.find(index);

      if (it != _idx_obj.end())
        return it->second;

      return nullptr;
    }

    void Env::set_object(Obj* thobj, int index)
    {
      if (index >= 0)
        _idx_obj[index] = thobj;
      else
        _list_obj.push_back(thobj);
    }

  }
}
