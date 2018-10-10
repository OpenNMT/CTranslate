#pragma once

#include <map>
#include <vector>

#include "onmt/onmt_export.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace th
  {

    class ONMT_EXPORT Env
    {
    public:
      ~Env();

      Obj* get_object(int index);
      void set_object(Obj* obj, int index = -1);

    private:
      std::map<int, Obj*> _idx_obj;
      std::vector<Obj*> _list_obj;
    };

  }
}
