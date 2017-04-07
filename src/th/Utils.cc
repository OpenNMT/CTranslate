#include "onmt/th/Utils.h"

namespace onmt
{
  namespace th
  {

    int get_number(Table* module_data, const std::string& name)
    {
      return get_scalar<int>(module_data, name);
    }

    bool get_boolean(Table* module_data, const std::string& name)
    {
      Boolean* b = get_field<Boolean*>(module_data, name);
      return b && b->get_value();
    }

  }
}
