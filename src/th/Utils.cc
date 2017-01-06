#include "onmt/th/Utils.h"

namespace onmt
{
  namespace th
  {

    int get_number(Table* module_data, const std::string& name)
    {
      Number* dim = get_field<Number*>(module_data, name);
      return dim ? static_cast<int>(dim->get_value()) : -1;
    }

    bool get_boolean(Table* module_data, const std::string& name)
    {
      Boolean* b = get_field<Boolean*>(module_data, name);
      return b && b->get_value();
    }

  }
}
