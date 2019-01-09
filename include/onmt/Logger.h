#pragma once

#include "onmt/onmt_export.h"
#include <string>

#include <boost/log/trivial.hpp>

namespace onmt
{

  class ONMT_EXPORT Logger
  {
  public:
    static void init(const std::string& log_file, bool disable_logs, const std::string& log_level);
    static boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level>& lg();

  private:
    static boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level> _lg;
  };

}
