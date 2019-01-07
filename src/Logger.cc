#include "onmt/Logger.h"

#include <memory>

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/support/date_time.hpp>

namespace onmt
{

  boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level> Logger::_lg;

  void Logger::init(const std::string& log_file, bool disable_logs, const std::string& log_level)
  {
    if (disable_logs || log_level == "NONE")
    {
      boost::log::core::get()->set_logging_enabled(false);
    }
    else
    {
      if (!log_file.empty())
      {
        boost::log::add_file_log
        (
          boost::log::keywords::file_name = log_file,
          boost::log::keywords::format =
          (
            boost::log::expressions::stream
              << '[' << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S")
              << ' ' << boost::log::trivial::severity
              << "] " << boost::log::expressions::smessage
          )
        );
      }

      boost::log::trivial::severity_level level = boost::log::trivial::info;
      if (!log_level.empty() && log_level != "INFO")
      {
        if (log_level == "DEBUG")
          level = boost::log::trivial::debug;
        else if (log_level == "WARNING")
          level = boost::log::trivial::warning;
        else if (log_level == "ERROR")
          level = boost::log::trivial::error;
        else
          std::cerr << "invalid log level specified: " << log_level << "; using default log level" << std::endl;
      }

      auto core = boost::log::core::get();
      core->set_filter
      (
        boost::log::trivial::severity >= level
      );

      core->add_global_attribute(boost::log::aux::default_attribute_names::timestamp(), boost::log::attributes::local_clock());
    }
  }

  boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level>& Logger::lg()
  {
    return _lg;
  }

}
