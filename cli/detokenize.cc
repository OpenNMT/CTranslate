#include <iostream>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include <onmt/onmt.h>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::options_description desc("Detokenization");
  desc.add_options()
    ("help", "display available options")
    ("joiner", po::value<std::string>()->default_value(onmt::Tokenizer::joiner_marker), "character used to annotate joiners")
    ("case_feature", po::bool_switch()->default_value(false), "first feature is the case")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cerr << desc << std::endl;
    return 1;
  }

  onmt::Tokenizer tokenizer(vm["case_feature"].as<bool>(), vm["joiner"].as<std::string>());

  std::string line;

  while (std::getline(std::cin, line))
  {
    if (!line.empty())
    {
      std::vector<std::string> tokens;
      boost::split(tokens, line, boost::is_any_of(" "));

      std::cout << tokenizer.detokenize(tokens);
    }

    std::cout << std::endl;
  }

  return 0;
}
