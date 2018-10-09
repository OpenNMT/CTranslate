#include <iostream>

#include <boost/program_options.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>

#include <onmt/onmt.h>

#include "BatchReader.h"
#include "BatchWriter.h"

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::options_description desc("OpenNMT Translator");
  desc.add_options()
    ("help", "display available options")
    ("model", po::value<std::string>(), "path to the OpenNMT model")
    ("src", po::value<std::string>(), "path to the file to translate (read from the standard input if not set)")
    ("tgt", po::value<std::string>(), "path to the output file (write to the standard output if not set")
    ("phrase_table", po::value<std::string>()->default_value(""), "path to the phrase table")
    ("vocab_mapping", po::value<std::string>()->default_value(""), "path to a vocabulary mapping table")
    ("replace_unk", po::bool_switch()->default_value(false), "replace unknown tokens by source tokens with the highest attention")
    ("batch_size", po::value<size_t>()->default_value(30), "batch size")
    ("beam_size", po::value<size_t>()->default_value(5), "beam size")
    ("n_best", po::value<size_t>()->default_value(1), "n best")
    ("max_sent_length", po::value<size_t>()->default_value(250), "maximum sentence length to produce")
    ("time", po::bool_switch()->default_value(false), "output average translation time")
    ("profiler", po::bool_switch()->default_value(false), "output per module computation time")
    ("parallel", po::value<size_t>()->default_value(1), "number of parallel translator")
    ("threads", po::value<size_t>()->default_value(0), "number of threads to use (set to 0 to use the number defined by OpenMP)")
    ("cuda", po::bool_switch()->default_value(false), "use cuda when available")
    ("qlinear", po::bool_switch()->default_value(false), "use quantized linear for speed-up")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cerr << desc << std::endl;
    return 1;
  }

  if (!vm.count("model"))
  {
    std::cerr << "missing model" << std::endl;
    return 1;
  }

  if (vm["threads"].as<size_t>() > 0)
    onmt::Threads::set(vm["threads"].as<size_t>());

  std::vector<std::unique_ptr<onmt::ITranslator>> translator_pool;
  translator_pool.emplace_back(onmt::TranslatorFactory::build(vm["model"].as<std::string>(),
                                                              vm["phrase_table"].as<std::string>(),
                                                              vm["vocab_mapping"].as<std::string>(),
                                                              vm["replace_unk"].as<bool>(),
                                                              vm["max_sent_length"].as<size_t>(),
                                                              vm["beam_size"].as<size_t>(),
                                                              vm["n_best"].as<size_t>(),
                                                              vm["cuda"].as<bool>(),
                                                              vm["qlinear"].as<bool>(),
                                                              vm["profiler"].as<bool>()));
  for (size_t i = 0; i < vm["parallel"].as<size_t>() - 1; ++i) {
    translator_pool.emplace_back(onmt::TranslatorFactory::clone(translator_pool.front()));
  }

  std::unique_ptr<BatchReader> reader;
  if (vm.count("src"))
    reader.reset(new BatchReader(vm["src"].as<std::string>(), vm["batch_size"].as<size_t>()));
  else
    reader.reset(new BatchReader(std::cin, vm["batch_size"].as<size_t>()));

  std::unique_ptr<BatchWriter> writer;
  if (vm.count("tgt"))
    writer.reset(new BatchWriter(vm["tgt"].as<std::string>()));
  else
    writer.reset(new BatchWriter(std::cout));

  std::chrono::high_resolution_clock::time_point t1, t2;

  if (vm["time"].as<bool>())
    t1 = std::chrono::high_resolution_clock::now();

  std::vector<std::future<bool>> futures;

  for (auto& translator: translator_pool)
  {
    futures.emplace_back(
      std::async(std::launch::async,
                 [](BatchReader* p_reader, BatchWriter* p_writer, onmt::ITranslator* p_trans)
                 {
                   while (true)
                   {
                     auto batch = p_reader->read_next();
                     if (batch.empty())
                       break;
                     auto res = p_trans->translate_batch(batch.get_data());
                     p_writer->write(BatchOutput(res, batch.get_id()));
                   }
                   return true;
                 },
                 reader.get(),
                 writer.get(),
                 translator.get()));
  }

  for (auto& f: futures)
    f.wait();

  if (vm["time"].as<bool>())
  {
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> sec = t2 - t1;
    size_t num_sents = reader->read_lines();
    std::cerr << "avg real (sentence/s)\t" << sec.count() / num_sents << std::endl;
  }

  return 0;
}
