#include <iostream>

#include <boost/program_options.hpp>
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>
#include <exception>
#include <cmath>
#include <iomanip>
#include <ios>

#include <onmt/onmt.h>
#include <onmt/Utils.h>
#ifdef WITH_BOOST_LOG
#  include <onmt/Logger.h>
#endif

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
    ("log_file", po::value<std::string>()->default_value(""), "path to the log file (write to standard output if not set)")
    ("disable_logs", po::bool_switch()->default_value(false), "if set, output nothing")
    ("log_level", po::value<std::string>()->default_value(""), "output logs at this level and above (accepted: DEBUG, INFO, WARNING, ERROR, NONE; default: INFO)")
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

#ifdef WITH_BOOST_LOG
  onmt::Logger::init(vm["log_file"].as<std::string>(), vm["disable_logs"].as<bool>(), vm["log_level"].as<std::string>());
#endif

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

                     std::vector<std::vector<std::string> > res;
                     std::vector<std::vector<float> > score;
                     std::vector<std::vector<size_t> > count_tgt_words, count_tgt_unk_words;
                     std::vector<size_t> count_src_words, count_src_unk_words;
                     try
                     {
                       res = p_trans->get_translations_batch(batch.get_input(), score, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words);
                     }
                     catch (const std::exception& e)
                     {
                       ONMT_LOG_STREAM_SEV(e.what(), boost::log::trivial::error);
                       throw;
                     }

                     batch.set_result(res, score, count_tgt_words, count_tgt_unk_words, count_src_words, count_src_unk_words);
                     p_writer->write(batch);
                   }
                   return true;
                 },
                 reader.get(),
                 writer.get(),
                 translator.get()));
  }

  for (auto& f: futures)
    f.wait();

  ONMT_LOG_STREAM_SEV("Translated " << writer->total_count_src_words() << " words, src unk count: " << writer->total_count_src_unk_words()
    << ", coverage: " << std::floor(writer->total_count_src_unk_words() * 1000.0 / writer->total_count_src_words()) / 10.0 << "%, "
    << "tgt words: " << writer->total_count_tgt_words() << " words, tgt unk count: " << writer->total_count_tgt_unk_words()
    << ", coverage: " << std::floor(writer->total_count_tgt_unk_words() * 1000.0 / writer->total_count_tgt_words()) / 10.0 << "%, ", boost::log::trivial::info);
  ONMT_LOG_STREAM_SEV("PRED AVG SCORE: " << std::fixed << std::setprecision(2) << writer->total_score() / writer->total_count_tgt_words()
    << ", PRED PPL: " << std::exp(-writer->total_score() / writer->total_count_tgt_words()), boost::log::trivial::info);

  if (vm["time"].as<bool>())
  {
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> sec = t2 - t1;
    size_t num_sents = reader->read_lines();
    std::cerr << "avg real (sentence/s)\t" << sec.count() / num_sents << std::endl;
  }

  return 0;
}
