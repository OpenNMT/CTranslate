#include "onmt/TranslatorFactory.h"

namespace onmt
{

  std::unique_ptr<ITranslator> TranslatorFactory::build(const std::string& model,
                                                        const std::string& phrase_table,
                                                        bool replace_unk,
                                                        size_t max_sent_length,
                                                        size_t beam_size,
                                                        bool cuda,
                                                        bool profiling)
  {
    ITranslator* t = nullptr;

    t = new DefaultTranslator<float>(model, phrase_table, replace_unk, max_sent_length, beam_size, cuda, profiling);

    return std::unique_ptr<ITranslator>(t);
  }

}
