#include "onmt/TranslatorFactory.h"

namespace onmt
{

  std::unique_ptr<ITranslator> TranslatorFactory::build(const std::string& model,
                                                        const std::string& phrase_table,
                                                        const std::string& vocab_mapping,
                                                        bool cuda,
                                                        bool qlinear,
                                                        bool profiling)
  {
    ITranslator* t = nullptr;

    t = new DefaultTranslator<float>(model,
                                     phrase_table,
                                     vocab_mapping,
                                     cuda,
                                     qlinear,
                                     profiling);

    return std::unique_ptr<ITranslator>(t);
  }

  std::unique_ptr<ITranslator>
  TranslatorFactory::clone(const std::unique_ptr<ITranslator>& translator)
  {
    ITranslator* t = new DefaultTranslator<float>(
      dynamic_cast<const DefaultTranslator<float>&>(*translator));
    return std::unique_ptr<ITranslator>(t);
  }

}
