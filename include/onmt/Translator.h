#pragma once

#include "Eigen/MatrixBatch.h"

#include "Model.h"
#include "Dictionary.h"
#include "PhraseTable.h"
#include "ITranslator.h"
#include "SubDict.h"

namespace onmt
{

  template <typename MatFwd = Eigen::MatrixBatch<float>,
            typename MatIn = Eigen::Map<const Eigen::RowMajorMat<float> >,
            typename MatEmb = Eigen::Map<const Eigen::RowMajorMat<float> >,
            typename ModelT = float>
  class Translator: public ITranslator
  {
  public:
    friend class TranslatorFactory;

    std::string translate(const std::string& text, ITokenizer& tokenizer) override;
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts, ITokenizer& tokenizer) override;

    TranslationResult
    translate(const std::vector<std::string>& tokens,
              const std::vector<std::vector<std::string> >& features) override;

    TranslationResult
    translate_batch(const std::vector<std::vector<std::string> >& batch_tokens,
                    const std::vector<std::vector<std::vector<std::string> > >& batch_features) override;

  protected:
    Translator(const std::string& model,
               const std::string& phrase_table,
               const std::string& vocab_mapping,
               bool replace_unk,
               size_t max_sent_length,
               size_t beam_size,
               bool cuda,
               bool qlinear,
               bool profiling);
    Translator(const Translator& other);

    /* profiling - starting first to profile load time */
    bool _profiling;
    Profiler _profiler;

    // Members shared accross translator instances.
    std::shared_ptr<Model<MatFwd, MatIn, MatEmb, ModelT>> _model;
    std::shared_ptr<const PhraseTable> _phrase_table;
    std::shared_ptr<const SubDict> _subdict;

    bool _cuda;
    bool _qlinear;

    bool _replace_unk;
    size_t _max_sent_length;
    size_t _beam_size;

    std::vector<MatFwd>
    get_encoder_input(size_t t,
                      const std::vector<std::vector<size_t> >& batch_ids,
                      const std::vector<std::vector<std::vector<size_t> > >& batch_feat_ids,
                      const std::vector<MatFwd>& rnn_state_enc) const;

    void
    encode(const std::vector<std::vector<std::string> >& batch_tokens,
           const std::vector<std::vector<size_t> >& batch_ids,
           const std::vector<std::vector<std::vector<size_t> > >& batch_feat_ids,
           std::vector<MatFwd>& rnn_state_enc,
           MatFwd& context);

    TranslationResult
    decode(const std::vector<std::vector<std::string> >& batch_tokens,
           size_t source_l,
           std::vector<MatFwd>& rnn_state_enc,
           MatFwd& context,
           const std::vector<size_t>& subvocab);

  private:
    void init_graph();

    nn::ModuleFactory<MatFwd, MatIn, MatEmb, ModelT> _factory;
    nn::Module<MatFwd>* _encoder;
    nn::Module<MatFwd>* _encoder_bwd;
    nn::Module<MatFwd>* _decoder;
    nn::Module<MatFwd>* _generator;
  };


  template <typename T>
  using DefaultTranslator = Translator<Eigen::MatrixBatch<T>,
                                       Eigen::Map<const Eigen::RowMajorMat<T> >,
                                       Eigen::Map<const Eigen::RowMajorMat<T> >,
                                       T>;

}

#include "Translator.hxx"
