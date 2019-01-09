#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <memory>

namespace onmt
{

  static std::vector<std::string> ids_to_words(const Dictionary& dict, const std::vector<size_t>& ids)
  {
    std::vector<std::string> words;
    words.reserve(ids.size());

    for (auto id: ids)
    {
      words.push_back(dict.get_id_word(id));
    }

    return words;
  }

  static std::vector<std::string> ids_to_words_replace(const Dictionary& dict,
                                                       const PhraseTable& phrase_table,
                                                       const std::vector<size_t>& ids,
                                                       const std::vector<std::string>& source,
                                                       const std::vector<std::vector<float> >& attn)
  {
    std::vector<std::string> words;
    words.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i)
    {
      if (ids[i] != Dictionary::unk_id)
        words.push_back(dict.get_id_word(ids[i]));
      else
      {
        // Get source word with the maximal attention.
        auto focus = std::max_element(attn[i].begin(), attn[i].end());
        size_t focus_on = std::distance(attn[i].begin(), focus);

        std::string replacement = source[focus_on];

        // If a phrase table is used, lookup its translation.
        if (!phrase_table.is_empty())
        {
          std::string tgt = phrase_table.lookup(replacement);
          if (!tgt.empty())
            replacement = tgt;
        }

        words.push_back(replacement);
      }
    }

    return words;
  }

  static std::vector<size_t> words_to_ids(const Dictionary& dict, const std::vector<std::string>& words)
  {
    std::vector<size_t> ids;
    ids.reserve(words.size());

    for (const auto& word: words)
    {
      ids.push_back(dict.get_word_id(word));
    }

    return ids;
  }

  static void pad_input(std::vector<size_t>& input, size_t pad_value, size_t max_len)
  {
    if (input.size() > max_len)
    {
      input.resize(max_len);
    }
    else if (input.size() < max_len)
    {
      input.reserve(max_len);
      size_t pad_size = max_len - input.size();
      input.insert(input.begin(), pad_size, pad_value);
    }
  }

  static size_t pad_inputs(std::vector<std::vector<size_t> >& inputs, size_t pad_value)
  {
    size_t cur_max_len = 0;

    for (const auto& input: inputs)
      cur_max_len = std::max(cur_max_len, input.size());

    for (auto& input: inputs)
      pad_input(input, pad_value, cur_max_len);

    return cur_max_len;
  }

  template <typename MatFwd>
  static std::vector<MatFwd> init_rnn_states(size_t num, size_t batch_size, size_t rnn_size)
  {
    std::vector<MatFwd> rnn_state;
    rnn_state.reserve(num);

    for (size_t l = 0; l < num; ++l)
    {
      rnn_state.emplace_back(batch_size, rnn_size);
      rnn_state.back().setZero();
    }

    return rnn_state;
  }

  template <typename MatFwd>
  static void mask_output(size_t t,
                          size_t source_l,
                          const std::vector<std::vector<std::string> >& inputs,
                          std::vector<MatFwd>& out)
  {
    for (size_t b = 0; b < inputs.size(); ++b)
    {
      if (t < source_l - inputs[b].size())
      {
        for (size_t j = 0; j < out.size(); ++j)
          out[j].row(b).setZero();
      }
    }
  }


  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  Translator<MatFwd, MatIn, MatEmb, ModelT>::Translator(const std::string& model,
                                                        const std::string& phrase_table,
                                                        const std::string& vocab_mapping,
                                                        bool replace_unk,
                                                        size_t max_sent_length,
                                                        size_t beam_size,
                                                        size_t n_best,
                                                        bool cuda,
                                                        bool qlinear,
                                                        bool profiling)
    : _profiling(profiling)
    , _profiler(profiling, true)
    , _model(new Model<MatFwd, MatIn, MatEmb, ModelT>(model))
    , _phrase_table(new PhraseTable(phrase_table))
    , _subdict(new SubDict(vocab_mapping, _model->get_tgt_dict()))
    , _cuda(cuda)
    , _qlinear(qlinear)
    , _replace_unk(replace_unk)
    , _max_sent_length(max_sent_length)
    , _beam_size(beam_size)
    , _n_best(n_best)
    , _factory(_profiler, cuda, qlinear)
  {
    init_graph();
    _profiler.stop("Initialization");
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  Translator<MatFwd, MatIn, MatEmb, ModelT>::Translator(
    const Translator<MatFwd, MatIn, MatEmb, ModelT>& other)
    : _profiling(other._profiling)
    , _profiler(_profiling, true)
    , _model(other._model)
    , _phrase_table(other._phrase_table)
    , _subdict(other._subdict)
    , _cuda(other._cuda)
    , _qlinear(other._qlinear)
    , _replace_unk(other._replace_unk)
    , _max_sent_length(other._max_sent_length)
    , _beam_size(other._beam_size)
    , _n_best(other._n_best)
    , _factory(_profiler, _cuda, _qlinear)
  {
    init_graph();
    _profiler.stop("Initialization");
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void Translator<MatFwd, MatIn, MatEmb, ModelT>::init_graph()
  {
    std::vector<nn::Module<MatFwd>*> encoder;
    std::vector<nn::Module<MatFwd>*> decoder;
    _model->create_graph(_factory, encoder, decoder);
    _encoder = encoder[0];
    _encoder_bwd = encoder.size() > 1 ? encoder[1] : nullptr;
    _decoder = decoder[0];
    _generator = decoder[1];
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  std::vector<std::string>
  Translator<MatFwd, MatIn, MatEmb, ModelT>::get_translations(const std::string& text,
                                                              ITokenizer& tokenizer,
                                                              std::vector<float>& scores,
                                                              std::vector<size_t>& count_tgt_words,
                                                              std::vector<size_t>& count_tgt_unk_words,
                                                              size_t& count_src_words,
                                                              size_t& count_src_unk_words)
  {
    std::vector<std::string> src_tokens;
    std::vector<std::vector<std::string> > src_features;

    tokenizer.tokenize(text, src_tokens, src_features);

    count_src_words = src_tokens.size();
    TranslationResult res = translate(src_tokens, src_features, count_src_unk_words);
    std::vector<std::string> tgt_texts;
    size_t count_results = res.count_job(0);
    tgt_texts.reserve(count_results);
    count_tgt_words.clear();
    count_tgt_words.reserve(count_results);

    for (size_t i = 0; i < count_results; ++i)
    {
      const auto& words = res.get_words(0, i);
      count_tgt_words.push_back(words.size());
      if (res.has_features())
        tgt_texts.push_back(tokenizer.detokenize(words, res.get_features(0, i)));
      else
        tgt_texts.push_back(tokenizer.detokenize(words, {}));
    }

    scores = res.get_score_job(0);
    count_tgt_unk_words = res.get_count_unk_words_job(0);
    return tgt_texts;
  }

  class tdict
  {
  public:
    tdict(size_t n)
      : _ndict(n)
    {}
    size_t _ndict;
    std::vector<size_t> subvocab;
  };

  template <typename MatFwd, typename MatIn, typename, typename ModelT>
  void* reduce_vocabulary(nn::Module<MatFwd>* M, void* t)
  {
    if (M->get_name() == "nn.Linear")
    {
      nn::Linear<MatFwd, MatIn, ModelT>* mL = (nn::Linear<MatFwd, MatIn, ModelT>*)M;
      tdict* data = (tdict*)t;
      if (mL->get_weight_rows() == data->_ndict)
        mL->apply_subdictionary(data->subvocab);
    }
    return 0;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  TranslationResult
  Translator<MatFwd, MatIn, MatEmb, ModelT>::translate(const std::vector<std::string>& tokens,
                                                       const std::vector<std::vector<std::string> >& features,
                                                       size_t& count_src_unk_words)
  {
    std::vector<std::vector<std::string> > src_batch(1, tokens);
    std::vector<std::vector<std::vector<std::string> > > src_feat_batch(1, features);
    std::vector<size_t> batch_count_src_unk_words;
    auto res = translate_batch(src_batch, src_feat_batch, batch_count_src_unk_words);
    count_src_unk_words = batch_count_src_unk_words.at(0);
    return res;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  std::vector<std::vector<std::string> >
  Translator<MatFwd, MatIn, MatEmb, ModelT>::get_translations_batch(const std::vector<std::string>& texts,
                                                                    ITokenizer& tokenizer,
                                                                    std::vector<std::vector<float> >& scores,
                                                                    std::vector<std::vector<size_t> >& count_tgt_words,
                                                                    std::vector<std::vector<size_t> >& count_tgt_unk_words,
                                                                    std::vector<size_t>& count_src_words,
                                                                    std::vector<size_t>& count_src_unk_words)
  {
    std::vector<std::vector<std::string> > batch_tokens;
    std::vector<std::vector<std::vector<std::string> > > batch_features;
    count_src_words.clear();

    for (const auto& text: texts)
    {
      std::vector<std::string> tokens;
      std::vector<std::vector<std::string> > features;
      tokenizer.tokenize(text, tokens, features);
      batch_tokens.push_back(tokens);
      batch_features.push_back(features);
      count_src_words.push_back(tokens.size());
    }

    TranslationResult res = translate_batch(batch_tokens, batch_features, count_src_unk_words);

    std::vector<std::vector<std::string> > tgt_texts;
    tgt_texts.reserve(texts.size());
    count_tgt_words.clear();
    count_tgt_words.reserve(texts.size());

    for (size_t i = 0; i < res.count(); ++i)
    {
      size_t count_results = res.count_job(i);
      tgt_texts.emplace_back();
      tgt_texts[i].reserve(count_results);
      count_tgt_words.emplace_back();
      count_tgt_words[i].reserve(count_results);
      for (size_t j = 0; j < count_results; ++j)
      {
        const auto& words = res.get_words(i, j);
        count_tgt_words[i].push_back(words.size());
        if (res.has_features())
          tgt_texts[i].push_back(tokenizer.detokenize(words, res.get_features(i, j)));
        else
          tgt_texts[i].push_back(tokenizer.detokenize(words, {}));
      }
    }

    scores = res.get_score_batch();
    count_tgt_unk_words = res.get_count_unk_words_batch();
    return tgt_texts;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  TranslationResult
  Translator<MatFwd, MatIn, MatEmb, ModelT>::translate_batch(
    const std::vector<std::vector<std::string> >& batch_tokens,
    const std::vector<std::vector<std::vector<std::string> > >& batch_features,
    std::vector<size_t>& batch_count_src_unk_words)
  {
    size_t batch_size = batch_tokens.size();
    batch_count_src_unk_words.clear();

    // Convert words to ids.
    std::vector<std::vector<size_t> > batch_ids;
    std::vector<std::vector<std::vector<size_t> > > batch_feat_ids;

    tdict data(_model->get_tgt_dict().get_size());
    if (!_subdict->empty())
    {
      std::set<size_t> e;
      for (const auto& it: batch_tokens)
        _subdict->extract(it, e);
      /* convert into vector */
      for (auto idx: e)
        data.subvocab.push_back(idx);
      /* modify generator weights and bias accordingly */
      _generator->apply(reduce_vocabulary<MatFwd, MatIn, MatEmb, ModelT>, &data);
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_ids.push_back(words_to_ids(_model->get_src_dict(), batch_tokens[b]));

      if (_model->get_src_feat_dicts().size() != batch_features[b].size())
        throw std::runtime_error("expected "
                                 + std::to_string(_model->get_src_feat_dicts().size())
                                 + " word feature(s), got "
                                 + std::to_string(batch_features[b].size())
                                 + " instead");
      else if (_model->get_src_feat_dicts().size() > 0)
      {
        batch_feat_ids.emplace_back();
        for (size_t j = 0; j < _model->get_src_feat_dicts().size(); ++j)
          batch_feat_ids[b].push_back(words_to_ids(_model->get_src_feat_dicts()[j], batch_features[b][j]));
      }

      batch_count_src_unk_words.push_back(std::count(batch_ids[b].begin(), batch_ids[b].end(), Dictionary::unk_id));
    }

    // Pad inputs to the left.
    size_t source_l = pad_inputs(batch_ids, Dictionary::pad_id);

    for (size_t j = 0; j < _model->get_src_feat_dicts().size(); ++j)
    {
      for (size_t b = 0; b < batch_size; ++b)
        pad_input(batch_feat_ids[b][j], Dictionary::pad_id, source_l);
    }

    // Encode source sequences then decode tgtet sequences.
    std::vector<MatFwd> rnn_state_enc;
    MatFwd context;

    encode(batch_tokens, batch_ids, batch_feat_ids, rnn_state_enc, context);
    return decode(batch_tokens, source_l, rnn_state_enc, context, data.subvocab);
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  std::vector<MatFwd>
  Translator<MatFwd, MatIn, MatEmb, ModelT>::get_encoder_input(
    size_t t,
    const std::vector<std::vector<size_t> >& batch_ids,
    const std::vector<std::vector<std::vector<size_t> > >& batch_feat_ids,
    const std::vector<MatFwd>& rnn_state_enc) const
  {
    size_t batch_size = batch_ids.size();

    std::vector<MatFwd> input;

    // Previous states
    for (size_t l = 0; l < rnn_state_enc.size(); ++l)
      input.push_back(rnn_state_enc[l]);

    // Words and features
    input.emplace_back(batch_size, 1 + _model->get_src_feat_dicts().size());
    for (size_t b = 0; b < batch_size; ++b)
    {
      input.back()(b, 0) = batch_ids[b][t];

      for (size_t j = 0; j < _model->get_src_feat_dicts().size(); ++j)
        input.back()(b, j + 1) = batch_feat_ids[b][j][t];
    }

    return input;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  void
  Translator<MatFwd, MatIn, MatEmb, ModelT>::encode(
    const std::vector<std::vector<std::string> >& batch_tokens,
    const std::vector<std::vector<size_t> >& batch_ids,
    const std::vector<std::vector<std::vector<size_t> > >& batch_feat_ids,
    std::vector<MatFwd>& rnn_state_enc,
    MatFwd& context)
  {
    size_t batch_size = batch_ids.size();
    size_t source_l = batch_ids[0].size();

    size_t num_layers = _model->template get_option_value<size_t>("layers");
    size_t rnn_size = _model->template get_option_value<size_t>("rnn_size");
    bool brnn = _model->get_option_string("encoder_type") == "brnn" || _model->get_option_flag("brnn");
    const std::string& brnn_merge = _model->get_option_string("brnn_merge");
    const std::string& rnn_type = _model->get_option_string("rnn_type");
    size_t num_states = num_layers * ((rnn_type.empty() || rnn_type == "LSTM") ? 2 : 1);

    if (brnn && brnn_merge == "concat")
      rnn_size /= 2;

    rnn_state_enc = init_rnn_states<MatFwd>(num_states, batch_size, rnn_size);
    context.resize(batch_size, source_l * rnn_size);

    for (size_t i = 0; i < source_l; ++i)
    {
      std::vector<MatFwd> input = get_encoder_input(i, batch_ids, batch_feat_ids, rnn_state_enc);

      rnn_state_enc = _encoder->forward(input);

      // Ignore outputs if the related input was padded.
      if (batch_size > 1)
        mask_output(i, source_l, batch_tokens, rnn_state_enc);

      MatFwd& rnn_out = rnn_state_enc.back();

      // Update context for this timestep (i.e. save the final output of the encoder for each batch).
      context.block(0, i * rnn_size, batch_size, rnn_size).noalias() = rnn_out;
    }

    if (brnn)
    {
      std::vector<MatFwd> final_rnn_state_enc_bwd =
        init_rnn_states<MatFwd>(rnn_state_enc.size(), batch_size, rnn_size);
      std::vector<MatFwd> rnn_state_enc_bwd =
        init_rnn_states<MatFwd>(rnn_state_enc.size(), batch_size, rnn_size);

      MatFwd context_bwd;
      context_bwd.resize(batch_size, source_l * rnn_size);

      for (int i = source_l - 1; i >= 0; --i)
      {
        std::vector<MatFwd> input = get_encoder_input(i, batch_ids, batch_feat_ids, rnn_state_enc_bwd);

        rnn_state_enc_bwd = _encoder_bwd->forward(input);

        // Ignore outputs if the related input was padded.
        if (batch_size > 1)
          mask_output(i, source_l, batch_tokens, rnn_state_enc_bwd);

        for (size_t b = 0; b < batch_size; ++b)
        {
          // All words of this sequence were forwarded so add its backward encoding
          // to the forward encoding.
          if (static_cast<size_t>(i) == source_l - batch_tokens[b].size())
          {
            for (size_t l = 0; l < rnn_state_enc.size(); ++l)
              final_rnn_state_enc_bwd[l].row(b).noalias() = rnn_state_enc_bwd[l].row(b);
          }
        }

        MatFwd& rnn_out = rnn_state_enc_bwd.back();
        context_bwd.block(0, i * rnn_size, batch_size, rnn_size).noalias() = rnn_out;
      }

      // Merge backward states with the forward states.
      for (size_t l = 0; l < rnn_state_enc.size(); ++l)
      {
        if (brnn_merge == "sum")
          rnn_state_enc[l].noalias() += final_rnn_state_enc_bwd[l];
        else
        {
          MatFwd state(rnn_state_enc[l].rows(), rnn_size * 2);
          state << rnn_state_enc[l], final_rnn_state_enc_bwd[l];
          rnn_state_enc[l] = state;
        }
      }

      if (brnn_merge == "sum")
        context.noalias() += context_bwd;
      else
      {
        size_t full_rnn_size = rnn_size * 2;
        MatFwd new_context(context.rows(), context.cols() + context_bwd.cols());
        for (size_t i = 0; i < source_l; ++i)
        {
          new_context.block(0, i * full_rnn_size, batch_size, rnn_size).noalias()
            = context.block(0, i * rnn_size, batch_size, rnn_size);
          new_context.block(0, i * full_rnn_size + rnn_size, batch_size, rnn_size).noalias()
            = context_bwd.block(0, i * rnn_size, batch_size, rnn_size);
        }

        context = new_context;
      }
    }
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  TranslationResult
  Translator<MatFwd, MatIn, MatEmb, ModelT>::decode(
    const std::vector<std::vector<std::string> >& batch_tokens,
    size_t source_l,
    const std::vector<MatFwd>& rnn_state_enc,
    const MatFwd& context,
    const std::vector<size_t>& subvocab)
  {
    if (_beam_size < _n_best)
      throw std::runtime_error("Beam size must be greater than or equal to the n-best list size");
    else if (_n_best == 0)
      throw std::runtime_error("N-best list size must not be zero");

    size_t batch_size = batch_tokens.size();

    size_t rnn_size = _model->template get_option_value<size_t>("rnn_size");
    bool with_input_feeding = _model->get_option_flag("input_feed", true);

    MatFwd context_dec(context);
    context_dec.setHiddenDim(source_l);

    std::vector<size_t> beam_size(batch_size, 1); // for the first decoding step, beam size is 1
    size_t total_beam_size = batch_size;
    std::unique_ptr<MatFwd> input_feed(with_input_feeding ? new MatFwd(total_beam_size, rnn_size) : nullptr);
    if (with_input_feeding)
      input_feed->setZero();

    // Copy encoder states to decoder states.
    size_t copy_offset = 0;
    if (_model->get_option_string("bridge") == "last")
    {
      size_t enc_layers = _model->template get_option_value<size_t>("enc_layers");
      size_t dec_layers = _model->template get_option_value<size_t>("dec_layers");
      copy_offset = enc_layers - dec_layers;
    }

    std::vector<MatFwd> rnn_state_dec;
    rnn_state_dec.reserve(rnn_state_enc.size() - copy_offset);
    for (size_t l = copy_offset; l < rnn_state_enc.size(); ++l)
    {
      rnn_state_dec.emplace_back(total_beam_size, rnn_size);
      rnn_state_dec.back() = rnn_state_enc[l];
    }

    std::vector<std::vector<std::vector<size_t> > > next_ys; // b x n x K
    std::vector<std::vector<std::vector<size_t> > > prev_ks; // b x n x K
    std::vector<std::vector<std::vector<float> > > scores; // b x n x K
    std::vector<std::vector<std::vector<std::vector<size_t> > > > next_features; // b x n x j x K
    std::vector<std::vector<MatFwd> > all_attention; // b x n x [ K x source_l ]

    // Get a pointer to the attention softmax module to mask its output.
    nn::Module<MatFwd>* softmax_attn = _decoder->find("softmaxAttn");

    // Prepare data structures for the beam search.
    for (size_t b = 0; b < batch_size; ++b)
    {
      std::vector<size_t> in1(1, Dictionary::bos_id);
      next_ys.emplace_back(1, in1);

      std::vector<size_t> prev_ks1(1, 1);
      prev_ks.emplace_back(1, prev_ks1);

      std::vector<float> scores1(1, 0.0);
      scores.emplace_back(1, scores1);

      if (_model->get_tgt_feat_dicts().size() > 0)
      {
        next_features.emplace_back();
        next_features.back().emplace_back();
        for (size_t j = 0; j < _model->get_tgt_feat_dicts().size(); ++j)
        {
          std::vector<size_t> in_feat1(1, Dictionary::eos_id);
          next_features.back().back().push_back(in_feat1);
        }
      }

      std::vector<MatFwd> attn1;
      attn1.emplace_back(1, source_l);
      attn1.back().setZero();
      all_attention.push_back(attn1);
    }

    size_t remaining_sents = batch_size;
    std::vector<bool> done(batch_size, false);

    std::vector<std::vector<float> > max_score(batch_size, std::vector<float>(_n_best, -std::numeric_limits<float>::max()));
    std::vector<std::vector<size_t> > best_k(batch_size, std::vector<size_t>(_n_best, 0));
    std::vector<std::vector<size_t> > best_finished_at(batch_size, std::vector<size_t>(_n_best, 0));

    size_t i;

    for (i = 1; remaining_sents > 0 && i <= _max_sent_length + 1; ++i)
    {
      // Prepare decoder input at timestep i.
      std::vector<MatFwd> input;

      // States.
      for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        input.push_back(rnn_state_dec[l]);

      // Words and features.
      input.emplace_back(total_beam_size, 1 + _model->get_tgt_feat_dicts().size());
      for (size_t b = 0, batch_offset = 0; b < batch_size; ++b)
      {
        if (done[b])
          continue;

        for (size_t k = 0; k < next_ys[b][i - 1].size(); ++k)
        {
          if (next_ys[b][i - 1][k] == Dictionary::eos_id)
            continue;

          size_t index = batch_offset++;
          input.back()(index, 0) = next_ys[b][i - 1][k];

          for (size_t j = 0; j < _model->get_tgt_feat_dicts().size(); ++j)
            input.back()(index, j + 1) = next_features[b][i - 1][j][k];
        }
      }

      // Context and input feed.
      input.push_back(context_dec);
      if (with_input_feeding)
        input.push_back(*input_feed);

      MatFwd attn_softmax_out;

      // Mask attention softmax output depending on input sentence size.
      softmax_attn->post_process_fun() = [&] (std::vector<MatFwd>& out) {
        if (batch_size == 1)
        {
          attn_softmax_out = out[0];
          return;
        }

        MatFwd& soft_out = out[0];

        for (size_t b = 0, batch_offset = 0; b < batch_size; ++b)
        {
          if (done[b])
            continue;

          size_t pad_len = source_l - batch_tokens[b].size();

          if (pad_len > 0)
          {
            for (size_t k = 0; k < beam_size[b]; ++k)
            {
              size_t index = batch_offset + k;
              soft_out.row(index).head(pad_len).setZero();
              soft_out.row(index) /= soft_out.row(index).sum(); // Normalization (softmax output).
            }
          }

          batch_offset += beam_size[b];
        }

        attn_softmax_out = soft_out;
      };

      // Forward into the decoder.
      std::vector<MatFwd> out_decoder = _decoder->forward(input);
      size_t out_decoder_pred_idx = out_decoder.size() - 1;
      std::vector<MatFwd> gen_input(1, out_decoder[out_decoder_pred_idx]);
      std::vector<MatFwd> gen_out = _generator->forward(gen_input);

      // Update rnn_state_dec based on decoder output.
      if (with_input_feeding)
        *input_feed = out_decoder[out_decoder_pred_idx];
      for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        rnn_state_dec[l] = out_decoder[l];

      MatFwd& out = gen_out[0];

      size_t new_remaining_sents = remaining_sents;

      _profiler.start();

      // Update beam path for all non finished sentences.
      total_beam_size = 0;
      for (size_t b = 0, batch_offset = 0; b < batch_size; batch_offset += beam_size[b++])
      {
        if (done[b])
          continue;

        // Penalize beam scores based on the previous step.
        for (size_t k = 0, l = 0; k < next_ys[b][i - 1].size(); ++k)
        {
          if (next_ys[b][i - 1][k] == Dictionary::eos_id)
            continue;

          out.row(batch_offset + l++).array() += scores[b][i - 1][k];
        }

        size_t new_beam_size = (i > _max_sent_length) ? beam_size[b] : _beam_size;
        prev_ks[b].emplace_back(new_beam_size, 0);
        next_ys[b].emplace_back(new_beam_size, 0);
        scores[b].emplace_back(new_beam_size, 0.0);
        all_attention[b].emplace_back(new_beam_size, source_l);

        if (i > _max_sent_length)
        {
          for (size_t k = 0, l = 0; k < next_ys[b][i - 1].size(); ++k)
          {
            if (next_ys[b][i - 1][k] == Dictionary::eos_id)
              continue;

            prev_ks[b][i][l] = k;
            next_ys[b][i][l] = Dictionary::eos_id;
            scores[b][i][l] = scores[b][i - 1][k];

            // Store the attention.
            all_attention[b][i].row(l) = attn_softmax_out.row(batch_offset + l);
            ++l;
          }
        }
        else
        {
          std::unique_ptr<MatFwd> block((remaining_sents > 1) ? new MatFwd(out.block(batch_offset, 0, beam_size[b], out.cols())) : nullptr);
          MatFwd& cur_sent_out = block ? *block : out;
          for (size_t k = 0; k < new_beam_size; ++k)
          {
            // Pick the best score across all beams.
            size_t best_score_id = 0;
            size_t from_beam = 0, from_beam_offset = 0;
            float best_score = cur_sent_out.maxCoeff(&from_beam_offset, &best_score_id);
            for (size_t l = 0, m = 0; l < next_ys[b][i - 1].size(); ++l)
            {
              if (next_ys[b][i - 1][l] == Dictionary::eos_id)
                continue;

              if (m++ == from_beam_offset)
              {
                from_beam = l;
                break;
              }
            }

            prev_ks[b][i][k] = from_beam;
            if (subvocab.size())
              /* restore the actual index */
              next_ys[b][i][k] = subvocab[best_score_id];
            else
              next_ys[b][i][k] = best_score_id;

            scores[b][i][k] = best_score;

            // Store the attention.
            all_attention[b][i].row(k) = attn_softmax_out.row(batch_offset + from_beam_offset);

            // Override the best to ignore it for the next beam.
            cur_sent_out(from_beam_offset, best_score_id) = -std::numeric_limits<float>::max();
          }
        }

        if (_model->get_tgt_feat_dicts().size() > 0)
        {
          next_features[b].emplace_back();
          for (size_t j = 0; j < _model->get_tgt_feat_dicts().size(); ++j)
          {
            next_features[b][i].emplace_back(new_beam_size, 0);
            for (size_t k = 0, l = 0; k < next_ys[b][i - 1].size(); ++k)
            {
              if (next_ys[b][i - 1][k] == Dictionary::eos_id)
                continue;

              auto it = std::find(prev_ks[b][i].begin(), prev_ks[b][i].end(), k);
              if (it != prev_ks[b][i].end())
              {
                size_t best_feature_val = 0;
                gen_out[1 + j].row(batch_offset + l).maxCoeff(&best_feature_val);

                do {
                  next_features[b][i][j][std::distance(prev_ks[b][i].begin(), it)] = best_feature_val;
                  it = std::find(++it, prev_ks[b][i].end(), k);
                } while (it != prev_ks[b][i].end());
              }

              ++l;
            }
          }
        }

        // Update the current best.
        size_t num_finished = 0;
        for (size_t k = 0; k < new_beam_size; ++k)
        {
          if (next_ys[b][i][k] == Dictionary::eos_id)
          {
            ++num_finished;

            // Add to the n-best list if applicable, and keep the list sorted.
            for (size_t n = 0; n < _n_best; ++n)
            {
              if (scores[b][i][k] > max_score[b][n])
              {
                for (size_t m = _n_best - 1; m > n; --m)
                {
                  max_score[b][m] = max_score[b][m - 1];
                  best_k[b][m] = best_k[b][m - 1];
                  best_finished_at[b][m] = best_finished_at[b][m - 1];
                }

                max_score[b][n] = scores[b][i][k];
                best_k[b][n] = k;
                best_finished_at[b][n] = i;
                break;
              }
            }
          }
        }

        // Done when n-best hypotheses are finished, and no scores of unfinished hypotheses higher than the n-best.
        done[b] = (num_finished > 0);
        if (done[b] && num_finished < new_beam_size)
        {
          for (size_t n = 0; n < _n_best; ++n)
          {
            if (max_score[b][n] == -std::numeric_limits<float>::max())
            {
              done[b] = false;
              break;
            }
          }

          if (done[b])
          {
            for (size_t k = 0; k < new_beam_size; ++k)
            {
              if (next_ys[b][i][k] != Dictionary::eos_id && scores[b][i][k] > max_score[b][_n_best - 1])
              {
                done[b] = false;
                break;
              }
            }
          }
        }

        if (done[b])
          new_remaining_sents--;
        else
          total_beam_size += new_beam_size - num_finished;
      }

      _profiler.stop("Beam search");

      if (new_remaining_sents > 0)
      {
        // Prepare new rnn_state_dec: ignore finished sentences and reorder states according to
        // the beam origin.
        std::vector<MatFwd> new_rnn_state_dec;

        for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        {
          new_rnn_state_dec.emplace_back(total_beam_size, rnn_size);
          new_rnn_state_dec.back().setZero();
        }

        // Also remove finished sentences from the context.
        MatFwd new_context(total_beam_size, source_l*rnn_size);
        std::unique_ptr<MatFwd> new_input_feed(with_input_feeding ? new MatFwd(total_beam_size, rnn_size) : nullptr);

        for (size_t b = 0, prev_batch_offset = 0, new_batch_offset = 0; b < batch_size; ++b)
        {
          size_t j = 0;
          if (!done[b])
          {
            for (size_t k = 0; k < next_ys[b][i].size(); ++k)
            {
              if (next_ys[b][i][k] == Dictionary::eos_id)
                continue;

              // Copy the states from the beam we came from.
              size_t prev_k_offset = prev_ks[b][i][k] - std::count(next_ys[b][i - 1].begin(), std::next(next_ys[b][i - 1].begin(), prev_ks[b][i][k] + 1), Dictionary::eos_id);
              for (size_t l = 0; l < new_rnn_state_dec.size(); ++l)
              {
                new_rnn_state_dec[l].row(new_batch_offset + j) =
                  rnn_state_dec[l].row(prev_batch_offset + prev_k_offset);
              }

              if (with_input_feeding)
              {
                new_input_feed->row(new_batch_offset + j) =
                  input_feed->row(prev_batch_offset + prev_k_offset);
              }

              new_context.row(new_batch_offset + j) =
                context_dec.row(prev_batch_offset + prev_k_offset);

              ++j;
            }

            new_batch_offset += j;
          }

          prev_batch_offset += beam_size[b];
          beam_size[b] = j;
        }

        rnn_state_dec = new_rnn_state_dec;
        context_dec = new_context;
        context_dec.setHiddenDim(source_l); // Do not forget to view it as a 3D tensor.
        if (with_input_feeding)
          input_feed.reset(new_input_feed.release());
      }

      remaining_sents = new_remaining_sents;
    }

    // Build final translation by following the beam path for each batch.
    std::vector<std::vector<std::vector<std::string> > > batch_tgt_tokens;
    std::vector<std::vector<std::vector<std::vector<std::string> > > > batch_tgt_features;
    std::vector<std::vector<std::vector<std::vector<float> > > > batch_attention;
    std::vector<std::vector<size_t> > batch_tgt_count_unk_words;

    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_tgt_tokens.emplace_back();
      batch_tgt_features.emplace_back();
      batch_attention.emplace_back();
      batch_tgt_count_unk_words.emplace_back();
      for (size_t n = 0; n < _n_best; ++n)
      {
        size_t start_k = best_k[b][n];
        size_t len = best_finished_at[b][n] + 1;
        if (len == 1)
          len = i;

        std::vector<size_t> tgt_ids;
        std::vector<std::vector<size_t> > tgt_feat_ids;
        std::vector<std::vector<float> > attention;

        tgt_ids.resize(len - 2); // Ignore <s> and </s>.

        for (size_t f = 0; f < _model->get_tgt_feat_dicts().size(); ++f)
          tgt_feat_ids.emplace_back(len - 2, 0);
        for (size_t l = 0; l < len - 2; ++l)
          attention.emplace_back(batch_tokens[b].size(), 0);

        for (size_t k = start_k, j = len - 1; j > 1; --j)
        {
          for (size_t f = 0; f < _model->get_tgt_feat_dicts().size(); ++f)
            tgt_feat_ids[f][j - 2] = next_features[b][j][f][k];
          
          k = prev_ks[b][j][k];

          tgt_ids[j - 2] = next_ys[b][j - 1][k];
          for (size_t l = 0; l < batch_tokens[b].size(); ++l)
          {
            size_t pad_length = source_l - batch_tokens[b].size();
            attention[j - 2][l] = all_attention[b][j - 1](k, l + pad_length);
          }
        }

        batch_attention[b].push_back(attention);
        batch_tgt_count_unk_words[b].push_back(std::count(tgt_ids.begin(), tgt_ids.end(), Dictionary::unk_id));

        if (_replace_unk)
          batch_tgt_tokens[b].push_back(ids_to_words_replace(_model->get_tgt_dict(),
                                                             *_phrase_table,
                                                             tgt_ids,
                                                             batch_tokens[b],
                                                             attention));
        else
          batch_tgt_tokens[b].push_back(ids_to_words(_model->get_tgt_dict(), tgt_ids));

        batch_tgt_features[b].emplace_back();
        for (size_t f = 0; f < _model->get_tgt_feat_dicts().size(); ++f)
          batch_tgt_features[b][n].push_back(ids_to_words(_model->get_tgt_feat_dicts()[f], tgt_feat_ids[f]));
      }
    }

    return TranslationResult(batch_tgt_tokens, batch_tgt_features, batch_attention, max_score, batch_tgt_count_unk_words);
  }

}
