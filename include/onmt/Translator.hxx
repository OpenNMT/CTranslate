#pragma once

#include <algorithm>
#include <iostream>
#include <limits>

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
      while (input.size() > max_len)
        input.pop_back();
    }
    else if (input.size() < max_len)
    {
      input.reserve(max_len);
      size_t pad_size = max_len - input.size();
      input.insert(input.begin(), pad_size, pad_value);
    }
  }

  static size_t pad_inputs(std::vector<std::vector<size_t> >& inputs, size_t pad_value, size_t max_len)
  {
    size_t cur_max_len = 0;

    for (const auto& input: inputs)
      cur_max_len = std::max(cur_max_len, input.size());

    cur_max_len = std::min(cur_max_len, max_len);

    for (auto& input: inputs)
      pad_input(input, pad_value, cur_max_len);

    return cur_max_len;
  }

  // The generator outputs a beam_size*batch_size x vocab_size matrix.
  // This function allows to easily query the output given a batch and a beam.
  static size_t get_offset(size_t batch, size_t beam, size_t batch_size)
  {
    return beam*batch_size + batch;
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
                                                        bool cuda,
                                                        bool profiling)
    : _profiler(profiling)
    , _model(model, _profiler, cuda)
    , _src_dict(_model.get_src_dict())
    , _tgt_dict(_model.get_tgt_dict())
    , _src_feat_dicts(_model.get_src_feat_dicts())
    , _tgt_feat_dicts(_model.get_tgt_feat_dicts())
    , _phrase_table(phrase_table)
    , _subdict(vocab_mapping, _tgt_dict)
    , _replace_unk(replace_unk)
    , _max_sent_length(max_sent_length)
    , _beam_size(beam_size)
    , _encoder(_model.get_encoder_module(0))
    , _encoder_bwd(_model.get_encoder_module(1))
    , _decoder(_model.get_decoder_module(0))
    , _generator(_model.get_decoder_module(1))
  {
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  std::string Translator<MatFwd, MatIn, MatEmb, ModelT>::translate(const std::string& text,
                                                                   ITokenizer& tokenizer)
  {
    std::vector<std::string> src_tokens;
    std::vector<std::vector<std::string> > src_features;

    tokenizer.tokenize(text, src_tokens, src_features);

    TranslationResult res = translate(src_tokens, src_features);

    return tokenizer.detokenize(res.get_words(), res.get_features());
  }

  class tdict
  {
  public:
    tdict(int n)
      : _ndict(n)
    {}
    int _ndict;
    std::vector<size_t> subvocab;
  };

  template <typename MatFwd, typename MatIn, typename, typename ModelT>
  void* reduce_vocabulary(nn::Module<MatFwd>* M, void* t)
  {
    if (M->get_name() == "nn.Linear")
    {
      nn::Linear<MatFwd, MatIn, ModelT>* mL = (nn::Linear<MatFwd, MatIn, ModelT>*)M;
      tdict* data = (tdict*)t;
      if (mL->get_weight().rows() == data->_ndict)
        SubDict::reduce_linearweight(mL->get_weight(),
                                     mL->get_bias(),
                                     mL->get_rweight(),
                                     mL->get_rbias(),
                                     data->subvocab);
    }
    return 0;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  TranslationResult
  Translator<MatFwd, MatIn, MatEmb, ModelT>::translate(const std::vector<std::string>& tokens,
                                                       const std::vector<std::vector<std::string> >& features)
  {
    std::vector<std::vector<std::string> > src_batch(1, tokens);
    std::vector<std::vector<std::vector<std::string> > > src_feat_batch(1, features);

    return translate_batch(src_batch, src_feat_batch);
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  std::vector<std::string>
  Translator<MatFwd, MatIn, MatEmb, ModelT>::translate_batch(const std::vector<std::string>& texts,
                                                             ITokenizer& tokenizer)
  {
    std::vector<std::vector<std::string> > batch_tokens;
    std::vector<std::vector<std::vector<std::string> > > batch_features;

    for (const auto& text: texts)
    {
      std::vector<std::string> tokens;
      std::vector<std::vector<std::string> > features;
      tokenizer.tokenize(text, tokens, features);
      batch_tokens.push_back(tokens);
      batch_features.push_back(features);
    }

    TranslationResult res = translate_batch(batch_tokens, batch_features);

    std::vector<std::string> tgt_texts;
    tgt_texts.reserve(texts.size());

    for (size_t i = 0; i < res.count(); ++i)
    {
      if (res.has_features())
        tgt_texts.push_back(tokenizer.detokenize(res.get_words(i), res.get_features(i)));
      else
        tgt_texts.push_back(tokenizer.detokenize(res.get_words(i), {}));
    }

    return tgt_texts;
  }

  template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
  TranslationResult
  Translator<MatFwd, MatIn, MatEmb, ModelT>::translate_batch(
    const std::vector<std::vector<std::string> >& batch_tokens,
    const std::vector<std::vector<std::vector<std::string> > >& batch_features)
  {
    size_t batch_size = batch_tokens.size();

    // Convert words to ids.
    std::vector<std::vector<size_t> > batch_ids;
    std::vector<std::vector<std::vector<size_t> > > batch_feat_ids;

    tdict data(_tgt_dict.get_size());
    if (!_subdict.empty())
    {
      std::set<size_t> e;
      for (const auto& it: batch_tokens)
        _subdict.extract(it, e);
      /* convert into vector */
      for (auto idx: e)
        data.subvocab.push_back(idx);
      /* modify generator weights and bias accordingly */
      _generator->apply(reduce_vocabulary<MatFwd, MatIn, MatEmb, ModelT>, &data);
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_ids.push_back(words_to_ids(_src_dict, batch_tokens[b]));

      if (_src_feat_dicts.size() != batch_features[b].size())
        throw std::runtime_error("expected "
                                 + std::to_string(_src_feat_dicts.size())
                                 + " word feature(s), got "
                                 + std::to_string(batch_features[b].size())
                                 + " instead");
      else if (_src_feat_dicts.size() > 0)
      {
        batch_feat_ids.emplace_back();
        for (size_t j = 0; j < _src_feat_dicts.size(); ++j)
          batch_feat_ids[b].push_back(words_to_ids(_src_feat_dicts[j], batch_features[b][j]));
      }
    }

    // Pad inputs to the left.
    size_t source_l = pad_inputs(batch_ids, Dictionary::pad_id, _max_sent_length);

    for (size_t j = 0; j < _src_feat_dicts.size(); ++j)
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
    input.emplace_back(batch_size, 1 + _src_feat_dicts.size());
    for (size_t b = 0; b < batch_size; ++b)
    {
      input.back()(b, 0) = batch_ids[b][t];

      for (size_t j = 0; j < _src_feat_dicts.size(); ++j)
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

    size_t num_layers = _model.template get_option_value<size_t>("layers");
    size_t rnn_size = _model.template get_option_value<size_t>("rnn_size");
    bool brnn = _model.get_option_string("encoder_type") == "brnn" || _model.get_option_flag("brnn");
    const std::string& brnn_merge = _model.get_option_string("brnn_merge");

    if (brnn && brnn_merge == "concat")
      rnn_size /= 2;

    rnn_state_enc = init_rnn_states<MatFwd>(num_layers * 2, batch_size, rnn_size);
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
    std::vector<MatFwd>& rnn_state_enc,
    MatFwd& context,
    const std::vector<size_t>& subvocab)
  {
    size_t batch_size = batch_tokens.size();

    size_t rnn_size = _model.template get_option_value<size_t>("rnn_size");
    bool with_input_feeding = _model.get_option_flag("input_feed", true);

    MatFwd context_dec = context.replicate(_beam_size, 1);
    context_dec.setHiddenDim(source_l);

    MatFwd input_feed(_beam_size * batch_size, rnn_size);
    input_feed.setZero();

    // Copy encoder states to decoder states.
    std::vector<MatFwd> rnn_state_dec;
    rnn_state_dec.reserve(rnn_state_enc.size());
    for (size_t l = 0; l < rnn_state_enc.size(); ++l)
    {
      rnn_state_dec.emplace_back(_beam_size * batch_size, rnn_size);
      rnn_state_dec.back() = rnn_state_enc[l].replicate(_beam_size, 1);
    }

    std::vector<std::vector<std::vector<size_t> > > next_ys; // b x n x K
    std::vector<std::vector<std::vector<size_t> > > prev_ks; // b x n x K
    std::vector<std::vector<std::vector<float> > > scores; // b x n x K
    std::vector<std::vector<std::vector<std::vector<size_t> > > > next_features; // b x n x j x K
    std::vector<std::vector<MatFwd> > all_attention; // b x n x [ K x source_l ]
    std::vector<int> batch_idx;

    // Get a pointer to the attention softmax module to mask its output.
    nn::Module<MatFwd>* softmax_attn = _decoder->find("softmaxAttn");

    // Prepare data structures for the beam search.
    for (size_t b = 0; b < batch_size; ++b)
    {
      std::vector<size_t> in1(_beam_size, Dictionary::pad_id);
      in1[0] = Dictionary::bos_id;
      next_ys.emplace_back(1, in1);

      std::vector<size_t> prev_ks1(_beam_size, 1);
      prev_ks.emplace_back(1, prev_ks1);

      std::vector<float> scores1(_beam_size, 0.0);
      scores.emplace_back(1, scores1);

      if (_tgt_feat_dicts.size() > 0)
      {
        next_features.emplace_back();
        next_features.back().emplace_back();
        for (size_t j = 0; j < _tgt_feat_dicts.size(); ++j)
        {
          std::vector<size_t> in_feat1(_beam_size, Dictionary::pad_id);
          in_feat1[0] = Dictionary::eos_id;
          next_features.back().back().push_back(in_feat1);
        }
      }

      std::vector<MatFwd> attn1;
      attn1.emplace_back(_beam_size, source_l);
      attn1.back().setZero();
      all_attention.push_back(attn1);

      batch_idx.push_back(b);
    }

    size_t remaining_sents = batch_size;
    std::vector<bool> done(batch_size, false);
    std::vector<bool> found_eos(batch_size, false);

    std::vector<float> end_score(batch_size, -std::numeric_limits<float>::max());
    std::vector<size_t> end_finished_at(batch_size, 0);

    std::vector<float> max_score(batch_size, -std::numeric_limits<float>::max());
    std::vector<size_t> best_k(batch_size, 0);
    std::vector<size_t> best_finished_at(batch_size, 0);

    size_t i;

    for (i = 1; remaining_sents > 0 && i < _max_sent_length; ++i)
    {
      // Prepare decoder input at timestep i.
      std::vector<MatFwd> input;

      // States.
      for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        input.push_back(rnn_state_dec[l]);

      // Words and features.
      input.emplace_back(_beam_size * remaining_sents, 1 + _tgt_feat_dicts.size());
      for (size_t b = 0; b < batch_size; ++b)
      {
        if (done[b])
          continue;

        int idx = batch_idx[b];

        for (size_t k = 0; k < _beam_size; ++k)
        {
          input.back()(get_offset(idx, k, remaining_sents), 0) = next_ys[b][i-1][k];

          for (size_t j = 0; j < _tgt_feat_dicts.size(); ++j)
            input.back()(get_offset(idx, k, remaining_sents), j + 1) = next_features[b][i-1][j][k];
        }

      }

      // Context and input feed.
      input.push_back(context_dec);
      if (with_input_feeding)
        input.push_back(input_feed);

      MatFwd attn_softmax_out;

      // Mask attention softmax output depending on input sentence size.
      softmax_attn->post_process_fun() = [&] (std::vector<MatFwd>& out) {
        if (batch_size == 1)
        {
          attn_softmax_out = out[0];
          return;
        }

        MatFwd& soft_out = out[0];

        for (size_t b = 0; b < batch_size; ++b)
        {
          if (done[b])
            continue;

          size_t pad_len = source_l - batch_tokens[b].size();

          if (pad_len > 0)
          {
            size_t idx = batch_idx[b];
            for (size_t k = 0; k < _beam_size; ++k)
            {
              size_t index = get_offset(idx, k, remaining_sents);
              soft_out.row(index).head(pad_len).setZero();
              soft_out.row(index) /= soft_out.row(index).sum(); // Normalization (softmax output).
            }
          }
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
        input_feed = out_decoder[out_decoder_pred_idx];
      for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        rnn_state_dec[l] = out_decoder[l];

      MatFwd& out = gen_out[0];
      out.setHiddenDim(remaining_sents); // beam x remaining_sents*vocab_size

      size_t new_remaining_sents = remaining_sents;

      _profiler.start();

      // Update beam path for all non finished sentences.
      for (size_t b = 0; b < batch_size; ++b)
      {
        if (done[b])
          continue;

        int idx = batch_idx[b];
        // Penalize beam scores based on the previous step.
        for (size_t k = 0; k < _beam_size; ++k)
          out.row(get_offset(idx, k, remaining_sents)).array() += scores[b][i-1][k];

        prev_ks[b].emplace_back(_beam_size, 0);
        next_ys[b].emplace_back(_beam_size, 0);
        scores[b].emplace_back(_beam_size, 0.0);
        all_attention[b].emplace_back(_beam_size, source_l);

        for (size_t k = 0; k < _beam_size; ++k)
        {
          float best_score = -std::numeric_limits<float>::max();
          size_t best_score_id = 0;
          size_t from_beam = 0;

          if (i == 1 || _beam_size == 1) // All outputs are the same on the first decoding step.
          {
            best_score_id = 0;
            best_score = out.row(idx).maxCoeff(&best_score_id);
          }
          else
          {
            std::vector<float> best_score_per_beam_size;
            std::vector<size_t> best_score_id_per_beam_size;

            // Find the best score across all beams.
            for (size_t k = 0; k < _beam_size; ++k)
            {
              size_t best_score_id_k = 0;
              float best_score_k = out
                .row(get_offset(idx, k, remaining_sents))
                .maxCoeff(&best_score_id_k);
              best_score_per_beam_size.push_back(best_score_k);
              best_score_id_per_beam_size.push_back(best_score_id_k);
            }

            // Pick the best.
            for (size_t k = 0; k < _beam_size; ++k)
            {
              if (best_score_per_beam_size[k] > best_score)
              {
                best_score = best_score_per_beam_size[k];
                best_score_id = best_score_id_per_beam_size[k];
                from_beam = k;
              }
            }
          }

          prev_ks[b][i][k] = from_beam;
          if (subvocab.size())
            /* restore the actual index */
            next_ys[b][i][k] = subvocab[best_score_id];
          else
            next_ys[b][i][k] = best_score_id;

          scores[b][i][k] = best_score;

          size_t from_beam_offset = get_offset(idx, from_beam, remaining_sents);

          // Store the attention.
          all_attention[b][i].row(k) = attn_softmax_out.row(from_beam_offset);

          // Override the best to ignore it for the next beam.
          out.row(from_beam_offset)(best_score_id) = -std::numeric_limits<float>::max();
        }

        if (_tgt_feat_dicts.size() > 0)
        {
          next_features[b].emplace_back();
          for (size_t j = 0; j < _tgt_feat_dicts.size(); ++j)
          {
            next_features[b][i].emplace_back();
            for (size_t k = 0; k < _beam_size; ++k)
            {
              size_t best_feature_val = 0;
              gen_out[1+j].row(get_offset(idx, k, remaining_sents)).maxCoeff(&best_feature_val);
              next_features[b][i][j].push_back(best_feature_val);
            }
          }
        }

        end_score[b] = scores[b][i][0];

        if (next_ys[b][i][0] == Dictionary::eos_id) // End of translation.
        {
          done[b] = true;
          found_eos[b] = true;
          end_finished_at[b] = i;
          new_remaining_sents--;
        }
        else // Otherwise, update the current best beam.
        {
          for (size_t k = 0; k < _beam_size; ++k)
          {
            if (next_ys[b][i][k] == Dictionary::eos_id)
            {
              found_eos[b] = true;
              if (scores[b][i][k] > max_score[b])
              {
                max_score[b] = scores[b][i][k];
                best_k[b] = k;
                best_finished_at[b] = i;
              }
            }
          }
        }
      }

      _profiler.stop("Beam search");

      if (new_remaining_sents > 0)
      {
        // Prepare new rnn_state_dec: ignore finished sentences and reorder states according to
        // the beam origin.
        std::vector<MatFwd> new_rnn_state_dec;

        for (size_t l = 0; l < rnn_state_dec.size(); ++l)
        {
          new_rnn_state_dec.emplace_back(_beam_size*new_remaining_sents, rnn_size);
          new_rnn_state_dec.back().setZero();
        }

        // Also remove finished sentences from the context.
        MatFwd new_context(_beam_size*new_remaining_sents, source_l*rnn_size);
        MatFwd new_input_feed(_beam_size*new_remaining_sents, rnn_size);

        size_t new_idx = 0; // Update each batch index.

        for (size_t b = 0; b < batch_size; ++b)
        {
          if (!done[b])
          {
            for (size_t k = 0; k < _beam_size; ++k)
            {
              for (size_t l = 0; l < new_rnn_state_dec.size(); ++l)
              {
                // Copy the states from the beam we came from.
                new_rnn_state_dec[l].row(get_offset(new_idx, k, new_remaining_sents)) =
                  rnn_state_dec[l].row(get_offset(batch_idx[b], prev_ks[b][i][k], remaining_sents));
              }

              if (with_input_feeding)
              {
                new_input_feed.row(get_offset(new_idx, k, new_remaining_sents)) =
                  input_feed.row(get_offset(batch_idx[b], k, remaining_sents));
              }

              new_context.row(get_offset(new_idx, k, new_remaining_sents)) =
                context_dec.row(get_offset(batch_idx[b], k, remaining_sents));
            }

            batch_idx[b] = new_idx++;
          }
          else
            batch_idx[b] = -1;
        }

        rnn_state_dec = new_rnn_state_dec;
        context_dec = new_context;
        context_dec.setHiddenDim(source_l); // Do not forget to view it as a 3D tensor.
        if (with_input_feeding)
          input_feed = new_input_feed;
      }

      remaining_sents = new_remaining_sents;
    }

    // Build final translation by following the beam path for each batch.
    std::vector<std::vector<std::string> > batch_tgt_tokens;
    std::vector<std::vector<std::vector<std::string> > > batch_tgt_features;
    std::vector<std::vector<std::vector<float> > > batch_attention;

    for (size_t b = 0; b < batch_size; ++b)
    {
      size_t start_k = best_k[b];
      size_t len = best_finished_at[b] + 1;
      if (end_score[b] > max_score[b]) // End score is the score of the top beam.
      {
        start_k = 0;
        len = end_finished_at[b] + 1;
      }

      if (len == 1)
        len = i;

      std::vector<size_t> tgt_ids;
      std::vector<std::vector<size_t> > tgt_feat_ids;
      std::vector<std::vector<float> > attention;

      tgt_ids.resize(len - 2); // Ignore <s> and </s>.

      for (size_t f = 0; f < _tgt_feat_dicts.size(); ++f)
        tgt_feat_ids.emplace_back(len - 2, 0);
      for (size_t l = 0; l < len - 2; ++l)
        attention.emplace_back(batch_tokens[b].size(), 0);

      for (size_t k = start_k, j = len - 1; j > 1; --j)
      {
        k = prev_ks[b][j][k];

        tgt_ids[j - 2] = next_ys[b][j - 1][k];
        for (size_t l = 0; l < batch_tokens[b].size(); ++l)
        {
          size_t pad_length = source_l - batch_tokens[b].size();
          attention[j - 2][l] = all_attention[b][j - 1](k, l + pad_length);
        }
        for (size_t f = 0; f < _tgt_feat_dicts.size(); ++f)
          tgt_feat_ids[f][j - 2] = next_features[b][j][f][k];
      }

      batch_attention.push_back(attention);

      if (_replace_unk)
        batch_tgt_tokens.push_back(ids_to_words_replace(_tgt_dict,
                                                         _phrase_table,
                                                         tgt_ids,
                                                         batch_tokens[b],
                                                         attention));
      else
        batch_tgt_tokens.push_back(ids_to_words(_tgt_dict, tgt_ids));

      batch_tgt_features.emplace_back();
      for (size_t f = 0; f < _tgt_feat_dicts.size(); ++f)
        batch_tgt_features[b].push_back(ids_to_words(_tgt_feat_dicts[f], tgt_feat_ids[f]));
    }

    return TranslationResult(batch_tgt_tokens, batch_tgt_features, batch_attention);
  }

}
