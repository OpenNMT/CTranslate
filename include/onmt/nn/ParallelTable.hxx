#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ParallelTable<MatFwd, MatIn, MatEmb, ModelT>::ParallelTable(th::Table* data,
                                                                ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
      : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ParallelTable", data, factory)
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> ParallelTable<MatFwd, MatIn, MatEmb, ModelT>::forward_impl(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      size_t idx = 0;

      for (auto& mod: this->_sequence)
      {
        std::vector<MatFwd> in;

        if (input.size() == 1 && this->_sequence.size() > 1)
        {
          // This is a special case when the input table is actually bundled in a single matrix.
          // The dimensions that do not have a corresponding module are all forwarded to the
          // last module in the sequence.
          if (idx == this->_sequence.size() - 1)
          {
            for (int i = idx; i < input[0].cols(); ++i)
              in.push_back(input[0].col(i));

            auto res = mod->forward(in);
            out.insert(out.end(), res.begin(), res.end());
          }
          else
          {
            in.push_back(input[0].col(idx++));
            out.push_back(mod->forward(in)[0]);
          }
        }
        else
        {
          in.push_back(input[idx++]);
          out.push_back(mod->forward(in)[0]);
        }
      }

      return out;
    }

  }
}
