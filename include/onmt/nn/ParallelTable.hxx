#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ParallelTable<MatFwd, MatIn, MatEmb, ModelT>::ParallelTable(th::Table* data)
      : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ParallelTable", data)
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> ParallelTable<MatFwd, MatIn, MatEmb, ModelT>::forward(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      out.reserve(input.size());
      size_t idx = 0;

      for (auto& mod: this->_sequence)
        out.push_back(mod->forward(input[idx++]));

      return Module<MatFwd>::wrap_return(out);
    }

  }
}
