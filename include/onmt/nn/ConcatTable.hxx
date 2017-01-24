#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ConcatTable<MatFwd, MatIn, MatEmb, ModelT>::ConcatTable(th::Table* data)
      : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ConcatTable", data)
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> ConcatTable<MatFwd, MatIn, MatEmb, ModelT>::forward(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      out.reserve(this->_sequence.size());

      for (auto& mod: this->_sequence)
      {
        std::vector<MatFwd> in(1, input[0]);
        out.push_back(mod->forward(in)[0]);
      }

      return Module<MatFwd>::wrap_return(out);
    }

  }
}
