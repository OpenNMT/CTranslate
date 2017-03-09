#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    ConcatTable<MatFwd, MatIn, MatEmb, ModelT>::ConcatTable(th::Table* data,
                                                            ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
      : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ConcatTable", data, factory)
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> ConcatTable<MatFwd, MatIn, MatEmb, ModelT>::forward_impl(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;
      out.reserve(this->_sequence.size());

      for (auto& mod: this->_sequence)
      {
        std::vector<MatFwd> in(1, input[0]);
        out.push_back(mod->forward(in)[0]);
      }

      return out;
    }

  }
}
