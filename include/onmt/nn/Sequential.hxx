#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Sequential<MatFwd, MatIn, MatEmb, ModelT>::Sequential(th::Table* data,
                                                          ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
      : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.Sequential", data, factory)
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> Sequential<MatFwd, MatIn, MatEmb, ModelT>::forward_impl(std::vector<MatFwd>& input) const
    {
      if (this->_sequence.empty())
        return input;

      auto it = this->_sequence.begin();
      std::vector<MatFwd> out = (*it)->forward(input);

      for (it++; it != this->_sequence.end(); it++)
      {
        out = (*it)->forward(out);
      }

      return out;
    }

  }
}
