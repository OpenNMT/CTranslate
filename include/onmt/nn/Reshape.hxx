#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Reshape<MatFwd>::Reshape(th::Table* data)
      : Module<MatFwd>("nn.Reshape")
      , _dims(th::get_storage_as_vector<long>(data, "size"))
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> Reshape<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      // also do the SplitTable
      std::vector<MatFwd> out;

      long leading_dim = _dims[0];

      for (long i = 0; i < leading_dim; ++i)
      {
        out.emplace_back(input[0].block(0,
                                        i * (input[0].cols() / leading_dim),
                                        input[0].rows(),
                                        input[0].cols() / leading_dim));
      }

      return out;
    }

  }
}
