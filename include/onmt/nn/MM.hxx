#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    MM<MatFwd>::MM(th::Table* data)
      : Module<MatFwd>("nn.MM")
      , _transA(get_boolean(data, "transA"))
      , _transB(get_boolean(data, "transB"))
    {
    }

    template <typename MatFwd>
    std::vector<MatFwd> MM<MatFwd>::forward_impl(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> out;

      out.emplace_back(input[0].rows(), input[0].virtualRows()*input[1].virtualCols());
      out.back().setHiddenDim(input[0].virtualRows());

      for (size_t i = 0; i < input[0].batches(); ++i)
      {
        MatFwd m1 = input[0].batch(i);
        MatFwd m2 = input[1].batch(i);

        if (_transA)
          m1.transposeInPlace();
        if (_transB)
          m2.transposeInPlace();

        MatFwd res = m1 * m2;

        out.back().assign(i, res);
      }

      return out;
    }

  }
}
