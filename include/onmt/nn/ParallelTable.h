#pragma once

#include "onmt/nn/Container.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class ParallelTable: public Container<MatFwd, MatIn, MatEmb, ModelT>
    {
    public:
      ParallelTable(th::Table* data, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
        : Container<MatFwd, MatIn, MatEmb, ModelT>("nn.ParallelTable", data, factory)
      {
      }

      void forward_impl(const std::vector<MatFwd>& inputs) override
      {
        this->_outputs.resize(this->_sequence.size());

        for (size_t i = 0; i < this->_sequence.size(); ++i)
        {
          if (inputs.size() == 1 && this->_sequence.size() > 1)
          {
            // This is a special case when the inputs table is actually bundled in a single matrix.
            // The dimensions that do not have a corresponding module are all forwarded to the
            // last module in the sequence.
            if (i == this->_sequence.size() - 1)
            {
              std::vector<MatFwd> in;
              for (int j = i; j < inputs[0].cols(); ++j)
                in.push_back(inputs[0].col(j));

              auto res = this->_sequence[i]->forward(in);

              for (size_t j = i; j - i < res.size(); ++j)
                this->_outputs[j] = res[j - i];
            }
            else
            {
              this->_outputs[i] = this->_sequence[i]->forward_one(inputs[0].col(i));
            }
          }
          else
          {
            this->_outputs[i] = this->_sequence[i]->forward_one(inputs[i]);
          }
        }
      }
    };

  }
}
