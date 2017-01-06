#pragma once

#include "onmt/nn/Node.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Graph: public Module<MatFwd>
    {
    public:
      Graph(th::Class* module, const std::string& name);

      virtual std::vector<MatFwd> forward(std::vector<MatFwd>& input) const override;

      Module<MatFwd>* find(const std::string& custom_name) override;

      // Dump the graph in the DOT format.
      void to_dot(const std::string& file, const std::string& name);

    private:
      std::map<size_t, Node<MatFwd> > _node_map;
      Node<MatFwd>& _root;

      Node<MatFwd>& build_graph(th::Obj* root);
    };

  }
}

#include "Graph.hxx"
