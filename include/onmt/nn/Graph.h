#pragma once

#include "onmt/nn/Node.h"
#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Obj.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Graph: public Module<MatFwd>
    {
    public:
      Graph(th::Class* module,
            const std::string& name,
            ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory);

      virtual std::vector<MatFwd> forward_impl(std::vector<MatFwd>& input) const override;

      Module<MatFwd>* find(const std::string& custom_name) override;

      // Dump the graph in the DOT format.
      void to_dot(const std::string& file, const std::string& name);

    private:
      std::map<size_t, Node<MatFwd> > _node_map;
      Node<MatFwd>& _root;

      Node<MatFwd>& build_graph(th::Obj* root, ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory);
    };

  }
}

#include "Graph.hxx"
