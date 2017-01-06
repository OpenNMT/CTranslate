#pragma once

#include <vector>
#include <fstream>
#include <map>

#include "onmt/nn/Module.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class Node
    {
    public:
      Node(size_t id);

      void set_id(size_t id);
      void set_select_index(int index);
      void add_child(Node& child);
      void set_module(nn::Module<MatFwd>* module);
      void add_input_index(size_t id);

      Module<MatFwd>* find(const std::string& custom_name);

      void to_dot(std::ostream& os) const;

      void forward(std::vector<MatFwd>& input, std::vector<MatFwd>& final_output, const Node* from);

    private:
      mutable bool _visited;
      size_t _id;
      int _select_index;
      int _expected_inputs;
      nn::Module<MatFwd>* _module;
      std::vector<Node*> _children;
      std::map<size_t, size_t> _input_to_index;
      std::vector<size_t> _index_to_input;
      std::vector<MatFwd> _input;
      std::vector<MatFwd> _output;
    };

  }
}

#include "onmt/nn/Node.hxx"
