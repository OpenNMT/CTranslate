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
      Node(size_t id)
        : _visited(false)
        , _id(id)
        , _select_index(-1)
        , _expected_inputs(0)
        , _children()
      {
      }

      void set_id(size_t id)
      {
        _id = id;
      }

      void set_select_index(int index)
      {
        _select_index = index;
      }

      void add_child(Node& child)
      {
        _children.push_back(&child);
      }

      void set_module(nn::Module<MatFwd>* module)
      {
        _module = module;
      }

      void add_input_index(size_t id)
      {
        size_t index = _index_to_input.size();
        _index_to_input.push_back(id);
        _input_to_index[id] = index;

        _module_inputs.resize(_module_inputs.size() + 1);
        _expected_inputs += 1;
      }

      Module<MatFwd>* find(const std::string& custom_name)
      {
        if (_module)
        {
          auto res = _module->find(custom_name);
          if (res)
            return res;
        }

        for (auto child: _children)
        {
          auto res = child->find(custom_name);
          if (res)
            return res;
        }

        return nullptr;
      }

      void* apply(void* (*func)(Module<MatFwd>*, void*), void* data)
      {
        if (_module)
          _module->apply(func, data);

        for (auto child: _children)
          child->apply(func, data);

        return nullptr;
      }

      void to_dot(std::ostream& os) const
      {
        if (_visited)
          return;

        _visited = true;

        os << _id << "  [label=\"Node" << _id;
        if (_module)
        {
          os << "\nmodule = " << _module->get_name();
          std::string details = _module->get_details();
          if (!details.empty())
            os << " " << details;
        }
        if (_select_index >= 0)
        {
          os << "\ninput = { }";
          os << "\nselectindex = " << _select_index;
        }
        if (_index_to_input.size() > 1)
        {
          os << "\nmapindex = {";
          for (auto it = _index_to_input.begin(); it != _index_to_input.end(); it++)
          {
            if (it != _index_to_input.begin())
              os << ",";
            os << "Node" << *it;
          }
          os << "}";
        }
        os << "\"];" << std::endl;

        for (auto child: _children)
        {
          os <<  _id << " -> " << child->_id << ";" << std::endl;
          child->to_dot(os);
        }
      }

      void forward(const std::vector<MatFwd>& node_inputs,
                   std::vector<MatFwd>& final_output,
                   const Node* from)
      {
        if (_select_index >= 0) // Pick input matrix from table.
          _module_inputs[0] = node_inputs[_select_index];
        else if (_index_to_input.size() > 1) // Map matrix into input table.
          _module_inputs[_input_to_index.at(from->_id)] = node_inputs[0];
        else // node_inputs is also input of the module.
          _module_inputs = node_inputs;

        _expected_inputs--;

        // Only forward into the module when all inputs were forwarded into the node.
        if (_expected_inputs <= 0)
        {
          if (_module && _module->get_name() != "nn.Identity")
            _output = _module->forward(_module_inputs);
          else
            _output = _module_inputs;

          // Reset input table and count to make the node reentrant.
          _expected_inputs = _index_to_input.size();
          _module_inputs.resize(_expected_inputs);

          if (_children.empty()) // No child == graph output.
            final_output = _output;
          else
          {
            // Forward output into every chilren node
            for (auto child: _children)
              child->forward(_output, final_output, this);
          }
        }
      }

    private:
      mutable bool _visited;
      size_t _id;
      int _select_index;
      int _expected_inputs;
      nn::Module<MatFwd>* _module;
      std::vector<Node*> _children;
      std::map<size_t, size_t> _input_to_index;
      std::vector<size_t> _index_to_input;
      std::vector<MatFwd> _module_inputs;
      std::vector<MatFwd> _output;
    };

  }
}
