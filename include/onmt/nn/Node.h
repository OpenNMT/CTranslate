#pragma once

#include <vector>
#include <fstream>
#include <map>
#include <unordered_map>
#include <limits>

#include "onmt/nn/ModuleFactory.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    class Node
    {
    public:
      Node(size_t id, const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory,
        std::unordered_map<size_t, Node<MatFwd, MatIn, MatEmb, ModelT>>& node_map)
        : _visited(false)
        , _id(id)
        , _select_index(-1)
        , _expected_inputs(0)
        , _children()
        , _factory(factory)
        , _node_map(node_map)
      {
      }

      Node(const Node& other, const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory,
        std::unordered_map<size_t, Node<MatFwd, MatIn, MatEmb, ModelT>>& node_map)
        : _visited(false)
        , _id(other._id)
        , _select_index(other._select_index)
        , _expected_inputs(other._index_to_input.size())
        , _module_id(other._module_id)
        , _children(other._children)
        , _input_to_index(other._input_to_index)
        , _index_to_input(other._index_to_input)
        , _module_inputs(other._index_to_input.size())
        , _factory(factory)
        , _node_map(node_map)
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

      void add_child(size_t child)
      {
        _children.push_back(child);
      }

      void set_module_id(size_t module_id)
      {
        _module_id = module_id;
      }

      void add_input_index(size_t id)
      {
        size_t index = _index_to_input.size();
        _index_to_input.push_back(id);
        _input_to_index[id] = index;

        _module_inputs.resize(_module_inputs.size() + 1);
        _expected_inputs += 1;
      }

      Module<MatFwd, MatIn, MatEmb, ModelT>* find(const std::string& custom_name)
      {
        if (_module_id != std::numeric_limits<size_t>::max())
        {
          auto mod = _factory.get_module(_module_id);
          auto res = mod->find(custom_name);
          if (res)
            return res;
        }

        for (auto child: _children)
        {
          auto& node = _node_map.at(child);
          auto res = node.find(custom_name);
          if (res)
            return res;
        }

        return nullptr;
      }

      void* apply(void* (*func)(Module<MatFwd, MatIn, MatEmb, ModelT>*, void*), void* data)
      {
        if (_visited)
          return nullptr;

        _visited = true;

        if (_module_id != std::numeric_limits<size_t>::max())
        {
          auto mod = _factory.get_module(_module_id);
          mod->apply(func, data);
        }

        for (auto child: _children)
        {
          auto& node = _node_map.at(child);
          node.apply(func, data);
        }

        return nullptr;
      }

      void to_dot(std::ostream& os) const
      {
        if (_visited)
          return;

        _visited = true;

        os << _id << "  [label=\"Node" << _id;
        if (_module_id != std::numeric_limits<size_t>::max())
        {
          auto mod = _factory.get_module(_module_id);
          os << "\nmodule = " << mod->get_name();
          std::string details = mod->get_details();
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
          auto& node = _node_map.at(child);
          os <<  _id << " -> " << node._id << ";" << std::endl;
          node.to_dot(os);
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
          Module<MatFwd, MatIn, MatEmb, ModelT>* mod = nullptr;
          if (_module_id != std::numeric_limits<size_t>::max())
            mod = _factory.get_module(_module_id);
          if (mod && mod->get_name() != "nn.Identity")
            _output = mod->forward(_module_inputs);
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
            {
              auto& node = _node_map.at(child);
              node.forward(_output, final_output, this);
            }
          }
        }
      }

      void set_unvisited()
      {
        _visited = false;
      }

    private:
      mutable bool _visited;
      size_t _id;
      int _select_index;
      int _expected_inputs;
      size_t _module_id;
      std::vector<size_t> _children;
      std::map<size_t, size_t> _input_to_index;
      std::vector<size_t> _index_to_input;
      std::vector<MatFwd> _module_inputs;
      std::vector<MatFwd> _output;
      const ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& _factory;
      std::unordered_map<size_t, Node<MatFwd, MatIn, MatEmb, ModelT>>& _node_map;
    };

  }
}
