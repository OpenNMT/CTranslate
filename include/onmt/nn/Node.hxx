#pragma once

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    Node<MatFwd>::Node(size_t id)
      : _visited(false)
      , _id(id)
      , _select_index(-1)
      , _expected_inputs(0)
      , _children()
    {
    }

    template <typename MatFwd>
    void Node<MatFwd>::set_id(size_t id)
    {
      _id = id;
    }

    template <typename MatFwd>
    void Node<MatFwd>::set_select_index(int index)
    {
      _select_index = index;
    }

    template <typename MatFwd>
    void Node<MatFwd>::add_child(Node& child)
    {
      _children.push_back(&child);
    }

    template <typename MatFwd>
    void Node<MatFwd>::set_module(nn::Module<MatFwd>* module)
    {
      _module = module;
    }

    template <typename MatFwd>
    Module<MatFwd>* Node<MatFwd>::find(const std::string& custom_name)
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

    template <typename MatFwd>
    void Node<MatFwd>::add_input_index(size_t id)
    {
      size_t index = _index_to_input.size();
      _index_to_input.push_back(id);
      _input_to_index[id] = index;

      _input.resize(_input.size() + 1);
      _expected_inputs += 1;
    }

    template <typename MatFwd>
    void Node<MatFwd>::to_dot(std::ostream& os) const
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

    template <typename MatFwd>
    void Node<MatFwd>::forward(std::vector<MatFwd>& input,
                               std::vector<MatFwd>& final_output,
                               const Node* from)
    {
      if (_select_index >= 0) // Pick input matrix from table.
        _input[0] = input[_select_index];
      else if (_index_to_input.size() > 1) // Map matrix into input table.
        _input[_input_to_index.at(from->_id)] = input[0];
      else // "input" is the actual input of the module.
        _input = input;

      _expected_inputs--;

      // Only forward into the module when all inputs were forwarded into the node.
      if (_expected_inputs <= 0)
      {
        if (_module)
          _output = _module->forward(_input);
        else
          _output = _input;

        // Reset input table and count to make the node reentrant.
        _expected_inputs = _index_to_input.size();
        _input.resize(_expected_inputs);

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

  }
}
