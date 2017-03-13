#pragma once

#include "onmt/nn/ModuleFactory.h"
#include "onmt/th/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Graph<MatFwd, MatIn, MatEmb, ModelT>::Graph(th::Class* module,
                                                const std::string& name,
                                                ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
      : Module<MatFwd>(name, false)
      , _root(build_graph(dynamic_cast<th::Table*>(
                            dynamic_cast<th::Table*>(module->get_data())
                            ->get_object().at("forwardnodes"))
                          ->get_array()[0],
                          factory))
    {
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    void Graph<MatFwd, MatIn, MatEmb, ModelT>::to_dot(const std::string& file, const std::string& name)
    {
      std::ofstream out(file.c_str());
      out << "digraph " << name << " {" << std::endl;
      _root.to_dot(out);
      out << "}" << std::endl;
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Module<MatFwd>* Graph<MatFwd, MatIn, MatEmb, ModelT>::find(const std::string& custom_name)
    {
      if (this->_custom_name == custom_name)
        return this;

      return _root.find(custom_name);
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    std::vector<MatFwd> Graph<MatFwd, MatIn, MatEmb, ModelT>::forward_impl(std::vector<MatFwd>& input) const
    {
      std::vector<MatFwd> output;
      _root.forward(input, output, nullptr);
      return output;
    }

    template <typename MatFwd, typename MatIn, typename MatEmb, typename ModelT>
    Node<MatFwd>&
    Graph<MatFwd, MatIn, MatEmb, ModelT>::build_graph(th::Obj* root,
                                                      ModuleFactory<MatFwd, MatIn, MatEmb, ModelT>& factory)
    {
      th::Class* root_class = dynamic_cast<th::Class*>(root);
      th::Table* root_fields = dynamic_cast<th::Table*>(root_class->get_data());
      th::Number* root_id = dynamic_cast<th::Number*>(root_fields->get_object().at("id"));

      size_t id = static_cast<size_t>(root_id->get_value());

      auto placeholder = _node_map.emplace(id, id);
      Node<MatFwd>& root_node = placeholder.first->second;

      if (!placeholder.second) // already exists
        return root_node;

      th::Table* root_data = th::get_field<th::Table*>(root_fields, "data");
      th::Class* module_class = th::get_field<th::Class*>(root_data, "module");
      if (!module_class)
        root_node.set_module(nullptr);
      else
        root_node.set_module(factory.build(module_class));
      th::Number* selectindex = th::get_field<th::Number*>(root_data, "selectindex");
      if (selectindex)
        root_node.set_select_index(static_cast<int>(selectindex->get_value())-1);
      th::Table* mapindex = th::get_field<th::Table*>(root_data, "mapindex");

      for (auto it = mapindex->get_array().begin(); it != mapindex->get_array().end(); it++)
      {
        th::Table* data = dynamic_cast<th::Table*>(*it);
        if (data)
        {
          th::Number* nodeid = th::get_field<th::Number*>(data, "forwardNodeId");
          size_t nodeid_val = static_cast<size_t>(nodeid->get_value());
          root_node.add_input_index(nodeid_val);
        }
      }

      th::Table* children = th::get_field<th::Table*>(root_fields, "children");

      if (children)
      {
        for (auto it = children->get_array().begin(); it != children->get_array().end(); it++)
        {
          if ((*it)->type() == th::ObjType::TORCH)
          {
            Node<MatFwd>& child = build_graph(*it, factory);
            root_node.add_child(child);
          }
        }
      }

      return root_node;
    }

  }
}
