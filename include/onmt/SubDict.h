#pragma once

#include <set>
#include <map>
#include <string>
#include <vector>
#include "onmt/Eigen/MatrixBatch.h"

#include "onmt/Dictionary.h"

namespace onmt {
  class SubDict {
  public:
    /* build subdict class given a dictionary and map file */
    SubDict(const std::string &map_file, const Dictionary &dict);

    /* given a sequence of words, extract sub-dictionary */
    void extract(const std::vector<std::string> &words, std::set<size_t> &r);

    bool empty() const { return _map_rules.size() == 0; }

    /* reduce a linear weigth matrix to a given extract */
    static Eigen::Map<const Eigen::RowMajorMat<float> > reduce_linearweight(
      const Eigen::Map<const Eigen::RowMajorMat<float> > &w,
      const std::vector<size_t> &v);
  private:
    std::vector< std::multimap<std::string, size_t> > _map_rules;
  };
}