#ifndef CONSTRAINTS_HPP_
#define CONSTRAINTS_HPP_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "globalType.hpp"

class Constraint {
 public:
  static pSB BroadcastRel_check(const SHAPE& sp1, const SHAPE& sp2);
  static pST BroadcastRel_generate(const DATATYPE& type, const SHAPE& shape);
  static void createTensorShapeRelation(NODE_PTR nd);

 private:
  static bool _shape_equalProduct(const SHAPE& sp1, const SHAPE& sp2);
};

#endif