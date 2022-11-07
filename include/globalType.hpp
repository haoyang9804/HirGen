#ifndef GLOBALTYPE_HPP_
#define GLOBALTYPE_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "node.hpp"

using DATATYPE = std::string;
using OPTYPE = std::string;
using SHAPE = std::vector<int>;
using NODE_PTR = std::shared_ptr<node>;
using NODE_PTRs = std::vector<NODE_PTR>;
using pST = std::pair<SHAPE, DATATYPE>;
using pSB = std::pair<SHAPE, bool>;

#include "stringUtils.hpp"

class SHAPE_STR_SQUARE {
 private:
  std::string shape_str;

 public:
  SHAPE_STR_SQUARE() = default;
  SHAPE_STR_SQUARE(SHAPE shape) { shape_str = SHAPE_to_string(shape, "[", "]"); }
  SHAPE_STR_SQUARE(std::string shape_str) {
    ASSERT(shape_str.front() == '[' && shape_str.back() == ']',
           "Ill SHAPE_STR_SQUARE form: " + shape_str + " with length " +
               std::to_string(shape_str.size()));
    this->shape_str = shape_str;
  }
  SHAPE_STR_SQUARE(const SHAPE_STR_SQUARE& shape_str) { this->shape_str = shape_str.str(); }
  SHAPE_STR_SQUARE& operator=(const SHAPE_STR_SQUARE& shape_str) {
    this->shape_str = shape_str.str();
    return *this;
  }
  ~SHAPE_STR_SQUARE() { shape_str.clear(); }
  std::string str() { return shape_str; }
  bool empty() { return shape_str.empty(); }
  std::string str() const { return shape_str; }
  bool empty() const { return shape_str.empty(); }

  friend std::ostream& operator<<(std::ostream& output, const SHAPE_STR_SQUARE& str) {
    output << str.shape_str;
    return output;
  }
};

bool operator<(const SHAPE_STR_SQUARE& s1, const SHAPE_STR_SQUARE& s2);

#endif