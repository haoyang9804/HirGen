/*!
 *\brief The header file defines the Observer class of Generator.
 */

#ifndef COVERAGE_HPP_
#define COVERAGE_HPP_

#include <set>
#include <tuple>

#include "globalType.hpp"
#include "logging.hpp"
#include "stringUtils.hpp"

#include <iostream>

class Coverage {
 public:
  static void load();
  static void save();
  static void update(const OPTYPE& optype, const DATATYPE& datatype, const SHAPE_STR_SQUARE& shape,
                     const OPTYPE& preoptype);
  static void load(const OPTYPE& optype, const DATATYPE& datatype, const SHAPE_STR_SQUARE& shape,
                   const OPTYPE& preoptype);
  static bool noProgress(const OPTYPE& optype, const DATATYPE& datatype,
                         const SHAPE& shape = SHAPE(), const OPTYPE& preoptype = "");
  static int get_score();

 private:
  // static std::set<OPTYPE> optypes;
  static std::set<std::tuple<OPTYPE, SHAPE_STR_SQUARE>> opTypeShape;
  static std::set<std::tuple<OPTYPE, OPTYPE>> opTypeEdges;
  static std::set<std::tuple<OPTYPE, DATATYPE>> optypeDatatypeTuple;

  // static std::set<OPTYPE> old_optypes;
  static std::set<std::tuple<OPTYPE, SHAPE_STR_SQUARE>> old_opTypeShape;
  static std::set<std::tuple<OPTYPE, OPTYPE>> old_opTypeEdges;
  static std::set<std::tuple<OPTYPE, DATATYPE>> old_optypeDatatypeTuple;
  static int score;

  // static void update_optypes(const OPTYPE& optype) { optypes.insert(optype); }
  static void update_opTypeEdges(const OPTYPE& optype, const OPTYPE& preoptype) {
    opTypeEdges.insert(std::make_tuple(preoptype, optype));
  }
  static void update_optypeDatatypeTuple(const OPTYPE& optype, const DATATYPE& datatype) {
    optypeDatatypeTuple.insert(std::make_tuple(optype, datatype));
  }
  static void update_opTypeShape(const OPTYPE& optype, const SHAPE_STR_SQUARE& shape) {
    opTypeShape.insert(std::make_tuple(optype, shape));
  }
  static void load_opTypeEdges(const OPTYPE& optype, const OPTYPE& preoptype) {
    old_opTypeEdges.insert(std::make_tuple(preoptype, optype));
  }
  static void load_optypeDatatypeTuple(const OPTYPE& optype, const DATATYPE& datatype) {
    old_optypeDatatypeTuple.insert(std::make_tuple(optype, datatype));
  }
  static void load_opTypeShape(const OPTYPE& optype, const SHAPE_STR_SQUARE& shape) {
    old_opTypeShape.insert(std::make_tuple(optype, shape));  
  }
  static void update_score(int extent) { score += extent; }
};

#endif