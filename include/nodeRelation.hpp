#ifndef NODERELATION_HPP_
#define NODERELATION_HPP_
/**
 * @brief nodeRelation is used for describing the relation of all types
 * of nodes except for funcNode
 * ? Why we need this class?
 * * we need to record all relationship to do test case reduction
 * * and subgraph reuse for another test.
 * ? What relation should we record?
 * * uopNode and bopNode have parent(s) (parent node type: uopNode, bopNode, varNode, constNode)
 * ! funcNode has parent(s) (parent node type: funcNode)
 * TODO: funcNode is planned to be placed in funcRelation
 */

#include <map>
#include <set>
#include <tuple>

#include "globalType.hpp"
#include "node.hpp"

class nodeRelation {
 private:
  static std::map<NODE_PTR, NODE_PTRs> parents;
  static std::map<NODE_PTR, NODE_PTRs> children;
  static std::set<std::tuple<int, int>> parent_child;
  NODE_PTR nd;

 public:
  nodeRelation() = default;
  nodeRelation(NODE_PTR nd) : nd(nd) {
    if (!parents.count(nd)) {
      parents.insert({nd, NODE_PTRs()});
    }
    if (!children.count(nd)) {
      children.insert({nd, NODE_PTRs()});
    }
  }
  nodeRelation* addParent(NODE_PTR pnd) {
    parents[nd].push_back(pnd);
    children[pnd].push_back(nd);
    return this;
  }
};

#endif