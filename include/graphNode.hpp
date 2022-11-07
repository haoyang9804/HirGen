#ifndef GRAPHNODE_HPP_
#define GRAPHNODE_HPP_

#include <set>
#include <string>
#include <unordered_map>

#include "globalType.hpp"
#include "graphNode.hpp"

struct node_compare {
  bool operator()(NODE_PTR nd1, NODE_PTR nd2) const { return nd1->ID_() < nd2->ID_(); }
};

class GNode {
 protected:
  static std::set<NODE_PTR, node_compare> outputs;
  static std::set<NODE_PTR, node_compare> inputs;
  static std::unordered_map<int, NODE_PTRs> layerToNodeptrs;
  static NODE_PTR tail_tuple_RelayStmt();

 public:
  static void clear();
};

class OutputNode : public GNode {
 public:
  static void clear();
  static size_t size();
  static bool empty();
  static NODE_PTRs mirror();
  static void insert(NODE_PTR nd);
  static void erase(NODE_PTR nd);
  static void update(NODE_PTR newtensor, NODE_PTR lhs = nullptr, NODE_PTR rhs = nullptr);
};

class InputNode : public GNode {
 public:
  static void clear();
  static size_t size();
  static bool empty();
  static NODE_PTRs mirror();
  static void insert(NODE_PTR nd);
  static void erase(NODE_PTR nd);
  static std::string inputString();
  static auto getInputs() {
    NODE_PTR rvNode = tail_tuple_RelayStmt();
    std::string inputnames_str = inputString();
    struct INPUT {
      std::string inputnames_str;
      NODE_PTR rvNode;
    };
    return INPUT{inputnames_str, rvNode};
  }
};

class Layer : public GNode {
 public:
  static void clear();
  static void fill(NODE_PTR nd);
  static NODE_PTRs get(int id);
};

#endif