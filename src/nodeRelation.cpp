#include <nodeRelation.hpp>

std::map<NODE_PTR, NODE_PTRs> nodeRelation::parents = {};
std::map<NODE_PTR, NODE_PTRs> nodeRelation::children = {};
std::set<std::tuple<int, int>> nodeRelation::parent_child = {};