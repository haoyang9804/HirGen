#include <math.h>

#include <algorithm>
#include <graphNode.hpp>
#include <pool.hpp>

NODE_PTRs IPool::nodePool = {};
NODE_PTRs IPool::callNodePool = {};
NODE_PTRs IPool::globalFuncPool = {};
NODE_PTRs IPool::localFuncPool = {};
NODE_PTRs GarbagePool::garbagePool = {};

NODE_PTR IPool::pickANodeFrom(NODE_PTRs pool) { return pool[rand() % pool.size()]; }

int IPool::getTheNumberofFuncNodes() { return localFuncPool.size() + globalFuncPool.size(); }

NODE_PTR IPool::randomlyPickAFuncNode() {
  int numberOfNodes = getTheNumberofFuncNodes();
  int nodeID = rand() % numberOfNodes;

  if (nodeID < localFuncPool.size()) return localFuncPool[nodeID];

  return globalFuncPool[nodeID - localFuncPool.size()];
}

void IPool::clear() {
  NodePool::clear();
  GlobalFuncPool::clear();
}

NODE_PTRs IPool::shapeCompatibleNodes(NODE_PTR nd) {
  NODE_PTRs available_nodes = {};
  for (NODE_PTR lnode : nodePool) {
    if (nd->shape_compatible(lnode)) available_nodes.push_back(lnode);
  }
  return available_nodes;
}

NODE_PTRs& IPool::globalFuncPool_() { return globalFuncPool; }

NODE_PTR NodePool::findByName(std::string name) {
  for (NODE_PTR node : nodePool) {
    if (node->name_() == name) {
      return node;
    }
  }
  ASSERT_FALSE("Found no node");
}

bool NodePool::empty() { return nodePool.empty(); }

void NodePool::insert(NODE_PTR ptr) { nodePool.push_back(ptr); }

void NodePool::clear() {
  GarbagePool::remain(nodePool);
  nodePool.clear();
}

NODE_PTR NodePool::select() {
  int layersNum = pow(2, Bound::maxlayer) - 1;
  int index = rand() % layersNum + 1;
  int tmplayer = 1;
  while (true) {
    if (index <= pow(2, tmplayer) - 1) break;
    tmplayer++;
  }
  NODE_PTRs tss = Layer::get(tmplayer);
  return tss[rand() % tss.size()];
}

bool NodePool::canGenerateCallNode() {
  if (getTheNumberofFuncNodes() == 0) return false;
  for (NODE_PTR funcNode : globalFuncPool) {
    if (funcNode->hasnoparams_()) return true;
  }
  for (NODE_PTR funcNode : localFuncPool) {
    if (funcNode->hasnoparams_()) return true;
  }
  if (nodePool.size() == 0) return false;
  return true;
}

NODE_PTR GlobalFuncPool::select() {
  int numberOfNodes = getTheNumberofFuncNodes();
  int nodeID = rand() % numberOfNodes;

  if (nodeID < localFuncPool.size()) return localFuncPool[nodeID];

  return globalFuncPool[nodeID - localFuncPool.size()];
}

NODE_PTRs GlobalFuncPool::getAvailableNodes() {
  NODE_PTRs availableFuncNodes;
  if (!nodePool.empty()) {
    std::merge(globalFuncPool.begin(), globalFuncPool.end(), localFuncPool.begin(),
               localFuncPool.end(), std::back_inserter(availableFuncNodes));
  } else {
    for (NODE_PTR funcNode : globalFuncPool) {
      if (funcNode->hasnoparams_()) availableFuncNodes.push_back(funcNode);
    }
    for (NODE_PTR funcNode : localFuncPool) {
      if (funcNode->hasnoparams_()) availableFuncNodes.push_back(funcNode);
    }
  }
  return availableFuncNodes;
}

void GlobalFuncPool::insert(NODE_PTR ptr) { globalFuncPool.push_back(ptr); }

void GlobalFuncPool::clear() { globalFuncPool.clear(); }