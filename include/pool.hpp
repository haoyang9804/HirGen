#ifndef POOL_HPP_
#define POOL_HPP_

#include "constraints.hpp"
#include "globalType.hpp"

class IPool {
 protected:
  static NODE_PTRs nodePool;
  static NODE_PTRs callNodePool;
  static NODE_PTRs globalFuncPool;
  static NODE_PTRs localFuncPool;
  static int getTheNumberofFuncNodes();

 private:
  friend class Constraint;

 public:
  static NODE_PTRs& globalFuncPool_();
  void clear();
  static NODE_PTR randomlyPickAFuncNode();
  static NODE_PTR pickANodeFrom(NODE_PTRs pool);
  static NODE_PTRs shapeCompatibleNodes(NODE_PTR nd);
};

class NodePool : public IPool {
 public:
  static NODE_PTR findByName(std::string name);
  static bool empty();
  static void insert(NODE_PTR ptr);
  static void clear();
  static NODE_PTR select();
  static bool canGenerateCallNode();
};

class GlobalFuncPool : public IPool {
 public:
  static NODE_PTR select();
  static NODE_PTRs getAvailableNodes();
  static void insert(NODE_PTR ptr);
  static void clear();
};

/**
 * ? WHY WE NEED THIS?
 * * Some derivative classes of node class contain weak_ptr to avoid
 * * recurrent reference introduced by shared_ptr. If we have no garbagePool
 * * to place the share_ptrs of these weak_ptrs, then these weak_ptrs will
 * * be ruthlessly expired.
 */
class GarbagePool {
 private:
  static NODE_PTRs garbagePool;

 public:
  static void remain(NODE_PTR garbage) { garbagePool.push_back(garbage); }
  static void remain(NODE_PTRs garbages) {
    for (auto ele : garbages) {
      garbagePool.push_back(ele);
    }
  }
};

#endif