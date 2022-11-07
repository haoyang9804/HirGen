#include <constraints.hpp>
#include <globalVar.hpp>
#include <iostream>
#include <pool.hpp>
#include <random.hpp>
#include <random>
#include <stringUtils.hpp>

bool Constraint::_shape_equalProduct(const SHAPE& sp1, const SHAPE& sp2) {
  int pd1 = 1, pd2 = 1;
  for (int ele : sp1) pd1 *= ele;
  for (int ele : sp2) pd2 *= ele;
  return pd1 == pd2;
}

pSB Constraint::BroadcastRel_check(const SHAPE& sp1, const SHAPE& sp2) {
  if (Custom::cLevel == relaxed) return pSB(picAShape(), true);
  size_t dim1 = sp1.size();
  size_t dim2 = sp2.size();
  size_t i = 1;
  SHAPE oshape = std::vector<int>();
  if (_shape_equalProduct(sp1, sp2)) return pSB(sp1, true);
  for (; i <= std::min(dim1, dim2); ++i) {
    int s1 = sp1[dim1 - i];
    int s2 = sp2[dim2 - i];
    if (s1 == 1) {
      oshape.push_back(s2);
    } else if (s2 == 1) {
      oshape.push_back(s1);
    } else if (s1 == -1) {  // s1 = relay.Any()
      oshape.push_back(s2);
    } else if (s2 == -1) {  // s2 = relay.Any()
      oshape.push_back(s1);
    } else if (s1 == s2) {
      oshape.push_back(s1);
    } else {
      return pSB(SHAPE(), false);
    }
  }
  size_t max_dim = std::max(dim1, dim2);
  auto& rshape = (dim1 > dim2) ? sp1 : sp2;
  for (; i <= max_dim; ++i) {
    oshape.push_back(rshape[max_dim - i]);
  }
  std::reverse(oshape.begin(), oshape.end());
  return pSB(oshape, true);
}

// generate a new node based on type, shape and BoradcaseRel rule
pST Constraint::BroadcastRel_generate(const DATATYPE& type, const SHAPE& shape) {
  if (Custom::cLevel == relaxed) return pST(picAShape(), rollAType());
  if (shape.empty()) {
    return pST(picAShape(), type);
  } else {
    SHAPE newshapeInv;
    size_t shape_size = shape.size();

    for (int i = 0; i < shape_size; i++) {
      if (shape[i] == 1)
        newshapeInv.push_back(rand() % Bound::dimValueUpbound + 1);
      else
        newshapeInv.push_back(shape[i]);
    }

    return pST(newshapeInv, type);
  }
}

void Constraint::createTensorShapeRelation(NODE_PTR nd) {
  auto localNodePool = IPool::nodePool;
  size_t siz = localNodePool.size();
  for (size_t i = 0; i < siz; i++) {
    if (localNodePool[i]->isVar() || localNodePool[i]->isConst()) {
      pSB res = BroadcastRel_check(nd->shape_(), localNodePool[i]->shape_());
      if (res.second) {
        nd->add_BroadCastRelTensorptr(localNodePool[i]);
        localNodePool[i]->add_BroadCastRelTensorptr(nd);
      }
    } else if (localNodePool[i]->isBop() || localNodePool[i]->isUop() ||
               localNodePool[i]->isCall()) {
      if (localNodePool[i]->resNode_()->isTuple()) continue;
      pSB res = BroadcastRel_check(nd->shape_(), localNodePool[i]->resNode_()->shape_());
      if (res.second) {
        nd->add_BroadCastRelTensorptr(localNodePool[i]);
        localNodePool[i]->add_BroadCastRelTensorptr(nd);
      }
    }
  }
}