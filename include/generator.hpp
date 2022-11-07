/*!
*\brief This header file includes class Generator and it's observer, which is
used for generate nodes and their corresponding expressions and record the
generation.
*/
#ifndef GENERATOR_HPP_
#define GENERATOR_HPP_
#include <functional>
#include <list>
#include <pythonFile.hpp>
#include <string>

// #include "coverage.hpp"
#include "globalType.hpp"
#include "globalVar.hpp"
#include "graphNode.hpp"
#include "logging.hpp"
#include "node.hpp"
#include "pool.hpp"
#include "random.hpp"
#include "stringUtils.hpp"

#define MAX(a, b) a > b ? a : b

#define tailStmt                    \
  generateTail();                   \
  int res = 1;                      \
  if (Custom::feature == "df") {    \
    generateInputsAndPredictions(); \
    OutputNode::clear();            \
  }

#define CONSTRUCTOR(typename)                                     \
  typename() = delete;                                            \
  typename(OPTYPE optype, std::function<DATATYPE()> randTypeFunc) \
      : optype(optype), randTypeFunc(randTypeFunc) {}

class IGenerator {
 public:
  virtual IGenerator* generate() = 0;
  virtual NODE_PTR get_newNode() = 0;
  NODE_PTR selectOrGenerateByShape(NODE_PTR nd);

 protected:
  void VarConstUpdate(NODE_PTR nd);
  inline DATATYPE _rvType(const OPTYPE& optype, const DATATYPE& datatype) {
    if (optype != "equal" && optype != "notEqual" && optype != "less" && optype != "lessEqual" &&
        optype != "greater" && optype != "greaterEqual" && optype != "isnan" &&
        optype != "isfinite" && optype != "isinf")
      return datatype;
    return "bool";
  }
};

class varconstGenerator : public IGenerator {
 public:
  IGenerator* generate() override {
    if (mode == 1) {
      generate_mode1();
    } else {
      generate_mode2();
    }
    return this;
  }
  NODE_PTR get_newNode() override { return newNode; }
  varconstGenerator() = delete;
  varconstGenerator(const DATATYPE& datatype) : datatype(datatype) {}
  varconstGenerator(const DATATYPE& datatype, const OPTYPE& binarytype, const SHAPE& shape)
      : datatype(datatype), binarytype(binarytype), shape(shape), mode(2) {}

 private:
  const DATATYPE datatype;
  const OPTYPE binarytype;
  const SHAPE shape;
  int mode = 1;
  NODE_PTR newNode = nullptr;
  void set_newNode(NODE_PTR newnode) { newNode = newnode; }

  void generate_mode1() {
    ASSERT(mode == 1, "mode isn't 1");
    int id = assignID();
    std::vector<int> shape = picAShape();
    if (Value::firstNode) {
      Value::firstNode = false;
      newNode = std::make_shared<varNode>(id, datatype, shape);
    } else {
      if (rand() % Bound::valueconstBound) {
        newNode = std::make_shared<varNode>(id, datatype, shape);
      } else {
        std::string value = generateValues(shape, 0, static_cast<int>(shape.size()), datatype);
        newNode = std::make_shared<constNode>(id, datatype, shape, value);
      }
    }
    ASSERT(newNode != nullptr, "newNode is nullptr, no varnode or constnode is generated.");
  }
  void generate_mode2() {
    ASSERT(mode == 2, "mode isn't 2");
    int id = assignID();
    pST st = Constraint::BroadcastRel_generate(datatype, shape);
    newNode = std::make_shared<varNode>(id, st.second, st.first);
    if (rand() % Bound::valueconstBound) {
      newNode = std::make_shared<varNode>(id, st.second, st.first);
    } else {
      std::string value = generateValues(st.first, 0, static_cast<int>(st.first.size()), st.second);
      newNode = std::make_shared<constNode>(id, st.second, st.first, value);
    }
    ASSERT(newNode != nullptr, "newNode is nullptr, no varnode or constnode is generated.");
  }
};

class uopGenerator : public IGenerator {
 public:
  CONSTRUCTOR(uopGenerator)
  IGenerator* generate() override;
  NODE_PTR get_newNode() override { return newNode; }

 private:
  OPTYPE optype;
  std::function<DATATYPE()> randTypeFunc;
  NODE_PTR newNode;
  void generateUnary_emptyPool(const DATATYPE& datatype, const OPTYPE& uoptype);
  void generateUnary_nonemptyPool(const DATATYPE& datatype, const OPTYPE& uoptype);
  void updateUopNode(NODE_PTR& vcnd, NODE_PTR& uopnd, const OPTYPE& uoptype,
                     const DATATYPE& datatype);
  NODE_PTR generateUopNode(const DATATYPE& datatype, const OPTYPE& uopType, const SHAPE& shape);
  void set_newNode(NODE_PTR newnode) { newNode = newnode; }
};

class bopGenerator : public IGenerator {
 public:
  CONSTRUCTOR(bopGenerator)
  IGenerator* generate() override;
  NODE_PTR get_newNode() override { return newNode; }

 private:
  OPTYPE optype;
  std::function<DATATYPE()> randTypeFunc;
  NODE_PTR newNode;
  void generateBinary_emptyPool(const DATATYPE& datatype, const OPTYPE& boptype);
  void generateBinary_nonemptyPool(const DATATYPE& datatype, const OPTYPE& boptype);
  void updateBopNode(NODE_PTR& lhs, NODE_PTR& rhs, NODE_PTR& bopnd, const OPTYPE& boptype,
                     const DATATYPE& datatype);
  NODE_PTR generateBopNode(const DATATYPE& datatype, const OPTYPE& binopType, const SHAPE& shape);
  void set_newNode(NODE_PTR newnode) { newNode = newnode; }
};

class funcGenerator : public IGenerator {
 public:
  IGenerator* generate() override;
  NODE_PTR get_newNode() override { return newNode; }

 protected:
  NODE_PTR fromFuncToVar(NODE_PTR funcnode);

 private:
  void mutateFunc(NODE_PTR funcnode);
  NODE_PTR newNode;
  void set_newNode(NODE_PTR newnode) { newNode = newnode; }
};

class callGenerator : public funcGenerator {
 public:
  IGenerator* generate() override;

 private:
  NODE_PTR newNode;
  void set_newNode(NODE_PTR newnode) { newNode = newnode; }
};

/*
uopGenerator* ug;
ug->option(NodePool::empty())->generate()
*/

/*implementation*/

/*==============================IGenerator==============================*/

NODE_PTR IGenerator::selectOrGenerateByShape(NODE_PTR nd) {
  NODE_PTRs available_nodes = IPool::shapeCompatibleNodes(nd);
  if (available_nodes.empty()) {
    if (rand() % 2) {
      NODE_PTR paramNode =
          std::make_shared<varNode>(assignID(), nd->dataType_(), reshape(nd->shape_()));
      VarConstUpdate(paramNode);
      return paramNode;
    }
    SHAPE newshape = reshape(nd->shape_());
    std::string value =
        generateValues(newshape, 0, static_cast<int>(newshape.size()), nd->dataType_());
    NODE_PTR paramNode = std::make_shared<constNode>(assignID(), nd->dataType_(), newshape, value);
    VarConstUpdate(paramNode);
    return paramNode;
  }
  return IPool::pickANodeFrom(available_nodes);
}

void IGenerator::VarConstUpdate(NODE_PTR nd) {
  if (nd->isVar()) InputNode::insert(nd);
  OutputNode::insert(nd);
  Layer::fill(nd);
  Constraint::createTensorShapeRelation(nd);
  NodePool::insert(nd);
  node_RelayStmt(nd);
}

/*==============================uopGenerator==============================*/

IGenerator* uopGenerator::generate() {
  auto tmpFunc = [&](auto roll, const OPTYPE& optype) {
    DATATYPE datatype = roll();
    if (NodePool::empty()) {
      generateUnary_emptyPool(datatype, optype);
    } else {
      generateUnary_nonemptyPool(datatype, optype);
    }
  };
  if (Custom::cLevel == strict) {
    tmpFunc(randTypeFunc, optype);
  } else if (Custom::cLevel == relaxed) {
    tmpFunc(rollAType, optype);
  } else {
    ASSERT_FALSE("uopGenerator's cLevel is unknown");
  }
  return this;
}

void uopGenerator::generateUnary_emptyPool(const DATATYPE& datatype, const OPTYPE& uoptype) {
  NODE_PTR vcnd = varconstGenerator(datatype).generate()->get_newNode();
  NODE_PTR uopnd =
      (uoptype != "reshape")
          ? generateUopNode(_rvType(uoptype, datatype), uoptype, vcnd->shape_())
          : generateUopNode(_rvType(uoptype, datatype), uoptype, reshape(vcnd->shape_()));
  if (Coverage::noProgress(uoptype, datatype, uopnd->shape_()))
    if (Custom::coverage) return;
  VarConstUpdate(vcnd);
  updateUopNode(vcnd, uopnd, uoptype, datatype);
}

void uopGenerator::generateUnary_nonemptyPool(const DATATYPE& datatype, const OPTYPE& uoptype) {
  NODE_PTR vcnd = NodePool::select();
  NODE_PTR uopnd =
      (uoptype != "reshape")
          ? generateUopNode(_rvType(uoptype, datatype), uoptype, vcnd->shape_())
          : generateUopNode(_rvType(uoptype, datatype), uoptype, reshape(vcnd->shape_()));
  if (vcnd->isUop() || vcnd->isBop()) {
    if (Coverage::noProgress(uoptype, datatype, uopnd->shape_(), vcnd->optype_()))
      if (Custom::coverage) return;
  } else if (Coverage::noProgress(uoptype, datatype, uopnd->shape_()))
    if (Custom::coverage) return;
  OutputNode::erase(vcnd);
  updateUopNode(vcnd, uopnd, uoptype, datatype);
}

void uopGenerator::updateUopNode(NODE_PTR& vcnd, NODE_PTR& uopnd, const OPTYPE& uoptype,
                                 const DATATYPE& datatype) {
  NODE_PTR uopnd_copy = nullptr, vcnd_copy = vcnd->copyNode();
  bool poison = false;
  using Accessory::name2mname;
  if (name2mname.count(vcnd->name_())) {
    poison = true;
    vcnd_copy->rename(name2mname[vcnd->name_()]);
  } else {
    vcnd_copy->rename(vcnd->name_());
  }
  if (poison) {
    uopnd_copy = generateUopNode(_rvType(uoptype, datatype), uoptype, vcnd->shape_());
    name2mname.insert({uopnd->name_(), uopnd_copy->name_()});
  }
  uopnd->assign_layer(vcnd->layer_() + 1);
  using Bound::maxlayer;
  maxlayer = MAX(maxlayer, uopnd->layer_());
  Constraint::createTensorShapeRelation(uopnd);
  Layer::fill(uopnd);
  generateUnaryRelayStmt NodePool::insert(uopnd);
  OutputNode::update(uopnd, vcnd);
  Value::opNum++;
  set_newNode(uopnd);
}

NODE_PTR uopGenerator::generateUopNode(const DATATYPE& datatype, const OPTYPE& uopType,
                                       const SHAPE& shape) {
  int id = assignID();
  std::shared_ptr<node> resNode = std::make_shared<varNode>(id, datatype, shape);
  auto rv = std::make_shared<uopNode>(id, uopType, resNode);
  GarbagePool::remain(resNode);
  return rv;
}

/*==============================bopGenerator==============================*/

IGenerator* bopGenerator::generate() {
  auto tmpFunc = [&](auto roll, const OPTYPE& optype) {
    DATATYPE datatype = roll();
    if (NodePool::empty()) {
      generateBinary_emptyPool(datatype, optype);
    } else {
      generateBinary_nonemptyPool(datatype, optype);
    }
  };
  if (Custom::cLevel == strict) {
    tmpFunc(randTypeFunc, optype);
  } else if (Custom::cLevel == relaxed) {
    tmpFunc(rollAType, optype);
  } else {
    ASSERT_FALSE("binopGenerator's cLevel is unknown");
  }
  return this;
}

void bopGenerator::generateBinary_emptyPool(const DATATYPE& datatype, const OPTYPE& boptype) {
  NODE_PTR lhs = varconstGenerator(datatype).generate()->get_newNode();
  NODE_PTR rhs =
      varconstGenerator(lhs->dataType_(), boptype, lhs->shape_()).generate()->get_newNode();
  SHAPE newshape = Constraint::BroadcastRel_check(lhs->shape_(), rhs->shape_()).first;
  NODE_PTR bopnd = generateBopNode(_rvType(boptype, datatype), boptype, newshape);
  if (Coverage::noProgress(boptype, datatype, bopnd->shape_()))
    if (Custom::coverage) return;
  VarConstUpdate(lhs);
  VarConstUpdate(rhs);
  updateBopNode(lhs, rhs, bopnd, boptype, datatype);
}

void bopGenerator::generateBinary_nonemptyPool(const DATATYPE& datatype, const OPTYPE& boptype) {
  NODE_PTR lhs, rhs;
  // auto _generateBinary_otherwise_lhs = [](NODE_PTR& lhs, const DATATYPE& datatype) {
  //   lhs = NodePool::select();
  //   OutputNode::erase(lhs);
  // };
  auto _generateBinary_otherwise_rhs_tryfromPool = [&](NODE_PTR& lhs, NODE_PTR& rhs,
                                                       const OPTYPE& bop_type) {
    NODE_PTRs BroadCastRelptrs = lhs->BroadCastRelptrs_();
    size_t siz = BroadCastRelptrs.size();
    if (siz == 0) {
      rhs = varconstGenerator(lhs->dataType_(), bop_type, lhs->shape_()).generate()->get_newNode();
      // VarConstUpdate(rhs);
      // lhs->add_BroadCastRelTensorptr(rhs);
      // rhs->add_BroadCastRelTensorptr(lhs);
      return true;
    } else {
      if (bop_type == "divide" or bop_type == "floor_divide" or bop_type == "mod" or
          bop_type == "floor_mod") {
        while (true) {
          rhs = BroadCastRelptrs[rand() % siz];
          if (rhs->isConst() && rhs->const_value_() == "0") {
            continue;
          } else {
            break;
          }
        }
      } else {
        rhs = BroadCastRelptrs[rand() % siz];
      }
      return false;
      // OutputNode::erase(rhs);
    }
  };
  auto _generateBinary_otherwise_rhs_newNode = [&](NODE_PTR& lhs, NODE_PTR& rhs,
                                                   const OPTYPE& bop_type) {
    rhs = varconstGenerator(lhs->dataType_(), bop_type, lhs->shape_()).generate()->get_newNode();
    // VarConstUpdate(rhs);
    // lhs->add_BroadCastRelTensorptr(rhs);
    // rhs->add_BroadCastRelTensorptr(lhs);
    return true;
  };
  auto _generateBinary_otherwise_rhs =
      [&_generateBinary_otherwise_rhs_tryfromPool, &_generateBinary_otherwise_rhs_newNode](
          NODE_PTR& lhs, NODE_PTR& rhs, const DATATYPE& datatype, const OPTYPE& bop_type) {
        if (trySelectFromPool()) {
          return _generateBinary_otherwise_rhs_tryfromPool(lhs, rhs, bop_type);
        } else {
          return _generateBinary_otherwise_rhs_newNode(lhs, rhs, bop_type);
        }
      };
  lhs = NodePool::select();
  bool newrhs = _generateBinary_otherwise_rhs(lhs, rhs, datatype, boptype);
  SHAPE newshape = Constraint::BroadcastRel_check(lhs->shape_(), rhs->shape_()).first;
  NODE_PTR bopnd = generateBopNode(_rvType(boptype, datatype), boptype, newshape);

  bool noprogress;

  if (lhs->isBop() || lhs->isUop()) {
    noprogress = Coverage::noProgress(boptype, datatype, bopnd->shape_(), lhs->optype_());
    if (rhs->isBop() || rhs->isUop()) {
      noprogress =
          noprogress && Coverage::noProgress(boptype, datatype, bopnd->shape_(), rhs->optype_());
    }
  } else if (rhs->isBop() || rhs->isUop()) {
    noprogress = Coverage::noProgress(boptype, datatype, bopnd->shape_(), rhs->optype_());
  } else {
    noprogress = Coverage::noProgress(boptype, datatype, bopnd->shape_());
  }
  if (noprogress)
    if (Custom::coverage) return;

  // update

  OutputNode::erase(lhs);
  if (newrhs) {
    VarConstUpdate(rhs);
    lhs->add_BroadCastRelTensorptr(rhs);
    rhs->add_BroadCastRelTensorptr(lhs);
  } else {
    OutputNode::erase(rhs);
  }
  updateBopNode(lhs, rhs, bopnd, boptype, datatype);
}

void bopGenerator::updateBopNode(NODE_PTR& lhs, NODE_PTR& rhs, NODE_PTR& bopnd,
                                 const OPTYPE& boptype, const DATATYPE& datatype) {
  NODE_PTR bopnd_copy = nullptr, lhs_copy = lhs->copyNode(), rhs_copy = rhs->copyNode();
  bool poison = false;
  using Accessory::name2mname;
  if (name2mname.count(lhs->name_())) {
    poison = true;
    lhs_copy->rename(name2mname[lhs->name_()]);
  } else {
    lhs_copy->rename(lhs->name_());
  }
  if (name2mname.count(rhs->name_())) {
    poison = true;
    rhs_copy->rename(name2mname[rhs->name_()]);
  } else {
    rhs_copy->rename(rhs->name_());
  }
  if (poison) {
    bopnd_copy = generateBopNode(_rvType(boptype, datatype), boptype, bopnd->shape_());
    name2mname.insert({bopnd->name_(), bopnd_copy->name_()});
  }
  bopnd->assign_layer(MAX(lhs->layer_(), rhs->layer_()));
  using Bound::maxlayer;
  maxlayer = MAX(maxlayer, bopnd->layer_());
  Constraint::createTensorShapeRelation(bopnd);
  Layer::fill(bopnd);
  generateBinaryRelayStmt NodePool::insert(bopnd);
  OutputNode::update(bopnd, lhs, rhs);
  Value::opNum++;
  set_newNode(bopnd);
}

NODE_PTR bopGenerator::generateBopNode(const DATATYPE& datatype, const OPTYPE& binopType,
                                       const SHAPE& shape) {
  int id = assignID();
  std::shared_ptr<node> resNode = std::make_shared<varNode>(id, datatype, shape);
  auto rv = std::make_shared<bopNode>(id, binopType, resNode);
  GarbagePool::remain(resNode);
  return rv;
}

/*==============================funcGenerator==============================*/

NODE_PTR funcGenerator::fromFuncToVar(NODE_PTR funcnode) {
  std::string funcCallName = "", mfuncCallName = "";
  using Accessory::name2mname;
  if (funcnode->isGlobal()) {
    funcCallName = funcnode->name_() + "_call";
    File::ofile << funcCallName << " = mod.get_global_var(\'" << funcnode->name_() << "\')"
                << std::endl;
    if (name2mname.count(funcnode->name_())) {
      mfuncCallName = name2mname[funcnode->name_()] + "_call";
      File::ofile << mfuncCallName << " = mutated_mod.get_global_var(\'"
                  << name2mname[funcnode->name_()] << "\')" << std::endl;
    }
  } else {
    funcCallName = funcnode->name_();
  }
  int nid;
  NODE_PTR rvnode;
  while (true) {
    std::tie(nid, rvnode) = funcnode->pickArvNode();
    if (rvnode->isFunc()) {
      funcnode = rvnode;
    } else {
      break;
    }
  }
  int id = assignID();
  NODE_PTR resnode = std::make_shared<varNode>(id, rvnode->dataType_(), rvnode->shape_());
  NODE_PTR callnode = std::make_shared<callNode>(id, nid, resnode, funcnode);
  GarbagePool::remain(resnode);
  std::string paramsStr = "";
  if (!funcnode->hasnoparams_()) {
    auto paramNode = funcnode->paramNode_();
    if (paramNode->isVar()) {
      NODE_PTR paramNodetmp = selectOrGenerateByShape(paramNode);
      paramsStr += "relay.reshape(" + paramNodetmp->name_() + ".astype(\'" +
                   paramNode->dataType_() + "\'), " +
                   SHAPE_to_string(paramNode->shape_(), "[", "]") + ")";
    } else if (paramNode->isTuple()) {
      auto members = paramNode->members_();
      for (int i = 0; i < members.size(); i++) {
        NODE_PTR paramNode = selectOrGenerateByShape(members[i]);
        paramsStr += "relay.reshape(" + paramNode->name_() + ".astype(\'" +
                     members[i]->dataType_() + "\'), " +
                     SHAPE_to_string(members[i]->shape_(), "[", "]") + "), ";
      }
      for (auto ele : members) ele.reset();
    } else {
      paramNode.reset();
      ASSERT_FALSE("funcnode->paramNode's type is neither tupleNode nor varNode");
    }
    paramNode.reset();
  }
  auto rvNode_ = funcnode->rvNode_();
  if (rvNode_->isTuple()) {
    File::ofile << callnode->name_() << " = relay.TupleGetItem(" << funcCallName << "(" << paramsStr
                << "), " << std::to_string(nid) << ")" << std::endl;
    if (mfuncCallName != "") {
      std::string call_str = "call_" + std::to_string(assignID());
      File::ofile << call_str << " = relay.TupleGetItem(" << mfuncCallName << "(" << paramsStr
                  << "), " << std::to_string(nid) << ")" << std::endl;
      name2mname.insert({callnode->name_(), call_str});
    }
  } else {
    File::ofile << callnode->name_() << " = " << funcCallName << "(" << paramsStr << ")"
                << std::endl;
    if (mfuncCallName != "") {
      std::string call_str = "call_" + std::to_string(assignID());
      File::ofile << call_str << " = " << funcCallName << "(" << paramsStr << ")" << std::endl;
      name2mname.insert({callnode->name_(), call_str});
    }
  }
  Layer::fill(callnode);
  Constraint::createTensorShapeRelation(callnode);
  NodePool::insert(callnode);
  OutputNode::insert(callnode);
  return callnode;
}

IGenerator* funcGenerator::generate() {
  if (OutputNode::empty()) return this;
  auto [inputnames_str, rvNode] = InputNode::getInputs();
  NODE_PTRs inputs_vec = InputNode::mirror();
  int fid = rvNode->isTuple() ? rvNode->ID_() : assignID();
  NODE_PTR funcnode;
  if (inputs_vec.size() == 0)
    funcnode = std::make_shared<funcNode>(fid, rvNode);
  else if (inputs_vec.size() == 1)
    funcnode = std::make_shared<funcNode>(fid, rvNode, inputs_vec[0]);
  else {
    NODE_PTR tmptupleNode = std::make_shared<tupleNode>(fid, inputs_vec);
    funcnode = std::make_shared<funcNode>(fid, rvNode, tmptupleNode);
    GarbagePool::remain(tmptupleNode);
  }

  GarbagePool::remain(rvNode);
  function_RelayStmt(funcnode->name_(), inputnames_str, "output");
  GNode::clear();
  NodePool::clear();
  init_maxlayer();

  // because relay.build a returning-function function will throw a warning
  // message according to
  // https://github.com/apache/tvm/pull/10502#event-6197333598, we temporarily
  // change randomization into determinization
  if (true) {
    funcnode->assign_global(true);
    GlobalFuncPool::insert(funcnode);
    File::ofile << "mod[\'" << funcnode->name_() << "\'] = " << funcnode->name_() << std::endl;
    File::ofile << "mod = relay.transform.InferType()(mod)" << std::endl;
    mutateFunc(funcnode);
    File::ofile << "mutated_mod = relay.transform.InferType()(mutated_mod)" << std::endl;
  } else {
    ASSERT_FALSE("No implementation");
    // funcnode->assign_global(false);
    // throwIntoLocalFuncPool(funcnode);
    // OutputNode::insert(funcnode);
  }
  set_newNode(funcnode);
  logger("+++" + funcnode->name_() + " " + std::to_string(funcnode->rvNode_() == nullptr));
  return this;
}

void funcGenerator::mutateFunc(NODE_PTR funcnode) {
  int seed = rand() % 2;
  using Accessory::name2mname;
  if (seed == 0) {
    File::ofile << "mutated_mod[\'" << funcnode->name_() << "\'] = " << funcnode->name_()
                << std::endl;
    File::ofile << "mutated_mod = relay.transform.InferType()(mutated_mod)" << std::endl;
    // Create the same function as the original one. The new function returns a
    // call the the original function
    /*
    def @f(%x:float32, %y:float32) {
        add(%x, %y)
    }
    ||
    def @g(%x:float32, %y:float32) {
        add(%x, %y)
    }
    def @f() {
        @g(%x, %y)
    }
    */
    if (funcnode->hasnoparams_()) {
      std::string funcCallName = funcnode->name_() + "_call";
      File::ofile << funcCallName << " = mutated_mod.get_global_var(\'" << funcnode->name_()
                  << "\')" << std::endl;
      NODE_PTR callnode = std::make_shared<callNode>(assignID(), -1, nullptr, funcnode);
      File::ofile << callnode->name_() << " = " << funcCallName << "()" << std::endl;
      File::ofile << "output = " << callnode->name_() << std::endl;
      std::string funcname = "func_" + std::to_string(assignID());
      function_RelayStmt(funcname, "", "output");
      name2mname.insert({funcnode->name_(), funcname});
      File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
    } else {
      NODE_PTR paramnode = funcnode->paramNode_();
      if (paramnode->isVar()) {
        NODE_PTR vnode = paramnode->copyNode();
        node_RelayStmt(vnode);
        std::string funcCallName = funcnode->name_() + "_call";
        File::ofile << funcCallName << " = mutated_mod.get_global_var(\'" << funcnode->name_()
                    << "\')" << std::endl;
        NODE_PTR callnode = std::make_shared<callNode>(assignID(), -1, nullptr, funcnode);
        File::ofile << callnode->name_() << " = " << funcCallName << "(" << vnode->name_() << ")"
                    << std::endl;
        File::ofile << "output = " << callnode->name_() << std::endl;
        std::string funcname = "func_" + std::to_string(assignID());
        function_RelayStmt(funcname, vnode->name_(), "output");
        name2mname.insert({funcnode->name_(), funcname});
        File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
      } else if (paramnode->isTuple()) {
        std::vector<std::shared_ptr<node>> members = paramnode->members_();
        std::string funcCallName = funcnode->name_() + "_call";
        File::ofile << funcCallName << " = mutated_mod.get_global_var(\'" << funcnode->name_()
                    << "\')" << std::endl;
        NODE_PTR callnode = std::make_shared<callNode>(assignID(), -1, nullptr, funcnode);
        std::string paramstr = "";
        for (NODE_PTR member : members) {
          ASSERT(member->isVar(), "The member " + member->name_() + " is not a varNode");
          NODE_PTR newmember = member->copyNode();
          node_RelayStmt(newmember);
          paramstr += newmember->name_() + ",";
        }
        File::ofile << callnode->name_() << " = " << funcCallName << "(" << paramstr << ")"
                    << std::endl;
        File::ofile << "output = " << callnode->name_() << std::endl;
        std::string funcname = "func_" + std::to_string(assignID());
        function_RelayStmt(funcname, paramstr, "output");
        name2mname.insert({funcnode->name_(), funcname});
        File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
      } else {
        ASSERT_FALSE("Invalid paramnode type: not varNode or tupleNode");
      }
      NODE_PTR newfuncnode = std::make_shared<funcNode>(assignID(), funcnode);
    }

  } else if (seed == 1) {
    /*
    def @f(%x:float32, %y:float32) {
        add(%x, %y)
    }
    ||
    def @f(%x:float32, %y:float32) {
         %0 = fn (%x: float32, %y: float32) {
            add(%x, %y)
        };
        %0(%x1, %y1)
    }
    */
    if (funcnode->hasnoparams_()) {
      File::ofile << "output = " << funcnode->name_() << "()" << std::endl;
      std::string funcname = "func_" + std::to_string(assignID());
      function_RelayStmt(funcname, "", "output");
      name2mname.insert({funcnode->name_(), funcname});
      File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
    } else {
      NODE_PTR paramnode = funcnode->paramNode_();
      if (paramnode->isVar()) {
        NODE_PTR vnode = paramnode->copyNode();
        node_RelayStmt(vnode);
        File::ofile << "output = " << funcnode->name_() << "(" << vnode->name_() << ")"
                    << std::endl;
        std::string funcname = "func_" + std::to_string(assignID());
        function_RelayStmt(funcname, vnode->name_(), "output");
        name2mname.insert({funcnode->name_(), funcname});
        File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
      } else if (paramnode->isTuple()) {
        std::vector<std::shared_ptr<node>> members = paramnode->members_();
        std::string paramstr = "";
        for (NODE_PTR member : members) {
          ASSERT(member->isVar(), "The member " + member->name_() + " is not a varNode");
          NODE_PTR newmember = member->copyNode();
          node_RelayStmt(newmember);
          paramstr += newmember->name_() + ",";
        }
        File::ofile << "output = " << funcnode->name_() << "(" << paramstr << ")" << std::endl;
        std::string funcname = "func_" + std::to_string(assignID());
        function_RelayStmt(funcname, paramstr, "output");
        name2mname.insert({funcnode->name_(), funcname});
        File::ofile << "mutated_mod[\'" << funcname << "\'] = " << funcname << std::endl;
      } else {
        ASSERT_FALSE("Invalid paramnode type: not varNode or tupleNode");
      }
      NODE_PTR newfuncnode = std::make_shared<funcNode>(assignID(), funcnode);
    }
  } else if (seed == 2) {
    // Use an empty function node to wrap this function node and return a
    // function This mutation strategy has been abandoned because of this bug of
    // TVM: https://github.com/apache/tvm/pull/10502#event-6197333598. According
    // to the developer of TVM, TVM does not currently support functions
    // returning function although Relay supports it in high-level IR.
    /*
    def @f(%x:float32, %y:float32) {
        add(%x, %y)
    }
    ||
    def @f() {
        g(%x:float32, %y:float32) {
            add(%x, %y)
        }
    }
    */
    // NODE_PTR newfuncnode = std::make_shared<funcNode>(assignID(), funcnode);
    // function_RelayStmt(funcnode->name_(), "");
    // File::ofile << "mutated_mod[\'" << newfuncnode->name_() << "\']
    // = " << newfuncnode->name_() << std::endl;
  }
}

/*==============================callGenerator==============================*/

IGenerator* callGenerator::generate() {
  logger("callGenerator generates");
  if (!NodePool::canGenerateCallNode()) return this;
  NODE_PTRs availableFuncNodes = GlobalFuncPool::getAvailableNodes();
  NODE_PTR funcnode = IPool::pickANodeFrom(availableFuncNodes);
  logger("---" + funcnode->name_() + " " + std::to_string(funcnode->rvNode_() == nullptr));
  set_newNode(fromFuncToVar(funcnode));
  return this;
}

#endif