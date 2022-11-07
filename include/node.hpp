/*!
 * \brief node and its descendants contains all information needed by
 *        kind of node to build a well-defined computational graph.
 */
#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "globalVar.hpp"
#include "logging.hpp"

template <typename Base, typename T>
inline bool isinstanceof(const T*) {
  return std::is_base_of<Base, T>::value;
}

class node {
 protected:
  int ID;
  std::string name;
  std::string nodeType;

 public:
  node() = default;
  void log() { logger("node " + name + " created"); }

  bool isVar() { return this->nodeType == "var"; }
  bool isConst() { return this->nodeType == "const"; }
  bool isFunc() { return this->nodeType == "func"; }
  bool isCall() { return this->nodeType == "call"; }
  bool isTuple() { return this->nodeType == "tuple"; }
  bool isBop() { return this->nodeType == "bop"; }
  bool isUop() { return this->nodeType == "uop"; }
  std::string name_() { return this->name; }
  std::string nodeType_() { return nodeType; }
  void change_nodeType(std::string nodeType) { this->nodeType = nodeType; }
  void rename(std::string name) { this->name = name; }
  int ID_() { return this->ID; }
  void setID(int id) { this->ID = id; }
  void setNodeType_(std::string nodeType) { this->nodeType = nodeType; }

  virtual std::shared_ptr<node> copyNode() { ASSERT_FALSE("copyNode unimplemented"); }

  // func
  virtual std::tuple<int, std::shared_ptr<node>> pickArvNode() {
    ASSERT_FALSE("pickArvNode unimplemented");
  }
  virtual bool hasnoparams_() { ASSERT_FALSE("hasnoparams_ unimplemented"); }
  virtual void assign_rvNode(std::shared_ptr<node> nd) {
    ASSERT_FALSE("assign_rvNode unimplemented");
  }
  virtual void assign_paramNode(std::shared_ptr<node> nd) {
    ASSERT_FALSE("assign_paramNode unimplemented");
  }
  virtual std::shared_ptr<node> rvNode_() { ASSERT_FALSE("rvNode_ unimplemented"); }
  virtual std::shared_ptr<node> paramNode_() { ASSERT_FALSE("paramNode_ unimplemented"); }
  virtual void assign_global(bool gb) { ASSERT_FALSE("assign_global unimplemented"); }
  virtual bool isGlobal() { ASSERT_FALSE("isGlobal unimplemented"); }

  // varConstTuple
  virtual void assign_layer(int layer) { ASSERT_FALSE("assign_layer unimplemented"); }
  virtual int layer_() { ASSERT_FALSE(this->nodeType + " layer_ unimplemented"); }

  // singleNode
  virtual void clear_ancestors() { ASSERT_FALSE("clear_ancestors unimplemented"); }
  virtual bool shape_compatible(std::shared_ptr<node> node_ptr) {
    ASSERT_FALSE(this->nodeType + " shape_compatible unimplemented");
  }
  virtual std::vector<int> shape_() {
    ASSERT_FALSE(this->nodeType + " " + std::to_string(this->ID) + " shape_ unimplemented");
  }
  virtual std::vector<std::shared_ptr<node>> BroadCastRelptrs_() {
    ASSERT_FALSE("BroadCastRelptrs_ unimplemented");
  }
  virtual void add_BroadCastRelTensorptr(std::shared_ptr<node> node_ptr) {
    ASSERT_FALSE("add_BroadCastRelTensorptr unimplemented");
  }
  virtual void assign_dataType(std::string str) { ASSERT_FALSE("assign_dataType unimplemented"); }
  virtual std::string dataType_() { ASSERT_FALSE("dataType_ unimplemented"); }
  virtual bool dataType_equal(std::shared_ptr<node> node_ptr) {
    ASSERT_FALSE("dataType_equal unimplemented");
  }

  // constNode
  virtual std::string const_value_() { ASSERT_FALSE("const_value_ unimplemented"); }

  // tupleNode
  virtual void addMember(std::shared_ptr<node> member) { ASSERT_FALSE("addMember unimplemented"); }
  virtual std::vector<std::shared_ptr<node>> members_() {
    ASSERT_FALSE(this->nodeType_() + " members_ unimplemented");
  }

  // opNode
  virtual std::string optype_() { ASSERT_FALSE("optype_ unimplemented"); }
  virtual void assign_resNode(std::shared_ptr<node> nd) {
    ASSERT_FALSE("assign_resNode unimplemented");
  }
  virtual std::shared_ptr<node> resNode_() { ASSERT_FALSE("resNode_ unimplemented"); }

  // bopNode
  virtual void assign_lhs(std::shared_ptr<node> lhs) { ASSERT_FALSE("assign_lhs unimplemented"); }
  virtual void assign_rhs(std::shared_ptr<node> rhs) { ASSERT_FALSE("assign_rhs unimplemented"); }
  virtual std::shared_ptr<node> lhs_() { ASSERT_FALSE("lhs_ unimplemented"); }
  virtual std::shared_ptr<node> rhs_() { ASSERT_FALSE("rhs_ unimplemented"); }

  // uopNode

  // callNode
  virtual void assign_funcNode(std::shared_ptr<node> nd) {
    ASSERT_FALSE("assign_funcNode unimplemented");
  }
  virtual std::shared_ptr<node> funcNode_() { ASSERT_FALSE("funcNode_ unimplemented"); }
  virtual int nid_() { ASSERT_FALSE("nid_ unimplemented"); }
};

class varConstTupleNode : public node {
 protected:
  int layer = 1;

 public:
  void assign_layer(int layer) override { this->layer = layer; }
  int layer_() override { return this->layer; }
  std::shared_ptr<node> copyNode() override {
    ASSERT_FALSE("varConstTupleNode: copyNode unimplemented");
  }
};

class singleNode : public varConstTupleNode {
 protected:
  std::vector<int> shape;
  std::vector<std::weak_ptr<node>> BroadCastRelptrs = {};
  std::string dataType;  // int8 int16 int32 int64,uint8 uint16 uint32
                         // uint64,float16,float32 float64 bool
 public:
  std::shared_ptr<node> copyNode() override { ASSERT_FALSE("singleNode: copyNode unimplemented"); }
  bool shape_compatible(std::shared_ptr<node> node_ptr) override {
    int pd1 = 1, pd2 = 1;
    for (int ele : this->shape) pd1 *= ele;
    for (int ele : node_ptr->shape_()) pd2 *= ele;
    return pd1 == pd2;
  }
  std::vector<int> shape_() override { return this->shape; }
  std::vector<std::shared_ptr<node>> BroadCastRelptrs_() override {
    std::vector<std::shared_ptr<node>> BroadCastRelptrsTmp;
    for (auto ele : this->BroadCastRelptrs) {
      if (std::shared_ptr<node> sp = ele.lock()) {
        BroadCastRelptrsTmp.push_back(sp);
      } else {
        ASSERT_FALSE("Cannot get the resource of node");
      }
    }
    return BroadCastRelptrsTmp;
  }
  void add_BroadCastRelTensorptr(std::shared_ptr<node> node_ptr) override {
    this->BroadCastRelptrs.push_back(std::weak_ptr<node>(node_ptr));
  }
  void assign_dataType(std::string str) override { this->dataType = str; }
  std::string dataType_() override { return this->dataType; }
  bool dataType_equal(std::shared_ptr<node> node_ptr) override {
    ASSERT(node_ptr->isVar() || node_ptr->isConst(),
           "node " + node_ptr->name_() + " has no dataType attribute");
    return this->dataType == node_ptr->dataType_();
  }
};

class tupleNode : public varConstTupleNode {
 protected:
  std::vector<std::weak_ptr<node>> members;

 public:
  tupleNode() {}
  tupleNode(int ID, std::vector<std::shared_ptr<node>> members = {}) {
    this->ID = ID;
    this->nodeType = "tuple";
    this->name = "tuple_" + std::to_string(ID);
    for (auto ele : members) {
      logger("add member " + ele->name_());
      this->members.push_back(std::weak_ptr<node>(ele));
    }
    this->log();
  }
  void addMember(std::shared_ptr<node> member) override {
    this->members.push_back(std::weak_ptr<node>(member));
  }
  std::vector<std::shared_ptr<node>> members_() override {
    std::vector<std::shared_ptr<node>> membersTmp;
    for (auto ele : this->members) {
      if (std::shared_ptr<node> sp = ele.lock()) {
        membersTmp.push_back(sp);
      } else {
        ASSERT_FALSE("Cannot get the resource of node in members_");
      }
    }
    return membersTmp;
  }
  std::shared_ptr<node> copyNode() override { ASSERT_FALSE("tupleNode: copyNode unimplemented"); }
};

class varNode : public singleNode {
 public:
  varNode() = default;
  varNode(int ID, std::string dataType, std::vector<int> shape) {
    this->ID = ID;
    this->name = "var_" + std::to_string(ID);
    this->dataType = dataType;
    this->shape = shape;
    this->nodeType = "var";
    this->log();
  }
  std::shared_ptr<node> copyNode() final {
    return std::make_shared<varNode>(assignID(), this->dataType_(), this->shape_());
  }
};

class constNode : public singleNode {
 private:
  std::string const_value;

 public:
  constNode() {}
  constNode(int ID, std::string dataType, std::vector<int> shape, std::string const_value = "") {
    this->ID = ID;
    this->name = "const_" + std::to_string(ID);
    this->dataType = dataType;
    this->shape = shape;
    this->nodeType = "const";
    this->const_value = const_value;
    this->log();
  }
  std::string const_value_() override { return this->const_value; }
  std::shared_ptr<node> copyNode() final {
    return std::make_shared<constNode>(assignID(), this->dataType_(), this->shape_(),
                                       this->const_value_());
  }
};

// the node wrapping a returned varNode
class wrapperNode : public node {
 protected:
  std::shared_ptr<node> resNode;

 public:
  int layer_() override { return this->resNode->layer_(); }
  std::string dataType_() override { return this->resNode->dataType_(); }
  std::vector<int> shape_() override { return this->resNode->shape_(); }
  void assign_layer(int layer) override { this->resNode->assign_layer(layer); }
  void assign_resNode(std::shared_ptr<node> nd) override { this->resNode = std::move(nd); }
  std::shared_ptr<node> resNode_() override { return this->resNode; }
  std::vector<std::shared_ptr<node>> BroadCastRelptrs_() override {
    return this->resNode->BroadCastRelptrs_();
  }
  void add_BroadCastRelTensorptr(std::shared_ptr<node> node_ptr) override {
    this->resNode->add_BroadCastRelTensorptr(node_ptr);
  }
  bool shape_compatible(std::shared_ptr<node> node_ptr) override {
    return this->resNode->shape_compatible(node_ptr);
  }
  std::shared_ptr<node> copyNode() override { ASSERT_FALSE("wrapperNode: copyNode unimplemented"); }
};

class opNode : public wrapperNode {
 protected:
  std::string optype;

 public:
  std::string optype_() override { return optype; }
  std::shared_ptr<node> copyNode() override { ASSERT_FALSE("opNode: copyNode unimplemented"); }
};

class bopNode : public opNode {
 private:
  std::weak_ptr<node> lhs;
  std::weak_ptr<node> rhs;

 public:
  bopNode() {}
  bopNode(int ID, std::string btype, std::shared_ptr<node> resNode,
          std::shared_ptr<node> lhs = nullptr, std::shared_ptr<node> rhs = nullptr) {
    this->ID = ID;
    this->nodeType = "bop";
    this->name = "bop_" + std::to_string(ID);
    this->optype = btype;
    this->resNode = resNode;
    if (lhs != nullptr) this->lhs = std::weak_ptr<node>(lhs);
    if (rhs != nullptr) this->rhs = std::weak_ptr<node>(rhs);
    this->log();
  }
  void assign_lhs(std::shared_ptr<node> lhs) override { this->lhs = std::weak_ptr<node>(lhs); }
  void assign_rhs(std::shared_ptr<node> rhs) override { this->rhs = std::weak_ptr<node>(rhs); }
  std::shared_ptr<node> lhs_() override {
    if (std::shared_ptr<node> sp = this->lhs.lock()) {
      return sp;
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<node> rhs_() override {
    if (std::shared_ptr<node> sp = this->rhs.lock()) {
      return sp;
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<node> copyNode() final {
    return std::make_shared<bopNode>(assignID(), this->optype_(), this->resNode_(), this->lhs_(),
                                     this->rhs_());
  }
};

class uopNode : public opNode {
 private:
  std::weak_ptr<node> paramNode;
  
 public:
  uopNode() {}
  uopNode(int ID, std::string utype, std::shared_ptr<node> resNode,
          std::shared_ptr<node> paramNode = nullptr) {
    this->ID = ID;
    this->nodeType = "uop";
    this->name = "uop_" + std::to_string(ID);
    this->optype = utype;
    this->resNode = resNode;
    if (paramNode != nullptr) this->paramNode = std::weak_ptr<node>(paramNode);
    this->log();
  }
  void assign_paramNode(std::shared_ptr<node> paramNode) override {
    this->paramNode = std::weak_ptr<node>(paramNode);
  }
  std::shared_ptr<node> paramNode_() override {
    if (std::shared_ptr<node> sp = this->paramNode.lock()) {
      return sp;
    } else {
      // throw std::logic_error("Cannot get the resource of node in paramNode_
      // in uopNode");
      return nullptr;
    }
  }
  std::shared_ptr<node> copyNode() final {
    return std::make_shared<uopNode>(assignID(), this->optype_(), this->resNode_(),
                                     this->paramNode_());
  }
};

// TODO: change the parent node of funcNode to wrapper node and change rvNodes
// to resNode
class funcNode : public node {
 private:
  // std::vector<std::shared_ptr<node>> rvNodes;
  std::weak_ptr<node> rvNode;
  std::weak_ptr<node> paramNode;
  // std::vector<std::shared_ptr<node>> params;
  // bool rvTypeisTuple; // will be deprecated later
  bool global;

 public:
  funcNode() {}
  funcNode(int ID, std::shared_ptr<node> rvNode = nullptr,
           std::shared_ptr<node> paramNode = nullptr) {
    this->ID = ID;
    this->nodeType = "func";
    this->name = "func_" + std::to_string(ID);
    if (rvNode != nullptr) {
      this->rvNode = std::weak_ptr<node>(rvNode);
    }
    if (paramNode != nullptr) {
      this->paramNode = std::weak_ptr<node>(paramNode);
    }
    this->log();
  }
  void assign_rvNode(std::shared_ptr<node> nd) override { this->rvNode = std::weak_ptr<node>(nd); }
  void assign_paramNode(std::shared_ptr<node> nd) override { this->paramNode = std::weak_ptr<node>(nd); }
  std::shared_ptr<node> rvNode_() override {
    if (std::shared_ptr<node> sp = this->rvNode.lock()) {
      return sp;
    } else {
      throw std::logic_error("Cannot get the resource of node in rvNode_ in funcNode");
    }
  }
  std::shared_ptr<node> paramNode_() override {
    if (std::shared_ptr<node> sp = this->paramNode.lock()) {
      return sp;
    } else {
      throw std::logic_error("Cannot get the resource of node in paramNode_ in funcNode");
    }
  }

  std::tuple<int, std::shared_ptr<node>> pickArvNode() override {
    std::shared_ptr<node> rvNodeTmp;
    logger(this->name);
    try {
      rvNodeTmp = this->rvNode.lock();
    } catch (std::exception e) {
      ASSERT_FALSE("Cannot get the resource of node in pickArvNode in funcNode");
    }
    logger(std::to_string(rvNodeTmp == nullptr));
    if (rvNodeTmp->isVar() || rvNodeTmp->isUop() || rvNodeTmp->isBop()) {
      return std::make_tuple(-1, rvNodeTmp);
    }
    if (rvNodeTmp->isTuple()) {
      int id = rand() % rvNodeTmp->members_().size();
      return std::make_tuple(id, rvNodeTmp->members_()[id]);
    }
    if (rvNodeTmp->isCall()) {
      return std::make_tuple(-1, rvNodeTmp->resNode_());
   }
    throw std::logic_error(
        "rvNode of a functionNode is not of type tupleNode, "
        "varNode, bopNode or uopNode, but " +
        rvNodeTmp->nodeType_());
  }

  bool hasnoparams_() override { return this->paramNode.expired(); }

  void assign_global(bool gb) override { this->global = gb; }
  bool isGlobal() override { return this->global; }
  int layer_() override { return 0; }
  std::shared_ptr<node> copyNode() override { ASSERT_FALSE("funcNode: copyNode unimplemented"); }
};

class callNode : public wrapperNode {
 private:
  std::weak_ptr<node> funcNode;
  int nid;

 public:
  callNode() {}
  callNode(int ID, int nid, std::shared_ptr<node> resNode = nullptr,
           std::shared_ptr<node> funcNode = nullptr) {
    this->ID = ID;
    this->nid = nid;
    this->nodeType = "call";
    this->name = "call_" + std::to_string(ID);
    this->resNode = resNode;
    if (funcNode != nullptr) this->funcNode = std::weak_ptr<node>(funcNode);
    this->log();
  }
  int nid_() override { return this->nid; }
  void assign_funcNode(std::shared_ptr<node> nd) override { this->funcNode = nd; }
  std::shared_ptr<node> funcNode_() override {
    if (std::shared_ptr<node> sp = funcNode.lock()) {
      return sp;
    } else {
      return nullptr;
    }
  }
  std::shared_ptr<node> copyNode() final {
    return std::make_shared<callNode>(assignID(), this->nid_(), this->resNode_(),
                                      this->funcNode_());
  }
};

#endif