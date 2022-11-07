#include <algorithm>
#include <chrono>
#include <globalVar.hpp>
#include <graphNode.hpp>
#include <iostream>
#include <pool.hpp>
#include <random.hpp>
#include <random>
#include <stringUtils.hpp>

bool _equalShape(NODE_PTR lhs, NODE_PTR rhs) {
  if (lhs->shape_().size() != rhs->shape_().size()) return false;
  for (int i = 0; i < lhs->shape_().size(); i++) {
    if (lhs->shape_()[i] != rhs->shape_()[i]) return false;
  }
  return true;
}

#define binaryOpRelayStmt(bopname)                                                                \
  std::string stat;                                                                               \
  if (Custom::cLevel == strict) {                                                                    \
    if (lhs->shape_().empty())                                                                    \
      stat = nd->name_() + " = relay." + #bopname + "(" + lhs->name_() + ".astype(\'" +           \
             nd->dataType_() + "\'), " + rhs->name_() + ".astype(\'" + nd->dataType_() + "\'))";  \
    else if (lhs->shape_compatible(rhs))                                                          \
      stat = nd->name_() + " = relay." + #bopname + "(" + lhs->name_() + ".astype(\'" +           \
             nd->dataType_() + "\'), relay.reshape(" + rhs->name_() + ".astype(\'" +              \
             nd->dataType_() + "\'), relay.shape_of(" + lhs->name_() + ")))";                     \
    else                                                                                          \
      stat = nd->name_() + " = relay." + #bopname + "(" + lhs->name_() + ".astype(\'" +           \
             nd->dataType_() + "\'), " + rhs->name_() + ".astype(\'" + nd->dataType_() + "\'))";  \
    File::ofile << stat << " # shape=" << SHAPE_to_string(nd->shape_(), "(", ")") << std::endl;   \
  } else {                                                                                        \
    stat = nd->name_() + " = relay." + #bopname + "(" + lhs->name_() + ", " + rhs->name_() + ")"; \
    File::ofile << stat << " # shape=" << SHAPE_to_string(nd->shape_(), "(", ")") << std::endl;   \
  }

#define unaryOpRelayStmt(uopname)                                                                 \
  if (Custom::cLevel == strict) {                                                                    \
    std::string stat = nd->name_() + " = relay." + #uopname + "(" + pnd->name_() + ".astype(\'" + \
                       nd->dataType_() + "\'))";                                                  \
    File::ofile << stat << " # shape=" << SHAPE_to_string(nd->shape_(), "(", ")") << std::endl;   \
  } else {                                                                                        \
    std::string stat = nd->name_() + " = relay." + #uopname + "(" + pnd->name_() + ")";           \
    File::ofile << stat << " # shape=" << SHAPE_to_string(nd->shape_(), "(", ")") << std::endl;   \
  }

std::string SHAPE_to_string(SHAPE shape, std::string beginstr, std::string endstr) {
  std::string str = beginstr;
  size_t siz = shape.size();
  for (size_t i = 0; i < siz; i++) {
    if (shape[i] == -1)
      ASSERT_FALSE("shape[i] = -1");
    else
      str += std::to_string(shape[i]);
    if (i != siz - 1) str += ", ";
  }
  if (siz == 1) str += ',';
  str += endstr;
  return str;
}

void generateStmt(std::string stat, NODE_PTR nd) {
  File::ofile << stat << " # shape is " << SHAPE_to_string(nd->shape_(), "(", ")")
              << " # data type is " << nd->dataType_() << std::endl;
}

void node_RelayStmt(NODE_PTR nd) {
  auto info = [nd]() {
    return "#candidate|" + std::to_string(nd->ID_()) + "|" +
           SHAPE_to_string(nd->shape_(), "(", ")") + "|" + nd->nodeType_() + "|" + nd->dataType_();
  };

  if (nd->isVar()) {
    std::string stat = nd->name_() + " = " + "relay.var(\"" + nd->name_() + "\", dtype = \"" +
                       nd->dataType_() + "\", shape = " + SHAPE_to_string(nd->shape_(), "(", ")") +
                       ")";
    File::ofile << stat << info() << std::endl;
  } else if (nd->isConst()) {
    std::string stat = nd->name_() + " = " + "relay.const(" + nd->const_value_() + ", dtype = \"" +
                       nd->dataType_() + "\")";
    File::ofile << stat << info() << std::endl;
  } else {
    throw std::logic_error("stringUtils.cpp -> node_RelayStmt -> nd has unexpected type");
  }
}

void floorDivide_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(floor_divide)
}
void floorMod_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(floor_mod) }
void bitwiseAnd_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(bitwise_and)
}
void bitwiseOr_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(bitwise_or) }
void bitwiseXor_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(bitwise_xor)
}
void notEqual_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(not_equal) }
void rightShift_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(right_shift)
}
void leftShift_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(left_shift) }

void add_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(add) }
void subtract_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(subtract) }
void multiply_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(multiply) }
void divide_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(divide) }
void power_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(power) }
void mod_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(mod) }
void logicalAnd_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(logical_and)
}
void logicalOr_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(logical_or) }
void logicalXor_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(logical_xor)
}
void equal_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(equal) }
void less_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(less) }
void lessEqual_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(less_equal) }
void greater_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(greater) }
void greaterEqual_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) {
  binaryOpRelayStmt(greater_equal)
}
void maximum_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(maximum) }
void minimum_RelayStmt(NODE_PTR nd, NODE_PTR lhs, NODE_PTR rhs) { binaryOpRelayStmt(minimum) }

void log2_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(log2) }
void log10_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(log10) }
void fastExp_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(fast_exp) }
void fastErf_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(fast_erf) }
void rsqrt_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(rsqrt) }
void trunc_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(trunc) }
void fastTanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(fast_tanh) }
void bitwiseNot_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(bitwise_not) }
void zerosLike_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(zeros_like) }
void onesLike_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(ones_like) }
void copy_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(copy) }
void isnan_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(isnan) }
void isfinite_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(isfinite) }
void isinf_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(isinf) }

void log_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(log) }
void tan_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(tan) }
void cos_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(cos) }
void cosh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(cosh) }
void sin_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(sin) }
void sinh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(sinh) }
void acos_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(acos) }
void acosh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(acosh) }
void asin_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(asin) }
void asinh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(asinh) }
void atan_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(atan) }
void atanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(atanh) }
void exp_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(exp) }
void erf_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(erf) }
void sqrt_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(sqrt) }
void sigmoid_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(sigmoid) }
void floor_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(floor) }
void ceil_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(ceil) }
void round_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(round) }
void abs_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(abs) }
void sign_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(sign) }
void tanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(tanh) }
void negative_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(negative) }
void logicalNot_RelayStmt(NODE_PTR nd, NODE_PTR pnd) { unaryOpRelayStmt(logical_not) }

void reshape_RelayStmt(NODE_PTR nd, NODE_PTR pnd) {
  std::string stat = nd->name_() + " = relay.reshape(" + pnd->name_() + ".astype(\'" +
                     nd->dataType_() + "\')," + SHAPE_to_string(nd->shape_(), "[", "]") + ")";
  File::ofile << stat << " # shape=" << SHAPE_to_string(nd->shape_(), "(", ")") << std::endl;
}

void header_RelayStmt() {
  File::ofile << "import tvm" << std::endl;
  File::ofile << "from tvm import relay" << std::endl;
  File::ofile << "from tvm.ir.transform import Sequential" << std::endl;
  File::ofile << "from tvm.contrib import graph_runtime" << std::endl;
  File::ofile << "import numpy as np" << std::endl;

  // add a python function(this is a workaroud)
  if (Custom::feature == "df") {
    std::string vmobj_to_list = "";
    vmobj_to_list += "def vmobj_to_list(o, dtype=\"float32\"):\n";
    vmobj_to_list += "    if isinstance(o, tvm.nd.NDArray):\n";
    vmobj_to_list += "        return [o]\n";
    vmobj_to_list += "    elif isinstance(o, tvm.runtime.container.ADT):\n";
    vmobj_to_list += "        result = []\n";
    vmobj_to_list += "        for f in o:\n";
    vmobj_to_list += "            result.extend(vmobj_to_list(f, dtype))\n";
    vmobj_to_list += "        return result\n";
    vmobj_to_list += "    else:\n";
    vmobj_to_list += "        return o\n";
    File::ofile << vmobj_to_list << std::endl;
    File::ofile << std::endl;
  }

  File::ofile << "mod = tvm.IRModule()" << std::endl;
  File::ofile << "mutated_mod = tvm.IRModule()" << std::endl;
}

void tail_mod_RelayStmt(std::string inputnames_str) {
  function_RelayStmt("F", inputnames_str, "output");
  File::ofile << "mod[\'main\'] = F" << std::endl;
  File::ofile << "mod = relay.transform.InferType()(mod)" << std::endl;
  File::ofile << "print(\'==========mod==========\')" << std::endl;
  File::ofile << "print(mod.astext(show_meta_data=False))" << std::endl;
  File::ofile << "print(\'===================================\')" << std::endl;

  function_RelayStmt("F", inputnames_str, "output2");

  File::ofile << "mutated_mod[\'main\'] = F" << std::endl;
  File::ofile << "mutated_mod = relay.transform.InferType()(mutated_mod)" << std::endl;
  File::ofile << "print(\'==========mutated_mod==========\')" << std::endl;
  File::ofile << "print(mutated_mod.astext(show_meta_data=False))" << std::endl;
  File::ofile << "print(\'===================================\')" << std::endl;
}

void tail_runtime_stmt(std::string modname, int id0, int id1, int id2, int id3,
                       std::string target) {
  File::ofile << "graph, lib, params = relay.build(" << modname << ", target=\'" << target << "\')"
              << std::endl;
  File::ofile << "module" << id0 << " = graph_runtime.create(graph, lib, tvm.device(\'" << target
              << "\',0))" << std::endl;

  File::ofile << "intrp" << id1 << " = relay.build_module.create_executor(\'graph\', " << modname
              << ", tvm.device(\'" << target << "\',0),\'" << target << "\')" << std::endl;
  File::ofile << "intrp" << id2 << " = relay.build_module.create_executor(\'debug\', " << modname
              << ", tvm.device(\'" << target << "\',0),\'" << target << "\')" << std::endl;
  File::ofile << "intrp" << id3 << " = relay.build_module.create_executor(\'vm\', " << modname
              << ", tvm.device(\'" << target << "\',0),\'" << target << "\')" << std::endl;
}

void tail_predicateAndOutput_stmt(int id0, int id1, int id2, int id3) {
  size_t outputs_size = OutputNode::size();
  for (size_t i = 0; i < outputs_size; i++) {
    File::ofile << "res" << id0 << "_" << std::to_string(i) << " = module" << id0 << ".get_output("
                << std::to_string(i) << ").asnumpy()" << std::endl;
    File::ofile << "res" << id1 << "_" << std::to_string(i) << " = res" << id1 << "["
                << std::to_string(i) << "].asnumpy()" << std::endl;
    File::ofile << "res" << id2 << "_" << std::to_string(i) << " = res" << id2 << "["
                << std::to_string(i) << "].asnumpy()" << std::endl;
    File::ofile << "res" << id3 << "_" << std::to_string(i) << " = res" << id3 << "["
                << std::to_string(i) << "].asnumpy()" << std::endl;

    File::ofile << "np.testing.assert_allclose(res" << id0 << "_" << std::to_string(i) << " ,res"
                << id1 << "_" << std::to_string(i) << ", atol=1e-3, rtol=1e-3)" << std::endl;
    File::ofile << "np.testing.assert_allclose(res" << id0 << "_" << std::to_string(i) << " ,res"
                << id2 << "_" << std::to_string(i) << ", atol=1e-3, rtol=1e-3)" << std::endl;
    File::ofile << "np.testing.assert_allclose(res" << id0 << "_" << std::to_string(i) << " ,res"
                << id3 << "_" << std::to_string(i) << ", atol=1e-3, rtol=1e-3)" << std::endl;
    File::ofile << "(res" << id0 << "_" << std::to_string(i) << " == res" << id1 << "_"
                << std::to_string(i) << ").all()" << std::endl;
    File::ofile << "(res" << id0 << "_" << std::to_string(i) << " == res" << id2 << "_"
                << std::to_string(i) << ").all()" << std::endl;
    File::ofile << "(res" << id0 << "_" << std::to_string(i) << " == res" << id3 << "_"
                << std::to_string(i) << ").all()" << std::endl;
  }
}

void tail_optimizations_RelayStmt() {
  std::vector<std::string> candidates = {"AlterOpLayout()",
                                         "AnnotateSpans()",
                                         "BatchingOps()",
                                         "CanonicalizeCast()",
                                         "CanonicalizeOps()",
                                         "DeadCodeElimination()",
                                         "DynamicToStatic()",
                                         "FastMath()",
                                         "FirstOrderGradient()",
                                         "EliminateCommonSubexpr()",
                                         "MergeCompilerRegions()",
                                         "Inline()",
                                         "LambdaLift()",
                                         "LazyGradientInit()",
                                         "PartialEvaluate()",
                                         "Legalize()",
                                         "FoldConstant()",
                                         "ToANormalForm()",
                                         "ToGraphNormalForm()",
                                         "SimplifyInference()",
                                         "ToBasicBlockNormalForm()",
                                         "FuseOps(3)",
                                         "DefuseOps()",
                                         "SimplifyExpr()",
                                         "InferType()"};
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine e(seed);
  std::shuffle(candidates.begin(), candidates.begin(), e);

  size_t candidates_num = candidates.size();
  int required_num = rand() % candidates_num + 1;
  File::ofile << "seq = Sequential([" << std::endl;
  for (int i = 0, j = 1; j <= required_num; i++, j++) {
    File::ofile << "\trelay.transform." + candidates[i] << "," << std::endl;
  }
  File::ofile << "])" << std::endl;
  File::ofile << "mod = seq(mod)" << std::endl;
  File::ofile << "print(mod.astext(show_meta_data=False))" << std::endl;
}

void tail_createOutputAndLoad_stmt(int id0, int id1, int id2, int id3, bool inputYes) {
  std::string inputnames_str = "";
  // load inputs
  for (NODE_PTR input : InputNode::mirror()) {
    NODE_PTR input_t = NodePool::findByName(input->name_());
    std::vector<int>&& input_shape = input_t->shape_();
    if (inputYes) {
      std::string&& value = generateValues(
          input_shape, 0, static_cast<int>(input_t->shape_().size()), input_t->dataType_());
      File::ofile << "input_" << input_t->ID_() << "= np.array(" << value << ", dtype='"
                  << input_t->dataType_() << "')" << std::endl;
    }
    File::ofile << "module" << id0 << ".set_input(\'" << input_t->name_() << "\', input_"
                << input_t->ID_() << ")" << std::endl;
    inputnames_str += "input_" + std::to_string(input_t->ID_()) + ", ";
  }

  File::ofile << "module" << id0 << ".set_input(**params)" << std::endl;
  File::ofile << "module" << id0 << ".run()" << std::endl;
  // create_executor : load inputs
  File::ofile << "res" << id1 << " = intrp" << id1 << ".evaluate()(" << inputnames_str << ")"
              << std::endl;
  File::ofile << "res" << id2 << " = intrp" << id2 << ".evaluate()(" << inputnames_str << ")"
              << std::endl;
  File::ofile << "res" << id3 << " = intrp" << id3 << ".evaluate()(" << inputnames_str << ")"
              << std::endl;

  File::ofile << "res" << id1 << " = vmobj_to_list(res" << id1 << ")" << std::endl;
  File::ofile << "res" << id2 << " = vmobj_to_list(res" << id2 << ")" << std::endl;
  File::ofile << "res" << id3 << " = vmobj_to_list(res" << id3 << ")" << std::endl;
}

void function_RelayStmt(std::string funcName, std::string inputNamesStr, std::string outputstr) {
  File::ofile << funcName + " = relay.Function([" + inputNamesStr + "], " << outputstr << ")"
              << std::endl;
}

std::vector<std::string> splitString(std::string str, std::string delimeter) {
  std::vector<std::string> splittedStrings = {};
  size_t pos = 0;

  while ((pos = str.find(delimeter)) != std::string::npos) {
    std::string token = str.substr(0, pos);
    if (token.length() > 0) splittedStrings.push_back(token);
    str.erase(0, pos + delimeter.length());
  }

  if (str.length() > 0) splittedStrings.push_back(str);
  return splittedStrings;
}

inline void _generateTail_build_and_predict(std::string modname, int id0, int id1, int id2, int id3,
                                            std::string device, bool inputYes = true) {
  tail_createOutputAndLoad_stmt(id0, id1, id2, id3, inputYes);
  tail_predicateAndOutput_stmt(id0, id1, id2, id3);
}

void _generateBuild() {
  tail_runtime_stmt("mod", 1, 2, 3, 4, "llvm");
  tail_runtime_stmt("mod", 5, 6, 7, 8, "cuda");
}

void _generateBuildForOpt() {
  tail_runtime_stmt("mod", 9, 10, 11, 12, "llvm");
  tail_runtime_stmt("mod", 13, 14, 15, 16, "cuda");
}

void _generateBuildForMutant() {
  tail_runtime_stmt("mutated_mod", 17, 18, 19, 20, "llvm");
  tail_runtime_stmt("mutated_mod", 21, 22, 23, 24, "cuda");
}

void generateTail() {
  if (Custom::runtimeMode == "debug") {
    File::ofile << "# delete until here" << std::endl;
  }
  auto [inputnames_str, _] = InputNode::getInputs();
  tail_mod_RelayStmt(inputnames_str);
  _generateBuild();
  tail_optimizations_RelayStmt();
  _generateBuildForOpt();
  _generateBuildForMutant();
}

void generateInputsAndPredictions() {
  _generateTail_build_and_predict("mod", 1, 2, 3, 4, "llvm");
  _generateTail_build_and_predict("mod", 5, 6, 7, 8, "cuda", false);

  _generateTail_build_and_predict("mod", 9, 10, 11, 12, "llvm", false);
  _generateTail_build_and_predict("mod", 13, 14, 15, 16, "cuda", false);

  _generateTail_build_and_predict("mutated_mod", 17, 18, 19, 20, "llvm", false);
  _generateTail_build_and_predict("mutated_mod", 21, 22, 23, 24, "cuda", false);
}

std::pair<std::string, std::string> split(std::string str, std::string delimiter) {
  if (!(str.empty())) {
    auto pos = str.find(delimiter);
    ASSERT(pos != std::string::npos, "split fails. No such delimiter " + delimiter);
    std::string first = str.substr(0, pos);
    str.erase(0, pos + delimiter.size());
    std::string second = str;
    return std::make_pair(first, second);
  }
  return std::make_pair("", "");
}