#ifndef STRINGUTILS_HPP_
#define STRINGUTILS_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "constraints.hpp"
#include "node.hpp"

#define generateBinaryRelayStmtByType_if(type)          \
  if (boptype == #type) {                               \
    type##_RelayStmt(bopnd, lhs, rhs);                  \
    if (poison) {                                       \
      type##_RelayStmt(bopnd_copy, lhs_copy, rhs_copy); \
    }                                                   \
  }

#define generateBinaryRelayStmtByType_elseif(type)      \
  else if (boptype == #type) {                          \
    type##_RelayStmt(bopnd, lhs, rhs);                  \
    if (poison) {                                       \
      type##_RelayStmt(bopnd_copy, lhs_copy, rhs_copy); \
    }                                                   \
  }

#define generateBinaryRelayStmt                                                                    \
  generateBinaryRelayStmtByType_if(add) generateBinaryRelayStmtByType_elseif(subtract)             \
      generateBinaryRelayStmtByType_elseif(multiply) generateBinaryRelayStmtByType_elseif(divide)  \
          generateBinaryRelayStmtByType_elseif(power) generateBinaryRelayStmtByType_elseif(mod)    \
              generateBinaryRelayStmtByType_elseif(floorMod) generateBinaryRelayStmtByType_elseif( \
                  floorDivide) generateBinaryRelayStmtByType_elseif(logicalAnd)                    \
                  generateBinaryRelayStmtByType_elseif(                                            \
                      logicalOr) generateBinaryRelayStmtByType_elseif(logicalXor)                  \
                      generateBinaryRelayStmtByType_elseif(                                        \
                          bitwiseAnd) generateBinaryRelayStmtByType_elseif(bitwiseOr)              \
                          generateBinaryRelayStmtByType_elseif(                                    \
                              bitwiseXor) generateBinaryRelayStmtByType_elseif(equal)              \
                              generateBinaryRelayStmtByType_elseif(                                \
                                  notEqual) generateBinaryRelayStmtByType_elseif(less)             \
                                  generateBinaryRelayStmtByType_elseif(                            \
                                      lessEqual) generateBinaryRelayStmtByType_elseif(greater)     \
                                      generateBinaryRelayStmtByType_elseif(greaterEqual)           \
                                          generateBinaryRelayStmtByType_elseif(maximum)            \
                                              generateBinaryRelayStmtByType_elseif(minimum)        \
                                                  generateBinaryRelayStmtByType_elseif(leftShift)  \
                                                      generateBinaryRelayStmtByType_elseif(        \
                                                          rightShift)

#define generateUnaryRelayStmtByType_if(type)  \
  if (uoptype == #type) {                      \
    type##_RelayStmt(uopnd, vcnd);             \
    if (poison) {                              \
      type##_RelayStmt(uopnd_copy, vcnd_copy); \
    }                                          \
  }

#define generateUnaryRelayStmtByType_elseif(type) \
  else if (uoptype == #type) {                    \
    type##_RelayStmt(uopnd, vcnd);                \
    if (poison) {                                 \
      type##_RelayStmt(uopnd_copy, vcnd_copy);    \
    }                                             \
  }

#define generateUnaryRelayStmt                                                                         \
  generateUnaryRelayStmtByType_if(log) generateUnaryRelayStmtByType_elseif(                            \
      log2) generateUnaryRelayStmtByType_elseif(log10) generateUnaryRelayStmtByType_elseif(tan)        \
      generateUnaryRelayStmtByType_elseif(cos) generateUnaryRelayStmtByType_elseif(                    \
          cosh) generateUnaryRelayStmtByType_elseif(sin) generateUnaryRelayStmtByType_elseif(sinh)     \
          generateUnaryRelayStmtByType_elseif(acos) generateUnaryRelayStmtByType_elseif(               \
              acosh) generateUnaryRelayStmtByType_elseif(asin)                                         \
              generateUnaryRelayStmtByType_elseif(asinh) generateUnaryRelayStmtByType_elseif(          \
                  atan) generateUnaryRelayStmtByType_elseif(atanh)                                     \
                  generateUnaryRelayStmtByType_elseif(exp) generateUnaryRelayStmtByType_elseif(        \
                      erf) generateUnaryRelayStmtByType_elseif(sqrt)                                   \
                      generateUnaryRelayStmtByType_elseif(rsqrt) generateUnaryRelayStmtByType_elseif(  \
                          sigmoid) generateUnaryRelayStmtByType_elseif(floor)                          \
                          generateUnaryRelayStmtByType_elseif(                                         \
                              ceil) generateUnaryRelayStmtByType_elseif(trunc)                         \
                              generateUnaryRelayStmtByType_elseif(                                     \
                                  round) generateUnaryRelayStmtByType_elseif(abs)                      \
                                  generateUnaryRelayStmtByType_elseif(                                 \
                                      sign) generateUnaryRelayStmtByType_elseif(tanh)                  \
                                      generateUnaryRelayStmtByType_elseif(                             \
                                          negative) generateUnaryRelayStmtByType_elseif(logicalNot)    \
                                          generateUnaryRelayStmtByType_elseif(bitwiseNot)              \
                                              generateUnaryRelayStmtByType_elseif(zerosLike)           \
                                                  generateUnaryRelayStmtByType_elseif(onesLike)        \
                                                      generateUnaryRelayStmtByType_elseif(copy)        \
                                                          generateUnaryRelayStmtByType_elseif(         \
                                                              isnan)                                   \
                                                              generateUnaryRelayStmtByType_elseif(     \
                                                                  isfinite)                            \
                                                                  generateUnaryRelayStmtByType_elseif( \
                                                                      isinf)

void node_RelayStmt(NODE_PTR var);
// void function_RelayStmt(NODE_PTR body, NODE_PTR var);

void add_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void subtract_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void multiply_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void divide_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void power_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void mod_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void logicalAnd_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void logicalOr_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void logicalXor_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void equal_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void less_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void lessEqual_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void greater_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void greaterEqual_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void maximum_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void minimum_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void floorDivide_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void floorMod_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void bitwiseAnd_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void bitwiseOr_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void bitwiseXor_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void notEqual_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void rightShift_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);
void leftShift_RelayStmt(NODE_PTR var, NODE_PTR lhs, NODE_PTR rhs);

void log2_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void log10_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void fastExp_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void fastErf_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void rsqrt_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void trunc_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void fastTanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void bitwiseNot_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void zerosLike_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void onesLike_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void copy_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void isnan_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void isfinite_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void isinf_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void log_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void tan_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void cos_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void cosh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void sin_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void sinh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void acos_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void acosh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void asin_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void asinh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void atan_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void atanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void exp_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void erf_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void sqrt_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void sigmoid_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void floor_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void ceil_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void round_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void abs_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void sign_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void tanh_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void negative_RelayStmt(NODE_PTR nd, NODE_PTR pnd);
void logicalNot_RelayStmt(NODE_PTR nd, NODE_PTR pnd);

void reshape_RelayStmt(NODE_PTR nd, NODE_PTR pnd);

void function_RelayStmt(std::string funcName, std::string inputNamesStr, std::string outputstr);

void header_RelayStmt();

void tail_mod_RelayStmt(std::string inputnames_str);
void tail_runtime_stmt(std::string modname, int id0, int id1, int id2, int id3,
                       std::string target = "llvm");

void tail_predicateAndOutput_stmt(int id0, int id1, int id2, int id3);
void tail_optimizations_RelayStmt();
void tail_createOutputAndLoad_stmt(int id0, int id1, int id2, int id3, bool inputYes);
std::string SHAPE_to_string(SHAPE shape, std::string beginstr, std::string endstr);

std::vector<std::string> splitString(std::string str, std::string delimeter);

void generateTail();

void generateInputsAndPredictions();

std::pair<std::string, std::string> split(std::string str, std::string delimiter);
#endif
