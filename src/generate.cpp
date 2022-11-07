#include <algorithm>
#include <coverage.hpp>
#include <functional>
#include <generator.hpp>
#include <globalVar.hpp>
#include <iostream>
#include <logging.hpp>
#include <pythonFile.hpp>
#include <random.hpp>
#include <string>
#include <stringUtils.hpp>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  ASSERT(argc <= 6, "More than 5 arguments, illegal!");
  std::string argv_strs[argc];
  for (int i = 1; i < argc; i++) {
    argv_strs[i] = std::string(argv[i]);
    argv_strs[i].erase(std::remove(argv_strs[i].begin(), argv_strs[i].end(), ' '),
                       argv_strs[i].end());
    auto [paramName, paramBody] = split(argv_strs[i], "=");
    if (paramName == "-num") {
      char* strpart;
      Custom::nodeNumUpBound = strtol(paramBody.c_str(), &strpart, 10);
      ASSERT(strpart[0] == '\0', "Incorrect node num, not integer!");
    }
    else if (paramName == "-rMode") {
      Custom::runtimeMode = paramBody;
    }
    else if (paramName == "-testing") {
      Custom::feature = paramBody;
    }
    else if (paramName == "-clevel") {
      if (paramBody == "strict") Custom::cLevel = strict;
      else if (paramBody == "disruptive") Custom::cLevel = relaxed;
      else ASSERT_FALSE("Illegal paramBody for -clevel: " + paramBody);
    }
    else if (paramName == "-coverage") {
      if (paramBody == "yes") Custom::coverage = true;
      else if (paramBody == "no") Custom::coverage = false;
      else ASSERT_FALSE("Illegal paramBody for -coverage: " + paramBody);
    }
    else {
      ASSERT_FALSE("Unrecognizable parameter: " + paramName);
    }
  }
  ASSERT(!(Custom::feature != "nodf" && Custom::feature != "df"),
         "Invalid Custom::feature: " + Custom::feature);
  ASSERT(!(Custom::runtimeMode != "release" && Custom::runtimeMode != "debug"),
         "Invalid Custom::runtimeMode: " + Custom::runtimeMode);
  initPythonFile();
  initSeed();
  header_RelayStmt();
  Coverage::load();
  std::vector<IGenerator*> generators = {
      new uopGenerator("log", rollAFloatType),
      new uopGenerator("log2", rollAFloatType),
      new uopGenerator("log10", rollAFloatType),
      new uopGenerator("tan", rollAFloatType),
      new uopGenerator("cos", rollAFloatType),
      new uopGenerator("cosh", rollAFloatType),
      new uopGenerator("sin", rollAFloatType),
      new uopGenerator("sinh", rollAFloatType),
      new uopGenerator("acos", rollAFloatType),
      new uopGenerator("acosh", rollAFloatType),
      new uopGenerator("asin", rollAFloatType),
      new uopGenerator("asinh", rollAFloatType),
      new uopGenerator("atan", rollAFloatType),
      new uopGenerator("atanh", rollAFloatType),
      new uopGenerator("exp", rollAFloatType),
      new uopGenerator("erf", rollAFloatType),
      new uopGenerator("sqrt", rollAFloatType),
      new uopGenerator("rsqrt", rollAFloatType),
      new uopGenerator("sigmoid", rollAFloatType),
      new bopGenerator("add", rollAIntorUIntorFloatType),
      new bopGenerator("subtract", rollAIntorUIntorFloatType),
      new bopGenerator("multiply", rollAIntorUIntorFloatType),
      new bopGenerator("divide", rollAFloatType),
      new bopGenerator("power", rollAFloatType),
      new bopGenerator("mod", rollAFloatType),
      new bopGenerator("floorMod", rollAFloatType),
      new bopGenerator("floorDivide", rollAFloatType),
      new bopGenerator("logicalAnd", rollABoolType),
      new bopGenerator("logicalOr", rollABoolType),
      new bopGenerator("logicalXor", rollAIntorUIntType),
      new bopGenerator("bitwiseAnd", rollAIntorUIntType),
      new bopGenerator("bitwiseOr", rollAIntorUIntType),
      new bopGenerator("bitwiseXor", rollAIntorUIntType),
      new bopGenerator("equal", rollAIntorUIntorFloatType),
      new bopGenerator("notEqual", rollAIntorUIntorFloatType),
      new bopGenerator("less", rollAIntorUIntorFloatType),
      new bopGenerator("lessEqual", rollAIntorUIntorFloatType),
      new bopGenerator("greater", rollAIntorUIntorFloatType),
      new bopGenerator("greaterEqual", rollAIntorUIntorFloatType),
      new bopGenerator("maximum", rollAIntorUIntorFloatType),
      new bopGenerator("minimum", rollAIntorUIntorFloatType),
      new bopGenerator("rightShift", rollAIntorUIntType),
      new bopGenerator("leftShift", rollAIntorUIntType),

      new funcGenerator(),
      new funcGenerator(),
      new funcGenerator(),
      new callGenerator(),
      new callGenerator(),
      new callGenerator()};

  int opnum = 0;
  auto generateOp = [&opnum, &generators]() {
    int funcID = rand() % generators.size();
    if (funcID >= generators.size() - 6) opnum -= 1;
    generators[funcID]->generate();
  };

  while (true) {
    opnum += 1;
    if (Value::opNum >= Custom::nodeNumUpBound) break;
    generateOp();
  }
  logger("Coverage Score: " + std::to_string(Coverage::get_score()));

  /**
   * @brief check ID order - Haoyang
   */
  // int preid = -1;
  // for (auto ele : OutputNode::mirror()) {
  //   // std::cout << ele->ID_() << std::endl;
  //   if (ele->ID_() <= preid) {
  //     std::cout << "WRONG" << std::endl;
  //   }
  //   preid = ele->ID_();
  // }
  /*Finish check*/

  if (Custom::runtimeMode == "release") {
    tailStmt stoplogging();
    Coverage::save();
    File::ifile.close();
    File::ofile.close();
    File::icsv.close();
    File::ocsv.close();
  }
  std::cout << "\033[1;31mSUCCESS!\033[0m" << std::endl;
}
