#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <string>
#include <vector>

#include "globalType.hpp"

std::vector<int> picAShape();
std::string generateAValue(const DATATYPE& dtype);
std::string generateValues(const SHAPE& shape, int dim, int siz, const DATATYPE& dtype);
DATATYPE rollABoolType();
DATATYPE rollAType();
DATATYPE rollAIntorUIntorFloatType();
DATATYPE rollAFloatType();
DATATYPE rollAIntorUIntType();
DATATYPE rollAIntorUIntorBoolType();
void initSeed();
bool trySelectFromPool();
std::vector<int> reshape(std::vector<int> shape);
#endif