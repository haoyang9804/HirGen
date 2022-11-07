#include <algorithm>
#include <globalVar.hpp>
#include <logging.hpp>
#include <random.hpp>
#include <random>
#include <stringUtils.hpp>

void initSeed() { srand(time(NULL)); }

std::vector<int> picAShape() {
  int rnum = rand() % Bound::shapeUpbound;
  std::vector<int> vec = std::vector<int>();
  while (rnum--) {
    int rrnum = rand();
    vec.push_back(rrnum % Bound::dimValueUpbound + 1);
  }
  return vec;
}

std::string generateAValue(const DATATYPE& dtype) {
  if (dtype == "bool") {
    return rand() % 2 ? "True" : "False";
  } else if (dtype.substr(0, 3) == "int" || dtype.substr(0, 4) == "uint") {
    int num = rand() % Bound::MY_INT_MAX + 1;
    return std::to_string(rand() % 2 == 0 ? num : -num);
  } else if (dtype.substr(0, 5) == "float") {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(0.001, Bound::MY_FLOAT_MAX);
    double num = distr(eng);
    return std::to_string(rand() % 2 == 0 ? num : -num);
  } else {
    throw std::logic_error("random.cpp > generateAValue >> bad dtype: " + dtype);
  }
}

std::string generateValues(const SHAPE& shape, int dim, int siz, const DATATYPE& dtype) {
  if (dim == siz) {
    return generateAValue(dtype);
  }
  std::string res = "[";
  for (int i = 1; i <= shape[dim]; i++) {
    res += generateValues(shape, dim + 1, siz, dtype);
    if (i < shape[dim]) res += ",";
  }
  res += "]";
  return res;
}

DATATYPE rollABoolType() { return "bool"; }

DATATYPE rollAType() {
  std::vector<std::string> type_all{"int64", "uint64", "float64", "int32", "uint32", "float32",
                                    "int16", "uint16",
                                    // "float16",
                                    "int8", "uint8", "bool"};
  std::string selected_type = type_all[rand() % type_all.size()];
  // selected_type = "float32";  // debug
  return selected_type;
}

DATATYPE rollAIntorUIntorFloatType() {
  std::vector<std::string> type_all{"int64", "uint64", "float64", "int32", "uint32", "float32",
                                    "int16", "uint16",
                                    // "float16",
                                    "int8", "uint8"};
  std::string selected_type = type_all[rand() % type_all.size()];
  // selected_type = "float32";  // debug
  return selected_type;
}

DATATYPE rollAFloatType() {
  std::vector<std::string> type_all{
      "float64", "float32",
      // "float16"
  };
  std::string selected_type = type_all[rand() % type_all.size()];
  return selected_type;
}

DATATYPE rollAIntorUIntType() {
  std::vector<std::string> type_all{"int64", "uint64", "int32", "uint32",
                                    "int16", "uint16", "int8",  "uint8"};
  std::string selected_type = type_all[rand() % type_all.size()];
  return selected_type;
}

DATATYPE rollAIntorUIntorBoolType() {
  std::vector<std::string> type_all{"int64",  "uint64", "int32", "uint32", "int16",
                                    "uint16", "int8",   "uint8", "bool"};
  std::string selected_type = type_all[rand() % type_all.size()];
  return selected_type;
}

bool trySelectFromPool() {
  if (rand() % 3 != 0) return true;
  return false;
}

std::vector<int> reshape(std::vector<int> shape) {
  if (shape.size() == 0) return std::vector<int>();
  int product = 1;
  for (int ele : shape) product *= ele;
  std::vector<int> primes;
  while (product % 2 == 0) {
    primes.push_back(2);
    product = product / 2;
  }
  for (int i = 3; i <= product; i = i + 2) {
    while (product % i == 0) {
      primes.push_back(i);
      product = product / i;
    }
  }
  int dim = rand() % Bound::shapeUpbound;
  std::vector<int> newshape;
  int low = 0, high;
  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(primes), std::end(primes), rng);
  for (int i = 1; i < dim - 1; i++) {
    high = low + rand() % (primes.size() - low + 1);
    int pd = 1;
    for (int j = low; j < high; j++) pd *= primes[j];
    newshape.push_back(pd);
    low = high;
  }
  int pd = 1;
  for (int j = low; j < primes.size(); j++) pd *= primes[j];
  newshape.push_back(pd);
  return newshape;
}