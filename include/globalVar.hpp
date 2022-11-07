#ifndef GRAPHVAR_HPP_
#define GRAPHVAR_HPP_

#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace Value {
extern int global_id;
extern bool firstNode;
extern int opNum;
}  // namespace Value

inline int assignID() { return Value::global_id++; }

namespace Bound {
extern int maxlayer;
extern const int shapeUpbound;
extern const int dimValueUpbound;
extern const int MY_FLOAT_MAX;
extern const int MY_INT_MAX;
extern const int valueconstBound;
}  // namespace Bound

inline void init_maxlayer() { Bound::maxlayer = 1; }

namespace File {
extern const char* pythonFilePath;
extern std::ifstream ifile;
extern std::ofstream ofile;
extern const char* pythonFileCopyPath;
extern const char* loggingFilePath;
extern std::ofstream ofs;
extern const char* csvPath;
extern std::ifstream icsv;
extern std::ofstream ocsv;
}  // namespace File

enum checking_level{
  relaxed,
  strict
};

namespace Custom {
extern int nodeNumUpBound;
extern std::string runtimeMode;
extern std::string feature;
extern checking_level cLevel;
extern bool coverage;
}  // namespace Custom

namespace Accessory {
extern std::unordered_map<std::string, std::string> name2mname;
}

#endif