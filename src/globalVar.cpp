#include <globalVar.hpp>

namespace Value {
int global_id = 0;
bool firstNode = true;
int opNum = 0;
}  // namespace Value

namespace Bound {
int maxlayer = 1;
const int shapeUpbound = 4;
const int dimValueUpbound = 16;
const int MY_FLOAT_MAX = 10;
const int MY_INT_MAX = 10;
const int valueconstBound = 4;
}  // namespace Bound

namespace File {
const char* pythonFilePath = "output.py";
std::ifstream ifile(pythonFilePath, std::ios::in);
std::ofstream ofile(pythonFilePath, std::ios::app);
const char* pythonFileCopyPath = "output_copy.py";
const char* loggingFilePath = "log.txt";
std::ofstream ofs(loggingFilePath, std::ios_base::out);
const char* csvPath = "cov.csv";
std::ifstream icsv(csvPath, std::ios::in);
std::ofstream ocsv(csvPath, std::ios::app);
}  // namespace File

namespace Custom {
int nodeNumUpBound = 100;
std::string runtimeMode = "release";
std::string feature = "nodf";
checking_level cLevel = strict;
bool coverage = true;
}  // namespace Custom

namespace Accessory {
std::unordered_map<std::string, std::string> name2mname;
}