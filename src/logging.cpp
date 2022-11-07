#include <globalVar.hpp>
#include <logging.hpp>

std::string getCurrentDateTime(std::string s) {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  if (s == "now")
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
  else if (s == "date")
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
  return std::string(buf);
};
void logger(std::string logMsg) {
  std::string now = getCurrentDateTime("now");
  using File::ofs;
  if (ofs.is_open()) ofs << now << "\t" << logMsg << std::endl;
}

void stoplogging() { File::ofs.close(); }