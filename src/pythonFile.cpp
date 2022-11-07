#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <globalVar.hpp>
#include <logging.hpp>
#include <pythonFile.hpp>
#include <string>

bool runPythonFile(const char* filename) {
  char exe[30] = "python ";
  strcat(exe, filename);
  int res = system(exe);
  if (res != 0) {
    return false;
  }
  return true;
}

void initPythonFile() {
  std::ofstream of(File::pythonFilePath, std::ios::out);
  of << "";
  of.clear();
}

void deletetheLastLineInConsole() {
  for (int i = 1; i <= 10; i++)
    std::cout << "\x1b[1A"   // Move cursor up one
              << "\x1b[2K";  // Delete the entire line
}

void re_freopen() { freopen(File::pythonFilePath, "a+", stdout); }

void backToScreen() {  // only support Linux
  freopen("/dev/tty", "a", stdout);
}

void deleteTailInFile() {
  std::ifstream in(File::pythonFilePath);
  std::ofstream out(File::pythonFileCopyPath);
  // some error checking...
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("# delete until here") != std::string::npos) {
      break;
    }

    out << line << std::endl;
  }
  in.close();
  out.close();
  using namespace File;
std:
  rename(pythonFileCopyPath, pythonFilePath);
  File::ofile.close();
  File::ofile.open(pythonFilePath, std::ios::app);
  File::ifile.close();
  File::ifile.open(pythonFilePath, std::ios::in);
}

bool runPythonFileAfterEachRound(const char* filename) {
  bool res = runPythonFile(filename);

  if (Value::opNum < Custom::nodeNumUpBound) {
    deleteTailInFile();
  }

  return res;
}