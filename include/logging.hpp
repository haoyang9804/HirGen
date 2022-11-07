#ifndef LOGGING_HPP_
#define LOGGING_HPP_
#include <stdexcept>
#include <string>

#define ASSERT(STATEMENT, STR) \
  if (!(STATEMENT)) throw std::logic_error(STR)

#define ASSERT_FALSE(STR) throw std::logic_error(STR)

void logger(std::string logMsg);
void stoplogging();

#endif
