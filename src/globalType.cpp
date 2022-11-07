#include <globalType.hpp>
#include <stringUtils.hpp>

bool operator<(const SHAPE_STR_SQUARE& s1, const SHAPE_STR_SQUARE& s2) {
  return s1.str() < s2.str();
}