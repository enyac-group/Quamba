#pragma once

#include <string>

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}
