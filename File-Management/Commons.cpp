#include "Commons.h"

#ifdef CXX11_ENABLED
std::string int64_to_string(int64_t value)
{
  return std::to_string (value);
}
#else
std::string int64_to_string(int64_t value)
{
  char buffer[65];
  snprintf(buffer, sizeof(buffer), "%ld", value);
  return std::string (buffer);
}
#endif
