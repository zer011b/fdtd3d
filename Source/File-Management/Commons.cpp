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

FileType
GridFileManager::getFileType (const std::string &str)
{
  uint32_t found = str.find_last_of(".");

  std::string type = str.substr (found + 1);

  if (type.compare ("bmp") == 0)
  {
    return FILE_TYPE_BMP;
  }
  else if (type.compare ("dat") == 0)
  {
    return FILE_TYPE_DAT;
  }
  else if (type.compare ("txt") == 0)
  {
    return FILE_TYPE_TXT;
  }

  UNREACHABLE;
}
