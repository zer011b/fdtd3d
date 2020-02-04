#include "Commons.h"

/**
 * Convert int64 to string
 */
#ifdef CXX11_ENABLED
std::string int64_to_string(int64_t value) /**< value */
{
  return std::to_string (value);
} /* int64_to_string */
#else
std::string int64_to_string(int64_t value) /**< value */
{
  char buffer[65];
  snprintf(buffer, sizeof(buffer), "%ld", value);
  return std::string (buffer);
} /* int64_to_string */
#endif

/**
 * Get type of file from its name
 *
 * @return type of file
 */
FileType
GridFileManager::getFileType (const std::string &str) /**< file name */
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
  return FILE_TYPE_COUNT;
} /* GridFileManager::getFileType */
