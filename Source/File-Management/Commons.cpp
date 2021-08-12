/*
 * Copyright (C) 2015 Gleb Balykov
 *
 * This file is part of fdtd3d.
 *
 * fdtd3d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * fdtd3d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with fdtd3d; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

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
  snprintf(buffer, sizeof(buffer), "%" PRId64, value);
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
