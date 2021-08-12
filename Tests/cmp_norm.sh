#!/bin/bash

#
#  Copyright (C) 2017 Gleb Balykov
#
#  This file is part of fdtd3d.
#
#  fdtd3d is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  fdtd3d is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with fdtd3d; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

#set -ex

input_cpu="$1"
input_gpu="$2"

norm_re="0.0"
norm_im="0.0"
norm="0.0"

max_re="0.0"
max_im="0.0"
max_mod="0.0"

function norm_diff()
{
  local val_prev=$1
  local first=$2
  local second=$3

  local val_tmp=$(echo $val_prev $first $second | awk '{val=$1 + ($2-$3)*($2-$3); printf "%.20f", val}')

  echo $val_tmp
}

function norm_mod()
{
  local re=$1
  local im=$2

  local val_tmp=$(echo $re $im | awk '{printf "%.20f", sqrt($1*$1+$2*$2)}')

  echo $val_tmp
}

function norm_normalized()
{
  local norm=$1
  local count=$2

  local val_tmp=$(echo $norm $count| awk '{printf "%.20f", sqrt($1*1.0/$2)}')

  echo $val_tmp
}

function norm_normalized_percent()
{
  local norm=$1
  local max=$2

  local val_tmp=$(echo $norm $max| awk '{printf "%.20f", $1*100.0/$2}')

  echo $val_tmp
}

function update_max()
{
  local max_prev=$1
  local val_to_check=$2

  val_to_check=$(echo $val_to_check | awk '{if ($1 < 0) {printf "%.20f", -$1} else {printf "%.20f", $1} }')

  local is_less=$(echo $max_prev $val_to_check | awk '{if ($1 < $2) {print 1;} else {print 0;}}')
  if [[ "$is_less" -eq "1" ]]; then
    max_prev=$(echo $val_to_check)
  fi

  echo $max_prev
}

while IFS= read -r line; do
  coord=$(echo $line | awk '{print $1}')
  re=$(echo $line | awk '{print $2}')
  im=$(echo $line | awk '{print $3}')
  mod=$(norm_mod $re $im)

  line_gpu=$(grep -e "^$coord " $input_gpu)
  re_gpu=$(echo $line_gpu | awk '{print $2}')
  im_gpu=$(echo $line_gpu | awk '{print $3}')
  mod_gpu=$(norm_mod $re_gpu $im_gpu)

  max_re=$(update_max $max_re $re)
  max_im=$(update_max $max_im $im)
  max_mod=$(update_max $max_mod $mod)

  norm_re=$(norm_diff $norm_re $re $re_gpu)
  norm_im=$(norm_diff $norm_im $im $im_gpu)
  norm=$(norm_diff $norm $mod $mod_gpu)
done < "$input_cpu"

count=$(echo $coord | awk '{print $1+1}')

norm_re=$(norm_normalized $norm_re $count)
norm_re_percent=$(norm_normalized_percent $norm_re $max_re)

norm_im=$(norm_normalized $norm_im $count)
norm_im_percent=$(norm_normalized_percent $norm_im $max_im)

norm=$(norm_normalized $norm $count)
norm_percent=$(norm_normalized_percent $norm $max_mod)

echo "Re: $norm_re of $max_re ($norm_re_percent %), $count"
echo "Im: $norm_im of $max_im ($norm_im_percent %), $count"
echo "Mod: $norm of $max_mod ($norm_percent %), $count"
