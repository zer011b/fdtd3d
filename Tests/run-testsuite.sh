#!/bin/bash

# To test all tests from test suite:
#   ./Tests/run-testsuite.sh ""

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

set -ex

# 1 to test MPI
TEST_MPI=$1; shift
# 1 to test in CUDA mode
CUDA_MODE=$1; shift

C_COMPILER=$1
CXX_COMPILER=$2

# check exit code
function check_res ()
{
  if [ $? -ne 0 ]; then
    echo "FAILED! ($percent_before%)"
    echo ""
    echo "==== Test suite: ERR ===="
    exit 1
  fi
}

CUR_DIR=`pwd`
SOURCE_DIR=$CUR_DIR

echo "==== Test suite: RUN ===="

test_num=$((1))
test_count=$(ls $CUR_DIR/Tests/suite | wc -l)

echo "Test suite size: $test_count"
echo "========================="
echo ""

for testdir in `ls $CUR_DIR/Tests/suite`; do
  percent_before=$(echo $test_num $test_count | awk '{val=100.0*($1-1)/$2; print val}')

  echo "$test_num. Testing <$testdir> ($percent_before%):"

  $CUR_DIR/Tests/run-test.sh $testdir $TEST_MPI $CUDA_MODE $CUR_DIR/Tests $SOURCE_DIR $C_COMPILER $CXX_COMPILER &> /dev/null
  check_res

  echo "OK!"
  echo ""
  test_num=$((test_num + 1))
done

echo "==== Test suite: OK! ===="
exit 0
