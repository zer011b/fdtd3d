#!/bin/bash

# To test all tests from test suite:
#   ./Tests/run-testsuite.sh ""

set -ex

# 0 for native tests, 1 for cuda tests
MODE=$1
SCRIPT="run-test.sh"
if [ "$MODE" == "1" ]; then
  SCRIPT="run-test-gpu.sh"
fi

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

  $CUR_DIR/Tests/$SCRIPT $testdir $CUR_DIR/Tests $SOURCE_DIR &> /dev/null
  check_res

  echo "OK!"
  echo ""
  test_num=$((test_num + 1))
done

echo "==== Test suite: OK! ===="
exit 0
