#!/usr/bin/env sh
set -e

BUILD=build/examples/spell_check
CURRENT_DIR=$(dirname `readlink -f $0`)

$BUILD/QueryPrediction.bin ${CURRENT_DIR}/module ${CURRENT_DIR}/test_input
