#!/usr/bin/env sh
set -e

BUILD=build/examples/spell_check

$BUILD/QueryPrediction.bin ${BUILD}/module ${BUILD}/test_input
