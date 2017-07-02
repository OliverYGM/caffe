#!/usr/bin/env sh
set -e

BUILD=build/examples/spell_check

mkdir -p ${BUILD}/module
wget "http://d2-test.oss-cn-hangzhou-zmf.aliyuncs.com/for_intel/caffe_module.prototxt" -O ${BUILD}/module/caffe_module.prototxt
wget "http://d2-test.oss-cn-hangzhou-zmf.aliyuncs.com/for_intel/caffe_module.net_parameter" -O ${BUILD}/module/caffe_module.net_parameter
wget "http://d2-test.oss-cn-hangzhou-zmf.aliyuncs.com/for_intel/test_input" -O ${BUILD}/test_input
