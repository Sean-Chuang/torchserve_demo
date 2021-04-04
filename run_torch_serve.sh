#!/bin/bash
# Java version need to higher than 11
export JAVA_HOME_14=$(/usr/libexec/java_home -v14)


model_name="mnist"
torchserve --stop
torchserve --start \
	--model-store model_store \
	--models ${model_name}=${model_name}.mar \
	--ts-config config.properties
