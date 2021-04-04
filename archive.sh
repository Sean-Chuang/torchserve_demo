#!/bin/bash
model_name="mnist"
torch-model-archiver \
	-f \
	--model-name $model_name \
	--version 1.0 \
	--model-file model/model.py \
	--serialized-file model/model.pt \
	--handler handler.py \
	--extra-files test_data/7.png

mv ${model_name}.mar model_store/
