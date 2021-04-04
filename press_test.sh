#!/bin/bash
curl http://127.0.0.1:8080/predictions/mnist -T test_data/0.png

ab -k -T application/jpg -n 10000 -c 100 \
	-u ../../../project/torchserve_demo/test_data/0.png \
	http://127.0.0.1:8080/predictions/mnist