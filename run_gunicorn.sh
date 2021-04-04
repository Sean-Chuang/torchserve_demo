#!/bin/bash
gunicorn -w 10 -b 0.0.0.0:8080 --preload server:app