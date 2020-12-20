#!/usr/bin/env bash

cd tests || exit 

make clean
make
cd ../
export PYTHONFAULTHANDLER=1
python3 -m unittest discover -s ./tests -p "*_test.py"


