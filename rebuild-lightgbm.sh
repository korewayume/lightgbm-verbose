#!/bin/bash

cwd=`pwd`

mkdir -p /Users/korewayume/Desktop/Gits/LightGBM/build
cd /Users/korewayume/Desktop/Gits/LightGBM/build

cmake \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib \
  ..

make -j4

cd /Users/korewayume/Desktop/Gits/LightGBM/python-package
pip install -e .

cd $cwd
