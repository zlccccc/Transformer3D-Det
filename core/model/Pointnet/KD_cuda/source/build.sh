nvcc -DGPU \
   -I/usr/local/cuda/include/ -DCUDNN  --compiler-options "-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC" \
    -c source/KDtree.cu -o KDtree.o -std=c++11
gcc -shared -o KDtree.so KDtree.o -L /usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
