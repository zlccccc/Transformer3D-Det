
g++ -Wall -Wfatal-errors -Ofast -fPIC --std=c++11 -o PointCloudToCylinderMap.o -c source/PointCloudToCylinderMap.cpp 
g++ -shared PointCloudToCylinderMap.o -o PointCloudToCylinderMap.so

