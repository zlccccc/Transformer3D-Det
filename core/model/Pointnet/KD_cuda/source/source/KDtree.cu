#define USE_CUDA
#define DEBUG_PROTECT
//#define DEBUG
#include "cuda_KDtree.hpp"
extern "C" {
void SearchForRadius(float* point, int point_size, float* query_points, int query_size, float distance, int max_count, int* result, int dim = 3)
{
    SearchRadius(point, point_size, query_points, query_size, distance, max_count, result, dim);
}
}
