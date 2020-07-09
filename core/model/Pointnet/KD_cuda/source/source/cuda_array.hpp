#ifndef _CUDA_ARRAY_HPP_
#define _CUDA_ARRAY_HPP_

#include "cuda_util.h"

template <typename scalar>
class MyCudaArray {// only for data transport
public:
    MyCudaArray(int length)
    {
        cudaMalloc((void**)&A, length * sizeof(scalar));
        LENGTH = length;
    }
    __host__ void Free()
    {
        cudaFree(A);
        A = nullptr;
    }
    __host__ void toGPU(scalar* src)
    {
        cudaMemcpy(A, src, LENGTH * sizeof(scalar), cudaMemcpyHostToDevice);
    }
    __host__ void toCPU(scalar* dst)
    {
        cudaMemcpy(dst, A, LENGTH * sizeof(scalar), cudaMemcpyDeviceToHost);
    }
    __host__ inline void toGPU(int id, const scalar *src)
    {
        cudaMemcpy(A+id, src, sizeof(scalar), cudaMemcpyHostToDevice);
    }
    __device__ inline scalar& operator[](int x)
    {
        return A[x];
    }

    scalar* A;
    size_t LENGTH;
private:
};
#endif