import cupy as cp
loaded_from_source = r'''
extern "C"{
    
__global__ void test_norm(double* vectors, const double* norm, const double* sign, unsigned int shape0, unsigned int shape1)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (tid < shape0 * shape1)
    {
            vectors[tid] = (-1.0)* sign[(tid/shape0)] * vectors[tid] / norm[(tid/shape0)] * sqrt(double(shape0));
    }
}

__global__ void test_discrete(double* vectors_discrete, const signed long long* labels, const signed long long* labels_count, unsigned int shape0, unsigned int shape1)

{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    if (tid < shape0 * shape1)
    {
        if(labels[tid % shape0] == (tid/shape0))

            vectors_discrete[tid] = 1 / sqrt(double(labels_count[tid/shape0]));
    }
}


}'''
mod= cp.RawModule(code=loaded_from_source, options=('-std=c++11',),backend=u'nvrtc')
ker_norm = mod.get_function('test_norm')
ker_discrete = mod.get_function('test_discrete')


