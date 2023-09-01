#include <functional>
#include <numeric>
#include "elementswise_op_impl.cu"

constexpr int thread_per_block = 256;

__host__ __device__ float GetDistance(const float &b1_x2_i, const float &b2_x2_i,
                                      const float &b1_x1_i, const float &b2_x1_i)
{
  return min(b1_x2_i, b2_x2_i) - max(b1_x1_i, b2_x1_i);
}

template <typename T>
struct FusedGetIntersectionAreaFunctor
{
  FusedGetIntersectionAreaFunctor() {}
  __device__ __forceinline__ T operator()(const T b1_x1, const T b1_x2,
                                          const T b2_x1, const T b2_x2, const T b1_y1,
                                          const T b1_y2, const T b2_y1, const T b2_y2) const
  {
    const T w = GetDistance(b1_x2, b2_x2, b1_x1, b2_x1);
    const T h = GetDistance(b1_y2, b2_y2, b1_y1, b2_y1);
    return (w > 0.0 && h > 0.0) ? w * h : 0.0;
  }
};

template <typename T>
void FusedGetIntersectionAreaKernel(const T *b1_x1, const T *b1_x2,
                                    const T *b2_x1, const T *b2_x2, const T *b1_y1,
                                    const T *b1_y2, const T *b2_y1, const T *b2_y2,
                                    T *output, const size_t count, cudaStream_t cuda_stream)
{
  FusedGetIntersectionAreaFunctor<T> functor;
  cuda::elementwise::EightInputs(functor, (uint)(count), output, b1_x1, b1_x2, b2_x1, b2_x2, b1_y1,
                                 b1_y2, b2_y1, b2_y2, cuda_stream);
}

__global__ void FusedGetIntersectionAreaGradKernel(
    const size_t size, const float *b1_x1, const float *b1_x2,
    const float *b2_x1, const float *b2_x2, const float *b1_y1,
    const float *b1_y2, const float *b2_y1, const float *b2_y2,
    const float *d_inter, float *d_b1_x1, float *d_b1_x2, float *d_b2_x1,
    float *d_b2_x2, float *d_b1_y1, float *d_b1_y2, float *d_b2_y1,
    float *d_b2_y2)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x)
  {
    float d_inter_i = d_inter[i];
    const float w = GetDistance(b1_x2[i], b2_x2[i], b1_x1[i], b2_x1[i]);
    const float h = GetDistance(b1_y2[i], b2_y2[i], b1_y1[i], b2_y1[i]);
    d_b1_x1[i] = 0.0;
    d_b1_x2[i] = 0.0;
    d_b2_x2[i] = 0.0;
    d_b2_x2[i] = 0.0;
    d_b1_y1[i] = 0.0;
    d_b1_y2[i] = 0.0;
    d_b2_y1[i] = 0.0;
    d_b2_y2[i] = 0.0;
    if (w > 0.0 && h > 0.0)
    {
      d_b1_x1[i] = (b1_x1[i] >= b2_x1[i]) ? -d_inter_i * h : 0.0;
      d_b2_x1[i] = (b1_x1[i] <= b2_x1[i]) ? -d_inter_i * h : 0.0;
      d_b1_x2[i] = (b1_x2[i] <= b2_x2[i]) ? d_inter_i * h : 0.0;
      d_b2_x2[i] = (b1_x2[i] >= b2_x2[i]) ? d_inter_i * h : 0.0;
      d_b1_y1[i] = (b1_y1[i] >= b2_y1[i]) ? -d_inter_i * w : 0.0;
      d_b2_y1[i] = (b1_y1[i] <= b2_y1[i]) ? -d_inter_i * w : 0.0;
      d_b1_y2[i] = (b1_y2[i] <= b2_y2[i]) ? d_inter_i * w : 0.0;
      d_b2_y2[i] = (b1_y2[i] >= b2_y2[i]) ? d_inter_i * w : 0.0;
    }
  }
}

extern "C" int FusedGetIntersectionArea(int nparam, void **params, int *ndims,
                                        int64_t **shapes, const char **dtypes,
                                        void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  int input_num = 9;
  int output_index = 8;
  // check input number
  if (nparam != input_num)
  {
    printf(
        "For FusedGetIntersectionArea, the number of input should be %d, "
        "but got %d.",
        input_num, nparam);
    return 1;
  }
  // check dytpe
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      printf(
          "For FusedGetIntersectionArea, the dtype of input should be %s, "
          "but got %s.",
          "float32", dtypes[i]);
      return 2;
    }
  }
  // read input and output parameters
  const float *b1_x1 = static_cast<const float *>(params[0]);
  const float *b1_x2 = static_cast<const float *>(params[1]);
  const float *b1_y1 = static_cast<const float *>(params[2]);
  const float *b1_y2 = static_cast<const float *>(params[3]);
  const float *b2_x1 = static_cast<const float *>(params[4]);
  const float *b2_x2 = static_cast<const float *>(params[5]);
  const float *b2_y1 = static_cast<const float *>(params[6]);
  const float *b2_y2 = static_cast<const float *>(params[7]);
  float *inter = static_cast<float *>(params[8]);

  // calculate the size of output
  size_t size = std::accumulate(shapes[output_index],
                                shapes[output_index] + ndims[output_index],
                                size_t(1), std::multiplies<size_t>());
  FusedGetIntersectionAreaKernel(b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, inter, size, custream);
  return 0;
}

extern "C" int FusedGetIntersectionAreaGrad(int nparam, void **params,
                                            int *ndims, int64_t **shapes,
                                            const char **dtypes, void *stream,
                                            void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  int input_num = 17;
  int output_index = 9;
  // check input number
  if (nparam != input_num)
  {
    printf(
        "For FusedGetIntersectionAreaGrad, the number of input should be "
        "%d, "
        "but got %d.",
        input_num, nparam);
    return 1;
  }
  // check dytpe
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      printf(
          "For FusedGetIntersectionAreaGrad, the dtype of input should be "
          "%s, "
          "but got %s.",
          "float32", dtypes[i]);
      return 2;
    }
  }
  // read input and output parameters
  const float *b1_x1 = static_cast<const float *>(params[0]);
  const float *b1_x2 = static_cast<const float *>(params[1]);
  const float *b1_y1 = static_cast<const float *>(params[2]);
  const float *b1_y2 = static_cast<const float *>(params[3]);
  const float *b2_x1 = static_cast<const float *>(params[4]);
  const float *b2_x2 = static_cast<const float *>(params[5]);
  const float *b2_y1 = static_cast<const float *>(params[6]);
  const float *b2_y2 = static_cast<const float *>(params[7]);
  const float *d_inter = static_cast<const float *>(params[8]);
  float *d_b1_x1 = static_cast<float *>(params[9]);
  float *d_b1_x2 = static_cast<float *>(params[10]);
  float *d_b1_y1 = static_cast<float *>(params[11]);
  float *d_b1_y2 = static_cast<float *>(params[12]);
  float *d_b2_x1 = static_cast<float *>(params[13]);
  float *d_b2_x2 = static_cast<float *>(params[14]);
  float *d_b2_y1 = static_cast<float *>(params[15]);
  float *d_b2_y2 = static_cast<float *>(params[16]);

  // calculate the size of output
  size_t size = std::accumulate(shapes[output_index],
                                shapes[output_index] + ndims[output_index],
                                size_t(1), std::multiplies<size_t>());
  int block_num = (size + thread_per_block - 1) / thread_per_block;
  FusedGetIntersectionAreaGradKernel<<<block_num, thread_per_block, 0,
                                       custream>>>(
      size, b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, d_inter,
      d_b1_x1, d_b1_x2, d_b1_y1, d_b1_y2, d_b2_x1, d_b2_x2, d_b2_y1, d_b2_y2);
  return 0;
}