#include <functional>
#include <numeric>

constexpr int thread_per_block = 256;

__global__ void FusedGetBounddingBoxesCoordKernel(
    const size_t size, const float *x1, const float *y1, const float *w1,
    const float *h1, const float *x2, const float *y2, const float *w2,
    const float *h2, float *b1_x1, float *b1_y1, float *b1_x2, float *b1_y2,
    float *b2_x1, float *b2_y1, float *b2_x2, float *b2_y2)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x)
  {
    const float w1_ = w1[i] / 2.0;
    const float h1_ = h1[i] / 2.0;
    const float w2_ = w2[i] / 2.0;
    const float h2_ = h2[i] / 2.0;
    const float x1_i = x1[i], y1_i = y1[i], x2_i = x2[i], y2_i = y2[i];
    b1_x1[i] = x1_i - w1_;
    b1_x2[i] = x1_i + w1_;
    b1_y1[i] = y1_i - h1_;
    b1_y2[i] = y1_i + h1_;
    b2_x1[i] = x2_i - w2_;
    b2_x2[i] = x2_i + w2_;
    b2_y1[i] = y2_i - h2_;
    b2_y2[i] = y2_i + h2_;
  }
}

__global__ void FusedGetBounddingBoxesCoordGradKernel(
    const size_t size, const float *d_b1_x1, const float *d_b1_x2,
    const float *d_b1_y1, const float *d_b1_y2, const float *d_b2_x1,
    const float *d_b2_x2, const float *d_b2_y1, const float *d_b2_y2,
    float *d_x1, float *d_y1, float *d_w1, float *d_h1, float *d_x2,
    float *d_y2, float *d_w2, float *d_h2)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x)
  {
    const float d_b1_x1_i = d_b1_x1[i];
    const float d_b1_x2_i = d_b1_x2[i];
    const float d_b1_y1_i = d_b1_y1[i];
    const float d_b1_y2_i = d_b1_y2[i];
    const float d_b2_x1_i = d_b2_x1[i];
    const float d_b2_x2_i = d_b2_x2[i];
    const float d_b2_y1_i = d_b2_y1[i];
    const float d_b2_y2_i = d_b2_y2[i];
    d_x1[i] = d_b1_x1_i + d_b1_x2_i;
    d_y1[i] = d_b1_y1_i + d_b1_y2_i;
    d_w1[i] = 0.0;
    d_h1[i] = 0.0;
    d_x2[i] = d_b2_x1_i + d_b2_x2_i;
    d_y2[i] = d_b2_y1_i + d_b2_y2_i;
    d_w2[i] = 0.0;
    d_h2[i] = 0.0;
  }
}

extern "C" int FusedGetBounddingBoxesCoord(int nparam, void **params,
                                           int *ndims, int64_t **shapes,
                                           const char **dtypes, void *stream,
                                           void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  int input_num = 16;
  int output_index = 8;
  // check input number
  if (nparam != input_num)
  {
    printf(
        "For FusedGetBounddingBoxesCoord, the number of input should be %d, "
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
          "For FusedGetBounddingBoxesCoord, the dtype of input should be %s, "
          "but got %s.",
          "float32", dtypes[i]);
      return 2;
    }
  }
  // read input and output parameters
  const float *x1 = static_cast<const float *>(params[0]);
  const float *y1 = static_cast<const float *>(params[1]);
  const float *w1 = static_cast<const float *>(params[2]);
  const float *h1 = static_cast<const float *>(params[3]);
  const float *x2 = static_cast<const float *>(params[4]);
  const float *y2 = static_cast<const float *>(params[5]);
  const float *w2 = static_cast<const float *>(params[6]);
  const float *h2 = static_cast<const float *>(params[7]);
  float *b1_x1 = static_cast<float *>(params[8]);
  float *b1_x2 = static_cast<float *>(params[9]);
  float *b1_y1 = static_cast<float *>(params[10]);
  float *b1_y2 = static_cast<float *>(params[11]);
  float *b2_x1 = static_cast<float *>(params[12]);
  float *b2_x2 = static_cast<float *>(params[13]);
  float *b2_y1 = static_cast<float *>(params[14]);
  float *b2_y2 = static_cast<float *>(params[15]);

  // calculate the size of output
  size_t size = std::accumulate(shapes[output_index],
                                shapes[output_index] + ndims[output_index],
                                size_t(1), std::multiplies<size_t>());
  int block_num = (size + thread_per_block - 1) / thread_per_block;
  FusedGetBounddingBoxesCoordKernel<<<block_num, thread_per_block, 0,
                                      custream>>>(
      size, x1, y1, w1, h1, x2, y2, w2, h2, b1_x1, b1_y1, b1_x2, b1_y2, b2_x1,
      b2_y1, b2_x2, b2_y2);
  return 0;
}

extern "C" int FusedGetBounddingBoxesCoordGrad(int nparam, void **params,
                                               int *ndims, int64_t **shapes,
                                               const char **dtypes,
                                               void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  int input_num = 16;
  int output_index = 8;
  // check input number
  if (nparam != input_num)
  {
    printf(
        "For FusedGetBounddingBoxesCoordGrad, the number of input should be "
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
          "For FusedGetBounddingBoxesCoordGrad, the dtype of input should be "
          "%s, "
          "but got %s.",
          "float32", dtypes[i]);
      return 2;
    }
  }
  // read input and output parameters
  const float *d_b1_x1 = static_cast<const float *>(params[0]);
  const float *d_b1_x2 = static_cast<const float *>(params[1]);
  const float *d_b1_y1 = static_cast<const float *>(params[2]);
  const float *d_b1_y2 = static_cast<const float *>(params[3]);
  const float *d_b2_x1 = static_cast<const float *>(params[4]);
  const float *d_b2_x2 = static_cast<const float *>(params[5]);
  const float *d_b2_y1 = static_cast<const float *>(params[6]);
  const float *d_b2_y2 = static_cast<const float *>(params[7]);
  float *d_x1 = static_cast<float *>(params[8]);
  float *d_y1 = static_cast<float *>(params[9]);
  float *d_w1 = static_cast<float *>(params[10]);
  float *d_h1 = static_cast<float *>(params[11]);
  float *d_x2 = static_cast<float *>(params[12]);
  float *d_y2 = static_cast<float *>(params[13]);
  float *d_w2 = static_cast<float *>(params[14]);
  float *d_h2 = static_cast<float *>(params[15]);

  // calculate the size of output
  size_t size = std::accumulate(shapes[output_index],
                                shapes[output_index] + ndims[output_index],
                                size_t(1), std::multiplies<size_t>());
  int block_num = (size + thread_per_block - 1) / thread_per_block;
  FusedGetBounddingBoxesCoordGradKernel<<<block_num, thread_per_block, 0,
                                          custream>>>(
      size, d_b1_x1, d_b1_x2, d_b1_y1, d_b1_y2, d_b2_x1, d_b2_x2, d_b2_y1, d_b2_y2, d_x1, d_y1, d_w1,
      d_h1, d_x2, d_y2, d_w2, d_h2);
  return 0;
}
