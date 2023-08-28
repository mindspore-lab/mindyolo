#include <stdio.h>

constexpr int THREADS = 256;
constexpr float EPS = 1e-7;

__global__ void FusedGetCiouKernel(const float *v, const float *iou, const float *rho2, const float *c2,
                                   float *alpha, float *out, const size_t size)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; idx < size;
       idx += step)
  {
    const float v_i = v[idx];
    const float iou_i = iou[idx];
    const float alpha_i = v_i / (v_i - iou_i + 1.0 + EPS);
    out[idx] = iou_i - (rho2[idx] / c2[idx] + v_i * alpha_i);
    alpha[idx] = alpha_i;
  }
}

extern "C" int FusedGetCiou(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                            void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 5;
  constexpr int TOTAL_PARAM_NUM = 6;
  if (nparam != TOTAL_PARAM_NUM)
  {
    printf("[Error] nparam is %d", nparam);
    return 1;
  }
  // check the data type is float32
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      return 2;
    }
  }
  // read input & output parameters
  void *v = params[0];
  void *iou = params[1];
  void *rho2 = params[2];
  void *c2 = params[3];
  void *alpha = params[4];
  void *out = params[5];
  // calculate the size to data to be processed
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  FusedGetCiouKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<const float *>(v), static_cast<const float *>(iou),
                                                      static_cast<const float *>(rho2), static_cast<const float *>(c2),
                                                      static_cast<float *>(alpha), static_cast<float *>(out), size);
  return 0;
}

__global__ void FusedGetCiouBpropKernel(const float *v, const float *iou, const float *rho2, const float *c2,
                                        const float *d_alpha, const float *d_out, float *d_v, float *d_iou,
                                        float *d_rho2, float *d_c2, const size_t size)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; idx < size;
       idx += step)
  {
    const float v_i = v[idx];
    const float iou_i = iou[idx];
    const float c2_i = c2[idx];
    const float d_out_i = d_out[idx];
    const float alpha_i = v_i / (v_i - iou_i + 1.0 + EPS);
    d_v[idx] = -alpha_i * d_out_i;
    d_iou[idx] = d_out_i;
    d_rho2[idx] = -d_out_i / c2_i;
    d_c2[idx] = rho2[idx] / (c2_i * c2_i) * d_out_i;
  }
}

extern "C" int FusedGetCiouBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                 void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 6;
  constexpr int TOTAL_PARAM_NUM = 10;
  if (nparam != TOTAL_PARAM_NUM)
    return 1;
  // check the data type is float32
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      return 2;
    }
  }
  // read input & output parameters
  void *v = params[0];
  void *iou = params[1];
  void *rho2 = params[2];
  void *c2 = params[3];
  void *d_alpha = params[4];
  void *d_out = params[5];
  void *d_v = params[6];
  void *d_iou = params[7];
  void *d_rho2 = params[8];
  void *d_c2 = params[9];
  // calculate the size to data to be processed
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  FusedGetCiouBpropKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<const float *>(v), static_cast<const float *>(iou),
                                                           static_cast<const float *>(rho2), static_cast<const float *>(c2),
                                                           static_cast<const float *>(d_alpha), static_cast<const float *>(d_out),
                                                           static_cast<float *>(d_v), static_cast<float *>(d_iou),
                                                           static_cast<float *>(d_rho2), static_cast<float *>(d_c2), size);
  return 0;
}