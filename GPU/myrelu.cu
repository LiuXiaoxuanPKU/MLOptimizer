#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void myrelu_forward_kernel(const float* __restrict__ data,
                                      int32_t* __restrict__ mask,
                                      float* __restrict__ output,
                                      int N) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < N) {
        int bit  = data[id] > 0;
        atomicOr(mask + id / 32, bit << (id % 32));
        output[id] = fmax(data[id], 0.0f);
    }
}


std::pair<torch::Tensor, torch::Tensor> myrelu_forward_cuda(
       torch::Tensor data) {
    int n_elements = 1;
    for (size_t i = 0; i < data.dim(); ++i) {
        n_elements *= data.size(i);
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor mask = torch::zeros({(n_elements + 31) / 32}, options);
    torch::Tensor output = torch::empty_like(data);

    int threads = 256;
    int blocks = (n_elements + threads - 1) / threads;

    myrelu_forward_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), mask.data_ptr<int32_t>(), output.data_ptr<float>(),
    n_elements);

    return std::make_pair(output, mask);
}

__global__ void myrelu_backward_kernel(const float* __restrict__ data,
                                                   int32_t* __restrict__ mask,
                                                   float* __restrict__ output,
                                                   int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N) {
    if ((mask[id / 32] >> (id % 32)) & 1) {
      output[id] = data[id];
    }
  }
}

std::vector<at::Tensor> myrelu_backward_cuda(torch::Tensor mask, torch::Tensor data){
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  int threads = 256;
  int blocks = (n_elements + threads - 1) / threads;

  torch::Tensor output = torch::zeros_like(data);

  myrelu_backward_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), mask.data_ptr<int32_t>(), output.data_ptr<float>(),
    n_elements);

  return output;
}