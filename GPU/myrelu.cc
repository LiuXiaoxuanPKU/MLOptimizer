#include <torch/extension.h>
#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor> myrelu_forward_cuda(torch::Tensor data);
torch::Tensor myrelu_backward_cuda(torch::Tensor mask, torch::Tensor data);

class MyReLU : public Function<ActQuantizedReLU> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input) {
    torch::Tensor mask, output;
    std::tie(output, mask) = myrelu_forward_cuda(input);
    ctx->save_for_backward({mask});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {myrelu_backward_cuda(saved[0], grad_outputs[0])};
  }
};

torch::Tensor myrelu(torch::Tensor input) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return MyReLU::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myrelu", &myrelu);
}