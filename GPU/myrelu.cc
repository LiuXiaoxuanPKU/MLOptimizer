#include <torch/extension.h>
#include <torch/torch.h>

using namespace torch::autograd;

std::pair<torch::Tensor, torch::Tensor> myrelu_forward_cuda(torch::Tensor data);
torch::Tensor myrelu_backward_cuda(torch::Tensor mask, torch::Tensor data);


// Helper for type check
#define CHECK_CUDA_TENSOR_TYPE(name, type)                                        \
  TORCH_CHECK(name.device().is_cuda(), #name " must be a CUDA tensor!");          \
  TORCH_CHECK(name.is_contiguous(), #name " must be contiguous!");                \
  TORCH_CHECK(name.dtype() == type, "The type of " #name " is not correct!");     \

torch::Tensor compress(torch::Tensor data) {
    float* float_data= data.data_ptr<float>();
    std::cout << float_data[0] << std::endl;
    return data;
}

torch::Tensor decompress(torch::Tensor data) {
    return data;
}

class MyReLU : public Function<MyReLU> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input) {
    torch::Tensor mask, output;
    std::tie(output, mask) = myrelu_forward_cuda(input);

    ctx->save_for_backward({compress(mask)});
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    return {myrelu_backward_cuda(decompress(saved[0]), grad_outputs[0])};
  }
};

torch::Tensor myrelu(torch::Tensor input) {
  CHECK_CUDA_TENSOR_TYPE(input, torch::kFloat32);
  return MyReLU::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myrelu", &myrelu);
}