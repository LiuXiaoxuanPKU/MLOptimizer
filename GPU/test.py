import numpy as np
import torch
from torch.nn import init, functional as F

from torch.utils.cpp_extension import load
myrelu = load(name="my_relu", sources=["myrelu.cc", "myrelu.cu"], verbose=True)



def test_relu_correctness():
    data_np = np.random.randn(128, 56, 56, 32).astype('float32')

    for device in ['cuda']:
        def test_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()

            output = func(data)
            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref =  test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(myrelu.myrelu)

        print("========== ReLU Correctness Test ==========")
        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


if __name__ == "__main__":
    test_relu_correctness()
