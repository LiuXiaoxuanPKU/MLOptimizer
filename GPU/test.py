import numpy as np
import torch
from torch.nn import init, functional as F

from torch.utils.cpp_extension import load
myrelu = load(name="my_relu", sources=["myrelu.cc", "myrelu.cu"], verbose=True)


def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f MB" % allocated)
        print("reserved:  %.2f MB" % reserved)
    return allocated

def test_relu_correctness():
    data_np = np.random.randn(28, 14, 14, 8).astype('float32')

    for device in ['cuda']:
        def test_implementation(func):
            data = torch.tensor(data_np).to(torch.device(device)).requires_grad_()

            before_size = get_memory_usage()
            output = func(data)
            after_size = get_memory_usage()
            print("Memory use:", after_size - before_size)

            output.backward(torch.ones_like(output))

            return [x.detach().cpu().numpy() for x in [output, data.grad]]

        output_ref, grad_data_ref =  test_implementation(F.relu)
        output_us, grad_data_us = test_implementation(myrelu.myrelu)

        print("========== ReLU Correctness Test ==========")
        np.testing.assert_allclose(output_ref, output_us)
        np.testing.assert_allclose(grad_data_ref, grad_data_us)


if __name__ == "__main__":
    test_relu_correctness()
