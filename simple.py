from unittest import TestCase
import torch
import unittest
import torch.optim as optim
from pytorch_block_sparse import BlockSparseLinear

import inspect
from gpu_mem_track import  MemTracker

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device =====>", device)

N, D_in, H1, H2 = 64, 32, 32 * 32, 32 * 32
D_out = D_in

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in).cuda()
#y = torch.randn(N, D_out)
y = 2 * x.mul(3.5)  * x + x + x.absolute().sqrt() + 1

print(x)
print(y)
model = torch.nn.Sequential(
    BlockSparseLinear(D_in, H1, True, 0.1),
    torch.nn.ReLU(),
    BlockSparseLinear(H1, H2, True, 0.1),
    torch.nn.ReLU(),
    BlockSparseLinear(H2, D_out, True, 0.1)
)

org_para_num = D_in * H1 + H1 * H2 + H2 * D_out
sparse_para_num = sum(p.numel() for p in model.parameters())

print("Original parameter number =========>", org_para_num)
print("Parameter Number =====>", sparse_para_num)

model = model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(50000):
    # Forward pass: compute predicted y by passing x to the model.
    gpu_tracker.track()
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)

    gpu_tracker.track()
    if t % 1000 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    gpu_tracker.track()


