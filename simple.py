from unittest import TestCase
import torch
import unittest
import torch.optim as optim
from pytorch_block_sparse import BlockSparseLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device =====>", device)

N, D_in, H, D_out = 64, 32, 32, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in).cuda()
#y = torch.randn(N, D_out)
y = 2 * x
print(x)
print(y)
model = BlockSparseLinear(32, 32, True, 1)
model = model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 10000 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



