from pytorch_block_sparse import BlockSparseLinear
import torch

N, D_in, H, D_out = 10, 100, 30, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

mode = torch.nn.Sequential(
        BlockSparseLinear(D_in, H, density=0.1)
        torch.nn.ReLU(),
        BlockSparseLinear(1024, 256, density=0.1) 
    )

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    print(i)

    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
