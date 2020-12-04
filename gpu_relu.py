import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Model on " + device)

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000).to(device=device)
y = torch.sin(x).to(device=device)
p = torch.tensor([1, 2, 3]).to(device=device)
# tensor (x, x^2, x^3).
xx = x.unsqueeze(-1).pow(p)

print(xx.shape)
print(y.shape)

# define model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.ReLU(),
    torch.nn.Flatten(0, 1)
)

model = model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

ini_mem = torch.cuda.memory_allocated()
max_mem = 0
for t in range(2000):
    torch.cuda.reset_max_memory_allocated()
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        print(torch.cuda.memory_summary())
        print(t, loss.item(), max_mem)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    max_mem = torch.cuda.max_memory_allocated() - ini_mem
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x +'
      f' {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


