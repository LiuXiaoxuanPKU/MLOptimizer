import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Model on " + device)


print("Initial Memory:", torch.cuda.memory_allocated())

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000).to(device=device)
y = torch.sin(x).to(device=device)
p = torch.tensor([1, 2, 3]).to(device=device)
# tensor (x, x^2, x^3).
xx = x.unsqueeze(-1).pow(p)

print("Memory after defining input/out:", torch.cuda.memory_allocated())

# define model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.ReLU(),
    torch.nn.Flatten(0, 1)
)

model = model.to(device)
print("Memory after defining model:", torch.cuda.memory_allocated())

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

print("Memory after defining optimizer:", torch.cuda.memory_allocated())

for t in range(2000):
    torch.cuda.reset_max_memory_allocated()
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        print(torch.cuda.memory_summary())
        print(t, loss.item())
        print("Memory after forward:", torch.cuda.memory_allocated())
        print("Memory after backward:", torch.cuda.memory_allocated())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    max_mem = torch.cuda.memory_allocated()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x +'
      f' {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


