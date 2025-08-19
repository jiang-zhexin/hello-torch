import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

# 使用 Adam 算法，需要提供模型的参数和 learning rate
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    y_pred: torch.Tensor = model(x)

    loss: torch.Tensor = loss_fn(y_pred, y)
    print(t, loss.item())

    # 梯度清零，原来调用的是 model.zero_grad，现在调用的是 optimizer 的 zero_grad
    optimizer.zero_grad()
    loss.backward()
    # 调用 optimizer.step 实现参数更新
    optimizer.step()
