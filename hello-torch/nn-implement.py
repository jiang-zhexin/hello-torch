import torch

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype, requires_grad=False)
y = torch.randn(N, D_out, device=device, dtype=dtype, requires_grad=False)

# 使用 nn 包来定义网络
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 常见的损失函数在 nn 包里也有，不需要我们自己实现
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    y_pred: torch.Tensor = model(x)
    loss: torch.Tensor = loss_fn(y_pred, y)
    print(t, loss.item())

    # 梯度清空，调用 Sequential 对象的 zero_grad 后所有里面的变量都会清零梯度
    model.zero_grad()

    # 反向计算梯度。我们通过 Module 定义的变量都会计算梯度
    loss.backward()

    # 更新参数，所有的参数都在 model.paramenters() 里
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
