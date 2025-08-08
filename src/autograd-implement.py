import torch

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机的 Tensor 作为输入和输出
# 输入和输出需要的 requires_grad=False (默认)，
# 因为我们不需要计算 loss 对它们的梯度
x = torch.randn(N, D_in, device=device, dtype=dtype, requires_grad=False)
y = torch.randn(N, D_out, device=device, dtype=dtype, requires_grad=False)

# 创建 weight 的 Tensor，需要设置 requires_grad=True
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用 autograd 进行反向计算。它会计算 loss 对所有对它有影响的
    # requires_grad=True 的 Tensor 的梯度
    loss.backward()

    # 手动使用梯度下降更新参数。一定要把更新的代码放到 torch.no_grad() 里
    # 否则下面的更新也会计算梯度
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 手动把梯度清零
        w1.grad.zero_()
        w2.grad.zero_()
