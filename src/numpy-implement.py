import numpy as np

# N 是 batch size；D_in 是输入大小
# H 是隐层的大小；D_out 是输出大小
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机产生输入与输出
x: np.ndarray = np.random.randn(N, D_in)  # 64 x 1000
y: np.ndarray = np.random.randn(N, D_out)  # 64 x 10

# 随机初始化参数
w1: np.ndarray = np.random.randn(D_in, H)  # 1000 x 100
w2: np.ndarray = np.random.randn(H, D_out)  # 100 x 10

# 学习率
learning_rate = 1e-6

for t in range(500):
    # 前向计算 y
    h: np.ndarray = x.dot(w1)  # 64 x 100
    h_relu: np.ndarray = np.maximum(h, 0)  # 64 x 100
    y_pred: np.ndarray = h_relu.dot(w2)  # 64 x 10

    # 计算loss
    loss: np.ndarray = np.square(y_pred - y).sum()  # 64 x 10
    print(t, loss)

    # 反向计算梯度
    # delta1
    grad_y_pred = 2.0 * (y_pred - y)  # 64 x 10
    # w2 的梯度
    grad_w2: np.ndarray = h_relu.T.dot(grad_y_pred)  # 100 x 10

    # delta2
    grad_h_relu: np.ndarray = grad_y_pred.dot(w2.T)  # 64 x 100
    grad_h: np.ndarray = grad_h_relu.copy()  # 64 x 100
    grad_h[h < 0] = 0  # 64 x 100
    # w1 的梯度
    grad_w1: np.ndarray = x.T.dot(grad_h)  # 1000 x 100

    # 更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
