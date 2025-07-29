import math
from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch


class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        
        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)  # 定义一个全局参数，可以通过反向传播进行优化
        self.grl = GradientReversal()  # 非参数梯度反转层

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)
# grl（GradientReversal 的实例）被应用于 global_T 和 lambda_。
# 使用了一个全局参数 global_T 和一个梯度反转层 grl。在模型的前向传播过程中，梯度反转层会对全局参数 global_T 进行操作，根据传入的 lambda_ 参数进行梯度反转。
from torch.autograd import Function
class GradientReversalFunction(Function):
    # 一个自定义的 Torch 自动微分函数
# GRL通过在反向传播期间乘以一个固定的负权重（一般为负数），来反转梯度的方向
    @staticmethod
    def forward(ctx, x, lambda_):
        # ctx是一个上下文对象，它用于在前向传播和反向传播之间存储中间结果。ctx 是一个可变的对象，可用于存储任意需在反向传播中使用的变量或张量。
        ctx.lambda_ = lambda_
        # 在前向传播过程中，可以通过 ctx 的属性或方法存储任意变量或张量，以便在反向传播时使用。
        # forward 方法在前向传播过程中执行，接受输入张量 x 和参数 lambda_，返回 x 的克隆张量
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        # 在反向传播时，你可以通过 ctx 获取之前存储的变量或张量，并使用它们计算梯度。
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads  # 为了实现梯度反转的操作
        # backward 方法在反向传播过程中执行，接受梯度张量 grads，并利用参数 lambda_ 计算得到 dx，即 x 对梯度的贡献
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


if __name__ == '__main__':

    model = Global_T()
    input = torch.rand(24,24,24)
    input2 = torch.rand(24,24,24)

    out = model(input, input2)
    
    print(out)






