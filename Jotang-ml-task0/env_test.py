# env_test.py —— 任务3第5步：环境验证脚本
# 运行: ~/ml-env/bin/python env_test.py
import sys
import torch

print("Python 版本 :", sys.version.split()[0])
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用   :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 设备    :", torch.cuda.get_device_name(0))

# 一次张量运算：在 GPU（如可用）上做矩阵乘法
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(3, 4, device=device)
b = torch.randn(4, 5, device=device)
c = a @ b

print("运算设备    :", device)
print("输入 a 形状 :", tuple(a.shape))
print("输入 b 形状 :", tuple(b.shape))
print("输出 c=a@b 形状:", tuple(c.shape))
print("c 的前两行:\n", c[:2])
