"""独立推理程序：输入一张本地图片，输出猫/狗判断和置信度。

用法: python infer.py <图片路径>
"""
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.240, 0.225]


class CatsDogsCNN(nn.Module):
    """与训练时完全相同的网络结构（加载权重要求结构一字不差）"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def predict(image_path, weights_path=None):
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cats_dogs_cnn.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatsDogsCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()

    if p > 0.5:
        return "狗", p
    return "猫", 1.0 - p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python infer.py <图片路径>")
        sys.exit(1)
    label, conf = predict(sys.argv[1])
    print(f"预测结果: {label} (置信度 {conf:.2%})")
