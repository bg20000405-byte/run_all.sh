import torch
import torch.nn as nn
import torch.optim as optim
import argparse, os

# 简单模型
class SimpleNet(nn.Module):
    def __init__(self, ncls=4):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(100, ncls)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ncls", type=int, default=4)
    args = parser.parse_args()

    model = SimpleNet(args.ncls)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 模拟数据 (100 维特征，分类任务)
    X = torch.randn(200, 100)
    y = torch.randint(0, args.ncls, (200,))

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/final.pth")
    print("✅ 模型已保存到 results/models/final.pth")
