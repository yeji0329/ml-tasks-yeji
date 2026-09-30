import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)

plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=15, label="Class 0", alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=15, label="Class 1", alpha=0.6)
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.title("make_moons (n=1000, noise=0.2)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train)
print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).reshape(-1, 1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)
print("X_train:", X_train_t.shape, X_train_t.dtype, "| y_train:", y_train_t.shape, y_train_t.dtype)

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
print(model)

x_fake = torch.randn(5, 2)
out = model(x_fake)
print(out.shape)
print(out)

import torch.nn as nn
from sklearn.metrics import accuracy_score

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 1000

hist_train_loss = []
hist_val_loss   = []
hist_val_acc    = []

for epoch in range(1, EPOCHS + 1):

    model.train()
    logit = model(X_train_t)
    loss = criterion(logit, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logit = model(X_val_t)
        val_loss = criterion(val_logit, y_val_t)
        pred = (val_logit > 0).float()
        acc = accuracy_score(y_val_t, pred)

    hist_train_loss.append(loss.item())
    hist_val_loss.append(val_loss.item())
    hist_val_acc.append(acc)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | train loss {loss.item():.4f} "
              f"| val loss {val_loss.item():.4f} | val acc {acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist_train_loss, label="train loss")
axes[0].plot(hist_val_loss, label="val loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[0].set_title("Loss curve"); axes[0].legend()

axes[1].plot(hist_val_acc, label="val acc", color="tab:red")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Val accuracy"); axes[1].legend()

plt.tight_layout()
plt.show()

import numpy as np

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_t = torch.tensor(grid, dtype=torch.float32)
with torch.no_grad():
    zz = model(grid_t)
zz = (zz > 0).float().numpy().reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, levels=1, alpha=0.25,
             colors=["#4C78A8", "#F58518"])
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            s=12, c="#4C78A8", label="Class 0")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            s=12, c="#F58518", label="Class 1")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.title("Decision boundary (MLP)")
plt.show()

import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_model(model, EPOCHS=1000, lr=1e-3):
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, EPOCHS + 1):

        model.train()
        loss = crit(model(X_train_t), y_train_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logit = model(X_val_t)
            val_loss = crit(val_logit, y_val_t)
            acc = accuracy_score(y_val_t, (val_logit > 0).float())

        hist["train_loss"].append(loss.item())
        hist["val_loss"].append(val_loss.item())
        hist["val_acc"].append(acc)
    return hist

linear_model = nn.Sequential(nn.Linear(2, 1))

mlp_model = MLP()

hist_linear = train_model(linear_model)
hist_mlp    = train_model(mlp_model)

print(f"Linear final val acc: {hist_linear['val_acc'][-1]:.4f}")
print(f"MLP    final val acc: {hist_mlp['val_acc'][-1]:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(hist_linear["val_acc"], label="linear (no hidden layer)")
plt.plot(hist_mlp["val_acc"], label="MLP")
plt.xlabel("epoch"); plt.ylabel("val accuracy")
plt.title("Linear vs MLP"); plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

model.eval()
with torch.no_grad():
    test_logit = model(X_test_t)
    test_pred = (test_logit > 0).float()

acc_test = accuracy_score(y_test_t, test_pred)
print(f"Test accuracy: {acc_test:.4f}")

cm = confusion_matrix(y_test_t, test_pred)
print("混淆矩阵（行=真实，列=预测）:")
print(cm)

plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 fontsize=20,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Confusion matrix (test set)")
plt.colorbar()
plt.show()

torch.save(model.state_dict(), "mlp_moons.pth")
print("saved: mlp_moons.pth")

model_loaded = MLP()
model_loaded.load_state_dict(torch.load("mlp_moons.pth"))
model_loaded.eval()

with torch.no_grad():
    logit = model_loaded(X_test_t)
    pred = (logit > 0).float()
print("Loaded model test acc:", accuracy_score(y_test_t, pred))
