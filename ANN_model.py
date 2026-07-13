import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data = load_breast_cancer() # یک شی از  <class 'sklearn.utils._bunch.Bunch'>


X = data.data # = data['data']
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

class CancerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CancerDataset(X_train, y_train)
test_dataset = CancerDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ساخت ANN
class Cancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=30, out_features=32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

model = Cancer()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for X_batch, y_batch in train_loader:
        prediction = model(X_batch)
        loss = loss_fn(prediction, y_batch) # تو پروژه های واقعی ضرر کل epoch حساب میشه
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch+1}, Loss: {loss.item():.4f}")

#ارزیابی مدل
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        prediction = model(X_batch)
        predicted = (prediction >= 0.5).float()
        correct += (predicted>=0.5).sum().item()
        total += y_batch.size(0)
accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")


new_patient = torch.tensor([[ 14.2, 20.1, 92.3, 620.0, 0.105,
    0.150, 0.120, 0.070, 0.180, 0.062,
    0.450, 1.10, 3.10, 42.0, 0.006,
    0.025, 0.030, 0.012, 0.020, 0.003,
    16.5, 27.0, 108.0, 820.0, 0.145,
    0.320, 0.410, 0.180, 0.310, 0.090]],dtype=torch.float32)
new_patient = scaler.transform(new_patient)
new_patient = torch.tensor(new_patient,dtype=torch.float32)
model.eval()

with torch.no_grad():
    prediction = model(new_patient)
    if prediction >= 0.5:
        print("Benign")
    else:
        print("Malignant")
    print(prediction)