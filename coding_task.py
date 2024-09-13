import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Tuple
import numpy as np
import requests
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

#### LOADING THE MODEL

from torchvision.models import resnet18

from helper_functions import accuracy_fn

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("out/models/01_MIA_67.pt", map_location="cpu")

model.load_state_dict(ckpt)

model.eval()  # Set the model to evaluation mode

# Define the data transform
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # normalize images according to training data of resnet18
])


#### DATASETS

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]



# Create dataset instances
pub_dataset: MembershipDataset = torch.load("out/data/01/pub.pt")  # dataset with membership labels
pub_dataset.transform = trans
priv_out_dataset: TaskDataset = torch.load("out/data/01/priv_out.pt")
priv_out_dataset.transform = trans  # dataset membership labels None

# Create DataLoader instances
pub_loader = DataLoader(pub_dataset, batch_size=32, shuffle=False)
priv_out_loader = DataLoader(priv_out_dataset, batch_size=32, shuffle=False)


#### Extract features
def extract_features(model, data, mem=True):
    features = []
    labels = []
    memberships = []

    for i in range(data.__len__()):
        ids, img, label, membership = data.__getitem__(i)
        img = img.to(torch.device("cpu"))
        outputs = model(img.unsqueeze(0))  # Get the outputs from the model
        o = outputs.detach().numpy()
        features.append(torch.from_numpy(o))
        #features.append(outputs.detach().numpy()) only neededfor option 3
        labels.append(torch.tensor([label]))
        if mem:
            memberships.append(torch.tensor([membership]))

    if mem:
        return np.vstack(features), np.hstack(labels), np.hstack(memberships)
    else:
        return np.vstack(features), np.hstack(labels)


# Extract features from PUB dataset
pub_features, pub_labels, pub_memberships = extract_features(model, pub_dataset)

# Extract features from PRIV OUT dataset
priv_out_features, priv_out_labels = extract_features(model, priv_out_dataset, False)

### Train attack model

"""
# For option 3

X = torch.from_numpy(pub_features).type(torch.float)
y = torch.from_numpy(pub_memberships).type(torch.float)
final_set = torch.from_numpy(priv_out_features).type(torch.float)
"""

X_train, X_test, y_train, y_test = train_test_split(pub_features,
                                                    pub_memberships,
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible



"""
### Option 1

# Train a logistic regression model
attack_model = LogisticRegression(solver='lbfgs', max_iter=1000)
attack_model.fit(pub_features, pub_memberships)

# Evaluate the attack model on the PUB dataset
pub_pred_scores = attack_model.predict_proba(pub_features)[:, 1]
roc_auc = roc_auc_score(pub_memberships, pub_pred_scores)
print(f"AUC on PUB dataset: {roc_auc}")
"""


### Option 2
attack_model = xgb.XGBClassifier(n_estimators=100)
attack_model.fit(X_train, y_train)

# Evaluate the attack model on the PUB dataset
pub_pred_scores = attack_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, pub_pred_scores)
print(f"AUC on PUB dataset: {roc_auc}")


"""
### Option 3

# Build model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=44, out_features=100)
        self.layer_2 = nn.Linear(in_features=100, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=100)
        self.layer_4 = nn.Linear(in_features=100, out_features=100)
        self.layer_5 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_5(self.relu(self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))))

attack_model = CircleModelV2().to("cpu")


# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(attack_model.parameters(), lr=0.1)

#eval model
# Fit the model
torch.manual_seed(42)
epochs = 1500

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = attack_model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    attack_model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = attack_model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))  # logits -> prediction probabilities -> prediction labels
        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

"""


# Predict membership scores for PRIV OUT dataset
priv_out_pred_scores = attack_model.predict_proba(priv_out_features)[:, 1]

"""
# Make predictions for option 3
#attack_model.eval()
#with torch.inference_mode():
#   priv_out_pred_scores = torch.round(torch.sigmoid(attack_model(final_set))).squeeze()
"""


#### EXAMPLE SUBMISSION

df = pd.DataFrame(
    {
        "ids": priv_out_dataset.ids,
        "score": priv_out_pred_scores,
    }
)
df.to_csv("sub.csv", index=None)


response = requests.post("http://34.71.138.79:9014/mia", files={"file": open("sub.csv", "rb")}, headers={"token": "20035462"})
print(response.json())
