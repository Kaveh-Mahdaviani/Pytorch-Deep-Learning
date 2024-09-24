import pandas as pd
from torch.utils.data import Dataset, DataLoader

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        # Load data to pandas DataFrame
        df = pd.read_csv(csv_path)
        # Convert data to a NumPy array and assign to self.data
        self.data = df.to_numpy()
        
    # Implement __len__ to return the number of data samples
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        # Assign last data column to label
        label = self.data[idx, -1]
        return features, label


# Create an instance of the WaterDataset
dataset_train = WaterDataset('water_train.csv')

# Create a DataLoader based on dataset_train
dataloader_train = DataLoader(
    dataset_train,
    batch_size = 2,
    shuffle = True,
)

# Get a batch of features and labels
features, labels = next(iter(dataloader_train))
print(features, labels)

###########
#creating the model

import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the three linear layers
        self.fc1 = nn.Linear(9,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        # Add two batch normalization layers
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

        # applying the He/Kiming initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity = "sigmoid")
        
    def forward(self, x):
        # Pass x through linear layers adding activations, we use elu not relu to prevent vanishing gradiant 
        x = nn.functional.elu(self.bn1(self.fc1(x)))
        x = nn.functional.elu(self.bn2(self.fc2(x)))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
    
###############
# training the model

import torch.optim as optim

net = Net()
criterion = nn.BCELoss()

def train_model(optimizer, net, num_epochs):
    for epoch in range(num_epochs):
        for features, labels in dataloader_train:
            optimizer.zero_grad()
            output = net(features)
            loss = criterion(output, labels.view(-1, 1))
            loss.backwards()
            optimizer.step()

# Define the SGD/ RMSprop / Adam optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.001)
# optimizer = optim.RMSprop(net.parameters(), lr=0.001)
optimizer = optim.Adam(net.parameters())

train_model(
    optimizer=optimizer,
    net=net,
    num_epochs=10,
)
##############
#evaluating the model:
from torchmetrics import Accuracy

# Set up binary accuracy metric
acc = Accuracy(task = 'binary')

net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        # Get predicted probabilities for test data batch
        outputs = net(features)
        preds = (outputs >= 0.5).float()
        acc(preds, labels.view(-1, 1))

# Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")