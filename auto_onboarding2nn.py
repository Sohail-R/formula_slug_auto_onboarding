import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Create Model class inheriting nn.Module

class Model(nn.Module):
    def __init__(self, in_features=4, h1=128,h2=64,out_features=5):
        super().__init__() #instantiates the nn.Module
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
    
    #Forward function moves things forward in the neural network
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
#Create a manual seed for randomization
torch.manual_seed(41)

#create instance of the model
model = Model()


#load in data into pandas w/ url
url = 'https://storage.googleapis.com/kagglesdsdata/datasets/7132156/11389151/clash_royale_cards.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251018%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251018T005936Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=90dbc5a5325f9a1c69b912c5d857c42f897f89139fb506848dea57ee65704ada390742900f73f546d51097e8e5d994410971c6aedeb47279fe6f1d4a12a3006ad2fa196738a94534b6478c60cae879efa34333c80e4b52efdf7a72d0d12b053728e83b6e7eba458bcd44ea4f1142965a6a0565af0358c00e711a558a24e7acd000c355885f1e43325a8d5ef8fd1c00026d20ba40d38fa52b73370aa667b5538fad6a2b457eb3f283b75339a9d272c3934aec0e6141d3665ff9e96160c80d73e786a936d6f9c6457173852e4ae1ebc79d3fa94e7b97ce119a2a287e9d5f0541147e7022da21ea8243a87ea548e12b87ad40fbdc3fded6e882d2902d804cb5ee9b'
df = pd.read_csv(url)

#Choose cards to keep (reduce dataset size)
# cards = ['Knight','Archers','Goblins','Giant','P.E.K.K.A']
# df = df[df['Card'].isin(cards)]


#Find the true number of classes
# Get all unique card names
all_cards = df['Card'].unique()
# Calculate the number of unique cards
num_classes = len(all_cards)
print(f"Total Unique Cards (Classes): {num_classes}")
#Change card names from text to numbers
# df['Card'] = df['Card'].replace('Knight',0)
# df['Card'] = df['Card'].replace('Archers',1)
# df['Card'] = df['Card'].replace('Goblins',2)
# df['Card'] = df['Card'].replace('Giant',3)
# df['Card'] = df['Card'].replace('P.E.K.K.A',4)

#Change rarity from text to numbers
df['rarity'] = df['rarity'].replace('common',0)
df['rarity'] = df['rarity'].replace('rare',1)
df['rarity'] = df['rarity'].replace('epic',2)
df['rarity'] = df['rarity'].replace('legendary',3)
df['rarity'] = df['rarity'].replace('champion',4)

#label map

label_map = {0:'common', 1: 'rare', 2: 'epic', 3: 'legendary', 4: 'champion'}
#reverse_label_map = {v: k for k, v in label_map.items()}
df['Card'] = df['Card'].map(label_map)


#Choose specific features
features = ['elixirCost','maxLevel','Usage', 'Win Rate']
df = df.dropna(subset=features)


#Train,Test

X = df[features].values
y = df['rarity'].values

#Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
epochs = 100
losses = []

for i in range(epochs):
    y_prediction = model(X_train)
    loss = loss_fn(y_prediction,y_train)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(f"prediction:{y_test}, actual {X_test}")

    if i % 10 == 0:
        print(f"Epoch {i}: - Loss {loss.item():.4f}")

#Evaluate
model.eval()
with torch.no_grad():
    y_eval = model(X_test)
    predicted = torch.argmax(y_eval, axis=1)

# Calculate accuracy
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Show actual vs predicted names
    print("Actual vs. Predicted:")
    for i in range(len(y_test)):
        actual_label = y_test[i].item()
        predicted_label = predicted[i].item()
        print(f"  Actual: {label_map[actual_label]:<10} → Predicted: {label_map[predicted_label]}")

# Plot loss over epochs
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()