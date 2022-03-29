import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import SGD
# from torchvision import transforms, utils

# prepate dataset:
# forget locaiton.csv, just use ssd_failure_tag.csv as failed and 20191231.csv as functional
# shape two into unified.
# combine some SMART in functional data
# combine two .csv in one
# smartData.join(locationData.set_index('disk_id').drop('model', axis=1), on='disk_id')

# ML

# incase the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    batch_size = 32

    print("Preparing dataset")
    dataPath = r'../data/'
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')

    # 'n_blocks' 'n_170', 'n_180'
    # 'r_program', 'n_program' 'n_171', 'n_181'
    # 'r_erase', 'n_erase' 'n_172', 'n_182'
    # 'n_wearout', 'n_173' 'n_177', 'n_233'
    colList = ['model', 'r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
               'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199',
               'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout',
               'r_241', 'n_241', 'r_242', 'n_242', 'r_9', 'n_9',
               'r_12', 'n_12', 'r_174', 'n_174', 'n_175', 'health']
    assignHealthyList = ['model', 'r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
                         'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199',
                         'r_241', 'n_241', 'r_242', 'n_242', 'r_9', 'n_9',
                         'r_12', 'n_12', 'r_174', 'n_174', 'n_175']
    assignFaultyList = ['model', 'r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
                        'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199',
                        'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout',
                        'r_241', 'n_241', 'r_242', 'n_242', 'r_9', 'n_9',
                        'r_12', 'n_12', 'r_174', 'n_174', 'n_175']

    datah = pd.DataFrame(columns=[colList])
    for colName in assignHealthyList:
        datah[colName] = smartData[colName]
    datah['n_blocks'] = smartData[['n_170', 'n_180']].max(axis=1)
    datah['r_program'] = smartData[['r_171', 'r_181']].max(axis=1)
    datah['n_program'] = smartData[['n_171', 'n_181']].max(axis=1)
    datah['r_erase'] = smartData[['r_172', 'r_182']].max(axis=1)
    datah['n_erase'] = smartData[['n_172', 'n_182']].max(axis=1)
    datah['n_wearout'] = smartData[['n_173', 'n_177', 'n_233']].max(axis=1)
    datah['health'] = 1

    dataf = pd.DataFrame(columns=[colList])
    for colName in assignFaultyList:
        dataf[colName] = failureData[colName]
    dataf['health'] = 0

    data = pd.concat([datah, dataf])  # .reset_index(drop=True)
    data['model'] = data['model'].replace(
        ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'C1', 'C2'],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # data = data.astype('float')
    data = data.fillna(-1)
    data = data.astype(float)

    # train, test = random_split(data, [3, len(data.index)-3])
    train = data.sample(frac=0.8, random_state=200).reset_index(drop=True)
    test = data.drop(train.index).sample(frac=1.0).reset_index(drop=True)

    # csvDataset is modified from pytorch data loading tutorial
    # source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    class ssdDataset(Dataset):
        """testing csv dataset."""

        def __init__(self, dataFrame, transform=None):
            """
            Args:
            dataFrame (panda dataFrame): source of ssd data, prepared.
            transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.dataFrame = dataFrame
            self.transform = transform

        def __len__(self):
            return len(self.dataFrame.index)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            #sample = self.dataFrame.loc[idx].drop(columns='health', level=0)
            sample = self.dataFrame.loc[idx][colList[:-1]]
            sample = torch.tensor(sample)
            label = self.dataFrame.loc[idx]['health']
            label = torch.tensor(label)
            if self.transform:
                sample = self.transform(sample)

            return sample, label

    # https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
    class MLP(nn.Module):
        '''
        Multilayer Perceptron.
        '''

        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
              nn.Flatten(),
              nn.Linear(32, 64),
              nn.ReLU(),
              nn.Linear(64, 32),
              nn.ReLU(),
              nn.Linear(32, 10),
              nn.ReLU(),
              nn.Linear(10, 1),
              nn.Sigmoid()
            )

        def forward(self, x):
            '''Forward pass'''
            return self.layers(x)

    # data_set = ssdDataset(data)
    # data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # inputs, classes = next(iter(data_loader))
    train_set = ssdDataset(train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = ssdDataset(test)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=True)
    # define the network
    model = MLP()
    # model.double()

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 1):  # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        # Set current loss value
        current_loss = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs.float())

            # Compute loss
            loss = loss_function(outputs, targets.unsqueeze(1).float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
