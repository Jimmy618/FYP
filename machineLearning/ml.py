import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
# from torchvision import transforms, utils

# ML

# incase the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    try:
        trainModel = sys.argv[1] == 'train'
    except IndexError:
        trainModel = False

    # torch.autograd.set_detect_anomaly(True)
    train_batch_size = 1024

    print("Preparing dataset")
    dataPath = r'../data/'
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')
    # location data is not used in ML because the location information (rank_id, disk_id, ...)
    # are not provided in Healthy SSDs data (smartData)
    # Although disk_id is indeed provided, serveral entries in 20191231.csv share the same disk_id,
    # presumably the same disk location is replaced serveral time or disk_id is resused in different location
    # so disk_id is not a key to join and find the locaiton
    # locationData = pd.read_csv(dataPath + 'location_info_of_ssd.csv')

    # since only portion of SMART data are provided in ssd_failure_tag.csv,
    # only correponding part of information in 20191231.csv are used
    # The provide SMART, as specificied in ssd_open_data READMO.md,
    # 'n_blocks': 'n_170', 'n_180'
    # 'r_program': 'n_program' 'n_171', 'n_181'
    # 'r_erase' 'n_erase': 'n_172', 'n_182'
    # 'n_wearout', 'n_173': 'n_177', 'n_233'
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

    # prepare health data
    # convert healthy data, use max approach
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

    # prepare failure data
    dataf = pd.DataFrame(columns=[colList])
    for colName in assignFaultyList:
        dataf[colName] = failureData[colName]
    dataf['health'] = 0

    # since number of health data is much larger than failure data 706182 > 18387,
    # almost all example feeded to the machine are health data (health label = 1)
    # it will predict every example to be health and still have high accuracy
    # perform undersampling to balance number of data
    datah = datah.sample(n=len(dataf.index))

    # join both data in one dataset
    data = pd.concat([datah, dataf])  # .reset_index(drop=True)
    data['model'] = data['model'].replace(
        ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'C1', 'C2'],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # treat nan as value -1
    data = data.fillna(-1).astype(float).reset_index(drop=True)

    # data_set = ssdDataset(data)
    # data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    # inputs, classes = next(iter(data_loader))

    # ssdDataset is modified from pytorch data loading tutorial
    # source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    class ssdDataset(Dataset):
        def __init__(self, dataFrame, transform=None):
            """
            Args:
            dataFrame (panda dataFrame): source of ssd data, prepared.
            transform (callable, optional): Optional transform to be applied on a sample.
            """
            # since most pre-processing are done before hand, not much changed are needed
            self.dataFrame = dataFrame
            self.transform = transform

        def __len__(self):
            return len(self.dataFrame.index)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            # sample = self.dataFrame.loc[idx].drop(columns='health', level=0)
            # remove the label 'health' from the example
            sample = self.dataFrame.loc[idx][colList[:-1]]
            sample = torch.tensor(sample)
            label = self.dataFrame.loc[idx]['health']
            label = torch.tensor(label)
            if self.transform:
                sample = self.transform(sample)

            return sample, label

    # define the network
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
              nn.Sigmoid()  # sigmoid function to convert result to [0,1]
            )

        def forward(self, x):
            '''Forward pass'''
            return self.layers(x)

    if trainModel:
        print("Training Neural Network")
        train = data.sample(frac=0.8).reset_index(drop=True)
        train_set = ssdDataset(train)
        train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        model = MLP()

        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(0, 1):
            print(f'Starting epoch {epoch+1}')
            current_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = loss_function(outputs, targets.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                    current_loss = 0.0

        print('Training process has finished.')
        torch.save(model.state_dict(), './model.pt')

    if not trainModel:
        print('Calculating Accuracy')

        test = data.sample(frac=0.1).reset_index(drop=True)
        test_set = ssdDataset(test)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=True)

        model = MLP()
        model.load_state_dict(torch.load('./model.pt'))

        correct_num = 0
        sample_num = len(test.index)
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, targets = data
                outputs = model(inputs.float())
                correct_num += (outputs.round() == targets.unsqueeze(1)).float().sum().item()

            print(f'Correct / Total : {correct_num} / {sample_num}')
            print(f'Percentage: {float(correct_num)/float(sample_num)*100: .2f} %')
