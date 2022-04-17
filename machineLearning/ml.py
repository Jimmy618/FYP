import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
# from torchvision import transforms, utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import warnings

# incase the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    try:
        model = sys.argv[1]
    except IndexError:
        model == 'knn'
    try:
        trainModel = sys.argv[2] == 'train'
    except IndexError:
        trainModel = False

    # torch.autograd.set_detect_anomaly(True)

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
    datah['health'] = 0

    # prepare failure data
    dataf = pd.DataFrame(columns=[colList])
    for colName in assignFaultyList:
        dataf[colName] = failureData[colName]
    dataf['health'] = 1

    # since number of health data is much larger than failure data 706182 > 18387,
    # almost all example feeded to the machine are health data (health label = 1)
    # it will predict every example to be health and still have high accuracy
    # perform undersampling to balance number of data
    datah = datah.sample(n=len(dataf.index)*3)

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

    # reference:
    # https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f
    # https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
    if model == 'knn':
        print("Training k nearest neighbors")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="dropping on a non-lexsorted multi-index without a level parameter may impact performance.")
            warnings.filterwarnings(
                "ignore", message="Feature names only support names that are all strings.")
            warnings.filterwarnings(
                "ignore", message="A column-vector y was passed when a 1d array was expected")

            knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                       algorithm='auto', leaf_size=28, p=2, metric='minkowski')

            x = data.drop(columns=['health'])
            y = data['health']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            print(f'Correct / Total : {tn + tp} / {tn+ fp+ fn+ tp}')
            print(f"Accuracy: {accuracy_score(y_test, y_pred)*100: .2f} %")
            print(f'False positive / Incorrect prediction: {fp}/{fp + fn}')
            print(f'Percentage: {float(fp)/float(fp + fn)*100: .2f} %')
            print(f'False negative / Incorrect prediction: {fn}/{fp + fn}')
            print(f'Percentage: {float(fn)/float(fp + fn)*100: .2f} %')

            hypertrain = False

            # below is copy of hyperparameters training in the reference
            if hypertrain:
                leaf_size = list(range(28, 34))
                n_neighbors = list(range(4, 7))
                hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors)
                knn_2 = KNeighborsClassifier()
                clf = GridSearchCV(knn_2, hyperparameters)
                best_model = clf.fit(x, y)
                print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
                print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

            if hypertrain:
                for i in range(25, 36):
                    knn = KNeighborsClassifier(n_neighbors=6, weights='uniform',
                                               algorithm='auto', leaf_size=i, p=2, metric='minkowski')
                    x = data.drop(columns=['health'])
                    y = data['health']
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

                    knn.fit(x_train, y_train)
                    y_pred = knn.predict(x_test)
                    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100: .2f} i = {i}")

    # reference:
    # https://scikit-learn.org/stable/modules/tree.html
    if model == 'dt':
        print("Training decision trees")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="dropping on a non-lexsorted multi-index without a level parameter may impact performance.")
            warnings.filterwarnings(
                "ignore", message="Feature names only support names that are all strings.")
            decision_tree = DecisionTreeClassifier(
                criterion='gini', random_state=0, min_samples_leaf=3, max_depth=9)
            x = data.drop(columns=['health'])
            y = data['health']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            decision_tree.fit(x_train, y_train)
            y_pred = decision_tree.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

            print(f'Correct / Total : {tn + tp} / {tn+ fp+ fn+ tp}')
            print(f"Accuracy: {accuracy_score(y_test, y_pred)*100: .2f} %")
            print(f'False positive / Incorrect prediction: {fp}/{fp + fn}')
            print(f'Percentage: {float(fp)/float(fp + fn)*100: .2f} %')
            print(f'False negative / Incorrect prediction: {fn}/{fp + fn}')
            print(f'Percentage: {float(fn)/float(fp + fn)*100: .2f} %')

            hypertrain = True

            if hypertrain:
                for i in range(5, 11):
                    decision_tree = DecisionTreeClassifier(
                        criterion='gini', random_state=0, min_samples_leaf=i, max_depth=8)
                    x = data.drop(columns=['health'])
                    y = data['health']
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

                    decision_tree.fit(x_train, y_train)
                    y_pred = decision_tree.predict(x_test)
                    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100: .2f} i = {i}")

    # ssdDataset is modified from pytorch data loading tutorial
    # source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    if model == 'mlp':
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
                  nn.Linear(32, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 128), nn.ReLU(),
                  nn.Linear(128, 1), nn.Sigmoid()  # sigmoid function to convert result to [0,1]
                )

            def forward(self, x):
                '''Forward pass'''
                return self.layers(x)

        if trainModel:
            print("Training Multilayer Perceptron")
            train_batch_size = 512
            train = data.sample(frac=0.8).reset_index(drop=True)
            # train = data.reset_index(drop=True)
            samples_weight = [1.1 if row['model'] == 9 else 1 for index, row in train.iterrows()]
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_set = ssdDataset(train)
            # train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
            train_loader = DataLoader(train_set, batch_size=train_batch_size, sampler=sampler)
            model = MLP()

            loss_function = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(0, 3):
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
            print('Calculating Accuracy of mlp model')

            test = data.sample(frac=0.5).reset_index(drop=True)
            test_set = ssdDataset(test)
            test_loader = DataLoader(test_set, batch_size=1024, shuffle=True)

            model = MLP()
            model.load_state_dict(torch.load('./model.pt'))

            # failed is 1, health is 0
            correct_num = 0
            false_positive = 0
            false_negative = 0
            sample_num = len(test.index)
            model.eval()

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, targets = data
                    outputs = model(inputs.float())
                    correct_num += (outputs.round() == targets.unsqueeze(1)).float().sum().item()
                    false_positive += (outputs.round() > targets.unsqueeze(1)).float().sum().item()
                    false_negative += (outputs.round() < targets.unsqueeze(1)).float().sum().item()

                incorrect_num = sample_num-correct_num
                print(f'Correct / Total : {correct_num} / {sample_num}')
                print(f'Percentage: {float(correct_num)/float(sample_num)*100: .2f} %')
                print(f'False positive / Incorrect prediction: {false_positive}/{incorrect_num}')
                print(f'Percentage: {float(false_positive)/float(incorrect_num)*100: .2f} %')
                print(f'False negative / Incorrect prediction: {false_negative}/{incorrect_num}')
                print(f'Percentage: {float(false_negative)/float(incorrect_num)*100: .2f} %')
