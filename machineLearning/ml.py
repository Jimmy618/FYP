import csv  # python built in module for processing csv file
import torch
from torch.utils.data import Dataset  # , DataLoader
import pandas as pd
# from torchvision import transforms, utils
# import panda

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

    data = pd.concat([datah, dataf])

    # csvDataset is modified from pytorch data loading tutorial
    # source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    class csvDataset(Dataset):
        """testing csv dataset."""

        def __init__(self, csv_file, transform=None):
            """
            Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            """
            #self.csv = open(path, newline='')
            # calculate csv length aka file count
            self.csvLength = sum(1 for row in self.csv)
            self.csv.seek(0)  # reset file pointer to starting position
            self.transform = transform

        def __len__(self):
            return self.csvLength

        def __getitem__(self, idx):
            if torch.istensor(idx):
                idx = idx.tolist()

            # extraction of data to be coded
            if self.transform:
                pass  # sample = self.transform(sample)

        def close():
            pass  # self.csv.close()

    # with open(path, newline='') as csvfile:
    #     csvReader = csv.reader(csvfile)
    #     print(csvReader)
    #     print(sum(1 for row in csvfile))
    #     csvfile.seek(0)
    #
    #     i = 0
    #     for row in csvReader:
    #         print(row)
    #         # print(row[0], row[1], row[2], row[3], row[4])
    #
    #         i += 1
    #         if i == 10:
    #             break
