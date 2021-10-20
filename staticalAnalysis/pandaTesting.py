import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

showGraph = False
to_csv = True
dataPath = r'../data/'
resultPath = r'../result/'

# since not every SMART is reported in failureData, 
# only selection of SMART data is used in analysis.
colList = ['r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
           'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199', 
           # to be added
           # 'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout', 
           'r_241', 'n_241','r_242', 'n_242', 'r_9', 'n_9', 
           'r_12', 'n_12', 'r_174', 'n_174', 'n_175']

def categoricalComposition(columnName, locationData, failureData):
    # this function intent to analyse composition of categorical data

    print("Processing", columnName, "composition")
    
    totalresult = locationData.groupby(columnName).size().to_frame('size')
    totalcount = locationData[columnName].count()
    totalresult['size / sizeTotal'] = totalresult['size'] / totalcount

    result = failureData.groupby(columnName).size().to_frame('failure size')
    count = failureData[columnName].count()
    result['failure size / failure sizeTotal'] = result['failure size'] / count

    output = totalresult.reset_index().merge(result.reset_index(), on=columnName)
    output['failure size / (size + failure size)'] = output['failure size'] / output['size']

    if to_csv:
        output.to_csv(resultPath + 'composition' + '_' + columnName + '.csv')

    if showGraph:
        # result.plot.pie(y='sizePercentage')
        plt.show()

def extractSMARTHelper(df, postfix):
    smart_df = pd.DataFrame(df, columns=colList)
    smart_quantile = smart_df.quantile(q=[0.25, 0.5, 0.75]).T
    smart_quantile = smart_quantile.rename(columns={0.25:'0.25 quantile' + postfix,
    0.5:'0.5 quantile' + postfix,
    0.75:'0.75 quantile' + postfix})
    smart_mean = smart_df.mean()
    smart_mean = smart_mean.rename('mean' + postfix)
    smart_std = smart_df.std()
    smart_std = smart_std.rename('std' + postfix)
    
    dataframes = [smart_quantile, smart_mean, smart_std]

    return pd.concat(dataframes, axis=1)

    # old code
    #return reduce(lambda  left,right: pd.merge(left,right,on=['index']), dataframes)

def numComposition(smartData, failureData):
    # this function intent to analyse composition of numerial data
    print("Processing SMART composition")
    
    result = extractSMARTHelper(failureData, '')
    working_result = extractSMARTHelper(smartData, ' (functional)')

    #output = result.merge(working_result, on='index')
    output = pd.concat([result, working_result], axis=1)

    if to_csv:
        output.to_csv(resultPath + 'composition' + '_' + 'SMART' + '.csv')

def categoricalSMART(columnName, smartData, failureData):
    # this function intent to analyse SMART of each instance of the categorical data

    print("Processing", columnName + "'s", "SMART")

    smart_df = smartData.groupby(columnName)
    smart_fail_df = failureData.groupby(columnName)
    temList = []
    list = []
    # assume both smartData and failureData share the same 'columnName' categorical variable
    for key, item in smart_df:
        # print(key)
        # print(smart_df.get_group(key))
        # print(smart_fail_df.get_group(key))
        fail_result = extractSMARTHelper(smart_df.get_group(key), '')
        result = extractSMARTHelper(smart_df.get_group(key), ' (functional)')
        #output = fail_result.merge(result, on='index')
        output = pd.concat([fail_result, result], axis=1)
        #print(output)
        temList.append(output)
        list.append(key)
        print(len(temList))
    print(temList)
    print(list)
    
    df = pd.concat(temList, keys=list)
    print(df)
    
    if to_csv:
        df.to_csv(resultPath + columnName + '\'s ' + 'group ' + 'SMART' + '.csv')
    
    # tem = pd.DataFrame(df.get_group('A1'), columns=colList)
    # result = extractsmarthelper(collist, smartdata, '')
    # print(result)
    # result = extractsmarthelper(collist, df.get_group('a1'), '')
    # print(result)
    # result = extractsmarthelper(collist, df.get_group('a2'), '')
    # print(result)


# In case the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    print("Preparing csv")
    locationData = pd.read_csv(dataPath + 'location_info_of_ssd.csv')
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')

    categoricalComposition('model', locationData, failureData)
    categoricalComposition('app', locationData, failureData)

    numComposition(smartData, failureData)

    # extract relation between differenct model and application
    categoricalSMART('model', smartData, failureData)

    # print(result)

    # df.n_5.dropna().sort_values().reset_index(drop=True).add_subplot()

    # discard header
    # df = pd.read_csv(path, skiprows = 1)
    # deal with missing value?
    # df = pd.read_csv(path, skiprows=1, na_values=['no info', '.'])

    # selecting specific columns
    # df = pd.DataFrame(df, columns=['app', 'rack_id'])
    # print(df[0:11])

    # panda guide reference :
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
    # print(df['app'].dtypes)
    # print(df['rack_id'].dtypes)
    # print(df['node_id'].dtypes)
    # print(df.head(5))

    # sorting
    # df.sort_index(axis=1, ascending=False)
    # df.sort_values(by="B")

    # selection:
    # df["A"] or df.A
    # df[0:3] or df["20130102":"20130104"]
    # selection by label
    # df.loc[dates[0]]
    # df.loc[:, ["A", "B"]]
    # For getting fast access to a scalar
    # df.at[dates[0], "A"]
    # selection by position
    # df.iloc[3]
