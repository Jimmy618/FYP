import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

showGraph = False
to_csv = True
dataPath = r'../data/'
resultPath = r'../result/'


def cateComposition(columnName, locationData, failureData):
    # this function intent to analysis composition of categorical data

    print("processing", columnName, "composition")
    
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


def numComposition(colList, smartData, failureData):
    print("processing SMART composition")
    
    smart_df = pd.DataFrame(failureData, columns=colList)
    smart_quantile = smart_df.quantile(q=[0.25, 0.5, 0.75]).T.reset_index()
    smart_quantile = smart_quantile.rename(columns={0.25:'0.25 quantile',
    0.5:'0.5 quantile',
    0.75:'0.75 quantile'})
    smart_mean = smart_df.mean().reset_index()
    smart_mean = smart_mean.rename(columns={0:'mean'})
    smart_std = smart_df.std().reset_index()
    smart_std = smart_std.rename(columns={0:'std'})
    
    dataframes = [smart_quantile, smart_mean, smart_std]

    result = reduce(lambda  left,right: pd.merge(left,right,on=['index']), dataframes)

    working_smart_df = pd.DataFrame(smartData, columns=colList)
    working_smart_quantile = working_smart_df.quantile(q=[0.25, 0.5, 0.75]).T.reset_index()
    working_smart_quantile = working_smart_quantile.rename(columns={0.25:'0.25 quantile (functional)',
    0.5:'0.5 quantile (functional)',
    0.75:'0.75 quantile (functional)'})
    working_smart_mean = working_smart_df.mean().reset_index()
    working_smart_mean = working_smart_mean.rename(columns={0:'mean (functional)'})
    working_smart_std = working_smart_df.std().reset_index()
    working_smart_std = working_smart_std.rename(columns={0:'std (functional)'})
    
    working_dataframes = [working_smart_quantile, working_smart_mean, working_smart_std]

    working_result = reduce(lambda  left,right: pd.merge(left,right,on=['index']), working_dataframes)

    output = result.merge(working_result, on='index')

    if to_csv:
        output.to_csv(resultPath + 'composition' + '_' + 'SMART' + '.csv')


# incase the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    print("preparing csv")
    locationData = pd.read_csv(dataPath + 'location_info_of_ssd.csv')
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')

    cateComposition('model', locationData, failureData)
    cateComposition('app', locationData, failureData)

    colList = ['r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
               'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199', 
               # 'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout', 
               'r_241', 'n_241','r_242', 'n_242', 'r_9', 'n_9', 
               'r_12', 'n_12', 'r_174', 'n_174', 'n_175']

    numComposition(colList, smartData, failureData)

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
