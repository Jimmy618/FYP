import pandas as pd
import matplotlib.pyplot as plt

showGraph = False
to_csv = True
dataPath = r'../data/'
resultPath = r'../result/'


def composition(columnName, data, failureData):
    # this function intent to analysis composition of categorical data

    totalresult = data.groupby(
        columnName).size().to_frame('size')
    totalcount = data[columnName].count()
    totalresult['size / sizeTotal'] = totalresult['size'] / totalcount

    result = failureData.groupby(columnName).size().to_frame('failure size')
    count = failureData[columnName].count()
    result['failure size / failure sizeTotal'] = result['failure size'] / count

    output = totalresult.reset_index().merge(result.reset_index(), on=columnName)
    output['failure size / size'] = output['failure size'] / output['size']

    if to_csv:
        output.to_csv(resultPath + 'composition' + '_' + columnName + '.csv')

    if showGraph:
        # result.plot.pie(y='sizePercentage')
        plt.show()


# incase the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':

    locationData = pd.read_csv(dataPath + 'location_info_of_ssd.csv')
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')

    composition('model', locationData, failureData)
    composition('app', locationData, failureData)

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
