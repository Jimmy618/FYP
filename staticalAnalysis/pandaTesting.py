import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

showGraph = False
to_csv = not(showGraph)
dataPath = r'../data/'
resultPath = r'../result/'
documentPrefix = 1

# since not every SMART is reported in failureData,
# only selection of SMART data is used in analysis.
colList = ['r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
           'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199',
           # to be added
           # 'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout',
           'r_241', 'n_241', 'r_242', 'n_242', 'r_9', 'n_9',
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

    dataframes = [totalresult, result]
    output = pd.concat(dataframes, axis=1)
    output['failure size / (size + failure size)'] = output['failure size'] / output['size']

    if to_csv:
        global documentPrefix
        output.to_csv(resultPath + str(documentPrefix) + '_'
                      + 'composition' + '_' + columnName + '.csv')
        documentPrefix += 1

    if showGraph:
        plot = output['size / sizeTotal'].plot.pie(subplots=True,
                                                   layout=(1, 1), figsize=(10, 10), autopct='%1.1f%%')
        plt.show()
        plot = output['failure size / failure sizeTotal'].plot.pie(
            subplots=True, layout=(1, 1), figsize=(10, 10), autopct='%1.1f%%')
        plt.show()


def modelAppComposition(locationData, failureData):
    # to be completed
    # this function intent to analyse composition of model to app data

    totalresult = locationData.groupby(['model', 'app']).size().to_frame('size')
    #totalcount = locationData[['model', 'app']].count()
    #totalresult['size / sizeTotal'] = totalresult['size'] / totalcount

    result = failureData.groupby(['model', 'app']).size().to_frame('failure size')
    #count = failureData[['model', 'app']].count()
    #result['failure size / failure sizeTotal'] = result['failure size'] / count


def extractSMARTHelper(df, postfix):
    smart_df = pd.DataFrame(df, columns=colList)
    smart_quantile = smart_df.quantile(q=[0.25, 0.5, 0.75]).T
    smart_quantile = smart_quantile.rename(columns={0.25: '0.25 quantile' + postfix,
                                                    0.5: '0.5 quantile' + postfix,
                                                    0.75: '0.75 quantile' + postfix})
    smart_mean = smart_df.mean()
    smart_mean = smart_mean.rename('mean' + postfix)
    smart_std = smart_df.std()
    smart_std = smart_std.rename('std' + postfix)

    dataframes = [smart_quantile, smart_mean, smart_std]

    return pd.concat(dataframes, axis=1)


def numComposition(smartData, failureData):
    # this function intent to analyse composition of numerial data
    print("Processing SMART composition")

    result = extractSMARTHelper(failureData, '')
    working_result = extractSMARTHelper(smartData, ' (functional)')

    output = pd.concat([result, working_result], axis=1)

    if to_csv:
        global documentPrefix
        output.to_csv(resultPath + str(documentPrefix) + '_'
                      + 'composition' + '_' + 'SMART' + '.csv')
        documentPrefix += 1


def categoricalSMART(columnName, smartData, failureData):
    # this function intent to analyse SMART of each instance of the categorical data
    print("Processing", columnName + "'s", "SMART")

    smart_df = smartData.groupby(columnName)
    smart_fail_df = failureData.groupby(columnName)
    cateDataFrameList = []
    catelist = []

    # assume both smartData and failureData share the same 'columnName' categorical variable
    for key, item in smart_df:
        result = extractSMARTHelper(smart_df.get_group(key), '')
        functional_result = extractSMARTHelper(smart_df.get_group(key), ' (functional)')

        dataFrame = pd.concat([result, functional_result], axis=1)

        cateDataFrameList.append(dataFrame)
        catelist.append(key)

    # combine the dataframe together
    output = pd.concat(cateDataFrameList, keys=catelist)

    if to_csv:
        global documentPrefix
        output.to_csv(resultPath + str(documentPrefix) + '_' + 'group_SMART_' + columnName + '.csv')
        documentPrefix += 1

    if showGraph:
        for column in colList:
            fig, ax = plt.subplots()
            for key, item in smart_fail_df:
                df = smart_fail_df.get_group(key)[column].dropna(
                ).sort_values().reset_index(drop=True)
                df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                    columns={column: column + ' ' + key}).plot(ax=ax)
            df = failureData[column].dropna().sort_values().reset_index(drop=True)
            df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                columns={column: column + ' ' + 'all'}).plot(ax=ax)
            plt.show()

            fig, ax = plt.subplots()
            for key, item in smart_df:
                df = smart_df.get_group(key)[column].dropna().sort_values().reset_index(drop=True)
                df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                    columns={column: column + ' ' + key + '(f)'}).plot(ax=ax)
            df = smartData[column].dropna().sort_values().reset_index(drop=True)
            df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                columns={column: column + ' ' + 'all' + '(f)'}).plot(ax=ax)
            plt.show()


def partialCategoricalSMART(columnName, failureData):
    # this function intent to analyse SMART of each instance of the categorical data
    # this function is provided because 'app' does not have functional data in all there dataset provided
    print("Processing", columnName + "'s", "SMART")

    smart_fail_df = failureData.groupby(columnName)
    cateDataFrameList = []
    catelist = []

    # assume both smartData and failureData share the same 'columnName' categorical variable
    for key, item in smart_fail_df:
        result = extractSMARTHelper(smart_fail_df.get_group(key), '')

        cateDataFrameList.append(result)
        catelist.append(key)

    # combine the dataframe together
    output = pd.concat(cateDataFrameList, keys=catelist)

    if to_csv:
        global documentPrefix
        output.to_csv(resultPath + str(documentPrefix) + '_' + 'group_SMART_' + columnName + '.csv')
        documentPrefix += 1

    if showGraph and False:
        for column in colList:
            fig, ax = plt.subplots()
            for key, item in smart_fail_df:
                df = smart_fail_df.get_group(key)[column].dropna(
                ).sort_values().reset_index(drop=True)
                df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                    columns={column: column + ' ' + key}).plot(ax=ax)
            df = failureData[column].dropna().sort_values().reset_index(drop=True)
            df.to_frame().set_index(np.linspace(0.0, 1.0, df.count())).rename(
                columns={column: column + ' ' + 'all'}).plot(ax=ax)
            plt.show()


def spatialFailureGroup(failureData):
    print("Processing", 'Spatial failure group')
    # idea: particular disk position may cause more failure due to moisture, heat, ...
    #       so find out failure distribution by *_id
    failure_group = failureData.set_index(['machine_room_id', 'rack_id', 'node_id']).sort_index()
    machine_room_id_count_occurrence = failure_group.groupby(level=[0]).size().value_counts(
    ).sort_index().rename(' machine_room_id Occurrence').rename_axis('machine_room_id Count')
    rack_id_count_occurrence = failure_group.groupby(level=[0, 1]).size().value_counts(
    ).sort_index().rename('rack_id Occurrence').rename_axis('rack_id Count')
    node_id_count_occurrence = failure_group.groupby(level=[0, 1, 2]).size().value_counts(
    ).sort_index().rename('node_id Occurrence').rename_axis('node_id Count')
    output = pd.concat([machine_room_id_count_occurrence.reset_index(
    ), rack_id_count_occurrence.reset_index(), node_id_count_occurrence.reset_index()], axis=1)

    if to_csv:
        global documentPrefix
        document_name = resultPath + str(documentPrefix) + '_' + 'spatial_failure_group' + '.csv'
        output.to_csv(document_name)
        documentPrefix += 1

    if showGraph:
        ax = machine_room_id_count_occurrence.plot.bar()
        ax.set_xlabel('Occurrence')
        ax.set_ylabel('machine_room_id Count')
        plt.show()
        ax = rack_id_count_occurrence.plot.bar()
        ax.set_xlabel('Occurrence')
        ax.set_ylabel('rack_id Count')
        plt.show()
        ax = node_id_count_occurrence.plot.bar()
        ax.set_xlabel('Occurrence')
        ax.set_ylabel('node_id Count')
        plt.show()


def spatialCategoricalFailure(failureData, columnName):
    print("Processing", 'Spatial categorical failure group')
    # idea: particular model/app in that particular position may fail more often,
    #       so failure distribution by *_id and group by model/app
    other = 'app' if columnName == 'model' else 'model'
    dropList = ['failure_time', 'failure', other, 'disk_id',
                'r_5', 'n_5', 'r_183', 'n_183', 'r_184', 'n_184', 'r_187', 'n_187',
                'r_195', 'n_195', 'r_197', 'n_197', 'r_199', 'n_199',
                'r_program', 'n_program', 'r_erase', 'n_erase', 'n_blocks', 'n_wearout',
                'r_241', 'n_241', 'r_242', 'n_242', 'r_9', 'n_9',
                'r_12', 'n_12', 'r_174', 'n_174', 'n_175']
    failure_group_c = failureData.drop(dropList, axis=1).set_index(
        ['machine_room_id', 'rack_id', 'node_id', columnName]).sort_index()

    failure_group_c['count'] = failure_group_c.groupby(level=[0, 1, 2, 3]).size()
    # discard group with small size, not needed for now
    # failure_group_c = failure_group_c[(failure_group_c['count'] > 1)]
    output = failure_group_c.reset_index().drop_duplicates()

    if to_csv:
        global documentPrefix
        document_name = resultPath + str(documentPrefix) + '_' + \
            'spatial_categorical_failure_' + columnName + '.csv'
        output.to_csv(document_name)
        documentPrefix += 1


def spatialSMARTFailure(failureData):
    print("Processing", 'Spatial SMART failure group')
    # idea: particular position trigger specific SMART and cause failure,
    #       so check for speciality in SMART, also look for similarity with correlation analysis

    dropList = ['failure_time', 'failure', 'model', 'app', 'disk_id']
    failure_group_s = failureData.drop(dropList, axis=1).set_index(
        ['machine_room_id', 'rack_id', 'node_id']).sort_index()

    failure_group_s['count'] = failure_group_s.groupby(level=[0, 1, 2]).size()
    # using failure_group_s as baseline, check for group with at least a number of disk using correlation
    list = []
    for i in range(1, 16):
        target = failure_group_s[failure_group_s['count'] == i]
        result = failure_group_s.corrwith(target, method='pearson')
        result = result.rename('SMART(size='+str(i)+')').rename_axis('SMART')
        list.append(result)
    output = pd.concat(list, axis=1)

    if to_csv:
        global documentPrefix
        document_name = resultPath + str(documentPrefix) + '_' + \
            'spatial_SMART_failure'+'.csv'
        output.to_csv(document_name)
        documentPrefix += 1


# In case the program is running in conda, check for '__main__'
# to ensure it run.
if __name__ == '__main__':
    print("Preparing csv")
    locationData = pd.read_csv(dataPath + 'location_info_of_ssd.csv')
    smartData = pd.read_csv(dataPath + '20191231.csv')
    failureData = pd.read_csv(dataPath + 'ssd_failure_tag.csv')

    # part 3.1 Basic analysis
    # 3.1.1
    categoricalComposition('model', locationData, failureData)
    categoricalComposition('app', locationData, failureData)
    numComposition(smartData, failureData)
    # 3.1.2
    # extract relation between differenct model and application
    categoricalSMART('model', smartData, failureData)
    partialCategoricalSMART('app', failureData)

    # part 3.2 Spatial analysis
    # 3.2.1 spatialFailureGroup
    spatialFailureGroup(failureData)
    # 3.2.2
    spatialCategoricalFailure(failureData, 'model')
    spatialCategoricalFailure(failureData, 'app')
    # 3.2.3
    spatialSMARTFailure(failureData)

    print("Analysis finished.")
