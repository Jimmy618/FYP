import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv



def plot_bar_chart(x_data, y_data, label_list, xlabel, ylabel, title):
    
    print(title)
    left = x_data
    height = y_data
    tick_label = label_list
    plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green'])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.title(title)
    plt.show()
    
def cal_time_count(time_diff):
    time = {
        "1 min": 60,
        "30 mins": 1800,
        "1 hour": 3600,
        "1 day": 3600*24,
        "1 week": 3600*24*7,
        "1 month": 3600*24*30,
        "1 year": 3600*24*365
    }
    
    time_count = {
        "<1 min": 0,
        "<30 mins": 0,
        "<1 hour": 0,
        "<1 day": 0,
        "<1 week": 0
    }
    
    for i in time_diff:
        i = i.total_seconds()
        if i<= time["1 min"]:
            time_count["<1 min"]+=1
        elif i>time["1 min"] and i <= time["30 mins"]:
            time_count["<30 mins"]+=1
        elif i>time["30 mins"] and i <= time["1 hour"]:
            time_count["<1 hour"]+=1
        elif i>time["1 hour"] and i <= time["1 week"]:
            time_count["<1 week"]+=1
    
    return time_count
    
    
def show_df_groupBy(dfx2):
    li = list(dfx2.groups.keys())  #  get the key list of the group
    print(li)
#     for key, item in dfx2:
# #         print(dfx2. get_group(key))
#         diff = item['failure_time'].diff()
#         print(key)
#         print(item['failure_time'])
#         print(diff)

def test(dfx2):
    df_keys = list(dfx2.groups.keys())
    time_count_dict = {}

    for key, item in dfx2:

        diff = item['failure_time'].diff()
        time_count_dict[key] = cal_time_count(diff)
        
    xlabel = "model"
    ylabel = "failure count"
    plot_stack_bar_chart("graph", df_keys, xlabel, ylabel, time_count_dict)
    

def plot_stack_bar_chart(title, key_list, xlabel, ylabel, time_count_dict):
    labels = key_list
    onemin = []
    halfhr = []
    onehr = []
    oneday = []
    oneweek = []
    for i in time_count_dict.keys():
        onemin.append(time_count_dict[i]["<1 min"])
        halfhr.append(time_count_dict[i]["<30 mins"])
        onehr.append(time_count_dict[i]["<1 hour"])
        oneday.append(time_count_dict[i]["<1 day"])
        oneweek.append(time_count_dict[i]["<1 week"])
    
    
    dict_all = {
        "Model": key_list,
        "<1 min": onemin,
        "<30 mins":halfhr,
        "<1 hour": onehr,
        "<1 day": oneday,
        "<1 week": oneweek
    }
    
    output = pd.DataFrame(data=dict_all)
    output.to_excel('model_analysis.xlsx')
    
    input_file = pd.read_excel('model_analysis.xlsx')
    input_file = input_file.drop(input_file.columns[0], 1)
    
    input_file.plot(
        x = 'Model',
        kind = 'barh',
        stacked = True,
        title = "sdsdf",
        mark_right = True)
    
    
    plt.show()
    plt.clf()



def temporal_analysis_factor(df_fail, factor):
    df_fail['failure_time'] = pd.to_datetime(df_fail['failure_time'])
    df = (df_fail.set_index(['failure_time'])
          .loc['2018-1']
          .reset_index()
          .reindex(columns=df_fail.columns))
    time_diff = df['failure_time'].diff()
    
    time_count = cal_time_count(time_diff)
    xlabel = 'Time interval'
    ylabel = 'Failure count'
    xdata = list(np.arange(1,5+1))
    plot_bar_chart(xdata, time_count.values(), list(time_count.keys()), xlabel, ylabel, "graph")
    
    if factor == "model":
        df_model = df.groupby("model")

        xlabel = 'model'
        ydata = list(df.groupby("model").size())
        xdata = list(np.arange(1,len(ydata)+1))
        
        dfx = df_fail.groupby(["model", "failure_time"])
    
        test(df.sort_values(['model','failure_time']).groupby('model'))

        plot_bar_chart(xdata, ydata, list(df_model.groups.keys()), xlabel, ylabel, 'graph')
    
    
        
        
df_all = pd.read_csv('20191231.csv')
df_fail = pd.read_csv('ssd_failure_tag.csv')
print("Total number of records: "+str(len(df_fail)))
df_locate = pd.read_csv('location_info_of_ssd.csv')



temporal_analysis_factor(df_fail, "model")



    