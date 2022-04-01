import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from collections import Counter



def plot_bar_chart(x_data, y_data, label_list, xlabel, ylabel, title):

    left = x_data
    height = y_data

    tick_label = label_list
    plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green'])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.title(title)
    plt.show()
    
def cal_time_count(time_diff, is_cumulative):
    total_time = 0
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
        "1 min": 0,
        "30 mins": 0,
        "1 hour": 0,
        "1 day": 0,
        "1 week": 0,
        "1 month": 0,
        "1 year":0,
        ">1 year": 0
    }
    if is_cumulative:
        for i in time_diff:
            i = i.total_seconds()
            if(not math.isnan(i)):
                total_time= total_time+i
                
            if i<= time["1 min"]:
                time_count["1 min"]+=1
            if i <= time["30 mins"]:
                time_count["30 mins"]+=1
            if i <= time["1 hour"]:
                time_count["1 hour"]+=1
            if i <= time["1 day"]:
                time_count["1 day"]+=1
            if i <= time["1 week"]:
                time_count["1 week"]+=1
            if i <= time["1 month"]:
                time_count["1 month"]+=1
            if i <= time["1 year"]:
                time_count["1 year"]+=1
    else:    
        for i in time_diff:
            
            i = i.total_seconds()
            if(not math.isnan(i)):
                total_time= total_time+i
            if i<= time["1 min"]:
                time_count["1 min"]+=1
            elif i>time["1 min"] and i <= time["30 mins"]:
                time_count["30 mins"]+=1
            elif i>time["30 mins"] and i <= time["1 hour"]:
                time_count["1 hour"]+=1
            elif i>time["1 hour"] and i <= time["1 day"]:
                time_count["1 day"]+=1
            elif i>time["1 day"] and i <= time["1 week"]:
                time_count["1 week"]+=1
            elif i>time["1 week"] and i <= time["1 month"]:
                time_count["1 month"]+=1
            elif i>time["1 month"] and i <= time["1 year"]:
                time_count["1 year"]+=1
            elif i>time["1 year"]:
                time_count[">1 year"]+=1
                
    return time_count
    
    


def plot_graph_by_df(dfx2, factor, title):
    df_keys = list(dfx2.groups.keys()) #  get the key list of the group
    time_count_dict = {}

    for key, item in dfx2:
#         print(key)
#         print(item)
        if len(item) > 1:
            
            diff = item['failure_time'].diff()
            dict1= cal_time_count(diff, is_cumulative=False)
           
            if key[0] in time_count_dict:
                
                for i in time_count_dict[key[0]]:
                    
                    if i in dict1:
                        time_count_dict[key[0]][i] = time_count_dict[key[0]][i] + dict1[i]
                    else:
                        pass
                
            else:
                time_count_dict[key[0]] = cal_time_count(diff, is_cumulative=False)
            
                
#         for i in time_count_dict[key]:
#             print("key: "+ i)
#             print(time_count_dict[key][i])
        
    
    xlabel = factor
    ylabel = "failure count"
  
    plot_stack_bar_chart(title, list(time_count_dict.keys()), xlabel, ylabel, time_count_dict, factor)
    

def plot_stack_bar_chart(title, key_list, xlabel, ylabel, time_count_dict, factor):
#     plt.rcParams["figure.figsize"] = (68, 66)
    plt.rcParams["figure.figsize"] = (10, 10)
    labels = key_list
    onemin = []
    halfhr = []
    onehr = []
    oneday = []
    oneweek = []
    onemonth = []
    oneyear = []
    oneyearmore = []
    for i in time_count_dict.keys():
        onemin.append(time_count_dict[i]["1 min"])
        halfhr.append(time_count_dict[i]["30 mins"])
        onehr.append(time_count_dict[i]["1 hour"])
        oneday.append(time_count_dict[i]["1 day"])
        oneweek.append(time_count_dict[i]["1 week"])
        onemonth.append(time_count_dict[i]["1 month"])
        oneyear.append(time_count_dict[i]["1 year"])
        oneyearmore.append(time_count_dict[i][">1 year"])
    
    dict_all = {
        factor: key_list,
        "0-1 min": onemin,
        ">1 min and <=30 mins": halfhr,
        ">30 mins and <=1 hour":onehr,
        ">1 hour and <=1 day": oneday,
        ">1 day and <=1 week": oneweek,
        ">1 week and <=1 month": onemonth,
        ">1 month and <=1 year": oneyear,
        ">1 year":oneyearmore
    }
    
    print(len(key_list))
    print(len(onemin))
    
    
    output = pd.DataFrame(data=dict_all)
    output.to_excel(factor+'_analysis.xlsx')
    
    output["sum"] = output.sum(axis=1)
    output["1 min (%)"] = (output["0-1 min"] /output["sum"])*100
    output["30 mins (%)"] = (output[">1 min and <=30 mins"]/output["sum"])*100
    output["1 hour (%)"] = (output[">30 mins and <=1 hour"] /output["sum"])*100
    output["1 day (%)"] = (output[">1 hour and <=1 day"]/output["sum"])*100
    output["1 week (%)"] = (output[">1 day and <=1 week"] /output["sum"])*100
    output["1 month (%)"] = (output[">1 week and <=1 month"] /output["sum"])*100
    output["1 year (%)"] = (output[">1 month and <=1 year"] /output["sum"])*100
    output[">1 year (%)"] = (output[">1 year"] /output["sum"])*100
    
    output.round(2)
    output.to_excel(factor+'_analysis_percentage.xlsx')
    input_file = pd.read_excel(factor+'_analysis.xlsx')
    input_file = input_file.drop(input_file.columns[0], 1)
    
    input_file.plot(
        x = factor,
        kind = 'barh',
        stacked = True,
        title = title,
        mark_right = True)
    
    
    
    plt.show()
    plt.clf()
    
    


def temporal_analysis_factor(df_fail, factor):
    x_label_list = ["0-1 min", ">1 min and <=30 mins", 
                    ">30 mins and <=1 hour", 
                    ">1 hour and <=1 day", ">1 day and <=1 week", 
                    ">1 week and <=1 month", ">1 month and <=1 year", "> 1 year"]
    df_fail['failure_time'] = pd.to_datetime(df_fail['failure_time'])
    start_from = ''
    selected_date = ''
    df = (df_fail.set_index(['failure_time'])
#           .loc[selected_date]
          .reset_index()
          .reindex(columns=df_fail.columns))
#     time_diff = df['failure_time'].diff()
    print("Total number of records for df: "+str(len(df)))
#     time_count = cal_time_count(time_diff, False)
    

    
    
    
    if factor == "model" or factor == "app":
        df_factor = df.groupby(factor)

        xlabel = factor
        ydata = list(df.groupby(factor).size())
        xdata = list(np.arange(1,len(ydata)+1))
        df2 = df.sort_values([factor,'failure_time']).groupby([factor, 'node_id', 'rack_id'])
        time_count_dict = {}
        time_count = {}
        df3 = df.sort_values(['failure_time']).groupby(['node_id', 'rack_id'])
        for key, item in df3:
            
            if len(item) > 1:

                time_diff = item['failure_time'].diff()
                
                dict1 = cal_time_count(time_diff, False)

                if len(time_count.keys()) !=0:

                    for i in time_count:
                        if i in dict1:

                            time_count[i] = time_count[i] + dict1[i]
                        else:
                            pass
                else:
                    time_count = dict1
                    print("first")
                    
        xlabel = 'Time interval'
        ylabel = 'Cumulative Failure Count'
        xdata = list(np.arange(1,len(time_count)+1))
        plt.figure(figsize=(20,20))
        plot_bar_chart(xdata, time_count.values(), x_label_list, xlabel, ylabel, selected_date)
        
                                                              
#         plot_graph_by_df(df2, factor, selected_date)
#         plot_bar_chart(xdata, ydata, list(df_factor.groups.keys()), xlabel, ylabel, "")
        
    elif factor == "r_9":
        df_fac = df.sort_values([factor,'failure_time'])
        print(df_fac)
    
        
        
df_all = pd.read_csv('20191231.csv')
df_fail = pd.read_csv('ssd_failure_tag.csv')
print("Total number of records: "+str(len(df_fail)))
df_locate = pd.read_csv('location_info_of_ssd.csv')


plt.figure(figsize=(16,16))
# temporal_analysis_factor(df_fail, "app")
# spatialSMARTFailure(df_fail)
temporalSMARTFailure(df_fail)



    