import pandas as pd
import numpy as np
import datetime
def holidays(freq, start="090117",end="040118"):
    dates = []
    Day_to_num = dict()
    month_to_num = dict()
    days = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(" ")
    months = "January February March April May June July August September October November December".split(" ")

    for i in range(len(months)):
        k = ''
        if i<9:
            k+="0"
        month_to_num[months[i]] = k+str(i+1)
    for i in range(len(days)):
        Day_to_num[days[i]] = str(i+1)

    with open('holidays.txt','r') as f:
        for line in f:
            Day, month, date_num, year = line.split(":")[0].split(" ")
            Day = Day_to_num[Day]
            month = month_to_num[month]
            dates.append(year+"-"+month+"-"+Day)
    holidays = pd.to_datetime(dates, format="%Y-%m-%d")
    #holidays = pd.DataFrame(np.ones(len(holidays)),index=holidays)
    dates = pd.date_range(start=pd.to_datetime('090117'), end = pd.to_datetime('040118'),freq="1d") #freq=str(freq)+"s")
    df = pd.DataFrame(np.zeros(len(dates)), index = dates, columns=["val"])
    df.loc[df.index.isin(holidays)] = 1
    df=df.resample(str(freq)+"s").mean().ffill()
    return df.val.values
