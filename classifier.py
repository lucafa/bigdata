#!/usr/bin/env python
#Convert date in a readable format

from datetime import datetime
import csv
import pandas as  pd
import numpy as np
from time import time
import sys
import os


import csv
year = "2018"
with open('alert_simple.csv', 'r') as f, open('alert_edited2.csv', 'a') as f_out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()  # For writing header
    for row in reader:
        if row['timestamp'] != '0':
            data=row['timestamp']
            data = data[:-1]
            data = year + "/" + data
            #print(data)
            a=datetime.strptime(data , '%Y/%m/%d-%H:%M:%S.%f')
            b = datetime(1970, 1, 1)
            microseconds_date = ((a-b).total_seconds())
            #print(microseconds_date)
            data_ok = str(microseconds_date)
            writer.writerow(data_ok)
            
exit()


#Read csv file to convert 
data = pd.read_csv("alert2.csv")
#print(data['timestamp'])
#print((datetime.strptime((data['timestamp']),"%m/%d-%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds())
#data['Timeframe'] = (data['timestamp'])
print(data['timestamp'].dt.hour)
exit()
#a=datetime.strptime(data2 , "%m/%d-%H:%M:%S.%f")
b = datetime(1970, 1, 1)
print(data2, "is type of: ", type(data2))
print(b, "is type of: ", type(b))

#print(-((a-b).total_seconds()))
#print(data2.Timestamp())
exit()


for index, row in data.itertuples():
    #print row["timestamp"], row["src"]
    edit_time=row['timestamp']
    print(edit_time())
    #a = datetime.strptime((row['timestamp']) , "%m/%d-%H:%M:%S.%f")
    #b = datetime(1970, 1, 1)
    #print(-((a-b).total_seconds()))


#data['timestamp'] = ((date.strptime(data['timestamp'],"%m/%d-%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds())
exit()
a = datetime.strptime('10/06-15:39:39.513121' , "%m/%d-%H:%M:%S.%f")
b = datetime(1970, 1, 1)
print(-((a-b).total_seconds()))


