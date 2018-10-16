#!/usr/bin/env python
#Convert date in a readable format
#****/
#    /
#    Write by Luca Facchin
#                        /
#                        /
#                        /
#TODO: Fix IndexError: list index out of range error
from datetime import datetime
import csv
import pandas as  pd
import numpy as np
from time import time
import sys
import os
import csv

#Variable declaration
x=1
year = "2018"


# Main Program
#Open file in read & write mode
with open('alert_simple.csv', 'r+') as f, open('alert_edited2.csv', 'a') as f_out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()  # For writing header
    data = [r for r in reader]
#While to edit row by row into the csv file
    while data:
        tempo = data[x]['timestamp']
        tempo = year+tempo
        #remove last charachter of the timestamp if it's a space character 
        if tempo.endswith(' '):
            tempo=tempo[:-1]
        #copy seconds value on new variable
        new_tempo = (((datetime.strptime(tempo,"%Y%m/%d-%H:%M:%S.%f")-datetime(1970,1,1)).total_seconds()))
        data[x]['timestamp'] = new_tempo
        #Write values on new file
        writer.writerow(data[x])
        x += 1

            
exit()
