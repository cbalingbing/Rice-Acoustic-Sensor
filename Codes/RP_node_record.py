#!/usr/bin/python3
import os
import time
from datetime import datetime
import logging
import csv
import configparser

# connect devices, other housekeeping
config = configparser.ConfigParser()

##################
# VARIABLE SETTING
##################

# load configuration from file
config.read('config_record.ini')
path = config['RECORD']['record_path']
log_path = config['LOG']['run_log']

# set logging
logging.basicConfig(filename=log_path, level=logging.DEBUG)

# obtain time-based data
ct = time.time()

#####################
# FUNCTION DEFINITION
#####################

def start_log():
   logging.info('Record Start: %s', datetime.now())

def end_log():
   logging.info('Record End: %s', datetime.now())
   logging.info('File : %s-recording.wav, Temperature:, Humidity:', ct)
   print('log saved...')

def write_csv():
   f = open(config['CSV']['csv_path'],'a')
   writer = csv.writer(f)
   row = [datetime.fromtimestamp(ct),str(ct)+'-recording.wav']
   writer.writerow(row)
   f.close()

def run_command():
   f = open(path + str(ct) + ".txt", "w")
   f.write("some content")
   f.close()

####################
# MAIN FUNCTIONALITY
####################

#1 log start, along with command string
start_log()

#2 obtain recording
run_command()

#3 end log
end_log()

#4 write to CSV
write_csv()

