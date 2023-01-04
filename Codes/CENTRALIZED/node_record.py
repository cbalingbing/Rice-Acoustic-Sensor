#!/usr/bin/python3
import os
import time
from datetime import datetime
import logging
import csv
import sys

##################
# VARIABLE SETTING
##################

# load configuration from file
path = sys.argv[1]
log_path = sys.argv[2]
csv_path = sys.argv[3]
node_counter = int(sys.argv[4])
command = 'some command\n'

# set logging
logging.basicConfig(filename=log_path, level=logging.DEBUG)

# obtain time-based data
ct = time.time()

#####################
# FUNCTION DEFINITION
#####################

def start_log():
   logging.info('Device %i Record Start: %s', node_counter, datetime.now())

def end_log():
   logging.info('Device %i Record End: %s', node_counter, datetime.now())
   logging.info('Device %i File : %s-recording.wav, Temperature:, Humidity:', node_counter, ct)
   print('log saved...')

def write_csv():
   f = open(csv_path,'a')
   writer = csv.writer(f)
   row = [node_counter,datetime.fromtimestamp(ct),str(ct)+'-recording.wav']
   writer.writerow(row)
   f.close()

def run_command():
   f = open(path + str(node_counter) + '_' + str(ct) + ".txt", "w")
   f.write(command)
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
