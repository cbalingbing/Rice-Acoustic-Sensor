#!/usr/bin/python3
import os
import time
from datetime import datetime
import logging
import csv
import sys
import board
import adafruit_ahtx0

# connect devices, other housekeeping
i2c = board.I2C()
sensor = adafruit_ahtx0.AHTx0(i2c)

##################
# VARIABLE SETTING
##################

# load configuration from arguments
path = sys.argv[1]
log_path = sys.argv[2]
csv_path = sys.argv[3]
node_counter = int(sys.argv[4])
duration = int(sys.argv[5])
device_name = sys.argv[6]
channel_count = sys.argv[7]

# set logging
logging.basicConfig(filename=log_path, level=logging.DEBUG)

# obtain time-based data
machine_time = time.localtime(time.time())
ct = str(machine_time.tm_mday)+ str(machine_time.tm_mon) + str(machine_time.tm_year) + '_' + str((machine_time.tm_hour)) + 'h' + str(machine_time.tm_min)+ 'm' + str(machine_time.tm_sec)
temp = sensor.temperature
hum = sensor.relative_humidity

# create arecord command
command = 'arecord -d ' + str(duration) + ' -D ' + device_name + ' -c' + str(channel_count) + ' -f S32_LE -t wav -V mono -v ' + path + str(ct) + '-recording.wav'

#####################
# FUNCTION DEFINITION
#####################

def start_log():
   logging.info('Device %i Record Start: %s', node_counter, datetime.now())
   logging.info('Command String: %s', command)

def end_log():
   logging.info('Device %i Record End: %s', node_counter, datetime.now())
   logging.info('Device %i File : %s-recording.wav, Temperature: %s, Humidity: %s', node_counter, ct, temp, hum)

def write_csv():
   f = open(csv_path,'a')
   writer = csv.writer(f)
   row = [node_counter,ct,str(ct)+'-recording.wav','{0:.2f}'.format(temp),'{0:.2f}'.format(hum)]
   writer.writerow(row)
   f.close()

def run_command():
   os.system(command)

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
