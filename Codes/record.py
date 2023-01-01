#!/usr/bin/python3
import os
import time
from datetime import datetime
import board
import adafruit_ahtx0
import logging
import csv
import configparser

# connect devices, other housekeeping
i2c = board.I2C()
sensor = adafruit_ahtx0.AHTx0(i2c)
config = configparser.ConfigParser()

# VARIABLE SETTING

# obtain timing-based data
ct = time.time()
temp = sensor.temperature
hum = sensor.relative_humidity

# load configuration from file
config.read('config_record.ini')
path = config['RECORD']['record_path']
log_path = config['LOG']['run_log']
channel_count = str(config['RECORD']['channel_count'])
device_name = config['RECORD']['device_name']
duration = int(config['RECORD']['record_duration']) * 60

# set logging
logging.basicConfig(filename=log_path, level=logging.DEBUG)

# create arecord command
# ex: 'arecord -d 600 -D plughw -c1 -f S32_LE -t wav -V mono -v /home/pi/DDMMYY-recording.wav'
command = 'arecord -d ' + str(duration) + ' -D ' + device_name + ' -c' + channel_count + ' -f S32_LE -t wav -V mono -v ' + path + str(ct) + '-recording.wav'

# FUNCTION DEFINITION

def start_log():
   logging.info('Record Start: %s', datetime.now())
   logging.info('Command String: %s', command)

def end_log():
   logging.info('Record End: %s', datetime.now())
   logging.info('File : %s-recording.wav, Temperature: %s, Humidity: %s', ct, temp, hum)
   print('log saved...')

def write_csv():
   f = open(config['CSV']['csv_path'],'a')
   writer = csv.writer(f)
   row = [datetime.fromtimestamp(ct),str(ct)+'-recording.wav','{0:.2f}'.format(temp),'{0:.2f}'.format(hum)]
   writer.writerow(row)
   f.close()

def run_command():
   os.system(command)

##################
# START
##################

#1 log start, along with command string
start_log()

#2 obtain recording
run_command()

#3 end log
end_log()

#4 write to CSV
write_csv()

##################
# END
##################