#!/usr/bin/python3
import os
import time
from datetime import datetime
import board
import adafruit_ahtx0
import logging
import csv
import configparser

# set logging
logging.basicConfig(filename='/home/pi/irri-logs.log', level=logging.DEBUG)

# connect devices, other housekeeping
i2c = board.I2C()
sensor = adafruit_ahtx0.AHTx0(i2c)
config = configparser.ConfigParser()
ct = time.time()

# load configuration from file
config.read('config_record.ini')
path = config['RECORD']['record_path']
duration = int(config['RECORD']['record_duration']) * 60

# obtain current sensor readings
temp = sensor.temperature
hum = sensor.relative_humidity

# start log
logging.info('Record Start: %s', datetime.now())

# obtain recording
# 'arecord -d 600 -D plughw -c1 -f S32_LE -t wav -V mono -v /home/pi/DDMMYY-recording.wav'
os.system('arecord -d ' + str(duration) + ' -D plughw -c1 -f S32_LE -t wav -V mono -v ' + path + str(ct) + '-recording.wav')     #deleted -r 48000 before -f to reduce file size

# end log
logging.info('Record End: %s', datetime.now())
logging.info('File : %s-recording.wav, Temperature: %s, Humidity: %s', ct, temp, hum)
print('log saved...')

#csv log
f = open(config['CSV']['csv_path'],'a')
writer = csv.writer(f)
row = [datetime.fromtimestamp(ct),str(ct)+'-recording.wav','{0:.2f}'.format(temp),'{0:.2f}'.format(hum)]
writer.writerow(row)
f.close()

