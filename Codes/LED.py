#!/user/bin/python
import RPi.GPIO as io
import time
io.setmode(io.BCM)
led1 = 27
led2 = 22
io.setup(led1, io.OUT)
io.setup(led2, io.OUT)
while 1
io.output(led1, True)
time.sleep(1)
io.output(led1, False)
io.output(led2, True)
time.sleep(1)
io.output(led2, False)
time.sleep(1)