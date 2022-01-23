#!/usr/bin/python3

import RPi.GPIO as GPIO
import time

vPin = 16

def setup():
    global pwm
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(vPin, GPIO.OUT)
    GPIO.output(vPin, GPIO.LOW)
    pwm = GPIO.PWM(vPin, 1000)   
    pwm.start(0)

def loop():
    while True:
        for dc in range(0, 101, 1):
            pwm.ChangeDutyCycle(dc)
            time.sleep(0.01)
        time.sleep(1)
        for dc in range(100, -1, -1):   
            pwm.ChangeDutyCycle(dc)
            time.sleep(0.01)
        time.sleep(1)

def destroy():
    pwm.stop()
    GPIO.output(vPin, GPIO.LOW)
    GPIO.cleanup()

if __name__ == '__main__':
    setup()
    try:
        print("entering loop")
        loop()
    except KeyboardInterrupt:
        print("ending")
        destroy()
