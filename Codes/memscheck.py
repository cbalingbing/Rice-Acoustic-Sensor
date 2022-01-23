#!/usr/bin/python3

import os
import time

def main():
    os.system('clear')
    #os.system('arecord -l')
    print('This script record a test sound for 30 secs and then play it.')
    time.sleep(2)
    print('Recording in 5')
    time.sleep(1)
    print('Recording in 4')
    time.sleep(1)
    print('Recording in 3')
    time.sleep(1)
    print('Recording in 2')
    time.sleep(1)
    print('Recording in 1')
    time.sleep(1)
    os.system('arecord -D dmic_sv -c2 -r 48000 -f S32_LE -t wav -V mono -v -d 30 /media/Caling/recording.wav')
    os.system('aplay recording.wav')

if __name__ == "__main__":
    main()
