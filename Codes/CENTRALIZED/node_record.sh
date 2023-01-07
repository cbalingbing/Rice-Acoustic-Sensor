#! /bin/bash

source config

# iterate IPs top to bottom
iterator=0
while [ $iterator -lt ${#IP_ARRAY[@]} ]
do
        # run command
        ssh $username@${IP_ARRAY[$iterator]} python < $python_script - $audio_path $log_path $csv_path $iterator $duration $mic_name $channel_count
        
        # next IP
        iterator=`expr $iterator + 1`
done
