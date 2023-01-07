#! /bin/bash

source ../config

# iterate IPs top to bottom
iterator=0
while [ $iterator -lt ${#IP_ARRAY[@]} ]
do
    ping -c 1 ${IP_ARRAY[$iterator]} > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "node ${IP_ARRAY[$iterator]} is up" 
    else
        echo "node ${IP_ARRAY[$iterator]} is down"
    fi

    # next IP
    iterator=`expr $iterator + 1`
done
