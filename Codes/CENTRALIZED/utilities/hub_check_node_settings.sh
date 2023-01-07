#! /bin/bash

source ../config

# iterate IPs top to bottom
iterator=0
while [ $iterator -lt ${#IP_ARRAY[@]} ]
do
	echo "###########################################"
	echo "Device $iterator IP ${IP_ARRAY[$iterator]}"
	echo "###########################################"
	
	echo "**** Contents of /etc/fstab ****"
   	ssh $username@${IP_ARRAY[$iterator]} cat /etc/fstab

	echo "**** All Mounted drives ****"
   	ssh $username@${IP_ARRAY[$iterator]} df -h

   	echo "**** Network folder access ****"
   	ssh $username@${IP_ARRAY[$iterator]} ls -lh /mnt/backup_disk/

    # next IP
    iterator=`expr $iterator + 1`
done
