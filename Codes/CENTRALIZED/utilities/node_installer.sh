#! /bin/bash

# update device first
sudo apt update
sudo apt upgrade -Y
sudo apt install smbclient
sudo apt install cifs-utils

# backup device install to image
wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
chmod +x pishrink.sh
sudo mv pishrink.sh /usr/local/bin
sudo dd if=/dev/mmcblk0 of=/mnt/backup_disk/rpi3_012023.img bs=1M
sudo pishrink.sh -z /mnt/backup_disk/rpi3_012023.img

# create mount point
sudo mkdir /mnt/backup_disk

# check if nfs is available
smbclient -L 192.168.1.108 -U pi

# /etc/fstab for main storage device
//192.168.1.108/MainStorage /mnt/backup_disk cifs rw,uid=1000,_netdev 0 1
