#! /bin/bash

# update device first
sudo apt update
sudo apt upgrade -Y

# backup device install to image
wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
chmod +x pishrink.sh
sudo mv pishrink.sh /usr/local/bin
sudo dd if=/dev/mmcblk0 of=/mnt/backup_disk/rpi3_012023.img bs=1M
sudo pishrink.sh -z /mnt/backup_disk/rpi3_012023.img

# create mount point and mount device
sudo mkdir /mnt/backup_disk
sudo mount /dev/sda1 /mnt/backup_disk

# create key and send to the node
ssh-keygen
ssh-copy-id -i ~/.ssh/id_rsa 192.168.1.234

# make script executable
chmod +x script.sh
./script.sh

# install samba and configure
sudo apt install samba
sudo apt install samba-common-bin
sudo nano /etc/samba/smb.conf
		[MainStorage]
		path = /mnt/backup_disk/NAS_SAVES
		comment = Main storage for all outputs
		writable = yes
		available = yes
		browseable = yes
		read only = no
		public = yes

# /etc/fstab for main storage device
PARTUUID=2c69c7b5-01    /mnt/backup_disk        vfat    rw,users,_netdev,umask=000      0       1

