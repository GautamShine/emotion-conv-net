#!/bin/sh
# Install and compile OpenCV and Caffe on NVIDIA Jetson TK1 Development Kit

# General updates:
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove

###############################################################################
# OpenCV 3.1
###############################################################################

# INSTALL THE DEPENDENCIES

# Build tools:
sudo apt-get install -y build-essential cmake

# GUI:
sudo apt-get install -y qt5-default libvtk6-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Java:
sudo apt-get install -y ant default-jdk

# Documentation:
sudo apt-get install -y doxygen

# INSTALL THE LIBRARY (YOU CAN CHANGE '3.0.0' FOR THE LAST STABLE VERSION)

sudo apt-get install -y unzip wget
wget https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip 3.1.0.zip
rm 3.1.0.zip
mv opencv-3.1.0 OpenCV
cd OpenCV
mkdir build
cd build
cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig

###############################################################################
# Caffe
###############################################################################

# Dependencies:
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install libprotobuf-dev protobuf-compiler gfortran \
libboost-all-dev cmake libleveldb-dev libsnappy-dev \
libboost-thread-dev libboost-system-dev \
libatlas-base-dev libhdf5-serial-dev libgflags-dev \
libgoogle-glog-dev liblmdb-dev -y

sudo usermod -a -G video $USER

# Install modules on base python:
sudo apt-get install python-pip
sudo pip install scipy
sudo pip install numpy
sudo pip install matplotlib
sudo pip install scikit-image
sudo pip install protobuf

# Download Caffe; need to modify Makefile to proceed
wget https://github.com/BVLC/caffe/archive/master.zip
unzip master.zip
cd caffe-master
cp Makefile.config.example Makefile.config
