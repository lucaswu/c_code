#! /bin/sh

nvcc -g -o test resize.cu `pkg-config --cflags --libs opencv`
if [ $? -ne 0 ];then
	echo "build  error !"
	exit 1
fi
./test /home/lucas/github/darknet/predictions.png out.jpg
