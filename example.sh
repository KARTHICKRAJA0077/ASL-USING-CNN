#!/bin/bash

#Ask For Input
#echo "enter the File Name\n"
#read path
#filename=$(basename $0)
#read path

#echo "the new file name is\n" 
#echo $path
#echo $filename

#$path 

convert -type Grayscale "imageconv/myimage.jpg"  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -resize 28x28\!  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -filter Mitchell  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -filter Robidoux  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -filter Catrom  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -filter Spline  "imageconv/gray.jpeg"

convert "imageconv/gray.jpeg"  -filter Hermite  "imageconv/gray.jpeg"

#python test.py
