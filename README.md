# Finding-Lane-Lines
### This project is a part of Self-Driving Car Engineer Nanodegree Program from Udacity.

## Objective:
### The goal of the project is to build a pipeline to find lane lines on the road using python and OpenCV.

## Instructions:
### To get started, you need to have some packages. click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) to setup all the requirements. Before creating the environment, change the text in the environment.yml file as provided in the repository.

### After activating the environment, type `jupyter notebook`. copy the link and paste it in browser to work on jupyter notebook. Additionally, I have also provided python file just in case.

## Pipeline:
### **Step 1 :** Read the image.
![Image](https://user-images.githubusercontent.com/85461865/122636591-1502d880-d0b8-11eb-9c19-04df283c068e.png)
### **Step 2 :** Convert the image to a gray scale.
### **Step 3 :** Apply Gaussian smoothing to reduce the noise in the picture.
### **Step 4 :** Apply Canny Edge Detection to extract edges. 
### **Step 5 :** Apply region of interest to identify only the lanes on road.
### **Step 6 :** Apply Hough Transform to get the lines from the image
### **Step 7 :** Apply weighted sum of input image and the image obtained after Hough Tranform to get the lanes on the road.








