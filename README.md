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
![gray](https://user-images.githubusercontent.com/85461865/122636742-dd486080-d0b8-11eb-8bfb-becf85ae3f3b.png)
### **Step 3 :** Apply Gaussian smoothing to reduce the noise in the picture.
![blur](https://user-images.githubusercontent.com/85461865/122636753-e6393200-d0b8-11eb-8c0d-34ae4f8a861c.png)
### **Step 4 :** Apply Canny Edge Detection to extract edges. 
![edges](https://user-images.githubusercontent.com/85461865/122636762-f3eeb780-d0b8-11eb-8146-926f507b1a33.png)
### **Step 5 :** Apply region of interest to identify only the lanes on road.
![Region](https://user-images.githubusercontent.com/85461865/122636771-ff41e300-d0b8-11eb-8e46-5f595e65c9e8.png)
### **Step 6 :** Apply Hough Transform to get the lines from the image
![Hough](https://user-images.githubusercontent.com/85461865/122636781-0963e180-d0b9-11eb-835a-ac4e3f916142.png)
### **Step 7 :** Apply weighted sum of input image and the image obtained after Hough Tranform to get the lanes on the road.
![lane_lines](https://user-images.githubusercontent.com/85461865/122636809-2f898180-d0b9-11eb-9f4b-2958c0090b59.png)
### In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating the slope, and depending on the sign of the slope (positive or negative), separated left and right lane line. Then, I took the average of the position of each of the lines and extrapolate to the top and bottom of the lane.
![extrapolated](https://user-images.githubusercontent.com/85461865/122637104-e5090480-d0ba-11eb-94e3-a15bfbb63da8.png)







