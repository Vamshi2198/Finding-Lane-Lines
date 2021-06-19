#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

# helper functions
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines_extrapolated(img, lines, color=[255, 0, 0], thickness=2):
    
    x_left=[]
    y_left=[]
    x_right=[]
    y_right=[]
    
#start interpolating
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1) / (x2-x1)
            
            if np.isnan(float(m)):
                continue
                
            if abs(m) < 0.3:
                continue
                
            if m > 0:
                x_right.append(x1)
                y_right.append(y1)
                x_right.append(x2)
                y_right.append(y2)
            elif m < 0:
                x_left.append(x1)
                x_left.append(x2)
                y_left.append(y1)
                y_left.append(y2)
#For left lane:
    left_lane= np.polyfit(x_left,y_left,1)
    m_left = left_lane[0]
    b_left = left_lane[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1_left = maxY
    x1_left = int((y1_left - b_left)/m_left)
    y2_left = int((maxY/2)) + 60
    x2_left = int((y2_left - b_left)/m_left)
    cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), [255, 0, 0], 4)

#For right lane:
    right_lane= np.polyfit(x_right,y_right,1)
    m_right = right_lane[0]
    b_right = right_lane[1]
    
    maxY = img.shape[0]
    maxX = img.shape[1]
    y1_right = maxY
    x1_right = int((y1_right - b_right)/m_right)
    y2_right = int((maxY/2)) + 60
    x2_right = int((y2_right - b_right)/m_right)
    cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), [255, 0, 0], 4)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, extrapolate=False):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if extrapolate:
        draw_lines_extrapolated(line_img, lines)
    else:
        draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import os
os.listdir("test_images/")

# pipeline
def lane_line(image):
    image = mpimg.imread("test_images/" + image)
    img_shape= image.shape
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size=5)
    edges = canny(blur_gray, low_threshold=100, high_threshold=150)
    vertices = np.array([[(0,img_shape[0]),(425, 315), (540, 315), (img_shape[1],img_shape[0])]], \
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    lines = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=40,
                              min_line_len=60, max_line_gap=30)
    lanes = weighted_img(img=lines, initial_img=image, α=0.8, β=1., γ=0.)
    plt.imshow(lanes)

# iterate
test_images = os.listdir('test_images/')
for image in test_images:
    lane_line(image)
    plt.show()

# pipeline for extrapolated lines
def extrapolated_lane_line(image):
    image = mpimg.imread("test_images/" + image)
    img_shape= image.shape
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size=3)
    edges = canny(blur_gray, low_threshold=100, high_threshold=150)
    vertices = np.array([[(0,img_shape[0]),(425, 315), (540, 315), (img_shape[1],img_shape[0])]], \
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    hough_lines_extrapolated = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=40,
                              min_line_len=60, max_line_gap=30, extrapolate=True)
    lanes_extrapolated = weighted_img(img=hough_lines_extrapolated, initial_img=image, α=0.8, β=1., γ=0.)
    plt.imshow(lanes_extrapolated)

#iterate
test_images = os.listdir('test_images/')
for image in test_images:
    extrapolated_lane_line(image)
    plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# pipeline for vedio
def process_image(image):
    img_shape= image.shape
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size=3)
    edges = canny(blur_gray, low_threshold=100, high_threshold=150)
    vertices = np.array([[(0,img_shape[0]),(425, 315), (540, 315), (img_shape[1],img_shape[0])]], \
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    hough_lines_extrapolated = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=40,
                              min_line_len=60, max_line_gap=30, extrapolate=True)
    lanes_extrapolated = weighted_img(img=hough_lines_extrapolated, initial_img=image, α=0.8, β=1., γ=0.)
    return lanes_extrapolated

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

def process_image_challenge(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    img_shape= image.shape
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size=3)
    edges = canny(blur_gray, low_threshold=5, high_threshold=170)
    vertices = np.array([[(0,img_shape[0]),(425, 315), (540, 315), (img_shape[1],img_shape[0])]], \
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    hough_lines_extrapolated = hough_lines(masked_edges, rho=0.5, theta=np.pi/360, threshold=10,
                              min_line_len=10, max_line_gap=150, extrapolate=True)
    lanes_extrapolated = weighted_img(img=hough_lines_extrapolated, initial_img=image, α=0.8, β=1., γ=0.)
    return lanes_extrapolated

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image_challenge)
%time challenge_clip.write_videofile(challenge_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))