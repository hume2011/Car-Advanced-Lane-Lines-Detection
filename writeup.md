## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced_lane_detection.ipynb". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

Pipeline is contained in the 10th code cell.

#### 1. Distortion-corrected.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. How I used color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step. 

![alt text][image3]

#### 3. How I performed a perspective transform.

The code for my perspective transform includes function `cv2.getPerspectiveTransform()` and function `cv2.warpPerspective()`. The `cv2.getPerspectiveTransform()` function takes as inputs a source (`dst1`) and a destination (`dst2`) points.  I chose to hardcode the source and destination points in the following manner:

```python

dst1 = np.float32([(0, 720), (553, 460),(728, 460), (1280, 720)])
dst2 = np.float32([(img_size[0]*0.1, img_size[1]*0.9), (img_size[0]*0.1, img_size[1]*0.1), (img_size[0]*0.9, img_size[1]*0.1), (img_size[0]*0.9, img_size[1]*0.9)])
```

I verified that my perspective transform was working as expected by checking the images before and after warped to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. How I identified lane-line pixels and fit their positions with a polynomial.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. How I calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

......

#### 6. An example image of my result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step below `#### Draw lane lines` in my code in `Pipeline`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. The final video output:

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion


When it comes to shadows or inconsistent colors of the road, this implementation can not work well, then the performance was better after I retuned the color and grandient threshold.

Still, it is totally not working for challenge videos, I should add other techniques to my implementation.
