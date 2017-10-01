# **Advanced Lane Finding Project**
---

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

[image1]: ./output_images/checkerboard_dist_undist.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/test1_undistorted.jpg "Undistorted Example"
[image4]: ./output_images/test1_binary.jpg "Binary Example"
[image5]: ./output_images/lines_with_warped_lines.png "Warp Example"
[image6]: ./output_images/lane_pixels.png "Lane Pixels"
[image7]: ./output_images/result_image.jpg "Result Image"

[video1]: ./output_images/project_video_output.mp4 "Video Output"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README

You're reading it!

### Camera Calibration

#### 1. How I computed the camera matrix and distortion coefficients.

The code for this step is contained in the **Camera Calibration** section (cells 3-5) of the [Advanced-Lane-Finding.ipynb](./Advanced-Lane-Finding.ipynb) notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

In the previous step, I wrote a `correct_images()` function that takes 3 arguments:
- List of image filepaths
- Camera calibration matrix
- Distortion coefficients

This applies the `cv2.undistort()` function to all of the images passed in. Below is what the above image looks like after undistorting. The difference is subtle, but you can notice that the vehicles on the right and far left sides of the image are now pulled closer towards the edges.

![alt text][image3]

#### 2. Thresholded binary image.

In the **Thresholded binary image** section (cells 8-11), there is a `color_and_gradient_pipeline()` function which takes in an image and applies color and gradient thresholding. When running on the test images, the gradient thresholding seemed to add a fair bit of noise to the image. While it initially seemed the saturation thresholding alone would be sufficient, I eventually added back gradient thresholding as it made lane detection slightly more robust.

Below is the same image from above with thresholding applied. You can see that the left lane marking is easily recognizable, while the stripped right lane markings visually harder to identify as a lane line. However, this is addressed in the next step.

![alt text][image4]

#### 3. Perspective transform

The code for my perspective transform, which is cells 12 & 13 in the **Perspective transform** section, includes two functions: `calculate_perspective_transforms()` and `image_warper()`.

The `calculate_perspective_transforms()` function takes sets of source (`src`) and destination (`dst`) points as inputs and returns a perspective transform (`M`), as well as an inverse perspective transform (`Minv`), which are later used to warp/unwarp a given image.

I chose the source and destination points by carefully inspecting the `straight_lines1.jpg` test image (corrected for distortion) and finding points where the lane lines start from the bottom of the image, as well as where they begin to fade into the horizon, which was at height = 450.

As a result, these are the source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 592, 450      | 320, 0        |
| 196, 720      | 320, 720      |
| 1114, 720     | 960, 720      |
| 686, 450      | 960, 0        |

Now that I have the perspective transform (`M`), I can use this to transform a given image. I do this using a convenience function called `image_warper()` which takes an image and a perspective transform. I chose to separate the image warping from the perspective calculation as the perspective calculation only needs to happen once, but the image warping will happen on every image.

I verified that my perspective transform was working by drawing lines on the corrected `straight_lines1.jpg` and then warping the images. As seen below, the drawn lines, which border the lane lines on the images, are relatively straight and parallel in the transformed image.

![alt text][image5]

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial

The code for identifying lane-line pixels and fitting their positions is in a function called `find_and_paint_lane()` in cell 16. This function is set up to take in parameters from a previous frame if lane lines have already been identified. If there are not pre-existing lane lines, the function starts by first taking a histogram of the bottom half of the binary, warped image from previous steps. Then, the x positions on the left and right sides of the image with the most pixels along the y axis are stored in `leftx_base` and `rightx_base`. Using these as a starting position, the function then creates "sliding windows" that are 200px wide by 80px high for the left and right lane. These windows records all nonzero value pixels in their respective windows. Then, the average is taken in each window to find the center. The window is then reposition and moved up to the next level. The end result is two arrays of points representing the left and right lane lines.

Using these points, a curve is fit using `np.polyfit()`. Now that polynomials have been fit for the left and right lane, x and y values are generated for plotting.

The image below show the pixels that were identified as left and right lane lines, as well as the polynomials that were fit to these pixels (in yellow) from the image shown above in sections 1 and 2.

![alt text][image6]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

The code for calculating the radius of curvature of the lane and the position of the vehicle with respect to center is show below. This is in the `find_and_paint_lane()` function (cell 16) in the notebook.

Similar to the step before, `np.polyfit()` is used to generate curves fit to the pixels, but this time the curves are calculated in terms of meters. With these curves, we can use the equation given for the radius of curvature given in the lessons to calculate the radius. This is done for both the left and right lane. However, I've chosen to only show the left lane curvature in the output.

For the vehicle position, the center of the lane can easily be calculate using the x position of each lane at the bottom of the image. With the x position of the lane, we can subtract the image midpoint to find the offset, the multiply this by the meter/pixel conversion. The result is the distance left or right of the center of the lane.

```python
# Find curve in radians
y_eval = np.max(ploty)
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
radius_text = "Radius of Curvature = {}(m)".format(int(left_curverad))

# Find distance from center of lane
lane_midpoint = ((right_fitx[-1] - left_fitx[-1]) / 2) + left_fitx[-1]
offset = lane_midpoint - midpoint
offset_in_m = offset * xm_per_pix
if offset_in_m >= 0:
    center_text = "Vehicle is {0:.2f}m left of center".format(round(offset_in_m,2))
else:
    center_text = "Vehicle is {0:.2f}m right of center".format(round(abs(offset_in_m),2))
```

#### 6. Result image

In the **Result image** section of the notebook (cell 17), I've run the original image from section 1 above through the entire pipeline. Below is the result.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.

[Video Output][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once I was able to get the basic pipeline performing well on the sample images, there were a few conditions which caused the lane finding to break down.

1. Dramatic changes in lighting/exposure on a given set of frames in the video, such as shadows.
2. The entry of a black sedan in the adjacent lane midway through the video.

There were a few changes I made to correct for these disruptions:
- While initially I didn't include color gradient thresholding, I chose to add this back in as white lane markings were not being detected reliably using saturation thresholding alone.
- The heavy shadows towards the end of the video created a lot of noise due to the fact that they are heavily saturated. To address this, I decided to do simple RGB color filtering and remove what I considered black, which were values less than 48 (out of 255) on the `R`, `G`, and `B` channels. This greatly improved performace.
- Finally, and similarly to above, I chose to actively include pixels that registered as white, as some of the white lane markings weren't strongly registering with the other color thresholding methods. While this improved performance on the project video, I'm now realizing this could cause problems if a white car appears in an adjacent lane.

A few improvements that come to mind that would make detection more robust include:
- More sophisticated tracking of previously detected lane lines. This was suggested in the project recommendations, but I have not yet implemented this.
- Better adaptation to changes in lighting and picture quality. Currently, my algorithm completely breaks down with the `harder_challenge_video.mp4` file.
