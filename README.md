# Project 04 - Advanced Lane Finding
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

[//]: # (References)

[image1]: ./figures/fig1_undistort.png "Original and Undistorted images"
[image2]: ./figures/fig2_warp_options.png "Warp options"
[image3]: ./figures/fig3_edges.png "Thresholded Image"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[laplacian]: http://www.eng.utah.edu/~hamburge/Road_Marking_Features_and_Processing_Steps.pdf "Road Marking Features and Processing"

[ref2]: https://pdfs.semanticscholar.org/2bce/94e1f0d921d6876cf346103f5f3e121bfdd8.pdf "Gradient-Enhancing Conversion for
Illumination-Robust Lane Detection"

<!-- "Gradient-Enhancing Conversion for
Illumination-Robust Lane Detection, Hunjae Yoo, Ukil Yang, and Kwanghoon Sohn, IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 1" -->

---
## Writeup / README

The code I used for doing this project can be found in `project04.py` and `Project04.ipynb`. The following sections go into further detail about the specific points described in the [Rubric](https://review.udacity.com/#!/rubrics/571/view).

## Camera Calibration and Distortion Correction

The camera calibration code is contained in lines 25-50 of `project04.py`. The camera calibration information is cached using `numpy.savez_compressed` in the interest of saving time.

The chessboard pattern calibration images contained 9 and 6 corners in the horizontal and vertical directions, respectively. First, a list of "object points", which are the (x, y, z) coordinates of these  chessboard corners in the real-world in 3D space, is compiled. The chessboard is assumed to be in the plane z=0, with the top-left corner at the origin. There is assumed to be unit spacing between the corners. These coordinates are stored in the array `objp`.

For each calibration image, the image coordinates (x,y), of the chessboard corners are computed using the `cv2.findChessboardCorners` function. These are appended to the `imgpoints` array and the `objp` array is appended to the `objpoints` array.

The two accumulated lists, `imgpoints` and `objpoints` are then passed into `cv2.calibrateCamera` to obtain the camera calibration and distortion coefficients. The input image is then undistorted (later in the image processing pipeline on line 409) using these coefficients and the `cv2.undistort` function. An example result is shown below:

![alt text][image1]

## Pipeline (single images)

### Overview
#### Order of operations
In the original pipeline, the perspective transform was applied after the edges were detected in the image using some thresholding techniques (described in a later section).

The effect of reversing this order, by applying the perspective transform first and then applying the edge detection was also examined. It was found that with the current set of hyper-parameters used for edge detection, the reverse process was able to reject more "distractions" at the cost of a decrease in the number of lane pixels found. This was particularly of interest in some parts of the challenge video. Here is an example of an image where the order of thresholding and warping made a difference:

![Effect of order of operations][image2]

However, the effect wasn't significant enough to merit it's use in the final version as it removed too many lane pixels. `challenge_video_out2.mp4` is an example of the video using the reverse order of operations. In the case of the challenge video, it gave much better results than the original pipeline. However, better methods for gradient and contrast enhancement might make the original pipeline just as good.

Therefore the pipeline used in the final version was:

**Distortion correction → Thresholding → Perspective Transform → Lane Detection (using Histogram or Masking methods) → Sanity Check**

The perspective transform points are computed dynamically in the beginning by using a Hough transform to detect the lane lane positions on the thresholded image. It falls back to a fail-safe in case the Hough transform doesn't work. This is described in detail in a later section.

#### Gradient enhancement
A color image can be converted to grayscale in many ways. The easiest is what is called equal-weight conversion where the red, green and blue values are given equal weights and averaged. However, the ratio between these weights can be modified to enhance the gradient of the edges of interest (such as lanes). According to [Ref2](ref2), grayscale images converted using the ratios 0.5 for red, 0.4 for green, and 0.1 for blue, are better at detecting yellow and white lanes. This method was used for converting images to gray scale in this project.

However, even this method faces problems under different color temperature illuminations, such as evening sun or artificial lighting conditions. A better method that continuously computes the best weighting vector based on prior frames is described in [Ref 2](ref2). This was not implemented in this project due to time constraints, and because reasonable results were obtained for the project and challenge videos without the use of such a method.

### Description of Pipeline
#### 1. Example of Distortion corrected image

![Example of Distortion corrected image][image1]

#### 2. Thresholding

Initially a number of combinations of color and gradient thresholds were attempted. It was found that none of these were very robust to changing conditions in lighting and contrast. After reviewing some literature on the subject, it was found that using a second derivative operation (a Laplacian) might be more suited to this purpose. By using a Laplacian operation on the image followed by thresholding it to highlight only the negative values (denoting a dark-bright-dark edge) it was possible to reject many of the false positives [ [Ref](http://www.eng.utah.edu/~hamburge/Road_Marking_Features_and_Processing_Steps.pdf) ]. The Laplacian resulted in better results than using combinations of Sobel gradients.


The thresholding operations used to detect edges in the images can be found in lines 149-175 of `project_04.py` in the function called `find_edges`. The thresholded binary mask obtained from the Laplacian is named `mask_one` in the code. It is first computed for the S-channel of the image in HLS colorspace. If too few pixels were detected by this method (less than 1% of total number of pixels), then the laplacian thresholding is attempted on the grayscale image.

The second thresholded mask, `mask_two`, is created using a simple threshold on the S-channel. And finally, a brightness mask (`gray_binary`) is used to reject any darker lines in the final result. These masks are combined as:
`combined_mask = gray_binary AND (mask_one OR mask_two)`

The results obtained using the edge detection algorithm for an image is shown below:

![alt text][image3]

#### 3. Perspective transform

The perspective transformation is computed using the functions `find_perspective_points` and `get_perspective_transform` in lines 52-147 of `project04.py`.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
