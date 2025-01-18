# Projective Reconstruction

## Image Rectification

To begin projective reconstruction, I find corresponding points between two images of the same scene. To do this I first begin by rectifying both the images such that all the corresponding points between the images lie in the same row of the second image.

Given that I have uncalibrated cameras, I begin by calculating the fundamental matrix (F) that represents the two cameras. To do this I manually find 14 corresponding points between the two images and use them to estimate F.

We know that give a pair of corresponding points (x,x ') and the fundamental matrix between the two cameras:

$$
X' T F X = 0
$$

Expanding this equation for a set of points we get:

$$
x' x f_{11} + x' y f_{12} + x' f_{13} + y' x f_{21} + y' y f_{22} + y' f_{23} + x f_{31} + y f_{32} + f_{33} = 0
$$

Where,

$$
X = 
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
\quad
X' = 
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
,\quad
F = 
\begin{bmatrix}
f_{11} & f_{12} & f_{13} \\
f_{21} & f_{22} & f_{23} \\
f_{31} & f_{31} & f_{33}
\end{bmatrix}
$$

This equation can be rewritten as:

$$
A = [x' {} x,\; x' {} y,\; x' {},\; y' {} x,\; y' {} y,\; y' {},\; x,\; y,\; 1]
$$

$$
f = [f_{11}, f_{12}, f_{13}, f_{21}, f_{22}, f_{23}, f_{31}, f_{32}, f_{33}] T
$$

$$
A f = 0
$$

Since F is homogenous, we have 8 unknowns. This means that we need at least 8 such equations to solve for all possible bales of F.  
So, for N points the A matrix becomes of size Nx9. Finally, we find the linear least square solution of the above equation to get an initial estimate of the fundamental matrix F.  
I personally annotated 14 points, so the A matrix used to find F was of size 14x9.

Once we have our initial estimate for F, we can make estimates of the left and right epipoles of the cameras. The left epipole is represented as e and the right epipole is represented as e '.  
The epipole of the left camera is the null space of F and the right epipole is the null space of F T.

Once we have the estimated epipoles we can use them to find the camera projection matrices for the two cameras. We assume that they are in canonical configuration.  
In canonical Configuration the left camera projections matrix is:

$$
P = 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

The right camera projection matrix is:

$$
P' = [ sF \bigm|e' ]
$$

Here s is any skew symmetric matrix. By convention we use the cross-product representation of e '.

To get the real-world 3D coordinates from the projection matrices we use the following equations:  
We can determine that for a correspondence (x,x ' ), which has the real-world coordinate X

$$
x \, P ' X = 0
$$

$$
x ' \, P ' X = 0
$$

Rewriting the above equation we can get 4 linearly independent constrains:

$$
y(P 3T X) - (P 2T X) = 0
$$

$$
x(P 3T X) - (P 1T X) = 0
$$

$$
y '(P' 3T X) - (P' 2T X) = 0
$$

$$
x '(P' 3T X) - (P' 1T X) = 0
$$

Converting these equations to a homogeneous system we get:

$$
A = 
\bigl[\,
x(P 3T) - (P 1T),\;
y(P 3T) - (P 2T),\;
x '(P' 3T) - (P' 1T),\;
y '(P' 3T) - (P' 2T)
\bigr]
$$

$$
A X = 0
$$

Solving the above equation using linear least squares gives us the initial guest for the 3D world points that are further refined by LM.  
Once we get our initial guesses for the camera projection matrices, and the 3D world points we use LM to optimize our estimate for P ' our predictions of the real-world 3D coordinates.

Once we have our optimized camera projection matrices and the real-world point estimates, we use them to calculate rectification homographies for the images.  
Calculating the rectification homographies involves sending the epipole of the image to infinity. This is done for the right image using the following steps:  
First, we translate the center of the image to the origin with the rotation matrix:

$$
T = 
\begin{bmatrix}
1 & 0 & -w/2 \\
0 & 1 & -h/2 \\
0 & 0 & 1
\end{bmatrix}
$$

Then we rotate the image so that the epipole rotates onto the x-axis. The angle we need is the inverse tangent of the location of the translated epipole:

$$
\theta = \tan {-1}\Bigl(\frac{-(e_y ' - w/2)}{(e_x ' - h/2)}\Bigr)
$$

We then create the rotation matrix:

$$
R = 
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Once this translation and rotation are applied to the epipole it ends up at the point [f,0,1] T  
We can solve for this distance f. The value for f we get is:

$$
f = (e_y ' - w/2)\cos(\theta) \;-\; (e_x ' - h/2)\sin(\theta)
$$

Finally, we find the G matrix that take the epipole from [f,0,1] T to [1,0,0] T.

$$
G =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
-1/f & 0 & 1
\end{bmatrix}
$$

Then the homography that rectifies the right image becomes:

$$
H ' = G \, R \, T
$$

Now we need to estimate the homography that rectifies the left image. This can be done with the following calculations:

$$
M = P ' P +
$$

where \( P + \) is the pseudo inverse of \( P \)

$$
H_0 = H ' M
$$

$$
H_a =
\begin{bmatrix}
a & b & c \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Where a, b, c = min┬abc ( Σ(ax '+by '+c - x ' ) 2 )  
We find this minimization using linear least squares and construct H_a  

Finally, the homography that rectifies the left image is

$$
H = H_a \, H_0
$$

This was we have found the homographies that rectify both our images.

---

## Finding Interest points in Rectified images

For edge detection I use the inbuilt CV2 function cv2.canny(). This function takes in two threshold values that control the sensitivity of edge detection. Higher values of the threshold will result in fewer and stronger edges, while lowering the values will give us more and potentially weaker edges. The output of the function is a binary image where the edges are represented by pixel values of 255 and the rest of the image has pixel values of 0.  
My threshold values are: 200 and 200  
These edge points are candidates for which we will find corresponding points in the two images.  
To match corresponding points, we pick a point in the left image and then search for all its potential matches in the same row of the right image. I use SSD to find the best candidate in the row of the right image.  
For the Sum of Squared Differences (SSD) method, the neighboring pixels are compared using the following formula:

$$
SSD = \sum_i \sum_j \bigl| f_1(i,j) - f_2(i,j) \bigr| ^{2}
$$

Here, f_1 (x,y) is the grayscale value of the pixel located at (x,y) in the first image  
And f_2 (x,y) is the grayscale value of the pixel located at (x,y) in the second image  
The summation here goes over all the pixels in the grid of 10×10 pixels described above  

My implementation of SSD correspondence calculates the SSD for all pairs of interest points in the two images. The point in the second image that has the lowest SSD value for a point in the first image is deemed the corresponding interest point of the first image in the second image.  
This gives us all the corresponding points in both the images.

---

## Projective Reconstruction

We begin with all the corresponding points we found in the rectified images. We multiply the points with the inverse of the corresponding rectified homographies to get the corresponding points in the original image coordinates. Next, we estimate the 3D real-world point for all these corresponding points using the same method described above. We then run LM on all these estimated real-world points. We use our refined values of P and P ' to convert these real-world points into the camera frame. Our loss is how far the transformed real-world points are from the actual correspondence we found in the image.  
The output from LM is all the coordinates of the real-world points that can be used for 3D scene construction.  
Finally, we plot all these points on a 3D axis to visually inspect all the points.

---

# Loop and Zhang Algorithm

The loop and Zhang Algorithm is used to rectify and find correspondences between stereo images. The algorithm essentially decomposes the rectification homographies H and H ' using the formulas below:

$$
H = H_{sh} \, H_{sim} \, H_{p}
$$

$$
H ' = H_{sh} ' \, H_{sim} ' \, H_{p} '
$$

H_sh  and H_sh ' are shearing homographies.  
H_sim and H_sim ' are similarity homographies  
H_p and H_p ' are purely projective homographies  

The projective homographies send the epipoles of the image to infinity. The similarity homographies translate, rotate and scale the images. Finally, the shearing homographies are used to eliminate all the non-linear distortions.

The output from the Loop and Zhang algorithm finds a lot more correspondence when compared to my implementation. This is likely because it doesn’t find correspondence only on a subset of point, like I do by using the canny edges. The distortion in the rectified images are a lot lower in the output of Loop and Zhang when compared to my rectified images. My outputs have a larger number of false positives that need to be filtered out. This is likely a weakness of SSD. Finally, my implementation required manual annotation of a few correspondences to work. Loop and Zhang’s algorithm didn’t need any such human intervention. This makes Loop and Zhang a more favorable method to find correspondence. The output from my pipeline is good, however, not as good as Loop and Zhang.

---

# Dense Stereo Matching

Dense stereo Matching is used to find pixel correspondences in two images. These images need to be rectified stereo images. The output of dense stereo matching is usually a disparity map. Since the images are rectified, the difference is the corresponding pixel locations is only along the column. The difference in the column of the corresponding pixels is considered as the disparity value and is stored as the disparity map.

Now to find the best match for correspondences in the same row of the images we use Census Transform.

### Census Transform

Census transform considers a window of pixels around the pixel for which is it finding a correspondence. For a pixel in the first image at location (p_1,p_2), we consider pixels in the second image that are located at (p_1-d,p_2), where d is all the possible disparity values.

Now for all the potential matches we create bit vector that is the size of the number of pixels in the window. Every pixel in the window casts a vote. In our case pixels values in the window that have a value strictly greater than the pixel we are interested in cast a vote of 1. The remaining pixels cast a vote of 0. These votes are what create the bit vector.

Similarly, we also create the same bit vector for the pixel in the first image. The bit vector from the first and second image are XOR’d together and the number of ones in the result are counted. The cost is the total number of ones. We find these costs between the pixel in the first image and all the candidate pixels in the second. The disparity value that gives is the lowest cost is then used as the value for that pixel in the disparity map.

The XOR essentially highlights where the votes of the pixels differ. More the differences in the votes, higher the cost and worse is that particular value of disparity.

From the outputs below we can see that increasing the window size increases accuracy. The larger window size also smooths the disparity map, there are less abrupt changes in the map for a larger window size. The larger the window size, the greater the number of votes. This makes every disparity calculation more accurate but also makes it computationally harder. In my testing, for our images, a window size of 20x20 yielded the best results. Increasing the window size after that gave diminishing returns.

---

# Automatic Extraction of Dense Correspondences

Automatic Extraction of dense Correspondence can be done when we have a pair of images and their depth maps. This is done by taking a point in one of the images, converting it to a real-world point and estimating its depth with the depth map. We then project this estimated point onto the other image. We check the depth of this projected image against the second depth map we have. If the depths match within some threshold we consider the points as matches. Using the depth map allows us to compare the depths of points and use it as a metric to find similarities between points in two images.

To get our 3D coordinate we multiply a pixel coordinate in the image with the inverse of the K matrix for the camera. This gives us any point on the ray that corresponds to that pixel in homogenous coordinates. We normalize the point by dividing all its elements by the last value. We then multiply this point with the depth we have in the depth map. This gives us the estimate of the 3D point.

We convert this 3D point into the world coordinate frame using the extrinsic parameters of the camera. The inverse of the extrinsic parameter matrix is multiplied with the homogenous 3D coordinate calculated in the previous step.

Once we have the point in the world coordinate frame, we estimate the point in the second cameras frame using the extrinsic camera matrix for the second camera. This gives us the estimate of the depth at that point in the second camera’s frame.

Finally, we project this point onto the second image and obtain its pixel location in the second image. We check the depth value from our second depth map and if the difference between our estimate and the value in the depth map is less than our threshold we considered the points as matches.
