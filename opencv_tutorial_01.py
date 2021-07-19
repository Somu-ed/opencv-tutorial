# importing packages
import imutils
import cv2

# loading i/p images and showing its dimentions
image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# display image
cv2.imshow("image", image)
cv2.waitKey(0)

# accessing a RGB pixel located at (x, y) = (50,100)
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
roi = image[60:160, 320:420]            # Array Slicing "image[startY:endY, startX:endX]"
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image,(200,200))
cv2.imshow("Fixed resizing", resized)
cv2.waitKey(0)

# resize the width to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w                           # Ratio of new:old width
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

# using imutils library
resized = imutils.resize(image, width=300)
cv2.imshow("imutils resize", resized)
cv2.waitKey(0)

# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# using imutils
rotated = imutils.rotate(image, -45)
cv2.imshow("imutils Rotation", rotated)
cv2.waitKey(0)

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# apply a Gaussian blur with a 11x11 kernel
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# draw a 2px thick red rectangle surrounding the face
output = image.copy()
cv2.rectangle(output, (320,60), (420,160), (0, 0, 225), 2)      # "(img, pt1, pt2, color, thickness)"
cv2.imshow("Rectangle", output)
cv2.waitKey(0)
# Since we are using OpenCV’s functions rather than NumPy operations we can supply our coordinates in (x, y) 
# order rather than (y, x) since we are not manipulating or accessing the NumPy array directly — OpenCV is taking 
# care of that for us.

# draw a blue 20px (filled in) circle on the image centered at x=300,y=150
output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)             # "(img, center, radius, color, thickness)"  -1 = solid
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)          # "(img, pt1, pt2, color, thickness)"
cv2.imshow("Line", output)
cv2.waitKey(0)

# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 255, 0), 2)              # "(img, text, start_pt, font, scale, color, thickness)"
cv2.imshow("Text", output)
cv2.waitKey(0)