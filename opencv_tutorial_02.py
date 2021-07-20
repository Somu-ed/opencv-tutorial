# imports
import argparse
import imutils
import cv2

# arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load & display img
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# edge detection
ed = cv2.Canny(gray, 30, 150)           # Canny Algorithm "(img, minVal, maxVal, Aperture_size[default=3])"
cv2.imshow("Edge detect", ed)
cv2.waitKey(0)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 225, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh Inv", thresh)
cv2.waitKey(0)

# finding contours
cnt = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = imutils.grab_contours(cnt)        # Compatibility line
output = image.copy()

# loop over contours
for c in cnt:
    # draw each contour on the output image with a 3px thick purple outline
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow("Countours", output)
    cv2.waitKey(0)

# display total no. of contours as purple text
text = "I found {} objects!".format(len(cnt))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# Erosion
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Erode", mask)
cv2.waitKey(0)

# Dilation
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilate", mask)
cv2.waitKey(0)

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)