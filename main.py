#importing library
import cv2
import numpy as np
#vudeo capture
capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.namedWindow("Window")

while True:
    #reading frames in images formate
    ret, image = capture.read()
    # resizing the frame
    image_resized = cv2.resize(image, None, fx=1, fy=1)
    # removing impurities from frames
    image_cleared = cv2.medianBlur(image_resized, 3)
    image_cleared= cv2.medianBlur(image_cleared, 3)
    image_cleared=cv2.medianBlur(image_cleared, 3)
    image_cleared=cv2.edgePreservingFilter(image_cleared, sigma_s=15)

    #bilatreal image filtering
    image_filtered = cv2.bilateralFilter(image_cleared, 3, 20, 5)
    for i in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)
    for i in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 5, 40, 10)

    #sharpening the image
    g_mask = cv2.GaussianBlur(image_filtered, (9,9), 5)
    image_sharp = cv2.addWeighted(image_filtered,1.5, g_mask, -0.5, 0)
    image_sharp=cv2.addWeighted(image_sharp,1.4, g_mask, -0.2, 20)
    #printing the frame
    cv2.imshow('Final Image', image_sharp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
