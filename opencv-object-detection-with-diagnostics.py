from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect Object
    object = object_cascade.detectMultiScale(frame_gray)
    #, minNeighbors=40,minSize=(100,100), maxSize=(450,450)
    for (x,y,w,h) in object:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        #Write Diagnostic Information to Screen
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame, 
                    ("X Axis: ")+str(center[0])+("  Y Axis: ")+str(center[1])+("  Height: ")+str(h)+("  Width: ")+str(w), 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv.LINE_4)

    cv.imshow('Capture - Object Detection with Diagnostics', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')

# set "default=" to location of cascade filter
parser.add_argument('--object_cascade', help='Path to Object cascade.', default='data/louis-404-cascade/cascade.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
object_cascade_name = args.object_cascade
object_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not object_cascade.load(cv.samples.findFile(object_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break