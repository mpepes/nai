#Authors: s19916 Kibort Jan & s19333 Michałek Piotr

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 15
STOP_ADV = False

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")

time.sleep(1.0)


cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(args["video"])
if (cap.isOpened()== False):
	  print("Error opening video stream, run program again")
 
ret_adv, frame_adv = video.read()

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    # if fileStream and not vs.more():
    #     break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    for rect in rects:
        if not STOP_ADV:
            ret_adv, frame_adv = video.read()
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER > EYE_AR_CONSEC_FRAMES:
                STOP_ADV = True
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
            STOP_ADV = False
        # reset the eye frame counter
            COUNTER = 0


    cv2.imshow("Frame", frame)
    cv2.imshow("Adv", frame_adv)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
