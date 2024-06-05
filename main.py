# Import required libraries
import cv2  # OpenCV library for computer vision tasks
import math  # Math functions and constants
import argparse  # Argument parsing library for command-line interfaces

# Function to highlight faces in an image
def highlightFace(net, frame, conf_threshold=0.7):
    # Make a copy of the frame
    frameOpencvDnn = frame.copy()
    # Get the height and width of the frame
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # Set the input to the network
    net.setInput(blob)
    # Perform face detection
    detections = net.forward()
    # Initialize list to store detected face boxes
    faceBoxes = []
    # Loop through the detections
    for i in range(detections.shape[2]):
        # Get the confidence of the detection
        confidence = detections[0, 0, i, 2]
        # Check if the confidence is above the threshold
        if confidence > conf_threshold:
            # Calculate the coordinates of the face box
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # Add the face box coordinates to the list
            faceBoxes.append([x1, y1, x2, y2])
            # Draw a rectangle around the face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    # Return the frame with highlighted faces and the list of face boxes
    return frameOpencvDnn, faceBoxes

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

# File paths for face detection, age detection, and gender detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values for age detection model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# List of age ranges
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# List of genders
genderList = ['Male', 'Female']

# Read face detection, age detection, and gender detection models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open the video capture device (camera)
video = cv2.VideoCapture(args.image if args.image else 0)
# Padding for face extraction
padding = 20

# Main loop for video processing
while True:
    # Read frame from video capture device
    hasFrame, frame = video.read()
    # If no frame is available, exit the loop
    if not hasFrame:
        cv2.waitKey()
        break
    # Highlight faces in the frame
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    # If no faces are detected, print a message
    if not faceBoxes:
        print("No face detected")
    # Loop through detected face boxes
    for faceBox in faceBoxes:
        # Extract the face region from the frame
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                     :min(faceBox[2] + padding, frame.shape[1] - 1)]
        # Create a blob from the face region
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Set input to gender detection model
        genderNet.setInput(blob)
        # Perform gender detection
        genderPreds = genderNet.forward()
        # Get the predicted gender
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        # Set input to age detection model
        ageNet.setInput(blob)
        # Perform age detection
        agePreds = ageNet.forward()
        # Get the predicted age range
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        # Put text on the result image with gender and age information
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        # Display the result image with annotations
        cv2.imshow("Detecting age and gender", resultImg)

    # Wait for a key press for a short duration
    key = cv2.waitKey(1)
    # If 'q' is pressed or the window is closed, exit the loop
    if key == ord("q") or cv2.getWindowProperty("Detecting age and gender", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release video capture device and close all windows
video.release()
cv2.destroyAllWindows()
