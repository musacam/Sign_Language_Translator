import cv2
import numpy as np
import os

# Train or test can change by its purpose.
mode = 'test'
directory = 'data/' + mode + '/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'zero': len(os.listdir(directory + "/0")),
             'one': len(os.listdir(directory + "/1")),
             'two': len(os.listdir(directory + "/2")),
             'three': len(os.listdir(directory + "/3")),
             'four': len(os.listdir(directory + "/4")),
             'five': len(os.listdir(directory + "/5")),
             'six': len(os.listdir(directory + "/6")),
             'seven': len(os.listdir(directory + "/7")),
             'eight': len(os.listdir(directory + "/8")),
             'nine': len(os.listdir(directory + "/9"))
             }

    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    ####################### Drawing the ROI ##############################
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    # Frame window
    cv2.imshow("Frame", frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)

    # Collect training data by clicking the number
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + '0/' + str(count['zero']) + '.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + '1/' + str(count['one']) + '.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + '2/' + str(count['two']) + '.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + '3/' + str(count['three']) + '.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + '4/' + str(count['four']) + '.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + '5/' + str(count['five']) + '.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory + '6/' + str(count['six']) + '.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory + '7/' + str(count['seven']) + '.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory + '8/' + str(count['eight']) + '.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory + '9/' + str(count['nine']) + '.jpg', roi)

    #This could be shortened to one line if we want specific input to train
    #But if we don't want to rerun program again and again we could think
    #to implement like this.



cap.release()
cv2.destroyAllWindows()