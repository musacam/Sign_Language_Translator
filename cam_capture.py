import cv2
import os

# Train or test can change by its purpose.
mode = 'train'
directory = 'data/' + mode + '/'
timer = 0
auto_capture = 0
vid = cv2.VideoCapture(0)

while True:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)

    # Count dictionary
    count = {'b': len(os.listdir(directory + "/b"))}

    # Track number of input data.
    cv2.putText(frame, "B : " + str(count['b']), (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

    # Coordinates of the ROI
    x1 = 320
    y1 = 10
    x2 = 630
    y2 = 320
    # Drawing the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    # Frame window
    cv2.imshow("Frame", frame)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)

    #Press "ESC" to stop auto recording
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break

    #Collect training data by clicking the number
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + '0/' + str(count['zero']) + '.jpg', roi)

    #Press "S" to start Auto Capture
    if interrupt & 0xFF == ord('b'):
        auto_capture = not auto_capture

    if auto_capture == 1:
        if timer > 2:
            cv2.imwrite(directory + 'b/' + str(count['b']) + '.jpg', roi)
            timer = 0
        timer = timer + 1

    # You can change the input in order to add new input.

vid.release()
cv2.destroyAllWindows()