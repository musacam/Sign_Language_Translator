from keras.models import model_from_json
import operator
import cv2

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")

cap = cv2.VideoCapture(0)

# Numbers dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR',
              5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE'}

# Background subtraction implementation ?

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#exercises


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Coordinates of the ROI
    x1 = 320
    y1 = 10
    x2 = 630
    y2 = 320
    # Drawing the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    # Resizing the ROI
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    cv2.imshow("test", test_image)

    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))

    prediction = {'ZERO': result[0][0],
                  'ONE': result[0][1],
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'SIX': result[0][6],
                  'SEVEN': result[0][7],
                  'EIGHT': result[0][8],
                  'NINE': result[0][9]}

    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    # Displaying the prediction
    cv2.putText(frame, prediction[0][0], (320, 360), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
