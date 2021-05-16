from tensorflow.keras.models import model_from_json
import operator
import cv2
import numpy as np

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")

vid = cv2.VideoCapture(0)

# Numbers dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR',
              5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE',
              10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
              15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'K',
              20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
              25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
              30: 'Y', 31: 'Z'}

categories = {value : key for (key, value) in categories.items()}

prediction_count = 0
predicted_word = ''
processtimer = 0

# Background subtraction implementation ?

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html


while True:
    _, frame = vid.read()
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
                  'NINE': result[0][9],
                  'A': result[0][10],
                  'B': result[0][11],
                  'C': result[0][12],
                  'D': result[0][13],
                  'E': result[0][14],
                  'F': result[0][15],
                  'G': result[0][16],
                  'H': result[0][17],
                  'I': result[0][18],
                  'K': result[0][19],
                  'L': result[0][20],
                  'M': result[0][21],
                  'N': result[0][22],
                  'O': result[0][23],
                  'P': result[0][24],
                  'R': result[0][25],
                  'S': result[0][26],
                  'T': result[0][27],
                  'U': result[0][28],
                  'V': result[0][29],
                  'Y': result[0][30],
                  'Z': result[0][31]
                  }

    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    cv2.imshow("Frame", frame)
    blackboard = np.zeros((300, 350), dtype=np.uint8)

    # Contour detection
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Hand detection and prediction on blackboard
    if len(contours) > 100:
        if prediction_count == 0:
            current_prediction = prediction[0][0]
        if current_prediction == prediction[0][0]:
            prediction_count = prediction_count + 1
        else:
            prediction_count = 0
        if prediction_count < 10:
            cv2.putText(blackboard, "Hold Still", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            processtimer = 0
        else:
            cv2.putText(blackboard, prediction[0][0], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            processtimer = processtimer + 1
        if processtimer > 10:
            predicted_word += str(categories.get(prediction[0][0]))
            processtimer = 0
    else:
        cv2.putText(blackboard, "No sign detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(blackboard, predicted_word, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Predictions', blackboard)

    # Delete last character from predicted word string by pressing S key.
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('s'):
        predicted_word = predicted_word[:-1]
    # Exit by pressing ESC key.
    if interrupt & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()
