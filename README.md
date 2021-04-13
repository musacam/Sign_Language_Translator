# Sign Language Translator

AI interpreter for Turkish Sign Language project for final capstone at Bahçeşehir University.

## How to Run

Load the needed modules. I am too lazy to create requirements text file. If you added new inputs
you need to train the data first. But you can use our trained model.
If you are ready to go just download and run `final.py`.

## What Are Those Python Files 

`cam_capture.py` for produce input data. You need to add filepath to the project and when running this press
on keyboard that you are giving to the input.

`train.py` for training and create trained model. Don't forget! Everytime you add input, you need to update
classifier.fit function. 

`final.py` for testing and observing result part.

`rotate_images.py` for enlarge the dataset and avoiding the mirror match.

`segment.py` and `threshold.py` for development purposes only let us handle it.


## TO-DO List

### Computer Engineering Dept.

- [x] Basic structure implementation
- [x] Numbers trained and desired result observed
- [x] Add rotate_images function
- [ ] Improve CNN structure
    - [ ] Add new layers maybe ?
- [ ] Think about more optimal solution about threshold values or background subtraction
    - [ ] Set hand histogram
    - [ ] BackgroundSubtractorMOG
- [ ] Add words and space structure (Optional for the last part)
- [ ] Text to speech (Optional for the last part)

### Software Engineering Dept.

- [ ] Enlarge the dataset (Currently 900 for training, 90 for test for every input)
- [ ] Add letters to the dataset
- [ ] UI implementation
- [ ] Different menus (Optional for the last part)
