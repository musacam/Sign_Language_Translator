# Sign_Language_Translator

Sign Language Translator project for final capstone project at Bahçeşehir University.


## How to run

Load the needed modules. I am too lazy to create requirements text file.

`cam_capture.py` for produce input data.

`train.py` for training and create json and h5 model.

`final.py` for testing and observing result part.

`segment.py` and `threshold.py` for development purposes only let us handle it.


## TO-DO List

### Computer Engineering Dept.

- [x] Basic structure implementation
- [x] Numbers trained and desired result observed
- [ ] Add letters and see if they work
- [ ] Improve CNN structure
- [ ] Think about more optimal solution about threshold values
    - [ ] Set hand histogram
- [ ] Add words and space structure (Optional for the last part)
- [ ] Text to speech (Optional for the last part)

### Software Engineering Dept.

- [ ] Enlargen dataset (Currently 900 for training 90 for test for every input)
- [ ] UI implementation
- [ ] Different menus (Optional for the last part)
