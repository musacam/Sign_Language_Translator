# Sign Language Translator

AI interpreter for Turkish Sign Language.

## How to Run

Load the needed modules via requirements file. If you added new inputs
you need to train the data first. But you can use our trained model.
If you are ready to go just download and run `final.py`.

## What Are Those Python Files 

`cam_capture.py` for produce input data. You need to add filepath to the project and when running this press
on keyboard that you are giving to the input.

`train.py` for training and create trained model. Don't forget! Everytime you add input, you need to update
classifier.fit function. 

`final.py` for testing and observing result part.

`rotate_images.py` for enlarge the dataset and avoiding the mirror match. If you need to...

`segment.py` for development purposes only let us handle it.


## TO-DO List

### Computer Engineering Dept.

- [ ] CNN structure improvement (Always need improvement)

#### Finished Tasks

- [x] Basic structure implementation
- [x] Numbers trained and desired result observed
- [x] Add rotate_images function
- [x] Think about moving signs and ask for advise
- [x] Add Dropout layers
- [x] Efficient output display (Countdown or steady still from user)
- [x] Implementation and integration of bg extraction if its needed (Better camera worked)
- [x] Accuracy increase
- [x] Add letters to the dataset (Not our job!)
- [x] Input to memory to sentence structure
- [x] Upgrade predicted word structure (Mostly done)

### Software Engineering Dept.

- [ ] Manage the dataset (Currently 900 for training, 90 for test for every input)
- [ ] Add letters to the dataset
- [ ] UI implementation
- [ ] Different menus (Optional for the last part)

#### Finished Tasks