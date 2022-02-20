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

## How Does UI Looks Like?

![Our UI](https://github.com/musacam/Sign_Language_Translator/blob/main/UI.jpg)
