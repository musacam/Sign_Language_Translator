from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()
# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the layers
classifier.add(Flatten())
# Fully connected layer
classifier.add(Dense(units=128, activation='relu')) # Lets think
classifier.add(Dense(units=10, activation='softmax'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train_set = train_data.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_data.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical')
classifier.fit(
        train_set,
        steps_per_epoch=900, # No of images in training set : 0 to 9 every folder has 900 images.
        epochs=10,
        validation_data=test_set,
        validation_steps=90) # No of images in test set : 0 to 9 every folder has 90 images.

# Steps per epoch * epoch = Batch size = All of the set data

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')

