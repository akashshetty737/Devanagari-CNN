# Following code is for building and training CNN model for recognising Devanagari
# Digits and saving our trained model under name called as dev_model.h5
# We will use the trained model called dev_model.h5 to predict our output
# We are using Python and Keras(Tensorflow at the backend)



import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pandas as pd


# Providing dataset path
train_data_path = 'dev/train'
validation_data_path = 'dev/val'
test_data_path = 'dev/test'

epoch = 10

# model begins here

img_width, img_height = 28,28

dev_model = Sequential()
dev_model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_width, img_height, 1)))
# change here at input shape as 28,28,1
dev_model.add(Conv2D(64, (3, 3), activation='relu'))
dev_model.add(MaxPooling2D(pool_size=(2, 2)))
dev_model.add(Dropout(0.25))
dev_model.add(Flatten())
dev_model.add(Dense(128, activation='relu'))
dev_model.add(Dropout(0.5))
dev_model.add(Dense(10, activation='softmax'))

dev_model.summary()


dev_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_width, img_height),
    color_mode="grayscale",
    batch_size=1,
    class_mode="categorical",
    shuffle=False )

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

dev_model.fit_generator(
    train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    epochs = epoch,
    validation_data = validation_generator,
    validation_steps = STEP_SIZE_VALID
)

dev_model.save('dev_model.h5')

filenames=test_generator.filenames
nb_samples = len(filenames)

dev_model.evaluate_generator(generator = validation_generator, steps = nb_samples)



test_generator.reset()
pred=dev_model.predict_generator(test_generator,verbose=1, steps = nb_samples)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]



results=pd.DataFrame({"Filenames":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)