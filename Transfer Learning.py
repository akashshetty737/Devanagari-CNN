import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


# Providing dataset path
train_data_path = 'dev/train'
validation_data_path = 'dev/val'
test_data_path = 'dev/test'

epoch = 100

# model begins here

img_width, img_height = 28,28

#use model that have been used for MNIST datset

model_tfl = load_model('cnn.h5')

#transfer learning

for layer in model_tfl.layers[:5]:
   layer.trainable = False

model_tfl.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model_tfl.summary()

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



callbacks = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='best_model_tfl2.h5', monitor='val_loss', save_best_only=True)]

history = model_tfl.fit_generator(
    train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    epochs = epoch,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = STEP_SIZE_VALID
)

print(history.history.keys())


#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


filenames=test_generator.filenames
nb_samples = len(filenames)

model_tfl.evaluate_generator(generator = validation_generator, steps = nb_samples)



test_generator.reset()
pred=model_tfl.predict_generator(test_generator,verbose=1, steps = nb_samples)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]



results4=pd.DataFrame({"Filenames":filenames,"Predictions":predictions})
results4.to_csv("results4.csv",index=False)