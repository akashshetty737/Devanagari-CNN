import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
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

dev_model_es = Sequential()
dev_model_es.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_width, img_height, 1)))
# change here at input shape as 28,28,1
dev_model_es.add(Conv2D(64, (3, 3), activation='relu'))
dev_model_es.add(MaxPooling2D(pool_size=(2, 2)))
dev_model_es.add(Dropout(0.25))
dev_model_es.add(Flatten())
dev_model_es.add(Dense(128, activation='relu'))
dev_model_es.add(Dropout(0.5))
dev_model_es.add(Dense(10, activation='softmax'))

dev_model_es.summary()


dev_model_es.compile(loss='binary_crossentropy',
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



#callbacks = [EarlyStopping(monitor='val_loss', patience=15),
#             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history = dev_model_es.fit_generator(
    train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    epochs = epoch,
#   callbacks = callbacks,
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
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

dev_model_es.save('dev_model_es.h5')

#filenames=test_generator.filenames
#nb_samples = len(filenames)

#best_model.evaluate_generator(generator = validation_generator, steps = nb_samples)



#test_generator.reset()
#pred=best_model.predict_generator(test_generator,verbose=1, steps = nb_samples)

#predicted_class_indices=np.argmax(pred,axis=1)

#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]



#results=pd.DataFrame({"Filenames":filenames,"Predictions":predictions})
#results.to_csv("results_es.csv",index=False)