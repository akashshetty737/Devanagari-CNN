from scipy.misc import imread, imresize
import numpy as np
import imageio
from PIL import Image
x = imageio.imread('7259.png',pilmode='L')
#compute a bit-wise inversion so black becomes white and vice versa
x = np.invert(x)
#make it the right size
x =np.array(Image.fromarray(x).resize([28,28]))
#convert to a 4D tensor to feed into our model
x = x.reshape(1,28,28,1)
x = x.astype('float32')
x /= 255

#perform the prediction
from keras.models import load_model
model = load_model('dev_model.h5')
out = model.predict(x)
print(np.argmax(out))