from keras.models import load_model
from keras.preprocessing import image
import imageio
import numpy as np
from keras.utils import plot_model
from matplotlib.pyplot import imshow

#returns a compiled model, identical to the previous one
model = load_model('my_model.h5')
#model.summary()


img_path = 'my0.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
print('Input image shape:', x.shape)

my_image = imageio.imread(img_path)
imshow(my_image)
print("Class Prediction Vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))
print("Estimation is " +str(np.argmax(model.predict(x))))
