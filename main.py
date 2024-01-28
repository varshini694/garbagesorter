from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('./1.jpg')

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array


prediction = model.predict(data)
print(prediction)

prediction_score =  np.argmax(prediction)

if prediction_score == 0 :
    print("PLASTIC")
elif prediction_score ==1 :
    print("BIO-WASTE")
elif prediction_score ==2 :
    print("E-WASTE")
elif prediction_score ==3 :
    print("GLASS")
elif prediction_score ==4 :
    print("METAL") 
elif prediction_score ==5 :
    print("TRASH") 


