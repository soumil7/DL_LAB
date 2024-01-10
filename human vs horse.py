#pip install Pillow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))  # Assuming binary classification
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Flow training images in batches using the generator
train_generator = ImageDataGenerator().flow_from_directory(
    r'train2',
    target_size=(128, 128),
    class_mode='categorical',
)

# Train the model
model.fit(
    train_generator,
    epochs=5
)

# Use Pillow's Image module to load and preprocess the image
image_path = r'C:\DL_LAB_PREV\DL_Final\EXP_9 CNN HUMAN AND HORSE\train2\horses\horse01-0.png'
img = Image.open(image_path).convert("RGB")  # Convert to RGB if the image has an alpha channel
img = img.resize((128, 128))
img_array = np.array(img)
img_array = img_array / 255.0  # Normalize pixel values
image = np.expand_dims(img_array, axis=0)  # Add a batch dimension

prediction = model.predict(image)

if prediction[0][0] <= 0.5:
    print("HORSE")
else:
    print("HUMAN")
