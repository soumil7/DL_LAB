from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Enhanced ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    epochs=5
)

# Prediction on a single image
image_path = 'train/dog/dog.1.jpg'
img = Image.open(image_path)
img = img.resize((128, 128))
img_array = np.array(img)
img_array = img_array / 255.0  # Normalize pixel values
image = np.expand_dims(img_array, axis=0)  # Add a batch dimension

prediction = model.predict(image)

if prediction[0][0] <= 0.5:
    print("DOG")
else:
    print("CAT")
