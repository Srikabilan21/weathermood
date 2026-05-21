import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train = ImageDataGenerator(rescale=1./255)

data = train.flow_from_directory(
        "dataset",
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, epochs=5)

model.save("weather_model.h5")