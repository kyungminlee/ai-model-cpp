import tensorflow as tf
import keras

def create_model():
    model = keras.Sequential([
      keras.layers.Conv2D(6, kernel_size=5, activation='relu'),
      keras.layers.MaxPool2D((2,2)),
      keras.layers.Conv2D(16, kernel_size=5, activation='relu'),
      keras.layers.MaxPool2D((2,2)),
      keras.layers.Reshape((16*5*5,)),
      keras.layers.Dense(120, activation='relu'),
      keras.layers.Dense(84, activation='relu'),
      keras.layers.Dense(10, activation='relu'),
    ])
    model.build(input_shape=(1, 32, 32, 3))
    return model

model = create_model()
w = model.get_weights()
for x in w:
    print(x.shape)
    x[...] = 0.1
model.set_weights(w)
# model.save('model.keras')
tf.saved_model.save(model, 'model.tf')
model.summary()

print(model(tf.ones((1, 32, 32, 3))))