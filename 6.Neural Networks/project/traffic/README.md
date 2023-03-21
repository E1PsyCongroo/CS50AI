# Project traffic for CS50AI
```python
Epoch 1/10
500/500 [==============================] - 10s 18ms/step - loss: 2.6467 - accuracy: 0.5147 
Epoch 2/10
500/500 [==============================] - 10s 21ms/step - loss: 0.5399 - accuracy: 0.8412
Epoch 3/10
500/500 [==============================] - 11s 21ms/step - loss: 0.2895 - accuracy: 0.9179
Epoch 4/10
500/500 [==============================] - 10s 20ms/step - loss: 0.2465 - accuracy: 0.9343
Epoch 5/10
500/500 [==============================] - 10s 19ms/step - loss: 0.1930 - accuracy: 0.9478
Epoch 6/10
500/500 [==============================] - 9s 18ms/step - loss: 0.1829 - accuracy: 0.9536
Epoch 7/10
500/500 [==============================] - 10s 20ms/step - loss: 0.1571 - accuracy: 0.9571
Epoch 8/10
500/500 [==============================] - 8s 17ms/step - loss: 0.1589 - accuracy: 0.9600
Epoch 9/10
500/500 [==============================] - 9s 17ms/step - loss: 0.1822 - accuracy: 0.9575
Epoch 10/10
500/500 [==============================] - 9s 17ms/step - loss: 0.1048 - accuracy: 0.9743
333/333 - 1s - loss: 0.0932 - accuracy: 0.9792 - 1s/epoch - 4ms/step
```

- The hidden layers are very significant
  - I have tried two versions of the hidden layer but nither was ideal
    ```python
      tf.keras.layers.Dense(NUM_CATEGORIES * 128, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(NUM_CATEGORIES * 64, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(NUM_CATEGORIES * 32, activation="relu"),
      tf.keras.layers.Dropout(0.2),
    ```
    This version has too many units to the inefficiency of the training
    ```python
      tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(NUM_CATEGORIES * 8, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(NUM_CATEGORIES * 4, activation="relu"),
      tf.keras.layers.Dropout(0.2),
    ```
    This version has too few cells in the hidden layer to reasonably classify the data images

