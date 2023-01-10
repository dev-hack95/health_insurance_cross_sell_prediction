import datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("./data/raw/train_processed.csv")
df = df.sample(n=50000)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

model_dl = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=0.1))
    ])

model_dl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cb_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),  tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

history = model_dl.fit(x_train, y_train,
                   epochs=60,
                   validation_split=0.3, callbacks=cb_list)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.savefig("accuracy.png")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savfig("loss.png")
