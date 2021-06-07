import pandas as pd
import json, os
import tensorflow as tf
import tensorflow as tf
from tensorflow.python.keras import optimizers
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers

with open('data_final.json') as json_file:
    dic_bueno = json.load(json_file)

#Poner nombre de la categoría que se vaya a entrenar
name_cat='Comedy'

#Coge tantos archivos que pertenezcan al género como archivos que no.
filenames = os.listdir("PNGs_")

dif = 0
categories = []
filenames_selected = []
cont_cat = 0
cont_no_cat = 0

for filename in filenames:
    film = filename[0:-4]
    category = dic_bueno[film][2]

    if name_cat in category:
        filenames_selected.append(filename)
        categories.append(name_cat)
        dif += 1
        cont_cat += 1
    else:
        if dif > 0:
            filenames_selected.append(filename)
            categories.append("non-" + name_cat)
            dif -= 1
            cont_no_cat += 1

df = pd.DataFrame({
    'filename': filenames_selected,
    'category': categories
})

print(cont_cat)
print(cont_no_cat)
print(df)

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence

#Se crean los df de train y test con un split del 75-25
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

datos_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,        # Normalizar la imagen
    zoom_range=0.2,        # Aplicar distintos zooms a las imágenes de training
    horizontal_flip=True   # Voltear las imágenes (modo espejo)
)

datos_test =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

tamanyo_batch=32 #64,128,256

generador_train = datos_train.flow_from_dataframe(
    df_train,
    "PNGs_",
    x_col='filename',
    y_col='category',
    target_size= (216,216),
    class_mode= 'binary',
    batch_size= tamanyo_batch
)

generador_test = datos_test.flow_from_dataframe(
    df_test,
    "PNGs_",
    x_col='filename',
    y_col='category',
    target_size=(216,216),
    class_mode= 'binary',
    batch_size=tamanyo_batch
)

#Red neuronal secuencial (Varias capas apiladas)
model = tf.keras.models.Sequential()

#La primera capa va a ser una convolución (reducir imagen) con 128 filtros, con un kernel de 3x3,
#con padding, y la función de activación es relu
model.add(tf.keras.layers.Conv2D(128,(3,3), padding='same',  activation = 'relu', input_shape = (216,216,3)))

#Capa de pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.Dropout(0.4))

#Poner la imagen que es muy profunda pero muy pequeña en una dimensión
model.add(tf.keras.layers.Flatten())

#Capa con 256 neuronas conectadas a la imagen aplanada en el paso anterior
model.add(tf.keras.layers.Dense(256, activation='relu'))


#Capa de normalizacion (entre 0.5 y 0.8).
model.add(tf.keras.layers.Dropout(0.2))

# Añadimos una capa sigmoid para que podamos clasificar las imágenes
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2 = model.fit_generator(
    generador_train,
    epochs=200,
    validation_data=generador_test,
    validation_steps=df_test.shape[0]//tamanyo_batch,
    steps_per_epoch=df_train.shape[0]//tamanyo_batch
)

from sklearn.externals import joblib

target_dir = 'modelos\\CNNs_binarias_'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('modelos\\CNNs_binarias_\\'+name_cat+'-v2.h5')
model.save_weights('modelos\\CNNs_binarias_\\'+name_cat+'-v2-pesos.h5')
with open('modelos\\CNNs_binarias_\\'+name_cat+'-v2.plk', 'wb') as f:
    joblib.dump(model2.history, f)

import matplotlib.pyplot as plt

acc = model2.history['accuracy']
val_acc = model2.history['val_accuracy']
loss = model2.history['loss']
val_loss = model2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
