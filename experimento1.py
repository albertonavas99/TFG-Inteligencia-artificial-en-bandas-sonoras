import pandas as pd
import json, os
import tensorflow as tf

#Se hace el DataFrame con los géneros y los nombres de los archivos

with open('data.json') as f_in:
    dic = json.load(f_in)

filenames = os.listdir("PNGs")

genero = {'Comedy':0, 'Drama':1, 'Sport':2, 'Music':3, 'Romance':4, 'Mystery':5, 'Sci-Fi':6, 'Thriller':7, 'Action':8, 'Fantasy':9, 'Western':10, 'Horror':11}

categories = []
for filename in filenames:
    film = filename[0:-4]
    category = dic[film][2]
    if isinstance(category, list):
        lista = []
        for cat in category:
            lista.append(genero[cat])
        categories.append(lista)
    else:
        lista = []
        lista.append(genero[category])
        categories.append(lista)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(df)

#Se hace One-Hot Encoding

from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

genero = {0:'Comedy', 1:'Drama', 2:'Sport', 3:'Music', 4:'Romance', 5:'Mystery', 6:'Sci-Fi', 7:'Thriller', 8:'Action', 9:'Fantasy', 10:'Western', 11:'Horror'}

mlb = MultiLabelBinarizer()


df2 = pd.DataFrame(mlb.fit_transform(df["category"]),columns=mlb.classes_)


df2.columns=[genero[x] for x in df2.columns]

df_row = pd.concat([df, df2],axis=1)
del df_row['category']

print(df_row)

from sklearn.model_selection import train_test_split
import tensorflow as tf

# Se crean los df de train y test con un split del 75-25
df_train, df_test = train_test_split(df_row, test_size=0.25, random_state=42)

datos_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,        # Normalizar la imagen
    zoom_range=0.2,        # Aplicar distintos zooms a las imágenes de training
    horizontal_flip=True   # Voltear las imágenes (modo espejo)
)

datos_test =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

tamanyo_batch=16 #64,128,256

categories = ['Comedy', 'Drama', 'Sport', 'Music', 'Romance', 'Mystery', 'Sci-Fi', 'Thriller', 'Action', 'Fantasy','Western','Horror']

generador_train = datos_train.flow_from_dataframe(
    df_train,
    "PNGs",
    x_col='filename',
    y_col=categories,
    target_size= (224,224),
    class_mode= 'raw',
    batch_size= tamanyo_batch
)

generador_test = datos_test.flow_from_dataframe(
    df_test,
    "PNGs",
    x_col='filename',
    y_col=categories,
    target_size=(224,224),
    class_mode='raw',
    batch_size=tamanyo_batch
)

#Se compone la CNN y se entrena
import tensorflow as tf
from tensorflow.python.keras import optimizers
from keras.layers.normalization import BatchNormalization

#Red neuronal secuencial (Varias capas apiladas)
model = tf.keras.models.Sequential()

#La primera capa va a ser una convolución (reducir imagen) con 32 filtros, con un kernel de 3x3,
#con padding, y la función de activación es relu
model.add(tf.keras.layers.Conv2D(32,(3,3), padding='same', activation = 'relu', input_shape = (224,224,3)))

#Capa de pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

#Más capas iguales incrementando los filtros
model.add(tf.keras.layers.Conv2D(64,(3,3), padding='same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(128,(3,3), padding='same', activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

#Poner la imagen que es muy profunda pero muy pequeña en una dimensión
model.add(tf.keras.layers.Flatten())

#Capa con 256 neuronas conectadas a la imagen aplanada en el paso anterior
model.add(tf.keras.layers.Dense(256, activation='relu'))

#Capa con 512 neuronas
model.add(tf.keras.layers.Dense(512, activation='relu'))

#Capa de normalizacion (entre 0.5 y 0.8).
model.add(tf.keras.layers.Dropout(0.5))

# Añadimos una capa sigmoid para que podamos clasificar las imágenes
model.add(tf.keras.layers.Dense(12, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2 = model.fit_generator(
    generador_train,
    epochs=150,
    validation_data=generador_test,
    validation_steps=df_test.shape[0]//tamanyo_batch,
    steps_per_epoch=df_train.shape[0]//tamanyo_batch
)

target_dir = 'modelos'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('modelos\\DifGeneros.h5')
model.save_weights('modelos\\DifGeneros-pesos.h5')


#Gráficos con resultados del entrenamiento y la validación

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


#Creación de matriz de confusión y tabla resumen

import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from pytube import YouTube
import os, subprocess
from pydub import AudioSegment
from platform import python_version_tuple

modelo = 'modelos\\DifGeneros.h5'
pesos_modelo = 'modelos\\DifGeneros-pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

x, y = izip(*(generador_test[i] for i in xrange(len(generador_test))))
X_test, y_test = np.vstack(x), np.vstack(y)

pred = cnn.predict(X_test, batch_size=tamanyo_batch, verbose=1)

import seaborn as sns


cm = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]])
total = [0,0,0,0,0,0,0,0,0,0,0,0]

for porcentajes, sionos in zip(pred, y_test):
    cont = 0
    si = []
    for siono in sionos:
        if siono == 1:
            si.append(cont)
            total[cont] += 1
        cont+=1
    cont = 0
    predicciones = []
    for porcentaje in porcentajes:
        if porcentaje >= 0.5:
            predicciones.append(cont)
        cont+=1

    quitar_si = []
    for s in si:
        if not predicciones:
            cm[s][12] += 1
        quitar_pred = []
        for prediccion in predicciones:
            if s == prediccion:
                cm[s][s] += 1
                quitar_si.append(s)
                quitar_pred.append(prediccion)
        for quita in quitar_pred:
            predicciones.remove(quita)

    for quita in quitar_si:
        si.remove(quita)

    for s in si:
        for prediccion in predicciones:
            cm[s][prediccion] += 1


#porcentaje de acierto

aciertos = [0,0,0,0,0,0,0,0,0,0,0,0]
cont = 0
for genero in cm:
    aciertos[cont] = genero[cont]/total[cont]
    cont += 1

print("Porcentaje de aciertos:")
print(aciertos)

print("Total:")
print(total)


# Visualiamos la matriz de confusión
clases_v2 = ['0:Comedy', '1:Drama', '2:Sport', '3:Music', '4:Romance', '5:Mystery', '6:Sci-Fi', '7:Thriller', '8:Action', '9:Fantasy', '10:Western', '11:Horror']

clases_x = ['0:Comedy', '1:Drama', '2:Sport', '3:Music', '4:Romance', '5:Mystery', '6:Sci-Fi', '7:Thriller', '8:Action', '9:Fantasy', '10:Western', '11:Horror','no predice ninguna']


con_mat_df = pd.DataFrame(cm, index = clases_v2, columns = clases_x)

figure = plt.figure(figsize=(13, 12))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues,fmt='g')

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


tabla = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]])

cont = 0
for acierto in aciertos:
    acierto = round(acierto*100, 2)
    tabla[0][cont] = acierto
    cont += 1

cont = 0
for tot in total:
    tabla[1][cont] = tot
    cont += 1

total_dataset = [0,0,0,0,0,0,0,0,0,0,0,0]

asignacion = {'Comedy':0, 'Drama':1, 'Sport':2, 'Music':3, 'Romance':4, 'Mystery':5, 'Sci-Fi':6, 'Thriller':7, 'Action':8, 'Fantasy':9, 'Western':10, 'Horror':11}

for filename in filenames:
    film = filename[0:-4]
    generos = dic[film][2]
    if isinstance(generos, list):
        for genero in generos:
            total_dataset[asignacion[genero]] += 1
    else:
        total_dataset[asignacion[generos]] += 1

cont = 0
for tot in total_dataset:
    tabla[2][cont] = tot
    cont += 1

con_mat_df = pd.DataFrame(tabla, index = ['porcentaje aciertos (%)', 'total validation', 'total dataset'], columns = clases_v2)

figure = plt.figure(figsize=(12, 3))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues, fmt='g')

plt.tight_layout()
plt.show() 
