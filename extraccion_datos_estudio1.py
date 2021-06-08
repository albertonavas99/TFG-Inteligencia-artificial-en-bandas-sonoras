#Obtiene todos las películas que tienen tráiler, guarda el id de su género/s y su id del tráiler de youtube en videos.json.

import requests
import json

pagina=1
peliculas = []
videos = {}
numAux = 0
while pagina<=500:

  URL = "https://api.themoviedb.org/3/movie/popular?api_key=921e79e9f8c557835bf8b77cb450a76a&page=" + str(pagina)

  r = requests.get(url = URL)

  data = r.json()
  peliculas.extend(data['results'])
  pagina += 1


for pelicula in peliculas:
  URL = "https://api.themoviedb.org/3/movie/" + str(pelicula['id']) + "?api_key=921e79e9f8c557835bf8b77cb450a76a&append_to_response=trailers"
  r = requests.get(url = URL)

  data = r.json()

  if 'trailers' in data and 'youtube' in data['trailers'] and len(data['trailers']['youtube']) != 0:
    trailer = data['trailers']['youtube'][0]['source']
    generos = []
    for genero in data['genres']:
      generos.append(genero['id'])
    videos[data['id']] = [generos,trailer]

  else:
    pass

  numAux +=1
  print(numAux)

print("Numero de videos:" + str(len(videos)) + " de:" + str(numAux))

with open('videos.json', 'w') as fp:
    json.dump(videos, fp, sort_keys=True, indent=4)

#Carga el archivo videos.json en un diccionario llamado videos.

import json

with open('videos.json') as f_in:
    videos = json.load(f_in)

#Descarga los trailers de youtube que no estuviesen descargados y guarda el .mp4 en la carpeta Dataset_Videos, el .wav en Dataset_Audios_Stereo y el .mid en Dataset_MIDIs.

import os, subprocess
from pytube import YouTube

os.chdir('Dataset_Videos')

existentes = [f for f in os.listdir('.') if os.path.isfile(f)]

os.chdir('../')

contador=0

for video in videos:

    contador +=1
    id = str(video)
    print(str(contador)+ ":" + id)

    if id+".mp4" in existentes:
        pass
    else:
        try:
            yt = YouTube("https://www.youtube.com/watch?v=" + videos[video][-1])

            #video
            stream = yt.streams.first()
            stream.download("Dataset_Videos",filename=id)

            #audio
            subprocess.check_output("ffmpeg -i Dataset_Videos\\"+id+".mp4 Dataset_Audios_Stereo\\"+id+".wav")

            #midi
            subprocess.check_output("python2 wavToMIDI\wavToMidi.py Dataset_Audios_Stereo\\"+id+".wav Dataset_MIDIs\\"+id+".mid 60")

        except:
            pass


print("Se han procesado "+str(contador)+" trailers")

#Imprime por pantalla los vídeos o audios que no se hayan podido convertir a MIDI para borrarlos y evitar problemas.


import os

os.chdir('Dataset_Audios_Stereo')

audios = [f for f in os.listdir('.') if os.path.isfile(f)]

os.chdir('../')

os.chdir('Dataset_Videos')

videos = [f for f in os.listdir('.') if os.path.isfile(f)]

os.chdir('../')

os.chdir('Dataset_MIDIs')

midis = [f for f in os.listdir('.') if os.path.isfile(f)]

print("Audios a borrar:")
for f in audios:
    if f[:-4]+".mid" not in midis:
        print(f[:-4])

print("Videos a borrar:")
for f in videos:
    if f[:-4]+".mid" not in midis:
        print(f[:-4])

os.chdir('../')

print("acaba")

#Separa la musica y la voz en Dataset_Mono_Split.

import os, subprocess

os.chdir('Dataset_Audios_Mono')
contador = 0

for f in os.listdir('.'):
    contador += 1
    subprocess.check_output("spleeter separate -p spleeter:2stems -o ..\\Dataset_Mono_Split\\" + f[0:-4] +" " + f)
    print(str(contador)+':'+ f)

os.chdir('../')
print("acaba")

#Convierte solo la música de los trailers a imagen y lo guardo en Dataset_Imagenes_Split

import os, subprocess

os.chdir('Dataset_Mono_Split')
files = os.listdir('.')
os.chdir('../')
contador = 0

for f in files:
    contador += 1
    subprocess.check_output("ffmpeg -i Dataset_Mono_Split\\" + f + "\\" + f + "\\accompaniment.wav -lavfi showspectrumpic=s=224x224:legend=disabled Dataset_Imagenes_Split\\"+ f +".png")
    print(str(contador)+':'+ f)

print("acaba")
