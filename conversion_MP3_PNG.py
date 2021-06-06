import librosa
import librosa.display
import pylab
from matplotlib import cm
import os
from pydub import AudioSegment
import numpy as np
import sys

files = os.listdir("MP3_Stereo_")
files_converted = os.listdir("PNGs_")
cont = 0
fallos = []

for file in files:

    if cont == 1000:
        break

    if file[:-4] + '.png' not in files_converted:

        try:

            print(str(cont) + ':' + file[:-4])

            y, sr = librosa.load("MP3_Stereo_\\" + file)

            mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)


            pylab.figure(figsize=(3,3))
            pylab.axis('off')
            librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
            pylab.savefig('PNGs_\\' + file[:-4] + '.png')
            pylab.close()

            cont += 1

        except KeyboardInterrupt:
            break
    
