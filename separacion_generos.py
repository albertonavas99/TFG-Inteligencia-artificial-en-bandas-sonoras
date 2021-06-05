import shutil
import json

generos = [Comedy, Drama, Sport, Music, Romance, Mystery, Sci-Fi, Thriller, Fantasy, Western, Action, Horror]

with open('data.json') as json_file:
    dic = json.load(json_file)

for gen in generos:

    for data in dic:
        if isinstance(dic[data][2], list):
            for genero in dic[data][2]:
                if genero == gen:
                    print(data)
                    shutil.copy('MIDIs\\' + data + '.mid', 'Magenta\\'+gen+'\\MIDIs\\' + data + '.mid')

        else:
            if dic[data][2]  == gen:
                print(data)
                shutil.copy('MIDIs\\' + data + '.mid', 'Magenta\\'+gen+'\\MIDIs\\' + data + '.mid')
