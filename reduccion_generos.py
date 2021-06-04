import os,json,copy

'''
Se realiza la reducción de géneros para la segunda extracción de datos
'''

with open('data.json') as json_file:
    dic = json.load(json_file)

for data in dic:
    if isinstance(dic[data][2], list):
        generos = copy.copy(dic[data][2])
        for genero in generos:
            if genero == 'Family' or genero == 'History' or genero == 'Biography' or genero == 'Animation':
                dic[data][2].remove(genero)
            if genero == 'Adventure' or genero == 'War':
                dic[data][2].remove(genero)
                if 'Action' not in dic[data][2]:
                    dic[data][2].append('Action')
            if genero == 'Musical':
                dic[data][2].remove(genero)
                if 'Music' not in dic[data][2]:
                    dic[data][2].append('Music')
            if genero == 'Crime':
                dic[data][2].remove(genero)
                if 'Thriller' not in dic[data][2]:
                    dic[data][2].append('Thriller')
    else:
        genero = dic[data][2]
        if genero == 'Family' or genero == 'History' or genero == 'Biography' or genero == 'Animation':
            print("NO DEBERIA ENTRAR")
        if genero == 'Adventure' or genero == 'War':
            dic[data][2] = 'Action'
        if genero == 'Musical':
            dic[data][2]='Music'
        if genero == 'Crime':
            dic[data][2]='Thriller'

with open('data.json', 'w') as fp:
    json.dump(dic, fp, indent=4)


'''
Se realiza la reducción de géneros para la tercera extracción de datos
'''

with open('data_final.json') as json_file:
    dic_bueno = json.load(json_file)

descargados = os.listdir('MP3_Stereo_')

for peli in descargados:
    for genero in dic_bueno[peli[:-4]][2]:
        if genero == 'Family' or genero == 'History' or genero == 'Biography' or genero == 'Animation' or genero == 'Documentary' or genero == 'Film-Noir' or genero == 'Short' or genero == 'Talk-Show' or genero == 'Reality-TV' or genero == 'News' or genero == 'Game-Show':
            dic_bueno[peli[:-4]][2].remove(genero)
        if genero == 'Adventure' or genero == 'War':
            dic_bueno[peli[:-4]][2].remove(genero)
            if 'Action' not in dic_bueno[peli[:-4]][2]:
                dic_bueno[peli[:-4]][2].append('Action')
        if genero == 'Musical':
            dic_bueno[peli[:-4]][2].remove(genero)
            if 'Music' not in dic_bueno[peli[:-4]][2]:
                dic_bueno[peli[:-4]][2].append('Music')
        if genero == 'Crime':
            dic_bueno[peli[:-4]][2].remove(genero)
            if 'Thriller' not in dic_bueno[peli[:-4]][2]:
                dic_bueno[peli[:-4]][2].append('Thriller')

with open('data_final.json', 'w') as fp:
    json.dump(dic_bueno, fp, indent=4, ensure_ascii=False)
