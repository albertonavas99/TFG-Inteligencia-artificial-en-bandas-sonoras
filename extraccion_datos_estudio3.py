from bs4 import BeautifulSoup
import requests
import string
import unidecode
from ytmusicapi import YTMusic
import youtube_dl

'''Se obtiene un listado de todas las bandas sonoras que hay en la página web
soundtrackcollector.com'''

url = 'http://www.soundtrackcollector.com'

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
print(letters)

dic={}

for letter in letters:
    print(letter)
    flag = 1
    number = 0
    while flag == 1:
        flag = 0
        html_content = requests.get(url+ '/title/browse/' +letter+"/" + str(number*60)).text
        soup = BeautifulSoup(html_content, "lxml")
        for entry in soup.find_all("div", {'class': 'clsListItemTitle'}):
            dic[entry.findChildren()[0].text] = url + entry.findChildren()[0].get('href')

        for link in soup.find_all("a"):
            for child in link.findChildren("img" , recursive=False):
                if 'http://img.soundtrackcollector.com/static/btn_next.gif' == child.get("src"):
                    flag = 1
        print(number)
        number += 1

with open('many_data.json', 'w') as fp:
    json.dump(dic, fp, indent=4)


'''
Se almacenan en dic_bueno los siguientes campos de todas las películas extraídas
en el anterior paso:

-url de la película en soundtrackcollector
-url de la película en imdb
-géneros cinematográficos
-compositor
-url en youtube music
'''

with open('many_data.json') as json_file:
    dic = json.load(json_file)

dic_bueno = {}
sin_genero = []
cont = 0

for title in dic:

    if cont % 100 == 0:
        with open('data_final.json', 'w') as fp:
            json.dump(dic_bueno, fp, indent=4, ensure_ascii=False)

    try:

        generos = []
        url = dic[title]

        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")

        for imdb in soup.find_all("a"):
            if 'http://www.imdb.com/Title?' in imdb.get('href'):
                link_genre = imdb.get('href')
                break

        for small in soup.find_all("small"):
            if 'Composer(s):' in small.text:
                composer= small.findNext('a').text
                composer = unidecode.unidecode(composer).strip()

        #codigo para sacar cada cancion de la película
        '''cont = 0
        for table in soup.find_all("table")[0].findChildren("table"):
            if table.get('cellpadding') == '0':
                for table2 in table.findChildren("table"):
                    if table2.get('width') == '600':
                        for td in table2.find_all("td"):
                            if '0.' in td.text or '1.' in td.text or '2.' in td.text or '3.' in td.text or '4.' in td.text or '5.' in td.text or '6.' in td.text or '7.' in td.text or '8.' in td.text or '9.' in td.text:
                                try:
                                    print(td.findNext('td').find_all("b")[0].text)
                                    cont+=1
                                except:
                                    pass
        '''



        response = requests.get(link_genre)
        html_content = response.text
        soup = BeautifulSoup(html_content, "lxml")
        for link in soup.find_all("td"):
            if 'primary_photo' in link.get("class"):
                link = link.find("a").get('href')
                break

        if link is not None and isinstance(link, str):
            link = "https://www.imdb.com/" + link
            html_content = requests.get(link).text
            soup = BeautifulSoup(html_content, "lxml")
            for div in soup.find_all("div", {'class','subtext'}):
                for genres in div.find_all("a"):
                    try:
                        if '/search/title?genres=' in genres.get('href'):
                            generos.append(genres.text)
                    except:
                        pass

            for h1 in soup.find_all("h1"):
                if h1.findNext('div').get('class')[0] == 'originalTitle':
                    titulo = h1.findNext('div').text
                    ori_title = h1.findNext('div').findChildren('span')[0].text
                    titulo  = titulo[0: -len(ori_title)]
                    titulo = unidecode.unidecode(titulo).strip()

                else:
                    titulo = h1.text
                    titulo = unidecode.unidecode(titulo).strip()

                nombre = titulo

                if ':' in nombre:
                    nombre = nombre.replace(":", "-")
                elif '/' in nombre:
                    nombre = nombre.replace("/", "-")
                elif '\\' in nombre:
                    nombre = nombre.replace("\\", "-")
                elif '<' in nombre:
                    nombre = nombre.replace("<", "-")
                elif '>' in nombre:
                    nombre = nombre.replace(">", "-")
                elif '|' in nombre:
                    nombre = nombre.replace("|", "-")
                elif '*' in nombre:
                    nombre = nombre.replace("*", " ")
                elif '"' in nombre:
                    nombre = nombre.replace('"', " ")
                elif '?' in nombre:
                    nombre = nombre.replace("?", " ")

                titulo = nombre

            if composer == 'Various' or generos == []:
                pass
            else:
                #print(str(cont) + ':' + titulo + ' - ' + str(generos) + ' - ' + composer)
                ytmusic = YTMusic('headers_auth.json')
                search_results = ytmusic.search(titulo + ' ' + composer, filter = 'songs')
                try:
                    youtube = 'https://music.youtube.com/watch?v=' + search_results[0]['videoId']
                    print(str(cont) + ':' + titulo + ' - ' + str(generos) + ' - ' + composer + ' - ' + youtube)
                    dic_bueno[titulo] = [url, link_genre, generos, composer, youtube]
                    cont += 1
                    print("------------------------------------------------------------------------")
                except:
                    search_results = ytmusic.search(titulo + ' ' + composer, filter = 'videos')
                    try:
                        youtube = 'https://music.youtube.com/watch?v=' + search_results[0]['videoId']
                        print(str(cont) + ':' + titulo + ' - ' + str(generos) + ' - ' + composer + ' - ' + youtube)
                        dic_bueno[titulo] = [url, link_genre, generos, composer, youtube]
                        cont += 1
                        print("------------------------------------------------------------------------")
                    except:
                        sin_genero.append(title)

        else:
            sin_genero.append(title)

    except KeyboardInterrupt:
        break
    except:
        sin_genero.append(title)

with open('data_final.json', 'w') as fp:
    json.dump(dic_bueno, fp, indent=4, ensure_ascii=False)


'''
Se descargan a continuación todas las bandas sonoras recopiladas en dic_bueno
'''

with open('data_final.json') as json_file:
    dic_bueno = json.load(json_file)

convertidos = os.listdir('MP3_Stereo_')

contador = 0

for title in dic_bueno:
    try:
        print(str(contador) + ': ' + title)
        if title + '.mp3' not in convertidos:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'MP3_Stereo_\\' + title + '.mp3',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([dic_bueno[title][4]])
        contador += 1
    except KeyboardInterrupt:
        break
    except:
        pass

with open('convertidos.json', 'w') as fp:
    json.dump(convertidos, fp, indent=4, ensure_ascii=False)
