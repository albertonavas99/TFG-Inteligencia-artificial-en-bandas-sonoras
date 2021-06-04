from bs4 import BeautifulSoup
import requests, json
import string

'''Se obtiene un listado de todas las bandas sonoras que hay en la página web
freemidi.org con los links de descarga tanto de los MIDIs como de los MP3 y
se guardan en dic.'''

url = 'https://freemidi.org/moviethemes-'

letters = string.ascii_lowercase

dic={}

cont = 0

for letter in letters:
    html_content = requests.get(url+letter).text
    soup = BeautifulSoup(html_content, "lxml")
    for music in soup.find_all("a"):
        if 'download' in music.get("href"):
            title = music.text
            id = music.get("href").split("-")[1]
            midi_url = "https://freemidi.org/getter-" + id
            mp3_url = "https://freemidi.org/getterm-" + id
            print(str(cont) + ": " + title)
            dic[title] = [midi_url,mp3_url,[]]
            cont += 1

html_content = requests.get(url+'0').text
soup = BeautifulSoup(html_content, "lxml")
for music in soup.find_all("a"):
    if 'download' in music.get("href"):
        title = music.text
        id = music.get("href").split("-")[1]
        midi_url = "https://freemidi.org/getter-" + id
        mp3_url = "https://freemidi.org/getterm-" + id
        print(str(cont) + ": " + title)
        dic[title] = [midi_url,mp3_url,[]]
        cont += 1


'''
Se obtienen los géneros de las películas que están en dic buscando esas películas
en imdb.com y extrayendo esos géneros del HTML
'''

url = "https://www.imdb.com/"

for nombre in dic.keys():

    data = None
    link = None

    nombre_mas = nombre.replace(" ", "+")
    print(nombre)

    response = requests.get(url+'find?q='+nombre_mas+'&s=tt&ttype=ft&ref_=fn_ft')
    html_content = response.text

    soup = BeautifulSoup(html_content, "lxml")

    for link in soup.find_all("td"):

        if 'primary_photo' in link.get("class"):
            link = link.find("a").get('href')
            break

    if link is not None:
        html_content = requests.get(url+link).text

        soup = BeautifulSoup(html_content, "lxml")

        data = json.loads(soup.find('script', type='application/ld+json').contents[0])
        print(data['genre'])
        dic[nombre][2] = data['genre']
    else:
        print("No encuentra película")

with open('data.json', 'w') as fp:
    json.dump(dic, fp, indent=4)

'''
Se descargan los MIDIs y los MP3 de los cuales se tenía ya su URL.
'''


with open('data.json') as json_file:
    dic = json.load(json_file)

cont = 0
problemas = []

for value in dic:

    nombre = value

    if ':' in nombre:
        nombre = nombre.replace(":", "-")

    print(str(cont) + ": " + nombre)

    midi_url = dic[value][0]
    mp3_url = dic[value][1]

    headers = {
                "Host": "freemidi.org",
                "Connection": "keep-alive",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
               }

    session = requests.Session()

    r = session.get(midi_url, headers=headers)
    r = session.get(midi_url, headers=headers)

    try:

        with open("MIDIs\\" + nombre + ".mid",'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)


        r = session.get(mp3_url, headers=headers)

        with open("MP3_Stereo\\"+ nombre + ".mp3",'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

    except:
        problemas.append(nombre)

    cont += 1
