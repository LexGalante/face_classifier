from requests import Request, Session
import requests
from face_classifier.model_generate import generate
import os


def create_data_set_folder():
    folder = "dataset-twitter"
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_image(url, value):
    create_data_set_folder()
    img_data = requests.get(url).content
    with open('dataset-twitter/'+value+'.jpg', 'wb') as handler:
        handler.write(img_data)


def request():
    hashTag = 'nike'
    url = 'https://instagramdimashirokovv1.p.rapidapi.com/tag/'+hashTag+'/optional'
    headers = {'x-rapidapi-key': 'a5c553e2ddmsh2f137b42ec50e8bp1ef7e3jsn1fa509e8e33a',
               'x-rapidapi-host': 'instagramdimashirokovv1.p.rapidapi.com'}

    s = Session()
    prepped = Request('GET', url, headers=headers).prepare()

    resp = s.send(prepped)
    return resp.json()


def start():
    body = request()
    edges = body['edges']

    for index in range(len(edges)):
        node = edges[index]
        url = node['node']['thumbnail_src']
        save_image(url, str(index))

    result = generate()

    print('Apareceu {}% o cr7 nos twitters da nike'.format(str(result)))


start()
