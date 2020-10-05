import os
import glob

sesiones = ["Session1","Session2","Session3","Session4","Session5"]

path_evaluaciones_ptr = "IEMOCAP_full_release/" + sesiones[0] + "/dialog/EmoEvaluation/Categorical/*.txt"
path_audios = "IEMOCAP_full_release/" + sesiones[0] + "/sentences/wav/"

evaluaciones_files   = glob.glob(path_evaluaciones_ptr)

def get_emotions(path):
    audios = []
    emociones = []
    with open(path,'r') as f:
        f = f.readlines()
        for linea in f:
            temp = linea.split(" ")
            audio = temp[0]
            emocion = temp[1][1:-1]
            audios.append(audio)
            emociones.append(emocion)
    return audios, emociones


def get_dataset(n=0):
    x = []
    y = []

    for eva in evaluaciones_files:
        audios, emociones = get_emotions(eva)
        x.extend(audios)
        y.extend(emociones)

    x_paths = []

    for audio in x:
        #print(audio)
        path = str(path_audios)
        folder = audio[:-5 ]
        path += folder
        path += "/"+  audio
        path += ".wav"
        x_paths.append(path)

    return x_paths, y

