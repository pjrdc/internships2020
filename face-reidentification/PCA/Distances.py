import pandas as pd
import numpy as np
import json

def Centroid(filename):
    data = pd.read_json(filename, orient='records')
    df = data.to_numpy()
    centroids = []
    sameperson = []
    previous = []
    result = []
    centroid = []
    for i in range(len(df)):
        sameperson = []
        centroid = []
        photo = df[i]
        if photo[len(photo)-1] not in previous:
            person_id = photo[len(photo)-1]
            previous.append(person_id)
            sameperson.append(photo[:len(photo)-1])
            for j in range(len(df) - 1, i, -1):
                next_photo = df[j]
                if str(person_id) == str(next_photo[len(next_photo)-1]):
                    sameperson.append(next_photo[:len(next_photo)-1])
            for dim in range(len(sameperson[0])):
                suma = []
                for k in range(len(sameperson)):
                    suma.append(sameperson[k][dim])
                centroid.append(np.mean(suma))
            centroids.append(centroid)
    return centroids, previous

def Distance_to_centroid(filename,x,y, savefile):
    data = pd.read_json(filename, orient='records')
    df = data.to_numpy()
    #centroid, group = Centroid(filename)
    centroid, group = x,y
    result = []
    for name in range(len(group)):
        array = []
        person = group[name]
        for i in range(len(df)):
            photo = df[i]
            if photo[len(photo)-1] == person:
                centr = np.array(centroid[name])
                compare = np.array(photo[:len(photo)-1])
                dist = np.linalg.norm(centr - compare)
                array.append(dist)
        result.append([array,person])
    json_object = json.dumps(result, indent=4)
    with open(savefile, "w") as outfile:
        outfile.write(json_object)

def Distance_centroids(x,y, savefile):
    centroid, group = x,y
    result = []
    for name in range(len(centroid)):
        array = []
        person = group[name]
        photo = centroid[name]
        rest = centroid[:name] + centroid[(name + 1):]
        for j in range(len(rest)):
            centr = np.array(photo)
            compare = np.array(rest[j])
            dist = np.linalg.norm(centr - compare)
            array.append(dist)
        result.append([array,person])
    json_object = json.dumps(result, indent=4)
    with open(savefile, "w") as outfile:
        outfile.write(json_object)