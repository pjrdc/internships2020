import pandas as pd
from statistics import mean, median
import json
import matplotlib.pyplot as plt
import numpy as np

'''
[0] - person
[1] - min
[2] - max
[3] - avg
[4] -  mediana
'''

def Informations(openfile,savefile):
    finalarr = []
    data = pd.read_json(openfile, orient='records')
    data = data.to_numpy()
    for i in range(len(data)):
        person = data[i]
        values = person[0]
        min_val_same = min(values)
        max_val_same = max(values)
        avg_val_same = mean(values)
        median_val_same = median(values)
        info = [str(person[1]), str(min_val_same), str(max_val_same),
                str(avg_val_same), str(median_val_same)]
        finalarr.append(info)
    json_object = json.dumps(finalarr, indent=4)
    with open(savefile, "w") as outfile:
        outfile.write(json_object)

def Analysis(dimensions,saveplot,info_to_path,info_centroid_path):
    for dimension in range(dimensions):
        info_to = pd.read_json(info_to_path + str(dimension) + '.json')
        info_centroid = pd.read_json(info_centroid_path + str(dimension) + '.json')
        saveplot = 'Comparisons/NS/Analiza/plot' + str(dimension) + '.png'
        correct_avg = 0
        wrong_avg = 0
        correct_med = 0
        wrong_med = 0
        for person_nr in range(len(info_to[0])):
            person_id = info_to[0][person_nr]
            avg_to = info_to[3][person_nr]
            med_to = info_to[3][person_nr]
            avg_centroid = info_centroid[4][person_nr]
            med_centroid = info_centroid[4][person_nr]
            dist_avg = avg_to - avg_centroid
            dist_med = med_to - med_centroid
            if dist_avg > 0:
                wrong_avg += 1
            else:
                correct_avg += 1
            if dist_med > 0:
                wrong_med += 1
            else:
                correct_med += 1
        n_groups = 2
        wrong = (wrong_avg, wrong_med)
        print(wrong)
        correct = (correct_avg, correct_med)
        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
        rects1 = plt.bar(index, wrong, bar_width,
                         alpha=opacity,
                         color='r',
                         label='Wrong')

        rects2 = plt.bar(index + bar_width, correct, bar_width,
                         alpha=opacity,
                         color='g',
                         label='Correct')
        plt.xlabel('Klasyfikacja')
        plt.ylabel('Ilosc')
        plt.title('Ilosc klasyfikacji dla ' + str(dimension) + 'wymiarów')
        plt.xticks(index + bar_width, ('Srednia', 'Mediana'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(saveplot)
        plt.show()