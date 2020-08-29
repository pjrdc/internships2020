import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition
import glob

# Load data from pickle
# First index - individual person
# Second index - picture of said person
def setDataset(dataset):
    label = []
    data = []
    for person in range(round(len(dataset))):
        for number_image in range(len(dataset[person])):
            label.append(person)
            data.append(dataset[person][number_image])
    label = np.array(label, dtype=np.int32)
    data = np.matrix(data, dtype=np.float32)
    return label, data

def TrainSVM(labels,trainingData):
    # Train the SVM
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(0.1)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e5), 1e-6))
    start = datetime.now()
    print(start)
    svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    end = datetime.now()
    print(end)
    runtime = end - start
    print(runtime)
    #svm.save(filename)
    return svm

def PredictPhoto(imagePath,svm):
    image = cv.imread(imagePath)
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    # compute the facial embedding for the face
    encoding = face_recognition.face_encodings(rgb, boxes)
    encoding = np.array(encoding, dtype=np.float32)
    try:
        print(imagePath, svm.predict(encoding)[1])
    except:
        print("Nie udal się encoding")

def PredictDataset(svm,trainingData,labels):
    for i in range(len(trainingData)):
        array = np.matrix(trainingData[i], dtype=np.float32)
        prediction = svm.predict(array)[1]
        print("Predicted: ", prediction, "Correct: ", labels[i])

# Load encodings and create trainingData with Labels
test = pd.read_pickle("embeddings_by_id_lfw.pickle")
labels, trainingData = setDataset(test)
trainingData = np.array(trainingData, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
labels = labels.reshape(-1,1)
svm = TrainSVM(labels, trainingData)
PredictDataset(svm,trainingData,labels)
#for file in glob.glob("LFW/*/*.jpg"):
    #PredictPhoto(file,svm)


