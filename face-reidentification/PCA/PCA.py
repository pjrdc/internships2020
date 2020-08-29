import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime

def PCA_own(dataset, savefile):
    df = pd.read_json(str(dataset))
    df = df.to_numpy()
    x = []
    y = []
    for i in range(len(df)):
        x.append(df[i][0])
        y.append((df[i][1]))
    res = pd.DataFrame(y, columns=['Y'])
    x = np.array(x)
    features = x.transpose()
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    for dim in range(1, len(vectors)):
        start = datetime.now()
        print("wymiary:", dim)
        print(start)
        arr = []
        principalComponents = []
        for i in range(0, dim):
            principalComponents.append(x.dot(vectors.T[i]))
        principalDf = pd.DataFrame(data=principalComponents)
        finalDf = pd.concat([principalDf.transpose(), res['Y']], axis=1)
        result = finalDf.to_json()
        #json_object = json.dumps(finalDf, indent=4)
        with open(str(savefile) + str(dim) + '.json', "w") as outfile:
            outfile.write(result)
        end = datetime.now()
        print(end)
        runtime = end - start
        print(runtime)

def PCA_sklearn(dataset, savefile):
    df = pd.read_json(str(dataset))
    df = df.to_numpy()
    x = []
    y = []
    for i in range(len(df)):
        x.append(df[i][0])
        y.append((df[i][1]))
    dimensions = len(x[0])
    res = pd.DataFrame(y, columns=['Y'])
    for dim in range(1, dimensions):
        print("wymiary:", dim)
        arr = []
        data = []
        pca = PCA(n_components=dim)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents)
        finalDf = pd.concat([principalDf, res['Y']], axis=1)
        #json_object = json.dumps(finalDf, indent=4)
        result = finalDf.to_json()
        with open(str(savefile)+str(dim)+'.json', "w") as outfile:
            outfile.write(result)


def PCA_own_centroid(x,y,savefile):
    x = np.array(x)
    y = np.array(y)
    res = pd.DataFrame(y, columns=['Y'])
    features = x.transpose()
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    for dim in range(1, len(vectors)):
        start = datetime.now()
        print("wymiary:", dim)
        print(start)
        arr = []
        principalComponents = []
        for i in range(0, dim):
            principalComponents.append(x.dot(vectors.T[i]))
        principalDf = pd.DataFrame(data=principalComponents)
        finalDf = pd.concat([principalDf.transpose(), res['Y']], axis=1)
        result = finalDf.to_json()
        with open(str(savefile) + str(dim) + '.json', "w") as outfile:
            outfile.write(result)
        end = datetime.now()
        print(end)
        runtime = end - start
        print(runtime)

def PCA_sklearn_centroid(x,y,savefile):
    x = np.array(x)
    y = np.array(y)
    res = pd.DataFrame(y, columns=['Y'])
    dimensions = len(x[0])
    for dim in range(1, dimensions):
        print("wymiary:", dim)
        arr = []
        data = []
        pca = PCA(n_components=dim)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents)
        finalDf = pd.concat([principalDf, res['Y']], axis=1)
        result = finalDf.to_json()
        with open(str(savefile)+str(dim)+'.json', "w") as outfile:
            outfile.write(result)