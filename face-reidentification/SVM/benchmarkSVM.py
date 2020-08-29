import cv2 as cv
import numpy as np
import random
'''
This is benchmark to test SVM by OpenCV
'''
NTRAINING_SAMPLES = 10 # Number of training samples per class
NCLASS_SAMPLES = 50 #Number of classes
DIMENSIONS = 128

#--------------------------0. Set up of data for clases ---------------------------------------
trainData = np.empty((NCLASS_SAMPLES*NTRAINING_SAMPLES, DIMENSIONS), dtype=np.float32)
labels = np.empty((NCLASS_SAMPLES*NTRAINING_SAMPLES, 1), dtype=np.int32)

#----------------------1.  Set up the labels and training data for the classes ----------------
for i in range(NCLASS_SAMPLES):
    labels[i*NTRAINING_SAMPLES:i*NTRAINING_SAMPLES+NTRAINING_SAMPLES] = i+1
for j in range(NCLASS_SAMPLES):
    for k in range(NTRAINING_SAMPLES):
        array = np.full((1,DIMENSIONS),random.uniform(j,j+0.5))
        trainData[j*NTRAINING_SAMPLES+k] = array

#------------------------ 2. Set up the support vector machines parameters --------------------
print('Starting training process')
## [init]
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(0.1)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e5), 1e-6))
## [init]
#------------------------ 3. Train the svm ----------------------------------------------------
## [train]
svm.train(trainData, cv.ml.ROW_SAMPLE, labels)
## [train]
print('Finished training process')

for i in range(NCLASS_SAMPLES):
    array = np.full((1,DIMENSIONS), i+0.25)
    vector = np.matrix(array, dtype=np.float32)
    response = svm.predict(vector)[1]
    print(response)
