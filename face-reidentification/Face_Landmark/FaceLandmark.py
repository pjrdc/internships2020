# import the necessary packages
import cv2
import json
import dlib  # Load the detector
import argparse
import os
from imutils import paths
import math
from itertools import combinations

'''
Function calculates combinations of points of interest choosen by us
[in]: none
[out]: list of combination
'''
def Combinations():
    points_of_interest = [17, 21, 22, 26, 36, 39, 42, 45, 27, 33, 31, 35, 48, 54]
    list_of_combinations =  combinations(points_of_interest,2)
    possibility = list(list_of_combinations)
    return possibility
'''
Function calculates distance between two points using Pythagorean theorem
[in]: X and Y of two points
[out]: Calculated distanced
'''
def Distance(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)
'''
Function returns list of proportions
[in]: Landmarks made by function CreateLandmarks
[out]: Proportions between Face Landmarks
'''
def ReturnProportions(landmarks):
    combination_list = Combinations()
    proportions = []
    ref_distance = Distance(landmarks.part(39).x, landmarks.part(42).x,
                           landmarks.part(39).y, landmarks.part(42).y)
    for pair in combination_list:
        dist = Distance(landmarks.part(pair[1]).x, landmarks.part(pair[0]).x,
                 landmarks.part(pair[1]).y, landmarks.part(pair[0]).y) / ref_distance
        proportions.append(dist)
    return proportions
'''
Function prints proportion of face landmarks with description 
[in]: Landmarks made by function CreateLandmarks
[out]: none
'''
def PrintProportions(landmarks):
    ### Ref_distance to długość między kącikami oczów bliżej nosa
    print("Proporcje")
    ref_distance = Distance(landmarks.part(39).x,landmarks.part(42).x,
                            landmarks.part(39).y,landmarks.part(42).y)
    print("Odległość referencyjna - długość między kącikami oczów bliżej nosa")
    print("Usta", Distance(landmarks.part(54).x, landmarks.part(48).x,
                           landmarks.part(54).y, landmarks.part(48).y)/ref_distance)

    print("Prawa brew", Distance(landmarks.part(21).x, landmarks.part(17).x ,
                                 landmarks.part(21).y, landmarks.part(17).y) / ref_distance)

    print("Lewe oko", Distance(landmarks.part(45).x, landmarks.part(42).x,
                               landmarks.part(45).y, landmarks.part(42).y) / ref_distance)

    print("Prawe oko", Distance(landmarks.part(39).x, landmarks.part(36).x ,
                                landmarks.part(39).y, landmarks.part(36).y) / ref_distance)

    print("Lewa brew", Distance(landmarks.part(26).x, landmarks.part(22).x,
                                landmarks.part(26).y, landmarks.part(22).y)  / ref_distance)

    print("Dlugosc nosa" ,Distance(landmarks.part(27).x, landmarks.part(33).x,
                                   landmarks.part(33).y, landmarks.part(27).y)/ref_distance)

    print("Szerokosc nosa", Distance(landmarks.part(35).x, landmarks.part(31).x,
                                     landmarks.part(35).y, landmarks.part(31).y)/ref_distance)

    print("Brew to brew zew:", Distance(landmarks.part(17).x, landmarks.part(26).x,
                                        landmarks.part(17).y, landmarks.part(26).y)/ref_distance)

    print("Oko to oko zew :", Distance(landmarks.part(36).x, landmarks.part(45).x,
                                       landmarks.part(36).y, landmarks.part(45).y)/ref_distance)

    print("Brew to brew wew:", Distance(landmarks.part(21).x, landmarks.part(22).x,
                                        landmarks.part(21).y, landmarks.part(22).y)/ref_distance)
'''
Function prints coordinates of face landmarks with description 
[in]: Landmarks made by function CreateLandmarks
[out]: none
'''
def PrintCoordinate(landmarks):
    print("Element, X, Y.")
    print("Prawa brew kraniec: ",landmarks.part(17).x, landmarks.part(17).y)
    print("Prawa brew nos: ",landmarks.part(21).x, landmarks.part(21).y)
    print("Lewa brew nos: ",landmarks.part(22).x, landmarks.part(22).y)
    print("Lewa brew kraniec: ",landmarks.part(26).x, landmarks.part(26).y)
    print("Prawe oko kraniec: ",landmarks.part(36).x, landmarks.part(36).y)
    print("Prawe oko nos: ",landmarks.part(39).x, landmarks.part(39).y)
    print("Lewe oko nos: ",landmarks.part(42).x, landmarks.part(42).y)
    print("Lewe oko kraniec: ",landmarks.part(45).x, landmarks.part(45).y)
    print("Gora nos: ",landmarks.part(27).x, landmarks.part(27).y)
    print("Dol nos: ",landmarks.part(33).x, landmarks.part(33).y)
    print("Prawy koniec nosa: ",landmarks.part(31).x, landmarks.part(31).y)
    print("Lewy koniec nosa: ",landmarks.part(35).x, landmarks.part(35).y)
    print("Prawy kacik ust: ",landmarks.part(48).x, landmarks.part(48).y)
    print("Lewy kacik ust: ",landmarks.part(54).x, landmarks.part(54).y)

'''
Function detect faces from dataset, calculates position of face landmark and save it to .json
file as well as return landmarks
[in]: path of dataset, path of savefile name
[out]: .json file of landmarks with person id(name of directory inside dataset)
'''
def CreateLandmarks(dataset,savefile):
    detector = dlib.get_frontal_face_detector()  # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # read the image
    points_of_interest = [17, 21, 22, 26, 36, 39, 42, 45, 27, 33, 31, 35, 48, 54]

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", default=dataset,
                    help="path to input directory of faces + images")
    args = vars(ap.parse_args())

    print("[INFO] loading faces...")
    imagePaths = list(paths.list_images(args["dataset"]))
    vectors = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] calculating image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        img = cv2.imread(imagePath)  # Convert image into grayscale
        try:
            gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)  # Use detector to find landmarks
        except:
            continue
        faces = detector(gray)
        for face in faces:
            data = 0
            landmarks = predictor(image=gray, box=face)  # Loop through all the points
            # for n in range(0, 68):
            data = ReturnProportions(landmarks)
            vectors.append(data)
            knownNames.append(name)
    cv2.waitKey(delay=0)  # Close all windows
    cv2.destroyAllWindows()
    data = {"vector": vectors, "names": knownNames}
    with open(savefile, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
'''
Function creates landmark for one image 
View image of person with landmarks on it 
[in]: path of image
[out]:
'''
def Landmark(image_path):
    detector = dlib.get_frontal_face_detector()  # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # read the image
    img = cv2.imread(image_path)
    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(image=gray, box=face)  # Loop through all the points
        for n in [17, 21, 22, 26, 36, 39, 42, 45, 27, 33, 31, 35, 48, 54]:
            x = landmarks.part(n).x
            y = landmarks.part(n).y  # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)  # show the image
    cv2.imshow(winname="Face", mat=img)  # Delay between every fram
    cv2.waitKey(delay=0)  # Close all windows
    cv2.destroyAllWindows()
    return landmarks