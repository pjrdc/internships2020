from Distances import Centroid, Distance_to_centroid, Distance_centroids
from PCA import  PCA_own, PCA_sklearn, PCA_own_centroid, PCA_sklearn_centroid
from functions import Analysis,Informations

print("PCA options:")
print("1. With Scikat-learn")
print("2. With eigenvectors")
option_1 = int(input())
print("Dataset path:")
dataset_path = str(input())
x,y = Centroid(dataset_path)
if option_1 == 1:
    print('Savefile of PCA on photos(no need to add extension):')
    save_photo = str(input())
    print('Savefile of PCA on centroids(no need to add extension):')
    save_centroid = str(input())
    PCA_sklearn(dataset=dataset_path,savefile=save_photo)
    PCA_sklearn_centroid(x,y,save_centroid)
    print("Want to analize results? \n y or n:")
    option_2 = str(input())
    if option_2 == 'y':
        print("Save path for distance between centroids (no need to add extension): ")
        save_distance_centroid = str(input())
        print("Save path for distance to centroid (no need to add extension): ")
        save_distance = str(input())
        Distance_centroids(x,y,save_distance_centroid)
        Distance_to_centroid(save_photo,x,y,save_distance)
        print("Path to save information to centroid:")
        info_to = str(input())
        print("Path to save information between centoids:")
        info_centroid = str(input())
        Informations(save_distance_centroid,info_centroid)
        Informations(save_distance,info_to)
        print("Directory path to save plots: ")
        saveplot_path = str(input())
        Analysis(len(x[0]),saveplot_path,info_to,info_centroid)
    elif option_2 == 'n':
        exit()
    else:
        print("Wrong option")

if option_1 == 2:
    print('Savefile of PCA on photos(no need to add extension):')
    save_photo = str(input())
    print('Savefile of PCA on centroids(no need to add extension):')
    save_centroid = str(input())
    PCA_own(dataset=dataset_path, savefile=save_photo)
    PCA_own_centroid(x, y, save_centroid)
    print("Want to analize results? \n y or n:")
    option_2 = str(input())
    if option_2 == 'y':
        print("Save path for distance between centroids (no need to add extension): ")
        save_distance_centroid = str(input())
        print("Save path for distance to centroid (no need to add extension): ")
        save_distance = str(input())
        Distance_centroids(x,y,save_distance_centroid)
        Distance_to_centroid(save_photo,x,y,save_distance)
        print("Path to save information to centroid:")
        info_to = str(input())
        print("Path to save information between centoids:")
        info_centroid = str(input())
        Informations(save_distance_centroid,info_centroid)
        Informations(save_distance,info_to)
        print("Directory path to save plots: ")
        saveplot_path = str(input())
        Analysis(len(x[0]),saveplot_path,info_to,info_centroid)
    elif option_2 == 'n':
        exit()
    else:
        print("Wrong option")1
else:
    print("Wrong option")
