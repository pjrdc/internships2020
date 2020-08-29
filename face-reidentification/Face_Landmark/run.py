from FaceLandmark import CreateLandmarks,Landmark,PrintCoordinate,PrintProportions

print("What would you like to do:")
print("1. Create landmarks for database")
print("2. Test creating landmark for one photo")
option = int(input())
if option == 1:
    print("Path of dataset:")
    path_dataset = str(input())
    print("Save file path:")
    savefile_path = str(input())
    CreateLandmarks(dataset=path_dataset,savefile=savefile_path)
elif option == 2:
    print("Image path: ")
    image_path = str(input())
    landmarks = Landmark(image_path=image_path)
    print("Whould you like to inspect data: \n 1-Yes \n 2-NO")
    foo = int(input())
    if foo==2:
        exit()
    elif foo==1:
        print("Options:")
        print("1 - Print proportions")
        print("2 - Print coordinates")
        print("3 - Print both")
        print("4 - Exit")
        opt = int(input())
        if opt == 4:
            exit()
        elif opt == 3:
            PrintCoordinate(landmarks)
            PrintProportions(landmarks)
        elif opt == 2:
            PrintCoordinate(landmarks)
        elif opt == 1:
            PrintCoordinate(landmarks)
        else:
            print("Wrong option")
else:
    print("wrong option")