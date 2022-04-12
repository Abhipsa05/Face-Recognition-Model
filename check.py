import cv2 as cv
import os 

haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people=['Ben']
dir = r'C:\Users\DELL\Desktop\python\Face Recognizer\Photos'
features=[]
labels=[]
for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            print(img_path)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            cv.imshow('imGE',gray)
            # face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

            # for (x,y,w,h) in face_rect:
            #     face_=gray[x:x+w,y:y+h]
            #     features.append(face_)
            #     labels.append(label)
            #     # if cv.empty(face_):
            #     #     print("Fault image")
            #     #     print(path)
            #     cv.imshow('image',face_)
            cv.waitKey(0)