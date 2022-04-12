import os
import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people=['Ben','pam']
dir = r'C:\Users\DELL\Desktop\python\Face Recognizer\Photos'
features=[]
labels=[]
def trainset():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

            for (x,y,w,h) in face_rect:
                face_=gray[x:x+w,y:y+h]
                features.append(face_)
                labels.append(label)

trainset()

features=np.array(features, dtype=object)
labels=np.array(labels)

model = cv.face.LBPHFaceRecognizer_create()
model.train(features,labels)

model.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

#testing the model

img = cv.imread(r'C:\Users\DELL\Desktop\python\Face Recognizer\test\pam.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

for (x,y,w,h) in faces_rect:
    faces_ = gray[y:y+h,x:x+w]

    label, confidence = model.predict(faces_)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    if(confidence<100):
        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.putText(img, str(confidence), (20,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
