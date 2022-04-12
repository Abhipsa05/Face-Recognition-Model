#testing the model using yml file
import cv2 as cv

model = cv.face.LBPHFaceRecognizer_create()
model.read('face_trained.yml')
people=['Ben']
haar_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread(r'C:\Users\DELL\Desktop\python\Face Recognizer\test\1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

for (x,y,w,h) in faces_rect:
    faces_ = gray[y:y+h,x:x+w]

    label, confidence = model.predict(faces_)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)