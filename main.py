import cv2
import numpy as np
import face_recognition
imgel=face_recognition.load_image_file('image/elon.jpg')
imgel=cv2.cvtColor(imgel,cv2.COLOR_BGR2RGB)
imgt=face_recognition.load_image_file('image/elontest.jpg')
imgt=cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgel)[0]
encode=face_recognition.face_encodings(imgel)[0]
cv2.rectangle(imgel,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facetest=face_recognition.face_locations(imgt)[0]
encodetest=face_recognition.face_encodings(imgt)[0]
cv2.rectangle(imgt,(facetest[3],facetest[0]),(facetest[1],facetest[2]),(255,0,255),2)

result=face_recognition.compare_faces([encode],encodetest)
facedis=face_recognition.face_distance([encode],encodetest)
print(result,facedis)
cv2.putText(imgt,f'{result}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)




cv2.imshow('elon',imgel)
cv2.imshow('elontest',imgt)
cv2.waitKey(0)
