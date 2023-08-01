import cv2
import numpy as np
import tensorflow as tf

cam=cv2.VideoCapture(0)

model=tf.keras.models.load_model('keras_model.h5')

while True:
    ret,frame=cam.read()
    img=cv2.resize(frame,(224,224))
    img=np.array(img,dtype=np.float32)
    img=np.expand_dims(img,axis=0)
    normalisedImg=img/255.0
    prediction=model.predict(normalisedImg)
    print("Prediction:",prediction)
    frame=cv2.flip(frame,1)

    cv2.imshow("cam",frame)
    if cv2.waitKey(2)==32:
        break

cv2.destroyAllWindows()
cam.release()