import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import os
import imutils
import time
import pose_estimation_class as pm
import mediapipe as mp

import cv2
from tkinter import *
from PIL import ImageTk, Image
import _tkinter # with underscore, and lowercase 't'

cap=cv2.VideoCapture(0)
json_file = open('model_pose.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_pose.h5")
print("Loaded model from disk")
label=['hand_wave','squat','standing','taking_phone','walking','yoga']
fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

detector = pm.PoseDetector()
while 1:
    nl,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    contours,hierchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if contours:
        areas = []

        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        
        max_area = max(areas or [0])

        max_area_index = areas.index(max_area)

        cnt = contours[max_area_index]

        M = cv2.moments(cnt)
        
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
        
        if h < w:
            j += 1
            
        if j > 10:
            #print ("FALL")
            cv2.putText(frame, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        if h > w:
            j = 0
            cv2.putText(frame, 'normal human', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    #cv2.imshow('fall Frame', frame)
    frame, p_landmarks, p_connections = detector.findPose(frame, False)
    mp.solutions.drawing_utils.draw_landmarks(frame, p_landmarks, p_connections)
    lmList = detector.getPosition(frame)
    cv2.imshow('pose frame',frame)
    
    #frame = cv2.resize(frame, (256, 256))
    
                       
    
    k=cv2.waitKey(1)
    if k%256==27:
        
                    
        print('close')
                    
        break
    elif k%256==32:             
        
                    
        print("image saved")
        
        cv2.imwrite('input.jpg',frame)
        
        #test_image = cv2.resize(frame, (256, 256)
        test_image = image.load_img('input.jpg', target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        fresult=np.max(result)
        label2=label[result.argmax()]
        print(label2)
        


win=tk.Tk()

def b1_click():
    global path2
    try:
        json_file = open('model1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model1.h5")
        print("Loaded model from disk")
        label=['hand_wave','squat','standing','taking_phone','walking','yoga']

        
        #lbl2=tk.Label(win,image=img)
        
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #img1=('F:/py/leaf_disease_final( COMPLETE )/1.jpg')


        #lbl2=tk.Label(win,image=img1)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #loading image 
        path2=filedialog.askopenfilename()
        print(path2)
        

        #img = ImageTk.PhotoImage(Image.open(path2))
        
        #lbl2=tk.Label(win,image=img)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")

        #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        #panel = tk.Label(win, image = img)
        #panel.pack( fill = "both", expand = "yes")
        #imr=cv2.imread(path2)
        #a=cv2.imshow(imr)
        #print(imr)
        test_image = image.load_img(path2, target_size = (128, 128))        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        #print(result)
        #print(result)
        fresult=np.max(result)
        label2=label[result.argmax()]
        print(label2)
        #lb2.configure(image=img)
        #lbl2.image=img
        lbl.configure(text=label2)
         
        
        #lbl2(ent.config(state='disabled'))
        win.mainloop()
        

    except IOError:
        pass


#button

#labelframe = LabelFrame(win, text="Leaf Disease Detection using OPENCV")
#labelframe.pack(fill="both", expand="yes")
label1 = Label(win, text=" Detection using OPENCV", fg ='blue')
label1.pack()
    
b1=tk.Button(win, text= 'browse image',width=25, height=3,fg ='red', command=b1_click)
b1.pack()
lbl = Label(win, text="Result", fg ='blue')
lbl.pack()

#image =ImageTk.PhotoImage(file='a.JPG')

#img1='1.JPG'
#lb2 = Label(win,image=image)
#lb2.pack()


#lbl.grid(column=0, row=0)
win.geometry("550x250")
win.title("Leaf Disease Detection using OPENCV")
win.bind("<Return>", b1_click)
win.mainloop() 
