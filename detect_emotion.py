import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from deepface import DeepFace
import webbrowser

class EmotionProcessor:
    def recv(self, frm):
        
        frame = frm.to_ndarray(format = "bgr24")
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(to_gray, 1.1, 4)

        for x,y,w,h in faces:
            
            try:
                detected_emotion = DeepFace.analyze(frame, actions=['emotion'])
            
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                
                cv2.rectangle(frame, (200, 400), (500, 500), (0, 0, 0), -1)
                cv2.putText(frame, detected_emotion[0]['dominant_emotion'], (230, 460), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 2, cv2.LINE_4)

                print(detected_emotion[0]['dominant_emotion'])
                np.save("emotion.npy", np.array([detected_emotion[0]['dominant_emotion']]))

            except:
                print('no face')
        
        return av.VideoFrame.from_ndarray(frame, format = "bgr24")

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""
    
webrtc_streamer(key = "key", desired_playing_state = True, video_processor_factory = EmotionProcessor)

btn = st.button("Detect my emotions")

if btn:
    if not(emotion):
        st.warning("Please show your face correctly")
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={emotion}+songs")     
