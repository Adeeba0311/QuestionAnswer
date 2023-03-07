import os
import speech_recognition as sr
from tkinter.filedialog import askopenfilename

def transcribe_video(filename):
   command = "ffmpeg -i "+filename+".mp4 "+filename+".mp3"
   os.system(command)

   commandwav = "ffmpeg -i "+filename+".mp3 "+filename+".wav"
   os.system(commandwav)

   AUDIO_FILE = filename+".wav"
   r = sr.Recognizer()
   audioFile = sr.AudioFile(AUDIO_FILE)

   with audioFile as source:
      audio = r.record(source, duration=100)

   print(type(audio))
   print("------------------------------------")
   text = r.recognize_google(audio)
   #text = r.recognize_sphinx(audio)
   return(text)



if __name__ == "__main__":
   filename = askopenfilename(filetypes=[("*","*.mp4")]) # queryImage
   temp = os.path.splitext(filename)
   text = transcribe_video(temp[0])
   #print(text)
