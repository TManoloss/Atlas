# Our Main file.
 
import speech_recognition as sr

# Cria um reconhecedor de voz
r = sr.Recognizer()

# Abre o microfone para captura
with sr.Microphone() as source:
    while True: 
     audio = r.listen(source) # Escuta o microfone
 
     print(r.recognize_google(audio, language= 'pt')) # Reconhece a fala
    