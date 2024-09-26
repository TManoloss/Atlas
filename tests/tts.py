import pyttsx3
engine = pyttsx3.init()


voices = engine.getProperty('voices')  
engine.setProperty('voice', voices[0].id)

engine.say("Olá sou Atlas e estou aqui para ajudar você. Como posso ajudar?")
engine.runAndWait()