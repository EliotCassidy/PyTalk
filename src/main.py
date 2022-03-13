import sys
from clint.textui import colored
from tkinter import *
from tkvideo import tkvideo
import os

error = colored.red('[ERROR]')
warning = colored.yellow('[WARNING]')
info = colored.blue('[INFO]')
sucess = colored.green('[SUCESS]')

c1 = "#e9f9f7"
c2 = "#bdccd7"
c3 = "#6e7c90"
c4 = "#404558"
c5 = "#2b2c35"

def redirect_recognition():
    os.system("python recognition.py ")

def redirect_training():
    os.system("python recognition.py")

def next_page():
    root.destroy()
    import parametre

root = Tk()


root.title("PyTalk")
root.geometry("1280x960")
root.wm_minsize(1080, 720)
root.config(background=c2)
try:
    root.iconbitmap("img/logo.ico")
except:
    print(f"{warning} NO ICON | Essayez d'éxécuter : 'cd src/'")



frame = Frame(root, bg=c2)

try:
    width, height = 431, 129
    image = PhotoImage(file="img/pytalk.png")
    canvas = Canvas(root, width=width, height=height, bg=c2, bd=0, highlightthickness=0)
    canvas.create_image(width/2, 129/2, image=image)
    canvas.pack(side=TOP)
except:
    print(f"{warning} NO IMAGE | Essayez d'éxécuter depuis 'src'")

title2 = Label(frame, text="Reconnaissance de la Langue des Signes en temps réel", font=("Courrier", 24), bg=c2, fg=c4)
title2.pack()


credit = Label(root, text="Distribué sous la license open soucre par Eliot CASSIDY et Antonin JOUVE ;)", font=("Courrier", 12), bg=c2, fg=c4)
credit.pack(side=BOTTOM)

parametre = Button(root, text="Paramètre", font=("Courrier", 18), bg=c4, fg=c2, activebackground=c2, activeforeground=c4, command=next_page)
parametre.pack(pady=25, side=BOTTOM)

try:
    recognizer = Button(frame, text="Reconnaitre", font=("Courrier", 24), bg=c4, fg=c2, activebackground=c2, activeforeground=c4, command=redirect_recognition)
    recognizer.pack(pady=25, fill=X, side=RIGHT)
except:
    print(f"{error} NO PROGRAM 'recognition.py' | Essayez d'éxécuter depuis le dossier 'src'")
    sys.exit(1)


try:
    trainer = Button(frame, text="Entrainer", font=("Courrier", 24), bg=c4, fg=c2, activebackground=c2, activeforeground=c4, command=redirect_training)
    trainer.pack(pady=25, fill=X, side=LEFT)
except:
    print(f"{error} NO PROGRAM 'training.py' | Essayez d'éxécuter depuis le dossier 'src'")
    sys.exit(1)


frame.pack(expand=YES)




try:
    vid_label = Label(root)
    vid_label.pack()
    video = tkvideo("video/video.mp4", vid_label, loop = 1, size = (640,480))
    video.play()
except:
    print(f"{warning} NO VIDEO | Essayez d'éxécuter : 'cd src/'")



root.mainloop()