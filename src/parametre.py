from clint.textui import colored
from tkinter import *
from tkvideo import tkvideo
import webbrowser

error = colored.red('[ERROR]')
warning = colored.yellow('[WARNING]')
info = colored.blue('[INFO]')
sucess = colored.green('[SUCESS]')

c1 = "#e9f9f7"
c2 = "#bdccd7"
c3 = "#6e7c90"
c4 = "#404558"
c5 = "#2b2c35"


root = Tk()
root.title("PyTalk")
root.geometry("1080x720")
root.wm_minsize(1050, 680)
root.config(background=c2)
try:
    root.iconbitmap("img/logo.ico")
except:
    print(f"{warning} NO ICON | Essayez d'éxécuter : 'cd src/'")
 
def nextPage():
    root.destroy()
    import main


propos = Label(root, text="A propos de nous :", font=("Courrier", 24), bg=c2, fg=c4)
propos.pack(side=TOP)

description = Label(root, text="Ce programme a été développé dans le cadre de 'Trophée NSI' dans le but de facilité la communication et l'inclusion de personne malentendante.\n Et de ce fait, lutter contre la discrimination.\nL'idée nous ai venus suite à une balade dans la rue où nous avons rencontré une personne pratiquant la langue des signes.\nOr, nous avons eu du mal à communiquer et l'idée d'un logiciel pouvant facilité la conversation semblait évidente !\nDe ce fait, on voulait faire de ce projet un projet qui aide la société\nNous sommes 2 à avoir travaillé activement sur ce projet, mais nous remercions grandement les ensignants, les amis.es...\nqui nous ont grandement aidé à améliorer des aspects du programme.\nNous avons décidé de mettre ce logiciel en open source sur github (voir ci-dessous) pour que le plus grand nombre puissent en profiter/l'améliorer.", font=("Courrier", 12), bg=c2, fg=c4)
description.pack(side=TOP)

amelioration = Label(root, text="Amélioration :", font=("Courrier", 24), bg=c2, fg=c4)
amelioration.pack(side=TOP)

pistes = Label(root, text="Ce programme nous a pris plus de 75h de travail. Ayant une date d'envoie du projet assez rapproché et étant dans une période de Bac,\nnous ne pouvions pas nous permettre d'inclure toutes les modifications envisagées.\nNéanmoins, nous chercherons à améliorer et corriger certains aspects de PyTalk via github.\nVoici certaines pistes d'amélioration :\nLa traduction de langue parlé à langyage des signes | Une fois la traduction effectuer, dicter à l'oral le résultat\n Facilité l'entrainement de nouveaux modèles | Choisir les différents mots à traduire\nFAIRE UNE VERSION WEB (prochainement...) | Corriger certains bug liés à tkinter (oups)", font=("Courrier", 12), bg=c2, fg=c4)
pistes.pack(side=TOP)


menu = Button(root, text="Retour au Menu", font=("Courrier", 24), command=nextPage)
menu.pack(fill=X, side=BOTTOM)

redirect = Frame(root, bg=c2)

github = Button(redirect, text="Projet", font=("Courrier", 18), bg=c4, fg=c2, activebackground=c2, activeforeground=c4, command=lambda: webbrowser.open("https://github.com/"))
github.pack(pady=25, side=LEFT, padx=5)

youtube = Button(redirect, text="Video", font=("Courrier", 18), bg=c4, fg=c2, activebackground=c2, activeforeground=c4, command=lambda: webbrowser.open("https://youtube.com/"))
youtube.pack(pady=25, side=RIGHT, padx=5)

redirect.pack(side=BOTTOM)

root.mainloop()