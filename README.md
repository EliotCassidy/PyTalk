# PyTalk
> Le code est distribué sous licence libre [GPL v3+](https://github.com/Trivador/PyTalk/blob/master/LICENSE) et le texte sous licence Creative Commons CC By-Sa, conformément aux demandes du concours.
###### PyTalk a été développé par Eliot CASSIDY et Antonin JOUVE, élèves faisant la spécialité NSI en 1ère Général.

PyTalk est un logiciel de traduction en temps réel de la langue des signes. Ce programme a été développé dans le cadre du concours [trophées NSI](https://trophees-nsi.fr/). 
La présentation technique du projet peut-être trouvé dans les fichiers du projet.


## Utilisation (windows)

### Installation

Pour installer les bibliothèques nécéssaires au fonctionnement du projet, éxécutez :
```md
pip3 install -r requirements.txt
```
###### Si une des versions spécifiés dans [requirements.txt](https://github.com/Trivador/PyTalk/blob/master/requirements.txt) est introuvable, essayez en installant la version la plus récente.

Python 3.9 a été utilisé pour ce projet. Veuillez télécharger et éxécuter ce programme en utilisant cette [version](https://www.python.org/downloads/release/python-3911/). Notez qu'il est impératif de mettre python 3.9 dans le PATH pour que le programme fonctionne correctement.

Voici une liste non exhaustif des librairies utilisées et leur utilité dans ce projet :

| Librairies                                                   | But                                                                     |
|--------------------------------------------------------------|-------------------------------------------------------------------------|
| [MediaPipe](https://mediapipe.dev/)                          | Reconnaitre des points de la main et du visage en temps réel            |
| [Numpy](https://numpy.org/)                                  | Stockage des points dans des tableaux numpy.                            |
| [OpenCV](https://opencv.org/)                                | Gestion de la webcam                                                    |
| [Sklearn](https://scikit-learn.org/stable/)                  | Machine learning basique                                                |
| [TensorFlow](https://www.tensorflow.org/?hl=fr)              | Machine learning complet permettant, entre autre, d'entrainer le modèle |
| [Tkinter](https://docs.python.org/fr/3/library/tkinter.html) | Interface graphique                                                     |

### Exécution
Avant de lancer le script n'oubliez pas d'éxécuter :
```md
cd src
```
Pour lancer le script, éxécutez :
```md
python main.py
```

### Personalisation :
Pour entrainer le modèle sur votre propre liste de mot, veuillez modifier cette [liste](https://github.com/Trivador/PyTalk/blob/83f749f4748220ace5d2cba4f03a5bc89488bd53/src/training.py#L74)
Pour modifer le nombre de frame et de vidéo par mot, veuillez modifier ces [valeurs](https://github.com/Trivador/PyTalk/blob/83f749f4748220ace5d2cba4f03a5bc89488bd53/src/training.py#L77). Notez que baisser ces valeurs diminuera la fiablité de la reconaissance. 

### Erreurs
Si il y a un problème d'instalation ou d'éxécution, verifez d'avoir :
- [Python](https://www.python.org/downloads/release/python-3911/) et de l'avoir déclaré dans le PATH
- pip : ```python -m pip install --upgrade pip```  pour mettre à jour
- Définis correctement la webcam en modifiant se [valeur](https://github.com/Trivador/PyTalk/blob/83f749f4748220ace5d2cba4f03a5bc89488bd53/src/training.py#L147)

Si le problème perciste, n'hésitez pas à nous en faire part dans l'onglet [problème](https://github.com/Trivador/PyTalk/issues). Nous essayerons d'être actif et de mettre à jour ce projet.
Notez également que ce programme nécésite une machine relativement puissante pour faire tourné le logiciel en temps réel.


## Améliorations
Nous avons pour ambition de continuer à suivre et à maintenir ce projet. Pour des soucis de temps liés à la date limite du concours, nous n'avons pas pu implémenter toutes les fonctionnalités que nous avions imaginé.
Voici une liste non exhaustive des mises à jour que nous souhaiterons faire :
- Ajouter une option graphique pour entraîner son modèle avec ses propres mots
- La retranscription d'une conversation en texte en temps réel
- Une fois que le logiciel à reconnu un mot, le dire à haute voix à l'aide d'un text-to speech
- Faire une version web de ce programme est le hoster en permanence (pas assez d'argent mais on a commencé)
- Corriger certains bug graphique pouvant survenir

Si vous avez d'autres pistes d'amélioration, veuillez nous en faire part dans l'onglet [idée](https://github.com/Trivador/PyTalk/discussions/categories/ideas)


## Remerciement
Nous tenons tout d'abords à remercier les membres ayant participé à l'organisation du Trophée NSI. Ce concours était une occasion de se surpasser et de gerer un projet dans son intégrité.
Puis, nous remercions grandement les professeurs de NSI qui nous ont aiguillé et aidé à de nombreuses reprises.
Enfin, nous avons une pensée particulière aux amis.es qui nous on apporté de précieux retours et conseils.
