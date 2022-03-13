# PyTalk

# Import des bibliothèques nécésaires
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Varibles pour mediapipe (holistic : faire détéction | drawing : dessiner détéction)
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """ summary : Permet la détéction avec mediapipe """
    # Conversion de BVR <=> RVB + on sauve un peu de mémoire lol
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """ Genère les traits des landmarks stylé grâce à mediapipe (et à la magie des docs) """
    
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    """ Extrait les valeurs de "results.X.landmark" et les stores dans 1 liste numpy (bcp de copié/collé lol) + Case handling si 0 points """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])






# Chemin pour la liste numpy
DATA_PATH = os.path.join('data') 

# Actions que l'on veut détecter pris de "https://youtu.be/defJsB_CJmo"
actions = np.array(["bonjour", "bonsoir", "ca_va"])

# 60 videos pour s'entrainer
no_sequences = 60

# Les videos seront 20 frames
sequence_length = 20

# Créer les dossiers pour mettres data et no_sequences liste numpy 
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



# Dictionaire des actions
label_map = {label:num for num, label in enumerate(actions)}


# Stock les valeures des listes numpy dans des grosseeees listes "sequences" et "labels"
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Pas compris c'est du stack overflow mais ça marhce lol
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# Definis le chemin pour les logs TensorBoard
log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)


# Deep Learning détection (stack overflowwwww)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# Compilation du modèle. On peut modifier l'optimiser
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# * MODIFIER EPOCHS POUR +/- PRESICION
# Train le modèle (modifier epochs pour plus d'accuracy) ça prend du temps à faire
model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])

model.save('action.h5')

""" Utiliser pour verifier modèle
# Verification du modèle
yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
print(accuracy_score(ytrue, yhat))
"""

# * Si rien ne s'ouvre, modifiez la valeur 0 en 1,2,..(si vous avez des webcam virtuelles par exemples)
cap = cv2.VideoCapture(0)


# * Jouez avec %confidence variables pour +/- de stabilité
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop dans liste des actions
    for action in actions:
        # Loop dans les videos (sequences)
        for sequence in range(no_sequences):
            # Loop sur la longueur de la video (sequence_lenght)
            for frame_num in range(sequence_length):

                # Lire capture
                ret, frame = cap.read()
                
                # Faire détéction
                image, results = mediapipe_detection(frame, holistic)
                
                # Dessiner les petits traits de connection
                draw_styled_landmarks(image, results)
                
                
                # Mise en place d'un délai entre chaque prise (peut êtr modifer) (un peu de copié/collé...)
                if frame_num == 0: 
                    cv2.putText(image, "Initialisation de la collection", (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, "Collection des images pour {} video {}".format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Montrer à l'écran
                    cv2.imshow('PyTalk', image)
                    cv2.waitKey(600)
                else: 
                    cv2.putText(image, "Collection des images pour {} video {}".format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Montrer à l'écran
                    cv2.imshow('PyTalk', image)

                # Export des listes numpy dans le dossier approprié
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)          

                # Quitte la loop si appuie sur Q
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


# Ferme tout clean :)
cap.release()
cv2.destroyAllWindows()
"""