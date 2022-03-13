# PyTalk

# Import des bibliothèques nécésaires
import sys
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow import keras
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

"""
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
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

model.save('action.h5')
"""

""" Utiliser pour verifier modèle
# Verification du modèle
yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
print(accuracy_score(ytrue, yhat))
"""
try:
    new_model = keras.models.load_model('action.h5')
except:
    print("ERROR aucun modèle")
    sys.exit(1)

# * Si rien ne s'ouvre, modifiez la valeur 0 en 1,2,..(si vous avez des webcam virtuelles par exemples)
cap = cv2.VideoCapture(0)

sequence = []
sentence = []


# Actions que l'on veut détecter pris de "https://youtu.be/defJsB_CJmo"
actions = np.array(["bonjour", "bonsoir", "ca_va"])

# 60 videos pour s'entrainer
no_sequences = 60

# Les videos seront 20 frames
sequence_length = 20


# * Modifier pour +/- sensibilité
threshold = 0.7

colors = [(245,117,16), (117,245,16), (16,117,245)]

# Fonction pour graphique bcp de stackoverflowww / github
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Lire capture
        ret, frame = cap.read()

        # Faire détections
        image, results = mediapipe_detection(frame, holistic)
        
        # Dessiner les petits traits de connection
        draw_styled_landmarks(image, results)

        # Prediction 
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-20:]
        
        if len(sequence) == 20:
            res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        # Visualitation du résultat
            # Ne montre à l'écran QUE si résultat précedent différent de celui présent
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Petit graph
            image = prob_viz(res, actions, image, colors)
        
        # Rendering
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Appuie sur Q pour quitter", (3,450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Montre sur l'écran
        cv2.imshow('PyTalk', image)

        # Quitte et détruit fenetre si appuie sur Q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()