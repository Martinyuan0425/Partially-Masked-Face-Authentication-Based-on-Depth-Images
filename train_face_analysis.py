from model.architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer, LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf

def read_image(image_path:str):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def initialize_model():
    face_encoder = InceptionResNetV2()
    path = "model/facenet_keras_weights.h5"
    
    # face_encoder = MobileNet()
    # path = r"C:\Users\606\Desktop\facenet-tf2-main\facenet-tf2-main\model_data/facenet_mobilenet.h5"
    
    face_encoder.load_weights(path)
    return face_encoder

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def encode_face(face, model, required_shape=(160, 160)):
    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    return model.predict(face_d)[0]

def normalize_encode(encode):
    l2_normalizer = Normalizer('l2')
    return l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]


def process_dataset():

    encoding_dict = {}
    model = initialize_model()
    dataset_path = r"C:\Users\606\Desktop\lab606\dataset\2_label"

    for face_names in os.listdir(dataset_path):
        
        if not os.path.isdir(os.path.join(dataset_path, face_names)):
            continue
        
        encodes = []
        person_dir = os.path.join(dataset_path, face_names)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir,image_name)
            img = read_image(image_path)
            
            encode = encode_face(img, model)
            encodes.append(encode)

        if len(encodes) > 0:
            encode = np.sum(encodes, axis=0 )
            encode = normalize_encode(encode)
            encoding_dict[face_names] = encode

    return encoding_dict

def train_classifier(encodings):
    num_faces = len(encodings)
    model = tf.keras.Sequential([
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(num_faces, activation="softmax"),
    ])
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer="adam",
        metrics=["accuracy"]
    )

    x_train = np.array([value for _, value in encodings.items()])
    y_train = np.array(list(encodings.keys()))
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    model.fit(x_train, y_train, epochs=10, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor = 'loss')])
    model.save("classifier")

    # model.summary()
    with open("labelmap.txt", "w") as file:
        for class_ in le.classes_:
            file.write(class_ + "\n")

def train():
    print("Processings Dataset")
    encodings = process_dataset()
    print("Training classifier")
    train_classifier(encodings)

def test(face, face_model, classifier):
    # path = r"C:\Users\606\Desktop\2_label_data\normal\0.jpg"
    # face = read_image(path)
    # face_model = initialize_model()
    # classifier = tf.keras.models.load_model("classifier/0707/")
    
    encoded_face = encode_face(face, face_model)
    encoded_face = np.expand_dims(encoded_face, axis=0)
    input_data = tf.constant(encoded_face)
        
    result = classifier(input_data).numpy()
    # print(result)
    
    return result.argmax()
if __name__ == "__main__":
    train()