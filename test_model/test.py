import tensorflow as tf
import librosa
import numpy as np
import argparse

help_message = "Use the -f argument to specify file to be tested. Currently tests only one file at a time."
parser = argparse.ArgumentParser(help=help_message)

parser.add_argument("-f", "--filename", help="Mention filename. Example: -f happy.wav")
parser.add_argument("-m", "--model_path", help="Mention directory of saved model. Example: -m saved_model/205_epochs_ravdees")
args = parser.parse_args()

if args.filename:
    save_file_path = args.filename
else:
    save_file_path = "happy.wav"

if args.model_path:
    save_model_path = args.model_path
else:
    save_model_path = "saved_model/205_epochs_ravdees"

def process_audio_clip(file_path):
    file_path = file_path
    audio, sr = librosa.load(file_path)
    mfcc = np.mean(librosa.feature.mfcc(audio, sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(audio, sr).T, axis=0)
    chromagram = np.mean(librosa.feature.chroma_stft(audio, sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(audio, sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(audio, sr).T, axis=0)
    extracted_features = tf.concat([mfcc, mel, chromagram, spectral, tonnetz], axis=0)
    return extracted_features

EMOTION_DICT_RAVDEES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

label_dict = EMOTION_DICT_RAVDEES

label_to_int = dict({(key, i) for i, key in enumerate(label_dict.keys())})

int_to_label = dict({(key, val) for val, key in label_to_int.items()})

Model = tf.keras.models.load_model(save_model_path)

features = process_audio_clip(save_file_path)

features = tf.expand_dims(features, -1)
features = tf.expand_dims(features, 0)

prediction = Model.predict(features)
label = tf.math.argmax(prediction[0])

emotion = label_dict[int_to_label[label.numpy()]]
print(f"Predicted Emotoin is: {emotion}")
