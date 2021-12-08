import tensorflow as tf
import librosa
import os
import numpy as np

EMOTION_DICT = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "Neutral",
}


def get_dataset(
    training_dir="./train_data",
    label_dict=EMOTION_DICT,
    validation_dir=None,
    val_split=0.2,
    batch_size=128,
    random_state=42,
    cache=False,
):
    """
    Creates a `tf.data.Dataset` object.
    Arguments:
    training_dir : String
    label_dict : dictionary with labels
    """

    def decompose_label(file_path: str):
        return label_to_int[file_path[-6:-5]]

    def process_audio_clip(file_path, label):
        file_path = file_path.numpy()
        audio, sr = librosa.load(file_path)
        mfcc = np.mean(librosa.feature.mfcc(audio, sr, n_mfcc=40).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(audio, sr).T, axis=0)
        chromagram = np.mean(librosa.feature.chroma_stft(audio, sr).T, axis=0)
        spectral = np.mean(librosa.feature.spectral_contrast(audio, sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(audio, sr).T, axis=0)
        extracted_features = tf.concat(
            [mfcc, mel, chromagram, spectral, tonnetz], axis=0
        )
        return extracted_features, label

    def tf_wrapper_process_audio_clip(file_path, label):
        extracted_features, label = tf.py_function(
            process_audio_clip, [file_path, label], [tf.float32, tf.int32]
        )
        extracted_features = tf.expand_dims(extracted_features, -1)
        return extracted_features, label

    file_path_list = os.listdir(training_dir)
    label_to_int = dict({(key, i) for i, key in enumerate(label_dict.keys())})
    labels = [decompose_label(file_path) for file_path in file_path_list]

    # Split into train and val sets
    if validation_dir is None:
        if val_split > 0:
            from sklearn.model_selection import train_test_split

            file_path_list = [
                os.path.join(training_dir, path) for path in file_path_list
            ]
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                file_path_list, labels, test_size=val_split, random_state=random_state
            )
    else:
        train_paths = file_path_list
        train_labels = file_path_list
        val_paths = os.listdir(validation_dir)
        val_labels = [decompose_label(file_path) for file_path in val_paths]
        val_paths = [os.path.join(training_dir, path) for path in val_paths]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    train_ds = train_ds.map(
        tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        tf_wrapper_process_audio_clip, num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def create_model(model_type="emoDB", num_output_classes=7):
    from tensorflow.keras import Model
    from tensorflow.keras import layers as L

    if model_type.lower() == "emodb":
        input_layer = L.Input(shape=(193, 1))

        cnn1 = L.Conv1D(256, (5))(input_layer)
        batch_norm1 = L.BatchNormalization()(cnn1)
        relu1 = L.ReLU()(batch_norm1)

        cnn2 = L.Conv1D(128, (5))(relu1)
        relu2 = L.ReLU()(cnn2)
        dropout1 = L.Dropout(0.1)(relu2)
        batch_norm2 = L.BatchNormalization()(dropout1)

        max_pool1 = L.MaxPool1D(8)(batch_norm2)

        conv3 = L.Conv1D(128, (5))(max_pool1)
        relu3 = L.ReLU()(conv3)
        conv4 = L.Conv1D(128, (5))(relu3)
        relu4 = L.ReLU()(conv4)
        conv5 = L.Conv1D(128, (5))(relu4)
        batch_norm4 = L.BatchNormalization()(conv5)
        relu5 = L.ReLU()(batch_norm4)
        dropout2 = L.Dropout(0.2)(relu5)

        conv6 = L.Conv1D(128, (5))(dropout2)
        flatten = L.Flatten()(conv6)
        dropout3 = L.Dropout(0.2)(flatten)

        output_layer = L.Dense(num_output_classes)(dropout3)
        batch_norm5 = L.BatchNormalization()(output_layer)
        softmax = L.Softmax()(batch_norm5)
        model = Model(inputs=[input_layer], outputs=[softmax])
        optimizer = tf.keras.optimizers.RMSprop(1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss)

    else:
        print("Model_type unknown. Please use one of \[emoDB, Ravdees, IEMOCAP\]")
        return None

    return model
