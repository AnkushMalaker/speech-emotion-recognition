import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model

class SpeechModel:
    def __init__(self, num_output_classes) -> None:
        self.num_output_classes = num_output_classes

    def getRAVDESS(self) -> Model:
        """Returns a tensorflow model that is according to specifications of the baseline CNN model in the paper."""
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

        output_logits = L.Dense(self.num_output_classes)(dropout3)
        batch_norm5 = L.BatchNormalization()(output_logits)
        softmax = L.Softmax()(batch_norm5)
        model = Model(inputs=[input_layer], outputs=[softmax])
        optimizer = tf.keras.optimizers.RMSprop(1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss)

        return model

    def getEmoDB(self) -> Model:
        """Returns a tensorflow model that is according to specifications of the EmoDB model A in the paper."""
        input_layer = L.Input(shape=(193, 1))

        cnn1 = L.Conv1D(256, (5))(input_layer)
        batch_norm1 = L.BatchNormalization()(cnn1)
        relu1 = L.ReLU()(batch_norm1)

        cnn2 = L.Conv1D(128, (5))(relu1)
        relu2 = L.ReLU()(cnn2)
        dropout1 = L.Dropout(0.1)(relu2)

        max_pool1 = L.MaxPool1D(8)(dropout1)

        conv3 = L.Conv1D(128, (5))(max_pool1)
        batch_norm2 = L.BatchNormalization()(conv3)
        relu3 = L.ReLU()(batch_norm2)
        dropout2 = L.Dropout(0.2)(relu3)

        flatten = L.Flatten()(dropout2)

        output_logits = L.Dense(self.num_output_classes)(flatten)
        softmax = L.Softmax()(output_logits)
        model = Model(inputs=[input_layer], outputs=[softmax])
        optimizer = tf.keras.optimizers.RMSprop(1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss)

        return model

    def getIEMOCAP(self) -> Model:
        """Returns a model that is same as EmoDB model in most senses, except dropout value before MaxPool layer is increased and model is compiled with Adam optimizer instead."""
        input_layer = L.Input(shape=(193, 1))

        cnn1 = L.Conv1D(256, (5))(input_layer)
        batch_norm1 = L.BatchNormalization()(cnn1)
        relu1 = L.ReLU()(batch_norm1)

        cnn2 = L.Conv1D(128, (5))(relu1)
        relu2 = L.ReLU()(cnn2)
        dropout1 = L.Dropout(0.2)(relu2)
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

        output_logits = L.Dense(self.num_output_classes)(dropout3)
        batch_norm5 = L.BatchNormalization()(output_logits)
        softmax = L.Softmax()(batch_norm5)
        model = Model(inputs=[input_layer], outputs=[softmax])
        optimizer = tf.keras.optimizers.Adam(1e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss)

        return model
