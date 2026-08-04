import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import fashion_mnist
from keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SELF_MODEL_PATH = "cloth_classifier_self.keras"
VGG_MODEL_PATH = "model_cloth_classifier_vgg16.keras"

SELF_HISTORY_PATH = "history_selfmade.npz"
VGG_HISTORY_PATH = "history_vgg16.npz"

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def train_self_model():

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    model.save(SELF_MODEL_PATH)

    np.savez(
        SELF_HISTORY_PATH,
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        accuracy=history.history["accuracy"],
        val_accuracy=history.history["val_accuracy"]
    )

    return model

def train_vgg16_model():

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    train_images = tf.image.resize(
        X_train[..., np.newaxis],
        (32, 32)
    )

    test_images = tf.image.resize(
        X_test[..., np.newaxis],
        (32, 32)
    )

    train_images = tf.image.grayscale_to_rgb(train_images).numpy().astype("float32")
    test_images = tf.image.grayscale_to_rgb(test_images).numpy().astype("float32")

    train_images /= 255.0
    test_images /= 255.0

    conv_base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3)
    )

    conv_base.trainable = False

    model = models.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    train_generator = train_datagen.flow(
        train_images,
        y_train,
        batch_size=32,
        shuffle=True
    )

    history = model.fit(
        train_generator,
        validation_data=(test_images, y_test),
        epochs=10,
        verbose=1
    )

    model.save(VGG_MODEL_PATH)

    np.savez(
        VGG_HISTORY_PATH,
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        accuracy=history.history["accuracy"],
        val_accuracy=history.history["val_accuracy"]
    )

    return model

# add cache collector for re-use purposes
@st.cache_resource
def load_or_train_model(model_type):

    if model_type == "VGG16":

        if os.path.exists(VGG_MODEL_PATH):
            return models.load_model(VGG_MODEL_PATH)

        return train_vgg16_model()

    else:

        if os.path.exists(SELF_MODEL_PATH):
            return models.load_model(SELF_MODEL_PATH)

        return train_self_model()

def predict_image(image_array, model_type):
    try:

        model = load_or_train_model(model_type)

        if model_type == "VGG16":

            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

        else:

            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=(0, -1))

        prediction = model.predict(img, verbose=0)

        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return class_names[class_index], confidence

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def evaluate_performance(model_type):
    history_path = (
        VGG_HISTORY_PATH
        if model_type == "VGG16"
        else SELF_HISTORY_PATH
    )

    if not os.path.exists(history_path):
        st.info("No training history found.")
        return

    try:
        data = np.load(history_path)

        loss = data["loss"]
        val_loss = data["val_loss"]
        acc = data["accuracy"]
        val_acc = data["val_accuracy"]

        epochs = np.arange(1, len(loss) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(epochs, loss, label="Train Loss")
        ax[0].plot(epochs, val_loss, label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].legend()

        ax[1].plot(epochs, acc, label="Train Accuracy")
        ax[1].plot(epochs, val_acc, label="Validation Accuracy")
        ax[1].set_title("Accuracy")
        ax[1].legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Performance display error: {e}")

# UI part of my dashboard
st.set_page_config(page_title="Fashion Classifier")

st.title("Fashion-MNIST Clothing Classifier")

choice = st.radio(
    "Choose model",
    ["VGG16", "Self-made model"]
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:

        st.error("Unable to read image.")

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:

            prediction, confidence = predict_image(
                img,
                choice
            )

            if prediction:

                st.subheader("Prediction")
                st.success(prediction)

                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

        st.divider()

        evaluate_performance(choice)