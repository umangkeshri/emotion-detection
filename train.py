import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Dropout,
    Flatten,
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Building the model architecture
def build_model():
    model = Sequential()

    model.add(
        Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)
        )
    )
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.0001, decay=1e-6),
        metrics=["accuracy"],
    )
    return model


def simple_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(7, 7),
            padding="same",
            name="image_array",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(7, 7), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=num_classes, kernel_size=(3, 3), padding="same"))
    model.add(GlobalAveragePooling2D())
    model.add(Activation("softmax", name="predictions"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# plots accuracy and loss curves
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(
        range(1, len(model_history.history["accuracy"]) + 1),
        model_history.history["accuracy"],
    )
    axs[0].plot(
        range(1, len(model_history.history["val_accuracy"]) + 1),
        model_history.history["val_accuracy"],
    )
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(
        np.arange(1, len(model_history.history["accuracy"]) + 1),
        len(model_history.history["accuracy"]) / 10,
    )
    axs[0].legend(["train", "val"], loc="best")
    # summarize history for loss
    axs[1].plot(
        range(1, len(model_history.history["loss"]) + 1),
        model_history.history["loss"],
    )
    axs[1].plot(
        range(1, len(model_history.history["val_loss"]) + 1),
        model_history.history["val_loss"],
    )
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(
        np.arange(1, len(model_history.history["loss"]) + 1),
        len(model_history.history["loss"]) / 10,
    )
    axs[1].legend(["train", "val"], loc="best")
    fig.savefig("plot.png")
    plt.show()


def main(args):
    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.data, "train"),
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(args.data, "test"),
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    model = build_model()
    # model = simple_CNN((48,48,1), 7)

    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        callbacks=[EarlyStopping(monitor="loss", patience=2)],
    )
    plot_model_history(model_info)
    model.save(os.path.join(args.model, "model.h5"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data", help="Path for the fer2013 dataset")
    args.add_argument("--model", help="Path to save the models")

    main(args.parse_args())
