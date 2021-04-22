import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logo_path = "../Logo_train"
logo_test = "../Logo_test"

### Tools
batch_size = 128
n_labels = len(os.listdir('../Logo_train'))

def generators(shape, preprocessing):
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 90,
        validation_split = 0.2,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        logo_path,
        target_size = (height, width),
        batch_size = batch_size,
        subset = 'training',
    )

    val_dataset = imgdatagen.flow_from_directory(
        logo_test,
        target_size = (height, width),
        batch_size = batch_size,
        subset = 'validation'
    )
    return train_dataset, val_dataset


def plot_history(history, yrange):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)

    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()


### Data sets
train_dataset, val_dataset = generators((224, 224), preprocessing=preprocess_input)

### Create Model
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=n_labels)
for layer in resnet.layers:
    layer.trainable = False

x = Flatten()(resnet.output)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
predictions = Dense(n_labels, activation='softmax')(x)

full_model = Model(inputs=resnet.input, outputs=predictions)

### training
full_model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adamax(lr=0.001),
                   metrics=['acc'])


history = full_model.fit_generator(
    train_dataset,
    validation_data= val_dataset,
    workers=10,
    epochs=5
)

plot_history(history, yrange=(0.9, 1))
full_model.save_weights('../src/resnet50.h5')

