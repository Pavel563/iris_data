import os
import zipfile
from random import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tqdm import tqdm

TEST_SIZE = 0.5
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
PATH = 'D:/data_science/labs/iris_data'
TRAIN_FOLDER = './train/'
TEST_FOLDER = './test/'
IMG_SIZE = 224
RESNET_WEIGHTS_PATH = 'D:/data_science/labs/iris_data/resnet50.h5'

train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")

with zipfile.ZipFile(train_image_path, "r") as z:
    z.extractall(".")

with zipfile.ZipFile(test_image_path, "r") as z:
    z.extractall(".")

train_image_list = os.listdir("./train/")[0:SAMPLE_SIZE]
test_image_list = os.listdir("./test/")


def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat':
        return [1, 0]
    elif pet == 'dog':
        return [0, 1]


def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER, img)
        if (isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img), np.array(label)])
    shuffle(data_df)
    return data_df


def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')


train = process_data(train_image_list, TRAIN_FOLDER)


def show_images(data, isTest=False):
    f, ax = plt.subplots(5, 5, figsize=(15, 15))
    for i, data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label == 1:
            str_label = 'Dog'
        elif label == 0:
            str_label = 'Cat'
        if (isTest):
            str_label = "None"
        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title("Label: {}".format(str_label))
    plt.show()


test = process_data(test_image_list, TEST_FOLDER, False)

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])

model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = True

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_model = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NO_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))


def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()


plot_accuracy_and_loss(train_model)

score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

predicted_classes = model.predict_classes(X_val)
y_true = np.argmax(y_val, axis=1)

correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]

target_names = ["Class {}:".format(i) for i in range(NUM_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

f, ax = plt.subplots(5, 5, figsize=(15, 15))
for i, data in enumerate(test[:25]):
    img_num = data[1]
    img_data = data[0]
    orig = img_data
    data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_predicted = 'Dog'
    else:
        str_predicted = 'Cat'
    ax[i // 5, i % 5].imshow(orig)
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_title("Predicted:{}".format(str_predicted))
plt.show()
