# authors: Piotr Michałek s19333 & Kibort Jan s19916

from sklearn.model_selection import train_test_split
import glob
import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.image import imsave
from random import randint
from random import shuffle
from random import seed
import cv2
import os
from tqdm import tqdm
from sklearn import metrics
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD

PATH = '/home/marcin/Pobrane/Zadanie_rekrutacyjne_ML_Engineer/data/'
MODE = {'data_show': False, 'data_prepare': False, 'clean': False, 'model_1': False, 'model_2': False, 'model_3': True, 'scores': False}
LABELS = ['all', 'disabled', 'men', 'mothers', 'none', 'women']
BATH_SIZE = 32
EPOCHS = 30 #100
seed(1)


def cleaner():
    """"
    Delete folder with photos
    """
    os.system(f'rm -r {PATH}post_procesing')


def transformer(img, angular=0, affine=0, transpose=0, brightness=0):
    """
    Prepare photos.
    :param img: img
    :param angular: float
    :param affine: float
    :param transpose: float
    :param brightness: float
    :return: img
    """

    if angular:
        ang_rot = np.random.uniform(angular) - angular / 2
        rows, cols, ch = img.shape
        rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
        img = cv2.warpAffine(img, rotation, (cols, rows))

    if transpose:
        x = transpose * np.random.uniform() - transpose / 2
        y = transpose * np.random.uniform() - transpose / 2
        transposition = np.float32([[1, 0, x], [0, 1, y]])
        img = cv2.warpAffine(img, transposition, (cols, rows))

    if affine:
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + affine * np.random.uniform() - affine / 2
        pt2 = 20 + affine * np.random.uniform() - affine / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, shear, (cols, rows))

    if brightness:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        random_bright = 0.25 + np.random.uniform()
        if random_bright > 1:
            random_bright = 1
        img[:, :, 2] = img[:, :, 2] * random_bright
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img


def show(data):
    """
    Plot random images.
    :param data: img paths []
    :return: nothing
    """

    for i in range(9):
        id = randint(0, len(data) - 1)
        plt.subplot(330 + 1 + i)
        image = imread(data[id])
        plt.imshow(image)
    plt.show()


def prepocesing():
    """
    Find photos. Show data histogram. Show photos before and after preprocessing.
    """
    path_dict = {'all': [], 'disabled': [], 'men': [], 'mothers': [], 'none': [], 'women': []}
    data_size = []

    for label in LABELS:
        path_dict[label] = glob.glob(f'{PATH}{label}/*.png')
        data_size.append(len(path_dict[label]))

    x = np.arange(len(LABELS))
    width = 0.6

    fig, ax = plt.subplots()
    rect = ax.bar(x, data_size, width)

    ax.set_ylabel('Values')
    ax.set_title('All values')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rect)
    fig.tight_layout()
    plt.show()

    for label in LABELS:
        show(path_dict[label])

    os.system(f'mkdir {PATH}post_procesing')
    for label in tqdm(LABELS):
        os.system(f'mkdir {PATH}post_procesing/{label}')
        for path in tqdm(path_dict[label]):
            image = imread(path)
            _, tail = os.path.split(path)
            file_name = tail.split('.')
            for i in range(20):
                if i:
                    img_post = transformer(image, 20, 10, 5, brightness=1)
                else:
                    img_post = image
                img_post = cv2.resize(img_post, (100, 100))
                imsave(f'{PATH}post_procesing/{label}/{file_name[0]}({i}).{file_name[1]}', img_post)


def get_data(paths_dict):
    length = len(paths_dict['mothers'])
    # length = 100
    data_x = []
    data_y = []
    key = 0

    for label in tqdm(LABELS):
        shuffle(paths_dict[label])
        paths_dict[label] = paths_dict[label][0:length]
        for path in tqdm(paths_dict[label]):
            data_x.append(imread(path))
            data_y.append(key)
        key += 1

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
    y_train = to_categorical(y_train, num_classes=6)
    y_test_norm = y_test
    y_test = to_categorical(y_test, num_classes=6)
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)

    return X_train, y_train, X_test, y_test, y_test_norm


def build_model_1(shape, activation):
    """"
    Create CNN model.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape, padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation=activation))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(len(LABELS)))
    model.add(Activation('softmax'))

    return model


def build_model_2(shape, activation):
    """"
    Create CNN model.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape, padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation=activation))
    model.add(SpatialDropout2D(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(len(LABELS)))
    model.add(Activation('softmax'))

    return model


def build_model_3():
    """"
    Use pretrained model - VGG16.
    """

    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(6, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def prepare_data(x_train, x_test):
    """
    Prepare sdata to model VGG16. Model needs reshape images from (100, 100, 4) to (224, 224, 3)
    """

    train = []
    test = []

    for i in tqdm(range(int(0.2*len(x_train)))):
        img = cv2.cvtColor(x_train[i], cv2.COLOR_BGRA2BGR)
        train.append(cv2.resize(img, (224, 224)))
    del x_train
    for i in tqdm(range(int(0.2*len(x_test)))):
        img = cv2.cvtColor(x_test[i], cv2.COLOR_BGRA2BGR)
        test.append(cv2.resize(img, (224, 224)))
    del x_test

    x_train = np.stack(train)
    x_test = np.stack(test)

    return x_train, x_test


def model_fit(model, x_train, y_train, x_test, y_test):
    """
    Training and evaluating a model.
    """
    print(model.summary())
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATH_SIZE, validation_split=0.1, verbose=1)
    scores = model.evaluate(x_test, y_test, verbose=1)
    date = datetime.datetime.now()
    model.save(f'{date.strftime("%X")}.h5')
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("Loss: %.2f%%" % (scores[0] * 100))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = [i for i in range(EPOCHS)]

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def prediction(x_test, y_test_norm):
    """
    Check predictions model.
    """

    paths = glob.glob(f'/home/marcin/PycharmProjects/Symbol_clasifier/*.h5', recursive=True)

    for path in paths:
        model = load_model(path)
        predictions = model.predict_classes(x_test)
        print(path)
        print(metrics.classification_report(y_test_norm, predictions))
        confusion_matrix = metrics.confusion_matrix(y_true=y_test_norm, y_pred=predictions)
        print(confusion_matrix)
        normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        print("")
        print("Confusion matrix (normalised to % of total test data):")
        print(normalised_confusion_matrix)

        width = 12
        height = 12
        # fig, ax = plt.subplots()
        plt.figure(figsize=(width, height))
        plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.rainbow)
        plt.title("Confusion matrix \n(normalized to the entire test set [%])")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(tick_marks, LABELS, rotation=90)
        plt.yticks(tick_marks, LABELS)
        plt.tight_layout()
        plt.ylabel('Real value')
        plt.xlabel('Prediction value')
        plt.show()


if __name__ == '__main__':
    path_dict = {'all': [], 'disabled': [], 'men': [], 'mothers': [], 'none': [], 'women': []}

    if MODE['clean']:
        cleaner()

    if MODE['data_prepare']:
        prepocesing()

    for label in LABELS:
        path_dict[label] = glob.glob(f'{PATH}post_procesing/{label}/*.png')

    if MODE['data_show']:
        for label in LABELS:
            show(path_dict[label])

    x_train, y_train, x_test, y_test, y_test_norm = get_data(path_dict)

    if MODE['model_1']:
        model = build_model_1(x_test[0].shape, 'relu')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_fit(model, x_train, y_train, x_test, y_test)

    if MODE['model_2']:
        model = build_model_2(x_test[0].shape, 'relu')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_fit(model, x_train, y_train, x_test, y_test)

    if MODE['model_3']:
        model = build_model_3()
        x_train, x_test = prepare_data(x_train, x_test)
        model_fit(model, x_train, y_train, x_test, y_test)

    if MODE['scores']:
        prediction(x_test, y_test_norm)