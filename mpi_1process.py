from mpi4py import MPI
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D,MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt
import time
import csv


directory_root = 'C:/Users/Tijana Atanasovska/PycharmProjects/plants_detection/plants_data'
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((128, 128))
image_size = 0
width=128
height=128
depth=3


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error in convert_image_to_Array() : {e}")
        return None


def convert_images(from_range, to_range):
    image_list, label_list = [], []
    #try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    print(root_dir)
    for directory in root_dir:
        # remove .DS_Store from list
        if directory == ".DS_Store":
            root_dir.remove(directory)

    for plant_folder in root_dir:
        print(listdir(f"{directory_root}/{plant_folder}"))
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")[from_range:to_range]




        for disease_folder in plant_disease_folder_list:
            # remove .DS_Store from list
            if disease_folder == ".DS_Store":
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

            for single_plant_disease_image in plant_disease_image_list:
                if single_plant_disease_image == ".DS_Store":
                    plant_disease_image_list.remove(single_plant_disease_image)

            f = "C:/Users/Tijana Atanasovska/PycharmProjects/plants_detection/data_arrays.csv"
            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)


    print("[INFO] Image loading completed")

    return image_list, label_list
    # except Exception as e:
    #     print(f"Error in convert_images()() : {e}")


def labelizer(label_list):
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)

    return image_labels, n_classes

def model(n_classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(128, (3, 3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    #print(model.summary())

    return model


def createmodel():

    print("creating model.......")
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest")

    model_ = model(n_classes=15)
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model_.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model_, aug

if rank == 0:
    print("Running process: ", comm.rank)

    time_start = time.time()
    image_list, label_list = convert_images(0,15)
    time_end = time.time()
    print("Time to process all data - rank 0: ", time_end - time_start)

    time_start = time.time()
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    time_end = time.time()
    print("Time to convert data to numpy arrays- rank 0: ", time_end - time_start)

    image_labels, n_classes = labelizer(label_list)

    print("[INFO] Splitting full data to train, test....")
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, \
                                                        test_size=0.2, shuffle=True, random_state=42)
    model_, aug = createmodel()
    # train the network
    print("[INFO] training network...")

    #for cycle to train the network so we can average weights from both tasks and update before every epoch
    time_full_training_s = time.time()

    with tensorflow.device('/cpu:0'):
        time_start = time.time()
        history = model_.fit_generator(
                aug.flow(x_train, y_train, batch_size=BS),
                validation_data=(x_test, y_test),
                steps_per_epoch=len(x_train) // BS,
                epochs=EPOCHS, verbose=1
            )
        time_end = time.time()
        print(f"Time to train 25 epochs - rank 0: ", time_end - time_start)

            #Wait for weights of the other task; sum and average the weights

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        with open('results_1process.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the data
            writer.writerow(acc)
            writer.writerow(val_acc)
            writer.writerow(loss)
            writer.writerow(val_loss)
