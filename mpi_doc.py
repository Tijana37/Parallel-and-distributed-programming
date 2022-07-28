from mpi4py import MPI
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from os import listdir
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow
import time
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2 import adam
from keras.optimizers import adam_v2


directory_root = 'C:/Users/Tijana Atanasovska/PycharmProjects/plants_detection/plants_data'
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
width=256
height=256
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
    opt = adam_v2.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model_.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model_, aug

if rank == 0:
    print("Running process: ", comm.rank)


    time_start = time.time()
    image_list, label_list = convert_images(0,8)
    time_end = time.time()
    print("Time to process half data - rank 0: ", time_end - time_start)
    time_start = time.time()
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0

    time_end = time.time()
    print("Time to convert all data to numpy arrays- rank 0: ", time_end - time_start)

    send_data = {'image_list0': np_image_list[:], 'label_list0': label_list[:]}
    print("Sending image list and label list......")
    comm.send(send_data, dest=1, tag=11)

    x_train_second, x_test_second, y_train_second, y_test_second = comm.recv(source=1, tag=11)

    print("Received shuffled data ......")
    #
    # model_, aug = createmodel()
    # # train the network
    # # print("[INFO] training network...")
    #
    # with tensorflow.device('/cpu:0'):
    #     history = model_.fit_generator(
    #             aug.flow(x_train_second, y_train_second, batch_size=BS),
    #             validation_data=(x_test_second, y_test_second),
    #             steps_per_epoch=len(x_train_second) // BS,
    #             epochs=10, verbose=1
    #         )
    # #
    #     print("Trained")
    #     acc = history.history['accuracy']
    #     val_acc = history.history['val_accuracy']
    #     loss = history.history['loss']
    #     val_loss = history.history['val_loss']
    #     epochs = range(1, len(acc) + 1)
    #     # Train and validation accuracy
    #     plt.plot(epochs, acc, 'b', label='Training accurarcy')
    #     plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    #     plt.title('Training and Validation accurarcy')
    #     plt.legend()
    #
    #     plt.figure()
    #     # Train and validation loss
    #     plt.plot(epochs, loss, 'b', label='Training loss')
    #     plt.plot(epochs, val_loss, 'r', label='Validation loss')
    #     plt.title('Training and Validation loss')
    #     plt.legend()
    #     plt.show()


#print(model_.get_weights())
    # w1 = model_.get_weights()
    # print(w1[0])
    # w2 = model_.get_weights()
    # w =  (w1 + w2) / 2
    # print(w[0])


elif rank == 1:
    print("Running process: ", comm.rank)

    time_start = time.time()
    image_list, label_list = convert_images(8,15)
    time_end = time.time()
    print("Time to process half data - rank 1: ", time_end - time_start)


    np_image_list = np.array(image_list, dtype=np.float16) / 225.0

    received_data = comm.recv(source=0, tag=11)

    np_image_list_full = np.concatenate((np_image_list, received_data['image_list0']), axis=0)

    np_image_labels_full = np.concatenate((label_list, received_data['label_list0']), axis=0)

    #labelizer must be on the whole dataset to make binary encoder for 15 classes,not 7 and 8 separetely
    image_labels, n_classes = labelizer(np_image_labels_full)

    print("Received image list and label list..........")

    print("[INFO] Splitting full data to train, test....")
    x_train, x_test, y_train, y_test = train_test_split(np_image_list_full, image_labels, test_size=0.2, shuffle=True, random_state=42)

    x_train_first, x_test_first, y_train_first, y_test_first = x_train[:len(x_train)//2], x_test[:len(x_test)//2],\
                                                               y_train[:len(y_train)//2], y_test[:len(y_test)//2]

    x_train_second, x_test_second, y_train_second, y_test_second = x_train[len(x_train) // 2:], x_test[len(x_test) // 2:], \
                                                               y_train[len(y_train) // 2:], y_test[len(y_test) // 2 :]

    send_data = [x_train_second, x_test_second, y_train_second, y_test_second]
    print("Sending shuffled data......")
    comm.send(send_data, dest=0, tag=11)

    model_, aug = createmodel()
    # # train the network
    # print("[INFO] training network...")

    time_start = time.time()

    with tensorflow.device('/cpu:0'):
        history = model_.fit_generator(
            aug.flow(x_train_first, y_train_first, batch_size=BS),
            validation_data=(x_test_first, y_test_first),
            steps_per_epoch=len(x_train_second) // BS,
            epochs=1, verbose=1
        )
        time_end = time.time()
        print("Time to train 1 epoch half data - rank 1: ", time_end - time_start)
    #
        print("Trained")
    #     acc = history.history['accuracy']
    #     val_acc = history.history['val_accuracy']
    #     loss = history.history['loss']
    #     val_loss = history.history['val_loss']
    #     epochs = range(1, len(acc) + 1)
    #     # Train and validation accuracy
    #     plt.plot(epochs, acc, 'b', label='Training accurarcy')
    #     plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    #     plt.title('Training and Validation accurarcy')
    #     plt.legend()
    #
    #     plt.figure()
    #     # Train and validation loss
    #     plt.plot(epochs, loss, 'b', label='Training loss')
    #     plt.plot(epochs, val_loss, 'r', label='Validation loss')
    #     plt.title('Training and Validation loss')
    #     plt.legend()
    #     plt.show()
    #
    #
    #
