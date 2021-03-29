import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt



DATASET_PATH = "./data.json"



def load_data(dataset_path):


    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # concert list into np array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return (X, y)



def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")


    # create error subplot
    axs[1].plot(history.history["loss"], label = "train error")
    axs[1].plot(history.history["val_loss"], label = "test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()



def prepare_datasets(train_size, validation_size, test_size):

    # load data
    X, y = load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)


    return (X_train, X_validation, X_test, y_train, y_validation, y_test)



def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    
    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))


    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))


    return (model)


if __name__ == "__main__":
    
    # get train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.6, 0.2, 0.2)

    
    # create network
    input_shape = (X_train.shape[1], X_train.shape[2])    # 130, 13
    model = build_model(input_shape)

    
    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])

    model.summary()

    
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

    
    # plot accuracy and error over the epochs
    plot_history(history)

    
    # evaluate model on the test set
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=2)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print("Accuracy on train set is {}".format(train_accuracy))
    print("Accuracy on test set is {}".format(test_accuracy))