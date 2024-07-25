from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score
import numpy as np
import matplotlib.pyplot as plt

class Model :

    def __init__(self, df) :

        self.df = df

        self.train = None
        self.test = None
        
        self.trainX = None
        self.trainY = None

        self.testX = None
        self.testY = None

        self.trainPredict = None
        self.testPredict = None
        self.predicted = None

        self.index = None

    def prep_data(self, feature_col = 'NORMAL (mm)', Tp = 2894, step = 10) :

        self.train = np.array(self.df[feature_col][:Tp])
        self.test = np.array(self.df[feature_col][Tp:])
        print("Train data length:", self.train.shape)       
        print("Test data length:", self.test.shape)

        self.train = self.train.reshape(-1, 1)
        self.test = self.test.reshape(-1, 1)

        # add step elements into train and test
        self.test = np.append(self.test, np.repeat(self.test[-1,], step))
        self.train = np.append(self.train, np.repeat(self.train[-1,], step))
        print("Train data length:", self.train.shape)
        print("Test data length:", self.test.shape)

    def plot_train_test(self, Tp = 2894):

        train_len = Tp
        test_len = len(self.df) - Tp

        plt.figure(figsize = (15, 4))
        plt.title("Train and test data plotted together", fontsize=16)

        # Plot train data
        plt.plot(np.arange(train_len), self.train[:train_len], c='blue')

        # Plot test data
        plt.plot(np.arange(train_len, train_len + test_len), self.test[:test_len], c = 'orange', alpha = 0.7)

        plt.legend(['Train', 'Test'])
        plt.grid(True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()


    @staticmethod
    def convertToMatrix(data, step = 10) :
        X, Y = [], []

        for i in range(len(data) - step) :
            d = i+step
            X.append(data[i : d, ])
            Y.append(data[d, ])
    
        return np.array(X), np.array(Y)

    def call_for_conv_mat(self, step = 10) :

        self.trainX, self.trainY = self.convertToMatrix(self.train, step)
        self.testX, self.testY = self.convertToMatrix(self.test, step)

        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

        print("Training data shape:", self.trainX.shape,', ',self.trainY.shape)
        print("Test data shape:", self.testX.shape,', ',self.testY.shape)

    @staticmethod
    def build_lstm(num_units = 128, embedding = 4, num_dense = 32, learning_rate = 0.001) :
        """
        Builds and compiles a simple RNN model
        Arguments:
                num_units: Number of units of the LSTM layer
                embedding: Embedding length
                num_dense: Number of neurons in the dense layer followed by the RNN layer
                learning_rate: Learning rate (uses RMSprop optimizer)
        Returns:
                A compiled Keras model.
        """

        model = Sequential()
    
        #1st lstm layer with input shape
        model.add(LSTM(units = num_units, input_shape = (1, embedding), activation = "relu", return_sequences = 'True'))

        #2nd lstm layer
        model.add(LSTM(units = num_units, activation = "relu", return_sequences = 'True'))

        #3rd LSTM layer
        model.add(LSTM(units = 50, activation = "relu"))

        #Dense layer
        model.add(Dense(num_dense, activation = "relu"))
        model.add(Dense(1))


        model.compile(loss = 'mean_squared_error', optimizer = RMSprop(learning_rate = learning_rate), metrics = ['mse'])
    
        return model
    
    def call_for_build(self) :
        self.model_rainfall = self.build_lstm(num_units = 150, num_dense = 50, embedding = 10, learning_rate = 0.0001)
        print(self.model_rainfall.summary())


    class MyCallback(Callback):
        def on_epoch_end(self, epoch, logs = None) : 
            if (epoch + 1) % 50 == 0 and epoch > 0 :
                print("Epoch number {} done".format(epoch + 1))

    def model_training(self) :

        #Batch size and number of epochs
        batch_size = 16
        num_epochs = 1000

        self.model_rainfall.fit(self.trainX, self.trainY, 
            epochs = num_epochs, 
            batch_size = batch_size, 
            callbacks = [self.MyCallback()], verbose = 0)
        
    def model_loss_curve(self) :
        """Monitors loss over epochs and plots model losses"""

        plt.figure(figsize = (7,5))
        plt.title("RMSE loss over epochs",fontsize = 16)
        plt.plot(np.sqrt(self.model_rainfall.history.history['loss']), c = 'k', lw = 2)
        plt.grid(True)
        plt.xlabel("Epochs", fontsize = 14)
        plt.ylabel("Root-mean-squared error",fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def model_saw_plot(self) :
        plt.figure(figsize = (15, 4))
        plt.title("This is what the model saw", fontsize = 18)
        plt.plot(self.trainX[:, 0][:, 0], c = 'blue')
        plt.grid(True)
        plt.show()

    def model_predicted_plot(self) :

        self.trainPredict = self.model_rainfall.predict(self.trainX)
        self.testPredict = self.model_rainfall.predict(self.testX)
        self.predicted = np.concatenate((self.trainPredict, self.testPredict), axis = 0)

        plt.figure(figsize = (15, 4))
        plt.title("This is what the model predicted", fontsize = 18)
        plt.plot(self.testPredict, c = 'orange')
        plt.grid(True)
        plt.show()

    def groundTruth_prediction_plot(self) :
        """Plots ground truth or actual graph of values and predicted graph by the model"""

        Tp = 2894
        self.index = self.df.index.values

        plt.figure(figsize = (15, 5))
        plt.title("Rainfall: Ground truth and prediction together", fontsize = 18)
        plt.plot(self.index, self.df['NORMAL (mm)'], c = 'blue')
        plt.plot(self.index, self.predicted, c = 'orange', alpha = 0.75)
        plt.legend(['True data','Predicted'], fontsize = 15)
        plt.axvline(x = Tp, c = "r")
        plt.grid(True)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(0, 50)     
        plt.show()

    def model_evaluation(self) :

        # Define Tp and N
        Tp = 2894  # Assign appropriate value
        N = 3617  # Define the upper limit for the index

        # Ensure predicted and df['NORMAL (mm)'] values are reshaped correctly
        predicted_values = self.predicted[Tp : N].reshape(-1)
        actual_values = self.df['NORMAL (mm)'][Tp : N].values.reshape(-1)

        # Calculate the errors
        error = predicted_values - actual_values

        # Flatten the error array if necessary
        error = np.array(error).ravel()     

        plt.figure(figsize = (7,5))
        plt.hist(error, bins = 25, edgecolor = 'k', color = 'orange')
        plt.show()
    
        plt.figure(figsize = (15, 4))
        plt.plot(error,c = 'blue', alpha = 0.75)
        plt.hlines(y = 0, xmin = -10, xmax = 650, color = 'k', lw = 3)
        plt.xlim(-10, 650)
        plt.grid(True)
        plt.show()

        # Assume testX and testY are already scaled appropriately
        self.predicted = self.model_rainfall.predict(self.testX)

        # Evaluate the model using the scaled values directly
        mse = MSE(self.testY, self.predicted)
        rmse = np.sqrt(mse)
        mae = MAE(self.testY, self.predicted)
        r2 = r2_score(self.testY, self.predicted)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")