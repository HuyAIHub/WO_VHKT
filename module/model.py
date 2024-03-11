import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score, mean_absolute_error

class TimeSeriesPredictionModel:
    def __init__(self, learn_r=30, training_rate=0.8):
        self.learn_r = learn_r
        self.training_rate = training_rate

    def data_input_model(self, data):
        df = pd.DataFrame(data, columns=['date', 'total_duration'])
        df.index = df.date
        df.drop('date', axis=1, inplace=True)

        # Chia tập dữ liệu
        rate = round(self.training_rate * df.shape[0])
        data = df.values

        train_data = data[:rate]
        test_data = data[rate:]

        # Chuẩn hóa dữ liệu
        sc = MinMaxScaler(feature_range=(0, 1))
        sc_train = sc.fit_transform(data)

        # Tạo vòng lặp các giá trị
        x_train, y_train = [], []
        for i in range(self.learn_r, len(train_data)):
            x_train.append(sc_train[i - self.learn_r:i, 0])  
            y_train.append(sc_train[i, 0])  

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        return x_train, y_train, train_data, test_data, data, df, sc, sc_train

    def build_model(self, x_train, y_train, epoch, batch_size, save_model):
        model = Sequential() 
        model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1), return_sequences=True))
        model.add(LSTM(units=64))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer='adam')

        best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=2, callbacks=[best_model])

        return model

    def evaluate(self, x_train, y_train, train_data, data, df, sc, final_model):
        y_train = sc.inverse_transform(y_train) 
        y_train_predict = final_model.predict(x_train) 
        y_train_predict = sc.inverse_transform(y_train_predict) 

        test = df[len(train_data) - self.learn_r:].values
        test = test.reshape(-1, 1)
        sc_test = sc.transform(test)

        x_test = []
        for i in range(self.learn_r, test.shape[0]):
            x_test.append(sc_test[i - self.learn_r:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        rate = round(self.training_rate * df.shape[0])
        y_test = data[rate:]

        y_test_predict = final_model.predict(x_test)
        y_test_predict = sc.inverse_transform(y_test_predict)

        train_data1 = df[self.learn_r:rate]
        test_data1 = df[rate:]

        print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
        print('Sai số tuyệt đối trung bình trên tập train (time):', mean_absolute_error(y_train, y_train_predict))
        print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
        print('Sai số tuyệt đối trung bình trên tập test (time):', mean_absolute_error(y_test, y_test_predict))

        return y_train_predict, y_test_predict

