import numpy as np

class FutureValuePredictor:
    def __init__(self, final_model, sc, learn_r = 30):
        self.final_model = final_model
        self.sc = sc
        self.learn_r = learn_r

    def predict_future(self, sc_train, num_future_values):
        predicted_values = []
        for _ in range(num_future_values):
            x_next = np.array([sc_train[-self.learn_r:, 0]])
            x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 1))
            y_next_predict = self.final_model.predict(x_next)
            y_next_predict = self.sc.inverse_transform(y_next_predict)
            predicted_values.append(y_next_predict)
            sc_train = np.append(sc_train, y_next_predict, axis=0)
        future_values = [prediction[0][0] for prediction in predicted_values]
        return future_values