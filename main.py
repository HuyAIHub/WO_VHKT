import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model #tải mô hình

from module.data_processing import preprocessing_data_init, load_data
from module.model import TimeSeriesPredictionModel
from module.predict import FutureValuePredictor


path_date = 'data/data_wo_split/'
# xu ly du lieu dau vao
df_days = load_data(path_date)

# xu ly data input model
Modle = TimeSeriesPredictionModel()
x_train_co_dien, y_train_co_dien , train_data_co_dien, test_data_co_dien , data_co_dien, df_co_dien, sc_co_dien, sc_train_co_dien = Modle.data_input_model(df_days['df_wo_co_dien_day'])
x_train_ksnn, y_train_ksnn , train_data_ksnn, test_data_ksnn , data_ksnn, df_ksnn, sc_ksnn, sc_train_ksnn = Modle.data_input_model(df_days['df_wo_ksnn_day'])
x_train_noc, y_train_noc , train_data_noc, test_data_noc , data_noc, df_noc, sc_noc, sc_train_noc = Modle.data_input_model(df_days['df_wo_noc_day'])
x_train_others, y_train_others , train_data_others, test_data_others , data_others, df_others, sc_others, sc_train_others = Modle.data_input_model(df_days['df_wo_others_day'])

#load model
final_model_co_dien = load_model("weights/model_co_dien.hdf5")
final_model_knss = load_model("weights/model_ksnn.hdf5")
final_model_noc = load_model("weights/model_noc.hdf5")
final_model_others = load_model("weights/model_others.hdf5")

#predict
co_dien_predictor = FutureValuePredictor(final_model_co_dien,sc_co_dien)
ksnn_predictor = FutureValuePredictor(final_model_knss,sc_ksnn)
noc_predictor = FutureValuePredictor(final_model_noc,sc_noc)
others_predictor = FutureValuePredictor(final_model_others,sc_others)


def convert_to_json(data):
    # Đảm bảo rằng dữ liệu không phải là một mảng hoặc một ma trận
    if isinstance(data, np.ndarray):
        data = data.tolist()  # Chuyển đổi numpy array sang list

    # Kiểm tra nếu dữ liệu là một từ điển
    if isinstance(data, dict):
        for key in data:
            # Nếu giá trị của từ điển là một numpy array, chuyển đổi sang list
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
            # Nếu giá trị của từ điển là một số float32, chuyển đổi sang float
            elif isinstance(data[key], np.float32):
                data[key] = float(data[key])
            # Đệ quy chuyển đổi các giá trị của từ điển khác
            else:
                data[key] = convert_to_json(data[key])
    # Nếu dữ liệu là một list
    elif isinstance(data, list):
        for i in range(len(data)):
            # Nếu giá trị trong list là một numpy array, chuyển đổi sang list
            if isinstance(data[i], np.ndarray):
                data[i] = data[i].tolist()
            # Nếu giá trị trong list là một số float32, chuyển đổi sang float
            elif isinstance(data[i], np.float32):
                data[i] = float(data[i])
            # Đệ quy chuyển đổi các giá trị khác trong list
            else:
                data[i] = convert_to_json(data[i])
    return data

def get_result_predict():
    future_co_dien = co_dien_predictor.predict_future(sc_train_co_dien,5)
    future_ksnn = ksnn_predictor.predict_future(sc_train_ksnn,5)
    future_noc = noc_predictor.predict_future(sc_train_noc,5)
    future_others = others_predictor.predict_future(sc_train_others,5)

    result_predict = {
        'co_dien': future_co_dien,
        'ksnn': future_ksnn,
        'noc': future_noc,
        'others': future_others,
    }
    print('--------result_predict_done---------')
    return convert_to_json(result_predict)