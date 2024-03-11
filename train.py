import pandas as pd
import numpy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from module.data_processing import preprocessing_data_init, load_data
from module.model import TimeSeriesPredictionModel


path_date = 'data/data_wo_split/'
# load data
df_days = load_data(path_date)

# xu ly data input model
x_train_co_dien, y_train_co_dien , train_data_co_dien, test_data_co_dien , data_co_dien, df_co_dien, sc_co_dien, sc_train_co_dien = TimeSeriesPredictionModel.data_input_model(df_days['df_wo_co_dien_day'])
x_train_knss, y_train_knss , train_data_knss, test_data_knss , data_knss, df_knss, sc_knss, sc_train_knss = TimeSeriesPredictionModel.data_input_model(df_days['df_wo_ksnn_day'])
x_train_noc, y_train_noc , train_data_noc, test_data_noc , data_noc, df_noc, sc_noc, sc_train_noc = TimeSeriesPredictionModel.data_input_model(df_days['df_wo_noc_day'])
x_train_others, y_train_others , train_data_others, test_data_others , data_others, df_others, sc_others, sc_train_others = TimeSeriesPredictionModel.data_input_model(df_days['df_wo_others_day'])


TimeSeriesPredictionModel.build_model()

#build model
model_co_dien = TimeSeriesPredictionModel.build_model(x_train_co_dien,y_train_co_dien,epoch = 100, batch_size = 100,save_model = 'weights/model_co_dien.hdf5')
model_knss = TimeSeriesPredictionModel.build_model(x_train_knss,y_train_knss,epoch = 100, batch_size = 100,save_model = 'weights/model_ksnn.hdf5')
model_noc = TimeSeriesPredictionModel.build_model(x_train_noc,y_train_noc,epoch = 100, batch_size = 100,save_model = 'weights/model_noc.hdf5')
model_others = TimeSeriesPredictionModel.build_model(x_train_others,y_train_others,epoch = 100, batch_size = 100,save_model = 'weights/model_others.hdf5')