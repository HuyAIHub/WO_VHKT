import pandas as pd
import numpy
import numpy as np
import plotly.graph_objects as go
import glob
def preprocessing_data_init(path):
    df = pd.read_csv(path)
    df.rename(columns = {'sum_duration':'total_duration'}, inplace = True)
    df = pd.DataFrame(df)

    # Tạo một từ điển để lưu trữ các DataFrame của từng phòng ban
    phong_ban_dict = {}
    # Tạo các DataFrame cho từng phòng ban và lưu vào từ điển
    for name, group in df.groupby('phong_ban'):
        phong_ban_dict[name] = group
    df_days = {'df_wo_co_dien_day':phong_ban_dict['wo_co_dien'],
                'df_wo_ksnn_day':phong_ban_dict['wo_ksnn'],
                'df_wo_noc_day':phong_ban_dict['wo_noc'],
                'df_wo_others_day':phong_ban_dict['wo_others']
                }
    return df_days


def load_data(path):
    df_wo_co_dien = pd.read_csv(path + 'df_wo_co_dien_day.csv')
    df_wo_ksnn = pd.read_csv(path + 'df_wo_ksnn_day.csv')
    df_wo_noc = pd.read_csv(path + 'df_wo_noc_day.csv')
    df_wo_others = pd.read_csv(path + 'df_wo_others_day.csv')

    df_days = {'df_wo_co_dien_day':df_wo_co_dien,
                    'df_wo_ksnn_day':df_wo_ksnn,
                    'df_wo_noc_day':df_wo_noc,
                    'df_wo_others_day':df_wo_others
                    }
    return df_days


def plot_view(_df,name):
  fig = go.Figure()

  # Add original and resampled time series
  fig.add_trace(go.Scatter(x=_df.index, y=_df['total_duration'], mode='lines', name='Original'))

  # Customize layout
  fig.update_layout(title='Biểu đồ '+ name,
                    xaxis_title='Day',
                    yaxis_title='Total Duration')

  # Show plot
  fig.show()

