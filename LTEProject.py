import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
import shap
import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from IPython.display import Image, display
from sklearn.preprocessing import QuantileTransformer
from scipy.ndimage.interpolation import shift

# Constants
DATASET_FOLDER_PATH=r'D:\ENSC813\Training\Training\LTE_Dataset\Dataset'
INPUT_FILE_PATH = DATASET_FOLDER_PATH +  r'\car\A_2018.01.18_14.37.56.csv'
OUTPUT_FILE_PATH= DATASET_FOLDER_PATH +  r'\car\LTE-out.csv'
DOWNLOAD_BITRATE_KEY='DL_bitrate'
DOWNLOAD_BITRATE_TEST_PRED_KEY='DL_bitrate_test_pred'
DOWNLOAD_BITRATE_TRAIN_PRED_KEY='DL_bitrate_train_pred'
DOWNLOAD_BITRATE_GROUND_TRUTH_KEY='Ground truth'
UPLOAD_BITRATE_KEY='UL_bitrate'
STATE_KEY='State'
NETWORK_MODE_KEY='NetworkMode'
OPERATOR_NAME_KEY='Operatorname'
TIMESTAMP_KEY='Timestamp'
NR_RX_RSRP='NRxRSRP'
NR_RX_RSRQ='NRxRSRQ'
NA_VALUES=['-']
WINDOW_LENGTH=20
FUTURE_OFFSETS=[4, 8, 12, 16, 20, 24, 32]
BATCH_SIZE=32
STRIDE=1
NODES=100
EPOCHS=200
SAMPLING_RATE=1
TEST_RATIO=0.2
USE_DROPOUT=False
DROPOUT_RATIO=0.5
MIN_KBPS = 10
ENABLE_DISPLAY=False
y_error_list = []
LONGITUDE_KEY='Longitude'
LATITUDE_KEY='Latitude'
SPEED_KEY='Speed'
RSRP_KEY='RSRP'
RSRQ_KEY='RSRQ'
RSSI_KEY='RSSI'
CQI_KEY='CQI'
SNR_KEY='SNR'
SERVING_CELL_LON='ServingCell_Lon'
SERVING_CELL_LAT='ServingCell_Lat'
SERVING_CELL_DIST='ServingCell_Distance'
DEBUG=0

# Load data
def create_dataset():
    print(INPUT_FILE_PATH)
    all_files = glob.glob(INPUT_FILE_PATH)
    li = []
    for file_name in all_files:
        df = pd.read_csv(file_name, header=0, index_col=0, na_values=['-'], parse_dates=[TIMESTAMP_KEY])
        li.append(df)

    dataset = pd.concat(li, axis=0, ignore_index=False)
    column_names = dataset.columns.tolist()

    # Set download column as first column
    column_names.remove(DOWNLOAD_BITRATE_KEY)
    column_names = [DOWNLOAD_BITRATE_KEY] + column_names
    dataset=dataset.reindex(columns=column_names)
    print(dataset.head())

    # Drop columns that are repetitive or unneeded
    dataset.drop(columns=[STATE_KEY, SPEED_KEY, SERVING_CELL_DIST, NETWORK_MODE_KEY, OPERATOR_NAME_KEY, NR_RX_RSRP, NR_RX_RSRQ], inplace=True)

    # Save to file
    dataset.to_csv(OUTPUT_FILE_PATH)
    print(dataset.head())
    print(dataset.describe())

# Create dataset
create_dataset()
df_original = pd.read_csv(OUTPUT_FILE_PATH, header=0, index_col=0, na_values=NA_VALUES, parse_dates=[TIMESTAMP_KEY])
df_original = df_original.fillna(method='ffill')

# Set max and min values for features
df_max_min = pd.DataFrame(columns=df_original.columns)
max_row = pd.Series({ DOWNLOAD_BITRATE_KEY: 1000000,
                      LONGITUDE_KEY: 180,
                      LATITUDE_KEY: 90,
                      RSRP_KEY: -44,
                      RSRQ_KEY:-3,
                      SNR_KEY: 30,
                      CQI_KEY:30,
                      RSSI_KEY:0,
                      UPLOAD_BITRATE_KEY: 500000,
                      SERVING_CELL_LON: 180,
                      SERVING_CELL_LAT: 90})

min_row = pd.Series({ DOWNLOAD_BITRATE_KEY: 0,
                      LONGITUDE_KEY: -180,
                      LATITUDE_KEY: -90,
                      RSRP_KEY: -140,
                      RSRQ_KEY:-20,
                      SNR_KEY: 1,
                      CQI_KEY:0,
                      RSSI_KEY:-100,
                      UPLOAD_BITRATE_KEY: 0,
                      SERVING_CELL_LON: -180,
                      SERVING_CELL_LAT: -90})
df_max_min = df_max_min.append(max_row, ignore_index=True)
df_max_min = df_max_min.append(min_row, ignore_index=True)
print(df_max_min)

future_offset_index = 0
for FUTURE_OFFSET in FUTURE_OFFSETS:
    df = df_original.copy()
    start_index = 0
    end_index=df.shape[0]
    scaler = MinMaxScaler(feature_range=(0,1))
    if DEBUG > 0:
        data_transform = df.to_numpy()
    else:
        # scale data between 0 and 1 for max/min values
        scaler.fit_transform(df_max_min)
        data_transform = scaler.transform(df)
  
    features_scaled=data_transform
    target_scaled=data_transform[:,0]
    target_scaled = np.reshape(target_scaled, (target_scaled.shape[0], 1))
    target_scaled_full = np.array([])
    
    if FUTURE_OFFSET > 0:
        # Drop last FUTURE_OFFSET rows
        df.drop(df.tail(FUTURE_OFFSET).index,inplace=True)

        target_scaled_full = target_scaled
        print (target_scaled_full.shape)
        print (target_scaled.shape)
        print (features_scaled.shape)
        x = 1
        while x < FUTURE_OFFSET:
            target_scaled_full = np.concatenate([target_scaled_full, np.roll(target_scaled, x * -1, axis=0)], axis=1)
            x = x + 1
        target_scaled_full = target_scaled_full[:(x-1) * -1]
        features_scaled = features_scaled[:(x-1) * -1]

    x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled_full, test_size=TEST_RATIO, random_state=1, shuffle=False)

    num_of_features=len(df.columns)
    train_generator = TimeseriesGenerator(x_train, y_train, length=WINDOW_LENGTH, stride=STRIDE, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)
    test_generator = TimeseriesGenerator(x_test, y_test, length=WINDOW_LENGTH, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)

    if DEBUG > 0:
        for i in range(len(train_generator)):
            x, y = train_generator[i]
            print('%s => %s' % (x, y))

    model = Sequential()
    model.add(LSTM(NODES, input_shape=(WINDOW_LENGTH, num_of_features), return_sequences=False))
    model.add(Dense(FUTURE_OFFSET))
    print(model.summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      mode='min')

    cb_list = [early_stopping]

    model.compile(loss=tf.losses.MeanSquaredError(),
                              optimizer=tf.optimizers.Adam(),
                              metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=cb_list, verbose=2, shuffle=False)
    model.evaluate(test_generator, verbose=0)

    # Predictions
    predictions = model.predict(test_generator)
    predictions_train = model.predict(train_generator)

    df_prediction = pd.concat([pd.DataFrame(predictions[:,future_offset_index]), pd.DataFrame(x_test[:,1:][WINDOW_LENGTH:])], axis=1)
    df_prediction_train = pd.concat([pd.DataFrame(predictions_train[:,future_offset_index]), pd.DataFrame(x_train[:,1:][WINDOW_LENGTH:])], axis=1)

    rev_trans_test_pred=scaler.inverse_transform(df_prediction)
    rev_trans_train_pred=scaler.inverse_transform(df_prediction_train)

    #
    # Use same metric found in research paper: Empowering Video Players in Cellular Throughput Prediction from Radio Network Measurements
    #
    # Absolute Residual Error (ARE) = abs (max(10, Ri) - max(10, Rp))/ (max(10, Ri)) * 100
    # 
    # where Ri = measured throughput and Rp = predicted throughput
    # 
    np.set_printoptions(threshold=sys.maxsize)
    y_true, y_pred = df[DOWNLOAD_BITRATE_KEY][rev_trans_test_pred.shape[0] * -1:],  rev_trans_test_pred[:,0]
    y_error =  np.abs((np.maximum(MIN_KBPS, y_true) - np.maximum(MIN_KBPS, y_pred))) / np.maximum(MIN_KBPS, y_pred) * 100
    y_error_list.append(y_error)

    if DEBUG > 0:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print (y_true)
        print (y_pred)
        print (y_error)
        print (np.max(y_error))
        print (np.min(y_error))

    if ENABLE_DISPLAY:
        # Display ground truth and predictions on plot
        x_data = range(df.shape[0])
        plt.title('Downlink Throughput (P'  + str(WINDOW_LENGTH) + 'F' + str(FUTURE_OFFSET) + ')')
        plt.plot(x_data[-y_true.shape[0]:], y_true, label='ground truth')
        plt.plot(x_data[-y_pred.shape[0]:], y_pred,  color='r', label='test prediction')
        plt.show()

        # Display MSE vs Epochs
        plt.title('MSE vs Epochs (P'  + str(WINDOW_LENGTH) + 'F' + str(FUTURE_OFFSET) + ')')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.legend(loc='upper right')
        plt.show()

    future_offset_index = future_offset_index + 1

i = 0
fig, axs = plt.subplots(nrows=1, ncols=len(FUTURE_OFFSETS), sharey='all')
fig.suptitle('History and Horizon Combination')
while i < len(FUTURE_OFFSETS):
    # Build the plot
    axs[i].boxplot(y_error_list[i], showmeans=True)
    axs[i].set_xticklabels(['P' + str(WINDOW_LENGTH) + 'F' + str(FUTURE_OFFSETS[i])])
    axs[i].yaxis.grid(True)
    i+=1
plt.setp(axs[:], ylabel='Absolute Value of Residual Error (%)')
plt.show()
