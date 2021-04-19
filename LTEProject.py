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
INPUT_FILE_PATH = DATASET_FOLDER_PATH +  r'\car\B_2018.01.18_14.38.07.csv'
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
FUTURE_OFFSET=32
BATCH_SIZE=32
STRIDE=1
NODES=100
EPOCHS=200
SAMPLING_RATE=1
TEST_RATIO=0.2
USE_DROPOUT=True

# Load data
def create_dataset():
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
	dataset.drop(columns=[STATE_KEY, NETWORK_MODE_KEY, OPERATOR_NAME_KEY], inplace=True)

	# Save to file
	dataset.to_csv(OUTPUT_FILE_PATH)
	print(dataset.head())
	print(dataset.describe())

create_dataset()
df = pd.read_csv(OUTPUT_FILE_PATH, header=0, index_col=0, na_values=NA_VALUES, parse_dates=[TIMESTAMP_KEY])
print(df)
print(df.info())
print(df.count())
print(df.describe())
df = df.fillna(method='ffill')

start_index = 0
end_index=df.shape[0]
scaler = MinMaxScaler(feature_range=(0,1))
# scale data between 0 and 1
data_transform = scaler.fit_transform(df)
features_scaled=data_transform
target_scaled=data_transform[:,0]
x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=TEST_RATIO, random_state=1, shuffle=False)

if FUTURE_OFFSET > 0:
	df.drop(df.tail(FUTURE_OFFSET).index,inplace=True) # drop last n rows
	y_train = shift(y_train, FUTURE_OFFSET * -1, cval=np.NaN)
	y_test = shift(y_test, FUTURE_OFFSET * -1, cval=np.NaN)
	y_train = y_train[:FUTURE_OFFSET * -1]
	y_test = y_test[:FUTURE_OFFSET * -1]
	x_train = x_train[:FUTURE_OFFSET * -1]
	x_test = x_test[:FUTURE_OFFSET * -1]

num_of_features=len(df.columns)
train_generator = TimeseriesGenerator(x_train, y_train, length=WINDOW_LENGTH, stride=STRIDE, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)
test_generator = TimeseriesGenerator(x_test, y_test, length=WINDOW_LENGTH, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)

print(train_generator[0])

model = Sequential()
model.add(GRU(NODES, input_shape=(WINDOW_LENGTH, num_of_features), return_sequences=True))
if USE_DROPOUT:
    model.add(Dropout(0.2))
model.add(GRU(NODES, return_sequences=False))
if USE_DROPOUT:
    model.add(Dropout(0.2))
model.add(Dense(1))
print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
												  patience=10,
											      mode='min')

cb_list = [early_stopping]

model.compile(loss=tf.losses.MeanAbsoluteError(),
		      optimizer=tf.optimizers.Adam(),
		      metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=cb_list, verbose=2, shuffle=False)
model.evaluate(test_generator, verbose=0)

# Predictions
predictions = model.predict(test_generator)
predictions_train = model.predict(train_generator)

df_prediction = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][WINDOW_LENGTH:])], axis=1)
df_prediction_train = pd.concat([pd.DataFrame(predictions_train), pd.DataFrame(x_train[:,1:][WINDOW_LENGTH:])], axis=1)

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
yerror =  np.abs((np.maximum(10, y_true) - np.maximum(10, y_pred))) / np.maximum(10, y_pred)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print (y_true)
print (y_pred)
print (yerror)
print (np.max(yerror))
print (np.min(yerror))

# Build the plot
fig, ax = plt.subplots()
labels = ['P' + str(WINDOW_LENGTH) + 'F' + str(FUTURE_OFFSET)]
x_pos = np.arange(len(labels))
ax.boxplot(yerror, positions=x_pos, showmeans=True)
ax.set_ylabel('Absolute Value of Residual Error (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('History and Horizon Combination')
ax.yaxis.grid(True)
plt.show()

# Display ground truth and predictions on plot
x_data = range(df.shape[0])
plt.plot(x_data[-rev_trans_test_pred.shape[0]:], y_true, label='ground truth')
plt.plot(x_data[-rev_trans_test_pred.shape[0]:], y_pred,  color='r', label='test prediction')
plt.show()

# Display MSE vs Epochs
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend(loc='upper right')
plt.show()


