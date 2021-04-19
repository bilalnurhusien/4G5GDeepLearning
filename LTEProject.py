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
WINDOW_LENGTH=10
BATCH_SIZE=32
STRIDE=1
LSTM_NODES=100
EPOCHS=200
SAMPLING_RATE=1
TEST_RATIO=0.2

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
	dataset = dataset[(dataset[DOWNLOAD_BITRATE_KEY] > 0 )]
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
quantile = MinMaxScaler(feature_range=(0,1))
# scale data between 0 and 1
data_transform = quantile.fit_transform(df)
features_scaled=data_transform
target_scaled=data_transform[:,0]
x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=TEST_RATIO, random_state=1, shuffle=False)

multi_target_train = shift(y_train, 0, cval=np.NaN)
multi_target_test = shift(y_test, 0, cval=np.NaN)
#multi_target_train = [shift(y_train, -12, cval=np.NaN), shift(y_train, -16, cval=np.NaN), shift(y_train, -32, cval=np.NaN)]
#multi_target_test = [shift(y_test, -12, cval=np.NaN), shift(y_test, -16, cval=np.NaN), shift(y_test, -32, cval=np.NaN)]
num_of_features=len(df.columns)
train_generator = TimeseriesGenerator(x_train, multi_target_train, length=WINDOW_LENGTH, stride=STRIDE, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)
test_generator = TimeseriesGenerator(x_test, multi_target_test, length=WINDOW_LENGTH, sampling_rate=SAMPLING_RATE, batch_size=BATCH_SIZE)

print(train_generator[0])
model = Sequential()
model.add(LSTM(LSTM_NODES, input_shape=(WINDOW_LENGTH,num_of_features), return_sequences=False))
model.add(Dense(1))
print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
												  patience=5,
											      mode='min')

cb_list = [early_stopping]

model.compile(loss=tf.losses.MeanAbsoluteError(),
		      optimizer=tf.optimizers.Adam(),
		      metrics=[tf.metrics.MeanSquaredError()])

history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=cb_list, verbose=2, shuffle=False)
model.evaluate(test_generator, verbose=0)


# Predictions
predictions = model.predict(test_generator)
predictions_train = model.predict(train_generator)

df_prediction = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][WINDOW_LENGTH:])], axis=1)
df_prediction_train = pd.concat([pd.DataFrame(predictions_train), pd.DataFrame(x_train[:,1:][WINDOW_LENGTH:])], axis=1)

rev_trans_test_pred=quantile.inverse_transform(df_prediction)
rev_trans_train_pred=quantile.inverse_transform(df_prediction_train)
y_true, y_pred = df[DOWNLOAD_BITRATE_KEY][rev_trans_test_pred.shape[0]  * -1:],  rev_trans_test_pred[:,0]
yerror_mean = mean_absolute_percentage_error(y_true, y_pred)
yerror_std = np.std(yerror_mean)

# Display ground truth and predictions on plot
x_data = range(df.shape[0])
plt.plot(x_data[-rev_trans_test_pred.shape[0]:], df[DOWNLOAD_BITRATE_KEY][-rev_trans_test_pred.shape[0]:], label='ground truth')
plt.plot(x_data[-rev_trans_test_pred.shape[0]:], rev_trans_test_pred[:,0],  color='r', label='test prediction')
plt.show()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend(loc='upper right')
plt.show()

# Build the plot
fig, ax = plt.subplots()
labels = ['P10F1']
x_pos = np.arange(len(labels))
ax.bar(x_pos, yerror_mean,
       yerr=yerror_std,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('Absolute Value of Residual Error (%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('History and Horizon Combination')
ax.yaxis.grid(True)

plt.show()
