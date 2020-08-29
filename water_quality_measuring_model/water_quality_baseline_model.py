'''
- Notice -
0) Baseline Model로서, 앙상블이나 다수 모델의 조합을 배제한 단일 모델에서의 성능기준 확보를 목표.
1) Tensorflow 2.x 사용, 단일 GPU 연산
2) 작업폴더에 train_input.csv, train_output.csv, test_input.csv, test_output_pred.csv 위치
'''

#%% 전처리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Lambda, RNN, LSTMCell
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam


# 데이터 불러오기
train_input = pd.read_csv('./train_input.csv')
train_output = pd.read_csv('./train_output.csv')
test_input = pd.read_csv('./test_input.csv')
test_output_pred = pd.read_csv('./test_output_pred.csv')

# 정렬
train_input.sort_values(by=['ID', 'I', 'J', 'K'], inplace=True, ignore_index=True)
train_output.sort_values(by=['ID', 'I', 'J', 'K'], inplace=True, ignore_index=True)

# nan을 0으로
train_output.fillna(0, inplace=True)

# output 값에서 ID, I, J, K, Time 제거 / input 값 에서 ID, I, J, K 제거
train_output = train_output.iloc[:,5:]
train_input = train_input.iloc[:,4:]
test_input = test_input.iloc[:,4:]

# output 컬럼명 별도 저장
output_cols = test_output_pred.columns.to_list()

# 1000배 스케일링
train_output = (train_output * 1000).to_numpy(dtype='float32')
train_input = (train_input * 1000).to_numpy(dtype='float32')
test_input = (test_input * 1000).to_numpy(dtype='float32')

# output 값을 시계열로 변형
output_list = []
t = int(len(train_output) / len(train_input))
for i in range(len(train_input)):
    output_list.append(train_output[i*t:(i+1)*t].reshape(1, t, -1))
train_output = np.concatenate(output_list, axis=0)

# 데이터 분할
train_x, val_x, train_y, val_y = train_test_split(train_input, train_output, test_size=0.1, random_state=256)

# y값 분리: 초기값(전체 고정값)과 예측값
train_y_init = train_y[:, 0, :]
val_y_init = val_y[:, 0, :]
train_y = train_y[:, 1:, :]
val_y = val_y[:, 1:, :]


#%% Baseline 모델 생성
def create_baseline_model(timestep):
    '''x를 cell state로'''
    x = Input(shape=60, name='Input_x')
    d1 = Dense(128, activation='relu', name='Dense_1')(x) # LSTM의 cell state로 사용
    d2 = Dense(128, activation='relu', name='Dense_2')(d1) # LSTM의 cell state로 사용
    lstm_input = tf.concat([tf.reshape(d2, [-1, 1, 128]), tf.zeros((tf.shape(d2)[0], timestep-1, 128))], axis=1)

    '''LSTM Part'''
    l1 = LSTM(128, return_sequences = True, name='GRU_1')(lstm_input)
    l2 = LSTM(128, return_sequences = True, name='GRU_2')(l1)
    l3 = LSTM(128, return_sequences = True, name='GRU_3')(l2)    
    l4 = Dense(128, activation='relu', name='Dense_3')(l3)
    yhat = Dense(16, activation='relu', name='activation')(l4)
    model = Model(inputs=x, outputs=yhat)
    return model

baseline_model = create_baseline_model(timestep=457)
baseline_model.summary()

# 최적 성능 지점에서 모델의 가중치를 저장하도록 하는 콜백 함수
## 모델 metric에 mape를 참고용으로 넣었지만 log가 취해진 값의 mape이므로 목적값의 mape와는 차이가 있음에 유의.
checkpointer = ModelCheckpoint(monitor='val_loss', filepath='./base_weights_simple.hdf5', 
                               verbose=1, save_best_only=True)
baseline_model.compile(loss=mean_absolute_percentage_error, optimizer=Adam(lr=0.001), metrics=['mae']) 


#%% 학습
model_train_input = np.concatenate([train_x, train_y_init], axis=1)
model_val_input = np.concatenate([val_x, val_y_init], axis=1)

hist1 = baseline_model.fit(model_train_input, train_y, batch_size=1024, epochs=10000,
                 validation_data=(model_val_input, val_y), callbacks=[checkpointer], 
                 verbose=1)

# 저장된 최적 모델 가져오기
baseline_model.load_weights('./base_weights_simple.hdf5')

#%% Plot hist: 플롯 그리기
# 1차 학습 플롯
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist1.history['loss'], 'y', label='train loss')
loss_ax.plot(hist1.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist1.history['mae'], 'b', label='train mae')
acc_ax.plot(hist1.history['val_mae'], 'g', label='val mae')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('mae')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

#%% validation 결과 비교
# MAPE ## TF 자체제공 MAPE는 값을 %단위로 표시하고 이하 함수는 소수점으로 표시함에 주의.
def mape(y_true, y_pred):
    epsilon = 0.000001
    return np.mean(np.abs(y_true - y_pred) / (y_true + epsilon))

val_pred = baseline_model.predict(model_val_input)
print(mape(val_y, val_pred))

#%% 결과 예측 및 저장
# 제출할 결과 예측하기
pred_init = test_output_pred[test_output_pred['Time']==6848].to_numpy()[:, 5:]
pred_init = (pred_init * 1000).astype('float32') # 스케일링
model_test_input = np.concatenate([test_input, pred_init], axis=1)
pred = baseline_model.predict(model_test_input)

#%% 결과 생성
# 제출결과를 제출형식에 맞게 수정하기
pred = np.concatenate([pred_init.reshape(-1, 1, 16), pred], axis=1)
pred = pred / 1000
# 예측 결과를 2차원으로 정렬
pred = pred.reshape(-1, 16)
# 예측결과를 test_output_pred와 합치기
pred = pd.DataFrame(data=pred, columns=output_cols[5:])
pred = pd.concat([test_output_pred.iloc[:,:5], pred], axis=1)
# 예측 결과를 csv로 저장
pred.to_csv('baseline_results_simple.csv', index=False)
