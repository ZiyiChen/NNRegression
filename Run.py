import numpy as np
import Analyze as any
import KerasModel as ksm
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def train(df_train, df_test, max_seq_engine_index, max_seq_trim_index, max_seq_engine, max_seq_trim):
    max_len = {
        'seq_engine': max_seq_engine_index + 5,
        'seq_trim': max_seq_trim_index + 5,
    }


    # scale target variable
    # df_train["target"] = np.log(df_train.price + 1)
    # target_scaler = MinMaxScaler(feature_range=(-1, 1))
    # df_train["target"] = target_scaler.fit_transform(df_train.target.values.reshape(-1, 1))

    # splitting training and development
    dtrain, ddev = train_test_split(df_train, random_state=123, train_size=0.90)
    xtrain = get_keras_data(dtrain, max_seq_engine, max_seq_trim)
    xdev = get_keras_data(ddev, max_seq_engine, max_seq_trim)
    print("dtrain: " + str(dtrain.shape))
    print("ddev: " + str(ddev.shape))
    model = ksm.get_model(xtrain, max_len)
    # training...
    history = model.fit(xtrain, dtrain.price, epochs=20, batch_size=128, validation_data=(xdev, ddev.price), verbose=1)
    val_preds = model.predict(xdev)
    # val_preds = target_scaler.inverse_transform(val_preds)
    # val_preds = np.exp(val_preds) + 1
    y_true = np.array(ddev.price.values)
    y_pred = val_preds[:, 0]
    mser = rmse(y_true, y_pred)
    print("MSE error on cross validation set: " + str(mser))
    return model, history


def plot_acc(history):
    # summarize history for loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def plot_err(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def rmse(y, y_pred):
    assert len(y) == len(y_pred)
    return np.sqrt(np.sum(np.square(y - y_pred)) / len(y))


def get_keras_data(dataset, MAX_ENGINE_SEQ, MAX_TRIM):
    X = {
        'seq_engine': pad_sequences(dataset.seq_engine, maxlen=MAX_ENGINE_SEQ)
        , 'seq_trim': pad_sequences(dataset.seq_trim, maxlen=MAX_TRIM)
        , 'year': np.array(dataset[["year"]])
        , 'make_id': np.array(dataset[["make_id"]])
        , 'model': np.array(dataset[["model"]])
        , 'drive_type': np.array(dataset[["drive_type"]])
        , 'door': np.array(dataset[["door"]])
        , 'transmission': np.array(dataset[["transmission"]])
        , 'base_msrp': np.array(dataset[["base_msrp"]])
        , 'transaction_msrp': np.array(dataset[["transaction_msrp"]])
        , 'destination': np.array(dataset[["destination"]])
        , 'bodytype': np.array(dataset[["bodytype"]])
        , 'zip': np.array(dataset[["zip"]])
        , 'State': np.array(dataset[["State"]])
        , 'dealercash': np.array(dataset[["dealercash"]])
        , 'customercash': np.array(dataset[["customercash"]])
        , 'finance': np.array(dataset[["finance"]])
        , 'lease': np.array(dataset[["lease"]])
        , 'cash': np.array(dataset[["cash"]])
        , 'longitude': np.array(dataset[["longitude"]])
        , 'latitude': np.array(dataset[["latitude"]])
        , 'sales_date_int': np.array(dataset[["sales_date_int"]])
    }
    return X


def predict_test_set(model, df_test, max_seq_engine, max_seq_trim):
    xtest = get_keras_data(df_test, max_seq_engine, max_seq_trim)
    val_preds = model.predict(xtest)
    y_pred = val_preds[:, 0]
    test_csv = pd.read_csv('hw_data_set_2.csv')
    test_csv['price'] = pd.Series(y_pred, index=test_csv.index)
    # any.print_with_line_separator("test_result head():\n" + str(test_csv.head()))
    test_csv.to_csv('hw_data_set_2_result.csv', index=False)


df_train = any.read_traning_data()
df_test = any.read_testing_data()
# any.plot_corr(df_train)
# any.plot_base_price(df_train)
df_train = any.clean_data(df_train)

# any.plot_price_log_price(df_train)
# any.plot_zip_hist(df_train)
# any.plot_make_id_hist(df_train)
# any.plot_trim_id_hist(df_train)
any.analyze_data(df_train, df_test)

df_train, df_test, max_seq_engine_index, max_seq_trim_index, max_seq_engine, max_seq_trim = any.date_processing(df_train, df_test)
max_seq_engine += 5
max_seq_trim += 5
# any.plot_corr(df_train)
# any.plot_eng_seq_hist(df_train)
# any.plot_trim_seq_hist(df_train)
# any.plot_time_hist(df_train)
# any.plot_base_price(df_train)
model, history = train(df_train, df_test, max_seq_engine_index, max_seq_trim_index, max_seq_engine, max_seq_trim)
# plot_acc(history)
plot_err(history)
predict_test_set(model, df_test, max_seq_engine, max_seq_trim)
