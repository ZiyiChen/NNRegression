import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



def print_with_line_separator(s):
    print("==================================================================================")
    print(s)


def read_traning_data():
    df_train = pd.read_csv('hw_data_set_1.csv')
    print_with_line_separator("training shape: \n" + str(df_train.shape))
    print_with_line_separator("training dtypes: \n" + str(df_train.dtypes))
    print_with_line_separator("training head: \n" + str(df_train.head()))
    print_with_line_separator("training price describe: \n" + str(df_train.price.describe()))
    print_with_line_separator("training make_id describe: \n" + str(df_train.make_id.describe()))
    print_with_line_separator("training door describe: \n" + str(df_train.door.describe()))
    print_with_line_separator("training base_msrp describe: \n" + str(df_train.base_msrp.describe()))
    print_with_line_separator("training transaction_msrp describe: \n" + str(df_train.transaction_msrp.describe()))
    print_with_line_separator("training destination describe: \n" + str(df_train.destination.describe()))
    print_with_line_separator("training zip describe: \n" + str(df_train.zip.describe()))
    print_with_line_separator("training longitude describe: \n" + str(df_train.longitude.describe()))
    print_with_line_separator("training latitude describe: \n" + str(df_train.latitude.describe()))
    print_with_line_separator("training base_msrp describe: \n" + str(df_train.base_msrp.describe()))
    print_with_line_separator("training transaction_msrp describe: \n" + str(df_train.transaction_msrp.describe()))
    print_with_line_separator("training cash describe: \n" + str(df_train.cash.describe()))
    print_with_line_separator("training dealercash describe: \n" + str(df_train.dealercash.describe()))
    print_with_line_separator("training customercash describe: \n" + str(df_train.customercash.describe()))

    return df_train


def clean_data(df_train):
    # delete dirty data
    df_checking = df_train[df_train.price >= 5e+05]
    print_with_line_separator("training data with price >= 5e+05:\n" + str(df_checking[['make', 'model', 'price']].head()))
    print_with_line_separator("df_checking price describe: \n" + str(df_checking.price.describe()))
    df_train = df_train[df_train.price < 5e+05]
    return df_train


def plot_time_hist(df_train):
    (df_train['sales_date_int']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('sales_date_int')
    plt.xlim([-2, 2])
    plt.ylabel('frequency')
    plt.title('Sales Time Distribution - Training Set')
    plt.show()


def plot_eng_seq_hist(df_train):
    df_train.seq_engine.apply(lambda x: len(x)).hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('seq_engine length')
    plt.ylabel('frequency')
    plt.title('seq_engine length Distribution - Training Set')
    plt.show()


def plot_trim_seq_hist(df_train):
    df_train.seq_trim.apply(lambda x: len(x)).hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('seq_trim length')
    plt.ylabel('frequency')
    plt.title('seq_trim length Distribution - Training Set')
    plt.show()


def plot_year_hist(df_train):
    (df_train['year']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('year')
    plt.xlim([2008, 2013])
    plt.ylabel('frequency')
    plt.title('Year Distribution - Training Set')
    plt.show()


def plot_zip_hist(df_train):
    (df_train['zip']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('zip')
    plt.ylabel('frequency')
    plt.title('Zip Distribution - Training Set')
    plt.show()


def plot_make_id_hist(df_train):
    (df_train['make_id']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('make_id')
    plt.ylabel('frequency')
    plt.title('Make_id Distribution - Training Set')
    plt.show()


def plot_trim_id_hist(df_train):
    (df_train['trim_id']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('trim_id')
    plt.ylabel('frequency')
    plt.title('trim_id Distribution - Training Set')
    plt.show()


def read_testing_data():
    df_test = pd.read_csv('hw_data_set_2.csv')
    print_with_line_separator("testing shape: \n" + str(df_test.shape))
    print_with_line_separator("testing dtypes: \n" + str(df_test.dtypes))
    print_with_line_separator("testing head: \n" + str(df_test.head()))
    # handle missing data
    df_test.sales_date.fillna(value=strftime("%d/%b%Y", gmtime()), inplace=True)
    return df_test


def plot_price_log_price(df_train):
    plt.subplot(1, 2, 1)
    (df_train['price']).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('price', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Price Distribution - Training Set', fontsize=17)

    plt.subplot(1, 2, 2)
    np.log(df_train['price'] + 1).plot.hist(bins=50, figsize=(20, 10), edgecolor='white')
    plt.xlabel('log(price)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.title('Log(Price+1) Distribution - Training Set', fontsize=17)
    plt.show()


def analyze_data(df_train, df_test):
    # year, make, model, model_id variables analysis
    train_year_make_model_seq = df_train['year'].map(str) + df_train['make'] + df_train['model']
    print_with_line_separator('there are ' + str(train_year_make_model_seq.nunique())
                              + ' unique year + make + model in df_train')
    train_year_make_model_mid_seq = train_year_make_model_seq + df_train['model_id'].map(str)
    print_with_line_separator('there are ' + str(train_year_make_model_mid_seq.nunique())
                              + ' unique year + make + model + model_id in df_train')
    print_with_line_separator('there are ' + str(df_train['model_id'].nunique()) + ' unique model_id in df_train')

    test_year_make_model_mid_seq = \
        df_test['year'].map(str) + df_test['make'] + df_test['model'] + df_test['model_id'].map(str)
    print_with_line_separator('there are ' + str(test_year_make_model_mid_seq.nunique())
                              + ' unique year + make + model + model_id in df_test')
    diff = set(test_year_make_model_mid_seq.unique()) - set(train_year_make_model_mid_seq.unique())
    print_with_line_separator('diff between test_year_make_model_mid_seq - train_year_make_model_mid_seq: \n'
                              + str(diff))
    diff = set(df_test['model_id'].unique()) - set(df_train['model_id'].unique())
    print_with_line_separator('diff between test_model_id - train_model_id: \n'
                              + str(diff))

    print_with_line_separator('there are ' + str(df_train['make'].nunique())
                              + ' unique make in df_train')
    print_with_line_separator('there are ' + str(df_train['make_id'].nunique())
                              + ' unique make_id in df_train')

    print_with_line_separator('there are ' + str(df_train['trim'].nunique())
                              + ' unique trim in df_train')
    print_with_line_separator('there are ' + str(df_train['trim_id'].nunique())
                              + ' unique trim_id in df_train')


def date_processing(df_train, df_test):
    # date time to processing
    print_with_line_separator(str(df_train.price.describe()))
    df_train['sales_date'] = pd.to_datetime(df_train['sales_date'], format="%d%b%Y")
    df_train['sales_date_int'] = df_train['sales_date'].dt.year * 10000 + df_train['sales_date'].dt.month * 100 + \
                                 df_train['sales_date'].dt.day
    print_with_line_separator("training sales_date_int describe: \n" + str(df_train.sales_date_int.describe()))
    df_checking = df_train[df_train.sales_date_int < 20000000]
    print_with_line_separator("df_train.sales_date_int < 20000000: \n" + str(df_checking))
    df_train = df_train[df_train.sales_date_int >= 20000000]
    df_train = df_train[df_train['sales_date_int']/10000 > df_train['year']]
    print_with_line_separator("training sales_date_int describe: \n" + str(df_train.sales_date_int.describe()))

    df_test['sales_date'] = pd.to_datetime(df_test['sales_date'], format="%d%b%Y")
    df_test['sales_date_int'] = df_test['sales_date'].dt.year * 10000 + df_test['sales_date'].dt.month * 100 + \
                                df_test['sales_date'].dt.day

    # target_scaler = MinMaxScaler(feature_range=(0, 15000))
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.sales_date_int.values.reshape(-1, 1))
    df_train["sales_date_int"] = target_scaler.transform(df_train.sales_date_int.values.reshape(-1, 1))
    df_test["sales_date_int"] = target_scaler.transform(df_test.sales_date_int.values.reshape(-1, 1))
    print_with_line_separator("training sales_date_int describe: \n" + str(df_train.sales_date_int.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.year.values.reshape(-1, 1))
    df_train["year"] = target_scaler.transform(df_train.year.values.reshape(-1, 1))
    df_test["year"] = target_scaler.transform(df_test.year.values.reshape(-1, 1))
    print_with_line_separator("training year describe: \n" + str(df_train.year.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.base_msrp.values.reshape(-1, 1))
    df_train["base_msrp"] = target_scaler.transform(df_train.base_msrp.values.reshape(-1, 1))
    df_test["base_msrp"] = target_scaler.transform(df_test.base_msrp.values.reshape(-1, 1))
    print_with_line_separator("training base_msrp describe: \n" + str(df_train.base_msrp.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.zip.values.reshape(-1, 1))
    df_train["zip"] = target_scaler.transform(df_train.zip.values.reshape(-1, 1))
    df_test["zip"] = target_scaler.transform(df_test.zip.values.reshape(-1, 1))
    print_with_line_separator("training zip describe: \n" + str(df_train.zip.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.transaction_msrp.values.reshape(-1, 1))
    df_train["transaction_msrp"] = target_scaler.transform(df_train.transaction_msrp.values.reshape(-1, 1))
    df_test["transaction_msrp"] = target_scaler.transform(df_test.transaction_msrp.values.reshape(-1, 1))
    print_with_line_separator("training transaction_msrp describe: \n" + str(df_train.transaction_msrp.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.destination.values.reshape(-1, 1))
    df_train["destination"] = target_scaler.transform(df_train.destination.values.reshape(-1, 1))
    df_test["destination"] = target_scaler.transform(df_test.destination.values.reshape(-1, 1))
    print_with_line_separator("training destination describe: \n" + str(df_train.destination.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.dealercash.values.reshape(-1, 1))
    df_train["dealercash"] = target_scaler.transform(df_train.dealercash.values.reshape(-1, 1))
    df_test["dealercash"] = target_scaler.transform(df_test.dealercash.values.reshape(-1, 1))
    print_with_line_separator("training dealercash describe: \n" + str(df_train.dealercash.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.customercash.values.reshape(-1, 1))
    df_train["customercash"] = target_scaler.transform(df_train.customercash.values.reshape(-1, 1))
    df_test["customercash"] = target_scaler.transform(df_test.customercash.values.reshape(-1, 1))
    print_with_line_separator("training customercash describe: \n" + str(df_train.customercash.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.make_id.values.reshape(-1, 1))
    df_train["make_id"] = target_scaler.transform(df_train.make_id.values.reshape(-1, 1))
    df_test["make_id"] = target_scaler.transform(df_test.make_id.values.reshape(-1, 1))
    print_with_line_separator("training make_id describe: \n" + str(df_train.make_id.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.longitude.values.reshape(-1, 1))
    df_train["longitude"] = target_scaler.transform(df_train.longitude.values.reshape(-1, 1))
    df_test["longitude"] = target_scaler.transform(df_test.longitude.values.reshape(-1, 1))
    print_with_line_separator("training longitude describe: \n" + str(df_train.longitude.describe()))

    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.latitude.values.reshape(-1, 1))
    df_train["latitude"] = target_scaler.transform(df_train.latitude.values.reshape(-1, 1))
    df_test["latitude"] = target_scaler.transform(df_test.latitude.values.reshape(-1, 1))
    print_with_line_separator("training latitude describe: \n" + str(df_train.latitude.describe()))

    # text processing
    le = LabelEncoder()
    le.fit(np.hstack([df_train['model'], df_test['model']]))
    df_train['model'] = le.transform(df_train['model'])
    df_test['model'] = le.transform(df_test['model'])
    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.model.values.reshape(-1, 1))
    df_train["model"] = target_scaler.transform(df_train.model.values.reshape(-1, 1))
    df_test["model"] = target_scaler.transform(df_test.model.values.reshape(-1, 1))
    print_with_line_separator("training model describe: \n" + str(df_train.model.describe()))

    le.fit(np.hstack([df_train['drive_type'], df_test['drive_type']]))
    df_train['drive_type'] = le.transform(df_train['drive_type'])
    df_test['drive_type'] = le.transform(df_test['drive_type'])
    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.drive_type.values.reshape(-1, 1))
    df_train["drive_type"] = target_scaler.transform(df_train.drive_type.values.reshape(-1, 1))
    df_test["drive_type"] = target_scaler.transform(df_test.drive_type.values.reshape(-1, 1))
    print_with_line_separator("training drive_type describe: \n" + str(df_train.drive_type.describe()))


    le.fit(np.hstack([df_train['transmission'], df_test['transmission']]))
    df_train['transmission'] = le.transform(df_train['transmission'])
    df_test['transmission'] = le.transform(df_test['transmission'])
    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.transmission.values.reshape(-1, 1))
    df_train["transmission"] = target_scaler.transform(df_train.transmission.values.reshape(-1, 1))
    df_test["transmission"] = target_scaler.transform(df_test.transmission.values.reshape(-1, 1))
    print_with_line_separator("training transmission describe: \n" + str(df_train.transmission.describe()))

    le.fit(np.hstack([df_train['bodytype'], df_test['bodytype']]))
    df_train['bodytype'] = le.transform(df_train['bodytype'])
    df_test['bodytype'] = le.transform(df_test['bodytype'])
    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.bodytype.values.reshape(-1, 1))
    df_train["bodytype"] = target_scaler.transform(df_train.bodytype.values.reshape(-1, 1))
    df_test["bodytype"] = target_scaler.transform(df_test.bodytype.values.reshape(-1, 1))
    print_with_line_separator("training bodytype describe: \n" + str(df_train.bodytype.describe()))

    le.fit(np.hstack([df_train['State'], df_test['State']]))
    df_train['State'] = le.transform(df_train['State'])
    df_test['State'] = le.transform(df_test['State'])
    del target_scaler
    target_scaler = StandardScaler()
    target_scaler.fit(df_train.State.values.reshape(-1, 1))
    df_train["State"] = target_scaler.transform(df_train.State.values.reshape(-1, 1))
    df_test["State"] = target_scaler.transform(df_test.State.values.reshape(-1, 1))
    print_with_line_separator("training State describe: \n" + str(df_train.State.describe()))

    del le

    # word seq processing
    raw_text = np.hstack([df_train.engine, df_test.engine])
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    df_train["seq_engine"] = tok_raw.texts_to_sequences(df_train.engine)
    df_test["seq_engine"] = tok_raw.texts_to_sequences(df_test.engine)
    max_seq_engine_index = len(tok_raw.word_index)

    del tok_raw
    raw_text = np.hstack([df_train.trim, df_test.trim])
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    df_train["seq_trim"] = tok_raw.texts_to_sequences(df_train.trim)
    df_test["seq_trim"] = tok_raw.texts_to_sequences(df_test.trim)
    max_seq_trim_index = len(tok_raw.word_index)

    print_with_line_separator("training head: \n"
                              + str(df_train[['model', 'drive_type', 'transmission', 'bodytype', 'State', 'seq_engine', 'seq_trim']].head()))
    print_with_line_separator("testing head: \n"
                              + str(df_test[['model', 'drive_type', 'transmission', 'bodytype', 'State', 'seq_engine', 'seq_trim']].head()))

    max_seq_engine = np.max([np.max(df_train.seq_engine.apply(lambda x: len(x))), np.max(df_test.seq_engine.apply(lambda x: len(x)))])
    max_seq_trim = np.max([np.max(df_train.seq_trim.apply(lambda x: len(x))), np.max(df_test.seq_trim.apply(lambda x: len(x)))])
    print_with_line_separator("max engine seq: " + str(max_seq_engine))
    print_with_line_separator("max trim seq: " + str(max_seq_trim))
    return df_train, df_test, max_seq_engine_index, max_seq_trim_index, max_seq_engine, max_seq_trim


def plot_corr(data):
    corr = data.corr()
    plt.figure(figsize=(20, 10))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.show()


def plot_base_price(df_train):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df_train.base_msrp, df_train.price, 'o')
    ax.set_title('base_msrp ,price relation')
    plt.xlabel('base_msrp')
    plt.ylabel('price')
    plt.show()
