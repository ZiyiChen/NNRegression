import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def read_traning_data(path):
    df = pd.read_csv(path)
    return df


def plot_base_price(df):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df.base_msrp, df.price, 'o')
    ax.set_title('base_msrp ,price relation')
    plt.xlabel('base_msrp')
    plt.ylabel('price')
    plt.show()


df = read_traning_data('hw_data_set_2_result.csv')
plot_base_price(df)
