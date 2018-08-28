from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model


def rmsle(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_model(X_train, max_len):

    bodytype = Input(shape=[X_train["bodytype"].shape[1]], name="bodytype")
    State = Input(shape=[X_train["State"].shape[1]], name="State")
    make_id = Input(shape=[X_train["make_id"].shape[1]], name="make_id")
    model = Input(shape=[X_train["model"].shape[1]], name="model")
    drive_type = Input(shape=[X_train["drive_type"].shape[1]], name="drive_type")
    transmission = Input(shape=[X_train["transmission"].shape[1]], name="transmission")
    zip = Input(shape=[X_train["zip"].shape[1]], name="zip")
    seq_engine = Input(shape=[X_train["seq_engine"].shape[1]], name="seq_engine")
    seq_trim = Input(shape=[X_train["seq_trim"].shape[1]], name="seq_trim")
    year = Input(shape=[X_train["year"].shape[1]], name="year")
    door = Input(shape=[X_train["door"].shape[1]], name="door")
    base_msrp = Input(shape=[X_train["base_msrp"].shape[1]], name="base_msrp")
    transaction_msrp = Input(shape=[X_train["transaction_msrp"].shape[1]], name="transaction_msrp")
    destination = Input(shape=[X_train["destination"].shape[1]], name="destination")
    dealercash = Input(shape=[X_train["dealercash"].shape[1]], name="dealercash")
    cash = Input(shape=[X_train["cash"].shape[1]], name="cash")
    customercash = Input(shape=[X_train["customercash"].shape[1]], name="customercash")
    finance = Input(shape=[X_train["finance"].shape[1]], name="finance")
    lease = Input(shape=[X_train["lease"].shape[1]], name="lease")
    longitude = Input(shape=[X_train["longitude"].shape[1]], name="longitude")
    latitude = Input(shape=[X_train["latitude"].shape[1]], name="latitude")
    sales_date_int = Input(shape=[X_train["sales_date_int"].shape[1]], name="sales_date_int")

    # Embeddings layers
    emb_seq_engine = Embedding(max_len['seq_engine'], 64)(seq_engine)
    emb_seq_trim = Embedding(max_len['seq_trim'], 64)(seq_trim)

    # rnn layer
    rnn_seq_engine = GRU(64)(emb_seq_engine)
    rnn_seq_trim = GRU(64)(emb_seq_trim)

    # main layer
    layer = concatenate([
        make_id
        , model
        , drive_type
        , transmission
        , bodytype
        , State
        , zip
        , rnn_seq_engine
        , rnn_seq_trim
        , year
        , door
        , base_msrp
        , transaction_msrp
        , destination
        , dealercash
        , cash
        , customercash
        , finance
        , lease
        , longitude
        , latitude
        , sales_date_int])

    layer = Dropout(0.1)(Dense(256)(layer))
    layer = Dropout(0.1)(Dense(128)(layer))

    # output
    output = Dense(1)(layer)

    # model
    model = Model([make_id, model, drive_type, transmission, bodytype, State, zip, seq_engine, seq_trim, year, door
                      , base_msrp, transaction_msrp, destination, cash, customercash, finance, lease
                      , dealercash, longitude, latitude, sales_date_int], output)
    model.compile(loss="mse", optimizer="adam")

    model.summary()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model
