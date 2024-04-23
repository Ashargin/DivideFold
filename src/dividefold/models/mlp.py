from keras.layers import Input, Dropout, Dense, Flatten, Softmax
from keras.models import Model


def MLP(input_shape=(None, 297)):
    # Create classifier model using transformer layer
    transformer_ff_dim = 64  # Feed forward network size inside transformer
    num_heads = 8  # Number of attention heads
    dropout_rate = 0.1
    middle_dense_dim = 16

    inputs = Input(shape=input_shape)
    transformed = (inputs)

    drop1 = Dropout(dropout_rate)(transformed)
    dense1 = Dense(middle_dense_dim, activation="relu")(drop1)
    drop2 = Dropout(dropout_rate)(dense1)

    # Cuts with independant probability at each position
    dense2 = Dense(1, activation="sigmoid")(drop2)
    pred_cuts = Flatten()(dense2)

    # Other method with ensemble probability (softmax)
    # dense2 = Dense(1, activation='linear')(drop2)
    # flattened = Flatten()(dense2)
    # pred_cuts = Softmax()(flattened)

    # Train and evaluate
    model = Model(inputs=inputs, outputs=pred_cuts)

    return model
