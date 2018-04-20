from keras import Model, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout, LSTM, Embedding
from keras.optimizers import RMSprop
from matplotlib import pyplot


def encoder_decoder_model(sequences_matrix,Y_train):
    num_decoder_tokens = 9
    max_words = 1000
    max_len = 150
    latent_dim = 64
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    encoder_embedding_layer = Embedding(max_words, 50, input_length=max_len)(encoder_inputs)
    encoder_lstm_layer, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding_layer)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm_layer = LSTM(latent_dim, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
    layer = Dense(256)(decoder_lstm_layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    decoder_outputs = Dense(num_decoder_tokens)(layer)
    decoder_outputs = Activation('sigmoid')(decoder_outputs)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Compile & run training
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!
    history = model.fit([sequences_matrix, Y_train], Y_train,
                    batch_size=10 , epochs=20,
                    validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(
         decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = Dense(num_decoder_tokens, activation='sigmoid')(decoder_outputs)
    decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    return encoder_model,decoder_model