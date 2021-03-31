import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dot, Embedding, Input
import tensorflow as tf
import keras.backend as K
from sklearn.linear_model import LinearRegression


class MCITR:

    def __init__(self, layer_enc=1, layer_dec=1, layer_cov=0, act_enc="linear", act_dec="linear", act_cov="linear", width_enc=None,
                        width_dec=None, width_embed=None, width_cov=None, optimizer="sgd", initializer="glorot_uniform", verbose=0, bias_enc=True, bias_dec=True, bias_cov=True):

        self.layer_enc, self.layer_dec = layer_enc, layer_dec # layer of encoder, decoder
        self.act_enc, self.act_dec = act_enc, act_dec # activation of encode, decoder
        self.width_enc, self.width_embed, self.width_dec = width_enc, width_embed, width_dec # width of encoder, decoder, embedding layers
        self.layer_cov, self.act_cov, self.width_cov = layer_cov, act_cov, width_cov # layer, activation and with for covariates embedding
        self.bias_enc, self.bias_dec, self.bias_cov = bias_enc, bias_dec, bias_cov
        self.optimizer = optimizer
        self.initializer = initializer
        self.verbose=verbose

    def model_s1_define(self, input_dim):

        ## Stage 1 Model, including autoencoder for treatment, encoder for covariates, and a whole model wrap everything

        trt_dim, cov_dim = input_dim # input_dim is a tuple

        with tf.device("/CPU:0"):

            def encoder():

                enc_layers = []
                
                trt = Input(shape=(trt_dim, ), name="treatment_input")

                enc_layers.append(trt)

                if self.layer_enc > 1:

                    if self.width_enc == None:
                        raise Exception("Encoder width is not specified.")
                    else:
                        for l in range(self.layer_enc - 1):

                            trt = Dense(self.width_enc, activation=self.act_enc, kernel_initializer=self.initializer, use_bias=self.bias_enc, name="treatment_encoder_{0}".format(l + 1))(trt)

                            enc_layers.append(trt)
                
                if self.width_embed == None:
                    self.width_embed = cov_dim
                    
                trt = Dense(self.width_embed, use_bias=self.bias_enc, kernel_initializer=self.initializer, name="treatment_embedding")(trt) # embedding layer

                enc_layers.append(trt)
                
                
                return enc_layers

            def decoder(trt):

                dec_layers = []

                if self.layer_dec > 1:
                    
                    if self.width_dec == None:
                        raise Exception("Decoder width is not specified.")
                    else:
                        for l in range(self.layer_dec - 1):

                            trt = Dense(self.width_dec, activation=self.act_dec, kernel_initializer=self.initializer, use_bias=self.bias_dec, name="treatment_decoder_{0}".format(l + 1))(trt)

                            dec_layers.append(trt)

                trt = Dense(trt_dim, activation="sigmoid", kernel_initializer=self.initializer, use_bias=self.bias_dec, name="treatment_output")(trt)

                dec_layers.append(trt)
 
                return dec_layers

            # define autoencoder
            enc_layers = encoder()
            dec_layers = decoder(enc_layers[-1])
            trt_autoencoder = keras.Model(inputs=enc_layers[0], outputs=dec_layers[-1])


            def covariate_embedding():
                
                cov_layers = []

                cov = Input(shape=(cov_dim, ), name="covariate_input")

                cov_layers.append(cov)

                if self.layer_cov > 0:

                    if self.width_cov == None:
                        raise Exception("Covariate embedding is not specified.")
                    else:
                        for l in range(self.layer_cov - 1):

                            cov = Dense(self.width_cov, activation=self.act_cov, kernel_initializer=self.initializer, use_bias=self.bias_cov, name="covariate_encoder_{0}".format(l + 1))(cov)

                            cov_layers.append(cov)
                        
                        cov = Dense(self.width_embed, kernel_initializer=self.initializer, use_bias=self.bias_cov, name="covariate_embedding")(cov)

                        cov_layers.append(cov)


                return cov_layers

            cov_layers = covariate_embedding()

            if self.layer_cov > 1:
                cov_encoder = keras.Model(inputs=cov_layers[0], outputs=cov_layers[-1])

            ## Stage 1 Model: fit the residuals with treatment and covariate embedding

            product = Dot(axes=1)([enc_layers[-1], cov_layers[-1]])

            model_s1 = keras.Model(inputs=[enc_layers[0], cov_layers[0]], 
                                   outputs=[dec_layers[-1], product])

        
            # define encoder
            trt_encoder = keras.Model(inputs=trt_autoencoder.layers[0].input, 
                                      outputs=trt_autoencoder.layers[self.layer_enc].output)

            # define decoder
            restored_w = []
            for w in trt_autoencoder.layers[(self.layer_enc + 1): ]:
                restored_w.extend(w.get_weights())

            dec_input = Input(shape=(self.width_embed, ))
            dec_layers = decoder(dec_input)
            trt_decoder = keras.Model(inputs=dec_input, outputs=dec_layers[-1])
            trt_decoder.set_weights(restored_w)


            self.model_s1, self.trt_encoder, self.trt_decoder = model_s1, trt_encoder, trt_decoder
            
            if self.layer_cov > 1:
                self.cov_encoder = cov_encoder


    def model_s1_fit(self, inputs, outputs, learning_rate, epochs):
        
        with tf.device("/CPU:0"):

            if self.optimizer == "sgd":
                self.model_s1.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss=["binary_crossentropy", "mse"], loss_weights=[0.2, 0.8])
            elif self.optimizer == "adam":
                self.model_s1.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=["binary_crossentropy", "mse"], loss_weights=[0.2, 0.8])
            self.model_s1.fit(inputs, outputs, epochs=epochs, verbose=self.verbose)


    def fit(self, Y, X, A, learning_rate, epochs, R=None):

        if A.ndim == 1:
            raise Exception("Only one channel treatment.")
        elif A.ndim > 1:
            if X.ndim == 1:
                input_dim = (A.shape[1], 1) 
            elif X.ndim > 1:
                input_dim = (A.shape[1], X.shape[1])

        if R is None:
            # compute residuals
            if X.ndim == 2:
                lm = LinearRegression().fit(X, Y)
                R = Y - lm.predict(X)
            elif X.ndim == 1:
                lm = LinearRegression().fit(X[:, np.newaxis], Y)
                R = Y - lm.predict(X[:, np.newaxis])

        inputs = [A, X]
        outputs = [A, R]

        # stage 1 model:
        self.model_s1_define(input_dim)
        self.model_s1_fit(inputs, outputs, learning_rate, epochs)


