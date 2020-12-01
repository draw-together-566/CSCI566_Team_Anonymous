# Team Anonymous
# Auto Generation Model for Drawings

from keras.models import Model
from keras.layers import Input

from keras.layers import Dense, LSTM, CuDNNLSTM, Bidirectional, Lambda
from keras.activations import softmax, exponential, tanh

from keras.layers.core import RepeatVector

from keras import backend as K
from keras.initializers import RandomNormal


import numpy as np
import random

from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy


def get_default_hparams():
    params_dict = {
        # Experiment:
        'is_training': True,  
        'data_set': 'cat',  
        'epochs': 50,  
        'batch_size': 100, 
        'accelerate_LSTM': False,  
        # Loss:
        'optimizer': 'adam',  
        'learning_rate': 0.0005,
        'decay_rate': 0.999999, 
        'min_learning_rate': .000005, 
        'kl_tolerance': 0.25,  
        'kl_weight': 0.5, 
        'kl_weight_start': 0.01,  
        'kl_decay_rate': 0.99995,  
        'grad': 1.0,  
        # Model:
        'z_size': 128, 
        'enc_rnn_size': 256,  
        'dec_rnn_size': 256,  
        'use_recurrent_dropout': True, 
        'recurrent_dropout_prob': 0.9,  
        'num_mixture': 50,  
        # Data:
        'random_scale_factor': 0.2,  
        'augment_stroke_prob': 0.15  
    }

    return params_dict


class rnnBasedVAE(object):

    def __init__(self, parameters):
        # Hyper parameters
        self.parameters = parameters
        self.model = self.buildModel()
        self.model.summary()

        # Optimizer
        if self.parameters['optimizer'] == 'adam':
            self.optimizer = Adam(lr=self.parameters['learning_rate'], clipvalue=self.parameters['grad'])
        elif self.parameters['optimizer'] == 'sgd':
            self.optimizer = SGD(lr=self.parameters['learning_rate'], momentum=0.8, clipvalue=self.parameters['grad'])
        else:
            raise ValueError('wrong optimizer!')
        # Loss Function
        self.loss_func = self.loss()
        self.sample_models = {}

    def buildModel(self):

        self.encoder = Input(shape=(self.parameters['max_seq_len'], 15), name='encoder')
        decoder = Input(shape=(self.parameters['max_seq_len'], 15), name='decoder')

        recurrent_dropout = 1.0-self.parameters['recurrent_dropout_prob'] if (self.parameters['use_recurrent_dropout'] and (self.parameters['accelerate_LSTM'] is False)) else 0

        if self.parameters['accelerate_LSTM'] and self.parameters['is_training']:
            lstm_incoder = CuDNNLSTM(units=self.parameters['enc_rnn_size'])
            lstm_decoder = CuDNNLSTM(units=self.parameters['dec_rnn_size'], return_sequences=True, return_state=True)
            self.parameters['use_recurrent_dropout'] = False
            print('Using CuDNNLSTM - No Recurrent Dropout!')
        else:
            lstm_incoder = LSTM(units=self.parameters['enc_rnn_size'], recurrent_dropout=recurrent_dropout)
            lstm_decoder = LSTM(units=self.parameters['dec_rnn_size'], recurrent_dropout=recurrent_dropout,
                                      return_sequences=True, return_state=True)

        # Bidirectional LSTM Encoder:
        encoder = Bidirectional(lstm_incoder, merge_mode='concat')(self.encoder)
        self.batch_z = self.latent_z(encoder)

        # Decoder:
        self.decoder = lstm_decoder

        self.initial_state = Dense(units=2*self.decoder.units, activation='tanh', name='dec_initial_state',
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.003))
        initial_state = self.initial_state(self.batch_z)

        init_h, init_c = (initial_state[:, :self.decoder.units], initial_state[:, self.decoder.units:])
        tile_z = RepeatVector(self.parameters['max_seq_len'])(self.batch_z)
        decoder_con_inputs = Concatenate()([decoder, tile_z])

        [decoder_output, final_state1, final_state_2] = self.decoder(decoder_con_inputs, initial_state=[init_h, init_c])
    

        n_out = (3 + self.parameters['num_mixture'] * 6)

        self.output = Dense(n_out, name='output')
        output = self.output(decoder_output)

        model_output = Model([self.encoder, decoder], output)

        return model_output

    def latent_z(self, encoder_output):

        def transform(z_params):
           
            mu, sigma = z_params
            sigma_exp = K.exp(sigma / 2.0)
            noise = mu + sigma_exp*K.random_normal(shape=K.shape(sigma_exp), mean=0.0, stddev=1.0)
            return noise
        self.mu = Dense(units=self.parameters['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
        self.sigma = Dense(units=self.parameters['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
        return Lambda(transform)([self.mu, self.sigma])

    def kl_loss(self, *args, **kwargs):
        kl_cost = -0.5*K.mean(1+self.sigma-K.square(self.mu)-K.exp(self.sigma))

        return K.maximum(kl_cost, self.parameters['kl_tolerance'])

    def total_reconstruction_loss(self, y_true, y_pred):
        out = self.process_coef(y_pred)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_drawing, o_drawing_logits] = out

        [x1_data, x2_data] = [y_true[:, :, 0], y_true[:, :, 1]]
        drawing_data = y_true[:, :, 2:5]
        pdf_values = self.keras_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)

        guassian = pdf_values * o_pi
        guassian = K.sum(guassian, 2, keepdims=True)

        gmm_loss = -K.log(guassian +1e-6) 

        fs = 1.0 - drawing_data[:, :, 2]
        fs = K.expand_dims(fs)
        gmm_loss = gmm_loss * fs
        drawing_state_loss = categorical_crossentropy(drawing_data, o_drawing)
        drawing_state_loss = K.expand_dims(drawing_state_loss)

        drawing_state_loss = K.switch(K.learning_phase(), drawing_state_loss, drawing_state_loss * fs)


        result = gmm_loss + drawing_state_loss

        reconstruction_loss = K.mean(result) 
        return reconstruction_loss

    def loss(self):
        # KL loss
        kl_loss = self.kl_loss
        # Reconstruction loss
        rec_loss = self.total_reconstruction_loss

        # weight w
        self.kl_weight = K.variable(self.parameters['kl_weight_start'], name='kl_weight')
        kl_weight = self.kl_weight

        def vaeModelLoss(y_true, y_pred):
            md_loss = rec_loss(y_true, y_pred)
            loss = kl_weight*kl_loss() + md_loss
            return loss

        return vaeModelLoss

    def process_coef(self, out_tensor):

        # drawing states:
        z_drawing_logits = out_tensor[:, :, 0:3]
        M = self.parameters['num_mixture']
        dist_params = [out_tensor[:, :, (3 + M * (n - 1)):(3 + M * n)] for n in range(1, 7)]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = dist_params


        z_pi = softmax(z_pi)
        z_drawing = softmax(z_drawing_logits)

        z_sigma1 = exponential(z_sigma1)
        z_sigma2 = exponential(z_sigma2)
        z_corr = tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_drawing, z_drawing_logits]
        return r

    def keras_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        M = mu1.shape[2]  # Number of mixtures
        norm1 = K.tile(K.expand_dims(x1), [1, 1, M]) - mu1
        norm2 = K.tile(K.expand_dims(x2), [1, 1, M]) - mu2
        s1s2 = s1 * s2
        z = K.square(norm1 / s1) + K.square(norm2 / s2) - 2.0 * (rho * norm1 * norm2) / s1s2
        neg_rho = 1.0 - K.square(rho)
        result = K.exp((-z) / (2 * neg_rho))
        denom = 2 * np.pi * s1s2 * K.sqrt(neg_rho)
        result = result / denom
        return result

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func,
                           metrics=[self.total_reconstruction_loss, self.kl_loss])
        print('Model Compiled!')

    def load_trained_weights(self, weights):
        self.model.load_weights(weights)
        print('Weights from {} loaded successfully'.format(weights))

    def make_sampling_models(self)
        models = {}

        # Phase 1
        batch_z = Input(shape=(self.parameters['z_size'],))
        initial_state = self.initial_state(batch_z)
        models['inital_z_state'] = Model(inputs=batch_z, outputs=initial_state)

        # Phase 2
        decoder = Input(shape=(1, 5))
        initial_h_input = Input(shape=(self.decoder.units,))
        initial_c_input = Input(shape=(self.decoder.units,))

        # Phase 3
        tile_z = RepeatVector(1)(batch_z)
        decoder_con_inputs = Concatenate()([decoder, tile_z])
        [decoder_output, final_state_1, final_state_2] = self.decoder(decoder_con_inputs,
                                                                      initial_state=[initial_h_input, initial_c_input])
        final_state = [final_state_1, final_state_2]
        model_outpututput = self.output(decoder_output)

        mixture_params = Lambda(self.process_coef)(model_output
utput)
        models['sample_output_model'] = Model(inputs=[decoder, initial_h_input, initial_c_input, batch_z],
                                              outputs=final_state + mixture_params)

        models['encoder_model'] = Model(inputs=self.encoder, outputs=self.batch_z)

        self.sample_models = models
        print('Model done')


def sample(rnnVAE_model, seq_len=250, temperature=1.0, greedy_mode=False, z=None):
    parameters = rnnVAE_model.parameters

    def adjust_t(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def pixel_idx(x, pdf, temp=1.0, greedy=False):
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_t(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        print('Wrong sampling')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1
    if z is None:
        z = np.random.randn(1, parameters['z_size'])
    inital_z_state = rnnVAE_model.sample_models['inital_z_state']

    prev_state = inital_z_state.predict(z)

    prev_state = [prev_state[:, :rnnVAE_model.decoder.units], prev_state[:, rnnVAE_model.decoder.units:]]

    sample_output_model = rnnVAE_model.sample_models['sample_output_model']

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []

    for i in range(seq_len):
        feed = {
            'decoder input': prev_x,
            'initial_state': prev_state,
            'batch_z': z
        }
        model_out_list = sample_output_model.predict([feed['decoder input'], feed['initial_state'][0], feed['initial_state'][1], feed['batch_z']])
        next_state = model_out_list[:2]
        mixture_params_val = model_out_list[2:]
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_drawing, _] = mixture_params_val

        idx = pixel_idx(random.random(), o_pi[0][0], temperature, greedy_mode)
        idx_eos = pixel_idx(random.random(), o_drawing[0][0], temperature, greedy_mode)
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][0][idx], o_mu2[0][0][idx], o_sigma1[0][0][idx], o_sigma2[0][0][idx], o_corr[0][0][idx], np.sqrt(temperature), greedy_mode)

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[0][0], o_mu1[0][0], o_mu2[0][0], o_sigma1[0][0], o_sigma2[0][0], o_corr[0][0],
            o_drawing[0][0]
        ]

        mixture_params.apdrawingd(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0][0] = np.array(strokes[i, :], dtype=np.float32)
        prev_state = next_state

    return strokes, mixture_params