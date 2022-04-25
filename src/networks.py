import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

class HeteroscedasticNetwork(tf.keras.Model):
    
    def __init__(self, n_params_d, n_params_s):
        super(HeteroscedasticNetwork, self).__init__()
        
        self.preprocessor = Sequential([
            GRU(64, return_sequences=True),
            LSTM(128, return_sequences=True),
            Dense(128, activation='selu', kernel_initializer='lecun_normal'),
        ])
        
        self.dynamic_predictor = Sequential([
            Dense(64, activation='selu', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(n_params_d)),
            tfpl.MultivariateNormalTriL(n_params_d)
        ])

        self.static_predictor = Sequential([
            LSTM(n_params_s),
            Dense(tfpl.MultivariateNormalTriL.params_size(n_params_s)),
            tfpl.MultivariateNormalTriL(n_params_s)
        ])
        
    def call(self, x):
        """
        Forward pass through the model.
        ----------
        Input:
        np.array of shape (batchsize, n_obs, 5)
        ----------
        Output:
        tf.tensor distribution of shape (batchsize, n_obs, n_params_d)
        tf.tensor distribution of shape (batchsize, n_params_s)
        """
        
        # obtain representation
        rep = self.preprocessor(x)
        
        # predict dynamic microscopic params
        preds_dyn = self.dynamic_predictor(rep)

        # predict static macroscopic params
        preds_stat = self.static_predictor(rep)

        return preds_dyn, preds_stat

class StaticHeteroscedasticNetwork(tf.keras.Model):
    
    def __init__(self, n_params):
        super(StaticHeteroscedasticNetwork, self).__init__()
        
        self.preprocessor = Sequential([
            GRU(64, return_sequences=True),
            LSTM(128, return_sequences=True),
            Dense(128, activation='selu', kernel_initializer='lecun_normal'),
        ])
        
        self.static_predictor = Sequential([
            Dense(64, activation='selu', kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(n_params)),
            tfpl.MultivariateNormalTriL(n_params)
        ])

        
    def call(self, x):
        """
        Forward pass through the model.
        ----------
        Input: np.array of shape (batchsize, n_obs, 5)
        ----------
        Output: tf.tensor distribution of shape (batchsize, n_obs, n_params)
        """

        # obtain representation
        rep = self.preprocessor(x)
        
        # predict static microscropic params
        preds_static = self.dynamic_predictor(rep)

        return preds_static