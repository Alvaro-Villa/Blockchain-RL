# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Creación del Cerebro




import tensorflow as tf

class Brain(object):
    def __init__(self, learning_rate = 0.001, number_actions = 7):
        self.learning_rate = learning_rate
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(3, )))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=number_actions, activation='softmax'))
        self.model = model
        self.model.compile = model.compile(optimizer='adam', loss='mse')
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        



'''
import tensorflow as tf
# Importar las librerías
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUCCIÓN DEL CEREBRO
model = tf.keras.models.Sequential()
class Brain(object):
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation = "sigmoid")(states)
        y = Dense(units = 32, activation = "sigmoid")(x)
        q_values = Dense(units = number_actions, activation = "softmax")(y) # Softmax elegimos la Q mas alta
        self.model = Model(inputs = states, output = q_values)
        self.model.compile(loss = "mse", optimizer = Adam(lr = learning_rate))'''