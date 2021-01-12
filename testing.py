# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Fase de testing


# Importar las librerí­as y otros ficheros de python
import os
import numpy as np
import random as rn
import tensorflow as tf 
import environment # no voy a usar dpq ni brain porque y tengo el modelo guardado

# Configurar las semillas para reproducibilidad
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS 
number_actions = 7
direction_boundary = 1


# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT
env = environment.Environment(optimal_price = (0.90, 1.1))

# CARGA DE UN MODELO PRE ENTRENADO
model = tf.keras.models.load_model("model_nvt.h5")


# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = False

# EJECUCIÓN DE UN AÑO DE SIMULACIÓN EN MODO INFERENCIA
env.train = train
current_state, _, _ = env.observe(1100)
fair_price_days = []
price_list = []
AUC = []
coin_list = []
for timestep in range(1100, 1197):
    pred_state, pred_price = env.predict_env(timestep)
    q_values = model.predict(pred_state)
    action = np.argmax(q_values[0])
    
    
    if (pred_price <= direction_boundary):
        direction = -1
    else:
        direction = 1
    if action == 0:
        coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 0
    elif action == 1:
        coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 0.5
    elif action == 2:
        coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 0.75
    elif action == 3:
        coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 1
    elif action == 4:
        if env.price_ai > 1/2:
            coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 1.25
        else:
            coin_var = env.coin_supplyai - 1
    elif action == 5:
        if env.price_ai > 1/3:
            coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 1.5
        else:
            coin_var = env.coin_supplyai - 1
    elif action == 6:
        if env.price_ai > 0.43:
            coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 1.75
        else:
            coin_var = env.coin_supplyai - 1
    coin_var = int(coin_var)
    next_state, reward, game_over = env.update_env(direction, coin_var, timestep)
    price_list.append(env.price_ai)
    coin_list.append(env.coin_supplyai)
    current_state = next_state
    if 0.9 < env.price_ai < 1.1:
        fair_price_days.append(1)
    else:
        fair_price_days.append(0)
    AUC.append(np.mean(fair_price_days))
import matplotlib.pyplot as plt
print(np.mean(fair_price_days))
plt.plot(price_list)
plt.plot(AUC)
plt.plot(coin_list)

            
# IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
