# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2
# Fase de entrenamiento de la IA

# Instalación de las librerí­as necesarias
# conda install -c conda-forge keras

# Importar las librerí­as y otros ficheros de python
import os
import numpy as np
import random as rn

import environment
import brain
import dqn

# Configurar las semillas para reproducibilidad
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS 
epsilon = 0.3
number_actions = 7
direction_boundary = 1
number_epochs = 100
max_memory = 30
batch_size = 10

env = environment.Environment(optimal_price = (0.9, 1.1))
# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT

# CONSTRUCCIÓN DEL CEREBRO CREANDO UN OBJETO DE LA CLASE BRAIN
brain = brain.Brain(learning_rate = 0.001, number_actions = number_actions)

# CONSTRUCCIÓN DEL MODELO DQN CREANDO UN OBJETO DE LA CLASE DQN
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = True

# ENTRENAR LA IA
env.train = train
model = brain.model

#early_stopping = True 
#patience = 10
#best_total_reward = -np.inf
#patience_count = 0
if (env.train):
    loss_list = []
    reward_list = []
    coin_list = []
    price_list = []
    action_list = []
    fair_price = []
    # INICIAR EL BUCLE DE TODAS LAS ÉPOCAS (1 Epoch = 5 Meses)
    for epoch in range(1, number_epochs+1):
        # INICIALIZACIÓN DE LAS VARIABLES DEL ENTORNO Y DEL BUCLE DE ENTRENAMIENTO
        total_reward = 0.
        loss = 0.
        env.reset()  
        game_over = False
        timestep = np.random.randint(1,800)
        current_state, _, _ = env.observe(timestep)
        stop = timestep + 360
        env.current_date = 0
        # INICIALIZACIÓN DEL BUCLE DE TIMESTEPS (Timestep = 1 minuto) EN UNA EPOCA
        while ((not game_over) and (timestep <= stop)):
            # EJECUTAR LA SIGUIENTE ACCIÓN POR EXPLORACIÓN
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                   
            # EJECUTAR LA SIGUIENTE ACCIÓN POR INFERENCIA
            else: 
                pred_state, pred_price = env.predict_env(timestep)
                q_values = model.predict(pred_state)
                action = np.argmax(q_values[0])
                
            coin_price = (env.current_market_cap/env.current_nvt)/env.coin_supplyai
            if (pred_price <= direction_boundary):
                direction = -1
            else:
                direction = 1
            if action == 0:
                coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 0. 
            elif action == 1:
                coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 0.5
            elif action == 2:
                coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 0.75
            elif action == 3:
                coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 1
            elif action == 4:
                if env.price_ai > 1/2:
                    coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 1.25
                else:
                    coin_var = env.coin_supplyai - 1
            elif action == 5:
                if env.price_ai > 1/3:
                    coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 1.5
                else:
                    coin_var = env.coin_supplyai - 1
            elif action == 6:
                if env.price_ai > 0.43:
                    coin_var = abs((pred_price * env.coin_supplyai) - env.coin_supplyai) * 1.75
                else:
                    coin_var = env.coin_supplyai - 1
            
            coin_var = int(coin_var)
            #coin_supplyai = abs(action - direction_boundary) * supply_step
            
            # ACTUALIZAR EL ENTORNO Y ALCANZAR EL SIGUIENTE ESTADO
            next_state, reward, game_over = env.update_env(direction, coin_var, timestep)
            total_reward = total_reward + reward
            reward_list.append(reward)
            coin_list.append(env.coin_supplyai)
            action_list.append(action)
            
            price_list.append(env.price_ai)
            # ALMACENAR LA NUEVA TRANSICIÓN EN LA MEMORIA
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            # OBTENER LOS DOS BLOQUES SEPARADOS DE ENTRADAS Y OBJETIVOS
            inputs, targets = dqn.get_batch(model, batch_size)
            
            # CALCULAR LA FUNCIÓN DE PÉRDIDAS UTILIZANDO TODO EL BLOQUE DE ENTRADA Y OBJETIVOS
            loss += model.train_on_batch(inputs, targets)
            loss_list.append(loss)
            if timestep % 50 == 0:
                print('Iteraci�n: ' + str(timestep))
            timestep += 1
            current_state = next_state

        # IMPRIMIR LOS RESULTADOS DEL ENTRENAMIENTO AL FINAL DEL EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print(" - Energia total gastada por el sistema con IA: {:.0f} J.".format(loss))
        print("game over en: {}".format(timestep))
        #print(" - Energia total gastada por el sistema sin IA: {:.0f} J.".format(env.total_energy_noai))
        # GUARDAR EL MODELO PARA SU USO FUTURO
        #if early_stopping:
         #   if (total_reward <= best_total_reward):
          #      patience_count += 1
           # else:
            #    best_total_reward = total_reward
             #   patience_count = 0
                
            #if patience_count >= patience:
             #   print("Ejecuci�n prematura del m�todo")
              #  break
        
    model.save("model_nvt.h5")

import matplotlib.pyplot as plt
plt.plot(coin_list, label = 'Funcion de perdidas en entrenamiento')
plt.plot(reward_list, label = 'Funcion de perdidas en entrenamiento')
plt.plot(loss_list, label = 'Funcion de perdidas en entrenamiento')
plt.xlim(xmin=300)
plt.plot(price_list, label = 'Funcion de perdidas en entrenamiento')
plt.ylim(ymax=2, ymin=0)
plt.plot(action_list)
plt.show()

