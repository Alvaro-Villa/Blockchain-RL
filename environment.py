# Inteligencia Artifical
# Creación del Entorno

# Importar las librerías
import numpy as np
import pandas as pd
import tensorflow as tf
data = pd.read_csv('bitcoin_dataset.csv')
data = data[['btc_market_price', 'btc_total_bitcoins', 'btc_market_cap','btc_trade_volume']]
data = data[50:1258]
data.dropna(inplace=True)



# CONSTRUIR EL ENTORNO EN UNA CLASE

class Environment(object): # Editando los parametros nos sirve tantos servidores tengamos
    
    # INTRODUCIR E INICIALIZAR LOS PARÁMETROS Y VARIABLES DEL ENTORNO
    def __init__(self, optimal_price = (0.9, 1.1), opt_nvt = 15, data = data):     
        self.data = data
        self.X_mc = pd.read_csv('C:/Users/Master/Desktop/E-3 An/5/TFG analytics/RNN - market cap y trading vol/X_mc.csv')
        self.X_tv = pd.read_csv('C:/Users/Master/Desktop/E-3 An/5/TFG analytics/RNN - market cap y trading vol/X_tv.csv')
        self.regressor_mc = tf.keras.models.load_model("market_cap_regressor.h5")
        self.regressor_tv = tf.keras.models.load_model("trade_vol_regressor.h5")
        self.optimal_price = optimal_price
        self.optimal_nvt= opt_nvt
        self.min_price = 0
        self.min_supply = data.btc_total_bitcoins.min()
        self.max_supply = data.btc_total_bitcoins.max()
        self.min_market_cap = data.btc_market_cap.min()
        self.max_market_cap = data.btc_market_cap.max()
        self.min_trading_volume = data.btc_trade_volume.min()
        self.max_trading_volume = data.btc_trade_volume.max()
        self.initial_date = 0
        self.current_date = self.initial_date
        self.initial_trading_volume = data.iloc[0,3]
        self.current_trading_volume = self.initial_trading_volume # Actualizacion de t-1 a t
        self.initial_market_cap = data.iloc[0,2]
        self.current_market_cap = self.initial_market_cap
        self.initial_price = data.iloc[0,0]
        self.initial_nvt = self.current_market_cap / self.current_trading_volume
        self.current_nvt = self.initial_nvt
        self.initial_supplyai = data.iloc[0,1]
        self.coin_supplyai = self.initial_supplyai
        self.price_ai = (self.current_market_cap/self.current_nvt) / self.coin_supplyai
        self.fair_price = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.game_over = 0 # Indica el fin de la simulacion, variable booleana
        self.train = 1 # En testing sera 0
    
    # CREAR UN MÉTODO QUE ACTUALICE EL ENTORNO JUSTO DESPUÉS DE QUE LA IA EJECUTE UNA ACCIÓN
    def update_env(self, direction, coin_var, current_date):
        reward_ia = 0.0
        self.coin_supplyai += coin_var * direction
        if self.coin_supplyai  < 1:
            self.coin_supplyai = 1    
        else:
           self.price_ai = (self.current_market_cap/self.current_nvt)/self.coin_supplyai  
        
        if(self.price_ai  < self.optimal_price[0]):
            reward_ia = self.price_ai - self.optimal_price[0] # Seria interesante multiplicar la reward para que no quede sobrepasada por la recompensa del NVT
        elif self.price_ai > self.optimal_price[1]:
            reward_ia = self.optimal_price[1] - self.price_ai
        else:
            reward_ia = 10
            self.fair_price += 1
        self.reward = 1e-3*reward_ia
        #self.total_reward += self.reward
        
        # OBTENCIÓN DEL SIGUIENTE ESTADO
        self.current_trading_volume = data.iloc[self.current_date+1,3]
        self.current_market_cap = data.iloc[self.current_date+1,2]
        self.current_nvt = self.current_market_cap / self.current_trading_volume
        self.current_date += 1
        
    
        # ESCALAR EL SIGUIENTE ESTADO
        scaled_trading_volume = (self.current_trading_volume - self.min_trading_volume)/(self.max_trading_volume - self.min_trading_volume)
        scaled_market_cap = (self.current_market_cap - self.min_market_cap)/(self.max_market_cap - self.min_market_cap)
        scaled_supply = (self.coin_supplyai - self.min_supply)/(self.max_supply - self.min_supply)
        next_state = np.matrix([scaled_market_cap, scaled_trading_volume, scaled_supply]) # vector fila
        #next_state = np.matrix([self.current_market_cap, self.current_trading_volume, self.coin_supplyai])
        reward = self.reward  
        game_over = self.game_over

        # DEVOLVER EL SIGUIENTE ESTADO, RECOMPENSA Y GAME OVER
        return next_state, reward, game_over
    
    # CREAR UN METODO QUE PREDIGA EL PROXIMO ESTADO
    def predict_env(self,timestep):
        X_tv = np.array([[i for i in self.X_tv.iloc[timestep,:]]])
        X_mc = np.array([[i for i in self.X_mc.iloc[timestep,:]]])
        X_mc = np.reshape(X_mc, (X_mc.shape[0], X_mc.shape[1],1))
        X_tv = np.reshape(X_tv, (X_tv.shape[0], X_tv.shape[1],1))
        [[pred_tv]] = self.regressor_tv.predict(X_tv)
        [[pred_mc]] = self.regressor_mc.predict(X_mc)
        self.pred_mc_unscaled = pred_mc * (self.max_market_cap - self.min_market_cap) + self.min_market_cap
        self.pred_tv_unscaled = pred_tv * (self.max_trading_volume - self.min_trading_volume) + self.min_trading_volume
        self.pred_nvt = self.pred_mc_unscaled/self.pred_tv_unscaled
        pred_price = (self.pred_mc_unscaled/self.pred_nvt )/self.coin_supplyai
        
        scaled_supply = (self.coin_supplyai - self.min_supply)/(self.max_supply - self.min_supply)
        pred_state = np.matrix([pred_mc, pred_tv, scaled_supply]) # vector fila
        
        return pred_state, pred_price
    
    # CREAR UN MÉTODO QUE REINICIE EL ENTORNO
            # reiniciamos despues de cada epoch
    def reset(self):
        self.current_date = self.initial_date
        self.current_market_cap = self.initial_market_cap
        self.current_trading_volume = self.initial_trading_volume
        self.coin_supplyai = self.initial_supplyai
        self.current_nvt = self.initial_nvt
        self.price_ai = (self.current_market_cap/self.current_nvt)/ self.coin_supplyai 
        self.fair_price = 0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1


    # CREAR UN MÉTODO QUE NOS DE EN CUALQUIER INSTANTE EL ESTADO ACTUAL, LA ÚLTIMA RECOMPENSA Y EL VALOR DE GAME OVER
    def observe(self, timestep):
        self.current_trading_volume = data.iloc[timestep+60,3]
        self.current_market_cap = data.iloc[timestep+60,2]
        self.coin_supplyai = data.iloc[timestep+60,1]
        scaled_trading_volume = (self.current_trading_volume - self.min_trading_volume)/(self.max_trading_volume - self.min_trading_volume)
        scaled_market_cap = (self.current_market_cap - self.min_market_cap)/(self.max_market_cap - self.min_market_cap)
        scaled_supply = (self.coin_supplyai - self.min_supply)/(self.max_supply - self.min_supply)
        current_state = np.matrix([scaled_market_cap, scaled_trading_volume, scaled_supply]) # vector fila
                # esta matriz sera la capa de entrada
        #current_state = np.matrix([self.current_market_cap, self.current_trading_volume, self.coin_supplyai])
        
        return current_state, self.reward, self.game_over
        
        
        
        
        
        
        
        
        
    
