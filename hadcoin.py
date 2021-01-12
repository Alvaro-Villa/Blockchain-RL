# Module 2 - Create a Cryptocurrency

# To be installed:
# Flask==0.12.2: pip install Flask==0.12.2
# Postman HTTP Client: https://www.getpostman.com/
# requests==2.18.4: pip install requests==2.18.4

# Importing the libraries
import datetime
import hashlib
import json
from flask import Flask, jsonify, request
import requests # utilizaremos paar conectar nodos
from uuid import uuid4 # crear un addres para cada nodo
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
import environment

env = environment.Environment(optimal_price = (0.90, 1.1))
direction_boundary = 1

train = False


env.train = train
current_state, _, _ = env.observe()
# Part 1 - Building a Blockchain

class Blockchain:

    def __init__(self):
        self.chain = []
        self.transactions = [] # objeto con transacciones que añadimos al block
        self.transactions_fees = []
        self.create_block(proof = 1, previous_hash = '0') # es importante que que esta funcion por debajo de transactions
        self.nodes = set() # conjunto vacio
        self.model = tf.keras.models.load_model("model_nvt.h5")
    
    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash,
                 'transactions': self.transactions} # Añadimos transacciones
        self.transactions = [] # borramos transactions
        self.chain.append(block)
        return block
    def create_reward_block(self, proof, previous_hash,node_address):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash,
                 'transactions': self.transactions_fees} # Añadimos transacciones
        self.transactions_fees = [] # borramos transactions
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
        return new_proof
    
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys = True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
    
    def is_chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
        return True
    
    def add_transaction(self, sender, receiver, amount, fee, miner ): # key elements of transactions, quian, a quien y cuanto
        self.transactions.append({'sender': sender,        # añadimos la transaccion a transactions
                                  'receiver': receiver,
                                  'amount': amount,
                                  'fee': fee})
        previous_block = self.get_previous_block()
        self.transactions_fees.append({'sender': sender,
                                       'miner':miner,
                                       'fee':fee})
        return previous_block['index'] + 1
    

    def add_node(self, address): # address es el numero del puerto
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc) # .netloc es un atributo generado por el parse
    
    def replace_chain(self): # cambia cualquier otra cadena que se duplique
        network = self.nodes
        longest_chain = None
        max_length = len(self.chain)
        for node in network: # con el bucle encontramos el node con el longest chain
            response = requests.get(f'http://{node}/get_chain') # requests obtenemos la chain y el leght
            # los nodos se diferencia por el port
            # f hacemos referencia al node fuera del str
            if response.status_code == 200: # comporbamos que todo OK
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.is_chain_valid(chain): # chequeamos que la chain sea valida
                    max_length = length
                    longest_chain = chain
        if longest_chain:
            self.chain = longest_chain
            return True
        return False
    #current_state = np.matrix([[9.54774713e-09, 8.71618556e-08, 0.00000000e+00]])
    def reward_distributor(self, current_state):
        current_state, _, _ = env.observe()
        q_values = self.model.predict([current_state])
        action = np.argmax(q_values[0])
                
        
        if (env.price_ai <= direction_boundary):
            direction = -1
        else:
            direction = 1
        if action == 0:
            coin_var = abs((env.price_ai * env.coin_supplyai) - env.coin_supplyai) * 0.1
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
        
        total_fee = 0
        for i in range(0,len(self.transactions_fees)):
            total_fee += self.transactions_fees[i]['fee']
    
        if direction == 1:
            fee_rate = (coin_var + total_fee)/total_fee
        else:
            fee_rate = (total_fee-coin_var)/total_fee
        
        for i in range(0,len(self.transactions_fees)):
            self.transactions_fees[i]['fee']=self.transactions_fees[i]['fee']*fee_rate
        return fee_rate
        
    def reward_demo(self):
        fee_rate = 0.8
        for i in range(0,len(self.transactions_fees)):
            self.transactions_fees[i]['fee'] = self.transactions_fees[i]['fee']*fee_rate
        

# Part 2 - Mining our Blockchain

# Creating a Web App
app = Flask(__name__)

# Creating an address for the node on Port 5000
node_address = str(uuid4()).replace('-', '') # uuid4 crea el address aleatorio unico que se asigna al nodo
                                             # Usamos replace para eliminar -
# Creating a Blockchain
blockchain = Blockchain()

# Mining a new block
@app.route('/mine_block', methods = ['GET'])
def mine_block():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    blockchain.add_transaction(sender = 'jose', receiver = 'Álvaro', amount = 1, fee = 1, miner= node_address)
    block = blockchain.create_block(proof, previous_hash)
    response = {'message': 'Congratulations, you just mined a block!',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash'],
                'transactions': block['transactions']}
    return jsonify(response), 200

# Mining reward block
@app.route('/mine_block_reward', methods = ['GET'])
def mine_block_reward():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    #blockchain.add_transaction(sender = 'node', receiver = 'Hadelin', amount = 1)
    blockchain.reward_distributor(current_state)
    block = blockchain.create_reward_block(proof, previous_hash, node_address)
    response = {'message': 'Congratulations, daily rewards have just been mined in this block!',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash'],
                'transactions': block['transactions']}
    return jsonify(response), 200

# Getting the full Blockchain
@app.route('/get_chain', methods = ['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return jsonify(response), 200

# Checking if the Blockchain is valid
@app.route('/is_valid', methods = ['GET'])
def is_valid():
    is_valid = blockchain.is_chain_valid(blockchain.chain)
    if is_valid:
        response = {'message': 'All good. The Blockchain is valid.'}
    else:
        response = {'message': 'Houston, we have a problem. The Blockchain is not valid.'}
    return jsonify(response), 200

# Adding a new transaction to the Blockchain
@app.route('/add_transaction', methods = ['POST'])
def add_transaction():
    json = request.get_json() # esto cogera el json y lo subira
    transaction_keys = ['sender', 'receiver', 'amount','fee', 'miner']
    if not all(key in json for key in transaction_keys): # si todas las key en el transaction_key no estan en json
        return 'Some elements of the transaction are missing', 400
    index = blockchain.add_transaction(json['sender'], json['receiver'], json['amount'], json['fee'], json['miner'])
    response = {'message': f'This transaction will be added to Block {index}'} # f' input the variable {}
    return jsonify(response), 201 # 201 todo ha ido bien Created

# Part 3 - Decentralizing our Blockchain

# Connecting new nodes
@app.route('/connect_node', methods = ['POST']) # POST la usamos para añadir info
def connect_node(): # crearemos un nuevo nodo y lo registraremos
    json = request.get_json()
    nodes = json.get('nodes') # obtenemos todos los nodos, nos devolvera el address de cada nodo
    if nodes is None:
        return "No node", 400
    for node in nodes:
        blockchain.add_node(node)
    response = {'message': 'All the nodes are now connected. The Hadcoin Blockchain now contains the following nodes:',
                'total_nodes': list(blockchain.nodes)}
    return jsonify(response), 201

# Replacing the chain by the longest chain if needed
@app.route('/replace_chain', methods = ['GET'])
def replace_chain():
    is_chain_replaced = blockchain.replace_chain()
    if is_chain_replaced:
        response = {'message': 'The nodes had different chains so the chain was replaced by the longest one.',
                    'new_chain': blockchain.chain}
    else:
        response = {'message': 'All good. The chain is the largest one.',
                    'actual_chain': blockchain.chain}
    return jsonify(response), 200

# Running the app
app.run(host = '0.0.0.0', port = 5000)
