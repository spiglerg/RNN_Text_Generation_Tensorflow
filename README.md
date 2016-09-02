# Text generation using a RNN

Text generation using a RNN (LSTM) using Tensorflow.


## Usage
To train the model:
1. Set the textfile you want to use to train the network in the code ("with open('data/shakespeare.txt', 'r') as f:")

2. 
    $ python rnn_tf.py

3. Run the network with the starting prefix you want to use to generate text:
    $ python rnn_tf.py saved/model.ckpt "The "


