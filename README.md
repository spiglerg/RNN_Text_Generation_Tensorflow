# Text generation using a RNN

Text generation using a RNN (LSTM) using Tensorflow.


## Usage

To train the model you can set the textfile you want to use to train the network by using command line options:

Run the network in train mode:

  $ python rnn_tf.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --mode=train

Run the network to generate text:

  $ python rnn_tf.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --test_prefix="The " --mode=talk
