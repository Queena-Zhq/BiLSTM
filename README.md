# BiLSTM
DataSpilit folder is original training data(5500) and test data(500),the training data has beed seperated randomly into 9:1,9 is used for training and 1 is used for evaluation.

PruingWordEmbedding folder contains the whole words embedding from glove algorithm with dimension 300, filtering the needed words from training data and storing into MaxEmbedding with all training words and MinEmbedding minus those words just appearing one time.

BiLSTM folder contains the main LSTM file and trained model. The parameters for model can be changed in config file : BiLSTM.ini

The program can be run by followinf command:

For train : python3 question_classifier.py train -config BiLSTM.ini

For test  : python3 question_classifier.py test -config BiLSTM.ini
