##################################################################################
# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
# Author: 10461991 Qi Zhang
# Date: 27th Feb 2020
##################################################################################

import configparser
import sys
import torch
import torch.nn as nn
from torch import optim as op, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy import *


class BiLSTMclassfier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size+1, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim,
                            embedding_dim,
                            num_layers=n_layers,
                            bidirectional = True,
                            dropout = dropout)


        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim* 2)
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim* 2, output_dim)


    def forward(self, embedds, hx=None):

        # initial hidden layer h0 as 
        lstm_out1, (hidden1, cell1) = self.lstm(embedds.view(len(embedds), 1, -1))

        back = lstm_out1[0].view(2,1,-1)
        forward = lstm_out1[len(embedds)-1].view(2,1,-1)

        back_out = back[1]
        forward_out = forward[0]
        hidden = torch.cat((forward_out,back_out), dim = 0)
        dense_outputs = self.fc1(hidden.view(1,-1))
        # print(dense_outputs)
        relu_vec = self.relu(dense_outputs)
        # print(relu_vec)
        out = self.fc2(relu_vec)
        # print(out)
        outputs = F.log_softmax(out.view(1, -1), dim=1)
        # print(outputs)
        return outputs


def test_BiLSTM(Bi_config,PATH_config,Train_config):
    words_list , label_list , _train_sentences = read_words(PATH_config["path_train"], PATH_config , Train_config )
    _words_list , _label_list , test_sentences = read_words(PATH_config["path_test"], PATH_config , Train_config )
    predicted_label = []
    true_label = []

    size_of_vocab = len(words_list)
    embedding_dim = int(Bi_config["embedding_dim"])
    num_hidden_nodes = int(Bi_config["num_hidden_nodes"])
    num_output_nodes = int(Bi_config["num_output_nodes"])
    num_layers = int(Bi_config["num_layers"])
    bidirection = bool(Bi_config["bidirection"])
    dropout = float(Bi_config["dropout"])

    
    model = BiLSTMclassfier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirection, dropout)
    # load model parameters
    model.load_state_dict(torch.load(PATH_config["path_model"]))
    print("Model has been loaded.")
    
    words_embedding = torch.load(PATH_config["final_embedding"])
    print("Embedding has been loaded.")

    # test start
    print("Test start! ")
    file = open(PATH_config["path_output"], "w", encoding = "utf-8")
    file.write("Predicted label / True label\n")
    train_acc = 0
    with torch.no_grad():
        for q in test_sentences:
            sentence = q[0]
            label = q[1]
            true_label.append(label)
            emb = []
            for num in sentence:
                if (num == '?') or (num == '``') or (num == '\'s') or (num == '\'\'') or (num == '.'):
                    continue
                else:
                    if num in words_list:
                        emb.append(words_list.index(num))
                    else:
                        emb.append(words_list.index("#UNK#"))
            emb = torch.tensor(emb, dtype=torch.long)
            sen_emb1 = words_embedding(emb)
            log_probs = model(sen_emb1)
            predicted_label.append(label_list[log_probs.argmax(1)])
            file.write(label_list[log_probs.argmax(1)])
            file.write(" / ")
            file.write(label)
            file.write("\n")
            if log_probs.argmax(1) == label_list.index(label):
                train_acc += 1
    train_acc /= len(test_sentences)

    con_matrix, precision_mean, recall_mean, f_score = evaluate(label_list, predicted_label, true_label, len(test_sentences))
    print("Confusion matrix: ")
    print(con_matrix)
    print("Precision: %f " % precision_mean)
    print("Recall: %f " % recall_mean)
    print("F_score: %f " % f_score)
    print("Accuracy: %f " % train_acc)
    file.write("\n")
    file.write("Total accuracy : ")
    file.write(str(train_acc))
    file.close()

    print("Test results have been writen in " +  PATH_config["path_output"])

    np.savetxt(PATH_config["confusion_matrix"], con_matrix ,fmt='%d')
    print("Confusion_Matrix has been writen in " +  PATH_config["confusion_matrix"])
    

def train_BiLSTM(Bi_config,PATH_config,Train_config):

    words_list , label_list , train_sentences = read_words(PATH_config["path_train"],PATH_config, Train_config)
    _words_list , _label_list , dev_sentences = read_words(PATH_config["path_dev"],PATH_config, Train_config)

    size_of_vocab = len(words_list)
    embedding_dim = int(Bi_config["embedding_dim"])
    num_hidden_nodes = int(Bi_config["num_hidden_nodes"])
    num_output_nodes = int(Bi_config["num_output_nodes"])
    num_layers = int(Bi_config["num_layers"])
    bidirection = bool(Bi_config["bidirection"])
    dropout = float(Bi_config["dropout"])

    lr = float(Train_config["lr"])
    iteration = int(Train_config["epoch"])
    randomly = Train_config["randomly_embedding"]
    early_stop = float(Train_config["early_stop"])
    lowercase = bool(Train_config["lowercase"])

    if randomly:
        print("Start randomly initial words embedding !")
        words_embedding = nn.Embedding(size_of_vocab,embedding_dim)
    else:
        print("Start glove Embedding ")
        vocabulary = glove_embedding(words_list,PATH_config["path_embedding"])
        # change data type from float to tensor
        vocabulary1 = torch.FloatTensor(vocabulary)
        # use from_pretrained to generate words embedding
        words_embedding = nn.Embedding.from_pretrained(vocabulary1, freeze = bool(Bi_config["freeze"]))

    model = BiLSTMclassfier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirection, dropout)
                   
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(iteration):
        print(epoch)
        train_data = []
        for question in train_sentences:
            target_list = []
            sen = question[0]
            label = question[1]
            model.zero_grad()

            emb = []
            emb_list = []
            for num in sen:
                emb.append(words_list.index(num))
            target = label_list.index(label)
            target_list.append(target)
            emb_list.append(emb)
            target1 = torch.LongTensor(target_list)

            emb = torch.tensor(emb,dtype=torch.long)
            sen_emb = words_embedding(emb)
            # print(sen_emb)
            log_probs = model(sen_emb)

            loss = loss_function(log_probs, target1)
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
        print("train finished, validation start! \n")
        acc = BiLSTM_validate(model , words_list, label_list, words_embedding, dev_sentences)

        if early_stop <= acc:
            print("The acc has meet the early stop condition, Train stop !")
            break
        else:
            continue
    torch.save(words_embedding, PATH_config["final_embedding"])
    # store_embedding(words_list, words_embedding, PATH_config["final_embedding"])
    print("Embedding has been save in " + PATH_config["final_embedding"])
    torch.save(model.state_dict(), PATH_config["path_model"])
    print("Model has been saved in " + PATH_config["path_model"])


def BiLSTM_validate(model , words_list, label_list, words_embedding, dev_sentences):
    train_acc = 0
    true_label = []
    predicted_label = []

    with torch.no_grad():
        for q in dev_sentences:
            sentence = q[0]
            label = q[1]
            true_label.append(label)
            emb = []
            for num in sentence:
                if (num == '?') or (num == '``') or (num == '\'s') or (num == '\'\'') or (num == '.'):
                    continue
                else:
                    if num in words_list:
                        emb.append(words_list.index(num))
                    else:
                        emb.append(words_list.index("#UNK#"))
            emb = torch.tensor(emb, dtype=torch.long)
            sen_emb1 = words_embedding(emb)
            log_probs = model(sen_emb1)
            predicted_label.append(label_list[log_probs.argmax(1)])
            if log_probs.argmax(1) == label_list.index(label):
                train_acc += 1
    train_acc /= len(dev_sentences)

    con_matrix, precision_mean, recall_mean, f_score=evaluate(label_list, predicted_label, true_label, len(dev_sentences))
    print("Precision: %f " % precision_mean)
    print("Recall: %f " % recall_mean)
    print("F_score: %f " % f_score)
    print("Accuracy: %f " % train_acc)
    return train_acc

# get precision recall and F1
def evaluate(label_list, predicted_label, true_label, num):
    con_matrix = np.zeros([len(label_list), len(label_list)])
    # get the confusion matrix
    for n in range(num):
        actual = true_label[n]
        predicted = predicted_label[n]
        actual_index = label_list.index(actual)
        pre_index = label_list.index(predicted)
        con_matrix[actual_index][pre_index] = con_matrix[actual_index][pre_index] + 1

    # get precision, recall, f-score
    label_num = []
    predicted_num = []
    correct_num = []
    precision_list = []
    recall_list = []
    n = 0
    m = 0
    for i in range(len(label_list)):
        for j in range(len(label_list)):
            n = n + con_matrix[i][j]
            m = m + con_matrix[j][i]
        label_num.append(n)
        predicted_num.append(m)
        correct_num.append(con_matrix[i][i])
        n = 0
        m = 0
        if (predicted_num[i] == 0) or (label_num[i] == 0) or (correct_num[i] == 0):
            precision = 0
            recall = 0
            precision_list.append(precision)
            recall_list.append(recall)
        else:
            precision = correct_num[i] / predicted_num[i]
            recall = correct_num[i] / label_num[i]
            precision_list.append(precision)
            recall_list.append(recall)
    precision_mean = mean(precision_list)
    recall_mean = mean(recall_list)
    f_score = (2 * recall_mean * precision_mean) / (recall_mean + precision_mean)

    return con_matrix, precision_mean, recall_mean, f_score


def glove_embedding(words_list,PATH):
    vocabulary = []
    file = open(PATH, "r", encoding="utf-8")
    embeddings = {}
    UNK_vec = [float(0)]*300
    # print(UNK_vec)
    # read every word vector from txt and stored in dict embeddings
    for lines in file.readlines():
        vec = []
        word = []
        vec_float = []
        lines = lines.replace('\n', '')
        word = lines.split('	')
        vec = word[1].split(' ')
        for num in range(len(vec)):
            vec_float.append(float(vec[num]))
        embeddings[word[0]] = vec_float

    # prune from training words, if the word does not exist, store UNK vector
    for word in words_list:
        if word not in embeddings.keys():
            golve_vec = UNK_vec
            # golve_vec = embeddings["#UNK#"]
        else:
            golve_vec = embeddings[word]
        # print(golve_vec)
        vocabulary.append(golve_vec)
        # print(vocabulary)
    return vocabulary

def get_Biconfig(PATH):
    config = configparser.ConfigParser()
    config.read(PATH)

    BiLSTM_config = {}
    PATH_config = {}
    Train_config = {}

    for cfg in config["BiLSTM"]:
        if cfg == "embedding_dim":
            BiLSTM_config[cfg] = config["BiLSTM"][cfg]
        elif cfg == "num_hidden_nodes":
            BiLSTM_config[cfg] = config["BiLSTM"][cfg]
        elif cfg == "num_output_nodes":
            BiLSTM_config[cfg] = config["BiLSTM"][cfg]
        elif cfg == "num_layers":
            BiLSTM_config[cfg] = config["BiLSTM"][cfg]
        elif cfg == "bidirection":
            BiLSTM_config[cfg] = config.getboolean("BiLSTM",cfg)
        elif cfg == "dropout":
            BiLSTM_config[cfg] = config["BiLSTM"][cfg]
        elif cfg == "freeze":
            BiLSTM_config[cfg] = config.getboolean("BiLSTM",cfg)
        else:
            print("Config {k} does not ues. ".format(k = cfg))
    
    for cfg in config["PATH"]:
        if cfg == "path_train":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "path_dev":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "path_test":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "path_model":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "path_embedding":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "final_embedding":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "confusion_matrix":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "path_output":
            PATH_config[cfg] = config["PATH"][cfg]
        elif cfg == "stop_words":
            PATH_config[cfg] = config["PATH"][cfg]
        else:
            print("Config {k} does not ues. ".format(k = cfg))

    for cfg in config["Train"]:
        if cfg == "epoch":
            Train_config[cfg] = config["Train"][cfg]
        elif cfg == "early_stop":
            Train_config[cfg] = config["Train"][cfg]
        elif cfg == "lowercase":
            Train_config[cfg] = config.getboolean("Train",cfg)
        elif cfg == "randomly_embedding":
            Train_config[cfg] = config.getboolean("Train",cfg)
        elif cfg == "lr":
            Train_config[cfg] = config["Train"][cfg]
        elif cfg == "remove_stopwords":
            Train_config[cfg] = config.getboolean("Train",cfg)
        else:
            print("Config {k} does not ues. ".format(k = cfg))
    return BiLSTM_config , PATH_config, Train_config


def read_words(PATH, PATH_config, Train_config):
    labels = {}
    words = {}
    stop_words = []
    stop = Train_config["remove_stopwords"]
    if stop:
        with open(PATH_config["stop_words"], "r", encoding = "utf-8") as file1:
            for word in file1.readlines():
                word = word.replace('\n', '')
                stop_words.append(word)
        file1.close()
    
    file = open(PATH, "r" ,encoding = "utf-8")
    sentences = []
    for lines in file.readlines():
        sentence = []
        lines = lines.replace('\n', '')
        list_sen = lines.split(' ')
        label = list_sen[0]
        if label not in labels.keys():
            labels[label] = 1
        sen = list_sen[1:]
        sen1 = []
        for num in range(0,len(sen)):
            if (sen[num] == '?') or (sen[num] == '``') or (sen[num] == '\'s') or (sen[num] == '\'\'') or (sen[num] == '.'):
                continue
            else:
                if sen[num] not in stop_words:
                    sen[num] = str.lower(sen[num])
                    sen1.append(sen[num])
                    if sen[num] not in words.keys():
                        words[sen[num]] = 1
                    else:
                        words[sen[num]] = words[sen[num]] + 1
                else:
                    continue
        sentence.append(sen1)
        sentence.append(label)
        # print(sentence)
        sentences.append(sentence)
    file.close()

    words["#UNK#"] = 100

    label_num = len(labels)
    words_num = len(words)

    label_list = list(labels.keys())
    words_list = list(words.keys())
    return words_list , label_list , sentences


# for train :   python3 question_classifier.py train -config BiLSTM.ini
# for train :   python3 question_classifier.py train -config BOW.ini
# for test  :   python3 question_classifier.py test -config BiLSTM.ini
# for test  :   python3 question_classifier.py test -config BOW.ini
if __name__ == '__main__':

    # get input parameters
    config = sys.argv
    torch.manual_seed(5)
    # initial algorithm, if algorithm == 1 : BOW; if algorithm == 2 : BiLSTM;
    algorithm = 0

    # get config parameters
    pam = config[3].split(".")
    if pam[0] == "BiLSTM":
        algorithm = 2
        Bi_config , PATH_config , Train_config = get_Biconfig(config[3])
    elif pam[0] == "BOW":
        algorithm = 1
        pass
    else:
        print("No such config file, please check the file name.")

    # train or test
    if config[1] == "train" :
        if algorithm == 1:
            pass
        elif algorithm == 2:
            train_BiLSTM(Bi_config,PATH_config,Train_config)
        else:
            print("No available Algorithm. ")
    elif config[1] == "test" :
        if algorithm == 1:
            pass
        elif algorithm ==2 :
            test_BiLSTM(Bi_config,PATH_config,Train_config)
        else:
            print("No available Algorithm. ")
    else:
        print("The second parameter doesn`t match.")
