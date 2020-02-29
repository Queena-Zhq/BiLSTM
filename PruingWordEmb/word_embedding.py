##################################################################################
# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
# Author: 10461991 Qi Zhang
# Date: 20th Feb 2020
##################################################################################
import json

import sys
import torch
import torch.nn as nn
from torch import optim as op, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def read_training_words():
    labels = {}
    words = {}
    file = open("train.txt", "r" ,encoding = "utf-8")
    sentences = []
    torch.manual_seed(5)
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
                sen[num] = str.lower(sen[num])
                sen1.append(sen[num])
                if sen[num] not in words.keys():
                    words[sen[num]] = 1
                else:
                    words[sen[num]] = words[sen[num]] + 1
        sentence.append(sen1)
        sentence.append(label)
        # print(sentence)
        sentences.append(sentence)
    file.close()

    words["#UNK#"] = 100

    words_num = len(words)

    words_list = list(words.keys())
    words_tf = []
    for word in words.keys():
        if words[word]>1:
            words_tf.append(word)

    return words_list, words_tf


def glove_embedding(words_list,words_tf):
    vocabulary = []
    file = open("glove.small.txt", "r", encoding="utf-8")
    embeddings = {}
    result1 = {}
    result2 = {}

    # read every word vector from txt and stored in dict embeddings
    for lines in file.readlines():
        lines = lines.replace('\n', '')
        word = lines.split('	')
        embeddings[word[0]] = word[1]

    file.close()

    # prune from training words, if the word does not exist, store UNK vector
    for word in words_list:
        if word not in embeddings.keys():
            golve_vec = embeddings["#UNK#"]
        else:
            golve_vec = embeddings[word]
        # print(golve_vec)
        result1[word] = golve_vec

    for word in words_tf:
        if word not in embeddings.keys():
            golve_vec = embeddings["#UNK#"]
        else:
            golve_vec = embeddings[word]
        # print(golve_vec)
        result2[word] = golve_vec
    return result1, result2

def write_all_words(all_embedding):
    with open("MaxEmbedding.txt", "w", encoding="utf-8") as file1:
        for keys in all_embedding.keys():
            print(keys)
            file1.write(keys)
            file1.write("	")

            file1.write(all_embedding[keys])
            if keys == "#UNK#":
                continue
            else:
                file1.write("\n")
    file1.close()

def write_tf_words(tf_embedding):
    with open("MinEmbedding.txt", "w", encoding="utf-8") as file2:
        for keys in tf_embedding.keys():
            print(keys)
            file2.write(keys)
            file2.write("	")

            file2.write(tf_embedding[keys])
            if keys == "#UNK#":
                continue
            else:
                file2.write("\n")
    file2.close()
    
if __name__ == "__main__":
    
    word_list , word_tf = read_training_words()
    all_embedding , tf_embedding = glove_embedding(word_list, word_tf)
    write_all_words(all_embedding)
    write_tf_words(tf_embedding)
