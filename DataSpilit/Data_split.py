##################################################################################
# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
# Author: Qi Zhang
# Date: 19th Feb 2020
##################################################################################

import random

def split(all_list, ratio=0.8):
    num = len(all_list)
    offset = int(num * ratio)
    random.shuffle(all_list)
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test


def write_split(file, train, test):
    infile = open(file, 'r', encoding='utf-8')
    tainfile = open(train, 'w', encoding='utf-8')
    testfile = open(test, 'w', encoding='utf-8')
    li = []
    for datas in infile.readlines():
        datas = datas.replace('\n', '')
        li.append(datas)
    traindatas, testdatas = split(li, ratio=0.9)
    for traindata in traindatas:
        tainfile.write(traindata + '\n')
    for testdata in testdatas:
        testfile.write(testdata + '\n')
    infile.close()
    tainfile.close()
    testfile.close()


if __name__ == "__main__":
    write_split('CW1_data.txt', 'train.txt', 'dev.txt')
