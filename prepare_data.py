
# coding: utf-8

import numpy as np
import random
import copy

import os
import argparse

# noise type
ADD = 0
DELETE = 1
UPDATE = 2
# noise type distribution
p_ADD = 0.0
p_DELETE = 0.0
p_UPDATE = 1.0
noise_type_distribution = [p_ADD, p_DELETE, p_UPDATE]
sample_num = 100
test_num = 10
noise_rate = 0.1

RAW_DATA_PATH = "/Users/tianxin//git/plan_correction/raw_data"
TRAIN_DATA_PATH = "/Users/tianxin/git/plan_correction/data"


def add_noise(target_plan, noise_indices, vocab, vocab_size):
    noise_plan = copy.deepcopy(target_plan)
#     length = len(noise_plan)
#     noise_num = int(noise_rate * length)
#     noise_indices = random.sample(range(length), noise_num)
    delete_num = 0
    add_num = 0
    for index in noise_indices:
        op_index = index + add_num - delete_num
        # print("sampleed index is {0}".format(index))
        noise_type, target_index = generate_noise(vocab_size)
        if noise_type == UPDATE:
            print("update in op_index {0}\torigin:{1}\tnoise:{2}".
                  format(index, noise_plan[op_index], vocab[target_index]))
            noise_plan[op_index] = vocab[target_index]
        elif noise_type == DELETE:
            print("delete in op_index {0}\tdelete_word:{1}".format(index, noise_plan[op_index]))
            del noise_plan[op_index]
            delete_num += 1
        elif noise_type == ADD:
            noise_plan.insert(vocab[target_index], op_index)
            add_num += 1
    return noise_plan


def build_vocab(filename):
    with open(filename) as ifile:
        words =  [line.strip() for line in ifile.readlines()]
        actions = words[4:]
        vocab = dict((index, action) for (index, action) in enumerate(actions))
        return vocab


def generate_noise(vocab_size):
    noise_type = np.random.choice(range(3), 1, p=noise_type_distribution)[0]
    # print(noise_type)
    target_index = -1
    if noise_type in [UPDATE, ADD]:
        target_index = np.random.choice(vocab_size, 1)[0]
    return noise_type, target_index


def commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', help='specify the domain', choices=["block", "driverlog", "depots"])
    parser.add_argument('--raw_data', help='data_file to be processed')

    args = parser.parse_args()
    raw_path = os.path.join(RAW_DATA_PATH, args.domain)
    train_path = os.path.join(TRAIN_DATA_PATH, args.domain)

    raw_file = os.path.join(raw_path, args.raw_data)
    vocab_file = os.path.join(raw_path, "vocab40000.from")
    noise_train = os.path.join(train_path, "noise_train")
    target_train = os.path.join(train_path, "target_train")
    noise_test = os.path.join(train_path, "noise_test")
    target_test = os.path.join(train_path, "target_test")
    return raw_file, vocab_file, noise_train, target_train, noise_test, target_test


if __name__ == "__main__":

    raw_file, vocab_file, noise_train, target_train, noise_test, target_test = commandline_args()

    vocab = build_vocab(vocab_file)
    vocab_size = len(vocab)

    with open(raw_file) as raw_data:
        with open(noise_train, "w") as noise_train_file, open(target_train,"w") as target_train_file:
                with open(noise_test, "w") as noise_test_file, open(target_test, "w") as target_test_file:
                    for line_num,plan in enumerate(raw_data):
                        target_plan = plan.strip().split()
                        # print("{0} plan".format(line_num))
                        # generate noise indices
                        length = len(target_plan)
                        noise_num = int(noise_rate * length)
                        noise_indices = random.sample(range(length), noise_num)
                        # for index in noise_indices:
                        #     print("{0}\t{1}".format(index, target_plan[index]))
                        # generate training data
                        for i in range(sample_num):
                            print("{0} plan {1} train sample".format(line_num, i))
                            noise_plan = add_noise(target_plan, noise_indices, vocab, vocab_size)
                            noise_train_file.write(" ".join(noise_plan) + "\n")
                            target_train_file.write(plan)
                        # generate test data
                        for i in range(test_num):
                            print("{0} plan {1} test sample".format(line_num, i))
                            noise_plan = add_noise(target_plan, noise_indices, vocab, vocab_size)
                            noise_test_file.write(" ".join(noise_plan) + "\n")
                            target_test_file.write(plan)
