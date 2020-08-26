"""
1 - Read TXT file with JSON string in each line
2 - Read randomly given weights to each file
3 - Tokenize and pack as .tfrecord files
"""
import os
import re
import csv
import sys
import json
import random
import argparse
import tensorflow as tf

from tokenizers import ByteLevelBPETokenizer

csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='Which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=1337,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '-base_fn',
    dest='base_fn',
    default='news2016zh_',
    type=str,
    help='We will output files that are like {base_fn}_{n}.tfrecord for n in 0, ..., 1023'
)

parser.add_argument(
    '-input_fn',
    dest='input_fn',
    default='./dataset/',
    type=str,
    help='Path to dataset in json format. THIS MUST BE A LOCAL FILE.'
)
parser.add_argument(
    '-max_seq_length',
    dest='max_seq_length',
    default=10240,
    type=int,
    help='Max sequence length',
)

parser.add_argument(
    '-vocab_file',
    dest='vocab_file',
    default='../vocabs/spanish/vocab.json',
    type=str,
    help='Tokenizer vocab.json file.'
)
parser.add_argument(
    '-merges_file',
    dest='merges_file',
    default='../vocabs/spanish/merges.txt',
    type=str,
    help='Tokenizer merges.txt file.'
)

args = parser.parse_args()


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file, dropout=0.1)

biggest_file_line_count = int(args.input_fn.split(",")[0])
paths = []

for path_repeat in args.input_fn.split(",")[1:]:
    splitted_path = path_repeat.split(":")
    path_dict = {
        "path": splitted_path[0],
        "probability": float(splitted_path[1]),
        "file_size": os.stat(splitted_path[0]).st_size,
        "file": open(splitted_path[0])
    }
    paths.append(path_dict)


def get_windows():
    for n in range(round(biggest_file_line_count / args.num_folds)):
        for path in paths:  # For each path
            if random.random() < path["probability"]:  # Account for sampling frequency
                random_point = random.randint(0, path["file_size"])
                path["file"].seek(random_point)
                path["file"].readline()  # skip this line to clear the partial line
                line_str = path["file"].readline()
                try:
                    json_obj = json.loads(line_str)
                except json.decoder.JSONDecodeError:
                    print("Json Error")
                    print(line_str)
                yield json_obj


train_file = args.base_fn + 'train_windowed_{}.tfrecord'.format(args.fold)
train_writer = tf.io.TFRecordWriter(train_file)


for window in get_windows():
    encoded_string = []
    for text in window:
        encoded_string.append(0)  # Appending <|endoftext|> token
        text = re.sub(r'(\W|\w)\n', r'\1', text)  # Clean line breaks
        encoded_string += tokenizer.encode(text).ids
    if len(encoded_string) < args.max_seq_length:
        print("Too short")
        continue

    if random.random() > 0.5:
        encoded_string = encoded_string[:args.max_seq_length]
    else:
        encoded_string = encoded_string[len(encoded_string) - args.max_seq_length:]
    assert len(encoded_string) == args.max_seq_length, "Window length {} is not equal to desired length {}".format(len(encoded_string), args.max_seq_length)

    features = {"input_ids": create_int_feature(encoded_string)}
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    train_writer.write(tf_example.SerializeToString())

train_writer.close()
