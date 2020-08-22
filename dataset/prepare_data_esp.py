"""
1 - Read TXT y CSV
2 - Conver to fragments (windows_size * 10)
3 - Filter, tokenize and pack as .tfrecord
"""
import os
import re
import csv
import sys
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


class text_list:
    def __init__(self):
        self.texts = []

    def new_file(self):
        if len(self.texts) > 0 and self.texts[-1] == "":
            return
        self.texts.append("")

    def add_text(self, text):
        self.texts[-1] += text

    def length(self):
        length = 0
        for t in self.texts:
            length += len(t)
        return length

    def pop(self, length):
        assert length <= self.length()
        to_pop = []
        to_remove = 0
        le = 0
        if self.texts[-1] == "":
            self.texts = self.texts[:-1]
        for t in self.texts:
            if le + len(t) < length:
                to_pop.append(t)
                le += len(t)
                to_remove += 1
            else:
                to_pop.append(t[:length - le])
                self.texts = self.texts[to_remove:]
                self.texts[0] = self.texts[0][length - le:]
                return to_pop


# Quick Tests for the text_list class
tt = text_list()
tt.new_file()
tt.add_text("1234")
assert tt.length() == 4
tt.new_file()
assert tt.length() == 4
tt.add_text("56789")
assert tt.length() == 9
p = tt.pop(6)
assert p == ["1234", "56"]
assert tt.length() == 3
assert tt.texts == ["789"]
tt.new_file()
assert len(tt.texts) == 2
tt.new_file()
assert len(tt.texts) == 2
p = tt.pop(3)
assert p == ["789"]
assert len(tt.texts) == 1
tt.add_text("1234")
assert tt.length() == 4
assert len(tt.texts) == 1
print("Tests passed")
# End of texts

paths = []
for (dirpath, _, fnames) in os.walk(args.input_fn):
    for fname in fnames:
        if fname.startswith("."):
            continue
        if fname.endswith(".txt") or fname.endswith(".csv"):
            paths.append(os.path.join(dirpath, fname))


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def is_clean(text, segment_size=128):
    """
    Check if a string have enough information
    by calculating the ratio between letters and all the characters
    """
    if len(text) == 0 or text == "\n":
        return True
    end = segment_size
    while 1:
        text_to_test = text[end - segment_size:end]
        letters_count = len(re.findall('[a-z]|[A-Z]|á|é|í|ó|í|Á|É|Í|Ó|Ú|ñ|Ñ', text_to_test))
        if letters_count / len(text_to_test) < 0.4:
            print(len(text_to_test), letters_count, text_to_test)
            return False
        end += segment_size
        if end > len(text):
            break
    return True


def get_windows(window_size=1024):
    """
    Feed windows to encode, check if the text is clean
    """
    text = text_list()
    count = -1  # -1, because the loops adds +1 at the start
    for path in paths:
        count += 1
        if count % args.num_folds != args.fold:
            continue
        print("{}/{} of files {}".format(count, len(paths), path))
        text.new_file()
        with open(path, 'r', encoding='utf-8', errors='ignore') as file_:
            if path.endswith(".txt"):
                for line in file_:
                    if is_clean(line):
                        text.add_text(line)
                        if text.length() >= window_size:
                            yield text.pop(window_size)
                    else:
                        text.new_file()
            elif path.endswith(".csv"):
                reader = csv.reader(file_, delimiter=',')
                for row in reader:
                    if is_clean(row[0]):
                        text.add_text(row[0])
                        if text.length() >= window_size:
                            yield text.pop(window_size)
                    else:
                        text.new_file()


tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file, dropout=0.1)

train_file = args.base_fn + 'train_windowed_{}.tfrecord'.format(args.fold)
train_writer = tf.io.TFRecordWriter(train_file)

for window in get_windows(args.max_seq_length * 6):
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
