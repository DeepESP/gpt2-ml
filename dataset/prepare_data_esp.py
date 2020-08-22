"""
1 - Read TXT y CSV
2 - Conver to fragments (windows_size * 10)
3 - Filter, tokenize and pack as .tfrecord
"""
import os
import re
import csv
import sys
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
    default='vocabs/spanish/vocab.json',
    type=str,
    help='Tokenizer vocab.json file.'
)
parser.add_argument(
    '-merges_file',
    dest='merges_file',
    default='vocabs/spanish/merges.txt',
    type=str,
    help='Tokenizer merges.txt file.'
)

args = parser.parse_args()

paths = []
for (dirpath, _, fnames) in os.walk(args.input_fn):
    for fname in fnames:
        if fname.startswith("."):
            continue
        if fname.endswith(".txt") or fname.endswith(".csv"):
            paths.append(os.path.join(dirpath, fname))

thread = 0


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def get_fragment(window_size=2048):
    text = ""
    count = -1  # -1, because the loops adds +1 at the start
    for path in paths:
        count += 1
        print("{}/{} of files {}".format(count, len(paths), path))
        if count % args.num_folds != args.fold:
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as file_:
            if path.endswith(".txt"):
                for line in file_:
                    text += line
                    if len(text) > window_size:
                        window_text = text[:window_size]
                        text = text[window_size:]
                        yield window_text
            elif path.endswith(".csv"):
                reader = csv.reader(file_, delimiter=',')
                for row in reader:
                    text += row[0]
                    if len(text) > window_size:
                        window_text = text[:window_size]
                        text = text[window_size:]
                        yield window_text


def is_clean(text, segment_size=128):
    end = segment_size
    while 1:
        text_to_test = text[end - segment_size:end]
        letters_count = len(re.findall('[a-z]|[A-Z]', text_to_test))
        if letters_count < 80:
            print(letters_count, text_to_test)
            return False
        end += segment_size
        if end > len(text):
            break
    return True


def get_windows(window_size=1024):
    for text in get_fragment(window_size * 2):
        while 1:
            window_text = text[:window_size]
            text = text[window_size:]
            if is_clean(window_text):
                yield window_text
            else:
                print("SKIP!")
                pass
            if window_size > len(text):
                break


tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merge_file, dropout=0.1)


total_written = 0
train_writer = None
for window in get_windows(args.max_seq_length * 6):
    if total_written == 0 or total_written % 1000000 == 0:
        if train_writer is not None:
            train_writer.close()
        train_file = args.base_fn + 'train_windowed_{}_{:04d}.tfrecord'.format(thread, total_written)
        train_writer = tf.io.TFRecordWriter(train_file)

    window = re.sub(r'(\W|\w)\n', r'\1', window)
    output = tokenizer.encode(window)
    if len(output.ids) < args.max_seq_length:
        print("Too short")
        continue
    features = {
        "input_ids": create_int_feature(output.ids[:args.max_seq_length])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    train_writer.write(tf_example.SerializeToString())
    total_written += 1

train_writer.close()
