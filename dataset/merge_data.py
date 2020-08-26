"""
1 - Read TXT y CSV
2 - Conver to fragments (windows_size * 10)
3 - Filter, and pack as TXT with a JSON string
    in each line.
"""
import os
import re
import csv
import sys
import json
import random
import argparse

csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-input_fn',
    dest='input_fn',
    default='./dataset/',
    type=str,
    help='Path to dataset in json format. THIS MUST BE A LOCAL FILE.'
)
parser.add_argument(
    '-output_fn',
    dest='output_fn',
    default='./dataset.txt',
    type=str,
    help='Path to the resulting text file.'
)
parser.add_argument(
    '-seq_length',
    dest='seq_length',
    default=10240,
    type=int,
    help='Max sequence length',
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


with open(args.output_fn, "a") as write_file:
    for window in get_windows(args.seq_length * 6):
        json_str = json.dumps(window, separators=(',', ':'))
        json_str += "\n"
        write_file.write(json_str)
