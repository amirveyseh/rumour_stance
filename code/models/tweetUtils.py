import os
from os import walk
import json
from pprint import pprint


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_files(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return f;


def json_parse(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data;