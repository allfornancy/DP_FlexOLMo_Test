import json
import math
import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import psutil
import smart_open
from tqdm import tqdm

from flexolmo.path_utils import glob_path

pid = os.getpid()
python_process = psutil.Process(pid)


def print_mem_use():
    memUse = python_process.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory use: %.1fGB" % memUse)


def read_file(path, p, skip_parse=False):
    np.random.seed(2025)
    docs = []
    with smart_open.open(path, "r") as f:
        for line in f:
            if p >= 1.0 or np.random.random() < p:
                if skip_parse:
                    docs.append(line)
                else:
                    dp = json.loads(line)
                    doc = dp["text"]
                    docs.append(doc)
    return path, docs


def read_reddit_file(path, p):
    np.random.seed(2025)
    docs = []
    with smart_open.open(path, "r") as f:
        for line in f:
            if p == 1.0 or np.random.random() < p:
                dp = json.loads(line)
                for c in dp["response"]["body"]["choices"]:
                    doc = c["message"]["content"]
                    assert isinstance(doc, str)
                    docs.append(doc)
    return path, docs


def load_data(domain, data_path, p=1.0, debug=False, parallel=True, shard_id=None, n_shards=None):

    paths = get_paths(data_path)

    if debug:
        paths = paths[:5]
    elif shard_id is not None and n_shards is not None:
        n_files_per_shard = math.ceil(len(paths) / n_shards)
        paths = paths[shard_id * n_files_per_shard : (shard_id + 1) * n_files_per_shard]
    else:
        assert shard_id is None and n_shards is None

    data = []
    with smart_open.open(paths[0], "r") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Start loading {len(paths)} paths for domain {domain}")
    start_time = time.time()
    _read_file = partial(read_reddit_file, p=p) if domain == "reddit" else partial(read_file, p=p)

    if parallel:
        all_docs_ = {}
        with Pool() as p, tqdm(total=len(paths), smoothing=0) as progress:
            for path, docs in p.imap_unordered(_read_file, paths):
                all_docs_[path] = docs
                progress.update(1)

        # this is to make sure docs are deterministic
        all_docs = [
            doc for _, docs in sorted(all_docs_.items(), key=lambda x: x[0]) for doc in docs
        ]

    else:
        all_docs = []
        for path in tqdm(paths):
            _, docs = _read_file(path)
            all_docs += docs

    print(f"Loaded {len(all_docs)} docs for {time.time() - start_time}s")
    print_mem_use()
    return all_docs


def get_paths(path: str):
    return list(glob_path(path))


def count_lines_file(path):
    cnt = 0
    with smart_open.open(path, "r") as f:
        for line in f:
            cnt += 1
    return cnt


def save_file(pair):
    data, save_path = pair
    with smart_open.open(save_path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def read_and_save(path, from_keyword, to_keyword):
    assert from_keyword in path
    save_path = path.replace(from_keyword, to_keyword)

    def find_min_index_with_sum(arr, target_sum=512):
        current_sum = 0
        for i, num in enumerate(arr):
            current_sum += num
            if current_sum >= target_sum:
                return i
        return None

    with smart_open.open(path, "r") as f, smart_open.open(save_path, "w") as f_w:
        for line in f:
            text = json.loads(line)["text"]
            paragraphs = text.split("\n\n")
            paragraphs = [par for par in paragraphs if len(par.strip()) > 0]
            n_words = [len(par.split()) for par in paragraphs]
            i = find_min_index_with_sum(n_words)
            if i is not None:
                text = "\n\n".join(paragraphs[:i])
            f_w.write(json.dumps({"text": text}) + "\n")


def count_words(path):
    line_cnt = 0
    word_cnt = 0
    with smart_open.open(path, "r") as f:
        for line in f:
            line_cnt += 1
            word_cnt += len(json.loads(line)["text"].split())
    return path, line_cnt, word_cnt


def sample_data(path, p=0.01):
    # if we train third-stage for 5B tokens, we only need about 1.25B tokens
    lines = []
    with smart_open.open(path, "r") as f:
        for line in f:
            if np.random.random() < p:
                lines.append(line)
    return path, lines
