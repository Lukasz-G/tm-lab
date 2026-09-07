"""Prepare IMDb for TMCore, keeping the vocabulary.

Mirrors FuzzyPatternTM's examples/IMDb/prepare_dataset.py — same n-gram range, same chi2 feature
selection — with two changes that matter here:

  * **The feature names are written out.** Upstream emits only bits and a label, which is enough to
    train and impossible to read. Mapping literal 4,412 back to the phrase "one of the worst" is the
    entire point of running IMDb rather than MNIST, so the vocabulary is the deliverable.
  * **No keras.** `keras.datasets.imdb` only downloads two files and applies a documented index
    shift; that is reproduced here directly, so the whole pipeline needs numpy and scikit-learn.

Output, in data/imdb/:
  train.txt, test.txt   one line per document: "<label> <index> <index> ..." (1-based, sparse)
  vocab.txt             one feature per line, in literal order

    python research/imdb-readability/prepare.py [--features 12800] [--max-ngram 4]
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data", "imdb"))


def fetch(name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print("downloading %s ..." % name, flush=True)
        urllib.request.urlretrieve("%s/%s" % (BASE, name), path)
    return path


def load_imdb(num_words, index_from, seed=113):
    """Reproduce keras.datasets.imdb.load_data without keras.

    The stored sequences hold 0-based ranks into the word index; keras shifts them by `index_from`
    and prefixes a start token, so the same shift has to happen here or every id maps to the wrong
    word. Words at or beyond `num_words` become the OOV token, as upstream does.
    """
    with np.load(fetch("imdb.npz"), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    rng = np.random.RandomState(seed)
    for xs, ys in ((x_train, y_train), (x_test, y_test)):
        idx = np.arange(len(xs))
        rng.shuffle(idx)
        xs[:] = xs[idx]
        ys[:] = ys[idx]

    start_char, oov_char = 1, 2
    def shift(xs):
        out = [[start_char] + [w + index_from for w in x] for x in xs]
        return [[w if w < num_words else oov_char for w in x] for x in out]

    return shift(x_train), y_train, shift(x_test), y_test


def id_to_word_map(index_from):
    with open(fetch("imdb_word_index.json"), encoding="utf-8") as fh:
        word_to_id = {k: v + index_from for k, v in json.load(fh).items()}
    word_to_id["<PAD>"], word_to_id["<START>"], word_to_id["<UNK>"] = 0, 1, 2
    return {v: k for k, v in word_to_id.items()}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=12800)
    ap.add_argument("--max-ngram", type=int, default=4)
    ap.add_argument("--num-words", type=int, default=40000)
    ap.add_argument("--index-from", type=int, default=2)
    a = ap.parse_args(argv)

    x_tr, y_tr, x_te, y_te = load_imdb(a.num_words, a.index_from)
    id2w = id_to_word_map(a.index_from)
    docs = lambda xs: [[id2w.get(w, "<UNK>").lower() for w in x] for x in xs]
    train_docs, test_docs = docs(x_tr), docs(x_te)
    print("documents: %d train, %d test" % (len(train_docs), len(test_docs)), flush=True)

    vec = CountVectorizer(tokenizer=lambda s: s, token_pattern=None,
                          ngram_range=(1, a.max_ngram), lowercase=False, binary=True)
    Xtr = vec.fit_transform(train_docs)
    Xte = vec.transform(test_docs)
    print("raw features: %d" % Xtr.shape[1], flush=True)

    skb = SelectKBest(chi2, k=a.features).fit(Xtr, y_tr.astype(np.uint32))
    keep = skb.get_support(indices=True)
    names = np.asarray(vec.get_feature_names_out())[keep]
    Xtr, Xte = skb.transform(Xtr), skb.transform(Xte)
    print("selected %d features" % len(names), flush=True)

    def write(path, X, y):
        X = X.tocsr()
        with open(path, "w", encoding="utf-8") as fh:
            for i in range(X.shape[0]):
                cols = X.indices[X.indptr[i]:X.indptr[i + 1]] + 1     # 1-based for Julia
                fh.write("%d %s\n" % (int(y[i]), " ".join(map(str, sorted(cols)))))

    write(os.path.join(DATA, "train.txt"), Xtr, y_tr)
    write(os.path.join(DATA, "test.txt"), Xte, y_te)
    with open(os.path.join(DATA, "vocab.txt"), "w", encoding="utf-8") as fh:
        for n in names:
            fh.write(n + "\n")
    print("wrote train.txt, test.txt, vocab.txt to %s" % DATA)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
