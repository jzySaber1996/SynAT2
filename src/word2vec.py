import os
import json
from gensim.models.word2vec import LineSentence, Word2Vec


def func(fin, fout):
    for line in fin:
        line = line.strip()
        if not line:
            continue
        sentence = json.loads(line)
        sentence = sentence["sentText"].strip().strip('"').lower()
        fout.write(sentence + '\n')


def make_corpus():
    #print("-------------haha")
    with open('../data_v0/tree_build/corpus.txt', 'wt', encoding='utf-8') as fout:
        with open('../data_v0/tree_build/train_aug.json', 'rt', encoding='utf-8') as fin:
            func(fin, fout)
        with open('../data_v0/tree_build/test_aug.json', 'rt', encoding='utf-8') as fin:
            func(fin, fout)


if __name__ == "__main__":
    if not os.path.exists('../data_v0/tree_build/corpus.txt'):
        make_corpus()

    sentences = LineSentence('../data_v0/tree_build/corpus.txt')
    model = Word2Vec(sentences, sg=1, size=300, workers=4, iter=8, negative=8)
    word_vectors = model.wv
    word_vectors.save('../data_v0/tree_build/word2vec')
    word_vectors.save_word2vec_format('../data_v0/tree_build/word2vec.txt', fvocab='../data_v0/tree_build/vocab.txt')