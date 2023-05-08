import json
from tree_build.utils import *


MAX_SENT_LENGTH = 100
MAX_TOKEN_LENGTH = 70


def search(pat, txt):#找到pat在txt子串的第一次出现位置
    i, N = 0, len(txt)
    j, M = 0, len(pat)
    while i < N and j < M:
        if txt[i] == pat[j]:
            j = j + 1
        else:
            i -= j
            j = 0
        i = i + 1
    if j == M:
        return i - M
    else:
        return -1


def make_tag_set(tag_set, relation_label):
    if relation_label == "None":
        return
    for pos in "BIES":
        for role in "12":
            tag_set.add("-".join([pos, relation_label, role]))#pos-relation_label-role


def update_tag_seq(em_text, sentence_text, relation_label, role, tag_set, tags_idx):
    overlap = False
    start = search(em_text, sentence_text)
    tag = "-".join(["S", relation_label, str(role)])
    if len(em_text) == 1:
        if tags_idx[start] != tag_set["O"]:
            overlap = True
        tags_idx[start] = tag_set[tag]
    else:
        tag = "B" + tag[1:]
        if tags_idx[start] != tag_set["O"]:
            overlap = True
        tags_idx[start] = tag_set[tag]
        tag = "E" + tag[1:]
        end = start + len(em_text) - 1
        if tags_idx[end] != tag_set["O"]:
            overlap = True
        tags_idx[end] = tag_set[tag]
        tag = "I" + tag[1:]
        for index in range(start + 1, end):
            if tags_idx[index] != tag_set["O"]:
                overlap = True
            tags_idx[index] = tag_set[tag]
    return overlap

    #prepare_data_set(fin, charset, vocab, relation_labels, entity_labels, tag_set, train, fout)
def prepare_data_set(fin, charset, vocab, relation_labels, entity_labels, tag_set, dataset, fout):
    num_overlap = 0
    data = json.load(fin)
    overlap=False
    for sentence in data:
        # if line == '[\n' or line == '[':
        #     continue
        # overlap = False
        # line = line.strip()
        # if not line:
        #     continue
        # sentence = json.loads(line)

        for relation_mention in sentence["relationMentions"]:
            relation_labels.add(relation_mention["label"])
            make_tag_set(tag_set, relation_mention["label"])
        for entity_mention in sentence["entityMentions"]:
            entity_labels.add(entity_mention["label"])

        sentence_text = sentence["sentText"].strip().strip('"').split()
        length_sent = len(sentence_text)
        if length_sent > MAX_SENT_LENGTH:
            continue

        lower_sentence_text = [token.lower() for token in sentence_text]
        sentence_idx = prepare_sequence(lower_sentence_text, vocab)

        tokens_idx = []
        for token in sentence_text:
            if len(token) <= MAX_TOKEN_LENGTH:
                tokens_idx.append(prepare_sequence(token, charset) + [charset["<pad>"]]*(MAX_TOKEN_LENGTH-len(token)))
            else:
                tokens_idx.append(prepare_sequence(token[0:13] + token[-7:], charset))

        tags_idx = [tag_set["O"]] * length_sent
        for relation_mention in sentence["relationMentions"]:
            if relation_mention["label"] == "None":
                continue
            em1_text = relation_mention["em1Text"].split()
            res1 = update_tag_seq(em1_text, sentence_text, relation_mention["label"], 1, tag_set, tags_idx)
            em2_text = relation_mention["em2Text"].split()
            res2 = update_tag_seq(em2_text, sentence_text, relation_mention["label"], 2, tag_set, tags_idx)
            if res1 or res2:
                num_overlap += 1
                overlap = True
        dataset.append((sentence_idx, tokens_idx, tags_idx))
        if overlap:
            fout.write(str(sentence)+"\n")
    return num_overlap


if __name__ == "__main__":
    charset = Charset()
    vocab = Vocabulary()
    vocab.load("../data_v0/NYT_CoType/vocab.txt")
    relation_labels = Index()
    entity_labels = Index()
    tag_set = Index()
    tag_set.add("O")

    with open("overlap.txt", "wt", encoding="utf-8") as fout:
        train = []
        with open('../data_v0/tree_build/train_aug.json', 'rt', encoding='utf-8') as fin:
            res = prepare_data_set(fin, charset, vocab, relation_labels, entity_labels, tag_set, train, fout)
            print("# of overlaps in train data: {}".format(res))
        save(train, '../data_v0/tree_build/train.pk')

        test = []
        with open('../data_v0/tree_build/test_aug.json', 'rt', encoding='utf-8') as fin:
            res = prepare_data_set(fin, charset, vocab, relation_labels, entity_labels, tag_set, test, fout)
            print("# of overlaps in test data: {}".format(res))
        save(test, '../data_v0/tree_build/test.pk')

    relation_labels.save('../data_v0/tree_build/relation_labels.txt')
    entity_labels.save('../data_v0/tree_build/entity_labels.txt')
    tag_set.save("../data_v0/tree_build/tag2id.txt")

# of overlaps in train data: 42924
# of overlaps in test data: 18
