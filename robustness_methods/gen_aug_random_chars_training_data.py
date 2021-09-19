import os
import random
import string
import nltk
import adver_tools
from copy import deepcopy
from nltk.corpus import stopwords


stop_words = stopwords.words("english")


def gen_aug_random_chars_data():
    all_ori_sents, all_ori_tags = adver_tools.read_data("../data/ontonotes_english/train.txt")
    _, entities_by_sid, _ = adver_tools.update_NER_dict(all_ori_sents, all_ori_tags)
    all_aug_sents = []
    all_aug_tags = []
    for sent_id, (sent, tags) in enumerate(zip(all_ori_sents, all_ori_tags)):
        entities = entities_by_sid[sent_id]
        if not entities:
            continue
        else:
            aug_sent = deepcopy(sent)
            ent_idxes = set()
            for ent in entities:
                ent_idxes.update(set(range(ent[1], ent[2]+1)))
            for token_idx in ent_idxes:
                token = sent[token_idx]
                if token.lower() in stop_words:
                    aug_sent[token_idx] = token
                    continue
                aug_token = ""
                for char in list(token):
                    if "a" <= char <= "z":
                        aug_token += random.choice(string.ascii_lowercase)
                    elif "A" <= char <= "Z":
                        aug_token += random.choice(string.ascii_uppercase)
                    else:
                        aug_token += char
                aug_sent[token_idx] = aug_token
            all_aug_sents.append(aug_sent)
            all_aug_tags.append(tags)
    return all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags


def output_aug_random_chars_data(all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags):
    with open("../data/ontonotes_english_aug_random_chars/train.txt", "w+") as f:
        for sent, tags in zip(all_ori_sents, all_ori_tags):
            for token, tag in zip(sent, tags):
                f.write(token + " " + tag + "\n")
            f.write("\n")
        for sent, tags in zip(all_aug_sents, all_aug_tags):
            for token, tag in zip(sent, tags):
                f.write(token + " " + tag + "\n")
            f.write("\n")


def main():
    random.seed(42)

    if not os.path.exists("../data/ontonotes_english_aug_random_chars"):
        os.system("mkdir -p ../data/ontonotes_english_aug_random_chars/")

    all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags = gen_aug_random_chars_data()

    output_aug_random_chars_data(all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags)


if __name__ == '__main__':
    main()
