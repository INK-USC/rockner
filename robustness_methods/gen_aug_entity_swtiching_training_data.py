import os
import random
import string
import adver_tools
from copy import deepcopy
from tqdm import tqdm


def gen_aug_entity_switching_data():
    all_ori_sents, all_ori_tags = adver_tools.read_data("../data/ontonotes_english/train.txt")
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_ori_sents, all_ori_tags)
    all_aug_sents, all_aug_tags = [], []
    for sent_id, (sent, tags) in enumerate(tqdm(zip(all_ori_sents, all_ori_tags), total=len(all_ori_sents))):
        entities = entities_by_sid[sent_id]
        if not entities:
            all_aug_sents.append(sent)
            all_aug_tags.append(tags)
        else:
            right_shift = 0
            aug_tags = deepcopy(tags)
            aug_sent = deepcopy(sent)
            for ent in entities:
                candidates = deepcopy(entities_by_type[ent[3]])
                candidates.remove(tuple(ent))
                sampled_ent = random.sample(candidates, k=1)[0]
                sampled_sent = all_ori_sents[sampled_ent[0]]
                sampled_tags = all_ori_tags[sampled_ent[0]]
                length_change = (sampled_ent[2]+1-sampled_ent[1]) - (ent[2]+1-ent[1])
                start_idx = ent[1] + right_shift
                # end_idx = ent[2] + right_shift + length_change
                aug_sent = aug_sent[:start_idx] + sampled_sent[sampled_ent[1]:sampled_ent[2]+1] + aug_sent[ent[2]+right_shift+1:]
                aug_tags = aug_tags[:start_idx] + sampled_tags[sampled_ent[1]:sampled_ent[2]+1] + aug_tags[ent[2]+right_shift+1:]
                right_shift += length_change
            all_aug_sents.append(aug_sent)
            all_aug_tags.append(aug_tags)
    return all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags


def output_aug_entity_switching_data(all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags):
    with open("../data/ontonotes_english_aug_entity_switching/train.txt", "w+") as f:
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

    if not os.path.exists("../data/ontonotes_english_aug_entity_switching"):
        os.system("mkdir -p ../data/ontonotes_english_aug_entity_switching/")

    all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags = gen_aug_entity_switching_data()

    output_aug_entity_switching_data(all_ori_sents, all_ori_tags, all_aug_sents, all_aug_tags)


if __name__ == '__main__':
    main()
