import string
import json
from collections import defaultdict
ENTITY_TYPE_LIST_BY_DATASET_NAME = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART',
                                                        'PERSON', 'PRODUCT', 'LANGUAGE'],
                                    "conll2003": ["PER", "ORG", "LOC"],
                                    "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART',
                                                          'PERSON', 'PRODUCT', 'LANGUAGE']}

ENTITY_TYPE_LIST = []

# read data from file path
def read_data(DATA_FILE_PATH):
    all_sents = []
    all_tags = []
    with open(DATA_FILE_PATH, "r") as f:
        lines = f.read().split("\n")
        sent = []
        tags = []
        for line in lines:  # for each word
            ls = line.split()
            if len(ls) == 0:
                if len(sent) == 0:  # careful for blank lines
                    continue
                all_sents.append(sent)
                all_tags.append(tags)
                sent = []
                tags = []
                continue
            word = ls[0]
            tag = ls[-1]
            sent.append(word)
            tags.append(tag)
    f.close()
    return all_sents, all_tags


# read results data from file path
def read_results_data(DATA_FILE_PATH):
    all_sents = []
    all_tags = []
    all_pred_tags = []
    with open(DATA_FILE_PATH, "r") as f:
        lines = f.read().split("\n")
        sent = []
        tags = []
        pred_tags = []
        for line in lines:  # for each word
            ls = line.split()
            if len(ls) == 0:
                if len(sent) == 0:  # careful for blank lines
                    continue
                all_sents.append(sent)
                all_tags.append(tags)
                all_pred_tags.append(pred_tags)
                sent = []
                tags = []
                pred_tags = []
                continue
            word = ls[0]
            tag = ls[1]
            pred_tag = ls[2]
            sent.append(word)
            tags.append(tag)
            pred_tags.append(pred_tag)
    f.close()
    return all_sents, all_tags, all_pred_tags


# transform sentence from array to string
def sent_arr_to_text(sent):
    sent_string = ""
    for ind, word in enumerate(sent):
        if ind == len(sent) - 1:
            if len(word) == 1 and word[0] in string.punctuation:
                sent_string = sent_string[0:]  # remove last space TODO: what's the point
            sent_string += word
        else:
            sent_string += word
            sent_string += " "
    return sent_string


# classify entities by type, sentence id and derive their set
def update_NER_dict(all_sents, all_tags):
    entities_by_sid = defaultdict(lambda: list())   # sid is sentence id
    entities_by_type = defaultdict(lambda: list())
    entities_set = set()    # set contains lower case of each entity seen
    for sid, (sent, tags) in enumerate(zip(all_sents, all_tags)):
        entities = []
        start_index = -1
        end_index = -1
        type = None
        for ind, (word, tag) in enumerate(zip(sent, tags)):
            if tag.startswith("B-"):
                start_index = ind
                end_index = ind
                type = tag[2:]
            elif tag.startswith("I-"):
                end_index = ind
            if tag == "O" or ind == len(tags) - 1 or tags[ind+1] == "B":  # O or end of sentence or about to  start new ent
                if type:
                    entities.append((sid, start_index, end_index, type))
                start_index = -1
                end_index = -1
                type = None
        for ent in entities:
            ent_type = ent[3]
            entities_by_type[ent_type].append(ent)
            entities_by_sid[sid].append(ent)
            ent_surface = [w.lower() for w in sent[ent[1]:ent[2] + 1]]
            entities_set.add(tuple(ent_surface))
    return entities_by_type, entities_by_sid, entities_set


# get tags of a sentence with one entity
def get_tags(sentence, entity):
    tags = ["O" for i in range(0, len(sentence))]
    if len(entity) == 4:
        entity = (entity[1], entity[2], entity[3])
    if entity[2] == "NONE":
        return tags
    start_pos = int(entity[0])
    end_pos = int(entity[1])
    label = str(entity[2])
    if start_pos == end_pos:
        tags[start_pos] = "B-" + label
    else:
        tags[start_pos] = "B-" + label
        tags[start_pos+1: end_pos+1] = ["I-" + label for i in range(start_pos+1, end_pos+1)]
    return tags


# get all tags of a sentence
def get_all_tags(sentence, predictions):
    tags = ["O" for _ in range(len(sentence))]
    for entity in predictions:
        if len(entity) == 4:
            ent = (entity[1], entity[2], entity[3])
        else:
            ent = (entity[0], entity[1], entity[2])
        label = str(ent[2])
        if label in ENTITY_TYPE_LIST:
            start_pos = int(ent[0])
            end_pos = int(ent[1])
            if start_pos == end_pos:
                try:
                    tags[start_pos] = "B-" + label
                except:
                    print(sentence, len(sentence), predictions, len(predictions), start_pos, end_pos, tags)
            else:
                try:
                    tags[start_pos] = "B-" + label
                    tags[start_pos+1: end_pos+1] = ["I-" + label for _ in range(start_pos+1, end_pos+1)]
                except:
                    print(sentence, len(sentence), predictions, len(predictions), start_pos, end_pos, tags)
    return tags




# get entity_name with entity and all_sents
def get_entity_name(all_sents, entity):
    sent = all_sents[entity[0]]
    entity_name_list = sent[entity[1]:entity[2]+1]
    entity_name = str(sent_arr_to_text(entity_name_list))
    return entity_name


# get entity_name with entity and one sentence
def get_entity_text(sentence, entity):
    if len(entity) == 4:
        entity_text = " ".join(sentence[entity[1]:entity[2]+1])
    else:
        entity_text = " ".join(sentence[entity[0]:entity[1]+1])
    return entity_text


# get entity_name_list from entity
def get_entity_name_list(all_sents, entity):
    sent = all_sents[entity[0]]
    entity_name_list = sent[entity[1]:entity[2]+1]
    return entity_name_list


# get subword position in word
def get_subword_position(word, subword, start_index):
    if int(start_index)+len(subword) == len(word):
        return "ending"
    elif start_index == 0:
        return "beginning"
    else:
        return "middle"


# read nchars data from file path
def read_nchars_data(ent_type, DATA_FOLDER_PATH):
    nchars_dict = defaultdict(lambda: list)
    nchars_dict[3] = []
    nchars_dict[4] = []
    nchars_dict[5] = []
    for nchars_size in range(3, 6):
        DATA_FILE_PATH = DATA_FOLDER_PATH + '/' + ent_type + '.' + str(nchars_size) + 'chars.jsonl'
        with open(DATA_FILE_PATH) as f:
            for line in f:
                nchars_dict[nchars_size].append(json.loads(line))
    return nchars_dict


# read correct cases
def read_correct_cases(DATA_FOLDER_PATH, MODEL):
    correct_cases = []
    with open(DATA_FOLDER_PATH + '/' + MODEL + '.jsonl') as f:
        for line in f:
            correct_cases.append(json.loads(line))
    return correct_cases


# read sentence-entity pairs
def read_sentence_entity_pairs(DATA_FOLDER_PATH, file_type):
    correct_cases = []
    with open(DATA_FOLDER_PATH + '/' + file_type + '.jsonl') as f:
        for line in f:
            correct_cases.append(json.loads(line))
    return correct_cases


# get GloVe vocabulary
def get_glove_vocabulary():
    vocabulary_set = set()
    with open('../data/glove.6B/glove.6B.50d.vocabulary.tsv') as f:
        for line in f:
            values = line.split()
            vocabulary_set.add(values[0])
    return vocabulary_set


# check if it is in GloVe vocabulary
def is_a_word(nchars, vocabulary):
    if str(nchars).lower() in vocabulary:
        return True
    return False


def read_all_types_freq_data(DATA_FOLDER_PATH, DATASET_NAME):
    word_freq = {}
    entity_freq = {}
    for ent_type in ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]:
        with open(DATA_FOLDER_PATH + '/' + ent_type + ".word.tsv") as word_file:
            word_lines = word_file.read().split('\n')[:-1]
            for line in word_lines:
                data = line.split('\t')
                if data[0] in word_freq.keys():
                    word_freq[data[0]]["freq"] += int(data[1])
                    word_freq[data[0]]["entity_num"] += int(data[2])
                else:
                    word_freq[data[0]] = {}
                    word_freq[data[0]]["freq"] = int(data[1])
                    word_freq[data[0]]["entity_num"] = int(data[2])
        with open(DATA_FOLDER_PATH + '/' + ent_type + ".entity.tsv") as entity_file:
            entity_lines = entity_file.read().split('\n')[:-1]
            for line in entity_lines:
                data = line.split('\t')
                if data[0] in entity_freq.keys():
                    entity_freq[data[0]] += int(data[1])
                else:
                    entity_freq[data[0]] = int(data[1])
    return word_freq, entity_freq


def get_context_words_freq(all_sents):
    context_words_freq = {}
    for sentence in all_sents:
        for word in sentence:
            if word not in context_words_freq.keys():
                context_words_freq[word] = 1
            else:
                context_words_freq[word] += 1
    return context_words_freq


def read_entities_and_entity_words_by_type(DATA_FOLDER_PATH, DATASET_NAME):
    entities = {}
    entity_words = {}
    for ent_type in ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]:
        entities[ent_type] = set()
        entity_words[ent_type] = set()
    for ent_type in ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]:
        with open(DATA_FOLDER_PATH+'/'+ent_type+'.entity.tsv', 'r') as f:
            for line in f:
                data = line.strip().split('\t')
                entities[ent_type].add(data[0])
        with open(DATA_FOLDER_PATH+'/'+ent_type+'.word.tsv', 'r') as f:
            for line in f:
                data = line.strip().split('\t')
                entity_words[ent_type].add(data[0])
    return entities, entity_words
