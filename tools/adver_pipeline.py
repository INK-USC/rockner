import adver_tools
import random
import string
import operator
from copy import deepcopy



DATASET_NAME = "ontonotes_names"
NCHARS_DATA_FOLDER_PATH = "./entity_statistics/" + DATASET_NAME
ENTITY_TYPE_LIST = ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART', 'PERSON', 'PRODUCT', 'LANGUAGE']
ENTITY_TYPE_LIST_WITHOUT_PERSON = []


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        from spacy.tokens import Doc
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def init():
    import spacy
    from spacy.tokens import DocBin
    from flair.data import Sentence
    from flair.models import SequenceTagger
    from happytransformer import HappyROBERTA, HappyBERT
    global tagger, nlp, bert, nchars_data
    bert = HappyBERT("bert-base-cased")
    tagger = SequenceTagger.load('ner-ontonotes-fast')
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    nchars_data = {}
    eval_result = {}
    for ent_type in ENTITY_TYPE_LIST:
        nchars_data[ent_type] = adver_tools.read_nchars_data(ent_type, NCHARS_DATA_FOLDER_PATH)
        eval_result[ent_type] = []
    return tagger, nlp, nchars_data, eval_result


def flair_init():
    from flair.data import Sentence
    from flair.models import SequenceTagger
    global tagger
    tagger = SequenceTagger.load('ner-ontonotes-fast')
    return tagger


def model_init(model, **kw):
    if model in ["self_trained_flair", "flair", "self_trained_flair_attacked"]:
        from flair.models import SequenceTagger
        global tagger
    elif model in ["spacy_lg", "spacy_md"]:
        import spacy
        global nlp
    elif model in ["stanza"]:
        import stanza
        global nlp
    elif model in ["bert_crf", "roberta_crf", "bert_crf_attacked", "roberta_crf_attacked", "roberta_crf_aug_random_chars",
                   "roberta_crf_aug_entity_switching", "roberta_crf_aug_mix_up", "blstm_crf", "blstm_crf_attacked",
                   "roberta_crf_aug_mixing_up", "roberta_crf_aug_random_masking"]:
        import sys
        sys.path.append("../pytorch_lstmcrf")
        global predictor
    if model == "self_trained_flair":
        tagger = SequenceTagger.load(model="../model_training/resources/taggers/self_trained_flair_gpu/best-model.pt")
        return tagger
    elif model == "flair":
        tagger = SequenceTagger.load('ner-ontonotes-fast')
        return tagger
    elif model == "self_trained_flair_attacked":
        tagger = SequenceTagger.load(model="../model_training/resources/taggers/self_trained_flair_gpu_attacked/best-model.pt")
        return tagger
    elif model == "spacy_lg":
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_lg")
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        return nlp
    elif model == "spacy_md":
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_md")
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        return nlp
    elif model == "stanza":
        nlp = stanza.Pipeline(lang='en', use_gpu=True, processors='tokenize,ner', tokenize_pretokenized=True)
        return nlp
    elif model in ["bert_crf", "roberta_crf", "bert_crf_attacked", "roberta_crf_attacked", 
                   "roberta_crf_aug_random_chars", "roberta_crf_aug_entity_switching", "roberta_crf_aug_mix_up",
                   "roberta_crf_aug_mixing_up", "roberta_crf_aug_random_masking"]:
        from transformers_predictor import TransformersNERPredictor
        predictor = TransformersNERPredictor(kw['model_path'], kw['device'])
        return predictor
    elif model == "blstm_crf_attacked":
        from ner_predictor import NERPredictor
        predictor = NERPredictor("../model_training/resources/taggers/blstm_crf_attacked/blstm_crf_attacked.tar.gz")
        return predictor
    elif model == "blstm_crf":
        from ner_predictor import NERPredictor
        predictor = NERPredictor("../model_training/resources/taggers/blstm_crf/blstm_crf.tar.gz")
        return predictor
    else:
        print("Model not supported.")
        return



def transform_input_data(sentence, entity):
    """Transforms initial data form into another which is compatible with model input.
    Args:
        sentence: A list of words.
        entity: A (sid, start_index_in_list, end_index_in_list, type) tuple.
    Returns:
        new_sentence: A string.
        new_entity: A dictionary.
    """
    new_sentence = adver_tools.sent_arr_to_text(sentence)
    entity_name_list = sentence[entity[1]:entity[2]+1]
    entity_name = str(adver_tools.sent_arr_to_text(entity_name_list))
    index = new_sentence.index(entity_name)
    rindex = new_sentence.rindex(entity_name)
    if index == rindex:
        new_entity = dict(text=entity_name, start_index=index, end_index=index + len(entity_name), label=entity[3])
    else:
        start_index = entity[1]     # entity[1] is the number of white spaces that will be added.
        for word_index in range(0, entity[1]):
            start_index += len(sentence[word_index])
        new_entity = dict(text=entity_name, start_index=start_index, end_index=start_index + len(entity_name), label=entity[3])
    return new_sentence, new_entity


def predict_sentence(sentence, model): 
    if model in ["self_trained_flair", "flair", "self_trained_flair_attacked"]:
        predictions = predict_with_flair(sentence)
    elif model in ["spacy_md", "spacy_lg"]:
        predictions = predict_with_spacy(sentence)
    elif model in ["stanza"]:
        predictions = predict_with_stanza(sentence)
    elif model in ["bert_crf", "roberta_crf", "bert_crf_attacked", "roberta_crf_attacked", "roberta_crf_aug_random_chars",
                   "roberta_crf_aug_entity_switching", "roberta_crf_aug_mix_up", "roberta_crf_aug_mixing_up", 
                   "roberta_crf_aug_random_masking"]:
        predictions = predict_with_transformer(sentence)
    elif model in ["blstm_crf", "blstm_crf_attacked"]:
        predictions = predict_with_blstm(sentence)
    else:
        print("Model not supported.")
        return
    return predictions


def verify_results(predictions, entity):
    if entity in predictions:
        return True
    return False


def predict_with_flair(sentence):
    from flair.data import Sentence
    predictions = []
    sentence_string = " ".join(sentence)
    flair_sentence = Sentence(sentence_string, use_tokenizer=False)
    if len(flair_sentence.tokens) != len(sentence):
        print(sentence, flair_sentence)
    tagger.predict(flair_sentence)
    spans = flair_sentence.get_spans('ner')
    for ent in spans:
        start_index = ent.tokens[0].idx - 1
        end_index = ent.tokens[-1].idx - 1
        predicted_entity = (start_index, end_index, ent.tag)
        predictions.append(predicted_entity)
    return predictions


def predict_with_spacy(sentence):
    predictions = []
    sentence_string = adver_tools.sent_arr_to_text(sentence)
    doc = nlp(sentence_string)
    for ent in doc.ents:
        start_index = len(sentence_string[:ent.start_char].split())
        end_index = start_index + len(ent.text.split()) - 1
        predicted_entity = (start_index, end_index, ent.label_)
        predictions.append(predicted_entity)
    return predictions


def predict_with_stanza(sentence):
    predictions = []
    sentence_string = " ".join(sentence)
    doc = nlp(sentence_string)
    for ent in doc.ents:
        ent_text = ent.text
        ent_type = ent.type
        start_index = len(sentence_string.split(ent_text)[0].split())
        end_index = start_index - 1 + len(ent_text.split())
        predicted_entity = (start_index, end_index, ent_type)
        predictions.append(predicted_entity)
    return predictions


def predict_with_transformer(sentence):
    predictions = []
    results = predictor.predict([sentence])
    start_idx = 0
    end_idx = 0
    for token_idx, token_type in enumerate(results[0]):
        if "S-" in token_type:
            predictions.append((token_idx, token_idx, token_type[2:]))
        elif "B-" in token_type:
            start_idx = token_idx
        elif "E-" in token_type:
            end_idx = token_idx
            predictions.append((start_idx, end_idx, token_type[2:]))
    return predictions


def predict_with_blstm(sentence):
    predictions = []
    result = predictor.predict(" ".join(sentence))
    start_idx = 0
    end_idx = 0
    for token_idx, token_type in enumerate(result):
        if "S-" in token_type:
            predictions.append((token_idx, token_idx, token_type[2:]))
        elif "B-" in token_type:
            start_idx = token_idx
        elif "E-" in token_type:
            end_idx = token_idx
            predictions.append((start_idx, end_idx, token_type[2:]))
    return predictions


def entity_level_subword_basic_attacker(sentence, entity, args):
    entity_type_list = ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART', 'PERSON', 'PRODUCT', 'LANGUAGE']
    if entity[2] in entity_type_list:
        entity_type_list.remove(entity[2])
    adver_list = []
    attack_num = args["attack_num"]
    attack_word_num = args["attack_word_num"]
    attack_subword_pos = args["attack_subword_pos"]
    nchars_data = args["nchars_data"]
    for index in range(attack_num):
        adver_sentence = sentence.copy()
        ent_type = random.choice(entity_type_list)
        nchars_size = random.choice([3, 4, 5])
        nchars_list = nchars_data[ent_type][nchars_size][:100]
        for entity_word_index in range(entity[0], min(entity[1]+1, entity[0]+attack_word_num)):
            nchars, position = random_choose_word_data(nchars_list, attack_subword_pos)
            entity_word = sentence[entity_word_index]
            adver_word = change_entity_subword(entity_word, nchars, nchars_size, position)
            adver_sentence[entity_word_index] = adver_word
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(entity)))
    return adver_list


def random_choose_word_data(nchars_list, attack_subword_pos):
    if attack_subword_pos not in ["all", "beginning", "middle", "ending"]:
        print("Error: ATTACK_POS is wrong!")
    position = ""
    while attack_subword_pos != position:
        item = random.choice(nchars_list)
        nchars = str(item["nchars"])
        word_data = random.choice(list(item["words"].values()))
        position = word_data["position"]
        if attack_subword_pos == "all":
            break
    return nchars, position

def change_entity_subword(word, nchars, nchars_size, position):
    word_length = len(word)
    if position == "beginning":
        if nchars_size < word_length:
            adver_word = nchars + word[nchars_size:]
        else:
            adver_word = nchars
    elif position == "middle":
        if nchars_size <= word_length-2:
            start_index = random.choice(range(1, word_length-nchars_size+1))
            adver_word = word[:start_index] + nchars + word[start_index+nchars_size:]
        else:
            if word_length <= 2:
                insert_index = 1
            else:
                insert_index = random.choice(range(1, word_length-1))
            adver_word = word[:insert_index] + nchars + word[insert_index:]
    else:
        if nchars_size < word_length:
            adver_word = word[:word_length-nchars_size] + nchars
        else:
            adver_word = word + nchars
    return adver_word

def context_level_basic_attacker(sentence, entity, args):
    attack_num = args["attack_num"]
    attack_type = args["attack_type"]
    attack_word_num = args["attack_word_num"]
    attack_word_pos = args["attack_word_pos"]
    if attack_word_pos not in ["front", "behind"]:
        print("Error: ATTACK_POS is wrong!")
        return
    if attack_type not in ["remove", "replace"]:
        print("Error: ATTACK_TYPE is wrong!")
        return

    adver_list = []

    if attack_type == "replace":
        glove_vocab = adver_tools.get_glove_vocabulary()
    for index in range(0, attack_num):
        if attack_type == "remove":
            adver_sentence, adver_entity = remove_word(sentence, entity, attack_word_num, attack_word_pos)
            if adver_sentence is None:
                continue
        else:
            adver_sentence, adver_entity = replace_word(sentence, entity, attack_word_num, attack_word_pos, glove_vocab)
            if adver_sentence is None:
                continue
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(adver_entity)))
    return adver_list


def remove_word(sentence, entity, attack_word_num, attack_word_pos):
    if (attack_word_pos == "front" and entity[0]-attack_word_num < 0) or (attack_word_pos == "behind" and entity[1]+attack_word_num > len(sentence)-1):
        print("Error: No word to remove!")
        return
    if attack_word_pos == "front":
        adver_sentence = sentence[:entity[0]-attack_word_num] + sentence[entity[0]:]
        adver_entity = (entity[0]-attack_word_num, entity[1]-attack_word_num, entity[2])
    else:
        adver_sentence = sentence[:entity[1]+1] + sentence[entity[1]+attack_word_num+1:]
        adver_entity = deepcopy(entity)
    return adver_sentence, adver_entity


def replace_word(sentence, entity, attack_word_num, attack_word_pos, glove_vocab):
    if (attack_word_pos == "front" and entity[0]-attack_word_num < 0) or (attack_word_pos == "behind" and entity[1]+attack_word_num > len(sentence)-1):
        print("Error: No word to replace!")
        return
    random_words = []
    for index in range(0, attack_word_num):
        random_words.append(random.choice(list(glove_vocab)))
    if attack_word_pos == "front":
        adver_sentence = sentence[:entity[0]-attack_word_num] + random_words + sentence[entity[0]:]
    else:
        adver_sentence = sentence[:entity[1]+1] + random_words + sentence[entity[1]+attack_word_num+1:]
    adver_entity = deepcopy(entity)
    return adver_sentence, adver_entity


def get_predicted_entity_type(predictions, entity):
    for prediction in predictions:
        if prediction[0] <= entity[0] and prediction[1] >= entity[1]:
            if prediction[2] in ENTITY_TYPE_LIST:
                return prediction[2]
    return "NONE"


def context_level_replace_with_BERT_attacker(sentence, entity, context_word_freq, glove_vocab, args):
    attack_num = args["attack_num"]
    attack_word_num = args["attack_word_num"]
    bert_results_num = args["bert_results_num"]
    adver_list = []
    pos_tags = get_pos_tags(sentence)
    replaceable_indexes = get_replaceable_indexes(pos_tags, entity)
    if len(replaceable_indexes) == 0:
        return adver_list
    replaceable_indexes = replaceable_indexes[:attack_word_num]
    replaceable_words = {}
    for index in replaceable_indexes:
        temp_sent = sentence.copy()
        temp_sent[index] = "[MASK]"
        # sent_string = adver_tools.sent_arr_to_text(temp_sent)
        sent_string = " ".join(temp_sent)
        bert_predict_results = bert.predict_mask(sent_string, num_results=bert_results_num)
        bert_results = bert_predict_results.copy()
        for result in bert_predict_results:
            if result["word"] == adver_tools.get_entity_text(sentence, entity) or result["word"].lower() not in glove_vocab:
                bert_results.remove(result)
        bert_results = sorted(bert_results, key=lambda x: context_word_freq.get(x["word"], 0))
        replaceable_words[index] = bert_results[:attack_num]
    for attack_round in range(attack_num):
        adver_sentence = sentence.copy()
        for index in replaceable_indexes:
            adver_sentence[index] = replaceable_words[index][attack_round]["word"]
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(entity), replaceable_indexes=replaceable_indexes))
    return adver_list


def get_pos_tags(sentence):
    pos_tags = []
    sentence_string = adver_tools.sent_arr_to_text(sentence)
    doc = nlp(sentence_string)
    for token in doc:
        pos_tags.append(token.pos_)
    return pos_tags


def get_replaceable_indexes(pos_tags, entity):
    replaceable_indexes = []
    for index, tag in enumerate(pos_tags):
        if entity[0] <= index <= entity[1]:
            continue
        if tag in ["NOUN", "VERB", "ADJ"]:
            replaceable_indexes.append(index)
    if len(replaceable_indexes) == 0:
        return []
    replaceable_indexes = sorted(replaceable_indexes, key=lambda x: abs(x-(entity[0]+entity[1])/2))
    return replaceable_indexes


def entity_level_replace_word_attacker(sentence, entity, context_word_freq, entity_set, glove_vocab, args):
    attack_num = args["attack_num"]
    attack_word_num = args["attack_word_num"]
    bert_results_num = args["bert_results_num"]
    adver_list = []
    attack_word_num = min(attack_word_num, entity[1]-entity[0]+1)
    bert_entity_predictions = {}
    for ent_index in range(entity[0], entity[1]+1):
        temp_sent = sentence.copy()
        temp_sent[ent_index] = "[MASK]"
        sent_string = " ".join(temp_sent)
        bert_predict_results = bert.predict_mask(sent_string, num_results=bert_results_num)
        bert_results = bert_predict_results.copy()
        for result in bert_predict_results:
            # TODO: double check glove vocab
            # TODO: check if new eneity in training set
            if ent_index < entity[1]:
                new_entity_text = " ".join(sentence[entity[0]:ent_index] + [result["word"]] + sentence[ent_index+1:entity[1]])
            else:
                new_entity_text = " ".join(sentence[entity[0]:entity[1]] + [result["word"]])
            if result["word"] == sentence[ent_index] or result["word"].lower() not in glove_vocab or new_entity_text in entity_set:
                bert_results.remove(result)
        bert_results = sorted(bert_results, key=lambda x: context_word_freq.get(x["word"], 0))
        bert_entity_predictions[ent_index] = bert_results[:attack_num]
    entity_all_indexes = list(range(entity[0], entity[1]+1))
    replaceable_entity_indexes = []
    for _ in range(attack_word_num):
        replacealbe_entity_index = random.choice(entity_all_indexes)
        replaceable_entity_indexes.append(replacealbe_entity_index)
        entity_all_indexes.remove(replacealbe_entity_index)
    replaceable_entity_indexes = sorted(replaceable_entity_indexes)
    for attack_round in range(attack_num):
        adver_sentence = sentence.copy()
        for index in replaceable_entity_indexes:
            adver_sentence[index] = bert_entity_predictions[index][attack_round]["word"]
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(entity)))
    return adver_list


def entity_only_attacker(sentence, entity):
    adver_list = []
    adver_sentence = sentence[entity[0]: entity[1]+1]
    adver_entity = (0, entity[1]-entity[0], entity[2])
    adver_list.append(dict(sentence=adver_sentence, entity=tuple(adver_entity)))
    return adver_list


def unseen_entity_only_attacker(sentence, entity, entity_freq):
    if adver_tools.get_entity_text(sentence, entity) not in set(entity_freq.keys()):
        adver_list = []
        adver_sentence = sentence[entity[0]: entity[1]+1]
        adver_entity = (0, entity[1]-entity[0], entity[2])
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(adver_entity)))
        return adver_list
    return []


def seen_entity_only_attacker(sentence, entity, entity_freq):
    if adver_tools.get_entity_text(sentence, entity) in set(entity_freq.keys()):
        adver_list = []
        adver_sentence = sentence[entity[0]: entity[1]+1]
        adver_entity = (0, entity[1]-entity[0], entity[2])
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(adver_entity)))
        return adver_list
    return []


def entity_level_replace_with_wikiinstances(sentence, entity, instances, instances_dict, args):
    attack_num = args["attack_num"]
    adver_list = []
    adver_instances = random.sample(list(instances[entity[2]]), k=attack_num)
    for attack_round in range(attack_num):
        instance = adver_instances[attack_round].split()
        adver_sentence = sentence[:entity[0]] + instance + sentence[entity[1]+1:]
        adver_entity = (entity[0], len(instance) + entity[0] - 1, entity[2])
        adver_list.append(dict(sentence=adver_sentence, entity=tuple(adver_entity),
                               derive_from=instances_dict[entity[2]][adver_instances[attack_round]]
                               if entity[2] != "PERSON" else ""))
    return adver_list


def entity_level_replace_with_several_wikiinstances(sentence, complete_ents, instances, instances_dict, args):
    attack_num = args["attack_num"]
    adver_sentences = {k: [] for k in range(attack_num)}
    adver_entities_by_round = {k: [] for k in range(attack_num)}
    adver_lists = {k: {} for k in range(attack_num)}
    # combined_entities = sorted(entities + non_sample_entities, key=lambda x: x[0], reverse=False)
    shifted_non_sample_ents = {k: [] for k in range(attack_num)}
    for attack_round in range(attack_num):
        entity_num = random.choice(range(1, len(complete_ents) + 1))
        sample_ents = [(ent[0], ent[1], ent[2]) for ent in random.sample(complete_ents, k=entity_num)]
        sample_ents = sorted(sample_ents, key=lambda x: x[0], reverse=False)
        entity_candidates = {ent_idx: random.sample(list(instances[sample_ents[ent_idx][2]]), k=1)
                             for ent_idx in range(entity_num)}
        # non_sample_ents = sorted(list(set(complete_ents) - set(sample_ents)), key=lambda x: x[0], reverse=False)
        adver_sentence = deepcopy(sentence)
        adver_entities = []
        ent_right_shift = 0
        ent_candidate_id = 0
        for ent_idx, entity in enumerate(complete_ents):
            if entity in sample_ents:
                instance = entity_candidates[ent_candidate_id][0].split()
                ent_len_change = len(instance) - (entity[1]+1-entity[0])
                start_idx = entity[0] + ent_right_shift
                end_idx = entity[1] + ent_right_shift + ent_len_change
                adver_sentence = adver_sentence[:start_idx] + instance + adver_sentence[entity[1]+ent_right_shift+1:]
                ent_right_shift += ent_len_change
                adver_entities.append((start_idx, end_idx, entity[2]))
                ent_candidate_id += 1
            else:
                shifted_non_sample_ents[attack_round].append((entity[0]+ent_right_shift,
                                                              entity[1]+ent_right_shift,
                                                              entity[2]))
        adver_sentences[attack_round] = adver_sentence
        adver_entities_by_round[attack_round] = adver_entities
        adver_lists[attack_round] = dict(original=dict(sentence=" ".join(sentence), all_entities=complete_ents,
                                                       sampled_entities=sample_ents),
                                         adver=dict(sentence=" ".join(adver_sentence), entities=adver_entities),
                                         entities_with_derivation=[dict(entity=sample_ents[ent_idx],
                                                                        instance=" ".join(instance),
                                                                        derive_from=instances_dict[sample_ents[ent_idx]
                                                                        [2]][entity_candidates[ent_idx][0]]
                                                                        if sample_ents[ent_idx][2] != "PERSON" else "")
                                                                   for ent_idx in range(entity_num)])
    return adver_sentences, adver_entities_by_round, adver_lists, shifted_non_sample_ents


def choose_entity_candidates(sample_ents, ent_classes, class_ents, class_instances):
    entity_candidates = {}
    for ent_index, ent in enumerate(sample_ents):
        ent_type = ent[3]
        if ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
            if not tuple(ent) in ent_classes[ent_type].keys():
                entity_candidates[ent_index] = {"status": "None QID or none class."}
                continue
            classes = {k: v for k, v in ent_classes[ent_type][tuple(ent)]['classes'].items()
                       if k in class_instances[ent[3]].keys() and class_instances[ent[3]][k]['instance_num'] != 0}
            class_keys = list(classes.keys())
            # random.shuffle(class_keys)
            # class_ent_num_dict = {k: class_ents[ent_type][k]['ent_num'] for k in class_keys}
            # chosen_class_id = max(class_ent_num_dict.items(), key=operator.itemgetter(1))[0]
            chosen_class_id = random.sample(class_keys, k=1)[0]
            chosen_class = class_instances[ent_type][chosen_class_id]
            chosen_class_instances = chosen_class['instances']
            if not chosen_class_instances:
                entity_candidates[ent_index] = {"status": "None instances for " + chosen_class_id + " " + chosen_class['class_title']}
                continue
            entity_candidates[ent_index] = {k: {"instance_title": chosen_class_instances[k],
                                                "class_info": dict(class_id=chosen_class_id,
                                                                   class_title=chosen_class['class_title'],
                                                                   instance_num=chosen_class['instance_num'],
                                                                   ent_num=class_ents[ent_type][chosen_class_id]['ent_num'],
                                                                   QID_num=class_ents[ent_type][chosen_class_id]['QID_num'])
                                                }
                                            for k in random.sample(chosen_class_instances.keys(), k=1)}
        else:
            entity_candidates[ent_index] = {k: {"instance_title": class_instances["PERSON"][k],
                                                "class_info": "N/A"}
                                            for k in random.sample(class_instances["PERSON"].keys(), k=1)}
    return entity_candidates


def entity_level_replace_with_homogeneous_wikiinstances(sentence, complete_ents, ent_classes, class_ents, class_instances, args):
    attack_num = args["attack_num"]
    adver_sentences = {k: [] for k in range(attack_num)}
    adver_entities_by_round = {k: [] for k in range(attack_num)}
    adver_lists = {k: {} for k in range(attack_num)}
    # combined_entities = sorted(entities + non_sample_entities, key=lambda x: x[0], reverse=False)
    shifted_non_sample_ents = {k: [] for k in range(attack_num)}
    none_data_sampled_entities = {k: {} for k in range(attack_num)}
    for attack_round in range(attack_num):
        random.seed(args["seeds"][attack_round])
        entity_num = random.choice(range(1, len(complete_ents) + 1))
        sample_ents = [ent for ent in random.sample(complete_ents, k=entity_num) if ent[3] == "PERSON" or ent in ent_classes[ent[3]].keys()]
        sample_ents = sorted(sample_ents, key=lambda x: x[1], reverse=False)
        entity_candidates = choose_entity_candidates(sample_ents, ent_classes, class_ents, class_instances)
        new_sample_ents = deepcopy(sample_ents)
        for ent_idx in entity_candidates.keys():
            status = entity_candidates[ent_idx].get("status", None)
            if status:
                none_data_sampled_entities[attack_round][tuple(sample_ents[ent_idx])] = status
                new_sample_ents.remove(sample_ents[ent_idx])
        ent_candidate_ids = sorted([candidate_id for candidate_id in entity_candidates.keys()
                                    if not entity_candidates[candidate_id].get("status")])
        entity_candidates = {k: entity_candidates[ent_candidate_ids[k]] for k in range(len(new_sample_ents))}
        # if len(new_sample_ents) >= 2:
        #     print('success')
        # entity_num = len(new_sample_ents)
        # non_sample_ents = sorted(list(set(complete_ents) - set(sample_ents)), key=lambda x: x[0], reverse=False)
        instances = []
        adver_sentence = deepcopy(sentence)
        adver_entities = []
        ent_right_shift = 0
        ent_candidate_id = 0
        for ent_idx, entity in enumerate(complete_ents):
            if entity in new_sample_ents:
                instance_id = list(entity_candidates[ent_candidate_id].keys())[0]
                instance_title = entity_candidates[ent_candidate_id][instance_id]['instance_title']
                instances.append({instance_id: instance_title})
                instance = instance_title.split()
                ent_len_change = len(instance) - (entity[2]+1-entity[1])
                start_idx = entity[1] + ent_right_shift
                end_idx = entity[2] + ent_right_shift + ent_len_change
                adver_sentence = adver_sentence[:start_idx] + instance + adver_sentence[entity[2]+ent_right_shift+1:]
                ent_right_shift += ent_len_change
                adver_entities.append((entity[0], start_idx, end_idx, entity[3]))
                ent_candidate_id += 1
            else:
                shifted_non_sample_ents[attack_round].append((entity[1]+ent_right_shift,
                                                              entity[2]+ent_right_shift,
                                                              entity[3]))
        adver_sentences[attack_round] = adver_sentence
        adver_entities_by_round[attack_round] = adver_entities
        adver_lists[attack_round] = dict(original=dict(sentence=" ".join(sentence), all_entities=complete_ents,
                                                       sampled_entities=new_sample_ents),
                                         adver=dict(sentence=" ".join(adver_sentence),
                                                    entities=adver_entities,
                                                    shifted_non_sample_entities=shifted_non_sample_ents[attack_round]),
                                         entities_with_derivation=
                                         [dict(entity=sampled_ent,
                                               ent_text=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['ent_text']
                                               if sampled_ent[3] != "PERSON" else " ".join(sentence[sampled_ent[1]:sampled_ent[2]+1]),
                                               linked_title=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['linked_title']
                                               if sampled_ent[3] != "PERSON" else "N/A",
                                               ent_id=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['ent_id']
                                               if sampled_ent[3] != "PERSON" else "N/A",
                                               instance=instance,
                                               derive_from=entity_candidates[ent_idx][list(instance.keys())[0]]['class_info'])
                                          for ent_idx, (sampled_ent, instance) in enumerate(zip(new_sample_ents, instances))]
                                         )
    return adver_sentences, adver_entities_by_round, adver_lists, shifted_non_sample_ents, none_data_sampled_entities


# def entity_level_replace_with_homogeneous_wikiinstances_concurrent(sentence, complete_ents, ent_classes, class_ents, class_instances, args):
def entity_level_replace_with_homogeneous_wikiinstances_concurrent(sentence, complete_ents, sampled_ents, ent_classes,
                                                                   class_ents, class_instances, args):
    # combined_entities = sorted(entities + non_sample_entities, key=lambda x: x[0], reverse=False)
    shifted_non_sample_ents = []
    none_data_sampled_entities = {}
    entity_candidates = choose_entity_candidates(sampled_ents, ent_classes, class_ents, class_instances)
    new_sample_ents = deepcopy(sampled_ents)
    for ent_idx in entity_candidates.keys():
        status = entity_candidates[ent_idx].get("status", None)
        if status:
            print("Entity irreplaceable.")
            none_data_sampled_entities[tuple(sampled_ents[ent_idx])] = status
            new_sample_ents.remove(sampled_ents[ent_idx])
    ent_candidate_ids = sorted([candidate_id for candidate_id in entity_candidates.keys()
                                if not entity_candidates[candidate_id].get("status")])
    entity_candidates = {k: entity_candidates[ent_candidate_ids[k]] for k in range(len(new_sample_ents))}
    instances = []
    adver_sentence = deepcopy(sentence)
    adver_entities = []
    ent_right_shift = 0
    ent_candidate_id = 0
    for ent_idx, entity in enumerate(complete_ents):
        if entity in new_sample_ents:
            instance_id = list(entity_candidates[ent_candidate_id].keys())[0]
            instance_title = entity_candidates[ent_candidate_id][instance_id]['instance_title']
            instances.append({instance_id: instance_title})
            instance = instance_title.split()
            ent_len_change = len(instance) - (entity[2]+1-entity[1])
            start_idx = entity[1] + ent_right_shift
            end_idx = entity[2] + ent_right_shift + ent_len_change
            adver_sentence = adver_sentence[:start_idx] + instance + adver_sentence[entity[2]+ent_right_shift+1:]
            ent_right_shift += ent_len_change
            if entity[3] != "NORP":
                adver_entities.append((entity[0], start_idx, end_idx, entity[3]))
            else:
                instance_class_id = entity_candidates[ent_candidate_id][instance_id]['class_info']['class_id']
                if instance_class_id not in class_instances['GPE'].keys():
                    adver_entities.append((entity[0], start_idx, end_idx, entity[3]))
                else:
                    adver_entities.append((entity[0], start_idx, end_idx, "GPE"))
            ent_candidate_id += 1
        else:
            shifted_non_sample_ents.append((entity[1]+ent_right_shift, entity[2]+ent_right_shift, entity[3]))
    adver_list = dict(original=dict(sentence=" ".join(sentence), all_entities=complete_ents,
                                    sampled_entities=new_sample_ents),
                      adver=dict(sentence=" ".join(adver_sentence),
                                 entities=adver_entities,
                                 shifted_non_sample_entities=shifted_non_sample_ents),
                      entities_with_derivation=[
                          dict(entity=sampled_ent,
                               ent_text=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['ent_text']
                               if sampled_ent[3] != "PERSON" else " ".join(sentence[sampled_ent[1]:sampled_ent[2]+1]),
                               linked_title=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['linked_title']
                               if sampled_ent[3] != "PERSON" else "N/A",
                               ent_id=ent_classes[sampled_ent[3]][tuple(sampled_ent)]['ent_id']
                               if sampled_ent[3] != "PERSON" else "N/A",
                               instance=instance,
                               derive_from=entity_candidates[ent_idx][list(instance.keys())[0]]['class_info'])
                          for ent_idx, (sampled_ent, instance) in enumerate(zip(new_sample_ents, instances))
                      ])
    return adver_sentence, adver_entities, adver_list, shifted_non_sample_ents
