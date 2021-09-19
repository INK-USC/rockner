import sys
import json
import wikidata_tools
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
CLASS_INSTANCES_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/class_instances/"
FLAIR_PROCESSED_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/flair_processed_class_instances/"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]



def predict_with_flair(sentence):
    global tagger
    predictions = []
    flair_sentence = Sentence(sentence)
    tagger.predict(flair_sentence)
    spans = flair_sentence.get_spans('ner')
    for ent in spans:
        start_index = ent.tokens[0].idx - 1
        end_index = ent.tokens[-1].idx - 1
        predicted_entity = (start_index, end_index, ent.tag)
        predictions.append(predicted_entity)
    return predictions


def verify(predictions, sentence, ent_type):
    # entity = (0, len(sentence.split())-1, ent_type)
    for prediction in predictions:
        if prediction[2] == ent_type and 0 <= prediction[0] <= prediction[1] <= len(sentence.split())-1:
            return True
    return False


def init_models():
    global tagger
    tagger = SequenceTagger.load('ner-ontonotes-fast')


def process_data_with_flair(class_instances, ent_type):
    processed_class_instances = {}
    removable_instance_id = set()
    for class_id, data in tqdm(class_instances[ent_type].items()):
        processed_class_instances[class_id] = {"class_title": data['class_title'], "instances": {}}
        for instance_id, instance_title in tqdm(data['instances'].items()):
            if instance_id in removable_instance_id:
                continue
            try:
                flair_predictions = predict_with_flair(instance_title)
            except Exception as e:
                removable_instance_id.add(instance_id)
                print(e)
                continue
            if verify(flair_predictions, instance_title, ent_type):
                removable_instance_id.add(instance_id)
            else:
                processed_class_instances[class_id]["instances"][instance_id] = instance_title
    processed_class_instances = {k: {"class_title": v['class_title'],
                                     "instance_num": len(v['instances']),
                                     "instances": v['instances']}
                                 for k, v in sorted(processed_class_instances.items(),
                                                    key=lambda x: len(x[1]['instances']),
                                                    reverse=True)}
    return processed_class_instances


def output_processed_class_instances(processed_class_instances, ent_type):
    with open(FLAIR_PROCESSED_PATH + ent_type + '.jsonl', 'w+') as f:
        for class_id, data in processed_class_instances.items():
            f.write(json.dumps(dict(class_id=class_id,
                                    class_title=data['class_title'],
                                    instance_num=data['instance_num'],
                                    instances=data['instances']))
                    + '\n')


def main():
    ent_type = sys.argv[1]
    init_models()
    class_instances = wikidata_tools.load_class_instances(CLASS_INSTANCES_PATH)
    processed_class_instances = process_data_with_flair(class_instances, ent_type)
    output_processed_class_instances(processed_class_instances, ent_type)


if __name__ == '__main__':
    main()
