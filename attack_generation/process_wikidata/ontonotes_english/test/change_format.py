import json
import sys

sys.path.append("../attack_generation")
import adver_tools

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
DATA_FILE_PATH = "../../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"

def main():
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    for ent_type in ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']:
    # for ent_type in ['GPE', 'ORG', 'FAC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART']:
        api_results = {}
        class_ents = {}
        entity_linked_title = {}
        entity_wikidata = {}
        ent_classes = {}
        n_class_ents = {}

        # with open("./backup_all_data/class_data/" + ent_type + ".ent_classes.jsonl", "r") as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         ent_id = data['ent_id']
        #         classes = data['classes']
        #         ent_text = " ".join(all_sents[ent[0]][ent[1]:ent[2]+1])
        #         linked_title = entity_linked_title[tuple(ent)]
        #         ent_classes[tuple(ent)] = dict(ent_text=ent_text,
        #                                        linked_title=linked_title,
        #                                        ent_id=ent_id,
        #                                        classes=classes)
        #
        # with open("./class_data/" + ent_type + ".ent_classes.jsonl", "w+") as f:
        #     for ent, data in ent_classes.items():
        #         f.write(json.dumps(dict(entity=ent, ent_text=data['ent_text'], linked_title=data['linked_title'],
        #                                 ent_id=data['ent_id'], classes=data['classes'])) + '\n')

        with open("./backup_all_data/class_data/" + ent_type + ".class_ents.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                class_id = data['class_id']
                class_title = data['class_title']
                entities = data['entities']
                QID_num = len(entities.keys())
                ent_num = sum([len(v) for v in entities.values()])
                class_ents[class_id] = dict(class_title=class_title, ent_num=ent_num, QID_num=QID_num, entities=entities)
        with open("./class_data/" + ent_type + ".class_ents.jsonl", 'w+') as f:
            class_ents = {k: v for k, v in sorted(class_ents.items(), key=lambda x: x[1]['ent_num'], reverse=True)}
            for class_id, data in class_ents.items():
                f.write(json.dumps(dict(class_id=class_id,
                                        class_title=data['class_title'],
                                        ent_num=data['ent_num'],
                                        QID_num=data['QID_num'],
                                        entities=data['entities'])) + '\n')


        # with open("./backup_all_data/api_results/" + ent_type + ".linked_wikiapi.jsonl", 'r') as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         api_results[tuple(ent)] = dict(ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
        #                                        linked_title=data['linked_title'],
        #                                        result=data['result'])
        # with open("./api_results/" + ent_type + ".linked_wikiapi.jsonl", 'w+') as f:
        #     for ent, data in api_results.items():
        #         f.write(json.dumps(dict(entity=ent,
        #                                 ent_text=data['ent_text'],
        #                                 linked_title=data['linked_title'],
        #                                 result=data['result'])) + '\n')
        # with open("./backup_all_data/entity_wikidata_pair/" + ent_type + ".linked_wikidata.jsonl", "r") as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         linked_title = data['linked_title']
        #         wikidata = data['wikidata']
        #         entity_linked_title[tuple(ent)] = linked_title
        #         entity_wikidata[tuple(ent)] = dict(ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
        #                                            linked_title=linked_title, wikidata=wikidata)
        # with open("./entity_wikidata_pair/" + ent_type + ".linked_wikidata.jsonl", "w+") as f:
        #     for ent, data in entity_wikidata.items():
        #         f.write(json.dumps(dict(entity=ent, ent_text=data['ent_text'], linked_title=data['linked_title'],
        #                                 wikidata=data['wikidata'])) + '\n')
        #
        # with open("./backup_all_data/class_data/" + ent_type + ".class_ents.jsonl", 'r') as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         class_id = data['class_id']
        #         class_title = data['class_title']
        #         entities = data['entities']
        #         new_entities = {}
        #         for ent_id, ent_lists in entities.items():
        #             new_entities[ent_id] = [dict(entity=ent,
        #                                          ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
        #                                          linked_title=entity_linked_title[tuple(ent)]) for ent in ent_lists]
        #         class_ents[class_id] = dict(class_title=class_title, entities=new_entities)
        # with open("./class_data/" + ent_type + ".class_ents.jsonl", 'w+') as f:
        #     for class_id, data in class_ents.items():
        #         f.write(json.dumps(dict(class_id=class_id,
        #                                 class_title=data['class_title'],
        #                                 entities=data['entities'])) + '\n')

        # with open("./backup_all_data/class_data/" + ent_type + ".ent_classes.jsonl", "r") as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         ent_id = data['ent_id']
        #         classes = data['classes']
        #         ent_text = " ".join(all_sents[ent[0]][ent[1]:ent[2]+1])
        #         linked_title = entity_linked_title[tuple(ent)]
        #         ent_classes[tuple(ent)] = dict(ent_text=ent_text,
        #                                        linked_title=linked_title,
        #                                        ent_id=ent_id,
        #                                        classes=classes)
        #
        # with open("./class_data/" + ent_type + ".ent_classes.jsonl", "w+") as f:
        #     for ent, data in ent_classes.items():
        #         f.write(json.dumps(dict(entity=ent, ent_text=data['ent_text'], linked_title=data['linked_title'],
        #                                 ent_id=data['ent_id'], classes=data['classes'])) + '\n')

        # with open("./backup_all_data/none_class_entities/" + ent_type + ".jsonl", 'r') as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         ent_id = data['ent_id']
        #         n_class_ents[tuple(ent)] = dict(ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
        #                                         linked_title=entity_linked_title[tuple(ent)],
        #                                         ent_id=ent_id)
        #
        # with open("./none_class_entities/" + ent_type + ".jsonl", 'w+') as f:
        #     for ent, data in n_class_ents.items():
        #         f.write(json.dumps(dict(entity=ent, ent_text=data['ent_text'], linked_title=data['linked_title'],
        #                            ent_id=data['ent_id'])) + '\n')

        # with open("./backup_all_data/none_id_entities/" + ent_type + ".linked_.jsonl", 'r') as f:
        #     for line in f:
        #         data = json.loads(line.strip())
        #         ent = data['entity']
        #         linked_title = data['linked_title']
        #         n_class_ents[tuple(ent)] = dict(ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
        #                                         linked_title=linked_title)
        #
        # with open("./none_id_entities/" + ent_type + ".jsonl", 'w+') as f:
        #     for ent, data in n_class_ents.items():
        #         f.write(json.dumps(dict(entity=ent, ent_text=data['ent_text'], linked_title=data['linked_title'])) + '\n')


if __name__ == '__main__':
    main()
