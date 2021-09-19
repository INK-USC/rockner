import json


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
START_IDX_LIST = {"test": [0, 3000], "dev": [0, 3000],
                  "train": [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000, 36000, 39000]}
ENTITY_NUM = {'test': 5461, 'dev': 5486, 'train': 39712}



def main():
    all_results = {}
    entity_lists = []
    for start_idx in START_IDX_LIST[DATASET_TYPE]:
        end_idx = min(start_idx + 3000, ENTITY_NUM[DATASET_TYPE])
        with open(DATASET_NAME + "/" + DATASET_TYPE
                  + "/titles/blink_results." + str(start_idx) + "_" + str(end_idx) + ".jsonl", "r") as f:
            for line in f:
                data = json.loads(line.strip())
                entity = data['entity']
                prediction = data['prediction']
                score = data['score']
                entity_lists.append(entity)
                all_results[tuple(entity)] = {"prediction": prediction, "score": score}
    with open(DATASET_NAME + "/" + DATASET_TYPE + "/titles/blink_results.jsonl", "w+") as f:
        for entity in entity_lists:
            f.write(json.dumps(dict(entity=entity,
                                    prediction=all_results[tuple(entity)]['prediction'],
                                    score=all_results[tuple(entity)]['score']))
                    + '\n')


if __name__ == "__main__":
    main()
