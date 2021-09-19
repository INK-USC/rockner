import os


models = ["spacy_lg", "stanza", "blstm_crf", "bert_crf", "self_trained_flair", "roberta_crf",
          "roberta_crf_aug_entity_switching", "roberta_crf_aug_random_masking", "roberta_crf_aug_mixing_up"]

precision = {model: [] for model in models}
recall = {model: [] for model in models}
f1_scores = {model: [] for model in models}


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"

for model in models:
    model_file_folder_name = model + "_entity_5attacks_concurrent_unseen_deleted"
    main_path = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + model_file_folder_name + "/"
    baseline_path = main_path + DATASET_TYPE + ".f1_report.txt"
    ent_path = main_path + "f1_reports/4.f1_report.txt"
    cxt_path = main_path + "context_replaced_robustness/context_replaced_filtered.f1_report.txt"
    both_path = main_path + "context_replaced_robustness/both_replaced_filtered.f1_report.txt"

    for path in [baseline_path, ent_path, cxt_path, both_path]:
        if not os.path.exists(path):
            precision[model].append("_")
            recall[model].append("_")
            f1_scores[model].append("_")
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                data = lines[2].strip()
                prec_start = data.find("precision:  ")+len("precision:  ")
                prec = str(data[prec_start:prec_start+5])

                recall_start = data.find("recall:  ")+len("recall:  ")
                rc = str(data[recall_start:recall_start+5])

                f1_start = data.find("FB1:  ")+len("FB1:  ")
                f1 = str(data[f1_start:f1_start+5])

                precision[model].append(prec)
                recall[model].append(rc)
                f1_scores[model].append(f1)


with open(DATASET_NAME + "." + DATASET_TYPE + ".robustness_table.txt", "w") as f:
    for model in models:
        f.write(model + "," + ",".join(precision[model]) + "," + ",".join(recall[model])
                + "," + ",".join(f1_scores[model]) + '\n')
