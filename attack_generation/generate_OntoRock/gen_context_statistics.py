import os


test_path = "../../OntoRock/test/OntoRock-C/filtered_log.txt"
dev_path = "../../OntoRock/dev/OntoRock-C/filtered_log.txt"

paths = [test_path, dev_path]

for path in paths:
    print(path)
    all_sent_counter = 0
    all_token_counter = 0
    atk_sent_counter = 0
    atk_token_counter = 0
    with open(path, 'r') as f:
        while True:
            ori_sent = f.readline().strip().split()
            rpl_sent = f.readline().strip().split()
            if len(ori_sent) == 0 and len(rpl_sent) == 0:
                break
            all_sent_counter += 1
            atk_token_num = 0
            f.readline()
            f.readline()
            for ori_token, rpl_token in zip(ori_sent, rpl_sent):
                all_token_counter += 1
                if ori_token != rpl_token:
                    atk_token_num += 1
            if atk_token_num != 0:
                atk_sent_counter += 1
                atk_token_counter += atk_token_num
    print(all_token_counter, atk_token_counter, all_sent_counter, atk_sent_counter)
