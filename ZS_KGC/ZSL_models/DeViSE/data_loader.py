import json, os
import random




def random_pick(some_list, probabilities):
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:break
    return item



def load_train_data(args, train_tasks, symbol2id, ent2id, e1rel_e2, rel2id, rela2label, rela_matrix):
    print('##LOADING CANDIDATES')
    rel2candidates = json.load(open(os.path.join(args.data_path, 'rel2candidates_all.json')))
    task_pool = sorted(train_tasks.keys())  # ensure the readout is the same

    # print(task_pool)

    while True:
        rel_batch, rel_neg_batch, query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels = [], [], [], [], [], [], [], [], []
        random.shuffle(task_pool)
        if len(rel2candidates[task_pool[0]]) <= 20:
            continue
        if len(rel2candidates[task_pool[1]]) <= 20:
            continue
        for query in task_pool[:args.batch_rela_num]:
            # print(query)
            relation_id = rel2id[query]
            candidates = rel2candidates[query]

            if args.dataset == 'Wiki':
                if len(candidates) <= 20:
                    # print 'not enough candidates'
                    continue

            train_and_test = train_tasks[query]

            random.shuffle(train_and_test)

            all_test_triples = train_and_test

            if len(all_test_triples) == 0:
                continue


            if len(all_test_triples) < args.batch_size:
                query_triples = [random.choice(all_test_triples) for _ in range(args.batch_size)]
            else:
                query_triples = random.sample(all_test_triples, args.batch_size)

            query_pairs += [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left += [ent2id[triple[0]] for triple in query_triples]
            query_right += [ent2id[triple[2]] for triple in query_triples]

            label = rela2label[query]

            # generate negative samples
            false_pairs_ = []
            false_left_ = []
            false_right_ = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if noise in ent2id.keys(): # ent2id.has_key(noise):
                        if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                            break
                false_pairs_.append([symbol2id[e_h], symbol2id[noise]])
                false_left_.append(ent2id[e_h])
                false_right_.append(ent2id[noise])

            false_pairs += false_pairs_
            false_left += false_left_
            false_right += false_right_

            rel_batch += [rel2id[query] for _ in range(args.batch_size)]

            neg_rel_batch = list()
            for _ in range(args.batch_size):
                while True:
                    neg_query = random.choice(task_pool)
                    if neg_query != query:
                        break

                neg_rel_batch.append(rel2id[neg_query])

            rel_neg_batch += neg_rel_batch


            labels += [rela2label[query]] * args.batch_size

        yield rela_matrix[rel_batch], rela_matrix[rel_neg_batch], query_pairs, query_left, query_right, false_pairs, false_left, false_right, labels
