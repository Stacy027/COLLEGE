import os, random
from transformers import RobertaTokenizer
from tqdm import tqdm
import nltk
# from tag import Annotate
import torch
import networkx as nx
from itertools import combinations
import json
import numpy as np
from multiprocessing import Pool
import sys
import os.path

import multiprocessing

import traceback

def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
    pass

#---------------------------------------#
#   seed
#---------------------------------------#
random_seed = 2022
random.seed(random_seed)

input_folder = "./pretrain_txt"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print(len(file_list),'# of files')
file_list.sort()


# def print_error(value):
#     print("error: ", value)

def load_data():
    entity2qid, qid2entity = {}, {}
    with open('../wikidata/wikidata5m_entity.txt', 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            l = line.strip().split('\t')
            if len(l) < 2:
                continue
            qid = l[0]
            for entity in l[1:]:
                qid2entity[qid] = entity
                entity2qid[entity] = qid
    print('wikidata5m_entity.txt (Wikidata5M) loaded!')

    relation2pid, pid2relation = {}, {}
    with open('../wikidata/wikidata5m_relation.txt', 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            r = line.strip().split('\t')
            if len(r) < 2:
                continue
            pid = r[0]
            pid2relation[pid] = r[1] 
            for rel in r[1:]:
                relation2pid[rel] = pid
    print('wikidata5m_relation.txt (Wikidata5M) loaded!')


    # This is to remove FewRel test set from our training data. If your need is not just reproducing the experiments,
    # you can discard this part. The `ernie_data` is obtained from https://github.com/thunlp/ERNIE
    fewrel_triples = set()
    with open('../wikidata/fewrel/test.json', 'r', encoding='utf-8') as fin:
        fewrel_data = json.load(fin)
        for ins in fewrel_data:
            r = ins['label']
            h, t = ins['ents'][0][0], ins['ents'][1][0]
            fewrel_triples.add((h, r, t))
    print('# triples in FewRel test set: {}'.format(len(fewrel_triples)))
    print(list(fewrel_triples)[0])

    head_cluster, tail_cluster = {}, {}
    head_tail_cluster = {}
    num_del = total = 0

    with open("../wikidata/wikidata5m_all_triplet.txt", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            tp = line.strip().split('\t')
            if len(tp) != 3:
                continue
            h, r, t = tp
            if (h, r, t) not in fewrel_triples:
                head_tail_cluster[(h, t)] = r
                if h in head_cluster:
                    head_cluster[h].append((r, t))
                else:
                    head_cluster[h] = [(r, t)]
                if t in tail_cluster:
                    tail_cluster[t].append((r, h))
                else:
                    tail_cluster[t] = [(r, h)]
                
            else:
                num_del += 1
            total += 1
    print(total, '- wikidata5m_triplet.txt (Wikidata5M) loaded!')
    print('deleted {} triples from Wikidata5M.'.format(num_del))
    return entity2qid, qid2entity, relation2pid, pid2relation, head_cluster, tail_cluster, head_tail_cluster


entity2qid, qid2entity, relation2pid, pid2relation, head_cluster, tail_cluster, head_tail_cluster = load_data()
max_extra_nodes = 200
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# wiki = nx.read_gpickle('./wikidata_alias.graph')
        
def wikinodes_add_relations(nodes):
    
    triplet_to_add = []
    entity_nodes = []
    for s in nodes:
        for t in nodes:
            flag = 0
            if (s, t) in head_tail_cluster:
                r = head_tail_cluster[(s,t)]
                if r in pid2relation:
                    flag = 1
            elif (t, s) in head_tail_cluster:
                r = head_tail_cluster[(t, s)]
                if r in pid2relation:
                    flag = 1
                # for e_attr in wiki[s][t].values():
            if flag == 1:
                entity_nodes.append(s)
                entity_nodes.append(t)
#                     print(s,t,e_attr['rel'])
                triplet_to_add.append((s, t, r))
    return triplet_to_add, entity_nodes

def find_extra_nodes(qids):
    bridge_nodes2 = []
    for q1 in qids:
        for q2 in qids:
            if q1 != q2 and q1 in qid2entity and q2 in qid2entity:
                triplets1 = []
                if q1 in head_cluster:
                    triplets1 = triplets1+ head_cluster[q1]
                if q1 in tail_cluster:
                    triplets1 = triplets1+ tail_cluster[q1]
                            
                triplets2 = []
                if q2 in head_cluster:
                    triplets2 = triplets2 + head_cluster[q2]
                if q2 in tail_cluster:
                    triplets2 = triplets2 + tail_cluster[q2]
                            
                bridge_nodes = set()
                q1_tails = [t for (r,t) in triplets1]
                q2_tails = [t for (r,t) in triplets2]
                bridge_nodes |= set(q1_tails) & set(q2_tails)
                
                for qid in bridge_nodes:
                    if qid in qid2entity:
                        bridge_nodes2.append(qid)
    return bridge_nodes2

def run_proc(index, n, file_list, min_seq_len=80, max_seq_len=200, n_samples_per_file=5000):
    output_folder_graph = '/share/project/yxw/output_graph_/' + str(index)
    if not os.path.exists(output_folder_graph):
        os.makedirs(output_folder_graph)
    output_folder_text = '/share/project/yxw/output_text_/' + str(index)
    if not os.path.exists(output_folder_text):
        os.makedirs(output_folder_text)
    j = index
    drop_samples = 0
    n_normal_data = 0
    is_large = False
    target_filename_graph = os.path.join(output_folder_graph, str(j))
    target_filename_text = os.path.join(output_folder_text, str(j))
    fout1 = open(target_filename_graph, 'w', encoding='utf-8')
    fout2 = open(target_filename_text, 'w', encoding='utf-8')
    for i in range(len(file_list)):
        if i % n == index:
            input_name = file_list[i]
            # print(input_name)
            # target_filename_graph = input_name.replace('/sharefs/yxw/output_text_filter', "/sharefs/yxw/output_graph_2hop")
            # fout1 = open(target_filename_graph, 'w', encoding='utf-8')
            print('[processing] # {}/{}: {}'.format(i, index, input_name))
            with open(input_name, 'r', encoding='utf-8') as fin:
                
                lines = fin.readlines()
                print("doc num:",len(lines))
            
                for line in lines:
                    segs      = line.strip().split("[_end_]")
                    content   = segs[0]  
                    map_ent   = segs[1:]
                    sentences = nltk.sent_tokenize(content)
                    
                    maps  = {}  # key-> value: entity mention -> QID
                    alias = {}
                    for x in map_ent:
                        v = x.split("[_map_]")  
                        if len(v) != 2:
                            continue
            
                        if v[1] in entity2qid:  
                            maps[v[0]]  = entity2qid[v[1]]   # v[0]: mentioned entity, v[1]: entity alias
                            alias[v[0]] = v[1]
                        elif v[1].lower() in entity2qid:
                            maps[v[0]]  = entity2qid[v[1].lower()]
                            alias[v[0]] = v[1].lower()

                    blocks, word_lst = [], []
                    s = ''
                    for sent in sentences:
                        s = '{} {}'.format(s, sent)
                        word_lst = tokenizer.encode(s)
                        if len(word_lst) >= min_seq_len:
                            blocks.append(s)
                            s = ''
        
                    for block in blocks:
                        entity_qid             = []  # store QID
                        entity_mentioned       = []
                        entity_alias           = [] # store entity alias
  
                        # pos                    = 1
                        text                   = ''
                        # soft_positions         = [0]
                        # type_ids               = []
                        # entity_start_positions = []
                        # entity_end_positions   = []
                        # words_ids              = [tokenizer.cls_token_id]
                        anchors                = [x.strip() for x in block.split("sepsepsep")]

                        #-----------------------#
                        # 实体链接
                        #-----------------------#
                        for x in anchors:
                            if len(x) < 1:
                                continue
                            else:
                                # words = tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True)
                                # words = words[:max_seq_len]
                                text = '{} {}'.format(text, x)
                                if x in maps and maps[x] not in entity_qid:
                                    entity_qid.append(maps[x])
                                    entity_mentioned.append(x)
                                    entity_alias.append(alias[x])
                                #     entity_start_positions.append(pos)
                                #     entity_end_positions.append(pos+len(words))
                    
                        # print(text)
                        # print(txt_nodes)
                           

                        if len(entity_qid) <= 1:  # no entity contained in the block
                            continue
                        else:
                            text_ids = tokenizer.encode(text, add_special_tokens=True, add_prefix_space=True)
                            text_ids = text_ids[:max_seq_len]
                            ins_text = {'input_ids': text_ids, 'ents': entity_qid}
                            # n_normal_data += 1
                            # fout.write(json.dumps(ins) + '\n') 


                            #-----------#
                            # graph
                            #-----------#
                            # line = json.loads(line)
                            # qids_ori = line['ents']
                        qids = []
                        for qid in entity_qid:
                            if qid in qid2entity:
                                qids.append(qid)
                        words_ids = [tokenizer.cls_token_id]
                        token_types_ids = [0]
                        if len(qids) > max_extra_nodes:
                            qids = qids[:max_extra_nodes//2]
                    
                        bridge_nodes = find_extra_nodes(qids)
                        if len(bridge_nodes) > max_extra_nodes:
                            bridge_nodes = bridge_nodes[:max_extra_nodes]
                        entity_nodes = set(qids + bridge_nodes)
                    
        
                        triplet_to_add, triplet_entities = wikinodes_add_relations(entity_nodes)
                        n_word_nodes = 1  # cxt node
                        token_types = [0]
                        # position_ids = [0]
                        G = nx.complete_graph(n_word_nodes)   # n(idx) nodes
        
                        qid2node = {}  # {key:qid, value:[e_node_alias]}
                        for qid in entity_nodes:
                            if qid not in qid2entity:
                                print(qid," is not in qid2entity")
                                continue
            
                            is_mentioned = False
                            e_nodes = []
                            if qid in qids:
                                is_mentioned = True
                            elif qid not in triplet_entities:
                                continue
                            entity_token = tokenizer.encode(qid2entity[qid], add_special_tokens=False, add_prefix_space=False)
                            # print(entity_token)
                
                            for t in entity_token:
                
                                e_node_alias = qid + '_' + str(t)
                                if G.has_node(e_node_alias):
                                    e_node_alias = e_node_alias + '_' + str(random.randint(1,2022))  # unique id
                                G.add_node(e_node_alias)
                                e_nodes.append(e_node_alias)
                                words_ids.append(t)
                                token_types_ids.append(1) 
                                
                                
                                if is_mentioned:
                                    G.add_edge(0, e_node_alias) # connected with cxt_node 
                                #     position_ids.append(1) 
                                # else:
                                #     position_ids.append(3) 
                            qid2node[qid] = e_nodes 
                            e_combins = [c for c in  combinations(e_nodes, 2)]  # add edges into entity span
                            G.add_edges_from(e_combins)
                    
            
                        relation_set = []
                        pid2node = {}  # {key:pid, value:[r_node_alias]}
                        if len(triplet_to_add) > 0:
                            for (h,t,pid) in triplet_to_add:
                                if pid not in pid2relation:
                                    print(pid," is not in pid2relation")
                                    continue
                                is_inserted = False
                                relation_token = tokenizer.encode(pid2relation[pid], add_special_tokens=False, add_prefix_space=False)
                                r_nodes = []
                                if pid in relation_set:
                                    is_inserted = True
                                else:
                                    relation_set.append(pid)
                                    # print(pid)
                                    for tk in relation_token:
                                        r_node_alias = pid + '_' + str(tk) 
                                        if G.has_node(r_node_alias):
                                            r_node_alias = r_node_alias + '_' + str(random.randint(1,2022))  # unique id
                                        G.add_node(r_node_alias)
                                        r_nodes.append(r_node_alias)
                                        # position_ids.append(2) 
                                        words_ids.append(tk)
                                        token_types_ids.append(2)
                                    pid2node[pid] = r_nodes 
                                head_realtion = pid2node[pid] + qid2node[h]
                                tail_realtion = pid2node[pid] + qid2node[t]
                                # print(head_realtion)
                                # print(tail_realtion)
                                r_combins = [c for c in  combinations(head_realtion, 2)] + [c for c in  combinations(tail_realtion, 2)] 
                                G.add_edges_from(r_combins)
        
                        position_ids = []
                        for node in G.nodes:
                            try:
                                relative = nx.shortest_path_length(G, source=0, target=node)
                                position_ids.append(relative)
                            except:
                                continue
                        #     # print(triplet_to_add)
                        #     # print(line['input_ids'])
                        #         # relative = 99
                        #         # continue
                        #     # print(G.nodes)
                        #     # print(words_ids)

                            
                        if len(G.nodes) != len(words_ids):
                            print('[warning] number of nodes does not match length of words_ids')
                            continue
                        if len(G.nodes) != len(token_types_ids):
                            print('[warning] number of nodes does not match length of token_types')
                            continue
                        if len(G.nodes) != len(position_ids):
                            print('[warning] number of nodes does not match length of position_ids')
                            continue
                        
                  
                        adj = np.array(nx.adjacency_matrix(G).todense())
                        adj = adj + np.eye(adj.shape[0], dtype=int)
                  
        
                        ins = {'nodes': [ids for ids in words_ids], 'position_ids': position_ids, 'adj': adj.tolist(),
                                'token_type_ids': token_types_ids}
                    # fout1.write(json.dumps(j) + '\n')
                        n_normal_data += 1

                        fout1.write(json.dumps(ins) + '\n')
                        fout2.write(json.dumps(ins_text) + '\n')
                        if n_normal_data >= n_samples_per_file:
                            n_normal_data = 0
                            fout1.close()
                            fout2.close()
                            j += 1
                            target_filename_graph = os.path.join(output_folder_graph, str(j))
                            target_filename_text = os.path.join(output_folder_text, str(j))
                            # fout_normal = open(target_filename, 'a+', encoding='utf-8')
                            fout1 = open(target_filename_graph, 'w', encoding='utf-8')
                            fout2 = open(target_filename_text, 'w', encoding='utf-8')
                            
    fout1.close()
    fout2.close()


multiprocessing.log_to_stderr()  

n = int(sys.argv[1])

p = Pool(n)
for i in range(n):
    p.apply_async(LogExceptions(run_proc), args=(i, n, file_list)) 
# run_proc(0, 1, file_list)
p.close()
p.join()