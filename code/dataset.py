import collections
import random
import numpy as np
from torch.utils.data import Dataset
import json
import torch
from transformers import RobertaTokenizer

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_mlm_labels(tokens, mask_index, vocab_size, masked_lm_prob=0.15, anchor_nodes=None):

    rng = random.Random(2022)
    cand_indexes = []
    if mask_index == 50264:  # indicates word nodes
        special_tokens = [0, 1, 2, 3]  # 0: <pad>, 1: <cls>, 2: <sep>, 3: <unk>
    else:
        print("unknown mask token")
        special_tokens = [0, 1]  # 0: <unk> 1: <pad>
    for (i, token) in enumerate(tokens):
        if token in special_tokens:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))
    masked_labels = []
    covered_indexes = set()
    for index in cand_indexes:
        if anchor_nodes is None: 
            if len(masked_labels) >= num_to_predict:
                break
            else:
                if index in covered_indexes:
                    continue
                covered_indexes.add(index)
                if rng.random() < 0.9:
                # if True:
                    masked_token = mask_index  # [MASK]
                else:
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    else:
                        masked_token = rng.randint(0, vocab_size - 1)
        else:
            if len(masked_labels) >= num_to_predict:
                break
            else:
                if index in covered_indexes:
                    continue
                covered_indexes.add(index)
                if tokens[index] in anchor_nodes:
                    if rng.random() <= 1.0:
                    # if rng.random() < 0.9:
                        masked_token = mask_index
                    else:
                        if rng.random() < 0.5:
                            masked_token = tokens[index]
                        else:
                            masked_token = rng.randint(0, vocab_size - 1)
                else:
                    continue
            # elif tokens[index] not in anchor_nodes:
            #     continue
            # else: # tokens[index] is anchor node
            #     if index in covered_indexes:
            #         continue
            #     covered_indexes.add(index)
            #     if rng.random() < 0.9:
            #         masked_token = mask_index
            #     else:
            #         if rng.random() < 0.5:
            #             masked_token = tokens[index]  # 以5%概率是本身
            #         else:
            #             masked_token = rng.randint(0, vocab_size - 1)
        output_tokens[index] = masked_token
        masked_labels.append(MaskedLmInstance(index=index, label=tokens[index]))
    masked_labels = sorted(masked_labels, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_labels:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    masked_labels = np.ones(len(tokens), dtype=int) * -1
    masked_labels[masked_lm_positions] = masked_lm_labels
    masked_labels = list(masked_labels)
    return output_tokens, masked_labels

class Dataset(Dataset):
    def __init__(self, text_list, graph_list, rank, n_workers, word_mask_index, word_vocab_size):
        self.text_list = text_list
        self.graph_list = graph_list
        self.word_mask_index = word_mask_index
        self.word_vocab_size = word_vocab_size

        # self.data= self.get_data()

        file_per_process = len(text_list) // n_workers
        if file_per_process * n_workers != len(text_list):
            if rank == 0:
                print('Drop {} files.'.format(len(text_list) - file_per_process * n_workers))
                print('# files per process: {}'.format(file_per_process))
        self.fps_t = text_list[rank * file_per_process:(rank + 1) * file_per_process]
        self.fps_g = graph_list[rank * file_per_process:(rank + 1) * file_per_process]

        self.current_file_idx = 0
        self.data = self.read_file(self.current_file_idx)
        self.num_samples_per_file = len(self.data)
        self.total_num_samples = self.num_samples_per_file * len(self.fps_t)

    def __len__(self):
        return self.total_num_samples

    def __getitem__(self, item: int):
        file_idx = item // self.num_samples_per_file
        if file_idx != self.current_file_idx:
            self.data = self.read_file(file_idx)
            self.current_file_idx = file_idx
        sample = self.data[item - file_idx * self.num_samples_per_file]

        return sample
    
    def _split_nodes(self, nodes, types):
        assert len(nodes) == len(types)
        cxt, anchors, others = [], [], []
        for node, type in zip(nodes, types):
            if type == 0:
                cxt.append(node)
            elif type == 1:
                anchors.append(node)
            # elif type == 2:
            #     others.append(node)
            else:
                others.append(node)
                # raise ValueError('unknown token type id.')
        return cxt, anchors, others
    
    def read_file(self, idx):
        data = []
        text_size = 0
        with open(self.fps_t[idx], 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            text_size = len(lines)
            for i in range(len(lines)):
                # print(x)
                x = lines[i]
                instance = json.loads(x)
                words = instance['input_ids']
                # [mask id, vocab size]:deberta same as roberta 
                words, word_mlm_labels = create_mlm_labels(words, self.word_mask_index, self.word_vocab_size, masked_lm_prob=0.15)
                attention_mask = [1] * len(words)
                token_type_ids = [0] * len(words)
                position_ids = [i for i in range(len(words))]
            # for x in fin:
                # print(x)
                # text = x.strip()
                new_text = {
                    'input_ids': words,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'masked_lm_labels': word_mlm_labels,
                }
                pair = {
                    'text': new_text,
                    'graph': ''
                }
                data.append(pair)
        with open(self.fps_g[idx], 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            assert text_size == len(lines)  # graph size == text size
            for i in range(len(lines)):
                # print(x)
                x = lines[i]
                instance = json.loads(x)
                words = instance['nodes']
                _, anchors, others = self._split_nodes(instance['nodes'], instance['position_ids'])
                words, word_mlm_labels = create_mlm_labels(words, self.word_mask_index, self.word_vocab_size, masked_lm_prob=1.0, anchor_nodes=anchors)

                # assert len(instance['nodes']) == len(words)
                assert len(instance['nodes']) == len(instance['position_ids'])
                assert len(instance['nodes']) == len(instance['adj'])
                assert len(instance['nodes']) == len(instance['token_type_ids'])
                new_graph = {
                    'input_ids': words,
                    'position_ids': instance['position_ids'],
                    'attention_mask': instance['adj'],
                    'token_type_ids': instance['token_type_ids'],
                    'masked_lm_labels': word_mlm_labels,
                    # 'ent_masked_lm_labels': entity_mlm_labels,
                    # 'rel_masked_lm_labels': relation_mlm_labels
                }
                
                data[i]['graph'] = new_graph
        # print(data)
        return data



token1 = RobertaTokenizer.from_pretrained('roberta-base')
WORD_PADDING_INDEX_ROBERTA = token1.pad_token_id

WORD_MAX_LENGTH = 512
KG_MAX_LENGTH = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def collate_fn(data):
    text = [i['text'] for i in data]
    graph = [i['graph'] for i in data]

    # text
    text_input_keys = ['input_ids', 'position_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels']
    text_target_keys = ['masked_lm_labels', 'word_seq_len']
    max_text_len = 0
    batch_text = []

    batch_text_x = {n: [] for n in text_input_keys}
    batch_text_y = {n: [] for n in text_target_keys}

    for sample in text:
        for n, v in sample.items():
            if n in text_input_keys:
                batch_text_x[n].append(v)
            if n in text_target_keys:
                batch_text_y[n].append(v)

        n_word_nodes = len(sample['input_ids'])
        # batch_text_x['n_word_nodes'] = n_word_nodes
        words = sample['input_ids']
        batch_text.append(words)
        max_text_len = len(words) if len(words) > max_text_len else max_text_len

        if max_text_len > WORD_MAX_LENGTH:
            max_text_len = WORD_MAX_LENGTH
            batch_text_y['word_seq_len'].append(max_text_len)
        else:
            batch_text_y['word_seq_len'].append(n_word_nodes)
    # pad
    seq_len = max_text_len
    for i in range(len(batch_text)):
        word_pad = max_text_len - len(batch_text[i])
        n_words = len(batch_text_x['input_ids'][i])

        assert n_words == len(batch_text_x['position_ids'][i])
        assert n_words == len(batch_text_x['token_type_ids'][i])

        if word_pad < 0:
            batch_text_x['input_ids'][i] = batch_text[i][:max_text_len]
            batch_text_x['position_ids'][i] = batch_text_x['position_ids'][i][:max_text_len]
            batch_text_x['token_type_ids'][i] = batch_text_x['token_type_ids'][i][:max_text_len] 
            batch_text_x['attention_mask'][i] = batch_text_x['attention_mask'][i][:max_text_len]
            batch_text_x['masked_lm_labels'][i] = batch_text_x['masked_lm_labels'][i][:max_text_len]
            batch_text_y['masked_lm_labels'][i] = batch_text_y['masked_lm_labels'][i][:max_text_len]
        else:
            batch_text_x['input_ids'][i] = batch_text[i] + [WORD_PADDING_INDEX_ROBERTA] * word_pad     
            batch_text_x['position_ids'][i] = batch_text_x['position_ids'][i] + [0] * word_pad 
            batch_text_x['token_type_ids'][i] = batch_text_x['token_type_ids'][i] + [0] * word_pad 
            batch_text_x['attention_mask'][i] = batch_text_x['attention_mask'][i] + [0] * word_pad 
            batch_text_x['masked_lm_labels'][i] = batch_text_x['masked_lm_labels'][i] + [-1] * word_pad
            batch_text_y['masked_lm_labels'][i] = batch_text_y['masked_lm_labels'][i] + [-1] * word_pad  


        assert len(batch_text_x['attention_mask'][i]) == seq_len
        assert len(batch_text_x['token_type_ids'][i]) == seq_len
        assert len(batch_text_x['position_ids'][i]) == seq_len
        
           

    for k, v in batch_text_x.items():     
        batch_text_x[k] = torch.tensor(v)#.to(device)
    for k, v in batch_text_y.items():
        batch_text_y[k] = torch.tensor(v)#.to(device)    


    # graph
    # data: [[x1:dict, y1:dict], [x2:dict, y2:dict], ...]
    input_keys = ['input_ids', 'position_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels']
    target_keys = ['masked_lm_labels', 'word_seq_len']
    max_word_nodes = 0
    batch_word = []

    batch_x = {n: [] for n in input_keys}
    batch_y = {n: [] for n in target_keys}
    for sample in graph:
        for n, v in sample.items():
            if n in input_keys:
                batch_x[n].append(v)
            if n in target_keys:
                batch_y[n].append(v)
        
        # n_word_nodes = sample['n_word_nodes']
        # n_kg_nodes = sample['n_entity_nodes'] + sample['n_relation_nodes']
        
        # batch_x['n_kg_nodes'].append(n_word_nodes)
        words = sample['input_ids']
        n_word_nodes = len(words)
        batch_word.append(words)


        max_word_nodes = len(words) if len(words) > max_word_nodes else max_word_nodes
        # max_kg_nodes = len(kg) if len(kg) > max_kg_nodes else max_kg_nodes
        if max_word_nodes > WORD_MAX_LENGTH:
            max_word_nodes = WORD_MAX_LENGTH
            batch_y['word_seq_len'].append(max_word_nodes)
        else:
            batch_y['word_seq_len'].append(n_word_nodes)
            
    
    # pad
    seq_len = max_word_nodes
    for i in range(len(batch_word)):
        word_pad = max_word_nodes - len(batch_word[i])
        # kg_pad = max_kg_nodes - len(batch_kg[i]) 
        n_words = batch_y['word_seq_len'][i]
        # n_kg_nodes = batch_x['n_entity_nodes'][i] + batch_x['n_relation_nodes'][i]
        if word_pad >= 0:
            
            batch_x['input_ids'][i] = batch_word[i] + [WORD_PADDING_INDEX_ROBERTA] * word_pad             
            batch_x['position_ids'][i] = batch_x['position_ids'][i] + [0] * word_pad    
            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i] + [0] * word_pad   
            batch_x['masked_lm_labels'][i] = batch_x['masked_lm_labels'][i] + [-1] * word_pad
            batch_y['masked_lm_labels'][i] = batch_y['masked_lm_labels'][i] + [-1] * word_pad       
            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
     
            adj = torch.cat((adj, torch.zeros(word_pad, adj.shape[1], dtype=torch.int)), dim=0)

            assert adj.shape[0] == seq_len
            adj = torch.cat((adj, torch.zeros(seq_len, word_pad, dtype=torch.int)), dim=1)


        elif word_pad < 0:
            batch_x['input_ids'][i] = batch_word[i][:max_word_nodes] 
            batch_x['position_ids'][i] = batch_x['position_ids'][i][:max_word_nodes]
            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i][:max_word_nodes] 
            batch_x['masked_lm_labels'][i] = batch_x['masked_lm_labels'][i][:max_word_nodes] 
            batch_y['masked_lm_labels'][i] = batch_y['masked_lm_labels'][i][:max_word_nodes]   
            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
            adj = adj[:max_word_nodes, :]

            assert adj.shape[0] == seq_len
            adj = adj[:, :max_word_nodes]

        batch_x['attention_mask'][i] = adj
        
    
    for k, v in batch_x.items():     
        if k == 'attention_mask':
            
            batch_x[k] = torch.stack(v, dim=0)#.to(device)
        else:
            batch_x[k] = torch.tensor(v)#.to(device)
    for k, v in batch_y.items():
        batch_y[k] = torch.tensor(v)#.to(device)
    
    # input_ids = data['input_ids'].to(device)
    # attention_mask = data['attention_mask'].to(device)
    # token_type_ids = data['token_type_ids'].to(device) 
    return (batch_text_x, batch_text_y, batch_x, batch_y)