from torch.utils.data import Dataset
import json
import torch
import numpy as np


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class ReasoningChainDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 max_ans_len,
                 nentity,
                 negative_num=5,
                 train=False,
                 max_node_num=7,
                 prepare_candidate=False
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len
        self.negative_num = negative_num
        self.train = train
        self.nentity = nentity
        self.max_node_num = max_node_num
        print(f"Loading data from {data_path}")

        with open(data_path, "r", encoding='utf-8') as r_f:
            self.data = json.load(r_f)['data']
        # for k in range(len(self.data)):
        #     self.data[k]['ans'] = list(set(self.data[k]['ans']) - set(self.data[k]['hard_ans']))

        self.type_id = {
            '1-chain': 0,
            '2-chain': 1,
            '3-chain': 2,
            '2-inter': 3,
            '3-inter': 4,
            'chain-inter': 5,
            'inter-chain': 6,
            '2-union': 7,
            'union-chain': 8
        }

        self.add_special_tokens_dict = {'anchor': self.tokenizer.convert_tokens_to_ids("[anchor]"),
                                        'rela': self.tokenizer.convert_tokens_to_ids("[rela]"),
                                        'target': self.tokenizer.convert_tokens_to_ids("[target]")
                                        }
        self.add_special_tokens_idx = {'anchor': 0, 'rela': 1, 'target': 2
                                       }

        if not train and prepare_candidate:
            entity_texts_file = "/".join(data_path.split("/")[:-1]) + "/entity_text.json"
            print(f"Loading entity texts from {entity_texts_file}")
            with open(entity_texts_file, "r", encoding='utf-8') as e_r_f:
                self.entities = json.load(e_r_f)

            self.features = {}
            candidate_input_ids_list = []
            candidate_atten_masks_list = []
            candidate_add_special_tokens_ids_list = []
            candidate_is_add_special_tokens_list = []
            for i in range(len(self.entities)):
                # candidate_inputs = self.tokenizer("[target] [CLS] {} [SEP]".format(self.entities[str(i)]),
                #                                   max_length=self.max_ans_len,
                #                                   truncation='longest_first',
                #                                   return_tensors="pt")
                candidate_inputs = self.tokenizer("[target] {}".format(self.entities[str(i)]),
                                                  max_length=self.max_ans_len,
                                                  truncation='longest_first',
                                                  return_tensors="pt", add_special_tokens=False)
                candidate_input_ids_list.append(candidate_inputs["input_ids"])
                candidate_atten_masks_list.append(candidate_inputs["attention_mask"])

                candidate_add_special_tokens_ids, \
                candidate_is_add_special_tokens = self.get_add_special_tokens(candidate_inputs['input_ids'][0],
                                                                              query_type='entity')

                candidate_add_special_tokens_ids_list.append(candidate_add_special_tokens_ids)
                candidate_is_add_special_tokens_list.append(candidate_is_add_special_tokens)

            self.features['entity_input_ids'] = collate_tokens([s.view(-1) for s in candidate_input_ids_list],
                                                               self.tokenizer.pad_token_id)
            self.features['entity_masks'] = collate_tokens([s.view(-1) for s in candidate_atten_masks_list],
                                                           self.tokenizer.pad_token_id)
            self.features['entity_add_special_tokens_ids'] = collate_tokens(
                [s for s in candidate_add_special_tokens_ids_list], len(self.add_special_tokens_dict))
            self.features['entity_is_add_special_tokens'] = collate_tokens(
                [s for s in candidate_is_add_special_tokens_list], self.tokenizer.pad_token_id)

    def get_add_special_tokens(self, input_ids, query_type):
        # is_add_special_tokens = torch.zeros_like(input_ids)
        add_special_tokens_ids = torch.ones_like(input_ids) * len(self.add_special_tokens_dict)
        if query_type != 'entity':
            anchor_positions = torch.where(input_ids == self.add_special_tokens_dict['anchor'])[0]
            relation_positions = torch.where(input_ids == self.add_special_tokens_dict['rela'])[0]
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]

            add_special_tokens_ids[anchor_positions] = self.add_special_tokens_idx['anchor']
            add_special_tokens_ids[relation_positions] = self.add_special_tokens_idx['rela']
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']

        else:
            assert query_type == 'entity'
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']
        is_add_special_tokens = torch.where(add_special_tokens_ids != len(self.add_special_tokens_dict), 1, 0)
        return add_special_tokens_ids, is_add_special_tokens

    def __getitem__(self, index):
        sample = self.data[index]
        query_type = sample["type"]
        assert query_type in ["1-chain", "2-chain", "3-chain"]

        # flatten_query_idx = [sample['query'][0]] + list(np.array(sample['query'][1]) + self.nentity)
        adj = torch.zeros((self.max_node_num, self.max_node_num))

        if query_type == "1-chain":
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 3)
            query = ["[anchor] {}".format(sample["query_text"][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][0].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 3)
            mask_col_position = [2]
            graph_row = [0, 0, 1, 1, 1, 2, 2]
            graph_col = [0, 1, 0, 1, 2, 1, 2]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2]
            # graph_col = [0, 1, 0, 1, 2, 0, 1, 2]
            adj[graph_row, graph_col] = 1
        elif query_type == "2-chain":
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 4)
            query = ["[anchor] {}".format(sample["query_text"][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][1].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 4)
            mask_col_position = [3]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
            graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
            # graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 0, 1, 2, 3]
            adj[graph_row, graph_col] = 1
        elif query_type == "3-chain":
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][2].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 5)
            query = ["[anchor] {}".format(sample["query_text"][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][1].strip()),
                     "[rela] {}".format(sample["query_text"][1][2].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 5)
            mask_col_position = [4]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
            graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4]
            # graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 1, 2, 3, 4]
            adj[graph_row, graph_col] = 1
        else:
            raise ValueError("Error query type!!!")

        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

        input_ids_list = []
        attention_mask_list = []
        query_add_special_tokens_ids_list = []
        query_is_add_special_tokens_list = []
        for i in range(len(query_inputs["input_ids"])):
            input_ids = query_inputs["input_ids"][i]  # .numpy().tolist()
            # edit_entity_positions, edit_relation_positions = self.get_query_flags(sample, query_type, query, input_ids)
            query_add_special_tokens_ids, \
            query_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type)
            input_ids_list.append(torch.LongTensor(input_ids))
            attention_mask_list.append(torch.LongTensor(query_inputs["attention_mask"][i]))
            query_add_special_tokens_ids_list.append(query_add_special_tokens_ids)
            query_is_add_special_tokens_list.append(query_is_add_special_tokens)

        query_inputs['input_ids'] = input_ids_list
        query_inputs['attention_mask'] = attention_mask_list
        query_inputs['add_special_tokens_ids'] = query_add_special_tokens_ids_list
        query_inputs['is_add_special_tokens'] = query_is_add_special_tokens_list

        type_index = self.type_id[query_type]
        return_dict = {
            "query_inputs": query_inputs,
            "adj": adj,
            "mask_col_position": mask_col_position,
            'type': torch.LongTensor([type_index]),
            "index": torch.LongTensor([index])
        }

        return return_dict

    def __len__(self):
        return len(self.data)


class ReasoningInterDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 max_ans_len,
                 nentity,
                 negative_num=5,
                 train=False,
                 max_node_num=7
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len
        self.negative_num = negative_num
        self.train = train
        self.nentity = nentity
        self.max_node_num = max_node_num
        print(f"Loading data from {data_path}")

        with open(data_path, "r", encoding='utf-8') as r_f:
            self.data = json.load(r_f)['data']
        # for k in range(len(self.data)):
        #     self.data[k]['ans'] = list(set(self.data[k]['ans']) - set(self.data[k]['hard_ans']))

        self.type_id = {
            '1-chain': 0,
            '2-chain': 1,
            '3-chain': 2,
            '2-inter': 3,
            '3-inter': 4,
            'chain-inter': 5,
            'inter-chain': 6,
            '2-union': 7,
            'union-chain': 8
        }

        self.add_special_tokens_dict = {'anchor': self.tokenizer.convert_tokens_to_ids("[anchor]"),
                                        'rela': self.tokenizer.convert_tokens_to_ids("[rela]"),
                                        'target': self.tokenizer.convert_tokens_to_ids("[target]")
                                        }
        self.add_special_tokens_idx = {'anchor': 0, 'rela': 1, 'target': 2
                                       }

    def get_2u_index(self, query, sample):
        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

        input_ids_list = []
        attention_mask_list = []
        query_add_special_tokens_ids_list = []
        query_is_add_special_tokens_list = []
        for i in range(len(query_inputs["input_ids"])):
            input_ids = query_inputs["input_ids"][i]  # .numpy().tolist()
            query_add_special_tokens_ids, \
            query_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type='1-chain')
            input_ids_list.append(torch.LongTensor(input_ids))
            attention_mask_list.append(torch.LongTensor(query_inputs["attention_mask"][i]))
            query_add_special_tokens_ids_list.append(query_add_special_tokens_ids)
            query_is_add_special_tokens_list.append(query_is_add_special_tokens)

        query_inputs['input_ids'] = input_ids_list
        query_inputs['attention_mask'] = attention_mask_list
        query_inputs['add_special_tokens_ids'] = query_add_special_tokens_ids_list
        query_inputs['is_add_special_tokens'] = query_is_add_special_tokens_list

        return query_inputs

    def get_add_special_tokens(self, input_ids, query_type):
        # is_add_special_tokens = torch.zeros_like(input_ids)
        add_special_tokens_ids = torch.ones_like(input_ids) * len(self.add_special_tokens_dict)
        if query_type != 'entity':
            anchor_positions = torch.where(input_ids == self.add_special_tokens_dict['anchor'])[0]
            relation_positions = torch.where(input_ids == self.add_special_tokens_dict['rela'])[0]
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]

            add_special_tokens_ids[anchor_positions] = self.add_special_tokens_idx['anchor']
            add_special_tokens_ids[relation_positions] = self.add_special_tokens_idx['rela']
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']

        else:
            assert query_type == 'entity'
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']
        is_add_special_tokens = torch.where(add_special_tokens_ids != len(self.add_special_tokens_dict), 1, 0)
        return add_special_tokens_ids, is_add_special_tokens

    def __getitem__(self, index):
        sample = self.data[index]
        query_type = sample["type"]

        assert query_type in ['2-inter', '3-inter', 'chain-inter', '2-union']

        adj = torch.zeros((self.max_node_num, self.max_node_num))

        if query_type in ["2-inter", "3-inter", "chain-inter"]:
            if query_type == '2-inter':
                # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
                #          "[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
                #          "[target]"] + ["[PAD]"] * (self.max_node_num - 5)
                query = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                         "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                         "[anchor] {}".format(sample["query_text"][1][0].strip()),
                         "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                         "[target]"] + ["[PAD]"] * (self.max_node_num - 5)
                # ent_real_in_query_flag = [0, 1, 2, 3]
                #
                # query_seq_position_ids = [0, 1, 2, 3, 4, 1, 2, 3, 4]
                # query_seq_attention_mask = [1] * len(query_seq_position_ids)
                mask_col_position = [4]
                graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
                graph_col = [0, 1, 0, 1, 4, 2, 3, 2, 3, 4, 1, 3, 4]
                # graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4]
                # graph_col = [0, 1, 0, 1, 4, 2, 3, 2, 3, 4, 0, 1, 2, 3, 4]
                adj[graph_row, graph_col] = 1
            elif query_type == '3-inter':
                # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
                #          "[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
                #          "[anchor] [CLS] {} [SEP]".format(sample["query_text"][2][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][2][1][0].strip()),
                #          "[target]"] + ["[PAD]"] * (self.max_node_num - 7)
                query = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                         "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                         "[anchor] {}".format(sample["query_text"][1][0].strip()),
                         "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                         "[anchor] {}".format(sample["query_text"][2][0].strip()),
                         "[rela] {}".format(sample["query_text"][2][1][0].strip()),
                         "[target]"] + ["[PAD]"] * (self.max_node_num - 7)
                # ent_real_in_query_flag = [0, 1, 2, 3, 4, 5]
                #
                # query_seq_position_ids = [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
                # query_seq_attention_mask = [1] * len(query_seq_position_ids)
                mask_col_position = [6]
                graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6]
                graph_col = [0, 1, 0, 1, 6, 2, 3, 2, 3, 6, 4, 5, 4, 5, 6, 1, 3, 5, 6]
                # graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6]
                # graph_col = [0, 1, 0, 1, 6, 2, 3, 2, 3, 6, 4, 5, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
                adj[graph_row, graph_col] = 1
            elif query_type == 'chain-inter':
                # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][1].strip()),
                #          "[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
                #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
                #          "[target]"] + ["[PAD]"] * (self.max_node_num - 6)
                query = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                         "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                         "[rela] {}".format(sample["query_text"][0][1][1].strip()),
                         "[anchor] {}".format(sample["query_text"][1][0].strip()),
                         "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                         "[target]"] + ["[PAD]"] * (self.max_node_num - 6)
                mask_col_position = [5]
                graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
                graph_col = [0, 1, 0, 1, 2, 1, 2, 5, 3, 4, 3, 4, 5, 2, 4, 5]
                # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5]
                # graph_col = [0, 1, 0, 1, 2, 1, 2, 5, 3, 4, 3, 4, 5, 0, 1, 2, 3, 4, 5]
                adj[graph_row, graph_col] = 1
            else:
                raise ValueError("Error query type!!!")

            query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

            input_ids_list = []
            attention_mask_list = []
            query_add_special_tokens_ids_list = []
            query_is_add_special_tokens_list = []
            for i in range(len(query_inputs["input_ids"])):
                input_ids = query_inputs["input_ids"][i]  # .numpy().tolist()
                # edit_entity_positions, edit_relation_positions = self.get_query_flags(sample, query_type, query, input_ids)
                query_add_special_tokens_ids, \
                query_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type)
                input_ids_list.append(torch.LongTensor(input_ids))
                attention_mask_list.append(torch.LongTensor(query_inputs["attention_mask"][i]))
                query_add_special_tokens_ids_list.append(query_add_special_tokens_ids)
                query_is_add_special_tokens_list.append(query_is_add_special_tokens)

            query_inputs['input_ids'] = input_ids_list
            query_inputs['attention_mask'] = attention_mask_list
            query_inputs['add_special_tokens_ids'] = query_add_special_tokens_ids_list
            query_inputs['is_add_special_tokens'] = query_is_add_special_tokens_list

            type_index = self.type_id[query_type]
            return_dict = {
                "query_inputs": query_inputs,
                "adj": adj,
                "mask_col_position": mask_col_position,
                'type': torch.LongTensor([type_index]),
                "index": torch.LongTensor([index])
            }

        else:
            assert query_type == '2-union'

            # query1 = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
            #           "[target]"] + ["[PAD]"] * (self.max_node_num - 3)
            #
            # query2 = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
            #           "[target]"] + ["[PAD]"] * (self.max_node_num - 3)

            query1 = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                      "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                      "[target]"] + ["[PAD]"] * (self.max_node_num - 3)

            query2 = ["[anchor] {}".format(sample["query_text"][1][0].strip()),
                      "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                      "[target]"] + ["[PAD]"] * (self.max_node_num - 3)

            mask_col_position = [2]
            graph_row = [0, 0, 1, 1, 1, 2, 2]
            graph_col = [0, 1, 0, 1, 2, 1, 2]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2]
            # graph_col = [0, 1, 0, 1, 2, 0, 1, 2]
            adj[graph_row, graph_col] = 1


            query_inputs1 = self.get_2u_index(query1, sample)
            query_inputs2 = self.get_2u_index(query2, sample)

            type_index = self.type_id[query_type]
            return_dict = {
                "query_inputs1": query_inputs1,
                "adj1": adj,
                "mask_col_position1": mask_col_position,

                "query_inputs2": query_inputs2,
                "adj2": adj,
                "mask_col_position2": mask_col_position,

                'type': torch.LongTensor([type_index]),
                "index": torch.LongTensor([index])
            }

        return return_dict

    def __len__(self):
        return len(self.data)


class ReasoningInterChainDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 max_ans_len,
                 nentity,
                 negative_num=5,
                 max_node_num=7,
                 train=False,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len
        self.negative_num = negative_num
        self.train = train
        self.nentity = nentity
        self.max_node_num = max_node_num
        print(f"Loading data from {data_path}")

        with open(data_path, "r", encoding='utf-8') as r_f:
            self.data = json.load(r_f)['data']
        # for k in range(len(self.data)):
        #     self.data[k]['ans'] = list(set(self.data[k]['ans']) - set(self.data[k]['hard_ans']))

        self.type_id = {
            '1-chain': 0,
            '2-chain': 1,
            '3-chain': 2,
            '2-inter': 3,
            '3-inter': 4,
            'chain-inter': 5,
            'inter-chain': 6,
            '2-union': 7,
            'union-chain': 8
        }

        self.add_special_tokens_dict = {'anchor': self.tokenizer.convert_tokens_to_ids("[anchor]"),
                                        'rela': self.tokenizer.convert_tokens_to_ids("[rela]"),
                                        'target': self.tokenizer.convert_tokens_to_ids("[target]")
                                        }
        self.add_special_tokens_idx = {'anchor': 0, 'rela': 1, 'target': 2
                                       }

    def get_add_special_tokens(self, input_ids, query_type):
        # is_add_special_tokens = torch.zeros_like(input_ids)
        add_special_tokens_ids = torch.ones_like(input_ids) * len(self.add_special_tokens_dict)
        if query_type != 'entity':
            anchor_positions = torch.where(input_ids == self.add_special_tokens_dict['anchor'])[0]
            relation_positions = torch.where(input_ids == self.add_special_tokens_dict['rela'])[0]
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]

            add_special_tokens_ids[anchor_positions] = self.add_special_tokens_idx['anchor']
            add_special_tokens_ids[relation_positions] = self.add_special_tokens_idx['rela']
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']

        else:
            assert query_type == 'entity'
            target_positions = torch.where(input_ids == self.add_special_tokens_dict['target'])[0]
            add_special_tokens_ids[target_positions] = self.add_special_tokens_idx['target']
        is_add_special_tokens = torch.where(add_special_tokens_ids != len(self.add_special_tokens_dict), 1, 0)
        return add_special_tokens_ids, is_add_special_tokens

    def get_up_index(self, query, sample):
        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

        input_ids_list = []
        attention_mask_list = []
        query_add_special_tokens_ids_list = []
        query_is_add_special_tokens_list = []
        for i in range(len(query_inputs["input_ids"])):
            input_ids = query_inputs["input_ids"][i]  # .numpy().tolist()
            query_add_special_tokens_ids, \
            query_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type='2-chain')
            input_ids_list.append(torch.LongTensor(input_ids))
            attention_mask_list.append(torch.LongTensor(query_inputs["attention_mask"][i]))
            query_add_special_tokens_ids_list.append(query_add_special_tokens_ids)
            query_is_add_special_tokens_list.append(query_is_add_special_tokens)

        query_inputs['input_ids'] = input_ids_list
        query_inputs['attention_mask'] = attention_mask_list
        query_inputs['add_special_tokens_ids'] = query_add_special_tokens_ids_list
        query_inputs['is_add_special_tokens'] = query_is_add_special_tokens_list
        return query_inputs

    def __getitem__(self, index):
        sample = self.data[index]
        query_type = sample["type"]
        assert query_type in ["inter-chain", "union-chain"]

        adj = torch.zeros((self.max_node_num, self.max_node_num))

        if query_type == "inter-chain":
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
            #          "[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][2].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 6)
            query = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                     "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                     "[anchor] {}".format(sample["query_text"][1][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                     "[rela] {}".format(sample["query_text"][2].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 6)
            mask_col_position = [5]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
            graph_col = [0, 1, 0, 1, 4, 2, 3, 2, 3, 4, 1, 3, 4, 5, 4, 5]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
            # graph_col = [0, 1, 0, 1, 4, 2, 3, 2, 3, 4, 1, 3, 4, 5, 0, 1, 2, 3, 4, 5]
            adj[graph_row, graph_col] = 1

            query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

            input_ids_list = []
            attention_mask_list = []
            query_add_special_tokens_ids_list = []
            query_is_add_special_tokens_list = []
            for i in range(len(query_inputs["input_ids"])):
                input_ids = query_inputs["input_ids"][i]  # .numpy().tolist()
                # edit_entity_positions, edit_relation_positions = self.get_query_flags(sample, query_type, query, input_ids)
                query_add_special_tokens_ids, \
                query_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type)
                input_ids_list.append(torch.LongTensor(input_ids))
                attention_mask_list.append(torch.LongTensor(query_inputs["attention_mask"][i]))
                query_add_special_tokens_ids_list.append(query_add_special_tokens_ids)
                query_is_add_special_tokens_list.append(query_is_add_special_tokens)

            query_inputs['input_ids'] = input_ids_list
            query_inputs['attention_mask'] = attention_mask_list
            query_inputs['add_special_tokens_ids'] = query_add_special_tokens_ids_list
            query_inputs['is_add_special_tokens'] = query_is_add_special_tokens_list

            type_index = self.type_id[query_type]
            return_dict = {
                "query_inputs": query_inputs,
                "adj": adj,
                "mask_col_position": mask_col_position,
                'type': torch.LongTensor([type_index]),
                "index": torch.LongTensor([index])
            }
        else:
            # query1 = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][0][1][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][2].strip()),
            #           "[target]"] + ["[PAD]"] * (self.max_node_num - 4)
            #
            # query2 = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1][0].strip()),
            #           "[rela] [CLS] {} [SEP]".format(sample["query_text"][2].strip()),
            #           "[target]"] + ["[PAD]"] * (self.max_node_num - 4)

            query1 = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                      "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                      "[rela] {}".format(sample["query_text"][2].strip()),
                      "[target]"] + ["[PAD]"] * (self.max_node_num - 4)

            query2 = ["[anchor] {}".format(sample["query_text"][1][0].strip()),
                      "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                      "[rela] {}".format(sample["query_text"][2].strip()),
                      "[target]"] + ["[PAD]"] * (self.max_node_num - 4)

            mask_col_position = [3]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
            graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
            # graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 0, 1, 2, 3]
            adj[graph_row, graph_col] = 1

            query_inputs1 = self.get_up_index(query1, sample)
            query_inputs2 = self.get_up_index(query2, sample)

            type_index = self.type_id[query_type]
            return_dict = {
                "query_inputs1": query_inputs1,
                "adj1": adj,
                "mask_col_position1": mask_col_position,

                "query_inputs2": query_inputs2,
                "adj2": adj,
                "mask_col_position2": mask_col_position,

                'type': torch.LongTensor([type_index]),
                "index": torch.LongTensor([index])
            }

        return return_dict

    def __len__(self):
        return len(self.data)
