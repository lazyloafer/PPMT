from torch.utils.data import Dataset
import json
import torch
import numpy as np
import random


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    # print(len(values), size)
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


class ReasoningDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 args,
                 train=True,
                 max_node_num=7
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.max_ans_len = args.max_ans_len
        self.negative_num = args.negative_num
        self.train = train
        self.args = args
        self.max_node_num = max_node_num

        self.add_special_tokens_dict = {'anchor': self.tokenizer.convert_tokens_to_ids("[anchor]"),
                                        'rela': self.tokenizer.convert_tokens_to_ids("[rela]"),
                                        'target': self.tokenizer.convert_tokens_to_ids("[target]"),
                                        'query': self.tokenizer.convert_tokens_to_ids("[query]")
                                        }
        self.add_special_tokens_idx = {'anchor': 0, 'rela': 1, 'target': 2, 'query': 3}
        print(f"Loading data from {args.train_file}")

        with open(args.train_file, "r", encoding='utf-8') as r_f:
            self.data = json.load(r_f)['data']

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

        entity_texts_file = "/".join(args.train_file.split("/")[:-1]) + "/entity_text.json"
        print(f"Loading entity texts from {entity_texts_file}")
        with open(entity_texts_file, "r", encoding='utf-8') as e_r_f:
            self.entities = json.load(e_r_f)

        assert len(self.entities) == self.args.nentity

        if not train:
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

        target_index = np.random.choice(range(len(sample["ans_text"])))
        # ans_inputs = self.tokenizer("[target] [CLS] {} [SEP]".format(sample["ans_text"][target_index]),
        #                             max_length=self.max_ans_len,
        #                             truncation='longest_first', return_tensors="pt", add_special_tokens=False)
        ans_inputs = self.tokenizer("[target] {}".format(sample["ans_text"][target_index]),
                                    max_length=self.max_ans_len,
                                    truncation='longest_first', return_tensors="pt", add_special_tokens=False)
        ans_add_special_tokens_ids, \
        ans_is_add_special_tokens = self.get_add_special_tokens(ans_inputs['input_ids'][0], query_type='entity')
        ans_inputs['add_special_tokens_ids'] = ans_add_special_tokens_ids.unsqueeze(0)
        ans_inputs['is_add_special_tokens'] = ans_is_add_special_tokens.unsqueeze(0)
        selected_ans = sample["ans"][target_index]

        adj = torch.zeros((self.max_node_num, self.max_node_num))

        if query_type == '1-chain':
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 3)
            query = ["[anchor] {}".format(sample["query_text"][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][0].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 3)
            # ent_real_in_query_flag = [0, 1]
            #
            # query_seq_position_ids = [0, 1, 2, 3, 4]
            # query_seq_attention_mask = [1] * len(query_seq_position_ids)
            mask_col_position = [2]
            graph_row = [0, 0, 1, 1, 1, 2, 2]
            graph_col = [0, 1, 0, 1, 2, 1, 2]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2]
            # graph_col = [0, 1, 0, 1, 2, 0, 1, 2]
            adj[graph_row, graph_col] = 1

            if self.args.add_path:
                path = ["[anchor] {}".format(sample["query_text"][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip())]
                ent_real_in_path_flag = [0, 1, 2]

        elif query_type == '2-chain':
            # query = ["[anchor] [CLS] {} [SEP]".format(sample["query_text"][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][0].strip()),
            #          "[rela] [CLS] {} [SEP]".format(sample["query_text"][1][1].strip()),
            #          "[target]"] + ["[PAD]"] * (self.max_node_num - 4)
            query = ["[anchor] {}".format(sample["query_text"][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][0].strip()),
                     "[rela] {}".format(sample["query_text"][1][1].strip()),
                     "[target]"] + ["[PAD]"] * (self.max_node_num - 4)
            # ent_real_in_query_flag = [0, 1, 2]
            #
            # query_seq_position_ids = [0, 1, 2, 3, 4, 5]
            # query_seq_attention_mask = [1] * len(query_seq_position_ids)
            mask_col_position = [3]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
            graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
            # graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 0, 1, 2, 3]
            adj[graph_row, graph_col] = 1

            if self.args.add_path:
                T1_idx = 0
                T1_text = ''
                T2_text = ''
                while T2_text != sample["ans_text"][target_index]:
                    T1_idx = np.random.choice(range(len(sample["T1"])))
                    T1_text = sample['T1_text'][T1_idx]

                    if len(sample["T2"][T1_idx]) == 0:
                        continue
                    T2_idx = np.random.choice(range(len(sample["T2"][T1_idx])))
                    T2_text = sample['T2_text'][T1_idx][T2_idx]
                path = ["[anchor] {}".format(sample["query_text"][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][1].strip()),
                        "[target] {}".format(T2_text.strip())]
                ent_real_in_path_flag = [0, 1, 2, 3]

        elif query_type == '3-chain':
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
            # ent_real_in_query_flag = [0, 1, 2, 3]
            #
            # query_seq_position_ids = [0, 1, 2, 3, 4, 5, 6]
            # query_seq_attention_mask = [1] * len(query_seq_position_ids)
            mask_col_position = [4]
            graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
            graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4]
            # graph_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4]
            # graph_col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 1, 2, 3, 4]
            adj[graph_row, graph_col] = 1

            if self.args.add_path:
                T1_idx = 0
                T1_text = ''
                T2_idx = 0
                T2_text = ''
                T3_text = ''
                while T3_text != sample["ans_text"][target_index]:
                    T1_idx = np.random.choice(range(len(sample["T1"])))
                    T1_text = sample['T1_text'][T1_idx]

                    if len(sample["T2"][T1_idx]) == 0:
                        continue
                    T2_idx = np.random.choice(range(len(sample["T2"][T1_idx])))
                    T2_text = sample['T2_text'][T1_idx][T2_idx]

                    if len(sample["T3"][T1_idx][T2_idx]) == 0:
                        continue
                    T3_idx = np.random.choice(range(len(sample["T3"][T1_idx][T2_idx])))
                    T3_text = sample['T3_text'][T1_idx][T2_idx][T3_idx]
                path = ["[anchor] {}".format(sample["query_text"][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][1].strip()),
                        "[rela] {}".format(sample["query_text"][1][2].strip()),
                        "[target] {}".format(T3_text.strip())]
                ent_real_in_path_flag = [0, 1, 2, 3, 4]

        elif query_type == '2-inter':
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

            if self.args.add_path:
                path = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                        "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip()),
                        "[anchor] {}".format(sample["query_text"][1][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip())]
                ent_real_in_path_flag = [0, 1, 2, 3, 4, 5]

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

            if self.args.add_path:
                path = ["[anchor] {}".format(sample["query_text"][0][0].strip()),
                        "[rela] {}".format(sample["query_text"][0][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip()),
                        "[anchor] {}".format(sample["query_text"][1][0].strip()),
                        "[rela] {}".format(sample["query_text"][1][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip()),
                        "[anchor] {}".format(sample["query_text"][2][0].strip()),
                        "[rela] {}".format(sample["query_text"][2][1][0].strip()),
                        "[target] {}".format(sample["ans_text"][target_index].strip())]
                ent_real_in_path_flag = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        else:
            raise ValueError("Illegal query type!!!!")

        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, add_special_tokens=False)

        input_ids_list = []
        attention_mask_list = []
        query_add_special_tokens_ids_list = []
        query_is_add_special_tokens_list = []
        for i in range(len(query_inputs["input_ids"])):
            input_ids = query_inputs["input_ids"][i]#.numpy().tolist()
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

        if self.args.add_path:
            path_inputs = self.tokenizer(path, max_length=self.max_seq_len)

            path_add_special_tokens_ids_list = []
            path_is_add_special_tokens_list = []
            for i in range(len(path_inputs["input_ids"])):
                input_ids = path_inputs["input_ids"][i]
                path_add_special_tokens_ids, \
                path_is_add_special_tokens = self.get_add_special_tokens(torch.tensor(input_ids), query_type='path')
                path_add_special_tokens_ids_list.append(path_add_special_tokens_ids)
                path_is_add_special_tokens_list.append(path_is_add_special_tokens)

            path_inputs['add_special_tokens_ids'] = path_add_special_tokens_ids_list
            path_inputs['is_add_special_tokens'] = path_is_add_special_tokens_list

        type_index = self.type_id[query_type]
        return_dict = {
            "query_inputs": query_inputs,
            "adj": adj,
            # "ent_real_in_query_flag": torch.LongTensor(ent_real_in_query_flag),
            # "query_seq_attention_mask": torch.LongTensor(query_seq_attention_mask),
            # "query_seq_position_ids": torch.LongTensor(query_seq_position_ids),
            "mask_col_position": mask_col_position,
            'type': torch.LongTensor([type_index]),
            "ans_inputs": ans_inputs,
            "ans": sample["ans"],
            "selected_ans": selected_ans,
            "ans_subsample_weight": torch.sqrt(1 / torch.Tensor([4 + len(sample["ans"])]))
        }

        if self.args.add_path:
            return_dict["path_inputs"] = path_inputs
            return_dict["ent_real_in_path_flag"] = torch.LongTensor(ent_real_in_path_flag)

        if not self.train:
            return_dict["index"] = torch.LongTensor([index])

        return return_dict

    def __len__(self):
        return len(self.data)


def qa_collate(samples, tokenizer, answer_features, args, pad_id=0):
    if len(samples) == 0:
        return {}
    negative_num = args.negative_num
    global_negative_num = args.global_negative_num

    if 'query_inputs' in samples[0]:
        query_input_ids = []
        query_mask = []
        query_add_special_tokens_ids = []
        query_is_add_special_tokens = []
        mask_col_position = []
        adj = []

        # query_len_cache = 0
        for i in range(len(samples)):
            query_input_ids += samples[i]["query_inputs"]["input_ids"]
            query_mask += samples[i]["query_inputs"]["attention_mask"]
            query_add_special_tokens_ids += samples[i]["query_inputs"]["add_special_tokens_ids"]
            query_is_add_special_tokens += samples[i]["query_inputs"]["is_add_special_tokens"]
            mask_col_position += samples[i]["mask_col_position"]
            adj.append(samples[i]["adj"])
        adj = torch.stack(adj, dim=0)

        batch = {
            'query_input_ids': collate_tokens(query_input_ids, pad_id),
            'query_mask': collate_tokens(query_mask, pad_id),
            'query_add_special_tokens_ids': collate_tokens(
                query_add_special_tokens_ids, len(tokenizer.additional_special_tokens)),
            'query_is_add_special_tokens': collate_tokens(query_is_add_special_tokens, pad_id),
            'mask_col_position': mask_col_position,
            'adj': adj
        }
    elif 'query_inputs1' in samples[0]:  # specially for 2u and up in test set
        assert 'query_inputs2' in samples[0]

        query_input_ids1 = []
        query_mask1 = []
        query_add_special_tokens_ids1 = []
        query_is_add_special_tokens1 = []
        mask_col_position1 = []
        adj1 = []

        query_input_ids2 = []
        query_mask2 = []
        query_add_special_tokens_ids2 = []
        query_is_add_special_tokens2 = []
        mask_col_position2 = []
        adj2 = []

        for i in range(len(samples)):
            query_input_ids1 += samples[i]["query_inputs1"]["input_ids"]
            query_mask1 += samples[i]["query_inputs1"]["attention_mask"]
            query_add_special_tokens_ids1 += samples[i]["query_inputs1"]["add_special_tokens_ids"]
            query_is_add_special_tokens1 += samples[i]["query_inputs1"]["is_add_special_tokens"]
            mask_col_position1 += samples[i]["mask_col_position1"]
            adj1.append(samples[i]["adj1"])

            query_input_ids2 += samples[i]["query_inputs2"]["input_ids"]
            query_mask2 += samples[i]["query_inputs2"]["attention_mask"]
            query_add_special_tokens_ids2 += samples[i]["query_inputs2"]["add_special_tokens_ids"]
            query_is_add_special_tokens2 += samples[i]["query_inputs2"]["is_add_special_tokens"]
            mask_col_position2 += samples[i]["mask_col_position2"]
            adj2.append(samples[i]["adj2"])

        adj1 = torch.stack(adj1, dim=0)
        adj2 = torch.stack(adj2, dim=0)

        batch = {
            'query_input_ids1': collate_tokens(query_input_ids1, pad_id),
            'query_mask1': collate_tokens(query_mask1, pad_id),
            'query_add_special_tokens_ids1': collate_tokens(
                query_add_special_tokens_ids1, len(tokenizer.additional_special_tokens)),
            'query_is_add_special_tokens1': collate_tokens(query_is_add_special_tokens1, pad_id),
            'mask_col_position1': mask_col_position1,
            'adj1': adj1,

            'query_input_ids2': collate_tokens(query_input_ids2, pad_id),
            'query_mask2': collate_tokens(query_mask2, pad_id),
            'query_add_special_tokens_ids2': collate_tokens(
                query_add_special_tokens_ids2, len(tokenizer.additional_special_tokens)),
            'query_is_add_special_tokens2': collate_tokens(query_is_add_special_tokens2, pad_id),
            'mask_col_position2': mask_col_position2,
            'adj2': adj2
        }
    else:
        raise ValueError("Error query type!!!")

    if "path_inputs" in samples[0]:
        path_input_ids = []
        path_mask = []
        path_add_special_tokens_ids = []
        path_is_add_special_tokens = []
        ent_real_in_path_flag = []
        path_len_cache = 0
        for i in range(len(samples)):
            path_input_ids += [torch.LongTensor(s) for s in samples[i]["path_inputs"]["input_ids"]]
            path_mask += [torch.LongTensor(s) for s in samples[i]["path_inputs"]["attention_mask"]]
            path_add_special_tokens_ids += samples[i]["path_inputs"]["add_special_tokens_ids"]
            path_is_add_special_tokens += samples[i]["path_inputs"]["is_add_special_tokens"]
            ent_real_in_path_flag.append(samples[i]["ent_real_in_path_flag"] + path_len_cache)
            path_len_cache += len(samples[i]["path_inputs"]["input_ids"])

        batch['path_inputs_ids'] = collate_tokens(path_input_ids, pad_id)
        batch['path_mask'] = collate_tokens(path_mask, pad_id)
        batch['path_add_special_tokens_ids'] = collate_tokens(
            path_add_special_tokens_ids, len(tokenizer.additional_special_tokens))
        batch['path_is_add_special_tokens'] = collate_tokens(path_is_add_special_tokens, pad_id)
        batch['ent_real_in_path_flag'] = ent_real_in_path_flag
        # batch["path_entity_positions"] = collate_tokens([s["path_entity_positions"] for s in samples], 0)
        # batch["path_relation_positions"] = collate_tokens([s["path_relation_positions"] for s in samples], 0)
        # batch["path_target_positions"] = collate_tokens([s["path_target_positions"] for s in samples], 0)

    if "ans_inputs" in samples[0]:
        batch.update({
            'ans_input_ids': collate_tokens([s["ans_inputs"]["input_ids"].view(-1) for s in samples], pad_id),
            'ans_masks': collate_tokens([s["ans_inputs"]["attention_mask"].view(-1) for s in samples], pad_id),
            'ans_add_special_tokens_ids': collate_tokens(
                [s["ans_inputs"]["add_special_tokens_ids"].view(-1) for s in samples],
                len(tokenizer.additional_special_tokens)),
            'ans_is_add_special_tokens': collate_tokens(
                [s["ans_inputs"]["is_add_special_tokens"].view(-1) for s in samples], pad_id),
            'ans_subsample_weight': torch.concat([s["ans_subsample_weight"].view(-1) for s in samples], dim=0)
        })

        if global_negative_num > 0:
            batch_ans = []
            for i in range(len(samples)):
                batch_ans += samples[i]["ans"]
            global_neg_ans_id = list(
                np.random.choice(list(set(range(args.nentity)).difference(set(batch_ans))),
                                 global_negative_num,
                                 replace=False)
            )
            batch.update({
                'global_neg_ans_id': torch.LongTensor(global_neg_ans_id),
                'global_neg_ans_input_ids': collate_tokens(
                    [answer_features["entity_input_ids"][i][answer_features["entity_masks"][i].bool()] for i in
                     global_neg_ans_id], pad_id),
                'global_neg_ans_atten_masks': collate_tokens(
                    [answer_features["entity_masks"][i][answer_features["entity_masks"][i].bool()] for i in
                     global_neg_ans_id], pad_id),
                'global_neg_ans_add_special_tokens_ids': collate_tokens(
                    [answer_features["entity_add_special_tokens_ids"][i][answer_features["entity_masks"][i].bool()] for
                     i in
                     global_neg_ans_id], pad_id),
                'global_neg_ans_is_add_special_tokens': collate_tokens(
                    [answer_features["entity_is_add_special_tokens"][i][answer_features["entity_masks"][i].bool()] for i
                     in
                     global_neg_ans_id], pad_id),
            })

    if "selected_ans" in samples[0]:
        batch["selected_ans"] = collate_tokens([torch.LongTensor([s["selected_ans"]]) for s in samples], -1)
        batch["ans"] = collate_tokens([torch.LongTensor(s["ans"]) for s in samples], -1)

        ## get in-batch neg ans
        negative_index_list = []
        only_negative_index_list = []
        for i in range(len(samples)):
            # pos random
            if len(samples) > negative_num:
                cur_nega_index = list(
                    np.random.choice(list(set(range(len(samples))) - {i}), negative_num, replace=False))
            else:
                cur_nega_index = list(
                    np.random.choice(list(set(range(len(samples))) - {i}), negative_num, replace=True))
            only_negative_index_list.append(torch.LongTensor(cur_nega_index))

            insert_index = np.random.choice(range(negative_num + 1))
            new_cur_nega_index = cur_nega_index[:insert_index] + [i] + cur_nega_index[insert_index:]
            negative_index_list.append(torch.LongTensor(new_cur_nega_index))

        batch["negative_index"] = collate_tokens(negative_index_list, -1)
        batch["true_negative_index"] = collate_tokens(only_negative_index_list, -1)

        tag_list = []
        for i in range(len(samples)):
            # pos random
            cur_tag = []
            pos_num = 0
            for j in range(negative_num + 1):
                if samples[negative_index_list[i][j]]["selected_ans"] in samples[i]["ans"]:
                    cur_tag.append(1.0)
                    pos_num += 1
                else:
                    cur_tag.append(0)
            tag_list.append(torch.tensor(cur_tag) / pos_num)
        batch["tags"] = collate_tokens(tag_list, -1)

    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)

    if "sep_index" in samples[0]:
        batch["sep_index"] = collate_tokens([s["sep_index"] for s in samples], -1)

    if "subsampling_weight" in samples[0]:
        batch["subsampling_weight"] = collate_tokens([s["subsampling_weight"] for s in samples], -1)

    if "entity_positions" in samples[0]:
        batch["entity_positions"] = collate_tokens([s["entity_positions"] for s in samples], 0)
        batch["relation_positions"] = collate_tokens([s["relation_positions"] for s in samples], 0)
    elif "entity_positions1" in samples[0]:
        assert "entity_positions2" in samples[0]
        batch["entity_positions1"] = collate_tokens([s["entity_positions1"] for s in samples], 0)
        batch["relation_positions1"] = collate_tokens([s["relation_positions1"] for s in samples], 0)
        batch["entity_positions2"] = collate_tokens([s["entity_positions2"] for s in samples], 0)
        batch["relation_positions2"] = collate_tokens([s["relation_positions2"] for s in samples], 0)
    # else:
    #     raise ValueError("Error query type!!!")

    if "mask_positions" in samples[0]:
        batch["mask_positions"] = collate_tokens([s["mask_positions"] for s in samples], 0)

    if "type" in samples[0]:
        batch["type"] = collate_tokens([s["type"] for s in samples], -1)

    if "union_label" in samples[0]:
        batch["union_label"] = collate_tokens([s["union_label"] for s in samples], -1)

    return batch