import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import BertPreTrainedModel, BertModel
from transformers.trainer_pt_utils import nested_numpify, nested_concat, distributed_concat
# from path_transformer import PathTransformer
from prefix_bert import MenBert

def _pad_across_processes(tensor, rank, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, rank, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, rank, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = _nested_gather(size, rank).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor

def _nested_gather(tensors, rank):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if rank != -1:
        tensors = distributed_concat(tensors)
    return tensors

def gather_data(feature, rank):
    gather_feature_host = None
    gather_feature = _pad_across_processes(feature, rank)
    gather_feature = _nested_gather(gather_feature, rank)
    gather_feature_host = gather_feature if gather_feature_host is None else nested_concat(gather_feature_host,
                                                                gather_feature,
                                                                padding_index=-100)
    return gather_feature_host

class ReasoningModel(BertPreTrainedModel):
    @staticmethod
    def euclidean_distance(rep1, rep2):
        distance = rep1 - rep2
        distance = torch.norm(distance, p=2, dim=-1)
        return distance

    @staticmethod
    def cross_entropy(p,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        log_likelihood = - torch.log(p[range(m),y]+1e-5)
        loss = torch.sum(log_likelihood) / m
        return loss

    def __init__(self, config, tokenizer, args=None):
        super().__init__(config)
        self.args = args
        self.pool_method = 'mean'  # mean  first
        self.bert = MenBert.from_pretrained(args.model_name)
        # self.path_encoder = PathTransformer(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
        #                                     num_attention_heads=12,
        #                                     attention_probs_dropout_prob=0.1,
        #                                     hidden_dropout_prob=0.1, layer_norm_eps=1e-12, num_hidden_layers=2)
        # self.path_decoder = PathTransformer(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
        #                                     num_attention_heads=12,
        #                                     attention_probs_dropout_prob=0.1,
        #                                     hidden_dropout_prob=0.1, layer_norm_eps=1e-12, num_hidden_layers=2)


        self.gamma_coff = 20
        self.gamma = 20

        self.tokenizer = tokenizer

        self.add_special_tokens_dict = {'anchor': self.tokenizer.convert_tokens_to_ids("[anchor]"),
                                        'rela': self.tokenizer.convert_tokens_to_ids("[rela]"),
                                        'target': self.tokenizer.convert_tokens_to_ids("[target]"),
                                        'query': self.tokenizer.convert_tokens_to_ids("[query]")
                                        }
        self.query_idx = torch.tensor(3, device='cuda')

        self.query_seq_len = {0: 5,
                              1: 6,
                              2: 7,
                              3: 9,
                              4: 13,
                              5: 10,
                              6: 11}

        self.add_special_tokens_embeddings = torch.nn.Embedding(len(self.add_special_tokens_dict) + 1, config.hidden_size)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def encode_ent_rela(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                        add_special_tokens_ids=None, is_add_special_tokens=None):
        if not self.args.fine_tuning:
            pretrain_seq_embedds = self.bert.embeddings.word_embeddings(input_ids) * (1 - is_add_special_tokens).unsqueeze(-1)
            add_special_tokens_embeddings = self.add_special_tokens_embeddings(add_special_tokens_ids) * is_add_special_tokens.unsqueeze(-1)
            inputs_embeds = pretrain_seq_embedds + add_special_tokens_embeddings
            input_ids = None
        else:
            inputs_embeds = None

        if self.args.is_memory:
            memory_attention_mask = torch.concat(
                [
                    torch.ones(size=(attention_mask.size(0), self.args.memory_size), device=attention_mask.device),
                    attention_mask[:, 1:]
                ], dim=-1
            )
        else:
            memory_attention_mask = attention_mask
            # a = attention_mask.detach().cpu().numpy()
        output = self.bert(input_ids=input_ids,
                           inputs_embeds=inputs_embeds,
                           attention_mask=memory_attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           use_memory=self.args.is_memory).last_hidden_state
        # pooled_output = torch.mean(output, dim=1)

        if self.pool_method == 'mean':
            pooled_output = torch.sum(output * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1,
                                                                                                keepdim=True)
        elif self.pool_method == 'first':
            pooled_output = output[:, 0, :]
        else:
            raise ValueError('error pool method!!!!')

        return pooled_output, output

    def encode_geometric_query(self, input_node_embeddings, types):
        query_embeddings = []
        query_index = []
        for i in range(len(input_node_embeddings)):
            query_index.append(i)
            if types[i][0] == 0:
                query_embeddings.append(input_node_embeddings[i][0] + input_node_embeddings[i][1])
            elif types[i][0] == 1:
                query_embeddings.append(
                    input_node_embeddings[i][0] +
                    input_node_embeddings[i][1] +
                    input_node_embeddings[i][2]
                )
            elif types[i][0] == 2:
                query_embeddings.append(
                    input_node_embeddings[i][0] +
                    input_node_embeddings[i][1] +
                    input_node_embeddings[i][2] +
                    input_node_embeddings[i][3]
                )
            elif types[i][0] == 3:
                query_embeddings.append(
                    (input_node_embeddings[i][0] + input_node_embeddings[i][1] +
                     input_node_embeddings[i][2] + input_node_embeddings[i][3]) / 2
                )
                # query_index.append(i)
                # query_embeddings.append(input_node_embeddings[i][2] + input_node_embeddings[i][3])
            elif types[i][0] == 4:
                query_embeddings.append(
                    (input_node_embeddings[i][0] + input_node_embeddings[i][1] +
                     input_node_embeddings[i][2] + input_node_embeddings[i][3] +
                     input_node_embeddings[i][4] + input_node_embeddings[i][5]) / 3
                )
                # query_index.append(i)
                # query_embeddings.append(input_node_embeddings[i][2] + input_node_embeddings[i][3])
                # query_index.append(i)
                # query_embeddings.append(input_node_embeddings[i][4] + input_node_embeddings[i][5])
        return torch.stack(query_embeddings, dim=0), query_index

    def encode_context_query(self, inputs_embeds, attention_mask=None, token_type_ids=None, position_ids=None,
                     add_special_tokens_ids=None, is_add_special_tokens=None):

        batch_size = inputs_embeds.size(0)

        add_special_tokens_embeddings = self.add_special_tokens_embeddings(self.query_idx).view(1, 1, -1).repeat(batch_size, 1, 1)
        inputs_embeds = torch.concat([add_special_tokens_embeddings, inputs_embeds], dim=1)

        a = torch.where(torch.sum(attention_mask, dim=-1, keepdim=True) > 0, 1, 0).squeeze(-1).unsqueeze(1)
        mask_temp = torch.concat([a, attention_mask], dim=1)
        b = torch.where(torch.sum(mask_temp, dim=-1, keepdim=True) > 0, 1, 0)

        if self.args.is_memory:
            memory_mask = b * torch.ones(size=(mask_temp.size(1), self.args.memory_size), device=attention_mask.device).repeat(batch_size, 1, 1)
            memory_attention_mask = torch.concat([memory_mask, mask_temp], dim=-1)
        else:
            memory_attention_mask = mask_temp
            # a = attention_mask.detach().cpu().numpy()
        output = self.bert(input_ids=None,
                           inputs_embeds=inputs_embeds,
                           attention_mask=memory_attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           use_memory=self.args.is_memory).last_hidden_state
        # pooled_output = torch.mean(output, dim=1)

        if self.pool_method == 'mean':
            pooled_output = torch.sum(output * b, dim=1) / torch.sum(b, dim=1)
        elif self.pool_method == 'first':
            pooled_output = output[:, 0, :]
        else:
            raise ValueError('error pool method!!!!')

        return pooled_output, output

    def training_loss(self, batch, lamd, task):

        ent_rel_embeddings = self.encode_ent_rela(input_ids=batch['query_input_ids'],
                                                  attention_mask=batch['query_mask'],
                                                  token_type_ids=batch.get('query_type_ids', None),
                                                  add_special_tokens_ids=batch['query_add_special_tokens_ids'],
                                                  is_add_special_tokens=batch['query_is_add_special_tokens'])[0]

        ans_rep = self.encode_ent_rela(input_ids=batch['ans_input_ids'],
                                       attention_mask=batch['ans_masks'],
                                       token_type_ids=batch.get('a_type_ids', None),
                                       add_special_tokens_ids=batch['ans_add_special_tokens_ids'],
                                       is_add_special_tokens=batch['ans_is_add_special_tokens']
                                       )[0]

        if self.args.global_negative_num > 0:
            global_neg_ans_rep = self.encode_ent_rela(input_ids=batch['global_neg_ans_input_ids'],
                                                      attention_mask=batch['global_neg_ans_atten_masks'],
                                                      add_special_tokens_ids=batch[
                                                          'global_neg_ans_add_special_tokens_ids'],
                                                      is_add_special_tokens=batch[
                                                          'global_neg_ans_is_add_special_tokens'])[0]


        input_node_embeddings = torch.stack(torch.chunk(ent_rel_embeddings, chunks=batch['adj'].size(0), dim=0), dim=0)

        if task == 'geo':
            query_embeddings_geo, query_index_geo = self.encode_geometric_query(input_node_embeddings, batch['type'])
            neg_ans_rep_geo = ans_rep[batch["negative_index"][query_index_geo]]  # [batch['mask_row_position']]
            neg_ans_rep_geo = neg_ans_rep_geo.transpose(-2, -1)
            tags_geo = batch['tags'][[query_index_geo]]
            logits_rk_1_geo = query_embeddings_geo.unsqueeze(1) @ neg_ans_rep_geo
            logits_rk_1_geo = logits_rk_1_geo.squeeze(1)
            if self.args.global_negative_num > 0:
                global_neg_ans_rep = global_neg_ans_rep.transpose(0, 1)
                logits_rk_2_geo = query_embeddings_geo @ global_neg_ans_rep
                logits_rk_1_geo = torch.concat([logits_rk_1_geo, logits_rk_2_geo], dim=-1)
                tags_geo = torch.concat([tags_geo, torch.zeros_like(logits_rk_2_geo)], dim=-1)

            rk_loss_geo = self.cls_loss_fn(logits_rk_1_geo / 1.0, tags_geo)
            return logits_rk_1_geo, rk_loss_geo
        else:
            seq_embeddings = self.encode_context_query(inputs_embeds=input_node_embeddings, attention_mask=batch['adj'])[0]
            neg_ans_rep = ans_rep[batch["negative_index"]]  # [batch['mask_row_position']]
            neg_ans_rep = neg_ans_rep.transpose(-2, -1)
            tags = batch['tags']  # [batch['mask_row_position']]
            logits_rk_1 = seq_embeddings.unsqueeze(1) @ neg_ans_rep
            logits_rk_1 = logits_rk_1.squeeze(1)
            if self.args.global_negative_num > 0:
                global_neg_ans_rep = global_neg_ans_rep.transpose(0, 1)
                logits_rk_2 = seq_embeddings @ global_neg_ans_rep
                logits_rk_1 = torch.concat([logits_rk_1, logits_rk_2], dim=-1)
                tags = torch.concat([tags, torch.zeros_like(logits_rk_2)], dim=-1)

            rk_loss = self.cls_loss_fn(logits_rk_1 / 1.0, tags)
            return logits_rk_1, rk_loss

        # P(q, a)

        # if self.args.global_negative_num > 0:
        #     global_neg_ans_rep = global_neg_ans_rep.transpose(0, 1)
        #     logits_rk_2 = seq_embeddings @ global_neg_ans_rep
        #     logits_rk_1 = torch.concat([logits_rk_1, logits_rk_2], dim=-1)
        #     tags = torch.concat([tags, torch.zeros_like(logits_rk_2)], dim=-1)
        #
        #     logits_rk_2_geo = query_embeddings_geo @ global_neg_ans_rep
        #     logits_rk_1_geo = torch.concat([logits_rk_1_geo, logits_rk_2_geo], dim=-1)
        #     tags_geo = torch.concat([tags_geo, torch.zeros_like(logits_rk_2_geo)], dim=-1)
        #
        # rk_loss = cls_loss_fn_2(logits_rk_1 / 1.0, tags)
        # rk_loss_geo = cls_loss_fn_2(logits_rk_1_geo / 1.0, tags_geo)
        # # loss = rk_loss

        # return rk_loss, rk_loss_geo  # lamd * classification_loss + 1 * loss

    def predict_score(self, batch, candidate_rep, schema):
        # print()
        if batch["type"][0][0] in [0, 1, 2, 3, 4, 5, 6]:
            ent_rel_embeddings = self.encode_ent_rela(input_ids=batch['query_input_ids'],
                                                      attention_mask=batch['query_mask'],
                                                      token_type_ids=batch.get('query_type_ids', None),
                                                      add_special_tokens_ids=batch['query_add_special_tokens_ids'],
                                                      is_add_special_tokens=batch['query_is_add_special_tokens'])[0]

            input_node_embeddings = torch.stack(torch.chunk(ent_rel_embeddings, chunks=batch['adj'].size(0), dim=0),
                                                dim=0)
            seq_embeddings = self.encode_context_query(inputs_embeds=input_node_embeddings, attention_mask=batch['adj'])[0]

        elif batch["type"][0][0] in [7, 8]:
            ent_rel_embeddings1 = self.encode_ent_rela(input_ids=batch['query_input_ids1'],
                                                       attention_mask=batch['query_mask1'],
                                                       token_type_ids=batch.get('query_type_ids1', None),
                                                       add_special_tokens_ids=batch['query_add_special_tokens_ids1'],
                                                       is_add_special_tokens=batch['query_is_add_special_tokens1'])[0]
            input_node_embeddings1 = torch.stack(torch.chunk(ent_rel_embeddings1, chunks=batch['adj1'].size(0), dim=0),
                                                 dim=0)
            seq_embeddings1 = self.encode_context_query(inputs_embeds=input_node_embeddings1, attention_mask=batch['adj1'])[0]

            ent_rel_embeddings2 = self.encode_ent_rela(input_ids=batch['query_input_ids2'],
                                                       attention_mask=batch['query_mask2'],
                                                       token_type_ids=batch.get('query_type_ids2', None),
                                                       add_special_tokens_ids=batch['query_add_special_tokens_ids2'],
                                                       is_add_special_tokens=batch['query_is_add_special_tokens2'])[0]
            input_node_embeddings2 = torch.stack(torch.chunk(ent_rel_embeddings2, chunks=batch['adj2'].size(0), dim=0),
                                                 dim=0)
            seq_embeddings2 = self.encode_context_query(inputs_embeds=input_node_embeddings2, attention_mask=batch['adj2'])[0]

            seq_embeddings = torch.stack([seq_embeddings1, seq_embeddings2], dim=0)

        else:
            raise ValueError("Error query type!!! Query type must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]")

        if batch["type"][0][0] < 7 or batch["type"][0][0] > 8:
            neg_ans_rep = candidate_rep.unsqueeze(0).repeat(seq_embeddings.size(0), 1, 1).transpose(-2, -1)
            seq_embeddings = seq_embeddings.unsqueeze(1)
            logits = (seq_embeddings @ neg_ans_rep).squeeze(1) / 1.0
            logits = logits.softmax(dim=-1)
            return logits
        else:
            neg_ans_rep = candidate_rep.unsqueeze(0).repeat(seq_embeddings.size(1), 1, 1).transpose(-2, -1)
            query_rep_1 = seq_embeddings[0].unsqueeze(1)
            logits_1 = (query_rep_1 @ neg_ans_rep).squeeze(1) / 1.0
            score_1 = logits_1.softmax(dim=-1)

            query_rep_2 = seq_embeddings[1].unsqueeze(1)
            logits_2 = (query_rep_2 @ neg_ans_rep).squeeze(1) / 1.0
            score_2 = logits_2.softmax(dim=-1)

            score = torch.max(torch.stack([score_1, score_2], dim=0), dim=0)[0]
            return score

    def forward(self, batch, is_prepare=False, lamd=0.3, candidate_rep=None, schema='matching', task='query'):
        if is_prepare:
            candidates_rep = self.encode_ent_rela(input_ids=batch['entity_input_ids'],
                                                  attention_mask=batch['entity_masks'],
                                                  token_type_ids=batch.get('entity_type_ids', None),
                                                  add_special_tokens_ids=batch['entity_add_special_tokens_ids'],
                                                  is_add_special_tokens=batch['entity_is_add_special_tokens'])[0]
            return candidates_rep
            
        elif candidate_rep == None:
            return self.training_loss(batch, lamd, task)
        else:
            return self.predict_score(batch, candidate_rep, schema)



