import logging
import os
import random
from datetime import date
from functools import partial
import collections
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import nested_numpify, nested_concat, distributed_concat
import torch.nn.functional as F
from config import train_args
from dataset_shuffle import ReasoningDataset, qa_collate
from dataset import ReasoningChainDataset, ReasoningInterDataset, ReasoningInterChainDataset
from model import ReasoningModel
from utils import AverageMeter, move_to_cuda, move_to_ds_cuda, load_saved
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    args = train_args()
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-bsz{args.train_batch_size}-lr{args.learning_rate}-epoch{args.num_train_epochs}-maxlen{args.max_seq_len}_neg{args.negative_num}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed()
        args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
    args.device = device
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # tokenizer.add_special_tokens({'additional_special_tokens': ["[anchor]", "[rela]", "[target]", "[qtype]",
    #                                                             "[projection]", "[intersection]", "[union]"]})
    tokenizer.add_special_tokens({'additional_special_tokens': ["[anchor]", "[rela]", "[target]", "[query]"]})
    print("*"*100)
    print(tokenizer.additional_special_tokens)

    model = ReasoningModel(bert_config, tokenizer, args)

    model.resize_token_embeddings(len(tokenizer))

    if not args.fine_tuning:
        for name, param in model.bert.named_parameters():
            if 'query' in name or 'key' in name or 'value' in name:
                if 'prefix_query' not in name and 'memory_key' not in name and 'memory_value' not in name:
                    print('frozen:', name)
                    param.requires_grad = args.fine_tuning
                else:
                    print('trained:', name)
            elif name.split('.')[0] == 'embeddings':
                print('frozen:', name)
                param.requires_grad = args.fine_tuning
            else:
                print('trained:', name)

    if args.do_train and args.max_seq_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    # torch.distributed.init_process_group(backend='nccl')

    test_dataset_1c = ReasoningChainDataset(tokenizer, args.test_file + "test_1c.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num, prepare_candidate=True)
    test_dataset_2c = ReasoningChainDataset(tokenizer, args.test_file + "test_2c.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_3c = ReasoningChainDataset(tokenizer, args.test_file + "test_3c.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_2i = ReasoningInterDataset(tokenizer, args.test_file + "test_2i.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_3i = ReasoningInterDataset(tokenizer, args.test_file + "test_3i.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_pi = ReasoningInterDataset(tokenizer, args.test_file + "test_pi.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_2u = ReasoningInterDataset(tokenizer, args.test_file + "test_2u.json", args.max_seq_len,
                                            args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_ip = ReasoningInterChainDataset(tokenizer, args.test_file + "test_ip.json", args.max_seq_len,
                                                 args.max_ans_len, args.nentity, args.negative_num)
    test_dataset_up = ReasoningInterChainDataset(tokenizer, args.test_file + "test_up.json", args.max_seq_len,
                                                 args.max_ans_len, args.nentity, args.negative_num)

    collate_fc = partial(qa_collate, tokenizer=tokenizer, answer_features=test_dataset_1c.features,
                         args=args, pad_id=tokenizer.pad_token_id)

    test_dataloader_1c = DataLoader(test_dataset_1c,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_2c = SequentialDistributedSampler(test_dataset_2c)
    test_dataloader_2c = DataLoader(test_dataset_2c,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_3c = SequentialDistributedSampler(test_dataset_3c)
    test_dataloader_3c = DataLoader(test_dataset_3c,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_2i = SequentialDistributedSampler(test_dataset_2i)
    test_dataloader_2i = DataLoader(test_dataset_2i,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_3i = SequentialDistributedSampler(test_dataset_3i)
    test_dataloader_3i = DataLoader(test_dataset_3i,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_pi = SequentialDistributedSampler(test_dataset_pi)
    test_dataloader_pi = DataLoader(test_dataset_pi,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_2u = SequentialDistributedSampler(test_dataset_2u)
    test_dataloader_2u = DataLoader(test_dataset_2u,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_ip = SequentialDistributedSampler(test_dataset_ip)
    test_dataloader_ip = DataLoader(test_dataset_ip,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    # test_sampler_up = SequentialDistributedSampler(test_dataset_up)
    test_dataloader_up = DataLoader(test_dataset_up,
                                    batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                    collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(
        f"Num of test batches: {len(test_dataloader_1c) + len(test_dataloader_2c) + len(test_dataloader_3c) + len(test_dataloader_2i) + len(test_dataloader_3i)}")
    test_dataloader = [test_dataloader_1c, test_dataloader_2c, test_dataloader_3c, test_dataloader_2i,
                       test_dataloader_3i, test_dataloader_ip, test_dataloader_pi, test_dataloader_2u,
                       test_dataloader_up]

    train_dataset = ReasoningDataset(tokenizer, args, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                  collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

    if args.do_eval:
        eval_dataset_1c = ReasoningChainDataset(tokenizer, args.predict_file + "dev_1c.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num, prepare_candidate=True)
        eval_dataset_2c = ReasoningChainDataset(tokenizer, args.predict_file + "dev_2c.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_3c = ReasoningChainDataset(tokenizer, args.predict_file + "dev_3c.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_2i = ReasoningInterDataset(tokenizer, args.predict_file + "dev_2i.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_3i = ReasoningInterDataset(tokenizer, args.predict_file + "dev_3i.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_pi = ReasoningInterDataset(tokenizer, args.predict_file + "dev_pi.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_2u = ReasoningInterDataset(tokenizer, args.predict_file + "dev_2u.json", args.max_seq_len,
                                                args.max_ans_len, args.negative_num)
        eval_dataset_ip = ReasoningInterChainDataset(tokenizer, args.predict_file + "dev_ip.json", args.max_seq_len,
                                                     args.max_ans_len, args.negative_num)
        eval_dataset_up = ReasoningInterChainDataset(tokenizer, args.predict_file + "dev_up.json", args.max_seq_len,
                                                     args.max_ans_len, args.negative_num)

        eval_dataloader_1c = DataLoader(eval_dataset_1c,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_2c = DataLoader(eval_dataset_2c,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_3c = DataLoader(eval_dataset_3c,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_2i = DataLoader(eval_dataset_2i,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_3i = DataLoader(eval_dataset_3i,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_pi = DataLoader(eval_dataset_pi,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_2u = DataLoader(eval_dataset_2u,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_ip = DataLoader(eval_dataset_ip,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
        eval_dataloader_up = DataLoader(eval_dataset_up,
                                        batch_size=int(args.predict_batch_size / torch.cuda.device_count()),
                                        collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)

        logger.info(
            f"Num of dev batches: {len(eval_dataloader_1c) + len(eval_dataloader_2c) + len(eval_dataloader_3c) + len(eval_dataloader_2i) + len(eval_dataloader_3i)}")
        eval_dataloader = [eval_dataloader_1c, eval_dataloader_2c, eval_dataloader_3c, eval_dataloader_2i,
                           eval_dataloader_3i, eval_dataloader_ip, eval_dataloader_pi, eval_dataloader_2u,
                           eval_dataloader_up]

    if args.init_checkpoint != "":
        logger.info(f"Loading model from {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])

    if not args.deepspeed:
        model.to(device)

    if not args.fine_tuning:
        for name, param in model.bert.named_parameters():
            if 'query' in name or 'key' in name or 'value' in name:
                if 'prefix_query' not in name and 'memory_key' not in name and 'memory_value' not in name:
                    assert param.requires_grad == args.fine_tuning
                else:
                    assert param.requires_grad != args.fine_tuning
            elif name.split('.')[0] == 'embeddings':
                assert param.requires_grad == args.fine_tuning
            else:
                assert param.requires_grad != args.fine_tuning

    # if not args.fine_tuning:
    #     for name, param in model.bert.named_parameters():
    #         if 'memory_embeddings_table' in name:
    #             assert param.requires_grad != args.fine_tuning
    #         else:
    #             assert param.requires_grad == args.fine_tuning
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # optimizer2 = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.local_rank != -1 and not args.deepspeed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.deepspeed:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0 # gradient update step
        batch_step = 0 # forward batch count
        best_score = 0
        best_epoch = 0
        if args.init_checkpoint != "" and args.general_checkpoint:
            train_loss_meter = checkpoint['loss']
        else:
            train_loss_meter = AverageMeter()
        model.train()
        train_dataset = ReasoningDataset(tokenizer, args, train=True)
        if args.deepspeed:
            train_sampler = DistributedSampler(train_dataset, seed=args.seed, num_replicas=torch.cuda.device_count())
            train_dataloader = DataLoader(train_dataset, batch_size=int(args.train_batch_size/torch.cuda.device_count()), pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        print("**"*10, "warmup_steps", warmup_steps)
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        scheduler2 = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # if not args.deepspeed:
        #     if args.init_checkpoint != "" and args.general_checkpoint:
        #         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #         print(scheduler.state_dict())

        if not args.init_checkpoint:
            # logger.info('Checking testing....')
            # result = predict(args, test_dataloader_1c.dataset.features, model, test_dataloader, logger, epoch=0)
            logger.info('Start training....')
            if args.init_checkpoint != "" and args.general_checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = 0

            for epoch in range(start_epoch, int(args.num_train_epochs)):

                if args.deepspeed:
                    train_dataloader.sampler.set_epoch(epoch)

                for batch in tqdm(train_dataloader):

                    batch_step += 1

                    if not args.deepspeed:
                        batch = move_to_cuda(batch)
                    else:
                        batch = move_to_ds_cuda(batch, args.device)

                    logits_rk_1, rk_loss = model(batch,
                                                 is_prepare=False,
                                                 schema=args.training_inference_schema,
                                                 task='query')
                    logits_rk_1 = F.softmax(logits_rk_1, dim=-1)

                    if n_gpu > 1:
                        rk_loss = rk_loss.mean()

                    if args.gradient_accumulation_steps > 1 and not args.deepspeed:
                        rk_loss = rk_loss / args.gradient_accumulation_steps
                    if args.deepspeed:
                        rk_loss = model.backward(rk_loss)
                    else:
                        rk_loss.backward()

                    train_loss_meter.update(rk_loss.item())

                    if args.deepspeed:
                        model.step()
                    if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                        if not args.deepspeed:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler1.step()
                        model.zero_grad()
                        # global_step += 1

                    if batch_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                        logger.info("********** Epoch: %d; Iteration: %d; current loss: %s; lr: %s", epoch, batch_step,
                                    str(round(train_loss_meter.avg, 4)), str(scheduler1.get_last_lr()[0]))

                        logs = {}
                        logs["loss"] = round(train_loss_meter.avg, 4)
                        logs["learning_rate1"] = scheduler1.get_last_lr()[0]
                        logs["epoch"] = epoch
                        logs = rewrite_logs(logs)

                    logits_rk_1_geo, rk_loss_geo = model(batch,
                                                         is_prepare=False,
                                                         schema=args.training_inference_schema,
                                                         task='geo')
                    logits_rk_1_geo = F.log_softmax(logits_rk_1_geo, -1)

                    kl = torch.nn.functional.kl_div(logits_rk_1_geo, logits_rk_1, reduction='mean')
                    # print(str(kl.item()))
                    if n_gpu > 1:
                        rk_loss_geo = rk_loss_geo.mean()

                    if args.gradient_accumulation_steps > 1 and not args.deepspeed:
                        rk_loss_geo = rk_loss_geo / args.gradient_accumulation_steps
                    if args.deepspeed:
                        rk_loss_geo = model.backward(rk_loss_geo)
                    else:
                        rk_loss_geo.backward()

                    # grads = {}
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and (parameter.grad != None):
                            # grads[name] = parameter.grad
                            # print(name)
                            parameter.grad *= kl

                    train_loss_meter.update(rk_loss_geo.item())

                    if args.deepspeed:
                        model.step()
                    if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                        if not args.deepspeed:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler2.step()
                        model.zero_grad()
                        global_step += 1

                    if batch_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                        logger.info("********** Epoch: %d; Iteration: %d; current loss: %s; lr: %s; kl: %s", epoch, batch_step,
                                    str(round(train_loss_meter.avg, 4)), str(scheduler2.get_last_lr()[0]), str(kl.item()))

                        logs = {}
                        logs["loss"] = round(train_loss_meter.avg, 4)
                        logs["learning_rate2"] = scheduler2.get_last_lr()[0]
                        logs["epoch"] = epoch
                        logs = rewrite_logs(logs)

                logger.info(f"Saving model at epoch {epoch}!")
                save_model(model, optimizer, args)
                if args.do_eval:
                    result = predict(args, test_dataloader_1c.dataset.features, model, eval_dataloader, logger, epoch=epoch)
                    print(result)

                # result = predict(args, test_dataloader_1c.dataset.features, model, test_dataloader, logger, epoch=epoch)
                # print(result)

            logger.info("Training finished!")

    if args.do_test:
        print("Prediction Here", "*"*40)
        result = predict(args, test_dataloader_1c.dataset.features, model, test_dataloader, logger, epoch=0)
        logger.info(f"test performance {result}")
    
    save_model(model, optimizer, args)


def predict(args, entity_to_feed, model, eval_dataloader, logger, epoch, is_test=False):
    if args.local_rank in [-1, 0]:
        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataloader[0].dataset)}")
        logger.info(f"  Batch size = {eval_dataloader[0].batch_size}")

    model.eval()

    with torch.no_grad():
        batch_num = (args.nentity // args.predict_batch_size) if (args.nentity % args.predict_batch_size) == 0 else (args.nentity // args.predict_batch_size) + 1
        candidate_entities = []
        for i in range(batch_num):
            batch_entity_to_feed = {'entity_input_ids': entity_to_feed['entity_input_ids'][i * args.predict_batch_size:(i + 1) * args.predict_batch_size],
                                    'entity_masks': entity_to_feed['entity_masks'][i * args.predict_batch_size:(i + 1) * args.predict_batch_size],
                                    'entity_add_special_tokens_ids': entity_to_feed['entity_add_special_tokens_ids'][i * args.predict_batch_size:(i + 1) * args.predict_batch_size],
                                    'entity_is_add_special_tokens': entity_to_feed['entity_is_add_special_tokens'][i * args.predict_batch_size:(i + 1) * args.predict_batch_size]}
            batch_entity_to_feed = move_to_ds_cuda(batch_entity_to_feed, args.device)
            candidate_entities.append(model(batch_entity_to_feed, is_prepare=True))
        candidate_entities = torch.concat(candidate_entities, dim=0)

    qtype = ['1-chain', '2-chain', '3-chain', '2-inter', '3-inter', 'inter-chain', 'chain-inter', '2-union', 'union-chain']
    # qtype = ['4-chain', '5-chain', '3-inter-chain', 'inter-2-chain']
    metrics = {}
    for i, each_eval_dataloader in enumerate(eval_dataloader):
        index_host = None
        all_index = None

        logits_host = None
        all_logits = None

        for batch in tqdm(each_eval_dataloader):
            batch_to_feed = move_to_ds_cuda(batch, args.device)
            batch_index = batch_to_feed["index"]

            with torch.no_grad():
                output = model(batch_to_feed, candidate_rep=candidate_entities, schema=args.training_inference_schema)
                logits = output.contiguous()

            logits = _pad_across_processes(logits, args)
            logits = _nested_gather(logits, args)
            logits_host = logits if logits_host is None else nested_concat(logits_host, logits,
                                                                            padding_index=-100)

            batch_index = _pad_across_processes(batch_index, args)
            batch_index = _nested_gather(batch_index, args)
            index_host = batch_index if index_host is None else nested_concat(index_host, batch_index,
                                                                            padding_index=-100)

        if index_host is not None:
            index = nested_numpify(index_host)
            all_index = index if all_index is None else nested_concat(all_index, index, padding_index=-100)
            
        if logits_host is not None:
            _logits = nested_numpify(logits_host)
            all_logits = _logits if all_logits is None else nested_concat(all_logits,
                                                                            _logits,
                                                                            padding_index=-100)

        all_logits = np.array(all_logits)
        all_index = np.array(all_index)
        all_ids = all_index.reshape(-1).tolist()

        eval_metrics_all = postprocess_qa_predictions(
            args,
            examples=each_eval_dataloader.dataset.data,
            all_indexes=all_ids,
            predictions=all_logits,
            logger=logger,
            is_test=is_test
        )
        for metric in eval_metrics_all.keys():
            if metric in metrics:
                metrics[metric] += eval_metrics_all[metric]
            else:   
                metrics[metric] = eval_metrics_all[metric]
        
        if args.local_rank in [-1, 0]:
            logger.info(f"{qtype[i]} Metrics: {eval_metrics_all}")
    
    for metric in metrics.keys():
        metrics[metric] = round(metrics[metric]/9, 4)

    if args.local_rank in [-1, 0]:
        logger.info("*"*100)
        logger.info(f"All Metrics: {metrics}")

    model.train()

    for key in list(metrics.keys()):
        if not key.startswith("eval_"):
            metrics[f"eval_{key}"] = metrics.pop(key)
    return metrics


def save_model(model, optimizer, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    # argparse_dict = vars(args)
    # with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
    #     json.dump(argparse_dict, fjson)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.output_dir, 'checkpoint')
    )


def postprocess_qa_predictions(
    args,
    examples,
    all_indexes,
    predictions,
    logger,
    is_test=False
):
    assert len(predictions) >= len(examples), f"Got {len(predictions)} predictions and {len(examples)} examples."

    # The dictionaries we have to fill.
    all_pred_sp = collections.OrderedDict()

    # Let's loop over all the examples!
    logs = []
    for i, example_index in tqdm(enumerate(all_indexes)):
        assert example_index < len(examples)
        example = examples[example_index]

        score = predictions[i]
        score = torch.Tensor(score).to(args.device)
        score = score.unsqueeze(0)
        
        ans_list = example["ans"]
        ans = set(ans_list)
        hard_ans_list = example["hard_ans"]
        all_idx = set(range(args.nentity))
        false_ans = all_idx - ans
        false_ans_list = list(false_ans)

        ans_idxs = np.array(hard_ans_list)
        vals = np.zeros((len(ans_idxs), args.nentity))
        vals[np.arange(len(ans_idxs)), ans_idxs] = 1
        axis2 = np.tile(false_ans_list, len(ans_idxs))
        axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
        vals[axis1, axis2] = 1
        b = torch.Tensor(vals).to(args.device)

        filter_score = b*score

        argsort = torch.argsort(filter_score, dim=1, descending=True)
        ans_tensor = torch.LongTensor(hard_ans_list).to(args.device)

        argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
        ranking = (argsort == 0).nonzero()
        ranking = ranking[:, 1]
        ranking = ranking + 1

        ans_vec = np.zeros(args.nentity)
        ans_vec[ans_list] = 1

        hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
        hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
        hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
        mrm = torch.mean(ranking.to(torch.float)).item()
        mrrm = torch.mean(1./ranking.to(torch.float)).item()
        num_ans = len(hard_ans_list)

        hits1m_newd = hits1m
        hits3m_newd = hits3m
        hits10m_newd = hits10m
        mrm_newd = mrm
        mrrm_newd = mrrm

        logs.append({
                'MRRm_new': mrrm_newd,
                'MRm_new': mrm_newd,
                'HITS_1m_new': hits1m_newd,
                'HITS_3m_new': hits3m_newd,
                'HITS_10m_new': hits10m_newd,
                'num_answer': num_ans,
                'type': example['type']
            })

    metrics = {}
    # num_answer = sum([log['num_answer'] for log in logs])
    for metric in logs[0].keys():
        if metric == 'num_answer' or metric == 'type':
            continue
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)

    return metrics


def _pad_across_processes(tensor, args, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, args, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, args, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = _nested_gather(size, args).cpu()

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


def _nested_gather(tensors, args):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if args.local_rank != -1:
        tensors = distributed_concat(tensors)
    return tensors


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


if __name__ == "__main__":

    main()

    #  python train.py --train_file ../Data/NELL/train.json --predict_file ../Data/NELL/ --test_file ../Data/NELL/ --do_train --do_eval --do_test --nentity 63361 --nrelation 400 --is_memory --memory_size 20 --prefix train_NELL --train_batch_size 128 --learning_rate 5e-5 --num_train_epochs 30