import argparse

def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str,
                        default="../Data/NELL/train.json") # FB15k-237-betae
    parser.add_argument("--predict_file", type=str,
                        default="../Data/NELL/")
    parser.add_argument("--test_file", type=str,
                        default="../Data/NELL/")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="for final test submission")
    parser.add_argument("--training_inference_schema", default="matching", type=str)  # classification matching

    # model
    parser.add_argument("--model_name",
                        default="bert-base-cased", type=str) # distilbert-base-cased  bert-base-cased
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--init_folder", type=str,
                        help="Initial folder (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=64, type=int)
    parser.add_argument("--max_ans_len", default=25, type=int)
    parser.add_argument('--fp16', default=True)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--predict_batch_size", default=10,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--save-prediction", default="", type=str)
    parser.add_argument("--negative_num", default=128, type=int)
    parser.add_argument("--nentity", default=63361, type=int)  # 63361  14505
    parser.add_argument("--nrelation", default=400, type=int)  # 400  474
    parser.add_argument("--add_path", default=False, type=bool)
    parser.add_argument("--global_negative_num", default=128, type=int)
    parser.add_argument("--fine_tuning", action='store_true', help="Whether fine-tune the PLM.")
    parser.add_argument("--is_memory", action='store_true', help="Whether use memory.")
    parser.add_argument("--memory_size", default=20, type=int)

    return parser

def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="train_NELL")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--num_q_per_gpu", default=1)
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval-period', type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--shared-norm", action="store_true")
    parser.add_argument("--use-adam", action="store_true", help="use adam or adamW")
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed")
    parser.add_argument("--general_checkpoint", action="store_true")
    parser.add_argument("--warmup-ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--label_smoothing_factor", default=0.0, type=float, help="label smoothing factor.")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    return parser.parse_args()
