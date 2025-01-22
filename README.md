# PPMT

The demo code of Paper "Progressive Prefix-Memory Tuning for Complex Logical Query Answering on Knowledge Graphs" for IJCAI 2025 anonymous review.

## Setting up
1. To set up the environment, please install the packages in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

2. Unzip `Data.zip` to the directory `/Data`.

3. Unzip `logs.zip` (fine-tuned model, which will be uploaded when the paper is accepted) to the directory `/src/logs`.

4. Download the pretrained BERT from [huggingface](https://huggingface.co/google-bert/bert-base-cased/tree/main)

5. Then you can run the following scripts to train the model:
```bash
cd src
python train.py --train_file ../Data/FB15k-237/train.json --predict_file ../Data/FB15k-237/ --test_file ../Data/FB15k-237/ --do_train --do_eval --do_test --nentity 14505 --nrelation 474 --is_memory --memory_size 20 --prefix train_FB15k-237 --train_batch_size 128 --learning_rate 5e-5 --num_train_epochs 30
```

6. If you want to first test the model, you can run the following scripts:
```bash
cd src
python train.py --train_file ../Data/FB15k-237/train.json --predict_file ../Data/FB15k-237/ --test_file ../Data/FB15k-237/ --do_test --init_checkpoint ./logs/06-28-2024/train_FB15k-237-bsz128-lr5e-05-epoch30.0-maxlen128_neg128/checkpoint --nentity 14505 --nrelation 474 --is_memory --memory_size 20 --prefix train_FB15k-237 --train_batch_size 128 --learning_rate 5e-5 --num_train_epochs 30
```

