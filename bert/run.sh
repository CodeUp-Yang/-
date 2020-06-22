
python3 run_classifier.py --task_name=my --do_train=true --do_eval=true --data_dir=data --vocab_file=uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=5 --output_dir=output
