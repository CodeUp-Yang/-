
python3 run_classifier.py --task_name=my --do_predict=true --data_dir=data --vocab_file=uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=output --max_seq_length=256 --learning_rate=5e-5  --output_dir=model_result/testset


# export BERT_BASE_DIR=/Users/imac/Desktop/finish/bert/uncased_L-12_H-768_A-12
# export DATA_DIR=data
# # TRAINED_CLASSIFIER为刚刚训练的输出目录，无需在进一步指定模型模型名称，否则分类结果会不对
# export TRAINED_CLASSIFIER=output/test
 


# python3 run_classifier.py \
#   --task_name=my \
#   --do_predict=true \
#   --data_dir=$DATA_DIR \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --bert_config_file=$BERT_BASE_DIR·/bert_config.json \
#   --init_checkpoint=$TRAINED_CLASSIFIER \
#   --max_seq_length=512 \
#   --output_dir=mymodel
