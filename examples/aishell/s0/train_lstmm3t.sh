export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/server/x86/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

nohup python wenet/bin/train_lstmm3t.py --gpu 1 \
      --config conf/train_lstmspner.yaml \
      --seed 777 \
      --ctcw 0.1 \
      --data_type raw \
      --symbol_table ner_data/dict/lstm_ner_lang_char.txt \
      --ner_dict ner_data/dict/ner_label.txt \
      --train_data ner_data/train/data.list \
      --cv_data ner_data/dev/data.list \
      --model_dir exp/lstm_m3t \
      --ddp.init_method file:///opt/data/private/slzhou/wenet/examples/aishell/s0/exp/transformer/ddp_init \
      --ddp.world_size 1 \
      --ddp.rank 0 \
      --ddp.dist_backend gloo \
      --num_workers 1 \
      --cmvn exp/transformer/global_cmvn \
      --checkpoint exp/lstm_m3t/current.pt \
      --pin_memory > lstm_m3t_ctcw0.1.log 2>&1 &