export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/server/x86/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

nohup python wenet/bin/train_speech_ner.py --gpu 0 \
      --config conf/tmp.yaml \
      --seed 777 \
      --ctcw 0.1 \
      --data_type raw \
      --symbol_table ner_data/dict/ner_lang_char.txt \
      --ner_dict ner_data/dict/ner_label.txt \
      --train_data ner_data/train/data.list \
      --cv_data ner_data/dev/data.list \
      --model_dir exp/m3t_ctcw0.1_h256 \
      --ddp.init_method file:///opt/data/private/slzhou/wenet/examples/aishell/s0/exp/transformer/ddp_init \
      --ddp.world_size 1 \
      --ddp.rank 0 \
      --ddp.dist_backend gloo \
      --num_workers 1 \
      --cmvn exp/transformer/global_cmvn \
      --pin_memory  > m3t_ctcw0.1_h256.log 2>&1 &