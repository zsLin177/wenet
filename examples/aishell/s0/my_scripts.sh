# nohup bash run.sh --stage 4 --stop-stage 4 > train_transformer.log 2>&1 &
# (204600)

# nohup bash run.sh --stage 4 --stop-stage 4 > reload_train_transformer.log 2>&1 &
# (102370)

export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/server/x86/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

python wenet/bin/recognize.py --gpu 1 \
      --mode ctc_greedy_search \
      --config exp/transformer/train.yaml \
      --data_type raw \
      --test_data data/test/data.list \
      --checkpoint exp/transformer/avg_10_before50.pt \
      --beam_size 10 \
      --batch_size 20 \
      --penalty 0.0 \
      --dict data/dict/lang_char.txt \
      --ctc_weight 0.5 \
      --reverse_weight 0.0 \
      --result_file exp/transformer/test_ctc_greedy_search/text