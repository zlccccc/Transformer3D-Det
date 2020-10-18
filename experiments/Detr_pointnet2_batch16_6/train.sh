mkdir -p log
CUDA_VISIBLE_DEVICES=$1 \
python ../../algorithm/main.py --config config.yaml $2  2>&1 |tee log/train_log.txt
