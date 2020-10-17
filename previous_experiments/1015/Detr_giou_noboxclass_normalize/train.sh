mkdir -p log
python ../../algorithm/main.py --config config.yaml --gpu $1 $2  2>&1 |tee log/train_log.txt
