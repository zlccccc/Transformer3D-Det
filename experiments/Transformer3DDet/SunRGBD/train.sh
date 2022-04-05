mkdir -p log
PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 \
python -u -m algorithm.main --config config.yaml  2>&1|tee log/train_log.txt
