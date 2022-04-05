mkdir -p log
PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --mpi=pmi2 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 \
'python -u -m algorithm.main --config config.yaml $2 2>&1|tee log/train_$3\_log.txt'
