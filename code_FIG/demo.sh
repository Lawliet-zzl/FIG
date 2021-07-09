
DATASET='CIFAR10'
MODEL='ResNet18'

python main_FIG.py --dataset ${DATASET} --model ${MODEL} --alg='Baseline'

python main_FIG.py --dataset ${DATASET} --model ${MODEL} --alg='FIG'