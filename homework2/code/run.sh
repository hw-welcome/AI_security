# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100

# for resnet
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 160 --batch-size 256 --lr 0.1 -ct 10
