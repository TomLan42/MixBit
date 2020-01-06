# python cifar_train_eval.py --model_name vgg \
#         --wbit_code 1 8 1 1 4 1 \
#         --abit_code 1 8 1 1 4 1 \
#         --ratio_code 1 1 1 1 1 1

# python cifar_train_eval.py --model_name vgg \
#         --wbit_code 1 4 4 1 4 4 \
#         --abit_code 1 4 4 1 4 4 \
#         --ratio_code 1 1 1 1 1 1

# python cifar_train_eval.py --model_name vgg \
#         --wbit_code 1 4 4 1 4 4 \
#         --abit_code 1 4 4 1 4 4 \
#         --ratio_code 1 1 1 1 1 1


python cifar_train_eval.py --model_name vgg  --max_epochs 30 --lr 0.01 \
        --wbit_code 1 1	1 1 4 1 \
        --abit_code 1 1	1 1 4 1 \
        --ratio_code 1 1 1 1 1 1 \
        --pretrain --pretrain_dir ./ckpt/vgg[111111][111111][1.01.01.01.01.01.0]cifar100/checkpoint.t7


python cifar_train_eval.py --model_name vgg  --max_epochs 30 --lr 0.01 \
        --wbit_code 1 1	1 1 1 4 \
        --abit_code 1 1	1 1 1 4 \
        --ratio_code 1 1 1 1 1 1 \
        --pretrain --pretrain_dir ./ckpt/vgg[111111][111111][1.01.01.01.01.01.0]cifar100/checkpoint.t7


python cifar_train_eval.py --model_name vgg  --max_epochs 30 --lr 0.01 \
        --wbit_code 1 1	1 4 1 1 \
        --abit_code 1 1	1 4 1 1 \
        --ratio_code 1 1 1 1 1 1 \
        --pretrain --pretrain_dir ./ckpt/vgg[111111][111111][1.01.01.01.01.01.0]cifar100/checkpoint.t7


python cifar_train_eval.py --model_name vgg  --max_epochs 30 --lr 0.01 \
        --wbit_code 1 1	4 1 1 1 \
        --abit_code 1 1	4 1 1 1 \
        --ratio_code 1 1 1 1 1 1 \
        --pretrain --pretrain_dir ./ckpt/vgg[111111][111111][1.01.01.01.01.01.0]cifar100/checkpoint.t7


python cifar_train_eval.py --model_name vgg  --max_epochs 30 --lr 0.01 \
        --wbit_code 1 4	1 1 1 1 \
        --abit_code 1 4	1 1 1 1 \
        --ratio_code 1 1 1 1 1 1 \
        --pretrain --pretrain_dir ./ckpt/vgg[111111][111111][1.01.01.01.01.01.0]cifar100/checkpoint.t7