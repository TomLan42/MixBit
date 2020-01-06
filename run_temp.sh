python cifar_train_eval.py --model_name vgg \
        --wbit_code 1 8 1 1 4 1 \
        --abit_code 1 8 1 1 4 1 \
        --ratio_code 1 1 1 1 1 1

python cifar_train_eval.py --model_name vgg \
        --wbit_code 1 4 4 1 4 4 \
        --abit_code 1 4 4 1 4 4 \
        --ratio_code 1 1 1 1 1 1

python cifar_train_eval.py --model_name vgg \
        --wbit_code 1 4 4 1 4 4 \
        --abit_code 1 4 4 1 4 4 \
        --ratio_code 1 1 1 1 1 1
