#!/bin/bash

run='False'
run_lr='False'
run_seed='False'
is_448=''
augs='medaugs'
ls=''
sd=''

device=0
serial=1
seed=1
lr=0.003

dataset_name='cub'
model_name='vit_b16 --classifier cls --cfg_method configs/methods/glsim.yaml'

lr_array=('0.03' '0.01' '0.003' '0.001')
seed_array=('1' '10' '100')

VALID_ARGS=$(getopt  -o '' --long run,run_lr,run_seed,is_448,weak_augs,ls,sd,device:,serial:,seed:,lr:,dataset_name:,model_name: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --run)
        run='True'
        shift 1
        ;;
    --run_lr)
        run_lr='True'
        shift 1
        ;;
    --run_seed)
        run_seed='True'
        shift 1
        ;;
    --is_448)
        is_448=' --cfg_is configs/settings/ft_is448.yaml'
        shift 1
        ;;
    --weak_augs)
        augs='weakaugs'
        shift 1
        ;;
    --ls)
        ls=' --ls'
        shift 1
        ;;
    --sd)
        sd=' --sd 0.1'
        shift 1
        ;;
    --device)
        device=${2}
        shift 2
        ;;
    --serial)
        serial=${2}
        shift 2
        ;;
    --seed)
        seed=${2}
        shift 2
        ;;
    --lr)
        lr=${2}
        shift 2
        ;;
    --dataset_name)
        dataset_name=${2}
        shift 2
        ;;
    --model_name)
        model_name=${2}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

# CMD="CUDA_VISIBLE_DEVICES=${device} nohup python -u tools/train.py --serial ${serial} --cfg configs/${dataset_name}_ft_is224_medaugs.yaml${is_448}"
CMD="nohup python -u tools/train.py --serial ${serial} --cfg configs/${dataset_name}_ft_is224_${augs}.yaml${is_448}${ls}${sd}"
echo "${CMD}"

# single run
if [[ "$run" == "True" ]]; then
    echo "${CMD} --seed ${seed} --lr ${lr} --model_name ${model_name}"
    ${CMD} --seed ${seed} --lr ${lr} --model_name ${model_name}
fi

# lr run
if [[ "$run_lr" == "True" ]]; then
    for rate in ${lr_array[@]}; do
        echo "${CMD} --seed ${seed} --lr ${rate} --model_name ${model_name} --train_trainval"
        ${CMD} --seed ${seed} --lr ${rate} --model_name ${model_name} --train_trainval
    done
fi


# seed run
if [[ "$run_seed" == "True" ]]; then
    for seed in ${seed_array[@]}; do
        echo "${CMD} --seed ${seed} --lr ${lr} --model_name ${model_name}"
        ${CMD} --seed ${seed} --lr ${lr} --model_name ${model_name}
    done
fi
