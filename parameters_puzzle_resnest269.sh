# Training parameters


## Train base scripts

### Train Puzzle
# --re_loss_option masking \
# --re_loss L1_Loss \
#CUDA_VISIBLE_DEVICES=1 python3 train_classification_with_puzzle.py \
#   --architecture resnest269 \
#   --alpha_schedule 0.50 --alpha 4.00 \
#   --lr 0.01 \
#   --tag ResNest269@Puzzle@CLF@noaug \
#   --data_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/ \
#   --loss_option cl_pcl_re_conf \
#   --re_loss_option masking \
#   --image_size 256 \
#   --batch_size 8 \
#   --augment colorjitter
#
#CUDA_VISIBLE_DEVICES=0 python3 inference_classification.py \
#    --architecture resnest269 \
#    --tag ResNest269@Puzzle@CLF@noaug \
#    --domain train_aug \
#    --data_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/
##
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
#    --experiment_name ResNest269@Puzzle@CLF@noaug@train@scale=0.5,1.0,1.5,2.0 \
#    --domain train \
#    --gt_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/SegmentationClass/ \


#> experiments/predictions/puzzlecam_train.txt

#CUDA_VISIBLE_DEVICES=0 python3 make_affinity_labels.py \
#    --experiment_name ResNet34@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 \
#    --domain train_aug --fg_threshold 0.40 --bg_threshold 0.10 \
#    --data_dir ../../queimadas/dataset2_bkp/
#
#CUDA_VISIBLE_DEVICES=0 python3 train_affinitynet.py --architecture resnet34 --tag AffinityNet@ResNet-34@Puzzle --label_name ResNet34@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10 --data_dir ../../queimadas/dataset2_bkp/ --batch_size 16 --image_size 256

#CUDA_VISIBLE_DEVICES=0 python3 inference_rw.py --architecture resnet34 --model_name AffinityNet@ResNet-34@Puzzle --cam_dir ResNet34@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train_aug --data_dir ../../queimadas/dataset2_bkp/
#```
#
#```bash
python3 make_pseudo_labels.py \
    --experiment_name ResNest269@Puzzle@CLF@noaug@train@scale=0.5,1.0,1.5,2.0 \
    --domain train_aug --threshold 0.65 --crf_iteration 0 \
    --data_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/
#
python3 evaluate.py \
    --experiment_name ResNest269@Puzzle@CLF@train@scale=0.5,1.0,1.5,2.0@crf=0_paletted\
    --mode png \
    --gt_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/SegmentationClass/ \


# python3 evaluate.py \
#     --experiment_name ResNest269@Puzzle@CLF@noaug@train@scale=0.5,1.0,1.5,2.0@crf=0\
#     --mode png \
#     --gt_dir /mnt/58B0FAA2B0FA85B2/max/datasets/pantanal/queimadas/dataset2_bkp/SegmentationClass/ \
# > experiments/predictions/pseudo_label.txt
#```
#
#```bash
#CUDA_VISIBLE_DEVICES=0 python3 train_segmentation.py --backbone resnet34 --mode bn --use_gn False --tag DeepLabv3+@ResNet-34@Fix@GN --label_name AffinityNet@ResNet-34@Puzzle@train@beta=10@exp_times=8@rw@crf=1 --data_dir ../../queimadas/dataset2_bkp/ --image_size 256 --batch_size 16
#```
#
#### Evaluate the model
#
#```bash
#CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnet34 --mode bn --use_gn False --tag DeepLabv3+@ResNet-34@Fix@GN --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir ../../queimadas/dataset2_bkp/
#
#CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnet34 --mode bn --use_gn False --tag DeepLabv3+@ResNet-34@Fix@GN --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir ../../queimadas/dataset2_bkp/ --domain test
#
#python3 evaluate.py --experiment_name DeepLabv3+@ResNet-34@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10 --domain val --mode png --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/val.txt
#
#python3 evaluate.py --experiment_name DeepLabv3+@ResNet-34@Fix@GN@test@scale=0.5,1.0,1.5,2.0@iteration=10 --domain test --mode png --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/test.txt
#```