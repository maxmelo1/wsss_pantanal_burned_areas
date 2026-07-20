# Training parameters


## Train base scripts

### Train Puzzle

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_classification_with_puzzle.py --architecture resnest101 --re_loss_option masking --re_loss L1_Loss --alpha_schedule 0.50 --alpha 4.00 --tag ResNeSt101@Puzzle@optimal --data_dir ../../queimadas/dataset2_bkp/ --loss_option cl_pcl_re_conf --image_size 256 --batch_size 16 --augment colorjitter
```

- Color jitter augmentation: brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1.

### Random walk

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_classification.py --architecture resnest101 --tag ResNeSt101@Puzzle@optimal --domain train_aug --data_dir ../../queimadas/dataset2_bkp/

python3 evaluate.py --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/puzzlecam_train.txt

python3 make_affinity_labels.py --experiment_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train_aug --fg_threshold 0.40 --bg_threshold 0.10 --data_dir ../../queimadas/dataset2_bkp/
```

- AffinityNet

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_affinitynet.py --architecture resnest101 --tag AffinityNet@ResNeSt-101@Puzzle --label_name ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0@aff_fg=0.40_bg=0.10 --data_dir ../../queimadas/dataset2_bkp/ --batch_size 16 --image_size 256
```


### Segmentation model

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_rw.py --architecture resnest101 --model_name AffinityNet@ResNeSt-101@Puzzle --cam_dir ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0 --domain train_aug --data_dir ../../queimadas/dataset2_bkp/
```

```bash
python3 make_pseudo_labels.py --experiment_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw --domain train_aug --threshold 0.35 --crf_iteration 1 --data_dir ../../queimadas/dataset2_bkp/

python3 evaluate.py --experiment_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw@crf=1 --mode png --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/pseudo_label.txt
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_segmentation.py --backbone resnest101 --mode bn --use_gn False --tag DeepLabv3+@ResNeSt-101@Fix@GN --label_name AffinityNet@ResNeSt-101@Puzzle@train@beta=10@exp_times=8@rw@crf=1 --data_dir ../../queimadas/dataset2_bkp/ --image_size 256 --batch_size 16
```

### Evaluate the model

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnest101 --mode bn --use_gn False --tag DeepLabv3+@ResNeSt-101@Fix@GN --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir ../../queimadas/dataset2_bkp/

CUDA_VISIBLE_DEVICES=0 python3 inference_segmentation.py --backbone resnest101 --mode bn --use_gn False --tag DeepLabv3+@ResNeSt-101@Fix@GN --scale 0.5,1.0,1.5,2.0 --iteration 10 --data_dir ../../queimadas/dataset2_bkp/ --domain test

python3 evaluate.py --experiment_name DeepLabv3+@ResNeSt-101@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10 --domain val --mode png --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/val.txt

python3 evaluate.py --experiment_name DeepLabv3+@ResNeSt-101@Fix@GN@test@scale=0.5,1.0,1.5,2.0@iteration=10 --domain test --mode png --gt_dir ../../queimadas/dataset2_bkp/SegmentationClass/ > experiments/predictions/test.txt
```