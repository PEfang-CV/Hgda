# Human-Guided Data Augmentation（under review）

## Introduction

This repository provides the related code for our research on Human-Guided Data Augmentation for Surface Defect Recognition, including Human Ranking Software, Reward Model, etc.

## Implementation

**Environment installation**

```shell
conda create -n hgda python==3.8
conda activate hgda
pip install -r requirements.txt
```

**Data preparation**

+ Download NEU-Seg dataset form [here](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
+ Download NEU-CLS dataset form [here](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
+ Download MT dataset form [here](https://github.com/abin24/Magnetic-tile-defect-datasets.)
+ The tire datasets is sourced from corporate collaborations. If you want to the data，please send  the email to us. we will send the download link once we receive and confirm your signed agreement.

**Human Ranking Software**

+ Set the working directory in the following format

    ```shell
    working directory/
    └── SourceData
        ├── GeneratedImg1
        ├── GeneratedImg2
        ├── GeneratedImg3
        ├── GeneratedImg4
        ├── SemanticLabel
        └── ClasssLabel 
    └── RankingResults
        ├── 1
        ├── 2
        ├── 3
        ├── 4
        ├── SemanticLabel   
        ├── ClasssLabel
        └── HunmanRanking.txt   # Record hunman ranking information
    ```

+ Set the working directory path in ./MakeHumanEvaluationDataset/humanRankingSoftware.py

+ run the human ranking software

  ```shell
  python ./MakeHumanEvaluationDataset/humanRankingSoftware.py
  ```


+ The ranking of software operations

  ![image-20240705093514242](./figs/image-20240705093514242.png)

**Reward Model**

* Set the HumanEvaluationDataset in the following format

    ```shell
    HumanEvaluationDataset/
    └── train
        ├── 1
        ├── 2
        ├── 3
        ├── 4
        ├── SemanticLabel   
        └── ClasssLabel
    └── test
        ├── 1
        ├── 2
        ├── 3
        ├── 4
        ├── SemanticLabel   
        └── ClasssLabel
    └── trainname.txt   # The name of the training image
    └── evalnname.txt   # The name of the validation image
    ```

* Training and validating the reward model

    ```shell
    python RewardModelTrain.py --train_path ./HumanEvaluationDataset/train --train_files ./HumanEvaluationDataset/train/trainname.txt --eval_path  ./HumanEvaluationDataset/test --eval_files ./HumanEvaluationDataset/train/evalnname.txt --save_dir ./chekpoints
    ```

* or just run the script

    ```shell
    sh RunRewardModel.sh
    ```

**Conditional Diffusion Model (CDM)**

+ We provide trained model available for download in the [here](https://drive.google.com/file/d/1M93euJcA1EUvlPcqhNfpcVszGjxB3ZYs/view?usp=drive_link)

+ Generating defect images  by CDM

  ```she
  MODEL_PATH="--checkmodel_path checkpoint.pt"
  SAVE_PATH="--out_file Results/NEU-Seg-demo-Results-ddim50"
  MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
  python CDM/sample.py $MODEL_PATH $SAVE_PATH $MODEL_FLAGS --classifier_scale 10.0  $SAMPLE_FLAGS --timestep_respacing ddim50
  ```
* or run the script
    ```shell
    cd CDM && sh run.sh
    ```

## Notes
+ The code for this project mainly refers to [guided-diffusion](https://github.com/openai/guided-diffusion). Thanks for the authors for their efforts.
+ If there are any issues with the code, please  send the email  to us.

