# Trusted Unified Feature-Neighborhood Dynamics for Multi-View Classification

## Introduction

Multi-view classification (MVC) is a challenging task due to the domain gaps and inconsistencies that often arise across different views, leading to uncertainties during the fusion process. Traditional methods, like Evidential Deep Learning (EDL), have been employed to address view uncertainty. However, these approaches often rely heavily on the Dempster-Shafer combination rule, which is sensitive to conflicting evidence and typically overlooks the critical role of neighborhood structures within multi-view data.

To overcome these limitations, we introduce **Trusted Unified Feature-NEighborhood Dynamics (TUNED)**, a novel model designed to enhance robustness in MVC tasks. Our method effectively integrates both local and global feature-neighborhood (F-N) structures to support robust decision-making. By extracting local F-N structures within each view and employing a selective Markov random field to manage cross-view dependencies, TUNED addresses potential uncertainties and conflicts more effectively than existing approaches.

Moreover, TUNED includes a shared parameterized evidence extractor that learns global consensus based on local F-N structures, thereby improving the integration of multi-view features. Our experiments on benchmark datasets demonstrate that TUNED significantly enhances accuracy and robustness, especially in scenarios with high uncertainty and conflicting views.



## How to Run

You can run the training scripts for each dataset individually by using the following commands.

### 1. PIE Dataset (Normal)

To train on the PIE dataset without conflicts:

```sh
python train_script.py --dataset PIE --model-path pie_normal --lr 0.001
```

### 2. PIE Dataset (Conflict)

To train on the PIE dataset with conflicts:

```sh
python train_script.py --dataset PIE --model-path pie_conflict --batch-size 200 --add-conflict
```

### 3. HandWritten Dataset (Normal)

To train on the HandWritten dataset without conflicts:

```sh
python train_script.py --dataset HandWritten --model-path handwritten --lr 0.001
```

### 4. HandWritten Dataset (Conflict)

To train on the HandWritten dataset with conflicts:

```sh
python train_script.py --dataset HandWritten --model-path handwritten --batch-size 200 --add-conflict
```

### 5. Scene Dataset (Normal)

To train on the Scene dataset without conflicts:

```sh
python train_script.py --dataset Scene --model-path scene --epochs 600 --annealing_step 100
```

### 6. Scene Dataset (Conflict)

To train on the Scene dataset with conflicts:

```sh
python train_script.py --dataset Scene --model-path scene --epochs 600 --annealing_step 100 --add-conflict
```

## Running All Experiments

If you want to run all the experiments in one go, you can execute the `train_eval.sh` script:

```sh
sh train_eval.sh
```

