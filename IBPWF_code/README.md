# Code for Nonparametric Bayesian Factorization for Continual Learning. 
## Please See `requirements.txt` for dependencies.

*****************

## Generate continual learning tasks for PermMNIST/SplitMNIST/SplitCIFAR10.
    
    Example:
    - Run `python create_benchmarks/create_permuted_mnist_tasks.py`

## Training

    Example:
    - Run `train_continual.py --config=configs/permutedMNIST10_ibpWF.yaml` to train the model. The checkpoint will be saved in `data/results/` directory.

## Evaluation

### Incremental Task setting
    
    Example:
    - Run `test_continual.py --config=data/results/split_mnist_results/seed1231_kappa0.5_alpha200.0/config.yaml --setting=multitask` to evaluate on incremental task setting.

#### Result for Incremental task

Accuracy = [# Correct] / [# Total ].

Task: 1         Testing Accuracy 0.978400

Task: 2         Testing Accuracy 0.978600

Task: 3         Testing Accuracy 0.977000

Task: 4         Testing Accuracy 0.978500

Task: 5         Testing Accuracy 0.977200

Task: 6         Testing Accuracy 0.978700

Task: 7         Testing Accuracy 0.978100

Task: 8         Testing Accuracy 0.974300

Task: 9         Testing Accuracy 0.978500

Task: 10        Testing Accuracy 0.979300

Average Accuracy (Across 10 classes/task): .9778

### Incremental Class setting

    Example:
    - Run `test_continual.py --config=data/results/split_mnist_results/seed1231_kappa0.5_alpha200.0/config.yaml  --setting=continual` to evaluate on incremental class setting.

#### Result for Incremental class

Task Inference Accuracy: 1.000000

Testing Accuracy (Across 10*10 classes): .9776 

