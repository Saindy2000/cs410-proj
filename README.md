# README

### Model Testing and Training

First, in console or terminal, run:

```python
cd code
```

If you want to load trained model parameters and test its performance:

```python
python ./code/train.py --load True
```

If you want to retrain the model,

```python
python ./code/train.py --load False
```

If you want to edit the hyperparameters, open dataset.py and edit class config.

### Required Packages

- pytorch (or pytorch-cuda)
- torchtext
- numpy
- scipy

### What's included in this repository

- proposal
- progress report
- final report
- ./code: codes and data
  - Digital_Music_5.json: raw dataset
  - text.pkl, test.pkl, user-item.pkl: Preprocessed dataset, including training dataset, testing dataset, user-item graph extracted from training dataset
  - model_best.pt: trained model parameters
  - log.txt, log_abla.txt: saved training logs for running our model and baseline
  - python files:
    - data_preprocess.py: codes for data preprocessing
    - visualization.py: codes for drawing loss and metric curves
    - model.py, lightGCN.py, transformer.py: implementation of our model
    - dataset.py: input data processing
    - train.py: training codes
    - utils.py: functional codes, e.g., implementation of ndcg, recall, prec, etc.