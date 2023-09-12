# Wyze Rule Recommendation Example Code
## Introduction
This is the example code for the Wyze rule recommendation challenge hosted in HF (https://huggingface.co/spaces/competitions/wyze-rule-recommendation), which reproduces the GraphRule algorithm for this dataset. The GraphRule is the centralized training of [FedRule](https://arxiv.org/abs/2211.06812). This is only for demonstration purposes and serves as a simple baseline model. We do not perform any hyperparameter optimization. The key steps implemented here include:

- Loading and preprocessing the Wyze rule and device datasets
- Constructing user-rule and user-device graphs from the data
- Applying graph neural network propagation and embedding techniques
- Training a model on the centralized graph embeddings
- Using the model to predict missing rules for new users

This is intended as a sample starter code to illustrate one modeling approach for the competition. There are many other innovative modeling techniques that could be applied to effectively recommend personalized rules on this dataset.

## Usage
```cli
python data_preprocess.py

python main.py

python output_result
```
