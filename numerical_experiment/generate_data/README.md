## Folder Description
This folder include python files to generate datasets for testing, person should expect to see the script to generate datasets with expected property they chose via tuning the hyperparameters

data reading script: read.py; could help you read the dataset in such format


### Hyperparameters

#### Sparsity
- Sparsity controls the proportion of zero entries in the dataset
- the data was stored in libsvm sparse data format


#### m
- number of observations (exact in number)

#### n
- maximum number of features (columns) the dataset have
- may vary for each observation depends on sparsity


#### num_cls
- the number of classes the dataset should have
- encoded in numbers i = 0 ... p, p is maximum number of class


