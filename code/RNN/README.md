# Files 

1. `preprocessing.py`: To preprocess the data and generate the pickle file
2. `gru_unrelated.py`: Code for training the unrelated vs related
3. `gru_stances.py`: Code for training the related stances
4. `restore_gru_stances.py`: To restore the weight from the checkpoint and run the test
5. `restore_gru_unrelated.py`: To restore the weight from the checkpoint and run the test

# Requirements

Tensorflow installed, either using virtualenv or directly using pip
nplk, glove, numpy, pandas, sklearn

# How to run

`python3 gru_stances.py` or `python3 gru_unrelated.py` 

It will then save the checkpoint file which we can use to run the test.
The pickle file must be generated first using the preprocessing.py.

