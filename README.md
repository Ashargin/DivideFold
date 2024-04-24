# DivideFold

This repository contains the DivideFold model for predicting the secondary structure of RNAs.
The goal of this method is to recursively partitionate the input sequence into smaller fragments and use an existing structure prediction tool to predict the secondary structures of the fragments. Then, the predictions are combined together to form the global prediction for the input sequence.
DivideFold allows to partitionate the sequence in a way that the structure is conserved as much as possible.

## Installation

DivideFold requires python 3.9 or higher.

``` console
git clone https://github.com/Ashargin/DivideFold
cd DivideFold
python3 -m venv myenv
source myenv/bin/activate
pip install -e .
```

We also recommend that you install LinearFold since it is the prediction function that we use by default : https://github.com/LinearFold/LinearFold

The prediction tools should be installed in the same folder as DivideFold.
For example, LinearFold should be installed at ../LinearFold.

The prediction tools can also be installed anywhere on your system.
In that case, the path must be specified with :

``` python
from dividefold.predict import linearfold_predict
sequence = "AUCG" * 1000  # example sequence
linearfold_prediction = linearfold_predict(sequence)
```

## Prediction

### You can predict a sequence's secondary structure using the prediction function :
``` python
from dividefold.predict import dividefold_predict
sequence = "AUCG" * 1000  # example sequence
prediction = dividefold_predict(sequence, path_linearfold="path/to/your/linearfold/repository")
```

### By default, the prediction tool to be applied after partitioning is LinearFold. However, DivideFold can use any function you like for the structure prediction part. If you would like to use a custom structure prediction function, you can use :
``` python
from dividefold.predict import dividefold_predict

def my_structure_prediction_function(seq):  # example prediction function
    return "(" * (len(seq) // 2) + "." * (len(seq) % 2) + ")" * (len(seq) // 2)

sequence = "AUCG" * 1000  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=my_structure_prediction_function)
```

### We also provide wrappers for LinearFold, RNAfold, UFold, MXfold2, RNAsubopt and ProbKnot. If the corresponding tool is installed on your system, you can use it as the prediction function for DivideFold :
``` python
from dividefold.predict import dividefold_predict, linearfold_predict, rnafold_predict, ufold_predict, mxfold2_predict, rnasubopt_predict, probknot_predict

sequence = "AUCG" * 1000  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=rnafold_predict)  # if you want to use RNAfold as the prediction function
```

### If you're only interested in the cut points, you can use :
``` python
from dividefold.predict import dividefold_predict
sequence = "AUCG" * 1000  # example sequence
fragments = dividefold_predict(sequence, return_cuts=True)
```
This will return the cut points at the final step in the recursive cutting process.

## Data availability
All data used in our training and experiments can be found in `data/data_structures/`, including our training set and our validation and test sets for both our sequence-wise and our family-wise split.