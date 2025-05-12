# DivideFold

This repository contains the DivideFold model for predicting the secondary structure of RNAs. \
The goal of this method is to recursively partitionate the input sequence into smaller fragments and use an existing structure prediction tool to predict the secondary structures of the fragments. Then, the predictions are combined together to form the global prediction for the input sequence. \
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

We also recommend that you install [LinearFold](https://github.com/LinearFold/LinearFold) since it is the prediction function that we use by default.

We provide wrappers for [KnotFold](https://github.com/gongtiansu/KnotFold), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2), [UFold](https://github.com/uci-cbcl/UFold), [ProbKnot](https://rna.urmc.rochester.edu/RNAstructure.html), [RNAfold](https://www.tbi.univie.ac.at/RNA/) and [RNAsubopt](https://www.tbi.univie.ac.at/RNA/). \
In order to use one of these prediction tools, it should be installed on your system, in the same parent folder as DivideFold. \
For example, [LinearFold](https://github.com/LinearFold/LinearFold) should be installed at `../LinearFold`.

The prediction tools can also be installed anywhere else on your system. \
In that case, the path must be specified with :

``` python
from dividefold.predict import linearfold_predict
sequence = "AUCG" * 100  # example sequence
linearfold_prediction = linearfold_predict(sequence, path_linearfold="path/to/your/linearfold/repository")
```

The installation paths should be specified when using [KnotFold](https://github.com/gongtiansu/KnotFold), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2), [UFold](https://github.com/uci-cbcl/UFold) or [ProbKnot](https://rna.urmc.rochester.edu/RNAstructure.html), if they are not already installed in the same parent folder as DivideFold. \
The paths for [RNAfold](https://www.tbi.univie.ac.at/RNA/) and [RNAsubopt](https://www.tbi.univie.ac.at/RNA/) do not matter and do not need to be specified.

## Prediction

### Usage

You can predict a sequence's secondary structure using the prediction function :
``` python
from dividefold.predict import dividefold_predict
sequence = "AUCG" * 100  # example sequence
prediction = dividefold_predict(sequence)
```

### Using a custom prediction function

By default, the prediction tool to be applied after partitioning is LinearFold. However, DivideFold can use any function you like for the structure prediction part. \
If you would like to use a custom structure prediction function, you can use :
``` python
from dividefold.predict import dividefold_predict

def my_structure_prediction_function(seq):  # example prediction function
    n = len(seq)
    return "(" * (n // 2) + "." * (n % 2) + ")" * (n // 2)

sequence = "AUCG" * 100  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=my_structure_prediction_function)
```

### Using other prediction tools

We also provide wrappers for [KnotFold](https://github.com/gongtiansu/KnotFold), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2), [UFold](https://github.com/uci-cbcl/UFold), [ProbKnot](https://rna.urmc.rochester.edu/RNAstructure.html), [RNAfold](https://www.tbi.univie.ac.at/RNA/) and [RNAsubopt](https://www.tbi.univie.ac.at/RNA/). \
If the corresponding tool is installed on your system, you can use it as the prediction function for DivideFold :
``` python
from dividefold.predict import dividefold_predict, knotfold_predict, linearfold_predict, rnafold_predict, ufold_predict, mxfold2_predict, rnasubopt_predict, probknot_predict

sequence = "AUCG" * 100  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=rnafold_predict)  # if you want to use RNAfold as the prediction function
```

### Obtaining cut points

If you're only interested in the cut points, you can use :
``` python
from dividefold.predict import dividefold_predict
sequence = "AUCG" * 100  # example sequence
fragments = dividefold_predict(sequence, return_cuts=True)
```
This will return the cut points at the final step in the recursive cutting process. \
No prediction function will be used, since only the cut points will be computed.

## Data availability
All data used in our training and experiments can be found in `data/data_structures/`, including our training set and our validation and test sets for both our sequence-wise and our family-wise split.
