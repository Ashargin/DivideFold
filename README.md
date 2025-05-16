# DivideFold

This repository contains the DivideFold model for predicting the secondary structure of long RNAs. \
The goal of this method is to recursively partition the input sequence into smaller fragments and use an existing structure prediction tool on the fragments. Then, the predicted structures are reassembled to form the global structure prediction for the input sequence. \
DivideFold aims to partition the sequence in a way that the structure is conserved as much as possible.

## Requirements

Python >= 3.9
Keras >= 3.2.1
PyTorch >= 2.5.0 or Tensorflow >= 2.16.1 as the Keras backend

## Installation

``` console
git clone https://github.com/Ashargin/DivideFold
cd DivideFold
python3 -m venv myenv
source myenv/bin/activate
pip install -e .
```

We also recommend that you install [KnotFold](https://github.com/gongtiansu/KnotFold) since it is the structure prediction function that we use by default.

We provide wrappers for [KnotFold](https://github.com/gongtiansu/KnotFold), [IPknot](https://github.com/satoken/ipknot), [pKiss](https://bibiserv.cebitec.uni-bielefeld.de/pkiss), [ProbKnot](https://rna.urmc.rochester.edu/RNAstructureWeb/Servers/ProbKnot/ProbKnot.html), [RNAfold](http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2) and [UFold](https://github.com/uci-cbcl/UFold). \
In order to use one of these structure prediction tools, it should be installed on your system, in the same parent folder as DivideFold. \
For example, [KnotFold](https://github.com/gongtiansu/KnotFold) should be installed at `../KnotFold`.

The structure prediction tools can also be installed anywhere else on your system. \
In that case, the path must be specified to the wrapper as the `dirpath` argument when using [KnotFold](https://github.com/gongtiansu/KnotFold), [ProbKnot](https://rna.urmc.rochester.edu/RNAstructureWeb/Servers/ProbKnot/ProbKnot.html), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2) or [UFold](https://github.com/uci-cbcl/UFold), if the tool is not already installed in the same parent folder as DivideFold. \
The paths for [IPknot](https://github.com/satoken/ipknot), [pKiss](https://bibiserv.cebitec.uni-bielefeld.de/pkiss) and [RNAfold](https://www.tbi.univie.ac.at/RNA/) do not matter and do not need to be specified.

## Usage

### Secondary structure prediction

You can predict a sequence's secondary structure using the prediction function:
``` python
from dividefold.predict import dividefold_predict
sequence = "".join(np.random.choice(["A", "U", "C", "G"], size=3000))  # example sequence
prediction = dividefold_predict(sequence)
```
By default, the structure prediction tool to be applied after partition is [KnotFold](https://github.com/gongtiansu/KnotFold).

### Specifying the structure prediction tool to be applied on the fragments 

We also provide wrappers for [IPknot](https://github.com/satoken/ipknot), [pKiss](https://bibiserv.cebitec.uni-bielefeld.de/pkiss), [ProbKnot](https://rna.urmc.rochester.edu/RNAstructureWeb/Servers/ProbKnot/ProbKnot.html), [RNAfold](http://rna.tbi.univie.ac.at/cgi-bin/RNAWebSuite/RNAfold.cgi), [LinearFold](https://github.com/LinearFold/LinearFold), [MXfold2](https://github.com/mxfold/mxfold2) and [UFold](https://github.com/uci-cbcl/UFold). \
If the corresponding tool is installed on your system, you can use it as the structure prediction function for DivideFold:
``` python
from dividefold.predict import dividefold_predict, knotfold_predict, ipknot_predict, pkiss_predict, probknot_predict, rnafold_predict, linearfold_predict, mxfold2_predict, ufold_predict
sequence = "".join(np.random.choice(["A", "U", "C", "G"], size=3000))  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=rnafold_predict)  # if you want to use RNAfold as the structure prediction function
```

### Using a custom structure prediction function

It is also possible to use any custom structure prediction function on the fragments after partition:
``` python
from dividefold.predict import dividefold_predict

def my_structure_prediction_function(seq):  # example structure prediction function
    n = len(seq)
    return "(" * (n // 2) + "." * (n % 2) + ")" * (n // 2)

sequence = "".join(np.random.choice(["A", "U", "C", "G"], size=3000))  # example sequence
prediction = dividefold_predict(sequence, predict_fnc=my_structure_prediction_function)
```

### Obtaining fragments coordinates

To obtain the fragments resulting from DivideFold's partition, use `return_fragments=True`:
``` python
from dividefold.predict import dividefold_predict
sequence = "".join(np.random.choice(["A", "U", "C", "G"], size=3000))  # example sequence
fragments, prediction = dividefold_predict(sequence, return_fragments=True)
```

If you're only interested in the fragments, and not in predicting the secondary structure, you can use `return_structure=False`:
``` python
from dividefold.predict import dividefold_predict
sequence = "".join(np.random.choice(["A", "U", "C", "G"], size=3000))  # example sequence
fragments = dividefold_predict(sequence, return_fragments=True, return_structure=False)
```
In this case, only the partition will be computed, and no structure prediction tool needs to be installed as none will be used.

## Data availability
All data used in our training and experiments can be found in `data/data_structures/`.