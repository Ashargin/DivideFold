import numpy as np
import pandas as pd
import keras
from pathlib import Path

from dividefold.utils import (
    format_data,
    optimize_pseudoknots,
    apply_mutation,
    evoaug_augment,
)
from dividefold.models.cnn_1d import CNN1D
from dividefold.predict import oracle_get_cuts

# Settings
MAX_MOTIFS = 200
MAX_DIL = 256
MIN_LEN = 400
DATA_AUGMENT_TYPE = "EVOAUG"
EPOCHS = 10  # change this if you want to train for fewer or more epochs

# Load model
model = CNN1D(max_motifs=MAX_MOTIFS, max_dil=MAX_DIL)
# Optional : if you want to load our pre-trained weights before fine-tuning :
# pretrained_model = keras.models.load_model(Path(__file__).parents[1] / "data/models/divide_model.keras")
# model.set_weights(pretrained_model.weights)
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    run_eagerly=True,
)


# Dataloader
def motif_data_generator(
    path_in,
    max_motifs=MAX_MOTIFS,
    min_len=MIN_LEN,
    max_len=None,
    data_augment_type=DATA_AUGMENT_TYPE,
    loss_lbda=0.5,
):
    # Load data
    df_in = pd.read_csv(path_in, index_col=0)
    df_in = df_in.sample(frac=1.0)
    if min_len is not None:
        df_in = df_in[df_in.seq.apply(len) > min_len]
    if max_len is not None:
        df_in = df_in[df_in.seq.apply(len) <= max_len]
    assert df_in.isna().sum().sum() == 0

    i = 0
    while True:
        # Get next observation
        row = df_in.iloc[i % df_in.shape[0]]
        seq, struct = row.seq, row.struct
        struct = optimize_pseudoknots(struct)

        # Apply data augmentation
        if data_augment_type == "MUTATION_SEQ":
            seq, struct = apply_mutation(
                seq,
                struct,
                mutation_proba=0.1 * np.random.random(),
                struct_deletion_proba=0.0,
            )
        elif data_augment_type == "MUTATION_SEQ_STRUCT":
            seq, struct = apply_mutation(
                seq,
                struct,
                mutation_proba=0.1 * np.random.random(),
                struct_deletion_proba=0.1 * np.random.random(),
            )
        elif data_augment_type == "EVOAUG":
            seq, struct = evoaug_augment(seq, struct)
        elif data_augment_type is not None:
            raise ValueError(
                'data_augment_type should be None or one of ["MUTATION_SEQ", "MUTATION_SEQ_STRUCT", "EVOAUG"].'
            )

        # Compute cut points from structure
        cuts, _ = oracle_get_cuts(struct)
        cuts = str(cuts).replace(",", "")

        # Format data
        seq_mat = format_data(seq, max_motifs=max_motifs)
        cuts_mat = np.array([float(c) for c in cuts[1:-1].split(" ")])

        # Inverse exponential distance to cut points loss
        loss_array = np.abs(
            cuts_mat.reshape((1, -1)) - np.arange(len(seq)).reshape((-1, 1))
        ).min(axis=1)
        loss_array = np.exp(-loss_lbda * loss_array)

        i += 1

        assert seq_mat.shape == (len(seq), max_motifs + 4)
        assert loss_array.shape == (len(seq),)

        yield seq_mat.reshape((1, len(seq), max_motifs + 4)), loss_array.reshape(
            (1, len(seq))
        )


# Fit model
# You can specify your data here : .csv file with "seq" and "struct" columns in header
# "seq" is the rna sequence, "struct" is the secondary structure in dot-bracket format
train_path = Path(__file__).parents[1] / "data/data_splits/train.csv"
val_path = Path(__file__).parents[1] / "data/data_splits/validation_sequencewise.csv"
train_gen = motif_data_generator(train_path)
val_gen = motif_data_generator(val_path, data_augment_type=None)
losses = []
for i in range(EPOCHS):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=4000,
        validation_steps=400,
        epochs=1,
    )
    this_loss = round(100000 * np.mean(history.history["val_loss"]))
    model.save(Path(__file__).parents[1] / "data/models/trained_epoch{i+1}.keras")
    losses.append(this_loss)
    print(f"Losses (epoch 1 to {i+1}):")
    print(losses)
    if min(losses) > 1000:
        print(
            "WARNING: if after 2 or 3 epochs, the loss has not gone below 1000, the model might be stale and failed in finding an optimization path. If this happens repeatedly, decrease MAX_DIL for an easier optimization path. Performances might decrease slightly."
        )
