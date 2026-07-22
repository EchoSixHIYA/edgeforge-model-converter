"""Download the official TF Hub YAMNet and package its original SavedModel."""
import argparse
import shutil
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub


parser = argparse.ArgumentParser()
parser.add_argument("output")
args = parser.parse_args()
archive = Path(args.output).resolve()
saved_model = archive.parent / "yamnet_saved_model"
if saved_model.exists():
    shutil.rmtree(saved_model)

model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
def infer(waveform):
    scores, embeddings, log_mel_spectrogram = model(waveform)
    return {"scores": scores, "embeddings": embeddings,
            "log_mel_spectrogram": log_mel_spectrogram}


tf.saved_model.save(model, str(saved_model), signatures=infer.get_concrete_function())
shutil.make_archive(str(archive.with_suffix("")), "zip", saved_model)
print(archive)
