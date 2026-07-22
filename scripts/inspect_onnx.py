"""Print model I/O shapes and metadata for exporter acceptance checks."""
import argparse

import onnx


def shape(value):
    return [dimension.dim_value or dimension.dim_param
            for dimension in value.type.tensor_type.shape.dim]


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()
model = onnx.load(args.model)
print("inputs", [(value.name, shape(value)) for value in model.graph.input])
print("outputs", [(value.name, shape(value)) for value in model.graph.output])
print("metadata", {item.key: item.value for item in model.metadata_props})
