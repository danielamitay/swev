#!/usr/bin/env python3
"""Bundle an explicit Swev contract with a converted Core ML model.

Requires coremltools. Does not infer recipes, convert weights, or establish parity.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, destination, contract_path):
    import coremltools as ct
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Output must be a new .mlpackage path")
    if destination.suffix != ".mlpackage":
        raise ValueError("Output must end in .mlpackage")
    records = json.loads(contract_path.read_text())
    required = {"swev.config", "swev.preprocessing", "swev.postprocessing"}
    if not required.issubset(records):
        raise ValueError("Contract requires config, preprocessing, and postprocessing")
    config = records["swev.config"]
    if config["execution"] != {"profile": "text-decision-v1", "inputAdapter": "text-recipe-v1"}:
        raise ValueError("Unsupported execution contract")
    model = ct.models.MLModel(str(source), skip_model_load=True)
    metadata = model.user_defined_metadata
    dtype_names = {65568: "float32", 131104: "int32"}

    def features(values):
        result = {}
        for value in values:
            if value.type.WhichOneof("Type") != "multiArrayType" or value.type.isOptional:
                raise ValueError("Expected required tensor inputs and outputs")
            array = value.type.multiArrayType
            if array.dataType not in dtype_names:
                raise ValueError("Expected Int32 or Float32 tensors")
            result[value.name] = {"shape": list(array.shape), "dtype": dtype_names[array.dataType]}
        return result

    spec = model.get_spec()
    signatures = {"inputs": features(spec.description.input), "outputs": features(spec.description.output)}
    pre = records["swev.preprocessing"]
    length, options = pre["sequenceLength"], pre["optionCapacity"]
    if not 8 <= length <= 2048 or not 2 <= options <= 32:
        raise ValueError("Unsupported export capacity")
    integer = lambda shape: {"shape": shape, "dtype": "int32"}
    expected = {"input_ids": integer([1, length]), "option_indices": integer([1, options])}
    if pre["tensors"] == "masked-options":
        expected.update(token_mask=integer([1, length]), option_mask=integer([1, options]), question_type=integer([1]))
    elif pre["tensors"] in ("causal-pointer", "causal-labels"):
        expected.update(position_ids=integer([1, length]), decision_indices=integer([1]),
                        attention_bias={"shape": [1, 1, length, length], "dtype": "float32"})
    else:
        raise ValueError("Unsupported tensor layout")
    if signatures != {"inputs": expected, "outputs": {"option_logits": {"shape": [1, options], "dtype": "float32"}}}:
        raise ValueError("Unexpected export signature")
    if "recipe" not in pre:
        raise ValueError("Missing declarative text recipe")
    for key, value in records.items():
        if not key.startswith("swev."):
            raise ValueError("Contract keys must use the swev namespace")
        metadata[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    assets = {}
    for key in metadata:
        if key.startswith("swev.tokenizer.") and key != "swev.tokenizer.asset-index":
            payload = metadata[key].encode()
            assets[key.removeprefix("swev.tokenizer.")] = {"key": key, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    if "tokenizer.json" not in assets:
        raise ValueError("Missing embedded tokenizer")
    metadata["swev.tokenizer.asset-index"] = json.dumps(assets)
    metadata["swev.signatures"] = json.dumps(signatures)
    model.short_description = "Swev text decision model: " + config["id"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    print(json.dumps({"model": config["id"], "path": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.contract)
