#!/usr/bin/env python3
"""Bundle text/image graphs with shared weights, or quantize a Swev package to INT4."""

import argparse
import base64
import json
import tempfile
from pathlib import Path


def check_output(output):
    if output.suffix != ".mlpackage" or output.exists():
        raise ValueError("Output must be a new .mlpackage path")
    output.parent.mkdir(parents=True, exist_ok=True)


def load(path):
    import coremltools as ct

    model = ct.models.MLModel(str(path), skip_model_load=True)
    config = json.loads(model.user_defined_metadata["swev.config"])
    if config.get("contractVersion") != "1.0":
        raise ValueError("Unsupported contractVersion; expected 1.0")
    return model, config


def weight_references(message):
    """Find blob references including constants inside nested operation blocks."""
    from google.protobuf.descriptor import FieldDescriptor

    found = set()
    for field, value in message.ListFields():
        if field.name == "blobFileValue":
            found.add(value.fileName)
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.is_repeated:
                children = (
                    value.values()
                    if field.message_type.GetOptions().map_entry
                    else value
                )
                for child in children:
                    found.update(weight_references(child))
            else:
                found.update(weight_references(value))
    return found


def bundle(text_path, image_path, output):
    import coremltools as ct
    from coremltools.proto import Model_pb2

    check_output(output)
    text, tc = load(text_path)
    image, ic = load(image_path)
    if (
        tc["execution"]["profile"] != "text-decision-v1"
        or ic["execution"]["profile"] != "vision-decision-v1"
    ):
        raise ValueError("Expected a text profile and an image profile")
    for key in ("modelVersion", "revision"):
        if tc.get(key) != ic.get(key):
            raise ValueError("Source model versions must agree")
    for key in ("maxQuestionsPerRequest", "maxOptionsPerQuestion"):
        if tc["capabilities"]["limits"][key] != ic["capabilities"]["limits"][key]:
            raise ValueError("Source capacities must agree")
    if (
        tc["capabilities"]["limits"]["maxSequenceTokens"]
        > ic["capabilities"]["limits"]["maxSequenceTokens"]
    ):
        raise ValueError("Text sequence capacity must not exceed image capacity")
    for key in set(text.user_defined_metadata) | set(image.user_defined_metadata):
        if (key.startswith("swev.tokenizer.") or key == "swev.postprocessing") and (
            text.user_defined_metadata.get(key) != image.user_defined_metadata.get(key)
        ):
            raise ValueError(
                "Text and image graphs must share tokenizer and postprocessing: " + key
            )
    with tempfile.TemporaryDirectory(prefix="swev-bundle-", dir=output.parent) as temp:
        package = Path(temp) / output.name
        descriptor = ct.utils.MultiFunctionDescriptor()
        descriptor.add_function(
            str(text_path), src_function_name="main", target_function_name="text"
        )
        descriptor.add_function(
            str(image_path), src_function_name="main", target_function_name="image"
        )
        descriptor.default_function_name = "image"
        # Used only to deduplicate weights at export time. Runtime graphs remain single-function.
        ct.utils.save_multifunction(descriptor, str(package))
        combined = ct.utils.load_spec(
            str(package / "Data/com.apple.CoreML/model.mlmodel")
        )

        def extract(name):
            spec = Model_pb2.Model()
            spec.CopyFrom(combined)
            spec.mlProgram.ClearField("functions")
            spec.mlProgram.functions["main"].CopyFrom(
                combined.mlProgram.functions[name]
            )
            description = next(
                f for f in combined.description.functions if f.name == name
            )
            for field in (
                "functions",
                "defaultFunctionName",
                "input",
                "output",
                "metadata",
            ):
                spec.description.ClearField(field)
            spec.description.input.extend(description.input)
            spec.description.output.extend(description.output)
            return spec

        root, embedded = extract("image"), extract("text")
        metadata = root.description.metadata.userDefined
        metadata.update(image.user_defined_metadata)
        ic["id"] = tc["id"] = output.stem
        ic["execution"]["profile"] = "routed-vision-decision-v1"
        metadata["swev.config"] = json.dumps(ic)
        overrides = {
            key: text.user_defined_metadata[key]
            for key in ("swev.config", "swev.preprocessing", "swev.signatures")
        }
        overrides["swev.config"] = json.dumps(tc)
        weights = {}
        for reference in weight_references(embedded):
            parts = reference.split("/")
            if (
                len(parts) != 3
                or parts[:2] != ["@model_path", "weights"]
                or parts[2] in ("", ".", "..")
                or "\\" in parts[2]
            ):
                raise ValueError("Unsupported weight reference: " + reference)
            weights[reference] = "/".join(parts[1:])
        specification = embedded.SerializeToString()
        if not 1 <= len(weights) <= 16 or len(specification) > 8 * 1024 * 1024:
            raise ValueError("Embedded text graph exceeds runtime limits")
        record = json.dumps(
            {
                "specification": base64.b64encode(specification).decode(),
                "metadata": overrides,
                "weights": weights,
            }
        )
        if len(record.encode()) > 12 * 1024 * 1024:
            raise ValueError("Embedded text record exceeds runtime limits")
        metadata["swev.text-model"] = record
        ct.utils.save_spec(root, str(package / "Data/com.apple.CoreML/model.mlmodel"))
        package.rename(output)


def quantize(source, output):
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )
    from coremltools.proto import Model_pb2

    check_output(output)
    model, config = load(source)
    optimization = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=32,
        )
    )
    with tempfile.TemporaryDirectory(prefix="swev-int4-", dir=output.parent) as temp:
        temp = Path(temp)

        def compress(model, name):
            result = linear_quantize_weights(model, config=optimization)
            provenance = json.loads(
                result.user_defined_metadata.get("swev.provenance", "{}")
            )
            provenance["quantization"] = {
                "dtype": "int4",
                "mode": "linear_symmetric",
                "blockSize": 32,
                "weightThreshold": 2048,
                "compute": "unchanged",
            }
            result.user_defined_metadata["swev.provenance"] = json.dumps(provenance)
            path = temp / (name + ".mlpackage")
            result.save(str(path))
            return path

        if config["execution"]["profile"] == "routed-vision-decision-v1":
            record = json.loads(model.user_defined_metadata["swev.text-model"])
            spec = Model_pb2.Model()
            spec.ParseFromString(
                base64.b64decode(record["specification"], validate=True)
            )
            metadata = dict(model.user_defined_metadata)
            del metadata["swev.text-model"]
            metadata.update(record["metadata"])
            spec.description.metadata.userDefined.update(metadata)
            text = ct.models.MLModel(
                spec, weights_dir=model.weights_dir, skip_model_load=True
            )
            config["execution"]["profile"] = "vision-decision-v1"
            model.user_defined_metadata["swev.config"] = json.dumps(config)
            del model.user_defined_metadata["swev.text-model"]
            text_path = compress(text, "text")
            image_path = compress(model, "image")
            bundle(text_path, image_path, output)
        else:
            config["id"] = output.stem
            model.user_defined_metadata["swev.config"] = json.dumps(config)
            compress(model, "model").rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("bundle", help="Combine prepared text and image packages")
    pair.add_argument("--text", required=True, type=Path)
    pair.add_argument("--image", required=True, type=Path)
    pair.add_argument("--output", required=True, type=Path)
    quant = sub.add_parser(
        "quantize", help="INT4 weights with unchanged floating-point computation"
    )
    quant.add_argument("source", type=Path)
    quant.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "bundle":
        bundle(args.text, args.image, args.output)
    else:
        quantize(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
