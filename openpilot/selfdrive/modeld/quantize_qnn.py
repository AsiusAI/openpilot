#!/usr/bin/env python3
"""Build the Dragon Q6A QNN model from driving_supercombo.onnx and modeld input captures."""

import argparse
import copy
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
from onnxruntime.quantization.execution_providers.qnn.preprocess import qnn_preprocess_model


FINAL_OUTPUT_HEADS = [
  "node_linear_30", "node_linear_31", "node_linear_32", "node_linear_33", "node_linear_34",
  "node_linear_62", "node_linear_63", "node_linear_64", "node_linear_65", "node_linear_66",
  "p_node_linear_20", "p_node_linear_21",
]
TEMPORAL_HISTORY_NODES = {"p_node_Transpose_0", "p_node_GatherND_7", "p_node_index"}
FIXED_SHAPE_RESHAPES = {"node_view", "p_node_view", "p_node_view_1", "p_node_view_2"}
SCALE_BUCKETED_CONVS = {"node_conv2d_57"}
SCALE_BUCKET_LOG2_WIDTH = 1
FUSE_BUCKET_SCALES = False


class ModeldCalibrationReader(CalibrationDataReader):
  def __init__(self, calibration_dir: Path):
    self.samples = iter(sorted(calibration_dir.glob("*.npz")))

  def get_next(self) -> dict[str, np.ndarray] | None:
    try:
      path = next(self.samples)
    except StopIteration:
      return None

    with np.load(path) as sample:
      inputs = {name: sample[name].copy() for name in sample.files}
    inputs["img"] = np.transpose(inputs["img"], (0, 2, 3, 1))
    inputs["big_img"] = np.transpose(inputs["big_img"], (0, 2, 3, 1))
    return inputs


def convert_fp16_to_fp32(source: Path, output: Path) -> None:
  model = onnx.load(source, load_external_data=False)

  # ORT's QNN preprocessor currently tries to merge metadata_props as a dict.
  # The original model metadata is not needed by the runtime model and trips
  # that code path, so discard it before preprocessing.
  del model.metadata_props[:]
  opsets: dict[str, int] = {}
  for opset in model.opset_import:
    opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
  del model.opset_import[:]
  model.opset_import.extend(helper.make_opsetid(domain, version) for domain, version in opsets.items())

  for index, initializer in enumerate(model.graph.initializer):
    if initializer.data_type == onnx.TensorProto.FLOAT16:
      model.graph.initializer[index].CopyFrom(
        numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float32), initializer.name)
      )

  for value_info in (*model.graph.input, *model.graph.output, *model.graph.value_info):
    if value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
      value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

  for node in model.graph.node:
    if node.op_type == "Cast":
      for attribute in node.attribute:
        if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16:
          attribute.i = onnx.TensorProto.FLOAT
    elif node.op_type == "Constant":
      for attribute in node.attribute:
        if attribute.name == "value" and attribute.t.data_type == onnx.TensorProto.FLOAT16:
          value = numpy_helper.to_array(attribute.t).astype(np.float32)
          attribute.t.CopyFrom(numpy_helper.from_array(value, attribute.t.name))

  onnx.checker.check_model(model, full_check=True)
  onnx.save(model, output)


def quantize_per_channel_reference(source: Path, output: Path, calibration_dir: Path) -> None:
  config = get_qnn_qdq_config(
    source,
    ModeldCalibrationReader(calibration_dir),
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QUInt16,
    weight_type=QuantType.QUInt8,
    per_channel=True,
    nodes_to_exclude=["p_node_masked_fill", "p_node_softmax", *FINAL_OUTPUT_HEADS],
  )
  quantize(source, output, config)


def factor_conv_scales(source: Path, per_channel_model: Path, output: Path) -> dict[str, list[dict]]:
  model = onnx.load(source, load_external_data=False)
  pc_model = onnx.load(per_channel_model, load_external_data=False)
  initializers = {initializer.name: initializer for initializer in model.graph.initializer}
  pc_initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in pc_model.graph.initializer}
  pc_producers = {tensor: node for node in pc_model.graph.node for tensor in node.output}
  pc_nodes = {node.name: node for node in pc_model.graph.node}
  new_nodes = []
  new_initializers = []
  overrides: dict[str, list[dict]] = {}

  for node in model.graph.node:
    if node.op_type != "Conv":
      new_nodes.append(node)
      continue

    pc_weight_dq = pc_producers[pc_nodes[node.name].input[1]]
    quantized = pc_initializers[pc_weight_dq.input[0]]
    channel_scale = pc_initializers[pc_weight_dq.input[1]].astype(np.float32)
    channel_zero_point = pc_initializers[pc_weight_dq.input[2]]
    if not np.all(channel_zero_point == 128):
      raise ValueError(f"{node.name} has non-symmetric per-channel weights")

    if node.name in SCALE_BUCKETED_CONVS:
      weight = numpy_helper.to_array(initializers[node.input[1]])
      group = next((attribute.i for attribute in node.attribute if attribute.name == "group"), 1)
      outputs_per_input = weight.shape[0] // group
      is_depthwise = group > 1 and weight.shape[1] == 1 and outputs_per_input * group == weight.shape[0]
      if group != 1 and not is_depthwise:
        raise ValueError(f"{node.name} has unsupported convolution groups")
      if np.any(channel_scale <= 0):
        raise ValueError(f"{node.name} has a non-positive channel scale")

      bias = None
      if len(node.input) >= 3:
        bias = numpy_helper.to_array(initializers[node.input[2]]).astype(np.float32)

      # QNN HTP only supports per-tensor convolution weights. A single tensor
      # scale loses almost all precision when the output-channel scales span
      # several orders of magnitude. Split output channels into power-of-two
      # scale buckets, then restore the original channel order. For depthwise
      # convolution, gather matching input channels (including duplicates for
      # a depth multiplier) so every bucket remains depthwise.
      bucket_outputs: list[str] = []
      channel_order: list[int] = []
      bucket_ids = np.floor(np.log2(channel_scale) / SCALE_BUCKET_LOG2_WIDTH).astype(np.int32)
      for bucket_id in sorted(np.unique(bucket_ids)):
        output_channels = np.flatnonzero(bucket_ids == bucket_id)
        suffix = f"m{-bucket_id}" if bucket_id < 0 else str(bucket_id)
        prefix = f"{node.name}_scale_bucket_{suffix}"

        bucket_input = node.input[0]
        if is_depthwise:
          input_channels = output_channels // outputs_per_input
          gather_indices_name = f"{prefix}_input_indices"
          bucket_input = f"{prefix}_input"
          new_initializers.append(numpy_helper.from_array(input_channels.astype(np.int32), gather_indices_name))
          new_nodes.append(helper.make_node(
            "Gather", [node.input[0], gather_indices_name], [bucket_input], name=f"{prefix}_input_gather", axis=1,
          ))

        bucket_scale = channel_scale[output_channels]
        base_scale = float(np.max(bucket_scale))
        weight_name = f"{prefix}_weight"
        if FUSE_BUCKET_SCALES:
          bucket_weight = weight[output_channels].astype(np.float32)
        else:
          bucket_weight = ((quantized[output_channels].astype(np.int16) - 128) * base_scale).astype(np.float32)
        new_initializers.append(numpy_helper.from_array(bucket_weight, weight_name))
        overrides[weight_name] = [{"quant_type": QuantType.QUInt8, "scale": base_scale, "zero_point": 128}]

        bucket_conv = copy.deepcopy(node)
        bucket_conv.name = prefix
        del bucket_conv.input[:]
        bucket_conv.input.extend([bucket_input, weight_name])
        unit_channel_scale = FUSE_BUCKET_SCALES or np.allclose(bucket_scale, base_scale, rtol=0.0, atol=0.0)
        if unit_channel_scale and bias is not None:
          bias_name = f"{prefix}_bias"
          new_initializers.append(numpy_helper.from_array(bias[output_channels], bias_name))
          bucket_conv.input.append(bias_name)
        before_scale = f"{prefix}_output" if unit_channel_scale else f"{prefix}_before_channel_scale"
        del bucket_conv.output[:]
        bucket_conv.output.append(before_scale)
        if is_depthwise:
          next(attribute for attribute in bucket_conv.attribute if attribute.name == "group").i = len(output_channels)
        new_nodes.append(bucket_conv)

        if unit_channel_scale:
          bucket_output = before_scale
        else:
          scale_name = f"{prefix}_channel_scale_value"
          new_initializers.append(numpy_helper.from_array((bucket_scale / base_scale).reshape(1, -1, 1, 1), scale_name))
          scaled_output = f"{prefix}_scaled" if bias is not None else f"{prefix}_output"
          new_nodes.append(helper.make_node(
            "Mul", [before_scale, scale_name], [scaled_output], name=f"{prefix}_channel_scale",
          ))
          if bias is not None:
            bias_name = f"{prefix}_bias"
            new_initializers.append(numpy_helper.from_array(bias[output_channels].reshape(1, -1, 1, 1), bias_name))
            bucket_output = f"{prefix}_output"
            new_nodes.append(helper.make_node(
              "Add", [scaled_output, bias_name], [bucket_output], name=f"{prefix}_post_scale_bias",
            ))
          else:
            bucket_output = scaled_output

        bucket_outputs.append(bucket_output)
        channel_order.extend(output_channels.tolist())

      concatenated = f"{node.output[0]}_scale_bucket_order"
      new_nodes.append(helper.make_node(
        "Concat", bucket_outputs, [concatenated], name=f"{node.name}_scale_bucket_concat", axis=1,
      ))
      restore_indices = np.argsort(np.asarray(channel_order)).astype(np.int32)
      restore_indices_name = f"{node.name}_restore_channel_indices"
      new_initializers.append(numpy_helper.from_array(restore_indices, restore_indices_name))
      new_nodes.append(helper.make_node(
        "Gather", [concatenated, restore_indices_name], node.output,
        name=f"{node.name}_restore_channel_order", axis=1,
      ))
      continue

    base_scale = float(np.max(channel_scale))
    weight = initializers[node.input[1]]
    weight.CopyFrom(numpy_helper.from_array(
      ((quantized.astype(np.int16) - 128) * base_scale).astype(np.float32), weight.name,
    ))
    overrides[weight.name] = [{"quant_type": QuantType.QUInt8, "scale": base_scale, "zero_point": 128}]

    bias = None
    if len(node.input) >= 3:
      bias = numpy_helper.to_array(initializers[node.input[2]]).astype(np.float32)
      del node.input[2:]

    original_output = node.output[0]
    node.output[0] = f"{original_output}_before_channel_scale"
    new_nodes.append(node)

    scale_name = f"{node.name}_channel_scale_value"
    new_initializers.append(numpy_helper.from_array((channel_scale / base_scale).reshape(1, -1, 1, 1), scale_name))
    scaled_output = original_output if bias is None else f"{original_output}_before_bias"
    new_nodes.append(helper.make_node(
      "Mul", [node.output[0], scale_name], [scaled_output], name=f"{node.name}_channel_scale",
    ))
    if bias is not None:
      bias_name = f"{node.name}_post_scale_bias"
      new_initializers.append(numpy_helper.from_array(bias.reshape(1, -1, 1, 1), bias_name))
      new_nodes.append(helper.make_node(
        "Add", [scaled_output, bias_name], [original_output], name=f"{node.name}_post_scale_bias",
      ))

  del model.graph.node[:]
  model.graph.node.extend(new_nodes)
  model.graph.initializer.extend(new_initializers)
  onnx.checker.check_model(model, full_check=True)
  onnx.save(model, output)
  return overrides


def rewrite_temporal_policy_for_qnn(source: Path, output: Path) -> None:
  """Replace model ops that the QNN HTP backend cannot compile."""
  model = onnx.load(source, load_external_data=False)
  new_nodes = []

  for node in model.graph.node:
    if node.name in FIXED_SHAPE_RESHAPES:
      # The exporter sets allowzero=1, which QNN treats as a dynamic reshape
      # and leaves on the CPU. None of these fixed shape tensors contains a
      # zero, so allowzero=0 is exactly equivalent and lets HTP own the nodes.
      allowzero = next(attribute for attribute in node.attribute if attribute.name == "allowzero")
      if allowzero.i != 1:
        raise ValueError(f"{node.name} has unexpected allowzero={allowzero.i}")
      allowzero.i = 0
      new_nodes.append(node)
    elif node.name == "p_node_Transpose_0":
      # The original Transpose -> GatherND -> Transpose selects the newest nine
      # entries from the fixed 25-entry feature history. QNN does not implement
      # GatherND, so express the same operation as a direct slice on axis 1.
      new_nodes.append(helper.make_node(
        "Slice",
        ["fb_full", "qnn_history_start", "qnn_history_end", "qnn_history_axis", "qnn_history_step"],
        ["p_index"],
        name="p_node_history_slice",
      ))
    elif node.name in TEMPORAL_HISTORY_NODES:
      continue
    elif node.name == "p_node_masked_fill":
      # Quantizing -inf produces an invalid scale. A finite additive causal mask
      # is equivalent for softmax and compiles as an HTP ElementWiseAdd.
      node.op_type = "Add"
      del node.input[:]
      node.input.extend(["p_mul", "qnn_causal_attention_mask"])
      new_nodes.append(node)
    elif node.op_type == "ReduceL2":
      # QNN has no ReduceL2 builder, but supports its primitive definition.
      squared = f"{node.output[0]}_squared"
      summed = f"{node.output[0]}_sum"
      new_nodes.extend([
        helper.make_node("Mul", [node.input[0], node.input[0]], [squared], name=f"{node.name}_square"),
        helper.make_node(
          "ReduceSum", [squared, node.input[1]], [summed], name=f"{node.name}_sum",
          keepdims=1, noop_with_empty_axes=0,
        ),
        helper.make_node("Sqrt", [summed], node.output, name=f"{node.name}_sqrt"),
      ])
    else:
      new_nodes.append(node)

  del model.graph.node[:]
  model.graph.node.extend(new_nodes)
  model.graph.initializer.extend([
    numpy_helper.from_array(np.array([16], dtype=np.int64), "qnn_history_start"),
    numpy_helper.from_array(np.array([25], dtype=np.int64), "qnn_history_end"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "qnn_history_axis"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "qnn_history_step"),
    numpy_helper.from_array(
      np.triu(np.full((1, 1, 9, 9), -100.0, dtype=np.float32), k=1),
      "qnn_causal_attention_mask",
    ),
  ])
  onnx.checker.check_model(model, full_check=True)
  onnx.save(model, output)


def quantize_for_qnn(source: Path, output: Path, calibration_dir: Path, overrides: dict[str, list[dict]],
                     calibration_method: CalibrationMethod) -> None:
  config = get_qnn_qdq_config(
    source,
    ModeldCalibrationReader(calibration_dir),
    calibrate_method=calibration_method,
    activation_type=QuantType.QUInt16,
    weight_type=QuantType.QUInt8,
    per_channel=True,
    init_overrides=overrides,
  )
  quantize(source, output, config)
  model = onnx.load(output, load_external_data=False)
  source_model = onnx.load(source, load_external_data=False)
  versions = {opset.domain: opset.version for opset in model.opset_import}
  domains = [opset.domain for opset in source_model.opset_import if opset.domain != "ai.onnx.preview"]
  preferred_domains = ["", "ai.onnx.ml", "ai.onnx.training", "ai.onnx.preview.training"]
  domains = [domain for domain in preferred_domains if domain in domains] + sorted(set(domains) - set(preferred_domains))
  del model.opset_import[:]
  model.opset_import.extend(helper.make_opsetid(domain, versions[domain]) for domain in domains)
  onnx.checker.check_model(model, full_check=True)
  onnx.save(model, output)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True, help="driving_supercombo.onnx")
  parser.add_argument("--calibration-dir", type=Path, required=True, help="directory of modeld input .npz captures")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--calibration-method", choices=("minmax", "percentile"), default="minmax")
  args = parser.parse_args()
  samples = sorted(args.calibration_dir.glob("*.npz"))
  if not samples:
    parser.error(f"no .npz calibration captures in {args.calibration_dir}")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="qnn-model-") as work_dir:
    work = Path(work_dir)
    fp32 = work / "driving_supercombo_fp32.onnx"
    preprocessed = work / "driving_supercombo_fp32_nhwc.onnx"
    per_channel = work / "driving_supercombo_pc16.onnx"
    factored = work / "driving_supercombo_factored.onnx"
    qnn_compatible = work / "driving_supercombo_qnn_compatible.onnx"

    convert_fp16_to_fp32(args.source, fp32)
    qnn_preprocess_model(
      fp32,
      preprocessed,
      fuse_layernorm=True,
      inputs_to_make_channel_last=["img", "big_img"],
    )
    preprocessed_model = onnx.load(preprocessed, load_external_data=False)
    for node in preprocessed_model.graph.node:
      if node.op_type == "Transpose" and node.input[0] in ("img", "big_img"):
        node.name = f"Transpose_channel_{0 if node.input[0] == 'img' else 1}"
    onnx.save(preprocessed_model, preprocessed)
    quantize_per_channel_reference(preprocessed, per_channel, args.calibration_dir)
    overrides = factor_conv_scales(preprocessed, per_channel, factored)
    rewrite_temporal_policy_for_qnn(factored, qnn_compatible)
    calibration_method = {
      "minmax": CalibrationMethod.MinMax,
      "percentile": CalibrationMethod.Percentile,
    }[args.calibration_method]
    quantize_for_qnn(qnn_compatible, args.output, args.calibration_dir, overrides, calibration_method)

  digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
  print(f"wrote {args.output} ({len(samples)} calibration samples, sha256 {digest})")


if __name__ == "__main__":
  main()
