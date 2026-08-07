#!/usr/bin/env python3
import argparse
import dataclasses
import os
import tempfile
import time
from functools import partial

import numpy as np

from openpilot.selfdrive.modeld.compile_modeld import (
  WARP_DEV,
  get_policy_npy_shapes,
  make_random_images,
  read_file_chunked_to_disk,
  sample_desire,
  sample_skip,
  shift_and_sample,
)
from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit


POLICY_INPUTS = ['feat_q', 'desire_q', 'packed_npy_inputs']


def make_input_queues(input_shapes, frame_skip, device):
  fb = input_shapes['features_buffer']
  dp = input_shapes['desire_pulse']

  shapes, sizes = get_policy_npy_shapes(input_shapes)
  packed_npy_inputs = np.zeros(sum(sizes), dtype=np.float16)
  npy = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed_npy_inputs, np.cumsum(sizes[:-1])), strict=True)}
  input_queues = {
    'feat_q': Tensor(np.zeros((frame_skip * fb[1], fb[0], fb[2]), dtype=np.float16), device=device).contiguous().realize(),
    'desire_q': Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float16), device=device).contiguous().realize(),
    'packed_npy_inputs': Tensor(packed_npy_inputs, device='NPY').realize(),
  }
  return input_queues, npy


def make_run_policy(model_runner, model_metadata, frame_skip):
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  npy_shapes, npy_sizes = get_policy_npy_shapes(model_metadata['input_shapes'])

  def run_policy(img, big_img, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(Device.DEFAULT)
    img = img.to(Device.DEFAULT)
    big_img = big_img.to(Device.DEFAULT)
    Tensor.realize(packed_npy_inputs, img, big_img)

    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(npy_sizes), npy_shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip_fn)

    inputs = {
      'img': img,
      'big_img': big_img,
      'features_buffer': feat_buf,
      'desire_pulse': desire_buf,
      'traffic_convention': traffic_convention,
      'action_t': action_t,
    }
    out = next(iter(model_runner(inputs).values())).cast('float32')
    return out,

  return run_policy


FUSED_LN9_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void KERNEL_NAME(__global half* data0_4608,
                          __global half* data1_4608,
                          __global float* data2_9,
                          __global float* data3_9,
                          __global half* data4_512,
                          __global half* data5_512) {
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  __attribute__ ((aligned (16))) __local float temp0[16];
  const int gidx0 = get_group_id(0);
  const int lidx0 = get_local_id(0);
  const int base = (gidx0 << 9) + (lidx0 << 5);

  for (int Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    half val0 = (*(data1_4608 + base + Ridx0));
    acc0 = acc0 + ((float)(val0));
  }
  *(temp0 + lidx0) = acc0;
  barrier(CLK_LOCAL_MEM_FENCE);

  float mean = 0.0f;
  for (int Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    float val1 = (*(temp0 + Ridx102));
    mean = mean + val1;
  }
  mean = mean * 0.001953125f;

  for (int Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    half val2 = (*(data1_4608 + base + Ridx0));
    float alu1 = (((float)(val2)) - mean);
    acc1 = acc1 + (alu1 * alu1);
  }
  *(temp0 + lidx0) = acc1;
  barrier(CLK_LOCAL_MEM_FENCE);

  float inv_std = 0.0f;
  for (int Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    float val3 = (*(temp0 + Ridx102));
    inv_std = inv_std + val3;
  }
  inv_std = (1 / sqrt(((inv_std * 0.001953125f) + 9.999999747378752e-06f)));

  for (int Ridx1 = 0; Ridx1 < 8; Ridx1++) {
    const int out_off = base + (Ridx1 << 2);
    half4 val4 = (*((__global half4*)((data1_4608 + out_off))));
    half4 val5 = (*((__global half4*)((data4_512 + (lidx0 << 5) + (Ridx1 << 2)))));
    half4 val6 = (*((__global half4*)((data5_512 + (lidx0 << 5) + (Ridx1 << 2)))));
    *((__global half4*)((data0_4608 + out_off))) = (half4)(
      ((((half)(((((float)(val4.x)) - mean) * inv_std))) * val5.x) + val6.x),
      ((((half)(((((float)(val4.y)) - mean) * inv_std))) * val5.y) + val6.y),
      ((((half)(((((float)(val4.z)) - mean) * inv_std))) * val5.z) + val6.z),
      ((((half)(((((float)(val4.w)) - mean) * inv_std))) * val5.w) + val6.w));
  }
}
"""


FUSED_LN64_SOURCE_TEMPLATE = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void KERNEL_NAME(__global half* data0_4608,
                          __global half* data1_13824,
                          __global float* data2_72,
                          __global float* data3_72,
                          __global half* data4_64,
                          __global half* data5_64) {
  __attribute__ ((aligned (16))) __local float temp0[16];
  const int gidx0 = get_group_id(0);
  const int gidx1 = get_group_id(1);
  const int lidx0 = get_local_id(0);
  const int in_off = (gidx0 * 1536) + QKV_OFFSET + (gidx1 << 6) + (lidx0 << 2);

  half4 val0 = (*((__global half4*)((data1_13824 + in_off))));
  float acc0 = 0.0f;
  acc0 = acc0 + ((float)(val0.x));
  acc0 = acc0 + ((float)(val0.y));
  acc0 = acc0 + ((float)(val0.z));
  acc0 = acc0 + ((float)(val0.w));
  *(temp0 + lidx0) = acc0;
  barrier(CLK_LOCAL_MEM_FENCE);

  float mean = 0.0f;
  for (int Ridx103 = 0; Ridx103 < 16; Ridx103++) {
    float val1 = (*(temp0 + Ridx103));
    mean = mean + val1;
  }
  mean = mean * 0.015625f;
  barrier(CLK_LOCAL_MEM_FENCE);

  float acc1 = 0.0f;
  float alu0 = (((float)(val0.x)) - mean);
  acc1 = acc1 + (alu0 * alu0);
  float alu1 = (((float)(val0.y)) - mean);
  acc1 = acc1 + (alu1 * alu1);
  float alu2 = (((float)(val0.z)) - mean);
  acc1 = acc1 + (alu2 * alu2);
  float alu3 = (((float)(val0.w)) - mean);
  acc1 = acc1 + (alu3 * alu3);
  *(temp0 + lidx0) = acc1;
  barrier(CLK_LOCAL_MEM_FENCE);

  float inv_std = 0.0f;
  for (int Ridx104 = 0; Ridx104 < 16; Ridx104++) {
    float val2 = (*(temp0 + Ridx104));
    inv_std = inv_std + val2;
  }
  inv_std = (1 / sqrt(((inv_std * 0.015625f) + 9.999999747378752e-06f)));

  const int out_head = gidx1 & 3;
  const int out_group = gidx1 >> 2;
  const int out_off = (out_group * 2304) + (out_head * 576) + (gidx0 << 6) + (lidx0 << 2);
  const int weight_off = (lidx0 << 2);
  half4 val3 = (*((__global half4*)((data4_64 + weight_off))));
  half4 val4 = (*((__global half4*)((data5_64 + weight_off))));
  *((__global half4*)((data0_4608 + out_off))) = (half4)(
    ((((half)(((((float)(val0.x)) - mean) * inv_std))) * val3.x) + val4.x),
    ((((half)(((((float)(val0.y)) - mean) * inv_std))) * val3.y) + val4.y),
    ((((half)(((((float)(val0.z)) - mean) * inv_std))) * val3.z) + val4.z),
    ((((half)(((((float)(val0.w)) - mean) * inv_std))) * val3.w) + val4.w));
}
"""

FUSED_OUTPUT_HEAD_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void KERNEL_NAME(__global float* output,
                          __global half* lane_input, __global half* lane_weights, __global half* lane_bias,
                          __global half* lane_prob_weights, __global half* lane_prob_bias,
                          __global half* edge_weights, __global half* edge_bias,
                          __global half* meta_weights, __global half* meta_bias,
                          __global half* desire_pred_weights, __global half* desire_pred_bias,
                          __global half* pose_input, __global half* pose_weights, __global half* pose_bias, __global half* pose_scale,
                          __global half* wide_weights, __global half* wide_bias,
                          __global half* transform_weights, __global half* transform_bias,
                          __global half* plan_input, __global half* plan_weights, __global half* plan_bias, __global half* plan_scale,
                          __global half* lead_input, __global half* lead_weights, __global half* lead_bias, __global half* lead_scale,
                          __global half* lead_prob_input, __global half* lead_prob_weights, __global half* lead_prob_bias,
                          __global half* desire_input, __global half* desire_weights, __global half* desire_bias,
                          __global half* action_input, __global half* action_weights, __global half* action_bias, __global half* action_scale,
                          __global half* hidden_state, __global half* pad) {
  const int out = get_global_id(0);
  if (out >= 2066) {
    output[out] = (float)(out < 2578 ? hidden_state[out - 2066] : pad[out - 2578]);
    return;
  }

  __global half* input = lane_input;
  __global half* weights = lane_weights;
  __global half* bias = lane_bias;
  __global half* scale = lane_bias;
  int base = 0;
  bool scaled = false;
  if (out < 528) {
    base = 0;
  } else if (out < 536) {
    weights = lane_prob_weights; bias = lane_prob_bias; base = 528;
  } else if (out < 800) {
    weights = edge_weights; bias = edge_bias; base = 536;
  } else if (out < 855) {
    weights = meta_weights; bias = meta_bias; base = 800;
  } else if (out < 887) {
    weights = desire_pred_weights; bias = desire_pred_bias; base = 855;
  } else if (out < 899) {
    input = pose_input; weights = pose_weights; bias = pose_bias; scale = pose_scale; base = 887; scaled = true;
  } else if (out < 905) {
    weights = wide_weights; bias = wide_bias; base = 899;
  } else if (out < 917) {
    weights = transform_weights; bias = transform_bias; base = 905;
  } else if (out < 1907) {
    input = plan_input; weights = plan_weights; bias = plan_bias; scale = plan_scale; base = 917; scaled = true;
  } else if (out < 2051) {
    input = lead_input; weights = lead_weights; bias = lead_bias; scale = lead_scale; base = 1907; scaled = true;
  } else if (out < 2054) {
    input = lead_prob_input; weights = lead_prob_weights; bias = lead_prob_bias; base = 2051;
  } else if (out < 2062) {
    input = desire_input; weights = desire_weights; bias = desire_bias; base = 2054;
  } else {
    input = action_input; weights = action_weights; bias = action_bias; scale = action_scale; base = 2062; scaled = true;
  }

  const int head_out = out - base;
  const int weight_base = head_out * 512;
  float acc = 0.0f;
  for (int reduction = 0; reduction < 128; reduction++) {
    const int offset = reduction * 4;
    const half4 input_values = *((__global half4*)(input + offset));
    const half4 weight_values = *((__global half4*)(weights + weight_base + offset));
    const half4 products = input_values * weight_values;
    acc = acc + (float)products.x;
    acc = acc + (float)products.y;
    acc = acc + (float)products.z;
    acc = acc + (float)products.w;
  }
  half value = (half)acc + bias[head_out];
  if (scaled) value = value * scale[head_out];
  output[out] = (float)value;
}
"""


def _program_name(call):
  from tinygrad.uop.ops import Ops
  return call.src[0].arg.function_name if call.src[0].op is Ops.PROGRAM else None


def _program_source(call) -> str:
  from tinygrad.uop.ops import Ops
  return next(node.arg for node in call.src[0].src if node.op is Ops.SOURCE)


def _program_argument_types(call) -> tuple[str, ...]:
  source = _program_source(call)
  signature = f"__kernel void {_program_name(call)}("
  args_start = source.index(signature) + len(signature)
  args_end = source.index(") {", args_start)
  return tuple(declaration.strip().rsplit(" ", 1)[0] for declaration in source[args_start:args_end].split(","))


def _replace_program(call, suffix: str, source: str, global_size: tuple[int | float, ...], local_size: tuple[int, ...],
                     globals_: tuple[int, ...], outs: tuple[int, ...], ins: tuple[int, ...], signature_copies: int = 1):
  from tinygrad.uop.ops import Ops

  ast = call.src[0]
  new_arg = dataclasses.replace(ast.arg, name=f"{ast.arg.name}_{suffix}", global_size=global_size,
                                local_size=local_size, globals=globals_, outs=outs, ins=ins)
  source = source.replace("KERNEL_NAME", new_arg.function_name)
  binary = source.encode()
  new_nodes = tuple(node.replace(arg=source) if node.op is Ops.SOURCE else
                    node.replace(arg=binary) if node.op is Ops.BINARY else node for node in ast.src)
  if signature_copies > 1:
    linear = new_nodes[1]
    params = tuple(node for node in linear.src if node.op is Ops.PARAM)
    buffer_count = len(call.src) - 1
    copies = tuple(node.replace(arg=dataclasses.replace(node.arg, slot=node.arg.slot + copy * buffer_count))
                   for copy in range(1, signature_copies) for node in params)
    new_nodes = (new_nodes[0], linear.replace(src=(*linear.src, *copies)), *new_nodes[2:])
  return call.replace(src=(ast.replace(arg=new_arg, src=new_nodes), *call.src[1:]))


def _merge_program_calls(calls, suffix: str, source: str, global_size: tuple[int, ...], local_size: tuple[int, ...]):
  base_ast = calls[0].src[0]
  buffer_count = len(calls[0].src) - 1
  outs, ins, buffers = [], [], []
  for copy, call in enumerate(calls):
    ast = call.src[0]
    if len(call.src) - 1 != buffer_count or ast.arg.globals != tuple(range(buffer_count)):
      raise ValueError(f"unsupported arguments for merged kernel {_program_name(call)}")
    if ast.arg.target != base_ast.arg.target:
      raise ValueError(f"incompatible metadata for merged kernel {_program_name(call)}")
    offset = copy * buffer_count
    outs.extend(index + offset for index in ast.arg.outs)
    ins.extend(index + offset for index in ast.arg.ins)
    buffers.extend(call.src[1:])

  merged = _replace_program(calls[0], suffix, source, global_size, local_size,
                            tuple(range(buffer_count * len(calls))), tuple(outs), tuple(ins), len(calls))
  return merged.replace(src=(merged.src[0], *buffers))


def _attention_ln64_quad_source() -> str:
  source = FUSED_LN64_SOURCE_TEMPLATE.replace("QKV_OFFSET", "((branch & 1) * 512)")
  signature = "__kernel void KERNEL_NAME("
  args_start = source.index(signature) + len(signature)
  args_end = source.index(") {", args_start)
  declarations = [arg.strip() for arg in source[args_start:args_end].split(",")]
  parsed = [declaration.rsplit(" ", 1) for declaration in declarations]
  if any(len(arg) != 2 for arg in parsed):
    raise ValueError("could not parse fused attention layer-normalization arguments")

  suffixes = ("a", "b", "c", "d")
  merged_declarations = [f"{prefix} {arg}_{suffix}" for suffix in suffixes for prefix, arg in parsed]
  aliases = ["  const int branch = get_group_id(2);"]
  for prefix, arg in parsed:
    choices = f"branch == 0 ? {arg}_a : branch == 1 ? {arg}_b : branch == 2 ? {arg}_c : {arg}_d"
    aliases.append(f"  {prefix} {arg} = {choices};")
  return (source[:args_start] + ", ".join(merged_declarations) + ") {\n" +
          "\n".join(aliases) + source[args_end + 3:])


def fuse_layer_norm_pairs(jit) -> bool:
  calls = list(jit.captured._linear.src)
  names = [_program_name(call) for call in calls]
  expected_types = ("__global half*", "__global half*", "__global float*",
                    "__global float*", "__global half*", "__global half*")
  fused, count, skipped = [], 0, 0
  index = 0
  pattern = ["r_9_16_32", "r_9_16_32", "r_9_16_32n1", "r_9_16_32n1",
             "E_3_128_4_3n1", "E_3_128_4_3n1"]
  while index < len(calls):
    if names[index:index + len(pattern)] != pattern:
      fused.append(calls[index])
      index += 1
      continue

    normalized_calls = calls[index + 4:index + 6]
    if any(_program_argument_types(call) != expected_types for call in normalized_calls):
      fused.extend(calls[index:index + len(pattern)])
      skipped += 1
      index += len(pattern)
      continue

    for call in normalized_calls:
      ast = call.src[0]
      normalized = _replace_program(call, "fusedln", FUSED_LN9_SOURCE, (9, 1, 1), (16, 1, 1),
                                    ast.arg.globals, (0, 2, 3), (1, 4, 5))
      fused.append(normalized)
    count += 1
    index += len(pattern)

  if count == 0:
    return False
  jit.captured._linear = jit.captured._linear.replace(src=tuple(fused))
  jit.captured.__dict__.pop("linear", None)
  print(f"fused {count} paired layer-normalization blocks ({skipped} incompatible blocks skipped)")
  return True


def fuse_attention_layer_norms(jit) -> bool:
  pattern = [
    "r_8_9_16_4", "r_8_9_16_4n1", "r_8_9_16_4", "r_8_9_16_4n1",
    "r_8_9_16_4n2", "r_8_9_16_4n3", "r_8_9_16_4n2", "r_8_9_16_4n3",
    "E_2_9_16_4_4", "E_2_9_16_4_4n1", "E_2_9_16_4_4", "E_2_9_16_4_4n1",
  ]
  calls = list(jit.captured._linear.src)
  names = [_program_name(call) for call in calls]
  fused, replaced = [], 0
  index = 0
  while index < len(calls):
    if names[index:index + len(pattern)] != pattern:
      fused.append(calls[index])
      index += 1
      continue

    attention_calls = calls[index + 8:index + 12]
    fused.append(_merge_program_calls(attention_calls, "fusedln64quad", _attention_ln64_quad_source(),
                                      (9, 8, 4), (16, 1, 1)))
    replaced += 4
    index += len(pattern)

  if replaced == 0:
    return False
  jit.captured._linear = jit.captured._linear.replace(src=tuple(fused))
  jit.captured.__dict__.pop("linear", None)
  print(f"fused {replaced} attention layer-normalization kernels into {replaced // 4} dispatches")
  return True


def fuse_output_heads(jit) -> bool:
  calls = list(jit.captured._linear.src)
  replaced = 0
  for index, call in enumerate(calls):
    name = _program_name(call)
    if name is None or not name.startswith("r_860_3_512_512_512_512_512_512_512_512_512_512_512_512_128_4"):
      continue
    ast = call.src[0]
    if len(call.src) != 41 or ast.arg.globals != tuple(range(40)):
      raise ValueError(f"unsupported arguments for fused output head {name}")
    calls[index] = _replace_program(call, "fusedheads", FUSED_OUTPUT_HEAD_SOURCE, (20.15625, 1, 1), (128, 1, 1),
                                    ast.arg.globals, ast.arg.outs, ast.arg.ins)
    replaced += 1

  if replaced == 0:
    return False
  jit.captured._linear = jit.captured._linear.replace(src=tuple(calls))
  jit.captured.__dict__.pop("linear", None)
  print(f"fused {replaced} multi-head output kernel")
  return True


def _pair_source(call, local_z: bool) -> str:
  source = _program_source(call)
  name = _program_name(call)
  signature = f"__kernel void {name}("
  signature_start = source.index(signature)
  args_start = signature_start + len(signature)
  args_end = source.index(") {", args_start)
  declarations = [arg.strip() for arg in source[args_start:args_end].split(",")]
  parsed = [declaration.rsplit(" ", 1) for declaration in declarations]
  if any(len(arg) != 2 for arg in parsed):
    raise ValueError(f"could not parse kernel arguments for {name}")

  paired_declarations = [f"{prefix} {arg}_a" for prefix, arg in parsed]
  paired_declarations += [f"{prefix} {arg}_b" for prefix, arg in parsed]
  pair_index = "get_local_id(2)" if local_z else "get_group_id(2)"
  aliases: list[str] = [f"  const int pair_idx = {pair_index};"]
  aliases += [f"  {prefix} {arg} = pair_idx == 0 ? {arg}_a : {arg}_b;" for prefix, arg in parsed]
  return (source[:signature_start] + "__kernel void KERNEL_NAME(" + ", ".join(paired_declarations) + ") {\n" +
          "\n".join(aliases) + source[args_end + 3:])


def _pair_has_cross_dependency(call_a, call_b) -> bool:
  ast_a, ast_b = call_a.src[0], call_b.src[0]
  a_outs = {call_a.src[index + 1] for index in ast_a.arg.outs}
  b_outs = {call_b.src[index + 1] for index in ast_b.arg.outs}
  a_ins = {call_a.src[index + 1] for index in ast_a.arg.ins}
  b_ins = {call_b.src[index + 1] for index in ast_b.arg.ins}
  return bool(a_outs & b_outs or a_outs & b_ins or b_outs & a_ins)


def _can_pair_calls(call_a, call_b) -> bool:
  from tinygrad.uop.ops import Ops

  if call_a.src[0].op is not Ops.PROGRAM or call_b.src[0].op is not Ops.PROGRAM:
    return False
  ast_a, ast_b = call_a.src[0], call_b.src[0]
  if ast_a.arg.function_name != ast_b.arg.function_name:
    return False
  if ast_a.arg.global_size != ast_b.arg.global_size or ast_a.arg.local_size != ast_b.arg.local_size:
    return False
  if ast_a.arg.global_size[2] != 1 or ast_a.arg.local_size is None or ast_a.arg.local_size[2] != 1:
    return False
  if ast_a.arg.vars or ast_b.arg.vars:
    return False
  source = _program_source(call_a)
  if any(token in source for token in ("get_global_id(2)", "get_group_id(2)", "get_local_id(2)")):
    return False
  return not _pair_has_cross_dependency(call_a, call_b)


def _pair_calls(call_a, call_b):
  ast_a, ast_b = call_a.src[0], call_b.src[0]
  buffer_count = len(call_a.src) - 1
  if ast_a.arg.globals != tuple(range(buffer_count)) or len(call_b.src) - 1 != buffer_count:
    raise ValueError(f"unsupported arguments for paired kernel {_program_name(call_a)}")

  outs = (*ast_a.arg.outs, *(index + buffer_count for index in ast_b.arg.outs))
  ins = (*ast_a.arg.ins, *(index + buffer_count for index in ast_b.arg.ins))

  source = _program_source(call_a)
  local_z = "__local" not in source and "barrier(" not in source
  if local_z:
    global_size = ast_a.arg.global_size
    local_size = (ast_a.arg.local_size[0], ast_a.arg.local_size[1], 2)
  else:
    global_size = (ast_a.arg.global_size[0], ast_a.arg.global_size[1], 2)
    local_size = ast_a.arg.local_size
  paired = _replace_program(call_a, "pairlz" if local_z else "pair", _pair_source(call_a, local_z),
                            global_size, local_size, tuple(range(buffer_count * 2)), outs, ins, 2)
  return paired.replace(src=(paired.src[0], *call_a.src[1:], *call_b.src[1:]))


def fuse_adjacent_calls(jit) -> bool:
  calls = list(jit.captured._linear.src)
  fused, counts = [], {}
  index = 0
  while index < len(calls):
    if index + 1 < len(calls) and _can_pair_calls(calls[index], calls[index + 1]):
      name = _program_name(calls[index])
      fused.append(_pair_calls(calls[index], calls[index + 1]))
      counts[name] = counts.get(name, 0) + 1
      index += 2
    else:
      fused.append(calls[index])
      index += 1

  if not counts:
    return False
  jit.captured._linear = jit.captured._linear.replace(src=tuple(fused))
  jit.captured.__dict__.pop("linear", None)
  detail = ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
  print(f"fused {sum(counts.values())} adjacent model calls ({detail})")
  return True


def vectorize_qkv_weight_loads(jit) -> bool:
  scalar_loads = """    half val0 = (*(data2_786432+(alu10+1)));
    half val1 = (*(data2_786432+(alu10+2)));
    half val2 = (*(data2_786432+(alu10+1536)));
    half val3 = (*(data2_786432+(alu10+1537)));
    half val4 = (*(data2_786432+(alu10+1538)));
    half val5 = (*(data2_786432+(alu10+3072)));
    half val6 = (*(data2_786432+(alu10+3073)));
    half val7 = (*(data2_786432+(alu10+3074)));
    half val8 = (*(data2_786432+(alu10+4608)));
    half val9 = (*(data2_786432+(alu10+4609)));
    half val10 = (*(data2_786432+(alu10+4610)));
    half val11 = (*(data2_786432+alu10));"""
  vector_loads = """    half3 weights0 = vload3(0, data2_786432+alu10);
    half3 weights1 = vload3(0, data2_786432+alu10+1536);
    half3 weights2 = vload3(0, data2_786432+alu10+3072);
    half3 weights3 = vload3(0, data2_786432+alu10+4608);
    half val0 = weights0.s1;
    half val1 = weights0.s2;
    half val2 = weights1.s0;
    half val3 = weights1.s1;
    half val4 = weights1.s2;
    half val5 = weights2.s0;
    half val6 = weights2.s1;
    half val7 = weights2.s2;
    half val8 = weights3.s0;
    half val9 = weights3.s1;
    half val10 = weights3.s2;
    half val11 = weights0.s0;"""

  calls = list(jit.captured._linear.src)
  replaced = 0
  for index, call in enumerate(calls):
    if _program_name(call) != "r_3_512_3_3_128_4_pairlz":
      continue
    ast = call.src[0]
    source = _program_source(call)
    if source.count(scalar_loads) != 1:
      raise ValueError("unsupported qkv weight-load kernel source")
    source = source.replace(scalar_loads, vector_loads)
    source = source.replace(f"__kernel void {ast.arg.function_name}(", "__kernel void KERNEL_NAME(", 1)
    calls[index] = _replace_program(call, "vecweights", source, ast.arg.global_size, ast.arg.local_size,
                                    ast.arg.globals, ast.arg.outs, ast.arg.ins)
    replaced += 1

  if replaced == 0:
    return False
  if replaced != 4:
    raise ValueError(f"expected 4 qkv weight-load kernels, found {replaced}")
  jit.captured._linear = jit.captured._linear.replace(src=tuple(calls))
  jit.captured.__dict__.pop("linear", None)
  print(f"vectorized weights in {replaced} qkv kernels")
  return True


def add_reqd_work_group_attrs(jit) -> bool:
  from tinygrad.uop.ops import Ops
  allowed = {"r_16_64_32_4_4_24_3_3"}
  calls = list(jit.captured._linear.src)
  changed = {}
  for index, call in enumerate(calls):
    if call.src[0].op is not Ops.PROGRAM:
      continue
    ast = call.src[0]
    name, local_size = ast.arg.function_name, ast.arg.local_size
    if name not in allowed or local_size is None:
      continue
    source = _program_source(call)
    signature = f"__kernel void {name}("
    if signature not in source or "reqd_work_group_size" in source:
      continue
    attribute = f"__attribute__((reqd_work_group_size({local_size[0]}, {local_size[1]}, {local_size[2]})))\n__kernel void KERNEL_NAME("
    calls[index] = _replace_program(call, "reqdwg", source.replace(signature, attribute, 1),
                                    ast.arg.global_size, local_size, ast.arg.globals,
                                    ast.arg.outs, ast.arg.ins)
    changed[name] = changed.get(name, 0) + 1

  if not changed:
    return False
  jit.captured._linear = jit.captured._linear.replace(src=tuple(calls))
  jit.captured.__dict__.pop("linear", None)
  detail = ", ".join(f"{name}:{count}" for name, count in sorted(changed.items()))
  print(f"added reqd_work_group_size attrs ({detail})")
  return True


def compile_jit(jit, make_random_inputs, input_keys, make_queues):
  exactness_seeds = (42, 43, 44)
  def random_inputs_run(fn, seed, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy = make_queues(Device.DEFAULT)
    rng = np.random.default_rng(seed)
    Tensor.manual_seed(seed)

    testing = test_val is not None or test_buffers is not None
    n_runs = 1 if testing else 3

    for i in range(n_runs):
      for v in npy.values():
        v[:] = rng.standard_normal(v.shape).astype(v.dtype)
      Device.default.synchronize()
      random_inputs = make_random_inputs()
      st = time.perf_counter()
      outs = fn(**{k: input_queues[k] for k in input_keys}, **random_inputs)
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
      print(f"  [{i+1}/{n_runs}] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

      if i == 0:
        val = [np.copy(v.numpy()) for v in outs]
        buffers = [np.copy(v.numpy().copy()) for v in input_queues.values()]

    if test_val is not None:
      match = all(np.array_equal(a, b) for a, b in zip(val, test_val, strict=True))
      assert match == expect_match, f"outputs {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    if test_buffers is not None:
      match = all(np.array_equal(a, b) for a, b in zip(buffers, test_buffers, strict=True))
      assert match == expect_match, f"buffers {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    return val, buffers

  print('capture + replay')
  test_cases = {seed: random_inputs_run(jit, seed) for seed in exactness_seeds}
  for optimize in (fuse_layer_norm_pairs, fuse_attention_layer_norms, fuse_adjacent_calls,
                   vectorize_qkv_weight_loads, fuse_output_heads, add_reqd_work_group_attrs):
    if optimize(jit):
      print(f'post-{optimize.__name__} exactness')
      for seed, (test_val, test_buffers) in test_cases.items():
        random_inputs_run(jit, seed, test_val, test_buffers, expect_match=True)
  print('pickle round trip')
  with tempfile.TemporaryFile(dir=".") as f:
    dump_oob(jit, f)
    f.seek(0)
    jit = load_oob(f)
  for seed, (test_val, test_buffers) in test_cases.items():
    random_inputs_run(jit, seed, test_val, test_buffers, expect_match=True)
  test_val, test_buffers = test_cases[exactness_seeds[0]]
  random_inputs_run(jit, exactness_seeds[-1] + 1, test_val, test_buffers, expect_match=False)
  return jit


if __name__ == "__main__":
  from tinygrad.nn.onnx import OnnxRunner
  from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict

  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--frame-skip', type=int, required=True)
  args = parser.parse_args()

  model_path = read_file_chunked_to_disk(args.onnx)
  model_runner = OnnxRunner(model_path)
  out = {'metadata': make_metadata_dict(model_path), 'direct_images': True}

  run_policy_jit = TinyJit(make_run_policy(model_runner, out['metadata'], args.frame_skip), prune=True)
  make_policy_queues = partial(make_input_queues, out['metadata']['input_shapes'], args.frame_skip)
  make_random_model_inputs = partial(make_random_images, keys=['img', 'big_img'],
                                     shape=out['metadata']['input_shapes']['img'], device=WARP_DEV)
  out['run_policy'] = compile_jit(run_policy_jit, make_random_model_inputs, POLICY_INPUTS, make_policy_queues)

  with open(args.output, "wb") as f:
    dump_oob(out, f)
  print(f"Saved JITs to {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
