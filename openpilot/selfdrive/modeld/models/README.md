## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

`driving_supercombo_qnn.onnx` is the QDQ model used by modeld on Asius/Dragon Q6A. QNN places all model arithmetic,
including the vision network and recurrent temporal policy, on the HTP. ONNX Runtime retains four fixed-shape `Reshape`
nodes as CPU bookkeeping. The generator rewrites unsupported fixed-shape operations and factors per-channel convolution
scales into HTP-supported per-tensor convolutions. One depthwise layer with a four-order-of-magnitude scale range is split
into power-of-two scale buckets to retain accuracy. The camera warp remains on the Qualcomm MSM GPU so the CPU only has to
marshal the warped tensor into QNN. Set `MODELD_DEV=QNN` (the Asius default) when building this configuration. Set
`MODELD_DEV=QCOM` to build the full tinygrad MSM model instead; this is the only modeld build-time backend switch.

The slower NumPy warp remains available for diagnostics with `MODEL_NUMPY_WARP=1`. Regenerate the QNN model from modeld
input captures with:

```sh
uv run --with onnx==1.22.0 --with onnxruntime==1.28.0 python openpilot/selfdrive/modeld/quantize_qnn.py \
  --source openpilot/selfdrive/modeld/models/driving_supercombo.onnx \
  --calibration-dir /path/to/modeld-inputs \
  --output openpilot/selfdrive/modeld/models/driving_supercombo_qnn.onnx
```

Set `MODEL_INPUTS_DUMP_DIR` while running the regular tinygrad model to capture calibration inputs.
For host replay, set `MODEL_BACKEND=onnx` and `MODEL_ONNX_PATH`; the `onnx` backend uses ONNX Runtime's CPU provider.
The QNN replay keeps action, pose, and the other driving outputs under comparison; it excludes only
`confidence`, `laneLineStds`, and `roadEdgeStds`, whose threshold calibration shifts under quantization.
Set `MODEL_QNN_STRICT_REPLAY=1` to include those uncertainty fields while auditing a newly generated model.

```sh
MODEL_BACKEND=onnx MODEL_ONNX_PATH=openpilot/selfdrive/modeld/models/driving_supercombo_qnn.onnx \
uv run --with matplotlib --with onnxruntime==1.28.0 \
  python openpilot/selfdrive/test/process_replay/model_replay.py
```
