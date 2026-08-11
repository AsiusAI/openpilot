## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

`driving_supercombo_qnn.onnx` is the experimental QDQ model for Asius/Dragon Q6A. QNN places all model arithmetic,
including the vision network and recurrent temporal policy, on the HTP. ONNX Runtime retains four fixed-shape `Reshape`
nodes as CPU bookkeeping. The generator rewrites unsupported fixed-shape operations and factors per-channel convolution
scales into HTP-supported per-tensor convolutions. One depthwise layer with a four-order-of-magnitude scale range is split
into power-of-two scale buckets to retain accuracy. The camera warp remains on the Qualcomm MSM GPU so the CPU only has to
marshal the warped tensor into QNN.

The HTP path is not car-ready yet: device replay currently shows recurrent output drift relative to the validated MSM
backend. Build it explicitly with `MODELD_DEV=QNN` for bench work. The safe default, and the configuration to use in a car,
is `MODELD_DEV=QCOM`; this is the only modeld build-time backend switch.

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
The replay compares uncertainty and action outputs and uses a tighter QNN tolerance than the general cross-hardware replay.
Do not relax or ignore those fields to make a quantized model pass; compare QNN and MSM output logs from the same Dragon run.

```sh
MODEL_BACKEND=onnx MODEL_ONNX_PATH=openpilot/selfdrive/modeld/models/driving_supercombo_qnn.onnx \
uv run --with matplotlib --with onnxruntime==1.28.0 \
  python openpilot/selfdrive/test/process_replay/model_replay.py
```
