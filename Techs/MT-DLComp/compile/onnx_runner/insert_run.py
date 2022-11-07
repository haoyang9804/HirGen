import os
import sys

import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as rt


model = onnx.load(sys.argv[1])
model = shape_inference.infer_shapes(model)

edge =  [e for e in model.graph.value_info if e.name == '390'][0]
input_data = np.load(sys.argv[2])

save_path = os.path.join(sys.argv[3], "model.onnx")

model.graph.output.insert(0, edge)
onnx.save(model, save_path)

sess = rt.InferenceSession(save_path)
input_names = sess.get_inputs()
output_name = [o.name for o in sess.get_outputs()]
if input_names:
    input_name = input_names[0].name

    out = sess.run(output_name, {input_name: input_data})
else:
    out = sess.run(output_name, {})
