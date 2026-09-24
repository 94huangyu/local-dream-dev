import onnx
from onnx import numpy_helper
import numpy as np
import sys
import os

def fix_onnx_model_for_qnn(model_path):
    print(f"[FIX_GRAPH] Processing {model_path} to remove QNN-incompatible NO-OPs...")
    model = onnx.load(model_path, load_external_data=False)
    
    nodes_to_remove = []
    new_nodes = []
    replaced = 0
    
    for i, node in enumerate(model.graph.node):
        # 1. Fix 1-element Split
        if node.op_type == "Split" and len(node.input) == 2:
            split_input = node.input[1]
            split_val = None
            for init in model.graph.initializer:
                if init.name == split_input:
                    split_val = np.frombuffer(init.raw_data, dtype=np.int64)
                    break
            if split_val is not None and len(split_val) == 1:
                print(f"  - Replacing 1-element Split: {node.name}")
                new_node = onnx.helper.make_node(
                    'Identity',
                    inputs=[node.input[0]],
                    outputs=node.output,
                    name=node.name + "_identity"
                )
                new_nodes.append(new_node)
                replaced += 1
                continue
                
        # 2. Fix 0-pad Pad
        if node.op_type == "Pad":
            pads_input = node.input[1]
            pads_val = None
            for init in model.graph.initializer:
                if init.name == pads_input:
                    pads_val = numpy_helper.to_array(init)
                    break
            if pads_val is not None and np.all(pads_val == 0):
                print(f"  - Replacing 0-pad Pad: {node.name}")
                new_node = onnx.helper.make_node(
                    'Identity',
                    inputs=[node.input[0]],
                    outputs=node.output,
                    name=node.name + "_identity"
                )
                new_nodes.append(new_node)
                replaced += 1
                continue
                
        # 3. Fix ScatterElements
        if node.op_type == "ScatterElements":
            print(f"  - Replacing ScatterElements: {node.name}")
            updates_input = node.input[2]
            new_node = onnx.helper.make_node(
                'Identity',
                inputs=[updates_input],
                outputs=node.output,
                name=node.name + "_identity"
            )
            new_nodes.append(new_node)
            replaced += 1
            continue

        new_nodes.append(node)
        
    if replaced > 0:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        print(f"  [FIX_GRAPH] Saving model {model_path} ({replaced} replacements made)...")
        onnx.save(model, model_path)
    else:
        print(f"  [FIX_GRAPH] No replacements needed.")
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_qnn_graph.py <onnx_path>")
        sys.exit(1)
    fix_onnx_model_for_qnn(sys.argv[1])
