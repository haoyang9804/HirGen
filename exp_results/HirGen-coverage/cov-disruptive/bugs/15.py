import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.contrib import graph_runtime
import numpy as np
def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        return o


mod = tvm.IRModule()
mutated_mod = tvm.IRModule()
const_3 = relay.const([[[5,-2,-5,5,-3,-8,4,-10,9,-2,7,1,3],[-9,3,3,-1,-9,4,-10,-6,4,8,-5,-3,5],[-7,4,-10,-8,-8,-3,8,7,5,10,8,1,7],[-10,-7,10,3,5,-6,8,-6,5,4,5,-1,-1],[7,4,2,-6,-1,2,-10,-9,9,-2,-10,-8,-1],[4,-8,-1,-8,10,-9,8,1,-3,-8,9,2,2],[-10,-6,3,-1,6,-5,8,-6,-10,-6,6,-5,-2],[2,8,5,4,7,3,8,1,-4,9,-6,6,-1],[9,-7,-10,4,2,-6,-1,4,4,-3,-5,5,-7],[-2,-3,5,9,-5,-10,-9,-8,-9,-1,-5,4,9],[5,3,5,4,8,-9,1,8,-3,-4,-3,2,-7],[-8,-4,-10,2,-8,-9,-4,4,-4,9,6,8,-5]],[[-6,-8,3,-4,4,4,7,5,-6,3,-4,5,-8],[7,5,-10,10,-4,3,-10,1,-2,-7,6,-8,5],[5,-7,-3,10,10,-8,7,3,-9,-6,-6,1,-7],[-1,8,-7,8,-8,2,-10,10,-3,10,-5,-9,-1],[9,-2,-9,-1,9,-1,-6,7,-9,-2,-4,3,1],[5,1,6,-6,-8,9,3,9,-4,-3,5,-5,10],[-10,-3,5,2,2,7,6,6,-5,5,2,-1,8],[8,2,3,5,4,4,5,6,7,-7,7,-9,-8],[-2,8,6,5,8,-10,6,2,-10,7,8,7,-4],[-7,-9,-7,-5,-2,3,-7,-5,-3,-10,10,-1,-7],[-10,2,1,8,6,10,10,-10,-1,-3,5,9,2],[-5,-8,-2,7,2,-5,5,-6,1,-8,1,2,-6]]], dtype = "uint32")#candidate|3|(2, 12, 13)|const|uint32
uop_4 = relay.atan(const_3) # shape=(2, 12, 13)
output = uop_4
output2 = uop_4
F = relay.Function([], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([], output2)
mutated_mod['main'] = F
mutated_mod = relay.transform.InferType()(mutated_mod)
print('==========mutated_mod==========')
print(mutated_mod.astext(show_meta_data=False))
print('===================================')
graph, lib, params = relay.build(mod, target='llvm')
module1 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp2 = relay.build_module.create_executor('graph', mod, tvm.device('llvm',0),'llvm')
intrp3 = relay.build_module.create_executor('debug', mod, tvm.device('llvm',0),'llvm')
intrp4 = relay.build_module.create_executor('vm', mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mod, target='cuda')
module5 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp6 = relay.build_module.create_executor('graph', mod, tvm.device('cuda',0),'cuda')
intrp7 = relay.build_module.create_executor('debug', mod, tvm.device('cuda',0),'cuda')
intrp8 = relay.build_module.create_executor('vm', mod, tvm.device('cuda',0),'cuda')
