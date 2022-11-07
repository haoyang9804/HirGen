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
const_20 = relay.const([[[10,3,2],[-5,-2,-5],[-2,10,7],[-6,10,4],[8,-8,7],[6,9,-7],[-5,4,-7],[10,5,-3],[-7,9,-10]],[[7,-7,8],[8,-3,-5],[-8,5,10],[2,-1,7],[9,4,1],[3,3,-5],[-5,-7,10],[9,4,9],[-7,-1,7]],[[-4,-4,-6],[-2,7,10],[6,-6,4],[-7,-5,8],[-5,6,3],[6,10,6],[8,-5,-3],[8,-10,4],[-2,8,10]]], dtype = "uint32")#candidate|20|(3, 9, 3)|const|uint32
uop_21 = relay.atan(const_20) # shape=(3, 9, 3)
output = uop_21
output2 = uop_21
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
