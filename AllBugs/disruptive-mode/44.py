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
const_78 = relay.const([[[2,9,-3,4,-7,-3,9,2,-7,7],[-6,-1,-10,2,5,-9,-6,-10,7,10]],[[-2,10,6,-9,10,6,-2,3,-10,-10],[-6,2,7,10,-6,8,5,1,2,8]],[[-7,8,-1,-3,-5,-4,10,-3,-7,-3],[-5,-3,10,4,8,-4,4,6,2,-6]],[[10,9,7,-1,9,7,4,4,8,6],[10,10,-1,-9,6,3,5,-10,-6,-7]]], dtype = "uint16")#candidate|78|(4, 2, 10)|const|uint16
uop_79 = relay.sqrt(const_78) # shape=(4, 2, 10)
output = relay.Tuple([uop_79,])
output2 = relay.Tuple([uop_79,])
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
