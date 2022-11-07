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
const_31 = relay.const([[-4,-10,-7,9,-8,6,-10,-9,1,-6],[5,-1,5,-3,-7,-2,4,-4,-10,2],[4,7,-4,-10,6,-5,8,-7,8,-9],[-5,-8,6,7,-6,4,-6,-4,7,-7],[-10,3,8,10,10,-1,-4,-3,4,4],[10,2,1,8,1,-1,6,10,-6,2],[-2,2,-3,-4,-4,8,8,-9,5,-7],[3,-7,3,-4,10,6,8,-7,2,9],[2,9,-5,-1,8,-7,7,9,-3,-1],[-9,-4,7,-10,-5,-5,6,8,-6,4],[-6,-4,-8,5,8,-6,-2,-3,-4,-8],[-4,7,-10,-5,-4,2,-1,5,1,-3],[-3,-5,-3,-6,5,-7,10,1,9,-8],[-7,6,-9,9,8,-9,5,10,6,5],[3,-10,-8,-4,-9,1,-3,5,-8,1]], dtype = "int8")#candidate|31|(15, 10)|const|int8
uop_32 = relay.atan(const_31) # shape=(15, 10)
output = uop_32
output2 = uop_32
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
