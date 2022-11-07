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
const_15 = relay.const([[-10,-8,5,1,-7,-3,-5,4,-9,-4,8,-5,6],[-7,-4,6,-7,4,-4,-3,9,3,-10,-7,-2,3],[7,-7,5,-7,6,-4,2,4,-4,-6,-6,-6,-6],[7,-6,-5,8,5,8,2,-7,-9,-6,-4,4,3],[9,-6,2,2,-4,8,-1,-5,5,4,-8,3,-8],[4,7,9,7,-2,9,-7,-2,-4,10,-3,9,-8],[-2,10,-10,-1,-2,-6,8,-9,1,-9,-3,-6,-2],[-9,-8,-7,2,2,2,10,8,4,-1,2,-9,4],[1,6,-1,-4,3,-3,1,-6,3,-8,1,4,4],[-5,4,-6,1,8,-8,5,6,-6,-5,9,-10,6],[10,5,-7,-9,-2,-6,7,-7,8,8,-1,-10,-7],[-4,-7,-8,2,1,6,10,2,-10,-2,6,8,8],[8,-6,9,4,10,-3,7,-4,1,-10,-2,4,7],[5,-5,5,-3,8,10,-7,6,-5,-1,8,2,7]], dtype = "uint64")#candidate|15|(14, 13)|const|uint64
uop_16 = relay.atanh(const_15) # shape=(14, 13)
output = uop_16
output2 = uop_16
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
