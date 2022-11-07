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
const_35 = relay.const([[-3,-1,-8,-5,-10,-4,-2,-3,8,10,-2,4],[-4,-5,9,-1,8,9,4,2,9,-8,-1,-1],[4,4,-2,4,-9,-7,6,1,1,-1,-6,-1],[-9,3,-2,8,-6,4,9,6,-6,-6,-10,-8],[7,-7,10,-7,1,-8,-4,8,7,10,9,5],[8,10,10,-10,-7,9,7,8,-8,10,1,2],[2,-5,7,-8,3,4,8,-10,10,-4,-7,-6],[7,7,-6,7,2,-10,-3,-5,-8,10,10,3]], dtype = "uint32")#candidate|35|(8, 12)|const|uint32
uop_36 = relay.acosh(const_35) # shape=(8, 12)
output = relay.Tuple([uop_36,])
output2 = relay.Tuple([uop_36,])
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
