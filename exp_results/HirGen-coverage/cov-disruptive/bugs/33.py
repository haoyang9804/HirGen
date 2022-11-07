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
const_48 = relay.const([[[-1,-4,-2,-6,5],[2,4,-4,7,8],[-10,8,10,-6,-2],[10,9,-4,-4,-1]],[[-2,-4,8,7,-2],[5,-2,7,-4,-4],[5,2,5,6,-8],[-6,-9,7,2,3]],[[2,5,9,-7,-6],[-1,-4,6,1,-9],[-6,1,9,3,3],[5,-3,1,-6,-5]],[[1,5,1,7,1],[2,2,5,1,-3],[5,4,-10,-10,-3],[-9,10,-10,-5,-4]],[[-2,6,-10,-7,10],[-9,-8,2,6,-2],[-8,-5,-5,1,3],[-6,4,1,4,6]],[[8,10,-10,3,6],[-7,6,-3,7,3],[-1,-7,3,-8,-5],[-3,-2,8,7,8]],[[-7,-10,-6,-6,-8],[5,7,2,-7,7],[5,-5,-2,6,4],[-7,-8,-4,1,-9]],[[8,-8,-3,5,3],[4,-1,-4,3,4],[-6,8,7,10,5],[-5,-1,-10,-5,6]],[[6,-3,-10,9,9],[-3,-2,10,-5,7],[10,-3,-7,7,7],[-8,-3,9,2,-9]]], dtype = "uint64")#candidate|48|(9, 4, 5)|const|uint64
uop_49 = relay.atanh(const_48) # shape=(9, 4, 5)
output = relay.Tuple([uop_49,])
output2 = relay.Tuple([uop_49,])
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
