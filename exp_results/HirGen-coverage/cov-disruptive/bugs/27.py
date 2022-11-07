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
const_7 = relay.const([[[False,False,True],[False,True,True],[True,True,True],[False,False,True],[False,True,True],[True,True,False],[True,True,True],[True,False,True],[True,True,False],[True,True,True],[True,True,True],[False,True,True]],[[True,False,False],[False,False,True],[True,True,False],[False,False,True],[False,False,False],[True,True,False],[True,False,False],[True,True,True],[False,False,False],[False,False,True],[True,True,True],[True,True,True]],[[False,False,False],[False,False,True],[False,False,True],[False,False,False],[True,True,True],[True,False,False],[False,True,True],[False,True,True],[True,False,True],[False,True,False],[True,False,True],[False,False,True]],[[True,False,False],[False,True,False],[False,False,True],[True,True,True],[False,True,False],[True,True,True],[False,False,True],[True,False,True],[False,True,True],[True,True,True],[False,False,False],[False,False,True]],[[False,True,True],[True,False,False],[True,False,True],[True,True,False],[True,False,False],[False,True,False],[True,True,True],[False,False,True],[False,True,True],[False,True,False],[True,False,True],[False,True,True]],[[False,False,False],[True,False,True],[True,True,True],[True,True,True],[True,True,False],[False,True,True],[True,True,False],[True,True,True],[True,False,True],[False,False,True],[True,False,True],[True,True,True]],[[True,False,False],[False,True,False],[True,False,True],[False,True,False],[True,False,False],[True,True,True],[False,False,False],[False,False,False],[True,False,True],[False,True,False],[False,False,True],[False,True,False]],[[False,False,True],[True,False,False],[False,True,False],[False,False,False],[True,True,False],[True,True,True],[False,False,True],[True,False,False],[True,False,True],[False,True,False],[True,True,False],[False,True,True]],[[False,True,False],[False,True,True],[False,False,False],[True,False,True],[False,False,True],[True,True,True],[True,False,False],[False,True,True],[False,False,False],[True,False,True],[False,False,False],[False,False,True]],[[True,True,False],[True,False,False],[False,False,False],[True,True,True],[True,False,True],[True,True,False],[False,True,False],[False,False,False],[False,False,False],[False,True,True],[False,False,False],[False,False,False]],[[False,False,False],[False,False,True],[True,True,True],[False,False,False],[True,False,False],[True,False,False],[False,False,True],[False,True,False],[True,True,False],[True,True,False],[True,True,True],[True,True,True]],[[False,False,False],[False,False,False],[False,True,False],[False,True,False],[True,True,True],[False,True,False],[False,True,True],[False,False,False],[True,False,True],[False,True,False],[True,False,False],[True,False,False]],[[True,False,False],[True,True,True],[True,False,False],[False,False,True],[False,False,False],[True,False,True],[True,True,True],[False,True,False],[False,False,False],[False,True,False],[True,False,True],[True,True,False]]], dtype = "bool")#candidate|7|(13, 12, 3)|const|bool
uop_8 = relay.atanh(const_7) # shape=(13, 12, 3)
output = uop_8
output2 = uop_8
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
