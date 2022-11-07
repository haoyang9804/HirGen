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
const_45 = relay.const([[[8,-8,1,-8,-8,-2,-8,-2,5,-6,4],[-2,5,-10,-7,-1,-10,9,-8,4,6,2],[-2,-10,-10,4,10,8,9,2,-4,-8,-3],[4,4,1,5,3,-8,4,9,3,-3,-7]],[[-7,-6,-2,-5,-5,-8,8,8,-3,-1,-6],[-10,-3,6,-1,8,-10,1,3,8,-5,-7],[-8,-8,-6,3,-4,-6,-3,2,-5,-6,-9],[8,-2,-7,5,-2,9,-5,9,-8,10,-1]],[[-6,8,5,6,-10,-5,-1,9,-8,-10,-1],[3,-2,4,-5,-6,9,-2,-8,10,3,10],[10,1,-2,-5,5,-4,9,8,2,2,-7],[-8,3,7,-3,-9,8,-1,10,-1,3,1]],[[-3,5,2,-9,-2,-8,-4,10,-9,10,-8],[8,-6,2,-6,1,-3,-7,4,5,-10,2],[5,-1,2,4,2,-1,-2,-8,-7,8,-7],[-4,-8,10,4,-3,-5,-5,-6,10,-2,-6]],[[3,5,-7,9,-7,-7,-6,4,-1,8,1],[-7,-6,-5,-8,-5,5,1,2,-5,5,-3],[-1,8,3,-9,1,9,3,-8,-5,1,2],[1,2,-4,-2,-7,-3,6,-2,8,3,3]],[[-3,6,-3,-7,9,-2,8,9,2,-2,10],[8,-2,-7,8,-10,10,-7,3,9,-4,6],[-4,9,-10,10,-8,3,-7,-7,-8,-5,5],[1,-8,1,-1,3,1,-3,4,2,-2,-5]],[[3,5,1,2,-2,3,3,-1,2,-2,-5],[-3,-8,2,-7,-10,-1,4,5,-7,8,6],[-5,-6,10,-9,2,-5,-8,-2,-5,7,-9],[7,8,-2,5,8,6,8,-1,-2,-10,5]],[[5,-5,2,-8,5,-5,-9,-9,-5,-6,-10],[-7,-1,8,-8,-1,10,5,10,-3,9,9],[-2,-6,-5,10,-5,10,5,-9,-1,-8,-6],[3,6,-2,1,-7,-10,-4,3,4,-10,-6]],[[-10,-10,5,2,-10,-9,2,-5,3,-9,-5],[1,6,10,10,3,-10,1,-10,7,1,-10],[9,5,7,6,7,-2,3,-9,7,-2,-6],[2,-9,7,9,2,-3,10,1,4,3,10]],[[-7,10,-10,2,-7,-3,-7,6,-10,-2,5],[4,7,-3,3,10,-2,-7,3,10,4,-6],[-6,7,-1,10,2,-4,-2,-1,2,-9,2],[-9,-3,-2,-4,10,-1,7,-7,-9,-3,-7]]], dtype = "uint8")#candidate|45|(10, 4, 11)|const|uint8
uop_46 = relay.atan(const_45) # shape=(10, 4, 11)
output = uop_46
output2 = uop_46
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
