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
const_64 = relay.const([[[1,-5,-8,2,10,4,-5,5,7,-7],[3,-3,-1,-9,10,-2,1,-1,-10,-1],[-2,-8,5,-8,-7,-3,4,1,5,-2],[-1,-5,4,1,8,4,-1,6,6,-2]],[[10,3,1,3,8,-10,-2,1,-2,9],[8,-9,-8,7,-10,8,2,6,-7,-9],[2,4,-4,-9,-1,6,4,10,-10,-5],[-6,10,6,3,-5,-10,2,1,1,4]],[[-10,5,8,5,2,10,-3,-8,-8,-4],[-3,-10,-8,-5,-2,-5,3,-4,5,1],[-9,7,-3,5,-6,-4,3,8,-6,-2],[-3,-1,-7,7,6,1,-9,9,3,-5]],[[3,8,5,7,8,-10,-8,-7,-5,7],[4,-9,6,7,4,-6,1,-4,1,-8],[3,-2,-8,-6,-5,6,-2,-2,-5,-6],[1,7,-10,-10,2,-4,5,-6,6,8]],[[8,1,-1,6,4,-4,-8,8,-9,10],[6,6,9,6,5,-7,10,-1,-2,4],[3,-10,-8,-10,9,-8,-4,-8,-8,8],[-10,4,-2,-8,6,-8,-1,-1,-1,8]],[[10,6,-2,-9,-1,-2,2,1,-5,-6],[-3,1,5,5,-4,5,-10,-10,-5,3],[-5,-1,4,-1,-2,-1,9,1,3,-9],[10,1,-3,4,-1,8,-5,5,10,8]],[[10,7,3,-5,-3,-3,8,-4,-8,-1],[-7,-7,9,-2,7,-10,-1,-8,4,-1],[10,3,2,9,6,-1,10,-7,-9,7],[-9,4,-4,6,-8,7,3,-3,7,-4]],[[-1,-10,-2,-9,4,-10,-3,-4,-9,-4],[-7,6,-3,6,-3,7,-7,7,3,-4],[-2,-1,1,1,-8,-8,8,-7,-4,-4],[-5,-1,7,-2,3,-9,-10,-7,6,10]],[[5,-10,-3,5,-10,-2,-2,-7,-10,1],[-7,-8,9,-10,5,10,10,-2,-7,-10],[-8,3,-7,8,9,3,-10,-3,-8,5],[-2,3,4,-5,-6,9,-9,-3,1,4]],[[-1,-6,-9,9,1,-3,10,7,3,7],[8,3,-8,-2,8,-1,-4,-1,7,10],[-5,-10,-1,8,-10,-3,-3,-6,3,-2],[-5,-1,1,3,3,5,-8,-2,4,3]],[[7,2,4,-10,7,9,6,-10,1,4],[-9,-3,-2,8,-7,9,6,-5,-1,-7],[-9,-8,-6,2,6,-3,-3,-3,-1,2],[-1,5,6,10,-7,4,5,2,4,1]],[[10,-6,2,1,-2,3,-5,1,-10,-9],[-9,-6,-6,-8,-4,2,-4,-8,-10,10],[-3,-5,5,2,-8,4,-3,-2,9,-1],[8,7,-6,8,6,-7,-5,9,-10,-5]],[[9,6,10,-3,7,-1,9,-6,-5,-10],[4,-1,-2,7,1,-8,1,-2,-1,9],[-9,2,-8,-5,-5,-7,-2,6,-9,6],[4,10,-8,-6,7,9,-6,6,-1,9]],[[5,3,-4,-3,-3,-3,-3,-4,-5,-3],[6,10,3,5,8,10,-2,3,5,9],[5,6,-2,2,-1,-2,7,8,-10,5],[-9,-1,9,4,-2,-1,4,-8,10,-6]],[[4,-9,-9,-2,9,7,4,8,-4,9],[-5,4,10,8,9,-2,-1,4,7,1],[3,6,9,-3,6,8,2,3,-1,5],[-4,-5,-7,6,-4,4,8,5,3,10]],[[-7,3,10,3,-10,3,5,-10,6,-10],[7,-8,-3,-10,-8,-6,-9,-6,8,2],[10,-1,-5,9,8,-2,3,-10,2,5],[-2,-6,-1,-10,4,8,-9,-8,1,-6]]], dtype = "int32")#candidate|64|(16, 4, 10)|const|int32
uop_65 = relay.sqrt(const_64) # shape=(16, 4, 10)
output = relay.Tuple([uop_65,])
output2 = relay.Tuple([uop_65,])
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
