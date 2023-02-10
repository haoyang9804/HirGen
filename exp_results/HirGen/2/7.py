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
var_26 = relay.var("var_26", dtype = "float64", shape = (5,))#candidate|26|(5,)|var|float64
uop_27 = relay.sigmoid(var_26.astype('float64')) # shape=(5,)
output = uop_27
output2 = uop_27
func_29 = relay.Function([var_26,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
mutated_mod['func_29'] = func_29
mutated_mod = relay.transform.InferType()(mutated_mod)
var_30 = relay.var("var_30", dtype = "float64", shape = (5,))#candidate|30|(5,)|var|float64
func_29_call = mutated_mod.get_global_var('func_29')
call_31 = func_29_call(var_30)
output = call_31
func_32 = relay.Function([var_30], output)
mutated_mod['func_32'] = func_32
mutated_mod = relay.transform.InferType()(mutated_mod)
const_34 = relay.const([-8,-9,-8,-4,-5,6,-1,4,-6], dtype = "uint8")#candidate|34|(9,)|const|uint8
var_35 = relay.var("var_35", dtype = "uint8", shape = (9,))#candidate|35|(9,)|var|uint8
bop_36 = relay.bitwise_xor(const_34.astype('uint8'), relay.reshape(var_35.astype('uint8'), relay.shape_of(const_34))) # shape=(9,)
uop_39 = relay.acosh(bop_36.astype('float64')) # shape=(9,)
bop_41 = relay.bitwise_or(uop_39.astype('int32'), relay.reshape(const_34.astype('int32'), relay.shape_of(uop_39))) # shape=(9,)
bop_44 = relay.add(const_34.astype('int32'), relay.reshape(bop_41.astype('int32'), relay.shape_of(const_34))) # shape=(9,)
const_47 = relay.const([1,6,-2,-9,-3,-3,-4,-8,9], dtype = "int32")#candidate|47|(9,)|const|int32
bop_48 = relay.subtract(bop_41.astype('float64'), relay.reshape(const_47.astype('float64'), relay.shape_of(bop_41))) # shape=(9,)
bop_53 = relay.equal(uop_39.astype('bool'), relay.reshape(bop_48.astype('bool'), relay.shape_of(uop_39))) # shape=(9,)
const_56 = relay.const([-6.797297,-7.865585,-1.301504,7.417431,-3.409476,-9.596064,-3.421512,5.423958,-0.877806], dtype = "float64")#candidate|56|(9,)|const|float64
bop_57 = relay.logical_xor(bop_48.astype('int8'), relay.reshape(const_56.astype('int8'), relay.shape_of(bop_48))) # shape=(9,)
uop_60 = relay.rsqrt(bop_36.astype('float64')) # shape=(9,)
bop_62 = relay.floor_divide(bop_44.astype('float64'), relay.reshape(var_35.astype('float64'), relay.shape_of(bop_44))) # shape=(9,)
uop_77 = relay.atan(uop_39.astype('float64')) # shape=(9,)
output = relay.Tuple([bop_53,bop_57,uop_60,bop_62,uop_77,])
output2 = relay.Tuple([bop_53,bop_57,uop_60,bop_62,uop_77,])
func_80 = relay.Function([var_35,], output)
mod['func_80'] = func_80
mod = relay.transform.InferType()(mod)
mutated_mod['func_80'] = func_80
mutated_mod = relay.transform.InferType()(mutated_mod)
var_81 = relay.var("var_81", dtype = "uint8", shape = (9,))#candidate|81|(9,)|var|uint8
func_80_call = mutated_mod.get_global_var('func_80')
call_82 = func_80_call(var_81)
output = call_82
func_83 = relay.Function([var_81], output)
mutated_mod['func_83'] = func_83
mutated_mod = relay.transform.InferType()(mutated_mod)
var_91 = relay.var("var_91", dtype = "float64", shape = (5, 5))#candidate|91|(5, 5)|var|float64
uop_92 = relay.sin(var_91.astype('float64')) # shape=(5, 5)
bop_94 = relay.right_shift(uop_92.astype('uint8'), relay.reshape(var_91.astype('uint8'), relay.shape_of(uop_92))) # shape=(5, 5)
func_29_call = mod.get_global_var('func_29')
func_32_call = mutated_mod.get_global_var('func_32')
const_100 = relay.const([1.381015,9.907970,-2.063932,-1.731952,8.747546], dtype = "float64")#candidate|100|(5,)|const|float64
call_99 = func_29_call(relay.reshape(const_100.astype('float64'), [5,]))
call_101 = func_29_call(relay.reshape(const_100.astype('float64'), [5,]))
uop_105 = relay.cos(uop_92.astype('float64')) # shape=(5, 5)
func_80_call = mod.get_global_var('func_80')
func_83_call = mutated_mod.get_global_var('func_83')
const_108 = relay.const([-4,-2,9,-4,-1,3,-6,-8,-10], dtype = "uint8")#candidate|108|(9,)|const|uint8
call_107 = relay.TupleGetItem(func_80_call(relay.reshape(const_108.astype('uint8'), [9,])), 2)
call_109 = relay.TupleGetItem(func_83_call(relay.reshape(const_108.astype('uint8'), [9,])), 2)
output = relay.Tuple([bop_94,call_99,const_100,uop_105,call_107,const_108,])
output2 = relay.Tuple([bop_94,call_101,const_100,uop_105,call_109,const_108,])
func_111 = relay.Function([var_91,], output)
mod['func_111'] = func_111
mod = relay.transform.InferType()(mod)
mutated_mod['func_111'] = func_111
mutated_mod = relay.transform.InferType()(mutated_mod)
var_112 = relay.var("var_112", dtype = "float64", shape = (5, 5))#candidate|112|(5, 5)|var|float64
func_111_call = mutated_mod.get_global_var('func_111')
call_113 = func_111_call(var_112)
output = call_113
func_114 = relay.Function([var_112], output)
mutated_mod['func_114'] = func_114
mutated_mod = relay.transform.InferType()(mutated_mod)
const_116 = relay.const([[[-9,6,7,4,10,7,6,-1,-10,-1,-2,-5,-6],[3,1,3,-9,5,-2,5,-8,-8,-10,1,7,6],[-4,4,3,5,5,3,8,9,-9,-1,-1,-9,5],[-2,1,-6,8,2,-9,-5,-6,8,8,-5,-4,-2],[-3,-7,-3,5,-6,9,-8,5,-2,4,1,-4,-9],[-6,5,-9,10,-1,4,9,4,4,-10,1,-1,10],[7,1,-4,3,4,10,-8,9,6,4,5,-1,-8],[7,-1,1,2,6,-10,7,-10,10,-7,-2,6,-2],[5,-5,10,-2,10,9,-9,10,-1,-5,-6,-9,-5],[-8,8,-1,-5,-5,10,-6,-5,4,9,3,-3,-2],[-3,-1,-5,-9,-9,5,-7,-1,-10,-7,1,8,7],[3,-10,-10,8,-3,-10,-4,-1,5,1,-4,2,-5],[3,6,-3,-2,3,-7,-3,4,3,4,-8,1,-6],[10,8,-2,7,3,-6,6,-8,2,3,7,-3,9]],[[2,3,-7,-6,8,-6,-8,8,5,-7,7,4,-3],[-4,-2,2,-5,1,-9,4,9,-4,-10,-2,6,-8],[7,1,-3,-10,9,10,-6,5,10,-3,-1,5,-9],[8,6,-6,-8,-7,-5,-7,-9,-8,7,-10,-3,5],[2,-7,7,-6,-1,-10,-1,-1,3,3,-4,9,-4],[-10,4,1,5,6,-3,-7,-5,5,-3,4,-4,-10],[-8,4,1,-9,9,-10,-9,-4,-9,9,7,3,10],[-8,6,7,-8,-6,-4,4,6,-6,2,-9,3,8],[2,2,9,-8,-7,-9,-5,3,-1,-3,-3,-3,-10],[-8,2,5,-5,8,6,-9,3,9,4,-7,1,9],[10,-8,4,-5,2,-2,8,3,8,-8,-1,-8,1],[7,7,5,-9,8,-1,-2,5,-1,4,9,-7,10],[4,-2,-5,-8,-8,10,7,3,-2,-10,8,2,-7],[-1,-9,-2,-1,9,-5,6,9,-3,3,4,-10,4]],[[8,-8,-10,9,2,-4,8,-8,-10,8,-2,8,6],[3,-8,4,5,-3,4,4,-7,-8,-2,10,3,-3],[-9,-10,-3,-3,-10,-9,10,9,7,-10,10,9,-6],[-3,-7,4,7,4,-9,-1,3,-3,5,5,-5,-6],[2,-8,4,-9,-8,-8,-7,5,7,-4,-9,3,10],[7,2,-4,-1,-5,4,6,7,6,-8,1,-6,-8],[-3,-10,9,-2,-1,9,9,-6,8,-4,8,10,-1],[-1,-5,4,10,-2,-9,1,-3,1,4,4,9,-8],[7,3,8,-7,-3,-2,-2,1,1,3,-2,-5,-2],[-10,-6,-8,-6,-1,4,-9,8,7,8,-3,-10,4],[-1,-8,-7,-3,-9,-7,9,1,-7,-10,4,7,7],[-2,8,-9,-4,-1,-9,5,5,-3,4,-10,6,6],[-7,-3,-4,-1,10,3,-9,6,-9,6,-7,-10,2],[-1,-8,-8,1,9,-6,-8,-2,-10,-5,9,-1,10]],[[3,-6,8,3,6,2,3,-3,-2,5,1,-7,6],[-4,10,-3,-4,7,-8,-9,-2,-10,-2,3,-5,10],[-6,4,-2,6,-2,6,-10,9,3,-4,9,-1,-8],[-9,-3,-9,-7,-8,3,4,5,-3,-2,4,-10,-4],[2,-4,-10,-3,3,1,-7,2,4,-7,-5,-5,3],[-4,-10,2,6,4,10,-8,-1,4,9,-10,-1,6],[-7,2,-4,3,-7,-8,2,8,1,-6,4,-2,5],[9,-1,9,-6,10,-9,4,9,-8,5,-4,9,-1],[-9,9,-7,9,3,-9,-10,9,-3,3,6,-4,7],[-9,1,3,-3,5,9,9,1,1,7,1,6,2],[7,-4,8,-3,3,9,6,-9,8,-9,8,-2,-1],[-4,7,8,6,8,-7,-2,-7,-10,5,-2,-7,-3],[10,-9,6,-9,-1,-9,-10,8,-2,-10,3,3,-9],[-10,5,-1,-7,-7,4,-10,-4,-10,9,-4,2,7]],[[-3,-3,7,-9,5,9,-8,8,2,-7,-8,1,-8],[-6,8,-4,-6,-5,10,-5,-2,10,-1,-4,-6,-1],[9,-10,8,5,-1,-9,8,2,-8,5,-3,1,-1],[9,7,4,4,3,1,-10,-1,-9,-4,6,-10,-1],[-6,-7,1,-6,7,-4,-3,-5,-9,7,3,-10,7],[-9,7,-9,4,1,3,-4,7,-3,9,7,2,-9],[-9,4,-4,-4,4,-5,-9,-4,6,-5,9,5,8],[7,5,-9,-2,-10,-7,-9,-8,-9,-1,-3,2,1],[-8,1,-10,9,2,-10,10,-1,4,8,-4,-1,2],[3,-2,7,-3,-7,5,-10,4,1,10,3,-2,8],[-9,-3,4,5,2,-9,8,-10,2,10,-8,1,3],[-3,-6,2,-3,6,-1,-5,-5,-8,2,6,7,-1],[-6,-4,8,-5,10,10,2,-1,9,6,2,-6,5],[8,9,7,-2,6,6,-4,9,-10,-2,-4,8,-6]],[[7,5,-6,-1,4,2,-9,-4,2,-1,-2,3,-3],[6,8,-4,-3,-6,5,6,9,3,-8,3,7,-2],[7,2,2,-3,-9,7,7,-3,-5,10,-6,-10,-2],[-6,-6,-7,-4,5,8,-6,5,10,-7,4,9,3],[-7,-2,5,7,6,6,-4,6,7,10,5,3,1],[3,-1,7,-1,-8,-7,-10,10,5,-5,7,-2,3],[5,9,-3,-7,-1,2,-5,10,9,9,6,-5,-4],[2,-7,1,-2,9,-5,10,3,9,4,4,-9,-3],[6,5,-10,5,-3,1,-9,1,-9,1,2,-3,-9],[-2,-1,-1,-4,-8,3,-6,-10,-2,-7,2,-1,-9],[-2,3,-9,-8,-1,9,7,-4,9,-7,-4,7,-1],[9,-1,-6,1,4,4,4,3,3,-1,10,-6,10],[-7,3,-5,9,3,10,-5,-7,6,-8,-5,9,-7],[-5,-8,-3,1,1,10,-5,-5,8,9,1,-5,4]],[[-10,-5,2,2,-4,-2,-4,-2,-5,9,1,-2,-1],[6,3,-7,3,-4,3,-4,4,3,-3,2,1,7],[-7,2,1,-7,8,-9,-7,7,-8,8,-8,-8,5],[8,1,9,4,7,9,-2,-7,7,2,5,-4,-10],[3,3,9,-5,-7,-10,6,-8,3,-10,-9,-6,1],[-10,2,7,-9,-5,-4,-1,-5,-3,8,5,-1,4],[4,-2,-1,9,-4,9,9,-4,-4,-3,5,9,-7],[-7,-1,-10,-10,1,-5,-5,9,2,7,8,-8,2],[4,9,8,-2,6,6,-2,-7,9,6,7,-3,-2],[8,1,7,1,-9,-1,-10,2,-7,4,3,-6,-9],[-7,-9,-3,-2,7,10,-5,-9,-9,-5,6,7,-6],[1,-9,6,5,6,-2,-2,6,-7,2,-7,-4,4],[-6,7,-1,-7,3,8,-9,-5,1,-10,-1,9,-10],[4,1,-3,7,1,2,-10,-1,10,1,3,-5,2]],[[-7,-9,5,2,-9,5,6,-6,3,-3,4,-4,-1],[10,9,1,-1,4,4,-4,-2,-1,-2,-9,2,-9],[5,3,2,3,-3,7,7,-2,1,3,2,4,-10],[-8,6,1,1,-9,-7,-2,9,6,3,6,-1,-9],[-1,-6,-7,3,10,-7,8,-3,8,-7,7,2,1],[-5,-5,5,9,-8,6,10,-7,-8,-2,-10,-8,-7],[10,6,7,6,-8,-1,4,3,-3,7,-9,5,-6],[2,-1,9,-2,5,-8,5,-3,1,8,-7,-7,10],[-1,-9,-9,-3,2,4,-8,-3,-9,10,-5,-9,-7],[3,-3,1,-10,-1,1,-1,-7,-7,3,9,10,-6],[-4,-9,3,4,7,-2,-6,-7,-8,-10,-3,-7,-8],[-2,8,5,-5,3,-6,-6,2,1,-10,5,-9,10],[-1,-10,-3,-9,-2,-3,9,-5,-2,-10,-8,4,-1],[3,-3,4,9,-5,8,-5,6,-7,3,10,4,4]],[[2,9,9,3,-6,8,-1,-10,-6,-1,-3,7,-2],[-3,-8,-10,7,4,-3,9,-6,-9,-9,7,-3,8],[-7,-4,-6,7,6,-6,9,10,-9,-3,4,-3,10],[10,8,-1,-7,-4,-4,8,-8,-4,4,-9,10,1],[4,9,-4,9,-9,7,-5,-6,3,10,8,-6,-7],[9,-9,-5,4,6,5,2,-6,-7,-3,10,8,-5],[6,-1,8,-4,10,-1,3,4,5,-1,6,-1,-2],[-3,1,-1,3,1,3,8,5,8,9,6,-10,2],[-7,-7,4,-1,9,-3,3,-4,9,10,3,2,1],[-3,-6,-9,10,2,-5,10,-7,8,6,8,4,-9],[-7,-9,10,-1,1,-7,-7,4,4,-8,6,2,-8],[1,-10,2,-3,8,-5,8,-1,6,6,2,5,-4],[9,-4,8,-10,-6,-6,3,2,2,-5,-7,-3,-2],[-6,8,-5,5,4,5,-2,6,1,9,-7,-3,6]],[[-3,4,7,-10,9,-1,-7,8,1,-6,-5,6,8],[6,6,-3,2,-4,-7,-3,-2,-3,-1,4,10,-9],[-4,-10,8,6,-8,3,-6,9,1,-2,-8,3,-5],[-1,6,-1,-5,-3,-6,5,5,6,-4,7,-8,-2],[3,8,-2,-2,-9,9,3,-6,-10,-7,-10,-6,-1],[-4,1,7,-1,3,-4,-4,-9,8,4,5,7,-8],[-6,-3,-9,-5,-3,6,5,-1,-4,3,-7,-5,5],[6,-1,-8,3,1,-10,-2,-9,9,8,5,5,3],[-7,7,-10,5,-9,8,6,-4,8,-9,4,2,-10],[-4,-5,8,-7,5,8,-9,-2,-1,-8,5,10,-2],[-4,6,9,6,9,7,9,-3,1,5,3,-2,-8],[-5,-10,5,-1,-5,-1,6,7,5,-5,-4,-4,-10],[3,-6,-6,1,9,-10,-9,1,-10,-1,8,4,6],[-10,4,-7,-9,-2,-7,-2,1,5,9,-1,2,-7]],[[-7,6,8,6,-5,1,-10,-2,5,1,8,-5,3],[7,7,-10,6,10,3,7,2,5,-10,9,2,3],[-10,-8,-6,10,10,10,7,3,-7,-1,10,-1,4],[-8,-1,-8,-3,4,1,5,-6,-1,-6,-7,5,-1],[6,5,-3,-10,-6,-3,-3,8,-1,-7,-9,-9,3],[7,-6,2,2,3,10,8,-2,-5,-6,-10,8,-3],[6,-10,-10,8,5,7,7,3,-5,-3,9,-1,3],[-3,9,-5,-8,5,-4,-3,10,8,-1,-5,9,-3],[-8,-5,-8,-9,4,1,3,-9,8,5,3,-5,4],[3,10,4,-4,6,-9,8,-2,-5,-6,1,-8,2],[-2,8,-9,-9,-8,-6,2,-4,-8,4,1,7,6],[1,-4,7,-2,-2,-7,1,-6,10,10,10,-4,9],[4,3,2,5,-9,-4,9,-3,10,4,2,3,7],[9,6,9,2,1,5,3,-8,6,-6,-6,4,10]],[[5,7,7,-3,4,4,-3,-1,10,-2,-3,-8,7],[5,3,7,-9,9,-10,8,4,1,8,6,8,-6],[6,1,-5,7,2,-7,-10,-6,9,-8,7,-8,3],[-10,-4,3,-10,5,1,-5,-10,3,-1,2,-4,-2],[-10,-3,-6,4,-1,10,-1,9,1,-4,-8,-10,-5],[-8,-5,-9,5,-7,-7,1,-10,-1,7,3,-4,6],[-6,-3,-1,10,-3,10,-3,9,2,7,-10,-2,-8],[3,-4,4,-7,10,7,-5,-1,8,-4,3,4,7],[-8,-1,6,10,4,7,-10,1,-9,2,1,-1,7],[-2,-3,7,10,2,3,-5,6,10,10,-7,-10,-6],[-7,10,2,6,-3,-9,-10,-8,-2,2,6,6,-5],[-5,-10,-8,10,2,9,7,1,-4,8,-6,6,8],[9,-10,-5,-9,-5,-3,-4,5,2,-5,10,6,3],[8,-7,5,-10,-10,9,-3,8,-9,10,-6,-8,9]],[[-10,5,6,-5,2,9,-5,9,10,-8,8,1,-5],[-2,9,2,6,-5,3,-7,9,6,-5,-8,7,-5],[7,-2,-4,5,8,3,7,8,9,4,1,4,5],[3,1,-8,4,1,8,1,5,2,3,1,-7,-10],[5,6,-7,5,-9,-6,-4,-2,3,8,7,-10,-10],[-10,6,-5,5,4,-2,-2,1,-5,3,7,9,3],[-3,-8,-4,2,-10,4,10,-1,-6,-3,-9,3,-4],[-9,4,-2,-7,-4,9,-2,2,-6,3,-5,10,-7],[6,2,-10,4,-3,5,-7,2,4,6,-7,8,-2],[10,-1,-1,-6,-7,-2,-4,-9,8,7,-3,1,7],[-4,9,-8,-7,5,-5,9,-9,5,9,-4,10,-4],[-4,-4,4,-8,5,-3,6,-8,-8,3,9,5,-1],[-4,2,-8,2,9,-2,-2,-2,9,2,-1,6,2],[-3,3,5,7,-5,6,-8,7,10,7,2,-2,-5]],[[7,-6,-7,3,2,-9,-7,-5,4,-8,3,8,-1],[-6,-7,5,-2,-2,-1,3,6,4,6,6,3,3],[-3,-4,3,9,-2,-5,-7,-7,5,2,10,5,-5],[10,-7,2,3,-6,9,-10,10,-7,-4,7,-4,9],[5,2,4,-8,4,-1,-7,5,2,8,1,-1,2],[4,-7,10,6,-3,-5,-3,2,-7,-1,-6,-6,-7],[2,1,5,7,-8,-3,-2,-7,9,5,-6,-10,-5],[-9,1,-7,3,-6,-3,10,4,10,-4,-3,8,2],[-2,-8,1,7,5,-6,4,-9,-6,10,4,6,-8],[-1,-6,-8,-7,5,-8,-1,-8,3,8,-9,-3,-4],[1,-5,8,-5,8,5,4,3,-2,-6,1,2,-1],[4,2,-1,10,1,-3,-2,-2,8,7,-7,-6,-9],[2,6,2,7,10,6,5,7,6,9,-8,-3,10],[4,-10,-3,-6,-6,-3,-10,-1,4,-4,4,-2,-7]]], dtype = "int16")#candidate|116|(14, 14, 13)|const|int16
var_117 = relay.var("var_117", dtype = "int16", shape = (14, 14, 13))#candidate|117|(14, 14, 13)|var|int16
bop_118 = relay.equal(const_116.astype('bool'), relay.reshape(var_117.astype('bool'), relay.shape_of(const_116))) # shape=(14, 14, 13)
func_29_call = mod.get_global_var('func_29')
func_32_call = mutated_mod.get_global_var('func_32')
const_122 = relay.const([[-2.400283,-1.251248,2.650906,8.936214,8.750775]], dtype = "float64")#candidate|122|(1, 5)|const|float64
call_121 = func_29_call(relay.reshape(const_122.astype('float64'), [5,]))
call_123 = func_29_call(relay.reshape(const_122.astype('float64'), [5,]))
func_111_call = mod.get_global_var('func_111')
func_114_call = mutated_mod.get_global_var('func_114')
var_127 = relay.var("var_127", dtype = "float64", shape = (25,))#candidate|127|(25,)|var|float64
call_126 = relay.TupleGetItem(func_111_call(relay.reshape(var_127.astype('float64'), [5, 5])), 2)
call_128 = relay.TupleGetItem(func_114_call(relay.reshape(var_127.astype('float64'), [5, 5])), 2)
uop_134 = relay.asin(const_116.astype('float64')) # shape=(14, 14, 13)
bop_137 = relay.floor_mod(uop_134.astype('float64'), relay.reshape(var_117.astype('float64'), relay.shape_of(uop_134))) # shape=(14, 14, 13)
func_80_call = mod.get_global_var('func_80')
func_83_call = mutated_mod.get_global_var('func_83')
const_141 = relay.const([[2,2,-9],[6,-10,-4],[-2,-8,7]], dtype = "uint8")#candidate|141|(3, 3)|const|uint8
call_140 = relay.TupleGetItem(func_80_call(relay.reshape(const_141.astype('uint8'), [9,])), 2)
call_142 = relay.TupleGetItem(func_83_call(relay.reshape(const_141.astype('uint8'), [9,])), 2)
output = relay.Tuple([bop_118,call_121,const_122,call_126,var_127,bop_137,call_140,const_141,])
output2 = relay.Tuple([bop_118,call_123,const_122,call_128,var_127,bop_137,call_142,const_141,])
func_144 = relay.Function([var_117,var_127,], output)
mod['func_144'] = func_144
mod = relay.transform.InferType()(mod)
var_145 = relay.var("var_145", dtype = "int16", shape = (14, 14, 13))#candidate|145|(14, 14, 13)|var|int16
var_146 = relay.var("var_146", dtype = "float64", shape = (25,))#candidate|146|(25,)|var|float64
output = func_144(var_145,var_146,)
func_147 = relay.Function([var_145,var_146,], output)
mutated_mod['func_147'] = func_147
mutated_mod = relay.transform.InferType()(mutated_mod)
const_149 = relay.const([[-7.572697,2.924968,-6.395616,8.870638,-9.275817,6.244445,-5.119335,8.934469,8.770267,-3.207254,2.482483,3.911728],[-5.770428,-4.153687,-8.392782,4.312381,-5.127206,0.940952,9.950502,-4.449179,-8.743852,6.906996,8.894952,-5.684900],[-5.074318,1.702434,-8.901416,-8.467321,5.907973,2.035415,7.076662,-0.706606,0.147957,-8.494876,-6.066461,-0.347682],[-8.840342,-2.651938,6.172652,-1.642450,2.305079,3.073482,3.574269,-9.097467,-2.826263,-3.423355,-8.100371,1.870463],[-3.513423,-4.703552,-4.975185,-7.539568,-1.282292,9.674037,7.398923,6.490402,9.521929,-1.737955,2.109589,3.441975],[8.591584,5.424206,-5.200204,8.031528,-1.650143,-6.220885,8.409593,0.756095,3.060214,5.373363,0.731443,1.739863],[9.603506,5.857517,-9.129365,-0.437364,-9.466550,7.494221,6.738390,3.981829,-8.175068,-3.591972,-3.443981,4.840228],[-1.935909,-2.103410,4.108356,9.203532,-0.719221,5.605890,6.912715,-5.803734,0.826124,7.119625,-0.941124,-3.278642],[-0.888479,-8.653797,5.269082,-0.529224,-3.970457,8.057471,-3.141381,-8.978386,2.631605,3.621890,-4.585731,-9.711795],[1.561544,-3.610106,-7.055693,-3.870011,0.423169,-3.948950,-6.763009,4.905359,0.255364,-4.096946,6.146548,-2.382989],[1.611631,4.489977,1.407069,0.281929,9.608359,4.324255,2.922174,-5.387657,-2.973802,7.389174,0.785323,-5.549957],[5.463431,-9.463693,7.773156,1.099749,-4.284902,-0.110601,-8.511458,-1.226638,6.075862,0.106744,2.293720,-3.076001],[-6.840118,-2.737018,-7.289445,1.303557,-1.142022,-4.531749,-9.320885,-2.253778,-4.623952,2.722121,-7.663497,0.318244],[6.729058,-7.060719,-3.694871,-1.117495,-0.775419,2.391552,8.426131,-7.792880,-0.011890,-1.626753,-2.845627,-7.190748]], dtype = "float64")#candidate|149|(14, 12)|const|float64
uop_150 = relay.atanh(const_149.astype('float64')) # shape=(14, 12)
output = uop_150
output2 = uop_150
func_152 = relay.Function([], output)
mod['func_152'] = func_152
mod = relay.transform.InferType()(mod)
output = func_152()
func_153 = relay.Function([], output)
mutated_mod['func_153'] = func_153
mutated_mod = relay.transform.InferType()(mutated_mod)
func_152_call = mod.get_global_var('func_152')
func_153_call = mutated_mod.get_global_var('func_153')
call_175 = func_152_call()
call_176 = func_152_call()
output = call_175
output2 = call_176
func_181 = relay.Function([], output)
mod['func_181'] = func_181
mod = relay.transform.InferType()(mod)
output = func_181()
func_182 = relay.Function([], output)
mutated_mod['func_182'] = func_182
mutated_mod = relay.transform.InferType()(mutated_mod)
var_190 = relay.var("var_190", dtype = "float32", shape = (11, 4, 1))#candidate|190|(11, 4, 1)|var|float32
uop_191 = relay.asinh(var_190.astype('float32')) # shape=(11, 4, 1)
output = uop_191
output2 = uop_191
func_193 = relay.Function([var_190,], output)
mod['func_193'] = func_193
mod = relay.transform.InferType()(mod)
mutated_mod['func_193'] = func_193
mutated_mod = relay.transform.InferType()(mutated_mod)
var_194 = relay.var("var_194", dtype = "float32", shape = (11, 4, 1))#candidate|194|(11, 4, 1)|var|float32
func_193_call = mutated_mod.get_global_var('func_193')
call_195 = func_193_call(var_194)
output = call_195
func_196 = relay.Function([var_194], output)
mutated_mod['func_196'] = func_196
mutated_mod = relay.transform.InferType()(mutated_mod)
func_152_call = mod.get_global_var('func_152')
func_153_call = mutated_mod.get_global_var('func_153')
call_235 = func_152_call()
call_236 = func_152_call()
uop_252 = relay.sigmoid(call_235.astype('float32')) # shape=(14, 12)
uop_254 = relay.sigmoid(call_236.astype('float32')) # shape=(14, 12)
uop_257 = relay.log(uop_252.astype('float64')) # shape=(14, 12)
uop_259 = relay.log(uop_254.astype('float64')) # shape=(14, 12)
uop_260 = relay.log10(uop_252.astype('float64')) # shape=(14, 12)
uop_262 = relay.log10(uop_254.astype('float64')) # shape=(14, 12)
bop_263 = relay.divide(uop_252.astype('float64'), relay.reshape(call_235.astype('float64'), relay.shape_of(uop_252))) # shape=(14, 12)
bop_266 = relay.divide(uop_254.astype('float64'), relay.reshape(call_236.astype('float64'), relay.shape_of(uop_254))) # shape=(14, 12)
bop_267 = relay.greater(uop_260.astype('bool'), relay.reshape(uop_252.astype('bool'), relay.shape_of(uop_260))) # shape=(14, 12)
bop_270 = relay.greater(uop_262.astype('bool'), relay.reshape(uop_254.astype('bool'), relay.shape_of(uop_262))) # shape=(14, 12)
uop_271 = relay.rsqrt(uop_260.astype('float32')) # shape=(14, 12)
uop_273 = relay.rsqrt(uop_262.astype('float32')) # shape=(14, 12)
bop_280 = relay.minimum(bop_263.astype('int32'), relay.reshape(uop_257.astype('int32'), relay.shape_of(bop_263))) # shape=(14, 12)
bop_283 = relay.minimum(bop_266.astype('int32'), relay.reshape(uop_259.astype('int32'), relay.shape_of(bop_266))) # shape=(14, 12)
bop_286 = relay.greater_equal(uop_271.astype('bool'), relay.reshape(uop_260.astype('bool'), relay.shape_of(uop_271))) # shape=(14, 12)
bop_289 = relay.greater_equal(uop_273.astype('bool'), relay.reshape(uop_262.astype('bool'), relay.shape_of(uop_273))) # shape=(14, 12)
uop_293 = relay.log(uop_271.astype('float32')) # shape=(14, 12)
uop_295 = relay.log(uop_273.astype('float32')) # shape=(14, 12)
var_299 = relay.var("var_299", dtype = "bool", shape = (14, 12))#candidate|299|(14, 12)|var|bool
bop_300 = relay.add(bop_286.astype('uint64'), relay.reshape(var_299.astype('uint64'), relay.shape_of(bop_286))) # shape=(14, 12)
bop_303 = relay.add(bop_289.astype('uint64'), relay.reshape(var_299.astype('uint64'), relay.shape_of(bop_289))) # shape=(14, 12)
var_306 = relay.var("var_306", dtype = "bool", shape = (14, 12))#candidate|306|(14, 12)|var|bool
bop_307 = relay.bitwise_and(bop_267.astype('int16'), relay.reshape(var_306.astype('int16'), relay.shape_of(bop_267))) # shape=(14, 12)
bop_310 = relay.bitwise_and(bop_270.astype('int16'), relay.reshape(var_306.astype('int16'), relay.shape_of(bop_270))) # shape=(14, 12)
bop_311 = relay.maximum(uop_293.astype('uint16'), relay.reshape(uop_252.astype('uint16'), relay.shape_of(uop_293))) # shape=(14, 12)
bop_314 = relay.maximum(uop_295.astype('uint16'), relay.reshape(uop_254.astype('uint16'), relay.shape_of(uop_295))) # shape=(14, 12)
bop_317 = relay.equal(bop_311.astype('bool'), relay.reshape(bop_267.astype('bool'), relay.shape_of(bop_311))) # shape=(14, 12)
bop_320 = relay.equal(bop_314.astype('bool'), relay.reshape(bop_270.astype('bool'), relay.shape_of(bop_314))) # shape=(14, 12)
var_321 = relay.var("var_321", dtype = "float32", shape = (14, 12))#candidate|321|(14, 12)|var|float32
bop_322 = relay.mod(uop_293.astype('float32'), relay.reshape(var_321.astype('float32'), relay.shape_of(uop_293))) # shape=(14, 12)
bop_325 = relay.mod(uop_295.astype('float32'), relay.reshape(var_321.astype('float32'), relay.shape_of(uop_295))) # shape=(14, 12)
uop_327 = relay.cosh(bop_322.astype('float32')) # shape=(14, 12)
uop_329 = relay.cosh(bop_325.astype('float32')) # shape=(14, 12)
bop_330 = relay.power(uop_327.astype('float64'), relay.reshape(var_321.astype('float64'), relay.shape_of(uop_327))) # shape=(14, 12)
bop_333 = relay.power(uop_329.astype('float64'), relay.reshape(var_321.astype('float64'), relay.shape_of(uop_329))) # shape=(14, 12)
const_334 = relay.const([[False,False,True,True,True,True,True,True,False,True,True,True],[False,False,True,False,True,False,True,True,False,True,True,False],[False,False,True,False,True,False,False,True,False,True,False,True],[False,False,True,True,True,False,False,False,False,True,False,True],[True,False,False,True,True,False,True,True,False,True,True,True],[True,True,False,True,True,True,True,True,True,False,False,False],[False,True,False,True,False,True,False,False,True,True,True,False],[True,True,True,True,False,True,True,True,False,True,False,True],[False,True,True,True,True,True,False,False,False,False,True,True],[True,True,True,False,False,False,False,True,True,False,True,True],[True,False,False,True,True,True,True,False,False,False,True,False],[True,True,False,False,False,True,True,True,False,False,False,True],[False,False,False,False,False,True,True,True,True,False,True,True],[True,False,True,True,False,False,True,True,False,True,True,False]], dtype = "bool")#candidate|334|(14, 12)|const|bool
bop_335 = relay.floor_mod(bop_286.astype('float32'), relay.reshape(const_334.astype('float32'), relay.shape_of(bop_286))) # shape=(14, 12)
bop_338 = relay.floor_mod(bop_289.astype('float32'), relay.reshape(const_334.astype('float32'), relay.shape_of(bop_289))) # shape=(14, 12)
uop_342 = relay.acosh(bop_300.astype('float64')) # shape=(14, 12)
uop_344 = relay.acosh(bop_303.astype('float64')) # shape=(14, 12)
func_193_call = mod.get_global_var('func_193')
func_196_call = mutated_mod.get_global_var('func_196')
const_347 = relay.const([2.074373,5.999806,-7.793753,-9.749243,-8.415750,-7.179647,-4.151939,5.662812,-5.594722,4.077792,-4.710520,-8.010283,6.960938,-6.331854,-0.909884,-8.641907,3.014448,8.989129,8.862083,-4.188340,-3.412074,-9.752180,0.528838,1.794481,-1.134587,-3.559948,-9.653809,-3.500176,-2.370221,-9.565347,-8.047751,-2.000862,-2.603683,-1.323966,-5.268759,3.606600,0.125977,3.096383,4.172884,-3.197113,5.982925,-9.815885,5.266682,3.217621], dtype = "float32")#candidate|347|(44,)|const|float32
call_346 = func_193_call(relay.reshape(const_347.astype('float32'), [11, 4, 1]))
call_348 = func_193_call(relay.reshape(const_347.astype('float32'), [11, 4, 1]))
var_351 = relay.var("var_351", dtype = "float32", shape = (14, 12))#candidate|351|(14, 12)|var|float32
bop_352 = relay.less_equal(uop_293.astype('bool'), relay.reshape(var_351.astype('bool'), relay.shape_of(uop_293))) # shape=(14, 12)
bop_355 = relay.less_equal(uop_295.astype('bool'), relay.reshape(var_351.astype('bool'), relay.shape_of(uop_295))) # shape=(14, 12)
bop_356 = relay.floor_divide(bop_330.astype('float32'), relay.reshape(bop_286.astype('float32'), relay.shape_of(bop_330))) # shape=(14, 12)
bop_359 = relay.floor_divide(bop_333.astype('float32'), relay.reshape(bop_289.astype('float32'), relay.shape_of(bop_333))) # shape=(14, 12)
var_363 = relay.var("var_363", dtype = "float64", shape = (14, 12))#candidate|363|(14, 12)|var|float64
bop_364 = relay.logical_xor(bop_330.astype('int16'), relay.reshape(var_363.astype('int16'), relay.shape_of(bop_330))) # shape=(14, 12)
bop_367 = relay.logical_xor(bop_333.astype('int16'), relay.reshape(var_363.astype('int16'), relay.shape_of(bop_333))) # shape=(14, 12)
bop_370 = relay.less(bop_364.astype('bool'), relay.reshape(bop_311.astype('bool'), relay.shape_of(bop_364))) # shape=(14, 12)
bop_373 = relay.less(bop_367.astype('bool'), relay.reshape(bop_314.astype('bool'), relay.shape_of(bop_367))) # shape=(14, 12)
output = relay.Tuple([bop_280,bop_307,bop_317,bop_335,uop_342,call_346,const_347,bop_352,bop_356,bop_370,])
output2 = relay.Tuple([bop_283,bop_310,bop_320,bop_338,uop_344,call_348,const_347,bop_355,bop_359,bop_373,])
func_381 = relay.Function([var_299,var_306,var_321,var_351,var_363,], output)
mod['func_381'] = func_381
mod = relay.transform.InferType()(mod)
var_382 = relay.var("var_382", dtype = "bool", shape = (14, 12))#candidate|382|(14, 12)|var|bool
var_383 = relay.var("var_383", dtype = "bool", shape = (14, 12))#candidate|383|(14, 12)|var|bool
var_384 = relay.var("var_384", dtype = "float32", shape = (14, 12))#candidate|384|(14, 12)|var|float32
var_385 = relay.var("var_385", dtype = "float32", shape = (14, 12))#candidate|385|(14, 12)|var|float32
var_386 = relay.var("var_386", dtype = "float64", shape = (14, 12))#candidate|386|(14, 12)|var|float64
output = func_381(var_382,var_383,var_384,var_385,var_386,)
func_387 = relay.Function([var_382,var_383,var_384,var_385,var_386,], output)
mutated_mod['func_387'] = func_387
mutated_mod = relay.transform.InferType()(mutated_mod)
var_397 = relay.var("var_397", dtype = "float32", shape = ())#candidate|397|()|var|float32
var_398 = relay.var("var_398", dtype = "float32", shape = (6,))#candidate|398|(6,)|var|float32
bop_399 = relay.power(var_397.astype('float32'), var_398.astype('float32')) # shape=(6,)
output = relay.Tuple([bop_399,])
output2 = relay.Tuple([bop_399,])
func_402 = relay.Function([var_397,var_398,], output)
mod['func_402'] = func_402
mod = relay.transform.InferType()(mod)
mutated_mod['func_402'] = func_402
mutated_mod = relay.transform.InferType()(mutated_mod)
func_402_call = mutated_mod.get_global_var('func_402')
var_404 = relay.var("var_404", dtype = "float32", shape = ())#candidate|404|()|var|float32
var_405 = relay.var("var_405", dtype = "float32", shape = (6,))#candidate|405|(6,)|var|float32
call_403 = func_402_call(var_404,var_405,)
output = call_403
func_406 = relay.Function([var_404,var_405,], output)
mutated_mod['func_406'] = func_406
mutated_mod = relay.transform.InferType()(mutated_mod)
func_152_call = mod.get_global_var('func_152')
func_153_call = mutated_mod.get_global_var('func_153')
call_411 = func_152_call()
call_412 = func_152_call()
output = call_411
output2 = call_412
func_424 = relay.Function([], output)
mod['func_424'] = func_424
mod = relay.transform.InferType()(mod)
output = func_424()
func_425 = relay.Function([], output)
mutated_mod['func_425'] = func_425
mutated_mod = relay.transform.InferType()(mutated_mod)
const_429 = relay.const([[-5.935191,-5.616549,-7.245698,-9.336414,-6.877370,3.010959,-3.510246,6.070278,-2.576804,2.434577,-7.980372,-9.000357,-1.389882,8.498714,4.376358,-4.058523],[-0.189146,1.332763,-4.281136,-4.576777,5.541697,1.236470,9.939803,-3.410101,2.544249,0.218830,9.467604,4.725132,-4.316685,-3.976392,4.764546,8.191054],[-1.431214,9.161871,5.355258,2.841809,7.116867,-4.971359,-8.716557,-5.681760,-5.055469,0.503368,0.494684,-2.026836,-3.980206,-4.063925,0.723280,8.745323],[-2.965762,-9.175952,3.472058,-4.079033,3.655462,2.619364,5.708927,2.494622,7.945260,-5.257752,1.214429,1.803085,0.670139,0.511833,2.549757,-3.720751],[2.128729,5.004666,9.050667,-5.734884,-1.804232,-3.659494,4.017765,-9.206340,-0.643480,-3.569849,7.131936,9.951071,0.999482,5.982157,-6.208361,-8.142467],[9.925631,-8.592734,4.103693,8.169053,-2.347735,2.700428,2.937286,4.925361,-1.839990,2.807597,1.067519,4.972329,5.420187,6.136883,-9.676188,-0.443907],[-4.063926,-1.536309,0.175352,1.201836,2.013447,6.051461,-1.826018,8.040871,-0.225055,-1.780471,5.992336,-7.428657,8.943378,-2.066621,2.980053,0.006299],[7.448520,4.386620,6.429425,9.019135,-4.302485,6.889255,9.429613,5.477480,7.869308,4.344895,1.769097,6.962361,3.921069,3.169443,-3.294779,-7.410961],[-1.605913,-8.282757,7.870429,5.923998,0.753910,-0.087124,0.205488,-7.665622,-5.394324,1.374172,-1.658268,-1.553070,1.208936,-1.613043,-9.798493,5.226582],[2.120885,-5.787843,1.137594,-8.243080,-9.750942,2.645615,-9.048043,-5.103267,2.838192,6.973858,-4.761886,-3.554289,6.263977,-0.558025,5.228877,-1.801452]], dtype = "float32")#candidate|429|(10, 16)|const|float32
var_430 = relay.var("var_430", dtype = "float32", shape = (10, 16))#candidate|430|(10, 16)|var|float32
bop_431 = relay.floor_mod(const_429.astype('float32'), relay.reshape(var_430.astype('float32'), relay.shape_of(const_429))) # shape=(10, 16)
func_381_call = mod.get_global_var('func_381')
func_387_call = mutated_mod.get_global_var('func_387')
const_436 = relay.const([True,False,False,False,True,True,True,True,True,False,False,True,True,True,True,False,True,False,True,False,False,True,False,True,True,True,True,False,True,True,True,False,False,True,False,True,True,True,False,False,True,False,True,False,True,False,True,True,False,False,True,False,True,True,False,True,False,True,True,True,True,False,True,True,False,True,False,True,False,False,True,True,True,False,True,False,False,False,True,False,False,False,True,False,True,True,True,True,False,False,False,True,True,True,False,True,False,False,False,False,True,True,True,False,True,True,False,True,True,False,True,False,False,False,False,False,True,True,True,False,True,False,True,False,True,False,True,False,False,True,False,True,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,True,False,False,True,False,True,True,True,True,True,True,False,False,False,True,True,False,True,False,True], dtype = "bool")#candidate|436|(168,)|const|bool
call_435 = relay.TupleGetItem(func_381_call(relay.reshape(const_436.astype('bool'), [14, 12]), relay.reshape(const_436.astype('bool'), [14, 12]), relay.reshape(const_436.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float64'), [14, 12]), ), 2)
call_437 = relay.TupleGetItem(func_387_call(relay.reshape(const_436.astype('bool'), [14, 12]), relay.reshape(const_436.astype('bool'), [14, 12]), relay.reshape(const_436.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float64'), [14, 12]), ), 2)
uop_438 = relay.tan(const_436.astype('float64')) # shape=(168,)
bop_446 = relay.equal(uop_438.astype('bool'), relay.reshape(call_435.astype('bool'), relay.shape_of(uop_438))) # shape=(168,)
bop_449 = relay.equal(uop_438.astype('bool'), relay.reshape(call_437.astype('bool'), relay.shape_of(uop_438))) # shape=(168,)
uop_450 = relay.sin(uop_438.astype('float32')) # shape=(168,)
uop_455 = relay.atanh(uop_450.astype('float64')) # shape=(168,)
func_381_call = mod.get_global_var('func_381')
func_387_call = mutated_mod.get_global_var('func_387')
call_460 = relay.TupleGetItem(func_381_call(relay.reshape(uop_455.astype('bool'), [14, 12]), relay.reshape(bop_446.astype('bool'), [14, 12]), relay.reshape(call_435.astype('float32'), [14, 12]), relay.reshape(bop_446.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float64'), [14, 12]), ), 0)
call_461 = relay.TupleGetItem(func_387_call(relay.reshape(uop_455.astype('bool'), [14, 12]), relay.reshape(bop_446.astype('bool'), [14, 12]), relay.reshape(call_435.astype('float32'), [14, 12]), relay.reshape(bop_446.astype('float32'), [14, 12]), relay.reshape(const_436.astype('float64'), [14, 12]), ), 0)
func_80_call = mod.get_global_var('func_80')
func_83_call = mutated_mod.get_global_var('func_83')
var_463 = relay.var("var_463", dtype = "uint8", shape = (9,))#candidate|463|(9,)|var|uint8
call_462 = relay.TupleGetItem(func_80_call(relay.reshape(var_463.astype('uint8'), [9,])), 2)
call_464 = relay.TupleGetItem(func_83_call(relay.reshape(var_463.astype('uint8'), [9,])), 2)
uop_467 = relay.acosh(bop_446.astype('float64')) # shape=(168,)
uop_469 = relay.acosh(bop_449.astype('float64')) # shape=(168,)
func_29_call = mod.get_global_var('func_29')
func_32_call = mutated_mod.get_global_var('func_32')
const_471 = relay.const([[-8.917384],[0.470988],[-9.122036],[2.950149],[-0.992668]], dtype = "float64")#candidate|471|(5, 1)|const|float64
call_470 = func_29_call(relay.reshape(const_471.astype('float64'), [5,]))
call_472 = func_29_call(relay.reshape(const_471.astype('float64'), [5,]))
bop_473 = relay.mod(uop_450.astype('float32'), relay.reshape(bop_446.astype('float32'), relay.shape_of(uop_450))) # shape=(168,)
bop_476 = relay.mod(uop_450.astype('float32'), relay.reshape(bop_449.astype('float32'), relay.shape_of(uop_450))) # shape=(168,)
bop_477 = relay.power(uop_455.astype('float64'), relay.reshape(call_460.astype('float64'), relay.shape_of(uop_455))) # shape=(168,)
bop_480 = relay.power(uop_455.astype('float64'), relay.reshape(call_461.astype('float64'), relay.shape_of(uop_455))) # shape=(168,)
bop_481 = relay.less(bop_473.astype('bool'), relay.reshape(bop_477.astype('bool'), relay.shape_of(bop_473))) # shape=(168,)
bop_484 = relay.less(bop_476.astype('bool'), relay.reshape(bop_480.astype('bool'), relay.shape_of(bop_476))) # shape=(168,)
bop_485 = relay.logical_or(uop_467.astype('bool'), relay.reshape(const_436.astype('bool'), relay.shape_of(uop_467))) # shape=(168,)
bop_488 = relay.logical_or(uop_469.astype('bool'), relay.reshape(const_436.astype('bool'), relay.shape_of(uop_469))) # shape=(168,)
uop_491 = relay.sqrt(bop_446.astype('float64')) # shape=(168,)
uop_493 = relay.sqrt(bop_449.astype('float64')) # shape=(168,)
bop_494 = relay.minimum(bop_477.astype('int32'), relay.reshape(bop_473.astype('int32'), relay.shape_of(bop_477))) # shape=(168,)
bop_497 = relay.minimum(bop_480.astype('int32'), relay.reshape(bop_476.astype('int32'), relay.shape_of(bop_480))) # shape=(168,)
bop_498 = relay.not_equal(bop_446.astype('bool'), relay.reshape(uop_455.astype('bool'), relay.shape_of(bop_446))) # shape=(168,)
bop_501 = relay.not_equal(bop_449.astype('bool'), relay.reshape(uop_455.astype('bool'), relay.shape_of(bop_449))) # shape=(168,)
bop_503 = relay.floor_divide(uop_455.astype('float64'), relay.reshape(bop_481.astype('float64'), relay.shape_of(uop_455))) # shape=(168,)
bop_506 = relay.floor_divide(uop_455.astype('float64'), relay.reshape(bop_484.astype('float64'), relay.shape_of(uop_455))) # shape=(168,)
func_402_call = mod.get_global_var('func_402')
func_406_call = mutated_mod.get_global_var('func_406')
const_509 = relay.const(3.037703, dtype = "float32")#candidate|509|()|const|float32
var_510 = relay.var("var_510", dtype = "float32", shape = (6,))#candidate|510|(6,)|var|float32
call_508 = relay.TupleGetItem(func_402_call(relay.reshape(const_509.astype('float32'), []), relay.reshape(var_510.astype('float32'), [6,]), ), 0)
call_511 = relay.TupleGetItem(func_406_call(relay.reshape(const_509.astype('float32'), []), relay.reshape(var_510.astype('float32'), [6,]), ), 0)
bop_513 = relay.bitwise_or(bop_481.astype('int8'), const_471.astype('int8')) # shape=(5, 168)
bop_516 = relay.bitwise_or(bop_484.astype('int8'), const_471.astype('int8')) # shape=(5, 168)
uop_518 = relay.atan(uop_491.astype('float32')) # shape=(168,)
uop_520 = relay.atan(uop_493.astype('float32')) # shape=(168,)
bop_522 = relay.left_shift(uop_455.astype('uint64'), relay.reshape(bop_473.astype('uint64'), relay.shape_of(uop_455))) # shape=(168,)
bop_525 = relay.left_shift(uop_455.astype('uint64'), relay.reshape(bop_476.astype('uint64'), relay.shape_of(uop_455))) # shape=(168,)
uop_529 = relay.log2(uop_491.astype('float32')) # shape=(168,)
uop_531 = relay.log2(uop_493.astype('float32')) # shape=(168,)
bop_534 = relay.bitwise_or(uop_438.astype('uint64'), relay.reshape(call_435.astype('uint64'), relay.shape_of(uop_438))) # shape=(168,)
bop_537 = relay.bitwise_or(uop_438.astype('uint64'), relay.reshape(call_437.astype('uint64'), relay.shape_of(uop_438))) # shape=(168,)
output = relay.Tuple([bop_431,call_462,var_463,call_470,bop_485,bop_494,bop_498,bop_503,call_508,const_509,var_510,bop_513,uop_518,bop_522,uop_529,bop_534,])
output2 = relay.Tuple([bop_431,call_464,var_463,call_472,bop_488,bop_497,bop_501,bop_506,call_511,const_509,var_510,bop_516,uop_520,bop_525,uop_531,bop_537,])
func_539 = relay.Function([var_430,var_463,var_510,], output)
mod['func_539'] = func_539
mod = relay.transform.InferType()(mod)
mutated_mod['func_539'] = func_539
mutated_mod = relay.transform.InferType()(mutated_mod)
func_539_call = mutated_mod.get_global_var('func_539')
var_541 = relay.var("var_541", dtype = "float32", shape = (10, 16))#candidate|541|(10, 16)|var|float32
var_542 = relay.var("var_542", dtype = "uint8", shape = (9,))#candidate|542|(9,)|var|uint8
var_543 = relay.var("var_543", dtype = "float32", shape = (6,))#candidate|543|(6,)|var|float32
call_540 = func_539_call(var_541,var_542,var_543,)
output = call_540
func_544 = relay.Function([var_541,var_542,var_543,], output)
mutated_mod['func_544'] = func_544
mutated_mod = relay.transform.InferType()(mutated_mod)
func_152_call = mod.get_global_var('func_152')
func_153_call = mutated_mod.get_global_var('func_153')
call_548 = func_152_call()
call_549 = func_152_call()
output = relay.Tuple([call_548,])
output2 = relay.Tuple([call_549,])
func_592 = relay.Function([], output)
mod['func_592'] = func_592
mod = relay.transform.InferType()(mod)
output = func_592()
func_593 = relay.Function([], output)
mutated_mod['func_593'] = func_593
mutated_mod = relay.transform.InferType()(mutated_mod)
func_152_call = mod.get_global_var('func_152')
func_153_call = mutated_mod.get_global_var('func_153')
call_594 = func_152_call()
call_595 = func_152_call()
output = relay.Tuple([call_594,])
output2 = relay.Tuple([call_595,])
func_602 = relay.Function([], output)
mod['func_602'] = func_602
mod = relay.transform.InferType()(mod)
mutated_mod['func_602'] = func_602
mutated_mod = relay.transform.InferType()(mutated_mod)
func_602_call = mutated_mod.get_global_var('func_602')
call_603 = func_602_call()
output = call_603
func_604 = relay.Function([], output)
mutated_mod['func_604'] = func_604
mutated_mod = relay.transform.InferType()(mutated_mod)
func_181_call = mod.get_global_var('func_181')
func_182_call = mutated_mod.get_global_var('func_182')
call_608 = func_181_call()
call_609 = func_181_call()
func_29_call = mod.get_global_var('func_29')
func_32_call = mutated_mod.get_global_var('func_32')
const_621 = relay.const([[-2.581521],[7.646254],[-1.820187],[-7.925406],[-9.750368]], dtype = "float64")#candidate|621|(5, 1)|const|float64
call_620 = func_29_call(relay.reshape(const_621.astype('float64'), [5,]))
call_622 = func_29_call(relay.reshape(const_621.astype('float64'), [5,]))
uop_625 = relay.asinh(call_620.astype('float64')) # shape=(5,)
uop_627 = relay.asinh(call_622.astype('float64')) # shape=(5,)
uop_628 = relay.cos(call_620.astype('float64')) # shape=(5,)
uop_630 = relay.cos(call_622.astype('float64')) # shape=(5,)
bop_638 = relay.subtract(uop_628.astype('uint64'), relay.reshape(uop_625.astype('uint64'), relay.shape_of(uop_628))) # shape=(5,)
bop_641 = relay.subtract(uop_630.astype('uint64'), relay.reshape(uop_627.astype('uint64'), relay.shape_of(uop_630))) # shape=(5,)
output = relay.Tuple([call_608,const_621,bop_638,])
output2 = relay.Tuple([call_609,const_621,bop_641,])
func_644 = relay.Function([], output)
mod['func_644'] = func_644
mod = relay.transform.InferType()(mod)
mutated_mod['func_644'] = func_644
mutated_mod = relay.transform.InferType()(mutated_mod)
func_644_call = mutated_mod.get_global_var('func_644')
call_645 = func_644_call()
output = call_645
func_646 = relay.Function([], output)
mutated_mod['func_646'] = func_646
mutated_mod = relay.transform.InferType()(mutated_mod)
var_696 = relay.var("var_696", dtype = "int32", shape = (14,))#candidate|696|(14,)|var|int32
var_697 = relay.var("var_697", dtype = "int32", shape = (14,))#candidate|697|(14,)|var|int32
bop_698 = relay.bitwise_xor(var_696.astype('int32'), relay.reshape(var_697.astype('int32'), relay.shape_of(var_696))) # shape=(14,)
output = bop_698
output2 = bop_698
func_702 = relay.Function([var_696,var_697,], output)
mod['func_702'] = func_702
mod = relay.transform.InferType()(mod)
mutated_mod['func_702'] = func_702
mutated_mod = relay.transform.InferType()(mutated_mod)
func_702_call = mutated_mod.get_global_var('func_702')
var_704 = relay.var("var_704", dtype = "int32", shape = (14,))#candidate|704|(14,)|var|int32
var_705 = relay.var("var_705", dtype = "int32", shape = (14,))#candidate|705|(14,)|var|int32
call_703 = func_702_call(var_704,var_705,)
output = call_703
func_706 = relay.Function([var_704,var_705,], output)
mutated_mod['func_706'] = func_706
mutated_mod = relay.transform.InferType()(mutated_mod)
func_644_call = mod.get_global_var('func_644')
func_646_call = mutated_mod.get_global_var('func_646')
call_712 = relay.TupleGetItem(func_644_call(), 1)
call_713 = relay.TupleGetItem(func_646_call(), 1)
output = relay.Tuple([call_712,])
output2 = relay.Tuple([call_713,])
func_727 = relay.Function([], output)
mod['func_727'] = func_727
mod = relay.transform.InferType()(mod)
output = func_727()
func_728 = relay.Function([], output)
mutated_mod['func_728'] = func_728
mutated_mod = relay.transform.InferType()(mutated_mod)
func_424_call = mod.get_global_var('func_424')
func_425_call = mutated_mod.get_global_var('func_425')
call_741 = func_424_call()
call_742 = func_424_call()
output = call_741
output2 = call_742
func_748 = relay.Function([], output)
mod['func_748'] = func_748
mod = relay.transform.InferType()(mod)
output = func_748()
func_749 = relay.Function([], output)
mutated_mod['func_749'] = func_749
mutated_mod = relay.transform.InferType()(mutated_mod)
var_802 = relay.var("var_802", dtype = "float64", shape = ())#candidate|802|()|var|float64
var_803 = relay.var("var_803", dtype = "float64", shape = (13, 11))#candidate|803|(13, 11)|var|float64
bop_804 = relay.floor_divide(var_802.astype('float64'), var_803.astype('float64')) # shape=(13, 11)
uop_810 = relay.asin(bop_804.astype('float32')) # shape=(13, 11)
var_812 = relay.var("var_812", dtype = "float32", shape = (13, 11))#candidate|812|(13, 11)|var|float32
bop_813 = relay.bitwise_or(uop_810.astype('uint64'), relay.reshape(var_812.astype('uint64'), relay.shape_of(uop_810))) # shape=(13, 11)
bop_816 = relay.maximum(bop_813.astype('uint16'), relay.reshape(uop_810.astype('uint16'), relay.shape_of(bop_813))) # shape=(13, 11)
bop_820 = relay.divide(var_802.astype('float64'), bop_813.astype('float64')) # shape=(13, 11)
bop_826 = relay.not_equal(bop_820.astype('bool'), var_802.astype('bool')) # shape=(13, 11)
output = relay.Tuple([bop_816,bop_826,])
output2 = relay.Tuple([bop_816,bop_826,])
func_829 = relay.Function([var_802,var_803,var_812,], output)
mod['func_829'] = func_829
mod = relay.transform.InferType()(mod)
var_830 = relay.var("var_830", dtype = "float64", shape = ())#candidate|830|()|var|float64
var_831 = relay.var("var_831", dtype = "float64", shape = (13, 11))#candidate|831|(13, 11)|var|float64
var_832 = relay.var("var_832", dtype = "float32", shape = (13, 11))#candidate|832|(13, 11)|var|float32
output = func_829(var_830,var_831,var_832,)
func_833 = relay.Function([var_830,var_831,var_832,], output)
mutated_mod['func_833'] = func_833
mutated_mod = relay.transform.InferType()(mutated_mod)
var_849 = relay.var("var_849", dtype = "float32", shape = (5, 6))#candidate|849|(5, 6)|var|float32
uop_850 = relay.atan(var_849.astype('float32')) # shape=(5, 6)
uop_853 = relay.rsqrt(uop_850.astype('float64')) # shape=(5, 6)
bop_858 = relay.subtract(uop_853.astype('int16'), relay.reshape(uop_850.astype('int16'), relay.shape_of(uop_853))) # shape=(5, 6)
bop_861 = relay.greater_equal(var_849.astype('bool'), relay.reshape(uop_850.astype('bool'), relay.shape_of(var_849))) # shape=(5, 6)
uop_864 = relay.sinh(uop_853.astype('float32')) # shape=(5, 6)
const_867 = relay.const([[2.295913,-9.645484,5.027218,8.095745,-7.587655,3.975414],[-7.523683,4.961764,-9.822512,-2.271650,5.136076,6.910819],[8.676246,-5.943015,6.776863,9.092967,-6.256047,-0.525908],[1.330595,-4.768012,2.701838,-5.091345,1.640449,-8.700201],[9.098614,6.749660,3.647216,9.563353,-6.467879,-9.903164]], dtype = "float32")#candidate|867|(5, 6)|const|float32
bop_868 = relay.equal(uop_864.astype('bool'), relay.reshape(const_867.astype('bool'), relay.shape_of(uop_864))) # shape=(5, 6)
uop_871 = relay.cos(bop_868.astype('float64')) # shape=(5, 6)
bop_873 = relay.less_equal(bop_868.astype('bool'), relay.reshape(uop_853.astype('bool'), relay.shape_of(bop_868))) # shape=(5, 6)
uop_876 = relay.asin(uop_871.astype('float32')) # shape=(5, 6)
uop_882 = relay.sqrt(uop_876.astype('float32')) # shape=(5, 6)
uop_884 = relay.sqrt(uop_882.astype('float64')) # shape=(5, 6)
output = relay.Tuple([bop_858,bop_861,bop_873,uop_884,])
output2 = relay.Tuple([bop_858,bop_861,bop_873,uop_884,])
func_891 = relay.Function([var_849,], output)
mod['func_891'] = func_891
mod = relay.transform.InferType()(mod)
mutated_mod['func_891'] = func_891
mutated_mod = relay.transform.InferType()(mutated_mod)
var_892 = relay.var("var_892", dtype = "float32", shape = (5, 6))#candidate|892|(5, 6)|var|float32
func_891_call = mutated_mod.get_global_var('func_891')
call_893 = func_891_call(var_892)
output = call_893
func_894 = relay.Function([var_892], output)
mutated_mod['func_894'] = func_894
mutated_mod = relay.transform.InferType()(mutated_mod)
func_424_call = mod.get_global_var('func_424')
func_425_call = mutated_mod.get_global_var('func_425')
call_928 = func_424_call()
call_929 = func_424_call()
output = relay.Tuple([call_928,])
output2 = relay.Tuple([call_929,])
func_940 = relay.Function([], output)
mod['func_940'] = func_940
mod = relay.transform.InferType()(mod)
mutated_mod['func_940'] = func_940
mutated_mod = relay.transform.InferType()(mutated_mod)
func_940_call = mutated_mod.get_global_var('func_940')
call_941 = func_940_call()
output = call_941
func_942 = relay.Function([], output)
mutated_mod['func_942'] = func_942
mutated_mod = relay.transform.InferType()(mutated_mod)
func_181_call = mod.get_global_var('func_181')
func_182_call = mutated_mod.get_global_var('func_182')
call_957 = func_181_call()
call_958 = func_181_call()
output = call_957
output2 = call_958
func_960 = relay.Function([], output)
mod['func_960'] = func_960
mod = relay.transform.InferType()(mod)
mutated_mod['func_960'] = func_960
mutated_mod = relay.transform.InferType()(mutated_mod)
func_960_call = mutated_mod.get_global_var('func_960')
call_961 = func_960_call()
output = call_961
func_962 = relay.Function([], output)
mutated_mod['func_962'] = func_962
mutated_mod = relay.transform.InferType()(mutated_mod)
func_940_call = mod.get_global_var('func_940')
func_942_call = mutated_mod.get_global_var('func_942')
call_963 = relay.TupleGetItem(func_940_call(), 0)
call_964 = relay.TupleGetItem(func_942_call(), 0)
uop_973 = relay.cos(call_963.astype('float64')) # shape=(14, 12)
uop_975 = relay.cos(call_964.astype('float64')) # shape=(14, 12)
uop_977 = relay.asinh(uop_973.astype('float64')) # shape=(14, 12)
uop_979 = relay.asinh(uop_975.astype('float64')) # shape=(14, 12)
output = uop_977
output2 = uop_979
func_983 = relay.Function([], output)
mod['func_983'] = func_983
mod = relay.transform.InferType()(mod)
mutated_mod['func_983'] = func_983
mutated_mod = relay.transform.InferType()(mutated_mod)
func_983_call = mutated_mod.get_global_var('func_983')
call_984 = func_983_call()
output = call_984
func_985 = relay.Function([], output)
mutated_mod['func_985'] = func_985
mutated_mod = relay.transform.InferType()(mutated_mod)
var_994 = relay.var("var_994", dtype = "float32", shape = (9,))#candidate|994|(9,)|var|float32
uop_995 = relay.log2(var_994.astype('float32')) # shape=(9,)
func_960_call = mod.get_global_var('func_960')
func_962_call = mutated_mod.get_global_var('func_962')
call_997 = func_960_call()
call_998 = func_960_call()
func_983_call = mod.get_global_var('func_983')
func_985_call = mutated_mod.get_global_var('func_985')
call_1002 = func_983_call()
call_1003 = func_983_call()
uop_1015 = relay.log10(uop_995.astype('float32')) # shape=(9,)
uop_1018 = relay.sinh(uop_1015.astype('float32')) # shape=(9,)
bop_1020 = relay.less(uop_1015.astype('bool'), relay.reshape(uop_1018.astype('bool'), relay.shape_of(uop_1015))) # shape=(9,)
output = relay.Tuple([call_997,call_1002,bop_1020,])
output2 = relay.Tuple([call_998,call_1003,bop_1020,])
func_1025 = relay.Function([var_994,], output)
mod['func_1025'] = func_1025
mod = relay.transform.InferType()(mod)
mutated_mod['func_1025'] = func_1025
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1026 = relay.var("var_1026", dtype = "float32", shape = (9,))#candidate|1026|(9,)|var|float32
func_1025_call = mutated_mod.get_global_var('func_1025')
call_1027 = func_1025_call(var_1026)
output = call_1027
func_1028 = relay.Function([var_1026], output)
mutated_mod['func_1028'] = func_1028
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1036 = relay.var("var_1036", dtype = "float64", shape = (7, 11))#candidate|1036|(7, 11)|var|float64
uop_1037 = relay.acos(var_1036.astype('float64')) # shape=(7, 11)
bop_1040 = relay.less_equal(var_1036.astype('bool'), relay.reshape(uop_1037.astype('bool'), relay.shape_of(var_1036))) # shape=(7, 11)
bop_1044 = relay.floor_mod(var_1036.astype('float64'), relay.reshape(bop_1040.astype('float64'), relay.shape_of(var_1036))) # shape=(7, 11)
bop_1048 = relay.left_shift(bop_1040.astype('int32'), relay.reshape(var_1036.astype('int32'), relay.shape_of(bop_1040))) # shape=(7, 11)
const_1051 = relay.const([[-6.835568,-6.928876,3.775794,8.439783,-7.604204,-7.128909,2.733858,5.489406,8.262305,-2.481122,-4.409813],[-3.197005,-5.456850,-7.916585,7.697005,8.835682,7.782584,-9.932029,-3.731844,3.285120,1.941028,9.182149],[0.449799,-4.327257,-9.270791,1.811419,-3.617462,-7.033822,3.104865,-1.236386,-8.597259,7.284143,2.306523],[3.614546,7.648451,0.791542,-1.349323,-2.862241,9.923523,-0.010738,5.309660,-1.989117,2.556622,-5.585044],[6.781288,3.769800,3.955526,-6.697449,-6.492057,-7.522899,-0.804814,-8.498581,-4.448117,-4.429441,-6.209718],[6.435507,-9.000614,1.516736,9.799490,3.899088,9.791895,-0.356767,9.401342,1.097240,-3.249843,0.473456],[9.200636,0.779862,-3.026814,0.905717,-1.416954,-0.332395,1.294124,-7.081480,-7.553447,2.382450,-5.707134]], dtype = "float64")#candidate|1051|(7, 11)|const|float64
bop_1052 = relay.subtract(bop_1044.astype('int64'), relay.reshape(const_1051.astype('int64'), relay.shape_of(bop_1044))) # shape=(7, 11)
output = relay.Tuple([bop_1048,bop_1052,])
output2 = relay.Tuple([bop_1048,bop_1052,])
func_1060 = relay.Function([var_1036,], output)
mod['func_1060'] = func_1060
mod = relay.transform.InferType()(mod)
var_1061 = relay.var("var_1061", dtype = "float64", shape = (7, 11))#candidate|1061|(7, 11)|var|float64
output = func_1060(var_1061)
func_1062 = relay.Function([var_1061], output)
mutated_mod['func_1062'] = func_1062
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1066 = relay.var("var_1066", dtype = "float64", shape = (5,))#candidate|1066|(5,)|var|float64
var_1067 = relay.var("var_1067", dtype = "float64", shape = (5,))#candidate|1067|(5,)|var|float64
bop_1068 = relay.power(var_1066.astype('float64'), relay.reshape(var_1067.astype('float64'), relay.shape_of(var_1066))) # shape=(5,)
bop_1072 = relay.mod(bop_1068.astype('float64'), relay.reshape(var_1066.astype('float64'), relay.shape_of(bop_1068))) # shape=(5,)
output = relay.Tuple([bop_1072,])
output2 = relay.Tuple([bop_1072,])
func_1080 = relay.Function([var_1066,var_1067,], output)
mod['func_1080'] = func_1080
mod = relay.transform.InferType()(mod)
var_1081 = relay.var("var_1081", dtype = "float64", shape = (5,))#candidate|1081|(5,)|var|float64
var_1082 = relay.var("var_1082", dtype = "float64", shape = (5,))#candidate|1082|(5,)|var|float64
output = func_1080(var_1081,var_1082,)
func_1083 = relay.Function([var_1081,var_1082,], output)
mutated_mod['func_1083'] = func_1083
mutated_mod = relay.transform.InferType()(mutated_mod)
func_602_call = mod.get_global_var('func_602')
func_604_call = mutated_mod.get_global_var('func_604')
call_1094 = relay.TupleGetItem(func_602_call(), 0)
call_1095 = relay.TupleGetItem(func_604_call(), 0)
func_1060_call = mod.get_global_var('func_1060')
func_1062_call = mutated_mod.get_global_var('func_1062')
var_1099 = relay.var("var_1099", dtype = "float64", shape = (77,))#candidate|1099|(77,)|var|float64
call_1098 = relay.TupleGetItem(func_1060_call(relay.reshape(var_1099.astype('float64'), [7, 11])), 1)
call_1100 = relay.TupleGetItem(func_1062_call(relay.reshape(var_1099.astype('float64'), [7, 11])), 1)
output = relay.Tuple([call_1094,call_1098,var_1099,])
output2 = relay.Tuple([call_1095,call_1100,var_1099,])
func_1102 = relay.Function([var_1099,], output)
mod['func_1102'] = func_1102
mod = relay.transform.InferType()(mod)
mutated_mod['func_1102'] = func_1102
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1103 = relay.var("var_1103", dtype = "float64", shape = (77,))#candidate|1103|(77,)|var|float64
func_1102_call = mutated_mod.get_global_var('func_1102')
call_1104 = func_1102_call(var_1103)
output = call_1104
func_1105 = relay.Function([var_1103], output)
mutated_mod['func_1105'] = func_1105
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1109 = relay.var("var_1109", dtype = "float64", shape = (4, 12, 7))#candidate|1109|(4, 12, 7)|var|float64
var_1110 = relay.var("var_1110", dtype = "float64", shape = (4, 12, 7))#candidate|1110|(4, 12, 7)|var|float64
bop_1111 = relay.floor_mod(var_1109.astype('float64'), relay.reshape(var_1110.astype('float64'), relay.shape_of(var_1109))) # shape=(4, 12, 7)
bop_1115 = relay.greater(var_1110.astype('bool'), relay.reshape(bop_1111.astype('bool'), relay.shape_of(var_1110))) # shape=(4, 12, 7)
uop_1118 = relay.log2(var_1110.astype('float32')) # shape=(4, 12, 7)
var_1120 = relay.var("var_1120", dtype = "float32", shape = (4, 12, 7))#candidate|1120|(4, 12, 7)|var|float32
bop_1121 = relay.logical_xor(uop_1118.astype('int16'), relay.reshape(var_1120.astype('int16'), relay.shape_of(uop_1118))) # shape=(4, 12, 7)
func_1080_call = mod.get_global_var('func_1080')
func_1083_call = mutated_mod.get_global_var('func_1083')
const_1130 = relay.const([-7.325791,-4.635322,5.596990,1.870310,-9.292715], dtype = "float64")#candidate|1130|(5,)|const|float64
call_1129 = relay.TupleGetItem(func_1080_call(relay.reshape(const_1130.astype('float64'), [5,]), relay.reshape(const_1130.astype('float64'), [5,]), ), 0)
call_1131 = relay.TupleGetItem(func_1083_call(relay.reshape(const_1130.astype('float64'), [5,]), relay.reshape(const_1130.astype('float64'), [5,]), ), 0)
var_1132 = relay.var("var_1132", dtype = "int16", shape = (4, 12, 7))#candidate|1132|(4, 12, 7)|var|int16
bop_1133 = relay.equal(bop_1121.astype('bool'), relay.reshape(var_1132.astype('bool'), relay.shape_of(bop_1121))) # shape=(4, 12, 7)
output = relay.Tuple([bop_1115,call_1129,const_1130,bop_1133,])
output2 = relay.Tuple([bop_1115,call_1131,const_1130,bop_1133,])
F = relay.Function([var_1109,var_1110,var_1120,var_1132,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1109,var_1110,var_1120,var_1132,], output2)
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
seq = Sequential([
	relay.transform.AlterOpLayout(),
	relay.transform.AnnotateSpans(),
	relay.transform.BatchingOps(),
	relay.transform.CanonicalizeCast(),
	relay.transform.CanonicalizeOps(),
	relay.transform.DeadCodeElimination(),
	relay.transform.DynamicToStatic(),
	relay.transform.FastMath(),
	relay.transform.FirstOrderGradient(),
	relay.transform.EliminateCommonSubexpr(),
	relay.transform.MergeCompilerRegions(),
	relay.transform.Inline(),
	relay.transform.LambdaLift(),
	relay.transform.LazyGradientInit(),
	relay.transform.PartialEvaluate(),
	relay.transform.Legalize(),
	relay.transform.FoldConstant(),
	relay.transform.ToANormalForm(),
])
mod = seq(mod)
print(mod.astext(show_meta_data=False))
graph, lib, params = relay.build(mod, target='llvm')
module9 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp10 = relay.build_module.create_executor('graph', mod, tvm.device('llvm',0),'llvm')
intrp11 = relay.build_module.create_executor('debug', mod, tvm.device('llvm',0),'llvm')
intrp12 = relay.build_module.create_executor('vm', mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mod, target='cuda')
module13 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp14 = relay.build_module.create_executor('graph', mod, tvm.device('cuda',0),'cuda')
intrp15 = relay.build_module.create_executor('debug', mod, tvm.device('cuda',0),'cuda')
intrp16 = relay.build_module.create_executor('vm', mod, tvm.device('cuda',0),'cuda')
graph, lib, params = relay.build(mutated_mod, target='llvm')
module17 = graph_runtime.create(graph, lib, tvm.device('llvm',0))
intrp18 = relay.build_module.create_executor('graph', mutated_mod, tvm.device('llvm',0),'llvm')
intrp19 = relay.build_module.create_executor('debug', mutated_mod, tvm.device('llvm',0),'llvm')
intrp20 = relay.build_module.create_executor('vm', mutated_mod, tvm.device('llvm',0),'llvm')
graph, lib, params = relay.build(mutated_mod, target='cuda')
module21 = graph_runtime.create(graph, lib, tvm.device('cuda',0))
intrp22 = relay.build_module.create_executor('graph', mutated_mod, tvm.device('cuda',0),'cuda')
intrp23 = relay.build_module.create_executor('debug', mutated_mod, tvm.device('cuda',0),'cuda')
intrp24 = relay.build_module.create_executor('vm', mutated_mod, tvm.device('cuda',0),'cuda')
input_1109= np.array([[[7.007891,8.952383,-0.412570,-5.103631,3.032711,-8.587980,2.028713],[-2.852993,-9.545680,0.962358,1.947952,-2.991819,-9.653635,6.418717],[-9.618961,-8.506272,-2.602143,8.169750,5.427727,-9.841292,2.427015],[-9.777546,7.278581,-2.421888,-4.486582,3.919913,-1.840628,6.297544],[3.790113,-0.370143,5.736439,4.249419,-8.266777,8.349805,-1.127435],[2.818173,-0.718799,-0.713707,-0.481889,0.603619,7.715418,-3.495141],[-1.936827,-6.252359,8.626680,7.865118,4.452929,-1.872904,4.815911],[8.407678,2.359419,7.260562,-3.623997,8.634482,4.685040,-1.268770],[2.476474,-5.849629,-4.921576,3.163656,9.724548,-3.000193,2.953215],[-2.989891,-5.360806,1.631856,-4.853208,-7.763835,-9.804625,-5.419296],[-4.284907,-9.976315,7.383543,-3.228572,6.205092,3.198347,-2.829792],[3.708114,-6.323803,9.399815,3.107900,-8.221885,0.444624,5.958598]],[[-7.011323,0.391301,-2.306744,6.561359,-9.977581,0.736358,2.577921],[-6.688538,-5.571455,-9.806990,-0.783098,-4.062479,-9.414483,9.118174],[5.223932,4.620259,-1.111787,-2.578307,9.753192,1.281342,7.824552],[8.051271,9.934325,7.094551,-2.438508,6.826360,0.648420,-1.677183],[-5.844405,2.703564,-8.591032,-7.764602,-1.796591,8.531935,-7.467744],[0.090420,1.704531,-6.222223,0.432676,2.779599,3.101659,-9.117116],[-8.572958,-6.613656,-7.457782,-0.216452,9.360609,-3.958071,7.195879],[7.922914,-7.541364,-0.004107,1.950933,-6.664526,-4.835749,-8.474508],[-8.192499,-9.145778,-2.550912,-7.467651,-4.699767,4.941511,4.133158],[2.517920,-3.243863,-6.701206,-7.831890,-1.418175,7.091987,-6.028754],[-9.125459,-3.441065,3.656469,4.288670,9.646641,-5.516655,-0.747464],[1.022010,5.097543,5.148564,7.513929,-5.970917,-5.305151,-6.635766]],[[-4.514401,1.005033,8.399203,2.383699,-7.569073,-8.651700,-3.690049],[4.245826,4.822155,-5.221094,-6.858684,-5.192292,8.387402,7.872306],[6.777267,-5.472436,-4.529623,1.368944,0.768153,-6.335798,2.850498],[7.733725,6.003755,4.302703,-0.697840,-3.513600,2.337831,-1.445591],[1.886963,-9.759517,8.596645,-3.873134,-7.664839,-4.343712,9.495540],[7.074142,4.979507,-4.893497,1.365915,6.255838,-2.670218,-3.233793],[-7.302771,-5.001506,-4.638446,0.602221,6.703594,0.242698,0.594217],[6.880252,-5.521315,7.714599,-4.434894,9.036950,6.339054,5.862326],[-4.866288,9.929502,-8.800096,-0.254185,-9.836492,1.328590,-6.195296],[4.261390,-9.434278,-4.406791,9.822050,-6.732277,4.873775,-4.026477],[7.586590,-5.135510,2.230366,-7.425092,-0.507021,9.103362,-3.078449],[-2.692917,2.123360,-8.816899,6.539272,0.227510,5.549014,-3.396759]],[[0.031009,1.001822,-4.784655,-8.309618,-2.427584,2.391819,7.673835],[2.183967,4.319761,-9.001301,-1.713429,-8.530335,3.473010,-6.150065],[-4.590480,-3.548989,5.371245,-4.037303,0.751722,-7.763260,0.457160],[-6.735979,-8.196165,5.888820,-6.228071,-6.970046,-9.591074,-9.105375],[-2.627266,-0.038892,1.615748,-0.658840,3.891650,-3.566211,3.035143],[-4.274685,-6.539626,-6.764115,-2.251212,5.782782,9.935751,3.384195],[-7.932953,-1.033079,6.767327,6.354996,7.569664,7.160993,-8.782410],[3.586066,-4.139274,1.186588,6.680889,3.824789,1.002493,-8.001282],[5.485427,-9.913732,5.539198,-5.959395,-7.051987,7.553350,-5.014486],[-4.416706,-7.757538,-2.514139,7.474943,9.956352,4.454582,-4.391056],[4.004971,-2.519232,2.962331,4.105984,2.025675,2.431043,3.954143],[4.792075,7.815326,1.881008,5.273524,4.773976,0.597222,5.343294]]], dtype='float64')
module1.set_input('var_1109', input_1109)
input_1110= np.array([[[3.073263,4.092146,-0.708686,4.780767,-9.163998,-2.282170,0.122009],[3.953445,-6.079997,-2.087529,-3.162552,4.108247,5.561752,-3.156889],[-2.791531,-2.091540,-6.420229,-5.069306,7.079917,-5.558774,-4.230254],[7.883169,-4.570357,9.139485,9.649957,-6.263252,3.929043,9.305972],[-3.924588,5.268091,-3.769201,4.642353,0.827543,0.943673,3.202613],[-9.365466,2.734628,3.311529,6.789681,-3.449774,-4.775610,-9.621473],[9.726158,6.579173,-6.082380,-5.101107,-4.465622,8.854914,8.548568],[-6.014791,-6.305967,8.772237,-0.261886,8.810349,3.653218,-9.967850],[-1.032671,0.228869,-8.022443,-9.660480,4.255723,7.192611,-8.952123],[6.748112,-4.819602,-4.130253,-8.918180,-6.712776,5.887293,-9.232155],[2.300519,-8.008528,-4.649520,6.974036,-7.172716,1.913783,-9.307557],[6.401732,8.937495,3.893409,-4.758446,-8.347961,4.298951,0.324244]],[[-9.531682,8.329432,-8.864729,6.923044,4.355215,-5.775342,-3.053769],[2.281635,-2.354514,8.954281,8.263661,8.304043,5.222479,-1.046473],[-5.438269,3.811291,-1.767849,1.785068,-0.546897,6.662015,4.728250],[-1.388333,6.953483,2.370450,-1.412835,0.580279,7.551048,-8.516551],[-1.687441,2.300563,-8.990685,7.572630,3.848555,-3.362772,5.847189],[5.221080,3.374362,0.963528,6.129601,-8.137819,3.691522,8.272659],[8.368939,4.415863,-6.913240,-5.090228,3.396799,5.288429,-6.602309],[0.027723,8.387205,9.141833,-6.711911,5.058169,6.956773,1.624345],[5.416511,1.737837,-4.659801,-6.387846,1.106294,-0.476251,-1.586048],[2.132515,2.154651,8.111989,2.802501,8.959172,5.726821,0.075945],[-4.813781,6.651522,7.979850,-6.032738,-1.034059,-8.486962,-4.708657],[-7.865663,4.910042,3.242311,-9.469118,1.323539,9.854037,2.005553]],[[-5.920008,5.637657,-8.239468,-4.558888,7.033319,3.767122,-4.408673],[5.741019,-0.042866,2.303507,9.168765,-4.785729,4.301063,7.572688],[-4.366288,5.771220,7.599178,-7.052282,-0.321151,0.271097,0.263425],[2.903624,1.112094,-0.979510,-0.614852,2.794991,0.140792,-2.505411],[2.114612,1.275105,-4.242375,-6.351545,-5.054313,7.474130,-1.371654],[-3.065157,-3.080508,2.041355,5.446659,7.323100,4.833140,2.081500],[7.726852,-6.950789,-1.340266,-3.762640,-5.093543,-3.063459,9.847299],[2.351138,2.311767,-4.883513,4.834530,0.834244,2.359476,9.211021],[2.539692,1.855299,-1.966318,3.279055,-4.487419,9.037887,1.905313],[3.058611,-6.473841,-8.133489,-2.381462,5.048789,-9.014277,-9.249556],[0.510469,-5.747839,5.321943,2.343538,9.770893,-8.661314,0.139133],[7.191724,3.424186,8.444776,1.575312,1.102841,-1.715200,-6.806713]],[[6.842689,1.471631,-9.961980,1.548545,0.167574,9.069104,-6.696258],[-0.452229,1.225401,-4.695961,-7.382268,-8.535499,1.563386,-5.193881],[-2.345429,-5.963799,2.702420,6.882012,2.501686,5.986529,0.602656],[-5.760561,-1.268047,3.879665,-9.389516,6.034268,-9.207532,-6.275333],[8.516706,9.718064,8.357165,-4.730423,7.184005,-6.038901,-1.838636],[-9.226871,-9.310686,7.181163,0.867231,1.370716,-8.722091,7.268349],[-5.211869,-0.044729,-4.926767,-4.073551,5.490186,-4.333102,-0.285215],[-5.058453,4.258838,-5.011103,1.531055,-2.309780,4.844874,-3.372639],[-1.799643,-2.575483,0.991928,3.071107,-5.941626,-0.163646,-8.790119],[2.733960,3.955277,2.986915,-3.411265,0.327372,-3.809030,-7.378419],[5.420382,2.633929,-2.827001,-8.077376,0.086559,0.659953,6.585534],[5.814653,6.299715,-4.356061,-5.509003,3.058089,-0.406572,-1.789721]]], dtype='float64')
module1.set_input('var_1110', input_1110)
input_1120= np.array([[[-5.534429,-1.874461,0.854655,-1.829429,8.552819,-4.929872,-8.021151],[-9.001838,6.137166,5.047207,-7.752529,2.596266,7.735070,4.630254],[5.608314,-0.250089,4.457664,7.769174,5.675512,-8.385325,8.922694],[4.190049,7.473533,0.195952,2.695238,9.539807,1.444405,6.424941],[7.384614,-6.139790,-2.777620,-9.444625,7.064534,5.090074,5.327531],[9.485467,-6.503418,-3.704756,-6.350754,-2.572122,-0.644019,-3.595262],[-7.495607,6.507530,-5.258622,2.155339,-0.029305,-2.823214,4.651117],[-0.112293,-6.222191,3.237102,-4.135761,-3.152874,-9.655034,5.820442],[-6.602954,-5.491028,8.533605,-4.270976,0.705292,-6.647229,2.986385],[9.478752,-3.542805,-5.946822,3.445352,9.552993,0.338449,-5.242347],[-4.917412,9.743956,3.311313,-2.958297,0.412255,1.426781,-6.700992],[-1.979183,-4.500886,-4.592742,-5.253381,5.304755,4.002895,7.943615]],[[-9.420523,-8.915759,0.352760,0.593109,8.155016,2.362909,-1.035377],[3.576493,3.221006,5.811713,8.142848,-7.662571,-1.741259,2.980847],[2.835675,-6.080740,-1.984731,-6.007837,-7.335552,7.007232,0.801927],[-3.551633,0.240815,-2.528727,-6.008412,-5.805039,-9.128647,4.224774],[8.533899,-4.672561,6.871289,-4.316058,8.987354,-6.602115,1.703875],[0.695266,-0.777788,-4.900605,5.889656,-1.617499,-1.277541,5.385563],[-9.276057,1.658027,0.248992,-1.855521,4.172933,-0.146846,3.948363],[-0.890851,-5.614423,-9.211137,-5.072145,-0.654631,1.715279,8.326030],[-8.008225,-4.207640,8.710654,-3.746292,7.922147,8.220787,4.489094],[4.842518,-7.818066,9.092116,9.830694,-1.304721,2.089529,3.514696],[5.444461,-6.831938,6.980310,-7.244291,7.978822,1.189819,8.127332],[7.023454,3.230115,-5.258863,-4.663944,-2.107390,0.095519,3.135791]],[[-0.222229,7.222428,3.884247,4.391733,0.891073,7.300531,-9.971733],[1.887055,-6.650085,-4.686681,-8.524824,0.566383,0.403471,-7.671829],[-4.975872,3.921341,5.867057,7.829435,-6.556459,3.856856,-4.564785],[-8.751198,-5.739443,9.206108,-7.748458,-1.341473,-2.113339,-1.246490],[5.211567,-1.319699,-2.342106,-4.260147,-3.926511,7.552695,-4.640942],[-7.092652,5.760764,9.057974,9.973789,-6.617886,7.544597,-4.767563],[0.085344,6.924003,5.408163,-9.787856,7.945613,7.702982,-3.554429],[-4.257369,-2.821772,-5.710626,-0.334296,2.173701,-7.640827,9.627768],[-6.826865,0.903847,-3.543789,-7.867227,-5.497873,-0.830927,-1.921124],[4.964274,-0.019998,4.657543,4.633025,-9.505939,-2.034028,5.712908],[-2.587425,-2.777431,-9.224993,-9.428425,-6.489378,-4.668575,5.280750],[-1.473072,0.048823,2.180081,6.587081,-7.191204,-8.582114,-5.608342]],[[-3.183443,0.701763,7.144405,0.445108,8.680355,-3.140381,-9.180662],[-4.877188,4.039499,8.825133,-3.072111,-4.516734,-9.445418,-7.142144],[-7.167560,3.986942,-4.856871,8.037507,-4.844314,3.513613,-0.044363],[7.767732,5.712315,-1.400661,-5.909542,4.246583,-6.359687,-2.715899],[-3.035187,7.743679,-2.196222,4.583522,7.086566,-8.331645,7.930522],[2.720133,3.947736,-6.756031,-5.800433,-1.554838,9.319408,-2.182179],[9.484719,-5.097999,6.574379,-3.832310,-4.839172,1.513975,-9.575483],[2.885361,7.179343,6.672640,8.442817,0.465034,7.516549,-6.224273],[8.831089,-3.072792,8.330545,-0.671577,0.288476,-8.238863,-0.482973],[6.202946,0.395539,-7.620807,-9.602384,-1.777264,-9.152408,0.249391],[2.993592,-2.918697,-8.106784,6.415819,4.547458,4.667929,8.188140],[-0.034177,0.329150,-7.362648,-2.778110,4.435141,8.385410,-2.451241]]], dtype='float32')
module1.set_input('var_1120', input_1120)
input_1132= np.array([[[8,-1,10,-2,1,1,-9],[-10,-8,-5,9,4,-2,8],[-8,8,-4,-3,-2,-9,2],[-5,-8,-5,6,-3,-7,-4],[3,-2,5,-9,10,-5,-8],[7,4,-1,5,-1,6,-7],[-10,-6,4,-5,8,8,-9],[-8,9,9,7,8,-7,-4],[-4,8,9,8,-1,-2,1],[5,-5,7,10,7,-7,9],[5,-9,-2,10,2,-7,-3],[2,-1,1,3,1,-7,1]],[[7,-10,-3,9,7,-9,9],[3,-6,-3,-5,-4,6,4],[-8,-9,-4,5,-9,5,6],[-6,9,-7,8,2,8,-4],[5,-8,-2,8,-6,-5,1],[-1,-3,2,-8,-7,-2,-10],[8,3,-9,-6,7,-8,1],[8,-5,-1,-5,2,-3,-10],[9,3,3,1,-8,4,5],[10,7,-9,6,1,-2,-7],[-2,-10,-6,-10,-3,-10,2],[-8,-7,-4,-6,-5,3,-6]],[[-8,2,9,-1,-6,-3,-4],[2,8,-2,8,-7,10,-9],[-7,10,9,5,3,-2,-9],[-10,7,-10,1,9,-7,-6],[-10,10,1,3,1,7,-8],[-6,3,1,-9,-6,-2,-10],[3,7,-10,-1,8,-2,-3],[-4,-9,7,6,3,-10,4],[5,-2,1,-10,-8,-3,-6],[-3,5,2,-8,5,1,-9],[-6,-3,-8,-7,-6,-9,9],[-9,6,-8,1,-4,-2,-3]],[[4,2,-5,-1,9,2,-7],[10,-3,-5,-2,2,2,4],[-5,-7,-1,1,3,10,9],[7,-3,4,-7,6,9,-5],[-6,-6,-8,-9,1,6,1],[-6,-6,-8,2,-4,10,-8],[-8,1,4,4,-2,2,3],[7,-1,-6,-10,5,-6,-2],[-9,-5,-5,-10,6,2,6],[-1,-1,-5,-3,-9,-3,6],[-5,6,-9,2,7,-5,-7],[1,3,2,-2,6,8,-10]]], dtype='int16')
module1.set_input('var_1132', input_1132)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res3 = intrp3.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res4 = intrp4.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res2 = vmobj_to_list(res2)
res3 = vmobj_to_list(res3)
res4 = vmobj_to_list(res4)
res1_0 = module1.get_output(0).asnumpy()
res2_0 = res2[0].asnumpy()
res3_0 = res3[0].asnumpy()
res4_0 = res4[0].asnumpy()
np.testing.assert_allclose(res1_0 ,res2_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_0 ,res3_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_0 ,res4_0, atol=1e-3, rtol=1e-3)
(res1_0 == res2_0).all()
(res1_0 == res3_0).all()
(res1_0 == res4_0).all()
res1_1 = module1.get_output(1).asnumpy()
res2_1 = res2[1].asnumpy()
res3_1 = res3[1].asnumpy()
res4_1 = res4[1].asnumpy()
np.testing.assert_allclose(res1_1 ,res2_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_1 ,res3_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_1 ,res4_1, atol=1e-3, rtol=1e-3)
(res1_1 == res2_1).all()
(res1_1 == res3_1).all()
(res1_1 == res4_1).all()
res1_2 = module1.get_output(2).asnumpy()
res2_2 = res2[2].asnumpy()
res3_2 = res3[2].asnumpy()
res4_2 = res4[2].asnumpy()
np.testing.assert_allclose(res1_2 ,res2_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res3_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res4_2, atol=1e-3, rtol=1e-3)
(res1_2 == res2_2).all()
(res1_2 == res3_2).all()
(res1_2 == res4_2).all()
res1_3 = module1.get_output(3).asnumpy()
res2_3 = res2[3].asnumpy()
res3_3 = res3[3].asnumpy()
res4_3 = res4[3].asnumpy()
np.testing.assert_allclose(res1_3 ,res2_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res3_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res4_3, atol=1e-3, rtol=1e-3)
(res1_3 == res2_3).all()
(res1_3 == res3_3).all()
(res1_3 == res4_3).all()
module5.set_input('var_1109', input_1109)
module5.set_input('var_1110', input_1110)
module5.set_input('var_1120', input_1120)
module5.set_input('var_1132', input_1132)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res7 = intrp7.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res8 = intrp8.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res6 = vmobj_to_list(res6)
res7 = vmobj_to_list(res7)
res8 = vmobj_to_list(res8)
res5_0 = module5.get_output(0).asnumpy()
res6_0 = res6[0].asnumpy()
res7_0 = res7[0].asnumpy()
res8_0 = res8[0].asnumpy()
np.testing.assert_allclose(res5_0 ,res6_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_0 ,res7_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_0 ,res8_0, atol=1e-3, rtol=1e-3)
(res5_0 == res6_0).all()
(res5_0 == res7_0).all()
(res5_0 == res8_0).all()
res5_1 = module5.get_output(1).asnumpy()
res6_1 = res6[1].asnumpy()
res7_1 = res7[1].asnumpy()
res8_1 = res8[1].asnumpy()
np.testing.assert_allclose(res5_1 ,res6_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_1 ,res7_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_1 ,res8_1, atol=1e-3, rtol=1e-3)
(res5_1 == res6_1).all()
(res5_1 == res7_1).all()
(res5_1 == res8_1).all()
res5_2 = module5.get_output(2).asnumpy()
res6_2 = res6[2].asnumpy()
res7_2 = res7[2].asnumpy()
res8_2 = res8[2].asnumpy()
np.testing.assert_allclose(res5_2 ,res6_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res7_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res8_2, atol=1e-3, rtol=1e-3)
(res5_2 == res6_2).all()
(res5_2 == res7_2).all()
(res5_2 == res8_2).all()
res5_3 = module5.get_output(3).asnumpy()
res6_3 = res6[3].asnumpy()
res7_3 = res7[3].asnumpy()
res8_3 = res8[3].asnumpy()
np.testing.assert_allclose(res5_3 ,res6_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res7_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res8_3, atol=1e-3, rtol=1e-3)
(res5_3 == res6_3).all()
(res5_3 == res7_3).all()
(res5_3 == res8_3).all()
module9.set_input('var_1109', input_1109)
module9.set_input('var_1110', input_1110)
module9.set_input('var_1120', input_1120)
module9.set_input('var_1132', input_1132)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res11 = intrp11.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res12 = intrp12.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res10 = vmobj_to_list(res10)
res11 = vmobj_to_list(res11)
res12 = vmobj_to_list(res12)
res9_0 = module9.get_output(0).asnumpy()
res10_0 = res10[0].asnumpy()
res11_0 = res11[0].asnumpy()
res12_0 = res12[0].asnumpy()
np.testing.assert_allclose(res9_0 ,res10_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_0 ,res11_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_0 ,res12_0, atol=1e-3, rtol=1e-3)
(res9_0 == res10_0).all()
(res9_0 == res11_0).all()
(res9_0 == res12_0).all()
res9_1 = module9.get_output(1).asnumpy()
res10_1 = res10[1].asnumpy()
res11_1 = res11[1].asnumpy()
res12_1 = res12[1].asnumpy()
np.testing.assert_allclose(res9_1 ,res10_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_1 ,res11_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_1 ,res12_1, atol=1e-3, rtol=1e-3)
(res9_1 == res10_1).all()
(res9_1 == res11_1).all()
(res9_1 == res12_1).all()
res9_2 = module9.get_output(2).asnumpy()
res10_2 = res10[2].asnumpy()
res11_2 = res11[2].asnumpy()
res12_2 = res12[2].asnumpy()
np.testing.assert_allclose(res9_2 ,res10_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res11_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res12_2, atol=1e-3, rtol=1e-3)
(res9_2 == res10_2).all()
(res9_2 == res11_2).all()
(res9_2 == res12_2).all()
res9_3 = module9.get_output(3).asnumpy()
res10_3 = res10[3].asnumpy()
res11_3 = res11[3].asnumpy()
res12_3 = res12[3].asnumpy()
np.testing.assert_allclose(res9_3 ,res10_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res11_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res12_3, atol=1e-3, rtol=1e-3)
(res9_3 == res10_3).all()
(res9_3 == res11_3).all()
(res9_3 == res12_3).all()
module13.set_input('var_1109', input_1109)
module13.set_input('var_1110', input_1110)
module13.set_input('var_1120', input_1120)
module13.set_input('var_1132', input_1132)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res15 = intrp15.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res16 = intrp16.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res14 = vmobj_to_list(res14)
res15 = vmobj_to_list(res15)
res16 = vmobj_to_list(res16)
res13_0 = module13.get_output(0).asnumpy()
res14_0 = res14[0].asnumpy()
res15_0 = res15[0].asnumpy()
res16_0 = res16[0].asnumpy()
np.testing.assert_allclose(res13_0 ,res14_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_0 ,res15_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_0 ,res16_0, atol=1e-3, rtol=1e-3)
(res13_0 == res14_0).all()
(res13_0 == res15_0).all()
(res13_0 == res16_0).all()
res13_1 = module13.get_output(1).asnumpy()
res14_1 = res14[1].asnumpy()
res15_1 = res15[1].asnumpy()
res16_1 = res16[1].asnumpy()
np.testing.assert_allclose(res13_1 ,res14_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_1 ,res15_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_1 ,res16_1, atol=1e-3, rtol=1e-3)
(res13_1 == res14_1).all()
(res13_1 == res15_1).all()
(res13_1 == res16_1).all()
res13_2 = module13.get_output(2).asnumpy()
res14_2 = res14[2].asnumpy()
res15_2 = res15[2].asnumpy()
res16_2 = res16[2].asnumpy()
np.testing.assert_allclose(res13_2 ,res14_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res15_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res16_2, atol=1e-3, rtol=1e-3)
(res13_2 == res14_2).all()
(res13_2 == res15_2).all()
(res13_2 == res16_2).all()
res13_3 = module13.get_output(3).asnumpy()
res14_3 = res14[3].asnumpy()
res15_3 = res15[3].asnumpy()
res16_3 = res16[3].asnumpy()
np.testing.assert_allclose(res13_3 ,res14_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res15_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res16_3, atol=1e-3, rtol=1e-3)
(res13_3 == res14_3).all()
(res13_3 == res15_3).all()
(res13_3 == res16_3).all()
module17.set_input('var_1109', input_1109)
module17.set_input('var_1110', input_1110)
module17.set_input('var_1120', input_1120)
module17.set_input('var_1132', input_1132)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res19 = intrp19.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res20 = intrp20.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res18 = vmobj_to_list(res18)
res19 = vmobj_to_list(res19)
res20 = vmobj_to_list(res20)
res17_0 = module17.get_output(0).asnumpy()
res18_0 = res18[0].asnumpy()
res19_0 = res19[0].asnumpy()
res20_0 = res20[0].asnumpy()
np.testing.assert_allclose(res17_0 ,res18_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_0 ,res19_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_0 ,res20_0, atol=1e-3, rtol=1e-3)
(res17_0 == res18_0).all()
(res17_0 == res19_0).all()
(res17_0 == res20_0).all()
res17_1 = module17.get_output(1).asnumpy()
res18_1 = res18[1].asnumpy()
res19_1 = res19[1].asnumpy()
res20_1 = res20[1].asnumpy()
np.testing.assert_allclose(res17_1 ,res18_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_1 ,res19_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_1 ,res20_1, atol=1e-3, rtol=1e-3)
(res17_1 == res18_1).all()
(res17_1 == res19_1).all()
(res17_1 == res20_1).all()
res17_2 = module17.get_output(2).asnumpy()
res18_2 = res18[2].asnumpy()
res19_2 = res19[2].asnumpy()
res20_2 = res20[2].asnumpy()
np.testing.assert_allclose(res17_2 ,res18_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res19_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res20_2, atol=1e-3, rtol=1e-3)
(res17_2 == res18_2).all()
(res17_2 == res19_2).all()
(res17_2 == res20_2).all()
res17_3 = module17.get_output(3).asnumpy()
res18_3 = res18[3].asnumpy()
res19_3 = res19[3].asnumpy()
res20_3 = res20[3].asnumpy()
np.testing.assert_allclose(res17_3 ,res18_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res19_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res20_3, atol=1e-3, rtol=1e-3)
(res17_3 == res18_3).all()
(res17_3 == res19_3).all()
(res17_3 == res20_3).all()
module21.set_input('var_1109', input_1109)
module21.set_input('var_1110', input_1110)
module21.set_input('var_1120', input_1120)
module21.set_input('var_1132', input_1132)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res23 = intrp23.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res24 = intrp24.evaluate()(input_1109, input_1110, input_1120, input_1132, )
res22 = vmobj_to_list(res22)
res23 = vmobj_to_list(res23)
res24 = vmobj_to_list(res24)
res21_0 = module21.get_output(0).asnumpy()
res22_0 = res22[0].asnumpy()
res23_0 = res23[0].asnumpy()
res24_0 = res24[0].asnumpy()
np.testing.assert_allclose(res21_0 ,res22_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_0 ,res23_0, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_0 ,res24_0, atol=1e-3, rtol=1e-3)
(res21_0 == res22_0).all()
(res21_0 == res23_0).all()
(res21_0 == res24_0).all()
res21_1 = module21.get_output(1).asnumpy()
res22_1 = res22[1].asnumpy()
res23_1 = res23[1].asnumpy()
res24_1 = res24[1].asnumpy()
np.testing.assert_allclose(res21_1 ,res22_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_1 ,res23_1, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_1 ,res24_1, atol=1e-3, rtol=1e-3)
(res21_1 == res22_1).all()
(res21_1 == res23_1).all()
(res21_1 == res24_1).all()
res21_2 = module21.get_output(2).asnumpy()
res22_2 = res22[2].asnumpy()
res23_2 = res23[2].asnumpy()
res24_2 = res24[2].asnumpy()
np.testing.assert_allclose(res21_2 ,res22_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res23_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res24_2, atol=1e-3, rtol=1e-3)
(res21_2 == res22_2).all()
(res21_2 == res23_2).all()
(res21_2 == res24_2).all()
res21_3 = module21.get_output(3).asnumpy()
res22_3 = res22[3].asnumpy()
res23_3 = res23[3].asnumpy()
res24_3 = res24[3].asnumpy()
np.testing.assert_allclose(res21_3 ,res22_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res23_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res24_3, atol=1e-3, rtol=1e-3)
(res21_3 == res22_3).all()
(res21_3 == res23_3).all()
(res21_3 == res24_3).all()

'''6: TVMFuncCall
5: _ZNSt17_Function_handlerIFvN3tvm7runtime7
4: tvm::runtime::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const [clone .isra.808]
3: tvm::runtime::GraphExecutorCreate(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module const&, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
2: tvm::runtime::GraphExecutor::Init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
1: tvm::runtime::GraphExecutor::SetupOpExecs()
0: tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::vector<DLTensor, std::allocator<DLTensor> > const&)

'''