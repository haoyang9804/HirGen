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
var_0 = relay.var("var_0", dtype = "float64", shape = (4, 4))#candidate|0|(4, 4)|var|float64
var_1 = relay.var("var_1", dtype = "float64", shape = (4, 4))#candidate|1|(4, 4)|var|float64
bop_2 = relay.not_equal(var_0.astype('bool'), relay.reshape(var_1.astype('bool'), relay.shape_of(var_0))) # shape=(4, 4)
bop_5 = relay.equal(var_0.astype('bool'), relay.reshape(var_1.astype('bool'), relay.shape_of(var_0))) # shape=(4, 4)
bop_8 = relay.maximum(var_1.astype('int8'), relay.reshape(bop_2.astype('int8'), relay.shape_of(var_1))) # shape=(4, 4)
uop_11 = relay.log10(var_1.astype('float32')) # shape=(4, 4)
bop_13 = relay.divide(uop_11.astype('float32'), relay.reshape(bop_8.astype('float32'), relay.shape_of(uop_11))) # shape=(4, 4)
bop_16 = relay.divide(bop_13.astype('float64'), relay.reshape(bop_8.astype('float64'), relay.shape_of(bop_13))) # shape=(4, 4)
uop_19 = relay.log2(bop_16.astype('float32')) # shape=(4, 4)
uop_21 = relay.acos(uop_19.astype('float32')) # shape=(4, 4)
uop_23 = relay.cosh(uop_19.astype('float64')) # shape=(4, 4)
bop_25 = relay.right_shift(uop_21.astype('uint16'), relay.reshape(bop_13.astype('uint16'), relay.shape_of(uop_21))) # shape=(4, 4)
output = relay.Tuple([bop_5,uop_23,bop_25,])
output2 = relay.Tuple([bop_5,uop_23,bop_25,])
func_28 = relay.Function([var_0,var_1,], output)
mod['func_28'] = func_28
mod = relay.transform.InferType()(mod)
mutated_mod['func_28'] = func_28
mutated_mod = relay.transform.InferType()(mutated_mod)
func_28_call = mutated_mod.get_global_var('func_28')
var_30 = relay.var("var_30", dtype = "float64", shape = (4, 4))#candidate|30|(4, 4)|var|float64
var_31 = relay.var("var_31", dtype = "float64", shape = (4, 4))#candidate|31|(4, 4)|var|float64
call_29 = func_28_call(var_30,var_31,)
output = call_29
func_32 = relay.Function([var_30,var_31,], output)
mutated_mod['func_32'] = func_32
mutated_mod = relay.transform.InferType()(mutated_mod)
var_34 = relay.var("var_34", dtype = "float64", shape = (16, 9, 9))#candidate|34|(16, 9, 9)|var|float64
uop_35 = relay.asin(var_34.astype('float64')) # shape=(16, 9, 9)
bop_37 = relay.less_equal(var_34.astype('bool'), relay.reshape(uop_35.astype('bool'), relay.shape_of(var_34))) # shape=(16, 9, 9)
bop_40 = relay.greater(var_34.astype('bool'), relay.reshape(uop_35.astype('bool'), relay.shape_of(var_34))) # shape=(16, 9, 9)
var_43 = relay.var("var_43", dtype = "float64", shape = (16, 9, 9))#candidate|43|(16, 9, 9)|var|float64
bop_44 = relay.power(var_34.astype('float32'), relay.reshape(var_43.astype('float32'), relay.shape_of(var_34))) # shape=(16, 9, 9)
uop_47 = relay.sinh(var_43.astype('float32')) # shape=(16, 9, 9)
const_49 = relay.const([[[True,False,False,True,True,True,True,False,True],[True,False,True,False,False,False,False,False,True],[True,True,False,False,True,False,False,False,True],[False,True,False,True,True,False,True,False,True],[False,True,False,True,True,False,False,True,False],[False,True,False,False,False,False,False,False,True],[False,False,True,True,True,False,True,False,True],[True,True,True,True,True,True,True,True,False],[True,True,True,True,False,True,False,False,True]],[[False,False,False,True,False,False,False,False,True],[False,True,True,False,True,False,True,False,False],[False,True,True,False,False,False,False,True,False],[True,True,False,False,True,False,False,False,True],[True,False,True,False,True,False,False,True,True],[False,False,True,False,True,False,True,True,False],[False,True,False,False,False,True,True,True,True],[True,True,True,False,False,False,True,True,True],[False,True,False,True,True,False,True,False,True]],[[True,True,True,False,True,False,False,False,True],[True,True,False,False,False,True,False,True,False],[False,False,True,True,False,False,True,False,True],[True,True,True,True,False,True,False,False,False],[False,False,False,True,False,True,True,False,False],[True,False,True,True,False,True,False,True,False],[False,False,False,True,False,True,True,True,True],[False,True,False,False,True,False,True,True,False],[False,False,True,False,True,True,True,False,False]],[[True,False,True,True,False,False,True,False,False],[False,True,True,True,True,False,True,True,True],[False,False,False,False,True,True,True,True,False],[True,True,True,True,False,True,False,True,False],[False,False,False,False,False,True,True,True,False],[True,True,True,True,True,False,True,True,True],[False,True,False,True,False,True,False,True,True],[True,True,False,True,False,False,True,False,False],[False,False,False,False,True,True,False,False,False]],[[False,False,True,True,False,False,True,True,False],[True,True,True,False,True,True,True,False,True],[True,False,True,False,False,True,False,True,True],[True,True,True,True,True,True,False,False,False],[True,True,True,True,False,True,True,True,False],[False,False,False,True,True,False,False,True,True],[False,True,False,True,False,True,False,True,False],[False,False,False,False,True,True,True,False,False],[False,True,True,False,True,True,True,False,False]],[[True,True,False,False,True,True,False,False,False],[True,False,True,True,False,True,True,False,False],[True,False,True,True,False,False,False,True,False],[True,False,False,True,True,True,True,False,False],[True,False,False,True,False,True,False,True,True],[False,True,False,False,False,False,True,True,False],[False,True,True,False,False,True,False,True,True],[False,True,True,False,False,True,True,True,True],[False,True,True,True,True,False,True,False,False]],[[True,True,True,False,True,False,True,True,False],[True,False,True,False,False,False,True,False,False],[False,True,True,False,True,True,True,True,False],[True,False,False,True,False,False,False,False,True],[False,True,True,False,False,True,True,False,True],[False,True,True,False,False,True,False,False,False],[True,True,True,True,False,False,False,True,False],[False,True,False,True,True,True,False,True,False],[True,False,False,False,False,False,False,True,False]],[[True,True,False,True,False,True,True,True,True],[True,True,False,True,True,True,True,True,False],[False,True,True,False,True,True,True,True,False],[True,True,True,True,False,False,True,False,False],[False,True,True,True,False,True,True,True,False],[False,False,True,False,False,True,True,True,False],[False,False,True,False,True,True,True,False,True],[True,True,True,True,True,False,True,False,False],[False,True,True,False,True,True,False,True,False]],[[True,False,True,True,False,True,False,True,False],[True,False,False,True,False,True,False,True,False],[True,False,False,True,False,True,True,True,False],[False,True,True,False,False,True,True,True,True],[False,True,False,False,True,True,False,False,True],[True,False,False,True,True,True,True,True,True],[False,False,False,False,False,True,True,True,True],[False,False,False,False,True,False,False,True,True],[True,False,True,False,True,True,True,True,True]],[[False,False,False,True,True,False,False,True,False],[True,True,True,True,True,False,True,True,True],[True,False,False,False,True,False,True,False,False],[True,True,True,False,True,True,False,False,False],[False,False,False,False,False,True,False,True,False],[False,False,False,True,False,False,True,False,True],[False,False,True,False,True,False,True,True,True],[False,True,False,True,True,False,True,True,False],[False,True,True,False,True,False,False,False,False]],[[False,False,False,False,False,False,True,False,True],[False,True,False,True,True,True,True,False,False],[False,True,False,False,True,True,False,False,True],[False,False,True,False,True,True,False,True,True],[True,False,True,False,False,False,True,False,False],[False,True,False,True,True,False,True,False,True],[False,False,True,True,False,False,True,False,True],[False,False,False,False,True,False,True,False,True],[False,True,True,False,True,False,False,False,False]],[[False,True,False,False,False,False,True,True,False],[True,False,False,False,True,False,False,True,False],[True,False,False,False,False,True,True,False,False],[True,True,True,True,True,False,True,True,False],[True,True,False,True,False,False,True,True,True],[False,True,False,False,False,True,False,False,True],[True,True,False,True,True,True,False,False,False],[True,False,False,True,True,True,True,True,True],[False,False,False,True,False,False,False,False,False]],[[True,False,True,False,True,False,False,True,True],[True,True,False,False,False,False,False,False,True],[True,True,False,False,True,True,True,True,True],[True,True,False,True,False,False,False,False,False],[True,True,True,False,False,False,False,False,True],[False,False,True,False,True,False,False,False,False],[True,True,True,True,False,False,True,True,False],[True,False,False,True,True,True,False,True,False],[True,False,False,False,False,True,True,False,False]],[[True,True,False,True,False,True,False,True,True],[False,False,True,False,False,True,True,True,False],[False,False,True,False,True,True,True,True,False],[False,False,False,False,True,True,True,True,False],[False,True,True,False,False,False,True,False,False],[False,True,True,False,False,True,True,False,False],[True,True,True,True,True,True,True,False,True],[True,True,False,True,True,True,False,True,True],[False,False,False,False,False,True,False,False,True]],[[True,False,False,False,True,True,True,False,True],[True,True,True,False,False,False,False,True,True],[True,False,True,True,False,True,True,True,False],[False,True,False,False,False,False,False,False,True],[True,False,True,False,True,True,True,True,True],[True,True,True,True,False,True,False,True,True],[True,False,False,True,True,True,False,True,False],[False,True,False,False,False,False,True,True,True],[False,False,False,False,False,True,True,True,False]],[[False,True,True,True,False,False,False,False,True],[True,False,False,True,False,True,False,False,True],[False,False,False,False,False,True,False,False,True],[False,True,False,False,True,True,True,True,True],[True,True,True,False,False,True,False,False,False],[True,False,False,True,False,False,True,False,True],[False,True,True,True,True,True,True,True,False],[False,False,True,False,False,False,True,False,True],[True,True,True,True,False,True,True,True,True]]], dtype = "bool")#candidate|49|(16, 9, 9)|const|bool
bop_50 = relay.mod(bop_37.astype('float64'), relay.reshape(const_49.astype('float64'), relay.shape_of(bop_37))) # shape=(16, 9, 9)
uop_53 = relay.cosh(const_49.astype('float64')) # shape=(16, 9, 9)
bop_55 = relay.not_equal(const_49.astype('bool'), relay.reshape(uop_53.astype('bool'), relay.shape_of(const_49))) # shape=(16, 9, 9)
uop_58 = relay.log(bop_37.astype('float32')) # shape=(16, 9, 9)
uop_60 = relay.exp(uop_35.astype('float64')) # shape=(16, 9, 9)
bop_62 = relay.power(uop_60.astype('float64'), relay.reshape(bop_50.astype('float64'), relay.shape_of(uop_60))) # shape=(16, 9, 9)
uop_65 = relay.erf(var_43.astype('float64')) # shape=(16, 9, 9)
bop_67 = relay.divide(uop_60.astype('float32'), relay.reshape(uop_65.astype('float32'), relay.shape_of(uop_60))) # shape=(16, 9, 9)
func_28_call = mod.get_global_var('func_28')
func_32_call = mutated_mod.get_global_var('func_32')
var_71 = relay.var("var_71", dtype = "float64", shape = (16,))#candidate|71|(16,)|var|float64
call_70 = relay.TupleGetItem(func_28_call(relay.reshape(var_71.astype('float64'), [4, 4]), relay.reshape(var_71.astype('float64'), [4, 4]), ), 2)
call_72 = relay.TupleGetItem(func_32_call(relay.reshape(var_71.astype('float64'), [4, 4]), relay.reshape(var_71.astype('float64'), [4, 4]), ), 2)
output = relay.Tuple([bop_40,bop_44,uop_47,bop_55,uop_58,bop_62,bop_67,call_70,var_71,])
output2 = relay.Tuple([bop_40,bop_44,uop_47,bop_55,uop_58,bop_62,bop_67,call_72,var_71,])
func_73 = relay.Function([var_34,var_43,var_71,], output)
mod['func_73'] = func_73
mod = relay.transform.InferType()(mod)
var_74 = relay.var("var_74", dtype = "float64", shape = (16, 9, 9))#candidate|74|(16, 9, 9)|var|float64
var_75 = relay.var("var_75", dtype = "float64", shape = (16, 9, 9))#candidate|75|(16, 9, 9)|var|float64
var_76 = relay.var("var_76", dtype = "float64", shape = (16,))#candidate|76|(16,)|var|float64
output = func_73(var_74,var_75,var_76,)
func_77 = relay.Function([var_74,var_75,var_76,], output)
mutated_mod['func_77'] = func_77
mutated_mod = relay.transform.InferType()(mutated_mod)
var_79 = relay.var("var_79", dtype = "float32", shape = ())#candidate|79|()|var|float32
uop_80 = relay.sin(var_79.astype('float32')) # shape=()
bop_82 = relay.less_equal(uop_80.astype('bool'), var_79.astype('bool')) # shape=()
output = relay.Tuple([bop_82,])
output2 = relay.Tuple([bop_82,])
func_85 = relay.Function([var_79,], output)
mod['func_85'] = func_85
mod = relay.transform.InferType()(mod)
var_86 = relay.var("var_86", dtype = "float32", shape = ())#candidate|86|()|var|float32
output = func_85(var_86)
func_87 = relay.Function([var_86], output)
mutated_mod['func_87'] = func_87
mutated_mod = relay.transform.InferType()(mutated_mod)
var_89 = relay.var("var_89", dtype = "int64", shape = (3, 12, 9))#candidate|89|(3, 12, 9)|var|int64
const_90 = relay.const([[[8,-9,-3,-5,5,-9,1,8,1],[-10,9,6,4,3,-3,-1,10,5],[-4,-6,-4,-10,8,9,-9,6,1],[-6,-5,-8,-4,1,-8,7,-7,-4],[8,7,9,-7,5,8,5,-2,3],[-1,10,10,8,-1,-2,7,-7,8],[-10,-8,-8,2,10,-6,-5,-9,8],[-4,-7,-3,-2,-9,-5,-6,-1,2],[1,-3,-10,-4,-5,-4,10,-3,-5],[-6,-7,5,-1,8,7,9,6,9],[-2,-4,5,2,-6,-2,1,5,4],[-3,2,5,5,-7,10,8,-8,9]],[[-2,9,-7,-6,-1,-10,2,-10,-7],[4,-10,3,-9,-10,-5,2,-3,5],[4,-8,-4,-1,-4,-3,-5,10,-7],[5,-6,-3,-7,10,6,8,7,-6],[-8,1,10,-10,-8,10,5,-1,4],[2,9,-9,-7,6,7,-9,10,5],[-2,3,-7,-5,-2,-3,-2,-6,-9],[7,-4,-10,-4,-5,8,4,-5,10],[-2,4,1,10,2,-4,1,-7,-2],[-7,7,-3,-2,-3,-4,8,-7,-9],[10,4,5,3,9,4,-6,-10,-3],[-7,7,-3,-2,6,3,-10,7,5]],[[1,-10,7,-2,5,5,6,-4,-2],[4,7,-5,-7,3,8,6,3,10],[9,4,-7,-3,-1,-5,-5,-8,9],[3,8,-10,2,-10,1,7,-3,6],[-10,-2,-8,-5,-6,-5,6,-2,2],[5,-9,10,-4,-2,8,8,6,10],[-6,7,7,10,-10,2,-6,5,-9],[-7,-2,1,-6,-9,-8,6,-5,8],[5,-7,-2,-6,-5,6,-5,-8,-2],[8,-7,-8,-9,1,-9,8,-6,-8],[4,-9,-6,5,-7,-10,-4,-5,-9],[-6,-9,-2,5,-4,-5,-1,8,-9]]], dtype = "int64")#candidate|90|(3, 12, 9)|const|int64
bop_91 = relay.logical_xor(var_89.astype('int64'), relay.reshape(const_90.astype('int64'), relay.shape_of(var_89))) # shape=(3, 12, 9)
var_94 = relay.var("var_94", dtype = "int64", shape = (3, 12, 9))#candidate|94|(3, 12, 9)|var|int64
bop_95 = relay.less_equal(const_90.astype('bool'), relay.reshape(var_94.astype('bool'), relay.shape_of(const_90))) # shape=(3, 12, 9)
bop_98 = relay.right_shift(const_90.astype('uint32'), relay.reshape(var_94.astype('uint32'), relay.shape_of(const_90))) # shape=(3, 12, 9)
uop_101 = relay.sin(bop_95.astype('float64')) # shape=(3, 12, 9)
bop_103 = relay.greater(const_90.astype('bool'), relay.reshape(var_89.astype('bool'), relay.shape_of(const_90))) # shape=(3, 12, 9)
var_106 = relay.var("var_106", dtype = "float64", shape = (3, 12, 9))#candidate|106|(3, 12, 9)|var|float64
bop_107 = relay.greater(uop_101.astype('bool'), relay.reshape(var_106.astype('bool'), relay.shape_of(uop_101))) # shape=(3, 12, 9)
var_110 = relay.var("var_110", dtype = "uint32", shape = (3, 12, 9))#candidate|110|(3, 12, 9)|var|uint32
bop_111 = relay.bitwise_xor(bop_98.astype('int8'), relay.reshape(var_110.astype('int8'), relay.shape_of(bop_98))) # shape=(3, 12, 9)
uop_114 = relay.log2(bop_103.astype('float64')) # shape=(3, 12, 9)
uop_116 = relay.log10(bop_107.astype('float64')) # shape=(3, 12, 9)
var_118 = relay.var("var_118", dtype = "float64", shape = (3, 12, 9))#candidate|118|(3, 12, 9)|var|float64
bop_119 = relay.greater_equal(uop_116.astype('bool'), relay.reshape(var_118.astype('bool'), relay.shape_of(uop_116))) # shape=(3, 12, 9)
bop_122 = relay.less_equal(bop_95.astype('bool'), relay.reshape(bop_103.astype('bool'), relay.shape_of(bop_95))) # shape=(3, 12, 9)
output = relay.Tuple([bop_91,bop_111,uop_114,bop_119,bop_122,])
output2 = relay.Tuple([bop_91,bop_111,uop_114,bop_119,bop_122,])
func_125 = relay.Function([var_89,var_94,var_106,var_110,var_118,], output)
mod['func_125'] = func_125
mod = relay.transform.InferType()(mod)
mutated_mod['func_125'] = func_125
mutated_mod = relay.transform.InferType()(mutated_mod)
func_125_call = mutated_mod.get_global_var('func_125')
var_127 = relay.var("var_127", dtype = "int64", shape = (3, 12, 9))#candidate|127|(3, 12, 9)|var|int64
var_128 = relay.var("var_128", dtype = "int64", shape = (3, 12, 9))#candidate|128|(3, 12, 9)|var|int64
var_129 = relay.var("var_129", dtype = "float64", shape = (3, 12, 9))#candidate|129|(3, 12, 9)|var|float64
var_130 = relay.var("var_130", dtype = "uint32", shape = (3, 12, 9))#candidate|130|(3, 12, 9)|var|uint32
var_131 = relay.var("var_131", dtype = "float64", shape = (3, 12, 9))#candidate|131|(3, 12, 9)|var|float64
call_126 = func_125_call(var_127,var_128,var_129,var_130,var_131,)
output = call_126
func_132 = relay.Function([var_127,var_128,var_129,var_130,var_131,], output)
mutated_mod['func_132'] = func_132
mutated_mod = relay.transform.InferType()(mutated_mod)
var_134 = relay.var("var_134", dtype = "float32", shape = ())#candidate|134|()|var|float32
var_135 = relay.var("var_135", dtype = "float32", shape = (6, 5, 10))#candidate|135|(6, 5, 10)|var|float32
bop_136 = relay.divide(var_134.astype('float32'), var_135.astype('float32')) # shape=(6, 5, 10)
var_139 = relay.var("var_139", dtype = "float32", shape = (6, 5, 10))#candidate|139|(6, 5, 10)|var|float32
bop_140 = relay.bitwise_and(bop_136.astype('uint8'), relay.reshape(var_139.astype('uint8'), relay.shape_of(bop_136))) # shape=(6, 5, 10)
const_143 = relay.const([[[-4.743199,-5.786616,-6.721007,5.596992,3.153326,-8.632384,4.833084,-2.184001,2.084644,-6.847326],[-7.476895,0.720046,5.624824,7.974559,4.215386,2.236768,-0.771444,-7.321817,-0.687889,-6.786361],[-1.964983,-4.046113,-2.522281,-4.192144,8.536226,-3.997925,9.117035,2.677267,5.479578,6.604053],[1.153132,-0.738413,-6.298065,-5.019412,7.381442,0.743713,-0.182898,5.693458,-5.190157,-4.602709],[-8.718980,0.921012,-5.742454,5.096868,-0.026048,-6.571292,6.712079,0.407544,-4.977194,-8.215314]],[[-2.665931,5.281439,-7.846449,-9.243929,-4.697863,-1.073655,4.273248,-2.856495,-0.574184,6.111915],[0.189030,-6.456857,-8.436121,-5.483239,3.112423,-6.222057,-7.168729,-5.155899,-3.622238,8.353888],[-4.075605,-2.256556,-4.557883,3.321480,-6.069052,1.712078,8.583119,-8.761492,9.075329,-4.190845],[7.672188,2.660174,-3.062683,-2.250455,-4.642187,3.301862,5.543966,-8.491367,9.634116,-7.341316],[-4.037554,8.234941,7.755105,2.904670,-9.128416,-1.559304,3.681745,-8.171314,4.926365,-9.617899]],[[-2.093444,-1.029377,0.750375,0.631474,0.718944,-3.639189,3.926054,3.475849,1.157471,-8.067420],[-6.195382,-0.053809,-8.899653,4.518576,6.311317,6.421804,9.964127,-7.351713,-2.570202,4.985791],[7.015100,8.552355,1.173030,-7.206285,9.319220,0.182016,8.104322,1.051528,-9.941143,1.272887],[2.604770,1.280099,4.397133,6.144976,-4.649909,9.359894,7.282283,-5.329691,1.328370,5.735864],[1.152342,-2.086574,-0.707378,-0.763585,-8.634259,9.607094,-4.044379,-7.609965,-2.253923,5.323051]],[[8.016641,-6.581034,9.471692,5.207731,5.276077,-6.014354,8.706502,0.256158,-0.991357,-6.229621],[-9.629602,-2.064569,9.315773,-2.295515,8.715563,-4.375606,-3.332331,4.079084,1.204593,-5.404384],[4.366261,-7.073041,1.997810,3.732307,4.431269,-5.489273,2.493292,1.566443,0.469302,-3.117608],[9.277636,3.671863,3.374442,-2.991414,-8.352744,-1.470505,2.577865,-9.758232,-7.585825,-5.476526],[-5.641121,3.444552,-4.194598,-3.102428,-7.077309,-8.874148,0.334728,0.596993,-1.009511,4.723294]],[[-0.646166,-8.645930,-9.511207,-9.983593,-5.185337,4.336878,9.730589,-7.606792,3.570646,-1.252807],[6.988824,4.824194,-6.170477,-0.756386,-3.866582,1.288898,7.586755,-2.913859,6.535911,-9.928924],[-3.702876,-1.730942,4.227763,0.747281,8.861062,-0.877169,-8.189883,5.333607,3.264291,9.839637],[5.731812,-8.949872,-3.296570,3.378826,-9.200848,-2.729849,8.730232,-6.217149,6.761587,1.666717],[8.898337,-1.723388,9.728595,0.008762,3.668792,9.864254,2.697651,4.576390,-0.855167,5.436804]],[[-2.106330,1.564910,-1.554327,-0.211888,9.245133,-4.492062,-8.699283,-5.689410,-1.840702,-3.971242],[-3.097158,9.092012,5.582412,-6.296747,3.573489,-4.830759,1.417350,7.198197,-9.461974,1.391179],[5.777415,-3.884667,-1.116428,8.119761,-8.265240,-1.570658,6.133309,4.689868,-3.144884,8.071102],[5.429156,-4.412936,6.454215,5.049456,7.414652,-4.829017,-1.568770,-3.507993,6.043686,-4.381721],[7.492465,8.672461,-2.378554,-6.233676,-4.540562,-9.261937,5.363986,-2.254638,-7.331287,-3.490478]]], dtype = "float32")#candidate|143|(6, 5, 10)|const|float32
bop_144 = relay.not_equal(var_135.astype('bool'), relay.reshape(const_143.astype('bool'), relay.shape_of(var_135))) # shape=(6, 5, 10)
const_147 = relay.const([[[False,False,False,False,True,True,False,True,True,True],[False,True,False,True,True,False,False,False,True,True],[True,False,False,False,True,True,True,False,True,False],[True,True,False,True,True,True,False,True,False,True],[True,True,False,True,False,False,True,True,False,False]],[[False,True,False,True,False,True,False,True,False,True],[True,True,False,False,False,True,True,False,True,False],[True,False,True,False,True,True,False,False,False,False],[False,True,False,False,False,False,True,False,True,True],[False,True,False,False,True,False,False,False,False,True]],[[False,False,True,True,False,False,True,False,False,True],[False,False,False,False,False,False,False,True,True,False],[True,True,True,True,True,False,False,True,False,False],[False,True,False,True,False,False,True,True,False,True],[True,True,True,True,True,True,False,False,True,True]],[[False,False,False,True,True,True,True,True,True,True],[False,True,False,False,True,True,True,False,False,True],[False,True,False,True,True,False,True,True,False,False],[False,False,False,False,True,True,True,False,True,False],[True,True,False,False,True,True,True,False,True,True]],[[False,True,True,False,True,False,False,False,True,False],[False,True,False,False,True,True,True,False,True,False],[True,True,True,True,True,True,False,False,True,True],[True,True,True,False,False,False,False,False,False,True],[True,False,False,True,False,True,True,True,False,False]],[[False,True,True,True,False,False,False,False,False,False],[True,False,True,False,False,True,False,True,False,False],[False,True,False,True,False,False,False,True,False,False],[False,False,True,True,True,True,False,False,True,False],[False,True,False,True,True,True,True,False,False,True]]], dtype = "bool")#candidate|147|(6, 5, 10)|const|bool
bop_148 = relay.subtract(bop_144.astype('uint8'), relay.reshape(const_147.astype('uint8'), relay.shape_of(bop_144))) # shape=(6, 5, 10)
bop_151 = relay.greater_equal(const_147.astype('bool'), relay.reshape(bop_144.astype('bool'), relay.shape_of(const_147))) # shape=(6, 5, 10)
output = relay.Tuple([bop_140,bop_148,bop_151,])
output2 = relay.Tuple([bop_140,bop_148,bop_151,])
func_154 = relay.Function([var_134,var_135,var_139,], output)
mod['func_154'] = func_154
mod = relay.transform.InferType()(mod)
var_155 = relay.var("var_155", dtype = "float32", shape = ())#candidate|155|()|var|float32
var_156 = relay.var("var_156", dtype = "float32", shape = (6, 5, 10))#candidate|156|(6, 5, 10)|var|float32
var_157 = relay.var("var_157", dtype = "float32", shape = (6, 5, 10))#candidate|157|(6, 5, 10)|var|float32
output = func_154(var_155,var_156,var_157,)
func_158 = relay.Function([var_155,var_156,var_157,], output)
mutated_mod['func_158'] = func_158
mutated_mod = relay.transform.InferType()(mutated_mod)
var_160 = relay.var("var_160", dtype = "int8", shape = (3, 14, 10))#candidate|160|(3, 14, 10)|var|int8
var_161 = relay.var("var_161", dtype = "int8", shape = (3, 14, 10))#candidate|161|(3, 14, 10)|var|int8
bop_162 = relay.bitwise_and(var_160.astype('int8'), relay.reshape(var_161.astype('int8'), relay.shape_of(var_160))) # shape=(3, 14, 10)
uop_165 = relay.log10(var_161.astype('float64')) # shape=(3, 14, 10)
uop_167 = relay.tan(uop_165.astype('float32')) # shape=(3, 14, 10)
output = relay.Tuple([bop_162,uop_167,])
output2 = relay.Tuple([bop_162,uop_167,])
func_169 = relay.Function([var_160,var_161,], output)
mod['func_169'] = func_169
mod = relay.transform.InferType()(mod)
mutated_mod['func_169'] = func_169
mutated_mod = relay.transform.InferType()(mutated_mod)
func_169_call = mutated_mod.get_global_var('func_169')
var_171 = relay.var("var_171", dtype = "int8", shape = (3, 14, 10))#candidate|171|(3, 14, 10)|var|int8
var_172 = relay.var("var_172", dtype = "int8", shape = (3, 14, 10))#candidate|172|(3, 14, 10)|var|int8
call_170 = func_169_call(var_171,var_172,)
output = call_170
func_173 = relay.Function([var_171,var_172,], output)
mutated_mod['func_173'] = func_173
mutated_mod = relay.transform.InferType()(mutated_mod)
var_175 = relay.var("var_175", dtype = "float64", shape = ())#candidate|175|()|var|float64
var_176 = relay.var("var_176", dtype = "float64", shape = (6, 9, 8))#candidate|176|(6, 9, 8)|var|float64
bop_177 = relay.mod(var_175.astype('float64'), var_176.astype('float64')) # shape=(6, 9, 8)
uop_180 = relay.acos(bop_177.astype('float32')) # shape=(6, 9, 8)
uop_182 = relay.erf(uop_180.astype('float32')) # shape=(6, 9, 8)
uop_184 = relay.sigmoid(uop_182.astype('float64')) # shape=(6, 9, 8)
uop_186 = relay.sqrt(uop_180.astype('float64')) # shape=(6, 9, 8)
output = relay.Tuple([uop_184,uop_186,])
output2 = relay.Tuple([uop_184,uop_186,])
func_188 = relay.Function([var_175,var_176,], output)
mod['func_188'] = func_188
mod = relay.transform.InferType()(mod)
var_189 = relay.var("var_189", dtype = "float64", shape = ())#candidate|189|()|var|float64
var_190 = relay.var("var_190", dtype = "float64", shape = (6, 9, 8))#candidate|190|(6, 9, 8)|var|float64
output = func_188(var_189,var_190,)
func_191 = relay.Function([var_189,var_190,], output)
mutated_mod['func_191'] = func_191
mutated_mod = relay.transform.InferType()(mutated_mod)
var_193 = relay.var("var_193", dtype = "uint16", shape = ())#candidate|193|()|var|uint16
var_194 = relay.var("var_194", dtype = "uint16", shape = ())#candidate|194|()|var|uint16
bop_195 = relay.not_equal(var_193.astype('bool'), var_194.astype('bool')) # shape=()
uop_198 = relay.tan(var_194.astype('float32')) # shape=()
bop_200 = relay.equal(var_194.astype('bool'), bop_195.astype('bool')) # shape=()
bop_203 = relay.equal(var_194.astype('bool'), bop_200.astype('bool')) # shape=()
uop_206 = relay.exp(uop_198.astype('float64')) # shape=()
output = relay.Tuple([bop_203,uop_206,])
output2 = relay.Tuple([bop_203,uop_206,])
func_208 = relay.Function([var_193,var_194,], output)
mod['func_208'] = func_208
mod = relay.transform.InferType()(mod)
mutated_mod['func_208'] = func_208
mutated_mod = relay.transform.InferType()(mutated_mod)
func_208_call = mutated_mod.get_global_var('func_208')
var_210 = relay.var("var_210", dtype = "uint16", shape = ())#candidate|210|()|var|uint16
var_211 = relay.var("var_211", dtype = "uint16", shape = ())#candidate|211|()|var|uint16
call_209 = func_208_call(var_210,var_211,)
output = call_209
func_212 = relay.Function([var_210,var_211,], output)
mutated_mod['func_212'] = func_212
mutated_mod = relay.transform.InferType()(mutated_mod)
var_214 = relay.var("var_214", dtype = "float32", shape = (5, 8))#candidate|214|(5, 8)|var|float32
uop_215 = relay.sinh(var_214.astype('float32')) # shape=(5, 8)
uop_217 = relay.asinh(uop_215.astype('float32')) # shape=(5, 8)
output = uop_217
output2 = uop_217
func_219 = relay.Function([var_214,], output)
mod['func_219'] = func_219
mod = relay.transform.InferType()(mod)
var_220 = relay.var("var_220", dtype = "float32", shape = (5, 8))#candidate|220|(5, 8)|var|float32
output = func_219(var_220)
func_221 = relay.Function([var_220], output)
mutated_mod['func_221'] = func_221
mutated_mod = relay.transform.InferType()(mutated_mod)
var_223 = relay.var("var_223", dtype = "float32", shape = (7, 12))#candidate|223|(7, 12)|var|float32
uop_224 = relay.atan(var_223.astype('float32')) # shape=(7, 12)
uop_226 = relay.log2(uop_224.astype('float64')) # shape=(7, 12)
bop_228 = relay.power(uop_226.astype('float32'), relay.reshape(var_223.astype('float32'), relay.shape_of(uop_226))) # shape=(7, 12)
bop_231 = relay.left_shift(bop_228.astype('uint16'), relay.reshape(var_223.astype('uint16'), relay.shape_of(bop_228))) # shape=(7, 12)
const_234 = relay.const([[2,8,-2,2,1,-6,-8,6,-9,-7,5,-9],[2,-10,4,8,-7,-3,-7,1,-10,1,-7,6],[-5,10,-5,-6,1,-7,-3,-2,-3,-1,7,-8],[-8,-5,-7,1,-7,2,10,-10,-6,-7,-4,-3],[-1,5,6,-6,-4,1,3,-3,-3,-2,-3,-3],[1,-9,-6,-2,-7,-6,2,-9,-3,10,-2,-6],[-8,-7,3,1,9,-9,-4,-3,-9,6,-1,8]], dtype = "uint16")#candidate|234|(7, 12)|const|uint16
bop_235 = relay.add(bop_231.astype('float64'), relay.reshape(const_234.astype('float64'), relay.shape_of(bop_231))) # shape=(7, 12)
uop_238 = relay.log10(var_223.astype('float64')) # shape=(7, 12)
uop_240 = relay.sin(uop_238.astype('float32')) # shape=(7, 12)
var_242 = relay.var("var_242", dtype = "float64", shape = (7, 12))#candidate|242|(7, 12)|var|float64
bop_243 = relay.floor_divide(uop_238.astype('float32'), relay.reshape(var_242.astype('float32'), relay.shape_of(uop_238))) # shape=(7, 12)
bop_246 = relay.floor_mod(bop_243.astype('float64'), relay.reshape(bop_228.astype('float64'), relay.shape_of(bop_243))) # shape=(7, 12)
bop_249 = relay.mod(bop_243.astype('float64'), relay.reshape(const_234.astype('float64'), relay.shape_of(bop_243))) # shape=(7, 12)
output = relay.Tuple([bop_235,uop_240,bop_246,bop_249,])
output2 = relay.Tuple([bop_235,uop_240,bop_246,bop_249,])
func_252 = relay.Function([var_223,var_242,], output)
mod['func_252'] = func_252
mod = relay.transform.InferType()(mod)
mutated_mod['func_252'] = func_252
mutated_mod = relay.transform.InferType()(mutated_mod)
func_252_call = mutated_mod.get_global_var('func_252')
var_254 = relay.var("var_254", dtype = "float32", shape = (7, 12))#candidate|254|(7, 12)|var|float32
var_255 = relay.var("var_255", dtype = "float64", shape = (7, 12))#candidate|255|(7, 12)|var|float64
call_253 = func_252_call(var_254,var_255,)
output = call_253
func_256 = relay.Function([var_254,var_255,], output)
mutated_mod['func_256'] = func_256
mutated_mod = relay.transform.InferType()(mutated_mod)
var_258 = relay.var("var_258", dtype = "float64", shape = (9, 14, 6))#candidate|258|(9, 14, 6)|var|float64
uop_259 = relay.asin(var_258.astype('float64')) # shape=(9, 14, 6)
output = relay.Tuple([uop_259,])
output2 = relay.Tuple([uop_259,])
func_261 = relay.Function([var_258,], output)
mod['func_261'] = func_261
mod = relay.transform.InferType()(mod)
var_262 = relay.var("var_262", dtype = "float64", shape = (9, 14, 6))#candidate|262|(9, 14, 6)|var|float64
output = func_261(var_262)
func_263 = relay.Function([var_262], output)
mutated_mod['func_263'] = func_263
mutated_mod = relay.transform.InferType()(mutated_mod)
const_265 = relay.const([[[-3.004978,5.024064,4.995852,5.215011,-6.510349,-2.973925],[-0.918125,-3.249845,1.062443,-7.428908,-8.330055,0.792244],[8.950240,-7.612703,0.448817,-8.496284,-0.681276,0.138517],[0.850969,2.460724,1.278234,7.920194,8.192688,-3.570281],[-2.968955,5.775928,5.286119,-5.480693,-6.407677,-4.212724],[-0.790621,9.103504,2.033019,4.624563,-1.607715,-8.223063]],[[-4.409726,3.090375,-8.284212,9.570202,-0.827647,0.712098],[3.451660,-5.779858,-8.934710,-3.732742,0.751666,2.575440],[-6.939380,2.909948,9.058188,-8.340282,8.214184,6.704194],[0.533879,-4.226756,4.175316,-3.465219,-1.997284,-8.041426],[2.435520,0.766088,5.071660,5.045050,9.439470,-2.682072],[9.846768,6.658909,-6.209573,-3.410868,3.468492,1.911080]],[[-0.473772,5.925997,6.848095,3.360478,-5.814230,0.274613],[2.197536,2.768931,-3.769755,0.477547,-3.048431,-9.846746],[-1.918262,4.338509,-0.884454,-8.008556,-4.450274,7.630362],[-2.192810,-1.227180,3.209241,-4.233313,5.087108,-8.671572],[2.567804,8.674816,-1.276455,-5.859853,-6.141186,-9.239648],[-6.582379,4.476428,-1.306127,-7.644932,3.607853,-0.326962]],[[4.539757,1.681361,-1.546452,-5.047507,0.362666,-8.294113],[5.781784,-9.543032,-4.799489,-1.604236,-8.378160,2.840748],[-7.735077,2.914440,-6.716011,1.665185,-0.308332,-9.741569],[-8.410450,-7.532169,-0.265211,5.941155,2.929827,1.972324],[-3.673045,-1.574418,7.370543,5.234909,-1.593073,0.830101],[-3.388802,-1.369904,8.788265,-8.511922,4.377443,-9.681422]],[[9.230104,3.446770,7.148952,-2.538266,6.058888,-5.254360],[-3.555697,-5.630538,-1.215245,8.016369,-5.215878,5.764366],[3.061448,0.691732,-9.528436,-8.722237,3.739443,-5.601832],[5.568204,-2.832029,-2.200772,9.712621,9.559124,5.761074],[-3.734219,-4.893419,-8.143174,-8.069383,6.468494,-5.522121],[8.641747,9.169632,-5.254489,-7.948653,-2.292363,-4.512651]]], dtype = "float64")#candidate|265|(5, 6, 6)|const|float64
uop_266 = relay.acosh(const_265.astype('float64')) # shape=(5, 6, 6)
uop_268 = relay.sin(const_265.astype('float32')) # shape=(5, 6, 6)
bop_270 = relay.greater(uop_266.astype('bool'), relay.reshape(uop_268.astype('bool'), relay.shape_of(uop_266))) # shape=(5, 6, 6)
uop_273 = relay.sin(uop_268.astype('float32')) # shape=(5, 6, 6)
uop_275 = relay.acos(uop_273.astype('float64')) # shape=(5, 6, 6)
var_277 = relay.var("var_277", dtype = "float32", shape = (5, 6, 6))#candidate|277|(5, 6, 6)|var|float32
bop_278 = relay.less(uop_273.astype('bool'), relay.reshape(var_277.astype('bool'), relay.shape_of(uop_273))) # shape=(5, 6, 6)
uop_281 = relay.acosh(uop_275.astype('float32')) # shape=(5, 6, 6)
uop_283 = relay.asin(uop_275.astype('float32')) # shape=(5, 6, 6)
uop_285 = relay.log10(bop_278.astype('float64')) # shape=(5, 6, 6)
var_287 = relay.var("var_287", dtype = "float64", shape = (5, 6, 6))#candidate|287|(5, 6, 6)|var|float64
bop_288 = relay.right_shift(uop_285.astype('int64'), relay.reshape(var_287.astype('int64'), relay.shape_of(uop_285))) # shape=(5, 6, 6)
uop_291 = relay.cos(uop_281.astype('float32')) # shape=(5, 6, 6)
uop_293 = relay.sinh(uop_291.astype('float64')) # shape=(5, 6, 6)
uop_295 = relay.log2(uop_293.astype('float32')) # shape=(5, 6, 6)
uop_297 = relay.sinh(uop_295.astype('float32')) # shape=(5, 6, 6)
bop_299 = relay.bitwise_xor(uop_297.astype('uint16'), relay.reshape(uop_293.astype('uint16'), relay.shape_of(uop_297))) # shape=(5, 6, 6)
func_28_call = mod.get_global_var('func_28')
func_32_call = mutated_mod.get_global_var('func_32')
var_303 = relay.var("var_303", dtype = "float64", shape = (16,))#candidate|303|(16,)|var|float64
call_302 = relay.TupleGetItem(func_28_call(relay.reshape(var_303.astype('float64'), [4, 4]), relay.reshape(var_303.astype('float64'), [4, 4]), ), 0)
call_304 = relay.TupleGetItem(func_32_call(relay.reshape(var_303.astype('float64'), [4, 4]), relay.reshape(var_303.astype('float64'), [4, 4]), ), 0)
const_305 = relay.const([[[0.489882,1.205160,-2.967335,5.257195,-6.902272,2.056338],[2.750428,1.302005,-8.977551,-9.475309,-4.506531,3.289066],[9.968637,-3.593314,5.431902,4.669467,2.728013,-9.167858],[-1.668393,-0.292760,-7.620157,9.986601,7.137107,-3.434539],[3.275280,-0.938733,-7.903878,2.939337,-4.423370,8.381339],[5.516023,-6.152107,5.094109,-7.324906,0.987831,-4.132627]],[[5.217735,4.309215,-0.700751,-4.215871,-5.490967,8.902147],[0.430466,9.199693,-6.613723,8.246701,6.696968,-4.038055],[-6.311352,-4.762675,5.340421,8.562455,-4.252081,2.362493],[-0.903365,-0.306852,9.226774,-4.125587,-7.836897,-3.940447],[-3.200168,5.527173,-4.217419,-4.444842,-6.200629,-0.411339],[8.210476,-6.309318,-6.485962,-4.679826,-0.732608,6.400093]],[[-0.556562,-9.684996,3.509415,8.137950,-9.900478,7.081050],[4.493399,3.339929,-0.708086,4.450275,7.412226,3.245719],[-0.909427,0.633918,-6.969271,-5.375112,-0.961236,-6.649856],[3.856634,8.527365,-6.890400,-8.018988,-5.645139,6.344997],[4.573466,-3.651880,6.637903,-5.850002,0.063105,-2.756962],[2.849502,-1.847404,0.076647,0.652770,-8.159018,-8.185959]],[[-7.475678,-3.539517,-1.956951,1.629585,4.032083,-6.422472],[-7.029303,-6.829137,-6.933113,6.365376,3.965011,2.824305],[-8.231229,5.553758,8.352872,0.812001,2.907410,-5.694636],[-2.161761,9.552297,2.230400,-7.889210,3.458958,6.227418],[2.499323,3.711009,-8.576842,6.209751,8.123350,-3.039069],[-9.222825,-2.816598,4.614665,7.107963,8.412161,7.698172]],[[-9.308378,-0.057500,-1.669243,-2.774774,-5.315163,-0.506107],[-9.487320,-4.912580,3.484639,1.086247,-6.020940,8.353067],[-1.426330,8.013624,3.923511,3.304282,-3.530998,7.013646],[7.706821,-5.575705,2.644981,3.459074,-2.342081,-3.250743],[-9.467889,2.054689,2.665375,-7.772888,5.060636,5.464553],[2.866628,8.066223,-8.998594,-3.937107,-3.033475,-6.110757]]], dtype = "float32")#candidate|305|(5, 6, 6)|const|float32
bop_306 = relay.subtract(uop_297.astype('int8'), relay.reshape(const_305.astype('int8'), relay.shape_of(uop_297))) # shape=(5, 6, 6)
bop_309 = relay.maximum(bop_299.astype('float64'), relay.reshape(uop_291.astype('float64'), relay.shape_of(bop_299))) # shape=(5, 6, 6)
func_208_call = mod.get_global_var('func_208')
func_212_call = mutated_mod.get_global_var('func_212')
const_313 = relay.const(-7, dtype = "uint16")#candidate|313|()|const|uint16
call_312 = relay.TupleGetItem(func_208_call(relay.reshape(const_313.astype('uint16'), []), relay.reshape(const_313.astype('uint16'), []), ), 1)
call_314 = relay.TupleGetItem(func_212_call(relay.reshape(const_313.astype('uint16'), []), relay.reshape(const_313.astype('uint16'), []), ), 1)
bop_315 = relay.divide(uop_295.astype('float64'), relay.reshape(bop_299.astype('float64'), relay.shape_of(uop_295))) # shape=(5, 6, 6)
bop_318 = relay.not_equal(uop_295.astype('bool'), relay.reshape(uop_268.astype('bool'), relay.shape_of(uop_295))) # shape=(5, 6, 6)
bop_321 = relay.floor_mod(uop_293.astype('float32'), relay.reshape(bop_270.astype('float32'), relay.shape_of(uop_293))) # shape=(5, 6, 6)
bop_324 = relay.logical_and(uop_291.astype('bool'), relay.reshape(bop_318.astype('bool'), relay.shape_of(uop_291))) # shape=(5, 6, 6)
bop_327 = relay.subtract(bop_324.astype('int32'), relay.reshape(uop_275.astype('int32'), relay.shape_of(bop_324))) # shape=(5, 6, 6)
uop_330 = relay.acos(bop_299.astype('float64')) # shape=(5, 6, 6)
bop_332 = relay.greater(uop_330.astype('bool'), relay.reshape(bop_309.astype('bool'), relay.shape_of(uop_330))) # shape=(5, 6, 6)
uop_335 = relay.sigmoid(uop_330.astype('float64')) # shape=(5, 6, 6)
uop_337 = relay.asinh(uop_335.astype('float32')) # shape=(5, 6, 6)
uop_339 = relay.sigmoid(bop_332.astype('float32')) # shape=(5, 6, 6)
uop_341 = relay.cos(bop_332.astype('float32')) # shape=(5, 6, 6)
output = relay.Tuple([uop_283,bop_288,call_302,var_303,bop_306,call_312,const_313,bop_315,bop_321,bop_327,uop_337,uop_339,uop_341,])
output2 = relay.Tuple([uop_283,bop_288,call_304,var_303,bop_306,call_314,const_313,bop_315,bop_321,bop_327,uop_337,uop_339,uop_341,])
func_343 = relay.Function([var_277,var_287,var_303,], output)
mod['func_343'] = func_343
mod = relay.transform.InferType()(mod)
var_344 = relay.var("var_344", dtype = "float32", shape = (5, 6, 6))#candidate|344|(5, 6, 6)|var|float32
var_345 = relay.var("var_345", dtype = "float64", shape = (5, 6, 6))#candidate|345|(5, 6, 6)|var|float64
var_346 = relay.var("var_346", dtype = "float64", shape = (16,))#candidate|346|(16,)|var|float64
output = func_343(var_344,var_345,var_346,)
func_347 = relay.Function([var_344,var_345,var_346,], output)
mutated_mod['func_347'] = func_347
mutated_mod = relay.transform.InferType()(mutated_mod)
var_349 = relay.var("var_349", dtype = "float64", shape = (14,))#candidate|349|(14,)|var|float64
uop_350 = relay.tan(var_349.astype('float64')) # shape=(14,)
uop_352 = relay.rsqrt(uop_350.astype('float64')) # shape=(14,)
bop_354 = relay.greater(uop_350.astype('bool'), relay.reshape(uop_352.astype('bool'), relay.shape_of(uop_350))) # shape=(14,)
bop_357 = relay.logical_and(bop_354.astype('bool'), relay.reshape(uop_352.astype('bool'), relay.shape_of(bop_354))) # shape=(14,)
bop_360 = relay.multiply(bop_357.astype('float32'), relay.reshape(uop_350.astype('float32'), relay.shape_of(bop_357))) # shape=(14,)
output = bop_360
output2 = bop_360
F = relay.Function([var_349,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_349,], output2)
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
input_349= np.array([-6.978770,-4.823430,2.108862,7.647959,-1.078706,-8.322586,-4.347443,6.813698,-2.603381,-6.041074,8.591941,-1.771886,3.241017,0.245506], dtype='float64')
module1.set_input('var_349', input_349)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_349, )
res3 = intrp3.evaluate()(input_349, )
res4 = intrp4.evaluate()(input_349, )
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
module5.set_input('var_349', input_349)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_349, )
res7 = intrp7.evaluate()(input_349, )
res8 = intrp8.evaluate()(input_349, )
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
module9.set_input('var_349', input_349)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_349, )
res11 = intrp11.evaluate()(input_349, )
res12 = intrp12.evaluate()(input_349, )
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
module13.set_input('var_349', input_349)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_349, )
res15 = intrp15.evaluate()(input_349, )
res16 = intrp16.evaluate()(input_349, )
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
module17.set_input('var_349', input_349)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_349, )
res19 = intrp19.evaluate()(input_349, )
res20 = intrp20.evaluate()(input_349, )
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
module21.set_input('var_349', input_349)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_349, )
res23 = intrp23.evaluate()(input_349, )
res24 = intrp24.evaluate()(input_349, )
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

'''25: TVMFuncCall
24: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
23: tvm::transform::Pass::operator()(tvm::IRModule) const
22: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
21: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
20: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
19: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
17: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
16: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
15: _ZNSt17_Function_handlerIFSt10sha
14: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
13: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''