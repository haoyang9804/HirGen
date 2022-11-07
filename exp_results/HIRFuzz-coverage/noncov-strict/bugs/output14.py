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
var_0 = relay.var("var_0", dtype = "float64", shape = (5,))#candidate|0|(5,)|var|float64
var_1 = relay.var("var_1", dtype = "float64", shape = (5,))#candidate|1|(5,)|var|float64
bop_2 = relay.power(var_0.astype('float64'), relay.reshape(var_1.astype('float64'), relay.shape_of(var_0))) # shape=(5,)
bop_5 = relay.bitwise_and(bop_2.astype('int8'), relay.reshape(var_0.astype('int8'), relay.shape_of(bop_2))) # shape=(5,)
uop_8 = relay.atan(var_0.astype('float32')) # shape=(5,)
var_10 = relay.var("var_10", dtype = "float64", shape = (5,))#candidate|10|(5,)|var|float64
bop_11 = relay.bitwise_or(var_0.astype('int32'), relay.reshape(var_10.astype('int32'), relay.shape_of(var_0))) # shape=(5,)
uop_14 = relay.cos(bop_2.astype('float64')) # shape=(5,)
uop_16 = relay.asinh(var_0.astype('float64')) # shape=(5,)
uop_18 = relay.atanh(uop_14.astype('float64')) # shape=(5,)
uop_20 = relay.asinh(uop_16.astype('float64')) # shape=(5,)
uop_22 = relay.atanh(uop_18.astype('float64')) # shape=(5,)
uop_24 = relay.sinh(uop_22.astype('float32')) # shape=(5,)
bop_26 = relay.floor_mod(uop_8.astype('float32'), relay.reshape(uop_18.astype('float32'), relay.shape_of(uop_8))) # shape=(5,)
var_29 = relay.var("var_29", dtype = "float64", shape = (5,))#candidate|29|(5,)|var|float64
bop_30 = relay.bitwise_and(uop_22.astype('int64'), relay.reshape(var_29.astype('int64'), relay.shape_of(uop_22))) # shape=(5,)
uop_33 = relay.log2(uop_24.astype('float64')) # shape=(5,)
var_35 = relay.var("var_35", dtype = "float64", shape = (5,))#candidate|35|(5,)|var|float64
bop_36 = relay.maximum(uop_33.astype('int64'), relay.reshape(var_35.astype('int64'), relay.shape_of(uop_33))) # shape=(5,)
var_39 = relay.var("var_39", dtype = "float64", shape = (5,))#candidate|39|(5,)|var|float64
bop_40 = relay.bitwise_or(uop_22.astype('uint8'), relay.reshape(var_39.astype('uint8'), relay.shape_of(uop_22))) # shape=(5,)
bop_43 = relay.logical_or(uop_33.astype('bool'), relay.reshape(var_29.astype('bool'), relay.shape_of(uop_33))) # shape=(5,)
var_46 = relay.var("var_46", dtype = "float32", shape = (5,))#candidate|46|(5,)|var|float32
bop_47 = relay.minimum(uop_24.astype('uint32'), relay.reshape(var_46.astype('uint32'), relay.shape_of(uop_24))) # shape=(5,)
uop_50 = relay.rsqrt(uop_24.astype('float64')) # shape=(5,)
uop_52 = relay.log(uop_24.astype('float64')) # shape=(5,)
bop_54 = relay.equal(uop_33.astype('bool'), relay.reshape(var_29.astype('bool'), relay.shape_of(uop_33))) # shape=(5,)
uop_57 = relay.sinh(uop_24.astype('float32')) # shape=(5,)
uop_59 = relay.log10(bop_43.astype('float32')) # shape=(5,)
bop_61 = relay.maximum(uop_59.astype('uint32'), relay.reshape(bop_2.astype('uint32'), relay.shape_of(uop_59))) # shape=(5,)
bop_64 = relay.equal(uop_18.astype('bool'), relay.reshape(bop_43.astype('bool'), relay.shape_of(uop_18))) # shape=(5,)
bop_67 = relay.not_equal(bop_54.astype('bool'), relay.reshape(bop_61.astype('bool'), relay.shape_of(bop_54))) # shape=(5,)
bop_70 = relay.subtract(uop_24.astype('int8'), relay.reshape(uop_8.astype('int8'), relay.shape_of(uop_24))) # shape=(5,)
uop_73 = relay.asin(uop_59.astype('float32')) # shape=(5,)
uop_75 = relay.log10(uop_73.astype('float32')) # shape=(5,)
uop_77 = relay.sqrt(uop_73.astype('float32')) # shape=(5,)
var_79 = relay.var("var_79", dtype = "float32", shape = (5,))#candidate|79|(5,)|var|float32
bop_80 = relay.not_equal(uop_73.astype('bool'), relay.reshape(var_79.astype('bool'), relay.shape_of(uop_73))) # shape=(5,)
bop_83 = relay.divide(uop_77.astype('float64'), relay.reshape(var_46.astype('float64'), relay.shape_of(uop_77))) # shape=(5,)
uop_86 = relay.erf(bop_83.astype('float64')) # shape=(5,)
uop_88 = relay.cosh(uop_86.astype('float32')) # shape=(5,)
uop_90 = relay.log2(uop_86.astype('float64')) # shape=(5,)
uop_92 = relay.log(uop_86.astype('float32')) # shape=(5,)
bop_94 = relay.minimum(uop_92.astype('uint64'), relay.reshape(uop_52.astype('uint64'), relay.shape_of(uop_92))) # shape=(5,)
output = relay.Tuple([bop_5,bop_11,uop_20,bop_26,bop_30,bop_36,bop_40,bop_47,uop_50,uop_57,bop_64,bop_67,bop_70,uop_75,bop_80,uop_88,uop_90,bop_94,])
output2 = relay.Tuple([bop_5,bop_11,uop_20,bop_26,bop_30,bop_36,bop_40,bop_47,uop_50,uop_57,bop_64,bop_67,bop_70,uop_75,bop_80,uop_88,uop_90,bop_94,])
func_97 = relay.Function([var_0,var_1,var_10,var_29,var_35,var_39,var_46,var_79,], output)
mod['func_97'] = func_97
mod = relay.transform.InferType()(mod)
mutated_mod['func_97'] = func_97
mutated_mod = relay.transform.InferType()(mutated_mod)
func_97_call = mutated_mod.get_global_var('func_97')
var_99 = relay.var("var_99", dtype = "float64", shape = (5,))#candidate|99|(5,)|var|float64
var_100 = relay.var("var_100", dtype = "float64", shape = (5,))#candidate|100|(5,)|var|float64
var_101 = relay.var("var_101", dtype = "float64", shape = (5,))#candidate|101|(5,)|var|float64
var_102 = relay.var("var_102", dtype = "float64", shape = (5,))#candidate|102|(5,)|var|float64
var_103 = relay.var("var_103", dtype = "float64", shape = (5,))#candidate|103|(5,)|var|float64
var_104 = relay.var("var_104", dtype = "float64", shape = (5,))#candidate|104|(5,)|var|float64
var_105 = relay.var("var_105", dtype = "float32", shape = (5,))#candidate|105|(5,)|var|float32
var_106 = relay.var("var_106", dtype = "float32", shape = (5,))#candidate|106|(5,)|var|float32
call_98 = func_97_call(var_99,var_100,var_101,var_102,var_103,var_104,var_105,var_106,)
output = call_98
func_107 = relay.Function([var_99,var_100,var_101,var_102,var_103,var_104,var_105,var_106,], output)
mutated_mod['func_107'] = func_107
mutated_mod = relay.transform.InferType()(mutated_mod)
var_109 = relay.var("var_109", dtype = "uint8", shape = (10,))#candidate|109|(10,)|var|uint8
var_110 = relay.var("var_110", dtype = "uint8", shape = (10,))#candidate|110|(10,)|var|uint8
bop_111 = relay.bitwise_and(var_109.astype('uint8'), relay.reshape(var_110.astype('uint8'), relay.shape_of(var_109))) # shape=(10,)
bop_114 = relay.not_equal(bop_111.astype('bool'), relay.reshape(var_110.astype('bool'), relay.shape_of(bop_111))) # shape=(10,)
uop_117 = relay.tan(var_110.astype('float32')) # shape=(10,)
const_119 = relay.const([-8.463113,-0.521413,-0.390424,4.156781,5.605933,0.570599,-5.324546,5.975733,5.372441,9.432070], dtype = "float32")#candidate|119|(10,)|const|float32
bop_120 = relay.not_equal(uop_117.astype('bool'), relay.reshape(const_119.astype('bool'), relay.shape_of(uop_117))) # shape=(10,)
uop_123 = relay.sqrt(bop_120.astype('float32')) # shape=(10,)
output = relay.Tuple([bop_114,uop_123,])
output2 = relay.Tuple([bop_114,uop_123,])
func_125 = relay.Function([var_109,var_110,], output)
mod['func_125'] = func_125
mod = relay.transform.InferType()(mod)
mutated_mod['func_125'] = func_125
mutated_mod = relay.transform.InferType()(mutated_mod)
func_125_call = mutated_mod.get_global_var('func_125')
var_127 = relay.var("var_127", dtype = "uint8", shape = (10,))#candidate|127|(10,)|var|uint8
var_128 = relay.var("var_128", dtype = "uint8", shape = (10,))#candidate|128|(10,)|var|uint8
call_126 = func_125_call(var_127,var_128,)
output = call_126
func_129 = relay.Function([var_127,var_128,], output)
mutated_mod['func_129'] = func_129
mutated_mod = relay.transform.InferType()(mutated_mod)
var_131 = relay.var("var_131", dtype = "uint64", shape = (12, 3, 1))#candidate|131|(12, 3, 1)|var|uint64
var_132 = relay.var("var_132", dtype = "uint64", shape = (12, 3, 8))#candidate|132|(12, 3, 8)|var|uint64
bop_133 = relay.bitwise_and(var_131.astype('uint64'), var_132.astype('uint64')) # shape=(12, 3, 8)
var_136 = relay.var("var_136", dtype = "uint64", shape = (12, 3, 13))#candidate|136|(12, 3, 13)|var|uint64
bop_137 = relay.power(var_131.astype('float32'), var_136.astype('float32')) # shape=(12, 3, 13)
uop_140 = relay.sqrt(bop_137.astype('float64')) # shape=(12, 3, 13)
bop_142 = relay.multiply(var_136.astype('int64'), relay.reshape(uop_140.astype('int64'), relay.shape_of(var_136))) # shape=(12, 3, 13)
bop_145 = relay.logical_and(bop_142.astype('bool'), var_131.astype('bool')) # shape=(12, 3, 13)
uop_148 = relay.log(bop_133.astype('float32')) # shape=(12, 3, 8)
bop_150 = relay.logical_xor(bop_137.astype('uint64'), relay.reshape(uop_140.astype('uint64'), relay.shape_of(bop_137))) # shape=(12, 3, 13)
const_153 = relay.const([[[4,-7,-3,6,1,10,2,-10,1,-3,8,-7,-3],[4,-10,3,-9,8,3,-5,-5,-2,-2,-1,2,5],[5,6,6,-1,-7,4,-5,9,8,2,-10,2,9]],[[-6,-9,-8,-4,-2,4,-1,-8,-10,-7,2,-4,9],[-1,7,-4,8,3,10,10,-8,-5,-9,3,-5,10],[-4,-4,-8,3,2,8,5,-7,7,-4,4,-5,2]],[[4,-8,7,-9,-6,5,-2,-9,8,-2,-8,2,7],[3,1,-9,5,9,1,4,-5,-7,9,-9,-9,-3],[1,8,3,7,-7,1,8,-5,-6,-6,-1,9,-8]],[[-8,-7,-3,6,6,3,7,2,1,6,8,-4,-2],[-7,4,9,-4,2,-7,-9,2,-6,1,5,-9,7],[-4,2,-5,-8,4,-10,6,-6,-2,3,8,-1,9]],[[-10,-8,4,7,9,-4,-10,4,9,-3,10,1,-9],[5,7,-9,-10,1,10,-1,9,-6,6,-2,-2,3],[10,7,2,-1,-8,5,9,4,-5,8,2,4,6]],[[6,-10,-1,-10,2,-2,2,-5,-8,4,-4,8,-2],[8,10,-1,3,3,4,2,-7,8,2,8,-8,3],[1,3,-2,1,-4,-8,6,9,-7,3,-3,5,6]],[[7,-8,-5,-10,2,7,5,6,8,-7,1,8,-9],[-6,-8,10,6,6,6,-10,-9,2,10,-9,1,5],[-9,-5,-6,5,-3,9,-2,1,-10,-4,9,3,5]],[[10,-8,-5,10,5,1,9,-4,1,3,2,2,2],[-4,-1,-7,-3,7,-1,-10,-2,6,-2,10,-6,5],[6,-6,1,3,8,-4,-4,3,-8,2,1,-3,-6]],[[-8,10,10,3,4,7,7,-7,10,-1,-7,-5,3],[6,-1,-3,-7,-4,-8,-5,2,-8,-9,1,-6,-4],[3,1,5,4,-10,-7,-1,-5,2,9,-2,-3,4]],[[2,-8,5,6,-9,-5,-9,1,3,-8,1,8,4],[-7,-3,-2,-2,-3,-2,-5,10,-6,-7,6,1,8],[-6,1,8,-10,10,-6,-2,1,8,7,-8,-8,3]],[[6,-1,-5,-3,-8,6,-2,10,1,8,8,-10,5],[-9,2,10,6,-4,-10,-9,10,8,4,-9,-2,6],[5,6,-3,-5,-6,5,-8,-5,8,9,-1,-9,5]],[[-7,6,-8,1,-3,10,9,4,-10,-5,-1,10,2],[-4,-4,8,7,-9,5,8,2,5,-6,7,1,2],[4,9,1,6,6,9,8,-8,-7,10,-6,1,3]]], dtype = "int64")#candidate|153|(12, 3, 13)|const|int64
bop_154 = relay.right_shift(bop_142.astype('int16'), relay.reshape(const_153.astype('int16'), relay.shape_of(bop_142))) # shape=(12, 3, 13)
output = relay.Tuple([bop_145,uop_148,bop_150,bop_154,])
output2 = relay.Tuple([bop_145,uop_148,bop_150,bop_154,])
func_157 = relay.Function([var_131,var_132,var_136,], output)
mod['func_157'] = func_157
mod = relay.transform.InferType()(mod)
var_158 = relay.var("var_158", dtype = "uint64", shape = (12, 3, 1))#candidate|158|(12, 3, 1)|var|uint64
var_159 = relay.var("var_159", dtype = "uint64", shape = (12, 3, 8))#candidate|159|(12, 3, 8)|var|uint64
var_160 = relay.var("var_160", dtype = "uint64", shape = (12, 3, 13))#candidate|160|(12, 3, 13)|var|uint64
output = func_157(var_158,var_159,var_160,)
func_161 = relay.Function([var_158,var_159,var_160,], output)
mutated_mod['func_161'] = func_161
mutated_mod = relay.transform.InferType()(mutated_mod)
const_163 = relay.const(0.956696, dtype = "float32")#candidate|163|()|const|float32
uop_164 = relay.cos(const_163.astype('float32')) # shape=()
const_166 = relay.const([-2.627316,6.237205], dtype = "float32")#candidate|166|(2,)|const|float32
bop_167 = relay.less_equal(uop_164.astype('bool'), const_166.astype('bool')) # shape=(2,)
bop_170 = relay.bitwise_and(bop_167.astype('uint16'), relay.reshape(const_166.astype('uint16'), relay.shape_of(bop_167))) # shape=(2,)
uop_173 = relay.asinh(const_163.astype('float32')) # shape=()
bop_175 = relay.right_shift(uop_164.astype('uint16'), const_163.astype('uint16')) # shape=()
var_178 = relay.var("var_178", dtype = "float32", shape = ())#candidate|178|()|var|float32
bop_179 = relay.mod(uop_164.astype('float32'), var_178.astype('float32')) # shape=()
bop_182 = relay.greater(const_163.astype('bool'), bop_170.astype('bool')) # shape=(2,)
uop_185 = relay.cosh(bop_170.astype('float64')) # shape=(2,)
func_125_call = mod.get_global_var('func_125')
func_129_call = mutated_mod.get_global_var('func_129')
var_188 = relay.var("var_188", dtype = "uint8", shape = (10, 1))#candidate|188|(10, 1)|var|uint8
call_187 = relay.TupleGetItem(func_125_call(relay.reshape(var_188.astype('uint8'), [10,]), relay.reshape(var_188.astype('uint8'), [10,]), ), 0)
call_189 = relay.TupleGetItem(func_129_call(relay.reshape(var_188.astype('uint8'), [10,]), relay.reshape(var_188.astype('uint8'), [10,]), ), 0)
output = relay.Tuple([uop_173,bop_175,bop_179,bop_182,uop_185,call_187,var_188,])
output2 = relay.Tuple([uop_173,bop_175,bop_179,bop_182,uop_185,call_189,var_188,])
func_190 = relay.Function([var_178,var_188,], output)
mod['func_190'] = func_190
mod = relay.transform.InferType()(mod)
mutated_mod['func_190'] = func_190
mutated_mod = relay.transform.InferType()(mutated_mod)
func_190_call = mutated_mod.get_global_var('func_190')
var_192 = relay.var("var_192", dtype = "float32", shape = ())#candidate|192|()|var|float32
var_193 = relay.var("var_193", dtype = "uint8", shape = (10, 1))#candidate|193|(10, 1)|var|uint8
call_191 = func_190_call(var_192,var_193,)
output = call_191
func_194 = relay.Function([var_192,var_193,], output)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
var_196 = relay.var("var_196", dtype = "float32", shape = (13,))#candidate|196|(13,)|var|float32
var_197 = relay.var("var_197", dtype = "float32", shape = (13,))#candidate|197|(13,)|var|float32
bop_198 = relay.greater_equal(var_196.astype('bool'), relay.reshape(var_197.astype('bool'), relay.shape_of(var_196))) # shape=(13,)
bop_201 = relay.floor_divide(var_197.astype('float32'), relay.reshape(var_196.astype('float32'), relay.shape_of(var_197))) # shape=(13,)
uop_204 = relay.tan(bop_201.astype('float32')) # shape=(13,)
uop_206 = relay.cos(uop_204.astype('float32')) # shape=(13,)
uop_208 = relay.asinh(uop_206.astype('float64')) # shape=(13,)
func_190_call = mod.get_global_var('func_190')
func_194_call = mutated_mod.get_global_var('func_194')
var_211 = relay.var("var_211", dtype = "float32", shape = ())#candidate|211|()|var|float32
var_212 = relay.var("var_212", dtype = "uint8", shape = (5, 2))#candidate|212|(5, 2)|var|uint8
call_210 = relay.TupleGetItem(func_190_call(relay.reshape(var_211.astype('float32'), []), relay.reshape(var_212.astype('uint8'), [10, 1]), ), 0)
call_213 = relay.TupleGetItem(func_194_call(relay.reshape(var_211.astype('float32'), []), relay.reshape(var_212.astype('uint8'), [10, 1]), ), 0)
uop_214 = relay.asin(uop_206.astype('float32')) # shape=(13,)
func_157_call = mod.get_global_var('func_157')
func_161_call = mutated_mod.get_global_var('func_161')
var_217 = relay.var("var_217", dtype = "uint64", shape = (36,))#candidate|217|(36,)|var|uint64
var_218 = relay.var("var_218", dtype = "uint64", shape = (288,))#candidate|218|(288,)|var|uint64
var_219 = relay.var("var_219", dtype = "uint64", shape = (468,))#candidate|219|(468,)|var|uint64
call_216 = relay.TupleGetItem(func_157_call(relay.reshape(var_217.astype('uint64'), [12, 3, 1]), relay.reshape(var_218.astype('uint64'), [12, 3, 8]), relay.reshape(var_219.astype('uint64'), [12, 3, 13]), ), 3)
call_220 = relay.TupleGetItem(func_161_call(relay.reshape(var_217.astype('uint64'), [12, 3, 1]), relay.reshape(var_218.astype('uint64'), [12, 3, 8]), relay.reshape(var_219.astype('uint64'), [12, 3, 13]), ), 3)
uop_221 = relay.erf(uop_208.astype('float64')) # shape=(13,)
var_223 = relay.var("var_223", dtype = "float32", shape = (13,))#candidate|223|(13,)|var|float32
bop_224 = relay.subtract(uop_204.astype('uint16'), relay.reshape(var_223.astype('uint16'), relay.shape_of(uop_204))) # shape=(13,)
var_227 = relay.var("var_227", dtype = "float64", shape = (13,))#candidate|227|(13,)|var|float64
bop_228 = relay.divide(uop_208.astype('float64'), relay.reshape(var_227.astype('float64'), relay.shape_of(uop_208))) # shape=(13,)
var_231 = relay.var("var_231", dtype = "float64", shape = (13,))#candidate|231|(13,)|var|float64
bop_232 = relay.less_equal(uop_221.astype('bool'), relay.reshape(var_231.astype('bool'), relay.shape_of(uop_221))) # shape=(13,)
uop_235 = relay.rsqrt(bop_228.astype('float32')) # shape=(13,)
bop_237 = relay.left_shift(uop_206.astype('uint32'), relay.reshape(var_231.astype('uint32'), relay.shape_of(uop_206))) # shape=(13,)
func_157_call = mod.get_global_var('func_157')
func_161_call = mutated_mod.get_global_var('func_161')
call_240 = relay.TupleGetItem(func_157_call(relay.reshape(var_217.astype('uint64'), [12, 3, 1]), relay.reshape(var_218.astype('uint64'), [12, 3, 8]), relay.reshape(call_216.astype('uint64'), [12, 3, 13]), ), 0)
call_241 = relay.TupleGetItem(func_161_call(relay.reshape(var_217.astype('uint64'), [12, 3, 1]), relay.reshape(var_218.astype('uint64'), [12, 3, 8]), relay.reshape(call_216.astype('uint64'), [12, 3, 13]), ), 0)
uop_242 = relay.sin(bop_232.astype('float64')) # shape=(13,)
bop_244 = relay.right_shift(uop_242.astype('int8'), relay.reshape(bop_237.astype('int8'), relay.shape_of(uop_242))) # shape=(13,)
uop_247 = relay.exp(bop_244.astype('float32')) # shape=(13,)
func_97_call = mod.get_global_var('func_97')
func_107_call = mutated_mod.get_global_var('func_107')
var_250 = relay.var("var_250", dtype = "float64", shape = (5, 1))#candidate|250|(5, 1)|var|float64
call_249 = relay.TupleGetItem(func_97_call(relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float32'), [5,]), relay.reshape(var_250.astype('float32'), [5,]), ), 11)
call_251 = relay.TupleGetItem(func_107_call(relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float64'), [5,]), relay.reshape(var_250.astype('float32'), [5,]), relay.reshape(var_250.astype('float32'), [5,]), ), 11)
bop_252 = relay.mod(uop_247.astype('float32'), relay.reshape(bop_237.astype('float32'), relay.shape_of(uop_247))) # shape=(13,)
bop_255 = relay.multiply(bop_232.astype('float64'), relay.reshape(bop_201.astype('float64'), relay.shape_of(bop_232))) # shape=(13,)
const_258 = relay.const([5.627855,-4.703369,-3.311174,0.075318,-1.688798,7.228238,5.085897,9.843874,1.085044,-1.710490,8.068304,3.858492,2.138759], dtype = "float32")#candidate|258|(13,)|const|float32
bop_259 = relay.less_equal(bop_252.astype('bool'), relay.reshape(const_258.astype('bool'), relay.shape_of(bop_252))) # shape=(13,)
bop_262 = relay.equal(uop_247.astype('bool'), call_216.astype('bool')) # shape=(12, 3, 13)
bop_265 = relay.equal(uop_247.astype('bool'), call_220.astype('bool')) # shape=(12, 3, 13)
bop_266 = relay.bitwise_xor(bop_259.astype('uint64'), relay.reshape(var_227.astype('uint64'), relay.shape_of(bop_259))) # shape=(13,)
bop_269 = relay.right_shift(bop_252.astype('int64'), relay.reshape(bop_201.astype('int64'), relay.shape_of(bop_252))) # shape=(13,)
bop_272 = relay.bitwise_or(uop_235.astype('int8'), call_210.astype('int8')) # shape=(13,)
bop_275 = relay.bitwise_or(uop_235.astype('int8'), call_213.astype('int8')) # shape=(13,)
bop_276 = relay.left_shift(bop_259.astype('int8'), relay.reshape(var_227.astype('int8'), relay.shape_of(bop_259))) # shape=(13,)
bop_279 = relay.power(bop_276.astype('float64'), relay.reshape(const_258.astype('float64'), relay.shape_of(bop_276))) # shape=(13,)
bop_282 = relay.greater(bop_272.astype('bool'), relay.reshape(bop_259.astype('bool'), relay.shape_of(bop_272))) # shape=(13,)
bop_285 = relay.greater(bop_275.astype('bool'), relay.reshape(bop_259.astype('bool'), relay.shape_of(bop_275))) # shape=(13,)
uop_286 = relay.acos(bop_244.astype('float64')) # shape=(13,)
var_288 = relay.var("var_288", dtype = "bool", shape = (13,))#candidate|288|(13,)|var|bool
bop_289 = relay.power(bop_259.astype('float64'), relay.reshape(var_288.astype('float64'), relay.shape_of(bop_259))) # shape=(13,)
uop_292 = relay.rsqrt(bop_244.astype('float32')) # shape=(13,)
bop_294 = relay.right_shift(bop_262.astype('int64'), uop_221.astype('int64')) # shape=(12, 3, 13)
bop_297 = relay.right_shift(bop_265.astype('int64'), uop_221.astype('int64')) # shape=(12, 3, 13)
var_298 = relay.var("var_298", dtype = "float32", shape = (13,))#candidate|298|(13,)|var|float32
bop_299 = relay.logical_xor(uop_235.astype('int8'), relay.reshape(var_298.astype('int8'), relay.shape_of(uop_235))) # shape=(13,)
uop_302 = relay.acosh(bop_279.astype('float32')) # shape=(13,)
bop_304 = relay.divide(bop_259.astype('float32'), relay.reshape(bop_266.astype('float32'), relay.shape_of(bop_259))) # shape=(13,)
bop_307 = relay.add(uop_302.astype('int32'), relay.reshape(var_197.astype('int32'), relay.shape_of(uop_302))) # shape=(13,)
bop_310 = relay.equal(uop_302.astype('bool'), call_216.astype('bool')) # shape=(12, 3, 13)
bop_313 = relay.equal(uop_302.astype('bool'), call_220.astype('bool')) # shape=(12, 3, 13)
bop_314 = relay.logical_and(bop_307.astype('bool'), relay.reshape(var_288.astype('bool'), relay.shape_of(bop_307))) # shape=(13,)
uop_317 = relay.atan(bop_307.astype('float32')) # shape=(13,)
uop_319 = relay.erf(bop_310.astype('float64')) # shape=(12, 3, 13)
uop_321 = relay.erf(bop_313.astype('float64')) # shape=(12, 3, 13)
bop_322 = relay.power(uop_319.astype('float32'), bop_299.astype('float32')) # shape=(12, 3, 13)
bop_325 = relay.power(uop_321.astype('float32'), bop_299.astype('float32')) # shape=(12, 3, 13)
output = relay.Tuple([bop_198,var_211,var_212,uop_214,var_217,var_218,var_219,bop_224,call_240,call_249,var_250,bop_255,bop_269,bop_282,uop_286,bop_289,uop_292,bop_294,bop_304,bop_314,uop_317,bop_322,])
output2 = relay.Tuple([bop_198,var_211,var_212,uop_214,var_217,var_218,var_219,bop_224,call_241,call_251,var_250,bop_255,bop_269,bop_285,uop_286,bop_289,uop_292,bop_297,bop_304,bop_314,uop_317,bop_325,])
func_326 = relay.Function([var_196,var_197,var_211,var_212,var_217,var_218,var_219,var_223,var_227,var_231,var_250,var_288,var_298,], output)
mod['func_326'] = func_326
mod = relay.transform.InferType()(mod)
var_327 = relay.var("var_327", dtype = "float32", shape = (13,))#candidate|327|(13,)|var|float32
var_328 = relay.var("var_328", dtype = "float32", shape = (13,))#candidate|328|(13,)|var|float32
var_329 = relay.var("var_329", dtype = "float32", shape = ())#candidate|329|()|var|float32
var_330 = relay.var("var_330", dtype = "uint8", shape = (5, 2))#candidate|330|(5, 2)|var|uint8
var_331 = relay.var("var_331", dtype = "uint64", shape = (36,))#candidate|331|(36,)|var|uint64
var_332 = relay.var("var_332", dtype = "uint64", shape = (288,))#candidate|332|(288,)|var|uint64
var_333 = relay.var("var_333", dtype = "uint64", shape = (468,))#candidate|333|(468,)|var|uint64
var_334 = relay.var("var_334", dtype = "float32", shape = (13,))#candidate|334|(13,)|var|float32
var_335 = relay.var("var_335", dtype = "float64", shape = (13,))#candidate|335|(13,)|var|float64
var_336 = relay.var("var_336", dtype = "float64", shape = (13,))#candidate|336|(13,)|var|float64
var_337 = relay.var("var_337", dtype = "float64", shape = (5, 1))#candidate|337|(5, 1)|var|float64
var_338 = relay.var("var_338", dtype = "bool", shape = (13,))#candidate|338|(13,)|var|bool
var_339 = relay.var("var_339", dtype = "float32", shape = (13,))#candidate|339|(13,)|var|float32
output = func_326(var_327,var_328,var_329,var_330,var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,)
func_340 = relay.Function([var_327,var_328,var_329,var_330,var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,], output)
mutated_mod['func_340'] = func_340
mutated_mod = relay.transform.InferType()(mutated_mod)
var_342 = relay.var("var_342", dtype = "float64", shape = (6, 7, 2))#candidate|342|(6, 7, 2)|var|float64
uop_343 = relay.log(var_342.astype('float64')) # shape=(6, 7, 2)
uop_345 = relay.sin(uop_343.astype('float32')) # shape=(6, 7, 2)
uop_347 = relay.sigmoid(uop_343.astype('float32')) # shape=(6, 7, 2)
uop_349 = relay.log10(uop_347.astype('float32')) # shape=(6, 7, 2)
uop_351 = relay.sinh(uop_347.astype('float32')) # shape=(6, 7, 2)
output = relay.Tuple([uop_345,uop_349,uop_351,])
output2 = relay.Tuple([uop_345,uop_349,uop_351,])
F = relay.Function([var_342,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_342,], output2)
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
input_342= np.array([[[8.327780,-7.864434],[1.756358,6.966873],[6.352609,-3.069213],[4.317967,-7.650031],[-6.664442,5.530161],[9.651814,8.993819],[-2.774050,8.281275]],[[7.400305,-3.032269],[-4.142331,7.912356],[-6.438595,5.503478],[-0.668054,-3.552117],[7.485296,-8.735841],[-1.671499,4.043941],[2.876387,1.585879]],[[-0.105335,-6.320769],[-9.436694,-5.867845],[1.486965,-4.060908],[8.173432,7.077044],[-4.506220,1.815377],[6.172170,6.486543],[-0.326219,4.951255]],[[6.238758,7.187792],[1.442461,-0.342446],[-8.748428,-1.104354],[-0.325319,-4.546888],[-0.145147,4.167047],[7.224268,4.011264],[-3.958274,-2.576549]],[[6.418612,-8.800899],[-7.229337,-5.308458],[-4.983882,-8.938125],[9.914385,-9.025935],[9.877894,6.207311],[7.951169,-5.448623],[-1.996497,2.669205]],[[-5.985663,9.614640],[1.848424,2.262542],[0.273785,2.348656],[-8.380889,-5.641793],[-2.761552,8.243631],[3.965440,-1.487882],[7.208817,8.227291]]], dtype='float64')
module1.set_input('var_342', input_342)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_342, )
res3 = intrp3.evaluate()(input_342, )
res4 = intrp4.evaluate()(input_342, )
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
module5.set_input('var_342', input_342)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_342, )
res7 = intrp7.evaluate()(input_342, )
res8 = intrp8.evaluate()(input_342, )
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
module9.set_input('var_342', input_342)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_342, )
res11 = intrp11.evaluate()(input_342, )
res12 = intrp12.evaluate()(input_342, )
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
module13.set_input('var_342', input_342)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_342, )
res15 = intrp15.evaluate()(input_342, )
res16 = intrp16.evaluate()(input_342, )
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
module17.set_input('var_342', input_342)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_342, )
res19 = intrp19.evaluate()(input_342, )
res20 = intrp20.evaluate()(input_342, )
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
module21.set_input('var_342', input_342)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_342, )
res23 = intrp23.evaluate()(input_342, )
res24 = intrp24.evaluate()(input_342, )
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

'''193: TVMFuncCall
192: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::vm::VMCompiler::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
191: tvm::relay::vm::VMCompiler::Lower(tvm::IRModule, tvm::runtime::Map<tvm::Integer, tvm::Target, void, void>, tvm::Target)
190: tvm::relay::vm::VMFunctionCompiler::Compile(tvm::GlobalVar const&, tvm::relay::Function const&)
189: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
188: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
187: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::FunctionNode const*)
186: tvm::relay::vm::VMFunctionCompiler::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
185: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
184: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
183: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
182: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
181: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
180: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
179: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
178: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
177: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
176: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
175: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
174: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
173: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
172: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
171: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
170: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
169: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
168: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
167: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
166: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
165: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
164: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
163: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
162: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
161: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
160: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
159: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
158: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
157: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
156: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
155: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
154: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
153: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
152: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
151: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
150: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
149: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
148: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
147: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
146: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
145: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
144: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
143: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
142: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
141: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
140: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
139: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
138: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
137: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
136: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
135: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
134: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
133: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
132: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
131: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
130: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
129: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
128: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
127: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
126: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
125: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
124: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
123: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
122: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
121: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
120: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
119: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
118: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
117: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
116: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
115: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
114: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
113: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
112: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
111: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
110: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
109: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
108: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
107: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
106: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
105: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
104: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
103: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
102: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
101: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
100: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
99: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
98: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
97: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
96: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
95: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
94: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
93: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
92: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
91: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
90: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
89: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
88: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
87: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
86: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
85: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
84: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
83: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
82: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
81: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
80: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
79: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
78: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
77: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
76: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
75: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
74: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
73: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
72: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
71: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
70: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
69: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
68: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
67: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
66: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
65: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
64: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
63: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
62: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
61: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
60: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
59: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
58: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
57: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
56: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
55: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
54: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
53: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
52: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
51: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
50: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
49: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
48: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
47: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
46: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
45: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
44: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
43: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
42: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
41: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
40: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
39: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
38: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
37: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
36: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
35: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
34: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
33: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
32: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
31: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
30: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
29: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
28: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
27: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
26: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
25: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
24: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
22: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
21: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
20: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
19: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
18: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
17: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
15: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
14: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
11: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
8: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
7: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
6: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
5: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
4: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
1: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
0: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)

'''