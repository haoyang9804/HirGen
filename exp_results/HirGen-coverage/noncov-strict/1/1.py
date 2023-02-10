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
var_0 = relay.var("var_0", dtype = "float64", shape = ())#candidate|0|()|var|float64
uop_1 = relay.rsqrt(var_0.astype('float64')) # shape=()
bop_3 = relay.logical_and(uop_1.astype('bool'), var_0.astype('bool')) # shape=()
bop_6 = relay.less_equal(uop_1.astype('bool'), bop_3.astype('bool')) # shape=()
bop_9 = relay.bitwise_and(uop_1.astype('int16'), var_0.astype('int16')) # shape=()
uop_12 = relay.sin(var_0.astype('float32')) # shape=()
bop_14 = relay.logical_and(bop_3.astype('bool'), bop_9.astype('bool')) # shape=()
bop_17 = relay.logical_xor(uop_1.astype('int64'), bop_9.astype('int64')) # shape=()
output = relay.Tuple([bop_6,uop_12,bop_14,bop_17,])
output2 = relay.Tuple([bop_6,uop_12,bop_14,bop_17,])
func_20 = relay.Function([var_0,], output)
mod['func_20'] = func_20
mod = relay.transform.InferType()(mod)
var_21 = relay.var("var_21", dtype = "float64", shape = ())#candidate|21|()|var|float64
output = func_20(var_21)
func_22 = relay.Function([var_21], output)
mutated_mod['func_22'] = func_22
mutated_mod = relay.transform.InferType()(mutated_mod)
var_24 = relay.var("var_24", dtype = "float64", shape = (3, 6))#candidate|24|(3, 6)|var|float64
uop_25 = relay.rsqrt(var_24.astype('float64')) # shape=(3, 6)
output = uop_25
output2 = uop_25
func_27 = relay.Function([var_24,], output)
mod['func_27'] = func_27
mod = relay.transform.InferType()(mod)
mutated_mod['func_27'] = func_27
mutated_mod = relay.transform.InferType()(mutated_mod)
var_28 = relay.var("var_28", dtype = "float64", shape = (3, 6))#candidate|28|(3, 6)|var|float64
func_27_call = mutated_mod.get_global_var('func_27')
call_29 = func_27_call(var_28)
output = call_29
func_30 = relay.Function([var_28], output)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
var_32 = relay.var("var_32", dtype = "float64", shape = ())#candidate|32|()|var|float64
uop_33 = relay.log10(var_32.astype('float64')) # shape=()
uop_35 = relay.erf(uop_33.astype('float64')) # shape=()
bop_37 = relay.greater_equal(uop_35.astype('bool'), uop_33.astype('bool')) # shape=()
func_27_call = mod.get_global_var('func_27')
func_30_call = mutated_mod.get_global_var('func_30')
const_41 = relay.const([[-0.353043],[-9.419562],[0.843837],[-1.469026],[1.434258],[7.307760],[-2.705194],[8.577402],[6.363337],[2.005302],[4.432207],[-5.143252],[-6.930925],[9.775554],[-1.216475],[4.659052],[3.452985],[8.516076]], dtype = "float64")#candidate|41|(18, 1)|const|float64
call_40 = func_27_call(relay.reshape(const_41.astype('float64'), [3, 6]))
call_42 = func_27_call(relay.reshape(const_41.astype('float64'), [3, 6]))
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_43 = relay.TupleGetItem(func_20_call(relay.reshape(uop_33.astype('float64'), [])), 1)
call_44 = relay.TupleGetItem(func_22_call(relay.reshape(uop_33.astype('float64'), [])), 1)
bop_45 = relay.bitwise_xor(uop_35.astype('int32'), call_40.astype('int32')) # shape=(3, 6)
bop_48 = relay.bitwise_xor(uop_35.astype('int32'), call_42.astype('int32')) # shape=(3, 6)
uop_49 = relay.cos(uop_33.astype('float64')) # shape=()
var_51 = relay.var("var_51", dtype = "float64", shape = (1, 7, 1))#candidate|51|(1, 7, 1)|var|float64
bop_52 = relay.greater(uop_33.astype('bool'), var_51.astype('bool')) # shape=(1, 7, 1)
uop_55 = relay.sin(uop_33.astype('float64')) # shape=()
uop_57 = relay.cosh(bop_52.astype('float32')) # shape=(1, 7, 1)
var_59 = relay.var("var_59", dtype = "bool", shape = (13, 7, 14))#candidate|59|(13, 7, 14)|var|bool
bop_60 = relay.floor_mod(bop_52.astype('float64'), var_59.astype('float64')) # shape=(13, 7, 14)
uop_63 = relay.log(bop_45.astype('float64')) # shape=(3, 6)
uop_65 = relay.log(bop_48.astype('float64')) # shape=(3, 6)
var_66 = relay.var("var_66", dtype = "float64", shape = (3, 6))#candidate|66|(3, 6)|var|float64
bop_67 = relay.power(uop_63.astype('float64'), relay.reshape(var_66.astype('float64'), relay.shape_of(uop_63))) # shape=(3, 6)
bop_70 = relay.power(uop_65.astype('float64'), relay.reshape(var_66.astype('float64'), relay.shape_of(uop_65))) # shape=(3, 6)
uop_71 = relay.log(uop_35.astype('float32')) # shape=()
uop_73 = relay.log2(uop_63.astype('float64')) # shape=(3, 6)
uop_75 = relay.log2(uop_65.astype('float64')) # shape=(3, 6)
var_76 = relay.var("var_76", dtype = "float64", shape = (3, 6))#candidate|76|(3, 6)|var|float64
bop_77 = relay.not_equal(uop_73.astype('bool'), relay.reshape(var_76.astype('bool'), relay.shape_of(uop_73))) # shape=(3, 6)
bop_80 = relay.not_equal(uop_75.astype('bool'), relay.reshape(var_76.astype('bool'), relay.shape_of(uop_75))) # shape=(3, 6)
bop_81 = relay.mod(bop_77.astype('float32'), relay.reshape(var_76.astype('float32'), relay.shape_of(bop_77))) # shape=(3, 6)
bop_84 = relay.mod(bop_80.astype('float32'), relay.reshape(var_76.astype('float32'), relay.shape_of(bop_80))) # shape=(3, 6)
var_85 = relay.var("var_85", dtype = "bool", shape = (3, 6))#candidate|85|(3, 6)|var|bool
bop_86 = relay.multiply(bop_77.astype('uint16'), relay.reshape(var_85.astype('uint16'), relay.shape_of(bop_77))) # shape=(3, 6)
bop_89 = relay.multiply(bop_80.astype('uint16'), relay.reshape(var_85.astype('uint16'), relay.shape_of(bop_80))) # shape=(3, 6)
bop_90 = relay.power(uop_35.astype('float64'), bop_81.astype('float64')) # shape=(3, 6)
bop_93 = relay.power(uop_35.astype('float64'), bop_84.astype('float64')) # shape=(3, 6)
bop_94 = relay.multiply(uop_33.astype('float32'), call_43.astype('float32')) # shape=()
bop_97 = relay.multiply(uop_33.astype('float32'), call_44.astype('float32')) # shape=()
uop_98 = relay.asin(uop_33.astype('float32')) # shape=()
uop_100 = relay.asinh(bop_81.astype('float32')) # shape=(3, 6)
uop_102 = relay.asinh(bop_84.astype('float32')) # shape=(3, 6)
var_103 = relay.var("var_103", dtype = "float32", shape = (3, 6))#candidate|103|(3, 6)|var|float32
bop_104 = relay.not_equal(uop_100.astype('bool'), relay.reshape(var_103.astype('bool'), relay.shape_of(uop_100))) # shape=(3, 6)
bop_107 = relay.not_equal(uop_102.astype('bool'), relay.reshape(var_103.astype('bool'), relay.shape_of(uop_102))) # shape=(3, 6)
uop_108 = relay.log2(bop_104.astype('float32')) # shape=(3, 6)
uop_110 = relay.log2(bop_107.astype('float32')) # shape=(3, 6)
uop_111 = relay.sin(uop_108.astype('float32')) # shape=(3, 6)
uop_113 = relay.sin(uop_110.astype('float32')) # shape=(3, 6)
bop_114 = relay.minimum(uop_111.astype('float64'), relay.reshape(call_40.astype('float64'), relay.shape_of(uop_111))) # shape=(3, 6)
bop_117 = relay.minimum(uop_113.astype('float64'), relay.reshape(call_42.astype('float64'), relay.shape_of(uop_113))) # shape=(3, 6)
uop_118 = relay.log(bop_77.astype('float32')) # shape=(3, 6)
uop_120 = relay.log(bop_80.astype('float32')) # shape=(3, 6)
uop_121 = relay.exp(uop_108.astype('float32')) # shape=(3, 6)
uop_123 = relay.exp(uop_110.astype('float32')) # shape=(3, 6)
bop_124 = relay.power(uop_121.astype('float32'), relay.reshape(uop_73.astype('float32'), relay.shape_of(uop_121))) # shape=(3, 6)
bop_127 = relay.power(uop_123.astype('float32'), relay.reshape(uop_75.astype('float32'), relay.shape_of(uop_123))) # shape=(3, 6)
bop_128 = relay.left_shift(uop_118.astype('uint8'), relay.reshape(uop_73.astype('uint8'), relay.shape_of(uop_118))) # shape=(3, 6)
bop_131 = relay.left_shift(uop_120.astype('uint8'), relay.reshape(uop_75.astype('uint8'), relay.shape_of(uop_120))) # shape=(3, 6)
bop_132 = relay.power(uop_108.astype('float32'), bop_94.astype('float32')) # shape=(3, 6)
bop_135 = relay.power(uop_110.astype('float32'), bop_97.astype('float32')) # shape=(3, 6)
var_136 = relay.var("var_136", dtype = "float32", shape = (3, 6))#candidate|136|(3, 6)|var|float32
bop_137 = relay.greater_equal(bop_124.astype('bool'), relay.reshape(var_136.astype('bool'), relay.shape_of(bop_124))) # shape=(3, 6)
bop_140 = relay.greater_equal(bop_127.astype('bool'), relay.reshape(var_136.astype('bool'), relay.shape_of(bop_127))) # shape=(3, 6)
uop_141 = relay.sinh(bop_132.astype('float32')) # shape=(3, 6)
uop_143 = relay.sinh(bop_135.astype('float32')) # shape=(3, 6)
var_144 = relay.var("var_144", dtype = "float64", shape = (3, 6))#candidate|144|(3, 6)|var|float64
bop_145 = relay.logical_and(uop_73.astype('bool'), relay.reshape(var_144.astype('bool'), relay.shape_of(uop_73))) # shape=(3, 6)
bop_148 = relay.logical_and(uop_75.astype('bool'), relay.reshape(var_144.astype('bool'), relay.shape_of(uop_75))) # shape=(3, 6)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_149 = relay.TupleGetItem(func_20_call(relay.reshape(uop_71.astype('float64'), [])), 0)
call_150 = relay.TupleGetItem(func_22_call(relay.reshape(uop_71.astype('float64'), [])), 0)
bop_151 = relay.multiply(uop_108.astype('int8'), relay.reshape(bop_86.astype('int8'), relay.shape_of(uop_108))) # shape=(3, 6)
bop_154 = relay.multiply(uop_110.astype('int8'), relay.reshape(bop_89.astype('int8'), relay.shape_of(uop_110))) # shape=(3, 6)
uop_155 = relay.cosh(bop_77.astype('float32')) # shape=(3, 6)
uop_157 = relay.cosh(bop_80.astype('float32')) # shape=(3, 6)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_158 = relay.TupleGetItem(func_20_call(relay.reshape(uop_35.astype('float64'), [])), 2)
call_159 = relay.TupleGetItem(func_22_call(relay.reshape(uop_35.astype('float64'), [])), 2)
bop_160 = relay.right_shift(uop_121.astype('uint32'), relay.reshape(bop_137.astype('uint32'), relay.shape_of(uop_121))) # shape=(3, 6)
bop_163 = relay.right_shift(uop_123.astype('uint32'), relay.reshape(bop_140.astype('uint32'), relay.shape_of(uop_123))) # shape=(3, 6)
bop_164 = relay.logical_or(bop_90.astype('bool'), relay.reshape(bop_104.astype('bool'), relay.shape_of(bop_90))) # shape=(3, 6)
bop_167 = relay.logical_or(bop_93.astype('bool'), relay.reshape(bop_107.astype('bool'), relay.shape_of(bop_93))) # shape=(3, 6)
uop_168 = relay.atan(bop_124.astype('float64')) # shape=(3, 6)
uop_170 = relay.atan(bop_127.astype('float64')) # shape=(3, 6)
bop_171 = relay.greater_equal(bop_160.astype('bool'), relay.reshape(var_136.astype('bool'), relay.shape_of(bop_160))) # shape=(3, 6)
bop_174 = relay.greater_equal(bop_163.astype('bool'), relay.reshape(var_136.astype('bool'), relay.shape_of(bop_163))) # shape=(3, 6)
var_175 = relay.var("var_175", dtype = "bool", shape = (3, 6))#candidate|175|(3, 6)|var|bool
bop_176 = relay.add(bop_137.astype('int8'), relay.reshape(var_175.astype('int8'), relay.shape_of(bop_137))) # shape=(3, 6)
bop_179 = relay.add(bop_140.astype('int8'), relay.reshape(var_175.astype('int8'), relay.shape_of(bop_140))) # shape=(3, 6)
uop_180 = relay.acos(uop_108.astype('float32')) # shape=(3, 6)
uop_182 = relay.acos(uop_110.astype('float32')) # shape=(3, 6)
var_183 = relay.var("var_183", dtype = "float64", shape = (3, 6))#candidate|183|(3, 6)|var|float64
bop_184 = relay.right_shift(uop_168.astype('int64'), relay.reshape(var_183.astype('int64'), relay.shape_of(uop_168))) # shape=(3, 6)
bop_187 = relay.right_shift(uop_170.astype('int64'), relay.reshape(var_183.astype('int64'), relay.shape_of(uop_170))) # shape=(3, 6)
uop_188 = relay.exp(uop_168.astype('float64')) # shape=(3, 6)
uop_190 = relay.exp(uop_170.astype('float64')) # shape=(3, 6)
bop_191 = relay.minimum(uop_188.astype('int64'), relay.reshape(bop_124.astype('int64'), relay.shape_of(uop_188))) # shape=(3, 6)
bop_194 = relay.minimum(uop_190.astype('int64'), relay.reshape(bop_127.astype('int64'), relay.shape_of(uop_190))) # shape=(3, 6)
bop_195 = relay.logical_and(uop_108.astype('bool'), bop_37.astype('bool')) # shape=(3, 6)
bop_198 = relay.logical_and(uop_110.astype('bool'), bop_37.astype('bool')) # shape=(3, 6)
var_199 = relay.var("var_199", dtype = "float32", shape = (3, 6))#candidate|199|(3, 6)|var|float32
bop_200 = relay.logical_xor(uop_180.astype('int8'), relay.reshape(var_199.astype('int8'), relay.shape_of(uop_180))) # shape=(3, 6)
bop_203 = relay.logical_xor(uop_182.astype('int8'), relay.reshape(var_199.astype('int8'), relay.shape_of(uop_182))) # shape=(3, 6)
uop_204 = relay.tan(uop_188.astype('float64')) # shape=(3, 6)
uop_206 = relay.tan(uop_190.astype('float64')) # shape=(3, 6)
var_207 = relay.var("var_207", dtype = "int64", shape = (3, 6))#candidate|207|(3, 6)|var|int64
bop_208 = relay.mod(bop_191.astype('float64'), relay.reshape(var_207.astype('float64'), relay.shape_of(bop_191))) # shape=(3, 6)
bop_211 = relay.mod(bop_194.astype('float64'), relay.reshape(var_207.astype('float64'), relay.shape_of(bop_194))) # shape=(3, 6)
output = relay.Tuple([const_41,uop_49,uop_55,uop_57,bop_60,bop_67,uop_71,uop_98,bop_114,bop_128,uop_141,bop_145,call_149,bop_151,uop_155,call_158,bop_164,bop_171,bop_176,bop_184,bop_195,bop_200,uop_204,bop_208,])
output2 = relay.Tuple([const_41,uop_49,uop_55,uop_57,bop_60,bop_70,uop_71,uop_98,bop_117,bop_131,uop_143,bop_148,call_150,bop_154,uop_157,call_159,bop_167,bop_174,bop_179,bop_187,bop_198,bop_203,uop_206,bop_211,])
func_212 = relay.Function([var_32,var_51,var_59,var_66,var_76,var_85,var_103,var_136,var_144,var_175,var_183,var_199,var_207,], output)
mod['func_212'] = func_212
mod = relay.transform.InferType()(mod)
mutated_mod['func_212'] = func_212
mutated_mod = relay.transform.InferType()(mutated_mod)
func_212_call = mutated_mod.get_global_var('func_212')
var_214 = relay.var("var_214", dtype = "float64", shape = ())#candidate|214|()|var|float64
var_215 = relay.var("var_215", dtype = "float64", shape = (1, 7, 1))#candidate|215|(1, 7, 1)|var|float64
var_216 = relay.var("var_216", dtype = "bool", shape = (13, 7, 14))#candidate|216|(13, 7, 14)|var|bool
var_217 = relay.var("var_217", dtype = "float64", shape = (3, 6))#candidate|217|(3, 6)|var|float64
var_218 = relay.var("var_218", dtype = "float64", shape = (3, 6))#candidate|218|(3, 6)|var|float64
var_219 = relay.var("var_219", dtype = "bool", shape = (3, 6))#candidate|219|(3, 6)|var|bool
var_220 = relay.var("var_220", dtype = "float32", shape = (3, 6))#candidate|220|(3, 6)|var|float32
var_221 = relay.var("var_221", dtype = "float32", shape = (3, 6))#candidate|221|(3, 6)|var|float32
var_222 = relay.var("var_222", dtype = "float64", shape = (3, 6))#candidate|222|(3, 6)|var|float64
var_223 = relay.var("var_223", dtype = "bool", shape = (3, 6))#candidate|223|(3, 6)|var|bool
var_224 = relay.var("var_224", dtype = "float64", shape = (3, 6))#candidate|224|(3, 6)|var|float64
var_225 = relay.var("var_225", dtype = "float32", shape = (3, 6))#candidate|225|(3, 6)|var|float32
var_226 = relay.var("var_226", dtype = "int64", shape = (3, 6))#candidate|226|(3, 6)|var|int64
call_213 = func_212_call(var_214,var_215,var_216,var_217,var_218,var_219,var_220,var_221,var_222,var_223,var_224,var_225,var_226,)
output = call_213
func_227 = relay.Function([var_214,var_215,var_216,var_217,var_218,var_219,var_220,var_221,var_222,var_223,var_224,var_225,var_226,], output)
mutated_mod['func_227'] = func_227
mutated_mod = relay.transform.InferType()(mutated_mod)
const_229 = relay.const([[[-7,-6,7,6,4,-3,7,-5,-7,7,6,-8],[-9,7,-3,7,3,-10,9,-10,-8,-4,-4,7],[-7,-1,-2,-3,5,2,2,10,5,6,6,6],[8,8,10,-7,-10,1,9,-2,3,2,3,3],[-2,3,4,3,-1,5,-6,8,3,9,-5,-6],[-7,6,-2,8,-7,-5,10,4,-9,-3,-3,5],[2,8,5,6,4,-1,-10,-2,2,7,-5,5],[9,-1,9,-10,9,6,4,7,-5,-6,-1,-1]],[[8,-7,10,9,2,-4,7,1,2,2,-1,-3],[1,-1,-8,7,10,-8,-8,2,6,-10,-9,-1],[-8,8,5,-6,-1,-9,7,6,10,10,1,-5],[-9,4,-5,1,-7,3,-10,-8,-3,-3,-1,-2],[3,8,-5,4,9,9,2,-7,9,-1,-8,2],[-6,-4,4,-7,-7,4,7,5,6,9,-4,-7],[8,-3,-6,10,10,8,-9,10,9,9,1,-3],[-7,-5,3,-5,-1,-9,-3,-1,-2,-6,6,-10]],[[-1,-3,2,1,10,-4,9,4,-10,-9,2,6],[10,9,-10,-3,8,3,-1,1,10,6,7,-1],[5,-9,-1,2,-9,-7,10,9,-5,-9,-9,-10],[8,-10,8,-1,1,-1,-7,4,-6,4,-1,5],[7,4,6,7,-2,8,-8,4,-9,2,9,-1],[-6,-2,-9,-5,-3,-4,-5,-3,-8,6,10,-7],[-6,-6,-4,-4,3,1,5,8,-5,6,-3,-6],[10,9,3,-2,5,-7,-6,-1,-9,10,-8,-9]],[[-3,2,6,9,1,7,-5,6,6,9,6,-10],[2,-8,-8,-3,4,9,5,-5,7,4,-10,2],[-6,9,-2,-9,-6,-3,4,7,3,-7,3,9],[-3,-4,5,-10,-8,5,5,-4,-8,9,-5,9],[2,10,-7,7,-2,-2,8,10,-10,-6,8,8],[-7,10,-3,1,7,1,2,6,1,-3,-2,5],[-7,-9,-2,-2,5,-4,3,4,-9,-5,-2,-5],[9,-2,-10,-4,-9,-1,-1,-4,5,5,-3,7]],[[3,-1,1,6,3,-2,4,-9,1,5,-4,-8],[-7,6,-3,10,-3,1,-10,-5,-6,7,2,-4],[-7,-5,-8,-10,8,-4,9,9,8,5,8,10],[-6,3,-2,-1,5,-10,10,7,9,1,7,9],[1,-1,5,-10,-6,-9,7,7,-7,-7,10,7],[7,-7,-9,8,-2,6,10,-6,-8,-7,6,-8],[-3,-2,-9,2,6,-10,6,4,9,-5,5,-7],[8,-9,-10,-9,-2,-5,7,-6,-7,-7,-4,-10]]], dtype = "int8")#candidate|229|(5, 8, 12)|const|int8
var_230 = relay.var("var_230", dtype = "int8", shape = (5, 8, 12))#candidate|230|(5, 8, 12)|var|int8
bop_231 = relay.maximum(const_229.astype('int8'), relay.reshape(var_230.astype('int8'), relay.shape_of(const_229))) # shape=(5, 8, 12)
bop_234 = relay.left_shift(bop_231.astype('uint16'), relay.reshape(var_230.astype('uint16'), relay.shape_of(bop_231))) # shape=(5, 8, 12)
uop_237 = relay.sigmoid(bop_234.astype('float64')) # shape=(5, 8, 12)
bop_239 = relay.greater(bop_234.astype('bool'), relay.reshape(const_229.astype('bool'), relay.shape_of(bop_234))) # shape=(5, 8, 12)
output = relay.Tuple([uop_237,bop_239,])
output2 = relay.Tuple([uop_237,bop_239,])
func_242 = relay.Function([var_230,], output)
mod['func_242'] = func_242
mod = relay.transform.InferType()(mod)
var_243 = relay.var("var_243", dtype = "int8", shape = (5, 8, 12))#candidate|243|(5, 8, 12)|var|int8
output = func_242(var_243)
func_244 = relay.Function([var_243], output)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
var_246 = relay.var("var_246", dtype = "float32", shape = (15, 5))#candidate|246|(15, 5)|var|float32
var_247 = relay.var("var_247", dtype = "float32", shape = (15, 5))#candidate|247|(15, 5)|var|float32
bop_248 = relay.mod(var_246.astype('float32'), relay.reshape(var_247.astype('float32'), relay.shape_of(var_246))) # shape=(15, 5)
uop_251 = relay.cosh(var_247.astype('float32')) # shape=(15, 5)
uop_253 = relay.rsqrt(uop_251.astype('float32')) # shape=(15, 5)
output = relay.Tuple([bop_248,uop_253,])
output2 = relay.Tuple([bop_248,uop_253,])
func_255 = relay.Function([var_246,var_247,], output)
mod['func_255'] = func_255
mod = relay.transform.InferType()(mod)
mutated_mod['func_255'] = func_255
mutated_mod = relay.transform.InferType()(mutated_mod)
func_255_call = mutated_mod.get_global_var('func_255')
var_257 = relay.var("var_257", dtype = "float32", shape = (15, 5))#candidate|257|(15, 5)|var|float32
var_258 = relay.var("var_258", dtype = "float32", shape = (15, 5))#candidate|258|(15, 5)|var|float32
call_256 = func_255_call(var_257,var_258,)
output = call_256
func_259 = relay.Function([var_257,var_258,], output)
mutated_mod['func_259'] = func_259
mutated_mod = relay.transform.InferType()(mutated_mod)
var_261 = relay.var("var_261", dtype = "int16", shape = (10, 7, 11))#candidate|261|(10, 7, 11)|var|int16
var_262 = relay.var("var_262", dtype = "int16", shape = (10, 7, 11))#candidate|262|(10, 7, 11)|var|int16
bop_263 = relay.logical_xor(var_261.astype('int16'), relay.reshape(var_262.astype('int16'), relay.shape_of(var_261))) # shape=(10, 7, 11)
var_266 = relay.var("var_266", dtype = "int16", shape = (10, 7, 11))#candidate|266|(10, 7, 11)|var|int16
bop_267 = relay.logical_or(var_262.astype('bool'), relay.reshape(var_266.astype('bool'), relay.shape_of(var_262))) # shape=(10, 7, 11)
bop_270 = relay.divide(bop_267.astype('float32'), relay.reshape(var_262.astype('float32'), relay.shape_of(bop_267))) # shape=(10, 7, 11)
bop_273 = relay.power(var_261.astype('float32'), relay.reshape(bop_263.astype('float32'), relay.shape_of(var_261))) # shape=(10, 7, 11)
output = relay.Tuple([bop_270,bop_273,])
output2 = relay.Tuple([bop_270,bop_273,])
func_276 = relay.Function([var_261,var_262,var_266,], output)
mod['func_276'] = func_276
mod = relay.transform.InferType()(mod)
var_277 = relay.var("var_277", dtype = "int16", shape = (10, 7, 11))#candidate|277|(10, 7, 11)|var|int16
var_278 = relay.var("var_278", dtype = "int16", shape = (10, 7, 11))#candidate|278|(10, 7, 11)|var|int16
var_279 = relay.var("var_279", dtype = "int16", shape = (10, 7, 11))#candidate|279|(10, 7, 11)|var|int16
output = func_276(var_277,var_278,var_279,)
func_280 = relay.Function([var_277,var_278,var_279,], output)
mutated_mod['func_280'] = func_280
mutated_mod = relay.transform.InferType()(mutated_mod)
var_282 = relay.var("var_282", dtype = "float32", shape = (15,))#candidate|282|(15,)|var|float32
uop_283 = relay.cosh(var_282.astype('float32')) # shape=(15,)
var_285 = relay.var("var_285", dtype = "float32", shape = (15,))#candidate|285|(15,)|var|float32
bop_286 = relay.subtract(uop_283.astype('int32'), relay.reshape(var_285.astype('int32'), relay.shape_of(uop_283))) # shape=(15,)
var_289 = relay.var("var_289", dtype = "float32", shape = (15,))#candidate|289|(15,)|var|float32
bop_290 = relay.greater_equal(uop_283.astype('bool'), relay.reshape(var_289.astype('bool'), relay.shape_of(uop_283))) # shape=(15,)
uop_293 = relay.sinh(bop_290.astype('float32')) # shape=(15,)
var_295 = relay.var("var_295", dtype = "float32", shape = (15,))#candidate|295|(15,)|var|float32
bop_296 = relay.greater_equal(uop_283.astype('bool'), relay.reshape(var_295.astype('bool'), relay.shape_of(uop_283))) # shape=(15,)
func_242_call = mod.get_global_var('func_242')
func_244_call = mutated_mod.get_global_var('func_244')
const_300 = relay.const([[3],[-5],[-2],[-9],[3],[3],[-5],[-8],[3],[-10],[1],[5],[-2],[4],[4],[-4],[-7],[-7],[-6],[-10],[-10],[-4],[-1],[-7],[-2],[-7],[9],[-2],[10],[4],[2],[-10],[9],[-8],[8],[5],[2],[-5],[2],[7],[1],[8],[-8],[4],[-8],[4],[8],[-1],[-5],[-1],[-6],[1],[4],[2],[9],[1],[-3],[-1],[3],[1],[-4],[3],[10],[7],[2],[6],[-4],[7],[2],[5],[1],[3],[-7],[7],[8],[1],[-4],[3],[8],[-7],[8],[8],[10],[-10],[-7],[6],[-3],[9],[3],[-9],[-5],[-4],[7],[-3],[7],[-3],[3],[7],[-7],[8],[3],[9],[2],[5],[-6],[-8],[-9],[2],[-10],[-8],[4],[4],[-3],[3],[-1],[5],[6],[10],[-3],[-5],[1],[2],[8],[5],[8],[6],[7],[9],[2],[-1],[2],[5],[-9],[8],[-9],[-1],[-8],[10],[6],[3],[9],[5],[-9],[5],[4],[-4],[-1],[6],[3],[-9],[7],[-3],[-2],[1],[8],[-3],[-5],[-2],[-3],[-6],[10],[-5],[9],[8],[5],[4],[-9],[9],[4],[3],[-5],[-9],[-5],[5],[-8],[6],[8],[-9],[-3],[9],[-8],[10],[-4],[9],[10],[-9],[8],[-7],[1],[-2],[-10],[-6],[-2],[-7],[6],[2],[1],[-9],[2],[10],[-2],[9],[9],[5],[-6],[8],[7],[-8],[1],[-3],[9],[8],[7],[4],[9],[-3],[5],[1],[8],[3],[-4],[3],[4],[-10],[5],[1],[9],[6],[9],[7],[-4],[-1],[4],[4],[-7],[-7],[-7],[-4],[-4],[-8],[-10],[-4],[8],[-6],[8],[7],[6],[2],[9],[2],[-2],[-8],[-3],[-1],[-9],[-3],[6],[9],[-7],[-1],[8],[8],[5],[4],[5],[-1],[-6],[5],[-3],[4],[3],[-2],[-7],[3],[6],[9],[-5],[10],[7],[5],[9],[-7],[6],[6],[-8],[3],[10],[-2],[-6],[10],[-10],[-6],[8],[-4],[-9],[6],[-6],[-6],[-5],[-2],[-3],[-4],[-10],[-1],[-1],[6],[2],[8],[-9],[2],[3],[9],[3],[2],[8],[-10],[-10],[-3],[3],[5],[-4],[7],[7],[5],[-10],[-6],[-7],[4],[10],[7],[-1],[8],[-1],[10],[-6],[-8],[-7],[8],[2],[2],[10],[-7],[-2],[-6],[-7],[2],[-10],[-7],[-3],[5],[-3],[-10],[9],[6],[1],[5],[5],[10],[-3],[2],[9],[-9],[-4],[10],[-3],[-7],[-1],[9],[-4],[3],[-3],[7],[-2],[-5],[-3],[-8],[1],[-1],[-2],[-2],[-7],[3],[5],[8],[-7],[5],[1],[5],[-8],[2],[-7],[6],[-6],[5],[-8],[-4],[6],[-3],[-1],[10],[-3],[-7],[-8],[-2],[7],[10],[2],[4],[-2],[5],[5],[-5],[9],[4],[8],[-10],[-1],[-9],[-3],[7],[-5],[-4],[4],[-5],[8],[-6],[-3],[6],[8],[4],[3],[9],[2],[5],[1],[5],[-1],[-2],[-8],[3],[9],[1],[-10],[-6],[6],[-6],[-5],[-5],[5],[4],[3],[1],[-2],[-3],[10],[-8],[-5],[4],[5],[-6],[9],[2],[-9],[3],[-8],[-1],[8],[-7],[4],[1],[-5],[-6],[-6],[-10],[10],[2],[9],[5],[10],[3]], dtype = "int8")#candidate|300|(480, 1)|const|int8
call_299 = relay.TupleGetItem(func_242_call(relay.reshape(const_300.astype('int8'), [5, 8, 12])), 0)
call_301 = relay.TupleGetItem(func_244_call(relay.reshape(const_300.astype('int8'), [5, 8, 12])), 0)
uop_302 = relay.cos(uop_293.astype('float64')) # shape=(15,)
uop_304 = relay.cosh(uop_293.astype('float32')) # shape=(15,)
uop_306 = relay.tan(uop_293.astype('float64')) # shape=(15,)
bop_308 = relay.floor_mod(uop_306.astype('float32'), relay.reshape(var_289.astype('float32'), relay.shape_of(uop_306))) # shape=(15,)
uop_311 = relay.log(uop_293.astype('float32')) # shape=(15,)
bop_313 = relay.greater_equal(bop_290.astype('bool'), relay.reshape(uop_283.astype('bool'), relay.shape_of(bop_290))) # shape=(15,)
var_316 = relay.var("var_316", dtype = "float32", shape = (15,))#candidate|316|(15,)|var|float32
bop_317 = relay.not_equal(uop_293.astype('bool'), relay.reshape(var_316.astype('bool'), relay.shape_of(uop_293))) # shape=(15,)
output = relay.Tuple([bop_286,bop_296,call_299,const_300,uop_302,uop_304,bop_308,uop_311,bop_313,bop_317,])
output2 = relay.Tuple([bop_286,bop_296,call_301,const_300,uop_302,uop_304,bop_308,uop_311,bop_313,bop_317,])
func_320 = relay.Function([var_282,var_285,var_289,var_295,var_316,], output)
mod['func_320'] = func_320
mod = relay.transform.InferType()(mod)
var_321 = relay.var("var_321", dtype = "float32", shape = (15,))#candidate|321|(15,)|var|float32
var_322 = relay.var("var_322", dtype = "float32", shape = (15,))#candidate|322|(15,)|var|float32
var_323 = relay.var("var_323", dtype = "float32", shape = (15,))#candidate|323|(15,)|var|float32
var_324 = relay.var("var_324", dtype = "float32", shape = (15,))#candidate|324|(15,)|var|float32
var_325 = relay.var("var_325", dtype = "float32", shape = (15,))#candidate|325|(15,)|var|float32
output = func_320(var_321,var_322,var_323,var_324,var_325,)
func_326 = relay.Function([var_321,var_322,var_323,var_324,var_325,], output)
mutated_mod['func_326'] = func_326
mutated_mod = relay.transform.InferType()(mutated_mod)
var_328 = relay.var("var_328", dtype = "float32", shape = (11, 16, 3))#candidate|328|(11, 16, 3)|var|float32
const_329 = relay.const([[[9.364840,-5.073978,6.472992],[7.255448,-4.124503,-1.361276],[7.663581,1.692809,1.610714],[-6.998607,-3.722040,-6.617076],[-7.534016,-2.121614,-9.205754],[-1.647874,-3.226334,7.792897],[-2.753933,-7.705646,-8.118576],[1.133137,7.824174,-3.666848],[-7.335063,-9.015197,-7.899149],[7.145819,-8.466907,0.343780],[3.578240,-0.542670,6.960305],[2.255519,-3.425357,-1.937555],[-8.223627,-3.214826,-6.812818],[-4.586317,-8.715012,7.708112],[-3.887165,3.781172,5.725505],[6.247124,-8.737782,-5.768588]],[[2.686546,4.771863,5.503788],[-3.331372,0.804189,7.038468],[-5.356142,-7.090332,-7.245868],[2.438666,-7.711476,8.133466],[-1.782913,-8.869783,-4.378576],[-4.019704,0.504943,6.084783],[5.490601,-2.305588,6.279034],[-8.060188,-6.852734,-2.342659],[1.723590,0.445425,-3.123987],[0.445380,1.173728,0.168938],[-4.740687,0.119309,3.968693],[6.781943,-7.913356,-3.262151],[4.034075,6.914984,9.181286],[5.126603,-4.743246,3.255395],[7.114422,9.719787,-7.171002],[4.038135,-4.023904,-7.566303]],[[2.787924,-9.283798,-7.240767],[2.134177,4.909675,6.547600],[-4.079091,6.347194,1.661205],[1.113915,8.988478,3.906524],[9.480040,7.604037,2.241571],[-6.423646,5.886454,-5.630632],[3.189460,4.354815,2.392609],[3.827494,0.200210,-3.375364],[6.619332,-9.577659,-4.862076],[0.942548,-0.440837,6.150295],[-5.396680,-5.627096,-5.386143],[0.837586,4.396553,-9.800402],[8.536567,-5.188400,8.850526],[8.335924,5.450722,0.691336],[6.110116,0.812521,5.169252],[6.335208,-9.680646,9.993886]],[[3.781170,-0.765555,-9.404073],[4.781799,6.844163,-1.503434],[-3.673547,9.009925,8.029153],[9.700615,8.834684,-3.041020],[1.179770,5.699457,-3.201519],[-5.449301,4.401549,-2.496419],[3.044361,-2.883079,5.178891],[9.766162,-5.800070,5.827607],[-3.729029,8.960744,-9.544194],[-2.118251,9.278718,2.172134],[6.269247,6.089352,1.662479],[-1.895584,4.316227,1.551386],[0.666284,-0.570653,1.526404],[-4.791611,-9.352715,-8.534348],[6.077866,-9.990372,-7.237560],[-4.220288,9.148053,-7.990539]],[[6.451525,-8.083103,3.558312],[-9.136071,-2.762155,2.398879],[-4.995012,0.043036,3.638786],[3.516298,5.386903,7.559434],[-6.228472,1.698025,-2.877756],[-6.559289,-4.520072,-3.783395],[-7.599395,-7.056815,-4.986940],[-9.320162,3.118808,1.501035],[8.360476,-9.677290,6.663226],[-2.632781,8.333970,5.840780],[-4.457951,-5.142449,-4.322197],[-3.963166,6.081376,-7.121491],[-8.705821,-9.706183,-6.709892],[-7.065214,1.188546,-4.936990],[0.868091,-5.215845,-6.015036],[-9.530845,7.471114,2.359764]],[[3.147167,-8.821993,7.759888],[-2.165301,6.040894,8.208449],[0.770296,-7.717942,-2.138359],[1.822376,5.840573,2.450081],[1.990448,-1.116034,-0.475527],[0.339711,-9.192588,-1.871633],[-1.574671,5.022103,1.847902],[8.097233,3.463031,3.509231],[3.162810,0.952578,-8.800839],[-1.610795,-2.196161,-6.738708],[4.071565,-4.929466,1.633899],[0.279626,-8.814871,-3.151218],[9.855820,-9.038526,9.970668],[2.533970,-3.754595,8.474707],[9.881954,-6.619149,5.477940],[-4.874421,-2.734683,-5.506721]],[[3.477142,4.303602,-6.398296],[1.810393,-4.934281,-8.315802],[-8.627277,-7.786370,-1.289417],[6.433634,0.031743,3.063440],[2.344429,1.080651,7.125295],[4.648830,2.048592,-8.633842],[-1.059649,-1.389173,8.303141],[-0.218235,-7.497981,-6.653271],[6.685153,-1.049592,3.574663],[6.435264,8.789577,0.263746],[-1.307490,9.889488,7.242547],[2.647214,5.071617,-0.473800],[2.335523,-9.949252,1.834514],[-4.525437,2.340583,-5.584439],[0.028241,2.605477,-0.153901],[5.029437,9.549573,6.173142]],[[-1.033682,4.352191,-6.352774],[-5.191460,-9.543702,2.655317],[4.918190,-8.796033,-0.041751],[8.748108,-3.150471,-8.269556],[0.172254,8.238182,-8.178115],[-7.354467,-0.377127,-7.223878],[1.249266,-9.484160,-1.211605],[-4.092560,5.742490,-9.093826],[8.311960,7.541639,-7.638391],[-8.028008,6.288332,-6.501031],[-5.689890,-9.814103,-1.506790],[5.227687,1.820566,-1.086986],[6.256222,1.556134,8.471262],[-1.675688,-5.753693,5.240259],[8.393260,-2.964527,4.242660],[-7.751679,4.414429,-9.599183]],[[-4.686035,-9.330490,8.738720],[5.851044,6.587075,-1.753480],[-4.157123,2.012701,-7.293041],[8.612851,-3.129842,9.194658],[-8.837186,-3.918297,-7.580864],[-3.583717,-3.766473,1.173565],[1.300555,8.714794,9.004261],[9.620162,-3.905633,-3.173933],[4.457837,8.955404,-1.857425],[-5.376980,-4.558164,-7.067732],[0.786905,8.651648,2.147257],[-5.233508,4.390948,7.604270],[2.338210,-9.413176,5.138573],[-2.765777,-4.113995,1.570988],[-8.970461,-3.069153,-3.865571],[-2.804007,9.382109,6.315399]],[[-7.720899,8.593622,2.538829],[-9.662783,3.498319,9.617369],[-5.321366,-7.074060,4.782952],[1.977913,2.932307,-6.339917],[-8.853397,6.566509,9.821534],[9.669245,-5.552053,9.636146],[7.348069,-4.972915,2.145241],[-9.211353,-4.082085,-0.640944],[-9.339978,9.956100,2.066397],[7.087473,-2.289656,1.433546],[4.226617,8.883410,6.935313],[-1.283085,0.825730,-0.521318],[-4.499173,-3.934722,0.165246],[-5.820566,-2.880183,5.135918],[5.237280,1.848045,4.955428],[2.195708,3.917613,9.537843]],[[-7.601473,-9.006170,-9.625660],[-9.674121,4.126298,4.993024],[7.472435,-0.464923,-3.045288],[-2.411446,-6.752759,4.066948],[-7.469228,-0.983382,-0.663329],[7.550557,4.794685,-6.120993],[-8.858342,-3.912039,1.571425],[-7.741329,9.404498,-0.343936],[-0.364396,-5.647263,-7.905167],[-6.660904,-1.131778,9.882770],[-8.555830,0.724698,-5.349580],[-9.790140,-5.025511,-5.189228],[-1.107002,0.226680,6.901366],[6.186177,-3.005526,-0.800562],[-4.056148,9.581397,6.441273],[2.387617,1.792067,-6.882667]]], dtype = "float32")#candidate|329|(11, 16, 3)|const|float32
bop_330 = relay.floor_divide(var_328.astype('float32'), relay.reshape(const_329.astype('float32'), relay.shape_of(var_328))) # shape=(11, 16, 3)
bop_333 = relay.right_shift(bop_330.astype('int32'), relay.reshape(var_328.astype('int32'), relay.shape_of(bop_330))) # shape=(11, 16, 3)
bop_336 = relay.not_equal(const_329.astype('bool'), relay.reshape(var_328.astype('bool'), relay.shape_of(const_329))) # shape=(11, 16, 3)
uop_339 = relay.cosh(bop_333.astype('float64')) # shape=(11, 16, 3)
bop_341 = relay.greater_equal(uop_339.astype('bool'), relay.reshape(bop_333.astype('bool'), relay.shape_of(uop_339))) # shape=(11, 16, 3)
bop_344 = relay.right_shift(uop_339.astype('uint64'), relay.reshape(var_328.astype('uint64'), relay.shape_of(uop_339))) # shape=(11, 16, 3)
const_347 = relay.const([[[-2,5,-4],[-1,1,-9],[8,10,-5],[-8,-1,8],[6,-4,4],[5,7,-5],[-4,-5,2],[10,1,-7],[-6,7,-6],[1,-4,-8],[-9,8,2],[-4,-2,3],[-6,1,4],[-6,-3,-10],[-8,2,8],[-4,-1,-2]],[[6,3,-8],[1,1,3],[-7,8,4],[10,10,5],[10,-6,2],[7,-7,-7],[7,9,7],[5,-3,-6],[2,-8,-6],[2,-4,8],[-8,5,5],[10,-7,4],[-1,2,4],[10,3,-10],[-2,-6,3],[-2,1,4]],[[-7,-2,3],[2,6,7],[-1,-8,6],[-5,2,-5],[-7,4,4],[6,4,10],[3,-1,-8],[-9,-7,5],[-6,-2,-4],[-1,-5,1],[-5,-3,-4],[-8,6,3],[10,-3,8],[-2,2,3],[2,9,1],[7,6,6]],[[-4,6,-4],[4,-3,-2],[-6,-5,8],[-4,-2,-6],[-4,10,4],[1,-10,7],[-8,-7,7],[3,-2,4],[8,10,3],[-4,10,7],[10,3,-7],[-6,-9,-9],[4,2,2],[-8,3,2],[-4,5,2],[2,-1,4]],[[5,-6,-4],[-3,-7,-4],[10,-3,8],[3,4,9],[-9,10,6],[-10,3,-5],[9,5,2],[8,1,-7],[7,-2,5],[6,2,3],[-9,7,9],[1,10,5],[-5,-2,-4],[-5,8,-1],[9,-8,-5],[-9,4,-8]],[[-9,-10,-5],[9,-9,8],[1,-4,1],[3,-5,2],[-1,-4,-8],[7,-1,4],[-3,-8,3],[6,1,4],[-7,3,7],[-2,8,-5],[-10,4,9],[-9,-1,5],[-5,10,10],[8,-3,9],[-8,-10,9],[-3,-8,-10]],[[4,-9,6],[-9,-4,-5],[-4,-10,-5],[9,-2,-10],[-4,9,9],[-9,1,-8],[-4,10,-9],[-2,9,10],[-3,-7,-7],[5,9,3],[7,3,-4],[5,-9,7],[1,10,3],[1,-10,2],[-5,-9,3],[4,-6,8]],[[-7,9,9],[3,5,-7],[3,6,-8],[1,5,10],[3,-10,1],[10,-10,-6],[-5,8,-2],[8,-6,-1],[3,10,1],[8,3,5],[-5,-2,4],[-5,-6,-10],[-9,-6,-3],[9,-8,-5],[-8,2,-2],[-5,-2,3]],[[6,-8,2],[-3,-6,-6],[-5,-6,6],[1,-6,-3],[-10,7,-5],[1,-10,-6],[-6,-3,-9],[1,6,6],[-1,-3,-1],[-7,-7,4],[-5,-5,-5],[7,-1,4],[-9,-3,-4],[-6,7,-2],[-10,-9,3],[10,2,5]],[[8,1,-10],[7,-9,6],[-9,-8,6],[5,1,4],[9,4,-9],[9,8,-5],[4,-2,2],[8,-2,10],[-8,-7,-10],[3,-3,1],[3,-9,9],[-1,-8,7],[3,-5,-4],[5,9,9],[3,9,8],[-9,-9,3]],[[-6,2,-9],[6,-6,7],[4,-2,-5],[4,-4,3],[-2,-2,-3],[-2,3,-3],[-5,-4,-7],[-1,-4,8],[5,-7,-5],[-8,3,2],[8,6,-10],[-4,5,-7],[9,-7,-2],[-2,9,3],[-10,-2,2],[-10,10,2]]], dtype = "uint64")#candidate|347|(11, 16, 3)|const|uint64
bop_348 = relay.bitwise_or(bop_344.astype('int32'), relay.reshape(const_347.astype('int32'), relay.shape_of(bop_344))) # shape=(11, 16, 3)
output = relay.Tuple([bop_336,bop_341,bop_348,])
output2 = relay.Tuple([bop_336,bop_341,bop_348,])
func_351 = relay.Function([var_328,], output)
mod['func_351'] = func_351
mod = relay.transform.InferType()(mod)
mutated_mod['func_351'] = func_351
mutated_mod = relay.transform.InferType()(mutated_mod)
var_352 = relay.var("var_352", dtype = "float32", shape = (11, 16, 3))#candidate|352|(11, 16, 3)|var|float32
func_351_call = mutated_mod.get_global_var('func_351')
call_353 = func_351_call(var_352)
output = call_353
func_354 = relay.Function([var_352], output)
mutated_mod['func_354'] = func_354
mutated_mod = relay.transform.InferType()(mutated_mod)
var_356 = relay.var("var_356", dtype = "float32", shape = (3,))#candidate|356|(3,)|var|float32
var_357 = relay.var("var_357", dtype = "float32", shape = (3,))#candidate|357|(3,)|var|float32
bop_358 = relay.floor_mod(var_356.astype('float32'), relay.reshape(var_357.astype('float32'), relay.shape_of(var_356))) # shape=(3,)
uop_361 = relay.exp(var_356.astype('float32')) # shape=(3,)
uop_363 = relay.sqrt(uop_361.astype('float32')) # shape=(3,)
uop_365 = relay.erf(uop_361.astype('float32')) # shape=(3,)
uop_367 = relay.asin(var_357.astype('float64')) # shape=(3,)
uop_369 = relay.cos(uop_363.astype('float64')) # shape=(3,)
bop_371 = relay.bitwise_and(uop_369.astype('uint16'), relay.reshape(uop_363.astype('uint16'), relay.shape_of(uop_369))) # shape=(3,)
var_374 = relay.var("var_374", dtype = "float32", shape = (3,))#candidate|374|(3,)|var|float32
bop_375 = relay.right_shift(uop_363.astype('uint8'), relay.reshape(var_374.astype('uint8'), relay.shape_of(uop_363))) # shape=(3,)
output = relay.Tuple([bop_358,uop_365,uop_367,bop_371,bop_375,])
output2 = relay.Tuple([bop_358,uop_365,uop_367,bop_371,bop_375,])
func_378 = relay.Function([var_356,var_357,var_374,], output)
mod['func_378'] = func_378
mod = relay.transform.InferType()(mod)
mutated_mod['func_378'] = func_378
mutated_mod = relay.transform.InferType()(mutated_mod)
func_378_call = mutated_mod.get_global_var('func_378')
var_380 = relay.var("var_380", dtype = "float32", shape = (3,))#candidate|380|(3,)|var|float32
var_381 = relay.var("var_381", dtype = "float32", shape = (3,))#candidate|381|(3,)|var|float32
var_382 = relay.var("var_382", dtype = "float32", shape = (3,))#candidate|382|(3,)|var|float32
call_379 = func_378_call(var_380,var_381,var_382,)
output = call_379
func_383 = relay.Function([var_380,var_381,var_382,], output)
mutated_mod['func_383'] = func_383
mutated_mod = relay.transform.InferType()(mutated_mod)
var_385 = relay.var("var_385", dtype = "uint64", shape = (5,))#candidate|385|(5,)|var|uint64
const_386 = relay.const([1,6,1,-7,-6], dtype = "uint64")#candidate|386|(5,)|const|uint64
bop_387 = relay.greater(var_385.astype('bool'), relay.reshape(const_386.astype('bool'), relay.shape_of(var_385))) # shape=(5,)
const_390 = relay.const([False,True,True,True,False], dtype = "bool")#candidate|390|(5,)|const|bool
bop_391 = relay.bitwise_and(bop_387.astype('int16'), relay.reshape(const_390.astype('int16'), relay.shape_of(bop_387))) # shape=(5,)
bop_394 = relay.power(bop_387.astype('float32'), relay.reshape(bop_391.astype('float32'), relay.shape_of(bop_387))) # shape=(5,)
uop_397 = relay.rsqrt(bop_394.astype('float64')) # shape=(5,)
bop_399 = relay.maximum(uop_397.astype('uint16'), relay.reshape(bop_387.astype('uint16'), relay.shape_of(uop_397))) # shape=(5,)
uop_402 = relay.log2(bop_399.astype('float32')) # shape=(5,)
bop_404 = relay.logical_or(const_386.astype('bool'), relay.reshape(const_390.astype('bool'), relay.shape_of(const_386))) # shape=(5,)
output = relay.Tuple([uop_402,bop_404,])
output2 = relay.Tuple([uop_402,bop_404,])
F = relay.Function([var_385,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_385,], output2)
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
input_385= np.array([10,5,3,2,8], dtype='uint64')
module1.set_input('var_385', input_385)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_385, )
res3 = intrp3.evaluate()(input_385, )
res4 = intrp4.evaluate()(input_385, )
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
module5.set_input('var_385', input_385)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_385, )
res7 = intrp7.evaluate()(input_385, )
res8 = intrp8.evaluate()(input_385, )
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
module9.set_input('var_385', input_385)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_385, )
res11 = intrp11.evaluate()(input_385, )
res12 = intrp12.evaluate()(input_385, )
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
module13.set_input('var_385', input_385)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_385, )
res15 = intrp15.evaluate()(input_385, )
res16 = intrp16.evaluate()(input_385, )
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
module17.set_input('var_385', input_385)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_385, )
res19 = intrp19.evaluate()(input_385, )
res20 = intrp20.evaluate()(input_385, )
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
module21.set_input('var_385', input_385)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_385, )
res23 = intrp23.evaluate()(input_385, )
res24 = intrp24.evaluate()(input_385, )
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

'''49: TVMFuncCall
48: _ZNSt17_Function_handlerIFvN3tvm7run
47: tvm::runtime::TypedPackedFunc<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>::AssignTypedLambda<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>(tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
46: tvm::relay::EvalFunction(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)
45: tvm::relay::Prepare(tvm::IRModule, tvm::CompilationConfig)
44: tvm::transform::Pass::operator()(tvm::IRModule) const
43: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
42: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
41: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
40: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
36: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
35: tvm::transform::Pass::operator()(tvm::IRModule) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
27: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
26: _ZN3tvm5relay9transform22Devic
25: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
24: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
23: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
22: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
21: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
20: tvm::relay::tec::LowerTensorExprMutator::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
16: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
15: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
14: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
13: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
12: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
11: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
9: tvm::relay::tec::TECompilerImpl::LowerShapeFunc(tvm::relay::tec::CCacheKey const&)
8: tvm::relay::tec::TECompilerImpl::LowerShapeFuncInternal(tvm::relay::tec::CCacheKey const&)
7: tvm::relay::tec::ShapeFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
6: tvm::relay::tec::MakeShapeFunc::Create(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
5: tvm::relay::tec::MakeShapeFunc::VisitExpr(tvm::RelayExpr const&)
4: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFNS_7runtime
1: tvm::relay::tec::MakeShapeFunc::VisitExpr_(tvm::relay::CallNode const*)
0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
3: TVMFuncCall
2: _ZNSt17_Function_handlerIFvN3tvm7run
1: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
0: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
4: TVMFuncCall
3: _ZNSt17_Function_handlerIFvN3tvm7run
2: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
1: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
0: tvm::runtime::Array<tvm::PrimExpr, void> tvm::runtime::TVMPODValue_::AsObjectRef<tvm::runtime::Array<tvm::PrimExpr, void> >() const

'''