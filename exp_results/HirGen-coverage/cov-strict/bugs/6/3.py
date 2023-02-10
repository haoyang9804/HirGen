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
var_7 = relay.var("var_7", dtype = "int32", shape = (1, 14, 10))#candidate|7|(1, 14, 10)|var|int32
var_8 = relay.var("var_8", dtype = "int32", shape = (6, 14, 10))#candidate|8|(6, 14, 10)|var|int32
bop_9 = relay.left_shift(var_7.astype('int32'), var_8.astype('int32')) # shape=(6, 14, 10)
output = bop_9
output2 = bop_9
func_15 = relay.Function([var_7,var_8,], output)
mod['func_15'] = func_15
mod = relay.transform.InferType()(mod)
var_16 = relay.var("var_16", dtype = "int32", shape = (1, 14, 10))#candidate|16|(1, 14, 10)|var|int32
var_17 = relay.var("var_17", dtype = "int32", shape = (6, 14, 10))#candidate|17|(6, 14, 10)|var|int32
output = func_15(var_16,var_17,)
func_18 = relay.Function([var_16,var_17,], output)
mutated_mod['func_18'] = func_18
mutated_mod = relay.transform.InferType()(mutated_mod)
var_20 = relay.var("var_20", dtype = "float64", shape = (7, 6))#candidate|20|(7, 6)|var|float64
var_21 = relay.var("var_21", dtype = "float64", shape = (7, 6))#candidate|21|(7, 6)|var|float64
bop_22 = relay.not_equal(var_20.astype('bool'), relay.reshape(var_21.astype('bool'), relay.shape_of(var_20))) # shape=(7, 6)
bop_27 = relay.divide(bop_22.astype('float32'), relay.reshape(var_21.astype('float32'), relay.shape_of(bop_22))) # shape=(7, 6)
bop_30 = relay.equal(var_20.astype('bool'), relay.reshape(bop_27.astype('bool'), relay.shape_of(var_20))) # shape=(7, 6)
bop_33 = relay.maximum(bop_27.astype('uint16'), relay.reshape(bop_22.astype('uint16'), relay.shape_of(bop_27))) # shape=(7, 6)
bop_36 = relay.subtract(bop_33.astype('uint16'), relay.reshape(bop_22.astype('uint16'), relay.shape_of(bop_33))) # shape=(7, 6)
bop_42 = relay.floor_mod(bop_36.astype('float32'), relay.reshape(bop_22.astype('float32'), relay.shape_of(bop_36))) # shape=(7, 6)
var_45 = relay.var("var_45", dtype = "float64", shape = (7, 6))#candidate|45|(7, 6)|var|float64
bop_46 = relay.left_shift(var_21.astype('uint8'), relay.reshape(var_45.astype('uint8'), relay.shape_of(var_21))) # shape=(7, 6)
uop_49 = relay.asinh(bop_22.astype('float64')) # shape=(7, 6)
var_55 = relay.var("var_55", dtype = "float64", shape = (7, 6))#candidate|55|(7, 6)|var|float64
bop_56 = relay.add(uop_49.astype('uint8'), relay.reshape(var_55.astype('uint8'), relay.shape_of(uop_49))) # shape=(7, 6)
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
var_60 = relay.var("var_60", dtype = "int32", shape = (140,))#candidate|60|(140,)|var|int32
var_61 = relay.var("var_61", dtype = "int32", shape = (840,))#candidate|61|(840,)|var|int32
call_59 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(var_61.astype('int32'), [6, 14, 10]), )
call_62 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(var_61.astype('int32'), [6, 14, 10]), )
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
call_66 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(call_59.astype('int32'), [6, 14, 10]), )
call_67 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(call_59.astype('int32'), [6, 14, 10]), )
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
call_70 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(var_61.astype('int32'), [6, 14, 10]), )
call_71 = func_15_call(relay.reshape(var_60.astype('int32'), [1, 14, 10]), relay.reshape(var_61.astype('int32'), [6, 14, 10]), )
bop_72 = relay.bitwise_and(uop_49.astype('int64'), relay.reshape(bop_46.astype('int64'), relay.shape_of(uop_49))) # shape=(7, 6)
output = relay.Tuple([bop_30,bop_42,bop_56,call_59,var_60,var_61,call_66,call_70,bop_72,])
output2 = relay.Tuple([bop_30,bop_42,bop_56,call_62,var_60,var_61,call_67,call_71,bop_72,])
func_75 = relay.Function([var_20,var_21,var_45,var_55,var_60,var_61,], output)
mod['func_75'] = func_75
mod = relay.transform.InferType()(mod)
var_76 = relay.var("var_76", dtype = "float64", shape = (7, 6))#candidate|76|(7, 6)|var|float64
var_77 = relay.var("var_77", dtype = "float64", shape = (7, 6))#candidate|77|(7, 6)|var|float64
var_78 = relay.var("var_78", dtype = "float64", shape = (7, 6))#candidate|78|(7, 6)|var|float64
var_79 = relay.var("var_79", dtype = "float64", shape = (7, 6))#candidate|79|(7, 6)|var|float64
var_80 = relay.var("var_80", dtype = "int32", shape = (140,))#candidate|80|(140,)|var|int32
var_81 = relay.var("var_81", dtype = "int32", shape = (840,))#candidate|81|(840,)|var|int32
output = func_75(var_76,var_77,var_78,var_79,var_80,var_81,)
func_82 = relay.Function([var_76,var_77,var_78,var_79,var_80,var_81,], output)
mutated_mod['func_82'] = func_82
mutated_mod = relay.transform.InferType()(mutated_mod)
const_87 = relay.const([[-9.236717,-3.375048,8.925025,-0.259823],[5.648386,-6.158099,-2.456800,-4.591862]], dtype = "float32")#candidate|87|(2, 4)|const|float32
uop_88 = relay.log10(const_87.astype('float32')) # shape=(2, 4)
bop_90 = relay.right_shift(uop_88.astype('uint16'), relay.reshape(const_87.astype('uint16'), relay.shape_of(uop_88))) # shape=(2, 4)
func_75_call = mod.get_global_var('func_75')
func_82_call = mutated_mod.get_global_var('func_82')
var_96 = relay.var("var_96", dtype = "float64", shape = (42,))#candidate|96|(42,)|var|float64
var_97 = relay.var("var_97", dtype = "int32", shape = (140,))#candidate|97|(140,)|var|int32
var_98 = relay.var("var_98", dtype = "int32", shape = (840,))#candidate|98|(840,)|var|int32
call_95 = relay.TupleGetItem(func_75_call(relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_97.astype('int32'), [140,]), relay.reshape(var_98.astype('int32'), [840,]), ), 2)
call_99 = relay.TupleGetItem(func_82_call(relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_96.astype('float64'), [7, 6]), relay.reshape(var_97.astype('int32'), [140,]), relay.reshape(var_98.astype('int32'), [840,]), ), 2)
uop_100 = relay.acosh(bop_90.astype('float32')) # shape=(2, 4)
bop_102 = relay.less(uop_100.astype('bool'), relay.reshape(bop_90.astype('bool'), relay.shape_of(uop_100))) # shape=(2, 4)
output = relay.Tuple([call_95,var_96,var_97,var_98,bop_102,])
output2 = relay.Tuple([call_99,var_96,var_97,var_98,bop_102,])
func_106 = relay.Function([var_96,var_97,var_98,], output)
mod['func_106'] = func_106
mod = relay.transform.InferType()(mod)
var_107 = relay.var("var_107", dtype = "float64", shape = (42,))#candidate|107|(42,)|var|float64
var_108 = relay.var("var_108", dtype = "int32", shape = (140,))#candidate|108|(140,)|var|int32
var_109 = relay.var("var_109", dtype = "int32", shape = (840,))#candidate|109|(840,)|var|int32
output = func_106(var_107,var_108,var_109,)
func_110 = relay.Function([var_107,var_108,var_109,], output)
mutated_mod['func_110'] = func_110
mutated_mod = relay.transform.InferType()(mutated_mod)
var_112 = relay.var("var_112", dtype = "int16", shape = ())#candidate|112|()|var|int16
var_113 = relay.var("var_113", dtype = "int16", shape = (8, 5))#candidate|113|(8, 5)|var|int16
bop_114 = relay.maximum(var_112.astype('int16'), var_113.astype('int16')) # shape=(8, 5)
bop_118 = relay.add(var_113.astype('int64'), var_112.astype('int64')) # shape=(8, 5)
bop_122 = relay.greater(var_112.astype('bool'), var_113.astype('bool')) # shape=(8, 5)
bop_125 = relay.floor_divide(bop_114.astype('float32'), relay.reshape(bop_118.astype('float32'), relay.shape_of(bop_114))) # shape=(8, 5)
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
const_129 = relay.const([10,1,3,3,1,-9,10,-4,-9,8,-6,2,-4,9,-8,5,10,-7,9,7,6,1,-3,4,9,-10,6,10,-8,4,-3,-6,-5,4,-5,-4,-8,-6,1,10,5,6,9,1,-6,7,-6,-2,8,-2,8,5,-2,-6,-2,-2,6,10,5,7,6,1,-7,-2,-9,-9,2,-5,-4,9,5,2,-1,-9,4,10,-7,4,3,8,-10,-1,10,9,6,7,9,-5,4,9,-9,9,-7,2,9,-4,-9,-3,2,1,10,-9,9,-1,7,10,6,-2,2,-5,-3,10,-7,-1,5,-8,9,-5,-3,-3,6,-6,3,-9,1,-6,4,-8,-6,-1,10,7,-9,10,-1,-2,-6,-3,-6,6], dtype = "int32")#candidate|129|(140,)|const|int32
var_130 = relay.var("var_130", dtype = "int32", shape = (840,))#candidate|130|(840,)|var|int32
call_128 = func_15_call(relay.reshape(const_129.astype('int32'), [1, 14, 10]), relay.reshape(var_130.astype('int32'), [6, 14, 10]), )
call_131 = func_15_call(relay.reshape(const_129.astype('int32'), [1, 14, 10]), relay.reshape(var_130.astype('int32'), [6, 14, 10]), )
uop_132 = relay.atan(bop_118.astype('float32')) # shape=(8, 5)
uop_135 = relay.log2(uop_132.astype('float32')) # shape=(8, 5)
uop_141 = relay.sin(uop_135.astype('float64')) # shape=(8, 5)
bop_144 = relay.less(uop_135.astype('bool'), var_112.astype('bool')) # shape=(8, 5)
bop_147 = relay.minimum(uop_141.astype('int32'), relay.reshape(bop_114.astype('int32'), relay.shape_of(uop_141))) # shape=(8, 5)
uop_150 = relay.sigmoid(bop_147.astype('float64')) # shape=(8, 5)
uop_152 = relay.asinh(uop_150.astype('float32')) # shape=(8, 5)
output = relay.Tuple([bop_122,bop_125,call_128,const_129,var_130,bop_144,uop_152,])
output2 = relay.Tuple([bop_122,bop_125,call_131,const_129,var_130,bop_144,uop_152,])
func_154 = relay.Function([var_112,var_113,var_130,], output)
mod['func_154'] = func_154
mod = relay.transform.InferType()(mod)
mutated_mod['func_154'] = func_154
mutated_mod = relay.transform.InferType()(mutated_mod)
func_154_call = mutated_mod.get_global_var('func_154')
var_156 = relay.var("var_156", dtype = "int16", shape = ())#candidate|156|()|var|int16
var_157 = relay.var("var_157", dtype = "int16", shape = (8, 5))#candidate|157|(8, 5)|var|int16
var_158 = relay.var("var_158", dtype = "int32", shape = (840,))#candidate|158|(840,)|var|int32
call_155 = func_154_call(var_156,var_157,var_158,)
output = call_155
func_159 = relay.Function([var_156,var_157,var_158,], output)
mutated_mod['func_159'] = func_159
mutated_mod = relay.transform.InferType()(mutated_mod)
var_176 = relay.var("var_176", dtype = "float64", shape = (7,))#candidate|176|(7,)|var|float64
uop_177 = relay.erf(var_176.astype('float64')) # shape=(7,)
bop_179 = relay.minimum(uop_177.astype('int64'), relay.reshape(var_176.astype('int64'), relay.shape_of(uop_177))) # shape=(7,)
bop_182 = relay.mod(bop_179.astype('float64'), relay.reshape(var_176.astype('float64'), relay.shape_of(bop_179))) # shape=(7,)
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
const_190 = relay.const([7,5,8,3,-4,1,-10,8,4,9,6,-7,3,-1,-5,-10,10,10,-3,-5,-7,6,5,-7,5,2,-1,-2,2,7,2,2,5,9,4,4,-10,-9,4,3,-2,-1,-5,9,-6,5,-6,-9,10,6,4,5,10,-9,10,4,-9,6,9,1,4,1,-4,8,10,8,-5,8,-7,-10,8,-10,10,-3,3,-4,2,4,8,2,9,-2,2,-10,6,9,-7,-5,7,5,1,-7,-7,-1,-5,2,2,8,5,-5,9,9,3,2,9,1,9,5,9,10,8,7,8,-3,5,-8,-2,-10,-8,-6,-2,-8,-5,4,8,9,5,-3,-3,-10,6,3,-2,4,5,-6,7,5,-7,-5], dtype = "int32")#candidate|190|(140,)|const|int32
var_191 = relay.var("var_191", dtype = "int32", shape = (840,))#candidate|191|(840,)|var|int32
call_189 = func_15_call(relay.reshape(const_190.astype('int32'), [1, 14, 10]), relay.reshape(var_191.astype('int32'), [6, 14, 10]), )
call_192 = func_15_call(relay.reshape(const_190.astype('int32'), [1, 14, 10]), relay.reshape(var_191.astype('int32'), [6, 14, 10]), )
var_193 = relay.var("var_193", dtype = "float64", shape = (7,))#candidate|193|(7,)|var|float64
bop_194 = relay.not_equal(bop_182.astype('bool'), relay.reshape(var_193.astype('bool'), relay.shape_of(bop_182))) # shape=(7,)
bop_197 = relay.bitwise_and(bop_194.astype('int32'), relay.reshape(bop_179.astype('int32'), relay.shape_of(bop_194))) # shape=(7,)
output = relay.Tuple([call_189,const_190,var_191,bop_197,])
output2 = relay.Tuple([call_192,const_190,var_191,bop_197,])
func_200 = relay.Function([var_176,var_191,var_193,], output)
mod['func_200'] = func_200
mod = relay.transform.InferType()(mod)
var_201 = relay.var("var_201", dtype = "float64", shape = (7,))#candidate|201|(7,)|var|float64
var_202 = relay.var("var_202", dtype = "int32", shape = (840,))#candidate|202|(840,)|var|int32
var_203 = relay.var("var_203", dtype = "float64", shape = (7,))#candidate|203|(7,)|var|float64
output = func_200(var_201,var_202,var_203,)
func_204 = relay.Function([var_201,var_202,var_203,], output)
mutated_mod['func_204'] = func_204
mutated_mod = relay.transform.InferType()(mutated_mod)
var_209 = relay.var("var_209", dtype = "float32", shape = (15,))#candidate|209|(15,)|var|float32
uop_210 = relay.exp(var_209.astype('float32')) # shape=(15,)
uop_212 = relay.rsqrt(uop_210.astype('float32')) # shape=(15,)
bop_214 = relay.less_equal(var_209.astype('bool'), relay.reshape(uop_212.astype('bool'), relay.shape_of(var_209))) # shape=(15,)
func_75_call = mod.get_global_var('func_75')
func_82_call = mutated_mod.get_global_var('func_82')
const_218 = relay.const([2.735840,6.453557,-8.955449,-1.554443,3.402678,-9.790050,5.159291,1.572008,1.121386,-2.158834,-4.856665,7.717315,-0.039966,1.117490,-6.800465,2.153620,-8.224434,-3.071231,-9.816360,0.918216,6.829341,-9.947532,-0.506761,-2.134629,-6.697584,6.035515,0.997464,4.974568,-1.592605,-9.958542,7.329196,5.699482,-2.334154,-9.503055,-3.085470,-8.406712,8.874010,-4.014224,-2.896659,-5.522753,4.305478,-3.189708], dtype = "float64")#candidate|218|(42,)|const|float64
const_219 = relay.const([[9,2,9,-9,7,6,-2,-9,-5,-5,7,3,-4,8,-6,-2,8,-5,3,-9,-9,2,-1,-4,-9,9,2,9,-4,2,9,3,10,-4,1,3,-8,3,-7,8,-6,1,10,4,7,1,1,-5,5,4,-5,-8,-2,-4,8,5,8,1,-8,-3,-4,2,2,8,9,3,-9,-8,-10,7,4,-1,10,4,1,-2,4,2,-3,3,-2,-7,7,-1,-8,6,-2,4,-3,7,-6,10,-5,7,-5,-10,-9,8,-2,7,-9,10,2,-3,-4,10,1,1,-4,-2,9,6,8,-5,4,8,7,-5,6,5,-8,10,-6,-5,-7,7,-10,-7,-10,-1,-4,-8,-9,-1,10,-2,10,-10,-3,7]], dtype = "int32")#candidate|219|(1, 140)|const|int32
var_220 = relay.var("var_220", dtype = "int32", shape = (840,))#candidate|220|(840,)|var|int32
call_217 = relay.TupleGetItem(func_75_call(relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_219.astype('int32'), [140,]), relay.reshape(var_220.astype('int32'), [840,]), ), 7)
call_221 = relay.TupleGetItem(func_82_call(relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_218.astype('float64'), [7, 6]), relay.reshape(const_219.astype('int32'), [140,]), relay.reshape(var_220.astype('int32'), [840,]), ), 7)
bop_222 = relay.equal(bop_214.astype('bool'), relay.reshape(var_209.astype('bool'), relay.shape_of(bop_214))) # shape=(15,)
uop_225 = relay.asin(bop_214.astype('float32')) # shape=(15,)
uop_227 = relay.atan(uop_225.astype('float32')) # shape=(15,)
bop_232 = relay.maximum(uop_225.astype('int16'), relay.reshape(uop_210.astype('int16'), relay.shape_of(uop_225))) # shape=(15,)
uop_235 = relay.acos(uop_227.astype('float32')) # shape=(15,)
var_237 = relay.var("var_237", dtype = "float32", shape = (15,))#candidate|237|(15,)|var|float32
bop_238 = relay.bitwise_or(uop_227.astype('uint16'), relay.reshape(var_237.astype('uint16'), relay.shape_of(uop_227))) # shape=(15,)
func_106_call = mod.get_global_var('func_106')
func_110_call = mutated_mod.get_global_var('func_110')
call_241 = relay.TupleGetItem(func_106_call(relay.reshape(const_218.astype('float64'), [42,]), relay.reshape(const_219.astype('int32'), [140,]), relay.reshape(var_220.astype('int32'), [840,]), ), 1)
call_242 = relay.TupleGetItem(func_110_call(relay.reshape(const_218.astype('float64'), [42,]), relay.reshape(const_219.astype('int32'), [140,]), relay.reshape(var_220.astype('int32'), [840,]), ), 1)
bop_243 = relay.not_equal(uop_235.astype('bool'), relay.reshape(bop_222.astype('bool'), relay.shape_of(uop_235))) # shape=(15,)
uop_252 = relay.tan(uop_227.astype('float32')) # shape=(15,)
bop_255 = relay.subtract(uop_235.astype('int32'), relay.reshape(bop_214.astype('int32'), relay.shape_of(uop_235))) # shape=(15,)
func_154_call = mod.get_global_var('func_154')
func_159_call = mutated_mod.get_global_var('func_159')
const_259 = relay.const(-5, dtype = "int16")#candidate|259|()|const|int16
const_260 = relay.const([-9,8,-9,5,3,2,-5,-4,10,2,10,5,6,-3,2,-6,10,8,-3,7,-10,-5,5,-4,5,4,7,-1,-3,-2,6,-5,-4,-6,-5,4,-3,10,8,7], dtype = "int16")#candidate|260|(40,)|const|int16
call_258 = relay.TupleGetItem(func_154_call(relay.reshape(const_259.astype('int16'), []), relay.reshape(const_260.astype('int16'), [8, 5]), relay.reshape(var_220.astype('int32'), [840,]), ), 0)
call_261 = relay.TupleGetItem(func_159_call(relay.reshape(const_259.astype('int16'), []), relay.reshape(const_260.astype('int16'), [8, 5]), relay.reshape(var_220.astype('int32'), [840,]), ), 0)
uop_262 = relay.asin(bop_238.astype('float64')) # shape=(15,)
output = relay.Tuple([call_217,const_218,const_219,var_220,bop_232,call_241,bop_243,uop_252,bop_255,call_258,const_259,const_260,uop_262,])
output2 = relay.Tuple([call_221,const_218,const_219,var_220,bop_232,call_242,bop_243,uop_252,bop_255,call_261,const_259,const_260,uop_262,])
func_267 = relay.Function([var_209,var_220,var_237,], output)
mod['func_267'] = func_267
mod = relay.transform.InferType()(mod)
mutated_mod['func_267'] = func_267
mutated_mod = relay.transform.InferType()(mutated_mod)
func_267_call = mutated_mod.get_global_var('func_267')
var_269 = relay.var("var_269", dtype = "float32", shape = (15,))#candidate|269|(15,)|var|float32
var_270 = relay.var("var_270", dtype = "int32", shape = (840,))#candidate|270|(840,)|var|int32
var_271 = relay.var("var_271", dtype = "float32", shape = (15,))#candidate|271|(15,)|var|float32
call_268 = func_267_call(var_269,var_270,var_271,)
output = call_268
func_272 = relay.Function([var_269,var_270,var_271,], output)
mutated_mod['func_272'] = func_272
mutated_mod = relay.transform.InferType()(mutated_mod)
var_319 = relay.var("var_319", dtype = "int32", shape = (8, 12))#candidate|319|(8, 12)|var|int32
var_320 = relay.var("var_320", dtype = "int32", shape = (8, 12))#candidate|320|(8, 12)|var|int32
bop_321 = relay.maximum(var_319.astype('int32'), relay.reshape(var_320.astype('int32'), relay.shape_of(var_319))) # shape=(8, 12)
const_326 = relay.const([[-9,10,-8,-7,-3,-8,-5,4,7,-4,-1,-8],[4,-2,6,-10,4,-7,3,9,-2,-3,-4,-10],[-1,1,-9,-3,-4,1,-8,5,1,6,5,-3],[2,5,7,6,2,-7,8,-4,-2,2,-4,-4],[10,-10,-5,1,-1,3,4,3,1,-4,10,1],[1,-4,4,-1,-9,-7,-5,2,-7,-3,6,-6],[6,6,-8,-7,-2,-5,8,9,10,-2,6,-1],[10,-3,-8,5,-9,-9,8,7,7,-6,-9,-2]], dtype = "int32")#candidate|326|(8, 12)|const|int32
bop_327 = relay.less(bop_321.astype('bool'), relay.reshape(const_326.astype('bool'), relay.shape_of(bop_321))) # shape=(8, 12)
output = bop_327
output2 = bop_327
func_330 = relay.Function([var_319,var_320,], output)
mod['func_330'] = func_330
mod = relay.transform.InferType()(mod)
mutated_mod['func_330'] = func_330
mutated_mod = relay.transform.InferType()(mutated_mod)
func_330_call = mutated_mod.get_global_var('func_330')
var_332 = relay.var("var_332", dtype = "int32", shape = (8, 12))#candidate|332|(8, 12)|var|int32
var_333 = relay.var("var_333", dtype = "int32", shape = (8, 12))#candidate|333|(8, 12)|var|int32
call_331 = func_330_call(var_332,var_333,)
output = call_331
func_334 = relay.Function([var_332,var_333,], output)
mutated_mod['func_334'] = func_334
mutated_mod = relay.transform.InferType()(mutated_mod)
var_336 = relay.var("var_336", dtype = "int16", shape = (16, 5))#candidate|336|(16, 5)|var|int16
var_337 = relay.var("var_337", dtype = "int16", shape = (16, 5))#candidate|337|(16, 5)|var|int16
bop_338 = relay.add(var_336.astype('int16'), relay.reshape(var_337.astype('int16'), relay.shape_of(var_336))) # shape=(16, 5)
uop_341 = relay.atanh(bop_338.astype('float32')) # shape=(16, 5)
output = uop_341
output2 = uop_341
func_344 = relay.Function([var_336,var_337,], output)
mod['func_344'] = func_344
mod = relay.transform.InferType()(mod)
mutated_mod['func_344'] = func_344
mutated_mod = relay.transform.InferType()(mutated_mod)
func_344_call = mutated_mod.get_global_var('func_344')
var_346 = relay.var("var_346", dtype = "int16", shape = (16, 5))#candidate|346|(16, 5)|var|int16
var_347 = relay.var("var_347", dtype = "int16", shape = (16, 5))#candidate|347|(16, 5)|var|int16
call_345 = func_344_call(var_346,var_347,)
output = call_345
func_348 = relay.Function([var_346,var_347,], output)
mutated_mod['func_348'] = func_348
mutated_mod = relay.transform.InferType()(mutated_mod)
var_370 = relay.var("var_370", dtype = "uint32", shape = (14,))#candidate|370|(14,)|var|uint32
var_371 = relay.var("var_371", dtype = "uint32", shape = (14,))#candidate|371|(14,)|var|uint32
bop_372 = relay.maximum(var_370.astype('uint32'), relay.reshape(var_371.astype('uint32'), relay.shape_of(var_370))) # shape=(14,)
uop_375 = relay.sigmoid(var_371.astype('float64')) # shape=(14,)
const_384 = relay.const([5.681710,4.922285,-6.014711,-5.158004,-1.644129,9.906172,4.650483,5.997581,-6.364006,9.293828,-9.467610,2.738030,9.332474,-1.823492], dtype = "float64")#candidate|384|(14,)|const|float64
bop_385 = relay.minimum(uop_375.astype('float64'), relay.reshape(const_384.astype('float64'), relay.shape_of(uop_375))) # shape=(14,)
bop_391 = relay.floor_divide(bop_385.astype('float64'), relay.reshape(uop_375.astype('float64'), relay.shape_of(bop_385))) # shape=(14,)
bop_394 = relay.left_shift(bop_372.astype('uint64'), relay.reshape(var_371.astype('uint64'), relay.shape_of(bop_372))) # shape=(14,)
uop_399 = relay.log(bop_394.astype('float32')) # shape=(14,)
func_106_call = mod.get_global_var('func_106')
func_110_call = mutated_mod.get_global_var('func_110')
const_405 = relay.const([3.566164,-3.265962,9.416851,-4.209865,6.184564,4.986298,-9.157637,6.751179,8.667903,0.550682,5.127426,-6.188330,-7.033002,-8.644179,-1.304746,1.612316,1.679825,1.110505,4.224822,1.453607,-2.044802,-2.191332,-3.087730,0.353237,-0.874383,-6.004283,1.593483,-5.217565,-5.341599,9.836795,2.640277,-6.307595,-3.912381,4.833759,-9.546164,-1.120469,6.992988,7.943373,0.817296,-2.532084,7.761665,2.157514], dtype = "float64")#candidate|405|(42,)|const|float64
const_406 = relay.const([-7,9,6,7,6,-2,-2,7,5,-2,6,-10,-1,-1,8,-2,-2,-10,3,-10,3,-1,-5,8,2,4,6,-3,4,-7,-4,9,8,-6,-7,1,-8,-6,3,-4,-3,-3,4,-8,-1,-3,2,3,-1,4,-5,9,-9,8,-10,7,3,-2,7,7,5,-7,2,-8,10,3,6,1,-9,-7,5,2,-5,7,6,-4,-2,7,-2,10,-6,-9,7,8,-3,4,-9,6,-5,-8,4,4,-4,1,8,9,-6,-6,2,-8,-7,1,-1,1,-9,-8,2,1,7,-1,-5,8,-1,-2,-8,-6,-3,4,3,1,-4,-4,-3,10,7,6,2,-3,10,-3,-6,-5,5,2,-5,-9,3,7,8,-7], dtype = "int32")#candidate|406|(140,)|const|int32
const_407 = relay.const([-9,7,-5,7,-7,-9,1,-4,-4,-9,-9,-8,-8,6,-3,8,-9,4,-4,-3,6,-5,-8,4,-9,-6,-3,7,-3,9,-9,8,6,3,6,2,8,9,-8,9,-3,-10,-9,10,-1,8,7,-9,1,-4,-5,-7,6,10,-8,-5,-4,-9,-1,7,-9,-1,2,-1,9,-7,4,1,1,-3,-2,4,-3,-3,4,-6,4,1,-7,-8,-2,5,3,5,9,5,-6,1,3,-6,3,-10,-9,-4,5,-5,9,2,-7,-7,5,-10,-2,2,3,2,10,-3,-3,7,9,7,-3,-1,4,-7,-2,9,7,1,9,-6,-5,10,-5,8,-1,3,3,-6,-4,5,-9,-1,-10,-8,8,-8,2,3,-4,-1,-5,9,-8,7,8,-9,-10,7,5,-7,-7,5,4,9,8,-5,6,-3,10,8,3,9,8,-1,9,-3,4,8,-6,10,-6,-7,10,7,-1,10,-2,5,-4,10,-4,-10,3,-10,2,2,-6,10,-10,-8,2,3,8,4,4,-5,-7,9,-4,8,9,-6,9,-1,-3,-8,-8,10,10,-7,-8,-3,9,6,2,-9,-1,6,-1,9,-9,-10,9,-2,-4,1,3,-8,7,6,-5,2,10,-8,-6,-4,5,1,-3,-2,6,-10,9,-3,-9,10,6,4,10,6,10,-3,3,10,1,-3,-7,-5,-1,7,-7,-8,6,6,3,-2,-1,9,-10,-2,-2,5,-9,6,-9,2,9,-7,-1,1,-7,8,-9,-6,7,-10,-2,-6,7,1,-4,-2,-4,-8,-3,8,1,2,2,-10,3,-7,5,5,5,10,10,4,10,-6,2,-3,-9,-2,-2,-1,-2,-9,2,-1,-10,10,-2,2,8,-4,-4,-1,3,-6,-8,7,7,-9,8,6,2,7,-10,-5,-9,-9,8,-7,-5,-4,-2,-7,5,-6,7,-2,-9,6,-10,-8,-9,-3,6,6,3,-8,-10,-3,-10,-9,-4,9,-4,3,-5,-4,-9,-3,-8,-10,-8,-7,-3,5,-1,4,7,-9,8,4,6,-10,-8,-1,2,9,-10,-10,7,10,-10,-5,10,8,7,-3,-4,7,-8,6,2,1,2,-6,-5,-10,3,-1,-1,-2,3,-3,-10,6,-1,-7,-10,6,10,-2,10,-9,7,-5,9,9,3,-8,6,-1,8,1,-8,1,-9,-2,-5,2,-1,2,-2,-3,-7,-8,-6,-10,-8,-4,8,-2,-8,-5,-3,5,-4,5,10,-3,-10,-9,-1,-3,-2,3,2,-8,-10,-1,2,9,5,-5,4,3,10,-2,-3,-10,-4,6,8,4,10,1,3,9,-5,-2,-4,-7,-7,5,-6,6,8,5,7,-4,-5,9,6,-6,4,-4,-3,5,1,-9,-7,-8,6,-10,3,9,8,5,-8,7,-9,2,-1,1,9,1,-6,8,2,-6,-2,2,5,-4,-8,-4,-9,8,-3,3,2,5,-1,-1,-3,-8,4,-4,7,1,-6,-6,-3,-8,5,8,3,5,-10,-9,-3,-7,8,3,7,3,-9,2,-3,-8,-2,-6,-4,1,7,-5,8,5,-3,-1,4,-6,5,7,-8,8,3,-5,7,9,7,-4,7,-2,10,9,-9,9,2,-4,8,8,-8,-8,-8,-8,-10,1,2,8,-7,5,1,-2,9,3,7,-9,-4,-7,4,9,-8,5,-4,-3,9,8,2,-5,3,-8,-1,3,-8,10,-8,7,-1,-7,5,-5,7,-6,-3,-2,1,7,1,-7,-2,10,-10,-7,9,-6,-1,-9,-6,9,-6,-6,-1,8,4,-10,4,8,6,5,-7,-2,-3,9,7,-9,-9,9,9,-3,-5,-9,2,5,3,7,-4,-10,-1,-8,-9,1,-3,7,10,-5,4,7,1,5,-3,-4,1,-6,7,9,3,1,2,8,-1,-8,6,10,-7,-9,-9,-9,-9,3,9,1,3,7,-3,7,7,8,9,9,4,10,3,6,-4,-9,2,-4,1,10,8,5,-3,9,2,9,7,1,-9,-3,7,-10,9,6,7,3,3,1,2,9,-9,2,-10,-6,-10,-3,-1,8,7,4,-9,-9,7,-10,9,5,-10,4,9,-8,-10,-10,4,-3,-1,6,5,4,-7,-3,6,4,6,-3,-9,6,3,-10,-3,9,1,-3,3,8,-4,6,-8,5,10,9,-8,-9,-10,-8,-2,-6,-6,-9,6,1,-8,-1,9,10,-3,-9,3,-4,4,5,-10,-3,10,7], dtype = "int32")#candidate|407|(840,)|const|int32
call_404 = relay.TupleGetItem(func_106_call(relay.reshape(const_405.astype('float64'), [42,]), relay.reshape(const_406.astype('int32'), [140,]), relay.reshape(const_407.astype('int32'), [840,]), ), 2)
call_408 = relay.TupleGetItem(func_110_call(relay.reshape(const_405.astype('float64'), [42,]), relay.reshape(const_406.astype('int32'), [140,]), relay.reshape(const_407.astype('int32'), [840,]), ), 2)
output = relay.Tuple([bop_391,uop_399,call_404,const_405,const_406,const_407,])
output2 = relay.Tuple([bop_391,uop_399,call_408,const_405,const_406,const_407,])
func_409 = relay.Function([var_370,var_371,], output)
mod['func_409'] = func_409
mod = relay.transform.InferType()(mod)
mutated_mod['func_409'] = func_409
mutated_mod = relay.transform.InferType()(mutated_mod)
func_409_call = mutated_mod.get_global_var('func_409')
var_411 = relay.var("var_411", dtype = "uint32", shape = (14,))#candidate|411|(14,)|var|uint32
var_412 = relay.var("var_412", dtype = "uint32", shape = (14,))#candidate|412|(14,)|var|uint32
call_410 = func_409_call(var_411,var_412,)
output = call_410
func_413 = relay.Function([var_411,var_412,], output)
mutated_mod['func_413'] = func_413
mutated_mod = relay.transform.InferType()(mutated_mod)
const_438 = relay.const([[[-3,2,-2,-8,2,3,-10,-5,-6,8],[6,4,9,4,4,-1,-8,-7,-7,5],[5,-5,4,3,2,1,1,3,-3,3],[2,-4,-6,1,8,-5,3,-9,-1,-8],[4,-10,-5,4,-1,-5,10,3,10,1],[9,10,5,10,-2,8,6,-10,2,-2],[10,8,-7,-7,-5,7,-2,6,8,3],[4,7,-3,4,-10,10,4,-1,-9,9],[9,-5,-5,-10,9,-5,7,3,-1,-7],[3,-1,5,7,-4,-7,-2,-4,5,2],[-9,7,10,-8,-8,2,-2,8,6,-1],[-10,3,-3,1,-1,4,-9,-6,2,10]],[[8,-1,6,6,-3,-4,7,6,-4,-8],[7,5,-2,7,-6,3,-8,4,4,-5],[9,1,-7,-4,4,-8,-5,-9,7,-10],[-10,6,-7,-3,10,-7,9,3,-5,-9],[-2,-9,1,3,-3,-10,3,-2,-6,-1],[-3,7,6,-8,-8,9,-5,6,-8,1],[-9,-4,-8,-5,1,5,6,-6,2,3],[-2,-10,1,-2,-3,10,-5,-8,-1,9],[1,-8,6,7,-9,-7,-3,7,-3,4],[-1,1,1,8,2,-5,-3,-8,-6,-6],[5,3,10,-3,8,-2,4,-4,-1,-6],[-2,-6,4,4,10,5,3,7,-10,8]],[[-10,5,-2,-8,3,-7,-8,2,8,-4],[-5,-6,-6,2,3,-4,-10,8,9,-4],[1,7,-6,-3,-2,7,-3,8,5,-7],[-1,5,9,-1,9,-9,1,-2,-3,-3],[10,-6,4,6,3,-10,-4,6,-8,2],[-4,-8,8,-4,-4,8,9,-9,-8,-2],[8,-7,-7,-8,2,2,2,3,-9,1],[-3,9,4,5,-5,-5,9,7,-7,-10],[-4,6,-2,6,-5,-7,7,-2,-6,-5],[4,-5,-3,8,7,10,-8,-9,-5,-2],[-1,1,-4,9,10,2,3,4,10,-6],[9,10,2,8,2,8,-10,1,1,-6]],[[3,4,5,-2,1,-2,3,8,9,3],[-7,-5,1,2,8,-5,-8,-7,-1,8],[8,-2,9,-8,-9,2,-3,4,-4,6],[-6,3,1,7,-10,4,4,-8,-8,-1],[-7,-5,-1,2,7,7,10,9,1,-3],[-2,8,-5,-9,3,1,5,10,-5,1],[-2,-7,-8,4,-4,-10,-4,-4,-5,1],[5,7,8,-5,-2,6,5,-6,10,2],[-9,-4,-3,1,1,7,5,-7,-3,-5],[3,9,-5,-10,-7,10,-9,8,-5,9],[2,-4,-9,-8,-7,7,4,-5,3,3],[2,-3,-10,-2,-9,-10,6,-7,-7,9]],[[-8,4,-6,-1,6,8,-1,-8,3,5],[2,2,3,5,4,4,4,-6,-6,-9],[2,2,-8,4,2,-3,9,2,-7,-10],[3,-5,-3,-7,-7,-3,4,4,-10,3],[-2,-8,-6,-10,9,3,5,9,8,-9],[4,6,9,-2,-10,1,-5,-2,-9,-3],[-2,10,-4,-9,9,-7,-2,-5,-5,-2],[-5,-9,-1,-1,-1,1,-1,-5,4,-7],[-2,7,7,4,10,9,8,6,-8,4],[-1,5,7,-7,5,-5,-8,10,-4,-9],[10,-8,7,2,10,5,10,-3,1,10],[-6,3,-7,1,-7,6,10,8,7,-3]],[[5,-8,9,3,-3,-5,8,-3,1,6],[6,-7,-7,4,-4,-1,-4,-3,-2,2],[-1,-2,-7,4,9,-7,-4,-3,-9,5],[5,-4,8,-1,-4,-4,2,-4,3,10],[-9,-8,3,-6,-10,8,8,6,9,-6],[2,10,8,-4,3,6,3,9,-7,8],[5,-5,-10,-10,-6,-4,6,4,-9,8],[4,-5,8,-1,-8,-2,-7,-7,8,5],[-4,4,-2,-6,7,-2,8,8,-1,2],[-9,4,3,-2,9,-5,-6,-4,5,9],[6,5,-1,9,-3,3,7,-8,1,9],[5,4,1,10,-8,2,-2,-7,1,7]],[[8,3,-9,6,-1,1,-9,9,-3,-7],[-10,2,-7,7,6,9,-5,-9,-3,6],[8,-2,-2,-6,-10,5,-4,-3,7,-9],[7,-7,2,1,-9,-1,-6,-3,-2,5],[-5,-3,9,-2,-10,-9,10,-9,2,-10],[-7,-7,4,3,-10,-9,7,6,9,-4],[10,6,-1,-4,7,3,5,-8,1,-4],[-8,-8,4,-2,-1,-10,10,-10,-10,-4],[-6,-10,-7,1,-10,1,-7,-2,2,-2],[10,1,-7,-3,-5,6,1,4,6,8],[-10,1,-5,-6,9,-9,9,1,2,5],[-8,-1,-8,-9,-2,-10,5,10,6,-8]]], dtype = "uint64")#candidate|438|(7, 12, 10)|const|uint64
var_439 = relay.var("var_439", dtype = "uint64", shape = (7, 12, 10))#candidate|439|(7, 12, 10)|var|uint64
bop_440 = relay.multiply(const_438.astype('uint64'), relay.reshape(var_439.astype('uint64'), relay.shape_of(const_438))) # shape=(7, 12, 10)
var_445 = relay.var("var_445", dtype = "uint64", shape = (7, 12, 10))#candidate|445|(7, 12, 10)|var|uint64
bop_446 = relay.equal(bop_440.astype('bool'), relay.reshape(var_445.astype('bool'), relay.shape_of(bop_440))) # shape=(7, 12, 10)
uop_461 = relay.asinh(bop_446.astype('float64')) # shape=(7, 12, 10)
func_409_call = mod.get_global_var('func_409')
func_413_call = mutated_mod.get_global_var('func_413')
var_464 = relay.var("var_464", dtype = "uint32", shape = (14, 1))#candidate|464|(14, 1)|var|uint32
call_463 = relay.TupleGetItem(func_409_call(relay.reshape(var_464.astype('uint32'), [14,]), relay.reshape(var_464.astype('uint32'), [14,]), ), 0)
call_465 = relay.TupleGetItem(func_413_call(relay.reshape(var_464.astype('uint32'), [14,]), relay.reshape(var_464.astype('uint32'), [14,]), ), 0)
bop_467 = relay.less_equal(var_445.astype('bool'), relay.reshape(bop_446.astype('bool'), relay.shape_of(var_445))) # shape=(7, 12, 10)
bop_470 = relay.bitwise_xor(var_464.astype('int8'), relay.reshape(call_463.astype('int8'), relay.shape_of(var_464))) # shape=(14, 1)
bop_473 = relay.bitwise_xor(var_464.astype('int8'), relay.reshape(call_465.astype('int8'), relay.shape_of(var_464))) # shape=(14, 1)
bop_475 = relay.greater_equal(var_439.astype('bool'), relay.reshape(bop_446.astype('bool'), relay.shape_of(var_439))) # shape=(7, 12, 10)
uop_479 = relay.erf(uop_461.astype('float64')) # shape=(7, 12, 10)
bop_481 = relay.logical_and(bop_475.astype('bool'), relay.reshape(const_438.astype('bool'), relay.shape_of(bop_475))) # shape=(7, 12, 10)
bop_485 = relay.minimum(uop_479.astype('int32'), relay.reshape(bop_481.astype('int32'), relay.shape_of(uop_479))) # shape=(7, 12, 10)
bop_488 = relay.left_shift(bop_475.astype('uint32'), relay.reshape(var_439.astype('uint32'), relay.shape_of(bop_475))) # shape=(7, 12, 10)
bop_494 = relay.less(bop_485.astype('bool'), relay.reshape(bop_488.astype('bool'), relay.shape_of(bop_485))) # shape=(7, 12, 10)
uop_497 = relay.asinh(uop_479.astype('float32')) # shape=(7, 12, 10)
bop_499 = relay.logical_or(var_445.astype('bool'), relay.reshape(bop_475.astype('bool'), relay.shape_of(var_445))) # shape=(7, 12, 10)
uop_505 = relay.exp(uop_497.astype('float64')) # shape=(7, 12, 10)
bop_509 = relay.not_equal(bop_494.astype('bool'), relay.reshape(bop_446.astype('bool'), relay.shape_of(bop_494))) # shape=(7, 12, 10)
bop_514 = relay.bitwise_xor(uop_505.astype('uint8'), relay.reshape(bop_499.astype('uint8'), relay.shape_of(uop_505))) # shape=(7, 12, 10)
uop_517 = relay.sqrt(bop_485.astype('float64')) # shape=(7, 12, 10)
bop_521 = relay.less(uop_497.astype('bool'), relay.reshape(uop_505.astype('bool'), relay.shape_of(uop_497))) # shape=(7, 12, 10)
uop_524 = relay.acosh(uop_517.astype('float32')) # shape=(7, 12, 10)
output = relay.Tuple([bop_467,bop_470,bop_509,bop_514,bop_521,uop_524,])
output2 = relay.Tuple([bop_467,bop_473,bop_509,bop_514,bop_521,uop_524,])
func_526 = relay.Function([var_439,var_445,var_464,], output)
mod['func_526'] = func_526
mod = relay.transform.InferType()(mod)
var_527 = relay.var("var_527", dtype = "uint64", shape = (7, 12, 10))#candidate|527|(7, 12, 10)|var|uint64
var_528 = relay.var("var_528", dtype = "uint64", shape = (7, 12, 10))#candidate|528|(7, 12, 10)|var|uint64
var_529 = relay.var("var_529", dtype = "uint32", shape = (14, 1))#candidate|529|(14, 1)|var|uint32
output = func_526(var_527,var_528,var_529,)
func_530 = relay.Function([var_527,var_528,var_529,], output)
mutated_mod['func_530'] = func_530
mutated_mod = relay.transform.InferType()(mutated_mod)
var_558 = relay.var("var_558", dtype = "int16", shape = ())#candidate|558|()|var|int16
var_559 = relay.var("var_559", dtype = "int16", shape = (12, 12, 1))#candidate|559|(12, 12, 1)|var|int16
bop_560 = relay.less(var_558.astype('bool'), var_559.astype('bool')) # shape=(12, 12, 1)
const_566 = relay.const([[[2,-1,-3,4],[-5,4,-2,5],[-7,6,8,-10],[7,5,-2,3],[-6,-5,-7,2],[-9,-9,1,-4],[8,2,7,6],[-1,4,-9,-9],[3,3,4,-3],[2,-5,-7,-1],[-9,10,7,2],[8,-3,-2,3]],[[-8,-7,-2,7],[-2,8,1,4],[-4,-8,2,-9],[7,-1,-6,-6],[3,-9,4,10],[8,5,1,1],[-6,6,-2,-2],[1,-8,8,-9],[7,-5,-4,7],[4,3,-3,-1],[-5,9,-6,-6],[4,1,-8,-10]],[[-10,8,-1,8],[-6,1,7,-8],[-2,10,8,4],[2,6,10,6],[3,4,6,-9],[2,2,3,2],[7,-3,9,-8],[7,-10,-2,-4],[2,-8,1,1],[-7,-3,-2,-8],[-8,-5,10,-8],[9,-3,6,-5]],[[7,-7,1,2],[-4,10,9,-5],[1,-5,-7,-10],[-5,-9,-10,1],[8,-6,9,-8],[-2,-9,6,-1],[5,-8,1,9],[2,-8,2,2],[9,-10,-5,-1],[7,-9,3,-10],[7,1,9,-5],[7,3,-5,1]],[[2,-2,8,-3],[3,-7,-3,-5],[8,-8,7,6],[-7,5,-4,5],[6,3,3,10],[-10,-2,-8,5],[6,-3,-7,9],[7,-7,10,7],[9,3,-5,-8],[2,10,-1,-5],[8,-10,-6,8],[-4,-6,-7,2]],[[10,2,10,10],[1,1,-5,-9],[-4,-8,9,-6],[3,7,3,-9],[-7,-8,-10,9],[10,-5,4,-8],[2,-9,-5,4],[-4,-9,-2,-2],[9,8,-10,-1],[4,-7,-10,9],[2,5,-2,5],[-1,5,-3,6]],[[8,-1,8,-1],[10,3,3,-5],[9,-4,10,6],[-6,-9,-7,10],[7,-6,-5,-9],[2,-8,3,2],[3,2,-5,-9],[9,-9,2,-4],[5,6,-4,7],[-9,3,-6,6],[10,5,6,8],[1,-4,1,2]],[[9,-5,4,-10],[-7,-9,-9,6],[-10,-9,-4,-4],[6,-8,9,2],[9,-2,1,10],[-7,3,-1,-5],[10,1,-3,-1],[-4,-8,10,-10],[9,-5,-4,-8],[-1,6,2,5],[7,9,1,-9],[-2,6,6,2]],[[8,6,2,10],[9,-5,9,-4],[-8,-4,-5,-3],[-1,9,-7,-3],[-8,-6,4,-4],[-3,-3,-3,-7],[-10,-9,6,-9],[3,4,-9,-4],[7,9,-5,6],[2,-8,-8,5],[8,2,5,10],[-1,7,-1,-9]],[[8,-4,9,-3],[-9,7,8,5],[7,9,-1,10],[-1,-2,-8,3],[-8,-1,6,1],[6,5,1,9],[7,10,7,10],[8,2,-2,4],[-3,7,8,7],[6,-8,-1,-7],[-7,8,8,1],[9,-2,2,3]],[[10,7,9,-3],[-5,4,-10,-6],[-7,-10,-2,2],[8,-6,-1,-6],[-8,10,6,-4],[6,6,-8,-2],[9,-3,5,-4],[-1,-3,-9,1],[3,-5,-5,-9],[7,-7,-2,-4],[-5,1,2,4],[6,-6,-8,2]],[[-10,-2,8,-6],[-9,-3,-5,1],[4,9,9,-3],[-3,-10,6,4],[-1,-1,-8,-3],[1,9,3,-6],[-1,-6,10,-5],[-8,6,9,2],[-4,8,-1,8],[-8,-9,-6,1],[2,10,-7,-7],[-10,-1,-10,-8]]], dtype = "int16")#candidate|566|(12, 12, 4)|const|int16
bop_567 = relay.divide(var_559.astype('float64'), const_566.astype('float64')) # shape=(12, 12, 4)
func_267_call = mod.get_global_var('func_267')
func_272_call = mutated_mod.get_global_var('func_272')
var_571 = relay.var("var_571", dtype = "float32", shape = (1, 15))#candidate|571|(1, 15)|var|float32
const_572 = relay.const([5,9,7,4,-8,3,2,2,1,-3,-10,-8,-3,-3,-10,-1,4,-8,9,-5,7,-8,-7,3,4,-10,4,-9,-10,8,-9,-7,-10,-3,9,5,-8,-6,-7,3,-6,4,-1,-9,-7,3,5,-10,6,-5,3,-2,-6,6,-7,3,-1,3,3,-3,8,-6,-7,-1,2,-10,7,6,8,7,10,-9,8,-3,6,9,3,9,4,4,6,-2,4,8,-9,-9,4,10,-2,1,8,-1,7,-10,10,8,-8,10,-10,5,-3,10,2,7,2,3,-10,-2,4,10,6,3,10,-5,2,-10,4,-2,10,2,9,-4,10,6,10,10,-3,9,6,2,3,5,-9,-6,6,3,-3,-4,4,9,-3,-7,3,10,9,-7,3,10,9,-3,5,-2,-2,3,6,8,-10,8,3,3,1,4,6,-5,1,-1,3,-5,2,9,-10,1,-3,5,8,-8,10,10,8,-2,-3,-5,8,1,10,2,-8,9,-2,7,-4,9,-7,-1,-6,5,-4,8,-2,6,9,-1,-10,6,5,-4,-8,-5,3,9,-7,-9,-5,-4,5,1,7,1,-10,-9,-2,-6,-7,-3,-4,-10,-9,3,7,2,-7,-6,-5,-4,-3,2,8,-6,-8,7,-3,5,7,7,5,-6,2,-9,-8,-2,5,-6,3,-10,-1,4,-4,-8,-7,5,-4,-6,-4,3,-7,-4,-10,-9,-4,-5,8,1,-8,-2,-3,7,-9,-4,-10,-6,-4,3,-5,-1,3,-6,-5,2,-7,10,-3,-7,-2,7,10,8,8,-9,4,-1,-6,-8,3,4,-4,2,-3,-2,-2,6,-4,5,8,4,4,8,2,10,-4,9,-1,5,-10,9,-6,1,9,-4,1,-8,-5,6,-2,7,5,6,-5,-9,8,6,-3,5,-6,-7,-1,4,1,5,8,-9,-8,-8,6,9,-7,9,5,-10,-5,7,-3,-3,-3,-10,-2,6,6,4,-3,-7,9,8,9,2,8,3,-10,-8,7,-4,6,1,-3,-3,8,-7,-5,-9,-5,-10,10,-4,10,-3,-9,5,8,5,-1,-10,-8,3,3,8,3,-3,-3,-5,-9,9,-9,3,-2,-9,-9,-9,-5,10,6,-6,-10,-4,-5,6,-1,-4,-1,10,-7,4,7,3,-8,9,-5,-5,-5,-7,-1,-6,8,-5,-9,2,-3,2,-4,3,9,7,9,8,2,4,-10,8,-2,-9,-7,7,-9,-3,3,6,9,8,5,4,3,3,-10,-1,5,3,3,3,10,-7,-1,5,-4,8,8,1,1,7,4,-4,-10,-10,-9,-6,-8,-2,-4,8,9,-1,9,-2,-5,-2,7,-9,-2,-9,-3,-10,8,9,8,-1,-5,-7,4,5,-8,-8,10,1,8,5,-5,-2,6,6,-1,2,-9,10,-10,-1,-1,-1,-8,-5,2,-8,4,1,-5,3,-4,-2,5,6,-9,2,-10,3,7,4,9,-10,6,-4,-8,7,-3,7,-2,-3,-2,1,10,-9,5,8,-9,-8,10,4,-8,-2,-9,-8,3,-2,-8,6,8,-5,-1,8,9,-3,-7,-7,-9,7,-9,10,8,-5,-6,4,-6,-8,2,-5,1,2,-8,-2,-9,6,6,7,8,10,-5,-10,3,-8,-2,-8,4,1,-8,-10,4,4,-3,-3,5,7,-10,-2,2,10,6,9,7,-3,-4,1,-10,7,-7,4,-8,3,7,2,9,5,1,7,-5,4,-3,7,-1,-7,-9,-2,6,-4,3,2,10,-9,-1,-1,-7,6,-2,-5,1,1,7,-6,1,5,8,5,-5,10,-1,4,8,9,-5,2,10,-6,9,9,-2,3,10,-7,-4,-2,-10,4,8,-5,-8,1,-2,-5,8,10,-8,9,1,-2,-5,-2,3,-10,-9,-4,-3,7,9,-6,-5,7,9,3,6,-7,-2,-5,-4,3,-3,2,-9,1,2,7,6,-3,3,-5,1,-9,10,7,-2,-9,7,-1,2,-8,-1,-2,-2,2,-10,4,8,4,9,5,-2,-6,-8,7,1,4,10,-1,-10,-8,9,-2,5,-8,-5,-6,9,-6,-9,9,-8,-3,1,-3,-8,-3,-2,5,-9,4,8,-3,-3,4,-3,3,-5,2,-2,-7,7,-7,-3,8,7,-9,3,-10,2,10,6,-9,-2,7,7,4,1,2,-6,-9,-4,8,2,-4,3,-2,-7,-9,5,3,-9,-5,-8,8,1,9,-6,3,-8,-4,2,1], dtype = "int32")#candidate|572|(840,)|const|int32
call_570 = relay.TupleGetItem(func_267_call(relay.reshape(var_571.astype('float32'), [15,]), relay.reshape(const_572.astype('int32'), [840,]), relay.reshape(var_571.astype('float32'), [15,]), ), 7)
call_573 = relay.TupleGetItem(func_272_call(relay.reshape(var_571.astype('float32'), [15,]), relay.reshape(const_572.astype('int32'), [840,]), relay.reshape(var_571.astype('float32'), [15,]), ), 7)
func_526_call = mod.get_global_var('func_526')
func_530_call = mutated_mod.get_global_var('func_530')
const_579 = relay.const([1,8,2,-10,-7,-7,4,-5,2,4,2,-1,-3,-8], dtype = "uint32")#candidate|579|(14,)|const|uint32
call_578 = relay.TupleGetItem(func_526_call(relay.reshape(const_572.astype('uint64'), [7, 12, 10]), relay.reshape(const_572.astype('uint64'), [7, 12, 10]), relay.reshape(const_579.astype('uint32'), [14, 1]), ), 2)
call_580 = relay.TupleGetItem(func_530_call(relay.reshape(const_572.astype('uint64'), [7, 12, 10]), relay.reshape(const_572.astype('uint64'), [7, 12, 10]), relay.reshape(const_579.astype('uint32'), [14, 1]), ), 2)
uop_586 = relay.log10(bop_567.astype('float64')) # shape=(12, 12, 4)
const_589 = relay.const([[[-2.481132,9.386763,3.747266,-7.901614],[0.377247,5.554600,0.703565,-8.166870],[-3.922345,-9.409568,1.328584,9.051009],[-8.575281,9.453939,4.562935,-6.195111],[8.216510,-8.504036,9.779176,7.065219],[9.444161,-5.401222,-3.644299,3.220692],[-1.293561,3.028201,4.730259,3.756211],[9.400500,8.440754,5.098089,-6.494975],[-6.662020,5.530458,4.503518,-7.979721],[8.615838,7.297409,8.757536,-7.223158],[7.216179,3.784387,-0.491496,-4.438040],[-9.077791,-8.877724,0.063308,-7.619877]],[[-1.499599,-4.843160,-7.254770,-1.923272],[6.892606,-0.861965,4.086597,-5.180791],[-0.054553,9.905790,-7.711197,-3.720362],[-8.863554,-2.392794,-9.620303,6.746595],[-9.222846,-9.155920,-6.714016,-9.602881],[2.784203,1.547747,-7.353892,2.018841],[8.801458,5.897038,-9.111155,-8.079066],[8.383202,-7.494233,-2.466214,-5.867138],[2.797145,5.555272,1.714278,3.282327],[-0.307583,1.954976,7.951149,-8.273747],[-9.559727,-6.190534,1.299737,7.013525],[-5.286040,-3.024815,4.305848,6.902797]],[[-4.289957,-2.468736,7.715679,-3.862155],[-5.291847,-3.815188,-9.007431,3.419693],[-8.663631,9.707349,-9.230997,-1.253981],[-2.926409,7.983187,4.779934,4.652184],[-6.565840,-0.544034,1.658893,8.038309],[-3.344339,8.039413,-2.968957,5.835008],[8.089485,8.961319,1.343307,-8.150989],[-4.146419,5.048365,-4.634297,6.772855],[9.275004,-5.928332,-8.190254,-6.552178],[7.783660,5.461534,-5.706428,7.863617],[5.057731,-2.948194,-7.725121,8.763061],[-7.854141,5.995492,1.962471,1.553836]],[[-8.974120,8.123726,-5.342896,6.465359],[-5.077263,2.939945,-5.518300,-5.923188],[0.427553,-9.135042,9.111567,0.687906],[4.447686,-5.180618,4.782868,3.464648],[-2.978854,4.693421,-6.403733,-4.362866],[1.595536,-7.575801,-7.980049,-0.686292],[9.945743,-5.329995,-3.721968,-8.177799],[-5.351813,-8.702250,7.988772,4.311185],[-4.941052,-2.569767,7.651985,8.951531],[-5.183082,-4.661850,-5.564815,5.805735],[-9.109437,-6.122008,6.640037,-3.569904],[-6.120254,7.315132,-2.808328,6.271530]],[[7.510617,7.709653,-9.111384,-2.217867],[-4.439716,-7.493017,5.299435,-2.780994],[2.295913,-8.420557,-4.581579,-7.898785],[3.490815,-1.385099,-7.219423,-6.747315],[9.824502,-5.516042,3.264135,-9.159123],[-9.171880,-9.202348,-8.779815,7.075536],[-7.411958,-3.570282,-5.564114,1.101498],[6.212256,6.603569,4.107492,1.543815],[9.807268,-1.166788,-9.812800,6.631420],[9.312706,-2.628981,-8.393761,5.495040],[4.466488,6.796440,-1.064277,1.151865],[-8.909086,7.308848,2.908932,-9.027675]],[[8.657108,7.086905,-0.740269,-4.714958],[-2.668082,0.016357,-9.107117,6.123695],[1.438551,4.224738,1.443317,8.960304],[8.090042,0.548283,7.009404,8.186017],[3.377153,5.599450,9.497218,8.365817],[-7.970243,7.468339,-0.689484,3.688675],[5.442069,9.065466,2.982194,9.744948],[-7.137706,4.611538,-6.339665,-4.954059],[3.448278,6.151078,1.409030,4.657543],[-2.102143,-7.612973,3.332469,-3.692028],[-7.251841,-3.106100,-8.250954,-9.628423],[-4.438528,-8.871814,-2.620735,-2.855853]],[[-9.199547,5.423261,6.993506,-2.304942],[1.078127,-8.657615,-1.885846,4.245417],[-4.822712,-7.822819,3.914678,4.012197],[8.552078,6.401814,-9.713997,8.940785],[9.383140,-8.707584,5.583827,-8.026873],[1.862700,-8.349022,7.887191,-1.965601],[5.197508,-5.127775,5.685510,-3.805619],[-1.925706,2.216757,-2.589545,8.522571],[4.533464,-6.985861,-6.195386,1.476177],[6.486040,5.620119,3.223206,-0.654434],[4.442720,-1.049887,3.030664,5.649024],[-1.725460,-8.003464,1.873283,-2.271637]],[[-5.329904,-5.330905,-4.638011,-2.978453],[6.475817,-2.954474,-0.413176,6.543826],[-2.064591,-2.051599,-7.978548,3.049296],[-9.264317,1.509984,4.103090,-0.445081],[-9.328384,8.173113,-8.949332,-3.709277],[7.541256,7.579900,-9.328692,2.397326],[-2.785462,-0.785291,5.808878,7.915036],[8.979209,-7.926526,-5.830737,-4.740600],[3.275841,6.721456,-3.127730,7.693762],[5.937649,1.557945,0.508418,-9.821915],[-9.406468,-6.003702,-9.090530,-8.069155],[9.240005,-4.918374,8.852238,-2.318764]],[[-4.253107,3.764570,-8.691927,3.085873],[3.347504,1.833410,8.775525,-8.449948],[-0.939657,-5.123573,-9.581904,-1.536063],[1.219828,0.922568,-1.257821,8.636100],[-0.469813,4.571659,7.824343,-1.927608],[4.468170,3.856303,7.960860,-8.587473],[-3.006354,-2.248005,6.495290,-3.257178],[9.054371,-2.110329,-9.257611,5.468722],[-8.113672,6.338658,6.629398,-4.573611],[2.782075,5.500115,5.091844,-9.987258],[-1.413000,-1.440270,7.561422,-3.684530],[-5.725330,-0.768066,-2.659906,3.443053]],[[-2.045891,-4.940244,-7.742031,-8.465061],[-2.557863,-2.001034,-6.684598,-3.589979],[7.040976,-3.190595,1.068461,-3.338314],[6.271342,-2.480512,-0.320893,-4.697098],[-4.789062,-6.548018,9.256818,-4.948860],[-8.532709,3.554385,4.916347,-1.504479],[-2.122498,-8.362433,1.292321,-3.902566],[8.489005,6.318956,-5.777668,4.511500],[-5.103239,-7.971974,-7.338047,-5.861313],[8.105960,6.955179,6.529641,-3.889787],[4.420104,5.695428,2.728058,9.136551],[9.594966,-3.986123,0.726582,-5.527058]],[[1.784951,7.941492,-2.176802,-9.703751],[-8.920077,-7.339988,4.451371,4.090722],[8.632059,-9.036405,-7.563210,-1.301338],[-1.781317,-5.007941,-6.734606,7.311545],[2.684183,7.758470,-9.721999,5.510577],[-9.094489,-2.438053,-8.997514,-8.551553],[9.859403,-2.991882,-6.097000,2.956895],[4.958566,-9.994425,-5.885246,3.682623],[0.574743,3.895128,3.218221,-8.389597],[6.740837,1.953239,-2.847269,9.622043],[-5.235711,-9.395213,-3.691599,4.723386],[7.958487,2.371904,-4.429373,7.197717]],[[-8.001983,7.329820,9.709514,3.418102],[1.370749,6.006228,-2.520910,9.713766],[-3.420849,5.142039,4.658494,6.432266],[9.280474,-4.770531,8.286122,4.782577],[4.020340,7.497000,-3.231404,1.007358],[-4.531022,4.804343,7.024898,0.284574],[-9.771703,6.158964,-2.062703,-8.976306],[6.396639,3.868252,-6.879838,-4.311273],[7.792397,5.958111,-2.823213,2.257285],[7.003119,0.732027,6.030919,-3.489249],[-4.511319,9.640724,-8.096319,-1.814935],[6.771618,8.122074,-2.547390,0.466363]]], dtype = "float64")#candidate|589|(12, 12, 4)|const|float64
bop_590 = relay.subtract(uop_586.astype('int8'), relay.reshape(const_589.astype('int8'), relay.shape_of(uop_586))) # shape=(12, 12, 4)
bop_595 = relay.floor_mod(uop_586.astype('float32'), relay.reshape(const_589.astype('float32'), relay.shape_of(uop_586))) # shape=(12, 12, 4)
var_599 = relay.var("var_599", dtype = "float64", shape = (12, 12, 4))#candidate|599|(12, 12, 4)|var|float64
bop_600 = relay.less(bop_567.astype('bool'), relay.reshape(var_599.astype('bool'), relay.shape_of(bop_567))) # shape=(12, 12, 4)
func_154_call = mod.get_global_var('func_154')
func_159_call = mutated_mod.get_global_var('func_159')
const_608 = relay.const([[2,-2,10,2,8,3,4,8,-2,-4,-1,9,1,-7,-3,-3,5,-2,5,-2],[-1,-9,-3,-9,-1,-1,-7,2,-2,-8,8,-4,7,7,-9,5,4,-4,-10,1]], dtype = "int16")#candidate|608|(2, 20)|const|int16
call_607 = relay.TupleGetItem(func_154_call(relay.reshape(var_558.astype('int16'), []), relay.reshape(const_608.astype('int16'), [8, 5]), relay.reshape(const_572.astype('int32'), [840,]), ), 5)
call_609 = relay.TupleGetItem(func_159_call(relay.reshape(var_558.astype('int16'), []), relay.reshape(const_608.astype('int16'), [8, 5]), relay.reshape(const_572.astype('int32'), [840,]), ), 5)
uop_610 = relay.cos(bop_590.astype('float32')) # shape=(12, 12, 4)
uop_612 = relay.asinh(uop_610.astype('float64')) # shape=(12, 12, 4)
uop_614 = relay.asin(uop_612.astype('float64')) # shape=(12, 12, 4)
var_617 = relay.var("var_617", dtype = "float64", shape = (12, 12, 4))#candidate|617|(12, 12, 4)|var|float64
bop_618 = relay.add(uop_612.astype('float32'), relay.reshape(var_617.astype('float32'), relay.shape_of(uop_612))) # shape=(12, 12, 4)
output = relay.Tuple([bop_560,call_570,var_571,const_572,call_578,const_579,bop_595,bop_600,call_607,const_608,uop_614,bop_618,])
output2 = relay.Tuple([bop_560,call_573,var_571,const_572,call_580,const_579,bop_595,bop_600,call_609,const_608,uop_614,bop_618,])
func_622 = relay.Function([var_558,var_559,var_571,var_599,var_617,], output)
mod['func_622'] = func_622
mod = relay.transform.InferType()(mod)
mutated_mod['func_622'] = func_622
mutated_mod = relay.transform.InferType()(mutated_mod)
func_622_call = mutated_mod.get_global_var('func_622')
var_624 = relay.var("var_624", dtype = "int16", shape = ())#candidate|624|()|var|int16
var_625 = relay.var("var_625", dtype = "int16", shape = (12, 12, 1))#candidate|625|(12, 12, 1)|var|int16
var_626 = relay.var("var_626", dtype = "float32", shape = (1, 15))#candidate|626|(1, 15)|var|float32
var_627 = relay.var("var_627", dtype = "float64", shape = (12, 12, 4))#candidate|627|(12, 12, 4)|var|float64
var_628 = relay.var("var_628", dtype = "float64", shape = (12, 12, 4))#candidate|628|(12, 12, 4)|var|float64
call_623 = func_622_call(var_624,var_625,var_626,var_627,var_628,)
output = call_623
func_629 = relay.Function([var_624,var_625,var_626,var_627,var_628,], output)
mutated_mod['func_629'] = func_629
mutated_mod = relay.transform.InferType()(mutated_mod)
var_636 = relay.var("var_636", dtype = "float64", shape = (1, 5, 13))#candidate|636|(1, 5, 13)|var|float64
uop_637 = relay.tan(var_636.astype('float64')) # shape=(1, 5, 13)
uop_640 = relay.exp(uop_637.astype('float64')) # shape=(1, 5, 13)
bop_642 = relay.logical_xor(uop_637.astype('int32'), relay.reshape(uop_640.astype('int32'), relay.shape_of(uop_637))) # shape=(1, 5, 13)
bop_647 = relay.left_shift(uop_637.astype('uint8'), relay.reshape(var_636.astype('uint8'), relay.shape_of(uop_637))) # shape=(1, 5, 13)
bop_652 = relay.bitwise_or(bop_642.astype('int8'), relay.reshape(uop_637.astype('int8'), relay.shape_of(bop_642))) # shape=(1, 5, 13)
func_15_call = mod.get_global_var('func_15')
func_18_call = mutated_mod.get_global_var('func_18')
const_656 = relay.const([-10,-5,1,7,10,2,-4,4,6,-7,6,6,3,10,-10,1,3,-6,-1,6,10,9,-5,-5,4,-5,1,6,3,-6,-1,4,5,10,10,9,-6,-5,3,9,3,8,9,-8,10,-7,2,-4,-9,3,10,3,-2,4,-10,1,-1,7,-2,2,8,-2,-1,7,9,-1,-2,5,-3,-9,4,-6,10,-4,-8,-6,-6,-7,3,8,4,2,-4,-1,-8,1,4,-6,1,-7,-6,5,1,10,-6,7,10,9,6,10,-2,-5,-8,-1,3,-6,-2,3,-3,-7,-3,-10,-4,4,10,-6,-7,-1,6,-4,1,-7,2,-6,5,5,-8,1,-5,5,-1,10,-6,6,4,-10,-1,9,9,10], dtype = "int32")#candidate|656|(140,)|const|int32
const_657 = relay.const([-1,4,2,-9,7,-10,9,-9,-5,-7,7,3,3,1,7,-1,-8,2,-6,-7,-10,-9,10,-4,-8,-3,8,1,8,-7,2,3,4,-8,-4,-1,-1,-3,-8,-9,9,3,5,-1,8,8,-5,-9,-6,-9,-3,5,1,-4,-9,7,-4,7,6,6,5,5,-5,-1,3,-3,10,-10,-10,-2,-9,6,1,5,-3,-3,-5,10,-5,2,1,-4,-5,-2,9,3,2,3,5,1,1,6,-4,-7,-7,9,-6,9,-7,6,4,4,-1,9,7,5,8,1,10,-3,8,-5,4,6,4,7,-3,3,-10,-1,-4,2,-8,6,-2,4,-10,4,2,5,-2,-1,-3,-9,7,-9,-4,5,-5,-9,9,-7,-2,-2,3,-10,10,-9,6,-6,-10,5,6,-8,-10,-10,-10,-5,9,7,-5,-6,6,7,-5,-8,10,9,-3,8,3,2,6,6,-7,9,9,6,-3,9,8,3,1,3,-10,-9,1,-8,1,6,8,7,-1,3,7,-3,9,10,-1,-8,-6,-6,10,-5,2,6,-7,5,7,-5,2,-9,4,-5,-2,8,9,8,9,-8,-7,1,9,-3,6,3,5,-10,-9,6,7,-3,9,6,5,1,10,5,7,-8,-7,2,-10,5,-9,-3,4,-5,-5,2,-6,-1,2,6,5,6,-2,6,-5,1,1,-2,-10,-9,-9,5,4,4,4,-9,3,-1,-6,-8,7,4,-3,-4,-5,-8,-5,-2,2,7,5,5,1,-6,4,-5,3,-10,8,-9,-6,-1,-3,10,-10,-1,2,4,1,-10,2,-5,-7,-1,-8,1,-1,-7,-5,10,-3,-10,-7,10,-4,1,6,-3,1,-9,8,5,2,4,10,-3,-6,-9,7,9,8,9,7,8,-8,9,8,-5,-7,-8,-4,-9,3,-1,7,-2,-1,-10,5,-8,-8,-8,2,-7,-2,-3,7,7,10,-10,10,-4,5,9,4,10,6,-10,-1,5,-3,10,-2,10,4,9,-4,-5,8,6,3,6,5,-6,-5,10,1,8,4,5,9,-2,-8,-1,-2,2,4,5,-4,-4,5,-6,3,-10,-5,-6,1,-9,-6,-9,9,4,-9,-1,-1,3,10,-2,7,-9,-9,9,5,-2,8,9,-1,1,1,7,6,10,9,-8,-6,-9,5,4,5,1,7,3,3,10,-3,-10,6,8,1,-4,-4,-9,-10,-1,1,10,-1,8,4,-10,6,9,-7,7,-4,5,5,-9,1,2,9,1,-5,-6,8,10,-3,6,-6,9,6,-8,-10,-8,6,-2,-1,6,4,7,-7,4,5,-1,-2,3,-6,1,-7,-6,8,-9,-3,-7,7,6,10,-6,-1,8,-9,3,1,1,2,10,-5,1,5,2,-1,-8,2,7,-10,9,5,5,8,3,-3,-2,-3,-6,-2,-2,9,-5,1,4,-9,1,5,-10,-1,-4,8,-1,-8,1,7,2,-3,-7,-1,8,-2,-5,-4,-1,-10,9,1,-10,9,9,-5,2,-2,-6,9,-4,-2,9,4,5,3,1,-6,-4,-2,-10,6,7,-2,10,7,-2,2,7,9,1,5,-7,3,-5,4,2,-9,7,-3,1,-10,-9,2,10,9,5,-3,5,6,1,-7,-10,-3,6,-4,6,-9,6,-2,-5,8,-6,-6,7,3,10,10,-4,6,10,8,4,2,1,10,6,-7,9,5,3,-5,-6,2,-9,-2,10,7,-7,6,3,-3,-4,-3,-4,-10,-4,10,1,10,2,-10,-1,-6,4,-6,8,7,-8,6,7,-8,2,-1,-2,2,4,-6,-9,-3,6,1,7,-5,-10,-8,8,10,-6,2,3,-1,6,-3,-2,9,6,-2,-9,1,-10,8,6,8,10,-1,-6,1,-2,8,-1,-1,-8,-1,10,8,-2,7,-6,-2,-7,-1,-5,-6,9,2,10,8,-8,6,-10,7,-1,-9,-1,3,-4,-10,9,8,7,-7,-5,6,-5,-1,-5,6,-5,2,8,2,2,-2,8,-8,-2,-4,1,2,10,-8,4,-2,-10,4,2,1,10,6,3,-2,8,-4,6,2,9,-3,-5,-3,1,-7,3,-8,9,-10,-7,-9,-10,3,-9,2,-4,-1,1,-8,-1,-5,2,9,7,-6,-4,9,-8,-10,7,4,-10,-2,1,1,-6,-6,2,-10,5,-3,-2,2,7,6,-7,-5,5,6,-9,8,3,-4,7,4,-3,-10,-3], dtype = "int32")#candidate|657|(840,)|const|int32
call_655 = func_15_call(relay.reshape(const_656.astype('int32'), [1, 14, 10]), relay.reshape(const_657.astype('int32'), [6, 14, 10]), )
call_658 = func_15_call(relay.reshape(const_656.astype('int32'), [1, 14, 10]), relay.reshape(const_657.astype('int32'), [6, 14, 10]), )
func_330_call = mod.get_global_var('func_330')
func_334_call = mutated_mod.get_global_var('func_334')
var_660 = relay.var("var_660", dtype = "int32", shape = (96,))#candidate|660|(96,)|var|int32
call_659 = func_330_call(relay.reshape(var_660.astype('int32'), [8, 12]), relay.reshape(var_660.astype('int32'), [8, 12]), )
call_661 = func_330_call(relay.reshape(var_660.astype('int32'), [8, 12]), relay.reshape(var_660.astype('int32'), [8, 12]), )
bop_663 = relay.multiply(uop_640.astype('uint32'), relay.reshape(var_636.astype('uint32'), relay.shape_of(uop_640))) # shape=(1, 5, 13)
uop_667 = relay.log(uop_640.astype('float32')) # shape=(1, 5, 13)
output = relay.Tuple([bop_647,bop_652,call_655,const_656,const_657,call_659,var_660,bop_663,uop_667,])
output2 = relay.Tuple([bop_647,bop_652,call_658,const_656,const_657,call_661,var_660,bop_663,uop_667,])
func_669 = relay.Function([var_636,var_660,], output)
mod['func_669'] = func_669
mod = relay.transform.InferType()(mod)
var_670 = relay.var("var_670", dtype = "float64", shape = (1, 5, 13))#candidate|670|(1, 5, 13)|var|float64
var_671 = relay.var("var_671", dtype = "int32", shape = (96,))#candidate|671|(96,)|var|int32
output = func_669(var_670,var_671,)
func_672 = relay.Function([var_670,var_671,], output)
mutated_mod['func_672'] = func_672
mutated_mod = relay.transform.InferType()(mutated_mod)
var_692 = relay.var("var_692", dtype = "float64", shape = (5,))#candidate|692|(5,)|var|float64
uop_693 = relay.acosh(var_692.astype('float64')) # shape=(5,)
uop_695 = relay.exp(uop_693.astype('float64')) # shape=(5,)
func_669_call = mod.get_global_var('func_669')
func_672_call = mutated_mod.get_global_var('func_672')
const_698 = relay.const([2.406856,5.862667,-1.527600,-3.288859,-6.444213,4.197833,-5.405552,6.994641,2.014227,4.928695,0.657480,-1.880807,5.134763,1.048349,-2.140544,-5.761924,-8.281297,-2.534566,-2.888123,-5.508448,-3.214756,-1.699486,-1.381786,2.160874,7.085428,-2.783509,6.624022,-9.151127,-0.074238,9.642594,7.288622,-8.278142,5.777211,-0.625278,-8.355365,5.859055,-0.961345,4.130966,2.341809,-9.208361,3.401179,9.341718,-6.666644,-2.492455,4.772401,6.435311,5.340926,-7.483838,-4.802673,-5.299587,-1.561923,7.610703,4.324903,9.023213,2.218749,8.689813,2.073180,-4.684197,-8.798268,-5.915693,-9.004981,-6.248865,-6.600556,2.323179,-7.081564], dtype = "float64")#candidate|698|(65,)|const|float64
const_699 = relay.const([7,5,1,-4,-1,8,8,-10,-7,5,-8,-3,-5,5,7,-7,5,4,1,-7,-8,4,9,7,6,-2,-8,-3,1,-9,3,1,-10,-7,4,3,-1,10,-4,-7,-10,5,5,-2,-7,3,7,-9,2,7,-10,-7,-9,2,-3,-4,-10,-10,-8,-9,3,4,-9,-10,8,4,10,-10,8,4,1,-4,-6,3,-9,8,6,6,2,-6,-3,-10,-4,2,2,-3,4,3,-4,-4,1,-6,2,2,-8,-6], dtype = "int32")#candidate|699|(96,)|const|int32
call_697 = relay.TupleGetItem(func_669_call(relay.reshape(const_698.astype('float64'), [1, 5, 13]), relay.reshape(const_699.astype('int32'), [96,]), ), 5)
call_700 = relay.TupleGetItem(func_672_call(relay.reshape(const_698.astype('float64'), [1, 5, 13]), relay.reshape(const_699.astype('int32'), [96,]), ), 5)
func_154_call = mod.get_global_var('func_154')
func_159_call = mutated_mod.get_global_var('func_159')
var_704 = relay.var("var_704", dtype = "int16", shape = ())#candidate|704|()|var|int16
const_705 = relay.const([-5,5,-6,5,-5,-4,7,3,-8,-8,-10,8,2,-6,-10,1,7,3,8,-3,-5,5,-7,-8,5,4,2,9,-5,-4,-4,-4,2,-4,5,5,-4,-2,1,8], dtype = "int16")#candidate|705|(40,)|const|int16
const_706 = relay.const([1,5,5,5,-6,-6,-1,8,-1,3,-3,-8,-10,6,-5,9,6,8,-3,-3,-3,9,-7,9,10,-4,1,-9,-3,8,-1,5,-2,-1,7,3,-4,4,-1,-9,-9,3,7,2,6,9,8,5,7,4,-1,-4,1,-6,4,-3,-10,-2,10,-10,-3,-4,-1,10,6,-3,-1,9,6,-4,1,-5,6,2,1,-2,9,7,6,9,9,8,1,-4,3,9,-7,-8,2,-1,-2,-9,-9,-2,-4,-9,-4,-5,-2,-4,-6,10,-9,-8,-8,8,-10,-2,3,-7,8,-2,-1,-9,-5,-2,10,-6,-10,-2,1,-10,-6,3,-8,-8,6,-1,6,-2,7,-4,5,5,-4,6,6,-8,8,-4,4,7,2,7,-9,3,-4,-2,6,-6,4,-6,8,3,-7,-6,-8,3,10,2,-10,5,1,-6,-5,-7,-4,1,1,10,2,-1,9,-2,3,-8,8,-10,3,6,-6,-8,8,-4,5,6,-10,-10,10,-9,8,7,-3,5,-1,-5,-3,-9,6,9,7,-1,-4,-7,-2,7,-6,-8,-1,6,-3,9,7,1,1,-5,7,5,-6,8,-1,-10,1,-7,-10,-4,-7,10,-1,5,1,10,5,-1,-10,-9,-4,-9,6,7,2,-8,6,-10,9,9,-7,-9,-6,5,1,-8,-2,2,-4,1,-2,-4,-7,5,2,-9,5,-5,-10,7,-2,-6,4,2,9,-4,2,-1,-2,6,-2,-7,5,-8,-10,-6,-1,-9,-2,-3,-4,-6,-4,-5,3,4,-8,-2,3,-8,4,7,6,-10,-7,-3,-9,5,-6,-2,6,6,-7,7,-8,-4,8,-10,7,5,1,3,9,-5,-2,5,5,2,-10,3,-3,-10,8,5,10,-9,2,-2,4,5,6,-2,1,-4,-2,-9,-4,5,-1,-7,-2,1,-10,4,-2,-10,9,1,7,-1,2,3,8,-10,-10,-7,5,6,-5,-1,-9,5,-2,-8,-2,7,8,2,2,5,10,4,-4,-3,5,3,-5,-4,5,-8,-2,2,9,-5,7,2,-4,5,9,4,-1,7,-7,5,-4,10,-7,5,-6,6,-2,1,5,5,-3,9,8,-2,-9,-9,-9,9,9,4,4,-9,8,-4,3,6,-8,9,-3,3,7,2,-10,4,5,-5,9,7,3,10,-1,-1,3,7,3,5,1,-8,10,-10,-1,4,-10,-7,4,-10,-4,-6,-3,1,4,-5,-5,3,4,-2,10,-3,9,-5,7,-10,6,-6,-10,4,4,8,-2,-9,-6,-9,7,9,4,-10,9,10,5,-6,-4,8,2,1,9,-8,9,1,3,-3,-6,-6,5,2,-5,-7,-1,-8,3,6,-7,4,4,-3,7,-4,1,8,7,7,10,-10,1,5,7,-3,-8,10,4,10,1,6,3,9,-4,3,-10,-7,5,4,-5,-6,-6,6,-2,-5,10,7,5,-5,-4,-5,5,7,7,-10,8,5,4,8,-8,3,2,1,-7,8,-7,8,-6,10,6,-6,-9,10,4,3,1,-10,-5,-10,-7,4,3,4,6,-3,-7,9,-1,1,-7,7,-8,10,-6,-6,1,7,-3,-3,6,-3,6,10,9,5,6,3,-9,-9,-3,7,8,-8,-6,2,-7,-6,-5,2,-10,1,7,1,-8,10,-9,5,-3,-8,4,-1,-5,5,5,-3,2,-2,9,-6,8,-8,5,4,-8,8,-4,-4,8,-8,-7,-10,6,-2,9,-2,6,-6,3,10,4,-7,7,-7,3,9,-4,-10,5,5,-2,3,-7,6,8,-3,-3,1,2,-6,3,6,8,-10,-10,9,2,-7,-2,8,10,-2,-2,7,-3,2,2,-3,-3,-5,8,4,-7,-6,5,-9,-7,-2,-6,-4,3,4,3,-6,-10,9,5,7,-6,3,9,8,2,5,-8,10,-7,5,-4,6,1,4,-1,-10,-4,-10,-4,8,-8,-8,2,-8,-2,6,-5,2,10,-4,6,7,8,2,-6,9,-3,-2,5,-3,4,-3,-4,7,-10,6,-6,6,10,5,10,2,3,10,4,6,-5,-7,5,-4,4,1,-2,-3,8,-2,5,2,10,7,5,1,2,10,3,-4,-5,8,-3,4,10,10,10,5,-5,-3,6,2,-6,-7,8,8,8,3,-9,-1,-2,6,4,-1,6,9,2,9,-6,6,10,9,4,-9,10,-3,2,4,5,-3,3,-10,-5,-10,-7,-5], dtype = "int32")#candidate|706|(840,)|const|int32
call_703 = relay.TupleGetItem(func_154_call(relay.reshape(var_704.astype('int16'), []), relay.reshape(const_705.astype('int16'), [8, 5]), relay.reshape(const_706.astype('int32'), [840,]), ), 5)
call_707 = relay.TupleGetItem(func_159_call(relay.reshape(var_704.astype('int16'), []), relay.reshape(const_705.astype('int16'), [8, 5]), relay.reshape(const_706.astype('int32'), [840,]), ), 5)
bop_709 = relay.greater_equal(uop_695.astype('bool'), call_703.astype('bool')) # shape=(8, 5)
bop_712 = relay.greater_equal(uop_695.astype('bool'), call_707.astype('bool')) # shape=(8, 5)
bop_714 = relay.add(uop_693.astype('float32'), relay.reshape(uop_695.astype('float32'), relay.shape_of(uop_693))) # shape=(5,)
func_344_call = mod.get_global_var('func_344')
func_348_call = mutated_mod.get_global_var('func_348')
const_721 = relay.const([-6,10,-1,-7,-10,3,-6,-1,7,9,6,2,6,-8,-6,4,-5,-9,5,-9,4,-2,-5,7,10,1,-6,8,4,4,-7,3,3,8,9,2,9,8,-7,-8,-2,3,-2,10,6,-6,-4,-5,-1,4,10,5,-2,-7,-7,5,2,7,-10,-9,-5,-9,2,-6,7,-4,10,10,2,10,-2,1,5,-9,-7,1,-1,-8,4,-9], dtype = "int16")#candidate|721|(80,)|const|int16
call_720 = func_344_call(relay.reshape(const_721.astype('int16'), [16, 5]), relay.reshape(const_721.astype('int16'), [16, 5]), )
call_722 = func_344_call(relay.reshape(const_721.astype('int16'), [16, 5]), relay.reshape(const_721.astype('int16'), [16, 5]), )
uop_724 = relay.cosh(uop_693.astype('float32')) # shape=(5,)
func_330_call = mod.get_global_var('func_330')
func_334_call = mutated_mod.get_global_var('func_334')
call_726 = func_330_call(relay.reshape(const_699.astype('int32'), [8, 12]), relay.reshape(const_699.astype('int32'), [8, 12]), )
call_727 = func_330_call(relay.reshape(const_699.astype('int32'), [8, 12]), relay.reshape(const_699.astype('int32'), [8, 12]), )
bop_728 = relay.bitwise_and(uop_693.astype('int8'), relay.reshape(var_692.astype('int8'), relay.shape_of(uop_693))) # shape=(5,)
output = relay.Tuple([call_697,const_698,const_699,var_704,const_705,const_706,bop_709,bop_714,call_720,const_721,uop_724,call_726,bop_728,])
output2 = relay.Tuple([call_700,const_698,const_699,var_704,const_705,const_706,bop_712,bop_714,call_722,const_721,uop_724,call_727,bop_728,])
func_732 = relay.Function([var_692,var_704,], output)
mod['func_732'] = func_732
mod = relay.transform.InferType()(mod)
mutated_mod['func_732'] = func_732
mutated_mod = relay.transform.InferType()(mutated_mod)
func_732_call = mutated_mod.get_global_var('func_732')
var_734 = relay.var("var_734", dtype = "float64", shape = (5,))#candidate|734|(5,)|var|float64
var_735 = relay.var("var_735", dtype = "int16", shape = ())#candidate|735|()|var|int16
call_733 = func_732_call(var_734,var_735,)
output = call_733
func_736 = relay.Function([var_734,var_735,], output)
mutated_mod['func_736'] = func_736
mutated_mod = relay.transform.InferType()(mutated_mod)
var_745 = relay.var("var_745", dtype = "float32", shape = (5, 11))#candidate|745|(5, 11)|var|float32
uop_746 = relay.exp(var_745.astype('float32')) # shape=(5, 11)
uop_752 = relay.sin(uop_746.astype('float32')) # shape=(5, 11)
bop_754 = relay.bitwise_and(uop_746.astype('uint64'), relay.reshape(var_745.astype('uint64'), relay.shape_of(uop_746))) # shape=(5, 11)
bop_758 = relay.multiply(uop_752.astype('int64'), relay.reshape(var_745.astype('int64'), relay.shape_of(uop_752))) # shape=(5, 11)
output = relay.Tuple([bop_754,bop_758,])
output2 = relay.Tuple([bop_754,bop_758,])
F = relay.Function([var_745,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_745,], output2)
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
	relay.transform.ToGraphNormalForm(),
	relay.transform.SimplifyInference(),
	relay.transform.ToBasicBlockNormalForm(),
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
input_745= np.array([[-2.004632,9.129258,2.831018,2.651060,9.895296,-0.898575,5.820191,-4.696962,-9.617423,1.002484,8.342237],[-0.929271,-2.379543,-8.342871,4.415662,7.468160,2.101743,8.962473,9.169230,-7.559737,4.743566,0.720504],[-4.723215,-8.463406,-0.429682,9.596992,-7.481623,-7.224789,5.077157,0.506773,-0.359527,-5.001554,1.798650],[-7.399373,-6.393581,6.325510,-3.996189,9.764411,-1.466267,7.125900,2.219227,-1.909201,-5.296399,-0.913251],[-1.897771,-6.393828,1.747823,-9.232918,8.907187,2.519301,8.289275,8.701494,5.771952,0.848776,0.486663]], dtype='float32')
module1.set_input('var_745', input_745)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_745, )
res3 = intrp3.evaluate()(input_745, )
res4 = intrp4.evaluate()(input_745, )
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
module5.set_input('var_745', input_745)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_745, )
res7 = intrp7.evaluate()(input_745, )
res8 = intrp8.evaluate()(input_745, )
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
module9.set_input('var_745', input_745)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_745, )
res11 = intrp11.evaluate()(input_745, )
res12 = intrp12.evaluate()(input_745, )
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
module13.set_input('var_745', input_745)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_745, )
res15 = intrp15.evaluate()(input_745, )
res16 = intrp16.evaluate()(input_745, )
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
module17.set_input('var_745', input_745)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_745, )
res19 = intrp19.evaluate()(input_745, )
res20 = intrp20.evaluate()(input_745, )
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
module21.set_input('var_745', input_745)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_745, )
res23 = intrp23.evaluate()(input_745, )
res24 = intrp24.evaluate()(input_745, )
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

'''22: TVMFuncCall
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
16: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
14: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
13: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
12: _ZNSt17_Function_handlerIFSt10sha
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''