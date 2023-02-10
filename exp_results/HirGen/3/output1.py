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
var_0 = relay.var("var_0", dtype = "int16", shape = (8,))#candidate|0|(8,)|var|int16
var_1 = relay.var("var_1", dtype = "int16", shape = (8,))#candidate|1|(8,)|var|int16
bop_2 = relay.equal(var_0.astype('bool'), relay.reshape(var_1.astype('bool'), relay.shape_of(var_0))) # shape=(8,)
bop_6 = relay.left_shift(var_1.astype('int16'), relay.reshape(var_0.astype('int16'), relay.shape_of(var_1))) # shape=(8,)
const_9 = relay.const([-7,8,10,9,-10,-8,9,9], dtype = "int16")#candidate|9|(8,)|const|int16
bop_10 = relay.bitwise_or(bop_6.astype('int16'), relay.reshape(const_9.astype('int16'), relay.shape_of(bop_6))) # shape=(8,)
bop_13 = relay.right_shift(bop_2.astype('uint8'), relay.reshape(var_1.astype('uint8'), relay.shape_of(bop_2))) # shape=(8,)
var_17 = relay.var("var_17", dtype = "bool", shape = (8,))#candidate|17|(8,)|var|bool
bop_18 = relay.less(bop_2.astype('bool'), relay.reshape(var_17.astype('bool'), relay.shape_of(bop_2))) # shape=(8,)
bop_25 = relay.not_equal(const_9.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(const_9))) # shape=(8,)
uop_28 = relay.log2(var_17.astype('float64')) # shape=(8,)
output = relay.Tuple([bop_10,bop_13,bop_18,bop_25,uop_28,])
output2 = relay.Tuple([bop_10,bop_13,bop_18,bop_25,uop_28,])
func_30 = relay.Function([var_0,var_1,var_17,], output)
mod['func_30'] = func_30
mod = relay.transform.InferType()(mod)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
func_30_call = mutated_mod.get_global_var('func_30')
var_32 = relay.var("var_32", dtype = "int16", shape = (8,))#candidate|32|(8,)|var|int16
var_33 = relay.var("var_33", dtype = "int16", shape = (8,))#candidate|33|(8,)|var|int16
var_34 = relay.var("var_34", dtype = "bool", shape = (8,))#candidate|34|(8,)|var|bool
call_31 = func_30_call(var_32,var_33,var_34,)
output = call_31
func_35 = relay.Function([var_32,var_33,var_34,], output)
mutated_mod['func_35'] = func_35
mutated_mod = relay.transform.InferType()(mutated_mod)
var_39 = relay.var("var_39", dtype = "float32", shape = (2, 4))#candidate|39|(2, 4)|var|float32
var_40 = relay.var("var_40", dtype = "float32", shape = (2, 4))#candidate|40|(2, 4)|var|float32
bop_41 = relay.floor_divide(var_39.astype('float32'), relay.reshape(var_40.astype('float32'), relay.shape_of(var_39))) # shape=(2, 4)
bop_44 = relay.power(var_39.astype('float32'), relay.reshape(bop_41.astype('float32'), relay.shape_of(var_39))) # shape=(2, 4)
output = relay.Tuple([bop_44,])
output2 = relay.Tuple([bop_44,])
func_47 = relay.Function([var_39,var_40,], output)
mod['func_47'] = func_47
mod = relay.transform.InferType()(mod)
mutated_mod['func_47'] = func_47
mutated_mod = relay.transform.InferType()(mutated_mod)
func_47_call = mutated_mod.get_global_var('func_47')
var_49 = relay.var("var_49", dtype = "float32", shape = (2, 4))#candidate|49|(2, 4)|var|float32
var_50 = relay.var("var_50", dtype = "float32", shape = (2, 4))#candidate|50|(2, 4)|var|float32
call_48 = func_47_call(var_49,var_50,)
output = call_48
func_51 = relay.Function([var_49,var_50,], output)
mutated_mod['func_51'] = func_51
mutated_mod = relay.transform.InferType()(mutated_mod)
var_53 = relay.var("var_53", dtype = "float64", shape = (5, 7))#candidate|53|(5, 7)|var|float64
uop_54 = relay.sin(var_53.astype('float64')) # shape=(5, 7)
bop_56 = relay.logical_or(var_53.astype('bool'), relay.reshape(uop_54.astype('bool'), relay.shape_of(var_53))) # shape=(5, 7)
output = relay.Tuple([bop_56,])
output2 = relay.Tuple([bop_56,])
func_59 = relay.Function([var_53,], output)
mod['func_59'] = func_59
mod = relay.transform.InferType()(mod)
var_60 = relay.var("var_60", dtype = "float64", shape = (5, 7))#candidate|60|(5, 7)|var|float64
output = func_59(var_60)
func_61 = relay.Function([var_60], output)
mutated_mod['func_61'] = func_61
mutated_mod = relay.transform.InferType()(mutated_mod)
var_72 = relay.var("var_72", dtype = "int8", shape = (9, 16))#candidate|72|(9, 16)|var|int8
var_73 = relay.var("var_73", dtype = "int8", shape = (9, 16))#candidate|73|(9, 16)|var|int8
bop_74 = relay.left_shift(var_72.astype('int8'), relay.reshape(var_73.astype('int8'), relay.shape_of(var_72))) # shape=(9, 16)
bop_77 = relay.floor_mod(var_72.astype('float64'), relay.reshape(bop_74.astype('float64'), relay.shape_of(var_72))) # shape=(9, 16)
bop_81 = relay.greater(var_72.astype('bool'), relay.reshape(bop_74.astype('bool'), relay.shape_of(var_72))) # shape=(9, 16)
func_59_call = mod.get_global_var('func_59')
func_61_call = mutated_mod.get_global_var('func_61')
var_85 = relay.var("var_85", dtype = "float64", shape = (35,))#candidate|85|(35,)|var|float64
call_84 = relay.TupleGetItem(func_59_call(relay.reshape(var_85.astype('float64'), [5, 7])), 0)
call_86 = relay.TupleGetItem(func_61_call(relay.reshape(var_85.astype('float64'), [5, 7])), 0)
output = relay.Tuple([bop_77,bop_81,call_84,var_85,])
output2 = relay.Tuple([bop_77,bop_81,call_86,var_85,])
func_87 = relay.Function([var_72,var_73,var_85,], output)
mod['func_87'] = func_87
mod = relay.transform.InferType()(mod)
mutated_mod['func_87'] = func_87
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mutated_mod.get_global_var('func_87')
var_89 = relay.var("var_89", dtype = "int8", shape = (9, 16))#candidate|89|(9, 16)|var|int8
var_90 = relay.var("var_90", dtype = "int8", shape = (9, 16))#candidate|90|(9, 16)|var|int8
var_91 = relay.var("var_91", dtype = "float64", shape = (35,))#candidate|91|(35,)|var|float64
call_88 = func_87_call(var_89,var_90,var_91,)
output = call_88
func_92 = relay.Function([var_89,var_90,var_91,], output)
mutated_mod['func_92'] = func_92
mutated_mod = relay.transform.InferType()(mutated_mod)
var_94 = relay.var("var_94", dtype = "float32", shape = (12,))#candidate|94|(12,)|var|float32
uop_95 = relay.sigmoid(var_94.astype('float32')) # shape=(12,)
output = relay.Tuple([uop_95,])
output2 = relay.Tuple([uop_95,])
func_98 = relay.Function([var_94,], output)
mod['func_98'] = func_98
mod = relay.transform.InferType()(mod)
mutated_mod['func_98'] = func_98
mutated_mod = relay.transform.InferType()(mutated_mod)
var_99 = relay.var("var_99", dtype = "float32", shape = (12,))#candidate|99|(12,)|var|float32
func_98_call = mutated_mod.get_global_var('func_98')
call_100 = func_98_call(var_99)
output = call_100
func_101 = relay.Function([var_99], output)
mutated_mod['func_101'] = func_101
mutated_mod = relay.transform.InferType()(mutated_mod)
const_110 = relay.const([-9,2,10,6,1,8,-7,1,-9,6,8,10,-5], dtype = "uint64")#candidate|110|(13,)|const|uint64
const_111 = relay.const([-7,9,-6,-9,5,7,-1,5,-4,-7,-7,8,-7], dtype = "uint64")#candidate|111|(13,)|const|uint64
bop_112 = relay.greater_equal(const_110.astype('bool'), relay.reshape(const_111.astype('bool'), relay.shape_of(const_110))) # shape=(13,)
bop_128 = relay.not_equal(bop_112.astype('bool'), relay.reshape(const_110.astype('bool'), relay.shape_of(bop_112))) # shape=(13,)
uop_131 = relay.sin(bop_112.astype('float64')) # shape=(13,)
var_133 = relay.var("var_133", dtype = "bool", shape = (13,))#candidate|133|(13,)|var|bool
bop_134 = relay.less(bop_128.astype('bool'), relay.reshape(var_133.astype('bool'), relay.shape_of(bop_128))) # shape=(13,)
bop_138 = relay.equal(uop_131.astype('bool'), relay.reshape(const_110.astype('bool'), relay.shape_of(uop_131))) # shape=(13,)
func_59_call = mod.get_global_var('func_59')
func_61_call = mutated_mod.get_global_var('func_61')
var_142 = relay.var("var_142", dtype = "float64", shape = (35,))#candidate|142|(35,)|var|float64
call_141 = relay.TupleGetItem(func_59_call(relay.reshape(var_142.astype('float64'), [5, 7])), 0)
call_143 = relay.TupleGetItem(func_61_call(relay.reshape(var_142.astype('float64'), [5, 7])), 0)
bop_144 = relay.less_equal(uop_131.astype('bool'), relay.reshape(bop_112.astype('bool'), relay.shape_of(uop_131))) # shape=(13,)
bop_147 = relay.add(bop_144.astype('int64'), relay.reshape(uop_131.astype('int64'), relay.shape_of(bop_144))) # shape=(13,)
bop_151 = relay.minimum(bop_134.astype('uint8'), relay.reshape(var_133.astype('uint8'), relay.shape_of(bop_134))) # shape=(13,)
bop_154 = relay.less_equal(bop_138.astype('bool'), relay.reshape(bop_128.astype('bool'), relay.shape_of(bop_138))) # shape=(13,)
var_157 = relay.var("var_157", dtype = "int64", shape = (13,))#candidate|157|(13,)|var|int64
bop_158 = relay.greater(bop_147.astype('bool'), relay.reshape(var_157.astype('bool'), relay.shape_of(bop_147))) # shape=(13,)
output = relay.Tuple([call_141,var_142,bop_151,bop_154,bop_158,])
output2 = relay.Tuple([call_143,var_142,bop_151,bop_154,bop_158,])
func_161 = relay.Function([var_133,var_142,var_157,], output)
mod['func_161'] = func_161
mod = relay.transform.InferType()(mod)
var_162 = relay.var("var_162", dtype = "bool", shape = (13,))#candidate|162|(13,)|var|bool
var_163 = relay.var("var_163", dtype = "float64", shape = (35,))#candidate|163|(35,)|var|float64
var_164 = relay.var("var_164", dtype = "int64", shape = (13,))#candidate|164|(13,)|var|int64
output = func_161(var_162,var_163,var_164,)
func_165 = relay.Function([var_162,var_163,var_164,], output)
mutated_mod['func_165'] = func_165
mutated_mod = relay.transform.InferType()(mutated_mod)
var_169 = relay.var("var_169", dtype = "int64", shape = (6, 5))#candidate|169|(6, 5)|var|int64
var_170 = relay.var("var_170", dtype = "int64", shape = (6, 5))#candidate|170|(6, 5)|var|int64
bop_171 = relay.left_shift(var_169.astype('int64'), relay.reshape(var_170.astype('int64'), relay.shape_of(var_169))) # shape=(6, 5)
uop_174 = relay.asin(var_170.astype('float64')) # shape=(6, 5)
var_176 = relay.var("var_176", dtype = "float64", shape = (6, 5))#candidate|176|(6, 5)|var|float64
bop_177 = relay.floor_divide(uop_174.astype('float64'), relay.reshape(var_176.astype('float64'), relay.shape_of(uop_174))) # shape=(6, 5)
const_180 = relay.const([[5.835545,6.041887,-4.166428,-1.724799,3.789908],[3.485238,-2.002199,4.653758,-1.700228,-8.315854],[5.674913,2.077578,-0.826073,2.121143,9.549028],[2.806389,-3.912028,-5.737468,8.873165,2.652857],[-5.303920,8.637950,7.753541,7.076562,2.547997],[-1.917454,-9.857187,6.403427,8.212200,7.347809]], dtype = "float64")#candidate|180|(6, 5)|const|float64
bop_181 = relay.less_equal(uop_174.astype('bool'), relay.reshape(const_180.astype('bool'), relay.shape_of(uop_174))) # shape=(6, 5)
uop_186 = relay.atan(bop_181.astype('float32')) # shape=(6, 5)
uop_188 = relay.tan(uop_186.astype('float32')) # shape=(6, 5)
bop_190 = relay.add(uop_188.astype('int64'), relay.reshape(bop_181.astype('int64'), relay.shape_of(uop_188))) # shape=(6, 5)
uop_193 = relay.sqrt(uop_188.astype('float32')) # shape=(6, 5)
uop_195 = relay.cosh(uop_174.astype('float32')) # shape=(6, 5)
bop_197 = relay.greater(uop_193.astype('bool'), relay.reshape(bop_190.astype('bool'), relay.shape_of(uop_193))) # shape=(6, 5)
bop_200 = relay.right_shift(bop_181.astype('int16'), relay.reshape(uop_188.astype('int16'), relay.shape_of(bop_181))) # shape=(6, 5)
bop_205 = relay.greater_equal(uop_195.astype('bool'), relay.reshape(bop_190.astype('bool'), relay.shape_of(uop_195))) # shape=(6, 5)
bop_208 = relay.less(var_169.astype('bool'), relay.reshape(var_176.astype('bool'), relay.shape_of(var_169))) # shape=(6, 5)
uop_211 = relay.acos(bop_197.astype('float64')) # shape=(6, 5)
uop_213 = relay.log(uop_193.astype('float32')) # shape=(6, 5)
var_215 = relay.var("var_215", dtype = "int16", shape = (6, 5))#candidate|215|(6, 5)|var|int16
bop_216 = relay.not_equal(bop_200.astype('bool'), relay.reshape(var_215.astype('bool'), relay.shape_of(bop_200))) # shape=(6, 5)
bop_220 = relay.less_equal(uop_213.astype('bool'), relay.reshape(bop_190.astype('bool'), relay.shape_of(uop_213))) # shape=(6, 5)
bop_224 = relay.subtract(uop_213.astype('uint64'), relay.reshape(uop_188.astype('uint64'), relay.shape_of(uop_213))) # shape=(6, 5)
bop_227 = relay.floor_mod(uop_193.astype('float32'), relay.reshape(bop_190.astype('float32'), relay.shape_of(uop_193))) # shape=(6, 5)
var_232 = relay.var("var_232", dtype = "float32", shape = (6, 5))#candidate|232|(6, 5)|var|float32
bop_233 = relay.bitwise_xor(uop_213.astype('uint16'), relay.reshape(var_232.astype('uint16'), relay.shape_of(uop_213))) # shape=(6, 5)
output = relay.Tuple([bop_171,bop_177,bop_205,bop_208,uop_211,bop_216,bop_220,bop_224,bop_227,bop_233,])
output2 = relay.Tuple([bop_171,bop_177,bop_205,bop_208,uop_211,bop_216,bop_220,bop_224,bop_227,bop_233,])
func_237 = relay.Function([var_169,var_170,var_176,var_215,var_232,], output)
mod['func_237'] = func_237
mod = relay.transform.InferType()(mod)
mutated_mod['func_237'] = func_237
mutated_mod = relay.transform.InferType()(mutated_mod)
func_237_call = mutated_mod.get_global_var('func_237')
var_239 = relay.var("var_239", dtype = "int64", shape = (6, 5))#candidate|239|(6, 5)|var|int64
var_240 = relay.var("var_240", dtype = "int64", shape = (6, 5))#candidate|240|(6, 5)|var|int64
var_241 = relay.var("var_241", dtype = "float64", shape = (6, 5))#candidate|241|(6, 5)|var|float64
var_242 = relay.var("var_242", dtype = "int16", shape = (6, 5))#candidate|242|(6, 5)|var|int16
var_243 = relay.var("var_243", dtype = "float32", shape = (6, 5))#candidate|243|(6, 5)|var|float32
call_238 = func_237_call(var_239,var_240,var_241,var_242,var_243,)
output = call_238
func_244 = relay.Function([var_239,var_240,var_241,var_242,var_243,], output)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
var_248 = relay.var("var_248", dtype = "float64", shape = (5,))#candidate|248|(5,)|var|float64
uop_249 = relay.atan(var_248.astype('float64')) # shape=(5,)
uop_251 = relay.erf(uop_249.astype('float64')) # shape=(5,)
func_87_call = mod.get_global_var('func_87')
func_92_call = mutated_mod.get_global_var('func_92')
const_255 = relay.const([[-8,4],[7,-4],[4,10],[9,10],[-4,7],[-10,9],[-1,2],[6,1],[-9,5],[-2,2],[2,3],[4,3],[8,5],[8,9],[-1,-3],[10,-1],[9,-4],[7,-7],[8,10],[-3,-1],[4,-10],[-10,-6],[-1,7],[10,-10],[-6,-2],[-10,3],[8,1],[8,-2],[-8,7],[-3,-8],[-1,-8],[7,-2],[2,-7],[2,4],[5,1],[9,-8],[-8,-6],[9,5],[-1,2],[1,-10],[-7,-7],[8,9],[9,7],[-5,-7],[-3,4],[-9,1],[-2,-9],[-7,7],[-5,-5],[8,-6],[-3,-6],[-9,1],[-6,-3],[4,4],[-9,8],[-1,1],[1,-10],[-9,-5],[1,-3],[4,4],[9,-4],[3,8],[2,-9],[-2,2],[-6,2],[3,10],[-9,-6],[-6,-3],[-10,-7],[-7,-7],[10,-6],[4,-6]], dtype = "int8")#candidate|255|(72, 2)|const|int8
const_256 = relay.const([-5.465976,9.258015,0.226855,4.452218,8.643205,-7.897759,-6.548272,0.024464,3.294241,2.437838,9.453965,4.582218,-8.733088,-0.538809,4.412843,-9.417639,-2.888724,8.245965,-2.364750,-9.230233,-5.165453,-4.060389,3.563363,4.881412,-7.989142,7.200074,-3.532450,9.633076,1.511621,4.698585,-0.935649,-0.449257,5.437448,-0.037848,-9.001259], dtype = "float64")#candidate|256|(35,)|const|float64
call_254 = relay.TupleGetItem(func_87_call(relay.reshape(const_255.astype('int8'), [9, 16]), relay.reshape(const_255.astype('int8'), [9, 16]), relay.reshape(const_256.astype('float64'), [35,]), ), 3)
call_257 = relay.TupleGetItem(func_92_call(relay.reshape(const_255.astype('int8'), [9, 16]), relay.reshape(const_255.astype('int8'), [9, 16]), relay.reshape(const_256.astype('float64'), [35,]), ), 3)
var_258 = relay.var("var_258", dtype = "float64", shape = (5,))#candidate|258|(5,)|var|float64
bop_259 = relay.equal(uop_251.astype('bool'), relay.reshape(var_258.astype('bool'), relay.shape_of(uop_251))) # shape=(5,)
uop_263 = relay.rsqrt(bop_259.astype('float32')) # shape=(5,)
bop_265 = relay.greater(var_258.astype('bool'), relay.reshape(uop_251.astype('bool'), relay.shape_of(var_258))) # shape=(5,)
bop_269 = relay.floor_divide(bop_259.astype('float32'), relay.reshape(uop_249.astype('float32'), relay.shape_of(bop_259))) # shape=(5,)
bop_272 = relay.multiply(uop_251.astype('uint8'), relay.reshape(bop_265.astype('uint8'), relay.shape_of(uop_251))) # shape=(5,)
func_161_call = mod.get_global_var('func_161')
func_165_call = mutated_mod.get_global_var('func_165')
const_276 = relay.const([[True],[True],[False],[False],[True],[False],[False],[False],[True],[True],[True],[False],[False]], dtype = "bool")#candidate|276|(13, 1)|const|bool
call_275 = relay.TupleGetItem(func_161_call(relay.reshape(const_276.astype('bool'), [13,]), relay.reshape(const_256.astype('float64'), [35,]), relay.reshape(const_276.astype('int64'), [13,]), ), 3)
call_277 = relay.TupleGetItem(func_165_call(relay.reshape(const_276.astype('bool'), [13,]), relay.reshape(const_256.astype('float64'), [35,]), relay.reshape(const_276.astype('int64'), [13,]), ), 3)
bop_278 = relay.equal(bop_269.astype('bool'), relay.reshape(bop_265.astype('bool'), relay.shape_of(bop_269))) # shape=(5,)
bop_281 = relay.multiply(bop_269.astype('float32'), relay.reshape(uop_251.astype('float32'), relay.shape_of(bop_269))) # shape=(5,)
bop_284 = relay.mod(bop_259.astype('float32'), relay.reshape(uop_251.astype('float32'), relay.shape_of(bop_259))) # shape=(5,)
uop_287 = relay.cosh(uop_263.astype('float32')) # shape=(5,)
output = relay.Tuple([call_254,const_255,const_256,bop_272,call_275,const_276,bop_278,bop_281,bop_284,uop_287,])
output2 = relay.Tuple([call_257,const_255,const_256,bop_272,call_277,const_276,bop_278,bop_281,bop_284,uop_287,])
func_289 = relay.Function([var_248,var_258,], output)
mod['func_289'] = func_289
mod = relay.transform.InferType()(mod)
var_290 = relay.var("var_290", dtype = "float64", shape = (5,))#candidate|290|(5,)|var|float64
var_291 = relay.var("var_291", dtype = "float64", shape = (5,))#candidate|291|(5,)|var|float64
output = func_289(var_290,var_291,)
func_292 = relay.Function([var_290,var_291,], output)
mutated_mod['func_292'] = func_292
mutated_mod = relay.transform.InferType()(mutated_mod)
var_300 = relay.var("var_300", dtype = "bool", shape = ())#candidate|300|()|var|bool
var_301 = relay.var("var_301", dtype = "bool", shape = (13,))#candidate|301|(13,)|var|bool
bop_302 = relay.logical_and(var_300.astype('bool'), var_301.astype('bool')) # shape=(13,)
func_237_call = mod.get_global_var('func_237')
func_244_call = mutated_mod.get_global_var('func_244')
var_306 = relay.var("var_306", dtype = "int64", shape = (1, 30))#candidate|306|(1, 30)|var|int64
call_305 = relay.TupleGetItem(func_237_call(relay.reshape(var_306.astype('int64'), [6, 5]), relay.reshape(var_306.astype('int64'), [6, 5]), relay.reshape(var_306.astype('float64'), [6, 5]), relay.reshape(var_306.astype('int16'), [6, 5]), relay.reshape(var_306.astype('float32'), [6, 5]), ), 7)
call_307 = relay.TupleGetItem(func_244_call(relay.reshape(var_306.astype('int64'), [6, 5]), relay.reshape(var_306.astype('int64'), [6, 5]), relay.reshape(var_306.astype('float64'), [6, 5]), relay.reshape(var_306.astype('int16'), [6, 5]), relay.reshape(var_306.astype('float32'), [6, 5]), ), 7)
bop_308 = relay.bitwise_and(var_306.astype('int8'), relay.reshape(call_305.astype('int8'), relay.shape_of(var_306))) # shape=(1, 30)
bop_311 = relay.bitwise_and(var_306.astype('int8'), relay.reshape(call_307.astype('int8'), relay.shape_of(var_306))) # shape=(1, 30)
var_313 = relay.var("var_313", dtype = "bool", shape = (13,))#candidate|313|(13,)|var|bool
bop_314 = relay.multiply(bop_302.astype('uint32'), relay.reshape(var_313.astype('uint32'), relay.shape_of(bop_302))) # shape=(13,)
uop_318 = relay.sinh(bop_314.astype('float64')) # shape=(13,)
uop_327 = relay.atan(bop_302.astype('float32')) # shape=(13,)
var_332 = relay.var("var_332", dtype = "float64", shape = (13,))#candidate|332|(13,)|var|float64
bop_333 = relay.bitwise_and(uop_318.astype('int32'), relay.reshape(var_332.astype('int32'), relay.shape_of(uop_318))) # shape=(13,)
bop_336 = relay.floor_divide(bop_333.astype('float32'), relay.reshape(uop_318.astype('float32'), relay.shape_of(bop_333))) # shape=(13,)
uop_341 = relay.sigmoid(bop_302.astype('float64')) # shape=(13,)
output = relay.Tuple([bop_308,uop_327,bop_336,uop_341,])
output2 = relay.Tuple([bop_311,uop_327,bop_336,uop_341,])
func_343 = relay.Function([var_300,var_301,var_306,var_313,var_332,], output)
mod['func_343'] = func_343
mod = relay.transform.InferType()(mod)
var_344 = relay.var("var_344", dtype = "bool", shape = ())#candidate|344|()|var|bool
var_345 = relay.var("var_345", dtype = "bool", shape = (13,))#candidate|345|(13,)|var|bool
var_346 = relay.var("var_346", dtype = "int64", shape = (1, 30))#candidate|346|(1, 30)|var|int64
var_347 = relay.var("var_347", dtype = "bool", shape = (13,))#candidate|347|(13,)|var|bool
var_348 = relay.var("var_348", dtype = "float64", shape = (13,))#candidate|348|(13,)|var|float64
output = func_343(var_344,var_345,var_346,var_347,var_348,)
func_349 = relay.Function([var_344,var_345,var_346,var_347,var_348,], output)
mutated_mod['func_349'] = func_349
mutated_mod = relay.transform.InferType()(mutated_mod)
var_369 = relay.var("var_369", dtype = "uint16", shape = ())#candidate|369|()|var|uint16
var_370 = relay.var("var_370", dtype = "uint16", shape = (4, 14))#candidate|370|(4, 14)|var|uint16
bop_371 = relay.less_equal(var_369.astype('bool'), var_370.astype('bool')) # shape=(4, 14)
output = bop_371
output2 = bop_371
func_374 = relay.Function([var_369,var_370,], output)
mod['func_374'] = func_374
mod = relay.transform.InferType()(mod)
var_375 = relay.var("var_375", dtype = "uint16", shape = ())#candidate|375|()|var|uint16
var_376 = relay.var("var_376", dtype = "uint16", shape = (4, 14))#candidate|376|(4, 14)|var|uint16
output = func_374(var_375,var_376,)
func_377 = relay.Function([var_375,var_376,], output)
mutated_mod['func_377'] = func_377
mutated_mod = relay.transform.InferType()(mutated_mod)
var_442 = relay.var("var_442", dtype = "uint32", shape = (1, 10, 15))#candidate|442|(1, 10, 15)|var|uint32
var_443 = relay.var("var_443", dtype = "uint32", shape = (1, 10, 15))#candidate|443|(1, 10, 15)|var|uint32
bop_444 = relay.bitwise_and(var_442.astype('uint32'), relay.reshape(var_443.astype('uint32'), relay.shape_of(var_442))) # shape=(1, 10, 15)
uop_448 = relay.asinh(var_443.astype('float32')) # shape=(1, 10, 15)
bop_450 = relay.right_shift(uop_448.astype('uint64'), relay.reshape(bop_444.astype('uint64'), relay.shape_of(uop_448))) # shape=(1, 10, 15)
bop_453 = relay.subtract(bop_450.astype('int32'), relay.reshape(var_443.astype('int32'), relay.shape_of(bop_450))) # shape=(1, 10, 15)
output = bop_453
output2 = bop_453
func_458 = relay.Function([var_442,var_443,], output)
mod['func_458'] = func_458
mod = relay.transform.InferType()(mod)
var_459 = relay.var("var_459", dtype = "uint32", shape = (1, 10, 15))#candidate|459|(1, 10, 15)|var|uint32
var_460 = relay.var("var_460", dtype = "uint32", shape = (1, 10, 15))#candidate|460|(1, 10, 15)|var|uint32
output = func_458(var_459,var_460,)
func_461 = relay.Function([var_459,var_460,], output)
mutated_mod['func_461'] = func_461
mutated_mod = relay.transform.InferType()(mutated_mod)
const_466 = relay.const([9.765593,8.422642,0.412319,-2.348176,0.491249], dtype = "float32")#candidate|466|(5,)|const|float32
uop_467 = relay.sinh(const_466.astype('float32')) # shape=(5,)
uop_470 = relay.acosh(const_466.astype('float64')) # shape=(5,)
output = relay.Tuple([uop_467,uop_470,])
output2 = relay.Tuple([uop_467,uop_470,])
func_473 = relay.Function([], output)
mod['func_473'] = func_473
mod = relay.transform.InferType()(mod)
mutated_mod['func_473'] = func_473
mutated_mod = relay.transform.InferType()(mutated_mod)
func_473_call = mutated_mod.get_global_var('func_473')
call_474 = func_473_call()
output = call_474
func_475 = relay.Function([], output)
mutated_mod['func_475'] = func_475
mutated_mod = relay.transform.InferType()(mutated_mod)
func_473_call = mod.get_global_var('func_473')
func_475_call = mutated_mod.get_global_var('func_475')
call_491 = relay.TupleGetItem(func_473_call(), 1)
call_492 = relay.TupleGetItem(func_475_call(), 1)
var_494 = relay.var("var_494", dtype = "float64", shape = (5,))#candidate|494|(5,)|var|float64
bop_495 = relay.logical_xor(call_491.astype('int32'), relay.reshape(var_494.astype('int32'), relay.shape_of(call_491))) # shape=(5,)
bop_498 = relay.logical_xor(call_492.astype('int32'), relay.reshape(var_494.astype('int32'), relay.shape_of(call_492))) # shape=(5,)
func_343_call = mod.get_global_var('func_343')
func_349_call = mutated_mod.get_global_var('func_349')
var_500 = relay.var("var_500", dtype = "bool", shape = ())#candidate|500|()|var|bool
var_501 = relay.var("var_501", dtype = "bool", shape = (13,))#candidate|501|(13,)|var|bool
var_502 = relay.var("var_502", dtype = "int64", shape = (30,))#candidate|502|(30,)|var|int64
call_499 = relay.TupleGetItem(func_343_call(relay.reshape(var_500.astype('bool'), []), relay.reshape(var_501.astype('bool'), [13,]), relay.reshape(var_502.astype('int64'), [1, 30]), relay.reshape(var_501.astype('bool'), [13,]), relay.reshape(var_501.astype('float64'), [13,]), ), 3)
call_503 = relay.TupleGetItem(func_349_call(relay.reshape(var_500.astype('bool'), []), relay.reshape(var_501.astype('bool'), [13,]), relay.reshape(var_502.astype('int64'), [1, 30]), relay.reshape(var_501.astype('bool'), [13,]), relay.reshape(var_501.astype('float64'), [13,]), ), 3)
uop_506 = relay.sqrt(call_499.astype('float64')) # shape=(13,)
uop_508 = relay.sqrt(call_503.astype('float64')) # shape=(13,)
const_516 = relay.const([6.169563,-3.350984,-0.050699,2.669417,0.628174,1.814715,-9.425425,2.776736,-8.447645,7.396072,-3.152053,2.195016,-9.633347], dtype = "float64")#candidate|516|(13,)|const|float64
bop_517 = relay.right_shift(uop_506.astype('int8'), relay.reshape(const_516.astype('int8'), relay.shape_of(uop_506))) # shape=(13,)
bop_520 = relay.right_shift(uop_508.astype('int8'), relay.reshape(const_516.astype('int8'), relay.shape_of(uop_508))) # shape=(13,)
output = relay.Tuple([bop_495,var_500,var_501,var_502,bop_517,])
output2 = relay.Tuple([bop_498,var_500,var_501,var_502,bop_520,])
func_523 = relay.Function([var_494,var_500,var_501,var_502,], output)
mod['func_523'] = func_523
mod = relay.transform.InferType()(mod)
mutated_mod['func_523'] = func_523
mutated_mod = relay.transform.InferType()(mutated_mod)
func_523_call = mutated_mod.get_global_var('func_523')
var_525 = relay.var("var_525", dtype = "float64", shape = (5,))#candidate|525|(5,)|var|float64
var_526 = relay.var("var_526", dtype = "bool", shape = ())#candidate|526|()|var|bool
var_527 = relay.var("var_527", dtype = "bool", shape = (13,))#candidate|527|(13,)|var|bool
var_528 = relay.var("var_528", dtype = "int64", shape = (30,))#candidate|528|(30,)|var|int64
call_524 = func_523_call(var_525,var_526,var_527,var_528,)
output = call_524
func_529 = relay.Function([var_525,var_526,var_527,var_528,], output)
mutated_mod['func_529'] = func_529
mutated_mod = relay.transform.InferType()(mutated_mod)
var_566 = relay.var("var_566", dtype = "float64", shape = (11, 2))#candidate|566|(11, 2)|var|float64
uop_567 = relay.log10(var_566.astype('float64')) # shape=(11, 2)
bop_570 = relay.logical_and(uop_567.astype('bool'), relay.reshape(var_566.astype('bool'), relay.shape_of(uop_567))) # shape=(11, 2)
bop_575 = relay.divide(bop_570.astype('float64'), relay.reshape(var_566.astype('float64'), relay.shape_of(bop_570))) # shape=(11, 2)
uop_578 = relay.exp(uop_567.astype('float64')) # shape=(11, 2)
bop_580 = relay.power(bop_575.astype('float64'), relay.reshape(uop_578.astype('float64'), relay.shape_of(bop_575))) # shape=(11, 2)
func_237_call = mod.get_global_var('func_237')
func_244_call = mutated_mod.get_global_var('func_244')
var_584 = relay.var("var_584", dtype = "int64", shape = (10, 3))#candidate|584|(10, 3)|var|int64
call_583 = relay.TupleGetItem(func_237_call(relay.reshape(var_584.astype('int64'), [6, 5]), relay.reshape(var_584.astype('int64'), [6, 5]), relay.reshape(var_584.astype('float64'), [6, 5]), relay.reshape(var_584.astype('int16'), [6, 5]), relay.reshape(var_584.astype('float32'), [6, 5]), ), 7)
call_585 = relay.TupleGetItem(func_244_call(relay.reshape(var_584.astype('int64'), [6, 5]), relay.reshape(var_584.astype('int64'), [6, 5]), relay.reshape(var_584.astype('float64'), [6, 5]), relay.reshape(var_584.astype('int16'), [6, 5]), relay.reshape(var_584.astype('float32'), [6, 5]), ), 7)
const_587 = relay.const([[1.060349,-4.970738],[-8.131029,-3.832206],[2.987763,-9.187985],[2.902958,-9.796502],[6.527366,7.236768],[0.838275,-7.799144],[1.990592,2.042479],[1.329179,-7.174312],[-9.105403,7.850303],[1.914266,4.283291],[0.010978,-8.549661]], dtype = "float64")#candidate|587|(11, 2)|const|float64
bop_588 = relay.add(bop_580.astype('uint16'), relay.reshape(const_587.astype('uint16'), relay.shape_of(bop_580))) # shape=(11, 2)
const_591 = relay.const([[-4.625558,4.080753],[-7.412990,6.406867],[3.579418,4.956924],[-2.208922,5.084237],[-2.343392,-9.014080],[-8.718591,-3.491931],[-2.480727,-4.394641],[8.539695,7.879233],[-2.577674,-6.630324],[-6.253866,-5.339461],[-5.408189,-8.594049]], dtype = "float64")#candidate|591|(11, 2)|const|float64
bop_592 = relay.subtract(uop_578.astype('uint32'), relay.reshape(const_591.astype('uint32'), relay.shape_of(uop_578))) # shape=(11, 2)
func_458_call = mod.get_global_var('func_458')
func_461_call = mutated_mod.get_global_var('func_461')
var_599 = relay.var("var_599", dtype = "uint32", shape = (150, 1))#candidate|599|(150, 1)|var|uint32
call_598 = func_458_call(relay.reshape(var_599.astype('uint32'), [1, 10, 15]), relay.reshape(var_599.astype('uint32'), [1, 10, 15]), )
call_600 = func_458_call(relay.reshape(var_599.astype('uint32'), [1, 10, 15]), relay.reshape(var_599.astype('uint32'), [1, 10, 15]), )
uop_601 = relay.asin(uop_567.astype('float64')) # shape=(11, 2)
func_289_call = mod.get_global_var('func_289')
func_292_call = mutated_mod.get_global_var('func_292')
const_604 = relay.const([7.336697,5.360296,1.064284,3.467528,5.351975], dtype = "float64")#candidate|604|(5,)|const|float64
call_603 = relay.TupleGetItem(func_289_call(relay.reshape(const_604.astype('float64'), [5,]), relay.reshape(const_604.astype('float64'), [5,]), ), 5)
call_605 = relay.TupleGetItem(func_292_call(relay.reshape(const_604.astype('float64'), [5,]), relay.reshape(const_604.astype('float64'), [5,]), ), 5)
uop_606 = relay.asinh(bop_592.astype('float32')) # shape=(11, 2)
output = relay.Tuple([call_583,var_584,bop_588,call_598,var_599,uop_601,call_603,const_604,uop_606,])
output2 = relay.Tuple([call_585,var_584,bop_588,call_600,var_599,uop_601,call_605,const_604,uop_606,])
func_608 = relay.Function([var_566,var_584,var_599,], output)
mod['func_608'] = func_608
mod = relay.transform.InferType()(mod)
var_609 = relay.var("var_609", dtype = "float64", shape = (11, 2))#candidate|609|(11, 2)|var|float64
var_610 = relay.var("var_610", dtype = "int64", shape = (10, 3))#candidate|610|(10, 3)|var|int64
var_611 = relay.var("var_611", dtype = "uint32", shape = (150, 1))#candidate|611|(150, 1)|var|uint32
output = func_608(var_609,var_610,var_611,)
func_612 = relay.Function([var_609,var_610,var_611,], output)
mutated_mod['func_612'] = func_612
mutated_mod = relay.transform.InferType()(mutated_mod)
var_623 = relay.var("var_623", dtype = "float32", shape = (10,))#candidate|623|(10,)|var|float32
var_624 = relay.var("var_624", dtype = "float32", shape = (10,))#candidate|624|(10,)|var|float32
bop_625 = relay.less(var_623.astype('bool'), relay.reshape(var_624.astype('bool'), relay.shape_of(var_623))) # shape=(10,)
bop_630 = relay.subtract(var_623.astype('uint8'), relay.reshape(bop_625.astype('uint8'), relay.shape_of(var_623))) # shape=(10,)
uop_635 = relay.cosh(bop_625.astype('float32')) # shape=(10,)
bop_640 = relay.greater(var_623.astype('bool'), relay.reshape(bop_625.astype('bool'), relay.shape_of(var_623))) # shape=(10,)
output = relay.Tuple([bop_630,uop_635,bop_640,])
output2 = relay.Tuple([bop_630,uop_635,bop_640,])
func_647 = relay.Function([var_623,var_624,], output)
mod['func_647'] = func_647
mod = relay.transform.InferType()(mod)
var_648 = relay.var("var_648", dtype = "float32", shape = (10,))#candidate|648|(10,)|var|float32
var_649 = relay.var("var_649", dtype = "float32", shape = (10,))#candidate|649|(10,)|var|float32
output = func_647(var_648,var_649,)
func_650 = relay.Function([var_648,var_649,], output)
mutated_mod['func_650'] = func_650
mutated_mod = relay.transform.InferType()(mutated_mod)
func_473_call = mod.get_global_var('func_473')
func_475_call = mutated_mod.get_global_var('func_475')
call_658 = relay.TupleGetItem(func_473_call(), 1)
call_659 = relay.TupleGetItem(func_475_call(), 1)
func_87_call = mod.get_global_var('func_87')
func_92_call = mutated_mod.get_global_var('func_92')
var_663 = relay.var("var_663", dtype = "int8", shape = (8, 18))#candidate|663|(8, 18)|var|int8
const_664 = relay.const([-5.950103,5.492454,-3.227657,-1.484688,2.219456,-9.144068,-8.167368,-4.416928,8.388353,1.153691,8.948345,9.229209,2.132612,3.265509,-9.549684,6.178198,-6.638942,-6.298278,7.193831,-1.054922,1.746991,-4.719292,-6.542421,3.811325,-1.852414,-9.062036,-0.935169,-2.913793,2.569321,-1.546634,-4.661585,1.730080,-4.871880,1.456478,-2.473697], dtype = "float64")#candidate|664|(35,)|const|float64
call_662 = relay.TupleGetItem(func_87_call(relay.reshape(var_663.astype('int8'), [9, 16]), relay.reshape(var_663.astype('int8'), [9, 16]), relay.reshape(const_664.astype('float64'), [35,]), ), 0)
call_665 = relay.TupleGetItem(func_92_call(relay.reshape(var_663.astype('int8'), [9, 16]), relay.reshape(var_663.astype('int8'), [9, 16]), relay.reshape(const_664.astype('float64'), [35,]), ), 0)
output = relay.Tuple([call_658,call_662,var_663,const_664,])
output2 = relay.Tuple([call_659,call_665,var_663,const_664,])
func_666 = relay.Function([var_663,], output)
mod['func_666'] = func_666
mod = relay.transform.InferType()(mod)
mutated_mod['func_666'] = func_666
mutated_mod = relay.transform.InferType()(mutated_mod)
var_667 = relay.var("var_667", dtype = "int8", shape = (8, 18))#candidate|667|(8, 18)|var|int8
func_666_call = mutated_mod.get_global_var('func_666')
call_668 = func_666_call(var_667)
output = call_668
func_669 = relay.Function([var_667], output)
mutated_mod['func_669'] = func_669
mutated_mod = relay.transform.InferType()(mutated_mod)
var_671 = relay.var("var_671", dtype = "float64", shape = (13, 5, 16))#candidate|671|(13, 5, 16)|var|float64
uop_672 = relay.sqrt(var_671.astype('float64')) # shape=(13, 5, 16)
func_374_call = mod.get_global_var('func_374')
func_377_call = mutated_mod.get_global_var('func_377')
var_676 = relay.var("var_676", dtype = "uint16", shape = ())#candidate|676|()|var|uint16
const_677 = relay.const([1,-10,5,8,-7,10,1,4,4,-1,-1,4,1,-10,-8,-6,8,1,7,7,-4,-8,2,7,7,2,6,-3,-10,7,-7,9,2,-1,9,3,-4,8,-2,-3,4,6,3,-6,-4,1,9,-9,9,-10,-9,-4,3,-10,-9,-5], dtype = "uint16")#candidate|677|(56,)|const|uint16
call_675 = func_374_call(relay.reshape(var_676.astype('uint16'), []), relay.reshape(const_677.astype('uint16'), [4, 14]), )
call_678 = func_374_call(relay.reshape(var_676.astype('uint16'), []), relay.reshape(const_677.astype('uint16'), [4, 14]), )
uop_681 = relay.cosh(uop_672.astype('float32')) # shape=(13, 5, 16)
bop_683 = relay.add(uop_681.astype('float32'), relay.reshape(var_671.astype('float32'), relay.shape_of(uop_681))) # shape=(13, 5, 16)
var_686 = relay.var("var_686", dtype = "float32", shape = (13, 5, 16))#candidate|686|(13, 5, 16)|var|float32
bop_687 = relay.maximum(uop_681.astype('float32'), relay.reshape(var_686.astype('float32'), relay.shape_of(uop_681))) # shape=(13, 5, 16)
bop_692 = relay.bitwise_or(bop_683.astype('uint16'), relay.reshape(uop_672.astype('uint16'), relay.shape_of(bop_683))) # shape=(13, 5, 16)
var_696 = relay.var("var_696", dtype = "float64", shape = (13, 5, 16))#candidate|696|(13, 5, 16)|var|float64
bop_697 = relay.logical_xor(uop_672.astype('uint32'), relay.reshape(var_696.astype('uint32'), relay.shape_of(uop_672))) # shape=(13, 5, 16)
var_700 = relay.var("var_700", dtype = "uint16", shape = (13, 5, 16))#candidate|700|(13, 5, 16)|var|uint16
bop_701 = relay.logical_or(bop_692.astype('bool'), relay.reshape(var_700.astype('bool'), relay.shape_of(bop_692))) # shape=(13, 5, 16)
uop_704 = relay.cos(bop_687.astype('float32')) # shape=(13, 5, 16)
func_98_call = mod.get_global_var('func_98')
func_101_call = mutated_mod.get_global_var('func_101')
var_707 = relay.var("var_707", dtype = "float32", shape = (12,))#candidate|707|(12,)|var|float32
call_706 = relay.TupleGetItem(func_98_call(relay.reshape(var_707.astype('float32'), [12,])), 0)
call_708 = relay.TupleGetItem(func_101_call(relay.reshape(var_707.astype('float32'), [12,])), 0)
const_709 = relay.const([[[6.011235,2.625468,-2.262802,-4.906834,-2.442269,3.110779,0.918139,-9.854823,2.548494,0.974269,-8.108577,-9.360568,3.654714,4.653976,7.248831,3.394564],[7.914154,-6.125458,-1.714778,-3.585207,8.901084,-9.971443,-2.863517,-4.281146,-2.816554,7.936264,-1.244337,0.026003,9.408353,-6.189446,8.456308,6.900275],[9.630245,7.016710,-1.404085,-3.577541,0.813746,0.799088,-2.613236,-6.175265,9.229232,9.800777,3.265383,3.915791,-8.144685,-9.173214,7.640944,-6.389898],[4.977071,-8.430897,3.754380,8.691111,-1.030146,-3.712602,9.921069,4.536828,-9.830446,-4.823795,0.578777,-1.468720,8.185514,-3.250655,4.740970,0.577067],[-4.946288,-5.920709,5.956817,-4.307164,-0.312382,-7.577108,3.148158,-1.470450,-4.451496,-2.979249,-8.393565,0.716697,7.831682,0.241148,-4.744851,2.899731]],[[-6.544242,4.015509,-0.227908,0.301271,-6.701079,-1.075427,-7.424001,-2.825794,0.472126,-6.645502,6.892981,9.092804,4.148027,1.492405,-5.644426,-0.996019],[1.204948,-8.568790,-8.806489,-9.785866,7.111850,-9.122510,-9.889606,-7.508447,0.624146,3.994303,1.280059,6.265636,1.093049,-6.862331,-4.640535,3.480611],[-1.029935,4.794517,3.636991,-4.698116,-1.307814,-3.980584,0.202460,-8.226359,-3.416899,5.151047,-1.508410,-8.402212,-8.731659,4.541239,-5.281001,-1.107766],[-8.038954,8.347522,4.324148,-6.278367,-4.316859,-3.639192,-8.748759,8.167812,8.782110,-7.538757,0.442969,5.041309,1.692806,-7.157925,8.248056,6.252001],[-9.174471,-0.273012,-7.653579,7.851091,3.919110,-2.185454,-1.117741,-0.089674,-4.334716,-3.797372,-6.816795,3.653705,-5.328054,5.973530,-2.871008,-9.177691]],[[4.732775,4.631425,1.876020,-8.360594,-2.334118,-3.782764,-8.680815,-0.249649,4.370996,3.461549,4.818967,-8.897707,-9.896802,0.897823,-9.854369,-1.271426],[-9.100555,3.553433,-7.039075,5.014323,-0.455435,-7.936997,-9.584160,9.909398,6.084929,6.048743,-2.733634,-6.814524,0.984360,3.649044,6.271973,7.970540],[4.443836,-9.172589,1.733624,8.931687,8.099242,-0.196818,-7.354126,-8.262478,-4.247612,-0.923848,6.793306,-3.387382,5.722731,-0.514163,4.569581,-1.974942],[-8.013370,-1.144050,-4.446685,8.018260,9.558216,-3.319159,8.486857,7.449483,-9.558896,-7.262515,9.659002,7.735917,8.668604,8.700684,4.507979,4.194544],[-6.883087,2.342083,0.557631,7.771563,9.397127,8.993758,-8.403240,-5.481878,-3.965311,-5.643441,5.472339,-5.334484,0.087241,2.407020,-7.672742,-6.376435]],[[4.878277,6.176463,-8.317131,2.873837,-2.198703,-1.336863,-2.820702,-5.711270,-3.111041,-3.820626,-7.441856,-8.925735,-5.883973,-6.401748,-6.712165,-6.284128],[2.382692,6.836533,-1.428202,9.073032,3.434940,4.113643,-9.598401,-6.014435,-3.952292,3.113007,-5.738271,-7.472692,4.560462,9.868338,3.112180,6.229871],[0.450202,-5.981571,-2.204139,7.825165,8.393382,0.049362,-1.098747,-5.267258,-8.375761,2.803715,9.931301,3.063534,-1.503843,4.502350,-7.183486,-4.467602],[0.846269,6.223348,-7.932663,8.322260,1.541736,-8.437676,-7.343926,-9.877817,-0.349124,2.052971,9.748036,-6.211412,7.660201,2.680685,6.856343,-4.394840],[-0.822429,-9.770676,-0.218255,3.212378,-3.723543,3.655849,-5.513993,2.458000,9.357382,4.594449,9.686604,-8.424562,2.459908,6.813085,-7.572973,3.366653]],[[9.574519,5.213581,9.983657,-8.710019,-4.505359,-7.565202,2.355153,9.855209,6.741028,-7.163756,6.296069,2.905935,-0.472967,7.128882,-5.602080,-9.980572],[-9.181006,5.066305,-0.561539,7.949414,5.663139,9.119235,4.398877,5.866754,9.360467,0.363942,5.196702,7.317880,3.831801,-9.966842,3.437680,-2.438627],[-0.995604,6.997621,7.791776,-5.562868,4.514093,0.971297,-9.909269,8.354390,-2.926038,-6.425774,3.865417,-5.378228,-2.201796,8.724022,4.686150,1.603815],[9.941737,-3.636423,5.552826,-5.044781,-0.621132,6.461562,-6.413904,7.195587,-1.375046,-5.292610,2.283928,-6.697983,2.161930,-7.614479,6.926910,6.972588],[-4.013313,5.800176,-9.658010,-9.362044,-5.282582,4.179122,-7.874788,8.879359,5.782646,8.028640,-6.774087,-9.241497,0.625738,-7.862270,6.797633,2.662222]],[[4.081916,6.290302,-6.377592,5.458511,-6.284227,-6.593022,0.968753,8.122217,5.966988,4.997304,-9.186445,-1.016965,-9.763869,5.546966,-6.874933,-5.020415],[4.735564,9.001810,2.095468,-6.895986,9.226736,0.361978,0.304273,8.435396,6.251086,-9.740166,6.401797,6.980955,-4.540683,0.422714,4.907617,-6.517526],[7.597635,0.530311,-4.311730,-8.566453,-0.584741,-8.301952,-0.547160,-1.312430,4.308823,-5.324094,2.038981,-3.107473,-6.088115,-2.374247,4.564996,-6.972271],[-7.105722,-0.137490,-6.132591,9.174068,-6.017980,-1.132068,2.498061,-7.037253,0.812357,9.636331,-4.584657,-8.382302,0.747436,-4.707426,-2.853642,-9.516844],[-3.264233,3.879212,9.018061,6.158268,0.107227,5.501987,4.372601,0.862672,-2.128490,2.745603,-8.008999,1.327499,9.972586,2.036028,-8.857503,-0.357852]],[[-5.468663,1.269065,-1.150763,4.098715,-0.503608,9.845510,-7.393491,8.361294,8.738985,0.170344,-0.376180,-9.581678,-6.382525,1.559252,1.215759,-5.429492],[-9.767842,5.371555,-4.138133,-5.010419,0.047598,-2.462154,-7.107670,-8.256483,7.934542,7.251140,-4.146340,9.917732,7.481228,-1.790297,-5.672972,-5.130239],[-3.562491,-8.546736,-4.805219,-0.584671,-0.682988,1.674553,-0.708215,-0.992838,1.613517,0.141026,3.799062,5.530663,-9.023866,6.571660,-1.057346,4.061959],[-4.239247,7.224406,-9.990859,-6.373558,0.067342,5.926453,-0.510961,9.407992,4.489361,2.152562,2.918119,-3.380933,-4.799005,-2.240499,2.426000,-5.435960],[6.201154,2.607120,2.123211,-3.052130,5.798885,-2.909899,-3.355660,-5.960652,-5.409880,-9.996976,-0.182539,7.325314,3.467901,9.427896,0.906941,-5.852158]],[[3.680434,-3.582614,9.504108,8.077497,-2.086203,-1.610600,1.610232,8.522092,5.397759,1.024188,-6.154072,-0.632373,9.355105,-8.833220,2.525650,8.996202],[-3.584833,7.511944,5.051756,2.486877,-3.632588,-0.131233,-0.178517,-2.426774,2.479286,1.754211,-4.854601,8.470098,4.863810,8.202074,-9.396156,5.847994],[-7.337053,9.542522,-9.219056,-9.078575,-9.165819,-2.917511,-7.774585,-1.642090,7.092381,4.658413,-4.210622,0.961142,-9.814426,-2.187908,9.710150,-2.186365],[4.312218,1.070734,-9.590360,-3.196202,-3.887703,3.998072,0.832109,7.850601,3.015919,1.673721,7.857820,4.134296,1.773054,0.328016,0.790593,-8.488944],[2.264715,-3.476307,2.367837,-8.725518,-1.525462,-7.199289,-9.653079,-1.516657,-5.421070,3.466670,-6.323785,-5.509607,-1.628757,-2.128137,2.581696,-7.233009]],[[2.877869,-0.819708,-8.684556,-5.210082,-8.107753,-4.279927,-8.998945,-2.320631,-6.535045,6.190858,-6.848700,-2.044995,6.754447,-6.162602,-0.669538,6.367334],[-2.653816,-3.336924,-8.682105,0.415606,-2.633819,2.751398,-3.874640,8.192331,0.357599,3.408081,-6.793972,4.562054,-0.748826,-8.741902,-4.400571,-0.877061],[7.917564,9.713177,-7.091891,-7.227926,8.373330,8.990637,4.981499,-5.833748,0.445806,-6.052584,-2.815375,4.849123,-4.885106,9.286519,7.186200,3.228612],[-4.395523,0.316042,0.097031,3.609050,7.707234,-3.523783,-1.609590,-2.356137,-1.613659,6.418043,-8.579133,1.168674,3.398732,7.970857,6.311514,9.330043],[-6.370889,-2.068226,1.002801,-9.454537,-4.513798,9.726992,0.379244,2.884276,9.027815,-8.185525,2.592148,-8.501887,-5.127283,-5.734144,-6.478153,-0.199331]],[[-0.285028,-2.293311,-3.825680,-4.482504,3.785886,9.484079,0.600385,-4.577444,-9.896754,-7.369812,-3.095317,-3.655795,4.306349,-6.618132,-5.908539,-3.645333],[0.360931,-3.693777,8.810869,1.591140,8.220006,9.273939,7.498625,4.717394,4.138593,6.128801,-1.578661,-1.759180,-2.316046,2.862623,2.830488,2.362751],[-8.256999,1.048014,1.747502,-6.928792,2.533865,4.623568,7.891845,-6.176836,4.982210,-1.488322,1.615137,3.250506,-0.046307,5.923424,-2.142286,-5.144051],[-6.282455,-6.259751,-5.066643,-0.238085,-4.173867,-7.089017,-6.824248,-7.935348,7.275314,9.442245,-3.777717,-8.699998,2.381602,-9.553754,8.930597,-6.202146],[-8.634308,5.729155,9.010028,2.252048,2.572005,2.657748,-7.226451,9.703798,9.519769,4.288771,5.020982,-7.706744,3.381907,-1.255822,8.603040,-7.314631]],[[6.143133,0.395520,-7.926024,7.857595,-7.036435,4.620582,-4.688574,-8.092310,-9.640072,2.098498,-4.254286,-5.689900,9.421441,-0.365543,-6.846417,-3.759771],[-3.311278,-8.887545,-3.472558,-0.674016,5.578206,-4.394608,3.500171,2.971396,-4.517847,9.668781,-7.515337,-7.312386,3.911782,-9.520681,6.927655,5.609277],[-2.984219,-6.664325,7.408983,-8.051902,5.137039,8.738269,7.838255,-5.705611,5.324475,-4.285873,2.547607,6.965969,6.040200,-4.800934,9.220630,-0.642403],[-7.119093,-5.011563,-0.792009,-4.653907,5.040535,-5.280469,-1.616247,-5.593690,-2.076604,7.791129,9.604949,-2.537317,-3.098955,-3.517269,6.467850,-3.822593],[6.786448,9.830491,6.660293,6.749856,6.265780,5.075768,-9.242989,-4.310404,1.681038,2.369939,-3.156402,0.591544,-8.467911,-0.175997,7.812169,2.121263]],[[-9.860652,-3.978079,-2.867480,-5.856813,6.137960,4.939956,-9.899462,-3.401909,9.746245,-4.282260,-6.860877,9.223503,0.074119,-9.575192,-5.686367,-0.198683],[-7.248496,-8.298461,-3.911701,4.915059,-0.610171,-6.383621,-1.226235,-5.456250,-3.822790,3.783222,4.384480,1.609798,7.452145,5.062821,-0.613983,-5.050436],[-7.556164,2.588462,3.331634,-7.747857,-7.979362,-5.893961,4.829442,-9.826320,-8.238673,-8.176300,-4.558716,-5.583043,3.981150,3.805952,3.012787,3.552955],[-3.655041,4.893079,5.120447,-8.100709,-6.051943,-1.867947,1.465405,1.486364,-6.300613,5.836291,9.981400,-9.738499,1.667203,-4.338205,5.372670,-1.377503],[5.753957,-1.215571,0.170786,-6.609846,2.211892,-7.891043,0.375526,-7.308677,7.315618,8.747480,-1.669730,-0.153952,3.369409,-1.173881,-3.285786,0.318484]],[[-4.517250,-2.667216,-5.401174,9.973812,4.982514,-8.481001,0.931995,-4.943729,3.907319,-6.198143,-4.025006,7.061406,8.269521,-1.927364,6.532776,2.721841],[3.050911,4.406546,-2.958243,-2.992150,-5.332495,1.465908,3.046910,9.606878,0.049617,-2.369099,-4.444002,7.253428,-9.774135,3.596618,4.930016,3.309060],[-3.689983,-3.184305,-2.558179,-5.583478,-6.477438,-1.408755,7.797424,-6.795900,4.851948,-0.915931,-5.543934,-3.229401,-9.864920,-0.005550,-9.850505,-9.395562],[5.199641,-1.832351,1.285054,-6.743844,-5.016454,-0.898753,-6.508780,-6.181074,0.940848,0.128308,-4.897570,-5.603009,2.542017,0.610968,6.454360,-2.182328],[-3.956669,-9.066604,9.440998,1.238851,2.447784,-8.856082,-8.230598,-7.884356,9.015106,-5.613680,3.132361,-1.998815,6.921569,-1.134578,-0.673795,6.533926]]], dtype = "float32")#candidate|709|(13, 5, 16)|const|float32
bop_710 = relay.right_shift(uop_704.astype('uint32'), relay.reshape(const_709.astype('uint32'), relay.shape_of(uop_704))) # shape=(13, 5, 16)
output = relay.Tuple([call_675,var_676,const_677,bop_697,bop_701,call_706,var_707,bop_710,])
output2 = relay.Tuple([call_678,var_676,const_677,bop_697,bop_701,call_708,var_707,bop_710,])
func_713 = relay.Function([var_671,var_676,var_686,var_696,var_700,var_707,], output)
mod['func_713'] = func_713
mod = relay.transform.InferType()(mod)
var_714 = relay.var("var_714", dtype = "float64", shape = (13, 5, 16))#candidate|714|(13, 5, 16)|var|float64
var_715 = relay.var("var_715", dtype = "uint16", shape = ())#candidate|715|()|var|uint16
var_716 = relay.var("var_716", dtype = "float32", shape = (13, 5, 16))#candidate|716|(13, 5, 16)|var|float32
var_717 = relay.var("var_717", dtype = "float64", shape = (13, 5, 16))#candidate|717|(13, 5, 16)|var|float64
var_718 = relay.var("var_718", dtype = "uint16", shape = (13, 5, 16))#candidate|718|(13, 5, 16)|var|uint16
var_719 = relay.var("var_719", dtype = "float32", shape = (12,))#candidate|719|(12,)|var|float32
output = func_713(var_714,var_715,var_716,var_717,var_718,var_719,)
func_720 = relay.Function([var_714,var_715,var_716,var_717,var_718,var_719,], output)
mutated_mod['func_720'] = func_720
mutated_mod = relay.transform.InferType()(mutated_mod)
func_473_call = mod.get_global_var('func_473')
func_475_call = mutated_mod.get_global_var('func_475')
call_722 = relay.TupleGetItem(func_473_call(), 0)
call_723 = relay.TupleGetItem(func_475_call(), 0)
func_713_call = mod.get_global_var('func_713')
func_720_call = mutated_mod.get_global_var('func_720')
const_725 = relay.const([9.631826,5.523932,4.878170,2.736864,2.391033,-3.543016,-7.143422,8.119391,4.783079,3.161272,-7.226283,-1.091905,-4.975194,6.257771,3.702040,-0.869841,-9.723014,-6.709607,9.382656,8.082825,-7.182876,-2.751254,1.610802,-1.129707,2.302174,8.387043,-5.886851,-2.788797,-2.095574,-3.565921,-6.490876,2.673506,-1.374439,-0.498358,8.014241,-1.992886,0.003852,8.710461,7.701634,9.547445,1.269608,-1.024605,8.316187,-7.282545,-8.853144,0.899176,8.164481,-9.267759,-6.781818,4.222270,-4.648408,-2.466648,-0.060342,-5.521905,3.264024,1.605733,4.073759,-9.268973,-8.568967,-1.791100,0.699089,4.120528,-5.665501,2.355984,-0.705477,9.963087,-9.575056,-0.383404,9.266375,-5.977560,3.402803,3.674797,-4.409736,6.681300,-3.507198,7.361625,6.005739,-9.723332,-1.391739,1.689428,-8.150221,-7.733133,-6.298122,-0.665383,8.627954,-3.788489,-1.777627,7.549804,-9.962831,2.542353,8.560070,-7.928805,3.828031,-3.013645,-7.300495,-0.215804,-0.984578,-7.920620,6.408686,-3.576163,9.292541,9.988176,-1.035697,-2.508879,6.896762,9.066149,4.185939,-4.670041,-7.056700,-1.130877,-0.036942,-5.846009,4.789303,1.572281,3.541449,-0.451612,-1.957510,-9.464622,-8.519666,1.934344,-5.426764,-6.902969,2.800495,-6.944233,-4.868425,-8.468889,8.344507,6.883192,3.246151,8.100387,9.555815,-2.507690,-7.835648,-1.783572,8.157960,-1.321684,5.141207,9.335669,9.814656,-8.195489,7.717505,-4.237767,3.323284,7.130692,-8.764281,2.818274,-2.650808,-1.602541,-8.848140,-0.626276,-0.567831,8.995559,2.875561,-4.046430,-7.324551,-0.590061,-0.057527,-0.264284,0.755548,-1.618860,7.241745,8.837543,8.440439,-7.717510,-1.508016,-5.850404,7.288857,-8.304520,-4.400733,-7.634390,-9.253509,-6.869542,8.373407,-7.437262,6.033417,1.675934,8.718591,-0.198950,-1.150329,-2.120422,0.572874,8.506093,1.674606,-5.718224,0.928556,-6.494089,0.522241,-6.007003,-9.761964,9.628977,8.426600,-2.597804,8.297749,-8.102423,3.423439,9.060955,1.200801,0.854514,-8.723728,-6.033689,-7.323996,3.533386,-9.013348,1.554864,5.450877,-1.129970,8.586003,8.185841,9.931384,-3.254862,-5.813299,-5.823500,-9.436000,-1.832667,1.071417,-9.332822,-2.350292,7.906776,3.889006,3.195118,-1.791425,-5.750679,-4.542964,-8.979482,7.189667,-9.034627,-1.178149,6.845253,-3.718830,-0.794703,-7.925797,-6.748817,-5.077028,6.164128,-7.829937,-4.831485,-0.926789,-6.512401,-0.648892,8.301199,8.582399,-0.994011,-2.267513,-5.763596,6.445876,-7.421757,-3.189948,-8.110654,-8.609152,-9.964134,-3.187464,0.143989,0.634835,6.281133,6.821836,4.267552,-1.399110,-9.941527,4.821224,-8.986165,3.500531,9.839793,9.834630,8.197218,9.569891,-4.775223,-7.795641,9.363849,9.107950,-4.987281,8.665768,3.081550,9.211676,-1.387672,-4.814567,4.472597,8.453010,0.518058,2.455415,-2.625276,5.151544,-8.705017,7.546032,2.072688,3.364937,7.214061,-7.178810,-5.028092,-6.588521,-6.078165,8.704107,5.865764,-8.837769,1.869842,2.870903,3.985449,-4.371333,-7.512336,9.067431,-7.481318,0.169455,1.993215,-4.763156,5.226125,-0.339082,-7.264695,-5.851110,-5.073130,-1.656829,-5.614136,4.313452,-9.352678,0.941842,0.072472,3.254649,5.741119,8.556929,-4.067946,3.503624,2.987249,8.301198,8.071647,5.269660,3.701986,-7.266524,4.108864,4.619284,8.953832,-7.180673,3.635312,7.905071,6.171971,7.063134,-6.641293,8.180716,-2.983741,9.486282,-3.345884,0.477562,3.995468,9.559792,1.223988,-6.686687,-7.923373,-5.208895,-3.246499,-1.067359,-0.443678,8.055326,-8.546516,-3.678561,-4.352566,8.419645,-9.887472,-9.646550,-8.457084,-6.710622,-9.728272,-3.467071,-1.323557,-7.261094,-7.229350,-5.959269,-4.470708,-8.527233,-6.237957,6.752984,-9.139387,-4.941436,2.188336,-9.619064,-9.514161,0.576251,-4.687765,6.541674,-7.461866,9.561909,5.470949,-1.156941,-1.268974,-2.388173,0.893575,2.715805,-5.205651,2.842819,-6.456980,2.319013,-8.435404,9.504264,6.148465,2.939498,-2.426314,-6.740296,-1.234322,9.880596,4.658385,-2.506350,3.245507,0.789353,4.040073,0.557958,-6.473664,-2.580478,3.542944,8.107536,9.523619,4.900959,-0.334818,8.366639,-8.632642,3.510335,-1.843378,-3.087588,0.037524,-3.353174,-1.250980,-9.917653,-0.174901,-5.454723,-6.986800,-3.000661,-7.916175,7.854458,-2.129009,-4.348631,9.382594,-4.836852,3.597987,7.343535,-5.290421,0.228352,7.984048,-1.495670,-0.803734,2.206522,-8.236353,-6.415007,5.113628,1.966168,-6.002672,-8.212360,1.817420,-1.147671,-5.589138,2.246529,9.151456,6.086148,-0.567205,-4.266685,-3.673658,7.957333,-0.420014,-3.292865,0.612342,9.328551,8.966145,5.573873,-1.828897,6.747456,5.937283,-5.905242,5.622439,0.515107,8.106721,-0.518334,-4.973625,-4.871870,-1.625602,-0.198329,7.249480,4.264346,-8.449149,0.367933,-8.497784,0.302813,-2.012728,-6.219936,3.592821,5.038297,-9.616693,1.900946,-5.703405,1.975952,-6.951409,-9.729122,4.206900,-5.360628,3.080746,-2.071170,-4.390491,9.532248,0.036338,-6.595117,-8.416733,7.551874,3.614315,1.432369,1.568506,-4.585486,-9.723895,1.238616,2.553453,1.141215,4.881542,-8.460536,1.583790,-2.172552,-4.330364,-1.338289,9.934924,4.261660,-8.447367,-4.700951,2.866055,7.630052,3.963374,6.082000,-4.622831,6.969721,-2.159958,5.709636,-3.844579,4.867341,-3.607601,9.077942,9.443668,-2.455785,7.407138,0.078244,-8.418232,-0.790311,8.122522,-6.924493,3.808021,-8.740986,1.213844,6.788667,1.938613,-8.355232,-3.475275,7.948672,-6.437316,9.245705,-8.428344,-6.263063,2.912994,5.325174,-6.431873,-1.083073,5.122620,-7.489427,-9.994546,-9.875353,-5.339213,5.957153,-0.785726,-1.201767,0.863871,-8.905481,1.957017,-3.440864,-8.641768,3.095909,6.078686,-1.685370,-7.784005,7.535827,-0.293850,5.855034,6.490517,-6.190526,4.795154,-9.416655,-0.125265,2.712505,-4.743575,7.627600,9.373120,2.222428,-2.239667,7.592915,-3.256270,-4.666664,2.959232,9.443622,-3.944737,-1.039413,3.557727,-1.750809,-1.512630,3.669446,5.383701,8.862590,8.866200,0.511601,-6.087623,3.849028,-6.832427,2.915332,9.896202,-2.218807,-3.599591,7.635756,5.170501,-2.078842,7.509440,-6.945359,-1.868595,1.687854,1.405582,2.100919,-8.464549,-9.873780,-0.227986,-2.360587,1.983659,-1.567907,0.183293,1.640630,0.355673,-1.335387,6.942230,-3.603261,-0.351534,1.409406,-3.966670,6.891698,9.249137,6.707230,-1.183258,9.758668,-5.963400,1.890649,-9.578389,4.375850,-2.851030,-8.512725,-1.742115,1.624210,1.695152,-0.790956,-2.167048,-5.063275,-2.201839,1.271067,7.690185,-8.948487,-9.433870,6.837182,-7.734084,4.937719,7.231602,-3.992830,-2.297299,-2.185506,-3.796725,3.286126,6.088001,4.476790,-3.990599,-5.320691,8.720263,8.319394,-4.169518,-4.598801,8.412530,-1.504542,-9.775207,6.336060,-1.322700,-5.800695,-0.117639,2.333676,4.089295,1.644234,-9.709064,8.216539,-4.635733,9.184209,8.055454,1.910731,0.831713,6.111404,-4.063772,-6.525161,0.590829,-0.752590,2.912684,2.478994,-6.168498,-5.447812,-8.502579,-9.371393,-5.436014,4.043921,-0.702465,4.744877,-8.370895,9.507792,9.146074,-2.776163,-6.269383,-2.704841,-1.948274,-6.905374,3.571282,3.434613,-8.395237,-8.613371,-7.548216,-8.459521,6.205482,-0.685942,6.583865,-2.750903,-7.677980,0.385558,6.245671,-4.761783,6.831698,-7.602355,6.812212,-6.988243,-5.966879,-5.289180,-4.644822,7.200579,-6.495177,-1.814115,0.580518,-6.994443,3.152912,-5.881726,-8.591922,-0.017541,5.379159,-2.524740,5.938722,-6.504059,3.869382,-3.668860,5.772399,2.535063,3.795018,5.245201,2.297716,-8.808856,7.160568,0.580655,9.905215,4.822308,8.666865,3.722441,-9.684671,-3.656813,7.671337,-5.741045,-0.135741,-8.578907,4.284924,-3.662053,-6.197451,7.268531,8.323350,4.672489,-2.408474,2.444394,-4.060513,-8.776451,-4.158156,-0.235766,7.117607,-0.276179,0.997140,-8.289222,2.361860,8.553137,7.792936,6.372065,8.696637,-0.735723,-6.962400,5.909083,-5.062472,7.656088,-3.664453,-1.059044,-1.468036,4.294568,-0.241797,-6.356023,-5.556115,1.123045,-2.779322,-9.980360,2.256679,-3.165829,-0.926105,-5.542874,2.493850,2.699809,2.497141,-7.847238,4.882574,9.819474,6.083594,0.289363,2.247140,4.545885,0.369343,-9.115617,-8.575219,-8.792082,-1.114699,-5.285816,3.384160,-1.550961,9.253019,-4.683006,-1.102339,-2.415813,-2.921350,-8.814524,-9.429842,6.916952,1.991137,-4.387159,-4.389214,6.904264,8.834089,7.110338,2.159193,-0.941756,-3.471435,5.549200,-3.250129,-6.850992,2.687250,2.880796,-2.360055,-5.508929,-8.079119,-7.350676,5.117131,3.659688,0.618860,2.178385,-0.794028,4.125940,0.675066,-7.997124,2.960842,-7.284405,8.098227,-9.720680,-6.999281,1.529921,-0.376116,-5.013206,4.728223,-2.378919,-3.566481,-8.635724,8.337851,3.518058,-9.164461,9.354321,-8.012647,6.319201,6.077150,7.144199,-5.959341,-9.840171,-5.727614,-2.402756,-5.512197,-9.302286,1.062628,-4.863493,1.663231,2.268440,6.970262,5.810369,-2.118198,5.036741,-8.250778,4.031678,6.235491,9.473482,-4.034838,-6.593504,1.218723,0.795523,1.359145,1.448492,4.798860,-3.934695,4.851261,-2.857927,-5.406008,6.608892,1.302324,4.501916,-7.308032,-2.630751,5.947783,-1.290364,-9.198703,3.544751,1.991975,-4.490286,7.891387,5.579276,-4.525556,-9.621867,1.161912,8.820748,-7.792608,0.048062,-5.499643,-3.628198,-9.439188,-4.247386,-6.046178,-1.063951,-9.629198,7.247684,-1.888296,7.092006,-2.230541,1.847428,9.375941,-5.189965,-5.422194,9.470658,-7.326978,-9.793091,-2.531732,9.361895,-8.992610,-6.852872,-4.439237,-7.142881,-8.430297,1.422660,2.760450,-2.369764,2.926923,-8.650861,-1.715023,-7.190382,3.853085,-6.488275,0.657738,2.820912,8.285642,5.312069,7.990634,-7.846363,0.548046,-6.687699,0.566929,8.898473,-6.185215,-4.036019,5.679839,6.254557,3.507912,-5.998138,4.768305,-2.901859,8.203932,-2.225750,0.616474,-6.459559,-8.601922,2.196687,0.609994,-2.241437,-0.934207,4.012077,-6.764878,3.250081,-4.115405,-6.573554,4.796044,-0.011883,-2.405680,6.894614,-6.165326,-7.666594,-3.633220,-7.565827,9.686393,8.222982,4.761159,4.901077,4.800522,1.183330,4.845678,9.262194,-8.551195,5.814230,-8.385639,-8.133247,3.347902,2.810937,-2.617350,0.463752,5.372381,1.981280,0.733225,-4.516597,8.509553,-6.709159,8.467208,2.937941,-7.905155,7.012381,4.110261,9.523940,-6.638247,6.447996,-1.201906,-1.223876,1.546495,-5.339281,0.267222,-9.400443,-5.891397,5.600552,-6.914974,6.659115,9.741730,-5.477668,-3.497671,2.952626,-5.384500,2.265411,-4.956050], dtype = "float64")#candidate|725|(1040,)|const|float64
var_726 = relay.var("var_726", dtype = "uint16", shape = ())#candidate|726|()|var|uint16
var_727 = relay.var("var_727", dtype = "float32", shape = (12, 1))#candidate|727|(12, 1)|var|float32
call_724 = relay.TupleGetItem(func_713_call(relay.reshape(const_725.astype('float64'), [13, 5, 16]), relay.reshape(var_726.astype('uint16'), []), relay.reshape(const_725.astype('float32'), [13, 5, 16]), relay.reshape(const_725.astype('float64'), [13, 5, 16]), relay.reshape(const_725.astype('uint16'), [13, 5, 16]), relay.reshape(var_727.astype('float32'), [12,]), ), 1)
call_728 = relay.TupleGetItem(func_720_call(relay.reshape(const_725.astype('float64'), [13, 5, 16]), relay.reshape(var_726.astype('uint16'), []), relay.reshape(const_725.astype('float32'), [13, 5, 16]), relay.reshape(const_725.astype('float64'), [13, 5, 16]), relay.reshape(const_725.astype('uint16'), [13, 5, 16]), relay.reshape(var_727.astype('float32'), [12,]), ), 1)
bop_729 = relay.equal(var_727.astype('bool'), var_726.astype('bool')) # shape=(12, 1)
func_289_call = mod.get_global_var('func_289')
func_292_call = mutated_mod.get_global_var('func_292')
call_733 = relay.TupleGetItem(func_289_call(relay.reshape(call_722.astype('float64'), [5,]), relay.reshape(call_722.astype('float64'), [5,]), ), 9)
call_734 = relay.TupleGetItem(func_292_call(relay.reshape(call_722.astype('float64'), [5,]), relay.reshape(call_722.astype('float64'), [5,]), ), 9)
bop_744 = relay.logical_or(bop_729.astype('bool'), call_722.astype('bool')) # shape=(12, 5)
bop_747 = relay.logical_or(bop_729.astype('bool'), call_723.astype('bool')) # shape=(12, 5)
output = relay.Tuple([call_724,const_725,call_733,bop_744,])
output2 = relay.Tuple([call_728,const_725,call_734,bop_747,])
func_749 = relay.Function([var_726,var_727,], output)
mod['func_749'] = func_749
mod = relay.transform.InferType()(mod)
var_750 = relay.var("var_750", dtype = "uint16", shape = ())#candidate|750|()|var|uint16
var_751 = relay.var("var_751", dtype = "float32", shape = (12, 1))#candidate|751|(12, 1)|var|float32
output = func_749(var_750,var_751,)
func_752 = relay.Function([var_750,var_751,], output)
mutated_mod['func_752'] = func_752
mutated_mod = relay.transform.InferType()(mutated_mod)
var_758 = relay.var("var_758", dtype = "uint8", shape = (3, 5, 15))#candidate|758|(3, 5, 15)|var|uint8
var_759 = relay.var("var_759", dtype = "uint8", shape = (3, 5, 15))#candidate|759|(3, 5, 15)|var|uint8
bop_760 = relay.left_shift(var_758.astype('uint8'), relay.reshape(var_759.astype('uint8'), relay.shape_of(var_758))) # shape=(3, 5, 15)
uop_763 = relay.sigmoid(bop_760.astype('float32')) # shape=(3, 5, 15)
output = uop_763
output2 = uop_763
F = relay.Function([var_758,var_759,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_758,var_759,], output2)
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
	relay.transform.FuseOps(3),
	relay.transform.DefuseOps(),
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
input_758= np.array([[[-10,-1,3,-3,-1,-1,1,3,-10,-4,8,6,9,-5,-8],[10,-9,2,7,8,-8,-8,-4,-6,7,-5,-4,-10,6,9],[-2,-1,1,2,9,-3,-10,9,-1,8,5,-3,-6,-3,8],[8,10,-10,5,9,3,-2,-3,9,-5,10,-10,-10,-5,1],[-4,3,10,-7,7,-6,9,5,10,6,1,-7,8,-9,2]],[[-6,1,-4,-3,4,8,2,10,7,-1,5,-10,5,2,3],[-4,-2,-6,-5,-7,1,-9,-3,10,10,5,-5,8,3,4],[5,2,-4,-2,-1,8,-4,-7,-7,-7,10,2,6,-5,-1],[-2,-8,4,7,1,-6,-10,-7,7,3,10,-5,1,2,10],[-7,-2,1,7,3,-5,-8,-9,-1,-6,10,10,-8,-9,-6]],[[-8,1,8,-7,9,-4,8,9,-4,6,5,-7,3,3,-8],[1,2,7,6,-6,5,-9,-9,-2,9,2,-4,-8,-6,9],[1,-1,3,-4,3,4,-7,3,-1,8,5,-3,-3,-7,3],[-5,7,-9,6,-9,-5,-7,8,-10,-4,-1,10,-7,-5,2],[-10,4,2,10,-9,5,4,6,-1,-2,-7,-2,-7,10,3]]], dtype='uint8')
module1.set_input('var_758', input_758)
input_759= np.array([[[1,8,5,-9,6,-5,-1,4,-2,-8,-9,-7,1,-10,-5],[-2,10,-2,-8,9,-8,-8,5,10,6,-2,-10,6,3,10],[-2,-4,2,7,-6,7,-3,5,-4,3,-3,-5,-6,-4,-8],[8,-9,-10,10,-3,6,-2,-3,-4,4,-1,6,9,6,2],[-8,-3,5,-9,3,-1,2,-7,-2,5,10,-10,1,-10,-3]],[[-10,3,1,9,2,2,10,-7,-10,-7,5,-1,6,6,-9],[-8,-1,4,-8,-2,4,4,-7,-8,1,1,3,-3,-4,-6],[-10,-3,5,5,-10,-4,10,-8,-10,-3,10,-7,8,3,-8],[4,-3,-9,4,-2,10,1,-5,-6,-3,-5,4,9,1,5],[-3,7,6,4,9,-6,-5,8,6,-4,2,-4,-3,-10,7]],[[6,-4,3,1,2,7,-1,-7,-2,-3,-1,-8,-5,8,-7],[-1,-4,-8,-5,-6,2,-6,-4,4,-3,-4,5,-9,-3,6],[-3,-3,-1,4,1,-4,-3,-8,3,-5,-10,-3,9,5,-1],[1,9,-1,-6,-8,-4,2,5,6,-3,-6,-8,-9,-1,2],[-9,-7,9,-9,-2,-4,3,6,7,8,1,-4,-1,9,-1]]], dtype='uint8')
module1.set_input('var_759', input_759)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_758, input_759, )
res3 = intrp3.evaluate()(input_758, input_759, )
res4 = intrp4.evaluate()(input_758, input_759, )
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
module5.set_input('var_758', input_758)
module5.set_input('var_759', input_759)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_758, input_759, )
res7 = intrp7.evaluate()(input_758, input_759, )
res8 = intrp8.evaluate()(input_758, input_759, )
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
module9.set_input('var_758', input_758)
module9.set_input('var_759', input_759)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_758, input_759, )
res11 = intrp11.evaluate()(input_758, input_759, )
res12 = intrp12.evaluate()(input_758, input_759, )
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
module13.set_input('var_758', input_758)
module13.set_input('var_759', input_759)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_758, input_759, )
res15 = intrp15.evaluate()(input_758, input_759, )
res16 = intrp16.evaluate()(input_758, input_759, )
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
module17.set_input('var_758', input_758)
module17.set_input('var_759', input_759)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_758, input_759, )
res19 = intrp19.evaluate()(input_758, input_759, )
res20 = intrp20.evaluate()(input_758, input_759, )
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
module21.set_input('var_758', input_758)
module21.set_input('var_759', input_759)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_758, input_759, )
res23 = intrp23.evaluate()(input_758, input_759, )
res24 = intrp24.evaluate()(input_758, input_759, )
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