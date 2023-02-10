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
var_0 = relay.var("var_0", dtype = "uint32", shape = (14,))#candidate|0|(14,)|var|uint32
const_1 = relay.const([3,8,9,5,-1,-4,-2,2,9,-4,7,-10,-2,-2], dtype = "uint32")#candidate|1|(14,)|const|uint32
bop_2 = relay.greater_equal(var_0.astype('bool'), relay.reshape(const_1.astype('bool'), relay.shape_of(var_0))) # shape=(14,)
uop_5 = relay.asinh(bop_2.astype('float32')) # shape=(14,)
uop_7 = relay.erf(const_1.astype('float64')) # shape=(14,)
output = relay.Tuple([uop_5,uop_7,])
output2 = relay.Tuple([uop_5,uop_7,])
func_9 = relay.Function([var_0,], output)
mod['func_9'] = func_9
mod = relay.transform.InferType()(mod)
mutated_mod['func_9'] = func_9
mutated_mod = relay.transform.InferType()(mutated_mod)
var_10 = relay.var("var_10", dtype = "uint32", shape = (14,))#candidate|10|(14,)|var|uint32
func_9_call = mutated_mod.get_global_var('func_9')
call_11 = func_9_call(var_10)
output = call_11
func_12 = relay.Function([var_10], output)
mutated_mod['func_12'] = func_12
mutated_mod = relay.transform.InferType()(mutated_mod)
var_14 = relay.var("var_14", dtype = "float32", shape = (7, 5))#candidate|14|(7, 5)|var|float32
uop_15 = relay.acos(var_14.astype('float32')) # shape=(7, 5)
bop_17 = relay.minimum(var_14.astype('uint64'), relay.reshape(uop_15.astype('uint64'), relay.shape_of(var_14))) # shape=(7, 5)
uop_20 = relay.erf(uop_15.astype('float32')) # shape=(7, 5)
uop_22 = relay.acos(uop_20.astype('float64')) # shape=(7, 5)
func_9_call = mod.get_global_var('func_9')
func_12_call = mutated_mod.get_global_var('func_12')
var_25 = relay.var("var_25", dtype = "uint32", shape = (14,))#candidate|25|(14,)|var|uint32
call_24 = relay.TupleGetItem(func_9_call(relay.reshape(var_25.astype('uint32'), [14,])), 1)
call_26 = relay.TupleGetItem(func_12_call(relay.reshape(var_25.astype('uint32'), [14,])), 1)
uop_27 = relay.exp(uop_15.astype('float64')) # shape=(7, 5)
uop_29 = relay.atan(uop_22.astype('float64')) # shape=(7, 5)
uop_31 = relay.rsqrt(uop_20.astype('float32')) # shape=(7, 5)
func_9_call = mod.get_global_var('func_9')
func_12_call = mutated_mod.get_global_var('func_12')
call_33 = relay.TupleGetItem(func_9_call(relay.reshape(var_25.astype('uint32'), [14,])), 0)
call_34 = relay.TupleGetItem(func_12_call(relay.reshape(var_25.astype('uint32'), [14,])), 0)
uop_35 = relay.log2(uop_29.astype('float32')) # shape=(7, 5)
func_9_call = mod.get_global_var('func_9')
func_12_call = mutated_mod.get_global_var('func_12')
call_37 = relay.TupleGetItem(func_9_call(relay.reshape(call_33.astype('uint32'), [14,])), 0)
call_38 = relay.TupleGetItem(func_12_call(relay.reshape(call_33.astype('uint32'), [14,])), 0)
bop_39 = relay.minimum(uop_29.astype('uint16'), relay.reshape(uop_35.astype('uint16'), relay.shape_of(uop_29))) # shape=(7, 5)
bop_42 = relay.add(uop_29.astype('int8'), relay.reshape(uop_20.astype('int8'), relay.shape_of(uop_29))) # shape=(7, 5)
uop_45 = relay.asin(bop_39.astype('float64')) # shape=(7, 5)
uop_47 = relay.sinh(uop_22.astype('float32')) # shape=(7, 5)
var_49 = relay.var("var_49", dtype = "float32", shape = (7, 5))#candidate|49|(7, 5)|var|float32
bop_50 = relay.equal(uop_35.astype('bool'), relay.reshape(var_49.astype('bool'), relay.shape_of(uop_35))) # shape=(7, 5)
output = relay.Tuple([bop_17,call_24,var_25,uop_27,uop_31,call_33,call_37,bop_42,uop_45,uop_47,bop_50,])
output2 = relay.Tuple([bop_17,call_26,var_25,uop_27,uop_31,call_34,call_38,bop_42,uop_45,uop_47,bop_50,])
func_53 = relay.Function([var_14,var_25,var_49,], output)
mod['func_53'] = func_53
mod = relay.transform.InferType()(mod)
mutated_mod['func_53'] = func_53
mutated_mod = relay.transform.InferType()(mutated_mod)
func_53_call = mutated_mod.get_global_var('func_53')
var_55 = relay.var("var_55", dtype = "float32", shape = (7, 5))#candidate|55|(7, 5)|var|float32
var_56 = relay.var("var_56", dtype = "uint32", shape = (14,))#candidate|56|(14,)|var|uint32
var_57 = relay.var("var_57", dtype = "float32", shape = (7, 5))#candidate|57|(7, 5)|var|float32
call_54 = func_53_call(var_55,var_56,var_57,)
output = call_54
func_58 = relay.Function([var_55,var_56,var_57,], output)
mutated_mod['func_58'] = func_58
mutated_mod = relay.transform.InferType()(mutated_mod)
var_60 = relay.var("var_60", dtype = "float32", shape = (1,))#candidate|60|(1,)|var|float32
uop_61 = relay.asinh(var_60.astype('float32')) # shape=(1,)
uop_63 = relay.atanh(uop_61.astype('float64')) # shape=(1,)
bop_65 = relay.greater(uop_61.astype('bool'), relay.reshape(uop_63.astype('bool'), relay.shape_of(uop_61))) # shape=(1,)
bop_68 = relay.maximum(uop_61.astype('int8'), relay.reshape(var_60.astype('int8'), relay.shape_of(uop_61))) # shape=(1,)
uop_71 = relay.log(bop_68.astype('float64')) # shape=(1,)
uop_73 = relay.tan(bop_68.astype('float32')) # shape=(1,)
bop_75 = relay.equal(uop_71.astype('bool'), relay.reshape(uop_61.astype('bool'), relay.shape_of(uop_71))) # shape=(1,)
bop_78 = relay.less(bop_65.astype('bool'), relay.reshape(bop_68.astype('bool'), relay.shape_of(bop_65))) # shape=(1,)
uop_81 = relay.cosh(bop_75.astype('float64')) # shape=(1,)
var_83 = relay.var("var_83", dtype = "float64", shape = (10,))#candidate|83|(10,)|var|float64
bop_84 = relay.greater_equal(uop_81.astype('bool'), var_83.astype('bool')) # shape=(10,)
uop_87 = relay.log2(bop_84.astype('float32')) # shape=(10,)
bop_89 = relay.equal(bop_84.astype('bool'), bop_68.astype('bool')) # shape=(10,)
uop_92 = relay.acosh(uop_87.astype('float32')) # shape=(10,)
var_94 = relay.var("var_94", dtype = "float32", shape = (10,))#candidate|94|(10,)|var|float32
bop_95 = relay.less(uop_92.astype('bool'), relay.reshape(var_94.astype('bool'), relay.shape_of(uop_92))) # shape=(10,)
output = relay.Tuple([uop_73,bop_78,bop_89,bop_95,])
output2 = relay.Tuple([uop_73,bop_78,bop_89,bop_95,])
func_98 = relay.Function([var_60,var_83,var_94,], output)
mod['func_98'] = func_98
mod = relay.transform.InferType()(mod)
mutated_mod['func_98'] = func_98
mutated_mod = relay.transform.InferType()(mutated_mod)
func_98_call = mutated_mod.get_global_var('func_98')
var_100 = relay.var("var_100", dtype = "float32", shape = (1,))#candidate|100|(1,)|var|float32
var_101 = relay.var("var_101", dtype = "float64", shape = (10,))#candidate|101|(10,)|var|float64
var_102 = relay.var("var_102", dtype = "float32", shape = (10,))#candidate|102|(10,)|var|float32
call_99 = func_98_call(var_100,var_101,var_102,)
output = call_99
func_103 = relay.Function([var_100,var_101,var_102,], output)
mutated_mod['func_103'] = func_103
mutated_mod = relay.transform.InferType()(mutated_mod)
var_105 = relay.var("var_105", dtype = "float64", shape = (3, 8))#candidate|105|(3, 8)|var|float64
uop_106 = relay.sigmoid(var_105.astype('float64')) # shape=(3, 8)
uop_108 = relay.tan(var_105.astype('float64')) # shape=(3, 8)
uop_110 = relay.asinh(var_105.astype('float64')) # shape=(3, 8)
bop_112 = relay.equal(uop_108.astype('bool'), relay.reshape(uop_110.astype('bool'), relay.shape_of(uop_108))) # shape=(3, 8)
var_115 = relay.var("var_115", dtype = "float64", shape = (3, 8))#candidate|115|(3, 8)|var|float64
bop_116 = relay.multiply(uop_110.astype('float32'), relay.reshape(var_115.astype('float32'), relay.shape_of(uop_110))) # shape=(3, 8)
bop_119 = relay.mod(bop_116.astype('float64'), relay.reshape(uop_106.astype('float64'), relay.shape_of(bop_116))) # shape=(3, 8)
bop_122 = relay.greater(bop_116.astype('bool'), relay.reshape(var_115.astype('bool'), relay.shape_of(bop_116))) # shape=(3, 8)
func_53_call = mod.get_global_var('func_53')
func_58_call = mutated_mod.get_global_var('func_58')
const_126 = relay.const([8.159684,3.141176,4.547136,2.307114,2.996610,-6.646319,-2.975361,-1.180902,-0.869595,-0.588056,-6.164492,-8.364479,8.315935,-2.230372,-1.970567,-4.845247,6.654131,0.923398,-1.528983,-4.065569,7.101624,-8.392851,2.279062,9.410322,-6.386475,2.041001,9.532975,1.837779,2.688996,3.038207,-6.888619,0.632971,-2.870573,5.711763,8.259040], dtype = "float32")#candidate|126|(35,)|const|float32
var_127 = relay.var("var_127", dtype = "uint32", shape = (1, 14))#candidate|127|(1, 14)|var|uint32
call_125 = relay.TupleGetItem(func_53_call(relay.reshape(const_126.astype('float32'), [7, 5]), relay.reshape(var_127.astype('uint32'), [14,]), relay.reshape(const_126.astype('float32'), [7, 5]), ), 6)
call_128 = relay.TupleGetItem(func_58_call(relay.reshape(const_126.astype('float32'), [7, 5]), relay.reshape(var_127.astype('uint32'), [14,]), relay.reshape(const_126.astype('float32'), [7, 5]), ), 6)
uop_129 = relay.cosh(var_127.astype('float32')) # shape=(1, 14)
var_131 = relay.var("var_131", dtype = "float64", shape = (3, 8))#candidate|131|(3, 8)|var|float64
bop_132 = relay.power(uop_110.astype('float64'), relay.reshape(var_131.astype('float64'), relay.shape_of(uop_110))) # shape=(3, 8)
bop_135 = relay.multiply(bop_119.astype('uint64'), relay.reshape(var_105.astype('uint64'), relay.shape_of(bop_119))) # shape=(3, 8)
uop_138 = relay.asin(bop_112.astype('float32')) # shape=(3, 8)
var_140 = relay.var("var_140", dtype = "float32", shape = (35,))#candidate|140|(35,)|var|float32
bop_141 = relay.multiply(const_126.astype('int64'), relay.reshape(var_140.astype('int64'), relay.shape_of(const_126))) # shape=(35,)
var_144 = relay.var("var_144", dtype = "float32", shape = (3, 8))#candidate|144|(3, 8)|var|float32
bop_145 = relay.power(uop_138.astype('float32'), relay.reshape(var_144.astype('float32'), relay.shape_of(uop_138))) # shape=(3, 8)
bop_148 = relay.power(bop_122.astype('float64'), relay.reshape(uop_110.astype('float64'), relay.shape_of(bop_122))) # shape=(3, 8)
uop_151 = relay.asin(const_126.astype('float64')) # shape=(35,)
var_153 = relay.var("var_153", dtype = "uint32", shape = (5, 14))#candidate|153|(5, 14)|var|uint32
bop_154 = relay.logical_and(var_127.astype('bool'), var_153.astype('bool')) # shape=(5, 14)
uop_157 = relay.acos(bop_145.astype('float64')) # shape=(3, 8)
uop_159 = relay.log10(bop_116.astype('float32')) # shape=(3, 8)
output = relay.Tuple([call_125,uop_129,bop_132,bop_135,bop_141,bop_148,uop_151,bop_154,uop_157,uop_159,])
output2 = relay.Tuple([call_128,uop_129,bop_132,bop_135,bop_141,bop_148,uop_151,bop_154,uop_157,uop_159,])
func_161 = relay.Function([var_105,var_115,var_127,var_131,var_140,var_144,var_153,], output)
mod['func_161'] = func_161
mod = relay.transform.InferType()(mod)
var_162 = relay.var("var_162", dtype = "float64", shape = (3, 8))#candidate|162|(3, 8)|var|float64
var_163 = relay.var("var_163", dtype = "float64", shape = (3, 8))#candidate|163|(3, 8)|var|float64
var_164 = relay.var("var_164", dtype = "uint32", shape = (1, 14))#candidate|164|(1, 14)|var|uint32
var_165 = relay.var("var_165", dtype = "float64", shape = (3, 8))#candidate|165|(3, 8)|var|float64
var_166 = relay.var("var_166", dtype = "float32", shape = (35,))#candidate|166|(35,)|var|float32
var_167 = relay.var("var_167", dtype = "float32", shape = (3, 8))#candidate|167|(3, 8)|var|float32
var_168 = relay.var("var_168", dtype = "uint32", shape = (5, 14))#candidate|168|(5, 14)|var|uint32
output = func_161(var_162,var_163,var_164,var_165,var_166,var_167,var_168,)
func_169 = relay.Function([var_162,var_163,var_164,var_165,var_166,var_167,var_168,], output)
mutated_mod['func_169'] = func_169
mutated_mod = relay.transform.InferType()(mutated_mod)
var_171 = relay.var("var_171", dtype = "float64", shape = (1, 10))#candidate|171|(1, 10)|var|float64
uop_172 = relay.sin(var_171.astype('float64')) # shape=(1, 10)
bop_174 = relay.not_equal(var_171.astype('bool'), relay.reshape(uop_172.astype('bool'), relay.shape_of(var_171))) # shape=(1, 10)
bop_177 = relay.maximum(bop_174.astype('uint64'), relay.reshape(uop_172.astype('uint64'), relay.shape_of(bop_174))) # shape=(1, 10)
uop_180 = relay.rsqrt(var_171.astype('float32')) # shape=(1, 10)
bop_182 = relay.minimum(var_171.astype('int8'), relay.reshape(bop_177.astype('int8'), relay.shape_of(var_171))) # shape=(1, 10)
bop_185 = relay.subtract(var_171.astype('float32'), relay.reshape(bop_182.astype('float32'), relay.shape_of(var_171))) # shape=(1, 10)
bop_188 = relay.bitwise_and(var_171.astype('int8'), relay.reshape(bop_174.astype('int8'), relay.shape_of(var_171))) # shape=(1, 10)
uop_191 = relay.log10(bop_177.astype('float32')) # shape=(1, 10)
uop_193 = relay.atanh(uop_191.astype('float64')) # shape=(1, 10)
output = relay.Tuple([uop_180,bop_185,bop_188,uop_193,])
output2 = relay.Tuple([uop_180,bop_185,bop_188,uop_193,])
func_195 = relay.Function([var_171,], output)
mod['func_195'] = func_195
mod = relay.transform.InferType()(mod)
mutated_mod['func_195'] = func_195
mutated_mod = relay.transform.InferType()(mutated_mod)
var_196 = relay.var("var_196", dtype = "float64", shape = (1, 10))#candidate|196|(1, 10)|var|float64
func_195_call = mutated_mod.get_global_var('func_195')
call_197 = func_195_call(var_196)
output = call_197
func_198 = relay.Function([var_196], output)
mutated_mod['func_198'] = func_198
mutated_mod = relay.transform.InferType()(mutated_mod)
var_200 = relay.var("var_200", dtype = "int8", shape = (4, 5))#candidate|200|(4, 5)|var|int8
var_201 = relay.var("var_201", dtype = "int8", shape = (4, 5))#candidate|201|(4, 5)|var|int8
bop_202 = relay.bitwise_and(var_200.astype('int8'), relay.reshape(var_201.astype('int8'), relay.shape_of(var_200))) # shape=(4, 5)
bop_205 = relay.bitwise_or(var_200.astype('int32'), relay.reshape(bop_202.astype('int32'), relay.shape_of(var_200))) # shape=(4, 5)
var_208 = relay.var("var_208", dtype = "int8", shape = (4, 5))#candidate|208|(4, 5)|var|int8
bop_209 = relay.subtract(var_200.astype('uint64'), relay.reshape(var_208.astype('uint64'), relay.shape_of(var_200))) # shape=(4, 5)
uop_212 = relay.sigmoid(var_201.astype('float32')) # shape=(4, 5)
uop_214 = relay.atan(bop_205.astype('float32')) # shape=(4, 5)
const_216 = relay.const([[-3,9,-6,-6,-6],[-2,5,-7,-8,1],[-9,10,10,7,-10],[-1,7,-9,-6,8]], dtype = "int32")#candidate|216|(4, 5)|const|int32
bop_217 = relay.greater(bop_205.astype('bool'), relay.reshape(const_216.astype('bool'), relay.shape_of(bop_205))) # shape=(4, 5)
var_220 = relay.var("var_220", dtype = "float32", shape = (4, 5))#candidate|220|(4, 5)|var|float32
bop_221 = relay.subtract(uop_214.astype('float32'), relay.reshape(var_220.astype('float32'), relay.shape_of(uop_214))) # shape=(4, 5)
var_224 = relay.var("var_224", dtype = "float32", shape = (4, 5))#candidate|224|(4, 5)|var|float32
bop_225 = relay.less(uop_214.astype('bool'), relay.reshape(var_224.astype('bool'), relay.shape_of(uop_214))) # shape=(4, 5)
bop_228 = relay.logical_and(bop_202.astype('bool'), relay.reshape(var_220.astype('bool'), relay.shape_of(bop_202))) # shape=(4, 5)
uop_231 = relay.tan(bop_205.astype('float64')) # shape=(4, 5)
var_233 = relay.var("var_233", dtype = "bool", shape = (4, 5))#candidate|233|(4, 5)|var|bool
bop_234 = relay.bitwise_xor(bop_225.astype('uint16'), relay.reshape(var_233.astype('uint16'), relay.shape_of(bop_225))) # shape=(4, 5)
bop_237 = relay.less(bop_234.astype('bool'), relay.reshape(var_201.astype('bool'), relay.shape_of(bop_234))) # shape=(4, 5)
const_240 = relay.const([[6.575409,7.855341,9.685416,2.705182,-3.520229],[-2.041481,-7.043130,-2.220958,-0.296527,-4.408156],[-4.064203,6.052481,4.488857,2.079242,3.846256],[-4.816356,1.349482,4.482320,6.679130,-7.866716]], dtype = "float32")#candidate|240|(4, 5)|const|float32
bop_241 = relay.minimum(bop_221.astype('uint16'), relay.reshape(const_240.astype('uint16'), relay.shape_of(bop_221))) # shape=(4, 5)
uop_244 = relay.atanh(bop_237.astype('float64')) # shape=(4, 5)
uop_246 = relay.sin(bop_237.astype('float64')) # shape=(4, 5)
output = relay.Tuple([bop_209,uop_212,bop_217,bop_228,uop_231,bop_241,uop_244,uop_246,])
output2 = relay.Tuple([bop_209,uop_212,bop_217,bop_228,uop_231,bop_241,uop_244,uop_246,])
func_248 = relay.Function([var_200,var_201,var_208,var_220,var_224,var_233,], output)
mod['func_248'] = func_248
mod = relay.transform.InferType()(mod)
var_249 = relay.var("var_249", dtype = "int8", shape = (4, 5))#candidate|249|(4, 5)|var|int8
var_250 = relay.var("var_250", dtype = "int8", shape = (4, 5))#candidate|250|(4, 5)|var|int8
var_251 = relay.var("var_251", dtype = "int8", shape = (4, 5))#candidate|251|(4, 5)|var|int8
var_252 = relay.var("var_252", dtype = "float32", shape = (4, 5))#candidate|252|(4, 5)|var|float32
var_253 = relay.var("var_253", dtype = "float32", shape = (4, 5))#candidate|253|(4, 5)|var|float32
var_254 = relay.var("var_254", dtype = "bool", shape = (4, 5))#candidate|254|(4, 5)|var|bool
output = func_248(var_249,var_250,var_251,var_252,var_253,var_254,)
func_255 = relay.Function([var_249,var_250,var_251,var_252,var_253,var_254,], output)
mutated_mod['func_255'] = func_255
mutated_mod = relay.transform.InferType()(mutated_mod)
var_257 = relay.var("var_257", dtype = "float64", shape = ())#candidate|257|()|var|float64
uop_258 = relay.asin(var_257.astype('float64')) # shape=()
const_260 = relay.const(1.073822, dtype = "float64")#candidate|260|()|const|float64
bop_261 = relay.mod(uop_258.astype('float32'), const_260.astype('float32')) # shape=()
bop_264 = relay.add(uop_258.astype('uint8'), const_260.astype('uint8')) # shape=()
bop_267 = relay.bitwise_xor(bop_261.astype('int64'), var_257.astype('int64')) # shape=()
var_270 = relay.var("var_270", dtype = "int64", shape = (6, 14, 16))#candidate|270|(6, 14, 16)|var|int64
bop_271 = relay.logical_or(bop_267.astype('bool'), var_270.astype('bool')) # shape=(6, 14, 16)
bop_274 = relay.subtract(bop_267.astype('int16'), var_257.astype('int16')) # shape=()
bop_277 = relay.floor_divide(bop_271.astype('float64'), var_257.astype('float64')) # shape=(6, 14, 16)
uop_280 = relay.acos(bop_271.astype('float32')) # shape=(6, 14, 16)
bop_282 = relay.floor_mod(bop_274.astype('float32'), bop_267.astype('float32')) # shape=()
bop_285 = relay.floor_divide(bop_261.astype('float64'), bop_271.astype('float64')) # shape=(6, 14, 16)
bop_288 = relay.power(uop_280.astype('float32'), relay.reshape(bop_271.astype('float32'), relay.shape_of(uop_280))) # shape=(6, 14, 16)
output = relay.Tuple([bop_264,bop_277,bop_282,bop_285,bop_288,])
output2 = relay.Tuple([bop_264,bop_277,bop_282,bop_285,bop_288,])
func_291 = relay.Function([var_257,var_270,], output)
mod['func_291'] = func_291
mod = relay.transform.InferType()(mod)
var_292 = relay.var("var_292", dtype = "float64", shape = ())#candidate|292|()|var|float64
var_293 = relay.var("var_293", dtype = "int64", shape = (6, 14, 16))#candidate|293|(6, 14, 16)|var|int64
output = func_291(var_292,var_293,)
func_294 = relay.Function([var_292,var_293,], output)
mutated_mod['func_294'] = func_294
mutated_mod = relay.transform.InferType()(mutated_mod)
var_296 = relay.var("var_296", dtype = "float64", shape = (9, 11))#candidate|296|(9, 11)|var|float64
uop_297 = relay.tan(var_296.astype('float64')) # shape=(9, 11)
uop_299 = relay.cosh(var_296.astype('float32')) # shape=(9, 11)
uop_301 = relay.log10(uop_299.astype('float32')) # shape=(9, 11)
uop_303 = relay.asin(uop_301.astype('float64')) # shape=(9, 11)
const_305 = relay.const([[-4.243588,4.711967,-1.049578,0.823051,8.067121,-9.542789,-6.874541,-6.769865,6.350508,5.923634,8.726532],[-3.499994,-8.303491,2.578039,-7.601465,-3.626663,2.342140,9.255073,-7.939023,-9.954580,1.024310,-9.413860],[-7.846869,-6.056326,-7.197006,-1.754233,-3.882311,8.150811,-2.827012,5.984133,3.806655,2.992016,2.445644],[-3.321506,-2.673855,-2.699230,7.190937,1.847561,6.120917,3.456785,3.438925,1.085961,2.638909,-1.700647],[-2.035051,-1.367542,-5.213279,-5.061224,-9.113595,5.630820,4.301973,-1.976165,-9.280382,-1.200904,4.952700],[0.431976,-9.119312,-1.581733,2.609817,5.380287,-0.861649,6.386653,7.525878,5.585700,-6.203158,-3.028406],[-4.471986,-4.064838,-1.678932,-7.291566,0.059263,9.145558,-7.593027,0.907830,-2.522755,7.544549,-1.492835],[3.533398,-2.149341,-3.282368,2.517326,-6.575757,6.014335,8.048454,-9.182804,-4.692149,5.825600,3.590648],[9.204077,-6.913780,9.031734,2.687062,-4.404621,7.932176,9.305070,-3.913069,8.275701,-8.970226,1.390842]], dtype = "float64")#candidate|305|(9, 11)|const|float64
bop_306 = relay.power(uop_303.astype('float64'), relay.reshape(const_305.astype('float64'), relay.shape_of(uop_303))) # shape=(9, 11)
uop_309 = relay.asin(bop_306.astype('float64')) # shape=(9, 11)
bop_311 = relay.bitwise_and(uop_309.astype('int64'), relay.reshape(uop_303.astype('int64'), relay.shape_of(uop_309))) # shape=(9, 11)
bop_314 = relay.logical_or(var_296.astype('bool'), relay.reshape(uop_297.astype('bool'), relay.shape_of(var_296))) # shape=(9, 11)
uop_317 = relay.sqrt(uop_301.astype('float32')) # shape=(9, 11)
bop_319 = relay.greater(uop_309.astype('bool'), relay.reshape(uop_303.astype('bool'), relay.shape_of(uop_309))) # shape=(9, 11)
output = relay.Tuple([bop_311,bop_314,uop_317,bop_319,])
output2 = relay.Tuple([bop_311,bop_314,uop_317,bop_319,])
func_322 = relay.Function([var_296,], output)
mod['func_322'] = func_322
mod = relay.transform.InferType()(mod)
var_323 = relay.var("var_323", dtype = "float64", shape = (9, 11))#candidate|323|(9, 11)|var|float64
output = func_322(var_323)
func_324 = relay.Function([var_323], output)
mutated_mod['func_324'] = func_324
mutated_mod = relay.transform.InferType()(mutated_mod)
var_326 = relay.var("var_326", dtype = "uint8", shape = (3, 2, 8))#candidate|326|(3, 2, 8)|var|uint8
var_327 = relay.var("var_327", dtype = "uint8", shape = (3, 2, 8))#candidate|327|(3, 2, 8)|var|uint8
bop_328 = relay.multiply(var_326.astype('uint8'), relay.reshape(var_327.astype('uint8'), relay.shape_of(var_326))) # shape=(3, 2, 8)
const_331 = relay.const([[[3,2,7,2,-10,-4,-10,2],[-7,-9,-3,-8,-5,10,-6,-7]],[[10,-10,5,1,-2,-2,-9,2],[-9,8,10,-5,-10,8,-2,7]],[[-4,1,9,-9,-6,-3,5,10],[-8,8,-4,-6,9,-10,-9,-3]]], dtype = "uint8")#candidate|331|(3, 2, 8)|const|uint8
bop_332 = relay.left_shift(bop_328.astype('int16'), relay.reshape(const_331.astype('int16'), relay.shape_of(bop_328))) # shape=(3, 2, 8)
uop_335 = relay.sigmoid(bop_332.astype('float32')) # shape=(3, 2, 8)
output = relay.Tuple([uop_335,])
output2 = relay.Tuple([uop_335,])
func_337 = relay.Function([var_326,var_327,], output)
mod['func_337'] = func_337
mod = relay.transform.InferType()(mod)
var_338 = relay.var("var_338", dtype = "uint8", shape = (3, 2, 8))#candidate|338|(3, 2, 8)|var|uint8
var_339 = relay.var("var_339", dtype = "uint8", shape = (3, 2, 8))#candidate|339|(3, 2, 8)|var|uint8
output = func_337(var_338,var_339,)
func_340 = relay.Function([var_338,var_339,], output)
mutated_mod['func_340'] = func_340
mutated_mod = relay.transform.InferType()(mutated_mod)
var_342 = relay.var("var_342", dtype = "uint32", shape = (14,))#candidate|342|(14,)|var|uint32
var_343 = relay.var("var_343", dtype = "uint32", shape = (14,))#candidate|343|(14,)|var|uint32
bop_344 = relay.multiply(var_342.astype('uint32'), relay.reshape(var_343.astype('uint32'), relay.shape_of(var_342))) # shape=(14,)
bop_347 = relay.logical_and(var_343.astype('bool'), relay.reshape(var_342.astype('bool'), relay.shape_of(var_343))) # shape=(14,)
output = relay.Tuple([bop_344,bop_347,])
output2 = relay.Tuple([bop_344,bop_347,])
func_350 = relay.Function([var_342,var_343,], output)
mod['func_350'] = func_350
mod = relay.transform.InferType()(mod)
mutated_mod['func_350'] = func_350
mutated_mod = relay.transform.InferType()(mutated_mod)
func_350_call = mutated_mod.get_global_var('func_350')
var_352 = relay.var("var_352", dtype = "uint32", shape = (14,))#candidate|352|(14,)|var|uint32
var_353 = relay.var("var_353", dtype = "uint32", shape = (14,))#candidate|353|(14,)|var|uint32
call_351 = func_350_call(var_352,var_353,)
output = call_351
func_354 = relay.Function([var_352,var_353,], output)
mutated_mod['func_354'] = func_354
mutated_mod = relay.transform.InferType()(mutated_mod)
var_356 = relay.var("var_356", dtype = "int64", shape = ())#candidate|356|()|var|int64
const_357 = relay.const(-3, dtype = "int64")#candidate|357|()|const|int64
bop_358 = relay.add(var_356.astype('int64'), const_357.astype('int64')) # shape=()
bop_361 = relay.not_equal(bop_358.astype('bool'), var_356.astype('bool')) # shape=()
output = relay.Tuple([bop_361,])
output2 = relay.Tuple([bop_361,])
F = relay.Function([var_356,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_356,], output2)
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
input_356= np.array(-4, dtype='int64')
module1.set_input('var_356', input_356)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_356, )
res3 = intrp3.evaluate()(input_356, )
res4 = intrp4.evaluate()(input_356, )
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
module5.set_input('var_356', input_356)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_356, )
res7 = intrp7.evaluate()(input_356, )
res8 = intrp8.evaluate()(input_356, )
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
module9.set_input('var_356', input_356)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_356, )
res11 = intrp11.evaluate()(input_356, )
res12 = intrp12.evaluate()(input_356, )
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
module13.set_input('var_356', input_356)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_356, )
res15 = intrp15.evaluate()(input_356, )
res16 = intrp16.evaluate()(input_356, )
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
module17.set_input('var_356', input_356)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_356, )
res19 = intrp19.evaluate()(input_356, )
res20 = intrp20.evaluate()(input_356, )
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
module21.set_input('var_356', input_356)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_356, )
res23 = intrp23.evaluate()(input_356, )
res24 = intrp24.evaluate()(input_356, )
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

'''45: TVMFuncCall
44: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
43: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
42: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
41: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
40: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
39: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
38: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
37: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
36: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
34: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
32: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
31: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
23: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
22: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
21: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
19: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
15: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
14: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
13: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
12: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
11: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
10: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
9: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
8: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
7: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
6: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
5: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
4: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9RelayEx
1: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::ConstructorNode const*)
0: tvm::IRModuleNode::LookupTypeDef(tvm::GlobalTypeVar const&) const

'''