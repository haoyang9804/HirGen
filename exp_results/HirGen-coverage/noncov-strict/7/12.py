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
var_0 = relay.var("var_0", dtype = "float32", shape = ())#candidate|0|()|var|float32
uop_1 = relay.log(var_0.astype('float32')) # shape=()
uop_3 = relay.log(uop_1.astype('float64')) # shape=()
uop_5 = relay.asinh(var_0.astype('float64')) # shape=()
uop_7 = relay.cos(uop_3.astype('float64')) # shape=()
uop_9 = relay.cos(uop_3.astype('float32')) # shape=()
bop_11 = relay.logical_and(uop_9.astype('bool'), uop_5.astype('bool')) # shape=()
bop_14 = relay.bitwise_and(uop_3.astype('int8'), uop_1.astype('int8')) # shape=()
uop_17 = relay.sin(uop_1.astype('float32')) # shape=()
bop_19 = relay.multiply(uop_7.astype('int32'), uop_1.astype('int32')) # shape=()
var_22 = relay.var("var_22", dtype = "float32", shape = ())#candidate|22|()|var|float32
bop_23 = relay.bitwise_xor(uop_1.astype('uint8'), var_22.astype('uint8')) # shape=()
output = relay.Tuple([bop_11,bop_14,uop_17,bop_19,bop_23,])
output2 = relay.Tuple([bop_11,bop_14,uop_17,bop_19,bop_23,])
func_26 = relay.Function([var_0,var_22,], output)
mod['func_26'] = func_26
mod = relay.transform.InferType()(mod)
mutated_mod['func_26'] = func_26
mutated_mod = relay.transform.InferType()(mutated_mod)
func_26_call = mutated_mod.get_global_var('func_26')
var_28 = relay.var("var_28", dtype = "float32", shape = ())#candidate|28|()|var|float32
var_29 = relay.var("var_29", dtype = "float32", shape = ())#candidate|29|()|var|float32
call_27 = func_26_call(var_28,var_29,)
output = call_27
func_30 = relay.Function([var_28,var_29,], output)
mutated_mod['func_30'] = func_30
mutated_mod = relay.transform.InferType()(mutated_mod)
var_32 = relay.var("var_32", dtype = "uint32", shape = ())#candidate|32|()|var|uint32
var_33 = relay.var("var_33", dtype = "uint32", shape = ())#candidate|33|()|var|uint32
bop_34 = relay.right_shift(var_32.astype('uint32'), var_33.astype('uint32')) # shape=()
var_37 = relay.var("var_37", dtype = "uint32", shape = (3, 3))#candidate|37|(3, 3)|var|uint32
bop_38 = relay.floor_mod(var_33.astype('float64'), var_37.astype('float64')) # shape=(3, 3)
var_41 = relay.var("var_41", dtype = "float64", shape = (3, 3))#candidate|41|(3, 3)|var|float64
bop_42 = relay.not_equal(bop_38.astype('bool'), relay.reshape(var_41.astype('bool'), relay.shape_of(bop_38))) # shape=(3, 3)
uop_45 = relay.tan(bop_42.astype('float64')) # shape=(3, 3)
uop_47 = relay.cosh(uop_45.astype('float64')) # shape=(3, 3)
var_49 = relay.var("var_49", dtype = "uint32", shape = (6,))#candidate|49|(6,)|var|uint32
bop_50 = relay.bitwise_xor(bop_34.astype('uint64'), var_49.astype('uint64')) # shape=(6,)
bop_53 = relay.equal(uop_47.astype('bool'), relay.reshape(var_37.astype('bool'), relay.shape_of(uop_47))) # shape=(3, 3)
bop_56 = relay.floor_mod(var_32.astype('float32'), var_49.astype('float32')) # shape=(6,)
uop_59 = relay.cosh(bop_53.astype('float32')) # shape=(3, 3)
uop_61 = relay.sinh(bop_53.astype('float32')) # shape=(3, 3)
var_63 = relay.var("var_63", dtype = "float64", shape = (3, 3))#candidate|63|(3, 3)|var|float64
bop_64 = relay.divide(uop_47.astype('float64'), relay.reshape(var_63.astype('float64'), relay.shape_of(uop_47))) # shape=(3, 3)
var_67 = relay.var("var_67", dtype = "float32", shape = (3, 3))#candidate|67|(3, 3)|var|float32
bop_68 = relay.subtract(uop_61.astype('float64'), relay.reshape(var_67.astype('float64'), relay.shape_of(uop_61))) # shape=(3, 3)
uop_71 = relay.asin(bop_53.astype('float64')) # shape=(3, 3)
output = relay.Tuple([bop_50,bop_56,uop_59,bop_64,bop_68,uop_71,])
output2 = relay.Tuple([bop_50,bop_56,uop_59,bop_64,bop_68,uop_71,])
func_73 = relay.Function([var_32,var_33,var_37,var_41,var_49,var_63,var_67,], output)
mod['func_73'] = func_73
mod = relay.transform.InferType()(mod)
var_74 = relay.var("var_74", dtype = "uint32", shape = ())#candidate|74|()|var|uint32
var_75 = relay.var("var_75", dtype = "uint32", shape = ())#candidate|75|()|var|uint32
var_76 = relay.var("var_76", dtype = "uint32", shape = (3, 3))#candidate|76|(3, 3)|var|uint32
var_77 = relay.var("var_77", dtype = "float64", shape = (3, 3))#candidate|77|(3, 3)|var|float64
var_78 = relay.var("var_78", dtype = "uint32", shape = (6,))#candidate|78|(6,)|var|uint32
var_79 = relay.var("var_79", dtype = "float64", shape = (3, 3))#candidate|79|(3, 3)|var|float64
var_80 = relay.var("var_80", dtype = "float32", shape = (3, 3))#candidate|80|(3, 3)|var|float32
output = func_73(var_74,var_75,var_76,var_77,var_78,var_79,var_80,)
func_81 = relay.Function([var_74,var_75,var_76,var_77,var_78,var_79,var_80,], output)
mutated_mod['func_81'] = func_81
mutated_mod = relay.transform.InferType()(mutated_mod)
var_83 = relay.var("var_83", dtype = "int8", shape = ())#candidate|83|()|var|int8
var_84 = relay.var("var_84", dtype = "int8", shape = (4,))#candidate|84|(4,)|var|int8
bop_85 = relay.greater(var_83.astype('bool'), var_84.astype('bool')) # shape=(4,)
bop_88 = relay.equal(bop_85.astype('bool'), var_83.astype('bool')) # shape=(4,)
bop_91 = relay.greater(bop_88.astype('bool'), relay.reshape(bop_85.astype('bool'), relay.shape_of(bop_88))) # shape=(4,)
uop_94 = relay.exp(bop_88.astype('float32')) # shape=(4,)
var_96 = relay.var("var_96", dtype = "int8", shape = (4,))#candidate|96|(4,)|var|int8
bop_97 = relay.greater_equal(var_84.astype('bool'), relay.reshape(var_96.astype('bool'), relay.shape_of(var_84))) # shape=(4,)
uop_100 = relay.log10(uop_94.astype('float32')) # shape=(4,)
bop_102 = relay.add(uop_94.astype('uint32'), relay.reshape(uop_100.astype('uint32'), relay.shape_of(uop_94))) # shape=(4,)
uop_105 = relay.log2(uop_94.astype('float64')) # shape=(4,)
uop_107 = relay.log10(uop_94.astype('float64')) # shape=(4,)
bop_109 = relay.bitwise_and(uop_100.astype('uint8'), relay.reshape(bop_88.astype('uint8'), relay.shape_of(uop_100))) # shape=(4,)
uop_112 = relay.atan(bop_88.astype('float32')) # shape=(4,)
uop_114 = relay.sqrt(bop_102.astype('float64')) # shape=(4,)
var_116 = relay.var("var_116", dtype = "uint8", shape = (4,))#candidate|116|(4,)|var|uint8
bop_117 = relay.less(bop_109.astype('bool'), relay.reshape(var_116.astype('bool'), relay.shape_of(bop_109))) # shape=(4,)
var_120 = relay.var("var_120", dtype = "float32", shape = (4,))#candidate|120|(4,)|var|float32
bop_121 = relay.less_equal(uop_94.astype('bool'), relay.reshape(var_120.astype('bool'), relay.shape_of(uop_94))) # shape=(4,)
bop_124 = relay.bitwise_and(bop_117.astype('uint32'), relay.reshape(bop_97.astype('uint32'), relay.shape_of(bop_117))) # shape=(4,)
func_73_call = mod.get_global_var('func_73')
func_81_call = mutated_mod.get_global_var('func_81')
const_128 = relay.const([4,-10,-1,-1,6,1,-5,5,1], dtype = "uint32")#candidate|128|(9,)|const|uint32
const_129 = relay.const([-9,9,-9,-1,-6,-5], dtype = "uint32")#candidate|129|(6,)|const|uint32
call_127 = relay.TupleGetItem(func_73_call(relay.reshape(var_83.astype('uint32'), []), relay.reshape(var_83.astype('uint32'), []), relay.reshape(const_128.astype('uint32'), [3, 3]), relay.reshape(const_128.astype('float64'), [3, 3]), relay.reshape(const_129.astype('uint32'), [6,]), relay.reshape(const_128.astype('float64'), [3, 3]), relay.reshape(const_128.astype('float32'), [3, 3]), ), 1)
call_130 = relay.TupleGetItem(func_81_call(relay.reshape(var_83.astype('uint32'), []), relay.reshape(var_83.astype('uint32'), []), relay.reshape(const_128.astype('uint32'), [3, 3]), relay.reshape(const_128.astype('float64'), [3, 3]), relay.reshape(const_129.astype('uint32'), [6,]), relay.reshape(const_128.astype('float64'), [3, 3]), relay.reshape(const_128.astype('float32'), [3, 3]), ), 1)
var_131 = relay.var("var_131", dtype = "float64", shape = (4,))#candidate|131|(4,)|var|float64
bop_132 = relay.less(uop_114.astype('bool'), relay.reshape(var_131.astype('bool'), relay.shape_of(uop_114))) # shape=(4,)
var_135 = relay.var("var_135", dtype = "bool", shape = (4,))#candidate|135|(4,)|var|bool
bop_136 = relay.bitwise_or(bop_117.astype('int32'), relay.reshape(var_135.astype('int32'), relay.shape_of(bop_117))) # shape=(4,)
bop_139 = relay.floor_divide(bop_132.astype('float64'), relay.reshape(bop_88.astype('float64'), relay.shape_of(bop_132))) # shape=(4,)
output = relay.Tuple([bop_91,uop_105,uop_107,uop_112,bop_121,bop_124,call_127,const_128,const_129,bop_136,bop_139,])
output2 = relay.Tuple([bop_91,uop_105,uop_107,uop_112,bop_121,bop_124,call_130,const_128,const_129,bop_136,bop_139,])
func_142 = relay.Function([var_83,var_84,var_96,var_116,var_120,var_131,var_135,], output)
mod['func_142'] = func_142
mod = relay.transform.InferType()(mod)
mutated_mod['func_142'] = func_142
mutated_mod = relay.transform.InferType()(mutated_mod)
func_142_call = mutated_mod.get_global_var('func_142')
var_144 = relay.var("var_144", dtype = "int8", shape = ())#candidate|144|()|var|int8
var_145 = relay.var("var_145", dtype = "int8", shape = (4,))#candidate|145|(4,)|var|int8
var_146 = relay.var("var_146", dtype = "int8", shape = (4,))#candidate|146|(4,)|var|int8
var_147 = relay.var("var_147", dtype = "uint8", shape = (4,))#candidate|147|(4,)|var|uint8
var_148 = relay.var("var_148", dtype = "float32", shape = (4,))#candidate|148|(4,)|var|float32
var_149 = relay.var("var_149", dtype = "float64", shape = (4,))#candidate|149|(4,)|var|float64
var_150 = relay.var("var_150", dtype = "bool", shape = (4,))#candidate|150|(4,)|var|bool
call_143 = func_142_call(var_144,var_145,var_146,var_147,var_148,var_149,var_150,)
output = call_143
func_151 = relay.Function([var_144,var_145,var_146,var_147,var_148,var_149,var_150,], output)
mutated_mod['func_151'] = func_151
mutated_mod = relay.transform.InferType()(mutated_mod)
var_153 = relay.var("var_153", dtype = "float64", shape = (1, 8))#candidate|153|(1, 8)|var|float64
uop_154 = relay.log10(var_153.astype('float64')) # shape=(1, 8)
var_156 = relay.var("var_156", dtype = "float64", shape = (7, 8))#candidate|156|(7, 8)|var|float64
bop_157 = relay.multiply(var_153.astype('int32'), var_156.astype('int32')) # shape=(7, 8)
uop_160 = relay.log(bop_157.astype('float64')) # shape=(7, 8)
uop_162 = relay.sigmoid(var_153.astype('float64')) # shape=(1, 8)
uop_164 = relay.cosh(uop_154.astype('float64')) # shape=(1, 8)
uop_166 = relay.sqrt(uop_164.astype('float64')) # shape=(1, 8)
uop_168 = relay.cos(uop_164.astype('float32')) # shape=(1, 8)
uop_170 = relay.atanh(uop_166.astype('float64')) # shape=(1, 8)
bop_172 = relay.equal(uop_168.astype('bool'), relay.reshape(uop_154.astype('bool'), relay.shape_of(uop_168))) # shape=(1, 8)
uop_175 = relay.atan(uop_170.astype('float64')) # shape=(1, 8)
uop_177 = relay.sin(uop_175.astype('float64')) # shape=(1, 8)
uop_179 = relay.sinh(uop_177.astype('float64')) # shape=(1, 8)
uop_181 = relay.acos(uop_177.astype('float64')) # shape=(1, 8)
uop_183 = relay.exp(uop_177.astype('float64')) # shape=(1, 8)
func_26_call = mod.get_global_var('func_26')
func_30_call = mutated_mod.get_global_var('func_30')
var_186 = relay.var("var_186", dtype = "float32", shape = ())#candidate|186|()|var|float32
call_185 = relay.TupleGetItem(func_26_call(relay.reshape(var_186.astype('float32'), []), relay.reshape(var_186.astype('float32'), []), ), 2)
call_187 = relay.TupleGetItem(func_30_call(relay.reshape(var_186.astype('float32'), []), relay.reshape(var_186.astype('float32'), []), ), 2)
uop_188 = relay.erf(uop_181.astype('float32')) # shape=(1, 8)
bop_190 = relay.not_equal(uop_179.astype('bool'), relay.reshape(uop_177.astype('bool'), relay.shape_of(uop_179))) # shape=(1, 8)
uop_193 = relay.acosh(uop_188.astype('float32')) # shape=(1, 8)
var_195 = relay.var("var_195", dtype = "float32", shape = (9, 8))#candidate|195|(9, 8)|var|float32
bop_196 = relay.equal(uop_193.astype('bool'), var_195.astype('bool')) # shape=(9, 8)
uop_199 = relay.sigmoid(uop_188.astype('float64')) # shape=(1, 8)
bop_201 = relay.power(uop_193.astype('float32'), bop_196.astype('float32')) # shape=(9, 8)
bop_204 = relay.divide(bop_196.astype('float64'), uop_199.astype('float64')) # shape=(9, 8)
bop_207 = relay.divide(uop_188.astype('float32'), var_186.astype('float32')) # shape=(1, 8)
var_210 = relay.var("var_210", dtype = "float64", shape = (13, 8))#candidate|210|(13, 8)|var|float64
bop_211 = relay.subtract(uop_175.astype('int32'), var_210.astype('int32')) # shape=(13, 8)
uop_214 = relay.acos(bop_196.astype('float64')) # shape=(9, 8)
bop_216 = relay.maximum(uop_214.astype('uint8'), relay.reshape(var_195.astype('uint8'), relay.shape_of(uop_214))) # shape=(9, 8)
bop_219 = relay.floor_mod(bop_216.astype('float64'), relay.reshape(bop_204.astype('float64'), relay.shape_of(bop_216))) # shape=(9, 8)
output = relay.Tuple([uop_160,uop_162,bop_172,uop_183,call_185,bop_190,bop_201,bop_207,bop_211,bop_219,])
output2 = relay.Tuple([uop_160,uop_162,bop_172,uop_183,call_187,bop_190,bop_201,bop_207,bop_211,bop_219,])
func_222 = relay.Function([var_153,var_156,var_186,var_195,var_210,], output)
mod['func_222'] = func_222
mod = relay.transform.InferType()(mod)
var_223 = relay.var("var_223", dtype = "float64", shape = (1, 8))#candidate|223|(1, 8)|var|float64
var_224 = relay.var("var_224", dtype = "float64", shape = (7, 8))#candidate|224|(7, 8)|var|float64
var_225 = relay.var("var_225", dtype = "float32", shape = ())#candidate|225|()|var|float32
var_226 = relay.var("var_226", dtype = "float32", shape = (9, 8))#candidate|226|(9, 8)|var|float32
var_227 = relay.var("var_227", dtype = "float64", shape = (13, 8))#candidate|227|(13, 8)|var|float64
output = func_222(var_223,var_224,var_225,var_226,var_227,)
func_228 = relay.Function([var_223,var_224,var_225,var_226,var_227,], output)
mutated_mod['func_228'] = func_228
mutated_mod = relay.transform.InferType()(mutated_mod)
const_230 = relay.const([[False,False,True,False],[True,True,False,True],[False,True,False,True],[True,False,False,True],[False,False,False,False],[True,True,True,False],[False,False,False,True],[False,True,True,True],[True,False,True,False],[True,False,False,True],[True,False,True,False]], dtype = "bool")#candidate|230|(11, 4)|const|bool
var_231 = relay.var("var_231", dtype = "bool", shape = (11, 4))#candidate|231|(11, 4)|var|bool
bop_232 = relay.logical_or(const_230.astype('bool'), relay.reshape(var_231.astype('bool'), relay.shape_of(const_230))) # shape=(11, 4)
bop_235 = relay.logical_xor(var_231.astype('uint16'), relay.reshape(const_230.astype('uint16'), relay.shape_of(var_231))) # shape=(11, 4)
bop_238 = relay.less(var_231.astype('bool'), relay.reshape(bop_232.astype('bool'), relay.shape_of(var_231))) # shape=(11, 4)
bop_241 = relay.maximum(bop_232.astype('uint16'), relay.reshape(var_231.astype('uint16'), relay.shape_of(bop_232))) # shape=(11, 4)
bop_244 = relay.less(const_230.astype('bool'), relay.reshape(bop_238.astype('bool'), relay.shape_of(const_230))) # shape=(11, 4)
const_247 = relay.const([[True,False,True,True],[True,True,True,False],[True,False,True,False],[False,True,True,False],[False,True,True,True],[False,False,True,False],[False,False,False,False],[True,True,False,False],[True,True,True,False],[True,True,False,False],[True,True,False,True]], dtype = "bool")#candidate|247|(11, 4)|const|bool
bop_248 = relay.mod(bop_244.astype('float64'), relay.reshape(const_247.astype('float64'), relay.shape_of(bop_244))) # shape=(11, 4)
uop_251 = relay.log2(const_247.astype('float32')) # shape=(11, 4)
uop_253 = relay.erf(uop_251.astype('float64')) # shape=(11, 4)
output = relay.Tuple([bop_235,bop_241,bop_248,uop_253,])
output2 = relay.Tuple([bop_235,bop_241,bop_248,uop_253,])
func_255 = relay.Function([var_231,], output)
mod['func_255'] = func_255
mod = relay.transform.InferType()(mod)
var_256 = relay.var("var_256", dtype = "bool", shape = (11, 4))#candidate|256|(11, 4)|var|bool
output = func_255(var_256)
func_257 = relay.Function([var_256], output)
mutated_mod['func_257'] = func_257
mutated_mod = relay.transform.InferType()(mutated_mod)
var_259 = relay.var("var_259", dtype = "uint8", shape = (10,))#candidate|259|(10,)|var|uint8
const_260 = relay.const([10,-2,5,-1,1,9,-8,1,-10,-6], dtype = "uint8")#candidate|260|(10,)|const|uint8
bop_261 = relay.subtract(var_259.astype('uint8'), relay.reshape(const_260.astype('uint8'), relay.shape_of(var_259))) # shape=(10,)
uop_264 = relay.asinh(bop_261.astype('float64')) # shape=(10,)
uop_266 = relay.asin(uop_264.astype('float32')) # shape=(10,)
const_268 = relay.const([-0.766136,5.530027,0.223214,-4.013060,-6.355907,-4.824557,-2.790051,-9.677941,-6.708141,0.669404], dtype = "float32")#candidate|268|(10,)|const|float32
bop_269 = relay.divide(uop_266.astype('float64'), relay.reshape(const_268.astype('float64'), relay.shape_of(uop_266))) # shape=(10,)
bop_272 = relay.bitwise_xor(uop_264.astype('uint64'), relay.reshape(var_259.astype('uint64'), relay.shape_of(uop_264))) # shape=(10,)
var_275 = relay.var("var_275", dtype = "float32", shape = (10,))#candidate|275|(10,)|var|float32
bop_276 = relay.power(uop_266.astype('float32'), relay.reshape(var_275.astype('float32'), relay.shape_of(uop_266))) # shape=(10,)
var_279 = relay.var("var_279", dtype = "float32", shape = (10,))#candidate|279|(10,)|var|float32
bop_280 = relay.power(bop_276.astype('float64'), relay.reshape(var_279.astype('float64'), relay.shape_of(bop_276))) # shape=(10,)
bop_283 = relay.mod(bop_269.astype('float64'), relay.reshape(var_279.astype('float64'), relay.shape_of(bop_269))) # shape=(10,)
bop_286 = relay.less_equal(bop_272.astype('bool'), relay.reshape(uop_264.astype('bool'), relay.shape_of(bop_272))) # shape=(10,)
bop_289 = relay.maximum(const_268.astype('uint64'), relay.reshape(bop_280.astype('uint64'), relay.shape_of(const_268))) # shape=(10,)
bop_292 = relay.bitwise_xor(bop_280.astype('int32'), relay.reshape(bop_286.astype('int32'), relay.shape_of(bop_280))) # shape=(10,)
uop_295 = relay.cos(const_260.astype('float64')) # shape=(10,)
var_297 = relay.var("var_297", dtype = "float64", shape = (10,))#candidate|297|(10,)|var|float64
bop_298 = relay.add(bop_269.astype('uint32'), relay.reshape(var_297.astype('uint32'), relay.shape_of(bop_269))) # shape=(10,)
uop_301 = relay.atan(bop_289.astype('float64')) # shape=(10,)
uop_303 = relay.tan(uop_266.astype('float32')) # shape=(10,)
uop_305 = relay.tan(uop_303.astype('float64')) # shape=(10,)
var_307 = relay.var("var_307", dtype = "float64", shape = (10,))#candidate|307|(10,)|var|float64
bop_308 = relay.logical_or(bop_280.astype('bool'), relay.reshape(var_307.astype('bool'), relay.shape_of(bop_280))) # shape=(10,)
bop_311 = relay.minimum(uop_305.astype('int32'), relay.reshape(uop_301.astype('int32'), relay.shape_of(uop_305))) # shape=(10,)
uop_314 = relay.log10(bop_311.astype('float32')) # shape=(10,)
const_316 = relay.const([1.928093,-2.842997,1.864258,-8.941345,8.187704,-0.757758,8.335031,7.049881,7.732538,-7.622886], dtype = "float32")#candidate|316|(10,)|const|float32
bop_317 = relay.bitwise_and(uop_314.astype('uint64'), relay.reshape(const_316.astype('uint64'), relay.shape_of(uop_314))) # shape=(10,)
var_320 = relay.var("var_320", dtype = "float32", shape = (10,))#candidate|320|(10,)|var|float32
bop_321 = relay.logical_or(uop_314.astype('bool'), relay.reshape(var_320.astype('bool'), relay.shape_of(uop_314))) # shape=(10,)
func_26_call = mod.get_global_var('func_26')
func_30_call = mutated_mod.get_global_var('func_30')
const_325 = relay.const(-9.858002, dtype = "float32")#candidate|325|()|const|float32
call_324 = relay.TupleGetItem(func_26_call(relay.reshape(const_325.astype('float32'), []), relay.reshape(const_325.astype('float32'), []), ), 3)
call_326 = relay.TupleGetItem(func_30_call(relay.reshape(const_325.astype('float32'), []), relay.reshape(const_325.astype('float32'), []), ), 3)
const_327 = relay.const([-2,-1,9,-9,-5,4,-2,1,-10,-3], dtype = "uint64")#candidate|327|(10,)|const|uint64
bop_328 = relay.left_shift(bop_317.astype('int64'), relay.reshape(const_327.astype('int64'), relay.shape_of(bop_317))) # shape=(10,)
func_73_call = mod.get_global_var('func_73')
func_81_call = mutated_mod.get_global_var('func_81')
const_332 = relay.const([-2,-3,6,3,9,-9,-5,4,4], dtype = "uint32")#candidate|332|(9,)|const|uint32
var_333 = relay.var("var_333", dtype = "uint32", shape = (3, 2))#candidate|333|(3, 2)|var|uint32
call_331 = relay.TupleGetItem(func_73_call(relay.reshape(const_325.astype('uint32'), []), relay.reshape(call_324.astype('uint32'), []), relay.reshape(const_332.astype('uint32'), [3, 3]), relay.reshape(const_332.astype('float64'), [3, 3]), relay.reshape(var_333.astype('uint32'), [6,]), relay.reshape(const_332.astype('float64'), [3, 3]), relay.reshape(const_332.astype('float32'), [3, 3]), ), 5)
call_334 = relay.TupleGetItem(func_81_call(relay.reshape(const_325.astype('uint32'), []), relay.reshape(call_324.astype('uint32'), []), relay.reshape(const_332.astype('uint32'), [3, 3]), relay.reshape(const_332.astype('float64'), [3, 3]), relay.reshape(var_333.astype('uint32'), [6,]), relay.reshape(const_332.astype('float64'), [3, 3]), relay.reshape(const_332.astype('float32'), [3, 3]), ), 5)
output = relay.Tuple([bop_283,bop_292,uop_295,bop_298,bop_308,bop_321,call_324,const_325,bop_328,call_331,const_332,var_333,])
output2 = relay.Tuple([bop_283,bop_292,uop_295,bop_298,bop_308,bop_321,call_326,const_325,bop_328,call_334,const_332,var_333,])
func_335 = relay.Function([var_259,var_275,var_279,var_297,var_307,var_320,var_333,], output)
mod['func_335'] = func_335
mod = relay.transform.InferType()(mod)
mutated_mod['func_335'] = func_335
mutated_mod = relay.transform.InferType()(mutated_mod)
func_335_call = mutated_mod.get_global_var('func_335')
var_337 = relay.var("var_337", dtype = "uint8", shape = (10,))#candidate|337|(10,)|var|uint8
var_338 = relay.var("var_338", dtype = "float32", shape = (10,))#candidate|338|(10,)|var|float32
var_339 = relay.var("var_339", dtype = "float32", shape = (10,))#candidate|339|(10,)|var|float32
var_340 = relay.var("var_340", dtype = "float64", shape = (10,))#candidate|340|(10,)|var|float64
var_341 = relay.var("var_341", dtype = "float64", shape = (10,))#candidate|341|(10,)|var|float64
var_342 = relay.var("var_342", dtype = "float32", shape = (10,))#candidate|342|(10,)|var|float32
var_343 = relay.var("var_343", dtype = "uint32", shape = (3, 2))#candidate|343|(3, 2)|var|uint32
call_336 = func_335_call(var_337,var_338,var_339,var_340,var_341,var_342,var_343,)
output = call_336
func_344 = relay.Function([var_337,var_338,var_339,var_340,var_341,var_342,var_343,], output)
mutated_mod['func_344'] = func_344
mutated_mod = relay.transform.InferType()(mutated_mod)
const_346 = relay.const([[[-5.734164,6.722870,-7.836628,-1.992162,1.180073,-2.311732,-1.858623,7.274136,7.574574,5.633349,2.739142],[-7.398258,-0.948252,8.077631,4.265968,-6.747818,-1.028787,0.175289,-7.765083,-4.097707,7.148547,4.374870],[4.144711,8.145175,2.757443,2.762510,-5.402276,-2.587541,5.457446,-8.538071,-7.406271,-0.347715,-3.730249],[6.812294,3.195401,-7.662077,-8.567739,-4.165515,-5.189194,-2.426298,7.109279,-0.934306,-8.166922,-6.553844],[2.241479,-2.485129,-3.654075,-9.497494,-4.292869,8.042640,-5.819207,-1.031669,4.049803,-0.641584,-2.772886],[0.962872,-4.203820,8.780266,-1.749217,7.502080,5.855551,7.895028,-3.630114,-0.553073,0.256889,7.562116],[-3.083071,8.494286,-5.924696,5.045043,-1.019534,-0.957628,0.586879,0.699682,-3.369038,7.228404,8.861744]]], dtype = "float32")#candidate|346|(1, 7, 11)|const|float32
uop_347 = relay.acosh(const_346.astype('float32')) # shape=(1, 7, 11)
const_349 = relay.const([[[9.482505,7.830703,-0.491731,-4.037333,5.304618,-6.030291,-5.841746,8.442601,-0.296342,5.355422,-2.855381],[-7.506660,-2.119799,-9.766897,0.086787,4.606702,0.891141,6.504663,8.633242,-1.690810,-9.548184,9.296008],[-3.336819,-3.626422,-6.003955,1.237961,-2.128987,9.111905,-1.222312,-0.968378,9.515511,9.658615,-3.246698],[-4.158814,-8.281830,-8.744716,-2.803672,7.404330,3.438295,9.417776,9.400049,-3.533205,-6.577096,3.630751],[8.115921,-2.655005,8.509362,4.977141,-0.939200,9.817664,-7.176509,6.412979,0.506368,5.440298,-4.990858],[-4.571492,-9.945682,8.655037,-7.044216,8.075374,-9.535146,-7.658546,7.841213,-4.332757,-3.203591,-0.908486],[6.370706,5.044141,-0.364551,4.788576,5.887441,3.378771,-3.403806,-8.617602,0.031077,-2.872858,6.791803]],[[6.431307,4.999907,8.750140,6.397884,-2.328267,9.412343,5.064439,8.451504,1.086064,-1.163740,-4.943018],[5.246260,-9.696522,-0.807636,1.287212,7.490322,9.304974,-8.857188,-8.065041,-3.839031,-3.700544,-3.360280],[-3.522940,8.367545,-1.483714,-0.366021,-7.355211,4.986808,-0.827759,-4.979781,-1.157865,-1.344188,-3.449529],[-8.433761,-3.383623,0.646103,-3.528132,-7.038120,7.314550,-8.910969,5.818221,-5.226313,3.860046,-5.041870],[-0.955883,9.970163,0.770561,-5.904308,-9.675561,-2.350598,9.979962,0.589145,-9.089427,-4.279045,7.937071],[2.809213,9.506231,3.182928,-0.297716,-8.684868,-1.474948,5.423757,6.496903,7.024104,-1.954385,-5.097359],[4.898645,4.946495,2.145524,-8.388182,2.015842,-6.908564,9.376778,8.049842,9.851773,-5.284417,9.672586]],[[4.646164,6.356467,-8.332413,2.862881,-8.773695,-0.191128,-9.152324,3.962629,0.776827,-3.227774,-8.931461],[3.151747,1.327918,5.120341,-1.375873,0.333870,-3.098380,-4.715494,-9.960648,4.398680,-7.702768,3.259202],[-9.625623,7.090628,2.660933,2.725732,-2.573747,5.350606,7.275658,3.054170,7.040388,-3.139534,-4.670695],[-3.607740,-2.894767,0.328555,6.917682,6.155941,5.330092,3.361532,-6.106006,-7.483696,2.615683,4.680473],[1.807258,-7.457445,4.352645,-4.549218,6.980270,0.997233,-5.556942,1.721841,4.841106,0.127742,5.733321],[8.208317,7.757778,-6.649901,7.823024,-0.311067,9.282024,9.306987,0.426836,-9.669631,-1.968198,-2.767031],[-3.557489,-3.659219,-0.090268,-5.776240,0.406631,0.251895,-4.772542,7.369763,0.047781,-5.067737,1.090026]],[[-8.252250,1.024503,7.576734,-9.752889,-1.709732,6.476516,-1.630545,9.542632,-8.404826,-9.503721,8.834264],[6.433695,-4.861190,-2.994575,0.719386,-0.543519,-7.482965,-2.114545,5.540903,6.858352,9.124121,-0.858351],[-3.476970,4.107730,-5.889777,9.309827,-3.740787,2.595197,4.943460,9.688627,6.349956,-2.723523,9.543042],[1.011001,3.132526,-6.586842,-1.674236,-5.715736,-1.793087,5.462020,3.176129,6.209127,2.432180,-9.043503],[-4.441952,-0.091472,8.044399,-6.456080,7.532812,6.406842,-8.763819,-1.940601,7.471783,-9.474230,-6.082805],[-8.561032,-6.640899,7.869689,-0.741478,-4.840891,4.282369,-0.567572,-7.866436,2.838947,3.654418,-2.127799],[-2.338112,-8.740600,-8.216525,-9.805709,-9.791538,-4.253878,-7.602381,-8.666751,-0.632681,2.670043,8.701210]],[[-3.421995,-8.286096,-1.948592,6.046676,-4.393450,5.386310,4.611563,3.873407,-4.794642,-0.100022,-3.337421],[-5.183639,-4.184145,5.271142,-8.651150,6.580118,-7.289356,-8.520946,8.844819,-7.948293,8.043947,-1.264203],[6.912201,-3.448797,7.373874,-4.557839,4.664786,0.604731,2.598828,-4.955075,3.209696,1.752613,1.700399],[-3.731935,7.171609,-3.639910,-0.983259,8.508016,-7.235454,5.099619,-0.022595,4.860046,-1.313848,4.528171],[-4.226249,5.646290,-0.983962,1.756687,-7.746039,-9.755175,-6.051446,6.273368,-2.474427,-6.949659,-5.072553],[-8.062857,-8.487843,4.857585,-5.064166,-6.108176,-5.931435,2.902616,-6.526874,-2.475287,-9.513004,-0.895484],[-7.718391,-1.897995,-2.033342,7.591868,-0.483872,-2.800235,-8.867031,-9.804435,-6.334377,0.630239,-2.352295]],[[9.159090,2.453658,-8.157560,1.049597,7.612713,-6.484003,-9.807416,-5.684828,1.561639,1.785859,3.547508],[9.479939,7.319769,4.622559,-7.783823,8.911022,-8.226329,-8.160555,-9.809541,-3.370011,5.685279,2.317379],[2.801926,7.676554,3.511826,-3.859329,-4.347860,-9.412576,-4.641911,-6.060684,8.299269,-7.510154,9.922678],[-5.384595,-9.370262,1.960888,4.603528,5.129076,-0.731131,-3.103149,-4.310877,3.487921,-1.323564,-6.608674],[8.167959,-7.646293,-5.375064,-0.289971,4.171273,5.697263,4.663499,7.773620,-9.475343,-0.712954,2.755104],[-3.329921,8.497921,8.281555,1.914894,-9.269063,-4.217567,4.029393,-6.930384,-0.419836,6.197400,0.779881],[-6.366029,3.654339,-7.670764,-8.002632,-6.127848,8.440897,-6.571753,-8.645162,-4.493665,-5.023899,8.353453]],[[5.825784,2.232188,8.439503,6.576935,-2.407652,2.429787,-9.084742,5.451165,1.439043,1.942936,9.860976],[7.021228,1.115025,0.797337,9.462630,-0.798620,-4.156434,-8.583784,-7.841374,-3.853730,-8.021014,-4.625307],[4.389689,9.202673,-3.964770,5.671918,-9.803645,0.097829,-3.926945,1.802769,4.767946,-6.624657,5.577173],[6.000655,0.724574,-2.402348,2.037116,-8.885697,-1.574678,8.086865,8.889525,5.014295,-5.267202,2.230307],[6.763599,-3.135984,-6.310966,-2.638250,0.637353,0.256886,0.759443,-3.459498,6.339916,-2.195114,-0.809718],[-8.860107,-0.761101,2.262988,-4.192186,-2.985816,0.636645,-4.921123,4.233521,-5.482568,-5.093557,4.368825],[4.406959,-6.963686,7.654775,8.613769,-7.052816,3.464357,1.944761,8.678269,2.629215,8.546187,-5.941321]],[[2.870045,-7.224366,1.208316,1.733267,-4.612895,-9.113964,8.772425,1.567886,-3.984234,-6.723177,5.651557],[-3.450201,2.855536,-1.076508,5.039993,-8.430785,-3.043047,-5.097595,5.391381,6.575589,-2.044204,-4.661555],[4.328699,-9.214643,-0.165995,0.933803,-3.541068,-2.342631,1.808288,-0.337903,-0.151706,8.815384,0.546185],[-8.817736,-0.534930,0.960349,4.917927,-6.998261,0.254190,-5.599167,-7.502855,8.746891,-8.261841,-5.695013],[-0.139090,-0.369880,-6.841033,-2.752822,8.857481,-4.243435,-8.374078,-7.560993,-0.663347,-7.595482,-7.117045],[5.662269,-6.244324,0.192199,-0.269230,-0.053173,3.914588,8.095854,8.719452,6.046596,-6.338289,-8.002501],[0.293878,8.856692,0.642443,-1.202688,-3.386777,-6.294285,-7.397560,4.953813,-5.072002,-1.051106,-0.588341]],[[4.510433,8.474877,-3.622641,-5.795068,-3.729419,-9.399729,1.561626,9.762716,3.093501,5.873532,-5.670943],[2.870550,6.807809,0.695441,7.771502,5.391258,8.432058,-2.291040,1.748764,-3.133498,-8.879886,9.676211],[3.774426,6.565850,-3.526666,-1.702620,1.460905,0.341955,0.645556,-1.019421,-0.365497,7.163175,-0.693486],[-6.173716,8.556021,3.029989,3.715946,3.374419,3.574797,3.971014,-5.648695,-2.088251,7.738641,-0.872979],[-5.668781,-5.598558,-0.945785,9.556855,7.836394,-1.216083,-1.057964,-9.177716,-3.186619,-0.213160,3.102858],[-9.742771,-1.885646,0.824276,-6.256228,-1.513606,-0.875213,9.802427,4.843105,0.704369,-6.526573,0.180235],[7.367586,-3.726507,1.173214,1.377287,-5.221133,-4.897504,-8.781718,3.603300,1.092997,8.886095,-7.618678]],[[2.595895,6.226241,-4.271335,-2.668708,3.610808,8.102968,-4.093766,-7.572537,6.382081,1.123013,6.088414],[7.377708,-7.371677,1.893563,-9.512992,-8.884187,9.707269,-8.210850,-8.111195,3.190827,-5.216086,5.203308],[2.767492,-3.166841,2.512771,3.724735,9.797052,2.245400,8.566300,-7.523805,-5.179107,0.722887,-9.393097],[8.428813,4.034474,-1.373467,2.945467,-2.618963,-2.015140,-0.436922,-0.132783,-2.550387,-3.653764,0.781552],[-3.901057,-5.346565,-6.343777,-9.805771,3.970577,6.638353,-9.090416,5.929916,-5.467106,7.150396,-5.258594],[-2.896725,5.235294,1.425714,-4.084707,7.802538,-7.524933,0.573210,-0.918553,7.663624,3.054331,-9.699234],[7.543047,5.471845,9.784941,-4.692270,-8.998707,-7.021493,7.436190,-9.670895,-4.893500,2.282973,1.769518]],[[3.385199,-5.687045,4.486494,-2.895807,-8.912362,9.895554,6.566295,-4.029004,6.326006,-8.543511,-4.212415],[2.156731,9.593928,-7.579396,-8.756616,2.931545,4.953584,-2.761366,9.321149,-4.461912,-2.984541,7.034659],[-3.746282,0.216884,8.374203,8.235840,-0.596154,-8.079649,-9.937969,-8.948085,-0.408757,9.344227,9.121376],[-5.696465,-3.387468,-0.419263,7.063163,-5.927634,5.478509,1.464289,7.172184,-1.179662,5.986006,9.474896],[-5.727657,-9.810199,8.217669,-4.298415,0.962026,5.174116,-6.618384,4.189427,8.367723,4.166646,0.943304],[4.097962,-6.194363,-9.589676,-8.608976,4.204534,4.117910,9.489853,2.769240,-2.802273,0.951596,-1.401665],[8.765770,0.563655,3.880792,-2.524399,5.706998,2.071808,7.886971,6.639433,7.243713,-2.108388,-5.613533]],[[2.457144,-5.890323,-0.246654,8.543155,0.602015,-0.642984,-5.097267,1.968440,5.955659,-1.932404,-8.486169],[-7.209491,-9.396822,8.289780,1.015455,-3.902588,6.620261,-3.676049,-3.692752,7.443107,-6.984209,-5.189904],[4.374768,3.021899,-8.301143,-8.700091,-7.528864,-6.723882,-9.010663,2.707810,6.190660,3.334207,-5.868581],[0.201640,5.707373,-4.826921,-8.300434,-1.396545,5.638926,-8.847457,-2.652026,-7.442053,-5.200308,7.645854],[-9.024427,-7.644417,-1.393009,6.693335,4.028547,-5.810131,7.876928,-4.329144,0.507723,4.123803,3.989881],[1.290762,-2.173545,-9.771665,-6.793864,-2.077072,-4.647129,6.387602,-2.848313,-5.266398,7.369619,-7.915287],[3.224822,-8.776710,-4.088765,4.952252,-5.972505,8.212574,4.541569,2.307772,1.288172,-6.831699,-6.435321]],[[-5.299957,-4.334974,-6.635379,5.009334,-7.876511,1.643807,-8.940673,8.885087,2.884459,-0.294178,-0.121194],[-7.021628,8.833665,5.058569,-7.774169,8.442873,-0.775760,2.045375,2.475327,-1.707807,5.030928,5.093455],[4.305383,-1.119853,-4.020960,1.470327,-3.321674,-6.936741,3.430055,9.213904,3.999645,-8.373563,3.251502],[-0.264096,0.725798,-2.935850,-6.343650,-7.447558,-4.882137,6.124468,4.054023,-7.664633,-8.776476,1.346667],[-7.993509,4.981791,-9.339101,-4.762824,9.366831,-7.642102,4.674891,-3.298140,-3.964030,1.910647,3.399455],[8.253549,7.492102,-9.600659,-8.714372,7.819620,-4.313544,-4.888896,-7.941325,-2.000474,8.312991,-8.611249],[-5.826167,8.482209,6.656968,5.289777,8.842440,0.978150,-1.405771,-3.767877,-9.358396,-7.657925,7.587706]]], dtype = "float32")#candidate|349|(13, 7, 11)|const|float32
bop_350 = relay.multiply(uop_347.astype('uint8'), const_349.astype('uint8')) # shape=(13, 7, 11)
bop_353 = relay.floor_mod(const_346.astype('float64'), bop_350.astype('float64')) # shape=(13, 7, 11)
output = relay.Tuple([bop_353,])
output2 = relay.Tuple([bop_353,])
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
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()()
res3 = intrp3.evaluate()()
res4 = intrp4.evaluate()()
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
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()()
res7 = intrp7.evaluate()()
res8 = intrp8.evaluate()()
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
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()()
res11 = intrp11.evaluate()()
res12 = intrp12.evaluate()()
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
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()()
res15 = intrp15.evaluate()()
res16 = intrp16.evaluate()()
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
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()()
res19 = intrp19.evaluate()()
res20 = intrp20.evaluate()()
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
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()()
res23 = intrp23.evaluate()()
res24 = intrp24.evaluate()()
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

'''2.739142],

'''