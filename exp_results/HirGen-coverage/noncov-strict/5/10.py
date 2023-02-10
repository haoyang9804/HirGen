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
var_0 = relay.var("var_0", dtype = "float32", shape = (1,))#candidate|0|(1,)|var|float32
var_1 = relay.var("var_1", dtype = "float32", shape = (2,))#candidate|1|(2,)|var|float32
bop_2 = relay.divide(var_0.astype('float32'), var_1.astype('float32')) # shape=(2,)
var_5 = relay.var("var_5", dtype = "float32", shape = (2,))#candidate|5|(2,)|var|float32
bop_6 = relay.logical_and(var_1.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(var_1))) # shape=(2,)
bop_9 = relay.right_shift(bop_2.astype('uint8'), var_0.astype('uint8')) # shape=(2,)
uop_12 = relay.log10(bop_2.astype('float32')) # shape=(2,)
var_14 = relay.var("var_14", dtype = "float32", shape = (2,))#candidate|14|(2,)|var|float32
bop_15 = relay.floor_mod(uop_12.astype('float32'), relay.reshape(var_14.astype('float32'), relay.shape_of(uop_12))) # shape=(2,)
var_18 = relay.var("var_18", dtype = "float32", shape = (2,))#candidate|18|(2,)|var|float32
bop_19 = relay.mod(bop_15.astype('float32'), relay.reshape(var_18.astype('float32'), relay.shape_of(bop_15))) # shape=(2,)
bop_22 = relay.right_shift(uop_12.astype('int8'), relay.reshape(var_18.astype('int8'), relay.shape_of(uop_12))) # shape=(2,)
uop_25 = relay.log2(bop_22.astype('float32')) # shape=(2,)
bop_27 = relay.greater_equal(uop_25.astype('bool'), relay.reshape(bop_9.astype('bool'), relay.shape_of(uop_25))) # shape=(2,)
uop_30 = relay.acosh(uop_25.astype('float64')) # shape=(2,)
bop_32 = relay.maximum(uop_30.astype('uint8'), relay.reshape(uop_12.astype('uint8'), relay.shape_of(uop_30))) # shape=(2,)
uop_35 = relay.erf(uop_30.astype('float64')) # shape=(2,)
var_37 = relay.var("var_37", dtype = "uint8", shape = (2,))#candidate|37|(2,)|var|uint8
bop_38 = relay.minimum(bop_32.astype('int32'), relay.reshape(var_37.astype('int32'), relay.shape_of(bop_32))) # shape=(2,)
bop_41 = relay.logical_xor(uop_35.astype('uint8'), relay.reshape(var_5.astype('uint8'), relay.shape_of(uop_35))) # shape=(2,)
bop_44 = relay.logical_xor(bop_41.astype('int16'), relay.reshape(bop_15.astype('int16'), relay.shape_of(bop_41))) # shape=(2,)
uop_47 = relay.sinh(bop_44.astype('float32')) # shape=(2,)
const_49 = relay.const([5.148185,-0.299486], dtype = "float32")#candidate|49|(2,)|const|float32
bop_50 = relay.floor_divide(uop_47.astype('float32'), relay.reshape(const_49.astype('float32'), relay.shape_of(uop_47))) # shape=(2,)
bop_53 = relay.bitwise_and(uop_47.astype('uint16'), relay.reshape(bop_15.astype('uint16'), relay.shape_of(uop_47))) # shape=(2,)
uop_56 = relay.exp(uop_47.astype('float32')) # shape=(2,)
uop_58 = relay.acos(uop_56.astype('float64')) # shape=(2,)
bop_60 = relay.not_equal(uop_58.astype('bool'), relay.reshape(var_37.astype('bool'), relay.shape_of(uop_58))) # shape=(2,)
uop_63 = relay.acos(bop_60.astype('float32')) # shape=(2,)
bop_65 = relay.floor_divide(uop_63.astype('float64'), relay.reshape(bop_9.astype('float64'), relay.shape_of(uop_63))) # shape=(2,)
bop_68 = relay.greater(uop_56.astype('bool'), relay.reshape(bop_41.astype('bool'), relay.shape_of(uop_56))) # shape=(2,)
var_71 = relay.var("var_71", dtype = "uint8", shape = (2,))#candidate|71|(2,)|var|uint8
bop_72 = relay.not_equal(bop_41.astype('bool'), relay.reshape(var_71.astype('bool'), relay.shape_of(bop_41))) # shape=(2,)
const_75 = relay.const([False,False], dtype = "bool")#candidate|75|(2,)|const|bool
bop_76 = relay.bitwise_xor(bop_60.astype('int64'), relay.reshape(const_75.astype('int64'), relay.shape_of(bop_60))) # shape=(2,)
uop_79 = relay.asin(bop_60.astype('float32')) # shape=(2,)
var_81 = relay.var("var_81", dtype = "float64", shape = (2,))#candidate|81|(2,)|var|float64
bop_82 = relay.less(bop_65.astype('bool'), relay.reshape(var_81.astype('bool'), relay.shape_of(bop_65))) # shape=(2,)
bop_85 = relay.logical_xor(bop_50.astype('uint8'), relay.reshape(bop_68.astype('uint8'), relay.shape_of(bop_50))) # shape=(2,)
bop_88 = relay.multiply(uop_63.astype('uint64'), relay.reshape(bop_68.astype('uint64'), relay.shape_of(uop_63))) # shape=(2,)
bop_91 = relay.not_equal(bop_88.astype('bool'), relay.reshape(bop_38.astype('bool'), relay.shape_of(bop_88))) # shape=(2,)
uop_94 = relay.log(bop_50.astype('float64')) # shape=(2,)
bop_96 = relay.less_equal(bop_60.astype('bool'), relay.reshape(var_37.astype('bool'), relay.shape_of(bop_60))) # shape=(2,)
var_99 = relay.var("var_99", dtype = "float32", shape = (2,))#candidate|99|(2,)|var|float32
bop_100 = relay.greater_equal(uop_63.astype('bool'), relay.reshape(var_99.astype('bool'), relay.shape_of(uop_63))) # shape=(2,)
var_103 = relay.var("var_103", dtype = "float32", shape = (2,))#candidate|103|(2,)|var|float32
bop_104 = relay.maximum(uop_63.astype('float64'), relay.reshape(var_103.astype('float64'), relay.shape_of(uop_63))) # shape=(2,)
var_107 = relay.var("var_107", dtype = "bool", shape = (2,))#candidate|107|(2,)|var|bool
bop_108 = relay.add(bop_96.astype('uint8'), relay.reshape(var_107.astype('uint8'), relay.shape_of(bop_96))) # shape=(2,)
bop_111 = relay.floor_divide(bop_108.astype('float64'), relay.reshape(bop_27.astype('float64'), relay.shape_of(bop_108))) # shape=(2,)
uop_114 = relay.sinh(bop_65.astype('float64')) # shape=(2,)
var_116 = relay.var("var_116", dtype = "bool", shape = (2,))#candidate|116|(2,)|var|bool
bop_117 = relay.logical_xor(bop_100.astype('uint32'), relay.reshape(var_116.astype('uint32'), relay.shape_of(bop_100))) # shape=(2,)
uop_120 = relay.atanh(bop_50.astype('float32')) # shape=(2,)
bop_122 = relay.multiply(uop_114.astype('float64'), relay.reshape(bop_44.astype('float64'), relay.shape_of(uop_114))) # shape=(2,)
uop_125 = relay.sqrt(bop_50.astype('float64')) # shape=(2,)
uop_127 = relay.sinh(bop_122.astype('float32')) # shape=(2,)
bop_129 = relay.left_shift(uop_127.astype('uint16'), relay.reshape(bop_96.astype('uint16'), relay.shape_of(uop_127))) # shape=(2,)
var_132 = relay.var("var_132", dtype = "uint16", shape = (2,))#candidate|132|(2,)|var|uint16
bop_133 = relay.minimum(bop_129.astype('uint32'), relay.reshape(var_132.astype('uint32'), relay.shape_of(bop_129))) # shape=(2,)
output = relay.Tuple([bop_6,bop_19,bop_53,bop_72,bop_76,uop_79,bop_82,bop_85,bop_91,uop_94,bop_104,bop_111,bop_117,uop_120,uop_125,bop_133,])
output2 = relay.Tuple([bop_6,bop_19,bop_53,bop_72,bop_76,uop_79,bop_82,bop_85,bop_91,uop_94,bop_104,bop_111,bop_117,uop_120,uop_125,bop_133,])
func_136 = relay.Function([var_0,var_1,var_5,var_14,var_18,var_37,var_71,var_81,var_99,var_103,var_107,var_116,var_132,], output)
mod['func_136'] = func_136
mod = relay.transform.InferType()(mod)
var_137 = relay.var("var_137", dtype = "float32", shape = (1,))#candidate|137|(1,)|var|float32
var_138 = relay.var("var_138", dtype = "float32", shape = (2,))#candidate|138|(2,)|var|float32
var_139 = relay.var("var_139", dtype = "float32", shape = (2,))#candidate|139|(2,)|var|float32
var_140 = relay.var("var_140", dtype = "float32", shape = (2,))#candidate|140|(2,)|var|float32
var_141 = relay.var("var_141", dtype = "float32", shape = (2,))#candidate|141|(2,)|var|float32
var_142 = relay.var("var_142", dtype = "uint8", shape = (2,))#candidate|142|(2,)|var|uint8
var_143 = relay.var("var_143", dtype = "uint8", shape = (2,))#candidate|143|(2,)|var|uint8
var_144 = relay.var("var_144", dtype = "float64", shape = (2,))#candidate|144|(2,)|var|float64
var_145 = relay.var("var_145", dtype = "float32", shape = (2,))#candidate|145|(2,)|var|float32
var_146 = relay.var("var_146", dtype = "float32", shape = (2,))#candidate|146|(2,)|var|float32
var_147 = relay.var("var_147", dtype = "bool", shape = (2,))#candidate|147|(2,)|var|bool
var_148 = relay.var("var_148", dtype = "bool", shape = (2,))#candidate|148|(2,)|var|bool
var_149 = relay.var("var_149", dtype = "uint16", shape = (2,))#candidate|149|(2,)|var|uint16
output = func_136(var_137,var_138,var_139,var_140,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,)
func_150 = relay.Function([var_137,var_138,var_139,var_140,var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,], output)
mutated_mod['func_150'] = func_150
mutated_mod = relay.transform.InferType()(mutated_mod)
var_152 = relay.var("var_152", dtype = "float32", shape = (13,))#candidate|152|(13,)|var|float32
uop_153 = relay.sigmoid(var_152.astype('float32')) # shape=(13,)
bop_155 = relay.power(uop_153.astype('float32'), relay.reshape(var_152.astype('float32'), relay.shape_of(uop_153))) # shape=(13,)
bop_158 = relay.power(uop_153.astype('float64'), relay.reshape(var_152.astype('float64'), relay.shape_of(uop_153))) # shape=(13,)
bop_161 = relay.floor_divide(uop_153.astype('float64'), relay.reshape(var_152.astype('float64'), relay.shape_of(uop_153))) # shape=(13,)
bop_164 = relay.logical_or(bop_158.astype('bool'), relay.reshape(bop_161.astype('bool'), relay.shape_of(bop_158))) # shape=(13,)
var_167 = relay.var("var_167", dtype = "float32", shape = (13,))#candidate|167|(13,)|var|float32
bop_168 = relay.left_shift(bop_155.astype('uint16'), relay.reshape(var_167.astype('uint16'), relay.shape_of(bop_155))) # shape=(13,)
uop_171 = relay.asin(bop_168.astype('float64')) # shape=(13,)
bop_173 = relay.bitwise_or(uop_171.astype('uint32'), relay.reshape(uop_153.astype('uint32'), relay.shape_of(uop_171))) # shape=(13,)
const_176 = relay.const([4,-9,-3,-6,9,9,6,-3,-1,5,7,-9,-4], dtype = "uint16")#candidate|176|(13,)|const|uint16
bop_177 = relay.maximum(bop_168.astype('float32'), relay.reshape(const_176.astype('float32'), relay.shape_of(bop_168))) # shape=(13,)
bop_180 = relay.greater(uop_171.astype('bool'), relay.reshape(bop_177.astype('bool'), relay.shape_of(uop_171))) # shape=(13,)
output = relay.Tuple([bop_164,bop_173,bop_180,])
output2 = relay.Tuple([bop_164,bop_173,bop_180,])
func_183 = relay.Function([var_152,var_167,], output)
mod['func_183'] = func_183
mod = relay.transform.InferType()(mod)
mutated_mod['func_183'] = func_183
mutated_mod = relay.transform.InferType()(mutated_mod)
func_183_call = mutated_mod.get_global_var('func_183')
var_185 = relay.var("var_185", dtype = "float32", shape = (13,))#candidate|185|(13,)|var|float32
var_186 = relay.var("var_186", dtype = "float32", shape = (13,))#candidate|186|(13,)|var|float32
call_184 = func_183_call(var_185,var_186,)
output = call_184
func_187 = relay.Function([var_185,var_186,], output)
mutated_mod['func_187'] = func_187
mutated_mod = relay.transform.InferType()(mutated_mod)
var_189 = relay.var("var_189", dtype = "uint32", shape = (9,))#candidate|189|(9,)|var|uint32
var_190 = relay.var("var_190", dtype = "uint32", shape = (9,))#candidate|190|(9,)|var|uint32
bop_191 = relay.not_equal(var_189.astype('bool'), relay.reshape(var_190.astype('bool'), relay.shape_of(var_189))) # shape=(9,)
uop_194 = relay.sigmoid(var_190.astype('float32')) # shape=(9,)
func_136_call = mod.get_global_var('func_136')
func_150_call = mutated_mod.get_global_var('func_150')
var_197 = relay.var("var_197", dtype = "float32", shape = (1,))#candidate|197|(1,)|var|float32
var_198 = relay.var("var_198", dtype = "float32", shape = (2,))#candidate|198|(2,)|var|float32
call_196 = relay.TupleGetItem(func_136_call(relay.reshape(var_197.astype('float32'), [1,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('uint8'), [2,]), relay.reshape(var_198.astype('uint8'), [2,]), relay.reshape(var_198.astype('float64'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('bool'), [2,]), relay.reshape(var_198.astype('bool'), [2,]), relay.reshape(var_198.astype('uint16'), [2,]), ), 1)
call_199 = relay.TupleGetItem(func_150_call(relay.reshape(var_197.astype('float32'), [1,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('uint8'), [2,]), relay.reshape(var_198.astype('uint8'), [2,]), relay.reshape(var_198.astype('float64'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('float32'), [2,]), relay.reshape(var_198.astype('bool'), [2,]), relay.reshape(var_198.astype('bool'), [2,]), relay.reshape(var_198.astype('uint16'), [2,]), ), 1)
bop_200 = relay.minimum(var_190.astype('uint32'), relay.reshape(uop_194.astype('uint32'), relay.shape_of(var_190))) # shape=(9,)
uop_203 = relay.sin(var_198.astype('float32')) # shape=(2,)
uop_205 = relay.atan(var_197.astype('float64')) # shape=(1,)
uop_207 = relay.tan(uop_194.astype('float32')) # shape=(9,)
bop_209 = relay.not_equal(bop_200.astype('bool'), relay.reshape(var_189.astype('bool'), relay.shape_of(bop_200))) # shape=(9,)
bop_212 = relay.bitwise_and(var_190.astype('int16'), var_197.astype('int16')) # shape=(9,)
var_215 = relay.var("var_215", dtype = "float32", shape = (9,))#candidate|215|(9,)|var|float32
bop_216 = relay.logical_or(uop_207.astype('bool'), relay.reshape(var_215.astype('bool'), relay.shape_of(uop_207))) # shape=(9,)
uop_219 = relay.asin(bop_216.astype('float32')) # shape=(9,)
uop_221 = relay.log10(uop_219.astype('float32')) # shape=(9,)
bop_223 = relay.equal(uop_219.astype('bool'), relay.reshape(var_189.astype('bool'), relay.shape_of(uop_219))) # shape=(9,)
const_226 = relay.const([True,True,True,True,True,False,False,True,False], dtype = "bool")#candidate|226|(9,)|const|bool
bop_227 = relay.divide(bop_216.astype('float64'), relay.reshape(const_226.astype('float64'), relay.shape_of(bop_216))) # shape=(9,)
output = relay.Tuple([bop_191,call_196,uop_203,uop_205,bop_209,bop_212,uop_221,bop_223,bop_227,])
output2 = relay.Tuple([bop_191,call_199,uop_203,uop_205,bop_209,bop_212,uop_221,bop_223,bop_227,])
func_230 = relay.Function([var_189,var_190,var_197,var_198,var_215,], output)
mod['func_230'] = func_230
mod = relay.transform.InferType()(mod)
mutated_mod['func_230'] = func_230
mutated_mod = relay.transform.InferType()(mutated_mod)
func_230_call = mutated_mod.get_global_var('func_230')
var_232 = relay.var("var_232", dtype = "uint32", shape = (9,))#candidate|232|(9,)|var|uint32
var_233 = relay.var("var_233", dtype = "uint32", shape = (9,))#candidate|233|(9,)|var|uint32
var_234 = relay.var("var_234", dtype = "float32", shape = (1,))#candidate|234|(1,)|var|float32
var_235 = relay.var("var_235", dtype = "float32", shape = (2,))#candidate|235|(2,)|var|float32
var_236 = relay.var("var_236", dtype = "float32", shape = (9,))#candidate|236|(9,)|var|float32
call_231 = func_230_call(var_232,var_233,var_234,var_235,var_236,)
output = call_231
func_237 = relay.Function([var_232,var_233,var_234,var_235,var_236,], output)
mutated_mod['func_237'] = func_237
mutated_mod = relay.transform.InferType()(mutated_mod)
const_239 = relay.const([3.513388,3.420584,-6.637218,0.420834,-7.763399,-2.536805,-7.748075,-0.159120,1.364369,-6.953552,-0.321729,5.712343], dtype = "float32")#candidate|239|(12,)|const|float32
uop_240 = relay.exp(const_239.astype('float32')) # shape=(12,)
bop_242 = relay.equal(uop_240.astype('bool'), relay.reshape(const_239.astype('bool'), relay.shape_of(uop_240))) # shape=(12,)
output = bop_242
output2 = bop_242
func_245 = relay.Function([], output)
mod['func_245'] = func_245
mod = relay.transform.InferType()(mod)
mutated_mod['func_245'] = func_245
mutated_mod = relay.transform.InferType()(mutated_mod)
func_245_call = mutated_mod.get_global_var('func_245')
call_246 = func_245_call()
output = call_246
func_247 = relay.Function([], output)
mutated_mod['func_247'] = func_247
mutated_mod = relay.transform.InferType()(mutated_mod)
var_248 = relay.var("var_248", dtype = "float64", shape = ())#candidate|248|()|var|float64
uop_249 = relay.acos(var_248.astype('float64')) # shape=()
bop_251 = relay.logical_xor(uop_249.astype('uint8'), var_248.astype('uint8')) # shape=()
bop_254 = relay.floor_mod(uop_249.astype('float32'), bop_251.astype('float32')) # shape=()
uop_257 = relay.asin(var_248.astype('float32')) # shape=()
func_136_call = mod.get_global_var('func_136')
func_150_call = mutated_mod.get_global_var('func_150')
var_260 = relay.var("var_260", dtype = "float32", shape = (2,))#candidate|260|(2,)|var|float32
call_259 = relay.TupleGetItem(func_136_call(relay.reshape(uop_257.astype('float32'), [1,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('uint8'), [2,]), relay.reshape(var_260.astype('uint8'), [2,]), relay.reshape(var_260.astype('float64'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('bool'), [2,]), relay.reshape(var_260.astype('bool'), [2,]), relay.reshape(var_260.astype('uint16'), [2,]), ), 3)
call_261 = relay.TupleGetItem(func_150_call(relay.reshape(uop_257.astype('float32'), [1,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('uint8'), [2,]), relay.reshape(var_260.astype('uint8'), [2,]), relay.reshape(var_260.astype('float64'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('float32'), [2,]), relay.reshape(var_260.astype('bool'), [2,]), relay.reshape(var_260.astype('bool'), [2,]), relay.reshape(var_260.astype('uint16'), [2,]), ), 3)
uop_262 = relay.log2(var_260.astype('float32')) # shape=(2,)
bop_264 = relay.bitwise_and(var_260.astype('int32'), var_248.astype('int32')) # shape=(2,)
output = relay.Tuple([bop_254,uop_257,call_259,uop_262,bop_264,])
output2 = relay.Tuple([bop_254,uop_257,call_261,uop_262,bop_264,])
func_267 = relay.Function([var_248,var_260,], output)
mod['func_267'] = func_267
mod = relay.transform.InferType()(mod)
var_268 = relay.var("var_268", dtype = "float64", shape = ())#candidate|268|()|var|float64
var_269 = relay.var("var_269", dtype = "float32", shape = (2,))#candidate|269|(2,)|var|float32
output = func_267(var_268,var_269,)
func_270 = relay.Function([var_268,var_269,], output)
mutated_mod['func_270'] = func_270
mutated_mod = relay.transform.InferType()(mutated_mod)
var_272 = relay.var("var_272", dtype = "float32", shape = ())#candidate|272|()|var|float32
uop_273 = relay.asinh(var_272.astype('float32')) # shape=()
bop_275 = relay.left_shift(var_272.astype('uint8'), uop_273.astype('uint8')) # shape=()
uop_278 = relay.log(uop_273.astype('float32')) # shape=()
uop_280 = relay.cosh(var_272.astype('float32')) # shape=()
uop_282 = relay.cos(uop_278.astype('float32')) # shape=()
output = relay.Tuple([bop_275,uop_280,uop_282,])
output2 = relay.Tuple([bop_275,uop_280,uop_282,])
func_284 = relay.Function([var_272,], output)
mod['func_284'] = func_284
mod = relay.transform.InferType()(mod)
mutated_mod['func_284'] = func_284
mutated_mod = relay.transform.InferType()(mutated_mod)
var_285 = relay.var("var_285", dtype = "float32", shape = ())#candidate|285|()|var|float32
func_284_call = mutated_mod.get_global_var('func_284')
call_286 = func_284_call(var_285)
output = call_286
func_287 = relay.Function([var_285], output)
mutated_mod['func_287'] = func_287
mutated_mod = relay.transform.InferType()(mutated_mod)
const_289 = relay.const(8.694929, dtype = "float32")#candidate|289|()|const|float32
uop_290 = relay.acos(const_289.astype('float32')) # shape=()
uop_292 = relay.asinh(uop_290.astype('float64')) # shape=()
bop_294 = relay.less_equal(uop_292.astype('bool'), const_289.astype('bool')) # shape=()
bop_297 = relay.not_equal(const_289.astype('bool'), uop_290.astype('bool')) # shape=()
uop_300 = relay.sigmoid(bop_294.astype('float32')) # shape=()
func_284_call = mod.get_global_var('func_284')
func_287_call = mutated_mod.get_global_var('func_287')
call_302 = relay.TupleGetItem(func_284_call(relay.reshape(uop_290.astype('float32'), [])), 2)
call_303 = relay.TupleGetItem(func_287_call(relay.reshape(uop_290.astype('float32'), [])), 2)
output = relay.Tuple([bop_297,uop_300,call_302,])
output2 = relay.Tuple([bop_297,uop_300,call_303,])
func_304 = relay.Function([], output)
mod['func_304'] = func_304
mod = relay.transform.InferType()(mod)
mutated_mod['func_304'] = func_304
mutated_mod = relay.transform.InferType()(mutated_mod)
func_304_call = mutated_mod.get_global_var('func_304')
call_305 = func_304_call()
output = call_305
func_306 = relay.Function([], output)
mutated_mod['func_306'] = func_306
mutated_mod = relay.transform.InferType()(mutated_mod)
const_307 = relay.const([[-6,8,-6,7,8,10,-4,10,-7,-6,-4,-1,4,-8,-10]], dtype = "int16")#candidate|307|(1, 15)|const|int16
var_308 = relay.var("var_308", dtype = "int16", shape = (1, 15))#candidate|308|(1, 15)|var|int16
bop_309 = relay.bitwise_or(const_307.astype('int16'), relay.reshape(var_308.astype('int16'), relay.shape_of(const_307))) # shape=(1, 15)
uop_312 = relay.sqrt(var_308.astype('float64')) # shape=(1, 15)
uop_314 = relay.log2(uop_312.astype('float32')) # shape=(1, 15)
uop_316 = relay.atan(uop_314.astype('float32')) # shape=(1, 15)
uop_318 = relay.acosh(uop_316.astype('float64')) # shape=(1, 15)
var_320 = relay.var("var_320", dtype = "float64", shape = (6, 15))#candidate|320|(6, 15)|var|float64
bop_321 = relay.add(uop_318.astype('uint64'), var_320.astype('uint64')) # shape=(6, 15)
bop_324 = relay.floor_divide(uop_318.astype('float32'), var_320.astype('float32')) # shape=(6, 15)
bop_327 = relay.add(uop_318.astype('uint16'), var_320.astype('uint16')) # shape=(6, 15)
func_267_call = mod.get_global_var('func_267')
func_270_call = mutated_mod.get_global_var('func_270')
var_331 = relay.var("var_331", dtype = "float64", shape = ())#candidate|331|()|var|float64
const_332 = relay.const([-2.121649,-5.106831], dtype = "float32")#candidate|332|(2,)|const|float32
call_330 = relay.TupleGetItem(func_267_call(relay.reshape(var_331.astype('float64'), []), relay.reshape(const_332.astype('float32'), [2,]), ), 1)
call_333 = relay.TupleGetItem(func_270_call(relay.reshape(var_331.astype('float64'), []), relay.reshape(const_332.astype('float32'), [2,]), ), 1)
var_334 = relay.var("var_334", dtype = "float32", shape = (9, 15))#candidate|334|(9, 15)|var|float32
bop_335 = relay.bitwise_xor(uop_316.astype('int8'), var_334.astype('int8')) # shape=(9, 15)
bop_338 = relay.bitwise_or(bop_327.astype('uint64'), uop_312.astype('uint64')) # shape=(6, 15)
uop_341 = relay.asinh(uop_318.astype('float32')) # shape=(1, 15)
uop_343 = relay.acos(uop_341.astype('float64')) # shape=(1, 15)
uop_345 = relay.acosh(uop_341.astype('float64')) # shape=(1, 15)
output = relay.Tuple([bop_309,bop_321,bop_324,call_330,var_331,const_332,bop_335,bop_338,uop_343,uop_345,])
output2 = relay.Tuple([bop_309,bop_321,bop_324,call_333,var_331,const_332,bop_335,bop_338,uop_343,uop_345,])
func_347 = relay.Function([var_308,var_320,var_331,var_334,], output)
mod['func_347'] = func_347
mod = relay.transform.InferType()(mod)
var_348 = relay.var("var_348", dtype = "int16", shape = (1, 15))#candidate|348|(1, 15)|var|int16
var_349 = relay.var("var_349", dtype = "float64", shape = (6, 15))#candidate|349|(6, 15)|var|float64
var_350 = relay.var("var_350", dtype = "float64", shape = ())#candidate|350|()|var|float64
var_351 = relay.var("var_351", dtype = "float32", shape = (9, 15))#candidate|351|(9, 15)|var|float32
output = func_347(var_348,var_349,var_350,var_351,)
func_352 = relay.Function([var_348,var_349,var_350,var_351,], output)
mutated_mod['func_352'] = func_352
mutated_mod = relay.transform.InferType()(mutated_mod)
var_354 = relay.var("var_354", dtype = "float64", shape = (13, 15, 14))#candidate|354|(13, 15, 14)|var|float64
uop_355 = relay.cos(var_354.astype('float64')) # shape=(13, 15, 14)
output = uop_355
output2 = uop_355
F = relay.Function([var_354,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_354,], output2)
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
input_354= np.array([[[8.289304,-3.168727,8.623747,1.299834,-0.761416,6.406805,-0.824890,-6.850901,6.527785,-1.826004,-4.052701,2.569795,8.155434,-7.236648],[3.565346,-4.457557,-2.104874,-9.839552,4.789526,-1.850292,7.868688,9.432915,7.533487,-8.344023,-6.658216,1.373059,-1.518590,-1.008982],[-6.205093,-8.493289,2.816646,-0.533200,-8.226387,0.987561,-8.230912,3.930771,-2.030841,-6.001874,-3.685897,-2.578728,0.908168,5.064732],[-8.333875,8.931647,-9.568014,-4.630907,0.451620,-1.488346,9.944465,1.105552,8.640730,6.943996,1.716923,0.190263,8.474438,-4.381399],[-0.864622,-3.677650,0.964387,1.880365,6.818668,2.217495,8.084108,-1.970825,-5.127796,-4.709661,-3.662539,9.824691,7.151448,-2.917598],[-7.492378,-7.706101,-6.692760,0.018287,-5.049437,-6.701312,-6.880932,-6.132753,6.233633,9.419143,-8.070182,8.727406,5.227427,0.760752],[-1.227897,6.710732,-8.382284,7.843193,-2.909703,7.913609,0.604098,3.183134,9.111839,0.950196,-5.089755,-3.209423,2.271545,-6.521907],[-4.879006,3.484580,3.980111,8.483896,-7.099532,-1.482692,8.865941,-4.350855,8.177221,7.331276,8.190378,5.718201,4.468874,1.143885],[-1.843221,0.022271,5.890311,4.154674,-0.964703,-4.020357,7.418632,1.882729,-4.370691,4.926566,3.322684,4.531710,-6.155883,7.264456],[-4.893474,-8.316354,-9.185680,-9.188583,-9.315654,-2.859327,-0.390255,-6.714310,0.009721,0.293285,1.392929,-6.657089,0.054260,0.073523],[-0.027545,2.789846,3.868416,9.181203,2.825472,5.027391,6.677247,-2.422857,1.272379,6.832018,1.115273,-2.162908,-0.848435,6.605665],[-7.325271,3.972993,3.315526,-2.467388,-0.968574,-2.354746,2.287688,6.624090,7.767102,-9.896500,-0.625706,-9.911282,-9.436455,-6.789871],[5.881963,-1.425388,2.611015,-9.405050,-8.806020,0.482908,9.530788,5.612660,-7.630430,6.134582,-9.447886,-1.708573,-2.104752,-7.722983],[8.322694,0.859387,4.629570,8.295844,2.736196,8.984197,-0.809931,-8.991920,6.308314,-5.546619,-1.775207,3.924287,3.368059,-7.040490],[-7.078431,1.198201,5.683614,-7.208620,-7.445147,3.126081,-5.576990,-9.586626,-2.772173,-1.814271,-6.687512,4.556386,-5.236944,7.007738]],[[-1.619422,2.627745,2.489580,-2.239444,3.670676,0.867752,-0.369111,7.997046,5.876799,3.311229,-8.296945,9.662080,-5.212553,-6.367417],[-9.599762,6.488213,7.911706,3.938386,3.981502,7.623050,9.647961,-1.183295,-9.401428,-4.236367,0.360153,5.043254,9.405060,-1.006353],[1.429157,-6.456572,9.313893,-1.844290,7.295671,3.623717,2.696453,2.523721,6.771528,0.622617,-9.970416,0.048630,6.173478,1.078398],[-2.721470,1.871595,-9.549099,1.347664,7.810056,-4.159062,7.118043,6.575724,5.958763,5.615728,-9.391579,-7.947003,-7.305880,1.434406],[-5.732939,-7.642690,-5.481775,-6.899028,-3.602069,-4.914914,3.526683,-3.418677,-2.885425,8.895020,-9.378763,-8.262963,1.021720,3.292554],[9.323928,-5.959938,-9.045931,-2.400250,-6.785395,4.470650,-4.802116,-8.130630,4.038328,-3.059573,-7.678061,3.245597,1.083300,1.327467],[-0.223456,-6.554258,9.732671,9.838745,-9.566691,7.585356,-7.969372,5.125879,-6.293809,-5.701833,-4.266491,-2.122291,-0.601322,6.715547],[9.523636,8.499300,-7.212582,3.852443,-8.771025,5.220807,-5.326386,5.477858,0.779349,3.905632,-8.588110,4.314709,2.139838,9.487180],[0.440795,8.066618,5.702047,-3.552572,-5.266252,-5.701843,-1.979060,7.944166,-7.288734,2.443211,8.484679,3.947543,-0.881970,-7.972196],[-3.593959,-8.084138,6.410350,8.001740,-2.563835,-1.586433,2.009916,6.124283,-5.447809,0.362143,7.252894,-6.705753,4.118909,-0.541665],[5.468718,1.645007,-4.181063,3.105693,2.451922,-4.766521,6.822888,6.065561,0.612004,-6.340925,7.784606,-1.705803,2.347168,-9.731915],[7.471635,-7.575578,7.873426,-0.262783,4.060307,2.350016,-5.993600,-4.412944,-1.204712,3.946596,-2.893982,2.321788,0.126067,-9.610128],[4.020683,6.125949,5.808380,8.658713,-4.884867,-6.580195,1.978139,-5.335592,-4.870716,9.545457,-4.926987,-7.214028,7.780024,5.101650],[-0.514806,3.430549,-3.279452,-2.928049,-5.482126,-4.788840,0.611478,6.055922,8.232697,1.952948,-5.355597,-1.914118,0.124896,-8.831441],[5.918380,3.997158,8.202029,5.104219,4.865904,-8.679098,1.089697,-2.981953,9.283690,4.127210,-2.495336,-8.260023,-9.676807,4.729237]],[[-0.602654,-1.323692,7.581741,7.782456,-3.789449,-6.035232,-6.827790,1.450149,-6.080357,-9.323877,1.145795,-7.516591,-7.481657,0.942897],[1.849756,-8.670483,-1.625467,2.106355,-0.500336,-4.550774,1.140838,0.639058,8.296033,0.468055,-1.100939,-5.510938,-8.080804,1.455282],[8.717277,3.159728,-3.456766,-4.735740,-9.880172,-0.578434,-3.544972,3.779021,-9.318882,9.331770,2.645293,1.000627,4.917558,-9.286068],[-1.043027,-6.688616,-3.128653,-1.426048,-6.160945,0.280813,8.026144,7.838280,-1.350180,7.263461,7.168419,8.296995,-6.750310,4.750350],[-1.353836,5.163783,2.286058,-4.740915,4.806009,-8.804281,-1.597566,-8.196728,-9.771075,5.219555,-9.840641,2.703303,-2.845864,-9.996513],[6.428894,-0.857370,2.602471,-3.090736,2.752873,-5.555952,5.913615,-1.884319,-7.089691,5.701960,6.764461,-4.720752,-9.346307,9.320068],[-1.196974,3.666473,4.503446,7.822838,8.742999,-2.311475,-9.552507,2.197542,3.950427,4.219389,-0.861764,-2.337913,-3.434234,8.584436],[-5.293085,1.165942,-5.144916,-7.711560,-1.446039,9.217887,5.015287,-0.459248,-2.262055,-3.949312,-1.314508,-9.217733,-9.715058,-8.651457],[3.382052,8.031988,-2.174177,-4.590421,5.433391,1.212945,-2.083761,4.832902,-4.407075,7.542009,5.235142,-2.822413,-8.912643,-3.413335],[-6.499275,6.909769,-1.087320,4.768618,5.233294,-1.058214,6.453266,-9.217331,-2.073472,7.381124,2.978033,4.968663,-1.232954,-6.548084],[-5.878959,-5.590977,4.254790,-9.340292,-8.517921,9.124872,9.535476,0.985053,4.389030,-6.259050,1.668543,-9.877499,5.496936,-4.341482],[2.816968,-6.479260,9.028019,-9.951282,-6.733221,4.079400,2.352907,-8.816600,-3.357267,9.319795,3.298173,7.265758,4.054445,5.866557],[9.544311,7.884862,1.866176,1.082554,-7.658809,2.719116,-0.147139,9.442848,4.883002,7.316661,3.957153,6.794469,-6.762358,-6.607071],[-9.437085,-5.973891,0.918825,0.506248,9.828230,3.992335,-1.128850,-4.350199,9.806144,-3.915360,5.023628,-9.473538,-2.136349,2.641552],[-3.755968,9.327498,-6.914583,0.190552,1.447644,-1.174841,3.997462,-6.656939,-6.590911,-7.222269,-6.538604,2.114917,-6.004904,7.589931]],[[1.650851,-6.671323,-6.388977,8.622102,8.392067,-6.988208,7.442027,-4.080975,-6.392287,-2.982678,5.932772,5.675051,-8.093591,-9.897077],[-3.799830,7.600629,-4.898963,-2.144798,5.408783,0.489242,-9.106265,6.834565,-0.342861,-9.068698,-0.995722,-9.318465,5.243572,-9.090350],[-7.343644,1.521497,-8.716744,-8.789150,-0.072719,1.069681,-2.872666,-9.914185,-2.798128,-9.166180,0.484639,5.681022,-6.839860,-5.764701],[7.903250,4.235747,7.167609,4.240205,6.427337,-2.110344,-3.310415,-1.344050,6.283211,2.798003,-3.281213,-9.241285,-9.316658,-0.045044],[3.111397,-1.182407,0.553895,-8.081375,-4.521184,-9.653290,-0.644702,6.377774,-3.303659,5.079060,2.369725,3.461983,4.079005,9.720596],[7.742302,-6.247977,-0.921641,-2.366703,-8.022887,-3.852492,-0.603396,-9.207733,-1.503594,1.288886,5.407651,-2.900264,0.311655,8.022027],[5.605912,-0.377651,-5.788491,-8.654496,4.090005,-8.315407,8.993846,0.739627,0.814948,-3.524426,9.623215,-6.682986,9.227767,9.790018],[-8.495034,8.024902,-5.739305,-0.990757,-1.436089,2.219661,0.072255,9.636160,6.439945,-3.343287,-0.938979,-6.160740,-9.549485,7.873451],[9.100826,-1.264698,1.101542,0.634186,0.274809,-2.689809,-0.990034,-1.642530,5.868115,5.943938,-8.113478,3.287377,-8.325653,-8.718625],[-8.150104,-8.553427,1.744801,8.298459,-0.818789,-8.337179,-2.430347,7.611465,-6.658986,3.821549,5.024749,-0.852264,-7.698389,2.034340],[4.925736,-8.827784,7.290207,1.408362,7.246096,5.381017,-2.356752,-9.497031,-4.699860,6.029398,0.414184,-9.160920,2.673791,-5.324409],[-9.907742,3.054325,7.044196,2.722329,-7.804794,7.322645,6.729962,-6.662033,-1.979763,-1.170529,-2.877140,-2.700853,-1.642019,4.765836],[2.725552,-6.814757,5.829076,-3.110481,3.452216,2.679043,-3.732223,2.270596,3.936634,8.212803,-3.003444,-7.216911,-1.255691,-7.878694],[1.300408,-5.372886,2.591066,-9.455031,-6.344294,-2.124849,-1.989915,-0.588843,-4.402661,-2.767379,8.748123,-3.183003,4.768197,-3.248435],[8.347210,4.288554,-2.016463,3.642444,-2.932983,-5.905377,-5.504245,-0.061110,-0.244186,-9.973896,-0.510647,-9.692383,-9.506849,-4.405230]],[[8.379385,7.542069,3.711882,0.917587,1.424196,-5.330688,-9.712384,-6.046458,6.826444,8.725385,5.687885,-3.372356,-9.552258,4.586184],[7.911226,8.596918,6.029211,8.273535,1.480539,-1.477219,-4.814303,-9.814403,-9.749033,5.778389,7.404121,2.890835,-8.256235,-7.541269],[9.146481,-2.992656,1.175395,1.366664,-2.698114,-0.080046,-1.539886,-1.172999,7.350590,2.412543,-3.230224,1.041721,-9.546483,-6.608029],[-7.439098,6.203443,6.583297,9.234772,4.861391,1.181214,0.040778,5.578905,-3.411757,-3.141209,-4.842006,8.843053,2.753345,-9.987890],[-0.483134,-5.390800,-7.787470,-0.910003,8.647188,-2.804257,-3.223954,1.530917,7.578441,3.612385,-4.061164,4.634732,-5.059860,6.572135],[7.676887,5.820747,7.777982,6.262888,9.002829,1.603219,8.549716,-0.097315,7.393274,8.452315,-7.684063,-8.323365,1.418444,-0.350746],[4.632701,3.711759,7.433401,-5.364268,-8.860114,-1.100663,3.785263,9.233918,-9.649114,-6.310841,1.210269,-8.586823,6.598483,-5.445368],[3.081846,-3.008325,7.337563,2.426153,-6.695386,0.877034,7.284578,-0.541552,6.305080,6.868834,7.740151,5.724381,-2.475214,7.661417],[-1.386003,-8.113541,-3.666017,-9.965206,-9.096245,-7.142481,0.018720,1.305684,-0.545398,0.753124,3.213364,3.404627,5.158372,1.683858],[-2.941975,9.551483,4.563276,-9.616684,-9.725424,5.375631,8.915308,2.735955,8.578937,9.927771,-3.474442,7.921186,-6.754598,2.093475],[0.642656,2.616037,7.519493,-5.789573,-0.394825,-4.483810,-1.714462,0.510663,1.121368,-7.365070,6.221520,-3.227448,4.332535,-6.672824],[-3.894386,5.486328,-4.301639,-2.506702,0.484441,-3.326343,5.202591,-1.528868,-6.901018,5.779419,-2.671449,-5.941170,-2.242874,9.163917],[-7.335387,3.817071,7.412594,-8.756329,9.724437,0.911390,-4.586213,-4.822957,-7.086923,9.757506,-2.131127,6.419253,-5.512935,-3.826667],[-5.210756,-3.409867,7.987162,-1.735208,-3.303495,3.999010,0.451117,-3.139268,-3.584581,1.468750,3.816774,-7.423184,-4.358677,-6.274378],[7.094550,0.695734,9.781202,7.589558,4.787060,1.231345,-4.648257,9.599961,-8.299855,5.914864,-3.231744,4.670097,1.328115,5.078971]],[[7.566017,4.572779,-8.294743,3.246498,-9.203444,1.386879,-7.297478,-4.689097,6.803448,-0.760890,4.750824,-9.688452,-0.941913,-6.627179],[0.718702,0.584418,4.339936,-4.028213,2.406213,5.760925,-3.159286,7.010540,-3.930124,9.533373,8.067949,5.065179,-6.913745,-2.139579],[8.535965,-7.889549,-8.041339,-1.832341,-2.887450,-9.455312,-6.321341,-8.964339,-2.033525,4.194607,3.712308,-6.258450,-6.148388,3.927680],[0.364346,7.623707,-4.596855,-3.844505,-9.162772,6.096314,2.591851,-3.147049,4.165094,-0.802910,8.629712,-1.936440,-9.759677,0.161343],[-0.967511,0.880177,2.466457,4.557224,3.862053,-5.006204,-9.270740,-7.605860,7.132099,4.802712,9.222351,-3.339707,2.986396,1.081083],[2.423136,-1.689411,-7.592607,-0.862246,-5.838919,5.936581,4.149324,5.696556,1.647574,3.243478,0.464370,-0.221062,-4.923226,9.900782],[2.613186,1.265740,6.065802,1.692511,-6.005433,4.214197,1.762958,-8.794054,5.580217,-9.512667,3.484117,0.331142,-3.530463,-4.209280],[4.031338,-3.732681,-5.991136,7.989757,4.689370,3.340607,-1.251564,-1.711824,-1.740569,-0.766317,6.744785,-3.006147,-7.443162,3.576725],[9.480548,7.508718,3.112695,-3.226640,-1.731342,8.807018,-3.509462,0.039744,-3.956764,-7.995214,-9.366918,-2.072754,6.021935,-1.710909],[-8.287128,-1.990264,7.238912,-0.839110,0.184054,9.989361,9.152528,4.544682,4.034745,-1.392910,-2.067524,-8.065618,-4.449637,-4.352790],[-1.996846,3.057760,-6.330291,-3.400579,-1.181974,-9.762844,4.543365,3.528888,3.675912,-0.889575,4.872705,-4.796237,2.378393,-5.983243],[4.729709,0.499731,7.256648,-0.023504,-2.319237,-2.117402,-7.143254,-3.926766,-0.162650,-1.341061,-6.381159,-2.008979,9.584285,-7.551089],[-2.036923,-7.887195,5.832926,7.151049,5.643570,7.290070,-6.302018,-1.983143,-7.074399,2.132242,-7.320345,-5.908137,-9.376537,-7.223664],[3.889968,5.183087,1.951402,0.171974,7.367569,0.499133,9.743257,3.079075,-5.944516,-2.704593,-0.564570,4.941939,8.679094,-7.026971],[0.026557,7.482681,9.728978,-1.596449,1.572575,6.106115,-3.461763,8.865841,0.900224,1.811760,-0.897694,4.599410,6.733781,9.333508]],[[-6.240922,3.330778,-7.802108,3.639381,0.852975,-5.269272,0.204371,7.719135,-9.180523,2.077586,4.004982,-9.267910,-1.081628,3.885188],[-7.260859,-1.116964,-2.793645,-0.193423,1.269480,-5.853052,9.850834,4.891425,-4.220034,-0.484193,5.558450,7.648210,0.591197,-1.944495],[2.003232,3.510541,-8.431742,-9.568764,6.845814,9.491360,-1.797299,1.385807,-5.890191,3.299808,-2.338215,4.308573,9.512371,-2.103181],[-5.690553,5.793794,-5.064293,-1.175704,-8.599250,9.960097,4.173284,-9.717423,-7.168786,-2.302768,-5.755758,-2.823876,7.831855,-1.967920],[-4.373305,0.305233,2.766154,-5.676137,-3.304270,-6.935342,3.305383,-3.367862,-5.541877,3.940737,4.497157,1.539510,7.562464,-5.562777],[-3.230524,9.396284,3.338824,4.357251,5.161118,7.570261,-3.013401,4.364794,4.876537,7.880315,-5.847799,6.330534,-9.281061,-0.450807],[-5.166377,-3.593583,2.868037,0.664276,1.284893,-2.580713,-9.882345,-1.304035,3.386446,-6.533406,8.296111,7.210680,-0.208167,8.467664],[7.858282,-5.457344,-5.581136,-5.211598,0.146480,9.669006,8.202181,7.669279,6.633260,-0.373631,2.919665,2.347086,-6.051461,6.792367],[2.782969,8.455043,-8.295714,-3.301731,2.609649,-6.167563,-3.548116,3.273304,2.603397,1.285555,-0.569991,-0.791223,8.497540,-0.275588],[-6.623846,-2.016767,8.877581,-2.683915,-0.963312,-6.711170,-0.661642,-1.313438,-5.762742,-6.710761,-8.285659,-6.698420,0.910383,1.827178],[7.831708,9.901034,4.491694,8.175433,3.161939,-0.611030,7.337527,1.670005,5.477030,-2.735522,0.363612,-1.235384,5.406639,-4.743453],[5.495606,4.636459,-8.863498,-8.683423,-2.879329,-7.158068,6.003178,7.349453,7.208362,-6.797797,-8.988169,6.137922,4.682454,-5.499109],[-7.576424,-4.191440,-4.402981,-8.386860,-2.601785,-2.176920,5.303093,-1.838893,2.560692,2.666035,6.103609,2.841285,-8.603964,1.553155],[-8.297580,-5.540721,7.353484,-4.057517,-6.999364,-0.912438,6.889002,-3.655522,7.817472,-6.238603,-0.009547,5.431748,0.587984,-6.976779],[7.284883,-2.621466,6.880913,2.584883,0.991633,-7.713031,-0.536512,-4.104549,4.482467,-3.149163,7.328120,1.986657,-7.577414,4.428184]],[[8.174636,-9.429417,-0.453396,3.879472,1.445722,-8.889794,-0.639885,0.455982,7.503197,1.240923,-4.036194,7.778825,-0.777844,3.220875],[8.976670,-3.521674,-1.182987,-2.042393,9.581613,-6.522799,-7.240586,6.768799,3.004034,3.449315,-0.748088,7.763344,7.849381,3.189153],[7.089881,-9.769307,4.108054,-6.468662,9.839441,-7.009482,-8.481204,-7.467573,6.834314,3.033616,-5.469899,7.937568,6.203549,9.891190],[2.503133,-2.047230,5.177489,-3.192245,-1.471740,0.145272,5.376853,-6.193703,-6.501607,-7.557387,2.988582,0.819535,-5.699930,-7.469277],[2.879837,-4.598177,-0.582541,-2.062446,-4.357905,1.515594,0.073003,-6.428120,-4.397142,-3.172328,2.281961,0.823610,-7.739317,-4.171648],[0.407067,8.574414,-8.868105,-4.660381,-8.152110,7.709112,8.634492,7.419143,3.221301,8.954738,5.617816,-4.928888,-1.804112,6.940612],[-7.785017,7.429737,-4.006045,0.050743,4.751356,-0.678463,-5.260445,-8.839537,-7.650347,-7.196879,7.666106,2.342032,1.334785,-0.487472],[8.262910,-8.586697,9.237985,-9.965782,-2.374162,8.084504,1.826944,-4.472693,7.107097,2.304288,-7.729984,6.952891,4.318295,-3.089595],[-0.375865,-5.015360,-3.246151,-2.514725,-4.236675,-5.224776,-2.643530,-9.724908,8.940021,7.294884,6.612643,-9.108567,-8.616649,-1.055988],[-3.465361,-5.756408,5.191200,2.490568,7.548055,4.040384,-8.626142,6.162490,0.683988,-2.341730,-7.909817,1.632825,-6.712539,2.964388],[6.851636,-4.363228,6.889603,2.409370,1.025074,-8.603467,-0.695430,-9.646334,6.647577,7.136619,-7.124834,5.308191,1.458095,-9.831749],[-8.949352,-0.330703,7.079344,-0.731418,2.626061,9.670072,-6.075638,5.643244,-9.274628,8.820167,7.559416,-7.371902,-3.025843,-2.522224],[8.807850,2.272107,8.416651,2.223009,-3.348325,0.626374,5.153257,-0.819184,-9.316115,-6.078663,5.302296,7.077265,-9.174905,-3.915215],[2.913648,-3.239667,-7.216011,6.015712,1.669014,-1.402766,-5.146523,0.898073,9.006925,7.596656,-3.059282,-2.034794,5.872197,-0.004179],[0.474820,-7.678685,-5.471427,3.125321,-8.087486,-4.722686,-4.943837,5.712436,-6.926213,-6.059694,-8.154336,1.126115,-7.914401,-9.842928]],[[2.253896,9.608316,-4.094331,-4.085986,6.253099,1.878658,-1.611166,3.273074,-6.693547,6.508165,-8.332727,-3.324162,5.422067,8.302623],[0.534600,8.838940,-9.795108,-9.235327,-0.069962,8.733851,-7.665498,2.525167,-0.223244,0.480270,9.619015,3.414534,9.121679,-2.319828],[-4.808366,-6.898860,3.083080,-8.014092,-9.769174,-2.794987,5.571577,-5.215568,-1.964419,-6.575033,6.603116,-7.887952,-5.897770,-9.344490],[3.118243,-9.792754,-7.848365,-8.547916,-5.050711,2.741417,5.982883,5.100102,-7.635089,-1.854779,5.598869,2.682551,3.407280,9.032965],[-7.155947,3.007274,-3.559394,8.995129,-7.941078,8.037264,-0.016909,3.604005,-5.817233,1.773239,5.891199,-9.318062,-5.584580,7.724242],[2.424628,-2.625484,-8.457374,2.534848,2.372369,1.322490,-3.166928,3.794000,9.679568,-5.799555,4.520532,-5.104889,-1.012075,-1.329269],[5.824039,-3.313711,-9.676355,-2.666607,-0.403278,-7.853193,-4.990535,5.928263,-4.227336,2.302979,9.371390,6.432849,2.540550,5.681069],[-7.271388,8.057962,5.913253,-8.124531,-4.710866,-2.575151,4.734731,-7.767222,-4.673739,-3.781327,-6.954271,9.188901,-4.212565,4.505785],[-0.096081,9.542079,-6.429229,-6.648239,-5.532186,5.321757,4.592721,6.338232,-6.736642,-7.673865,5.528440,3.931162,6.310979,5.239638],[-4.942192,5.896101,-9.671148,0.743117,4.756520,-3.788013,5.953402,-7.381893,-1.861582,8.018257,-5.791051,1.483885,-0.259426,1.728891],[7.052910,8.124430,3.357790,9.131985,4.749377,-8.465859,-2.091908,-6.823788,1.990693,5.025046,-6.936235,-0.420971,-0.197387,-5.976563],[5.735969,-8.283539,1.252489,-2.410950,2.161911,-5.001331,-0.847827,8.678886,3.837792,-2.963726,4.020953,-2.376510,-6.992815,-3.231369],[2.200798,-1.806061,-4.687800,6.158328,-9.130386,9.134691,8.045710,6.177502,-5.129196,8.740658,-9.135995,-0.239186,8.372691,-0.547389],[-7.486748,-8.800087,4.007771,-0.788017,-6.007866,2.036994,7.544348,-4.070137,-7.006602,-6.612226,-4.423272,2.057804,-0.773256,-4.272830],[-3.965180,5.332403,4.029774,-4.631563,-6.941858,1.718602,7.942566,-6.202960,0.054292,3.621856,-4.227894,-8.587792,9.554773,6.425435]],[[-6.307191,9.221653,-3.584225,4.021747,6.205076,0.075039,-0.048065,-8.003672,6.006277,-7.372736,5.257919,4.436631,0.053788,-8.294100],[0.078806,-4.406468,-7.047397,-5.553527,3.199940,-8.568630,-4.961780,-1.110620,-6.321824,-2.831553,9.765571,-0.642145,-8.908162,-1.821332],[-5.971543,2.960388,0.520569,8.191425,2.003636,-7.019959,5.472035,9.471092,-0.218891,-4.579855,-3.432294,5.010621,9.263472,-7.841407],[6.342729,9.236678,-3.711712,5.966052,4.520710,4.918792,-1.521752,5.556585,-0.021362,5.127140,-6.621098,-4.486932,-3.710418,2.504736],[9.650822,-8.229505,-2.608058,-0.107559,-2.135174,-4.347428,-7.246448,-5.026154,-9.259229,-1.664060,3.154479,1.508221,3.033583,-0.521111],[4.043507,0.618222,-5.253912,7.124495,-9.558903,0.649618,0.856035,-6.884157,1.902545,6.677478,-5.388227,-2.403766,8.824901,5.144187],[7.158826,4.116084,2.704359,-2.784167,-3.321567,5.548104,8.600984,9.699212,-5.146865,9.895636,-6.235524,6.952465,-2.622992,-9.985122],[-5.413529,4.156885,-8.183094,-6.650568,7.512609,0.256904,1.961266,-2.296391,5.881878,9.589019,4.650569,8.497086,3.461502,-5.923020],[-4.616482,-3.696854,7.962041,7.211801,-0.567040,2.688507,-4.503087,9.728362,7.242666,-2.843568,1.322923,7.636949,-6.154860,-0.760248],[0.652820,-3.341230,-4.992186,-6.098170,-8.795967,4.810663,-1.831728,-8.422018,6.897002,-4.580392,-0.857611,0.898248,-2.212177,-3.064050],[0.239112,8.838444,-0.299190,-6.486527,-1.118247,-0.969654,-3.718622,2.980691,-1.612725,9.416029,0.012577,7.902834,4.528062,0.693896],[5.652498,-6.060799,-6.890159,4.938245,4.754360,1.231574,6.099018,-4.457790,2.778097,-0.865643,-9.767841,2.947875,5.596181,2.399018],[8.972253,-7.906938,-3.313888,9.252869,-9.970218,0.161820,2.179028,4.522451,-8.207405,-5.711020,1.935618,-1.216114,8.736527,6.163091],[-5.656825,5.362440,0.857323,-8.125945,-3.117473,5.748941,-6.175814,8.739005,9.965638,-7.084847,-4.692870,-8.415636,4.610584,7.893908],[-9.477608,1.495501,0.056622,0.668669,2.064701,8.429175,2.118323,-3.188544,2.230892,0.595254,0.212604,7.937146,8.393703,5.690215]],[[-5.808945,3.687882,-6.102742,1.224874,7.732323,-2.110940,-9.143709,-8.995317,-0.819518,5.498657,-4.642516,2.625246,7.064824,-7.025004],[-9.857288,6.823628,-3.989231,5.199130,-8.965290,5.934081,6.279380,-7.924738,6.385493,-3.058866,5.950645,7.065894,-2.890112,6.412152],[3.769522,-5.091804,-1.040123,-0.142686,-1.583057,9.187931,-7.404051,-1.530871,-6.283128,6.446461,-3.587516,-0.048669,-2.741420,8.056774],[-8.229354,-4.123886,-2.371308,1.900187,-7.671876,-4.840697,4.183560,4.472463,-4.175688,-5.976575,-7.745733,-8.401151,2.890148,3.180788],[-4.130238,-8.379737,0.696713,9.958961,-7.133427,-2.692535,-6.773118,2.743364,-4.994548,-7.067349,6.760813,-2.602311,-0.457324,-2.839568],[5.167477,6.478137,-8.879806,-7.338501,-9.856818,-6.462119,-5.706431,-0.270126,0.607346,-9.630107,-2.416549,-1.625290,3.912217,-6.688471],[2.307044,9.152061,-6.899379,-2.572626,0.382775,-1.585302,-1.911321,-6.120029,-7.861196,-7.773356,-9.078381,2.058588,7.355490,-3.973138],[-6.386175,-4.253236,5.221874,-7.624384,0.269698,6.405667,8.954950,-0.361050,-3.493486,-8.884430,4.230819,-2.382040,-7.515407,5.932499],[4.855807,-5.051466,-0.973650,3.697547,5.237668,9.148412,-0.800285,1.639183,-4.530750,-4.589758,-0.103488,8.298968,8.039941,5.828078],[-0.798506,3.574344,9.870627,6.273663,-4.185902,8.983067,4.743240,-5.098471,8.341884,3.542485,-4.365453,-4.511297,5.790899,-4.009587],[6.362219,-6.226881,-0.978246,4.660183,8.803272,9.644395,8.582113,1.036415,2.031194,9.071694,1.616314,6.847127,-0.793491,-0.828676],[1.153772,-5.872223,0.357485,-8.856760,-4.575287,0.273984,4.064561,9.387274,0.197098,5.583969,1.316078,-6.764255,6.316365,-9.181896],[7.559771,9.214012,5.375215,-1.285930,-8.892340,2.650566,-8.844413,8.009194,7.160637,3.101095,1.435232,-0.998418,6.600561,-1.012698],[-2.382513,-2.356581,1.047779,-0.261995,9.453729,2.163054,-1.017023,-8.677976,7.444285,-3.655455,-9.833626,-0.784081,-4.163357,-0.160528],[3.561984,-3.815060,2.944823,4.754769,0.769360,-1.975674,-6.202863,-1.807013,-0.523236,-6.753777,-0.611757,-2.399158,-9.169616,1.844216]],[[9.905534,-0.201976,-0.543188,-9.581506,0.141039,-9.508858,-8.434249,8.104718,4.156600,6.333469,-0.605784,5.418744,-2.678733,1.093609],[-4.528622,7.925988,9.710518,-7.461231,-3.280680,9.965147,4.475314,-2.482000,-9.772999,-1.977693,-9.809233,8.835516,6.234441,6.999571],[0.389233,-0.067054,4.389777,1.328781,1.138168,-0.452913,6.054260,5.968121,-1.372822,-3.104574,2.086242,-2.448161,-8.430550,-7.157569],[-6.222998,7.018066,9.326709,-5.057542,4.433371,0.390730,1.438032,-4.535823,6.813951,2.913308,-2.668171,5.871765,9.088014,1.106143],[1.404799,3.713593,7.489153,3.094073,4.702078,-3.116502,-1.161149,4.944171,5.229940,-3.492598,-9.451139,-8.471628,0.528140,-6.094383],[6.588722,-4.898977,-3.407913,7.346828,-0.524809,-9.418256,-6.804602,-2.094142,-8.128696,-4.945627,3.072051,-3.376705,2.823600,-5.408871],[-3.140784,7.021227,-4.940657,7.465487,8.347816,8.570931,9.441487,3.596570,-2.361490,-3.041744,-8.973978,-1.960824,6.177104,9.571766],[-4.025189,6.464224,-2.798051,-0.151717,-1.822808,8.997759,-3.794464,3.973930,-5.619214,9.151036,-6.689931,6.487868,3.858099,0.189252],[7.150775,5.818223,-1.871046,-7.799838,4.720806,-0.417953,-0.044598,0.279107,-8.978286,1.750287,-6.422104,6.000167,-8.828298,7.590663],[-7.684449,7.418995,4.330314,7.119515,8.366145,-9.305875,1.436392,3.033957,1.506272,-3.921625,7.688939,-9.362024,9.374815,2.189049],[4.280287,9.183008,9.150913,8.883177,5.525286,-1.304365,-5.675137,4.346752,5.298153,-2.280256,5.995308,-4.165124,-0.590063,-9.613780],[-0.817750,3.436278,-1.821228,-8.977727,9.180420,-1.375655,-7.010495,-5.012708,-8.433488,-2.981251,-7.215745,-1.215354,-0.602061,-2.066364],[0.587403,-2.934851,-5.643981,8.261344,-3.367967,-3.897160,4.336251,-8.018379,-1.897861,7.796630,-6.656629,-1.882910,-3.858205,-9.409798],[-5.011103,5.319789,-8.555280,9.932065,5.487573,4.825312,-3.575627,-8.225560,4.122464,-3.730796,2.256931,-2.033095,2.537248,-3.926816],[1.780418,-0.614993,6.533714,4.846251,3.747352,4.272530,-6.901600,-5.031037,-9.603058,-8.727213,1.742945,6.810548,-3.162189,-4.853095]],[[1.850552,3.029022,-7.361260,-9.253822,0.328919,7.924860,-5.138055,-6.652904,4.318241,-0.948029,2.075824,2.304199,3.725512,0.011451],[-4.514095,9.776788,-6.538981,-0.262208,8.411283,-3.669179,5.915482,0.799420,-5.893411,-3.731231,0.251367,-2.751964,2.328808,6.124955],[-0.675071,1.625512,9.106460,-8.518662,9.587157,-4.923934,1.551862,-5.846830,-8.335192,4.095274,3.832206,-3.977423,-1.864523,7.515300],[2.895161,-6.571547,8.015849,-5.154130,-0.382816,-7.365618,-8.186304,-7.513688,0.007708,-3.943528,2.425369,8.259317,5.742187,2.763762],[-0.551166,4.917027,0.394039,4.329140,1.539441,5.616969,7.896148,-6.952662,-6.325216,5.484703,4.685564,0.040106,7.321580,4.142805],[8.079096,-3.458338,8.458461,9.593181,4.587745,7.865027,-0.356392,3.454905,-8.857537,4.836101,-7.546292,-3.760555,-0.864653,-0.020020],[-2.941349,-5.930231,-3.701700,9.943615,7.103285,6.642903,-1.555140,3.052324,6.542796,-7.043215,-0.424145,5.641192,-9.761332,-7.607410],[4.946438,5.287924,-4.071153,2.486341,-6.596159,-2.459112,0.455561,7.197769,-5.448074,8.693475,4.866148,4.876685,3.790158,-0.381124],[4.448703,7.271413,-3.847559,-9.516769,-6.329515,0.821486,6.806191,-1.768034,2.152548,-5.242562,8.369152,-5.018328,8.824104,-0.800183],[-7.232085,-4.938030,-4.065198,-0.282367,-7.423311,9.302108,6.122933,-8.690988,8.712257,8.714700,-3.737802,-0.999539,9.088150,-5.312053],[2.875316,-8.225717,8.182424,7.921933,-4.985687,-2.995945,-4.865059,6.531529,6.629790,-5.030081,3.442242,3.057608,3.190886,5.276094],[-9.478011,3.491385,-4.589142,4.853787,9.585136,7.646294,5.406295,-8.563318,-2.987559,3.739632,4.850221,-0.094875,5.819916,-4.603982],[1.704154,-2.822568,1.983699,2.473739,1.077576,-2.517761,8.275229,-6.611683,4.415991,9.482813,-0.467845,0.134460,-2.738661,-1.106574],[-8.326818,5.977627,-2.884734,6.030549,0.182994,-3.364604,-8.985915,7.611356,-1.429863,-9.608564,1.367396,5.446980,-1.887644,0.603467],[-8.152422,-8.664182,2.367844,4.888117,2.442340,8.183358,0.820683,6.816292,-7.201665,-4.408940,-3.814775,7.994624,-9.454432,-1.113515]]], dtype='float64')
module1.set_input('var_354', input_354)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_354, )
res3 = intrp3.evaluate()(input_354, )
res4 = intrp4.evaluate()(input_354, )
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
module5.set_input('var_354', input_354)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_354, )
res7 = intrp7.evaluate()(input_354, )
res8 = intrp8.evaluate()(input_354, )
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
module9.set_input('var_354', input_354)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_354, )
res11 = intrp11.evaluate()(input_354, )
res12 = intrp12.evaluate()(input_354, )
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
module13.set_input('var_354', input_354)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_354, )
res15 = intrp15.evaluate()(input_354, )
res16 = intrp16.evaluate()(input_354, )
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
module17.set_input('var_354', input_354)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_354, )
res19 = intrp19.evaluate()(input_354, )
res20 = intrp20.evaluate()(input_354, )
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
module21.set_input('var_354', input_354)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_354, )
res23 = intrp23.evaluate()(input_354, )
res24 = intrp24.evaluate()(input_354, )
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

'''43: TVMFuncCall
42: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
41: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
40: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
39: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
38: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
37: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
36: tvm::transform::Pass::operator()(tvm::IRModule) const
35: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
34: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
30: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
29: tvm::transform::Pass::operator()(tvm::IRModule) const
28: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
27: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
26: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
25: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
22: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
20: _ZN3tvm5relay9transform22Devic
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
18: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
17: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
15: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
14: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
8: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
7: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
6: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
5: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
3: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
2: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
1: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
0: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)

'''