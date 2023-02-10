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
var_1 = relay.var("var_1", dtype = "float64", shape = (4,))#candidate|1|(4,)|var|float64
bop_2 = relay.power(var_0.astype('float64'), var_1.astype('float64')) # shape=(4,)
uop_5 = relay.asinh(var_0.astype('float64')) # shape=()
const_7 = relay.const([-0.265695,-2.222821,7.359836,-6.663968], dtype = "float64")#candidate|7|(4,)|const|float64
bop_8 = relay.floor_divide(var_1.astype('float64'), relay.reshape(const_7.astype('float64'), relay.shape_of(var_1))) # shape=(4,)
bop_11 = relay.minimum(uop_5.astype('uint8'), const_7.astype('uint8')) # shape=(4,)
uop_14 = relay.log2(bop_11.astype('float64')) # shape=(4,)
uop_16 = relay.cosh(bop_11.astype('float64')) # shape=(4,)
bop_18 = relay.add(uop_16.astype('float32'), var_0.astype('float32')) # shape=(4,)
uop_21 = relay.acos(uop_14.astype('float64')) # shape=(4,)
uop_23 = relay.atanh(uop_21.astype('float64')) # shape=(4,)
bop_25 = relay.add(bop_11.astype('int16'), relay.reshape(bop_8.astype('int16'), relay.shape_of(bop_11))) # shape=(4,)
bop_28 = relay.maximum(bop_8.astype('uint8'), relay.reshape(bop_25.astype('uint8'), relay.shape_of(bop_8))) # shape=(4,)
bop_31 = relay.logical_and(uop_23.astype('bool'), relay.reshape(uop_16.astype('bool'), relay.shape_of(uop_23))) # shape=(4,)
bop_34 = relay.right_shift(uop_21.astype('int8'), relay.reshape(var_1.astype('int8'), relay.shape_of(uop_21))) # shape=(4,)
bop_37 = relay.right_shift(bop_34.astype('uint32'), relay.reshape(uop_16.astype('uint32'), relay.shape_of(bop_34))) # shape=(4,)
uop_40 = relay.rsqrt(bop_31.astype('float32')) # shape=(4,)
const_42 = relay.const([-5.453225,-5.327069,2.943904,-1.808404], dtype = "float32")#candidate|42|(4,)|const|float32
bop_43 = relay.logical_or(uop_40.astype('bool'), relay.reshape(const_42.astype('bool'), relay.shape_of(uop_40))) # shape=(4,)
var_46 = relay.var("var_46", dtype = "bool", shape = (4,))#candidate|46|(4,)|var|bool
bop_47 = relay.left_shift(bop_43.astype('int32'), relay.reshape(var_46.astype('int32'), relay.shape_of(bop_43))) # shape=(4,)
output = relay.Tuple([bop_2,bop_18,bop_28,bop_37,bop_47,])
output2 = relay.Tuple([bop_2,bop_18,bop_28,bop_37,bop_47,])
func_50 = relay.Function([var_0,var_1,var_46,], output)
mod['func_50'] = func_50
mod = relay.transform.InferType()(mod)
mutated_mod['func_50'] = func_50
mutated_mod = relay.transform.InferType()(mutated_mod)
func_50_call = mutated_mod.get_global_var('func_50')
var_52 = relay.var("var_52", dtype = "float64", shape = ())#candidate|52|()|var|float64
var_53 = relay.var("var_53", dtype = "float64", shape = (4,))#candidate|53|(4,)|var|float64
var_54 = relay.var("var_54", dtype = "bool", shape = (4,))#candidate|54|(4,)|var|bool
call_51 = func_50_call(var_52,var_53,var_54,)
output = call_51
func_55 = relay.Function([var_52,var_53,var_54,], output)
mutated_mod['func_55'] = func_55
mutated_mod = relay.transform.InferType()(mutated_mod)
var_57 = relay.var("var_57", dtype = "float64", shape = ())#candidate|57|()|var|float64
uop_58 = relay.log10(var_57.astype('float64')) # shape=()
var_60 = relay.var("var_60", dtype = "float64", shape = (6,))#candidate|60|(6,)|var|float64
bop_61 = relay.greater_equal(uop_58.astype('bool'), var_60.astype('bool')) # shape=(6,)
uop_64 = relay.sinh(uop_58.astype('float64')) # shape=()
bop_66 = relay.greater_equal(bop_61.astype('bool'), uop_58.astype('bool')) # shape=(6,)
output = relay.Tuple([uop_64,bop_66,])
output2 = relay.Tuple([uop_64,bop_66,])
func_69 = relay.Function([var_57,var_60,], output)
mod['func_69'] = func_69
mod = relay.transform.InferType()(mod)
mutated_mod['func_69'] = func_69
mutated_mod = relay.transform.InferType()(mutated_mod)
func_69_call = mutated_mod.get_global_var('func_69')
var_71 = relay.var("var_71", dtype = "float64", shape = ())#candidate|71|()|var|float64
var_72 = relay.var("var_72", dtype = "float64", shape = (6,))#candidate|72|(6,)|var|float64
call_70 = func_69_call(var_71,var_72,)
output = call_70
func_73 = relay.Function([var_71,var_72,], output)
mutated_mod['func_73'] = func_73
mutated_mod = relay.transform.InferType()(mutated_mod)
var_75 = relay.var("var_75", dtype = "float32", shape = (3,))#candidate|75|(3,)|var|float32
var_76 = relay.var("var_76", dtype = "float32", shape = (3,))#candidate|76|(3,)|var|float32
bop_77 = relay.minimum(var_75.astype('float32'), relay.reshape(var_76.astype('float32'), relay.shape_of(var_75))) # shape=(3,)
bop_80 = relay.bitwise_xor(bop_77.astype('uint64'), relay.reshape(var_75.astype('uint64'), relay.shape_of(bop_77))) # shape=(3,)
const_83 = relay.const([-8.158053,6.371368,3.471104], dtype = "float32")#candidate|83|(3,)|const|float32
bop_84 = relay.bitwise_and(var_76.astype('uint64'), relay.reshape(const_83.astype('uint64'), relay.shape_of(var_76))) # shape=(3,)
bop_87 = relay.power(bop_77.astype('float64'), relay.reshape(const_83.astype('float64'), relay.shape_of(bop_77))) # shape=(3,)
bop_90 = relay.subtract(const_83.astype('int32'), relay.reshape(var_75.astype('int32'), relay.shape_of(const_83))) # shape=(3,)
bop_93 = relay.not_equal(bop_84.astype('bool'), relay.reshape(bop_77.astype('bool'), relay.shape_of(bop_84))) # shape=(3,)
bop_96 = relay.not_equal(bop_90.astype('bool'), relay.reshape(var_76.astype('bool'), relay.shape_of(bop_90))) # shape=(3,)
bop_99 = relay.left_shift(bop_77.astype('uint64'), relay.reshape(bop_84.astype('uint64'), relay.shape_of(bop_77))) # shape=(3,)
var_102 = relay.var("var_102", dtype = "uint64", shape = (3,))#candidate|102|(3,)|var|uint64
bop_103 = relay.bitwise_or(bop_99.astype('uint64'), relay.reshape(var_102.astype('uint64'), relay.shape_of(bop_99))) # shape=(3,)
uop_106 = relay.cosh(bop_80.astype('float32')) # shape=(3,)
uop_108 = relay.sin(bop_87.astype('float64')) # shape=(3,)
var_110 = relay.var("var_110", dtype = "float64", shape = (3,))#candidate|110|(3,)|var|float64
bop_111 = relay.left_shift(uop_108.astype('int32'), relay.reshape(var_110.astype('int32'), relay.shape_of(uop_108))) # shape=(3,)
var_114 = relay.var("var_114", dtype = "uint64", shape = (3,))#candidate|114|(3,)|var|uint64
bop_115 = relay.multiply(bop_103.astype('uint32'), relay.reshape(var_114.astype('uint32'), relay.shape_of(bop_103))) # shape=(3,)
var_118 = relay.var("var_118", dtype = "float32", shape = (3,))#candidate|118|(3,)|var|float32
bop_119 = relay.bitwise_xor(uop_106.astype('int8'), relay.reshape(var_118.astype('int8'), relay.shape_of(uop_106))) # shape=(3,)
uop_122 = relay.cos(bop_119.astype('float32')) # shape=(3,)
bop_124 = relay.multiply(bop_119.astype('float64'), relay.reshape(bop_93.astype('float64'), relay.shape_of(bop_119))) # shape=(3,)
uop_127 = relay.cosh(uop_122.astype('float64')) # shape=(3,)
bop_129 = relay.divide(uop_122.astype('float64'), relay.reshape(bop_119.astype('float64'), relay.shape_of(uop_122))) # shape=(3,)
bop_132 = relay.logical_or(uop_127.astype('bool'), relay.reshape(var_76.astype('bool'), relay.shape_of(uop_127))) # shape=(3,)
var_135 = relay.var("var_135", dtype = "bool", shape = (3,))#candidate|135|(3,)|var|bool
bop_136 = relay.divide(bop_132.astype('float64'), relay.reshape(var_135.astype('float64'), relay.shape_of(bop_132))) # shape=(3,)
output = relay.Tuple([bop_96,bop_111,bop_115,bop_124,bop_129,bop_136,])
output2 = relay.Tuple([bop_96,bop_111,bop_115,bop_124,bop_129,bop_136,])
func_139 = relay.Function([var_75,var_76,var_102,var_110,var_114,var_118,var_135,], output)
mod['func_139'] = func_139
mod = relay.transform.InferType()(mod)
var_140 = relay.var("var_140", dtype = "float32", shape = (3,))#candidate|140|(3,)|var|float32
var_141 = relay.var("var_141", dtype = "float32", shape = (3,))#candidate|141|(3,)|var|float32
var_142 = relay.var("var_142", dtype = "uint64", shape = (3,))#candidate|142|(3,)|var|uint64
var_143 = relay.var("var_143", dtype = "float64", shape = (3,))#candidate|143|(3,)|var|float64
var_144 = relay.var("var_144", dtype = "uint64", shape = (3,))#candidate|144|(3,)|var|uint64
var_145 = relay.var("var_145", dtype = "float32", shape = (3,))#candidate|145|(3,)|var|float32
var_146 = relay.var("var_146", dtype = "bool", shape = (3,))#candidate|146|(3,)|var|bool
output = func_139(var_140,var_141,var_142,var_143,var_144,var_145,var_146,)
func_147 = relay.Function([var_140,var_141,var_142,var_143,var_144,var_145,var_146,], output)
mutated_mod['func_147'] = func_147
mutated_mod = relay.transform.InferType()(mutated_mod)
var_149 = relay.var("var_149", dtype = "float32", shape = (9,))#candidate|149|(9,)|var|float32
uop_150 = relay.cos(var_149.astype('float32')) # shape=(9,)
output = relay.Tuple([uop_150,])
output2 = relay.Tuple([uop_150,])
func_152 = relay.Function([var_149,], output)
mod['func_152'] = func_152
mod = relay.transform.InferType()(mod)
var_153 = relay.var("var_153", dtype = "float32", shape = (9,))#candidate|153|(9,)|var|float32
output = func_152(var_153)
func_154 = relay.Function([var_153], output)
mutated_mod['func_154'] = func_154
mutated_mod = relay.transform.InferType()(mutated_mod)
var_156 = relay.var("var_156", dtype = "float64", shape = (16,))#candidate|156|(16,)|var|float64
uop_157 = relay.sin(var_156.astype('float64')) # shape=(16,)
bop_159 = relay.greater_equal(var_156.astype('bool'), relay.reshape(uop_157.astype('bool'), relay.shape_of(var_156))) # shape=(16,)
uop_162 = relay.sqrt(uop_157.astype('float64')) # shape=(16,)
uop_164 = relay.atan(bop_159.astype('float32')) # shape=(16,)
uop_166 = relay.cos(uop_164.astype('float32')) # shape=(16,)
uop_168 = relay.sigmoid(var_156.astype('float32')) # shape=(16,)
uop_170 = relay.acos(uop_166.astype('float32')) # shape=(16,)
uop_172 = relay.atan(uop_157.astype('float32')) # shape=(16,)
bop_174 = relay.minimum(uop_166.astype('uint64'), relay.reshape(uop_157.astype('uint64'), relay.shape_of(uop_166))) # shape=(16,)
var_177 = relay.var("var_177", dtype = "float32", shape = (16,))#candidate|177|(16,)|var|float32
bop_178 = relay.not_equal(uop_170.astype('bool'), relay.reshape(var_177.astype('bool'), relay.shape_of(uop_170))) # shape=(16,)
bop_181 = relay.add(uop_157.astype('uint8'), relay.reshape(bop_178.astype('uint8'), relay.shape_of(uop_157))) # shape=(16,)
uop_184 = relay.sin(bop_159.astype('float64')) # shape=(16,)
bop_186 = relay.floor_mod(uop_184.astype('float64'), relay.reshape(bop_159.astype('float64'), relay.shape_of(uop_184))) # shape=(16,)
uop_189 = relay.cos(bop_178.astype('float64')) # shape=(16,)
const_191 = relay.const([-7.897594,1.688318,0.830874,-6.858515,-5.546716,9.625374,1.572648,1.489694,-8.364892,-3.989709,3.260993,-0.225086,7.624559,4.006043,1.774902,-2.893162], dtype = "float64")#candidate|191|(16,)|const|float64
bop_192 = relay.divide(uop_189.astype('float32'), relay.reshape(const_191.astype('float32'), relay.shape_of(uop_189))) # shape=(16,)
uop_195 = relay.rsqrt(bop_192.astype('float64')) # shape=(16,)
uop_197 = relay.sin(bop_192.astype('float64')) # shape=(16,)
var_199 = relay.var("var_199", dtype = "float64", shape = (16,))#candidate|199|(16,)|var|float64
bop_200 = relay.add(uop_197.astype('int16'), relay.reshape(var_199.astype('int16'), relay.shape_of(uop_197))) # shape=(16,)
uop_203 = relay.log10(bop_192.astype('float64')) # shape=(16,)
uop_205 = relay.atan(bop_200.astype('float64')) # shape=(16,)
uop_207 = relay.log10(bop_200.astype('float64')) # shape=(16,)
output = relay.Tuple([uop_162,uop_168,uop_172,bop_174,bop_181,bop_186,uop_195,uop_203,uop_205,uop_207,])
output2 = relay.Tuple([uop_162,uop_168,uop_172,bop_174,bop_181,bop_186,uop_195,uop_203,uop_205,uop_207,])
func_209 = relay.Function([var_156,var_177,var_199,], output)
mod['func_209'] = func_209
mod = relay.transform.InferType()(mod)
var_210 = relay.var("var_210", dtype = "float64", shape = (16,))#candidate|210|(16,)|var|float64
var_211 = relay.var("var_211", dtype = "float32", shape = (16,))#candidate|211|(16,)|var|float32
var_212 = relay.var("var_212", dtype = "float64", shape = (16,))#candidate|212|(16,)|var|float64
output = func_209(var_210,var_211,var_212,)
func_213 = relay.Function([var_210,var_211,var_212,], output)
mutated_mod['func_213'] = func_213
mutated_mod = relay.transform.InferType()(mutated_mod)
var_215 = relay.var("var_215", dtype = "float64", shape = (4,))#candidate|215|(4,)|var|float64
uop_216 = relay.erf(var_215.astype('float64')) # shape=(4,)
uop_218 = relay.sigmoid(uop_216.astype('float32')) # shape=(4,)
bop_220 = relay.greater_equal(uop_218.astype('bool'), relay.reshape(uop_216.astype('bool'), relay.shape_of(uop_218))) # shape=(4,)
var_223 = relay.var("var_223", dtype = "float32", shape = (4,))#candidate|223|(4,)|var|float32
bop_224 = relay.equal(uop_218.astype('bool'), relay.reshape(var_223.astype('bool'), relay.shape_of(uop_218))) # shape=(4,)
bop_227 = relay.maximum(var_223.astype('int64'), relay.reshape(uop_218.astype('int64'), relay.shape_of(var_223))) # shape=(4,)
uop_230 = relay.tan(var_223.astype('float32')) # shape=(4,)
uop_232 = relay.atanh(uop_218.astype('float64')) # shape=(4,)
uop_234 = relay.tan(uop_232.astype('float32')) # shape=(4,)
uop_236 = relay.exp(uop_234.astype('float32')) # shape=(4,)
bop_238 = relay.floor_mod(uop_216.astype('float32'), relay.reshape(uop_234.astype('float32'), relay.shape_of(uop_216))) # shape=(4,)
func_152_call = mod.get_global_var('func_152')
func_154_call = mutated_mod.get_global_var('func_154')
const_242 = relay.const([[0.292287,8.509156,-9.697176],[-1.786254,4.272809,9.569880],[-3.034046,0.988892,4.567691]], dtype = "float32")#candidate|242|(3, 3)|const|float32
call_241 = relay.TupleGetItem(func_152_call(relay.reshape(const_242.astype('float32'), [9,])), 0)
call_243 = relay.TupleGetItem(func_154_call(relay.reshape(const_242.astype('float32'), [9,])), 0)
bop_244 = relay.minimum(uop_236.astype('int32'), relay.reshape(bop_238.astype('int32'), relay.shape_of(uop_236))) # shape=(4,)
output = relay.Tuple([bop_220,bop_224,bop_227,uop_230,call_241,const_242,bop_244,])
output2 = relay.Tuple([bop_220,bop_224,bop_227,uop_230,call_243,const_242,bop_244,])
func_247 = relay.Function([var_215,var_223,], output)
mod['func_247'] = func_247
mod = relay.transform.InferType()(mod)
var_248 = relay.var("var_248", dtype = "float64", shape = (4,))#candidate|248|(4,)|var|float64
var_249 = relay.var("var_249", dtype = "float32", shape = (4,))#candidate|249|(4,)|var|float32
output = func_247(var_248,var_249,)
func_250 = relay.Function([var_248,var_249,], output)
mutated_mod['func_250'] = func_250
mutated_mod = relay.transform.InferType()(mutated_mod)
var_252 = relay.var("var_252", dtype = "float32", shape = (6, 3))#candidate|252|(6, 3)|var|float32
uop_253 = relay.cos(var_252.astype('float32')) # shape=(6, 3)
uop_255 = relay.log10(uop_253.astype('float64')) # shape=(6, 3)
output = relay.Tuple([uop_255,])
output2 = relay.Tuple([uop_255,])
func_257 = relay.Function([var_252,], output)
mod['func_257'] = func_257
mod = relay.transform.InferType()(mod)
var_258 = relay.var("var_258", dtype = "float32", shape = (6, 3))#candidate|258|(6, 3)|var|float32
output = func_257(var_258)
func_259 = relay.Function([var_258], output)
mutated_mod['func_259'] = func_259
mutated_mod = relay.transform.InferType()(mutated_mod)
const_261 = relay.const([[[8,4,-1,-6,7,-2],[7,2,-1,7,-2,6],[5,6,-8,-7,8,-2],[4,-3,6,-8,9,9],[5,-9,-6,-2,-8,-1],[7,-5,-3,7,7,-8]],[[5,-3,4,-6,10,7],[-4,-7,7,9,9,-10],[-2,-8,7,1,-2,6],[-8,6,-10,4,-3,-6],[-10,-8,-10,-1,-7,3],[-5,-6,1,-4,6,5]],[[-6,-2,1,-1,3,-1],[-8,-9,-1,9,-9,4],[8,-6,-9,-2,1,4],[-2,-6,4,9,-10,-9],[1,9,-2,9,-1,6],[-8,-5,-2,5,-3,10]],[[6,1,6,-9,5,-6],[-10,-9,-9,-3,3,8],[-8,-1,1,10,9,-5],[9,-1,-9,5,1,2],[-9,9,-2,9,-7,1],[-2,-7,-2,9,4,10]],[[-8,-5,-7,10,-4,9],[-3,6,-9,-10,-7,5],[7,3,-3,-9,5,9],[-2,9,-8,-2,7,1],[-6,-8,-7,-2,-4,-8],[-5,-7,2,4,4,10]],[[2,-3,1,-4,6,-9],[2,9,-6,4,-5,-4],[6,3,-3,-4,-6,-5],[-4,3,10,4,3,1],[-2,1,-5,-6,-4,5],[-3,-2,7,-5,8,-2]],[[10,-2,9,10,-8,3],[-7,6,-8,-9,-1,10],[-10,-10,3,-7,-8,-3],[10,-10,-8,2,2,7],[1,10,8,-2,3,5],[4,8,-10,-8,-8,-9]],[[-1,-2,-2,-2,-6,9],[-2,-2,-4,2,10,-9],[5,-1,8,-9,-4,4],[1,-1,10,-1,-10,9],[10,2,-5,8,-10,3],[10,1,-5,10,-2,-3]],[[10,7,6,3,-9,-6],[6,4,-5,-8,-1,1],[-3,-1,1,-9,9,8],[-6,-5,-6,9,7,1],[-2,-8,-7,8,9,5],[5,-8,4,2,-9,-6]],[[8,-8,-1,-10,8,-6],[9,5,3,-8,7,-3],[-6,-10,-3,-1,-4,-7],[-9,-7,-7,2,-7,3],[-8,-6,-7,-6,5,-7],[4,1,-6,-10,7,2]],[[2,-7,-10,7,-1,2],[4,-10,-4,9,-4,4],[1,5,-3,10,-10,4],[-9,-9,-6,8,-9,-10],[5,8,2,-9,8,-8],[2,8,-7,3,3,-4]]], dtype = "int32")#candidate|261|(11, 6, 6)|const|int32
const_262 = relay.const([[[-2,5,-2,-5,5,-5],[-4,-7,-5,-10,-6,10],[-7,1,10,7,2,3],[10,6,-2,4,-7,-6],[7,-7,-1,2,1,-1],[-7,-2,-2,-5,-3,-8]],[[2,-2,10,-1,6,-4],[4,6,-1,6,-5,9],[3,10,4,10,9,-5],[2,7,-6,4,7,7],[6,8,-2,-1,-10,3],[-8,-10,-7,-5,4,4]],[[5,2,-7,-1,-10,4],[-5,-9,5,-1,-6,-1],[7,-9,-2,-1,-10,-8],[-1,10,10,-5,7,2],[-7,9,3,6,10,8],[-10,-8,3,-7,-6,4]],[[-1,3,-5,1,7,7],[2,1,-9,-3,-10,-9],[-6,5,-5,8,-10,-3],[-7,10,6,5,1,9],[4,8,2,7,6,-10],[3,-7,10,-9,-4,-10]],[[-6,-10,8,6,-3,8],[-2,1,-6,-10,-10,3],[5,-4,3,-1,8,4],[3,10,4,1,-2,7],[5,9,2,-5,6,-8],[-3,7,-2,-7,-3,-5]],[[6,7,10,10,-1,1],[2,10,5,-6,9,-10],[-6,-10,-6,8,-10,-10],[8,4,-8,3,-10,8],[3,4,9,4,10,4],[8,-6,6,-7,-5,-2]],[[-7,-4,9,4,8,-3],[9,10,-1,3,-6,-8],[-6,-5,-7,5,3,2],[-9,-4,-2,4,4,1],[8,-7,-5,-5,1,2],[-8,-1,-4,2,7,10]],[[3,-3,8,4,-9,2],[-7,3,-8,3,9,8],[-9,9,-8,-7,5,-10],[3,-10,-7,6,-9,2],[-5,-5,1,10,-6,5],[-9,7,-4,6,-10,-2]],[[9,1,-9,7,-9,8],[6,9,-8,5,9,-9],[6,-10,4,-3,10,-2],[6,2,-10,5,-10,-5],[1,4,-4,4,-10,8],[6,6,-7,4,-6,9]],[[9,-9,10,-7,-5,-2],[-4,9,-1,-8,10,5],[-10,9,7,1,3,5],[5,-6,-3,-9,5,10],[7,-4,-10,4,-8,-6],[-2,1,7,1,-7,2]],[[5,8,-3,9,6,9],[-2,10,6,7,2,-5],[7,3,-3,6,8,2],[-2,-1,4,-8,3,-8],[-3,6,6,-6,-7,7],[-4,2,6,-3,-4,-9]]], dtype = "int32")#candidate|262|(11, 6, 6)|const|int32
bop_263 = relay.greater(const_261.astype('bool'), relay.reshape(const_262.astype('bool'), relay.shape_of(const_261))) # shape=(11, 6, 6)
bop_266 = relay.bitwise_and(bop_263.astype('int8'), relay.reshape(const_261.astype('int8'), relay.shape_of(bop_263))) # shape=(11, 6, 6)
uop_269 = relay.log2(const_261.astype('float64')) # shape=(11, 6, 6)
uop_271 = relay.sinh(uop_269.astype('float64')) # shape=(11, 6, 6)
output = relay.Tuple([bop_266,uop_271,])
output2 = relay.Tuple([bop_266,uop_271,])
func_273 = relay.Function([], output)
mod['func_273'] = func_273
mod = relay.transform.InferType()(mod)
mutated_mod['func_273'] = func_273
mutated_mod = relay.transform.InferType()(mutated_mod)
func_273_call = mutated_mod.get_global_var('func_273')
call_274 = func_273_call()
output = call_274
func_275 = relay.Function([], output)
mutated_mod['func_275'] = func_275
mutated_mod = relay.transform.InferType()(mutated_mod)
var_276 = relay.var("var_276", dtype = "float64", shape = (7, 3, 10))#candidate|276|(7, 3, 10)|var|float64
uop_277 = relay.cosh(var_276.astype('float64')) # shape=(7, 3, 10)
uop_279 = relay.tan(uop_277.astype('float32')) # shape=(7, 3, 10)
var_281 = relay.var("var_281", dtype = "float64", shape = (7, 3, 10))#candidate|281|(7, 3, 10)|var|float64
bop_282 = relay.divide(uop_277.astype('float32'), relay.reshape(var_281.astype('float32'), relay.shape_of(uop_277))) # shape=(7, 3, 10)
uop_285 = relay.cosh(var_276.astype('float64')) # shape=(7, 3, 10)
const_287 = relay.const([[[8.410881,-1.937147,-7.904980,-2.283894,-2.006666,0.049476,4.261835,-5.117905,-3.463929,2.066316],[-5.824207,-3.578480,-3.496539,0.579494,2.208382,1.359026,1.072563,9.079570,-5.211758,-9.997978],[2.633080,-9.727861,0.161764,0.969849,-4.665594,4.248468,-0.279240,-1.965387,-3.415849,2.775078]],[[-2.306910,9.301348,2.636495,0.020926,-3.588157,-9.072204,4.462102,-2.313816,2.135103,-7.940888],[-8.604360,7.862761,2.740107,7.040656,2.858756,-1.421987,2.299397,9.285074,-1.188263,9.726192],[-4.294511,1.781164,-1.629082,-5.708340,7.858682,-3.010798,-1.664718,-8.157000,6.808324,-3.148747]],[[9.539145,-2.000548,-5.833885,2.966919,-3.011221,1.275663,-5.524958,-5.767005,-8.828883,7.432582],[-3.259636,-2.999951,0.219377,-6.521637,-1.159650,4.319376,9.897171,0.411323,2.713274,7.617677],[2.371331,-4.547466,1.434211,-8.707156,1.552750,3.159710,5.264470,2.753742,7.917638,3.946283]],[[-6.132201,0.274923,-8.595460,3.370154,2.322600,8.392422,8.691502,-6.913742,7.173910,-4.765000],[-3.634050,-9.413143,9.629828,-2.645794,1.119986,0.896960,-3.136216,-9.535263,6.163327,-2.327214],[-0.764216,8.070936,4.593714,-0.462268,-1.670559,-2.737886,-0.019252,7.906409,-8.344713,-4.647539]],[[1.162965,5.975453,-0.221080,9.429781,7.103095,-6.987026,8.108048,5.059565,-7.479542,4.639118],[-8.108505,1.954822,-5.637140,3.710846,3.173997,0.782692,0.815368,-5.012673,-6.780104,1.070140],[-6.193621,0.601246,7.958312,-8.291746,-2.939526,-2.153036,3.613857,7.411138,-5.253328,-7.239525]],[[5.995112,3.457430,-8.264908,-3.405322,0.341195,-4.349132,-5.175281,4.873359,-0.274211,4.824913],[-5.924985,-5.024627,3.278643,6.432556,-6.020486,8.994837,5.406679,-1.185351,0.387057,-4.542879],[-6.170290,-9.508548,-5.836813,-3.921180,7.788348,2.261867,5.772108,-6.872192,-1.476096,6.417277]],[[8.780189,-8.064469,9.428376,0.355478,7.968963,4.999652,-1.486358,-8.468392,8.251494,-6.519035],[-1.092526,-6.540786,8.867799,4.563609,0.799721,-4.673586,2.878024,9.131376,0.613062,3.953046],[7.474651,-5.284914,-3.526043,-1.080740,0.831466,6.208968,7.548508,7.194582,-4.129430,-3.888915]]], dtype = "float32")#candidate|287|(7, 3, 10)|const|float32
bop_288 = relay.equal(uop_279.astype('bool'), relay.reshape(const_287.astype('bool'), relay.shape_of(uop_279))) # shape=(7, 3, 10)
uop_291 = relay.asin(bop_288.astype('float64')) # shape=(7, 3, 10)
uop_293 = relay.rsqrt(bop_288.astype('float32')) # shape=(7, 3, 10)
uop_295 = relay.asin(uop_277.astype('float64')) # shape=(7, 3, 10)
var_297 = relay.var("var_297", dtype = "bool", shape = (7, 3, 10))#candidate|297|(7, 3, 10)|var|bool
bop_298 = relay.not_equal(bop_288.astype('bool'), relay.reshape(var_297.astype('bool'), relay.shape_of(bop_288))) # shape=(7, 3, 10)
uop_301 = relay.asinh(uop_291.astype('float64')) # shape=(7, 3, 10)
var_303 = relay.var("var_303", dtype = "bool", shape = (7, 3, 10))#candidate|303|(7, 3, 10)|var|bool
bop_304 = relay.floor_divide(bop_298.astype('float32'), relay.reshape(var_303.astype('float32'), relay.shape_of(bop_298))) # shape=(7, 3, 10)
uop_307 = relay.asin(uop_293.astype('float64')) # shape=(7, 3, 10)
bop_309 = relay.less(uop_307.astype('bool'), relay.reshape(var_281.astype('bool'), relay.shape_of(uop_307))) # shape=(7, 3, 10)
uop_312 = relay.acosh(uop_301.astype('float32')) # shape=(7, 3, 10)
uop_314 = relay.asin(bop_282.astype('float32')) # shape=(7, 3, 10)
output = relay.Tuple([uop_285,uop_295,bop_304,bop_309,uop_312,uop_314,])
output2 = relay.Tuple([uop_285,uop_295,bop_304,bop_309,uop_312,uop_314,])
func_316 = relay.Function([var_276,var_281,var_297,var_303,], output)
mod['func_316'] = func_316
mod = relay.transform.InferType()(mod)
var_317 = relay.var("var_317", dtype = "float64", shape = (7, 3, 10))#candidate|317|(7, 3, 10)|var|float64
var_318 = relay.var("var_318", dtype = "float64", shape = (7, 3, 10))#candidate|318|(7, 3, 10)|var|float64
var_319 = relay.var("var_319", dtype = "bool", shape = (7, 3, 10))#candidate|319|(7, 3, 10)|var|bool
var_320 = relay.var("var_320", dtype = "bool", shape = (7, 3, 10))#candidate|320|(7, 3, 10)|var|bool
output = func_316(var_317,var_318,var_319,var_320,)
func_321 = relay.Function([var_317,var_318,var_319,var_320,], output)
mutated_mod['func_321'] = func_321
mutated_mod = relay.transform.InferType()(mutated_mod)
const_323 = relay.const(False, dtype = "bool")#candidate|323|()|const|bool
const_324 = relay.const([True,False,False,False,False,True,True], dtype = "bool")#candidate|324|(7,)|const|bool
bop_325 = relay.logical_or(const_323.astype('bool'), const_324.astype('bool')) # shape=(7,)
const_328 = relay.const([True,True,True,True,True,False,False], dtype = "bool")#candidate|328|(7,)|const|bool
bop_329 = relay.left_shift(bop_325.astype('int64'), relay.reshape(const_328.astype('int64'), relay.shape_of(bop_325))) # shape=(7,)
bop_332 = relay.maximum(bop_329.astype('int8'), relay.reshape(const_324.astype('int8'), relay.shape_of(bop_329))) # shape=(7,)
uop_335 = relay.log2(bop_332.astype('float32')) # shape=(7,)
output = relay.Tuple([uop_335,])
output2 = relay.Tuple([uop_335,])
func_337 = relay.Function([], output)
mod['func_337'] = func_337
mod = relay.transform.InferType()(mod)
mutated_mod['func_337'] = func_337
mutated_mod = relay.transform.InferType()(mutated_mod)
func_337_call = mutated_mod.get_global_var('func_337')
call_338 = func_337_call()
output = call_338
func_339 = relay.Function([], output)
mutated_mod['func_339'] = func_339
mutated_mod = relay.transform.InferType()(mutated_mod)
var_340 = relay.var("var_340", dtype = "float64", shape = (16,))#candidate|340|(16,)|var|float64
uop_341 = relay.exp(var_340.astype('float64')) # shape=(16,)
output = relay.Tuple([uop_341,])
output2 = relay.Tuple([uop_341,])
F = relay.Function([var_340,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_340,], output2)
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
input_340= np.array([-3.181691,-5.660684,-7.039163,5.715290,9.010070,0.667024,3.500029,2.885475,7.354916,4.665624,-9.944164,5.972369,-2.526269,9.510670,-8.183248,4.635479], dtype='float64')
module1.set_input('var_340', input_340)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_340, )
res3 = intrp3.evaluate()(input_340, )
res4 = intrp4.evaluate()(input_340, )
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
module5.set_input('var_340', input_340)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_340, )
res7 = intrp7.evaluate()(input_340, )
res8 = intrp8.evaluate()(input_340, )
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
module9.set_input('var_340', input_340)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_340, )
res11 = intrp11.evaluate()(input_340, )
res12 = intrp12.evaluate()(input_340, )
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
module13.set_input('var_340', input_340)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_340, )
res15 = intrp15.evaluate()(input_340, )
res16 = intrp16.evaluate()(input_340, )
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
module17.set_input('var_340', input_340)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_340, )
res19 = intrp19.evaluate()(input_340, )
res20 = intrp20.evaluate()(input_340, )
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
module21.set_input('var_340', input_340)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_340, )
res23 = intrp23.evaluate()(input_340, )
res24 = intrp24.evaluate()(input_340, )
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

'''47: TVMFuncCall
46: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
45: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
44: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
43: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
42: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
41: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
40: tvm::transform::Pass::operator()(tvm::IRModule) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
36: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
35: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
34: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
33: tvm::transform::Pass::operator()(tvm::IRModule) const
32: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
30: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
29: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
28: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
27: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
26: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
25: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
24: _ZN3tvm5relay9transform22Devic
23: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
22: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
21: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
20: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
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