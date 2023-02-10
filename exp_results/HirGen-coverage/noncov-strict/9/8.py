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
uop_1 = relay.log2(var_0.astype('float64')) # shape=()
uop_3 = relay.acosh(var_0.astype('float64')) # shape=()
bop_5 = relay.divide(uop_3.astype('float64'), var_0.astype('float64')) # shape=()
bop_8 = relay.less(bop_5.astype('bool'), var_0.astype('bool')) # shape=()
uop_11 = relay.sinh(bop_5.astype('float64')) # shape=()
bop_13 = relay.less(uop_11.astype('bool'), uop_1.astype('bool')) # shape=()
bop_16 = relay.power(var_0.astype('float32'), uop_11.astype('float32')) # shape=()
uop_19 = relay.acosh(var_0.astype('float64')) # shape=()
output = relay.Tuple([bop_8,bop_13,bop_16,uop_19,])
output2 = relay.Tuple([bop_8,bop_13,bop_16,uop_19,])
func_21 = relay.Function([var_0,], output)
mod['func_21'] = func_21
mod = relay.transform.InferType()(mod)
mutated_mod['func_21'] = func_21
mutated_mod = relay.transform.InferType()(mutated_mod)
var_22 = relay.var("var_22", dtype = "float64", shape = ())#candidate|22|()|var|float64
func_21_call = mutated_mod.get_global_var('func_21')
call_23 = func_21_call(var_22)
output = call_23
func_24 = relay.Function([var_22], output)
mutated_mod['func_24'] = func_24
mutated_mod = relay.transform.InferType()(mutated_mod)
var_26 = relay.var("var_26", dtype = "float64", shape = (5, 2))#candidate|26|(5, 2)|var|float64
uop_27 = relay.log(var_26.astype('float64')) # shape=(5, 2)
output = uop_27
output2 = uop_27
func_29 = relay.Function([var_26,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
var_30 = relay.var("var_30", dtype = "float64", shape = (5, 2))#candidate|30|(5, 2)|var|float64
output = func_29(var_30)
func_31 = relay.Function([var_30], output)
mutated_mod['func_31'] = func_31
mutated_mod = relay.transform.InferType()(mutated_mod)
var_33 = relay.var("var_33", dtype = "float64", shape = (8, 10, 13))#candidate|33|(8, 10, 13)|var|float64
var_34 = relay.var("var_34", dtype = "float64", shape = (8, 10, 13))#candidate|34|(8, 10, 13)|var|float64
bop_35 = relay.floor_divide(var_33.astype('float64'), relay.reshape(var_34.astype('float64'), relay.shape_of(var_33))) # shape=(8, 10, 13)
uop_38 = relay.atan(var_34.astype('float32')) # shape=(8, 10, 13)
bop_40 = relay.right_shift(var_34.astype('int64'), relay.reshape(uop_38.astype('int64'), relay.shape_of(var_34))) # shape=(8, 10, 13)
bop_43 = relay.floor_mod(uop_38.astype('float32'), relay.reshape(var_33.astype('float32'), relay.shape_of(uop_38))) # shape=(8, 10, 13)
bop_46 = relay.floor_mod(bop_40.astype('float64'), relay.reshape(var_33.astype('float64'), relay.shape_of(bop_40))) # shape=(8, 10, 13)
uop_49 = relay.rsqrt(bop_40.astype('float32')) # shape=(8, 10, 13)
uop_51 = relay.asin(uop_49.astype('float64')) # shape=(8, 10, 13)
bop_53 = relay.floor_divide(uop_38.astype('float32'), relay.reshape(var_34.astype('float32'), relay.shape_of(uop_38))) # shape=(8, 10, 13)
var_56 = relay.var("var_56", dtype = "float64", shape = (8, 10, 13))#candidate|56|(8, 10, 13)|var|float64
bop_57 = relay.logical_and(uop_51.astype('bool'), relay.reshape(var_56.astype('bool'), relay.shape_of(uop_51))) # shape=(8, 10, 13)
bop_60 = relay.add(bop_46.astype('uint8'), relay.reshape(uop_51.astype('uint8'), relay.shape_of(bop_46))) # shape=(8, 10, 13)
uop_63 = relay.cos(bop_53.astype('float32')) # shape=(8, 10, 13)
uop_65 = relay.log2(bop_60.astype('float32')) # shape=(8, 10, 13)
bop_67 = relay.maximum(uop_65.astype('uint8'), relay.reshape(var_33.astype('uint8'), relay.shape_of(uop_65))) # shape=(8, 10, 13)
bop_70 = relay.power(uop_65.astype('float32'), relay.reshape(bop_60.astype('float32'), relay.shape_of(uop_65))) # shape=(8, 10, 13)
output = relay.Tuple([bop_35,bop_43,bop_57,uop_63,bop_67,bop_70,])
output2 = relay.Tuple([bop_35,bop_43,bop_57,uop_63,bop_67,bop_70,])
func_73 = relay.Function([var_33,var_34,var_56,], output)
mod['func_73'] = func_73
mod = relay.transform.InferType()(mod)
var_74 = relay.var("var_74", dtype = "float64", shape = (8, 10, 13))#candidate|74|(8, 10, 13)|var|float64
var_75 = relay.var("var_75", dtype = "float64", shape = (8, 10, 13))#candidate|75|(8, 10, 13)|var|float64
var_76 = relay.var("var_76", dtype = "float64", shape = (8, 10, 13))#candidate|76|(8, 10, 13)|var|float64
output = func_73(var_74,var_75,var_76,)
func_77 = relay.Function([var_74,var_75,var_76,], output)
mutated_mod['func_77'] = func_77
mutated_mod = relay.transform.InferType()(mutated_mod)
var_79 = relay.var("var_79", dtype = "float32", shape = (12, 5))#candidate|79|(12, 5)|var|float32
uop_80 = relay.cos(var_79.astype('float32')) # shape=(12, 5)
uop_82 = relay.atan(uop_80.astype('float32')) # shape=(12, 5)
uop_84 = relay.atanh(uop_80.astype('float64')) # shape=(12, 5)
bop_86 = relay.bitwise_and(uop_80.astype('int8'), relay.reshape(uop_82.astype('int8'), relay.shape_of(uop_80))) # shape=(12, 5)
bop_89 = relay.logical_or(uop_84.astype('bool'), relay.reshape(var_79.astype('bool'), relay.shape_of(uop_84))) # shape=(12, 5)
bop_92 = relay.minimum(uop_82.astype('uint64'), relay.reshape(bop_86.astype('uint64'), relay.shape_of(uop_82))) # shape=(12, 5)
output = relay.Tuple([bop_89,bop_92,])
output2 = relay.Tuple([bop_89,bop_92,])
func_95 = relay.Function([var_79,], output)
mod['func_95'] = func_95
mod = relay.transform.InferType()(mod)
var_96 = relay.var("var_96", dtype = "float32", shape = (12, 5))#candidate|96|(12, 5)|var|float32
output = func_95(var_96)
func_97 = relay.Function([var_96], output)
mutated_mod['func_97'] = func_97
mutated_mod = relay.transform.InferType()(mutated_mod)
const_99 = relay.const(6.805978, dtype = "float32")#candidate|99|()|const|float32
uop_100 = relay.sigmoid(const_99.astype('float32')) # shape=()
var_102 = relay.var("var_102", dtype = "float32", shape = (6,))#candidate|102|(6,)|var|float32
bop_103 = relay.floor_divide(uop_100.astype('float32'), var_102.astype('float32')) # shape=(6,)
bop_106 = relay.maximum(uop_100.astype('int16'), var_102.astype('int16')) # shape=(6,)
uop_109 = relay.acosh(uop_100.astype('float32')) # shape=()
const_111 = relay.const([[[5.536155,-6.989135,-9.900265],[-0.455916,-5.516908,-1.247180],[-9.505330,3.220989,6.702793],[6.122625,-5.219464,-6.590389],[7.754944,5.869355,2.856949],[2.000707,1.928607,5.225234],[-7.501989,5.500209,-5.527913],[-9.781234,5.331319,-7.766170],[7.485725,8.852150,8.156780]]], dtype = "float32")#candidate|111|(1, 9, 3)|const|float32
bop_112 = relay.right_shift(uop_109.astype('uint32'), const_111.astype('uint32')) # shape=(1, 9, 3)
bop_115 = relay.logical_or(uop_109.astype('bool'), uop_100.astype('bool')) # shape=()
var_118 = relay.var("var_118", dtype = "uint32", shape = (14, 9, 3))#candidate|118|(14, 9, 3)|var|uint32
bop_119 = relay.divide(bop_112.astype('float64'), var_118.astype('float64')) # shape=(14, 9, 3)
var_122 = relay.var("var_122", dtype = "float32", shape = (6,))#candidate|122|(6,)|var|float32
bop_123 = relay.greater(var_102.astype('bool'), relay.reshape(var_122.astype('bool'), relay.shape_of(var_102))) # shape=(6,)
func_29_call = mod.get_global_var('func_29')
func_31_call = mutated_mod.get_global_var('func_31')
const_127 = relay.const([8.735076,-6.314982,2.288622,-5.523834,-9.804691,7.720346,-5.491985,-7.070484,-3.445079,-6.672193], dtype = "float64")#candidate|127|(10,)|const|float64
call_126 = func_29_call(relay.reshape(const_127.astype('float64'), [5, 2]))
call_128 = func_29_call(relay.reshape(const_127.astype('float64'), [5, 2]))
output = relay.Tuple([bop_103,bop_106,bop_115,bop_119,bop_123,call_126,const_127,])
output2 = relay.Tuple([bop_103,bop_106,bop_115,bop_119,bop_123,call_128,const_127,])
func_129 = relay.Function([var_102,var_118,var_122,], output)
mod['func_129'] = func_129
mod = relay.transform.InferType()(mod)
var_130 = relay.var("var_130", dtype = "float32", shape = (6,))#candidate|130|(6,)|var|float32
var_131 = relay.var("var_131", dtype = "uint32", shape = (14, 9, 3))#candidate|131|(14, 9, 3)|var|uint32
var_132 = relay.var("var_132", dtype = "float32", shape = (6,))#candidate|132|(6,)|var|float32
output = func_129(var_130,var_131,var_132,)
func_133 = relay.Function([var_130,var_131,var_132,], output)
mutated_mod['func_133'] = func_133
mutated_mod = relay.transform.InferType()(mutated_mod)
const_135 = relay.const([True,True,True,False,True,False], dtype = "bool")#candidate|135|(6,)|const|bool
const_136 = relay.const([True,True,True,False,True,False], dtype = "bool")#candidate|136|(6,)|const|bool
bop_137 = relay.logical_and(const_135.astype('bool'), relay.reshape(const_136.astype('bool'), relay.shape_of(const_135))) # shape=(6,)
uop_140 = relay.log(const_136.astype('float32')) # shape=(6,)
var_142 = relay.var("var_142", dtype = "float32", shape = (6,))#candidate|142|(6,)|var|float32
bop_143 = relay.subtract(uop_140.astype('int32'), relay.reshape(var_142.astype('int32'), relay.shape_of(uop_140))) # shape=(6,)
bop_146 = relay.greater(const_135.astype('bool'), relay.reshape(var_142.astype('bool'), relay.shape_of(const_135))) # shape=(6,)
output = relay.Tuple([bop_137,bop_143,bop_146,])
output2 = relay.Tuple([bop_137,bop_143,bop_146,])
func_149 = relay.Function([var_142,], output)
mod['func_149'] = func_149
mod = relay.transform.InferType()(mod)
mutated_mod['func_149'] = func_149
mutated_mod = relay.transform.InferType()(mutated_mod)
var_150 = relay.var("var_150", dtype = "float32", shape = (6,))#candidate|150|(6,)|var|float32
func_149_call = mutated_mod.get_global_var('func_149')
call_151 = func_149_call(var_150)
output = call_151
func_152 = relay.Function([var_150], output)
mutated_mod['func_152'] = func_152
mutated_mod = relay.transform.InferType()(mutated_mod)
const_154 = relay.const([[0.087927,-9.915638,1.764391,-2.282517,-8.207174,-3.335785,-8.680351,8.437772],[5.721106,-7.162790,-4.118866,-7.690014,9.948840,-5.548962,5.902436,5.348798],[2.528221,-1.745202,4.579270,-2.750288,-0.586012,8.329739,-4.611148,1.328192],[-9.482927,-8.846747,-6.415001,-6.190280,-6.467347,4.916094,-3.752695,-9.685225],[-0.297376,-0.117776,-6.566787,-3.398294,3.321166,0.472500,-6.885147,9.123416]], dtype = "float64")#candidate|154|(5, 8)|const|float64
var_155 = relay.var("var_155", dtype = "float64", shape = (5, 8))#candidate|155|(5, 8)|var|float64
bop_156 = relay.less(const_154.astype('bool'), relay.reshape(var_155.astype('bool'), relay.shape_of(const_154))) # shape=(5, 8)
uop_159 = relay.sinh(const_154.astype('float64')) # shape=(5, 8)
bop_161 = relay.equal(uop_159.astype('bool'), relay.reshape(const_154.astype('bool'), relay.shape_of(uop_159))) # shape=(5, 8)
uop_164 = relay.sigmoid(uop_159.astype('float64')) # shape=(5, 8)
uop_166 = relay.acosh(uop_164.astype('float32')) # shape=(5, 8)
output = relay.Tuple([bop_156,bop_161,uop_166,])
output2 = relay.Tuple([bop_156,bop_161,uop_166,])
func_168 = relay.Function([var_155,], output)
mod['func_168'] = func_168
mod = relay.transform.InferType()(mod)
mutated_mod['func_168'] = func_168
mutated_mod = relay.transform.InferType()(mutated_mod)
var_169 = relay.var("var_169", dtype = "float64", shape = (5, 8))#candidate|169|(5, 8)|var|float64
func_168_call = mutated_mod.get_global_var('func_168')
call_170 = func_168_call(var_169)
output = call_170
func_171 = relay.Function([var_169], output)
mutated_mod['func_171'] = func_171
mutated_mod = relay.transform.InferType()(mutated_mod)
const_173 = relay.const([[[-5,1]],[[-8,5]],[[-9,5]],[[5,7]],[[7,2]],[[4,-6]],[[6,-9]],[[5,8]],[[10,3]],[[8,7]],[[7,7]]], dtype = "int8")#candidate|173|(11, 1, 2)|const|int8
var_174 = relay.var("var_174", dtype = "int8", shape = (11, 13, 2))#candidate|174|(11, 13, 2)|var|int8
bop_175 = relay.less(const_173.astype('bool'), var_174.astype('bool')) # shape=(11, 13, 2)
bop_178 = relay.greater(bop_175.astype('bool'), relay.reshape(var_174.astype('bool'), relay.shape_of(bop_175))) # shape=(11, 13, 2)
uop_181 = relay.sqrt(bop_175.astype('float64')) # shape=(11, 13, 2)
bop_183 = relay.logical_xor(bop_178.astype('uint64'), const_173.astype('uint64')) # shape=(11, 13, 2)
uop_186 = relay.acosh(bop_183.astype('float32')) # shape=(11, 13, 2)
bop_188 = relay.not_equal(uop_181.astype('bool'), relay.reshape(bop_175.astype('bool'), relay.shape_of(uop_181))) # shape=(11, 13, 2)
bop_191 = relay.divide(bop_188.astype('float32'), relay.reshape(bop_178.astype('float32'), relay.shape_of(bop_188))) # shape=(11, 13, 2)
bop_194 = relay.subtract(bop_175.astype('uint32'), relay.reshape(bop_191.astype('uint32'), relay.shape_of(bop_175))) # shape=(11, 13, 2)
var_197 = relay.var("var_197", dtype = "bool", shape = (11, 13, 2))#candidate|197|(11, 13, 2)|var|bool
bop_198 = relay.logical_or(bop_178.astype('bool'), relay.reshape(var_197.astype('bool'), relay.shape_of(bop_178))) # shape=(11, 13, 2)
bop_201 = relay.floor_mod(bop_194.astype('float32'), relay.reshape(bop_175.astype('float32'), relay.shape_of(bop_194))) # shape=(11, 13, 2)
var_204 = relay.var("var_204", dtype = "float32", shape = (11, 13, 2))#candidate|204|(11, 13, 2)|var|float32
bop_205 = relay.bitwise_xor(bop_191.astype('int64'), relay.reshape(var_204.astype('int64'), relay.shape_of(bop_191))) # shape=(11, 13, 2)
uop_208 = relay.sqrt(bop_188.astype('float32')) # shape=(11, 13, 2)
bop_210 = relay.subtract(uop_186.astype('float64'), relay.reshape(bop_198.astype('float64'), relay.shape_of(uop_186))) # shape=(11, 13, 2)
bop_213 = relay.right_shift(uop_208.astype('int16'), relay.reshape(bop_175.astype('int16'), relay.shape_of(uop_208))) # shape=(11, 13, 2)
uop_216 = relay.log2(uop_186.astype('float64')) # shape=(11, 13, 2)
bop_218 = relay.floor_mod(bop_210.astype('float32'), relay.reshape(uop_216.astype('float32'), relay.shape_of(bop_210))) # shape=(11, 13, 2)
var_221 = relay.var("var_221", dtype = "bool", shape = (11, 13, 2))#candidate|221|(11, 13, 2)|var|bool
bop_222 = relay.mod(bop_188.astype('float64'), relay.reshape(var_221.astype('float64'), relay.shape_of(bop_188))) # shape=(11, 13, 2)
bop_225 = relay.floor_mod(bop_210.astype('float32'), relay.reshape(bop_183.astype('float32'), relay.shape_of(bop_210))) # shape=(11, 13, 2)
var_228 = relay.var("var_228", dtype = "float64", shape = (11, 13, 2))#candidate|228|(11, 13, 2)|var|float64
bop_229 = relay.logical_and(bop_210.astype('bool'), relay.reshape(var_228.astype('bool'), relay.shape_of(bop_210))) # shape=(11, 13, 2)
var_232 = relay.var("var_232", dtype = "float32", shape = (11, 13, 2))#candidate|232|(11, 13, 2)|var|float32
bop_233 = relay.logical_xor(bop_218.astype('uint64'), relay.reshape(var_232.astype('uint64'), relay.shape_of(bop_218))) # shape=(11, 13, 2)
bop_236 = relay.not_equal(bop_198.astype('bool'), relay.reshape(bop_222.astype('bool'), relay.shape_of(bop_198))) # shape=(11, 13, 2)
uop_239 = relay.asinh(bop_178.astype('float64')) # shape=(11, 13, 2)
uop_241 = relay.rsqrt(uop_216.astype('float32')) # shape=(11, 13, 2)
bop_243 = relay.bitwise_xor(uop_208.astype('int64'), relay.reshape(bop_191.astype('int64'), relay.shape_of(uop_208))) # shape=(11, 13, 2)
uop_246 = relay.atanh(uop_241.astype('float32')) # shape=(11, 13, 2)
var_248 = relay.var("var_248", dtype = "float32", shape = (11, 13, 2))#candidate|248|(11, 13, 2)|var|float32
bop_249 = relay.floor_divide(uop_241.astype('float32'), relay.reshape(var_248.astype('float32'), relay.shape_of(uop_241))) # shape=(11, 13, 2)
bop_252 = relay.right_shift(uop_241.astype('int16'), relay.reshape(bop_249.astype('int16'), relay.shape_of(uop_241))) # shape=(11, 13, 2)
bop_255 = relay.logical_xor(bop_252.astype('int8'), relay.reshape(var_204.astype('int8'), relay.shape_of(bop_252))) # shape=(11, 13, 2)
bop_258 = relay.not_equal(uop_246.astype('bool'), relay.reshape(uop_186.astype('bool'), relay.shape_of(uop_246))) # shape=(11, 13, 2)
uop_261 = relay.sinh(uop_246.astype('float64')) # shape=(11, 13, 2)
var_263 = relay.var("var_263", dtype = "float32", shape = (11, 13, 2))#candidate|263|(11, 13, 2)|var|float32
bop_264 = relay.bitwise_and(uop_246.astype('uint16'), relay.reshape(var_263.astype('uint16'), relay.shape_of(uop_246))) # shape=(11, 13, 2)
bop_267 = relay.multiply(uop_241.astype('float64'), relay.reshape(bop_191.astype('float64'), relay.shape_of(uop_241))) # shape=(11, 13, 2)
uop_270 = relay.cos(uop_261.astype('float32')) # shape=(11, 13, 2)
uop_272 = relay.asin(bop_252.astype('float64')) # shape=(11, 13, 2)
bop_274 = relay.greater(uop_270.astype('bool'), relay.reshape(bop_229.astype('bool'), relay.shape_of(uop_270))) # shape=(11, 13, 2)
bop_277 = relay.bitwise_and(bop_267.astype('int32'), relay.reshape(bop_210.astype('int32'), relay.shape_of(bop_267))) # shape=(11, 13, 2)
bop_280 = relay.floor_divide(bop_264.astype('float64'), relay.reshape(uop_272.astype('float64'), relay.shape_of(bop_264))) # shape=(11, 13, 2)
bop_283 = relay.logical_xor(bop_274.astype('uint64'), relay.reshape(bop_188.astype('uint64'), relay.shape_of(bop_274))) # shape=(11, 13, 2)
bop_286 = relay.floor_mod(uop_272.astype('float32'), relay.reshape(bop_267.astype('float32'), relay.shape_of(uop_272))) # shape=(11, 13, 2)
bop_289 = relay.greater_equal(bop_274.astype('bool'), relay.reshape(var_197.astype('bool'), relay.shape_of(bop_274))) # shape=(11, 13, 2)
bop_292 = relay.divide(bop_274.astype('float64'), relay.reshape(bop_188.astype('float64'), relay.shape_of(bop_274))) # shape=(11, 13, 2)
uop_295 = relay.asinh(var_221.astype('float32')) # shape=(11, 13, 2)
uop_297 = relay.log10(bop_274.astype('float32')) # shape=(11, 13, 2)
var_299 = relay.var("var_299", dtype = "float64", shape = (11, 13, 2))#candidate|299|(11, 13, 2)|var|float64
bop_300 = relay.maximum(bop_292.astype('uint16'), relay.reshape(var_299.astype('uint16'), relay.shape_of(bop_292))) # shape=(11, 13, 2)
uop_303 = relay.cos(uop_297.astype('float64')) # shape=(11, 13, 2)
var_305 = relay.var("var_305", dtype = "float32", shape = (11, 13, 2))#candidate|305|(11, 13, 2)|var|float32
bop_306 = relay.power(uop_297.astype('float64'), relay.reshape(var_305.astype('float64'), relay.shape_of(uop_297))) # shape=(11, 13, 2)
uop_309 = relay.rsqrt(uop_297.astype('float64')) # shape=(11, 13, 2)
const_311 = relay.const([[[-5.782042,-6.047730],[7.486760,-5.319946],[9.812953,7.635352],[-6.399350,4.675716],[-6.186366,-8.897587],[-3.343520,-9.243394],[2.141354,-2.688339],[-0.597903,8.093890],[-2.208987,1.360476],[-6.609492,-7.069087],[6.128707,-6.632783],[8.790612,6.009466],[6.835839,-3.419008]],[[-9.360674,2.719208],[1.172087,3.561384],[3.570326,6.526256],[-1.489382,5.357793],[-2.145984,-3.790965],[5.975040,3.861135],[7.220555,-5.434101],[8.664002,-6.968910],[-6.896975,0.154334],[-5.947007,2.613344],[-5.548011,7.913822],[-0.038447,5.204939],[-6.856013,-0.378541]],[[9.838386,-6.147906],[-6.438658,5.031648],[3.762249,4.356945],[0.530927,7.072481],[4.142580,-4.805751],[7.416550,7.625010],[-4.617890,6.995236],[-8.525656,9.924623],[8.331083,-1.099636],[-5.633721,4.819819],[-2.174075,7.277242],[-0.845409,2.852708],[-5.310403,0.091608]],[[7.715216,6.417207],[2.453236,-5.320917],[-9.405609,3.875374],[7.369739,6.255157],[-8.950132,3.848918],[0.193418,-7.178698],[1.690733,0.983912],[1.818164,-2.955083],[1.514512,8.359930],[-0.166916,5.823361],[4.542956,-3.092265],[-2.725572,-0.208342],[6.997843,6.796766]],[[9.036607,-1.512275],[6.786092,-5.473328],[-5.823193,4.202557],[-4.476672,-8.633716],[-0.008840,7.131735],[2.264112,-0.790132],[7.595741,-1.798244],[-0.722070,4.565167],[1.573795,8.734580],[2.784141,-7.007503],[-8.626934,-7.288797],[-0.742204,4.540442],[-6.774721,-4.738106]],[[-4.187966,4.469098],[4.257491,-7.875271],[-0.466554,0.765957],[5.871380,-4.870296],[-2.779937,-0.138519],[3.735736,9.272109],[-3.992080,-1.436779],[5.575143,-0.076368],[-0.066351,9.923940],[4.534107,3.486153],[7.243404,3.264733],[8.822581,-3.487745],[-8.792744,-6.284845]],[[4.388803,1.287260],[-3.062443,-9.400675],[-1.407872,-6.292393],[-5.994129,0.003627],[6.326908,1.265742],[6.316793,-4.273383],[1.716604,6.589560],[5.024353,9.554424],[9.937313,-8.360736],[-2.188948,-6.967869],[4.917817,1.785337],[-0.911435,6.909308],[-4.197779,6.365538]],[[-1.042763,-8.362861],[-6.276188,0.753634],[-4.682529,-1.833286],[-1.700469,7.805219],[8.207832,7.916251],[-6.482689,-9.519685],[2.140850,4.560390],[8.973773,-1.129518],[-9.650457,2.282424],[-3.262167,-2.299982],[7.758335,-9.396950],[-6.330846,-3.288364],[6.830362,-8.021946]],[[0.732900,6.332773],[0.876341,-5.832828],[1.213352,-0.606292],[7.338783,-1.221830],[-1.877829,-2.058821],[5.423681,5.871106],[7.425861,5.807923],[-1.121547,-7.644551],[-7.859143,-9.569723],[-7.053466,-3.654096],[1.529346,3.762595],[-8.757530,-7.319352],[9.427009,-8.034058]],[[-4.696129,-8.905771],[-1.530592,-6.619314],[8.249383,-1.447048],[-4.314404,4.107963],[1.962863,-8.808994],[-0.804031,2.369856],[4.663629,-5.498886],[1.932143,7.324344],[-2.670203,2.253354],[6.002195,5.639838],[-6.687622,0.839497],[3.155160,-0.869382],[-2.165038,8.549590]],[[-1.019900,-4.131743],[2.118533,-9.879101],[4.058118,-2.207324],[8.920590,-9.394541],[3.261502,-3.177411],[6.633987,2.133860],[-2.645991,2.243563],[-0.284209,-8.180321],[-9.904104,5.603500],[8.405142,-1.021106],[2.458718,-3.477066],[-0.195309,-3.172108],[0.198846,4.222531]]], dtype = "float32")#candidate|311|(11, 13, 2)|const|float32
bop_312 = relay.add(uop_186.astype('uint16'), relay.reshape(const_311.astype('uint16'), relay.shape_of(uop_186))) # shape=(11, 13, 2)
bop_315 = relay.divide(uop_297.astype('float32'), relay.reshape(bop_274.astype('float32'), relay.shape_of(uop_297))) # shape=(11, 13, 2)
uop_318 = relay.atan(bop_289.astype('float64')) # shape=(11, 13, 2)
bop_320 = relay.floor_mod(uop_303.astype('float64'), relay.reshape(var_228.astype('float64'), relay.shape_of(uop_303))) # shape=(11, 13, 2)
bop_323 = relay.divide(bop_320.astype('float64'), relay.reshape(uop_186.astype('float64'), relay.shape_of(bop_320))) # shape=(11, 13, 2)
bop_326 = relay.power(uop_309.astype('float64'), relay.reshape(bop_236.astype('float64'), relay.shape_of(uop_309))) # shape=(11, 13, 2)
output = relay.Tuple([bop_201,bop_205,bop_213,bop_225,bop_233,uop_239,bop_243,bop_255,bop_258,bop_277,bop_280,bop_283,bop_286,uop_295,bop_300,bop_306,bop_312,bop_315,uop_318,bop_323,bop_326,])
output2 = relay.Tuple([bop_201,bop_205,bop_213,bop_225,bop_233,uop_239,bop_243,bop_255,bop_258,bop_277,bop_280,bop_283,bop_286,uop_295,bop_300,bop_306,bop_312,bop_315,uop_318,bop_323,bop_326,])
func_329 = relay.Function([var_174,var_197,var_204,var_221,var_228,var_232,var_248,var_263,var_299,var_305,], output)
mod['func_329'] = func_329
mod = relay.transform.InferType()(mod)
mutated_mod['func_329'] = func_329
mutated_mod = relay.transform.InferType()(mutated_mod)
func_329_call = mutated_mod.get_global_var('func_329')
var_331 = relay.var("var_331", dtype = "int8", shape = (11, 13, 2))#candidate|331|(11, 13, 2)|var|int8
var_332 = relay.var("var_332", dtype = "bool", shape = (11, 13, 2))#candidate|332|(11, 13, 2)|var|bool
var_333 = relay.var("var_333", dtype = "float32", shape = (11, 13, 2))#candidate|333|(11, 13, 2)|var|float32
var_334 = relay.var("var_334", dtype = "bool", shape = (11, 13, 2))#candidate|334|(11, 13, 2)|var|bool
var_335 = relay.var("var_335", dtype = "float64", shape = (11, 13, 2))#candidate|335|(11, 13, 2)|var|float64
var_336 = relay.var("var_336", dtype = "float32", shape = (11, 13, 2))#candidate|336|(11, 13, 2)|var|float32
var_337 = relay.var("var_337", dtype = "float32", shape = (11, 13, 2))#candidate|337|(11, 13, 2)|var|float32
var_338 = relay.var("var_338", dtype = "float32", shape = (11, 13, 2))#candidate|338|(11, 13, 2)|var|float32
var_339 = relay.var("var_339", dtype = "float64", shape = (11, 13, 2))#candidate|339|(11, 13, 2)|var|float64
var_340 = relay.var("var_340", dtype = "float32", shape = (11, 13, 2))#candidate|340|(11, 13, 2)|var|float32
call_330 = func_329_call(var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,)
output = call_330
func_341 = relay.Function([var_331,var_332,var_333,var_334,var_335,var_336,var_337,var_338,var_339,var_340,], output)
mutated_mod['func_341'] = func_341
mutated_mod = relay.transform.InferType()(mutated_mod)
const_343 = relay.const(-1, dtype = "int64")#candidate|343|()|const|int64
var_344 = relay.var("var_344", dtype = "int64", shape = (9, 7, 5))#candidate|344|(9, 7, 5)|var|int64
bop_345 = relay.left_shift(const_343.astype('int64'), var_344.astype('int64')) # shape=(9, 7, 5)
output = bop_345
output2 = bop_345
F = relay.Function([var_344,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_344,], output2)
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
input_344= np.array([[[-9,9,-3,10,-5],[-7,3,6,-7,-6],[-8,5,10,5,4],[-1,7,2,-4,-5],[-1,-2,6,4,4],[-4,-8,2,4,6],[-7,3,-10,1,-9]],[[-1,-9,-5,-2,-8],[1,-1,-10,-7,-10],[4,-1,-1,-2,-2],[-9,1,7,4,-7],[8,-4,-4,-9,1],[-3,7,10,4,-10],[-6,1,-8,-3,10]],[[5,-7,4,-5,2],[2,-9,-1,7,-4],[4,-6,7,-7,-10],[-2,2,4,-1,-1],[-5,8,2,-4,5],[-5,-7,-5,-7,-5],[-9,2,6,-9,6]],[[-8,3,-6,5,3],[7,-10,9,9,10],[-4,10,-2,3,3],[-1,2,-7,8,-10],[4,9,2,-1,-3],[-6,2,3,4,5],[-9,8,-10,-2,9]],[[-1,7,-6,7,1],[-10,7,3,-4,-5],[-4,-5,6,7,8],[-9,-8,-1,7,-4],[6,-7,6,2,-9],[-10,3,-5,-3,-8],[8,-1,6,-3,-8]],[[3,-2,8,10,-9],[8,2,-5,10,-7],[10,-8,4,-1,-6],[5,-4,8,7,1],[-7,-5,8,-3,-9],[-1,-8,8,-10,8],[-9,-5,-1,4,-1]],[[4,-1,6,-1,2],[-9,-7,-9,-8,5],[6,10,6,4,-1],[-4,-2,6,-6,7],[6,-10,-3,-9,9],[9,-9,-7,-2,6],[6,10,7,-6,-4]],[[-3,-8,-5,-2,-4],[-10,6,-5,6,-6],[1,-4,-9,-9,-4],[-7,3,-9,-6,3],[-10,4,2,-1,-7],[1,-8,-2,-4,2],[-4,-6,-4,-5,4]],[[2,-1,-3,8,-9],[2,4,5,-9,-7],[-8,1,9,-8,8],[4,-7,-8,10,-4],[-10,1,6,-2,-4],[7,-9,5,-4,4],[-5,1,3,-4,-5]]], dtype='int64')
module1.set_input('var_344', input_344)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_344, )
res3 = intrp3.evaluate()(input_344, )
res4 = intrp4.evaluate()(input_344, )
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
module5.set_input('var_344', input_344)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_344, )
res7 = intrp7.evaluate()(input_344, )
res8 = intrp8.evaluate()(input_344, )
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
module9.set_input('var_344', input_344)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_344, )
res11 = intrp11.evaluate()(input_344, )
res12 = intrp12.evaluate()(input_344, )
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
module13.set_input('var_344', input_344)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_344, )
res15 = intrp15.evaluate()(input_344, )
res16 = intrp16.evaluate()(input_344, )
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
module17.set_input('var_344', input_344)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_344, )
res19 = intrp19.evaluate()(input_344, )
res20 = intrp20.evaluate()(input_344, )
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
module21.set_input('var_344', input_344)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_344, )
res23 = intrp23.evaluate()(input_344, )
res24 = intrp24.evaluate()(input_344, )
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

'''0,                -1024,

'''