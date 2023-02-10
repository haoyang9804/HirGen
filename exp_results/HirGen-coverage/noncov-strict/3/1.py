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
var_0 = relay.var("var_0", dtype = "float32", shape = (4,))#candidate|0|(4,)|var|float32
uop_1 = relay.acos(var_0.astype('float32')) # shape=(4,)
var_3 = relay.var("var_3", dtype = "float32", shape = (4,))#candidate|3|(4,)|var|float32
bop_4 = relay.floor_mod(uop_1.astype('float32'), relay.reshape(var_3.astype('float32'), relay.shape_of(uop_1))) # shape=(4,)
bop_7 = relay.greater(uop_1.astype('bool'), relay.reshape(var_3.astype('bool'), relay.shape_of(uop_1))) # shape=(4,)
uop_10 = relay.tan(var_3.astype('float64')) # shape=(4,)
bop_12 = relay.power(var_3.astype('float32'), relay.reshape(uop_1.astype('float32'), relay.shape_of(var_3))) # shape=(4,)
uop_15 = relay.sinh(var_0.astype('float32')) # shape=(4,)
uop_17 = relay.asin(bop_7.astype('float64')) # shape=(4,)
uop_19 = relay.rsqrt(uop_17.astype('float64')) # shape=(4,)
output = relay.Tuple([bop_4,uop_10,bop_12,uop_15,uop_19,])
output2 = relay.Tuple([bop_4,uop_10,bop_12,uop_15,uop_19,])
func_21 = relay.Function([var_0,var_3,], output)
mod['func_21'] = func_21
mod = relay.transform.InferType()(mod)
var_22 = relay.var("var_22", dtype = "float32", shape = (4,))#candidate|22|(4,)|var|float32
var_23 = relay.var("var_23", dtype = "float32", shape = (4,))#candidate|23|(4,)|var|float32
output = func_21(var_22,var_23,)
func_24 = relay.Function([var_22,var_23,], output)
mutated_mod['func_24'] = func_24
mutated_mod = relay.transform.InferType()(mutated_mod)
var_26 = relay.var("var_26", dtype = "float64", shape = (1, 6))#candidate|26|(1, 6)|var|float64
uop_27 = relay.acosh(var_26.astype('float64')) # shape=(1, 6)
var_29 = relay.var("var_29", dtype = "float64", shape = (6, 6))#candidate|29|(6, 6)|var|float64
bop_30 = relay.divide(var_26.astype('float64'), var_29.astype('float64')) # shape=(6, 6)
bop_33 = relay.greater_equal(uop_27.astype('bool'), bop_30.astype('bool')) # shape=(6, 6)
bop_36 = relay.divide(bop_33.astype('float32'), var_26.astype('float32')) # shape=(6, 6)
var_39 = relay.var("var_39", dtype = "float64", shape = (2, 6))#candidate|39|(2, 6)|var|float64
bop_40 = relay.less(uop_27.astype('bool'), var_39.astype('bool')) # shape=(2, 6)
bop_43 = relay.bitwise_or(bop_40.astype('uint8'), var_26.astype('uint8')) # shape=(2, 6)
bop_46 = relay.floor_mod(bop_33.astype('float64'), uop_27.astype('float64')) # shape=(6, 6)
var_49 = relay.var("var_49", dtype = "float32", shape = (6, 6))#candidate|49|(6, 6)|var|float32
bop_50 = relay.left_shift(bop_36.astype('int16'), relay.reshape(var_49.astype('int16'), relay.shape_of(bop_36))) # shape=(6, 6)
uop_53 = relay.exp(var_39.astype('float32')) # shape=(2, 6)
bop_55 = relay.mod(var_26.astype('float64'), bop_30.astype('float64')) # shape=(6, 6)
uop_58 = relay.rsqrt(bop_33.astype('float32')) # shape=(6, 6)
bop_60 = relay.maximum(uop_53.astype('int32'), relay.reshape(var_39.astype('int32'), relay.shape_of(uop_53))) # shape=(2, 6)
bop_63 = relay.greater(bop_60.astype('bool'), relay.reshape(bop_40.astype('bool'), relay.shape_of(bop_60))) # shape=(2, 6)
uop_66 = relay.sin(uop_58.astype('float32')) # shape=(6, 6)
uop_68 = relay.erf(uop_66.astype('float32')) # shape=(6, 6)
uop_70 = relay.cos(uop_68.astype('float32')) # shape=(6, 6)
bop_72 = relay.divide(uop_70.astype('float64'), var_26.astype('float64')) # shape=(6, 6)
bop_75 = relay.minimum(uop_68.astype('uint8'), relay.reshape(uop_58.astype('uint8'), relay.shape_of(uop_68))) # shape=(6, 6)
uop_78 = relay.sinh(bop_72.astype('float64')) # shape=(6, 6)
uop_80 = relay.acosh(uop_78.astype('float32')) # shape=(6, 6)
bop_82 = relay.greater(uop_78.astype('bool'), relay.reshape(bop_50.astype('bool'), relay.shape_of(uop_78))) # shape=(6, 6)
var_85 = relay.var("var_85", dtype = "float32", shape = (6, 6))#candidate|85|(6, 6)|var|float32
bop_86 = relay.left_shift(uop_80.astype('int32'), relay.reshape(var_85.astype('int32'), relay.shape_of(uop_80))) # shape=(6, 6)
uop_89 = relay.cos(uop_70.astype('float64')) # shape=(6, 6)
uop_91 = relay.atanh(uop_78.astype('float64')) # shape=(6, 6)
bop_93 = relay.less(uop_78.astype('bool'), relay.reshape(uop_68.astype('bool'), relay.shape_of(uop_78))) # shape=(6, 6)
uop_96 = relay.erf(bop_86.astype('float32')) # shape=(6, 6)
uop_98 = relay.log10(uop_89.astype('float64')) # shape=(6, 6)
const_100 = relay.const([[-8.898288,2.498500,2.541057,-7.988224,-6.142132,-7.794844],[-0.370570,-3.255451,0.502877,1.934156,6.982689,-4.859288],[-1.853629,-1.487699,-8.594250,-2.373317,6.440113,-1.438513],[4.541511,4.629890,-6.309281,1.831479,1.757588,1.363725],[2.403626,7.007373,-6.957602,-4.024911,8.483698,-1.389581],[-2.579462,-7.079660,-2.754330,9.390270,-2.399027,4.921989]], dtype = "float32")#candidate|100|(6, 6)|const|float32
bop_101 = relay.maximum(uop_70.astype('float32'), relay.reshape(const_100.astype('float32'), relay.shape_of(uop_70))) # shape=(6, 6)
uop_104 = relay.sinh(bop_86.astype('float64')) # shape=(6, 6)
uop_106 = relay.acos(bop_101.astype('float32')) # shape=(6, 6)
uop_108 = relay.atanh(bop_75.astype('float64')) # shape=(6, 6)
output = relay.Tuple([bop_43,bop_46,bop_55,bop_63,bop_82,uop_91,bop_93,uop_96,uop_98,uop_104,uop_106,uop_108,])
output2 = relay.Tuple([bop_43,bop_46,bop_55,bop_63,bop_82,uop_91,bop_93,uop_96,uop_98,uop_104,uop_106,uop_108,])
func_110 = relay.Function([var_26,var_29,var_39,var_49,var_85,], output)
mod['func_110'] = func_110
mod = relay.transform.InferType()(mod)
var_111 = relay.var("var_111", dtype = "float64", shape = (1, 6))#candidate|111|(1, 6)|var|float64
var_112 = relay.var("var_112", dtype = "float64", shape = (6, 6))#candidate|112|(6, 6)|var|float64
var_113 = relay.var("var_113", dtype = "float64", shape = (2, 6))#candidate|113|(2, 6)|var|float64
var_114 = relay.var("var_114", dtype = "float32", shape = (6, 6))#candidate|114|(6, 6)|var|float32
var_115 = relay.var("var_115", dtype = "float32", shape = (6, 6))#candidate|115|(6, 6)|var|float32
output = func_110(var_111,var_112,var_113,var_114,var_115,)
func_116 = relay.Function([var_111,var_112,var_113,var_114,var_115,], output)
mutated_mod['func_116'] = func_116
mutated_mod = relay.transform.InferType()(mutated_mod)
var_118 = relay.var("var_118", dtype = "float64", shape = (16,))#candidate|118|(16,)|var|float64
uop_119 = relay.rsqrt(var_118.astype('float64')) # shape=(16,)
uop_121 = relay.log10(uop_119.astype('float32')) # shape=(16,)
uop_123 = relay.exp(uop_121.astype('float32')) # shape=(16,)
func_110_call = mod.get_global_var('func_110')
func_116_call = mutated_mod.get_global_var('func_116')
const_126 = relay.const([-5.184957,4.898999,-8.304918,-9.330661,9.771532,5.294607], dtype = "float64")#candidate|126|(6,)|const|float64
var_127 = relay.var("var_127", dtype = "float64", shape = (36,))#candidate|127|(36,)|var|float64
var_128 = relay.var("var_128", dtype = "float64", shape = (12,))#candidate|128|(12,)|var|float64
call_125 = relay.TupleGetItem(func_110_call(relay.reshape(const_126.astype('float64'), [1, 6]), relay.reshape(var_127.astype('float64'), [6, 6]), relay.reshape(var_128.astype('float64'), [2, 6]), relay.reshape(var_127.astype('float32'), [6, 6]), relay.reshape(var_127.astype('float32'), [6, 6]), ), 4)
call_129 = relay.TupleGetItem(func_116_call(relay.reshape(const_126.astype('float64'), [1, 6]), relay.reshape(var_127.astype('float64'), [6, 6]), relay.reshape(var_128.astype('float64'), [2, 6]), relay.reshape(var_127.astype('float32'), [6, 6]), relay.reshape(var_127.astype('float32'), [6, 6]), ), 4)
bop_130 = relay.power(uop_123.astype('float32'), relay.reshape(var_118.astype('float32'), relay.shape_of(uop_123))) # shape=(16,)
bop_133 = relay.greater_equal(bop_130.astype('bool'), relay.reshape(uop_119.astype('bool'), relay.shape_of(bop_130))) # shape=(16,)
bop_136 = relay.multiply(bop_133.astype('int64'), relay.reshape(uop_123.astype('int64'), relay.shape_of(bop_133))) # shape=(16,)
bop_139 = relay.equal(bop_130.astype('bool'), relay.reshape(var_118.astype('bool'), relay.shape_of(bop_130))) # shape=(16,)
output = relay.Tuple([call_125,const_126,var_127,var_128,bop_136,bop_139,])
output2 = relay.Tuple([call_129,const_126,var_127,var_128,bop_136,bop_139,])
func_142 = relay.Function([var_118,var_127,var_128,], output)
mod['func_142'] = func_142
mod = relay.transform.InferType()(mod)
mutated_mod['func_142'] = func_142
mutated_mod = relay.transform.InferType()(mutated_mod)
func_142_call = mutated_mod.get_global_var('func_142')
var_144 = relay.var("var_144", dtype = "float64", shape = (16,))#candidate|144|(16,)|var|float64
var_145 = relay.var("var_145", dtype = "float64", shape = (36,))#candidate|145|(36,)|var|float64
var_146 = relay.var("var_146", dtype = "float64", shape = (12,))#candidate|146|(12,)|var|float64
call_143 = func_142_call(var_144,var_145,var_146,)
output = call_143
func_147 = relay.Function([var_144,var_145,var_146,], output)
mutated_mod['func_147'] = func_147
mutated_mod = relay.transform.InferType()(mutated_mod)
const_149 = relay.const(-5, dtype = "int16")#candidate|149|()|const|int16
var_150 = relay.var("var_150", dtype = "int16", shape = (8, 14))#candidate|150|(8, 14)|var|int16
bop_151 = relay.less_equal(const_149.astype('bool'), var_150.astype('bool')) # shape=(8, 14)
output = bop_151
output2 = bop_151
func_154 = relay.Function([var_150,], output)
mod['func_154'] = func_154
mod = relay.transform.InferType()(mod)
mutated_mod['func_154'] = func_154
mutated_mod = relay.transform.InferType()(mutated_mod)
var_155 = relay.var("var_155", dtype = "int16", shape = (8, 14))#candidate|155|(8, 14)|var|int16
func_154_call = mutated_mod.get_global_var('func_154')
call_156 = func_154_call(var_155)
output = call_156
func_157 = relay.Function([var_155], output)
mutated_mod['func_157'] = func_157
mutated_mod = relay.transform.InferType()(mutated_mod)
var_159 = relay.var("var_159", dtype = "float32", shape = (2, 7, 12))#candidate|159|(2, 7, 12)|var|float32
var_160 = relay.var("var_160", dtype = "float32", shape = (2, 7, 12))#candidate|160|(2, 7, 12)|var|float32
bop_161 = relay.floor_mod(var_159.astype('float32'), relay.reshape(var_160.astype('float32'), relay.shape_of(var_159))) # shape=(2, 7, 12)
bop_164 = relay.bitwise_or(var_159.astype('uint8'), relay.reshape(var_160.astype('uint8'), relay.shape_of(var_159))) # shape=(2, 7, 12)
uop_167 = relay.exp(var_159.astype('float64')) # shape=(2, 7, 12)
bop_169 = relay.power(uop_167.astype('float32'), relay.reshape(bop_161.astype('float32'), relay.shape_of(uop_167))) # shape=(2, 7, 12)
bop_172 = relay.mod(bop_169.astype('float32'), relay.reshape(bop_161.astype('float32'), relay.shape_of(bop_169))) # shape=(2, 7, 12)
bop_175 = relay.bitwise_xor(var_159.astype('int32'), relay.reshape(var_160.astype('int32'), relay.shape_of(var_159))) # shape=(2, 7, 12)
bop_178 = relay.multiply(uop_167.astype('uint16'), relay.reshape(bop_172.astype('uint16'), relay.shape_of(uop_167))) # shape=(2, 7, 12)
bop_181 = relay.greater(bop_164.astype('bool'), relay.reshape(uop_167.astype('bool'), relay.shape_of(bop_164))) # shape=(2, 7, 12)
uop_184 = relay.asinh(bop_169.astype('float64')) # shape=(2, 7, 12)
var_186 = relay.var("var_186", dtype = "float64", shape = (2, 7, 12))#candidate|186|(2, 7, 12)|var|float64
bop_187 = relay.equal(uop_184.astype('bool'), relay.reshape(var_186.astype('bool'), relay.shape_of(uop_184))) # shape=(2, 7, 12)
var_190 = relay.var("var_190", dtype = "bool", shape = (2, 7, 12))#candidate|190|(2, 7, 12)|var|bool
bop_191 = relay.power(bop_181.astype('float64'), relay.reshape(var_190.astype('float64'), relay.shape_of(bop_181))) # shape=(2, 7, 12)
bop_194 = relay.greater(bop_187.astype('bool'), relay.reshape(bop_169.astype('bool'), relay.shape_of(bop_187))) # shape=(2, 7, 12)
var_197 = relay.var("var_197", dtype = "float64", shape = (2, 7, 12))#candidate|197|(2, 7, 12)|var|float64
bop_198 = relay.bitwise_or(uop_184.astype('uint64'), relay.reshape(var_197.astype('uint64'), relay.shape_of(uop_184))) # shape=(2, 7, 12)
func_154_call = mod.get_global_var('func_154')
func_157_call = mutated_mod.get_global_var('func_157')
var_202 = relay.var("var_202", dtype = "int16", shape = (112,))#candidate|202|(112,)|var|int16
call_201 = func_154_call(relay.reshape(var_202.astype('int16'), [8, 14]))
call_203 = func_154_call(relay.reshape(var_202.astype('int16'), [8, 14]))
uop_204 = relay.sinh(bop_169.astype('float64')) # shape=(2, 7, 12)
var_206 = relay.var("var_206", dtype = "bool", shape = (2, 7, 12))#candidate|206|(2, 7, 12)|var|bool
bop_207 = relay.power(bop_187.astype('float32'), relay.reshape(var_206.astype('float32'), relay.shape_of(bop_187))) # shape=(2, 7, 12)
uop_210 = relay.rsqrt(bop_207.astype('float32')) # shape=(2, 7, 12)
uop_212 = relay.atanh(bop_198.astype('float64')) # shape=(2, 7, 12)
bop_214 = relay.greater(uop_210.astype('bool'), relay.reshape(bop_198.astype('bool'), relay.shape_of(uop_210))) # shape=(2, 7, 12)
uop_217 = relay.asin(uop_210.astype('float32')) # shape=(2, 7, 12)
var_219 = relay.var("var_219", dtype = "float32", shape = (2, 7, 12))#candidate|219|(2, 7, 12)|var|float32
bop_220 = relay.maximum(uop_210.astype('int64'), relay.reshape(var_219.astype('int64'), relay.shape_of(uop_210))) # shape=(2, 7, 12)
bop_223 = relay.greater(uop_212.astype('bool'), relay.reshape(uop_204.astype('bool'), relay.shape_of(uop_212))) # shape=(2, 7, 12)
uop_226 = relay.sinh(uop_217.astype('float32')) # shape=(2, 7, 12)
uop_228 = relay.sin(uop_217.astype('float64')) # shape=(2, 7, 12)
bop_230 = relay.subtract(bop_194.astype('uint16'), relay.reshape(bop_169.astype('uint16'), relay.shape_of(bop_194))) # shape=(2, 7, 12)
uop_233 = relay.sqrt(uop_217.astype('float32')) # shape=(2, 7, 12)
uop_235 = relay.sinh(uop_217.astype('float64')) # shape=(2, 7, 12)
bop_237 = relay.minimum(uop_228.astype('float64'), relay.reshape(bop_207.astype('float64'), relay.shape_of(uop_228))) # shape=(2, 7, 12)
uop_240 = relay.acos(uop_217.astype('float32')) # shape=(2, 7, 12)
bop_242 = relay.less(uop_217.astype('bool'), relay.reshape(var_219.astype('bool'), relay.shape_of(uop_217))) # shape=(2, 7, 12)
output = relay.Tuple([bop_175,bop_178,bop_191,call_201,var_202,bop_214,bop_220,bop_223,uop_226,bop_230,uop_233,uop_235,bop_237,uop_240,bop_242,])
output2 = relay.Tuple([bop_175,bop_178,bop_191,call_203,var_202,bop_214,bop_220,bop_223,uop_226,bop_230,uop_233,uop_235,bop_237,uop_240,bop_242,])
func_245 = relay.Function([var_159,var_160,var_186,var_190,var_197,var_202,var_206,var_219,], output)
mod['func_245'] = func_245
mod = relay.transform.InferType()(mod)
var_246 = relay.var("var_246", dtype = "float32", shape = (2, 7, 12))#candidate|246|(2, 7, 12)|var|float32
var_247 = relay.var("var_247", dtype = "float32", shape = (2, 7, 12))#candidate|247|(2, 7, 12)|var|float32
var_248 = relay.var("var_248", dtype = "float64", shape = (2, 7, 12))#candidate|248|(2, 7, 12)|var|float64
var_249 = relay.var("var_249", dtype = "bool", shape = (2, 7, 12))#candidate|249|(2, 7, 12)|var|bool
var_250 = relay.var("var_250", dtype = "float64", shape = (2, 7, 12))#candidate|250|(2, 7, 12)|var|float64
var_251 = relay.var("var_251", dtype = "int16", shape = (112,))#candidate|251|(112,)|var|int16
var_252 = relay.var("var_252", dtype = "bool", shape = (2, 7, 12))#candidate|252|(2, 7, 12)|var|bool
var_253 = relay.var("var_253", dtype = "float32", shape = (2, 7, 12))#candidate|253|(2, 7, 12)|var|float32
output = func_245(var_246,var_247,var_248,var_249,var_250,var_251,var_252,var_253,)
func_254 = relay.Function([var_246,var_247,var_248,var_249,var_250,var_251,var_252,var_253,], output)
mutated_mod['func_254'] = func_254
mutated_mod = relay.transform.InferType()(mutated_mod)
const_256 = relay.const([[-7.368360,-3.803457,-8.485743,8.123816,1.433387,4.626706,-2.014175,-3.590013,-8.156097,1.282789,-8.161397,-2.167362,-8.298134,-1.462241,-3.625041,4.289687],[4.485700,5.379428,-1.447001,-6.486897,6.121894,-3.155522,2.707988,-6.011025,-4.291565,0.745657,-8.848277,-1.852985,8.832002,9.859883,-9.943792,-6.455753],[-5.137641,2.643754,8.496032,-1.420489,-7.600279,-7.229665,-1.203278,9.950683,-0.635575,9.352574,-5.845554,-6.187988,5.170952,-3.380527,-7.539114,0.088044],[-8.437777,4.140182,-7.761864,2.308376,-6.538782,-9.860429,-1.994613,5.436956,-3.861051,6.729826,-6.368452,6.721961,2.590018,8.337312,-0.069188,-7.389610],[-7.227221,-5.650907,-6.857496,5.226434,2.061958,9.468961,9.493186,9.468680,2.425807,0.056585,-9.103759,4.035838,-5.365403,6.419889,4.337855,5.991797],[4.986712,-7.423636,0.483482,-3.647271,-2.152864,5.551046,-5.096217,2.929039,1.429079,2.965900,0.959389,-8.178181,8.886511,9.782968,0.143130,-2.101411],[-0.529978,-4.839673,-3.884874,7.691187,-8.659976,0.222202,4.694675,2.434355,3.635325,-8.197184,0.123127,-1.221266,-6.071497,-6.432232,-8.010343,-7.851596],[7.443778,9.738521,3.490076,-3.011941,6.177884,-2.483770,4.423852,4.440007,-0.818560,6.514542,-6.886231,-8.983677,2.525125,8.731601,-6.339372,4.022017],[-6.406621,6.917510,2.424161,-0.853061,2.273926,6.212222,-2.115917,9.903174,-6.626111,-9.457026,-4.207395,6.656941,8.930608,-1.027540,-9.402967,-3.450788],[-7.551852,-8.575769,1.353866,-4.102419,3.433286,4.028857,-1.764125,-6.957766,-6.842689,-7.749595,4.932390,-1.605536,-2.201812,-0.450560,-8.748948,4.762392],[4.228826,-4.096597,-4.361016,9.340256,-3.241086,3.562490,-6.734066,5.654658,3.501634,5.915608,3.424529,6.376059,-6.315681,-8.269267,-0.310971,3.002105]], dtype = "float32")#candidate|256|(11, 16)|const|float32
uop_257 = relay.acos(const_256.astype('float32')) # shape=(11, 16)
bop_259 = relay.bitwise_and(uop_257.astype('uint32'), relay.reshape(const_256.astype('uint32'), relay.shape_of(uop_257))) # shape=(11, 16)
func_154_call = mod.get_global_var('func_154')
func_157_call = mutated_mod.get_global_var('func_157')
const_263 = relay.const([-8,-1,4,4,-2,8,-2,3,2,-4,-5,-2,3,-3,9,-5,-5,-7,-10,4,6,1,-3,-3,8,-3,-6,-9,-6,-10,-10,-1,-1,-2,9,-3,10,-5,8,9,5,1,9,-8,2,-2,-5,7,-1,-7,7,-7,5,1,8,-1,3,4,-4,9,-4,-10,3,-3,8,-7,3,1,4,1,-1,-6,-4,10,-7,-6,1,-8,-3,9,2,-4,1,-2,9,-8,-6,7,-7,8,6,-4,9,-1,2,2,7,-10,-3,-3,6,3,8,-1,-6,8,4,1,2,4,2,-2], dtype = "int16")#candidate|263|(112,)|const|int16
call_262 = func_154_call(relay.reshape(const_263.astype('int16'), [8, 14]))
call_264 = func_154_call(relay.reshape(const_263.astype('int16'), [8, 14]))
bop_265 = relay.maximum(bop_259.astype('int64'), relay.reshape(uop_257.astype('int64'), relay.shape_of(bop_259))) # shape=(11, 16)
uop_268 = relay.log2(bop_259.astype('float32')) # shape=(11, 16)
bop_270 = relay.minimum(uop_257.astype('int8'), relay.reshape(bop_265.astype('int8'), relay.shape_of(uop_257))) # shape=(11, 16)
output = relay.Tuple([call_262,const_263,uop_268,bop_270,])
output2 = relay.Tuple([call_264,const_263,uop_268,bop_270,])
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
const_276 = relay.const([[-2.383210,9.670442]], dtype = "float64")#candidate|276|(1, 2)|const|float64
uop_277 = relay.log2(const_276.astype('float64')) # shape=(1, 2)
bop_279 = relay.subtract(uop_277.astype('float32'), relay.reshape(const_276.astype('float32'), relay.shape_of(uop_277))) # shape=(1, 2)
bop_282 = relay.logical_xor(uop_277.astype('uint32'), relay.reshape(bop_279.astype('uint32'), relay.shape_of(uop_277))) # shape=(1, 2)
bop_285 = relay.logical_and(uop_277.astype('bool'), relay.reshape(const_276.astype('bool'), relay.shape_of(uop_277))) # shape=(1, 2)
uop_288 = relay.acosh(bop_282.astype('float64')) # shape=(1, 2)
bop_290 = relay.logical_and(uop_288.astype('bool'), relay.reshape(bop_285.astype('bool'), relay.shape_of(uop_288))) # shape=(1, 2)
var_293 = relay.var("var_293", dtype = "bool", shape = (2, 2))#candidate|293|(2, 2)|var|bool
bop_294 = relay.floor_mod(bop_290.astype('float64'), var_293.astype('float64')) # shape=(2, 2)
bop_297 = relay.maximum(bop_279.astype('uint64'), relay.reshape(bop_290.astype('uint64'), relay.shape_of(bop_279))) # shape=(1, 2)
output = relay.Tuple([bop_294,bop_297,])
output2 = relay.Tuple([bop_294,bop_297,])
func_300 = relay.Function([var_293,], output)
mod['func_300'] = func_300
mod = relay.transform.InferType()(mod)
mutated_mod['func_300'] = func_300
mutated_mod = relay.transform.InferType()(mutated_mod)
var_301 = relay.var("var_301", dtype = "bool", shape = (2, 2))#candidate|301|(2, 2)|var|bool
func_300_call = mutated_mod.get_global_var('func_300')
call_302 = func_300_call(var_301)
output = call_302
func_303 = relay.Function([var_301], output)
mutated_mod['func_303'] = func_303
mutated_mod = relay.transform.InferType()(mutated_mod)
var_305 = relay.var("var_305", dtype = "int8", shape = (5, 1, 2))#candidate|305|(5, 1, 2)|var|int8
var_306 = relay.var("var_306", dtype = "int8", shape = (5, 16, 2))#candidate|306|(5, 16, 2)|var|int8
bop_307 = relay.left_shift(var_305.astype('int8'), var_306.astype('int8')) # shape=(5, 16, 2)
bop_310 = relay.bitwise_xor(var_306.astype('uint8'), relay.reshape(bop_307.astype('uint8'), relay.shape_of(var_306))) # shape=(5, 16, 2)
uop_313 = relay.log2(var_305.astype('float32')) # shape=(5, 1, 2)
uop_315 = relay.rsqrt(bop_310.astype('float32')) # shape=(5, 16, 2)
bop_317 = relay.multiply(uop_313.astype('uint64'), relay.reshape(var_305.astype('uint64'), relay.shape_of(uop_313))) # shape=(5, 1, 2)
func_154_call = mod.get_global_var('func_154')
func_157_call = mutated_mod.get_global_var('func_157')
var_321 = relay.var("var_321", dtype = "int16", shape = (4, 28))#candidate|321|(4, 28)|var|int16
call_320 = func_154_call(relay.reshape(var_321.astype('int16'), [8, 14]))
call_322 = func_154_call(relay.reshape(var_321.astype('int16'), [8, 14]))
bop_323 = relay.multiply(uop_313.astype('int8'), uop_315.astype('int8')) # shape=(5, 16, 2)
func_110_call = mod.get_global_var('func_110')
func_116_call = mutated_mod.get_global_var('func_116')
var_327 = relay.var("var_327", dtype = "float64", shape = (3, 2))#candidate|327|(3, 2)|var|float64
const_328 = relay.const([1.812799,-6.483581,2.191475,-6.362816,-5.962309,5.718070,-9.886865,1.166543,-2.872476,-9.729519,7.477888,-1.583930,9.266794,-4.257403,6.837721,3.461619,6.138672,1.517831,-4.202539,-7.623306,5.026752,-1.055574,4.092509,-7.658444,2.688145,-4.726629,1.207968,-1.357964,-4.899585,5.102607,-8.117041,-3.908219,9.443134,-5.104700,-9.383207,-3.788947], dtype = "float64")#candidate|328|(36,)|const|float64
const_329 = relay.const([-5.824886,-0.939720,-7.402274,-5.915379,7.248949,-6.787070,-4.462341,-6.983778,-3.576196,-3.933318,-4.935871,1.415055], dtype = "float64")#candidate|329|(12,)|const|float64
call_326 = relay.TupleGetItem(func_110_call(relay.reshape(var_327.astype('float64'), [1, 6]), relay.reshape(const_328.astype('float64'), [6, 6]), relay.reshape(const_329.astype('float64'), [2, 6]), relay.reshape(const_328.astype('float32'), [6, 6]), relay.reshape(const_328.astype('float32'), [6, 6]), ), 5)
call_330 = relay.TupleGetItem(func_116_call(relay.reshape(var_327.astype('float64'), [1, 6]), relay.reshape(const_328.astype('float64'), [6, 6]), relay.reshape(const_329.astype('float64'), [2, 6]), relay.reshape(const_328.astype('float32'), [6, 6]), relay.reshape(const_328.astype('float32'), [6, 6]), ), 5)
uop_331 = relay.asin(bop_310.astype('float32')) # shape=(5, 16, 2)
output = relay.Tuple([bop_317,call_320,var_321,bop_323,call_326,var_327,const_328,const_329,uop_331,])
output2 = relay.Tuple([bop_317,call_322,var_321,bop_323,call_330,var_327,const_328,const_329,uop_331,])
func_333 = relay.Function([var_305,var_306,var_321,var_327,], output)
mod['func_333'] = func_333
mod = relay.transform.InferType()(mod)
var_334 = relay.var("var_334", dtype = "int8", shape = (5, 1, 2))#candidate|334|(5, 1, 2)|var|int8
var_335 = relay.var("var_335", dtype = "int8", shape = (5, 16, 2))#candidate|335|(5, 16, 2)|var|int8
var_336 = relay.var("var_336", dtype = "int16", shape = (4, 28))#candidate|336|(4, 28)|var|int16
var_337 = relay.var("var_337", dtype = "float64", shape = (3, 2))#candidate|337|(3, 2)|var|float64
output = func_333(var_334,var_335,var_336,var_337,)
func_338 = relay.Function([var_334,var_335,var_336,var_337,], output)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
var_340 = relay.var("var_340", dtype = "float64", shape = (14, 13))#candidate|340|(14, 13)|var|float64
uop_341 = relay.sin(var_340.astype('float64')) # shape=(14, 13)
bop_343 = relay.multiply(uop_341.astype('int32'), relay.reshape(var_340.astype('int32'), relay.shape_of(uop_341))) # shape=(14, 13)
uop_346 = relay.atan(var_340.astype('float32')) # shape=(14, 13)
bop_348 = relay.left_shift(bop_343.astype('int64'), relay.reshape(var_340.astype('int64'), relay.shape_of(bop_343))) # shape=(14, 13)
output = relay.Tuple([uop_346,bop_348,])
output2 = relay.Tuple([uop_346,bop_348,])
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
input_340= np.array([[-6.081894,1.688211,8.863938,0.450313,3.658550,-6.067565,7.061671,2.014809,-7.879157,-3.152569,3.920208,3.771378,7.780847],[-7.048978,3.573173,9.953483,-8.458519,9.142625,3.565912,2.967682,-4.476932,1.122879,-7.716005,-0.892664,0.962443,-2.950634],[-0.747490,-4.610365,-3.892335,3.334718,0.426529,-6.030983,-5.069765,7.828269,-8.287925,-8.001916,-1.816026,-4.620293,-9.171271],[7.633899,3.210136,-5.837453,3.894311,8.760505,1.061247,0.013295,-3.264703,-5.762706,-0.626804,-5.664658,-7.305108,1.474614],[-0.482066,3.086328,3.138733,-4.813295,-4.697121,-5.203742,-6.301762,-2.103166,-7.251576,-3.210757,8.779840,0.025432,-6.948223],[-9.333296,-3.452690,4.213558,6.004463,5.404001,8.235911,0.761446,-6.648291,5.425480,-5.862124,-2.674444,-4.642153,7.341071],[2.718011,3.942019,-8.390547,4.183923,0.973236,9.075961,4.983919,5.443943,0.498912,9.433122,-0.919577,-4.898525,-5.182837],[-8.547205,0.279835,-4.653112,-3.738262,-6.557659,8.847289,-6.143720,-0.827840,1.717820,-5.237555,-3.567659,8.266287,-1.538957],[-1.561540,-7.221732,2.078381,2.462837,-3.529523,-4.861321,1.490190,7.581186,-8.709127,-1.090660,-2.178901,-0.826628,-9.351673],[-1.645608,-5.047815,-3.004089,6.496135,0.690299,4.159830,7.026188,-7.510585,-5.871989,-2.456537,-9.882791,6.635940,8.628064],[-0.245026,7.947543,-3.993602,8.059947,-1.866236,4.353052,-2.444119,-4.659597,-9.153465,-6.164794,0.460857,-0.698117,-1.814895],[-2.889114,7.294925,3.075698,-3.482269,5.832503,-3.928709,6.775033,-1.901464,-9.259232,-6.660877,-3.674448,-8.290554,6.726459],[7.270344,8.613815,8.491096,-9.460163,3.655817,-0.035229,-1.408555,-4.074809,-1.036567,0.105652,4.999232,-4.533436,-0.259235],[-3.402005,2.879328,7.919599,2.972291,-0.536798,-5.346886,-1.823985,1.858085,3.078311,-2.225635,-3.179755,-2.004668,5.607353]], dtype='float64')
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