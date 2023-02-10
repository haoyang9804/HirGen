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
uop_1 = relay.acos(var_0.astype('float64')) # shape=()
bop_3 = relay.logical_xor(uop_1.astype('int32'), var_0.astype('int32')) # shape=()
var_6 = relay.var("var_6", dtype = "float64", shape = (6, 7))#candidate|6|(6, 7)|var|float64
bop_7 = relay.right_shift(uop_1.astype('int16'), var_6.astype('int16')) # shape=(6, 7)
bop_10 = relay.mod(uop_1.astype('float32'), var_6.astype('float32')) # shape=(6, 7)
uop_13 = relay.sinh(var_0.astype('float32')) # shape=()
output = relay.Tuple([bop_3,bop_7,bop_10,uop_13,])
output2 = relay.Tuple([bop_3,bop_7,bop_10,uop_13,])
func_15 = relay.Function([var_0,var_6,], output)
mod['func_15'] = func_15
mod = relay.transform.InferType()(mod)
mutated_mod['func_15'] = func_15
mutated_mod = relay.transform.InferType()(mutated_mod)
func_15_call = mutated_mod.get_global_var('func_15')
var_17 = relay.var("var_17", dtype = "float64", shape = ())#candidate|17|()|var|float64
var_18 = relay.var("var_18", dtype = "float64", shape = (6, 7))#candidate|18|(6, 7)|var|float64
call_16 = func_15_call(var_17,var_18,)
output = call_16
func_19 = relay.Function([var_17,var_18,], output)
mutated_mod['func_19'] = func_19
mutated_mod = relay.transform.InferType()(mutated_mod)
var_21 = relay.var("var_21", dtype = "uint32", shape = (16,))#candidate|21|(16,)|var|uint32
var_22 = relay.var("var_22", dtype = "uint32", shape = (16,))#candidate|22|(16,)|var|uint32
bop_23 = relay.left_shift(var_21.astype('uint32'), relay.reshape(var_22.astype('uint32'), relay.shape_of(var_21))) # shape=(16,)
bop_26 = relay.bitwise_xor(var_21.astype('int16'), relay.reshape(bop_23.astype('int16'), relay.shape_of(var_21))) # shape=(16,)
uop_29 = relay.sin(bop_26.astype('float32')) # shape=(16,)
uop_31 = relay.asin(uop_29.astype('float32')) # shape=(16,)
bop_33 = relay.equal(bop_26.astype('bool'), relay.reshape(uop_29.astype('bool'), relay.shape_of(bop_26))) # shape=(16,)
const_36 = relay.const([-6.787079,-2.800273,-5.151850,-0.733103,-5.699338,1.415213,-4.697955,1.301878,3.017296,8.196749,-0.090495,9.881385,-3.159040,7.464255,-9.387236,1.157175], dtype = "float32")#candidate|36|(16,)|const|float32
bop_37 = relay.left_shift(uop_31.astype('int8'), relay.reshape(const_36.astype('int8'), relay.shape_of(uop_31))) # shape=(16,)
bop_40 = relay.floor_mod(uop_31.astype('float32'), relay.reshape(var_22.astype('float32'), relay.shape_of(uop_31))) # shape=(16,)
const_43 = relay.const([5.806467,-8.077951,-3.247168,-8.046144,3.078225,0.426660,4.649936,-7.354181,-7.514023,8.403929,-1.905746,-2.164798,0.101778,1.569315,-2.388452,5.749845], dtype = "float32")#candidate|43|(16,)|const|float32
bop_44 = relay.less_equal(uop_31.astype('bool'), relay.reshape(const_43.astype('bool'), relay.shape_of(uop_31))) # shape=(16,)
uop_47 = relay.tan(bop_44.astype('float64')) # shape=(16,)
var_49 = relay.var("var_49", dtype = "float64", shape = (16,))#candidate|49|(16,)|var|float64
bop_50 = relay.floor_divide(uop_47.astype('float32'), relay.reshape(var_49.astype('float32'), relay.shape_of(uop_47))) # shape=(16,)
bop_53 = relay.floor_divide(uop_31.astype('float32'), relay.reshape(var_22.astype('float32'), relay.shape_of(uop_31))) # shape=(16,)
uop_56 = relay.asinh(uop_47.astype('float64')) # shape=(16,)
var_58 = relay.var("var_58", dtype = "float64", shape = (16,))#candidate|58|(16,)|var|float64
bop_59 = relay.greater_equal(uop_47.astype('bool'), relay.reshape(var_58.astype('bool'), relay.shape_of(uop_47))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_19_call = mutated_mod.get_global_var('func_19')
const_63 = relay.const(-3.184912, dtype = "float64")#candidate|63|()|const|float64
const_64 = relay.const([6.968843,-1.151473,6.968346,-0.085939,-2.027448,-7.559170,6.469422,9.342177,-4.782910,-2.975737,-6.666197,6.579127,3.079347,6.786433,-4.837116,-8.399480,2.075056,-1.008744,5.496617,-2.199648,-7.672871,0.269361,5.853009,3.939779,-0.104528,0.003623,-1.050664,-6.297417,-0.565862,-6.083562,-2.935668,-9.789432,-0.969821,-5.791761,4.620646,2.069764,-2.691571,5.611739,8.541245,8.181624,-5.938368,-6.606793], dtype = "float64")#candidate|64|(42,)|const|float64
call_62 = relay.TupleGetItem(func_15_call(relay.reshape(const_63.astype('float64'), []), relay.reshape(const_64.astype('float64'), [6, 7]), ), 1)
call_65 = relay.TupleGetItem(func_19_call(relay.reshape(const_63.astype('float64'), []), relay.reshape(const_64.astype('float64'), [6, 7]), ), 1)
bop_66 = relay.logical_xor(uop_29.astype('int64'), relay.reshape(const_36.astype('int64'), relay.shape_of(uop_29))) # shape=(16,)
func_15_call = mod.get_global_var('func_15')
func_19_call = mutated_mod.get_global_var('func_19')
call_69 = relay.TupleGetItem(func_15_call(relay.reshape(const_63.astype('float64'), []), relay.reshape(call_62.astype('float64'), [6, 7]), ), 3)
call_70 = relay.TupleGetItem(func_19_call(relay.reshape(const_63.astype('float64'), []), relay.reshape(call_62.astype('float64'), [6, 7]), ), 3)
uop_71 = relay.log2(uop_29.astype('float64')) # shape=(16,)
uop_73 = relay.exp(bop_66.astype('float32')) # shape=(16,)
uop_75 = relay.sinh(const_63.astype('float64')) # shape=()
bop_77 = relay.less_equal(uop_47.astype('bool'), relay.reshape(var_58.astype('bool'), relay.shape_of(uop_47))) # shape=(16,)
bop_80 = relay.bitwise_or(bop_50.astype('uint8'), relay.reshape(bop_59.astype('uint8'), relay.shape_of(bop_50))) # shape=(16,)
output = relay.Tuple([bop_33,bop_37,bop_40,bop_53,uop_56,call_62,const_64,call_69,uop_71,uop_73,uop_75,bop_77,bop_80,])
output2 = relay.Tuple([bop_33,bop_37,bop_40,bop_53,uop_56,call_65,const_64,call_70,uop_71,uop_73,uop_75,bop_77,bop_80,])
func_83 = relay.Function([var_21,var_22,var_49,var_58,], output)
mod['func_83'] = func_83
mod = relay.transform.InferType()(mod)
mutated_mod['func_83'] = func_83
mutated_mod = relay.transform.InferType()(mutated_mod)
func_83_call = mutated_mod.get_global_var('func_83')
var_85 = relay.var("var_85", dtype = "uint32", shape = (16,))#candidate|85|(16,)|var|uint32
var_86 = relay.var("var_86", dtype = "uint32", shape = (16,))#candidate|86|(16,)|var|uint32
var_87 = relay.var("var_87", dtype = "float64", shape = (16,))#candidate|87|(16,)|var|float64
var_88 = relay.var("var_88", dtype = "float64", shape = (16,))#candidate|88|(16,)|var|float64
call_84 = func_83_call(var_85,var_86,var_87,var_88,)
output = call_84
func_89 = relay.Function([var_85,var_86,var_87,var_88,], output)
mutated_mod['func_89'] = func_89
mutated_mod = relay.transform.InferType()(mutated_mod)
var_91 = relay.var("var_91", dtype = "float32", shape = (7,))#candidate|91|(7,)|var|float32
var_92 = relay.var("var_92", dtype = "float32", shape = (7,))#candidate|92|(7,)|var|float32
bop_93 = relay.floor_mod(var_91.astype('float32'), relay.reshape(var_92.astype('float32'), relay.shape_of(var_91))) # shape=(7,)
output = bop_93
output2 = bop_93
func_96 = relay.Function([var_91,var_92,], output)
mod['func_96'] = func_96
mod = relay.transform.InferType()(mod)
mutated_mod['func_96'] = func_96
mutated_mod = relay.transform.InferType()(mutated_mod)
func_96_call = mutated_mod.get_global_var('func_96')
var_98 = relay.var("var_98", dtype = "float32", shape = (7,))#candidate|98|(7,)|var|float32
var_99 = relay.var("var_99", dtype = "float32", shape = (7,))#candidate|99|(7,)|var|float32
call_97 = func_96_call(var_98,var_99,)
output = call_97
func_100 = relay.Function([var_98,var_99,], output)
mutated_mod['func_100'] = func_100
mutated_mod = relay.transform.InferType()(mutated_mod)
const_102 = relay.const([[[-6.900441,8.965534,-3.371693,8.106820,1.043510,-1.894168],[7.385034,-5.196709,-5.643038,5.823361,2.664729,-3.597109]],[[9.973365,0.024753,-8.951888,-7.253370,-2.274117,0.957086],[-1.191150,-0.584470,-0.628968,1.422346,-7.973966,-5.253500]],[[0.141683,9.626653,8.218327,4.678471,-5.337740,-7.435718],[-0.453902,-1.861322,8.998393,9.654637,-5.888183,1.818706]]], dtype = "float32")#candidate|102|(3, 2, 6)|const|float32
uop_103 = relay.erf(const_102.astype('float32')) # shape=(3, 2, 6)
bop_105 = relay.mod(const_102.astype('float32'), relay.reshape(uop_103.astype('float32'), relay.shape_of(const_102))) # shape=(3, 2, 6)
const_108 = relay.const([[[-0.949985,-7.051402,4.982307,6.095530,7.182500,-8.317797],[7.347566,1.666529,-7.351126,-2.536873,-9.831380,1.294238]],[[7.298308,-5.903363,2.393428,6.635101,-2.154165,5.936287],[-9.059117,6.435547,1.555949,-2.571814,-0.360542,6.851156]],[[3.785663,0.301052,4.119084,-3.594903,5.059127,1.555927],[4.106325,-7.269995,-4.791356,4.553965,-7.985991,-2.539750]]], dtype = "float32")#candidate|108|(3, 2, 6)|const|float32
bop_109 = relay.equal(uop_103.astype('bool'), relay.reshape(const_108.astype('bool'), relay.shape_of(uop_103))) # shape=(3, 2, 6)
var_112 = relay.var("var_112", dtype = "float32", shape = (3, 2, 6))#candidate|112|(3, 2, 6)|var|float32
bop_113 = relay.floor_divide(const_108.astype('float32'), relay.reshape(var_112.astype('float32'), relay.shape_of(const_108))) # shape=(3, 2, 6)
bop_116 = relay.mod(bop_113.astype('float64'), relay.reshape(uop_103.astype('float64'), relay.shape_of(bop_113))) # shape=(3, 2, 6)
uop_119 = relay.atan(uop_103.astype('float32')) # shape=(3, 2, 6)
output = relay.Tuple([bop_105,bop_109,bop_116,uop_119,])
output2 = relay.Tuple([bop_105,bop_109,bop_116,uop_119,])
func_121 = relay.Function([var_112,], output)
mod['func_121'] = func_121
mod = relay.transform.InferType()(mod)
mutated_mod['func_121'] = func_121
mutated_mod = relay.transform.InferType()(mutated_mod)
var_122 = relay.var("var_122", dtype = "float32", shape = (3, 2, 6))#candidate|122|(3, 2, 6)|var|float32
func_121_call = mutated_mod.get_global_var('func_121')
call_123 = func_121_call(var_122)
output = call_123
func_124 = relay.Function([var_122], output)
mutated_mod['func_124'] = func_124
mutated_mod = relay.transform.InferType()(mutated_mod)
var_126 = relay.var("var_126", dtype = "float32", shape = ())#candidate|126|()|var|float32
uop_127 = relay.log2(var_126.astype('float32')) # shape=()
bop_129 = relay.floor_divide(uop_127.astype('float64'), var_126.astype('float64')) # shape=()
uop_132 = relay.log(var_126.astype('float32')) # shape=()
bop_134 = relay.add(uop_132.astype('uint16'), uop_127.astype('uint16')) # shape=()
bop_137 = relay.multiply(var_126.astype('float64'), bop_129.astype('float64')) # shape=()
bop_140 = relay.bitwise_or(bop_134.astype('int64'), bop_129.astype('int64')) # shape=()
bop_143 = relay.greater_equal(bop_137.astype('bool'), uop_127.astype('bool')) # shape=()
uop_146 = relay.acos(bop_143.astype('float32')) # shape=()
uop_148 = relay.log(uop_146.astype('float64')) # shape=()
uop_150 = relay.sinh(uop_148.astype('float64')) # shape=()
uop_152 = relay.cosh(uop_150.astype('float32')) # shape=()
bop_154 = relay.right_shift(uop_152.astype('uint16'), bop_129.astype('uint16')) # shape=()
uop_157 = relay.acosh(bop_154.astype('float64')) # shape=()
output = relay.Tuple([bop_140,uop_157,])
output2 = relay.Tuple([bop_140,uop_157,])
func_159 = relay.Function([var_126,], output)
mod['func_159'] = func_159
mod = relay.transform.InferType()(mod)
var_160 = relay.var("var_160", dtype = "float32", shape = ())#candidate|160|()|var|float32
output = func_159(var_160)
func_161 = relay.Function([var_160], output)
mutated_mod['func_161'] = func_161
mutated_mod = relay.transform.InferType()(mutated_mod)
const_163 = relay.const(7.597997, dtype = "float32")#candidate|163|()|const|float32
uop_164 = relay.sigmoid(const_163.astype('float32')) # shape=()
uop_166 = relay.sin(uop_164.astype('float32')) # shape=()
uop_168 = relay.asinh(uop_164.astype('float32')) # shape=()
uop_170 = relay.tan(uop_168.astype('float64')) # shape=()
uop_172 = relay.sigmoid(uop_164.astype('float32')) # shape=()
uop_174 = relay.atan(uop_170.astype('float64')) # shape=()
uop_176 = relay.sinh(uop_170.astype('float64')) # shape=()
uop_178 = relay.rsqrt(uop_174.astype('float32')) # shape=()
uop_180 = relay.atan(const_163.astype('float32')) # shape=()
uop_182 = relay.rsqrt(uop_170.astype('float32')) # shape=()
uop_184 = relay.asin(uop_176.astype('float64')) # shape=()
var_186 = relay.var("var_186", dtype = "float32", shape = (5,))#candidate|186|(5,)|var|float32
bop_187 = relay.right_shift(uop_172.astype('int8'), var_186.astype('int8')) # shape=(5,)
output = relay.Tuple([uop_166,uop_178,uop_180,uop_182,uop_184,bop_187,])
output2 = relay.Tuple([uop_166,uop_178,uop_180,uop_182,uop_184,bop_187,])
func_190 = relay.Function([var_186,], output)
mod['func_190'] = func_190
mod = relay.transform.InferType()(mod)
mutated_mod['func_190'] = func_190
mutated_mod = relay.transform.InferType()(mutated_mod)
var_191 = relay.var("var_191", dtype = "float32", shape = (5,))#candidate|191|(5,)|var|float32
func_190_call = mutated_mod.get_global_var('func_190')
call_192 = func_190_call(var_191)
output = call_192
func_193 = relay.Function([var_191], output)
mutated_mod['func_193'] = func_193
mutated_mod = relay.transform.InferType()(mutated_mod)
var_195 = relay.var("var_195", dtype = "int16", shape = (8, 10))#candidate|195|(8, 10)|var|int16
var_196 = relay.var("var_196", dtype = "int16", shape = (8, 10))#candidate|196|(8, 10)|var|int16
bop_197 = relay.subtract(var_195.astype('int16'), relay.reshape(var_196.astype('int16'), relay.shape_of(var_195))) # shape=(8, 10)
var_200 = relay.var("var_200", dtype = "int16", shape = (8, 10))#candidate|200|(8, 10)|var|int16
bop_201 = relay.right_shift(bop_197.astype('uint8'), relay.reshape(var_200.astype('uint8'), relay.shape_of(bop_197))) # shape=(8, 10)
var_204 = relay.var("var_204", dtype = "int16", shape = (8, 10))#candidate|204|(8, 10)|var|int16
bop_205 = relay.floor_divide(bop_197.astype('float64'), relay.reshape(var_204.astype('float64'), relay.shape_of(bop_197))) # shape=(8, 10)
bop_208 = relay.bitwise_xor(var_204.astype('int64'), relay.reshape(bop_197.astype('int64'), relay.shape_of(var_204))) # shape=(8, 10)
uop_211 = relay.sinh(bop_201.astype('float32')) # shape=(8, 10)
uop_213 = relay.sigmoid(bop_208.astype('float64')) # shape=(8, 10)
bop_215 = relay.add(var_204.astype('uint8'), relay.reshape(bop_208.astype('uint8'), relay.shape_of(var_204))) # shape=(8, 10)
uop_218 = relay.sqrt(uop_211.astype('float32')) # shape=(8, 10)
uop_220 = relay.cos(uop_218.astype('float64')) # shape=(8, 10)
bop_222 = relay.bitwise_or(uop_220.astype('uint64'), relay.reshape(var_200.astype('uint64'), relay.shape_of(uop_220))) # shape=(8, 10)
bop_225 = relay.greater_equal(uop_220.astype('bool'), relay.reshape(bop_205.astype('bool'), relay.shape_of(uop_220))) # shape=(8, 10)
func_15_call = mod.get_global_var('func_15')
func_19_call = mutated_mod.get_global_var('func_19')
const_229 = relay.const(-8.419164, dtype = "float64")#candidate|229|()|const|float64
var_230 = relay.var("var_230", dtype = "float64", shape = (42,))#candidate|230|(42,)|var|float64
call_228 = relay.TupleGetItem(func_15_call(relay.reshape(const_229.astype('float64'), []), relay.reshape(var_230.astype('float64'), [6, 7]), ), 0)
call_231 = relay.TupleGetItem(func_19_call(relay.reshape(const_229.astype('float64'), []), relay.reshape(var_230.astype('float64'), [6, 7]), ), 0)
bop_232 = relay.maximum(bop_225.astype('int64'), relay.reshape(uop_220.astype('int64'), relay.shape_of(bop_225))) # shape=(8, 10)
bop_235 = relay.logical_xor(uop_213.astype('int16'), relay.reshape(uop_220.astype('int16'), relay.shape_of(uop_213))) # shape=(8, 10)
bop_238 = relay.bitwise_xor(uop_218.astype('uint16'), relay.reshape(var_195.astype('uint16'), relay.shape_of(uop_218))) # shape=(8, 10)
uop_241 = relay.cos(uop_213.astype('float32')) # shape=(8, 10)
bop_243 = relay.multiply(bop_235.astype('int16'), relay.reshape(bop_208.astype('int16'), relay.shape_of(bop_235))) # shape=(8, 10)
func_121_call = mod.get_global_var('func_121')
func_124_call = mutated_mod.get_global_var('func_124')
const_247 = relay.const([[5.493589,-8.593898,8.821931,2.629100],[2.864636,-0.673584,-7.858243,8.082905],[0.262855,8.024841,-1.771536,-3.442370],[0.495053,6.793523,8.567842,3.034069],[5.433889,-6.625785,-9.316359,3.011923],[-8.221561,6.488531,-6.301398,-8.592150],[-5.658248,4.461610,7.236077,2.207578],[2.798129,7.676695,6.897204,9.070981],[-4.578752,-9.960130,4.729667,-4.813206]], dtype = "float32")#candidate|247|(9, 4)|const|float32
call_246 = relay.TupleGetItem(func_121_call(relay.reshape(const_247.astype('float32'), [3, 2, 6])), 0)
call_248 = relay.TupleGetItem(func_124_call(relay.reshape(const_247.astype('float32'), [3, 2, 6])), 0)
uop_249 = relay.atanh(bop_225.astype('float32')) # shape=(8, 10)
func_121_call = mod.get_global_var('func_121')
func_124_call = mutated_mod.get_global_var('func_124')
call_251 = relay.TupleGetItem(func_121_call(relay.reshape(call_246.astype('float32'), [3, 2, 6])), 0)
call_252 = relay.TupleGetItem(func_124_call(relay.reshape(call_246.astype('float32'), [3, 2, 6])), 0)
bop_253 = relay.bitwise_or(uop_213.astype('int64'), relay.reshape(uop_241.astype('int64'), relay.shape_of(uop_213))) # shape=(8, 10)
bop_256 = relay.logical_or(bop_222.astype('bool'), relay.reshape(bop_225.astype('bool'), relay.shape_of(bop_222))) # shape=(8, 10)
bop_259 = relay.greater(bop_238.astype('bool'), relay.reshape(bop_208.astype('bool'), relay.shape_of(bop_238))) # shape=(8, 10)
uop_262 = relay.sin(uop_249.astype('float64')) # shape=(8, 10)
uop_264 = relay.sinh(uop_262.astype('float32')) # shape=(8, 10)
uop_266 = relay.sin(uop_264.astype('float32')) # shape=(8, 10)
var_268 = relay.var("var_268", dtype = "float32", shape = (8, 10))#candidate|268|(8, 10)|var|float32
bop_269 = relay.equal(uop_266.astype('bool'), relay.reshape(var_268.astype('bool'), relay.shape_of(uop_266))) # shape=(8, 10)
bop_272 = relay.add(uop_249.astype('uint64'), relay.reshape(bop_243.astype('uint64'), relay.shape_of(uop_249))) # shape=(8, 10)
bop_275 = relay.subtract(uop_262.astype('int16'), relay.reshape(var_195.astype('int16'), relay.shape_of(uop_262))) # shape=(8, 10)
uop_278 = relay.log(uop_264.astype('float64')) # shape=(8, 10)
uop_280 = relay.erf(uop_262.astype('float64')) # shape=(8, 10)
const_282 = relay.const([[-8.839818,-7.673682,-6.619967,9.437013,8.036360,5.627267,3.731989,8.065034,-6.090501,-4.938311],[-9.866126,-0.865900,-6.796154,-7.716606,-6.940430,2.309136,6.259659,-0.594282,0.858590,0.729901],[6.831616,1.105196,3.333369,5.302837,-0.016950,2.706849,-7.754310,-8.166232,8.804227,1.529212],[-4.334982,-6.632542,-1.935995,-1.390171,4.319931,-6.088633,-4.379596,9.877453,-0.238165,1.358351],[9.534910,6.606866,-4.621611,-0.851584,6.861286,1.828931,-8.526931,8.919621,5.337238,7.595315],[2.933390,7.133870,8.430164,-1.754222,0.812922,-0.777898,-1.662328,-6.234827,-1.858046,-3.863836],[-9.816762,9.816560,-1.609031,-2.986863,-2.581642,-7.665804,9.980554,-1.922642,-9.929796,5.285028],[-2.170599,-8.195936,5.310435,-4.288544,-8.960542,6.852286,-3.307621,2.530685,-7.146923,-0.908615]], dtype = "float32")#candidate|282|(8, 10)|const|float32
bop_283 = relay.less(uop_266.astype('bool'), relay.reshape(const_282.astype('bool'), relay.shape_of(uop_266))) # shape=(8, 10)
var_286 = relay.var("var_286", dtype = "bool", shape = (8, 10))#candidate|286|(8, 10)|var|bool
bop_287 = relay.floor_mod(bop_283.astype('float32'), relay.reshape(var_286.astype('float32'), relay.shape_of(bop_283))) # shape=(8, 10)
const_290 = relay.const([[9,-2,-6,5,4,7,-8,9,-6,3],[-8,-6,-2,-8,10,-10,9,-1,-3,-8],[-4,6,7,5,2,7,-3,-5,-7,-1],[4,-5,-3,9,-5,-3,-10,-2,-4,-10],[-5,6,-1,-4,-5,-5,-6,3,9,7],[9,-2,9,-5,10,-1,4,6,10,2],[-5,-6,9,2,1,3,-1,8,10,-5],[1,-7,3,8,3,1,-1,1,-8,10]], dtype = "int16")#candidate|290|(8, 10)|const|int16
bop_291 = relay.bitwise_xor(bop_275.astype('uint16'), relay.reshape(const_290.astype('uint16'), relay.shape_of(bop_275))) # shape=(8, 10)
var_294 = relay.var("var_294", dtype = "bool", shape = (8, 10))#candidate|294|(8, 10)|var|bool
bop_295 = relay.logical_or(bop_283.astype('bool'), relay.reshape(var_294.astype('bool'), relay.shape_of(bop_283))) # shape=(8, 10)
var_298 = relay.var("var_298", dtype = "float32", shape = (8, 10))#candidate|298|(8, 10)|var|float32
bop_299 = relay.power(uop_266.astype('float64'), relay.reshape(var_298.astype('float64'), relay.shape_of(uop_266))) # shape=(8, 10)
bop_302 = relay.right_shift(uop_280.astype('uint16'), relay.reshape(uop_220.astype('uint16'), relay.shape_of(uop_280))) # shape=(8, 10)
bop_305 = relay.not_equal(bop_302.astype('bool'), relay.reshape(uop_266.astype('bool'), relay.shape_of(bop_302))) # shape=(8, 10)
bop_308 = relay.logical_xor(bop_287.astype('uint8'), relay.reshape(bop_283.astype('uint8'), relay.shape_of(bop_287))) # shape=(8, 10)
uop_311 = relay.log2(bop_287.astype('float64')) # shape=(8, 10)
const_313 = relay.const([[0.372054,-9.813011,-8.972232,-3.803825,-5.991929,8.543745,-9.899511,0.007392,-0.604597,-4.156856],[-8.857956,-4.244701,8.174434,-1.604079,5.208549,-8.069959,0.187266,-9.001096,9.267539,6.851514],[7.056772,3.339523,-7.814916,6.581839,9.135016,2.821525,-0.227238,3.154286,-1.342941,9.185341],[-2.365458,-4.675779,-4.500860,1.394441,-5.469615,8.420410,9.923508,7.485557,-4.313202,-2.462239],[2.476057,7.753439,8.965157,5.503500,-6.660826,-4.213580,-7.733610,1.285533,6.488104,-7.347535],[8.006817,6.929313,-9.438427,7.604449,-0.615133,2.963215,4.566230,3.048612,2.268495,-3.976324],[-0.640884,-5.389518,-8.280213,4.291025,-9.919873,8.630147,-7.117971,-7.748166,2.836637,5.347306],[-3.084037,7.362652,8.639186,-1.039856,0.623362,4.011045,0.674320,-1.864106,0.926571,-1.558782]], dtype = "float64")#candidate|313|(8, 10)|const|float64
bop_314 = relay.logical_or(uop_311.astype('bool'), relay.reshape(const_313.astype('bool'), relay.shape_of(uop_311))) # shape=(8, 10)
const_317 = relay.const([[8.312385,9.309231,-3.334686,8.073571,9.361243,8.003155,1.226493,-2.975797,7.341160,-2.422742],[-6.562349,-2.659012,-6.591259,-6.636102,-6.414887,2.189155,9.547018,5.180835,-9.715523,7.284515],[6.091094,5.107689,4.678655,7.064076,-3.273495,8.664004,1.057101,4.209855,2.840563,-2.229388],[-1.901531,9.867507,-3.911566,3.193820,-8.217711,-7.500290,7.395651,-1.267155,7.396265,9.233500],[7.079979,-9.376580,0.671956,-2.840626,-0.248387,-0.590125,-4.322862,-3.436382,-3.166287,-4.813778],[-9.215496,-6.519792,-5.454888,-6.811445,-9.510560,7.215433,2.937346,8.457336,5.824067,0.576431],[-2.129613,-0.141741,-8.182520,2.359130,-4.550756,4.503291,4.447721,-2.300021,-5.760595,3.695461],[4.680009,-2.845047,6.411994,5.712298,-6.085834,-1.908848,-9.464702,5.047734,6.660853,-4.543316]], dtype = "float64")#candidate|317|(8, 10)|const|float64
bop_318 = relay.minimum(uop_280.astype('int64'), relay.reshape(const_317.astype('int64'), relay.shape_of(uop_280))) # shape=(8, 10)
func_121_call = mod.get_global_var('func_121')
func_124_call = mutated_mod.get_global_var('func_124')
call_321 = relay.TupleGetItem(func_121_call(relay.reshape(call_246.astype('float32'), [3, 2, 6])), 2)
call_322 = relay.TupleGetItem(func_124_call(relay.reshape(call_246.astype('float32'), [3, 2, 6])), 2)
output = relay.Tuple([bop_215,call_228,const_229,var_230,bop_232,call_246,const_247,call_251,bop_253,bop_256,bop_259,bop_269,bop_272,uop_278,bop_291,bop_295,bop_299,bop_305,bop_308,bop_314,bop_318,call_321,])
output2 = relay.Tuple([bop_215,call_231,const_229,var_230,bop_232,call_248,const_247,call_252,bop_253,bop_256,bop_259,bop_269,bop_272,uop_278,bop_291,bop_295,bop_299,bop_305,bop_308,bop_314,bop_318,call_322,])
func_323 = relay.Function([var_195,var_196,var_200,var_204,var_230,var_268,var_286,var_294,var_298,], output)
mod['func_323'] = func_323
mod = relay.transform.InferType()(mod)
mutated_mod['func_323'] = func_323
mutated_mod = relay.transform.InferType()(mutated_mod)
func_323_call = mutated_mod.get_global_var('func_323')
var_325 = relay.var("var_325", dtype = "int16", shape = (8, 10))#candidate|325|(8, 10)|var|int16
var_326 = relay.var("var_326", dtype = "int16", shape = (8, 10))#candidate|326|(8, 10)|var|int16
var_327 = relay.var("var_327", dtype = "int16", shape = (8, 10))#candidate|327|(8, 10)|var|int16
var_328 = relay.var("var_328", dtype = "int16", shape = (8, 10))#candidate|328|(8, 10)|var|int16
var_329 = relay.var("var_329", dtype = "float64", shape = (42,))#candidate|329|(42,)|var|float64
var_330 = relay.var("var_330", dtype = "float32", shape = (8, 10))#candidate|330|(8, 10)|var|float32
var_331 = relay.var("var_331", dtype = "bool", shape = (8, 10))#candidate|331|(8, 10)|var|bool
var_332 = relay.var("var_332", dtype = "bool", shape = (8, 10))#candidate|332|(8, 10)|var|bool
var_333 = relay.var("var_333", dtype = "float32", shape = (8, 10))#candidate|333|(8, 10)|var|float32
call_324 = func_323_call(var_325,var_326,var_327,var_328,var_329,var_330,var_331,var_332,var_333,)
output = call_324
func_334 = relay.Function([var_325,var_326,var_327,var_328,var_329,var_330,var_331,var_332,var_333,], output)
mutated_mod['func_334'] = func_334
mutated_mod = relay.transform.InferType()(mutated_mod)
var_336 = relay.var("var_336", dtype = "int16", shape = (16, 14, 11))#candidate|336|(16, 14, 11)|var|int16
var_337 = relay.var("var_337", dtype = "int16", shape = (16, 14, 11))#candidate|337|(16, 14, 11)|var|int16
bop_338 = relay.multiply(var_336.astype('int16'), relay.reshape(var_337.astype('int16'), relay.shape_of(var_336))) # shape=(16, 14, 11)
uop_341 = relay.atan(var_337.astype('float32')) # shape=(16, 14, 11)
var_343 = relay.var("var_343", dtype = "float32", shape = (16, 14, 11))#candidate|343|(16, 14, 11)|var|float32
bop_344 = relay.less(uop_341.astype('bool'), relay.reshape(var_343.astype('bool'), relay.shape_of(uop_341))) # shape=(16, 14, 11)
bop_347 = relay.bitwise_and(uop_341.astype('uint8'), relay.reshape(var_336.astype('uint8'), relay.shape_of(uop_341))) # shape=(16, 14, 11)
bop_350 = relay.less_equal(bop_344.astype('bool'), relay.reshape(bop_338.astype('bool'), relay.shape_of(bop_344))) # shape=(16, 14, 11)
output = relay.Tuple([bop_347,bop_350,])
output2 = relay.Tuple([bop_347,bop_350,])
F = relay.Function([var_336,var_337,var_343,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_336,var_337,var_343,], output2)
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
input_336= np.array([[[-5,-7,7,9,-4,-7,-10,5,9,-9,5],[-1,-8,-2,-2,-3,7,9,-1,8,-5,8],[-8,-7,6,-1,-9,7,7,-8,-2,-7,-1],[4,5,7,-1,-5,9,-5,-7,-8,-9,4],[4,6,-3,6,-9,5,-4,-4,-9,-3,-1],[3,-9,2,6,5,-8,-8,3,-4,-6,-6],[1,-7,-7,9,1,3,1,-7,3,-8,2],[-1,-10,7,4,-6,3,5,4,-7,-6,-6],[-8,-3,-6,4,-7,-6,2,2,-7,10,2],[10,-1,-4,9,-3,6,10,9,1,-10,2],[10,-5,2,9,-1,3,-5,-9,8,3,-10],[-5,6,-4,5,-8,9,4,-1,9,-1,5],[1,10,6,3,-6,-4,8,-3,-3,10,6],[-7,-6,-2,9,2,-3,-9,-8,6,5,-8]],[[-9,6,2,6,-9,-9,-7,1,-10,8,-9],[-2,-1,1,-10,7,-2,1,-8,-5,10,-3],[5,-5,-10,6,-9,-4,-9,-4,-8,8,3],[-3,8,1,2,-5,9,1,2,-9,-8,-9],[10,7,4,7,9,-2,9,-5,9,-9,-9],[-8,-8,1,-9,-4,-4,8,-10,-7,8,8],[-3,8,10,-2,8,1,6,3,-10,8,-1],[4,-3,2,10,-9,-10,7,-6,10,-9,-9],[1,-2,-8,7,-7,1,-1,1,3,-6,-7],[-5,-2,-7,6,-5,6,1,8,-3,10,4],[9,5,-5,4,-6,7,4,8,-9,-10,-4],[1,1,-6,-2,-4,-8,-10,10,7,3,-3],[-9,8,7,3,8,-1,1,-2,10,1,-3],[-10,-7,8,-2,-8,1,7,5,-7,-7,4]],[[8,1,9,7,-5,6,4,1,-4,-7,6],[-3,5,3,-2,1,2,9,4,-5,3,-10],[-5,2,-1,-4,-4,-5,9,-2,5,-1,5],[-6,4,-7,-3,3,-9,7,6,6,-10,-1],[-9,8,-8,8,-8,2,8,-10,-8,10,10],[7,1,-9,4,-10,5,-4,-6,8,-9,-8],[2,9,-3,-7,-10,-2,-8,2,1,3,-9],[-10,-1,2,7,-8,-10,-8,10,5,-10,1],[-7,9,4,-3,1,1,-10,-2,1,-7,-9],[-3,10,8,-6,-7,-1,-9,8,-1,5,-1],[-3,5,2,-6,-7,-1,8,9,8,6,-4],[-8,7,3,1,9,6,-9,-6,-3,-5,-3],[4,6,-10,3,9,10,9,10,-7,4,-4],[6,-8,-6,3,-6,1,-9,6,9,-1,3]],[[-2,2,3,7,-8,9,1,-9,-5,1,8],[-9,4,-2,-6,-7,-8,-10,-6,8,9,2],[-3,6,7,-3,1,-1,-9,10,-8,6,7],[-4,-4,-10,2,-4,-6,-2,-2,-10,-9,8],[6,5,-4,-9,-8,7,-7,-4,-3,-5,-3],[-2,4,-7,-4,2,2,5,10,-10,-9,10],[-6,-7,5,5,-9,-2,-10,-9,-6,6,10],[-4,7,9,-1,-4,-3,5,10,-5,-1,-5],[8,7,5,-9,-7,7,9,2,5,8,-4],[6,-6,-7,10,8,-6,-9,10,-7,-3,3],[-10,-3,-10,3,-3,-9,1,-6,8,8,-8],[-6,-5,7,5,9,4,10,-2,-7,-3,-3],[-2,7,2,-5,-2,-9,4,-6,-6,4,-8],[6,-7,-6,3,7,8,-7,-1,1,5,-4]],[[-5,-6,7,5,-4,5,7,-4,-1,-6,-6],[1,7,-5,-3,-9,-7,-1,7,-2,-8,4],[10,-4,10,-4,-2,8,-9,2,-9,-10,6],[-3,-9,5,1,3,-7,4,6,6,3,7],[-4,-1,10,1,7,10,-8,3,2,3,-2],[9,-5,8,-8,9,6,6,3,1,-1,8],[-3,1,9,-7,2,-1,-2,6,-3,-3,-5],[4,-8,10,5,-1,10,-6,-9,-2,-10,8],[1,1,7,5,-4,-1,-4,-4,-10,-9,-10],[-1,-1,-7,-5,4,-1,2,-10,-1,5,-9],[-7,-10,-9,-9,-8,-5,3,-10,1,8,9],[-7,8,5,1,4,6,-5,5,-5,4,1],[-4,-10,5,4,-8,4,2,-9,3,1,7],[6,1,2,3,-7,-4,-1,-4,6,-9,-8]],[[-2,-6,-10,6,-2,-7,6,-10,-3,-6,-3],[8,6,-7,-6,8,4,3,-10,9,-1,-6],[-6,-6,3,-7,-10,2,-1,-1,-6,3,2],[3,-4,3,-9,-4,5,-5,1,3,-1,7],[5,-8,-6,2,6,-7,6,-8,9,1,2],[-10,-1,6,9,-3,9,7,2,-3,-3,-3],[-2,-5,4,5,-3,1,-10,-6,-8,5,-10],[-5,2,-3,10,-5,6,3,7,6,-3,-5],[-3,5,-2,1,5,-4,-5,3,4,10,-7],[10,1,3,2,-10,-9,-8,-5,2,-6,6],[-7,9,-6,-1,-1,6,-6,9,-7,5,-4],[-8,4,9,-10,3,2,9,10,1,-9,-5],[-6,-5,3,-2,4,6,-1,10,-7,-7,3],[-9,5,6,6,1,6,-6,9,-8,7,-3]],[[-8,-10,-3,6,-4,-1,-6,5,8,6,5],[7,-2,-6,-8,7,-1,-2,5,-4,-4,8],[-4,2,8,-10,-1,-6,1,-2,-1,6,8],[-10,-10,10,9,-6,10,5,-7,10,1,1],[-7,2,1,-2,9,-2,6,3,-9,-8,-6],[-6,9,10,-1,3,-3,-8,6,5,-3,4],[-8,-5,-5,-5,-10,-5,8,-1,6,9,-1],[-4,-8,-2,-4,10,-3,-9,2,1,-7,-10],[-2,6,-5,5,-3,-4,-8,3,5,3,2],[10,8,-3,6,1,-4,-5,-1,8,-9,-4],[1,8,3,-7,5,-7,7,-1,-5,2,-9],[-10,-7,-6,-3,3,10,-1,5,7,3,5],[8,-6,8,2,-5,10,3,10,6,9,5],[8,7,-9,1,-2,-10,2,8,-5,-8,-7]],[[-1,-6,-10,-1,-4,4,6,4,9,-6,6],[-7,2,7,9,1,-1,-4,-2,6,6,9],[-5,-9,9,-3,5,-1,-4,-9,2,-2,6],[4,3,-4,-1,-2,6,-1,1,1,-6,5],[1,-4,1,-8,3,7,-1,8,-5,-8,-4],[-5,-5,-10,-1,10,6,-6,9,6,-5,10],[3,10,8,9,9,4,-2,-4,7,6,-7],[2,4,7,-10,-7,-5,-1,7,5,-5,9],[-5,-2,-6,-8,1,-6,-10,6,2,-3,10],[-3,5,2,-9,9,5,-8,10,-3,-8,7],[4,7,7,-6,-5,6,7,10,-1,9,10],[5,5,-10,9,-2,1,10,9,-6,2,-4],[2,-2,3,-6,-3,3,-1,3,-2,-2,6],[2,-5,-10,2,4,8,-6,-1,3,7,2]],[[-1,-7,-1,8,10,-3,-10,-2,6,8,-5],[-4,9,10,5,4,-8,1,-10,9,10,-7],[-4,5,-6,-2,9,9,3,-4,3,6,-10],[5,7,2,-5,-10,-7,-6,9,-7,-6,-8],[8,-1,1,4,-4,1,-2,-1,-1,5,7],[4,3,-6,-8,2,-8,8,-8,-3,3,6],[-9,4,9,2,5,1,9,-9,-2,4,-3],[3,5,6,10,-5,2,-10,3,4,-8,-5],[2,4,4,6,-10,-1,-5,-7,10,5,-3],[2,7,-3,-8,-1,-4,-4,5,-6,-1,2],[-3,-2,-2,6,9,3,-3,-10,7,-3,8],[-7,-9,10,1,5,-2,10,1,-9,-3,10],[-9,9,4,6,9,-7,-8,8,8,3,2],[3,-9,-6,7,2,-8,3,-3,6,-2,-2]],[[-9,-10,-8,6,-8,3,-3,2,7,-7,5],[7,-7,-1,3,9,-8,8,2,3,-4,7],[-7,7,-5,9,4,-7,6,-10,-3,-2,8],[4,5,-9,-6,-6,7,7,8,9,1,-3],[-6,-7,-3,-10,10,-2,3,-3,8,-2,-3],[-5,4,-4,-2,1,3,-6,8,9,2,-2],[-1,-6,4,-3,8,1,2,6,-4,1,-6],[4,-5,-3,-2,-3,-1,3,-8,6,2,7],[-1,-7,9,-2,5,10,-7,-8,-7,-4,-2],[6,3,8,-7,9,-8,-5,-4,-6,3,10],[9,10,9,-6,1,2,2,-9,-7,-1,5],[-3,-2,-8,5,4,-6,-1,8,-7,-9,-5],[1,2,-4,-2,-8,8,6,-5,-3,-8,-1],[5,6,10,-3,6,1,-1,-2,2,1,-10]],[[-7,9,-1,-1,6,4,8,-7,1,4,-3],[4,9,5,-7,7,9,-6,-5,-5,-10,4],[4,-8,7,6,-6,-5,-6,5,5,-9,-9],[5,3,8,-8,1,4,-9,-1,10,3,4],[-6,-7,2,5,3,9,7,-9,-8,9,4],[-2,10,4,-7,-7,7,4,-1,3,7,2],[-9,-1,5,-10,9,-4,4,-4,-6,6,8],[3,6,-5,-9,-9,10,-5,3,4,7,-10],[-3,4,9,3,9,-7,3,10,-1,-10,10],[-7,-1,4,6,9,-8,1,4,-5,1,7],[-3,9,5,-6,5,9,-2,-3,5,-2,-6],[2,-8,-2,6,6,-4,5,2,-8,5,-4],[8,-7,-9,8,3,-7,7,-1,-7,-4,2],[4,-3,3,-8,4,4,9,-4,1,6,-10]],[[5,6,10,5,-3,8,-1,-1,9,5,1],[-9,10,10,-9,-1,-4,8,-10,8,4,-7],[-6,-2,-6,-8,-10,9,8,9,4,2,8],[-3,-5,-10,-4,3,-7,-5,-3,-1,-2,4],[4,7,-3,5,1,10,9,-6,-8,3,6],[-10,2,2,4,4,-7,-7,10,-2,1,8],[3,-5,2,6,-7,-7,-2,3,-4,5,7],[-4,5,9,-9,8,-2,-2,-10,-4,-7,-5],[9,6,-6,5,-4,-2,4,-6,8,-4,6],[5,-4,-2,2,-1,-9,4,-9,-9,-9,-1],[2,-7,-1,3,5,-1,8,10,-5,1,-10],[-9,8,-10,-7,-2,-10,1,8,-7,3,1],[-5,1,6,4,-9,-2,9,3,-5,-6,4],[4,-3,-6,-1,8,-2,6,3,1,-2,-8]],[[-7,8,-8,6,-2,-5,-3,1,-9,-2,8],[-6,-9,-2,2,-2,2,8,2,3,6,1],[6,7,-3,-6,-8,-3,-1,-8,6,9,-9],[10,-1,-9,9,-8,4,-2,-4,-7,9,1],[-4,1,10,9,-7,3,4,-8,-2,9,2],[2,-8,-10,8,6,-10,-3,9,10,-3,-7],[-4,7,3,-1,-7,-3,-1,-1,4,-10,-8],[-7,-5,6,-1,-2,-10,-1,6,8,-5,8],[5,3,-9,-6,-8,-6,3,10,5,-6,-2],[5,-9,1,5,-9,1,10,7,-6,10,-7],[-2,-6,6,1,1,7,-1,3,10,-4,4],[-8,1,-5,6,3,-8,4,-1,3,10,-1],[8,7,-5,-8,7,3,2,9,1,-7,-3],[2,-7,-5,3,-10,-1,1,-7,9,7,-2]],[[-10,-8,-4,-10,7,10,-5,10,2,-2,-4],[1,2,-7,-1,-7,-9,4,9,8,-3,9],[-5,7,9,-10,-2,6,10,4,-2,6,5],[5,-3,-9,3,-6,7,6,3,-8,-7,-1],[-10,2,-6,-5,8,7,5,8,4,-6,7],[6,8,-7,2,5,10,6,-6,7,1,5],[-9,-3,9,3,-3,-6,-7,-7,-8,1,-2],[6,-7,3,-6,-3,-3,8,-10,-8,2,-4],[-6,7,7,-5,-6,8,1,-1,-2,-2,3],[-7,-10,3,-8,2,2,-9,9,8,3,9],[7,-3,-2,-4,-4,7,-7,-6,9,-8,-9],[-10,2,-6,-3,-1,2,-1,-8,1,-7,5],[-1,8,1,5,-4,-2,2,-3,2,-7,-9],[-3,3,-8,4,-4,-6,-9,8,-1,-8,1]],[[1,-6,8,-1,10,-4,10,3,9,1,8],[4,-5,-10,-8,-2,7,-5,-1,-1,-10,4],[-9,4,-4,3,6,-10,2,3,-5,10,2],[6,-2,4,2,-2,7,-8,2,5,-1,-8],[-6,-3,-5,-2,10,10,-5,-6,2,-9,-5],[-10,-10,-2,1,-4,-10,-10,1,1,10,-7],[-6,1,2,-9,-6,-3,-5,7,-4,2,9],[-8,4,4,-9,-6,-4,2,-6,3,9,8],[3,-9,6,1,-3,-6,8,-8,2,1,2],[4,-6,10,4,-1,-1,2,-4,-1,-3,4],[8,10,10,4,7,-10,6,10,5,-5,-4],[4,-1,4,3,-9,2,8,1,3,-1,-10],[-2,5,2,4,8,-3,-6,-4,5,-9,-3],[6,-3,9,-2,3,9,-9,-9,-8,-4,5]],[[-6,-10,-4,-5,-8,-8,-1,-4,-8,-5,6],[-2,1,-9,9,-7,-5,-5,4,6,2,-2],[3,-4,-5,-9,-4,-4,-9,10,-9,2,-1],[-4,8,8,4,-10,6,-9,2,-1,-9,1],[-8,-7,-7,-10,9,-5,4,4,5,-6,-10],[3,-2,-10,7,-2,-2,1,-1,10,-5,-4],[10,8,2,6,6,3,2,-1,-5,-6,-4],[-8,8,-3,-3,-3,-5,-8,-4,8,-6,9],[6,-8,5,-10,1,-1,6,-3,1,3,6],[2,10,9,4,-5,-10,-3,-7,3,3,-6],[-8,-9,4,4,3,-7,9,-8,-3,-3,-6],[-7,3,2,4,4,-6,2,5,9,3,3],[-3,4,4,9,6,-4,6,7,-5,-1,-6],[-2,-8,-10,-8,-4,8,6,-5,-3,-1,-2]]], dtype='int16')
module1.set_input('var_336', input_336)
input_337= np.array([[[-4,-4,-1,-9,-7,-7,1,-9,-4,9,-10],[3,10,-8,4,5,-10,4,-7,8,1,10],[-5,-2,-1,-9,4,6,-10,4,-4,-10,1],[6,7,1,-9,2,9,3,3,-8,-5,-2],[9,-9,-6,-9,-2,-2,-1,7,-3,4,-2],[-1,-8,-5,-3,10,-4,7,7,-6,3,8],[-10,-5,-3,-2,-5,-2,-2,-8,4,9,6],[-8,3,1,-2,1,-5,3,-5,-5,-10,3],[-3,-8,-10,7,-4,10,10,3,-2,4,-1],[7,3,-6,10,4,-4,-2,-1,2,-7,1],[4,9,-10,-5,-7,-7,-4,-5,4,-8,4],[-3,-1,8,-7,-3,-5,-7,5,-8,-4,-5],[9,1,-9,-4,-8,-8,1,-7,7,9,4],[8,8,2,-2,10,-6,7,-9,8,7,8]],[[-4,-6,5,-6,1,-8,10,3,-6,-5,2],[-8,4,-10,-4,9,-1,10,10,7,-9,-2],[-5,-3,8,3,-4,2,-7,-9,9,8,2],[-10,-5,8,3,5,5,-1,9,-4,5,9],[6,-2,5,-9,-1,1,-5,9,9,6,-3],[-8,10,-3,5,6,9,10,-1,1,5,7],[-1,8,2,5,-10,8,-4,-5,-3,4,-1],[6,-1,-6,-8,-8,1,10,-9,-9,-9,-1],[7,4,8,-4,2,3,-6,2,1,-3,-6],[-6,-10,-6,-3,10,2,2,1,1,5,-7],[-4,-4,-10,9,-2,3,-1,8,7,-4,3],[-10,-1,8,-6,8,5,-5,1,-1,9,6],[6,-4,4,-3,-5,3,-6,1,-7,10,-10],[-7,8,-5,-3,-9,-7,-1,2,-1,-6,2]],[[-7,-4,1,-8,-10,6,-4,-7,4,-5,10],[5,1,8,10,9,-4,-4,8,9,-7,-6],[-6,8,-6,2,-1,9,-10,-9,-6,10,5],[6,-3,9,-8,1,-5,-5,-3,8,-3,-6],[-3,-5,-7,-10,8,4,-1,-3,10,-6,-4],[8,7,3,-5,2,8,-1,-2,-7,-9,2],[-6,10,9,-10,6,-2,8,-10,6,2,-9],[9,4,-2,9,-4,8,-10,1,-2,10,7],[-1,2,2,3,7,-3,-2,-5,-8,5,-2],[-3,-4,9,10,1,-6,3,7,9,2,6],[-8,2,-9,-4,-9,-5,3,-7,1,-6,-1],[4,5,-1,-5,10,7,1,-8,-10,-1,1],[10,-7,-6,-7,-2,-4,-4,5,-3,5,6],[9,-1,2,-3,-8,-1,-3,9,-5,-2,-9]],[[7,8,-8,1,10,-7,-6,-5,-2,-1,7],[1,7,-9,8,-7,-4,4,-9,8,-4,10],[-6,10,-7,2,-8,-10,-10,3,-6,4,7],[-4,4,8,8,1,-5,1,-3,-6,5,-5],[4,4,10,2,5,-8,9,-8,-5,9,9],[2,3,4,-5,-3,-1,-10,3,10,9,9],[-3,-3,6,-8,3,9,-9,4,-4,-4,-1],[7,6,4,-5,-3,9,-9,-1,-2,-1,2],[5,-6,2,5,-4,-9,-2,-8,-5,-7,9],[1,-5,6,-5,-5,-4,-3,-1,-2,-3,-5],[-4,-1,-4,-5,-4,-2,-9,-4,7,-7,8],[-7,-10,-4,-10,7,-2,2,7,-8,4,-2],[8,-9,3,6,-6,2,2,1,3,-5,-6],[-10,6,6,7,3,3,-2,8,10,3,-1]],[[6,-8,-10,3,-1,1,-10,-7,4,-4,-2],[5,-4,-5,5,-3,-8,9,3,-5,4,2],[-2,10,-9,-7,1,-10,-5,4,-10,2,-8],[-7,-4,5,-10,-1,-8,-2,-5,7,-9,3],[-1,-5,-3,-1,9,-2,-2,-8,-6,5,-1],[-2,-6,-4,-7,10,-3,-8,-3,7,-7,-2],[8,1,10,-9,-5,-7,1,5,-8,7,-9],[10,-6,-1,-6,3,-4,7,4,-2,8,-9],[1,-10,-8,-10,3,-10,-7,-3,-5,-8,-5],[-10,-2,8,-2,-2,3,-1,8,-8,3,2],[-9,6,5,3,-10,9,-5,-5,-5,7,-8],[9,-2,-9,-6,-7,-9,-1,7,-4,-6,1],[1,-6,1,-6,-6,7,3,2,-8,-6,-2],[-1,10,-5,10,8,2,-2,9,-3,3,2]],[[-7,-7,5,-2,-8,-5,-10,-7,-8,4,8],[4,3,2,-7,-4,-1,1,-2,-3,8,1],[2,-8,2,2,-7,-7,4,-6,6,-7,-9],[-4,-1,-3,7,-5,-1,-9,7,3,8,4],[-1,-2,10,-5,8,-2,-6,-6,-9,4,-7],[5,6,6,9,-5,10,-9,-7,-9,-6,3],[-4,-7,-3,-4,-5,2,8,3,-8,8,-6],[-8,3,-7,6,3,1,4,-2,-4,-9,8],[4,7,-10,2,3,-8,-5,7,8,-1,1],[-3,-6,-2,3,4,-1,-1,-7,-6,3,7],[2,-9,-4,-9,6,10,1,8,-9,-5,-5],[-1,1,1,5,2,-3,-8,-1,-2,1,10],[-3,9,-1,4,1,8,-5,4,8,7,-6],[8,4,-6,-3,8,-5,2,1,3,1,2]],[[2,-8,-3,-4,10,6,-4,-10,1,7,-4],[5,2,7,-1,3,-6,-6,6,8,1,-1],[-8,-6,-8,6,-9,-9,-9,6,5,5,-10],[8,10,3,10,10,6,7,-6,-8,-4,-6],[3,-4,1,8,10,-9,-2,4,8,-9,10],[10,9,-1,-5,7,-10,6,8,-5,8,9],[-6,-1,8,3,-8,-6,-1,1,3,-5,6],[-4,-3,-3,3,-2,-8,7,3,-6,1,9],[8,-1,5,-3,7,-5,6,-3,3,-2,-6],[-9,-7,1,6,3,7,4,-7,-5,4,-5],[2,-3,-1,6,-2,-10,9,-10,-1,-10,6],[-4,-2,-9,1,-1,2,-7,-7,-4,-9,1],[-8,-4,3,-9,9,1,-2,-8,-4,7,-2],[-1,-1,-9,1,-8,-8,-8,-6,9,-2,-5]],[[-3,-9,-1,-5,3,2,-2,4,-1,-2,3],[-2,-9,-2,-3,5,-1,5,3,7,-5,-1],[1,2,-8,-1,-8,-9,3,-10,-10,4,-9],[5,6,-9,-2,10,6,3,10,-4,-8,5],[-6,-10,10,-10,-1,-3,7,-10,4,2,5],[5,-3,4,-4,3,-6,6,6,-8,-1,6],[5,-8,-9,1,10,-9,-7,-4,-4,10,-1],[7,-4,-4,6,-4,-7,1,5,-8,-2,-9],[2,9,-2,7,7,1,-8,6,-5,6,3],[8,1,-4,-2,1,8,-9,1,4,-9,1],[9,3,6,-8,7,8,9,-1,6,-7,4],[-7,-4,-2,7,10,1,-10,3,7,-3,7],[8,4,-10,10,-1,-8,9,7,5,2,3],[4,-9,5,7,-3,-10,9,-2,-3,-6,-7]],[[2,9,-5,3,2,-3,8,4,-10,10,-5],[-1,8,1,6,8,-6,-5,-7,-9,8,3],[-10,3,3,-9,2,7,-3,-3,8,8,10],[3,4,8,3,-1,-8,4,-10,2,-4,-4],[-9,7,10,-1,9,-10,7,-8,-1,7,5],[-1,-6,-9,-1,-7,-9,10,-2,-10,7,5],[-7,-8,3,-8,-5,-5,1,1,-4,4,5],[-3,-7,9,4,-9,8,-1,-1,3,-7,8],[2,-4,10,-2,-4,-3,9,-4,-7,-5,-2],[-9,-7,-10,-1,-4,10,9,-6,4,-5,10],[10,-8,10,-1,2,-5,2,-5,6,6,-7],[-5,-4,-6,7,-7,-1,3,10,4,-5,3],[6,10,7,-9,9,-1,4,-3,3,-7,5],[-6,5,-9,8,-6,9,-6,10,-3,9,3]],[[1,-6,8,-9,-3,-9,6,9,-1,-10,-10],[9,2,-2,-7,-6,-3,-3,6,-7,8,9],[6,-1,2,-7,8,4,-7,-2,-1,3,1],[-6,-7,-7,-9,-10,-4,-9,-2,7,-10,1],[7,-2,-4,-2,-10,-3,1,3,-6,-2,3],[7,10,5,-6,9,-2,4,7,-2,-4,3],[6,-7,8,5,3,-9,5,-2,2,6,4],[9,-6,8,7,7,-7,8,-4,-1,4,-9],[-5,1,2,-1,-7,-8,3,-3,7,-6,4],[4,-6,6,-4,-6,-6,4,4,6,8,-2],[-10,-5,-7,-3,-4,9,-9,-7,-10,10,2],[7,1,-7,9,-2,-8,-6,-8,6,-4,-2],[-1,-6,4,10,1,-3,-3,1,-4,9,-10],[7,-3,-1,-9,2,2,3,-5,3,8,-10]],[[10,3,6,6,-8,-5,-7,10,9,-5,3],[8,-9,-5,9,-10,-9,-3,-4,1,5,4],[4,5,9,4,-6,-9,-3,3,-4,-3,5],[-5,6,-10,-6,10,-10,3,6,1,-10,-8],[7,-5,4,3,3,-8,-8,1,-5,-3,-7],[-4,2,7,-6,-4,-8,-5,8,-2,5,-1],[8,-6,4,-6,4,6,-5,-1,-9,3,6],[8,5,-7,9,3,2,8,9,-7,-6,-2],[-5,2,-7,-9,-3,9,8,8,5,-3,-3],[5,9,-8,4,-5,-2,-10,2,-4,-8,-2],[-10,-2,-2,-6,-9,8,10,2,5,-2,2],[9,-3,8,-7,-6,1,-6,6,-9,-2,3],[-4,-5,-2,-9,10,8,-1,3,-7,7,-10],[10,-4,10,-9,-7,10,1,10,6,-7,1]],[[3,6,-4,7,-7,-9,9,6,-4,8,-6],[4,10,-7,-4,-5,-4,-6,-7,2,3,4],[1,-7,-2,9,-5,-5,2,-4,-10,-3,1],[1,-6,1,-4,-9,-2,10,-1,-3,1,-3],[-9,9,-5,4,-6,5,9,1,-8,10,7],[5,8,-6,5,-4,-6,2,7,-9,-2,-7],[8,-4,-8,-8,4,-5,-9,-3,-2,-9,-10],[-8,-6,2,-1,8,-6,-8,5,-1,7,-1],[-9,-3,1,-7,-2,10,4,-7,-10,4,7],[2,8,-1,1,3,8,3,7,-2,9,-8],[-5,-7,10,-3,-5,-1,10,8,-7,3,7],[7,-8,-3,-3,9,8,5,1,-6,-4,-9],[-8,8,-10,-3,9,-2,-10,-3,8,3,7],[-3,-5,-5,6,5,-3,-1,4,-3,-3,1]],[[2,8,-5,7,10,-2,-8,6,-5,10,-4],[2,-5,2,-9,-3,8,-10,-4,-6,-6,-1],[-9,10,10,1,-1,1,-9,-5,-9,-9,8],[9,-8,-8,7,-1,-6,10,3,9,6,1],[6,-4,-2,9,2,-4,7,7,1,10,-10],[-3,-2,6,-4,-3,-1,-3,-8,5,-2,-4],[-2,3,-8,-1,3,-9,8,-3,2,-5,3],[-8,-2,-2,-3,-5,7,8,-1,2,-5,-9],[8,-7,9,3,-2,2,-5,10,3,5,-5],[3,-2,-2,-8,1,-4,-6,4,5,-2,10],[3,5,7,-6,-5,-8,6,5,2,9,8],[-1,7,-6,4,6,8,-5,-3,7,10,4],[-7,9,5,-5,5,5,1,1,-9,-7,10],[4,-3,-1,8,8,-4,-7,6,4,-8,-5]],[[10,8,-5,-10,7,8,5,-2,3,-2,-6],[8,1,-3,-8,-1,4,-2,2,-4,4,-8],[9,5,9,-1,-6,-7,10,7,-10,-8,6],[3,-2,9,-7,8,-5,2,-8,-1,1,-1],[-9,10,-7,-4,-8,2,-4,-7,-6,-8,8],[6,-4,-10,4,7,2,-5,-6,1,-5,-6],[-6,1,10,4,1,1,9,-10,7,-8,-5],[-7,1,-4,2,7,-5,-1,-8,-10,-7,4],[-6,-10,-4,-9,3,-9,-5,3,5,5,-8],[4,1,3,-8,-5,1,-5,-6,-5,2,5],[9,5,10,5,-6,2,-6,9,-3,-6,2],[1,-10,8,10,-9,5,-4,-4,3,-1,-9],[-3,9,-3,-2,8,2,10,10,-7,-9,-3],[-7,6,-8,6,-4,-8,-5,-9,7,-8,7]],[[-10,8,-8,-10,-3,-9,-5,-4,6,10,-2],[-6,-9,10,1,7,-6,9,-3,-4,-3,3],[-10,1,-6,2,3,9,-9,1,-6,-3,3],[-6,-10,3,3,-3,7,-1,-4,-3,10,-1],[2,-1,2,1,-6,4,-7,-1,-6,-5,10],[-8,-8,-3,4,7,8,4,-2,5,-4,-3],[-7,3,-9,-4,6,-5,-4,5,4,4,2],[4,-5,-1,10,-9,-8,-8,-1,-8,-8,-5],[1,2,4,1,-6,-9,6,9,3,2,-10],[-10,7,-10,-1,7,5,-1,-5,-4,-7,-1],[2,-2,-8,10,-7,9,-10,-6,5,5,3],[8,8,9,4,2,-10,5,-7,7,7,7],[-6,8,-8,-1,7,-7,3,2,-7,-5,7],[6,5,9,9,9,10,-9,8,-9,9,-2]],[[3,-10,-8,-9,5,10,-9,3,-6,-5,10],[7,4,-5,-2,-7,-4,-7,-2,-9,4,-10],[-2,-2,9,-7,3,-9,1,-10,-2,10,-6],[10,5,-6,1,-3,2,3,5,10,-5,-5],[-2,1,-3,-2,10,-6,3,6,-1,9,4],[-5,-2,-5,2,5,-4,-10,7,-6,3,-4],[4,-10,-6,-8,-1,-9,6,1,-3,-3,-5],[6,6,1,-3,-6,2,2,7,8,3,-10],[-1,-10,-9,5,-8,6,-2,3,-9,-2,7],[10,-6,-1,-8,-1,-1,-7,-5,-4,3,7],[6,-1,-1,7,9,-1,10,5,2,-8,6],[-5,8,-4,-1,4,-10,-2,8,-8,-9,-1],[10,6,1,-7,9,-7,-1,-6,3,8,7],[-4,-1,1,3,8,5,5,8,-4,-4,-4]]], dtype='int16')
module1.set_input('var_337', input_337)
input_343= np.array([[[-8.645921,2.697940,1.537957,-5.327475,7.363347,-3.693041,-7.382070,3.606899,2.823678,-0.893715,-1.670673],[2.597064,7.516619,-5.446015,-1.396045,4.520321,-5.408655,0.189048,8.919889,-4.682269,-4.804703,-7.770639],[-4.205601,2.777864,-7.910073,4.617976,-0.264747,3.818830,9.933178,8.693424,2.292572,-5.622347,5.478201],[3.046584,5.680670,-2.161912,3.132027,1.877043,-2.515141,0.961191,-2.326717,-2.075803,8.260063,-5.864765],[6.700772,-4.174592,-9.127269,0.285988,9.725916,7.390370,-5.218028,-2.765094,-9.503765,-3.478478,-3.403831],[5.420780,-5.328535,4.356066,-1.999073,-3.098737,-2.332496,-7.986760,3.145441,-3.347540,-5.946847,-1.257952],[7.231411,-3.570423,-7.538555,7.051327,-4.019452,8.521399,-0.643942,5.464554,-5.860863,-0.660071,-2.626711],[1.582590,-9.427430,-4.519354,3.350905,-5.487171,7.054043,-3.300975,5.871331,-6.433945,0.743531,-5.136144],[4.082982,-3.056317,-2.610504,-6.477481,0.476637,-4.295427,0.129788,-1.983810,8.035082,-0.671035,-3.879900],[-1.555518,-8.457552,7.593565,-1.991184,7.934105,0.624320,9.943746,-3.269709,0.902289,-5.787360,-7.634715],[-5.811930,-9.187768,8.369918,4.824600,-3.219084,4.314851,-4.229091,-7.571043,-2.294292,-9.393207,5.589979],[2.067155,3.690250,2.449374,8.001736,-9.542874,-1.049178,6.066936,4.848370,5.949810,3.393353,-5.998142],[-7.603430,-0.736992,-4.350435,-2.684771,4.386148,6.008806,-3.321078,-7.554267,-4.622602,-7.559759,9.245916],[-3.756265,-0.945183,-7.562927,-8.050859,3.362570,5.431649,-2.513603,7.705277,1.148324,-8.443297,5.009750]],[[9.012822,-8.207797,-6.699447,2.696107,1.659695,-2.764233,5.329686,4.808598,5.420571,-6.307691,-5.561670],[-6.783751,-3.042199,-0.754097,7.417349,6.098429,1.512457,-2.832327,-5.083872,-9.119486,-3.976223,4.530089],[2.888081,3.683649,-1.309239,-0.033353,-3.207853,-5.084468,-1.045335,-4.697148,-4.527145,2.867708,1.224318],[-9.653082,9.596598,7.821380,0.489136,8.858046,6.277683,-0.953330,9.270550,1.098892,8.011271,-9.254352],[-9.101349,0.824657,-2.336586,5.753788,2.950942,-7.737343,-3.848887,-5.367502,-7.283938,-0.616278,-0.658261],[5.837030,0.616640,-4.986604,-4.102928,3.074757,1.138700,-2.483166,2.910670,9.771912,7.018861,0.932355],[-5.820530,3.575963,1.173704,-1.490447,-2.415270,3.460995,-5.711256,3.705539,-4.502091,-1.317209,0.964375],[-8.936512,-1.651346,4.037398,4.231210,2.280536,-7.559473,-4.981788,-4.625440,-7.694854,8.172448,0.512686],[6.746987,-2.362816,9.307781,2.225483,8.341492,7.855389,8.477433,3.818708,3.048633,-6.814056,-5.096512],[7.128731,-5.778684,0.686575,-5.899791,4.987973,6.903320,4.354297,-4.122724,-1.228369,-1.460934,2.367207],[-3.373722,-1.181662,1.006744,-3.389734,-5.076235,4.406512,0.860479,-4.714208,1.624039,0.324215,1.670572],[5.277101,4.972666,-4.875768,5.968938,6.204908,-6.671943,4.134377,9.759590,6.636901,1.638955,-2.876741],[-3.900990,-0.055344,-8.160856,-7.934652,6.095650,-6.343526,1.242090,-3.016360,-8.649742,4.375986,-0.195775],[0.641962,-2.127350,5.320723,0.860049,-5.974241,-9.232172,5.487287,-4.494703,-7.501532,4.032308,-5.421122]],[[1.492327,4.882045,-9.719856,9.172043,1.785577,-0.008652,7.622270,2.643910,3.983483,-2.860075,7.713644],[6.110894,-9.285137,9.620900,-8.610757,-3.173132,-9.679994,9.276338,-4.412697,5.613931,0.251608,1.427971],[2.305969,4.195377,-3.736657,-4.016969,9.242723,8.481684,-8.933758,7.171131,-5.288763,-4.976260,0.112094],[-4.605205,-7.127668,3.346142,0.906141,5.033696,-5.342859,-4.175870,-6.457537,-1.557408,-4.253289,5.348254],[-2.317771,3.412363,9.956480,3.627272,3.133987,-1.623766,-2.774958,-2.154817,1.401785,-0.771336,-8.833398],[-0.210540,-9.395380,3.476011,5.636828,9.257233,0.363071,-3.719107,-0.434755,-3.662961,6.382229,-9.244121],[-1.204714,-0.753102,-5.906777,1.947253,8.646010,9.240257,-7.794591,-7.666968,-8.025956,8.180396,-4.438443],[-1.546358,-8.114165,-9.946339,4.739824,0.428652,4.567298,5.581621,-0.704375,2.117864,6.399136,2.133332],[7.256729,6.235893,3.357060,2.936057,-2.224678,6.133242,-0.945678,2.684085,-5.953031,-2.374116,-2.230694],[3.674767,-0.767718,-9.639190,3.900257,-7.862775,7.208829,-7.965041,-6.557130,-2.852198,1.611000,1.247057],[5.078347,-9.766344,9.393933,-3.776667,-2.463190,6.808333,-9.681061,-2.547927,4.708349,-1.838628,-1.469738],[-2.502432,-9.412532,-7.712878,-5.524741,-1.541729,-6.371080,3.328308,1.257981,2.873676,3.199516,-3.912341],[-0.145987,5.013720,6.991321,-0.113363,0.034829,3.818513,5.341432,9.182821,8.759914,7.886430,-2.908647],[-3.568614,-6.707081,-1.425383,-3.644945,6.864402,5.284279,-8.255467,-7.595292,5.844834,1.095455,-4.624459]],[[-3.401527,-1.971156,6.342619,4.689472,-1.539031,9.775647,1.707625,-2.103441,-2.384467,-2.739461,0.254005],[6.147615,5.641231,0.851022,4.381700,0.047050,8.244348,7.037677,6.266690,8.912045,-8.453803,9.443207],[-8.555800,5.286229,0.176248,-3.048291,9.249487,6.831009,-0.780752,0.419373,-3.102090,8.295653,9.028189],[-2.078272,7.009166,-1.072023,-5.621052,5.893631,-1.783039,2.019290,-6.440350,-1.403558,2.481885,-6.515992],[-7.674302,0.089331,-6.984623,-9.923795,-5.058835,4.299326,-8.553186,2.585585,0.521803,0.132066,-9.669391],[9.076599,3.161054,-2.295572,5.379253,-1.538380,-1.517998,-7.073598,1.021279,-8.572192,7.765594,9.732483],[-4.823786,-1.910155,-9.811430,5.593392,2.934398,0.381490,-7.848411,8.200252,6.365131,1.501516,-5.403299],[-6.003266,-4.023963,6.982444,-9.985305,-1.936860,5.456182,-2.358424,-1.034555,-2.653216,4.011956,-1.057802],[9.215166,9.928762,-0.204684,9.520912,6.895528,-3.540435,-1.442243,4.279228,-6.150555,2.694295,9.299786],[4.761651,2.167229,5.107328,-8.700952,-0.080177,2.682199,-0.262441,-8.293644,-4.406313,2.651887,-1.596433],[-9.729168,2.358024,2.083814,4.389750,-7.576736,-2.082161,-5.544747,-0.685222,-2.622250,5.466979,3.995732],[7.027243,0.738739,9.008202,-6.565463,-9.309106,3.625937,4.075528,7.472220,1.376153,-0.627822,2.319190],[0.685197,3.009464,2.612592,-0.461945,-6.986097,-8.086510,1.529395,-2.861334,9.782052,1.729619,-1.792295],[0.785545,6.471431,-3.508211,2.697425,-0.355391,6.428129,-4.644366,-3.264449,7.316081,-1.228771,-4.017876]],[[6.654656,7.035064,-3.550328,-6.255948,3.808790,-8.123219,-7.374497,-4.498110,-8.443754,4.012481,-5.693721],[-8.874270,-8.852683,3.473480,-9.922630,-3.462335,5.062117,-7.358760,-2.102034,-9.061797,3.507331,-3.061754],[5.998432,5.755971,-6.076395,-6.329589,-2.440620,1.394039,-0.446642,6.616673,2.263674,1.540102,2.463830],[-5.636860,-2.449665,4.282887,2.786965,7.700331,3.183687,-9.369563,6.667764,-8.673980,-6.551455,-0.721555],[3.456842,2.166846,-0.734275,1.183718,-6.320516,7.831883,-7.560768,-0.294917,6.240082,2.597332,-3.153910],[-3.212265,-1.673718,8.620500,-7.740469,5.707306,8.381064,-2.364726,0.628719,4.299687,2.808661,-5.773702],[-3.333937,7.665814,6.278153,-4.419363,1.856601,9.438198,9.264901,-8.088614,5.104585,2.159045,-4.773352],[-7.069913,9.795299,8.876493,-9.987701,4.937749,0.085620,-9.183365,9.503406,-3.086744,0.418827,-4.363777],[0.759577,5.124688,-8.111743,9.619346,-6.926397,2.853251,-3.625683,7.197395,-3.456532,0.608309,5.186135],[-9.098288,-9.949832,6.060392,2.558390,-8.472100,-3.599206,7.054953,9.102539,5.860536,-1.752221,3.265479],[-8.381773,7.441882,8.412449,-1.774469,5.551194,9.680578,6.717576,2.439569,-2.867400,6.840760,4.166462],[-7.319085,-4.167410,1.638131,-6.444644,3.932389,8.076777,9.059593,9.136221,4.811080,3.052031,-8.690559],[-4.825063,6.396006,2.818552,2.009240,9.643509,3.695432,-2.869753,-7.986776,-1.361100,7.433565,-7.431245],[-6.911747,-8.474642,-0.024028,2.754752,-3.714876,8.193445,-8.694051,2.335502,6.935450,4.544723,-5.161638]],[[5.562924,-7.291102,-6.063383,0.543989,9.758809,-8.733520,-6.917789,3.783670,7.457247,1.920937,0.937892],[5.891428,8.738250,7.905264,-4.547016,-7.097132,-2.803522,2.393428,-3.165507,0.917295,-7.588506,1.911799],[8.715341,-0.781546,2.273766,6.818961,-0.980500,8.427992,5.726217,-7.798847,-8.774209,-3.349362,-4.857091],[9.952519,-3.604179,-5.292514,-5.368605,6.211062,-5.658662,-3.478772,9.628193,-9.168796,-6.144522,-5.138047],[-9.320298,1.623786,3.314937,4.505980,2.607440,3.671331,1.135016,1.991947,6.078689,8.071979,-2.893205],[2.197133,6.299583,3.195449,9.033649,-3.065350,6.257923,-9.672156,7.314813,-0.677789,-2.869161,-1.489084],[0.361000,0.447730,0.141998,-1.296503,-7.226423,8.627591,3.259096,-9.829658,-9.223672,-7.518951,-0.623395],[4.792583,-1.952327,4.770952,3.053604,-6.222216,9.716150,5.019719,4.291038,-9.222630,-0.139799,1.228798],[0.647427,-3.888167,-0.213449,1.832428,-4.521108,-3.687997,-8.541323,8.501542,-5.008551,-6.807466,-7.869265],[-5.284168,0.620114,8.799755,-8.360657,9.382181,-2.518207,-6.112847,-0.274578,-0.905148,-1.642388,7.617550],[-3.752906,-6.069895,-7.154046,-3.929868,-6.218442,-2.236630,1.031567,7.982731,0.515337,0.323489,8.889043],[-5.795060,8.567918,-8.810028,-6.850743,-1.194749,-1.453982,3.484658,9.488462,1.285328,-5.635887,6.599300],[-3.663735,4.336377,2.802118,3.103198,-8.309463,-7.441626,-3.842972,5.750657,-8.880375,2.188043,-4.680799],[-4.627195,-4.175952,-6.040283,9.031189,8.979311,-4.910002,8.783509,2.234234,-9.995580,2.587649,-5.391609]],[[7.347796,-7.322657,6.578830,6.578364,-2.581343,-9.110117,1.571961,0.763996,-6.302103,-0.263560,9.727790],[4.112444,5.811599,2.979175,-7.658559,-7.514502,9.743601,-7.762052,4.259984,0.486578,7.026143,-3.950041],[-8.473206,9.223225,-7.231318,1.193393,-3.322876,6.431479,4.474740,-6.695800,-2.100320,3.953137,5.359184],[-0.499899,6.050812,-3.115439,9.771784,-2.127190,-1.010218,9.696428,8.839367,-1.597575,6.630688,9.006551],[-2.861659,-8.922547,0.556041,0.954557,3.527546,6.100965,8.288421,9.153492,-4.194742,-6.138560,-9.984548],[-3.355846,-7.216291,6.152289,-4.900460,-9.517078,-0.906959,4.330456,8.809381,-9.591909,-7.805111,5.984578],[2.476963,9.722384,-3.233471,-2.766832,6.280117,4.141555,3.198063,0.435813,7.204339,9.709394,-1.979632],[0.187981,9.231657,6.462908,3.747664,4.250958,1.098438,2.841775,-0.530620,-0.543904,-4.785565,5.758905],[-7.466808,2.480357,3.755916,2.546154,7.599461,0.503596,-1.834626,-9.572415,2.729132,-6.731069,-9.623648],[8.625882,5.262223,-0.294631,9.180398,-7.085211,-4.773860,9.047919,-2.586496,-2.975462,-0.890734,-6.656250],[-8.930195,-2.772943,-1.521659,-4.791572,-7.907646,2.190374,-3.657311,8.473850,2.389080,3.744628,8.841694],[1.536989,-6.572558,5.129920,5.518933,8.639783,5.850310,-6.899024,3.556694,-2.567505,-2.753919,8.405318],[-5.122149,-1.281353,-1.955945,3.624044,-2.125326,8.662356,-4.358541,8.130654,-6.967200,4.007782,4.820066],[4.850962,9.277262,-2.443059,0.495114,-4.472371,0.413083,-2.076763,-3.040243,1.051947,-1.947722,1.992845]],[[-0.133642,8.058132,-2.036752,-4.642129,-2.173384,-7.735878,8.454617,9.074133,-4.759585,-3.518633,-4.441247],[-4.558476,7.262348,2.626106,-5.149617,-5.837189,-3.080460,-0.198759,1.838034,-1.480727,-4.290963,5.159700],[4.630897,2.387620,-7.288002,4.082426,-0.369265,6.441348,-3.144274,0.185844,-0.185775,3.863246,7.622859],[8.098022,-2.443968,-6.076077,-4.243567,-9.131101,-5.144323,9.850149,-9.451637,5.963148,-3.262467,-2.891345],[-4.604531,-3.942913,2.210136,6.646837,5.831533,9.342247,-9.875148,-1.229815,2.058979,-8.940514,-0.030084],[-8.841030,8.101102,2.123422,0.960539,-0.029379,-0.771255,-0.383729,-4.389699,-9.063073,-8.821855,7.489497],[-3.208631,8.393523,-7.446455,8.408131,2.980847,4.476798,-3.095906,1.746786,-0.882170,9.733510,-1.559088],[9.899666,8.196773,-3.730074,6.566924,-5.967322,4.801429,-4.497438,3.207066,7.798904,7.877318,8.434712],[1.612048,5.159210,-5.474897,-3.219854,-3.214549,2.388291,-6.510334,2.989486,3.444490,8.926361,4.544832],[9.089449,4.530768,-6.947901,2.697917,-6.196009,-7.505724,2.167464,-0.268874,8.604941,8.149264,-7.823721],[-6.292446,3.083672,3.679628,-3.412619,-5.588666,-1.956267,-6.092318,1.840655,-8.923739,6.780949,1.837311],[-5.018744,-9.971847,6.044078,-2.951594,6.890949,8.143643,2.366154,5.491200,-9.389077,9.934331,-1.735160],[8.098806,0.439574,4.206074,-1.724409,8.470632,4.617663,-4.399331,8.500192,-4.353936,5.997301,-1.747952],[6.807687,-6.800247,9.325735,-9.689016,-1.722829,7.802140,1.928640,-8.837178,-9.143335,-7.813318,9.069466]],[[-6.610331,-7.187200,-1.174403,-5.406483,-0.079338,-5.917138,8.028686,-7.463375,9.312971,4.531455,-4.252605],[-3.864322,-9.360877,4.552213,-1.457203,3.413433,9.808546,3.900007,9.578211,-8.720091,3.068883,4.225773],[9.513970,-6.427130,-8.263562,1.311040,3.237703,5.963302,-9.221651,-9.135010,3.414157,0.252364,9.725489],[-6.083896,4.394634,-3.161709,-9.421698,4.622355,4.868801,-8.153673,-8.519608,1.826320,0.431269,5.950996],[7.644486,-6.904922,-9.461084,7.666463,5.366928,-7.215653,4.781112,1.258847,-4.829197,9.732043,0.863774],[2.589100,3.734541,0.328043,9.398965,7.095522,-0.772238,-9.145691,6.246434,9.707931,0.012912,0.929911],[-6.802294,-7.738899,-7.657552,4.472952,-7.947922,5.387071,5.917784,-3.345296,3.374716,0.264492,-2.226611],[-7.123063,5.696579,-1.612679,-2.485165,7.001549,-5.502411,8.042529,3.678594,-5.464775,-6.041099,-2.408042],[-0.624252,-5.842578,-9.234963,-4.767697,7.356286,-9.455347,-1.238238,-3.163410,0.683103,2.271264,9.441438],[-8.335568,4.056106,-7.093711,-0.606953,1.271588,-8.171898,-3.190776,1.419285,7.592791,-0.686234,-9.582329],[-0.118440,9.796612,-4.121331,7.051529,1.027294,7.385231,6.946356,-3.196980,-2.318429,-2.142011,-1.890390],[9.043006,3.935661,-0.033828,-4.470880,-8.933381,4.249868,-0.800555,8.861878,-0.068665,3.368321,9.033632],[2.449282,-1.760269,-5.790287,-6.706410,-0.386115,-6.720858,-0.648407,0.550051,1.861324,1.648768,5.590541],[7.923417,-3.033350,2.162653,2.127015,-1.735027,8.948133,-9.006161,2.780308,-1.181147,-5.281995,-0.382536]],[[2.046135,2.370716,0.633943,-9.128275,7.332858,1.102609,2.814021,9.216628,-1.354204,3.229477,5.331059],[4.489304,3.909636,-6.569581,-5.623592,-1.232906,-6.693428,5.054980,6.932807,-1.055087,2.163509,-6.961256],[-9.779344,9.963483,2.767638,-8.117160,-6.841862,1.864412,4.524928,6.022593,-7.957562,-1.141013,6.621417],[5.792644,8.169495,0.220094,1.566024,3.741097,-3.841942,-5.395920,7.239215,-7.892748,-4.843649,8.524631],[2.818040,-7.903801,0.620358,-3.050096,-0.395781,9.124284,4.051591,-8.418565,-5.446392,-3.185936,-5.166215],[-0.005972,-0.944061,6.689144,-8.370225,-4.246013,9.343544,-3.335237,1.986647,3.829516,-3.152078,2.789720],[-3.853515,-3.749197,1.577611,2.151675,-2.929411,-1.792272,-8.735229,6.807532,-0.373708,-6.808091,-2.409015],[-4.440709,3.050473,1.336304,-7.209409,9.822402,-4.843979,3.505626,-3.122875,3.904471,-6.737324,7.525891],[6.187210,-9.468163,2.539007,9.409562,4.382111,1.629629,-1.283120,-6.897855,-4.547646,3.234451,6.275603],[-4.072118,9.307220,-4.595264,9.983083,-5.385904,3.875647,9.179285,0.197916,-2.772049,-5.122378,9.521848],[1.896659,-5.109121,4.666672,-5.641708,-0.722676,-5.569837,6.182265,-0.309466,3.475291,3.976443,1.776925],[6.359661,-3.957558,6.911465,6.262832,6.061822,-5.249444,-6.320880,0.287292,-8.880736,-7.708194,8.765049],[-3.890142,-0.948509,-9.307018,-4.791348,-1.184480,-6.370335,3.644520,-8.322695,-0.482294,0.453678,5.802729],[-9.108763,2.584793,-9.591035,9.205953,9.884595,8.465929,1.367037,0.843160,9.465763,7.713250,-8.511591]],[[6.985971,-3.457732,-0.146424,-3.790877,5.632238,-3.563148,-0.923902,-2.980749,-4.730388,7.966729,-8.767507],[2.850815,-6.049158,4.283258,-8.903194,-9.603070,-2.974054,3.570044,4.327845,-1.213677,-8.277875,3.312858],[-4.025885,4.623502,6.860502,-9.389117,4.465375,4.903429,1.645876,-4.718105,4.535160,0.106829,3.807780],[-5.624892,8.641797,2.816318,-9.260225,-1.128991,5.127526,8.579198,-6.465433,-3.438016,-0.213210,6.594219],[-0.451785,3.224114,-4.596859,-7.248954,-2.897665,-2.548487,9.130067,3.355738,-9.098238,-0.934441,3.681890],[-1.206314,-8.357975,-8.681570,3.612544,-1.420326,9.221620,5.779757,-5.630187,9.634843,-6.128353,-4.033648],[5.941141,8.702234,3.115481,9.694149,-5.900735,-3.077883,-1.903118,6.833207,-4.401757,-7.799969,5.412482],[-1.299121,7.930537,-6.294282,7.271050,6.461120,9.008608,2.528626,6.587477,7.119276,-1.534565,-4.247663],[-1.188475,-2.685205,5.233784,-7.229550,-2.708186,-0.189267,-4.966875,-3.842504,-5.816453,8.986983,-1.162776],[-4.077525,9.359517,8.581860,-1.555548,5.412663,6.906404,6.041540,-4.643550,-4.794342,-1.504847,-7.360206],[3.740937,7.101789,-0.352229,7.189250,1.820436,-6.169949,-8.352070,-3.811199,7.576024,9.204645,-0.174305],[4.992906,3.809713,6.562118,9.636877,-0.414764,-2.218677,-8.246170,-4.437327,-6.369715,9.231396,4.515321],[-8.512479,-6.090977,0.613907,9.248151,2.868814,-3.384115,-3.754410,-2.705632,4.176731,-0.305343,-2.883904],[-4.140669,4.373153,-2.104613,-2.987898,-3.402698,-4.226796,-2.733900,-5.999057,3.049305,9.581637,-6.697017]],[[8.250180,4.603082,-9.074097,5.934113,-0.227668,4.830021,-1.489718,-2.583912,9.738322,7.748888,-1.012212],[0.569976,3.250005,-8.716439,-4.646467,-9.047314,5.000714,-9.102707,-9.730546,6.393748,-9.655225,-3.947992],[-7.571067,6.563808,0.061972,9.349286,-7.644453,-7.003119,8.927386,-4.150247,-0.078296,-6.038946,-4.390760],[0.528274,-4.024846,1.228563,6.737475,8.921325,5.092026,-3.347694,8.453203,-0.597850,-2.096600,-3.198699],[2.245715,-7.877350,7.646498,9.798114,6.553907,-7.410227,9.351671,-1.497581,0.119839,1.912351,-2.664418],[6.203739,5.631618,-6.070311,0.504266,-8.952000,4.029877,-9.916555,2.661196,7.377540,-0.294467,-8.869131],[4.559621,-3.024083,-2.972119,7.052508,4.331380,-8.376733,-7.595821,3.218915,7.070578,-6.813383,-0.257385],[-7.888910,3.093985,-1.655696,7.105295,7.991007,-3.587885,-3.046971,9.993074,9.255024,-7.510982,-9.384120],[-3.367454,-8.664229,6.232395,6.347127,9.407586,9.646158,7.028732,-1.884847,-9.018520,5.152004,-3.603207],[6.889714,-0.379865,-9.739480,-9.363211,3.932017,-0.201495,0.116125,4.105948,4.164233,-0.666331,2.357952],[7.470347,-5.252417,2.587115,-2.821460,0.646974,-9.722583,-5.430784,3.626229,7.723899,-6.847855,-5.984953],[8.806670,-6.913791,8.311454,4.434488,-1.889826,-8.196741,-1.511504,-1.887693,3.211769,-7.597166,3.803348],[6.377006,8.950825,-9.137912,-2.016402,5.884921,-7.409228,7.460707,-1.166358,-7.808883,5.572107,-0.923865],[-1.157593,-0.735733,-0.281827,7.565928,8.580161,-7.805179,2.517534,-7.531723,-8.831069,1.662508,1.497221]],[[-5.779529,5.075136,-2.768012,4.223838,-7.804606,-5.567741,-4.935561,5.929697,-4.442003,-9.234284,-8.956259],[3.063244,1.198712,-5.906475,-6.142727,1.894870,-2.530370,4.952950,0.086165,4.209985,6.341502,-9.697464],[-1.377692,2.477531,6.438595,-2.476471,-2.623123,-5.306021,8.385850,-3.816634,2.052221,-7.567160,1.958301],[-8.458662,-7.451104,-2.905508,5.692982,2.082130,2.638346,9.533277,-7.164221,-4.479491,4.098789,0.124158],[9.430779,3.115335,9.205681,-5.837531,1.568502,-0.014445,-6.546772,8.911286,8.528660,-2.638888,-7.437966],[0.474755,4.068053,7.688118,4.865347,6.211491,-2.518993,6.586579,-1.691488,-7.387502,-4.272636,7.217205],[-7.446108,5.944153,-7.634215,-8.851936,8.036157,5.178666,4.053718,8.807199,5.220481,7.315333,8.385295],[-9.886645,5.382727,-5.055050,5.702945,-0.014519,-9.910870,4.687951,0.148262,6.035191,-8.975265,2.184668],[6.805295,-2.779485,4.535919,5.149322,-3.360747,-5.321784,-9.509358,7.218726,0.617128,8.727948,1.391468],[-2.337618,-6.459147,3.471374,-9.640744,-7.889428,-1.675987,5.747158,0.126564,-9.200012,-6.698204,7.459152],[3.084827,-6.657099,-7.626323,0.812033,-2.847602,3.565338,4.044576,8.965832,5.095254,-2.355991,-4.839695],[-0.878916,-4.147462,2.883029,7.067884,4.960942,-2.345615,4.199364,-8.472925,-3.195942,-7.563181,9.817807],[9.235307,-0.718751,5.534623,-1.708031,-3.087044,2.264452,4.136491,9.649023,3.152608,-3.852936,-5.919178],[-5.286922,-2.910676,8.612979,-4.481865,-9.361967,-6.942861,0.083997,8.514996,4.686213,-4.572460,-0.336532]],[[-1.357390,-8.611644,9.602698,4.667035,7.486361,-4.373297,-8.938727,-2.764305,-4.343686,-7.790245,5.998096],[-2.972677,1.660695,-4.163825,-1.884151,-5.610234,-6.435282,-2.043662,8.285735,2.220490,-5.566922,-9.953677],[-1.046024,9.320967,9.310831,8.624195,2.710100,-0.978791,8.279853,-5.029630,7.188146,-6.945546,-5.546129],[8.128666,-9.414665,-8.040880,-6.205359,-3.239270,8.214135,-5.213906,3.352505,-2.038917,1.625705,-1.523883],[3.131656,5.075436,4.021286,6.858931,-3.481132,4.581682,4.695840,6.036641,-2.255739,-4.961244,-9.970367],[-3.485660,-6.017132,-8.889615,2.204820,8.711114,-4.756576,-4.299015,-0.015963,2.471459,-3.680171,-0.852115],[-5.826443,-8.812536,6.383060,-1.935868,7.022548,0.618960,8.238670,1.353179,-2.821928,-9.721803,5.423911],[-3.433456,-8.691528,7.889871,-8.143586,-1.034879,6.654912,3.118856,-3.581956,-6.707883,7.112311,4.998901],[-7.156776,8.837547,0.424311,0.132524,-6.512656,8.592504,4.271879,-0.895545,-5.909451,-8.140633,5.593096],[-5.447922,6.839558,4.750449,-7.640719,-2.137192,1.413378,-7.786925,7.014761,-3.200746,7.390926,-2.465762],[-5.969420,6.800201,4.976404,-3.351802,1.278866,-9.054762,-3.337935,7.741919,1.265154,7.681345,-7.107640],[0.482408,9.423863,8.521212,8.289607,7.091051,-9.792926,-5.706504,-6.237489,-9.596750,7.643289,-0.716331],[-8.741250,0.930983,8.473559,9.791144,-2.671740,1.411932,-6.805050,-5.464978,-5.350631,2.098331,-5.315005],[1.252305,-4.292092,9.723906,-3.389653,5.794177,8.209741,-6.556984,8.092665,-7.480220,3.066400,9.751002]],[[7.677883,9.082305,3.078821,-1.143657,4.098884,-5.310963,-7.171794,1.595108,-4.009762,6.749276,0.070427],[-3.301541,8.327620,-1.410684,0.903680,-3.993373,-2.206997,-6.743793,-4.329615,5.065284,-8.103762,-1.555035],[9.140348,-9.243146,-7.196804,-4.905734,3.000772,4.771661,6.520022,1.028928,5.996253,7.121680,-0.355177],[-1.991071,1.193376,-8.305706,2.099236,-4.164658,-0.612240,-5.281387,-8.761843,9.906554,-2.479857,9.610416],[-5.231721,-5.220138,-0.882688,9.140280,3.855340,-5.070360,-4.111174,2.925410,9.772000,-5.228615,-7.752107],[-5.706996,4.240460,-3.216829,-4.432149,-9.820737,8.058945,9.142581,-9.416173,-0.707778,-0.882288,-7.093021],[3.474139,-3.007692,6.178312,-9.185893,-8.363860,8.078579,-0.316886,5.506136,6.159638,-4.528894,-3.878838],[-5.949023,-9.368947,-0.407067,5.961576,7.857077,-7.855824,6.238009,5.548524,3.661589,5.620898,8.447000],[3.080131,-1.839161,-7.389046,9.259213,-8.730534,-8.882527,-6.454872,6.401203,5.020397,-4.744382,1.165016],[-5.719791,8.524025,-5.564117,-2.186144,5.149943,-5.913467,-3.082991,-9.910657,4.177312,7.074175,5.728166],[-8.859397,-5.093706,-4.612436,-0.855136,-9.300256,-8.873832,-9.937968,-6.045256,-0.886199,-0.852260,-9.387673],[-7.787724,-1.588847,0.938275,7.412867,7.665131,-8.146992,2.892959,-8.949560,-6.670666,-8.916647,1.380340],[3.435099,0.907502,4.357333,-8.348744,-9.144661,-3.210865,-9.427143,2.209746,-4.223561,2.687981,-9.921471],[7.693170,-2.277221,2.936117,-1.100386,-7.914108,8.297659,-0.688173,3.850199,-7.766719,2.185993,2.169408]],[[-1.504180,0.946383,0.763791,8.747098,1.472003,9.450065,-3.495765,1.711258,7.299514,-4.725062,5.540002],[1.520856,4.029765,5.616202,-2.695226,-6.875853,9.458300,7.576930,6.671275,-8.654943,-5.408390,-2.984032],[-7.080355,5.372223,0.204667,-3.844843,-0.453146,-8.538995,9.025454,-5.624690,-4.193806,6.732182,-6.116299],[-1.599776,6.204356,4.496438,-2.434090,8.886845,1.179174,-1.512236,-6.624563,8.321242,-0.063125,4.115950],[9.656965,8.598677,-2.905057,8.245305,-6.032333,-2.622213,-0.950003,4.288724,-9.465685,-4.595446,7.628218],[-7.881650,7.073289,5.689928,2.731185,-2.255362,-8.146195,-4.573794,-8.288344,-4.198461,-6.261811,-5.846578],[-3.603390,2.958917,-4.717752,-6.617848,2.024325,-6.379693,-6.593537,-4.363129,-5.546809,0.136289,5.954862],[4.680728,1.581638,-2.244795,1.088930,4.183732,-4.314119,-4.163942,-3.411456,-2.434723,7.394518,-3.923044],[-4.601415,-6.305473,4.035200,-1.209170,1.166011,-3.330750,4.596266,-8.204764,-4.550785,-7.402409,0.076820],[0.837264,7.551133,3.223337,-6.384740,-7.350394,-5.407962,7.748520,-7.757703,-4.025604,-7.212680,8.809274],[5.506611,-4.550240,0.515416,-4.183160,2.640164,-8.842771,-9.508872,9.425539,5.229975,9.367413,-4.198377],[-4.914580,-5.284325,2.343057,0.579846,-4.276590,-5.685152,-7.463586,4.555745,-8.762465,9.536222,0.351726],[-5.020337,-1.726659,2.747855,-7.704605,0.232645,-7.573400,0.010086,-8.771860,-6.775454,-2.626506,0.427699],[-1.454466,1.948137,3.902565,9.069087,-3.892438,6.202086,-5.375586,8.861130,1.431419,6.891341,-3.984435]]], dtype='float32')
module1.set_input('var_343', input_343)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_336, input_337, input_343, )
res3 = intrp3.evaluate()(input_336, input_337, input_343, )
res4 = intrp4.evaluate()(input_336, input_337, input_343, )
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
module5.set_input('var_336', input_336)
module5.set_input('var_337', input_337)
module5.set_input('var_343', input_343)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_336, input_337, input_343, )
res7 = intrp7.evaluate()(input_336, input_337, input_343, )
res8 = intrp8.evaluate()(input_336, input_337, input_343, )
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
module9.set_input('var_336', input_336)
module9.set_input('var_337', input_337)
module9.set_input('var_343', input_343)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_336, input_337, input_343, )
res11 = intrp11.evaluate()(input_336, input_337, input_343, )
res12 = intrp12.evaluate()(input_336, input_337, input_343, )
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
module13.set_input('var_336', input_336)
module13.set_input('var_337', input_337)
module13.set_input('var_343', input_343)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_336, input_337, input_343, )
res15 = intrp15.evaluate()(input_336, input_337, input_343, )
res16 = intrp16.evaluate()(input_336, input_337, input_343, )
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
module17.set_input('var_336', input_336)
module17.set_input('var_337', input_337)
module17.set_input('var_343', input_343)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_336, input_337, input_343, )
res19 = intrp19.evaluate()(input_336, input_337, input_343, )
res20 = intrp20.evaluate()(input_336, input_337, input_343, )
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
module21.set_input('var_336', input_336)
module21.set_input('var_337', input_337)
module21.set_input('var_343', input_343)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_336, input_337, input_343, )
res23 = intrp23.evaluate()(input_336, input_337, input_343, )
res24 = intrp24.evaluate()(input_336, input_337, input_343, )
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