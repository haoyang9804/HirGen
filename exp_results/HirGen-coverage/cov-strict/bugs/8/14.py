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
var_0 = relay.var("var_0", dtype = "float32", shape = (5, 4))#candidate|0|(5, 4)|var|float32
var_1 = relay.var("var_1", dtype = "float32", shape = (5, 4))#candidate|1|(5, 4)|var|float32
bop_2 = relay.power(var_0.astype('float32'), relay.reshape(var_1.astype('float32'), relay.shape_of(var_0))) # shape=(5, 4)
bop_5 = relay.bitwise_and(var_1.astype('int8'), relay.reshape(bop_2.astype('int8'), relay.shape_of(var_1))) # shape=(5, 4)
bop_8 = relay.add(var_0.astype('int8'), relay.reshape(bop_2.astype('int8'), relay.shape_of(var_0))) # shape=(5, 4)
uop_11 = relay.sinh(bop_5.astype('float64')) # shape=(5, 4)
output = relay.Tuple([bop_8,uop_11,])
output2 = relay.Tuple([bop_8,uop_11,])
func_13 = relay.Function([var_0,var_1,], output)
mod['func_13'] = func_13
mod = relay.transform.InferType()(mod)
mutated_mod['func_13'] = func_13
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13_call = mutated_mod.get_global_var('func_13')
var_15 = relay.var("var_15", dtype = "float32", shape = (5, 4))#candidate|15|(5, 4)|var|float32
var_16 = relay.var("var_16", dtype = "float32", shape = (5, 4))#candidate|16|(5, 4)|var|float32
call_14 = func_13_call(var_15,var_16,)
output = call_14
func_17 = relay.Function([var_15,var_16,], output)
mutated_mod['func_17'] = func_17
mutated_mod = relay.transform.InferType()(mutated_mod)
const_19 = relay.const(1.936042, dtype = "float32")#candidate|19|()|const|float32
var_20 = relay.var("var_20", dtype = "float32", shape = ())#candidate|20|()|var|float32
bop_21 = relay.equal(const_19.astype('bool'), var_20.astype('bool')) # shape=()
uop_24 = relay.sigmoid(const_19.astype('float64')) # shape=()
bop_26 = relay.equal(uop_24.astype('bool'), var_20.astype('bool')) # shape=()
output = relay.Tuple([bop_21,bop_26,])
output2 = relay.Tuple([bop_21,bop_26,])
func_29 = relay.Function([var_20,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
var_30 = relay.var("var_30", dtype = "float32", shape = ())#candidate|30|()|var|float32
output = func_29(var_30)
func_31 = relay.Function([var_30], output)
mutated_mod['func_31'] = func_31
mutated_mod = relay.transform.InferType()(mutated_mod)
var_33 = relay.var("var_33", dtype = "float32", shape = (11, 2))#candidate|33|(11, 2)|var|float32
uop_34 = relay.tan(var_33.astype('float32')) # shape=(11, 2)
const_36 = relay.const([[-3.870699,0.213541],[9.056479,-4.923415],[-3.601027,8.772423],[0.229063,-3.647017],[-2.550587,-6.159072],[-8.309378,8.364154],[3.420911,2.312329],[-4.061866,0.412052],[-8.229866,0.325784],[0.503196,1.328219],[0.538490,5.370295]], dtype = "float32")#candidate|36|(11, 2)|const|float32
bop_37 = relay.right_shift(uop_34.astype('int16'), relay.reshape(const_36.astype('int16'), relay.shape_of(uop_34))) # shape=(11, 2)
var_40 = relay.var("var_40", dtype = "float32", shape = (11, 2))#candidate|40|(11, 2)|var|float32
bop_41 = relay.multiply(const_36.astype('int16'), relay.reshape(var_40.astype('int16'), relay.shape_of(const_36))) # shape=(11, 2)
bop_44 = relay.greater_equal(uop_34.astype('bool'), relay.reshape(const_36.astype('bool'), relay.shape_of(uop_34))) # shape=(11, 2)
bop_47 = relay.floor_divide(uop_34.astype('float64'), relay.reshape(var_40.astype('float64'), relay.shape_of(uop_34))) # shape=(11, 2)
var_50 = relay.var("var_50", dtype = "bool", shape = (11, 2))#candidate|50|(11, 2)|var|bool
bop_51 = relay.greater(bop_44.astype('bool'), relay.reshape(var_50.astype('bool'), relay.shape_of(bop_44))) # shape=(11, 2)
func_13_call = mod.get_global_var('func_13')
func_17_call = mutated_mod.get_global_var('func_17')
const_55 = relay.const([[-3.870719,7.650829,-7.648559,3.071500,5.964210,-7.816886,6.452557,-4.662067,-6.351765,5.884036,2.358876,-5.201859,-3.395452,-0.457225,6.003906,8.831987,1.410215,2.509799,0.538054,4.714561]], dtype = "float32")#candidate|55|(1, 20)|const|float32
call_54 = relay.TupleGetItem(func_13_call(relay.reshape(const_55.astype('float32'), [5, 4]), relay.reshape(const_55.astype('float32'), [5, 4]), ), 1)
call_56 = relay.TupleGetItem(func_17_call(relay.reshape(const_55.astype('float32'), [5, 4]), relay.reshape(const_55.astype('float32'), [5, 4]), ), 1)
bop_57 = relay.bitwise_or(bop_44.astype('int32'), relay.reshape(uop_34.astype('int32'), relay.shape_of(bop_44))) # shape=(11, 2)
uop_60 = relay.sin(bop_41.astype('float64')) # shape=(11, 2)
bop_62 = relay.greater(bop_37.astype('bool'), relay.reshape(var_40.astype('bool'), relay.shape_of(bop_37))) # shape=(11, 2)
bop_65 = relay.mod(bop_51.astype('float64'), relay.reshape(uop_60.astype('float64'), relay.shape_of(bop_51))) # shape=(11, 2)
uop_68 = relay.sin(bop_37.astype('float32')) # shape=(11, 2)
uop_70 = relay.sigmoid(uop_68.astype('float64')) # shape=(11, 2)
uop_72 = relay.sinh(uop_70.astype('float64')) # shape=(11, 2)
uop_74 = relay.log2(uop_70.astype('float64')) # shape=(11, 2)
uop_76 = relay.log10(uop_74.astype('float64')) # shape=(11, 2)
bop_78 = relay.left_shift(uop_76.astype('int32'), relay.reshape(bop_57.astype('int32'), relay.shape_of(uop_76))) # shape=(11, 2)
bop_81 = relay.greater_equal(uop_76.astype('bool'), relay.reshape(bop_47.astype('bool'), relay.shape_of(uop_76))) # shape=(11, 2)
uop_84 = relay.rsqrt(bop_78.astype('float64')) # shape=(11, 2)
var_86 = relay.var("var_86", dtype = "float64", shape = (11, 2))#candidate|86|(11, 2)|var|float64
bop_87 = relay.less_equal(uop_72.astype('bool'), relay.reshape(var_86.astype('bool'), relay.shape_of(uop_72))) # shape=(11, 2)
func_29_call = mod.get_global_var('func_29')
func_31_call = mutated_mod.get_global_var('func_31')
const_91 = relay.const(-0.542074, dtype = "float32")#candidate|91|()|const|float32
call_90 = relay.TupleGetItem(func_29_call(relay.reshape(const_91.astype('float32'), [])), 0)
call_92 = relay.TupleGetItem(func_31_call(relay.reshape(const_91.astype('float32'), [])), 0)
uop_93 = relay.acos(uop_84.astype('float32')) # shape=(11, 2)
bop_95 = relay.left_shift(uop_84.astype('int8'), relay.reshape(bop_81.astype('int8'), relay.shape_of(uop_84))) # shape=(11, 2)
bop_98 = relay.add(bop_81.astype('uint32'), relay.reshape(uop_34.astype('uint32'), relay.shape_of(bop_81))) # shape=(11, 2)
output = relay.Tuple([call_54,const_55,bop_62,bop_65,bop_87,call_90,const_91,uop_93,bop_95,bop_98,])
output2 = relay.Tuple([call_56,const_55,bop_62,bop_65,bop_87,call_92,const_91,uop_93,bop_95,bop_98,])
func_101 = relay.Function([var_33,var_40,var_50,var_86,], output)
mod['func_101'] = func_101
mod = relay.transform.InferType()(mod)
mutated_mod['func_101'] = func_101
mutated_mod = relay.transform.InferType()(mutated_mod)
func_101_call = mutated_mod.get_global_var('func_101')
var_103 = relay.var("var_103", dtype = "float32", shape = (11, 2))#candidate|103|(11, 2)|var|float32
var_104 = relay.var("var_104", dtype = "float32", shape = (11, 2))#candidate|104|(11, 2)|var|float32
var_105 = relay.var("var_105", dtype = "bool", shape = (11, 2))#candidate|105|(11, 2)|var|bool
var_106 = relay.var("var_106", dtype = "float64", shape = (11, 2))#candidate|106|(11, 2)|var|float64
call_102 = func_101_call(var_103,var_104,var_105,var_106,)
output = call_102
func_107 = relay.Function([var_103,var_104,var_105,var_106,], output)
mutated_mod['func_107'] = func_107
mutated_mod = relay.transform.InferType()(mutated_mod)
var_109 = relay.var("var_109", dtype = "float64", shape = (11, 2))#candidate|109|(11, 2)|var|float64
var_110 = relay.var("var_110", dtype = "float64", shape = (11, 2))#candidate|110|(11, 2)|var|float64
bop_111 = relay.mod(var_109.astype('float64'), relay.reshape(var_110.astype('float64'), relay.shape_of(var_109))) # shape=(11, 2)
bop_114 = relay.floor_divide(var_110.astype('float32'), relay.reshape(var_109.astype('float32'), relay.shape_of(var_110))) # shape=(11, 2)
bop_117 = relay.logical_and(var_110.astype('bool'), relay.reshape(bop_111.astype('bool'), relay.shape_of(var_110))) # shape=(11, 2)
uop_120 = relay.cosh(var_109.astype('float32')) # shape=(11, 2)
bop_122 = relay.maximum(uop_120.astype('float32'), relay.reshape(bop_111.astype('float32'), relay.shape_of(uop_120))) # shape=(11, 2)
bop_125 = relay.not_equal(bop_122.astype('bool'), relay.reshape(bop_117.astype('bool'), relay.shape_of(bop_122))) # shape=(11, 2)
output = relay.Tuple([bop_114,bop_125,])
output2 = relay.Tuple([bop_114,bop_125,])
func_128 = relay.Function([var_109,var_110,], output)
mod['func_128'] = func_128
mod = relay.transform.InferType()(mod)
mutated_mod['func_128'] = func_128
mutated_mod = relay.transform.InferType()(mutated_mod)
func_128_call = mutated_mod.get_global_var('func_128')
var_130 = relay.var("var_130", dtype = "float64", shape = (11, 2))#candidate|130|(11, 2)|var|float64
var_131 = relay.var("var_131", dtype = "float64", shape = (11, 2))#candidate|131|(11, 2)|var|float64
call_129 = func_128_call(var_130,var_131,)
output = call_129
func_132 = relay.Function([var_130,var_131,], output)
mutated_mod['func_132'] = func_132
mutated_mod = relay.transform.InferType()(mutated_mod)
const_134 = relay.const(-1, dtype = "uint64")#candidate|134|()|const|uint64
const_135 = relay.const([8,8,-8,-4,8,9,-8,1,10,-7,-9,10,6,3,6,-3], dtype = "uint64")#candidate|135|(16,)|const|uint64
bop_136 = relay.greater(const_134.astype('bool'), const_135.astype('bool')) # shape=(16,)
bop_139 = relay.power(const_134.astype('float32'), const_135.astype('float32')) # shape=(16,)
bop_142 = relay.not_equal(bop_136.astype('bool'), const_134.astype('bool')) # shape=(16,)
uop_145 = relay.acosh(bop_136.astype('float32')) # shape=(16,)
bop_147 = relay.logical_and(uop_145.astype('bool'), relay.reshape(bop_136.astype('bool'), relay.shape_of(uop_145))) # shape=(16,)
bop_150 = relay.left_shift(bop_147.astype('int64'), relay.reshape(bop_139.astype('int64'), relay.shape_of(bop_147))) # shape=(16,)
uop_153 = relay.atanh(uop_145.astype('float64')) # shape=(16,)
uop_155 = relay.acos(uop_145.astype('float32')) # shape=(16,)
bop_157 = relay.minimum(bop_139.astype('uint8'), const_134.astype('uint8')) # shape=(16,)
bop_160 = relay.logical_or(uop_145.astype('bool'), relay.reshape(bop_157.astype('bool'), relay.shape_of(uop_145))) # shape=(16,)
bop_163 = relay.less_equal(uop_153.astype('bool'), relay.reshape(bop_142.astype('bool'), relay.shape_of(uop_153))) # shape=(16,)
output = relay.Tuple([bop_150,uop_155,bop_160,bop_163,])
output2 = relay.Tuple([bop_150,uop_155,bop_160,bop_163,])
func_166 = relay.Function([], output)
mod['func_166'] = func_166
mod = relay.transform.InferType()(mod)
mutated_mod['func_166'] = func_166
mutated_mod = relay.transform.InferType()(mutated_mod)
func_166_call = mutated_mod.get_global_var('func_166')
call_167 = func_166_call()
output = call_167
func_168 = relay.Function([], output)
mutated_mod['func_168'] = func_168
mutated_mod = relay.transform.InferType()(mutated_mod)
var_169 = relay.var("var_169", dtype = "uint8", shape = (1,))#candidate|169|(1,)|var|uint8
const_170 = relay.const([5,-1,4,7,-4,-9,6,3,-7,8,-10,4,7,8], dtype = "uint8")#candidate|170|(14,)|const|uint8
bop_171 = relay.minimum(var_169.astype('uint8'), const_170.astype('uint8')) # shape=(14,)
bop_174 = relay.multiply(var_169.astype('int64'), bop_171.astype('int64')) # shape=(14,)
func_29_call = mod.get_global_var('func_29')
func_31_call = mutated_mod.get_global_var('func_31')
call_177 = relay.TupleGetItem(func_29_call(relay.reshape(var_169.astype('float32'), [])), 1)
call_178 = relay.TupleGetItem(func_31_call(relay.reshape(var_169.astype('float32'), [])), 1)
uop_179 = relay.cosh(bop_174.astype('float32')) # shape=(14,)
var_181 = relay.var("var_181", dtype = "float32", shape = (14,))#candidate|181|(14,)|var|float32
bop_182 = relay.bitwise_or(uop_179.astype('uint8'), relay.reshape(var_181.astype('uint8'), relay.shape_of(uop_179))) # shape=(14,)
bop_185 = relay.power(var_181.astype('float32'), var_169.astype('float32')) # shape=(14,)
uop_188 = relay.asin(uop_179.astype('float64')) # shape=(14,)
var_190 = relay.var("var_190", dtype = "float32", shape = (14,))#candidate|190|(14,)|var|float32
bop_191 = relay.bitwise_xor(uop_179.astype('uint8'), relay.reshape(var_190.astype('uint8'), relay.shape_of(uop_179))) # shape=(14,)
var_194 = relay.var("var_194", dtype = "float64", shape = (14,))#candidate|194|(14,)|var|float64
bop_195 = relay.right_shift(uop_188.astype('uint32'), relay.reshape(var_194.astype('uint32'), relay.shape_of(uop_188))) # shape=(14,)
bop_198 = relay.floor_mod(uop_188.astype('float32'), relay.reshape(const_170.astype('float32'), relay.shape_of(uop_188))) # shape=(14,)
bop_201 = relay.right_shift(bop_182.astype('uint16'), relay.reshape(const_170.astype('uint16'), relay.shape_of(bop_182))) # shape=(14,)
uop_204 = relay.cos(bop_195.astype('float32')) # shape=(14,)
uop_206 = relay.log(uop_188.astype('float64')) # shape=(14,)
bop_208 = relay.floor_mod(var_169.astype('float32'), var_190.astype('float32')) # shape=(14,)
bop_211 = relay.minimum(uop_204.astype('uint32'), relay.reshape(uop_179.astype('uint32'), relay.shape_of(uop_204))) # shape=(14,)
const_214 = relay.const([7,10,8,4,-9,4,-6,3,-8,6,-8,8,-2,6], dtype = "uint32")#candidate|214|(14,)|const|uint32
bop_215 = relay.less_equal(bop_211.astype('bool'), relay.reshape(const_214.astype('bool'), relay.shape_of(bop_211))) # shape=(14,)
bop_218 = relay.greater(bop_211.astype('bool'), var_169.astype('bool')) # shape=(14,)
bop_221 = relay.logical_or(uop_204.astype('bool'), relay.reshape(bop_174.astype('bool'), relay.shape_of(uop_204))) # shape=(14,)
uop_224 = relay.atanh(bop_198.astype('float32')) # shape=(14,)
const_226 = relay.const([0.025657,6.884842,4.870121,-8.508311,-0.076969,7.604705,1.440099,3.076195,2.283080,-0.026719,9.722101,-8.496888,6.052770,0.067390], dtype = "float64")#candidate|226|(14,)|const|float64
bop_227 = relay.mod(uop_206.astype('float32'), relay.reshape(const_226.astype('float32'), relay.shape_of(uop_206))) # shape=(14,)
bop_230 = relay.multiply(var_190.astype('float64'), relay.reshape(bop_221.astype('float64'), relay.shape_of(var_190))) # shape=(14,)
uop_233 = relay.acos(bop_218.astype('float32')) # shape=(14,)
var_235 = relay.var("var_235", dtype = "float32", shape = (14,))#candidate|235|(14,)|var|float32
bop_236 = relay.bitwise_and(uop_233.astype('int8'), relay.reshape(var_235.astype('int8'), relay.shape_of(uop_233))) # shape=(14,)
uop_239 = relay.sqrt(bop_198.astype('float32')) # shape=(14,)
bop_241 = relay.logical_xor(bop_236.astype('uint8'), relay.reshape(var_190.astype('uint8'), relay.shape_of(bop_236))) # shape=(14,)
uop_244 = relay.sigmoid(uop_233.astype('float64')) # shape=(14,)
bop_246 = relay.floor_divide(uop_204.astype('float32'), relay.reshape(bop_227.astype('float32'), relay.shape_of(uop_204))) # shape=(14,)
bop_249 = relay.add(uop_244.astype('uint8'), relay.reshape(bop_230.astype('uint8'), relay.shape_of(uop_244))) # shape=(14,)
func_128_call = mod.get_global_var('func_128')
func_132_call = mutated_mod.get_global_var('func_132')
const_253 = relay.const([-3.226770,-5.760572,2.135757,-7.070503,7.945820,6.282624,-4.102918,-1.858138,7.133181,1.387650,-3.961833,4.058757,4.088263,-1.160195,-3.128051,-2.247659,-8.436001,-7.041798,-3.770947,-7.439583,1.045192,7.402886], dtype = "float64")#candidate|253|(22,)|const|float64
call_252 = relay.TupleGetItem(func_128_call(relay.reshape(const_253.astype('float64'), [11, 2]), relay.reshape(const_253.astype('float64'), [11, 2]), ), 1)
call_254 = relay.TupleGetItem(func_132_call(relay.reshape(const_253.astype('float64'), [11, 2]), relay.reshape(const_253.astype('float64'), [11, 2]), ), 1)
bop_255 = relay.bitwise_or(bop_249.astype('int32'), relay.reshape(var_235.astype('int32'), relay.shape_of(bop_249))) # shape=(14,)
var_258 = relay.var("var_258", dtype = "bool", shape = (11, 2))#candidate|258|(11, 2)|var|bool
bop_259 = relay.floor_divide(call_252.astype('float32'), relay.reshape(var_258.astype('float32'), relay.shape_of(call_252))) # shape=(11, 2)
bop_262 = relay.floor_divide(call_254.astype('float32'), relay.reshape(var_258.astype('float32'), relay.shape_of(call_254))) # shape=(11, 2)
bop_263 = relay.bitwise_and(uop_244.astype('int16'), relay.reshape(uop_233.astype('int16'), relay.shape_of(uop_244))) # shape=(14,)
func_13_call = mod.get_global_var('func_13')
func_17_call = mutated_mod.get_global_var('func_17')
const_267 = relay.const([-9.323646,-5.691127,-1.032139,6.530501,7.913674,6.903957,-4.976530,7.250660,-2.032178,-8.561028,8.510327,3.821775,2.965368,3.825550,8.875345,-2.870425,7.485798,-2.697249,-3.611251,1.280480], dtype = "float32")#candidate|267|(20,)|const|float32
call_266 = relay.TupleGetItem(func_13_call(relay.reshape(const_267.astype('float32'), [5, 4]), relay.reshape(const_267.astype('float32'), [5, 4]), ), 0)
call_268 = relay.TupleGetItem(func_17_call(relay.reshape(const_267.astype('float32'), [5, 4]), relay.reshape(const_267.astype('float32'), [5, 4]), ), 0)
bop_269 = relay.logical_xor(bop_263.astype('int8'), relay.reshape(uop_188.astype('int8'), relay.shape_of(bop_263))) # shape=(14,)
uop_272 = relay.tan(bop_263.astype('float64')) # shape=(14,)
bop_274 = relay.add(uop_272.astype('int16'), relay.reshape(bop_198.astype('int16'), relay.shape_of(uop_272))) # shape=(14,)
func_13_call = mod.get_global_var('func_13')
func_17_call = mutated_mod.get_global_var('func_17')
call_277 = relay.TupleGetItem(func_13_call(relay.reshape(const_267.astype('float32'), [5, 4]), relay.reshape(call_266.astype('float32'), [5, 4]), ), 1)
call_278 = relay.TupleGetItem(func_17_call(relay.reshape(const_267.astype('float32'), [5, 4]), relay.reshape(call_266.astype('float32'), [5, 4]), ), 1)
bop_279 = relay.subtract(bop_255.astype('uint32'), relay.reshape(bop_246.astype('uint32'), relay.shape_of(bop_255))) # shape=(14,)
uop_282 = relay.log10(uop_272.astype('float32')) # shape=(14,)
var_284 = relay.var("var_284", dtype = "int8", shape = (14,))#candidate|284|(14,)|var|int8
bop_285 = relay.add(bop_269.astype('int32'), relay.reshape(var_284.astype('int32'), relay.shape_of(bop_269))) # shape=(14,)
uop_288 = relay.sqrt(uop_272.astype('float64')) # shape=(14,)
bop_290 = relay.not_equal(bop_274.astype('bool'), relay.reshape(bop_195.astype('bool'), relay.shape_of(bop_274))) # shape=(14,)
uop_293 = relay.acosh(uop_282.astype('float32')) # shape=(14,)
uop_295 = relay.atan(bop_274.astype('float32')) # shape=(14,)
bop_297 = relay.equal(uop_282.astype('bool'), relay.reshape(var_235.astype('bool'), relay.shape_of(uop_282))) # shape=(14,)
output = relay.Tuple([call_177,bop_185,bop_191,bop_201,bop_208,bop_215,uop_224,uop_239,bop_241,const_253,bop_259,call_266,const_267,call_277,bop_279,bop_285,uop_288,bop_290,uop_293,uop_295,bop_297,])
output2 = relay.Tuple([call_178,bop_185,bop_191,bop_201,bop_208,bop_215,uop_224,uop_239,bop_241,const_253,bop_262,call_268,const_267,call_278,bop_279,bop_285,uop_288,bop_290,uop_293,uop_295,bop_297,])
func_300 = relay.Function([var_169,var_181,var_190,var_194,var_235,var_258,var_284,], output)
mod['func_300'] = func_300
mod = relay.transform.InferType()(mod)
var_301 = relay.var("var_301", dtype = "uint8", shape = (1,))#candidate|301|(1,)|var|uint8
var_302 = relay.var("var_302", dtype = "float32", shape = (14,))#candidate|302|(14,)|var|float32
var_303 = relay.var("var_303", dtype = "float32", shape = (14,))#candidate|303|(14,)|var|float32
var_304 = relay.var("var_304", dtype = "float64", shape = (14,))#candidate|304|(14,)|var|float64
var_305 = relay.var("var_305", dtype = "float32", shape = (14,))#candidate|305|(14,)|var|float32
var_306 = relay.var("var_306", dtype = "bool", shape = (11, 2))#candidate|306|(11, 2)|var|bool
var_307 = relay.var("var_307", dtype = "int8", shape = (14,))#candidate|307|(14,)|var|int8
output = func_300(var_301,var_302,var_303,var_304,var_305,var_306,var_307,)
func_308 = relay.Function([var_301,var_302,var_303,var_304,var_305,var_306,var_307,], output)
mutated_mod['func_308'] = func_308
mutated_mod = relay.transform.InferType()(mutated_mod)
const_310 = relay.const([-2.623165,1.280760,9.501852,-4.290344,-3.917868,-4.925362,-9.504598,-7.871977,-5.944467,7.366909,-8.796623,-8.774829,3.447341,0.069395,0.961416], dtype = "float64")#candidate|310|(15,)|const|float64
var_311 = relay.var("var_311", dtype = "float64", shape = (15,))#candidate|311|(15,)|var|float64
bop_312 = relay.divide(const_310.astype('float64'), relay.reshape(var_311.astype('float64'), relay.shape_of(const_310))) # shape=(15,)
uop_315 = relay.log2(bop_312.astype('float32')) # shape=(15,)
bop_317 = relay.mod(var_311.astype('float64'), relay.reshape(bop_312.astype('float64'), relay.shape_of(var_311))) # shape=(15,)
uop_320 = relay.log10(bop_312.astype('float64')) # shape=(15,)
bop_322 = relay.logical_xor(bop_317.astype('int32'), relay.reshape(var_311.astype('int32'), relay.shape_of(bop_317))) # shape=(15,)
bop_325 = relay.power(uop_315.astype('float64'), relay.reshape(uop_320.astype('float64'), relay.shape_of(uop_315))) # shape=(15,)
uop_328 = relay.atan(const_310.astype('float64')) # shape=(15,)
output = relay.Tuple([bop_322,bop_325,uop_328,])
output2 = relay.Tuple([bop_322,bop_325,uop_328,])
func_330 = relay.Function([var_311,], output)
mod['func_330'] = func_330
mod = relay.transform.InferType()(mod)
mutated_mod['func_330'] = func_330
mutated_mod = relay.transform.InferType()(mutated_mod)
var_331 = relay.var("var_331", dtype = "float64", shape = (15,))#candidate|331|(15,)|var|float64
func_330_call = mutated_mod.get_global_var('func_330')
call_332 = func_330_call(var_331)
output = call_332
func_333 = relay.Function([var_331], output)
mutated_mod['func_333'] = func_333
mutated_mod = relay.transform.InferType()(mutated_mod)
var_335 = relay.var("var_335", dtype = "bool", shape = ())#candidate|335|()|var|bool
const_336 = relay.const(False, dtype = "bool")#candidate|336|()|const|bool
bop_337 = relay.logical_or(var_335.astype('bool'), const_336.astype('bool')) # shape=()
bop_340 = relay.floor_divide(var_335.astype('float64'), const_336.astype('float64')) # shape=()
uop_343 = relay.sinh(bop_340.astype('float32')) # shape=()
uop_345 = relay.log(uop_343.astype('float32')) # shape=()
bop_347 = relay.bitwise_and(uop_343.astype('uint32'), bop_337.astype('uint32')) # shape=()
uop_350 = relay.acosh(uop_345.astype('float32')) # shape=()
output = relay.Tuple([bop_347,uop_350,])
output2 = relay.Tuple([bop_347,uop_350,])
F = relay.Function([var_335,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_335,], output2)
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
input_335= np.array(True, dtype='bool')
module1.set_input('var_335', input_335)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_335, )
res3 = intrp3.evaluate()(input_335, )
res4 = intrp4.evaluate()(input_335, )
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
module5.set_input('var_335', input_335)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_335, )
res7 = intrp7.evaluate()(input_335, )
res8 = intrp8.evaluate()(input_335, )
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
module9.set_input('var_335', input_335)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_335, )
res11 = intrp11.evaluate()(input_335, )
res12 = intrp12.evaluate()(input_335, )
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
module13.set_input('var_335', input_335)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_335, )
res15 = intrp15.evaluate()(input_335, )
res16 = intrp16.evaluate()(input_335, )
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
module17.set_input('var_335', input_335)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_335, )
res19 = intrp19.evaluate()(input_335, )
res20 = intrp20.evaluate()(input_335, )
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
module21.set_input('var_335', input_335)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_335, )
res23 = intrp23.evaluate()(input_335, )
res24 = intrp24.evaluate()(input_335, )
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

'''75: TVMFuncCall
74: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
73: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
72: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
71: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
70: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
69: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
68: tvm::transform::Pass::operator()(tvm::IRModule) const
67: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
66: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
65: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
64: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
63: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
62: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
61: tvm::transform::Pass::operator()(tvm::IRModule) const
60: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
59: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
58: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
57: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
56: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
55: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
54: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
53: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
52: _ZN3tvm5relay9transform22Devic
51: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
50: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
49: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
48: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
47: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
46: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
45: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
44: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
43: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
42: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
41: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
40: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
39: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
38: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
37: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
36: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
35: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
34: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
33: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::te::Tensor, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
32: tvm::LowerSchedule(tvm::te::Schedule, tvm::runtime::Array<tvm::runtime::ObjectRef, void> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer, std::hash<tvm::te::Tensor>, std::equal_to<tvm::te::Tensor>, std::allocator<std::pair<tvm::te::Tensor const, tvm::tir::Buffer> > > const&, bool)
31: tvm::LowerWithPassList(tvm::IRModule, tvm::runtime::Array<tvm::transform::Pass, void>)
30: tvm::transform::Pass::operator()(tvm::IRModule) const
29: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
28: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
27: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
26: tvm::tir::transform::PrimFuncPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
25: _ZNSt17_Function_handlerIFvN3tvm7
24: tvm::runtime::TypedPackedFunc<tvm::tir::PrimFunc (tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::tir::transform::NarrowDataType(int)::{lambda(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::tir::transform::NarrowDataType(int)::{lambda(tvm::tir::PrimFunc, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
23: tvm::tir::DataTypeRewriter::operator()(tvm::tir::Stmt)
22: _ZZN3tvm3tir11StmtFunctorIFNS0_4StmtERKS
21: tvm::tir::DataTypeRewriter::VisitStmt_(tvm::tir::StoreNode const*)
20: tvm::tir::StmtExprMutator::VisitExpr(tvm::PrimExpr const&)
19: _ZZN3tvm3tir11ExprFunctorIFNS_8PrimExprE
18: _ZThn16_N3tvm3tir16DataTyp
17: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::CallNode const*)
16: tvm::tir::ExprMutator::VisitExpr_(tvm::tir::CallNode const*)
15: non-virtual thunk to tvm::tir::StmtExprMutator::VisitExpr(tvm::PrimExpr const&)
14: _ZZN3tvm3tir11ExprFunctorIFNS_8PrimExprERKS
13: _ZThn16_N3tvm3tir16DataTyp
12: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::CastNode const*)
11: tvm::tir::ExprMutator::VisitExpr_(tvm::tir::CastNode const*)
10: non-virtual thunk to tvm::tir::StmtExprMutator::VisitExpr(tvm::PrimExpr const&)
9: _ZZN3tvm3tir11ExprFunctorIFNS_8PrimExprE
8: _ZThn16_N3tvm3tir16DataTyp
7: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::CallNode const*)
6: tvm::tir::ExprMutator::VisitExpr_(tvm::tir::CallNode const*)
5: non-virtual thunk to tvm::tir::StmtExprMutator::VisitExpr(tvm::PrimExpr const&)
4: _ZZN3tvm3tir11ExprFunctorIFNS_8PrimExprE
3: _ZThn16_N3tvm3tir16DataTyp
2: tvm::tir::DataTypeRewriter::VisitExpr_(tvm::tir::DivNode const*)
1: tvm::div(tvm::PrimExpr, tvm::PrimExpr, tvm::Span)
0: tvm::PrimExpr tvm::arith::TryConstFold<tvm::tir::Div>(tvm::PrimExpr, tvm::PrimExpr)

'''