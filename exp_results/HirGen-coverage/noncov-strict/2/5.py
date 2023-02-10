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
var_0 = relay.var("var_0", dtype = "float32", shape = (15,))#candidate|0|(15,)|var|float32
uop_1 = relay.log2(var_0.astype('float32')) # shape=(15,)
uop_3 = relay.log10(var_0.astype('float32')) # shape=(15,)
var_5 = relay.var("var_5", dtype = "float32", shape = (15,))#candidate|5|(15,)|var|float32
bop_6 = relay.not_equal(uop_3.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(uop_3))) # shape=(15,)
bop_9 = relay.mod(uop_3.astype('float32'), relay.reshape(bop_6.astype('float32'), relay.shape_of(uop_3))) # shape=(15,)
const_12 = relay.const([3.622926,5.891281,-4.131011,-0.767875,-9.057228,7.503339,-9.370011,4.508640,9.478085,-1.017868,7.812288,-3.205818,3.267250,-3.260917,8.142259], dtype = "float32")#candidate|12|(15,)|const|float32
bop_13 = relay.power(bop_9.astype('float64'), relay.reshape(const_12.astype('float64'), relay.shape_of(bop_9))) # shape=(15,)
bop_16 = relay.greater_equal(bop_13.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(bop_13))) # shape=(15,)
bop_19 = relay.less(var_5.astype('bool'), relay.reshape(uop_3.astype('bool'), relay.shape_of(var_5))) # shape=(15,)
uop_22 = relay.sin(var_5.astype('float64')) # shape=(15,)
bop_24 = relay.less(bop_16.astype('bool'), relay.reshape(bop_19.astype('bool'), relay.shape_of(bop_16))) # shape=(15,)
bop_27 = relay.not_equal(uop_1.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(uop_1))) # shape=(15,)
var_30 = relay.var("var_30", dtype = "bool", shape = (15,))#candidate|30|(15,)|var|bool
bop_31 = relay.right_shift(bop_27.astype('uint16'), relay.reshape(var_30.astype('uint16'), relay.shape_of(bop_27))) # shape=(15,)
bop_34 = relay.bitwise_xor(var_30.astype('int16'), relay.reshape(uop_22.astype('int16'), relay.shape_of(var_30))) # shape=(15,)
bop_37 = relay.less_equal(uop_1.astype('bool'), relay.reshape(bop_31.astype('bool'), relay.shape_of(uop_1))) # shape=(15,)
const_40 = relay.const([-5.210307,1.588270,3.211488,4.792380,-2.607320,-5.345927,-5.466983,-7.316165,1.201430,2.180389,-8.274731,-0.592180,9.118979,-5.667256,4.676166], dtype = "float32")#candidate|40|(15,)|const|float32
bop_41 = relay.right_shift(bop_9.astype('int32'), relay.reshape(const_40.astype('int32'), relay.shape_of(bop_9))) # shape=(15,)
uop_44 = relay.asinh(bop_24.astype('float32')) # shape=(15,)
bop_46 = relay.right_shift(uop_44.astype('uint64'), relay.reshape(const_12.astype('uint64'), relay.shape_of(uop_44))) # shape=(15,)
uop_49 = relay.acos(var_0.astype('float32')) # shape=(15,)
uop_51 = relay.sqrt(uop_44.astype('float32')) # shape=(15,)
const_53 = relay.const([-2.225667,8.975230,0.579001,-1.961202,-0.651203,-4.720014,7.199444,-1.396875,0.506645,-0.303379,3.664926,-4.466552,7.407368,5.329063,-0.520250], dtype = "float32")#candidate|53|(15,)|const|float32
bop_54 = relay.right_shift(uop_44.astype('int64'), relay.reshape(const_53.astype('int64'), relay.shape_of(uop_44))) # shape=(15,)
bop_57 = relay.add(const_53.astype('int64'), relay.reshape(bop_27.astype('int64'), relay.shape_of(const_53))) # shape=(15,)
bop_60 = relay.subtract(bop_46.astype('int64'), relay.reshape(bop_24.astype('int64'), relay.shape_of(bop_46))) # shape=(15,)
bop_63 = relay.greater_equal(bop_46.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(bop_46))) # shape=(15,)
bop_66 = relay.floor_mod(bop_60.astype('float64'), relay.reshape(bop_27.astype('float64'), relay.shape_of(bop_60))) # shape=(15,)
bop_69 = relay.bitwise_and(bop_54.astype('uint32'), relay.reshape(bop_41.astype('uint32'), relay.shape_of(bop_54))) # shape=(15,)
const_72 = relay.const([6.217116,-6.233647,4.626307,7.705879,2.828869,1.787784,6.922318,-0.359524,1.928526,-0.474343,-4.627881,8.537638,9.890224,7.824460,3.216137], dtype = "float32")#candidate|72|(15,)|const|float32
bop_73 = relay.multiply(uop_51.astype('int64'), relay.reshape(const_72.astype('int64'), relay.shape_of(uop_51))) # shape=(15,)
uop_76 = relay.sigmoid(bop_60.astype('float64')) # shape=(15,)
var_78 = relay.var("var_78", dtype = "float32", shape = (15,))#candidate|78|(15,)|var|float32
bop_79 = relay.multiply(uop_51.astype('uint8'), relay.reshape(var_78.astype('uint8'), relay.shape_of(uop_51))) # shape=(15,)
bop_82 = relay.multiply(uop_51.astype('float64'), relay.reshape(bop_34.astype('float64'), relay.shape_of(uop_51))) # shape=(15,)
uop_85 = relay.sin(const_72.astype('float32')) # shape=(15,)
uop_87 = relay.acosh(bop_82.astype('float64')) # shape=(15,)
uop_89 = relay.sinh(uop_87.astype('float32')) # shape=(15,)
bop_91 = relay.not_equal(uop_87.astype('bool'), relay.reshape(bop_34.astype('bool'), relay.shape_of(uop_87))) # shape=(15,)
uop_94 = relay.log(uop_89.astype('float64')) # shape=(15,)
output = relay.Tuple([bop_37,uop_49,bop_57,bop_63,bop_66,bop_69,bop_73,uop_76,bop_79,uop_85,bop_91,uop_94,])
output2 = relay.Tuple([bop_37,uop_49,bop_57,bop_63,bop_66,bop_69,bop_73,uop_76,bop_79,uop_85,bop_91,uop_94,])
func_96 = relay.Function([var_0,var_5,var_30,var_78,], output)
mod['func_96'] = func_96
mod = relay.transform.InferType()(mod)
mutated_mod['func_96'] = func_96
mutated_mod = relay.transform.InferType()(mutated_mod)
func_96_call = mutated_mod.get_global_var('func_96')
var_98 = relay.var("var_98", dtype = "float32", shape = (15,))#candidate|98|(15,)|var|float32
var_99 = relay.var("var_99", dtype = "float32", shape = (15,))#candidate|99|(15,)|var|float32
var_100 = relay.var("var_100", dtype = "bool", shape = (15,))#candidate|100|(15,)|var|bool
var_101 = relay.var("var_101", dtype = "float32", shape = (15,))#candidate|101|(15,)|var|float32
call_97 = func_96_call(var_98,var_99,var_100,var_101,)
output = call_97
func_102 = relay.Function([var_98,var_99,var_100,var_101,], output)
mutated_mod['func_102'] = func_102
mutated_mod = relay.transform.InferType()(mutated_mod)
var_104 = relay.var("var_104", dtype = "int32", shape = ())#candidate|104|()|var|int32
var_105 = relay.var("var_105", dtype = "int32", shape = ())#candidate|105|()|var|int32
bop_106 = relay.bitwise_or(var_104.astype('int32'), var_105.astype('int32')) # shape=()
bop_109 = relay.floor_divide(var_105.astype('float32'), var_104.astype('float32')) # shape=()
bop_112 = relay.equal(var_105.astype('bool'), bop_106.astype('bool')) # shape=()
func_96_call = mod.get_global_var('func_96')
func_102_call = mutated_mod.get_global_var('func_102')
const_116 = relay.const([9.908179,9.554868,9.920120,7.023339,9.738182,8.457269,3.467540,1.999191,5.611191,4.637445,1.546423,9.325064,4.149229,5.338091,-8.665171], dtype = "float32")#candidate|116|(15,)|const|float32
call_115 = relay.TupleGetItem(func_96_call(relay.reshape(const_116.astype('float32'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), relay.reshape(const_116.astype('bool'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), ), 4)
call_117 = relay.TupleGetItem(func_102_call(relay.reshape(const_116.astype('float32'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), relay.reshape(const_116.astype('bool'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), ), 4)
bop_118 = relay.less(bop_106.astype('bool'), var_105.astype('bool')) # shape=()
bop_121 = relay.logical_xor(bop_106.astype('uint64'), bop_109.astype('uint64')) # shape=()
bop_124 = relay.bitwise_or(call_115.astype('uint32'), bop_106.astype('uint32')) # shape=(15,)
bop_127 = relay.bitwise_or(call_117.astype('uint32'), bop_106.astype('uint32')) # shape=(15,)
bop_128 = relay.add(var_104.astype('int8'), var_105.astype('int8')) # shape=()
uop_131 = relay.atan(bop_118.astype('float64')) # shape=()
uop_133 = relay.atanh(uop_131.astype('float32')) # shape=()
bop_135 = relay.maximum(uop_133.astype('uint64'), bop_118.astype('uint64')) # shape=()
uop_138 = relay.erf(bop_121.astype('float64')) # shape=()
uop_140 = relay.rsqrt(bop_124.astype('float64')) # shape=(15,)
uop_142 = relay.rsqrt(bop_127.astype('float64')) # shape=(15,)
uop_143 = relay.asinh(uop_138.astype('float32')) # shape=()
var_145 = relay.var("var_145", dtype = "float32", shape = (5, 3))#candidate|145|(5, 3)|var|float32
bop_146 = relay.less(uop_133.astype('bool'), var_145.astype('bool')) # shape=(5, 3)
func_96_call = mod.get_global_var('func_96')
func_102_call = mutated_mod.get_global_var('func_102')
call_149 = relay.TupleGetItem(func_96_call(relay.reshape(bop_146.astype('float32'), [15,]), relay.reshape(uop_140.astype('float32'), [15,]), relay.reshape(uop_140.astype('bool'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), ), 10)
call_150 = relay.TupleGetItem(func_102_call(relay.reshape(bop_146.astype('float32'), [15,]), relay.reshape(uop_140.astype('float32'), [15,]), relay.reshape(uop_140.astype('bool'), [15,]), relay.reshape(const_116.astype('float32'), [15,]), ), 10)
bop_151 = relay.floor_divide(bop_135.astype('float32'), uop_133.astype('float32')) # shape=()
var_154 = relay.var("var_154", dtype = "float32", shape = ())#candidate|154|()|var|float32
bop_155 = relay.mod(uop_143.astype('float64'), var_154.astype('float64')) # shape=()
bop_158 = relay.floor_divide(bop_135.astype('float64'), bop_118.astype('float64')) # shape=()
bop_161 = relay.bitwise_xor(bop_135.astype('uint16'), call_149.astype('uint16')) # shape=(15,)
bop_164 = relay.bitwise_xor(bop_135.astype('uint16'), call_150.astype('uint16')) # shape=(15,)
uop_165 = relay.asin(uop_131.astype('float64')) # shape=()
bop_167 = relay.logical_and(bop_161.astype('bool'), uop_138.astype('bool')) # shape=(15,)
bop_170 = relay.logical_and(bop_164.astype('bool'), uop_138.astype('bool')) # shape=(15,)
uop_171 = relay.log2(uop_140.astype('float64')) # shape=(15,)
uop_173 = relay.log2(uop_142.astype('float64')) # shape=(15,)
bop_174 = relay.add(var_145.astype('uint8'), uop_143.astype('uint8')) # shape=(5, 3)
bop_177 = relay.right_shift(bop_124.astype('uint8'), bop_158.astype('uint8')) # shape=(15,)
bop_180 = relay.right_shift(bop_127.astype('uint8'), bop_158.astype('uint8')) # shape=(15,)
bop_181 = relay.floor_divide(bop_146.astype('float64'), relay.reshape(var_145.astype('float64'), relay.shape_of(bop_146))) # shape=(5, 3)
uop_184 = relay.cosh(bop_174.astype('float32')) # shape=(5, 3)
uop_186 = relay.sqrt(uop_184.astype('float64')) # shape=(5, 3)
uop_188 = relay.log2(uop_186.astype('float32')) # shape=(5, 3)
var_190 = relay.var("var_190", dtype = "float64", shape = (5, 3))#candidate|190|(5, 3)|var|float64
bop_191 = relay.maximum(uop_186.astype('float32'), relay.reshape(var_190.astype('float32'), relay.shape_of(uop_186))) # shape=(5, 3)
const_194 = relay.const([[-1.149145,-0.446934,1.142033],[2.800430,8.410043,-3.655528],[-2.483686,4.439264,2.575476],[6.112233,-1.179917,-4.805109],[2.768855,9.346431,-9.299156]], dtype = "float32")#candidate|194|(5, 3)|const|float32
bop_195 = relay.logical_and(uop_188.astype('bool'), relay.reshape(const_194.astype('bool'), relay.shape_of(uop_188))) # shape=(5, 3)
var_198 = relay.var("var_198", dtype = "float64", shape = (14, 14))#candidate|198|(14, 14)|var|float64
bop_199 = relay.bitwise_and(uop_138.astype('int16'), var_198.astype('int16')) # shape=(14, 14)
uop_202 = relay.exp(uop_186.astype('float32')) # shape=(5, 3)
const_204 = relay.const([[-0.685777,-1.983170],[-8.685923,8.317356],[-1.648364,3.818179],[-3.124437,-2.243335],[-9.049050,-3.629655],[-2.317718,-3.270672],[-5.443238,4.566950],[2.041336,8.889960],[-9.459236,7.459598],[-2.266613,2.423830],[6.246728,3.711131],[5.534981,4.048677]], dtype = "float32")#candidate|204|(12, 2)|const|float32
bop_205 = relay.power(uop_133.astype('float32'), const_204.astype('float32')) # shape=(12, 2)
uop_208 = relay.acos(bop_195.astype('float32')) # shape=(5, 3)
uop_210 = relay.atanh(uop_208.astype('float64')) # shape=(5, 3)
uop_212 = relay.atanh(uop_210.astype('float32')) # shape=(5, 3)
bop_214 = relay.bitwise_xor(uop_208.astype('uint64'), relay.reshape(bop_167.astype('uint64'), relay.shape_of(uop_208))) # shape=(5, 3)
bop_217 = relay.bitwise_xor(uop_208.astype('uint64'), relay.reshape(bop_170.astype('uint64'), relay.shape_of(uop_208))) # shape=(5, 3)
output = relay.Tuple([bop_112,const_116,bop_128,bop_151,bop_155,uop_165,uop_171,bop_177,bop_181,bop_191,bop_199,uop_202,bop_205,uop_212,bop_214,])
output2 = relay.Tuple([bop_112,const_116,bop_128,bop_151,bop_155,uop_165,uop_173,bop_180,bop_181,bop_191,bop_199,uop_202,bop_205,uop_212,bop_217,])
func_218 = relay.Function([var_104,var_105,var_145,var_154,var_190,var_198,], output)
mod['func_218'] = func_218
mod = relay.transform.InferType()(mod)
var_219 = relay.var("var_219", dtype = "int32", shape = ())#candidate|219|()|var|int32
var_220 = relay.var("var_220", dtype = "int32", shape = ())#candidate|220|()|var|int32
var_221 = relay.var("var_221", dtype = "float32", shape = (5, 3))#candidate|221|(5, 3)|var|float32
var_222 = relay.var("var_222", dtype = "float32", shape = ())#candidate|222|()|var|float32
var_223 = relay.var("var_223", dtype = "float64", shape = (5, 3))#candidate|223|(5, 3)|var|float64
var_224 = relay.var("var_224", dtype = "float64", shape = (14, 14))#candidate|224|(14, 14)|var|float64
output = func_218(var_219,var_220,var_221,var_222,var_223,var_224,)
func_225 = relay.Function([var_219,var_220,var_221,var_222,var_223,var_224,], output)
mutated_mod['func_225'] = func_225
mutated_mod = relay.transform.InferType()(mutated_mod)
var_227 = relay.var("var_227", dtype = "int32", shape = (2, 14, 10))#candidate|227|(2, 14, 10)|var|int32
var_228 = relay.var("var_228", dtype = "int32", shape = (2, 14, 10))#candidate|228|(2, 14, 10)|var|int32
bop_229 = relay.minimum(var_227.astype('int32'), relay.reshape(var_228.astype('int32'), relay.shape_of(var_227))) # shape=(2, 14, 10)
bop_232 = relay.subtract(var_228.astype('int32'), relay.reshape(var_227.astype('int32'), relay.shape_of(var_228))) # shape=(2, 14, 10)
var_235 = relay.var("var_235", dtype = "int32", shape = (2, 14, 10))#candidate|235|(2, 14, 10)|var|int32
bop_236 = relay.add(bop_232.astype('int16'), relay.reshape(var_235.astype('int16'), relay.shape_of(bop_232))) # shape=(2, 14, 10)
var_239 = relay.var("var_239", dtype = "int16", shape = (2, 14, 10))#candidate|239|(2, 14, 10)|var|int16
bop_240 = relay.left_shift(bop_236.astype('int32'), relay.reshape(var_239.astype('int32'), relay.shape_of(bop_236))) # shape=(2, 14, 10)
bop_243 = relay.right_shift(var_227.astype('uint8'), relay.reshape(var_235.astype('uint8'), relay.shape_of(var_227))) # shape=(2, 14, 10)
output = relay.Tuple([bop_229,bop_240,bop_243,])
output2 = relay.Tuple([bop_229,bop_240,bop_243,])
func_246 = relay.Function([var_227,var_228,var_235,var_239,], output)
mod['func_246'] = func_246
mod = relay.transform.InferType()(mod)
var_247 = relay.var("var_247", dtype = "int32", shape = (2, 14, 10))#candidate|247|(2, 14, 10)|var|int32
var_248 = relay.var("var_248", dtype = "int32", shape = (2, 14, 10))#candidate|248|(2, 14, 10)|var|int32
var_249 = relay.var("var_249", dtype = "int32", shape = (2, 14, 10))#candidate|249|(2, 14, 10)|var|int32
var_250 = relay.var("var_250", dtype = "int16", shape = (2, 14, 10))#candidate|250|(2, 14, 10)|var|int16
output = func_246(var_247,var_248,var_249,var_250,)
func_251 = relay.Function([var_247,var_248,var_249,var_250,], output)
mutated_mod['func_251'] = func_251
mutated_mod = relay.transform.InferType()(mutated_mod)
var_253 = relay.var("var_253", dtype = "float32", shape = ())#candidate|253|()|var|float32
uop_254 = relay.log2(var_253.astype('float32')) # shape=()
bop_256 = relay.mod(uop_254.astype('float64'), var_253.astype('float64')) # shape=()
var_259 = relay.var("var_259", dtype = "float32", shape = ())#candidate|259|()|var|float32
bop_260 = relay.greater(uop_254.astype('bool'), var_259.astype('bool')) # shape=()
uop_263 = relay.log2(bop_260.astype('float32')) # shape=()
uop_265 = relay.sin(uop_263.astype('float64')) # shape=()
const_267 = relay.const([[[6.383031,9.188867,-0.594752,5.519801,-1.171677,5.490239,8.950474,-4.901859,-7.973021,6.153000,-4.501007],[0.477514,2.246807,-4.843679,3.388897,-4.988113,3.175478,5.466377,-1.713628,-3.178053,-4.520000,-4.760614],[-0.334392,0.896257,3.689207,3.795676,1.798735,0.694716,-8.702768,9.893042,5.207215,-1.521416,7.161481],[-0.740251,3.373312,7.667896,3.214042,9.574870,-4.814126,-5.144618,-8.682606,0.445977,-6.529149,-6.845495],[-8.108898,-9.386742,6.944380,-0.364171,-9.923078,-9.233925,-7.741806,-9.059640,1.121227,1.583071,-1.175752],[6.814430,2.438807,-5.545281,0.719113,-8.958900,5.462933,-8.044283,-7.624579,6.536913,6.139177,-9.031416],[4.472644,7.851850,-6.499612,-5.261088,-6.264607,8.719468,0.131207,3.010054,-4.701356,-7.830638,2.482261],[-8.334873,-5.603377,-6.316690,4.039685,1.971796,0.442039,-3.366599,7.914621,9.329676,-0.947899,4.321765],[-8.074683,8.243428,-4.174771,-2.080268,-9.699002,1.798883,1.646414,-2.480334,-1.192028,7.367738,5.624538],[-9.529179,8.856788,-1.293443,-5.462584,2.083721,0.428471,-2.838678,-8.554366,2.532140,5.382382,1.594335],[-2.156667,-5.946789,1.766779,8.087641,2.050564,7.380991,7.512026,-3.500982,5.591642,-8.108515,-3.152862],[3.928899,-6.778656,5.505945,6.720710,-1.058384,6.227174,4.013931,-2.517499,3.980542,-8.080349,-3.776868],[-3.933873,4.628665,-2.780893,-1.480994,-2.519983,-4.160149,-9.888131,2.770987,-0.487521,-5.326199,-7.344667],[5.135618,-5.036557,-7.431876,4.755018,-2.849032,8.279607,-5.270170,5.707016,8.114245,-0.544264,-1.546508]],[[3.234021,-6.305578,1.685413,5.919669,-0.780405,8.050077,6.762082,9.527612,-8.385736,3.335998,-2.368179],[-4.058573,-0.820639,9.831788,5.411467,-4.621357,9.245572,-2.994585,0.177255,-2.670301,-9.721382,-1.760372],[7.864631,7.755148,2.521397,-7.114445,-1.419080,2.390066,-6.309058,5.442934,5.123305,-8.251345,-7.780195],[0.765377,1.461356,-2.568859,0.758752,2.482913,-8.312481,5.860422,9.670801,1.203003,-3.196523,1.309771],[-1.251579,-7.038566,-3.732507,-5.427203,2.088435,-3.258308,-4.117573,1.449694,2.190220,-8.984850,9.419203],[7.144281,5.544446,-1.282356,8.313471,-9.745090,-0.773923,0.441842,-4.502711,4.564117,-1.085709,-4.571107],[-0.378290,-8.937850,-9.739455,-1.344692,-9.488010,4.567855,-1.799663,1.445879,0.317909,2.587842,-0.683923],[-7.094829,2.770903,3.213120,9.347034,-8.374679,8.923286,9.072030,2.532098,8.244908,1.287933,-0.201829],[-2.196377,9.242253,3.594179,4.075925,-8.167734,-6.724297,9.383962,9.897666,0.553189,6.981965,-9.461933],[9.684669,-0.731400,-8.424431,1.551750,2.862339,-4.544679,6.789352,-4.817243,-6.597371,-7.115449,-8.658280],[-4.486145,-8.656803,9.124822,-8.828422,-7.833674,2.512242,-7.961948,-8.173717,-0.993650,4.770182,-7.760914],[0.660183,6.624117,4.463348,-8.951530,4.905158,2.654379,9.525841,6.973977,1.881045,2.015158,-5.836669],[-6.961921,6.185675,0.283231,0.597185,8.904251,-7.837458,-1.145665,-2.078091,3.799152,8.347686,7.848700],[7.471776,-3.473196,-2.260484,9.160557,8.597443,6.198393,4.270696,1.969808,-5.575528,9.621117,8.447251]],[[7.941571,-8.396506,-0.113837,9.109834,-9.952905,-4.909550,5.959387,-2.966545,-5.280184,-3.587272,-0.383092],[-4.372045,9.090060,-0.223945,4.354093,-0.259715,1.373266,7.639500,9.244249,5.678765,7.801086,-4.211380],[-9.591094,1.272604,5.500722,-7.285811,4.932013,6.303203,-9.326764,4.632194,-4.530496,-8.915305,7.544566],[7.068820,-9.130655,-8.713674,-8.841750,6.329261,-6.901582,2.996576,-0.802293,3.877414,-2.865385,-5.766103],[4.357565,-7.293034,-6.906438,1.668512,-1.230031,-6.282224,-2.883961,-7.518396,9.996674,7.037080,7.786560],[7.597413,8.685321,6.375676,-8.392942,-4.489192,-6.333901,4.954525,2.935439,-5.387718,6.692136,-4.459918],[5.136239,-8.681833,-0.351117,-8.658849,6.982610,-0.950469,8.959471,-4.457263,1.652161,0.609338,-0.172718],[-9.681476,9.159221,5.150464,8.638769,-6.792760,2.635100,-8.418875,6.332751,3.502371,-5.833287,2.605161],[3.494432,0.239516,-5.559083,-6.809489,3.975774,-2.744031,8.047687,8.421653,1.358181,-5.239112,0.009346],[6.395653,9.665470,8.453760,-8.604096,8.730341,-0.094591,-6.535976,2.196510,9.591534,-4.716483,-3.944748],[-0.746388,-1.863488,0.285484,-9.534523,7.823189,4.545163,-2.995955,4.573053,3.325552,-9.568185,9.630346],[-4.232357,-6.778530,-6.083628,8.956378,-5.121563,-9.524781,-7.197612,6.297972,-6.936108,-3.430427,-2.852933],[-8.310384,5.036493,-1.794126,-0.232507,-9.822379,5.225952,-7.774517,0.973655,-2.906693,2.947126,3.567780],[-8.301025,-2.541477,4.342821,4.874622,4.564283,6.988913,0.719415,2.165619,-6.840420,4.217882,-8.529704]],[[6.656539,3.239418,6.652255,-3.852303,-7.791473,-4.992879,-8.846309,5.958455,8.509734,6.878892,0.358687],[-3.899943,-9.921202,8.969689,-6.073171,-5.601330,1.358051,6.157044,-6.819999,5.775637,3.227672,-3.817457],[-7.704632,6.185780,-5.679008,-4.825773,4.344939,-3.104840,0.069153,0.962229,9.861980,7.199154,-6.789886],[-6.537293,1.832935,5.950897,6.108725,-9.633845,-8.677141,2.235935,-9.950046,-7.177388,-8.583923,-5.623409],[-4.332121,-9.520786,6.021800,6.732831,-8.807968,-0.170260,7.861854,-1.278053,5.205300,-5.642834,-4.158732],[9.239954,-8.055231,5.650396,-4.413737,-2.381388,6.944074,3.131717,9.002207,-1.699140,-0.407600,5.521146],[0.036810,-2.075556,-8.118797,-9.206554,-7.778894,8.727710,1.303191,4.461735,-4.994934,7.394420,-9.141888],[-5.281991,4.512631,-1.945037,-7.533308,6.599278,9.899561,-2.133295,9.021870,4.885192,-2.691212,-3.480184],[8.682340,-7.532458,8.193778,7.908322,-1.599101,3.808776,2.515011,4.546729,9.938762,5.549799,-7.541559],[-5.882836,-5.029023,2.339961,3.644161,-4.618615,6.804303,-4.526516,-2.700717,6.697748,-9.168860,-6.844069],[-0.810766,6.065365,9.731595,-5.634097,-0.847267,1.907145,-9.560447,-1.846772,-9.649602,-7.812828,1.293719],[-5.953132,4.288954,-7.252139,6.409651,3.347099,-5.169322,1.752511,5.733317,9.475117,1.933971,-7.852864],[3.632109,2.641978,6.316592,2.197329,-4.050212,2.125194,9.663943,4.478543,-2.938865,-2.025118,5.566055],[-0.242088,-7.800245,2.085108,-9.416233,1.414300,-7.825723,6.453002,6.575677,-7.277180,2.325345,9.618626]],[[7.390892,8.036894,5.981430,-0.349349,2.763482,5.991492,-4.879605,2.596784,-9.948624,-1.256901,4.278506],[-9.160036,-6.131457,1.744398,-0.877449,0.670693,4.265118,-7.487008,-9.219262,-6.993073,7.435649,0.470919],[-1.372714,6.994760,-8.568215,9.301523,8.208600,-8.255869,-2.343655,3.295917,-3.859731,-1.717296,1.677544],[-0.657032,1.705813,-5.004594,-9.960894,-6.362079,-2.280895,5.418295,7.790710,0.335825,-6.095357,-3.623371],[7.160618,-2.758042,-1.941637,8.376573,7.393134,-2.312280,-0.852530,7.340804,-8.084883,5.301391,8.421535],[5.795011,2.459324,-1.602678,-2.670990,-2.532304,-6.572324,6.320861,3.208530,1.780678,-2.036329,7.504759],[-8.747565,9.175524,-9.597932,1.813209,7.329982,-5.030563,9.058868,0.809357,-8.325842,4.654577,-9.657048],[8.883593,-2.252132,-0.845403,-1.074090,1.796579,-0.258595,-7.139520,-4.296563,3.028582,-9.552670,-1.528959],[-5.358294,-9.752107,-6.181639,4.867700,-1.908124,-7.355111,9.402315,-5.913195,3.880432,-9.160659,-0.104407],[-6.657166,9.594886,8.297423,-3.584598,2.698926,4.313953,8.147859,-9.693529,-8.508273,3.029900,2.492056],[7.355649,0.507309,8.220319,-6.805303,-7.763519,-6.485191,-3.332540,4.338237,-6.020048,8.582170,-1.344715],[6.792651,-7.985213,-7.911915,-0.544706,-3.250502,2.614029,-6.956532,-8.060330,-4.589226,-6.614790,-7.747250],[-9.395755,7.916956,-9.319849,-1.375045,6.646421,0.210788,3.505769,8.694985,2.721921,1.484460,5.536854],[-1.563422,0.712440,-4.840587,4.104419,-2.898563,1.656307,-8.103918,2.268564,-3.300725,-0.520959,-5.189123]]], dtype = "float64")#candidate|267|(5, 14, 11)|const|float64
bop_268 = relay.logical_and(uop_265.astype('bool'), const_267.astype('bool')) # shape=(5, 14, 11)
bop_271 = relay.greater(uop_265.astype('bool'), bop_260.astype('bool')) # shape=()
var_274 = relay.var("var_274", dtype = "float32", shape = (8, 1, 8))#candidate|274|(8, 1, 8)|var|float32
bop_275 = relay.bitwise_xor(uop_263.astype('int64'), var_274.astype('int64')) # shape=(8, 1, 8)
bop_278 = relay.power(bop_268.astype('float64'), var_253.astype('float64')) # shape=(5, 14, 11)
uop_281 = relay.atan(uop_254.astype('float64')) # shape=()
bop_283 = relay.minimum(bop_271.astype('int8'), const_267.astype('int8')) # shape=(5, 14, 11)
uop_286 = relay.log2(uop_263.astype('float64')) # shape=()
uop_288 = relay.sigmoid(uop_286.astype('float64')) # shape=()
uop_290 = relay.sigmoid(uop_288.astype('float32')) # shape=()
bop_292 = relay.bitwise_and(bop_283.astype('int16'), bop_271.astype('int16')) # shape=(5, 14, 11)
uop_295 = relay.log(uop_290.astype('float64')) # shape=()
uop_297 = relay.cos(uop_295.astype('float64')) # shape=()
output = relay.Tuple([bop_256,bop_275,bop_278,uop_281,bop_292,uop_297,])
output2 = relay.Tuple([bop_256,bop_275,bop_278,uop_281,bop_292,uop_297,])
func_299 = relay.Function([var_253,var_259,var_274,], output)
mod['func_299'] = func_299
mod = relay.transform.InferType()(mod)
mutated_mod['func_299'] = func_299
mutated_mod = relay.transform.InferType()(mutated_mod)
func_299_call = mutated_mod.get_global_var('func_299')
var_301 = relay.var("var_301", dtype = "float32", shape = ())#candidate|301|()|var|float32
var_302 = relay.var("var_302", dtype = "float32", shape = ())#candidate|302|()|var|float32
var_303 = relay.var("var_303", dtype = "float32", shape = (8, 1, 8))#candidate|303|(8, 1, 8)|var|float32
call_300 = func_299_call(var_301,var_302,var_303,)
output = call_300
func_304 = relay.Function([var_301,var_302,var_303,], output)
mutated_mod['func_304'] = func_304
mutated_mod = relay.transform.InferType()(mutated_mod)
var_306 = relay.var("var_306", dtype = "int64", shape = (3, 1))#candidate|306|(3, 1)|var|int64
var_307 = relay.var("var_307", dtype = "int64", shape = (3, 4))#candidate|307|(3, 4)|var|int64
bop_308 = relay.add(var_306.astype('int64'), var_307.astype('int64')) # shape=(3, 4)
uop_311 = relay.acos(bop_308.astype('float64')) # shape=(3, 4)
bop_313 = relay.mod(uop_311.astype('float64'), relay.reshape(var_307.astype('float64'), relay.shape_of(uop_311))) # shape=(3, 4)
uop_316 = relay.exp(uop_311.astype('float64')) # shape=(3, 4)
uop_318 = relay.atanh(uop_311.astype('float32')) # shape=(3, 4)
output = relay.Tuple([bop_313,uop_316,uop_318,])
output2 = relay.Tuple([bop_313,uop_316,uop_318,])
func_320 = relay.Function([var_306,var_307,], output)
mod['func_320'] = func_320
mod = relay.transform.InferType()(mod)
var_321 = relay.var("var_321", dtype = "int64", shape = (3, 1))#candidate|321|(3, 1)|var|int64
var_322 = relay.var("var_322", dtype = "int64", shape = (3, 4))#candidate|322|(3, 4)|var|int64
output = func_320(var_321,var_322,)
func_323 = relay.Function([var_321,var_322,], output)
mutated_mod['func_323'] = func_323
mutated_mod = relay.transform.InferType()(mutated_mod)
const_325 = relay.const([[5.562999,-9.375353,7.322800,-5.801673,-9.346151,-5.310651,-9.304374,6.118693,-8.623595,8.781098,-8.774252,2.645917,1.518763,2.945083],[-4.566308,-9.981324,7.090451,2.896784,3.269737,5.019123,4.565941,7.164873,-3.654863,-9.918633,3.891109,6.069185,-6.282470,2.675445],[-4.261146,8.794345,-4.965523,-8.392582,-1.812603,-2.446589,-0.082617,7.523497,4.575632,1.582580,-1.681118,-1.133854,-1.888362,5.847213],[7.816508,-3.912543,6.352027,-1.812916,5.463906,-6.872126,-1.101303,-9.841839,-1.134637,6.682825,-2.087252,2.732992,-5.739608,-3.870605],[5.070120,4.046782,-9.928648,9.792580,7.983625,6.652350,-5.611221,-1.132440,-2.401036,6.642441,8.891598,1.220765,-3.080517,-4.786875],[-5.452106,2.241262,-9.872219,-2.808056,-4.485432,9.953574,-7.440013,7.912423,-3.311828,4.310882,-0.764480,9.365970,4.020281,3.510767],[5.974393,8.158215,-0.281569,7.508798,6.682500,3.585647,7.580576,-6.088117,-3.776162,0.166046,1.492543,0.303298,0.267802,4.324886],[8.669592,-2.687194,-0.666481,2.680036,-4.052951,-9.386998,-3.653446,-9.548740,-7.613384,5.145269,-8.361825,9.451416,-0.062473,1.590024],[9.906936,-8.818908,6.566048,8.189604,-4.015111,-3.844827,8.394413,3.800783,-0.774225,-0.911176,-5.975942,2.060700,-6.410829,-7.014670],[-2.563797,-3.726479,4.399706,-5.301006,-5.293276,-6.115932,-6.715142,6.957408,1.486905,-7.468788,-4.930101,-7.458075,-3.402497,9.427209],[-8.941357,8.530930,9.758435,-6.190110,3.938772,9.612045,-8.752683,-7.758966,-1.136146,5.026635,-5.773275,-8.571674,-6.796255,9.451036],[-9.082827,0.637011,0.710650,3.602530,-1.921170,9.304386,9.925156,4.775259,-9.327992,-8.149945,-7.295845,0.456017,7.152804,6.159647],[-9.473673,8.400620,-9.289030,-9.438993,0.883198,-3.608277,6.449298,3.907139,8.603376,0.945089,1.616443,-8.803023,2.021658,-6.612241],[0.183190,-9.359177,7.465005,2.010645,-4.574507,8.396421,0.357665,8.834576,9.974608,2.976522,-1.069163,-2.312785,7.432871,7.468167],[-9.996632,9.178373,3.397092,-5.342160,8.068652,-5.325377,5.945784,-1.309911,-3.532156,8.423639,-5.624790,-1.876189,4.464286,-8.707007],[1.393543,4.695784,-9.695359,-5.492123,-5.786679,-8.836548,-3.540025,9.469753,-9.420584,-1.421839,-3.421257,-4.527793,4.343781,6.126898]], dtype = "float32")#candidate|325|(16, 14)|const|float32
const_326 = relay.const([[1.104713,9.824448,1.962514,0.702112,7.766375,-1.794527,-9.702028,1.018266,-3.090093,7.750743,-7.901916,-9.700488,-8.849430,8.842984],[4.728556,-6.271154,7.608287,0.444730,5.040220,-2.840405,5.468638,-6.609269,7.943337,5.940832,5.926851,0.958851,-0.246118,-2.474983],[7.250433,-7.088843,-7.422433,2.888446,-9.709858,-3.872994,5.733558,2.785428,2.397288,-8.574772,7.808731,-3.793182,-2.734074,-6.707349],[-6.987866,-0.708945,-3.965585,-6.749988,7.718012,-9.526406,-6.749925,3.529154,-2.967849,-7.534356,-4.788305,-1.281472,8.109399,-9.034561],[-2.244130,-2.324942,6.871888,-2.876103,-8.860087,-4.821374,-2.816084,1.631210,-7.265846,1.058479,0.548459,-3.196983,-8.669260,6.448430],[-9.471791,-4.568327,-5.733895,6.655851,7.609114,1.937706,-2.559582,4.083638,2.112937,9.558924,1.129667,-4.715463,4.010055,-2.816600],[1.546143,9.501548,6.958200,0.280228,-1.811256,7.115567,-8.449853,5.178851,-4.149432,-4.918013,-9.388924,6.771676,-5.678224,-4.284622],[8.622552,7.022452,0.343871,-9.262937,-1.216990,-2.391128,-1.346401,-5.552083,9.636193,0.425878,-0.445487,0.605623,2.735714,-0.651193],[-3.994682,9.048310,2.831578,-4.501708,4.274316,7.358403,-3.486377,1.354673,9.614491,-1.812343,9.944379,0.890737,3.801481,-1.862399],[3.247508,-1.606022,-8.962207,9.395761,-7.332497,-1.173659,-3.417662,-0.646695,0.245192,0.490801,1.061114,0.088680,7.946226,0.437939],[9.909964,-1.230099,-5.698668,-4.030537,-7.457763,7.083641,-6.436518,9.827976,2.425024,3.138804,5.125461,-9.468708,-3.058545,8.113999],[-8.713704,-5.135982,1.909097,-8.846332,2.820582,-5.364571,9.421262,8.067912,7.927777,-3.526377,-8.126779,3.050713,8.019075,-3.313295],[-5.332002,3.916896,-3.535402,-3.769901,-7.508326,-7.518856,-7.682784,-0.640414,-6.281211,2.185387,-7.523298,-7.667037,6.522294,-1.471859],[7.717973,1.733404,-8.883654,-0.826866,-8.725427,-5.780669,-5.366183,7.199057,3.546502,-7.583469,5.709470,0.189719,8.825693,-1.713504],[-6.460335,7.091468,1.683262,7.490386,-2.649652,5.442908,-5.744258,4.450246,-7.045189,5.156884,-1.943346,6.306037,9.057861,9.109052],[-9.925931,-7.531105,-5.220251,7.043998,-7.308203,-2.436131,-1.530095,4.873834,9.667950,7.700955,3.964611,-2.969861,-8.870641,4.727360]], dtype = "float32")#candidate|326|(16, 14)|const|float32
bop_327 = relay.subtract(const_325.astype('float32'), relay.reshape(const_326.astype('float32'), relay.shape_of(const_325))) # shape=(16, 14)
bop_330 = relay.floor_divide(const_326.astype('float32'), relay.reshape(const_325.astype('float32'), relay.shape_of(const_326))) # shape=(16, 14)
uop_333 = relay.acosh(bop_330.astype('float64')) # shape=(16, 14)
bop_335 = relay.right_shift(bop_327.astype('uint32'), relay.reshape(bop_330.astype('uint32'), relay.shape_of(bop_327))) # shape=(16, 14)
output = relay.Tuple([uop_333,bop_335,])
output2 = relay.Tuple([uop_333,bop_335,])
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

'''9: TVMFuncCall
8: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
7: tvm::transform::Pass::operator()(tvm::IRModule) const
6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::LazyGradientInit()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::LazyGradientInit(tvm::RelayExpr const&, tvm::IRModule)
0: tvm::relay::CheckFeature(tvm::RelayExpr const&, tvm::relay::FeatureSet const&)

'''