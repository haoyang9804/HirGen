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
var_14 = relay.var("var_14", dtype = "float32", shape = (2, 8, 4))#candidate|14|(2, 8, 4)|var|float32
var_15 = relay.var("var_15", dtype = "float32", shape = (2, 8, 4))#candidate|15|(2, 8, 4)|var|float32
bop_16 = relay.divide(var_14.astype('float32'), relay.reshape(var_15.astype('float32'), relay.shape_of(var_14))) # shape=(2, 8, 4)
bop_19 = relay.logical_or(var_15.astype('bool'), relay.reshape(var_14.astype('bool'), relay.shape_of(var_15))) # shape=(2, 8, 4)
bop_29 = relay.multiply(bop_19.astype('uint8'), relay.reshape(bop_16.astype('uint8'), relay.shape_of(bop_19))) # shape=(2, 8, 4)
bop_32 = relay.not_equal(var_15.astype('bool'), relay.reshape(bop_29.astype('bool'), relay.shape_of(var_15))) # shape=(2, 8, 4)
bop_35 = relay.less(var_14.astype('bool'), relay.reshape(bop_29.astype('bool'), relay.shape_of(var_14))) # shape=(2, 8, 4)
bop_38 = relay.logical_or(bop_19.astype('bool'), relay.reshape(bop_29.astype('bool'), relay.shape_of(bop_19))) # shape=(2, 8, 4)
bop_47 = relay.not_equal(bop_32.astype('bool'), relay.reshape(bop_19.astype('bool'), relay.shape_of(bop_32))) # shape=(2, 8, 4)
bop_50 = relay.mod(bop_16.astype('float32'), relay.reshape(bop_38.astype('float32'), relay.shape_of(bop_16))) # shape=(2, 8, 4)
var_59 = relay.var("var_59", dtype = "float32", shape = (2, 8, 4))#candidate|59|(2, 8, 4)|var|float32
bop_60 = relay.logical_and(bop_50.astype('bool'), relay.reshape(var_59.astype('bool'), relay.shape_of(bop_50))) # shape=(2, 8, 4)
bop_63 = relay.minimum(bop_47.astype('float32'), relay.reshape(bop_35.astype('float32'), relay.shape_of(bop_47))) # shape=(2, 8, 4)
bop_66 = relay.maximum(bop_16.astype('uint64'), relay.reshape(bop_32.astype('uint64'), relay.shape_of(bop_16))) # shape=(2, 8, 4)
uop_69 = relay.asin(bop_19.astype('float64')) # shape=(2, 8, 4)
const_71 = relay.const([[[3.738501,2.698359,-5.024716,-1.610273],[9.748001,7.758495,-3.526964,-1.562920],[-3.290869,5.438290,-0.529742,-4.327133],[-4.490136,3.327142,-3.142015,5.284670],[-7.313375,-9.523585,-8.620797,-8.654881],[-4.520607,-6.038046,-4.592301,2.291502],[-8.125642,-0.685672,4.298451,-3.826980],[-6.364692,5.971322,-7.375590,-2.325260]],[[3.183640,0.125623,-0.174576,1.629142],[-7.545338,3.417550,8.986260,2.464900],[0.955062,-7.963200,7.052134,-4.487442],[-0.236052,-0.806764,-9.008522,5.642495],[1.217389,-0.285012,-1.998833,-3.340113],[7.309709,-1.045810,-2.818646,7.046902],[4.164597,-2.062454,-8.691742,9.912888],[-7.246953,-3.571163,-3.771822,-0.441129]]], dtype = "float64")#candidate|71|(2, 8, 4)|const|float64
bop_72 = relay.left_shift(uop_69.astype('int32'), relay.reshape(const_71.astype('int32'), relay.shape_of(uop_69))) # shape=(2, 8, 4)
var_75 = relay.var("var_75", dtype = "bool", shape = (2, 8, 4))#candidate|75|(2, 8, 4)|var|bool
bop_76 = relay.logical_xor(bop_32.astype('uint8'), relay.reshape(var_75.astype('uint8'), relay.shape_of(bop_32))) # shape=(2, 8, 4)
bop_79 = relay.greater_equal(bop_38.astype('bool'), relay.reshape(bop_60.astype('bool'), relay.shape_of(bop_38))) # shape=(2, 8, 4)
bop_82 = relay.equal(bop_72.astype('bool'), relay.reshape(uop_69.astype('bool'), relay.shape_of(bop_72))) # shape=(2, 8, 4)
uop_87 = relay.atanh(uop_69.astype('float32')) # shape=(2, 8, 4)
bop_91 = relay.bitwise_and(uop_87.astype('uint16'), relay.reshape(bop_76.astype('uint16'), relay.shape_of(uop_87))) # shape=(2, 8, 4)
uop_95 = relay.log(bop_82.astype('float32')) # shape=(2, 8, 4)
var_100 = relay.var("var_100", dtype = "bool", shape = (2, 8, 4))#candidate|100|(2, 8, 4)|var|bool
bop_101 = relay.power(bop_79.astype('float64'), relay.reshape(var_100.astype('float64'), relay.shape_of(bop_79))) # shape=(2, 8, 4)
bop_108 = relay.add(uop_87.astype('uint64'), relay.reshape(const_71.astype('uint64'), relay.shape_of(uop_87))) # shape=(2, 8, 4)
bop_112 = relay.bitwise_or(bop_91.astype('uint64'), relay.reshape(bop_47.astype('uint64'), relay.shape_of(bop_91))) # shape=(2, 8, 4)
output = relay.Tuple([bop_63,bop_66,uop_95,bop_101,bop_108,bop_112,])
output2 = relay.Tuple([bop_63,bop_66,uop_95,bop_101,bop_108,bop_112,])
func_118 = relay.Function([var_14,var_15,var_59,var_75,var_100,], output)
mod['func_118'] = func_118
mod = relay.transform.InferType()(mod)
mutated_mod['func_118'] = func_118
mutated_mod = relay.transform.InferType()(mutated_mod)
func_118_call = mutated_mod.get_global_var('func_118')
var_120 = relay.var("var_120", dtype = "float32", shape = (2, 8, 4))#candidate|120|(2, 8, 4)|var|float32
var_121 = relay.var("var_121", dtype = "float32", shape = (2, 8, 4))#candidate|121|(2, 8, 4)|var|float32
var_122 = relay.var("var_122", dtype = "float32", shape = (2, 8, 4))#candidate|122|(2, 8, 4)|var|float32
var_123 = relay.var("var_123", dtype = "bool", shape = (2, 8, 4))#candidate|123|(2, 8, 4)|var|bool
var_124 = relay.var("var_124", dtype = "bool", shape = (2, 8, 4))#candidate|124|(2, 8, 4)|var|bool
call_119 = func_118_call(var_120,var_121,var_122,var_123,var_124,)
output = call_119
func_125 = relay.Function([var_120,var_121,var_122,var_123,var_124,], output)
mutated_mod['func_125'] = func_125
mutated_mod = relay.transform.InferType()(mutated_mod)
const_139 = relay.const([1.590423,-7.009311,6.431273,-3.753953,-6.791344,-4.964531,7.055398,-2.901797,6.147274,5.432510,-8.153414,-9.299346,-4.747630,4.102433,8.382793,3.284033], dtype = "float32")#candidate|139|(16,)|const|float32
uop_140 = relay.atan(const_139.astype('float32')) # shape=(16,)
bop_142 = relay.not_equal(uop_140.astype('bool'), relay.reshape(const_139.astype('bool'), relay.shape_of(uop_140))) # shape=(16,)
uop_145 = relay.log10(bop_142.astype('float32')) # shape=(16,)
uop_147 = relay.log2(bop_142.astype('float32')) # shape=(16,)
func_118_call = mod.get_global_var('func_118')
func_125_call = mutated_mod.get_global_var('func_125')
const_152 = relay.const([4.858883,-0.256299,-8.893753,-2.732453,-6.806493,2.855357,9.155024,-2.227063,3.366849,1.607483,8.192847,-5.839124,-2.516823,6.620647,-1.973934,1.078226,3.232032,1.171181,-8.984887,-7.249459,2.836269,-1.816376,0.162725,0.801034,5.024596,1.393032,0.369452,7.716622,0.778722,0.662242,-6.642408,-2.184606,-2.310881,7.577824,9.115473,0.790369,8.442155,4.938177,8.408552,2.603718,4.703605,0.411225,-2.122417,-1.480454,-0.990731,6.366297,-0.006114,-4.632556,6.681430,3.699982,-6.727013,0.552664,3.530034,-1.390881,-1.824726,1.636718,-0.808568,-0.340079,-4.750787,9.143801,-5.029635,4.617181,-9.139778,0.463046], dtype = "float32")#candidate|152|(64,)|const|float32
call_151 = relay.TupleGetItem(func_118_call(relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), ), 2)
call_153 = relay.TupleGetItem(func_125_call(relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), ), 2)
var_158 = relay.var("var_158", dtype = "float32", shape = (16,))#candidate|158|(16,)|var|float32
bop_159 = relay.subtract(uop_147.astype('int8'), relay.reshape(var_158.astype('int8'), relay.shape_of(uop_147))) # shape=(16,)
bop_162 = relay.divide(bop_142.astype('float32'), relay.reshape(uop_140.astype('float32'), relay.shape_of(bop_142))) # shape=(16,)
uop_167 = relay.sqrt(uop_140.astype('float64')) # shape=(16,)
func_118_call = mod.get_global_var('func_118')
func_125_call = mutated_mod.get_global_var('func_125')
call_170 = relay.TupleGetItem(func_118_call(relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), ), 0)
call_171 = relay.TupleGetItem(func_125_call(relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(call_151.astype('float32'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), relay.reshape(const_152.astype('bool'), [2, 8, 4]), ), 0)
bop_173 = relay.mod(uop_167.astype('float32'), relay.reshape(uop_140.astype('float32'), relay.shape_of(uop_167))) # shape=(16,)
uop_178 = relay.exp(bop_142.astype('float64')) # shape=(16,)
bop_185 = relay.bitwise_or(uop_140.astype('uint32'), relay.reshape(uop_145.astype('uint32'), relay.shape_of(uop_140))) # shape=(16,)
var_190 = relay.var("var_190", dtype = "uint32", shape = (16,))#candidate|190|(16,)|var|uint32
bop_191 = relay.left_shift(bop_185.astype('uint16'), relay.reshape(var_190.astype('uint16'), relay.shape_of(bop_185))) # shape=(16,)
output = relay.Tuple([call_151,const_152,bop_159,bop_162,call_170,bop_173,uop_178,bop_191,])
output2 = relay.Tuple([call_153,const_152,bop_159,bop_162,call_171,bop_173,uop_178,bop_191,])
func_196 = relay.Function([var_158,var_190,], output)
mod['func_196'] = func_196
mod = relay.transform.InferType()(mod)
mutated_mod['func_196'] = func_196
mutated_mod = relay.transform.InferType()(mutated_mod)
func_196_call = mutated_mod.get_global_var('func_196')
var_198 = relay.var("var_198", dtype = "float32", shape = (16,))#candidate|198|(16,)|var|float32
var_199 = relay.var("var_199", dtype = "uint32", shape = (16,))#candidate|199|(16,)|var|uint32
call_197 = func_196_call(var_198,var_199,)
output = call_197
func_200 = relay.Function([var_198,var_199,], output)
mutated_mod['func_200'] = func_200
mutated_mod = relay.transform.InferType()(mutated_mod)
var_219 = relay.var("var_219", dtype = "uint16", shape = (16, 14))#candidate|219|(16, 14)|var|uint16
var_220 = relay.var("var_220", dtype = "uint16", shape = (16, 14))#candidate|220|(16, 14)|var|uint16
bop_221 = relay.minimum(var_219.astype('uint16'), relay.reshape(var_220.astype('uint16'), relay.shape_of(var_219))) # shape=(16, 14)
bop_225 = relay.greater(var_219.astype('bool'), relay.reshape(bop_221.astype('bool'), relay.shape_of(var_219))) # shape=(16, 14)
bop_228 = relay.multiply(var_220.astype('int8'), relay.reshape(var_219.astype('int8'), relay.shape_of(var_220))) # shape=(16, 14)
uop_231 = relay.sigmoid(bop_221.astype('float32')) # shape=(16, 14)
bop_233 = relay.maximum(uop_231.astype('int8'), relay.reshape(bop_225.astype('int8'), relay.shape_of(uop_231))) # shape=(16, 14)
output = relay.Tuple([bop_228,bop_233,])
output2 = relay.Tuple([bop_228,bop_233,])
func_237 = relay.Function([var_219,var_220,], output)
mod['func_237'] = func_237
mod = relay.transform.InferType()(mod)
var_238 = relay.var("var_238", dtype = "uint16", shape = (16, 14))#candidate|238|(16, 14)|var|uint16
var_239 = relay.var("var_239", dtype = "uint16", shape = (16, 14))#candidate|239|(16, 14)|var|uint16
output = func_237(var_238,var_239,)
func_240 = relay.Function([var_238,var_239,], output)
mutated_mod['func_240'] = func_240
mutated_mod = relay.transform.InferType()(mutated_mod)
const_286 = relay.const([[[-6,3,8,2,6,-3,-4,4,-7,-8],[10,9,-5,-9,-2,6,-4,6,8,9]],[[-10,4,-3,2,3,-4,7,5,8,2],[6,10,8,-1,-5,-4,-3,9,-9,8]],[[10,-4,-7,8,4,-7,-9,-10,4,5],[2,-2,-8,-2,1,-10,4,-9,-5,-8]],[[10,-7,4,-8,-1,-8,2,9,-3,8],[3,-9,4,-9,7,-5,-5,-8,2,-9]],[[-8,-5,9,10,-4,-9,10,3,-6,-2],[10,-1,-4,5,-4,-8,-3,3,-6,5]],[[-5,7,2,9,2,2,3,-8,4,-2],[10,-3,4,5,2,3,6,-5,2,6]],[[-7,8,6,6,1,-10,-1,-4,-2,-8],[10,-4,-1,-7,-2,7,-3,-1,4,1]],[[-9,-8,6,5,2,-4,3,3,6,-7],[2,10,-2,4,-10,1,6,10,-3,1]]], dtype = "uint64")#candidate|286|(8, 2, 10)|const|uint64
var_287 = relay.var("var_287", dtype = "uint64", shape = (8, 2, 10))#candidate|287|(8, 2, 10)|var|uint64
bop_288 = relay.logical_xor(const_286.astype('uint64'), relay.reshape(var_287.astype('uint64'), relay.shape_of(const_286))) # shape=(8, 2, 10)
bop_295 = relay.logical_or(bop_288.astype('bool'), relay.reshape(const_286.astype('bool'), relay.shape_of(bop_288))) # shape=(8, 2, 10)
output = bop_295
output2 = bop_295
func_298 = relay.Function([var_287,], output)
mod['func_298'] = func_298
mod = relay.transform.InferType()(mod)
var_299 = relay.var("var_299", dtype = "uint64", shape = (8, 2, 10))#candidate|299|(8, 2, 10)|var|uint64
output = func_298(var_299)
func_300 = relay.Function([var_299], output)
mutated_mod['func_300'] = func_300
mutated_mod = relay.transform.InferType()(mutated_mod)
var_342 = relay.var("var_342", dtype = "uint8", shape = (13, 14))#candidate|342|(13, 14)|var|uint8
const_343 = relay.const([[3,1,8,5,-1,-3,8,6,-9,-4,6,2,-8,2],[7,3,7,-4,-1,10,1,8,-5,9,-9,-8,-10,-1],[-3,10,-10,-9,-8,-4,9,-10,9,8,-4,-10,2,10],[-10,-2,7,-4,6,3,-6,4,1,6,4,4,2,1],[-1,6,5,6,-10,6,-4,2,6,-5,-10,6,6,-9],[1,-2,2,-6,8,6,2,5,-3,-6,-5,-5,-2,8],[-1,6,-7,-6,8,6,4,5,-6,-4,9,5,10,8],[10,-8,3,5,-4,-10,5,-8,10,6,4,-2,-7,7],[-6,9,-1,1,3,-3,-8,-9,9,-3,1,-4,7,-7],[2,4,-1,7,2,-3,3,3,-10,-4,2,-9,-7,7],[6,10,2,-7,-1,-2,-10,9,2,5,-6,3,6,-10],[-3,-9,-3,-9,5,7,-8,-8,-5,-2,-10,-8,-5,4],[-3,6,-3,4,-4,-10,-9,-6,5,5,-2,6,-3,-10]], dtype = "uint8")#candidate|343|(13, 14)|const|uint8
bop_344 = relay.left_shift(var_342.astype('uint8'), relay.reshape(const_343.astype('uint8'), relay.shape_of(var_342))) # shape=(13, 14)
uop_347 = relay.log2(bop_344.astype('float32')) # shape=(13, 14)
bop_349 = relay.floor_mod(uop_347.astype('float64'), relay.reshape(var_342.astype('float64'), relay.shape_of(uop_347))) # shape=(13, 14)
uop_354 = relay.acos(uop_347.astype('float32')) # shape=(13, 14)
var_374 = relay.var("var_374", dtype = "float32", shape = (13, 14))#candidate|374|(13, 14)|var|float32
bop_375 = relay.logical_or(uop_354.astype('bool'), relay.reshape(var_374.astype('bool'), relay.shape_of(uop_354))) # shape=(13, 14)
output = relay.Tuple([bop_349,bop_375,])
output2 = relay.Tuple([bop_349,bop_375,])
func_378 = relay.Function([var_342,var_374,], output)
mod['func_378'] = func_378
mod = relay.transform.InferType()(mod)
mutated_mod['func_378'] = func_378
mutated_mod = relay.transform.InferType()(mutated_mod)
func_378_call = mutated_mod.get_global_var('func_378')
var_380 = relay.var("var_380", dtype = "uint8", shape = (13, 14))#candidate|380|(13, 14)|var|uint8
var_381 = relay.var("var_381", dtype = "float32", shape = (13, 14))#candidate|381|(13, 14)|var|float32
call_379 = func_378_call(var_380,var_381,)
output = call_379
func_382 = relay.Function([var_380,var_381,], output)
mutated_mod['func_382'] = func_382
mutated_mod = relay.transform.InferType()(mutated_mod)
var_386 = relay.var("var_386", dtype = "int32", shape = ())#candidate|386|()|var|int32
const_387 = relay.const([[-2,-3,-5,9,5],[-7,4,-9,10,3],[4,-6,-8,-8,-9],[-9,6,-7,-2,-8],[-2,10,-5,-3,10],[9,-9,6,-5,8],[-9,-5,10,-8,8],[1,-1,8,-3,-2],[-6,-6,-5,-10,-9],[6,-9,3,-5,-2],[-6,-5,-8,-6,-1]], dtype = "int32")#candidate|387|(11, 5)|const|int32
bop_388 = relay.greater_equal(var_386.astype('bool'), const_387.astype('bool')) # shape=(11, 5)
bop_393 = relay.minimum(bop_388.astype('int64'), relay.reshape(const_387.astype('int64'), relay.shape_of(bop_388))) # shape=(11, 5)
uop_399 = relay.log2(bop_388.astype('float64')) # shape=(11, 5)
bop_403 = relay.bitwise_or(uop_399.astype('uint8'), relay.reshape(bop_388.astype('uint8'), relay.shape_of(uop_399))) # shape=(11, 5)
bop_410 = relay.not_equal(bop_403.astype('bool'), var_386.astype('bool')) # shape=(11, 5)
var_413 = relay.var("var_413", dtype = "float64", shape = (11, 5))#candidate|413|(11, 5)|var|float64
bop_414 = relay.subtract(uop_399.astype('uint16'), relay.reshape(var_413.astype('uint16'), relay.shape_of(uop_399))) # shape=(11, 5)
bop_419 = relay.less(bop_410.astype('bool'), relay.reshape(bop_414.astype('bool'), relay.shape_of(bop_410))) # shape=(11, 5)
func_118_call = mod.get_global_var('func_118')
func_125_call = mutated_mod.get_global_var('func_125')
var_427 = relay.var("var_427", dtype = "float32", shape = (64,))#candidate|427|(64,)|var|float32
call_426 = relay.TupleGetItem(func_118_call(relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), ), 5)
call_428 = relay.TupleGetItem(func_125_call(relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), ), 5)
bop_429 = relay.less_equal(bop_388.astype('bool'), relay.reshape(bop_414.astype('bool'), relay.shape_of(bop_388))) # shape=(11, 5)
func_118_call = mod.get_global_var('func_118')
func_125_call = mutated_mod.get_global_var('func_125')
call_433 = relay.TupleGetItem(func_118_call(relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('bool'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), ), 1)
call_434 = relay.TupleGetItem(func_125_call(relay.reshape(var_427.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('float32'), [2, 8, 4]), relay.reshape(call_426.astype('bool'), [2, 8, 4]), relay.reshape(var_427.astype('bool'), [2, 8, 4]), ), 1)
bop_436 = relay.maximum(bop_419.astype('uint8'), relay.reshape(bop_429.astype('uint8'), relay.shape_of(bop_419))) # shape=(11, 5)
const_440 = relay.const([[False,True,False,True,True],[True,True,True,False,False],[True,False,False,True,True],[False,True,True,True,False],[False,True,True,True,True],[True,True,True,False,False],[True,False,True,True,False],[False,False,True,True,True],[True,True,True,True,False],[False,False,True,True,True],[False,False,False,True,True]], dtype = "bool")#candidate|440|(11, 5)|const|bool
bop_441 = relay.divide(bop_429.astype('float64'), relay.reshape(const_440.astype('float64'), relay.shape_of(bop_429))) # shape=(11, 5)
bop_445 = relay.left_shift(uop_399.astype('uint64'), relay.reshape(bop_403.astype('uint64'), relay.shape_of(uop_399))) # shape=(11, 5)
bop_451 = relay.greater_equal(bop_429.astype('bool'), relay.reshape(bop_388.astype('bool'), relay.shape_of(bop_429))) # shape=(11, 5)
uop_454 = relay.asinh(bop_451.astype('float32')) # shape=(11, 5)
func_196_call = mod.get_global_var('func_196')
func_200_call = mutated_mod.get_global_var('func_200')
var_457 = relay.var("var_457", dtype = "float32", shape = (16, 1))#candidate|457|(16, 1)|var|float32
call_456 = relay.TupleGetItem(func_196_call(relay.reshape(var_457.astype('float32'), [16,]), relay.reshape(var_457.astype('uint32'), [16,]), ), 0)
call_458 = relay.TupleGetItem(func_200_call(relay.reshape(var_457.astype('float32'), [16,]), relay.reshape(var_457.astype('uint32'), [16,]), ), 0)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
var_460 = relay.var("var_460", dtype = "uint64", shape = (160,))#candidate|460|(160,)|var|uint64
call_459 = func_298_call(relay.reshape(var_460.astype('uint64'), [8, 2, 10]))
call_461 = func_298_call(relay.reshape(var_460.astype('uint64'), [8, 2, 10]))
output = relay.Tuple([bop_393,call_426,var_427,call_433,bop_436,bop_441,bop_445,uop_454,call_456,var_457,call_459,var_460,])
output2 = relay.Tuple([bop_393,call_428,var_427,call_434,bop_436,bop_441,bop_445,uop_454,call_458,var_457,call_461,var_460,])
func_462 = relay.Function([var_386,var_413,var_427,var_457,var_460,], output)
mod['func_462'] = func_462
mod = relay.transform.InferType()(mod)
mutated_mod['func_462'] = func_462
mutated_mod = relay.transform.InferType()(mutated_mod)
func_462_call = mutated_mod.get_global_var('func_462')
var_464 = relay.var("var_464", dtype = "int32", shape = ())#candidate|464|()|var|int32
var_465 = relay.var("var_465", dtype = "float64", shape = (11, 5))#candidate|465|(11, 5)|var|float64
var_466 = relay.var("var_466", dtype = "float32", shape = (64,))#candidate|466|(64,)|var|float32
var_467 = relay.var("var_467", dtype = "float32", shape = (16, 1))#candidate|467|(16, 1)|var|float32
var_468 = relay.var("var_468", dtype = "uint64", shape = (160,))#candidate|468|(160,)|var|uint64
call_463 = func_462_call(var_464,var_465,var_466,var_467,var_468,)
output = call_463
func_469 = relay.Function([var_464,var_465,var_466,var_467,var_468,], output)
mutated_mod['func_469'] = func_469
mutated_mod = relay.transform.InferType()(mutated_mod)
const_506 = relay.const([[-5,4],[9,9],[-1,2],[2,-7]], dtype = "uint64")#candidate|506|(4, 2)|const|uint64
var_507 = relay.var("var_507", dtype = "uint64", shape = (4, 2))#candidate|507|(4, 2)|var|uint64
bop_508 = relay.greater_equal(const_506.astype('bool'), relay.reshape(var_507.astype('bool'), relay.shape_of(const_506))) # shape=(4, 2)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
const_514 = relay.const([5,-9,7,7,2,1,2,-4,3,7,-6,4,2,9,-8,7,8,-10,-10,6,-7,-3,-2,10,-2,1,-1,1,-5,-9,5,-6,-3,-8,3,2,-3,-9,-1,-2,5,3,8,-1,4,-8,5,-4,10,3,7,3,2,7,-8,-6,-2,-1,3,-5,-2,-8,7,-9,-3,10,8,10,-3,3,5,8,-10,-8,-1,-2,6,-3,-3,9,5,-7,-5,4,3,-1,9,-8,8,-4,10,3,-9,10,9,9,1,8,3,3,-1,6,6,1,-1,3,2,-9,2,-2,-9,4,6,-7,6,-5,-8,7,5,8,-2,7,-8,-6,6,-3,10,8,-1,-3,6,-4,-10,-8,8,3,10,-3,-8,1,6,8,1,-1,2,-9,-1,3,-3,-8,-1,-8,-1,-9,1,-1,-2,-4,9,-5], dtype = "uint64")#candidate|514|(160,)|const|uint64
call_513 = func_298_call(relay.reshape(const_514.astype('uint64'), [8, 2, 10]))
call_515 = func_298_call(relay.reshape(const_514.astype('uint64'), [8, 2, 10]))
output = relay.Tuple([bop_508,call_513,const_514,])
output2 = relay.Tuple([bop_508,call_515,const_514,])
func_519 = relay.Function([var_507,], output)
mod['func_519'] = func_519
mod = relay.transform.InferType()(mod)
mutated_mod['func_519'] = func_519
mutated_mod = relay.transform.InferType()(mutated_mod)
var_520 = relay.var("var_520", dtype = "uint64", shape = (4, 2))#candidate|520|(4, 2)|var|uint64
func_519_call = mutated_mod.get_global_var('func_519')
call_521 = func_519_call(var_520)
output = call_521
func_522 = relay.Function([var_520], output)
mutated_mod['func_522'] = func_522
mutated_mod = relay.transform.InferType()(mutated_mod)
var_561 = relay.var("var_561", dtype = "int8", shape = (3,))#candidate|561|(3,)|var|int8
var_562 = relay.var("var_562", dtype = "int8", shape = (3,))#candidate|562|(3,)|var|int8
bop_563 = relay.bitwise_or(var_561.astype('int8'), relay.reshape(var_562.astype('int8'), relay.shape_of(var_561))) # shape=(3,)
uop_569 = relay.sigmoid(bop_563.astype('float64')) # shape=(3,)
uop_576 = relay.log2(uop_569.astype('float64')) # shape=(3,)
uop_578 = relay.log(uop_576.astype('float32')) # shape=(3,)
const_580 = relay.const([-0.473128,0.012314,9.464239], dtype = "float32")#candidate|580|(3,)|const|float32
bop_581 = relay.less(uop_578.astype('bool'), relay.reshape(const_580.astype('bool'), relay.shape_of(uop_578))) # shape=(3,)
var_586 = relay.var("var_586", dtype = "float64", shape = (3,))#candidate|586|(3,)|var|float64
bop_587 = relay.equal(uop_576.astype('bool'), relay.reshape(var_586.astype('bool'), relay.shape_of(uop_576))) # shape=(3,)
uop_590 = relay.asin(bop_587.astype('float32')) # shape=(3,)
const_593 = relay.const([0.338839,7.707932,5.641306], dtype = "float64")#candidate|593|(3,)|const|float64
bop_594 = relay.minimum(uop_576.astype('int64'), relay.reshape(const_593.astype('int64'), relay.shape_of(uop_576))) # shape=(3,)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
const_598 = relay.const([4,-8,4,-10,-8,1,-5,6,7,-5,10,-8,-7,10,-9,-5,5,-8,1,-8,-1,1,-3,9,7,-8,-6,-9,-7,6,10,-9,-5,5,-6,-6,-4,-10,-4,-9,-5,-7,3,-3,5,-6,-1,9,1,3,-7,-4,-6,9,5,2,-3,2,4,-6,-5,1,5,-9,6,-6,10,5,4,7,-2,-1,5,10,-9,8,9,7,1,-2,9,-1,8,7,8,3,-1,4,10,-7,2,-10,-5,7,-6,-5,7,-5,10,6,6,-9,-8,6,8,-4,7,3,2,9,7,2,-10,8,1,-9,5,-5,-10,10,4,-10,7,-4,1,8,9,-1,-9,8,-5,8,-4,5,-5,4,7,10,-7,-5,-10,-8,7,8,10,-7,-3,-5,9,5,-2,-3,4,5,10,10,4,-8,9,8], dtype = "uint64")#candidate|598|(160,)|const|uint64
call_597 = func_298_call(relay.reshape(const_598.astype('uint64'), [8, 2, 10]))
call_599 = func_298_call(relay.reshape(const_598.astype('uint64'), [8, 2, 10]))
var_600 = relay.var("var_600", dtype = "float64", shape = (3,))#candidate|600|(3,)|var|float64
bop_601 = relay.floor_mod(uop_569.astype('float32'), relay.reshape(var_600.astype('float32'), relay.shape_of(uop_569))) # shape=(3,)
func_118_call = mod.get_global_var('func_118')
func_125_call = mutated_mod.get_global_var('func_125')
const_608 = relay.const([6.475973,2.650557,-6.063712,-3.283369,0.455959,-8.758481,-5.475071,-2.659296,0.472498,1.526173,0.460233,-0.794074,9.481505,-7.015486,-5.239867,-4.551713,-4.346753,1.182823,3.210917,-2.630296,1.271523,-5.744831,4.250452,-4.895381,-9.892154,-5.876687,9.059233,1.884589,-5.018667,9.310681,5.539939,-8.362586,3.365618,-1.704370,5.912508,-3.071464,1.728417,-4.681097,5.743670,-9.745163,7.828192,-6.351440,8.764691,1.582450,7.815116,-3.193992,-7.330440,-0.388879,9.631066,3.354795,7.515742,1.683472,-3.987353,4.093426,-2.660131,3.497719,-7.333344,-5.783449,8.123882,1.697229,4.880109,1.507169,1.972559,0.357289], dtype = "float32")#candidate|608|(64,)|const|float32
call_607 = relay.TupleGetItem(func_118_call(relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('bool'), [2, 8, 4]), relay.reshape(const_608.astype('bool'), [2, 8, 4]), ), 4)
call_609 = relay.TupleGetItem(func_125_call(relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('float32'), [2, 8, 4]), relay.reshape(const_608.astype('bool'), [2, 8, 4]), relay.reshape(const_608.astype('bool'), [2, 8, 4]), ), 4)
bop_610 = relay.mod(bop_581.astype('float64'), relay.reshape(bop_594.astype('float64'), relay.shape_of(bop_581))) # shape=(3,)
bop_615 = relay.greater_equal(uop_578.astype('bool'), relay.reshape(const_580.astype('bool'), relay.shape_of(uop_578))) # shape=(3,)
output = relay.Tuple([uop_590,call_597,const_598,bop_601,call_607,const_608,bop_610,bop_615,])
output2 = relay.Tuple([uop_590,call_599,const_598,bop_601,call_609,const_608,bop_610,bop_615,])
func_619 = relay.Function([var_561,var_562,var_586,var_600,], output)
mod['func_619'] = func_619
mod = relay.transform.InferType()(mod)
mutated_mod['func_619'] = func_619
mutated_mod = relay.transform.InferType()(mutated_mod)
func_619_call = mutated_mod.get_global_var('func_619')
var_621 = relay.var("var_621", dtype = "int8", shape = (3,))#candidate|621|(3,)|var|int8
var_622 = relay.var("var_622", dtype = "int8", shape = (3,))#candidate|622|(3,)|var|int8
var_623 = relay.var("var_623", dtype = "float64", shape = (3,))#candidate|623|(3,)|var|float64
var_624 = relay.var("var_624", dtype = "float64", shape = (3,))#candidate|624|(3,)|var|float64
call_620 = func_619_call(var_621,var_622,var_623,var_624,)
output = call_620
func_625 = relay.Function([var_621,var_622,var_623,var_624,], output)
mutated_mod['func_625'] = func_625
mutated_mod = relay.transform.InferType()(mutated_mod)
var_672 = relay.var("var_672", dtype = "int64", shape = (16, 13))#candidate|672|(16, 13)|var|int64
const_673 = relay.const([[-4,1,-10,-10,-4,-6,7,8,8,-10,4,9,1],[9,-5,7,1,9,-10,10,7,3,-7,5,6,8],[7,-9,7,-2,-1,10,-8,-5,-7,10,-2,7,5],[-5,-7,-1,2,3,-8,-2,5,-2,2,-1,4,-4],[-6,6,-2,2,-5,4,-5,-9,-5,3,-6,2,2],[-6,5,-3,-10,2,-9,3,-9,3,5,8,-4,9],[6,5,-7,6,-10,2,-7,4,-1,10,-4,-2,3],[-9,1,-7,1,6,-5,-1,1,-7,-10,-4,5,-8],[5,8,-2,-6,-6,9,-7,7,-7,-6,-2,7,-1],[-2,5,7,-4,9,-1,8,-2,-7,3,-10,10,1],[9,5,-8,-10,1,-2,-2,2,10,3,8,-8,3],[3,5,7,-1,6,3,2,-3,10,5,-1,1,-4],[3,2,-3,-9,-1,2,3,1,6,-2,6,-6,-2],[-6,10,-4,6,-9,2,-7,8,-5,1,-7,-7,-8],[10,6,1,-7,-3,-4,-8,10,-2,-5,8,1,5],[-8,6,-7,-2,-8,-4,3,-7,-8,9,7,5,-2]], dtype = "int64")#candidate|673|(16, 13)|const|int64
bop_674 = relay.bitwise_xor(var_672.astype('int64'), relay.reshape(const_673.astype('int64'), relay.shape_of(var_672))) # shape=(16, 13)
uop_678 = relay.asin(bop_674.astype('float64')) # shape=(16, 13)
func_378_call = mod.get_global_var('func_378')
func_382_call = mutated_mod.get_global_var('func_382')
const_682 = relay.const([[-3],[-8],[-5],[7],[1],[-8],[5],[2],[-10],[7],[1],[-3],[3],[-8],[5],[4],[10],[4],[-3],[5],[-1],[6],[-4],[1],[-3],[6],[-10],[-8],[2],[-1],[-4],[4],[7],[1],[9],[1],[-8],[-5],[5],[8],[6],[3],[-3],[-10],[9],[-4],[6],[-6],[6],[-10],[-5],[6],[-10],[-8],[8],[-5],[2],[-10],[-9],[3],[-2],[8],[5],[-1],[-5],[-7],[-6],[6],[3],[-2],[-1],[-5],[5],[1],[-6],[1],[-1],[7],[4],[6],[-1],[-2],[7],[4],[1],[1],[3],[9],[3],[-1],[4],[8],[10],[9],[5],[5],[-2],[4],[2],[-9],[2],[-7],[8],[-2],[-5],[10],[2],[3],[5],[-5],[-1],[6],[-7],[5],[8],[6],[-6],[4],[8],[4],[6],[3],[4],[-9],[-9],[-9],[7],[6],[-8],[7],[4],[6],[6],[-5],[6],[8],[-7],[-2],[-6],[-10],[-9],[5],[-6],[4],[-1],[10],[4],[-7],[-7],[-5],[-4],[-9],[-3],[-2],[3],[-1],[-2],[-10],[9],[-4],[5],[-6],[-3],[-7],[5],[-5],[4],[-8],[4],[3],[-3],[-10],[5],[3],[4],[-4],[5],[-7],[-3],[9],[6],[-7]], dtype = "uint8")#candidate|682|(182, 1)|const|uint8
call_681 = relay.TupleGetItem(func_378_call(relay.reshape(const_682.astype('uint8'), [13, 14]), relay.reshape(const_682.astype('float32'), [13, 14]), ), 1)
call_683 = relay.TupleGetItem(func_382_call(relay.reshape(const_682.astype('uint8'), [13, 14]), relay.reshape(const_682.astype('float32'), [13, 14]), ), 1)
func_619_call = mod.get_global_var('func_619')
func_625_call = mutated_mod.get_global_var('func_625')
var_686 = relay.var("var_686", dtype = "int8", shape = (3,))#candidate|686|(3,)|var|int8
call_685 = relay.TupleGetItem(func_619_call(relay.reshape(var_686.astype('int8'), [3,]), relay.reshape(var_686.astype('int8'), [3,]), relay.reshape(var_686.astype('float64'), [3,]), relay.reshape(var_686.astype('float64'), [3,]), ), 0)
call_687 = relay.TupleGetItem(func_625_call(relay.reshape(var_686.astype('int8'), [3,]), relay.reshape(var_686.astype('int8'), [3,]), relay.reshape(var_686.astype('float64'), [3,]), relay.reshape(var_686.astype('float64'), [3,]), ), 0)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
var_689 = relay.var("var_689", dtype = "uint64", shape = (160,))#candidate|689|(160,)|var|uint64
call_688 = func_298_call(relay.reshape(var_689.astype('uint64'), [8, 2, 10]))
call_690 = func_298_call(relay.reshape(var_689.astype('uint64'), [8, 2, 10]))
bop_693 = relay.less_equal(uop_678.astype('bool'), relay.reshape(const_673.astype('bool'), relay.shape_of(uop_678))) # shape=(16, 13)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
call_696 = func_298_call(relay.reshape(call_688.astype('uint64'), [8, 2, 10]))
call_697 = func_298_call(relay.reshape(call_688.astype('uint64'), [8, 2, 10]))
uop_698 = relay.log10(bop_693.astype('float64')) # shape=(16, 13)
bop_700 = relay.less(bop_693.astype('bool'), relay.reshape(const_673.astype('bool'), relay.shape_of(bop_693))) # shape=(16, 13)
bop_703 = relay.bitwise_or(uop_698.astype('uint16'), relay.reshape(uop_678.astype('uint16'), relay.shape_of(uop_698))) # shape=(16, 13)
bop_709 = relay.bitwise_xor(uop_698.astype('uint64'), relay.reshape(bop_703.astype('uint64'), relay.shape_of(uop_698))) # shape=(16, 13)
var_712 = relay.var("var_712", dtype = "float64", shape = (16, 13))#candidate|712|(16, 13)|var|float64
bop_713 = relay.multiply(uop_698.astype('float32'), relay.reshape(var_712.astype('float32'), relay.shape_of(uop_698))) # shape=(16, 13)
bop_717 = relay.add(bop_703.astype('uint32'), relay.reshape(uop_698.astype('uint32'), relay.shape_of(bop_703))) # shape=(16, 13)
output = relay.Tuple([call_681,const_682,call_685,var_686,call_688,var_689,call_696,bop_700,bop_709,bop_713,bop_717,])
output2 = relay.Tuple([call_683,const_682,call_687,var_686,call_690,var_689,call_697,bop_700,bop_709,bop_713,bop_717,])
func_720 = relay.Function([var_672,var_686,var_689,var_712,], output)
mod['func_720'] = func_720
mod = relay.transform.InferType()(mod)
var_721 = relay.var("var_721", dtype = "int64", shape = (16, 13))#candidate|721|(16, 13)|var|int64
var_722 = relay.var("var_722", dtype = "int8", shape = (3,))#candidate|722|(3,)|var|int8
var_723 = relay.var("var_723", dtype = "uint64", shape = (160,))#candidate|723|(160,)|var|uint64
var_724 = relay.var("var_724", dtype = "float64", shape = (16, 13))#candidate|724|(16, 13)|var|float64
output = func_720(var_721,var_722,var_723,var_724,)
func_725 = relay.Function([var_721,var_722,var_723,var_724,], output)
mutated_mod['func_725'] = func_725
mutated_mod = relay.transform.InferType()(mutated_mod)
var_729 = relay.var("var_729", dtype = "int32", shape = (4, 13))#candidate|729|(4, 13)|var|int32
const_730 = relay.const([[-1,-5,4,-10,-10,2,-4,6,9,-7,-1,-5,-2],[1,-10,6,-5,-4,4,8,8,-9,-8,-10,-10,4],[-7,8,-10,3,-3,-3,-1,7,-10,-1,-5,-3,-5],[2,10,-9,-4,6,10,-6,-10,1,-6,9,5,6]], dtype = "int32")#candidate|730|(4, 13)|const|int32
bop_731 = relay.minimum(var_729.astype('int32'), relay.reshape(const_730.astype('int32'), relay.shape_of(var_729))) # shape=(4, 13)
func_237_call = mod.get_global_var('func_237')
func_240_call = mutated_mod.get_global_var('func_240')
const_739 = relay.const([1,-10,-1,7,-7,-1,1,7,-1,-10,2,7,8,-8,-4,-6,1,4,-2,4,-3,4,4,1,10,2,-1,-7,-7,-9,8,-5,-5,8,2,-7,-6,-1,4,10,5,6,-2,-4,10,7,-1,6,3,-3,8,10,-3,-8,4,1,-9,5,1,-8,5,10,10,-3,-4,8,6,-7,2,6,-3,-5,-5,2,2,-1,-7,-2,-9,-2,1,4,-9,-8,6,6,2,-2,4,5,3,-10,8,-7,-6,1,-8,-5,-8,9,7,-5,-7,-8,-10,8,10,6,-9,-5,2,2,-1,-9,8,-4,3,5,-6,-2,5,-2,7,3,10,3,-3,4,-10,4,-10,4,-10,3,4,-2,7,-6,-3,2,-6,-7,-10,7,-3,9,-1,-9,5,-7,-9,-8,5,1,-10,1,-8,8,-1,-7,8,-10,-2,-2,-10,4,9,4,-3,-2,-7,-2,5,-2,-5,-1,-6,1,-7,-3,-1,-6,8,-4,6,7,-3,10,-6,-2,-1,3,-3,7,9,-9,-8,3,-9,-10,-7,10,-3,7,3,2,2,3,-2,-7,2,-3,3,8,-3,8,-3,-9,9,8,5,5,-7,-5], dtype = "uint16")#candidate|739|(224,)|const|uint16
call_738 = relay.TupleGetItem(func_237_call(relay.reshape(const_739.astype('uint16'), [16, 14]), relay.reshape(const_739.astype('uint16'), [16, 14]), ), 0)
call_740 = relay.TupleGetItem(func_240_call(relay.reshape(const_739.astype('uint16'), [16, 14]), relay.reshape(const_739.astype('uint16'), [16, 14]), ), 0)
output = relay.Tuple([bop_731,call_738,const_739,])
output2 = relay.Tuple([bop_731,call_740,const_739,])
func_743 = relay.Function([var_729,], output)
mod['func_743'] = func_743
mod = relay.transform.InferType()(mod)
mutated_mod['func_743'] = func_743
mutated_mod = relay.transform.InferType()(mutated_mod)
var_744 = relay.var("var_744", dtype = "int32", shape = (4, 13))#candidate|744|(4, 13)|var|int32
func_743_call = mutated_mod.get_global_var('func_743')
call_745 = func_743_call(var_744)
output = call_745
func_746 = relay.Function([var_744], output)
mutated_mod['func_746'] = func_746
mutated_mod = relay.transform.InferType()(mutated_mod)
var_787 = relay.var("var_787", dtype = "float32", shape = (14,))#candidate|787|(14,)|var|float32
uop_788 = relay.acosh(var_787.astype('float32')) # shape=(14,)
uop_794 = relay.log10(uop_788.astype('float32')) # shape=(14,)
uop_797 = relay.acos(uop_794.astype('float64')) # shape=(14,)
uop_799 = relay.asinh(uop_797.astype('float64')) # shape=(14,)
bop_801 = relay.power(uop_797.astype('float64'), relay.reshape(uop_788.astype('float64'), relay.shape_of(uop_797))) # shape=(14,)
uop_804 = relay.sinh(uop_797.astype('float64')) # shape=(14,)
bop_807 = relay.bitwise_xor(uop_799.astype('int64'), relay.reshape(uop_794.astype('int64'), relay.shape_of(uop_799))) # shape=(14,)
bop_812 = relay.subtract(var_787.astype('float32'), relay.reshape(uop_804.astype('float32'), relay.shape_of(var_787))) # shape=(14,)
const_815 = relay.const([7.159348,4.595766,-5.603446,4.328169,-5.630936,-7.121349,5.459040,-6.893014,-0.224905,-7.230833,-0.841542,-1.202213,3.255462,-8.205559], dtype = "float64")#candidate|815|(14,)|const|float64
bop_816 = relay.logical_and(uop_797.astype('bool'), relay.reshape(const_815.astype('bool'), relay.shape_of(uop_797))) # shape=(14,)
func_462_call = mod.get_global_var('func_462')
func_469_call = mutated_mod.get_global_var('func_469')
const_826 = relay.const(4, dtype = "int32")#candidate|826|()|const|int32
const_827 = relay.const([-1.879158,-9.915289,3.304654,-9.557820,-7.811047,5.572449,1.664270,7.472754,9.409436,1.538449,-4.923647,-6.889689,-7.315471,9.518214,7.454971,-5.299532,7.039218,-7.761338,6.975279,-5.983896,4.183363,-8.321228,8.427074,-0.490786,9.197367,2.472556,-2.609988,-0.942442,5.611564,-3.220239,-9.576529,-7.796857,-4.619536,-3.504685,9.666556,4.389246,8.409078,6.984477,3.008034,6.330867,-2.327195,-9.872488,-5.555393,0.880735,4.377789,6.780111,4.269800,6.910302,-6.910492,4.194256,-2.631028,-4.481035,-5.688644,-8.575201,3.907647], dtype = "float64")#candidate|827|(55,)|const|float64
var_828 = relay.var("var_828", dtype = "float32", shape = (64,))#candidate|828|(64,)|var|float32
var_829 = relay.var("var_829", dtype = "float32", shape = (16,))#candidate|829|(16,)|var|float32
var_830 = relay.var("var_830", dtype = "uint64", shape = (160,))#candidate|830|(160,)|var|uint64
call_825 = relay.TupleGetItem(func_462_call(relay.reshape(const_826.astype('int32'), []), relay.reshape(const_827.astype('float64'), [11, 5]), relay.reshape(var_828.astype('float32'), [64,]), relay.reshape(var_829.astype('float32'), [16, 1]), relay.reshape(var_830.astype('uint64'), [160,]), ), 9)
call_831 = relay.TupleGetItem(func_469_call(relay.reshape(const_826.astype('int32'), []), relay.reshape(const_827.astype('float64'), [11, 5]), relay.reshape(var_828.astype('float32'), [64,]), relay.reshape(var_829.astype('float32'), [16, 1]), relay.reshape(var_830.astype('uint64'), [160,]), ), 9)
func_237_call = mod.get_global_var('func_237')
func_240_call = mutated_mod.get_global_var('func_240')
var_833 = relay.var("var_833", dtype = "uint16", shape = (224,))#candidate|833|(224,)|var|uint16
call_832 = relay.TupleGetItem(func_237_call(relay.reshape(var_833.astype('uint16'), [16, 14]), relay.reshape(var_833.astype('uint16'), [16, 14]), ), 0)
call_834 = relay.TupleGetItem(func_240_call(relay.reshape(var_833.astype('uint16'), [16, 14]), relay.reshape(var_833.astype('uint16'), [16, 14]), ), 0)
func_743_call = mod.get_global_var('func_743')
func_746_call = mutated_mod.get_global_var('func_746')
var_836 = relay.var("var_836", dtype = "int32", shape = (52,))#candidate|836|(52,)|var|int32
call_835 = relay.TupleGetItem(func_743_call(relay.reshape(var_836.astype('int32'), [4, 13])), 1)
call_837 = relay.TupleGetItem(func_746_call(relay.reshape(var_836.astype('int32'), [4, 13])), 1)
func_298_call = mod.get_global_var('func_298')
func_300_call = mutated_mod.get_global_var('func_300')
call_838 = func_298_call(relay.reshape(var_830.astype('uint64'), [8, 2, 10]))
call_839 = func_298_call(relay.reshape(var_830.astype('uint64'), [8, 2, 10]))
uop_840 = relay.log2(bop_807.astype('float32')) # shape=(14,)
var_842 = relay.var("var_842", dtype = "float32", shape = (14,))#candidate|842|(14,)|var|float32
bop_843 = relay.equal(uop_840.astype('bool'), relay.reshape(var_842.astype('bool'), relay.shape_of(uop_840))) # shape=(14,)
bop_852 = relay.bitwise_and(uop_840.astype('int16'), call_835.astype('int16')) # shape=(16, 14)
bop_855 = relay.bitwise_and(uop_840.astype('int16'), call_837.astype('int16')) # shape=(16, 14)
bop_856 = relay.bitwise_or(bop_852.astype('uint64'), relay.reshape(call_835.astype('uint64'), relay.shape_of(bop_852))) # shape=(16, 14)
bop_859 = relay.bitwise_or(bop_855.astype('uint64'), relay.reshape(call_837.astype('uint64'), relay.shape_of(bop_855))) # shape=(16, 14)
output = relay.Tuple([bop_801,bop_812,bop_816,call_825,const_826,const_827,var_828,var_829,var_830,call_832,var_833,var_836,call_838,bop_843,bop_856,])
output2 = relay.Tuple([bop_801,bop_812,bop_816,call_831,const_826,const_827,var_828,var_829,var_830,call_834,var_833,var_836,call_839,bop_843,bop_859,])
func_860 = relay.Function([var_787,var_828,var_829,var_830,var_833,var_836,var_842,], output)
mod['func_860'] = func_860
mod = relay.transform.InferType()(mod)
var_861 = relay.var("var_861", dtype = "float32", shape = (14,))#candidate|861|(14,)|var|float32
var_862 = relay.var("var_862", dtype = "float32", shape = (64,))#candidate|862|(64,)|var|float32
var_863 = relay.var("var_863", dtype = "float32", shape = (16,))#candidate|863|(16,)|var|float32
var_864 = relay.var("var_864", dtype = "uint64", shape = (160,))#candidate|864|(160,)|var|uint64
var_865 = relay.var("var_865", dtype = "uint16", shape = (224,))#candidate|865|(224,)|var|uint16
var_866 = relay.var("var_866", dtype = "int32", shape = (52,))#candidate|866|(52,)|var|int32
var_867 = relay.var("var_867", dtype = "float32", shape = (14,))#candidate|867|(14,)|var|float32
output = func_860(var_861,var_862,var_863,var_864,var_865,var_866,var_867,)
func_868 = relay.Function([var_861,var_862,var_863,var_864,var_865,var_866,var_867,], output)
mutated_mod['func_868'] = func_868
mutated_mod = relay.transform.InferType()(mutated_mod)
const_870 = relay.const([3.922378,6.018823,-3.881977,0.090813,-6.195462,-6.186168,-7.624687,-3.317088,9.722451,0.582993,6.771823,0.501965,2.194747], dtype = "float32")#candidate|870|(13,)|const|float32
uop_871 = relay.log10(const_870.astype('float32')) # shape=(13,)
uop_873 = relay.exp(uop_871.astype('float64')) # shape=(13,)
bop_875 = relay.greater_equal(const_870.astype('bool'), relay.reshape(uop_873.astype('bool'), relay.shape_of(const_870))) # shape=(13,)
bop_883 = relay.equal(uop_871.astype('bool'), relay.reshape(const_870.astype('bool'), relay.shape_of(uop_871))) # shape=(13,)
uop_888 = relay.log2(bop_883.astype('float32')) # shape=(13,)
var_891 = relay.var("var_891", dtype = "float64", shape = (13,))#candidate|891|(13,)|var|float64
bop_892 = relay.power(uop_873.astype('float32'), relay.reshape(var_891.astype('float32'), relay.shape_of(uop_873))) # shape=(13,)
bop_897 = relay.right_shift(uop_873.astype('uint64'), relay.reshape(var_891.astype('uint64'), relay.shape_of(uop_873))) # shape=(13,)
output = relay.Tuple([bop_875,uop_888,bop_892,bop_897,])
output2 = relay.Tuple([bop_875,uop_888,bop_892,bop_897,])
F = relay.Function([var_891,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_891,], output2)
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
input_891= np.array([-5.314238,-9.573036,-1.761176,4.179846,7.028057,5.103951,-4.826578,8.282042,5.042925,-1.581223,-0.230528,0.327489,-1.196418], dtype='float64')
module1.set_input('var_891', input_891)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_891, )
res3 = intrp3.evaluate()(input_891, )
res4 = intrp4.evaluate()(input_891, )
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
res1_2 = module1.get_output(2).asnumpy()
res2_2 = res2[2].asnumpy()
res3_2 = res3[2].asnumpy()
res4_2 = res4[2].asnumpy()
np.testing.assert_allclose(res1_2 ,res2_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res3_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res4_2, atol=1e-3, rtol=1e-3)
(res1_2 == res2_2).all()
(res1_2 == res3_2).all()
(res1_2 == res4_2).all()
res1_3 = module1.get_output(3).asnumpy()
res2_3 = res2[3].asnumpy()
res3_3 = res3[3].asnumpy()
res4_3 = res4[3].asnumpy()
np.testing.assert_allclose(res1_3 ,res2_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res3_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_3 ,res4_3, atol=1e-3, rtol=1e-3)
(res1_3 == res2_3).all()
(res1_3 == res3_3).all()
(res1_3 == res4_3).all()
module5.set_input('var_891', input_891)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_891, )
res7 = intrp7.evaluate()(input_891, )
res8 = intrp8.evaluate()(input_891, )
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
res5_2 = module5.get_output(2).asnumpy()
res6_2 = res6[2].asnumpy()
res7_2 = res7[2].asnumpy()
res8_2 = res8[2].asnumpy()
np.testing.assert_allclose(res5_2 ,res6_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res7_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res8_2, atol=1e-3, rtol=1e-3)
(res5_2 == res6_2).all()
(res5_2 == res7_2).all()
(res5_2 == res8_2).all()
res5_3 = module5.get_output(3).asnumpy()
res6_3 = res6[3].asnumpy()
res7_3 = res7[3].asnumpy()
res8_3 = res8[3].asnumpy()
np.testing.assert_allclose(res5_3 ,res6_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res7_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_3 ,res8_3, atol=1e-3, rtol=1e-3)
(res5_3 == res6_3).all()
(res5_3 == res7_3).all()
(res5_3 == res8_3).all()
module9.set_input('var_891', input_891)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_891, )
res11 = intrp11.evaluate()(input_891, )
res12 = intrp12.evaluate()(input_891, )
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
res9_2 = module9.get_output(2).asnumpy()
res10_2 = res10[2].asnumpy()
res11_2 = res11[2].asnumpy()
res12_2 = res12[2].asnumpy()
np.testing.assert_allclose(res9_2 ,res10_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res11_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res12_2, atol=1e-3, rtol=1e-3)
(res9_2 == res10_2).all()
(res9_2 == res11_2).all()
(res9_2 == res12_2).all()
res9_3 = module9.get_output(3).asnumpy()
res10_3 = res10[3].asnumpy()
res11_3 = res11[3].asnumpy()
res12_3 = res12[3].asnumpy()
np.testing.assert_allclose(res9_3 ,res10_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res11_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_3 ,res12_3, atol=1e-3, rtol=1e-3)
(res9_3 == res10_3).all()
(res9_3 == res11_3).all()
(res9_3 == res12_3).all()
module13.set_input('var_891', input_891)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_891, )
res15 = intrp15.evaluate()(input_891, )
res16 = intrp16.evaluate()(input_891, )
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
res13_2 = module13.get_output(2).asnumpy()
res14_2 = res14[2].asnumpy()
res15_2 = res15[2].asnumpy()
res16_2 = res16[2].asnumpy()
np.testing.assert_allclose(res13_2 ,res14_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res15_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res16_2, atol=1e-3, rtol=1e-3)
(res13_2 == res14_2).all()
(res13_2 == res15_2).all()
(res13_2 == res16_2).all()
res13_3 = module13.get_output(3).asnumpy()
res14_3 = res14[3].asnumpy()
res15_3 = res15[3].asnumpy()
res16_3 = res16[3].asnumpy()
np.testing.assert_allclose(res13_3 ,res14_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res15_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_3 ,res16_3, atol=1e-3, rtol=1e-3)
(res13_3 == res14_3).all()
(res13_3 == res15_3).all()
(res13_3 == res16_3).all()
module17.set_input('var_891', input_891)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_891, )
res19 = intrp19.evaluate()(input_891, )
res20 = intrp20.evaluate()(input_891, )
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
res17_2 = module17.get_output(2).asnumpy()
res18_2 = res18[2].asnumpy()
res19_2 = res19[2].asnumpy()
res20_2 = res20[2].asnumpy()
np.testing.assert_allclose(res17_2 ,res18_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res19_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res20_2, atol=1e-3, rtol=1e-3)
(res17_2 == res18_2).all()
(res17_2 == res19_2).all()
(res17_2 == res20_2).all()
res17_3 = module17.get_output(3).asnumpy()
res18_3 = res18[3].asnumpy()
res19_3 = res19[3].asnumpy()
res20_3 = res20[3].asnumpy()
np.testing.assert_allclose(res17_3 ,res18_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res19_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_3 ,res20_3, atol=1e-3, rtol=1e-3)
(res17_3 == res18_3).all()
(res17_3 == res19_3).all()
(res17_3 == res20_3).all()
module21.set_input('var_891', input_891)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_891, )
res23 = intrp23.evaluate()(input_891, )
res24 = intrp24.evaluate()(input_891, )
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
res21_2 = module21.get_output(2).asnumpy()
res22_2 = res22[2].asnumpy()
res23_2 = res23[2].asnumpy()
res24_2 = res24[2].asnumpy()
np.testing.assert_allclose(res21_2 ,res22_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res23_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res24_2, atol=1e-3, rtol=1e-3)
(res21_2 == res22_2).all()
(res21_2 == res23_2).all()
(res21_2 == res24_2).all()
res21_3 = module21.get_output(3).asnumpy()
res22_3 = res22[3].asnumpy()
res23_3 = res23[3].asnumpy()
res24_3 = res24[3].asnumpy()
np.testing.assert_allclose(res21_3 ,res22_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res23_3, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_3 ,res24_3, atol=1e-3, rtol=1e-3)
(res21_3 == res22_3).all()
(res21_3 == res23_3).all()
(res21_3 == res24_3).all()

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