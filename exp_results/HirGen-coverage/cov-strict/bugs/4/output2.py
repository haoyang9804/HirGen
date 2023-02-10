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
const_38 = relay.const([[[0.191755,-4.747551,-8.664948,-7.360424,-4.915304,-2.095567],[-1.907064,-2.197218,-6.438810,9.897054,8.772280,-0.918283],[9.542670,-8.432312,2.962102,-8.644170,0.853669,-7.828227]],[[4.046029,2.173546,-9.963657,-8.480234,-7.158134,-5.915417],[-4.735352,-1.420829,2.792470,-7.957833,-7.366517,7.243793],[-7.443952,3.172060,-0.968279,-8.648586,-0.755916,-8.432390]],[[3.810194,0.431354,7.373703,3.954687,1.361664,0.198000],[-0.687721,2.423024,-8.823308,-5.172075,-3.786978,7.241162],[5.198474,9.107823,9.521405,-8.227096,-3.370519,1.751768]],[[2.553776,5.789359,-2.748687,-3.951210,1.663088,-4.355315],[-0.814411,-2.808731,-9.553142,5.344339,0.861846,1.469839],[-8.601271,-0.692337,-3.999270,-5.826174,-0.243115,-0.851198]],[[-4.703895,9.676481,5.372738,-5.952777,-4.511586,-8.255020],[-8.689866,-8.562351,-5.222813,-2.017320,4.254368,3.260680],[-2.202929,-6.183549,-6.167639,-9.596430,5.481875,-6.798200]],[[7.617522,-3.100716,9.684056,-4.253519,-0.736633,-3.399502],[-6.430376,-5.213045,0.309575,5.348097,0.158224,-1.080141],[1.299586,5.977001,-7.707238,1.414224,-9.478766,3.763535]],[[-3.328876,3.877486,-9.770268,-8.288058,-9.075945,-2.851532],[-2.488934,-5.585936,1.854838,6.034116,8.933751,3.817794],[-5.947228,2.726520,-8.636209,8.571614,-1.716521,-7.596311]],[[-3.000772,3.848232,4.886588,-4.465477,-5.255203,1.319818],[3.414168,-5.387367,7.886842,-4.507509,-1.049689,-6.252595],[-3.888657,2.384757,-1.316112,5.424936,-3.421034,-2.984314]],[[-5.954523,-5.333761,9.698709,4.268070,-9.305471,2.629164],[1.040776,-0.195240,6.091376,8.132707,-8.248357,-7.133365],[-6.875128,8.892801,-4.034493,-6.854394,-1.912216,1.609543]],[[-4.857269,-2.865373,-6.904202,8.876199,7.328959,8.371180],[-5.854267,-2.007412,-2.403138,-4.945895,-5.471265,6.934380],[5.828964,2.235783,-5.457294,7.736017,7.846302,-8.068081]],[[1.394754,-5.958365,4.296212,3.719317,-0.827570,-9.656710],[-2.272717,5.072837,-3.214055,8.259099,-2.611239,4.774718],[-1.562273,1.789811,-9.752910,1.373492,0.816718,-5.712009]],[[-3.550668,2.315762,7.428883,1.536275,-7.493857,-8.924589],[0.658196,-1.555555,-6.264501,7.598727,-8.051500,4.958734],[-6.325894,-4.733104,-0.520636,-4.795846,-1.163466,9.063576]],[[6.347569,0.491312,8.986419,-0.191175,2.120158,-4.432502],[-3.556110,-9.139224,-4.154089,8.008219,3.766221,0.837323],[-6.721774,-9.518529,1.071091,-9.239331,-2.559603,-1.657961]],[[0.799898,-3.793854,4.142075,-3.532980,-6.733589,-8.620927],[-3.223008,7.807143,5.676792,8.381471,-8.401228,-1.992123],[0.052714,-8.840826,-1.103415,3.035741,3.615681,-4.892330]],[[-2.282442,-5.553541,-2.244570,-1.944699,-5.352383,-0.586431],[-4.181891,9.218505,8.745221,-2.299057,-9.031326,-1.185045],[6.550515,-6.364987,1.902621,7.003418,-0.877872,0.669344]]], dtype = "float64")#candidate|38|(15, 3, 6)|const|float64
uop_39 = relay.log2(const_38.astype('float64')) # shape=(15, 3, 6)
bop_41 = relay.divide(const_38.astype('float64'), relay.reshape(uop_39.astype('float64'), relay.shape_of(const_38))) # shape=(15, 3, 6)
bop_47 = relay.greater(const_38.astype('bool'), relay.reshape(uop_39.astype('bool'), relay.shape_of(const_38))) # shape=(15, 3, 6)
output = relay.Tuple([bop_41,bop_47,])
output2 = relay.Tuple([bop_41,bop_47,])
func_51 = relay.Function([], output)
mod['func_51'] = func_51
mod = relay.transform.InferType()(mod)
output = func_51()
func_52 = relay.Function([], output)
mutated_mod['func_52'] = func_52
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_58 = relay.TupleGetItem(func_51_call(), 1)
call_59 = relay.TupleGetItem(func_52_call(), 1)
var_62 = relay.var("var_62", dtype = "bool", shape = (15, 3, 6))#candidate|62|(15, 3, 6)|var|bool
bop_63 = relay.maximum(call_58.astype('uint32'), relay.reshape(var_62.astype('uint32'), relay.shape_of(call_58))) # shape=(15, 3, 6)
bop_66 = relay.maximum(call_59.astype('uint32'), relay.reshape(var_62.astype('uint32'), relay.shape_of(call_59))) # shape=(15, 3, 6)
output = relay.Tuple([bop_63,])
output2 = relay.Tuple([bop_66,])
func_67 = relay.Function([var_62,], output)
mod['func_67'] = func_67
mod = relay.transform.InferType()(mod)
var_68 = relay.var("var_68", dtype = "bool", shape = (15, 3, 6))#candidate|68|(15, 3, 6)|var|bool
output = func_67(var_68)
func_69 = relay.Function([var_68], output)
mutated_mod['func_69'] = func_69
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_93 = relay.TupleGetItem(func_51_call(), 1)
call_94 = relay.TupleGetItem(func_52_call(), 1)
uop_102 = relay.acos(call_93.astype('float64')) # shape=(15, 3, 6)
uop_104 = relay.acos(call_94.astype('float64')) # shape=(15, 3, 6)
output = relay.Tuple([uop_102,])
output2 = relay.Tuple([uop_104,])
func_108 = relay.Function([], output)
mod['func_108'] = func_108
mod = relay.transform.InferType()(mod)
output = func_108()
func_109 = relay.Function([], output)
mutated_mod['func_109'] = func_109
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_113 = relay.TupleGetItem(func_51_call(), 1)
call_114 = relay.TupleGetItem(func_52_call(), 1)
const_119 = relay.const([[[False,True,False,True,False,False],[False,False,True,True,False,True],[True,True,False,False,True,False]],[[True,True,True,False,True,True],[True,False,False,False,False,False],[False,False,True,False,False,True]],[[True,False,True,False,True,True],[True,False,False,True,True,True],[False,False,False,True,False,False]],[[False,True,False,True,True,True],[True,False,True,True,False,True],[False,True,False,False,False,True]],[[True,True,False,False,True,True],[True,True,True,False,False,True],[False,False,False,False,True,False]],[[True,True,False,True,False,False],[False,False,False,False,False,False],[False,False,True,False,False,False]],[[True,True,True,False,True,True],[True,True,False,True,False,True],[True,True,False,True,False,False]],[[False,True,True,False,True,True],[False,True,True,True,True,True],[False,False,True,True,False,False]],[[True,True,False,True,True,False],[False,False,True,True,False,False],[True,False,True,False,False,False]],[[False,False,False,True,True,True],[True,True,False,False,True,False],[False,False,False,False,True,True]],[[False,True,True,False,False,True],[False,False,True,True,False,True],[True,False,True,True,False,True]],[[True,True,False,True,True,True],[True,True,True,True,False,False],[False,False,False,False,False,False]],[[True,False,False,True,True,True],[False,True,True,False,False,True],[True,True,False,True,False,True]],[[True,False,True,False,True,True],[True,False,True,True,False,False],[True,True,False,False,False,False]],[[True,True,True,False,True,True],[False,False,True,False,True,True],[False,False,True,True,True,True]]], dtype = "bool")#candidate|119|(15, 3, 6)|const|bool
bop_120 = relay.minimum(call_113.astype('uint8'), relay.reshape(const_119.astype('uint8'), relay.shape_of(call_113))) # shape=(15, 3, 6)
bop_123 = relay.minimum(call_114.astype('uint8'), relay.reshape(const_119.astype('uint8'), relay.shape_of(call_114))) # shape=(15, 3, 6)
var_124 = relay.var("var_124", dtype = "uint8", shape = (15, 3, 6))#candidate|124|(15, 3, 6)|var|uint8
bop_125 = relay.bitwise_xor(bop_120.astype('uint16'), relay.reshape(var_124.astype('uint16'), relay.shape_of(bop_120))) # shape=(15, 3, 6)
bop_128 = relay.bitwise_xor(bop_123.astype('uint16'), relay.reshape(var_124.astype('uint16'), relay.shape_of(bop_123))) # shape=(15, 3, 6)
var_130 = relay.var("var_130", dtype = "bool", shape = (15, 3, 6))#candidate|130|(15, 3, 6)|var|bool
bop_131 = relay.subtract(call_113.astype('int8'), relay.reshape(var_130.astype('int8'), relay.shape_of(call_113))) # shape=(15, 3, 6)
bop_134 = relay.subtract(call_114.astype('int8'), relay.reshape(var_130.astype('int8'), relay.shape_of(call_114))) # shape=(15, 3, 6)
uop_136 = relay.erf(var_130.astype('float32')) # shape=(15, 3, 6)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_139 = relay.TupleGetItem(func_51_call(), 0)
call_140 = relay.TupleGetItem(func_52_call(), 0)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_146 = relay.TupleGetItem(func_51_call(), 1)
call_147 = relay.TupleGetItem(func_52_call(), 1)
bop_150 = relay.mod(uop_136.astype('float64'), relay.reshape(bop_131.astype('float64'), relay.shape_of(uop_136))) # shape=(15, 3, 6)
bop_153 = relay.mod(uop_136.astype('float64'), relay.reshape(bop_134.astype('float64'), relay.shape_of(uop_136))) # shape=(15, 3, 6)
bop_162 = relay.greater_equal(bop_150.astype('bool'), relay.reshape(call_139.astype('bool'), relay.shape_of(bop_150))) # shape=(15, 3, 6)
bop_165 = relay.greater_equal(bop_153.astype('bool'), relay.reshape(call_140.astype('bool'), relay.shape_of(bop_153))) # shape=(15, 3, 6)
uop_166 = relay.cosh(bop_150.astype('float32')) # shape=(15, 3, 6)
uop_168 = relay.cosh(bop_153.astype('float32')) # shape=(15, 3, 6)
uop_170 = relay.cos(bop_150.astype('float32')) # shape=(15, 3, 6)
uop_172 = relay.cos(bop_153.astype('float32')) # shape=(15, 3, 6)
bop_173 = relay.power(bop_162.astype('float64'), relay.reshape(call_139.astype('float64'), relay.shape_of(bop_162))) # shape=(15, 3, 6)
bop_176 = relay.power(bop_165.astype('float64'), relay.reshape(call_140.astype('float64'), relay.shape_of(bop_165))) # shape=(15, 3, 6)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_177 = relay.TupleGetItem(func_51_call(), 0)
call_178 = relay.TupleGetItem(func_52_call(), 0)
uop_179 = relay.cos(uop_136.astype('float64')) # shape=(15, 3, 6)
bop_181 = relay.add(uop_179.astype('uint16'), relay.reshape(var_130.astype('uint16'), relay.shape_of(uop_179))) # shape=(15, 3, 6)
bop_186 = relay.logical_xor(bop_150.astype('uint16'), relay.reshape(bop_125.astype('uint16'), relay.shape_of(bop_150))) # shape=(15, 3, 6)
bop_189 = relay.logical_xor(bop_153.astype('uint16'), relay.reshape(bop_128.astype('uint16'), relay.shape_of(bop_153))) # shape=(15, 3, 6)
output = relay.Tuple([call_146,uop_166,uop_170,bop_173,call_177,bop_181,bop_186,])
output2 = relay.Tuple([call_147,uop_168,uop_172,bop_176,call_178,bop_181,bop_189,])
func_198 = relay.Function([var_124,var_130,], output)
mod['func_198'] = func_198
mod = relay.transform.InferType()(mod)
var_199 = relay.var("var_199", dtype = "uint8", shape = (15, 3, 6))#candidate|199|(15, 3, 6)|var|uint8
var_200 = relay.var("var_200", dtype = "bool", shape = (15, 3, 6))#candidate|200|(15, 3, 6)|var|bool
output = func_198(var_199,var_200,)
func_201 = relay.Function([var_199,var_200,], output)
mutated_mod['func_201'] = func_201
mutated_mod = relay.transform.InferType()(mutated_mod)
var_205 = relay.var("var_205", dtype = "uint8", shape = (5, 14))#candidate|205|(5, 14)|var|uint8
var_206 = relay.var("var_206", dtype = "uint8", shape = (5, 14))#candidate|206|(5, 14)|var|uint8
bop_207 = relay.less_equal(var_205.astype('bool'), relay.reshape(var_206.astype('bool'), relay.shape_of(var_205))) # shape=(5, 14)
uop_214 = relay.log(var_206.astype('float32')) # shape=(5, 14)
bop_217 = relay.logical_xor(uop_214.astype('uint16'), relay.reshape(bop_207.astype('uint16'), relay.shape_of(uop_214))) # shape=(5, 14)
output = relay.Tuple([bop_217,])
output2 = relay.Tuple([bop_217,])
func_220 = relay.Function([var_205,var_206,], output)
mod['func_220'] = func_220
mod = relay.transform.InferType()(mod)
mutated_mod['func_220'] = func_220
mutated_mod = relay.transform.InferType()(mutated_mod)
func_220_call = mutated_mod.get_global_var('func_220')
var_222 = relay.var("var_222", dtype = "uint8", shape = (5, 14))#candidate|222|(5, 14)|var|uint8
var_223 = relay.var("var_223", dtype = "uint8", shape = (5, 14))#candidate|223|(5, 14)|var|uint8
call_221 = func_220_call(var_222,var_223,)
output = call_221
func_224 = relay.Function([var_222,var_223,], output)
mutated_mod['func_224'] = func_224
mutated_mod = relay.transform.InferType()(mutated_mod)
const_260 = relay.const([[[-7.461819,2.647219,-2.198454,-2.149873,-3.607591,-1.937591,3.095964,8.228227,-0.606674,-2.472660]],[[-4.670032,1.944305,-3.647059,8.409444,6.624152,3.776227,-7.067236,-0.805611,-4.741695,-7.335558]],[[-0.328515,4.888181,8.826856,5.801809,-9.401974,-3.173855,-3.303740,-0.856323,-8.174955,0.456034]],[[3.868212,0.040865,-9.472527,-9.122760,-9.682788,9.022555,6.354068,3.271390,-7.973653,-1.169327]],[[-5.647084,7.099293,4.090635,9.768268,-0.371271,0.677796,2.526886,3.592652,-6.956318,2.991558]]], dtype = "float64")#candidate|260|(5, 1, 10)|const|float64
uop_261 = relay.atanh(const_260.astype('float64')) # shape=(5, 1, 10)
bop_263 = relay.mod(uop_261.astype('float64'), relay.reshape(const_260.astype('float64'), relay.shape_of(uop_261))) # shape=(5, 1, 10)
uop_270 = relay.log10(uop_261.astype('float64')) # shape=(5, 1, 10)
bop_272 = relay.left_shift(uop_270.astype('uint8'), relay.reshape(uop_261.astype('uint8'), relay.shape_of(uop_270))) # shape=(5, 1, 10)
bop_275 = relay.bitwise_xor(bop_272.astype('uint8'), relay.reshape(uop_261.astype('uint8'), relay.shape_of(bop_272))) # shape=(5, 1, 10)
bop_279 = relay.add(const_260.astype('uint32'), relay.reshape(bop_275.astype('uint32'), relay.shape_of(const_260))) # shape=(5, 1, 10)
bop_282 = relay.floor_mod(uop_261.astype('float32'), relay.reshape(bop_272.astype('float32'), relay.shape_of(uop_261))) # shape=(5, 1, 10)
func_67_call = mod.get_global_var('func_67')
func_69_call = mutated_mod.get_global_var('func_69')
const_287 = relay.const([False,True,True,False,True,True,False,False,False,True,True,True,True,False,True,True,True,True,False,True,False,True,False,False,False,True,False,True,False,False,True,False,True,False,False,False,True,False,False,True,False,False,True,True,False,False,False,True,True,True,False,True,False,False,False,False,True,False,False,False,False,True,False,True,False,True,True,True,True,True,True,True,True,False,True,True,False,True,False,False,False,False,True,False,True,True,True,False,True,True,False,True,False,True,False,False,False,True,False,True,True,True,True,False,True,False,False,True,True,False,True,False,True,True,False,False,False,True,False,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,True,True,False,True,True,False,True,False,False,False,True,False,False,False,False,True,False,False,False,True,True,False,True,True,False,True,False,True,True,True,False,False,False,False,True,True,False,True,False,False,True,True,True,False,True,True,True,True,True,True,True,True,True,False,False,True,True,False,False,True,True,False,True,False,True,True,True,True,False,True,False,True,True,True,True,False,False,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,False,True,False,True,False,False,True,False,True,True,False,False,False,True,True,False,False,True,True,True,False,False,False,True,False,True,False,True,True,False,False,False,False,False,True,False,False,False,True,False,True,True,False,True], dtype = "bool")#candidate|287|(270,)|const|bool
call_286 = relay.TupleGetItem(func_67_call(relay.reshape(const_287.astype('bool'), [15, 3, 6])), 0)
call_288 = relay.TupleGetItem(func_69_call(relay.reshape(const_287.astype('bool'), [15, 3, 6])), 0)
bop_290 = relay.right_shift(bop_263.astype('int64'), relay.reshape(bop_275.astype('int64'), relay.shape_of(bop_263))) # shape=(5, 1, 10)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_293 = relay.TupleGetItem(func_51_call(), 0)
call_294 = relay.TupleGetItem(func_52_call(), 0)
func_198_call = mod.get_global_var('func_198')
func_201_call = mutated_mod.get_global_var('func_201')
call_297 = relay.TupleGetItem(func_198_call(relay.reshape(call_293.astype('uint8'), [15, 3, 6]), relay.reshape(call_293.astype('bool'), [15, 3, 6]), ), 1)
call_298 = relay.TupleGetItem(func_201_call(relay.reshape(call_293.astype('uint8'), [15, 3, 6]), relay.reshape(call_293.astype('bool'), [15, 3, 6]), ), 1)
func_108_call = mod.get_global_var('func_108')
func_109_call = mutated_mod.get_global_var('func_109')
call_302 = relay.TupleGetItem(func_108_call(), 0)
call_303 = relay.TupleGetItem(func_109_call(), 0)
uop_305 = relay.tan(bop_290.astype('float32')) # shape=(5, 1, 10)
var_307 = relay.var("var_307", dtype = "float32", shape = (5, 9, 10))#candidate|307|(5, 9, 10)|var|float32
bop_308 = relay.floor_divide(uop_305.astype('float64'), var_307.astype('float64')) # shape=(5, 9, 10)
bop_311 = relay.less(bop_308.astype('bool'), bop_279.astype('bool')) # shape=(5, 9, 10)
bop_314 = relay.not_equal(uop_305.astype('bool'), relay.reshape(uop_261.astype('bool'), relay.shape_of(uop_305))) # shape=(5, 1, 10)
output = relay.Tuple([bop_282,call_286,const_287,call_293,call_297,call_302,bop_311,bop_314,])
output2 = relay.Tuple([bop_282,call_288,const_287,call_294,call_298,call_303,bop_311,bop_314,])
func_317 = relay.Function([var_307,], output)
mod['func_317'] = func_317
mod = relay.transform.InferType()(mod)
var_318 = relay.var("var_318", dtype = "float32", shape = (5, 9, 10))#candidate|318|(5, 9, 10)|var|float32
output = func_317(var_318)
func_319 = relay.Function([var_318], output)
mutated_mod['func_319'] = func_319
mutated_mod = relay.transform.InferType()(mutated_mod)
var_362 = relay.var("var_362", dtype = "bool", shape = (8,))#candidate|362|(8,)|var|bool
const_363 = relay.const([True,True,False,True,False,False,True,False], dtype = "bool")#candidate|363|(8,)|const|bool
bop_364 = relay.logical_or(var_362.astype('bool'), relay.reshape(const_363.astype('bool'), relay.shape_of(var_362))) # shape=(8,)
output = relay.Tuple([bop_364,])
output2 = relay.Tuple([bop_364,])
func_370 = relay.Function([var_362,], output)
mod['func_370'] = func_370
mod = relay.transform.InferType()(mod)
mutated_mod['func_370'] = func_370
mutated_mod = relay.transform.InferType()(mutated_mod)
var_371 = relay.var("var_371", dtype = "bool", shape = (8,))#candidate|371|(8,)|var|bool
func_370_call = mutated_mod.get_global_var('func_370')
call_372 = func_370_call(var_371)
output = call_372
func_373 = relay.Function([var_371], output)
mutated_mod['func_373'] = func_373
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_375 = relay.TupleGetItem(func_51_call(), 1)
call_376 = relay.TupleGetItem(func_52_call(), 1)
var_379 = relay.var("var_379", dtype = "bool", shape = (15, 3, 6))#candidate|379|(15, 3, 6)|var|bool
bop_380 = relay.less_equal(call_375.astype('bool'), relay.reshape(var_379.astype('bool'), relay.shape_of(call_375))) # shape=(15, 3, 6)
bop_383 = relay.less_equal(call_376.astype('bool'), relay.reshape(var_379.astype('bool'), relay.shape_of(call_376))) # shape=(15, 3, 6)
output = bop_380
output2 = bop_383
func_384 = relay.Function([var_379,], output)
mod['func_384'] = func_384
mod = relay.transform.InferType()(mod)
mutated_mod['func_384'] = func_384
mutated_mod = relay.transform.InferType()(mutated_mod)
var_385 = relay.var("var_385", dtype = "bool", shape = (15, 3, 6))#candidate|385|(15, 3, 6)|var|bool
func_384_call = mutated_mod.get_global_var('func_384')
call_386 = func_384_call(var_385)
output = call_386
func_387 = relay.Function([var_385], output)
mutated_mod['func_387'] = func_387
mutated_mod = relay.transform.InferType()(mutated_mod)
func_108_call = mod.get_global_var('func_108')
func_109_call = mutated_mod.get_global_var('func_109')
call_425 = relay.TupleGetItem(func_108_call(), 0)
call_426 = relay.TupleGetItem(func_109_call(), 0)
const_428 = relay.const([[[-9.368486,6.326622,0.478734,0.361733,-2.846826,-4.264561],[-6.128377,1.926875,-9.093896,-8.921746,-2.248918,-8.552599],[-9.544712,-5.897165,7.202510,-5.746063,-2.192343,1.518472]],[[6.756122,9.406011,-3.296998,-4.852214,6.591095,8.311270],[-5.050191,-7.842518,7.950334,6.229883,4.275034,-3.277259],[-8.523602,6.185794,6.550639,-0.459212,1.232621,-8.635717]],[[-3.008050,-8.729727,-1.418208,2.485680,-4.488646,7.463123],[-4.607147,6.865949,7.322262,-4.716539,-1.856941,-4.653712],[-4.425964,-4.050611,-7.341471,2.023269,-7.080237,5.507854]],[[8.664636,4.939828,-7.918845,3.744593,-9.255347,1.988005],[0.556692,8.042377,9.903874,5.274002,-6.076389,6.699494],[-1.652840,4.791158,2.876150,-6.126302,2.763748,-3.011795]],[[-7.621165,-4.680891,-5.682478,-6.672277,5.086287,-5.965314],[0.688590,-0.443990,9.739695,8.484854,-7.448213,-8.794870],[3.085074,3.329573,0.322963,-2.901879,6.943253,-8.833911]],[[-2.067284,9.465357,-7.190188,-3.982141,2.336007,9.715729],[7.659161,9.597171,-7.370551,0.607577,-1.032355,-8.956212],[-6.669344,7.128599,-1.069893,3.964471,9.828170,-9.447637]],[[-2.906050,5.055042,-9.550691,7.463843,4.330661,7.183873],[-9.577269,3.944961,7.353294,-1.216301,2.123499,-7.224671],[1.346878,-4.347282,-3.947940,-3.523634,-5.675426,3.047611]],[[0.241283,-7.011919,7.470493,-1.790553,-8.741787,-6.426815],[1.184386,4.127370,-6.258639,-8.890427,8.904082,-3.438389],[-5.780946,8.214700,1.384125,1.820489,0.678128,9.842785]],[[8.551850,-8.279329,-2.181016,-1.917857,-6.490901,-3.528081],[1.418289,-2.914453,4.430311,-9.389748,-1.080896,-2.716025],[-7.225049,-2.410383,5.901873,-2.054953,5.570854,-1.812855]],[[9.232608,8.133740,-8.574000,7.385591,7.325518,5.612384],[-9.483259,-9.566231,3.278583,-2.839870,-4.344055,6.838549],[4.038339,-1.970272,-1.691775,-8.273219,0.412311,2.553813]],[[3.183527,2.257688,6.145483,5.374920,7.628016,-5.230705],[2.692433,-5.807721,4.465653,5.652285,-3.689914,8.405598],[3.918310,-1.189857,9.363448,-9.472516,-5.563017,3.138699]],[[-6.304596,-4.554482,-6.774181,-7.960173,8.590752,9.520723],[-6.189462,6.016269,1.423532,-8.801162,-8.002424,2.823777],[-8.242548,-6.109251,-0.275297,-6.490975,4.127838,-2.290154]],[[5.282590,8.529563,8.868457,5.196282,3.539380,-7.248774],[4.907671,7.685227,-0.965618,-9.108098,2.592728,2.795317],[-2.481027,-6.541892,-6.665526,-4.614047,-7.622735,4.008449]],[[-8.192821,4.431296,1.354692,-5.876844,-3.136680,1.782292],[7.496715,-5.314644,-6.737649,-6.931772,9.779990,-1.985121],[4.828421,-4.546458,-2.565014,3.414927,-1.000642,9.529952]],[[0.328877,0.621585,5.320461,-2.658892,-2.346479,6.566464],[-6.752222,9.752520,8.191390,-0.789472,-4.911670,-8.160215],[-8.403738,-4.427454,-3.773360,-6.215737,0.888282,-7.153341]]], dtype = "float64")#candidate|428|(15, 3, 6)|const|float64
bop_429 = relay.less(call_425.astype('bool'), relay.reshape(const_428.astype('bool'), relay.shape_of(call_425))) # shape=(15, 3, 6)
bop_432 = relay.less(call_426.astype('bool'), relay.reshape(const_428.astype('bool'), relay.shape_of(call_426))) # shape=(15, 3, 6)
func_220_call = mod.get_global_var('func_220')
func_224_call = mutated_mod.get_global_var('func_224')
var_434 = relay.var("var_434", dtype = "uint8", shape = (70,))#candidate|434|(70,)|var|uint8
call_433 = relay.TupleGetItem(func_220_call(relay.reshape(var_434.astype('uint8'), [5, 14]), relay.reshape(var_434.astype('uint8'), [5, 14]), ), 0)
call_435 = relay.TupleGetItem(func_224_call(relay.reshape(var_434.astype('uint8'), [5, 14]), relay.reshape(var_434.astype('uint8'), [5, 14]), ), 0)
output = relay.Tuple([bop_429,call_433,var_434,])
output2 = relay.Tuple([bop_432,call_435,var_434,])
func_440 = relay.Function([var_434,], output)
mod['func_440'] = func_440
mod = relay.transform.InferType()(mod)
mutated_mod['func_440'] = func_440
mutated_mod = relay.transform.InferType()(mutated_mod)
var_441 = relay.var("var_441", dtype = "uint8", shape = (70,))#candidate|441|(70,)|var|uint8
func_440_call = mutated_mod.get_global_var('func_440')
call_442 = func_440_call(var_441)
output = call_442
func_443 = relay.Function([var_441], output)
mutated_mod['func_443'] = func_443
mutated_mod = relay.transform.InferType()(mutated_mod)
const_451 = relay.const([[-0.823401,2.222578,-0.461728,-5.819528,-7.234441,0.502233],[-5.973752,-0.761596,5.516678,-1.088462,-0.385781,-9.419237],[-4.501437,-0.262946,-6.872270,9.484886,6.126596,8.183972],[-3.205517,9.788608,-9.548028,-2.218286,-7.405740,9.823149],[8.091782,-0.751052,-0.316228,6.763530,1.957500,-7.295499],[4.233620,2.450985,-1.738896,7.338539,-7.742575,-1.057975],[5.893430,1.421326,9.390907,-9.081423,4.707299,-0.221478],[9.084252,-2.619902,1.549576,-4.677796,0.480170,-6.694403],[-7.997775,-8.450568,-1.676707,2.673081,7.861030,4.430702],[4.178046,8.673135,-2.293915,7.733932,-0.868465,7.546300],[-5.264431,-6.463384,0.506356,3.759365,-6.038796,-6.778002],[-0.750210,4.945989,9.684707,8.433471,-7.853425,7.707358],[-7.497837,-8.934415,-3.837182,2.124073,0.144607,-5.891750],[-1.437846,-2.622589,3.761144,6.990692,-5.102530,9.922686],[-4.912635,6.473132,-8.296273,1.991803,6.448245,8.571387],[3.769279,-1.121698,-1.147250,0.869249,-0.076359,-9.260406]], dtype = "float32")#candidate|451|(16, 6)|const|float32
uop_452 = relay.log2(const_451.astype('float32')) # shape=(16, 6)
bop_457 = relay.mod(const_451.astype('float32'), relay.reshape(uop_452.astype('float32'), relay.shape_of(const_451))) # shape=(16, 6)
output = relay.Tuple([bop_457,])
output2 = relay.Tuple([bop_457,])
func_460 = relay.Function([], output)
mod['func_460'] = func_460
mod = relay.transform.InferType()(mod)
output = func_460()
func_461 = relay.Function([], output)
mutated_mod['func_461'] = func_461
mutated_mod = relay.transform.InferType()(mutated_mod)
const_481 = relay.const([[-0.393594,-4.900080,-0.289191,-1.935533,-9.804994,-7.893868,6.780932,-6.123343,-6.918446],[4.867371,-3.220875,-0.690181,6.902999,9.684858,-6.556191,4.055245,8.293670,-5.536138]], dtype = "float32")#candidate|481|(2, 9)|const|float32
uop_482 = relay.cos(const_481.astype('float32')) # shape=(2, 9)
bop_484 = relay.divide(const_481.astype('float64'), relay.reshape(uop_482.astype('float64'), relay.shape_of(const_481))) # shape=(2, 9)
const_487 = relay.const([[8.723697,4.516620,-4.765286,3.657752,7.857633,4.611493,-4.975851,-7.763004,5.188462],[7.005763,9.679598,2.855223,-9.022193,-6.257756,4.723855,-1.358931,-8.980455,9.765007]], dtype = "float64")#candidate|487|(2, 9)|const|float64
bop_488 = relay.multiply(bop_484.astype('uint16'), relay.reshape(const_487.astype('uint16'), relay.shape_of(bop_484))) # shape=(2, 9)
uop_492 = relay.tan(bop_488.astype('float32')) # shape=(2, 9)
var_496 = relay.var("var_496", dtype = "float32", shape = (2, 9))#candidate|496|(2, 9)|var|float32
bop_497 = relay.logical_or(uop_492.astype('bool'), relay.reshape(var_496.astype('bool'), relay.shape_of(uop_492))) # shape=(2, 9)
uop_500 = relay.log10(uop_482.astype('float64')) # shape=(2, 9)
uop_505 = relay.sigmoid(bop_484.astype('float64')) # shape=(2, 9)
bop_507 = relay.bitwise_and(uop_500.astype('int64'), relay.reshape(bop_488.astype('int64'), relay.shape_of(uop_500))) # shape=(2, 9)
bop_511 = relay.logical_and(bop_507.astype('bool'), relay.reshape(uop_500.astype('bool'), relay.shape_of(bop_507))) # shape=(2, 9)
var_517 = relay.var("var_517", dtype = "float32", shape = (2, 9))#candidate|517|(2, 9)|var|float32
bop_518 = relay.bitwise_xor(uop_482.astype('uint64'), relay.reshape(var_517.astype('uint64'), relay.shape_of(uop_482))) # shape=(2, 9)
uop_521 = relay.exp(bop_511.astype('float32')) # shape=(2, 9)
bop_524 = relay.bitwise_or(uop_521.astype('int32'), relay.reshape(bop_497.astype('int32'), relay.shape_of(uop_521))) # shape=(2, 9)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_528 = relay.TupleGetItem(func_51_call(), 1)
call_529 = relay.TupleGetItem(func_52_call(), 1)
uop_530 = relay.exp(bop_524.astype('float32')) # shape=(2, 9)
output = relay.Tuple([uop_505,bop_518,call_528,uop_530,])
output2 = relay.Tuple([uop_505,bop_518,call_529,uop_530,])
func_532 = relay.Function([var_496,var_517,], output)
mod['func_532'] = func_532
mod = relay.transform.InferType()(mod)
mutated_mod['func_532'] = func_532
mutated_mod = relay.transform.InferType()(mutated_mod)
func_532_call = mutated_mod.get_global_var('func_532')
var_534 = relay.var("var_534", dtype = "float32", shape = (2, 9))#candidate|534|(2, 9)|var|float32
var_535 = relay.var("var_535", dtype = "float32", shape = (2, 9))#candidate|535|(2, 9)|var|float32
call_533 = func_532_call(var_534,var_535,)
output = call_533
func_536 = relay.Function([var_534,var_535,], output)
mutated_mod['func_536'] = func_536
mutated_mod = relay.transform.InferType()(mutated_mod)
const_542 = relay.const([0.745469,9.064549,-4.100441,6.141526,7.985574,3.686179,-1.526384,6.078849,6.932365,-5.873196,1.830998,-1.692640,5.238240,5.907839,-4.724883], dtype = "float64")#candidate|542|(15,)|const|float64
const_543 = relay.const([0.434141,-7.610276,-4.955739,-8.818084,1.841633,8.611231,-6.615146,-1.724745,5.652668,-8.709478,-3.701298,5.788706,3.066440,-3.352877,4.384825], dtype = "float64")#candidate|543|(15,)|const|float64
bop_544 = relay.minimum(const_542.astype('float64'), relay.reshape(const_543.astype('float64'), relay.shape_of(const_542))) # shape=(15,)
output = relay.Tuple([bop_544,])
output2 = relay.Tuple([bop_544,])
func_547 = relay.Function([], output)
mod['func_547'] = func_547
mod = relay.transform.InferType()(mod)
output = func_547()
func_548 = relay.Function([], output)
mutated_mod['func_548'] = func_548
mutated_mod = relay.transform.InferType()(mutated_mod)
var_563 = relay.var("var_563", dtype = "float64", shape = (10, 14, 15))#candidate|563|(10, 14, 15)|var|float64
var_564 = relay.var("var_564", dtype = "float64", shape = (10, 14, 15))#candidate|564|(10, 14, 15)|var|float64
bop_565 = relay.floor_mod(var_563.astype('float64'), relay.reshape(var_564.astype('float64'), relay.shape_of(var_563))) # shape=(10, 14, 15)
output = bop_565
output2 = bop_565
func_568 = relay.Function([var_563,var_564,], output)
mod['func_568'] = func_568
mod = relay.transform.InferType()(mod)
var_569 = relay.var("var_569", dtype = "float64", shape = (10, 14, 15))#candidate|569|(10, 14, 15)|var|float64
var_570 = relay.var("var_570", dtype = "float64", shape = (10, 14, 15))#candidate|570|(10, 14, 15)|var|float64
output = func_568(var_569,var_570,)
func_571 = relay.Function([var_569,var_570,], output)
mutated_mod['func_571'] = func_571
mutated_mod = relay.transform.InferType()(mutated_mod)
const_590 = relay.const([[-2,-5,-6,-2,-9,3,-4,1,10,-6,6,10,-5,7,-1,-5],[5,-3,-8,-1,7,2,3,-7,-3,-1,4,-2,10,10,10,2],[5,6,10,2,-1,1,10,-7,10,-4,3,3,5,5,-9,3],[-6,-7,8,4,-4,3,10,8,2,-4,8,-8,-3,2,-4,5],[-6,4,2,3,-2,3,-10,4,3,-5,-7,7,8,1,-4,1],[9,-6,-3,8,-7,9,-1,7,1,6,7,-8,-3,-6,-7,9],[-2,6,3,-3,2,8,-3,7,7,1,-9,8,10,-10,-2,-3],[-5,-10,-3,-1,-2,-8,3,10,3,5,-2,3,-10,-9,4,-5]], dtype = "int8")#candidate|590|(8, 16)|const|int8
var_591 = relay.var("var_591", dtype = "int8", shape = (8, 16))#candidate|591|(8, 16)|var|int8
bop_592 = relay.bitwise_or(const_590.astype('int8'), relay.reshape(var_591.astype('int8'), relay.shape_of(const_590))) # shape=(8, 16)
output = relay.Tuple([bop_592,])
output2 = relay.Tuple([bop_592,])
func_595 = relay.Function([var_591,], output)
mod['func_595'] = func_595
mod = relay.transform.InferType()(mod)
mutated_mod['func_595'] = func_595
mutated_mod = relay.transform.InferType()(mutated_mod)
var_596 = relay.var("var_596", dtype = "int8", shape = (8, 16))#candidate|596|(8, 16)|var|int8
func_595_call = mutated_mod.get_global_var('func_595')
call_597 = func_595_call(var_596)
output = call_597
func_598 = relay.Function([var_596], output)
mutated_mod['func_598'] = func_598
mutated_mod = relay.transform.InferType()(mutated_mod)
func_108_call = mod.get_global_var('func_108')
func_109_call = mutated_mod.get_global_var('func_109')
call_621 = relay.TupleGetItem(func_108_call(), 0)
call_622 = relay.TupleGetItem(func_109_call(), 0)
func_547_call = mod.get_global_var('func_547')
func_548_call = mutated_mod.get_global_var('func_548')
call_623 = relay.TupleGetItem(func_547_call(), 0)
call_624 = relay.TupleGetItem(func_548_call(), 0)
func_220_call = mod.get_global_var('func_220')
func_224_call = mutated_mod.get_global_var('func_224')
var_630 = relay.var("var_630", dtype = "uint8", shape = (7, 10))#candidate|630|(7, 10)|var|uint8
call_629 = relay.TupleGetItem(func_220_call(relay.reshape(var_630.astype('uint8'), [5, 14]), relay.reshape(var_630.astype('uint8'), [5, 14]), ), 0)
call_631 = relay.TupleGetItem(func_224_call(relay.reshape(var_630.astype('uint8'), [5, 14]), relay.reshape(var_630.astype('uint8'), [5, 14]), ), 0)
bop_638 = relay.logical_and(call_629.astype('bool'), relay.reshape(var_630.astype('bool'), relay.shape_of(call_629))) # shape=(5, 14)
bop_641 = relay.logical_and(call_631.astype('bool'), relay.reshape(var_630.astype('bool'), relay.shape_of(call_631))) # shape=(5, 14)
output = relay.Tuple([call_621,call_623,bop_638,])
output2 = relay.Tuple([call_622,call_624,bop_641,])
func_642 = relay.Function([var_630,], output)
mod['func_642'] = func_642
mod = relay.transform.InferType()(mod)
var_643 = relay.var("var_643", dtype = "uint8", shape = (7, 10))#candidate|643|(7, 10)|var|uint8
output = func_642(var_643)
func_644 = relay.Function([var_643], output)
mutated_mod['func_644'] = func_644
mutated_mod = relay.transform.InferType()(mutated_mod)
const_658 = relay.const([[[-8.577237,-2.676520,3.633116,4.072047,6.743605,2.594789,5.471718,-1.165021,6.578458,-7.364448,4.236825,8.020432,8.291662,9.641316]]], dtype = "float32")#candidate|658|(1, 1, 14)|const|float32
uop_659 = relay.sqrt(const_658.astype('float32')) # shape=(1, 1, 14)
bop_661 = relay.bitwise_or(const_658.astype('int8'), relay.reshape(uop_659.astype('int8'), relay.shape_of(const_658))) # shape=(1, 1, 14)
output = relay.Tuple([bop_661,])
output2 = relay.Tuple([bop_661,])
func_665 = relay.Function([], output)
mod['func_665'] = func_665
mod = relay.transform.InferType()(mod)
mutated_mod['func_665'] = func_665
mutated_mod = relay.transform.InferType()(mutated_mod)
func_665_call = mutated_mod.get_global_var('func_665')
call_666 = func_665_call()
output = call_666
func_667 = relay.Function([], output)
mutated_mod['func_667'] = func_667
mutated_mod = relay.transform.InferType()(mutated_mod)
const_668 = relay.const([[-2.325336],[-0.496559],[-9.344072],[4.845085],[7.145676],[8.082039],[0.054663],[7.777349],[2.630665],[-5.585637],[-5.474007],[1.141786],[4.154165],[7.363039],[-4.780107],[3.166705]], dtype = "float64")#candidate|668|(16, 1)|const|float64
var_669 = relay.var("var_669", dtype = "float64", shape = (16, 8))#candidate|669|(16, 8)|var|float64
bop_670 = relay.power(const_668.astype('float64'), var_669.astype('float64')) # shape=(16, 8)
bop_676 = relay.floor_divide(const_668.astype('float32'), bop_670.astype('float32')) # shape=(16, 8)
uop_683 = relay.sigmoid(bop_676.astype('float32')) # shape=(16, 8)
func_547_call = mod.get_global_var('func_547')
func_548_call = mutated_mod.get_global_var('func_548')
call_685 = relay.TupleGetItem(func_547_call(), 0)
call_686 = relay.TupleGetItem(func_548_call(), 0)
uop_688 = relay.atanh(uop_683.astype('float32')) # shape=(16, 8)
uop_692 = relay.log2(uop_688.astype('float64')) # shape=(16, 8)
func_384_call = mod.get_global_var('func_384')
func_387_call = mutated_mod.get_global_var('func_387')
var_695 = relay.var("var_695", dtype = "bool", shape = (270,))#candidate|695|(270,)|var|bool
call_694 = func_384_call(relay.reshape(var_695.astype('bool'), [15, 3, 6]))
call_696 = func_384_call(relay.reshape(var_695.astype('bool'), [15, 3, 6]))
bop_698 = relay.logical_xor(uop_683.astype('uint8'), relay.reshape(uop_688.astype('uint8'), relay.shape_of(uop_683))) # shape=(16, 8)
uop_703 = relay.atanh(uop_692.astype('float32')) # shape=(16, 8)
bop_705 = relay.logical_or(uop_692.astype('bool'), relay.reshape(uop_683.astype('bool'), relay.shape_of(uop_692))) # shape=(16, 8)
uop_710 = relay.exp(uop_703.astype('float32')) # shape=(16, 8)
uop_712 = relay.atan(bop_698.astype('float64')) # shape=(16, 8)
bop_716 = relay.bitwise_and(uop_710.astype('uint32'), relay.reshape(bop_705.astype('uint32'), relay.shape_of(uop_710))) # shape=(16, 8)
bop_719 = relay.logical_or(uop_703.astype('bool'), relay.reshape(bop_676.astype('bool'), relay.shape_of(uop_703))) # shape=(16, 8)
func_440_call = mod.get_global_var('func_440')
func_443_call = mutated_mod.get_global_var('func_443')
var_730 = relay.var("var_730", dtype = "uint8", shape = (70,))#candidate|730|(70,)|var|uint8
call_729 = relay.TupleGetItem(func_440_call(relay.reshape(var_730.astype('uint8'), [70,])), 1)
call_731 = relay.TupleGetItem(func_443_call(relay.reshape(var_730.astype('uint8'), [70,])), 1)
uop_732 = relay.rsqrt(bop_719.astype('float32')) # shape=(16, 8)
output = relay.Tuple([call_685,call_694,var_695,uop_712,bop_716,call_729,var_730,uop_732,])
output2 = relay.Tuple([call_686,call_696,var_695,uop_712,bop_716,call_731,var_730,uop_732,])
func_736 = relay.Function([var_669,var_695,var_730,], output)
mod['func_736'] = func_736
mod = relay.transform.InferType()(mod)
mutated_mod['func_736'] = func_736
mutated_mod = relay.transform.InferType()(mutated_mod)
func_736_call = mutated_mod.get_global_var('func_736')
var_738 = relay.var("var_738", dtype = "float64", shape = (16, 8))#candidate|738|(16, 8)|var|float64
var_739 = relay.var("var_739", dtype = "bool", shape = (270,))#candidate|739|(270,)|var|bool
var_740 = relay.var("var_740", dtype = "uint8", shape = (70,))#candidate|740|(70,)|var|uint8
call_737 = func_736_call(var_738,var_739,var_740,)
output = call_737
func_741 = relay.Function([var_738,var_739,var_740,], output)
mutated_mod['func_741'] = func_741
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_751 = relay.TupleGetItem(func_51_call(), 0)
call_752 = relay.TupleGetItem(func_52_call(), 0)
var_753 = relay.var("var_753", dtype = "float64", shape = (15, 3, 6))#candidate|753|(15, 3, 6)|var|float64
bop_754 = relay.not_equal(call_751.astype('bool'), relay.reshape(var_753.astype('bool'), relay.shape_of(call_751))) # shape=(15, 3, 6)
bop_757 = relay.not_equal(call_752.astype('bool'), relay.reshape(var_753.astype('bool'), relay.shape_of(call_752))) # shape=(15, 3, 6)
bop_763 = relay.multiply(bop_754.astype('int8'), relay.reshape(call_751.astype('int8'), relay.shape_of(bop_754))) # shape=(15, 3, 6)
bop_766 = relay.multiply(bop_757.astype('int8'), relay.reshape(call_752.astype('int8'), relay.shape_of(bop_757))) # shape=(15, 3, 6)
bop_767 = relay.subtract(bop_754.astype('int8'), relay.reshape(var_753.astype('int8'), relay.shape_of(bop_754))) # shape=(15, 3, 6)
bop_770 = relay.subtract(bop_757.astype('int8'), relay.reshape(var_753.astype('int8'), relay.shape_of(bop_757))) # shape=(15, 3, 6)
bop_772 = relay.right_shift(bop_763.astype('int16'), relay.reshape(call_751.astype('int16'), relay.shape_of(bop_763))) # shape=(15, 3, 6)
bop_775 = relay.right_shift(bop_766.astype('int16'), relay.reshape(call_752.astype('int16'), relay.shape_of(bop_766))) # shape=(15, 3, 6)
bop_776 = relay.right_shift(bop_767.astype('int32'), relay.reshape(bop_772.astype('int32'), relay.shape_of(bop_767))) # shape=(15, 3, 6)
bop_779 = relay.right_shift(bop_770.astype('int32'), relay.reshape(bop_775.astype('int32'), relay.shape_of(bop_770))) # shape=(15, 3, 6)
uop_780 = relay.exp(bop_776.astype('float64')) # shape=(15, 3, 6)
uop_782 = relay.exp(bop_779.astype('float64')) # shape=(15, 3, 6)
bop_785 = relay.logical_and(uop_780.astype('bool'), relay.reshape(var_753.astype('bool'), relay.shape_of(uop_780))) # shape=(15, 3, 6)
bop_788 = relay.logical_and(uop_782.astype('bool'), relay.reshape(var_753.astype('bool'), relay.shape_of(uop_782))) # shape=(15, 3, 6)
const_789 = relay.const([[[-8.062130,-7.947480,-6.939002,-9.791183,-3.006263,0.425127],[8.683791,-7.346387,5.146683,8.672001,-8.261553,-6.594099],[7.356251,-6.740058,-1.170703,0.319439,4.606842,8.312019]],[[9.013218,-4.170686,-7.478854,2.660891,9.682272,8.166725],[5.608864,8.744286,6.398370,5.256396,-6.556752,-8.044969],[6.337630,6.796642,4.295359,-8.946070,-1.161126,0.951967]],[[-1.727971,3.525955,-2.247339,-5.957992,9.206336,-9.024010],[-9.252552,8.063390,1.850404,1.882618,-4.988424,-9.408166],[1.301378,-0.753631,4.550844,-5.075645,-8.406248,1.325256]],[[-5.393745,-4.838513,-3.304801,5.092680,-4.501308,3.910647],[-7.399108,5.076099,3.572522,-4.632468,-5.217084,-1.243323],[-3.226308,-0.711554,-1.672013,-0.552154,3.488355,-6.016222]],[[8.550184,-9.476530,9.446565,9.967284,-5.056932,-3.156474],[-3.080397,-1.371649,6.852194,8.264852,9.549833,0.013656],[9.401366,4.230263,-3.539583,-0.696977,6.561268,-5.549656]],[[-7.210317,-0.802995,-7.929401,-7.538592,3.651982,2.466561],[-7.759882,1.030431,-8.306831,1.370999,-8.773025,3.228691],[4.865479,-1.649189,-0.562060,6.720322,-1.612708,-9.050987]],[[-0.278198,-5.027927,1.538099,-9.705233,-0.191075,-9.244418],[-4.500680,-3.847607,-8.471210,-5.768514,-0.156391,-2.054886],[8.709554,5.276553,8.565901,-2.764233,7.883106,1.528289]],[[-9.972340,-0.946474,6.410049,-3.696718,-0.246630,-7.524306],[-7.294588,-3.733873,5.130483,5.001965,-8.595387,5.398310],[-9.708543,0.996921,-3.371962,2.291068,1.775699,9.798892]],[[-3.374179,-2.956852,-3.939030,3.106447,3.033558,2.139074],[-6.397675,5.141444,-3.445285,-1.414971,-7.941131,-5.081776],[-0.872718,-5.928346,4.816236,2.578570,2.627782,-9.111418]],[[-8.037526,-1.562170,-5.171981,0.798648,-3.476385,1.962685],[3.728907,-4.638138,-7.166480,-8.959475,-7.942082,-3.099793],[-7.920742,7.967562,7.231146,8.020570,9.696087,-4.783663]],[[-0.065696,-1.492232,-0.840020,0.051162,7.506683,-2.772250],[-8.439980,-1.336103,0.390845,-9.005327,-6.980723,7.644229],[8.447182,-5.451439,-4.339260,-0.356780,-8.355920,4.176697]],[[2.578577,-2.925782,-5.577515,-0.409445,3.748291,1.196758],[0.789159,-2.695397,2.683175,4.830882,-9.125941,6.569162],[-7.433587,5.150084,-7.724263,-1.437616,4.327725,-0.124907]],[[4.241344,-6.215620,-0.063396,1.877632,7.401058,1.359433],[0.988720,-0.525544,-8.901010,-2.274665,2.709950,0.934950],[-0.799363,9.167505,-6.536583,-7.072334,-7.035892,-9.387087]],[[-6.878485,-6.554118,7.488780,4.972548,7.284556,2.383579],[1.383325,-2.399606,9.059233,5.170449,0.480891,-1.946350],[-7.091994,3.442474,5.325900,9.406395,9.195754,-7.599213]],[[-3.747120,6.663561,2.389358,3.645622,-0.997803,1.982267],[-5.430731,-5.935570,-4.550693,0.565292,5.616942,4.314674],[4.272300,-7.612622,6.898614,1.576330,-4.581380,2.097763]]], dtype = "float64")#candidate|789|(15, 3, 6)|const|float64
bop_790 = relay.minimum(uop_780.astype('int16'), relay.reshape(const_789.astype('int16'), relay.shape_of(uop_780))) # shape=(15, 3, 6)
bop_793 = relay.minimum(uop_782.astype('int16'), relay.reshape(const_789.astype('int16'), relay.shape_of(uop_782))) # shape=(15, 3, 6)
uop_797 = relay.rsqrt(bop_790.astype('float32')) # shape=(15, 3, 6)
uop_799 = relay.rsqrt(bop_793.astype('float32')) # shape=(15, 3, 6)
uop_813 = relay.tan(uop_797.astype('float32')) # shape=(15, 3, 6)
uop_815 = relay.tan(uop_799.astype('float32')) # shape=(15, 3, 6)
func_568_call = mod.get_global_var('func_568')
func_571_call = mutated_mod.get_global_var('func_571')
const_818 = relay.const([2.479733,-8.020892,-9.803743,-6.455838,-7.992793,4.068950,-5.999315,7.087416,0.519162,7.623509,5.182204,5.240400,5.015913,-7.636311,-9.322669,-7.106735,-8.606801,-4.409041,2.841899,7.025097,-3.943099,0.551752,6.509006,-7.888280,-6.000968,-7.762580,-4.508643,-9.382459,-6.128804,6.292718,-3.606217,-3.730319,-6.437325,9.681046,5.875989,8.965046,1.207850,-9.697712,0.567344,8.741120,0.464132,9.352050,5.648806,4.663304,-3.734219,-1.199890,8.538405,-3.811767,8.196023,1.042668,-8.112733,-8.526638,-9.540307,-8.397296,8.185011,3.706043,8.428261,1.018950,-2.823062,-9.970425,7.782890,4.506245,2.735542,-3.667747,7.800525,-3.466571,-6.957497,7.339062,7.957564,-2.748316,0.400502,6.168212,-9.746149,0.103890,-9.758795,1.938109,-1.369903,-5.631262,-7.920282,-8.069308,-5.321389,8.427724,-2.092537,0.185879,-8.368671,-9.249552,8.439386,-9.517116,-0.668481,-5.746991,2.593672,-7.230119,-8.229548,-3.049542,5.229654,-6.833725,2.573894,-5.555779,-8.977083,0.250414,3.297852,-8.612427,-4.113958,-3.480630,-5.154337,8.155648,8.691844,7.356468,-5.737388,-4.091535,0.412649,1.705563,7.349953,-5.516965,-2.714452,-8.242424,3.959286,-0.833681,-6.333859,6.561447,-9.360228,-2.640977,-2.834765,5.168247,2.293789,7.698202,2.610970,2.141939,-3.006500,-9.804592,-3.905726,7.517585,-4.115997,5.520559,-7.325323,4.972860,1.303555,-9.892067,-8.929989,-4.646544,-5.500917,-7.838313,-7.813463,-3.694298,6.666319,1.628835,3.833948,6.588415,7.968994,-5.468066,5.622833,-0.183824,-3.965383,3.186999,-3.093673,-4.891418,1.223641,-0.049935,6.663861,-3.223878,-7.267862,-7.146987,-4.335771,7.455879,-7.637866,-7.691707,9.620723,-0.089825,2.415938,-4.167054,5.131634,-3.708716,4.269138,-3.606053,9.043157,4.502297,9.713042,-7.219001,-8.112525,8.993610,9.887748,-2.879246,-0.565664,-5.424886,-2.341827,7.646362,-9.649379,-6.978203,2.330617,-3.296244,-4.127658,-4.719111,5.103681,6.295935,-6.680758,-9.629138,-6.351653,-0.044967,3.659914,2.167088,-9.745397,8.858696,-5.091825,-6.222302,-3.275219,7.584124,4.199254,-7.855005,-3.749681,-0.523066,-4.086163,-6.597847,1.215789,4.937764,-4.690620,5.396415,-1.737465,7.416757,-1.419030,-0.988048,-5.152663,0.073114,8.177096,-2.131538,-1.960356,-9.832438,4.934379,8.175259,2.471323,7.567720,8.686159,-0.629443,0.357324,8.988630,-6.993059,-9.560703,2.084547,-7.971837,-5.801710,-3.798263,4.971809,5.725117,8.319883,8.755491,-4.189098,-6.213210,0.324488,4.864690,-6.258842,1.663765,-3.417665,8.724930,3.023273,-5.661872,-3.094215,-8.532800,7.243986,-0.405405,6.875277,1.776081,-8.684042,2.910707,4.128552,-3.201361,7.512359,-4.169631,-4.240783,4.501485,7.049934,7.587604,-0.403291,0.448306,7.362592,0.630215,4.480326,-1.910264,-8.720258,-2.877509,-6.820912,-0.941189,-8.751046,9.532044,-5.520428,-6.675782,-6.714774,-4.059025,-4.824084,-0.533568,5.036305,-2.652907,5.053530,0.191857,-8.704090,4.410919,-4.104013,3.941258,-3.622365,6.475911,0.619710,-4.077035,9.766875,7.638310,-9.170466,-7.249930,0.118104,-2.612317,6.345938,-5.867464,2.758520,3.162034,4.715824,-1.426860,0.887235,9.469085,-6.594568,4.239476,-2.710004,7.313890,-6.912342,-0.755801,-7.407784,-7.673089,-0.676315,4.301332,-8.676988,-0.958201,2.932902,6.988437,-3.100233,-5.723591,-8.021033,3.213040,-4.804834,6.654571,-6.318749,-1.163526,3.925852,-7.602105,8.312804,0.988198,-9.706034,-1.839052,7.635890,6.134963,-1.465032,-6.807338,2.318321,3.586112,-0.029060,-4.302864,-8.973810,6.545533,-6.125383,-7.169558,1.195672,9.531999,1.057764,-5.084209,-1.743056,-3.135100,5.742307,7.968724,-3.679173,-7.149491,-8.723370,5.718595,1.048455,-3.433509,2.978963,8.597049,-6.041373,-0.521618,-6.939830,-1.310746,-3.384024,8.218628,5.421834,-0.386537,6.101353,-7.262052,-8.189123,2.173660,-7.266505,5.603777,-8.467885,8.217137,0.981275,-2.061828,-6.086341,-2.582226,7.296996,-4.997472,3.242477,0.656672,3.151418,-0.302118,6.303322,2.477224,8.090713,7.003288,1.742651,-7.267476,4.120582,4.983829,-4.765290,-3.466835,6.245504,-4.868288,-6.862926,-4.043925,-4.531689,4.214562,-2.581211,7.520252,4.944795,3.643325,7.894181,1.996879,-1.332537,-6.098459,3.763860,-6.773109,0.657091,4.943515,-9.174012,0.635913,-8.253030,3.183848,5.014828,-5.785814,7.364363,-0.875106,0.166602,5.430854,-3.186689,-1.690818,2.565419,-2.338876,-6.328057,-3.098871,7.939353,9.341632,-5.899688,-7.465629,2.436660,-9.086259,-6.071637,-5.700169,-1.992557,9.073705,1.717089,-8.877182,3.900143,-2.905532,-0.358700,-0.662219,-8.123821,1.089402,-6.314577,1.259201,-3.958555,8.527182,-0.487879,5.467954,8.543429,5.235293,-2.873471,2.851722,-2.439169,5.449901,-8.414646,5.423484,5.479716,9.999463,-0.581511,-1.006598,-5.089797,-0.681212,3.775690,-9.205093,-3.808976,9.445981,7.982442,-0.986257,-3.490353,6.763299,-0.643058,-3.535384,2.643142,7.768926,-7.834318,0.594985,4.602696,4.016188,6.794643,-0.302250,0.508052,-1.078557,-5.936685,-1.206007,4.702535,-0.841532,-3.166788,2.912785,-0.345121,-2.629271,-8.054012,-0.459309,6.986229,-1.463384,3.096319,-5.470580,4.072883,5.329801,-2.095381,-9.854064,1.805317,3.501780,4.761450,-3.145115,2.328536,-4.286638,-9.290985,-2.155268,-0.647015,-2.229236,1.392192,-8.677941,8.864577,-0.703036,-6.269019,2.933445,2.780465,0.845095,-1.143363,2.562482,-0.201116,9.027427,-2.697001,-3.681264,9.394227,-7.927728,-1.342736,6.571946,-4.651870,0.719459,-3.405477,5.451367,7.257152,1.389890,-7.336607,0.707624,-9.372585,8.213116,-6.444845,-6.148732,-3.825901,-1.362289,9.205957,-6.061215,-6.771744,-6.373108,3.779690,-7.782548,9.753354,-9.827269,-3.404172,-7.794265,-6.452163,1.424820,-5.944762,-6.868370,-4.677274,6.287212,-1.454711,-0.373551,0.408714,-2.056907,-6.128268,1.232346,-8.524313,7.591107,3.737283,3.167184,9.386173,2.789313,-9.598865,-4.008271,-8.842509,0.047467,8.256386,9.883634,-9.193060,3.577881,7.720219,-5.753378,-7.801063,-0.395871,-5.856322,-3.433492,8.419272,2.593237,2.227626,-0.819415,6.142054,5.014844,8.892382,3.896951,-1.337193,-6.338931,2.540507,4.195471,-2.809221,5.206277,3.356981,-7.843792,4.980167,2.788065,4.457145,5.318077,6.645858,-6.891782,1.052629,-4.473350,-8.584429,8.127348,-9.096278,-5.872767,5.540289,4.256020,1.304199,2.805430,7.717606,1.528126,-2.536798,9.542233,0.912000,3.639267,1.135397,-2.881717,-3.926606,8.293390,-6.067103,-0.489253,4.111000,6.646939,6.203510,-1.578613,0.162567,-1.972526,-4.559064,0.082020,2.897891,-6.440326,-5.370748,-4.589816,8.965674,6.592198,4.497271,9.697267,6.579723,6.833735,7.397317,5.147463,-8.405884,-0.241090,3.822652,-6.867797,-0.492049,8.821964,-0.253855,-3.613152,7.035493,4.246202,8.895314,-8.802276,1.545573,2.391877,6.141205,9.256608,-9.314695,-2.789922,3.298532,-5.162608,1.325350,0.687125,-9.306075,5.585843,9.172602,-6.299389,2.724711,-7.817150,0.825862,0.433271,-3.890593,-6.523412,9.179634,5.766409,3.703304,-9.521927,0.159322,-1.994718,6.128549,-0.209818,5.550892,5.652598,7.178817,5.736129,-6.922498,5.805369,-1.232188,6.770926,-0.472336,-2.940208,5.445811,-2.020875,-9.981912,-3.264790,3.961496,5.161837,2.910272,5.095594,-1.881121,6.297098,-3.909077,7.699164,-6.593262,-0.239221,8.561202,2.685554,9.426022,-5.344654,-6.036904,-6.738578,2.650630,-3.019827,-7.914653,3.103576,3.678174,8.830603,6.625604,-5.428128,-5.497766,-4.828924,4.645849,-1.813629,5.035606,6.324859,1.315085,2.853010,0.035476,1.990932,-4.179625,-7.915723,-9.563319,5.472530,-7.633228,1.290401,-5.184229,5.634300,8.208690,-7.562773,-0.901972,3.199393,3.823351,-4.833063,-1.143919,4.723637,-7.825887,-2.937253,9.306464,-0.673153,0.635442,-0.853715,8.753161,-3.132371,4.611970,0.933674,-6.994115,3.127404,-7.021619,-0.887782,-7.819141,8.725020,9.458737,0.713989,8.759341,-7.926730,1.052222,-9.162329,-4.991473,-8.580644,1.929847,9.786862,-3.151353,-2.896260,-2.170998,7.069065,-2.225927,1.205908,0.409018,-9.898479,-4.116653,8.006888,-5.085899,-1.066614,-0.294002,-9.290641,4.341718,7.921492,0.237323,-6.756346,-7.143811,-4.099236,-6.319922,-8.333807,-7.776754,-9.254769,-3.665261,-1.474888,6.261374,2.834030,8.231244,9.553277,-3.067809,-5.582175,-2.562388,2.177327,5.909247,3.653737,-2.600767,0.413472,5.373438,1.988055,0.689965,-8.252613,5.155893,4.195886,6.137973,3.870562,3.259465,8.338966,-4.837661,9.521270,-5.304406,9.250677,-0.912669,5.515562,-8.176495,6.564668,3.341673,-9.035425,-6.200061,1.330797,-4.120789,6.961027,2.674496,-1.870810,9.095220,6.038962,-7.295839,-9.649577,9.185600,-1.232071,4.732113,8.786888,2.217498,6.488812,8.890884,-1.509790,7.908813,-1.896204,-9.347286,-2.308944,-8.920568,-9.895248,-1.941803,9.027507,-8.117443,9.450748,-6.166882,-3.500409,-7.606444,7.505427,7.772654,1.980008,0.525833,1.578622,-8.889162,0.490440,9.799299,-6.180046,5.466491,1.697066,8.099826,1.571919,4.187378,-2.016918,7.492187,-2.722573,6.157858,6.816861,6.194931,-7.386169,-9.995172,-8.281791,0.541992,2.143807,-8.729865,7.757882,4.668157,0.908204,-0.971679,-8.599064,9.081072,-0.147470,-1.581707,0.105606,-9.668915,-3.616596,-2.508862,-9.931542,2.370179,-6.720895,-6.012598,0.951561,-9.288567,2.634951,-9.920955,-6.480609,-2.485362,-8.095232,-8.608416,-3.148774,-4.389827,0.034421,8.307756,-5.648396,4.606150,-9.281391,-4.505815,1.938367,4.279620,-3.606517,-6.018766,2.662133,0.801163,0.682988,-7.245251,3.857323,-0.661571,8.833741,-1.148982,-9.616191,-7.674736,-8.919903,0.612586,9.350481,-9.181095,-9.065355,-6.048991,6.624819,-4.404270,6.594162,-8.718072,7.505885,0.643381,-6.433734,-5.299377,4.220587,7.438358,-4.785466,2.057812,0.065243,1.470492,-5.887810,-3.021364,2.996867,-9.977299,8.883211,3.781220,-0.917112,0.404546,7.666169,0.260919,-3.315184,-5.736161,1.642816,-3.451900,0.153907,9.941528,2.827762,3.054933,-1.172081,0.762991,-0.966522,-8.669557,8.490494,7.686261,0.537717,4.451076,2.158600,-4.725641,1.922463,-3.570913,-9.315082,-9.645869,9.650816,8.548451,5.065022,5.036153,4.892904,-7.060244,0.009195,1.120876,8.964790,-2.124996,-3.058167,1.284219,3.026314,-8.574222,6.399030,3.704268,-3.074883,-5.953966,9.600126,7.493824,3.371167,5.971778,6.258013,-5.399382,5.411488,-1.566853,2.261294,8.583699,0.396856,9.737094,-7.891467,3.232798,9.597258,-6.191287,8.424342,1.551244,-6.887611,7.921698,-8.279836,-6.339619,-1.439361,4.211955,-9.882479,-9.177809,0.038810,0.500836,-2.464506,-2.462355,-6.843609,-8.311912,8.518972,5.925278,-4.067045,1.125336,-3.271734,-4.913984,-8.729948,-5.033493,-6.741126,-6.111499,-6.136979,4.415868,-3.655957,-9.464993,-9.999943,-4.922723,4.010000,8.734010,-9.284669,-0.592784,-9.311660,0.313009,-3.278050,-1.955956,-7.802367,-1.230586,-6.317775,0.909825,-5.385931,5.848762,-3.066234,-4.673556,0.316559,5.131579,-2.872633,-4.846352,-6.082104,-6.926560,-1.522876,5.865848,0.015940,8.746320,6.841996,7.155960,3.295520,6.505683,-6.113251,9.419703,7.734223,-8.944505,-2.328564,0.026732,-2.955269,3.315091,-1.268171,-4.585862,1.632737,9.874732,-7.040581,-2.059152,9.427602,-6.881965,7.253161,-8.682987,-9.252907,-3.160519,7.666552,-1.824558,3.106321,-5.018015,-3.667736,-7.003696,3.699362,5.095410,-6.434339,8.249483,9.696431,5.847927,4.466928,3.428936,-7.482766,-7.142436,4.468661,-3.226291,-2.636666,-4.672820,6.476577,-6.522581,-1.777004,-4.408850,4.736020,7.130975,7.981906,6.281507,-0.587418,4.970646,4.354681,-1.415369,-2.843650,1.957927,-3.815931,5.347158,-2.605909,-8.202978,1.603732,2.358507,-9.604049,3.329173,-1.674514,-3.361493,-6.309077,5.600741,9.657237,-9.547773,-5.574411,-5.581906,-2.775034,-5.707031,-2.860835,7.536540,-1.526521,-7.049914,7.314622,-6.016037,6.203048,-5.461934,-1.500832,2.602180,-1.337005,8.748297,2.260664,2.624368,-8.807233,0.541105,-9.081761,-5.211846,1.209244,2.988669,2.872752,-7.710974,-5.409966,-4.616610,5.898282,-3.339064,4.805634,-3.249814,0.926442,-4.576367,-5.991226,-9.669117,8.802936,2.459111,-6.398553,1.884226,-6.907397,-3.315722,7.650449,-3.812111,-1.901613,-0.263407,-8.687094,-4.995527,-0.103988,5.627393,-3.505716,3.629513,8.859149,4.530526,6.620769,7.714721,-1.324473,-5.579212,-2.603730,-5.949162,1.760102,-8.231080,-4.833757,0.364323,-6.814132,7.427892,3.921718,-4.092067,3.407538,-1.776428,-7.856246,-4.318351,9.916957,-6.851789,2.645045,-1.122607,5.327323,8.494558,3.636087,-2.571429,8.011591,0.975367,-8.956645,0.292539,1.940789,-5.252630,2.654705,-2.937124,8.436476,-3.251573,-1.569203,3.771330,-7.488283,7.232857,-2.425679,1.851630,0.490086,-3.707082,-7.684595,9.081698,9.513124,1.625153,-8.038863,9.056631,1.621568,8.436974,-9.608801,8.043050,8.926470,-3.160065,3.130755,0.349405,-0.548554,-5.193060,-7.834858,-6.795390,-6.901200,-9.848691,-8.977444,-9.371179,-8.951860,8.951545,7.202058,5.400750,8.355858,-7.992969,5.378747,8.421549,2.034063,-7.222150,9.464985,0.268933,-4.946038,-6.959658,1.251006,-1.314648,-6.644208,-5.153910,0.791942,-1.878810,6.326683,7.696756,5.594182,5.887340,-5.898985,-1.517652,-4.593016,8.294276,5.121694,8.146151,8.087829,-0.776838,6.481852,0.100959,-2.416856,3.202415,-6.827581,-9.828504,1.812977,-7.899600,9.235950,6.644119,-6.768753,2.604408,1.074143,-0.837275,-2.822887,-7.049979,7.002700,-4.918617,-2.468983,5.934231,-0.653484,-7.563474,3.283546,-5.040744,7.590890,7.816655,-9.686759,9.692371,4.901544,-6.629463,5.508887,-2.437843,-2.214590,-0.216210,-3.203748,9.495722,-3.977424,9.008349,-1.616200,-9.188781,0.874124,0.370303,-7.594342,7.983194,-7.579532,-2.443993,5.078735,-0.448285,9.477115,-3.739994,6.544877,8.257057,0.755711,2.107036,-6.427202,0.892187,9.662830,5.726225,0.092975,-5.157724,7.597657,8.942511,2.448560,-9.526468,9.975258,-1.050125,8.768098,-6.534376,-5.896845,-9.196007,9.596148,8.166086,5.466471,5.794002,3.010988,-5.627809,-5.694574,0.380130,-6.442154,-6.316338,-4.102955,-7.824044,-5.704188,-3.942650,-1.139605,4.844575,-7.342436,1.938742,-9.713077,9.332166,-5.458270,-8.974184,7.407046,-8.991092,7.089909,6.171207,0.482975,2.321877,-0.260296,8.909735,9.763250,8.947787,8.565162,0.465920,0.380366,2.080377,-9.265307,-0.412105,-6.108282,7.329725,8.079534,3.608193,-4.454978,8.095628,8.321142,-0.971472,4.773434,-7.499869,-3.057254,-7.839050,-6.532869,2.212329,2.759488,6.132367,-3.692887,-1.170204,6.300174,-3.628521,-1.790882,9.141190,-1.185367,-6.775710,0.474234,-8.663545,-7.673359,1.476950,0.181517,-8.680397,-2.088589,-8.892673,-3.066460,-7.163343,-1.604172,-3.664074,-8.879405,-4.322665,9.215379,4.118555,-5.255596,-2.168834,-3.842895,3.954192,-0.629501,9.288782,0.714932,6.251477,-4.713967,-3.532840,-3.867560,2.582703,-9.505242,-7.349581,3.719129,-4.665589,5.001818,-6.535723,-7.474146,3.597480,9.797165,1.456183,-6.432417,1.307823,-0.138945,-5.275157,-1.136670,-1.602959,3.853016,-5.783702,-9.231688,-3.772860,4.542491,0.763188,6.133436,-5.114166,9.883066,8.169049,-6.185747,-8.515336,-9.132355,0.852383,2.534300,0.370412,-2.337526,7.085115,-8.311617,6.944443,-7.585313,0.191746,3.798317,9.080779,3.798678,4.349389,5.565807,2.805365,-2.086053,-6.197417,-3.305715,-6.756695,8.908776,4.145225,-7.335675,7.856735,5.414505,-5.132152,-3.883357,-5.500820,-0.251686,5.694397,5.660487,-7.061531,4.380717,3.948388,-4.034164,0.610261,3.484123,-5.516816,2.547647,1.197558,-0.309489,6.682409,3.257336,7.049098,8.118086,8.590119,7.553970,-7.153099,-5.739165,-1.558664,-1.596424,-8.750478,-1.184282,6.570677,-2.477517,-7.634962,-2.175710,1.620548,9.227657,6.750842,9.612541,-8.880848,-4.680569,1.618387,-1.083708,-0.740692,7.349388,3.841094,4.926135,9.279946,7.146128,-5.611194,-5.189732,0.471483,-0.319662,-3.269558,6.446227,-0.593037,-8.963701,-3.507167,7.792376,1.502689,7.278261,-2.193496,3.214537,-8.784170,-0.263344,-0.952036,-7.214860,-8.477757,-5.940977,-9.454991,8.402706,-3.998257,-1.373535,8.292121,1.342790,8.280581,1.876732,9.572011,2.467987,4.242749,-4.064241,-1.807169,1.249423,-3.493018,-8.128612,0.890376,7.958529,6.783602,-7.224872,-3.218522,3.586820,-6.571312,3.824164,3.572412,7.335424,-7.810002,-7.192919,-0.986075,-7.269371,7.731102,3.691184,-4.012851,-0.383692,-3.515010,-2.400344,-9.668911,-1.058289,-3.438743,-3.136479,2.758705,-5.658085,-1.786480,-0.811636,-0.948088,6.253145,1.804147,-2.476552,5.246672,8.974125,4.002362,-5.006520,4.288614,-7.863476,7.203915,7.492263,-5.756383,-7.184402,-6.455174,3.914897,3.273255,8.151271,5.074875,9.821840,-4.379219,4.249855,-4.650273,-8.977745,-5.146726,-8.241257,9.266425,-2.442694,5.052610,-7.160194,-1.189395,6.717063,-0.624319,-1.021251,-4.455672,7.121319,4.173099,6.953527,-3.079141,1.193418,-7.592591,-8.409586,-1.282656,-8.333331,-8.723107,1.748814,-6.997740,-3.834524,1.521593,8.572108,-1.279735,1.651375,9.095430,-5.035179,-7.597152,-1.229722,2.946392,-9.290491,-2.432071,0.529563,-4.716450,2.239833,3.944217,2.588299,-0.677594,4.768344,0.928564,-3.269642,8.975133,5.666831,2.857157,-8.702768,-0.916657,-5.875095,6.539474,5.325604,-5.213131,6.445507,1.282231,0.303396,7.037680,-9.364736,2.592713,3.343478,6.208391,-4.463033,-4.360139,5.856953,3.986583,4.629012,-5.230350,-0.422973,7.145911,-4.177393,5.811549,7.606850,-5.800808,3.529034,-4.066399,-6.977719,7.894108,-4.490306,-7.076643,7.017098,8.664859,-7.088445,9.013104,9.720153,-8.962639,7.178715,2.954130,-1.957150,5.189049,1.163628,-0.394748,0.581803,-2.738306,-3.154556,-9.846118,-8.564036,-4.737032,1.719337,-7.611738,8.128688,-1.582103,-9.834102,1.218467,-0.958435,-9.783886,-1.462654,6.610945,0.204248,-0.608205,-0.355107,2.430579,-0.598418,-9.600809,7.856832,-5.666427,3.049015,-5.634414,-5.011097,9.307931,-9.179585,-2.396099,-8.447168,-3.712358,1.996686,4.870497,7.303588,8.342141,8.613384,9.012537,-4.833692,1.508284,1.997275,2.418913,2.881110,-5.967160,-7.141716,-1.444184,-2.242971,-5.287774,1.806116,-7.762640,-6.734310,0.126151,2.859381,0.237986,-0.793545,2.508265,-5.957193,-9.892776,5.223690,8.376147,9.885193,8.715683,-7.928981,-0.544150,5.744722,6.445211,-0.094192,5.268215,2.948774,4.729635,9.158968,2.411986,4.622741,9.472525,-8.784766,-3.633553,5.580122,-9.078851,-8.495862,3.614705,6.576986,0.696184,9.287684,6.394520,5.884326,8.105311,0.627860,-8.072773,-4.283295,-2.803517,-4.495024,-8.161484,-9.607756,-5.463158,1.618492,-1.935118,-4.052216,-4.707169,2.124552,-1.348788,-7.039704,8.375392,7.646163,-8.840497,-3.225925,-5.907839,-4.045471,-5.366084,6.138188,9.685479,-3.224418,1.335056,8.707526,5.225543,6.749135,-3.887829,2.274315,-2.589204,7.212912,-7.548602,3.344934,2.531668,3.492251,-0.841874,-7.399902,-6.730569,1.742338,2.759158,7.158470,2.319553,-4.616969,2.326130,-5.426012,-7.496582,-3.472815,6.588395,-9.819236,2.586200,-4.354709,-5.840969,9.282499,5.575108,0.134423,-5.305708,3.476901,2.351895,6.967216,5.765759,-8.270405,4.404641,-3.097593,-1.574606,3.368953,4.855714,3.521266,-1.754656,-2.050695,-3.240302,-7.943457,-0.867799,-3.679282,-4.239353,-4.459936,2.922113,1.337744,-1.491252,0.037890,-0.540183,-5.039921,-9.856446,-8.600581,6.668661,3.010642,-5.470699,-8.859224,5.234470,-6.129045,-9.885426,-9.373866,-1.498750,-1.485168,4.427584,0.825005,9.392375,5.852197,-8.983709,-6.231511,-6.040129,4.806248,-7.738044,-5.874939,-4.318580,1.061572,0.821175,2.036642,3.280353,-5.130375,5.885056,1.884863,-8.081077,2.422254,-7.673088,-5.095215,5.328285,0.359728,7.427386,1.081443,-6.469301,-0.793357,-4.362038,3.701321,1.429032,2.035214,-6.286034,0.702380,1.941264,6.481817,2.694822,3.087999,9.028395,-4.699320,-6.058625,-4.537645,-4.538735,-2.373986,-5.541383,-9.361683,-4.256873,6.381030,-6.682612,4.953484,2.493162,6.160527,-2.337372,9.213611,9.137227,-6.311501,2.366355,6.422070,9.196580,1.068940,4.068418,4.183557,2.955090,-5.888466,-5.867860,3.078525,-6.398391,-3.649485,5.753077,-2.286610,3.270879,-3.399602,-2.890419,-1.922167,-5.506172,7.919685,3.164648,8.153836,4.565431,1.767707,4.216470,0.683280,1.987226,-5.858752,8.986073,-7.741927,3.435866,-4.211824,-3.376506,9.782831,-1.379290,-8.845795,6.386149,-9.732392,-2.126533,-0.967799,-7.066740,6.179078,0.814792,6.091857,5.744750,6.471732,-5.980622,-7.815723,-8.983721,7.943145,-6.487710,5.435370,0.339200,3.224101,7.924595,4.010409,5.922295,7.020491,-0.019402,-2.650206,-9.185977,-3.358081,6.826378,3.668364,-9.603362,-5.275771,5.338157,2.549026,9.300040,-5.334707,-2.445624,-8.285279,-4.668734,5.871398,-2.114246,-7.152619,1.040232,-8.252140,3.028721,3.111135,-4.526645,1.946654,1.948389,3.684884,9.178237,0.120305,4.065032,7.574652,1.125560,7.651184,-6.732526,4.818198,-3.811174,-1.642960,5.487160,-6.798757,7.637778,0.211033,-3.949500,4.748586,-3.157385,6.638460], dtype = "float64")#candidate|818|(2100,)|const|float64
call_817 = func_568_call(relay.reshape(const_818.astype('float64'), [10, 14, 15]), relay.reshape(const_818.astype('float64'), [10, 14, 15]), )
call_819 = func_568_call(relay.reshape(const_818.astype('float64'), [10, 14, 15]), relay.reshape(const_818.astype('float64'), [10, 14, 15]), )
uop_822 = relay.asinh(bop_785.astype('float64')) # shape=(15, 3, 6)
uop_824 = relay.asinh(bop_788.astype('float64')) # shape=(15, 3, 6)
uop_825 = relay.cos(uop_813.astype('float32')) # shape=(15, 3, 6)
uop_827 = relay.cos(uop_815.astype('float32')) # shape=(15, 3, 6)
var_838 = relay.var("var_838", dtype = "float32", shape = (15, 3, 6))#candidate|838|(15, 3, 6)|var|float32
bop_839 = relay.less_equal(uop_825.astype('bool'), relay.reshape(var_838.astype('bool'), relay.shape_of(uop_825))) # shape=(15, 3, 6)
bop_842 = relay.less_equal(uop_827.astype('bool'), relay.reshape(var_838.astype('bool'), relay.shape_of(uop_827))) # shape=(15, 3, 6)
uop_847 = relay.asin(uop_825.astype('float64')) # shape=(15, 3, 6)
uop_849 = relay.asin(uop_827.astype('float64')) # shape=(15, 3, 6)
func_67_call = mod.get_global_var('func_67')
func_69_call = mutated_mod.get_global_var('func_69')
call_851 = relay.TupleGetItem(func_67_call(relay.reshape(uop_825.astype('bool'), [15, 3, 6])), 0)
call_852 = relay.TupleGetItem(func_69_call(relay.reshape(uop_825.astype('bool'), [15, 3, 6])), 0)
func_67_call = mod.get_global_var('func_67')
func_69_call = mutated_mod.get_global_var('func_69')
call_856 = relay.TupleGetItem(func_67_call(relay.reshape(call_751.astype('bool'), [15, 3, 6])), 0)
call_857 = relay.TupleGetItem(func_69_call(relay.reshape(call_751.astype('bool'), [15, 3, 6])), 0)
output = relay.Tuple([call_817,const_818,uop_822,bop_839,uop_847,call_851,call_856,])
output2 = relay.Tuple([call_819,const_818,uop_824,bop_842,uop_849,call_852,call_857,])
func_860 = relay.Function([var_753,var_838,], output)
mod['func_860'] = func_860
mod = relay.transform.InferType()(mod)
mutated_mod['func_860'] = func_860
mutated_mod = relay.transform.InferType()(mutated_mod)
func_860_call = mutated_mod.get_global_var('func_860')
var_862 = relay.var("var_862", dtype = "float64", shape = (15, 3, 6))#candidate|862|(15, 3, 6)|var|float64
var_863 = relay.var("var_863", dtype = "float32", shape = (15, 3, 6))#candidate|863|(15, 3, 6)|var|float32
call_861 = func_860_call(var_862,var_863,)
output = call_861
func_864 = relay.Function([var_862,var_863,], output)
mutated_mod['func_864'] = func_864
mutated_mod = relay.transform.InferType()(mutated_mod)
var_886 = relay.var("var_886", dtype = "uint64", shape = (11, 9, 14))#candidate|886|(11, 9, 14)|var|uint64
var_887 = relay.var("var_887", dtype = "uint64", shape = (11, 9, 14))#candidate|887|(11, 9, 14)|var|uint64
bop_888 = relay.subtract(var_886.astype('uint64'), relay.reshape(var_887.astype('uint64'), relay.shape_of(var_886))) # shape=(11, 9, 14)
uop_891 = relay.acosh(bop_888.astype('float32')) # shape=(11, 9, 14)
uop_893 = relay.sqrt(uop_891.astype('float32')) # shape=(11, 9, 14)
uop_897 = relay.acos(uop_893.astype('float64')) # shape=(11, 9, 14)
bop_900 = relay.mod(var_886.astype('float64'), relay.reshape(uop_891.astype('float64'), relay.shape_of(var_886))) # shape=(11, 9, 14)
func_370_call = mod.get_global_var('func_370')
func_373_call = mutated_mod.get_global_var('func_373')
var_904 = relay.var("var_904", dtype = "bool", shape = (8, 1))#candidate|904|(8, 1)|var|bool
call_903 = relay.TupleGetItem(func_370_call(relay.reshape(var_904.astype('bool'), [8,])), 0)
call_905 = relay.TupleGetItem(func_373_call(relay.reshape(var_904.astype('bool'), [8,])), 0)
output = relay.Tuple([uop_897,bop_900,call_903,var_904,])
output2 = relay.Tuple([uop_897,bop_900,call_905,var_904,])
func_907 = relay.Function([var_886,var_887,var_904,], output)
mod['func_907'] = func_907
mod = relay.transform.InferType()(mod)
var_908 = relay.var("var_908", dtype = "uint64", shape = (11, 9, 14))#candidate|908|(11, 9, 14)|var|uint64
var_909 = relay.var("var_909", dtype = "uint64", shape = (11, 9, 14))#candidate|909|(11, 9, 14)|var|uint64
var_910 = relay.var("var_910", dtype = "bool", shape = (8, 1))#candidate|910|(8, 1)|var|bool
output = func_907(var_908,var_909,var_910,)
func_911 = relay.Function([var_908,var_909,var_910,], output)
mutated_mod['func_911'] = func_911
mutated_mod = relay.transform.InferType()(mutated_mod)
const_969 = relay.const([0.492833,-8.966735,-3.510469,-0.852202], dtype = "float64")#candidate|969|(4,)|const|float64
var_970 = relay.var("var_970", dtype = "float64", shape = (4,))#candidate|970|(4,)|var|float64
bop_971 = relay.divide(const_969.astype('float64'), relay.reshape(var_970.astype('float64'), relay.shape_of(const_969))) # shape=(4,)
uop_976 = relay.tan(bop_971.astype('float64')) # shape=(4,)
uop_979 = relay.cos(const_969.astype('float32')) # shape=(4,)
func_532_call = mod.get_global_var('func_532')
func_536_call = mutated_mod.get_global_var('func_536')
const_985 = relay.const([9.820838,-7.463258,4.006285,-4.660235,4.609172,1.003622,-8.766072,-3.478997,0.825215,-6.844126,-9.755048,6.369723,-0.651603,-4.182749,-4.665287,8.189908,-1.523137,2.955018], dtype = "float32")#candidate|985|(18,)|const|float32
call_984 = relay.TupleGetItem(func_532_call(relay.reshape(const_985.astype('float32'), [2, 9]), relay.reshape(const_985.astype('float32'), [2, 9]), ), 2)
call_986 = relay.TupleGetItem(func_536_call(relay.reshape(const_985.astype('float32'), [2, 9]), relay.reshape(const_985.astype('float32'), [2, 9]), ), 2)
uop_990 = relay.atanh(uop_976.astype('float64')) # shape=(4,)
bop_993 = relay.minimum(uop_990.astype('int8'), relay.reshape(const_969.astype('int8'), relay.shape_of(uop_990))) # shape=(4,)
func_532_call = mod.get_global_var('func_532')
func_536_call = mutated_mod.get_global_var('func_536')
call_996 = relay.TupleGetItem(func_532_call(relay.reshape(const_985.astype('float32'), [2, 9]), relay.reshape(const_985.astype('float32'), [2, 9]), ), 0)
call_997 = relay.TupleGetItem(func_536_call(relay.reshape(const_985.astype('float32'), [2, 9]), relay.reshape(const_985.astype('float32'), [2, 9]), ), 0)
output = relay.Tuple([uop_979,call_984,const_985,bop_993,call_996,])
output2 = relay.Tuple([uop_979,call_986,const_985,bop_993,call_997,])
func_1004 = relay.Function([var_970,], output)
mod['func_1004'] = func_1004
mod = relay.transform.InferType()(mod)
mutated_mod['func_1004'] = func_1004
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1005 = relay.var("var_1005", dtype = "float64", shape = (4,))#candidate|1005|(4,)|var|float64
func_1004_call = mutated_mod.get_global_var('func_1004')
call_1006 = func_1004_call(var_1005)
output = call_1006
func_1007 = relay.Function([var_1005], output)
mutated_mod['func_1007'] = func_1007
mutated_mod = relay.transform.InferType()(mutated_mod)
func_51_call = mod.get_global_var('func_51')
func_52_call = mutated_mod.get_global_var('func_52')
call_1041 = relay.TupleGetItem(func_51_call(), 0)
call_1042 = relay.TupleGetItem(func_52_call(), 0)
uop_1052 = relay.sqrt(call_1041.astype('float64')) # shape=(15, 3, 6)
uop_1054 = relay.sqrt(call_1042.astype('float64')) # shape=(15, 3, 6)
uop_1057 = relay.atanh(call_1041.astype('float64')) # shape=(15, 3, 6)
uop_1059 = relay.atanh(call_1042.astype('float64')) # shape=(15, 3, 6)
func_220_call = mod.get_global_var('func_220')
func_224_call = mutated_mod.get_global_var('func_224')
const_1066 = relay.const([1,2,-6,-10,-8,10,-5,10,-8,-8,-4,-7,-10,8,-10,-8,5,-7,-2,9,4,-1,-8,10,-9,-8,-7,3,6,5,-8,-7,4,5,-2,5,4,7,-7,9,-5,-9,6,10,-5,9,9,-8,9,8,-2,9,8,-9,1,3,8,8,5,-7,-6,-4,10,7,-7,-3,9,6,-6,-4], dtype = "uint8")#candidate|1066|(70,)|const|uint8
call_1065 = relay.TupleGetItem(func_220_call(relay.reshape(const_1066.astype('uint8'), [5, 14]), relay.reshape(const_1066.astype('uint8'), [5, 14]), ), 0)
call_1067 = relay.TupleGetItem(func_224_call(relay.reshape(const_1066.astype('uint8'), [5, 14]), relay.reshape(const_1066.astype('uint8'), [5, 14]), ), 0)
output = relay.Tuple([uop_1052,uop_1057,call_1065,const_1066,])
output2 = relay.Tuple([uop_1054,uop_1059,call_1067,const_1066,])
func_1068 = relay.Function([], output)
mod['func_1068'] = func_1068
mod = relay.transform.InferType()(mod)
output = func_1068()
func_1069 = relay.Function([], output)
mutated_mod['func_1069'] = func_1069
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1078 = relay.var("var_1078", dtype = "uint64", shape = (5, 1))#candidate|1078|(5, 1)|var|uint64
var_1079 = relay.var("var_1079", dtype = "uint64", shape = (5, 6))#candidate|1079|(5, 6)|var|uint64
bop_1080 = relay.greater(var_1078.astype('bool'), var_1079.astype('bool')) # shape=(5, 6)
output = bop_1080
output2 = bop_1080
func_1089 = relay.Function([var_1078,var_1079,], output)
mod['func_1089'] = func_1089
mod = relay.transform.InferType()(mod)
mutated_mod['func_1089'] = func_1089
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1089_call = mutated_mod.get_global_var('func_1089')
var_1091 = relay.var("var_1091", dtype = "uint64", shape = (5, 1))#candidate|1091|(5, 1)|var|uint64
var_1092 = relay.var("var_1092", dtype = "uint64", shape = (5, 6))#candidate|1092|(5, 6)|var|uint64
call_1090 = func_1089_call(var_1091,var_1092,)
output = call_1090
func_1093 = relay.Function([var_1091,var_1092,], output)
mutated_mod['func_1093'] = func_1093
mutated_mod = relay.transform.InferType()(mutated_mod)
func_665_call = mod.get_global_var('func_665')
func_667_call = mutated_mod.get_global_var('func_667')
call_1113 = relay.TupleGetItem(func_665_call(), 0)
call_1114 = relay.TupleGetItem(func_667_call(), 0)
output = call_1113
output2 = call_1114
func_1117 = relay.Function([], output)
mod['func_1117'] = func_1117
mod = relay.transform.InferType()(mod)
mutated_mod['func_1117'] = func_1117
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1117_call = mutated_mod.get_global_var('func_1117')
call_1118 = func_1117_call()
output = call_1118
func_1119 = relay.Function([], output)
mutated_mod['func_1119'] = func_1119
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1120 = relay.const([[[8,-3,-4,4,10,1,-6,3,-5,6,3],[7,2,-2,6,7,5,4,-1,-1,4,4],[-7,5,-4,-7,10,-10,10,9,1,-1,-10],[-6,-3,-1,-10,-6,3,-9,2,-5,-10,7],[9,7,4,8,2,4,-10,-4,-1,7,4],[3,2,8,-5,1,3,-5,5,-10,7,4],[-10,-10,-1,-4,-3,-8,-7,6,-3,5,6]],[[-2,5,-9,7,-10,2,-7,6,-5,-10,-3],[-1,-8,-7,-2,-5,3,-8,2,6,4,3],[10,-5,-8,7,-4,2,5,-7,2,1,3],[8,7,-1,-1,-2,-4,5,4,5,2,-9],[7,-4,-9,-6,6,-2,-7,4,7,-10,-3],[-3,7,-4,-5,6,9,1,-8,-4,1,7],[-9,10,9,7,3,10,9,-8,6,-6,-7]],[[7,-4,-10,-7,4,2,-3,4,-4,3,-8],[9,-7,-4,-4,-10,6,3,8,5,-7,-8],[7,-2,10,8,-8,-3,-3,-5,-8,9,-8],[10,-4,2,3,6,1,-4,3,-8,-8,-5],[-8,2,-7,1,-5,1,-7,3,-2,-9,-6],[5,-4,4,6,5,-2,-1,7,-2,3,4],[-8,-10,-7,-7,-4,4,-7,-10,-8,6,-1]],[[8,-9,10,8,1,-10,-10,2,2,2,-1],[-6,2,5,5,7,4,4,9,8,9,3],[-7,-7,-8,-4,-5,-2,-4,7,3,-3,-3],[10,10,-1,8,-8,-8,1,-1,-7,-5,7],[7,9,9,6,-8,3,-6,6,-10,-6,-7],[-8,-4,-9,5,-8,-10,-10,-8,3,5,-3],[2,-5,-4,-3,-10,3,6,-2,-8,3,5]],[[8,-2,-9,-5,-9,4,9,7,-10,3,-10],[5,-9,-10,10,-6,1,7,6,-4,1,9],[5,2,-2,9,5,3,7,-8,1,5,-3],[-1,6,-6,-1,-7,-10,2,-5,3,4,-5],[1,8,-7,-3,-5,1,-2,-8,-1,-1,-5],[4,-3,9,-5,10,8,-8,-3,5,-7,-1],[3,-2,-4,7,6,-9,-8,-8,-5,-9,-8]]], dtype = "int64")#candidate|1120|(5, 7, 11)|const|int64
const_1121 = relay.const([[[10,4,4,9,5,-6,-5,7,5,-5,-6],[-4,10,6,2,-10,9,-2,-5,9,-3,1],[3,5,5,-3,-7,-9,-2,3,-8,6,8],[-2,-1,2,-6,-1,10,-7,7,6,-8,3],[10,1,8,5,5,8,10,1,5,-5,9],[-10,9,-10,-10,6,-6,8,-6,9,-4,-9],[-7,10,-2,6,6,-8,3,3,-9,-3,-8]],[[1,6,5,-5,4,-7,-5,-1,2,9,2],[6,-5,-6,1,1,-6,-2,-7,-7,10,-7],[-5,-5,6,7,-6,-5,9,1,-3,-4,9],[10,-4,7,3,10,-10,4,-9,-4,-3,5],[-10,4,9,2,3,-2,-1,2,10,-1,-6],[-6,4,-9,-1,-9,-3,2,-2,-7,-6,-3],[-9,10,-6,2,4,7,10,-4,-4,-1,7]],[[-8,8,3,6,3,-10,5,-6,8,-6,7],[-2,-6,6,-3,3,-10,-7,-6,-9,-8,-4],[5,-9,1,9,-1,10,7,-5,9,-2,-8],[3,9,7,-3,10,2,-10,-5,10,9,-5],[8,-7,-9,-7,7,5,4,-5,2,4,7],[6,4,-9,-5,-1,6,1,8,-4,-10,1],[8,2,3,3,-1,-3,-4,7,-9,-4,6]],[[-10,-3,-9,1,6,8,-9,-3,1,1,8],[8,-7,-6,10,4,8,4,8,-7,-7,4],[8,-8,3,4,-7,9,-7,-4,-6,-9,-4],[2,-10,-10,-6,3,-4,-4,2,-8,-10,4],[-3,6,-7,-6,2,4,-8,9,9,-9,-4],[-9,1,-9,6,8,-2,-1,-3,-9,-6,7],[9,10,-9,-2,4,-3,-1,-7,8,8,-2]],[[-8,-7,-10,-3,-7,-4,-2,-5,-9,-8,-5],[-5,-7,10,4,6,8,10,-6,-4,5,9],[-10,2,3,-2,-8,-3,-1,-10,-4,8,2],[10,10,1,2,3,-4,3,-2,8,-3,4],[-5,-10,3,-2,5,2,-6,1,3,-9,-8],[4,9,-1,-6,-8,-9,9,8,6,-3,-1],[9,4,7,-9,-8,-4,-7,-1,9,-2,-10]]], dtype = "int64")#candidate|1121|(5, 7, 11)|const|int64
bop_1122 = relay.subtract(const_1120.astype('int64'), relay.reshape(const_1121.astype('int64'), relay.shape_of(const_1120))) # shape=(5, 7, 11)
bop_1133 = relay.bitwise_xor(const_1120.astype('uint32'), relay.reshape(bop_1122.astype('uint32'), relay.shape_of(const_1120))) # shape=(5, 7, 11)
uop_1137 = relay.sinh(bop_1122.astype('float64')) # shape=(5, 7, 11)
uop_1140 = relay.acosh(bop_1133.astype('float64')) # shape=(5, 7, 11)
output = relay.Tuple([uop_1137,uop_1140,])
output2 = relay.Tuple([uop_1137,uop_1140,])
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