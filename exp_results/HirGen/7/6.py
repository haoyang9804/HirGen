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
var_74 = relay.var("var_74", dtype = "float32", shape = (11, 2, 5))#candidate|74|(11, 2, 5)|var|float32
var_75 = relay.var("var_75", dtype = "float32", shape = (11, 2, 5))#candidate|75|(11, 2, 5)|var|float32
bop_76 = relay.multiply(var_74.astype('float32'), relay.reshape(var_75.astype('float32'), relay.shape_of(var_74))) # shape=(11, 2, 5)
uop_86 = relay.asinh(bop_76.astype('float32')) # shape=(11, 2, 5)
var_88 = relay.var("var_88", dtype = "float32", shape = (11, 2, 5))#candidate|88|(11, 2, 5)|var|float32
bop_89 = relay.greater_equal(uop_86.astype('bool'), relay.reshape(var_88.astype('bool'), relay.shape_of(uop_86))) # shape=(11, 2, 5)
bop_92 = relay.less_equal(uop_86.astype('bool'), relay.reshape(bop_76.astype('bool'), relay.shape_of(uop_86))) # shape=(11, 2, 5)
bop_100 = relay.floor_divide(bop_92.astype('float32'), relay.reshape(bop_89.astype('float32'), relay.shape_of(bop_92))) # shape=(11, 2, 5)
uop_114 = relay.sigmoid(uop_86.astype('float32')) # shape=(11, 2, 5)
var_116 = relay.var("var_116", dtype = "float32", shape = (11, 2, 5))#candidate|116|(11, 2, 5)|var|float32
bop_117 = relay.logical_xor(uop_114.astype('uint8'), relay.reshape(var_116.astype('uint8'), relay.shape_of(uop_114))) # shape=(11, 2, 5)
output = relay.Tuple([bop_100,bop_117,])
output2 = relay.Tuple([bop_100,bop_117,])
func_120 = relay.Function([var_74,var_75,var_88,var_116,], output)
mod['func_120'] = func_120
mod = relay.transform.InferType()(mod)
mutated_mod['func_120'] = func_120
mutated_mod = relay.transform.InferType()(mutated_mod)
func_120_call = mutated_mod.get_global_var('func_120')
var_122 = relay.var("var_122", dtype = "float32", shape = (11, 2, 5))#candidate|122|(11, 2, 5)|var|float32
var_123 = relay.var("var_123", dtype = "float32", shape = (11, 2, 5))#candidate|123|(11, 2, 5)|var|float32
var_124 = relay.var("var_124", dtype = "float32", shape = (11, 2, 5))#candidate|124|(11, 2, 5)|var|float32
var_125 = relay.var("var_125", dtype = "float32", shape = (11, 2, 5))#candidate|125|(11, 2, 5)|var|float32
call_121 = func_120_call(var_122,var_123,var_124,var_125,)
output = call_121
func_126 = relay.Function([var_122,var_123,var_124,var_125,], output)
mutated_mod['func_126'] = func_126
mutated_mod = relay.transform.InferType()(mutated_mod)
var_139 = relay.var("var_139", dtype = "float32", shape = (16, 3, 5))#candidate|139|(16, 3, 5)|var|float32
uop_140 = relay.rsqrt(var_139.astype('float32')) # shape=(16, 3, 5)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
const_143 = relay.const([[3.706282,-3.224387,1.400223,1.203970,-6.539704,0.232990,0.128129,-0.175557,-0.470634,-6.797087,-9.797564,-9.786638,8.825299,-8.465257,4.607307,6.853818,4.798844,-4.030009,-6.518794,6.341496,5.293885,-2.229747,0.982578,-5.050253,-4.909606,1.040524,6.136174,-7.100076,-5.039556,-9.290976,3.667388,-5.632810,9.347690,8.627889,-4.694049,-1.437953,9.876816,-8.203647,5.632303,2.303913,6.588413,-5.306469,-4.833105,3.047209,-3.637834,-1.274083,6.548497,-7.867543,6.773185,-9.197676,-5.932397,7.263679,7.370313,-9.004257,9.087248,4.914044,8.865649,0.433743,-6.195565,-3.478918,-6.555767,-9.182562,2.960409,5.314230,5.494720,-1.706178,-3.244689,3.370609,0.360969,8.049643,9.773104,0.438068,-4.374011,-7.770464,0.812124,7.071226,-8.604134,9.200161,0.642456,-5.828809,-2.183631,-5.877569,4.899939,-8.368361,-0.268397,0.581049,-3.653445,-8.874243,6.861346,-5.477693,5.240458,-7.244887,8.278585,3.128355,-8.454683,5.440610,-0.832946,-8.196113,-1.125913,-3.200562,-6.566594,-7.209851,-6.400020,2.757987,-8.556001,-4.024048,-7.454742,1.939130,-8.581760,-4.741770]], dtype = "float32")#candidate|143|(1, 110)|const|float32
call_142 = relay.TupleGetItem(func_120_call(relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), ), 0)
call_144 = relay.TupleGetItem(func_126_call(relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), relay.reshape(const_143.astype('float32'), [11, 2, 5]), ), 0)
output = relay.Tuple([uop_140,call_142,const_143,])
output2 = relay.Tuple([uop_140,call_144,const_143,])
func_147 = relay.Function([var_139,], output)
mod['func_147'] = func_147
mod = relay.transform.InferType()(mod)
var_148 = relay.var("var_148", dtype = "float32", shape = (16, 3, 5))#candidate|148|(16, 3, 5)|var|float32
output = func_147(var_148)
func_149 = relay.Function([var_148], output)
mutated_mod['func_149'] = func_149
mutated_mod = relay.transform.InferType()(mutated_mod)
var_207 = relay.var("var_207", dtype = "bool", shape = (10, 11, 12))#candidate|207|(10, 11, 12)|var|bool
var_208 = relay.var("var_208", dtype = "bool", shape = (10, 11, 12))#candidate|208|(10, 11, 12)|var|bool
bop_209 = relay.logical_and(var_207.astype('bool'), relay.reshape(var_208.astype('bool'), relay.shape_of(var_207))) # shape=(10, 11, 12)
output = relay.Tuple([bop_209,])
output2 = relay.Tuple([bop_209,])
func_215 = relay.Function([var_207,var_208,], output)
mod['func_215'] = func_215
mod = relay.transform.InferType()(mod)
mutated_mod['func_215'] = func_215
mutated_mod = relay.transform.InferType()(mutated_mod)
func_215_call = mutated_mod.get_global_var('func_215')
var_217 = relay.var("var_217", dtype = "bool", shape = (10, 11, 12))#candidate|217|(10, 11, 12)|var|bool
var_218 = relay.var("var_218", dtype = "bool", shape = (10, 11, 12))#candidate|218|(10, 11, 12)|var|bool
call_216 = func_215_call(var_217,var_218,)
output = call_216
func_219 = relay.Function([var_217,var_218,], output)
mutated_mod['func_219'] = func_219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_254 = relay.var("var_254", dtype = "float64", shape = (9, 14))#candidate|254|(9, 14)|var|float64
uop_255 = relay.erf(var_254.astype('float64')) # shape=(9, 14)
func_147_call = mod.get_global_var('func_147')
func_149_call = mutated_mod.get_global_var('func_149')
var_259 = relay.var("var_259", dtype = "float32", shape = (240,))#candidate|259|(240,)|var|float32
call_258 = relay.TupleGetItem(func_147_call(relay.reshape(var_259.astype('float32'), [16, 3, 5])), 1)
call_260 = relay.TupleGetItem(func_149_call(relay.reshape(var_259.astype('float32'), [16, 3, 5])), 1)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
call_261 = relay.TupleGetItem(func_120_call(relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), ), 1)
call_262 = relay.TupleGetItem(func_126_call(relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), relay.reshape(call_258.astype('float32'), [11, 2, 5]), ), 1)
bop_263 = relay.divide(uop_255.astype('float64'), relay.reshape(var_254.astype('float64'), relay.shape_of(uop_255))) # shape=(9, 14)
bop_270 = relay.mod(uop_255.astype('float64'), relay.reshape(bop_263.astype('float64'), relay.shape_of(uop_255))) # shape=(9, 14)
bop_273 = relay.less_equal(uop_255.astype('bool'), relay.reshape(bop_263.astype('bool'), relay.shape_of(uop_255))) # shape=(9, 14)
uop_278 = relay.rsqrt(bop_273.astype('float64')) # shape=(9, 14)
bop_281 = relay.left_shift(uop_278.astype('uint64'), relay.reshape(uop_255.astype('uint64'), relay.shape_of(uop_278))) # shape=(9, 14)
var_286 = relay.var("var_286", dtype = "float64", shape = (9, 14))#candidate|286|(9, 14)|var|float64
bop_287 = relay.add(uop_278.astype('float64'), relay.reshape(var_286.astype('float64'), relay.shape_of(uop_278))) # shape=(9, 14)
uop_290 = relay.erf(bop_281.astype('float64')) # shape=(9, 14)
bop_310 = relay.subtract(uop_290.astype('int16'), relay.reshape(bop_270.astype('int16'), relay.shape_of(uop_290))) # shape=(9, 14)
func_147_call = mod.get_global_var('func_147')
func_149_call = mutated_mod.get_global_var('func_149')
call_319 = relay.TupleGetItem(func_147_call(relay.reshape(var_259.astype('float32'), [16, 3, 5])), 0)
call_320 = relay.TupleGetItem(func_149_call(relay.reshape(var_259.astype('float32'), [16, 3, 5])), 0)
output = relay.Tuple([call_258,var_259,call_261,bop_287,bop_310,call_319,])
output2 = relay.Tuple([call_260,var_259,call_262,bop_287,bop_310,call_320,])
func_328 = relay.Function([var_254,var_259,var_286,], output)
mod['func_328'] = func_328
mod = relay.transform.InferType()(mod)
var_329 = relay.var("var_329", dtype = "float64", shape = (9, 14))#candidate|329|(9, 14)|var|float64
var_330 = relay.var("var_330", dtype = "float32", shape = (240,))#candidate|330|(240,)|var|float32
var_331 = relay.var("var_331", dtype = "float64", shape = (9, 14))#candidate|331|(9, 14)|var|float64
output = func_328(var_329,var_330,var_331,)
func_332 = relay.Function([var_329,var_330,var_331,], output)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
const_368 = relay.const([[-4,2,6,1,9,6,-8,8,8,4,1,-1,-1]], dtype = "uint64")#candidate|368|(1, 13)|const|uint64
var_369 = relay.var("var_369", dtype = "uint64", shape = (7, 13))#candidate|369|(7, 13)|var|uint64
bop_370 = relay.greater(const_368.astype('bool'), var_369.astype('bool')) # shape=(7, 13)
uop_376 = relay.cosh(var_369.astype('float32')) # shape=(7, 13)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
const_380 = relay.const([9.720095,9.171078,6.581056,-1.645105,2.575432,4.631463,-6.470812,-0.448232,0.859750,-5.663900,-8.413856,-4.119286,8.006302,3.213427,-3.812361,1.178857,5.616006,2.613458,9.126468,-1.868993,-2.229573,-6.590331,-9.505759,-2.786056,0.122912,-0.010322,1.248034,6.961624,0.092698,2.753642,2.762984,5.094371,4.858807,7.614516,-4.778097,0.220724,-6.820753,3.734880,-3.059184,-0.363480,9.810408,-8.791992,5.713007,9.157038,-2.449419,-4.313777,7.983720,-7.744965,7.307755,-7.248973,0.008402,-6.532213,5.316212,-8.506670,8.278457,7.916988,9.373511,-0.734854,-3.070024,-8.415633,-8.694293,-8.669782,-2.680210,4.995490,1.007247,-8.872323,5.928805,-6.386398,-5.636704,4.996026,0.473497,-1.120295,-2.476595,3.835681,5.557523,3.195695,8.934681,2.733703,-7.455546,8.158584,-2.612790,-2.174964,-0.667909,-2.574029,3.712137,-4.491641,9.225595,3.117754,5.633367,-4.183839,-6.416515,6.391815,0.580890,-2.315461,8.135816,4.784005,0.354540,8.963066,-3.282667,-7.330854,4.531699,-7.592649,-0.103933,-5.956751,1.796757,-2.311255,-2.970077,2.708086,-3.509797,7.308125], dtype = "float32")#candidate|380|(110,)|const|float32
call_379 = relay.TupleGetItem(func_120_call(relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), ), 1)
call_381 = relay.TupleGetItem(func_126_call(relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), relay.reshape(const_380.astype('float32'), [11, 2, 5]), ), 1)
bop_382 = relay.bitwise_and(uop_376.astype('int8'), relay.reshape(var_369.astype('int8'), relay.shape_of(uop_376))) # shape=(7, 13)
bop_389 = relay.mod(uop_376.astype('float32'), const_368.astype('float32')) # shape=(7, 13)
output = relay.Tuple([bop_370,call_379,const_380,bop_382,bop_389,])
output2 = relay.Tuple([bop_370,call_381,const_380,bop_382,bop_389,])
func_393 = relay.Function([var_369,], output)
mod['func_393'] = func_393
mod = relay.transform.InferType()(mod)
var_394 = relay.var("var_394", dtype = "uint64", shape = (7, 13))#candidate|394|(7, 13)|var|uint64
output = func_393(var_394)
func_395 = relay.Function([var_394], output)
mutated_mod['func_395'] = func_395
mutated_mod = relay.transform.InferType()(mutated_mod)
const_424 = relay.const([[9,-4,3,-2,7,7],[9,9,9,-3,4,1],[4,-2,1,-3,-4,-1],[-2,9,-6,-1,8,-2],[6,2,-7,3,10,-4],[-7,-3,8,10,-8,-9],[-5,-9,8,2,1,5],[-4,9,6,-8,2,6],[7,10,-4,10,-1,2],[4,-5,-7,-1,1,-6],[-5,1,-8,-9,-2,-2],[10,-6,3,-2,2,-2],[-10,-10,2,9,4,7]], dtype = "uint16")#candidate|424|(13, 6)|const|uint16
var_425 = relay.var("var_425", dtype = "uint16", shape = (13, 6))#candidate|425|(13, 6)|var|uint16
bop_426 = relay.add(const_424.astype('uint16'), relay.reshape(var_425.astype('uint16'), relay.shape_of(const_424))) # shape=(13, 6)
output = bop_426
output2 = bop_426
func_430 = relay.Function([var_425,], output)
mod['func_430'] = func_430
mod = relay.transform.InferType()(mod)
mutated_mod['func_430'] = func_430
mutated_mod = relay.transform.InferType()(mutated_mod)
var_431 = relay.var("var_431", dtype = "uint16", shape = (13, 6))#candidate|431|(13, 6)|var|uint16
func_430_call = mutated_mod.get_global_var('func_430')
call_432 = func_430_call(var_431)
output = call_432
func_433 = relay.Function([var_431], output)
mutated_mod['func_433'] = func_433
mutated_mod = relay.transform.InferType()(mutated_mod)
const_435 = relay.const([[[-8],[4],[4],[-9],[6],[10],[-5],[1],[-10],[5],[10]],[[-2],[-3],[-10],[-10],[2],[-2],[7],[2],[8],[3],[-7]],[[4],[-8],[-1],[-8],[4],[-4],[2],[5],[4],[9],[8]],[[1],[-1],[-7],[-8],[-5],[-2],[-7],[-5],[8],[7],[-8]],[[7],[-3],[5],[-7],[8],[-10],[-9],[-3],[3],[1],[10]],[[10],[5],[9],[-1],[2],[-2],[-1],[7],[-9],[-7],[3]],[[-8],[-5],[6],[-6],[-4],[-4],[-8],[2],[9],[7],[1]],[[10],[7],[-2],[-10],[10],[-1],[9],[-9],[-4],[5],[3]]], dtype = "int8")#candidate|435|(8, 11, 1)|const|int8
var_436 = relay.var("var_436", dtype = "int8", shape = (8, 11, 10))#candidate|436|(8, 11, 10)|var|int8
bop_437 = relay.not_equal(const_435.astype('bool'), var_436.astype('bool')) # shape=(8, 11, 10)
bop_440 = relay.add(const_435.astype('uint8'), bop_437.astype('uint8')) # shape=(8, 11, 10)
bop_444 = relay.multiply(bop_440.astype('float64'), relay.reshape(bop_437.astype('float64'), relay.shape_of(bop_440))) # shape=(8, 11, 10)
func_147_call = mod.get_global_var('func_147')
func_149_call = mutated_mod.get_global_var('func_149')
const_451 = relay.const([3.502512,4.647294,-4.776206,-1.450769,3.791419,7.900762,-0.745025,-0.193655,5.963309,-6.979725,5.570769,-9.621760,0.490772,7.998722,1.505685,-8.515671,9.226490,-1.288137,-9.163095,-6.566928,-9.289896,1.699922,-8.194774,-2.109253,-9.895818,-7.870905,-3.013940,2.707753,3.866040,7.071562,4.053502,7.498356,6.030076,-5.060907,-2.913319,8.784532,-9.844830,-8.536362,4.552768,5.648155,1.976505,5.382495,-5.223722,8.843309,6.527445,1.089288,-5.164928,8.845974,-1.065069,9.281736,-0.750360,6.404243,-8.682394,-9.257225,0.997173,2.001749,3.100802,-0.081417,8.937906,9.130378,-9.769584,-7.371939,-5.453649,2.018632,3.870073,0.485200,9.279482,8.455617,-6.789833,2.811200,7.875528,-6.376500,1.521786,-3.676296,7.569257,-3.942075,-5.810076,-0.707451,-8.618925,-6.579915,7.952116,-1.279144,-6.706444,-3.557217,4.119221,-7.319625,-3.982933,4.642594,-4.961546,-1.629708,-6.482162,7.531379,6.296561,4.328220,9.781400,-1.541549,9.673945,-0.782609,-5.803655,4.011217,-0.889338,-5.094633,-2.681598,4.174439,4.449581,-7.938318,-0.659687,5.984406,-1.200001,2.235746,-3.239865,-4.817561,-8.343786,1.686962,9.720371,-3.662611,3.426529,-5.928270,4.734927,-3.170920,5.479865,-5.132877,-8.941092,-4.926750,-6.545827,9.193667,2.138033,-7.218536,-4.612497,-0.344461,-3.019014,6.185631,9.861897,-7.274868,1.996195,1.436979,7.979145,-0.787325,3.970565,-5.242377,9.554448,-9.280086,5.414517,-7.066521,5.501054,8.855656,-1.421540,0.734852,-9.344591,-4.718259,-5.709408,5.462092,3.648545,9.327849,-9.547279,-0.182072,4.525427,-9.562827,8.507946,-4.537590,4.856098,6.989977,6.745506,1.420103,-6.807825,1.212283,2.151205,-3.398583,9.639047,-1.069249,-2.073363,9.963989,8.791230,-8.400945,-7.076901,9.026239,1.496756,3.958241,4.726841,-3.016116,-1.694942,-3.937508,-1.581383,-0.276031,1.603905,9.492257,6.657160,9.271269,-5.480596,-4.258122,5.169607,-4.326948,-5.114924,9.461161,-1.917806,1.165218,9.508453,9.497553,-7.857296,5.486081,-3.220472,2.713907,3.968847,-7.140321,8.245111,9.501062,0.327171,4.909729,7.450509,6.074198,-7.687364,2.304698,0.609919,-3.697682,-0.120629,7.909223,-0.635937,3.540752,7.767477,-9.116717,-1.269400,8.232506,8.724043,1.944479,5.440533,9.859624,3.063007,-2.884452,4.038372,-6.572576,-7.884422,-5.279638,9.909674,8.879399,1.624090,1.768699,7.778351,2.478229,0.457204,4.492499], dtype = "float32")#candidate|451|(240,)|const|float32
call_450 = relay.TupleGetItem(func_147_call(relay.reshape(const_451.astype('float32'), [16, 3, 5])), 1)
call_452 = relay.TupleGetItem(func_149_call(relay.reshape(const_451.astype('float32'), [16, 3, 5])), 1)
var_454 = relay.var("var_454", dtype = "bool", shape = (8, 11, 10))#candidate|454|(8, 11, 10)|var|bool
bop_455 = relay.mod(bop_437.astype('float32'), relay.reshape(var_454.astype('float32'), relay.shape_of(bop_437))) # shape=(8, 11, 10)
bop_460 = relay.floor_mod(bop_437.astype('float32'), relay.reshape(var_454.astype('float32'), relay.shape_of(bop_437))) # shape=(8, 11, 10)
uop_465 = relay.sinh(bop_437.astype('float64')) # shape=(8, 11, 10)
output = relay.Tuple([bop_444,call_450,const_451,bop_455,bop_460,uop_465,])
output2 = relay.Tuple([bop_444,call_452,const_451,bop_455,bop_460,uop_465,])
func_467 = relay.Function([var_436,var_454,], output)
mod['func_467'] = func_467
mod = relay.transform.InferType()(mod)
var_468 = relay.var("var_468", dtype = "int8", shape = (8, 11, 10))#candidate|468|(8, 11, 10)|var|int8
var_469 = relay.var("var_469", dtype = "bool", shape = (8, 11, 10))#candidate|469|(8, 11, 10)|var|bool
output = func_467(var_468,var_469,)
func_470 = relay.Function([var_468,var_469,], output)
mutated_mod['func_470'] = func_470
mutated_mod = relay.transform.InferType()(mutated_mod)
var_514 = relay.var("var_514", dtype = "float32", shape = (10, 13))#candidate|514|(10, 13)|var|float32
uop_515 = relay.rsqrt(var_514.astype('float32')) # shape=(10, 13)
var_518 = relay.var("var_518", dtype = "float32", shape = (10, 13))#candidate|518|(10, 13)|var|float32
bop_519 = relay.greater_equal(var_514.astype('bool'), relay.reshape(var_518.astype('bool'), relay.shape_of(var_514))) # shape=(10, 13)
func_430_call = mod.get_global_var('func_430')
func_433_call = mutated_mod.get_global_var('func_433')
const_523 = relay.const([[5,-10,-8,7,-4,9,3,-6,-9,7,8,1,-2,3,10,-10,7,-3,10,-4,-10,-8,8,2,-1,-3,6,6,5,-1,-1,2,-3,-4,7,-7,10,10,3,7,8,10,-6,-3,6,-7,-7,8,8,-4,1,-9,-1,10,10,8,5,-3,6,3,-2,-10,1,-10,9,3,-7,10,-3,2,8,-3,-9,6,-4,9,9,-6]], dtype = "uint16")#candidate|523|(1, 78)|const|uint16
call_522 = func_430_call(relay.reshape(const_523.astype('uint16'), [13, 6]))
call_524 = func_430_call(relay.reshape(const_523.astype('uint16'), [13, 6]))
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
var_531 = relay.var("var_531", dtype = "bool", shape = (1320,))#candidate|531|(1320,)|var|bool
call_530 = relay.TupleGetItem(func_215_call(relay.reshape(var_531.astype('bool'), [10, 11, 12]), relay.reshape(var_531.astype('bool'), [10, 11, 12]), ), 0)
call_532 = relay.TupleGetItem(func_219_call(relay.reshape(var_531.astype('bool'), [10, 11, 12]), relay.reshape(var_531.astype('bool'), [10, 11, 12]), ), 0)
bop_537 = relay.power(uop_515.astype('float32'), relay.reshape(var_514.astype('float32'), relay.shape_of(uop_515))) # shape=(10, 13)
func_393_call = mod.get_global_var('func_393')
func_395_call = mutated_mod.get_global_var('func_395')
const_544 = relay.const([2,-2,1,-7,8,-10,7,3,5,-2,2,-10,10,-6,-4,3,-1,9,10,1,-4,-4,10,-4,-1,-5,6,-6,5,3,2,-5,10,-5,2,6,-10,6,-8,-6,-7,-1,-8,9,-10,-1,8,-5,2,7,-9,-1,-3,-8,9,9,-5,-9,9,4,-10,-2,-3,-7,9,2,-6,7,-1,-10,7,4,-10,-4,-1,-6,5,9,7,4,9,3,8,5,-3,10,2,1,-7,1,-3], dtype = "uint64")#candidate|544|(91,)|const|uint64
call_543 = relay.TupleGetItem(func_393_call(relay.reshape(const_544.astype('uint64'), [7, 13])), 0)
call_545 = relay.TupleGetItem(func_395_call(relay.reshape(const_544.astype('uint64'), [7, 13])), 0)
bop_546 = relay.mod(uop_515.astype('float32'), relay.reshape(bop_519.astype('float32'), relay.shape_of(uop_515))) # shape=(10, 13)
var_552 = relay.var("var_552", dtype = "float32", shape = (10, 13))#candidate|552|(10, 13)|var|float32
bop_553 = relay.divide(bop_537.astype('float64'), relay.reshape(var_552.astype('float64'), relay.shape_of(bop_537))) # shape=(10, 13)
uop_556 = relay.acosh(bop_546.astype('float64')) # shape=(10, 13)
output = relay.Tuple([call_522,const_523,call_530,var_531,call_543,const_544,bop_553,uop_556,])
output2 = relay.Tuple([call_524,const_523,call_532,var_531,call_545,const_544,bop_553,uop_556,])
func_563 = relay.Function([var_514,var_518,var_531,var_552,], output)
mod['func_563'] = func_563
mod = relay.transform.InferType()(mod)
var_564 = relay.var("var_564", dtype = "float32", shape = (10, 13))#candidate|564|(10, 13)|var|float32
var_565 = relay.var("var_565", dtype = "float32", shape = (10, 13))#candidate|565|(10, 13)|var|float32
var_566 = relay.var("var_566", dtype = "bool", shape = (1320,))#candidate|566|(1320,)|var|bool
var_567 = relay.var("var_567", dtype = "float32", shape = (10, 13))#candidate|567|(10, 13)|var|float32
output = func_563(var_564,var_565,var_566,var_567,)
func_568 = relay.Function([var_564,var_565,var_566,var_567,], output)
mutated_mod['func_568'] = func_568
mutated_mod = relay.transform.InferType()(mutated_mod)
var_590 = relay.var("var_590", dtype = "int8", shape = (3, 9))#candidate|590|(3, 9)|var|int8
var_591 = relay.var("var_591", dtype = "int8", shape = (3, 9))#candidate|591|(3, 9)|var|int8
bop_592 = relay.equal(var_590.astype('bool'), relay.reshape(var_591.astype('bool'), relay.shape_of(var_590))) # shape=(3, 9)
output = relay.Tuple([bop_592,])
output2 = relay.Tuple([bop_592,])
func_602 = relay.Function([var_590,var_591,], output)
mod['func_602'] = func_602
mod = relay.transform.InferType()(mod)
mutated_mod['func_602'] = func_602
mutated_mod = relay.transform.InferType()(mutated_mod)
func_602_call = mutated_mod.get_global_var('func_602')
var_604 = relay.var("var_604", dtype = "int8", shape = (3, 9))#candidate|604|(3, 9)|var|int8
var_605 = relay.var("var_605", dtype = "int8", shape = (3, 9))#candidate|605|(3, 9)|var|int8
call_603 = func_602_call(var_604,var_605,)
output = call_603
func_606 = relay.Function([var_604,var_605,], output)
mutated_mod['func_606'] = func_606
mutated_mod = relay.transform.InferType()(mutated_mod)
var_616 = relay.var("var_616", dtype = "float32", shape = (11, 11, 5))#candidate|616|(11, 11, 5)|var|float32
uop_617 = relay.asin(var_616.astype('float32')) # shape=(11, 11, 5)
bop_619 = relay.bitwise_xor(uop_617.astype('int64'), relay.reshape(var_616.astype('int64'), relay.shape_of(uop_617))) # shape=(11, 11, 5)
uop_622 = relay.acosh(var_616.astype('float32')) # shape=(11, 11, 5)
output = relay.Tuple([bop_619,uop_622,])
output2 = relay.Tuple([bop_619,uop_622,])
func_624 = relay.Function([var_616,], output)
mod['func_624'] = func_624
mod = relay.transform.InferType()(mod)
var_625 = relay.var("var_625", dtype = "float32", shape = (11, 11, 5))#candidate|625|(11, 11, 5)|var|float32
output = func_624(var_625)
func_626 = relay.Function([var_625], output)
mutated_mod['func_626'] = func_626
mutated_mod = relay.transform.InferType()(mutated_mod)
var_699 = relay.var("var_699", dtype = "uint32", shape = (10, 16, 5))#candidate|699|(10, 16, 5)|var|uint32
var_700 = relay.var("var_700", dtype = "uint32", shape = (10, 16, 5))#candidate|700|(10, 16, 5)|var|uint32
bop_701 = relay.multiply(var_699.astype('uint32'), relay.reshape(var_700.astype('uint32'), relay.shape_of(var_699))) # shape=(10, 16, 5)
bop_705 = relay.logical_and(bop_701.astype('bool'), relay.reshape(var_700.astype('bool'), relay.shape_of(bop_701))) # shape=(10, 16, 5)
uop_708 = relay.asin(bop_701.astype('float32')) # shape=(10, 16, 5)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
var_711 = relay.var("var_711", dtype = "float32", shape = (110,))#candidate|711|(110,)|var|float32
call_710 = relay.TupleGetItem(func_120_call(relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), ), 1)
call_712 = relay.TupleGetItem(func_126_call(relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), relay.reshape(var_711.astype('float32'), [11, 2, 5]), ), 1)
bop_720 = relay.logical_or(uop_708.astype('bool'), relay.reshape(bop_705.astype('bool'), relay.shape_of(uop_708))) # shape=(10, 16, 5)
uop_725 = relay.exp(uop_708.astype('float32')) # shape=(10, 16, 5)
func_467_call = mod.get_global_var('func_467')
func_470_call = mutated_mod.get_global_var('func_470')
var_728 = relay.var("var_728", dtype = "int8", shape = (880,))#candidate|728|(880,)|var|int8
call_727 = relay.TupleGetItem(func_467_call(relay.reshape(var_728.astype('int8'), [8, 11, 10]), relay.reshape(var_728.astype('bool'), [8, 11, 10]), ), 0)
call_729 = relay.TupleGetItem(func_470_call(relay.reshape(var_728.astype('int8'), [8, 11, 10]), relay.reshape(var_728.astype('bool'), [8, 11, 10]), ), 0)
uop_732 = relay.tan(bop_720.astype('float64')) # shape=(10, 16, 5)
var_735 = relay.var("var_735", dtype = "float64", shape = (10, 16, 5))#candidate|735|(10, 16, 5)|var|float64
bop_736 = relay.floor_mod(uop_732.astype('float64'), relay.reshape(var_735.astype('float64'), relay.shape_of(uop_732))) # shape=(10, 16, 5)
output = relay.Tuple([call_710,var_711,uop_725,call_727,var_728,bop_736,])
output2 = relay.Tuple([call_712,var_711,uop_725,call_729,var_728,bop_736,])
func_740 = relay.Function([var_699,var_700,var_711,var_728,var_735,], output)
mod['func_740'] = func_740
mod = relay.transform.InferType()(mod)
var_741 = relay.var("var_741", dtype = "uint32", shape = (10, 16, 5))#candidate|741|(10, 16, 5)|var|uint32
var_742 = relay.var("var_742", dtype = "uint32", shape = (10, 16, 5))#candidate|742|(10, 16, 5)|var|uint32
var_743 = relay.var("var_743", dtype = "float32", shape = (110,))#candidate|743|(110,)|var|float32
var_744 = relay.var("var_744", dtype = "int8", shape = (880,))#candidate|744|(880,)|var|int8
var_745 = relay.var("var_745", dtype = "float64", shape = (10, 16, 5))#candidate|745|(10, 16, 5)|var|float64
output = func_740(var_741,var_742,var_743,var_744,var_745,)
func_746 = relay.Function([var_741,var_742,var_743,var_744,var_745,], output)
mutated_mod['func_746'] = func_746
mutated_mod = relay.transform.InferType()(mutated_mod)
const_753 = relay.const(-8, dtype = "int32")#candidate|753|()|const|int32
var_754 = relay.var("var_754", dtype = "int32", shape = (11, 12))#candidate|754|(11, 12)|var|int32
bop_755 = relay.add(const_753.astype('int32'), var_754.astype('int32')) # shape=(11, 12)
uop_759 = relay.atanh(bop_755.astype('float32')) # shape=(11, 12)
uop_764 = relay.acosh(uop_759.astype('float32')) # shape=(11, 12)
uop_766 = relay.rsqrt(uop_759.astype('float64')) # shape=(11, 12)
bop_768 = relay.logical_or(uop_764.astype('bool'), relay.reshape(bop_755.astype('bool'), relay.shape_of(uop_764))) # shape=(11, 12)
uop_778 = relay.tan(uop_759.astype('float32')) # shape=(11, 12)
func_328_call = mod.get_global_var('func_328')
func_332_call = mutated_mod.get_global_var('func_332')
var_785 = relay.var("var_785", dtype = "float64", shape = (126,))#candidate|785|(126,)|var|float64
const_786 = relay.const([1.697930,7.535832,6.586056,-8.646329,3.872649,-0.655143,6.333337,4.875494,7.211973,-2.035117,-2.055396,1.098434,-1.389121,7.864255,8.080786,-8.873159,-9.325228,-7.882625,0.760732,3.290281,0.916053,5.721924,0.470332,0.564105,3.853931,3.507697,-8.519012,4.885667,-5.664046,-2.642679,9.651372,-7.341936,8.892865,6.704544,2.367110,6.880937,-1.952835,3.533191,8.871551,-9.358337,5.995813,-8.272044,-7.672358,-7.235177,-0.892482,9.455200,9.584862,-1.501526,-2.540034,3.916687,-9.923461,-2.260659,2.167664,-0.558465,-1.126936,-9.087754,6.151036,-6.439699,-4.716514,-9.082391,8.768701,-6.507410,-0.976979,5.791673,-9.101024,-6.199432,1.860321,1.035443,7.188607,8.994977,-4.327231,8.481449,6.744519,-6.710361,-7.668894,-4.996449,-5.248349,5.498823,1.889966,6.447421,7.804553,5.323689,-8.094336,-0.621286,-2.184157,1.107087,2.205525,-6.723330,-4.340025,-0.373498,-4.247930,2.713069,0.390824,3.335061,6.077637,-8.529974,-2.000255,-6.570495,-6.211922,-2.590792,-1.615061,6.393649,-4.395974,-8.145914,-3.050189,-5.948541,-1.923833,4.929635,-9.117588,-3.415869,3.407889,-8.705569,-2.818662,-8.095993,4.555259,-4.851719,2.663367,8.041366,9.644441,-3.965470,-0.720353,-5.507457,-3.677882,-3.378596,-7.812449,0.622852,9.602393,-2.473217,-1.897164,-1.113711,0.575739,2.510014,8.197454,9.488399,6.814161,-6.912647,-7.461490,-5.105113,2.391988,1.476133,5.234073,3.722629,-6.287308,1.272823,8.428808,2.367986,-6.978746,5.356284,8.992476,-0.463105,-0.370011,1.711019,-2.035118,-6.725798,-4.093502,8.593003,-6.838370,5.285576,-8.630642,8.126313,-5.994938,3.947330,7.360940,-2.437637,2.135877,7.045184,-5.155103,-5.380121,-4.022903,-7.035004,-8.021095,6.986234,-4.833665,8.653867,9.021934,-4.891821,0.137569,-7.828009,-7.411861,-9.792609,6.396602,-1.960759,-9.659881,-0.089448,2.162553,8.364837,7.625329,-7.617136,0.632245,-7.623075,-5.388683,6.648436,-1.785892,-5.354213,-1.382666,-4.863657,2.797081,5.606828,6.076589,-9.984972,-7.526412,-8.554513,-0.414932,-7.090449,4.600764,-6.761707,4.638847,8.584414,6.180246,0.736662,-5.978483,-7.754234,-3.376078,3.316370,2.380268,-5.178150,-5.528409,0.199272,0.881532,-0.375545,8.508049,-5.109752,-3.433985,-4.559582,7.573146,-9.048402,-3.705388,4.862408,-2.041016,-2.761503,1.703495,-9.365787,-6.401623,-7.434734,3.154638,-1.981644,-9.119274,8.146016,-8.211021,-4.070021], dtype = "float32")#candidate|786|(240,)|const|float32
call_784 = relay.TupleGetItem(func_328_call(relay.reshape(var_785.astype('float64'), [9, 14]), relay.reshape(const_786.astype('float32'), [240,]), relay.reshape(var_785.astype('float64'), [9, 14]), ), 5)
call_787 = relay.TupleGetItem(func_332_call(relay.reshape(var_785.astype('float64'), [9, 14]), relay.reshape(const_786.astype('float32'), [240,]), relay.reshape(var_785.astype('float64'), [9, 14]), ), 5)
output = relay.Tuple([uop_766,bop_768,uop_778,call_784,var_785,const_786,])
output2 = relay.Tuple([uop_766,bop_768,uop_778,call_787,var_785,const_786,])
func_790 = relay.Function([var_754,var_785,], output)
mod['func_790'] = func_790
mod = relay.transform.InferType()(mod)
var_791 = relay.var("var_791", dtype = "int32", shape = (11, 12))#candidate|791|(11, 12)|var|int32
var_792 = relay.var("var_792", dtype = "float64", shape = (126,))#candidate|792|(126,)|var|float64
output = func_790(var_791,var_792,)
func_793 = relay.Function([var_791,var_792,], output)
mutated_mod['func_793'] = func_793
mutated_mod = relay.transform.InferType()(mutated_mod)
var_822 = relay.var("var_822", dtype = "float32", shape = (6, 4, 7))#candidate|822|(6, 4, 7)|var|float32
uop_823 = relay.atanh(var_822.astype('float32')) # shape=(6, 4, 7)
uop_825 = relay.sqrt(uop_823.astype('float32')) # shape=(6, 4, 7)
func_602_call = mod.get_global_var('func_602')
func_606_call = mutated_mod.get_global_var('func_606')
const_828 = relay.const([10,-4,-1,3,7,9,6,-1,-4,2,10,-8,-3,-10,10,1,5,5,8,-5,-10,7,2,-2,8,-8,8], dtype = "int8")#candidate|828|(27,)|const|int8
call_827 = relay.TupleGetItem(func_602_call(relay.reshape(const_828.astype('int8'), [3, 9]), relay.reshape(const_828.astype('int8'), [3, 9]), ), 0)
call_829 = relay.TupleGetItem(func_606_call(relay.reshape(const_828.astype('int8'), [3, 9]), relay.reshape(const_828.astype('int8'), [3, 9]), ), 0)
output = relay.Tuple([uop_825,call_827,const_828,])
output2 = relay.Tuple([uop_825,call_829,const_828,])
func_831 = relay.Function([var_822,], output)
mod['func_831'] = func_831
mod = relay.transform.InferType()(mod)
var_832 = relay.var("var_832", dtype = "float32", shape = (6, 4, 7))#candidate|832|(6, 4, 7)|var|float32
output = func_831(var_832)
func_833 = relay.Function([var_832], output)
mutated_mod['func_833'] = func_833
mutated_mod = relay.transform.InferType()(mutated_mod)
var_880 = relay.var("var_880", dtype = "float32", shape = (4,))#candidate|880|(4,)|var|float32
uop_881 = relay.acosh(var_880.astype('float32')) # shape=(4,)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
var_884 = relay.var("var_884", dtype = "bool", shape = (1, 1320))#candidate|884|(1, 1320)|var|bool
call_883 = relay.TupleGetItem(func_215_call(relay.reshape(var_884.astype('bool'), [10, 11, 12]), relay.reshape(var_884.astype('bool'), [10, 11, 12]), ), 0)
call_885 = relay.TupleGetItem(func_219_call(relay.reshape(var_884.astype('bool'), [10, 11, 12]), relay.reshape(var_884.astype('bool'), [10, 11, 12]), ), 0)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
call_886 = relay.TupleGetItem(func_215_call(relay.reshape(call_883.astype('bool'), [10, 11, 12]), relay.reshape(call_883.astype('bool'), [10, 11, 12]), ), 0)
call_887 = relay.TupleGetItem(func_219_call(relay.reshape(call_883.astype('bool'), [10, 11, 12]), relay.reshape(call_883.astype('bool'), [10, 11, 12]), ), 0)
func_790_call = mod.get_global_var('func_790')
func_793_call = mutated_mod.get_global_var('func_793')
const_890 = relay.const([-3,8,1,-8,3,-2,4,9,5,-7,-2,-8,10,-4,-8,-5,-3,4,5,8,-9,-4,10,-7,5,-7,4,1,-9,-8,8,-7,8,-3,-7,-2,-3,-1,10,-7,-1,-9,10,-9,-6,5,10,-3,5,-1,8,1,10,4,8,-4,-6,6,-4,2,-1,4,-8,2,-2,8,-8,3,8,9,-1,-9,-9,-10,9,5,-10,3,-3,-3,-10,8,9,8,-9,7,4,-6,10,-8,-1,-9,1,8,9,-10,4,6,9,5,3,-9,-8,9,-4,-7,9,7,9,5,-9,-4,-5,8,-4,-4,-7,-3,-10,2,-1,-1,6,10,3,5,10,-7,-2,9,-10,-5], dtype = "int32")#candidate|890|(132,)|const|int32
const_891 = relay.const([7.675004,-4.958861,-0.934326,-1.753712,8.275795,3.999971,-0.299994,5.861683,-8.717535,7.387689,7.379105,6.809596,9.730145,-7.131201,1.282526,-3.799354,-5.101412,9.122154,-0.558155,-7.397353,-5.310332,1.342273,-4.475434,-2.560297,3.521257,9.863171,6.149403,0.333304,-4.664972,-2.035372,-7.160916,-1.095020,6.272967,5.252689,0.757066,4.864313,5.081442,-7.850293,1.137378,-6.354985,2.037049,-7.528818,9.967228,4.163540,6.229843,7.889993,-9.429230,-5.672680,7.868966,5.917127,-9.000867,8.307884,6.479211,7.834158,-6.805726,-1.393788,9.425607,5.698900,-5.750895,-2.665245,-9.700838,2.120301,4.498617,-9.966558,-3.848373,8.223637,6.745261,-9.526720,1.127938,9.273520,-4.155180,1.811529,-2.512336,-4.614486,5.057620,-4.388378,-6.037440,9.555960,-0.476171,9.682753,6.260539,5.893932,7.078432,-2.123143,0.332288,-5.583678,8.089091,0.166743,-3.817096,-7.039022,-3.750674,7.039756,4.026468,-8.095987,8.977224,-5.428474,-3.491109,2.878974,3.657664,-6.537354,1.149844,-0.244293,-6.757629,3.035607,9.539033,1.722327,-9.232358,1.035078,1.706771,3.117202,3.854104,3.787690,7.720888,3.240893,-2.118341,8.183040,-2.934784,-2.996338,7.698639,-3.418844,-8.003847,-9.925875,-2.919067,-6.542104,5.689206,-7.051931], dtype = "float64")#candidate|891|(126,)|const|float64
call_889 = relay.TupleGetItem(func_790_call(relay.reshape(const_890.astype('int32'), [11, 12]), relay.reshape(const_891.astype('float64'), [126,]), ), 1)
call_892 = relay.TupleGetItem(func_793_call(relay.reshape(const_890.astype('int32'), [11, 12]), relay.reshape(const_891.astype('float64'), [126,]), ), 1)
func_328_call = mod.get_global_var('func_328')
func_332_call = mutated_mod.get_global_var('func_332')
var_895 = relay.var("var_895", dtype = "float32", shape = (240,))#candidate|895|(240,)|var|float32
call_894 = relay.TupleGetItem(func_328_call(relay.reshape(const_891.astype('float64'), [9, 14]), relay.reshape(var_895.astype('float32'), [240,]), relay.reshape(const_891.astype('float64'), [9, 14]), ), 4)
call_896 = relay.TupleGetItem(func_332_call(relay.reshape(const_891.astype('float64'), [9, 14]), relay.reshape(var_895.astype('float32'), [240,]), relay.reshape(const_891.astype('float64'), [9, 14]), ), 4)
output = relay.Tuple([uop_881,call_883,var_884,call_886,call_889,const_890,const_891,call_894,var_895,])
output2 = relay.Tuple([uop_881,call_885,var_884,call_887,call_892,const_890,const_891,call_896,var_895,])
func_899 = relay.Function([var_880,var_884,var_895,], output)
mod['func_899'] = func_899
mod = relay.transform.InferType()(mod)
var_900 = relay.var("var_900", dtype = "float32", shape = (4,))#candidate|900|(4,)|var|float32
var_901 = relay.var("var_901", dtype = "bool", shape = (1, 1320))#candidate|901|(1, 1320)|var|bool
var_902 = relay.var("var_902", dtype = "float32", shape = (240,))#candidate|902|(240,)|var|float32
output = func_899(var_900,var_901,var_902,)
func_903 = relay.Function([var_900,var_901,var_902,], output)
mutated_mod['func_903'] = func_903
mutated_mod = relay.transform.InferType()(mutated_mod)
var_913 = relay.var("var_913", dtype = "int64", shape = ())#candidate|913|()|var|int64
var_914 = relay.var("var_914", dtype = "int64", shape = (10, 2, 7))#candidate|914|(10, 2, 7)|var|int64
bop_915 = relay.multiply(var_913.astype('int64'), var_914.astype('int64')) # shape=(10, 2, 7)
func_328_call = mod.get_global_var('func_328')
func_332_call = mutated_mod.get_global_var('func_332')
const_924 = relay.const([[7.610363,5.970316,-2.354317,8.848845,9.647237,-6.276953,1.890096,-4.372270,-4.957242,4.237820,0.164435,-4.586213,-5.009710,-2.895727,0.428641,-0.499088,8.335659,9.786506,2.094435,5.651973,-8.057338,-9.163110,-5.470380,8.554136,1.577142,-8.653879,-0.548762,7.062905,-1.377537,-2.143768,-9.531945,-1.690371,-6.698260,0.868608,-7.436420,-1.389061,-8.606041,-6.660478,6.679086,8.278666,-3.853512,2.312971],[-8.912484,8.031496,0.036123,7.965710,-2.289021,3.523734,8.212114,-8.387629,8.842972,-9.094638,7.542078,-8.452595,-0.710888,5.397464,4.434705,-0.136132,-8.056111,-2.971044,2.105246,7.972657,8.444234,-0.466707,-4.779342,-6.765454,7.777192,6.597139,-8.761003,-7.624949,-2.589335,0.839206,-2.710199,8.506820,-2.560174,-5.420761,8.335999,3.437033,-9.387947,9.974029,8.417747,3.500179,-7.405438,9.376618],[-1.465492,-7.845501,-2.102002,4.014073,4.568645,6.704933,-7.167925,4.758390,9.034740,2.272303,-8.392804,-6.640232,-9.130438,-7.595367,5.497747,7.387636,2.279512,-7.437641,-0.647010,0.101225,-0.956511,6.298596,-7.156605,3.436952,3.116850,-0.945492,2.466189,7.738449,-7.010252,5.175144,-5.696944,1.493254,-8.148844,3.208360,6.969479,-4.433654,7.739207,3.493874,-1.387375,3.595881,1.022188,5.862517]], dtype = "float64")#candidate|924|(3, 42)|const|float64
const_925 = relay.const([[3.642486,8.840599,8.137496,-7.537216,-1.316273,2.433588,1.491913,-0.103203,-0.253935,2.943072,-9.978661,6.431443,-8.337453,8.369937,-6.083325,-6.889456,-4.762284,3.398466,9.978718,-6.426255,5.589170,-4.667562,-6.740737,1.583280,2.079078,-7.842713,8.540294,5.415258,-3.530981,3.088618,-3.057189,-1.983916,6.227649,-1.929720,0.303585,-9.625433,-4.256766,-5.353298,3.446009,-9.594454,-3.199349,-2.677173,-9.724944,6.653343,-6.577711,-6.983481,-4.613083,1.009281,-0.616564,-6.156782,-5.921499,-2.975238,7.423712,4.765782,2.571997,8.070985,1.059134,7.620763,3.109389,-4.386773,3.207043,6.979880,-0.435959,9.282681,-4.197355,-1.120295,-0.669358,-6.064860,1.795825,-6.381906,8.565933,5.819764,-1.728096,-8.555272,1.881553,0.572982,6.215986,4.446188,-9.633790,-3.417227,-8.660392,7.256252,-3.460750,9.882685,-0.141508,-9.943011,0.492821,-6.400151,-2.515513,-4.368476,3.042259,5.770391,-5.898729,-5.526526,6.545477,5.805003,-2.904794,7.575617,-4.543403,-0.632942,-5.562590,-4.052127,-2.700521,6.373385,-5.034096,3.242804,-3.329287,-3.895255,1.293332,1.162287,-9.241506,-4.578440,1.591358,2.618043,-9.690808,-1.926023,-3.655412,2.252332,7.244023,0.560157,-4.576076,-2.307579,4.492364,8.047947,8.976250,-1.801146,6.139461,-0.282600,-1.123020,3.149000,-9.516025,5.131897,-2.291999,5.061728,-4.835320,7.493365,7.769745,2.189081,9.485031,0.067507,5.660196,-1.322045,7.238077,3.192687,-1.680405,-6.614644,7.208952,3.230603,-8.581134,1.290796,9.800476,8.086891,-7.928297,-3.426245,0.534159,6.006906,4.296528,-9.681482,-3.852096,-6.942799,-7.504785,4.595202,-1.213389,4.361191,-5.677469,6.349775,-6.777476,-1.928411,1.254288,-5.976170,5.380518,5.271617,3.519880,2.094133,-8.654775,-8.949433,-0.041753,-1.120324,-9.811588,-9.189227,-9.906434,-2.826093,-7.381170,-9.408151,0.376795,9.028415,-0.378367,0.936431,-8.220455,9.621793,-3.761353,2.345433,-4.786298,-0.079278,0.183886,1.190715,-9.814162,-7.712872,-0.028271,4.666404,9.718175,-2.144874,3.582802,8.563178,-5.363643,-7.614579,-6.313634,-6.360054,4.695888,3.026206,0.644814,3.979096,3.292238,-8.749262,-4.133666,0.137924,-8.901734,6.350758,-8.609008,9.422168,6.578915,9.410491,9.421594,-5.207153,-5.635469,-7.834337,-0.346785,-1.857959,6.553559,-8.603736,-9.939966,-6.987925,8.770482,-3.053952,-2.407923,8.248687,-4.303941,-6.342632,-7.922412,-2.600472]], dtype = "float32")#candidate|925|(1, 240)|const|float32
call_923 = relay.TupleGetItem(func_328_call(relay.reshape(const_924.astype('float64'), [9, 14]), relay.reshape(const_925.astype('float32'), [240,]), relay.reshape(const_924.astype('float64'), [9, 14]), ), 0)
call_926 = relay.TupleGetItem(func_332_call(relay.reshape(const_924.astype('float64'), [9, 14]), relay.reshape(const_925.astype('float32'), [240,]), relay.reshape(const_924.astype('float64'), [9, 14]), ), 0)
bop_927 = relay.floor_divide(var_913.astype('float32'), const_924.astype('float32')) # shape=(3, 42)
bop_933 = relay.minimum(var_914.astype('float32'), relay.reshape(bop_915.astype('float32'), relay.shape_of(var_914))) # shape=(10, 2, 7)
var_936 = relay.var("var_936", dtype = "int64", shape = (5,))#candidate|936|(5,)|var|int64
bop_937 = relay.greater(var_913.astype('bool'), var_936.astype('bool')) # shape=(5,)
uop_942 = relay.erf(call_923.astype('float32')) # shape=(11, 2, 5)
uop_944 = relay.erf(call_926.astype('float32')) # shape=(11, 2, 5)
uop_946 = relay.log(uop_942.astype('float64')) # shape=(11, 2, 5)
uop_948 = relay.log(uop_944.astype('float64')) # shape=(11, 2, 5)
uop_952 = relay.rsqrt(uop_946.astype('float64')) # shape=(11, 2, 5)
uop_954 = relay.rsqrt(uop_948.astype('float64')) # shape=(11, 2, 5)
uop_957 = relay.exp(uop_946.astype('float32')) # shape=(11, 2, 5)
uop_959 = relay.exp(uop_948.astype('float32')) # shape=(11, 2, 5)
bop_967 = relay.add(uop_942.astype('uint64'), bop_937.astype('uint64')) # shape=(11, 2, 5)
bop_970 = relay.add(uop_944.astype('uint64'), bop_937.astype('uint64')) # shape=(11, 2, 5)
bop_971 = relay.bitwise_xor(uop_957.astype('uint16'), relay.reshape(uop_952.astype('uint16'), relay.shape_of(uop_957))) # shape=(11, 2, 5)
bop_974 = relay.bitwise_xor(uop_959.astype('uint16'), relay.reshape(uop_954.astype('uint16'), relay.shape_of(uop_959))) # shape=(11, 2, 5)
output = relay.Tuple([const_925,bop_927,bop_933,bop_967,bop_971,])
output2 = relay.Tuple([const_925,bop_927,bop_933,bop_970,bop_974,])
func_975 = relay.Function([var_913,var_914,var_936,], output)
mod['func_975'] = func_975
mod = relay.transform.InferType()(mod)
mutated_mod['func_975'] = func_975
mutated_mod = relay.transform.InferType()(mutated_mod)
func_975_call = mutated_mod.get_global_var('func_975')
var_977 = relay.var("var_977", dtype = "int64", shape = ())#candidate|977|()|var|int64
var_978 = relay.var("var_978", dtype = "int64", shape = (10, 2, 7))#candidate|978|(10, 2, 7)|var|int64
var_979 = relay.var("var_979", dtype = "int64", shape = (5,))#candidate|979|(5,)|var|int64
call_976 = func_975_call(var_977,var_978,var_979,)
output = call_976
func_980 = relay.Function([var_977,var_978,var_979,], output)
mutated_mod['func_980'] = func_980
mutated_mod = relay.transform.InferType()(mutated_mod)
var_982 = relay.var("var_982", dtype = "float32", shape = (16, 16, 14))#candidate|982|(16, 16, 14)|var|float32
const_983 = relay.const([[[8.199100,0.257917,3.823635,6.371646,6.364883,-5.666800,-5.517663,-8.814682,5.745724,3.983441,9.427510,-9.927132,-3.760935,9.584671],[7.468382,5.561920,-5.870543,-9.203653,2.503830,-1.993080,8.071403,-7.822198,-7.681733,-1.116574,-8.473616,-3.081780,4.731512,9.113977],[9.753228,-0.584097,2.550758,1.068422,-0.234957,8.564853,8.378283,0.578339,8.157450,5.628943,-0.449575,6.308294,4.727722,-8.873674],[8.262760,4.198949,-1.646861,4.101346,4.167144,3.971167,7.345666,-7.688519,-7.984787,6.587559,9.335657,-4.177451,-4.710373,-1.114982],[5.228578,-8.805501,-6.557474,7.955154,4.760506,7.645186,5.061199,8.113881,5.929829,-9.621373,1.804649,8.158401,-5.989531,-4.463277],[-0.971659,-1.338081,8.766729,-3.512053,-2.610396,-3.000876,-2.515765,2.166738,-0.330741,-5.431076,-9.273416,8.412192,-2.048311,-7.298885],[-2.670699,3.002093,-0.446020,-6.624933,0.950604,1.546923,-1.123583,2.349076,7.209887,2.180802,9.210161,-4.422548,-0.271988,0.663052],[-9.502949,6.214565,-0.629683,5.045706,-5.672170,-9.131549,-8.963041,-6.855880,9.301980,-5.147559,-0.775837,-8.672179,7.654752,5.709664],[6.907812,7.324987,-8.060012,-4.072826,3.129026,8.986799,-2.275985,1.982935,-5.682093,0.649264,-2.075461,-1.741378,9.205787,-1.718361],[4.772910,-6.302596,6.646584,6.984538,-4.306337,-3.297799,7.144745,-9.094894,-1.102320,8.478757,6.473603,-6.787375,-0.776769,-3.333835],[-3.192430,-9.105893,-9.757122,-8.919745,-3.113498,8.750954,4.018940,6.970102,-6.184390,-6.424473,4.247238,0.838721,0.052446,-2.139721],[-6.273181,6.110324,-8.061015,1.265067,9.788993,-6.964288,5.072466,-8.563761,5.879011,5.009174,9.871674,-5.595152,5.497205,8.261299],[-0.704343,-0.019193,-2.245936,5.976819,-5.323879,7.341396,-4.994968,4.972630,-2.778227,-8.312231,6.805091,4.522879,-7.913347,-6.862388],[9.016769,-7.288552,5.694357,-1.175619,-0.453014,-0.928416,-5.994787,-1.401932,4.559782,-1.626192,-7.352186,-6.433817,1.926745,-6.521844],[5.262837,3.275043,6.015148,-6.854614,1.722296,-3.457716,-0.920586,-2.933463,5.448316,5.867684,-6.723646,-5.283946,8.869780,5.556983],[-8.934190,-0.807602,-2.048624,-6.553748,3.500551,-9.384303,6.969357,-8.203032,-5.243731,6.857523,-3.099441,-7.776385,-8.944936,8.347238]],[[1.260062,-2.286337,-2.891437,-4.920295,7.062162,-0.474959,5.882495,5.579740,-0.565868,3.329580,-5.509597,-2.853671,7.539816,2.480759],[6.343373,9.648004,0.704536,-7.000580,-9.108385,-7.215792,2.412019,-8.881341,-2.995123,6.760646,-8.712780,4.111090,1.150911,1.216694],[-7.656866,4.402594,7.739212,4.712216,-7.009058,-3.344104,-5.573473,-7.252960,-4.811226,-4.839619,-7.404946,3.774209,-9.909522,6.069590],[-1.065435,6.036767,-0.791391,-5.302431,6.054844,-4.069467,2.964164,3.989373,-6.166584,8.011410,-6.074637,3.081353,8.937685,7.662088],[0.894005,-2.968633,2.492823,-1.272424,-2.643610,9.420681,-1.583050,5.362088,9.986197,8.652491,8.287603,-1.727474,8.140464,-4.717729],[-5.697444,-5.523574,7.096761,1.967380,-8.396945,-1.359389,-8.050103,0.513544,9.334493,-5.302659,6.533422,-9.151971,-2.048613,-8.808150],[-5.425735,-3.426183,6.927819,-0.465883,0.213965,8.420503,-4.072440,-0.404804,-7.773242,3.674217,1.770935,-4.214038,-6.785213,0.581048],[1.297281,-9.139355,-0.758623,-6.744102,5.160784,-2.965575,-1.299375,9.509145,7.856697,6.105604,0.085377,-2.287191,-8.136485,8.377511],[5.349234,7.368905,-2.515597,4.434928,-6.432875,-8.078537,-4.804568,1.349271,8.858947,1.161242,-9.722567,-5.512420,-3.145503,-8.869346],[7.512545,7.160435,-6.408142,9.840528,-5.927707,-1.252129,-2.599609,-2.865194,2.019375,8.149654,-7.262441,-6.067388,8.794148,-3.984815],[8.660903,-1.982694,-5.441247,9.327243,-3.233320,-7.582197,8.271449,4.668115,3.779736,9.700616,9.922495,1.944669,2.356072,2.843843],[-5.340568,-3.881589,-1.398553,-1.518688,-7.066721,-1.777635,-9.421879,4.065674,6.366159,5.118880,5.145374,5.965827,5.013355,-0.571049],[-5.572693,6.580647,-7.271542,-0.217597,-0.816089,0.997410,8.194590,1.221418,-6.773408,1.340505,-3.205503,4.155624,7.694120,-2.789241],[6.628568,-9.392663,-2.231020,9.272236,9.204554,-3.145961,-1.869722,1.948800,2.518698,-7.183102,2.036689,2.280661,-4.680283,9.319446],[7.800487,6.534615,0.649206,0.240521,2.588087,-5.006120,-6.815837,8.843312,-8.642539,-5.718162,2.781494,-1.735636,2.607904,-5.997722],[-6.260582,9.866739,-1.580310,-1.438219,-3.751563,9.048633,-4.107171,5.530369,-1.484512,-5.075073,0.753660,-1.898566,9.814074,0.890588]],[[-6.570667,-9.241095,-4.975482,7.713613,-2.616529,-5.211894,6.688510,2.748855,7.300555,-2.532247,2.815427,-5.463409,-1.128449,6.641547],[0.382559,-7.587864,1.862526,-7.000277,-7.412212,-3.214043,-3.452292,-9.659839,-9.019011,2.642107,-3.086614,-1.414790,-6.180150,-3.799321],[7.448928,-9.259407,-4.706859,8.676828,-2.221820,6.119298,8.820389,9.817687,-1.181228,3.402110,7.116825,-5.259486,-2.690410,1.650998],[-3.426958,5.073367,-3.368891,-4.248280,-6.619316,-0.058921,-9.350859,-3.665766,6.224838,9.508768,1.115404,-6.673692,7.099008,-2.434923],[7.191871,1.226755,7.615862,-7.490925,-4.582352,-8.906066,-9.566864,7.772318,1.668962,-3.614967,-3.177239,-1.780938,-0.224578,-0.073175],[-2.806982,9.582806,4.850612,7.035308,-0.755288,-3.029098,-2.300399,3.589514,9.753110,4.547217,-4.814232,3.392451,4.270802,-9.132843],[-1.698259,-6.615352,2.661310,3.837945,-2.376511,2.351739,-9.214249,8.674552,7.581270,9.532344,-0.096111,7.123319,-9.490827,6.601381],[-7.517564,0.888512,-1.787041,5.527035,3.790481,-9.014742,8.871684,-6.742595,5.568287,7.867713,4.259146,1.941137,0.121064,-1.593852],[0.052863,-4.413080,-3.365905,6.267418,5.599473,-7.696740,6.812855,-9.979841,1.689923,-7.023921,-1.136763,2.995697,-3.271919,6.608793],[1.264479,2.525079,3.280603,-4.909558,0.649745,6.660096,7.598184,8.198523,-1.240584,6.663237,5.228857,4.420287,5.648076,3.855721],[1.161503,-2.446203,-8.865804,-1.199079,3.301781,8.991684,-4.946222,7.603340,5.395841,-8.547736,-9.462232,1.335244,3.814865,4.313290],[5.609760,-3.220814,-3.303282,8.965776,-5.624061,-1.363909,-4.505993,-7.212345,-3.071081,-3.721439,-7.948309,-3.925740,6.328287,-5.881209],[-5.516358,6.757763,-9.990764,-1.900946,-5.765843,-2.020601,7.244208,-4.576789,-8.297424,-5.500309,-4.687949,-3.178224,4.200519,0.251931],[-1.749980,3.575286,-5.427732,-9.051628,-9.906443,9.955914,-4.397184,4.254037,-1.159344,-3.852935,-4.594282,-8.908237,9.678488,-9.052589],[6.470196,7.128966,5.444244,-5.561545,-5.431833,0.797350,9.883215,1.049545,-1.398195,-2.921716,0.446818,0.588815,8.422111,-2.951350],[0.217566,-2.300772,4.997625,-4.980913,-7.667746,-7.534839,-2.999480,-1.182783,-7.747832,5.908596,-9.474167,7.697218,-7.873404,-0.272217]],[[-9.236352,2.078487,2.308534,-6.251260,2.894469,7.598177,5.256366,-3.938788,6.691666,-3.926816,3.860670,-4.220332,5.037686,8.538555],[2.458702,0.539126,-7.717920,2.638730,-7.418006,-2.126706,-2.120319,-1.697375,5.346870,2.315784,3.744793,-4.449249,-0.676415,-0.179314],[7.255914,5.405328,2.835372,-2.043369,3.314907,0.061292,-4.504246,-8.197228,-9.324717,-7.489280,1.880943,-4.089252,0.303833,1.444882],[-4.652847,9.263387,3.202192,-5.398924,-2.013782,3.142156,-4.995502,0.815231,-4.202568,8.989153,-2.790119,4.351677,-4.779567,-7.048190],[-6.001105,5.783953,-0.749545,-9.274115,0.494364,-0.030414,-4.898857,-8.098597,6.581253,4.477451,8.202596,-7.372727,-1.884783,3.891897],[2.787745,4.850472,-2.514171,-1.853620,3.373136,-0.381239,6.945669,-5.163399,-1.060307,-0.413915,6.111230,-6.906471,4.038861,-1.028112],[-4.907160,-3.831469,-2.613156,9.709389,-0.852794,2.462332,6.105599,-2.226053,-9.968203,-0.703555,5.198609,-0.388018,-3.496735,9.433445],[4.579491,5.623289,-1.084123,-8.222743,7.779903,4.688659,2.447027,-9.886903,-6.775716,-3.766197,3.181606,-3.850869,1.994751,1.178559],[4.578942,8.530288,5.218602,-9.104117,-1.450228,-1.206721,-0.216848,7.027444,-5.585432,-9.105504,8.702431,0.012136,1.398123,3.033658],[0.164325,-7.088979,-0.177755,2.203950,5.716583,6.962255,-3.805633,5.046366,7.023009,8.034634,-5.429424,-9.635036,-3.278021,9.046716],[5.778528,-3.401941,6.254516,2.605762,6.479339,2.012041,-8.357556,-7.620333,-7.363529,-4.803589,1.246026,2.815091,0.808632,0.058497],[-3.346980,-0.533008,4.521914,-6.664061,4.705471,-2.537099,-6.205960,4.814370,-9.293934,-1.887881,-6.779396,-3.108939,-2.881925,6.223303],[-5.776838,5.815183,8.312777,-7.200507,-5.159803,-8.275579,4.631481,-0.273360,-8.990190,-2.226033,-1.350337,-1.298587,9.677495,-8.377190],[-0.725992,-6.027448,-2.541421,6.088375,0.034219,-2.223824,-0.371267,4.799208,-9.429145,-7.341169,-8.873071,0.768241,6.672652,4.694426],[0.558389,-6.297269,7.750455,-7.920260,-3.589142,-8.265515,8.388567,9.019283,9.062940,0.735000,0.605680,-8.267237,5.747171,-9.549814],[1.517676,5.522913,6.566314,3.188655,-4.845173,-3.032938,0.954286,-8.726840,0.799967,0.639008,3.909445,-1.097026,-5.036041,-7.064617]],[[0.312652,-2.407694,-1.381062,9.054074,7.885843,-3.426910,-2.456359,-1.133952,6.281092,-4.885699,-9.525557,0.172095,0.743493,-2.113190],[3.466477,4.198297,5.840807,3.070110,1.919878,8.590342,4.659312,4.613653,-9.948011,-7.665316,-9.899346,0.108817,-4.681977,-6.723282],[-2.671034,6.488801,8.542288,-1.337911,-5.386384,-2.886189,-8.429360,-7.996237,-4.594035,4.890666,0.481105,-7.968520,9.863249,-1.085991],[4.283210,0.922263,-1.635702,9.089245,3.806034,-3.540045,0.955019,1.443786,7.317026,-2.781322,8.435716,-0.602071,9.426777,0.854796],[-0.657402,-8.530220,-0.933405,-7.147203,-9.160018,2.750095,1.591576,4.400884,-5.857799,7.793467,-5.175291,9.719864,9.992675,-6.963546],[1.536398,1.641981,8.616274,3.445702,5.816838,-7.394338,0.238635,1.642685,-8.318606,0.525825,-9.603581,-3.054121,-8.565250,-7.618134],[7.716633,-9.539115,-1.395416,3.729955,9.506546,-7.770318,-1.356070,-2.403005,-8.757969,-8.481974,-9.825791,7.924106,-8.237369,0.819888],[-8.358147,5.163904,-2.814801,-0.221026,8.579083,0.955700,-3.685677,-3.224071,-0.675044,-4.518954,-7.136590,8.292401,-3.221032,4.841647],[-3.885771,-8.797620,6.683117,1.990796,5.434874,-3.978093,3.549879,-2.471536,8.412630,-8.984228,2.700057,-2.580092,4.881705,9.056208],[6.153625,9.644522,0.003652,-3.387290,3.053861,8.554334,3.881230,9.294435,5.392494,-9.186123,-3.891478,-2.961584,7.064873,-2.641820],[9.819056,-1.799355,-5.292419,-2.680849,1.956918,-2.104729,-8.892117,-4.529626,-7.931709,-2.350976,7.611771,-6.047114,2.937762,2.531442],[-5.773250,-6.364983,0.299184,-4.449097,-0.455973,-9.546343,0.473735,-1.368498,-4.749271,5.346044,-6.398893,-4.092533,-9.106309,-1.482932],[1.293658,-0.512659,8.912764,1.768896,5.442292,-5.528773,6.823122,-7.315411,-8.584286,-7.615683,4.857505,3.782402,-0.975792,5.295187],[-2.517237,-8.580550,7.986583,6.577815,5.675651,-6.992750,-1.764211,-5.760645,6.766083,-3.726749,6.696763,-8.371397,-4.112579,9.296309],[4.697267,1.311065,-4.584934,0.895775,-7.379050,-2.662296,-7.422552,2.547993,2.443290,-5.334558,-6.429298,-8.107748,4.470777,-8.333789],[-2.730512,-0.254216,-2.410767,-7.527927,-0.139657,-6.374311,-9.959397,-6.381523,6.387952,-6.618940,-4.836919,2.630506,1.856897,-0.711259]],[[-6.539505,3.680497,-6.093379,-7.906234,6.377265,3.658199,7.510842,7.571898,-6.758031,-1.421296,7.489166,7.179415,1.576432,-3.809808],[-6.551079,9.465923,9.866166,-2.143754,8.221736,-5.931152,1.592450,-0.624245,4.386366,-2.364131,8.332985,-4.030626,4.179244,-5.760298],[-7.447867,-6.806495,-0.783666,-8.994219,-6.514124,5.472858,0.259872,-7.744204,-3.286340,8.970047,-3.849099,5.858305,-8.378024,-8.294205],[2.581078,-9.565221,0.631684,7.027632,9.805850,2.274650,-1.757202,6.426048,-8.122062,-7.184722,-7.606708,7.483077,2.230107,-5.096952],[-6.030867,-3.334219,-6.599782,7.119207,3.803520,1.283703,-7.848614,-7.023782,3.220484,4.754651,1.784172,-6.844376,9.721305,4.312805],[1.646380,1.123050,-2.557764,9.827426,-3.728360,2.867178,0.575285,-9.956920,8.420599,-9.529415,-6.649259,9.877161,6.646286,3.245767],[0.706033,-1.294705,1.273742,-6.232899,6.689382,-3.168569,-2.992737,8.403285,-3.867363,-3.376721,-4.827640,-6.780695,-3.503752,-5.510788],[-9.320644,-3.703740,-2.654966,-6.622091,-6.254085,-8.505243,-4.182177,-0.631846,-7.810304,1.083602,0.900175,-9.020084,-1.178915,9.499028],[-4.626751,9.295477,6.705561,-7.213005,-8.138719,8.352573,7.123774,-6.551508,-3.816664,2.519290,-9.210350,6.963816,-6.047545,5.618765],[2.078412,3.100653,-0.716284,-5.522356,-1.422162,-5.530209,0.105194,-7.358427,8.673574,-6.353213,0.302635,-0.168888,-1.871139,2.716453],[7.552262,-9.350696,5.512581,1.166269,-0.726328,-9.331327,-1.846209,4.089634,-8.205084,4.993654,-4.878457,-9.473728,1.684683,1.500127],[-2.402103,-0.831225,1.347497,-3.513299,-0.879194,-2.504371,8.219638,2.461028,6.901721,0.784567,-3.791717,9.781401,4.069566,-2.591061],[-5.719183,-9.037375,8.188626,2.855493,5.692909,6.534995,1.733997,-3.851217,-7.138695,-1.829977,-6.832660,-2.114009,-5.391567,5.636409],[3.858483,-5.308387,0.539286,-6.127451,7.659470,9.143907,-9.159116,-2.977668,7.575347,-7.254675,2.870408,2.954263,6.221669,-7.324627],[-7.234968,0.095675,-5.024693,6.614882,-2.051898,-7.751902,8.688050,8.021196,-1.735991,6.549775,-0.016747,-5.063804,3.471623,-5.915777],[5.296802,-4.479089,-1.949362,4.296294,-8.147628,9.165077,-1.645110,-6.112990,2.024094,7.196249,-5.466541,5.798293,9.403564,-6.759390]],[[-2.165939,3.732316,9.293327,3.626619,1.516182,-8.125849,3.158671,-6.652864,2.723356,-1.295610,-8.351659,8.037711,-5.830467,-8.656119],[-4.636365,-0.339236,3.290590,-5.469004,-6.278738,-8.291775,-6.094754,2.040233,-7.507727,3.109057,-9.969325,7.837323,-1.952905,8.897868],[7.048966,-4.032945,-7.994620,-9.810470,-2.279019,2.543445,-6.081257,-6.582144,-6.634574,4.938435,5.423176,2.482137,-2.235167,1.566046],[2.806106,9.404460,-8.647713,-2.257759,6.038996,-6.897281,9.104789,-2.396891,0.324597,-6.900658,-7.007446,5.117090,-1.178623,4.920960],[3.648750,3.169882,-4.806735,7.888036,-3.344868,8.822956,2.489274,6.969072,3.390596,-2.875386,-3.522322,0.214386,-5.932670,2.510522],[1.919537,2.750339,6.661715,0.556043,-9.986055,-2.501570,-6.458514,-9.686038,3.851014,-9.967896,1.882439,4.074134,0.763299,-7.369865],[7.662841,-3.279028,-3.621428,4.543901,6.986777,6.776832,-8.791839,-4.271705,-4.106259,-6.346555,1.918558,-5.007469,1.333484,-5.242512],[-7.932438,8.576406,-2.109091,-1.795008,4.421750,-3.214183,-5.798945,-0.633045,1.197087,5.486016,0.534101,1.513735,-0.163186,8.763201],[-7.273022,-9.507867,-5.423586,-8.602859,-1.552648,-7.296558,5.339488,-7.024106,-3.284978,-1.804056,5.962304,5.394246,8.850290,1.923085],[1.829875,3.488895,8.573148,-2.872387,4.814946,-6.463425,-5.044538,-6.234881,-7.962168,8.231782,6.441206,-5.731239,-4.303218,-6.202990],[3.190517,8.881147,-8.107943,-7.210220,5.098640,-0.570191,9.438957,-5.889544,0.953982,6.444380,9.715259,-1.093504,-5.730453,2.605001],[2.086708,-9.621457,-2.883601,1.662705,9.474590,-4.105385,-3.490391,8.780655,6.545513,8.548851,-7.780248,-1.084785,-7.804782,4.406737],[-7.609253,7.895437,-4.171333,-5.141207,2.474867,8.499805,9.988661,9.720013,-1.123465,7.166613,-3.673086,-9.312403,-9.512804,-9.526013],[-5.847600,6.141423,2.939866,-5.230647,1.032338,-6.760002,-9.688820,2.299102,7.932887,4.299168,6.836178,0.445101,-6.840703,3.140739],[2.650071,0.113800,3.913911,-7.068187,2.919298,-0.996308,3.253263,6.885326,-7.744860,1.355158,-5.847312,-0.239629,0.629511,4.304781],[5.098718,-6.966923,2.662643,-7.123424,9.648852,4.445803,8.365207,3.215100,-8.860480,-8.582082,3.954497,-9.540698,1.004256,3.852696]],[[-2.865198,-5.599788,5.167418,-0.893345,-2.366364,0.920052,1.682658,-0.946517,-1.351434,0.519640,-6.287545,-8.115005,3.877983,0.544656],[1.700560,-8.434435,-1.843081,2.932064,3.323021,-8.446084,2.682670,6.898995,-2.323008,2.116032,1.825787,3.160868,1.058323,2.497802],[-0.191490,1.162237,-4.171775,1.371716,-7.164526,-7.604182,3.146439,9.255079,8.198742,2.865457,-1.674209,-8.724639,-7.733743,4.196272],[5.871691,-3.669977,-2.752269,3.013737,0.673838,8.072568,-1.607915,-0.574338,-9.550522,-4.483866,-7.160700,6.446986,-1.564554,5.860717],[1.004324,-5.241806,4.261804,-1.459256,-1.338113,-0.766445,-5.246118,8.233259,-7.075841,-4.722437,9.410244,-8.406873,6.651075,-5.699246],[1.148186,-6.788195,8.579567,-9.295033,5.491726,-5.854905,-4.019439,-6.341898,-1.613635,6.599701,1.213774,9.359089,-6.034374,-7.595854],[4.276101,-1.243589,-2.179724,1.399996,1.824018,1.125699,-4.449199,7.145595,-3.580944,1.632783,9.824376,5.998376,2.518393,-5.526892],[-7.953501,6.843932,8.500260,7.043990,-5.614756,9.304171,-7.737260,-9.364363,-9.943389,7.948337,4.661411,3.405562,-0.243613,7.785285],[7.148687,2.643181,9.130380,3.591088,-3.648886,-2.646861,8.289953,0.939953,-3.077032,-5.247077,3.754563,-0.210422,1.861941,5.789769],[-1.568267,1.313440,-6.387343,-1.404974,2.109080,8.675728,-7.127016,-1.416928,8.430683,4.268504,9.470431,2.359639,1.845923,-9.449689],[7.637978,-1.563529,-6.761371,-8.468277,8.425654,-7.877440,-8.620490,-8.581223,-6.194463,0.857905,-3.188080,2.343805,-4.579093,-6.316219],[-9.289756,-5.955025,2.598218,-5.285304,-7.802571,-5.086333,7.064160,-5.302797,-4.489994,-4.465223,9.861804,5.270663,-1.192568,-0.385359],[1.422371,-5.657652,7.806338,1.130961,-9.160155,-0.712956,-3.465333,-6.263888,-5.126253,-9.291536,0.978680,5.499892,-2.301405,0.188581],[8.665606,6.062758,-7.848823,4.232655,5.372366,-6.481551,-8.436981,-1.618336,-6.513167,-3.376987,-2.087066,3.688763,2.633544,-6.264206],[1.355071,-0.422299,3.171635,6.940303,0.243339,2.080462,-1.194372,-9.587678,5.244089,1.659383,9.996796,-7.925367,-2.556521,-6.587972],[-5.349007,6.428885,-3.156850,-2.882947,-0.366196,5.335101,-3.529438,-6.656385,8.508323,9.537081,3.031896,-4.302120,-1.760561,-6.345170]],[[-7.982438,-1.961340,-2.772698,3.714081,6.420203,-4.144526,4.387299,-3.928731,3.030579,-6.123732,6.791965,2.286214,1.430569,-9.947817],[-2.904023,-7.657990,7.882020,3.259308,7.819471,-4.635858,-3.278735,-4.435590,-7.762795,-4.307823,3.556529,-7.254045,-1.093502,6.637254],[8.446528,-8.799506,4.054518,4.933312,9.677263,-5.004163,8.179945,-7.704414,8.081625,3.877468,6.238223,6.024584,9.128403,9.123572],[3.075020,1.789273,-7.651676,-1.826406,-0.102274,-5.621573,-5.971985,-5.356445,1.062374,-0.350317,8.646666,5.101933,0.016459,9.075808],[-7.704858,-4.474016,-4.452938,6.049000,3.443652,-5.340973,9.481552,-2.977797,4.454401,8.255248,4.169524,1.085664,5.729070,6.003900],[-6.505758,0.100323,6.718082,-1.267901,5.241220,7.852715,-7.329733,7.173819,-1.345432,0.203220,-9.449847,8.579725,-4.985558,6.407674],[2.830400,-1.905041,1.585716,-1.017686,-4.108888,-8.662346,-9.855039,-5.083234,5.253105,-2.093720,3.919638,8.458227,-4.419423,3.682272],[-7.804334,0.631318,4.992354,8.601652,3.118955,-9.423792,-6.403166,-1.360674,-0.578185,1.918273,-4.172537,3.255644,-7.626184,4.938801],[8.897330,3.617509,6.359652,1.172256,5.486292,-3.405837,3.118860,-9.341348,8.651816,-1.063705,8.825078,-1.826269,-4.395426,-5.763478],[-7.370371,1.501825,-6.951197,7.844864,0.578138,2.103144,5.171540,-6.653909,-0.055520,8.614470,8.821120,7.239175,8.729748,-1.987680],[0.935531,-6.763381,-2.820061,-0.433084,8.226869,7.869060,-1.103925,2.429559,-8.859922,-3.244400,-3.387688,-1.169474,-3.212219,-1.177768],[7.266860,0.082852,9.741101,8.573031,8.533113,0.313724,8.639022,-6.190650,8.467342,-1.532945,8.446545,-5.098913,-2.403654,1.571438],[-5.890256,-4.855869,-0.817029,-8.721607,-6.834068,0.571365,5.025595,-3.843725,2.242181,3.842073,-8.909554,-1.439459,-4.359567,6.536739],[-9.419751,0.848165,-5.600997,-0.782924,1.694007,5.889888,-8.702195,-9.341031,9.152000,7.523471,1.661085,7.769001,-5.813963,6.762614],[-8.557300,5.342558,6.381895,6.720574,-6.414007,-7.160850,-8.573783,0.447333,-2.823987,1.621845,-3.118116,8.348361,1.808399,3.996684],[-8.794391,-3.548659,9.461447,9.133521,8.761211,7.893651,4.722048,1.805349,3.045286,-9.580194,-5.512732,6.878775,-9.705963,-3.333742]],[[9.187809,2.056786,-2.091688,-4.389275,4.909785,-5.000076,-8.210220,7.947774,3.085868,5.587044,5.827288,7.466394,7.748240,4.125444],[1.530679,9.505929,3.480153,-6.033384,7.698096,-4.723436,7.688906,2.512665,-3.440895,0.654534,8.072081,-2.131161,-9.880727,7.350202],[-2.814055,-0.450805,-3.268966,-4.782911,-9.858728,-7.536957,1.247494,9.862547,3.474765,-5.159720,4.094435,-5.429139,-8.951893,4.726496],[-4.035907,5.084355,5.453768,-9.889676,0.413022,-9.211624,3.040817,0.215084,7.026707,4.308011,2.729202,-1.026010,6.768198,-2.936538],[7.754169,6.089621,-3.227732,-9.231630,-9.786520,9.531151,3.133932,9.435942,-9.661140,9.646563,6.731618,9.766915,-4.876515,9.934659],[-0.594237,6.335366,-5.241721,0.075228,1.853091,-7.151991,-3.946078,-5.318459,7.851508,-9.918490,-6.688758,1.900188,3.159927,7.480268],[-3.619363,4.313338,-9.942353,-3.245740,0.006618,6.351380,4.361846,-0.070817,3.402840,8.130360,2.173271,5.443753,1.170722,5.446662],[9.823584,-7.956169,-0.461664,-0.789436,5.975229,1.185361,-5.092318,0.670882,-6.463017,4.647335,-7.223534,7.396415,-6.407094,-6.478580],[4.744438,-4.318612,-3.721226,4.560298,8.414195,6.950385,-3.339940,5.029594,1.583946,0.768683,9.852557,3.411354,4.641299,0.124039],[4.312651,-7.129398,5.926568,4.587477,2.503530,-7.711948,-1.326929,7.338504,-5.670114,0.604156,-8.131810,-9.876929,0.756702,8.517644],[-8.618780,1.502132,-0.823005,-8.940449,-9.868227,7.215911,-1.974147,-9.520276,3.883061,-8.447825,6.609756,1.217070,-1.283813,1.678701],[-7.338047,-5.054369,4.087351,-1.005500,5.963570,5.244555,0.479783,-7.036181,6.820745,-2.180909,-0.493207,-5.874379,8.960351,8.054919],[6.157311,0.874157,-4.249874,-0.748692,9.274728,0.905512,-3.596321,5.828961,-9.935305,2.960184,-5.432883,-4.292284,-7.639098,-3.516796],[8.903139,7.648509,-3.184780,-1.205205,6.001610,-4.588067,9.747786,5.895499,3.157757,4.439843,-6.526840,1.293486,6.698227,-7.947398],[-4.567921,9.338321,-0.338801,2.878929,8.686329,0.720875,-3.870791,-6.291907,-8.665958,9.250751,-9.260327,8.642175,-0.823667,4.305290],[-3.568156,7.570276,8.085599,-7.075856,-4.409986,-8.494287,9.786757,-8.488394,-1.102832,-8.815797,-0.249670,-5.373636,-4.281392,-6.604456]],[[-1.887899,-4.345179,3.348292,1.513409,-6.880964,-9.386627,9.527869,6.479638,7.667014,1.108280,-9.768320,-9.961053,2.732472,1.404648],[-2.646298,-8.256896,2.877147,5.532728,5.376910,8.124046,1.831376,-1.316557,3.703276,7.742132,3.514191,-6.463101,-2.288308,-5.323821],[6.311580,-0.803896,3.604585,-1.905380,1.601660,4.919646,-8.993261,1.518052,1.047799,0.929179,4.564714,8.199126,1.308294,-4.460820],[-0.157466,5.100339,-9.959415,4.246788,2.726499,-0.859275,3.673076,2.116090,1.823569,5.213052,-8.095444,7.003723,-6.839102,-2.288740],[-4.924183,2.681621,0.091417,-7.944949,-1.474347,-6.177980,6.292297,4.576691,-8.877367,-4.295932,1.985307,-4.112989,-1.000074,5.049475],[-9.734058,-5.374621,-9.636953,0.091257,7.060350,1.434751,6.529285,2.535184,-0.386521,-9.079490,0.215583,-9.463857,-1.439326,-3.150911],[-5.373037,7.576027,8.981938,3.789998,-6.280360,5.611454,-3.693339,1.840092,-7.148204,-1.220691,5.163657,-8.172236,-3.846033,3.996350],[8.192690,7.037543,5.469676,7.659650,1.560345,-2.731360,1.465181,6.184398,-4.336404,-4.195476,6.395838,-3.491838,9.888698,3.196968],[9.913371,-6.735096,-2.119732,0.037447,6.185889,-3.185208,5.701569,-3.988812,-3.519751,7.851526,-2.307862,-1.627479,-1.399062,1.467041],[2.918092,2.005688,5.183005,4.784442,-2.910785,-2.571175,1.376113,-0.574500,2.398047,-0.840436,-1.181947,3.391814,6.195095,-3.428385],[-7.281529,6.109675,-6.024042,-9.941218,-3.941795,2.282670,-9.628889,-0.770914,-9.845698,5.674325,2.908625,7.356328,1.848630,6.578879],[-0.593163,-6.987333,8.841758,-1.229408,1.527903,7.061693,9.680373,-9.034589,-9.901972,5.564574,0.472836,7.467474,-1.861840,-2.682945],[1.684610,-3.305432,5.314823,-0.837337,-2.119981,-8.467501,0.943800,-0.614698,-6.316932,5.077629,4.113609,2.040071,6.071610,-8.683321],[8.500005,3.911468,-1.127829,6.059820,-9.058366,-3.742560,-5.165218,-9.121668,-5.269371,-2.721423,1.156277,-9.525853,9.005993,4.700389],[-1.771582,-4.122100,-8.072797,-9.886833,-9.670582,-9.841940,2.113447,5.617163,8.156603,7.232554,-8.876914,-0.356966,4.143178,5.336694],[-6.812972,5.503368,-9.691788,5.346562,9.717767,2.532624,8.957653,4.553800,-4.781080,-6.490356,-3.213823,2.668200,-3.485046,1.142472]],[[-7.393721,-0.209148,4.411116,1.665417,1.339831,5.357301,-0.253652,-8.782122,-0.973851,-0.750250,6.075214,1.610572,6.390480,-5.680097],[-8.077492,3.687211,-2.526028,-6.244129,-7.706663,6.683912,-6.086710,-2.158861,3.701720,-2.665305,-3.446044,-3.432465,5.573746,-2.542160],[5.706814,-9.355740,-3.912337,-4.801289,8.060532,-6.343494,0.399568,-9.727437,-6.608674,-2.505547,9.412472,-2.042852,-9.577170,6.354903],[-4.214212,-4.595584,2.769321,7.005548,-4.598566,-2.879932,-9.937151,8.391011,8.734993,5.910945,3.443490,3.902425,-1.218065,-6.818090],[5.567502,0.234159,5.324655,7.570623,-8.861730,-5.416006,-7.117386,7.020692,-4.354418,-9.757694,-8.417221,0.864488,-4.653835,-5.589633],[-9.076365,9.501141,9.787377,4.433543,-7.900014,5.106096,-9.638172,4.284700,-8.832022,0.837750,-4.034834,-2.754259,-5.894473,-1.131163],[0.219715,1.266355,2.977832,8.690112,3.627966,-4.858475,7.271957,4.653480,5.521567,-0.674263,1.995300,-3.308526,-7.209427,-6.237335],[8.860191,8.193275,8.342113,-8.720589,6.529029,0.874928,8.530372,-5.754140,3.126734,-2.832761,-3.738060,9.119412,-2.119408,8.136887],[-2.692295,6.703787,-4.399643,-7.129297,-1.510391,7.583582,-6.233743,-9.419877,-9.886956,-0.612213,-7.602704,-8.197158,-5.654245,-5.190109],[-1.627072,4.986860,3.808549,-5.964942,5.073319,-7.170709,3.107331,7.057777,-9.000164,6.390858,-9.683441,5.413903,-2.372193,4.070941],[9.968979,-1.144166,-8.010222,-7.645923,-9.077256,6.842343,-2.421467,9.322614,6.741439,9.831416,9.495525,-8.736081,8.410593,-9.410726],[-6.004683,-8.968512,-7.530068,3.913558,-6.916364,-0.130312,-9.605911,-6.086290,0.839469,-1.077986,-9.957124,7.439754,7.054981,8.745731],[8.963348,-1.610880,7.207534,8.020413,8.196759,-9.640590,-2.592615,9.168395,2.776670,9.618900,3.285424,6.577977,-2.251759,3.252469],[8.019833,-1.515955,9.803946,-7.951858,3.869611,-3.520020,-7.830874,-4.775997,7.973302,6.129898,5.317224,-7.519328,-9.628894,-1.638084],[-9.826906,-1.912131,-2.472560,-4.155037,-8.272065,-4.446084,-8.117687,-5.582077,-6.247820,9.729673,-6.811770,-2.462436,-4.376787,9.518869],[-0.453379,5.125269,7.717728,-5.284198,-4.688751,6.292367,5.397708,-2.340622,-4.481427,-8.143404,3.351427,7.455545,-4.513005,2.850681]],[[-3.439435,-9.009506,-8.929816,9.152212,8.624729,1.240850,-4.291761,-5.708527,-4.276766,4.913825,-5.366277,2.077227,0.884050,6.021865],[-9.290863,-6.750333,7.297643,-8.835896,-0.397821,-4.499627,2.218493,8.647222,-8.133816,-8.982880,-5.673360,-9.895233,2.096970,-9.414642],[-5.743316,8.870381,-7.559948,1.319872,-2.829787,0.885620,6.056954,-5.478762,6.446487,-8.169028,-2.044604,2.699005,-7.349608,7.142617],[3.604588,-3.409053,9.551331,5.844426,-6.306636,-0.616706,-8.005827,5.671736,4.511286,-2.182929,-4.400821,-6.320808,7.266736,4.257983],[6.699949,-4.898940,5.384416,-2.074348,-6.742021,-4.354492,-4.195207,4.054029,6.885341,-3.805574,-0.573431,0.511469,5.734046,8.932806],[4.280781,-7.330792,1.380003,-5.739411,-1.475820,-8.182573,-9.261828,4.586830,3.744246,6.661858,3.331881,0.854343,-7.339487,-5.114474],[-6.181979,6.069384,9.931296,6.513571,-7.744014,2.433762,-7.750249,6.311080,-6.486450,-7.986508,0.797597,-5.943259,6.231593,-6.315483],[1.688440,-0.443201,1.667431,4.858774,3.718133,3.523008,-7.852361,-8.265225,-7.096332,1.481623,-8.338137,-2.981323,9.002170,4.825303],[-7.583881,0.339755,-8.208896,-2.951475,2.378938,-1.516967,-0.632512,-6.065478,-0.257989,6.197250,-9.543524,-5.300141,-8.879578,-4.862888],[9.056350,8.517683,9.387858,0.457181,-6.693677,7.826659,-6.016483,-8.741341,-8.444458,9.896852,-7.730998,5.532718,-0.009104,9.859672],[4.034658,-4.090116,1.474081,-5.483098,-7.567954,1.045197,8.250938,-1.206101,-0.784363,-2.222042,2.480377,-3.940776,0.502762,-3.993122],[8.574454,-3.906057,5.050872,1.578784,-0.077215,-3.444298,-9.221166,2.006419,-4.492660,1.340998,7.681723,7.078275,4.303148,-9.765235],[-5.722531,-6.260493,-5.788920,-6.209617,-0.861541,8.082278,4.003630,9.428848,0.432358,-5.970619,2.357405,-1.938252,9.879894,2.458233],[5.978526,4.562874,3.551548,9.125611,8.185939,7.312586,-0.654427,-1.019666,-1.573438,0.651524,4.603744,-6.499592,1.910027,1.373871],[-7.674037,5.961620,-7.036330,0.344340,4.541111,9.896580,-4.228656,-5.530257,0.394845,8.781718,-6.514347,6.983167,-0.608182,-3.599464],[5.723709,-8.978242,7.243038,7.529372,3.197688,3.357476,-9.993193,1.653838,2.803468,3.420215,-1.480893,5.476898,3.830181,-7.798659]],[[-3.326826,0.888119,-6.138237,9.186211,7.298811,5.527034,-7.947089,2.679349,-5.754933,4.610362,-2.906588,-7.572177,5.705750,-2.702747],[0.128940,-8.523975,9.166244,9.859529,-4.537275,4.104266,2.433464,-4.199498,8.069015,-6.510974,-5.303967,-0.686233,-5.820230,0.366901],[-5.172160,-7.773796,8.785309,2.414035,-8.251153,-2.784556,5.694757,3.425070,0.858377,-2.009982,2.030685,-9.355355,4.538374,9.969198],[0.206822,8.878863,-3.054850,2.631254,6.415531,-0.234150,2.142526,-7.643915,5.712954,-5.340962,5.528066,0.573356,2.354129,-2.063488],[-3.174703,4.585706,-9.004081,0.525733,-8.736389,-8.582604,8.671418,-0.553951,-3.773212,-7.575297,-8.641602,-6.795978,2.787026,-0.290132],[3.207448,5.841629,-3.845862,0.527867,-3.735404,7.547591,-0.898259,-6.032322,2.414003,-3.675140,8.754052,5.199290,5.361685,1.676726],[8.501067,-6.492794,-6.681680,7.905224,-8.169706,7.502922,3.328998,2.067823,7.510796,-3.076354,-7.858198,-6.426376,5.083722,4.555280],[7.533983,5.417600,5.491693,5.667271,-0.394639,-4.366501,2.152674,9.436255,6.521212,-4.082279,-0.749870,6.484194,-0.030561,-4.619659],[8.933241,-5.321234,-1.255084,1.271883,6.625311,2.487785,6.599577,-5.504434,7.685327,-4.056083,2.127049,7.270119,8.424743,-6.548572],[-6.749106,6.708360,9.758312,-6.792775,6.220030,4.730225,-7.010573,-0.705172,-6.854094,5.455029,-9.957753,-2.286869,-7.725745,1.626808],[-9.266320,5.102449,-8.888160,-9.808957,2.692653,7.089868,5.894991,1.132826,1.243056,2.197715,4.594281,-2.885871,-9.401913,5.754302],[-0.499856,-5.757507,-1.537440,-6.039354,-9.598710,-8.881709,4.608305,2.595017,-0.499236,6.440467,-0.010586,-0.037968,8.479640,-7.772558],[3.918434,-7.333059,-3.550330,6.424719,2.796957,-4.232226,-7.373428,4.481195,-4.660780,-3.825725,0.908722,6.832938,9.673068,0.103734],[-9.836352,-7.286534,5.649769,2.487604,-9.640341,0.883908,-1.972736,7.231352,-9.461719,0.425626,9.156874,-1.487625,4.817851,4.789855],[5.187015,4.020612,-5.102957,0.282512,-3.234834,3.274497,0.186506,-7.558196,-4.372892,6.435008,-4.489425,-4.104849,-5.889534,3.850065],[-7.290942,-0.629532,5.663988,4.348185,-1.216094,-6.227435,1.241780,7.272717,-1.281070,3.977819,4.628380,7.261458,3.013678,-7.280363]],[[-7.235622,6.235391,-0.574984,-5.222865,4.993341,0.775302,8.223801,-1.531874,9.923497,2.716083,-8.061050,3.003836,4.450769,7.436099],[-3.603723,9.365832,0.509931,2.014009,-6.343239,-3.399551,7.750571,6.403909,9.074057,8.450498,4.805897,-5.296777,5.924959,-1.128927],[-0.218004,9.069802,0.379478,-8.191886,1.079859,-7.678280,5.414518,9.852597,-4.550641,1.376572,9.750435,-7.634572,-6.206832,-7.894310],[-9.969935,-5.519082,-4.111950,4.143651,-5.447104,5.765512,5.943809,-7.385709,-3.759380,0.852062,-6.506617,-9.161731,-8.927674,4.060169],[1.780049,-8.860679,-9.771732,-5.649589,9.508706,-4.152413,7.544024,1.326764,0.752837,-0.673091,-1.941138,-7.465426,-5.991545,-1.304439],[0.946746,7.699360,5.704826,2.060666,3.150920,-5.096002,2.927495,-5.675180,-6.397831,-2.991548,1.857047,6.806168,-8.974643,1.485555],[-2.046094,7.802514,1.517270,-4.350418,0.462041,-9.816936,-0.969773,5.412921,4.273271,-9.203300,7.723823,1.327993,3.491701,-4.729781],[-8.187917,2.732285,2.917224,-5.022258,9.286499,7.259557,-3.020950,8.230921,2.985343,9.530381,6.229584,-3.924152,-4.346397,4.497438],[3.722379,0.344755,6.809736,-2.058467,-5.177739,9.046766,-7.263093,-6.781422,-3.668162,2.589172,-5.532777,-3.245014,-8.990763,-0.954709],[-4.484092,1.397089,-0.458697,9.993489,4.202559,-9.854260,1.294666,9.503811,5.748615,-6.833512,-5.265469,8.320731,-6.185976,-5.800366],[-7.720673,8.843623,-6.687931,-5.694946,-5.925316,6.298444,-1.587187,3.641844,5.582076,8.202813,-0.270873,2.462814,4.192798,5.357398],[-6.546080,-7.952823,-4.497989,-8.672180,-8.999562,8.299122,-1.414173,-8.707336,0.079337,-7.848890,5.833909,0.529503,-9.833261,-0.860885],[8.720553,9.207654,1.127440,-5.269382,-8.146223,-4.372060,-0.708135,6.513635,-1.363631,4.942722,9.421709,2.482269,0.950791,-5.685392],[1.555916,9.258428,-7.297560,-6.924887,-9.355847,4.355012,0.888434,-0.461370,-1.800175,4.107251,3.341368,-9.425337,-4.607267,6.384088],[-7.756546,9.413139,-8.059158,-9.069661,-6.165936,-1.421555,3.588038,6.680515,-0.365613,-0.220717,1.063600,6.360080,-4.900688,3.350360],[2.683784,-7.634531,-5.698039,8.177519,-0.660611,1.735496,6.958090,-2.860432,4.095806,3.231914,7.207788,-6.057104,6.116089,0.472855]],[[1.176754,5.877924,-4.840201,-8.386118,3.760509,0.963078,5.940392,-2.182336,7.221248,-1.811042,-1.479712,-1.803496,5.394113,-0.829684],[-5.709972,1.168374,-0.343506,7.158682,-8.835861,4.883080,9.319420,-8.829062,2.459055,9.543751,1.837065,-5.199544,-5.022438,2.721132],[4.535136,-8.335662,4.886901,-3.555127,-5.107220,-7.978782,3.061714,3.546726,5.341981,-8.776329,-8.957236,2.213799,9.243602,-0.018645],[-5.649595,0.357681,2.266281,6.522105,2.156770,6.927436,9.473896,6.010644,6.690041,-6.677646,-3.231933,-5.436229,-5.484220,-5.853916],[0.300257,9.125326,-5.247480,6.736680,2.191726,-9.345628,-3.570257,-5.547331,-3.830549,4.410722,-5.168615,-0.224778,-1.067486,-9.805577],[-5.546173,-6.943042,6.298294,8.802404,-7.131821,8.761084,9.586038,7.182600,1.091252,-4.439275,3.108166,-6.986407,7.736201,-7.980808],[2.160197,-2.476095,1.727929,4.630154,2.829028,-3.325877,2.827425,3.196964,5.256940,5.040377,-5.278791,-4.813105,9.228892,-9.903734],[8.853614,-8.895134,8.694353,-8.065651,8.953320,7.435403,-8.026028,-9.697041,1.498818,5.618516,-0.515141,-7.774844,-2.504510,-9.688765],[0.892102,-3.150807,2.482707,5.221449,8.218583,9.835488,9.678800,9.591422,-3.532997,5.530801,1.641864,4.263470,2.822288,2.667897],[-9.490379,9.482710,-0.033348,2.552538,-1.529012,-0.866567,-6.869902,-1.548275,-5.314570,-8.592063,-7.342285,-3.848901,-2.250591,5.315951],[1.896717,5.692774,4.736279,1.840564,-5.458179,2.645398,1.786542,-9.145770,-3.080170,5.107827,7.607579,5.715380,-3.635054,5.445067],[6.337304,-5.349404,3.260135,4.587220,8.680613,-1.034329,1.682026,-5.886105,6.992397,-3.193618,-5.921622,-4.346337,8.153714,2.225986],[-5.847158,-2.199350,2.469845,-7.767549,-1.803410,2.302762,6.661549,5.552238,-3.557458,6.998796,8.786781,5.784400,-1.029841,6.849301],[6.574062,6.098665,0.979550,-9.043452,-1.225256,-9.165847,-3.188752,-5.250666,2.320255,-3.973898,-7.950830,-9.985132,6.048369,7.296758],[9.724987,-1.450622,5.636791,-7.504138,0.354180,7.004410,-0.102342,0.376777,-0.031163,-4.233883,8.591022,1.006379,-6.459670,7.215135],[8.821304,8.718303,1.683416,5.571407,6.437151,-6.749511,-9.362340,3.028933,9.868459,5.747874,-4.401456,3.023542,-5.539066,0.559534]]], dtype = "float32")#candidate|983|(16, 16, 14)|const|float32
bop_984 = relay.power(var_982.astype('float32'), relay.reshape(const_983.astype('float32'), relay.shape_of(var_982))) # shape=(16, 16, 14)
func_393_call = mod.get_global_var('func_393')
func_395_call = mutated_mod.get_global_var('func_395')
var_988 = relay.var("var_988", dtype = "uint64", shape = (91,))#candidate|988|(91,)|var|uint64
call_987 = relay.TupleGetItem(func_393_call(relay.reshape(var_988.astype('uint64'), [7, 13])), 4)
call_989 = relay.TupleGetItem(func_395_call(relay.reshape(var_988.astype('uint64'), [7, 13])), 4)
uop_990 = relay.sqrt(const_983.astype('float32')) # shape=(16, 16, 14)
bop_992 = relay.minimum(uop_990.astype('uint16'), relay.reshape(const_983.astype('uint16'), relay.shape_of(uop_990))) # shape=(16, 16, 14)
const_996 = relay.const([[[1.949361,5.852866,1.332696,5.833602,-8.324475,-0.211131,-3.032401,5.242648,8.598818,1.504980,4.849523,9.634350,-4.576434,6.173521],[6.248537,5.459025,1.948567,8.799999,7.097141,-8.375054,3.294162,4.254730,-7.104693,4.328243,2.645770,-2.728904,-0.662901,-6.102434],[3.467501,-4.900986,-4.653063,9.665516,-6.926385,3.270810,5.966344,2.525319,-9.060129,-9.238020,8.338896,-4.732475,-0.588075,7.338494],[-8.525837,-6.386381,2.480961,-0.363103,-5.475018,-8.586458,-2.055948,-6.017077,4.972552,-2.848422,-8.704866,6.940556,1.340847,-2.569042],[-3.530886,-8.357204,-4.269755,-0.476309,0.613133,2.413577,-5.737244,-1.683735,6.450991,-3.048749,-3.608140,-2.497088,-7.959307,-0.480242],[9.671504,3.005089,1.176271,-9.505778,-2.921910,1.467404,3.771328,-9.076816,-6.447002,-7.265389,5.369724,7.412855,-1.730664,3.565674],[6.491564,-9.547847,-8.198999,-9.444974,2.850221,1.879586,-0.981499,6.719864,-5.919912,6.019433,-2.981113,-8.279773,1.102458,0.836530],[-7.786138,-0.017097,-8.827588,-1.612817,-4.695306,-8.251778,-0.722620,-4.476424,3.126288,-8.745433,2.463898,-5.070170,-3.216766,0.600996],[-1.175598,2.083181,4.044040,-4.196164,-9.777719,7.181292,-7.542554,-3.614338,3.322298,3.382024,-3.370402,-9.771984,6.167543,3.339061],[-7.113477,9.246116,5.443577,1.543474,-5.642236,-1.897783,-5.662201,-9.998190,-5.497164,7.528440,9.289688,-9.211829,7.305939,9.716754],[2.590026,-1.367568,5.108119,3.330556,-5.105877,-0.058182,-5.860568,0.147988,-0.372233,8.538956,-5.898792,-9.540444,1.045517,2.716394],[-4.022225,-7.843041,7.373126,0.101968,-1.652020,5.143344,0.944263,-9.074656,-0.501781,-7.365521,2.330339,7.884196,4.410119,1.658612],[-7.902396,5.397869,6.734510,4.204895,-7.655394,9.531170,8.504160,9.101562,-5.511009,-1.190286,6.312709,1.471298,1.155302,-1.860382],[2.690258,0.254102,-1.121388,-3.015219,-7.174557,-8.860471,-7.001595,-8.468978,-3.074621,-3.893965,5.564776,1.490620,6.367065,5.468722],[-4.511610,0.956800,9.702770,9.495544,6.654652,6.232565,3.669722,-9.016689,8.142775,0.634828,-3.006643,7.664687,4.564055,-7.024766],[4.395339,1.862908,5.538274,4.771844,3.426866,-5.370499,-4.195537,4.838068,7.658450,0.090188,-6.210242,-8.437168,-5.215526,-8.719594]],[[6.365682,8.314505,-6.639587,-4.739322,2.884487,-1.554673,-2.855218,-2.345496,-5.991270,6.766671,0.863496,-1.085620,4.946498,4.167017],[-4.637048,8.505930,-1.577062,8.585561,0.366801,2.354174,3.626125,-9.134296,-0.657874,7.234992,-5.098443,4.280204,5.650841,8.569093],[-4.945657,7.586235,-1.216790,-1.913946,1.134207,8.933532,1.136497,1.382810,-1.038157,6.846049,-7.416667,5.979639,8.115565,9.829621],[-9.442179,6.021590,9.543245,-3.827565,-0.794768,-2.836875,-1.631076,-5.659137,-6.681460,-6.069029,5.176134,-1.195228,-3.339990,-1.020826],[-5.428178,8.036801,-6.047263,9.282747,9.611376,8.209919,5.563333,8.446392,0.645757,9.239928,7.033641,-2.673080,3.648262,2.752086],[-8.232251,7.017070,1.729553,-0.951443,-8.855998,8.739922,7.396493,0.529733,-8.082868,-5.801776,-2.232327,-9.882607,6.698498,6.418973],[1.583056,5.423886,-5.159388,-0.767725,2.834805,-6.233234,-0.857026,3.170232,-8.281646,-9.967287,8.532840,3.675913,1.506273,1.226399],[-1.255907,2.246832,5.176293,-3.099491,-5.657240,8.945652,0.432774,6.327717,3.783364,1.720426,6.293027,7.444819,-4.224096,-1.049749],[-4.991291,-0.999362,-7.211807,-0.217328,-4.274700,-6.361462,3.882181,-5.271252,1.118199,-9.098933,-5.167547,-4.054537,-9.460978,-8.075749],[-0.003138,-5.832492,9.827890,-1.646198,-6.458712,9.301956,0.218798,8.059017,-1.974440,8.191694,7.443829,-4.993293,6.966884,5.153718],[-2.529200,-9.603534,8.206610,6.548705,0.963086,-5.600765,-2.385767,3.208640,8.346621,-0.649345,-3.288117,7.039790,-8.779205,-8.827985],[9.107974,9.269170,9.234837,-6.140458,-2.025451,9.619616,7.014316,1.888837,-8.218888,3.456739,-8.037746,9.268528,3.714810,-4.332797],[-2.951305,9.386368,7.859337,-9.206117,-0.938367,3.169804,-8.074112,4.507419,-6.183856,-2.624787,0.378709,7.446522,6.279197,-1.575922],[6.286963,-4.028740,3.574605,5.206488,-5.774160,-8.061178,3.111210,9.184704,-7.755867,1.955490,-0.364827,8.601212,-0.645505,6.132925],[2.891186,-4.925574,-4.273866,5.950146,3.480318,-6.923841,-4.423695,-5.362190,-6.176275,0.143284,-8.359196,-7.429319,-2.262031,-5.814778],[-8.549150,7.834862,-1.314599,-9.342986,-0.005551,-7.239440,5.135542,-0.072566,5.666012,9.059010,-1.907057,4.914947,1.611573,0.725957]],[[8.739037,-4.677259,3.236574,-2.923957,-7.381067,8.881074,-3.208806,-0.455224,-3.578022,1.710950,1.083685,-4.064143,3.009391,0.100381],[1.161237,0.574755,-7.629855,-8.478676,7.164173,2.136670,8.368017,-6.400777,7.236101,1.852195,4.650453,2.556123,3.202952,7.010168],[7.766432,-1.245694,0.394224,-7.075469,8.655635,-7.424751,7.186563,-7.987944,-6.703471,-7.617034,9.721694,9.622698,-3.696410,-5.709680],[-4.980185,-5.596105,-3.076973,-3.110495,-9.201637,8.742337,7.809633,-0.313416,-5.459963,-9.111779,-5.060286,-6.149322,-1.006599,-6.445232],[-7.068349,2.100754,-4.134607,0.845967,-7.232046,3.993725,-6.866297,-5.956167,-2.724017,-1.238127,-7.289297,3.555434,6.399776,-8.138455],[2.894664,-6.142969,5.466446,4.605649,2.258424,-9.105921,-6.357580,9.457872,8.883483,1.601505,-0.754555,-2.087430,-8.206866,1.627651],[1.685941,2.376876,3.422678,-1.219557,9.602620,-1.253608,-4.977605,-2.173313,-2.945206,5.202107,-9.646235,1.143783,-6.058640,9.391532],[4.776526,-9.344367,-5.602790,-8.406319,1.579086,4.534681,-0.375114,-0.644624,-3.091104,3.513369,-5.591948,-0.396654,1.630294,5.546192],[6.020764,-0.362246,-8.553203,8.655258,2.165845,-1.562913,7.827975,5.741241,5.609475,-0.982051,4.325528,1.186555,2.679643,-5.064191],[8.211807,-8.169072,-0.432390,-0.681846,-1.958209,6.470275,2.430812,-1.310561,0.599786,-9.916396,2.848705,1.136024,-4.641454,-6.583799],[-1.588995,5.811395,6.376179,8.240373,-0.189772,-6.667620,1.771631,-3.451422,8.754078,3.151570,1.286050,5.296123,3.486252,1.533219],[9.080926,0.833868,-3.613602,-6.968408,-3.233866,3.904688,4.970162,8.469263,6.485037,1.053084,4.052616,-8.170551,-1.615116,0.029752],[-6.306240,9.097565,9.328239,3.093984,4.659583,1.267993,7.981777,1.976064,-7.485971,-7.888815,-9.320164,-8.185477,-2.863930,-7.980896],[-9.757237,-0.029444,-9.642629,2.355030,-6.637073,5.169546,-9.934727,-0.120302,2.652459,-0.867504,-0.943387,-5.533375,-2.505021,-6.572009],[2.370644,9.970774,2.588257,-0.946637,8.504907,1.709742,-0.559149,-9.872634,8.356181,6.961664,-0.388807,-9.336769,-7.624461,2.580083],[0.410266,8.609626,0.646160,-0.881939,2.830206,-1.096056,-0.370274,-5.072967,6.977005,2.989766,3.190181,9.410247,-9.127194,2.440310]],[[-3.693807,7.902160,-4.926033,-0.690610,0.660062,-6.961813,7.203535,-4.639222,-2.729457,3.574786,0.225221,-7.945385,8.378952,-1.808218],[6.913998,-4.932279,-9.232511,8.084669,-8.389821,-0.148201,2.538164,-0.259966,-0.821501,-9.884710,-7.845393,-5.592330,0.050777,-3.805916],[-1.007643,-1.996598,-4.848532,-0.619339,8.223946,-2.828585,5.891530,8.882900,4.170398,-4.173461,-4.963506,1.467936,-2.158655,-6.365028],[-9.720032,-9.836363,9.654173,3.228661,5.413050,2.496529,4.088694,-7.578322,-2.322984,1.553038,-2.651956,9.996310,4.536022,6.198931],[-1.264568,2.130298,-0.039358,-1.295294,-7.502078,-6.959759,2.787915,-4.752772,0.372392,5.165628,0.053410,8.751402,-8.422847,-8.992729],[8.883417,5.920559,-8.513409,6.947724,-2.646320,-8.621957,2.723505,-2.800172,-9.421655,9.188812,-2.580247,-4.505527,-4.757953,3.145978],[-8.188244,-4.224289,6.253749,-2.150765,-1.493069,-4.767796,1.677095,2.320140,7.137901,8.722037,2.123669,6.851680,6.708055,9.066549],[-8.140572,9.217843,-2.514888,-6.180812,8.490074,7.279795,-5.756837,0.182516,2.305688,-3.778465,-9.179529,-0.607513,2.515150,6.416565],[7.749492,-4.489151,9.038663,4.629228,6.514389,9.511669,-2.734294,-0.162501,2.207660,-6.736116,-2.592649,7.434237,-6.304831,-5.816652],[-6.563802,-2.657154,-9.267809,2.365153,-2.940378,-2.742525,-1.655546,6.388535,-5.480185,-5.584831,1.487660,-4.948546,4.283956,-2.249689],[-0.013385,9.875775,8.298168,-0.019019,-4.739239,0.368032,-0.604955,-3.686959,-4.388970,3.549606,9.407834,-6.965451,-3.177847,7.108734],[-1.815491,5.827106,-6.671501,1.066250,1.670919,-4.913217,3.511719,-4.136009,5.455170,6.215630,-3.841127,-4.998750,-3.272942,-7.778949],[-4.204014,-5.000136,-3.634912,6.871968,2.987885,-4.552657,-5.064014,-7.781505,-0.186512,-2.825590,-3.896663,6.688842,-9.939169,-8.834517],[-7.940973,-4.715072,-4.553366,-5.394201,-3.634390,-1.071048,-3.703095,-8.640601,9.714808,2.063466,-9.915935,1.046149,3.536870,8.978274],[-6.419377,-4.400356,4.495697,6.686398,7.308801,-9.215528,3.394652,3.983904,5.661460,3.201002,7.154101,-5.841904,-6.549519,-6.517342],[9.060798,4.492133,7.900343,-5.923934,-5.045926,9.727390,9.923461,-6.752890,-7.298570,4.076288,0.963909,-5.046236,4.976126,-0.204676]],[[-0.395870,5.599351,0.055116,2.941395,-6.068298,5.518482,-6.010948,-8.661925,-0.215760,-7.126727,8.379405,-8.666900,-2.053401,9.711218],[3.165231,3.413233,-3.537462,1.516387,-8.714739,-3.373416,4.250012,4.911612,-1.166510,5.474748,-1.784542,4.387412,2.798765,0.710394],[-5.259689,4.501584,-7.722739,4.444712,6.793669,-7.869494,-8.722770,-4.528058,-6.303959,0.659735,-8.121231,9.796960,-5.206648,-9.900663],[-3.971533,-3.713981,-6.294137,8.179674,-8.914700,6.734594,5.432389,1.584881,-6.327546,5.720495,-6.512630,-9.644176,-6.852355,6.958234],[-0.695701,-9.892215,2.426252,-3.689327,4.943874,7.099963,-8.748006,3.991950,-1.774332,7.723396,6.522535,-1.267254,5.638493,-2.225649],[-9.832610,9.992341,1.517288,-1.917603,-1.501264,-8.440073,-0.327225,2.778390,-4.657280,-4.091300,-4.271484,7.111498,7.869943,2.716697],[-9.278568,-9.402151,5.375602,-5.377731,3.091059,-0.366875,8.758249,3.665751,-6.439113,-5.947729,-5.803132,9.159378,6.667623,-3.288998],[-0.872699,1.788609,3.012197,-1.892989,0.303764,7.515579,0.656488,-1.669721,-7.781464,-6.283368,9.394019,2.616612,4.579205,-3.524122],[8.017505,8.120136,-8.894037,0.685237,-0.001851,-6.455888,-4.999149,4.954003,2.834322,-6.729866,8.044004,-5.152191,9.975562,-1.471798],[0.737554,3.320921,8.371617,-4.763866,-6.428607,0.320975,8.007617,-5.017876,9.977990,1.977320,-7.478311,-6.291044,-5.963526,-1.145310],[-8.025818,6.885747,0.546481,8.066092,5.064744,-7.909261,8.959185,6.175805,4.673792,-4.687319,1.591014,-3.589725,3.712868,0.754019],[5.315922,3.421914,0.970000,2.037023,-3.240887,6.649156,4.732945,5.224380,-1.741125,5.872698,-1.891272,-7.798745,-7.865017,6.326954],[-0.609611,4.604114,-6.252412,5.215180,5.099048,-5.568856,4.757741,-6.450154,7.409480,7.480627,-1.625064,-8.738409,-5.133246,-5.787343],[0.102014,-3.613938,-0.224976,9.864383,1.265255,-0.437205,-4.619644,-0.774748,9.358366,-9.103759,-3.338549,-4.935914,-1.831084,0.826445],[4.996671,1.432236,-4.912232,-2.445244,-7.689354,6.437001,2.961381,-9.406631,8.604350,0.205932,2.184052,9.888599,-8.435242,-9.324180],[-6.759544,8.982996,7.136523,-4.170807,1.085986,7.108539,0.725996,9.098382,-8.610696,-3.872250,-2.348577,-8.526543,4.348817,0.221341]],[[2.671781,-5.486298,2.084428,-0.814239,4.340074,-9.564584,2.517560,-5.670040,1.957843,8.964027,5.037911,8.586324,-8.952116,-0.668620],[6.995181,6.370106,9.596890,4.116497,2.046617,-5.172168,-5.506256,7.200463,-2.254891,-7.280920,9.139719,5.469410,9.109515,-7.304026],[1.334173,2.710594,-2.742688,2.358163,2.666205,5.805658,-3.821785,2.188634,-8.307026,0.148646,-1.084630,-2.558812,5.432154,-3.419914],[-3.867232,-9.417174,8.015788,4.338563,-6.478142,-2.415471,8.610691,7.416141,9.467590,-6.367669,6.992078,-4.662704,-8.320059,-4.697339],[-6.238545,-5.209100,0.273188,-2.867226,2.438560,6.252331,5.637438,2.246849,8.107179,-0.315446,6.912522,-0.529484,-6.202539,-7.130918],[-6.847352,0.774896,3.131090,5.081338,-6.397352,-2.309635,0.307648,-5.172036,2.277960,1.782154,-0.526626,7.058998,7.485344,-7.970880],[-9.990680,-9.156168,6.204745,4.608258,8.070594,-4.089022,7.503005,7.848032,-9.176852,1.102785,7.484763,-7.830359,-5.846717,8.474822],[9.464421,-1.095551,-5.383203,-1.885962,-4.033862,0.538171,7.895441,7.478136,-9.522711,0.429912,-3.969085,-3.823395,-5.066028,4.078746],[-9.738898,-6.936298,9.135535,2.158907,3.718609,4.883189,6.101252,5.029713,-5.196541,5.856442,-8.838300,6.290457,5.860743,-5.305330],[-2.991692,2.707681,5.777621,3.821493,-4.251479,-3.771169,-1.345915,5.835186,-9.797133,-7.488842,6.341231,9.564023,2.534337,0.408932],[-6.444092,-7.540013,4.342947,5.167116,2.367503,8.923567,2.822920,6.719954,-6.713183,-2.757468,3.681250,7.832442,-1.204552,-5.385007],[-6.921474,9.512141,-5.739783,9.863774,1.966785,-3.788225,1.070949,-1.004743,6.149794,2.770472,-8.561802,8.174493,-3.266205,-1.909439],[5.453944,-4.774589,-9.162176,7.385824,9.253228,3.991185,4.342777,3.537125,7.939391,-3.115162,2.772745,-6.873293,7.068783,-7.376199],[2.527100,-4.611708,0.498202,-3.422130,-2.437990,-8.604002,-0.470561,-8.550549,0.706237,6.977717,4.264665,-7.456062,3.100281,-4.250645],[-5.717124,-8.296042,-0.116931,-4.916734,-3.867207,4.003393,-0.458394,-4.648694,8.173074,8.177163,9.219873,-7.757048,-8.059288,0.962470],[1.633536,-0.391916,-7.251747,3.572643,4.953538,7.059852,-3.892919,5.219776,-1.773646,-1.074017,-7.155090,9.441316,-2.605514,2.519309]],[[-1.871832,-7.424294,-5.548252,3.415463,1.590588,9.051716,-2.885466,-7.034018,-1.550449,-3.443181,-2.628402,-6.081777,-3.944188,-3.936912],[3.565674,0.828405,-3.132632,-9.713099,-9.893407,-8.684162,-9.432651,-1.456782,-4.497894,-7.000385,-5.447890,4.570206,7.420274,-0.412328],[7.403231,-4.994460,-2.653064,-1.047042,2.795617,8.831196,-4.707672,7.669854,-9.256456,-8.159626,3.647710,8.075898,4.025984,-7.073024],[-6.028794,-6.871110,6.241710,7.243159,-4.728565,-0.747022,-6.022721,9.464280,8.500422,7.024934,-2.590890,-7.843846,-2.593585,-6.189574],[9.711274,-1.132360,0.477527,4.039619,0.028606,-9.646076,4.984964,4.927494,-6.588001,-2.017487,-7.870413,0.636939,4.952951,-5.189766],[0.181548,2.758779,5.284690,7.648609,-8.005386,5.364338,7.171976,4.322902,-5.282694,-4.763134,4.274760,4.571505,-9.375141,8.422325],[-8.326485,-7.197357,-3.688703,-9.767353,1.911496,-0.179443,9.772444,-3.006373,0.647146,7.080546,-1.603309,-1.252691,-5.393835,9.860449],[3.701010,2.006084,-6.035906,9.904480,5.630769,-4.094896,2.569843,9.618159,-4.913587,1.776436,6.746175,-2.701608,-9.429713,-2.060022],[-5.345029,-4.552202,-7.901951,8.417266,6.842019,3.910131,8.934582,7.471585,-3.104693,0.472557,-4.921748,-7.851517,7.594960,-9.256651],[7.194183,-1.982017,-0.260041,1.797497,-7.416255,-2.573126,-7.642522,8.805651,2.850590,-6.144752,6.007105,9.619193,8.711519,1.378670],[-5.073427,-7.530699,-8.268888,0.355284,1.901699,2.390404,5.005542,3.349274,6.062872,3.207572,-5.867028,-7.768806,8.077221,2.530125],[6.755536,5.625998,-8.284775,-6.788825,4.040554,-6.475601,-2.385427,-4.344402,2.552115,8.216352,-0.274988,8.817628,0.515030,-6.001638],[8.032666,1.777610,3.651810,-2.036472,-8.749732,-4.733106,-8.220794,9.248100,9.665052,3.895333,2.659413,5.603725,4.254797,-3.329389],[6.566481,-6.805054,1.450048,4.490327,-6.895521,-9.853703,-0.895107,-4.531429,5.004757,-9.624478,-6.626583,4.096011,-7.173954,0.375239],[-4.903908,-3.373758,5.806685,-4.150916,-7.291106,9.696920,-8.490080,-8.833425,-8.514870,2.097047,-6.166471,-2.741508,4.599521,-9.917564],[1.990574,-2.919611,-7.128214,7.162577,0.801166,-6.179271,1.777274,-8.349477,4.560850,1.833863,-7.509982,-7.073449,-7.652017,3.808678]],[[-4.973167,9.493230,6.850333,2.828773,9.534593,5.208845,0.505579,-0.993734,9.342142,-6.715348,6.299862,9.457378,-6.632485,-8.194496],[-1.129894,9.930948,-2.053381,-9.129609,-3.582249,3.401765,-3.711738,-6.321678,-2.948993,5.157864,8.910226,-4.352315,3.513652,-4.261417],[9.799612,-6.616561,-8.185883,2.299259,-2.839026,4.139383,7.948338,-1.734379,6.610911,6.701956,-6.832649,-2.110066,-8.528302,-9.033442],[-8.797239,-1.631108,-1.036838,6.504299,-2.440445,1.903736,8.420585,4.340486,8.392787,-6.172941,-4.369373,-1.206620,-8.415535,-0.501212],[7.601346,-1.286979,-9.908461,-0.503855,6.685805,2.322443,-6.335983,3.366129,7.160193,-9.273374,-5.216018,-4.149552,3.201852,4.823189],[6.942714,-8.632263,9.452566,-1.289219,9.050264,-3.168643,-1.970127,-4.460288,-9.773038,-9.974950,-8.663283,-9.436051,-7.856087,0.706768],[3.878130,3.611862,8.541388,3.057465,0.905792,-0.584871,-5.508678,4.365749,-4.234123,6.691115,4.991231,1.112052,2.293248,1.856498],[-2.753333,5.402701,6.937339,-0.106538,-8.586804,9.105856,3.498955,1.693204,-6.458059,-3.901573,-5.280394,8.669026,-8.132820,-3.871206],[-0.921717,1.848942,-4.285949,-4.515999,2.129381,-2.757911,8.099663,3.582491,6.514022,-6.125541,-6.875276,5.281567,7.866526,-2.534631],[1.563259,7.061948,-8.731711,-3.301871,3.009029,0.403313,3.172851,-5.519181,6.192491,8.158116,-5.002731,5.680911,-5.438521,-4.844110],[2.479586,9.890337,0.990301,6.421186,5.720968,-3.294833,-5.074534,-3.522800,-0.867666,-5.525043,-5.915848,-2.853228,-5.675104,1.589857],[-4.389680,-0.831937,5.662019,-0.096612,-0.665881,-2.949341,2.749123,-0.277229,-1.660971,5.823759,0.772765,1.363793,4.194171,9.724960],[0.645562,3.533581,-7.974716,3.087790,9.458951,-1.884449,6.318774,-7.970714,9.850331,5.197925,7.708400,-7.925743,-9.079412,-7.427182],[-4.991041,3.453417,8.710482,1.493857,-9.900216,8.938951,6.178879,2.826119,-6.971808,5.978867,-9.735731,-6.855008,-0.017772,6.811429],[-5.164507,-1.298792,0.911319,-4.219624,-4.274383,-5.673803,-8.345145,6.871507,1.194131,5.189654,-1.250956,1.386462,7.578223,-0.712340],[7.241661,-6.602016,3.654483,-3.017602,2.295501,6.113464,-1.457106,-7.269061,6.930103,-4.799219,2.059180,4.106210,3.161656,-4.493492]],[[-6.383188,-4.699635,-0.061100,0.020182,9.169768,-2.931837,-6.952709,5.813089,0.154620,1.272764,4.584271,1.636950,9.158155,-1.211071],[-0.385007,7.707116,9.085645,-0.189784,9.883121,5.778914,6.714048,3.143368,-7.664170,-8.103608,-6.312665,-7.833543,-3.817720,-0.510491],[-8.043279,-3.207936,-1.973260,6.421200,5.964358,9.509315,1.658589,5.292275,7.865530,-4.266483,5.827452,1.024536,3.043755,4.974293],[-1.941533,5.905262,-9.123518,6.823759,-7.997892,-6.549010,-7.008295,-6.477008,8.606223,-5.789209,-9.233160,-2.867522,0.733778,9.335734],[2.806955,-6.241671,6.106460,2.382047,2.445718,-6.451800,7.560592,4.524734,-2.636670,-8.431540,-5.557409,-5.093058,6.562181,-9.308019],[9.466557,6.495669,9.768399,-5.984168,-8.586003,-3.123811,-2.475411,0.336174,-9.981500,5.702926,-3.104211,-6.767191,-0.858860,2.103964],[9.784576,-7.759386,2.881016,8.047110,-8.857212,0.408345,7.143719,-3.581400,-1.970265,-4.087748,4.302838,-1.093350,0.488082,-3.860479],[9.066568,9.651683,7.412114,8.375125,-9.863288,6.663542,-7.810650,3.472282,1.326594,7.914871,9.872276,-4.892041,5.159323,-9.464674],[3.530440,-4.117507,-1.060509,3.118738,-7.368394,-1.339063,4.410375,3.422428,8.643001,-8.617212,-8.400975,-5.854298,7.160819,-8.557363],[0.295114,1.330645,0.609775,-7.057370,6.430913,-5.917167,-4.965875,-3.926748,-8.918088,-6.349435,-5.943773,-9.941149,-3.651250,-8.353871],[3.630552,6.487798,9.614566,5.530915,-9.365453,-1.594080,4.441914,4.024938,-8.572189,-3.770066,6.948691,-2.897657,0.644178,1.291173],[5.045984,1.604692,-2.183177,-4.756698,0.985646,4.554546,7.441969,2.131609,-8.221907,-0.275666,4.198675,2.509980,6.211802,-0.755005],[8.697881,2.404609,-6.918702,9.350220,8.027953,-5.577853,-8.687056,-4.064862,-8.693469,-9.130976,7.192965,3.974699,9.458763,3.373214],[-4.675280,0.919806,9.634809,-3.358810,3.187776,2.403890,9.062139,2.167761,0.546109,9.818114,-2.382372,-7.134829,0.324824,-8.905089],[-1.817191,2.248537,3.755527,-1.506979,6.849009,-9.422621,4.916512,3.053645,-1.106134,-7.251326,7.359021,8.565917,0.084465,7.636909],[9.231581,-5.715937,9.093178,-5.291104,-3.225380,-4.900645,7.588489,4.479171,-0.435707,-1.881712,9.561514,-3.162991,-2.973738,-1.161288]],[[9.094651,-3.434595,0.418902,-6.404311,4.964791,-7.421147,-0.392915,0.837121,9.641283,-5.720734,-4.684613,-4.424108,9.338565,-8.406138],[-3.078874,3.112552,-4.750530,-3.664159,-2.168205,-1.554905,8.180690,-2.599340,8.598664,-8.488491,-3.406479,-5.211460,7.399527,-1.040388],[7.786383,-6.564091,-5.873716,6.696291,-1.840076,-6.128513,3.677511,-0.884060,0.762976,-1.794141,-7.733653,-4.538610,1.122852,4.287480],[9.782480,4.764582,-8.107255,7.462944,2.786946,7.975624,-7.386258,7.374537,5.379545,0.453861,-3.427530,9.437835,-1.455179,4.194434],[7.255025,-2.925919,-6.064304,5.432266,4.050794,3.473047,-0.695824,-4.201691,5.426095,-1.594812,2.186538,8.725871,5.141859,1.097256],[-8.942616,6.527960,6.609474,9.004439,3.652233,8.056734,2.664847,0.813917,8.719405,-0.983405,5.853947,1.862893,-0.101974,-3.888184],[-0.091807,1.464691,-8.195678,-0.137403,-7.720522,-5.941792,-2.099504,3.853251,-2.694590,0.067153,-4.580014,-1.473315,-2.374640,4.578146],[7.294681,-6.278393,7.278094,-1.560130,-1.880773,6.456683,-6.124815,-8.920390,5.963337,-4.308954,7.295039,-9.265898,-1.204888,4.358176],[-2.484154,2.662604,5.155150,6.140607,-2.635535,-0.966715,-2.867189,3.966107,7.587300,-3.110800,2.705648,3.771259,-5.629369,4.445651],[-0.587456,-3.207773,8.247903,-9.259981,8.401119,9.248239,-3.102572,7.237398,1.783579,8.023275,-1.426825,-9.877969,0.148336,-6.718211],[0.060427,-0.184491,5.298528,-9.023670,-7.714508,2.068829,8.608098,7.705927,-2.783401,-5.094527,-4.952471,8.983341,8.133553,-2.818771],[7.518860,5.095747,-8.919184,-3.475127,-7.104272,-0.991394,-4.724452,-9.934531,-3.739419,5.478460,-7.079297,2.776059,3.084536,-6.943665],[-0.832061,6.779814,6.054803,-0.597662,-1.679246,4.169843,5.363534,-1.448101,7.804966,0.771606,-2.918099,1.985515,5.502702,0.832262],[7.257400,4.849418,9.439277,3.685735,3.981413,-6.883691,-9.519779,-8.516982,-8.995247,2.375336,-0.215856,6.952203,4.322770,6.396126],[6.113148,-3.368234,-4.346307,-8.562311,-8.189855,7.892215,0.703397,-5.899797,2.216814,-7.147243,0.626630,-5.902117,-7.078924,-3.252066],[-5.936770,-4.590827,1.393089,-8.003711,9.884171,9.144351,-2.229602,3.969946,-3.176438,9.103001,7.912580,7.500399,-9.251289,-7.564456]],[[-3.274157,-1.537431,-8.763568,-0.648318,3.397966,-0.807569,-6.825304,9.439369,-4.797250,-0.983628,-8.170325,-0.911480,7.601673,-7.500660],[4.504362,0.019225,8.367848,-3.401312,-3.479447,2.799551,0.620918,-5.285487,1.381585,-7.948344,5.684839,4.909564,4.959488,8.617301],[2.157515,-0.588160,-7.353568,-3.671074,0.234406,7.654788,-0.394876,-1.241953,-8.561600,7.984897,-3.070086,3.489499,-7.579010,6.455900],[4.631044,5.956634,-7.676910,0.469448,4.491971,-4.463328,4.641733,-8.774831,0.008036,0.757315,-1.769246,4.925979,4.691556,-1.739900],[1.074334,1.517592,-5.861005,-6.257071,-3.039434,1.726780,5.646014,-5.275433,4.708617,-6.131349,1.758322,2.198579,-3.917105,-0.560061],[5.182715,-5.829431,-4.604365,2.847509,-5.984549,8.195188,-3.379045,-5.986048,6.154784,-5.584380,8.331154,9.022755,-7.053536,-7.611146],[9.299386,-6.038931,4.460798,-4.438990,-5.460786,-5.728951,9.636140,4.998339,-3.781642,9.614728,0.254312,2.563724,0.870583,6.606013],[0.358272,-1.090777,0.623288,8.184370,9.920683,-7.911017,-4.844170,-2.202583,3.095162,0.015470,2.623354,9.209259,-1.844000,-6.048810],[5.476661,2.697608,1.425116,0.475622,3.461250,-2.552992,-7.641675,-5.590293,2.312562,-2.333632,3.764135,6.133591,5.819797,2.276227],[8.107148,1.630025,2.048633,3.773036,-4.789324,2.109475,5.440200,6.263739,4.072932,-6.875776,-2.380108,0.357937,-2.623984,4.735312],[3.052606,6.587624,-7.502053,8.261953,-0.991982,-1.560796,9.659016,-9.010387,3.337364,5.821466,2.212082,-9.663484,-5.066531,7.879808],[-3.302224,-0.355550,1.036339,-3.550922,-2.052740,-1.545291,-9.336677,-7.115989,-6.034546,-5.605746,-2.341780,-8.387164,-2.602949,2.648273],[-6.600702,9.865922,1.827904,-1.087980,-4.266331,-4.768525,-7.204894,4.897960,-9.946643,3.595997,-2.291810,-0.557610,1.089885,-3.443030],[7.561389,-9.504497,6.292741,-5.339162,-0.838322,9.489700,4.917288,3.275272,-5.261175,-4.062496,-3.468705,-1.219223,-1.946498,0.116316],[0.079451,7.900788,-2.600237,2.480823,8.956339,-0.199438,-5.680789,-3.188642,-4.936263,-1.873874,-0.584079,2.887137,-6.133007,0.948853],[-4.472824,-0.757743,5.757178,9.064313,0.548998,1.183590,-5.745767,-4.111237,0.665268,2.570792,4.574482,4.147682,-8.943042,-3.982169]],[[-8.326736,0.205196,9.765856,-0.557679,8.088610,-5.044981,-8.540287,-0.410781,8.751419,-9.913550,1.650564,-9.551581,1.904493,6.802913],[-3.667350,-2.018208,4.126340,-2.167982,6.804472,6.067399,-6.128560,1.278251,3.320984,1.141393,3.169753,-2.027622,2.254734,1.226412],[-9.005904,8.562183,-6.510149,3.294765,4.965794,-1.525718,-6.660595,-4.173753,0.032822,-8.380211,2.157738,-6.716406,6.792462,5.309213],[5.331188,3.269299,4.953667,4.679987,-8.241887,-7.692565,-4.480083,-0.201289,-6.026483,8.429306,-4.085455,5.981601,6.085302,-1.216519],[-1.850659,-7.732132,-9.085035,1.243285,-4.069981,-4.116440,6.586251,-9.001783,4.059285,-7.104975,1.817878,-3.988688,9.816959,4.085184],[4.875181,5.303533,-9.188021,7.887169,8.542270,-4.214360,3.375517,-6.231981,0.288125,-6.793009,5.759484,0.316729,-2.164318,9.928722],[9.043807,3.058034,-8.737576,-5.743143,-2.888611,-6.342258,-5.416198,6.669005,5.358856,-4.379001,-3.137730,5.526124,0.743505,0.550736],[-9.530822,-3.460948,2.353100,-9.188197,-0.837200,-7.142342,5.301411,0.625897,-3.595731,0.015218,-4.946002,-6.517998,8.804071,9.181894],[-4.625683,-3.111415,5.359529,-7.770036,-5.310944,-3.724238,9.019065,1.172959,9.189239,-2.154045,2.299368,8.861208,7.375869,4.047713],[-3.449217,-4.066912,6.625687,7.475790,2.232956,3.466714,4.689749,-3.173429,9.892520,5.999759,-3.556033,-8.077567,8.235500,-3.682444],[6.478905,7.117372,-5.578489,4.681097,1.344555,-7.404701,-7.665378,-1.786549,3.668505,4.473768,2.288005,-1.121605,-4.611727,2.320522],[-2.750881,7.869168,9.816833,3.431591,-4.306518,7.339169,-9.470726,-1.733721,-5.920328,-1.454925,2.879532,-0.830751,-0.314470,9.771237],[2.875492,-7.396358,-1.431942,7.499511,2.256927,0.954463,3.963765,8.709825,-3.355169,4.146517,-0.861398,-3.429121,8.122863,-8.946340],[2.898266,-7.144354,3.460697,0.674102,-1.019081,7.432742,1.591473,2.058808,-7.798355,-5.385031,-7.744885,8.760026,1.896100,0.047313],[-3.185662,-4.232680,4.597644,-3.222631,-2.477050,-5.544108,-0.331421,-5.993766,-5.631472,1.898637,-7.203370,8.296906,3.508406,9.918305],[-1.693061,8.950413,8.496295,0.932292,-9.997616,0.721190,1.631313,-5.388038,6.805507,1.049902,-5.281525,-4.586399,0.238405,-3.959288]],[[-0.509714,4.467175,-5.467962,6.009932,1.892086,-1.123868,-7.250982,-4.452938,2.996802,-2.990911,5.422501,-2.692942,-3.770246,-1.376389],[-1.093404,-4.930894,-4.499329,1.218741,-4.503003,-1.503228,6.623077,6.473395,-0.167623,4.778239,-3.030094,7.204140,-8.499381,-7.526155],[-5.969925,-9.271215,9.440751,-6.094820,-9.496714,-6.209840,-8.749937,-0.979461,7.523723,-1.565413,9.352250,0.249487,4.381694,9.528215],[-4.689946,-5.215950,-0.894884,1.842562,4.010864,-7.865112,7.897925,-4.362402,6.072233,-0.163303,-8.708313,1.994949,-0.775909,-8.532103],[2.329334,6.717419,1.028545,-4.201844,-2.887438,0.843145,2.289324,-5.984263,-6.611878,7.890393,6.089581,-4.129717,-0.899309,-9.841334],[-0.148518,-4.657515,-2.393161,4.382886,5.961508,-1.207734,0.159196,5.924337,0.034949,7.538038,-0.367893,9.391414,-4.020350,7.428215],[0.571942,-2.367411,-1.037716,9.021981,-7.679181,-9.728773,0.944739,-1.668319,-9.500662,7.176408,1.631411,6.959271,0.889402,0.437203],[-1.643350,-4.805601,-1.752626,9.811162,5.039934,-6.136301,5.007479,1.626170,8.804148,7.215844,1.782623,0.543303,-4.193523,-5.571755],[6.484948,1.102340,7.496844,-5.018634,-8.939356,-6.624523,-6.380446,-4.238999,-8.624902,-2.625769,6.295604,8.766415,6.231530,1.253674],[-6.114501,7.843342,2.361568,7.169897,1.653688,0.133878,-9.113188,1.676015,-9.451552,-0.984539,-8.664845,-5.761854,6.744058,-7.307405],[-3.615677,-3.438264,-2.295041,-9.602527,-2.099551,-0.221794,3.662037,-8.096383,7.654297,4.931695,4.252651,-7.635675,-6.942921,1.823108],[5.544228,-6.540603,3.095865,-8.750914,9.160163,5.484667,-7.328740,4.595268,0.366341,4.365820,9.825431,-6.284112,7.523192,-5.445383],[2.717272,1.519323,-9.149083,-7.400107,-2.520801,9.581622,-4.996783,9.098159,-3.744334,-0.635545,-6.837917,-5.532141,-0.304576,-0.401674],[-2.317712,-8.264327,-0.920685,-3.883753,3.945413,-7.725011,9.481579,0.373579,9.459939,-2.700119,9.420154,1.003495,6.681086,-4.145658],[8.910398,4.182102,5.057818,3.184986,0.268112,-1.544575,1.127945,-4.864180,-5.960805,-3.394675,-3.284890,4.737287,5.090283,-5.129107],[2.769150,4.337769,7.278869,-8.456922,-7.588977,5.601049,9.695683,-4.525076,-8.382285,8.842548,4.300727,-9.782769,-7.392026,1.128623]],[[8.878093,-6.564776,-5.380230,8.807662,-5.740634,5.083475,6.546073,-9.419176,-5.303766,8.249478,2.425279,-6.634701,-7.620459,4.515490],[-4.651737,2.231619,-3.220886,-3.885582,0.594962,-1.758396,7.806221,-5.400043,-8.682994,4.641406,6.891839,-1.567736,6.457655,2.570774],[5.797626,9.006599,-2.529978,9.540881,-8.550342,0.821662,2.551966,-8.437095,7.189067,-5.611567,8.144681,2.371121,-7.750586,-9.746010],[-6.810083,-7.722794,-0.339153,1.146036,-6.741828,8.485735,0.603930,-7.780222,-8.212580,7.991985,9.196675,-6.623555,-3.485399,0.006346],[5.381557,-7.637099,-9.970898,3.845367,-3.399899,7.025696,-8.598108,-8.985559,6.822484,-0.591948,2.799005,-7.541420,0.610390,7.198693],[-2.770973,5.695451,-1.107149,4.038666,-2.949849,9.156186,1.337439,4.284021,8.607804,4.546921,-0.798305,3.318748,-0.237423,-2.959417],[-2.813876,1.437137,-3.514930,-9.890069,-0.942621,7.179491,-3.700295,-3.020328,3.245620,0.687686,4.897354,-0.949860,8.460896,5.593462],[9.035161,8.445218,4.784239,-6.824171,3.577256,6.998244,-0.159464,-0.049258,8.170950,-8.975790,-8.724735,-9.475926,6.053443,6.478291],[-0.393768,-5.098517,-4.172291,6.664042,-3.834620,-5.914694,-3.746143,6.602978,-8.021910,6.396879,-6.551400,6.083182,-9.352459,-3.206007],[-3.371470,-9.336177,-9.406728,-8.502308,-8.658106,8.333887,5.230943,1.428398,7.248317,-9.838836,-1.313780,6.383022,3.422374,-3.958781],[-8.782541,5.835595,-8.688895,2.590620,-5.915059,-3.307836,5.050178,1.423460,5.272727,-2.346923,-5.895646,-5.060174,5.818824,2.752330],[-6.293135,-6.129243,5.783774,3.012994,8.980792,-0.447920,-2.432160,-7.185153,-1.937067,-3.446146,-0.577976,-3.790297,5.741799,-7.470379],[9.742779,-1.410375,9.150095,-6.034838,-5.522610,-3.754176,1.187884,-5.351306,8.868455,0.152300,-7.115882,2.549581,2.767972,8.814683],[3.881968,3.257176,-5.009003,-7.376686,-1.172283,-2.255899,9.063008,-7.826211,8.212383,-0.473438,4.631413,0.405749,-5.874821,0.210339],[-2.718070,-9.790853,-9.778919,-4.926343,3.840174,-0.306163,-5.430010,7.796370,1.860552,-5.148178,-0.953535,-5.573552,-3.694463,1.964219],[-4.864197,-2.317745,-4.634197,-4.674836,-8.934984,4.262076,-4.655621,6.059334,-1.183823,-7.818490,-7.173454,-1.437226,3.658848,1.283448]],[[-4.905580,-6.377413,0.827421,-9.372214,-1.060039,-5.085373,2.356645,-4.349852,2.900911,-8.058314,-4.207847,7.596898,3.313831,1.658329],[-4.531277,5.323892,-6.853980,-5.178464,-1.002233,1.003881,4.755524,-0.147268,5.621163,-1.515396,-5.672145,-9.716501,2.407629,-2.544944],[9.898384,-5.300210,-2.534350,3.499881,4.208923,-5.309104,-0.634839,-8.136100,6.382777,-7.895907,-1.502727,4.395927,-9.173681,4.678843],[4.276270,-5.598383,-8.008209,1.995877,-4.195909,5.506466,-8.395411,-4.934061,9.280272,-2.318917,8.576889,1.156829,-6.258672,-2.440014],[-4.791555,-1.343507,2.972534,-6.043813,4.020365,-7.374420,-7.152189,-0.757223,9.589042,7.662880,5.302311,5.661052,5.205194,-4.226080],[0.943636,-1.270184,9.828489,-6.087693,2.380971,-1.931695,-1.005581,9.760637,-5.120916,2.778593,-5.503330,-8.426987,-8.795362,-2.211981],[-2.989485,5.721511,6.574124,5.371293,8.593986,-0.531132,-5.887763,1.090165,6.574265,-4.669113,-3.917794,-3.999009,-5.521161,3.155879],[-2.345146,-8.613796,-7.469872,2.607942,6.281240,-1.950315,-3.686048,-5.811410,0.477977,6.435719,-7.255781,-1.494923,1.295986,8.945411],[4.630312,-4.405900,-6.844210,-1.502835,0.048718,-8.018786,-4.641843,3.499946,6.839851,-9.892740,1.322895,-5.830589,9.862384,9.629077],[8.319189,-7.973253,6.221815,-9.657633,-6.945646,-9.804874,-7.252851,-2.805336,-6.336374,3.788090,0.829045,-5.540511,9.986975,-0.805315],[-4.045172,3.943951,-7.776409,-7.992784,3.656523,1.375443,-2.615931,8.262440,5.540203,7.103176,4.068576,1.552448,-9.126363,0.972186],[-5.770087,-1.712061,4.617951,-7.732110,-5.709103,6.620014,-6.592455,4.405524,3.448962,2.269252,-5.830173,-0.614250,-7.695802,1.180848],[9.927006,-9.947237,-9.576551,-2.868967,6.474178,-7.596440,-5.390094,8.458531,-8.085771,4.637814,5.836079,-8.503880,7.668739,7.040041],[9.326227,7.387928,3.054831,-2.367631,-5.257689,-7.241238,-1.013191,0.837311,-7.926947,3.182803,7.918466,-8.118276,0.118253,7.968073],[6.418016,-9.409994,7.160563,-7.438091,-2.924360,-5.420172,0.996276,-4.182125,1.905770,-0.075202,-7.551726,6.664137,-6.470844,-2.039587],[-2.436137,6.984215,-0.811194,-0.330290,4.544873,-4.942718,-6.816581,-1.857316,9.961733,2.110766,3.245028,2.256210,7.590156,6.085044]],[[-5.943793,7.555000,2.313522,4.079962,7.587511,4.159043,-7.784972,-4.329997,2.907128,-9.091664,5.954170,5.422424,-8.121285,-2.782441],[8.271677,4.021072,8.601138,3.747175,0.409665,-0.206422,-4.758248,9.881011,5.528187,7.903009,8.248743,-4.848941,7.890059,4.959716],[-8.152623,4.771144,8.867479,2.490793,-3.984446,-7.973864,0.772561,-9.383817,-0.334433,-5.796649,3.823575,-9.559595,-0.626319,1.886397],[0.213720,9.639421,-9.348262,1.516778,5.575925,-5.156081,8.784009,-3.239515,3.409504,9.275636,-2.419966,7.815362,6.546337,-3.312535],[-9.151347,-1.539143,-4.946113,2.768458,-4.723196,3.113466,-6.120566,8.441065,-3.207634,-2.090246,-5.527114,3.765657,-7.197293,-7.805214],[9.566545,9.246747,-5.236899,9.476635,4.595229,5.159155,8.755183,1.503868,-8.585045,-3.841400,-6.209191,-3.572912,-5.855703,-8.517938],[2.465797,-3.127768,4.416411,-7.073451,8.591131,6.132624,9.229753,1.326110,3.088234,-4.657264,6.845416,-7.231494,3.459758,2.113363],[-7.769908,-3.100278,-1.140772,-5.922522,-6.700105,-6.194975,-1.948370,-4.225284,-4.311315,0.992923,4.004875,-4.567353,-4.690641,-3.837285],[3.014265,3.825072,9.472006,8.079553,-9.797790,-2.846994,5.737095,3.324534,-2.270265,7.151066,0.822270,-9.126022,-3.728792,9.999100],[4.335630,-6.448463,3.607168,-7.562781,-0.108111,0.972842,-2.144297,2.650862,-6.890806,4.374266,1.929598,6.545103,9.610959,6.750294],[-6.313039,-1.328575,-2.771934,1.217240,-0.545520,2.254262,2.564696,-6.573730,-7.801617,5.864875,-8.176627,2.355418,3.080092,-5.929416],[-1.438689,-9.644255,-6.039669,-8.461674,-7.244338,7.335855,4.196050,5.381481,1.099184,-1.522015,9.814087,-3.571264,9.608740,3.765477],[-8.443541,7.060630,8.235887,-3.537656,7.817955,9.110988,-6.410572,-8.523013,4.294583,-2.991204,6.095143,-6.541451,4.287423,-2.658035],[-8.587300,7.561311,9.697555,4.481048,0.209370,3.976437,-3.103784,-3.596425,8.073037,-5.415583,-5.755314,5.438817,-2.768857,-3.049572],[-9.475217,-1.638103,4.538095,2.236834,-7.531963,4.052905,-7.495758,-0.362028,0.263449,4.380305,-0.784804,6.107488,5.253003,-4.054640],[-5.082594,3.871503,-0.329429,2.660967,1.432526,-5.924701,7.523278,6.145334,-1.660045,-1.743880,-3.405121,-6.221370,-9.051765,5.035230]]], dtype = "float32")#candidate|996|(16, 16, 14)|const|float32
bop_997 = relay.floor_mod(uop_990.astype('float64'), relay.reshape(const_996.astype('float64'), relay.shape_of(uop_990))) # shape=(16, 16, 14)
uop_1001 = relay.acosh(uop_990.astype('float64')) # shape=(16, 16, 14)
bop_1003 = relay.multiply(uop_1001.astype('uint64'), relay.reshape(bop_984.astype('uint64'), relay.shape_of(uop_1001))) # shape=(16, 16, 14)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
const_1007 = relay.const([-6.375851,-8.868508,4.741348,0.518659,0.675194,-7.782006,3.964079,8.371991,1.700067,-1.967102,3.149121,-3.477721,6.620690,9.163990,-7.257035,4.028632,-7.864968,6.186338,-0.663506,-8.819000,-4.879584,-1.823583,0.416254,0.800884,-2.981710,-8.426151,8.364078,5.028146,-3.188865,5.931935,6.896411,-3.520031,-3.188068,1.305555,-0.135704,-6.517654,6.205653,1.068307,-6.943890,3.530631,-5.216770,-9.567157,-2.968168,-5.772429,9.552492,8.207593,-7.856662,-2.199329,-6.164334,2.516725,3.590308,7.171951,5.922901,3.743681,-9.224632,-3.644749,9.283265,-5.848336,-9.768096,-8.461433,-7.495221,-8.120570,3.878427,7.683757,4.095488,2.939035,4.258344,1.473897,3.922291,-1.667066,2.170615,-5.087341,-1.085137,-9.371691,-1.036241,-7.962333,-6.659048,6.175316,4.433070,9.972381,-9.630959,-5.659313,-5.409121,-2.239766,-1.653317,5.965642,0.527594,-7.621499,-9.309412,-5.658224,-1.099210,7.727961,4.045429,-0.291886,7.922252,7.092844,-6.408198,7.518011,0.843981,-1.723353,6.872261,1.696393,-0.329855,-1.156362,-5.319066,3.350976,5.026091,-5.621397,7.520291,9.125876], dtype = "float32")#candidate|1007|(110,)|const|float32
call_1006 = relay.TupleGetItem(func_120_call(relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), ), 0)
call_1008 = relay.TupleGetItem(func_126_call(relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), relay.reshape(const_1007.astype('float32'), [11, 2, 5]), ), 0)
func_328_call = mod.get_global_var('func_328')
func_332_call = mutated_mod.get_global_var('func_332')
const_1011 = relay.const([4.356262,6.186539,8.225938,9.745306,-9.302467,7.009799,1.031168,-9.381252,-2.589411,-7.670870,-6.464749,-4.991953,-0.033573,8.380268,-8.336717,4.128383,-6.037828,-3.128099,-4.662496,8.341264,-5.735081,9.834436,2.898396,0.909893,-3.835372,9.732888,-3.122528,0.349741,-8.955161,-9.720136,-1.081179,-9.140670,4.357388,2.078771,8.409323,-0.830886,6.051550,3.267279,5.113710,-8.296037,8.935475,-3.471658,4.491683,-1.348641,-6.688524,-4.260886,-0.569238,-5.857478,6.011107,-3.781826,-8.407663,1.832571,-7.527626,-8.383267,9.355471,2.196162,-7.526348,-5.970447,-0.633728,-9.107355,-7.187806,5.924106,3.580393,-0.733820,0.885752,8.159426,0.916789,6.511300,-0.033486,7.293158,-3.632787,-6.002141,0.110009,-8.061795,1.758813,-9.908574,-8.157908,4.079207,0.814776,-7.913788,-6.505252,-8.555756,-9.465247,6.316714,-6.483757,-2.685178,-9.665135,7.725638,-9.310047,4.998749,-5.892533,8.697485,1.358968,4.475799,-7.155742,-7.791260,3.032994,-3.724207,-1.247586,-8.804617,-0.358163,-5.177023,7.568068,-3.129425,9.550762,-2.042754,-5.481377,-1.947074,-6.810236,-7.372692,-3.937387,5.363099,-7.020039,6.165074,3.498405,1.389225,3.078828,-5.044981,8.436142,-4.702358,-0.146585,7.730386,-6.302783,0.596810,5.943235,0.224561], dtype = "float64")#candidate|1011|(126,)|const|float64
var_1012 = relay.var("var_1012", dtype = "float32", shape = (240,))#candidate|1012|(240,)|var|float32
call_1010 = relay.TupleGetItem(func_328_call(relay.reshape(const_1011.astype('float64'), [9, 14]), relay.reshape(var_1012.astype('float32'), [240,]), relay.reshape(const_1011.astype('float64'), [9, 14]), ), 3)
call_1013 = relay.TupleGetItem(func_332_call(relay.reshape(const_1011.astype('float64'), [9, 14]), relay.reshape(var_1012.astype('float32'), [240,]), relay.reshape(const_1011.astype('float64'), [9, 14]), ), 3)
output = relay.Tuple([call_987,var_988,bop_992,bop_997,bop_1003,call_1006,const_1007,call_1010,const_1011,var_1012,])
output2 = relay.Tuple([call_989,var_988,bop_992,bop_997,bop_1003,call_1008,const_1007,call_1013,const_1011,var_1012,])
func_1018 = relay.Function([var_982,var_988,var_1012,], output)
mod['func_1018'] = func_1018
mod = relay.transform.InferType()(mod)
var_1019 = relay.var("var_1019", dtype = "float32", shape = (16, 16, 14))#candidate|1019|(16, 16, 14)|var|float32
var_1020 = relay.var("var_1020", dtype = "uint64", shape = (91,))#candidate|1020|(91,)|var|uint64
var_1021 = relay.var("var_1021", dtype = "float32", shape = (240,))#candidate|1021|(240,)|var|float32
output = func_1018(var_1019,var_1020,var_1021,)
func_1022 = relay.Function([var_1019,var_1020,var_1021,], output)
mutated_mod['func_1022'] = func_1022
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1026 = relay.var("var_1026", dtype = "float64", shape = (7, 1))#candidate|1026|(7, 1)|var|float64
var_1027 = relay.var("var_1027", dtype = "float64", shape = (7, 7))#candidate|1027|(7, 7)|var|float64
bop_1028 = relay.power(var_1026.astype('float64'), var_1027.astype('float64')) # shape=(7, 7)
output = bop_1028
output2 = bop_1028
func_1046 = relay.Function([var_1026,var_1027,], output)
mod['func_1046'] = func_1046
mod = relay.transform.InferType()(mod)
var_1047 = relay.var("var_1047", dtype = "float64", shape = (7, 1))#candidate|1047|(7, 1)|var|float64
var_1048 = relay.var("var_1048", dtype = "float64", shape = (7, 7))#candidate|1048|(7, 7)|var|float64
output = func_1046(var_1047,var_1048,)
func_1049 = relay.Function([var_1047,var_1048,], output)
mutated_mod['func_1049'] = func_1049
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1056 = relay.var("var_1056", dtype = "float32", shape = (1, 15, 2))#candidate|1056|(1, 15, 2)|var|float32
uop_1057 = relay.rsqrt(var_1056.astype('float32')) # shape=(1, 15, 2)
bop_1059 = relay.subtract(uop_1057.astype('float32'), relay.reshape(var_1056.astype('float32'), relay.shape_of(uop_1057))) # shape=(1, 15, 2)
bop_1071 = relay.logical_or(bop_1059.astype('bool'), relay.reshape(uop_1057.astype('bool'), relay.shape_of(bop_1059))) # shape=(1, 15, 2)
var_1077 = relay.var("var_1077", dtype = "float32", shape = (15, 15, 2))#candidate|1077|(15, 15, 2)|var|float32
bop_1078 = relay.floor_mod(var_1056.astype('float32'), var_1077.astype('float32')) # shape=(15, 15, 2)
bop_1081 = relay.less_equal(bop_1071.astype('bool'), bop_1078.astype('bool')) # shape=(15, 15, 2)
func_831_call = mod.get_global_var('func_831')
func_833_call = mutated_mod.get_global_var('func_833')
const_1094 = relay.const([7.695461,-3.421093,4.867620,8.481569,6.700819,0.546307,7.290502,-1.942445,8.203379,-8.264120,-2.356293,8.825919,-5.102434,2.917218,-3.469680,-3.404005,-3.978239,8.974932,1.415857,-8.420536,-3.301453,2.813094,9.102854,-8.417514,-9.810046,-1.637818,-3.058072,4.762289,3.122867,-6.098167,1.252317,-2.389066,5.976509,6.797909,-0.293805,-4.021436,5.586503,-6.831194,3.505334,-1.228063,-5.651256,-4.238887,-3.482398,6.960678,1.096560,3.737575,7.058975,-6.270027,0.783624,9.865223,3.694652,6.061547,5.293845,2.216413,-5.291780,-8.843259,6.310548,4.627274,-8.696839,7.050847,-0.117802,1.042264,-6.708006,-7.832644,7.808182,-2.562644,5.071604,-5.114542,0.055924,-2.159921,5.792474,-9.262009,4.148213,-0.440646,2.427713,6.160685,5.460937,0.892292,-2.130491,5.229259,0.166429,4.586638,8.500964,-6.387006,8.139490,-5.837380,3.592719,8.123196,-4.999183,3.829302,-6.934332,2.494935,8.487226,2.148173,3.622081,2.995688,-3.807791,1.023478,-5.498434,5.311615,-9.638430,-8.931908,-7.398555,2.009136,-1.145259,-0.720774,3.601333,-3.360627,-6.925765,5.019095,-4.274354,7.911099,4.014031,6.817529,-4.035954,9.460588,-3.598925,-1.943532,-5.686900,-5.006265,-4.987110,8.011171,-6.830134,-1.789197,3.306480,-9.828564,2.356357,2.417942,0.470264,-4.046238,2.320708,-2.048545,-4.905017,-1.891354,-8.484217,8.337741,-2.344402,7.752684,-8.125595,0.732823,5.057104,0.586499,7.393435,5.816023,0.195528,-6.581703,9.254572,6.517335,-8.757113,-3.866725,-6.353242,3.526663,-5.274945,9.173988,5.904023,3.277753,3.194151,1.786999,9.386889,4.469500,-2.186151,7.861739,-7.463611,8.152740,-5.694398,-8.798650,1.600191,3.477879], dtype = "float32")#candidate|1094|(168,)|const|float32
call_1093 = relay.TupleGetItem(func_831_call(relay.reshape(const_1094.astype('float32'), [6, 4, 7])), 0)
call_1095 = relay.TupleGetItem(func_833_call(relay.reshape(const_1094.astype('float32'), [6, 4, 7])), 0)
func_1046_call = mod.get_global_var('func_1046')
func_1049_call = mutated_mod.get_global_var('func_1049')
var_1106 = relay.var("var_1106", dtype = "float64", shape = (1, 7))#candidate|1106|(1, 7)|var|float64
const_1107 = relay.const([6.366601,-8.173250,-6.413263,8.414891,6.748029,-4.896483,6.979260,-7.694262,6.838393,-3.151217,9.836798,-5.123688,-4.866624,-6.485401,3.782224,-3.103146,-8.994633,-2.806123,-0.742467,2.139298,5.542574,9.433563,-6.419774,-6.665983,2.559635,4.707949,5.505969,-7.093232,1.897354,-8.360143,-3.328439,8.157378,8.283947,2.912416,-6.706754,4.091866,0.603025,-9.887528,-3.672408,6.323351,6.359441,0.490690,-8.368390,-6.624946,-4.594714,-4.566822,5.574124,-3.746694,-1.348438], dtype = "float64")#candidate|1107|(49,)|const|float64
call_1105 = func_1046_call(relay.reshape(var_1106.astype('float64'), [7, 1]), relay.reshape(const_1107.astype('float64'), [7, 7]), )
call_1108 = func_1046_call(relay.reshape(var_1106.astype('float64'), [7, 1]), relay.reshape(const_1107.astype('float64'), [7, 7]), )
output = relay.Tuple([bop_1081,call_1093,const_1094,call_1105,var_1106,const_1107,])
output2 = relay.Tuple([bop_1081,call_1095,const_1094,call_1108,var_1106,const_1107,])
func_1109 = relay.Function([var_1056,var_1077,var_1106,], output)
mod['func_1109'] = func_1109
mod = relay.transform.InferType()(mod)
var_1110 = relay.var("var_1110", dtype = "float32", shape = (1, 15, 2))#candidate|1110|(1, 15, 2)|var|float32
var_1111 = relay.var("var_1111", dtype = "float32", shape = (15, 15, 2))#candidate|1111|(15, 15, 2)|var|float32
var_1112 = relay.var("var_1112", dtype = "float64", shape = (1, 7))#candidate|1112|(1, 7)|var|float64
output = func_1109(var_1110,var_1111,var_1112,)
func_1113 = relay.Function([var_1110,var_1111,var_1112,], output)
mutated_mod['func_1113'] = func_1113
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1205 = relay.var("var_1205", dtype = "uint64", shape = ())#candidate|1205|()|var|uint64
var_1206 = relay.var("var_1206", dtype = "uint64", shape = (6,))#candidate|1206|(6,)|var|uint64
bop_1207 = relay.maximum(var_1205.astype('uint64'), var_1206.astype('uint64')) # shape=(6,)
bop_1218 = relay.bitwise_and(bop_1207.astype('int64'), var_1205.astype('int64')) # shape=(6,)
func_120_call = mod.get_global_var('func_120')
func_126_call = mutated_mod.get_global_var('func_126')
const_1223 = relay.const([-0.497008,9.917552,-6.350821,-3.287105,-9.618397,0.200841,-6.582548,7.820567,8.999998,-6.553694,-3.074431,0.589382,-1.433985,1.291699,-4.225316,6.878902,7.518226,3.692117,2.548003,8.400978,8.579243,9.539568,8.575676,1.531426,9.694964,-4.762429,9.548125,-4.857111,-8.533598,-5.257161,-7.249346,-0.702923,-6.783610,0.018992,6.373217,-0.545153,3.978235,-9.044088,-9.772376,9.660008,2.338976,5.828989,9.719724,8.793306,9.935016,-6.684376,9.873515,8.859032,-5.755134,-2.652294,-5.149284,-3.094252,-0.034203,-4.945530,4.253360,1.917415,5.877946,2.748723,-1.679041,-4.670374,-6.524525,1.937840,6.786958,-4.130771,-4.255621,-6.217977,2.877202,-8.296440,1.664417,-6.129740,6.989397,-7.249326,-5.396871,7.614572,-2.620679,-9.849731,2.879495,-9.129833,-2.305735,-8.230327,7.195966,0.314100,-0.218782,4.581207,-6.065377,-1.947916,9.404545,-0.691469,-8.346667,-8.887727,-0.044183,-4.072835,-8.247928,-4.012636,3.407693,6.410327,4.728026,5.533472,0.162680,-9.226691,-9.078650,8.583799,9.991854,3.080218,0.416070,-2.259846,-5.021944,3.940820,9.581583,9.272538], dtype = "float32")#candidate|1223|(110,)|const|float32
call_1222 = relay.TupleGetItem(func_120_call(relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), ), 0)
call_1224 = relay.TupleGetItem(func_126_call(relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), relay.reshape(const_1223.astype('float32'), [11, 2, 5]), ), 0)
output = relay.Tuple([bop_1218,call_1222,const_1223,])
output2 = relay.Tuple([bop_1218,call_1224,const_1223,])
func_1225 = relay.Function([var_1205,var_1206,], output)
mod['func_1225'] = func_1225
mod = relay.transform.InferType()(mod)
var_1226 = relay.var("var_1226", dtype = "uint64", shape = ())#candidate|1226|()|var|uint64
var_1227 = relay.var("var_1227", dtype = "uint64", shape = (6,))#candidate|1227|(6,)|var|uint64
output = func_1225(var_1226,var_1227,)
func_1228 = relay.Function([var_1226,var_1227,], output)
mutated_mod['func_1228'] = func_1228
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1369 = relay.var("var_1369", dtype = "int64", shape = (9, 11))#candidate|1369|(9, 11)|var|int64
const_1370 = relay.const([[-1,7,-8,-7,1,-8,5,9,-7,7,1],[8,-3,-3,-9,6,-10,4,-3,6,4,-8],[6,-3,2,-4,-3,-1,3,3,-9,-2,2],[-8,-8,1,-10,1,8,10,-5,3,7,4],[-7,2,6,6,-1,-2,-1,-10,-2,7,-6],[-10,-1,7,-6,-10,8,9,-7,1,-9,8],[8,1,1,-1,-4,8,-3,-7,-7,-4,-10],[-8,-7,-6,4,6,-2,-1,6,-9,-5,8],[-1,-9,-2,3,5,9,-10,10,-9,-5,5]], dtype = "int64")#candidate|1370|(9, 11)|const|int64
bop_1371 = relay.not_equal(var_1369.astype('bool'), relay.reshape(const_1370.astype('bool'), relay.shape_of(var_1369))) # shape=(9, 11)
func_740_call = mod.get_global_var('func_740')
func_746_call = mutated_mod.get_global_var('func_746')
var_1379 = relay.var("var_1379", dtype = "uint32", shape = (800,))#candidate|1379|(800,)|var|uint32
const_1380 = relay.const([[2.209655,5.409383,9.171306,-3.016487,-5.983192,5.679957,-7.376003,4.232549,8.022684,5.790945],[1.472231,8.419471,-7.829887,7.330989,0.592419,6.130166,-1.253826,6.224580,-3.079444,8.135737],[-1.321403,-9.400451,8.458555,-6.625876,3.947636,0.575793,4.499675,-5.494711,7.877242,-7.101375],[-1.078054,0.175696,-0.194072,-9.139898,-0.723791,9.059922,-5.574031,6.780459,0.967235,-7.133610],[9.085739,2.339908,-2.430939,-1.082325,-3.497492,-4.558804,-0.904770,7.438254,-2.086416,3.172228],[8.048889,-0.810989,4.013899,2.372176,1.727610,1.944586,2.826377,1.124017,-7.705527,5.476499],[-3.054844,9.396190,8.608933,0.101705,-4.647824,-4.308260,5.387597,3.607008,-4.433036,-8.469920],[0.256936,7.793206,-1.305820,-0.059982,-0.142358,3.735096,-7.191130,-2.973979,0.364023,0.246080],[-9.844472,8.022258,-3.008008,-3.980174,-0.018318,1.268753,-6.762801,-8.788657,5.581329,-3.084071],[1.203090,3.633625,-9.320929,5.391335,0.297258,-5.039302,6.129963,-1.474445,-4.520925,-5.642577],[6.869848,-1.931101,-9.311898,-9.752362,-2.122880,-0.145652,1.981005,2.647322,5.229177,6.479737]], dtype = "float32")#candidate|1380|(11, 10)|const|float32
const_1381 = relay.const([2,4,5,4,-8,-3,-3,9,-7,10,-2,4,-1,5,3,-6,1,10,4,4,-1,6,-7,9,-10,9,1,5,8,-7,1,-7,-3,-10,1,9,-1,3,4,4,-2,7,6,-5,-3,5,-6,-10,-9,8,-4,-1,5,-7,-7,4,-1,1,4,-2,-8,3,-9,-3,-1,8,7,4,-3,-1,-9,-10,-6,8,6,-7,6,8,-10,6,9,1,-3,-8,-4,-9,-3,-3,-9,-9,2,-1,5,-6,10,10,-5,-7,-6,7,5,-2,4,-7,10,-9,-1,1,-2,5,-8,-3,-6,1,7,-1,-3,-1,10,-9,-4,-3,9,-3,2,2,2,4,6,-9,5,-7,5,-5,-6,5,3,6,-3,-2,8,-6,-7,-10,8,-10,-1,3,-5,-1,3,10,6,-5,-5,-6,10,5,2,2,7,5,9,6,-6,-2,4,4,-9,9,-7,9,6,-7,-9,10,-3,6,5,-3,-10,4,10,-4,-9,3,7,-5,3,-1,3,10,10,-5,1,-8,6,5,-9,4,-1,6,-6,2,3,6,7,-1,4,-2,7,-6,-3,8,1,10,9,-1,-6,-6,6,-8,-10,10,6,-10,3,5,-8,-7,-7,10,-2,-10,-2,9,8,1,-6,-1,2,-7,-7,2,-7,4,-4,7,3,-6,4,3,5,-2,-4,5,-4,2,5,9,-10,8,-7,-4,4,5,7,2,1,-3,4,-10,10,-2,-5,9,-3,-3,-3,1,7,5,-6,-4,-8,-4,4,-5,-8,-1,9,10,8,4,-10,-9,5,-3,-1,-9,-5,-3,7,-5,-4,-9,6,-7,-3,-8,-8,-4,-10,-3,3,2,-5,8,3,-8,-8,2,-6,-10,9,-4,2,-7,4,-7,-2,-6,4,-10,5,-5,-6,-9,-3,-3,-5,-7,-6,3,-4,6,7,-2,6,-9,-1,6,-9,-2,-5,5,2,9,8,-8,10,8,-8,2,4,6,6,-7,-10,9,1,-5,8,1,-6,8,-4,5,9,-7,1,-2,-9,3,-5,-5,8,1,4,2,4,-5,-8,-2,10,-2,5,7,3,3,4,-6,8,8,3,1,2,6,8,9,-7,10,-8,-6,-9,-9,-9,-3,6,8,-3,-5,-4,-7,8,10,-10,5,-7,9,2,-9,4,-7,-3,-8,-3,-3,5,-7,7,-8,8,-5,6,3,9,-8,-3,-3,-9,5,7,9,8,-5,-5,-2,-6,-10,-8,-2,-2,-1,-7,-3,9,-10,-4,8,-7,6,8,9,-4,8,-1,-10,5,9,-8,-10,-10,3,3,7,6,4,-1,8,1,7,8,-8,-9,-5,-9,3,-9,-3,-10,4,4,-10,10,-10,5,4,5,6,-9,10,-10,-9,-1,-3,3,-7,-5,3,4,-8,-1,2,-6,-9,-3,-5,3,-4,-6,-5,4,7,-8,5,-4,-3,-7,2,-5,-10,-1,1,-1,-10,2,8,-7,5,5,-9,5,5,-7,9,9,10,4,-6,6,2,9,-4,-4,4,-1,8,5,9,-10,3,8,9,-5,-9,-9,9,-2,8,1,3,5,7,-5,-3,7,-4,1,10,-8,-7,1,-7,-10,-7,7,5,3,5,10,-5,-3,-3,10,-10,-2,5,4,6,-9,9,7,1,5,10,-1,4,1,8,-1,-2,-7,8,-1,7,9,9,9,-6,5,10,2,-9,1,6,8,-7,-3,6,9,-7,1,-4,5,3,-7,-5,6,2,3,-1,4,-8,1,4,-9,-8,-4,-1,2,8,-8,10,6,3,-8,10,2,-10,2,-3,-4,3,3,6,7,-7,-4,3,9,-9,2,-10,9,-7,5,-4,-3,6,1,3,7,-5,9,2,3,6,1,10,-7,3,-7,10,3,5,-9,2,-10,-6,4,-6,9,2,7,2,-9,3,-1,5,10,-6,10,-2,4,7,8,-2,-5,-7,-8,-6,3,4,-4,-7,-1,-3,10,9,-5,-1,2,-10,-10,1,5,7,-7,-3,10,-5,-5,-9,-4,-1,7,-2,7,-7,-8,-7,6,-4,-4,3,-8,-2,7,-10,-2,-9,-2,9,4,6,-6,8,6,-6,8,-4,-8,-1,-10,5,1,10,-9,-3,-8,-7,-4,8,-7,-10,-2,5,-1,-6,-5,-2,6,-10,-4,8,-3,-6,-1,7,-9,-7,3,2,6,8,10,10,3,-1,-4,-9,-5,6,-1,6,-3,-10,4,-2,-1,-4,10,7,5,-2,-8,-7,-8,-10,7,-7,-8,-7,-3,3,-9,5,10,-5,-3,-3,10,3,-5,6,-9,-10,-4,-9,6,-8,-6,6,-8,-10,-2,9,5,9,-7,1,10,4,8,8,3,1,-1,-7], dtype = "int8")#candidate|1381|(880,)|const|int8
call_1378 = relay.TupleGetItem(func_740_call(relay.reshape(var_1379.astype('uint32'), [10, 16, 5]), relay.reshape(var_1379.astype('uint32'), [10, 16, 5]), relay.reshape(const_1380.astype('float32'), [110,]), relay.reshape(const_1381.astype('int8'), [880,]), relay.reshape(var_1379.astype('float64'), [10, 16, 5]), ), 0)
call_1382 = relay.TupleGetItem(func_746_call(relay.reshape(var_1379.astype('uint32'), [10, 16, 5]), relay.reshape(var_1379.astype('uint32'), [10, 16, 5]), relay.reshape(const_1380.astype('float32'), [110,]), relay.reshape(const_1381.astype('int8'), [880,]), relay.reshape(var_1379.astype('float64'), [10, 16, 5]), ), 0)
func_430_call = mod.get_global_var('func_430')
func_433_call = mutated_mod.get_global_var('func_433')
const_1390 = relay.const([2,-7,3,10,-7,-2,-7,1,6,-1,1,8,9,-1,7,-4,1,1,-1,5,1,-6,4,8,7,-4,9,10,-3,8,-3,1,6,-10,-3,2,-2,2,-8,-7,-10,-5,-5,6,-8,-10,-7,-4,10,-7,-9,-7,-9,-1,1,-2,-1,1,4,3,-2,-2,3,1,2,-8,-8,5,6,-4,4,-8,-3,-8,-6,8,6,-6], dtype = "uint16")#candidate|1390|(78,)|const|uint16
call_1389 = func_430_call(relay.reshape(const_1390.astype('uint16'), [13, 6]))
call_1391 = func_430_call(relay.reshape(const_1390.astype('uint16'), [13, 6]))
bop_1392 = relay.equal(call_1389.astype('bool'), relay.reshape(const_1390.astype('bool'), relay.shape_of(call_1389))) # shape=(13, 6)
bop_1395 = relay.equal(call_1391.astype('bool'), relay.reshape(const_1390.astype('bool'), relay.shape_of(call_1391))) # shape=(13, 6)
uop_1399 = relay.sqrt(const_1370.astype('float64')) # shape=(9, 11)
output = relay.Tuple([bop_1371,call_1378,var_1379,const_1380,const_1381,bop_1392,uop_1399,])
output2 = relay.Tuple([bop_1371,call_1382,var_1379,const_1380,const_1381,bop_1395,uop_1399,])
func_1401 = relay.Function([var_1369,var_1379,], output)
mod['func_1401'] = func_1401
mod = relay.transform.InferType()(mod)
var_1402 = relay.var("var_1402", dtype = "int64", shape = (9, 11))#candidate|1402|(9, 11)|var|int64
var_1403 = relay.var("var_1403", dtype = "uint32", shape = (800,))#candidate|1403|(800,)|var|uint32
output = func_1401(var_1402,var_1403,)
func_1404 = relay.Function([var_1402,var_1403,], output)
mutated_mod['func_1404'] = func_1404
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1588 = relay.var("var_1588", dtype = "uint16", shape = (1, 7))#candidate|1588|(1, 7)|var|uint16
var_1589 = relay.var("var_1589", dtype = "uint16", shape = (7, 7))#candidate|1589|(7, 7)|var|uint16
bop_1590 = relay.greater_equal(var_1588.astype('bool'), var_1589.astype('bool')) # shape=(7, 7)
bop_1593 = relay.mod(var_1588.astype('float32'), var_1589.astype('float32')) # shape=(7, 7)
func_1401_call = mod.get_global_var('func_1401')
func_1404_call = mutated_mod.get_global_var('func_1404')
const_1598 = relay.const([[-10],[6],[10],[-10],[9],[-8],[-7],[4],[2],[5],[-2],[-1],[-1],[-4],[3],[-9],[-2],[9],[-1],[5],[-4],[-3],[-5],[-4],[4],[9],[10],[-8],[3],[3],[5],[8],[-9],[9],[3],[-9],[7],[9],[8],[9],[5],[-7],[9],[3],[-7],[-7],[10],[-8],[-10],[10],[1],[4],[1],[7],[7],[3],[-5],[-6],[-8],[-6],[5],[7],[-7],[-6],[-4],[-4],[-4],[-10],[-10],[-8],[7],[10],[4],[6],[-4],[8],[6],[-7],[-4],[1],[7],[-2],[4],[10],[3],[4],[7],[-9],[9],[-4],[-8],[2],[-10],[6],[-2],[-4],[4],[4],[10]], dtype = "int64")#candidate|1598|(99, 1)|const|int64
const_1599 = relay.const([[-4,-6],[10,-6],[-3,-5],[1,9],[6,-4],[6,9],[-8,-2],[8,-9],[-6,1],[4,-3],[8,-5],[8,2],[6,8],[2,7],[-2,1],[4,6],[4,6],[-7,-7],[6,7],[6,8],[7,-5],[-1,-1],[3,7],[-9,-7],[-10,-2],[-1,3],[-3,-3],[-2,-10],[-5,10],[-9,-3],[2,-10],[-10,-5],[-2,-7],[-8,-2],[3,-7],[-6,-1],[-3,-9],[1,-5],[-1,-3],[3,-3],[-7,-10],[3,3],[9,8],[-8,-6],[-2,-5],[8,-6],[-6,2],[-9,6],[7,8],[-2,-5],[-4,4],[-4,4],[7,10],[9,6],[7,7],[-5,1],[-3,-4],[-1,-3],[2,8],[8,-3],[-1,-2],[-1,6],[-6,-3],[7,5],[-4,-4],[5,-1],[-5,-6],[-10,9],[2,-10],[10,10],[-2,-9],[8,1],[-10,-6],[-10,-6],[5,-9],[-1,-6],[2,7],[-3,-7],[3,-6],[-1,-2],[9,8],[6,-6],[5,7],[6,10],[-9,-7],[5,9],[-4,-8],[1,-4],[9,-2],[7,5],[-3,-8],[4,-4],[-7,7],[-9,-10],[3,-5],[-10,-5],[-2,-2],[-4,3],[-9,-6],[-8,-8],[-5,9],[-3,-4],[6,1],[10,-5],[-10,3],[3,-1],[6,-7],[-10,-1],[2,-9],[-8,-1],[2,-2],[-9,-7],[1,-5],[6,7],[5,-8],[9,-10],[-9,5],[-1,4],[2,8],[-3,10],[-6,10],[-1,4],[3,1],[-1,-2],[-9,8],[5,7],[-7,9],[-1,4],[2,-9],[-3,8],[-3,7],[-9,4],[5,-10],[-1,-10],[-4,-8],[7,-2],[8,-1],[4,-4],[-8,1],[-9,5],[-7,8],[4,8],[-8,4],[-10,4],[10,6],[7,-1],[2,-10],[6,-9],[-2,3],[-7,-5],[7,10],[-8,5],[2,-1],[3,1],[3,8],[-9,-2],[-6,-9],[6,-5],[7,-10],[10,4],[7,8],[-7,-7],[4,-10],[6,6],[-7,-2],[2,7],[3,-6],[-3,4],[6,-1],[-2,-3],[-8,9],[-5,4],[1,6],[-3,3],[-10,-8],[9,-2],[3,8],[-4,2],[-6,-5],[-6,-4],[5,2],[-4,-4],[4,-7],[6,-8],[5,-6],[10,2],[-9,-6],[6,-3],[1,8],[10,2],[-8,7],[-10,-2],[-4,7],[-6,-8],[8,3],[-4,-6],[4,2],[8,8],[-3,-9],[-1,-2],[-1,6],[10,6],[8,1],[-6,6],[-2,3],[-2,-7],[7,10],[1,2],[-10,3],[4,7],[9,-3],[6,-4],[-10,2],[8,-9],[-7,5],[-10,2],[-7,2],[-4,-3],[-9,-5],[-10,5],[6,3],[7,10],[-10,4],[2,-2],[8,-2],[2,-5],[-4,6],[-1,-5],[8,-2],[-10,1],[-9,-10],[8,2],[3,-10],[9,-6],[6,-10],[-2,-6],[3,-7],[4,2],[2,4],[1,8],[10,-1],[-3,-4],[10,8],[9,4],[3,6],[1,1],[-10,-3],[-3,-9],[9,3],[-8,3],[-3,1],[4,-5],[-1,4],[-2,5],[-10,-5],[9,7],[5,-2],[-9,2],[-3,-3],[2,-3],[2,3],[2,6],[-6,4],[-6,-3],[-1,5],[-9,1],[1,8],[5,-4],[-1,-10],[7,-5],[-8,8],[2,-4],[-8,2],[-7,-9],[8,-2],[1,7],[-7,4],[-5,-6],[-10,7],[-6,6],[2,-1],[-10,7],[-8,-8],[-9,10],[3,9],[-5,-1],[9,-8],[8,9],[1,-9],[10,8],[9,6],[4,-3],[10,-7],[6,9],[-5,-10],[-10,-5],[1,-9],[-1,1],[-6,7],[3,-6],[-1,9],[10,-1],[-6,2],[-2,-4],[-2,-5],[3,-5],[3,6],[2,-10],[-5,-10],[-8,2],[4,9],[-10,10],[-7,4],[-1,8],[1,-9],[-5,7],[1,-5],[-8,2],[-10,7],[-5,1],[-10,-3],[-6,-6],[-4,-9],[5,9],[3,4],[4,3],[-8,-5],[5,-6],[-2,9],[-1,-2],[-9,3],[9,9],[-9,-3],[-10,6],[-7,2],[6,-9],[8,-4],[-3,5],[-1,4],[-2,4],[-6,4],[-3,4],[1,2],[-6,4],[10,-5],[6,3],[-10,-7],[-1,-4],[8,-5],[-8,1],[-6,7],[-5,-8],[-8,-8],[-4,-2],[-1,7],[-2,1],[10,-10],[-6,8],[-5,6],[5,-9],[-10,9],[-9,8],[-4,1],[9,5],[-7,-8],[8,-5],[-10,-7],[6,1],[-4,-9],[3,-4],[-6,2],[-9,3],[-4,9],[8,6],[-8,4],[-8,9],[5,-4],[5,7],[5,7],[10,9],[2,5],[3,-7],[-6,-4],[-5,-8],[-1,3],[2,-9],[1,8],[5,-5],[-5,-4],[-4,1],[3,-4],[-8,-1],[5,-5],[3,-2],[10,-6],[-7,-10],[1,-1],[5,3],[-7,8],[-9,6]], dtype = "uint32")#candidate|1599|(400, 2)|const|uint32
call_1597 = relay.TupleGetItem(func_1401_call(relay.reshape(const_1598.astype('int64'), [9, 11]), relay.reshape(const_1599.astype('uint32'), [800,]), ), 1)
call_1600 = relay.TupleGetItem(func_1404_call(relay.reshape(const_1598.astype('int64'), [9, 11]), relay.reshape(const_1599.astype('uint32'), [800,]), ), 1)
const_1603 = relay.const([[-4,-9],[8,-5],[7,-8],[6,-10],[-5,-6],[1,5],[1,8],[-10,5],[6,1],[3,2],[-7,-7],[-4,-3],[-5,9],[10,3],[-10,3],[4,4],[-3,5],[10,-8],[-10,9],[-1,4],[-1,9],[6,3],[6,6],[3,1],[-4,3],[-6,10],[-5,3],[-5,-1],[1,-10],[-3,5],[-8,-9],[-2,10],[-9,-1],[-4,-2],[-3,-5],[1,5],[-6,-8],[-7,-10],[5,-7],[-2,-2],[-1,-5],[-5,-4],[-1,7],[6,3],[-4,-3],[-4,-7],[4,6],[2,-6],[-3,6],[7,8],[-2,1],[6,2],[-5,-2],[-9,7],[-2,9],[3,5],[-6,7],[4,3],[-10,-5],[-8,-6],[-3,-1],[-5,8],[-1,1],[4,-3],[-8,6],[-6,7],[-5,2],[-9,-4],[7,7],[-4,-9],[-1,1],[8,-8],[2,8],[6,4],[10,-3],[-4,-3],[2,3],[4,3],[6,-3],[-7,9],[-8,9],[-4,5],[9,-3],[7,-6],[-8,7],[4,-9],[3,3],[4,9],[-10,-4],[-1,8],[-9,7],[-1,-6],[-3,-10],[2,-7],[1,3],[3,5],[-10,10],[10,-3],[3,6],[2,-2],[5,-2],[-9,-2],[-1,6],[-8,-9],[-2,2],[-1,2],[7,-9],[-9,4],[10,1],[-4,-2],[9,-9],[8,7],[-9,1],[-8,-5],[10,1],[-2,3],[10,-5],[-8,-1],[4,-6],[-10,-10],[6,2],[-6,-6],[-10,8],[-4,5],[-7,-10],[7,6],[3,-2],[8,5],[-5,7],[-3,-10],[2,4],[8,-9],[2,-4],[5,10],[3,-3],[8,-5],[5,4],[7,-9],[-1,4],[-4,6],[4,-5],[-1,7],[1,-5],[9,9],[5,10],[-9,-10],[3,10],[7,-3],[8,-10],[-7,3],[8,-4],[-1,3],[-8,6],[3,8],[10,-5],[-3,-7],[-8,-10],[-7,-2],[10,-4],[-9,-6],[9,6],[10,9],[-1,-6],[-5,-5],[-4,-10],[5,-6],[8,-7],[6,-5],[7,-6],[-5,-6],[1,1],[7,-9],[-1,1],[-8,3],[6,2],[5,-6],[9,7],[9,10],[-7,6],[-9,6],[1,-5],[4,-7],[-10,-2],[-6,-7],[-10,-1],[-10,-1],[-6,-3],[-1,2],[-6,-2],[-7,-9],[-8,3],[-7,3],[7,-10],[-8,3],[7,6],[5,-1],[5,9],[6,2],[-9,10],[-5,7],[8,10],[10,8],[1,7],[-2,-9],[6,10],[8,-6],[6,6],[10,7],[1,-1],[-1,8],[5,-10],[-6,5],[-8,3],[-8,4],[-6,7],[-4,-1],[-2,-7],[5,6],[5,-6],[-6,-4],[-5,-2],[-6,9],[7,-8],[-2,9],[5,-6],[-3,10],[-2,-8],[7,-5],[-1,-7],[3,8],[4,2],[-7,10],[9,2],[-10,-2],[9,1],[4,-10],[2,1],[-3,-5],[7,3],[-2,-7],[8,-3],[-2,-3],[4,8],[-9,-2],[-6,-8],[2,-8],[-3,-4],[-3,-6],[-10,1],[5,-4],[-7,9],[-1,-9],[-4,7],[-10,-6],[5,10],[10,-9],[6,-6],[-4,-1],[3,9],[1,-2],[-6,-10],[2,-1],[-5,-7],[10,9],[5,-8],[6,-10],[-7,4],[8,2],[2,10],[7,3],[4,-1],[5,2],[-5,-1],[-10,-1],[-9,-5],[-8,-8],[-9,8],[-10,10],[-10,-10],[-10,7],[5,-8],[8,-9],[4,-5],[-3,7],[-3,-7],[-6,1],[-9,5],[5,7],[-1,-2],[6,-7],[9,7],[6,3],[-6,4],[-9,-10],[-8,-8],[-6,-3],[-1,5],[8,10],[-5,-9],[-8,2],[-3,8],[1,-10],[4,2],[-4,6],[-1,3],[-2,-9],[9,-9],[8,-4],[8,-5],[-1,-10],[3,-4],[8,-10],[-10,10],[-7,10],[7,-4],[-1,-8],[-2,7],[-7,10],[-8,5],[-4,-5],[-2,-10],[-4,-7],[-10,-7],[-1,9],[7,-7],[-3,-4],[-1,7],[6,4],[8,8],[-2,4],[8,-4],[-6,2],[-9,8],[8,-2],[-3,-6],[2,-8],[4,4],[-1,-10],[2,7],[7,3],[-8,5],[-5,5],[-8,6],[3,-2],[-9,2],[7,-10],[6,-8],[7,-8],[-2,-1],[1,-4],[10,-1],[-3,-9],[9,4],[-5,-6],[-10,-5],[-6,3],[7,-7],[-1,9],[9,6],[3,4],[-9,10],[3,-5],[2,9],[3,10],[-6,-10],[7,2],[-1,10],[4,-8],[2,-6],[-5,7],[3,9],[-10,7],[5,-6],[-1,10],[7,-6],[6,5],[-1,5],[4,-5],[-2,6],[-5,-9],[-2,-5],[-5,5],[-7,-2],[-2,-4],[-10,10],[3,-2],[2,-6],[-2,-7],[3,-7],[9,10],[-2,10],[1,-8],[-3,6],[6,10],[7,4],[2,-4],[-5,1],[5,-1],[-6,-3],[-7,8]], dtype = "uint32")#candidate|1603|(400, 2)|const|uint32
bop_1604 = relay.left_shift(const_1599.astype('uint32'), relay.reshape(const_1603.astype('uint32'), relay.shape_of(const_1599))) # shape=(400, 2)
func_215_call = mod.get_global_var('func_215')
func_219_call = mutated_mod.get_global_var('func_219')
const_1618 = relay.const([True,False,False,True,False,False,False,True,False,True,False,False,False,False,False,True,False,False,True,True,True,False,True,False,True,False,True,True,True,True,True,False,True,False,True,True,False,True,False,False,False,True,True,False,True,True,True,True,True,True,False,False,True,False,True,False,False,False,False,True,True,True,True,False,True,False,True,False,True,True,False,False,False,True,False,True,False,False,True,False,True,True,False,False,True,True,True,False,True,True,True,False,False,True,False,False,True,True,False,True,True,False,True,True,False,True,True,False,True,False,False,False,True,True,True,True,False,False,True,False,True,False,False,True,True,True,True,True,False,True,False,True,False,True,True,False,False,False,False,False,False,True,False,True,False,True,False,False,True,True,False,False,False,True,False,True,False,True,False,False,True,False,False,True,True,True,True,False,True,True,False,True,False,False,False,False,False,True,True,True,False,True,False,False,False,False,False,False,True,False,True,False,True,True,True,False,False,False,False,True,False,False,False,False,True,False,True,True,True,False,False,False,True,False,False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,True,True,False,True,True,True,False,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,True,True,True,True,False,True,False,False,False,True,False,False,False,True,True,True,True,True,False,True,True,False,False,True,True,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False,True,False,False,True,False,False,False,True,True,True,False,True,True,True,False,False,True,False,False,False,True,True,True,True,False,True,True,False,True,True,False,False,False,False,True,False,False,False,True,False,True,False,True,False,True,True,True,False,True,True,True,False,True,False,True,True,True,False,True,False,False,True,False,False,True,True,False,False,True,True,False,False,True,True,True,False,False,False,True,True,True,False,False,False,False,True,True,True,False,False,True,False,True,True,False,True,False,False,True,False,True,True,False,True,False,True,True,False,True,False,True,True,False,True,True,False,True,True,True,True,True,False,True,True,True,True,False,False,True,True,False,False,False,False,True,False,False,True,False,True,True,True,False,False,True,False,False,False,True,False,True,False,False,False,True,False,True,True,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,True,False,True,True,True,True,False,False,True,False,True,True,True,True,True,True,True,True,True,False,False,True,False,False,False,True,False,True,True,True,True,False,True,False,True,True,True,True,True,True,True,False,False,False,True,False,True,True,True,False,True,True,False,True,False,False,False,False,True,True,True,True,True,True,True,False,False,True,True,True,False,False,True,False,False,True,False,True,False,False,False,True,True,False,False,True,False,True,False,False,False,True,True,False,False,False,False,False,True,False,True,True,False,True,False,True,False,False,False,False,False,False,True,False,False,True,True,True,False,True,True,True,True,False,True,True,False,True,False,False,True,True,True,False,False,True,True,False,False,True,False,False,False,True,False,False,True,False,True,True,True,False,False,False,False,True,False,True,True,False,True,False,True,False,False,False,False,True,False,False,True,True,False,True,False,True,True,True,True,True,True,False,True,True,True,False,True,True,True,False,True,False,False,False,False,True,False,False,False,True,False,True,False,True,False,False,False,False,False,True,True,True,True,False,False,False,False,True,True,True,True,False,True,False,True,False,True,True,False,True,False,True,True,False,False,True,True,False,True,True,True,False,False,False,True,False,True,True,False,False,True,True,True,False,True,False,False,False,True,True,False,False,False,True,False,False,False,True,False,False,False,True,False,False,True,True,True,False,True,True,True,False,False,False,False,False,False,True,False,True,False,False,True,False,True,False,False,False,True,False,False,False,True,False,False,False,False,True,True,True,False,False,True,True,False,True,True,False,False,True,True,False,False,True,False,True,True,False,True,False,False,True,False,True,False,True,False,False,False,True,True,True,True,False,False,True,True,True,True,False,False,False,False,False,True,True,False,False,True,True,True,False,True,True,True,True,False,True,True,True,False,False,False,True,False,False,False,True,True,True,True,True,False,False,False,True,True,False,False,False,True,True,False,False,False,False,True,True,True,False,False,False,False,False,True,False,False,False,False,True,True,True,False,True,True,False,True,False,False,True,True,False,False,True,False,False,True,False,True,True,False,True,True,True,True,False,True,True,False,True,False,False,True,True,True,False,True,False,True,False,True,False,False,True,True,False,False,True,False,True,False,True,True,True,False,False,True,True,False,False,True,False,False,False,True,True,False,True,False,True,True,True,True,True,True,True,True,True,False,False,False,False,True,True,True,True,False,False,False,False,False,True,False,False,True,False,False,False,True,False,True,False,True,True,True,False,False,False,True,False,False,False,False,True,True,True,False,True,True,True,True,False,False,False,False,False,False,False,False,True,False,True,True,False,False,False,False,False,False,False,False,True,False,False,False,True,True,True,True,True,False,False,True,False,False,True,False,False,False,False,True,False,False,False,False,False,False,True,True,True,True,True,False,True,False,False,False,True,True,True,False,True,False,True,False,False,True,False,True,True,True,False,True,True,True,False,True,True,True,False,False,False,False,False,True,False,True,True,True,False,True,False,False,True,True,False,True,False,False,False,True,True,True,True,False,False,True,False,True,False,False,False,False,False,False,True,False,True,False,False,False,True,False,False,False,True,False,False,False,False,False,True,False,True,False,False,True,True,False,True,True,True,True,True,True,True,False,False,True,True,False,True,False,False,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,False,True,False,True,False,False,True,False,False,True,True,True,True,False,False,True,True,True,False,False,False,False,True,True,True,True,False,False,True,True,True,True,False,False,True,True,False,False,True,False,True,False,False,False,False,False,True,True,True,True,True,True,True,False,False,False,True,True,True,False,False,False,False,False,False,True,True,False,True,False,True,False,False,True,False,False,True,True,True,False,False,False,True,False,True,True,False,False,False,True,True,False,False,True,True,False,False,False,True,True,True,False,True,True,True,True,False,False,False,True,False,False,False,True,False,True,False,True,True,True,False,False,True,False,True,False,True,True,True,False,False,False,False,True,True,True,False,True,True,False,True,True,True,True,False,True,False,False,False,True,True,True,False,True,True], dtype = "bool")#candidate|1618|(1320,)|const|bool
call_1617 = relay.TupleGetItem(func_215_call(relay.reshape(const_1618.astype('bool'), [10, 11, 12]), relay.reshape(const_1618.astype('bool'), [10, 11, 12]), ), 0)
call_1619 = relay.TupleGetItem(func_219_call(relay.reshape(const_1618.astype('bool'), [10, 11, 12]), relay.reshape(const_1618.astype('bool'), [10, 11, 12]), ), 0)
func_430_call = mod.get_global_var('func_430')
func_433_call = mutated_mod.get_global_var('func_433')
const_1622 = relay.const([10,-10,2,5,-6,-2,-3,-1,-2,-9,-5,1,-9,9,-8,10,10,6,-4,6,1,-1,9,-9,6,1,4,-1,3,-1,-8,10,10,2,7,4,-5,2,-6,4,-10,-1,7,1,2,10,2,-10,3,6,9,8,6,8,5,-2,-9,3,-7,-5,-4,-10,4,-9,-3,9,-5,-4,8,-3,-5,10,-9,9,5,8,-10,6], dtype = "uint16")#candidate|1622|(78,)|const|uint16
call_1621 = func_430_call(relay.reshape(const_1622.astype('uint16'), [13, 6]))
call_1623 = func_430_call(relay.reshape(const_1622.astype('uint16'), [13, 6]))
func_899_call = mod.get_global_var('func_899')
func_903_call = mutated_mod.get_global_var('func_903')
const_1625 = relay.const([6.215043,1.223447,2.954028,-1.512402], dtype = "float32")#candidate|1625|(4,)|const|float32
const_1626 = relay.const([-5.346292,4.200031,8.380196,5.789170,-1.375114,-5.525816,9.964649,-1.766294,7.537153,-7.397353,8.957201,-6.647032,-3.222529,-3.700680,-8.941598,-7.070345,0.978826,6.551417,-3.266565,3.448843,-1.413365,-8.650554,3.177374,-9.518961,-5.818372,9.402406,7.487250,4.112354,1.503903,1.691928,-9.997972,-1.070622,9.549574,-4.716048,-8.443908,-2.803446,9.354100,5.965788,8.290752,5.144056,-0.193975,1.447454,-8.661593,5.003819,9.996436,2.650212,-8.858907,9.136101,8.513978,-1.275765,0.890292,-8.250767,4.779644,8.824479,-4.251375,6.854764,-1.688605,-6.221136,5.185416,-7.519908,-3.229551,-7.312440,5.233339,-3.890822,8.254195,2.265214,5.882891,9.270368,3.213078,4.450706,-6.549539,-6.640512,-6.836285,2.163709,-9.927202,-8.298838,-0.361040,-4.349610,-5.913985,-0.993932,7.609394,7.838273,-4.346277,5.193572,0.601514,3.760155,7.880266,-0.855398,-3.301205,-3.161069,7.669403,4.734779,3.755342,-2.499910,-0.124032,-8.819184,-5.158269,-8.086473,-1.864404,-2.013471,-5.527535,7.545802,8.235467,3.930238,-4.008940,-4.910540,-6.523712,2.154575,1.545114,-8.238892,-1.600428,9.287448,-6.939977,6.230383,-9.058395,-1.187407,8.039098,-3.709605,-4.628497,-6.520905,7.523410,-7.999311,-1.988873,-0.234615,2.914627,5.832402,4.655008,-8.101462,-1.769661,-0.584952,0.453523,-4.121702,7.697215,8.335869,-0.751560,-2.827180,7.806798,-4.368500,-6.743616,8.546297,1.808682,2.047623,-3.990241,-9.513203,-6.045100,8.704038,4.613352,-7.724414,-6.432943,-7.644553,-9.426094,2.983510,-1.069331,0.634782,-2.898104,-9.733199,-6.131852,-9.802793,-3.508109,2.665778,4.308481,-9.401062,-6.978187,6.061168,0.322134,-8.487406,-7.221220,0.695152,7.122498,1.055710,5.366276,0.655368,-5.649224,-7.787928,-7.312297,8.335060,-0.247861,-6.108271,-7.761236,-3.818645,-2.584545,6.750663,-3.073342,6.676860,-7.456824,4.220652,-6.832059,1.432286,-7.662626,5.857866,3.012316,-5.444959,6.786685,3.058021,7.886579,5.717977,-5.703477,-1.190530,0.954401,-5.105590,6.354204,6.820077,-8.066340,-0.294362,6.400784,2.667722,-1.900400,-4.743094,-9.653506,1.214704,4.687515,8.533102,-0.201597,-0.258900,1.893033,3.168621,-7.034688,1.674946,7.758904,3.656733,3.933582,7.080674,7.207695,4.710112,5.580278,4.681528,7.949906,-6.097820,-4.724908,7.369556,3.661176,-5.771130,-3.170168,-7.224228,2.263732,-0.729589,-0.309151,-6.989774,2.765976,-6.574202], dtype = "float32")#candidate|1626|(240,)|const|float32
call_1624 = relay.TupleGetItem(func_899_call(relay.reshape(const_1625.astype('float32'), [4,]), relay.reshape(call_1617.astype('bool'), [1, 1320]), relay.reshape(const_1626.astype('float32'), [240,]), ), 1)
call_1627 = relay.TupleGetItem(func_903_call(relay.reshape(const_1625.astype('float32'), [4,]), relay.reshape(call_1617.astype('bool'), [1, 1320]), relay.reshape(const_1626.astype('float32'), [240,]), ), 1)
output = relay.Tuple([bop_1590,bop_1593,call_1597,const_1598,bop_1604,call_1617,const_1618,call_1621,const_1622,call_1624,const_1625,const_1626,])
output2 = relay.Tuple([bop_1590,bop_1593,call_1600,const_1598,bop_1604,call_1619,const_1618,call_1623,const_1622,call_1627,const_1625,const_1626,])
func_1644 = relay.Function([var_1588,var_1589,], output)
mod['func_1644'] = func_1644
mod = relay.transform.InferType()(mod)
var_1645 = relay.var("var_1645", dtype = "uint16", shape = (1, 7))#candidate|1645|(1, 7)|var|uint16
var_1646 = relay.var("var_1646", dtype = "uint16", shape = (7, 7))#candidate|1646|(7, 7)|var|uint16
output = func_1644(var_1645,var_1646,)
func_1647 = relay.Function([var_1645,var_1646,], output)
mutated_mod['func_1647'] = func_1647
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1735 = relay.const([[-8,7,1,5,-5,9,9,6,-6,3,6,9,2,-7,-1,4],[-1,9,6,2,-2,-9,-7,-4,5,-2,3,-8,-9,9,-4,-1],[6,5,-5,-3,-9,8,5,-1,-9,-1,4,-9,-4,5,3,-1]], dtype = "int8")#candidate|1735|(3, 16)|const|int8
var_1736 = relay.var("var_1736", dtype = "int8", shape = (3, 16))#candidate|1736|(3, 16)|var|int8
bop_1737 = relay.greater(const_1735.astype('bool'), relay.reshape(var_1736.astype('bool'), relay.shape_of(const_1735))) # shape=(3, 16)
output = relay.Tuple([bop_1737,])
output2 = relay.Tuple([bop_1737,])
func_1740 = relay.Function([var_1736,], output)
mod['func_1740'] = func_1740
mod = relay.transform.InferType()(mod)
var_1741 = relay.var("var_1741", dtype = "int8", shape = (3, 16))#candidate|1741|(3, 16)|var|int8
output = func_1740(var_1741)
func_1742 = relay.Function([var_1741], output)
mutated_mod['func_1742'] = func_1742
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1807 = relay.var("var_1807", dtype = "float32", shape = (7, 8, 16))#candidate|1807|(7, 8, 16)|var|float32
uop_1808 = relay.sin(var_1807.astype('float32')) # shape=(7, 8, 16)
uop_1810 = relay.sqrt(var_1807.astype('float32')) # shape=(7, 8, 16)
uop_1812 = relay.acos(uop_1810.astype('float64')) # shape=(7, 8, 16)
bop_1814 = relay.bitwise_xor(uop_1808.astype('int32'), relay.reshape(uop_1810.astype('int32'), relay.shape_of(uop_1808))) # shape=(7, 8, 16)
bop_1823 = relay.logical_xor(uop_1808.astype('int16'), relay.reshape(bop_1814.astype('int16'), relay.shape_of(uop_1808))) # shape=(7, 8, 16)
uop_1826 = relay.asinh(var_1807.astype('float32')) # shape=(7, 8, 16)
func_1740_call = mod.get_global_var('func_1740')
func_1742_call = mutated_mod.get_global_var('func_1742')
const_1831 = relay.const([-4,4,8,-8,-8,1,9,10,-7,6,-5,-6,1,-4,-10,3,-3,9,2,-2,-8,-6,9,4,8,9,-6,-6,9,-8,-8,-4,10,2,-1,-2,-7,7,-6,-4,-2,2,8,-3,-7,-9,-3,-5], dtype = "int8")#candidate|1831|(48,)|const|int8
call_1830 = relay.TupleGetItem(func_1740_call(relay.reshape(const_1831.astype('int8'), [3, 16])), 0)
call_1832 = relay.TupleGetItem(func_1742_call(relay.reshape(const_1831.astype('int8'), [3, 16])), 0)
output = relay.Tuple([uop_1812,bop_1823,uop_1826,call_1830,const_1831,])
output2 = relay.Tuple([uop_1812,bop_1823,uop_1826,call_1832,const_1831,])
func_1835 = relay.Function([var_1807,], output)
mod['func_1835'] = func_1835
mod = relay.transform.InferType()(mod)
mutated_mod['func_1835'] = func_1835
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1836 = relay.var("var_1836", dtype = "float32", shape = (7, 8, 16))#candidate|1836|(7, 8, 16)|var|float32
func_1835_call = mutated_mod.get_global_var('func_1835')
call_1837 = func_1835_call(var_1836)
output = call_1837
func_1838 = relay.Function([var_1836], output)
mutated_mod['func_1838'] = func_1838
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1851 = relay.var("var_1851", dtype = "int32", shape = (5, 1, 11))#candidate|1851|(5, 1, 11)|var|int32
var_1852 = relay.var("var_1852", dtype = "int32", shape = (5, 10, 11))#candidate|1852|(5, 10, 11)|var|int32
bop_1853 = relay.subtract(var_1851.astype('int32'), var_1852.astype('int32')) # shape=(5, 10, 11)
bop_1857 = relay.greater_equal(var_1851.astype('bool'), bop_1853.astype('bool')) # shape=(5, 10, 11)
output = relay.Tuple([bop_1857,])
output2 = relay.Tuple([bop_1857,])
func_1877 = relay.Function([var_1851,var_1852,], output)
mod['func_1877'] = func_1877
mod = relay.transform.InferType()(mod)
mutated_mod['func_1877'] = func_1877
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1877_call = mutated_mod.get_global_var('func_1877')
var_1879 = relay.var("var_1879", dtype = "int32", shape = (5, 1, 11))#candidate|1879|(5, 1, 11)|var|int32
var_1880 = relay.var("var_1880", dtype = "int32", shape = (5, 10, 11))#candidate|1880|(5, 10, 11)|var|int32
call_1878 = func_1877_call(var_1879,var_1880,)
output = call_1878
func_1881 = relay.Function([var_1879,var_1880,], output)
mutated_mod['func_1881'] = func_1881
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1899 = relay.var("var_1899", dtype = "float32", shape = (1, 9, 9))#candidate|1899|(1, 9, 9)|var|float32
uop_1900 = relay.sinh(var_1899.astype('float32')) # shape=(1, 9, 9)
bop_1904 = relay.add(var_1899.astype('int64'), relay.reshape(uop_1900.astype('int64'), relay.shape_of(var_1899))) # shape=(1, 9, 9)
func_602_call = mod.get_global_var('func_602')
func_606_call = mutated_mod.get_global_var('func_606')
var_1908 = relay.var("var_1908", dtype = "int8", shape = (3, 9))#candidate|1908|(3, 9)|var|int8
call_1907 = relay.TupleGetItem(func_602_call(relay.reshape(var_1908.astype('int8'), [3, 9]), relay.reshape(var_1908.astype('int8'), [3, 9]), ), 0)
call_1909 = relay.TupleGetItem(func_606_call(relay.reshape(var_1908.astype('int8'), [3, 9]), relay.reshape(var_1908.astype('int8'), [3, 9]), ), 0)
bop_1919 = relay.minimum(uop_1900.astype('int16'), relay.reshape(bop_1904.astype('int16'), relay.shape_of(uop_1900))) # shape=(1, 9, 9)
var_1923 = relay.var("var_1923", dtype = "float32", shape = (10, 9, 9))#candidate|1923|(10, 9, 9)|var|float32
bop_1924 = relay.maximum(var_1899.astype('int32'), var_1923.astype('int32')) # shape=(10, 9, 9)
bop_1931 = relay.less(uop_1900.astype('bool'), relay.reshape(var_1899.astype('bool'), relay.shape_of(uop_1900))) # shape=(1, 9, 9)
var_1936 = relay.var("var_1936", dtype = "float32", shape = (14, 9, 9))#candidate|1936|(14, 9, 9)|var|float32
bop_1937 = relay.mod(uop_1900.astype('float32'), var_1936.astype('float32')) # shape=(14, 9, 9)
output = relay.Tuple([call_1907,var_1908,bop_1919,bop_1924,bop_1931,bop_1937,])
output2 = relay.Tuple([call_1909,var_1908,bop_1919,bop_1924,bop_1931,bop_1937,])
F = relay.Function([var_1899,var_1908,var_1923,var_1936,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1899,var_1908,var_1923,var_1936,], output2)
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
input_1899= np.array([[[-4.544065,9.856496,-7.695308,-7.880887,1.441820,-4.652680,-9.645736,7.366749,-9.575327],[-8.805610,9.964667,-6.267512,-7.289329,6.281416,-0.220166,-3.297144,-6.508258,5.517760],[1.680794,5.097634,5.298235,-6.223358,-1.632428,-3.323674,-6.304911,-6.647558,9.693788],[0.515869,9.806259,1.295470,8.452937,-1.445688,-4.224544,2.148207,2.439779,-8.933867],[-6.567978,6.968121,-5.189617,8.558130,-4.417906,-5.452758,8.078298,2.952033,-2.690798],[-5.841813,-0.506773,0.589855,-2.672158,4.197412,9.302435,-5.514244,-1.537487,-8.026185],[-9.878412,5.642718,8.995184,-4.192008,4.537136,-3.233599,9.745398,6.681815,5.822749],[-8.312092,6.064653,-5.163721,3.289226,4.324919,-1.549089,4.263026,6.953720,7.983844],[-8.558525,5.931213,8.677405,-6.214950,3.811414,7.781268,-6.009621,-3.613272,2.147783]]], dtype='float32')
module1.set_input('var_1899', input_1899)
input_1908= np.array([[-7,-6,-5,-3,-8,-10,8,3,4],[-4,-5,-9,-3,6,4,-5,-8,-3],[1,-4,10,-9,10,10,-1,-6,-1]], dtype='int8')
module1.set_input('var_1908', input_1908)
input_1923= np.array([[[3.139807,8.284115,-4.402584,-1.124337,9.352853,4.273777,-2.331146,4.823340,-2.896180],[-6.990803,-7.536670,-8.137816,2.483850,-8.473943,0.944658,-4.293752,1.708925,-9.779701],[3.913085,-2.349588,1.073990,-2.039546,-2.632671,6.711748,-3.924638,5.919437,-2.536485],[1.993825,-3.709457,-0.855368,-1.520779,-4.252050,8.561872,3.940811,0.398377,2.859024],[1.035429,-9.935434,-8.653853,-6.928195,4.162420,2.951877,-8.583787,-9.260654,7.551889],[-7.974698,5.153057,0.237134,4.381339,2.468541,-4.221764,-9.903603,-8.056625,-4.517238],[-1.368138,4.321946,-5.031927,1.249393,5.134906,5.397216,-2.339255,7.971980,-9.931894],[-6.421973,0.450089,-9.968054,3.952829,-8.380673,1.029872,-9.020659,8.056928,-3.040983],[-2.879644,-0.526141,0.922808,-6.406766,7.046326,2.585704,6.064383,-2.588551,-3.865977]],[[-1.008999,9.006936,3.696649,0.964278,-6.482077,-7.308095,-1.008887,-1.297551,-8.728179],[-0.885200,3.542671,-7.254823,9.916406,9.347647,8.273431,-1.972649,9.590074,-0.957153],[7.187789,-2.644167,-7.129513,-2.274421,6.954604,-4.099279,-7.109676,9.621825,-6.076570],[-6.658246,0.767696,9.160686,0.753414,-9.914909,9.226041,-0.897517,2.221452,-7.503543],[8.434313,-0.608340,6.645703,-5.553845,-5.817483,-5.320356,-6.546424,-6.870974,-3.292539],[-9.663533,-6.465369,-8.363022,7.729790,-1.815535,7.228507,-4.210782,9.600630,4.963644],[7.552925,-4.512058,1.014399,3.614689,8.754616,2.066500,0.907107,-5.734167,8.743831],[-0.219304,9.466888,9.956379,5.819083,5.516828,-9.434332,-0.533255,-8.420117,0.101924],[9.317547,1.110992,8.574602,-9.290793,-7.920972,-7.946018,8.202827,-8.929299,6.969571]],[[9.469868,-6.931415,7.692666,8.223298,-2.832527,-6.861454,-8.926793,-9.610268,-6.588665],[-0.155639,4.762833,2.010484,-1.453274,-4.598569,8.356091,-7.014730,-4.566816,5.532556],[-5.328018,1.771437,7.986279,-6.733289,0.519489,3.935149,-8.335728,-8.285147,7.293829],[0.303773,-8.442752,-1.607790,4.405497,-5.050283,6.820593,7.707872,3.537965,7.798970],[-1.938337,-2.090231,-7.368865,-5.823486,9.999958,-8.372184,-1.297043,3.242504,8.840735],[5.184856,-1.839591,8.980047,2.665856,4.769132,0.407653,5.368380,5.708317,-6.317115],[-1.271841,-2.272171,3.951605,-6.928183,8.283099,2.385973,6.398852,9.052647,-8.539777],[-5.711077,-1.967093,-5.129156,-2.697787,7.065636,3.259566,6.676330,-5.870536,-0.069135],[6.457270,-4.008360,-9.157242,7.188630,-1.125168,3.220812,6.027248,-4.443948,-3.987450]],[[-2.097617,4.057584,-7.896727,8.879167,-6.041568,1.252568,8.791903,2.630305,9.099908],[7.184076,6.521090,-4.485021,6.671087,-3.646899,4.237905,-1.057932,5.555177,7.580970],[8.936580,-0.754290,9.591380,-2.705458,-9.318904,-9.670988,4.362664,-2.052638,7.315977],[6.161348,4.115111,-3.438262,-6.082672,-5.022587,-4.715762,4.576799,-4.412215,1.346390],[-4.411929,-9.628272,0.583449,-7.448622,7.556473,-8.189886,2.050135,6.967264,9.088574],[6.969964,-7.362310,-1.497931,-0.391483,-8.614880,6.612510,-0.751258,8.339720,-3.854278],[1.972394,2.524984,5.855964,6.332277,-8.260120,1.694542,-1.231312,3.014140,-7.193785],[-1.284971,-1.936557,0.721908,-3.530921,4.594826,1.168858,7.389223,-7.677500,5.960220],[-1.858397,8.253826,2.315273,-5.637376,0.382823,-6.046689,3.905688,-4.870223,-3.899836]],[[0.434880,9.747537,-8.351723,-7.658611,9.665372,4.164088,9.203436,5.637093,-5.790719],[8.290408,8.917262,-5.655980,2.831375,-9.985584,6.198187,0.705681,1.188258,6.682268],[8.241756,5.245663,8.254790,1.913340,-6.149773,9.396538,4.196681,-5.774174,8.434160],[9.654376,-3.833630,8.306014,-7.647600,4.061876,2.933938,0.494878,-8.340814,-1.307249],[5.233090,-9.648359,-2.787942,-3.507591,1.886203,-2.189956,-9.215359,8.763291,7.069168],[-6.428200,8.843030,-2.173052,7.698258,-8.426031,-5.930105,0.387595,-4.691240,-7.670468],[9.318890,1.501752,6.844055,5.776756,5.020176,-8.380313,3.251107,-9.057582,-3.184100],[-1.497497,-6.517704,-9.174827,0.932350,2.896337,4.620867,-0.354817,-1.640185,0.102190],[-4.703143,0.868775,6.486860,-0.494318,0.144898,-3.544326,8.774029,4.965791,8.756588]],[[5.025902,2.640967,-6.283719,-3.669604,9.778036,-9.064911,-5.332859,2.792754,-5.515858],[5.713297,-4.545596,9.649380,0.460535,6.902799,2.145538,-6.531888,6.003778,4.196064],[-3.871477,-6.987149,-5.974402,1.265291,4.710092,2.724582,3.920869,1.112213,3.655375],[-8.280160,6.907100,1.708065,-8.743036,1.100672,3.880517,-9.793682,-5.827554,1.605789],[3.557381,3.419883,5.868532,2.935045,-6.986389,-5.767629,0.933682,-2.468979,-2.378587],[5.180822,9.108590,-6.861479,3.785907,-2.970821,-7.511963,1.970210,-7.314670,-1.811135],[5.015101,0.060865,-5.797184,7.386449,-3.704888,9.938575,0.723696,-0.820965,3.259049],[0.583962,0.586976,6.854345,4.422184,3.965743,4.597653,3.218892,1.323151,-4.149940],[5.955333,-8.593710,6.507736,-5.652738,-1.013619,7.790851,-1.005182,7.030575,8.184989]],[[6.861363,3.287805,-3.905716,-2.754695,7.026974,-8.713352,-8.512093,9.808701,-3.528712],[-1.138922,8.877349,7.686522,-8.200203,5.162824,7.731698,-5.032505,0.995028,3.116403],[1.456717,9.518478,-8.004389,-2.385573,4.314903,8.909534,8.645256,6.470601,-7.585515],[9.007709,-1.452737,-5.467185,7.789074,-5.536616,-0.978646,9.216716,8.117687,-4.890830],[-1.829169,-5.442717,-9.690080,3.974559,5.070603,-5.150079,7.240276,4.080923,-9.604336],[0.824494,-8.288948,-4.208973,-9.113561,-4.510704,6.752906,0.540489,-3.424318,6.423257],[5.086346,-9.736688,6.044012,-4.818249,-1.995013,-6.441110,-2.203732,1.554201,8.732935],[1.153588,8.545817,-4.106572,7.256000,-3.965805,6.713060,-7.697907,0.534114,-2.203937],[-8.499373,7.121827,-5.658497,1.196361,-5.377569,5.490172,5.150199,5.072010,4.243712]],[[3.340102,7.657288,-5.048406,3.403915,1.047578,3.153503,-7.960770,-2.964677,-0.721079],[3.403795,7.824294,-8.552840,-4.869443,-6.785542,-1.988567,3.003225,-5.598908,-9.354628],[1.850988,6.923324,-4.415337,-8.537423,-6.978641,-2.860444,3.379995,6.295713,5.797533],[-5.824642,3.890830,7.708506,-7.059647,6.750771,9.972707,3.491822,9.256297,6.102341],[6.120197,-6.444846,8.574844,-1.301658,4.302123,7.908064,-7.697585,-0.431053,-9.517535],[4.083910,-5.810980,2.063250,-7.982023,-7.649193,-4.343921,5.742760,-6.602355,8.850864],[6.422267,-8.862038,4.253873,4.304373,4.517873,9.281499,-6.418985,-0.956238,0.573446],[-7.660057,-8.443865,-9.668506,3.452460,-1.568126,6.953498,5.626948,-8.031566,0.356719],[9.097426,1.782586,-8.331041,4.830759,5.025391,1.986011,5.153388,4.926621,5.604112]],[[-2.219526,7.778080,-3.785535,7.395041,6.550830,0.526601,4.975268,-5.732536,-9.563928],[-3.010556,7.621430,6.316006,-6.720356,-2.344435,5.049456,4.735038,-6.820649,-3.762967],[-5.252939,-4.199937,7.289073,-6.085167,2.323991,2.729176,-6.109262,4.396389,-0.286553],[-8.574047,8.431593,-1.695044,-0.274698,3.692075,-0.749458,5.089483,3.189240,7.353743],[-1.691971,1.265244,-2.654449,9.257138,-2.187105,-5.559280,1.496192,-8.101944,8.913237],[5.949886,-6.864011,0.123879,2.063409,7.201628,-3.160552,9.433224,0.067700,-2.320134],[-8.038559,-3.137227,-5.541527,6.334234,6.426900,1.862508,-0.181507,4.828024,5.236529],[2.037042,0.662127,4.636047,9.811934,-4.631969,2.941010,-6.683087,-3.536258,2.727489],[8.562529,9.391402,-8.849956,-0.606212,7.875633,4.541915,-3.127493,5.822803,-6.852567]],[[1.950474,-2.623432,-1.655154,8.413267,5.630784,2.780385,-4.786447,4.300449,3.506588],[-8.894629,-7.210806,0.278990,-2.388187,-9.296132,-1.520629,-8.721729,6.669356,6.727368],[0.492939,-4.887813,3.560523,4.276898,7.414136,6.612567,-3.677555,-8.057407,6.251311],[-2.086854,5.689011,-8.163995,5.448715,8.837495,0.846920,-4.517683,6.531891,0.842937],[-9.099148,1.002954,7.132220,8.843913,-9.921189,-9.207249,5.451474,-8.072054,0.340570],[-5.764703,7.744024,7.878613,9.911809,4.134320,8.272702,5.613161,6.161449,2.353716],[4.789368,-7.789650,-0.058990,4.437038,-0.468946,-0.010934,-1.270981,-5.649774,-4.673726],[-5.702975,-4.476523,0.847282,-6.889659,7.288729,0.688553,-7.333824,4.789353,-1.608863],[3.896111,-8.829990,1.721645,-7.883313,9.207941,-9.600786,-5.155682,4.729776,-4.048729]]], dtype='float32')
module1.set_input('var_1923', input_1923)
input_1936= np.array([[[-1.477084,0.076703,3.592896,-7.625324,0.110286,-1.934073,7.833758,-2.358694,9.919096],[-7.707850,8.695637,1.019262,-1.389099,6.044780,-2.886340,-3.635229,-7.986783,-3.017038],[-6.048115,-4.713719,3.254182,4.501970,-6.853586,-6.187316,-7.205182,8.268086,-5.297099],[3.537133,9.508970,6.046201,-3.011127,-8.273262,3.638702,-4.462142,-2.695746,-6.567366],[-0.355089,-2.431723,6.648893,-4.342082,-9.864923,2.509795,-9.730750,5.475487,9.572288],[0.452014,-3.635058,4.034457,8.646626,9.591249,-5.796023,9.610082,6.594228,7.831359],[-3.881711,-4.436964,-8.069122,7.368784,-6.359135,-7.390833,8.989804,-6.843927,3.613801],[-0.830489,0.867414,-8.796538,0.365950,-5.444588,-1.072048,4.673404,-9.234771,5.447498],[2.633638,6.771843,8.851941,-7.394001,-4.054879,-7.055480,-2.659020,-4.449164,-9.375187]],[[7.916096,-6.156736,-0.346318,3.295723,9.090880,-2.780477,-9.733450,1.415843,5.094573],[1.564355,-7.523498,-5.935131,-5.025943,2.486015,3.339389,8.749950,6.148247,-8.528407],[-2.193255,2.827391,1.021063,-1.363992,7.373308,-8.512206,-1.857575,-6.107242,3.501923],[8.859432,9.667940,-3.153709,-0.914485,7.114379,7.934737,-3.853145,-5.959842,4.265874],[3.151075,9.549031,8.690872,2.884692,-1.393094,-3.871123,6.215300,2.833292,-8.571824],[8.709557,1.737705,-1.935265,-3.792679,4.128042,-5.898537,9.287062,-7.108341,2.904199],[-7.724233,-5.437626,-8.117513,-0.519317,-6.237182,5.558723,1.871692,-0.441172,7.399090],[8.545918,0.831189,-3.613923,4.544835,6.042827,5.317503,1.771758,6.824131,-1.716802],[9.033484,0.552826,-9.109354,-0.691791,7.422644,-4.109117,-8.041076,0.631617,-5.566179]],[[3.420226,3.572896,-9.801565,6.290166,-9.157745,4.753118,6.939898,3.965043,-6.031474],[9.258387,2.921373,4.909752,3.472474,2.409212,8.992853,8.018103,0.734930,6.464174],[9.035396,0.488769,-4.120373,-4.119219,8.080664,-2.475231,7.810428,1.025873,4.285174],[-8.430163,-5.330302,9.455930,-5.772756,-6.509678,3.169062,8.804591,6.524799,-7.185756],[3.170554,9.847903,-7.036041,-8.515968,4.076084,-2.989732,2.312964,8.860598,-0.038286],[1.848025,2.860631,-5.451074,-4.882697,3.100652,-8.814110,2.609676,-9.789401,6.439361],[-7.360375,-7.832588,3.845735,-2.494627,-1.951066,-6.911480,-4.053566,9.507825,9.324585],[-4.492516,5.645942,5.980116,7.333100,3.060059,2.752622,-0.149807,-8.171452,3.064084],[0.826999,-6.706833,-3.839941,-7.591341,6.676724,-8.946016,3.135530,-7.865722,9.527303]],[[7.120732,-5.967687,-9.006318,4.979718,3.693993,-6.142730,5.824557,5.751053,3.287596],[-7.437735,-3.438209,7.176079,-2.452104,-2.314307,1.396128,7.538180,-9.241349,6.652279],[5.995886,-2.553663,-0.054496,-3.385286,-2.751379,-8.401638,0.980884,-6.263621,-6.503005],[-8.539114,-3.862684,6.504813,-0.291897,-7.740313,-0.233375,-3.831904,-3.695908,-3.910010],[9.397333,-8.518437,-9.911135,2.787521,2.873833,-4.809383,2.902582,4.076811,4.815366],[6.130018,3.828083,-2.828645,0.607859,9.272531,4.689574,-5.264402,-6.933554,3.454450],[1.119577,-7.609321,-1.056182,-4.221788,-6.663229,7.769137,-8.270333,9.504437,9.005853],[3.494126,-3.955596,-0.224287,1.611044,-7.224847,-7.608368,-4.990875,-5.456082,-9.366987],[7.425840,-9.508038,-8.465149,0.362192,-5.935791,-0.299246,2.750687,-6.340655,7.459642]],[[2.809712,-4.779053,-3.216341,1.216788,-8.460380,-2.291348,-5.510939,-1.098811,5.372742],[4.913162,2.652763,2.438276,9.299529,-0.434348,2.875337,-1.283865,-6.672407,-3.848142],[4.414375,8.965140,-2.517107,8.711211,-2.280598,7.850052,-2.048137,-7.891657,7.641985],[-6.187405,-8.468864,-3.745669,-6.092802,2.404301,4.816046,8.623185,0.339935,9.890521],[-7.835255,5.467653,-8.236291,-7.080285,4.744518,-9.156950,5.082154,5.813120,4.032216],[7.126560,-4.708552,-5.900619,-8.436323,8.228953,3.694709,9.164968,2.710087,-0.721138],[-0.874863,-6.133143,0.265945,-5.077542,3.302694,7.006916,1.193027,-4.262189,6.049805],[-8.339249,1.270267,2.962373,-0.823768,-3.576965,4.152942,9.525111,-5.866753,4.061691],[-1.594267,-0.566504,4.324984,7.593071,-7.868135,5.423917,-0.138437,6.878209,1.070766]],[[-1.075758,-7.561729,9.641543,8.355228,2.148182,-0.492975,-7.787963,-5.334463,4.904816],[-3.488083,-2.714376,-3.166589,-5.268987,5.893771,-8.908922,-5.037472,-9.248257,-9.972351],[5.372854,-0.603792,1.951505,2.026955,-9.676802,-5.524013,-7.052326,-3.310971,3.191647],[-8.500641,6.982423,7.430159,-3.464211,9.691078,-4.578866,-3.263259,9.087313,-4.410160],[8.890215,-3.669898,4.341233,2.278816,4.782679,2.110808,4.129349,-6.313243,8.823161],[-0.127148,-3.658179,1.635992,6.859295,-2.707715,-7.646842,6.306213,-2.090555,7.888144],[-4.648177,4.490720,-9.992368,-5.431644,-1.546692,1.562269,3.284103,7.956608,3.985286],[-1.297218,-5.321294,9.217306,-2.961530,2.904493,0.855155,-2.505428,9.632557,1.399404],[-2.178332,9.212474,0.819472,2.627503,9.731252,-6.376836,3.315788,2.228587,5.337180]],[[-9.816790,-8.605791,-3.835264,8.216480,7.461698,2.455374,-6.451562,9.023077,-0.982442],[-1.848181,8.070060,8.042064,-2.769431,-6.742547,-3.700034,1.138050,7.120332,-3.497396],[2.204126,6.949013,2.251397,7.810676,-9.741995,3.192059,1.141818,-4.747334,8.866986],[-3.754717,-4.473683,-7.751948,-8.765750,-8.049148,3.507013,-0.143625,-7.925251,0.963445],[-8.356080,8.418599,7.801745,9.039853,3.019008,2.846575,2.838102,-6.685404,0.898605],[1.740905,-3.239867,3.411680,-1.451452,-7.910004,4.788384,-1.075273,0.561463,8.976580],[5.182671,9.965509,-8.929938,0.502579,5.080640,-8.105407,-2.101176,-4.558722,3.605155],[-0.546567,8.634793,-3.130489,0.112132,-5.921078,-1.454017,7.610318,6.892337,-3.036432],[-9.091350,2.299270,-8.188303,-3.020789,3.429832,8.608101,-0.079166,0.781018,0.465937]],[[6.516200,-0.490829,7.844530,5.193066,-5.338134,6.298881,-1.414732,4.261156,8.062317],[4.505788,-5.290854,2.375748,3.898621,-3.875183,6.915333,-1.660568,-2.540897,2.807833],[-0.623272,-2.845524,4.235169,5.698777,1.999382,7.082425,-9.526720,-4.560435,-1.046430],[5.515018,-4.454670,-3.631501,4.149725,-3.161339,0.041473,8.039969,-6.256495,5.971019],[4.485118,0.153284,2.156649,-4.890279,-2.983171,-0.158975,-8.029926,-9.061401,8.968653],[-4.850727,-5.146290,-1.226543,-2.236923,9.858790,-2.375171,-7.090817,4.948988,-3.547443],[7.910360,6.598766,-5.585730,-0.414314,8.057677,9.871618,5.789028,4.301856,-4.744179],[3.168861,9.129874,8.347637,6.701616,5.318714,5.024231,7.138581,-3.582453,-8.973779],[8.313240,4.899489,-1.731658,9.473345,3.812560,6.830436,6.974172,-0.551156,9.781699]],[[-3.178535,-3.667960,5.697238,7.624656,-5.806558,9.975457,-5.988901,7.142935,1.744693],[-7.298103,3.359635,1.835975,3.832661,7.100005,2.188511,1.339669,-1.668547,-5.387026],[-0.788851,-7.303232,7.289829,2.575563,-2.914874,-1.364368,-3.693809,5.813151,-5.086136],[-0.659960,6.124152,7.047987,-0.192208,-2.117093,-8.231327,5.761307,-9.333588,1.179984],[0.806118,-9.165101,3.652291,-3.534654,0.351459,0.701996,-3.423257,-8.741636,-5.788782],[0.468874,-0.090249,1.715118,-0.675236,2.018075,-0.860278,-5.632644,-2.845046,-7.667974],[5.541388,2.605827,-2.205552,-6.126938,5.754828,-5.349771,5.909230,-3.904450,6.737444],[-2.035703,-9.885392,-6.429808,6.826020,7.410222,1.276757,3.083468,-5.026077,-5.360765],[-8.003466,0.026187,0.161426,6.628281,7.343263,-5.578679,3.943624,-3.896770,2.888100]],[[-9.787084,-8.348853,-0.311333,4.036657,-6.758815,-3.044948,-0.944828,-3.630776,-9.564612],[2.625161,-2.244236,-8.653751,2.582297,2.431778,8.165541,5.291299,7.804418,5.521871],[2.430145,-1.600932,-4.678203,-1.468400,9.713291,2.758040,-3.974716,1.093413,5.610295],[-3.722205,4.200736,3.723928,-7.543672,-4.741244,-7.845069,9.083290,0.856355,0.221409],[-5.709912,-4.387783,-4.083350,-3.703518,1.536078,1.802402,4.787629,1.869577,-0.163388],[3.302614,5.578607,-9.142451,-8.087650,-1.995790,0.168225,5.441326,5.775312,7.181183],[3.645767,6.738706,7.254090,-6.847543,-1.186106,7.747775,-0.178780,4.659829,2.439381],[3.659566,6.833638,9.226141,2.041222,-8.168465,-9.776948,2.000617,7.563795,-5.269656],[7.395662,6.059122,9.729464,-8.588933,-4.209577,9.450146,-3.401320,4.495674,-2.937745]],[[-6.410933,5.462216,8.469476,9.174310,-2.501235,7.338368,6.097642,0.601856,-5.302127],[-2.210440,-2.124480,-6.082778,-6.541961,-9.890094,-6.381638,-3.856942,-6.345314,8.351166],[5.710713,-7.701313,-7.598664,0.187557,7.873725,-3.940524,1.312028,-2.982699,0.950730],[0.913786,2.836547,8.430118,9.071142,6.411696,-8.812727,6.022353,2.414867,3.049568],[8.465537,4.382158,1.552626,-2.534463,-2.934428,-0.887983,3.355479,1.467235,8.680986],[-7.837609,-7.988916,-7.493447,-4.574315,5.866753,-6.537334,6.014212,8.051628,-6.845647],[0.444607,1.919222,7.603595,2.050851,2.557858,-2.569097,-4.121465,6.461687,-6.262107],[5.593104,2.070163,-8.932075,1.459949,0.761345,-3.629266,3.916871,-2.594177,-5.809780],[-9.082175,-9.924614,-5.743286,-9.076967,0.680570,-1.534873,2.262048,2.316070,-9.367469]],[[-1.716397,0.880447,-5.743337,3.350816,9.357213,-6.586959,-3.560786,-8.823166,-4.493863],[4.658405,2.334156,7.295478,-3.143770,3.672986,7.265553,7.277458,4.001285,-3.558747],[-3.561923,5.111624,3.133498,0.338971,-4.813276,4.365417,2.639780,7.170377,2.254519],[-2.034371,6.796337,8.733144,6.984266,3.432008,-4.510462,-4.011931,1.000873,-6.683044],[9.113468,-4.829623,4.054477,-5.624443,-7.022876,7.189544,-7.420716,4.831896,6.987520],[4.366802,4.466103,0.142762,-1.426722,6.257479,2.236374,-1.476795,3.912908,-2.400763],[-2.828888,8.530771,-4.791552,8.687855,-8.581980,0.048486,3.955505,-2.998223,8.301574],[-0.476151,9.347554,6.480379,-2.110133,4.890827,1.073544,-4.687531,8.891093,-0.393887],[2.090975,-4.275877,5.071695,7.751659,-8.678481,6.544143,-7.799633,0.095593,0.644155]],[[-2.869529,3.263053,-8.339355,-2.408667,-2.716293,-5.562273,5.316906,-0.877629,-3.550883],[1.862355,6.692056,0.316340,7.419772,-9.089306,-0.978669,-0.899196,0.248021,-3.274989],[-9.242065,8.920827,-2.512478,6.444500,3.996134,-9.812397,1.991992,2.690974,1.363171],[-8.530781,-9.926393,-9.345755,-1.134410,-8.607304,-2.659443,5.745623,8.970459,0.589191],[9.677804,1.859851,2.977313,-5.155700,-1.592055,2.610379,-8.551312,-2.666120,-4.285909],[8.667584,8.308006,6.165020,7.091948,-8.714437,9.678768,-1.776044,6.007177,2.887944],[7.041857,7.702472,3.987219,-3.426709,-5.426897,6.695818,1.546359,-0.785113,-9.922686],[-0.387890,-3.889466,-6.661658,-3.705889,-0.032411,-1.677106,-1.487472,7.056275,2.632798],[-6.716504,6.638837,4.445621,-0.219344,4.712347,5.405834,-5.530458,6.588036,5.137503]],[[-9.155234,7.640845,3.685047,-1.915402,0.317406,4.660553,-7.594118,-7.347587,6.622746],[-8.251784,-7.930922,-8.836528,8.241929,6.847380,9.000486,-9.599832,7.882999,2.984377],[-5.942299,-1.088831,9.338727,-6.199832,-8.466085,5.329725,-4.193471,3.430453,-8.387359],[6.527601,-5.695914,-9.610580,2.009864,1.317032,-7.569941,2.291775,-5.967380,8.886803],[1.410029,0.791157,-4.768653,5.554804,1.318953,-3.097178,-0.217776,0.507591,-6.953297],[6.887077,3.221012,-7.642447,6.376709,-5.306583,-6.921751,4.609039,0.436712,5.909833],[-4.530099,2.944733,7.438892,6.762054,5.224300,-1.745905,-1.182002,3.800121,-1.256403],[-7.606299,4.750036,3.286963,-7.345641,0.137846,4.142594,6.395616,-9.514608,4.770038],[-6.500205,3.202094,-0.685909,-1.579110,7.933018,-9.423029,3.929769,8.269720,2.080479]]], dtype='float32')
module1.set_input('var_1936', input_1936)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res3 = intrp3.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res4 = intrp4.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res1_4 = module1.get_output(4).asnumpy()
res2_4 = res2[4].asnumpy()
res3_4 = res3[4].asnumpy()
res4_4 = res4[4].asnumpy()
np.testing.assert_allclose(res1_4 ,res2_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res3_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_4 ,res4_4, atol=1e-3, rtol=1e-3)
(res1_4 == res2_4).all()
(res1_4 == res3_4).all()
(res1_4 == res4_4).all()
res1_5 = module1.get_output(5).asnumpy()
res2_5 = res2[5].asnumpy()
res3_5 = res3[5].asnumpy()
res4_5 = res4[5].asnumpy()
np.testing.assert_allclose(res1_5 ,res2_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_5 ,res3_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_5 ,res4_5, atol=1e-3, rtol=1e-3)
(res1_5 == res2_5).all()
(res1_5 == res3_5).all()
(res1_5 == res4_5).all()
module5.set_input('var_1899', input_1899)
module5.set_input('var_1908', input_1908)
module5.set_input('var_1923', input_1923)
module5.set_input('var_1936', input_1936)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res7 = intrp7.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res8 = intrp8.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res5_4 = module5.get_output(4).asnumpy()
res6_4 = res6[4].asnumpy()
res7_4 = res7[4].asnumpy()
res8_4 = res8[4].asnumpy()
np.testing.assert_allclose(res5_4 ,res6_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res7_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_4 ,res8_4, atol=1e-3, rtol=1e-3)
(res5_4 == res6_4).all()
(res5_4 == res7_4).all()
(res5_4 == res8_4).all()
res5_5 = module5.get_output(5).asnumpy()
res6_5 = res6[5].asnumpy()
res7_5 = res7[5].asnumpy()
res8_5 = res8[5].asnumpy()
np.testing.assert_allclose(res5_5 ,res6_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_5 ,res7_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_5 ,res8_5, atol=1e-3, rtol=1e-3)
(res5_5 == res6_5).all()
(res5_5 == res7_5).all()
(res5_5 == res8_5).all()
module9.set_input('var_1899', input_1899)
module9.set_input('var_1908', input_1908)
module9.set_input('var_1923', input_1923)
module9.set_input('var_1936', input_1936)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res11 = intrp11.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res12 = intrp12.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res9_4 = module9.get_output(4).asnumpy()
res10_4 = res10[4].asnumpy()
res11_4 = res11[4].asnumpy()
res12_4 = res12[4].asnumpy()
np.testing.assert_allclose(res9_4 ,res10_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res11_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_4 ,res12_4, atol=1e-3, rtol=1e-3)
(res9_4 == res10_4).all()
(res9_4 == res11_4).all()
(res9_4 == res12_4).all()
res9_5 = module9.get_output(5).asnumpy()
res10_5 = res10[5].asnumpy()
res11_5 = res11[5].asnumpy()
res12_5 = res12[5].asnumpy()
np.testing.assert_allclose(res9_5 ,res10_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_5 ,res11_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_5 ,res12_5, atol=1e-3, rtol=1e-3)
(res9_5 == res10_5).all()
(res9_5 == res11_5).all()
(res9_5 == res12_5).all()
module13.set_input('var_1899', input_1899)
module13.set_input('var_1908', input_1908)
module13.set_input('var_1923', input_1923)
module13.set_input('var_1936', input_1936)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res15 = intrp15.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res16 = intrp16.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res13_4 = module13.get_output(4).asnumpy()
res14_4 = res14[4].asnumpy()
res15_4 = res15[4].asnumpy()
res16_4 = res16[4].asnumpy()
np.testing.assert_allclose(res13_4 ,res14_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res15_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_4 ,res16_4, atol=1e-3, rtol=1e-3)
(res13_4 == res14_4).all()
(res13_4 == res15_4).all()
(res13_4 == res16_4).all()
res13_5 = module13.get_output(5).asnumpy()
res14_5 = res14[5].asnumpy()
res15_5 = res15[5].asnumpy()
res16_5 = res16[5].asnumpy()
np.testing.assert_allclose(res13_5 ,res14_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_5 ,res15_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_5 ,res16_5, atol=1e-3, rtol=1e-3)
(res13_5 == res14_5).all()
(res13_5 == res15_5).all()
(res13_5 == res16_5).all()
module17.set_input('var_1899', input_1899)
module17.set_input('var_1908', input_1908)
module17.set_input('var_1923', input_1923)
module17.set_input('var_1936', input_1936)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res19 = intrp19.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res20 = intrp20.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res17_4 = module17.get_output(4).asnumpy()
res18_4 = res18[4].asnumpy()
res19_4 = res19[4].asnumpy()
res20_4 = res20[4].asnumpy()
np.testing.assert_allclose(res17_4 ,res18_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res19_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_4 ,res20_4, atol=1e-3, rtol=1e-3)
(res17_4 == res18_4).all()
(res17_4 == res19_4).all()
(res17_4 == res20_4).all()
res17_5 = module17.get_output(5).asnumpy()
res18_5 = res18[5].asnumpy()
res19_5 = res19[5].asnumpy()
res20_5 = res20[5].asnumpy()
np.testing.assert_allclose(res17_5 ,res18_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_5 ,res19_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_5 ,res20_5, atol=1e-3, rtol=1e-3)
(res17_5 == res18_5).all()
(res17_5 == res19_5).all()
(res17_5 == res20_5).all()
module21.set_input('var_1899', input_1899)
module21.set_input('var_1908', input_1908)
module21.set_input('var_1923', input_1923)
module21.set_input('var_1936', input_1936)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res23 = intrp23.evaluate()(input_1899, input_1908, input_1923, input_1936, )
res24 = intrp24.evaluate()(input_1899, input_1908, input_1923, input_1936, )
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
res21_4 = module21.get_output(4).asnumpy()
res22_4 = res22[4].asnumpy()
res23_4 = res23[4].asnumpy()
res24_4 = res24[4].asnumpy()
np.testing.assert_allclose(res21_4 ,res22_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res23_4, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_4 ,res24_4, atol=1e-3, rtol=1e-3)
(res21_4 == res22_4).all()
(res21_4 == res23_4).all()
(res21_4 == res24_4).all()
res21_5 = module21.get_output(5).asnumpy()
res22_5 = res22[5].asnumpy()
res23_5 = res23[5].asnumpy()
res24_5 = res24[5].asnumpy()
np.testing.assert_allclose(res21_5 ,res22_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_5 ,res23_5, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_5 ,res24_5, atol=1e-3, rtol=1e-3)
(res21_5 == res22_5).all()
(res21_5 == res23_5).all()
(res21_5 == res24_5).all()

'''6: TVMFuncCall
5: _ZNSt17_Function_handlerIFvN3tvm7runtime7
4: tvm::runtime::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const [clone .isra.808]
3: tvm::runtime::GraphExecutorCreate(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module const&, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
2: tvm::runtime::GraphExecutor::Init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::Module, std::vector<DLDevice, std::allocator<DLDevice> > const&, tvm::runtime::PackedFunc)
1: tvm::runtime::GraphExecutor::SetupOpExecs()
0: tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::vector<DLTensor, std::allocator<DLTensor> > const&)

'''