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
var_0 = relay.var("var_0", dtype = "float32", shape = (9,))#candidate|0|(9,)|var|float32
uop_1 = relay.sin(var_0.astype('float32')) # shape=(9,)
output = relay.Tuple([uop_1,])
output2 = relay.Tuple([uop_1,])
func_4 = relay.Function([var_0,], output)
mod['func_4'] = func_4
mod = relay.transform.InferType()(mod)
var_5 = relay.var("var_5", dtype = "float32", shape = (9,))#candidate|5|(9,)|var|float32
output = func_4(var_5)
func_6 = relay.Function([var_5], output)
mutated_mod['func_6'] = func_6
mutated_mod = relay.transform.InferType()(mutated_mod)
const_47 = relay.const([[-10,-5,8],[1,1,-5],[-9,4,-1],[8,-5,-7],[4,5,6],[1,10,4],[-1,4,-2],[10,4,-9],[-6,5,7],[4,-3,-1]], dtype = "uint64")#candidate|47|(10, 3)|const|uint64
var_48 = relay.var("var_48", dtype = "uint64", shape = (10, 3))#candidate|48|(10, 3)|var|uint64
bop_49 = relay.left_shift(const_47.astype('uint64'), relay.reshape(var_48.astype('uint64'), relay.shape_of(const_47))) # shape=(10, 3)
bop_52 = relay.greater(const_47.astype('bool'), relay.reshape(bop_49.astype('bool'), relay.shape_of(const_47))) # shape=(10, 3)
uop_60 = relay.log10(bop_52.astype('float64')) # shape=(10, 3)
output = relay.Tuple([uop_60,])
output2 = relay.Tuple([uop_60,])
func_62 = relay.Function([var_48,], output)
mod['func_62'] = func_62
mod = relay.transform.InferType()(mod)
mutated_mod['func_62'] = func_62
mutated_mod = relay.transform.InferType()(mutated_mod)
var_63 = relay.var("var_63", dtype = "uint64", shape = (10, 3))#candidate|63|(10, 3)|var|uint64
func_62_call = mutated_mod.get_global_var('func_62')
call_64 = func_62_call(var_63)
output = call_64
func_65 = relay.Function([var_63], output)
mutated_mod['func_65'] = func_65
mutated_mod = relay.transform.InferType()(mutated_mod)
var_67 = relay.var("var_67", dtype = "float64", shape = (9, 3, 13))#candidate|67|(9, 3, 13)|var|float64
uop_68 = relay.asin(var_67.astype('float64')) # shape=(9, 3, 13)
uop_71 = relay.sqrt(uop_68.astype('float64')) # shape=(9, 3, 13)
uop_75 = relay.tan(uop_71.astype('float64')) # shape=(9, 3, 13)
var_77 = relay.var("var_77", dtype = "float64", shape = (9, 3, 13))#candidate|77|(9, 3, 13)|var|float64
bop_78 = relay.not_equal(uop_68.astype('bool'), relay.reshape(var_77.astype('bool'), relay.shape_of(uop_68))) # shape=(9, 3, 13)
uop_83 = relay.exp(uop_75.astype('float64')) # shape=(9, 3, 13)
bop_85 = relay.minimum(uop_83.astype('uint16'), relay.reshape(var_77.astype('uint16'), relay.shape_of(uop_83))) # shape=(9, 3, 13)
output = relay.Tuple([bop_78,bop_85,])
output2 = relay.Tuple([bop_78,bop_85,])
func_88 = relay.Function([var_67,var_77,], output)
mod['func_88'] = func_88
mod = relay.transform.InferType()(mod)
mutated_mod['func_88'] = func_88
mutated_mod = relay.transform.InferType()(mutated_mod)
func_88_call = mutated_mod.get_global_var('func_88')
var_90 = relay.var("var_90", dtype = "float64", shape = (9, 3, 13))#candidate|90|(9, 3, 13)|var|float64
var_91 = relay.var("var_91", dtype = "float64", shape = (9, 3, 13))#candidate|91|(9, 3, 13)|var|float64
call_89 = func_88_call(var_90,var_91,)
output = call_89
func_92 = relay.Function([var_90,var_91,], output)
mutated_mod['func_92'] = func_92
mutated_mod = relay.transform.InferType()(mutated_mod)
var_112 = relay.var("var_112", dtype = "float64", shape = (16, 6, 11))#candidate|112|(16, 6, 11)|var|float64
uop_113 = relay.sin(var_112.astype('float64')) # shape=(16, 6, 11)
uop_118 = relay.acos(uop_113.astype('float32')) # shape=(16, 6, 11)
uop_120 = relay.atanh(uop_118.astype('float64')) # shape=(16, 6, 11)
bop_124 = relay.mod(uop_120.astype('float64'), relay.reshape(uop_113.astype('float64'), relay.shape_of(uop_120))) # shape=(16, 6, 11)
bop_127 = relay.not_equal(uop_118.astype('bool'), relay.reshape(bop_124.astype('bool'), relay.shape_of(uop_118))) # shape=(16, 6, 11)
bop_132 = relay.left_shift(uop_118.astype('uint32'), relay.reshape(uop_120.astype('uint32'), relay.shape_of(uop_118))) # shape=(16, 6, 11)
bop_135 = relay.right_shift(uop_118.astype('uint64'), relay.reshape(uop_113.astype('uint64'), relay.shape_of(uop_118))) # shape=(16, 6, 11)
func_62_call = mod.get_global_var('func_62')
func_65_call = mutated_mod.get_global_var('func_65')
const_140 = relay.const([-10,1,-5,3,9,3,-7,-1,-2,6,1,2,6,-1,8,4,-6,10,-7,3,4,-5,2,7,9,2,5,-7,9,9], dtype = "uint64")#candidate|140|(30,)|const|uint64
call_139 = relay.TupleGetItem(func_62_call(relay.reshape(const_140.astype('uint64'), [10, 3])), 0)
call_141 = relay.TupleGetItem(func_65_call(relay.reshape(const_140.astype('uint64'), [10, 3])), 0)
bop_145 = relay.greater(bop_135.astype('bool'), relay.reshape(bop_127.astype('bool'), relay.shape_of(bop_135))) # shape=(16, 6, 11)
output = relay.Tuple([bop_132,call_139,const_140,bop_145,])
output2 = relay.Tuple([bop_132,call_141,const_140,bop_145,])
func_153 = relay.Function([var_112,], output)
mod['func_153'] = func_153
mod = relay.transform.InferType()(mod)
mutated_mod['func_153'] = func_153
mutated_mod = relay.transform.InferType()(mutated_mod)
var_154 = relay.var("var_154", dtype = "float64", shape = (16, 6, 11))#candidate|154|(16, 6, 11)|var|float64
func_153_call = mutated_mod.get_global_var('func_153')
call_155 = func_153_call(var_154)
output = call_155
func_156 = relay.Function([var_154], output)
mutated_mod['func_156'] = func_156
mutated_mod = relay.transform.InferType()(mutated_mod)
var_174 = relay.var("var_174", dtype = "float32", shape = ())#candidate|174|()|var|float32
var_175 = relay.var("var_175", dtype = "float32", shape = (1, 6))#candidate|175|(1, 6)|var|float32
bop_176 = relay.add(var_174.astype('float32'), var_175.astype('float32')) # shape=(1, 6)
func_88_call = mod.get_global_var('func_88')
func_92_call = mutated_mod.get_global_var('func_92')
const_183 = relay.const([-0.200699,-6.323727,-4.709326,0.424721,-0.857946,-4.161253,6.567237,4.089284,3.566223,-1.683361,-1.857381,-3.307410,-2.314975,8.268402,6.320118,-8.858945,8.323246,8.843653,8.529267,-4.988219,3.181332,6.260890,-4.431018,7.248875,9.659453,3.050774,-5.375375,8.130653,4.636860,9.594843,-7.511862,-9.758298,-6.068782,-0.730004,-2.635415,-5.536209,0.888866,-3.577150,-4.049687,-2.646691,8.630896,-5.894050,0.022979,-8.830485,-5.196123,2.466415,6.231888,-6.930628,-3.090565,3.696366,6.832721,-8.341214,3.130728,4.763507,8.067201,2.204574,6.353029,-2.567197,-8.835508,2.030906,-5.955464,4.181974,-2.671036,-0.912369,-1.243864,3.336217,9.520278,8.579545,8.602777,-6.512702,-0.146848,1.290973,8.905192,-0.536250,0.270343,-9.252290,7.410458,3.394151,8.132586,-0.326969,-9.925050,-5.331760,4.409445,-3.604909,-5.127436,5.591831,7.990358,-6.919704,9.771735,-6.080100,-6.504735,-1.657072,-9.476355,6.806915,7.851156,-6.969201,2.966747,8.820777,-4.917563,6.041812,5.436639,5.177187,-3.350293,5.410117,-3.928347,-9.275415,1.855012,-0.320805,-3.988058,0.756261,3.519902,6.617088,-1.658913,-5.762206,3.378826,4.065994,-1.851841,0.603426,-7.289426,-7.017161,-2.463592,7.158308,-7.399828,4.773662,8.687798,-9.404807,-6.267604,-0.809882,2.078431,4.528091,-4.249775,9.636075,7.543344,7.465921,1.935393,2.058915,-3.432364,7.266028,-9.229490,-7.748305,-4.369352,1.622551,-0.750156,0.457303,-9.171321,-8.581472,-3.529840,3.478135,-6.879334,7.766235,1.046554,7.730115,-5.449029,-0.510124,-1.681118,-0.334988,3.832147,5.502217,4.368079,1.788539,-8.593560,0.846316,2.658785,-4.705616,2.648567,-2.915785,-8.941623,4.360104,-1.546031,-7.329674,-7.027147,3.224826,-5.672977,2.815362,-6.010942,6.441805,2.516395,9.848217,7.326081,-7.512823,5.719407,2.759769,4.493506,-6.768122,-9.386618,-5.485051,-2.326980,0.016047,-6.755706,-1.001130,5.987604,9.690250,-8.351019,-4.985096,2.962219,7.785472,9.277598,-4.413543,0.596826,-5.373904,9.966923,-3.559881,0.928099,-0.620327,-8.822945,-6.978315,-8.742477,-1.648279,-9.798285,0.887681,-9.581715,-2.260564,6.814476,-7.711842,-9.909436,4.119965,1.834265,1.174059,4.321887,-5.419071,-3.379125,9.178665,7.725313,-1.324246,-1.134071,7.488247,-9.127594,-0.493118,-0.694680,-1.130475,-7.131466,1.335417,8.131331,-2.175484,-5.402215,-1.491266,0.502810,-0.389319,-6.997985,-9.759561,-2.128246,2.225558,2.418651,-0.444293,-6.346815,-1.781964,-2.335703,-3.356956,-1.695876,-2.919102,3.445979,-2.855135,4.598235,9.225324,4.364585,-5.816788,6.913838,-4.028959,-2.923711,9.792625,8.755428,6.143623,8.031557,3.365126,-1.638795,-2.338411,-5.761895,-7.598194,9.325671,7.068768,2.536158,-1.561737,3.324995,8.275188,-5.165429,-1.754763,9.696993,7.028654,0.798761,-7.667495,5.183859,-5.501344,0.710456,6.962486,-6.383023,4.203207,-9.231910,8.429019,-1.494259,5.831785,7.546844,-9.634028,4.152542,4.718234,-4.303711,-0.499629,-1.462761,-1.136871,8.733948,5.568135,-4.999287,0.830150,-7.161900,9.128992,-2.705743,3.766234,-9.876186,-6.917778,-2.545114,2.124837,8.696721,-6.780922,-2.578180,-2.430257,-3.106521,9.678681,-1.203208,3.943953,8.091915,-1.744609,-6.353347,8.366074,1.006104,-6.501717,9.978050,7.229561,-2.401899,0.650080,-8.478252,-2.989798,0.005524,3.382357,7.623483,-2.429046,4.672960,-1.484636,7.024571,7.116406,8.709113,-3.641889,9.787018,4.904871,4.717257,-6.232471,-9.544949,-1.688180,-4.268123,-2.507754,-0.514801,5.455554,9.094742], dtype = "float64")#candidate|183|(351,)|const|float64
call_182 = relay.TupleGetItem(func_88_call(relay.reshape(const_183.astype('float64'), [9, 3, 13]), relay.reshape(const_183.astype('float64'), [9, 3, 13]), ), 0)
call_184 = relay.TupleGetItem(func_92_call(relay.reshape(const_183.astype('float64'), [9, 3, 13]), relay.reshape(const_183.astype('float64'), [9, 3, 13]), ), 0)
output = relay.Tuple([bop_176,call_182,const_183,])
output2 = relay.Tuple([bop_176,call_184,const_183,])
func_190 = relay.Function([var_174,var_175,], output)
mod['func_190'] = func_190
mod = relay.transform.InferType()(mod)
mutated_mod['func_190'] = func_190
mutated_mod = relay.transform.InferType()(mutated_mod)
func_190_call = mutated_mod.get_global_var('func_190')
var_192 = relay.var("var_192", dtype = "float32", shape = ())#candidate|192|()|var|float32
var_193 = relay.var("var_193", dtype = "float32", shape = (1, 6))#candidate|193|(1, 6)|var|float32
call_191 = func_190_call(var_192,var_193,)
output = call_191
func_194 = relay.Function([var_192,var_193,], output)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
var_196 = relay.var("var_196", dtype = "float32", shape = (12,))#candidate|196|(12,)|var|float32
var_197 = relay.var("var_197", dtype = "float32", shape = (12,))#candidate|197|(12,)|var|float32
bop_198 = relay.floor_mod(var_196.astype('float32'), relay.reshape(var_197.astype('float32'), relay.shape_of(var_196))) # shape=(12,)
bop_201 = relay.power(var_196.astype('float64'), relay.reshape(bop_198.astype('float64'), relay.shape_of(var_196))) # shape=(12,)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
var_207 = relay.var("var_207", dtype = "float32", shape = (9,))#candidate|207|(9,)|var|float32
call_206 = relay.TupleGetItem(func_4_call(relay.reshape(var_207.astype('float32'), [9,])), 0)
call_208 = relay.TupleGetItem(func_6_call(relay.reshape(var_207.astype('float32'), [9,])), 0)
uop_209 = relay.log10(var_197.astype('float32')) # shape=(12,)
var_213 = relay.var("var_213", dtype = "float32", shape = (12,))#candidate|213|(12,)|var|float32
bop_214 = relay.floor_mod(uop_209.astype('float32'), relay.reshape(var_213.astype('float32'), relay.shape_of(uop_209))) # shape=(12,)
bop_223 = relay.left_shift(bop_214.astype('uint16'), relay.reshape(uop_209.astype('uint16'), relay.shape_of(bop_214))) # shape=(12,)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
call_235 = relay.TupleGetItem(func_4_call(relay.reshape(var_207.astype('float32'), [9,])), 0)
call_236 = relay.TupleGetItem(func_6_call(relay.reshape(var_207.astype('float32'), [9,])), 0)
var_237 = relay.var("var_237", dtype = "float32", shape = (12,))#candidate|237|(12,)|var|float32
bop_238 = relay.multiply(uop_209.astype('float32'), relay.reshape(var_237.astype('float32'), relay.shape_of(uop_209))) # shape=(12,)
uop_241 = relay.erf(bop_223.astype('float32')) # shape=(12,)
bop_249 = relay.subtract(uop_241.astype('uint64'), relay.reshape(var_197.astype('uint64'), relay.shape_of(uop_241))) # shape=(12,)
bop_256 = relay.less(uop_241.astype('bool'), relay.reshape(var_213.astype('bool'), relay.shape_of(uop_241))) # shape=(12,)
bop_259 = relay.greater(uop_209.astype('bool'), relay.reshape(bop_256.astype('bool'), relay.shape_of(uop_209))) # shape=(12,)
var_264 = relay.var("var_264", dtype = "float32", shape = (12,))#candidate|264|(12,)|var|float32
bop_265 = relay.less_equal(uop_209.astype('bool'), relay.reshape(var_264.astype('bool'), relay.shape_of(uop_209))) # shape=(12,)
uop_268 = relay.exp(bop_259.astype('float32')) # shape=(12,)
var_270 = relay.var("var_270", dtype = "float32", shape = (12,))#candidate|270|(12,)|var|float32
bop_271 = relay.add(uop_268.astype('uint8'), relay.reshape(var_270.astype('uint8'), relay.shape_of(uop_268))) # shape=(12,)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
call_275 = relay.TupleGetItem(func_4_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
call_276 = relay.TupleGetItem(func_6_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
uop_277 = relay.erf(uop_241.astype('float32')) # shape=(12,)
uop_279 = relay.cos(bop_256.astype('float64')) # shape=(12,)
bop_281 = relay.bitwise_and(bop_259.astype('uint8'), relay.reshape(var_270.astype('uint8'), relay.shape_of(bop_259))) # shape=(12,)
bop_284 = relay.mod(bop_265.astype('float64'), relay.reshape(var_196.astype('float64'), relay.shape_of(bop_265))) # shape=(12,)
bop_287 = relay.subtract(uop_268.astype('int16'), relay.reshape(bop_256.astype('int16'), relay.shape_of(uop_268))) # shape=(12,)
uop_296 = relay.sqrt(uop_277.astype('float64')) # shape=(12,)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
call_300 = relay.TupleGetItem(func_4_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
call_301 = relay.TupleGetItem(func_6_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
bop_302 = relay.floor_divide(uop_296.astype('float64'), relay.reshape(bop_259.astype('float64'), relay.shape_of(uop_296))) # shape=(12,)
func_190_call = mod.get_global_var('func_190')
func_194_call = mutated_mod.get_global_var('func_194')
var_316 = relay.var("var_316", dtype = "float32", shape = ())#candidate|316|()|var|float32
var_317 = relay.var("var_317", dtype = "float32", shape = (6,))#candidate|317|(6,)|var|float32
call_315 = relay.TupleGetItem(func_190_call(relay.reshape(var_316.astype('float32'), []), relay.reshape(var_317.astype('float32'), [1, 6]), ), 1)
call_318 = relay.TupleGetItem(func_194_call(relay.reshape(var_316.astype('float32'), []), relay.reshape(var_317.astype('float32'), [1, 6]), ), 1)
func_88_call = mod.get_global_var('func_88')
func_92_call = mutated_mod.get_global_var('func_92')
call_319 = relay.TupleGetItem(func_88_call(relay.reshape(call_315.astype('float64'), [9, 3, 13]), relay.reshape(call_315.astype('float64'), [9, 3, 13]), ), 0)
call_320 = relay.TupleGetItem(func_92_call(relay.reshape(call_315.astype('float64'), [9, 3, 13]), relay.reshape(call_315.astype('float64'), [9, 3, 13]), ), 0)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
call_321 = relay.TupleGetItem(func_4_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
call_322 = relay.TupleGetItem(func_6_call(relay.reshape(call_206.astype('float32'), [9,])), 0)
func_153_call = mod.get_global_var('func_153')
func_156_call = mutated_mod.get_global_var('func_156')
var_327 = relay.var("var_327", dtype = "float64", shape = (1056,))#candidate|327|(1056,)|var|float64
call_326 = relay.TupleGetItem(func_153_call(relay.reshape(var_327.astype('float64'), [16, 6, 11])), 0)
call_328 = relay.TupleGetItem(func_156_call(relay.reshape(var_327.astype('float64'), [16, 6, 11])), 0)
bop_330 = relay.not_equal(uop_296.astype('bool'), relay.reshape(var_196.astype('bool'), relay.shape_of(uop_296))) # shape=(12,)
func_62_call = mod.get_global_var('func_62')
func_65_call = mutated_mod.get_global_var('func_65')
var_337 = relay.var("var_337", dtype = "uint64", shape = (30,))#candidate|337|(30,)|var|uint64
call_336 = relay.TupleGetItem(func_62_call(relay.reshape(var_337.astype('uint64'), [10, 3])), 0)
call_338 = relay.TupleGetItem(func_65_call(relay.reshape(var_337.astype('uint64'), [10, 3])), 0)
output = relay.Tuple([bop_201,call_206,var_207,call_235,bop_238,bop_249,bop_271,call_275,uop_279,bop_281,bop_284,bop_287,call_300,bop_302,call_315,var_316,var_317,call_319,call_321,call_326,var_327,bop_330,call_336,var_337,])
output2 = relay.Tuple([bop_201,call_208,var_207,call_236,bop_238,bop_249,bop_271,call_276,uop_279,bop_281,bop_284,bop_287,call_301,bop_302,call_318,var_316,var_317,call_320,call_322,call_328,var_327,bop_330,call_338,var_337,])
func_339 = relay.Function([var_196,var_197,var_207,var_213,var_237,var_264,var_270,var_316,var_317,var_327,var_337,], output)
mod['func_339'] = func_339
mod = relay.transform.InferType()(mod)
mutated_mod['func_339'] = func_339
mutated_mod = relay.transform.InferType()(mutated_mod)
func_339_call = mutated_mod.get_global_var('func_339')
var_341 = relay.var("var_341", dtype = "float32", shape = (12,))#candidate|341|(12,)|var|float32
var_342 = relay.var("var_342", dtype = "float32", shape = (12,))#candidate|342|(12,)|var|float32
var_343 = relay.var("var_343", dtype = "float32", shape = (9,))#candidate|343|(9,)|var|float32
var_344 = relay.var("var_344", dtype = "float32", shape = (12,))#candidate|344|(12,)|var|float32
var_345 = relay.var("var_345", dtype = "float32", shape = (12,))#candidate|345|(12,)|var|float32
var_346 = relay.var("var_346", dtype = "float32", shape = (12,))#candidate|346|(12,)|var|float32
var_347 = relay.var("var_347", dtype = "float32", shape = (12,))#candidate|347|(12,)|var|float32
var_348 = relay.var("var_348", dtype = "float32", shape = ())#candidate|348|()|var|float32
var_349 = relay.var("var_349", dtype = "float32", shape = (6,))#candidate|349|(6,)|var|float32
var_350 = relay.var("var_350", dtype = "float64", shape = (1056,))#candidate|350|(1056,)|var|float64
var_351 = relay.var("var_351", dtype = "uint64", shape = (30,))#candidate|351|(30,)|var|uint64
call_340 = func_339_call(var_341,var_342,var_343,var_344,var_345,var_346,var_347,var_348,var_349,var_350,var_351,)
output = call_340
func_352 = relay.Function([var_341,var_342,var_343,var_344,var_345,var_346,var_347,var_348,var_349,var_350,var_351,], output)
mutated_mod['func_352'] = func_352
mutated_mod = relay.transform.InferType()(mutated_mod)
var_376 = relay.var("var_376", dtype = "bool", shape = (5, 16))#candidate|376|(5, 16)|var|bool
var_377 = relay.var("var_377", dtype = "bool", shape = (5, 16))#candidate|377|(5, 16)|var|bool
bop_378 = relay.logical_and(var_376.astype('bool'), relay.reshape(var_377.astype('bool'), relay.shape_of(var_376))) # shape=(5, 16)
bop_381 = relay.greater_equal(bop_378.astype('bool'), relay.reshape(var_376.astype('bool'), relay.shape_of(bop_378))) # shape=(5, 16)
uop_391 = relay.log2(bop_381.astype('float64')) # shape=(5, 16)
uop_394 = relay.cos(uop_391.astype('float32')) # shape=(5, 16)
uop_396 = relay.sqrt(uop_394.astype('float32')) # shape=(5, 16)
uop_403 = relay.log(uop_394.astype('float32')) # shape=(5, 16)
bop_405 = relay.greater(uop_396.astype('bool'), relay.reshape(var_377.astype('bool'), relay.shape_of(uop_396))) # shape=(5, 16)
func_153_call = mod.get_global_var('func_153')
func_156_call = mutated_mod.get_global_var('func_156')
var_410 = relay.var("var_410", dtype = "float64", shape = (1056, 1))#candidate|410|(1056, 1)|var|float64
call_409 = relay.TupleGetItem(func_153_call(relay.reshape(var_410.astype('float64'), [16, 6, 11])), 1)
call_411 = relay.TupleGetItem(func_156_call(relay.reshape(var_410.astype('float64'), [16, 6, 11])), 1)
output = relay.Tuple([uop_403,bop_405,call_409,var_410,])
output2 = relay.Tuple([uop_403,bop_405,call_411,var_410,])
func_412 = relay.Function([var_376,var_377,var_410,], output)
mod['func_412'] = func_412
mod = relay.transform.InferType()(mod)
mutated_mod['func_412'] = func_412
mutated_mod = relay.transform.InferType()(mutated_mod)
func_412_call = mutated_mod.get_global_var('func_412')
var_414 = relay.var("var_414", dtype = "bool", shape = (5, 16))#candidate|414|(5, 16)|var|bool
var_415 = relay.var("var_415", dtype = "bool", shape = (5, 16))#candidate|415|(5, 16)|var|bool
var_416 = relay.var("var_416", dtype = "float64", shape = (1056, 1))#candidate|416|(1056, 1)|var|float64
call_413 = func_412_call(var_414,var_415,var_416,)
output = call_413
func_417 = relay.Function([var_414,var_415,var_416,], output)
mutated_mod['func_417'] = func_417
mutated_mod = relay.transform.InferType()(mutated_mod)
var_438 = relay.var("var_438", dtype = "float32", shape = (1,))#candidate|438|(1,)|var|float32
uop_439 = relay.cosh(var_438.astype('float32')) # shape=(1,)
uop_442 = relay.sinh(uop_439.astype('float32')) # shape=(1,)
func_88_call = mod.get_global_var('func_88')
func_92_call = mutated_mod.get_global_var('func_92')
const_445 = relay.const([-3.231095,-3.348234,-3.029711,-5.418618,7.024061,-0.127529,9.428712,-8.309195,-7.324234,3.306849,6.466933,-4.937219,-8.349965,-3.589109,4.313533,-4.641642,0.191288,-5.523206,7.014935,5.603013,9.270446,-4.526738,-5.232223,-8.895392,1.699069,0.686663,7.561543,-8.209138,5.570427,2.455224,0.709714,-7.935939,0.395829,-1.721622,-9.126159,9.342855,8.021137,-6.001043,0.386036,2.129428,2.973124,9.538955,-6.796503,-9.640669,-0.661632,-0.726414,-0.867401,3.080253,-0.386111,-3.768602,2.448428,-8.533660,4.790146,1.055444,0.296931,5.898824,6.240732,3.407424,-2.915437,0.827166,-4.248711,9.046553,2.991512,-9.802464,-7.570259,-8.135361,-6.128036,4.844848,2.767816,-2.907203,3.009337,1.461266,9.982510,-4.911412,2.329969,-1.570437,0.064247,-6.208782,-1.696060,-4.599974,2.702696,8.733048,2.406863,4.241136,5.670246,3.068407,4.970952,3.335950,-8.538358,4.437393,9.159597,2.117762,5.279709,6.017381,-1.839939,-7.258218,-5.429455,9.498604,-1.342932,4.098309,8.672171,-2.191734,4.373197,2.716461,7.537348,9.305077,-7.412709,-9.130041,-1.335249,-6.233757,1.186846,-2.904074,-4.535242,-9.037185,-9.093227,-5.417442,-7.193954,6.807014,-2.909412,9.608608,3.493557,-3.039488,2.222840,9.742900,1.489254,-9.324600,-1.938520,-9.082942,6.089321,-9.020519,-5.169534,2.254572,3.969575,2.460354,6.374506,-7.187985,3.152467,-6.417564,5.457849,8.646765,8.451682,3.682302,-6.324894,-2.947552,-8.005515,-8.485820,8.581054,-8.719301,-5.153078,-0.547644,-8.988700,-8.531565,-3.003200,1.267206,-5.405707,-7.823842,-3.005210,7.484122,8.985503,-3.684649,-0.561398,6.947553,-1.362122,2.017579,0.635624,-6.431717,-2.768219,5.973867,-2.047447,-1.331862,1.166437,-8.569696,-4.358360,5.888647,-0.751437,-5.267100,-9.288835,-4.059211,7.851007,0.346449,-4.430940,-6.646684,-5.177024,9.940554,-8.547006,0.400418,-6.392109,-5.266422,0.087317,-6.320256,4.672268,1.425939,-3.876893,8.585228,9.759308,-0.624381,-8.857171,-8.239147,8.419738,6.089679,0.073065,5.921741,0.904234,-8.164449,0.644458,-0.495020,4.043733,-3.938465,8.683293,6.546526,-8.945728,-8.659540,7.252078,5.658296,-9.767476,-3.162098,5.793548,3.976307,7.806794,-4.443918,7.830873,-8.665490,-7.327549,-2.815051,-8.989623,-5.227316,-7.077332,7.219728,4.831249,-9.426983,-4.438390,1.629577,-4.180409,-5.691300,-6.945090,3.579642,1.487314,1.462034,-1.840523,6.372380,5.149493,-3.931972,3.249765,5.064117,-9.522686,-1.218519,9.224553,8.051275,0.085960,8.361332,-2.518115,5.201752,-9.992708,9.255428,-1.520717,0.928316,-5.139981,5.756408,-1.679458,-5.388267,-5.575652,1.189737,-8.703416,-8.510787,-9.523029,9.774859,-2.691613,-3.755930,5.319378,4.276469,-8.860157,-5.449327,-9.022820,-4.093647,-2.920601,-1.358016,-5.023467,-5.462532,-1.582598,-0.529940,7.840600,3.785946,-8.084250,-8.700728,1.268351,0.554956,7.557175,-1.889192,-3.345481,-8.024770,1.387265,9.157670,-7.072212,-3.356613,-6.981736,4.738360,-6.736887,2.007519,-2.279704,3.422705,-1.375976,5.043579,-9.323221,0.877063,9.521827,7.941060,-2.704015,-3.173080,7.912387,4.812733,8.992248,6.473156,0.665383,-3.317932,-1.078748,7.501093,-9.090143,-9.203105,-2.810769,2.753040,0.569836,-7.290240,-6.100890,-5.352447,3.428203,6.341108,-0.313214,1.439322,-1.695500,1.772640,6.496730,9.155082,-2.114726,-8.951466,2.146341,-6.905197,-4.538991,8.849202,5.320354,-9.144673,2.299943,7.447361,-7.538686,-2.356876,-8.403396,-5.609147,-1.581435,7.592768,8.042965,-4.101465,-8.977827], dtype = "float64")#candidate|445|(351,)|const|float64
call_444 = relay.TupleGetItem(func_88_call(relay.reshape(const_445.astype('float64'), [9, 3, 13]), relay.reshape(const_445.astype('float64'), [9, 3, 13]), ), 0)
call_446 = relay.TupleGetItem(func_92_call(relay.reshape(const_445.astype('float64'), [9, 3, 13]), relay.reshape(const_445.astype('float64'), [9, 3, 13]), ), 0)
func_4_call = mod.get_global_var('func_4')
func_6_call = mutated_mod.get_global_var('func_6')
const_449 = relay.const([[-0.418851,2.066414,7.136987],[1.486515,-3.144975,7.654411],[6.613531,-8.184206,2.929221]], dtype = "float32")#candidate|449|(3, 3)|const|float32
call_448 = relay.TupleGetItem(func_4_call(relay.reshape(const_449.astype('float32'), [9,])), 0)
call_450 = relay.TupleGetItem(func_6_call(relay.reshape(const_449.astype('float32'), [9,])), 0)
bop_451 = relay.less(uop_442.astype('bool'), call_444.astype('bool')) # shape=(9, 3, 13)
bop_454 = relay.less(uop_442.astype('bool'), call_446.astype('bool')) # shape=(9, 3, 13)
uop_455 = relay.erf(uop_439.astype('float64')) # shape=(1,)
uop_458 = relay.asinh(uop_455.astype('float32')) # shape=(1,)
bop_460 = relay.floor_divide(uop_458.astype('float32'), relay.reshape(uop_439.astype('float32'), relay.shape_of(uop_458))) # shape=(1,)
output = relay.Tuple([const_445,call_448,const_449,bop_451,bop_460,])
output2 = relay.Tuple([const_445,call_450,const_449,bop_454,bop_460,])
func_465 = relay.Function([var_438,], output)
mod['func_465'] = func_465
mod = relay.transform.InferType()(mod)
mutated_mod['func_465'] = func_465
mutated_mod = relay.transform.InferType()(mutated_mod)
var_466 = relay.var("var_466", dtype = "float32", shape = (1,))#candidate|466|(1,)|var|float32
func_465_call = mutated_mod.get_global_var('func_465')
call_467 = func_465_call(var_466)
output = call_467
func_468 = relay.Function([var_466], output)
mutated_mod['func_468'] = func_468
mutated_mod = relay.transform.InferType()(mutated_mod)
var_487 = relay.var("var_487", dtype = "float32", shape = (7, 5))#candidate|487|(7, 5)|var|float32
uop_488 = relay.asin(var_487.astype('float32')) # shape=(7, 5)
uop_491 = relay.atan(uop_488.astype('float32')) # shape=(7, 5)
output = relay.Tuple([uop_491,])
output2 = relay.Tuple([uop_491,])
func_493 = relay.Function([var_487,], output)
mod['func_493'] = func_493
mod = relay.transform.InferType()(mod)
var_494 = relay.var("var_494", dtype = "float32", shape = (7, 5))#candidate|494|(7, 5)|var|float32
output = func_493(var_494)
func_495 = relay.Function([var_494], output)
mutated_mod['func_495'] = func_495
mutated_mod = relay.transform.InferType()(mutated_mod)
const_544 = relay.const([[[-1,-1,-2,9,9,6,-6,1,-10,-1,-7,3,-3],[8,-1,10,-5,-6,-3,4,8,-2,-9,-8,5,-6],[-4,6,7,7,-6,1,6,-10,-2,-1,-4,-5,-7],[4,4,-6,2,-6,-3,4,-10,5,-2,-6,-9,5],[1,6,2,-10,1,-3,-6,-1,-3,-2,-2,2,-10],[-8,-7,-2,-10,-6,10,7,2,6,10,-10,-2,10],[7,-8,10,10,7,-6,2,-9,8,6,7,-7,-9],[4,9,-5,-3,7,-9,1,-7,-4,8,3,-9,-3],[1,5,-9,3,-2,4,1,1,8,-5,8,6,-1],[10,-8,-8,5,10,4,-1,2,-2,-9,-3,4,2],[5,9,-6,8,4,-6,-1,4,-2,2,2,-6,-6],[8,7,3,7,-8,1,-1,-5,-5,-5,-2,5,10],[-7,-6,7,4,-3,4,8,-7,9,7,6,10,2],[6,-7,3,-7,8,5,6,8,-10,-3,9,-6,5]],[[8,-1,1,-6,-8,-4,-9,6,6,-4,3,3,2],[9,-8,-7,-2,-5,-8,1,3,8,10,8,6,-2],[-5,-5,7,-2,-10,1,9,-5,10,-10,5,-5,-2],[1,-4,-8,-8,-5,-9,-1,-7,-1,2,-3,5,-3],[3,-5,10,6,-1,-4,-9,5,7,-9,10,-3,7],[3,9,-2,6,9,3,2,10,-4,-5,9,5,-6],[-7,7,5,5,10,8,5,3,-3,-4,1,1,7],[5,7,8,2,-9,-9,-5,-2,-4,2,-6,2,-5],[-8,-8,8,6,7,-9,-9,-10,-3,-7,7,3,-1],[-4,1,-6,8,5,-9,1,-4,7,-9,2,-3,8],[-2,-1,1,6,5,-2,-5,-10,7,10,-2,-5,-5],[6,7,8,-6,-2,4,-9,-9,-3,6,-1,-7,-4],[-10,-1,-7,-5,-2,6,3,2,-4,-6,2,7,5],[2,-1,-10,-5,-7,9,-1,-6,8,2,7,-3,-9]],[[6,-2,6,-2,10,7,6,-2,-8,7,-7,-3,5],[7,7,-2,3,-4,9,5,2,7,-6,5,9,7],[-9,6,-2,-3,-8,2,10,-10,1,-10,-7,-1,5],[8,9,1,6,3,10,-9,-8,10,4,-9,-5,-10],[-4,-7,8,-5,9,-7,5,6,-4,2,5,-2,-1],[-6,9,-1,8,4,-3,4,-6,8,-9,-7,-1,-6],[2,-7,-10,1,10,2,2,-5,6,-7,-10,-6,-1],[-3,1,-1,1,7,-3,-1,-4,3,1,-9,7,3],[-2,8,1,10,-3,6,10,-7,6,8,-3,4,-8],[4,10,5,-6,9,-6,-1,9,5,3,-2,10,-6],[10,-6,4,6,-7,10,-5,1,8,-2,-9,8,-1],[-8,-10,-3,-6,-9,-10,6,7,9,5,-3,1,-1],[4,-2,-10,5,5,-10,-8,-3,3,4,-7,2,-4],[-6,2,-5,9,-3,-8,-8,-8,-5,-5,-5,-10,-10]],[[5,-1,-7,-5,-6,-7,-10,3,3,-2,-8,-9,5],[-3,6,-2,-3,9,-9,3,7,-10,2,-2,5,-6],[-10,7,-1,-10,2,-8,-6,9,1,1,-1,2,6],[-10,9,5,-3,7,-1,-10,1,-1,5,2,-1,-6],[6,8,4,-7,4,-4,-2,-2,8,-9,-2,-5,10],[4,2,6,10,4,-1,-8,-10,-3,-3,-9,-9,-1],[-9,-9,-6,4,5,-8,3,-8,10,9,-10,4,7],[4,2,-8,-2,1,2,-5,-5,-3,-4,1,8,-6],[4,10,3,-9,1,-7,-6,10,4,-6,-2,-7,1],[10,10,8,6,-5,-5,-8,-2,9,3,-6,7,-8],[-3,5,5,-5,-3,-8,4,2,8,4,-2,9,-2],[-6,9,10,-1,7,-2,10,9,6,3,-3,-5,2],[9,5,-6,-2,7,8,9,-1,-10,-8,-10,2,-10],[4,-3,-3,10,6,10,-3,-2,-2,-6,3,7,7]],[[-4,1,-6,-9,3,-1,-1,1,5,-8,-2,-9,5],[-6,-5,-8,-6,5,5,-10,8,-3,8,-10,-7,1],[9,10,4,-6,1,-8,-4,8,6,10,-9,3,-9],[10,3,5,-1,-8,1,2,8,8,2,3,-3,-4],[-8,-10,1,-4,3,-8,4,-1,-8,8,4,9,-1],[-6,5,-9,-6,-1,10,3,5,-10,-5,-6,6,8],[10,-6,10,-5,1,-10,-4,-5,-8,-10,8,-4,2],[6,3,-8,1,-4,-7,8,5,-6,-10,-3,-7,-3],[-4,-8,6,6,7,-2,9,1,-7,5,-6,-10,2],[-7,8,-6,5,-3,-9,2,-5,6,10,-2,-8,-10],[-8,3,6,8,9,8,-9,7,-10,-4,3,8,9],[2,-7,-4,-3,-4,10,-4,-8,-3,3,-5,-10,7],[10,7,-7,9,-1,8,3,-4,-6,9,9,10,-5],[5,3,7,9,-5,2,10,-8,-7,2,1,10,-1]],[[6,-2,-8,4,-3,5,6,-8,-5,-10,-9,-1,2],[7,3,-2,-1,10,3,3,9,-8,9,1,-4,-4],[-2,4,-8,-4,10,2,10,3,9,9,-3,1,3],[-5,-5,-2,5,-10,6,5,-2,9,2,7,-7,9],[3,3,4,-5,-6,-7,-2,3,-1,-1,-7,6,4],[9,10,-3,-8,1,4,-2,-4,3,-3,1,5,8],[-4,-1,8,10,-10,2,-9,1,-10,6,-1,9,5],[10,4,4,-2,8,5,-9,3,4,-6,-5,-9,2],[10,-3,-1,-5,-1,-5,-3,5,3,4,9,7,-6],[-4,3,-2,7,-8,-8,-5,3,2,-5,4,4,5],[10,9,-4,9,-8,4,8,-6,-6,-5,-3,5,-5],[7,8,3,3,4,-6,9,-5,3,-5,-10,-10,-2],[-4,-10,-7,-10,-8,-5,-8,2,-1,10,-6,-9,-2],[3,-6,4,3,10,-10,9,3,10,-4,-5,-5,-2]],[[10,1,-2,3,-1,-7,8,8,-8,-7,-2,4,1],[-10,-6,7,-6,9,5,6,-8,2,-2,5,7,3],[-9,3,8,-8,10,2,4,-3,-3,4,-6,1,-7],[10,-10,-1,-9,1,-2,4,4,5,7,-8,-9,-6],[-9,2,3,9,-4,2,-7,-3,-4,10,-2,6,4],[-10,6,-7,2,2,1,9,-4,-3,8,1,-2,7],[-9,-1,-9,-6,-8,-2,4,8,-5,-3,1,-1,9],[-10,4,-4,2,-6,8,-8,4,-9,-3,-4,-5,7],[-2,-5,-10,-3,-2,-7,-4,-7,-2,-1,4,9,-8],[-4,-1,-10,-1,5,2,8,-6,-10,6,3,4,-8],[-5,-4,9,7,-10,2,-4,5,-6,-9,5,-1,9],[-1,-9,1,-6,-6,-2,-3,-9,-8,-4,8,5,10],[-9,3,8,-2,2,10,8,-1,8,7,-10,2,-7],[4,10,-1,7,-8,7,10,5,7,6,6,-5,5]],[[5,9,-8,3,7,8,10,-8,5,-8,10,4,5],[-3,-2,-6,-6,10,-10,-3,-10,2,4,-9,-2,1],[-2,9,4,8,8,2,10,7,6,-8,8,4,-2],[3,10,5,1,-1,-3,1,-2,-7,-7,-10,-4,-7],[8,6,-6,5,7,10,-3,-2,-10,10,-8,-2,-1],[7,-3,-8,8,3,5,2,6,-9,-2,5,-4,4],[2,-6,-8,-8,7,-2,-10,-9,-2,8,-4,9,7],[-10,1,-4,7,5,3,8,9,-7,4,-7,-8,4],[-1,-9,-9,9,10,4,-1,5,-4,6,-3,-4,8],[3,4,9,2,-2,-5,-2,4,6,-5,-1,-6,-2],[-1,-5,-6,6,-8,7,-3,3,-2,4,-1,1,-3],[-4,-3,-7,-3,-3,3,-10,4,-2,-6,4,-2,-2],[-1,8,9,-3,10,-5,4,-3,7,7,1,-8,-6],[5,-5,10,1,-9,4,3,4,9,5,-5,5,6]],[[1,1,10,-5,4,-10,5,-5,3,-3,5,-7,1],[-5,-3,-2,-6,-4,-10,1,4,9,3,8,-4,-5],[2,9,3,10,6,-2,7,-1,5,-9,5,10,-3],[-6,-9,6,-4,-5,-1,-4,-6,-1,2,-10,-8,-8],[8,-3,-6,8,1,-7,10,-3,-6,10,-5,7,-7],[-6,7,-7,2,5,-8,-3,10,-3,1,-3,-5,1],[8,-9,6,-9,9,-5,8,9,7,-5,4,-7,-4],[-4,-5,-8,4,7,5,2,-4,5,8,-7,9,-4],[-10,10,3,-4,10,1,-9,5,-6,8,-3,-1,-10],[-8,2,2,2,7,-5,-4,-3,2,-3,8,3,9],[-10,6,1,-8,5,5,1,3,-3,2,1,-9,-10],[-4,-10,8,-3,1,2,7,9,-5,5,6,-4,-5],[-4,-9,-9,9,4,-3,-6,1,2,-5,1,1,9],[-8,-10,8,-7,-10,10,10,-9,-2,-5,-4,4,-10]],[[-1,-3,-10,3,-5,-4,10,1,-5,5,-3,-2,-2],[7,-6,-3,8,7,-8,-6,3,-5,5,-2,-6,6],[-6,8,-5,-3,2,5,-1,-5,-3,2,-4,9,-5],[-2,8,10,8,-1,3,-1,-2,-2,-1,4,2,1],[4,-7,-1,-8,7,-2,-6,-6,-8,5,-5,5,-5],[-1,-4,5,10,-5,-9,1,-2,2,7,10,-7,7],[1,-10,-10,5,1,10,-2,-4,9,-8,10,3,-8],[-5,6,-8,6,5,-9,10,10,3,6,-2,5,-7],[10,2,-1,-2,1,-5,-8,-2,1,-6,1,-5,-9],[1,-7,-8,-6,-4,7,5,-5,-6,-9,9,10,2],[-5,8,4,-7,-8,9,-5,-3,10,-2,-8,-10,-9],[-6,-4,-2,-5,-2,-2,1,6,-5,2,5,6,9],[-8,-9,-7,5,5,-10,-8,1,-3,-9,4,-2,-10],[10,-6,9,7,-8,7,4,1,-4,6,8,-8,2]],[[10,10,-1,-7,-9,-9,-6,-9,-10,-5,-6,-9,-4],[6,3,6,2,-3,-4,9,7,-8,-7,-6,-1,-10],[-7,3,-6,-3,-7,-2,3,-1,8,-2,8,-7,-8],[-3,-1,-10,-2,10,-2,-10,-4,-7,10,5,5,-10],[-9,9,-10,-8,5,-1,-1,-8,-8,1,1,-8,-5],[6,2,3,4,7,-4,7,3,8,-3,-6,-5,2],[4,-4,2,-8,-5,-6,-5,-3,9,10,-9,5,-5],[2,7,-3,-3,9,6,-2,10,-7,10,-6,1,-1],[2,-4,-9,6,-2,9,7,-3,-9,1,-4,4,2],[4,-6,4,3,7,-4,-1,4,-4,-2,-2,-3,-10],[-5,10,6,1,-1,2,-2,-7,3,-5,-10,8,5],[2,-6,4,7,-7,7,5,-2,3,5,2,-10,5],[-5,1,-2,7,-3,9,4,3,5,-1,-4,-3,-4],[4,-2,-3,1,-7,7,1,5,-2,-8,-10,4,2]]], dtype = "uint32")#candidate|544|(11, 14, 13)|const|uint32
var_545 = relay.var("var_545", dtype = "uint32", shape = (11, 14, 13))#candidate|545|(11, 14, 13)|var|uint32
bop_546 = relay.right_shift(const_544.astype('uint32'), relay.reshape(var_545.astype('uint32'), relay.shape_of(const_544))) # shape=(11, 14, 13)
bop_549 = relay.greater_equal(const_544.astype('bool'), relay.reshape(bop_546.astype('bool'), relay.shape_of(const_544))) # shape=(11, 14, 13)
bop_552 = relay.divide(bop_546.astype('float32'), relay.reshape(var_545.astype('float32'), relay.shape_of(bop_546))) # shape=(11, 14, 13)
uop_557 = relay.sin(bop_552.astype('float32')) # shape=(11, 14, 13)
uop_560 = relay.rsqrt(bop_552.astype('float32')) # shape=(11, 14, 13)
bop_562 = relay.subtract(uop_557.astype('float32'), relay.reshape(var_545.astype('float32'), relay.shape_of(uop_557))) # shape=(11, 14, 13)
uop_567 = relay.acos(bop_549.astype('float32')) # shape=(11, 14, 13)
uop_569 = relay.log(bop_549.astype('float64')) # shape=(11, 14, 13)
bop_571 = relay.power(uop_567.astype('float64'), relay.reshape(uop_569.astype('float64'), relay.shape_of(uop_567))) # shape=(11, 14, 13)
uop_577 = relay.tan(uop_560.astype('float32')) # shape=(11, 14, 13)
uop_582 = relay.exp(var_545.astype('float64')) # shape=(11, 14, 13)
bop_584 = relay.maximum(uop_557.astype('uint32'), relay.reshape(bop_562.astype('uint32'), relay.shape_of(uop_557))) # shape=(11, 14, 13)
const_590 = relay.const([[[-6.145053,9.365609,0.547121,6.828908,2.869787,-9.405270,8.590064,-9.353956,-2.146039,4.466951,-9.948340,8.332160,4.111066],[1.945396,8.453651,-2.219039,-3.932322,-8.710590,7.417736,8.452972,-6.914019,-0.774507,-8.722105,4.184432,2.565639,6.522889],[-4.191232,6.129011,-8.783193,6.668980,-4.856609,-0.260720,1.063071,-7.579917,-9.148970,-6.975797,0.515408,-7.740470,9.705453],[2.313867,4.899896,2.024804,0.877501,8.836141,1.145291,2.620642,2.789586,-2.750096,-5.485597,1.888795,5.596863,2.918035],[3.711271,-8.400724,-5.222155,6.845306,-1.905825,5.341346,-4.070310,-1.618502,-8.385891,6.007455,1.712342,-8.507354,-0.235887],[-9.618098,6.629020,2.840341,-2.824611,-6.042390,3.617116,-6.041828,-4.538907,1.733283,-2.532247,-0.043221,8.294649,-9.821336],[-3.207481,-8.485887,5.813163,-9.683545,-6.326112,3.545644,6.022963,9.989169,0.294514,4.600687,-6.328303,-3.702812,-2.604064],[0.177661,-8.402647,-0.533650,-9.479749,-8.121330,8.288381,5.216958,-8.098854,-9.466457,-9.183876,-2.306925,7.585711,2.894884],[-8.321336,9.488814,-5.613325,-3.613620,-0.340905,9.407470,1.308089,-2.371216,-6.458265,-5.844344,-8.203600,4.777526,3.670141],[2.722434,2.115678,-8.519720,-9.479729,-7.555473,-5.407604,-5.251764,6.792979,2.395879,7.285250,0.781881,8.058496,-2.993036],[8.361772,-7.244589,3.747452,6.102750,-2.748273,-9.961861,6.434557,-6.350503,3.055797,2.805663,-0.529217,4.504788,-7.553966],[-2.403793,-9.480469,-0.081261,-2.325438,-1.272066,-2.366937,-0.093099,7.181375,1.364116,5.395142,-9.816999,-9.041806,5.073421],[1.184082,-0.818718,7.143584,8.509403,0.076254,6.883777,7.091229,1.120204,-3.895493,-1.766089,-2.599233,-8.198478,2.560533],[-2.126271,-7.348592,9.042751,-4.418015,-4.200616,-4.364021,-0.922407,7.563221,2.240103,9.731519,-6.904068,-1.467298,-5.444546]],[[-2.917606,-5.944259,8.011067,-0.153140,-3.517530,7.304484,1.385243,-1.440594,-1.474752,-6.645424,9.344429,-3.144238,-1.048997],[-2.542095,9.776237,-4.927813,1.322178,-7.534210,8.390846,7.927388,-8.855513,6.971005,-8.412842,5.686123,-5.818649,-5.521131],[4.562550,-2.462977,4.950081,-3.394620,5.301292,-4.775076,9.361154,-3.497262,2.037946,-3.015528,-9.438696,-6.437480,4.498057],[2.076549,3.557408,6.784047,-3.654662,-0.960962,-5.202099,9.053957,3.602583,4.837993,-2.613516,-9.086382,5.115681,6.893741],[-6.274197,-8.130660,5.606242,6.386817,3.504522,4.177033,-5.600199,6.379219,-8.881081,4.811644,-4.284716,-9.977259,-1.016243],[0.465538,1.182227,7.918655,-8.069185,8.157102,-5.623817,-7.484648,-2.476535,2.144101,6.992298,0.458879,8.405668,-3.960811],[2.965034,-5.164777,5.309351,8.812150,-7.629592,-3.044805,-7.921233,-1.134274,-4.130004,-5.477619,-8.591159,4.260915,-8.272716],[0.431360,3.161170,-3.974503,-0.251055,-2.529919,-2.494337,-8.215865,5.653536,6.599946,3.854798,-7.749541,-9.770627,-0.858603],[-7.978115,-3.459495,-9.732044,5.076961,2.230560,3.608092,-1.047407,2.703052,9.606699,-6.150512,-2.496007,8.300508,7.278181],[7.108129,-6.820498,0.787701,-1.823160,-2.686998,8.598556,-9.178749,7.978174,9.818637,-6.398209,-5.399718,-9.174461,-9.894211],[-1.945932,-5.325459,4.121379,0.317865,9.635849,-4.277648,-5.226165,0.888751,-3.681078,7.991399,2.576583,-6.256800,2.197800],[7.875589,7.932040,8.657979,8.089500,7.069337,2.569971,8.896418,6.241311,-8.065222,-3.968882,-0.903843,-3.326745,-1.761565],[-5.540346,2.048913,0.520453,9.494703,-6.656206,-2.563999,4.186002,-8.161504,-4.911050,0.339570,2.845946,5.119698,8.217654],[8.636966,6.795270,5.443430,-3.634130,-6.643495,-2.922753,-4.041304,-9.186112,-1.170320,-7.356844,-1.458175,-3.753544,7.913752]],[[-9.416899,5.249432,-9.308949,8.500775,-6.454439,9.018635,3.974081,-4.787871,0.099096,-7.710775,2.501270,8.732105,1.951979],[-2.746648,-5.451496,2.399683,3.491950,-7.568091,6.370886,7.242650,6.808781,-6.943398,-3.413700,-2.856854,0.277442,7.565589],[7.305243,9.660743,5.634879,8.579170,8.750043,-5.274911,6.596513,-8.192616,-8.521159,1.532771,-4.419624,-7.362416,-7.505479],[-8.131734,3.194085,7.152995,8.627815,0.649226,-5.733852,-4.786050,7.681993,-1.534647,1.303138,-6.962586,4.304791,1.651689],[6.976123,-9.902709,-6.932292,-8.202409,5.203653,5.081995,-4.314718,2.687311,5.411284,-8.883139,-3.338333,4.894868,5.349990],[6.563888,9.085761,-3.176484,-9.255570,4.886641,9.715132,-1.282901,4.992746,2.319973,-9.244340,-2.590801,-4.237105,-8.538297],[-0.418605,5.812993,3.471969,-5.157983,0.089085,-1.024212,0.325790,1.566939,9.261199,6.456407,0.032468,-7.482373,8.724733],[2.190026,9.001568,-6.251835,0.480271,5.406726,-5.916378,-0.224285,-5.938971,9.303647,-3.746427,-1.795537,-5.713785,-3.498210],[2.808489,1.007398,2.308671,-4.353836,3.006918,-0.885933,-0.428824,0.596588,8.186498,0.549211,-0.814141,-0.623462,7.885119],[-5.039893,-6.293320,9.879928,0.681295,6.720690,2.005637,4.072527,-8.020898,7.156889,4.326416,-2.725516,-3.147262,-9.052233],[-5.408442,1.478346,-8.379746,-6.705956,-5.766433,-7.760083,-9.009573,4.380690,8.679974,-0.194830,-2.685821,1.404224,9.052337],[6.955427,4.628841,-0.802821,-5.205537,1.720853,-4.547240,2.626622,2.669422,-5.268101,5.965927,5.141122,8.766183,7.582596],[9.153716,5.093990,-5.808934,4.268475,0.666105,2.005378,7.752110,-1.909029,-3.182028,7.153544,9.737146,-4.920876,3.942423],[3.650531,2.670173,-8.503064,8.248166,6.364095,-7.243156,4.103209,5.339356,7.991626,7.845038,-5.077081,-8.808350,2.097691]],[[5.041587,-1.365719,0.172761,5.960532,-2.754649,-8.112278,-0.423764,0.002929,-9.517817,-2.703655,-6.529619,-8.291297,7.829188],[1.403473,-2.036499,0.368857,-8.657539,-0.803843,2.639565,1.164632,-9.651974,-7.090061,7.205250,9.867985,-7.755210,7.325956],[-4.039858,8.873602,-3.031006,2.176584,4.807166,9.930831,-5.585228,-5.601335,6.206732,-1.206237,3.722179,-4.571942,-8.629960],[-9.123915,6.914339,1.173471,8.023554,3.445353,-9.868520,7.960522,0.314916,8.782845,-0.031509,-5.161823,8.653821,1.449644],[6.954759,-3.285839,-2.532939,-0.138404,-7.003045,1.127221,-8.368585,-1.271433,6.095256,0.827018,-9.770176,-4.913537,-6.076904],[-4.363956,5.448289,-8.845831,-9.577228,5.798905,5.955302,-1.060705,5.049240,7.965359,2.033038,-4.069510,6.266809,3.187578],[4.392515,6.012174,-4.895515,5.114527,9.055458,-8.845010,-1.568518,-7.714991,-0.365155,-9.236080,-6.596182,8.180639,9.531175],[-1.657010,7.217194,-4.769850,-4.977741,-4.957407,5.469507,-9.901088,1.298962,-0.619201,-2.747038,0.792986,-6.571191,1.927235],[-1.481364,-6.222034,-0.689171,-2.990999,-8.372886,-6.260530,-8.769005,-1.642331,5.498740,9.003197,3.207689,-1.040402,-7.269142],[-1.561621,9.695374,-9.215853,1.279811,6.495057,1.222407,7.318186,-4.644934,-2.221715,4.822983,1.152721,-3.395281,6.923952],[-0.690181,3.668901,-3.555142,0.703950,3.606295,3.198728,-6.110636,4.825722,-5.696694,-4.294199,-3.317057,-6.048558,7.405986],[-1.470202,-9.139666,8.552028,-0.057542,3.157258,7.880763,-2.851978,-5.579721,4.074709,9.221933,2.837537,0.132965,-8.861910],[-5.752784,3.904336,-2.441171,0.434059,-9.638628,-9.472939,-2.751400,3.755775,-6.726633,-9.688230,1.751503,-7.076555,-5.178617],[-1.706112,9.013680,-9.762591,3.885459,1.865548,8.826269,-2.092999,9.039165,-2.956615,5.869445,4.640137,-5.649839,-1.656245]],[[2.486573,-5.800787,-5.016176,-0.269766,2.481146,-4.756862,-7.178295,5.788384,-2.539216,1.626643,-7.800880,6.222408,1.838388],[3.455419,8.852705,3.887677,-7.918163,-2.250814,4.398477,-3.042300,9.042185,7.268483,-1.708322,-4.771967,4.082344,6.796401],[1.278411,0.359521,7.039358,-9.533017,-4.395676,-2.328382,0.061995,-7.032154,0.653797,-5.041463,6.984482,-6.604846,-2.666903],[-7.593788,6.034077,0.677058,4.525783,0.350412,5.044930,4.630375,-8.630318,9.083325,-6.640343,-0.876865,-6.861816,4.773437],[-1.834883,-8.192944,-4.027246,-0.273417,-2.986208,-1.436574,-2.414475,-8.988837,-9.453537,-0.801405,1.297342,-7.081021,2.142974],[9.633670,2.642822,1.266902,3.738290,-5.468578,-4.397153,9.950292,-2.277961,-9.168146,8.465501,9.914243,-1.768811,-2.200807],[3.459728,-2.801846,3.783915,-4.509536,-3.589318,0.739732,2.723165,6.953081,-9.535499,-3.452600,-2.799849,-6.500660,-1.357431],[2.116174,8.368272,-5.844141,7.400827,7.086566,5.784465,8.664339,1.203088,1.987797,2.287040,-5.631182,4.187734,-3.465961],[-5.971602,8.728695,-7.213816,4.316818,9.060688,-8.391156,-6.337284,9.429282,-2.335930,3.274811,7.087614,-0.759787,9.869179],[9.799912,-7.499145,2.238277,-7.066407,7.917398,6.458742,-5.567830,-7.599187,0.611670,-8.791724,-0.781988,-8.162520,-6.087042],[-8.134594,-1.608824,8.769164,-9.622072,0.428791,-7.472043,9.217573,9.729341,4.536111,8.279000,-8.959119,-7.069613,3.463206],[8.541713,6.352976,-1.877866,8.348501,4.553785,-7.041500,-8.273164,6.637681,5.542994,-6.294050,-7.199064,-4.506078,8.330344],[-3.596935,1.868734,8.557095,5.395085,-5.223566,-1.060087,-5.041744,-1.356446,-7.321053,8.828149,-7.796522,-0.541747,8.385986],[-2.140608,-4.975991,-9.810231,5.209597,-0.406163,8.860513,8.480437,9.024285,2.762652,-9.267253,-5.884239,-8.586702,-7.230499]],[[-5.831581,8.683240,5.049442,5.494327,9.231685,7.186961,9.138112,5.385370,3.227563,-0.161564,4.307580,1.409146,-5.853346],[-9.459584,2.316683,7.143188,4.689628,4.273475,-2.622611,-5.728093,-9.645199,-7.155543,8.809150,-1.918820,4.599244,-9.661945],[-8.441954,-2.823964,4.065682,7.297594,5.195997,1.444935,9.180511,-3.815457,1.780963,-6.098851,-5.939109,-1.430615,-0.950336],[-6.471462,1.097130,-3.691131,-9.282925,-2.500830,2.832690,-3.526717,-7.192398,2.901063,-9.089005,4.743067,-4.322055,-7.153757],[5.090306,4.593498,2.706983,6.670683,-1.025629,1.738959,-4.183204,-5.960045,1.044962,-0.251224,-0.918645,-7.256369,2.940907],[1.081617,1.793615,-8.154016,-3.808318,-7.832064,-9.593878,-7.834095,-7.891980,9.704333,-6.855324,-6.516488,8.840201,5.401502],[1.864149,-9.124360,-9.342482,-5.606510,0.426309,-6.920711,-8.127813,-5.162966,-6.074111,7.666554,-3.046829,1.747299,2.604659],[-9.089463,-0.253728,-4.610322,2.147292,7.090372,-2.695376,5.749007,-7.999410,6.086528,6.839408,0.191414,7.208232,-5.601120],[-6.132744,-2.046986,4.342961,-7.677430,-0.945411,6.805121,3.482446,4.722962,7.227214,-8.243285,-0.779960,-3.718053,3.030903],[2.298612,5.366346,-4.016867,5.383032,7.109535,1.433998,0.793686,-2.889559,-3.534091,6.060502,-8.164043,-8.172164,-5.433897],[-7.028142,-4.135922,7.189174,-4.649276,9.053604,-3.037937,3.882149,-0.461113,1.829465,8.790934,-0.126777,2.689585,6.051428],[-0.517970,-8.188669,0.033564,-9.190570,-8.229729,2.190138,-0.522838,4.922963,-6.132572,-9.365961,-1.854038,-6.008109,9.314218],[4.364922,-1.178693,-8.759939,-7.821189,9.813794,-7.430282,-3.539509,3.180406,5.246161,-0.203070,-5.691082,-2.899261,9.786318],[-2.023828,-5.597117,3.501660,7.709472,-2.584292,-8.256035,-5.492117,-9.437814,2.294679,-1.880598,0.590882,-4.937544,-5.226677]],[[-8.945393,4.887959,3.053728,3.834919,-6.010110,0.661902,-6.196587,7.159295,5.129676,-2.457405,-3.038100,-4.513125,2.680308],[1.947908,1.114470,-4.285273,-2.455825,-1.200817,-1.769625,6.855682,-8.102130,-7.398910,-1.505334,8.019501,3.849267,-0.350739],[-4.387288,-5.253964,-6.695050,8.219915,-1.339242,8.084785,-8.238233,-1.084307,2.712345,7.860611,3.182291,-4.242986,0.979426],[8.147576,7.339428,9.280158,-3.740613,2.297686,1.948656,-9.675079,6.791242,-7.607609,9.071725,-2.646366,-9.468617,-5.128227],[7.017605,1.922329,6.832648,5.526390,-9.858736,-1.790461,5.640559,0.951617,-9.545724,-2.239204,7.196690,2.363888,7.174754],[7.606113,5.310160,9.821516,-1.229002,-1.050991,0.880594,5.878292,-0.461643,-5.400298,1.482879,-5.786131,-8.947131,0.416964],[2.469044,-9.929626,-3.799689,-3.287982,-0.777680,-3.787227,-0.198200,-8.493997,1.363330,-7.065865,9.458374,4.682193,-8.637405],[-1.402094,-0.371939,8.247370,9.963555,-1.695073,8.378358,4.979747,7.027333,7.963587,-1.916289,5.957846,9.530505,3.058914],[6.644599,4.618069,-9.201299,-6.277371,-8.519929,-8.090908,4.677176,6.904839,2.466965,-4.033912,4.673477,3.449111,2.798156],[4.625861,-6.012156,2.758409,0.788771,0.488802,6.665659,-5.440417,2.694693,0.371855,6.542026,-9.538812,8.887181,8.028757],[-0.876823,6.709847,-0.072239,-4.995705,3.809984,-0.373970,-7.815369,3.578591,0.641203,6.334374,-2.007195,3.952810,-8.970327],[4.758424,2.466984,-3.419412,9.633604,8.581171,-9.553653,-8.068031,-9.415559,-8.946119,-4.562939,-0.376907,2.751003,6.483531],[-4.914455,0.948824,-7.040641,-5.503782,-3.396781,6.358522,-0.542660,1.495253,-5.689891,4.637068,-4.092114,6.527064,2.285699],[-8.769666,-2.723832,-8.380565,-1.465148,4.667634,-2.035700,-5.261553,-0.747301,-8.540398,4.556586,3.908404,9.918719,-3.115872]],[[-0.128442,5.743466,1.594602,-4.889910,3.376250,-1.552708,-5.302168,-7.697653,9.664599,3.144932,-6.999132,0.795652,6.138476],[6.918883,2.190378,-9.886062,9.439029,0.426066,3.011719,-0.528478,-4.234331,7.252444,-7.906260,-3.407834,-1.996639,1.794243],[6.451363,-9.764363,9.697464,-3.677015,4.305154,-4.561079,-5.006551,3.725560,9.740784,-4.700649,-7.412457,1.651450,8.539142],[4.260348,1.901560,-5.622736,7.885964,-7.106575,-3.511277,9.790877,4.051153,-3.776619,2.871248,-0.328987,-9.789164,-4.983054],[-4.025922,1.519899,4.318226,5.898454,-5.602333,-5.194150,0.486998,-6.820483,9.256605,6.914546,-3.187750,-6.370995,-1.767672],[-8.764347,6.898986,4.309129,-4.362622,4.015835,7.945306,3.889250,-5.504152,7.889164,-7.572461,2.372166,9.014360,-1.400161],[-8.196650,9.438496,7.176138,2.599561,4.499326,-0.507682,-2.085887,7.894403,3.163390,5.479769,-7.301600,1.213546,-2.820073],[-7.453252,2.290793,4.176104,0.097071,-9.085697,9.737841,1.559556,9.722998,-4.228340,0.484384,1.333905,-9.604924,-3.629172],[-7.621125,3.019162,-8.538022,-8.933074,9.148378,-2.099153,7.877788,5.340231,-0.951647,2.465229,1.320921,7.539511,2.874149],[8.299825,0.379133,1.997862,4.016772,8.433632,-5.617451,6.703956,0.162717,4.668325,9.196730,7.712904,8.517926,1.670865],[9.675724,-6.380523,5.692369,-4.975345,7.434449,-2.340429,8.556290,1.976707,-2.301632,3.424745,-0.131808,-9.535083,1.282423],[0.274075,-1.260135,0.460171,-5.648902,-5.807316,7.567009,-6.114118,-6.365929,6.711171,-1.029702,-0.879260,-8.269687,6.886874],[-5.001534,-8.468175,1.166369,-1.227905,-3.236135,3.927942,5.831960,-6.106640,2.101980,-1.021902,-7.786760,1.394671,-9.536384],[6.325316,4.927682,2.684602,-9.383381,9.101235,3.482011,2.049796,-5.153875,-2.244190,-7.603704,-3.464321,-7.687376,-1.258552]],[[6.703728,8.666000,5.474869,-5.700685,5.359992,-6.626149,0.519197,-4.575941,1.554918,-9.233919,-5.661051,-8.283054,0.201597],[-1.055254,4.374001,-4.108201,-4.581517,-8.742592,-6.146564,-5.952658,-4.345711,7.566180,2.311847,-6.315984,5.239319,-9.511812],[-0.048272,2.710787,1.060761,2.101779,-8.979371,2.305809,3.542696,-7.329637,-0.517117,-2.817454,0.240710,-0.933208,7.818874],[2.677891,5.237752,-5.417371,-5.360078,-5.494894,-6.009353,-6.193227,3.935370,5.412062,7.585803,9.620428,9.625881,-7.424655],[4.423996,2.074875,-2.321719,-9.028787,-5.227976,7.798811,-9.048761,-1.574348,9.016643,0.447421,5.113518,7.993689,-4.631163],[-6.567178,-9.677300,-1.326646,-5.989882,-4.704791,-2.805385,-6.612706,-1.422380,-9.338080,4.659302,9.017552,3.784651,-1.846493],[8.247378,-7.084795,-3.626178,0.354891,3.053691,-8.431837,2.776340,-4.046739,7.575782,-5.312580,-7.015409,-6.805784,-3.039940],[-5.774946,-3.015552,-4.270696,6.656235,4.612028,7.306166,-2.415897,-8.779822,-5.415480,-2.684801,-8.633893,-4.175536,0.774944],[6.885816,-5.912517,8.192496,0.889422,9.195859,2.805463,-5.868939,-8.355010,-3.217238,4.347145,-6.259286,-0.113988,-0.351456],[-1.611110,-8.712061,-8.246965,-8.625880,2.709609,9.572951,6.179939,-2.995214,0.702700,-0.707240,-4.009125,3.648744,8.303986],[-6.455678,-7.365715,-3.642821,1.626093,-5.428012,-1.293166,-4.205730,-4.695469,3.061801,5.581228,-5.583947,-5.383209,-7.324968],[8.273132,-6.291584,3.963367,7.694221,-6.214064,2.336989,-4.062098,1.282670,-0.347624,-6.760794,3.586713,-1.620015,9.659306],[-4.639556,3.362403,-3.719317,-0.918009,-9.798892,5.072244,7.409293,1.692922,8.497050,-7.805351,5.217814,5.428430,5.046469],[2.314094,0.592118,-1.692895,-9.325479,-7.460692,6.861486,-0.727208,9.867002,7.481246,6.704330,3.481116,-8.621440,4.807343]],[[-7.373878,2.612348,5.350973,0.531010,2.110120,-9.160084,-6.269931,-3.148891,3.235305,8.578291,8.639858,2.891045,1.906642],[8.429599,-1.326311,1.847011,1.594248,-7.472530,6.881105,9.551629,-6.563872,2.941599,3.859839,-7.928297,-9.554859,7.217500],[-6.773407,-9.398959,2.507571,6.992224,-5.366435,-9.907461,5.693032,-7.728834,-5.029994,8.381627,-3.508601,0.291243,-0.656370],[-4.056527,5.902589,-2.386657,0.276639,-4.901041,1.020534,-4.393169,-9.020315,4.399060,-6.426568,8.294013,-5.471349,0.510776],[6.844186,-2.783363,-1.351471,-0.175009,-6.027714,9.101698,1.932404,-0.107419,0.677471,-1.685958,9.780829,9.761870,8.441670],[-6.241556,0.781653,-7.234398,5.149912,-0.165508,-0.360740,8.032007,-2.384771,-4.481174,-1.551678,-2.584822,3.946164,-6.295093],[-4.645666,-5.208409,-1.588091,9.798791,-5.093277,-3.804805,-9.567178,5.200139,0.940222,4.110234,4.064218,0.304350,-7.213320],[3.831296,-6.519633,-7.933073,1.006046,-8.202352,-3.570758,4.480001,-1.873881,-6.997566,0.333913,9.965779,-2.301013,-6.595800],[-7.605778,-8.957148,1.033148,-1.207163,6.278491,-9.560266,-0.190083,-5.758851,6.918805,8.909411,8.364012,-5.915088,-4.533871],[5.580396,2.873013,-6.068238,9.141820,-0.464718,5.762288,-4.558258,-2.006429,5.661536,2.281960,7.079811,-4.894108,-5.339863],[-4.435339,-4.745791,-3.931771,2.065001,0.783287,8.604172,-5.657040,1.978511,1.526921,-6.111700,-6.475613,-1.111300,4.519432],[-7.520268,-1.518296,8.373638,-3.019278,0.791970,8.599755,-6.491139,2.662388,1.171092,5.707344,8.793841,-1.836451,-9.116474],[5.905859,1.753736,-8.371306,-1.131314,-4.299062,2.994219,3.221315,-0.161969,-1.750604,7.398235,-7.202375,2.766958,3.461930],[-8.530395,3.518842,5.596772,6.309894,0.531594,-2.055829,7.551861,-8.101581,4.862871,9.442882,-6.287526,-3.764451,2.176894]],[[5.095089,9.795476,0.989765,-5.115439,-9.566340,-8.010352,-6.332760,-3.631067,4.456854,-7.581378,-4.997801,-4.097254,2.933501],[3.275089,-3.795979,-1.001680,-4.806287,5.930000,-6.600162,-5.623282,9.968432,-1.693298,-7.154492,-6.780359,-6.198321,4.937630],[7.749379,5.619884,7.454088,-4.388823,0.742110,8.886449,-7.712508,-5.011831,-9.985528,-9.377937,1.793684,-6.560440,9.167613],[0.462539,6.643223,-0.099550,-6.225320,7.686755,3.178079,-1.105716,-7.434648,-3.471399,-9.534433,4.967207,5.005711,-0.889300],[0.610777,-0.277155,0.811671,-4.266272,2.117600,9.963555,-7.998653,6.226732,7.783816,9.591476,4.977851,-6.500267,-1.712819],[-5.337225,6.730081,-2.943093,3.508929,9.199978,-6.126738,5.236296,5.185927,-7.751775,7.055334,0.425992,1.392338,9.421766],[-7.228169,-1.673479,8.322959,-0.281704,5.891103,3.705055,9.841119,5.818761,7.982582,9.437351,-8.460270,-3.477806,4.857479],[-6.702938,-3.016313,0.244754,5.797805,7.731446,1.502792,-9.553265,-8.040627,7.026686,-0.131686,-5.976291,-0.305320,-9.473540],[4.269513,-2.077312,-5.027575,0.002082,-9.078470,9.189956,-1.802584,5.345838,-1.564519,-9.404011,5.147030,4.817218,-2.432983],[9.700352,4.322872,3.429808,4.843555,-3.697015,-5.115297,-9.567870,-2.486923,0.312032,7.797393,-2.210449,-5.988322,-4.871495],[4.519937,9.775302,7.643674,-8.757203,9.025539,0.966619,9.921814,-8.973283,9.715011,5.219446,6.789399,-9.882619,2.573538],[-7.297608,4.632186,1.617575,-6.948047,6.819740,-4.791871,6.829015,1.888506,-4.233665,-4.321461,9.632930,7.215789,5.471622],[8.251652,4.395542,2.700412,-1.402910,5.341406,2.848806,-4.180960,0.734027,-1.583022,-6.308975,-1.175479,-4.039381,-5.676528],[-8.400666,-2.873077,-0.771371,-0.387919,-4.359458,4.140247,-3.935653,5.417338,4.006542,-3.766552,-1.347010,2.482419,2.520965]]], dtype = "float32")#candidate|590|(11, 14, 13)|const|float32
bop_591 = relay.floor_mod(uop_577.astype('float32'), relay.reshape(const_590.astype('float32'), relay.shape_of(uop_577))) # shape=(11, 14, 13)
func_62_call = mod.get_global_var('func_62')
func_65_call = mutated_mod.get_global_var('func_65')
var_595 = relay.var("var_595", dtype = "uint64", shape = (30,))#candidate|595|(30,)|var|uint64
call_594 = relay.TupleGetItem(func_62_call(relay.reshape(var_595.astype('uint64'), [10, 3])), 0)
call_596 = relay.TupleGetItem(func_65_call(relay.reshape(var_595.astype('uint64'), [10, 3])), 0)
func_190_call = mod.get_global_var('func_190')
func_194_call = mutated_mod.get_global_var('func_194')
const_599 = relay.const(0.660803, dtype = "float32")#candidate|599|()|const|float32
const_600 = relay.const([-1.522136,4.655798,-4.798592,2.891326,-9.696297,-7.241845], dtype = "float32")#candidate|600|(6,)|const|float32
call_598 = relay.TupleGetItem(func_190_call(relay.reshape(const_599.astype('float32'), []), relay.reshape(const_600.astype('float32'), [1, 6]), ), 2)
call_601 = relay.TupleGetItem(func_194_call(relay.reshape(const_599.astype('float32'), []), relay.reshape(const_600.astype('float32'), [1, 6]), ), 2)
uop_604 = relay.log(uop_557.astype('float64')) # shape=(11, 14, 13)
bop_608 = relay.greater(bop_562.astype('bool'), relay.reshape(bop_571.astype('bool'), relay.shape_of(bop_562))) # shape=(11, 14, 13)
output = relay.Tuple([uop_582,bop_584,bop_591,call_594,var_595,call_598,const_599,const_600,uop_604,bop_608,])
output2 = relay.Tuple([uop_582,bop_584,bop_591,call_596,var_595,call_601,const_599,const_600,uop_604,bop_608,])
func_615 = relay.Function([var_545,var_595,], output)
mod['func_615'] = func_615
mod = relay.transform.InferType()(mod)
mutated_mod['func_615'] = func_615
mutated_mod = relay.transform.InferType()(mutated_mod)
func_615_call = mutated_mod.get_global_var('func_615')
var_617 = relay.var("var_617", dtype = "uint32", shape = (11, 14, 13))#candidate|617|(11, 14, 13)|var|uint32
var_618 = relay.var("var_618", dtype = "uint64", shape = (30,))#candidate|618|(30,)|var|uint64
call_616 = func_615_call(var_617,var_618,)
output = call_616
func_619 = relay.Function([var_617,var_618,], output)
mutated_mod['func_619'] = func_619
mutated_mod = relay.transform.InferType()(mutated_mod)
var_627 = relay.var("var_627", dtype = "float64", shape = (4, 15, 7))#candidate|627|(4, 15, 7)|var|float64
uop_628 = relay.sigmoid(var_627.astype('float64')) # shape=(4, 15, 7)
bop_632 = relay.left_shift(uop_628.astype('int8'), relay.reshape(var_627.astype('int8'), relay.shape_of(uop_628))) # shape=(4, 15, 7)
uop_635 = relay.acosh(bop_632.astype('float64')) # shape=(4, 15, 7)
output = relay.Tuple([uop_635,])
output2 = relay.Tuple([uop_635,])
func_639 = relay.Function([var_627,], output)
mod['func_639'] = func_639
mod = relay.transform.InferType()(mod)
var_640 = relay.var("var_640", dtype = "float64", shape = (4, 15, 7))#candidate|640|(4, 15, 7)|var|float64
output = func_639(var_640)
func_641 = relay.Function([var_640], output)
mutated_mod['func_641'] = func_641
mutated_mod = relay.transform.InferType()(mutated_mod)
var_656 = relay.var("var_656", dtype = "float32", shape = (8, 13, 11))#candidate|656|(8, 13, 11)|var|float32
uop_657 = relay.erf(var_656.astype('float32')) # shape=(8, 13, 11)
bop_659 = relay.bitwise_or(uop_657.astype('int8'), relay.reshape(var_656.astype('int8'), relay.shape_of(uop_657))) # shape=(8, 13, 11)
uop_664 = relay.cosh(bop_659.astype('float32')) # shape=(8, 13, 11)
var_667 = relay.var("var_667", dtype = "float32", shape = (8, 13, 11))#candidate|667|(8, 13, 11)|var|float32
bop_668 = relay.greater(uop_664.astype('bool'), relay.reshape(var_667.astype('bool'), relay.shape_of(uop_664))) # shape=(8, 13, 11)
bop_672 = relay.minimum(bop_668.astype('uint32'), relay.reshape(uop_664.astype('uint32'), relay.shape_of(bop_668))) # shape=(8, 13, 11)
uop_676 = relay.asinh(bop_672.astype('float32')) # shape=(8, 13, 11)
uop_678 = relay.log(bop_668.astype('float32')) # shape=(8, 13, 11)
uop_682 = relay.sin(uop_678.astype('float32')) # shape=(8, 13, 11)
bop_685 = relay.multiply(bop_659.astype('int8'), relay.reshape(var_667.astype('int8'), relay.shape_of(bop_659))) # shape=(8, 13, 11)
const_688 = relay.const([[[0.217682,1.832657,9.022048,4.715287,3.107469,-0.592954,0.897576,8.749038,7.059793,9.417169,-6.853464],[9.878892,-3.379147,-2.258003,-3.823013,0.753217,9.032249,-0.184470,-7.380574,4.694063,7.839431,-3.025946],[-7.707201,5.899379,0.102060,-1.028060,7.774292,-6.266166,-2.026782,3.843192,4.074731,-9.343309,1.044035],[5.302947,-1.835562,-3.371204,-9.572089,-3.553789,-2.160722,-2.936064,-9.941261,-0.675484,-4.194902,-3.347181],[8.845102,-7.029940,-5.971063,4.321625,6.043851,-4.490663,6.012202,8.600305,2.574123,-1.159099,2.009793],[5.354921,5.887774,4.224368,-9.808838,-7.196447,5.805965,-9.148448,6.557435,-0.784651,-1.280064,7.907545],[3.048203,3.779463,-2.590088,-7.782610,-8.208165,-7.724301,7.205186,-2.409331,7.573913,3.006111,2.366492],[-1.084126,9.842721,2.776411,3.172236,9.345905,2.927101,3.538969,9.525140,9.763230,3.243332,5.835286],[2.089151,7.810053,2.930396,6.983386,-3.380152,5.794911,-8.504544,-4.944208,5.113840,-7.608188,-2.573955],[8.659883,-5.437874,-6.364297,-6.027445,-0.508145,4.186488,-2.797261,9.775755,6.852977,9.003992,2.134770],[5.276757,-3.985984,-0.838516,5.404545,-1.311911,-5.120949,1.562343,5.411162,-1.015227,3.623499,4.331508],[-5.198223,3.807498,-3.575112,4.593244,7.967047,6.750922,6.737519,-3.902571,2.879817,6.753422,7.563531],[-8.946428,-7.708207,0.826828,-8.151153,2.349448,2.609754,-6.683440,-4.590919,9.590246,-0.606574,6.706472]],[[-0.197720,-8.655564,-2.092649,9.542271,-1.504821,-7.868829,8.286405,-4.105454,-6.373984,-2.587434,-3.837461],[-3.123340,-6.197231,-2.418170,-1.698700,9.169059,-7.383305,-1.717388,9.701303,-8.018250,1.373030,4.793696],[-9.571917,-2.985392,8.534194,-6.496138,2.026052,-6.613451,1.142657,-8.340263,-9.990802,-8.918678,1.011973],[-8.765917,2.551424,-5.233074,7.032614,8.697625,7.242517,3.977431,-9.313530,4.965831,-1.618351,-8.353011],[-2.007179,3.704156,-1.701378,1.294019,4.229392,-6.326212,5.177112,7.160405,-0.345011,-1.089762,4.212748],[7.235040,-2.905231,7.355949,-1.948685,-5.894263,2.493497,-1.659155,8.765661,9.914864,1.470799,4.582520],[1.841253,5.726086,7.789377,4.072191,2.681281,1.574066,9.065390,-7.291911,-1.927343,7.656736,4.512296],[8.178775,7.832418,8.387682,-3.654389,8.437657,1.494017,-5.632277,1.371914,0.233653,-6.847385,-4.242431],[-4.584970,8.138282,6.803685,-0.829656,-5.640432,6.285621,-0.245312,-4.369030,-4.156346,-2.342522,9.710205],[-1.324676,2.323129,5.129676,-3.207413,7.905190,8.125852,4.900756,6.553610,6.517099,7.300338,-2.847532],[2.683205,2.341956,-3.349714,3.429972,-2.608771,-1.745840,-0.770799,3.504052,6.318345,9.094878,1.554798],[4.195514,-4.186964,1.844275,-2.288444,8.321207,-7.715617,6.057348,0.645356,-2.316786,-1.132877,5.126006],[8.970801,-8.237800,5.542406,7.466614,-5.003152,-2.219559,4.338867,0.763923,9.431592,9.594509,4.490707]],[[-5.255387,3.080188,-3.925912,-9.560598,-5.266741,-3.724828,-4.620742,-6.605556,-4.118757,2.624278,3.332976],[-4.900605,-8.308324,-1.134534,-1.536603,-3.859294,-5.695239,7.427320,0.929782,-5.309631,8.166361,-4.844761],[-2.295038,6.006904,-5.562389,4.508020,0.278180,-6.261390,9.305875,7.595946,-0.635463,8.100353,-7.493169],[4.111903,-0.328440,7.190512,-3.036897,3.411067,3.092474,5.732556,2.994447,0.658204,9.038848,-4.826497],[-8.708090,-8.810727,5.430585,6.395379,-9.682945,-4.864947,-8.431252,2.661807,0.016877,-6.098209,8.120840],[-9.890390,-2.435156,-0.206947,9.799248,3.948274,-3.290566,-4.448268,4.493491,7.596269,-0.551750,-9.118073],[-2.426058,7.133480,-4.134289,-7.102805,-2.051541,2.969755,-8.953600,-2.792750,-3.739672,-2.195978,9.095959],[6.675975,-3.769130,-3.907948,-7.539614,-5.734303,-3.606073,-1.709515,1.636542,5.049575,-9.195354,-5.883063],[-8.066368,-3.839972,-4.442865,5.655521,-6.787947,-3.074191,-1.499436,5.551689,5.624278,2.121557,-5.292300],[8.467181,-4.259925,7.456840,1.176943,6.859486,-7.528684,-4.343045,-5.594213,-3.463112,-8.842683,-5.839632],[9.696434,8.003558,1.139440,0.962377,8.854291,0.391127,4.244139,-1.630273,-7.607428,-7.130084,3.241378],[-9.430192,-3.102981,-1.021912,8.710918,8.288578,9.280524,-5.293096,2.539309,-5.495641,-9.527513,-4.265869],[-1.679117,-4.237162,4.997210,0.414168,3.846865,-4.818386,-3.839406,8.359277,6.235622,-4.657239,1.786416]],[[5.626823,-8.203054,3.372143,7.464615,-0.104294,-4.226643,2.038562,3.416452,0.861384,-9.223537,1.855018],[-4.309439,-6.354387,3.546110,-8.258607,7.317500,-4.970074,6.521495,-6.972643,5.722341,-9.379234,9.946442],[8.426813,-1.274420,-7.288777,1.573226,4.509597,-0.880570,2.570710,9.444480,-3.383927,-5.233038,-4.743568],[0.146670,-2.328782,-2.431746,-3.272145,-4.988664,-5.541301,-7.558904,6.824598,-0.668794,-3.167483,0.667976],[2.110745,1.379676,1.058911,-8.692065,-5.510312,9.594665,-5.085437,0.562222,0.595755,9.472100,1.590561],[-9.746606,4.695248,7.791273,4.200242,3.302598,6.339398,9.578580,-4.944687,7.351645,9.938741,-7.562091],[-2.765258,-5.803404,5.999916,5.690498,-1.356948,-5.725784,4.686988,6.396484,-5.786473,6.550051,6.362039],[-4.118340,-2.601807,-7.906074,-5.657768,6.511744,4.507295,2.912169,6.222475,7.763339,-4.809879,5.609198],[5.786904,2.127301,-6.881594,-0.065823,0.027700,2.519369,-2.684395,7.860116,-1.885843,8.272988,-9.623339],[5.426886,2.120592,8.558104,-4.919129,4.119259,0.700709,0.590205,5.423826,-1.985046,-6.458410,-3.025361],[6.356481,-8.850290,5.257664,8.465978,-6.620727,5.088644,1.896816,4.193898,-4.595901,-5.302623,2.731648],[8.300550,6.944139,0.190522,4.962467,-8.209926,2.147613,8.249520,-2.578515,0.096054,9.196279,-4.572222],[2.182713,-2.681522,3.162449,-0.882663,-6.888267,8.600028,1.807596,8.610812,4.993070,2.296583,-3.805226]],[[2.002598,-1.449987,7.424486,5.605165,-6.367768,-8.889190,-7.903834,6.740676,-5.008685,-7.374934,2.078666],[5.398845,9.956692,-4.874863,8.582004,5.276697,9.000099,7.506197,0.429471,6.437517,-1.152983,-8.836102],[1.371967,8.910459,9.831870,7.834370,9.034313,0.605711,1.312563,7.762830,6.880205,5.087141,-6.199158],[6.969856,-2.143962,-9.361523,-3.764969,7.469872,-2.545967,-2.249071,5.376616,0.941082,-2.772903,6.750803],[-9.432017,-5.033311,6.485157,0.623848,7.350019,7.137709,1.064936,-3.368880,-5.560246,9.086712,-0.720962],[-8.092316,9.464816,-4.713324,-1.212661,-6.170701,-7.738088,-9.844837,-2.275341,-4.123884,7.882068,4.731388],[9.141725,-8.092645,-1.424307,-4.652195,9.326979,-9.396817,-1.268688,-0.444793,-3.488266,-0.142125,-2.593144],[-3.810375,-4.398055,-4.240882,-7.925530,-3.084615,9.151378,9.235850,-4.162428,-1.630958,-2.636950,-8.424496],[2.063551,2.635573,8.065010,-9.839624,8.701457,0.329247,7.751321,5.592497,0.103354,4.090283,-5.256598],[-2.837295,5.311738,3.088258,2.481201,-4.642195,-0.866861,-4.198129,0.679883,2.166451,5.897762,-2.365881],[-6.880384,-0.422235,2.204765,-7.069513,-6.068809,-1.924859,5.279955,6.614875,-6.827594,9.608899,-2.711399],[-0.289266,1.888358,-4.334096,-2.913221,0.805666,-7.146644,1.227162,-7.149775,4.587220,-1.180402,-5.953719],[3.120757,2.935477,9.994161,3.731571,-7.990661,-5.384821,9.675119,0.344068,9.340403,9.817866,-6.362197]],[[-8.296253,-5.726923,-7.084642,4.196462,9.239077,-4.314409,-5.198367,9.961686,4.805252,-1.065157,7.375998],[-2.671517,5.417134,-8.129083,2.347276,1.789229,7.314888,-1.427767,3.414520,-2.002909,-5.556404,1.252258],[-7.885148,-9.046212,-5.724559,7.012648,2.326414,-0.359594,-2.509753,2.997873,-4.688409,8.620423,-9.849465],[0.443742,1.197539,-5.812883,-0.157980,6.234661,1.627660,5.870283,-9.410455,5.564968,-3.693406,-3.293723],[-4.684186,3.529882,3.161322,-5.199108,-1.708874,9.198089,3.241404,8.644089,5.319627,0.665866,6.711776],[-8.709772,5.924609,3.797685,-3.103628,-3.030698,0.074982,1.868266,-7.250307,-7.883237,1.552679,5.598979],[-5.379703,-3.024233,2.918586,-1.331743,-0.208650,-1.132839,-7.486296,-1.802388,4.174555,-7.558866,-7.523752],[2.223893,1.280966,4.051071,4.128476,-7.774423,2.440471,3.937277,-3.371837,6.063428,9.168330,-3.529859],[2.450280,-0.012096,-3.450878,7.143529,-7.621940,0.861336,2.927532,-3.540999,3.113668,-7.771948,4.148560],[1.557054,1.373847,0.432914,-9.919588,-4.970351,-5.151327,9.802639,2.139933,1.437764,1.717367,-3.554198],[6.019568,-2.817808,0.077078,9.954451,-5.019557,-6.301938,-1.635219,-6.792654,1.102299,-0.434432,8.564651],[-0.169613,6.837377,-7.216497,7.946305,7.185695,0.842320,8.178428,-7.696310,-0.464811,-4.742121,-9.167972],[-6.797879,2.813750,6.141365,9.298274,3.917236,-0.199986,0.952772,-2.598355,2.066900,6.006646,3.006865]],[[1.291730,3.461797,-7.803068,-6.625329,-1.910437,-0.516399,-8.359942,-7.578410,-0.954284,-9.040399,-9.830720],[-5.907061,-0.458415,-8.920069,-5.524663,-7.462090,4.248401,8.232376,0.547022,-7.202557,-7.887005,-5.620866],[-0.163112,-9.672190,-6.455373,4.371593,-7.338201,7.188208,4.353071,-0.659213,4.872951,8.484064,3.119080],[4.498571,-8.729871,-4.442403,-1.916860,-0.021210,4.099067,7.149252,5.880049,-1.031446,-9.057196,-5.012057],[5.358352,-2.596677,6.898047,9.656727,-2.599393,-6.361858,-9.197745,-7.128201,8.532985,-5.617773,2.349665],[-1.403494,-7.036786,6.051140,0.872259,9.957337,-9.346047,2.719345,5.057915,-3.203137,4.835166,4.115261],[-8.180661,-2.024448,-2.946981,-5.991429,7.756925,-5.817715,-4.769799,-8.552835,5.988465,-8.069242,8.197768],[0.477599,9.168072,9.541325,-8.801668,-8.992728,-4.627049,-9.987330,8.156530,-8.782055,-0.256434,6.538158],[-6.100263,-2.457890,2.605883,-0.526889,-6.734917,1.143588,2.776038,1.737070,1.963160,-8.708266,-1.161536],[-4.365857,-1.429118,-4.260907,5.019411,8.886353,-0.295130,5.217103,-7.936588,-4.516910,-6.941423,-1.094872],[-2.291626,4.199990,-5.457637,3.606770,1.821284,-0.260606,1.691698,-7.606769,-3.695737,-4.261315,6.400900],[-6.675623,1.895913,-4.293150,8.108153,-2.174397,-9.579137,7.849849,1.020310,-8.050014,6.814697,-6.966406],[8.173681,5.217462,-8.993098,-3.730425,-7.268981,-3.892634,-7.709007,7.769171,7.309231,7.312638,5.710314]],[[-8.563200,-6.605597,5.394377,-4.279888,-3.442865,-6.498589,1.077005,6.685835,-7.791753,1.322953,-9.401350],[-8.920515,0.564820,4.401731,3.767967,4.910521,2.995673,-9.064845,7.444202,6.598118,0.937900,-9.004560],[-9.844757,-9.762031,8.986931,2.834978,9.805716,2.216355,-6.024682,-7.792826,5.794655,9.284977,5.756564],[-5.815605,-5.696486,-0.923006,5.573214,3.753738,-3.924200,4.848785,8.828062,4.193561,-7.529318,2.854720],[3.932723,-1.988796,1.885169,-7.761814,-3.671133,-3.297548,-6.717967,-7.364091,4.576860,0.455705,1.639503],[0.486256,-8.230750,-0.060080,9.406062,5.921476,6.690795,4.814236,2.554447,9.492535,-3.307948,8.080845],[-9.870054,6.218409,5.408698,4.745529,5.833426,0.149633,-9.488511,-0.201076,3.317979,-3.280540,8.660347],[4.202106,7.695744,-1.826223,-5.971187,-9.904374,7.586412,-1.239406,7.495592,-0.040397,-5.094926,-9.977887],[1.682078,-3.183080,-7.684719,6.696261,5.192142,-7.844474,8.129824,-5.930935,-7.269001,-3.967478,-8.958960],[-9.691216,-8.883314,-1.170535,-2.543392,3.609758,5.808197,-5.086307,1.186583,-8.247788,5.274246,1.922253],[0.735728,-9.480390,8.608606,-8.231151,-3.869360,2.273913,9.393352,4.105304,-4.876956,2.036317,6.919609],[5.868184,2.301376,8.131491,-5.241406,3.446699,-8.943101,-9.986454,-3.209558,-6.300973,3.028171,2.696818],[6.511373,4.060763,-2.476706,-2.011006,-2.642981,-9.336322,7.236970,-3.045174,-5.314798,3.752711,-8.470427]]], dtype = "float32")#candidate|688|(8, 13, 11)|const|float32
bop_689 = relay.mod(uop_676.astype('float64'), relay.reshape(const_688.astype('float64'), relay.shape_of(uop_676))) # shape=(8, 13, 11)
func_639_call = mod.get_global_var('func_639')
func_641_call = mutated_mod.get_global_var('func_641')
var_693 = relay.var("var_693", dtype = "float64", shape = (420,))#candidate|693|(420,)|var|float64
call_692 = relay.TupleGetItem(func_639_call(relay.reshape(var_693.astype('float64'), [4, 15, 7])), 0)
call_694 = relay.TupleGetItem(func_641_call(relay.reshape(var_693.astype('float64'), [4, 15, 7])), 0)
func_639_call = mod.get_global_var('func_639')
func_641_call = mutated_mod.get_global_var('func_641')
call_699 = relay.TupleGetItem(func_639_call(relay.reshape(var_693.astype('float64'), [4, 15, 7])), 0)
call_700 = relay.TupleGetItem(func_641_call(relay.reshape(var_693.astype('float64'), [4, 15, 7])), 0)
uop_701 = relay.sigmoid(uop_682.astype('float64')) # shape=(8, 13, 11)
bop_703 = relay.less_equal(uop_682.astype('bool'), relay.reshape(bop_659.astype('bool'), relay.shape_of(uop_682))) # shape=(8, 13, 11)
var_706 = relay.var("var_706", dtype = "bool", shape = (8, 13, 11))#candidate|706|(8, 13, 11)|var|bool
bop_707 = relay.greater_equal(bop_703.astype('bool'), relay.reshape(var_706.astype('bool'), relay.shape_of(bop_703))) # shape=(8, 13, 11)
output = relay.Tuple([bop_685,bop_689,call_692,var_693,call_699,uop_701,bop_707,])
output2 = relay.Tuple([bop_685,bop_689,call_694,var_693,call_700,uop_701,bop_707,])
func_721 = relay.Function([var_656,var_667,var_693,var_706,], output)
mod['func_721'] = func_721
mod = relay.transform.InferType()(mod)
var_722 = relay.var("var_722", dtype = "float32", shape = (8, 13, 11))#candidate|722|(8, 13, 11)|var|float32
var_723 = relay.var("var_723", dtype = "float32", shape = (8, 13, 11))#candidate|723|(8, 13, 11)|var|float32
var_724 = relay.var("var_724", dtype = "float64", shape = (420,))#candidate|724|(420,)|var|float64
var_725 = relay.var("var_725", dtype = "bool", shape = (8, 13, 11))#candidate|725|(8, 13, 11)|var|bool
output = func_721(var_722,var_723,var_724,var_725,)
func_726 = relay.Function([var_722,var_723,var_724,var_725,], output)
mutated_mod['func_726'] = func_726
mutated_mod = relay.transform.InferType()(mutated_mod)
var_728 = relay.var("var_728", dtype = "float64", shape = (2, 10))#candidate|728|(2, 10)|var|float64
const_729 = relay.const([[-5.191254,2.236858,-3.800622,-1.182304,3.744306,-1.074249,-7.910174,1.393692,4.341413,5.071555],[7.342143,-3.243351,-5.495646,-7.977960,9.216464,-9.904077,-5.582419,-3.673591,-7.199046,-4.017410]], dtype = "float64")#candidate|729|(2, 10)|const|float64
bop_730 = relay.greater_equal(var_728.astype('bool'), relay.reshape(const_729.astype('bool'), relay.shape_of(var_728))) # shape=(2, 10)
bop_738 = relay.greater(bop_730.astype('bool'), relay.reshape(var_728.astype('bool'), relay.shape_of(bop_730))) # shape=(2, 10)
func_153_call = mod.get_global_var('func_153')
func_156_call = mutated_mod.get_global_var('func_156')
var_742 = relay.var("var_742", dtype = "float64", shape = (1056,))#candidate|742|(1056,)|var|float64
call_741 = relay.TupleGetItem(func_153_call(relay.reshape(var_742.astype('float64'), [16, 6, 11])), 1)
call_743 = relay.TupleGetItem(func_156_call(relay.reshape(var_742.astype('float64'), [16, 6, 11])), 1)
uop_744 = relay.log2(bop_738.astype('float32')) # shape=(2, 10)
uop_747 = relay.sinh(uop_744.astype('float32')) # shape=(2, 10)
const_749 = relay.const([[2.135807,0.332865,-6.429311,1.400148,3.241058,-2.410043,3.628543,3.746908,9.061201,3.665794],[2.441298,-6.369431,1.598007,2.020269,-4.132812,9.514951,3.548659,-9.065409,0.837495,-5.335716]], dtype = "float32")#candidate|749|(2, 10)|const|float32
bop_750 = relay.multiply(uop_747.astype('int32'), relay.reshape(const_749.astype('int32'), relay.shape_of(uop_747))) # shape=(2, 10)
func_615_call = mod.get_global_var('func_615')
func_619_call = mutated_mod.get_global_var('func_619')
const_756 = relay.const([[-5,-9,8,8,1,-7,9,3,6,7,-4,3,-9,6,-6,-6,-5,1,-6,5,4,8,3,-7,7,-7,1,4,-6,9,3,4,-2,-3,4,8,2,-1,1,-4,-2,-1,2,2,6,-9,-1,-10,7,-3,-3,-9,9,10,8,8,4,2,5,-3,9,3,-4,-3,6,2,-7,4,8,3,7,-7,-3,6,-7,10,-10,4,-3,5,-2,-1,-5,-2,9,3,9,7,-1,9,-4,1,-9,8,-1,3,9,2,-9,8,-2,-3,1,2,-5,-7,9,2,5,1,-2,-9,8,-6,-6,2,5,8,7,7,2,-10,-4,-8,-2,-4,3,7,5,-8,-8,-8,8,-9,7,-7,-2,7,-7,4,5,-4,-1,2,4,8,2,5,3,5,1,-8,-8,8,1,-7,-10,-9,-1,7,-2,-7,1,6,-5,-4,4,-9,9,10,-1,-10,-9,4,-1,5,-5,10,4,2,9,-6,-9,7,-6,8,5,8,7,4,-2,1,-1,5,-7,7,-4,10,9,9,-4,3,-7,1,-5,-1,-4,-6,5,4,-10,-9,5,-8,-7,-2,-6,-10,7,8,-6,1,-3,-9,-2,-2,4,-9,-1,3,-1,-8,-9,-1,-10,10,-8,-10,-2,9,1,-6,-3,6,-1,-3,-6,-1,-9,-3,1,-7,-6,10,-5,10,2,-3,-3,10,-9,-6,4,-2,-10,-4,-10,-10,7,4,8,-9,2,-5,2,10,-10,8,4,7,-7,-4,3,3,3,1,1,-4,-4,9,6,-9,6,10,-9,5,-1,1,5,1,1,-3,-4,7,-8,5,-6,2,-4,-3,1,-9,6,-4,5,-1,1,-9,9,-4,4,2,-6,3,9,2,-1,4,-10,8,-6,-2,-3,2,10,-10,7,-4,8,-2,-3,-7,-5,4,9,-1,-3,4,7,-7,-1,-1,10,-1,5,5,8,-2,3,8,-10,-9,-5,8,-6,-1,1,-9,-7,1,-4,2,6,-4,-2,1,3,-4,2,2,-6,-8,5,9,-6,6,5,-1,-1,-2,-9,-4,4,3,-7,-3,-8,-9,-8,6,9,3,1,-8,1,5,4,4,-5,-3,4,5,-9,-6,7,1,6,-1,-2,-9,-9,-4,-2,-1,5,-1,-4,-2,3,1,2,5,10,6,4,-4,10,6,-6,10,2,10,-10,-8,10,4,-3,-1,6,4,3,-3,-6,4,7,-3,5,-3,-8,7,-4,2,-1,-2,-2,-7,6,5,-10,-4,-5,3,2,5,1,-10,5,4,-6,-1,5,-1,-9,-1,-7,5,3,2,-8,-7,-10,-5,10,-1,8,-5,1,1,6,10,-8,7,-4,1,5,10,7,-10,-7,-5,2,-5,8,4,8,9,-8,-1,5,-4,-10,-8,2,-3,-1,-3,8,8,3,-2,-6,9,9,-4,-10,7,-5,-6,-6,-5,7,2,5,1,-3,-9,6,-1,-7,-9,5,6,2,-8,-1,7,-2,-1,-10,5,9,-2,-3,10,-10,-7,-7,-7,-7,7,-8,5,-3,7,1,7,-7,5,-2,7,3,8,-1,9,2,5,2,1,6,-9,9,9,4,5,-8,-9,8,-1,4,6,-7,7,-9,-5,6,4,-9,-3,-8,9,1,-6,-5,8,8,7,-3,5,3,-10,3,6,8,6,-6,-9,5,3,-6,-1,-5,10,-3,3,-6,-4,-7,-2,8,1,5,10,-9,9,-6,-8,1,10,-6,9,-3,2,3,-7,1,4,9,-5,-7,-2,9,3,-1,-1,3,8,4,10,-1,8,-4,-10,-6,-3,-3,5,6,-9,8,-3,-1,9,-10,-3,-5,-5,-1,5,-3,-10,3,8,1,6,9,-2,-2,-8,-1,3,1,-8,6,2,2,-6,7,-4,-1,-2,4,5,-7,6,3,8,9,8,9,-5,-4,-8,4,1,-6,10,-7,4,7,3,-1,-5,7,6,-5,-9,-1,-10,-10,-2,2,-6,6,-2,-2,-4,6,-5,-5,8,-1,7,-7,7,-7,-4,1,-9,-10,8,8,6,1,6,10,6,-5,3,1,9,-9,7,3,-6,9,-6,8,-2,3,7,-3,7,7,10,-2,-3,2,-10,10,-9,-10,7,-6,8,-6,-3,7,9,-1,-9,-7,-3,-6,3,9,-4,-6,-6,1,-4,-9,8,-8,-10,-7,10,-1,3,6,-3,10,3,-10,10,4,-8,10,1,-4,-7,-9,-5,1,-10,-5,-1,10,-3,-4,-2,7,-3,8,5,4,3,8,-4,7,6,-6,-9,9,-9,-4,-1,-10,6,-8,-10,-2,4,4,-8,3,-4,-1,7,4,-2,6,-7,-10,6,5,-10,2,8,-3,6,-2,-7,5,4,-2,10,-2,6,-2,-9,10,-6,3,-8,-10,-4,-9,7,7,-5,-8,8,-2,1,9,-10,9,-1,10,1,-6,-4,6,-8,3,-1,-8,10,-5,3,2,3,9,-4,-5,-1,-9,-7,8,-7,1,-1,-1,7,3,8,-3,-5,-6,5,5,4,3,-9,9,4,-8,-6,-8,-4,5,4,-4,-4,1,4,7,10,-1,3,-6,7,3,-2,-8,-2,8,-3,-9,5,-6,6,6,3,6,7,-4,8,10,10,6,-3,10,-9,4,-10,2,6,-3,-5,-7,-3,-9,-2,-10,-8,-6,1,6,-6,-3,-3,-4,2,7,8,-2,-7,-1,9,-1,8,-8,5,8,-6,-8,3,-7,-1,4,8,-10,6,-6,-3,-4,6,9,-7,3,-7,2,2,4,-4,3,-4,-1,-4,-7,-5,9,4,3,4,-5,-6,10,-8,1,5,-8,-9,-9,-5,2,-9,1,9,-5,1,-2,10,-7,8,-2,6,-2,-8,-7,-9,-1,10,-2,-10,-1,-8,10,-7,-6,-6,10,10,-6,4,9,2,4,4,2,4,4,6,-1,10,-8,6,10,-1,-4,10,10,5,-8,3,-4,-3,10,2,-2,9,-3,-9,10,-1,-5,-8,-7,-1,2,4,1,-4,-9,-2,-3,3,5,10,4,-10,4,-5,10,5,10,-10,9,1,4,-8,-4,4,-5,-5,-8,7,2,3,8,8,7,8,-3,3,2,-4,-2,-4,-3,-7,3,-2,-8,8,3,3,-5,-8,-9,6,2,-6,10,-5,-10,-5,6,7,-4,-9,3,1,-2,6,1,3,-4,2,-2,9,7,2,-3,5,-5,6,-8,-8,5,-4,-8,2,7,-9,8,8,2,-9,10,-4,-10,-3,-7,-9,-6,1,-9,7,-6,4,6,1,3,-5,-8,7,9,-6,-10,-1,7,2,4,3,-8,7,-10,-2,-10,5,-5,-6,2,-1,9,1,-3,-2,-4,8,-2,6,3,5,4,-3,-2,-5,3,1,7,7,-6,-4,6,4,7,1,7,-3,-8,2,-7,6,-3,1,-10,-8,1,-7,-6,-6,1,2,3,-3,-4,-1,-1,1,6,4,2,4,-7,-10,-4,-5,4,-4,-7,-8,-1,-3,9,-8,6,3,-5,-1,8,5,-8,6,2,-10,9,9,10,4,-6,4,-6,9,-4,4,-4,3,-1,-7,-10,-1,5,-9,-4,4,-5,1,7,-3,-2,2,-7,-1,5,1,-3,4,-4,4,-6,-5,1,2,-8,9,-1,-8,-8,-7,5,8,-10,-9,-3,-2,3,10,5,-2,-5,5,-1,-4,-1,2,1,-7,2,7,9,6,9,8,8,-9,10,-10,-6,-2,7,5,3,9,6,-3,-7,8,7,-2,-5,-7,-4,6,5,-6,-1,10,-1,-1,7,3,-5,2,9,6,-6,9,-1,-9,-2,-9,-4,-8,10,-5,5,2,10,-3,-9,-5,1,-4,-10,9,8,6,8,10,-3,-6,-2,2,4,3,4,8,-3,-8,-5,-4,6,6,3,-2,8,9,-6,-2,-4,-6,4,-4,8,9,-10,8,-4,-7,-10,-5,7,-10,5,-5,-10,7,-7,-10,-4,6,-1,-1,-1,-4,-9,1,5,1,7,10,7,9,10,-8,-8,10,1,6,10,-2,3,5,7,9,-1,9,-6,-3,10,1,-8,-1,2,8,9,3,5,9,3,4,-7,-6,-5,-9,2,7,1,-8,5,-5,8,-6,7,3,-5,5,-3,-5,2,5,4,-4,8,-7,8,7,1,6,-1,7,-9,-8,-5,-9,8,3,-6,6,8,8,5,-1,9,4,8,5,-6,-10,-7,10,-4,9,9,-5,9,5,9,10,-3,5,6,1,-3,4,1,-7,-6,-5,4,-6,-5,6,-4,9,5,-3,6,-4,6,-10,2,1,6,5,-8,-2,1,-7,8,-4,-4,3,-2,8,3,-2,5,-2,-9,-10,9,2,-10,3,-6,1,6,7,-9,4,5,6,1,-1,-7,9,-5,-4,-1,2,-5,-7,-7,7,-1,7,-8,6,-2,-2,-9,3,-4,-8,-5,2,7,1,6,-9,10,3,-10,-9,-5,3,2,10,2,-2,-2,7,6,1,-5,-8,-8,-4,9,5,2,10,4,3,9,5,2,1,9,9,-4,-10,8,-5,7,1,-5,-7,-10,-7,6,2,-3,7,5,-9,-2,3,4,7,7,-1,10,3,8,10,-1,3,3,7,5,9,4,-9,10,-10,-7,-9,-8,-6,-2,2,9,9,-3,-2,-2,-8,-2,7,-7,7,9,-1,-6,-5,5,4,-1,6,9,4,8,10,-3,10,-7,5,2,-4,3,9,-5,2,5,-1,5,-6,9,2,6,-3,-10,-8,2,-3,9,5,-6,-1,-4,-5,-10,-3,-8,3,9,-4,2,-3,4,-3,9,-9,4,6,-6,10,-2,7,10,-2,5,9,-10,6,-10,8,-7,-4,9,-1,-3,-1,4,2,7,4,-4,1,-7,9,-6,-5,-6,-5,1,9,9,-8,7,8,6,8,-4,5,8,9,10,6,-4,-10,7,8,1,-8,8,6,7,-5,5,-4,-3,-4,6,-1,-10,-5,2,2,1,-4,7,2,-4,-5,-6,-1,-2,4,-6,10,7,5,-1,5,-9,1,-8,10,-10,10,-5,-9,-5,-7,-9,7,2,-5,8,-5,6,-3,-8,6,-7,6,-5,-1,7,1,4,-9,-7,9,4,-2,-7,7,-7,3,-9,-6,-3,7,4,5,-4,-2,1,-8,-5,-9,7,-8,-1,8,10,-6,-10,5,1,-2,6,-9,7,-4,-10,-5,-9,1,1,-9,5,6,-10,-9,-10,-7,-5,3,-4,4,-2,-2,9,-2,-5,3,-10,-7,-9,1,-7,4,-7,-6,8,4,9,4,1,10,8,-8,-2,9,1,-9,-10,-4,3,-5,-10,-10,7,8,-9,4,2,8,-6,5,6,-8,8,1]], dtype = "uint32")#candidate|756|(1, 2002)|const|uint32
call_755 = relay.TupleGetItem(func_615_call(relay.reshape(const_756.astype('uint32'), [11, 14, 13]), relay.reshape(call_741.astype('uint64'), [30,]), ), 9)
call_757 = relay.TupleGetItem(func_619_call(relay.reshape(const_756.astype('uint32'), [11, 14, 13]), relay.reshape(call_741.astype('uint64'), [30,]), ), 9)
bop_758 = relay.not_equal(uop_747.astype('bool'), relay.reshape(bop_738.astype('bool'), relay.shape_of(uop_747))) # shape=(2, 10)
uop_761 = relay.sigmoid(uop_747.astype('float64')) # shape=(2, 10)
bop_764 = relay.logical_and(uop_761.astype('bool'), relay.reshape(bop_738.astype('bool'), relay.shape_of(uop_761))) # shape=(2, 10)
var_769 = relay.var("var_769", dtype = "bool", shape = (2, 10))#candidate|769|(2, 10)|var|bool
bop_770 = relay.divide(bop_764.astype('float64'), relay.reshape(var_769.astype('float64'), relay.shape_of(bop_764))) # shape=(2, 10)
bop_773 = relay.floor_divide(bop_750.astype('float32'), relay.reshape(const_729.astype('float32'), relay.shape_of(bop_750))) # shape=(2, 10)
output = relay.Tuple([call_741,var_742,call_755,const_756,bop_758,bop_770,bop_773,])
output2 = relay.Tuple([call_743,var_742,call_757,const_756,bop_758,bop_770,bop_773,])
func_777 = relay.Function([var_728,var_742,var_769,], output)
mod['func_777'] = func_777
mod = relay.transform.InferType()(mod)
var_778 = relay.var("var_778", dtype = "float64", shape = (2, 10))#candidate|778|(2, 10)|var|float64
var_779 = relay.var("var_779", dtype = "float64", shape = (1056,))#candidate|779|(1056,)|var|float64
var_780 = relay.var("var_780", dtype = "bool", shape = (2, 10))#candidate|780|(2, 10)|var|bool
output = func_777(var_778,var_779,var_780,)
func_781 = relay.Function([var_778,var_779,var_780,], output)
mutated_mod['func_781'] = func_781
mutated_mod = relay.transform.InferType()(mutated_mod)
var_813 = relay.var("var_813", dtype = "uint64", shape = (13, 15))#candidate|813|(13, 15)|var|uint64
const_814 = relay.const([[7,4,-2,5,-10,4,8,-9,-10,-4,5,4,-8,-2,8],[-5,-5,4,3,-3,-2,5,4,4,3,3,-10,7,-9,-5],[8,4,2,-1,9,-10,-10,1,-10,-9,5,-10,7,4,-8],[-5,-2,2,-6,9,-4,8,-5,-9,6,-6,8,-7,-5,2],[-2,4,5,6,-6,1,-3,3,10,2,9,-8,-6,-3,7],[-8,4,3,-5,-5,5,10,-4,7,-6,1,-7,-4,5,-5],[5,-1,5,-2,-2,-1,10,-4,-8,-10,10,-10,10,1,3],[8,-9,-9,-3,-7,6,4,1,2,-9,-4,2,5,9,-9],[-1,-10,-6,-7,2,5,-1,-5,7,2,10,6,-9,5,-8],[-8,-1,-9,9,-7,-9,-8,3,3,-3,-7,-8,-4,-3,-4],[-5,-10,-1,3,-7,1,-5,6,-2,-8,-1,-7,-8,-5,2],[-2,2,-3,-2,-2,-9,3,-2,9,-8,-1,4,6,-1,-2],[10,1,-7,4,-1,8,-1,-3,2,9,1,-6,9,4,5]], dtype = "uint64")#candidate|814|(13, 15)|const|uint64
bop_815 = relay.bitwise_or(var_813.astype('uint64'), relay.reshape(const_814.astype('uint64'), relay.shape_of(var_813))) # shape=(13, 15)
bop_827 = relay.logical_xor(bop_815.astype('uint16'), relay.reshape(const_814.astype('uint16'), relay.shape_of(bop_815))) # shape=(13, 15)
func_465_call = mod.get_global_var('func_465')
func_468_call = mutated_mod.get_global_var('func_468')
const_832 = relay.const([5.222625], dtype = "float32")#candidate|832|(1,)|const|float32
call_831 = relay.TupleGetItem(func_465_call(relay.reshape(const_832.astype('float32'), [1,])), 3)
call_833 = relay.TupleGetItem(func_468_call(relay.reshape(const_832.astype('float32'), [1,])), 3)
uop_859 = relay.acos(call_831.astype('float64')) # shape=(9, 3, 13)
uop_861 = relay.acos(call_833.astype('float64')) # shape=(9, 3, 13)
uop_862 = relay.cosh(uop_859.astype('float32')) # shape=(9, 3, 13)
uop_864 = relay.cosh(uop_861.astype('float32')) # shape=(9, 3, 13)
output = relay.Tuple([bop_827,const_832,uop_862,])
output2 = relay.Tuple([bop_827,const_832,uop_864,])
F = relay.Function([var_813,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_813,], output2)
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
input_813= np.array([[7,9,-9,8,-5,3,-3,-10,7,-5,-6,2,2,-7,-6],[-6,-10,6,2,-3,6,10,8,-8,-3,8,1,9,9,-4],[7,-6,8,4,-8,8,4,3,-9,-9,-8,-4,-8,-10,7],[10,7,8,-7,-5,1,9,1,-7,7,-5,-4,3,9,7],[-9,-9,4,-4,-2,-8,-4,10,2,-8,-5,3,-9,9,10],[-4,-6,7,-10,3,-2,7,-7,-1,-4,10,-9,9,9,-6],[6,10,-10,-2,-10,4,-5,7,-8,2,5,-3,5,2,10],[3,-7,9,-1,-10,5,2,5,-3,8,-6,9,7,1,-8],[-9,-6,5,6,6,7,-9,5,-10,10,-6,-1,8,-9,-9],[-10,9,9,-3,9,10,-3,9,-7,-8,7,5,-9,5,-2],[6,-7,9,-10,-10,2,-7,2,-4,-4,-8,-6,-8,6,6],[10,-9,10,4,8,9,5,-6,-1,-8,3,-6,8,8,7],[-1,8,5,-8,8,-2,-8,2,-6,6,-7,-1,-2,5,5]], dtype='uint64')
module1.set_input('var_813', input_813)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_813, )
res3 = intrp3.evaluate()(input_813, )
res4 = intrp4.evaluate()(input_813, )
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
module5.set_input('var_813', input_813)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_813, )
res7 = intrp7.evaluate()(input_813, )
res8 = intrp8.evaluate()(input_813, )
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
module9.set_input('var_813', input_813)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_813, )
res11 = intrp11.evaluate()(input_813, )
res12 = intrp12.evaluate()(input_813, )
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
module13.set_input('var_813', input_813)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_813, )
res15 = intrp15.evaluate()(input_813, )
res16 = intrp16.evaluate()(input_813, )
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
module17.set_input('var_813', input_813)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_813, )
res19 = intrp19.evaluate()(input_813, )
res20 = intrp20.evaluate()(input_813, )
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
module21.set_input('var_813', input_813)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_813, )
res23 = intrp23.evaluate()(input_813, )
res24 = intrp24.evaluate()(input_813, )
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

'''104: TVMFuncCall
103: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
102: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
101: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
100: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
99: tvm::transform::Pass::operator()(tvm::IRModule) const
98: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
97: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
96: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
95: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
94: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
93: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
92: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
91: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
90: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
89: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
88: _ZN3tvm5relay16MixedModeMutato
87: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
86: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
85: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
84: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
83: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
82: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
81: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
80: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
79: _ZN3tvm5relay16MixedModeMutato
78: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
77: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
76: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
75: tvm::relay::MixedModeMutator::VisitExpr_(tvm::relay::CallNode const*)
74: tvm::relay::DynamicToStaticMutator::Rewrite_(tvm::relay::CallNode const*, tvm::RelayExpr const&)
73: std::_Function_handler<tvm::RelayExpr (tvm::relay::CallNode const*), tvm::relay::DynamicToStaticMutator::DynamicToStaticMutator(tvm::IRModule, tvm::relay::Function)::{lambda(tvm::relay::CallNode const*)#1}>::_M_invoke(std::_Any_data const&, tvm::relay::CallNode const*&&)
72: tvm::relay::DynamicToStaticMutator::PrepareArgs(tvm::relay::CallNode const*)
71: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
70: tvm::transform::Pass::operator()(tvm::IRModule) const
69: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
68: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
67: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FoldConstant()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FoldConstant()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
66: tvm::relay::transform::FoldConstantExpr(tvm::RelayExpr const&, tvm::IRModule const&)
65: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
64: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
63: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
62: _ZN3tvm5relay16MixedModeMutato
61: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
60: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
59: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
58: tvm::relay::transform::(anonymous namespace)::ConstantFolder::VisitExpr_(tvm::relay::FunctionNode const*)
57: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
56: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
55: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
54: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
53: _ZN3tvm5relay16MixedModeMutato
52: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
51: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
50: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
49: tvm::relay::MixedModeMutator::VisitExpr_(tvm::relay::CallNode const*)
48: tvm::relay::transform::(anonymous namespace)::ConstantFolder::Rewrite_(tvm::relay::CallNode const*, tvm::RelayExpr const&)
47: tvm::relay::transform::(anonymous namespace)::ConstantFolder::ConstEvaluate(tvm::RelayExpr const&)
46: tvm::relay::Eval(tvm::RelayExpr, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void>, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >, DLDevice, tvm::Target)
45: tvm::relay::Prepare(tvm::IRModule, tvm::CompilationConfig)
44: tvm::transform::Pass::operator()(tvm::IRModule) const
43: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
42: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
41: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
40: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
36: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
35: tvm::transform::Pass::operator()(tvm::IRModule) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
27: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
26: _ZN3tvm5relay9transform22Devic
25: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
24: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
23: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
22: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
21: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
20: tvm::relay::tec::LowerTensorExprMutator::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
16: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
15: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
14: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
13: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
12: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
11: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
9: tvm::relay::tec::TECompilerImpl::LowerShapeFunc(tvm::relay::tec::CCacheKey const&)
8: tvm::relay::tec::TECompilerImpl::LowerShapeFuncInternal(tvm::relay::tec::CCacheKey const&)
7: tvm::relay::tec::ShapeFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
6: tvm::relay::tec::MakeShapeFunc::Create(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
5: tvm::relay::tec::MakeShapeFunc::VisitExpr(tvm::RelayExpr const&)
4: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFNS_7runtime
1: tvm::relay::tec::MakeShapeFunc::VisitExpr_(tvm::relay::CallNode const*)
0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
3: TVMFuncCall
2: _ZNSt17_Function_handlerIFvN3tvm7run
1: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
0: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
4: TVMFuncCall
3: _ZNSt17_Function_handlerIFvN3tvm7run
2: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
1: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
0: tvm::runtime::Array<tvm::PrimExpr, void> tvm::runtime::TVMPODValue_::AsObjectRef<tvm::runtime::Array<tvm::PrimExpr, void> >() const

'''