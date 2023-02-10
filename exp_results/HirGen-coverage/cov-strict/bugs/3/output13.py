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
var_88 = relay.var("var_88", dtype = "float32", shape = (16, 5))#candidate|88|(16, 5)|var|float32
uop_89 = relay.rsqrt(var_88.astype('float32')) # shape=(16, 5)
bop_94 = relay.bitwise_and(var_88.astype('int32'), relay.reshape(uop_89.astype('int32'), relay.shape_of(var_88))) # shape=(16, 5)
bop_100 = relay.logical_and(var_88.astype('bool'), relay.reshape(uop_89.astype('bool'), relay.shape_of(var_88))) # shape=(16, 5)
output = relay.Tuple([bop_94,bop_100,])
output2 = relay.Tuple([bop_94,bop_100,])
func_109 = relay.Function([var_88,], output)
mod['func_109'] = func_109
mod = relay.transform.InferType()(mod)
var_110 = relay.var("var_110", dtype = "float32", shape = (16, 5))#candidate|110|(16, 5)|var|float32
output = func_109(var_110)
func_111 = relay.Function([var_110], output)
mutated_mod['func_111'] = func_111
mutated_mod = relay.transform.InferType()(mutated_mod)
var_122 = relay.var("var_122", dtype = "float64", shape = (5, 8))#candidate|122|(5, 8)|var|float64
const_123 = relay.const([[4.846423,-7.273556,4.381621,0.802563,8.848953,8.576487,-6.563215,5.405862],[1.810867,6.853375,2.545692,-8.438767,1.411971,5.115030,-3.397504,1.437414],[-7.460769,-2.068745,9.874140,-4.660369,7.873467,-1.972971,-4.617445,-5.326308],[-7.320847,-5.401266,-9.271981,-2.192772,-5.643904,3.258251,-8.966301,-5.716660],[-2.136171,-0.766550,9.749237,-3.548185,-2.270359,-5.055986,6.602551,-2.625558]], dtype = "float64")#candidate|123|(5, 8)|const|float64
bop_124 = relay.power(var_122.astype('float64'), relay.reshape(const_123.astype('float64'), relay.shape_of(var_122))) # shape=(5, 8)
func_109_call = mod.get_global_var('func_109')
func_111_call = mutated_mod.get_global_var('func_111')
var_132 = relay.var("var_132", dtype = "float32", shape = (80,))#candidate|132|(80,)|var|float32
call_131 = relay.TupleGetItem(func_109_call(relay.reshape(var_132.astype('float32'), [16, 5])), 0)
call_133 = relay.TupleGetItem(func_111_call(relay.reshape(var_132.astype('float32'), [16, 5])), 0)
uop_141 = relay.asin(var_132.astype('float32')) # shape=(80,)
output = relay.Tuple([bop_124,call_131,uop_141,])
output2 = relay.Tuple([bop_124,call_133,uop_141,])
func_143 = relay.Function([var_122,var_132,], output)
mod['func_143'] = func_143
mod = relay.transform.InferType()(mod)
mutated_mod['func_143'] = func_143
mutated_mod = relay.transform.InferType()(mutated_mod)
func_143_call = mutated_mod.get_global_var('func_143')
var_145 = relay.var("var_145", dtype = "float64", shape = (5, 8))#candidate|145|(5, 8)|var|float64
var_146 = relay.var("var_146", dtype = "float32", shape = (80,))#candidate|146|(80,)|var|float32
call_144 = func_143_call(var_145,var_146,)
output = call_144
func_147 = relay.Function([var_145,var_146,], output)
mutated_mod['func_147'] = func_147
mutated_mod = relay.transform.InferType()(mutated_mod)
var_167 = relay.var("var_167", dtype = "float32", shape = (1, 4))#candidate|167|(1, 4)|var|float32
uop_168 = relay.tan(var_167.astype('float32')) # shape=(1, 4)
bop_187 = relay.right_shift(var_167.astype('uint8'), relay.reshape(uop_168.astype('uint8'), relay.shape_of(var_167))) # shape=(1, 4)
output = relay.Tuple([bop_187,])
output2 = relay.Tuple([bop_187,])
func_192 = relay.Function([var_167,], output)
mod['func_192'] = func_192
mod = relay.transform.InferType()(mod)
mutated_mod['func_192'] = func_192
mutated_mod = relay.transform.InferType()(mutated_mod)
var_193 = relay.var("var_193", dtype = "float32", shape = (1, 4))#candidate|193|(1, 4)|var|float32
func_192_call = mutated_mod.get_global_var('func_192')
call_194 = func_192_call(var_193)
output = call_194
func_195 = relay.Function([var_193], output)
mutated_mod['func_195'] = func_195
mutated_mod = relay.transform.InferType()(mutated_mod)
var_276 = relay.var("var_276", dtype = "float32", shape = (1, 9))#candidate|276|(1, 9)|var|float32
uop_277 = relay.cos(var_276.astype('float32')) # shape=(1, 9)
bop_280 = relay.multiply(uop_277.astype('uint16'), relay.reshape(var_276.astype('uint16'), relay.shape_of(uop_277))) # shape=(1, 9)
uop_284 = relay.tan(uop_277.astype('float64')) # shape=(1, 9)
func_109_call = mod.get_global_var('func_109')
func_111_call = mutated_mod.get_global_var('func_111')
var_291 = relay.var("var_291", dtype = "float32", shape = (1, 80))#candidate|291|(1, 80)|var|float32
call_290 = relay.TupleGetItem(func_109_call(relay.reshape(var_291.astype('float32'), [16, 5])), 0)
call_292 = relay.TupleGetItem(func_111_call(relay.reshape(var_291.astype('float32'), [16, 5])), 0)
uop_293 = relay.log2(uop_284.astype('float64')) # shape=(1, 9)
bop_295 = relay.power(uop_284.astype('float32'), relay.reshape(bop_280.astype('float32'), relay.shape_of(uop_284))) # shape=(1, 9)
var_301 = relay.var("var_301", dtype = "float64", shape = (9, 9))#candidate|301|(9, 9)|var|float64
bop_302 = relay.greater_equal(uop_293.astype('bool'), var_301.astype('bool')) # shape=(9, 9)
output = relay.Tuple([call_290,var_291,bop_295,bop_302,])
output2 = relay.Tuple([call_292,var_291,bop_295,bop_302,])
func_305 = relay.Function([var_276,var_291,var_301,], output)
mod['func_305'] = func_305
mod = relay.transform.InferType()(mod)
mutated_mod['func_305'] = func_305
mutated_mod = relay.transform.InferType()(mutated_mod)
func_305_call = mutated_mod.get_global_var('func_305')
var_307 = relay.var("var_307", dtype = "float32", shape = (1, 9))#candidate|307|(1, 9)|var|float32
var_308 = relay.var("var_308", dtype = "float32", shape = (1, 80))#candidate|308|(1, 80)|var|float32
var_309 = relay.var("var_309", dtype = "float64", shape = (9, 9))#candidate|309|(9, 9)|var|float64
call_306 = func_305_call(var_307,var_308,var_309,)
output = call_306
func_310 = relay.Function([var_307,var_308,var_309,], output)
mutated_mod['func_310'] = func_310
mutated_mod = relay.transform.InferType()(mutated_mod)
var_340 = relay.var("var_340", dtype = "float64", shape = (14, 4, 13))#candidate|340|(14, 4, 13)|var|float64
var_341 = relay.var("var_341", dtype = "float64", shape = (14, 4, 13))#candidate|341|(14, 4, 13)|var|float64
bop_342 = relay.floor_divide(var_340.astype('float64'), relay.reshape(var_341.astype('float64'), relay.shape_of(var_340))) # shape=(14, 4, 13)
var_352 = relay.var("var_352", dtype = "float64", shape = (14, 4, 13))#candidate|352|(14, 4, 13)|var|float64
bop_353 = relay.less_equal(bop_342.astype('bool'), relay.reshape(var_352.astype('bool'), relay.shape_of(bop_342))) # shape=(14, 4, 13)
var_356 = relay.var("var_356", dtype = "float64", shape = (14, 4, 13))#candidate|356|(14, 4, 13)|var|float64
bop_357 = relay.equal(bop_342.astype('bool'), relay.reshape(var_356.astype('bool'), relay.shape_of(bop_342))) # shape=(14, 4, 13)
var_375 = relay.var("var_375", dtype = "float64", shape = (14, 4, 13))#candidate|375|(14, 4, 13)|var|float64
bop_376 = relay.minimum(var_356.astype('uint32'), relay.reshape(var_375.astype('uint32'), relay.shape_of(var_356))) # shape=(14, 4, 13)
uop_380 = relay.acos(var_341.astype('float32')) # shape=(14, 4, 13)
output = relay.Tuple([bop_353,bop_357,bop_376,uop_380,])
output2 = relay.Tuple([bop_353,bop_357,bop_376,uop_380,])
func_384 = relay.Function([var_340,var_341,var_352,var_356,var_375,], output)
mod['func_384'] = func_384
mod = relay.transform.InferType()(mod)
mutated_mod['func_384'] = func_384
mutated_mod = relay.transform.InferType()(mutated_mod)
func_384_call = mutated_mod.get_global_var('func_384')
var_386 = relay.var("var_386", dtype = "float64", shape = (14, 4, 13))#candidate|386|(14, 4, 13)|var|float64
var_387 = relay.var("var_387", dtype = "float64", shape = (14, 4, 13))#candidate|387|(14, 4, 13)|var|float64
var_388 = relay.var("var_388", dtype = "float64", shape = (14, 4, 13))#candidate|388|(14, 4, 13)|var|float64
var_389 = relay.var("var_389", dtype = "float64", shape = (14, 4, 13))#candidate|389|(14, 4, 13)|var|float64
var_390 = relay.var("var_390", dtype = "float64", shape = (14, 4, 13))#candidate|390|(14, 4, 13)|var|float64
call_385 = func_384_call(var_386,var_387,var_388,var_389,var_390,)
output = call_385
func_391 = relay.Function([var_386,var_387,var_388,var_389,var_390,], output)
mutated_mod['func_391'] = func_391
mutated_mod = relay.transform.InferType()(mutated_mod)
const_393 = relay.const([[-9,10,6,-8,-2,-2,4,8,1,10,-1,8],[-9,9,-1,1,-3,-10,8,7,-4,2,-1,1],[-4,1,-2,-1,8,-4,-4,-10,7,-7,8,4],[-7,2,-7,-7,4,6,8,8,7,8,-6,-10],[8,7,-5,10,9,7,4,5,-3,5,-8,1],[6,1,-6,6,-10,2,6,-5,-3,-1,8,3],[8,-3,3,-4,-3,-3,-5,3,-2,3,4,-4],[-10,-3,10,-10,-6,-5,-6,9,5,8,-1,-7],[5,-2,8,7,-1,-8,6,7,10,-2,2,-8],[3,9,7,6,6,3,6,-3,10,-7,-7,8],[-10,-6,10,-9,9,7,-6,10,-3,-5,2,-7],[9,-9,-2,-6,7,-5,7,10,-4,4,1,1],[6,4,-3,-2,-9,-1,-8,-1,10,-2,2,2]], dtype = "int32")#candidate|393|(13, 12)|const|int32
var_394 = relay.var("var_394", dtype = "int32", shape = (13, 12))#candidate|394|(13, 12)|var|int32
bop_395 = relay.multiply(const_393.astype('int32'), relay.reshape(var_394.astype('int32'), relay.shape_of(const_393))) # shape=(13, 12)
bop_398 = relay.equal(const_393.astype('bool'), relay.reshape(bop_395.astype('bool'), relay.shape_of(const_393))) # shape=(13, 12)
func_305_call = mod.get_global_var('func_305')
func_310_call = mutated_mod.get_global_var('func_310')
var_405 = relay.var("var_405", dtype = "float32", shape = (9,))#candidate|405|(9,)|var|float32
const_406 = relay.const([5.280781,2.512636,7.826594,-0.852892,7.275079,1.972685,-6.957125,-7.210367,8.502878,-7.151692,7.120866,4.387769,8.294636,1.160899,-4.407935,-6.249754,9.337330,6.142881,1.143541,-5.573622,2.072405,-1.844575,-2.936144,5.434498,1.882275,7.388146,4.797612,-4.684639,-4.679819,4.898022,1.636693,-1.334939,7.855118,1.742427,-9.515377,-9.020390,-9.678794,1.839986,7.938905,-4.233835,-9.308464,8.086848,7.549867,-7.199942,-1.839167,-3.583441,-6.031814,-6.390093,-5.799620,-5.534446,6.809428,-3.375406,1.512304,-3.853911,0.760599,1.900474,-8.719688,8.233801,9.413759,6.115291,0.708981,9.484879,2.758760,-6.130424,4.692938,-0.886309,8.222664,-6.002599,-6.194061,4.887730,-5.300143,0.424787,4.721051,-1.871976,-7.657486,-0.165632,1.339971,7.449601,5.046872,-0.253137], dtype = "float32")#candidate|406|(80,)|const|float32
const_407 = relay.const([-3.616961,4.651187,-0.762840,-1.418005,-8.456326,-0.822374,-7.887885,-3.677366,-3.186914,-7.939734,-9.928781,-5.498131,3.350249,-6.457754,9.650880,1.569647,-5.239870,-7.946153,1.459462,8.193296,-8.111665,-8.212580,-2.118650,-8.276086,0.557517,-2.307240,-0.365751,-1.945547,9.015125,-2.360948,-2.771502,-1.935269,2.296946,-9.512563,2.063376,-2.401971,9.073507,-6.831159,1.891336,-4.790193,-6.493982,-0.864719,-2.479899,-6.501559,5.007767,-4.497554,2.319467,-5.068099,-1.121322,4.868364,-1.037321,5.237788,-3.244332,8.349953,-9.315777,3.159685,-3.280879,-8.037887,-6.928622,3.752903,0.939802,9.744885,-8.438814,4.054698,-3.802787,7.455180,-3.132474,-7.650860,-6.932341,-6.120908,-0.504217,7.567143,4.642522,7.373635,8.794529,-5.631302,-3.946260,8.728696,0.788320,0.152750,3.736693], dtype = "float64")#candidate|407|(81,)|const|float64
call_404 = relay.TupleGetItem(func_305_call(relay.reshape(var_405.astype('float32'), [1, 9]), relay.reshape(const_406.astype('float32'), [1, 80]), relay.reshape(const_407.astype('float64'), [9, 9]), ), 3)
call_408 = relay.TupleGetItem(func_310_call(relay.reshape(var_405.astype('float32'), [1, 9]), relay.reshape(const_406.astype('float32'), [1, 80]), relay.reshape(const_407.astype('float64'), [9, 9]), ), 3)
bop_410 = relay.minimum(const_393.astype('int32'), relay.reshape(bop_398.astype('int32'), relay.shape_of(const_393))) # shape=(13, 12)
bop_420 = relay.less(bop_410.astype('bool'), relay.reshape(bop_398.astype('bool'), relay.shape_of(bop_410))) # shape=(13, 12)
bop_427 = relay.left_shift(const_393.astype('uint64'), relay.reshape(var_394.astype('uint64'), relay.shape_of(const_393))) # shape=(13, 12)
bop_435 = relay.bitwise_xor(var_394.astype('int8'), relay.reshape(bop_410.astype('int8'), relay.shape_of(var_394))) # shape=(13, 12)
output = relay.Tuple([call_404,var_405,const_406,const_407,bop_420,bop_427,bop_435,])
output2 = relay.Tuple([call_408,var_405,const_406,const_407,bop_420,bop_427,bop_435,])
func_440 = relay.Function([var_394,var_405,], output)
mod['func_440'] = func_440
mod = relay.transform.InferType()(mod)
mutated_mod['func_440'] = func_440
mutated_mod = relay.transform.InferType()(mutated_mod)
func_440_call = mutated_mod.get_global_var('func_440')
var_442 = relay.var("var_442", dtype = "int32", shape = (13, 12))#candidate|442|(13, 12)|var|int32
var_443 = relay.var("var_443", dtype = "float32", shape = (9,))#candidate|443|(9,)|var|float32
call_441 = func_440_call(var_442,var_443,)
output = call_441
func_444 = relay.Function([var_442,var_443,], output)
mutated_mod['func_444'] = func_444
mutated_mod = relay.transform.InferType()(mutated_mod)
const_465 = relay.const([[-0.637591,-7.727575,-0.215172,-4.945068],[2.957524,5.493532,-3.198277,-2.675872],[6.311932,2.349237,7.767893,-1.128441],[-6.379639,6.384633,-2.693486,6.806088]], dtype = "float64")#candidate|465|(4, 4)|const|float64
uop_466 = relay.log2(const_465.astype('float64')) # shape=(4, 4)
var_477 = relay.var("var_477", dtype = "float64", shape = (4, 4))#candidate|477|(4, 4)|var|float64
bop_478 = relay.power(uop_466.astype('float64'), relay.reshape(var_477.astype('float64'), relay.shape_of(uop_466))) # shape=(4, 4)
uop_493 = relay.atanh(uop_466.astype('float32')) # shape=(4, 4)
func_440_call = mod.get_global_var('func_440')
func_444_call = mutated_mod.get_global_var('func_444')
var_497 = relay.var("var_497", dtype = "int32", shape = (156,))#candidate|497|(156,)|var|int32
var_498 = relay.var("var_498", dtype = "float32", shape = (3, 3))#candidate|498|(3, 3)|var|float32
call_496 = relay.TupleGetItem(func_440_call(relay.reshape(var_497.astype('int32'), [13, 12]), relay.reshape(var_498.astype('float32'), [9,]), ), 4)
call_499 = relay.TupleGetItem(func_444_call(relay.reshape(var_497.astype('int32'), [13, 12]), relay.reshape(var_498.astype('float32'), [9,]), ), 4)
uop_501 = relay.log10(bop_478.astype('float64')) # shape=(4, 4)
output = relay.Tuple([uop_493,call_496,var_497,var_498,uop_501,])
output2 = relay.Tuple([uop_493,call_499,var_497,var_498,uop_501,])
func_503 = relay.Function([var_477,var_497,var_498,], output)
mod['func_503'] = func_503
mod = relay.transform.InferType()(mod)
var_504 = relay.var("var_504", dtype = "float64", shape = (4, 4))#candidate|504|(4, 4)|var|float64
var_505 = relay.var("var_505", dtype = "int32", shape = (156,))#candidate|505|(156,)|var|int32
var_506 = relay.var("var_506", dtype = "float32", shape = (3, 3))#candidate|506|(3, 3)|var|float32
output = func_503(var_504,var_505,var_506,)
func_507 = relay.Function([var_504,var_505,var_506,], output)
mutated_mod['func_507'] = func_507
mutated_mod = relay.transform.InferType()(mutated_mod)
const_527 = relay.const([[[-9,2,-6,-6,-3,5,3,3,8,-10,7,2,-7,-2,2,-5],[-8,8,5,-8,-6,8,-2,-1,-4,-3,-4,-4,5,-1,3,-7],[-3,9,-6,-4,10,2,-2,2,-7,-9,-1,-1,7,3,-10,-3],[5,8,-8,4,6,-10,2,-6,-7,7,6,-6,3,-6,-10,-3]],[[10,-4,6,6,5,-2,-6,-5,3,-10,-1,-5,6,7,-10,-8],[-2,1,-2,3,1,-6,4,-1,-3,-10,-7,-1,6,-8,-8,9],[3,-9,-6,-4,-8,5,2,-10,7,1,7,-9,-10,1,7,8],[-1,5,-9,-6,9,-10,-7,8,-4,-2,-3,-4,-1,-8,7,6]],[[-5,7,10,-3,-7,2,7,10,8,-2,4,-3,1,3,9,7],[-2,8,-8,-3,-7,-10,4,6,3,-5,4,-2,-9,-2,-4,-10],[3,-8,-10,7,7,-4,7,-5,7,-7,-7,-5,-4,3,-6,-4],[-3,-2,-5,-2,8,5,7,-8,-1,-5,-6,3,9,-9,10,-6]],[[6,-8,8,-5,-8,-2,10,1,-4,8,-9,9,-10,3,-2,5],[-6,10,4,10,6,9,1,7,-6,-5,-9,-1,3,-9,2,4],[6,3,9,-4,-5,-1,4,-2,-5,7,3,3,3,-10,5,-4],[8,-2,9,4,-10,-10,4,-7,1,3,4,-10,7,2,4,2]],[[1,10,10,6,8,-8,-2,-3,1,-4,1,9,-4,1,-5,3],[-7,-8,9,-10,-3,-1,-3,5,10,6,5,-3,-6,-10,-7,9],[-6,-2,-4,-3,-1,10,-1,-6,8,4,9,2,-10,-8,-5,-3],[-5,-7,-7,6,7,2,4,-8,-1,-4,10,6,10,8,-3,8]],[[8,-7,-8,-5,-2,4,4,-6,6,7,-10,5,7,5,6,7],[1,-1,10,5,-10,6,2,-10,6,6,2,-3,-7,-10,4,-9],[3,3,3,-5,-8,8,5,-10,-7,6,-8,-8,-9,-9,6,-5],[7,4,7,-4,2,-5,-4,-7,-1,-2,-2,-5,-1,-4,-4,7]],[[9,3,3,-9,1,-2,-10,6,4,4,10,-8,6,7,-3,3],[-8,-7,4,-3,-3,10,6,7,-5,7,-2,4,7,10,-8,1],[9,-6,-6,-8,6,4,-5,3,2,5,4,-2,3,-2,-6,8],[3,-10,4,9,9,3,4,4,-5,10,10,-2,-7,-7,-9,10]],[[-8,-2,2,2,7,-5,-3,-9,-8,10,2,-3,6,3,5,-9],[6,5,7,1,-5,4,-5,-6,1,-3,8,3,-7,3,-10,5],[6,-5,-1,6,1,6,3,9,8,9,9,3,-8,8,-7,-3],[-4,5,-5,3,-10,-1,9,-9,3,6,-1,8,3,7,-8,4]],[[-1,9,-9,1,9,2,-2,-8,8,7,-3,9,10,8,8,-1],[-3,4,-5,9,7,-4,10,-5,8,-10,-8,6,6,-7,-8,-5],[3,10,4,-1,-9,-3,-9,-9,-2,-5,-9,10,7,-2,-10,-3],[-6,-10,-2,-5,-8,-2,-7,-3,2,-9,10,-8,7,-9,5,-9]],[[-5,-5,-9,-8,9,1,2,2,3,7,-1,10,5,4,-3,2],[9,5,-10,-5,4,2,2,-4,-7,8,-7,5,-10,-1,1,3],[3,-10,-2,1,-10,3,-8,5,-9,-3,-3,-8,5,-8,-5,6],[10,2,-8,-9,5,-10,-1,10,10,-6,-2,-10,2,-3,10,6]]], dtype = "uint32")#candidate|527|(10, 4, 16)|const|uint32
var_528 = relay.var("var_528", dtype = "uint32", shape = (10, 4, 16))#candidate|528|(10, 4, 16)|var|uint32
bop_529 = relay.greater_equal(const_527.astype('bool'), relay.reshape(var_528.astype('bool'), relay.shape_of(const_527))) # shape=(10, 4, 16)
bop_532 = relay.logical_xor(var_528.astype('int16'), relay.reshape(bop_529.astype('int16'), relay.shape_of(var_528))) # shape=(10, 4, 16)
func_384_call = mod.get_global_var('func_384')
func_391_call = mutated_mod.get_global_var('func_391')
const_536 = relay.const([-6.233201,-0.968225,3.920484,8.266633,0.359097,-1.608571,-8.834922,-3.997985,2.152571,8.077491,-4.495385,-7.972951,-7.203695,0.194106,-5.610855,1.648735,4.009560,-0.458045,7.235563,6.111355,-4.459241,-5.427917,6.511244,4.411790,7.358210,-5.375785,-8.123521,-8.280654,-1.349497,-0.242781,-0.659842,0.466743,6.459367,-7.387034,-5.193253,5.774602,5.941323,1.469047,5.219933,5.270506,-0.144784,-6.352978,-4.423113,8.214261,0.671886,8.515063,-8.277352,9.320392,0.175041,-5.065056,-0.834576,-7.227502,-7.119521,-4.449406,0.312762,-1.388312,2.587444,-0.884013,5.430266,0.648445,9.613883,-8.287529,5.133972,9.484793,4.023233,-1.002414,2.271035,6.542218,0.999919,0.202491,-5.337527,-4.420551,2.037441,5.764201,-8.770637,9.474577,-8.936167,1.287868,9.547957,-9.893819,2.329277,-4.357579,0.866553,-6.234366,-6.661291,0.378749,9.026598,-9.480198,-6.567774,-5.532541,-4.321977,-3.117403,5.988339,1.621402,-8.291703,-3.360497,-3.139016,0.828733,-1.230961,-6.695722,0.224611,7.454529,2.886395,9.753431,-4.124279,1.067271,0.463466,8.474400,3.740182,-7.674366,-9.012428,9.756543,9.258109,-6.480691,-2.534433,-5.281576,9.624785,2.477114,-6.001471,-9.455539,-1.071789,7.556246,-0.217136,-5.033042,3.448723,5.443890,0.168096,0.660889,5.654408,2.045124,-4.413551,-4.049849,2.665137,6.780250,-4.931881,-0.262710,8.873553,8.139533,-0.759246,4.220936,-6.592283,4.046914,-0.735228,-3.713535,7.749639,1.030032,-8.685032,4.853389,0.484837,6.452083,-3.093770,-5.514882,2.938429,1.187470,-1.129843,-2.490978,-3.823109,-8.623422,-1.985656,-0.971435,-8.471670,3.415974,4.954122,2.562703,8.028990,4.305163,-2.703785,8.746086,4.843689,-4.772870,-1.696959,-9.867297,-7.455158,5.571717,8.187284,-4.430619,8.407570,-8.438767,-7.822901,-2.323075,-0.214039,-4.574957,3.663074,8.408944,-4.433425,-5.970762,-0.684633,0.703622,-0.763337,4.111431,7.201429,8.277144,9.426415,3.539069,7.757695,-0.984613,3.587813,5.710214,-5.157160,5.875026,-0.696699,6.696501,-8.501287,0.784019,8.687324,5.553311,5.379066,-3.965462,-6.311465,-0.534348,0.833457,-1.658647,3.226295,0.307943,-4.813405,6.979159,9.146980,4.782431,7.685009,-7.999057,8.070195,1.867950,-6.418907,9.968032,-8.563467,2.381719,-3.258934,-3.182930,0.468606,-0.703296,7.046424,-1.459601,-1.142685,-8.328840,-2.389232,0.228089,-3.012360,-3.669424,-7.096583,-5.581879,-7.352031,-0.273244,0.186010,-8.155567,-4.073099,-1.733391,-0.113847,-2.047164,-6.486765,2.707017,8.069446,-8.822986,8.811096,4.676595,-1.293376,-8.806210,5.955741,-6.448658,-0.295353,8.188463,3.273951,-0.898496,-6.621472,-9.285257,6.663695,9.897594,-9.473409,3.315833,0.389097,4.808849,-2.190332,-1.292961,-3.399016,-5.621151,1.436429,-8.395025,5.752457,9.933532,6.663278,-7.637576,4.292869,-5.901617,3.575688,7.786438,-7.942586,-6.689099,7.041070,-6.742170,8.591037,-7.352024,-0.071646,8.211678,5.915346,3.065328,9.168700,9.811967,9.294428,-5.870573,6.533757,2.000096,2.259710,-5.252348,-4.825604,-0.500447,8.361965,2.927769,8.870815,0.329207,8.099242,-4.632666,-3.890970,8.567234,5.406027,-2.559416,5.640081,-9.397820,-0.141578,8.305280,6.854074,-0.723879,-4.095681,-8.689628,2.439225,-9.004000,-0.176612,2.785190,7.425643,-8.809507,-4.703045,7.884073,9.837915,-7.496100,7.491890,-1.493935,-8.055493,3.734820,-5.294268,-8.394632,-6.655624,-2.318684,1.974360,0.854811,-1.703319,0.629081,3.120025,0.729826,-7.023117,-4.609053,4.216515,-4.643728,8.627416,3.592125,2.574806,6.881485,-5.884260,-0.288062,5.587107,-8.742017,-7.618961,2.106940,3.455445,-7.930925,-4.511476,1.060101,3.961918,4.615083,0.730457,6.799011,6.595983,-0.197056,-2.874377,2.440882,5.611002,-0.796204,-3.548461,0.830811,-3.162571,4.645724,-2.169764,-0.162364,3.349933,-3.015348,-3.362741,1.413648,0.569906,-7.129313,0.263756,8.409409,6.883235,-7.561590,4.487794,1.127781,4.646766,8.353096,4.642844,9.792872,3.472505,-1.298167,9.421201,6.850329,6.904085,2.487573,-6.712925,9.387022,-1.149325,9.605508,7.756795,7.488311,4.291499,-0.789044,-1.646541,0.849211,-4.342823,-5.860938,-9.487192,-5.038992,2.454604,-0.985001,-9.385000,0.198849,3.481148,7.384350,-2.973236,1.672155,4.321653,-3.438728,4.769273,0.269030,9.249372,-8.576768,-3.653904,-7.464507,-9.635799,0.508481,-4.595483,6.040602,8.421459,3.865285,3.533115,8.816631,-3.245704,-0.645489,-5.347691,2.284735,7.540639,9.627273,-2.694063,6.560215,-4.652626,-8.478925,-2.243334,-6.557027,-9.935831,4.286975,-7.226270,-9.783094,-6.360039,-7.993010,-0.354017,5.344989,0.870789,2.884093,-8.051476,-0.908587,0.184377,8.064897,-9.770943,9.254103,3.094396,-5.855787,-2.235569,9.728899,-0.043502,9.374325,9.604435,-5.888776,3.098193,2.091382,0.491380,-1.303477,-0.304054,-4.988443,1.365491,4.169692,-3.318138,-5.865640,-2.204090,0.014280,7.942326,9.608404,1.052145,-6.225406,-6.260350,-9.009852,8.542120,-6.979804,4.519849,3.476191,0.024844,3.724128,-3.210404,-5.776146,0.067832,3.162440,-4.637770,-1.624426,8.478518,-2.704977,-7.082137,5.614688,6.141547,0.889332,-2.644392,7.501605,5.711151,7.589054,-3.313286,-2.118122,7.239018,-6.544722,-3.049727,-1.202394,3.532681,3.156438,4.940417,-1.347658,1.535814,7.872584,-0.573258,1.884629,-2.754451,3.390054,-7.736800,-7.923215,-4.918012,7.130005,4.364226,-9.365683,-1.527441,-8.675620,-3.527588,-1.596477,-0.606380,2.127066,-9.064503,-6.428090,0.301348,0.823653,6.170009,1.520648,-2.914927,2.239914,-4.929324,0.150121,1.612345,0.755895,-6.910615,9.239755,5.989852,1.236956,6.984944,-9.462320,2.090960,-5.657034,-5.378689,-3.738178,-6.866203,-2.817597,7.910459,2.165703,-5.169161,-4.056980,-0.365085,4.399710,-5.923551,0.038484,4.079758,-5.526240,4.320129,8.227875,5.530983,9.347221,0.420613,-2.991423,0.292048,-3.874799,-0.630826,-7.088596,-4.150102,7.022218,-5.850905,7.343106,-7.475898,-9.419895,-2.184450,3.614187,-2.397903,-9.888744,6.655143,-9.653582,-8.392798,-5.826000,-4.344694,-5.969483,-3.628477,6.578977,-8.880241,-7.706489,3.844904,9.918137,0.771786,9.015486,7.156749,9.531330,-3.446848,6.222169,-0.298857,-3.338759,-6.363734,-9.534106,-7.230475,6.444178,-3.409830,1.457983,-2.575426,8.743190,-8.435965,1.873474,-5.282167,-2.491679,5.634984,-8.340320,6.787900,-6.519207,1.164023,-5.433464,-7.052890,-6.429057,-3.127252,-1.925033,-3.027553,-8.276985,-9.480525,-3.766043,-8.110018,-4.103685,-0.437372,-3.504973,8.516938,5.672468,7.806560,-0.210000,0.923364,-3.533139,-6.184346,1.418325,-5.997344,3.454171,6.316542,-2.333159,-5.537226,3.301289,7.779755,-6.633996,-7.482364,8.439856,-0.666210,-8.107542,-5.651959,7.570855,4.815490,-9.902795,-4.645236,8.682927,5.767613,1.111131,-8.767475,2.616641,-0.070049,-6.492925,5.488050,-7.127431,1.065732,-8.279656,6.686435,-0.054688,-2.634541,4.119627,2.924866,-0.819623,-1.089826,-1.382673,3.687580,-8.402214,-2.162583,-8.129746,-0.955388,3.158077,1.153053,7.733554,9.219431,-7.109593,-0.825534,-2.759077,-4.789126,5.938679,-1.066954,1.828474,2.615301,-6.116991,-2.454447,2.007042,9.990663,-5.793484,-7.722397,-6.269206,2.579286,5.903142,-5.586753,3.010543,7.502301,2.003581,0.945685,5.619065,-8.123844,-6.182430,-1.605432,3.586892], dtype = "float64")#candidate|536|(728,)|const|float64
call_535 = relay.TupleGetItem(func_384_call(relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), ), 0)
call_537 = relay.TupleGetItem(func_391_call(relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), relay.reshape(const_536.astype('float64'), [14, 4, 13]), ), 0)
func_440_call = mod.get_global_var('func_440')
func_444_call = mutated_mod.get_global_var('func_444')
const_541 = relay.const([-10,-5,4,7,7,10,1,10,-7,-7,10,9,-10,-7,2,8,10,8,-7,1,-8,-7,-2,-1,7,-4,-6,4,-3,-6,1,-6,7,-9,8,8,1,-6,-6,4,7,-2,-2,-8,9,9,-7,-5,6,1,-5,1,-8,-1,-6,-8,4,5,-8,2,-6,-1,3,-7,1,-4,-9,-2,-3,-9,-7,-1,7,-10,8,4,10,-7,1,-3,-2,-9,-3,-7,-4,-5,-3,6,4,-7,-8,6,-6,3,7,-2,-5,1,1,-4,8,8,6,3,9,-2,-9,-7,1,4,7,10,-6,10,1,-2,3,-10,5,-9,-3,3,1,-9,-9,-2,-9,-8,5,2,-2,-5,-6,4,-2,8,-3,-7,1,9,-1,8,-3,-4,-8,4,-1,-5,-8,10,1,5,-9,10,-10,-3], dtype = "int32")#candidate|541|(156,)|const|int32
var_542 = relay.var("var_542", dtype = "float32", shape = (9,))#candidate|542|(9,)|var|float32
call_540 = relay.TupleGetItem(func_440_call(relay.reshape(const_541.astype('int32'), [13, 12]), relay.reshape(var_542.astype('float32'), [9,]), ), 5)
call_543 = relay.TupleGetItem(func_444_call(relay.reshape(const_541.astype('int32'), [13, 12]), relay.reshape(var_542.astype('float32'), [9,]), ), 5)
output = relay.Tuple([bop_532,call_535,const_536,call_540,const_541,var_542,])
output2 = relay.Tuple([bop_532,call_537,const_536,call_543,const_541,var_542,])
func_556 = relay.Function([var_528,var_542,], output)
mod['func_556'] = func_556
mod = relay.transform.InferType()(mod)
mutated_mod['func_556'] = func_556
mutated_mod = relay.transform.InferType()(mutated_mod)
func_556_call = mutated_mod.get_global_var('func_556')
var_558 = relay.var("var_558", dtype = "uint32", shape = (10, 4, 16))#candidate|558|(10, 4, 16)|var|uint32
var_559 = relay.var("var_559", dtype = "float32", shape = (9,))#candidate|559|(9,)|var|float32
call_557 = func_556_call(var_558,var_559,)
output = call_557
func_560 = relay.Function([var_558,var_559,], output)
mutated_mod['func_560'] = func_560
mutated_mod = relay.transform.InferType()(mutated_mod)
var_648 = relay.var("var_648", dtype = "float64", shape = ())#candidate|648|()|var|float64
const_649 = relay.const([[[2.421761],[4.720235],[-2.741463]],[[-8.340346],[-1.380663],[7.728524]],[[1.302753],[-5.209031],[-5.367919]],[[7.581444],[-8.118039],[-4.228735]],[[6.659012],[7.029781],[8.309812]],[[-3.386558],[8.772681],[-9.563138]],[[-7.717507],[5.370018],[5.410581]],[[-1.294106],[-4.512027],[5.446165]],[[4.373506],[-6.726164],[-9.761567]]], dtype = "float64")#candidate|649|(9, 3, 1)|const|float64
bop_650 = relay.greater_equal(var_648.astype('bool'), const_649.astype('bool')) # shape=(9, 3, 1)
output = bop_650
output2 = bop_650
func_655 = relay.Function([var_648,], output)
mod['func_655'] = func_655
mod = relay.transform.InferType()(mod)
mutated_mod['func_655'] = func_655
mutated_mod = relay.transform.InferType()(mutated_mod)
var_656 = relay.var("var_656", dtype = "float64", shape = ())#candidate|656|()|var|float64
func_655_call = mutated_mod.get_global_var('func_655')
call_657 = func_655_call(var_656)
output = call_657
func_658 = relay.Function([var_656], output)
mutated_mod['func_658'] = func_658
mutated_mod = relay.transform.InferType()(mutated_mod)
var_679 = relay.var("var_679", dtype = "int64", shape = (14, 16, 5))#candidate|679|(14, 16, 5)|var|int64
var_680 = relay.var("var_680", dtype = "int64", shape = (14, 16, 5))#candidate|680|(14, 16, 5)|var|int64
bop_681 = relay.bitwise_or(var_679.astype('int64'), relay.reshape(var_680.astype('int64'), relay.shape_of(var_679))) # shape=(14, 16, 5)
output = relay.Tuple([bop_681,])
output2 = relay.Tuple([bop_681,])
func_688 = relay.Function([var_679,var_680,], output)
mod['func_688'] = func_688
mod = relay.transform.InferType()(mod)
var_689 = relay.var("var_689", dtype = "int64", shape = (14, 16, 5))#candidate|689|(14, 16, 5)|var|int64
var_690 = relay.var("var_690", dtype = "int64", shape = (14, 16, 5))#candidate|690|(14, 16, 5)|var|int64
output = func_688(var_689,var_690,)
func_691 = relay.Function([var_689,var_690,], output)
mutated_mod['func_691'] = func_691
mutated_mod = relay.transform.InferType()(mutated_mod)
var_728 = relay.var("var_728", dtype = "float64", shape = (16, 15, 14))#candidate|728|(16, 15, 14)|var|float64
uop_729 = relay.asin(var_728.astype('float64')) # shape=(16, 15, 14)
output = relay.Tuple([uop_729,])
output2 = relay.Tuple([uop_729,])
func_731 = relay.Function([var_728,], output)
mod['func_731'] = func_731
mod = relay.transform.InferType()(mod)
var_732 = relay.var("var_732", dtype = "float64", shape = (16, 15, 14))#candidate|732|(16, 15, 14)|var|float64
output = func_731(var_732)
func_733 = relay.Function([var_732], output)
mutated_mod['func_733'] = func_733
mutated_mod = relay.transform.InferType()(mutated_mod)
var_771 = relay.var("var_771", dtype = "int64", shape = (10, 13, 1))#candidate|771|(10, 13, 1)|var|int64
var_772 = relay.var("var_772", dtype = "int64", shape = (10, 13, 10))#candidate|772|(10, 13, 10)|var|int64
bop_773 = relay.add(var_771.astype('int64'), var_772.astype('int64')) # shape=(10, 13, 10)
func_192_call = mod.get_global_var('func_192')
func_195_call = mutated_mod.get_global_var('func_195')
var_779 = relay.var("var_779", dtype = "float32", shape = (2, 2))#candidate|779|(2, 2)|var|float32
call_778 = relay.TupleGetItem(func_192_call(relay.reshape(var_779.astype('float32'), [1, 4])), 0)
call_780 = relay.TupleGetItem(func_195_call(relay.reshape(var_779.astype('float32'), [1, 4])), 0)
uop_782 = relay.asinh(bop_773.astype('float32')) # shape=(10, 13, 10)
uop_800 = relay.asin(uop_782.astype('float64')) # shape=(10, 13, 10)
bop_807 = relay.logical_or(uop_782.astype('bool'), relay.reshape(var_772.astype('bool'), relay.shape_of(uop_782))) # shape=(10, 13, 10)
bop_811 = relay.bitwise_or(uop_800.astype('int32'), relay.reshape(var_772.astype('int32'), relay.shape_of(uop_800))) # shape=(10, 13, 10)
bop_816 = relay.logical_xor(uop_800.astype('uint8'), relay.reshape(bop_773.astype('uint8'), relay.shape_of(uop_800))) # shape=(10, 13, 10)
bop_821 = relay.right_shift(bop_811.astype('uint8'), relay.reshape(var_772.astype('uint8'), relay.shape_of(bop_811))) # shape=(10, 13, 10)
uop_834 = relay.log(bop_821.astype('float64')) # shape=(10, 13, 10)
uop_847 = relay.cosh(uop_834.astype('float64')) # shape=(10, 13, 10)
uop_852 = relay.sin(uop_847.astype('float32')) # shape=(10, 13, 10)
func_731_call = mod.get_global_var('func_731')
func_733_call = mutated_mod.get_global_var('func_733')
var_855 = relay.var("var_855", dtype = "float64", shape = (140, 24))#candidate|855|(140, 24)|var|float64
call_854 = relay.TupleGetItem(func_731_call(relay.reshape(var_855.astype('float64'), [16, 15, 14])), 0)
call_856 = relay.TupleGetItem(func_733_call(relay.reshape(var_855.astype('float64'), [16, 15, 14])), 0)
bop_864 = relay.logical_and(uop_852.astype('bool'), relay.reshape(bop_807.astype('bool'), relay.shape_of(uop_852))) # shape=(10, 13, 10)
output = relay.Tuple([call_778,var_779,bop_816,call_854,var_855,bop_864,])
output2 = relay.Tuple([call_780,var_779,bop_816,call_856,var_855,bop_864,])
func_871 = relay.Function([var_771,var_772,var_779,var_855,], output)
mod['func_871'] = func_871
mod = relay.transform.InferType()(mod)
mutated_mod['func_871'] = func_871
mutated_mod = relay.transform.InferType()(mutated_mod)
func_871_call = mutated_mod.get_global_var('func_871')
var_873 = relay.var("var_873", dtype = "int64", shape = (10, 13, 1))#candidate|873|(10, 13, 1)|var|int64
var_874 = relay.var("var_874", dtype = "int64", shape = (10, 13, 10))#candidate|874|(10, 13, 10)|var|int64
var_875 = relay.var("var_875", dtype = "float32", shape = (2, 2))#candidate|875|(2, 2)|var|float32
var_876 = relay.var("var_876", dtype = "float64", shape = (140, 24))#candidate|876|(140, 24)|var|float64
call_872 = func_871_call(var_873,var_874,var_875,var_876,)
output = call_872
func_877 = relay.Function([var_873,var_874,var_875,var_876,], output)
mutated_mod['func_877'] = func_877
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1065 = relay.var("var_1065", dtype = "float64", shape = (9, 13, 12))#candidate|1065|(9, 13, 12)|var|float64
var_1066 = relay.var("var_1066", dtype = "float64", shape = (9, 13, 12))#candidate|1066|(9, 13, 12)|var|float64
bop_1067 = relay.less_equal(var_1065.astype('bool'), relay.reshape(var_1066.astype('bool'), relay.shape_of(var_1065))) # shape=(9, 13, 12)
bop_1070 = relay.bitwise_or(var_1065.astype('int16'), relay.reshape(bop_1067.astype('int16'), relay.shape_of(var_1065))) # shape=(9, 13, 12)
uop_1077 = relay.sin(bop_1067.astype('float64')) # shape=(9, 13, 12)
bop_1083 = relay.divide(uop_1077.astype('float64'), relay.reshape(bop_1067.astype('float64'), relay.shape_of(uop_1077))) # shape=(9, 13, 12)
output = relay.Tuple([bop_1070,bop_1083,])
output2 = relay.Tuple([bop_1070,bop_1083,])
func_1088 = relay.Function([var_1065,var_1066,], output)
mod['func_1088'] = func_1088
mod = relay.transform.InferType()(mod)
mutated_mod['func_1088'] = func_1088
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1088_call = mutated_mod.get_global_var('func_1088')
var_1090 = relay.var("var_1090", dtype = "float64", shape = (9, 13, 12))#candidate|1090|(9, 13, 12)|var|float64
var_1091 = relay.var("var_1091", dtype = "float64", shape = (9, 13, 12))#candidate|1091|(9, 13, 12)|var|float64
call_1089 = func_1088_call(var_1090,var_1091,)
output = call_1089
func_1092 = relay.Function([var_1090,var_1091,], output)
mutated_mod['func_1092'] = func_1092
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1126 = relay.var("var_1126", dtype = "float32", shape = (2, 14))#candidate|1126|(2, 14)|var|float32
uop_1127 = relay.erf(var_1126.astype('float32')) # shape=(2, 14)
output = relay.Tuple([uop_1127,])
output2 = relay.Tuple([uop_1127,])
func_1129 = relay.Function([var_1126,], output)
mod['func_1129'] = func_1129
mod = relay.transform.InferType()(mod)
mutated_mod['func_1129'] = func_1129
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1130 = relay.var("var_1130", dtype = "float32", shape = (2, 14))#candidate|1130|(2, 14)|var|float32
func_1129_call = mutated_mod.get_global_var('func_1129')
call_1131 = func_1129_call(var_1130)
output = call_1131
func_1132 = relay.Function([var_1130], output)
mutated_mod['func_1132'] = func_1132
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1290 = relay.var("var_1290", dtype = "float32", shape = (5, 16))#candidate|1290|(5, 16)|var|float32
uop_1291 = relay.sin(var_1290.astype('float32')) # shape=(5, 16)
func_305_call = mod.get_global_var('func_305')
func_310_call = mutated_mod.get_global_var('func_310')
const_1294 = relay.const([[8.600091],[0.954118],[-3.698403],[8.303555],[7.736256],[5.588478],[3.221638],[8.573229],[4.775817]], dtype = "float32")#candidate|1294|(9, 1)|const|float32
var_1295 = relay.var("var_1295", dtype = "float64", shape = (81,))#candidate|1295|(81,)|var|float64
call_1293 = relay.TupleGetItem(func_305_call(relay.reshape(const_1294.astype('float32'), [1, 9]), relay.reshape(var_1290.astype('float32'), [1, 80]), relay.reshape(var_1295.astype('float64'), [9, 9]), ), 2)
call_1296 = relay.TupleGetItem(func_310_call(relay.reshape(const_1294.astype('float32'), [1, 9]), relay.reshape(var_1290.astype('float32'), [1, 80]), relay.reshape(var_1295.astype('float64'), [9, 9]), ), 2)
bop_1297 = relay.greater(uop_1291.astype('bool'), relay.reshape(var_1290.astype('bool'), relay.shape_of(uop_1291))) # shape=(5, 16)
uop_1301 = relay.sigmoid(uop_1291.astype('float32')) # shape=(5, 16)
output = relay.Tuple([call_1293,const_1294,var_1295,bop_1297,uop_1301,])
output2 = relay.Tuple([call_1296,const_1294,var_1295,bop_1297,uop_1301,])
func_1303 = relay.Function([var_1290,var_1295,], output)
mod['func_1303'] = func_1303
mod = relay.transform.InferType()(mod)
var_1304 = relay.var("var_1304", dtype = "float32", shape = (5, 16))#candidate|1304|(5, 16)|var|float32
var_1305 = relay.var("var_1305", dtype = "float64", shape = (81,))#candidate|1305|(81,)|var|float64
output = func_1303(var_1304,var_1305,)
func_1306 = relay.Function([var_1304,var_1305,], output)
mutated_mod['func_1306'] = func_1306
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1348 = relay.var("var_1348", dtype = "float32", shape = (8, 7, 6))#candidate|1348|(8, 7, 6)|var|float32
uop_1349 = relay.atanh(var_1348.astype('float32')) # shape=(8, 7, 6)
func_1303_call = mod.get_global_var('func_1303')
func_1306_call = mutated_mod.get_global_var('func_1306')
const_1354 = relay.const([-7.669959,2.032484,3.024434,-3.640789,-0.649130,8.059853,-4.287253,-2.264284,-2.479922,-3.902518,0.767286,-1.131787,6.699565,1.637330,9.185337,4.944984,-1.025440,-4.747797,-1.195135,-7.066950,0.268873,-8.204044,9.164094,3.218361,-0.179644,4.966060,-5.172442,-7.813602,3.222897,-4.409635,4.420408,-6.731142,8.615278,1.756820,9.966862,-9.019498,3.969991,3.257153,-7.885608,6.239535,-0.806490,-4.181214,-5.707094,-4.880163,-5.216917,-9.435980,8.571549,3.386457,2.753453,-9.553007,-7.024814,-2.319777,7.567293,-5.357513,-0.096733,-2.488321,-2.741889,3.114863,-1.303419,-0.714825,2.441300,-8.405471,2.710982,6.412714,-7.193152,-9.058907,-1.745121,6.353180,-6.433354,5.809996,3.712304,9.865888,6.329574,-6.544069,-9.250974,-5.374020,-1.030773,-8.169648,8.307002,-6.718380], dtype = "float32")#candidate|1354|(80,)|const|float32
const_1355 = relay.const([[-9.341881,-7.085876,5.923772,7.013502,8.617400,9.180623,-6.708174,9.848252,3.693272,3.197816,8.673587,4.205187,6.890610,-9.508641,8.206858,6.917431,4.193036,7.092298,8.513756,4.282888,3.938384,-6.450395,8.268069,3.640696,4.065697,-8.947392,9.207806,4.733100,0.407288,-3.591172,6.627773,-1.396321,9.516688,0.867675,-4.030547,-5.365840,-8.107377,-8.912281,-7.884909,-3.521421,-8.963434,-9.687709,8.985564,-2.197460,-1.510572,1.385850,-5.765648,-1.950069,6.741672,-1.077400,5.989449,-4.461670,-7.083100,0.310507,-9.077829,-6.273262,-9.733650,-0.786775,-2.678580,-3.348609,8.351154,4.851811,8.900853,-5.062089,2.574816,4.773925,4.706602,-3.664113,-1.884266,-6.118392,-2.288863,7.314147,6.251247,-7.030026,6.168240,-9.579426,-9.317926,-4.700835,-0.920698,-7.206701,-7.882209]], dtype = "float64")#candidate|1355|(1, 81)|const|float64
call_1353 = relay.TupleGetItem(func_1303_call(relay.reshape(const_1354.astype('float32'), [5, 16]), relay.reshape(const_1355.astype('float64'), [81,]), ), 0)
call_1356 = relay.TupleGetItem(func_1306_call(relay.reshape(const_1354.astype('float32'), [5, 16]), relay.reshape(const_1355.astype('float64'), [81,]), ), 0)
bop_1357 = relay.less(uop_1349.astype('bool'), relay.reshape(var_1348.astype('bool'), relay.shape_of(uop_1349))) # shape=(8, 7, 6)
output = relay.Tuple([call_1353,const_1354,const_1355,bop_1357,])
output2 = relay.Tuple([call_1356,const_1354,const_1355,bop_1357,])
func_1368 = relay.Function([var_1348,], output)
mod['func_1368'] = func_1368
mod = relay.transform.InferType()(mod)
var_1369 = relay.var("var_1369", dtype = "float32", shape = (8, 7, 6))#candidate|1369|(8, 7, 6)|var|float32
output = func_1368(var_1369)
func_1370 = relay.Function([var_1369], output)
mutated_mod['func_1370'] = func_1370
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1383 = relay.var("var_1383", dtype = "uint64", shape = (4, 14, 16))#candidate|1383|(4, 14, 16)|var|uint64
var_1384 = relay.var("var_1384", dtype = "uint64", shape = (4, 14, 16))#candidate|1384|(4, 14, 16)|var|uint64
bop_1385 = relay.greater(var_1383.astype('bool'), relay.reshape(var_1384.astype('bool'), relay.shape_of(var_1383))) # shape=(4, 14, 16)
output = bop_1385
output2 = bop_1385
func_1409 = relay.Function([var_1383,var_1384,], output)
mod['func_1409'] = func_1409
mod = relay.transform.InferType()(mod)
mutated_mod['func_1409'] = func_1409
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1409_call = mutated_mod.get_global_var('func_1409')
var_1411 = relay.var("var_1411", dtype = "uint64", shape = (4, 14, 16))#candidate|1411|(4, 14, 16)|var|uint64
var_1412 = relay.var("var_1412", dtype = "uint64", shape = (4, 14, 16))#candidate|1412|(4, 14, 16)|var|uint64
call_1410 = func_1409_call(var_1411,var_1412,)
output = call_1410
func_1413 = relay.Function([var_1411,var_1412,], output)
mutated_mod['func_1413'] = func_1413
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1459 = relay.var("var_1459", dtype = "float32", shape = (15, 11, 16))#candidate|1459|(15, 11, 16)|var|float32
uop_1460 = relay.cosh(var_1459.astype('float32')) # shape=(15, 11, 16)
func_688_call = mod.get_global_var('func_688')
func_691_call = mutated_mod.get_global_var('func_691')
const_1463 = relay.const([-2,4,4,4,8,-1,-4,5,3,-4,-1,-5,2,2,10,-8,8,8,3,8,-7,8,6,-9,-5,7,4,5,-4,9,-4,10,-8,4,8,3,-10,7,6,4,3,-9,3,-2,-3,-5,-10,7,-1,9,2,7,3,-8,9,2,5,-8,7,7,1,-5,7,8,4,9,7,9,-9,-3,-1,2,-8,4,10,-3,5,1,3,-10,-7,4,5,8,-2,-2,-5,9,-5,-9,6,6,-4,-5,-8,-6,6,-9,-5,6,-7,10,-5,7,4,4,2,5,-9,-2,5,-2,-8,-3,9,2,-10,-9,-3,-4,2,9,10,-9,4,5,-9,-2,1,2,7,-5,-6,1,-1,-8,-2,9,3,-10,1,-6,-4,7,-4,-5,-3,-7,7,10,-2,2,-5,6,1,3,-3,2,7,-3,10,-1,10,-4,5,-6,-10,1,-1,3,4,6,5,5,9,5,7,2,6,-4,-10,8,-1,5,-7,-6,6,-10,-9,-5,4,8,7,-9,-8,2,5,7,-8,-8,3,-9,5,10,-6,-2,6,-2,-2,6,3,-7,7,-2,8,-3,3,2,1,2,8,8,8,6,8,-9,9,9,1,4,-3,-2,9,4,5,-9,-2,4,-6,9,2,7,5,9,9,-5,10,-7,-1,-1,7,-9,5,-8,-6,5,4,-9,-7,-2,10,9,7,9,8,-6,9,3,-9,-5,7,-2,1,9,10,6,-1,1,2,-7,1,-5,-1,2,-1,-4,8,-1,-7,-9,-4,-8,4,6,-1,-6,8,-1,-2,-8,2,3,-10,-1,9,-4,6,9,-3,-6,9,5,5,-6,-2,10,-8,6,-1,5,-4,2,-2,2,5,-8,-9,-4,2,1,-6,-8,8,-4,-8,7,-9,5,-10,-7,-4,1,-1,-10,-7,-4,4,9,5,-10,-2,8,6,5,8,-10,8,-8,6,-1,-8,8,-9,5,10,-9,-2,6,-8,-3,2,1,-5,-8,-4,6,3,6,-10,-6,-8,-3,-1,8,9,-8,-7,8,5,-4,4,-2,3,9,-9,-2,-1,7,5,1,-9,2,7,7,-7,-2,-1,9,2,-7,-4,-10,-5,7,4,-1,-9,8,-2,-10,6,-1,-6,-10,10,-1,-6,1,8,9,-8,-5,-3,4,10,6,8,2,8,2,1,2,2,9,1,-2,-10,7,1,-1,-7,-6,4,9,-4,-10,4,-2,-5,-6,3,-7,-6,8,-6,-10,-8,10,-6,4,9,5,-7,-10,10,-1,6,7,2,3,6,3,-10,6,1,6,-5,3,-7,-8,-1,-1,-5,-10,2,5,7,-2,2,-4,7,4,2,5,2,-8,2,-7,6,4,-1,7,-10,4,4,-4,10,2,5,1,3,-8,9,-2,5,4,-6,1,-10,7,-2,7,-6,4,-8,-9,-6,-4,2,-1,-1,-2,-6,4,-9,1,1,-5,-7,-10,-6,6,4,9,7,5,-5,-8,10,-8,-4,-10,5,3,6,8,10,6,-5,2,-6,-9,-5,-5,-10,4,2,-8,1,10,3,-9,-2,-9,-1,7,9,2,5,-9,-10,9,-8,8,8,-7,6,6,-3,4,-9,5,9,1,7,-10,-4,-9,-6,4,-7,-7,-8,-10,1,9,5,7,3,1,-10,7,10,6,4,-1,2,-5,-2,-5,9,2,-10,-1,9,-1,2,2,-3,-7,-8,-8,-6,-3,8,-3,4,4,-6,-6,-6,3,4,-10,-2,6,5,4,-6,2,-7,-4,4,-4,4,5,-7,-8,2,10,6,7,9,-5,-8,-1,-3,7,-3,-8,-1,7,-1,-10,4,8,5,1,5,-9,4,8,-4,-7,-6,-4,-9,-4,5,-3,6,-10,5,5,9,2,5,-3,5,-8,6,-9,-2,-5,-8,7,-8,-3,10,2,5,5,4,-10,9,-1,9,-9,-2,7,1,-4,-3,-1,-6,-2,-4,10,4,-8,-8,-7,-10,6,-1,4,1,10,-4,6,-1,-10,7,5,-9,-6,2,4,2,-6,4,-4,5,-8,-4,-8,7,1,5,9,9,-1,-1,-4,4,7,-9,5,-6,4,9,6,-7,4,-5,8,-8,-8,3,-3,4,-6,-2,9,-4,-7,-5,3,-2,-4,9,2,10,-8,1,-9,-3,10,-6,3,-1,6,-5,-8,5,-5,-5,-4,8,9,5,-4,4,4,-2,3,9,-1,-10,3,-2,-2,9,1,5,-7,-9,9,-7,-3,-1,-6,6,7,6,-1,-10,7,-8,7,10,4,-4,-1,7,2,10,10,-5,4,-9,-2,-10,10,-7,1,-10,-5,9,7,-1,8,9,1,-5,7,-10,4,-8,-5,-3,2,-3,-8,-4,2,6,-1,6,-8,5,-8,4,4,-8,7,-7,-5,-8,5,-9,-9,7,-1,10,-5,-7,-1,-6,-3,5,-8,-8,5,7,10,-8,-3,9,-7,7,7,6,-5,2,-9,-7,-9,9,3,-3,3,3,6,2,-4,-7,1,8,6,-3,7,3,6,-2,-7,5,-6,-6,2,-6,-4,-9,6,2,-7,8,-6,-2,2,-5,-8,-9,-9,-1,-2,9,-9,5,-4,-9,2,9,-10,-2,6,1,8,-10,-4,7,-7,-4,-5,-6,-5,-1,6,-7,-10,7,6,2,10,10,5,10,6,-2,4,-9,5,-6,-2,5,3,10,2,5,3,5,-5,-10,10,1,10,-3,-1,-3,-3,5,-2,7,-6,-7,-10,-5,-7,7,6,3,9,10,4,-3,2,3,4,2,-2,6,-9,-8,-7,8,-2,-1,5,-1,5,-6,-5,-4,10,-6,3,6,-7,2,7,9,-1,10,-4,-6,6,-7,-2,-7,-5,5,-9,3,5,-9,-5,2,1,-1,1,-3,1,6,-10,-6,-8,-6,1,-2,-3,9,-1,4,-2,-8,-4,2,2,-1,-2,9,9,8,5,6,-1,5,-7,4,6,2,3,-9,1,9,-6,-3,4,-5,-1], dtype = "int64")#candidate|1463|(1120,)|const|int64
call_1462 = relay.TupleGetItem(func_688_call(relay.reshape(const_1463.astype('int64'), [14, 16, 5]), relay.reshape(const_1463.astype('int64'), [14, 16, 5]), ), 0)
call_1464 = relay.TupleGetItem(func_691_call(relay.reshape(const_1463.astype('int64'), [14, 16, 5]), relay.reshape(const_1463.astype('int64'), [14, 16, 5]), ), 0)
func_1303_call = mod.get_global_var('func_1303')
func_1306_call = mutated_mod.get_global_var('func_1306')
const_1472 = relay.const([-0.504043,-5.954572,-2.936679,-9.021337,-0.343247,1.554073,2.018747,-6.799615,3.949609,4.336137,1.707055,6.264373,-4.272249,-9.036217,-5.032001,-9.849690,1.071501,-9.367588,0.988253,-2.057949,8.709525,1.417635,5.961418,9.761990,-1.794156,7.746216,3.146235,-4.941682,-5.799938,-6.045291,-6.238599,7.792735,1.125765,0.245684,-4.583397,-5.104036,-6.262547,6.240354,-4.828305,-0.473617,1.699306,-9.416256,9.667214,-6.227224,4.678898,-1.498699,4.093662,0.167315,2.755832,9.687482,4.528296,-9.657975,3.922933,6.672934,-7.281013,-8.451661,6.045332,-9.547417,-2.435610,-9.459352,1.868928,2.645072,-0.242284,7.427871,-5.811078,9.695636,8.834526,7.655155,7.950063,-3.641152,-1.937518,9.073684,5.854042,-0.259784,-9.757762,9.621047,6.758565,3.554760,-2.031253,-7.445404], dtype = "float32")#candidate|1472|(80,)|const|float32
const_1473 = relay.const([[-4.218624,-0.395200,-9.699033,8.359616,-8.441757,2.103402,-8.307252,6.398137,-6.003645],[3.759485,-2.383027,-3.257773,-4.799148,-3.163698,-5.398913,-9.983496,6.151662,3.332206],[6.841008,8.723083,0.433364,1.884411,0.456287,-4.188590,0.755101,6.129540,8.291194],[-0.408045,3.865276,-6.515776,-0.092577,-3.146667,7.369147,9.607129,6.275147,-5.802630],[-9.627156,-7.241316,-9.023763,1.332744,-6.233388,3.428636,-1.347490,1.548553,-8.821018],[-9.006977,-0.279628,-4.993196,-9.523236,4.707611,8.935731,4.881862,5.912386,0.288975],[-2.338798,1.434173,3.814722,-8.528941,3.921005,5.303991,8.729846,-9.388592,3.183083],[9.633939,9.998359,3.897093,-0.175716,-7.603592,-3.191780,4.755481,-9.881573,-4.796477],[6.655707,5.496002,-5.601639,-9.750386,-0.397515,-7.821340,-7.954542,-5.873788,-0.524835]], dtype = "float64")#candidate|1473|(9, 9)|const|float64
call_1471 = relay.TupleGetItem(func_1303_call(relay.reshape(const_1472.astype('float32'), [5, 16]), relay.reshape(const_1473.astype('float64'), [81,]), ), 2)
call_1474 = relay.TupleGetItem(func_1306_call(relay.reshape(const_1472.astype('float32'), [5, 16]), relay.reshape(const_1473.astype('float64'), [81,]), ), 2)
output = relay.Tuple([uop_1460,call_1462,const_1463,call_1471,const_1472,const_1473,])
output2 = relay.Tuple([uop_1460,call_1464,const_1463,call_1474,const_1472,const_1473,])
func_1475 = relay.Function([var_1459,], output)
mod['func_1475'] = func_1475
mod = relay.transform.InferType()(mod)
var_1476 = relay.var("var_1476", dtype = "float32", shape = (15, 11, 16))#candidate|1476|(15, 11, 16)|var|float32
output = func_1475(var_1476)
func_1477 = relay.Function([var_1476], output)
mutated_mod['func_1477'] = func_1477
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1530 = relay.var("var_1530", dtype = "uint8", shape = (1, 13, 14))#candidate|1530|(1, 13, 14)|var|uint8
var_1531 = relay.var("var_1531", dtype = "uint8", shape = (12, 13, 14))#candidate|1531|(12, 13, 14)|var|uint8
bop_1532 = relay.bitwise_and(var_1530.astype('uint8'), var_1531.astype('uint8')) # shape=(12, 13, 14)
output = bop_1532
output2 = bop_1532
func_1540 = relay.Function([var_1530,var_1531,], output)
mod['func_1540'] = func_1540
mod = relay.transform.InferType()(mod)
var_1541 = relay.var("var_1541", dtype = "uint8", shape = (1, 13, 14))#candidate|1541|(1, 13, 14)|var|uint8
var_1542 = relay.var("var_1542", dtype = "uint8", shape = (12, 13, 14))#candidate|1542|(12, 13, 14)|var|uint8
output = func_1540(var_1541,var_1542,)
func_1543 = relay.Function([var_1541,var_1542,], output)
mutated_mod['func_1543'] = func_1543
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1554 = relay.var("var_1554", dtype = "float64", shape = (3, 11))#candidate|1554|(3, 11)|var|float64
uop_1555 = relay.asinh(var_1554.astype('float64')) # shape=(3, 11)
output = relay.Tuple([uop_1555,])
output2 = relay.Tuple([uop_1555,])
func_1567 = relay.Function([var_1554,], output)
mod['func_1567'] = func_1567
mod = relay.transform.InferType()(mod)
mutated_mod['func_1567'] = func_1567
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1568 = relay.var("var_1568", dtype = "float64", shape = (3, 11))#candidate|1568|(3, 11)|var|float64
func_1567_call = mutated_mod.get_global_var('func_1567')
call_1569 = func_1567_call(var_1568)
output = call_1569
func_1570 = relay.Function([var_1568], output)
mutated_mod['func_1570'] = func_1570
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1574 = relay.var("var_1574", dtype = "uint64", shape = ())#candidate|1574|()|var|uint64
const_1575 = relay.const([[-5,1,4,-8,-4,-10,6,10],[-5,-2,-4,6,-3,-8,1,10],[4,7,-9,-9,-7,-6,5,3],[-4,4,-8,-7,7,-9,3,6],[1,7,8,3,-7,8,-2,-3],[6,4,8,-4,7,3,3,9],[-1,-3,2,-4,-4,-2,3,-5],[-5,-10,-10,-5,-8,-9,8,-4],[-4,-7,3,-7,6,-5,7,8],[9,-2,-4,6,-7,-4,2,-3],[-5,5,4,6,1,8,1,7],[-6,7,-10,-2,9,-10,-4,-9],[3,-10,6,-6,-7,-3,10,8],[-3,-6,7,7,-3,-9,2,-7]], dtype = "uint64")#candidate|1575|(14, 8)|const|uint64
bop_1576 = relay.less_equal(var_1574.astype('bool'), const_1575.astype('bool')) # shape=(14, 8)
output = relay.Tuple([bop_1576,])
output2 = relay.Tuple([bop_1576,])
func_1591 = relay.Function([var_1574,], output)
mod['func_1591'] = func_1591
mod = relay.transform.InferType()(mod)
var_1592 = relay.var("var_1592", dtype = "uint64", shape = ())#candidate|1592|()|var|uint64
output = func_1591(var_1592)
func_1593 = relay.Function([var_1592], output)
mutated_mod['func_1593'] = func_1593
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1686 = relay.const([[False,False,True,False,True,False,True,True,False,True,True,False,False,True],[True,False,True,False,True,True,True,False,True,False,False,True,True,True],[True,False,True,True,True,False,False,False,True,True,True,True,False,True],[False,True,False,True,True,True,False,False,True,True,False,False,False,True],[False,True,False,True,False,True,False,True,False,False,True,True,False,True],[False,False,False,False,True,False,False,True,True,False,True,False,True,False]], dtype = "bool")#candidate|1686|(6, 14)|const|bool
var_1687 = relay.var("var_1687", dtype = "bool", shape = (6, 14))#candidate|1687|(6, 14)|var|bool
bop_1688 = relay.logical_and(const_1686.astype('bool'), relay.reshape(var_1687.astype('bool'), relay.shape_of(const_1686))) # shape=(6, 14)
bop_1692 = relay.equal(var_1687.astype('bool'), relay.reshape(bop_1688.astype('bool'), relay.shape_of(var_1687))) # shape=(6, 14)
bop_1695 = relay.less(bop_1688.astype('bool'), relay.reshape(const_1686.astype('bool'), relay.shape_of(bop_1688))) # shape=(6, 14)
func_1475_call = mod.get_global_var('func_1475')
func_1477_call = mutated_mod.get_global_var('func_1477')
const_1700 = relay.const([[-7.775382,7.495844,8.853651,6.754606,-4.524511,2.768273,-8.375002,4.121049,-6.671828,-1.204265,3.423838,5.903137,-5.531442,1.297260,-6.511206,-6.358366,9.142098,4.781784,8.881295,2.314447,-6.493524,2.306707,-7.154241,-6.595659,3.698094,-1.441627,-9.571738,-3.585047,-3.147046,7.709007,3.787709,-4.342047,-4.889235,-5.192424,-1.310725,0.303224,-2.449399,-2.288616,6.854607,5.913482,7.094196,-9.190480,0.520373,-3.119750,-5.039990,-9.617881,-1.129636,-7.739396,4.384087,-9.877669,-5.430606,-6.449478,-7.365186,0.116508,-5.187387,0.847052,3.289180,8.517572,-6.707076,-6.319906,0.313783,-5.692827,1.586164,-7.822622,1.831436,7.809443,7.904580,1.428274,-4.186661,5.183381,6.736415,-3.878754,-1.775917,-0.369522,-8.548948,8.943337,6.916334,4.433758,-1.640923,6.153200,-9.719839,-9.250006,8.609901,8.888531,-4.417821,-9.348521,6.934017,-0.316538,-1.209300,-5.253582,-0.003878,-7.119440,-9.289683,-4.538141,-8.733684,-5.999577,-2.102002,-4.845190,-4.157432,1.440392,-9.636344,3.086578,4.223968,-7.643878,-7.573158,-5.012764,-7.264927,-2.033185,-7.037813,-0.617850,7.331610,1.417486,9.237062,6.344067,2.060445,-5.438496,-0.910351,-9.929302,-2.080604,9.571081,1.383836,5.892582,-6.110341,-7.628860,-8.875660,9.330841,-8.055913,-5.616417,-7.536647,2.544297,-9.927407,-9.961785,6.054326,-4.208987,2.830395,-4.033844,8.677273,-3.063647,-7.738980,-8.194078,2.228383,-5.845933,-5.012314,2.215387,-1.086325,5.528758,4.512785,3.258133,-8.000645,-8.801740,1.659912,-5.953041,-4.411504,-0.042863,-3.665900,7.202178,4.431788,1.096626,-2.688455,-4.478087,1.460816,-9.184444,9.191140,-2.560010,0.926471,8.374361,3.715250,-0.648186,4.310870,-8.157594,6.739882,2.752215,2.037989,3.666250,-6.220876,-6.512093,6.818596,-1.378054,-2.279584,-3.142285,8.572949,-9.439731,6.164668,-4.507397,-0.983500,-7.441329,-7.703864,-5.810531,-7.005314,-7.374425,0.339190,5.639794,1.652555,-1.949336,-6.454460,6.874427,-4.658868,-0.723747,-5.511483,-2.918123,-1.073532,8.068678,4.025623,-7.469224,4.531062,-3.720686,-6.161499,9.159181,5.589464,0.328690,-7.335577,4.821375,7.725980,-1.119932,-3.866023,-1.508430,-5.775873,1.675652,0.522325,9.680226],[-3.586406,6.082390,6.525540,-4.771582,-7.146143,-9.637422,0.492591,-2.760044,-3.766738,-6.186803,9.463498,8.141287,-4.189367,3.460387,-8.065070,-9.719745,-3.631336,2.593556,-9.979420,-6.163988,-2.593422,7.426348,-5.319593,-5.964395,0.552902,8.975953,7.218324,-6.792395,-7.147277,2.045595,-2.247985,3.374799,7.473310,3.482743,8.004919,4.505255,-2.682499,9.046278,-1.499352,8.052614,-4.149876,-5.268976,8.334042,-8.526790,6.497290,9.296290,9.505842,-3.329251,8.624192,2.691372,-5.088577,-6.429166,3.431946,7.977381,9.495757,8.219390,-3.441898,9.808129,-4.981658,6.454748,9.234005,-6.155765,5.960289,-5.253884,-9.958545,4.692616,-8.091726,5.974060,9.795591,2.742564,-6.862427,9.151118,-5.129442,-6.633422,-2.509354,-8.796243,-0.358768,-5.907765,-8.287623,9.431198,-5.479741,1.664560,-1.937861,1.563614,5.433679,6.997977,8.765806,-5.635593,4.282556,5.407530,-4.181760,-2.315952,-6.301096,0.905084,6.015108,8.659477,4.653465,-7.896500,8.362144,-7.001692,-2.656237,-3.283679,-7.375972,3.724130,-2.316564,3.780041,8.311225,0.706161,-3.824287,-0.588418,1.095399,-6.339981,-9.181068,6.592675,-6.724915,-0.304794,0.877052,-1.505584,-8.427712,1.205799,-5.201060,5.422180,-7.631949,0.546045,9.036346,-8.209257,2.356629,-1.613628,0.519045,-5.359803,9.139810,-2.380300,0.083782,-1.088637,2.587538,-4.549516,-8.208099,0.447524,5.633069,5.769498,-5.126812,4.407512,-4.559168,0.198917,1.711596,9.532347,-7.814495,1.528310,7.200832,0.002943,-4.087278,-1.751110,-5.655832,8.523005,2.233477,-5.541483,3.628731,7.011592,1.540334,8.329887,-3.548466,8.301225,5.490691,4.275738,-3.460283,4.712373,-0.203890,4.727662,8.684293,-1.108936,-0.431109,-4.111943,1.524048,0.655879,-5.371042,6.137885,-9.384610,9.856599,-6.968441,-8.140687,-7.228037,1.589138,5.583526,5.322888,0.838508,8.144889,-6.359923,9.912566,-7.300217,8.390310,-1.123059,7.889741,0.470724,-7.772319,2.110736,2.343878,-2.902191,5.161867,3.749905,-8.574949,-5.845843,-7.388956,7.861872,-7.274766,5.087321,-0.189500,9.273487,-0.638867,-5.361522,-1.985413,8.234269,8.837871,-7.302283,5.543860,1.307608,-9.894937,-9.534250,0.628335,4.625946,0.417403],[2.539625,-1.292204,2.327788,3.029521,3.544239,-9.474910,7.478409,-1.618814,-6.606107,-0.494782,6.910487,4.724720,8.623775,3.952094,8.044624,7.113620,-4.466633,0.309867,-4.914329,5.441074,-1.053344,-3.724589,-7.446074,3.888353,6.220163,-7.714549,-0.980001,-9.116335,-3.288394,-7.186258,-4.720301,-2.349216,6.148559,-8.526178,6.763281,2.649701,5.022204,2.083185,-8.302180,-2.684817,-4.048752,-6.447221,6.345863,7.689722,-1.731807,6.676870,3.301168,4.414687,3.109436,2.094197,-4.873681,-8.706631,-8.729992,1.288637,-4.135628,-0.113817,-2.959419,3.066765,4.315567,-7.811598,7.372242,-6.559066,4.276296,1.209706,-1.808060,9.073214,-0.032895,-1.296009,-3.520729,6.275557,-4.048571,1.903326,5.531862,-9.249411,7.463529,-6.951462,-8.174533,-8.002928,5.128790,-7.372180,-2.256021,-7.050912,5.396286,1.892979,-4.118615,-1.588485,1.123362,4.902722,-7.148921,8.180272,-5.872534,-4.138192,4.307321,0.711129,4.249786,-8.034852,7.862444,-5.563118,4.331656,-7.878107,-7.853845,-9.083626,0.350892,-3.307565,4.717852,1.796314,-6.623313,-0.371326,-0.569007,-9.910346,5.537351,9.997117,4.728045,1.115971,-7.163876,-8.547335,-8.936833,-7.717708,-4.922636,-2.902429,-4.309275,-0.588648,6.133319,-9.541598,-1.585729,2.601465,2.161952,-9.179061,-1.156137,-8.762812,-0.063382,-2.611621,1.536091,-3.513275,8.305809,-5.454167,-4.861290,-5.063689,2.554422,-0.470725,2.361866,-3.493900,-1.128924,2.552439,-7.061197,4.062279,-7.571321,-2.934419,-6.811004,-6.862052,5.315708,-8.833388,2.014470,3.399356,5.394048,-5.933255,-9.414193,-7.822584,3.004421,5.641652,9.256956,-2.497649,-3.271522,2.196806,9.666352,6.203121,-3.184616,5.097896,-7.877247,8.732217,-9.725165,4.560309,-9.878680,8.781698,1.779357,1.317032,-7.424386,4.619061,-7.204085,3.528805,-3.090737,-7.186543,-0.428239,-0.845220,7.444960,0.709252,8.163972,-8.117770,-8.152630,0.697837,-1.957036,-4.328008,8.965903,-5.215497,-9.519512,1.929874,-5.204408,-7.904327,0.150627,6.136424,-4.043627,-9.846960,4.603559,1.812981,7.750651,-7.967457,6.071221,-0.112549,-3.377010,3.296065,-7.335181,4.561087,-8.190601,-5.168347,-6.976917,-6.917902,-0.838641,-4.638472,3.943862,5.454564],[-6.972773,3.322597,-3.672109,-2.476090,-2.279607,2.484874,3.833738,8.185781,-7.079516,4.285794,6.145205,8.776157,5.582292,-2.071516,6.285388,7.333982,3.900216,-5.969978,-3.923381,-0.965920,-8.620143,1.654344,-7.173065,1.868145,-0.285551,0.278894,4.537066,5.689279,-5.323898,3.237714,2.428576,5.784153,7.783058,-1.601650,4.798269,-8.729352,4.552696,3.609164,-6.242305,-8.208290,7.672956,8.199937,3.890959,4.723725,-0.636941,2.026727,2.336330,-0.658436,-8.582390,-7.790526,7.812619,4.592139,4.678243,-4.869868,-6.426758,-6.711787,-7.596455,-2.389360,7.930489,9.257918,-2.086116,0.737022,9.904404,9.563655,5.151549,1.581380,-6.838203,6.170250,9.850030,-0.224121,-7.173348,-0.470395,-2.317644,-6.452151,-2.199365,5.720329,0.567727,9.329345,-9.157066,-7.501400,-8.125476,-2.549695,6.780870,-4.226036,1.753991,-0.907341,-4.484592,-8.090153,0.960687,-8.226554,-9.152933,-2.232640,-9.342460,-7.450305,-3.658264,-6.977515,7.694215,-7.656092,-3.822124,8.749968,9.851896,-9.055628,-3.204545,-2.808788,5.610982,-9.140356,-8.182226,9.844595,-1.565444,0.388484,2.546839,8.127113,-1.968317,2.158517,9.365957,-5.061290,-8.765002,-1.708161,8.364072,-1.367490,0.214401,-3.573037,7.477381,-1.370693,0.821759,7.144332,-1.079601,3.675611,-7.196271,4.291341,-8.695877,-5.082883,-1.954098,6.231780,5.619929,4.424623,-7.210848,-1.191590,8.444409,7.095288,-3.304896,0.303336,-3.588965,5.024759,1.432598,-6.936091,-5.268032,-7.157615,4.728149,-2.617828,1.515584,8.865968,2.514847,-5.837366,-5.283405,3.684828,-5.087384,7.922810,-6.534973,5.376827,-4.106924,6.060928,-5.331770,9.749854,1.547913,-7.317907,4.485861,5.508521,8.899036,-2.740479,6.033706,-9.907872,-0.249162,-7.239561,-8.696618,4.991521,9.170466,0.706044,-2.326631,9.465702,-9.929881,3.338181,0.052159,-6.038183,-7.416036,7.831652,-4.856086,6.268699,3.012527,7.690673,7.542839,-2.502358,1.364013,-7.835933,5.578864,-7.639822,3.734423,6.780363,-0.753541,3.244353,-0.819706,-4.309794,-2.221875,-4.244085,-2.144065,0.931068,-9.057938,-7.106984,2.913223,4.126478,4.591663,-8.605291,9.797175,5.529343,1.558310,-0.126569,0.813457,-0.168843,-8.441331,-5.158148],[4.017954,-1.055192,6.601554,6.093424,4.047629,7.026233,-7.601766,3.984138,-0.104438,1.874288,2.401491,1.713256,4.565813,-3.170032,-1.325894,-5.553099,-3.660301,7.626745,0.971663,-3.120928,-8.537127,6.764860,5.932487,-7.221674,-4.009593,0.423792,-0.915801,-7.387054,4.043088,1.148710,0.642959,1.817085,6.490679,-5.214575,0.018781,4.855828,-2.325288,3.708852,-7.834412,8.974633,8.341730,-3.506933,-3.168676,9.928833,-4.093604,8.319719,-0.961951,0.939824,-1.536548,9.460817,0.291456,1.092631,8.473511,8.317211,-0.974689,-2.662469,-5.350648,4.135497,-2.867918,-5.941387,-9.689277,-1.417018,-3.241684,-8.059175,0.454093,-5.039618,-5.432527,4.326589,-4.938989,1.341955,7.349637,1.311674,0.526814,-3.499804,6.456317,-6.905879,2.779503,6.429965,-6.013647,-7.087312,5.774403,-8.122727,-7.720383,5.527481,8.728017,5.258038,0.502583,-3.830096,-7.551277,-3.163415,6.315643,8.799961,4.283469,-7.567467,-4.738419,-5.785841,5.652575,2.301868,-3.528596,9.733511,-7.168596,-3.835708,4.513980,-9.572146,-5.343640,4.214800,4.269172,-3.520214,3.183906,-7.908332,0.695746,2.736015,-7.491353,-4.170145,-8.828597,-2.996757,-4.720304,-2.873367,0.817242,8.697171,7.294582,5.905842,8.813730,4.638702,-8.737878,-6.086239,-4.230510,-9.038852,0.733627,5.460345,-7.801827,-7.664395,4.849177,-9.266946,8.435215,-5.768826,4.735857,7.966908,8.026028,8.951152,-6.331170,0.887143,-1.437872,4.939732,-8.856414,1.734212,-1.802446,9.673175,-6.042609,-2.997822,-0.128674,-3.731741,-0.935824,-6.361229,-6.882314,1.970784,-8.017904,-2.386796,5.164508,-3.994699,-3.041037,-3.245560,3.880104,-6.736479,-9.422807,2.141913,5.168734,-3.142969,5.523947,7.934204,-2.204422,-9.279570,1.616953,8.547861,1.388857,-6.121519,2.305933,-1.704323,2.566290,-2.763119,-3.033076,-9.510002,4.908927,1.231344,2.075287,9.430313,-7.721241,-9.968900,-6.924167,-1.810852,9.064085,1.154666,1.958149,8.428638,4.731796,-4.141431,7.470491,3.021802,8.865998,5.016901,-6.320914,1.404128,4.682497,-3.689724,5.540652,2.121307,-3.018080,-9.191325,-1.279826,-0.025727,8.255905,4.552106,6.787056,-2.751484,-7.602099,5.806816,-8.015211,4.974148,7.403384,6.299618],[-0.856983,1.699387,8.868959,-3.461084,-9.074685,6.166777,-6.205266,-3.347377,-6.330280,-2.691078,7.574463,3.805355,-6.020391,9.229359,-3.310994,3.487530,5.759618,5.266258,-0.714292,-1.439782,-1.969826,-2.950102,8.804731,-2.298462,3.896914,-4.909914,2.464355,-0.779127,-5.829167,6.117073,-2.297658,8.276924,1.655544,5.979524,-9.918894,-0.751448,5.022789,3.495034,-7.258805,-4.768120,9.621306,-3.648526,-0.123347,-0.982073,-1.847694,-1.447015,-5.677741,-0.718694,-7.205280,9.100866,4.495605,-5.750694,5.596602,0.551753,7.994375,4.394772,-1.743881,3.756861,5.930017,7.736607,6.781002,-7.331556,7.713544,-3.162361,-7.681642,9.661823,9.452209,4.949264,1.322003,-7.284747,-3.582309,1.687191,0.365699,-8.552003,4.291846,-1.723785,2.193687,-2.026743,0.855838,-5.388196,4.475991,-6.050438,2.541629,4.263243,-3.654186,-4.130503,4.855446,-1.930358,-0.661952,9.894025,-9.763596,4.568027,2.113371,8.327354,-1.559165,-8.117075,2.686514,-6.657537,-8.403167,8.621726,-2.568139,-0.540073,-4.861733,-9.602842,4.998439,-9.571838,-1.991845,3.808721,1.850229,-3.116149,-9.531345,9.824410,4.693756,6.467185,8.074901,9.128977,-2.036951,-1.631838,-7.732494,2.416867,-9.643180,8.535259,1.311344,-6.164619,8.905860,-3.086119,-0.258797,0.208350,-8.850219,0.705437,-3.180614,8.868484,-6.475401,2.377330,-5.418037,-2.619367,-2.378627,4.684798,-3.398402,-9.031612,8.079178,-3.044109,9.463620,9.899004,-2.461177,2.466262,4.253383,7.792421,-6.002800,-5.040582,2.832385,-6.810243,4.683483,-4.496285,0.335968,3.822540,0.441773,-6.845907,-8.267158,-5.835327,7.676935,8.462382,-6.224626,-3.483604,5.500089,-2.963025,-9.859683,-9.776339,-1.560498,4.495635,5.348807,-3.835059,-0.182590,0.197030,9.028422,5.971551,-1.103076,2.920185,3.914695,7.091414,5.110608,-7.329503,-1.370892,3.066684,0.119555,2.419425,0.776525,9.476163,-7.429642,-5.251881,1.871012,-6.369324,-9.654386,-7.410710,-2.606488,-1.270000,9.070907,9.588478,9.508345,9.840221,8.284479,-8.932569,-4.824340,7.780321,-9.538063,-5.175242,7.206435,0.695232,4.323695,-3.473310,-3.498779,4.048011,8.778479,7.101459,8.483059,2.877308,4.596228,8.321560,9.027374,1.617757],[4.816246,4.690853,-6.344240,-6.563423,8.458005,4.310956,-1.040517,3.672630,5.699193,-8.892302,6.053592,-5.121009,1.296337,4.355914,-2.313953,2.102268,-2.002603,-9.145310,6.909801,-8.714360,9.538223,-6.354781,-3.507590,7.489275,7.638980,-1.952735,9.669293,1.540987,5.534493,-0.787387,1.318573,1.780330,-2.898499,4.827739,-3.373859,-4.350578,7.412692,0.320443,6.367242,-8.186821,0.223054,5.739528,4.716598,9.507762,9.980533,-1.957460,4.864541,2.491983,0.124169,-4.241561,-9.743384,-2.748230,4.974611,-2.164470,-6.097258,8.914838,8.765415,4.211814,-6.696740,6.006884,-7.297394,-5.336024,3.513416,8.218662,-7.037712,0.548073,4.825780,9.024238,7.534821,1.358276,-7.505950,3.393517,1.317634,-2.018532,-1.815779,-9.257280,7.320529,-7.735274,-8.460269,0.800953,5.522893,0.545117,-3.361548,-1.976957,-3.525589,-0.529408,-3.934840,7.077958,-2.317165,8.270884,4.360439,8.436004,1.786983,-4.334950,5.629393,-6.349727,-4.242563,1.703089,-6.753404,-2.777381,6.948517,9.800051,0.972542,-3.346448,-4.233015,-3.660122,2.572724,-5.868070,6.843958,-4.775581,5.596598,-4.440300,-1.499484,-1.566522,3.062868,-4.144619,1.989430,-4.603241,-1.521413,-4.865812,2.982854,-8.964536,-5.698836,7.054540,3.681222,5.915341,-7.807438,-8.629986,0.561778,-1.814174,-4.745108,-7.495147,-7.408326,-1.515308,1.283524,-2.960752,3.730394,4.409921,9.688006,8.780114,-1.349463,9.129577,-2.334540,5.143404,2.574997,-1.745721,-4.796557,7.195059,-7.844904,1.018326,0.785753,-5.047409,2.231148,-3.230153,-7.174938,8.406715,-1.630607,0.602934,1.228258,9.491474,-1.015188,-3.913038,-5.016038,-9.173456,8.089569,-9.286356,8.228545,5.284987,-9.704282,8.145281,4.633276,7.454687,-1.234180,8.923850,1.473467,-7.219350,-8.274276,5.926364,-6.578543,6.745165,0.756936,-7.572517,-9.704164,8.947593,-5.792668,0.511473,-4.055640,4.668682,-2.171814,-9.023079,2.831949,0.650501,5.604487,-4.868392,-1.535203,7.501835,5.105737,-2.758609,7.981842,-1.118263,7.752632,4.213114,-8.196957,-5.378457,0.319752,6.320614,2.510441,8.859056,7.232241,-4.044337,3.251210,3.967799,-2.470488,5.006367,-8.363854,0.478402,6.490854,1.173860,1.975713,-2.474012],[-7.536797,-5.106925,7.232221,-6.473867,1.313977,-5.050900,-3.190815,-5.820600,-0.709812,-6.588144,3.598623,-9.965876,-8.347616,0.802274,6.436024,6.644607,9.658339,3.479368,6.728142,2.621780,-9.290729,7.414814,5.330617,0.355991,2.349604,-6.896898,6.653574,-0.832556,-7.298490,-7.187350,5.135872,0.989766,9.816693,6.726548,-5.665537,0.233969,2.778925,5.401257,-0.034250,-2.403676,-7.482802,3.020706,7.778549,-2.137368,4.744346,0.659470,-7.684547,1.791858,0.986125,-0.407745,1.112669,-2.566088,-5.192958,1.472675,1.186960,0.390581,6.457304,6.665772,-0.358682,-7.765650,-1.992500,-2.792239,5.973542,-1.309913,0.178365,-9.836356,8.413852,7.210300,0.703941,-9.302561,-4.235784,-5.348712,-0.448325,-6.629132,3.815066,-5.639648,-7.049403,-0.459149,-5.848802,-3.419281,3.501991,-4.085671,5.094164,9.243257,3.496262,9.730938,9.139163,2.530908,-9.262550,-4.040719,-0.758564,7.311009,2.040967,-0.941351,3.302156,1.622354,-3.820900,2.848469,0.017901,-7.309959,-4.463415,-5.344330,5.290152,-1.562352,1.599832,-6.318367,0.478976,-3.123146,9.181972,4.773182,6.876341,-8.842366,-5.106065,-0.687688,-0.235092,-7.369231,-4.135565,-7.334759,-7.967000,6.268112,5.797392,-2.610005,-9.896729,2.436096,2.253433,-5.705860,-9.212645,-2.942987,-6.049789,-9.057698,3.542992,6.261283,-6.482550,-8.205643,-0.439915,-4.876024,3.999988,5.449921,-6.871464,6.380108,2.490684,-7.420921,-9.430423,-1.568476,2.466648,7.351084,-5.541369,5.942565,-8.545965,2.330911,9.970413,-3.183607,-8.271156,-7.036339,5.883860,0.766004,9.297058,-2.651306,-5.047970,-6.915832,3.604272,3.225991,2.197609,-3.064964,-8.336585,-8.059595,-0.411755,-3.167920,-1.595265,6.178224,-2.205456,-6.965065,-7.606892,-8.595298,-4.508460,2.096945,-3.891639,0.875166,2.372545,-4.348735,8.807207,1.274811,3.481031,9.571426,5.903213,8.868047,4.802585,7.285618,-8.136059,-1.128199,8.049924,5.295356,-2.356557,6.303236,-8.852403,2.830460,-6.640443,1.667513,-4.771468,-8.475387,9.379134,-7.482448,3.436083,9.311760,2.717331,-8.247828,9.947540,-4.738381,-6.546062,9.360545,8.727288,1.211704,1.440237,-9.202478,1.211911,9.894404,-5.289292,5.589573,-7.117926,-6.185124],[-6.581234,-4.547426,-1.576811,-1.441450,-7.746717,8.395582,-8.283095,1.840540,-9.258354,-8.885893,-3.491582,-7.282287,0.963845,1.950077,-2.112906,7.378644,-1.796786,-8.907631,-8.672387,-6.440046,-8.847124,1.550933,-6.779437,7.093499,-3.693735,-4.383317,2.599819,3.891552,-4.741414,-0.739643,-9.923423,0.173868,4.796205,7.552479,-1.956317,-8.883770,-0.972022,-4.991675,7.985972,7.452492,6.696196,-4.688611,-0.991303,5.940442,-8.648379,-3.347883,6.122169,-4.534092,1.014696,-7.386858,8.066632,6.166211,-5.539271,-1.078045,1.990418,5.304221,2.235103,9.754772,-3.446333,-2.253276,5.826323,7.536815,-4.424682,4.610979,-3.879970,-5.599244,7.964888,9.444606,0.775764,3.514305,2.341360,8.610388,-6.899557,-7.707924,8.778061,-8.708525,7.658606,3.262522,9.330274,-9.372307,-1.095881,3.692940,-5.790265,-9.537398,-4.213033,-9.607260,-7.947840,-5.000089,-1.935000,4.071439,3.153537,-5.628190,-1.827092,-8.645980,5.787314,1.413640,2.806119,0.402247,7.066819,8.234703,2.539491,4.301499,1.649706,-1.236257,-3.230420,9.243015,1.002974,2.443354,6.032390,4.698090,-4.279934,4.633573,0.946192,2.572759,-5.022565,-7.951468,-6.490629,6.896127,3.918997,2.004678,-0.939971,1.773227,3.819420,3.991745,7.039382,3.985781,1.701114,7.469220,0.700406,3.377046,-0.133105,6.611610,4.416266,-2.999031,-8.185530,-2.954398,7.213314,3.117711,-0.090655,8.975379,8.550425,3.062144,5.712709,-9.418663,3.121803,-0.527635,8.951602,-1.645310,5.929935,-9.196591,2.362786,-9.258182,-3.324190,7.957347,-9.993189,-3.338568,6.557996,-6.907710,7.149073,0.742759,-5.754110,-3.273407,3.167333,-8.228258,6.512258,-8.431561,2.090034,5.513717,-7.574881,4.885525,-1.227331,3.744921,4.928276,-8.314203,-4.717600,-1.386747,-0.344872,-0.562913,5.775427,-3.972439,1.788118,6.578980,9.639290,1.593851,6.094286,-3.522455,-3.443185,0.451685,8.475795,-3.428246,8.000165,-8.988792,1.203727,0.987173,2.965694,-1.494226,8.226584,8.300349,-8.316130,-1.605240,1.264207,0.211313,-5.381325,-6.400117,2.878299,2.513741,7.093450,-5.085208,0.993891,6.176565,2.027230,9.386672,-6.343881,9.091091,0.995989,-7.133167,-3.490700,4.995019,1.926913,-3.251800],[-7.902346,2.058800,6.185194,9.882479,-7.328966,8.226977,-0.934880,-1.223691,9.740720,3.952616,4.740792,-7.247998,-5.206132,8.120392,3.294493,-8.627653,5.994566,1.862125,7.127141,4.422201,2.912288,5.747207,-4.979564,-5.300881,3.950218,-8.526237,-1.273190,-9.560240,-0.546653,-7.732892,7.749725,-6.536481,-5.785920,2.321697,-2.921956,8.039177,4.333004,0.092957,-6.591387,0.639348,1.750322,1.189474,-1.365887,-9.311331,3.747496,-8.792284,2.437590,0.690543,-0.248332,9.641269,-1.712558,5.400661,3.237604,0.330376,-6.637809,0.069706,-4.324897,6.133261,-1.912446,-1.287067,-7.389560,-1.950211,8.486143,-2.421524,-2.781664,-7.258489,-7.914348,-6.685372,-5.358788,-0.873282,5.989007,-8.087173,-2.897697,-9.009224,-5.128345,-2.918750,-1.980826,-3.370858,-0.289790,6.764061,9.118446,7.764568,0.539792,2.124994,2.547210,3.622044,-1.802045,7.285512,2.509644,3.273579,-8.734792,8.648297,7.070704,-1.046257,-9.050584,-8.334005,0.099840,2.509464,-8.306845,-8.991541,-4.282768,-0.412743,7.910545,9.003775,7.857297,-3.156979,-1.519411,-3.318831,-4.107578,-2.118215,8.250338,-4.117436,-9.595383,6.459691,-1.660593,1.313272,4.732298,5.795164,8.048993,-5.473451,-0.524253,-0.961296,-5.656802,-6.763306,9.004643,7.611246,1.384687,2.851156,0.588900,-6.921560,-2.160333,7.304561,6.750419,-5.610352,6.919865,-3.176452,-8.481647,9.387379,6.188201,6.267583,-3.248778,1.337373,-7.380377,2.077989,-4.908526,9.889068,5.424846,-9.478353,-4.454757,9.046102,4.145312,5.323485,-6.739589,-5.822527,-8.909148,-1.713753,-1.074579,-6.520033,-1.531052,9.571413,-6.646217,6.898193,6.536026,-5.869664,1.293850,6.612343,1.927890,-8.833558,8.889463,-9.695136,-6.618478,-4.076765,-4.343528,8.186433,5.670990,9.554304,3.352889,9.881000,1.514370,-0.618887,-8.281546,2.771559,-9.707789,2.800252,6.975076,0.313355,3.798599,-1.073295,6.974752,-9.801118,-2.314721,-5.413393,1.651564,-7.092963,4.344959,9.497941,9.623318,-4.973989,-9.934242,7.969463,8.433202,6.757007,2.823180,-1.564966,6.327043,7.244859,7.960997,6.317282,2.250894,2.783321,7.118500,-2.979096,3.909772,-6.437858,7.638417,2.631221,3.432811,0.325416,1.358407,6.097369],[-7.366090,-3.999315,-6.010515,-2.964064,-8.560754,-5.764493,-5.335933,-6.294734,7.799759,9.032362,-8.489476,7.109585,4.328586,-3.904977,5.947811,1.822313,-1.697512,3.930760,-6.701899,0.133539,8.519219,-3.512108,-8.398223,8.496154,1.227815,-2.027901,-3.030262,5.241712,-5.669339,-8.124099,6.883667,-7.672840,1.673439,-2.630678,5.071752,-7.160628,-8.462657,1.046839,5.284274,-8.250949,7.294628,1.770541,4.562944,-9.242609,-7.972915,0.004216,-3.146227,-4.155877,3.433627,2.478064,-8.005535,7.020547,-5.403082,3.154372,9.413517,2.856776,2.880573,-2.588357,9.225677,-4.513201,9.873341,-6.988650,0.234317,-4.002097,6.014660,-0.090802,8.220817,-6.851861,-2.844890,-3.731075,-9.731281,-2.995703,-2.054028,-6.960718,5.222961,6.825073,-0.228102,2.550723,-2.288414,-1.156248,3.587725,4.814973,-6.400522,0.314666,-7.692431,-2.697987,4.883032,-7.445210,-3.947370,2.433245,5.379238,-4.112842,-4.644017,-1.791599,9.651459,9.400310,6.691868,-1.712988,-8.639350,-0.516757,8.083356,4.170950,7.901259,-8.105508,-1.974271,-2.503598,-9.657357,9.219163,-5.922522,-6.446373,-0.857593,-7.238080,-7.456404,7.433259,-0.295101,8.314825,1.085134,-9.885361,-1.806532,-4.770702,1.407168,2.143575,8.126074,-5.950275,-6.279276,-0.590583,-7.311554,-0.084822,2.091257,-3.735685,3.618723,9.674592,-2.061345,6.129433,-9.813650,-9.675637,-7.239917,5.245465,-4.046057,1.989053,-1.537088,-8.351094,6.677872,-3.594421,-2.203530,-1.329159,-1.833586,-1.304671,-5.017081,-7.579546,3.324606,-6.069546,-5.242762,-7.475459,6.054187,8.001051,0.996122,0.371350,-4.921412,6.173336,-9.377889,-8.707044,3.235812,2.070960,-8.943952,-7.685817,-7.310157,-0.688200,-1.860922,-7.439218,-6.277405,-7.717603,0.943304,-6.037612,8.886886,-7.146422,-4.460638,-1.546949,-1.557754,0.772300,5.709784,-0.866371,-2.042467,-1.517159,7.849460,-1.557983,-2.539738,3.179716,-8.522693,5.773985,8.187540,7.764303,-6.047583,4.002669,-1.099057,-6.673177,-7.922943,8.687112,3.627654,-1.672407,-7.375803,-1.532237,7.495520,-0.156369,-8.720967,7.507429,-0.352195,9.342448,-9.088018,4.205699,0.434707,7.136450,-0.141924,-4.363491,-0.728203,5.503197,0.447729,-9.529756,1.315493,3.629139],[-7.054236,-5.373453,8.351982,-0.378872,-8.315132,-5.632779,9.266867,-8.281853,4.637504,2.049931,1.174716,-3.522956,-2.766282,7.951533,-0.246163,1.973248,8.721814,4.537712,6.630047,-1.737048,1.865741,7.400247,-7.686405,-4.459859,-2.665329,-2.666026,-3.221833,2.263991,2.623083,1.035245,7.585897,-7.207853,-5.224930,8.477826,0.109405,2.842942,6.763994,2.454087,-6.811757,4.908358,7.285668,-3.381079,-5.267351,0.344829,-9.721430,1.395037,5.635922,-3.973090,4.999998,-5.172480,-9.322694,4.643983,-2.706554,4.769220,5.625003,-3.371757,-0.878347,-9.092016,-0.667554,-7.613616,-0.236161,-0.584478,-9.433132,6.954528,-6.901492,-9.042962,-0.991838,-1.011668,5.917275,0.483673,-8.368888,8.054753,4.560291,-6.812842,2.723009,-2.012690,-9.768893,-0.570384,-7.501219,7.199877,7.743863,3.503402,5.067113,-9.203563,9.667425,7.595421,7.926488,1.140003,8.633541,-7.619863,-4.477290,-9.531094,1.480227,5.108006,5.565862,3.249033,6.821237,-8.118442,-9.868058,3.643765,-6.259013,-8.973459,7.082091,-1.182319,2.428646,-5.474969,-4.935809,-0.007568,9.156477,4.441180,-9.325543,0.957009,4.498653,4.911765,-0.441913,2.096726,8.441250,6.270543,6.423728,9.716795,-4.666203,5.125079,2.038025,0.081515,0.284700,8.909938,5.453779,9.088569,-6.738399,-3.240809,6.766427,4.958356,5.436861,-2.048967,8.049120,5.832336,4.157287,-2.609570,4.278535,8.354731,-2.506981,-2.992584,3.392002,-2.162002,-1.271345,-5.498950,-1.333425,7.909451,-3.134697,3.116869,1.775667,-8.149605,0.288727,3.588254,-2.686344,3.876691,7.152013,-1.290753,-2.964116,-4.778239,6.615992,-2.778450,3.743394,-2.019967,1.982630,0.847403,-5.301224,3.012416,-1.024494,-1.344890,5.025494,9.324004,-6.941576,6.506737,-4.119407,9.674123,9.190087,7.668237,7.210582,-6.821970,2.261100,-5.499889,4.804453,2.878763,-9.076704,-8.766317,-0.014975,8.489621,3.959521,7.891596,-1.341273,-6.604047,-0.986347,-8.459753,7.147424,9.651883,-1.187540,-2.732345,3.946922,3.213136,-6.891382,-2.433546,2.224500,7.191650,-3.733154,-8.386226,2.550058,-1.166015,3.335815,-1.160993,-9.232858,5.771899,2.013220,-9.369642,-5.697793,-2.037461,-8.864590,0.522651,-8.234493,-7.653024]], dtype = "float32")#candidate|1700|(12, 220)|const|float32
call_1699 = relay.TupleGetItem(func_1475_call(relay.reshape(const_1700.astype('float32'), [15, 11, 16])), 2)
call_1701 = relay.TupleGetItem(func_1477_call(relay.reshape(const_1700.astype('float32'), [15, 11, 16])), 2)
uop_1703 = relay.rsqrt(bop_1688.astype('float32')) # shape=(6, 14)
uop_1710 = relay.log(uop_1703.astype('float64')) # shape=(6, 14)
bop_1713 = relay.mod(uop_1710.astype('float32'), relay.reshape(uop_1703.astype('float32'), relay.shape_of(uop_1710))) # shape=(6, 14)
uop_1737 = relay.sinh(bop_1695.astype('float64')) # shape=(6, 14)
var_1744 = relay.var("var_1744", dtype = "float64", shape = (6, 14))#candidate|1744|(6, 14)|var|float64
bop_1745 = relay.logical_and(uop_1710.astype('bool'), relay.reshape(var_1744.astype('bool'), relay.shape_of(uop_1710))) # shape=(6, 14)
output = relay.Tuple([bop_1692,call_1699,const_1700,bop_1713,uop_1737,bop_1745,])
output2 = relay.Tuple([bop_1692,call_1701,const_1700,bop_1713,uop_1737,bop_1745,])
func_1750 = relay.Function([var_1687,var_1744,], output)
mod['func_1750'] = func_1750
mod = relay.transform.InferType()(mod)
var_1751 = relay.var("var_1751", dtype = "bool", shape = (6, 14))#candidate|1751|(6, 14)|var|bool
var_1752 = relay.var("var_1752", dtype = "float64", shape = (6, 14))#candidate|1752|(6, 14)|var|float64
output = func_1750(var_1751,var_1752,)
func_1753 = relay.Function([var_1751,var_1752,], output)
mutated_mod['func_1753'] = func_1753
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1764 = relay.var("var_1764", dtype = "uint32", shape = (14, 16, 16))#candidate|1764|(14, 16, 16)|var|uint32
var_1765 = relay.var("var_1765", dtype = "uint32", shape = (14, 16, 16))#candidate|1765|(14, 16, 16)|var|uint32
bop_1766 = relay.add(var_1764.astype('uint32'), relay.reshape(var_1765.astype('uint32'), relay.shape_of(var_1764))) # shape=(14, 16, 16)
uop_1773 = relay.asinh(var_1765.astype('float32')) # shape=(14, 16, 16)
var_1775 = relay.var("var_1775", dtype = "float32", shape = (14, 16, 16))#candidate|1775|(14, 16, 16)|var|float32
bop_1776 = relay.left_shift(uop_1773.astype('int32'), relay.reshape(var_1775.astype('int32'), relay.shape_of(uop_1773))) # shape=(14, 16, 16)
func_1409_call = mod.get_global_var('func_1409')
func_1413_call = mutated_mod.get_global_var('func_1413')
const_1780 = relay.const([[2,5,-6,-8,-3,-5,-7,-10,-2,10,8,3,6,-10,5,-2,-7,6,-4,-8,-9,4,7,3,-8,7,-5,-5,-4,-1,-9,5],[10,1,8,-2,4,5,9,-5,10,-4,-10,5,-6,-4,9,-2,1,-4,-7,-5,2,6,-7,10,-3,-2,5,-1,-6,-7,4,-6],[5,5,-8,-4,-2,4,3,7,-10,1,10,6,6,-3,-3,9,-10,-1,-5,-9,-9,-6,-6,-8,-4,2,2,-2,-1,1,5,-3],[4,-3,1,8,4,-10,-7,-7,-1,8,6,4,-4,-3,-6,-4,-8,-9,-7,-9,-1,1,9,-2,6,-6,8,4,-6,1,-1,2],[1,10,-8,-1,8,-8,2,9,-8,2,4,9,3,2,6,3,3,5,10,2,-10,7,10,1,9,-5,-9,5,-6,-6,-8,10],[-10,-5,-3,-10,-9,6,-1,9,-10,4,-1,-1,-3,9,-6,3,4,-8,-6,1,-3,9,3,5,4,-3,8,4,-3,-3,4,6],[-5,-6,-1,10,-3,8,-7,-9,2,-8,6,-2,-8,9,-4,-1,-7,3,5,4,10,-4,-4,-10,5,-9,-3,-1,9,-3,2,1],[1,9,-7,4,4,-6,2,-9,5,9,10,-9,9,9,5,5,5,-4,7,-2,9,-3,4,-10,-9,-7,1,7,7,5,9,-2],[4,-9,-5,3,-8,6,10,3,7,10,5,9,-5,-3,-4,9,2,-10,8,2,-2,10,-6,-3,2,-10,1,6,-4,10,-6,6],[8,5,-9,-4,7,5,8,6,-9,4,10,6,10,1,-7,-4,-3,5,-5,-5,2,6,5,-9,-2,6,4,3,-10,-2,3,3],[4,-10,-6,1,-4,-9,9,-3,10,3,1,-10,2,9,-9,-3,-10,-1,-7,-6,9,8,-3,9,2,9,-7,6,2,-7,8,3],[-3,6,-9,9,-10,-6,-9,8,8,-7,-8,-6,-8,3,-5,6,5,5,6,9,-2,9,-1,-5,6,-6,4,-9,7,8,10,-7],[6,8,8,8,-2,1,7,4,9,10,-10,-8,2,-8,-2,7,6,1,-3,-4,-7,-10,-3,8,4,9,-8,10,7,-2,2,-8],[-7,2,8,-6,-3,7,-3,-1,-2,1,-7,10,9,-2,3,1,9,-9,1,9,-4,-9,-3,3,1,7,-5,2,9,6,6,-10],[-1,1,6,6,-6,8,7,7,5,6,9,3,5,-9,-7,5,10,9,-6,10,2,-3,9,-7,-4,-4,2,-10,-9,8,-8,5],[9,-5,-1,-6,3,1,1,-6,-10,7,-10,-9,-8,6,-9,5,8,-2,-7,3,-10,-1,4,-6,-10,8,-2,10,3,-5,8,2],[-1,5,5,-5,-4,-7,-6,3,-6,7,-8,-7,2,5,1,5,4,-10,-8,5,5,-10,-5,-6,-4,8,-9,-4,6,-10,1,-10],[7,-3,7,9,-7,3,-2,2,2,-1,-4,2,6,-6,-7,-1,6,-9,9,-5,7,-10,-6,6,-4,-9,-3,8,-2,6,1,-6],[-1,-4,-3,1,-6,10,5,2,-4,-8,3,-8,-10,-6,4,-2,9,7,4,3,-1,3,-10,7,-9,2,-1,-9,-5,-4,9,3],[-9,-9,10,-2,-4,9,4,-9,-1,5,9,2,8,-10,1,4,-6,2,6,9,1,-5,5,7,3,9,3,6,9,-10,-1,-10],[-9,5,8,-5,-3,3,-4,-6,-2,9,-8,9,-2,8,3,-2,-2,-10,-10,-9,-5,1,-5,-4,-1,2,-9,7,-2,-7,-4,-9],[-6,7,4,-7,2,-6,-4,-6,7,-4,-1,-10,-5,7,-7,-8,-5,-7,4,-6,3,-5,-4,1,9,8,6,2,-6,-3,-5,-4],[-7,-7,-8,-9,-7,-7,5,2,3,3,-9,-3,2,-10,8,-5,-6,7,-7,4,-1,3,-10,10,-7,5,6,5,-3,-6,8,5],[6,-7,-10,-7,5,-7,-6,-10,-7,-3,10,-4,-10,3,4,1,4,4,-2,-5,2,-1,6,-3,6,-10,3,-1,-4,10,-6,7],[2,4,-2,-10,10,3,3,-10,2,-7,-2,8,5,-4,-1,-3,-2,-7,-5,-9,8,-8,7,-2,-8,6,9,5,-3,7,10,9],[4,-8,-2,-3,-8,3,-10,4,8,-7,-3,-1,1,-2,-7,-5,-6,-1,10,1,4,3,3,-2,-9,-7,1,-9,-7,5,4,8],[-5,-10,-4,4,9,2,7,2,-2,8,-9,6,-10,2,1,3,9,-10,6,-1,-1,6,8,4,7,1,-9,-1,-8,-2,-1,6],[8,-2,-1,-9,7,-2,3,5,8,-5,-5,-5,-7,3,-9,9,4,-4,-6,7,8,8,-5,-1,-9,9,1,8,-8,8,-7,-9]], dtype = "uint64")#candidate|1780|(28, 32)|const|uint64
call_1779 = func_1409_call(relay.reshape(const_1780.astype('uint64'), [4, 14, 16]), relay.reshape(const_1780.astype('uint64'), [4, 14, 16]), )
call_1781 = func_1409_call(relay.reshape(const_1780.astype('uint64'), [4, 14, 16]), relay.reshape(const_1780.astype('uint64'), [4, 14, 16]), )
output = relay.Tuple([bop_1766,bop_1776,call_1779,const_1780,])
output2 = relay.Tuple([bop_1766,bop_1776,call_1781,const_1780,])
func_1784 = relay.Function([var_1764,var_1765,var_1775,], output)
mod['func_1784'] = func_1784
mod = relay.transform.InferType()(mod)
mutated_mod['func_1784'] = func_1784
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1784_call = mutated_mod.get_global_var('func_1784')
var_1786 = relay.var("var_1786", dtype = "uint32", shape = (14, 16, 16))#candidate|1786|(14, 16, 16)|var|uint32
var_1787 = relay.var("var_1787", dtype = "uint32", shape = (14, 16, 16))#candidate|1787|(14, 16, 16)|var|uint32
var_1788 = relay.var("var_1788", dtype = "float32", shape = (14, 16, 16))#candidate|1788|(14, 16, 16)|var|float32
call_1785 = func_1784_call(var_1786,var_1787,var_1788,)
output = call_1785
func_1789 = relay.Function([var_1786,var_1787,var_1788,], output)
mutated_mod['func_1789'] = func_1789
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1842 = relay.var("var_1842", dtype = "float32", shape = (11, 9, 9))#candidate|1842|(11, 9, 9)|var|float32
uop_1843 = relay.atan(var_1842.astype('float32')) # shape=(11, 9, 9)
func_556_call = mod.get_global_var('func_556')
func_560_call = mutated_mod.get_global_var('func_560')
const_1848 = relay.const([2,8,6,-5,6,-7,5,7,-9,-2,-2,9,2,-9,5,-8,5,-7,1,-3,-7,-1,2,4,-6,8,-7,-9,10,-7,-9,2,-2,-2,10,9,8,10,9,-9,8,2,4,-2,1,-1,-6,-4,-3,5,-4,1,-6,1,-4,-8,-8,8,4,-10,3,5,6,10,-6,5,9,1,8,1,-2,6,9,-6,6,-8,-7,10,4,5,7,9,5,-3,2,4,-1,6,4,5,-7,-1,-9,10,-1,3,-4,-3,-4,3,-10,-2,7,3,9,3,4,-10,3,4,4,3,10,8,-2,-2,8,-3,1,8,5,-5,5,10,5,3,3,-9,2,9,-10,6,4,-5,-7,-6,-4,3,-8,7,-9,5,8,5,3,6,4,3,5,-6,9,-7,2,-2,3,4,-2,5,2,2,5,-1,1,-8,10,-9,5,3,7,10,5,-3,-2,8,8,1,9,-2,-4,7,4,4,-3,7,-2,-10,-10,5,3,7,-3,-5,-8,3,9,9,-7,3,2,9,-8,1,6,-1,9,-10,-4,-5,4,-8,9,6,-10,7,10,6,-7,-7,4,1,6,6,-6,6,-10,-3,-2,6,-10,6,8,-10,-10,9,-7,-5,-2,4,-3,-8,-5,-3,-9,-10,9,1,-7,-4,9,9,4,10,-5,3,-7,3,1,-8,5,9,4,-1,2,6,-1,-9,6,-10,-4,-1,6,-2,9,5,-9,-8,-8,-2,8,-2,-2,3,-10,1,8,4,7,10,7,-2,4,9,-10,-2,8,7,-7,-10,-5,-3,-6,-8,-5,10,4,10,1,9,10,-10,-5,10,5,-1,2,5,10,-8,-2,9,-10,3,6,-1,-3,-7,-7,-4,-7,-5,8,-5,-5,-3,3,2,-1,-5,7,5,6,-2,-9,7,9,4,-4,-6,-4,-4,1,2,-3,8,2,1,5,3,8,-9,4,4,4,6,9,7,-8,-3,-4,10,-6,6,-4,7,-8,1,-10,1,7,2,-9,-7,7,-7,-10,-9,-10,10,-3,-8,-3,-5,5,9,-5,-7,1,1,-3,1,7,-4,4,-1,-2,9,9,10,-10,-8,-3,9,-1,4,-9,1,-6,3,10,-8,-6,-6,1,-10,9,5,-5,6,5,3,-6,9,1,1,-9,-2,5,9,7,8,7,6,10,-1,-1,-8,-9,8,-3,1,-9,5,2,4,3,-3,-5,-4,2,-9,-8,-1,-6,-8,9,8,-2,-4,7,4,-9,-10,-6,9,-10,3,5,-8,3,9,7,10,2,-6,3,1,-4,8,7,-10,-6,-6,8,-2,-2,-1,-10,-8,8,-5,1,2,-1,9,-8,7,-3,4,9,-10,-9,-7,5,-3,-9,-3,6,-8,-8,5,6,-9,-4,-10,-1,3,-5,-10,8,-3,-4,1,-2,7,6,5,5,-4,-10,-4,-7,10,-9,-2,6,8,-10,1,8,3,1,9,-3,-7,-7,4,9,-6,8,7,-1,-3,-4,4,-4,9,6,-3,10,4,2,1,-5,10,-5,5,-4,7,-5,-3,10,6,9,-7,-1,6,10,-5,-6,-6,3,-7,4,-8,-8,9,2,-4,-7,10,3,-3,7,5,4,6,-5,-8,-8,-7,7,6,1,10,2,-6,-8,-5,-7,-10,-2,-10,-1,5,3,8,7,-9,4,6,9,-9,1,6,4,10,-4,4,-7,7], dtype = "uint32")#candidate|1848|(640,)|const|uint32
var_1849 = relay.var("var_1849", dtype = "float32", shape = (9, 1))#candidate|1849|(9, 1)|var|float32
call_1847 = relay.TupleGetItem(func_556_call(relay.reshape(const_1848.astype('uint32'), [10, 4, 16]), relay.reshape(var_1849.astype('float32'), [9,]), ), 2)
call_1850 = relay.TupleGetItem(func_560_call(relay.reshape(const_1848.astype('uint32'), [10, 4, 16]), relay.reshape(var_1849.astype('float32'), [9,]), ), 2)
var_1853 = relay.var("var_1853", dtype = "float32", shape = (11, 9, 9))#candidate|1853|(11, 9, 9)|var|float32
bop_1854 = relay.left_shift(uop_1843.astype('uint64'), relay.reshape(var_1853.astype('uint64'), relay.shape_of(uop_1843))) # shape=(11, 9, 9)
func_143_call = mod.get_global_var('func_143')
func_147_call = mutated_mod.get_global_var('func_147')
const_1859 = relay.const([3.610288,3.972844,7.994752,-9.116365,-4.486741,-4.223652,5.790579,-8.041760,1.232049,-4.171802,7.166522,-5.752115,5.824222,-0.028917,1.175892,4.515929,9.457954,-8.625888,-6.465194,-8.689595,-5.836159,-9.785650,8.358303,4.309754,-3.049843,7.457155,5.429935,-1.582177,-7.130170,3.108839,6.114477,-1.224617,2.222430,5.110104,7.675900,-8.597072,-1.158910,6.232749,-4.216389,-6.862041], dtype = "float64")#candidate|1859|(40,)|const|float64
const_1860 = relay.const([6.404799,2.013405,-8.455423,-1.886876,-3.797353,-0.155970,4.759810,2.725554,2.282420,-7.123402,-4.293708,-3.012853,-7.573054,7.911089,-9.665039,7.622307,-2.571820,1.751805,3.961605,-9.499748,3.726560,-9.403274,-9.228955,3.018457,4.578503,-2.703400,-5.752121,-4.717629,3.501278,-7.688309,-7.244706,-5.843779,-8.029954,5.291976,9.725221,9.653483,-1.207037,5.637214,9.026617,7.111761,7.257094,-1.285018,-4.151663,-2.664022,-4.548994,6.862092,-1.542728,1.386700,3.381888,5.425017,-2.449008,0.161963,-0.908983,-3.423003,-2.021828,-8.756704,8.075372,7.954929,-8.852554,3.948133,-3.685262,3.811212,-3.982787,6.181394,3.181963,-4.380236,-9.371587,-9.326507,9.391626,-7.420706,-2.913556,0.820219,-8.506560,1.786360,-9.459299,5.652202,7.844435,-7.276808,-6.482333,-7.163697], dtype = "float32")#candidate|1860|(80,)|const|float32
call_1858 = relay.TupleGetItem(func_143_call(relay.reshape(const_1859.astype('float64'), [5, 8]), relay.reshape(const_1860.astype('float32'), [80,]), ), 0)
call_1861 = relay.TupleGetItem(func_147_call(relay.reshape(const_1859.astype('float64'), [5, 8]), relay.reshape(const_1860.astype('float32'), [80,]), ), 0)
var_1862 = relay.var("var_1862", dtype = "float32", shape = (11, 9, 9))#candidate|1862|(11, 9, 9)|var|float32
bop_1863 = relay.subtract(uop_1843.astype('int64'), relay.reshape(var_1862.astype('int64'), relay.shape_of(uop_1843))) # shape=(11, 9, 9)
uop_1878 = relay.cosh(const_1859.astype('float32')) # shape=(40,)
output = relay.Tuple([call_1847,const_1848,var_1849,bop_1854,call_1858,const_1860,bop_1863,uop_1878,])
output2 = relay.Tuple([call_1850,const_1848,var_1849,bop_1854,call_1861,const_1860,bop_1863,uop_1878,])
func_1886 = relay.Function([var_1842,var_1849,var_1853,var_1862,], output)
mod['func_1886'] = func_1886
mod = relay.transform.InferType()(mod)
var_1887 = relay.var("var_1887", dtype = "float32", shape = (11, 9, 9))#candidate|1887|(11, 9, 9)|var|float32
var_1888 = relay.var("var_1888", dtype = "float32", shape = (9, 1))#candidate|1888|(9, 1)|var|float32
var_1889 = relay.var("var_1889", dtype = "float32", shape = (11, 9, 9))#candidate|1889|(11, 9, 9)|var|float32
var_1890 = relay.var("var_1890", dtype = "float32", shape = (11, 9, 9))#candidate|1890|(11, 9, 9)|var|float32
output = func_1886(var_1887,var_1888,var_1889,var_1890,)
func_1891 = relay.Function([var_1887,var_1888,var_1889,var_1890,], output)
mutated_mod['func_1891'] = func_1891
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1900 = relay.var("var_1900", dtype = "float64", shape = (3, 9, 13))#candidate|1900|(3, 9, 13)|var|float64
uop_1901 = relay.log2(var_1900.astype('float64')) # shape=(3, 9, 13)
uop_1917 = relay.rsqrt(uop_1901.astype('float64')) # shape=(3, 9, 13)
bop_1920 = relay.less_equal(uop_1901.astype('bool'), relay.reshape(uop_1917.astype('bool'), relay.shape_of(uop_1901))) # shape=(3, 9, 13)
func_192_call = mod.get_global_var('func_192')
func_195_call = mutated_mod.get_global_var('func_195')
const_1924 = relay.const([-0.209707,0.415544,-1.458935,5.725443], dtype = "float32")#candidate|1924|(4,)|const|float32
call_1923 = relay.TupleGetItem(func_192_call(relay.reshape(const_1924.astype('float32'), [1, 4])), 0)
call_1925 = relay.TupleGetItem(func_195_call(relay.reshape(const_1924.astype('float32'), [1, 4])), 0)
output = relay.Tuple([bop_1920,call_1923,const_1924,])
output2 = relay.Tuple([bop_1920,call_1925,const_1924,])
func_1927 = relay.Function([var_1900,], output)
mod['func_1927'] = func_1927
mod = relay.transform.InferType()(mod)
mutated_mod['func_1927'] = func_1927
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1928 = relay.var("var_1928", dtype = "float64", shape = (3, 9, 13))#candidate|1928|(3, 9, 13)|var|float64
func_1927_call = mutated_mod.get_global_var('func_1927')
call_1929 = func_1927_call(var_1928)
output = call_1929
func_1930 = relay.Function([var_1928], output)
mutated_mod['func_1930'] = func_1930
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1959 = relay.var("var_1959", dtype = "float32", shape = (6, 3, 14))#candidate|1959|(6, 3, 14)|var|float32
uop_1960 = relay.tan(var_1959.astype('float32')) # shape=(6, 3, 14)
bop_1962 = relay.subtract(uop_1960.astype('uint32'), relay.reshape(var_1959.astype('uint32'), relay.shape_of(uop_1960))) # shape=(6, 3, 14)
func_440_call = mod.get_global_var('func_440')
func_444_call = mutated_mod.get_global_var('func_444')
var_1966 = relay.var("var_1966", dtype = "int32", shape = (156,))#candidate|1966|(156,)|var|int32
const_1967 = relay.const([7.546155,1.276681,-5.787810,-1.184766,8.437249,4.469506,5.513001,-5.287973,0.669062], dtype = "float32")#candidate|1967|(9,)|const|float32
call_1965 = relay.TupleGetItem(func_440_call(relay.reshape(var_1966.astype('int32'), [13, 12]), relay.reshape(const_1967.astype('float32'), [9,]), ), 3)
call_1968 = relay.TupleGetItem(func_444_call(relay.reshape(var_1966.astype('int32'), [13, 12]), relay.reshape(const_1967.astype('float32'), [9,]), ), 3)
bop_1978 = relay.not_equal(uop_1960.astype('bool'), relay.reshape(var_1959.astype('bool'), relay.shape_of(uop_1960))) # shape=(6, 3, 14)
output = relay.Tuple([bop_1962,call_1965,var_1966,const_1967,bop_1978,])
output2 = relay.Tuple([bop_1962,call_1968,var_1966,const_1967,bop_1978,])
func_1987 = relay.Function([var_1959,var_1966,], output)
mod['func_1987'] = func_1987
mod = relay.transform.InferType()(mod)
var_1988 = relay.var("var_1988", dtype = "float32", shape = (6, 3, 14))#candidate|1988|(6, 3, 14)|var|float32
var_1989 = relay.var("var_1989", dtype = "int32", shape = (156,))#candidate|1989|(156,)|var|int32
output = func_1987(var_1988,var_1989,)
func_1990 = relay.Function([var_1988,var_1989,], output)
mutated_mod['func_1990'] = func_1990
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2015 = relay.var("var_2015", dtype = "float64", shape = (8, 10))#candidate|2015|(8, 10)|var|float64
uop_2016 = relay.rsqrt(var_2015.astype('float64')) # shape=(8, 10)
uop_2019 = relay.sqrt(uop_2016.astype('float32')) # shape=(8, 10)
bop_2022 = relay.bitwise_or(uop_2019.astype('uint64'), relay.reshape(var_2015.astype('uint64'), relay.shape_of(uop_2019))) # shape=(8, 10)
func_1475_call = mod.get_global_var('func_1475')
func_1477_call = mutated_mod.get_global_var('func_1477')
const_2026 = relay.const([[-3.091939,-8.054897,-3.483313,0.633384,-5.932534,-8.257245,3.597930,-1.351994,9.456330,-8.087271,-8.933643,3.522535,0.603262,9.659295,-6.093255,0.045164,0.712333,2.806499,-1.380300,-9.165870,-2.294600,5.409213,-6.934073,-9.917942,4.495715,-6.493842,8.918826,-3.374162,7.619553,1.949354,1.105668,-3.805228,-8.326676,-6.414528,-0.208695,6.603871,5.300375,2.973998,-2.597103,-2.906764,-2.039626,3.858542,-4.858651,-0.118714,-1.289149,9.175541,7.507672,-9.880048,-2.149774,-3.398557,0.363943,8.803920,-8.887099,8.082539,-2.437487,6.976048,-7.309102,2.843859,-6.192837,-0.801691,0.190437,-9.606322,-1.508124,-9.236835,4.204456,3.018381,-8.509909,-8.010805,9.194051,6.327425,2.732627,5.663778,5.600598,-4.380582,-2.346600,-1.973801,-4.127782,-3.971596,2.644661,7.710237,-3.144798,-1.348196,8.629096,4.502612,-7.401891,5.031950,1.652039,8.950256,0.389470,3.710716,3.875957,1.647422,-6.517432,-3.488383,-6.428106,1.327192,-0.643947,9.030836,-6.322689,6.978372,8.490383,-3.234905,6.601930,1.151979,3.925266,-1.157084,6.959784,-3.683829,-0.727886,6.635119,3.111644,1.652259,-6.532740,0.597233,0.671812,4.650502,9.456768,-2.990948,-5.491554,9.091962,-0.764454,-0.802758,8.648377,4.293561,0.718038,-2.135840,1.310091,-6.288427,-5.256545,-8.697489,-5.532462,-7.052683,3.873067,-1.590838,2.655074,9.512664,-9.765967,3.208303,3.406088,8.927987,3.436555,3.107127,5.231525,1.938265,5.285523,5.432529,8.418312,-6.550775,-8.620422,-2.641863,-4.060256,2.582760,8.855658,3.780759,-3.257221,9.620044,-6.433340,-9.352215,7.259023,-8.441306,-5.566450,-3.952848,5.575962,-1.210999,-9.973490,3.520457,6.815972,3.595543,4.629167,2.274433,9.811591,-8.920236,8.154879,4.367097,-5.063820,3.469998,2.622726,-8.745675,-5.132993,0.248124,-6.620231,-6.373880,7.775515,-1.172430,-2.190164,-9.052085,4.511267,0.316758,-8.098226,1.279771,-7.907214,5.653192,-9.636174,5.321793,-5.381534,2.412335,9.114347,-8.208529,8.908855,-8.802076,-4.277613,5.475749,6.944946,-7.816638,8.494913,-8.942465,5.145962,-0.674983,6.004619,-4.156094,-4.343085,-6.488523,2.966037,-3.710742,-0.143056,-1.153130,-3.790279,-7.455974,7.647312,1.150057,-2.902440,-2.416796,9.757124,9.449098,-2.191351,-8.788259,-4.250491,9.400463,-2.270533,-8.811485,-5.700112,6.175113,2.406331,-2.139264,3.813406,3.232748,0.058420,2.865557,-3.858532,8.410469,-2.815733,4.272574,6.200189,5.230210,0.042528,-7.845708,-4.191885,-7.269805,4.008402,-7.985405,-9.993406,-8.881584,-1.191000,-3.629276,-2.655300,3.571750,5.396323,8.596220,9.783369,-2.959654,6.425889,-4.117600,0.909318,9.612662,0.840431,8.971782,1.393633,8.471482,9.075483,2.393509,-4.586854,-0.175671,7.293798,-2.461557,-8.483109,7.868535,2.523319,-8.855182,-0.720140,1.870584,2.548670,9.925545,-2.168717,-2.142663,-6.923076,0.283264,1.449562,1.686852,-0.732665,7.814066,-5.086114,-0.321805,0.409002,-2.924629,-5.300383,7.657279,-7.147799,-2.491867,-9.798680,-5.896327,-0.776669,7.671703,-5.840335,2.084322,-6.991088,2.013371,7.902449,-3.773225,-9.398686,4.346417,-2.134621,8.139479,2.839447,3.336839,-5.808732,8.293661,5.603015,-1.407219,2.120996,-7.438908,5.865516,-8.770093,0.329309,1.517962,-0.864446,-7.058164,5.418096,8.923440,-8.279430,-7.643714,6.765011,0.297235,-4.392811,-0.865742,8.876660,7.589144,-3.077627,-0.219724,-3.492527,-8.194071,-6.491721,8.437603,-7.367910,-4.663038,7.265262,7.536498,-1.531210,-4.908298,4.450603,-6.990068,9.432897,9.211715,-4.065221,-5.909399,2.487411,7.392025,4.012046,8.946781,7.310853,-1.652815,0.598475,9.541532,8.571100,-3.315781,-4.834885,8.304811,0.249543,-3.050297,-5.055808,-8.300821,2.351960,2.408466,7.470137,-9.163622,7.864707,9.204148,-5.875144,-6.277077,-4.047577,-4.589461,-6.355711,4.360677,-8.473762,6.827549,-9.642117,-9.674015,2.215288,-6.366670,1.447192,6.963751,-0.943896,6.981701,-0.608632,-7.530253,-8.882043,7.171933,-0.601293,-8.341993,-4.847537,-5.264283,-8.201500,-6.402954,-0.374442,-5.896345,3.260165,-2.409401,-5.123556,5.284983,-5.345156,-8.987597,-3.833518,2.880728,-6.567684,-5.709248,5.064942,9.805733,7.848888,8.556979,-9.192218,7.126519,1.412907,9.393765,9.530943,-3.828106,-7.333464,5.692886,9.836246,-8.265603,-5.307539,-5.632259,3.705949,3.202163,-5.238001,-5.109540,-4.856235,-2.985967,-9.260355,-2.070237,-5.449242,8.771314,9.100391,-6.837686,6.694419,-8.794217,4.669206,9.446028,-5.385636,4.746472,9.858423,3.442984,-2.848343,0.042877,-8.851379,-9.306534,6.960582,7.964550,-8.048208,8.654055,-7.581067,-8.657717,-8.997307,-8.138359,-2.940261,6.280237,-5.161285,-4.498262,7.398795,9.661753,4.686031,-8.846202,3.186471,4.905474,4.020861,4.052852,-2.043318,9.435489,5.010505,4.650918,-0.371731,5.601533,9.106819,7.234161,4.774399,-2.612252,-2.523311,2.087528,6.754286,-7.058034,1.570291,-6.432418,9.290713,9.996957,-7.880891,-9.480600,7.623619,7.030566,4.625572,6.545549,8.185233,0.841427,2.922307,8.860200,3.667771,4.362205,2.954946,-1.433411,-5.514912,6.706867,-3.258984,2.863404,3.934356,0.762338,4.399037,9.912712,-5.414908,-8.236001,-7.966243,-5.523624,3.694180,-9.135245,-2.985213,6.877738,-5.559553,2.739431,8.173861,5.764564,5.576164,6.738068,8.453852,8.954422,-9.466445,8.023558,5.604485,-4.077760,-4.694165,2.884470,-9.591008,0.943353,8.232280,9.731062,9.727816,9.778135,5.899583,9.281997,6.316621,4.951610,-0.569741,-5.169198,-0.469686,7.685979,-3.218070,1.274103,2.318946,3.528750,0.568307,3.932265,9.405278,9.455967,-5.769112,-5.532427,5.187758,7.694082,-6.454135,3.635154,-0.922651,3.863946,0.703741,8.475784,9.944268,0.222948,-9.844774,2.084317,1.223721,-0.567057,4.799053,-6.409479,-7.727886,8.937391,1.080069,-9.227961,6.640362,-4.823450,-2.630970,6.974634,7.015576,9.137979,-3.130670,2.544213,3.849031,7.411058,-9.852499,-5.286716,9.660784,-7.056929,-9.028744,-1.546851,-2.086608,5.100718,4.375121,9.361703,7.367705,-1.195189,7.629539,-0.443377,1.414778,-9.008504,9.737090,6.948521,-9.638233,8.598226,-8.165519,-8.258217,7.083969,1.145594,-4.092110,8.417601,7.365922,2.796641,6.821536,-0.373108,7.385565,5.589690,8.468225,0.049029,-3.731483,-8.674150,0.487098,8.112934,-7.924265,8.082956,2.638626,5.137226,-8.949572,-2.626598,-0.355801,-1.165514,6.031642,-4.229225,-1.546756,-2.282309,-0.074612,5.083473,4.969790,-3.634661,-1.111142,-0.837520,-2.895008,-2.910269,8.795112,1.206897,7.518191,7.048242,0.219470,6.993244,-7.085719,-2.059524,8.456531,-7.033310,-4.269094,-6.063781,-2.982688,0.545898,-5.426586,3.750908,-2.963922,9.037781,-1.545969,-6.997257,-4.770719,-9.575182,1.520186,-1.145613,-7.764998,3.201557,6.991292,4.320489,-0.213103,6.648412,1.747989,-2.014969,3.520120,6.085215,1.219844,5.307326,-2.945118,-7.405530,-2.507510,-3.371496,7.847593,2.316383,5.136140,-4.582751,9.250793,7.558716,-2.949511,-8.817080,7.559398,4.930701,2.148230,1.184586,-2.091992,9.452100,-8.120923,-3.250727,9.142553,-5.599381,-2.693493,4.643919,-0.405667,-8.844233,-1.498256,-3.744260,2.542597,-1.593765,-6.660388,-4.217814,2.212227,3.558476,7.085547,3.334116,-0.718766,6.602889,-3.687806,-8.463777,9.318878,5.931779,-7.271812,-1.782439,-1.482222,-4.597926,-4.930704,4.058590,2.084167,4.683608,-1.661296,9.149077,-4.024807,-7.453797,7.450534,-0.223469,7.771055,1.358090,6.136515,6.410293,-6.046744,2.652153,-5.088648,1.905966,7.665136,-6.208328,1.919876,-5.016687,4.128517,3.741758,-2.060678,6.439100,3.815886,1.027467,-8.392975,-9.602594,-4.458359,9.439272,3.389889,-2.867962,8.670316,-9.455333,2.932857,4.005196,3.755478,2.944741,-2.993100,-6.340112,6.041556,-2.594624,-3.879336,9.510585,-2.613806,-9.386496,1.198072,0.319815,7.821813,8.644476,-1.595181,9.985448,4.475571,4.796832,-1.739162,-7.954511,1.404920,0.995755,8.005402,8.621091,-8.109473,9.272573,-3.323957,-0.084842,2.681264,-1.245252,-7.146710,3.111144,0.483909,0.094665,-9.252850,3.806658,8.066491,-5.805772,9.498102,-6.168516,8.897565,4.589791,-8.970879,4.518431,-0.177742,-1.509026,-1.136012,-7.678473,-7.178871,8.047905,0.172324,7.076494,7.017864,-6.685387,9.350976,-4.925000,-3.161190,-7.668093,-2.008927,-7.753641,-4.627053,5.451230,4.540003,-6.156111,-9.399162,-6.020551,-4.220831,-8.927130,2.508336,2.872323,6.808663,0.299303,2.722190,6.377753,-2.227740,-3.448527,-8.963099,8.951940,-8.393016,2.741372,6.033259,-9.950406,9.866781,5.847537,8.814393,9.598574,5.464371,2.868005,3.589660,-6.068247,-8.278273,-2.099414,-8.446834,-5.223098,7.813279,0.473734,-7.429416,-8.618807,-2.471863,-0.214977,-0.947607,2.681958,8.849724,7.527761,9.019664,-3.285909,-9.062350,-4.872641,-0.404598,0.466156,-0.307245,-9.872352,2.640194,1.121434,-5.566383,9.607161,8.209818,-9.153970,9.041045,9.227419,3.001850,1.110284,-5.373031,7.571287,-0.916484,-8.325520,1.389645,3.376492,-6.463617,5.513585,8.580874,8.556758,5.952745,5.545799,7.956354,6.755355,-1.328577,4.449023,-5.152729,1.540810,1.296678,0.197800,-2.936407,9.824999,7.768027,1.617488,-4.056976,-7.313080,7.663743,-1.170937,-3.669644,7.401348,-9.563650,-2.996620,8.989848,-2.717126,0.294083,-2.750616,-3.616596,-8.061179,-6.396852,-4.441144,6.185004,-3.178755,-5.168879,8.320778,-6.225453,8.533328,5.303145,3.667421,0.782567,-1.250484,9.334835,-9.956094,-4.965283,-5.635729,6.986752,8.256340,6.722913,9.567703,7.176882,-6.042572,1.145086,-1.465781,4.575546,-2.447674,7.621056,-8.764144,9.444075,9.844709,6.212290,-5.099779,-8.517632,6.380616,6.848031,-0.859864,-3.382937,-4.306646,-6.658840,-9.107209,-4.967885,9.133663,1.652467,-2.236318,-2.646442,8.300665,7.786031,-6.778981,9.456714,9.203458,-2.804304,3.359481,9.860692,1.726980,3.998845,9.062517,-3.911491,6.447617,-1.369315,-0.541419,2.463135,8.172350,8.903436,5.560255,7.069969,4.035567,-9.176044,-7.277274,-9.630421,8.767887,-8.894972,0.359240,5.014090,-8.863568,-3.164427,3.168063,-9.699476,3.166878,-1.255795,7.361296,4.885294,-5.257008,6.270088,6.746789,-0.933533,8.107491,-0.096540,3.926710,-0.780517,5.258910,-1.854414,-9.370744,4.029796,-3.198131,-3.039876,-6.084773,-2.350691,4.670130,7.607460,0.240694,-0.752796,-4.005757,0.124043,-3.698966,-3.651677,-8.937973,-5.732270,5.695615,-1.156295,-4.635802,6.489373,8.887881,2.328006,-5.090958,6.425858,-3.639745,-5.062112,-0.638939,-7.318442,0.604671,9.975049,3.968882,0.743618,5.283908,-3.798523,-4.534130,-8.868584,7.512267,1.183102,3.490067,8.239064,-2.931920,-4.288665,3.831697,4.537050,4.251692,-4.841983,2.981712,-5.228906,-5.563370,-2.381888,-5.857475,-8.455189,-2.197044,3.306993,-8.652397,7.432438,7.732531,-2.346059,-3.944397,9.240919,-1.401939,-2.689966,3.298023,-3.098694,1.546268,3.403135,-8.844826,8.144624,1.257979,-0.209426,9.746847,-1.224391,3.526088,4.875263,-3.766032,-6.041966,-1.918249,-1.375434,3.273073,3.194127,4.806753,-0.257341,-7.359104,-8.222320,5.564997,-7.745019,-8.591484,6.055924,5.262465,6.294501,2.310455,-5.810175,-5.282824,-8.287595,5.434533,-3.437415,-2.950854,-9.895136,-4.266716,-5.064967,5.959068,-7.022785,2.908435,-7.360534,-4.909761,-4.535376,0.687525,-0.221312,-6.130842,3.005254,-5.013474,-4.765265,-7.582145,-5.186971,-7.803043,-4.316536,0.885770,3.206749,-7.899425,0.477697,-6.750712,9.342947,-8.841080,0.448135,-5.482974,-8.725855,2.604808,1.649225,-1.444262,-5.030160,-6.504988,-3.256704,-7.752319,-1.398782,6.275095,9.326324,4.280261,2.092104,-0.471571,-5.141689,8.239433,-8.101964,-3.069970,9.317071,8.076636,9.658106,-7.086165,1.163264,6.771585,-3.157360,4.272535,-4.019266,-3.779293,-5.392079,7.611912,3.156953,2.746401,1.952101,1.693163,-4.855573,-6.452556,-3.439002,-7.321807,1.748580,8.449889,-1.768144,-4.704604,9.716989,9.016516,9.160832,7.127190,8.420426,-6.240462,-8.403621,2.389957,-9.082094,-1.548085,3.254553,-3.331240,-6.396123,-4.477831,-7.529641,-1.638974,1.839303,1.221495,-2.685081,0.658459,-4.623340,-0.262682,7.453356,0.697872,2.064944,-9.905064,-4.179719,-4.115707,-5.738462,-9.182435,8.956552,-6.324523,-3.542891,8.029810,-4.211997,-4.417285,2.014663,4.706973,2.011021,-7.875570,-2.809155,7.209725,-2.768142,-3.848365,-1.950386,1.165513,8.474961,-4.135406,-0.772931,-9.426660,-0.365172,4.161494,8.760419,-5.568542,1.308809,7.026595,1.326261,7.204412,-8.521749,-5.760633,-0.719840,-1.348246,7.374167,9.828155,4.751546,7.025053,-1.073535,7.998330,9.302960,-2.552816,-1.692409,6.064241,-0.460600,2.152499,-6.735765,2.698570,-4.465864,-4.714860,-9.781472,0.987124,6.024405,8.096028,4.801339,0.055265,-6.951852,2.609837,0.293024,-5.391931,9.137719,1.276683,7.815679,-0.811425,-4.268071,-9.127149,-6.395250,-8.689210,-2.719405,-3.835690,-6.151831,-4.563214,9.283958,-7.675466,-7.511449,1.378265,-5.748070,3.518947,6.326110,6.855819,-3.073108,5.400808,7.791584,-5.710114,9.533464,6.795499,-9.703922,1.561522,-0.535723,6.912031,-2.513826,-9.138774,1.973604,6.575536,-3.649317,-5.458733,-0.417481,7.194209,-0.412404,9.988642,-7.652582,2.118525,-8.719611,0.621624,-0.923474,-2.984250,4.686298,-0.111010,-8.195832,7.592832,-9.769611,-7.764423,4.365397,3.832111,2.641930,-9.533300,8.635095,9.391446,-4.748090,-2.810543,0.647436,6.786613,-0.244441,4.447203,6.136951,9.691662,-9.715435,-2.682980,7.199623,3.065538,-6.590436,4.483919,-4.237838,-7.270903,6.356900,7.530149,5.858484,9.495127,-9.095725,-8.815810,6.201744,-2.498413,7.370353,-2.632593,-3.769480,-2.942819,5.561363,-5.705581,-4.423224,-7.710733,-4.574334,-1.690653,-2.218951,3.650152,-3.781145,2.024672,0.182709,6.423726,-5.144196,4.496573,-6.881835,-5.601714,4.324246,3.466268,0.855504,2.977338,2.608964,2.216769,5.249721,1.228647,-4.391522,-7.338833,-8.614618,5.585661,2.742418,5.811513,-1.877026,0.466735,-0.327589,2.527948,-4.409853,6.231485,-0.231040,2.363909,3.997614,2.646764,7.980242,0.333520,4.936786,1.852836,-3.884979,-0.186198,-6.732963,-0.192114,-1.671291,-0.548719,1.250547,6.643174,-6.326500,9.616065,0.108337,7.160964,-5.902684,-5.780485,9.838451,-0.128620,-9.260454,-9.039912,-6.005909,8.244797,-3.774743,9.201297,-9.931344,0.406856,-2.504697,-8.828988,6.869542,-2.243150,1.403967,-5.189519,-7.305306,-5.163391,5.770379,7.305472,5.399972,3.552902,-5.298291,0.210050,1.846973,5.341370,-7.502217,5.232883,-0.838185,0.684323,-4.773800,-1.491195,7.232892,3.039558,-3.455375,-9.034989,7.846816,7.290270,2.893868,-3.670395,-4.419269,-5.035236,-9.473077,-0.095531,0.916070,-7.072701,3.405178,0.637939,3.931916,-0.047045,7.242175,7.693065,-5.995577,8.081356,-0.462568,-5.907289,5.081963,1.336132,-3.809760,-5.274171,7.619578,5.344558,-8.277639,4.821837,6.150841,6.634775,9.350380,7.188349,5.941361,7.366951,2.470928,-5.331392,-8.554505,-9.334465,-0.207701,-6.281434,9.788841,-0.140031,-8.796406,1.414013,9.154570,1.010233,3.931695,-6.373282,3.763522,-2.311677,4.687545,1.635368,-6.927326,-9.039590,1.014133,0.773621,-9.661519,-2.409575,1.600099,-1.141721,-1.517022,3.636096,6.063329,-6.281528,1.718163,5.993472,-0.898165,-0.993800,-2.632581,1.572288,-0.753287,-3.184053,-7.748569,5.154263,-9.633109,4.649609,9.570515,-0.421343,-7.919973,3.629300,-2.474098,6.167145,-3.186101,1.774363,-9.194448,-6.001658,5.127608,-4.888163,0.187394,-3.113088,2.848724,-7.623491,-8.330436,9.186022,9.949132,9.757795,5.576145,-9.823697,-4.098989,-5.989191,-6.845341,9.783278,6.565953,-3.958383,-6.629105,-7.644333,-1.232569,-3.660560,-7.774940,5.111665,7.651207,4.965155,8.004976,-9.255215,4.577622,8.248571,0.229708,-2.843249,-3.169473,9.335467,9.815020,-7.043059,-8.128438,-0.468962,5.516722,-5.557755,-3.976074,3.736373,-2.829684,-5.031918,8.758780,4.620384,-5.538096,4.488056,-4.898604,9.306172,6.633041,6.849463,7.051302,-8.855494,-9.588076,8.171525,-1.772329,-4.633123,-7.980041,-4.670969,-6.659496,-3.197921,4.408097,3.114508,-6.527298,2.994927,3.602916,1.324240,7.096367,3.466785,-2.222665,-8.050398,4.752606,1.485836,7.735059,9.411806,-5.735489,5.622164,-7.767747,1.536222,2.476942,-7.362073,-1.329715,6.611696,6.731278,6.542292,0.403703,-2.803919,-0.213383,-3.033827,0.885161,-5.308037,5.383998,-2.394563,-7.933393,-4.424501,-0.728814,-6.967900,-8.939262,-7.315271,-4.771214,-2.499785,-8.953947,2.934292,-9.965755,-9.945684,1.154230,7.446641,-1.086505,-5.967801,-9.493628,8.954742,0.005692,2.200396,6.539972,9.580889,9.101837,-0.164187,-6.313119,-9.526922,5.034649,2.962867,3.455812,-8.782726,-0.601406,6.330781,2.564023,2.980265,-0.563308,5.531996,9.975686,-0.349231,-7.625344,-2.684740,-8.438462,0.925601,4.558991,-3.462729,-2.565035,5.711829,-9.605037,9.837500,8.784987,-3.440953,3.755090,2.934704,6.744716,3.529631,-3.401344,-3.413609,-0.214203,3.439368,-5.397873,4.373379,-6.258143,9.830879,6.357992,-0.051194,-8.643409,-5.280616,-6.930379,-2.406799,9.291690,5.451311,2.033579,-2.773818,-3.059893,3.119941,-6.773260,0.960821,2.751871,5.430836,8.574173,9.833689,-9.739992,6.255384,-7.758766,-3.245472,2.798113,6.129090,-7.351652,-6.545333,7.710918,3.911311,-5.780446,-2.847696,-7.160050,-9.567661,-3.239272,9.991561,-4.051167,6.448539,-9.256383,-4.854257,-2.323847,-6.045437,-2.899214,2.041649,-1.849525,3.964262,3.237164,3.490701,5.561932,5.163345,2.778180,-9.606618,4.150660,-1.743481,-8.877399,6.882404,9.384361,8.919594,-3.213097,9.521080,8.175328,-7.362624,5.585526,-3.108893,5.412411,-7.413968,-7.160683,-2.374060,-6.848248,2.471924,-3.678719,2.966686,-8.760158,2.814383,3.574915,-0.047500,5.183326,1.930456,-1.393433,0.689180,0.756016,-3.198323,-6.955133,1.243663,-2.044489,4.524037,6.853121,-1.653771,9.157321,4.081446,-1.746133,-9.905464,-4.130258,-7.009885,8.807491,-7.193205,3.390866,3.012895,-1.293177,8.216224,0.910881,9.581834,0.613819,8.697286,2.165823,-9.077908,-2.846368,0.686888,-3.295963,6.030009,8.915718,8.284893,-8.159073,-2.954194,3.922011,7.506699,4.594081,-2.392188,-7.096259,9.353181,-1.943586,-8.676618,-0.825366,7.796973,3.976207,3.647558,-5.465052,-8.198699,1.564534,4.133746,-1.993595,6.526052,8.035639,-9.957401,7.820498,9.980574,2.283770,-0.609256,4.988907,0.473448,-6.801624,8.003127,4.121731,-4.403353,3.246091,7.598920,-5.211705,7.575815,-1.351071,-4.624334,2.861688,8.611694,2.774836,8.112945,6.157474,7.173636,8.669507,8.455902,-1.205624,8.352876,5.845203,4.841508,4.612717,-3.827622,3.203381,-6.227254,-3.558724,1.283410,5.556570,9.262763,4.601547,0.038170,3.727623,0.186430,-1.241796,5.555339,-9.350248,7.778291,2.163199,7.763529,-0.898390,-7.510436,-3.226640,-9.102385,-3.564610,-8.339476,7.504724,-9.479760,-7.489874,-7.695664,-8.837541,0.814888,5.698460,-3.641687,-3.773378,1.156826,3.817426,7.267637,7.451499,-8.689589,8.817552,-5.598348,-3.768162,-4.034053,-5.748504,8.264899,-0.258186,9.014157,8.489230,7.467390,-6.026653,1.953552,-7.465697,3.754040,-7.071693,8.403693,-1.276198,-0.006127,-1.725627,-0.496438,-1.931641,7.940117,-6.199063,-9.106480,-2.351587,6.648093,-1.391689,3.239826,8.015479,-1.199820,-2.119757,-3.297510,4.713010,9.206378,7.197196,4.126595,0.219794,-6.610830,7.420722,3.986296,-7.452789,-1.726949,2.236892,7.904256,-1.328507,4.183650,0.663903,-6.143632,-9.430210,-4.288832,7.921889,6.876565,-5.813678,-0.857692,-2.879288,-7.800704,-4.784473,0.420596,-3.239721,-4.684016,1.683127,2.539740,2.930091,8.159833,5.003013,1.058932,0.461124,3.302315,7.743880,-4.002428,8.543226,-0.095093,4.413901,-4.811135,5.337270,5.584323,-4.804371,4.291456,-1.893928,4.972757,6.958875,-8.336496,-0.495323,-9.254575,9.505745,0.312339,6.288292,-9.405283,-0.343060,-0.502009,3.920357,-5.570670,-8.421543,-0.452695,-8.886868,-6.453861,-1.216367,6.168419,-3.200774,-2.380712,-9.736479,4.375529,-8.145989,-4.005963,-6.153204,-9.481893,-6.606879,1.023079,-3.989914,1.932828,0.810953,-1.496742,-4.375070,-6.378010,-0.173068,-2.490428,0.143389,-6.766530,1.460494,-4.791877,-5.018889,-0.955398,0.975416,3.668534,3.089168,-2.160524,3.794079,-1.655136,-0.789022,-5.510720,-9.806043,3.527960,-5.145256,8.274581,7.537900,7.842941,-6.733185,-6.635985,0.318100,3.976347,0.267548,-8.789195,2.497744,-3.771475,6.086809,-9.739909,-2.377562,4.862244,9.559916,-1.393171,-1.133076,-1.030091,9.945895,-4.011936,6.210356,0.246202,-6.855095,2.623035,1.886141,0.867448,-8.899395,-7.941575,-5.229206,-3.012975,-2.034806,-8.387283,-4.032745,9.032816,4.711086,4.903593,8.529897,7.306498,-7.801834,7.840607,9.750321,-0.045838,-0.497190,-1.629472,2.163273,9.627772,3.290699,-2.972693,-4.658428,-3.029332,-6.526346,-9.865494,-1.326449,7.593973,-6.608105,-7.512499,2.135082,-7.913710,1.429390,-8.616010,2.475275,-2.768710,2.037435,-2.217195,-3.190570,-6.324378,-0.679078,2.611566,-8.941126,9.435107,3.357842,-5.989816,7.065631,3.102172,6.818246,8.423878,2.542062,-2.705474,-0.749549,-5.543003,8.681137,5.056097,-3.330747,-3.033051,4.403454,7.797513,-7.403508,-8.326369,-1.822986,-5.946532,8.624609,-0.529288,-6.296714,-1.904243,-0.317968,4.155520,2.897071,1.533910,2.823788,-5.690495,-6.431333,-5.913463,-2.641991,-4.754797,-8.433249,-7.601871,-2.430763,-6.697880,2.595591,4.278313,0.114117,1.331767,-1.242745,-5.030769,5.992345,-1.951842,1.945792,-0.331566,2.886912,-3.647935,4.499260,-2.601658,-0.601013,-3.178087,1.319757,2.532808,7.162606,7.405889,8.067592,1.344293,-2.301047,-7.064595,-6.562949,1.170006,9.199025,3.696170,9.435453,-3.452786,-5.603245,0.347097,4.541879,9.874528,-6.523643,-7.028013,-3.485509,-1.402258,2.466073,-0.349630,7.646713,-1.621895,-2.914481,9.958363,5.687925,2.519410,5.595769,2.027578,4.914387,3.679295,6.633941,6.288314,2.598025,-7.522131,4.188196,4.988244,8.462340,2.269904,0.156673,3.918536,6.507479,7.340246,4.670621,8.707773,4.700570,-6.396608,-0.716931,-3.910293,5.749640,-3.623348,-2.359300,6.030567,3.891183,8.265452,-8.798916,3.774373,8.314766,-4.956199,7.835541,6.803691,-7.530067,1.989485,5.315931,-1.860933,5.399506,8.724577,-6.994408,0.012395,-5.010307,3.835437,3.431867,-4.109272,5.878093,8.912390,-9.626279,5.861190,0.118792,9.168178,-8.663080,1.105921,-8.229811,2.042498,-1.558383,5.367839,1.849588,4.814196,3.573698,6.882176,-8.225705,-8.371640,0.227148,5.716366,-3.810826,-4.476008,-2.296318,-9.053845,-8.073646,-1.175680,-1.013489,0.816130,-3.008295,-0.526885,-9.934301,-1.896311,2.313226,1.682699,-3.795876,9.188382,7.421910,1.121186,8.976749,2.867496,-7.169676,-6.318398,7.005765,-0.564591,-4.332586,6.908418,7.543397,7.421819,-7.408523,6.852136,4.804467,-9.804711,-3.835679,1.203765,0.279394,-9.944479,-6.626613,2.947866,-2.027696,7.650460,-5.473869,-0.090047,-0.600853,-3.856624,-3.724398,6.576393,-5.256320,8.639739,9.107774,8.206755,8.731451,-2.167097,-0.406160,1.960237,-7.348040,-8.220795,6.395021,-8.534055,-4.426126,8.884570,-2.662859,7.610116,-4.553480,-1.994326,6.560992,-9.606532,8.162179,-1.230040,-0.016970,-3.394247,-8.048190,-1.956889,5.668476,7.386640,6.747780,1.685045,7.811700,2.877302,2.315539,4.833407,3.402872,6.887310,5.091668,-6.994344,0.039272,-3.600906,3.606497,6.654604,3.863806,1.012208,8.570082,-1.016503,5.762887,-6.026489,8.877973,5.299431,-6.981199,8.429469,-0.965939,3.738046,5.153866,5.848815,-2.263088,0.560553,8.494778,5.238415,7.819762,2.721666,6.568123,9.902468,0.918453,9.579882,9.899885,1.754472,-1.151820,1.628228,-5.301947,1.323840,-9.870716,-8.514209,5.521924,-6.700033,-5.663898,1.505326,-1.578195,-5.194020,8.799337,1.745205,7.490806,-0.001590,4.228064,0.624710,-4.267070,7.228617,5.056224,-2.444860,5.438357,-3.750456,-0.374913,5.166561,-1.906284,-6.368625,6.947111,9.743230,4.752056,-1.984713,2.809892,-1.804812,-4.963296,-3.638143,-0.392765,-8.464424,-6.322217,2.339876,-7.214756,2.681425,-9.545875,-8.749766,8.142862,-4.657205,0.477398,6.353596,-9.855030,-9.553562,5.138725,6.992835,5.811574,8.706798,-7.665858,-3.371452,4.481415,1.525245,-3.149557,8.916237,9.352922,-5.891942,1.838024,-3.770839,7.051012,-4.203628,0.668738,-1.325105,-2.338720,-9.917906,-3.102725,4.336381,-3.262456,-0.863224,4.630137,-0.926033,0.219391,3.960770,5.943838,6.893441,3.420235,6.627267,7.760887,9.496697,-8.861138,-1.724170,8.758871,-5.916009,-1.865347,-3.496665,9.071170,3.255075,4.102276,9.379518,-0.184864,1.998638,-6.779718,-6.196675,0.140266,6.167343,-7.764579,-9.413111,-4.611927,9.737380,0.271448,-6.456745,6.065425,-7.263736,-7.245948,2.420140,-6.127607,-2.448047,8.252073,-1.964530,1.246549,-2.700095,5.630555,3.279783,-0.745914,-1.628268,-2.719909,-9.789204,-8.314670,-2.846287,-8.370245,4.831837,-1.750262,9.739192,9.846848,-4.433024,-3.931587,-3.868030,-5.515735,8.324802,-7.126594,-5.562224,-6.674056,-5.224366,5.185895,2.752680,-2.452172,-2.148460,2.633179,4.956501,0.913907,2.621330,-6.630355,6.385009,1.812579,1.299122,5.121325,-4.867007,-6.247912,-4.915216,6.753986,0.540094,8.969129,-9.479891,-5.696365,4.177475,-1.735609,3.011234,-2.257897,4.178524,2.917172,2.231229,-1.435384,-4.529265,8.278538,9.693684,-6.109175,-6.921659,-8.838066,-8.233391,-1.890336,8.311195,-1.254584,-4.514692,1.255455,-2.313819,-9.968283,9.407309,4.194766,-8.905861,2.208215,6.218925,-3.840166,9.917213,0.771346,0.160739,9.023168,-2.594583,5.362284,-9.784942,-4.469371,6.817047,-1.708260,5.954552,-3.058554,-6.450513,0.666684,-0.569582,-1.811621,6.150380,5.088820,1.902282,8.926439,4.125882,-9.881093,9.780707,-3.861972,-5.690449,-3.698695,0.424936,1.905819,-7.389167,5.960236,-3.334764,-9.666162,0.071425,6.525241,-4.696936,-0.933173,0.082487,9.975508,2.410364,-8.456426,-3.983346,-6.063225,4.825771,-2.402773,-4.720870,-3.157141,9.502817,-1.989139,8.575287,4.418299,-6.970153,7.382044,8.673963,6.943333,-1.871022,-8.695643,-6.932219,-9.411886,-7.948221,6.211683,6.774190,1.431312,-2.530124,8.033312,0.120365,0.823447,7.515698,-2.138838,9.351656,6.931259,2.056037,-0.749341,-5.404001,7.932489,1.814245,-7.541144,9.975598,7.685629,8.131884,6.774381,-1.104210,0.061649,4.636467,-7.215604,1.459189,-1.983278,1.279534,-9.273384,-8.360373,-7.664524,-3.970631]], dtype = "float32")#candidate|2026|(1, 2640)|const|float32
call_2025 = relay.TupleGetItem(func_1475_call(relay.reshape(const_2026.astype('float32'), [15, 11, 16])), 5)
call_2027 = relay.TupleGetItem(func_1477_call(relay.reshape(const_2026.astype('float32'), [15, 11, 16])), 5)
uop_2030 = relay.sin(var_2015.astype('float64')) # shape=(8, 10)
output = relay.Tuple([bop_2022,call_2025,const_2026,uop_2030,])
output2 = relay.Tuple([bop_2022,call_2027,const_2026,uop_2030,])
func_2032 = relay.Function([var_2015,], output)
mod['func_2032'] = func_2032
mod = relay.transform.InferType()(mod)
mutated_mod['func_2032'] = func_2032
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2033 = relay.var("var_2033", dtype = "float64", shape = (8, 10))#candidate|2033|(8, 10)|var|float64
func_2032_call = mutated_mod.get_global_var('func_2032')
call_2034 = func_2032_call(var_2033)
output = call_2034
func_2035 = relay.Function([var_2033], output)
mutated_mod['func_2035'] = func_2035
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2047 = relay.var("var_2047", dtype = "float64", shape = (3, 13))#candidate|2047|(3, 13)|var|float64
uop_2048 = relay.sin(var_2047.astype('float64')) # shape=(3, 13)
uop_2061 = relay.cos(var_2047.astype('float64')) # shape=(3, 13)
uop_2064 = relay.log2(var_2047.astype('float32')) # shape=(3, 13)
output = relay.Tuple([uop_2048,uop_2061,uop_2064,])
output2 = relay.Tuple([uop_2048,uop_2061,uop_2064,])
func_2067 = relay.Function([var_2047,], output)
mod['func_2067'] = func_2067
mod = relay.transform.InferType()(mod)
var_2068 = relay.var("var_2068", dtype = "float64", shape = (3, 13))#candidate|2068|(3, 13)|var|float64
output = func_2067(var_2068)
func_2069 = relay.Function([var_2068], output)
mutated_mod['func_2069'] = func_2069
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2162 = relay.var("var_2162", dtype = "float64", shape = (16, 11, 1))#candidate|2162|(16, 11, 1)|var|float64
uop_2163 = relay.erf(var_2162.astype('float64')) # shape=(16, 11, 1)
bop_2167 = relay.less_equal(uop_2163.astype('bool'), relay.reshape(var_2162.astype('bool'), relay.shape_of(uop_2163))) # shape=(16, 11, 1)
func_1567_call = mod.get_global_var('func_1567')
func_1570_call = mutated_mod.get_global_var('func_1570')
var_2183 = relay.var("var_2183", dtype = "float64", shape = (33,))#candidate|2183|(33,)|var|float64
call_2182 = relay.TupleGetItem(func_1567_call(relay.reshape(var_2183.astype('float64'), [3, 11])), 0)
call_2184 = relay.TupleGetItem(func_1570_call(relay.reshape(var_2183.astype('float64'), [3, 11])), 0)
output = relay.Tuple([bop_2167,call_2182,var_2183,])
output2 = relay.Tuple([bop_2167,call_2184,var_2183,])
func_2187 = relay.Function([var_2162,var_2183,], output)
mod['func_2187'] = func_2187
mod = relay.transform.InferType()(mod)
mutated_mod['func_2187'] = func_2187
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2187_call = mutated_mod.get_global_var('func_2187')
var_2189 = relay.var("var_2189", dtype = "float64", shape = (16, 11, 1))#candidate|2189|(16, 11, 1)|var|float64
var_2190 = relay.var("var_2190", dtype = "float64", shape = (33,))#candidate|2190|(33,)|var|float64
call_2188 = func_2187_call(var_2189,var_2190,)
output = call_2188
func_2191 = relay.Function([var_2189,var_2190,], output)
mutated_mod['func_2191'] = func_2191
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2265 = relay.var("var_2265", dtype = "float32", shape = (11, 5))#candidate|2265|(11, 5)|var|float32
var_2266 = relay.var("var_2266", dtype = "float32", shape = (11, 5))#candidate|2266|(11, 5)|var|float32
bop_2267 = relay.floor_divide(var_2265.astype('float32'), relay.reshape(var_2266.astype('float32'), relay.shape_of(var_2265))) # shape=(11, 5)
output = bop_2267
output2 = bop_2267
func_2276 = relay.Function([var_2265,var_2266,], output)
mod['func_2276'] = func_2276
mod = relay.transform.InferType()(mod)
mutated_mod['func_2276'] = func_2276
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2276_call = mutated_mod.get_global_var('func_2276')
var_2278 = relay.var("var_2278", dtype = "float32", shape = (11, 5))#candidate|2278|(11, 5)|var|float32
var_2279 = relay.var("var_2279", dtype = "float32", shape = (11, 5))#candidate|2279|(11, 5)|var|float32
call_2277 = func_2276_call(var_2278,var_2279,)
output = call_2277
func_2280 = relay.Function([var_2278,var_2279,], output)
mutated_mod['func_2280'] = func_2280
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2343 = relay.const([[0.258137,-7.517421,-0.522343,-5.465197,-7.719228,6.006838,-9.345803,-7.790993,4.136434,-4.789141,3.589601,-8.366240,9.625350,1.296399,-4.280425],[4.627504,-7.313013,-7.032443,-4.013222,-0.829167,2.637801,1.157836,3.130999,5.616286,4.113284,2.040933,-6.960753,-2.357592,1.969859,-8.433998],[-8.931759,3.410211,1.294578,5.957519,-7.169035,2.632360,0.956879,5.167578,-6.797276,6.187817,-6.124471,-0.026936,-2.417323,-9.857570,9.754944],[-4.316551,-9.516693,-4.147239,5.238024,5.491689,-5.310764,-5.495432,-2.471411,-0.567488,-8.630147,-7.552395,1.497955,2.080629,3.998281,4.109196],[-8.695908,-6.331022,6.372448,6.708025,-6.813621,-7.195310,9.330765,-1.409011,-6.514244,-9.609806,8.725760,6.143644,7.789980,-8.042913,-8.038931],[3.199324,9.110096,8.687736,8.215078,-6.244168,6.236860,-5.716890,1.070306,-4.972467,9.555412,-7.557585,-7.167457,6.266576,-5.819351,-9.112149],[1.959170,1.760288,-5.408920,0.766074,4.522711,8.382573,-3.386876,5.967944,3.659638,4.022629,-0.589377,7.007716,0.621127,6.626205,-2.332656],[-9.653776,8.700871,4.079269,-3.180834,3.149874,-1.273618,8.022130,-8.257986,-2.499543,-1.997822,6.886466,2.364891,4.743231,3.541879,-5.432665],[0.924963,-7.761270,-3.034449,-9.469113,-1.287287,8.098586,-7.204924,-1.564874,9.387343,-2.580309,-2.754332,9.434295,-5.642531,1.243443,7.591097],[-0.495047,-0.554069,1.948269,-5.228793,-7.569884,-2.714840,6.965119,-9.994701,4.176079,-4.109609,9.050688,3.653272,-7.966337,-8.319260,-2.747145],[1.817461,-6.262068,2.500086,-7.500338,1.471512,-1.350639,-0.247362,-2.991824,4.790013,4.634530,-7.451910,7.382492,8.623321,2.398398,1.938142],[3.864672,-1.531217,-4.326450,7.928010,6.969743,4.657919,-0.828726,9.352497,-5.633847,8.311706,-4.293729,2.517330,7.280174,0.517938,-8.014695],[-5.499923,5.309437,1.100229,-2.242693,3.274134,3.579296,-0.646174,-7.871298,-5.814515,-1.044099,-0.309281,5.673487,-5.939214,-4.170797,-7.405461],[-4.880724,-9.432781,4.934756,-2.591960,6.346094,-5.277957,-7.952557,-0.609502,-0.411930,-4.750864,-2.745046,-3.543700,-9.722982,2.131683,-0.897886]], dtype = "float32")#candidate|2343|(14, 15)|const|float32
var_2344 = relay.var("var_2344", dtype = "float32", shape = (14, 15))#candidate|2344|(14, 15)|var|float32
bop_2345 = relay.floor_mod(const_2343.astype('float32'), relay.reshape(var_2344.astype('float32'), relay.shape_of(const_2343))) # shape=(14, 15)
func_1750_call = mod.get_global_var('func_1750')
func_1753_call = mutated_mod.get_global_var('func_1753')
var_2350 = relay.var("var_2350", dtype = "bool", shape = (84,))#candidate|2350|(84,)|var|bool
call_2349 = relay.TupleGetItem(func_1750_call(relay.reshape(var_2350.astype('bool'), [6, 14]), relay.reshape(var_2350.astype('float64'), [6, 14]), ), 3)
call_2351 = relay.TupleGetItem(func_1753_call(relay.reshape(var_2350.astype('bool'), [6, 14]), relay.reshape(var_2350.astype('float64'), [6, 14]), ), 3)
output = relay.Tuple([bop_2345,call_2349,var_2350,])
output2 = relay.Tuple([bop_2345,call_2351,var_2350,])
func_2356 = relay.Function([var_2344,var_2350,], output)
mod['func_2356'] = func_2356
mod = relay.transform.InferType()(mod)
mutated_mod['func_2356'] = func_2356
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2356_call = mutated_mod.get_global_var('func_2356')
var_2358 = relay.var("var_2358", dtype = "float32", shape = (14, 15))#candidate|2358|(14, 15)|var|float32
var_2359 = relay.var("var_2359", dtype = "bool", shape = (84,))#candidate|2359|(84,)|var|bool
call_2357 = func_2356_call(var_2358,var_2359,)
output = call_2357
func_2360 = relay.Function([var_2358,var_2359,], output)
mutated_mod['func_2360'] = func_2360
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2386 = relay.var("var_2386", dtype = "uint8", shape = (10, 9, 9))#candidate|2386|(10, 9, 9)|var|uint8
const_2387 = relay.const([[[7,10,1,-2,-10,-4,2,-9,-8],[7,9,-9,-8,3,-6,-5,-3,9],[-9,-6,5,-4,-7,-2,-3,1,6],[7,10,7,3,9,5,1,8,-6],[5,-2,2,-9,2,1,1,-3,-9],[7,2,-7,-5,-4,-6,-7,-10,-7],[-2,1,8,3,-9,-10,-7,-10,2],[-8,8,1,-10,-8,5,5,4,3],[-8,-1,3,5,8,-8,5,-1,-6]],[[8,9,8,-9,-3,-1,-5,3,7],[-9,3,-10,-2,-7,-7,-3,10,-8],[8,-3,2,-9,4,7,7,8,5],[-3,2,-10,-2,3,-7,-10,2,-7],[-7,5,8,5,-8,-5,10,10,7],[-2,-4,-1,-7,5,-4,6,-9,-5],[3,-2,4,3,-7,-7,-4,-3,3],[7,-5,-1,-5,4,-10,-2,-10,3],[6,-9,1,-5,-4,-3,-2,-1,5]],[[10,1,-5,-3,-1,5,-9,9,2],[-6,5,7,9,8,-6,3,2,-10],[-2,-6,-7,8,2,6,9,4,1],[2,9,-1,-3,-3,-6,-4,5,5],[2,8,3,9,5,6,5,8,-7],[10,-1,5,-7,10,-8,6,8,-6],[-9,5,-8,-9,-6,-6,-6,-3,3],[-8,4,4,3,-3,-5,6,-4,7],[-9,-5,-5,-9,9,-2,7,-3,-8]],[[-6,9,5,-6,8,-9,2,-3,-4],[9,-1,-5,-3,5,-7,2,-5,3],[-5,-1,9,-7,-10,9,9,9,-8],[-3,1,6,-2,-1,9,8,-3,6],[9,-4,-2,-4,4,-10,-10,-3,2],[8,10,-10,-5,3,-5,1,-2,4],[-9,6,-2,6,-9,4,8,5,3],[-7,4,-6,10,-5,-2,10,-9,-5],[-1,4,9,3,-4,-7,6,7,5]],[[10,-1,-10,6,-6,-4,-8,3,-9],[2,4,-10,2,5,5,3,5,-2],[8,-7,-5,-7,-8,-4,5,-8,-1],[9,4,3,4,-7,-9,-6,-9,1],[-5,-4,1,9,4,-6,-6,6,-9],[-3,4,-1,-9,10,-1,-9,-1,9],[-10,9,-6,7,-8,-5,-4,9,8],[-1,-9,6,-5,-6,9,10,4,3],[2,5,5,-8,-10,-8,-9,7,1]],[[-9,8,5,-3,-7,2,-10,-2,-6],[-6,-7,-1,7,4,-9,-7,3,6],[-5,3,-7,-1,-9,1,-9,8,-5],[1,-1,6,-3,-10,2,-6,-4,-8],[4,-7,2,8,5,9,7,2,6],[9,4,-6,3,5,9,-8,5,-7],[3,2,3,10,1,9,-3,-7,-6],[-5,-2,-8,-6,-7,-4,-9,10,-7],[-1,3,2,-9,10,1,-8,-7,-9]],[[-5,-3,-5,5,-6,-5,5,1,-10],[-2,7,5,-8,-6,-5,-9,-9,1],[9,-10,-10,-7,1,-7,-10,-9,10],[5,-9,2,8,-2,-4,-3,-10,6],[-7,9,3,-6,4,10,-1,2,9],[-2,5,-6,-9,-10,3,1,10,-6],[5,-6,9,-4,4,-10,10,-8,8],[-2,4,-5,9,-4,7,4,4,-2],[-9,-5,2,3,-9,-5,-6,3,-10]],[[-4,-8,1,-10,9,6,-9,8,7],[-3,-2,-1,9,-5,-5,5,-5,10],[8,3,-4,-9,-6,-8,8,1,-4],[-6,-2,-9,5,8,7,-8,-6,2],[-9,-2,1,8,-3,-7,3,-2,-2],[2,5,9,-6,6,-7,-1,-3,-1],[-2,2,-8,-2,2,-4,7,4,-5],[-3,-10,-2,-9,-3,-5,5,-5,-5],[-7,6,3,4,7,9,10,3,-10]],[[7,8,-2,-6,2,-5,6,4,-1],[-3,6,-2,-10,-6,2,-7,-7,-1],[-2,7,10,-1,-4,-10,3,-10,3],[-3,-6,-5,-6,9,9,-2,7,-5],[1,-5,2,-10,9,3,3,5,6],[8,-2,5,5,2,-9,8,4,7],[-9,4,-6,-9,2,8,4,10,-5],[9,4,-3,-8,-5,4,-2,-8,10],[3,4,-5,-3,-10,-7,10,6,-5]],[[7,8,10,5,-7,-8,8,8,9],[4,-4,-7,-1,5,8,-4,5,-10],[-6,-9,-7,-9,10,-7,-3,-3,7],[-8,-4,8,2,1,-6,6,-6,-4],[-4,8,-4,-3,5,-6,5,-3,-6],[10,-2,7,4,9,-8,2,10,6],[-4,3,-6,-2,-2,7,-3,-5,-2],[3,8,3,-9,-8,7,-4,-4,8],[5,-8,-4,-2,5,-4,-10,10,-10]]], dtype = "uint8")#candidate|2387|(10, 9, 9)|const|uint8
bop_2388 = relay.bitwise_xor(var_2386.astype('uint8'), relay.reshape(const_2387.astype('uint8'), relay.shape_of(var_2386))) # shape=(10, 9, 9)
output = relay.Tuple([bop_2388,])
output2 = relay.Tuple([bop_2388,])
func_2394 = relay.Function([var_2386,], output)
mod['func_2394'] = func_2394
mod = relay.transform.InferType()(mod)
var_2395 = relay.var("var_2395", dtype = "uint8", shape = (10, 9, 9))#candidate|2395|(10, 9, 9)|var|uint8
output = func_2394(var_2395)
func_2396 = relay.Function([var_2395], output)
mutated_mod['func_2396'] = func_2396
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2433 = relay.var("var_2433", dtype = "uint64", shape = (16, 12))#candidate|2433|(16, 12)|var|uint64
var_2434 = relay.var("var_2434", dtype = "uint64", shape = (16, 12))#candidate|2434|(16, 12)|var|uint64
bop_2435 = relay.right_shift(var_2433.astype('uint64'), relay.reshape(var_2434.astype('uint64'), relay.shape_of(var_2433))) # shape=(16, 12)
uop_2448 = relay.erf(var_2433.astype('float32')) # shape=(16, 12)
bop_2450 = relay.bitwise_and(uop_2448.astype('uint64'), relay.reshape(var_2433.astype('uint64'), relay.shape_of(uop_2448))) # shape=(16, 12)
func_1475_call = mod.get_global_var('func_1475')
func_1477_call = mutated_mod.get_global_var('func_1477')
var_2457 = relay.var("var_2457", dtype = "float32", shape = (2640,))#candidate|2457|(2640,)|var|float32
call_2456 = relay.TupleGetItem(func_1475_call(relay.reshape(var_2457.astype('float32'), [15, 11, 16])), 5)
call_2458 = relay.TupleGetItem(func_1477_call(relay.reshape(var_2457.astype('float32'), [15, 11, 16])), 5)
bop_2459 = relay.floor_divide(uop_2448.astype('float32'), relay.reshape(bop_2435.astype('float32'), relay.shape_of(uop_2448))) # shape=(16, 12)
bop_2467 = relay.greater_equal(bop_2435.astype('bool'), relay.reshape(bop_2459.astype('bool'), relay.shape_of(bop_2435))) # shape=(16, 12)
func_556_call = mod.get_global_var('func_556')
func_560_call = mutated_mod.get_global_var('func_560')
var_2471 = relay.var("var_2471", dtype = "uint32", shape = (2, 320))#candidate|2471|(2, 320)|var|uint32
const_2472 = relay.const([9.403488,-0.618835,-3.306102,-4.194094,8.682058,3.500146,6.994545,8.633561,7.453198], dtype = "float32")#candidate|2472|(9,)|const|float32
call_2470 = relay.TupleGetItem(func_556_call(relay.reshape(var_2471.astype('uint32'), [10, 4, 16]), relay.reshape(const_2472.astype('float32'), [9,]), ), 4)
call_2473 = relay.TupleGetItem(func_560_call(relay.reshape(var_2471.astype('uint32'), [10, 4, 16]), relay.reshape(const_2472.astype('float32'), [9,]), ), 4)
output = relay.Tuple([bop_2450,call_2456,var_2457,bop_2467,call_2470,var_2471,const_2472,])
output2 = relay.Tuple([bop_2450,call_2458,var_2457,bop_2467,call_2473,var_2471,const_2472,])
func_2495 = relay.Function([var_2433,var_2434,var_2457,var_2471,], output)
mod['func_2495'] = func_2495
mod = relay.transform.InferType()(mod)
mutated_mod['func_2495'] = func_2495
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2495_call = mutated_mod.get_global_var('func_2495')
var_2497 = relay.var("var_2497", dtype = "uint64", shape = (16, 12))#candidate|2497|(16, 12)|var|uint64
var_2498 = relay.var("var_2498", dtype = "uint64", shape = (16, 12))#candidate|2498|(16, 12)|var|uint64
var_2499 = relay.var("var_2499", dtype = "float32", shape = (2640,))#candidate|2499|(2640,)|var|float32
var_2500 = relay.var("var_2500", dtype = "uint32", shape = (2, 320))#candidate|2500|(2, 320)|var|uint32
call_2496 = func_2495_call(var_2497,var_2498,var_2499,var_2500,)
output = call_2496
func_2501 = relay.Function([var_2497,var_2498,var_2499,var_2500,], output)
mutated_mod['func_2501'] = func_2501
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2595 = relay.var("var_2595", dtype = "float64", shape = (11, 9, 3))#candidate|2595|(11, 9, 3)|var|float64
uop_2596 = relay.erf(var_2595.astype('float64')) # shape=(11, 9, 3)
uop_2598 = relay.acosh(uop_2596.astype('float64')) # shape=(11, 9, 3)
func_1303_call = mod.get_global_var('func_1303')
func_1306_call = mutated_mod.get_global_var('func_1306')
var_2606 = relay.var("var_2606", dtype = "float32", shape = (1, 80))#candidate|2606|(1, 80)|var|float32
const_2607 = relay.const([2.845799,-8.342673,-7.774508,-0.035070,-6.601806,2.057957,6.075450,6.931216,1.379610,-6.288245,-8.060101,2.640956,1.182240,-7.465227,0.616719,-0.040542,5.616741,-4.111592,6.729137,1.181286,8.844906,-6.095879,1.793908,6.486216,-0.410510,-6.035796,-1.124134,9.420429,-2.222732,-3.366909,4.599923,3.745887,6.784241,1.935419,-4.823010,-1.328634,9.031173,-5.467481,4.641760,6.328786,6.485434,-4.872934,5.100505,6.904654,-6.919835,1.744924,-5.115178,-0.588765,-8.830597,4.418824,-2.199558,-7.486620,-9.244299,-0.035265,-6.041468,3.534988,-5.090150,9.574738,8.797754,2.341987,-1.628491,6.409465,2.359159,-6.577956,9.486179,-0.339496,7.101108,1.623572,6.846160,8.639597,0.811430,1.340916,6.525878,1.135380,7.854857,-6.895960,5.465617,9.193351,1.183100,-5.537600,0.266366], dtype = "float64")#candidate|2607|(81,)|const|float64
call_2605 = relay.TupleGetItem(func_1303_call(relay.reshape(var_2606.astype('float32'), [5, 16]), relay.reshape(const_2607.astype('float64'), [81,]), ), 4)
call_2608 = relay.TupleGetItem(func_1306_call(relay.reshape(var_2606.astype('float32'), [5, 16]), relay.reshape(const_2607.astype('float64'), [81,]), ), 4)
func_440_call = mod.get_global_var('func_440')
func_444_call = mutated_mod.get_global_var('func_444')
const_2618 = relay.const([-6,-4,7,-1,-5,6,-10,3,-8,7,8,4,-9,8,-1,-5,-7,7,6,9,-7,-3,-1,-2,-7,7,-2,-7,-7,4,-8,-6,5,1,3,3,4,8,2,6,-8,5,4,-3,-1,-9,-6,-3,10,4,-3,5,3,-3,-3,6,-2,3,7,4,-5,-4,-10,1,1,5,-2,-1,2,-2,5,-10,1,2,1,3,-1,-4,3,2,3,-8,-6,2,5,6,-1,6,10,-7,5,7,-8,8,-8,-5,-10,6,4,-5,-1,8,7,-4,-5,8,-3,-6,5,2,8,6,2,3,1,4,3,6,9,7,-2,-2,9,8,5,9,9,1,-9,-9,6,-4,5,-7,1,8,8,-2,1,-2,2,-10,-9,4,7,10,-7,2,-4,8,-4,1,-8,10,-3,-6], dtype = "int32")#candidate|2618|(156,)|const|int32
var_2619 = relay.var("var_2619", dtype = "float32", shape = (9,))#candidate|2619|(9,)|var|float32
call_2617 = relay.TupleGetItem(func_440_call(relay.reshape(const_2618.astype('int32'), [13, 12]), relay.reshape(var_2619.astype('float32'), [9,]), ), 1)
call_2620 = relay.TupleGetItem(func_444_call(relay.reshape(const_2618.astype('int32'), [13, 12]), relay.reshape(var_2619.astype('float32'), [9,]), ), 1)
bop_2622 = relay.floor_mod(uop_2598.astype('float64'), relay.reshape(var_2595.astype('float64'), relay.shape_of(uop_2598))) # shape=(11, 9, 3)
output = relay.Tuple([call_2605,var_2606,const_2607,call_2617,const_2618,var_2619,bop_2622,])
output2 = relay.Tuple([call_2608,var_2606,const_2607,call_2620,const_2618,var_2619,bop_2622,])
F = relay.Function([var_2595,var_2606,var_2619,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_2595,var_2606,var_2619,], output2)
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
input_2595= np.array([[[6.140720,-1.914704,2.503924],[-1.991256,-8.320010,-4.801344],[0.947037,-1.693263,7.956305],[1.575304,-1.577209,-5.888902],[-4.692784,1.194598,9.303482],[-8.420000,4.793105,-9.191615],[-3.547273,3.143553,-2.764159],[-8.738549,6.130966,5.517440],[-6.573974,0.637482,5.194517]],[[2.990557,-3.911089,-7.982161],[-5.393473,-7.238471,-8.446558],[1.987335,8.659495,8.134913],[-7.612325,2.745374,-1.746930],[-0.446884,-1.354436,7.704041],[-2.490200,4.936204,6.854182],[-3.953315,4.282165,-1.118731],[-4.922870,-6.499259,-0.893838],[8.589452,4.650427,5.935854]],[[6.297964,-0.688750,5.825630],[1.799095,-0.914179,-8.788605],[-3.462945,-3.312779,1.047421],[7.412089,-5.823736,5.658585],[-8.429515,3.327037,-1.667545],[0.929100,-4.685097,9.250281],[-4.166933,9.137627,7.929107],[-4.897249,1.259469,2.534010],[3.613273,-7.287522,-8.187002]],[[6.581363,-3.976297,-8.719330],[7.263527,-6.853034,8.430477],[3.643710,-0.502666,3.115427],[2.247983,1.362611,-4.722896],[1.070049,-0.268542,4.939918],[-7.815749,7.779751,0.987623],[9.999068,3.256085,-5.984970],[1.011990,-0.614080,7.570224],[5.208407,-4.097372,3.790805]],[[-2.149413,-7.277910,-0.595357],[7.775783,-4.519132,7.185721],[1.518688,-6.292833,-3.106928],[0.007724,4.839791,9.769714],[3.111391,3.196890,5.716908],[-7.357622,4.783297,-3.759638],[-8.427867,-0.053743,-1.865744],[-0.282957,-8.115384,2.099787],[-8.002388,-4.353363,-8.558851]],[[-5.211536,2.227362,7.226652],[-4.258119,-7.207227,8.346742],[5.264164,-7.113560,-3.943375],[4.304594,-4.514535,-7.408342],[-8.253151,5.105269,-4.103405],[-2.791683,2.042016,-0.034747],[-1.994407,2.804481,-1.656723],[-1.689811,-5.786961,-9.322420],[8.673185,-1.788257,8.130362]],[[8.619638,9.529015,3.489606],[4.992682,-7.229999,2.154604],[9.474171,2.461621,-1.592258],[1.072438,7.084305,8.752602],[5.124197,6.001061,9.431486],[-4.540894,-9.634504,3.788313],[3.089189,2.452228,2.450913],[-8.727910,-6.412044,7.835885],[6.705536,9.319423,6.631026]],[[-8.234534,-0.611589,-1.611086],[0.683860,-1.548642,-2.413802],[0.053050,-6.264890,6.352105],[1.778543,5.796511,0.839831],[-2.191362,1.089760,1.906785],[-8.529061,9.113204,5.850017],[-0.232284,-9.238739,-1.087332],[-0.265580,-1.111409,-9.248371],[9.423376,9.521807,5.996589]],[[5.263620,-6.508638,-0.358662],[8.474113,3.691114,5.422757],[-6.076435,2.306640,-4.112884],[0.884706,3.944340,-5.971579],[0.986429,7.199006,-7.464151],[5.423862,-9.627216,-6.455310],[1.221978,-6.566069,2.352122],[9.235718,0.669086,-2.531781],[-0.162950,9.373475,-7.785588]],[[9.036448,4.619146,-0.605145],[-1.932855,8.853962,9.785871],[4.872801,-9.198388,-3.186002],[1.228871,7.730346,-2.445486],[-1.880798,-6.605711,8.738618],[0.450414,-6.227882,8.341655],[-5.389726,3.785316,0.008170],[-7.112270,2.774080,-9.268886],[-1.895942,-5.826981,-9.007173]],[[3.200830,5.250713,-3.324091],[-8.342292,-3.963356,-1.064839],[-0.322278,6.812723,2.481005],[-5.673380,6.226173,-2.721918],[8.958810,8.473388,-1.715512],[0.187530,-1.366365,4.828878],[6.273904,0.380174,3.497888],[4.002091,-5.918942,1.392202],[4.764568,7.811617,3.851741]]], dtype='float64')
module1.set_input('var_2595', input_2595)
input_2606= np.array([[-8.530195,-5.026731,-2.960211,-4.112189,4.955405,4.971064,6.541141,-0.464904,-0.620116,9.461618,7.138462,-9.877533,0.004600,1.066016,-1.824870,-2.554023,8.540376,-1.775077,3.919611,5.495576,1.484638,-2.721521,6.830426,9.178797,-4.708252,-0.941130,1.631006,-4.335068,-4.030054,8.356095,-9.958674,-7.379426,6.622086,-1.420839,9.356303,4.188060,-3.713285,3.437136,-6.879345,9.200975,0.270972,3.882220,-7.090929,2.345951,9.376624,1.054291,-4.194255,7.329182,-0.781759,2.189334,-4.503447,-3.437704,-0.191895,-2.970517,9.806094,6.243883,5.694548,7.815909,-6.056003,2.901128,6.981150,3.023507,-9.407252,9.959516,-0.221540,-9.564827,6.427138,6.919496,-1.815771,9.021833,7.862198,4.854414,1.713459,-5.475188,3.330301,5.688813,-9.916192,9.117184,5.869159,0.749254]], dtype='float32')
module1.set_input('var_2606', input_2606)
input_2619= np.array([2.409116,-4.440823,7.562095,-6.127488,-8.477477,1.234035,-6.230465,-7.000226,1.141352], dtype='float32')
module1.set_input('var_2619', input_2619)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_2595, input_2606, input_2619, )
res3 = intrp3.evaluate()(input_2595, input_2606, input_2619, )
res4 = intrp4.evaluate()(input_2595, input_2606, input_2619, )
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
res1_6 = module1.get_output(6).asnumpy()
res2_6 = res2[6].asnumpy()
res3_6 = res3[6].asnumpy()
res4_6 = res4[6].asnumpy()
np.testing.assert_allclose(res1_6 ,res2_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_6 ,res3_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_6 ,res4_6, atol=1e-3, rtol=1e-3)
(res1_6 == res2_6).all()
(res1_6 == res3_6).all()
(res1_6 == res4_6).all()
module5.set_input('var_2595', input_2595)
module5.set_input('var_2606', input_2606)
module5.set_input('var_2619', input_2619)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_2595, input_2606, input_2619, )
res7 = intrp7.evaluate()(input_2595, input_2606, input_2619, )
res8 = intrp8.evaluate()(input_2595, input_2606, input_2619, )
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
res5_6 = module5.get_output(6).asnumpy()
res6_6 = res6[6].asnumpy()
res7_6 = res7[6].asnumpy()
res8_6 = res8[6].asnumpy()
np.testing.assert_allclose(res5_6 ,res6_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_6 ,res7_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_6 ,res8_6, atol=1e-3, rtol=1e-3)
(res5_6 == res6_6).all()
(res5_6 == res7_6).all()
(res5_6 == res8_6).all()
module9.set_input('var_2595', input_2595)
module9.set_input('var_2606', input_2606)
module9.set_input('var_2619', input_2619)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_2595, input_2606, input_2619, )
res11 = intrp11.evaluate()(input_2595, input_2606, input_2619, )
res12 = intrp12.evaluate()(input_2595, input_2606, input_2619, )
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
res9_6 = module9.get_output(6).asnumpy()
res10_6 = res10[6].asnumpy()
res11_6 = res11[6].asnumpy()
res12_6 = res12[6].asnumpy()
np.testing.assert_allclose(res9_6 ,res10_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_6 ,res11_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_6 ,res12_6, atol=1e-3, rtol=1e-3)
(res9_6 == res10_6).all()
(res9_6 == res11_6).all()
(res9_6 == res12_6).all()
module13.set_input('var_2595', input_2595)
module13.set_input('var_2606', input_2606)
module13.set_input('var_2619', input_2619)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_2595, input_2606, input_2619, )
res15 = intrp15.evaluate()(input_2595, input_2606, input_2619, )
res16 = intrp16.evaluate()(input_2595, input_2606, input_2619, )
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
res13_6 = module13.get_output(6).asnumpy()
res14_6 = res14[6].asnumpy()
res15_6 = res15[6].asnumpy()
res16_6 = res16[6].asnumpy()
np.testing.assert_allclose(res13_6 ,res14_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_6 ,res15_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_6 ,res16_6, atol=1e-3, rtol=1e-3)
(res13_6 == res14_6).all()
(res13_6 == res15_6).all()
(res13_6 == res16_6).all()
module17.set_input('var_2595', input_2595)
module17.set_input('var_2606', input_2606)
module17.set_input('var_2619', input_2619)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_2595, input_2606, input_2619, )
res19 = intrp19.evaluate()(input_2595, input_2606, input_2619, )
res20 = intrp20.evaluate()(input_2595, input_2606, input_2619, )
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
res17_6 = module17.get_output(6).asnumpy()
res18_6 = res18[6].asnumpy()
res19_6 = res19[6].asnumpy()
res20_6 = res20[6].asnumpy()
np.testing.assert_allclose(res17_6 ,res18_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_6 ,res19_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_6 ,res20_6, atol=1e-3, rtol=1e-3)
(res17_6 == res18_6).all()
(res17_6 == res19_6).all()
(res17_6 == res20_6).all()
module21.set_input('var_2595', input_2595)
module21.set_input('var_2606', input_2606)
module21.set_input('var_2619', input_2619)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_2595, input_2606, input_2619, )
res23 = intrp23.evaluate()(input_2595, input_2606, input_2619, )
res24 = intrp24.evaluate()(input_2595, input_2606, input_2619, )
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
res21_6 = module21.get_output(6).asnumpy()
res22_6 = res22[6].asnumpy()
res23_6 = res23[6].asnumpy()
res24_6 = res24[6].asnumpy()
np.testing.assert_allclose(res21_6 ,res22_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_6 ,res23_6, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_6 ,res24_6, atol=1e-3, rtol=1e-3)
(res21_6 == res22_6).all()
(res21_6 == res23_6).all()
(res21_6 == res24_6).all()

'''42: TVMFuncCall
41: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
40: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
39: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
38: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
37: tvm::transform::Pass::operator()(tvm::IRModule) const
36: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
35: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
30: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
29: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
28: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
27: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
26: _ZN3tvm5relay16MixedModeMutato
25: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
22: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeMutator::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.535]
19: tvm::relay::MixedModeMutator::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::DynamicToStaticMutator::DispatchVisitExpr(tvm::RelayExpr const&)
17: _ZN3tvm5relay16MixedModeMutato
16: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
15: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
14: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
13: tvm::relay::MixedModeMutator::VisitExpr_(tvm::relay::CallNode const*)
12: tvm::relay::DynamicToStaticMutator::Rewrite_(tvm::relay::CallNode const*, tvm::RelayExpr const&)
11: std::_Function_handler<tvm::RelayExpr (tvm::relay::CallNode const*), tvm::relay::DynamicToStaticMutator::DynamicToStaticMutator(tvm::IRModule, tvm::relay::Function)::{lambda(tvm::relay::CallNode const*)#1}>::_M_invoke(std::_Any_data const&, tvm::relay::CallNode const*&&)
10: tvm::relay::DynamicToStaticMutator::PrepareArgs(tvm::relay::CallNode const*)
9: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
8: tvm::transform::Pass::operator()(tvm::IRModule) const
7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
6: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::Pass::operator()(tvm::IRModule) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}::operator()(tvm::IRModule, tvm::transform::PassContext const&) const [clone .isra.813]
0: tvm::DiagnosticContext::Render()

'''