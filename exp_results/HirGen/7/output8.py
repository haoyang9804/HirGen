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
const_25 = relay.const([[[5.408955,-9.107442,2.406967,3.663875,3.987650,-6.743532],[-9.947389,-2.199327,-6.146947,1.387405,-4.158320,-1.122570],[5.374751,7.244721,3.747718,2.199076,2.520398,-0.872151],[1.996634,2.582636,9.834806,-7.330638,5.956805,-5.500968],[0.082121,-4.149903,5.511800,-2.517542,-7.882953,-7.250900],[9.484294,-0.749115,-6.380986,6.378887,3.086203,-7.432997],[-0.362905,-9.000910,-7.141176,0.280073,-2.521605,1.453324],[8.957902,-5.495084,1.130506,1.930051,-4.657301,0.283521],[-4.829628,-6.715250,9.746859,-1.461070,-6.470275,-3.808924],[8.651047,-7.153672,8.770853,3.073163,3.642465,-5.160961],[9.456179,5.104612,-1.027010,-2.210398,6.310868,-1.169391],[9.784885,-8.719743,0.949260,-7.323422,9.333179,-7.227616],[-0.116465,3.766306,1.393162,-4.122021,2.367630,-2.245657],[-4.475972,0.831379,-1.749743,4.772114,-6.617181,4.306328]],[[-8.865157,3.960090,-3.687044,-4.859469,0.884556,-1.795972],[-2.694231,8.639508,5.066797,2.511422,-9.451414,8.740268],[-1.389443,3.186137,7.584597,-5.205138,-0.954544,0.863736],[9.983950,-5.706628,2.092905,9.759249,7.652694,8.985752],[-2.222238,4.231891,6.118004,6.615952,9.359690,4.830643],[6.332056,-4.148790,2.441324,-7.266395,0.871375,1.730115],[-4.638835,-5.304466,-1.438860,-8.203370,-5.387878,6.085492],[-5.056038,3.576784,6.744215,-0.886251,-3.265011,1.445672],[-5.221928,-3.131084,-8.690821,-0.195561,-7.955319,-5.108077],[0.773081,-3.931398,-6.344817,5.798495,-2.574728,-2.011227],[7.009801,6.890691,1.685612,3.399834,-8.641318,-2.729168],[-0.810088,3.375910,0.019881,5.787181,-9.350775,6.231210],[3.489772,-0.975165,0.204558,-8.326821,1.402513,0.506858],[-1.083519,7.027984,-9.556773,3.964938,-3.272543,-9.400585]]], dtype = "float32")#candidate|25|(2, 14, 6)|const|float32
var_26 = relay.var("var_26", dtype = "float32", shape = (2, 14, 6))#candidate|26|(2, 14, 6)|var|float32
bop_27 = relay.multiply(const_25.astype('float32'), relay.reshape(var_26.astype('float32'), relay.shape_of(const_25))) # shape=(2, 14, 6)
uop_35 = relay.atanh(const_25.astype('float64')) # shape=(2, 14, 6)
output = relay.Tuple([bop_27,uop_35,])
output2 = relay.Tuple([bop_27,uop_35,])
func_40 = relay.Function([var_26,], output)
mod['func_40'] = func_40
mod = relay.transform.InferType()(mod)
mutated_mod['func_40'] = func_40
mutated_mod = relay.transform.InferType()(mutated_mod)
var_41 = relay.var("var_41", dtype = "float32", shape = (2, 14, 6))#candidate|41|(2, 14, 6)|var|float32
func_40_call = mutated_mod.get_global_var('func_40')
call_42 = func_40_call(var_41)
output = call_42
func_43 = relay.Function([var_41], output)
mutated_mod['func_43'] = func_43
mutated_mod = relay.transform.InferType()(mutated_mod)
var_158 = relay.var("var_158", dtype = "float32", shape = (9, 2))#candidate|158|(9, 2)|var|float32
uop_159 = relay.tan(var_158.astype('float32')) # shape=(9, 2)
uop_166 = relay.sqrt(var_158.astype('float64')) # shape=(9, 2)
uop_168 = relay.sin(uop_159.astype('float32')) # shape=(9, 2)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
var_171 = relay.var("var_171", dtype = "float32", shape = (1, 168))#candidate|171|(1, 168)|var|float32
call_170 = relay.TupleGetItem(func_40_call(relay.reshape(var_171.astype('float32'), [2, 14, 6])), 0)
call_172 = relay.TupleGetItem(func_43_call(relay.reshape(var_171.astype('float32'), [2, 14, 6])), 0)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
call_173 = relay.TupleGetItem(func_40_call(relay.reshape(var_171.astype('float32'), [2, 14, 6])), 1)
call_174 = relay.TupleGetItem(func_43_call(relay.reshape(var_171.astype('float32'), [2, 14, 6])), 1)
uop_175 = relay.atanh(uop_168.astype('float64')) # shape=(9, 2)
var_180 = relay.var("var_180", dtype = "float64", shape = (9, 2))#candidate|180|(9, 2)|var|float64
bop_181 = relay.less_equal(uop_166.astype('bool'), relay.reshape(var_180.astype('bool'), relay.shape_of(uop_166))) # shape=(9, 2)
bop_186 = relay.greater_equal(uop_175.astype('bool'), relay.reshape(uop_168.astype('bool'), relay.shape_of(uop_175))) # shape=(9, 2)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
call_189 = relay.TupleGetItem(func_40_call(relay.reshape(call_173.astype('float32'), [2, 14, 6])), 1)
call_190 = relay.TupleGetItem(func_43_call(relay.reshape(call_173.astype('float32'), [2, 14, 6])), 1)
output = relay.Tuple([call_170,var_171,call_173,bop_181,bop_186,call_189,])
output2 = relay.Tuple([call_172,var_171,call_174,bop_181,bop_186,call_190,])
func_193 = relay.Function([var_158,var_171,var_180,], output)
mod['func_193'] = func_193
mod = relay.transform.InferType()(mod)
mutated_mod['func_193'] = func_193
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mutated_mod.get_global_var('func_193')
var_195 = relay.var("var_195", dtype = "float32", shape = (9, 2))#candidate|195|(9, 2)|var|float32
var_196 = relay.var("var_196", dtype = "float32", shape = (1, 168))#candidate|196|(1, 168)|var|float32
var_197 = relay.var("var_197", dtype = "float64", shape = (9, 2))#candidate|197|(9, 2)|var|float64
call_194 = func_193_call(var_195,var_196,var_197,)
output = call_194
func_198 = relay.Function([var_195,var_196,var_197,], output)
mutated_mod['func_198'] = func_198
mutated_mod = relay.transform.InferType()(mutated_mod)
var_265 = relay.var("var_265", dtype = "float64", shape = (7, 1, 7))#candidate|265|(7, 1, 7)|var|float64
uop_266 = relay.sinh(var_265.astype('float64')) # shape=(7, 1, 7)
bop_268 = relay.add(uop_266.astype('uint8'), relay.reshape(var_265.astype('uint8'), relay.shape_of(uop_266))) # shape=(7, 1, 7)
bop_271 = relay.bitwise_and(uop_266.astype('uint16'), relay.reshape(var_265.astype('uint16'), relay.shape_of(uop_266))) # shape=(7, 1, 7)
uop_275 = relay.cosh(bop_268.astype('float32')) # shape=(7, 1, 7)
uop_282 = relay.sigmoid(uop_275.astype('float32')) # shape=(7, 1, 7)
output = relay.Tuple([bop_271,uop_282,])
output2 = relay.Tuple([bop_271,uop_282,])
func_286 = relay.Function([var_265,], output)
mod['func_286'] = func_286
mod = relay.transform.InferType()(mod)
var_287 = relay.var("var_287", dtype = "float64", shape = (7, 1, 7))#candidate|287|(7, 1, 7)|var|float64
output = func_286(var_287)
func_288 = relay.Function([var_287], output)
mutated_mod['func_288'] = func_288
mutated_mod = relay.transform.InferType()(mutated_mod)
var_317 = relay.var("var_317", dtype = "uint32", shape = (7, 10))#candidate|317|(7, 10)|var|uint32
var_318 = relay.var("var_318", dtype = "uint32", shape = (7, 10))#candidate|318|(7, 10)|var|uint32
bop_319 = relay.equal(var_317.astype('bool'), relay.reshape(var_318.astype('bool'), relay.shape_of(var_317))) # shape=(7, 10)
var_324 = relay.var("var_324", dtype = "bool", shape = (7, 10))#candidate|324|(7, 10)|var|bool
bop_325 = relay.floor_mod(bop_319.astype('float32'), relay.reshape(var_324.astype('float32'), relay.shape_of(bop_319))) # shape=(7, 10)
bop_331 = relay.add(bop_319.astype('uint16'), relay.reshape(var_324.astype('uint16'), relay.shape_of(bop_319))) # shape=(7, 10)
uop_334 = relay.log10(bop_319.astype('float32')) # shape=(7, 10)
bop_336 = relay.subtract(var_324.astype('uint32'), relay.reshape(uop_334.astype('uint32'), relay.shape_of(var_324))) # shape=(7, 10)
output = relay.Tuple([bop_325,bop_331,bop_336,])
output2 = relay.Tuple([bop_325,bop_331,bop_336,])
func_340 = relay.Function([var_317,var_318,var_324,], output)
mod['func_340'] = func_340
mod = relay.transform.InferType()(mod)
var_341 = relay.var("var_341", dtype = "uint32", shape = (7, 10))#candidate|341|(7, 10)|var|uint32
var_342 = relay.var("var_342", dtype = "uint32", shape = (7, 10))#candidate|342|(7, 10)|var|uint32
var_343 = relay.var("var_343", dtype = "bool", shape = (7, 10))#candidate|343|(7, 10)|var|bool
output = func_340(var_341,var_342,var_343,)
func_344 = relay.Function([var_341,var_342,var_343,], output)
mutated_mod['func_344'] = func_344
mutated_mod = relay.transform.InferType()(mutated_mod)
var_388 = relay.var("var_388", dtype = "float64", shape = (6, 15))#candidate|388|(6, 15)|var|float64
var_389 = relay.var("var_389", dtype = "float64", shape = (6, 15))#candidate|389|(6, 15)|var|float64
bop_390 = relay.divide(var_388.astype('float64'), relay.reshape(var_389.astype('float64'), relay.shape_of(var_388))) # shape=(6, 15)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
const_395 = relay.const([[4.543362,-3.955138,5.980570,8.126715,-8.544652,-5.620794,3.609298],[6.009050,6.170698,0.360934,-1.427173,-7.899098,-8.753938,3.113082],[2.651902,4.345688,3.833874,-1.807704,1.817275,-8.936757,9.171604],[-8.509492,5.927390,-3.459403,4.046106,-4.089914,-5.395987,8.272767],[9.593358,0.740153,-3.996767,3.025292,2.460794,4.566802,6.017801],[-7.051960,-3.312430,5.071156,-6.792986,0.081366,6.093962,0.482305],[-3.673444,9.461525,5.399005,-7.699524,9.438415,-2.215389,3.778323]], dtype = "float64")#candidate|395|(7, 7)|const|float64
call_394 = relay.TupleGetItem(func_286_call(relay.reshape(const_395.astype('float64'), [7, 1, 7])), 0)
call_396 = relay.TupleGetItem(func_288_call(relay.reshape(const_395.astype('float64'), [7, 1, 7])), 0)
output = relay.Tuple([bop_390,call_394,const_395,])
output2 = relay.Tuple([bop_390,call_396,const_395,])
func_401 = relay.Function([var_388,var_389,], output)
mod['func_401'] = func_401
mod = relay.transform.InferType()(mod)
var_402 = relay.var("var_402", dtype = "float64", shape = (6, 15))#candidate|402|(6, 15)|var|float64
var_403 = relay.var("var_403", dtype = "float64", shape = (6, 15))#candidate|403|(6, 15)|var|float64
output = func_401(var_402,var_403,)
func_404 = relay.Function([var_402,var_403,], output)
mutated_mod['func_404'] = func_404
mutated_mod = relay.transform.InferType()(mutated_mod)
var_424 = relay.var("var_424", dtype = "int32", shape = (14, 2, 3))#candidate|424|(14, 2, 3)|var|int32
const_425 = relay.const([[[-5,6,5],[-4,-5,9]],[[-5,-9,-3],[1,1,2]],[[10,-4,-1],[4,-3,3]],[[-6,-6,-9],[1,10,-3]],[[4,10,2],[7,9,-2]],[[-2,8,9],[-5,10,5]],[[6,5,7],[10,-4,7]],[[2,7,5],[6,8,-5]],[[7,1,3],[7,6,-7]],[[4,6,-10],[1,7,5]],[[-9,-9,5],[5,6,-7]],[[4,8,10],[2,-7,-9]],[[1,10,7],[-2,-4,-5]],[[-7,-9,6],[1,4,9]]], dtype = "int32")#candidate|425|(14, 2, 3)|const|int32
bop_426 = relay.less(var_424.astype('bool'), relay.reshape(const_425.astype('bool'), relay.shape_of(var_424))) # shape=(14, 2, 3)
bop_429 = relay.add(const_425.astype('uint32'), relay.reshape(bop_426.astype('uint32'), relay.shape_of(const_425))) # shape=(14, 2, 3)
uop_432 = relay.log2(bop_426.astype('float64')) # shape=(14, 2, 3)
bop_434 = relay.bitwise_xor(uop_432.astype('uint8'), relay.reshape(const_425.astype('uint8'), relay.shape_of(uop_432))) # shape=(14, 2, 3)
var_437 = relay.var("var_437", dtype = "float64", shape = (14, 2, 3))#candidate|437|(14, 2, 3)|var|float64
bop_438 = relay.logical_xor(uop_432.astype('uint16'), relay.reshape(var_437.astype('uint16'), relay.shape_of(uop_432))) # shape=(14, 2, 3)
uop_443 = relay.cosh(uop_432.astype('float64')) # shape=(14, 2, 3)
bop_451 = relay.power(bop_438.astype('float64'), relay.reshape(uop_443.astype('float64'), relay.shape_of(bop_438))) # shape=(14, 2, 3)
bop_457 = relay.minimum(bop_451.astype('int64'), relay.reshape(var_424.astype('int64'), relay.shape_of(bop_451))) # shape=(14, 2, 3)
uop_460 = relay.asinh(uop_443.astype('float32')) # shape=(14, 2, 3)
uop_462 = relay.cosh(uop_460.astype('float64')) # shape=(14, 2, 3)
bop_469 = relay.divide(uop_462.astype('float64'), relay.reshape(bop_434.astype('float64'), relay.shape_of(uop_462))) # shape=(14, 2, 3)
bop_478 = relay.floor_mod(bop_457.astype('float32'), relay.reshape(bop_426.astype('float32'), relay.shape_of(bop_457))) # shape=(14, 2, 3)
uop_481 = relay.sqrt(bop_469.astype('float64')) # shape=(14, 2, 3)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
const_487 = relay.const([4.284754,6.603751,4.404153,-4.209276,-9.409251,9.938605,8.210061,-9.576535,7.056087,7.629968,-9.918206,-3.722909,-2.115090,-7.559971,-0.116726,-3.903979,-8.851424,-3.318396,2.615981,-0.986222,3.182905,2.669378,4.490106,7.449172,-5.139049,6.812941,7.994583,-6.871081,9.437836,-9.340186,-5.628003,8.109754,-5.851782,8.261696,-8.838565,1.263670,6.352238,-2.970712,-1.869496,9.432535,1.434110,-1.574893,5.165804,-4.019132,0.129770,-7.343338,-0.446461,-8.756798,8.214382], dtype = "float64")#candidate|487|(49,)|const|float64
call_486 = relay.TupleGetItem(func_286_call(relay.reshape(const_487.astype('float64'), [7, 1, 7])), 0)
call_488 = relay.TupleGetItem(func_288_call(relay.reshape(const_487.astype('float64'), [7, 1, 7])), 0)
uop_490 = relay.sinh(uop_481.astype('float32')) # shape=(14, 2, 3)
bop_493 = relay.floor_divide(uop_481.astype('float32'), relay.reshape(var_424.astype('float32'), relay.shape_of(uop_481))) # shape=(14, 2, 3)
bop_498 = relay.mod(uop_462.astype('float32'), relay.reshape(bop_434.astype('float32'), relay.shape_of(uop_462))) # shape=(14, 2, 3)
uop_502 = relay.erf(bop_438.astype('float32')) # shape=(14, 2, 3)
uop_505 = relay.sigmoid(uop_460.astype('float64')) # shape=(14, 2, 3)
bop_516 = relay.maximum(uop_460.astype('uint32'), relay.reshape(bop_478.astype('uint32'), relay.shape_of(uop_460))) # shape=(14, 2, 3)
var_520 = relay.var("var_520", dtype = "float32", shape = (14, 2, 3))#candidate|520|(14, 2, 3)|var|float32
bop_521 = relay.left_shift(uop_490.astype('int64'), relay.reshape(var_520.astype('int64'), relay.shape_of(uop_490))) # shape=(14, 2, 3)
func_401_call = mod.get_global_var('func_401')
func_404_call = mutated_mod.get_global_var('func_404')
const_525 = relay.const([-7.052733,7.979641,8.743803,-7.531825,-5.950127,-0.924758,-4.545637,-6.921858,-8.578316,-0.623782,-5.482080,7.527645,5.702852,9.145837,-7.896894,4.263111,1.280454,-0.649784,6.678607,-7.630780,-4.191432,-2.763164,-4.228990,9.715619,7.947614,-3.772094,5.186794,2.854609,-6.435772,2.415618,2.862185,6.298243,7.773845,5.295932,-0.752919,-0.450261,2.244407,4.243795,6.599235,-1.108739,-0.761435,-0.929319,4.010910,-2.880828,-4.314207,-2.468122,9.086341,8.501813,0.794288,8.127839,-6.678943,-2.812729,-8.676211,9.622317,4.948737,-3.895915,7.944646,3.079343,-6.763975,-7.857212,3.416791,-9.795272,-1.390824,-7.192093,8.204331,1.246775,0.033474,6.589709,5.026767,-8.311231,-4.309384,9.645229,8.013252,-2.225797,-0.669665,6.972308,-3.001577,-0.407821,4.745857,-3.809426,-2.980079,-1.456754,-4.907209,7.652243,4.229045,-0.298274,-2.423371,8.036667,-6.982606,-7.688341], dtype = "float64")#candidate|525|(90,)|const|float64
call_524 = relay.TupleGetItem(func_401_call(relay.reshape(const_525.astype('float64'), [6, 15]), relay.reshape(const_525.astype('float64'), [6, 15]), ), 0)
call_526 = relay.TupleGetItem(func_404_call(relay.reshape(const_525.astype('float64'), [6, 15]), relay.reshape(const_525.astype('float64'), [6, 15]), ), 0)
bop_536 = relay.bitwise_xor(uop_481.astype('uint16'), relay.reshape(uop_432.astype('uint16'), relay.shape_of(uop_481))) # shape=(14, 2, 3)
func_193_call = mod.get_global_var('func_193')
func_198_call = mutated_mod.get_global_var('func_198')
const_540 = relay.const([-3.260634,-6.415126,-5.864479,9.968737,0.206181,-2.961945,4.367791,5.573571,-2.974613,-8.807823,5.307766,-8.527930,5.442720,9.438294,4.131859,1.554719,-0.307951,0.775100], dtype = "float32")#candidate|540|(18,)|const|float32
var_541 = relay.var("var_541", dtype = "float32", shape = (168,))#candidate|541|(168,)|var|float32
call_539 = relay.TupleGetItem(func_193_call(relay.reshape(const_540.astype('float32'), [9, 2]), relay.reshape(var_541.astype('float32'), [1, 168]), relay.reshape(const_540.astype('float64'), [9, 2]), ), 5)
call_542 = relay.TupleGetItem(func_198_call(relay.reshape(const_540.astype('float32'), [9, 2]), relay.reshape(var_541.astype('float32'), [1, 168]), relay.reshape(const_540.astype('float64'), [9, 2]), ), 5)
func_340_call = mod.get_global_var('func_340')
func_344_call = mutated_mod.get_global_var('func_344')
const_544 = relay.const([7,-3,-9,-7,-5,7,-2,-2,7,-4,3,-6,7,2,8,4,-7,2,-6,-10,-5,-8,1,3,-5,10,1,7,-6,2,10,-9,-3,-2,10,-9,-8,-10,-10,7,-10,-1,5,-6,7,-2,6,-4,2,7,8,7,-4,3,1,-7,-6,-9,6,-9,7,-10,5,-9,9,-1,3,-1,4,-1], dtype = "uint32")#candidate|544|(70,)|const|uint32
call_543 = relay.TupleGetItem(func_340_call(relay.reshape(const_544.astype('uint32'), [7, 10]), relay.reshape(const_544.astype('uint32'), [7, 10]), relay.reshape(const_544.astype('bool'), [7, 10]), ), 0)
call_545 = relay.TupleGetItem(func_344_call(relay.reshape(const_544.astype('uint32'), [7, 10]), relay.reshape(const_544.astype('uint32'), [7, 10]), relay.reshape(const_544.astype('bool'), [7, 10]), ), 0)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
call_551 = relay.TupleGetItem(func_40_call(relay.reshape(call_539.astype('float32'), [2, 14, 6])), 1)
call_552 = relay.TupleGetItem(func_43_call(relay.reshape(call_539.astype('float32'), [2, 14, 6])), 1)
bop_553 = relay.logical_and(bop_493.astype('bool'), relay.reshape(bop_426.astype('bool'), relay.shape_of(bop_493))) # shape=(14, 2, 3)
bop_556 = relay.logical_or(uop_490.astype('bool'), relay.reshape(bop_521.astype('bool'), relay.shape_of(uop_490))) # shape=(14, 2, 3)
func_401_call = mod.get_global_var('func_401')
func_404_call = mutated_mod.get_global_var('func_404')
call_559 = relay.TupleGetItem(func_401_call(relay.reshape(call_524.astype('float64'), [6, 15]), relay.reshape(call_524.astype('float64'), [6, 15]), ), 0)
call_560 = relay.TupleGetItem(func_404_call(relay.reshape(call_524.astype('float64'), [6, 15]), relay.reshape(call_524.astype('float64'), [6, 15]), ), 0)
bop_564 = relay.multiply(bop_493.astype('uint16'), relay.reshape(bop_516.astype('uint16'), relay.shape_of(bop_493))) # shape=(14, 2, 3)
output = relay.Tuple([bop_429,call_486,const_487,bop_498,uop_502,uop_505,call_524,const_525,bop_536,call_539,const_540,var_541,call_543,const_544,call_551,bop_553,bop_556,call_559,bop_564,])
output2 = relay.Tuple([bop_429,call_488,const_487,bop_498,uop_502,uop_505,call_526,const_525,bop_536,call_542,const_540,var_541,call_545,const_544,call_552,bop_553,bop_556,call_560,bop_564,])
func_567 = relay.Function([var_424,var_437,var_520,var_541,], output)
mod['func_567'] = func_567
mod = relay.transform.InferType()(mod)
mutated_mod['func_567'] = func_567
mutated_mod = relay.transform.InferType()(mutated_mod)
func_567_call = mutated_mod.get_global_var('func_567')
var_569 = relay.var("var_569", dtype = "int32", shape = (14, 2, 3))#candidate|569|(14, 2, 3)|var|int32
var_570 = relay.var("var_570", dtype = "float64", shape = (14, 2, 3))#candidate|570|(14, 2, 3)|var|float64
var_571 = relay.var("var_571", dtype = "float32", shape = (14, 2, 3))#candidate|571|(14, 2, 3)|var|float32
var_572 = relay.var("var_572", dtype = "float32", shape = (168,))#candidate|572|(168,)|var|float32
call_568 = func_567_call(var_569,var_570,var_571,var_572,)
output = call_568
func_573 = relay.Function([var_569,var_570,var_571,var_572,], output)
mutated_mod['func_573'] = func_573
mutated_mod = relay.transform.InferType()(mutated_mod)
var_603 = relay.var("var_603", dtype = "int16", shape = (8, 7))#candidate|603|(8, 7)|var|int16
const_604 = relay.const([[3,3,-10,6,-6,1,-9],[-1,-2,9,6,6,-7,-9],[5,6,-3,9,-6,1,1],[6,-8,-7,10,2,9,1],[7,9,-8,-7,8,6,-10],[-7,-8,-2,-2,1,-4,-7],[-10,-3,10,6,-8,1,-10],[9,-4,10,-1,-10,-1,4]], dtype = "int16")#candidate|604|(8, 7)|const|int16
bop_605 = relay.left_shift(var_603.astype('int16'), relay.reshape(const_604.astype('int16'), relay.shape_of(var_603))) # shape=(8, 7)
bop_612 = relay.mod(bop_605.astype('float32'), relay.reshape(const_604.astype('float32'), relay.shape_of(bop_605))) # shape=(8, 7)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
const_622 = relay.const([[7.659241,6.364092,1.023226,2.895773,1.186291,-8.051233,-3.067931],[-4.537521,-6.218904,6.171913,-6.478027,4.945992,8.797275,0.060081],[7.990149,7.450160,-4.294279,2.832277,-9.826032,-7.151711,-3.274526],[-8.377057,-1.740273,8.071037,-4.803178,3.927412,0.875923,0.215886],[1.259375,-8.404424,-4.103043,9.954628,2.581091,-1.121068,-1.109548],[7.717682,-4.886925,1.091300,-6.586908,7.670852,4.131801,-9.154031],[2.236273,-2.298405,-2.081686,8.044672,-9.301259,-2.688316,6.460313]], dtype = "float64")#candidate|622|(7, 7)|const|float64
call_621 = relay.TupleGetItem(func_286_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 1)
call_623 = relay.TupleGetItem(func_288_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 1)
bop_626 = relay.logical_or(bop_612.astype('bool'), relay.reshape(var_603.astype('bool'), relay.shape_of(bop_612))) # shape=(8, 7)
bop_629 = relay.add(bop_626.astype('uint8'), relay.reshape(bop_612.astype('uint8'), relay.shape_of(bop_626))) # shape=(8, 7)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
call_634 = relay.TupleGetItem(func_286_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 0)
call_635 = relay.TupleGetItem(func_288_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 0)
uop_636 = relay.tan(bop_612.astype('float32')) # shape=(8, 7)
uop_640 = relay.tan(uop_636.astype('float32')) # shape=(8, 7)
uop_642 = relay.log2(uop_636.astype('float64')) # shape=(8, 7)
func_193_call = mod.get_global_var('func_193')
func_198_call = mutated_mod.get_global_var('func_198')
const_646 = relay.const([[-8.590264],[-9.813693],[-1.331991],[2.816918],[2.259056],[-1.286757],[4.595732],[-3.568803],[2.568815],[0.596116],[4.725362],[4.090946],[5.416879],[9.407908],[1.522986],[-4.134352],[-4.616443],[6.250783]], dtype = "float32")#candidate|646|(18, 1)|const|float32
var_647 = relay.var("var_647", dtype = "float32", shape = (168,))#candidate|647|(168,)|var|float32
call_645 = relay.TupleGetItem(func_193_call(relay.reshape(const_646.astype('float32'), [9, 2]), relay.reshape(var_647.astype('float32'), [1, 168]), relay.reshape(const_646.astype('float64'), [9, 2]), ), 4)
call_648 = relay.TupleGetItem(func_198_call(relay.reshape(const_646.astype('float32'), [9, 2]), relay.reshape(var_647.astype('float32'), [1, 168]), relay.reshape(const_646.astype('float64'), [9, 2]), ), 4)
var_649 = relay.var("var_649", dtype = "float32", shape = (8, 7))#candidate|649|(8, 7)|var|float32
bop_650 = relay.not_equal(uop_636.astype('bool'), relay.reshape(var_649.astype('bool'), relay.shape_of(uop_636))) # shape=(8, 7)
bop_656 = relay.divide(uop_642.astype('float32'), relay.reshape(var_603.astype('float32'), relay.shape_of(uop_642))) # shape=(8, 7)
func_193_call = mod.get_global_var('func_193')
func_198_call = mutated_mod.get_global_var('func_198')
call_662 = relay.TupleGetItem(func_193_call(relay.reshape(const_646.astype('float32'), [9, 2]), relay.reshape(var_647.astype('float32'), [1, 168]), relay.reshape(const_646.astype('float64'), [9, 2]), ), 0)
call_663 = relay.TupleGetItem(func_198_call(relay.reshape(const_646.astype('float32'), [9, 2]), relay.reshape(var_647.astype('float32'), [1, 168]), relay.reshape(const_646.astype('float64'), [9, 2]), ), 0)
uop_664 = relay.log10(bop_650.astype('float32')) # shape=(8, 7)
bop_669 = relay.less_equal(uop_642.astype('bool'), call_621.astype('bool')) # shape=(7, 8, 7)
bop_672 = relay.less_equal(uop_642.astype('bool'), call_623.astype('bool')) # shape=(7, 8, 7)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
call_673 = relay.TupleGetItem(func_40_call(relay.reshape(call_662.astype('float32'), [2, 14, 6])), 0)
call_674 = relay.TupleGetItem(func_43_call(relay.reshape(call_662.astype('float32'), [2, 14, 6])), 0)
func_40_call = mod.get_global_var('func_40')
func_43_call = mutated_mod.get_global_var('func_43')
call_677 = relay.TupleGetItem(func_40_call(relay.reshape(var_647.astype('float32'), [2, 14, 6])), 1)
call_678 = relay.TupleGetItem(func_43_call(relay.reshape(var_647.astype('float32'), [2, 14, 6])), 1)
uop_679 = relay.atanh(uop_664.astype('float32')) # shape=(8, 7)
uop_683 = relay.acosh(bop_650.astype('float32')) # shape=(8, 7)
bop_687 = relay.not_equal(uop_640.astype('bool'), call_634.astype('bool')) # shape=(7, 8, 7)
bop_690 = relay.not_equal(uop_640.astype('bool'), call_635.astype('bool')) # shape=(7, 8, 7)
var_696 = relay.var("var_696", dtype = "float32", shape = (8, 7))#candidate|696|(8, 7)|var|float32
bop_697 = relay.greater(uop_679.astype('bool'), relay.reshape(var_696.astype('bool'), relay.shape_of(uop_679))) # shape=(8, 7)
bop_701 = relay.bitwise_or(uop_640.astype('int16'), relay.reshape(uop_642.astype('int16'), relay.shape_of(uop_640))) # shape=(8, 7)
bop_704 = relay.greater_equal(bop_697.astype('bool'), relay.reshape(uop_640.astype('bool'), relay.shape_of(bop_697))) # shape=(8, 7)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
call_715 = relay.TupleGetItem(func_286_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 0)
call_716 = relay.TupleGetItem(func_288_call(relay.reshape(const_622.astype('float64'), [7, 1, 7])), 0)
output = relay.Tuple([const_622,bop_629,call_645,const_646,var_647,bop_656,call_662,bop_669,call_673,call_677,uop_683,bop_687,bop_701,bop_704,call_715,])
output2 = relay.Tuple([const_622,bop_629,call_648,const_646,var_647,bop_656,call_663,bop_672,call_674,call_678,uop_683,bop_690,bop_701,bop_704,call_716,])
func_720 = relay.Function([var_603,var_647,var_649,var_696,], output)
mod['func_720'] = func_720
mod = relay.transform.InferType()(mod)
mutated_mod['func_720'] = func_720
mutated_mod = relay.transform.InferType()(mutated_mod)
func_720_call = mutated_mod.get_global_var('func_720')
var_722 = relay.var("var_722", dtype = "int16", shape = (8, 7))#candidate|722|(8, 7)|var|int16
var_723 = relay.var("var_723", dtype = "float32", shape = (168,))#candidate|723|(168,)|var|float32
var_724 = relay.var("var_724", dtype = "float32", shape = (8, 7))#candidate|724|(8, 7)|var|float32
var_725 = relay.var("var_725", dtype = "float32", shape = (8, 7))#candidate|725|(8, 7)|var|float32
call_721 = func_720_call(var_722,var_723,var_724,var_725,)
output = call_721
func_726 = relay.Function([var_722,var_723,var_724,var_725,], output)
mutated_mod['func_726'] = func_726
mutated_mod = relay.transform.InferType()(mutated_mod)
var_734 = relay.var("var_734", dtype = "int64", shape = ())#candidate|734|()|var|int64
var_735 = relay.var("var_735", dtype = "int64", shape = (7, 16))#candidate|735|(7, 16)|var|int64
bop_736 = relay.not_equal(var_734.astype('bool'), var_735.astype('bool')) # shape=(7, 16)
uop_751 = relay.sigmoid(var_735.astype('float32')) # shape=(7, 16)
bop_756 = relay.add(uop_751.astype('int32'), var_734.astype('int32')) # shape=(7, 16)
output = relay.Tuple([bop_736,bop_756,])
output2 = relay.Tuple([bop_736,bop_756,])
func_760 = relay.Function([var_734,var_735,], output)
mod['func_760'] = func_760
mod = relay.transform.InferType()(mod)
mutated_mod['func_760'] = func_760
mutated_mod = relay.transform.InferType()(mutated_mod)
func_760_call = mutated_mod.get_global_var('func_760')
var_762 = relay.var("var_762", dtype = "int64", shape = ())#candidate|762|()|var|int64
var_763 = relay.var("var_763", dtype = "int64", shape = (7, 16))#candidate|763|(7, 16)|var|int64
call_761 = func_760_call(var_762,var_763,)
output = call_761
func_764 = relay.Function([var_762,var_763,], output)
mutated_mod['func_764'] = func_764
mutated_mod = relay.transform.InferType()(mutated_mod)
var_806 = relay.var("var_806", dtype = "int32", shape = (16, 6))#candidate|806|(16, 6)|var|int32
var_807 = relay.var("var_807", dtype = "int32", shape = (16, 6))#candidate|807|(16, 6)|var|int32
bop_808 = relay.equal(var_806.astype('bool'), relay.reshape(var_807.astype('bool'), relay.shape_of(var_806))) # shape=(16, 6)
var_813 = relay.var("var_813", dtype = "bool", shape = (16, 6))#candidate|813|(16, 6)|var|bool
bop_814 = relay.bitwise_and(bop_808.astype('int8'), relay.reshape(var_813.astype('int8'), relay.shape_of(bop_808))) # shape=(16, 6)
uop_817 = relay.sin(bop_814.astype('float64')) # shape=(16, 6)
bop_823 = relay.less_equal(uop_817.astype('bool'), relay.reshape(bop_808.astype('bool'), relay.shape_of(uop_817))) # shape=(16, 6)
var_828 = relay.var("var_828", dtype = "bool", shape = (16, 6))#candidate|828|(16, 6)|var|bool
bop_829 = relay.multiply(bop_823.astype('int16'), relay.reshape(var_828.astype('int16'), relay.shape_of(bop_823))) # shape=(16, 6)
bop_832 = relay.logical_and(bop_823.astype('bool'), relay.reshape(var_806.astype('bool'), relay.shape_of(bop_823))) # shape=(16, 6)
uop_837 = relay.acosh(bop_823.astype('float32')) # shape=(16, 6)
uop_842 = relay.cosh(uop_837.astype('float64')) # shape=(16, 6)
uop_844 = relay.atanh(uop_842.astype('float64')) # shape=(16, 6)
bop_846 = relay.divide(uop_842.astype('float32'), relay.reshape(bop_832.astype('float32'), relay.shape_of(uop_842))) # shape=(16, 6)
bop_855 = relay.bitwise_or(uop_837.astype('int64'), relay.reshape(bop_846.astype('int64'), relay.shape_of(uop_837))) # shape=(16, 6)
uop_858 = relay.asin(uop_837.astype('float64')) # shape=(16, 6)
func_760_call = mod.get_global_var('func_760')
func_764_call = mutated_mod.get_global_var('func_764')
const_861 = relay.const(-1, dtype = "int64")#candidate|861|()|const|int64
const_862 = relay.const([[-8,-1,-7,-5,-7,-4,1,8,9,-9,-5,-7,4,3,-3,-1,10,-6,-8,6,7,-4,-2,-10,4,3,-4,10,-10,9,-6,6,4,-4,2,10,2,8,8,-2,7,-2,8,10,5,-1,10,8,1,8,6,3,-6,9,-8,4,8,6,-8,-6,-8,3,-6,3,6,9,3,-7,2,1,-9,-5,1,-7,-2,1,9,7,6,3,3,-4,-5,-3,1,8,1,10,-9,-2,-7,-10,-8,4,-6,7,-8,-6,-6,-9,-7,8,2,-9,-3,9,-5,-10,5,6,-8,3]], dtype = "int64")#candidate|862|(1, 112)|const|int64
call_860 = relay.TupleGetItem(func_760_call(relay.reshape(const_861.astype('int64'), []), relay.reshape(const_862.astype('int64'), [7, 16]), ), 0)
call_863 = relay.TupleGetItem(func_764_call(relay.reshape(const_861.astype('int64'), []), relay.reshape(const_862.astype('int64'), [7, 16]), ), 0)
bop_864 = relay.power(uop_844.astype('float32'), relay.reshape(uop_842.astype('float32'), relay.shape_of(uop_844))) # shape=(16, 6)
uop_869 = relay.asinh(var_813.astype('float64')) # shape=(16, 6)
bop_871 = relay.bitwise_xor(uop_837.astype('uint64'), relay.reshape(bop_823.astype('uint64'), relay.shape_of(uop_837))) # shape=(16, 6)
bop_874 = relay.minimum(bop_855.astype('uint16'), relay.reshape(uop_869.astype('uint16'), relay.shape_of(bop_855))) # shape=(16, 6)
uop_877 = relay.sigmoid(bop_846.astype('float32')) # shape=(16, 6)
const_879 = relay.const([[-7.916525,4.619570,0.576495,-9.237258,-0.463289,7.475677],[0.986457,7.510678,-3.157362,-4.759654,-7.509747,-9.678066],[6.520569,-3.268504,-0.852082,5.359195,-9.595792,-8.953148],[2.368487,2.970499,-7.720259,-3.374212,-5.672640,8.215958],[-0.954047,-9.157181,6.018716,2.310409,-0.931508,-6.057022],[5.447511,4.924329,-9.424051,5.331023,-3.455040,5.129637],[-9.506890,7.219065,-7.497648,8.206196,-8.772381,0.204438],[-6.851829,-5.249992,8.486533,6.152046,-6.045589,-8.870906],[-8.056083,3.248604,4.459966,-3.198654,-8.131515,-5.397851],[-9.028178,3.658240,9.323375,-0.851671,-1.889075,-2.646544],[2.244718,-6.336232,-6.052243,1.528666,4.728972,-8.417426],[0.757741,-5.771829,-6.042246,-8.461679,-8.077948,9.921111],[4.053943,4.858517,-0.909959,6.352351,-8.094231,6.754386],[-2.858719,9.005634,8.357052,-7.357249,-7.728971,5.100100],[-9.252350,-9.653428,4.108852,-5.488615,1.643886,-9.840678],[-2.150287,-1.905882,-9.789895,9.349277,-1.351708,-6.156943]], dtype = "float32")#candidate|879|(16, 6)|const|float32
bop_880 = relay.less(bop_864.astype('bool'), relay.reshape(const_879.astype('bool'), relay.shape_of(bop_864))) # shape=(16, 6)
uop_892 = relay.sinh(bop_864.astype('float64')) # shape=(16, 6)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
const_897 = relay.const([[-9.382826],[-4.271229],[4.465975],[8.420307],[7.767411],[-0.260397],[8.678411],[9.095719],[3.710339],[5.876766],[-3.135905],[2.773496],[9.632201],[8.607686],[0.488808],[2.389895],[3.275744],[-2.827212],[6.023175],[-3.247857],[-7.700586],[4.675764],[-6.688939],[1.825015],[-9.415312],[1.683011],[-5.502298],[-5.004734],[0.482569],[3.919956],[-9.320493],[-1.169593],[6.897763],[-5.851881],[-8.653667],[9.172689],[-9.130768],[-1.414712],[1.041924],[-9.436099],[-7.283029],[-6.104343],[-0.196113],[8.605686],[-4.753291],[6.941176],[5.110475],[-0.758814],[-4.715774]], dtype = "float64")#candidate|897|(49, 1)|const|float64
call_896 = relay.TupleGetItem(func_286_call(relay.reshape(const_897.astype('float64'), [7, 1, 7])), 1)
call_898 = relay.TupleGetItem(func_288_call(relay.reshape(const_897.astype('float64'), [7, 1, 7])), 1)
uop_900 = relay.tan(uop_844.astype('float64')) # shape=(16, 6)
bop_902 = relay.less_equal(uop_837.astype('bool'), relay.reshape(bop_855.astype('bool'), relay.shape_of(uop_837))) # shape=(16, 6)
uop_906 = relay.log(uop_892.astype('float32')) # shape=(16, 6)
bop_908 = relay.mod(uop_906.astype('float64'), relay.reshape(bop_829.astype('float64'), relay.shape_of(uop_906))) # shape=(16, 6)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
call_914 = relay.TupleGetItem(func_286_call(relay.reshape(const_897.astype('float64'), [7, 1, 7])), 1)
call_915 = relay.TupleGetItem(func_288_call(relay.reshape(const_897.astype('float64'), [7, 1, 7])), 1)
output = relay.Tuple([uop_858,call_860,const_861,const_862,bop_871,bop_874,uop_877,bop_880,call_896,const_897,uop_900,bop_902,bop_908,call_914,])
output2 = relay.Tuple([uop_858,call_863,const_861,const_862,bop_871,bop_874,uop_877,bop_880,call_898,const_897,uop_900,bop_902,bop_908,call_915,])
func_916 = relay.Function([var_806,var_807,var_813,var_828,], output)
mod['func_916'] = func_916
mod = relay.transform.InferType()(mod)
var_917 = relay.var("var_917", dtype = "int32", shape = (16, 6))#candidate|917|(16, 6)|var|int32
var_918 = relay.var("var_918", dtype = "int32", shape = (16, 6))#candidate|918|(16, 6)|var|int32
var_919 = relay.var("var_919", dtype = "bool", shape = (16, 6))#candidate|919|(16, 6)|var|bool
var_920 = relay.var("var_920", dtype = "bool", shape = (16, 6))#candidate|920|(16, 6)|var|bool
output = func_916(var_917,var_918,var_919,var_920,)
func_921 = relay.Function([var_917,var_918,var_919,var_920,], output)
mutated_mod['func_921'] = func_921
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1017 = relay.var("var_1017", dtype = "float64", shape = (1, 1))#candidate|1017|(1, 1)|var|float64
var_1018 = relay.var("var_1018", dtype = "float64", shape = (16, 7))#candidate|1018|(16, 7)|var|float64
bop_1019 = relay.mod(var_1017.astype('float64'), var_1018.astype('float64')) # shape=(16, 7)
bop_1029 = relay.bitwise_xor(bop_1019.astype('int64'), relay.reshape(var_1018.astype('int64'), relay.shape_of(bop_1019))) # shape=(16, 7)
uop_1033 = relay.rsqrt(bop_1029.astype('float64')) # shape=(16, 7)
const_1040 = relay.const([[-0.131464,6.067793,-9.484195,6.216958,-7.463885,-8.691136,-0.819580],[1.466263,-6.512756,5.859892,9.668931,4.972306,7.637107,-3.113369],[-6.807146,1.152354,-7.978156,-5.388454,6.010549,5.688434,4.118361],[-1.277887,-5.435223,-0.730661,-0.907687,3.667005,-6.930877,2.265910],[2.423254,-2.411428,-0.755138,3.785548,-8.902725,1.847259,6.095352],[-9.106923,-5.396585,-6.700850,-2.858975,-5.581967,-0.629295,4.147074],[-7.082934,7.436570,-1.380853,-1.446334,7.313071,-9.649650,3.801527],[-7.854884,-1.233751,8.893372,6.952989,3.194271,-3.683704,-2.377770],[4.928437,3.951976,7.712615,-4.563307,3.388977,-6.830115,-3.779944],[-3.273778,-6.279512,-1.212668,8.763682,-7.204975,2.286114,1.173666],[4.591887,7.309849,6.715175,-2.717353,2.652095,-0.370536,3.489148],[5.818220,6.984246,5.273943,-2.758214,-7.766010,2.493049,-2.934353],[9.344083,2.936095,-9.844559,5.754136,0.953239,-0.103920,-2.232783],[0.122086,3.628673,3.303210,2.319771,-5.669474,-9.539419,7.709577],[6.114471,9.884868,0.154568,2.839936,5.404276,4.713093,3.211789],[0.926267,8.599702,0.773886,6.837936,1.833691,-9.515386,5.727606]], dtype = "float64")#candidate|1040|(16, 7)|const|float64
bop_1041 = relay.add(uop_1033.astype('float32'), relay.reshape(const_1040.astype('float32'), relay.shape_of(uop_1033))) # shape=(16, 7)
var_1047 = relay.var("var_1047", dtype = "float32", shape = (16, 7))#candidate|1047|(16, 7)|var|float32
bop_1048 = relay.add(bop_1041.astype('uint8'), relay.reshape(var_1047.astype('uint8'), relay.shape_of(bop_1041))) # shape=(16, 7)
bop_1051 = relay.not_equal(var_1017.astype('bool'), bop_1029.astype('bool')) # shape=(16, 7)
uop_1056 = relay.cosh(uop_1033.astype('float32')) # shape=(16, 7)
func_286_call = mod.get_global_var('func_286')
func_288_call = mutated_mod.get_global_var('func_288')
const_1060 = relay.const([[1.811490,-1.814824,2.264930,0.902421,0.319283,6.764076,-7.004555,-6.831162,-7.918026,9.600049,8.471998,-5.606282,5.638267,6.880193,-7.792194,-0.799400,-9.160781,-0.342095,2.845522,4.958615,-9.159064,4.575708,-4.827693,4.769286,2.015670,0.811420,5.959507,-8.648579,-3.544094,-1.757817,8.843244,-5.310843,-8.569237,6.704709,9.496381,-6.957051,5.204523,-2.847935,4.417461,-8.898442,-3.000482,9.883329,-6.957459,-0.371748,-6.185709,1.283801,3.529468,9.413863,-1.043239]], dtype = "float64")#candidate|1060|(1, 49)|const|float64
call_1059 = relay.TupleGetItem(func_286_call(relay.reshape(const_1060.astype('float64'), [7, 1, 7])), 1)
call_1061 = relay.TupleGetItem(func_288_call(relay.reshape(const_1060.astype('float64'), [7, 1, 7])), 1)
uop_1062 = relay.log2(bop_1041.astype('float64')) # shape=(16, 7)
func_401_call = mod.get_global_var('func_401')
func_404_call = mutated_mod.get_global_var('func_404')
const_1065 = relay.const([-4.177319,0.981250,9.999218,-4.677987,-3.384672,6.520136,-4.374546,2.709626,0.457582,8.554274,-9.059792,0.877039,4.838215,-9.446615,-6.764565,-4.107280,8.137133,-9.570698,8.615441,-3.305480,4.233021,5.303004,-5.469142,6.661120,-9.288266,-0.586406,8.582536,1.133847,2.042827,-8.762297,4.408425,-3.443981,-2.756565,-7.820878,8.992002,5.025001,-8.909056,-3.285116,7.214116,6.996975,1.579914,-7.437416,7.367828,4.920850,-1.716156,-3.105479,-4.145267,-3.108605,4.039485,-0.180815,2.880369,5.408669,6.445885,7.968586,3.132718,-7.099049,-1.397189,3.839103,-4.931306,-7.796632,-8.839005,-2.420028,-1.477700,8.714135,2.440773,-1.599355,1.448953,5.476119,-9.622409,-6.984224,0.419355,-6.513341,2.668550,8.630153,-2.710156,-0.783980,-1.235766,4.727626,-7.466585,-4.087043,0.190409,-9.974493,-6.072413,9.736754,-0.194973,-4.512685,-7.819249,-8.502093,-8.659044,5.209241], dtype = "float64")#candidate|1065|(90,)|const|float64
call_1064 = relay.TupleGetItem(func_401_call(relay.reshape(const_1065.astype('float64'), [6, 15]), relay.reshape(const_1065.astype('float64'), [6, 15]), ), 1)
call_1066 = relay.TupleGetItem(func_404_call(relay.reshape(const_1065.astype('float64'), [6, 15]), relay.reshape(const_1065.astype('float64'), [6, 15]), ), 1)
bop_1071 = relay.minimum(bop_1048.astype('uint64'), var_1017.astype('uint64')) # shape=(16, 7)
var_1077 = relay.var("var_1077", dtype = "float32", shape = (16, 7))#candidate|1077|(16, 7)|var|float32
bop_1078 = relay.bitwise_and(uop_1056.astype('int8'), relay.reshape(var_1077.astype('int8'), relay.shape_of(uop_1056))) # shape=(16, 7)
output = relay.Tuple([bop_1051,call_1059,const_1060,uop_1062,call_1064,const_1065,bop_1071,bop_1078,])
output2 = relay.Tuple([bop_1051,call_1061,const_1060,uop_1062,call_1066,const_1065,bop_1071,bop_1078,])
func_1083 = relay.Function([var_1017,var_1018,var_1047,var_1077,], output)
mod['func_1083'] = func_1083
mod = relay.transform.InferType()(mod)
var_1084 = relay.var("var_1084", dtype = "float64", shape = (1, 1))#candidate|1084|(1, 1)|var|float64
var_1085 = relay.var("var_1085", dtype = "float64", shape = (16, 7))#candidate|1085|(16, 7)|var|float64
var_1086 = relay.var("var_1086", dtype = "float32", shape = (16, 7))#candidate|1086|(16, 7)|var|float32
var_1087 = relay.var("var_1087", dtype = "float32", shape = (16, 7))#candidate|1087|(16, 7)|var|float32
output = func_1083(var_1084,var_1085,var_1086,var_1087,)
func_1088 = relay.Function([var_1084,var_1085,var_1086,var_1087,], output)
mutated_mod['func_1088'] = func_1088
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1171 = relay.var("var_1171", dtype = "float32", shape = (7, 9))#candidate|1171|(7, 9)|var|float32
uop_1172 = relay.cos(var_1171.astype('float32')) # shape=(7, 9)
uop_1181 = relay.sin(uop_1172.astype('float64')) # shape=(7, 9)
output = relay.Tuple([uop_1181,])
output2 = relay.Tuple([uop_1181,])
func_1184 = relay.Function([var_1171,], output)
mod['func_1184'] = func_1184
mod = relay.transform.InferType()(mod)
var_1185 = relay.var("var_1185", dtype = "float32", shape = (7, 9))#candidate|1185|(7, 9)|var|float32
output = func_1184(var_1185)
func_1186 = relay.Function([var_1185], output)
mutated_mod['func_1186'] = func_1186
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1281 = relay.var("var_1281", dtype = "int64", shape = ())#candidate|1281|()|var|int64
var_1282 = relay.var("var_1282", dtype = "int64", shape = (7, 11, 15))#candidate|1282|(7, 11, 15)|var|int64
bop_1283 = relay.multiply(var_1281.astype('int64'), var_1282.astype('int64')) # shape=(7, 11, 15)
uop_1289 = relay.log(bop_1283.astype('float32')) # shape=(7, 11, 15)
output = relay.Tuple([uop_1289,])
output2 = relay.Tuple([uop_1289,])
F = relay.Function([var_1281,var_1282,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1281,var_1282,], output2)
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
	relay.transform.FoldConstant(),
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
input_1281= np.array(-7, dtype='int64')
module1.set_input('var_1281', input_1281)
input_1282= np.array([[[-9,8,-3,6,-10,-3,-7,8,-2,7,-8,-9,2,7,6],[-7,-7,6,5,5,-10,2,9,9,6,-3,10,3,7,4],[-7,10,-5,3,-8,-9,-7,6,-8,-9,10,9,6,-4,3],[8,-8,-4,-5,-4,3,3,-4,-8,-5,-7,9,-5,-7,7],[6,9,-4,-8,10,-1,-7,3,-5,-4,-7,3,8,-10,8],[6,-3,-9,-10,5,9,2,-2,7,-7,-8,-3,-10,-1,-7],[-10,-5,2,1,3,3,-3,9,-2,-5,-7,-5,3,3,-10],[5,7,-4,-5,2,-2,-1,-2,2,-7,4,9,-6,7,-8],[-4,-6,4,2,6,9,-1,-2,-1,6,1,-1,1,3,-1],[-4,6,7,-3,-1,-5,4,-1,8,1,7,5,-2,-9,4],[-8,6,2,3,-4,-2,-5,6,5,7,1,-9,3,-3,-6]],[[2,-7,7,-4,3,-5,9,-3,5,3,7,-5,10,-8,10],[-10,-10,-3,-6,-7,4,-6,-1,8,-9,5,4,-2,8,7],[-6,10,-6,2,9,3,-9,-6,7,6,4,-2,1,-3,4],[7,-9,-5,-2,-8,-4,2,-9,-6,2,-4,1,-7,10,1],[1,1,5,-6,-2,-7,7,-2,10,5,-9,-10,-5,-5,8],[6,-6,-1,6,7,6,-6,3,-1,7,-7,-7,9,4,6],[1,-1,-10,7,2,3,8,-10,7,-8,-9,1,9,-7,-7],[7,4,7,2,9,-10,3,2,-2,7,3,2,4,-5,8],[-4,2,-3,-4,-4,3,5,-1,9,6,-3,6,3,1,-10],[2,4,-6,4,3,2,2,-2,-6,8,9,-1,9,4,5],[-8,10,2,8,5,-6,7,-6,-10,-10,-6,-10,-3,-4,10]],[[10,8,-8,5,8,7,-9,-9,3,-1,8,3,4,-8,10],[-8,3,-10,6,-9,9,8,10,5,9,-10,3,6,7,8],[3,8,1,9,5,-4,3,2,-3,1,-3,2,2,3,-3],[9,-6,4,-4,9,-2,-6,8,10,-5,5,2,5,-7,-5],[1,8,2,-9,6,8,10,3,-1,9,-6,7,-2,-3,-5],[9,4,-5,9,3,-1,-3,6,-1,9,-5,4,2,-4,-7],[5,5,5,-6,-6,-2,-1,-2,-1,1,2,-6,8,4,-4],[3,9,9,-1,-10,1,6,7,8,-8,9,-7,-8,-2,8],[-5,5,10,10,6,-8,2,-10,1,-6,-9,1,-7,-10,3],[-6,1,-6,7,4,-8,4,-6,8,4,-8,-8,-6,10,-7],[-6,-3,6,9,7,-9,-4,4,7,9,5,-4,-8,9,8]],[[3,9,-2,9,-10,-3,-1,2,-8,7,-2,-7,-7,-9,-7],[2,-1,-7,-8,4,-3,9,-5,7,-9,3,9,8,-8,-4],[-3,4,1,-9,-10,-9,2,-6,8,-2,-10,-10,6,-6,-6],[9,1,5,5,-1,9,-7,-9,-7,6,5,4,-6,-5,7],[7,3,9,7,4,-8,-9,1,10,4,7,4,7,2,-4],[-3,4,6,6,7,5,-2,-2,4,-10,10,10,-9,-3,8],[1,-8,-9,-8,2,-4,3,-1,8,-7,-4,8,-10,-10,7],[2,-9,-2,-9,-3,6,-7,-3,3,6,2,8,3,-4,-3],[-10,10,9,10,-2,8,2,6,6,7,-1,1,6,9,8],[4,9,9,-5,4,4,7,9,-9,3,4,-4,5,-10,1],[-6,-8,-10,1,-6,5,8,5,-7,10,-10,-6,-1,6,5]],[[7,-4,-4,-2,10,-3,9,6,9,-8,-5,-7,-5,-2,-8],[-8,-6,-10,7,-9,-3,1,-4,-5,2,5,-4,-8,9,-1],[7,-1,-4,-5,-6,-5,9,-10,4,-7,9,-4,2,-5,2],[-7,-5,-3,-3,9,5,8,-3,6,-2,-4,4,-7,9,-9],[-5,-1,-5,1,-3,7,8,1,-4,7,-1,-8,-10,-3,-6],[-2,-9,3,-7,3,-9,-9,8,8,-8,-7,-7,-8,-9,-8],[5,5,-4,9,-5,3,-5,-10,-4,-2,-5,-5,-3,1,-4],[-6,-8,-3,9,6,-4,-9,3,1,-10,-10,-1,9,-8,5],[-9,10,-2,-2,7,-9,8,10,-9,-9,-4,-6,-5,-1,5],[-4,-9,-2,1,3,3,-9,-10,-4,-3,-2,3,-4,-2,3],[10,-10,4,-9,-5,5,6,8,-6,-7,-5,-6,6,9,4]],[[3,-9,-5,6,-10,-8,-8,-8,10,-9,10,5,-2,-5,-9],[1,3,9,-5,-3,4,9,-6,8,-10,3,9,-8,-1,4],[-10,-3,6,-4,-7,-5,8,4,-8,10,-9,10,2,-1,-10],[4,9,-6,1,6,-5,-8,-3,-7,-5,3,-9,-2,-2,6],[7,2,7,-2,8,-6,-9,7,7,6,-2,-3,-3,-2,7],[5,-7,-9,6,5,-9,8,5,1,1,3,-8,2,1,-4],[-5,-10,7,9,1,9,4,10,-3,1,-10,-5,1,10,5],[-8,-5,3,10,-7,-5,-2,2,-10,5,-8,8,-3,-3,10],[-7,8,-9,-3,-2,-3,-5,-10,7,3,-3,-8,4,-9,4],[-5,-8,-4,4,9,8,-5,-4,-3,4,4,-4,-10,6,5],[9,-3,-1,-5,4,2,-3,-8,-1,2,4,-7,-10,9,4]],[[-6,-4,-9,5,1,-7,1,8,-9,-10,-6,-6,-5,10,10],[9,-7,2,7,2,-7,1,-9,9,9,6,4,-10,6,-6],[-4,-2,-8,-8,4,-5,9,10,2,-3,-9,-8,-10,-5,-6],[5,-3,-2,-5,-8,-10,-1,6,4,-1,10,-7,4,-7,-6],[-9,5,-1,-5,-5,5,-3,-4,-8,-8,7,-10,8,-4,4],[-8,-2,-6,-9,-9,-1,2,1,-4,4,9,-2,6,4,5],[-7,-2,7,3,-7,2,7,6,8,6,-4,3,-3,-7,6],[-6,6,-3,-5,9,-9,4,3,-3,-9,-4,3,10,6,-4],[1,5,-7,10,-9,6,8,-3,7,5,-4,-9,-6,-4,8],[3,7,5,-10,-7,-4,6,-2,-4,-4,5,7,8,6,-4],[-9,2,-6,-9,-3,-7,9,6,2,-10,-8,6,-2,-9,-6]]], dtype='int64')
module1.set_input('var_1282', input_1282)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1281, input_1282, )
res3 = intrp3.evaluate()(input_1281, input_1282, )
res4 = intrp4.evaluate()(input_1281, input_1282, )
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
module5.set_input('var_1281', input_1281)
module5.set_input('var_1282', input_1282)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1281, input_1282, )
res7 = intrp7.evaluate()(input_1281, input_1282, )
res8 = intrp8.evaluate()(input_1281, input_1282, )
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
module9.set_input('var_1281', input_1281)
module9.set_input('var_1282', input_1282)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1281, input_1282, )
res11 = intrp11.evaluate()(input_1281, input_1282, )
res12 = intrp12.evaluate()(input_1281, input_1282, )
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
module13.set_input('var_1281', input_1281)
module13.set_input('var_1282', input_1282)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1281, input_1282, )
res15 = intrp15.evaluate()(input_1281, input_1282, )
res16 = intrp16.evaluate()(input_1281, input_1282, )
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
module17.set_input('var_1281', input_1281)
module17.set_input('var_1282', input_1282)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1281, input_1282, )
res19 = intrp19.evaluate()(input_1281, input_1282, )
res20 = intrp20.evaluate()(input_1281, input_1282, )
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
module21.set_input('var_1281', input_1281)
module21.set_input('var_1282', input_1282)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1281, input_1282, )
res23 = intrp23.evaluate()(input_1281, input_1282, )
res24 = intrp24.evaluate()(input_1281, input_1282, )
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

'''45: TVMFuncCall
44: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
43: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
42: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
41: tvm::IRModule::FromExpr(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&)
40: tvm::IRModule::FromExprInContext(tvm::RelayExpr const&, tvm::runtime::Map<tvm::GlobalVar, tvm::BaseFunc, void, void> const&, tvm::runtime::Map<tvm::GlobalTypeVar, tvm::TypeData, void, void> const&, std::unordered_set<tvm::runtime::String, std::hash<tvm::runtime::String>, std::equal_to<tvm::runtime::String>, std::allocator<tvm::runtime::String> >)
39: tvm::IRModuleNode::Add(tvm::GlobalVar const&, tvm::BaseFunc const&, bool)
38: tvm::WarnIfMalformed(tvm::IRModule const&, tvm::relay::Function)
37: tvm::relay::FreeTypeVars(tvm::RelayExpr const&, tvm::IRModule const&)
36: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
35: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
34: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
33: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
32: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
31: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
30: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
29: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
28: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
27: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
26: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
25: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
24: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
23: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
22: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
21: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
20: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
19: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
15: tvm::relay::ExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
14: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
13: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
12: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
11: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
10: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
9: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::relay::LetNode const*)
8: tvm::relay::ExpandANormalForm(tvm::relay::LetNode const*, std::function<void (tvm::relay::LetNode const*)>, std::function<void (tvm::relay::LetNode const*)>)
7: _ZNSt17_Function_handlerIFvPKN3tvm5relay7LetNodeEEZNS1_15TypeVarEVis
6: tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)
5: void tvm::relay::ExpandDataflow<tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::MixedModeVisitor::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.533]
4: tvm::relay::MixedModeVisitor::VisitLeaf(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9RelayEx
1: tvm::relay::TypeVarEVisitor::VisitExpr_(tvm::ConstructorNode const*)
0: tvm::IRModuleNode::LookupTypeDef(tvm::GlobalTypeVar const&) const

'''