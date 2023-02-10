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
var_14 = relay.var("var_14", dtype = "int64", shape = (16, 7, 3))#candidate|14|(16, 7, 3)|var|int64
var_15 = relay.var("var_15", dtype = "int64", shape = (16, 7, 3))#candidate|15|(16, 7, 3)|var|int64
bop_16 = relay.maximum(var_14.astype('int64'), relay.reshape(var_15.astype('int64'), relay.shape_of(var_14))) # shape=(16, 7, 3)
uop_22 = relay.erf(var_14.astype('float32')) # shape=(16, 7, 3)
bop_24 = relay.less(uop_22.astype('bool'), relay.reshape(var_15.astype('bool'), relay.shape_of(uop_22))) # shape=(16, 7, 3)
bop_32 = relay.greater_equal(bop_24.astype('bool'), relay.reshape(var_14.astype('bool'), relay.shape_of(bop_24))) # shape=(16, 7, 3)
bop_42 = relay.logical_and(var_14.astype('bool'), relay.reshape(bop_16.astype('bool'), relay.shape_of(var_14))) # shape=(16, 7, 3)
bop_53 = relay.less_equal(bop_24.astype('bool'), relay.reshape(bop_42.astype('bool'), relay.shape_of(bop_24))) # shape=(16, 7, 3)
bop_59 = relay.subtract(bop_16.astype('uint8'), relay.reshape(bop_32.astype('uint8'), relay.shape_of(bop_16))) # shape=(16, 7, 3)
output = relay.Tuple([bop_53,bop_59,])
output2 = relay.Tuple([bop_53,bop_59,])
func_73 = relay.Function([var_14,var_15,], output)
mod['func_73'] = func_73
mod = relay.transform.InferType()(mod)
var_74 = relay.var("var_74", dtype = "int64", shape = (16, 7, 3))#candidate|74|(16, 7, 3)|var|int64
var_75 = relay.var("var_75", dtype = "int64", shape = (16, 7, 3))#candidate|75|(16, 7, 3)|var|int64
output = func_73(var_74,var_75,)
func_76 = relay.Function([var_74,var_75,], output)
mutated_mod['func_76'] = func_76
mutated_mod = relay.transform.InferType()(mutated_mod)
const_143 = relay.const([[[2.778752,1.064729,9.855523,-3.124377,4.457833,-8.486184,-5.923477,6.522772,6.076459,9.684745,8.250855,4.682079,2.619515,-2.167653],[-3.272155,8.721828,7.548271,9.126987,-6.846013,-0.519850,-5.395731,7.114289,-8.666166,9.600075,-1.481288,0.490993,-6.015448,-5.567426],[-9.043142,-3.716932,2.252707,-7.893347,-8.197143,8.318728,9.661164,-9.056505,-4.757609,-3.197112,3.757342,-2.115422,-9.710158,6.337875],[-8.811113,7.848542,-5.196468,-9.226993,3.789052,9.181199,-8.768942,-0.410883,-3.324880,4.476622,-1.211952,-1.981535,4.289884,9.685461]],[[-6.953168,9.353707,6.773925,7.263503,-3.096462,3.997106,-7.335642,1.731074,4.628397,-1.776164,-7.917602,-3.364904,3.634535,-7.453130],[-3.271707,1.063467,0.340681,7.913017,8.099389,-4.358344,-9.315688,0.505397,-3.115640,7.528939,-9.732513,-1.608232,-3.267806,-6.318984],[4.242644,-3.453272,-2.501859,-3.762664,-3.741643,-5.193782,-3.833084,5.193893,-7.317864,2.072756,1.691569,-0.336363,-9.515824,8.076489],[-2.639406,-4.579120,-2.602608,7.612308,-6.648356,-8.037803,0.209755,-3.300435,-5.511259,3.410716,-4.185585,4.120900,9.875838,-3.524274]],[[-2.011045,-9.950410,6.461267,-7.758911,-2.341657,-6.266679,6.157902,1.423658,-2.421099,-3.726937,-5.094922,1.281563,-8.449478,-7.931466],[1.903014,0.628526,-3.892098,-2.584587,1.657107,-3.270066,-0.676492,-9.517263,6.208209,-6.791002,-5.953032,-1.294480,-8.314885,5.150526],[9.015669,2.190177,-6.128298,-6.775597,-7.214135,-6.354412,-9.025963,3.439227,-2.701727,-1.687385,-8.950439,5.979985,-2.334245,2.607054],[6.389473,6.056789,-1.094770,0.680158,-6.537869,-2.069105,-5.784119,-5.085409,4.963767,4.716701,5.593228,7.595117,4.657126,-7.649240]]], dtype = "float64")#candidate|143|(3, 4, 14)|const|float64
const_144 = relay.const([[[-1.325246,5.116152,6.421697,-5.501937,-7.800812,-7.223157,-8.251230,-2.890980,6.513693,-5.583314,6.987607,9.845587,6.662140,0.045111],[-9.543652,8.027358,5.158315,-3.237766,9.859617,-8.837386,6.756471,-2.624086,9.321051,-8.688477,-5.874219,-3.743249,7.399320,-3.486786],[-8.891707,-8.395528,-2.168131,6.147157,-6.183584,-2.540869,2.233464,-1.974501,-8.413774,-3.539604,8.687575,-0.511955,0.469314,2.562839],[-8.749006,-2.547676,3.930520,0.703898,-5.794232,1.807818,7.701400,-3.204802,-7.385339,2.202959,1.526345,5.665364,6.995735,-3.844504]],[[-9.892185,0.751026,-1.557985,0.485217,-8.605244,3.041317,8.218298,-8.929810,5.566825,3.301708,0.158089,-5.923400,-0.304201,8.479931],[5.583793,0.560635,5.125408,2.453377,-4.842705,9.872545,1.924048,7.553943,9.274159,3.339972,0.979394,8.606903,-4.053560,4.538207],[6.243648,-7.567541,1.785625,-4.578844,-5.086520,-6.454370,-3.573355,2.325376,-6.882060,-1.932069,-6.060927,-2.565756,1.119525,-3.474583],[9.839814,-7.491999,-1.842509,-3.265855,-4.240913,-6.010212,-1.909221,4.484167,-4.953457,-5.252707,-4.705377,7.239205,3.646906,-8.970616]],[[6.093949,-2.553131,-1.070196,3.797040,5.438781,-7.743294,-8.449576,-5.488593,9.120946,0.770501,-3.287505,6.062993,-9.056273,-4.852652],[-2.088426,-7.166294,6.218790,0.790975,-4.176363,9.701107,-2.532496,6.056285,-5.585196,3.557829,-6.368961,-7.931550,-0.915058,4.884219],[-3.861757,-4.642537,-2.925933,-0.936008,9.580397,0.196325,-4.421372,5.734016,9.284186,3.717531,-6.451916,6.920936,3.298457,-3.406869],[7.013737,-2.261708,9.625529,6.119086,-7.672704,4.466379,7.851763,4.343459,0.034423,-2.728281,7.273067,0.351744,-8.757875,-9.046595]]], dtype = "float64")#candidate|144|(3, 4, 14)|const|float64
bop_145 = relay.divide(const_143.astype('float64'), relay.reshape(const_144.astype('float64'), relay.shape_of(const_143))) # shape=(3, 4, 14)
func_73_call = mod.get_global_var('func_73')
func_76_call = mutated_mod.get_global_var('func_76')
const_150 = relay.const([10,-9,-5,-1,2,1,1,-3,4,9,2,-7,3,-5,10,9,9,9,9,-9,1,4,6,6,-1,9,6,4,6,2,-1,-6,-8,-6,-6,-2,5,-10,5,-8,-2,4,3,-9,-7,-4,1,9,-4,8,10,-6,8,-5,-1,-1,-1,1,3,2,-1,10,-6,-3,6,2,-1,-4,-6,2,-3,10,-10,-10,3,6,-10,-7,-7,-9,-2,-2,1,4,4,-5,-10,5,-8,-4,-8,-6,-7,5,3,-1,-6,8,9,-3,9,-2,-2,9,-3,-2,5,7,1,8,-7,9,-1,1,10,-3,-3,-7,3,7,-3,4,9,10,7,-5,6,-10,-2,7,6,-2,6,-5,9,4,3,8,1,-1,7,-5,-10,1,5,-6,9,-1,9,-5,-8,-10,7,2,-7,10,-6,8,9,10,9,-8,-5,-7,-5,-5,-1,3,-5,8,7,-8,3,9,-7,-8,6,-9,5,5,8,2,-5,7,-1,3,-5,10,-9,1,-9,-6,-7,6,-1,-1,-3,8,1,9,2,-9,-1,6,2,-7,8,-6,8,-6,-4,-10,3,4,-3,5,3,6,-5,-7,10,7,-7,5,-6,-4,10,-7,2,-7,4,-2,-6,-2,-1,-7,3,10,-9,-5,6,5,-5,-10,-2,2,1,-5,-4,-9,8,-7,1,10,2,3,5,7,8,7,-6,-8,-6,-3,9,-7,-9,-7,4,8,-3,6,-1,-9,10,-1,-6,-3,-3,5,10,1,6,10,5,6,3,7,-6,-2,-4,-5,3,5,6,-8,-1,8,6,7,-9,1,-6,-3,7,-1,8,3,5,9,-9,-2,7,8,-9,7,10,8,-8,-10,-6,-8,-6,10,-1,5,-6,-10,6,-6,-5,-4,3,-2,-3,-10], dtype = "int64")#candidate|150|(336,)|const|int64
call_149 = relay.TupleGetItem(func_73_call(relay.reshape(const_150.astype('int64'), [16, 7, 3]), relay.reshape(const_150.astype('int64'), [16, 7, 3]), ), 1)
call_151 = relay.TupleGetItem(func_76_call(relay.reshape(const_150.astype('int64'), [16, 7, 3]), relay.reshape(const_150.astype('int64'), [16, 7, 3]), ), 1)
output = relay.Tuple([bop_145,call_149,const_150,])
output2 = relay.Tuple([bop_145,call_151,const_150,])
func_157 = relay.Function([], output)
mod['func_157'] = func_157
mod = relay.transform.InferType()(mod)
output = func_157()
func_158 = relay.Function([], output)
mutated_mod['func_158'] = func_158
mutated_mod = relay.transform.InferType()(mutated_mod)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_183 = relay.TupleGetItem(func_157_call(), 1)
call_184 = relay.TupleGetItem(func_158_call(), 1)
const_188 = relay.const([[[8,5,-3],[4,-8,9],[4,10,-2],[3,8,-5],[7,-9,-3],[-6,8,-2],[4,2,5]],[[-9,7,-10],[10,8,5],[10,-5,7],[-1,-8,-2],[-1,6,-10],[5,9,9],[-10,1,1]],[[-5,-2,-10],[-6,-1,10],[-5,-2,2],[-2,-1,8],[10,-3,8],[-9,-2,3],[1,3,-6]],[[-5,-1,3],[-9,2,-8],[2,-10,9],[6,1,10],[-2,-3,-2],[7,9,-6],[-3,7,-7]],[[-8,1,3],[-4,-9,-8],[8,9,-4],[-8,10,-9],[10,-1,9],[-2,-6,-3],[-4,-9,3]],[[-1,9,10],[-4,-6,-6],[1,1,2],[10,5,8],[7,6,-6],[-5,-9,10],[-2,10,7]],[[7,3,-9],[-5,1,-7],[9,-5,-1],[-3,-8,5],[-4,2,-6],[-9,8,8],[8,-10,-5]],[[-2,-10,1],[-7,-8,-9],[-5,-9,-4],[-6,8,8],[5,1,-7],[-10,-3,5],[2,10,-7]],[[2,-9,-6],[5,-9,-10],[-9,3,8],[-8,-8,-1],[-7,-2,-2],[6,-8,-2],[-2,-10,10]],[[-8,-3,-8],[-10,1,1],[10,4,7],[-1,-9,-1],[-7,8,4],[7,6,2],[3,4,1]],[[-6,3,2],[-6,-4,6],[-6,8,10],[3,-3,-7],[6,-6,-6],[6,-7,3],[-1,-6,4]],[[10,-5,-5],[8,4,10],[6,-3,-6],[1,-2,5],[2,2,10],[-2,-5,9],[-3,9,-8]],[[-1,5,-1],[5,-5,4],[2,-5,-6],[2,7,6],[-9,-3,-8],[-7,9,5],[3,4,6]],[[-7,4,5],[-8,9,-6],[1,3,4],[-5,2,-1],[4,6,-1],[3,-5,3],[-8,-6,-6]],[[1,-8,1],[8,2,-10],[7,-8,-9],[9,10,-1],[-1,4,8],[-2,9,-3],[-3,-8,1]],[[8,1,2],[7,-4,-5],[-10,7,-6],[5,-2,-9],[7,-8,-6],[1,8,2],[-4,1,-5]]], dtype = "uint8")#candidate|188|(16, 7, 3)|const|uint8
bop_189 = relay.divide(call_183.astype('float32'), relay.reshape(const_188.astype('float32'), relay.shape_of(call_183))) # shape=(16, 7, 3)
bop_192 = relay.divide(call_184.astype('float32'), relay.reshape(const_188.astype('float32'), relay.shape_of(call_184))) # shape=(16, 7, 3)
output = bop_189
output2 = bop_192
func_193 = relay.Function([], output)
mod['func_193'] = func_193
mod = relay.transform.InferType()(mod)
output = func_193()
func_194 = relay.Function([], output)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_231 = relay.TupleGetItem(func_157_call(), 0)
call_232 = relay.TupleGetItem(func_158_call(), 0)
func_73_call = mod.get_global_var('func_73')
func_76_call = mutated_mod.get_global_var('func_76')
var_234 = relay.var("var_234", dtype = "int64", shape = (336,))#candidate|234|(336,)|var|int64
call_233 = relay.TupleGetItem(func_73_call(relay.reshape(var_234.astype('int64'), [16, 7, 3]), relay.reshape(var_234.astype('int64'), [16, 7, 3]), ), 1)
call_235 = relay.TupleGetItem(func_76_call(relay.reshape(var_234.astype('int64'), [16, 7, 3]), relay.reshape(var_234.astype('int64'), [16, 7, 3]), ), 1)
output = relay.Tuple([call_231,call_233,var_234,])
output2 = relay.Tuple([call_232,call_235,var_234,])
func_237 = relay.Function([var_234,], output)
mod['func_237'] = func_237
mod = relay.transform.InferType()(mod)
var_238 = relay.var("var_238", dtype = "int64", shape = (336,))#candidate|238|(336,)|var|int64
output = func_237(var_238)
func_239 = relay.Function([var_238], output)
mutated_mod['func_239'] = func_239
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_288 = func_193_call()
call_289 = func_193_call()
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_301 = func_193_call()
call_302 = func_193_call()
output = relay.Tuple([call_288,call_301,])
output2 = relay.Tuple([call_289,call_302,])
func_312 = relay.Function([], output)
mod['func_312'] = func_312
mod = relay.transform.InferType()(mod)
mutated_mod['func_312'] = func_312
mutated_mod = relay.transform.InferType()(mutated_mod)
func_312_call = mutated_mod.get_global_var('func_312')
call_313 = func_312_call()
output = call_313
func_314 = relay.Function([], output)
mutated_mod['func_314'] = func_314
mutated_mod = relay.transform.InferType()(mutated_mod)
var_335 = relay.var("var_335", dtype = "float32", shape = (13, 15))#candidate|335|(13, 15)|var|float32
const_336 = relay.const([[2.886624,8.082726,7.572695,-1.994968,2.222586,9.331005,-1.272896,-3.074271,-6.059258,-7.159831,-4.847860,-3.536971,-2.709783,-7.012266,-2.566125],[1.898546,5.595321,6.422666,1.666396,-7.183624,9.042847,3.437396,-8.756548,-3.284173,-7.953808,4.335645,-2.886161,-4.344441,6.970623,9.800653],[-1.104185,-8.225043,7.204854,-1.407010,3.892060,3.343751,3.461012,-3.067399,-7.396931,-9.177991,1.731947,3.621232,8.913410,-4.502265,-6.340138],[6.406396,-0.478984,1.896094,2.386768,-6.913285,-5.491677,4.401724,9.733007,2.938275,8.981843,-8.802480,1.042314,-8.692240,0.445948,-7.347316],[-9.149507,-4.306963,1.770222,-0.313933,9.091667,6.328561,-1.341413,1.679755,-1.371172,-9.904305,0.840890,2.268359,-9.490546,6.287199,-3.943757],[-0.005881,6.109221,-6.703631,-8.244585,-0.365484,3.356732,6.234018,-8.228037,0.108712,4.724960,-1.122520,-3.636527,5.894329,-7.381053,-0.227991],[-4.693161,6.308686,3.844829,-2.174010,3.202290,-5.670432,-4.254440,-0.305557,-3.303007,-6.516166,9.146092,-5.792135,-4.836902,9.543108,7.367397],[6.346916,-5.235719,4.836225,-3.193264,5.293429,-1.127054,4.947097,5.465160,-8.357210,1.776476,4.485306,0.914173,3.099284,6.807844,-2.221466],[-8.866155,8.696683,2.550629,1.598168,-4.791157,5.249447,-5.954712,-1.341224,-7.030615,4.019581,4.399439,6.560161,8.378418,-4.007772,2.596615],[7.368947,-3.612489,-0.619191,3.497385,-2.457932,-7.761382,9.706777,-3.734963,-6.670809,-5.704305,-4.517047,-3.910299,-0.129794,-5.810883,-1.308586],[-6.341964,-9.500068,-9.387610,-2.570527,-5.966865,-1.354538,-8.569307,6.041959,4.925577,5.319175,1.829674,9.631821,5.254474,3.645058,-3.980096],[3.666146,4.992773,2.342584,-4.085310,-6.999480,-3.441922,2.204575,-0.424844,3.929002,-8.988123,5.074537,4.788069,7.380909,-8.279849,-4.236660],[0.575905,9.873187,9.280125,-0.853674,-4.927286,-0.473339,0.743926,6.711122,-9.641349,4.833649,-8.004761,6.574413,7.902583,-6.615754,0.410571]], dtype = "float32")#candidate|336|(13, 15)|const|float32
bop_337 = relay.greater_equal(var_335.astype('bool'), relay.reshape(const_336.astype('bool'), relay.shape_of(var_335))) # shape=(13, 15)
output = bop_337
output2 = bop_337
func_346 = relay.Function([var_335,], output)
mod['func_346'] = func_346
mod = relay.transform.InferType()(mod)
mutated_mod['func_346'] = func_346
mutated_mod = relay.transform.InferType()(mutated_mod)
var_347 = relay.var("var_347", dtype = "float32", shape = (13, 15))#candidate|347|(13, 15)|var|float32
func_346_call = mutated_mod.get_global_var('func_346')
call_348 = func_346_call(var_347)
output = call_348
func_349 = relay.Function([var_347], output)
mutated_mod['func_349'] = func_349
mutated_mod = relay.transform.InferType()(mutated_mod)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_354 = relay.TupleGetItem(func_157_call(), 1)
call_355 = relay.TupleGetItem(func_158_call(), 1)
output = relay.Tuple([call_354,])
output2 = relay.Tuple([call_355,])
func_356 = relay.Function([], output)
mod['func_356'] = func_356
mod = relay.transform.InferType()(mod)
mutated_mod['func_356'] = func_356
mutated_mod = relay.transform.InferType()(mutated_mod)
func_356_call = mutated_mod.get_global_var('func_356')
call_357 = func_356_call()
output = call_357
func_358 = relay.Function([], output)
mutated_mod['func_358'] = func_358
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_370 = func_193_call()
call_371 = func_193_call()
uop_380 = relay.sinh(call_370.astype('float64')) # shape=(16, 7, 3)
uop_382 = relay.sinh(call_371.astype('float64')) # shape=(16, 7, 3)
bop_387 = relay.floor_mod(uop_380.astype('float32'), relay.reshape(call_370.astype('float32'), relay.shape_of(uop_380))) # shape=(16, 7, 3)
bop_390 = relay.floor_mod(uop_382.astype('float32'), relay.reshape(call_371.astype('float32'), relay.shape_of(uop_382))) # shape=(16, 7, 3)
uop_391 = relay.asinh(uop_380.astype('float64')) # shape=(16, 7, 3)
uop_393 = relay.asinh(uop_382.astype('float64')) # shape=(16, 7, 3)
output = relay.Tuple([bop_387,uop_391,])
output2 = relay.Tuple([bop_390,uop_393,])
func_397 = relay.Function([], output)
mod['func_397'] = func_397
mod = relay.transform.InferType()(mod)
mutated_mod['func_397'] = func_397
mutated_mod = relay.transform.InferType()(mutated_mod)
func_397_call = mutated_mod.get_global_var('func_397')
call_398 = func_397_call()
output = call_398
func_399 = relay.Function([], output)
mutated_mod['func_399'] = func_399
mutated_mod = relay.transform.InferType()(mutated_mod)
var_402 = relay.var("var_402", dtype = "bool", shape = (4, 3, 7))#candidate|402|(4, 3, 7)|var|bool
var_403 = relay.var("var_403", dtype = "bool", shape = (4, 3, 7))#candidate|403|(4, 3, 7)|var|bool
bop_404 = relay.logical_or(var_402.astype('bool'), relay.reshape(var_403.astype('bool'), relay.shape_of(var_402))) # shape=(4, 3, 7)
func_346_call = mod.get_global_var('func_346')
func_349_call = mutated_mod.get_global_var('func_349')
var_408 = relay.var("var_408", dtype = "float32", shape = (195,))#candidate|408|(195,)|var|float32
call_407 = func_346_call(relay.reshape(var_408.astype('float32'), [13, 15]))
call_409 = func_346_call(relay.reshape(var_408.astype('float32'), [13, 15]))
output = relay.Tuple([bop_404,call_407,var_408,])
output2 = relay.Tuple([bop_404,call_409,var_408,])
func_415 = relay.Function([var_402,var_403,var_408,], output)
mod['func_415'] = func_415
mod = relay.transform.InferType()(mod)
mutated_mod['func_415'] = func_415
mutated_mod = relay.transform.InferType()(mutated_mod)
func_415_call = mutated_mod.get_global_var('func_415')
var_417 = relay.var("var_417", dtype = "bool", shape = (4, 3, 7))#candidate|417|(4, 3, 7)|var|bool
var_418 = relay.var("var_418", dtype = "bool", shape = (4, 3, 7))#candidate|418|(4, 3, 7)|var|bool
var_419 = relay.var("var_419", dtype = "float32", shape = (195,))#candidate|419|(195,)|var|float32
call_416 = func_415_call(var_417,var_418,var_419,)
output = call_416
func_420 = relay.Function([var_417,var_418,var_419,], output)
mutated_mod['func_420'] = func_420
mutated_mod = relay.transform.InferType()(mutated_mod)
func_356_call = mod.get_global_var('func_356')
func_358_call = mutated_mod.get_global_var('func_358')
call_422 = relay.TupleGetItem(func_356_call(), 0)
call_423 = relay.TupleGetItem(func_358_call(), 0)
var_426 = relay.var("var_426", dtype = "uint8", shape = (16, 7, 3))#candidate|426|(16, 7, 3)|var|uint8
bop_427 = relay.logical_xor(call_422.astype('uint32'), relay.reshape(var_426.astype('uint32'), relay.shape_of(call_422))) # shape=(16, 7, 3)
bop_430 = relay.logical_xor(call_423.astype('uint32'), relay.reshape(var_426.astype('uint32'), relay.shape_of(call_423))) # shape=(16, 7, 3)
bop_433 = relay.bitwise_xor(call_422.astype('int32'), relay.reshape(var_426.astype('int32'), relay.shape_of(call_422))) # shape=(16, 7, 3)
bop_436 = relay.bitwise_xor(call_423.astype('int32'), relay.reshape(var_426.astype('int32'), relay.shape_of(call_423))) # shape=(16, 7, 3)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_437 = func_193_call()
call_438 = func_193_call()
bop_439 = relay.greater(call_422.astype('bool'), relay.reshape(bop_427.astype('bool'), relay.shape_of(call_422))) # shape=(16, 7, 3)
bop_442 = relay.greater(call_423.astype('bool'), relay.reshape(bop_430.astype('bool'), relay.shape_of(call_423))) # shape=(16, 7, 3)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_445 = func_193_call()
call_446 = func_193_call()
bop_449 = relay.bitwise_and(var_426.astype('int8'), relay.reshape(bop_433.astype('int8'), relay.shape_of(var_426))) # shape=(16, 7, 3)
bop_452 = relay.bitwise_and(var_426.astype('int8'), relay.reshape(bop_436.astype('int8'), relay.shape_of(var_426))) # shape=(16, 7, 3)
bop_455 = relay.left_shift(call_445.astype('uint64'), relay.reshape(bop_427.astype('uint64'), relay.shape_of(call_445))) # shape=(16, 7, 3)
bop_458 = relay.left_shift(call_446.astype('uint64'), relay.reshape(bop_430.astype('uint64'), relay.shape_of(call_446))) # shape=(16, 7, 3)
func_312_call = mod.get_global_var('func_312')
func_314_call = mutated_mod.get_global_var('func_314')
call_460 = relay.TupleGetItem(func_312_call(), 0)
call_461 = relay.TupleGetItem(func_314_call(), 0)
func_415_call = mod.get_global_var('func_415')
func_420_call = mutated_mod.get_global_var('func_420')
const_466 = relay.const([True,True,True,True,True,True,True,False,False,False,False,False,True,False,False,True,True,False,False,True,True,False,True,True,False,True,True,False,True,False,True,False,True,False,True,True,True,False,True,True,False,True,True,False,True,True,True,True,True,False,False,False,False,True,True,False,False,True,True,True,True,False,True,False,False,False,True,True,True,True,False,True,False,True,True,False,False,True,True,True,True,True,True,True], dtype = "bool")#candidate|466|(84,)|const|bool
const_467 = relay.const([-6.734325,-6.412734,-3.220368,0.597956,6.323958,-2.096389,5.734579,-3.280726,-3.588748,7.313579,4.556763,2.323629,-5.652844,-7.563344,0.605973,-7.171465,9.539225,-8.809364,9.230032,8.885042,-2.160182,-8.631164,-8.242714,0.858662,1.947231,6.211546,-7.772866,4.238388,0.028238,-4.711769,9.517017,-0.013950,-9.352072,-5.159315,5.768522,-1.981782,0.469126,9.158273,4.023454,-8.057060,6.840731,-1.220112,-6.900089,9.518526,0.700796,-3.297719,-2.298446,3.093568,7.995705,-6.411396,6.653276,-5.722189,-7.221637,-6.825907,-9.869105,-1.427287,-2.497255,8.360234,-6.861463,4.805109,-1.028732,-5.202183,-9.320889,3.307976,4.463704,-8.974415,5.758035,0.196397,-4.078439,8.540275,-3.664586,2.380684,-7.338422,6.158339,6.454003,2.064445,-9.915127,-7.187087,3.856230,-2.072513,0.526905,-4.873766,1.432088,-9.089088,8.307083,-6.841087,-9.598721,0.923409,-4.525305,2.721266,4.528579,-5.622145,2.587537,-6.225254,-9.830556,5.938015,-7.370636,-5.301123,-2.537169,4.498675,4.142179,8.759225,4.811298,-4.265709,-4.637711,2.459999,-8.581153,8.661828,-1.447418,5.698294,9.209724,9.249979,-5.483175,5.042094,-0.541381,-4.801485,2.114929,9.324026,-7.264548,-0.916434,-4.511759,-5.579839,4.075292,-7.086265,-7.867804,6.839942,-8.252312,2.141342,-4.852708,0.465171,0.874845,-8.655867,-5.064200,-7.448714,-3.335186,2.784588,-8.705836,8.051023,6.161267,-7.975230,5.768057,2.958041,-3.428549,-1.507119,-9.725285,0.203239,-6.566672,-7.481242,-8.664091,6.647848,7.042998,1.884669,6.388269,-7.556057,-5.091152,-1.336768,-0.845966,4.640307,-8.253457,3.149599,-4.621262,-2.763449,7.721444,1.956575,2.268479,-5.370816,2.015302,5.621955,-8.407301,9.881093,-2.751525,7.116316,-0.696977,8.666322,-1.933996,8.464242,4.021183,6.920570,-4.310225,-9.977421,-2.821947,-2.235216,-2.864046,-9.163499,5.843017,7.273412,6.276032,-7.245529,-6.106270,-9.082652,2.862361,6.527921,0.864005,9.930219,6.095277], dtype = "float32")#candidate|467|(195,)|const|float32
call_465 = relay.TupleGetItem(func_415_call(relay.reshape(const_466.astype('bool'), [4, 3, 7]), relay.reshape(const_466.astype('bool'), [4, 3, 7]), relay.reshape(const_467.astype('float32'), [195,]), ), 1)
call_468 = relay.TupleGetItem(func_420_call(relay.reshape(const_466.astype('bool'), [4, 3, 7]), relay.reshape(const_466.astype('bool'), [4, 3, 7]), relay.reshape(const_467.astype('float32'), [195,]), ), 1)
output = relay.Tuple([call_437,bop_439,bop_449,bop_455,call_460,call_465,const_466,const_467,])
output2 = relay.Tuple([call_438,bop_442,bop_452,bop_458,call_461,call_468,const_466,const_467,])
func_472 = relay.Function([var_426,], output)
mod['func_472'] = func_472
mod = relay.transform.InferType()(mod)
mutated_mod['func_472'] = func_472
mutated_mod = relay.transform.InferType()(mutated_mod)
var_473 = relay.var("var_473", dtype = "uint8", shape = (16, 7, 3))#candidate|473|(16, 7, 3)|var|uint8
func_472_call = mutated_mod.get_global_var('func_472')
call_474 = func_472_call(var_473)
output = call_474
func_475 = relay.Function([var_473], output)
mutated_mod['func_475'] = func_475
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_485 = func_193_call()
call_486 = func_193_call()
uop_496 = relay.log(call_485.astype('float64')) # shape=(16, 7, 3)
uop_498 = relay.log(call_486.astype('float64')) # shape=(16, 7, 3)
uop_500 = relay.cosh(uop_496.astype('float32')) # shape=(16, 7, 3)
uop_502 = relay.cosh(uop_498.astype('float32')) # shape=(16, 7, 3)
output = uop_500
output2 = uop_502
func_507 = relay.Function([], output)
mod['func_507'] = func_507
mod = relay.transform.InferType()(mod)
output = func_507()
func_508 = relay.Function([], output)
mutated_mod['func_508'] = func_508
mutated_mod = relay.transform.InferType()(mutated_mod)
var_520 = relay.var("var_520", dtype = "float64", shape = (14, 4, 5))#candidate|520|(14, 4, 5)|var|float64
var_521 = relay.var("var_521", dtype = "float64", shape = (14, 4, 5))#candidate|521|(14, 4, 5)|var|float64
bop_522 = relay.power(var_520.astype('float64'), relay.reshape(var_521.astype('float64'), relay.shape_of(var_520))) # shape=(14, 4, 5)
func_73_call = mod.get_global_var('func_73')
func_76_call = mutated_mod.get_global_var('func_76')
var_527 = relay.var("var_527", dtype = "int64", shape = (336,))#candidate|527|(336,)|var|int64
call_526 = relay.TupleGetItem(func_73_call(relay.reshape(var_527.astype('int64'), [16, 7, 3]), relay.reshape(var_527.astype('int64'), [16, 7, 3]), ), 0)
call_528 = relay.TupleGetItem(func_76_call(relay.reshape(var_527.astype('int64'), [16, 7, 3]), relay.reshape(var_527.astype('int64'), [16, 7, 3]), ), 0)
func_472_call = mod.get_global_var('func_472')
func_475_call = mutated_mod.get_global_var('func_475')
call_530 = relay.TupleGetItem(func_472_call(relay.reshape(var_527.astype('uint8'), [16, 7, 3])), 4)
call_531 = relay.TupleGetItem(func_475_call(relay.reshape(var_527.astype('uint8'), [16, 7, 3])), 4)
uop_533 = relay.tan(call_530.astype('float32')) # shape=(16, 7, 3)
uop_535 = relay.tan(call_531.astype('float32')) # shape=(16, 7, 3)
var_537 = relay.var("var_537", dtype = "float32", shape = (16, 7, 3))#candidate|537|(16, 7, 3)|var|float32
bop_538 = relay.minimum(uop_533.astype('int32'), relay.reshape(var_537.astype('int32'), relay.shape_of(uop_533))) # shape=(16, 7, 3)
bop_541 = relay.minimum(uop_535.astype('int32'), relay.reshape(var_537.astype('int32'), relay.shape_of(uop_535))) # shape=(16, 7, 3)
bop_542 = relay.bitwise_and(bop_538.astype('int64'), relay.reshape(var_537.astype('int64'), relay.shape_of(bop_538))) # shape=(16, 7, 3)
bop_545 = relay.bitwise_and(bop_541.astype('int64'), relay.reshape(var_537.astype('int64'), relay.shape_of(bop_541))) # shape=(16, 7, 3)
output = relay.Tuple([bop_522,call_526,var_527,bop_542,])
output2 = relay.Tuple([bop_522,call_528,var_527,bop_545,])
func_555 = relay.Function([var_520,var_521,var_527,var_537,], output)
mod['func_555'] = func_555
mod = relay.transform.InferType()(mod)
mutated_mod['func_555'] = func_555
mutated_mod = relay.transform.InferType()(mutated_mod)
func_555_call = mutated_mod.get_global_var('func_555')
var_557 = relay.var("var_557", dtype = "float64", shape = (14, 4, 5))#candidate|557|(14, 4, 5)|var|float64
var_558 = relay.var("var_558", dtype = "float64", shape = (14, 4, 5))#candidate|558|(14, 4, 5)|var|float64
var_559 = relay.var("var_559", dtype = "int64", shape = (336,))#candidate|559|(336,)|var|int64
var_560 = relay.var("var_560", dtype = "float32", shape = (16, 7, 3))#candidate|560|(16, 7, 3)|var|float32
call_556 = func_555_call(var_557,var_558,var_559,var_560,)
output = call_556
func_561 = relay.Function([var_557,var_558,var_559,var_560,], output)
mutated_mod['func_561'] = func_561
mutated_mod = relay.transform.InferType()(mutated_mod)
func_507_call = mod.get_global_var('func_507')
func_508_call = mutated_mod.get_global_var('func_508')
call_576 = func_507_call()
call_577 = func_507_call()
output = call_576
output2 = call_577
func_585 = relay.Function([], output)
mod['func_585'] = func_585
mod = relay.transform.InferType()(mod)
output = func_585()
func_586 = relay.Function([], output)
mutated_mod['func_586'] = func_586
mutated_mod = relay.transform.InferType()(mutated_mod)
func_397_call = mod.get_global_var('func_397')
func_399_call = mutated_mod.get_global_var('func_399')
call_590 = relay.TupleGetItem(func_397_call(), 0)
call_591 = relay.TupleGetItem(func_399_call(), 0)
uop_611 = relay.exp(call_590.astype('float64')) # shape=(16, 7, 3)
uop_613 = relay.exp(call_591.astype('float64')) # shape=(16, 7, 3)
uop_616 = relay.log2(uop_611.astype('float64')) # shape=(16, 7, 3)
uop_618 = relay.log2(uop_613.astype('float64')) # shape=(16, 7, 3)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_621 = relay.TupleGetItem(func_157_call(), 0)
call_622 = relay.TupleGetItem(func_158_call(), 0)
bop_625 = relay.floor_divide(uop_616.astype('float32'), relay.reshape(uop_611.astype('float32'), relay.shape_of(uop_616))) # shape=(16, 7, 3)
bop_628 = relay.floor_divide(uop_618.astype('float32'), relay.reshape(uop_613.astype('float32'), relay.shape_of(uop_618))) # shape=(16, 7, 3)
output = relay.Tuple([call_621,bop_625,])
output2 = relay.Tuple([call_622,bop_628,])
func_631 = relay.Function([], output)
mod['func_631'] = func_631
mod = relay.transform.InferType()(mod)
mutated_mod['func_631'] = func_631
mutated_mod = relay.transform.InferType()(mutated_mod)
func_631_call = mutated_mod.get_global_var('func_631')
call_632 = func_631_call()
output = call_632
func_633 = relay.Function([], output)
mutated_mod['func_633'] = func_633
mutated_mod = relay.transform.InferType()(mutated_mod)
func_585_call = mod.get_global_var('func_585')
func_586_call = mutated_mod.get_global_var('func_586')
call_636 = func_585_call()
call_637 = func_585_call()
func_631_call = mod.get_global_var('func_631')
func_633_call = mutated_mod.get_global_var('func_633')
call_638 = relay.TupleGetItem(func_631_call(), 1)
call_639 = relay.TupleGetItem(func_633_call(), 1)
func_346_call = mod.get_global_var('func_346')
func_349_call = mutated_mod.get_global_var('func_349')
var_644 = relay.var("var_644", dtype = "float32", shape = (195,))#candidate|644|(195,)|var|float32
call_643 = func_346_call(relay.reshape(var_644.astype('float32'), [13, 15]))
call_645 = func_346_call(relay.reshape(var_644.astype('float32'), [13, 15]))
output = relay.Tuple([call_636,call_638,call_643,var_644,])
output2 = relay.Tuple([call_637,call_639,call_645,var_644,])
func_664 = relay.Function([var_644,], output)
mod['func_664'] = func_664
mod = relay.transform.InferType()(mod)
var_665 = relay.var("var_665", dtype = "float32", shape = (195,))#candidate|665|(195,)|var|float32
output = func_664(var_665)
func_666 = relay.Function([var_665], output)
mutated_mod['func_666'] = func_666
mutated_mod = relay.transform.InferType()(mutated_mod)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_692 = relay.TupleGetItem(func_157_call(), 2)
call_693 = relay.TupleGetItem(func_158_call(), 2)
output = relay.Tuple([call_692,])
output2 = relay.Tuple([call_693,])
func_695 = relay.Function([], output)
mod['func_695'] = func_695
mod = relay.transform.InferType()(mod)
output = func_695()
func_696 = relay.Function([], output)
mutated_mod['func_696'] = func_696
mutated_mod = relay.transform.InferType()(mutated_mod)
var_715 = relay.var("var_715", dtype = "uint64", shape = (9, 4, 14))#candidate|715|(9, 4, 14)|var|uint64
var_716 = relay.var("var_716", dtype = "uint64", shape = (9, 4, 14))#candidate|716|(9, 4, 14)|var|uint64
bop_717 = relay.bitwise_or(var_715.astype('uint64'), relay.reshape(var_716.astype('uint64'), relay.shape_of(var_715))) # shape=(9, 4, 14)
func_397_call = mod.get_global_var('func_397')
func_399_call = mutated_mod.get_global_var('func_399')
call_721 = relay.TupleGetItem(func_397_call(), 1)
call_722 = relay.TupleGetItem(func_399_call(), 1)
output = relay.Tuple([bop_717,call_721,])
output2 = relay.Tuple([bop_717,call_722,])
func_730 = relay.Function([var_715,var_716,], output)
mod['func_730'] = func_730
mod = relay.transform.InferType()(mod)
mutated_mod['func_730'] = func_730
mutated_mod = relay.transform.InferType()(mutated_mod)
func_730_call = mutated_mod.get_global_var('func_730')
var_732 = relay.var("var_732", dtype = "uint64", shape = (9, 4, 14))#candidate|732|(9, 4, 14)|var|uint64
var_733 = relay.var("var_733", dtype = "uint64", shape = (9, 4, 14))#candidate|733|(9, 4, 14)|var|uint64
call_731 = func_730_call(var_732,var_733,)
output = call_731
func_734 = relay.Function([var_732,var_733,], output)
mutated_mod['func_734'] = func_734
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_761 = relay.TupleGetItem(func_695_call(), 0)
call_762 = relay.TupleGetItem(func_696_call(), 0)
uop_776 = relay.tan(call_761.astype('float32')) # shape=(336,)
uop_778 = relay.tan(call_762.astype('float32')) # shape=(336,)
output = uop_776
output2 = uop_778
func_784 = relay.Function([], output)
mod['func_784'] = func_784
mod = relay.transform.InferType()(mod)
output = func_784()
func_785 = relay.Function([], output)
mutated_mod['func_785'] = func_785
mutated_mod = relay.transform.InferType()(mutated_mod)
const_791 = relay.const([[7.474747,-5.569360,-1.447325,2.585786,-2.545796,-5.429463,2.622278,0.903670,3.127419,2.826822,-4.208339,-2.886713],[2.282057,-5.317412,-5.216162,-5.422462,0.487577,7.228565,-1.681371,6.901234,-7.848012,4.316082,0.264329,-8.460983]], dtype = "float64")#candidate|791|(2, 12)|const|float64
uop_792 = relay.atanh(const_791.astype('float64')) # shape=(2, 12)
output = relay.Tuple([uop_792,])
output2 = relay.Tuple([uop_792,])
func_796 = relay.Function([], output)
mod['func_796'] = func_796
mod = relay.transform.InferType()(mod)
mutated_mod['func_796'] = func_796
mutated_mod = relay.transform.InferType()(mutated_mod)
func_796_call = mutated_mod.get_global_var('func_796')
call_797 = func_796_call()
output = call_797
func_798 = relay.Function([], output)
mutated_mod['func_798'] = func_798
mutated_mod = relay.transform.InferType()(mutated_mod)
func_507_call = mod.get_global_var('func_507')
func_508_call = mutated_mod.get_global_var('func_508')
call_834 = func_507_call()
call_835 = func_507_call()
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_851 = func_193_call()
call_852 = func_193_call()
func_784_call = mod.get_global_var('func_784')
func_785_call = mutated_mod.get_global_var('func_785')
call_854 = func_784_call()
call_855 = func_784_call()
bop_870 = relay.right_shift(call_834.astype('uint64'), relay.reshape(call_851.astype('uint64'), relay.shape_of(call_834))) # shape=(16, 7, 3)
bop_873 = relay.right_shift(call_835.astype('uint64'), relay.reshape(call_852.astype('uint64'), relay.shape_of(call_835))) # shape=(16, 7, 3)
func_555_call = mod.get_global_var('func_555')
func_561_call = mutated_mod.get_global_var('func_561')
const_889 = relay.const([-6.561673,-2.642912,6.105504,-9.858005,8.121361,-1.587607,3.257805,-1.752710,-8.106220,-9.604522,9.073411,7.216270,8.165059,7.382309,-6.636039,-8.105595,0.565774,7.043856,-6.587850,9.112127,-9.285011,7.690085,-3.993543,0.996308,-9.689152,-5.669185,-3.948045,-8.295234,7.623529,-1.293959,3.357971,-2.445668,9.253043,-3.033323,-1.974734,-3.785228,6.731085,-5.015232,7.438972,6.196535,0.961373,1.313680,7.828972,5.232719,-4.151923,-4.976269,-1.101692,-3.523161,7.848547,9.185558,-6.450735,-5.319296,2.361095,4.413036,-1.980958,3.428797,5.485236,-5.669583,-2.744300,3.194202,0.513679,7.542005,2.456621,-7.294469,-3.588107,-3.559690,4.192061,-0.337479,7.053548,5.164463,-4.301541,9.742016,-5.318125,-4.074344,6.649924,1.985658,-1.388686,3.199031,-9.477362,-5.559490,1.391796,8.989248,0.447935,-1.054595,-7.044992,-2.344194,-4.133365,-1.129427,9.505093,2.706569,-1.993313,-6.337853,8.292668,-7.951646,4.371751,-9.719536,9.884987,1.799736,-9.522386,7.195820,7.296527,3.041412,3.245898,-4.312009,7.491962,-1.578963,-4.516082,-0.850740,-4.964411,9.166819,5.901192,-0.391395,-1.348239,6.096949,0.981918,8.012608,-9.325082,-0.873912,-7.648996,9.226261,4.411860,6.956273,-0.516360,3.359388,5.087509,-9.872264,9.551531,0.669293,-9.968059,-1.312159,-8.588927,-0.440469,-6.239161,-4.657268,-1.763889,-8.039530,7.625616,9.313589,7.138575,-4.715370,-1.702010,9.584886,-2.569291,7.181216,9.318715,-8.932618,2.467940,4.192598,-7.631502,-8.093972,4.808471,-9.117525,-9.738497,-0.079591,-2.563403,-8.746050,0.001017,-5.688047,1.961947,-2.014606,7.564523,-2.686365,4.821940,4.014104,1.859695,-5.399099,-5.182702,-5.860632,2.439020,2.396208,2.739330,-3.433521,2.105864,-0.835495,-0.872180,-2.579004,-3.771916,-9.705752,-2.579908,1.898255,9.559426,-4.456844,-5.029085,-4.320668,9.843034,3.288809,-5.890728,6.097990,0.685995,-7.467937,-1.586189,4.910708,9.608296,-9.123216,5.375594,0.930400,-9.165767,0.421627,-2.553822,-4.095037,9.605325,4.693974,9.476199,9.767279,-9.810914,-8.416111,-7.412890,7.964610,2.372534,7.517388,8.455762,0.665798,-3.964149,-7.883017,-5.740054,-4.336388,5.512206,4.442995,-1.618136,8.732355,-6.678526,3.332231,9.673990,-7.411129,8.395526,-1.290549,3.732444,-5.724908,-1.193789,-5.811422,7.401225,-7.345011,-5.058822,0.205386,-7.915050,-2.988848,-1.804554,-6.480755,-1.228515,-3.785489,-5.368575,0.035970,-2.977441,-6.961267,-4.840267,1.540579,5.889989,-3.888826,-8.931573,4.569878,-8.229224,6.210515,4.427232,0.307353,0.572262,4.856277,-1.408367,4.753635,-1.983359,9.993195,-1.531823,-2.192841,-8.552207,-0.858981,-6.428966,5.743148,0.714803,4.726785,-8.744481,-4.414577,-3.276664,-7.424318,-4.121510,9.939662,9.587984,-3.484810,-0.017020,6.203159,5.236916,0.619332], dtype = "float64")#candidate|889|(280,)|const|float64
call_888 = relay.TupleGetItem(func_555_call(relay.reshape(const_889.astype('float64'), [14, 4, 5]), relay.reshape(const_889.astype('float64'), [14, 4, 5]), relay.reshape(call_854.astype('int64'), [336,]), relay.reshape(call_854.astype('float32'), [16, 7, 3]), ), 3)
call_890 = relay.TupleGetItem(func_561_call(relay.reshape(const_889.astype('float64'), [14, 4, 5]), relay.reshape(const_889.astype('float64'), [14, 4, 5]), relay.reshape(call_854.astype('int64'), [336,]), relay.reshape(call_854.astype('float32'), [16, 7, 3]), ), 3)
func_472_call = mod.get_global_var('func_472')
func_475_call = mutated_mod.get_global_var('func_475')
call_898 = relay.TupleGetItem(func_472_call(relay.reshape(call_888.astype('uint8'), [16, 7, 3])), 0)
call_899 = relay.TupleGetItem(func_475_call(relay.reshape(call_888.astype('uint8'), [16, 7, 3])), 0)
bop_902 = relay.power(bop_870.astype('float64'), relay.reshape(call_888.astype('float64'), relay.shape_of(bop_870))) # shape=(16, 7, 3)
bop_905 = relay.power(bop_873.astype('float64'), relay.reshape(call_890.astype('float64'), relay.shape_of(bop_873))) # shape=(16, 7, 3)
func_397_call = mod.get_global_var('func_397')
func_399_call = mutated_mod.get_global_var('func_399')
call_913 = relay.TupleGetItem(func_397_call(), 1)
call_914 = relay.TupleGetItem(func_399_call(), 1)
var_916 = relay.var("var_916", dtype = "float64", shape = (16, 7, 3))#candidate|916|(16, 7, 3)|var|float64
bop_917 = relay.power(bop_902.astype('float64'), relay.reshape(var_916.astype('float64'), relay.shape_of(bop_902))) # shape=(16, 7, 3)
bop_920 = relay.power(bop_905.astype('float64'), relay.reshape(var_916.astype('float64'), relay.shape_of(bop_905))) # shape=(16, 7, 3)
func_397_call = mod.get_global_var('func_397')
func_399_call = mutated_mod.get_global_var('func_399')
call_924 = relay.TupleGetItem(func_397_call(), 1)
call_925 = relay.TupleGetItem(func_399_call(), 1)
uop_926 = relay.log10(call_888.astype('float64')) # shape=(16, 7, 3)
uop_928 = relay.log10(call_890.astype('float64')) # shape=(16, 7, 3)
var_932 = relay.var("var_932", dtype = "float32", shape = (16, 7, 3))#candidate|932|(16, 7, 3)|var|float32
bop_933 = relay.logical_or(call_851.astype('bool'), relay.reshape(var_932.astype('bool'), relay.shape_of(call_851))) # shape=(16, 7, 3)
bop_936 = relay.logical_or(call_852.astype('bool'), relay.reshape(var_932.astype('bool'), relay.shape_of(call_852))) # shape=(16, 7, 3)
uop_939 = relay.cosh(uop_926.astype('float32')) # shape=(16, 7, 3)
uop_941 = relay.cosh(uop_928.astype('float32')) # shape=(16, 7, 3)
func_631_call = mod.get_global_var('func_631')
func_633_call = mutated_mod.get_global_var('func_633')
call_959 = relay.TupleGetItem(func_631_call(), 0)
call_960 = relay.TupleGetItem(func_633_call(), 0)
output = relay.Tuple([call_854,const_889,call_898,call_913,bop_917,call_924,bop_933,uop_939,call_959,])
output2 = relay.Tuple([call_855,const_889,call_899,call_914,bop_920,call_925,bop_936,uop_941,call_960,])
func_968 = relay.Function([var_916,var_932,], output)
mod['func_968'] = func_968
mod = relay.transform.InferType()(mod)
var_969 = relay.var("var_969", dtype = "float64", shape = (16, 7, 3))#candidate|969|(16, 7, 3)|var|float64
var_970 = relay.var("var_970", dtype = "float32", shape = (16, 7, 3))#candidate|970|(16, 7, 3)|var|float32
output = func_968(var_969,var_970,)
func_971 = relay.Function([var_969,var_970,], output)
mutated_mod['func_971'] = func_971
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1004 = relay.var("var_1004", dtype = "bool", shape = (15,))#candidate|1004|(15,)|var|bool
var_1005 = relay.var("var_1005", dtype = "bool", shape = (15,))#candidate|1005|(15,)|var|bool
bop_1006 = relay.logical_and(var_1004.astype('bool'), relay.reshape(var_1005.astype('bool'), relay.shape_of(var_1004))) # shape=(15,)
output = relay.Tuple([bop_1006,])
output2 = relay.Tuple([bop_1006,])
func_1009 = relay.Function([var_1004,var_1005,], output)
mod['func_1009'] = func_1009
mod = relay.transform.InferType()(mod)
var_1010 = relay.var("var_1010", dtype = "bool", shape = (15,))#candidate|1010|(15,)|var|bool
var_1011 = relay.var("var_1011", dtype = "bool", shape = (15,))#candidate|1011|(15,)|var|bool
output = func_1009(var_1010,var_1011,)
func_1012 = relay.Function([var_1010,var_1011,], output)
mutated_mod['func_1012'] = func_1012
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1014 = relay.var("var_1014", dtype = "int64", shape = (1, 12))#candidate|1014|(1, 12)|var|int64
const_1015 = relay.const([[-5,-10,-9,-5,1,-6,-3,-2,-8,5,-5,10],[-3,-10,5,9,-8,-10,1,4,7,4,1,-10],[6,8,4,-3,-6,-8,-2,2,-5,9,-10,-6],[4,-9,-8,6,2,-7,-4,3,-5,3,-5,9],[9,-6,2,-8,-1,-5,-3,3,5,-4,-2,10],[-6,-5,10,-3,-4,8,10,-6,10,-10,-10,-10],[6,-3,9,-6,-6,9,4,5,-10,6,8,1],[-3,10,-7,-10,4,-9,-9,-9,-5,3,10,-1],[-2,10,-1,7,-3,1,7,3,7,6,9,2],[-4,7,-2,6,-5,3,7,-2,-5,2,-4,-5]], dtype = "int64")#candidate|1015|(10, 12)|const|int64
bop_1016 = relay.left_shift(var_1014.astype('int64'), const_1015.astype('int64')) # shape=(10, 12)
var_1029 = relay.var("var_1029", dtype = "int64", shape = (10, 12))#candidate|1029|(10, 12)|var|int64
bop_1030 = relay.subtract(const_1015.astype('int64'), relay.reshape(var_1029.astype('int64'), relay.shape_of(const_1015))) # shape=(10, 12)
bop_1037 = relay.floor_divide(var_1029.astype('float64'), relay.reshape(bop_1016.astype('float64'), relay.shape_of(var_1029))) # shape=(10, 12)
output = relay.Tuple([bop_1030,bop_1037,])
output2 = relay.Tuple([bop_1030,bop_1037,])
func_1042 = relay.Function([var_1014,var_1029,], output)
mod['func_1042'] = func_1042
mod = relay.transform.InferType()(mod)
mutated_mod['func_1042'] = func_1042
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1042_call = mutated_mod.get_global_var('func_1042')
var_1044 = relay.var("var_1044", dtype = "int64", shape = (1, 12))#candidate|1044|(1, 12)|var|int64
var_1045 = relay.var("var_1045", dtype = "int64", shape = (10, 12))#candidate|1045|(10, 12)|var|int64
call_1043 = func_1042_call(var_1044,var_1045,)
output = call_1043
func_1046 = relay.Function([var_1044,var_1045,], output)
mutated_mod['func_1046'] = func_1046
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1082 = relay.var("var_1082", dtype = "int8", shape = (14, 1, 14))#candidate|1082|(14, 1, 14)|var|int8
const_1083 = relay.const([[[2,-2,10,-5,10,6,5,5,-5,-10,10,-10,-6,6],[5,3,7,-8,9,-7,-7,3,4,-2,2,5,-1,-1],[-9,-8,2,-2,7,9,5,-7,8,4,7,-10,-7,10],[-2,-8,-8,5,-6,-3,7,10,-8,4,-4,-10,6,1],[7,-3,4,1,-6,4,-3,6,8,2,3,-4,6,-2],[-5,8,-6,-4,10,4,10,-1,-4,-3,-5,8,2,-6],[5,-3,-3,-4,-1,1,2,1,-5,6,-9,2,5,3],[4,-8,-9,9,1,9,2,7,-7,-9,-1,2,6,-6],[-7,-10,8,10,-8,-7,4,3,10,-3,8,-9,-7,-1],[-8,5,5,4,3,3,3,-10,-8,5,3,-4,10,-4],[-6,-1,1,10,3,-1,8,-6,6,9,9,1,-4,6],[10,-10,-4,-1,-8,3,6,8,10,6,9,9,1,-1],[6,-8,5,5,8,-6,-8,10,5,-8,1,10,-1,-10]],[[-10,9,6,4,-5,3,-6,2,8,-8,1,9,-1,-3],[-7,-7,2,2,-3,-1,7,4,3,-7,-9,5,2,1],[10,-4,-7,-5,-8,8,3,-10,5,4,-9,10,-8,5],[6,1,-9,6,-7,6,-8,-5,-6,-3,-10,-7,-8,6],[-2,3,-6,-6,9,-5,-3,-2,-5,-1,-3,-2,-1,-1],[-5,-4,7,-7,-4,-9,6,-1,-5,7,-1,3,-3,6],[-3,3,-1,7,6,10,8,3,5,3,7,2,-10,-2],[9,9,3,9,8,-2,1,-4,-3,-5,1,8,1,7],[-4,-8,-9,4,6,-7,-1,-9,-3,5,10,-4,-9,-2],[-4,-6,10,-3,9,-10,-3,1,8,-8,3,-8,-3,-3],[-7,-5,-3,-1,-4,-1,-4,-10,-10,10,-9,10,2,-4],[-10,-9,1,10,-4,-9,7,-3,-3,-7,2,-2,8,4],[9,6,6,-6,9,2,-7,4,-1,5,10,4,1,-6]],[[7,8,-9,-5,10,-1,-9,2,4,-2,-3,-6,2,6],[7,2,-1,1,-5,-4,-2,-3,9,-9,-1,5,-1,-8],[4,-4,-5,6,9,5,-4,8,-5,-6,-6,5,-7,5],[5,1,-1,-10,-3,2,2,5,-8,3,-6,8,-6,-3],[-7,-3,6,-2,10,1,10,5,-5,-9,7,-8,1,-9],[3,9,2,2,7,3,10,1,9,2,-10,9,-3,2],[-5,10,-1,-7,3,9,7,-7,7,-7,-6,2,-9,3],[8,-5,-4,-10,-6,-10,-8,8,10,-2,-3,-3,5,-5],[6,6,3,8,8,4,7,-2,-2,10,-6,-1,1,-6],[-9,1,-8,1,10,-9,-1,2,-5,-5,-7,-9,9,-1],[-3,-7,7,10,-8,-7,2,9,-8,-1,7,10,-3,6],[8,-4,-6,5,-10,-1,8,7,3,-5,-6,4,4,-9],[2,1,-5,-4,8,4,-6,6,-2,-6,-10,-8,7,2]],[[-5,-10,-8,-10,8,2,3,3,1,-3,-6,-10,8,-5],[-1,6,-9,6,7,-10,7,-10,10,-10,-5,-3,-4,9],[2,-6,-2,8,7,1,-9,-9,-5,-9,-2,-3,3,-2],[-1,-8,-7,-4,7,2,-8,-7,4,-5,7,-1,4,8],[9,6,-10,-3,-9,6,2,1,1,-9,6,-3,8,1],[-7,1,10,-8,-4,-7,-9,-10,-1,1,10,6,-3,-10],[2,-5,2,-8,3,-5,8,-1,-5,8,-1,9,-6,-9],[5,-10,-3,-9,-10,9,5,-2,6,-10,4,10,-6,1],[7,2,1,10,-6,-1,7,4,6,5,-1,2,4,3],[2,-1,-3,-10,5,-10,-10,-1,-3,-9,1,10,10,9],[5,-1,5,-5,-6,-5,-6,6,-8,-2,-2,-9,-4,8],[2,-5,9,-5,1,5,6,10,4,9,-5,-4,-2,3],[-3,-10,-6,3,5,10,-9,-5,4,8,-3,2,8,4]],[[-9,7,-8,9,6,-9,-2,-6,1,-8,7,5,-5,-8],[10,-4,-9,-1,10,10,1,-4,8,4,3,6,-9,10],[5,8,9,4,-5,-4,-2,8,4,6,5,-1,1,-6],[8,8,-3,-9,-10,-10,-8,7,-9,-5,10,6,1,-2],[5,-5,-4,-6,-1,9,8,4,4,-5,-10,-7,-2,-4],[10,-5,5,-10,5,8,-3,-7,6,8,-5,-4,-5,-5],[-6,10,-6,3,-7,10,2,-9,-5,-9,-8,-1,-2,-7],[-8,4,3,9,8,-9,-6,-6,9,7,-10,6,-3,-10],[-4,10,1,-3,7,-2,-4,8,3,2,9,-7,4,-8],[10,2,7,3,7,2,8,6,8,5,2,-3,-7,-2],[6,9,-10,-10,-4,-6,-3,-2,-4,-4,-6,4,-8,-9],[-9,-5,-2,1,-5,2,-7,10,-9,-10,3,-3,-3,-5],[6,-6,-6,10,-5,-2,3,2,2,-8,4,-4,-2,-4]],[[-6,7,-4,6,4,-7,-6,-3,-3,-9,-7,-8,4,6],[10,10,2,7,8,5,9,3,1,4,-1,10,9,6],[9,9,5,-5,-5,-2,-6,-9,7,10,-7,1,-9,-5],[9,4,-5,-9,-10,-7,5,-2,-6,-5,10,-2,-10,-5],[4,-9,-1,-4,4,9,6,-1,-7,3,1,-8,-3,-9],[1,-2,-6,-1,4,5,7,-3,4,-7,6,-4,-7,-1],[-1,3,-6,1,8,3,7,1,4,10,-8,8,-6,1],[7,4,-4,-1,6,4,-6,-7,5,-1,-9,-8,10,1],[10,-9,-5,4,4,10,1,-3,1,2,-5,6,5,-3],[9,1,5,6,-5,-9,-10,-6,-1,8,6,-9,-6,-6],[-8,-6,-4,-8,-8,-9,7,-1,-7,7,-5,-1,6,5],[4,5,-2,-8,-9,-5,1,8,-4,-4,-6,-10,-6,6],[-10,6,-2,-5,-1,9,1,6,10,9,-5,-6,4,3]],[[-6,3,-8,1,9,4,-10,1,9,-3,-1,4,5,-8],[5,-10,10,8,-2,10,4,5,-9,5,-2,-8,1,1],[-3,7,-10,-10,10,-8,8,8,-2,-7,-7,-10,1,6],[-1,9,5,5,-1,-6,4,7,-4,3,6,-8,2,-6],[5,6,8,-1,9,6,10,5,8,5,5,-7,-9,5],[-2,-6,-2,2,-8,-1,-8,1,1,-1,7,10,-5,2],[1,-5,10,-1,-4,2,-5,-8,-2,-10,-4,1,-6,-3],[7,4,5,-4,-9,-9,7,4,4,4,3,-8,-6,8],[1,4,9,7,-7,10,5,8,1,-3,-6,-1,-2,5],[10,-1,-5,1,3,7,-1,6,6,-6,-8,2,-6,10],[-1,-6,-8,-4,-2,-4,-4,-4,10,9,-6,3,-5,-5],[-7,1,-9,9,-3,-10,-7,-9,7,-6,-5,-3,1,2],[-2,-9,3,-6,-6,-10,7,6,-4,7,-10,8,-7,8]],[[-8,5,9,-8,-3,-3,-9,-10,10,-4,-10,-6,1,10],[7,-7,-8,9,-6,8,-1,-2,5,-1,-7,1,-4,-2],[3,5,7,8,-4,-3,-4,-2,3,7,6,-2,3,-2],[2,-3,3,-6,-10,8,-1,-4,1,10,-10,1,-7,10],[-8,-6,-4,7,7,-2,2,-2,-8,-8,4,2,-6,-1],[7,2,4,10,-7,9,-2,2,8,7,-10,-6,4,2],[2,5,-5,-9,-5,8,5,5,-8,-3,10,-4,-6,7],[-2,-10,-2,-6,-1,7,3,7,3,-7,-8,-7,10,-1],[4,4,4,-2,10,5,-4,-9,-10,-5,10,-3,-10,-9],[4,7,-5,-5,7,-9,-5,10,9,-4,-8,3,1,-8],[-8,3,6,-9,-1,2,-2,-6,-5,6,-8,4,-1,7],[1,-7,2,-1,10,-1,-6,8,4,-7,-2,7,-5,-9],[-6,-10,-10,7,9,-2,10,9,-8,6,-2,-3,5,-6]],[[10,9,-10,2,8,-5,7,-5,3,-5,-10,3,3,8],[-7,-10,-10,1,-8,1,4,2,-5,10,-10,9,3,5],[9,4,8,2,-3,-9,8,8,7,-5,8,2,-10,1],[4,-9,-9,-10,-9,6,-7,-6,6,9,5,-8,9,4],[-1,-3,2,10,9,10,-2,1,-5,4,-10,-5,-9,9],[-4,-9,9,-2,6,-2,-1,2,-3,3,-9,-9,2,-5],[8,-10,8,-7,-1,-7,8,1,-6,-3,7,4,4,8],[-9,6,-6,8,-6,-4,4,6,-3,-3,2,-7,-3,7],[-6,1,8,1,-6,2,5,-8,7,5,1,-10,-6,-1],[8,4,5,-7,7,-8,10,2,4,-6,-2,-1,-1,-5],[8,3,6,-9,-7,3,-8,-4,9,-4,5,-3,3,1],[7,-8,-2,10,-2,-6,-9,2,7,4,1,-8,9,5],[-7,5,-10,6,6,9,-7,7,-3,-1,-5,8,-6,4]],[[10,-7,-8,9,-5,-10,-5,9,4,-7,-6,-9,7,1],[-4,7,6,10,-6,2,8,-2,-2,4,-10,-3,1,-9],[1,-10,-7,-10,-2,-9,-1,-10,6,1,-3,-4,-6,-9],[-10,3,-10,-6,-9,4,-7,10,5,-9,1,8,5,-7],[8,-7,-8,-10,5,-7,-4,-8,3,1,7,10,-9,2],[-8,-4,6,-7,-4,-3,4,-5,-1,1,10,-6,1,-3],[-2,-4,-3,-1,2,3,4,6,6,-8,4,-2,9,-9],[4,1,-6,-8,-5,-8,-6,4,-5,-8,2,9,-5,-10],[9,-5,1,8,-10,6,5,2,-5,9,-7,-2,-5,6],[-1,4,-4,-2,-4,-4,-7,2,6,-1,-10,-4,1,7],[6,6,10,9,9,-10,-2,-9,-7,-10,5,-6,4,-4],[-6,1,-4,3,-8,-9,6,10,-6,-6,2,-9,-7,-1],[-5,-6,-6,6,5,-6,-4,8,6,7,-10,6,6,6]],[[10,-4,2,9,6,-8,7,-4,10,-10,-1,-8,-7,-3],[-10,6,-8,-7,10,-6,3,9,-9,8,4,-4,4,-10],[-8,4,3,10,-2,-10,4,3,4,-5,5,-2,3,8],[10,8,10,-6,4,4,2,10,-4,3,-7,8,-6,-2],[6,6,-7,-4,4,2,5,-6,2,4,1,1,-10,9],[-1,2,-3,3,-3,-9,-6,-5,-6,9,6,-3,5,-4],[-5,8,4,-5,-10,-6,-5,10,-7,6,7,-1,-1,2],[10,-10,1,-3,1,1,-6,6,4,-5,-2,-4,8,-4],[2,4,3,7,5,-8,-7,8,8,5,10,2,-2,-1],[-4,9,2,-7,1,-9,-5,-3,-6,6,-6,7,-4,10],[-9,-6,7,-8,1,5,1,-8,-1,-9,-1,2,-7,-3],[-9,-6,-3,1,10,-10,5,2,-9,2,-1,-6,7,10],[-8,6,-6,-8,-7,-1,-8,-9,-6,9,-3,-3,4,-7]],[[-1,5,-7,-9,-5,4,-7,10,-5,10,-2,-5,2,9],[1,-6,-10,2,-4,-2,9,1,-10,3,2,7,4,10],[3,-3,5,5,-10,5,-7,3,5,-1,-8,2,-8,3],[2,2,-3,8,-4,-3,9,10,-7,1,8,-6,-8,4],[7,10,3,5,-8,-5,-7,9,-6,-3,2,4,-7,4],[-1,3,-8,-9,6,-10,-8,-7,1,2,-2,2,-10,2],[9,5,4,7,-1,-4,2,-1,9,7,1,9,4,-9],[-8,-4,-9,-6,-8,-7,-1,6,-7,6,-5,5,-2,-1],[-6,3,-3,2,7,-5,2,4,4,-7,-8,-6,3,9],[6,2,-2,9,-3,7,10,2,5,3,7,10,-2,6],[-8,1,-9,-9,9,8,8,9,-9,-9,-1,4,2,-1],[10,2,-4,-1,5,6,7,-5,9,-4,8,-3,-8,5],[10,-6,5,6,8,-4,9,-10,-4,-3,10,-1,3,-7]],[[2,-10,-10,-3,-4,-6,-10,9,2,6,-4,-9,8,-3],[6,3,2,2,-2,6,-6,8,5,-5,-7,-1,2,5],[9,-3,8,5,-1,-1,-2,9,-9,3,-6,-1,9,-2],[-1,3,5,-7,3,-2,-8,7,9,-5,-3,-4,-7,2],[6,-9,2,-9,9,9,-7,-6,1,-3,1,9,-10,-3],[-8,5,10,3,5,9,9,5,7,10,-7,10,-5,-2],[-4,2,-1,-7,8,-2,-1,6,-7,5,2,-5,10,-9],[-6,8,7,1,-4,10,8,4,1,9,-10,-3,10,5],[3,3,-6,-9,-10,2,1,-7,-4,-10,2,4,-4,8],[6,3,9,1,8,10,-2,-3,6,6,2,4,5,-8],[6,-4,1,-2,-5,8,4,-3,-10,-1,-8,-6,-3,4],[10,-9,-10,-7,-10,3,-3,-1,9,-3,-2,-7,3,3],[-8,4,-1,-4,-5,5,-1,5,9,1,1,9,2,8]],[[-1,-1,-7,7,-9,-2,2,-4,-6,-9,4,-5,5,10],[9,-3,8,1,1,5,9,-8,5,-5,8,7,5,-9],[-7,-8,-9,-9,6,-2,8,-9,-5,7,9,-10,-4,-1],[3,9,-10,-6,-1,6,5,-4,-3,9,2,-7,-5,-7],[8,-10,-7,8,-8,-6,1,6,-8,-8,-5,-1,2,5],[-9,-8,10,7,-9,-5,-9,-7,1,-6,-4,-1,3,6],[-4,-5,-3,9,3,9,10,10,-2,-10,8,3,-4,3],[-8,5,9,-4,-4,-6,-8,-10,8,6,5,-5,5,7],[4,-9,8,-6,-5,-9,-3,8,9,-10,8,-9,3,7],[3,9,3,-2,9,8,-10,5,-2,-7,-2,8,-2,-2],[10,-4,3,-10,9,-9,-2,6,-6,7,3,8,10,7],[10,8,-1,3,-6,-2,5,-5,10,10,2,3,1,3],[7,3,1,9,4,6,5,4,3,8,3,-1,-5,-2]]], dtype = "int8")#candidate|1083|(14, 13, 14)|const|int8
bop_1084 = relay.right_shift(var_1082.astype('int8'), const_1083.astype('int8')) # shape=(14, 13, 14)
func_585_call = mod.get_global_var('func_585')
func_586_call = mutated_mod.get_global_var('func_586')
call_1095 = func_585_call()
call_1096 = func_585_call()
output = relay.Tuple([bop_1084,call_1095,])
output2 = relay.Tuple([bop_1084,call_1096,])
func_1101 = relay.Function([var_1082,], output)
mod['func_1101'] = func_1101
mod = relay.transform.InferType()(mod)
var_1102 = relay.var("var_1102", dtype = "int8", shape = (14, 1, 14))#candidate|1102|(14, 1, 14)|var|int8
output = func_1101(var_1102)
func_1103 = relay.Function([var_1102], output)
mutated_mod['func_1103'] = func_1103
mutated_mod = relay.transform.InferType()(mutated_mod)
func_356_call = mod.get_global_var('func_356')
func_358_call = mutated_mod.get_global_var('func_358')
call_1105 = relay.TupleGetItem(func_356_call(), 0)
call_1106 = relay.TupleGetItem(func_358_call(), 0)
func_730_call = mod.get_global_var('func_730')
func_734_call = mutated_mod.get_global_var('func_734')
var_1146 = relay.var("var_1146", dtype = "uint64", shape = (6, 84))#candidate|1146|(6, 84)|var|uint64
call_1145 = relay.TupleGetItem(func_730_call(relay.reshape(var_1146.astype('uint64'), [9, 4, 14]), relay.reshape(var_1146.astype('uint64'), [9, 4, 14]), ), 0)
call_1147 = relay.TupleGetItem(func_734_call(relay.reshape(var_1146.astype('uint64'), [9, 4, 14]), relay.reshape(var_1146.astype('uint64'), [9, 4, 14]), ), 0)
func_1101_call = mod.get_global_var('func_1101')
func_1103_call = mutated_mod.get_global_var('func_1103')
const_1152 = relay.const([-8,-4,-9,-1,8,4,4,2,3,-7,8,-10,9,-10,-7,-5,-5,10,-10,6,1,-9,5,9,5,-4,9,-9,-7,3,-7,1,-9,-7,-5,6,2,-3,9,8,-7,8,-8,4,8,-1,8,-7,4,6,7,-5,7,8,9,9,-10,-2,4,4,-3,-6,-7,-8,7,10,5,-4,5,-6,10,2,10,8,10,-5,7,4,4,-6,-5,4,-4,-10,-2,5,2,10,1,-8,-10,-4,4,-10,7,1,2,1,4,9,8,6,3,-5,-5,-5,-8,1,-4,-2,7,-4,9,-7,9,-8,1,-8,-3,-7,6,3,-1,-7,5,8,1,1,1,2,3,6,6,-6,-7,-7,-7,8,-3,4,4,-10,-6,-6,6,-7,-10,2,-5,4,3,6,2,-9,10,-4,5,-9,4,10,-5,-10,-7,10,-3,5,-8,6,-6,-6,8,1,6,6,7,-10,2,5,2,-5,-4,1,-3,3,-3,6,-10,5,-6,6,7,-6,-7,10,8,4], dtype = "int8")#candidate|1152|(196,)|const|int8
call_1151 = relay.TupleGetItem(func_1101_call(relay.reshape(const_1152.astype('int8'), [14, 1, 14])), 0)
call_1153 = relay.TupleGetItem(func_1103_call(relay.reshape(const_1152.astype('int8'), [14, 1, 14])), 0)
output = relay.Tuple([call_1105,call_1145,var_1146,call_1151,const_1152,])
output2 = relay.Tuple([call_1106,call_1147,var_1146,call_1153,const_1152,])
func_1161 = relay.Function([var_1146,], output)
mod['func_1161'] = func_1161
mod = relay.transform.InferType()(mod)
var_1162 = relay.var("var_1162", dtype = "uint64", shape = (6, 84))#candidate|1162|(6, 84)|var|uint64
output = func_1161(var_1162)
func_1163 = relay.Function([var_1162], output)
mutated_mod['func_1163'] = func_1163
mutated_mod = relay.transform.InferType()(mutated_mod)
func_157_call = mod.get_global_var('func_157')
func_158_call = mutated_mod.get_global_var('func_158')
call_1167 = relay.TupleGetItem(func_157_call(), 2)
call_1168 = relay.TupleGetItem(func_158_call(), 2)
const_1171 = relay.const([-2,1,1,10,3,6,-7,-10,7,-7,-4,8,6,4,10,1,10,-6,5,3,-1,-2,7,5,5,-2,1,-8,9,2,5,9,-5,-6,-5,8,9,2,-3,10,1,8,4,-10,8,-10,7,-8,-6,-5,-3,-7,-5,10,8,-6,4,2,10,10,-5,-2,-10,7,5,7,6,-2,-4,7,-2,2,6,3,7,-5,4,7,8,-1,5,-2,-3,2,-9,-7,3,-9,3,-7,5,-5,9,-2,5,9,1,-10,8,-10,-10,-9,8,2,-10,-4,-10,4,10,-6,9,-6,7,-5,-8,3,5,1,-10,7,-3,-4,1,-4,-6,-7,-8,-7,8,1,8,4,-3,-7,1,2,-6,-8,-6,9,5,-9,8,7,8,4,1,1,4,-9,-1,-8,9,3,2,-1,7,-1,6,3,-7,1,-4,4,-2,2,5,-2,5,2,-7,-4,3,-1,3,-4,-8,8,9,9,-4,7,-8,6,4,3,-10,-8,-4,1,-3,-10,2,-5,10,2,-3,-7,-8,8,-8,-5,-10,3,-6,-1,-1,-1,9,-4,8,-6,7,-5,4,-8,4,3,10,6,-2,4,8,-6,9,-7,-10,-9,-4,-3,-8,3,4,-6,3,10,-8,9,-8,-4,-7,-6,-4,3,1,-6,-6,-1,-4,5,-8,1,9,-6,6,-4,1,-9,6,2,-9,6,-7,6,-3,-7,5,-9,-9,-1,-4,-8,-4,-9,1,8,10,-5,-3,8,-1,-9,2,-3,2,6,9,-1,2,-5,-3,-5,9,2,1,9,3,-10,-7,-8,9,7,9,-2,9,-8,-10,-7,-4,-2,7,9,-1,-8,4,-9,1,2,5,-1,-9,9,1,-8,-7,2,-2,-5,-9,-3,-9,9,-10,10,-3,2], dtype = "int64")#candidate|1171|(336,)|const|int64
bop_1172 = relay.equal(call_1167.astype('bool'), relay.reshape(const_1171.astype('bool'), relay.shape_of(call_1167))) # shape=(336,)
bop_1175 = relay.equal(call_1168.astype('bool'), relay.reshape(const_1171.astype('bool'), relay.shape_of(call_1168))) # shape=(336,)
output = relay.Tuple([bop_1172,])
output2 = relay.Tuple([bop_1175,])
func_1179 = relay.Function([], output)
mod['func_1179'] = func_1179
mod = relay.transform.InferType()(mod)
mutated_mod['func_1179'] = func_1179
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1179_call = mutated_mod.get_global_var('func_1179')
call_1180 = func_1179_call()
output = call_1180
func_1181 = relay.Function([], output)
mutated_mod['func_1181'] = func_1181
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1186 = relay.var("var_1186", dtype = "float32", shape = (2,))#candidate|1186|(2,)|var|float32
uop_1187 = relay.asinh(var_1186.astype('float32')) # shape=(2,)
func_1101_call = mod.get_global_var('func_1101')
func_1103_call = mutated_mod.get_global_var('func_1103')
const_1199 = relay.const([5,-10,7,-8,8,4,1,-2,-10,5,-8,-1,-6,-8,-4,-5,-5,2,7,-2,7,8,-1,-9,9,-7,10,9,-6,-6,-10,-1,-8,-6,3,9,3,10,-2,-7,2,10,-3,2,8,10,-1,-2,2,-5,7,4,3,-5,2,6,7,-10,8,10,9,3,-8,8,-4,-4,-7,10,4,-7,-9,-2,2,9,-3,-6,-6,2,-4,4,3,1,-6,-4,-8,7,-7,-4,-1,8,9,-8,-5,-8,1,8,-10,6,10,8,-2,-5,-6,-9,-1,2,-3,6,-1,6,10,5,-3,-8,2,10,3,-4,-4,7,-10,-7,9,10,2,1,-3,2,9,10,-7,7,6,6,-4,7,-2,-8,2,3,7,-5,-2,6,-8,3,7,-3,3,4,4,5,8,1,3,-3,7,1,-7,-1,5,10,7,-6,-7,-10,-2,1,8,7,5,4,-9,1,5,-3,-7,-2,-8,-5,-9,-8,-1,10,-10,-3,-6,-6,3,3,9,-8,-8,-3,1,5], dtype = "int8")#candidate|1199|(196,)|const|int8
call_1198 = relay.TupleGetItem(func_1101_call(relay.reshape(const_1199.astype('int8'), [14, 1, 14])), 0)
call_1200 = relay.TupleGetItem(func_1103_call(relay.reshape(const_1199.astype('int8'), [14, 1, 14])), 0)
uop_1201 = relay.sinh(call_1198.astype('float32')) # shape=(14, 13, 14)
uop_1203 = relay.sinh(call_1200.astype('float32')) # shape=(14, 13, 14)
uop_1204 = relay.sigmoid(uop_1201.astype('float32')) # shape=(14, 13, 14)
uop_1206 = relay.sigmoid(uop_1203.astype('float32')) # shape=(14, 13, 14)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
var_1210 = relay.var("var_1210", dtype = "int64", shape = (336,))#candidate|1210|(336,)|var|int64
call_1209 = relay.TupleGetItem(func_237_call(relay.reshape(var_1210.astype('int64'), [336,])), 0)
call_1211 = relay.TupleGetItem(func_239_call(relay.reshape(var_1210.astype('int64'), [336,])), 0)
uop_1212 = relay.cosh(uop_1204.astype('float32')) # shape=(14, 13, 14)
uop_1214 = relay.cosh(uop_1206.astype('float32')) # shape=(14, 13, 14)
var_1215 = relay.var("var_1215", dtype = "float32", shape = (14, 13, 14))#candidate|1215|(14, 13, 14)|var|float32
bop_1216 = relay.subtract(uop_1212.astype('float64'), relay.reshape(var_1215.astype('float64'), relay.shape_of(uop_1212))) # shape=(14, 13, 14)
bop_1219 = relay.subtract(uop_1214.astype('float64'), relay.reshape(var_1215.astype('float64'), relay.shape_of(uop_1214))) # shape=(14, 13, 14)
bop_1220 = relay.logical_and(uop_1212.astype('bool'), relay.reshape(bop_1216.astype('bool'), relay.shape_of(uop_1212))) # shape=(14, 13, 14)
bop_1223 = relay.logical_and(uop_1214.astype('bool'), relay.reshape(bop_1219.astype('bool'), relay.shape_of(uop_1214))) # shape=(14, 13, 14)
uop_1224 = relay.log10(uop_1212.astype('float64')) # shape=(14, 13, 14)
uop_1226 = relay.log10(uop_1214.astype('float64')) # shape=(14, 13, 14)
bop_1228 = relay.greater_equal(uop_1224.astype('bool'), relay.reshape(uop_1212.astype('bool'), relay.shape_of(uop_1224))) # shape=(14, 13, 14)
bop_1231 = relay.greater_equal(uop_1226.astype('bool'), relay.reshape(uop_1214.astype('bool'), relay.shape_of(uop_1226))) # shape=(14, 13, 14)
uop_1232 = relay.erf(uop_1224.astype('float32')) # shape=(14, 13, 14)
uop_1234 = relay.erf(uop_1226.astype('float32')) # shape=(14, 13, 14)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_1235 = func_193_call()
call_1236 = func_193_call()
output = relay.Tuple([uop_1187,const_1199,call_1209,var_1210,bop_1220,bop_1228,uop_1232,call_1235,])
output2 = relay.Tuple([uop_1187,const_1199,call_1211,var_1210,bop_1223,bop_1231,uop_1234,call_1236,])
func_1240 = relay.Function([var_1186,var_1210,var_1215,], output)
mod['func_1240'] = func_1240
mod = relay.transform.InferType()(mod)
var_1241 = relay.var("var_1241", dtype = "float32", shape = (2,))#candidate|1241|(2,)|var|float32
var_1242 = relay.var("var_1242", dtype = "int64", shape = (336,))#candidate|1242|(336,)|var|int64
var_1243 = relay.var("var_1243", dtype = "float32", shape = (14, 13, 14))#candidate|1243|(14, 13, 14)|var|float32
output = func_1240(var_1241,var_1242,var_1243,)
func_1244 = relay.Function([var_1241,var_1242,var_1243,], output)
mutated_mod['func_1244'] = func_1244
mutated_mod = relay.transform.InferType()(mutated_mod)
func_356_call = mod.get_global_var('func_356')
func_358_call = mutated_mod.get_global_var('func_358')
call_1281 = relay.TupleGetItem(func_356_call(), 0)
call_1282 = relay.TupleGetItem(func_358_call(), 0)
var_1299 = relay.var("var_1299", dtype = "uint8", shape = (16, 7, 3))#candidate|1299|(16, 7, 3)|var|uint8
bop_1300 = relay.add(call_1281.astype('int32'), relay.reshape(var_1299.astype('int32'), relay.shape_of(call_1281))) # shape=(16, 7, 3)
bop_1303 = relay.add(call_1282.astype('int32'), relay.reshape(var_1299.astype('int32'), relay.shape_of(call_1282))) # shape=(16, 7, 3)
output = bop_1300
output2 = bop_1303
func_1316 = relay.Function([var_1299,], output)
mod['func_1316'] = func_1316
mod = relay.transform.InferType()(mod)
var_1317 = relay.var("var_1317", dtype = "uint8", shape = (16, 7, 3))#candidate|1317|(16, 7, 3)|var|uint8
output = func_1316(var_1317)
func_1318 = relay.Function([var_1317], output)
mutated_mod['func_1318'] = func_1318
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1340 = relay.var("var_1340", dtype = "float32", shape = (7, 5))#candidate|1340|(7, 5)|var|float32
const_1341 = relay.const([[-7.500507,5.447888,-8.466675,7.253118,-0.454026],[-9.967460,7.429923,0.672343,8.743023,-8.971140],[0.860375,-7.030781,3.984448,-2.945199,4.546197],[-4.362857,9.545231,-0.445627,7.067597,-4.753907],[-9.894005,9.718762,3.424938,-2.147229,4.806728],[-8.336014,-9.496619,-1.344430,-6.275037,-7.487422],[-1.558620,3.577054,-7.571380,6.450316,-0.189340]], dtype = "float32")#candidate|1341|(7, 5)|const|float32
bop_1342 = relay.mod(var_1340.astype('float32'), relay.reshape(const_1341.astype('float32'), relay.shape_of(var_1340))) # shape=(7, 5)
output = bop_1342
output2 = bop_1342
func_1348 = relay.Function([var_1340,], output)
mod['func_1348'] = func_1348
mod = relay.transform.InferType()(mod)
mutated_mod['func_1348'] = func_1348
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1349 = relay.var("var_1349", dtype = "float32", shape = (7, 5))#candidate|1349|(7, 5)|var|float32
func_1348_call = mutated_mod.get_global_var('func_1348')
call_1350 = func_1348_call(var_1349)
output = call_1350
func_1351 = relay.Function([var_1349], output)
mutated_mod['func_1351'] = func_1351
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1353 = relay.var("var_1353", dtype = "float32", shape = (9, 3, 1))#candidate|1353|(9, 3, 1)|var|float32
var_1354 = relay.var("var_1354", dtype = "float32", shape = (9, 3, 15))#candidate|1354|(9, 3, 15)|var|float32
bop_1355 = relay.mod(var_1353.astype('float32'), var_1354.astype('float32')) # shape=(9, 3, 15)
bop_1362 = relay.divide(var_1354.astype('float64'), relay.reshape(bop_1355.astype('float64'), relay.shape_of(var_1354))) # shape=(9, 3, 15)
uop_1370 = relay.acosh(var_1354.astype('float32')) # shape=(9, 3, 15)
output = relay.Tuple([bop_1362,uop_1370,])
output2 = relay.Tuple([bop_1362,uop_1370,])
func_1375 = relay.Function([var_1353,var_1354,], output)
mod['func_1375'] = func_1375
mod = relay.transform.InferType()(mod)
mutated_mod['func_1375'] = func_1375
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1375_call = mutated_mod.get_global_var('func_1375')
var_1377 = relay.var("var_1377", dtype = "float32", shape = (9, 3, 1))#candidate|1377|(9, 3, 1)|var|float32
var_1378 = relay.var("var_1378", dtype = "float32", shape = (9, 3, 15))#candidate|1378|(9, 3, 15)|var|float32
call_1376 = func_1375_call(var_1377,var_1378,)
output = call_1376
func_1379 = relay.Function([var_1377,var_1378,], output)
mutated_mod['func_1379'] = func_1379
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1393 = relay.var("var_1393", dtype = "uint32", shape = (2, 8, 5))#candidate|1393|(2, 8, 5)|var|uint32
var_1394 = relay.var("var_1394", dtype = "uint32", shape = (2, 8, 5))#candidate|1394|(2, 8, 5)|var|uint32
bop_1395 = relay.bitwise_xor(var_1393.astype('uint32'), relay.reshape(var_1394.astype('uint32'), relay.shape_of(var_1393))) # shape=(2, 8, 5)
bop_1402 = relay.add(bop_1395.astype('uint16'), relay.reshape(var_1394.astype('uint16'), relay.shape_of(bop_1395))) # shape=(2, 8, 5)
func_664_call = mod.get_global_var('func_664')
func_666_call = mutated_mod.get_global_var('func_666')
const_1413 = relay.const([-1.434300,-7.232573,4.076899,-9.181955,2.739627,-1.229942,4.714545,-7.761061,-7.782330,3.741074,-1.493580,9.541496,9.181571,-8.000955,-6.468012,-6.283499,-8.040204,-1.144778,7.895575,7.633600,-6.933135,1.339714,2.054376,9.347423,-5.971240,1.582061,1.384494,1.148404,3.168118,7.630749,-8.916836,-9.583618,8.042599,-2.629750,-4.538859,3.137518,1.734960,-8.440562,8.540600,3.332123,-2.393821,-6.933552,0.841113,-5.459514,-5.424999,7.162478,-8.628657,9.338398,-4.008861,-5.032389,-3.494596,6.063650,1.250688,-7.493925,0.449924,-6.551690,-8.088099,9.291192,-5.649299,-5.499061,-3.200457,5.011327,-6.518067,-0.573312,8.708766,1.150061,-2.667056,2.287453,-5.793979,-6.179950,2.952503,8.471872,-9.351409,-9.629299,-5.319807,8.693516,-5.786605,6.661921,1.349700,7.494375,2.580687,-0.735327,0.727506,6.651217,6.228280,0.025294,-1.953368,0.278695,-1.761437,1.167783,-4.205098,7.851515,-5.769073,6.801431,-1.528950,-4.372362,7.920578,8.635289,-0.393908,-6.569746,0.773740,0.866377,-0.614932,-5.376682,-4.293258,-2.945824,-4.027500,4.887053,-4.911401,-0.908291,1.001873,-2.919656,-9.569218,8.088406,-5.855134,-7.932133,6.964839,5.739903,-6.299736,-2.954026,-7.182283,-3.644792,-2.985265,3.222962,-1.241566,7.884197,-8.054889,0.832824,-2.492672,7.948874,-4.904849,-5.931033,8.419543,-4.771022,5.717748,-4.252416,6.819668,-9.138893,-6.773810,-2.409895,5.734545,-8.761603,-0.020423,-1.457828,-7.065233,3.413402,-1.869197,1.591133,-2.725134,5.617416,-3.772564,3.069469,-4.460693,-1.503660,9.867855,-8.781908,-2.023489,-0.196293,-6.748441,9.068565,-1.214738,9.163925,-6.920255,4.807164,-2.031979,-2.043696,-1.803687,-1.449271,8.695047,-6.448230,-3.854127,-0.020702,1.659462,8.764310,8.187279,6.164995,2.201824,-2.341522,0.918236,-2.968175,3.986953,-4.346693,-7.272870,-9.863210,4.927117,-2.804566,2.787069,0.343264,1.535768,-0.811357,4.708739,1.908285,4.930483,6.800248,1.860414], dtype = "float32")#candidate|1413|(195,)|const|float32
call_1412 = relay.TupleGetItem(func_664_call(relay.reshape(const_1413.astype('float32'), [195,])), 1)
call_1414 = relay.TupleGetItem(func_666_call(relay.reshape(const_1413.astype('float32'), [195,])), 1)
bop_1424 = relay.right_shift(var_1393.astype('uint16'), relay.reshape(bop_1395.astype('uint16'), relay.shape_of(var_1393))) # shape=(2, 8, 5)
uop_1428 = relay.atan(var_1394.astype('float64')) # shape=(2, 8, 5)
bop_1434 = relay.maximum(uop_1428.astype('float64'), relay.reshape(bop_1395.astype('float64'), relay.shape_of(uop_1428))) # shape=(2, 8, 5)
bop_1438 = relay.multiply(uop_1428.astype('int64'), relay.reshape(bop_1434.astype('int64'), relay.shape_of(uop_1428))) # shape=(2, 8, 5)
uop_1441 = relay.sin(bop_1434.astype('float64')) # shape=(2, 8, 5)
bop_1446 = relay.greater(uop_1441.astype('bool'), relay.reshape(bop_1438.astype('bool'), relay.shape_of(uop_1441))) # shape=(2, 8, 5)
var_1452 = relay.var("var_1452", dtype = "float64", shape = (2, 8, 5))#candidate|1452|(2, 8, 5)|var|float64
bop_1453 = relay.divide(uop_1428.astype('float64'), relay.reshape(var_1452.astype('float64'), relay.shape_of(uop_1428))) # shape=(2, 8, 5)
uop_1456 = relay.asinh(bop_1446.astype('float64')) # shape=(2, 8, 5)
output = relay.Tuple([bop_1402,call_1412,const_1413,bop_1424,bop_1453,uop_1456,])
output2 = relay.Tuple([bop_1402,call_1414,const_1413,bop_1424,bop_1453,uop_1456,])
func_1461 = relay.Function([var_1393,var_1394,var_1452,], output)
mod['func_1461'] = func_1461
mod = relay.transform.InferType()(mod)
var_1462 = relay.var("var_1462", dtype = "uint32", shape = (2, 8, 5))#candidate|1462|(2, 8, 5)|var|uint32
var_1463 = relay.var("var_1463", dtype = "uint32", shape = (2, 8, 5))#candidate|1463|(2, 8, 5)|var|uint32
var_1464 = relay.var("var_1464", dtype = "float64", shape = (2, 8, 5))#candidate|1464|(2, 8, 5)|var|float64
output = func_1461(var_1462,var_1463,var_1464,)
func_1465 = relay.Function([var_1462,var_1463,var_1464,], output)
mutated_mod['func_1465'] = func_1465
mutated_mod = relay.transform.InferType()(mutated_mod)
func_397_call = mod.get_global_var('func_397')
func_399_call = mutated_mod.get_global_var('func_399')
call_1481 = relay.TupleGetItem(func_397_call(), 0)
call_1482 = relay.TupleGetItem(func_399_call(), 0)
output = relay.Tuple([call_1481,])
output2 = relay.Tuple([call_1482,])
func_1490 = relay.Function([], output)
mod['func_1490'] = func_1490
mod = relay.transform.InferType()(mod)
output = func_1490()
func_1491 = relay.Function([], output)
mutated_mod['func_1491'] = func_1491
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1501 = relay.const([[[4.284933,9.572663,-4.952715,-7.285958,8.155479,5.741583,7.650105,3.773739,-2.237973,-0.523426],[8.657158,3.890307,5.477186,-7.605431,-2.669417,1.030315,3.213022,-1.136464,6.092326,7.798512],[-7.029631,3.604115,1.651550,7.769213,-6.785838,-7.755331,-9.979484,-9.215281,1.886536,0.921019],[6.401611,4.868887,7.289182,-6.691174,0.865791,9.231290,-2.360264,9.066812,4.523607,8.139030],[-1.511616,-9.945872,1.892843,8.094439,3.560057,-0.829431,8.807675,1.123489,9.351658,-7.515269],[7.164969,-1.735877,-4.413782,-6.008505,-7.738387,4.552020,0.370495,3.243829,3.305316,9.612141]],[[5.643499,3.754575,-3.014216,-8.499035,-4.946153,-5.306698,-8.987428,3.096821,-8.459215,1.165936],[2.842665,8.081625,-1.891165,0.212143,2.513407,-8.133479,-9.460018,-0.572426,-1.383366,-2.396448],[4.438470,7.635198,0.037812,-5.745905,-8.777702,3.675841,4.191301,-3.449416,3.540049,6.484741],[-9.898158,-9.186650,5.986742,7.009098,3.577976,-3.766116,-4.655893,-6.123191,-3.129653,7.907482],[-1.029116,-9.214016,-3.009904,5.252894,-0.269329,-3.571453,-7.986672,7.009504,6.642093,-6.177256],[-2.224033,7.647939,-7.894368,-7.325366,6.221433,3.019461,-4.439298,4.028212,-4.967418,4.421555]],[[2.901863,7.202913,-0.461032,4.869298,-2.444146,-9.238401,-3.963717,0.741724,8.630318,8.999175],[-7.217422,8.268197,-9.045124,3.306914,0.304124,4.655831,-2.085442,2.538193,4.363491,0.337327],[-7.586933,-3.315101,7.133974,6.647432,4.249630,6.521075,6.793321,4.969532,4.265930,-5.271037],[2.351456,7.755075,-0.137415,-2.310123,0.341032,5.587935,6.654405,-4.164526,-6.715534,1.563918],[-1.379243,6.941630,8.240623,8.830899,7.798848,-4.823833,9.279067,-5.556426,-5.980985,7.073640],[-8.441746,2.501890,-4.532031,2.031515,2.627058,2.376831,9.093900,8.745241,7.970496,-1.272857]]], dtype = "float64")#candidate|1501|(3, 6, 10)|const|float64
var_1502 = relay.var("var_1502", dtype = "float64", shape = (3, 6, 10))#candidate|1502|(3, 6, 10)|var|float64
bop_1503 = relay.add(const_1501.astype('float64'), relay.reshape(var_1502.astype('float64'), relay.shape_of(const_1501))) # shape=(3, 6, 10)
bop_1511 = relay.floor_mod(const_1501.astype('float64'), relay.reshape(var_1502.astype('float64'), relay.shape_of(const_1501))) # shape=(3, 6, 10)
output = relay.Tuple([bop_1503,bop_1511,])
output2 = relay.Tuple([bop_1503,bop_1511,])
func_1514 = relay.Function([var_1502,], output)
mod['func_1514'] = func_1514
mod = relay.transform.InferType()(mod)
var_1515 = relay.var("var_1515", dtype = "float64", shape = (3, 6, 10))#candidate|1515|(3, 6, 10)|var|float64
output = func_1514(var_1515)
func_1516 = relay.Function([var_1515], output)
mutated_mod['func_1516'] = func_1516
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1535 = relay.var("var_1535", dtype = "float64", shape = (1, 1))#candidate|1535|(1, 1)|var|float64
uop_1536 = relay.sqrt(var_1535.astype('float64')) # shape=(1, 1)
uop_1540 = relay.acosh(uop_1536.astype('float64')) # shape=(1, 1)
uop_1548 = relay.sinh(var_1535.astype('float32')) # shape=(1, 1)
var_1551 = relay.var("var_1551", dtype = "float64", shape = (10, 16))#candidate|1551|(10, 16)|var|float64
bop_1552 = relay.subtract(uop_1540.astype('float32'), var_1551.astype('float32')) # shape=(10, 16)
output = relay.Tuple([uop_1548,bop_1552,])
output2 = relay.Tuple([uop_1548,bop_1552,])
func_1559 = relay.Function([var_1535,var_1551,], output)
mod['func_1559'] = func_1559
mod = relay.transform.InferType()(mod)
mutated_mod['func_1559'] = func_1559
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1559_call = mutated_mod.get_global_var('func_1559')
var_1561 = relay.var("var_1561", dtype = "float64", shape = (1, 1))#candidate|1561|(1, 1)|var|float64
var_1562 = relay.var("var_1562", dtype = "float64", shape = (10, 16))#candidate|1562|(10, 16)|var|float64
call_1560 = func_1559_call(var_1561,var_1562,)
output = call_1560
func_1563 = relay.Function([var_1561,var_1562,], output)
mutated_mod['func_1563'] = func_1563
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1576 = relay.TupleGetItem(func_695_call(), 0)
call_1577 = relay.TupleGetItem(func_696_call(), 0)
uop_1587 = relay.sigmoid(call_1576.astype('float32')) # shape=(336,)
uop_1589 = relay.sigmoid(call_1577.astype('float32')) # shape=(336,)
func_356_call = mod.get_global_var('func_356')
func_358_call = mutated_mod.get_global_var('func_358')
call_1593 = relay.TupleGetItem(func_356_call(), 0)
call_1594 = relay.TupleGetItem(func_358_call(), 0)
bop_1599 = relay.bitwise_xor(uop_1587.astype('int16'), relay.reshape(call_1576.astype('int16'), relay.shape_of(uop_1587))) # shape=(336,)
bop_1602 = relay.bitwise_xor(uop_1589.astype('int16'), relay.reshape(call_1577.astype('int16'), relay.shape_of(uop_1589))) # shape=(336,)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_1603 = func_193_call()
call_1604 = func_193_call()
func_1490_call = mod.get_global_var('func_1490')
func_1491_call = mutated_mod.get_global_var('func_1491')
call_1612 = relay.TupleGetItem(func_1490_call(), 0)
call_1613 = relay.TupleGetItem(func_1491_call(), 0)
uop_1614 = relay.log10(bop_1599.astype('float64')) # shape=(336,)
uop_1616 = relay.log10(bop_1602.astype('float64')) # shape=(336,)
uop_1623 = relay.erf(uop_1587.astype('float32')) # shape=(336,)
uop_1625 = relay.erf(uop_1589.astype('float32')) # shape=(336,)
bop_1626 = relay.less_equal(uop_1614.astype('bool'), relay.reshape(bop_1599.astype('bool'), relay.shape_of(uop_1614))) # shape=(336,)
bop_1629 = relay.less_equal(uop_1616.astype('bool'), relay.reshape(bop_1602.astype('bool'), relay.shape_of(uop_1616))) # shape=(336,)
bop_1634 = relay.bitwise_and(bop_1599.astype('int8'), relay.reshape(bop_1626.astype('int8'), relay.shape_of(bop_1599))) # shape=(336,)
bop_1637 = relay.bitwise_and(bop_1602.astype('int8'), relay.reshape(bop_1629.astype('int8'), relay.shape_of(bop_1602))) # shape=(336,)
uop_1638 = relay.atan(bop_1599.astype('float32')) # shape=(336,)
uop_1640 = relay.atan(bop_1602.astype('float32')) # shape=(336,)
func_1240_call = mod.get_global_var('func_1240')
func_1244_call = mutated_mod.get_global_var('func_1244')
const_1648 = relay.const([[1.405295,8.897520]], dtype = "float32")#candidate|1648|(1, 2)|const|float32
const_1649 = relay.const([-8.301283,1.795678,-4.342701,-2.606870,-6.697416,4.923938,-8.876626,5.818262,-6.352570,-2.483914,0.791482,4.680503,1.547165,5.496058,-2.392674,-8.080690,-1.565459,3.742729,-6.362542,4.561946,9.218202,-6.923776,-5.566511,-8.162459,1.852369,-1.348375,-1.158296,-1.672503,7.123762,-7.911530,6.694512,-0.231145,7.920449,-6.424624,9.833294,-2.676501,-6.887412,-6.401564,-5.872789,-2.894792,-0.377461,-6.874904,-8.756219,-2.962449,0.891109,3.532536,-5.788793,-2.585852,-9.590271,1.335906,-6.796388,-1.564771,8.343048,-8.655196,6.267920,2.068403,5.972144,-5.483240,-4.879530,6.740863,-6.313466,8.910656,-5.894449,-3.606340,-6.216097,6.636336,2.075693,-3.775611,-0.774055,-9.515821,7.329897,8.678872,-5.210665,-3.473818,6.488011,-4.631609,2.115020,-4.029719,1.786910,-7.800025,-3.763438,-1.821541,1.303189,-3.104794,9.317999,-4.526559,4.331408,-8.401088,2.821144,-5.136310,-7.691430,-6.411114,-6.800524,-5.441443,6.286307,-1.790330,-5.628273,6.942652,1.344030,2.740460,5.808333,6.393940,-3.526488,-2.977510,-3.870296,-6.561057,7.920419,-2.922904,5.799754,7.176723,5.416288,-9.226756,-6.621651,-9.220111,-6.634801,1.677703,6.573445,-6.789807,-9.203884,4.623442,7.787881,2.771313,0.127648,9.518018,-2.761996,8.954144,-3.363264,1.294855,-1.288707,-5.162717,-8.234677,-6.471035,-0.635810,3.941545,5.461049,9.973303,-4.806287,5.282938,5.939386,-1.666397,1.760808,3.495601,1.486054,-0.783296,-0.649809,-8.015701,-9.051928,-6.214505,0.358972,5.609734,9.681973,-6.098056,-6.002151,3.142053,-9.959026,0.317069,4.214756,7.548552,-1.098729,-3.984217,-7.514142,3.089067,1.277590,2.439333,9.642163,4.073271,4.680051,-2.038831,5.762372,1.537590,7.668087,5.216178,-3.976700,9.037064,7.814083,2.900979,0.459557,-6.397712,3.897118,5.770630,8.758778,6.864266,-2.569589,-0.665996,2.279002,1.477734,-7.676598,2.062440,8.249444,3.806436,1.943409,-8.819436,0.355085,4.564573,-9.126553,7.098329,-8.535119,1.765117,-9.899248,-3.823820,3.385696,-9.982453,0.163934,-1.212394,2.194092,5.954740,7.243333,6.369790,-4.229104,2.657723,0.648481,-5.411975,2.888916,0.752499,1.692647,0.368567,9.996596,7.244561,3.443048,-2.410131,3.823663,0.804327,3.648590,-3.844149,-5.558435,0.863736,-4.205581,8.502299,7.078966,-2.651035,-7.197169,3.757312,7.063286,-9.959587,9.561745,2.784492,-1.868133,2.942133,4.982625,-7.201948,2.288311,-5.196651,2.555790,6.802833,-1.797847,-0.288645,8.276050,-4.733005,-1.537640,1.952765,9.891872,-3.452438,-7.048443,-3.603286,0.271901,3.461580,-0.240396,6.877376,3.915198,-8.354310,-1.983748,-2.519544,-1.243324,-9.324870,-7.633930,-3.820040,0.432523,5.531854,-6.405463,2.181003,-4.894193,-9.151435,-2.156167,0.834134,-3.392239,2.071646,-6.650866,-4.756424,5.616400,3.246880,5.871226,7.110026,-2.054196,-1.151844,-0.581006,4.645136,-7.636307,6.869498,1.231779,-2.203938,-9.906521,-6.704691,2.953439,-7.886683,0.069067,-5.492620,2.448404,7.666964,8.523181,0.178880,-8.744326,-4.484558,-6.149827,7.026888,-5.626071,-1.790645,-7.864257,3.108582,9.351104,-2.453323,9.230448,0.569981,-7.796291,6.942974,-5.278390,-4.696916,1.838365,5.213262,-5.265889,3.243660,-8.157704,-7.784599,-7.518679,2.936444,3.436585,-6.325035,-9.940836,6.527437,2.225451,-8.656954,7.836814,-8.953900,-4.615728,6.928100,-9.342752,2.373969,-6.459473,7.082879,-4.040710,-9.015352,-5.516032,-7.057781,9.172816,-2.996198,-3.753636,-3.264963,9.052006,-1.331238,0.118628,-7.185235,-1.543699,-9.459938,6.431410,0.897613,-4.676753,2.261187,9.008158,-1.585839,-7.137025,5.090655,5.327642,-1.110019,-5.506751,-0.322329,-9.705343,7.049152,3.735518,1.301844,-4.727898,-7.213255,1.051039,9.647955,2.502969,9.323287,-3.239168,9.295406,-4.046875,-3.007491,-1.189285,-1.576737,-5.797289,-0.426702,-7.667037,-4.539148,-9.550448,9.950110,4.247812,-2.886240,-4.560128,-5.053358,-7.691637,2.894416,7.923634,2.585902,-7.030206,8.495195,-3.136555,-6.765508,4.485789,3.129659,7.316482,8.237198,1.821075,-6.624855,6.225602,7.286270,-1.588245,6.115589,-1.335828,0.171736,-8.031277,9.508511,-2.956018,2.554390,8.938800,-4.888958,-0.371769,6.019340,8.565540,8.818720,-2.064075,0.657197,1.287778,-9.683547,1.819691,8.817790,-5.524352,-0.166519,-6.166401,8.002827,-9.722425,7.418841,3.816700,5.668693,-8.578839,-2.301371,2.729184,2.787757,-0.488433,-3.883640,1.215765,1.614463,2.320192,6.028062,-9.138361,7.720414,-5.679675,0.401671,6.385777,-6.548407,9.077763,-4.003198,-8.147771,-5.017834,4.571274,5.212881,-1.221682,9.578163,-9.511457,9.248625,5.786294,-8.134251,9.079993,5.068386,-0.596229,-7.131727,-1.593827,-8.099470,-9.336551,2.208094,7.771575,4.824218,-4.365973,5.923614,9.305913,7.565236,5.020210,-2.543107,5.029689,7.541988,6.933467,9.307733,-9.584370,2.837452,-8.044579,3.346541,4.007841,1.542966,0.794024,-2.751614,-3.777640,8.993998,7.991465,-4.173649,7.426984,-8.125929,-9.959525,2.083734,7.573895,6.439357,6.562775,-7.813863,5.567167,-8.889443,-1.042285,3.714271,-9.333227,4.414559,-3.187607,-1.796994,3.300788,0.516851,5.595272,9.118597,2.439943,-6.668998,8.156122,4.223637,7.037711,5.740341,-7.839818,-5.131672,0.759183,2.375161,6.605241,8.872515,-2.020551,-4.607155,1.331578,3.689983,3.560902,-0.427113,-4.770637,1.062422,7.159038,4.678709,-6.668585,2.465546,4.951423,7.817754,-6.485383,-6.677323,3.143490,-9.067648,-4.339012,0.335058,-9.058622,-7.690039,2.860436,-5.841677,-3.754887,3.767515,5.350069,-4.253233,4.751320,3.794753,-8.232011,-0.951028,0.758855,1.627193,-8.201871,3.190229,-1.305261,5.145584,8.695353,-5.352553,-5.435909,-1.127393,-6.520985,-9.368533,1.052232,-1.570040,1.468836,8.129078,5.670029,-6.569087,-7.042574,-0.416373,8.010644,-4.160647,2.299608,-9.034037,-6.910973,8.914098,0.179744,-9.679017,-8.705119,8.666208,-3.452042,-3.445784,3.972673,8.048764,-7.374656,-2.436729,-2.885322,-0.760172,3.116302,8.482328,-7.113900,0.368508,-4.132573,-0.193053,-8.435578,9.560657,-8.119413,-5.269969,-7.403945,7.929131,-6.455592,6.030470,2.639732,-3.350163,-7.616929,-8.706256,1.588893,3.904750,-7.121780,-7.201432,7.592654,7.369988,5.950910,0.914962,4.884488,-2.403552,-0.643863,-1.844135,5.485991,-6.305366,-6.971925,-6.876621,-1.725522,-9.769876,3.092495,9.709801,-7.803683,4.146616,8.370874,9.978003,6.948196,-8.290669,7.093548,7.485393,4.830499,4.466027,4.858715,8.830083,8.663554,-5.385776,-6.804157,4.973466,-1.659628,-7.181440,6.503470,-0.304464,7.294109,-4.601459,-6.354103,-7.829153,1.966237,7.003656,2.467679,4.450465,-1.192081,-4.027123,7.129892,1.912221,-4.669652,8.417288,-3.717115,-5.081173,-6.921795,-4.745608,9.510058,8.682584,-3.468933,7.897543,4.096103,-0.834694,-1.983749,3.088596,-5.526450,2.444238,-3.418272,-1.525529,-1.931657,8.919607,0.473889,4.065715,-9.901695,-8.762410,-1.462576,-5.349233,-2.235922,0.064399,7.252731,-4.946795,-1.306734,-1.187335,1.717240,4.072718,-2.232507,-5.149237,-1.731427,-7.944133,-3.514018,-8.391640,-5.003945,5.295400,0.937399,5.596252,5.946980,1.389642,1.807577,-1.000072,-6.693202,-6.497456,-5.855245,7.110644,5.615438,3.959562,-3.642745,-0.486597,2.752447,0.759571,-5.966268,0.829057,-3.765958,7.193832,1.660741,-2.381558,7.498731,-7.087858,-1.060394,0.578000,6.818347,4.902770,-9.096641,-1.698852,0.224002,-0.592357,-8.195762,8.299437,-8.783772,-0.687834,7.351768,-4.884277,1.219469,-4.759776,-6.024845,9.310878,9.053159,0.248074,-9.035108,3.750525,5.786294,-7.280670,3.045032,-5.879932,-1.252084,3.718478,-4.827413,3.133510,-4.803917,6.679920,0.798146,-9.429128,1.582475,-4.848804,8.071473,-3.301792,9.570868,-8.669546,-8.414339,1.056121,3.427453,-0.137829,-4.446246,-7.965132,-3.807126,6.406648,-8.747625,-8.078990,5.734850,-5.199617,-9.403342,2.994526,2.170640,-9.275549,-7.337418,-7.736582,2.790363,2.491875,-4.507195,-0.345912,7.808654,-1.673598,-0.832347,-0.166086,8.027460,-7.593140,7.526424,-5.764096,-4.103764,-4.981676,-6.180242,-0.997800,-9.157168,9.532064,3.788455,4.485053,2.238385,6.318726,-7.969399,9.193153,-7.793376,8.218468,9.046445,-9.134634,-0.017322,-4.198658,1.881672,-9.086299,7.498423,5.199724,-9.085794,3.187033,-4.169197,9.928853,-9.023794,5.750610,-0.104986,-2.872827,-5.628929,3.501350,-0.162123,6.955828,7.546144,6.560077,9.470174,8.893424,3.156997,6.230986,9.301036,-7.469587,8.301084,-5.080627,-8.308853,2.653967,6.450836,7.044807,-5.931658,-3.976199,4.319770,-6.913487,-6.065511,-1.615516,8.790059,4.922030,-2.187329,-0.972611,3.659320,4.888667,-8.301257,-1.986012,-0.242397,8.256868,-1.969390,-4.948005,2.835655,-7.907628,-3.919869,2.552774,-6.861963,0.109965,-5.945087,-1.015591,-8.648211,3.789082,-7.267092,5.771710,1.281789,-7.440518,-6.012244,3.542680,6.670776,2.658117,-1.819148,2.674959,6.844648,8.345938,3.691365,-3.100808,-6.770104,-4.582388,7.609019,3.278732,-8.688678,5.960918,-9.676266,-5.224292,-3.029108,2.836117,-7.518954,-2.068527,5.274586,7.292500,7.923371,8.060454,7.672506,3.261712,4.652726,7.807903,-2.443411,-7.115471,-6.423791,-3.322390,-9.781880,6.772943,9.051825,-5.972533,-9.924255,1.208599,8.745725,3.298164,-6.364977,-0.270117,9.207005,7.492574,-7.579212,-9.281071,-0.358668,9.449817,-0.593463,8.672541,-9.618853,7.157648,6.290636,0.137517,5.235940,9.102016,4.598308,4.473328,7.233664,-7.724269,-6.626953,-8.808524,9.560257,1.859165,1.795177,6.451282,-1.830286,-8.022406,1.678877,6.245445,-1.065795,-2.189699,-1.659457,-2.126691,-5.517637,1.790758,3.873368,3.386004,0.630327,-2.581031,8.097197,4.986099,-5.104532,4.878546,4.215555,-7.344508,4.394019,8.603276,-6.783853,-8.678662,-3.535674,8.132542,0.344045,-2.662879,1.729061,6.296600,8.793995,1.964177,-4.530919,9.718611,1.362139,1.122893,8.287269,-8.591783,-4.630161,-5.957444,-8.874920,-1.785038,3.720156,-9.009260,-1.676372,9.139157,-9.667846,2.092407,-8.260269,-4.877536,0.118949,-0.513789,-2.301528,-5.678176,3.953440,9.565141,6.915883,-7.980320,7.845308,8.090456,-0.341142,7.554125,8.381761,-9.175353,4.678377,3.085324,-3.105445,7.824685,-8.951134,-0.907567,4.242770,5.241497,8.755545,4.853434,1.870908,-2.535582,4.898777,-5.436359,2.778474,-3.712358,-2.598220,5.612838,-5.513923,1.697918,-5.065780,-9.542501,-1.935206,-3.823394,9.634179,-3.242619,2.364346,7.868279,-5.320002,6.551081,-8.718530,-6.080671,-3.086391,1.934350,5.585360,2.065264,-8.975326,5.698312,-9.876586,1.054640,9.756487,-4.301514,-7.182552,6.769382,6.639812,-2.928670,9.560216,-6.001117,-4.340826,-3.447522,8.232729,-6.307321,3.325931,-4.261362,1.105870,-4.308891,2.365744,7.621695,-6.467100,-7.859471,-5.304003,5.044858,3.788293,6.294449,1.457200,6.812535,3.162644,4.955933,0.560531,-7.086019,2.071751,6.466805,0.758646,-8.631636,-1.168300,-4.548630,-0.988336,-7.185967,3.122140,5.255384,2.196653,4.193271,-5.999033,9.120529,-4.448275,2.739374,-6.716749,-9.299916,7.532843,8.026273,4.955180,-3.822991,6.411197,-9.410624,-4.145816,-5.920372,-3.328739,8.137737,-9.293774,5.308961,-9.030970,-4.527618,4.529888,0.650560,2.345229,-3.949312,-1.426004,-4.867742,8.017514,-2.635618,8.418932,8.793246,9.424290,-2.085449,-6.062842,-3.105147,-4.653692,4.769364,-9.532479,0.544184,-0.376532,-2.342213,-4.521902,-0.618291,8.159184,-9.533513,4.534167,5.827691,-2.111870,-1.429582,5.378586,8.827420,2.359199,4.988258,9.940412,-1.341177,7.562641,-2.921359,9.503432,8.006070,-6.764892,5.645342,-7.924923,-6.110938,9.940702,9.536265,0.271588,-5.725329,3.652163,-1.696044,-2.244017,-7.570971,-7.006036,-3.089988,-1.412522,-5.165418,3.260602,-7.390828,-7.759079,-5.786947,-3.898544,-4.976512,-9.990986,-2.946054,9.904559,4.521001,2.434694,7.097280,-0.690159,-1.447335,7.646832,4.455038,-1.399223,-0.498795,-9.479440,-0.571307,-0.933843,9.530572,-8.387664,5.349494,-1.659506,9.538032,-1.748845,7.738992,-0.603654,0.204944,9.635351,-1.076007,0.647872,-0.587087,7.053646,-0.375689,-6.269538,5.943994,9.512539,-5.343699,8.461343,-7.787038,-2.280633,-9.448447,3.615082,-8.139457,4.174362,-1.693528,-8.318047,-5.957461,-6.419683,8.882430,-1.105921,1.906574,2.553162,-1.133357,9.573318,8.575591,-5.184121,2.474386,1.882921,-4.151436,-0.671173,3.934752,-8.733420,7.679436,-6.541493,1.996535,4.439863,8.494077,-1.769688,2.707996,-5.251864,8.040324,-4.910773,-7.629609,-1.599849,-1.281552,9.279023,6.496471,-0.827226,-8.748655,-1.362518,1.840414,-0.743615,-4.976296,-7.281521,5.911538,-9.814462,-5.314240,1.924006,3.395268,0.888207,-4.354034,5.957533,-6.512370,-3.024422,8.282594,-8.831013,-9.643878,8.010467,5.254255,-2.974603,-4.935079,0.078983,-4.895436,5.880865,9.594047,4.603013,2.757369,8.576573,-9.825182,2.333427,-1.145104,2.992572,-0.816319,-1.358803,-8.109553,-1.614319,4.031985,5.817912,-0.201412,8.030219,-8.534460,-1.554384,-7.324786,4.534911,7.165087,-3.357111,1.597119,0.066143,-6.493893,0.061436,-6.843559,0.464015,-4.954746,6.940210,6.439014,-2.647119,-7.971336,9.915543,-3.339274,2.588241,8.846370,-1.818540,6.505279,5.113219,7.789663,5.497843,-8.402004,9.632582,7.476430,-7.550820,-7.561571,-6.589261,7.534504,1.042647,2.124036,2.461269,-8.255847,3.425462,6.823922,9.329166,6.504890,5.559125,3.022579,-8.534649,8.958540,3.974159,9.697372,-7.752664,9.071035,1.970398,5.657794,-6.997970,-1.564982,9.600668,-4.411370,-9.850451,3.346074,0.862180,-8.392250,-1.035636,-7.521446,7.751315,2.239648,-4.595537,2.051577,9.242151,0.275298,-8.595664,3.221415,4.548129,-4.298414,-2.866954,6.108112,-1.981078,3.379202,-7.271783,-3.415656,-4.980204,7.530363,0.114135,-2.832979,5.034230,-3.485766,3.784721,-5.608794,1.696964,-9.595907,-0.460577,3.440763,7.783959,3.825740,7.262416,-0.653168,-3.099107,-6.352191,-0.948216,-6.457863,-5.832905,-7.686144,5.625558,5.116161,8.265713,-3.982297,8.662914,0.752810,-0.669025,-5.107566,-7.615093,1.354413,2.110602,-8.544219,-6.380746,2.774843,2.537921,-5.486486,-4.225423,2.094198,9.524386,-5.082847,-4.542704,-8.100368,-0.856365,-7.187148,3.263559,1.360692,2.969872,-6.335578,5.740521,3.014083,8.308468,6.846775,0.530626,-7.269489,-4.651053,0.410389,-6.048470,2.181213,5.974698,6.470027,-5.246037,8.904506,0.994797,8.521930,1.123653,6.360933,8.638780,8.495285,-0.854189,0.343245,-9.574130,5.082141,-9.299799,3.221826,-9.713197,-3.324422,-7.840987,-9.717455,-6.248431,-4.301876,-2.603887,-8.418634,-1.755665,-5.416877,2.605082,7.963167,2.385997,-0.090458,3.510737,0.219191,7.079865,-8.486705,6.364356,3.496737,3.087218,3.810160,0.205533,1.635620,1.151601,-9.154428,8.718148,-0.805573,-3.477357,6.650690,4.359996,5.847120,-4.458736,-7.905017,0.193379,4.236781,8.018617,-5.152513,7.388917,-1.943160,-9.783656,3.710123,-1.388773,-8.170660,5.632734,-3.153734,-8.476308,2.167161,-5.980947,-0.467467,7.866813,-3.320813,-5.137371,8.772804,-5.014057,7.387413,4.968541,-6.373744,-2.584413,-4.184239,-0.614626,-5.469735,7.631971,-2.916829,7.389421,7.078596,-9.485062,1.118606,-4.924185,-1.285908,-0.449117,4.642064,3.575969,0.130489,-0.408500,6.481401,5.400821,8.181567,2.943982,-3.029594,-3.677700,5.728292,6.580529,-6.021532,2.869598,-1.255684,-0.149784,5.347224,6.342340,5.663596,1.994827,5.442118,3.283329,4.315631,-0.457781,3.808855,8.544067,6.431618,-5.835201,-4.569109,5.246186,2.625467,2.354015,9.044086,-7.930853,-5.856250,0.270344,-2.546354,2.824911,2.218540,3.621590,-2.483796,8.095569,5.432784,8.886699,5.821480,-4.794764,-3.875374,-2.015044,-4.794549,7.616326,-6.176843,-9.067451,9.112955,-0.394977,5.229438,8.759851,-5.099798,-6.435205,-7.676760,2.620361,-6.478874,8.149120,7.604734,-3.934365,-2.567994,1.923303,-1.037186,6.468064,3.529447,-2.861285,9.452162,-7.993364,-8.457085,4.747664,9.563380,-2.473750,-0.292165,-0.688839,7.454328,7.196000,-7.566298,-1.440816,1.229720,-6.970126,9.946630,9.697366,9.834415,-4.563735,-7.681730,-1.791115,9.058760,2.263128,-8.200010,5.800101,-0.071316,7.506222,2.313024,5.632123,2.873648,5.300303,-4.967527,2.616472,3.178543,1.659137,2.424850,6.147855,8.008334,-5.792477,-7.783222,7.500446,9.940703,-9.267740,6.759024,3.159504,8.668005,6.884123,9.525467,1.938585,5.151320,9.910955,2.507630,8.408835,1.211819,-4.161140,6.960089,6.143992,3.436748,-1.012407,7.913656,0.997351,-3.547632,5.267658,-2.680132,-2.049707,6.735519,-5.792179,-7.408807,8.838362,-5.240774,-8.572288,-6.288939,-5.552346,7.044277,-2.453382,-5.851360,1.994998,-0.687044,-5.610131,-4.646422,-5.866193,-6.369213,-8.549260,-9.450994,1.713955,9.000196,8.358743,1.355703,3.821619,-5.014759,5.400777,-5.410524,8.901352,3.742624,5.155997,2.858654,-3.913704,-2.337941,-1.185148,-0.666443,8.658487,9.942488,9.888278,-0.462365,2.666813,-4.261474,1.331839,-2.038727,-4.756046,8.270303,4.714995,3.873229,-6.145624,-2.051520,4.784320,-8.040131,-3.588696,0.648393,-7.195682,8.473501,-3.076363,5.652636,-3.891421,-7.679478,5.159791,-7.439625,-5.657748,-7.272536,0.122500,-2.760180,-7.220129,-7.876764,-1.805959,-9.941835,6.588289,2.486616,-3.036975,-7.202786,-0.520599,0.079521,-0.444515,-1.547245,4.589486,9.186920,3.109486,5.924005,7.313679,-4.677030,0.749348,-3.191112,-6.270674,-3.030949,-7.911954,0.882263,-1.251229,7.682446,-7.554443,8.905729,-3.425112,1.961236,-0.029805,8.799569,-2.728284,-0.319307,9.330145,-4.853073,-2.324581,-8.303638,0.782327,3.475577,-6.623548,-9.399512,-3.647458,-2.647536,0.520139,-9.601157,7.832387,4.403682,3.780249,9.052514,-8.173185,8.751973,9.426622,-1.765597,0.797271,-6.329818,-2.988160,8.770607,-1.977226,8.247423,7.110695,-2.995030,-8.697367,-0.707796,3.784026,-9.767523,8.513028,-2.473866,7.654840,-5.047536,-9.606075,-0.611593,6.792726,8.590875,0.013359,2.686667,-8.653094,1.286693,7.301971,-4.605552,-9.117178,0.358947,-3.528341,-7.311791,-1.788943,2.066610,9.589073,4.821700,7.464521,-8.590768,2.106090,7.857000,3.976689,-9.677172,-3.188471,4.551814,6.959199,-2.190368,-4.484275,4.648495,2.457022,7.650100,-8.989243,-1.210982,2.282267,9.942942,-5.768858,-6.005352,4.436619,7.654403,-3.438260,-0.846578,8.583070,0.763068,5.011401,-9.084546,7.385257,5.556012,-7.067813,4.580671,1.252449,-2.082240,6.257712,6.353679,-8.846157,-1.639859,-9.992862,-4.116019,-1.400710,6.152692,0.249045,-9.262811,-9.237473,-0.024942,3.905705,-0.522579,-6.553289,5.240281,-3.308903,1.737267,5.631827,2.562484,-6.264830,0.368758,5.777327,8.682391,-1.189756,-1.838056,1.899597,6.104837,-8.805130,6.263041,-0.684948,1.476700,-2.278764,0.247591,-7.996295,9.000979,9.837877,3.298097,7.416884,4.811739,6.097050,-1.156781,-2.843561,5.795169,-7.886476,-3.771721,7.892327,3.284352,-4.590899,5.013052,2.556537,2.441808,0.278745,5.501049,6.500672,-2.267861,-3.757683,-0.446932,-2.739113,-6.996433,-7.824534,-7.788878,-2.051002,8.917914,-5.482024,2.775164,7.740365,-9.642228,-9.775502,-0.272309,-4.277663,-3.406876,4.859889,-5.839708,-4.638145,-3.052665,9.194743,5.168354,-7.563986,-8.936700,0.668123,-5.640937,-5.168241,-3.826579,1.255326,0.312452,7.044230,-8.412042,-5.829227,2.182871,2.153569,9.371994,-5.032657,-4.801734,-8.142294,7.037197,9.495960,-0.827596,9.791418,-8.867895,0.289426,-9.805264,9.260175,-0.330994,1.243805,-5.368764,2.574837,1.775337,-1.212234,5.697208,-7.177518,1.797372,-4.517181,-3.560472,-8.308595,-2.575213,-6.215778,-6.685108,-7.675422,8.769905,-4.748219,-8.924485,7.826037,2.524852,-2.209566,-5.668831,-5.569951,-9.668179,-3.543174,-4.011863,7.214567,-5.372505,1.714603,0.507692,8.614918,-7.017800,-4.375875,-4.072038,-9.530079,-7.108443,-1.830833,-4.165881,-5.131224,9.325654,7.594505,5.733995,1.178797,-2.362143,-6.809026,-9.954086,-8.995963,-8.914090,4.752205,-1.814065,-4.191633,-8.817242,-8.808249,9.413469,7.486592,8.185958,-6.321452,0.155968,-2.883447,5.619944,2.387605,9.831595,-8.562526,6.497808,-5.613556,8.758071,-4.869537,2.820108,-2.508093,0.059153,9.036198,-2.992825,9.340395,3.958640,5.257023,-5.780763,-8.066076,-0.152772,3.984499,4.667291,-6.832118,5.357584,-2.115718,9.013667,2.843375,8.766464,5.314840,8.682256,6.485414,-7.830488,5.067384,-4.756566,-8.999103,-4.228141,-1.480367,3.298303,-4.135270,3.205793,1.118389,-3.345901,0.181091,-4.022690,4.067922,6.253550,-0.383520,-6.834062,-6.027841,-9.198035,9.128045,0.232394,-1.090198,-8.572719,9.323891,8.773919,-9.869954,5.954575,3.442872,-0.371689,9.643356,-5.235971,-9.340064,3.791610,4.423399,4.431166,-9.119022,-2.418255,-3.760981,-6.341328,-1.688274,-1.320636,8.624428,-0.016681,7.007424,3.476583,-7.371209,6.914252,-8.235151,6.884091,-0.474994,7.719602,1.440584,0.860796,-6.576719,9.743023,8.958586,-3.056695,-2.199793,-0.264550,8.207794,-1.032298,3.343397,-3.117628,1.478364,7.233651,1.590999,-3.897254,-5.154065,-9.123987,6.894647,8.533643,2.193003,5.804197,-6.521994,-8.208693,-6.659874,5.585508,-0.772156,2.718253,5.515646,9.726155,0.061989,-7.440305,5.435613,7.873792,-2.702659,5.673126,-3.902164,4.080892,-6.671968,-6.757560,7.607200,-0.229182,-1.733724,6.270282,-4.898231,-4.253406,-7.706774,0.964702,6.377217,1.408318,8.936640,6.943394,1.296765,-0.150020,-6.346657,-3.490872,-7.666893,7.972867,3.674726,1.925659,4.640604,2.051386,-2.494091,4.364694,0.574165,3.978933,-3.175673,-9.881690,-2.699939,9.318208,-6.734598,-8.907565,5.156414,4.210164,-8.572481,-0.759385,-5.436646,0.972532,9.261322,-5.768501,7.148181,7.711202,-9.800277,-8.565801,2.834165,-3.007132,7.772051,-7.768981,-1.595538,1.156478,-1.643564,6.530799,-6.097394,7.902931,5.673278,-0.547631,-1.748294,-6.231646,-7.455367,1.689870,1.500386,3.490746,-9.103289,7.224889,3.602335,2.656587,-7.267418,-7.062009,3.210190,-8.940553,7.802606,-1.112885,9.770711,-3.826934,4.280717,8.850379,1.384721,5.637105,-1.338543,0.779798,-3.117855,-3.042954,5.250162,2.664444,-6.463267,2.997005,-8.959899,0.886175,6.787941,7.142798,0.252177,0.023992,8.752488,-3.755544,5.960624,0.631316,-7.372336,-2.779607,-6.742523,1.116572,-2.808279,-0.906111,-2.117037,-4.986641,-9.451010,-4.801211,-1.120231,-3.382774,-7.589005,5.782388,-4.758177,8.140804,-0.124290,-5.125595,9.040393,-8.489810,2.945156,0.693769,-2.914766,5.029634,1.074651,0.931297,5.629158,-5.391822,-3.862830,6.947413,-9.317466,-4.745292,-4.902665,-7.845617,-0.482364,4.630822,9.515304,7.095133,-1.865639,-9.078767,-9.831037,3.253261,-6.288703,6.136285,-7.103405,8.746603,-7.194789,2.577279,6.125032,-5.427112,-9.817739,4.816981,-0.035098,-5.903624,5.404355,1.313979,-7.052445,5.115018,-0.353748,9.206165,2.127704,5.936807,-2.880545,-7.622605,-9.409889,1.582114,9.456391,6.145349,0.022855,-1.647347,-4.980465,-8.900542,2.348337,1.383302,8.436580,-6.792880,-4.409205,-2.502354,-9.682239,-3.663663,0.755359,-8.579103,-6.081984,1.953989,4.726756,-9.211749,1.053840,-6.483078,-9.541590,-5.287496,-1.382298,-4.694470,-7.988977,-0.513182,3.826248,-4.616024,-9.560092,1.432223,-1.918988,1.494917,8.764513,2.478587,-4.290573,8.537793,-7.802289,-0.512276,-2.471549,2.738826,7.104496,3.457601,-7.916643,-4.129646,-2.927341,2.710241,-1.686277,-6.625755,-5.584772,-0.970704,2.347833,7.575252,0.697055,5.128387,-8.548060,-6.736706,-4.329993,-2.835025,0.057448,9.801846,3.568057,-2.042616,1.047421,1.367504,6.388895,8.103956,7.500162,-8.540247,-2.790941,-7.718981,-7.852804,6.045242,-7.978890,7.580828,-4.573930,-6.708638,5.154981,4.467987,8.992112,-2.674108,-3.723546,-5.429657,-8.942378,-8.684650,-2.790617,0.381165,2.437621,9.250126,-5.206549,-6.227075,4.169039,-2.348067,1.610616,-7.331497,-3.311090,6.975846,-2.125666,-7.653539,-3.470420,7.867306,0.986675,0.333220,7.452629,9.397560,3.762101,-5.701404,-9.404419,9.033414,-3.789657,-8.451108,-8.851586,1.903063,4.263821,4.276344,-5.384095,-8.851627,7.333875,8.272576,-6.304906,6.647742,-0.447984,1.533326,-1.476282,-9.191489,-4.504161,-3.273541,7.245738,2.943417,0.709265,-5.897462,9.672942,-4.699734,7.731124,-6.490773,-0.666867,5.972811,6.323304,8.716201,-4.184049,3.907041,-3.606729,4.271392,1.351440,0.369963,-4.990365,2.211304,-5.976947,-0.207137,9.623159,-2.394304,-6.782300,-2.779100,-6.304279,-4.774582,0.520755,4.860619,2.739800,1.138424,9.068356,1.871479,3.535605,8.544525,-8.816503,7.467082,-5.979412,6.769759,0.496913,4.832979,9.689103,-0.040954,2.970454,1.884875,1.319841,8.350019,4.638464,-5.327091,2.368257,-0.880874,9.715766,3.863210,5.756351,6.887923,-6.694817,-8.802681,-5.590929,8.099938,-6.317494,-0.747579,5.433198,-8.566217,-0.736599,3.396578,3.399799,-2.628825,5.015666,-8.310298,-5.417002,-9.894085,-8.005963,5.527519,3.614092,-3.247126,6.765007,1.386893,6.868591,7.318860,-7.780750,0.061670,-3.898540,8.339381,-8.188753,2.302378,8.881508,5.923427,-9.156004,-6.050160,6.277532,-8.510558,4.045386,-2.984910,6.945287,-1.376235,-7.790353,9.984828,4.987493,2.128068,2.001133,8.528167,-1.958487,-7.527307,8.907151,-4.378429,-7.809315,0.740512,1.943581,-1.667002,7.026254,-5.308969,-6.676892,-6.598943,-4.575839,5.338389,-1.900884,-4.855295,-8.577612,-9.233686,9.024922,-3.790864,2.560468,3.718505,1.489126,7.076649,4.745417,1.889900,-3.012115,1.092321,1.811889,-0.490920,0.090103,-0.081612,6.445192,2.523432], dtype = "float32")#candidate|1649|(2548,)|const|float32
call_1647 = relay.TupleGetItem(func_1240_call(relay.reshape(const_1648.astype('float32'), [2,]), relay.reshape(uop_1623.astype('int64'), [336,]), relay.reshape(const_1649.astype('float32'), [14, 13, 14]), ), 3)
call_1650 = relay.TupleGetItem(func_1244_call(relay.reshape(const_1648.astype('float32'), [2,]), relay.reshape(uop_1623.astype('int64'), [336,]), relay.reshape(const_1649.astype('float32'), [14, 13, 14]), ), 3)
bop_1654 = relay.mod(uop_1614.astype('float64'), relay.reshape(call_1593.astype('float64'), relay.shape_of(uop_1614))) # shape=(336,)
bop_1657 = relay.mod(uop_1616.astype('float64'), relay.reshape(call_1594.astype('float64'), relay.shape_of(uop_1616))) # shape=(336,)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1658 = relay.TupleGetItem(func_695_call(), 0)
call_1659 = relay.TupleGetItem(func_696_call(), 0)
bop_1662 = relay.add(uop_1614.astype('float32'), relay.reshape(call_1612.astype('float32'), relay.shape_of(uop_1614))) # shape=(336,)
bop_1665 = relay.add(uop_1616.astype('float32'), relay.reshape(call_1613.astype('float32'), relay.shape_of(uop_1616))) # shape=(336,)
uop_1666 = relay.cos(bop_1662.astype('float64')) # shape=(336,)
uop_1668 = relay.cos(bop_1665.astype('float64')) # shape=(336,)
func_631_call = mod.get_global_var('func_631')
func_633_call = mutated_mod.get_global_var('func_633')
call_1671 = relay.TupleGetItem(func_631_call(), 1)
call_1672 = relay.TupleGetItem(func_633_call(), 1)
uop_1673 = relay.sinh(uop_1666.astype('float32')) # shape=(336,)
uop_1675 = relay.sinh(uop_1668.astype('float32')) # shape=(336,)
var_1682 = relay.var("var_1682", dtype = "float32", shape = (336,))#candidate|1682|(336,)|var|float32
bop_1683 = relay.maximum(uop_1673.astype('int16'), relay.reshape(var_1682.astype('int16'), relay.shape_of(uop_1673))) # shape=(336,)
bop_1686 = relay.maximum(uop_1675.astype('int16'), relay.reshape(var_1682.astype('int16'), relay.shape_of(uop_1675))) # shape=(336,)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
call_1695 = relay.TupleGetItem(func_237_call(relay.reshape(call_1576.astype('int64'), [336,])), 0)
call_1696 = relay.TupleGetItem(func_239_call(relay.reshape(call_1576.astype('int64'), [336,])), 0)
output = relay.Tuple([call_1603,uop_1623,bop_1634,uop_1638,call_1647,const_1648,const_1649,bop_1654,call_1658,call_1671,bop_1683,call_1695,])
output2 = relay.Tuple([call_1604,uop_1625,bop_1637,uop_1640,call_1650,const_1648,const_1649,bop_1657,call_1659,call_1672,bop_1686,call_1696,])
func_1698 = relay.Function([var_1682,], output)
mod['func_1698'] = func_1698
mod = relay.transform.InferType()(mod)
mutated_mod['func_1698'] = func_1698
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1699 = relay.var("var_1699", dtype = "float32", shape = (336,))#candidate|1699|(336,)|var|float32
func_1698_call = mutated_mod.get_global_var('func_1698')
call_1700 = func_1698_call(var_1699)
output = call_1700
func_1701 = relay.Function([var_1699], output)
mutated_mod['func_1701'] = func_1701
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1749 = relay.TupleGetItem(func_695_call(), 0)
call_1750 = relay.TupleGetItem(func_696_call(), 0)
func_1514_call = mod.get_global_var('func_1514')
func_1516_call = mutated_mod.get_global_var('func_1516')
const_1760 = relay.const([-3.635919,-9.882375,9.276382,-3.865748,5.080086,0.622295,-6.716869,-9.774938,-5.620366,5.834531,-7.740268,9.838788,-5.254664,-7.002616,-7.136743,-1.647928,-3.565428,3.139684,8.907840,-0.086169,6.730742,-2.698858,-3.442250,7.868117,9.253676,9.342835,0.935204,5.226326,-4.611922,6.805101,-4.749163,7.112513,9.215265,-0.618446,7.257794,6.579133,3.313898,-9.452863,-8.343184,-5.071865,5.144710,-4.829712,7.497754,-2.868109,0.942399,-2.471961,8.516905,-6.221568,3.700408,4.413364,-2.941877,7.150626,7.341228,3.300658,5.251622,0.041442,5.572426,-1.331463,0.930109,-2.578946,-8.727383,-6.943433,6.381462,-3.012994,-9.296276,3.822418,2.510569,-6.362811,-0.103985,-3.423744,5.531228,-2.997883,4.505277,5.033032,2.252394,-0.843723,2.573631,-5.570193,8.766356,0.858803,-9.570844,-4.830678,4.567978,-1.602922,-1.384846,-6.895643,6.310846,1.878171,6.389570,9.772488,-9.209583,-6.421508,9.024212,-4.167644,-9.197982,-5.401778,-9.474985,-0.808443,2.654468,-0.471053,9.205429,9.371976,6.971937,-5.192472,-0.177103,-5.000068,4.659187,-7.752016,7.853041,4.121531,-5.147242,-5.002197,6.146216,-5.717262,-7.941702,-9.017766,8.091725,-0.516285,-6.227317,7.672930,-6.692567,-0.197786,9.138401,-2.512928,7.081365,-0.163770,8.364663,0.178632,6.262554,1.326615,-7.500735,1.750964,-4.814119,-4.678965,-1.926321,9.441968,2.262676,-7.797410,-1.795098,2.697681,6.837851,6.052613,8.805578,6.922856,-7.600910,-1.542835,-0.151168,0.604431,5.987781,-8.813256,3.981645,-4.339801,1.743982,9.402681,-2.006623,5.594486,-6.671802,-5.637790,7.868278,-0.159251,7.454068,-5.625926,5.979497,-1.840162,-8.135675,-0.557559,-0.457667,-3.907735,-8.753286,6.053176,5.673671,-5.151186,5.184579,5.605684,-7.527898,-3.985054,-7.136302,8.055143,-9.076270,-0.453194], dtype = "float64")#candidate|1760|(180,)|const|float64
call_1759 = relay.TupleGetItem(func_1514_call(relay.reshape(const_1760.astype('float64'), [3, 6, 10])), 0)
call_1761 = relay.TupleGetItem(func_1516_call(relay.reshape(const_1760.astype('float64'), [3, 6, 10])), 0)
func_415_call = mod.get_global_var('func_415')
func_420_call = mutated_mod.get_global_var('func_420')
var_1781 = relay.var("var_1781", dtype = "bool", shape = (84,))#candidate|1781|(84,)|var|bool
var_1782 = relay.var("var_1782", dtype = "float32", shape = (39, 5))#candidate|1782|(39, 5)|var|float32
call_1780 = relay.TupleGetItem(func_415_call(relay.reshape(var_1781.astype('bool'), [4, 3, 7]), relay.reshape(var_1781.astype('bool'), [4, 3, 7]), relay.reshape(var_1782.astype('float32'), [195,]), ), 1)
call_1783 = relay.TupleGetItem(func_420_call(relay.reshape(var_1781.astype('bool'), [4, 3, 7]), relay.reshape(var_1781.astype('bool'), [4, 3, 7]), relay.reshape(var_1782.astype('float32'), [195,]), ), 1)
uop_1785 = relay.asin(const_1760.astype('float64')) # shape=(180,)
bop_1788 = relay.power(uop_1785.astype('float32'), relay.reshape(call_1759.astype('float32'), relay.shape_of(uop_1785))) # shape=(180,)
bop_1791 = relay.power(uop_1785.astype('float32'), relay.reshape(call_1761.astype('float32'), relay.shape_of(uop_1785))) # shape=(180,)
func_664_call = mod.get_global_var('func_664')
func_666_call = mutated_mod.get_global_var('func_666')
call_1793 = relay.TupleGetItem(func_664_call(relay.reshape(call_1780.astype('float32'), [195,])), 2)
call_1794 = relay.TupleGetItem(func_666_call(relay.reshape(call_1780.astype('float32'), [195,])), 2)
bop_1796 = relay.floor_divide(bop_1788.astype('float64'), relay.reshape(call_1759.astype('float64'), relay.shape_of(bop_1788))) # shape=(180,)
bop_1799 = relay.floor_divide(bop_1791.astype('float64'), relay.reshape(call_1761.astype('float64'), relay.shape_of(bop_1791))) # shape=(180,)
uop_1800 = relay.cos(var_1781.astype('float64')) # shape=(84,)
output = relay.Tuple([call_1749,call_1780,var_1782,call_1793,bop_1796,uop_1800,])
output2 = relay.Tuple([call_1750,call_1783,var_1782,call_1794,bop_1799,uop_1800,])
func_1807 = relay.Function([var_1781,var_1782,], output)
mod['func_1807'] = func_1807
mod = relay.transform.InferType()(mod)
var_1808 = relay.var("var_1808", dtype = "bool", shape = (84,))#candidate|1808|(84,)|var|bool
var_1809 = relay.var("var_1809", dtype = "float32", shape = (39, 5))#candidate|1809|(39, 5)|var|float32
output = func_1807(var_1808,var_1809,)
func_1810 = relay.Function([var_1808,var_1809,], output)
mutated_mod['func_1810'] = func_1810
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1490_call = mod.get_global_var('func_1490')
func_1491_call = mutated_mod.get_global_var('func_1491')
call_1825 = relay.TupleGetItem(func_1490_call(), 0)
call_1826 = relay.TupleGetItem(func_1491_call(), 0)
output = call_1825
output2 = call_1826
func_1867 = relay.Function([], output)
mod['func_1867'] = func_1867
mod = relay.transform.InferType()(mod)
output = func_1867()
func_1868 = relay.Function([], output)
mutated_mod['func_1868'] = func_1868
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1867_call = mod.get_global_var('func_1867')
func_1868_call = mutated_mod.get_global_var('func_1868')
call_1960 = func_1867_call()
call_1961 = func_1867_call()
output = relay.Tuple([call_1960,])
output2 = relay.Tuple([call_1961,])
func_1971 = relay.Function([], output)
mod['func_1971'] = func_1971
mod = relay.transform.InferType()(mod)
output = func_1971()
func_1972 = relay.Function([], output)
mutated_mod['func_1972'] = func_1972
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1985 = relay.const([[-3,9,6,-4,-1,9],[-9,-6,-6,-5,-8,2]], dtype = "uint32")#candidate|1985|(2, 6)|const|uint32
var_1986 = relay.var("var_1986", dtype = "uint32", shape = (2, 6))#candidate|1986|(2, 6)|var|uint32
bop_1987 = relay.bitwise_or(const_1985.astype('uint32'), relay.reshape(var_1986.astype('uint32'), relay.shape_of(const_1985))) # shape=(2, 6)
output = relay.Tuple([bop_1987,])
output2 = relay.Tuple([bop_1987,])
func_2001 = relay.Function([var_1986,], output)
mod['func_2001'] = func_2001
mod = relay.transform.InferType()(mod)
var_2002 = relay.var("var_2002", dtype = "uint32", shape = (2, 6))#candidate|2002|(2, 6)|var|uint32
output = func_2001(var_2002)
func_2003 = relay.Function([var_2002], output)
mutated_mod['func_2003'] = func_2003
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2021 = relay.var("var_2021", dtype = "uint8", shape = (2, 8))#candidate|2021|(2, 8)|var|uint8
const_2022 = relay.const([[-7,-8,-7,2,-1,-1,-4,-10],[1,-2,9,-1,1,-3,2,-6]], dtype = "uint8")#candidate|2022|(2, 8)|const|uint8
bop_2023 = relay.less(var_2021.astype('bool'), relay.reshape(const_2022.astype('bool'), relay.shape_of(var_2021))) # shape=(2, 8)
var_2029 = relay.var("var_2029", dtype = "uint8", shape = (2, 8))#candidate|2029|(2, 8)|var|uint8
bop_2030 = relay.add(const_2022.astype('uint64'), relay.reshape(var_2029.astype('uint64'), relay.shape_of(const_2022))) # shape=(2, 8)
bop_2036 = relay.not_equal(const_2022.astype('bool'), relay.reshape(bop_2030.astype('bool'), relay.shape_of(const_2022))) # shape=(2, 8)
bop_2042 = relay.divide(const_2022.astype('float32'), relay.reshape(bop_2023.astype('float32'), relay.shape_of(const_2022))) # shape=(2, 8)
bop_2052 = relay.multiply(bop_2036.astype('int32'), relay.reshape(bop_2030.astype('int32'), relay.shape_of(bop_2036))) # shape=(2, 8)
func_1179_call = mod.get_global_var('func_1179')
func_1181_call = mutated_mod.get_global_var('func_1181')
call_2062 = relay.TupleGetItem(func_1179_call(), 0)
call_2063 = relay.TupleGetItem(func_1181_call(), 0)
uop_2065 = relay.rsqrt(bop_2023.astype('float64')) # shape=(2, 8)
func_585_call = mod.get_global_var('func_585')
func_586_call = mutated_mod.get_global_var('func_586')
call_2067 = func_585_call()
call_2068 = func_585_call()
output = relay.Tuple([bop_2042,bop_2052,call_2062,uop_2065,call_2067,])
output2 = relay.Tuple([bop_2042,bop_2052,call_2063,uop_2065,call_2068,])
func_2079 = relay.Function([var_2021,var_2029,], output)
mod['func_2079'] = func_2079
mod = relay.transform.InferType()(mod)
var_2080 = relay.var("var_2080", dtype = "uint8", shape = (2, 8))#candidate|2080|(2, 8)|var|uint8
var_2081 = relay.var("var_2081", dtype = "uint8", shape = (2, 8))#candidate|2081|(2, 8)|var|uint8
output = func_2079(var_2080,var_2081,)
func_2082 = relay.Function([var_2080,var_2081,], output)
mutated_mod['func_2082'] = func_2082
mutated_mod = relay.transform.InferType()(mutated_mod)
func_312_call = mod.get_global_var('func_312')
func_314_call = mutated_mod.get_global_var('func_314')
call_2084 = relay.TupleGetItem(func_312_call(), 1)
call_2085 = relay.TupleGetItem(func_314_call(), 1)
func_1971_call = mod.get_global_var('func_1971')
func_1972_call = mutated_mod.get_global_var('func_1972')
call_2092 = relay.TupleGetItem(func_1971_call(), 0)
call_2093 = relay.TupleGetItem(func_1972_call(), 0)
output = relay.Tuple([call_2084,call_2092,])
output2 = relay.Tuple([call_2085,call_2093,])
func_2097 = relay.Function([], output)
mod['func_2097'] = func_2097
mod = relay.transform.InferType()(mod)
mutated_mod['func_2097'] = func_2097
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2097_call = mutated_mod.get_global_var('func_2097')
call_2098 = func_2097_call()
output = call_2098
func_2099 = relay.Function([], output)
mutated_mod['func_2099'] = func_2099
mutated_mod = relay.transform.InferType()(mutated_mod)
func_356_call = mod.get_global_var('func_356')
func_358_call = mutated_mod.get_global_var('func_358')
call_2124 = relay.TupleGetItem(func_356_call(), 0)
call_2125 = relay.TupleGetItem(func_358_call(), 0)
func_2001_call = mod.get_global_var('func_2001')
func_2003_call = mutated_mod.get_global_var('func_2003')
var_2130 = relay.var("var_2130", dtype = "uint32", shape = (12,))#candidate|2130|(12,)|var|uint32
call_2129 = relay.TupleGetItem(func_2001_call(relay.reshape(var_2130.astype('uint32'), [2, 6])), 0)
call_2131 = relay.TupleGetItem(func_2003_call(relay.reshape(var_2130.astype('uint32'), [2, 6])), 0)
func_346_call = mod.get_global_var('func_346')
func_349_call = mutated_mod.get_global_var('func_349')
var_2136 = relay.var("var_2136", dtype = "float32", shape = (195,))#candidate|2136|(195,)|var|float32
call_2135 = func_346_call(relay.reshape(var_2136.astype('float32'), [13, 15]))
call_2137 = func_346_call(relay.reshape(var_2136.astype('float32'), [13, 15]))
uop_2140 = relay.tan(call_2135.astype('float32')) # shape=(13, 15)
uop_2142 = relay.tan(call_2137.astype('float32')) # shape=(13, 15)
var_2144 = relay.var("var_2144", dtype = "float32", shape = (13, 15))#candidate|2144|(13, 15)|var|float32
bop_2145 = relay.floor_mod(uop_2140.astype('float32'), relay.reshape(var_2144.astype('float32'), relay.shape_of(uop_2140))) # shape=(13, 15)
bop_2148 = relay.floor_mod(uop_2142.astype('float32'), relay.reshape(var_2144.astype('float32'), relay.shape_of(uop_2142))) # shape=(13, 15)
output = relay.Tuple([call_2124,call_2129,var_2130,var_2136,bop_2145,])
output2 = relay.Tuple([call_2125,call_2131,var_2130,var_2136,bop_2148,])
func_2155 = relay.Function([var_2130,var_2136,var_2144,], output)
mod['func_2155'] = func_2155
mod = relay.transform.InferType()(mod)
var_2156 = relay.var("var_2156", dtype = "uint32", shape = (12,))#candidate|2156|(12,)|var|uint32
var_2157 = relay.var("var_2157", dtype = "float32", shape = (195,))#candidate|2157|(195,)|var|float32
var_2158 = relay.var("var_2158", dtype = "float32", shape = (13, 15))#candidate|2158|(13, 15)|var|float32
output = func_2155(var_2156,var_2157,var_2158,)
func_2159 = relay.Function([var_2156,var_2157,var_2158,], output)
mutated_mod['func_2159'] = func_2159
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2177 = relay.var("var_2177", dtype = "int64", shape = (8, 5, 5))#candidate|2177|(8, 5, 5)|var|int64
const_2178 = relay.const([[[-4,-3,-5,-1,-10],[9,-1,10,3,10],[-6,-6,-6,1,-10],[-1,7,-7,8,3],[-6,-2,1,4,5]],[[-8,6,-7,-6,-10],[-7,1,4,-8,8],[10,6,-1,-3,2],[9,-10,4,9,-6],[-4,-9,6,-5,-6]],[[8,4,-8,-10,10],[-9,-2,5,-6,5],[3,-4,-6,-6,7],[-3,7,-8,4,8],[-9,-4,5,6,4]],[[-3,3,-9,-8,-2],[-4,9,-2,-8,2],[-2,-2,4,8,9],[3,8,9,-5,-10],[-3,1,8,6,10]],[[2,1,-10,9,4],[9,2,-1,5,-10],[-4,3,-3,-9,-9],[-7,2,-2,2,10],[3,-1,-1,4,2]],[[-2,6,-2,-8,-6],[-1,-5,10,5,3],[7,-9,9,-3,-5],[-8,1,-4,-1,-7],[7,-1,-1,2,-4]],[[-3,10,-9,1,3],[-7,3,-7,5,9],[-6,-10,-1,-9,-8],[9,-3,-5,-10,-4],[-2,-8,-1,2,-2]],[[7,-9,4,-7,4],[-5,7,6,10,-8],[-10,5,5,-2,1],[-10,-1,-3,-9,-7],[6,-8,2,2,-9]]], dtype = "int64")#candidate|2178|(8, 5, 5)|const|int64
bop_2179 = relay.multiply(var_2177.astype('int64'), relay.reshape(const_2178.astype('int64'), relay.shape_of(var_2177))) # shape=(8, 5, 5)
bop_2182 = relay.not_equal(var_2177.astype('bool'), relay.reshape(const_2178.astype('bool'), relay.shape_of(var_2177))) # shape=(8, 5, 5)
output = relay.Tuple([bop_2179,bop_2182,])
output2 = relay.Tuple([bop_2179,bop_2182,])
F = relay.Function([var_2177,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_2177,], output2)
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
	relay.transform.ToANormalForm(),
	relay.transform.ToGraphNormalForm(),
	relay.transform.SimplifyInference(),
	relay.transform.ToBasicBlockNormalForm(),
	relay.transform.FuseOps(3),
	relay.transform.DefuseOps(),
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
input_2177= np.array([[[6,4,5,7,-1],[7,-3,-4,-1,8],[8,9,1,5,5],[-9,-1,6,5,3],[-2,1,-3,10,5]],[[-7,9,9,5,-6],[-1,5,8,-6,-7],[-8,1,6,-8,9],[-6,4,-2,5,-9],[10,-6,2,8,-4]],[[8,6,5,5,-8],[-5,-9,-10,-4,-10],[6,2,9,-3,-1],[2,2,1,7,1],[10,-2,6,2,8]],[[2,-4,-5,8,9],[-6,-1,8,6,-7],[3,-4,6,6,2],[10,8,-2,-8,-1],[-2,-6,7,-4,-1]],[[1,-6,-8,7,-10],[10,4,-5,-8,-7],[-8,8,-1,-1,-8],[-5,6,-4,-6,2],[1,-8,-7,3,6]],[[6,10,-1,7,-5],[-4,-3,-2,7,-7],[3,-1,5,4,5],[8,5,5,7,5],[8,2,10,-10,1]],[[-7,7,7,-8,-8],[10,-6,3,10,7],[4,5,-9,-2,-5],[6,9,-2,5,-7],[4,-10,2,-4,-3]],[[-8,-6,-1,-9,4],[7,-5,-5,-3,6],[-5,9,1,-9,4],[3,7,4,-6,2],[-6,-2,-5,8,9]]], dtype='int64')
module1.set_input('var_2177', input_2177)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_2177, )
res3 = intrp3.evaluate()(input_2177, )
res4 = intrp4.evaluate()(input_2177, )
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
module5.set_input('var_2177', input_2177)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_2177, )
res7 = intrp7.evaluate()(input_2177, )
res8 = intrp8.evaluate()(input_2177, )
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
module9.set_input('var_2177', input_2177)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_2177, )
res11 = intrp11.evaluate()(input_2177, )
res12 = intrp12.evaluate()(input_2177, )
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
module13.set_input('var_2177', input_2177)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_2177, )
res15 = intrp15.evaluate()(input_2177, )
res16 = intrp16.evaluate()(input_2177, )
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
module17.set_input('var_2177', input_2177)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_2177, )
res19 = intrp19.evaluate()(input_2177, )
res20 = intrp20.evaluate()(input_2177, )
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
module21.set_input('var_2177', input_2177)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_2177, )
res23 = intrp23.evaluate()(input_2177, )
res24 = intrp24.evaluate()(input_2177, )
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