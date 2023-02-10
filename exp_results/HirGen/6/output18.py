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
const_60 = relay.const([[[6.774595],[3.453253],[6.223274],[2.305386],[8.643169],[-7.689814],[-2.439119],[0.079805],[-1.688202],[-8.623747],[-8.340080],[-6.585798],[-3.080390],[-7.208266],[7.375502],[-8.635610]]], dtype = "float64")#candidate|60|(1, 16, 1)|const|float64
uop_61 = relay.exp(const_60.astype('float64')) # shape=(1, 16, 1)
uop_64 = relay.erf(uop_61.astype('float32')) # shape=(1, 16, 1)
bop_66 = relay.logical_xor(uop_64.astype('int32'), relay.reshape(uop_61.astype('int32'), relay.shape_of(uop_64))) # shape=(1, 16, 1)
uop_72 = relay.atan(bop_66.astype('float32')) # shape=(1, 16, 1)
bop_76 = relay.maximum(uop_72.astype('int8'), relay.reshape(uop_61.astype('int8'), relay.shape_of(uop_72))) # shape=(1, 16, 1)
output = bop_76
output2 = bop_76
func_87 = relay.Function([], output)
mod['func_87'] = func_87
mod = relay.transform.InferType()(mod)
mutated_mod['func_87'] = func_87
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mutated_mod.get_global_var('func_87')
call_88 = func_87_call()
output = call_88
func_89 = relay.Function([], output)
mutated_mod['func_89'] = func_89
mutated_mod = relay.transform.InferType()(mutated_mod)
var_93 = relay.var("var_93", dtype = "float32", shape = (6, 1))#candidate|93|(6, 1)|var|float32
uop_94 = relay.cosh(var_93.astype('float32')) # shape=(6, 1)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_97 = func_87_call()
call_98 = func_87_call()
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_117 = func_87_call()
call_118 = func_87_call()
output = relay.Tuple([uop_94,call_97,call_117,])
output2 = relay.Tuple([uop_94,call_98,call_118,])
func_123 = relay.Function([var_93,], output)
mod['func_123'] = func_123
mod = relay.transform.InferType()(mod)
mutated_mod['func_123'] = func_123
mutated_mod = relay.transform.InferType()(mutated_mod)
var_124 = relay.var("var_124", dtype = "float32", shape = (6, 1))#candidate|124|(6, 1)|var|float32
func_123_call = mutated_mod.get_global_var('func_123')
call_125 = func_123_call(var_124)
output = call_125
func_126 = relay.Function([var_124], output)
mutated_mod['func_126'] = func_126
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_168 = func_87_call()
call_169 = func_87_call()
var_187 = relay.var("var_187", dtype = "int8", shape = (6, 16, 6))#candidate|187|(6, 16, 6)|var|int8
bop_188 = relay.not_equal(call_168.astype('bool'), var_187.astype('bool')) # shape=(6, 16, 6)
bop_191 = relay.not_equal(call_169.astype('bool'), var_187.astype('bool')) # shape=(6, 16, 6)
bop_192 = relay.left_shift(bop_188.astype('int16'), relay.reshape(var_187.astype('int16'), relay.shape_of(bop_188))) # shape=(6, 16, 6)
bop_195 = relay.left_shift(bop_191.astype('int16'), relay.reshape(var_187.astype('int16'), relay.shape_of(bop_191))) # shape=(6, 16, 6)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
const_198 = relay.const([[-7.372921,-7.713230],[-5.706482,7.588318],[-7.728062,0.383260]], dtype = "float32")#candidate|198|(3, 2)|const|float32
call_197 = relay.TupleGetItem(func_123_call(relay.reshape(const_198.astype('float32'), [6, 1])), 0)
call_199 = relay.TupleGetItem(func_126_call(relay.reshape(const_198.astype('float32'), [6, 1])), 0)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
call_213 = relay.TupleGetItem(func_123_call(relay.reshape(const_198.astype('float32'), [6, 1])), 2)
call_214 = relay.TupleGetItem(func_126_call(relay.reshape(const_198.astype('float32'), [6, 1])), 2)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
call_215 = relay.TupleGetItem(func_123_call(relay.reshape(call_197.astype('float32'), [6, 1])), 0)
call_216 = relay.TupleGetItem(func_126_call(relay.reshape(call_197.astype('float32'), [6, 1])), 0)
output = relay.Tuple([bop_192,call_197,const_198,call_213,call_215,])
output2 = relay.Tuple([bop_195,call_199,const_198,call_214,call_216,])
func_218 = relay.Function([var_187,], output)
mod['func_218'] = func_218
mod = relay.transform.InferType()(mod)
var_219 = relay.var("var_219", dtype = "int8", shape = (6, 16, 6))#candidate|219|(6, 16, 6)|var|int8
output = func_218(var_219)
func_220 = relay.Function([var_219], output)
mutated_mod['func_220'] = func_220
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_246 = func_87_call()
call_247 = func_87_call()
output = call_246
output2 = call_247
func_254 = relay.Function([], output)
mod['func_254'] = func_254
mod = relay.transform.InferType()(mod)
output = func_254()
func_255 = relay.Function([], output)
mutated_mod['func_255'] = func_255
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_351 = func_87_call()
call_352 = func_87_call()
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_359 = func_87_call()
call_360 = func_87_call()
bop_362 = relay.not_equal(call_359.astype('bool'), relay.reshape(call_351.astype('bool'), relay.shape_of(call_359))) # shape=(1, 16, 1)
bop_365 = relay.not_equal(call_360.astype('bool'), relay.reshape(call_352.astype('bool'), relay.shape_of(call_360))) # shape=(1, 16, 1)
uop_367 = relay.sigmoid(bop_362.astype('float64')) # shape=(1, 16, 1)
uop_369 = relay.sigmoid(bop_365.astype('float64')) # shape=(1, 16, 1)
var_382 = relay.var("var_382", dtype = "float64", shape = (4, 16, 10))#candidate|382|(4, 16, 10)|var|float64
bop_383 = relay.subtract(uop_367.astype('uint64'), var_382.astype('uint64')) # shape=(4, 16, 10)
bop_386 = relay.subtract(uop_369.astype('uint64'), var_382.astype('uint64')) # shape=(4, 16, 10)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
const_390 = relay.const([-3.486775,-9.556424,2.280372,6.722756,-0.922741,1.766222], dtype = "float32")#candidate|390|(6,)|const|float32
call_389 = relay.TupleGetItem(func_123_call(relay.reshape(const_390.astype('float32'), [6, 1])), 0)
call_391 = relay.TupleGetItem(func_126_call(relay.reshape(const_390.astype('float32'), [6, 1])), 0)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_394 = func_254_call()
call_395 = func_254_call()
output = relay.Tuple([bop_383,call_389,const_390,call_394,])
output2 = relay.Tuple([bop_386,call_391,const_390,call_395,])
func_396 = relay.Function([var_382,], output)
mod['func_396'] = func_396
mod = relay.transform.InferType()(mod)
mutated_mod['func_396'] = func_396
mutated_mod = relay.transform.InferType()(mutated_mod)
var_397 = relay.var("var_397", dtype = "float64", shape = (4, 16, 10))#candidate|397|(4, 16, 10)|var|float64
func_396_call = mutated_mod.get_global_var('func_396')
call_398 = func_396_call(var_397)
output = call_398
func_399 = relay.Function([var_397], output)
mutated_mod['func_399'] = func_399
mutated_mod = relay.transform.InferType()(mutated_mod)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_401 = func_254_call()
call_402 = func_254_call()
output = relay.Tuple([call_401,])
output2 = relay.Tuple([call_402,])
func_408 = relay.Function([], output)
mod['func_408'] = func_408
mod = relay.transform.InferType()(mod)
output = func_408()
func_409 = relay.Function([], output)
mutated_mod['func_409'] = func_409
mutated_mod = relay.transform.InferType()(mutated_mod)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_432 = func_254_call()
call_433 = func_254_call()
var_438 = relay.var("var_438", dtype = "int8", shape = (14, 16, 8))#candidate|438|(14, 16, 8)|var|int8
bop_439 = relay.less_equal(call_432.astype('bool'), var_438.astype('bool')) # shape=(14, 16, 8)
bop_442 = relay.less_equal(call_433.astype('bool'), var_438.astype('bool')) # shape=(14, 16, 8)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_443 = func_87_call()
call_444 = func_87_call()
uop_446 = relay.sin(call_443.astype('float64')) # shape=(1, 16, 1)
uop_448 = relay.sin(call_444.astype('float64')) # shape=(1, 16, 1)
bop_458 = relay.logical_and(uop_446.astype('bool'), var_438.astype('bool')) # shape=(14, 16, 8)
bop_461 = relay.logical_and(uop_448.astype('bool'), var_438.astype('bool')) # shape=(14, 16, 8)
uop_465 = relay.sqrt(var_438.astype('float64')) # shape=(14, 16, 8)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_468 = func_254_call()
call_469 = func_254_call()
output = relay.Tuple([bop_439,bop_458,uop_465,call_468,])
output2 = relay.Tuple([bop_442,bop_461,uop_465,call_469,])
func_478 = relay.Function([var_438,], output)
mod['func_478'] = func_478
mod = relay.transform.InferType()(mod)
mutated_mod['func_478'] = func_478
mutated_mod = relay.transform.InferType()(mutated_mod)
var_479 = relay.var("var_479", dtype = "int8", shape = (14, 16, 8))#candidate|479|(14, 16, 8)|var|int8
func_478_call = mutated_mod.get_global_var('func_478')
call_480 = func_478_call(var_479)
output = call_480
func_481 = relay.Function([var_479], output)
mutated_mod['func_481'] = func_481
mutated_mod = relay.transform.InferType()(mutated_mod)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_519 = relay.TupleGetItem(func_408_call(), 0)
call_520 = relay.TupleGetItem(func_409_call(), 0)
output = relay.Tuple([call_519,])
output2 = relay.Tuple([call_520,])
func_542 = relay.Function([], output)
mod['func_542'] = func_542
mod = relay.transform.InferType()(mod)
mutated_mod['func_542'] = func_542
mutated_mod = relay.transform.InferType()(mutated_mod)
func_542_call = mutated_mod.get_global_var('func_542')
call_543 = func_542_call()
output = call_543
func_544 = relay.Function([], output)
mutated_mod['func_544'] = func_544
mutated_mod = relay.transform.InferType()(mutated_mod)
const_555 = relay.const([[[6.841568,-1.447632,1.518947,-4.267687,4.002342,1.754840,-9.119515,0.178947,-1.475306,-0.098971,1.112851],[2.365888,-5.933117,7.225717,-8.269646,5.751909,-8.021583,-1.940532,6.347002,-0.355264,-8.145034,6.270272],[-0.966760,-0.253350,9.441054,1.599169,8.541661,1.098598,9.604403,-5.717396,0.183702,2.151538,4.541291],[7.638722,-8.656271,8.393100,-6.098443,1.200416,5.513418,7.970905,-0.523941,4.987022,8.267486,2.428258],[-5.098980,-2.232425,-0.621857,7.119176,5.684646,-0.420079,-4.421362,-9.571838,7.462517,5.224910,-2.043682],[4.663121,6.270585,-0.323261,5.492505,7.810746,6.925144,-8.504185,-5.327920,-8.715590,-0.431230,0.510106],[-3.020333,2.511080,-9.875003,2.513952,-3.677319,5.898568,3.753311,-8.251881,-5.043392,-5.227525,-7.319896],[1.827771,-3.167177,-1.004033,-8.898375,5.590340,-8.030154,-1.618829,-2.160293,7.894068,4.158621,-0.773160],[-8.648034,2.878671,7.881560,8.373294,-8.372651,-2.573855,-2.456692,-8.071247,-9.762306,1.175784,-4.923403],[9.287100,6.130805,0.212000,-1.185626,-2.781162,4.315020,5.464684,5.587964,-4.200465,6.993616,7.694969],[0.856883,-6.694229,8.146177,-5.979844,-4.159716,-9.445428,-3.298652,-2.000615,-5.054938,-7.049905,-1.366438],[-0.875205,-2.463107,-1.764761,8.637530,9.262085,3.340960,-9.970575,4.421157,-5.444413,3.786608,-6.266317],[3.159013,-2.550881,1.444262,5.462226,-8.624428,5.927268,-6.859764,-1.144931,5.575611,-3.216283,6.759630],[-4.834771,-8.071112,-0.147607,1.511829,4.601028,9.908764,-3.723760,-8.019161,0.226977,4.299512,5.340108],[-4.448689,5.979791,7.292699,-7.604949,6.861016,-5.920522,0.388796,9.410720,2.555841,7.441884,-7.197651]],[[9.000527,8.865984,0.591832,3.293665,-3.882491,-3.842727,-8.558588,-3.893853,-1.278094,6.379977,5.479958],[-3.715119,9.765075,0.663362,-4.316204,3.005852,2.358536,-7.257586,9.889972,6.637428,4.620644,-0.683195],[-8.618032,-0.687382,-9.276069,1.350767,2.578964,-0.378069,0.295716,5.713813,3.165650,8.738867,-8.662411],[6.555851,3.926756,9.776883,-3.324599,-9.824701,-1.871396,3.369861,-7.931860,-5.855327,-1.894989,5.069221],[8.156770,-2.485779,4.897295,9.272726,6.212519,8.309232,0.234060,-2.260467,-4.142306,-3.895740,3.948655],[3.994778,-5.371020,6.424855,-2.856575,-8.784654,-9.015516,-4.124858,-4.620660,8.280636,-6.065317,9.684276],[2.775203,8.030407,-2.463834,2.729686,-4.304356,-2.482569,-2.736470,4.774157,-9.361557,-8.875484,-4.409106],[-1.955555,-3.145251,2.292280,-5.890662,8.893622,-3.830728,-8.826905,-2.555478,-3.646555,-0.813051,-7.799743],[4.367353,-5.941182,9.179910,-2.613604,1.033927,0.956706,-9.939697,5.041623,9.515999,-3.155317,4.060844],[-4.375227,-6.135298,-2.449902,1.326818,-5.621205,3.701419,-9.848054,9.652596,-2.384078,9.971874,6.869853],[-0.158571,9.058823,9.025934,0.328854,-1.300486,-5.387942,6.258890,5.381991,2.352288,3.542544,-4.490671],[-1.440612,-2.176632,9.666790,-5.163888,3.300079,4.910539,-7.116587,-4.070617,1.927232,1.024054,-3.147297],[8.646913,-7.213609,9.278311,3.271762,8.133668,1.318505,-2.487834,-7.898404,9.272503,-1.064986,-5.413421],[2.817202,-1.308413,4.178320,8.847236,-2.133451,7.729370,6.451093,-3.589471,4.193092,-3.143169,4.992493],[7.730566,9.204550,6.542658,3.736988,-6.002576,6.725480,6.078007,3.905779,-9.937191,0.217907,-3.152673]],[[-1.679095,9.247690,-9.691743,-3.434589,4.668184,9.896848,3.270500,-1.521101,-4.831657,4.012565,7.509359],[-0.682138,9.716003,3.425376,0.009826,-0.541226,-0.969511,9.993577,9.570034,8.223687,2.722466,-8.749327],[8.477190,4.209002,2.031668,4.668555,6.885344,1.906156,-5.697457,-8.011592,-0.744646,3.982985,-3.035924],[-5.995402,5.316617,-5.524886,-2.159753,5.997126,9.191913,-7.224987,2.373725,-7.177511,9.487558,-5.443693],[-1.430867,7.129060,8.817746,0.441581,1.436333,0.455849,0.391033,8.620613,-9.638720,-8.045793,8.219721],[2.845347,-5.733553,-4.562157,8.428744,-0.510275,1.765519,1.558664,-7.284538,-9.739869,-2.367059,-6.610311],[-5.905944,1.906577,-4.432589,-9.340636,-2.715512,3.827798,7.396274,3.883465,-0.411449,-1.723357,3.109542],[-7.563649,5.727641,-1.009478,-8.433558,9.219485,-1.604572,-4.803297,-6.655876,8.364517,-4.941945,-3.486620],[-9.207559,-3.840096,1.550171,-7.285921,-6.885661,-4.984316,3.794994,3.771353,-6.062903,-4.608531,-4.337636],[4.770575,-7.401826,0.107239,8.695103,-5.751601,5.667247,-4.809727,-1.562463,-5.020288,6.551778,-1.831389],[3.950059,9.577227,-0.327522,-1.044912,-3.745500,-2.188899,-9.857387,7.143519,1.001527,7.026326,-8.170144],[9.830593,-0.972725,-1.119577,-3.402983,3.973907,-1.506464,6.621251,-1.296105,1.683464,-0.953543,2.078271],[-5.990454,-4.531720,7.518269,-2.493032,-7.147135,-8.727089,9.267410,-0.890233,2.769329,8.567340,-3.803048],[-9.763570,7.459268,8.212595,8.897127,-4.619880,7.016992,4.184294,3.477096,3.991954,6.785953,-4.577177],[-0.852314,4.759534,-8.124539,8.854360,3.602715,-5.612934,-1.278727,-6.949914,-8.370030,1.787303,-6.430931]],[[-8.183623,-2.164862,0.237813,6.718061,6.938919,-3.345661,7.646348,1.572919,-0.460117,5.116081,8.765596],[-5.885965,5.673222,8.071725,5.162624,-9.961115,6.985694,0.031780,-9.399512,-7.125415,-8.890860,-4.217563],[-1.442563,-9.011777,1.681086,7.478885,9.817732,-7.244203,-4.339609,7.088515,5.614357,-9.036243,0.956851],[0.002882,-9.909836,4.920597,1.965304,4.038017,3.995291,-9.071189,4.124756,-5.117862,3.277029,1.791622],[-1.647093,9.993281,-0.328926,-9.488759,8.441064,-4.112760,9.237891,-0.156924,1.365579,-6.216430,-8.597752],[5.280744,-6.053036,-1.897394,0.965101,-7.312661,9.696655,1.693227,5.917161,4.649182,-4.635988,-2.194133],[9.205607,-3.381263,-7.769072,4.617938,5.614104,-2.246720,-9.596930,7.774075,0.037129,9.932642,-2.860126],[-6.137053,-6.172757,-2.316566,7.696930,0.246973,-3.161899,-4.685064,-4.387872,9.985732,-7.799923,-8.598084],[-2.996203,-6.409577,9.832511,-8.109322,7.393952,0.424632,-7.695148,-4.703837,-1.063727,-3.905041,2.658787],[-1.400698,4.860909,6.094742,9.377971,-6.216575,8.941907,7.918245,7.657462,-4.789474,3.304149,-6.277826],[-5.955811,0.989935,-1.396830,0.803515,-5.644627,-3.225551,4.118715,-7.310351,1.407624,-3.852246,6.542827],[9.401940,-7.312194,5.073239,2.544596,9.084525,-7.218707,-3.505099,2.280903,-1.797810,0.838913,4.130700],[-1.283032,-3.675436,-6.583120,2.689329,-2.637240,-9.789526,-9.478383,-4.360355,-8.137116,0.805209,3.017931],[5.329066,7.813821,-1.828971,-8.070019,7.006597,6.881050,-9.960326,9.865612,-9.613799,-0.566378,-6.156267],[-2.695270,-9.096319,-0.233175,-6.391567,-0.724766,-6.808881,7.975750,-4.040709,-4.284361,1.199909,3.969053]]], dtype = "float64")#candidate|555|(4, 15, 11)|const|float64
var_556 = relay.var("var_556", dtype = "float64", shape = (4, 15, 11))#candidate|556|(4, 15, 11)|var|float64
bop_557 = relay.divide(const_555.astype('float64'), relay.reshape(var_556.astype('float64'), relay.shape_of(const_555))) # shape=(4, 15, 11)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_563 = relay.TupleGetItem(func_408_call(), 0)
call_564 = relay.TupleGetItem(func_409_call(), 0)
uop_566 = relay.erf(var_556.astype('float32')) # shape=(4, 15, 11)
func_218_call = mod.get_global_var('func_218')
func_220_call = mutated_mod.get_global_var('func_220')
var_575 = relay.var("var_575", dtype = "int8", shape = (576,))#candidate|575|(576,)|var|int8
call_574 = relay.TupleGetItem(func_218_call(relay.reshape(var_575.astype('int8'), [6, 16, 6])), 3)
call_576 = relay.TupleGetItem(func_220_call(relay.reshape(var_575.astype('int8'), [6, 16, 6])), 3)
var_582 = relay.var("var_582", dtype = "float64", shape = (4, 15, 11))#candidate|582|(4, 15, 11)|var|float64
bop_583 = relay.bitwise_and(var_556.astype('int16'), relay.reshape(var_582.astype('int16'), relay.shape_of(var_556))) # shape=(4, 15, 11)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_586 = func_87_call()
call_587 = func_87_call()
output = relay.Tuple([bop_557,call_563,uop_566,call_574,var_575,bop_583,call_586,])
output2 = relay.Tuple([bop_557,call_564,uop_566,call_576,var_575,bop_583,call_587,])
func_592 = relay.Function([var_556,var_575,var_582,], output)
mod['func_592'] = func_592
mod = relay.transform.InferType()(mod)
var_593 = relay.var("var_593", dtype = "float64", shape = (4, 15, 11))#candidate|593|(4, 15, 11)|var|float64
var_594 = relay.var("var_594", dtype = "int8", shape = (576,))#candidate|594|(576,)|var|int8
var_595 = relay.var("var_595", dtype = "float64", shape = (4, 15, 11))#candidate|595|(4, 15, 11)|var|float64
output = func_592(var_593,var_594,var_595,)
func_596 = relay.Function([var_593,var_594,var_595,], output)
mutated_mod['func_596'] = func_596
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_683 = func_87_call()
call_684 = func_87_call()
func_542_call = mod.get_global_var('func_542')
func_544_call = mutated_mod.get_global_var('func_544')
call_691 = relay.TupleGetItem(func_542_call(), 0)
call_692 = relay.TupleGetItem(func_544_call(), 0)
output = relay.Tuple([call_683,call_691,])
output2 = relay.Tuple([call_684,call_692,])
func_695 = relay.Function([], output)
mod['func_695'] = func_695
mod = relay.transform.InferType()(mod)
output = func_695()
func_696 = relay.Function([], output)
mutated_mod['func_696'] = func_696
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_705 = relay.TupleGetItem(func_695_call(), 0)
call_706 = relay.TupleGetItem(func_696_call(), 0)
uop_709 = relay.asin(call_705.astype('float64')) # shape=(1, 16, 1)
uop_711 = relay.asin(call_706.astype('float64')) # shape=(1, 16, 1)
bop_729 = relay.divide(uop_709.astype('float64'), relay.reshape(call_705.astype('float64'), relay.shape_of(uop_709))) # shape=(1, 16, 1)
bop_732 = relay.divide(uop_711.astype('float64'), relay.reshape(call_706.astype('float64'), relay.shape_of(uop_711))) # shape=(1, 16, 1)
bop_734 = relay.less_equal(bop_729.astype('bool'), relay.reshape(uop_709.astype('bool'), relay.shape_of(bop_729))) # shape=(1, 16, 1)
bop_737 = relay.less_equal(bop_732.astype('bool'), relay.reshape(uop_711.astype('bool'), relay.shape_of(bop_732))) # shape=(1, 16, 1)
var_741 = relay.var("var_741", dtype = "int8", shape = (11, 16, 7))#candidate|741|(11, 16, 7)|var|int8
bop_742 = relay.minimum(call_705.astype('uint8'), var_741.astype('uint8')) # shape=(11, 16, 7)
bop_745 = relay.minimum(call_706.astype('uint8'), var_741.astype('uint8')) # shape=(11, 16, 7)
output = relay.Tuple([bop_734,bop_742,])
output2 = relay.Tuple([bop_737,bop_745,])
func_749 = relay.Function([var_741,], output)
mod['func_749'] = func_749
mod = relay.transform.InferType()(mod)
mutated_mod['func_749'] = func_749
mutated_mod = relay.transform.InferType()(mutated_mod)
var_750 = relay.var("var_750", dtype = "int8", shape = (11, 16, 7))#candidate|750|(11, 16, 7)|var|int8
func_749_call = mutated_mod.get_global_var('func_749')
call_751 = func_749_call(var_750)
output = call_751
func_752 = relay.Function([var_750], output)
mutated_mod['func_752'] = func_752
mutated_mod = relay.transform.InferType()(mutated_mod)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_777 = func_254_call()
call_778 = func_254_call()
func_592_call = mod.get_global_var('func_592')
func_596_call = mutated_mod.get_global_var('func_596')
var_789 = relay.var("var_789", dtype = "float64", shape = (1, 660))#candidate|789|(1, 660)|var|float64
var_790 = relay.var("var_790", dtype = "int8", shape = (576,))#candidate|790|(576,)|var|int8
call_788 = relay.TupleGetItem(func_592_call(relay.reshape(var_789.astype('float64'), [4, 15, 11]), relay.reshape(var_790.astype('int8'), [576,]), relay.reshape(var_789.astype('float64'), [4, 15, 11]), ), 3)
call_791 = relay.TupleGetItem(func_596_call(relay.reshape(var_789.astype('float64'), [4, 15, 11]), relay.reshape(var_790.astype('int8'), [576,]), relay.reshape(var_789.astype('float64'), [4, 15, 11]), ), 3)
uop_801 = relay.asin(var_790.astype('float64')) # shape=(576,)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_804 = relay.TupleGetItem(func_408_call(), 0)
call_805 = relay.TupleGetItem(func_409_call(), 0)
uop_806 = relay.exp(uop_801.astype('float32')) # shape=(576,)
bop_810 = relay.less_equal(call_804.astype('bool'), uop_806.astype('bool')) # shape=(1, 16, 576)
bop_813 = relay.less_equal(call_805.astype('bool'), uop_806.astype('bool')) # shape=(1, 16, 576)
output = relay.Tuple([call_777,call_788,var_789,bop_810,])
output2 = relay.Tuple([call_778,call_791,var_789,bop_813,])
func_819 = relay.Function([var_789,var_790,], output)
mod['func_819'] = func_819
mod = relay.transform.InferType()(mod)
mutated_mod['func_819'] = func_819
mutated_mod = relay.transform.InferType()(mutated_mod)
func_819_call = mutated_mod.get_global_var('func_819')
var_821 = relay.var("var_821", dtype = "float64", shape = (1, 660))#candidate|821|(1, 660)|var|float64
var_822 = relay.var("var_822", dtype = "int8", shape = (576,))#candidate|822|(576,)|var|int8
call_820 = func_819_call(var_821,var_822,)
output = call_820
func_823 = relay.Function([var_821,var_822,], output)
mutated_mod['func_823'] = func_823
mutated_mod = relay.transform.InferType()(mutated_mod)
func_542_call = mod.get_global_var('func_542')
func_544_call = mutated_mod.get_global_var('func_544')
call_856 = relay.TupleGetItem(func_542_call(), 0)
call_857 = relay.TupleGetItem(func_544_call(), 0)
output = call_856
output2 = call_857
func_861 = relay.Function([], output)
mod['func_861'] = func_861
mod = relay.transform.InferType()(mod)
mutated_mod['func_861'] = func_861
mutated_mod = relay.transform.InferType()(mutated_mod)
func_861_call = mutated_mod.get_global_var('func_861')
call_862 = func_861_call()
output = call_862
func_863 = relay.Function([], output)
mutated_mod['func_863'] = func_863
mutated_mod = relay.transform.InferType()(mutated_mod)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_873 = func_254_call()
call_874 = func_254_call()
output = call_873
output2 = call_874
func_878 = relay.Function([], output)
mod['func_878'] = func_878
mod = relay.transform.InferType()(mod)
mutated_mod['func_878'] = func_878
mutated_mod = relay.transform.InferType()(mutated_mod)
func_878_call = mutated_mod.get_global_var('func_878')
call_879 = func_878_call()
output = call_879
func_880 = relay.Function([], output)
mutated_mod['func_880'] = func_880
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_890 = func_87_call()
call_891 = func_87_call()
output = relay.Tuple([call_890,])
output2 = relay.Tuple([call_891,])
func_897 = relay.Function([], output)
mod['func_897'] = func_897
mod = relay.transform.InferType()(mod)
output = func_897()
func_898 = relay.Function([], output)
mutated_mod['func_898'] = func_898
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_905 = relay.TupleGetItem(func_695_call(), 1)
call_906 = relay.TupleGetItem(func_696_call(), 1)
output = call_905
output2 = call_906
func_907 = relay.Function([], output)
mod['func_907'] = func_907
mod = relay.transform.InferType()(mod)
mutated_mod['func_907'] = func_907
mutated_mod = relay.transform.InferType()(mutated_mod)
func_907_call = mutated_mod.get_global_var('func_907')
call_908 = func_907_call()
output = call_908
func_909 = relay.Function([], output)
mutated_mod['func_909'] = func_909
mutated_mod = relay.transform.InferType()(mutated_mod)
func_878_call = mod.get_global_var('func_878')
func_880_call = mutated_mod.get_global_var('func_880')
call_910 = func_878_call()
call_911 = func_878_call()
func_542_call = mod.get_global_var('func_542')
func_544_call = mutated_mod.get_global_var('func_544')
call_918 = relay.TupleGetItem(func_542_call(), 0)
call_919 = relay.TupleGetItem(func_544_call(), 0)
func_478_call = mod.get_global_var('func_478')
func_481_call = mutated_mod.get_global_var('func_481')
const_926 = relay.const([10,-9,1,-2,-9,2,9,-10,-2,-8,3,1,-4,-1,4,4,-1,8,-1,6,3,9,7,6,-10,6,-6,-3,7,-3,10,-6,-10,-5,-7,3,9,5,9,7,-8,5,2,-6,2,2,9,-2,-2,-2,-8,-2,6,4,1,-7,6,2,-1,-9,-8,-6,-8,-5,-5,-3,1,1,5,3,7,-4,3,3,4,9,-2,8,3,-2,10,3,-9,9,2,-9,-1,1,8,-9,5,-6,-3,7,-10,4,2,-3,4,6,-3,4,-7,7,-9,10,2,-1,10,1,-6,-2,-4,3,7,5,-6,2,-4,-1,-5,8,-6,-4,-8,-10,2,5,7,8,-7,3,-2,7,6,-2,10,10,-9,-4,-1,-1,6,-1,-4,-6,1,-4,-4,10,10,1,-2,-6,7,-9,-6,-1,6,7,10,3,-3,-4,-10,9,8,2,10,2,9,6,-4,10,-7,-2,-2,-8,-3,-2,-1,-5,-10,1,-4,-9,-6,10,10,-10,9,-5,8,-6,8,-2,-10,7,3,-1,-3,-9,3,-9,-9,1,-2,9,-10,8,2,8,-8,-1,3,5,-2,4,-2,-8,1,-1,6,10,-4,5,-10,-9,7,7,-1,7,5,-9,-4,1,6,8,-1,-9,9,-2,-6,7,-10,7,-7,5,6,10,-2,-5,1,10,4,-1,-3,3,-4,7,3,7,8,1,-5,6,-1,6,6,6,-7,-4,9,-4,-8,-9,4,-10,-8,-6,3,10,5,-6,4,-1,-10,-3,-2,4,-5,10,10,2,2,-3,-2,-1,-1,-7,10,7,-3,-10,7,-5,-10,-4,7,-8,-5,-8,-4,-5,-5,-1,2,-6,8,2,-2,8,3,2,-9,1,-1,-3,5,1,4,-6,10,1,-7,-3,-3,-1,-2,-6,-3,-5,7,1,-7,3,-4,8,-2,2,-6,-10,-6,3,6,-1,10,3,-5,9,-7,-5,-1,7,-4,3,-5,-6,6,3,3,-10,1,8,10,-1,-10,-1,1,-4,-8,4,-4,6,-2,-1,-2,-7,8,10,1,-6,10,-3,5,6,8,9,-2,1,4,10,5,-1,-1,1,-4,4,-4,5,-6,-3,10,-4,6,8,-9,8,-4,-9,1,10,6,-1,10,-10,-8,-1,-6,3,-2,4,9,-7,10,-9,-5,10,2,-4,7,1,8,4,-6,-5,-7,-10,-8,9,-3,3,-7,1,-2,4,-10,7,-6,9,-9,10,3,4,9,3,4,-1,-7,-2,7,10,-2,5,-9,2,9,6,4,-5,-5,-4,-2,-10,7,-10,10,4,-9,2,3,8,5,-6,-5,-3,-9,7,-2,-2,1,-5,-4,-4,8,-10,9,-5,-3,-5,-3,-9,5,-8,5,6,-4,6,2,10,6,-4,-8,-9,-1,-9,3,1,-4,-5,-10,4,3,-1,-1,-8,-1,-2,-1,-9,-5,9,6,4,-10,8,1,4,-3,-3,-8,-1,-1,-9,2,-7,8,-5,-5,-4,-9,4,2,7,5,-10,-9,1,2,4,-9,2,-6,-10,7,7,4,-9,-1,4,6,-8,-4,7,-10,3,4,5,-4,3,10,-1,-3,-5,-7,-10,-4,8,-7,6,-7,-7,6,-3,8,4,-9,-4,-7,7,-7,-5,-5,-9,-9,10,-3,9,10,10,9,7,2,10,-10,1,4,-6,-10,-1,-9,-7,9,-2,9,8,7,-7,7,-8,6,2,-9,9,3,-10,-3,3,-4,-1,-8,-2,2,-5,9,-6,5,7,8,3,7,4,-10,-2,-3,-9,1,5,5,4,3,10,-10,10,-4,9,-4,-3,-9,1,7,-2,-5,4,5,3,8,4,10,-5,-6,-6,-7,-7,6,-10,-9,-3,-7,2,5,-2,9,-2,-9,-3,-1,-5,10,7,-9,-8,9,-3,-8,-10,10,-5,-7,8,-2,-3,10,3,2,-3,-9,1,2,-4,-9,-1,-4,8,3,-9,6,10,-4,5,7,8,9,-1,3,1,-10,3,-1,7,-8,1,-2,2,-7,9,5,9,-7,6,8,4,-2,-3,-9,-8,-9,-2,1,-5,1,2,3,1,-2,4,4,7,6,-2,-8,-3,-1,-8,-9,1,-10,7,4,7,-4,-5,8,8,-10,8,-1,9,-6,-4,-1,-4,-8,6,8,8,-2,-2,-5,-4,-9,4,-1,-3,9,1,-2,3,-7,-4,4,-9,-6,-9,-9,5,5,-9,-4,5,-10,5,5,-4,1,-9,1,-3,9,8,-5,10,-4,-8,3,-4,-2,-7,-9,1,-10,-4,10,-9,-9,1,-2,6,1,-6,7,-9,1,8,-8,9,-1,2,2,2,-8,7,-5,-8,7,3,6,-5,-10,6,-1,-10,9,4,1,5,1,1,-3,4,-3,-3,1,-5,-10,10,7,4,10,-9,7,2,4,-6,2,-8,-9,5,7,-10,2,-10,10,7,3,-2,-10,8,10,2,4,-1,10,-2,8,6,3,-3,10,-2,-7,-3,3,-2,5,-6,-5,3,-3,-2,8,4,-6,2,-6,8,1,10,-5,6,8,1,7,9,-1,-2,1,-8,-3,-2,7,2,-3,3,-7,-8,-10,-6,-4,-8,8,-6,6,-3,-6,-7,6,2,-4,-5,-5,3,3,4,-6,6,2,2,-6,6,1,7,-5,-7,6,-9,5,6,-3,9,3,-5,7,-3,-4,5,6,-5,10,9,1,5,-10,-8,-1,6,2,9,1,-2,-4,-2,-8,-1,-6,4,-8,7,-2,-3,-2,9,10,-5,3,-8,6,-9,-7,-3,-1,-2,10,7,6,8,-6,-4,-6,-9,-6,-8,-7,7,7,6,-10,7,10,3,3,-4,8,-7,-8,-5,-3,9,10,-3,7,-3,10,7,5,7,-9,6,2,-2,-2,-5,10,5,9,6,-9,-10,-10,-10,7,10,-2,-3,-4,5,4,10,-3,5,-3,7,-10,1,-8,-5,-2,-10,3,-9,10,2,-3,1,7,9,7,-4,-6,6,-7,6,-6,-4,-10,8,4,-6,-4,4,-3,1,-10,8,9,-8,5,-8,2,1,-6,4,1,-8,8,-6,4,2,4,7,-6,-2,1,-3,5,-5,-5,10,10,10,-9,-9,-4,6,-3,-3,-10,-9,8,-3,5,3,-8,5,2,-6,-2,3,-7,5,-2,8,-5,2,-6,6,7,-1,-2,-3,-2,5,-5,1,7,-4,6,2,9,-5,-1,6,8,-4,-4,5,7,1,8,6,7,8,-9,5,6,-3,-10,-4,-8,1,7,-9,5,6,3,-3,-9,-6,-3,8,-10,6,-4,8,-1,6,-4,10,-7,7,8,10,2,10,6,3,-1,-8,9,6,7,-6,9,4,-4,-5,10,-10,3,8,-7,9,-9,-2,-5,8,2,-2,-2,6,7,6,1,10,7,-5,-8,-4,-3,5,7,-7,9,-10,8,-5,-7,2,9,4,-9,-2,-5,-6,-3,6,4,8,3,-4,-4,-5,2,-1,-9,-8,-3,-7,-7,7,-4,9,-6,-1,5,-1,-4,9,2,-8,-2,1,-9,4,-1,-5,5,3,2,-10,9,3,-8,1,2,-2,8,-10,2,-2,-6,-8,-4,2,-4,3,4,7,8,-3,-9,-5,-8,5,-10,3,-5,-6,-9,-5,1,-6,-3,8,-3,4,-5,1,5,-1,3,8,4,-7,2,-8,9,9,-9,5,-9,-3,9,6,-10,-6,3,-10,-10,2,10,-9,3,3,7,3,-9,10,2,2,5,2,-4,8,9,9,9,-8,-6,-6,1,8,-8,7,-6,-2,7,-10,-8,6,-8,7,-3,-4,8,-3,-1,-8,-1,1,-6,-10,-10,-5,-1,-8,-7,8,1,-10,-2,-9,-6,-4,-4,4,-1,-2,-7,9,-7,6,-3,-4,4,5,-7,-1,-1,-9,1,2,10,-10,10,-2,8,1,-2,-6,-6,-4,-5,-5,-1,1,10,3,9,6,5,-6,-7,4,9,6,1,10,9,2,10,2,-7,7,-8,-5,10,-4,8,6,8,-7,-6,-4,-2,4,10,-1,1,-4,2,10,3,-1,-3,-10,-1,-9,4,1,-10,10,-1,6,1,-8,9,7,5,-3,-6,-1,-2,3,4,3,-9,5,-4,-1,-2,9,9,3,7,2,1,-10,4,8,-9,5,4,9,10,6,-9,3,-5,1,2,-3,9,-5,-1,10,-4,10,-4,9,-9,5,-9,5,5,-6,-1,10,-4,-2,8,7,-8,1,5,7,-2,-3,3,7,-6,-4,1,9,-1,-2,-5,-2,-5,-7,10,4,6,-6,-7,-4,-5,-4,-6,-6,5,7,-7,5,9,-10,-3,2,-6,-6,-1,-8,3,-5,-4,-6,-8,-9,2,-5,-5,2,3,-9,2,2,8,-2,-1,-1,-7,-3,2,10,-7,6,1,-2,-2,5,-6,9,-6,-6,5,1,-4,9,-6,9,3,7,-3,-2,10,9,4,-4,9,9,-7,8,-5,-9,-5,6,4,-3,4,4,-3,-2,10,-9,-5,-3,-9,4,-7,6,-3,-9,8,6,7,8,2,9,5,3,8,-2,8,-10,-1,4,-1,-3,-4,2,8,3,9,3,2,4,1,-7,-6,-1,-5,2,-7,6,-1,-5,6,8,2,9,-7,-5,1,-9,-1,5,-7,-2,10,-8,-6,-2,2,8,-1,-9,-2,-9,6,-1,8,3,-2,2,4,-5,-4,3,-6,2,-9,-8,-5,7,6,-10,-3,2,9,7,5,-4,-8,-6,7,2,9,9,8,7,4,-8,10,4,5,3,-3,10,-3,6,6,-7], dtype = "int8")#candidate|926|(1792,)|const|int8
call_925 = relay.TupleGetItem(func_478_call(relay.reshape(const_926.astype('int8'), [14, 16, 8])), 0)
call_927 = relay.TupleGetItem(func_481_call(relay.reshape(const_926.astype('int8'), [14, 16, 8])), 0)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_931 = func_254_call()
call_932 = func_254_call()
func_819_call = mod.get_global_var('func_819')
func_823_call = mutated_mod.get_global_var('func_823')
var_946 = relay.var("var_946", dtype = "float64", shape = (660,))#candidate|946|(660,)|var|float64
var_947 = relay.var("var_947", dtype = "int8", shape = (576,))#candidate|947|(576,)|var|int8
call_945 = relay.TupleGetItem(func_819_call(relay.reshape(var_946.astype('float64'), [1, 660]), relay.reshape(var_947.astype('int8'), [576,]), ), 1)
call_948 = relay.TupleGetItem(func_823_call(relay.reshape(var_946.astype('float64'), [1, 660]), relay.reshape(var_947.astype('int8'), [576,]), ), 1)
uop_949 = relay.cosh(call_910.astype('float64')) # shape=(1, 16, 1)
uop_951 = relay.cosh(call_911.astype('float64')) # shape=(1, 16, 1)
func_861_call = mod.get_global_var('func_861')
func_863_call = mutated_mod.get_global_var('func_863')
call_959 = func_861_call()
call_960 = func_861_call()
bop_966 = relay.subtract(uop_949.astype('uint64'), const_926.astype('uint64')) # shape=(1, 16, 1792)
bop_969 = relay.subtract(uop_951.astype('uint64'), const_926.astype('uint64')) # shape=(1, 16, 1792)
bop_971 = relay.maximum(var_946.astype('int32'), uop_949.astype('int32')) # shape=(1, 16, 660)
bop_974 = relay.maximum(var_946.astype('int32'), uop_951.astype('int32')) # shape=(1, 16, 660)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
const_994 = relay.const([[-5.732190,-2.945826,-1.717453,-7.360665,7.554959,1.548414]], dtype = "float32")#candidate|994|(1, 6)|const|float32
call_993 = relay.TupleGetItem(func_123_call(relay.reshape(const_994.astype('float32'), [6, 1])), 0)
call_995 = relay.TupleGetItem(func_126_call(relay.reshape(const_994.astype('float32'), [6, 1])), 0)
uop_999 = relay.sinh(var_947.astype('float64')) # shape=(576,)
func_592_call = mod.get_global_var('func_592')
func_596_call = mutated_mod.get_global_var('func_596')
call_1006 = relay.TupleGetItem(func_592_call(relay.reshape(var_946.astype('float64'), [4, 15, 11]), relay.reshape(uop_999.astype('int8'), [576,]), relay.reshape(var_946.astype('float64'), [4, 15, 11]), ), 6)
call_1007 = relay.TupleGetItem(func_596_call(relay.reshape(var_946.astype('float64'), [4, 15, 11]), relay.reshape(uop_999.astype('int8'), [576,]), relay.reshape(var_946.astype('float64'), [4, 15, 11]), ), 6)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_1010 = func_87_call()
call_1011 = func_87_call()
uop_1015 = relay.atanh(uop_999.astype('float32')) # shape=(576,)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_1026 = func_254_call()
call_1027 = func_254_call()
uop_1031 = relay.log2(call_925.astype('float32')) # shape=(14, 16, 8)
uop_1033 = relay.log2(call_927.astype('float32')) # shape=(14, 16, 8)
func_396_call = mod.get_global_var('func_396')
func_399_call = mutated_mod.get_global_var('func_399')
const_1047 = relay.const([-7.555678,-6.513088,2.363192,-3.508577,6.272925,-5.862350,4.499363,3.333771,-0.200388,-1.296513,-6.410530,-6.622348,-6.444659,-6.127105,5.042134,2.553980,-1.475370,0.113592,0.627498,8.451447,-3.853711,0.437154,-2.127387,-0.975992,-2.652251,-6.703095,7.772566,6.735410,-6.519756,8.283936,3.935059,2.337100,-0.546521,7.660974,-8.504443,-8.406837,7.069069,7.544847,6.203671,-8.069540,-9.972337,-5.610149,-8.496383,3.124435,1.165689,-5.973295,2.991112,-8.553919,-2.094151,4.575990,-8.135929,4.941084,2.394808,-5.369497,-0.859632,2.657570,8.081467,-7.879879,1.432030,3.552462,8.591874,8.122279,9.607454,-9.223666,1.515763,5.632520,-7.096249,0.436397,5.777622,-5.323897,6.776374,-7.822591,6.116084,-2.587346,-5.027118,6.197348,5.183744,-8.427873,-4.311274,-4.174763,-1.774037,9.692141,4.926203,8.311394,-3.150376,-4.997117,0.093582,3.103951,-3.573639,0.073178,3.383518,-7.865450,6.559590,0.120940,9.008483,6.988061,0.148500,-8.640008,-7.523367,4.236666,0.039798,-2.832414,-4.735870,7.918460,2.070177,4.067481,9.756899,2.020593,-8.736437,-3.042194,-5.127260,-8.606594,3.304748,-4.472564,-1.139151,-4.637407,-9.522689,-2.978124,-5.235255,4.505295,-8.518747,-9.633746,-5.808622,-1.538712,7.491817,-2.591803,-0.620885,9.282925,-5.461004,9.564376,-9.906039,-7.615099,-9.722977,7.717698,5.147830,-5.694563,-9.282297,5.849639,-9.035861,2.662721,9.392027,-5.683381,-8.632868,2.421724,9.517607,6.754448,-4.180512,-3.735974,-5.395785,-9.716999,-7.076157,4.950782,8.645826,1.069016,-3.726389,9.361111,6.383400,-6.596550,-7.939952,-7.043246,-7.560853,0.581806,0.989566,-2.316032,4.434584,9.000487,7.143610,-1.068195,-3.577343,4.882611,6.149601,-1.399828,-3.817312,-8.140724,-4.795200,-2.867095,9.137292,4.133397,-0.940929,-9.216883,-3.956416,3.060640,3.234496,1.185510,-5.805686,-3.586025,9.753498,-6.311646,4.963418,-1.326134,0.851269,8.454531,-7.724890,1.188942,-5.341084,1.498701,-9.873256,-8.975508,-9.489682,5.272623,-9.364233,-7.595181,-8.886490,5.103405,-9.440786,2.783612,7.386654,-8.957877,0.232957,-9.552710,6.364937,9.069479,-6.314041,8.527564,1.645927,6.942617,4.158499,2.947996,-0.331031,9.367319,6.884802,-9.898133,8.885007,-1.447003,8.438883,-5.518636,-6.236365,-8.948136,-7.805541,-0.905090,-2.504231,3.563363,4.456074,7.428967,-5.516250,-9.700154,9.524521,-0.839080,4.184797,9.641079,5.502526,-1.450791,5.011124,-5.605598,-2.841628,4.004723,6.071758,-0.764162,-8.481242,-4.363653,2.588367,-7.289883,-5.442911,9.414265,4.660870,-2.463539,0.389451,1.034769,1.964159,-9.628604,-7.659189,-3.733488,-6.127247,-2.368022,-4.161955,5.720081,-5.012650,-7.772723,-8.862594,-2.975728,-8.383169,-2.971154,1.460132,3.344259,-6.003469,6.519912,5.875941,-9.089842,-0.894688,-5.841931,6.472370,-8.411068,4.772665,-4.432038,7.429018,-3.229373,9.714520,2.952986,-5.321364,0.356390,-4.764340,0.239325,-8.299391,2.893989,-5.863082,6.912520,-0.456401,7.147545,-5.404426,4.440758,8.767781,-8.542811,7.958447,7.278504,-9.174025,-5.739381,1.690730,-0.316672,8.748272,3.550170,-8.116231,7.083186,-7.818148,-9.825158,7.918938,-4.412599,0.160618,2.138576,-3.763713,-7.455632,-1.129662,9.686830,-0.814635,5.885644,-7.053342,-9.628199,-2.634296,7.494901,-5.786952,8.954746,6.586575,-7.968234,-9.388493,3.011157,7.974193,9.104572,-0.018626,2.299552,7.950187,-5.863098,7.427628,-2.157177,5.207132,0.485630,4.847818,8.985057,-4.466678,-1.417755,8.464463,-8.745997,8.485573,-6.745482,-7.760173,8.842001,3.449997,0.695330,-7.638367,-7.412336,1.575254,3.674867,-2.277966,-7.605371,0.020709,-3.293793,-8.182500,8.563049,-7.531484,-7.590090,3.705179,-5.664470,2.665499,-7.993891,5.480285,8.941772,-7.551905,-4.483259,4.578239,9.982116,5.437761,-7.315485,-5.992230,7.898685,2.107438,-8.879745,1.647032,2.177483,-5.772342,-5.613947,-9.562164,6.050650,2.071989,-9.903249,-9.982327,9.020538,2.130328,2.116255,9.181527,-5.779006,-2.542084,0.230206,-6.850516,3.483984,-7.246474,-7.301752,5.537408,7.439807,6.460109,-7.015570,-9.569503,1.998907,5.769256,0.730000,-7.326074,9.868657,-6.927073,-7.232781,-4.342855,-4.699358,-5.381340,5.824582,-9.767289,-2.341639,-1.607180,1.082095,-7.795486,-8.743286,2.884949,-6.567527,-8.803901,0.691513,-6.919249,9.433894,3.203916,-6.992486,-3.848252,3.974753,-2.309504,7.732903,-4.940740,2.306464,9.476094,-8.322853,-6.005686,-1.385626,2.152417,7.427780,-9.158446,-6.843208,-1.773896,9.218749,-0.955841,5.898874,-2.181708,-4.635717,7.362994,0.457350,2.246817,-2.175257,-5.239886,-1.007651,-1.359408,5.027163,-5.072925,-7.098553,-9.632136,-9.465059,-6.824968,-0.685963,-6.326642,-5.783197,-6.715721,-7.682690,0.885345,4.334240,-8.286256,1.865853,-4.492382,2.897969,-1.101440,9.458353,2.543393,-1.796402,-3.918997,-5.928686,5.799697,-1.717542,-8.587900,7.329381,8.255537,2.013131,9.893958,-7.973487,0.345945,-4.273545,3.006812,3.540851,2.079547,3.857637,-2.641606,-1.549458,-0.559385,3.703633,-4.114088,-9.658178,-2.471551,4.753194,3.673446,7.994349,4.562247,-5.257697,9.253734,8.816026,3.286913,-7.719509,-7.648857,0.952295,-2.588465,1.318997,-8.013583,-9.317964,1.129643,-1.346445,-7.775707,7.530807,7.056303,-9.382355,8.233585,-2.753294,-3.308571,-2.372939,3.315517,0.009201,2.245027,-5.579331,6.788482,-3.985967,-4.216756,7.139417,-2.323600,-4.687578,-9.007201,-4.000155,-5.947038,7.464341,-1.551058,0.539315,4.364475,1.481719,4.039214,-7.509852,-7.730472,6.807268,5.464807,5.532046,-6.269452,-7.784885,4.355509,-3.343518,0.448946,-0.027915,-6.842072,1.350436,9.116111,-7.047105,8.166455,1.010235,7.140883,-5.104937,6.912989,-8.621176,-0.137471,-4.593255,6.622538,7.543222,-3.413134,-0.660588,-2.681629,0.162872,-1.596279,-2.279117,-8.611296,-0.578947,1.889320,6.731276,-0.125557,-3.605585,-6.839647,-8.459735,3.965352,-6.299518,-7.980726,-5.892214,-5.725053,-8.648918,9.976419,-1.219499,-5.039690,9.852408,-3.152056,-3.808271,-8.091249,2.067152,6.075234,8.409207,4.138487,2.974734,-9.184743,-9.589703,6.412961,-1.327323,-7.080773,-9.982662,0.812452,-0.331724,-9.190230,-0.141524,7.681962,5.675777,-6.995738,3.101386,-1.602559,3.997018,4.925989,-5.598982,-7.814996,9.186415,9.743896,8.666850,-8.650783,-3.982992,-5.143330,8.876468,-8.387953,-9.215434,2.855957,7.249217,-6.700479,9.298437,-7.834938,-1.834089], dtype = "float64")#candidate|1047|(640,)|const|float64
call_1046 = relay.TupleGetItem(func_396_call(relay.reshape(const_1047.astype('float64'), [4, 16, 10])), 0)
call_1048 = relay.TupleGetItem(func_399_call(relay.reshape(const_1047.astype('float64'), [4, 16, 10])), 0)
uop_1055 = relay.asin(var_946.astype('float64')) # shape=(660,)
func_218_call = mod.get_global_var('func_218')
func_220_call = mutated_mod.get_global_var('func_220')
call_1075 = relay.TupleGetItem(func_218_call(relay.reshape(var_947.astype('int8'), [6, 16, 6])), 3)
call_1076 = relay.TupleGetItem(func_220_call(relay.reshape(var_947.astype('int8'), [6, 16, 6])), 3)
var_1080 = relay.var("var_1080", dtype = "int8", shape = (3, 16, 12))#candidate|1080|(3, 16, 12)|var|int8
bop_1081 = relay.minimum(call_1075.astype('uint16'), var_1080.astype('uint16')) # shape=(3, 16, 12)
bop_1084 = relay.minimum(call_1076.astype('uint16'), var_1080.astype('uint16')) # shape=(3, 16, 12)
output = relay.Tuple([call_918,call_931,call_945,call_959,bop_966,bop_971,call_993,const_994,call_1006,call_1010,uop_1015,call_1026,uop_1031,call_1046,const_1047,uop_1055,bop_1081,])
output2 = relay.Tuple([call_919,call_932,call_948,call_960,bop_969,bop_974,call_995,const_994,call_1007,call_1011,uop_1015,call_1027,uop_1033,call_1048,const_1047,uop_1055,bop_1084,])
func_1093 = relay.Function([var_946,var_947,var_1080,], output)
mod['func_1093'] = func_1093
mod = relay.transform.InferType()(mod)
mutated_mod['func_1093'] = func_1093
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1093_call = mutated_mod.get_global_var('func_1093')
var_1095 = relay.var("var_1095", dtype = "float64", shape = (660,))#candidate|1095|(660,)|var|float64
var_1096 = relay.var("var_1096", dtype = "int8", shape = (576,))#candidate|1096|(576,)|var|int8
var_1097 = relay.var("var_1097", dtype = "int8", shape = (3, 16, 12))#candidate|1097|(3, 16, 12)|var|int8
call_1094 = func_1093_call(var_1095,var_1096,var_1097,)
output = call_1094
func_1098 = relay.Function([var_1095,var_1096,var_1097,], output)
mutated_mod['func_1098'] = func_1098
mutated_mod = relay.transform.InferType()(mutated_mod)
func_878_call = mod.get_global_var('func_878')
func_880_call = mutated_mod.get_global_var('func_880')
call_1173 = func_878_call()
call_1174 = func_878_call()
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1177 = relay.TupleGetItem(func_695_call(), 1)
call_1178 = relay.TupleGetItem(func_696_call(), 1)
bop_1179 = relay.power(call_1173.astype('float64'), relay.reshape(call_1177.astype('float64'), relay.shape_of(call_1173))) # shape=(1, 16, 1)
bop_1182 = relay.power(call_1174.astype('float64'), relay.reshape(call_1178.astype('float64'), relay.shape_of(call_1174))) # shape=(1, 16, 1)
func_907_call = mod.get_global_var('func_907')
func_909_call = mutated_mod.get_global_var('func_909')
call_1186 = func_907_call()
call_1187 = func_907_call()
bop_1190 = relay.bitwise_or(call_1186.astype('int32'), relay.reshape(call_1177.astype('int32'), relay.shape_of(call_1186))) # shape=(1, 16, 1)
bop_1193 = relay.bitwise_or(call_1187.astype('int32'), relay.reshape(call_1178.astype('int32'), relay.shape_of(call_1187))) # shape=(1, 16, 1)
func_861_call = mod.get_global_var('func_861')
func_863_call = mutated_mod.get_global_var('func_863')
call_1194 = func_861_call()
call_1195 = func_861_call()
output = relay.Tuple([bop_1179,bop_1190,call_1194,])
output2 = relay.Tuple([bop_1182,bop_1193,call_1195,])
func_1197 = relay.Function([], output)
mod['func_1197'] = func_1197
mod = relay.transform.InferType()(mod)
mutated_mod['func_1197'] = func_1197
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1198 = func_1197_call()
output = call_1198
func_1199 = relay.Function([], output)
mutated_mod['func_1199'] = func_1199
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1199_call = mutated_mod.get_global_var('func_1199')
call_1285 = relay.TupleGetItem(func_1197_call(), 0)
call_1286 = relay.TupleGetItem(func_1199_call(), 0)
const_1287 = relay.const([[[3.843847,-3.989809,4.934081,-5.139531,-6.684674,6.759227,-1.375184,0.927902,4.824155],[-0.503280,5.093657,5.390887,9.367256,-2.543157,-4.017634,9.431281,-9.096266,2.199087],[-4.061271,-4.291322,-0.828460,7.566662,6.345465,-2.879084,-5.434002,2.004090,-1.240667],[-0.083532,-9.035428,-6.144293,-8.858748,3.177709,8.664026,-5.083117,-8.804611,5.564793],[1.225523,1.093038,0.995964,9.615680,4.869662,8.008019,-5.019994,9.125680,-2.272567],[1.984816,-4.385689,5.314817,0.974484,0.975568,-2.354247,1.622464,-5.904233,-4.046071],[-1.674777,3.894147,9.862875,6.326404,-5.226093,-2.953341,-3.436755,3.506869,-0.649909],[9.844614,-3.264587,-1.037485,2.416277,-6.982933,-3.291539,9.995173,2.682199,-5.530555],[7.132416,-8.867771,5.766105,-7.135604,-5.341819,-1.392861,-8.806146,5.998386,-3.925075],[-2.190471,2.461871,1.202287,6.722303,-5.794094,1.334311,0.799244,3.589170,-2.808551],[6.758181,-0.071187,-3.066835,-3.409076,-4.190801,-0.461665,7.563604,-8.221889,0.410524],[5.964624,-4.598488,4.830295,-8.480269,-5.027150,-7.835664,-6.696050,8.358803,-3.111440],[5.523010,5.389676,-9.737774,4.867606,-5.061210,-0.796415,7.872415,-4.818482,2.100683],[-5.644197,5.254814,8.785446,0.721604,4.384238,1.751183,5.210407,-8.333524,-8.777184],[-9.753947,0.643058,-5.506442,-4.891958,9.357170,3.183398,5.679433,-3.315660,1.464780],[-9.684520,-0.590987,6.738862,8.715447,-0.137995,2.720664,-3.083037,-1.825676,-6.967471]],[[6.680326,5.797720,-8.596366,-8.160697,-0.206001,-2.294391,-9.183538,-6.588976,-7.137026],[-0.749821,-6.834882,8.439767,-1.620473,5.831156,9.280235,0.732047,8.511012,3.167356],[8.573236,8.185976,6.863264,6.638063,-5.884102,-6.493942,-4.326195,7.081334,5.697022],[-4.385170,-9.831383,1.558135,-7.699803,1.329607,7.396916,6.393304,-0.823289,-5.440348],[-0.658740,1.815472,5.806691,9.944496,9.126177,4.975076,6.752717,-3.172852,8.009147],[-9.996076,-7.066368,4.231216,-3.104023,-5.398576,6.143417,-7.021412,5.218730,9.787383],[3.719636,-2.677455,6.348946,-8.238334,5.295433,-3.184938,-2.342925,-5.568828,-1.188683],[-3.252615,-3.690040,6.026468,3.903383,0.401180,-7.453979,2.191637,2.675494,-6.439282],[1.031125,-7.955707,6.357817,-2.737233,4.977789,-7.128310,-2.631967,-7.542855,3.291931],[4.977113,-5.077631,2.053497,5.713079,-6.785341,-6.801823,9.050107,3.747213,7.594556],[-3.105227,-5.757651,-1.084021,-0.818559,1.637598,-2.243767,-0.045172,0.442500,-6.750060],[0.931942,-3.992498,6.268911,-9.726514,-4.373078,-8.997217,-0.583437,9.390819,-1.350383],[3.315714,2.043238,2.540751,-8.606676,0.065922,-8.443638,-8.098596,6.908091,-9.669380],[-9.732489,2.840926,-4.139956,-7.664180,-2.586934,-9.956939,1.994852,4.531854,-2.227423],[-9.576279,2.214598,8.587298,-6.235217,5.106933,-0.737379,-8.238088,3.823126,9.650731],[5.877442,-5.546051,-2.782125,-5.758816,4.311627,-5.047378,-4.782108,-6.640486,-9.651252]],[[-3.201398,4.677906,-0.924483,3.784253,-2.078697,-0.920758,-1.368680,7.331393,0.441351],[1.677546,-3.580887,-0.498972,3.823651,4.260350,-0.948305,8.485975,-5.026506,-3.238933],[-8.059381,6.363897,-5.458045,-0.948684,4.993442,-0.936313,7.201476,-6.430945,-8.857695],[-5.910058,-8.980216,3.909548,4.259703,9.601700,8.196453,-3.137946,8.863317,5.313622],[0.060253,-8.049578,2.374248,-7.592600,1.430698,-4.337503,9.078308,7.546793,3.577745],[-2.195014,6.749421,-3.081228,9.728431,-5.113812,0.980612,5.624700,1.112364,1.608998],[-8.138786,-2.684225,2.852220,3.116944,1.493765,-3.662342,3.204576,-2.447823,-0.897702],[5.930643,5.001266,-8.325886,0.860749,-8.775422,9.315407,-3.098758,8.723782,9.533978],[1.823888,4.361019,0.728505,1.615827,-3.623146,7.703898,8.752406,-6.803378,-3.616892],[6.712004,-3.735526,3.967481,1.243761,2.575524,-0.080588,0.405667,1.816017,-3.713891],[-7.515692,9.742651,7.406369,1.282323,-0.250022,-4.028055,-3.842098,-3.528407,4.477272],[3.461082,8.693998,8.192976,0.995336,-2.155605,2.684471,-3.745633,-1.135742,-6.760747],[-7.908985,-9.190064,2.621085,-9.811939,-3.379441,9.045613,-7.777482,3.174048,9.370367],[6.170728,3.526286,-3.926676,-4.819664,5.901452,-8.686260,1.649674,0.245458,5.396830],[-6.485983,8.448837,6.424400,-1.146941,7.007295,3.507029,-1.792390,6.682019,-5.229940],[-5.487335,-2.927413,-3.991885,2.408622,-2.249666,5.647662,6.057078,9.611449,4.811892]],[[2.184296,-7.253075,1.571945,-1.419818,-1.320949,9.239971,6.109176,1.652809,6.965638],[-1.360893,3.303153,-4.558805,1.246523,-2.153533,-7.194864,6.340892,6.179285,-8.729279],[6.317961,-2.739444,-6.460329,5.395408,9.944918,-7.530143,-3.451779,2.228090,7.616693],[-2.537336,-7.363444,5.000955,-4.720898,-1.116707,-6.014996,-5.971105,0.425749,5.081807],[6.138823,6.452903,9.660790,5.100176,1.108424,-6.583881,-0.505704,8.227112,2.426142],[3.347032,2.280870,-2.459019,-6.231337,5.262091,1.273859,-0.402899,4.197123,-2.232743],[2.528770,-5.146525,-3.157410,2.589404,-9.530098,6.208706,8.155105,8.007899,7.370680],[-4.571094,4.983999,7.465428,-9.144814,8.857678,-1.813002,-7.740549,8.600676,-2.589324],[1.991578,5.802023,-1.414412,-7.297539,7.328659,-8.835020,5.508430,0.802585,-5.559574],[8.080141,-2.778481,0.949780,-7.295777,-8.828319,-8.567402,-0.829177,-0.266633,2.965039],[-4.036168,-3.182007,-1.055231,-6.003473,5.073842,-5.494567,5.665967,-9.536874,-0.456819],[-7.723884,9.484663,-3.833069,8.427482,-8.084903,-3.614787,-7.738534,5.903266,8.537657],[7.345482,6.974375,3.202650,-7.126638,4.527569,-0.763831,-1.399014,3.410135,8.353289],[-6.955702,-7.438422,-4.341833,-0.539554,2.819818,7.287037,9.968617,8.905226,6.647523],[-7.421585,4.759239,-2.810756,2.294328,-4.163387,-1.924443,5.488457,-9.958862,0.425471],[-5.043446,7.724533,3.310086,-1.280650,1.102576,9.522415,-0.681300,-9.050535,-4.618782]],[[9.049800,-0.197586,-4.981279,5.026267,6.721349,1.790872,-2.194790,-3.508993,4.620607],[-5.627014,2.376436,0.090002,-8.946239,-5.073229,8.267642,9.948829,2.241826,-1.576488],[6.569633,6.028954,1.276276,3.670921,-0.744858,6.750427,-0.649011,-5.122019,3.338814],[-7.054706,7.247778,1.371019,6.747310,-6.402497,-8.251460,-2.642015,-5.660129,-5.536092],[-8.819222,-9.277150,3.540512,9.347489,5.588112,5.864998,8.881288,7.079704,6.126052],[-4.983286,5.918509,1.161062,9.734090,8.231001,3.651197,-6.425453,-9.475419,-7.399905],[-0.264476,7.407701,5.549062,9.663557,-9.373224,-0.988033,9.282338,2.471570,4.472616],[-6.738748,-4.729177,-6.895663,7.810079,-8.157454,3.792669,7.141890,-7.630363,-2.917245],[-6.832887,-6.893616,-6.249828,-6.538806,4.275181,-5.364479,-5.768830,-7.445275,1.432199],[-0.513013,3.601003,-2.926281,5.197642,-5.747386,-5.669900,-1.806217,-9.188470,7.561949],[8.854162,-0.059578,5.339683,7.319155,0.718824,2.137565,-2.839540,0.582693,-4.587682],[8.215283,-7.675549,8.171804,-4.384948,6.427094,-8.929573,1.638475,-2.542136,3.917935],[-8.822743,2.577584,-9.871413,-6.420103,-9.437459,-4.288786,3.062038,-5.212048,8.638254],[-3.768252,6.155834,6.713856,4.461468,5.589485,-1.399246,1.265793,1.385878,0.749297],[1.118858,2.036941,0.321861,-9.579684,1.053771,-6.673450,-3.158330,-7.727519,-9.699346],[-3.021319,-2.620593,8.519102,-9.213850,6.761557,3.688638,5.430507,1.501782,-9.239238]],[[-7.727912,7.554094,-3.782487,-1.678198,4.373364,-2.817323,-1.354250,1.909440,3.248193],[-1.273726,8.344671,4.489367,-5.160750,-0.155303,2.006875,-5.123273,5.072251,1.656240],[-2.272938,7.659921,-9.329272,9.858041,-6.439120,7.242798,-7.537665,3.567967,-6.075973],[-1.056425,0.133103,-3.317061,-0.959384,-8.677833,-7.106719,1.740847,4.524090,-8.396664],[-1.539605,7.032276,-5.005635,-6.298494,-0.326859,6.257444,6.351126,7.856392,-0.630380],[0.771230,-2.660077,-9.964743,-6.590549,2.256231,-8.114345,4.004351,-0.612445,2.560225],[-2.425689,7.556598,1.908878,0.419274,-4.660938,5.425976,-7.927957,8.002148,-1.621427],[1.619061,3.078816,6.669792,2.737858,5.388668,2.905123,-0.314155,-4.963697,-0.495337],[-5.328417,-8.167878,-8.129020,8.399426,0.907440,6.367144,4.469173,-4.142389,-9.954650],[-9.256093,-1.382615,5.044699,4.758453,7.917363,9.044832,6.300556,6.427898,-9.614593],[-2.806730,-1.761815,2.912368,5.150137,0.851457,6.409485,7.347758,0.309798,9.946692],[4.834751,-1.325523,1.082773,-4.444230,-4.091014,-2.604328,6.020167,-6.957191,-9.920213],[-2.859382,-5.283511,7.658099,0.306558,-3.414681,8.414321,7.956778,-9.347212,3.371947],[2.075860,-1.390287,-1.025115,-5.502698,6.087513,8.175485,-8.474834,-8.100324,5.211029],[8.596341,-9.805718,2.670501,3.467046,2.268811,9.098735,2.472483,-6.007809,-3.090190],[4.043382,3.788526,4.095762,-1.892084,-5.170724,8.185106,9.848657,-6.764623,-3.124007]],[[0.539524,-4.593934,2.595169,6.988021,-9.042223,-6.266583,-3.126387,7.594765,2.046148],[7.030711,9.006593,-1.359235,8.136303,4.801900,7.094809,-3.025901,0.215176,4.479665],[-9.715117,-7.898212,9.095877,5.218393,-0.994044,1.307638,2.680438,9.923800,-6.218930],[0.528517,0.382184,3.074339,-2.987078,5.736858,-1.239633,-0.839876,-7.356559,2.699511],[-2.303480,7.012996,9.268463,-3.232563,5.316815,1.789863,1.377517,-8.489706,4.124495],[9.622569,7.272530,-3.308022,-5.963277,-6.559220,6.975231,-4.683685,-6.126074,-5.019481],[-6.306711,-7.470039,-0.073725,0.626971,3.584666,6.155263,4.323740,-3.818875,1.680943],[-1.748011,-8.245050,-2.677128,-8.460258,5.806388,6.754699,-8.127748,-5.185451,1.367575],[4.842172,-0.281476,-8.499146,0.816105,-9.117569,-0.560438,-6.969184,5.051020,2.139215],[-5.930648,-3.845154,-9.703697,3.590191,1.849376,-6.445898,0.882924,5.626581,-5.988003],[0.416470,0.338268,1.880191,4.682213,-2.893213,-2.795043,5.440728,6.382055,-4.951402],[9.734839,1.825287,1.994791,1.103701,2.443290,-9.132054,1.878148,2.023600,-4.687427],[-6.762269,8.655471,-4.664941,5.523036,-6.549130,-7.646125,-9.232511,3.343537,-4.140151],[7.632331,2.762296,0.307534,-8.830897,9.836590,1.577585,6.605384,7.892216,7.003164],[-1.268025,5.436159,4.841441,-0.127823,0.806165,0.127859,-2.415016,-2.590677,8.342459],[-7.035418,-4.466109,-9.108364,7.226519,8.114820,-4.515552,-8.492489,6.730897,3.310665]],[[4.319919,9.870109,4.376459,4.373374,0.945346,6.551847,7.641836,8.040797,6.635466],[7.060539,8.080599,-3.759299,5.307489,-7.089487,-5.159223,-2.458214,5.971078,3.003231],[-4.279105,7.248827,-5.697443,1.983494,-8.540335,8.847122,-6.910507,-5.234381,8.029835],[6.554377,9.682811,-2.463558,8.147417,-5.677050,-8.043847,-0.322754,-6.107626,-2.098442],[-1.261595,-1.645954,-6.909922,0.026881,-3.536302,-5.960376,-4.136257,6.964509,-1.801807],[8.871196,-8.164996,-6.396465,4.376470,8.819446,-3.352778,-1.296454,9.737370,8.915237],[-0.425855,-0.596009,-5.100405,6.815524,-0.062132,4.623421,-4.346354,3.685737,-0.294499],[9.544221,-2.845858,7.868080,-6.186132,4.997867,-3.195471,-1.936848,0.216845,0.015987],[8.856070,-9.834512,9.282708,-7.176822,0.047130,-9.894605,-8.069399,5.295096,1.248432],[9.470085,4.452195,6.794981,-9.468496,-2.474256,2.981604,8.239998,-9.062778,-6.421164],[7.709057,4.748349,-9.159155,-1.412295,7.752255,2.691468,-8.338292,3.500230,-7.143410],[2.278352,-9.139661,-6.066422,-0.173601,-7.353632,-0.321060,-6.377833,-7.722260,-0.429200],[-7.217926,0.024102,-2.970238,-1.302999,0.021544,-9.582800,-2.633815,-4.256475,-3.864519],[-1.605526,8.534552,2.377746,-5.748377,3.697681,-3.417374,1.821135,8.713408,-6.927693],[-5.311348,-8.684351,-6.731866,2.869590,1.177071,1.313737,-2.437717,-7.283837,2.101083],[5.250889,0.648703,-9.040739,-6.199205,-3.717166,-8.189205,-7.506557,3.805379,-9.166890]],[[8.764699,8.917222,-8.113834,-4.484902,-3.572802,-9.809561,7.318802,7.594217,-4.667767],[-2.403146,-5.927459,-4.999062,2.220467,1.548774,-0.826409,5.243318,0.577030,-8.019542],[8.596526,-5.832871,2.771240,5.848826,9.273018,7.622387,-1.607166,-2.436483,2.000565],[5.974526,1.979955,4.807064,1.171136,-3.192069,5.714346,-5.616249,5.398640,0.058061],[3.196393,1.913785,3.056647,-5.495340,-2.479330,-7.549325,-6.835249,4.694718,-6.662302],[2.895924,7.291216,-0.940255,-2.967688,9.858365,-0.815051,-9.538742,-7.453832,-6.665318],[-9.997720,6.161026,1.942629,-1.815882,-2.153662,-7.390684,4.115392,-9.454324,8.717042],[9.858025,3.186043,8.986810,6.149748,7.268922,-2.557216,-7.215906,-7.466174,8.535763],[4.342386,1.910616,7.212261,9.164922,5.390600,4.804910,-3.703269,-7.490592,-3.099117],[0.696472,6.676745,7.687682,-0.503989,6.183706,4.180935,7.957911,-0.316286,-4.708034],[-8.389692,-9.349149,6.500330,-6.269178,8.600665,4.929420,-5.760906,1.488804,0.937942],[2.294181,-9.662046,7.659976,-4.072187,8.461350,2.256781,-9.457327,8.704241,-4.158666],[1.541699,-6.109897,0.082762,-9.433541,4.954927,-2.460833,-3.462248,-2.622898,-4.834145],[-7.026924,-1.565373,2.991208,-8.236492,7.855952,5.884438,-2.977870,-3.024277,6.759789],[4.546601,-7.154220,6.279352,9.147142,-9.527955,1.237308,9.237736,1.027192,8.000110],[-3.321043,9.338735,1.679143,9.306294,1.037229,-8.324307,2.966904,-7.627218,-9.588054]],[[-2.262685,4.169797,-2.198656,1.085600,-0.534706,6.054143,-9.865514,5.762653,-9.033511],[-5.659620,6.969735,5.982311,-6.903332,1.628703,-6.758553,-4.378064,8.046826,-5.802560],[-1.337030,-6.617410,-0.286133,-6.156228,8.852868,-2.956977,-5.136985,8.251610,-7.655656],[-1.704888,8.882619,7.895983,5.346334,7.058832,2.083245,-1.444778,3.301583,7.800427],[7.958167,-9.381362,1.167701,-9.340294,3.787209,9.599138,-7.457100,-3.709810,7.599832],[-5.295732,-8.726909,-1.577236,2.107688,2.210449,9.335366,-4.237002,6.900994,1.350185],[-5.051141,-5.756434,8.505551,8.742020,8.940516,0.259999,-8.944721,-4.361744,6.387728],[-5.253296,3.692906,-0.605326,-1.911233,8.576969,0.343218,5.949685,-2.895618,-6.695944],[6.852448,-6.533179,4.818423,-8.181101,3.139858,-2.546475,3.650233,7.717341,3.148646],[5.483755,-0.487483,5.695366,4.913423,6.838366,-5.200606,8.792610,-5.982684,7.766230],[9.942801,9.934148,-0.820732,-1.434553,-4.803625,-2.385714,0.737343,-6.465751,9.201425],[5.141393,-4.440460,-1.431979,-6.253807,-8.556388,8.404751,7.064460,3.700826,5.338864],[-5.899810,0.197903,1.222011,-1.879103,3.378878,7.034528,-7.807874,0.550242,8.730221],[-4.897569,4.541787,-9.848482,-9.310313,-4.070946,-4.764022,6.144458,9.614695,-7.399951],[-4.676522,2.175280,5.413862,-2.541211,8.775825,-7.197968,-5.627442,0.592536,1.241009],[-7.959603,3.943244,-2.402146,7.064971,-2.977013,-7.386238,8.225226,-0.925752,3.004234]],[[6.973969,2.370438,1.042630,-6.142076,-9.339781,-1.480548,5.224319,0.517228,6.035561],[0.407187,0.318613,4.673225,-3.725961,3.409211,2.741135,-1.702223,-3.491383,2.728209],[9.665374,5.532868,8.410158,1.810123,2.209170,2.285179,-8.116480,2.053272,1.649480],[-7.899811,8.810622,-7.745564,-7.582723,-4.917643,-2.221160,-8.207853,3.062673,-9.153383],[1.328168,4.390421,-4.881308,0.895925,1.338557,5.241065,8.418003,-6.697704,1.675130],[6.720324,3.464157,-3.913862,-2.488058,8.325635,2.845907,-9.437871,-4.850828,6.596159],[0.398778,0.558608,7.136701,5.929999,-4.726703,9.890734,-4.862297,9.117176,-8.977214],[2.700183,6.094034,-1.363975,-3.925815,1.951792,8.629001,-8.096407,-7.023181,9.744043],[-5.420435,-8.942986,8.036460,-9.681347,2.690657,9.953289,9.548711,-8.592958,-2.147214],[5.163514,3.166161,4.967285,7.263586,2.554473,2.615814,8.600395,7.827497,1.659161],[-4.549061,-1.752205,3.722873,5.981682,1.618544,5.496754,5.290479,-9.589150,-2.364808],[9.724394,9.171344,9.055735,9.113160,-9.340437,-1.417986,9.025067,4.495418,-9.184134],[-4.441161,-9.986857,7.091814,6.550334,-1.491853,-1.762375,3.449188,0.518841,-7.859293],[0.649179,3.282664,9.246588,3.393079,-0.299340,-2.187853,-3.196737,2.244741,-8.822217],[-3.729312,7.159481,-0.581431,3.633999,3.418228,-0.698051,3.406328,0.028957,-7.532578],[0.636987,-4.395609,-5.411704,-6.683765,2.717263,1.180970,2.309976,4.628316,1.815296]],[[-6.610639,8.695235,1.832585,7.524693,4.633427,2.096606,5.752658,-1.914426,9.800466],[2.469432,8.752308,8.164646,-4.142775,-0.972421,8.510532,6.732695,3.944810,6.251430],[8.848160,0.858045,8.512523,-6.980470,1.405128,-0.584127,4.700979,1.709452,-3.844724],[-9.073042,4.118886,-2.724177,-4.236863,-1.706211,-8.541414,-5.396013,-6.060553,0.418586],[-9.444987,4.027254,-3.767356,-0.382737,6.983030,-6.069295,-4.084142,-6.762302,4.936531],[-2.518730,7.188882,4.251689,-3.290750,8.264682,-8.012721,9.789125,-8.018242,-7.634094],[-0.774841,-3.600095,-2.275442,-5.285382,4.045034,-7.069386,6.498601,-1.457919,4.001231],[5.384586,7.546089,9.941502,8.392973,0.743571,8.113822,-6.467295,-4.229235,6.119067],[5.465888,-9.331729,-6.027280,1.188298,2.730665,-4.788323,-7.651899,3.376702,-5.718921],[8.271862,7.156928,5.178984,-8.365220,-9.225185,0.170030,-8.107041,9.303580,0.695912],[7.233884,-2.144990,1.799333,-8.613129,-3.044863,5.166330,-4.430064,-9.648511,9.275546],[-7.463602,9.951378,3.276572,-7.458949,1.281803,-6.618713,0.377249,7.213912,-5.690142],[2.983873,-3.244020,-1.619988,-9.164403,-4.366262,-1.564418,7.762198,-7.010843,-5.872851],[0.946077,7.670506,-3.989519,0.430112,0.921279,0.811722,6.774079,-8.352796,-9.751823],[-0.538081,4.849210,7.197593,-0.788291,-2.787260,4.306805,-9.672414,4.513374,8.662912],[8.483779,-0.823050,-3.302453,1.548405,-7.475115,4.909154,-1.869314,0.247655,0.222166]]], dtype = "float64")#candidate|1287|(12, 16, 9)|const|float64
bop_1288 = relay.minimum(call_1285.astype('float64'), const_1287.astype('float64')) # shape=(12, 16, 9)
bop_1291 = relay.minimum(call_1286.astype('float64'), const_1287.astype('float64')) # shape=(12, 16, 9)
output = bop_1288
output2 = bop_1291
func_1295 = relay.Function([], output)
mod['func_1295'] = func_1295
mod = relay.transform.InferType()(mod)
output = func_1295()
func_1296 = relay.Function([], output)
mutated_mod['func_1296'] = func_1296
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1300 = relay.var("var_1300", dtype = "float64", shape = (5, 4))#candidate|1300|(5, 4)|var|float64
uop_1301 = relay.log(var_1300.astype('float64')) # shape=(5, 4)
output = relay.Tuple([uop_1301,])
output2 = relay.Tuple([uop_1301,])
func_1315 = relay.Function([var_1300,], output)
mod['func_1315'] = func_1315
mod = relay.transform.InferType()(mod)
mutated_mod['func_1315'] = func_1315
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1316 = relay.var("var_1316", dtype = "float64", shape = (5, 4))#candidate|1316|(5, 4)|var|float64
func_1315_call = mutated_mod.get_global_var('func_1315')
call_1317 = func_1315_call(var_1316)
output = call_1317
func_1318 = relay.Function([var_1316], output)
mutated_mod['func_1318'] = func_1318
mutated_mod = relay.transform.InferType()(mutated_mod)
func_542_call = mod.get_global_var('func_542')
func_544_call = mutated_mod.get_global_var('func_544')
call_1347 = relay.TupleGetItem(func_542_call(), 0)
call_1348 = relay.TupleGetItem(func_544_call(), 0)
func_749_call = mod.get_global_var('func_749')
func_752_call = mutated_mod.get_global_var('func_752')
var_1352 = relay.var("var_1352", dtype = "int8", shape = (1232,))#candidate|1352|(1232,)|var|int8
call_1351 = relay.TupleGetItem(func_749_call(relay.reshape(var_1352.astype('int8'), [11, 16, 7])), 1)
call_1353 = relay.TupleGetItem(func_752_call(relay.reshape(var_1352.astype('int8'), [11, 16, 7])), 1)
uop_1359 = relay.acosh(call_1351.astype('float32')) # shape=(11, 16, 7)
uop_1361 = relay.acosh(call_1353.astype('float32')) # shape=(11, 16, 7)
bop_1366 = relay.not_equal(uop_1359.astype('bool'), call_1347.astype('bool')) # shape=(11, 16, 7)
bop_1369 = relay.not_equal(uop_1361.astype('bool'), call_1348.astype('bool')) # shape=(11, 16, 7)
output = relay.Tuple([var_1352,bop_1366,])
output2 = relay.Tuple([var_1352,bop_1369,])
func_1377 = relay.Function([var_1352,], output)
mod['func_1377'] = func_1377
mod = relay.transform.InferType()(mod)
mutated_mod['func_1377'] = func_1377
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1378 = relay.var("var_1378", dtype = "int8", shape = (1232,))#candidate|1378|(1232,)|var|int8
func_1377_call = mutated_mod.get_global_var('func_1377')
call_1379 = func_1377_call(var_1378)
output = call_1379
func_1380 = relay.Function([var_1378], output)
mutated_mod['func_1380'] = func_1380
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1422 = relay.var("var_1422", dtype = "float32", shape = (5, 12, 9))#candidate|1422|(5, 12, 9)|var|float32
uop_1423 = relay.acos(var_1422.astype('float32')) # shape=(5, 12, 9)
const_1425 = relay.const([[[7.839050,1.807865,-2.152194,-6.334020,-0.865799,-3.509596,-7.575909,0.986349,-9.622759],[6.178241,-8.936302,0.382648,-1.148213,4.434198,1.519787,4.002149,-5.802687,5.436674],[5.841191,-0.040675,-0.215509,-3.943628,-3.609579,0.682390,1.216915,6.443640,-3.778826],[5.141907,0.215583,7.568233,8.161620,3.326647,-2.946564,1.582876,-5.140729,7.206838],[-2.744370,-3.141679,0.076276,9.434735,-1.237248,-0.144281,4.248336,4.057948,-9.000525],[2.622911,-0.231066,-1.213773,-6.971865,-8.949709,1.396355,5.912430,2.752684,-8.764784],[5.580786,8.836473,3.585920,-4.230812,0.610199,7.475473,3.939019,-9.248159,9.079499],[-7.762895,-3.413490,-8.898293,-1.322992,0.884371,8.731087,-3.786203,8.878699,0.537297],[0.885193,-3.652406,8.711349,-8.320151,-2.842187,-5.774844,8.769267,3.890122,6.484166],[4.997551,2.705697,-3.082320,8.520285,9.465580,-3.472756,9.457109,6.174859,6.233153],[4.302494,2.088989,-7.597871,2.881954,-3.264498,4.597901,-1.754245,5.207437,8.973972],[0.152324,-9.054754,6.642875,5.450451,-0.373259,-9.839626,5.728105,4.949144,-8.681257]],[[5.063024,-8.214479,-2.487632,1.386990,-3.052293,1.874441,-9.219760,-6.900139,6.294600],[-7.422952,-1.814779,2.379335,-4.175000,-2.290872,1.610942,1.196516,-7.796357,-4.567319],[0.179099,-8.297156,-1.621302,0.437985,-6.705480,1.140846,1.883145,-6.133300,-4.175117],[7.355565,-4.143858,2.568164,-4.733908,-2.700645,-7.340266,3.673955,8.292040,0.127910],[9.531990,-8.129708,7.582677,-3.251396,8.648041,-0.534847,-0.943163,-6.035685,-5.164912],[1.245343,-1.101223,4.582943,-0.600492,-5.728986,-0.996626,-9.660738,-6.763929,0.355012],[-6.715351,6.108798,-2.477506,-3.341735,5.722248,-7.909410,-8.169290,-3.283500,4.920728],[8.447771,-6.859741,7.091560,0.922544,4.466079,0.953512,7.065544,-2.164833,6.808886],[0.714285,0.175202,4.658160,-7.727719,2.070159,-8.588667,-2.884640,5.722288,-9.601899],[-8.746660,-3.622359,0.170911,-3.821440,3.477871,4.298571,-1.808422,-9.809695,9.711313],[8.553511,9.724676,-3.292144,9.516758,6.978916,-7.036001,-7.770520,-5.573685,-8.039755],[-0.259484,-8.094361,7.723350,-2.322120,-0.815486,4.811562,-6.864543,9.675344,-6.387095]],[[-9.900904,0.654377,-3.201858,0.372177,-6.115334,4.100081,9.239122,0.181223,3.574260],[-7.443001,-9.969398,5.830330,-0.445049,-0.092769,3.720182,-9.781714,8.252263,-5.870970],[5.331069,-0.766600,2.104869,-9.444169,0.446234,-9.887523,-5.891404,-1.872739,-3.782416],[-4.847681,-3.870544,-8.613545,2.422778,2.217884,-4.322827,-7.297896,4.920744,6.533164],[-6.601906,0.688165,-2.059483,7.888282,-8.215810,7.894547,0.689454,-3.812533,0.441452],[7.278884,1.638481,4.435264,-6.995501,2.779243,-9.372456,-2.693023,-4.952684,-2.254954],[-2.501935,7.784266,3.099324,7.834879,-4.891449,-2.794723,-1.609682,7.079825,-9.987077],[-9.882286,-5.151292,-7.909541,-1.109391,-4.891390,3.501320,3.169286,-8.771484,-8.844920],[-2.359448,-0.325765,1.118591,-3.335049,-8.432561,9.128048,-7.870225,-0.869433,5.755017],[9.932862,1.171662,-0.231883,-8.326820,-0.638744,-0.131705,-3.641289,-8.624680,-6.728653],[-2.772055,-4.923729,-3.810659,1.125974,0.396113,5.645491,2.303715,-5.724375,-3.196941],[9.391876,9.657581,6.189112,-6.857874,-1.371108,-6.763004,2.484664,1.083789,-6.556870]],[[4.670494,-0.979955,9.712151,-7.444418,-4.429597,8.873937,5.388515,5.808244,8.452156],[8.971438,-7.868747,-5.937595,-2.246645,2.365290,0.009086,9.894350,-3.911508,3.021017],[0.479118,-7.992900,6.298777,5.810755,-1.679179,8.544403,2.539115,2.694737,-9.697235],[2.749202,5.240097,-4.106981,-2.715160,-1.026264,9.815766,-1.251685,6.085605,-7.416697],[-4.680294,3.759494,-6.810748,-0.120537,8.573727,7.209235,-7.933670,-8.753253,-7.275033],[-6.259720,-1.618948,7.011562,5.332441,3.298128,-5.598458,9.083559,5.877629,6.270315],[7.210400,7.731533,0.019928,-8.263485,1.169296,-4.061309,1.934295,-5.513343,6.913657],[1.208000,7.375738,1.524432,-9.050356,7.241200,3.767134,9.481076,-3.461185,-2.732505],[7.402858,6.349914,0.595786,-9.061097,9.562373,7.895921,-4.759986,9.589234,2.672454],[3.030709,5.676430,2.446853,1.525395,8.732756,5.162581,-2.138658,-2.113652,-2.945672],[7.976617,-1.820228,0.394641,9.273218,-8.063883,-2.398895,0.389523,8.742011,-1.482216],[3.826702,3.324507,1.545127,-2.340080,2.861828,-8.215671,1.768382,8.861409,-3.926937]],[[5.620373,-4.615917,-2.455004,4.338315,5.587272,-8.757172,3.281933,1.711023,-1.923321],[2.661070,-3.427059,3.432029,-3.013764,-9.479585,-0.423979,3.137005,-5.910640,8.352046],[-9.664063,-5.793341,5.547786,6.522569,5.082339,9.331106,5.826243,-2.272349,0.255822],[-6.587534,-1.264168,5.870995,1.594192,-5.095662,1.122546,-0.040833,-4.766526,4.828890],[9.666603,-7.681194,7.712089,-0.701783,4.400487,3.481445,-5.197046,-2.116000,-5.034119],[2.721681,-3.985450,-2.193782,3.748901,8.469081,1.549463,1.218034,0.247965,7.260239],[8.090850,-2.020646,1.575842,-7.463108,7.936864,-2.261722,-3.454737,-1.173110,-3.949499],[-3.402880,4.663040,8.323420,-4.416088,-9.834474,2.883081,5.843617,6.455302,4.217085],[6.237483,9.605345,-2.871743,-9.958198,3.464387,-3.401951,4.120354,-1.547035,-7.996143],[-5.405936,-2.214703,5.838053,-5.797324,3.242006,-1.503711,-2.996301,-0.138391,-6.235128],[-2.892372,3.970124,4.932720,3.667868,4.852461,-1.694024,4.095594,-8.172954,6.428486],[4.303745,-9.462092,1.570770,6.176044,-5.590499,4.860057,3.474242,-2.857070,-1.803564]]], dtype = "float32")#candidate|1425|(5, 12, 9)|const|float32
bop_1426 = relay.less(var_1422.astype('bool'), relay.reshape(const_1425.astype('bool'), relay.shape_of(var_1422))) # shape=(5, 12, 9)
func_1315_call = mod.get_global_var('func_1315')
func_1318_call = mutated_mod.get_global_var('func_1318')
const_1431 = relay.const([6.249422,-1.814627,-2.853673,5.598921,-3.834214,4.248048,-9.335013,6.506789,1.464921,-4.426398,8.361775,-8.070610,1.602826,-9.422397,-0.906601,4.579463,1.955135,3.755073,-0.543477,4.666858], dtype = "float64")#candidate|1431|(20,)|const|float64
call_1430 = relay.TupleGetItem(func_1315_call(relay.reshape(const_1431.astype('float64'), [5, 4])), 0)
call_1432 = relay.TupleGetItem(func_1318_call(relay.reshape(const_1431.astype('float64'), [5, 4])), 0)
func_1315_call = mod.get_global_var('func_1315')
func_1318_call = mutated_mod.get_global_var('func_1318')
call_1451 = relay.TupleGetItem(func_1315_call(relay.reshape(const_1431.astype('float64'), [5, 4])), 0)
call_1452 = relay.TupleGetItem(func_1318_call(relay.reshape(const_1431.astype('float64'), [5, 4])), 0)
const_1453 = relay.const([[[9.412703,-2.044768,2.206060,4.112943,-5.164979,-5.022712,-4.662684,0.882768,8.655797],[3.730127,-7.295527,4.132498,8.297844,-2.527561,-4.216422,2.363334,7.428240,0.494431],[-9.704433,-5.554441,-4.230548,8.034674,-6.229906,0.833072,0.576523,4.205939,4.826484],[5.244774,9.128548,8.429282,7.123665,8.164290,3.245168,2.385066,-2.922254,-3.213022],[4.739995,1.040970,-5.404535,6.727741,5.123867,-8.217869,7.420561,3.779580,7.390280],[-3.233497,7.099949,-9.033668,-0.986970,7.458864,2.719429,-6.243734,8.037511,-7.668833],[-6.629699,-5.085021,-6.466051,4.153064,-9.652834,9.418329,0.413336,4.325555,5.407225],[7.812109,3.559261,-4.788395,-2.012114,7.616309,-3.894508,-1.946519,8.280699,-4.937095],[1.025062,7.387240,-6.852386,7.259000,6.719460,5.825902,-5.325221,-7.021919,5.372195],[-8.117383,8.229533,8.929556,-0.552907,8.180383,-8.928842,6.087791,3.485165,-1.687195],[7.954084,1.422569,-3.731880,-5.237897,-1.645868,-5.950061,6.094460,4.199063,2.906411],[7.268811,-2.546360,2.272010,-4.779117,-3.357723,-9.982924,-8.289374,3.490174,-1.229963]],[[-4.719376,-7.586785,7.032451,-7.738776,-2.033503,-5.751691,-4.140885,0.855783,-8.406870],[-6.935245,6.963309,-4.621846,7.430550,-0.987395,-4.353249,-3.213859,3.561602,0.737747],[3.712929,1.307679,-1.186684,1.030595,7.249489,7.346088,-0.320236,7.245782,3.618231],[2.363456,-5.403637,3.371717,-1.866712,0.049190,-2.311142,-1.908428,-8.364371,9.884046],[2.885239,4.866081,4.054774,-5.449328,-6.455130,-2.445230,0.190166,-5.302138,0.803772],[0.197902,3.017457,3.391117,9.576371,-6.366865,0.208298,-0.257631,-4.253752,-6.835034],[3.066035,2.046719,-1.461622,9.882125,7.892616,9.256473,3.089341,-8.319209,7.308127],[-0.900047,-7.869714,-2.725245,6.975458,-1.049669,-9.194865,4.383092,-7.231946,-4.706893],[-1.415498,-3.649020,6.439176,-4.038337,-3.955044,-7.265615,-3.489084,2.591363,0.726958],[1.781933,-0.533207,-1.310033,-8.168042,-7.628233,4.399739,1.925327,-2.654075,9.803503],[3.928229,-6.445559,4.289349,0.367516,-5.259392,-8.373960,-2.559138,-6.819368,3.000714],[1.770611,-4.645092,-2.574944,-2.829106,-7.423717,-7.531440,4.923051,2.257742,1.756575]],[[-9.108307,6.438821,1.917333,-8.494755,8.374242,0.034442,7.013425,-8.723573,-6.718240],[6.500106,-9.604875,-8.999105,-5.532159,-8.547116,0.847223,-0.744386,-8.722192,-1.595548],[7.989518,1.135711,0.946143,5.249049,2.948104,6.417698,3.272026,3.009910,-9.517004],[-1.782146,8.160999,-4.274353,-6.475631,-7.377975,-2.361317,9.700170,7.531071,-1.890543],[9.242076,1.036256,7.391727,-3.241877,-6.070151,-6.786665,0.032489,3.421600,2.202149],[-3.795068,-7.975448,-9.796894,6.006231,-9.053046,-4.254912,-4.094022,-2.354722,0.720009],[-2.173976,-7.140085,1.553848,4.526730,7.426592,6.443621,-5.515262,6.313433,-5.272138],[-8.968503,9.759480,-0.647275,2.742785,8.969563,-3.813441,-2.882770,-9.224860,4.800598],[8.346941,7.010628,0.352056,-7.304721,-7.109708,-6.214156,0.993639,-1.783415,8.370858],[6.011264,6.725029,8.581885,2.511379,-7.832405,-4.589719,2.007459,-6.406232,8.339318],[7.909539,-6.045309,1.832503,-0.171774,5.097629,3.567686,7.693119,4.755646,0.837639],[-1.541242,-4.632758,-0.002506,4.392073,2.554461,-1.926670,6.642876,-5.145583,4.413251]],[[6.997430,-3.294688,0.538080,5.668654,-5.062647,9.940312,3.312397,-2.363013,5.511911],[9.183389,-3.566522,-4.403855,0.961697,-5.939610,2.790435,9.790563,0.992123,1.416664],[0.735733,7.523551,-7.731854,7.521017,8.968612,2.013359,-4.972937,5.231179,9.292292],[-9.307234,4.867904,-0.066716,-6.690418,7.530461,-4.207800,-3.999686,-2.473275,6.026871],[5.998694,-5.151545,5.288242,7.581749,-8.149323,-0.670182,-9.690381,-3.745588,-9.263173],[8.814923,-9.274836,-1.191515,5.106255,-2.921055,2.538784,-6.803931,-5.333567,4.501780],[6.797935,-9.323463,5.476753,9.909106,2.465630,6.445686,5.211581,-9.319249,-0.146699],[-1.481167,-0.800215,2.491898,-0.211546,-3.657020,-6.355426,-0.570742,-0.885610,-5.841659],[-6.974409,4.883108,6.952983,4.068154,7.482521,4.771179,-1.873779,-5.716909,-0.866204],[-0.194182,1.722864,-7.250588,-3.718126,1.076747,1.374229,-3.246551,-2.585195,7.032721],[6.900100,-5.869855,-2.317756,-5.482096,8.450359,4.465407,-0.817347,-3.464709,-1.455392],[1.647182,-0.790403,4.148039,-2.457794,7.614081,-7.133083,3.227123,9.952243,-5.173688]],[[8.145235,-7.929222,0.650395,-5.392872,-7.113370,-0.131705,1.233452,9.360554,-0.381735],[3.321206,2.978369,6.849402,2.011676,2.809164,-1.688989,3.409007,-6.951598,-7.493889],[0.433328,2.735848,-8.617212,5.456022,2.214393,0.223064,2.297456,2.867826,6.793105],[-3.912352,0.851687,8.303881,-9.940764,2.893118,-1.503046,-2.196156,-1.418930,9.955368],[8.180876,1.085053,-6.774459,2.033935,7.348261,-9.042085,7.563281,-7.250999,-7.750241],[3.399362,-7.148886,4.800131,-7.450466,-7.330127,7.303499,9.756937,-6.811805,5.798229],[7.101011,8.229546,6.669197,4.450771,-1.353487,9.012467,9.289904,3.775934,7.087512],[1.306189,7.167906,1.044656,5.359846,1.103605,7.061128,-8.941615,5.093830,4.164549],[6.097751,-4.224054,-4.526824,-8.503456,-9.212623,0.630739,-9.428950,1.772610,7.889551],[-9.795422,-6.155304,-2.030460,2.602572,-8.792340,-2.350104,4.199361,7.676600,-8.055042],[-9.008739,6.408986,-7.014243,-3.960836,9.933463,-4.303231,-7.257320,-5.714881,7.190175],[-3.596924,2.798661,7.260966,5.071287,-0.889038,-3.253294,-8.561008,9.377954,2.620491]]], dtype = "float32")#candidate|1453|(5, 12, 9)|const|float32
bop_1454 = relay.maximum(uop_1423.astype('float64'), relay.reshape(const_1453.astype('float64'), relay.shape_of(uop_1423))) # shape=(5, 12, 9)
output = relay.Tuple([bop_1426,call_1430,const_1431,call_1451,bop_1454,])
output2 = relay.Tuple([bop_1426,call_1432,const_1431,call_1452,bop_1454,])
func_1463 = relay.Function([var_1422,], output)
mod['func_1463'] = func_1463
mod = relay.transform.InferType()(mod)
mutated_mod['func_1463'] = func_1463
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1464 = relay.var("var_1464", dtype = "float32", shape = (5, 12, 9))#candidate|1464|(5, 12, 9)|var|float32
func_1463_call = mutated_mod.get_global_var('func_1463')
call_1465 = func_1463_call(var_1464)
output = call_1465
func_1466 = relay.Function([var_1464], output)
mutated_mod['func_1466'] = func_1466
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1475 = relay.TupleGetItem(func_695_call(), 1)
call_1476 = relay.TupleGetItem(func_696_call(), 1)
func_396_call = mod.get_global_var('func_396')
func_399_call = mutated_mod.get_global_var('func_399')
var_1483 = relay.var("var_1483", dtype = "float64", shape = (1, 640))#candidate|1483|(1, 640)|var|float64
call_1482 = relay.TupleGetItem(func_396_call(relay.reshape(var_1483.astype('float64'), [4, 16, 10])), 2)
call_1484 = relay.TupleGetItem(func_399_call(relay.reshape(var_1483.astype('float64'), [4, 16, 10])), 2)
func_542_call = mod.get_global_var('func_542')
func_544_call = mutated_mod.get_global_var('func_544')
call_1488 = relay.TupleGetItem(func_542_call(), 0)
call_1489 = relay.TupleGetItem(func_544_call(), 0)
bop_1491 = relay.right_shift(call_1482.astype('int8'), call_1475.astype('int8')) # shape=(1, 16, 6)
bop_1494 = relay.right_shift(call_1484.astype('int8'), call_1476.astype('int8')) # shape=(1, 16, 6)
func_1463_call = mod.get_global_var('func_1463')
func_1466_call = mutated_mod.get_global_var('func_1466')
const_1508 = relay.const([4.703013,-9.435418,0.502704,-4.701916,8.068544,-5.148351,2.933357,1.694573,-5.513213,-1.903717,-7.403715,7.724930,-5.207551,2.929127,2.086835,4.900986,-3.530849,-6.234842,0.333265,6.827310,-5.931198,-4.855083,5.350626,7.989408,4.514766,-3.585828,3.622119,-6.504057,-8.013827,2.835249,-9.579214,9.225573,-9.076283,-0.008472,-1.654603,4.843267,-5.458717,-4.218674,3.137482,5.466052,7.305044,-0.523763,-1.851414,-6.195163,-2.493397,-0.996354,2.058595,6.295884,0.242315,1.636939,-0.782144,7.154858,-2.901760,-0.966835,9.327204,0.620977,-9.616084,9.119572,-3.324997,3.029320,9.971383,4.389493,3.725069,0.266904,1.361912,-3.519188,5.781091,-9.032710,-4.072645,2.255900,-0.471455,-0.114564,-1.662609,5.340242,-1.337334,-9.698162,-5.075909,-8.521147,-1.756649,4.338803,-5.945339,3.316023,6.989491,5.078399,9.559713,3.139982,2.579537,-9.245329,-0.145715,2.467472,-3.784961,-5.635610,6.308337,8.429079,-4.082644,2.354549,-8.435090,-6.419206,-0.604385,5.275966,-2.752984,-6.359569,2.220384,-5.314093,-7.026536,-5.424535,5.167379,-3.853973,0.155574,-3.935472,-9.242991,-2.698267,5.706296,-3.474765,-5.841050,5.172986,-0.071638,5.228850,-1.376325,8.531790,6.579178,7.118858,7.824969,0.148499,5.569232,-9.894414,-8.342482,0.701964,-8.204139,6.723632,5.546816,2.771919,-5.526688,3.881080,-1.376513,-7.147731,-5.807098,-8.201947,5.956412,-6.220347,-5.088975,-7.664261,8.795338,-5.870770,6.324263,2.032191,-3.544244,-7.987906,9.045844,5.054591,2.886279,8.841006,8.410222,5.679112,7.930496,-2.994437,-7.563682,-7.437854,-7.617793,8.742365,0.999222,-8.757203,-0.008638,-6.368779,8.973033,2.478966,1.973305,-0.053153,4.665948,7.796660,-4.350167,-2.853333,-0.995433,-9.106222,1.940378,-1.270994,-6.408963,-6.489700,-8.142946,-4.633385,-4.737715,-4.538076,-6.200830,-6.743862,-7.857217,3.262027,5.091724,5.928742,-9.335346,-8.277318,-9.357990,-1.095807,-1.626165,5.533687,-8.308500,-7.861378,5.322802,-6.485977,0.630690,1.239244,7.583641,-1.984017,-7.822208,-1.024147,-0.850098,-1.649923,-6.041950,3.592346,4.618071,7.083672,1.551478,-1.414955,-2.601942,-3.781405,-1.710737,3.184181,-3.526596,-4.853407,8.685221,-5.124211,-4.582776,-9.249367,5.168907,6.121215,-2.797826,-8.545417,-1.183717,0.116476,-7.285791,-8.767086,4.443937,-7.306111,-9.325832,5.777929,9.687536,0.948365,-8.118590,-3.142325,4.375021,9.318706,-9.700392,1.279821,-4.048058,7.365314,1.876372,6.029570,3.143628,-8.173165,9.370848,-4.088690,1.073351,-1.398995,3.512452,-6.905673,-1.995866,9.190487,4.866711,2.574999,0.473562,-7.186870,-2.990084,7.700363,9.010807,9.886801,2.189118,-3.316383,8.475540,6.203992,3.185748,-2.872713,6.332791,0.541195,-9.203780,-6.816127,6.022518,-2.421985,2.812494,-1.250708,-4.559777,1.677565,7.858906,-9.511186,-2.328314,5.437706,2.915013,6.303122,9.785220,-6.663171,7.046436,5.101596,1.830203,-0.045792,-1.237857,-3.393300,5.176409,-9.198790,9.074185,8.522496,-8.813320,8.230605,-2.493903,-1.021974,1.689779,-3.271303,-7.557551,-8.750035,9.309413,-1.654585,4.048599,-2.207156,-1.247252,2.785893,-8.369853,9.250629,0.369239,-4.371875,3.955149,-9.351260,7.902346,5.572644,-2.037868,-3.251993,-1.371980,4.828443,8.346409,-3.625164,1.036342,7.868077,8.385802,-5.576550,1.106652,-7.836306,-6.300802,-5.507157,9.863238,2.479741,8.645014,-8.027545,-1.294430,1.568223,2.304337,4.107202,0.623144,-5.991810,7.215087,4.855648,-9.522375,4.969855,-0.214993,-4.739377,0.800799,-3.793846,1.035909,-5.941970,-3.069066,5.411244,-0.626384,-7.480701,7.611441,-9.325906,-1.880931,-1.004636,7.355112,2.222796,5.459371,-4.371613,3.857893,7.964734,6.006959,-8.704002,0.023400,2.800173,-4.295257,1.466338,9.820434,-8.832450,6.648377,-9.485528,-3.759696,7.249127,5.440773,-1.674111,-8.643177,-3.220447,7.506296,2.319365,-2.636687,-1.501864,3.162923,1.357425,4.374436,-7.374850,-5.627730,-5.136714,-5.622770,-8.472363,9.278983,-5.492390,-7.004312,1.233637,1.880213,-3.543582,4.069713,-4.577139,-4.250623,1.212760,2.609300,-5.456532,-9.643590,-5.325392,-3.399106,4.778925,9.486660,6.022636,-3.258098,7.619410,-5.666056,6.255443,6.414257,-0.920128,5.144439,4.977224,1.190113,-2.331277,-2.509576,-8.414718,9.752763,-1.668554,5.167953,-0.295378,-5.214548,0.408787,4.886733,0.385424,-7.676471,-1.106992,9.673786,-2.482389,-3.348885,7.651875,7.190240,0.444458,8.444757,5.587246,-2.616940,-2.845696,1.297998,2.288462,-2.118417,7.036016,-0.150448,7.631985,-3.451263,-5.918300,-4.302904,6.113252,7.611917,8.168206,8.758715,9.874009,-2.951108,-7.009831,5.947003,3.203017,-2.760502,-0.766492,-5.103794,6.015752,8.579634,3.847747,5.234785,6.160501,-0.466522,5.669615,2.335044,6.369889,-7.364471,6.198716,2.015447,0.544949,-1.166386,-0.010900,4.714449,4.219345,2.030940,3.450722,2.888895,6.105465,1.307683,2.071662,0.406609,-2.460489,6.047369,3.923291,-8.665103,-0.653932,-1.344514,-1.361006,-0.197947,-5.574063,-8.437538,-9.241056,-0.676491,5.245716,-6.050645,-7.520172,5.878562,8.120738,-0.275505,9.661335,-5.223364,8.764955,-8.339100,-9.298082,8.568211,-5.312808,-7.081367,-0.437373,-6.157169,-5.344875,-0.956455,-1.360894,-4.792738,-1.153329,-7.146520,1.509026,-5.389884,7.074046,-3.007505,6.991098,-3.052182,-1.626292,5.857874,9.166274,-4.772892,-8.587492,-6.860946,2.925741,6.330316,-0.181308], dtype = "float32")#candidate|1508|(540,)|const|float32
call_1507 = relay.TupleGetItem(func_1463_call(relay.reshape(const_1508.astype('float32'), [5, 12, 9])), 4)
call_1509 = relay.TupleGetItem(func_1466_call(relay.reshape(const_1508.astype('float32'), [5, 12, 9])), 4)
output = relay.Tuple([var_1483,call_1488,bop_1491,call_1507,const_1508,])
output2 = relay.Tuple([var_1483,call_1489,bop_1494,call_1509,const_1508,])
func_1527 = relay.Function([var_1483,], output)
mod['func_1527'] = func_1527
mod = relay.transform.InferType()(mod)
var_1528 = relay.var("var_1528", dtype = "float64", shape = (1, 640))#candidate|1528|(1, 640)|var|float64
output = func_1527(var_1528)
func_1529 = relay.Function([var_1528], output)
mutated_mod['func_1529'] = func_1529
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1535 = relay.var("var_1535", dtype = "float32", shape = (9, 9, 4))#candidate|1535|(9, 9, 4)|var|float32
uop_1536 = relay.sigmoid(var_1535.astype('float32')) # shape=(9, 9, 4)
output = uop_1536
output2 = uop_1536
func_1540 = relay.Function([var_1535,], output)
mod['func_1540'] = func_1540
mod = relay.transform.InferType()(mod)
mutated_mod['func_1540'] = func_1540
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1541 = relay.var("var_1541", dtype = "float32", shape = (9, 9, 4))#candidate|1541|(9, 9, 4)|var|float32
func_1540_call = mutated_mod.get_global_var('func_1540')
call_1542 = func_1540_call(var_1541)
output = call_1542
func_1543 = relay.Function([var_1541], output)
mutated_mod['func_1543'] = func_1543
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_1592 = func_87_call()
call_1593 = func_87_call()
var_1596 = relay.var("var_1596", dtype = "int8", shape = (15, 16, 13))#candidate|1596|(15, 16, 13)|var|int8
bop_1597 = relay.multiply(call_1592.astype('uint8'), var_1596.astype('uint8')) # shape=(15, 16, 13)
bop_1600 = relay.multiply(call_1593.astype('uint8'), var_1596.astype('uint8')) # shape=(15, 16, 13)
bop_1602 = relay.equal(var_1596.astype('bool'), relay.reshape(bop_1597.astype('bool'), relay.shape_of(var_1596))) # shape=(15, 16, 13)
bop_1605 = relay.equal(var_1596.astype('bool'), relay.reshape(bop_1600.astype('bool'), relay.shape_of(var_1596))) # shape=(15, 16, 13)
bop_1615 = relay.bitwise_and(call_1592.astype('int16'), bop_1602.astype('int16')) # shape=(15, 16, 13)
bop_1618 = relay.bitwise_and(call_1593.astype('int16'), bop_1605.astype('int16')) # shape=(15, 16, 13)
uop_1624 = relay.acos(bop_1597.astype('float64')) # shape=(15, 16, 13)
uop_1626 = relay.acos(bop_1600.astype('float64')) # shape=(15, 16, 13)
output = relay.Tuple([bop_1615,uop_1624,])
output2 = relay.Tuple([bop_1618,uop_1626,])
func_1635 = relay.Function([var_1596,], output)
mod['func_1635'] = func_1635
mod = relay.transform.InferType()(mod)
mutated_mod['func_1635'] = func_1635
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1636 = relay.var("var_1636", dtype = "int8", shape = (15, 16, 13))#candidate|1636|(15, 16, 13)|var|int8
func_1635_call = mutated_mod.get_global_var('func_1635')
call_1637 = func_1635_call(var_1636)
output = call_1637
func_1638 = relay.Function([var_1636], output)
mutated_mod['func_1638'] = func_1638
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1199_call = mutated_mod.get_global_var('func_1199')
call_1640 = relay.TupleGetItem(func_1197_call(), 2)
call_1641 = relay.TupleGetItem(func_1199_call(), 2)
func_1540_call = mod.get_global_var('func_1540')
func_1543_call = mutated_mod.get_global_var('func_1543')
var_1645 = relay.var("var_1645", dtype = "float32", shape = (324,))#candidate|1645|(324,)|var|float32
call_1644 = func_1540_call(relay.reshape(var_1645.astype('float32'), [9, 9, 4]))
call_1646 = func_1540_call(relay.reshape(var_1645.astype('float32'), [9, 9, 4]))
uop_1650 = relay.tan(call_1644.astype('float32')) # shape=(9, 9, 4)
uop_1652 = relay.tan(call_1646.astype('float32')) # shape=(9, 9, 4)
var_1658 = relay.var("var_1658", dtype = "float32", shape = (324,))#candidate|1658|(324,)|var|float32
bop_1659 = relay.logical_xor(var_1645.astype('int16'), relay.reshape(var_1658.astype('int16'), relay.shape_of(var_1645))) # shape=(324,)
func_1527_call = mod.get_global_var('func_1527')
func_1529_call = mutated_mod.get_global_var('func_1529')
var_1673 = relay.var("var_1673", dtype = "float64", shape = (640,))#candidate|1673|(640,)|var|float64
call_1672 = relay.TupleGetItem(func_1527_call(relay.reshape(var_1673.astype('float64'), [1, 640])), 3)
call_1674 = relay.TupleGetItem(func_1529_call(relay.reshape(var_1673.astype('float64'), [1, 640])), 3)
uop_1675 = relay.cos(uop_1650.astype('float64')) # shape=(9, 9, 4)
uop_1677 = relay.cos(uop_1652.astype('float64')) # shape=(9, 9, 4)
bop_1684 = relay.bitwise_and(uop_1675.astype('uint32'), relay.reshape(var_1658.astype('uint32'), relay.shape_of(uop_1675))) # shape=(9, 9, 4)
bop_1687 = relay.bitwise_and(uop_1677.astype('uint32'), relay.reshape(var_1658.astype('uint32'), relay.shape_of(uop_1677))) # shape=(9, 9, 4)
output = relay.Tuple([call_1640,bop_1659,call_1672,var_1673,bop_1684,])
output2 = relay.Tuple([call_1641,bop_1659,call_1674,var_1673,bop_1687,])
func_1691 = relay.Function([var_1645,var_1658,var_1673,], output)
mod['func_1691'] = func_1691
mod = relay.transform.InferType()(mod)
mutated_mod['func_1691'] = func_1691
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1691_call = mutated_mod.get_global_var('func_1691')
var_1693 = relay.var("var_1693", dtype = "float32", shape = (324,))#candidate|1693|(324,)|var|float32
var_1694 = relay.var("var_1694", dtype = "float32", shape = (324,))#candidate|1694|(324,)|var|float32
var_1695 = relay.var("var_1695", dtype = "float64", shape = (640,))#candidate|1695|(640,)|var|float64
call_1692 = func_1691_call(var_1693,var_1694,var_1695,)
output = call_1692
func_1696 = relay.Function([var_1693,var_1694,var_1695,], output)
mutated_mod['func_1696'] = func_1696
mutated_mod = relay.transform.InferType()(mutated_mod)
func_897_call = mod.get_global_var('func_897')
func_898_call = mutated_mod.get_global_var('func_898')
call_1704 = relay.TupleGetItem(func_897_call(), 0)
call_1705 = relay.TupleGetItem(func_898_call(), 0)
const_1716 = relay.const([[[-6,-7,-2,7,10,-9],[-8,7,6,-6,-10,6],[7,8,-1,-9,7,-1],[10,4,-7,-4,8,3],[-7,-2,3,6,-5,9],[-9,4,-2,-5,-4,4],[-4,6,-8,-5,-7,9],[8,-1,-2,-10,7,-8],[-10,1,4,5,4,1],[8,6,-4,7,-4,2],[9,10,-7,-4,3,-10],[3,-6,-5,-10,2,10],[2,-7,-6,-4,-2,-10],[3,5,3,5,-1,2],[10,10,-7,10,-8,8],[6,10,8,-9,7,-9]],[[1,6,-6,7,10,10],[8,1,-6,3,-2,-4],[-6,-1,8,3,5,1],[-8,7,-6,9,6,-1],[8,-6,8,3,2,-6],[6,-10,-9,-10,1,-6],[6,-2,-5,9,-8,8],[-7,-1,4,9,-7,9],[-4,-4,10,-2,7,-1],[9,-10,4,-3,8,10],[-1,10,-2,5,-1,10],[1,8,-6,-4,8,1],[1,-5,-10,-1,-5,-9],[1,4,-5,1,-6,5],[9,-6,5,-1,1,-7],[10,-2,-8,-4,6,-1]],[[10,-6,1,8,-5,7],[2,-6,1,-4,1,7],[-6,10,-7,-5,-3,5],[4,-5,8,9,-9,2],[10,2,3,-3,-10,-3],[2,-6,-8,-10,-6,3],[7,5,9,-8,5,-10],[-1,7,4,8,5,2],[7,-10,-6,-7,5,1],[-9,-2,-10,3,5,-1],[-8,-6,3,10,1,2],[-3,10,-3,3,-1,8],[-7,-1,9,-1,2,-1],[7,-6,-1,6,9,3],[9,6,1,1,3,2],[5,8,-3,2,-10,8]]], dtype = "int8")#candidate|1716|(3, 16, 6)|const|int8
bop_1717 = relay.bitwise_xor(call_1704.astype('uint16'), const_1716.astype('uint16')) # shape=(3, 16, 6)
bop_1720 = relay.bitwise_xor(call_1705.astype('uint16'), const_1716.astype('uint16')) # shape=(3, 16, 6)
func_749_call = mod.get_global_var('func_749')
func_752_call = mutated_mod.get_global_var('func_752')
const_1739 = relay.const([[-3,-7,10,-5,5,-8,8,5,-5,7,2,4,-2,-3,-10,-9,1,-10,-8,-10,6,3,9,-3,-1,-6,-6,-2,3,-9,10,5,-5,-4,1,-5,-3,-1,8,-9,-6,-4,7,1,3,2,1,7,-4,1,7,2,-4,-2,-1,-5,-8,-9,7,5,-3,5,6,-10,-5,6,6,-1,5,5,6,2,9,1,1,-5,-6,-8,2,-3,-10,-10,10,-7,7,-10,-3,9,4,-9,-7,-1,5,-5,7,6,10,-4,-7,7,6,-9,10,3,9,10,6,-10,1,10,7,5,-3,-10,-1,-4,-2,8,-2,-7,2,-8,-9,3,-8,-7,-10,4,3,-4,9,-3,-3,-7,1,8,3,4,-8,1,-2,1,-1,7,-9,10,1,10,-4,10,3,9,-4,6,-1,-8,-5,9,5,2,7,3,7,3,-2,-7,1,-9,-10,-6,3,-6,-4,6,10,-5,-2,-8,-2,10,-4,-9,4,9,-4,-5,-1,-5,7,-4,9,-6,8,10,3,-5,9,-3,1,-7,8,-2,-4,-6,6,7,-9,-6,5,6,8,-8,-5,-1,2,3,-9,-3,-10,3,-6,1,3,4,-3,-7,-4,-7,-5,-7,-6,1,3,-10,10,9,6,-3,7,-9,8,1,-3,2,7,-7,10,-8,-9,-8,-2,4,8,6,-5,-6,-1,8,5,3,9,-3,-5,4,7,-10,-10,4,-4,-3,-7,3,-6,-3,-10,1,4,10,4,3,7,-10,-2,5,7,4,-9,-2,-9,-3,1,10,-9,5,4,5,-5,-10,-9,10,5,10,2,-8,-6,2,7,3,10,-9,9,-10,7,9,5,4,7,5,-5,-2,5,10,-4,10,-2,2,3,6,-2,-7,-8,8,-2,-5,-5,10,1,-4,3,-10,-2,-8,4,6,3,-4,6,-6,-6,-7,-8,-1,-7,3,-8,6,-5,7,-10,-7,8,-4,5,5,1,-10,-9,9,7,-5,-4,7,3,3,8,-6,1,-3,10,-8,7,1,3,1,1,-5,5,6,-7,-3,9,1,9,-8,1,-1,6,-6,1,6,-3,9,-4,-8,-4,-6,-2,4,-3,10,8,4,-3,-2,-8,-6,-1,9,-10,6,9,8,1,3,-4,-5,6,10,-10,4,1,8,-7,-9,5,-10,1,6,-1,3,-3,10,8,9,-2,-7,-4,-7,4,-7,-5,-5,-6,6,1,-3,-8,2,-4,-5,3,-9,6,8,-6,-9,2,7,9,5,3,-4,2,-8,-6,-3,2,8,2,8,-10,6,4,-2,-7,3,-7,-10,3,-7,10,10,4,1,-3,-1,-7,5,-10,-4,-5,6,2,-1,5,-5,-8,6,4,-7,6,2,-6,-7,-5,5,8,10,-1,-6,-8,9,8,9,1,-9,-5,10,-9,10,-8,3,-8,7,-5,4,-3,-1,-8,-3,8,6,4,-7,-5,-8,-1,8,-4,-9,-7,4,5,3,8,2,1,-1,6,-9,-2,3,-7,-3,7,2,6,-7,3,-6,-7,7,8,2,-4,2,-9,-1,7,9,-7,-9,-9,8,8,3,4,-3,9,-4,-8,1,5,-4,-8,-8,6,-2,6,8,-3,9,-2,8,10,6,-2,3,-1,-10,2,7,-5,-2,8,5],[3,-1,-8,8,-2,-6,-3,2,8,6,7,-5,-2,-3,8,-4,-4,6,5,-2,8,1,-6,7,4,3,5,-4,-6,-1,-7,-3,-9,4,3,-1,1,2,7,-3,-1,2,2,-6,5,3,2,4,5,7,6,-6,-2,-6,8,-9,4,8,9,6,-3,7,-5,9,1,-5,-4,3,1,-4,-1,-9,-6,2,-3,9,-9,1,9,1,6,8,9,7,-3,1,5,8,9,7,10,9,-2,1,-2,3,10,-9,5,1,6,3,-1,3,2,-3,-3,2,-5,-1,7,7,7,10,-10,3,10,7,8,-1,-9,-3,-9,10,-9,-10,-6,8,3,-7,-7,-10,6,-8,9,-3,6,-1,-2,-8,1,-7,-7,10,1,4,8,4,-3,-6,-4,1,6,4,4,-9,1,3,1,9,-1,-1,-7,2,-1,-2,-9,4,6,3,3,-4,10,-4,9,4,-8,-8,3,-5,-8,7,2,3,3,-6,10,3,-7,2,2,-7,-2,-5,5,-7,-9,-2,-10,-10,5,-9,8,-6,9,-1,3,-1,-3,7,-6,-2,-1,-7,3,3,-5,-4,6,-6,-7,6,-2,-4,3,2,-8,-6,3,9,5,4,8,-3,-10,-9,-5,-7,-4,-4,4,10,-2,-10,-4,-1,-3,-6,-6,3,-7,3,9,-1,-2,-4,10,-8,-10,9,7,2,-8,-1,9,-10,-8,8,-4,-4,5,5,-2,10,-9,-6,1,-8,-8,5,3,-4,10,-10,-3,3,10,3,-8,4,2,-6,1,5,-3,7,-6,6,-2,-2,-1,-1,2,-5,-7,-10,10,-3,1,10,-10,9,5,4,2,-2,-1,-5,7,3,-8,4,-8,9,-6,-1,-6,-2,6,10,-2,-2,-7,3,-7,-9,-9,5,9,7,4,2,10,3,-2,5,-7,-6,-6,-10,7,-9,4,4,-7,9,-10,-9,6,8,7,1,-10,2,-8,-10,4,9,-1,-1,3,8,-2,-6,3,8,5,7,8,5,2,1,5,5,-5,4,8,-3,-3,1,-1,7,8,-5,-3,-4,-4,-2,7,4,-6,1,-4,-6,-3,2,-6,4,-3,4,-9,-1,-1,3,2,9,-8,4,5,9,-9,-4,2,-2,-6,10,-3,8,6,4,-9,-8,3,10,-5,6,2,10,-9,-8,7,10,-9,-5,-2,-4,2,-9,-5,-3,-8,4,-10,-3,-7,3,8,-5,4,3,1,-6,-7,-9,5,-3,-4,2,-1,2,5,1,1,-2,10,-10,1,5,2,-2,-10,10,2,-8,9,1,7,9,2,2,7,2,-7,-9,2,6,8,8,9,5,7,-4,-9,-4,-6,-10,10,2,10,-6,5,-6,7,-3,1,-6,3,5,-5,-3,-7,4,7,-3,-9,3,1,5,5,-4,-7,5,2,-3,8,8,9,1,9,-6,4,-2,-9,3,-6,-10,-9,2,4,-8,1,8,-7,-9,4,5,-5,5,4,9,6,3,-4,-9,-2,1,-3,5,-7,-8,-2,-10,-10,10,-2,-5,10,-4,-5,-3,9,2,10,-4,6,7,-5,-2,10,10,-2,-10,3,7,8,8,-3,-1,-2,1,7,3,-4,8,8,8,10,-2,-6,4,-10,-3,-5,-7,9,2]], dtype = "int8")#candidate|1739|(2, 616)|const|int8
call_1738 = relay.TupleGetItem(func_749_call(relay.reshape(const_1739.astype('int8'), [11, 16, 7])), 1)
call_1740 = relay.TupleGetItem(func_752_call(relay.reshape(const_1739.astype('int8'), [11, 16, 7])), 1)
bop_1743 = relay.bitwise_or(const_1716.astype('uint32'), call_1704.astype('uint32')) # shape=(3, 16, 6)
bop_1746 = relay.bitwise_or(const_1716.astype('uint32'), call_1705.astype('uint32')) # shape=(3, 16, 6)
func_1635_call = mod.get_global_var('func_1635')
func_1638_call = mutated_mod.get_global_var('func_1638')
var_1756 = relay.var("var_1756", dtype = "int8", shape = (3120,))#candidate|1756|(3120,)|var|int8
call_1755 = relay.TupleGetItem(func_1635_call(relay.reshape(var_1756.astype('int8'), [15, 16, 13])), 1)
call_1757 = relay.TupleGetItem(func_1638_call(relay.reshape(var_1756.astype('int8'), [15, 16, 13])), 1)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_1760 = func_87_call()
call_1761 = func_87_call()
output = relay.Tuple([bop_1717,call_1738,const_1739,bop_1743,call_1755,var_1756,call_1760,])
output2 = relay.Tuple([bop_1720,call_1740,const_1739,bop_1746,call_1757,var_1756,call_1761,])
func_1762 = relay.Function([var_1756,], output)
mod['func_1762'] = func_1762
mod = relay.transform.InferType()(mod)
var_1763 = relay.var("var_1763", dtype = "int8", shape = (3120,))#candidate|1763|(3120,)|var|int8
output = func_1762(var_1763)
func_1764 = relay.Function([var_1763], output)
mutated_mod['func_1764'] = func_1764
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_1777 = relay.TupleGetItem(func_695_call(), 0)
call_1778 = relay.TupleGetItem(func_696_call(), 0)
output = relay.Tuple([call_1777,])
output2 = relay.Tuple([call_1778,])
func_1788 = relay.Function([], output)
mod['func_1788'] = func_1788
mod = relay.transform.InferType()(mod)
mutated_mod['func_1788'] = func_1788
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1788_call = mutated_mod.get_global_var('func_1788')
call_1789 = func_1788_call()
output = call_1789
func_1790 = relay.Function([], output)
mutated_mod['func_1790'] = func_1790
mutated_mod = relay.transform.InferType()(mutated_mod)
func_861_call = mod.get_global_var('func_861')
func_863_call = mutated_mod.get_global_var('func_863')
call_1793 = func_861_call()
call_1794 = func_861_call()
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_1801 = func_254_call()
call_1802 = func_254_call()
output = relay.Tuple([call_1793,call_1801,])
output2 = relay.Tuple([call_1794,call_1802,])
func_1812 = relay.Function([], output)
mod['func_1812'] = func_1812
mod = relay.transform.InferType()(mod)
output = func_1812()
func_1813 = relay.Function([], output)
mutated_mod['func_1813'] = func_1813
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1295_call = mod.get_global_var('func_1295')
func_1296_call = mutated_mod.get_global_var('func_1296')
call_1817 = func_1295_call()
call_1818 = func_1295_call()
uop_1826 = relay.tan(call_1817.astype('float32')) # shape=(12, 16, 9)
uop_1828 = relay.tan(call_1818.astype('float32')) # shape=(12, 16, 9)
output = relay.Tuple([uop_1826,])
output2 = relay.Tuple([uop_1828,])
func_1829 = relay.Function([], output)
mod['func_1829'] = func_1829
mod = relay.transform.InferType()(mod)
mutated_mod['func_1829'] = func_1829
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1829_call = mutated_mod.get_global_var('func_1829')
call_1830 = func_1829_call()
output = call_1830
func_1831 = relay.Function([], output)
mutated_mod['func_1831'] = func_1831
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1199_call = mutated_mod.get_global_var('func_1199')
call_1868 = relay.TupleGetItem(func_1197_call(), 2)
call_1869 = relay.TupleGetItem(func_1199_call(), 2)
func_897_call = mod.get_global_var('func_897')
func_898_call = mutated_mod.get_global_var('func_898')
call_1872 = relay.TupleGetItem(func_897_call(), 0)
call_1873 = relay.TupleGetItem(func_898_call(), 0)
func_1463_call = mod.get_global_var('func_1463')
func_1466_call = mutated_mod.get_global_var('func_1466')
const_1879 = relay.const([[4.483578,0.919081,6.181624,9.242641,9.756593,6.087542,6.441019,-0.566174,6.498340,0.246213],[-5.180756,6.534085,-0.537558,2.830586,3.979043,-6.568670,9.711115,5.551848,-6.423978,-3.942743],[4.734330,1.223355,6.909879,6.917787,2.289762,6.279972,-8.482803,-4.228796,-2.791344,5.887688],[-6.850467,-6.367176,3.296372,-8.964372,5.501254,1.092116,-6.706182,9.564657,3.005376,-7.124972],[2.632792,-7.014049,-6.799707,-6.021504,2.255125,-4.423936,4.109042,3.292917,6.886859,7.112748],[-1.361999,4.766500,3.419945,1.465581,2.761635,-5.361773,7.584192,-0.831194,9.134762,-8.072527],[-0.044056,-5.617074,-8.535984,-1.285259,3.230442,-8.057819,-3.725032,-5.767470,-4.827110,-2.196208],[-4.075907,7.515334,-2.801384,6.756065,-8.477447,-1.642007,5.005589,1.210879,-0.320973,2.555365],[9.754666,7.502178,4.405963,2.624782,6.574146,6.092525,-6.010337,-2.831276,-5.984992,-5.894420],[9.648806,2.334109,1.589893,-0.973130,-4.450232,0.573540,0.804567,7.620921,7.169885,8.438862],[9.544718,-2.861606,4.750194,-8.715747,-5.762618,-5.039029,9.973234,-2.243974,-1.768917,-8.520091],[-5.198041,-2.241447,-4.491388,-0.434233,4.747807,9.877683,-0.776306,-0.145743,-1.470450,2.424086],[-9.005058,-5.196149,5.212247,-8.050152,-9.008629,-3.066021,8.469430,-3.449258,7.752522,4.189673],[-7.740233,6.647595,-6.403278,-3.777496,-9.107991,5.054674,-7.586868,-2.863263,2.646312,2.016963],[3.196989,-6.681779,3.951181,-0.667613,-3.462111,7.390179,-6.538874,7.143774,-5.072690,7.311644],[-1.067215,8.784527,7.572769,-2.074931,2.134510,-7.721921,-0.738861,2.171424,1.124034,-2.807768],[8.949872,8.753971,-2.143189,-7.451047,-5.385284,1.817480,-9.053794,7.886379,-3.110978,-1.983061],[-0.832380,-9.783439,-9.269844,-8.386186,2.579476,7.363155,-5.630030,3.080685,9.846072,3.264891],[1.365633,-8.603149,-3.858850,7.900851,-5.073910,-4.916014,-3.451927,0.377620,-2.056116,3.011103],[-4.519631,-8.106530,9.930980,8.282964,3.650972,-9.891756,0.921568,-1.125817,1.739014,-3.778100],[-1.269586,-7.479260,3.155255,4.827308,0.748954,7.444596,3.358664,-3.933136,5.245867,4.281727],[-4.118896,-1.982269,5.461768,3.884805,-9.329626,-6.189112,-0.790952,-3.725681,-9.257173,8.757082],[-5.844557,4.401858,-5.177858,-5.560527,6.535543,-8.786711,1.048797,5.241946,-7.246489,1.268960],[-4.675193,3.151961,-5.665562,-7.231707,5.440451,-8.314585,-2.365949,7.700537,1.132679,1.251798],[0.718676,1.851757,-5.236247,3.949797,3.417005,8.417578,-7.334448,6.996272,-8.132658,2.669012],[3.063657,1.845995,1.929241,0.633011,-6.106526,5.779412,-1.465663,6.912066,6.795264,4.229416],[7.290872,-4.642128,2.551857,9.054099,2.562728,8.375698,-2.582948,9.934593,0.282863,9.734622],[2.191706,-4.103446,8.492657,-6.533676,-0.675547,7.454625,-8.167549,-3.341115,1.709735,-1.558567],[-0.737179,-9.630149,-2.076393,9.138604,-3.287266,-1.836269,0.856361,9.370190,-5.916247,-1.120903],[-3.440699,-2.212417,7.643553,-1.832542,-5.627318,-0.163779,4.400793,-4.018433,-5.395698,2.634404],[-7.747372,-1.143770,-8.565424,-7.578210,9.827080,-5.824443,-3.058858,-6.458009,2.026334,8.128590],[5.903383,-9.507325,-6.238335,3.734753,-2.296024,6.638101,-7.143477,7.941696,5.300656,3.672862],[-2.300575,-1.318901,-8.348346,-8.125435,-4.134797,-2.156538,1.436990,-7.584016,1.328298,-3.299410],[-7.812256,-0.804336,1.107641,-0.034431,7.363276,6.767633,5.206754,-8.599253,1.295052,1.759869],[-6.625410,7.898944,-5.336840,2.346073,9.679717,-1.706792,2.516028,-9.439752,-3.124076,-0.153953],[-9.462321,2.597124,0.803645,-8.658836,-8.745871,-3.215438,0.809719,5.906984,5.453835,2.490552],[-6.897982,2.389315,-8.781742,0.196386,-5.041126,-0.696246,8.099044,-1.510361,4.451908,0.493004],[-1.536460,6.875674,-3.597209,4.711010,7.377905,-1.497868,-1.484348,-1.079029,-5.238826,4.009813],[4.657827,2.771429,2.669735,1.765602,-8.857940,1.522392,8.708311,-5.878396,9.009325,4.780875],[-0.632268,-5.701292,8.495300,2.863247,-2.658699,-9.711414,-7.007673,4.448180,8.043316,7.338003],[7.444371,-9.082113,9.830592,-6.098596,6.795370,8.467327,-2.646662,-9.940875,-9.404291,3.595986],[7.537721,-4.878193,7.005029,2.071210,0.113064,7.302578,9.742438,2.672177,-5.484479,9.849737],[3.519719,-6.826971,9.978935,-4.573322,-4.293791,-3.017929,3.987031,-9.998256,-6.150137,-0.632433],[-2.312131,7.147488,3.618119,-2.079474,-1.592781,8.069025,-0.069456,6.516993,8.018039,-5.150575],[5.667610,3.504298,8.553351,-8.568919,3.036813,6.134288,-3.662776,-3.653348,2.417748,5.232228],[-0.430666,-0.019551,-8.879767,-5.959874,8.379753,-2.255897,-6.130422,0.363716,5.311617,1.224939],[-0.872618,-3.454868,2.346145,8.622896,4.379516,6.624590,9.509784,-2.576721,3.280553,3.608139],[6.782232,-4.439806,0.957461,-1.197860,4.301969,-6.912339,-6.607586,-3.206599,0.013134,1.212337],[-5.362472,-3.352938,-7.904578,-1.734732,-9.851517,-6.134695,9.672671,8.954523,-2.585143,5.481555],[2.322660,-4.008158,-8.530652,-9.915147,-6.238593,-6.910022,-2.191758,-5.038663,8.876032,-5.264077],[6.439823,-4.512956,7.005137,1.540365,5.640045,9.825166,-3.094225,-0.397759,-2.712307,-4.788035],[-5.135213,-2.260738,-9.942651,9.635453,6.470752,2.868646,-7.352423,5.639022,1.819171,-0.124614],[8.794822,9.784573,8.000741,-3.219133,-8.797032,-6.646395,9.440405,9.779378,3.519947,2.372916],[1.324312,4.828180,-7.332601,3.890843,-0.536358,-9.723759,-1.335604,5.922928,-0.058063,5.952521]], dtype = "float32")#candidate|1879|(54, 10)|const|float32
call_1878 = relay.TupleGetItem(func_1463_call(relay.reshape(const_1879.astype('float32'), [5, 12, 9])), 0)
call_1880 = relay.TupleGetItem(func_1466_call(relay.reshape(const_1879.astype('float32'), [5, 12, 9])), 0)
uop_1886 = relay.log10(call_1878.astype('float32')) # shape=(5, 12, 9)
uop_1888 = relay.log10(call_1880.astype('float32')) # shape=(5, 12, 9)
func_123_call = mod.get_global_var('func_123')
func_126_call = mutated_mod.get_global_var('func_126')
const_1907 = relay.const([0.857061,4.595935,-3.557982,-4.423739,-2.531026,6.166626], dtype = "float32")#candidate|1907|(6,)|const|float32
call_1906 = relay.TupleGetItem(func_123_call(relay.reshape(const_1907.astype('float32'), [6, 1])), 2)
call_1908 = relay.TupleGetItem(func_126_call(relay.reshape(const_1907.astype('float32'), [6, 1])), 2)
uop_1924 = relay.rsqrt(uop_1886.astype('float64')) # shape=(5, 12, 9)
uop_1926 = relay.rsqrt(uop_1888.astype('float64')) # shape=(5, 12, 9)
var_1932 = relay.var("var_1932", dtype = "int8", shape = (11, 16, 1))#candidate|1932|(11, 16, 1)|var|int8
bop_1933 = relay.less_equal(call_1868.astype('bool'), var_1932.astype('bool')) # shape=(11, 16, 1)
bop_1936 = relay.less_equal(call_1869.astype('bool'), var_1932.astype('bool')) # shape=(11, 16, 1)
func_1377_call = mod.get_global_var('func_1377')
func_1380_call = mutated_mod.get_global_var('func_1380')
var_1939 = relay.var("var_1939", dtype = "int8", shape = (1232,))#candidate|1939|(1232,)|var|int8
call_1938 = relay.TupleGetItem(func_1377_call(relay.reshape(var_1939.astype('int8'), [1232,])), 0)
call_1940 = relay.TupleGetItem(func_1380_call(relay.reshape(var_1939.astype('int8'), [1232,])), 0)
var_1943 = relay.var("var_1943", dtype = "float32", shape = (5, 12, 9))#candidate|1943|(5, 12, 9)|var|float32
bop_1944 = relay.bitwise_xor(uop_1886.astype('uint8'), relay.reshape(var_1943.astype('uint8'), relay.shape_of(uop_1886))) # shape=(5, 12, 9)
bop_1947 = relay.bitwise_xor(uop_1888.astype('uint8'), relay.reshape(var_1943.astype('uint8'), relay.shape_of(uop_1888))) # shape=(5, 12, 9)
output = relay.Tuple([call_1872,const_1879,call_1906,const_1907,uop_1924,bop_1933,call_1938,var_1939,bop_1944,])
output2 = relay.Tuple([call_1873,const_1879,call_1908,const_1907,uop_1926,bop_1936,call_1940,var_1939,bop_1947,])
func_1949 = relay.Function([var_1932,var_1939,var_1943,], output)
mod['func_1949'] = func_1949
mod = relay.transform.InferType()(mod)
var_1950 = relay.var("var_1950", dtype = "int8", shape = (11, 16, 1))#candidate|1950|(11, 16, 1)|var|int8
var_1951 = relay.var("var_1951", dtype = "int8", shape = (1232,))#candidate|1951|(1232,)|var|int8
var_1952 = relay.var("var_1952", dtype = "float32", shape = (5, 12, 9))#candidate|1952|(5, 12, 9)|var|float32
output = func_1949(var_1950,var_1951,var_1952,)
func_1953 = relay.Function([var_1950,var_1951,var_1952,], output)
mutated_mod['func_1953'] = func_1953
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1985 = relay.const([[[-6.916541,4.668486,7.026580,-3.418904,-2.999259,-0.401055,-5.134020,-9.639083,5.407061],[-7.312738,0.762028,-1.913071,-1.936112,2.246586,0.895382,-9.450963,2.520261,3.029735],[3.832716,7.417629,2.183571,-1.421807,-0.392495,3.405449,-4.765799,-3.089433,-8.960268],[6.160070,6.010648,4.513638,-2.217732,-0.239559,3.518950,-9.313537,-1.418074,-4.287363],[7.299506,7.534084,-8.355825,8.595596,9.217053,-1.717943,-5.574032,-9.547154,6.537021],[5.668462,-7.996154,9.740158,4.896925,-1.047929,-0.787707,9.066382,3.616731,9.367871],[7.760645,-1.305020,-5.669398,0.871147,0.619572,9.602887,3.922710,-2.246052,-3.071878],[4.484745,8.417260,0.944163,-4.924588,-5.574811,-8.877035,9.666630,-2.706781,-5.893322],[0.712537,-7.858166,0.472176,4.661594,-3.811737,-1.015638,7.458236,-3.732909,-4.306758],[-6.954054,-9.805964,-9.812849,3.512395,-7.380654,-9.136677,-5.379450,-8.318455,-4.896121],[-9.957214,-0.182363,0.964288,-1.027326,-9.736296,6.609341,-5.240267,-4.118705,-1.086858],[7.027218,-1.080493,-4.106294,-0.557319,-8.907227,4.933163,5.313311,-2.716562,-7.852874],[-3.604889,6.088830,1.098974,3.339969,-9.043769,-7.089981,5.526046,-1.574580,0.622476],[-1.269053,-2.444384,3.457330,3.808292,3.163751,-4.644125,7.244622,-4.148986,-3.256592],[4.612865,-1.392354,8.877305,2.469831,-3.794066,-7.872215,-0.716996,-4.235523,-3.166468],[-9.590436,-2.811096,9.982099,2.381136,2.116426,0.804907,4.485365,-5.045438,4.043176]],[[-6.193633,-8.255020,-3.460585,-3.950208,3.131869,6.871606,-3.398474,3.375038,-9.177564],[-4.832393,1.565811,8.138025,1.319276,7.857147,-1.057538,5.434350,0.698690,-8.839516],[-9.961105,-4.468721,8.422394,3.794860,2.658028,-5.686053,-9.395604,0.262945,-3.986719],[-4.042278,0.325994,8.131647,-5.516093,-4.926493,-1.134287,-8.741487,-3.944811,-4.827163],[-2.934710,2.159934,-4.293483,6.319003,-9.789531,-8.275070,-7.045592,-5.468240,6.288232],[5.726576,-4.903273,4.479053,-1.482593,-9.030067,6.258474,-8.963142,-4.752086,1.475586],[6.951399,7.220152,-3.120660,1.706376,9.371543,-3.363346,3.809856,-4.234745,-9.399021],[1.503660,8.136642,4.367429,-1.687287,5.206216,2.808193,-6.001992,9.192120,9.277825],[9.404241,-1.162527,-8.693577,6.672860,4.694168,-9.705285,-4.882072,-0.824132,6.966292],[-5.897749,-5.802211,-3.957472,-4.659179,-8.362140,8.136108,7.941266,-8.799286,9.957031],[0.610243,9.515686,-6.319117,-2.167146,4.866286,1.322075,-3.296890,-4.717562,9.574314],[-3.833148,4.330689,4.730107,-9.045286,-0.566981,8.026892,-2.454966,-8.627381,1.565860],[3.602153,4.306317,-8.031536,3.181883,-3.384876,8.751176,-5.256505,-9.534322,8.936906],[-9.749108,-4.287842,-2.770039,-0.497356,-0.867475,-4.109317,-5.064798,7.679782,-8.959762],[-4.071078,-8.524692,-9.980149,-6.844038,-6.851789,-2.720819,-2.287932,0.458516,9.106219],[-2.482036,-8.078438,0.829246,-8.822389,-2.260562,2.711598,-4.727806,-1.753912,0.399204]],[[-0.571529,-7.007244,-2.390030,-3.668656,7.342176,8.166588,-1.255239,2.128006,1.891316],[6.740308,-0.699337,1.078419,4.433295,6.918976,5.006673,-3.460283,-8.908962,-1.743365],[6.478350,4.568931,-4.106925,-5.818560,1.817694,-8.274691,-7.262495,-6.192515,4.761858],[-2.130493,5.351095,6.941758,-2.482841,2.862489,-7.151892,2.083986,-3.928393,-5.633399],[-8.486721,6.592879,-6.040418,-8.285425,-0.464755,3.130869,-6.296444,-0.906766,1.776189],[-7.315334,9.173414,-3.099338,2.972269,8.533204,8.922097,7.074197,-2.932553,3.211597],[-5.930769,7.160696,-0.768138,-4.435256,8.986972,0.152055,-8.066642,-4.488596,0.915122],[3.913314,6.326709,-2.252077,2.465454,-0.189334,2.028370,-2.583068,0.423195,-1.457532],[1.046643,-0.119942,5.026085,6.270315,-4.075981,6.694702,2.744773,-5.029466,7.433109],[4.769738,-0.408344,2.912036,8.100956,8.844296,1.624304,-2.944334,4.082272,0.191848],[-3.478569,-0.444405,3.902582,-1.133439,2.883848,1.127069,-1.647102,4.831302,-1.988407],[-6.744184,-8.125535,-4.463851,8.597282,-4.195309,0.221921,7.501629,0.181626,-8.020942],[5.920734,9.028338,-5.220888,8.477267,7.451582,4.292074,7.021944,3.771970,-5.625617],[-5.090372,-8.249622,-2.963990,-7.602617,-8.124896,7.216430,-9.973301,6.682681,1.255457],[-4.243927,-0.073619,1.689052,9.353272,1.672438,7.903365,-9.850613,2.516197,-1.339012],[0.129658,9.354777,-1.090885,-8.490391,1.662647,-7.442077,4.380685,-9.367861,-2.589953]],[[-3.443143,-1.342979,-5.523293,5.461405,1.634693,-0.698438,-7.704297,0.393545,9.666742],[-7.853045,-2.008746,4.250087,0.807645,-9.483871,4.079906,7.999776,-5.419735,3.267209],[0.321055,-6.429953,-9.952535,3.769057,-0.644522,-0.793885,-2.522247,5.613698,8.622155],[-3.765219,-1.857283,-1.935071,8.923347,1.314364,1.000729,4.104003,4.152369,-2.596618],[-9.424323,-0.935294,-0.755626,-9.309671,5.908738,7.936354,-2.724417,6.834509,-5.982240],[-3.642508,-1.667936,-0.173453,2.489447,-2.780200,6.444194,-7.101550,-2.075752,-4.196918],[9.044663,-4.902269,-2.810550,7.187770,3.299052,1.551243,-3.510895,7.771681,9.585846],[3.246854,3.138423,0.695808,-1.319582,-0.733499,-3.269565,8.608122,9.151905,6.017283],[0.774156,0.957762,9.673818,6.428591,-3.106683,-5.578732,-6.818418,-7.654719,-5.146942],[-6.057277,1.875796,4.345015,3.144488,-1.532167,-4.788471,-0.045181,-9.011505,-1.393899],[2.884695,-2.241165,8.074423,1.685408,-4.022238,5.409846,-8.722305,6.020822,3.024378],[9.503357,9.113872,9.356674,6.042370,9.804831,2.935446,-2.468937,8.167121,6.729145],[7.257287,-5.361109,-2.744262,-5.301757,1.932750,0.340441,8.645880,6.884946,-8.877305],[-9.793941,8.485620,8.760210,-0.385012,9.387527,-0.248334,-8.195196,8.851225,3.985622],[-1.109632,-6.157559,9.781206,-3.110474,4.852030,4.915002,1.514549,4.721729,-0.802039],[0.076992,-4.089207,-1.023913,7.811004,5.402562,-4.182634,8.223390,-4.698578,-3.719044]],[[8.407924,-7.834522,0.203944,-3.792362,-8.661783,7.862746,-6.034794,0.840124,6.462603],[-4.044376,-7.476639,9.253063,-2.350020,6.123867,7.288551,3.105355,0.097333,4.113823],[1.139760,6.900716,2.818300,-0.790081,3.206347,6.694155,-5.352765,-5.974942,9.436325],[9.204392,-5.083361,-9.004371,-4.500697,-6.056675,-6.890171,-7.434067,7.779908,4.841480],[-6.693686,-9.660487,8.482289,-1.478310,3.770140,-9.286965,0.047474,2.863318,-7.219270],[9.527802,3.838982,-1.681100,2.373983,1.560768,-5.791419,3.723680,0.167770,4.078739],[8.589626,-6.975219,-7.034688,8.356933,-4.881294,5.121837,1.384885,6.949480,-4.599827],[-7.570353,4.554052,-3.115917,-5.181542,-2.124501,9.865785,-2.134221,-9.993025,-3.301185],[8.194077,-3.120011,-1.055554,-6.296762,-4.346547,-5.659083,-6.114224,-7.294683,-7.693584],[8.746769,-5.728021,-2.241265,4.315721,0.100379,0.599324,-0.339414,4.676918,-5.220114],[-8.607257,6.274887,7.214670,7.551538,-1.551777,6.543408,-8.255559,3.823899,-4.359037],[6.387985,-3.362658,4.995052,-3.464218,-9.429284,-0.068552,0.287675,-5.838118,4.323336],[-4.809610,4.593913,-1.050725,7.686146,1.870223,-3.605778,2.541776,-8.438630,-1.303590],[5.846128,8.020622,-0.214913,9.911899,2.922138,1.687894,0.165629,2.353547,-3.218441],[2.061075,3.956364,5.632904,-0.877149,5.605373,-5.536019,2.285791,-6.322608,-9.498466],[-0.255894,-1.883450,9.768059,0.597674,3.858453,2.724144,-6.255366,9.876703,8.728981]],[[1.726584,8.169516,-9.961508,-9.417486,-7.982186,9.306888,-4.494323,-0.124903,2.648364],[-8.190297,-1.963710,6.673783,3.345511,-8.440468,6.946532,6.011436,5.605946,1.253737],[6.431738,1.878204,-7.448219,-6.091756,4.888804,9.345166,-1.420350,7.149925,0.377381],[-6.848567,-2.772109,-6.342557,-2.748962,5.770491,-3.989534,-0.855553,-8.544336,7.743137],[-8.262335,2.611957,-8.218321,-8.092045,-5.516389,6.913056,-3.957832,9.248965,-0.799138],[4.488101,2.187553,5.265508,1.437573,1.148357,6.615787,0.222031,-9.690486,0.837242],[0.269534,0.425970,6.736384,-3.708096,-5.152451,7.401053,5.391643,-5.782163,7.539018],[-4.536990,6.692818,-9.759948,3.865380,-9.807053,5.971813,-0.639743,4.684763,-9.506735],[1.709139,-3.598474,-2.028341,-4.906126,-7.144573,1.954824,-3.334353,0.812074,7.100489],[7.543058,7.207834,1.389247,5.272580,0.161690,0.789706,-4.025404,-5.124405,0.133987],[-4.210613,-3.604046,-9.246676,-7.479470,-3.920679,-3.304643,5.162837,-8.033041,3.944197],[9.716135,0.180591,1.748540,7.540070,8.385013,-2.869672,-2.771615,3.786083,-3.910447],[-8.587811,-3.043876,-2.083073,9.986272,-3.810498,-7.337951,1.457416,4.678410,-2.346795],[1.956533,-8.895155,-9.928911,3.295586,5.922280,6.443308,-7.752641,-8.994519,-7.120450],[7.179920,-6.816100,3.682255,0.121100,3.781380,-5.326000,3.844144,8.608613,-7.983623],[6.017674,-9.581070,-9.072736,-3.264544,-8.182815,-0.937601,3.087636,-2.224045,4.575835]],[[-3.084249,-4.719889,0.740596,6.658709,-3.311927,-7.942390,-7.086080,3.520803,-7.119744],[3.177281,-7.124536,7.115361,-4.019218,5.712313,8.689182,8.706577,8.068766,2.816375],[-2.987803,-5.519584,0.297843,0.950034,-2.538743,-2.522328,9.108307,1.071603,1.762639],[-9.835241,3.981414,-8.612209,-2.458653,-4.764821,5.996451,9.037575,-8.211249,8.038937],[-7.316220,-7.134829,7.630004,-7.681090,-1.870042,-2.945430,-9.499307,6.039194,-1.974614],[-8.994418,1.598655,4.849143,-9.083556,-9.062256,-4.791978,-9.831052,3.614191,0.176687],[-0.551970,2.073769,1.264025,-8.511323,-0.120397,2.429150,-0.370821,5.799649,-4.098041],[-2.296786,4.858166,4.167971,-6.205508,6.634172,-1.424557,-7.541579,-6.094063,0.295033],[-2.587726,1.793617,-2.568436,4.088083,3.675804,-0.762113,2.848830,-8.786508,-0.620166],[-0.373661,-3.397890,-0.749940,-5.603536,9.309243,-2.109705,-9.362600,-5.896177,1.096993],[-5.084938,2.076179,0.665113,3.937458,9.976333,-2.810168,-6.695310,-1.322715,-8.823174],[0.817073,-7.566280,6.032092,-5.728277,0.018142,4.309515,5.494126,5.945333,1.394472],[-4.421339,-0.772098,6.717497,0.249715,7.637628,-6.521257,-0.126358,0.320708,-8.132717],[0.579199,-8.750379,5.123362,5.745940,-4.722055,-6.987724,-3.210718,-7.911327,-9.834251],[3.681482,2.131893,1.974134,-2.383131,-2.488768,-1.225172,-0.735253,5.361721,-1.509086],[-3.403468,6.982107,3.950502,8.437837,-6.125281,-3.474830,0.620791,6.781010,-2.401000]],[[-0.765226,-4.565778,-3.063115,7.252564,4.577108,-1.304060,0.715532,3.292964,2.633739],[-6.972865,-5.613751,-9.077996,5.532996,-4.484527,0.859125,-9.830931,1.913896,-9.359354],[1.821897,-0.541448,-1.341572,4.433839,-9.151571,-5.847077,8.323967,-6.588077,0.235736],[-2.375721,-0.660222,5.243952,1.388391,2.728299,5.129093,6.405033,9.980659,5.681117],[-6.150455,0.043112,7.745668,-5.333359,9.347902,3.837434,3.921573,0.905599,-7.336165],[3.282201,-4.422763,5.605966,-3.291162,4.383158,-8.027440,5.317102,7.653879,7.274433],[-2.973240,7.864447,-2.030892,9.273956,-2.865142,7.995069,5.428211,-4.710649,3.130339],[9.343232,-0.275214,3.179940,7.455259,7.561472,-7.531746,-3.709033,-6.903886,-7.021089],[-6.640219,7.494715,-2.145250,1.789122,9.205553,-1.265620,6.329535,4.823142,-1.013454],[4.897418,0.447586,-9.658056,7.326649,8.515573,-5.849154,5.238052,9.115983,9.485589],[5.741308,6.370426,-0.755497,-9.409197,-1.413739,-4.186172,-1.250985,-8.568605,-6.565929],[3.582456,5.648425,-5.708434,-1.964156,-4.045257,-3.476338,-8.640168,4.795664,-1.799585],[3.566643,1.992467,-3.503119,-5.041781,7.957465,2.759619,2.367419,-0.406486,1.789238],[-7.382563,-5.898726,5.207901,-1.471313,-8.981879,4.673181,9.691534,9.246712,-7.915389],[-9.428785,0.453482,-7.371680,-9.571975,5.896408,-5.368861,0.313321,9.311015,-3.501010],[-9.699273,-0.887587,-9.081466,1.948493,-5.520876,-3.658183,1.285024,3.195777,3.456949]],[[3.269320,2.072307,-6.928789,0.902808,-7.101039,4.997977,3.516030,1.106241,-2.356883],[6.221756,5.746405,4.795715,8.275259,4.501890,6.195949,-7.371503,-0.545458,6.720794],[5.675966,-7.013430,8.478343,-5.325740,0.545756,-6.282248,3.994034,9.448066,2.555980],[6.965875,6.345924,9.900104,2.238531,9.114970,3.691311,-4.415096,7.208852,-4.482365],[-2.490254,1.572178,-1.905611,0.958994,-8.728899,4.761585,8.519469,-7.696317,7.038781],[3.279754,7.743397,-9.700109,-7.177392,8.038577,1.131261,-9.581885,7.812512,8.574417],[-2.342778,9.825008,2.668419,-3.465430,5.139476,-3.799249,4.435594,1.341067,-4.298499],[1.106885,-3.022220,1.737939,9.884192,9.416359,4.051096,-5.537612,-1.122606,-8.627218],[-5.231241,-5.584960,0.438241,-9.619741,9.114440,1.590603,8.256593,-0.223639,-0.560538],[5.852587,4.228664,-3.917686,1.691734,-6.425100,-6.181310,-7.408459,-3.990212,9.532603],[8.295952,-9.505637,5.605720,-7.340322,-6.397446,2.049415,-9.192395,-8.055843,6.683355],[1.884669,-9.504940,-2.584003,-1.880009,7.483823,-9.173546,8.303412,1.258292,-6.443501],[8.325649,4.810518,7.403653,-2.897419,-9.690710,4.589566,3.127204,-5.979561,3.265545],[3.698004,2.862927,-8.639215,7.276260,4.980739,1.845380,8.548030,8.250898,-9.871559],[3.944254,-5.551172,-1.178428,-0.795157,-0.960546,5.953158,8.705296,-4.476362,1.734776],[-5.415340,-4.506208,7.786469,8.160471,-0.207267,-2.946397,0.182889,-0.432747,6.266106]],[[-3.589546,-0.279701,-0.887888,-2.878130,-6.523752,2.030014,7.302152,-4.025540,1.913882],[5.229372,0.461488,5.655382,-3.305095,3.637464,5.782896,9.390478,-5.481718,-5.140555],[5.491524,8.581635,6.545851,-0.653658,-8.106749,-9.680161,-8.707675,8.787025,-3.927383],[7.679361,6.113250,4.183225,4.520210,-9.464175,-3.725998,5.850606,6.147284,-9.083313],[8.458775,4.320959,2.680900,5.963695,9.439008,7.927097,-5.650767,4.850856,-1.360536],[-0.134621,7.612772,2.165280,7.059940,-9.185827,3.870080,-0.365848,0.231338,9.104551],[9.260514,-4.055210,6.689965,-6.343019,8.177345,2.188321,-5.004908,1.742841,7.893685],[-7.927204,7.759869,0.978196,0.749182,1.718298,-5.366393,3.369767,-5.615735,-2.099717],[-0.205201,9.790257,-8.957572,3.631651,-6.327054,8.551605,9.626596,-4.195548,-7.000721],[-6.246462,6.300282,-1.548404,-9.621906,-6.535748,6.034590,-7.214433,1.618721,3.749768],[-0.605159,2.920467,-2.586848,-9.296998,-0.549614,-2.728759,7.088635,-8.552855,8.630067],[-5.186416,-4.751587,-9.231724,3.100670,5.562274,-1.729060,4.761165,2.669253,4.349125],[2.591589,-3.984424,-4.816132,-7.581009,7.649892,8.899726,9.296790,-6.494373,-8.060590],[5.824583,2.745048,-4.701751,-3.215891,-0.665857,-8.643034,8.465647,-0.839871,9.772029],[-9.673790,-5.620544,-0.709692,-1.886009,7.212922,-3.674071,4.951880,5.109668,-6.154038],[-1.610444,4.836198,1.620887,-2.547316,3.049257,-9.160146,-6.882145,-2.173678,-4.956600]]], dtype = "float32")#candidate|1985|(10, 16, 9)|const|float32
uop_1986 = relay.sinh(const_1985.astype('float32')) # shape=(10, 16, 9)
bop_1988 = relay.mod(uop_1986.astype('float64'), relay.reshape(const_1985.astype('float64'), relay.shape_of(uop_1986))) # shape=(10, 16, 9)
var_1992 = relay.var("var_1992", dtype = "float32", shape = (10, 16, 9))#candidate|1992|(10, 16, 9)|var|float32
bop_1993 = relay.multiply(const_1985.astype('float64'), relay.reshape(var_1992.astype('float64'), relay.shape_of(const_1985))) # shape=(10, 16, 9)
func_1949_call = mod.get_global_var('func_1949')
func_1953_call = mutated_mod.get_global_var('func_1953')
const_2024 = relay.const([3,8,-5,3,-8,-3,-10,1,9,-10,-7,-10,8,9,-9,3,-6,1,-1,7,3,-8,-6,6,2,-8,-1,-9,-10,-10,9,-1,4,4,-8,-5,-1,-6,3,2,-8,-6,9,-9,-6,-8,-4,-7,2,-4,-2,-3,-3,10,1,-5,4,-1,9,-2,10,-7,8,1,-7,2,5,10,4,-5,-2,5,2,-9,-5,5,10,10,-7,-9,7,2,7,-9,4,6,-4,3,5,8,-2,5,-4,3,-7,6,6,-3,3,-3,-6,2,8,-8,7,4,9,-5,-6,8,1,-5,-3,-9,10,3,1,-6,1,5,-1,1,-10,-5,-5,2,-7,6,-7,8,6,-5,-7,7,6,9,-1,-7,9,1,-7,3,-2,-5,1,-6,5,8,-2,-3,9,6,-4,-2,5,8,-8,7,7,7,-5,7,-6,-5,1,7,-2,9,5,9,-9,-3,6,-5,2,-3], dtype = "int8")#candidate|2024|(176,)|const|int8
var_2025 = relay.var("var_2025", dtype = "int8", shape = (1232,))#candidate|2025|(1232,)|var|int8
const_2026 = relay.const([[7.303999,0.797384,-9.900505,-8.444189,2.334921,6.726020,2.518628,-7.268566,1.670809,3.288895,-8.522313,2.689470,5.854086,7.502139,4.937612,3.011156,6.896347,8.391803,-3.501194,-7.336226,6.766017,3.313947,2.015666,3.339000,0.057547,-8.046395,-7.076442,5.566572,-4.360493,3.542529],[-7.358189,9.635909,3.689853,-8.946380,-0.411161,7.355017,-6.800266,-6.935738,-3.259326,-6.823823,-1.740532,-2.860315,5.207084,3.092014,-2.768332,5.770125,0.773704,-5.461165,6.823384,-1.696256,1.963169,7.426686,-0.476050,5.627433,5.531557,0.586256,2.984564,7.746166,4.061340,-2.787901],[4.737939,9.069649,-0.695085,-4.800210,-7.289932,1.332137,-2.725236,4.209232,-5.401253,-9.809087,-6.584761,-1.704474,2.425082,-7.378575,-9.120530,-9.621723,0.452574,-7.369791,3.003235,1.153544,8.880475,-6.725484,-9.262617,-6.981642,-5.109288,-7.404235,-1.435537,-7.613825,-3.270690,5.537299],[-6.578662,2.692819,0.681462,0.675007,-6.046635,-5.396270,-7.939746,3.425004,-8.396393,4.378287,-6.652207,9.104254,-6.033000,-5.743012,2.501851,5.290862,4.241868,1.901148,-8.825041,-9.933645,1.770128,-1.087572,0.217141,-2.589971,-4.259531,-8.685664,-6.077292,5.769985,-6.721905,1.568737],[0.417592,4.823930,2.011677,8.697818,2.232198,-1.574717,-5.837285,-0.340089,2.538036,1.481008,4.555630,-8.580523,-9.895389,-0.591345,-3.208320,-1.806135,6.193813,-0.658529,-9.703753,-1.348899,0.236780,-7.563909,-5.713086,1.573421,-9.996904,4.765586,6.526199,9.375254,6.541308,-1.124171],[3.028105,-4.570068,-8.214979,-4.978911,-8.933970,-5.921286,0.878623,-6.909158,-6.516342,7.067142,-1.380694,-9.861604,3.505620,0.867847,2.896746,-1.561646,-3.525808,5.641215,4.370537,7.570685,7.465183,-1.428645,7.997139,-9.203288,-9.109658,-2.040589,-4.754229,-3.370138,-7.402951,-0.601106],[1.879819,-6.809731,7.273730,-7.361251,9.885032,0.746205,-7.638839,-3.553106,-7.130653,7.039962,-0.922192,0.535174,-5.840281,-3.505058,-0.194025,-3.575115,1.662579,7.989835,2.153295,-3.749366,-1.704010,4.094038,1.291906,-6.922251,-5.723773,-7.044404,0.455591,4.465328,-2.475732,-2.055829],[-4.635233,-1.950266,1.967894,8.226741,-0.764191,-3.243374,9.710632,1.800667,7.343129,-4.141812,1.306086,-5.812289,-0.957976,-7.500692,-1.604427,8.580768,6.256128,-5.926424,-7.103593,7.793628,3.316033,0.043520,2.306968,9.269283,-5.988070,-2.640327,8.212113,-1.922061,-3.924844,-4.860837],[5.194213,-9.306573,0.291296,3.140413,-0.393392,-9.575740,-4.601937,-8.493928,-6.771240,-2.397550,7.505190,-4.850053,-4.854639,-4.026270,1.304422,3.425705,6.208979,-7.406948,-7.478410,-3.728550,-1.635543,6.185326,-7.445428,2.768723,9.459267,3.680703,-7.965125,2.036922,-5.613886,-6.431100],[-3.721682,-6.648904,3.586671,8.465486,-6.778815,-0.344575,-2.403297,-8.036394,0.916441,-6.070480,8.997014,6.813163,3.419523,-1.797521,-1.332740,-3.730747,-1.909765,-2.352570,8.213342,-6.301699,8.300961,-7.725883,-7.093987,-3.519364,-7.561603,-2.815367,-0.641958,-4.699779,2.715643,1.500303],[3.965816,-7.618964,-7.259210,7.561473,-1.939619,-0.012369,-4.673539,-5.682123,0.771910,-9.303574,6.427524,7.545440,-8.875310,3.576474,6.198563,4.659349,-6.780450,-7.761229,0.961090,-6.484220,1.072183,2.567226,-4.026545,-4.880133,3.661184,4.471500,-6.765189,-3.443435,-2.795820,-2.671833],[5.014916,-3.716288,5.862583,-7.610489,-4.753017,1.979113,4.697697,3.506888,-6.025815,8.589794,-2.204885,-9.390121,-9.531105,6.456914,-1.196250,-2.791275,-0.155145,-1.294535,5.917582,-7.646282,4.868377,-5.966664,-6.051271,-6.292283,1.957675,-6.856500,4.609099,-2.480031,-2.268116,-3.325491],[2.927752,-1.351827,-3.256544,-2.033513,6.704108,7.531493,-9.663780,-3.179500,-2.844303,0.050418,-4.771798,6.490004,-5.524038,8.968712,-4.150564,8.329341,-5.679453,7.915343,-5.701572,1.926286,-4.116977,2.857001,-2.829770,5.978661,-9.545468,-4.260526,2.077449,-2.679519,-3.360558,-7.722674],[-4.248385,-5.394005,8.801454,9.606220,1.131246,1.205412,1.158778,-4.656367,-5.166902,-2.164738,-5.990243,-8.751480,1.196614,6.595800,-6.951901,-6.755084,-1.123219,-5.885877,-9.678578,7.870992,-8.471378,5.851336,5.615160,9.482595,3.887821,6.736099,-9.531020,3.488358,-1.053869,4.424973],[-7.647835,6.175922,9.709297,-4.530117,7.969372,3.875039,1.822531,-2.055615,6.442803,-1.339054,9.532219,-8.311453,0.694935,2.317192,8.293369,2.342063,-6.284037,-8.399718,-3.515871,7.301594,-9.226766,1.764412,3.714299,0.081839,2.596958,-7.924248,6.093912,-8.079118,-5.235451,-5.503146],[3.994877,-0.670799,1.739890,8.458696,0.534685,2.059205,4.434593,4.194261,-2.366520,3.730973,6.391539,7.446097,0.245111,7.731971,7.762249,9.223111,0.011676,9.458199,-4.446162,-1.538328,5.952702,6.875954,-6.339270,-0.942918,6.364146,-7.237625,1.688839,5.587854,-9.282600,-8.757763],[2.860208,-1.709136,3.674988,3.892521,-2.201338,-1.693745,8.870323,-5.018432,-6.803885,3.742400,3.623420,-9.029205,2.155681,0.780097,8.369044,1.532642,0.369594,5.525949,7.716105,-0.857068,6.904992,-1.481667,-9.373914,-0.904994,2.079634,4.223350,-8.015990,3.572801,7.937613,8.297817],[-3.600219,4.902444,-6.064409,6.817000,0.380249,4.010605,-8.002800,-6.909667,-7.923297,1.880534,-9.086620,-0.438938,5.230170,-8.387370,-7.999236,7.102225,-5.206917,-9.667188,3.909247,2.564077,5.296263,3.728348,-6.211139,0.698317,7.228439,-9.717364,3.779611,-8.067971,-3.004660,3.236631]], dtype = "float32")#candidate|2026|(18, 30)|const|float32
call_2023 = relay.TupleGetItem(func_1949_call(relay.reshape(const_2024.astype('int8'), [11, 16, 1]), relay.reshape(var_2025.astype('int8'), [1232,]), relay.reshape(const_2026.astype('float32'), [5, 12, 9]), ), 8)
call_2027 = relay.TupleGetItem(func_1953_call(relay.reshape(const_2024.astype('int8'), [11, 16, 1]), relay.reshape(var_2025.astype('int8'), [1232,]), relay.reshape(const_2026.astype('float32'), [5, 12, 9]), ), 8)
bop_2033 = relay.floor_divide(bop_1988.astype('float64'), relay.reshape(uop_1986.astype('float64'), relay.shape_of(bop_1988))) # shape=(10, 16, 9)
uop_2038 = relay.tan(bop_1993.astype('float32')) # shape=(10, 16, 9)
output = relay.Tuple([call_2023,const_2024,var_2025,const_2026,bop_2033,uop_2038,])
output2 = relay.Tuple([call_2027,const_2024,var_2025,const_2026,bop_2033,uop_2038,])
func_2048 = relay.Function([var_1992,var_2025,], output)
mod['func_2048'] = func_2048
mod = relay.transform.InferType()(mod)
mutated_mod['func_2048'] = func_2048
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2048_call = mutated_mod.get_global_var('func_2048')
var_2050 = relay.var("var_2050", dtype = "float32", shape = (10, 16, 9))#candidate|2050|(10, 16, 9)|var|float32
var_2051 = relay.var("var_2051", dtype = "int8", shape = (1232,))#candidate|2051|(1232,)|var|int8
call_2049 = func_2048_call(var_2050,var_2051,)
output = call_2049
func_2052 = relay.Function([var_2050,var_2051,], output)
mutated_mod['func_2052'] = func_2052
mutated_mod = relay.transform.InferType()(mutated_mod)
func_695_call = mod.get_global_var('func_695')
func_696_call = mutated_mod.get_global_var('func_696')
call_2085 = relay.TupleGetItem(func_695_call(), 0)
call_2086 = relay.TupleGetItem(func_696_call(), 0)
func_396_call = mod.get_global_var('func_396')
func_399_call = mutated_mod.get_global_var('func_399')
const_2111 = relay.const([-1.177004,2.942054,0.436024,-2.895725,-7.674545,2.824784,-8.940878,-4.826055,-3.116619,-2.300570,-9.102464,5.304149,1.261038,9.535684,2.764681,-2.760644,-6.604429,4.356017,-3.620275,-1.433525,9.222538,-2.646024,2.675814,-6.845365,2.107307,-8.057599,3.742735,2.270084,-4.690478,-4.670728,-3.899707,3.961229,-9.594791,-2.725838,-8.676230,1.734322,5.793778,0.333660,-7.766704,-4.505155,-2.443925,6.301119,-9.375387,6.805134,0.328540,-7.406301,-6.568827,-6.997632,1.356258,-3.322636,-9.076347,8.431333,6.094565,-4.125489,1.945889,-2.633578,1.982315,9.655494,-2.827303,-7.864436,-4.715581,3.754135,-3.668433,0.339769,1.874279,3.179625,0.507775,0.059814,-7.078143,-7.988994,-4.463777,6.275142,6.941448,7.997576,8.748500,3.886034,4.119345,2.569847,3.974602,1.193906,-1.406418,-2.736038,0.065871,-2.080565,9.739454,2.679559,4.322359,6.824034,-5.909062,0.107524,-9.677838,4.701166,2.168732,2.350074,-4.136435,0.147275,8.566776,-1.810533,6.434272,-3.540959,-4.935847,-9.402149,2.336395,-4.586530,5.828175,5.899311,-4.137926,6.246600,7.671923,-6.131408,9.788193,-9.314348,6.019259,4.907818,-1.997202,1.926029,-3.574164,-8.583140,5.941355,0.804792,-3.381797,-0.188918,7.235659,4.353158,-7.278375,-8.232644,3.425004,4.052687,-2.777525,-5.461785,-9.703939,1.497697,6.626774,-0.670520,-3.744799,8.503417,6.043286,7.710135,8.303198,0.919855,9.628958,4.495668,5.811771,2.123612,-5.319234,-0.581837,-8.193344,7.523013,6.969687,-1.964400,5.205112,7.016023,-0.638698,7.808126,9.240278,5.246159,8.939855,4.769903,5.566883,-9.061904,-6.753050,8.021671,-9.662045,8.562187,-6.900169,5.677727,3.042219,-1.044971,2.671053,-0.500197,6.148157,0.653866,-9.884067,9.100723,-6.944112,1.955183,-9.137768,6.789539,5.516041,-4.552399,-6.744884,2.549404,-4.031522,3.898225,-7.251012,-2.489555,-0.171981,-7.727702,1.978602,-5.284970,8.796917,-5.695366,-8.350186,-2.900808,-3.935866,-4.853993,-3.960101,2.532926,0.433844,-3.535656,-6.418334,1.905310,9.059669,3.218230,1.206301,-3.266126,-9.462362,-8.724633,-2.014061,-8.608442,-9.341381,4.235076,2.355243,7.524299,-8.795681,-7.568146,6.247673,1.086455,3.363725,7.494304,-5.079730,4.732771,-9.706741,-9.825278,-9.081409,-0.836567,4.593817,4.580524,-2.088524,6.157967,-2.969154,7.121164,-6.865008,-0.214439,5.594023,-6.475723,7.418226,-6.250600,-9.076900,-2.760392,-7.495900,4.533487,0.429125,-9.030215,8.615292,-4.060241,8.657532,0.238042,-4.259514,1.858418,9.077827,3.269053,6.389846,2.474497,-5.753343,-7.139534,-7.657449,-2.736242,-8.872605,2.372375,7.357477,1.991086,7.247632,-9.880977,0.449481,5.675739,7.095574,6.384689,6.188713,-8.712796,8.484902,-1.001595,-5.797705,4.082538,7.222571,6.083083,-2.229012,7.579097,6.275145,9.976525,2.897736,-6.142460,3.889092,1.075465,-4.699554,0.055621,-7.252863,4.442147,-0.346549,2.456734,6.905293,-6.827942,6.256945,3.976564,2.583021,7.168158,4.077296,-7.418040,-8.728833,1.683917,1.152812,-3.751876,-9.700581,2.922841,-3.079723,-4.283108,6.577934,3.161180,-4.329348,8.644892,2.811269,-4.723072,-0.274933,0.351710,-8.475181,4.163439,2.360477,6.036141,1.845544,5.518481,5.974992,0.826795,-4.542617,-5.178879,2.915789,3.383925,-3.044970,-3.102867,-5.916880,0.052514,-6.536105,-4.340075,-1.094927,-8.945211,-4.132371,9.683311,3.694366,-1.385797,3.849785,1.872728,3.553891,9.404646,-9.552748,-2.992983,-2.558857,-2.458200,8.339217,-1.650635,-0.255109,5.531738,-8.189827,7.517166,-0.108492,-7.289519,-5.237632,-5.327493,-9.727775,9.886033,6.442165,8.546260,-5.360469,8.294237,-3.380912,8.961845,-9.116575,-0.989302,-5.270814,-4.135933,4.676168,-0.648832,-8.492098,7.749846,2.743328,-0.830744,0.109580,-3.722499,9.515064,1.955276,7.013751,-4.703784,8.119370,9.668606,-8.770643,-0.153970,-2.754951,4.201349,-0.270314,9.484542,-8.218054,-4.724014,0.753939,3.392477,-2.819960,-6.316954,2.321840,5.318647,3.588860,-7.181406,6.803354,-2.295060,-1.725045,-8.881011,-8.516205,-7.814423,4.881712,-9.332855,9.859196,1.154883,-3.973303,8.418137,5.125388,3.704172,0.088176,-0.818725,-6.084607,-4.460615,1.438023,8.597610,-7.233909,-1.258158,3.909686,-0.109996,-2.414719,-3.365973,-6.703181,-5.496370,9.906566,-1.412190,-5.572527,3.338847,1.526463,4.380398,-5.222766,7.322768,-9.816554,2.714411,-0.049789,-6.853964,9.153905,-5.159566,-2.236129,1.040702,-9.556892,0.548334,7.412785,6.367172,-8.384683,5.023287,-9.877220,-0.076107,-7.072565,-3.772855,5.822774,5.094228,5.608889,-0.734881,-1.620156,6.006282,-5.791546,0.244607,0.733638,-7.689726,4.494281,9.074335,-4.447997,-1.764594,6.352841,-0.884946,8.555270,5.497008,5.235217,1.563808,-0.062748,-4.833390,6.178357,-7.293142,-7.240014,-8.916588,-2.257150,3.830810,1.690310,1.715576,-4.149920,-0.930537,1.926096,4.286859,0.247937,-2.479662,7.046458,-8.992919,2.761134,-2.257383,-9.977826,5.083721,-6.020778,2.477008,1.137997,-7.220417,-8.744698,8.081614,-0.473534,-0.164929,1.679883,1.665740,1.765828,3.202053,-7.249803,7.669569,5.062735,4.184612,8.598236,9.614540,8.498291,0.879355,-9.832642,1.613380,2.531024,5.371358,9.777281,9.137069,-8.405362,1.247633,-6.793897,2.225826,7.680120,2.841507,-8.900162,8.338409,-4.466668,5.931106,-2.910625,-2.193025,0.357575,-5.180946,-3.547235,8.314042,8.632336,-6.039329,5.380298,2.744480,-9.314162,-3.337178,2.928168,0.098684,-8.799161,6.402352,4.681771,-6.979335,8.501225,5.207529,-8.311508,-0.451094,4.619288,-3.131675,1.621147,-9.326424,-4.010219,-1.928153,-2.185442,-0.328644,-2.322375,7.621428,5.307386,-2.941753,0.523158,-4.877981,0.217931,1.754043,1.710546,9.098423,2.993587,3.012863,-3.270200,7.518042,2.073232,9.719618,8.217409,3.449333,-9.254773,1.984651,-8.990288,-3.664637,3.914018,-8.871580,3.598057,8.043212,4.458129,-9.798831,-6.821856,0.406185,7.686381,5.138673,3.988077,2.490575,9.969442,9.124455,9.883702,7.456644,8.938703,0.489290,7.744261,3.507700,4.385659,5.141369,-9.562252,5.548148,0.475989,-4.226967,-2.670453,6.323288,3.920823,8.806164,-7.771160,5.606395,-2.328398,-9.919331,4.601684,7.244534,3.713837,-3.017015,0.733233,-9.925762,-4.315501,1.404625,-3.822972,-2.959229,-2.396708,-1.221393,-9.245203,-4.053202,3.612429,3.516202,4.018245,0.533221,3.622747,-7.980664,-9.275458,7.899198,0.088431,9.435780], dtype = "float64")#candidate|2111|(640,)|const|float64
call_2110 = relay.TupleGetItem(func_396_call(relay.reshape(const_2111.astype('float64'), [4, 16, 10])), 2)
call_2112 = relay.TupleGetItem(func_399_call(relay.reshape(const_2111.astype('float64'), [4, 16, 10])), 2)
output = relay.Tuple([call_2085,call_2110,const_2111,])
output2 = relay.Tuple([call_2086,call_2112,const_2111,])
func_2117 = relay.Function([], output)
mod['func_2117'] = func_2117
mod = relay.transform.InferType()(mod)
output = func_2117()
func_2118 = relay.Function([], output)
mutated_mod['func_2118'] = func_2118
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1295_call = mod.get_global_var('func_1295')
func_1296_call = mutated_mod.get_global_var('func_1296')
call_2135 = func_1295_call()
call_2136 = func_1295_call()
output = relay.Tuple([call_2135,])
output2 = relay.Tuple([call_2136,])
func_2137 = relay.Function([], output)
mod['func_2137'] = func_2137
mod = relay.transform.InferType()(mod)
mutated_mod['func_2137'] = func_2137
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2137_call = mutated_mod.get_global_var('func_2137')
call_2138 = func_2137_call()
output = call_2138
func_2139 = relay.Function([], output)
mutated_mod['func_2139'] = func_2139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_907_call = mod.get_global_var('func_907')
func_909_call = mutated_mod.get_global_var('func_909')
call_2148 = func_907_call()
call_2149 = func_907_call()
output = call_2148
output2 = call_2149
func_2151 = relay.Function([], output)
mod['func_2151'] = func_2151
mod = relay.transform.InferType()(mod)
mutated_mod['func_2151'] = func_2151
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2151_call = mutated_mod.get_global_var('func_2151')
call_2152 = func_2151_call()
output = call_2152
func_2153 = relay.Function([], output)
mutated_mod['func_2153'] = func_2153
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2187 = relay.const([[5,1,-9,3,1,-8,6,-8,-7,-1,8],[-3,-7,6,-5,3,9,9,10,3,-7,8],[2,10,-4,4,-8,-8,2,-7,6,-6,-6],[2,8,-7,-6,-10,-10,6,3,7,-1,-8],[-10,-9,-5,-1,3,-6,9,-9,8,-1,-2],[-4,1,8,-8,-4,-1,7,1,8,-1,6],[-10,3,6,8,-3,5,-6,3,-9,-4,-1]], dtype = "int64")#candidate|2187|(7, 11)|const|int64
var_2188 = relay.var("var_2188", dtype = "int64", shape = (7, 11))#candidate|2188|(7, 11)|var|int64
bop_2189 = relay.maximum(const_2187.astype('int64'), relay.reshape(var_2188.astype('int64'), relay.shape_of(const_2187))) # shape=(7, 11)
output = bop_2189
output2 = bop_2189
func_2208 = relay.Function([var_2188,], output)
mod['func_2208'] = func_2208
mod = relay.transform.InferType()(mod)
mutated_mod['func_2208'] = func_2208
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2209 = relay.var("var_2209", dtype = "int64", shape = (7, 11))#candidate|2209|(7, 11)|var|int64
func_2208_call = mutated_mod.get_global_var('func_2208')
call_2210 = func_2208_call(var_2209)
output = call_2210
func_2211 = relay.Function([var_2209], output)
mutated_mod['func_2211'] = func_2211
mutated_mod = relay.transform.InferType()(mutated_mod)
func_87_call = mod.get_global_var('func_87')
func_89_call = mutated_mod.get_global_var('func_89')
call_2247 = func_87_call()
call_2248 = func_87_call()
func_897_call = mod.get_global_var('func_897')
func_898_call = mutated_mod.get_global_var('func_898')
call_2270 = relay.TupleGetItem(func_897_call(), 0)
call_2271 = relay.TupleGetItem(func_898_call(), 0)
output = relay.Tuple([call_2247,call_2270,])
output2 = relay.Tuple([call_2248,call_2271,])
func_2280 = relay.Function([], output)
mod['func_2280'] = func_2280
mod = relay.transform.InferType()(mod)
output = func_2280()
func_2281 = relay.Function([], output)
mutated_mod['func_2281'] = func_2281
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2151_call = mod.get_global_var('func_2151')
func_2153_call = mutated_mod.get_global_var('func_2153')
call_2293 = func_2151_call()
call_2294 = func_2151_call()
func_1527_call = mod.get_global_var('func_1527')
func_1529_call = mutated_mod.get_global_var('func_1529')
const_2303 = relay.const([[-9.882496,7.561233,-2.884250,3.210959],[0.944796,-8.056209,-4.076338,8.760117],[1.065956,-4.057382,4.470355,5.238404],[-6.649636,-4.105465,-7.651511,-1.304891],[5.793268,4.808506,-8.128255,3.906858],[-3.068040,-2.583928,0.270760,-5.880639],[0.888808,-4.152832,3.066706,8.487716],[9.807438,-1.715246,8.912982,-2.918467],[1.664639,-3.677212,-9.070447,1.734021],[4.636936,1.309874,4.161696,4.416954],[-6.371556,-5.031522,4.699607,1.901909],[4.900146,-9.165109,-0.516031,9.571599],[5.343016,5.772650,-2.473595,-0.854076],[0.189481,-6.223365,3.114256,8.738577],[4.491587,0.616013,-8.148916,6.634412],[1.296025,-1.135704,-3.221512,5.091979],[7.395876,7.658176,1.386918,4.729564],[1.287868,-5.364632,8.162950,4.381818],[8.580264,0.256277,7.127172,2.958862],[-9.437716,3.971308,-4.210306,-8.938288],[1.073860,1.751287,8.466695,0.043727],[-7.216177,-0.750324,-2.795873,-2.567827],[-2.837216,1.659220,2.110692,-6.260229],[-1.763301,-2.460669,-4.001170,-5.763197],[9.363354,0.840006,-2.260231,3.792701],[-2.695140,-9.524277,1.145921,-4.461203],[-6.485803,-7.819225,-2.348605,8.724741],[-1.373431,5.431364,3.578167,-8.125598],[3.979951,1.361531,1.082788,-2.336059],[-5.318492,-3.644294,-6.144236,-5.724039],[-4.127437,-2.186839,3.295374,0.282985],[9.825005,9.829410,-8.817900,5.253840],[7.177728,2.110593,-4.604049,-3.465965],[-0.233427,-3.471283,9.353879,0.854371],[3.088020,-7.099273,1.210873,-6.152033],[4.451536,5.504177,-6.823439,3.819648],[-4.172581,-3.159720,1.943825,6.739110],[3.436208,-7.989063,-9.504029,-6.923448],[9.542400,0.597975,-8.399203,5.475155],[5.888753,8.095228,-3.867299,1.394719],[0.880680,6.925148,-7.092520,-7.562252],[-5.807768,-9.636841,-7.531825,9.693815],[0.689974,-7.638268,-8.881334,9.754613],[6.237292,7.018233,-2.650134,-1.386268],[-3.433621,-0.405622,-3.664766,-0.844912],[0.588050,-5.250055,7.681730,3.809602],[-4.848418,-8.380891,2.655522,-6.660590],[-9.815608,-3.815792,-5.364802,-5.915077],[-7.902025,9.830499,6.959304,-1.372165],[-5.897704,-0.498984,-2.501439,-3.903761],[-1.261165,3.770500,1.843741,-2.097608],[-3.422948,-6.584229,0.800214,4.116816],[3.218159,-7.649101,4.701504,8.763269],[4.136403,7.154562,9.823803,-5.980141],[8.605976,6.226925,3.788702,-3.446646],[8.126684,-9.584500,-9.376884,-3.885720],[-3.303621,-4.915211,8.275813,6.022756],[-9.905399,-5.439864,5.360172,8.963551],[8.976004,7.718232,-1.184863,-4.541776],[-4.934382,-9.103488,-3.747056,-6.186303],[3.511349,-8.168882,-6.630998,-1.372565],[9.961258,-2.750470,8.868930,6.710814],[-8.291206,-7.641753,-4.119084,-7.444203],[2.061061,4.029350,-6.505773,6.443761],[0.989401,-8.011236,6.931481,-2.802787],[-2.207575,5.031494,-9.082621,-4.304544],[5.304145,6.510699,3.989317,-9.515750],[-1.062148,-4.611029,8.368350,-7.894391],[-0.583605,-0.070513,0.975527,-0.643744],[0.807052,-1.456433,-3.010714,0.612289],[4.140783,7.300761,-1.815746,3.349905],[-3.062769,6.070514,8.493119,-6.535763],[1.003774,-8.764659,2.868347,-3.941257],[-4.112890,-7.933973,2.407660,8.868556],[-9.981804,9.107432,-2.867689,3.178683],[-1.774670,5.141219,-6.495761,1.088541],[-8.655011,5.238131,-0.819806,1.084869],[-5.210423,3.919545,2.952346,-5.480769],[-6.689590,-9.936408,-3.968640,6.399760],[6.000512,8.479656,-2.484973,9.627523],[-3.832215,-3.497413,-3.063582,5.642110],[0.144101,-0.124333,2.433235,-6.419657],[-7.005651,5.889766,-8.849784,0.677565],[7.360488,8.285310,-0.805741,-9.611755],[6.959817,4.085507,-1.314638,-7.223315],[-4.602912,-4.968730,7.492549,4.331210],[-9.975854,1.383075,0.754727,-0.938999],[8.991635,-0.104483,-1.936405,-8.626609],[2.523257,8.102474,-8.052270,4.224112],[-9.472829,3.806123,-2.416129,7.970598],[9.644994,9.283932,2.697522,9.003122],[7.492387,-7.456125,-0.792435,-5.394028],[6.004882,-6.556138,4.341224,-1.118317],[4.666652,-8.421242,-4.327892,-6.023179],[-1.696761,-7.855964,5.214863,-4.797209],[0.867014,-4.324650,7.102575,2.245281],[-3.808376,-2.794573,5.619612,8.204205],[-0.414410,-1.908319,6.569287,-6.834316],[-0.071319,-7.027220,-2.383567,-8.366276],[4.461888,-5.587160,2.295033,0.630032],[7.569218,-3.166264,-9.545149,4.815468],[7.245214,9.375431,1.279384,-3.496229],[-4.094593,3.045048,8.482010,-1.610013],[0.637328,8.471698,-9.556803,-2.594303],[-3.335463,-8.220181,4.705054,-8.606339],[4.870850,9.655118,0.571200,-7.494845],[-8.690595,2.633187,-3.161260,2.208893],[-4.978864,-7.281391,1.837276,-5.651329],[4.995102,2.712852,-9.380738,1.639467],[3.007581,-3.444978,-8.441405,0.546566],[5.533554,-4.815378,-9.607108,8.292522],[-5.880795,-7.572444,-1.789969,-2.756888],[8.997077,-2.587881,2.745307,-1.044322],[9.443777,7.309030,5.986872,-5.545763],[1.787792,-0.367357,-9.966775,-6.336894],[-4.397179,6.020159,3.451721,-2.613420],[0.785039,3.675356,9.419098,0.862580],[-1.953275,-6.656158,9.462046,-9.933746],[-5.240251,3.639581,-2.132267,0.023352],[-2.378482,-3.316440,-1.065273,-1.287804],[7.743254,-6.041947,1.018493,3.341676],[9.018876,-5.677437,7.810033,5.559790],[0.570508,-9.346600,6.613655,-5.495167],[-3.126003,2.681301,-1.566679,-8.301342],[5.507354,-5.629441,0.465403,5.381202],[1.203081,3.725910,-0.049717,-1.596023],[0.185534,-3.065227,4.989520,-9.915469],[0.871358,-3.068943,-3.366791,0.342974],[-7.147380,-9.375258,-9.873796,-6.515114],[8.701231,-2.118971,-8.896134,-1.121096],[8.040933,-2.711310,2.349456,0.837360],[3.144718,-5.910690,-3.770910,0.742693],[3.568139,-3.257654,3.746138,-3.166939],[2.686835,2.923899,6.146445,1.811883],[-2.706383,7.992159,3.939306,-3.117473],[4.564849,-9.453829,-8.380724,-1.823666],[8.549850,2.229457,0.242859,-7.268393],[-8.328976,-9.771044,1.324860,9.832733],[-9.282519,4.411484,0.565182,-1.010988],[5.645918,-8.197164,-0.381068,3.987232],[-5.896460,2.687689,-7.725989,-4.343205],[3.044937,1.878683,-1.710748,-3.663018],[3.156724,-7.211361,9.209117,4.433713],[4.919472,-4.212848,-0.241889,-0.860702],[6.724541,-8.170264,8.831103,-6.423136],[-3.742899,7.172442,-9.898611,5.551268],[8.780076,-1.627926,-2.337265,8.808601],[-4.733170,4.244744,-0.028585,2.449090],[7.142588,3.010253,-6.075099,7.956221],[8.275219,3.670969,-9.924610,-8.684557],[1.338576,-6.516437,-6.541110,1.062725],[-4.619594,-2.546410,-4.634783,-1.544099],[8.969198,-8.935353,5.210152,-2.865289],[-1.442162,3.976317,-3.102625,7.042069],[-6.316595,6.100089,9.790249,2.846283],[-4.232335,-8.243892,7.121303,-9.474904],[2.690373,-1.101909,4.276414,8.856036],[3.211642,-4.470100,-0.848801,1.952817],[-7.987063,-7.261932,4.620438,0.992027],[7.893394,5.472021,-9.213451,4.503929]], dtype = "float64")#candidate|2303|(160, 4)|const|float64
call_2302 = relay.TupleGetItem(func_1527_call(relay.reshape(const_2303.astype('float64'), [1, 640])), 4)
call_2304 = relay.TupleGetItem(func_1529_call(relay.reshape(const_2303.astype('float64'), [1, 640])), 4)
output = relay.Tuple([call_2293,call_2302,const_2303,])
output2 = relay.Tuple([call_2294,call_2304,const_2303,])
func_2328 = relay.Function([], output)
mod['func_2328'] = func_2328
mod = relay.transform.InferType()(mod)
mutated_mod['func_2328'] = func_2328
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2328_call = mutated_mod.get_global_var('func_2328')
call_2329 = func_2328_call()
output = call_2329
func_2330 = relay.Function([], output)
mutated_mod['func_2330'] = func_2330
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2362 = relay.var("var_2362", dtype = "float64", shape = (7, 12))#candidate|2362|(7, 12)|var|float64
uop_2363 = relay.cosh(var_2362.astype('float64')) # shape=(7, 12)
output = uop_2363
output2 = uop_2363
func_2367 = relay.Function([var_2362,], output)
mod['func_2367'] = func_2367
mod = relay.transform.InferType()(mod)
mutated_mod['func_2367'] = func_2367
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2368 = relay.var("var_2368", dtype = "float64", shape = (7, 12))#candidate|2368|(7, 12)|var|float64
func_2367_call = mutated_mod.get_global_var('func_2367')
call_2369 = func_2367_call(var_2368)
output = call_2369
func_2370 = relay.Function([var_2368], output)
mutated_mod['func_2370'] = func_2370
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1199_call = mutated_mod.get_global_var('func_1199')
call_2393 = relay.TupleGetItem(func_1197_call(), 2)
call_2394 = relay.TupleGetItem(func_1199_call(), 2)
func_2328_call = mod.get_global_var('func_2328')
func_2330_call = mutated_mod.get_global_var('func_2330')
call_2397 = relay.TupleGetItem(func_2328_call(), 0)
call_2398 = relay.TupleGetItem(func_2330_call(), 0)
bop_2401 = relay.right_shift(call_2393.astype('uint32'), relay.reshape(call_2397.astype('uint32'), relay.shape_of(call_2393))) # shape=(1, 16, 1)
bop_2404 = relay.right_shift(call_2394.astype('uint32'), relay.reshape(call_2398.astype('uint32'), relay.shape_of(call_2394))) # shape=(1, 16, 1)
output = bop_2401
output2 = bop_2404
func_2405 = relay.Function([], output)
mod['func_2405'] = func_2405
mod = relay.transform.InferType()(mod)
mutated_mod['func_2405'] = func_2405
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2405_call = mutated_mod.get_global_var('func_2405')
call_2406 = func_2405_call()
output = call_2406
func_2407 = relay.Function([], output)
mutated_mod['func_2407'] = func_2407
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1295_call = mod.get_global_var('func_1295')
func_1296_call = mutated_mod.get_global_var('func_1296')
call_2447 = func_1295_call()
call_2448 = func_1295_call()
uop_2455 = relay.asinh(call_2447.astype('float64')) # shape=(12, 16, 9)
uop_2457 = relay.asinh(call_2448.astype('float64')) # shape=(12, 16, 9)
bop_2464 = relay.bitwise_or(call_2447.astype('uint32'), relay.reshape(uop_2455.astype('uint32'), relay.shape_of(call_2447))) # shape=(12, 16, 9)
bop_2467 = relay.bitwise_or(call_2448.astype('uint32'), relay.reshape(uop_2457.astype('uint32'), relay.shape_of(call_2448))) # shape=(12, 16, 9)
output = relay.Tuple([bop_2464,])
output2 = relay.Tuple([bop_2467,])
func_2470 = relay.Function([], output)
mod['func_2470'] = func_2470
mod = relay.transform.InferType()(mod)
mutated_mod['func_2470'] = func_2470
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2470_call = mutated_mod.get_global_var('func_2470')
call_2471 = func_2470_call()
output = call_2471
func_2472 = relay.Function([], output)
mutated_mod['func_2472'] = func_2472
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1197_call = mod.get_global_var('func_1197')
func_1199_call = mutated_mod.get_global_var('func_1199')
call_2489 = relay.TupleGetItem(func_1197_call(), 2)
call_2490 = relay.TupleGetItem(func_1199_call(), 2)
var_2494 = relay.var("var_2494", dtype = "int8", shape = (13, 16, 16))#candidate|2494|(13, 16, 16)|var|int8
bop_2495 = relay.power(call_2489.astype('float64'), var_2494.astype('float64')) # shape=(13, 16, 16)
bop_2498 = relay.power(call_2490.astype('float64'), var_2494.astype('float64')) # shape=(13, 16, 16)
output = relay.Tuple([bop_2495,])
output2 = relay.Tuple([bop_2498,])
func_2508 = relay.Function([var_2494,], output)
mod['func_2508'] = func_2508
mod = relay.transform.InferType()(mod)
mutated_mod['func_2508'] = func_2508
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2509 = relay.var("var_2509", dtype = "int8", shape = (13, 16, 16))#candidate|2509|(13, 16, 16)|var|int8
func_2508_call = mutated_mod.get_global_var('func_2508')
call_2510 = func_2508_call(var_2509)
output = call_2510
func_2511 = relay.Function([var_2509], output)
mutated_mod['func_2511'] = func_2511
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2405_call = mod.get_global_var('func_2405')
func_2407_call = mutated_mod.get_global_var('func_2407')
call_2559 = func_2405_call()
call_2560 = func_2405_call()
output = relay.Tuple([call_2559,])
output2 = relay.Tuple([call_2560,])
func_2587 = relay.Function([], output)
mod['func_2587'] = func_2587
mod = relay.transform.InferType()(mod)
output = func_2587()
func_2588 = relay.Function([], output)
mutated_mod['func_2588'] = func_2588
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1829_call = mod.get_global_var('func_1829')
func_1831_call = mutated_mod.get_global_var('func_1831')
call_2620 = relay.TupleGetItem(func_1829_call(), 0)
call_2621 = relay.TupleGetItem(func_1831_call(), 0)
uop_2622 = relay.cos(call_2620.astype('float32')) # shape=(12, 16, 9)
uop_2624 = relay.cos(call_2621.astype('float32')) # shape=(12, 16, 9)
uop_2636 = relay.acosh(uop_2622.astype('float64')) # shape=(12, 16, 9)
uop_2638 = relay.acosh(uop_2624.astype('float64')) # shape=(12, 16, 9)
const_2639 = relay.const([[[-3.418441,9.918427,1.178469,-3.333427,-9.398839,-4.302180,0.527675,-3.244967,-6.397598],[-1.297746,6.374563,8.849934,8.896006,8.376495,6.824252,-5.318469,-2.958823,2.460434],[-0.186020,0.561618,-1.260406,-2.473302,-2.542304,3.714931,5.492656,8.810903,-2.369601],[-2.265118,-7.125052,3.558937,-9.758654,3.143389,-9.572502,1.867410,-9.118585,5.648020],[-9.266713,-7.796128,-9.092501,6.426248,6.383750,-8.187627,-1.483178,-8.804976,5.548458],[-3.665099,9.114609,-8.152219,-6.652169,-9.921374,-7.741047,-1.415219,2.349216,3.518520],[-6.813418,-2.114128,-1.993269,3.807196,7.023174,2.373118,-8.288016,-0.067876,6.896017],[3.850410,-3.216220,9.319767,5.406370,7.746628,-3.688614,7.445567,-5.171926,2.092749],[-9.192950,6.345961,-6.162438,-3.099830,-8.949068,-0.532850,2.189413,5.857334,-6.165811],[-7.643292,-4.265784,-2.541076,5.506789,1.276126,3.251802,-4.840606,-8.346763,1.226856],[-2.601987,2.473887,8.757238,-1.086439,6.937947,-5.113264,-0.736696,7.177925,3.908754],[-6.733225,3.705938,-0.765466,-9.901044,0.699366,-1.170229,4.538200,-1.513989,9.549837],[-2.506617,0.872871,8.962876,1.638176,-2.222933,9.044812,3.163499,-5.953827,6.409326],[8.982905,9.997825,-9.194682,-9.353301,-4.237304,-8.965029,-3.298553,-6.172820,-1.101890],[8.994834,8.851423,9.264580,8.286146,-9.980875,1.835385,-7.829680,9.295104,2.903642],[6.268824,4.775865,7.181710,6.225178,8.455516,1.316521,-1.588266,0.654342,-2.106966]],[[-0.218121,7.033268,-1.166442,-4.807288,-9.172159,-4.795837,-7.893059,0.869642,-2.996189],[2.421799,-6.774541,4.207303,9.394316,-1.156750,9.183094,6.781079,7.052661,6.212307],[8.418136,-2.721782,9.586417,-1.710817,9.555629,-9.300631,-9.435299,5.608852,-4.435415],[-1.311332,-3.594776,-9.082819,8.929568,8.135837,9.811006,-6.680317,0.526392,-9.196268],[-9.049882,-4.132107,-6.779674,0.768498,-0.457786,1.322466,6.369013,-7.859212,-6.671020],[4.666882,-4.159749,-5.366824,7.521903,1.924343,-1.141869,-7.514048,-3.038707,-2.330189],[2.882956,4.551792,-4.489720,-8.563636,-3.249169,-7.121955,5.990829,-3.909362,-9.191813],[1.583830,-3.963865,-8.863624,-4.223306,5.695785,8.157111,4.650742,6.025341,6.149837],[5.411510,4.787830,-8.949541,4.530818,8.543252,-5.412718,-0.901524,-6.684769,-4.501841],[3.734009,1.835372,6.841479,-9.928919,0.982142,3.103931,-9.839957,-3.931881,-2.212284],[9.990264,-5.881276,5.642399,-2.345353,9.748033,-5.549052,-0.511616,-2.193010,-6.410281],[-5.572517,2.334915,-0.180238,-3.973452,9.365071,-7.544562,-1.285776,6.481427,2.308199],[7.586443,4.840593,-9.331443,-7.042190,9.987633,-5.758920,-8.700113,6.958582,-5.532329],[-0.759063,-8.119251,2.213023,7.988817,-0.755229,-0.886767,3.493378,7.712937,-0.345108],[3.329562,-7.282344,-6.021617,-5.649521,-0.640174,-1.164251,-4.638316,0.060017,-3.789460],[9.035605,-9.437714,6.866556,1.894440,-2.652021,4.384712,-3.074017,7.898687,1.611752]],[[8.358937,-4.642774,5.082913,-7.497174,1.433675,-3.577839,-2.854975,0.054186,2.795171],[-5.896402,-7.913687,-1.473635,0.795382,-1.871247,2.105619,-4.405442,2.891033,-0.490494],[2.652794,-8.112209,0.052003,7.840298,0.285979,-0.745947,0.275803,7.085411,-5.593123],[4.355480,6.712450,-0.421015,3.723532,1.020760,-8.044966,-6.091002,7.912633,-2.667167],[7.714388,-8.536448,4.693931,-2.087617,8.468122,-7.940866,9.107618,-1.339942,2.016899],[5.463430,9.915411,5.922623,1.967801,-0.282899,4.579650,7.027212,-8.995406,8.023906],[-2.571608,-7.669685,3.984531,1.801642,-6.062747,3.911973,2.473259,-1.175221,-0.218935],[-5.051007,8.912056,-3.195542,1.962060,-9.411789,6.633346,7.584225,6.685340,-5.700270],[-6.326507,5.083849,8.795140,-4.041064,6.137530,6.427865,9.682950,1.935015,-1.742168],[7.703686,4.554774,6.633401,3.271900,6.095389,-1.405607,4.544726,3.626415,7.701643],[-8.685482,2.374257,7.582033,9.022962,-5.941783,5.807841,-5.453142,4.220656,-0.945946],[-1.344369,4.695624,-6.430089,4.067750,9.216747,-2.436776,6.294846,-9.417507,0.631277],[-6.234414,-0.460107,7.925152,3.033266,-9.191502,-1.460157,-1.100890,-1.744110,-3.074279],[6.327574,8.162268,-8.422095,-5.279569,-6.841325,2.213114,-5.510592,-9.770565,-0.440406],[-1.770222,7.062958,-9.495924,-8.029204,-6.096341,4.802097,2.208645,0.710369,2.800784],[2.069568,7.155740,-6.299204,7.915001,-3.991664,-7.361434,7.586442,7.719685,5.857033]],[[-5.630578,-2.631131,9.898708,0.577760,-5.697570,9.402480,8.451870,7.337652,-8.635928],[1.662470,-5.807073,1.935774,-4.009565,-4.462499,7.266264,-3.223600,8.785726,-8.294423],[-4.036360,1.166341,-6.191220,-9.311659,8.196835,0.403959,4.791316,9.764688,-0.021091],[-7.027434,-6.574090,-9.544368,4.225857,8.223644,2.815071,8.340992,-8.765014,4.888744],[3.731741,-4.577078,5.513418,-4.759406,-5.153598,6.281082,-9.857047,6.092028,-6.241036],[-8.764863,-1.505000,-0.935505,9.141322,3.766257,-3.457829,3.620623,-4.497040,9.710566],[6.564549,8.749869,9.580881,-4.445900,-7.134777,-2.085023,-9.107146,-1.901372,5.900749],[-3.831215,-3.784530,-9.812774,-4.212949,-7.582709,1.774891,4.519009,3.943870,-1.714891],[6.709592,-0.175295,-7.041615,-7.472763,1.724320,-3.802855,6.316561,-1.656877,-3.381273],[-1.699522,-2.782833,-1.377761,-3.823283,-6.386919,-9.795052,2.848427,5.703163,2.417986],[-4.790378,-2.406785,2.450150,-5.915188,-8.674026,-0.111292,1.547009,6.616016,-1.897590],[5.704048,6.859149,-3.731169,-8.813605,5.228807,-0.251541,-0.245818,-8.971723,-4.362089],[8.789006,-9.701862,8.257282,-1.072885,-7.737769,-0.128070,0.354790,3.071658,6.121955],[-2.610368,1.508103,-3.666129,1.267117,7.490864,6.083057,8.273532,-5.476702,-9.596827],[-7.976273,0.807826,4.803959,6.692771,8.195146,7.783370,2.424523,9.905384,-8.530439],[-6.225777,-2.701386,8.284458,9.773844,-9.402502,4.523151,-0.941893,7.454146,-9.134261]],[[5.028306,-2.893042,-4.448022,-0.380543,8.893190,9.752217,3.233655,3.807996,3.353705],[5.707920,7.692122,-5.742968,8.306541,6.763905,-5.593886,9.263266,3.107492,6.058206],[2.698302,2.480116,5.802438,-0.785227,-9.740974,-4.371311,2.277155,3.579451,3.612853],[5.073184,-0.646689,0.822151,-7.894433,-4.162735,-2.862579,9.576929,9.680437,1.601195],[4.030904,6.187580,7.909280,1.578272,-0.647921,-2.437752,4.855332,-4.999463,-6.085194],[-6.723571,-1.070560,-9.510597,-8.922088,-5.911129,7.151214,-3.629520,-0.852130,-1.826212],[4.675971,-9.861941,-8.974364,5.828571,-5.519712,1.306104,-5.467088,5.455672,7.633362],[9.508811,5.791145,4.506975,7.791569,-3.175423,-0.040341,-7.061364,-7.539630,4.517393],[6.615349,-1.413520,-6.289036,-4.063718,-4.337478,8.830550,8.772548,7.676915,-4.477883],[2.742148,1.974148,3.739447,8.919794,2.669326,-3.081687,-2.502569,-6.909837,6.575758],[8.158420,3.911894,5.438689,1.245264,9.626601,-4.616313,8.536210,-5.624910,2.809584],[-7.364833,5.287459,-3.844618,-3.855594,4.847822,-1.634076,3.107481,-0.455440,6.468958],[8.715965,-0.559305,6.610209,3.641820,7.762798,3.072780,8.147652,4.903834,-9.488258],[5.023341,-3.862838,5.636259,5.035110,-4.788987,1.572461,-1.821464,-4.819303,2.783719],[3.249879,0.722668,-6.704005,7.613878,-2.620126,-5.312558,-1.736722,-3.619863,-9.660238],[8.216615,-8.432858,6.041178,8.211334,7.998060,0.147509,-1.471586,7.916259,7.291800]],[[-2.758537,6.690028,8.484184,6.837229,4.058590,-6.571662,9.640538,-1.732639,-8.942051],[2.082471,9.575737,9.384028,-9.374292,4.460578,3.805613,6.537157,2.178827,8.187410],[7.015794,-0.620064,-9.217301,-2.580269,1.830122,9.422910,6.303747,2.236846,9.743446],[0.618076,-9.200727,9.645138,1.491599,-8.461430,-7.571223,2.977461,-0.708797,-9.963551],[-1.860880,9.598582,3.908487,2.379946,8.312219,6.698650,-1.776275,-8.750020,6.012403],[-1.157925,0.511060,5.050294,-3.544245,6.425941,6.865063,0.704618,-0.048502,6.162419],[-2.260147,-0.710145,5.851426,-6.232188,-8.709415,3.932439,1.649591,-1.771902,-5.065984],[-3.475357,-2.594788,1.040081,1.075090,-0.285018,2.748916,-7.155032,-3.539239,-4.253241],[-5.758572,4.147423,3.943653,8.985149,4.905891,0.403095,3.503332,-0.575076,7.094151],[1.833476,9.493144,2.045184,0.751495,-9.511870,-7.939127,-9.763086,5.651288,-5.034908],[-2.690765,5.275153,6.249280,0.322047,-6.205046,7.770055,8.327997,8.675740,-2.335200],[-2.490443,-5.839314,3.730196,3.086804,5.702066,-4.787093,6.040425,7.284920,-2.977649],[-7.981593,-8.920863,2.914568,-5.642041,-3.126601,7.287015,-2.462165,1.762614,-0.334483],[-8.940365,-5.785262,8.768694,4.529229,7.059837,5.178081,3.105997,8.835642,6.369242],[4.339799,2.107826,6.297103,-3.973687,-9.631496,-2.321880,2.139345,-3.867796,-6.893887],[-5.993411,8.340108,5.660084,0.605480,-2.154656,-2.642544,3.636502,7.705595,6.690018]],[[-4.507379,2.784517,7.055369,1.429545,-7.220570,-9.837279,7.782129,-0.211588,-2.090166],[-7.739413,-0.935316,-0.328662,-9.198646,6.331595,8.108883,-1.783460,-5.187737,-1.035015],[5.445989,-5.739064,-7.875337,-0.600245,1.688053,-8.389374,-7.898321,2.399811,9.639116],[4.483324,-7.673090,6.075658,-4.064876,3.922385,3.679130,-1.493058,7.629022,-8.589907],[-1.875963,-6.453528,4.875711,8.979400,8.690188,6.388926,6.737755,-4.404799,4.191835],[7.851183,2.119753,-0.211937,-2.143054,2.958803,-9.371311,6.708093,-8.681203,-7.553344],[-4.954917,-8.165759,6.861445,-1.324298,-7.378716,-7.308203,1.938085,2.004807,-2.902653],[0.230225,2.659091,-1.470998,0.503333,-9.910725,3.547505,1.728537,-7.037331,7.404850],[0.088474,-4.090734,-4.879450,2.539935,-5.059055,-5.203596,1.792144,-0.874752,-8.239178],[-7.318759,-1.692373,-0.570178,1.372812,-5.560825,5.930951,0.665458,3.705065,-2.360171],[-3.740500,4.428813,5.290282,9.368399,-3.559683,2.883496,5.356421,-4.345683,-8.207987],[7.643116,-8.281074,9.520607,-4.099180,4.902813,1.173281,1.555896,8.783133,-6.365984],[0.018226,5.917167,3.141094,-4.355170,-1.312911,6.126560,8.640404,6.766059,-1.658728],[7.526670,7.875758,-4.397929,8.728536,9.295523,6.253969,2.975451,9.693984,-6.995574],[5.405024,7.962137,1.419418,-8.021082,-1.796593,-8.870725,-4.660903,2.758875,-5.981266],[-5.405008,9.045533,8.017332,-0.974491,1.283134,0.031747,-2.040046,9.027093,9.112825]],[[-5.166230,9.462850,8.859961,2.091995,-4.883510,8.846995,8.522377,-3.128282,0.892642],[9.689295,-0.880912,-9.564241,-4.819035,-3.596530,-1.136614,-8.106681,6.079818,9.179953],[3.992414,-3.074809,9.663084,0.069624,-8.553125,-3.574270,5.059269,8.403961,-4.298191],[4.425648,-5.348820,-2.995584,8.018459,-8.263703,3.490292,2.083638,-5.923857,-8.810011],[-6.240958,-0.757467,3.479977,-1.737891,-1.947108,-1.428758,1.098687,0.885643,4.241937],[6.201993,-1.563484,1.927383,9.321883,2.070849,-8.811289,-9.794627,1.488665,0.448800],[4.635954,7.238243,1.756480,-0.192829,8.897054,7.204344,-3.278987,-4.947815,-8.979874],[-4.311705,-0.424136,8.031893,0.025472,9.587279,5.518186,2.901377,6.035001,-1.765431],[-4.089790,4.695384,6.215742,-4.847808,3.875156,-6.042081,-5.965033,-7.376040,-9.259443],[9.631814,8.022114,-2.797496,-4.045506,9.618131,-7.712594,-8.779283,-5.119510,5.073221],[-7.033056,7.577380,-2.163944,8.801199,-4.296201,5.832982,-6.081659,-6.900706,-6.020227],[-1.021176,-9.772853,-6.846326,0.452507,2.594367,-9.223745,0.045977,-4.319766,0.882913],[7.049445,0.371429,-7.525934,-6.740783,-3.874324,-7.907939,-0.971915,0.611057,-7.343927],[6.563035,-4.056220,2.234192,8.747908,-3.178101,-4.346653,-1.395468,-9.031466,3.472147],[3.432562,3.777654,2.356201,-0.291057,-3.936506,-0.117319,6.414788,5.485591,0.638486],[-9.087116,2.060331,-0.293886,-8.076409,8.062845,4.469949,8.872686,7.748009,-2.622443]],[[-1.784768,-8.660504,-0.291950,0.256632,-9.723691,1.522301,-3.459274,2.568474,-4.321319],[6.071338,-9.515051,-3.531997,1.261311,-0.200378,-7.017026,1.084492,8.162199,-1.443535],[1.324384,3.284363,-2.943707,6.560982,0.293324,-4.849321,-2.637863,-1.331709,4.984415],[-8.438610,6.952088,9.903983,6.453243,-0.760451,-7.580506,-7.249533,0.001108,5.066894],[7.586941,-8.524268,0.044179,-2.162733,-1.787440,1.805277,7.108781,-0.615578,-7.421903],[3.744553,4.608788,7.753858,-7.906689,0.119936,2.097817,7.795769,0.761360,5.416090],[-9.158757,-8.861035,5.413352,-2.895854,-6.235755,5.248668,-6.112735,-9.477508,-5.930939],[4.625457,-0.574719,-9.015687,-0.008844,-2.918496,4.913928,-2.970899,8.116098,-5.350785],[-9.211394,-0.488807,-9.209193,-7.942893,-1.983768,-1.014789,-6.239532,6.894077,-5.168964],[-6.764612,7.946160,-9.675239,1.590445,-3.180137,6.829547,0.618766,-3.310689,-9.101065],[7.772020,0.622643,-1.007528,-0.502588,-3.860716,2.099006,-3.471147,5.297886,-0.108241],[-9.545234,-2.410773,-7.236584,-3.431292,-6.264948,3.652525,0.355571,8.108377,-7.840393],[-1.584367,-2.299404,-6.615728,2.603927,5.911570,-2.482440,-4.289078,4.571746,-7.172427],[-6.796116,6.077464,6.115307,-8.786980,9.743173,8.551410,1.070285,3.024091,-0.425271],[7.837161,-1.463094,-9.692975,-9.622759,5.704801,-0.035264,5.522613,-9.599511,4.625751],[-3.514452,-4.961678,7.166446,9.859569,6.984395,-7.160020,8.561762,7.617384,-6.684102]],[[-9.409482,-2.132249,5.825199,0.641227,-8.914824,0.913688,1.372383,9.594114,-1.279363],[-8.474485,0.438568,-7.789540,4.780204,0.898341,7.421106,8.968172,-1.446433,2.864596],[-1.914812,-9.573059,0.037130,-2.196619,4.954996,-7.985959,-3.609441,-6.304885,9.415910],[1.090668,-5.986860,7.870486,-5.183526,5.998035,6.758732,-2.767275,-2.184911,-5.313154],[2.517453,-7.701858,-7.261565,-4.398189,4.657770,-9.646453,5.351794,-0.614235,-7.060140],[7.778373,-8.508964,9.239608,0.442506,8.439382,-3.813294,8.074614,-3.806056,9.199665],[6.722819,9.082495,-4.055214,6.508805,3.519600,6.250684,8.945827,-0.117107,4.769411],[3.185941,-1.968260,-7.466614,-4.900734,-2.147656,-1.794140,5.989196,7.146293,-9.574046],[-6.889662,4.640101,5.153029,5.974378,1.931823,-2.880412,6.435846,7.601594,-0.369783],[7.413246,-4.380517,3.825504,9.530659,-1.456092,9.108626,-0.488798,-7.483210,6.018093],[-7.295982,-9.318676,6.758001,-1.406573,8.011870,-2.010766,-6.040009,-6.560272,5.219958],[3.747822,9.316564,5.330971,-7.076538,-0.485606,5.477433,7.528726,1.435739,4.333942],[-5.041479,9.731670,5.468394,-1.580504,1.207726,-4.337538,-9.147751,7.739239,3.828419],[2.359575,-2.868124,-2.524826,9.340030,0.628295,-6.663982,9.419745,6.267919,-8.651999],[-2.702924,-3.873142,6.826233,-3.621933,-6.020348,0.821921,-3.115372,0.486191,8.919355],[-6.515962,8.951023,6.421166,-2.054804,6.517698,2.802448,7.455443,-2.558780,-4.403056]],[[-7.012247,5.826855,-9.412915,6.780452,3.068258,-1.904948,-1.305544,-4.273602,-3.606821],[9.501235,-7.715033,-2.890857,-5.874791,-7.850393,5.809035,1.114183,4.412804,-5.796506],[2.911205,0.165484,0.442542,8.457136,-6.644305,5.884422,8.348584,-1.392332,4.210632],[-2.224381,-9.837807,-4.954204,7.659238,4.531884,4.341783,-6.808775,1.214554,2.865685],[9.639792,1.201251,-5.245829,2.015754,7.278254,-5.389311,-3.491184,3.876247,1.502778],[1.702220,8.996601,9.806099,-6.837505,1.354814,6.215851,-8.871818,-3.607250,8.720703],[2.130893,-7.889148,-3.809699,7.148337,6.491042,7.111499,8.179042,0.313416,9.321595],[7.927113,7.917214,-6.891888,2.790973,9.658753,-2.853403,4.020273,8.121820,-6.712113],[-1.471933,6.285292,-5.698758,-5.660330,3.468994,-7.343202,-7.046517,-4.976071,8.192833],[3.849730,-1.452109,-8.607485,7.210117,-0.252818,7.157519,-4.016968,-4.451585,-2.592377],[-0.046543,-6.182985,-0.096276,-6.674458,-2.102733,-9.867857,0.802590,1.683062,8.884182],[-1.207783,4.543561,1.812291,-1.640116,-3.412344,5.898428,7.915848,4.968376,5.826018],[5.312187,-5.687962,2.395346,3.284844,-2.776095,-3.284438,-2.708554,-5.800878,5.898761],[-6.299192,8.249496,-9.385752,0.855469,-7.792714,-9.634105,6.330895,-1.972572,9.589647],[-4.207653,-8.744951,2.392493,-0.968504,-3.867431,7.301147,3.576094,7.635445,-2.963307],[1.759040,0.599997,-7.626845,-9.286455,0.320583,-5.293362,-4.632106,0.347097,2.269689]],[[3.333782,-0.941852,9.835755,4.580468,-8.420717,0.520092,-1.472996,-0.128954,1.552893],[9.457901,-9.702572,-2.304732,-2.519516,-2.978613,0.012489,-6.563813,4.516609,-5.278882],[5.953148,0.963528,-7.559304,-1.408664,-8.082693,4.207915,-8.355832,9.778490,3.297100],[8.205167,-5.987402,-7.178924,5.195182,-9.097468,0.676969,7.713607,-5.633242,-5.576590],[-1.548487,3.971540,-5.971369,-0.407858,-2.931137,0.285914,3.151124,8.376365,-3.229540],[3.922263,-4.248501,8.876920,7.299500,-2.259111,2.021676,-6.363925,-5.616990,-6.902641],[-0.725929,3.399897,-9.618742,7.285889,9.846570,5.283400,-5.700848,0.420018,-0.881358],[-1.346250,-4.041888,4.214634,-7.966878,8.943273,-4.054725,6.857318,-3.079319,0.844585],[3.754729,-6.384449,0.591133,9.089916,-3.233873,-7.215859,4.446256,-3.708094,-1.846576],[8.474182,-2.494781,1.073512,4.208310,5.136120,4.209333,-5.554247,6.998686,4.886955],[8.572115,-5.274505,3.763934,-0.318676,-1.830695,-3.008437,6.401096,3.775788,-1.714730],[-3.291353,4.405151,4.779893,-5.355996,8.627676,-4.990523,-5.438702,9.015313,-1.728015],[2.892607,1.127015,6.594691,-4.374315,-0.956158,-0.520609,-1.029729,-7.901438,2.510680],[-9.725393,0.428382,7.691485,-3.111089,5.497219,1.092009,7.456232,2.768694,-6.311216],[-4.738209,8.742772,-0.110028,-2.496245,-7.756671,-3.188840,-3.453671,2.233154,-7.281730],[-1.519975,-7.953007,2.776706,7.387170,-1.575868,3.884290,5.693457,-0.709274,-6.255362]]], dtype = "float64")#candidate|2639|(12, 16, 9)|const|float64
bop_2640 = relay.greater_equal(uop_2636.astype('bool'), relay.reshape(const_2639.astype('bool'), relay.shape_of(uop_2636))) # shape=(12, 16, 9)
bop_2643 = relay.greater_equal(uop_2638.astype('bool'), relay.reshape(const_2639.astype('bool'), relay.shape_of(uop_2638))) # shape=(12, 16, 9)
bop_2666 = relay.floor_mod(bop_2640.astype('float32'), relay.reshape(uop_2622.astype('float32'), relay.shape_of(bop_2640))) # shape=(12, 16, 9)
bop_2669 = relay.floor_mod(bop_2643.astype('float32'), relay.reshape(uop_2624.astype('float32'), relay.shape_of(bop_2643))) # shape=(12, 16, 9)
output = relay.Tuple([bop_2666,])
output2 = relay.Tuple([bop_2669,])
func_2672 = relay.Function([], output)
mod['func_2672'] = func_2672
mod = relay.transform.InferType()(mod)
output = func_2672()
func_2673 = relay.Function([], output)
mutated_mod['func_2673'] = func_2673
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2676 = relay.var("var_2676", dtype = "float32", shape = (5, 4, 11))#candidate|2676|(5, 4, 11)|var|float32
uop_2677 = relay.cos(var_2676.astype('float32')) # shape=(5, 4, 11)
output = uop_2677
output2 = uop_2677
func_2682 = relay.Function([var_2676,], output)
mod['func_2682'] = func_2682
mod = relay.transform.InferType()(mod)
var_2683 = relay.var("var_2683", dtype = "float32", shape = (5, 4, 11))#candidate|2683|(5, 4, 11)|var|float32
output = func_2682(var_2683)
func_2684 = relay.Function([var_2683], output)
mutated_mod['func_2684'] = func_2684
mutated_mod = relay.transform.InferType()(mutated_mod)
func_907_call = mod.get_global_var('func_907')
func_909_call = mutated_mod.get_global_var('func_909')
call_2708 = func_907_call()
call_2709 = func_907_call()
var_2710 = relay.var("var_2710", dtype = "int8", shape = (3, 16, 5))#candidate|2710|(3, 16, 5)|var|int8
bop_2711 = relay.less(call_2708.astype('bool'), var_2710.astype('bool')) # shape=(3, 16, 5)
bop_2714 = relay.less(call_2709.astype('bool'), var_2710.astype('bool')) # shape=(3, 16, 5)
var_2717 = relay.var("var_2717", dtype = "int8", shape = (3, 16, 5))#candidate|2717|(3, 16, 5)|var|int8
bop_2718 = relay.less_equal(var_2710.astype('bool'), relay.reshape(var_2717.astype('bool'), relay.shape_of(var_2710))) # shape=(3, 16, 5)
func_478_call = mod.get_global_var('func_478')
func_481_call = mutated_mod.get_global_var('func_481')
var_2725 = relay.var("var_2725", dtype = "int8", shape = (1792,))#candidate|2725|(1792,)|var|int8
call_2724 = relay.TupleGetItem(func_478_call(relay.reshape(var_2725.astype('int8'), [14, 16, 8])), 1)
call_2726 = relay.TupleGetItem(func_481_call(relay.reshape(var_2725.astype('int8'), [14, 16, 8])), 1)
func_1295_call = mod.get_global_var('func_1295')
func_1296_call = mutated_mod.get_global_var('func_1296')
call_2728 = func_1295_call()
call_2729 = func_1295_call()
output = relay.Tuple([bop_2711,bop_2718,call_2724,var_2725,call_2728,])
output2 = relay.Tuple([bop_2714,bop_2718,call_2726,var_2725,call_2729,])
func_2734 = relay.Function([var_2710,var_2717,var_2725,], output)
mod['func_2734'] = func_2734
mod = relay.transform.InferType()(mod)
var_2735 = relay.var("var_2735", dtype = "int8", shape = (3, 16, 5))#candidate|2735|(3, 16, 5)|var|int8
var_2736 = relay.var("var_2736", dtype = "int8", shape = (3, 16, 5))#candidate|2736|(3, 16, 5)|var|int8
var_2737 = relay.var("var_2737", dtype = "int8", shape = (1792,))#candidate|2737|(1792,)|var|int8
output = func_2734(var_2735,var_2736,var_2737,)
func_2738 = relay.Function([var_2735,var_2736,var_2737,], output)
mutated_mod['func_2738'] = func_2738
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1829_call = mod.get_global_var('func_1829')
func_1831_call = mutated_mod.get_global_var('func_1831')
call_2758 = relay.TupleGetItem(func_1829_call(), 0)
call_2759 = relay.TupleGetItem(func_1831_call(), 0)
output = call_2758
output2 = call_2759
func_2777 = relay.Function([], output)
mod['func_2777'] = func_2777
mod = relay.transform.InferType()(mod)
output = func_2777()
func_2778 = relay.Function([], output)
mutated_mod['func_2778'] = func_2778
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2817 = relay.var("var_2817", dtype = "float32", shape = (3, 2))#candidate|2817|(3, 2)|var|float32
uop_2818 = relay.log2(var_2817.astype('float32')) # shape=(3, 2)
output = uop_2818
output2 = uop_2818
func_2820 = relay.Function([var_2817,], output)
mod['func_2820'] = func_2820
mod = relay.transform.InferType()(mod)
var_2821 = relay.var("var_2821", dtype = "float32", shape = (3, 2))#candidate|2821|(3, 2)|var|float32
output = func_2820(var_2821)
func_2822 = relay.Function([var_2821], output)
mutated_mod['func_2822'] = func_2822
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2830 = relay.var("var_2830", dtype = "float32", shape = (16, 5, 11))#candidate|2830|(16, 5, 11)|var|float32
uop_2831 = relay.log10(var_2830.astype('float32')) # shape=(16, 5, 11)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_2833 = relay.TupleGetItem(func_408_call(), 0)
call_2834 = relay.TupleGetItem(func_409_call(), 0)
uop_2842 = relay.asinh(var_2830.astype('float64')) # shape=(16, 5, 11)
func_1829_call = mod.get_global_var('func_1829')
func_1831_call = mutated_mod.get_global_var('func_1831')
call_2849 = relay.TupleGetItem(func_1829_call(), 0)
call_2850 = relay.TupleGetItem(func_1831_call(), 0)
output = relay.Tuple([uop_2831,call_2833,uop_2842,call_2849,])
output2 = relay.Tuple([uop_2831,call_2834,uop_2842,call_2850,])
func_2854 = relay.Function([var_2830,], output)
mod['func_2854'] = func_2854
mod = relay.transform.InferType()(mod)
var_2855 = relay.var("var_2855", dtype = "float32", shape = (16, 5, 11))#candidate|2855|(16, 5, 11)|var|float32
output = func_2854(var_2855)
func_2856 = relay.Function([var_2855], output)
mutated_mod['func_2856'] = func_2856
mutated_mod = relay.transform.InferType()(mutated_mod)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_2876 = relay.TupleGetItem(func_408_call(), 0)
call_2877 = relay.TupleGetItem(func_409_call(), 0)
var_2889 = relay.var("var_2889", dtype = "int8", shape = (11, 16, 16))#candidate|2889|(11, 16, 16)|var|int8
bop_2890 = relay.less_equal(call_2876.astype('bool'), var_2889.astype('bool')) # shape=(11, 16, 16)
bop_2893 = relay.less_equal(call_2877.astype('bool'), var_2889.astype('bool')) # shape=(11, 16, 16)
bop_2894 = relay.greater_equal(var_2889.astype('bool'), call_2876.astype('bool')) # shape=(11, 16, 16)
bop_2897 = relay.greater_equal(var_2889.astype('bool'), call_2877.astype('bool')) # shape=(11, 16, 16)
output = relay.Tuple([bop_2890,bop_2894,])
output2 = relay.Tuple([bop_2893,bop_2897,])
func_2898 = relay.Function([var_2889,], output)
mod['func_2898'] = func_2898
mod = relay.transform.InferType()(mod)
var_2899 = relay.var("var_2899", dtype = "int8", shape = (11, 16, 16))#candidate|2899|(11, 16, 16)|var|int8
output = func_2898(var_2899)
func_2900 = relay.Function([var_2899], output)
mutated_mod['func_2900'] = func_2900
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2907 = relay.const([[7,5,1],[-3,-5,8],[8,4,-2],[6,-5,-1],[-7,9,-5],[2,-8,-4]], dtype = "int32")#candidate|2907|(6, 3)|const|int32
var_2908 = relay.var("var_2908", dtype = "int32", shape = (6, 3))#candidate|2908|(6, 3)|var|int32
bop_2909 = relay.maximum(const_2907.astype('int32'), relay.reshape(var_2908.astype('int32'), relay.shape_of(const_2907))) # shape=(6, 3)
func_592_call = mod.get_global_var('func_592')
func_596_call = mutated_mod.get_global_var('func_596')
var_2923 = relay.var("var_2923", dtype = "float64", shape = (660,))#candidate|2923|(660,)|var|float64
const_2924 = relay.const([[-6,-4],[-3,-9],[-3,-10],[-1,-9],[2,-10],[7,-10],[-9,7],[3,-10],[2,9],[7,6],[-6,-10],[-5,9],[-3,2],[8,-2],[1,2],[-3,-4],[-6,-2],[-8,4],[5,2],[-3,9],[-10,7],[3,7],[-1,7],[-1,10],[-3,-10],[-3,6],[10,-9],[-10,8],[9,10],[-9,5],[7,-3],[-4,-8],[7,5],[-6,-7],[-10,-7],[5,10],[5,6],[-8,3],[5,-8],[10,4],[-5,5],[-2,6],[-7,2],[-2,-5],[-4,-10],[-4,-10],[8,-9],[-3,4],[-4,7],[-1,9],[8,9],[-4,7],[-7,-10],[8,1],[-7,4],[5,-6],[-7,6],[6,6],[-7,-1],[1,-10],[-5,-1],[-10,8],[-5,7],[-2,-9],[-8,9],[-2,-7],[9,-3],[6,-9],[2,3],[-2,-1],[-7,7],[5,-10],[-2,-4],[-1,-7],[-9,-1],[2,7],[9,9],[-4,6],[-1,1],[-8,-7],[-10,-7],[4,-3],[4,1],[3,-1],[-9,-4],[1,6],[-3,-9],[10,3],[9,-10],[6,7],[7,-6],[-7,3],[-5,10],[-6,-7],[-3,-2],[-2,-10],[-7,-7],[4,-2],[3,1],[3,5],[10,-4],[-2,-7],[-9,10],[6,-6],[2,7],[-7,-7],[7,10],[4,9],[-3,-4],[8,-1],[-2,10],[-1,-10],[-10,-9],[8,-9],[7,-6],[-5,7],[-10,2],[-2,-6],[-4,-2],[-5,-9],[1,-3],[-5,-3],[-2,-5],[4,6],[3,-1],[-2,1],[3,2],[6,-2],[9,-8],[-10,9],[8,2],[6,-5],[5,-3],[-2,3],[-8,3],[6,4],[3,-10],[-3,-8],[2,4],[-5,6],[4,8],[-4,9],[-8,7],[-8,-7],[-3,5],[-7,6],[2,-5],[8,8],[1,-5],[-9,-10],[-9,-8],[9,7],[6,7],[-6,1],[2,4],[9,1],[-9,6],[5,2],[-2,8],[-10,3],[-10,9],[3,5],[-8,-7],[-2,-7],[-9,7],[-1,-5],[-3,-4],[1,-9],[10,-8],[-10,10],[1,-6],[-10,-3],[10,-2],[-3,1],[3,9],[7,-4],[7,-9],[4,-6],[-1,-7],[7,-1],[-2,-5],[-1,10],[-6,-7],[-8,7],[4,2],[-9,2],[-1,10],[-2,7],[5,10],[-10,7],[9,8],[-6,3],[-3,7],[-3,2],[9,6],[-5,7],[2,-4],[-6,-10],[-7,-2],[-3,3],[10,-8],[9,9],[-1,9],[-5,-6],[4,1],[10,-6],[-10,-2],[-3,-1],[-2,-6],[-8,-9],[9,7],[-6,2],[2,9],[5,-1],[-4,-9],[7,-10],[8,2],[-10,-9],[10,-2],[8,6],[7,3],[1,-8],[8,-1],[-7,8],[6,3],[10,4],[5,-8],[-8,-4],[10,-9],[-1,5],[6,6],[9,-9],[3,-1],[-4,-5],[-1,-4],[7,-5],[5,-7],[3,-8],[10,-2],[-6,-5],[5,-4],[-3,7],[-9,-2],[-1,3],[-1,2],[4,2],[-4,6],[7,8],[7,8],[10,3],[-8,-8],[-2,-3],[-3,4],[7,-3],[10,10],[1,9],[10,8],[-5,-1],[-6,2],[-2,-8],[1,2],[9,8],[5,9],[6,3],[-7,7],[-2,5],[-9,-3],[-1,4],[4,-10],[1,4],[-6,1],[-3,7],[5,-2],[-4,2],[-5,2],[5,-9],[-2,-4],[9,-1],[3,4],[-6,5],[-8,-8],[-7,6],[5,-7],[-4,1],[-7,10],[-3,10],[3,-1],[6,-10]], dtype = "int8")#candidate|2924|(288, 2)|const|int8
call_2922 = relay.TupleGetItem(func_592_call(relay.reshape(var_2923.astype('float64'), [4, 15, 11]), relay.reshape(const_2924.astype('int8'), [576,]), relay.reshape(var_2923.astype('float64'), [4, 15, 11]), ), 4)
call_2925 = relay.TupleGetItem(func_596_call(relay.reshape(var_2923.astype('float64'), [4, 15, 11]), relay.reshape(const_2924.astype('int8'), [576,]), relay.reshape(var_2923.astype('float64'), [4, 15, 11]), ), 4)
uop_2927 = relay.acosh(var_2923.astype('float64')) # shape=(660,)
output = relay.Tuple([bop_2909,call_2922,const_2924,uop_2927,])
output2 = relay.Tuple([bop_2909,call_2925,const_2924,uop_2927,])
func_2931 = relay.Function([var_2908,var_2923,], output)
mod['func_2931'] = func_2931
mod = relay.transform.InferType()(mod)
var_2932 = relay.var("var_2932", dtype = "int32", shape = (6, 3))#candidate|2932|(6, 3)|var|int32
var_2933 = relay.var("var_2933", dtype = "float64", shape = (660,))#candidate|2933|(660,)|var|float64
output = func_2931(var_2932,var_2933,)
func_2934 = relay.Function([var_2932,var_2933,], output)
mutated_mod['func_2934'] = func_2934
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2777_call = mod.get_global_var('func_2777')
func_2778_call = mutated_mod.get_global_var('func_2778')
call_2936 = func_2777_call()
call_2937 = func_2777_call()
output = relay.Tuple([call_2936,])
output2 = relay.Tuple([call_2937,])
func_2938 = relay.Function([], output)
mod['func_2938'] = func_2938
mod = relay.transform.InferType()(mod)
output = func_2938()
func_2939 = relay.Function([], output)
mutated_mod['func_2939'] = func_2939
mutated_mod = relay.transform.InferType()(mutated_mod)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_2995 = func_254_call()
call_2996 = func_254_call()
output = relay.Tuple([call_2995,])
output2 = relay.Tuple([call_2996,])
func_3001 = relay.Function([], output)
mod['func_3001'] = func_3001
mod = relay.transform.InferType()(mod)
output = func_3001()
func_3002 = relay.Function([], output)
mutated_mod['func_3002'] = func_3002
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3037 = relay.var("var_3037", dtype = "uint8", shape = (6, 1, 5))#candidate|3037|(6, 1, 5)|var|uint8
var_3038 = relay.var("var_3038", dtype = "uint8", shape = (6, 14, 5))#candidate|3038|(6, 14, 5)|var|uint8
bop_3039 = relay.bitwise_or(var_3037.astype('uint8'), var_3038.astype('uint8')) # shape=(6, 14, 5)
func_1949_call = mod.get_global_var('func_1949')
func_1953_call = mutated_mod.get_global_var('func_1953')
const_3048 = relay.const([2,-2,3,5,3,-9,-2,8,-6,-6,6,-2,-2,-7,7,5,-1,-6,-9,6,-2,4,7,4,3,-8,2,3,-10,6,-4,9,-8,-2,6,8,2,6,3,5,-1,-6,3,-8,2,10,-5,-1,-6,1,-10,-3,-10,-4,-8,3,-3,4,-10,9,-8,-6,-10,7,4,9,6,8,1,-1,4,10,-4,-3,7,-4,3,-8,-1,-10,-1,-8,-8,-3,10,-10,9,-10,-4,-5,7,6,1,4,8,-6,-10,8,-1,-9,-6,-3,-8,-7,-8,-3,-4,-6,4,-9,-3,-1,-9,2,-1,-9,1,1,-7,-4,-3,-3,-8,1,10,-2,5,5,-6,2,-2,-4,4,-4,6,-9,5,-5,2,4,-8,-4,-4,6,8,5,10,-4,-5,-9,-8,-7,4,-7,9,-8,-8,1,1,-1,-7,2,-6,-6,-7,-1,5,10,-1,6,6,9,8,7,-7,-10], dtype = "int8")#candidate|3048|(176,)|const|int8
const_3049 = relay.const([-2,9,1,2,2,9,9,-7,-9,10,-9,8,-8,4,-7,-2,-7,6,-10,-7,8,-2,-5,-1,-4,-4,1,-10,-7,1,-3,-5,-6,8,9,-8,-3,-3,1,-5,-9,9,1,1,6,-3,4,7,1,8,3,9,6,4,-8,-4,10,-6,-7,-10,-6,-9,-3,2,4,-3,6,2,-4,1,7,-8,3,9,-1,-10,-2,-9,5,-8,-3,-5,3,-3,6,-10,1,4,-1,3,-2,9,4,-10,1,-10,-8,-3,4,9,-2,-7,-6,7,3,-7,5,-8,4,-7,-7,9,-2,8,10,8,-6,-7,5,8,-5,3,-3,-8,1,3,-2,3,7,-2,9,-1,6,-9,-1,10,-4,-8,6,-5,3,6,10,-1,-1,4,-4,8,2,-4,-4,2,9,-6,-6,-3,9,-6,9,6,10,-1,-6,8,-4,-4,9,8,-6,5,5,2,-7,1,-4,7,6,2,-7,-1,-5,1,8,3,4,-5,6,-1,4,1,6,3,8,-3,8,-7,-8,6,-8,10,-5,-8,-2,7,10,5,-8,9,6,10,-7,-6,5,-9,-10,-2,-5,9,-9,7,-4,5,-3,-9,-6,4,8,-6,-3,-6,7,-10,-8,2,-5,9,1,5,-8,-4,5,-8,9,6,3,7,-9,2,7,-7,1,4,-9,8,3,-2,2,-1,-5,5,6,8,10,-5,2,-1,-9,6,2,-5,-6,3,-6,8,-1,9,-8,3,7,-7,1,3,-3,-5,-3,-1,-7,6,1,5,9,-9,3,7,1,10,-3,1,-3,10,9,6,8,-9,6,-8,2,-3,1,-6,5,-4,2,-4,-4,-3,-8,8,9,-6,3,-1,3,-9,-6,-7,-2,3,8,9,4,6,-10,-9,-4,-5,8,2,-2,6,-6,10,5,2,-1,-9,4,8,-10,5,3,-2,6,4,5,-5,-2,10,8,-5,6,5,-7,-4,-1,4,-6,5,-8,-9,9,-5,9,-2,-9,8,-9,5,-8,-7,-7,1,5,7,-6,-8,2,-8,9,10,5,3,-2,1,9,-9,7,2,-7,-7,7,8,9,2,5,5,-1,3,-9,5,2,-4,3,3,5,5,-5,3,8,-1,-7,-3,7,6,-10,-5,7,1,5,2,-4,-4,-9,-7,2,9,6,-10,8,10,10,-2,5,9,-7,-10,7,4,-2,-2,-8,-4,3,-7,3,-10,-3,10,-6,-7,-9,6,2,-3,-10,1,-7,2,10,-4,-9,7,-7,-9,3,7,-5,-9,4,3,8,4,8,7,7,4,6,-7,8,7,4,-5,-9,-9,-6,-9,-8,-10,-2,6,-6,5,-2,6,-2,3,-7,8,3,-8,-10,1,-3,4,-8,5,3,-8,4,-6,7,4,-8,-9,-10,7,-8,-7,1,-4,-4,-5,-9,-3,-8,-3,10,8,-2,6,4,1,6,-2,-9,-8,-5,9,7,-1,3,-10,3,-1,-5,-8,-4,2,7,7,-10,9,7,3,10,-10,10,-7,8,-10,7,-3,-7,1,8,-7,2,-8,-4,4,7,-8,-6,1,-5,8,-1,4,-7,-1,2,7,-6,2,6,-3,-2,1,7,1,-7,5,1,5,2,-4,3,-8,-6,8,-8,8,-3,-5,3,2,6,-7,5,-6,-7,3,4,9,-4,4,8,-10,-3,1,3,-7,9,8,5,2,4,3,1,-8,1,-6,5,9,-3,5,2,-4,5,6,6,5,-7,8,2,-2,-1,9,-6,-6,-9,-4,-5,3,-2,4,-8,10,9,5,7,2,6,5,-2,-1,-4,-3,7,10,6,9,-5,1,-10,-3,4,4,1,-1,10,5,9,4,-3,-7,-10,-4,7,9,2,2,-1,5,6,4,-10,-2,6,-6,2,-9,2,-9,9,-6,-4,-6,-7,-5,-8,10,10,6,-8,7,8,4,1,10,-3,-2,7,10,-10,-6,5,-6,-8,1,9,-5,5,-6,3,8,-2,-1,-10,-2,7,1,-6,5,1,7,3,6,1,2,-10,10,-1,-1,-3,5,1,-9,5,10,-9,6,10,10,6,7,-4,-7,9,10,-3,9,-1,2,4,1,-8,7,-1,-3,-1,-4,-7,-10,1,6,4,-9,-6,10,10,3,-10,-9,-7,9,-3,3,-4,-6,-2,8,-2,1,-5,-7,-4,-6,-3,5,-7,10,-5,-10,5,-3,-10,4,-9,9,-6,10,7,4,3,1,-4,-8,3,10,-3,6,-8,-7,-5,-1,-1,3,-9,1,8,1,-3,5,-8,6,9,-7,4,4,-10,-2,-7,10,-4,9,-3,6,-9,3,-5,1,3,-3,-1,6,-3,4,-5,-6,-5,5,-5,-5,-1,2,-9,1,2,-9,-2,-6,-5,-5,-5,1,7,-8,-6,-7,-3,-4,1,-4,-3,7,10,-10,-2,-5,-7,-6,4,-5,1,1,10,-6,-3,5,5,-1,5,-10,-8,5,5,10,-9,2,-8,4,10,-10,-2,1,10,2,5,4,9,1,-3,5,-7,-10,10,2,3,5,10,5,-7,-5,1,8,-1,5,-9,3,-7,-10,8,9,5,-7,9,4,8,-10,1,10,6,-5,6,-2,3,-2,-3,2,-6,-4,-10,-10,-2,-9,8,-4,-7,-9,6,-7,-5,-10,1,-8,-1,-1,5,-1,-4,1,-2,-6,3,-3,9,10,9,2,-1,9,5,3,3,-8,-6,9,-5,8,-10,-8,1,3,-7,-1,5,10,-2,-3,-9,-4,-1,7,-7,3,8,-8,-9,-10,-6,-5,-3,-3,10,-1,-9,5,9,-4,-9,-6,8,9,-3,-4,-1,9,7,-2,10,-5,-10,8,6,-6,1,-3,3,6,10,6,10,9,9,-4,5,-9,4,2,1,5,9,-9,-10,8,-7,-3,-10,-6,-4,-10,-2,3,6,3,3,-9,6,5,-5,5,-7,-8,-3,-8,5,-5,8,-10,-3,-4,-1,-9,3,2,-6,1,-7,2,3,6,-9,6,1,4,9,10,-10,-6,10,-4,-4,10,4,-1,4,-7,-10,-10,9,5,-1,1,-2,8,-4,8,5,-1,-10,6,-2,8,-7,3,2,5,2,7,-2,10,5,-8,-8,9,-1,-3,-9,-5,9,2,7,-5,5,7,-6,5,-4,4,-2,-9,10,-8,-3,7,4,-3,9,1,1,1,-6,-3,-1,-2,1,-8,5,2,-10,2,-6,-6,-3,9,7,-5,8,3,4,6,1,6,1,-6,7,-6,2,-10,-9,1,10,-8,4,10,1,3,2,-9,-9,4], dtype = "int8")#candidate|3049|(1232,)|const|int8
var_3050 = relay.var("var_3050", dtype = "float32", shape = (540,))#candidate|3050|(540,)|var|float32
call_3047 = relay.TupleGetItem(func_1949_call(relay.reshape(const_3048.astype('int8'), [11, 16, 1]), relay.reshape(const_3049.astype('int8'), [1232,]), relay.reshape(var_3050.astype('float32'), [5, 12, 9]), ), 5)
call_3051 = relay.TupleGetItem(func_1953_call(relay.reshape(const_3048.astype('int8'), [11, 16, 1]), relay.reshape(const_3049.astype('int8'), [1232,]), relay.reshape(var_3050.astype('float32'), [5, 12, 9]), ), 5)
func_2820_call = mod.get_global_var('func_2820')
func_2822_call = mutated_mod.get_global_var('func_2822')
var_3054 = relay.var("var_3054", dtype = "float32", shape = (6, 1))#candidate|3054|(6, 1)|var|float32
call_3053 = func_2820_call(relay.reshape(var_3054.astype('float32'), [3, 2]))
call_3055 = func_2820_call(relay.reshape(var_3054.astype('float32'), [3, 2]))
output = relay.Tuple([bop_3039,call_3047,const_3048,const_3049,var_3050,call_3053,var_3054,])
output2 = relay.Tuple([bop_3039,call_3051,const_3048,const_3049,var_3050,call_3055,var_3054,])
func_3061 = relay.Function([var_3037,var_3038,var_3050,var_3054,], output)
mod['func_3061'] = func_3061
mod = relay.transform.InferType()(mod)
var_3062 = relay.var("var_3062", dtype = "uint8", shape = (6, 1, 5))#candidate|3062|(6, 1, 5)|var|uint8
var_3063 = relay.var("var_3063", dtype = "uint8", shape = (6, 14, 5))#candidate|3063|(6, 14, 5)|var|uint8
var_3064 = relay.var("var_3064", dtype = "float32", shape = (540,))#candidate|3064|(540,)|var|float32
var_3065 = relay.var("var_3065", dtype = "float32", shape = (6, 1))#candidate|3065|(6, 1)|var|float32
output = func_3061(var_3062,var_3063,var_3064,var_3065,)
func_3066 = relay.Function([var_3062,var_3063,var_3064,var_3065,], output)
mutated_mod['func_3066'] = func_3066
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3102 = relay.var("var_3102", dtype = "float32", shape = (7, 4, 15))#candidate|3102|(7, 4, 15)|var|float32
var_3103 = relay.var("var_3103", dtype = "float32", shape = (7, 4, 15))#candidate|3103|(7, 4, 15)|var|float32
bop_3104 = relay.floor_mod(var_3102.astype('float32'), relay.reshape(var_3103.astype('float32'), relay.shape_of(var_3102))) # shape=(7, 4, 15)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_3117 = func_254_call()
call_3118 = func_254_call()
uop_3121 = relay.sigmoid(bop_3104.astype('float64')) # shape=(7, 4, 15)
output = relay.Tuple([call_3117,uop_3121,])
output2 = relay.Tuple([call_3118,uop_3121,])
func_3126 = relay.Function([var_3102,var_3103,], output)
mod['func_3126'] = func_3126
mod = relay.transform.InferType()(mod)
mutated_mod['func_3126'] = func_3126
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3126_call = mutated_mod.get_global_var('func_3126')
var_3128 = relay.var("var_3128", dtype = "float32", shape = (7, 4, 15))#candidate|3128|(7, 4, 15)|var|float32
var_3129 = relay.var("var_3129", dtype = "float32", shape = (7, 4, 15))#candidate|3129|(7, 4, 15)|var|float32
call_3127 = func_3126_call(var_3128,var_3129,)
output = call_3127
func_3130 = relay.Function([var_3128,var_3129,], output)
mutated_mod['func_3130'] = func_3130
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1812_call = mod.get_global_var('func_1812')
func_1813_call = mutated_mod.get_global_var('func_1813')
call_3134 = relay.TupleGetItem(func_1812_call(), 0)
call_3135 = relay.TupleGetItem(func_1813_call(), 0)
output = relay.Tuple([call_3134,])
output2 = relay.Tuple([call_3135,])
func_3139 = relay.Function([], output)
mod['func_3139'] = func_3139
mod = relay.transform.InferType()(mod)
mutated_mod['func_3139'] = func_3139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3139_call = mutated_mod.get_global_var('func_3139')
call_3140 = func_3139_call()
output = call_3140
func_3141 = relay.Function([], output)
mutated_mod['func_3141'] = func_3141
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3191 = relay.var("var_3191", dtype = "float32", shape = (3, 5, 10))#candidate|3191|(3, 5, 10)|var|float32
var_3192 = relay.var("var_3192", dtype = "float32", shape = (3, 5, 10))#candidate|3192|(3, 5, 10)|var|float32
bop_3193 = relay.subtract(var_3191.astype('float32'), relay.reshape(var_3192.astype('float32'), relay.shape_of(var_3191))) # shape=(3, 5, 10)
bop_3203 = relay.mod(var_3191.astype('float64'), relay.reshape(bop_3193.astype('float64'), relay.shape_of(var_3191))) # shape=(3, 5, 10)
output = bop_3203
output2 = bop_3203
func_3213 = relay.Function([var_3191,var_3192,], output)
mod['func_3213'] = func_3213
mod = relay.transform.InferType()(mod)
var_3214 = relay.var("var_3214", dtype = "float32", shape = (3, 5, 10))#candidate|3214|(3, 5, 10)|var|float32
var_3215 = relay.var("var_3215", dtype = "float32", shape = (3, 5, 10))#candidate|3215|(3, 5, 10)|var|float32
output = func_3213(var_3214,var_3215,)
func_3216 = relay.Function([var_3214,var_3215,], output)
mutated_mod['func_3216'] = func_3216
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2280_call = mod.get_global_var('func_2280')
func_2281_call = mutated_mod.get_global_var('func_2281')
call_3233 = relay.TupleGetItem(func_2280_call(), 1)
call_3234 = relay.TupleGetItem(func_2281_call(), 1)
uop_3244 = relay.log(call_3233.astype('float32')) # shape=(1, 16, 1)
uop_3246 = relay.log(call_3234.astype('float32')) # shape=(1, 16, 1)
output = relay.Tuple([uop_3244,])
output2 = relay.Tuple([uop_3246,])
func_3248 = relay.Function([], output)
mod['func_3248'] = func_3248
mod = relay.transform.InferType()(mod)
output = func_3248()
func_3249 = relay.Function([], output)
mutated_mod['func_3249'] = func_3249
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3333 = relay.var("var_3333", dtype = "float32", shape = (7, 15, 8))#candidate|3333|(7, 15, 8)|var|float32
uop_3334 = relay.log(var_3333.astype('float32')) # shape=(7, 15, 8)
func_1812_call = mod.get_global_var('func_1812')
func_1813_call = mutated_mod.get_global_var('func_1813')
call_3337 = relay.TupleGetItem(func_1812_call(), 0)
call_3338 = relay.TupleGetItem(func_1813_call(), 0)
func_1829_call = mod.get_global_var('func_1829')
func_1831_call = mutated_mod.get_global_var('func_1831')
call_3341 = relay.TupleGetItem(func_1829_call(), 0)
call_3342 = relay.TupleGetItem(func_1831_call(), 0)
func_2280_call = mod.get_global_var('func_2280')
func_2281_call = mutated_mod.get_global_var('func_2281')
call_3356 = relay.TupleGetItem(func_2280_call(), 0)
call_3357 = relay.TupleGetItem(func_2281_call(), 0)
output = relay.Tuple([uop_3334,call_3337,call_3341,call_3356,])
output2 = relay.Tuple([uop_3334,call_3338,call_3342,call_3357,])
func_3371 = relay.Function([var_3333,], output)
mod['func_3371'] = func_3371
mod = relay.transform.InferType()(mod)
var_3372 = relay.var("var_3372", dtype = "float32", shape = (7, 15, 8))#candidate|3372|(7, 15, 8)|var|float32
output = func_3371(var_3372)
func_3373 = relay.Function([var_3372], output)
mutated_mod['func_3373'] = func_3373
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2328_call = mod.get_global_var('func_2328')
func_2330_call = mutated_mod.get_global_var('func_2330')
call_3447 = relay.TupleGetItem(func_2328_call(), 0)
call_3448 = relay.TupleGetItem(func_2330_call(), 0)
var_3452 = relay.var("var_3452", dtype = "int8", shape = (12, 16, 16))#candidate|3452|(12, 16, 16)|var|int8
bop_3453 = relay.less(call_3447.astype('bool'), var_3452.astype('bool')) # shape=(12, 16, 16)
bop_3456 = relay.less(call_3448.astype('bool'), var_3452.astype('bool')) # shape=(12, 16, 16)
uop_3458 = relay.log10(var_3452.astype('float64')) # shape=(12, 16, 16)
uop_3461 = relay.log2(uop_3458.astype('float64')) # shape=(12, 16, 16)
func_878_call = mod.get_global_var('func_878')
func_880_call = mutated_mod.get_global_var('func_880')
call_3467 = func_878_call()
call_3468 = func_878_call()
output = relay.Tuple([bop_3453,uop_3461,call_3467,])
output2 = relay.Tuple([bop_3456,uop_3461,call_3468,])
func_3473 = relay.Function([var_3452,], output)
mod['func_3473'] = func_3473
mod = relay.transform.InferType()(mod)
var_3474 = relay.var("var_3474", dtype = "int8", shape = (12, 16, 16))#candidate|3474|(12, 16, 16)|var|int8
output = func_3473(var_3474)
func_3475 = relay.Function([var_3474], output)
mutated_mod['func_3475'] = func_3475
mutated_mod = relay.transform.InferType()(mutated_mod)
func_861_call = mod.get_global_var('func_861')
func_863_call = mutated_mod.get_global_var('func_863')
call_3495 = func_861_call()
call_3496 = func_861_call()
output = call_3495
output2 = call_3496
func_3498 = relay.Function([], output)
mod['func_3498'] = func_3498
mod = relay.transform.InferType()(mod)
output = func_3498()
func_3499 = relay.Function([], output)
mutated_mod['func_3499'] = func_3499
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3546 = relay.var("var_3546", dtype = "float32", shape = ())#candidate|3546|()|var|float32
var_3547 = relay.var("var_3547", dtype = "float32", shape = (10, 1))#candidate|3547|(10, 1)|var|float32
bop_3548 = relay.mod(var_3546.astype('float32'), var_3547.astype('float32')) # shape=(10, 1)
func_254_call = mod.get_global_var('func_254')
func_255_call = mutated_mod.get_global_var('func_255')
call_3570 = func_254_call()
call_3571 = func_254_call()
output = relay.Tuple([bop_3548,call_3570,])
output2 = relay.Tuple([bop_3548,call_3571,])
func_3574 = relay.Function([var_3546,var_3547,], output)
mod['func_3574'] = func_3574
mod = relay.transform.InferType()(mod)
var_3575 = relay.var("var_3575", dtype = "float32", shape = ())#candidate|3575|()|var|float32
var_3576 = relay.var("var_3576", dtype = "float32", shape = (10, 1))#candidate|3576|(10, 1)|var|float32
output = func_3574(var_3575,var_3576,)
func_3577 = relay.Function([var_3575,var_3576,], output)
mutated_mod['func_3577'] = func_3577
mutated_mod = relay.transform.InferType()(mutated_mod)
func_897_call = mod.get_global_var('func_897')
func_898_call = mutated_mod.get_global_var('func_898')
call_3579 = relay.TupleGetItem(func_897_call(), 0)
call_3580 = relay.TupleGetItem(func_898_call(), 0)
output = call_3579
output2 = call_3580
func_3584 = relay.Function([], output)
mod['func_3584'] = func_3584
mod = relay.transform.InferType()(mod)
mutated_mod['func_3584'] = func_3584
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3584_call = mutated_mod.get_global_var('func_3584')
call_3585 = func_3584_call()
output = call_3585
func_3586 = relay.Function([], output)
mutated_mod['func_3586'] = func_3586
mutated_mod = relay.transform.InferType()(mutated_mod)
func_408_call = mod.get_global_var('func_408')
func_409_call = mutated_mod.get_global_var('func_409')
call_3601 = relay.TupleGetItem(func_408_call(), 0)
call_3602 = relay.TupleGetItem(func_409_call(), 0)
output = relay.Tuple([call_3601,])
output2 = relay.Tuple([call_3602,])
func_3612 = relay.Function([], output)
mod['func_3612'] = func_3612
mod = relay.transform.InferType()(mod)
output = func_3612()
func_3613 = relay.Function([], output)
mutated_mod['func_3613'] = func_3613
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3698 = relay.var("var_3698", dtype = "float32", shape = (1, 13, 16))#candidate|3698|(1, 13, 16)|var|float32
var_3699 = relay.var("var_3699", dtype = "float32", shape = (5, 13, 16))#candidate|3699|(5, 13, 16)|var|float32
bop_3700 = relay.less_equal(var_3698.astype('bool'), var_3699.astype('bool')) # shape=(5, 13, 16)
bop_3707 = relay.equal(bop_3700.astype('bool'), var_3698.astype('bool')) # shape=(5, 13, 16)
output = bop_3707
output2 = bop_3707
func_3711 = relay.Function([var_3698,var_3699,], output)
mod['func_3711'] = func_3711
mod = relay.transform.InferType()(mod)
var_3712 = relay.var("var_3712", dtype = "float32", shape = (1, 13, 16))#candidate|3712|(1, 13, 16)|var|float32
var_3713 = relay.var("var_3713", dtype = "float32", shape = (5, 13, 16))#candidate|3713|(5, 13, 16)|var|float32
output = func_3711(var_3712,var_3713,)
func_3714 = relay.Function([var_3712,var_3713,], output)
mutated_mod['func_3714'] = func_3714
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3742 = relay.var("var_3742", dtype = "float64", shape = (12, 8))#candidate|3742|(12, 8)|var|float64
uop_3743 = relay.sin(var_3742.astype('float64')) # shape=(12, 8)
func_1691_call = mod.get_global_var('func_1691')
func_1696_call = mutated_mod.get_global_var('func_1696')
var_3757 = relay.var("var_3757", dtype = "float32", shape = (324,))#candidate|3757|(324,)|var|float32
const_3758 = relay.const([1.860355,-9.184334,6.758875,-1.476108,-6.846765,9.954287,9.663380,6.309597,-0.624144,4.820257,2.264272,0.898362,5.277055,1.218471,2.894856,6.858376,4.286155,9.196319,1.854311,-0.091129,8.651452,-5.336666,-0.671725,8.518705,0.476558,-2.254852,-9.694306,3.161666,5.569813,-1.323723,3.663096,-0.831452,-8.988017,8.829766,8.038602,2.529594,-9.250760,9.459973,-5.910777,7.283242,2.989365,-0.111187,7.290720,-4.020225,3.901182,9.968908,-0.972769,3.091224,6.894839,-3.630790,-6.288646,-4.293922,-4.171267,8.528037,-3.822191,-2.279421,3.132464,-4.026963,5.370747,8.463423,1.659384,3.460140,-9.658921,-6.110358,-6.534522,0.285740,6.760191,0.083181,6.032944,-9.933828,4.516943,-6.154915,5.048972,8.502205,2.265707,0.069577,-0.095253,-5.014541,-9.964212,-2.749260,2.830956,0.003826,0.753659,-0.269787,-0.443499,4.848596,-7.587911,-6.376567,-1.980295,-1.782469,-6.872283,-7.423144,-2.063367,-2.430757,-3.262008,8.025481,-9.645133,-8.832736,6.673592,-7.983873,9.875188,4.961907,8.893319,9.703237,-0.184275,2.677935,-1.947583,1.544123,-7.246688,1.312962,-3.460045,9.765002,8.628891,4.544368,-0.454225,-0.315938,7.812102,4.863368,0.580219,-2.663028,-7.962064,6.659104,6.237512,-7.803579,-7.025100,-8.977043,-5.721475,8.969673,0.701377,0.021429,-0.758192,7.432331,1.015563,0.763600,-7.671828,-3.933545,4.730784,5.930307,-3.769333,8.231849,5.670076,-8.291373,5.151206,7.405452,-9.700841,-9.364394,6.113633,-2.957340,7.050535,5.639532,6.482944,-2.894872,8.247412,5.710389,0.617397,-7.591499,0.297912,1.356027,-1.696458,4.706720,3.108860,-1.928758,-2.392099,3.981325,-0.962193,4.380418,9.443802,-6.151892,0.640699,-7.731971,-0.428032,8.243041,8.200364,-9.175458,8.447766,-8.140896,-9.049152,4.364201,8.331544,-2.519025,-9.194667,6.853717,3.396633,-7.762525,-6.272654,-1.225319,-0.552048,-2.262062,-7.754479,3.388635,-3.978176,-7.229098,-9.978498,0.552517,-4.471047,6.233438,1.632872,-6.002558,2.431923,1.575396,-6.062383,-5.907080,8.354063,-5.998630,-6.848822,-4.592239,-8.982641,1.387201,-3.691987,-0.601672,-0.376934,0.493346,7.300994,-1.489780,4.261523,-3.158352,0.788382,-3.693785,6.992512,-2.866795,-0.316323,-2.690765,1.067128,-3.203850,5.863208,0.936309,-3.177427,9.427226,-4.790698,-2.960648,-9.402645,1.429045,-3.240366,-6.342505,4.300320,1.775300,4.633687,-8.083040,2.992511,5.956209,5.987771,-3.787899,7.605030,0.451808,3.864185,2.631692,-6.067563,-0.050884,-4.019575,-4.302422,4.976578,-8.071801,-9.644926,-0.144146,9.707792,-9.583738,-2.076016,-6.035673,-4.286909,8.906134,-7.475474,1.606256,-0.512877,3.959287,5.736078,-9.204016,6.948785,3.884675,9.607279,6.068511,1.378164,7.328024,-1.938094,-0.943436,-5.462347,-7.500629,-8.746602,6.876180,3.970175,-0.926338,1.434904,-8.602813,4.408726,-5.113163,9.663492,1.600009,3.040911,-0.769470,-5.163174,-4.975629,-6.449742,5.862276,9.672451,0.014072,0.321644,5.285180,-5.298177,7.218165,5.969724,-5.016519,3.288337,-2.733241,1.903755,-8.574909,7.654247,-1.679039,-8.167760,-2.467093,-1.208621,-0.387470,7.888003,-9.618884,5.553975,3.103191,7.884592,4.047393,4.236575,6.246294,-1.177329,6.347418,8.814546,-6.040843,0.888087,8.845174,-7.963286,-0.654397,5.999577,-0.044702,-4.894710,3.930791,0.147151,8.600505,-6.103977,9.098640,-9.514057,-8.996437,-7.433253,1.240879,-6.665035,-5.358944,6.008835,-1.753751,0.366401,0.793485,2.094870,-1.173289,-4.452398,3.245038,-7.607222,5.323276,7.402351,-3.526769,-0.854242,-5.636196,-9.541357,-8.427596,8.807722,-2.727071,0.051507,-5.251884,1.935477,3.314921,-6.221539,-3.533652,-8.053467,-5.988719,-6.884630,2.850151,-2.469330,9.630024,-5.453694,-1.337241,-7.286718,2.156348,6.077237,-6.736629,-5.617728,-0.012227,9.760572,9.702946,-2.939688,2.630823,9.266463,-7.930520,-5.483419,-4.658309,6.088658,-9.654413,-3.146664,1.364381,-9.010288,-2.125389,3.185060,1.119697,1.917409,-9.960643,-9.409876,-4.506935,7.151492,-1.594538,-0.118927,-4.230029,2.945924,-9.363492,-4.890229,1.584147,2.641228,1.477193,-8.781551,2.422899,-4.030348,-1.699990,-8.595920,-3.235649,8.387211,3.588756,8.156268,7.466068,9.757161,-0.459830,1.282908,-3.951796,2.951659,0.918527,-5.230250,-7.860347,4.947623,-8.383708,0.200428,1.981975,3.010706,2.681907,-3.458055,8.413063,-0.912684,-1.892636,-1.825677,-0.169441,-0.466618,9.525322,9.490602,3.327691,-8.792717,-0.745541,-0.218219,-0.357542,-1.699353,9.995226,-6.949527,-1.029368,-6.221850,-0.441460,8.641302,2.501880,7.378126,9.166436,-7.608766,6.021510,-8.275610,-5.948072,0.775851,-0.062131,-4.141732,-3.283532,-5.373850,5.173101,7.114189,4.247880,9.565405,8.978056,-1.486123,4.117863,0.409546,3.990526,-3.831341,5.009341,0.508058,4.061087,2.709979,-4.389416,0.051332,-1.439301,-0.261630,6.584887,-3.156701,-3.280314,7.331456,3.973457,-9.546987,-2.867592,7.683041,-8.745749,5.953940,-5.001842,6.194055,9.585508,3.487369,-3.636530,5.224009,9.597089,-2.074246,-0.277145,8.547068,-7.086042,9.693497,-1.215952,-5.991445,0.065025,-5.614690,-7.798407,-3.905866,-2.796849,-9.975044,1.032373,3.649184,4.738320,-8.854557,9.187659,-4.043936,0.430409,5.364332,-9.694573,-3.100184,9.461878,1.710074,5.289378,2.522220,0.965441,-5.774765,-8.383252,7.584017,3.017190,4.132387,5.609374,-4.713406,9.114688,-1.853749,-3.982510,2.017213,-1.097021,6.381445,8.984604,4.218798,-3.329834,1.143386,7.444040,3.686324,0.795069,-2.905940,-4.894167,4.485081,-7.016261,3.632906,4.555576,-9.220830,2.188764,0.337969,5.777073,-8.694755,-5.293360,-4.700601,-0.011738,-3.448762,-2.054831,3.375137,5.184343,7.451085,-8.746640,-6.279387,6.336447,1.653434,9.759557,5.862446,-1.939356,-2.703982,-8.879769,-7.947736,-6.887910,-5.826262,1.997334,3.865275,-6.372665,-0.800648,0.844322,-3.055150,-9.966945,-4.137417,-2.765154,-9.416066,3.498391,5.049305,1.598128,-3.070166,-2.736316,-6.410043,-0.169308,0.615413,-3.355702,-2.779739,-2.964096,9.336687,-1.654483,-0.483318,2.068878,5.412324,-7.505521,-4.118014,3.992147,2.068568,2.334921,4.665035,-3.163893,8.117799,9.417729,-5.415711,-4.158627,-2.906945,4.561275,-6.219610,9.436331,4.982389,-4.853298,1.527403,1.869767,-0.354779,-2.601552,-0.454726,-1.082658,-4.349479,6.630672,3.190324,-3.820388,6.929000,7.317039,-5.199056,0.746571,-6.346660,-8.412673,9.605457,-8.808254], dtype = "float64")#candidate|3758|(640,)|const|float64
call_3756 = relay.TupleGetItem(func_1691_call(relay.reshape(var_3757.astype('float32'), [324,]), relay.reshape(var_3757.astype('float32'), [324,]), relay.reshape(const_3758.astype('float64'), [640,]), ), 0)
call_3759 = relay.TupleGetItem(func_1696_call(relay.reshape(var_3757.astype('float32'), [324,]), relay.reshape(var_3757.astype('float32'), [324,]), relay.reshape(const_3758.astype('float64'), [640,]), ), 0)
func_1812_call = mod.get_global_var('func_1812')
func_1813_call = mutated_mod.get_global_var('func_1813')
call_3762 = relay.TupleGetItem(func_1812_call(), 0)
call_3763 = relay.TupleGetItem(func_1813_call(), 0)
func_218_call = mod.get_global_var('func_218')
func_220_call = mutated_mod.get_global_var('func_220')
const_3765 = relay.const([[2,-6,-1,-6,3,-1,-8,4,2,-7,1,-7,-7,-9,-7,-7,-4,-7,-5,7,-2,3,-3,-3,2,-3,-3,1,3,3,3,-10,-4,10,-2,9,7,4,-6,8,9,4,2,-4,-5,5,-8,5,-6,3,-3,-1,-2,-4,-4,-10,-4,6,8,-2,-8,3,4,10,-2,-9,3,9,7,-6,6,7,-10,7,-5,-3,8,1,-3,-3,3,-1,-1,4,2,2,-7,9,2,-5,6,1,3,5,4,10],[7,-6,10,8,-3,1,7,-7,-5,-7,-10,5,2,-1,6,-6,3,-1,3,4,-10,-5,-4,-10,-5,-3,5,2,6,6,5,4,3,10,9,-3,10,7,-10,-4,-4,-4,5,-6,9,2,3,1,-5,3,4,2,6,7,-8,5,-8,-1,-4,3,-1,5,-2,10,-7,1,3,-3,-5,-6,-1,-5,9,1,2,-5,10,-7,-8,2,-3,7,8,-4,-10,6,-3,-2,6,4,-7,3,10,-10,8,-6],[-6,-7,-4,-7,-3,-8,5,-7,-3,-1,1,-10,5,-1,-3,-3,1,3,-3,2,5,-10,-9,-1,9,5,-7,1,4,-6,3,5,-1,1,-9,8,4,4,5,8,-1,-4,-7,-2,-10,-7,8,-7,8,-1,-3,-1,-6,-10,4,9,-10,-4,9,-7,2,4,-4,9,10,-1,10,10,8,-1,10,8,2,-9,7,1,-8,9,7,5,9,-2,7,5,4,8,3,-7,7,7,2,3,-5,3,2,9],[4,-10,-3,-6,10,-2,1,9,3,-4,-6,-6,-1,10,7,-7,-1,1,4,1,-5,-9,2,6,6,10,-9,2,5,5,3,4,9,-8,-7,8,-1,8,-7,-1,-4,-6,9,-3,1,-8,-8,-8,10,-1,-4,8,-6,-1,1,-7,-2,8,-2,-6,8,7,6,6,-1,4,-6,3,7,8,-7,-4,9,-10,4,-3,-10,10,-8,-4,3,-6,-8,-2,4,-4,-4,-8,10,-5,-4,7,-3,-9,-3,-9],[1,-5,4,-4,9,7,-4,-10,5,-9,6,-4,-2,7,8,-10,-1,5,-6,4,-7,-3,-10,-7,8,5,-1,2,-3,-7,-7,7,-5,3,-3,9,10,3,4,-4,-6,-6,-7,8,-7,1,3,-2,-4,2,-2,7,2,9,-10,8,-1,2,-1,1,7,-2,2,1,-8,3,-3,-1,10,2,-4,-5,5,4,8,-5,-7,-5,10,-5,8,-8,10,-7,1,2,10,-10,-7,-10,-10,-1,-2,6,-4,-10],[-2,-3,-10,-3,-8,8,-2,-2,3,-3,-8,9,-1,-2,9,7,-2,2,-9,-5,-9,6,3,10,1,-10,8,-9,-6,5,-2,-5,-3,6,-1,10,4,-1,5,3,-5,-4,-4,-3,3,6,-10,-10,7,4,7,-7,-1,7,8,-3,5,-7,10,9,-7,1,-5,2,-6,-9,-2,3,-5,-1,-4,-5,2,8,-1,7,4,-7,8,-9,6,5,-4,-3,-1,-1,9,-4,9,2,2,8,-7,3,4,8]], dtype = "int8")#candidate|3765|(6, 96)|const|int8
call_3764 = relay.TupleGetItem(func_218_call(relay.reshape(const_3765.astype('int8'), [6, 16, 6])), 3)
call_3766 = relay.TupleGetItem(func_220_call(relay.reshape(const_3765.astype('int8'), [6, 16, 6])), 3)
func_3061_call = mod.get_global_var('func_3061')
func_3066_call = mutated_mod.get_global_var('func_3066')
const_3768 = relay.const([-10,3,5,1,-4,-2,-4,-9,-1,2,-6,-10,7,10,-6,3,-10,-10,-8,8,4,8,-3,4,1,-9,5,7,1,-1], dtype = "uint8")#candidate|3768|(30,)|const|uint8
var_3769 = relay.var("var_3769", dtype = "uint8", shape = (5, 84))#candidate|3769|(5, 84)|var|uint8
const_3770 = relay.const([7.818640,-2.652289,-5.397981,1.445819,-1.197331,-5.221243,-0.228949,-0.792956,2.937625,-7.749838,-0.151420,-2.109077,1.349813,5.071041,5.453088,4.019397,1.107692,-7.005146,0.123204,9.533233,6.534702,-8.409087,2.926608,9.352418,-6.406695,-6.980874,-0.790590,7.064472,-6.987984,5.428680,4.477721,-0.235293,-3.109326,-5.600649,-9.934256,0.425916,0.396998,4.800438,5.749862,5.803690,-5.864097,-5.501817,2.830686,-8.089116,-0.360258,4.600343,7.299942,2.560252,-8.752215,8.870260,8.521335,-1.680311,-7.632769,-4.756325,-2.543272,-0.027146,2.345653,-8.227546,-3.757394,-3.855008,-7.023097,2.039209,-4.655805,1.250355,-6.365974,2.494618,-5.733280,1.780048,-1.527844,-8.707935,9.766610,7.143320,9.919887,8.163292,1.937115,9.831470,0.548568,7.026370,5.810159,5.695955,4.423844,6.052824,-6.190921,2.527405,-4.944452,-9.928234,-7.482196,4.367883,7.917083,4.819171,-3.771547,-0.273294,8.116866,5.194502,-6.870742,7.301407,-1.645418,0.679656,0.236519,1.401045,-2.043252,6.764092,7.521763,9.742829,-4.281970,6.533505,6.545283,-6.985510,5.240768,1.406786,-6.163339,-1.264930,2.860794,-3.426294,-3.112352,8.938957,2.220731,5.009875,7.616767,4.427027,6.278300,-9.866056,-2.805196,-2.165188,1.708399,8.411561,-6.135710,-4.681966,-3.923003,-0.669529,-2.569977,0.370208,-5.783458,-3.051297,3.556076,9.895983,5.689918,2.145216,2.194561,8.893479,7.360306,-8.996454,-9.399712,-9.326597,2.174814,4.452908,-2.731336,3.067131,4.938719,-1.055651,6.371991,-0.822760,3.049204,-2.304409,6.109799,0.397500,9.760574,-6.870943,-1.262796,-8.591140,1.968216,9.785669,-8.497014,-6.072745,1.046870,4.244554,8.016826,1.037096,8.625529,9.086806,0.738135,-2.333591,-2.184004,9.309537,8.832005,3.972713,6.412434,-6.589533,9.019672,-7.554360,0.646792,-8.240915,0.802743,4.995457,8.720718,9.904955,-7.509433,5.692686,-0.694881,5.237275,-9.848381,-3.571260,-8.738024,-0.813405,6.829834,-7.130880,-0.159552,6.884306,-7.992899,-9.053646,0.354939,-3.203061,1.179310,8.973913,-8.923971,2.529085,5.731134,8.023182,-0.610180,1.191104,-9.236269,-9.832631,-3.684171,-4.898217,1.784234,9.683886,-8.638988,-2.208012,8.383521,1.184431,-4.612851,3.570487,-6.485714,7.271980,-4.690294,-2.136594,-8.824703,3.663849,5.269837,1.211691,-4.474595,0.267545,3.471998,7.749788,0.481376,-0.881591,9.214653,0.176852,-8.098298,1.898979,-5.823346,8.203728,-3.156685,9.415559,-1.588315,-6.316647,0.649180,5.390201,6.369046,-6.049640,1.426683,2.845358,-7.978942,-1.287867,8.726765,0.611466,-6.444194,2.514842,8.510078,-7.221402,1.264983,-3.815293,-8.304178,8.957459,-1.269717,5.803537,0.331244,8.509798,0.932070,-6.215629,8.025683,-7.335817,3.900772,-5.680507,8.189290,-2.974333,-3.051594,3.914380,6.316584,-4.104110,-5.750233,4.709853,-5.618798,0.163063,5.400517,5.560614,3.969659,-9.989402,4.992698,7.521710,1.701538,0.220151,7.318737,-3.918943,-8.411898,-1.703955,-1.832738,-8.451002,-9.969546,7.610302,7.764336,-0.485955,-3.397820,4.007218,-8.862217,-1.268526,0.588442,4.419872,6.845503,3.693149,-2.298438,-0.485542,4.700465,-7.613573,5.387663,6.710748,-7.809056,4.324227,-5.003521,-2.108156,-3.759037,-0.048760,-6.959152,-8.455797,-0.197200,2.695270,9.830041,6.450797,-2.002800,2.974920,5.767171,7.739275,-9.026976,4.780000,7.317581,9.074588,-9.481502,0.320447,0.116804,-3.360264,4.182651,8.411375,-3.441546,-3.356491,-2.215187,-8.968442,-6.305406,8.479797,-5.819409,-6.673530,-5.320319,3.720548,5.902416,7.839129,-6.282319,-4.668775,-6.880237,5.733347,1.201742,7.892747,9.958127,7.021492,8.929646,-6.034389,6.699616,-8.079694,3.916970,6.210133,-0.834534,1.186666,-6.141581,9.187917,4.263963,9.973358,-9.868136,-0.964335,-4.089869,7.677873,-0.131952,-3.708072,-3.931635,7.504104,-0.984022,5.392929,-7.007765,-0.650010,-8.028448,9.447411,-5.181743,-8.747998,3.013335,-9.340293,0.809943,-8.498211,2.644391,2.270221,9.251113,1.813693,4.608287,-5.839233,-8.283015,5.841862,-2.892204,-0.620507,2.265300,2.714401,-7.111707,4.947808,-0.450838,3.857578,-6.631013,4.143906,-5.714954,4.506037,9.353003,3.617765,-3.103103,1.351776,5.824584,3.357045,-5.837327,2.766164,-5.602542,-8.980111,-1.341018,5.788768,4.319571,-6.551358,9.835135,0.194724,-0.901936,-4.971823,6.702874,3.766377,-8.752519,-7.404689,-8.590577,7.050993,-8.595593,3.998420,2.361043,0.567748,5.876578,-6.623219,2.636524,-0.771800,-7.226780,0.348385,-5.363494,-8.581213,9.140020,4.097215,8.908409,5.632660,5.996169,-8.440885,1.991778,4.589264,-3.096706,4.657509,3.728257,1.276246,7.778802,-7.321718,1.903724,-9.892088,7.896342,-4.825159,-6.600945,-6.493761,-1.499705,-0.412708,-0.793291,0.067242,-2.340412,-1.761095,-5.868723,5.744520,-1.265732,3.689479,6.260003,2.515610,7.037735,5.115128,0.340911,9.559132,-1.292515,0.876272,0.785079,1.356564,2.169736,-4.727709,4.481707,7.524646,2.398944,-9.180749,-9.178816,2.546440,5.836079,1.609847,-9.670083,3.976062,3.140731,9.685396,3.494085,-1.413768,-1.328867,-4.591884,-5.910714,-8.032773,-7.888504,-3.357129,-8.239236,-5.221034,-1.912900,-8.453202,5.902476,-4.936054,-6.312830,1.476353,-0.376766,-9.097348,-2.104161,0.938584,7.088118,-5.189819,-0.692367,-7.693076,-5.617086,-9.565709,7.295197,4.311528,-3.244124,2.605481,8.561780,-5.883679,-4.857311,1.551599,6.817048,4.989476], dtype = "float32")#candidate|3770|(540,)|const|float32
var_3771 = relay.var("var_3771", dtype = "float32", shape = (6,))#candidate|3771|(6,)|var|float32
call_3767 = relay.TupleGetItem(func_3061_call(relay.reshape(const_3768.astype('uint8'), [6, 1, 5]), relay.reshape(var_3769.astype('uint8'), [6, 14, 5]), relay.reshape(const_3770.astype('float32'), [540,]), relay.reshape(var_3771.astype('float32'), [6, 1]), ), 0)
call_3772 = relay.TupleGetItem(func_3066_call(relay.reshape(const_3768.astype('uint8'), [6, 1, 5]), relay.reshape(var_3769.astype('uint8'), [6, 14, 5]), relay.reshape(const_3770.astype('float32'), [540,]), relay.reshape(var_3771.astype('float32'), [6, 1]), ), 0)
func_1093_call = mod.get_global_var('func_1093')
func_1098_call = mutated_mod.get_global_var('func_1098')
var_3774 = relay.var("var_3774", dtype = "float64", shape = (660, 1))#candidate|3774|(660, 1)|var|float64
call_3773 = relay.TupleGetItem(func_1093_call(relay.reshape(var_3774.astype('float64'), [660,]), relay.reshape(const_3765.astype('int8'), [576,]), relay.reshape(const_3765.astype('int8'), [3, 16, 12]), ), 12)
call_3775 = relay.TupleGetItem(func_1098_call(relay.reshape(var_3774.astype('float64'), [660,]), relay.reshape(const_3765.astype('int8'), [576,]), relay.reshape(const_3765.astype('int8'), [3, 16, 12]), ), 12)
func_2672_call = mod.get_global_var('func_2672')
func_2673_call = mutated_mod.get_global_var('func_2673')
call_3777 = relay.TupleGetItem(func_2672_call(), 0)
call_3778 = relay.TupleGetItem(func_2673_call(), 0)
func_3371_call = mod.get_global_var('func_3371')
func_3373_call = mutated_mod.get_global_var('func_3373')
var_3828 = relay.var("var_3828", dtype = "float32", shape = (840,))#candidate|3828|(840,)|var|float32
call_3827 = relay.TupleGetItem(func_3371_call(relay.reshape(var_3828.astype('float32'), [7, 15, 8])), 3)
call_3829 = relay.TupleGetItem(func_3373_call(relay.reshape(var_3828.astype('float32'), [7, 15, 8])), 3)
func_1949_call = mod.get_global_var('func_1949')
func_1953_call = mutated_mod.get_global_var('func_1953')
var_3832 = relay.var("var_3832", dtype = "int8", shape = (176, 1))#candidate|3832|(176, 1)|var|int8
const_3833 = relay.const([5,-9,7,-9,9,4,-3,-1,8,-1,5,-9,-1,-5,3,-6,-10,-5,2,4,5,7,3,-5,4,-1,-7,7,-1,-10,4,3,2,-1,-3,6,-4,-5,3,-8,-7,9,-1,-3,-9,-7,2,-7,9,9,-2,9,9,3,-6,-10,3,-7,6,-1,7,-7,9,1,1,-9,6,1,1,-9,8,-8,-9,-9,6,-9,7,10,-2,-1,3,-10,-7,-8,8,5,-9,8,9,4,-7,-10,10,8,3,7,-1,-8,-9,-7,6,-2,5,4,9,-1,2,1,-1,6,-10,1,1,10,-6,8,8,-6,3,-9,9,-9,1,-6,10,-6,6,-6,7,-8,4,6,7,-7,3,3,7,-4,3,-3,-2,7,-6,-7,-8,9,-3,6,-6,-2,-10,-10,8,-7,6,6,-9,-7,6,10,-1,8,4,5,8,7,-4,-6,5,-3,-1,10,-6,7,-3,7,-5,-7,2,9,1,7,4,-8,-7,1,-4,10,-10,-1,3,6,-5,-9,6,7,3,-3,-9,10,-3,-6,2,4,-10,-4,5,2,-2,6,-6,-7,9,10,2,6,10,4,-7,-5,-5,1,-8,9,2,2,-8,-1,-1,-7,-8,10,-2,-2,-5,7,6,8,-10,6,2,-2,-6,-9,3,-2,6,10,6,-8,-7,-9,-2,-6,5,4,1,10,-8,-8,5,-3,10,1,-3,7,6,4,-6,-10,-6,10,10,-7,-1,1,-5,-8,-8,-2,-2,3,-6,5,9,5,-10,3,7,2,6,-7,-8,5,6,6,-6,-5,8,-4,-5,-10,-4,-9,-2,-2,-4,9,-3,-7,4,8,-4,2,-3,7,-5,-5,10,3,-4,-2,5,-10,10,1,1,-3,2,-7,2,-3,2,10,-3,-8,10,-3,5,-5,-1,6,6,8,-5,-3,10,-5,-5,6,10,4,6,-8,1,4,10,5,-9,3,10,10,3,4,-2,7,6,5,1,5,-7,-5,10,4,-4,3,5,4,1,-8,-8,-7,8,-8,-5,10,-1,8,4,-1,9,2,7,1,-4,1,6,6,-1,-3,-7,-6,-10,10,-6,-4,-2,10,2,9,4,10,9,-6,-2,7,-3,-8,9,7,-6,2,-7,1,-6,9,-2,10,5,-2,-6,-8,1,-10,-3,-6,9,-9,10,8,-2,-6,6,-6,3,-9,6,-5,-1,-7,-3,-5,1,-9,-6,9,-3,-2,2,-1,-5,-1,6,-10,-3,3,-2,-10,-4,5,7,-10,2,1,-8,-8,-3,-6,7,9,-2,7,3,-6,6,7,-9,6,-8,5,9,1,-4,-10,4,2,2,-10,9,-1,9,-10,8,-10,-1,4,-10,-2,3,-7,3,-3,-1,1,-10,6,-4,-10,10,-7,-4,-3,-6,-9,8,8,-7,8,5,-9,2,6,10,8,-5,1,8,6,-8,4,2,9,4,-9,-5,-4,2,4,-6,-1,9,-2,5,-10,-2,-1,8,-6,-7,1,-1,-10,-5,-4,1,1,7,-7,-1,7,3,10,8,-6,-8,9,-2,-4,-7,9,4,-6,-5,-1,-1,-10,6,8,3,-1,2,2,-10,10,-5,-3,-9,7,-3,-10,9,-9,-9,1,4,7,-3,-8,5,10,9,6,4,3,4,-2,-5,-7,9,-8,-3,6,4,-9,-6,-7,-4,-6,-8,6,4,4,8,-4,1,-9,-7,-10,-3,-9,10,5,-1,-10,9,1,-3,-1,-8,7,8,-4,3,7,-5,-4,-2,-6,-9,4,-4,9,1,6,6,-9,-9,9,-8,-9,-3,-5,3,10,10,10,-10,-9,1,-8,-9,3,-9,2,-1,10,6,-8,7,-1,-3,-10,9,4,5,9,9,3,-6,-7,8,10,-3,10,-10,6,1,-6,3,10,-2,-3,-3,-10,3,-6,5,2,-6,-5,-3,-10,6,5,-8,1,-8,-10,1,1,-7,5,-5,-9,1,-9,-6,-9,7,-6,-2,4,9,10,5,-6,-8,-6,7,-8,-1,3,8,-2,9,-1,-4,9,-8,-5,-10,-5,1,-3,-9,7,8,7,-3,10,-8,-10,-7,-8,-8,7,-6,-8,6,-5,10,4,1,3,-2,-5,7,7,-4,6,9,-2,10,4,-9,10,3,1,-5,8,-9,-10,-8,-10,-1,6,2,7,-8,-7,-1,10,10,8,6,-3,-8,-3,9,3,-2,9,9,-7,6,5,2,1,-7,-7,2,-9,-4,-1,6,2,2,-3,9,9,-3,-5,2,-6,3,-3,5,9,4,4,-5,1,10,-10,3,6,-6,10,-7,8,-2,-8,4,-8,-1,-2,9,-4,5,-7,5,-4,-6,-10,3,-6,-2,-3,6,-5,-4,-1,-8,3,9,4,5,5,-4,-6,-7,-3,-9,-3,8,9,-5,-8,-5,8,10,8,-10,9,-1,1,9,7,10,2,2,-1,-4,-1,9,3,-1,-7,7,5,10,6,-9,5,6,7,-1,-4,9,-3,5,8,-8,-9,10,3,-9,7,-5,-7,-10,-1,-5,3,10,-2,4,5,4,7,7,4,-2,-5,-1,-9,7,3,5,-5,4,4,-7,9,2,-7,-7,-6,-10,-5,-3,6,-4,-1,-7,-4,4,-6,-6,4,1,8,2,7,4,5,-8,-9,-9,1,-9,-4,4,3,7,9,-7,-7,10,-1,10,4,-6,-10,7,9,-10,-6,7,6,5,5,10,-2,-8,5,-8,5,3,-7,9,-6,-2,-3,2,-6,-5,-2,3,-7,5,-8,-2,5,8,-8,1,-8,-6,4,10,-8,1,2,-2,-2,3,-9,2,-2,9,9,-4,3,-8,10,6,-7,-6,10,7,4,-3,5,-9,8,-5,-8,-4,10,7,-3,6,2,8,9,-6,-5,7,4,7,4,9,-2,2,-6,-2,5,-9,-9,1,-2,2,-3,9,5,-9,9,-10,2,3,-10,-9,2,-4,-4,-10,9,6,-2,-1,-1,5,-8,-1,6,8,-8,3,-4,-2,1,7,-5,-3,-5,4,-5,-4,-4,3,10,8,-8,-7,-7,3,2,-9,2,-3,-4,-3,-8,2,-3,-1,-6,-10,-1,1,7,-7,5,-7,6,5,9,-1,-6,-8,-7,3,-10,-2,-1,1,8,-8,-4,1,7,3,4,2,8,1,10,1,-7,1,7,-6,9,-9,1,6,5,-5,-3,9,8,7,3,8,-4,-10,2,-5,1,8,3,1,1,-7,4,6,-7,10,4,2,8,10,-9,2,10,-2,-4,-2,-9,-9,-6,-5,-1,-5,-4,4,10,-3,-10,8,7], dtype = "int8")#candidate|3833|(1232,)|const|int8
call_3831 = relay.TupleGetItem(func_1949_call(relay.reshape(var_3832.astype('int8'), [11, 16, 1]), relay.reshape(const_3833.astype('int8'), [1232,]), relay.reshape(const_3770.astype('float32'), [5, 12, 9]), ), 1)
call_3834 = relay.TupleGetItem(func_1953_call(relay.reshape(var_3832.astype('int8'), [11, 16, 1]), relay.reshape(const_3833.astype('int8'), [1232,]), relay.reshape(const_3770.astype('float32'), [5, 12, 9]), ), 1)
output = relay.Tuple([uop_3743,call_3756,var_3757,const_3758,call_3762,call_3764,const_3765,call_3767,const_3768,var_3769,const_3770,var_3771,call_3773,var_3774,call_3777,call_3827,var_3828,call_3831,var_3832,const_3833,])
output2 = relay.Tuple([uop_3743,call_3759,var_3757,const_3758,call_3763,call_3766,const_3765,call_3772,const_3768,var_3769,const_3770,var_3771,call_3775,var_3774,call_3778,call_3829,var_3828,call_3834,var_3832,const_3833,])
func_3858 = relay.Function([var_3742,var_3757,var_3769,var_3771,var_3774,var_3828,var_3832,], output)
mod['func_3858'] = func_3858
mod = relay.transform.InferType()(mod)
mutated_mod['func_3858'] = func_3858
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3858_call = mutated_mod.get_global_var('func_3858')
var_3860 = relay.var("var_3860", dtype = "float64", shape = (12, 8))#candidate|3860|(12, 8)|var|float64
var_3861 = relay.var("var_3861", dtype = "float32", shape = (324,))#candidate|3861|(324,)|var|float32
var_3862 = relay.var("var_3862", dtype = "uint8", shape = (5, 84))#candidate|3862|(5, 84)|var|uint8
var_3863 = relay.var("var_3863", dtype = "float32", shape = (6,))#candidate|3863|(6,)|var|float32
var_3864 = relay.var("var_3864", dtype = "float64", shape = (660, 1))#candidate|3864|(660, 1)|var|float64
var_3865 = relay.var("var_3865", dtype = "float32", shape = (840,))#candidate|3865|(840,)|var|float32
var_3866 = relay.var("var_3866", dtype = "int8", shape = (176, 1))#candidate|3866|(176, 1)|var|int8
call_3859 = func_3858_call(var_3860,var_3861,var_3862,var_3863,var_3864,var_3865,var_3866,)
output = call_3859
func_3867 = relay.Function([var_3860,var_3861,var_3862,var_3863,var_3864,var_3865,var_3866,], output)
mutated_mod['func_3867'] = func_3867
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3872 = relay.const(-9, dtype = "uint16")#candidate|3872|()|const|uint16
var_3873 = relay.var("var_3873", dtype = "uint16", shape = (15, 11))#candidate|3873|(15, 11)|var|uint16
bop_3874 = relay.add(const_3872.astype('uint16'), var_3873.astype('uint16')) # shape=(15, 11)
uop_3890 = relay.log(bop_3874.astype('float32')) # shape=(15, 11)
output = uop_3890
output2 = uop_3890
F = relay.Function([var_3873,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_3873,], output2)
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
input_3873= np.array([[-1,-1,-6,-2,3,7,2,-2,-10,-10,9],[-4,4,-4,2,-1,4,7,5,4,-1,-7],[2,-2,6,-3,-8,1,6,-5,7,-1,-7],[2,10,-9,-6,4,8,10,7,-10,2,-1],[9,7,1,8,8,2,-5,-9,-9,-8,-6],[-2,-4,2,-6,-2,-8,-2,9,-10,9,5],[5,4,4,3,-2,4,1,-10,8,1,-3],[8,-2,-10,-10,-6,8,1,2,3,10,10],[5,10,3,1,6,-1,8,1,6,-3,5],[-10,9,-5,-3,10,-6,-3,-2,7,4,6],[9,9,3,4,7,9,9,5,-8,10,-7],[-2,-7,-7,-9,2,2,-7,-7,-7,7,10],[-1,-8,2,-2,2,8,-7,10,-9,5,-8],[-8,-5,9,10,-8,4,9,-3,-5,-9,2],[-7,3,9,-7,2,1,-10,-9,-2,3,-10]], dtype='uint16')
module1.set_input('var_3873', input_3873)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_3873, )
res3 = intrp3.evaluate()(input_3873, )
res4 = intrp4.evaluate()(input_3873, )
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
module5.set_input('var_3873', input_3873)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_3873, )
res7 = intrp7.evaluate()(input_3873, )
res8 = intrp8.evaluate()(input_3873, )
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
module9.set_input('var_3873', input_3873)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_3873, )
res11 = intrp11.evaluate()(input_3873, )
res12 = intrp12.evaluate()(input_3873, )
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
module13.set_input('var_3873', input_3873)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_3873, )
res15 = intrp15.evaluate()(input_3873, )
res16 = intrp16.evaluate()(input_3873, )
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
module17.set_input('var_3873', input_3873)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_3873, )
res19 = intrp19.evaluate()(input_3873, )
res20 = intrp20.evaluate()(input_3873, )
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
module21.set_input('var_3873', input_3873)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_3873, )
res23 = intrp23.evaluate()(input_3873, )
res24 = intrp24.evaluate()(input_3873, )
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

'''11.090248, 11.783418, 11.783357, 11.783357, 11.090355],
11.090248, 11.090187, 11.090065, 11.090065,      -inf],

'''