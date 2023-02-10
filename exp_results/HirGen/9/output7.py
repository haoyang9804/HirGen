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
var_3 = relay.var("var_3", dtype = "float64", shape = (11, 14, 4))#candidate|3|(11, 14, 4)|var|float64
uop_4 = relay.asinh(var_3.astype('float64')) # shape=(11, 14, 4)
output = uop_4
output2 = uop_4
func_7 = relay.Function([var_3,], output)
mod['func_7'] = func_7
mod = relay.transform.InferType()(mod)
var_8 = relay.var("var_8", dtype = "float64", shape = (11, 14, 4))#candidate|8|(11, 14, 4)|var|float64
output = func_7(var_8)
func_9 = relay.Function([var_8], output)
mutated_mod['func_9'] = func_9
mutated_mod = relay.transform.InferType()(mutated_mod)
var_41 = relay.var("var_41", dtype = "float32", shape = (3, 9))#candidate|41|(3, 9)|var|float32
uop_42 = relay.asin(var_41.astype('float32')) # shape=(3, 9)
var_54 = relay.var("var_54", dtype = "float32", shape = (3, 9))#candidate|54|(3, 9)|var|float32
bop_55 = relay.right_shift(uop_42.astype('uint64'), relay.reshape(var_54.astype('uint64'), relay.shape_of(uop_42))) # shape=(3, 9)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
var_62 = relay.var("var_62", dtype = "float64", shape = (616,))#candidate|62|(616,)|var|float64
call_61 = func_7_call(relay.reshape(var_62.astype('float64'), [11, 14, 4]))
call_63 = func_7_call(relay.reshape(var_62.astype('float64'), [11, 14, 4]))
output = relay.Tuple([bop_55,call_61,var_62,])
output2 = relay.Tuple([bop_55,call_63,var_62,])
func_66 = relay.Function([var_41,var_54,var_62,], output)
mod['func_66'] = func_66
mod = relay.transform.InferType()(mod)
mutated_mod['func_66'] = func_66
mutated_mod = relay.transform.InferType()(mutated_mod)
func_66_call = mutated_mod.get_global_var('func_66')
var_68 = relay.var("var_68", dtype = "float32", shape = (3, 9))#candidate|68|(3, 9)|var|float32
var_69 = relay.var("var_69", dtype = "float32", shape = (3, 9))#candidate|69|(3, 9)|var|float32
var_70 = relay.var("var_70", dtype = "float64", shape = (616,))#candidate|70|(616,)|var|float64
call_67 = func_66_call(var_68,var_69,var_70,)
output = call_67
func_71 = relay.Function([var_68,var_69,var_70,], output)
mutated_mod['func_71'] = func_71
mutated_mod = relay.transform.InferType()(mutated_mod)
var_224 = relay.var("var_224", dtype = "float64", shape = (6, 9))#candidate|224|(6, 9)|var|float64
uop_225 = relay.sigmoid(var_224.astype('float64')) # shape=(6, 9)
output = uop_225
output2 = uop_225
func_227 = relay.Function([var_224,], output)
mod['func_227'] = func_227
mod = relay.transform.InferType()(mod)
mutated_mod['func_227'] = func_227
mutated_mod = relay.transform.InferType()(mutated_mod)
var_228 = relay.var("var_228", dtype = "float64", shape = (6, 9))#candidate|228|(6, 9)|var|float64
func_227_call = mutated_mod.get_global_var('func_227')
call_229 = func_227_call(var_228)
output = call_229
func_230 = relay.Function([var_228], output)
mutated_mod['func_230'] = func_230
mutated_mod = relay.transform.InferType()(mutated_mod)
const_326 = relay.const([[7.562653,0.499500],[1.076762,4.113001],[-5.451189,-7.714428],[2.019194,9.624153],[-0.231621,0.970384],[8.516593,-4.802802],[-6.789853,6.318918],[-6.084199,0.249521],[2.067189,9.071086],[-2.698984,1.622005],[-0.919628,-1.892009]], dtype = "float32")#candidate|326|(11, 2)|const|float32
uop_327 = relay.asinh(const_326.astype('float32')) # shape=(11, 2)
output = uop_327
output2 = uop_327
func_331 = relay.Function([], output)
mod['func_331'] = func_331
mod = relay.transform.InferType()(mod)
output = func_331()
func_332 = relay.Function([], output)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
const_333 = relay.const([[[-8.452334,4.425323,-6.229416,-1.493796,6.630992,0.561019,0.233630,-5.101789,-0.357198,2.527360,5.301116,9.086895,9.364551,2.356297,-5.869153,-6.513943]],[[-1.399321,-2.003725,3.303670,-7.234445,-0.568260,-2.881093,3.625302,1.648632,-2.967728,8.347443,-5.251118,-1.257311,-2.704841,6.779404,-7.412351,5.164419]]], dtype = "float64")#candidate|333|(2, 1, 16)|const|float64
uop_334 = relay.rsqrt(const_333.astype('float64')) # shape=(2, 1, 16)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
const_338 = relay.const([-8.234933,-2.406052,-3.902417,9.757647,6.277252,1.797100,-4.326412,8.871188,-4.273286,0.893881,-0.369623,-7.681983,5.834826,7.362567,7.221882,-9.278433,-6.916757,-5.912626,-2.909701,-8.069132,-7.793870,7.021130,1.965159,-3.684703,-0.227733,-5.590174,6.232301,-5.147841,-8.299931,1.237513,-1.303953,7.916909,-7.060283,6.535971,3.593271,-6.972051,6.890787,-9.062742,-9.179778,7.832602,-1.193541,4.926928,-8.709720,-8.007742,-8.468411,-3.298511,2.450204,2.824425,6.633697,8.285570,-2.475385,8.899260,3.678443,-0.209390,-7.288671,-3.135272,3.319994,-4.076614,4.933402,6.225931,4.420465,-5.924123,2.489769,-5.058681,8.527000,-1.944585,-5.131632,7.575093,1.844913,2.288410,8.492897,3.446276,-0.189630,2.832838,-0.287559,8.272634,-6.880873,8.220308,0.409547,3.968641,2.571518,3.349981,9.856676,9.925111,-6.100097,-1.325656,-3.461708,0.691884,-9.565158,-1.982690,7.795135,-9.127604,3.439728,6.624167,6.375827,3.356216,-9.228224,-8.826063,-8.390025,2.139159,3.161369,-7.108007,1.937905,-7.345636,-4.373680,-0.760916,-2.220701,-0.115722,-4.595215,-6.499892,-3.361668,-3.585731,-7.026093,-8.385228,-5.706304,-2.418480,5.068564,8.718014,-9.499681,-1.600737,-5.037299,-8.213670,3.074750,1.115769,-4.615691,-5.371679,9.871980,-3.330516,3.273634,-9.285908,-2.586761,5.141387,-3.559761,-5.439136,-3.218999,0.446094,5.439152,8.134769,-8.468432,0.288962,9.687282,7.187524,-2.528190,-2.953294,-9.616181,-4.774160,6.887538,5.702307,-5.409094,-9.297889,-2.425473,-0.339314,8.931367,7.173813,-1.336137,3.804272,-2.452825,-5.327750,-8.397783,-6.207590,-0.777745,7.872583,0.022931,4.534552,-4.754468,-9.597154,2.953096,-7.406344,9.969114,4.502743,-0.409101,8.737022,5.368933,-2.379864,-9.253726,6.708345,3.910107,7.863762,6.123053,-9.023075,-9.135488,-0.845871,6.866574,0.915455,-4.368675,-9.671601,2.476061,8.182080,-7.960326,-6.938599,7.567799,7.284440,-7.048106,7.266856,7.714454,9.482947,-1.362732,4.250886,-0.747589,-8.000840,4.038964,-9.644102,-0.449712,-3.252852,4.350757,-6.307163,-7.512558,1.485580,-7.521296,-1.811540,-1.474906,9.425346,9.877229,7.424505,2.441670,7.457661,-9.357203,0.022603,4.207859,6.299619,-0.911176,9.098384,9.644785,8.040147,4.189045,2.552896,6.922422,8.378709,-3.591630,0.384500,-4.178189,-6.225201,-6.822448,-9.573630,3.680222,-5.992038,9.510243,-9.977648,-2.725484,-3.824935,1.874221,2.362685,-7.306163,-0.848392,9.572449,8.331800,-2.729623,4.285124,8.264984,-0.158838,5.981394,8.192650,1.599682,-8.247090,8.066137,5.414954,-0.189484,-3.514666,5.042764,9.603943,-9.125834,1.879476,3.664686,4.014640,-2.246544,9.213667,-0.595641,-0.398812,3.966378,4.144402,9.337446,4.144647,2.096064,-6.536224,-8.890724,-3.885711,-3.430322,0.068468,-3.915041,8.303371,7.419693,-0.825154,9.603596,9.287734,9.200116,0.100016,4.606623,0.932289,-6.354614,-9.509686,1.141213,0.598597,-3.168179,2.522157,2.432737,5.008472,-4.058585,-8.483128,-3.468781,-7.999314,-0.813147,-9.821244,-9.432608,0.072554,3.085960,-4.820701,-8.870363,3.210922,-3.735214,9.406985,5.101555,-4.125708,-4.946667,4.067239,-9.834726,7.365911,8.775727,3.600516,0.533684,2.286007,-2.120295,5.102741,6.348727,8.820703,-4.621521,-7.781833,7.386487,8.314916,3.351759,-1.482552,-7.801437,8.663861,9.913247,7.815370,0.310800,8.216074,-1.295766,-6.293597,3.040582,-7.797996,-8.850811,8.816515,6.054032,0.059537,3.975711,1.686795,1.936837,-0.649123,7.824154,7.063343,-3.758232,-3.263651,-7.344062,-3.517059,-3.256757,6.977437,9.544392,-5.078882,5.105833,-1.523063,3.062690,-0.258016,-2.921223,5.312227,-6.555075,-9.058353,2.297126,-6.255906,7.160074,-4.557333,6.603676,-7.000290,-5.537226,4.411781,-6.280261,7.686070,8.159938,3.560310,-5.169057,9.302415,8.923836,1.902521,-9.063373,-7.114691,-2.287409,6.577144,-7.574751,6.668950,-8.616934,-5.753383,-2.341419,-6.489046,9.323854,4.396162,-7.990954,5.208624,-3.081879,7.579511,-0.530459,-9.455184,-4.121041,-2.216740,6.088160,0.098115,-2.595515,-9.307553,0.080581,2.802419,-0.420798,-6.244117,8.676095,0.091995,-1.715966,3.445043,-4.760772,-6.433936,9.663207,-9.585717,-7.880186,6.252013,7.624535,6.902252,-6.425367,-9.101070,2.008221,-2.911793,-0.944693,-6.062531,-6.115546,9.376103,3.271287,0.995605,-7.866853,0.906947,6.594362,8.443417,-0.286850,-6.893307,2.433971,-0.490191,8.802521,-5.199714,-8.106072,0.156670,-1.207117,0.943856,-4.471681,-9.947818,-5.404427,5.162747,-8.311774,3.562290,1.758633,-8.596634,-8.688197,2.519000,8.417703,6.871851,-4.349688,-8.304376,5.195899,-6.263104,-2.393532,-1.439829,-3.731978,-4.561422,-0.495501,2.759087,5.957921,7.066459,-7.110684,-8.421423,-8.647371,5.710914,-3.661696,8.541216,-5.731490,3.971922,-4.633107,1.255825,0.423031,-7.534693,4.489680,0.816515,0.232459,-8.370357,4.958146,3.793848,-7.675221,-3.195913,-6.698272,-3.858316,9.152420,7.877676,2.952726,-9.810138,-1.399407,5.014627,-0.748016,5.513590,7.021570,-4.559590,-2.857684,-1.674528,-7.247784,4.179824,-8.179789,1.650176,7.348527,2.170534,-0.751082,6.470952,-5.057961,-6.830395,8.280866,-5.353731,-4.804536,9.623370,-0.209406,3.567376,-0.061202,7.725133,9.349987,-9.150566,7.582918,0.179849,8.231649,5.861281,0.866061,-1.940839,8.869866,8.978576,4.684909,-8.557145,-0.091113,1.043655,-2.044496,1.915905,2.791890,-5.817152,-3.620496,-9.498824,-3.254978,6.075983,-5.315877,-9.465507,0.401341,1.787802,-5.339770,-5.603101,0.875701,8.991884,0.911577,4.152850,-0.661854,0.828732,-6.287202,-6.572089,8.282798,-8.816260,4.876464,0.353568,0.963488,-1.542127,-3.393065,-2.435932,-7.458552,0.235907,9.237504,-8.649781,-2.320774,-9.191485,-5.433120,2.879132,-0.664431,3.883071,-4.140184,-9.953448,6.123948,2.614909,-8.871927,0.957069,0.085108,-1.198061,-5.754534,-9.959480,-2.822713,8.618488,2.899598,1.654250,-6.973477,-7.008544,5.581503,-8.304887,5.710891,2.871281,-2.279991,-6.773588,1.388187,-6.719910,7.444460,1.017564,9.990969,-6.037749,2.285094,-0.134944,-7.211080,1.810848,-5.373311,-1.275472,7.875728,9.332316,0.057575,-6.771547,-1.249797,-7.851061], dtype = "float64")#candidate|338|(616,)|const|float64
call_337 = func_7_call(relay.reshape(const_338.astype('float64'), [11, 14, 4]))
call_339 = func_7_call(relay.reshape(const_338.astype('float64'), [11, 14, 4]))
bop_340 = relay.add(uop_334.astype('int64'), relay.reshape(const_333.astype('int64'), relay.shape_of(uop_334))) # shape=(2, 1, 16)
bop_352 = relay.greater(bop_340.astype('bool'), relay.reshape(uop_334.astype('bool'), relay.shape_of(bop_340))) # shape=(2, 1, 16)
var_355 = relay.var("var_355", dtype = "int64", shape = (2, 16, 16))#candidate|355|(2, 16, 16)|var|int64
bop_356 = relay.divide(bop_340.astype('float64'), var_355.astype('float64')) # shape=(2, 16, 16)
bop_363 = relay.power(bop_352.astype('float64'), bop_356.astype('float64')) # shape=(2, 16, 16)
bop_366 = relay.logical_xor(bop_356.astype('int16'), relay.reshape(var_355.astype('int16'), relay.shape_of(bop_356))) # shape=(2, 16, 16)
uop_370 = relay.log2(bop_363.astype('float64')) # shape=(2, 16, 16)
uop_373 = relay.sqrt(uop_370.astype('float32')) # shape=(2, 16, 16)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_375 = func_331_call()
call_376 = func_331_call()
bop_382 = relay.bitwise_xor(uop_370.astype('int8'), relay.reshape(bop_366.astype('int8'), relay.shape_of(uop_370))) # shape=(2, 16, 16)
uop_385 = relay.tan(bop_366.astype('float64')) # shape=(2, 16, 16)
uop_392 = relay.acosh(call_337.astype('float32')) # shape=(11, 14, 4)
uop_394 = relay.acosh(call_339.astype('float32')) # shape=(11, 14, 4)
bop_397 = relay.minimum(call_337.astype('int32'), relay.reshape(uop_392.astype('int32'), relay.shape_of(call_337))) # shape=(11, 14, 4)
bop_400 = relay.minimum(call_339.astype('int32'), relay.reshape(uop_394.astype('int32'), relay.shape_of(call_339))) # shape=(11, 14, 4)
bop_401 = relay.less_equal(bop_340.astype('bool'), uop_373.astype('bool')) # shape=(2, 16, 16)
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
var_405 = relay.var("var_405", dtype = "float32", shape = (27,))#candidate|405|(27,)|var|float32
call_404 = relay.TupleGetItem(func_66_call(relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(const_338.astype('float64'), [616,]), ), 0)
call_406 = relay.TupleGetItem(func_71_call(relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(const_338.astype('float64'), [616,]), ), 0)
uop_408 = relay.atan(uop_373.astype('float32')) # shape=(2, 16, 16)
bop_410 = relay.bitwise_or(uop_408.astype('int32'), bop_352.astype('int32')) # shape=(2, 16, 16)
uop_414 = relay.acosh(bop_340.astype('float32')) # shape=(2, 1, 16)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_417 = func_331_call()
call_418 = func_331_call()
uop_421 = relay.sin(bop_410.astype('float32')) # shape=(2, 16, 16)
bop_423 = relay.add(uop_421.astype('uint64'), uop_414.astype('uint64')) # shape=(2, 16, 16)
bop_429 = relay.bitwise_and(bop_423.astype('int32'), relay.reshape(uop_373.astype('int32'), relay.shape_of(bop_423))) # shape=(2, 16, 16)
uop_432 = relay.atanh(uop_408.astype('float64')) # shape=(2, 16, 16)
func_227_call = mod.get_global_var('func_227')
func_230_call = mutated_mod.get_global_var('func_230')
var_446 = relay.var("var_446", dtype = "float64", shape = (54,))#candidate|446|(54,)|var|float64
call_445 = func_227_call(relay.reshape(var_446.astype('float64'), [6, 9]))
call_447 = func_227_call(relay.reshape(var_446.astype('float64'), [6, 9]))
func_227_call = mod.get_global_var('func_227')
func_230_call = mutated_mod.get_global_var('func_230')
call_455 = func_227_call(relay.reshape(call_445.astype('float64'), [6, 9]))
call_456 = func_227_call(relay.reshape(call_445.astype('float64'), [6, 9]))
bop_457 = relay.greater_equal(bop_410.astype('bool'), uop_414.astype('bool')) # shape=(2, 16, 16)
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
call_463 = relay.TupleGetItem(func_66_call(relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(call_337.astype('float64'), [616,]), ), 2)
call_464 = relay.TupleGetItem(func_71_call(relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(var_405.astype('float32'), [3, 9]), relay.reshape(call_337.astype('float64'), [616,]), ), 2)
bop_468 = relay.logical_or(uop_432.astype('bool'), uop_334.astype('bool')) # shape=(2, 16, 16)
uop_471 = relay.acos(bop_457.astype('float32')) # shape=(2, 16, 16)
bop_483 = relay.not_equal(uop_471.astype('bool'), bop_340.astype('bool')) # shape=(2, 16, 16)
uop_486 = relay.tan(bop_401.astype('float64')) # shape=(2, 16, 16)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
call_493 = func_7_call(relay.reshape(call_337.astype('float64'), [11, 14, 4]))
call_494 = func_7_call(relay.reshape(call_337.astype('float64'), [11, 14, 4]))
output = relay.Tuple([const_338,call_375,bop_382,uop_385,bop_397,call_404,var_405,call_417,bop_429,call_445,var_446,call_455,call_463,bop_468,bop_483,uop_486,call_493,])
output2 = relay.Tuple([const_338,call_376,bop_382,uop_385,bop_400,call_406,var_405,call_418,bop_429,call_447,var_446,call_456,call_464,bop_468,bop_483,uop_486,call_494,])
func_495 = relay.Function([var_355,var_405,var_446,], output)
mod['func_495'] = func_495
mod = relay.transform.InferType()(mod)
mutated_mod['func_495'] = func_495
mutated_mod = relay.transform.InferType()(mutated_mod)
func_495_call = mutated_mod.get_global_var('func_495')
var_497 = relay.var("var_497", dtype = "int64", shape = (2, 16, 16))#candidate|497|(2, 16, 16)|var|int64
var_498 = relay.var("var_498", dtype = "float32", shape = (27,))#candidate|498|(27,)|var|float32
var_499 = relay.var("var_499", dtype = "float64", shape = (54,))#candidate|499|(54,)|var|float64
call_496 = func_495_call(var_497,var_498,var_499,)
output = call_496
func_500 = relay.Function([var_497,var_498,var_499,], output)
mutated_mod['func_500'] = func_500
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_504 = func_331_call()
call_505 = func_331_call()
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
var_509 = relay.var("var_509", dtype = "float32", shape = (27,))#candidate|509|(27,)|var|float32
var_510 = relay.var("var_510", dtype = "float64", shape = (616,))#candidate|510|(616,)|var|float64
call_508 = relay.TupleGetItem(func_66_call(relay.reshape(var_509.astype('float32'), [3, 9]), relay.reshape(var_509.astype('float32'), [3, 9]), relay.reshape(var_510.astype('float64'), [616,]), ), 0)
call_511 = relay.TupleGetItem(func_71_call(relay.reshape(var_509.astype('float32'), [3, 9]), relay.reshape(var_509.astype('float32'), [3, 9]), relay.reshape(var_510.astype('float64'), [616,]), ), 0)
uop_514 = relay.atan(call_504.astype('float64')) # shape=(11, 2)
uop_516 = relay.atan(call_505.astype('float64')) # shape=(11, 2)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
call_522 = func_7_call(relay.reshape(var_510.astype('float64'), [11, 14, 4]))
call_523 = func_7_call(relay.reshape(var_510.astype('float64'), [11, 14, 4]))
output = relay.Tuple([call_508,var_509,var_510,uop_514,call_522,])
output2 = relay.Tuple([call_511,var_509,var_510,uop_516,call_523,])
func_524 = relay.Function([var_509,var_510,], output)
mod['func_524'] = func_524
mod = relay.transform.InferType()(mod)
var_525 = relay.var("var_525", dtype = "float32", shape = (27,))#candidate|525|(27,)|var|float32
var_526 = relay.var("var_526", dtype = "float64", shape = (616,))#candidate|526|(616,)|var|float64
output = func_524(var_525,var_526,)
func_527 = relay.Function([var_525,var_526,], output)
mutated_mod['func_527'] = func_527
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_585 = func_331_call()
call_586 = func_331_call()
uop_590 = relay.cos(call_585.astype('float32')) # shape=(11, 2)
uop_592 = relay.cos(call_586.astype('float32')) # shape=(11, 2)
uop_593 = relay.erf(uop_590.astype('float32')) # shape=(11, 2)
uop_595 = relay.erf(uop_592.astype('float32')) # shape=(11, 2)
func_524_call = mod.get_global_var('func_524')
func_527_call = mutated_mod.get_global_var('func_527')
const_599 = relay.const([-7.793949,1.784859,7.284193,-7.346537,9.985721,3.386007,-4.635068,-7.800245,-8.425200,9.759021,6.879365,-4.717384,-7.569388,-4.027979,-2.325694,-2.959413,0.345300,-1.043412,4.678143,0.802171,5.229430,-2.351994,3.573190,-1.369733,8.420539,2.589241,-3.783059], dtype = "float32")#candidate|599|(27,)|const|float32
var_600 = relay.var("var_600", dtype = "float64", shape = (616,))#candidate|600|(616,)|var|float64
call_598 = relay.TupleGetItem(func_524_call(relay.reshape(const_599.astype('float32'), [27,]), relay.reshape(var_600.astype('float64'), [616,]), ), 1)
call_601 = relay.TupleGetItem(func_527_call(relay.reshape(const_599.astype('float32'), [27,]), relay.reshape(var_600.astype('float64'), [616,]), ), 1)
bop_603 = relay.floor_divide(uop_593.astype('float64'), relay.reshape(call_585.astype('float64'), relay.shape_of(uop_593))) # shape=(11, 2)
bop_606 = relay.floor_divide(uop_595.astype('float64'), relay.reshape(call_586.astype('float64'), relay.shape_of(uop_595))) # shape=(11, 2)
uop_607 = relay.atanh(bop_603.astype('float32')) # shape=(11, 2)
uop_609 = relay.atanh(bop_606.astype('float32')) # shape=(11, 2)
var_612 = relay.var("var_612", dtype = "float32", shape = (11, 2))#candidate|612|(11, 2)|var|float32
bop_613 = relay.logical_and(uop_593.astype('bool'), relay.reshape(var_612.astype('bool'), relay.shape_of(uop_593))) # shape=(11, 2)
bop_616 = relay.logical_and(uop_595.astype('bool'), relay.reshape(var_612.astype('bool'), relay.shape_of(uop_595))) # shape=(11, 2)
bop_617 = relay.bitwise_and(uop_607.astype('int16'), relay.reshape(bop_613.astype('int16'), relay.shape_of(uop_607))) # shape=(11, 2)
bop_620 = relay.bitwise_and(uop_609.astype('int16'), relay.reshape(bop_616.astype('int16'), relay.shape_of(uop_609))) # shape=(11, 2)
uop_623 = relay.log10(uop_607.astype('float64')) # shape=(11, 2)
uop_625 = relay.log10(uop_609.astype('float64')) # shape=(11, 2)
var_626 = relay.var("var_626", dtype = "float32", shape = (11, 2))#candidate|626|(11, 2)|var|float32
bop_627 = relay.equal(uop_607.astype('bool'), relay.reshape(var_626.astype('bool'), relay.shape_of(uop_607))) # shape=(11, 2)
bop_630 = relay.equal(uop_609.astype('bool'), relay.reshape(var_626.astype('bool'), relay.shape_of(uop_609))) # shape=(11, 2)
bop_633 = relay.right_shift(uop_623.astype('int8'), relay.reshape(uop_593.astype('int8'), relay.shape_of(uop_623))) # shape=(11, 2)
bop_636 = relay.right_shift(uop_625.astype('int8'), relay.reshape(uop_595.astype('int8'), relay.shape_of(uop_625))) # shape=(11, 2)
uop_639 = relay.rsqrt(uop_623.astype('float32')) # shape=(11, 2)
uop_641 = relay.rsqrt(uop_625.astype('float32')) # shape=(11, 2)
bop_642 = relay.mod(bop_633.astype('float64'), relay.reshape(uop_590.astype('float64'), relay.shape_of(bop_633))) # shape=(11, 2)
bop_645 = relay.mod(bop_636.astype('float64'), relay.reshape(uop_592.astype('float64'), relay.shape_of(bop_636))) # shape=(11, 2)
uop_646 = relay.sqrt(bop_633.astype('float32')) # shape=(11, 2)
uop_648 = relay.sqrt(bop_636.astype('float32')) # shape=(11, 2)
bop_650 = relay.maximum(uop_646.astype('float32'), relay.reshape(var_612.astype('float32'), relay.shape_of(uop_646))) # shape=(11, 2)
bop_653 = relay.maximum(uop_648.astype('float32'), relay.reshape(var_612.astype('float32'), relay.shape_of(uop_648))) # shape=(11, 2)
func_495_call = mod.get_global_var('func_495')
func_500_call = mutated_mod.get_global_var('func_500')
const_655 = relay.const([[-4],[-1],[3],[-5],[3],[1],[2],[-8],[6],[4],[7],[-5],[-9],[-3],[-10],[3],[-5],[7],[10],[-4],[-10],[9],[-2],[6],[9],[2],[-7],[-7],[5],[1],[8],[-9],[10],[-2],[10],[3],[10],[9],[1],[9],[-5],[-7],[8],[-1],[3],[-8],[7],[5],[-5],[-8],[-2],[4],[3],[3],[-1],[5],[5],[-10],[-6],[-8],[4],[2],[-5],[-9],[-4],[5],[5],[8],[-3],[5],[5],[-1],[4],[3],[-4],[-8],[7],[2],[-7],[-2],[5],[6],[5],[2],[-7],[3],[9],[-2],[5],[2],[8],[-8],[10],[-7],[-10],[-7],[-4],[-6],[9],[10],[-5],[7],[5],[6],[7],[-9],[-7],[-10],[-8],[3],[5],[4],[-3],[10],[-8],[-7],[3],[8],[5],[10],[-3],[-9],[-10],[7],[-9],[8],[10],[-3],[2],[-4],[-6],[-5],[7],[9],[-9],[-6],[-9],[-3],[7],[9],[-10],[-9],[-8],[-5],[10],[1],[2],[-6],[-9],[-1],[-5],[-1],[5],[5],[3],[7],[10],[-6],[-10],[1],[10],[-5],[9],[-1],[8],[9],[2],[-8],[-3],[8],[-5],[-3],[-3],[-5],[10],[1],[7],[-4],[9],[5],[7],[1],[-3],[6],[-1],[5],[-1],[3],[7],[-7],[4],[-9],[-7],[-7],[8],[4],[-5],[7],[-3],[7],[-9],[-2],[-1],[8],[-4],[7],[4],[1],[8],[-10],[10],[-4],[-8],[-7],[5],[-6],[4],[-9],[-6],[-3],[2],[8],[-5],[-9],[-7],[-1],[-2],[5],[5],[-2],[-4],[7],[8],[-5],[10],[5],[-9],[10],[-4],[-8],[6],[1],[4],[8],[8],[2],[-10],[-1],[2],[5],[3],[6],[4],[5],[2],[10],[-5],[-8],[7],[-10],[7],[6],[-2],[8],[-3],[8],[4],[8],[-9],[-8],[8],[-1],[-9],[4],[6],[-10],[10],[10],[4],[-6],[2],[5],[-9],[4],[9],[8],[-9],[2],[9],[-5],[-2],[8],[4],[-3],[-8],[7],[6],[-8],[6],[2],[-5],[3],[-3],[-4],[-3],[-7],[-6],[8],[-5],[-6],[4],[6],[5],[5],[-7],[7],[9],[-6],[4],[-7],[10],[3],[-1],[4],[4],[10],[5],[-5],[8],[-2],[-9],[7],[-8],[2],[-6],[1],[1],[10],[3],[-7],[-5],[-2],[-2],[1],[-9],[7],[3],[-5],[10],[4],[-6],[-9],[-7],[10],[-4],[8],[5],[10],[-10],[-7],[7],[6],[9],[-4],[1],[-4],[5],[4],[4],[-3],[-1],[10],[2],[1],[-2],[-1],[-4],[2],[10],[-4],[-5],[8],[2],[1],[5],[4],[6],[2],[-3],[-2],[-3],[7],[-1],[3],[-6],[4],[5],[-9],[-3],[-1],[-8],[-6],[9],[4],[1],[9],[8],[3],[5],[-1],[-5],[2],[7],[7],[4],[10],[-2],[-6],[-4],[-5],[4],[-9],[9],[2],[-10],[-7],[-7],[6],[-10],[6],[10],[-6],[4],[-5],[2],[-3],[5],[-9],[-6],[10],[-3],[2],[4],[4],[9],[-10],[5],[3],[-7],[8],[-10],[8],[-3],[-10],[7],[2],[7],[-9],[-1],[6],[2],[1],[1],[-7],[-4],[-8],[3],[2],[2],[-2],[3],[2],[-9],[-4],[-6],[10],[-6],[5],[7],[-8],[6],[7],[-3],[-5],[4],[7],[-5],[-9],[9],[-5],[-6],[3],[8],[-1],[-2],[7],[-6],[-7],[-5],[5],[2],[-7],[8],[1],[1],[-4],[-3],[-1],[7],[-7],[1],[-4]], dtype = "int64")#candidate|655|(512, 1)|const|int64
const_656 = relay.const([1.921663,-0.764448,-0.893636,0.971345,6.495995,8.747875,0.378189,-2.467039,-6.526940,2.746313,0.796646,-4.148767,-2.169612,7.506259,-8.224784,9.385278,5.324891,1.421282,9.846673,-0.805259,8.618204,-4.710011,-3.544784,-0.546469,-2.417616,7.266235,-6.827223,8.229834,-5.801842,-1.049526,-4.301280,-0.815916,3.597788,-0.438944,-0.875719,3.537462,-2.069975,-6.998342,-7.192180,8.871580,7.209025,-5.266055,6.127951,-5.117907,8.435924,-8.781659,-8.781078,5.681584,-3.502019,8.438704,7.218279,0.457070,-5.104835,-9.940778], dtype = "float64")#candidate|656|(54,)|const|float64
call_654 = relay.TupleGetItem(func_495_call(relay.reshape(const_655.astype('int64'), [2, 16, 16]), relay.reshape(call_598.astype('float32'), [27,]), relay.reshape(const_656.astype('float64'), [54,]), ), 8)
call_657 = relay.TupleGetItem(func_500_call(relay.reshape(const_655.astype('int64'), [2, 16, 16]), relay.reshape(call_598.astype('float32'), [27,]), relay.reshape(const_656.astype('float64'), [54,]), ), 8)
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
call_659 = relay.TupleGetItem(func_66_call(relay.reshape(const_599.astype('float32'), [3, 9]), relay.reshape(const_599.astype('float32'), [3, 9]), relay.reshape(var_600.astype('float64'), [616,]), ), 1)
call_660 = relay.TupleGetItem(func_71_call(relay.reshape(const_599.astype('float32'), [3, 9]), relay.reshape(const_599.astype('float32'), [3, 9]), relay.reshape(var_600.astype('float64'), [616,]), ), 1)
uop_667 = relay.sinh(bop_642.astype('float64')) # shape=(11, 2)
uop_669 = relay.sinh(bop_645.astype('float64')) # shape=(11, 2)
output = relay.Tuple([call_598,const_599,var_600,bop_617,bop_627,uop_639,bop_650,call_654,const_655,const_656,call_659,uop_667,])
output2 = relay.Tuple([call_601,const_599,var_600,bop_620,bop_630,uop_641,bop_653,call_657,const_655,const_656,call_660,uop_669,])
func_670 = relay.Function([var_600,var_612,var_626,], output)
mod['func_670'] = func_670
mod = relay.transform.InferType()(mod)
var_671 = relay.var("var_671", dtype = "float64", shape = (616,))#candidate|671|(616,)|var|float64
var_672 = relay.var("var_672", dtype = "float32", shape = (11, 2))#candidate|672|(11, 2)|var|float32
var_673 = relay.var("var_673", dtype = "float32", shape = (11, 2))#candidate|673|(11, 2)|var|float32
output = func_670(var_671,var_672,var_673,)
func_674 = relay.Function([var_671,var_672,var_673,], output)
mutated_mod['func_674'] = func_674
mutated_mod = relay.transform.InferType()(mutated_mod)
var_704 = relay.var("var_704", dtype = "float32", shape = (4, 1, 3))#candidate|704|(4, 1, 3)|var|float32
const_705 = relay.const([[[9.766756,-3.800084,5.429838],[1.006925,-2.402470,-9.491563],[-0.043379,3.541001,5.786311],[-0.193283,-3.948130,7.926977],[-8.500062,-6.453578,9.090281],[-0.068981,-8.516148,8.348701],[-7.251264,-2.965112,1.446804],[-2.762593,5.263101,-5.683015],[-0.014820,-9.441799,-8.419011],[-8.150858,2.496742,2.851335],[-5.622320,-4.557028,-2.774461],[-8.994917,-9.393085,7.495549],[-6.856745,3.414004,1.734540]],[[-9.878280,-3.862001,2.472341],[-9.819083,6.583524,-7.902844],[-9.297136,-0.527696,2.038010],[5.946116,-5.835255,-4.183108],[8.166543,1.350479,7.101802],[9.729915,-0.525279,-5.989343],[-9.167428,-7.906508,4.335115],[-6.109764,1.946440,-2.019072],[0.207690,7.394355,0.202216],[9.370298,-0.404295,-9.368085],[-1.768524,8.305650,0.459817],[-0.463583,-1.123943,-4.397636],[6.541455,2.028633,2.002107]],[[4.287614,2.585882,-6.851989],[0.171309,-7.377023,9.751114],[5.829585,-9.324102,-6.030600],[-1.094585,5.515397,9.794925],[-2.020223,5.561794,-5.018047],[4.010381,2.655704,-2.636553],[-4.974176,-1.946234,4.437849],[4.146304,1.283569,0.896836],[5.086310,-4.098175,-7.982748],[-0.116716,-0.790798,-3.774777],[0.995702,-2.140690,3.062887],[-5.341830,-4.855872,-9.630930],[-5.298393,-4.705241,2.323984]],[[-8.620622,-6.420289,1.756925],[-5.910230,5.152439,4.144091],[7.451415,-0.922078,-4.574734],[-5.561442,0.024966,9.250402],[-2.173509,1.289473,3.086342],[-0.522519,4.234262,-3.387089],[9.396173,1.361485,4.537031],[8.133454,7.680775,-1.787234],[4.102946,-3.927446,-2.048587],[-3.946489,-9.359306,0.626499],[-1.565972,7.192753,-1.770170],[-7.978402,-0.498979,4.771957],[0.623577,4.803793,-9.349455]]], dtype = "float32")#candidate|705|(4, 13, 3)|const|float32
bop_706 = relay.divide(var_704.astype('float32'), const_705.astype('float32')) # shape=(4, 13, 3)
uop_712 = relay.acosh(bop_706.astype('float32')) # shape=(4, 13, 3)
bop_715 = relay.subtract(uop_712.astype('int64'), relay.reshape(bop_706.astype('int64'), relay.shape_of(uop_712))) # shape=(4, 13, 3)
uop_724 = relay.sinh(uop_712.astype('float32')) # shape=(4, 13, 3)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
const_727 = relay.const([2.059477,-2.083299,-3.238646,-4.933570,5.882440,9.800465,9.513153,-9.976158,-4.198736,-5.655204,5.731303,0.665466,4.160900,-0.527303,-6.614064,9.698885,0.768709,5.192102,-3.497174,0.888382,9.214274,0.380012,-4.759251,-9.156410,-7.077670,-8.757983,-3.352966,2.018340,-9.819623,-9.156665,8.468444,3.132915,1.374266,-5.098424,-7.634119,-7.951526,-5.667335,0.150812,3.380070,2.260276,-9.011605,1.489094,6.195882,-5.492171,-6.693702,8.330323,-5.179483,6.365834,3.884736,6.962481,2.425980,1.758095,4.793003,-5.805928,-9.039780,9.670482,3.701441,-1.334007,5.950220,-0.905474,7.649626,5.952919,-4.657358,-1.980642,-9.453500,-7.744145,7.592844,-5.496026,-1.852543,3.584725,-5.963697,2.135646,5.949545,2.745697,5.581443,-2.644301,2.907842,-0.601509,-2.423014,3.161250,0.089204,-7.392106,-9.850334,4.752883,1.951137,2.002889,3.201393,9.101161,-7.466528,-4.729227,3.829860,5.039758,-4.463432,-6.352362,-1.594868,-9.998327,0.370945,-0.067991,3.488201,-3.988489,-3.839980,-4.895851,2.808475,-3.184046,-4.365468,6.569980,2.772273,-2.972821,-1.587939,3.415159,3.392768,-3.601558,-1.481554,-6.340514,8.138075,7.019515,-0.486286,1.490429,7.312601,-2.336937,-9.929568,3.162099,-1.904082,-1.287439,1.076445,4.190856,2.730452,3.146532,-8.046390,7.182235,1.143730,7.266796,-2.322860,1.272785,-3.968497,-1.158836,4.179035,-4.505438,8.901032,-3.399725,-9.113740,5.044820,-4.290206,-0.990154,-2.725943,-4.010296,-3.895441,-8.117034,-3.250614,-1.015043,5.213824,-3.272216,7.000227,-7.583934,3.008948,2.646562,-8.491783,4.818024,7.214868,6.223258,0.827695,0.020704,8.025236,-9.994504,3.201950,-8.295942,9.240125,8.068458,8.076202,-3.026153,6.745251,2.909721,-0.457978,-1.660594,-4.374962,-3.335719,3.331697,2.753794,1.135682,-0.749844,-3.532745,6.113938,8.814941,-6.058212,-3.086726,3.990190,-9.413987,5.087154,6.295282,8.246716,1.373820,3.865954,8.464356,3.040390,5.556224,-2.807460,-5.380583,0.155378,-4.250826,-8.163639,-9.194108,-4.438384,-9.979997,-5.890656,1.857086,1.145797,5.231762,5.043913,3.602342,2.281777,4.047604,8.793995,9.561490,9.153269,-9.583081,-2.014148,-0.810293,-6.060948,-8.315756,-9.518059,-1.904648,-4.080714,6.679336,-1.261614,-3.795164,1.737711,1.091117,4.567774,3.878777,6.858487,-3.275937,0.917191,-6.561251,-5.721495,-0.885052,0.521358,-4.394444,-4.831805,1.344629,7.659356,-8.991582,3.990924,6.996381,-3.607876,-9.226664,-9.537159,0.712727,2.781982,8.327508,-7.384729,-7.478904,-3.673995,8.065837,-8.909198,-5.225637,-6.626254,-7.297003,-2.805527,-0.203995,1.397923,-8.010185,3.241616,4.620000,-0.749735,-6.872332,-8.682549,-4.248910,-0.311807,1.183278,-1.261892,-2.446561,-8.905798,1.763668,-7.758852,7.457475,-3.852407,0.052773,4.161808,-8.905433,-2.967259,-4.002033,6.149392,1.429768,-2.271666,3.784857,-5.020648,7.692210,-1.733579,-4.418104,-4.880563,-1.834323,9.654668,-2.178670,-3.342763,-7.012739,-2.180361,0.393006,1.069989,9.961653,7.271073,6.219236,-5.840980,-7.562970,1.808753,8.564590,-7.921649,-5.178465,6.946219,-6.174358,6.650318,-2.197027,5.950991,7.104126,8.904953,-2.422263,3.342702,-9.742185,-0.650845,9.479505,1.286664,8.224394,-1.349551,-9.879408,8.135981,-4.053967,7.640078,-6.284062,-9.419859,-9.524930,-4.579162,9.147236,-9.714751,2.050633,-1.076226,-3.242329,9.649420,-9.093222,9.727959,0.917377,1.652980,9.021530,6.969978,-6.339257,8.774033,8.643615,5.183119,-7.646338,-8.526362,0.133720,-3.704798,0.835189,3.167132,2.521633,-9.947529,1.622528,-8.464674,-2.016990,-6.948865,-2.015630,4.817230,3.384426,-7.082561,-0.420896,9.490452,7.624471,5.700641,2.431667,-7.431328,5.375274,-6.550955,7.331979,6.931348,-6.134483,-4.572515,-2.581148,-5.344648,-7.578633,4.893836,8.444057,-9.170625,-9.654896,8.740005,7.990810,-6.178107,-2.785121,5.342356,-0.528547,-3.583376,8.551592,2.156005,-6.267310,7.980418,-4.643632,-8.224949,-5.167400,-1.165368,-5.416375,-3.014160,-0.582116,8.312222,1.234979,-3.577396,8.214316,-0.197892,2.347399,-3.961072,8.043127,-6.105683,-7.435008,-6.644874,-1.788513,0.622687,-5.607765,-2.020994,3.310240,-6.049184,-3.111835,-2.710391,1.886827,-4.294707,-3.495460,-6.788237,9.557633,3.870229,6.374308,-2.446312,-8.280799,7.352302,7.267232,2.097895,8.149630,-1.890107,-5.739543,6.507850,0.011144,-7.805864,4.065438,-3.713704,-5.196070,-3.893483,0.305720,6.874638,0.721901,-7.252510,-9.420333,7.295054,1.152309,7.411926,-7.600259,8.905119,5.585624,3.977979,5.842062,6.749092,7.789867,1.541001,-7.573566,-9.688775,1.457670,-5.721743,6.385872,1.030432,7.067466,-1.294799,-5.374669,-9.107016,-4.231545,-6.586958,5.394511,6.522837,2.398738,9.836540,3.844653,3.033324,-7.699009,-7.199066,4.257449,7.418634,-5.833500,-6.060094,4.017736,-2.464953,-6.668325,4.853393,-7.085172,-9.135595,-7.176192,8.560783,-3.092370,-2.877020,2.009398,-1.858912,2.642090,0.807822,0.838153,-4.859168,-1.277221,-8.883986,5.611948,-9.336183,-8.468284,9.169073,-4.323189,-8.526554,-3.927542,-8.547355,1.981995,-4.738798,-1.976438,0.190382,-5.268635,-4.135849,-6.503259,3.582131,5.347845,-1.251834,0.420302,-5.854137,-5.137099,-3.495122,8.655097,-6.687225,6.310642,8.758136,6.149168,8.350212,-4.942678,-4.535853,9.890292,-9.406289,-1.962679,-0.866958,5.432460,4.525714,-8.101722,-8.121506,-1.040447,-5.624636,1.641480,0.601476,-0.453935,-1.551969,1.999767,-5.187052,7.289734,0.461644,9.038416,9.122350,-2.305921,-8.235030,-6.558945,-5.543539,-0.884150,-2.923256,5.454037,-3.423567,-5.043787,-7.014707,4.612738,-6.149846,-3.292942,-4.517710,-0.340907,-5.064144,6.635964,1.544063,7.602776,-3.859482,-0.641416,2.549970,0.277427,-8.392521,7.900358,5.845751,2.867575,4.252326,3.018783,0.380885,9.638414,-1.532869,-9.632123,-8.822376,4.438890,2.941856,6.180429,4.763503,-4.684523,-8.257652,7.786221,4.598740,-5.316088,-5.777624,-8.481766,8.955034,0.702733,-5.525752,7.486317,2.467818,-0.734735,-2.532462,0.923187,-0.540499,3.639596,4.883535,-6.060716,8.708943,4.991537,-3.729789,1.662230,7.980244,9.185518,2.109193,-8.690686,6.550833,5.118426,-4.503784], dtype = "float64")#candidate|727|(616,)|const|float64
call_726 = func_7_call(relay.reshape(const_727.astype('float64'), [11, 14, 4]))
call_728 = func_7_call(relay.reshape(const_727.astype('float64'), [11, 14, 4]))
bop_729 = relay.bitwise_or(uop_724.astype('int32'), relay.reshape(bop_706.astype('int32'), relay.shape_of(uop_724))) # shape=(4, 13, 3)
bop_734 = relay.greater(bop_706.astype('bool'), relay.reshape(const_705.astype('bool'), relay.shape_of(bop_706))) # shape=(4, 13, 3)
output = relay.Tuple([bop_715,call_726,const_727,bop_729,bop_734,])
output2 = relay.Tuple([bop_715,call_728,const_727,bop_729,bop_734,])
func_737 = relay.Function([var_704,], output)
mod['func_737'] = func_737
mod = relay.transform.InferType()(mod)
var_738 = relay.var("var_738", dtype = "float32", shape = (4, 1, 3))#candidate|738|(4, 1, 3)|var|float32
output = func_737(var_738)
func_739 = relay.Function([var_738], output)
mutated_mod['func_739'] = func_739
mutated_mod = relay.transform.InferType()(mutated_mod)
var_741 = relay.var("var_741", dtype = "uint16", shape = ())#candidate|741|()|var|uint16
const_742 = relay.const([[-2,-1],[5,4]], dtype = "uint16")#candidate|742|(2, 2)|const|uint16
bop_743 = relay.bitwise_and(var_741.astype('uint16'), const_742.astype('uint16')) # shape=(2, 2)
bop_748 = relay.divide(const_742.astype('float64'), relay.reshape(bop_743.astype('float64'), relay.shape_of(const_742))) # shape=(2, 2)
bop_754 = relay.less(const_742.astype('bool'), relay.reshape(bop_748.astype('bool'), relay.shape_of(const_742))) # shape=(2, 2)
var_761 = relay.var("var_761", dtype = "bool", shape = (2, 2))#candidate|761|(2, 2)|var|bool
bop_762 = relay.bitwise_and(bop_754.astype('uint64'), relay.reshape(var_761.astype('uint64'), relay.shape_of(bop_754))) # shape=(2, 2)
const_766 = relay.const([[True,False],[True,False]], dtype = "bool")#candidate|766|(2, 2)|const|bool
bop_767 = relay.right_shift(bop_754.astype('int64'), relay.reshape(const_766.astype('int64'), relay.shape_of(bop_754))) # shape=(2, 2)
bop_770 = relay.power(bop_748.astype('float64'), relay.reshape(bop_767.astype('float64'), relay.shape_of(bop_748))) # shape=(2, 2)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_773 = func_331_call()
call_774 = func_331_call()
func_670_call = mod.get_global_var('func_670')
func_674_call = mutated_mod.get_global_var('func_674')
var_777 = relay.var("var_777", dtype = "float64", shape = (616,))#candidate|777|(616,)|var|float64
call_776 = relay.TupleGetItem(func_670_call(relay.reshape(var_777.astype('float64'), [616,]), relay.reshape(call_773.astype('float32'), [11, 2]), relay.reshape(call_773.astype('float32'), [11, 2]), ), 5)
call_778 = relay.TupleGetItem(func_674_call(relay.reshape(var_777.astype('float64'), [616,]), relay.reshape(call_773.astype('float32'), [11, 2]), relay.reshape(call_773.astype('float32'), [11, 2]), ), 5)
uop_781 = relay.exp(bop_743.astype('float64')) # shape=(2, 2)
uop_785 = relay.atanh(uop_781.astype('float32')) # shape=(2, 2)
uop_788 = relay.tan(uop_785.astype('float32')) # shape=(2, 2)
bop_795 = relay.subtract(uop_785.astype('uint16'), relay.reshape(const_742.astype('uint16'), relay.shape_of(uop_785))) # shape=(2, 2)
bop_798 = relay.logical_and(uop_781.astype('bool'), relay.reshape(bop_743.astype('bool'), relay.shape_of(uop_781))) # shape=(2, 2)
bop_801 = relay.less_equal(bop_795.astype('bool'), relay.reshape(uop_781.astype('bool'), relay.shape_of(bop_795))) # shape=(2, 2)
uop_810 = relay.sin(uop_785.astype('float64')) # shape=(2, 2)
output = relay.Tuple([bop_762,bop_770,call_773,call_776,var_777,uop_788,bop_798,bop_801,uop_810,])
output2 = relay.Tuple([bop_762,bop_770,call_774,call_778,var_777,uop_788,bop_798,bop_801,uop_810,])
func_814 = relay.Function([var_741,var_761,var_777,], output)
mod['func_814'] = func_814
mod = relay.transform.InferType()(mod)
var_815 = relay.var("var_815", dtype = "uint16", shape = ())#candidate|815|()|var|uint16
var_816 = relay.var("var_816", dtype = "bool", shape = (2, 2))#candidate|816|(2, 2)|var|bool
var_817 = relay.var("var_817", dtype = "float64", shape = (616,))#candidate|817|(616,)|var|float64
output = func_814(var_815,var_816,var_817,)
func_818 = relay.Function([var_815,var_816,var_817,], output)
mutated_mod['func_818'] = func_818
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_854 = func_331_call()
call_855 = func_331_call()
output = relay.Tuple([call_854,])
output2 = relay.Tuple([call_855,])
func_865 = relay.Function([], output)
mod['func_865'] = func_865
mod = relay.transform.InferType()(mod)
output = func_865()
func_866 = relay.Function([], output)
mutated_mod['func_866'] = func_866
mutated_mod = relay.transform.InferType()(mutated_mod)
var_911 = relay.var("var_911", dtype = "int8", shape = (11, 6))#candidate|911|(11, 6)|var|int8
const_912 = relay.const([[8,5,1,3,-7,-2],[6,-10,-10,-3,-5,7],[6,-5,-1,-10,5,7],[5,-3,4,-5,-9,-5],[9,8,-2,8,-3,1],[-2,-4,-4,2,-3,-6],[-10,9,6,-8,9,1],[-1,9,1,-7,-1,-7],[-6,-4,6,5,-1,-5],[8,3,9,1,10,3],[2,-4,3,-5,2,5]], dtype = "int8")#candidate|912|(11, 6)|const|int8
bop_913 = relay.multiply(var_911.astype('int8'), relay.reshape(const_912.astype('int8'), relay.shape_of(var_911))) # shape=(11, 6)
var_916 = relay.var("var_916", dtype = "int8", shape = (11, 6))#candidate|916|(11, 6)|var|int8
bop_917 = relay.minimum(bop_913.astype('int32'), relay.reshape(var_916.astype('int32'), relay.shape_of(bop_913))) # shape=(11, 6)
bop_925 = relay.bitwise_and(var_916.astype('uint8'), relay.reshape(bop_913.astype('uint8'), relay.shape_of(var_916))) # shape=(11, 6)
uop_931 = relay.acosh(const_912.astype('float32')) # shape=(11, 6)
bop_935 = relay.bitwise_xor(bop_917.astype('uint8'), relay.reshape(bop_913.astype('uint8'), relay.shape_of(bop_917))) # shape=(11, 6)
var_943 = relay.var("var_943", dtype = "float32", shape = (11, 6))#candidate|943|(11, 6)|var|float32
bop_944 = relay.less(uop_931.astype('bool'), relay.reshape(var_943.astype('bool'), relay.shape_of(uop_931))) # shape=(11, 6)
output = relay.Tuple([bop_925,bop_935,bop_944,])
output2 = relay.Tuple([bop_925,bop_935,bop_944,])
func_947 = relay.Function([var_911,var_916,var_943,], output)
mod['func_947'] = func_947
mod = relay.transform.InferType()(mod)
mutated_mod['func_947'] = func_947
mutated_mod = relay.transform.InferType()(mutated_mod)
func_947_call = mutated_mod.get_global_var('func_947')
var_949 = relay.var("var_949", dtype = "int8", shape = (11, 6))#candidate|949|(11, 6)|var|int8
var_950 = relay.var("var_950", dtype = "int8", shape = (11, 6))#candidate|950|(11, 6)|var|int8
var_951 = relay.var("var_951", dtype = "float32", shape = (11, 6))#candidate|951|(11, 6)|var|float32
call_948 = func_947_call(var_949,var_950,var_951,)
output = call_948
func_952 = relay.Function([var_949,var_950,var_951,], output)
mutated_mod['func_952'] = func_952
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_983 = relay.TupleGetItem(func_865_call(), 0)
call_984 = relay.TupleGetItem(func_866_call(), 0)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
const_1003 = relay.const([[4.561160],[-6.819537],[-4.502673],[-6.265051],[8.873908],[4.792069],[4.091568],[0.615253],[5.523926],[6.204261],[-6.820786],[9.852152],[5.836753],[7.149037],[2.642713],[-4.939863],[5.092076],[5.879354],[4.839590],[-8.953915],[3.314097],[2.694560],[0.637838],[6.280979],[0.906538],[0.953259],[-2.826920],[-1.454089],[6.485220],[-8.767549],[6.115331],[1.781494],[8.173632],[-6.017945],[-6.402480],[3.237103],[-4.646226],[-6.875782],[4.346562],[-6.385132],[-9.242970],[2.096546],[6.219645],[-8.914459],[6.620680],[6.929309],[-8.763455],[-0.013004],[8.089890],[-3.714590],[9.011144],[-4.428804],[-4.401986],[-9.585815],[-4.153121],[-4.915125],[-4.510254],[-9.292290],[0.994988],[-7.100732],[9.409042],[2.399665],[-8.894946],[2.737761],[-2.090083],[2.469623],[-1.049135],[7.526081],[9.636119],[-7.348849],[3.236442],[-0.745136],[-9.022958],[8.960383],[-8.145330],[3.627969],[4.754818],[2.965531],[-0.654381],[5.703413],[-3.074895],[-7.941511],[-4.910233],[3.635374],[6.209979],[-4.426204],[-5.784625],[-1.007843],[1.696543],[-9.913622],[5.714996],[1.276647],[-2.988233],[-1.068768],[3.906769],[3.252187],[7.213874],[-4.961494],[3.836681],[6.995937],[-5.695253],[1.762008],[-2.899320],[1.267841],[4.883595],[1.287596],[2.442116],[1.817706],[7.354900],[-8.300108],[-9.083639],[-9.953108],[-4.676039],[6.187974],[-4.498919],[-9.031570],[-9.554798],[3.402791],[-1.724159],[-8.078783],[-1.214481],[-1.643678],[-7.804157],[6.227132],[-8.747897],[9.964764],[5.982111],[-5.709769],[-3.976321],[-6.969909],[-1.066262],[6.233950],[-6.524171],[3.731064],[-4.319788],[2.173255],[-4.537488],[-1.360349],[7.818745],[-1.292622],[-6.134302],[-8.206527],[-8.901239],[5.581578],[8.974075],[1.788055],[-2.796310],[-1.138094],[-2.566137],[6.598849],[-3.979577],[8.660164],[8.211380],[1.574085],[-0.494358],[-2.179481],[3.389086],[-5.182020],[5.422137],[3.150928],[9.421567],[-1.898964],[6.437316],[8.463962],[-8.181212],[-7.782540],[6.981842],[6.239670],[-3.773447],[8.335804],[9.383656],[5.751329],[7.114018],[-4.011287],[1.688372],[2.682660],[-1.591255],[-7.735436],[-7.389608],[5.029519],[6.182501],[8.919941],[-0.138343],[7.931268],[-8.691975],[7.282829],[-8.902025],[-0.204249],[-2.565054],[-1.689311],[-5.606865],[9.392707],[3.197198],[5.298379],[8.267372],[-5.188916],[-6.474467],[4.400398],[2.796972],[5.968179],[2.572411],[3.236922],[8.130134],[5.975321],[-4.846615],[1.129941],[9.381511],[9.283163],[6.236053],[7.039997],[-0.520189],[4.533030],[3.988104],[8.648397],[4.410868],[-1.730702],[1.659966],[-3.183176],[-0.275673],[-9.116717],[9.727901],[3.475722],[-4.157286],[2.224949],[9.625837],[-2.236133],[-9.287774],[2.302087],[-0.758141],[-8.274907],[2.455908],[7.089888],[-7.384271],[-5.053699],[8.679681],[1.532853],[-6.344652],[-0.330911],[-5.305895],[-9.415924],[-0.464921],[1.324961],[-2.950926],[-9.906237],[4.000391],[-3.943219],[-2.250196],[9.695190],[5.543758],[0.940468],[-1.731969],[9.847944],[0.899692],[-2.376112],[3.745959],[-3.157441],[1.079751],[9.638134],[-7.586523],[2.444657],[-8.985731],[-5.028112],[4.762295],[-8.908125],[8.759828],[9.969461],[-7.186525],[5.801353],[-0.827827],[2.949400],[-8.280687],[8.850924],[6.691113],[-7.039857],[7.156798],[1.764516],[8.195899],[-2.645353],[1.344682],[-5.996869],[-0.029912],[9.133778],[-1.070287],[-3.926185],[-5.513184],[6.288557],[0.320172],[-4.837874],[9.120236],[-5.755152],[-6.562779],[-2.723570],[-2.301583],[-7.825123],[2.783684],[-7.397154],[9.606560],[2.804067],[-0.481649],[-9.971708],[9.964314],[-2.737418],[1.443427],[-5.947245],[9.721243],[2.964082],[-0.318889],[5.685424],[-8.597424],[5.089336],[-1.625944],[8.296674],[8.073889],[-3.885838],[-6.882227],[-3.986355],[-3.048568],[0.008034],[1.818718],[-9.551540],[-8.001392],[9.317342],[9.986827],[3.301419],[-8.837951],[-4.420575],[3.274421],[-4.078234],[-1.820048],[-1.688766],[-5.023435],[-7.189020],[2.065812],[-2.449658],[3.222990],[1.006345],[-6.483026],[2.919272],[-7.133671],[7.093050],[3.503419],[6.526328],[1.560262],[0.078262],[-8.855308],[-5.733019],[-6.422458],[5.444830],[-3.270831],[-1.936142],[-7.276899],[-3.485082],[-3.260544],[6.519544],[-3.120242],[-4.604837],[-3.667328],[-2.778934],[5.107106],[7.828889],[-9.560616],[-4.052523],[-7.277628],[-9.094331],[2.153223],[3.429203],[9.137523],[-9.235844],[9.936011],[-7.625195],[-3.700044],[3.565096],[-5.060832],[-1.522797],[0.798361],[-3.348332],[-5.184994],[-9.728019],[-5.685665],[5.239274],[0.487204],[-6.510156],[-5.752845],[3.837696],[-2.632518],[-6.362366],[-1.707996],[2.261654],[2.027904],[-5.561574],[3.838042],[-4.089921],[-1.560902],[-6.467598],[-6.928870],[-9.999973],[-7.724134],[-8.859281],[-0.525598],[-5.462600],[7.681289],[5.207913],[-2.394328],[-4.996049],[6.618644],[-4.501517],[4.825690],[-7.376002],[4.559252],[1.429330],[-9.416774],[-8.989309],[-1.370304],[-6.461976],[-1.181298],[9.785652],[8.085471],[8.990288],[3.120920],[2.851391],[-9.341794],[6.615272],[1.364698],[9.174133],[-5.463769],[-7.908592],[0.856776],[-6.760808],[9.205418],[-8.525969],[9.455419],[5.677337],[-5.045484],[8.417305],[-9.897866],[-0.196322],[-6.866655],[0.686139],[8.873243],[7.819145],[5.028519],[0.220743],[-5.802592],[-7.874934],[-7.483930],[1.611137],[-9.672020],[-0.245992],[4.756113],[-2.450098],[-2.260510],[-5.386540],[-8.363089],[-9.681940],[2.531605],[5.508056],[6.305126],[6.475144],[-4.000410],[5.647245],[-6.735335],[0.242139],[1.655055],[8.424164],[5.561088],[4.066957],[6.981050],[-5.734509],[2.522598],[6.779994],[-4.967518],[9.970253],[1.135852],[6.645962],[-2.052443],[-2.684842],[7.437536],[0.175807],[2.454239],[0.614033],[-8.601549],[9.534033],[6.005598],[1.048889],[-5.482531],[7.606633],[5.478907],[7.110990],[1.940520],[6.668031],[4.740908],[-2.663483],[5.803538],[-1.127326],[-0.115041],[0.136504],[-4.435394],[-8.464683],[-9.179505],[-6.213777],[0.674082],[-5.757501],[-3.828911],[7.951012],[-9.048770],[1.009855],[9.441561],[5.400227],[8.962499],[4.245555],[9.141546],[-6.056525],[4.307136],[9.264174],[-9.094861],[6.155475],[6.207685],[-4.509292],[9.635775],[8.315405],[-0.368376],[6.387504],[-5.175893],[-7.697508],[-7.450798],[-6.207035],[-0.208815],[0.249447],[9.085690],[-0.886755],[-4.398825],[3.179771],[4.197562],[3.584350],[-1.564385],[-0.358186],[9.644332],[-9.411668],[-1.617447],[7.658732],[-0.844284],[6.685702],[-1.687881],[9.909218],[2.779315],[9.717892],[4.081063],[-8.306423],[0.545033],[3.958297],[-3.896190],[-4.796619],[7.330975],[-6.376712],[-4.435279],[4.076545],[1.604308],[0.692664],[4.195634],[9.974597],[-6.191415],[1.906702],[-0.377577],[-1.788215],[9.655128],[7.650928],[-4.806305],[-4.569566],[-9.330919],[7.041916],[-7.013872],[9.650640],[2.868298],[-4.870067],[9.150213],[-2.772929],[-6.424489],[5.388331],[5.233992],[-0.535073],[1.592480],[-6.691853],[-8.066568],[-3.511578],[8.494392],[3.271274],[-5.651881],[1.711353],[4.717084],[-6.585659],[-6.472723],[5.469014],[-3.954205],[7.460464],[-2.354303],[1.340581],[4.649486],[1.102201],[-5.883734],[-1.158969],[-3.303340],[8.230218],[-7.777896],[-4.148440],[-5.186754],[6.082613],[6.909274],[-4.438488],[2.354850],[6.697986],[2.473136],[-7.804418],[5.450623],[-0.223001],[9.663239],[0.482115]], dtype = "float64")#candidate|1003|(616, 1)|const|float64
call_1002 = func_7_call(relay.reshape(const_1003.astype('float64'), [11, 14, 4]))
call_1004 = func_7_call(relay.reshape(const_1003.astype('float64'), [11, 14, 4]))
output = relay.Tuple([call_983,call_1002,const_1003,])
output2 = relay.Tuple([call_984,call_1004,const_1003,])
func_1025 = relay.Function([], output)
mod['func_1025'] = func_1025
mod = relay.transform.InferType()(mod)
output = func_1025()
func_1026 = relay.Function([], output)
mutated_mod['func_1026'] = func_1026
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1042 = relay.var("var_1042", dtype = "int16", shape = (2,))#candidate|1042|(2,)|var|int16
var_1043 = relay.var("var_1043", dtype = "int16", shape = (2,))#candidate|1043|(2,)|var|int16
bop_1044 = relay.multiply(var_1042.astype('int16'), relay.reshape(var_1043.astype('int16'), relay.shape_of(var_1042))) # shape=(2,)
func_814_call = mod.get_global_var('func_814')
func_818_call = mutated_mod.get_global_var('func_818')
const_1048 = relay.const(-1, dtype = "uint16")#candidate|1048|()|const|uint16
const_1049 = relay.const([False,True,True,True], dtype = "bool")#candidate|1049|(4,)|const|bool
var_1050 = relay.var("var_1050", dtype = "float64", shape = (14, 44))#candidate|1050|(14, 44)|var|float64
call_1047 = relay.TupleGetItem(func_814_call(relay.reshape(const_1048.astype('uint16'), []), relay.reshape(const_1049.astype('bool'), [2, 2]), relay.reshape(var_1050.astype('float64'), [616,]), ), 5)
call_1051 = relay.TupleGetItem(func_818_call(relay.reshape(const_1048.astype('uint16'), []), relay.reshape(const_1049.astype('bool'), [2, 2]), relay.reshape(var_1050.astype('float64'), [616,]), ), 5)
func_814_call = mod.get_global_var('func_814')
func_818_call = mutated_mod.get_global_var('func_818')
call_1052 = relay.TupleGetItem(func_814_call(relay.reshape(const_1048.astype('uint16'), []), relay.reshape(const_1049.astype('bool'), [2, 2]), relay.reshape(var_1050.astype('float64'), [616,]), ), 3)
call_1053 = relay.TupleGetItem(func_818_call(relay.reshape(const_1048.astype('uint16'), []), relay.reshape(const_1049.astype('bool'), [2, 2]), relay.reshape(var_1050.astype('float64'), [616,]), ), 3)
output = relay.Tuple([bop_1044,call_1047,const_1048,const_1049,var_1050,call_1052,])
output2 = relay.Tuple([bop_1044,call_1051,const_1048,const_1049,var_1050,call_1053,])
func_1054 = relay.Function([var_1042,var_1043,var_1050,], output)
mod['func_1054'] = func_1054
mod = relay.transform.InferType()(mod)
var_1055 = relay.var("var_1055", dtype = "int16", shape = (2,))#candidate|1055|(2,)|var|int16
var_1056 = relay.var("var_1056", dtype = "int16", shape = (2,))#candidate|1056|(2,)|var|int16
var_1057 = relay.var("var_1057", dtype = "float64", shape = (14, 44))#candidate|1057|(14, 44)|var|float64
output = func_1054(var_1055,var_1056,var_1057,)
func_1058 = relay.Function([var_1055,var_1056,var_1057,], output)
mutated_mod['func_1058'] = func_1058
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1060 = relay.var("var_1060", dtype = "int64", shape = (1,))#candidate|1060|(1,)|var|int64
var_1061 = relay.var("var_1061", dtype = "int64", shape = (16,))#candidate|1061|(16,)|var|int64
bop_1062 = relay.equal(var_1060.astype('bool'), var_1061.astype('bool')) # shape=(16,)
bop_1068 = relay.logical_and(bop_1062.astype('bool'), var_1060.astype('bool')) # shape=(16,)
func_524_call = mod.get_global_var('func_524')
func_527_call = mutated_mod.get_global_var('func_527')
var_1083 = relay.var("var_1083", dtype = "float32", shape = (27,))#candidate|1083|(27,)|var|float32
const_1084 = relay.const([0.716212,-9.459722,5.706422,-7.015190,-8.421111,6.710282,2.423915,5.964897,-9.069950,5.261944,-5.190865,1.391300,9.109604,-6.873484,7.121494,5.471173,8.419022,0.319569,-3.759261,-0.743558,0.134641,-7.546917,0.169066,3.644641,7.319709,1.483939,1.991900,4.412554,-9.604637,1.714660,-1.709029,-9.707034,9.501190,-9.859951,4.470571,-6.986270,5.590498,3.058291,-2.917636,-2.284168,8.656075,6.567958,-3.365265,0.053396,-3.449009,7.527113,3.054166,5.984836,2.679969,-9.269081,-1.875852,7.884650,-1.553574,-2.846986,1.612429,-2.786018,4.334499,1.582015,-0.423974,-6.295133,-2.623428,1.512429,5.765981,-5.147807,3.561377,0.462799,7.373951,5.356947,1.436191,-0.583186,-8.988688,4.936664,1.547716,4.324721,-0.551969,-8.888062,1.105260,-2.477798,-8.100882,-4.310965,-9.229520,-9.882472,-3.485091,9.533223,1.646761,0.663253,-3.341396,1.908293,9.828017,-2.980742,-6.182041,-7.190741,-9.347090,-2.737004,1.310138,-6.337726,-4.246161,9.258917,-1.569292,-0.935271,7.678573,8.067281,1.475605,5.885724,-7.186096,-0.923847,-7.377720,-2.573085,4.843180,-7.912281,2.136661,-0.983593,4.818917,6.466302,-3.027042,8.959120,0.705476,-2.837661,-3.519115,1.255960,6.037150,6.660956,0.456528,-9.674961,2.669444,5.725088,-1.957506,-1.618899,-2.866453,4.578247,-6.736578,-0.815643,-1.391591,-7.113464,-2.794322,9.937195,1.462204,9.227037,-2.828473,1.815231,-9.045151,0.697539,9.331758,-8.213584,6.379664,-2.458767,9.732450,0.201252,5.785338,-3.699865,6.792275,9.905838,-4.131534,3.392726,3.951641,-5.442424,-0.359286,-3.295716,-6.257729,8.084868,-8.599934,1.944030,-9.592268,7.463447,-4.502555,7.024658,6.513534,-6.757313,1.697663,6.485540,-5.425897,3.103474,9.836378,-8.368832,-0.061105,8.957172,-6.401480,-4.477618,3.375866,-2.006274,0.293787,7.164299,2.953266,9.288563,-3.652981,4.461853,-1.991935,9.676268,-4.339871,6.955782,0.257796,0.829034,4.015854,-5.039529,-9.932076,-5.916634,-9.169649,-4.671788,8.938058,-2.276489,-4.366119,5.122496,-8.403945,-8.877081,-9.368566,-5.203417,-2.496817,8.327897,6.766459,-5.076635,3.321694,-1.648594,3.227817,7.098564,-7.793174,-5.629733,2.867573,7.311597,-3.153626,-4.128074,-2.489869,-5.175458,-2.338063,-3.475687,6.126930,9.672614,-1.085273,-6.472452,3.240233,-1.162453,3.452691,-0.147483,-6.715034,8.312923,-5.812421,-6.075652,-7.927567,8.527396,-8.503108,-5.299134,9.406594,-2.127064,5.981602,9.579436,-6.042909,-0.475403,-7.842279,-7.047094,2.566675,9.416431,5.643700,-4.760317,-6.810802,1.628660,3.422953,-4.929336,1.088140,-2.554076,5.382696,8.540716,-7.221801,7.373066,6.238338,0.888291,5.878830,-9.062274,-6.387898,-2.714253,-1.362358,-5.524816,-2.875097,-5.035342,9.742655,-6.975682,-4.640743,5.355574,-4.556751,4.436070,-3.587724,-1.596966,9.136749,9.250358,8.714187,-7.253520,3.969361,7.397524,4.503833,0.809205,7.282898,3.704969,-2.483691,-4.533059,-4.385538,-0.334064,-1.456989,-2.190716,5.913949,-7.156458,-1.387185,-1.358723,7.857472,7.389361,-7.123130,4.843229,-9.419966,8.876590,0.187178,9.345283,-8.022213,5.945032,-1.122234,-1.446499,1.906804,-3.677028,-9.865036,9.972655,2.137628,-5.909227,9.500216,3.441207,9.075060,-5.583919,-0.829395,-5.149956,2.570852,5.261728,5.802629,1.643193,-2.211431,-9.844712,8.130621,-6.130324,-2.016428,-5.670891,0.787527,-3.846407,-3.112195,8.755361,-3.696593,1.386202,0.846056,-9.973852,1.106279,3.205482,0.625581,-0.141422,5.616554,7.719261,7.709510,-1.516811,5.880594,5.559355,7.316963,-7.890199,-5.958585,-1.739188,8.634577,4.081779,-1.004152,-4.981413,-2.133792,-4.065522,-0.158035,-4.577225,1.481729,-9.679913,6.038545,5.383096,-6.633395,-7.522092,8.758565,9.809998,3.088649,8.920087,5.938761,-9.181824,-3.954171,-4.342702,-5.903959,5.904936,6.743434,9.704092,2.660089,5.169911,-2.442170,4.115848,-8.326679,-0.737783,3.782151,-1.426697,-6.182029,-9.670698,7.806196,0.487814,4.456528,-7.990328,-1.338349,6.861118,-5.810491,8.301420,9.259673,-5.854014,1.160953,6.271811,3.420686,-7.786845,-5.700758,-3.361328,1.707222,-4.469807,-0.252666,5.849108,-0.480195,9.990508,3.882527,4.691368,9.422752,9.322078,1.063064,4.670975,-5.364684,-0.106293,-2.487488,-8.096988,-4.203138,8.226290,4.296261,5.206164,3.955602,-3.835053,8.443117,4.791106,-5.325369,-2.934772,-7.715897,-5.899894,0.107842,3.831424,7.006763,5.271732,-8.236543,9.946559,8.249394,7.912258,3.866264,6.950598,-6.968945,1.433772,7.409881,-2.759675,7.400942,-9.491884,5.803437,-5.613809,-3.979762,-8.465858,-4.353340,-3.048568,-6.967327,2.583016,-4.106613,-8.417971,8.869078,7.217579,1.286658,-6.259011,-3.455160,9.728368,0.357141,-7.769625,3.057240,-8.810308,-9.234810,9.049681,6.259666,-9.167746,-4.845494,-8.419794,-5.103771,-8.050738,2.455971,-1.766516,-1.607209,1.643461,-8.652115,9.608353,-7.846462,3.235263,-8.390792,2.552812,4.674511,5.894065,-9.346459,-6.889105,7.473277,3.519156,2.775902,-1.118296,6.463048,6.036574,-4.201848,-0.435194,1.325137,2.072848,-8.376063,1.581054,4.676881,7.678869,-4.116736,-0.552456,8.555284,-7.504955,-0.571642,-3.003432,-9.665379,6.675581,-9.933906,6.142062,2.734941,1.866978,0.472083,-6.580054,-9.292731,-7.531538,7.116825,-6.306625,-8.602865,8.024652,-4.760940,5.754006,-2.741077,0.874885,-8.773090,-0.746495,9.724870,4.797807,-7.192789,8.776864,-7.704961,9.572116,3.241649,-9.918461,-6.987050,-2.408337,7.650867,4.252053,-9.601135,-3.998586,7.652436,-9.345905,-2.746677,0.390563,0.147732,7.452903,-2.753292,0.192930,4.253529,-3.160731,-8.163446,-3.062410,-3.069646,9.345732,-4.387225,5.723520,-6.887243,-8.443093,7.442456,5.524471,0.585527,-5.473430,3.625199,9.225012,4.034546,-4.447627,-6.053546,9.430067,-9.791126,0.066376,-7.506271,-9.078789,-2.945391,-7.821788,-6.517932,2.651505,-8.447355,5.635071,9.165907,-8.798800,-4.512217,-4.694486,9.954098,1.829035,-7.075099,-5.052678,7.591939,1.037137,-9.532053,-8.784852,4.183361,-8.160173,7.350525,3.287058,-7.609859,-7.592048,-7.722047,9.784005,-8.332417,2.415525,8.931308,-7.346577,-4.859633,-6.982734,6.792299,3.717143,-2.507969,-7.820808], dtype = "float64")#candidate|1084|(616,)|const|float64
call_1082 = relay.TupleGetItem(func_524_call(relay.reshape(var_1083.astype('float32'), [27,]), relay.reshape(const_1084.astype('float64'), [616,]), ), 2)
call_1085 = relay.TupleGetItem(func_527_call(relay.reshape(var_1083.astype('float32'), [27,]), relay.reshape(const_1084.astype('float64'), [616,]), ), 2)
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
call_1087 = relay.TupleGetItem(func_66_call(relay.reshape(var_1083.astype('float32'), [3, 9]), relay.reshape(var_1083.astype('float32'), [3, 9]), relay.reshape(const_1084.astype('float64'), [616,]), ), 1)
call_1088 = relay.TupleGetItem(func_71_call(relay.reshape(var_1083.astype('float32'), [3, 9]), relay.reshape(var_1083.astype('float32'), [3, 9]), relay.reshape(const_1084.astype('float64'), [616,]), ), 1)
func_737_call = mod.get_global_var('func_737')
func_739_call = mutated_mod.get_global_var('func_739')
var_1093 = relay.var("var_1093", dtype = "float32", shape = (12,))#candidate|1093|(12,)|var|float32
call_1092 = relay.TupleGetItem(func_737_call(relay.reshape(var_1093.astype('float32'), [4, 1, 3])), 2)
call_1094 = relay.TupleGetItem(func_739_call(relay.reshape(var_1093.astype('float32'), [4, 1, 3])), 2)
bop_1097 = relay.less(call_1087.astype('bool'), var_1060.astype('bool')) # shape=(11, 14, 4)
bop_1100 = relay.less(call_1088.astype('bool'), var_1060.astype('bool')) # shape=(11, 14, 4)
output = relay.Tuple([bop_1068,call_1082,var_1083,const_1084,call_1092,var_1093,bop_1097,])
output2 = relay.Tuple([bop_1068,call_1085,var_1083,const_1084,call_1094,var_1093,bop_1100,])
func_1101 = relay.Function([var_1060,var_1061,var_1083,var_1093,], output)
mod['func_1101'] = func_1101
mod = relay.transform.InferType()(mod)
var_1102 = relay.var("var_1102", dtype = "int64", shape = (1,))#candidate|1102|(1,)|var|int64
var_1103 = relay.var("var_1103", dtype = "int64", shape = (16,))#candidate|1103|(16,)|var|int64
var_1104 = relay.var("var_1104", dtype = "float32", shape = (27,))#candidate|1104|(27,)|var|float32
var_1105 = relay.var("var_1105", dtype = "float32", shape = (12,))#candidate|1105|(12,)|var|float32
output = func_1101(var_1102,var_1103,var_1104,var_1105,)
func_1106 = relay.Function([var_1102,var_1103,var_1104,var_1105,], output)
mutated_mod['func_1106'] = func_1106
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1025_call = mod.get_global_var('func_1025')
func_1026_call = mutated_mod.get_global_var('func_1026')
call_1119 = relay.TupleGetItem(func_1025_call(), 1)
call_1120 = relay.TupleGetItem(func_1026_call(), 1)
func_1054_call = mod.get_global_var('func_1054')
func_1058_call = mutated_mod.get_global_var('func_1058')
var_1134 = relay.var("var_1134", dtype = "int16", shape = (2, 1))#candidate|1134|(2, 1)|var|int16
call_1133 = relay.TupleGetItem(func_1054_call(relay.reshape(var_1134.astype('int16'), [2,]), relay.reshape(var_1134.astype('int16'), [2,]), relay.reshape(call_1119.astype('float64'), [14, 44]), ), 1)
call_1135 = relay.TupleGetItem(func_1058_call(relay.reshape(var_1134.astype('int16'), [2,]), relay.reshape(var_1134.astype('int16'), [2,]), relay.reshape(call_1119.astype('float64'), [14, 44]), ), 1)
func_1101_call = mod.get_global_var('func_1101')
func_1106_call = mutated_mod.get_global_var('func_1106')
var_1180 = relay.var("var_1180", dtype = "int64", shape = (1,))#candidate|1180|(1,)|var|int64
const_1181 = relay.const([-1,4,9,9,7,6,8,4,2,6,4,-10,6,3,8,-8], dtype = "int64")#candidate|1181|(16,)|const|int64
var_1182 = relay.var("var_1182", dtype = "float32", shape = (27,))#candidate|1182|(27,)|var|float32
var_1183 = relay.var("var_1183", dtype = "float32", shape = (12,))#candidate|1183|(12,)|var|float32
call_1179 = relay.TupleGetItem(func_1101_call(relay.reshape(var_1180.astype('int64'), [1,]), relay.reshape(const_1181.astype('int64'), [16,]), relay.reshape(var_1182.astype('float32'), [27,]), relay.reshape(var_1183.astype('float32'), [12,]), ), 6)
call_1184 = relay.TupleGetItem(func_1106_call(relay.reshape(var_1180.astype('int64'), [1,]), relay.reshape(const_1181.astype('int64'), [16,]), relay.reshape(var_1182.astype('float32'), [27,]), relay.reshape(var_1183.astype('float32'), [12,]), ), 6)
bop_1196 = relay.left_shift(var_1134.astype('uint64'), const_1181.astype('uint64')) # shape=(2, 16)
func_66_call = mod.get_global_var('func_66')
func_71_call = mutated_mod.get_global_var('func_71')
call_1200 = relay.TupleGetItem(func_66_call(relay.reshape(var_1182.astype('float32'), [3, 9]), relay.reshape(var_1182.astype('float32'), [3, 9]), relay.reshape(call_1119.astype('float64'), [616,]), ), 0)
call_1201 = relay.TupleGetItem(func_71_call(relay.reshape(var_1182.astype('float32'), [3, 9]), relay.reshape(var_1182.astype('float32'), [3, 9]), relay.reshape(call_1119.astype('float64'), [616,]), ), 0)
bop_1203 = relay.left_shift(var_1180.astype('uint16'), call_1119.astype('uint16')) # shape=(11, 14, 4)
bop_1206 = relay.left_shift(var_1180.astype('uint16'), call_1120.astype('uint16')) # shape=(11, 14, 4)
uop_1208 = relay.atanh(call_1179.astype('float64')) # shape=(11, 14, 4)
uop_1210 = relay.atanh(call_1184.astype('float64')) # shape=(11, 14, 4)
output = relay.Tuple([call_1133,var_1182,var_1183,bop_1196,call_1200,bop_1203,uop_1208,])
output2 = relay.Tuple([call_1135,var_1182,var_1183,bop_1196,call_1201,bop_1206,uop_1210,])
func_1213 = relay.Function([var_1134,var_1180,var_1182,var_1183,], output)
mod['func_1213'] = func_1213
mod = relay.transform.InferType()(mod)
mutated_mod['func_1213'] = func_1213
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1213_call = mutated_mod.get_global_var('func_1213')
var_1215 = relay.var("var_1215", dtype = "int16", shape = (2, 1))#candidate|1215|(2, 1)|var|int16
var_1216 = relay.var("var_1216", dtype = "int64", shape = (1,))#candidate|1216|(1,)|var|int64
var_1217 = relay.var("var_1217", dtype = "float32", shape = (27,))#candidate|1217|(27,)|var|float32
var_1218 = relay.var("var_1218", dtype = "float32", shape = (12,))#candidate|1218|(12,)|var|float32
call_1214 = func_1213_call(var_1215,var_1216,var_1217,var_1218,)
output = call_1214
func_1219 = relay.Function([var_1215,var_1216,var_1217,var_1218,], output)
mutated_mod['func_1219'] = func_1219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1224 = relay.var("var_1224", dtype = "float32", shape = (8, 5))#candidate|1224|(8, 5)|var|float32
uop_1225 = relay.cos(var_1224.astype('float32')) # shape=(8, 5)
output = relay.Tuple([uop_1225,])
output2 = relay.Tuple([uop_1225,])
func_1227 = relay.Function([var_1224,], output)
mod['func_1227'] = func_1227
mod = relay.transform.InferType()(mod)
var_1228 = relay.var("var_1228", dtype = "float32", shape = (8, 5))#candidate|1228|(8, 5)|var|float32
output = func_1227(var_1228)
func_1229 = relay.Function([var_1228], output)
mutated_mod['func_1229'] = func_1229
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1233 = relay.TupleGetItem(func_865_call(), 0)
call_1234 = relay.TupleGetItem(func_866_call(), 0)
func_227_call = mod.get_global_var('func_227')
func_230_call = mutated_mod.get_global_var('func_230')
var_1236 = relay.var("var_1236", dtype = "float64", shape = (54,))#candidate|1236|(54,)|var|float64
call_1235 = func_227_call(relay.reshape(var_1236.astype('float64'), [6, 9]))
call_1237 = func_227_call(relay.reshape(var_1236.astype('float64'), [6, 9]))
output = relay.Tuple([call_1233,call_1235,var_1236,])
output2 = relay.Tuple([call_1234,call_1237,var_1236,])
func_1238 = relay.Function([var_1236,], output)
mod['func_1238'] = func_1238
mod = relay.transform.InferType()(mod)
mutated_mod['func_1238'] = func_1238
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1239 = relay.var("var_1239", dtype = "float64", shape = (54,))#candidate|1239|(54,)|var|float64
func_1238_call = mutated_mod.get_global_var('func_1238')
call_1240 = func_1238_call(var_1239)
output = call_1240
func_1241 = relay.Function([var_1239], output)
mutated_mod['func_1241'] = func_1241
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1269 = relay.TupleGetItem(func_865_call(), 0)
call_1270 = relay.TupleGetItem(func_866_call(), 0)
func_7_call = mod.get_global_var('func_7')
func_9_call = mutated_mod.get_global_var('func_9')
const_1272 = relay.const([2.741254,-2.529420,-0.939478,6.364677,-6.517691,-2.639467,-3.907985,-7.318801,7.358664,0.986970,-4.718487,1.661759,5.335075,-5.290792,6.816786,-2.045058,3.127951,-5.136779,2.720383,2.398630,6.674873,-9.526953,-7.369459,-7.203351,-6.389379,-5.114172,-1.289393,1.102827,-1.875833,-1.941631,-5.628690,-7.068519,9.597771,9.681787,-7.242571,-2.180753,-2.511887,4.783480,8.720416,5.346801,0.702426,-0.133971,3.150263,2.499885,7.851752,2.224206,-7.278245,-2.464654,-4.095516,-1.980698,-6.640372,-5.990788,4.086923,5.437122,-1.660308,0.048967,-7.371876,4.252701,1.101378,0.894399,-4.370188,-8.187192,-3.375533,9.276696,9.569809,7.490225,-5.477028,-6.100719,5.192220,3.417847,-1.750728,4.039682,-1.199692,6.163569,9.742755,3.942710,8.550237,-1.235528,-4.197225,3.487122,-0.841156,5.115465,-7.181708,-1.072699,3.262875,0.251039,-2.177979,-6.668902,2.007096,5.067816,-0.312286,8.043212,-9.845632,0.485640,1.473612,-0.535357,9.008054,-4.408427,-3.810173,0.577346,-0.381189,3.685782,3.688409,-4.226158,1.799767,1.431081,-1.230449,-2.478454,-7.027555,9.688932,-2.861932,9.043829,6.293861,6.864219,8.779894,9.535089,-9.352280,-1.456885,-6.034031,-7.124312,-9.280211,8.196585,-2.010857,-5.727546,7.871099,-5.160489,4.912957,9.423455,-2.511962,-1.031066,0.306511,9.059820,2.445465,7.026821,-3.311237,9.917769,7.181495,9.512601,-9.184124,-9.878129,0.491922,8.532573,9.088350,9.636295,-8.268883,2.192391,6.919779,9.632510,-8.266414,-1.191464,-7.780794,-9.085386,-3.187443,-4.214340,1.411644,-8.651026,0.635796,6.039212,-8.796617,-6.629559,0.200744,-3.088144,2.425069,7.180176,-1.938423,-1.343730,1.165394,-4.582496,8.939141,8.435290,-9.987153,6.930668,2.720222,-9.182899,-2.979803,-4.753663,-6.184891,-4.105735,-6.418837,8.560951,-3.409676,-1.979083,-7.650333,6.544746,4.987558,-3.146211,0.302230,0.719670,5.474691,-6.293639,3.428076,3.810994,-3.964133,7.722994,8.230016,7.214716,-1.652706,5.310436,1.816404,-9.558844,7.593327,-2.085619,5.897755,-4.463775,0.810200,-5.466006,6.573785,-0.140611,2.921574,6.313973,-3.881671,-9.129230,-6.617625,-0.704418,6.041986,-8.173938,8.578639,1.938123,-0.916557,7.262532,-9.984916,-2.008426,9.738701,8.478447,-6.690551,1.992933,-0.951311,-4.736438,-1.239100,-6.104573,0.768296,-0.288848,7.251033,9.616187,6.401625,2.021586,-4.365456,-5.555056,-2.676657,6.055821,-6.577606,8.325811,-6.403254,6.258361,-1.585615,-1.027515,-9.482288,8.029352,-1.762703,8.811723,4.116458,-6.162219,7.827146,-2.986515,-8.973209,-0.034829,-1.022458,4.024255,1.482215,2.722900,-1.316186,-7.095288,7.543727,-9.969040,-6.349212,4.010673,-6.618766,-7.249948,-4.442007,6.072029,-0.695055,3.763486,-3.717984,5.806281,2.204272,9.532505,6.937049,-0.698504,8.524811,-0.678402,8.777764,-0.051924,1.301615,4.288698,2.139721,1.800616,-1.978439,-7.051412,4.375444,9.342828,6.907962,4.136581,-9.879783,6.483056,-8.903386,1.737722,5.861836,-9.808057,-2.009190,7.230980,-5.867689,9.050238,5.576152,9.926971,-5.242070,-3.915503,-1.178012,-1.197520,8.612254,-1.858225,7.484446,1.453119,0.967245,-6.297299,3.943075,5.631512,-3.330700,2.633037,7.987687,-7.731049,4.234329,7.903471,-9.858640,-5.057124,8.806178,-3.221045,-1.707311,0.301026,4.354952,-3.285561,6.832327,-5.174097,-7.210841,-1.885480,8.748432,8.254410,8.403076,-6.418439,-7.907018,5.458910,6.170850,1.019980,-3.826787,-1.240574,-1.837288,-3.590626,-6.274775,6.586989,-0.769711,-8.167176,-7.177118,-3.420962,-8.646405,-2.433443,0.294294,-4.631359,0.419048,4.364235,3.025728,1.236266,-1.423949,8.210538,-9.482457,1.961031,-8.872172,-0.583977,-3.059069,-1.438502,9.215932,8.079332,9.994095,-2.513881,9.911470,-2.883100,9.162499,-8.002723,6.362408,-4.531222,-8.865585,0.212247,-5.849772,-2.099988,-2.882333,5.049818,6.940825,0.874243,9.973669,2.551494,4.752892,8.167873,2.467202,-4.551057,7.876117,-2.206419,-6.002862,1.335018,-1.598133,0.047218,-0.227288,-2.197876,-2.771174,-2.816772,5.729869,-8.113479,7.258801,7.816771,-4.967350,-8.777267,4.555487,3.556207,-1.562063,-9.941846,-0.995084,4.552212,-6.835992,-8.353114,8.571742,-8.272779,-6.772278,6.771602,-4.198410,-7.524035,-8.087089,7.294971,-9.690162,-4.835639,9.804217,5.554425,-3.892463,-3.290755,-2.616908,5.056817,-3.521657,7.111841,3.024192,-0.144420,9.129649,-8.925620,1.584627,8.520012,-2.847630,-7.636999,-4.576391,-4.640586,5.632773,-2.658951,9.203760,5.244576,7.391375,1.195429,-3.646448,4.879284,-3.869092,4.458035,0.035290,3.410911,-4.418022,0.312270,7.889599,-1.911975,-0.056700,4.860220,-0.546101,6.311094,9.108027,-5.946892,5.602711,9.927232,2.971294,8.001979,-3.107055,3.061910,8.075537,5.497837,-5.878175,9.267638,-2.013896,-5.985275,3.622019,6.566973,-1.123953,-8.323349,0.077326,-3.563420,1.917876,6.967795,-6.260820,-7.404416,1.549957,4.966395,5.503385,-6.686484,1.697129,6.135416,1.572900,3.450319,-5.137661,0.756358,2.032763,1.723831,5.322372,-9.025386,6.506421,8.894047,3.737963,-9.289994,4.983104,-2.618657,3.753401,-5.203459,-7.829315,3.156617,5.130616,-2.661242,-8.713379,3.563881,-0.694190,5.631997,0.474500,-6.565448,5.902709,5.789036,-3.796081,1.822876,7.220044,-3.036853,4.895564,4.666897,-6.302233,-2.395450,5.773352,-8.372541,0.944991,1.584473,7.010155,5.094558,-1.538505,2.028620,5.511240,-5.801506,-7.983946,-9.070882,-5.301496,2.987254,-7.485989,6.197293,6.341456,5.402418,-2.737875,7.873026,1.952267,-1.688177,-1.961080,-2.521968,3.081896,-5.719747,9.800519,7.154809,-7.569673,-5.790469,-0.563131,-7.502774,4.709687,7.772717,-2.698344,9.402228,8.333740,7.826410,-6.646645,3.172341,8.010830,0.553818,8.286419,-5.537192,2.733041,-9.805759,1.135940,-6.655032,-5.990979,-8.963032,-5.484966,-9.053426,4.803952,5.949228,1.939939,4.271096,3.844768,4.729844,-7.742604,-8.876216,-0.539336,5.889114,-8.198023,5.874003,1.348752,0.684458,0.845154,-0.731054,-0.512471,-8.103026,-1.129594,-7.864461,9.873949,-7.186386,1.425736,5.824255,3.304117,9.877134,-0.343741,9.078589,-5.260876,-1.282743,-4.525004,3.722087,-9.482132,1.501269], dtype = "float64")#candidate|1272|(616,)|const|float64
call_1271 = func_7_call(relay.reshape(const_1272.astype('float64'), [11, 14, 4]))
call_1273 = func_7_call(relay.reshape(const_1272.astype('float64'), [11, 14, 4]))
uop_1279 = relay.cos(const_1272.astype('float32')) # shape=(616,)
func_1227_call = mod.get_global_var('func_1227')
func_1229_call = mutated_mod.get_global_var('func_1229')
const_1282 = relay.const([[-3.218212,-2.133848,-3.553572,-0.660783,-1.747840,-9.722283,-0.410838,-5.683722,-7.968497,-0.289992,-2.206657,0.446873,-9.361998,-2.320402,4.882578,3.337880,2.081550,-7.921404,4.776894,0.739046,1.839092,-9.927693,-3.748777,-3.821606,5.903622,-8.671857,-6.795115,-6.790293,-0.199708,4.582735,3.498502,9.806534,-7.059495,-4.881819,-9.930688,3.864822,-8.917974,-6.797758,6.640546,5.818933]], dtype = "float32")#candidate|1282|(1, 40)|const|float32
call_1281 = relay.TupleGetItem(func_1227_call(relay.reshape(const_1282.astype('float32'), [8, 5])), 0)
call_1283 = relay.TupleGetItem(func_1229_call(relay.reshape(const_1282.astype('float32'), [8, 5])), 0)
func_947_call = mod.get_global_var('func_947')
func_952_call = mutated_mod.get_global_var('func_952')
var_1285 = relay.var("var_1285", dtype = "int8", shape = (66,))#candidate|1285|(66,)|var|int8
call_1284 = relay.TupleGetItem(func_947_call(relay.reshape(var_1285.astype('int8'), [11, 6]), relay.reshape(var_1285.astype('int8'), [11, 6]), relay.reshape(var_1285.astype('float32'), [11, 6]), ), 1)
call_1286 = relay.TupleGetItem(func_952_call(relay.reshape(var_1285.astype('int8'), [11, 6]), relay.reshape(var_1285.astype('int8'), [11, 6]), relay.reshape(var_1285.astype('float32'), [11, 6]), ), 1)
uop_1287 = relay.sqrt(uop_1279.astype('float64')) # shape=(616,)
bop_1295 = relay.multiply(uop_1287.astype('uint64'), relay.reshape(call_1271.astype('uint64'), relay.shape_of(uop_1287))) # shape=(616,)
bop_1298 = relay.multiply(uop_1287.astype('uint64'), relay.reshape(call_1273.astype('uint64'), relay.shape_of(uop_1287))) # shape=(616,)
output = relay.Tuple([call_1269,call_1281,const_1282,call_1284,var_1285,bop_1295,])
output2 = relay.Tuple([call_1270,call_1283,const_1282,call_1286,var_1285,bop_1298,])
func_1303 = relay.Function([var_1285,], output)
mod['func_1303'] = func_1303
mod = relay.transform.InferType()(mod)
var_1304 = relay.var("var_1304", dtype = "int8", shape = (66,))#candidate|1304|(66,)|var|int8
output = func_1303(var_1304)
func_1305 = relay.Function([var_1304], output)
mutated_mod['func_1305'] = func_1305
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1310 = relay.var("var_1310", dtype = "float32", shape = (11, 9, 14))#candidate|1310|(11, 9, 14)|var|float32
uop_1311 = relay.erf(var_1310.astype('float32')) # shape=(11, 9, 14)
bop_1315 = relay.multiply(uop_1311.astype('float64'), relay.reshape(var_1310.astype('float64'), relay.shape_of(uop_1311))) # shape=(11, 9, 14)
bop_1320 = relay.not_equal(bop_1315.astype('bool'), relay.reshape(uop_1311.astype('bool'), relay.shape_of(bop_1315))) # shape=(11, 9, 14)
output = relay.Tuple([bop_1320,])
output2 = relay.Tuple([bop_1320,])
func_1327 = relay.Function([var_1310,], output)
mod['func_1327'] = func_1327
mod = relay.transform.InferType()(mod)
var_1328 = relay.var("var_1328", dtype = "float32", shape = (11, 9, 14))#candidate|1328|(11, 9, 14)|var|float32
output = func_1327(var_1328)
func_1329 = relay.Function([var_1328], output)
mutated_mod['func_1329'] = func_1329
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1351 = relay.var("var_1351", dtype = "float64", shape = (10, 7, 2))#candidate|1351|(10, 7, 2)|var|float64
uop_1352 = relay.log2(var_1351.astype('float64')) # shape=(10, 7, 2)
const_1355 = relay.const([[[-4.939732,-1.906991],[-2.880491,8.709384],[-5.280031,-5.631781],[-6.252982,-9.645593],[5.011400,6.936213],[-6.307183,-8.186478],[-3.762489,8.553542]],[[3.733550,8.082167],[8.869117,-2.197950],[5.175381,2.501913],[4.950758,3.439846],[-8.509216,4.233402],[8.873737,-7.671498],[2.939657,-8.538707]],[[-6.097297,-9.999230],[-1.634326,-9.105208],[0.578842,-9.903643],[-3.076428,6.067850],[6.625369,1.799734],[-9.206680,0.975120],[4.055398,2.796811]],[[-7.184575,-3.161541],[9.826945,-6.445485],[-2.751000,4.259086],[0.615466,3.530121],[0.298484,5.418086],[7.178791,-5.896718],[9.799391,-7.688698]],[[2.385908,2.628557],[6.853480,6.382546],[0.798893,3.694834],[-1.511499,0.182643],[-4.201244,6.946061],[2.746825,-5.562062],[9.948140,-9.082551]],[[-9.327151,9.237080],[7.408737,-0.886192],[5.655533,7.051281],[9.674393,-1.538083],[-6.922202,-6.394288],[-6.351149,-5.782649],[-1.615836,0.955696]],[[-0.856272,6.310586],[-6.668083,-4.072134],[3.431050,-7.975792],[-5.979787,8.858433],[-5.656526,5.707329],[-1.909087,1.929403],[9.800594,-5.659231]],[[-6.760205,2.979305],[-9.916007,-4.759578],[-6.271530,-6.766945],[7.394876,-9.144839],[-0.419471,7.940731],[3.675951,9.913045],[-7.862771,0.162146]],[[8.710303,-5.696195],[8.286927,-3.540827],[-2.587945,-8.856355],[1.291934,-3.586624],[2.269584,1.229665],[4.663223,5.487555],[5.185710,-0.802570]],[[0.550760,-3.829179],[7.067500,0.468707],[-6.011020,-5.774387],[-8.016059,2.237482],[5.531765,-5.751911],[-6.666171,8.033247],[-4.509472,5.187739]]], dtype = "float64")#candidate|1355|(10, 7, 2)|const|float64
bop_1356 = relay.not_equal(uop_1352.astype('bool'), relay.reshape(const_1355.astype('bool'), relay.shape_of(uop_1352))) # shape=(10, 7, 2)
bop_1359 = relay.logical_and(var_1351.astype('bool'), relay.reshape(uop_1352.astype('bool'), relay.shape_of(var_1351))) # shape=(10, 7, 2)
output = relay.Tuple([bop_1356,bop_1359,])
output2 = relay.Tuple([bop_1356,bop_1359,])
func_1362 = relay.Function([var_1351,], output)
mod['func_1362'] = func_1362
mod = relay.transform.InferType()(mod)
var_1363 = relay.var("var_1363", dtype = "float64", shape = (10, 7, 2))#candidate|1363|(10, 7, 2)|var|float64
output = func_1362(var_1363)
func_1364 = relay.Function([var_1363], output)
mutated_mod['func_1364'] = func_1364
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1025_call = mod.get_global_var('func_1025')
func_1026_call = mutated_mod.get_global_var('func_1026')
call_1366 = relay.TupleGetItem(func_1025_call(), 2)
call_1367 = relay.TupleGetItem(func_1026_call(), 2)
output = relay.Tuple([call_1366,])
output2 = relay.Tuple([call_1367,])
func_1389 = relay.Function([], output)
mod['func_1389'] = func_1389
mod = relay.transform.InferType()(mod)
mutated_mod['func_1389'] = func_1389
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1389_call = mutated_mod.get_global_var('func_1389')
call_1390 = func_1389_call()
output = call_1390
func_1391 = relay.Function([], output)
mutated_mod['func_1391'] = func_1391
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1389_call = mod.get_global_var('func_1389')
func_1391_call = mutated_mod.get_global_var('func_1391')
call_1448 = relay.TupleGetItem(func_1389_call(), 0)
call_1449 = relay.TupleGetItem(func_1391_call(), 0)
var_1452 = relay.var("var_1452", dtype = "float64", shape = (616, 11))#candidate|1452|(616, 11)|var|float64
bop_1453 = relay.greater(call_1448.astype('bool'), var_1452.astype('bool')) # shape=(616, 11)
bop_1456 = relay.greater(call_1449.astype('bool'), var_1452.astype('bool')) # shape=(616, 11)
uop_1461 = relay.sin(call_1448.astype('float64')) # shape=(616, 1)
uop_1463 = relay.sin(call_1449.astype('float64')) # shape=(616, 1)
uop_1464 = relay.log2(uop_1461.astype('float64')) # shape=(616, 1)
uop_1466 = relay.log2(uop_1463.astype('float64')) # shape=(616, 1)
output = relay.Tuple([bop_1453,uop_1464,])
output2 = relay.Tuple([bop_1456,uop_1466,])
func_1471 = relay.Function([var_1452,], output)
mod['func_1471'] = func_1471
mod = relay.transform.InferType()(mod)
mutated_mod['func_1471'] = func_1471
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1472 = relay.var("var_1472", dtype = "float64", shape = (616, 11))#candidate|1472|(616, 11)|var|float64
func_1471_call = mutated_mod.get_global_var('func_1471')
call_1473 = func_1471_call(var_1472)
output = call_1473
func_1474 = relay.Function([var_1472], output)
mutated_mod['func_1474'] = func_1474
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1499 = relay.var("var_1499", dtype = "int16", shape = (8, 1, 8))#candidate|1499|(8, 1, 8)|var|int16
var_1500 = relay.var("var_1500", dtype = "int16", shape = (8, 5, 8))#candidate|1500|(8, 5, 8)|var|int16
bop_1501 = relay.greater(var_1499.astype('bool'), var_1500.astype('bool')) # shape=(8, 5, 8)
uop_1508 = relay.sin(bop_1501.astype('float32')) # shape=(8, 5, 8)
bop_1516 = relay.logical_or(uop_1508.astype('bool'), var_1499.astype('bool')) # shape=(8, 5, 8)
uop_1530 = relay.cos(bop_1516.astype('float32')) # shape=(8, 5, 8)
uop_1532 = relay.sinh(uop_1530.astype('float32')) # shape=(8, 5, 8)
bop_1534 = relay.power(uop_1530.astype('float64'), relay.reshape(bop_1501.astype('float64'), relay.shape_of(uop_1530))) # shape=(8, 5, 8)
output = relay.Tuple([uop_1532,bop_1534,])
output2 = relay.Tuple([uop_1532,bop_1534,])
func_1539 = relay.Function([var_1499,var_1500,], output)
mod['func_1539'] = func_1539
mod = relay.transform.InferType()(mod)
var_1540 = relay.var("var_1540", dtype = "int16", shape = (8, 1, 8))#candidate|1540|(8, 1, 8)|var|int16
var_1541 = relay.var("var_1541", dtype = "int16", shape = (8, 5, 8))#candidate|1541|(8, 5, 8)|var|int16
output = func_1539(var_1540,var_1541,)
func_1542 = relay.Function([var_1540,var_1541,], output)
mutated_mod['func_1542'] = func_1542
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1571 = relay.TupleGetItem(func_865_call(), 0)
call_1572 = relay.TupleGetItem(func_866_call(), 0)
var_1573 = relay.var("var_1573", dtype = "float32", shape = (11, 2))#candidate|1573|(11, 2)|var|float32
bop_1574 = relay.subtract(call_1571.astype('uint8'), relay.reshape(var_1573.astype('uint8'), relay.shape_of(call_1571))) # shape=(11, 2)
bop_1577 = relay.subtract(call_1572.astype('uint8'), relay.reshape(var_1573.astype('uint8'), relay.shape_of(call_1572))) # shape=(11, 2)
output = bop_1574
output2 = bop_1577
func_1582 = relay.Function([var_1573,], output)
mod['func_1582'] = func_1582
mod = relay.transform.InferType()(mod)
var_1583 = relay.var("var_1583", dtype = "float32", shape = (11, 2))#candidate|1583|(11, 2)|var|float32
output = func_1582(var_1583)
func_1584 = relay.Function([var_1583], output)
mutated_mod['func_1584'] = func_1584
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1589 = relay.TupleGetItem(func_865_call(), 0)
call_1590 = relay.TupleGetItem(func_866_call(), 0)
output = relay.Tuple([call_1589,])
output2 = relay.Tuple([call_1590,])
func_1598 = relay.Function([], output)
mod['func_1598'] = func_1598
mod = relay.transform.InferType()(mod)
output = func_1598()
func_1599 = relay.Function([], output)
mutated_mod['func_1599'] = func_1599
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1389_call = mod.get_global_var('func_1389')
func_1391_call = mutated_mod.get_global_var('func_1391')
call_1633 = relay.TupleGetItem(func_1389_call(), 0)
call_1634 = relay.TupleGetItem(func_1391_call(), 0)
output = call_1633
output2 = call_1634
func_1635 = relay.Function([], output)
mod['func_1635'] = func_1635
mod = relay.transform.InferType()(mod)
output = func_1635()
func_1636 = relay.Function([], output)
mutated_mod['func_1636'] = func_1636
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1635_call = mod.get_global_var('func_1635')
func_1636_call = mutated_mod.get_global_var('func_1636')
call_1641 = func_1635_call()
call_1642 = func_1635_call()
output = call_1641
output2 = call_1642
func_1651 = relay.Function([], output)
mod['func_1651'] = func_1651
mod = relay.transform.InferType()(mod)
mutated_mod['func_1651'] = func_1651
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1651_call = mutated_mod.get_global_var('func_1651')
call_1652 = func_1651_call()
output = call_1652
func_1653 = relay.Function([], output)
mutated_mod['func_1653'] = func_1653
mutated_mod = relay.transform.InferType()(mutated_mod)
func_331_call = mod.get_global_var('func_331')
func_332_call = mutated_mod.get_global_var('func_332')
call_1704 = func_331_call()
call_1705 = func_331_call()
output = relay.Tuple([call_1704,])
output2 = relay.Tuple([call_1705,])
func_1713 = relay.Function([], output)
mod['func_1713'] = func_1713
mod = relay.transform.InferType()(mod)
mutated_mod['func_1713'] = func_1713
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1713_call = mutated_mod.get_global_var('func_1713')
call_1714 = func_1713_call()
output = call_1714
func_1715 = relay.Function([], output)
mutated_mod['func_1715'] = func_1715
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1389_call = mod.get_global_var('func_1389')
func_1391_call = mutated_mod.get_global_var('func_1391')
call_1734 = relay.TupleGetItem(func_1389_call(), 0)
call_1735 = relay.TupleGetItem(func_1391_call(), 0)
uop_1736 = relay.acos(call_1734.astype('float32')) # shape=(616, 1)
uop_1738 = relay.acos(call_1735.astype('float32')) # shape=(616, 1)
bop_1744 = relay.power(call_1734.astype('float32'), relay.reshape(uop_1736.astype('float32'), relay.shape_of(call_1734))) # shape=(616, 1)
bop_1747 = relay.power(call_1735.astype('float32'), relay.reshape(uop_1738.astype('float32'), relay.shape_of(call_1735))) # shape=(616, 1)
var_1748 = relay.var("var_1748", dtype = "float32", shape = (616, 15))#candidate|1748|(616, 15)|var|float32
bop_1749 = relay.multiply(uop_1736.astype('uint64'), var_1748.astype('uint64')) # shape=(616, 15)
bop_1752 = relay.multiply(uop_1738.astype('uint64'), var_1748.astype('uint64')) # shape=(616, 15)
output = relay.Tuple([bop_1744,bop_1749,])
output2 = relay.Tuple([bop_1747,bop_1752,])
F = relay.Function([var_1748,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1748,], output2)
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
input_1748= np.array([[0.221989,0.560797,8.046842,5.635768,7.232847,7.658404,-7.230615,-5.331082,-5.656340,-4.243739,-9.535837,-6.931537,-9.983718,-7.694557,-2.049017],[-2.264084,5.531896,4.472834,-6.041092,-5.559181,0.080267,2.458744,-7.740449,6.912811,1.620530,-9.129429,7.150742,4.981822,1.460181,4.407640],[-6.178939,-0.736411,9.044459,-6.828120,-5.059766,9.733590,-6.544581,-3.373321,1.908837,2.255690,2.662657,-2.151499,-5.869408,1.142698,5.881226],[1.586041,-8.328569,-6.782622,3.085435,-1.174611,6.095716,6.877680,-4.277519,-3.786170,3.672992,-5.228160,-9.770856,7.453694,-5.949222,-1.234290],[4.402244,3.660558,2.101449,2.307919,-2.764879,5.148304,-6.712105,5.102774,-8.873138,-1.432328,5.358594,-7.361199,6.085818,-1.148893,-8.366058],[-9.730034,-8.095030,-9.334376,3.320333,-3.133248,0.596977,0.509210,-6.493593,-3.703470,-0.478926,3.766519,8.909182,5.791478,2.004883,-8.480679],[2.337458,-5.280534,-9.826277,7.820523,-4.579666,5.242986,1.581390,5.683034,8.675468,-2.066906,-5.742681,2.043570,-4.962188,7.833488,-1.918457],[6.546870,-7.243523,6.173330,-4.842083,-0.078224,-2.828171,-3.645210,-0.756703,8.041577,5.660611,2.902131,2.939575,6.994495,-2.627675,-6.380580],[-3.878033,-4.708324,6.617457,5.139553,-9.653461,-6.418348,4.302954,2.644948,8.032446,8.725216,-8.239627,-9.842862,5.215967,0.778921,-2.953114],[-7.247319,-6.196836,0.732912,-9.949514,2.007119,-1.787075,1.293951,5.986673,2.095144,-3.093989,1.772909,-8.098798,-3.912399,-3.292201,4.676941],[9.695960,0.167366,-8.470239,5.097954,-5.288161,0.006031,9.192028,-9.745401,8.910927,7.992960,2.444286,3.995789,-6.089439,6.459544,5.615854],[-3.891819,5.905071,-7.062769,-1.351249,-9.111368,1.899806,-7.393829,9.400630,4.513341,-8.570219,-5.166678,5.986106,1.415302,2.042452,-2.265402],[7.454240,-1.523017,4.830975,-7.250760,-6.196378,-8.097428,5.058128,-3.887511,3.230731,7.649183,-3.122225,2.657919,5.586065,-5.652275,-6.020784],[-4.195044,3.426117,-4.346514,3.115283,-1.684592,7.091086,6.750378,5.425999,6.512759,9.502907,2.429219,-8.892614,-3.723754,5.695926,8.402946],[2.099146,-5.630905,-3.899295,1.159277,5.623858,3.552146,-8.863514,1.988125,7.930366,3.550226,-3.844819,-7.989945,2.242068,-6.725672,8.983651],[-1.894069,8.723007,0.659645,8.947171,-7.518366,3.366109,-9.286127,-4.440383,8.283786,-1.916217,8.136257,4.730703,-2.568874,-5.295598,-8.287872],[-5.879843,-7.958614,9.169415,7.781131,4.285686,0.490667,4.267488,-2.155926,-0.263412,7.370851,-3.409600,0.396663,4.586729,5.918735,-5.022090],[5.279146,-3.015176,7.860545,-7.344822,3.428610,-5.524183,-2.415912,-2.549350,0.311878,-5.996272,9.036354,4.881922,7.783870,-6.106567,-3.013651],[-1.587934,4.864484,-8.103297,-2.232847,5.913580,-2.639984,-7.992451,-3.200944,9.312459,4.016643,-0.367441,5.845570,0.801240,-5.969061,3.049328],[0.942391,2.866606,-1.489469,2.201601,-7.731484,-0.745932,-3.340308,7.087757,2.520228,-3.518028,-4.076146,-3.465937,5.297457,-3.580396,0.708760],[-8.318679,3.205676,-8.670960,5.609192,-2.868948,-1.801616,-3.498604,-2.214366,2.993698,8.671211,-0.030023,6.499003,7.405754,1.326984,-5.860902],[3.258997,0.838641,-3.049096,-2.450766,7.169044,0.451340,-3.937947,-0.706407,3.799330,-2.997921,8.885570,7.142269,8.540325,5.921914,-3.443899],[-3.776469,0.970883,1.019379,5.346582,0.792056,-4.018676,-1.676329,9.819471,0.892378,1.675833,8.931624,4.112292,6.994605,4.375346,5.440204],[-2.770882,6.149076,2.572647,-7.039157,-0.307971,7.208046,-8.644175,1.764073,-2.232241,6.875167,9.881128,-0.033028,8.796672,4.134244,6.664253],[-0.100920,-0.209441,3.170870,-6.013273,-6.763956,4.211948,-1.236521,-4.868235,5.083820,-1.758630,-4.097470,6.716962,-1.097298,-8.412622,4.552023],[-8.905297,1.549047,2.127340,-8.698056,-3.962086,4.998344,-3.754157,-8.332414,5.446828,2.741634,-4.518636,3.375091,3.099355,-1.062768,-7.498351],[8.283526,6.152399,6.330782,5.439203,4.662934,4.079434,9.407737,-5.521229,-9.715407,-6.360001,9.568810,9.353980,-1.243065,3.427433,-5.462692],[7.225081,-3.579494,-7.377723,4.157342,-0.898104,-3.195764,4.152503,9.085404,3.110028,-6.101905,-8.965548,-0.882559,-7.428469,-9.364052,2.737938],[6.791983,-3.092154,5.675482,-8.409015,-9.240055,9.484992,-4.861656,-6.285599,-4.493927,0.958947,1.608233,2.851124,8.246955,9.687103,2.094418],[-4.843514,7.268117,-2.219231,2.554482,7.241951,9.572403,-2.955449,-6.461495,-2.704130,-6.806494,9.659091,6.763591,7.166364,-1.962864,-2.555855],[2.220371,5.092188,3.867335,-6.639020,-0.986201,-8.551678,-4.312652,0.487419,2.863256,-8.107493,7.076591,-7.309602,-8.077457,8.132535,-8.039017],[-5.476221,-2.833929,-5.257715,1.219215,-2.277216,-5.924028,-1.040811,-8.537792,7.417225,7.488996,6.538066,5.476801,1.939146,-8.074583,9.635871],[-3.494266,-6.974187,9.983932,-0.681522,3.600127,-0.393460,7.630280,0.947802,-5.661472,4.961105,-1.462494,-9.466196,-5.982505,8.828340,-4.303247],[0.885838,-6.833827,3.143037,-7.353741,-5.575620,-2.592926,-5.716205,9.282545,7.543170,-9.160448,6.194672,-8.210636,5.116306,4.736485,4.518132],[8.507264,-2.784427,-0.576508,6.831796,3.310121,0.230591,-4.457413,2.831289,6.163092,3.559553,-6.807653,-3.005567,-9.178680,1.237896,-6.379748],[4.110804,-9.235698,1.351876,4.190102,8.880685,-9.180806,-5.219576,-1.743591,0.178926,1.108863,-8.418783,7.992845,-8.458658,-5.380230,-9.974054],[-7.178633,-7.619841,4.015811,8.102685,-8.548955,-4.245349,4.141217,8.807407,-5.615831,7.274512,8.330146,5.988960,-6.580940,-6.731513,-1.165336],[1.813678,-2.520716,6.014007,5.990610,-1.455266,9.889773,0.412417,6.169377,8.395175,7.000796,3.372371,-3.528112,9.479484,-8.225718,0.796932],[-9.735482,8.542305,-2.808706,7.266340,-2.178101,4.397437,-2.315495,-9.560606,9.656123,6.888130,1.518830,2.106268,-4.169877,-4.274882,-0.792366],[4.541640,7.211634,7.696771,6.414063,2.345931,-1.021126,1.425758,1.027383,-0.464203,8.983643,9.128188,4.213855,-4.891969,-3.237009,-4.667979],[-2.363192,2.425450,3.411255,2.139137,0.812366,-0.458756,8.195736,-9.659528,8.264376,8.297554,4.125312,0.888856,0.084853,-9.661070,2.146017],[-6.371836,-8.401938,1.908250,-7.138451,-8.177237,0.784450,1.095886,-0.134724,-8.665952,5.969420,7.713587,-8.656399,4.690445,-8.141839,8.302731],[-1.998232,-2.540214,-1.449445,-0.759698,-4.661205,-4.375608,8.091767,3.445706,-4.069183,-2.704207,8.212385,-2.614077,-7.504416,-4.028179,7.031173],[-6.392601,5.723954,-4.819962,-9.516195,7.152432,5.833505,4.351328,9.201469,-5.553001,-5.799367,5.970361,-7.514037,5.199092,5.485489,-2.172546],[-5.216266,4.007799,3.018703,0.844581,-8.346124,-8.945943,-3.269619,7.029066,-3.420612,2.631877,-0.194280,9.825625,-4.292407,2.826391,-0.029966],[-7.460117,0.917230,-8.686716,1.469644,-5.181632,-4.811697,7.506733,-8.757955,5.251677,7.858444,3.747145,1.555149,-3.745108,-2.672421,-7.356567],[0.572571,7.455874,-4.525987,9.677843,4.394176,1.339044,1.683102,-6.563825,1.311302,-8.977329,6.414830,-0.290786,-7.253151,-6.568515,9.106449],[4.743103,-8.496467,1.624382,9.171757,-1.762228,-0.265112,-0.080319,8.625318,-3.644725,-4.410305,1.062495,-5.730537,2.865976,-5.055636,4.303063],[-1.927502,5.604065,4.840885,0.183952,5.450345,4.526912,6.616176,4.562308,4.663329,-9.109906,4.193398,9.976816,2.016076,-8.365923,-9.673208],[6.463099,2.434054,6.001332,6.718793,9.066778,5.614024,6.638930,-2.549688,4.372002,-1.012170,-4.239653,0.034520,6.081269,-1.376192,-1.124177],[-1.139312,0.863551,-3.839979,-3.243910,8.463128,3.137364,2.532132,2.026127,4.313921,7.841607,-6.831835,7.065620,4.136900,8.527416,4.317829],[-1.226860,4.935998,1.392821,3.431620,-0.463562,2.274997,3.013128,-2.868972,-6.223365,7.107996,8.893043,7.705224,8.043416,6.784231,-2.494010],[-5.291339,-5.852001,-1.528168,-4.689661,-6.361846,-8.184351,-2.268366,-2.095940,6.924491,-0.778682,-5.785956,-8.965620,9.551370,-9.364024,-7.014488],[3.025854,7.238393,1.752849,9.255510,5.009142,-9.190677,3.262731,8.165859,8.569835,-1.111366,3.574630,9.184497,5.468814,1.231376,2.361856],[-7.427333,0.042886,5.818861,-4.480294,-9.869224,-8.071073,8.460804,6.587367,4.359827,7.450455,-2.391619,-7.291521,0.572942,-7.361516,1.244391],[-2.909344,-3.237902,3.618891,-4.337267,-7.352082,-2.195412,6.763272,-5.665084,-5.655912,9.103186,-7.147919,2.156717,0.222195,-0.754653,9.549811],[-6.175312,3.734228,0.020171,-5.954590,-0.394871,-1.633017,0.702321,4.994336,-3.041876,8.230073,2.482617,-2.555844,-1.942104,0.730930,0.636378],[-7.845348,4.306318,5.895515,-6.479296,-2.178039,-0.907440,0.006813,0.053087,-9.972361,3.656445,8.510203,5.833726,1.940142,-3.481588,-0.566978],[8.910201,0.612314,-3.862694,-3.841078,-0.467470,-0.579044,2.658309,-0.337698,-0.594582,3.610472,-5.656983,-5.191350,-3.705159,0.617933,5.538472],[-9.772582,-6.009848,8.475290,-4.958242,5.554992,-1.088392,-3.457329,-6.764484,-8.743964,7.604967,-9.351856,8.930718,1.246941,-0.711725,-2.648745],[-7.602212,-4.893530,-1.059060,0.171588,1.941772,0.774888,-9.461202,-4.610170,-6.184088,4.819571,-1.444338,3.859651,0.152943,9.118451,9.155257],[0.544474,-7.859215,-6.851394,9.653578,-9.839970,-0.385510,-5.683371,2.072249,6.216176,-8.927851,7.594935,-4.396154,-2.391542,0.892376,4.254198],[0.285444,-3.112970,9.114620,-7.890919,-0.279517,3.081238,-1.065085,-3.768233,-4.834139,5.513939,-6.770579,8.977959,3.449691,-0.191397,0.509130],[3.316534,-6.425797,9.895992,-2.002875,-1.778934,-6.857492,4.721548,-9.851850,-5.980860,-7.513987,8.248880,-9.434976,8.498261,-4.647450,-2.735331],[-1.742604,-3.741667,0.764680,-6.428259,8.764080,-0.509706,-4.470548,-8.768496,7.256847,-8.359420,-7.056459,6.637379,-4.929804,-1.133459,-4.928351],[9.150838,-7.202608,-1.978921,4.556348,5.296155,1.963455,-6.066648,1.519076,-1.936263,1.650658,-3.852837,-5.196968,-4.550039,5.688788,5.419499],[3.380995,-2.590296,-3.192179,4.070868,0.499504,-4.648672,-0.010691,-8.745239,4.103466,2.741880,0.272575,-7.550975,4.176736,8.660020,8.949477],[-5.700715,5.953212,-2.112206,7.855549,6.890724,-0.546580,-2.917338,-2.632093,3.758170,4.834763,-4.769795,-1.302665,8.196006,6.232820,-7.289130],[1.779216,7.437201,6.276082,5.724365,8.921639,5.428909,-3.642377,-7.123702,7.772513,2.380966,-3.072432,7.719287,-6.889720,-6.078421,9.882363],[6.616877,-6.811411,3.623696,-2.498504,-2.140613,3.706650,-0.230300,7.664905,-2.799425,-4.103681,-9.258978,0.359720,3.534302,-0.413158,0.421091],[0.441619,-7.152204,7.145790,0.447789,-4.044654,2.011295,-6.739464,-1.848922,2.744458,-2.600714,-7.651262,-3.127287,-0.747101,8.033532,-8.021519],[3.845081,3.995581,0.850067,8.076335,5.883618,-7.059738,-6.742153,-3.061191,2.382761,0.286090,2.619449,-6.169572,-2.687276,4.307538,3.260873],[-2.272183,9.387446,-2.006607,2.641691,-9.346499,-9.868939,7.753305,2.724571,8.809270,6.110637,-6.684711,-9.998436,1.350729,4.029251,-1.190577],[-3.688979,2.458290,0.312435,-0.810547,-0.899885,7.352320,8.979948,1.955354,-7.308788,7.443802,6.528219,-9.273208,-5.356365,-4.833784,-5.874640],[-5.327406,-6.649707,4.810475,-2.210275,-2.552897,-6.629696,9.068246,-5.333700,-7.238739,7.777608,5.360899,2.342981,-2.291362,0.349467,0.339937],[2.317165,-3.462224,-7.562314,6.386536,3.842593,2.016820,1.094513,0.534744,6.853920,-6.275053,-9.488956,4.385300,6.083017,0.105396,4.587389],[2.322237,8.075863,-4.161871,5.722576,-5.110829,9.519299,-4.342665,-0.781972,-5.438619,1.001956,-3.463525,-3.949857,3.725492,4.041210,-0.476506],[6.615735,9.165505,7.862854,-6.883740,-6.478804,6.349936,-8.658523,-7.673849,-3.534432,-0.329944,-4.545826,9.479291,6.720555,-5.622009,8.165508],[4.500222,-5.378920,9.639894,-0.955155,-5.158584,-1.909676,-2.500083,5.668406,3.828796,7.572545,8.142532,-3.265118,0.347511,8.765986,-2.937179],[-3.903267,-7.718732,2.301690,1.396065,3.988856,-2.161094,8.774874,4.356189,1.243358,-4.238818,-6.773403,-3.572293,-0.097028,-2.922958,-9.368445],[3.044290,-6.488340,1.642824,1.820705,4.398714,4.416002,4.849711,-1.185271,7.479854,8.834635,8.254616,-2.610328,1.999717,7.783715,-1.304874],[-6.661814,-8.531033,1.541618,-9.016380,-6.304740,-8.382132,5.027226,6.684576,-0.351231,-1.908914,-8.695904,5.808637,5.990233,-9.442977,8.463657],[-9.590013,-4.318960,-7.961770,2.340070,-4.118238,9.491402,9.998098,-3.556650,-1.859507,4.663183,2.928190,-0.687506,-3.641114,8.125480,-1.186317],[-0.460542,-9.203965,-5.437789,-4.215884,-8.629875,7.414142,7.052726,-8.848550,3.968128,-9.502371,4.046668,9.592512,8.946676,-3.575260,-8.168604],[9.391857,8.414287,4.710785,-1.705478,5.135558,-5.281346,-3.761457,1.201882,-7.902789,-9.353304,-0.665578,-0.751072,4.320089,2.217967,-3.758186],[0.459002,-4.143915,7.299000,-0.842119,-1.786151,-9.804502,-6.344386,-7.026484,-8.920373,4.255308,-8.825784,-5.416329,6.296538,-3.104044,4.780605],[-6.089146,-8.044170,-6.241986,-5.284403,7.121020,-0.192537,9.528121,5.053753,2.738081,-1.696614,-4.188713,-9.671952,2.820948,-2.640242,-3.793100],[-7.295491,-0.847276,-4.825993,3.200517,0.324741,1.365629,-3.161069,3.741008,-4.893800,4.716532,2.340928,7.129824,1.511825,9.465322,-8.536432],[0.497088,-3.018442,1.581164,-4.873375,5.971451,-0.123490,-1.656007,5.452852,-7.794755,-4.417859,7.653576,0.877940,5.115585,5.354991,-7.172704],[-0.704846,2.173305,-8.939838,3.473859,4.490803,-5.021082,9.297552,-1.305017,-8.012662,3.540711,7.783941,-5.523113,2.402075,8.493705,-3.613318],[2.676486,5.004391,8.043820,0.587117,8.820829,2.215546,-3.399228,-7.831041,-5.321097,9.586610,7.166819,-6.597976,9.811042,6.501283,-6.640007],[6.169383,-8.556214,-5.076469,-6.562765,1.678069,-7.525967,8.411888,4.176949,-0.376554,6.866411,8.700423,-6.980245,-0.276343,-9.184157,-1.471878],[-2.357375,-2.462783,-7.131476,-7.146947,-9.022293,-6.898068,4.510998,-2.340188,3.139683,-9.493261,-9.586093,2.668085,3.157988,1.616779,3.363611],[-3.005374,2.066365,5.741403,-3.663237,-6.281444,1.165733,5.647911,-7.916220,5.866726,-7.037852,-6.667805,-4.422391,9.545723,-9.502417,0.563776],[-5.134142,5.891542,-4.368382,-1.421918,-2.617474,4.508100,7.022868,-2.596425,-1.301698,-8.888699,9.596769,-3.081178,-8.139854,1.201684,-7.382611],[1.342360,9.820408,-6.215990,4.960766,-7.566834,7.322800,8.311466,-6.254152,-1.712229,9.059700,-9.055171,-8.609374,-0.854833,-2.828914,3.623047],[-6.891847,-8.566801,6.714148,2.010016,4.683019,-7.782061,-2.790875,1.237045,-7.899995,2.109789,-6.031107,-7.038209,-7.508175,-8.824640,2.121024],[4.682347,-7.714047,4.046119,-0.521664,-7.900389,-5.664466,-6.823050,-1.503183,2.423170,-1.132527,-7.518406,-3.647029,7.962565,-4.803914,-5.557790],[7.538802,9.415597,1.108589,0.869380,-9.050923,7.061924,-5.473299,3.791423,4.360564,6.719976,7.804199,4.564746,3.327678,5.608151,-6.062232],[2.372673,0.787565,2.985596,4.621554,9.167716,-6.245212,-7.421388,-8.152333,1.087522,-5.199561,0.582341,-7.355576,-4.760963,4.968516,2.903428],[3.114425,1.860729,-2.091968,6.345971,8.868366,0.552411,2.401296,7.111285,6.261359,1.979502,3.590042,3.540087,6.442292,6.274477,8.518121],[-7.165310,9.488462,-7.033337,-2.144204,9.751542,-3.886579,5.429626,-6.322460,5.188907,0.288415,-2.722075,-2.484074,0.451244,8.133780,-2.038317],[3.723040,2.831213,7.298907,-3.189664,0.829028,3.120795,-0.604222,8.965069,7.213108,-9.414320,1.896112,-1.909612,3.406846,-5.870094,-7.774520],[7.135498,3.708839,1.270323,-3.948587,1.466317,9.132328,0.280671,7.476162,9.055961,-0.592460,-8.833202,-4.556304,1.930505,-5.301903,-9.117722],[-8.257410,-2.872998,-6.749422,-5.636370,-3.622777,-8.805270,0.561344,3.987378,-1.109531,3.671495,9.331358,-8.377857,-9.187897,7.847385,8.701407],[-5.293248,0.948915,-5.340222,-8.191024,9.113081,-5.051658,-2.462235,7.063047,-3.623910,-6.720129,-4.421303,4.411233,7.639679,-4.012778,-2.154734],[-6.225632,-2.366970,-6.285282,5.077818,1.686678,9.043152,-6.696037,0.504810,7.956759,3.351831,-1.869413,5.063760,5.961045,8.843545,7.938885],[5.364545,-7.733143,-7.272076,-0.659199,4.651973,-5.836230,8.621834,6.245403,-1.922809,6.787616,-9.338296,-7.556267,9.088689,0.699431,4.281903],[1.738605,-1.022953,-1.134863,-9.444253,3.223197,6.827143,-4.133165,-9.513851,2.237245,-5.177192,5.770513,-3.190367,6.335084,8.781042,9.238475],[2.697962,8.016872,-3.982808,-9.240028,0.366820,3.172301,3.729874,6.749186,5.217147,3.869725,7.020296,-0.740592,-3.525447,-1.706545,-1.258964],[-8.632652,-5.512524,-1.590690,-3.480671,8.271770,-2.068997,-6.394882,-2.285221,7.582624,-7.631397,4.567749,3.790386,-1.594261,5.140404,7.754738],[-8.397669,-4.901048,-2.551525,7.875422,1.676142,-6.504143,8.791141,-6.669740,-9.801959,2.705417,-4.260812,6.999962,1.857152,0.951007,-8.716402],[-4.895913,-9.771832,6.688488,2.916885,9.422320,9.789766,-4.415955,-7.123774,-2.200490,7.374728,-1.414593,-1.052061,0.701329,9.385980,8.608421],[-1.165720,-7.602389,-2.176651,1.492844,-2.449593,-7.517495,-1.552184,1.118661,7.062421,1.755399,1.250357,-3.521251,8.078566,3.332713,-9.451786],[-0.490375,7.546361,-6.684452,-3.913497,6.070826,-5.272429,-6.414439,2.224822,-9.260046,5.188478,7.492489,3.474458,1.885097,7.534936,6.362108],[0.904934,-7.025273,-3.787187,-6.053931,-9.675276,6.095661,6.359045,1.291285,9.801305,-1.217786,-2.319812,-0.121115,5.446381,-0.882744,-8.355380],[1.429648,-7.654220,-7.368988,-0.312510,8.364720,2.698197,0.813713,4.252681,9.225818,-9.593084,4.003873,0.944178,-3.162547,9.668312,5.158542],[-4.079917,7.287322,4.906858,6.050953,-1.298319,-0.714194,-4.647275,6.175718,1.312030,-1.072869,-3.850841,-6.483250,1.630139,-3.995610,3.038783],[6.291725,-9.319237,-5.869184,-2.663797,8.357936,8.499810,-4.149794,-4.562517,5.865923,-7.255976,3.527282,3.628396,-8.995198,5.473148,-3.436183],[8.306797,-0.363429,-4.661695,1.434723,1.333665,-2.971613,-7.916988,-4.808831,-6.453134,-2.129984,5.593778,6.228636,5.634385,-1.346269,-2.679537],[-2.047403,-3.724846,-3.749368,0.325241,5.715022,-7.852009,9.364544,1.347990,0.678442,0.836644,-4.560432,-5.299053,-0.998362,-8.185287,-1.074901],[5.732117,-6.221224,8.817636,-5.076267,-8.125543,9.876208,7.816203,0.321641,-5.975315,-4.808500,2.284051,9.895697,-8.161053,2.643721,-7.658439],[-3.838125,-4.524101,3.766200,5.254100,5.636705,2.451147,-4.234462,4.977984,3.136273,1.892162,-3.292315,1.728836,-5.472038,9.999980,-5.537458],[8.748826,0.571826,9.426824,9.892165,-7.190311,-8.982416,-7.793831,9.673754,-6.426617,7.736924,-9.101052,-2.026721,9.721430,-1.758972,8.249304],[-6.052084,1.676443,-9.810833,-9.619814,3.240148,-2.409383,9.213943,-6.232168,2.266425,2.291629,-5.369557,-0.056641,5.729063,8.210823,-3.277283],[-2.058658,7.658833,-7.381734,-9.956278,-0.282712,9.157530,1.339509,8.217495,1.415708,8.097887,9.919457,1.068061,-5.212697,-5.306470,-6.055228],[-7.435035,4.897232,-8.639651,-7.361541,2.162082,-7.962581,8.367725,8.629477,2.459871,2.574377,6.702354,8.127901,-9.024011,-7.035697,1.664706],[5.608473,9.722775,-9.601305,-6.666148,0.086493,7.115339,3.929877,4.457343,7.360083,4.790764,6.290115,5.524437,7.907917,3.151110,-9.841244],[-3.488154,-8.533021,7.371401,4.625211,7.323753,6.172265,-7.949051,-1.201356,6.135493,8.918383,-0.093208,2.452306,6.629054,-8.367689,-4.324429],[2.835059,-6.080489,0.134406,-8.084163,-0.025958,7.543187,-9.103439,-8.952082,3.510204,-4.537706,-3.871950,-6.649314,5.043682,3.475864,-9.488037],[-3.811315,-1.642758,0.284659,-0.861227,7.606221,-0.149939,-7.805814,-4.759459,4.072819,-8.613068,-8.460543,-3.385136,8.646922,-0.257515,1.677158],[-0.951231,2.533508,6.132574,-5.684823,-6.091447,-2.060864,-0.325436,-3.050033,2.817307,8.932695,3.819175,-2.821258,-9.294821,8.299799,-9.820581],[4.653681,7.216232,-4.524496,4.451110,-7.190915,-2.490101,-1.017565,-7.805703,6.199843,-5.357876,6.781663,5.666861,-4.555971,3.225810,-1.774803],[-7.210447,1.511216,-1.029819,-9.007506,-8.643233,-7.418775,7.873726,-3.964718,5.732760,8.773167,-2.929037,1.053363,8.414154,6.757173,-6.347656],[-3.321906,4.935135,-1.052665,5.513830,1.684443,9.784297,-0.003825,9.277094,-1.018213,7.318727,-0.251159,6.486859,1.184358,2.732311,2.997865],[-9.233845,8.217687,7.209077,-0.867291,-6.994588,9.206492,1.804685,5.421523,-3.545381,9.411369,8.638650,2.418221,4.797299,5.580864,9.480626],[-9.496691,5.256802,2.182169,2.861959,8.152576,1.282159,-7.945512,3.035995,3.546095,4.578055,3.897770,4.178973,9.121354,1.965844,1.625867],[-6.383230,-8.242450,3.414653,-3.484876,2.671236,4.926741,-8.680588,4.469451,3.966715,1.472603,2.259611,3.381653,7.570718,-6.916754,-7.261915],[0.485151,0.827861,-8.131307,7.680141,-3.352272,-2.801645,-6.896266,6.420837,-3.090183,-2.876825,9.907063,-7.119512,-3.361948,-3.506500,-0.523922],[9.709882,0.583329,-3.775017,2.712971,-0.787930,-3.695079,3.041273,-2.652552,8.993093,7.720691,-9.631334,2.808993,-0.887545,8.307180,-2.518260],[6.478253,5.668868,7.806068,-5.370224,-9.108222,-3.178171,9.923748,4.099478,-4.107596,9.973759,-9.455185,-9.006585,-8.490787,-9.320415,0.691367],[-6.430945,-8.456346,1.784147,5.070033,-8.303639,1.922488,-8.522855,0.652481,-0.454562,-1.945753,1.027462,6.382973,1.051546,-3.244895,0.157281],[-4.635731,-7.234435,-2.401436,-6.989206,0.116400,7.423589,0.137863,-6.997805,8.810786,-4.216742,-8.249659,-1.039464,-1.506280,8.690047,2.971066],[-1.537814,-6.590043,-5.702137,2.088920,-6.765250,-4.533980,8.337263,-3.899118,-1.739152,-2.316569,7.247554,-6.883385,-8.567677,3.267801,-2.405985],[7.483341,6.837959,4.421838,-8.530109,-2.197359,-1.973443,-9.556688,9.880152,3.621188,-3.197317,-5.924420,-2.067771,2.082911,2.676444,-4.492900],[-5.969479,9.104877,-4.998462,9.248676,2.087686,4.440742,-9.197774,7.762355,-8.964464,3.388345,-4.891850,8.083612,0.414822,6.483308,5.047841],[-1.951225,5.817626,0.563968,-8.249935,-2.492889,5.700212,2.409891,9.962422,1.126519,3.369335,-7.272025,-2.390617,-8.594526,9.038170,5.861080],[-8.919454,-1.141096,4.555588,0.534298,-3.754654,3.623869,6.186845,-5.632944,0.001346,9.529955,-5.651786,-6.925308,2.552725,-7.137758,-5.096525],[5.067611,9.686310,3.257937,-3.479903,6.773010,-1.018263,-7.242931,5.214652,4.289416,-3.896500,-3.059729,-3.731998,7.464335,5.625589,-9.520204],[2.346949,-9.447000,9.815666,9.467346,-6.437666,7.421530,9.125014,3.401576,-3.996716,4.834193,0.776009,8.199000,-8.270221,6.444134,5.922317],[-0.798563,4.751173,4.722953,-5.353502,-4.966025,6.998535,4.371224,3.030012,-1.672587,8.891177,-1.697810,6.084903,8.878934,8.800851,3.032244],[-7.165453,3.080488,7.864804,6.864601,-5.144393,-6.642294,6.295777,-3.531438,-8.268574,-0.613531,3.666568,-7.041733,-4.234097,-0.253684,-5.349711],[-7.636392,7.808596,3.269461,8.438913,-3.647445,-5.045104,3.073009,8.660688,0.194492,-7.238557,2.920549,-4.442734,-4.621600,2.119593,-3.498029],[9.765614,3.172276,4.585964,3.496082,5.238591,-6.944461,-0.893523,4.520156,7.966162,-0.846916,-7.583912,7.707491,5.235068,7.334399,8.894748],[4.024406,0.489715,2.568719,8.709725,3.166977,6.357504,-8.652243,7.431172,3.857252,-8.270698,9.769130,0.059140,-6.131980,-9.345120,-7.644227],[5.081311,-7.457399,-3.607375,4.711695,-2.627275,-5.910259,-4.322503,-8.933806,-3.427921,5.138146,2.431082,8.031583,0.340544,5.810905,8.139082],[3.933399,8.839543,4.147800,-7.529498,1.324621,2.505547,-1.846585,4.028471,-4.179756,-7.885811,-2.454024,-0.105036,-0.524861,7.985529,1.739452],[6.532212,-2.090423,0.203596,-1.605710,-9.611392,-6.199743,8.533260,-2.535207,2.863660,-4.034364,-1.797362,7.530379,-8.073980,-9.436766,-8.295356],[4.344386,0.427263,-5.723345,3.470336,-4.530055,4.259098,-8.286239,2.414096,9.555861,0.571342,-5.742042,-0.241051,-1.194735,0.863004,4.915153],[-5.189830,-9.263318,-1.704605,-1.601133,2.372955,4.366383,-4.377318,-7.631179,-5.301365,-8.457394,5.140458,9.789030,3.043481,1.036152,3.656448],[-5.361640,9.539610,8.542615,9.150157,5.882093,-0.996787,9.948950,-7.888010,-2.640028,1.488195,-8.741549,1.803527,8.192244,7.399290,1.739428],[4.545378,4.488700,-2.382545,-1.881260,-5.136599,1.308666,-5.055461,6.763242,-6.333518,-5.402678,5.191898,-7.803585,-6.555739,7.889969,-7.648458],[-4.155753,-8.947256,-4.122296,-6.449008,-4.411518,4.694168,0.165614,7.467743,-4.151410,-7.812953,7.387553,-2.477357,-0.190163,8.254258,-5.120152],[4.153539,3.662987,-9.747395,-0.632699,-1.170905,-4.389203,-4.743211,-4.412368,-6.150277,8.761674,0.917018,-9.138097,5.140254,-7.647979,-1.324596],[-8.330014,-2.307657,6.914148,-4.910856,2.260030,1.211334,-8.659567,-6.281144,6.965519,8.063628,2.040333,9.018874,-8.178233,7.149109,-9.342511],[-1.701424,5.953406,-1.218694,6.614633,-7.806704,-3.537783,-2.819779,-3.231596,9.477959,-9.139635,-3.469606,2.246777,1.371660,-5.275033,-2.052275],[6.615805,-1.437070,9.133498,3.945726,6.488631,-0.731121,8.550572,-1.974912,4.347206,9.970385,-6.098365,2.205995,-8.645915,-8.678202,7.028880],[0.579392,1.614967,6.116749,-3.849383,8.028417,1.387398,8.320434,2.613402,-9.336821,1.046298,-6.893829,4.795505,3.270133,-4.458687,3.388841],[7.281843,-5.606692,-3.601600,2.193583,7.617547,-5.374037,-0.599829,9.012394,5.598132,-5.597949,-9.553070,-4.437424,0.651082,7.860803,5.264921],[5.307922,8.186263,6.220741,-3.575641,-6.456171,-6.177993,-7.511408,0.765296,-8.893321,8.311143,5.603696,2.186094,3.845234,-6.403188,-1.742321],[7.008693,-1.536698,-4.617635,-4.663690,-4.725547,-8.274740,0.005716,-4.666399,-6.733716,8.831831,1.074561,0.087369,6.512119,8.047892,2.502365],[7.830990,-3.808705,5.382962,-2.508445,5.360563,7.000845,6.535782,-2.219754,3.376289,-9.841797,-3.447164,5.980212,-9.646037,-3.168650,-7.384417],[4.164719,4.082723,0.690717,-5.461431,-8.759735,4.415508,7.470204,5.078638,-3.061478,8.019864,1.155390,-1.366764,4.968066,4.015215,4.969700],[-0.293226,-2.350388,-2.506519,-4.851393,3.851091,-2.267239,-1.229717,3.141734,7.293363,-8.437994,-9.235411,-1.473359,9.966571,9.980227,8.127541],[-7.757351,8.089796,9.762633,9.345257,7.323043,2.420543,3.502678,9.025015,3.921126,-1.490023,7.630187,8.964402,0.331188,-4.468470,-0.070268],[6.965947,4.772193,7.890165,-9.051438,-4.798491,5.358855,9.395068,-6.462715,9.106992,7.342229,4.425366,-5.288028,-1.688191,1.510551,-2.985482],[7.879327,2.929671,-1.878381,5.899985,8.874396,-7.910503,7.496023,8.322617,7.793193,5.440976,-5.987842,2.673328,-8.075003,-8.842971,-6.081714],[4.925627,2.643913,-3.609281,1.064952,-8.437533,-7.856334,5.769216,0.047301,7.245859,6.579311,2.627921,8.226652,-3.760880,6.715830,-5.765504],[0.681138,7.423999,-4.034396,-8.724396,7.352219,-4.248696,-2.192273,-8.537266,-8.378676,-6.181703,-1.359793,9.939195,-8.849813,4.094351,-1.777529],[-1.646864,0.803998,-1.039211,0.654920,4.681704,-1.632941,-3.257078,5.839319,-9.592308,-1.171414,8.653376,-3.738428,5.986511,-7.668550,-6.508646],[-8.488934,-6.937434,-7.762216,1.232223,-1.642829,7.630356,-4.159822,4.387517,-5.564693,2.001841,-4.094965,4.512591,8.493300,5.853989,-0.033336],[-3.593621,-6.982257,-2.243923,-7.988250,1.723826,6.270555,1.801858,-8.682939,6.509138,-9.013552,-1.065608,-1.906862,0.777325,-3.284466,4.487414],[6.025925,1.684672,5.056909,-4.943075,1.752564,-0.535422,-3.378502,-0.454561,-1.413728,7.040622,-6.784109,0.371982,-0.563586,-5.895145,1.093081],[0.035564,9.775126,1.440456,-8.927891,-6.733545,2.292034,-3.301515,-2.632433,-9.632381,5.749419,4.506151,8.027772,-4.283067,-3.284403,-1.343185],[-4.847429,-2.798990,-2.828321,-4.892372,3.720194,-8.425146,9.938451,3.737046,8.326677,-1.799791,7.662806,-6.562857,-8.193355,-6.718982,4.816199],[0.589272,-1.008472,-1.643157,5.080950,-9.703003,8.937158,3.399675,6.003298,0.637346,-4.061297,8.461424,5.223300,4.526559,-0.830684,-1.849325],[-9.346460,-7.528071,5.912280,-3.268545,2.255060,-0.903013,9.416170,-2.987690,-8.767989,-1.858970,4.651517,-4.868126,3.663040,8.988050,-0.227228],[-5.067071,4.264256,9.987154,6.888487,9.816422,-4.755533,5.550522,1.073648,7.778245,6.276605,-7.720626,8.990382,-6.994083,-5.922656,9.984655],[0.852922,-9.089912,-0.552696,-1.377754,2.994030,-9.959489,8.174473,5.302083,1.018569,-8.046054,-3.049108,-7.932699,5.675889,4.179935,-1.003368],[-8.585255,-8.946147,-6.282007,7.815887,-2.311624,-3.956640,-4.465113,-6.048104,-6.362830,-7.924114,3.135350,3.100588,7.158192,-5.945584,4.758352],[1.283387,-6.665758,-3.422753,-5.674106,8.642118,-2.069955,3.233506,1.182715,-2.489475,-1.150160,-0.520557,-5.454980,6.244022,7.024874,-6.194179],[-9.436761,-6.077267,2.073893,6.996335,-0.831174,-4.774270,-5.070524,-3.613371,7.080589,6.538799,2.393004,2.237449,8.014119,1.909236,-7.350552],[9.059033,9.045422,-1.039233,-3.469674,-0.399652,-3.183471,3.000971,4.137615,5.804934,5.721768,-4.555240,-5.066560,-6.723693,-1.383437,-1.375793],[9.968567,1.326356,-6.311026,3.099967,2.138816,1.765637,-7.593754,-2.009795,-3.138646,-9.489667,-9.343920,5.390809,3.293126,-0.629420,3.408170],[-4.858750,7.367029,-2.194670,7.443445,-7.976369,4.638946,0.736371,8.796618,8.451734,7.382655,2.673660,-5.606053,-2.392361,-9.485020,9.266833],[-0.560826,-8.946854,6.688226,7.545980,-9.528190,2.978215,2.429922,3.628122,-7.311754,7.194075,-7.523979,4.655865,1.095763,-3.995688,1.860321],[5.162279,-9.979302,5.895938,-1.581548,-2.975743,-9.364397,9.781326,-5.241033,-5.728911,2.323189,-9.118947,-3.370270,5.239056,-1.491902,5.372065],[7.693245,6.913247,-9.767518,-0.479020,2.701780,5.954188,-2.231730,4.043122,9.809122,-9.150836,2.961059,7.886291,-7.468558,4.871539,-0.977209],[-2.202363,-7.007967,0.485420,-3.252782,1.182246,2.953344,-6.677205,8.107183,-5.737381,5.174189,-3.824027,-0.847257,0.469150,-9.838417,4.107269],[5.867318,-6.892071,6.123744,2.808403,5.700304,6.932789,0.017539,4.679775,2.143437,-0.693038,-3.306375,-2.001627,-5.031932,9.812662,-8.888009],[0.640851,7.616650,1.447040,-7.641295,-8.167847,-6.206152,-6.083942,3.896041,-5.728649,1.432722,-6.344598,6.717716,-7.203742,-4.059399,-9.694000],[-1.219687,-1.060300,-0.531602,-5.130722,-3.409732,-7.599421,-0.008293,6.663311,-4.629852,2.449483,-9.385515,8.300487,-0.447353,4.986285,-4.342945],[8.193988,0.417394,-8.686512,-0.313514,-9.595791,9.158955,3.358590,9.180886,-3.042654,-8.668587,-1.331635,6.370346,8.245610,0.979951,-5.015819],[-4.704899,7.595917,4.008827,6.790429,-3.157647,2.488069,1.563606,-3.070644,0.530286,-9.880133,-9.354025,-9.639016,0.014297,8.407063,-4.887420],[-5.388330,2.025697,-2.771543,7.912923,-5.209049,-3.879849,3.513814,-4.099838,-2.553041,-7.124417,-7.313668,1.820090,-1.601847,-7.992708,-0.980195],[4.485046,6.790440,-3.622468,-5.845332,-1.912919,4.726105,-3.723495,-3.323542,-6.059909,-2.084586,3.791399,1.349293,7.248626,9.730772,6.841031],[-6.616147,-1.453187,-8.147350,7.638536,-5.273647,9.251250,-5.397533,0.983527,9.380821,-4.120292,-6.448274,-9.034168,-7.096688,-6.861354,7.618178],[9.655680,-5.001028,6.810471,-3.425049,9.921380,-9.275940,6.265361,-7.163859,5.972900,-6.990686,0.686641,-3.596320,9.786978,4.673764,-4.423989],[7.969367,-5.480756,-1.386952,-5.211443,2.867739,4.171144,7.683606,-0.220912,1.425491,3.898569,8.055937,5.808270,-4.236118,8.597774,-9.221174],[-3.398220,2.869153,9.851205,-9.293383,3.509687,-8.213373,-2.681899,9.756405,7.494896,-1.981086,-7.637970,-1.892302,0.123969,-0.499358,-0.155259],[5.104317,1.203874,-0.473194,-5.384098,-6.970930,-0.921241,-2.830997,-7.350408,3.654271,2.181259,-3.100939,-8.930624,6.830237,8.568201,-6.851499],[-2.227800,-7.404125,-3.154558,-9.540183,-2.769284,-8.664493,8.637227,1.087131,-5.452556,-3.667075,-3.957000,-8.874545,6.816544,9.748541,2.141755],[-1.378966,8.867510,5.344197,1.999719,-4.747881,6.526010,-5.019773,6.238297,-6.307476,4.140102,1.836032,-4.150475,-7.959076,6.729062,-2.246698],[6.720940,-4.352960,-8.013647,-7.315819,-4.047964,5.847842,-5.143736,-4.692072,8.866734,1.281448,9.460723,-9.500320,-8.525119,3.217808,-0.294652],[8.973216,3.284080,-1.515785,4.584204,6.590581,3.715091,8.897542,2.174773,-6.249621,5.077105,3.925356,-2.041774,-8.233173,-4.255507,3.289183],[2.205232,-6.714348,-0.005864,-2.157573,6.642017,2.334012,-6.903926,-7.913018,9.141841,-8.705281,9.172883,9.297349,-1.770826,-2.855264,-8.883610],[3.362804,-1.954769,-7.972916,9.791856,7.365090,-4.942224,6.769911,0.511388,-8.438158,-7.837209,6.955937,-9.251940,-6.155752,-4.249635,8.159600],[-4.376367,3.969010,-1.920712,2.244134,-3.655295,2.808726,-2.431268,-2.548915,-0.913367,-5.471510,3.695370,-9.524654,-0.388859,-0.582965,-1.520516],[9.304717,6.476940,8.108758,-6.062553,3.712737,8.323939,-2.803656,3.765079,-3.977631,7.406453,6.465581,-8.375323,5.612019,-5.796953,-5.151419],[4.928365,6.272971,-1.848029,2.333792,7.604670,1.337786,3.345961,-4.752756,-9.300919,-3.441239,1.370775,1.386297,-5.045642,5.824254,-5.931316],[6.295390,7.282638,-6.058695,4.930750,0.288803,-5.241094,-5.161740,-8.294329,0.469385,1.183763,-3.915418,4.966948,-3.560478,-8.170868,-4.123054],[6.657725,1.208840,-8.965549,-8.143546,7.553730,-6.229033,6.642498,2.908879,-4.282042,-9.498382,-5.246876,-5.988714,-2.482205,3.443796,-1.298950],[-5.722675,9.309586,1.787090,6.732708,-9.359777,5.040431,1.634897,9.584455,-0.579133,5.353858,4.290462,2.633224,3.969840,-5.487959,1.087458],[-6.543716,2.246545,2.047408,6.895761,-1.734901,0.660833,-8.142842,-9.625035,7.622200,4.517673,-8.869238,-5.447516,9.066687,2.096057,5.601063],[-3.287704,-8.814096,3.798159,-2.891263,0.254632,-6.813399,-0.380448,9.776493,-8.378232,4.019035,2.216671,-5.218383,-8.687902,1.356484,-2.402209],[-2.738137,-8.077653,-5.952803,-2.802958,-3.591218,3.950518,-4.720466,1.260879,2.537410,-8.500483,-1.054066,-5.405856,8.539782,-0.546689,-3.286403],[5.060908,9.163534,-6.747153,-6.533396,3.800237,-8.162641,5.827392,-1.201010,-8.053908,2.001581,5.347378,-4.545782,-9.678774,-9.250856,-8.059714],[1.186159,4.894436,1.379238,-0.569085,-6.116678,-1.592990,7.469311,6.077644,-1.596954,3.166680,-8.405898,6.117324,-2.039759,-7.865170,-8.278838],[7.788483,1.297790,4.419218,-1.521776,-6.163144,9.692166,9.629996,-9.172247,-8.900212,-0.400880,-3.668139,-6.350149,8.719498,-2.615817,4.254172],[-5.337888,-0.781175,7.326460,-3.047230,-1.943361,-7.055380,6.510007,-7.819770,-7.187277,5.302674,-5.936604,2.246901,5.916894,0.334175,1.247892],[4.024803,5.109173,0.993137,4.620609,8.264069,-3.184941,1.609277,1.292614,6.064503,-1.117849,3.442475,8.248826,-3.805084,4.918358,-7.395246],[-5.564181,7.453817,2.910819,1.692323,-4.288260,-1.919970,-3.279672,7.553490,8.420396,-8.223704,5.418652,-8.080571,-0.876202,4.842725,-5.254428],[-0.451554,2.784603,1.039778,-0.599506,6.934571,4.159963,7.324962,6.932320,1.971910,-2.574314,1.005828,6.864827,-8.765704,-0.374994,-1.979127],[6.849442,9.269877,-3.627369,8.193054,3.303900,3.906986,5.499506,7.648678,2.453375,7.232035,-3.912901,4.168525,8.002165,2.827718,-7.438406],[-6.407684,0.024345,-9.130630,-0.424284,-9.728547,1.235677,-8.688874,0.578252,9.089808,1.397089,-6.188830,2.573565,5.743561,8.994580,-2.174007],[-5.427781,6.343085,2.580903,2.682179,8.459741,5.686633,-0.484781,1.156383,7.448155,-6.012674,-8.952258,-2.712054,6.748916,-9.102524,4.055214],[-8.614380,8.145081,3.214045,2.125163,8.105892,-2.810267,3.539425,-8.582393,-5.805926,-9.394758,-1.679408,1.380341,-0.837606,8.875776,0.341497],[-0.025004,-6.993905,0.656959,-7.928188,3.236264,-5.181830,-4.936896,-9.452308,-6.126426,9.429738,0.525763,5.958353,-0.353612,2.248363,-2.862040],[-2.185219,-0.643372,3.845685,-1.957925,7.756170,9.507770,-8.876175,6.722425,-6.189471,0.289737,-0.115085,-9.929550,-3.011360,-1.658469,-7.097116],[-7.566158,9.234338,9.946115,8.890157,-2.306511,5.175931,-5.903367,4.492779,-8.931798,4.715168,7.205909,4.764751,-1.440493,1.408810,5.622662],[5.814675,-9.226369,8.240010,8.178100,-5.875992,3.137954,5.731611,9.615752,5.532990,1.354234,3.323789,7.519662,-7.904972,-2.199683,-3.084340],[3.558033,-1.061390,-4.570530,5.470155,-0.328959,8.570368,-7.646106,7.937583,9.410008,8.805856,6.977199,-3.657690,0.918713,-5.321128,-7.738232],[-4.517314,6.813233,1.450861,-1.225385,6.929617,-4.371474,-7.445387,-9.572000,-4.745993,-7.988211,-5.122862,4.120390,-5.367444,6.394151,-8.952249],[1.098016,5.998338,1.409059,-1.412341,0.778285,-6.467774,-2.154205,-8.253220,-5.665449,0.192775,8.977518,1.211806,-9.025739,9.474364,-5.567329],[9.045947,-6.969487,-9.443691,8.264220,2.198448,8.880504,3.062682,-6.488944,-9.569482,-3.070932,-0.972052,5.807034,-1.727318,9.736533,7.192356],[-8.541806,-0.328131,1.257629,-0.336406,2.004327,4.459149,-4.354332,4.031074,4.968659,6.374276,2.199076,7.704785,4.835669,-4.381884,6.183711],[5.923802,-2.220230,6.823828,-7.313251,-2.148470,5.731927,-8.101858,-3.284511,0.027584,4.746910,-5.297100,-3.042036,5.612154,0.778345,-6.806514],[5.679883,-0.528300,9.089587,9.711841,-5.352880,-2.368821,-3.672626,9.492245,-7.338236,-3.608772,3.274823,-4.647748,-6.120696,-2.581174,7.375706],[-3.893256,-6.923094,-4.994256,-9.461761,8.178734,0.230426,3.516031,4.361530,8.578178,3.052266,8.494012,-5.580394,-0.859576,-3.903923,-0.493948],[3.702435,-1.386006,-5.202769,-7.006659,1.401501,4.135201,4.529227,-1.858241,4.352000,-2.194669,2.359320,5.698467,4.533054,-4.973289,-2.688897],[2.249425,3.782673,6.494600,-5.004923,5.892696,-6.735490,-3.092194,5.341306,-0.707814,-3.298129,6.625948,-8.203120,5.597508,-8.239737,6.004797],[9.174760,-3.748247,7.757039,-8.118171,2.394907,2.406909,-1.067006,5.802219,-8.610349,-4.475500,-5.758339,-1.785067,-9.112968,-7.990737,2.065978],[-2.204056,-7.503351,8.125827,-3.677973,3.994337,7.072828,1.942142,8.627317,7.503429,-5.088928,-6.282430,1.038324,-0.496147,4.251204,-0.061341],[-0.308224,2.004426,-4.626150,-6.569794,-2.972281,-3.732423,-8.701379,8.469784,1.911074,9.956954,-3.863062,-2.160574,-2.002872,4.202506,-6.513198],[-2.969941,-3.754704,4.173877,1.663065,-0.389519,-4.117080,5.481400,-9.844506,-2.062559,-2.285160,-5.530722,1.927296,-3.291299,9.303635,2.598421],[-3.988872,-6.517849,3.388032,8.449451,-1.616103,-9.096064,-0.189928,0.227372,-7.079629,-7.355803,-6.988389,6.393508,-6.000343,6.001161,-8.232417],[4.266154,1.472949,5.308706,7.125197,3.920995,-4.648154,-5.372004,6.275532,-5.260743,9.465648,-5.866574,2.398995,-8.915850,-2.157510,3.562151],[-2.002082,9.007907,-5.660726,2.490831,7.530961,9.084284,-2.132350,-7.340820,-2.023007,-3.372748,-9.924585,3.196960,-6.717085,5.346286,0.634461],[9.841378,9.812137,0.034389,-0.834889,6.237812,3.113868,8.392090,2.674372,-0.600260,-4.515051,5.954267,5.283425,-0.350024,-5.716768,-7.337673],[-6.545374,-0.407284,-8.444345,-8.167099,-8.295064,-6.746095,-4.728948,7.529599,-8.085346,6.921052,-7.985120,4.902584,1.760779,-2.666405,9.163241],[-9.745226,-7.483574,7.666792,-3.517557,6.878858,8.428166,3.263896,-9.598588,8.292911,-2.081912,2.144434,-3.334082,-8.903563,8.311725,6.583335],[-2.893115,-8.169535,-9.389467,6.858131,7.516805,6.159574,5.261657,0.966883,-3.369758,-8.522192,6.847664,0.777364,-0.005707,-2.923450,-4.238262],[-1.492885,6.132207,-6.776830,-3.936216,-3.928772,-3.704204,-0.381304,-1.477968,8.548317,8.428743,5.862255,-4.017613,-1.021263,4.158263,-9.184004],[-3.063241,-3.131354,6.820681,-4.329951,-5.736065,-0.661799,-0.139725,-4.997863,-0.193667,4.649836,9.320792,-1.610091,-7.977982,-0.170249,4.483020],[4.808362,7.550267,2.963801,-7.315892,-8.709311,1.142468,-0.292830,-0.251179,-8.490129,-6.131244,-7.375137,3.323199,7.270446,8.361790,3.621735],[-1.317853,-1.414390,-4.503070,0.990894,2.518321,8.527042,-0.801092,3.054103,1.789615,0.433843,3.070217,0.881864,-8.732518,-6.217712,4.365765],[4.000504,8.530775,8.802379,8.727825,-4.637925,7.117601,9.744154,7.695539,-9.978657,0.243453,0.449347,5.678724,6.141523,1.302425,2.757764],[2.008202,7.199885,-7.388389,-2.334642,4.474020,9.992078,-0.180171,-1.243825,0.959198,-3.612543,-8.802008,6.708069,1.480793,4.310413,1.044488],[7.975147,5.111316,1.196566,3.714100,-0.976946,-8.965327,1.480518,-9.026144,2.151056,-0.040107,-2.780711,1.861803,-7.354233,-1.910028,0.802264],[-6.719435,-2.689582,9.009445,-6.171023,9.772090,8.614620,-7.417213,7.247317,-0.518074,-5.356010,-3.601666,9.923970,-2.918947,-0.528273,-8.412286],[-0.243602,-3.040879,-2.927145,-1.587564,2.398627,6.965082,-3.201552,9.304761,5.865361,-8.805367,3.083944,-6.330482,-7.622719,-2.160255,-1.402202],[-3.358944,-9.890768,0.398658,8.194178,4.453344,-0.219567,1.034393,-7.067468,-5.889880,9.389872,8.011095,2.053552,1.841171,-9.397421,4.989834],[-0.115508,-5.821551,-7.254081,-8.931389,8.120866,-0.047971,-0.006486,-7.939731,5.696607,-6.281807,1.373186,4.628709,-9.058748,-2.305153,-9.418812],[5.484129,-9.891652,8.221549,4.500551,-5.982062,2.632116,-8.763412,-3.735213,-6.348173,7.347216,-2.908352,-1.668352,-0.193743,6.657394,-5.895190],[-4.150262,-3.825656,6.809532,4.450214,4.616554,5.761499,6.893527,9.400590,8.538854,8.525776,-1.058879,0.204010,4.726921,3.991796,2.457115],[-9.802381,3.261489,-4.296325,2.932113,-1.759706,5.737758,9.895808,6.810769,-4.145265,6.304760,-3.455729,-3.336800,-5.718945,8.110920,-6.945280],[7.646587,-7.299642,6.002982,5.246725,-2.803196,-5.217466,-1.816946,-3.493328,-5.006504,-3.312170,-1.791101,6.882741,-7.352993,-6.432713,-3.604820],[-1.602737,-9.092323,-4.723939,4.030529,-1.332651,8.363411,7.597619,-5.010196,-7.649930,1.262115,-3.558896,9.382714,-5.103326,-6.494390,9.241543],[3.896964,-5.260304,-7.992962,0.639149,3.596780,4.839347,-5.861522,-7.324049,6.084443,4.330145,1.950351,-4.145276,-7.056283,-6.936995,-4.897444],[7.896288,-0.314204,6.507016,-1.023171,-5.410824,-1.678065,7.218298,4.272654,2.135699,-1.582259,2.649373,7.336275,-4.622696,-3.495496,-1.998097],[-3.623746,-3.952490,0.282798,5.015614,-3.366578,0.484834,8.249158,-3.438764,-9.611709,5.691065,-2.082958,-1.234934,1.978866,5.614769,-0.444917],[-0.801701,3.037997,9.051445,-0.726437,1.105853,-4.017542,7.576668,4.780438,-7.378805,5.080843,-8.030222,-6.456630,3.082415,5.906608,0.566197],[-7.576557,-0.831737,-5.800424,1.486070,-3.320666,-9.534796,4.417295,-9.890081,4.720756,8.761729,1.310822,-0.501268,-0.142753,0.183360,6.683370],[2.953096,-8.320794,2.085390,9.234852,1.339578,8.610791,8.304304,3.446166,5.266011,-3.208345,0.368315,-0.107137,-9.207889,-6.653007,-5.712461],[-0.771268,8.104943,1.701559,-0.708454,2.247488,-1.929368,8.403352,-4.897798,-1.726818,3.634310,-8.362368,-7.546086,-0.482467,4.900986,-8.466619],[-9.307738,-8.065578,7.356779,1.205090,-0.289745,1.048139,1.838788,-8.761190,8.094775,-6.371296,-3.523840,3.952156,5.167521,7.343364,-6.556325],[9.232886,-2.589798,2.907502,5.852088,4.340898,0.350719,5.354351,9.773016,-1.475971,-4.219917,8.169810,2.324241,6.441031,7.195105,-5.275000],[-2.984003,-7.676996,0.674583,-2.433479,-9.004616,-7.949350,-2.678650,3.391685,-9.063196,-9.726785,-4.873004,7.607690,-7.662134,-2.071078,1.286766],[-8.122887,-0.996714,-7.735585,-9.886347,5.180373,1.567665,-9.932105,7.745969,6.655150,-8.457775,-2.071319,-0.917572,-1.943844,-4.087703,-7.377549],[3.805905,5.177460,4.886437,3.440847,-3.310342,1.098177,4.069627,3.861938,4.988692,-9.837751,-1.549799,-6.397055,-5.017267,5.082420,5.812554],[-0.094671,-7.685724,8.737397,9.172187,2.023119,5.295038,7.871683,3.548798,3.308693,-1.408529,-5.122710,-5.422237,1.385835,4.944169,4.742186],[-8.197024,-0.961466,8.955440,-7.989661,-4.215874,-3.379661,-7.514074,-5.641636,-8.331268,-9.849687,8.352628,4.522898,3.016399,-2.729289,5.618655],[-9.763316,7.612549,4.238664,-7.373605,-5.695600,8.341658,-9.593033,-4.355342,7.613037,-8.763362,9.782752,-5.186610,8.764110,0.541986,8.620556],[6.562653,-2.744590,-3.286506,8.980287,-6.511011,5.941556,-9.764164,4.731292,9.074254,9.405045,3.793779,1.533941,2.211625,4.209724,2.030933],[8.608645,-8.874444,3.930880,6.225746,3.679683,-0.302656,-5.594622,-2.899750,-5.048006,-4.165587,6.783395,-7.527785,-6.609248,9.862247,4.488875],[-9.283601,5.189906,-5.523164,6.494476,6.435699,5.445929,4.361350,-4.329404,-3.368618,9.714530,-3.700565,-8.924030,-1.957648,-9.258413,-7.850383],[-0.148027,5.421509,-8.189736,-8.748991,1.315962,-4.180228,-5.378493,-6.787828,-6.287436,2.093094,-3.473758,-3.933849,-2.369472,6.185505,6.380983],[-9.342296,-1.506535,6.879156,-8.272879,-5.669920,-4.522344,-9.184629,-1.116855,8.181172,6.753502,2.807368,3.444794,-6.744387,-8.054263,-4.714549],[-2.192407,0.524317,-8.771613,6.837139,-2.163434,7.431370,-6.524022,1.116278,-2.740428,3.322069,7.346432,0.474894,2.088544,0.645883,9.565095],[7.990849,-5.786451,-0.073893,9.888135,4.031670,7.963774,-2.154512,8.203936,0.232713,5.662215,5.118377,8.463329,4.175506,3.800582,-5.651628],[-0.199579,-6.962527,-8.015795,-1.718796,-2.795964,7.219651,-9.086045,3.283489,-8.223887,1.308305,-5.634895,-1.864743,-2.324816,-8.097334,-5.988029],[-4.220298,-3.368610,-6.003797,-9.498491,-0.093134,-9.471161,-5.764762,-9.529632,-7.951310,-3.678083,-2.584587,4.499319,-1.976939,-2.349753,1.861969],[6.762701,1.484743,-0.306053,-0.011902,-4.929074,6.803864,8.381193,4.591811,4.283655,-1.677742,-2.312660,-9.470529,5.218351,1.985247,7.681933],[2.534166,-5.753139,-0.310391,-2.445862,7.653215,-1.058128,3.812006,2.360560,3.259554,4.607905,-4.651223,-0.837346,1.896277,-0.243750,2.927027],[2.451452,-1.527622,1.343483,-7.233317,-4.914103,3.742355,-9.404581,-5.182821,6.481063,-0.088473,4.339270,-1.346563,-5.027343,8.689422,-1.870032],[-3.717837,5.407657,-2.070662,-7.997920,-9.698763,-4.451405,4.133041,4.923856,-2.239828,8.982005,1.104476,-9.992367,-6.145732,2.225197,5.024436],[-1.059819,2.393139,-1.173283,-7.067593,-5.270203,3.971089,-9.911675,0.896748,-1.158972,6.848984,-2.672477,-4.043664,-2.445820,0.975002,5.916239],[-0.579529,3.753500,6.491568,7.570819,-9.074237,6.779977,-2.415934,-5.583014,5.612909,-9.992575,-8.208423,9.673068,8.940477,0.894738,4.412685],[8.941362,-6.017087,5.913912,-6.590052,5.989813,-1.795697,9.986139,1.777677,-7.420738,-9.079708,5.961847,2.630647,-5.680462,-8.382651,-5.073864],[-4.297393,6.954028,-1.011651,-9.922532,5.758681,1.407938,-1.119516,0.280091,-6.148103,-1.817180,-0.636701,6.005966,-1.099195,-1.450717,8.976750],[-6.841821,-1.072676,-2.329007,-6.486418,8.321747,-1.883959,3.072823,5.563564,-9.315763,-2.861221,5.061983,-4.085116,2.435817,-7.080603,-0.813428],[-6.393377,6.814330,-7.211345,2.272099,-6.740473,2.915531,1.328606,-9.588485,9.936756,-2.037196,1.329809,-7.149241,-7.512313,9.990528,6.239674],[-5.836617,-1.113006,-9.259769,8.986393,-5.083765,-6.224315,-5.311016,-9.027498,1.423678,-4.301612,-9.821964,9.475424,6.283794,-8.424141,-5.084897],[-2.828305,9.069067,-3.764319,-5.275075,2.685963,5.420875,-6.279760,8.462650,-3.902276,-3.716555,-4.330948,2.294088,2.251917,2.857730,6.822599],[1.651646,-5.826762,-1.096668,-2.454553,-1.051741,8.884773,2.080388,3.602636,8.425435,2.597957,-0.687700,-5.287055,7.087564,2.110563,7.705844],[-5.203739,-5.202214,6.341471,7.576472,4.515739,2.638503,-6.421987,-9.542711,4.567558,0.364703,0.227191,-2.765732,0.050906,8.448749,-6.385972],[1.947889,5.806051,1.387025,2.798901,-0.554793,-0.553156,3.924753,7.499637,-5.732022,3.855842,7.603089,-2.329339,-8.416332,-3.088154,-6.945462],[-0.807743,8.666902,4.445716,-8.296263,-4.935773,-8.550171,-4.512803,7.397204,0.024077,-4.152624,6.820912,-5.181446,4.654596,-9.681287,-8.121314],[-3.364561,-7.613046,0.766597,-0.556036,-2.296394,-5.256229,9.927390,6.776467,-5.401460,8.694641,9.607857,0.279277,-1.305661,1.444335,-6.320429],[7.563121,-1.589315,-0.842822,1.135522,-5.232390,7.024107,-1.094088,5.284571,5.150666,8.858555,8.533145,-3.079163,-4.251843,6.867441,5.597398],[6.535937,7.009731,-3.612781,3.133927,-6.768432,-9.213174,5.636588,9.043222,-6.981645,-6.446278,3.925551,-5.285535,-6.762699,-3.464276,-4.568680],[0.116590,-3.215181,-9.518472,9.520561,-0.778473,2.918223,8.905269,-1.263551,1.380021,-0.014285,-8.426407,2.993723,4.803717,6.554129,-7.243962],[1.533735,3.877398,-1.693520,-0.079991,-3.440701,5.280075,6.898879,-4.744828,1.847336,-6.572281,7.094928,9.454698,-8.885720,6.151462,-3.412674],[7.947173,8.025987,3.132217,-2.231502,3.096873,4.389022,-1.877159,-8.587514,-7.969898,9.879789,7.459596,7.192460,0.646730,2.971432,7.513786],[-8.931196,2.566000,7.594180,5.465338,-3.677504,-6.534628,-5.779078,5.080211,4.180206,-0.693291,-0.997507,-2.864925,8.636590,2.499449,-2.698379],[-1.296152,-7.420753,-3.585013,7.481305,4.712655,5.863021,0.015883,7.127974,-0.739024,-0.226186,6.566754,-2.151534,-0.196670,2.477497,-1.606615],[-5.404519,7.428620,-3.099544,-7.104065,1.466297,0.691690,-9.322996,-7.572784,5.684038,-2.960254,-4.628473,5.930150,9.493651,-3.548600,0.483275],[-5.845133,2.426514,-4.919642,-7.845466,2.271978,6.536025,-0.601608,7.105496,4.156773,3.039457,-3.077520,0.555380,-8.150413,9.580300,7.741502],[5.873635,-2.155391,8.574173,-5.513307,-8.496124,-9.938739,-8.639947,4.511531,0.779895,8.253395,-6.595540,-2.197731,6.137796,4.943868,4.330225],[9.615080,-4.389471,-6.952602,1.995086,8.472303,-0.778867,0.766022,-6.008328,2.008727,7.097782,-2.211613,-3.665271,2.347526,9.762745,0.113674],[4.845359,5.641738,-5.151423,-0.123583,-7.364984,4.023151,3.456775,-4.301363,3.897826,5.885561,-2.651700,9.620672,6.004020,-9.741669,5.897748],[6.590160,-4.780856,-5.325114,-8.277802,-2.300326,2.702049,-3.532070,-9.956616,-3.613999,-5.203716,-5.609480,-2.578292,2.556837,-6.094507,-0.401167],[7.114195,1.811973,5.765280,8.612205,-6.473335,-0.584109,8.143669,-7.679635,5.353929,-5.944740,-8.472645,-4.778289,-4.268533,-8.437951,2.883232],[-0.656874,4.595687,2.192664,1.452645,-4.214813,-4.192552,4.494000,3.849637,-2.181911,1.658847,-3.993428,4.809242,-6.627233,3.904385,0.594112],[1.273343,5.896325,6.321728,1.574532,6.818664,-0.114701,4.204003,-9.201088,8.965560,4.915129,8.632410,-8.797249,7.969461,-4.612821,-6.156292],[0.557414,7.749641,-5.645330,7.875758,6.070637,4.259239,2.667599,-5.873657,-0.713130,-9.964738,-6.790536,2.598514,-7.496722,4.195826,1.702735],[-2.270354,6.905267,-5.612359,-2.886601,-6.258749,-9.059764,5.196072,-6.279094,4.077806,6.328317,-6.417325,7.323391,3.089486,-3.727583,-8.428507],[-8.663685,3.581474,-1.767358,8.858270,8.868318,4.335197,2.434087,4.195714,-2.517092,-6.759504,-4.470367,8.411121,9.782580,-6.997559,9.743215],[5.227779,4.713332,4.109058,-7.168692,8.489674,-3.766440,-3.340335,2.704583,1.476795,-6.925383,-3.058671,7.602422,0.917562,-6.637135,-8.865086],[-6.356258,9.086362,-2.684753,-4.247009,8.361293,7.476836,-6.763412,4.567832,3.441758,2.676365,9.773102,1.751552,0.632833,5.935498,-0.411559],[0.763810,2.908462,-9.226321,9.834034,4.313936,-2.994720,3.957523,-0.297714,3.127799,6.092212,1.805941,-6.899780,7.070363,3.167593,5.398802],[1.099557,4.979893,3.840386,-6.374672,-6.128079,9.686398,-2.350176,6.559090,9.327714,8.850743,2.645452,3.058032,4.922201,7.731004,6.445476],[-4.689611,0.019040,-4.724842,7.259574,2.818005,-0.992601,7.686791,6.191275,4.073449,6.632262,-1.258954,3.539027,-5.867155,-6.767990,-3.514145],[-5.146411,-0.353990,-7.155347,-9.029536,-7.380818,7.092655,-2.889014,1.533141,6.394965,-2.942969,4.074095,1.378746,5.684515,9.404741,7.385028],[9.226872,-8.594570,1.227342,-5.543609,1.036267,2.534780,7.642807,4.985147,-4.408547,2.514529,0.726032,0.982366,-5.680459,-5.253057,-3.082878],[8.183695,0.955775,-9.180126,-2.562976,-9.202569,5.459074,-8.488686,2.491896,0.096277,-1.553010,-1.205307,-8.231381,-2.990086,-0.361242,-6.077137],[-2.724832,5.864693,-6.150314,-3.731331,-8.468094,-5.170616,-0.443888,4.388645,-5.118820,0.647328,0.908265,-5.868163,1.751328,-9.166336,0.726705],[2.742460,-3.996528,8.597222,-2.916487,4.378930,-8.906584,-1.344058,-7.384540,-8.888870,-9.669202,1.613953,1.404426,4.654864,-4.005087,4.251646],[4.138574,8.666492,9.340782,-5.396223,-5.161484,-8.397639,9.396555,-6.243524,-4.918291,-8.663782,-6.530779,9.912827,1.707370,4.610778,-5.155811],[5.245001,7.448079,-8.905711,7.611837,-0.735771,-3.213960,-5.971694,6.792759,-6.875084,6.312953,-6.969785,-7.565347,9.514590,9.860655,-5.922683],[0.105707,5.952079,-3.732243,5.633974,-8.157740,-3.965952,4.899496,-6.350942,0.331054,-8.055591,0.920480,-6.692297,-8.636626,2.410352,-8.862880],[-9.947017,4.572162,-8.033686,6.709715,8.477110,2.004161,-7.796936,-6.401738,2.102398,0.614624,-5.218667,7.519774,6.347475,-5.698387,0.755095],[-6.333538,0.666035,1.642610,2.810771,4.435859,-4.949306,-6.036487,0.985304,0.769147,-8.738266,9.497188,1.399249,-2.497482,-2.383617,3.687489],[9.667224,0.562630,5.135311,9.155064,5.104327,0.106192,7.972908,5.329626,-7.132992,-9.546170,2.031207,0.572154,-8.736579,2.198283,-5.296770],[-9.430214,-4.695643,-0.026804,-1.225785,6.043726,2.147358,7.214918,-7.172276,5.472993,5.287918,-6.430267,2.714147,1.529316,4.703096,1.634025],[1.730569,4.658351,4.436566,2.455737,3.530993,-2.910892,1.777411,-8.564417,-0.288046,-2.778177,6.922491,-9.710561,-0.306623,-5.682325,7.374428],[0.640596,0.090553,-8.728902,1.814268,-8.263444,-2.577396,5.935600,-4.734434,-1.323639,0.550787,-6.376648,7.156437,-9.983710,-5.414628,5.319856],[-6.151674,-1.656619,6.880010,-5.454550,-4.770569,-1.446982,9.104052,-2.179569,3.510842,-7.481003,9.800933,5.162702,-0.354096,-0.952382,-5.748198],[-5.056074,-3.677069,-5.194873,1.869740,-3.477334,5.647288,0.762960,-4.021057,-5.596307,-2.266009,2.146806,6.108808,-6.749347,-0.537753,-4.109286],[-6.377338,4.383334,5.659229,-4.094495,-9.319592,-9.565337,8.167851,3.300223,7.250924,6.559314,-5.346873,3.015906,5.345262,7.148929,-9.207989],[-3.600318,-3.617511,9.945561,1.342090,4.331406,9.748902,9.075050,2.148159,-2.197280,-1.990426,-4.065700,-8.087998,-8.432070,6.098460,6.581442],[0.723297,-6.729307,2.025762,8.801246,4.783425,1.666279,-5.925789,-5.819481,7.624317,-4.627106,-6.354953,-3.493678,-3.842730,3.728338,5.791701],[0.862659,-4.348483,-1.295964,-0.429321,1.770793,-5.730289,-4.647209,9.838257,-1.303082,9.171563,9.254402,6.850839,6.575016,-4.385400,8.196038],[2.240250,4.304581,-4.734432,-3.619339,6.654873,2.386630,-4.253114,7.187405,-5.568418,-0.592657,-5.178107,7.586158,9.861355,-8.513957,0.333902],[1.818009,-6.932506,0.064285,3.584185,0.638570,7.360095,-8.991716,-3.902520,8.473209,-0.742795,7.570973,5.032902,-5.922217,1.480161,0.065471],[0.883321,8.315493,5.401847,-6.059149,-5.776409,9.016284,-1.412983,8.333565,8.838021,6.475564,-2.335515,3.597464,-8.673222,8.145542,-7.627896],[-2.642415,8.968764,3.318294,-8.061179,3.315218,-3.270141,-1.525043,1.078230,3.902873,5.646179,-6.671680,7.887469,5.395846,8.120703,2.695592],[5.949755,8.316602,-9.770241,3.580175,3.338864,2.161688,-4.223470,-8.186642,-9.715464,-7.650519,7.105006,0.673229,-2.954268,-5.111672,9.123406],[6.687248,0.039411,-1.580178,-7.774844,4.943282,-2.655821,8.554575,-4.519833,-9.060059,7.414775,-9.421017,6.216608,-1.910373,0.691312,-1.738239],[-4.884627,5.231212,-9.554330,8.364016,-3.729023,-0.789122,-9.849604,1.989895,-8.689189,2.154959,-9.869422,-0.358738,9.842802,-9.842448,9.178850],[-8.092256,-5.419252,1.875435,6.303889,3.832522,8.551464,-2.970911,6.993079,8.480651,-6.821453,-6.471995,-9.719315,-9.212201,3.632406,-7.826114],[4.058541,-5.309529,5.832966,7.070385,7.469107,-6.069857,-8.914492,-4.294071,-8.619030,9.598572,-3.241527,3.579915,-8.200238,-6.501227,-6.704861],[-2.615805,0.062955,-4.089338,5.893816,5.548154,-9.505383,1.592810,9.707630,4.806388,4.003394,-3.886908,-9.052988,3.794506,3.816005,-6.206463],[-6.592092,-7.627986,5.081157,-9.244480,-4.067888,0.383423,4.114886,4.043609,-9.747711,-4.258258,-6.513816,5.240306,3.323283,2.556192,3.380358],[-9.554199,2.115076,7.896715,8.977990,0.925987,-6.538799,2.556406,5.607218,-0.249769,4.047439,-3.288837,2.368751,-2.742484,-1.620790,1.215038],[7.523234,3.394003,-5.095615,4.341222,7.159606,3.810603,-8.324398,8.295544,-6.474735,3.035653,1.715911,3.486832,-4.262348,3.416729,0.967684],[-8.794092,-5.241795,9.219731,-4.144024,0.615884,9.584239,4.554727,3.053265,-4.475700,-3.558061,1.331777,3.214869,-2.642129,-8.899656,-6.066620],[1.489222,-6.159271,-6.657875,-7.333917,-8.743687,-1.718700,0.716502,6.813732,1.932023,-7.138414,-6.124654,0.059325,-9.370353,0.493674,6.320622],[-4.794689,-6.675207,-3.273378,7.664283,1.119281,-3.875928,3.038141,-3.854810,-0.352784,-8.535421,6.567104,-3.107806,-4.000175,0.461997,1.839745],[5.290041,2.768211,-7.569756,7.117063,-1.734290,-6.454231,-5.419657,-3.576926,-8.804838,-5.752880,7.985225,0.816661,-9.904037,-7.507777,0.139670],[-6.958672,-0.562920,-7.292583,-5.449188,-0.718693,0.251086,0.638677,-5.156822,-3.033471,3.671261,-0.199884,-0.940812,-1.065311,9.997875,-2.946638],[7.829203,-2.037046,-6.436625,-7.294764,-2.651433,-4.407586,6.815255,4.335244,8.983683,4.429592,0.985907,-9.250313,9.585565,-5.900578,9.255152],[-1.572704,-3.831797,-1.607533,7.456148,9.164645,1.833855,6.238712,0.278457,0.058103,-7.882106,3.154373,-5.459919,-9.985312,2.508520,-9.422828],[7.873753,5.072497,6.182149,1.715789,-9.423550,-0.366265,-7.051364,7.002286,0.650661,-9.706964,1.855388,9.375088,1.116908,6.883587,-1.914070],[-5.751853,-6.735837,9.396858,8.413022,0.006380,6.029762,-1.172131,1.954561,-1.133865,-0.255213,8.860282,-9.532687,7.729864,-0.215560,-0.814505],[-2.053058,-4.291322,-9.045896,0.578981,-2.919834,-1.071179,-9.797690,2.319433,-1.321464,-9.471917,9.531339,-4.649610,-2.940626,9.623546,-6.080286],[-0.315868,-3.257256,-4.020776,-4.473281,1.361698,-5.260090,8.907788,-3.312402,-1.684242,-2.420218,5.937948,-5.511658,1.835464,2.655320,7.198149],[1.025055,-9.252637,3.542399,-4.712071,-3.604574,-6.135622,2.665890,-1.029949,9.000386,-9.346634,0.265111,4.770705,6.480366,2.756052,2.607059],[8.996927,-0.163427,9.228626,-7.444040,8.395479,3.087646,0.797308,0.848134,-5.639631,-4.769171,-2.016009,-1.878467,-6.982079,8.825354,-7.316343],[-0.462412,3.837759,3.000096,-1.202538,-5.257685,6.233388,-1.678681,0.005321,-5.555775,-0.109777,-2.894422,-2.322029,-6.095178,4.626084,-2.445869],[-9.884823,5.093466,0.675723,-4.238066,-3.165828,-7.945089,-1.413468,-0.716230,-8.278330,8.221143,1.659002,-6.668843,-8.932612,-2.777995,-8.587616],[9.278231,-2.583674,-1.641894,-2.293707,2.365657,2.174085,-9.262792,-2.589300,7.664182,3.554378,5.849265,6.639804,-5.653681,-8.381202,0.926807],[-6.812402,-2.401367,4.762600,-1.859133,4.551162,-1.235150,4.328080,6.561393,9.587646,8.695430,1.013779,2.596929,-1.463526,4.311839,3.574305],[7.773781,9.115102,-4.554042,-5.292033,-8.423285,-4.878483,-3.280797,3.584475,-9.557373,7.698175,1.865699,-7.039989,3.803401,9.723058,-1.594935],[1.119315,-0.347966,3.230333,3.959035,0.033530,3.746665,-8.709359,-1.487852,3.413816,-3.032819,-9.756030,-8.182202,-5.326087,1.147949,-2.554138],[-3.814558,6.739310,-3.041035,7.235080,-8.483691,-1.505556,8.420267,5.091674,-9.833918,-9.759862,3.198724,8.095794,1.785014,4.473292,1.590867],[-6.686401,0.517669,-2.613680,-6.628645,0.950420,-2.536037,1.608949,-2.543048,3.157439,1.046635,6.522518,-3.924559,-5.591097,9.932701,-1.086130],[5.965699,-7.418668,-5.460056,8.356716,9.947956,0.470567,-9.022136,0.752406,-2.404260,6.300045,-6.545733,-8.554737,4.327627,-8.258515,9.853877],[3.785349,-2.255008,7.390508,-7.393678,8.741206,-8.311757,-7.602758,9.019187,4.265735,-0.655611,-8.894914,-7.204193,-5.037177,5.133117,-9.635247],[2.667171,-5.124075,2.674118,8.512012,-4.412119,1.636135,6.560773,0.370471,4.508127,-1.245794,7.757502,-5.026501,-5.262125,-8.372626,-7.063340],[-8.047449,-9.820745,8.590506,-2.509493,-1.179777,1.908150,2.860892,2.815527,-8.342225,-3.386344,-0.194024,3.549416,9.172513,5.736091,5.430562],[-9.828815,-8.594340,-1.438612,-2.738129,-9.939311,2.796456,0.157123,-0.433686,3.858439,2.422968,-2.993568,-8.197801,1.892947,9.944878,4.688316],[-6.945108,-1.705925,-0.399409,-4.921810,4.041737,2.573448,-5.095756,-2.095642,3.281293,6.900251,8.996397,2.068782,9.436256,3.751592,5.599418],[-5.487022,-8.088830,-3.016538,6.066102,4.936798,8.495604,-8.092448,7.734116,7.070637,-2.908247,-2.774060,-9.770975,8.384123,-9.312174,-3.930992],[9.349665,0.621631,5.429783,2.036572,-2.414411,5.630734,4.598093,4.448751,-3.708842,-4.779318,-6.383542,-0.108489,-5.973301,-2.780828,-5.600330],[-1.206820,4.615897,2.171217,-3.613562,-5.631041,-7.684713,-6.355047,1.711592,-1.280906,-2.175454,-0.994595,3.687603,6.925183,2.655060,-5.020309],[-5.403784,7.506319,-0.863813,-8.347264,5.160466,5.315025,5.981950,-8.422323,-2.827448,-1.643516,1.435033,5.038532,-0.597074,-5.743104,-9.961775],[8.673504,6.487517,5.642062,-9.631377,-1.525392,-6.937641,9.372760,-8.410804,-3.662197,-1.838815,5.785260,5.946845,4.044978,-4.525819,5.189373],[-2.978863,6.033926,-6.859030,-7.850281,5.014983,-4.992407,-3.168025,0.319343,9.711312,9.198491,-4.123916,8.944152,7.940473,2.424249,-3.110259],[2.067548,-0.506238,2.408219,5.099949,7.862288,-5.583688,-0.015876,2.734725,4.348477,4.093115,-2.871037,-0.786838,3.566105,-0.640441,7.802527],[-6.820824,-6.234545,2.990787,9.055195,0.178132,7.419968,-7.935021,-9.683140,9.144747,8.503782,9.685980,3.932341,2.485003,6.910266,6.908851],[8.388881,8.896988,-1.460709,8.395586,5.103749,-9.072830,-4.449748,-0.690277,-0.846780,0.385505,-8.901939,-6.677594,-9.903488,2.793231,3.016736],[-8.549407,-1.481086,-2.262830,-4.343519,-4.269612,-5.597444,6.775603,3.958862,-2.514026,4.267969,7.950277,-5.621313,5.553716,-0.470465,-4.118085],[-3.815782,-6.567786,6.031693,5.629006,-3.883682,1.504985,8.815782,1.841299,6.151271,-8.322729,1.670672,-8.127884,1.443063,-7.465122,-1.504252],[-4.937593,1.729666,1.160075,5.264109,5.458252,-5.901614,7.390502,5.457016,9.265784,-3.883377,7.216853,7.647441,4.370087,-5.742840,-0.363500],[-7.051184,9.441920,5.512866,-6.652922,5.457780,-6.303551,0.880897,7.946156,9.309948,9.261414,3.746193,1.056130,-3.657308,4.292275,8.382597],[-8.351894,-1.598775,7.691721,-2.263766,9.677378,6.407770,-0.500436,2.280863,7.304830,-1.730519,-1.691250,-2.073283,-5.392539,-7.339696,2.828467],[-9.182496,9.220608,1.766916,-5.762917,6.492537,7.475677,-2.564137,1.357908,-2.590298,-2.043552,4.591141,-9.136211,-0.560036,-1.318524,-5.244132],[-3.265438,1.814060,-3.117912,4.120491,8.756935,-2.160655,8.894477,-5.999138,-7.882534,-4.664110,1.516654,4.812436,2.240117,-9.428880,-3.133151],[9.773921,9.377490,-6.442494,2.653893,1.159322,-0.008087,3.038119,-5.492431,-1.114742,-8.387077,2.286074,-5.384397,6.699059,-8.434020,-0.682670],[5.104731,-9.275967,-0.116696,-3.921231,-8.481590,-7.555802,9.723340,-0.635736,6.478274,-8.217899,4.621163,3.283147,-6.147898,4.585606,3.540327],[3.681198,9.002386,4.282971,-6.595569,9.155503,6.642628,-8.117652,-7.513982,-7.600746,8.476024,5.545283,-3.069907,3.850319,-4.019786,-7.736385],[-6.760234,-5.377846,3.134182,5.888699,1.821911,-6.907267,5.662738,2.248531,9.511317,4.647169,-6.453453,-1.863792,-7.471594,2.163723,-3.305529],[-3.841547,8.924500,4.337634,-2.982764,-9.745764,6.463692,5.594457,9.703115,-4.634995,-9.786903,-5.926396,2.740953,-1.640881,-2.618456,-5.879882],[7.710969,2.158406,9.271919,7.988806,0.401181,8.589953,-1.507948,5.016499,5.164692,-4.817148,-6.558396,-4.025607,7.330345,4.073524,-2.276313],[0.013309,-7.351634,-5.376531,7.991880,2.436956,6.257265,2.965834,0.812986,-9.573145,0.737950,2.903309,2.445197,1.232822,-5.195938,-5.706691],[-8.394866,-4.647673,-2.978916,-0.215359,-9.425716,-2.513401,-8.142520,1.523355,9.939986,-4.668540,-0.432911,-7.571813,0.251215,4.941079,-5.443010],[-5.572113,6.577836,1.572135,4.925401,9.496436,8.423367,0.040645,9.241471,-6.298066,-3.840989,3.070753,-2.600907,-4.808153,3.164628,5.184125],[6.451583,-9.740487,-5.819462,-5.311058,3.169856,3.454832,-7.751133,-6.230138,7.570923,-3.363312,4.241723,-7.811754,0.254985,8.230522,-1.362575],[-1.741577,-2.811693,-6.218391,-4.355067,-9.639561,-8.846420,-5.723652,9.658043,3.272724,4.647829,-5.956264,4.473291,-5.337510,-8.905670,-2.995009],[-2.371488,-5.977716,9.593935,5.128498,-5.419508,-2.425415,5.942168,7.223395,8.693735,7.686342,-7.258802,-9.317109,4.652840,-9.724094,-8.659484],[-1.996012,6.872372,-4.823817,-2.741610,-5.796819,5.129367,7.557756,2.013711,1.823392,8.191006,-3.847241,9.631667,-6.742299,9.779242,-4.076604],[5.624485,8.743627,4.759736,9.705407,2.286379,-8.898667,-0.078136,-6.769177,-2.958796,5.548093,-6.650648,3.847693,-1.176498,-6.491781,-4.274319],[4.067161,-1.473472,-4.245817,-2.680639,8.334752,0.750083,9.611872,1.001586,-8.590567,5.373289,-0.479638,0.907190,9.951047,1.658686,6.766071],[-2.739339,0.958823,9.979042,0.031356,1.502297,-7.248604,-8.584131,2.809363,7.792249,-5.120936,4.513527,-3.020752,-4.053073,-3.732984,-5.515578],[-8.650370,-7.460146,4.336297,6.763918,-2.764049,9.063909,-5.333066,-5.328822,9.638140,1.249448,-6.878329,-9.801351,9.265499,8.536496,-8.681134],[3.965397,-4.274804,-5.778841,-6.622025,-1.954643,6.061211,6.686224,-0.747371,6.569376,2.149964,0.324881,1.552923,-5.687245,-8.870570,-5.302812],[6.841094,8.843214,1.038628,4.758546,-2.710269,-5.130890,4.060115,5.028090,3.967742,-2.473770,9.928468,1.393191,2.160740,2.006003,1.179059],[9.672158,4.322247,-0.046614,-7.698925,-1.445975,-4.019920,-0.864023,-8.162758,1.214257,-8.016595,-4.029824,2.690647,-1.890320,-7.769093,-5.960198],[9.050042,-4.412282,0.938162,-1.531198,9.035489,-7.772015,9.270094,8.020113,-9.123642,-1.125019,-4.202024,-3.814616,-3.403448,7.203114,-3.316872],[-5.366699,1.577161,-5.195854,3.796564,-9.792380,-2.918564,-4.453786,-2.578272,0.551247,-6.886763,9.820531,1.504449,-0.091310,8.465288,-8.668960],[-8.067588,3.971423,8.176749,-3.734008,-3.354657,5.262691,9.811111,6.335015,-4.416417,-0.591559,-9.959792,8.261281,9.072990,7.442357,0.080787],[-5.498081,-9.127345,4.109856,3.409447,-5.183198,2.114442,-6.292716,-5.217267,-1.424840,9.514768,2.376812,-5.259695,2.537602,8.226201,-0.407597],[-3.602480,-7.706202,1.481590,-2.493194,7.961530,-6.959430,3.706810,2.847817,-7.041265,-4.252152,-0.664952,3.153011,-7.584015,-2.318995,9.792652],[0.878949,0.551401,-4.609198,7.608759,7.248177,-7.471739,6.364122,1.322569,2.021834,-9.513131,7.027971,8.429021,-0.029327,9.989026,7.901872],[4.430850,-4.343165,-0.851693,5.903397,4.919867,-0.203258,5.084369,0.557568,-2.667018,-7.922326,-3.337452,5.677160,0.245289,-7.014148,-5.259931],[8.913315,-4.174154,-0.272029,6.914830,-2.642118,1.548246,-1.898922,5.288582,2.992843,-7.324059,-5.635942,3.237823,-4.500958,7.980715,7.607970],[4.203973,-4.865979,-5.744006,-2.109573,-9.745337,-0.739252,-7.713074,-7.648656,1.699721,8.439956,-5.855113,-4.534238,-5.479479,-0.649664,5.497412],[4.617360,-8.380372,-6.049025,1.064400,-6.298901,-1.275378,2.096740,7.901760,-7.100049,6.163981,-4.873110,9.135007,3.411404,-4.197862,-3.231727],[1.133773,-1.972068,8.254413,-2.014359,2.611232,-6.253419,4.410595,-0.013067,-2.644156,1.520495,1.664424,2.313881,1.583575,-0.223187,-8.123364],[8.210878,-3.571066,0.740335,6.401743,-9.831378,4.722564,-1.673204,-2.657914,1.420286,0.097185,-8.948690,6.406591,9.130351,-0.667090,-5.261914],[-6.229325,-1.373143,0.850300,-8.541053,2.490020,7.529683,-6.946763,-4.511926,3.334815,1.642808,-7.587381,2.865342,1.725114,-9.460916,-2.337564],[-0.181241,9.681855,6.851701,-4.149648,6.464254,9.938220,-9.063591,-3.386183,-1.613455,1.357612,-4.822559,-9.798938,4.619343,-4.794451,7.445382],[-1.128943,-9.086062,9.077266,-2.947008,6.743565,9.296548,-9.688234,-9.155800,-6.593292,-2.368326,-1.828845,-1.363235,-5.963455,-8.694630,8.781571],[0.026657,0.054831,2.102529,-9.692010,7.946697,6.284182,-6.783557,-6.835962,-8.371537,0.256607,-0.381624,7.416911,-0.612406,-8.410050,7.614398],[6.245473,0.505497,-6.882097,4.786281,-3.439301,-2.061899,6.791947,-4.552862,-6.410841,9.614493,6.065387,6.893511,-2.975133,7.046888,8.175494],[-5.610504,2.462270,6.883795,1.663823,-9.630407,3.549600,4.261437,8.198795,8.119798,2.856470,-2.373139,-1.252763,7.408400,0.436046,-4.553789],[3.177148,8.514668,-7.311226,-7.023165,9.739021,-7.608712,-0.936611,-6.279760,1.898557,4.781714,-1.866050,3.637074,0.049337,9.126996,3.202581],[3.415205,-6.226794,-6.074131,-3.643820,-0.100452,7.506643,-6.208616,8.945484,-1.160824,-2.988600,3.251020,9.784734,-2.582892,5.880786,8.756964],[0.037352,-5.090474,-4.114494,-1.971382,3.200713,-6.061554,1.976441,-9.511596,6.576203,0.377399,-8.643323,-8.952625,-1.288550,6.608342,-7.965602],[-8.218589,4.091376,-3.475531,8.949631,-2.747309,1.715823,-7.753153,-4.094475,3.626909,6.665412,5.906705,2.713165,2.572377,-1.296973,-2.598254],[9.331059,-3.342813,3.704196,-2.636041,-3.564978,6.499616,-1.653003,-1.303162,5.231342,-7.757325,4.367678,-6.239470,-7.973893,-5.362859,-9.740530],[1.284368,0.357468,0.040959,-1.478140,-3.160977,-3.464564,3.307858,6.260238,4.372086,7.718366,7.947865,-9.788286,-0.322147,2.222313,8.123450],[3.769838,1.492114,7.491222,1.857353,1.635768,-7.978476,9.020221,-0.434056,3.698151,0.938123,2.174582,7.099599,6.014813,-7.707656,-2.659461],[-7.636101,2.158204,0.697225,-2.952544,-5.127068,-3.240418,-9.816667,-1.023356,-8.827762,-9.907070,7.782968,-2.538873,1.553120,-3.263899,-7.047314],[-1.850023,-6.895406,-8.708398,-2.368876,-3.233192,-7.772092,8.858169,-0.408318,3.136631,9.585221,9.463324,-7.489259,-2.159065,2.758159,3.332392],[3.130460,-9.739039,8.041977,9.943165,-3.818363,8.816654,-9.322497,8.449946,-1.173593,0.892627,-1.253239,-2.628276,-9.520840,1.862608,6.051623],[4.257178,-4.354854,-1.583843,1.497065,0.220463,0.159364,-4.994977,4.213248,1.524043,4.286753,2.999886,8.693082,-1.683647,-7.541738,3.935997],[-2.721535,4.982211,-2.647568,9.985225,0.836750,3.986650,7.402519,4.219687,5.521143,-0.187238,5.956766,1.281987,-8.191499,0.068063,8.676725],[-6.394047,2.341203,-9.165207,-1.754623,1.714153,-8.903389,-8.638507,2.321817,-9.572990,-9.025105,3.081137,-1.145972,-2.865685,-0.467971,-4.885981],[-8.858543,1.996210,-9.207335,-2.500330,1.854887,-3.942152,-4.653191,5.157399,-2.284968,9.363194,1.737443,2.429793,0.900357,-7.258609,8.872887],[5.508340,1.735483,3.110864,-9.732879,-9.377433,-2.639559,4.874710,-1.414889,-6.702419,-6.372616,3.008539,-9.450940,5.358256,9.803525,9.881729],[-3.453996,-6.376121,4.473470,2.336335,5.164014,-8.930186,2.229714,1.369227,-2.653428,-1.156322,4.768058,-5.196159,-2.243493,5.018491,9.621563],[-5.155039,8.100859,-8.853616,4.035488,-9.062879,5.676496,-8.319482,-1.087861,-2.469904,6.378813,0.252637,7.586244,-8.422821,8.639913,2.170130],[-5.874695,3.754160,-9.315829,-9.763708,4.518442,8.586824,5.434283,6.919271,5.550548,0.565058,4.775893,6.282888,-9.066064,-9.476437,3.394504],[5.523980,-9.712348,4.985394,-6.119990,-1.552195,-9.427857,-0.563657,5.111934,5.083284,-4.612386,1.120286,2.257362,-6.978937,-9.941836,7.030166],[-2.907463,1.201994,4.735861,7.608206,-0.805428,-4.196218,2.870482,3.173587,-5.274458,0.594898,1.523757,-5.476130,2.717607,-3.342260,-3.817693],[5.344686,-2.349404,9.966498,5.277571,9.592469,-1.172961,-9.607582,3.920104,-7.011836,-6.137213,-1.077908,-4.933126,-1.720898,9.578190,8.297114],[6.365214,-9.381409,0.613034,3.747615,-2.862628,-7.111843,-4.809070,2.001066,-5.764517,-5.178214,9.178511,-4.406393,4.471318,-9.214250,-8.252614],[-5.809217,-6.319387,1.152629,-2.101235,6.619888,-5.597986,9.035888,-0.412614,-0.784033,-3.852664,-0.921569,9.837919,6.147672,1.126033,3.329280],[9.911056,2.080736,-9.938441,1.394675,-8.999342,6.604675,8.939940,8.081812,4.101093,-5.925934,-6.111117,-5.387800,-1.231452,3.768992,6.982387],[8.179140,-8.802720,-2.672818,9.597405,-9.264562,-1.018628,-9.372346,-5.468213,-9.466714,8.168689,-4.649925,4.444180,6.500101,-1.550285,4.797872],[2.579346,1.451613,-7.526499,-6.561038,8.964362,8.756598,2.430741,4.060870,0.438895,1.493862,4.500684,7.418192,-1.215544,-0.949892,0.459037],[-3.618573,-2.743208,-7.182230,-7.561311,-3.418017,3.470335,4.972102,-1.259463,0.872342,-9.123065,-6.315750,-6.233131,-6.840981,6.299273,4.010092],[8.146969,1.462924,8.956399,-7.982533,5.574938,-3.138674,-4.292114,9.686028,-5.368034,-7.836150,2.751997,-7.542530,-3.671130,1.209917,6.392400],[-5.048105,-4.297724,-4.898212,-4.474152,5.772187,-0.416189,-7.955297,-7.337523,6.612951,-1.847262,5.998747,-4.061213,3.817976,6.615288,-8.389753],[-3.398361,7.244157,0.335630,-6.959247,-2.831751,5.209543,9.564581,9.734608,8.173894,1.451126,-2.115032,0.804866,-2.107417,-3.126170,4.885382],[1.404331,7.599062,9.533602,-2.929928,-7.728081,8.724031,6.418306,-3.721178,-2.559684,5.592530,5.816136,6.656256,0.633341,-9.990973,7.897890],[-1.304212,7.291077,8.071456,-3.157778,-2.699436,-3.402498,-2.979188,2.482605,-9.336078,-9.859085,9.967578,6.115080,2.811508,-9.485575,-8.288888],[9.436741,-6.353933,0.121490,2.162873,-9.248256,-1.199871,-9.191050,-3.079551,8.618129,6.947267,6.037888,-6.873102,7.469953,2.422157,5.565157],[0.305029,-2.180924,7.189602,5.314527,-5.122803,-1.667749,-6.728084,7.903826,-0.856504,7.175859,2.025598,-5.846145,-0.786812,5.238854,-5.325412],[4.107493,0.615238,8.709364,1.522083,-5.831377,-6.427245,8.124070,2.925284,-1.727989,4.334114,1.206781,-0.935455,-8.349069,-7.867565,-4.513854],[-0.428522,-1.468224,4.304526,-4.663453,1.088830,-9.275778,-2.113217,-1.658345,4.445337,6.647445,8.471448,6.576292,9.545705,-4.170994,4.063681],[-4.857503,-9.321616,1.882698,-4.345585,1.001068,-8.911371,-2.851924,0.096078,-4.762215,5.270642,0.234280,6.989574,-1.751642,6.129266,-5.987124],[3.597211,-8.948615,0.051326,1.543958,-2.533806,2.542031,2.258206,1.872599,3.314301,6.752359,6.800296,5.145695,9.257086,3.252795,-0.685120],[3.255396,-3.194111,2.930718,0.477225,-4.061438,5.468197,8.539307,2.811832,1.776994,-2.456057,8.603731,-8.161367,-6.184467,-1.932215,-3.212650],[1.818760,-1.202624,5.934416,-9.209642,-2.962200,-5.410005,-3.415974,-1.533504,-6.766681,-5.223178,-3.095990,1.102734,-4.959812,-9.074586,1.689454],[7.686003,-2.908357,2.551550,-5.101325,5.898399,-1.896196,-8.144747,3.615437,-9.394849,7.077211,-2.748630,-8.017668,-7.180031,-0.674963,0.952340],[1.459712,-9.708390,-3.462126,7.532613,3.679604,9.991002,1.052445,8.772849,9.284083,-2.006221,-4.630850,-0.042566,-3.686888,-4.769982,-1.895007],[2.890642,-5.595543,8.338125,8.834243,3.960807,0.621489,-7.744878,2.867557,8.552119,3.312656,6.067095,2.043670,-5.033287,-0.501616,-4.909676],[-5.499627,-9.329626,1.000597,0.219460,-3.867438,-7.381547,1.735661,-1.021759,-2.572698,5.682193,-5.330584,6.263814,0.386340,5.187165,-2.034424],[-3.638327,9.651718,-0.783500,-3.627813,6.526258,-1.667871,7.236774,-5.129951,-7.062795,4.043353,-7.354790,-0.393323,6.877479,2.007539,1.137524],[-4.724228,-4.381305,-9.898134,9.246361,-7.329521,-6.951005,-6.258080,7.293093,9.417108,6.889572,3.414515,-6.661210,5.700575,1.476289,1.321716],[9.403515,-0.333567,6.099646,-0.503682,-6.689735,-8.823306,8.115049,-2.563027,5.497441,8.193222,-6.267678,2.543055,-6.106974,6.338135,1.406465],[2.188209,-5.741481,4.837247,-4.654225,-7.810382,-3.062165,7.685789,6.981509,-7.194404,-7.183082,9.899517,5.775825,6.698104,-3.222855,8.116122],[3.648509,-0.994256,3.375378,-3.822450,2.724003,-3.796360,7.382503,1.084059,2.778923,9.757367,4.410833,-2.757594,0.363450,6.189879,-1.074288],[-3.274376,8.781015,-8.941509,-4.963152,-9.671558,9.358597,4.541106,6.013209,-5.812603,8.703581,-2.179436,-9.577364,0.468132,-6.060855,6.277212],[1.800236,0.022631,-6.243038,8.074103,-4.192961,-7.727312,-6.277209,-5.223548,3.296202,-7.217833,-6.971096,0.323477,-5.473280,5.517919,2.989067],[0.086040,-9.194963,0.781158,-7.254985,4.270004,6.825535,2.194853,3.273408,0.677718,-1.909197,7.981199,-1.040812,9.335592,8.316716,9.731068],[-6.752957,-1.827374,3.596278,4.227110,-5.332199,-9.196899,7.745185,-6.296158,2.288934,7.391986,6.156350,7.290811,9.169126,-0.932937,3.732110],[-3.058463,6.878653,-7.257797,-0.618715,-0.775576,-7.309906,-4.983954,-7.460678,-7.577638,-2.292867,0.689750,8.571505,8.981260,-4.757631,2.351279],[-9.116346,0.159527,-7.751704,-3.105174,7.944328,-0.602100,-1.643714,6.468536,2.621618,-0.202334,6.018402,2.124284,-9.669143,-6.194173,6.850219],[-0.712331,4.959708,0.425090,9.778767,-5.434903,-0.269855,-0.861630,-6.781130,-8.819846,-6.582166,6.476249,-7.113844,-4.040931,-2.102723,3.003054],[-3.961996,8.449583,5.439163,-7.310963,-5.102400,-0.271653,1.071166,2.620532,-6.120442,7.623787,4.946705,9.305584,8.569478,-4.505226,-1.581407],[4.901231,-4.814318,-6.721421,1.947681,-0.972713,3.448901,2.447459,-7.528704,-6.443338,-9.837624,5.171530,4.475600,7.562627,-8.330722,-6.995703],[-2.626210,7.473112,-6.623291,-9.430585,-9.286500,2.343954,1.361385,-4.973549,-4.676656,3.524028,-4.410844,-0.998499,4.783282,9.085551,1.969842],[-3.992084,3.505868,-7.601014,6.972270,-0.441718,-6.157361,9.294185,-9.077249,2.919418,1.380414,3.846944,2.170147,1.919797,9.391383,-1.483468],[-1.799652,-4.812958,-2.727513,4.749501,-7.055201,-7.988613,8.341269,-3.538621,8.311250,-2.475367,-3.128195,7.316875,1.777005,-4.064784,1.750248],[-9.375231,8.330504,4.170307,3.764828,0.811274,-1.964959,-9.595918,-5.915127,0.309729,-5.921327,-5.595598,2.114224,-4.042001,-2.809591,-6.706548],[4.143384,2.463204,8.123497,5.445734,2.364872,-4.762686,-4.240351,7.248572,-9.501360,-6.775678,8.559396,6.068932,-8.833383,1.949556,5.052004],[-2.992857,-3.412400,7.880930,7.921524,2.756644,-4.090434,-8.117784,-1.117152,8.424856,5.756918,-5.470015,-4.211936,7.391935,6.933868,8.177719],[-7.578334,-3.742601,7.861719,7.194713,-6.142184,6.456063,-3.011330,9.233450,0.924228,1.996581,-5.601422,0.327032,-0.182854,-5.818745,-0.674647],[-1.034921,5.398771,3.130899,-9.731609,0.087287,7.071419,9.749302,6.698911,4.921278,2.784659,0.793401,-0.574031,-8.726589,3.630528,-0.825785],[-7.511485,0.061107,9.912398,-3.869517,5.041012,-5.452446,-0.093899,-8.819639,-2.293056,0.812039,2.599679,-1.727565,2.825896,-2.767293,2.249924],[-5.983919,9.285404,5.764122,9.500900,-5.292206,-4.220326,4.630345,3.131084,-1.687008,0.014198,9.836409,-9.792135,-3.254758,3.620246,-0.408969],[-0.342390,-3.954105,-7.272464,-3.148184,-5.179719,-8.412456,-3.920197,7.228353,-5.454629,2.455592,8.877945,-0.734921,-9.591800,-2.723350,8.471994],[-0.191340,1.021463,0.132811,2.289309,7.458442,-2.522434,-4.226679,-4.362796,-3.413212,3.350456,-5.911709,9.456094,-3.602722,-9.876114,6.278325],[1.344812,2.816847,-9.583985,1.425127,-9.975155,9.728209,-5.375499,7.057152,-9.047256,1.554321,-4.538868,-0.150985,-0.474347,1.245529,4.025632],[-1.626422,5.089321,-2.582148,-4.559004,0.274226,-8.701272,-0.488993,-7.158794,3.348338,4.201726,-3.330679,5.840162,7.521681,7.863271,-8.030313],[3.517805,9.358508,-2.037452,8.798700,9.546434,0.734312,9.826415,6.796399,-0.469783,-4.614350,0.695572,8.320693,-5.847054,0.673773,5.502572],[-1.255275,-2.726573,7.350452,9.709054,-4.839473,2.245102,2.732893,3.313353,-6.735763,4.487501,2.713246,6.100536,6.506545,-0.137748,4.919043],[-4.591962,-8.894917,-4.195171,4.078705,-6.894371,-4.904783,-1.500196,4.453396,-6.099822,1.428019,-2.885140,2.435223,8.832190,1.929371,8.816930],[4.914943,2.223646,-9.252159,-6.038217,8.760958,0.978585,-1.706039,3.318932,0.683204,-5.564474,-3.083949,-4.476278,-2.795314,-4.264841,2.888107],[9.416261,-0.971340,-4.122007,-2.094389,-6.686860,-9.324723,4.047025,0.200987,-3.144696,4.050282,6.005102,9.829765,4.117591,4.296906,-2.370526],[3.601404,-9.818107,-1.201516,-1.275233,2.269117,-1.211119,5.062492,-8.225625,9.910884,2.820971,2.218587,-0.370101,-9.120922,7.066350,3.636250],[-7.750672,8.990101,-0.715654,-1.908994,-8.710482,-5.671876,7.087053,-2.003584,-6.861482,-3.397988,-7.073729,2.905779,-9.375787,-0.723319,9.321671],[3.126459,-5.836730,-2.424254,-1.569011,-2.282890,-2.913123,-4.727089,-6.730563,-7.208040,-4.285425,-3.425551,-2.261355,2.316984,-7.102686,-2.770263],[1.505631,1.340192,-9.747573,6.549126,-7.148267,8.372722,-7.646098,-7.927033,4.834498,4.368453,1.094032,-9.191202,4.767411,6.281000,-3.289162],[-8.686069,1.039730,8.943533,9.538835,9.944507,-1.708125,9.970541,-4.222579,8.560815,-3.699186,3.353474,4.054732,9.607448,2.351829,-1.469628],[5.519838,8.993666,-9.322063,-9.168507,1.322550,-0.615786,8.531797,-5.381320,8.268467,7.034954,7.024135,-0.923818,-5.140774,7.285420,-5.292029],[2.866041,-9.260060,-5.552627,-6.186789,7.190083,-9.827375,4.765766,9.360033,6.897939,-1.252880,-3.776062,6.681454,-5.451445,-6.743937,9.352535],[5.262283,-1.883556,-5.903441,-0.303123,-1.200397,-0.465668,5.333806,-8.505796,-7.630348,-3.805683,-8.433092,-1.269014,5.467301,1.343479,-4.509384],[-8.113471,-2.354431,-2.466286,7.577477,3.868320,-4.395054,2.809578,4.191735,-8.568384,3.951645,-6.921797,4.175105,-7.972485,8.237251,-0.017593],[-4.331469,5.143720,-6.115978,8.799637,2.397102,5.434818,0.696283,3.851097,-2.254665,-8.921828,-1.079758,3.153595,2.630104,-2.998673,5.462931],[-6.326739,7.816150,-7.494558,6.864669,-4.385869,-4.236379,-0.504494,-3.753098,0.725804,5.183968,-0.267616,-3.047208,5.733439,7.542884,2.677146],[5.848263,3.593845,3.664827,-0.066548,2.151210,1.558996,-2.607833,2.146617,-5.417457,7.777302,-0.810196,8.086013,4.989447,-2.477901,3.602812],[5.482973,-7.334455,7.871217,1.654131,-6.486298,-2.043430,-1.416251,-8.550527,1.627505,-0.364773,-2.491037,1.061030,-5.259512,9.248717,2.084721],[-2.273456,3.966731,9.429561,-2.262210,-7.300148,-6.510696,7.296226,5.901649,-4.489611,-8.660135,3.913707,-6.550310,-9.420299,4.956172,1.905256],[-9.317760,4.968976,-8.455474,4.664751,8.548397,-7.230132,-5.140707,2.564775,0.529924,6.817862,-3.552220,-2.012365,4.251792,6.329346,-5.944648],[3.418946,-9.016990,-5.866087,-9.213838,-4.919740,4.335220,9.698168,-8.097287,4.128223,-1.792700,1.432880,2.468739,9.850923,-3.833613,4.921163],[-9.175867,-5.551182,-3.845536,8.193400,-2.801917,-0.860350,-6.840143,8.319048,-1.527406,-9.062127,0.383897,-6.847376,-9.944114,3.453520,-0.022723],[1.199866,-6.150783,-1.569145,6.157005,2.964497,3.126201,9.729198,6.724275,-3.224861,-5.398212,-8.273937,-0.119659,-4.914573,8.758241,0.929658],[7.620140,-1.439222,-0.144020,-7.623439,-1.654902,8.360163,9.577680,-2.085308,-2.578969,2.297094,8.151920,-4.800335,-1.447166,-8.086028,2.480537],[8.645984,4.784464,-5.447826,-4.722243,1.078038,-7.856886,-7.848560,0.254203,-6.745369,1.629927,-0.925364,2.306121,-1.130661,-9.601019,4.871601],[3.176306,-9.863392,-1.646344,-2.852134,4.127525,-0.122486,4.466257,0.402584,6.990224,-8.477633,7.036459,2.370251,9.610027,-3.137298,4.137331],[8.138144,4.536860,7.530213,-4.439424,-8.148318,1.158166,7.729788,-8.937368,4.228934,-4.133698,-9.865951,9.825279,2.574889,-5.273475,-0.175475],[5.787504,-2.770790,6.029258,8.923635,3.631098,6.769023,-0.429656,2.360791,3.496535,-7.025754,-5.134543,-0.498701,-6.899083,1.623880,8.422616],[-0.840590,5.262058,5.279434,-1.624856,2.765583,8.874828,-9.457056,8.023393,-3.343376,4.953180,4.472880,1.200246,8.087794,9.566482,-0.191469],[-6.418516,7.673538,2.020287,-0.439003,9.998964,-0.312436,-2.303179,-5.701527,-3.647180,-1.349865,8.018359,-4.709152,6.796955,6.488686,-7.661156],[-0.944231,-4.311391,-5.936402,-8.079123,-1.387085,-1.814833,-4.742599,2.803869,6.557187,9.964617,2.079592,4.054077,-9.348699,4.217700,3.100887],[7.697802,-4.685755,7.871848,-1.141418,7.238894,-1.386770,2.804785,-2.884205,7.852231,-3.906803,-6.753861,-7.425029,3.217687,-3.688611,-4.309228],[-8.979888,9.494865,7.247024,-7.864568,9.295316,-3.990109,9.490915,-1.863147,3.598202,8.455727,9.389650,6.724973,8.646400,-6.031756,-8.714658],[3.142258,-0.084089,0.339058,-9.638255,1.855592,-0.817774,4.800488,4.418106,5.612149,5.643152,7.309008,-7.861392,-1.723932,7.325626,-4.890682],[5.227838,5.763403,-0.590678,-6.820910,-7.855474,-7.038375,7.734863,-7.951843,-6.669841,4.489429,4.110475,-6.341253,-1.890920,4.082729,1.626386],[3.321160,-3.567850,-8.755270,6.066860,8.675328,0.498063,-9.349912,8.086955,4.951215,8.467476,7.493997,4.082075,-5.397953,4.653847,-2.182684],[7.046439,7.101253,-0.312570,1.796309,-6.447783,7.662555,-2.581012,1.236087,-0.121187,1.322979,0.415512,-1.140409,-3.563182,-6.609426,7.870397],[-4.704789,-1.774538,-0.596384,-1.054242,6.300927,-9.094935,-8.508156,-8.125117,-8.159540,-8.862221,-5.362112,-4.549032,5.236877,-5.918768,-4.872876],[-2.548441,-1.713579,5.190149,7.667769,-5.975486,-1.568118,1.191769,-1.481189,-2.412938,-6.521483,-6.766029,6.074438,-3.336154,-7.874678,-4.964660],[-1.993903,6.698572,-5.170737,2.108462,-8.967126,-3.195295,-8.616892,3.352848,1.014588,9.954712,9.338179,1.157824,-6.380189,8.237113,-1.513559],[-4.767293,-5.745113,6.764145,-2.969306,-2.687422,0.876132,8.070275,-1.678513,-1.151487,7.607347,4.565739,6.779998,7.704506,-6.908475,-9.097580],[-7.181168,1.124945,6.114468,2.429012,6.549308,-0.047370,-4.401133,-9.408887,8.060556,5.800871,-2.993428,9.231301,0.346636,-5.820992,7.427170],[-8.787259,0.680715,-1.625432,-5.452988,-5.308045,7.899275,-6.442080,-5.883308,-1.880391,0.149730,-3.903928,-0.744705,-2.983073,-9.958181,-2.592399],[4.549340,-7.042544,-1.904778,1.090542,-6.082891,7.662947,4.791250,-7.236647,-8.644590,3.262020,-2.980104,-0.640935,1.662355,-3.022328,5.178892],[7.532642,9.651431,2.857458,-6.099185,-6.080731,-5.465488,-3.199488,-4.230676,6.363606,0.533320,-8.868319,9.610211,-4.583658,1.928010,-8.915931],[5.669994,-2.882234,5.283881,3.146569,-5.792851,8.105743,0.551734,-0.416396,-2.444587,-0.909959,-4.722820,5.133229,7.427898,-4.595921,-2.075077],[2.609592,-4.209556,-3.351553,5.875302,0.155332,3.940333,-5.708423,-5.158095,-8.145765,9.733320,-6.126529,-4.529064,0.050400,-3.994631,-6.047415],[-1.358371,-9.516087,3.811239,6.959130,-2.852647,2.918361,5.078146,-7.657668,-6.128176,-8.970209,-4.288714,-3.218211,-7.248084,-8.832964,4.999238],[8.327520,9.191386,9.543656,-1.235539,8.614126,7.803503,-1.934829,1.197842,6.661769,-0.868779,9.160222,-6.911341,3.516427,3.410422,0.876693],[-9.677610,-1.193533,7.650519,6.813271,-1.062773,-5.876593,4.656527,-8.575924,-9.533223,-3.060972,9.980175,9.702173,4.870202,-8.366429,-8.211105],[-0.923343,-0.911606,-1.562697,-3.681595,9.060914,9.713016,-5.994861,0.795680,7.957239,-4.281862,-3.208691,4.638160,-8.899270,-7.247219,0.920890],[-2.166538,1.952922,-6.904915,-1.534599,4.491902,-7.855142,-7.379495,6.880581,1.416441,1.407313,0.375969,-9.250394,8.073253,2.592662,8.335618],[-2.651220,-4.348316,0.473133,-5.610568,9.458211,2.690553,-5.609513,2.498938,9.593890,-5.486890,-0.164897,-1.291867,-8.981574,4.934739,6.814823],[-7.134060,0.132163,-5.631628,6.760360,-8.400019,-0.265006,-6.832315,4.962192,7.831916,9.187822,2.905006,7.620612,-8.275453,1.723010,0.516537],[-3.395041,1.009850,0.253092,-4.648726,-8.246296,5.175971,1.117736,2.744750,8.199604,5.591475,-7.250064,7.799104,4.611544,9.921176,8.924198],[3.233013,-8.826205,6.925032,-8.036407,-7.024513,2.812690,-5.087700,-0.431405,3.906696,-0.845185,-0.820462,-6.261570,-2.825552,5.286358,-2.214292],[-0.507419,-9.615080,-8.223593,-3.930134,-8.297485,8.868854,9.324115,-1.898730,7.820409,2.362542,-3.865392,-4.897501,0.575937,-8.031322,-9.653714],[-2.130152,4.998177,9.892158,-5.045372,-8.816235,6.709321,-0.247259,3.454637,-0.343595,9.885684,-1.681352,5.146506,-6.588796,2.507876,5.526477],[2.575681,9.506026,-5.127251,2.797169,-6.803411,1.332630,9.984161,-6.141322,-6.294166,-2.818393,8.492134,-2.117971,8.582174,6.858775,0.361407],[2.929917,-4.588215,7.808522,2.815913,-2.629491,6.269239,1.204547,9.482808,7.842445,-4.502550,6.695673,8.555398,-2.939367,-5.260815,5.932676],[-5.214463,9.807502,0.705610,-1.833255,2.872879,9.029036,-7.301281,8.802055,-3.391423,-7.445698,-1.631736,-9.164729,6.939520,9.053900,2.697345],[5.772099,6.272370,-1.976721,-2.890342,1.285237,3.631132,-2.168601,-1.260781,-1.643413,-7.004599,9.560833,-7.006428,-4.853909,7.314740,-1.675634],[7.819335,-0.658361,-4.308374,3.976439,1.696812,-9.170045,3.070594,2.141631,-9.511471,5.800155,-5.338116,-3.129862,-9.717126,-0.483969,-0.829446],[-1.144161,-1.696222,-1.528092,-2.803325,9.015343,-0.609429,-5.880442,-4.210974,2.836663,3.972393,-4.719107,-4.772979,0.833541,3.193660,-3.303309],[-8.364146,7.127162,9.033785,8.942369,-7.937234,-5.235816,-9.382765,-6.990352,-9.114343,3.775915,-8.946098,5.700988,-5.680653,-2.381392,-4.552871],[5.844285,6.890194,4.261007,-9.113924,-2.399504,-5.280321,-8.897623,-7.923962,0.450795,-7.365778,-0.806948,-2.819277,2.183106,-7.507688,1.884222],[-3.199792,6.583425,9.984895,7.235121,1.651389,-8.414419,-9.888299,-6.644825,-0.037667,3.412591,1.883106,5.503961,0.513663,-1.829502,-1.028838],[-4.391639,4.270486,1.351016,-6.829979,-4.094925,-1.403897,2.617535,1.022627,8.010602,-5.919590,3.336777,2.753687,5.234680,6.323550,-3.537720],[5.213645,3.590399,1.726230,-6.024782,0.138646,1.556617,8.863929,-6.024759,8.307872,-4.609251,-3.682974,0.372900,4.525552,-8.111101,-2.469455],[-1.528306,1.067460,-9.093845,-3.380997,-4.512499,-5.881224,-4.914818,0.411680,-8.011056,-7.847673,-5.018148,-8.582248,-2.013719,-8.124206,8.478629],[-1.033407,5.199550,4.415272,-8.313897,-0.047624,3.614291,-2.056963,-6.509295,-1.478914,-1.419386,-6.979946,9.695078,-7.156277,-9.585480,-3.159232],[-4.444498,8.624422,-5.681765,8.352547,1.563487,0.165755,-2.043556,-5.858338,3.964972,-1.292672,9.507204,-0.380326,1.644587,9.794177,-3.571614],[5.479855,-4.195636,-2.601039,4.983612,6.504344,9.026005,1.011702,-4.574883,-0.133475,-1.663024,-2.436971,8.242850,-0.064085,8.050703,-5.190777],[3.109990,-4.593324,3.808568,-6.499411,4.281222,4.047647,2.350385,-1.174179,-7.755718,0.616390,2.954017,-6.950911,2.623457,-8.335675,-6.778377],[-9.527359,-4.264355,1.017588,3.980553,7.148223,2.725617,7.830027,6.768673,7.479804,-3.871325,3.394909,-1.905651,7.862168,-2.516717,-0.900853],[-1.782935,4.532711,-2.949176,-3.506375,-9.970342,-0.103396,-3.379006,-1.771958,1.394932,1.976326,3.451857,-2.683645,1.423082,1.268375,6.278744],[-4.056007,-5.320325,-9.171322,7.926447,-0.539808,-1.515717,7.246874,-0.487656,5.565642,7.992359,8.632549,7.116453,-6.635542,5.887336,-3.595287],[-0.162462,7.612166,-4.680634,3.449991,-3.594975,-3.085564,-9.817710,-5.088908,6.000367,-5.505099,-5.653628,3.128226,0.491786,-7.337940,4.138078],[1.496465,-6.833161,-0.388795,5.828127,-3.311753,5.631004,-5.441425,-6.487996,-2.434682,-2.230703,2.729816,3.896993,-4.907055,-3.798650,7.405442],[-4.021686,0.591051,9.497428,6.312010,4.341483,4.423824,-6.383120,-4.560347,-5.029306,6.032118,-2.584591,3.766190,6.127162,-5.952197,6.400889],[-8.936093,5.670807,-0.870680,1.241385,4.582058,6.050946,-0.991360,-9.718272,-4.766768,7.921641,-8.891123,-1.944943,0.854416,7.420733,-2.976072],[-9.170947,-5.331259,-6.766241,-9.412124,1.716447,1.030183,-7.910631,-5.579085,-1.998793,5.344311,-9.597373,7.883483,-0.459250,5.544755,0.167939],[-3.618839,-0.442737,7.463413,2.260801,-7.356061,7.754468,1.282938,7.988075,-0.363089,-1.449965,-0.677835,6.762295,-5.287306,-7.235707,-5.975268],[3.428905,9.106949,5.845736,-8.312927,4.181791,4.162198,-5.328787,-8.465760,9.151177,-8.337931,1.310485,-4.219794,-0.076472,-2.025583,-0.216050],[-9.107292,0.230009,4.537947,-5.113374,1.710530,7.365327,-2.471951,8.702928,7.227023,-2.172904,-7.585702,-3.251700,-6.168335,6.826719,-4.532464],[8.379408,1.403581,-8.950731,-1.603476,8.507196,-2.913194,-3.725480,-6.660169,-6.254302,-2.752173,3.217878,-0.347281,-9.836369,5.774993,8.357293],[1.710399,-2.246498,3.757756,-0.315991,-5.765813,2.553098,-5.310818,9.029439,7.221429,-2.085288,-6.133044,-1.461699,-5.441090,-3.220862,5.096337]], dtype='float32')
module1.set_input('var_1748', input_1748)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1748, )
res3 = intrp3.evaluate()(input_1748, )
res4 = intrp4.evaluate()(input_1748, )
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
module5.set_input('var_1748', input_1748)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1748, )
res7 = intrp7.evaluate()(input_1748, )
res8 = intrp8.evaluate()(input_1748, )
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
module9.set_input('var_1748', input_1748)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1748, )
res11 = intrp11.evaluate()(input_1748, )
res12 = intrp12.evaluate()(input_1748, )
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
module13.set_input('var_1748', input_1748)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1748, )
res15 = intrp15.evaluate()(input_1748, )
res16 = intrp16.evaluate()(input_1748, )
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
module17.set_input('var_1748', input_1748)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1748, )
res19 = intrp19.evaluate()(input_1748, )
res20 = intrp20.evaluate()(input_1748, )
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
module21.set_input('var_1748', input_1748)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1748, )
res23 = intrp23.evaluate()(input_1748, )
res24 = intrp24.evaluate()(input_1748, )
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

'''21: TVMFuncCall
20: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
19: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
18: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
17: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
16: tvm::transform::Pass::operator()(tvm::IRModule) const
15: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
14: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
13: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
12: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
11: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
10: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
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