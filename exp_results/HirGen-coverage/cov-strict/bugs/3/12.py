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
var_83 = relay.var("var_83", dtype = "float32", shape = (9, 3, 14))#candidate|83|(9, 3, 14)|var|float32
uop_84 = relay.cosh(var_83.astype('float32')) # shape=(9, 3, 14)
var_87 = relay.var("var_87", dtype = "float32", shape = (9, 3, 14))#candidate|87|(9, 3, 14)|var|float32
bop_88 = relay.logical_or(uop_84.astype('bool'), relay.reshape(var_87.astype('bool'), relay.shape_of(uop_84))) # shape=(9, 3, 14)
bop_96 = relay.bitwise_and(bop_88.astype('uint32'), relay.reshape(var_83.astype('uint32'), relay.shape_of(bop_88))) # shape=(9, 3, 14)
uop_121 = relay.asinh(uop_84.astype('float32')) # shape=(9, 3, 14)
bop_123 = relay.floor_divide(uop_121.astype('float64'), relay.reshape(uop_84.astype('float64'), relay.shape_of(uop_121))) # shape=(9, 3, 14)
output = relay.Tuple([bop_96,bop_123,])
output2 = relay.Tuple([bop_96,bop_123,])
func_127 = relay.Function([var_83,var_87,], output)
mod['func_127'] = func_127
mod = relay.transform.InferType()(mod)
mutated_mod['func_127'] = func_127
mutated_mod = relay.transform.InferType()(mutated_mod)
func_127_call = mutated_mod.get_global_var('func_127')
var_129 = relay.var("var_129", dtype = "float32", shape = (9, 3, 14))#candidate|129|(9, 3, 14)|var|float32
var_130 = relay.var("var_130", dtype = "float32", shape = (9, 3, 14))#candidate|130|(9, 3, 14)|var|float32
call_128 = func_127_call(var_129,var_130,)
output = call_128
func_131 = relay.Function([var_129,var_130,], output)
mutated_mod['func_131'] = func_131
mutated_mod = relay.transform.InferType()(mutated_mod)
const_187 = relay.const([[[-3,7,4,9,4,-10,10,1,-3],[-4,2,-5,4,-1,-4,-2,-9,2],[-9,-7,-2,3,8,-8,7,-3,-5],[-3,-2,1,9,6,-4,-9,9,6],[-4,5,-3,10,8,2,-2,4,9],[4,9,-4,-8,4,1,-1,2,-5],[-1,-6,10,2,-3,8,2,-8,9]],[[1,-6,-5,8,-4,-3,5,5,3],[-9,7,-1,-4,4,3,-5,-3,1],[10,5,10,-2,-2,-8,2,1,-1],[-2,-7,-6,-2,-5,2,4,-7,-5],[-6,4,3,2,1,-3,-6,-7,3],[-3,6,1,-4,-1,5,8,4,1],[5,-3,7,-9,2,7,-3,2,6]],[[9,-10,3,-6,-9,2,-8,-3,3],[6,4,10,-3,2,2,9,-10,-2],[4,-3,-8,3,1,-4,6,8,10],[-3,-3,-10,-3,-8,8,5,8,5],[4,-2,3,-1,8,1,9,2,-9],[7,-9,-4,-5,-1,10,6,5,10],[6,1,1,1,-5,2,-3,-5,-5]],[[1,1,-2,-9,6,-8,4,-6,-5],[6,-8,1,7,2,10,5,7,5],[-9,-8,8,9,-8,-6,5,1,3],[5,7,3,5,9,4,-9,-8,-5],[-3,-8,-1,-5,6,-10,-7,-2,4],[-2,-3,-1,6,-9,-7,1,-3,10],[-5,10,1,2,-9,-6,10,-10,-10]],[[-2,-10,-1,7,8,2,4,5,1],[-6,-7,9,-7,-1,4,7,8,-6],[6,-2,3,-4,-5,3,-6,1,5],[7,9,-9,1,-8,5,9,-7,-8],[9,9,3,-10,-4,-4,-10,-3,1],[3,-9,-3,5,6,7,4,5,9],[-9,7,-9,4,-4,-10,8,8,-6]]], dtype = "int32")#candidate|187|(5, 7, 9)|const|int32
var_188 = relay.var("var_188", dtype = "int32", shape = (5, 7, 9))#candidate|188|(5, 7, 9)|var|int32
bop_189 = relay.bitwise_xor(const_187.astype('int32'), relay.reshape(var_188.astype('int32'), relay.shape_of(const_187))) # shape=(5, 7, 9)
bop_195 = relay.multiply(var_188.astype('float32'), relay.reshape(const_187.astype('float32'), relay.shape_of(var_188))) # shape=(5, 7, 9)
var_198 = relay.var("var_198", dtype = "int32", shape = (5, 7, 9))#candidate|198|(5, 7, 9)|var|int32
bop_199 = relay.floor_mod(var_188.astype('float64'), relay.reshape(var_198.astype('float64'), relay.shape_of(var_188))) # shape=(5, 7, 9)
const_204 = relay.const([[[1,9,-10,1,8,-10,-3,1,-2],[6,-5,-7,3,10,-10,-1,-6,4],[-9,-2,9,9,4,9,3,8,-2],[-6,2,5,4,2,8,-6,3,5],[3,-3,-4,-3,-10,8,-5,-10,-2],[-8,3,-6,7,8,-1,1,-7,-1],[1,-1,-6,-6,3,5,-3,2,4]],[[-6,2,-1,9,9,10,-8,-5,-3],[-5,-9,5,6,8,3,-7,-2,-5],[4,-9,-1,-4,-2,-7,9,4,-10],[5,-8,-3,-4,1,9,-3,3,7],[-10,2,-9,7,-1,-7,3,-9,4],[3,4,8,-4,-8,10,3,-6,2],[-4,-5,-1,9,6,-6,-2,2,4]],[[1,10,7,7,-2,4,10,4,-6],[10,10,5,-8,-6,5,2,-10,-7],[2,6,9,-7,-10,6,7,-1,-4],[2,3,4,-10,6,-9,-8,9,5],[5,-6,-5,1,10,9,-6,8,1],[-7,-10,-2,6,-10,-3,10,8,-3],[8,-5,5,2,8,-3,4,1,-3]],[[-5,-10,-3,-1,-1,10,-4,-7,9],[-4,3,3,-3,4,-4,-9,-7,5],[4,-6,7,-8,-6,-9,-4,10,-8],[-3,-3,-7,5,6,-6,3,-7,4],[7,-4,-8,-10,9,-10,-4,-8,-2],[-3,-2,1,10,-10,5,10,10,4],[8,-10,5,-3,-5,-5,-9,-5,7]],[[-1,-2,-3,-8,8,6,4,8,-2],[4,-2,-8,5,7,-10,-6,1,-5],[4,7,9,-3,-1,7,-6,1,-9],[6,8,-3,-10,-1,-6,7,-8,-1],[-10,-10,-7,-4,-7,2,-4,7,4],[10,9,-4,8,3,6,2,10,6],[-8,-2,-3,-10,-3,-7,-9,10,8]]], dtype = "int32")#candidate|204|(5, 7, 9)|const|int32
bop_205 = relay.logical_or(var_188.astype('bool'), relay.reshape(const_204.astype('bool'), relay.shape_of(var_188))) # shape=(5, 7, 9)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
const_212 = relay.const([5.199193,6.282644,-5.924538,0.597784,-3.965488,8.765956,2.003190,-5.805000,8.726145,1.502139,-2.986530,-8.716155,-9.587124,2.897338,8.054948,-2.798597,0.083756,4.231403,-1.399756,4.978449,-6.891142,8.030109,1.131306,-9.330831,-0.132285,6.480974,-2.107280,-9.832416,-9.305818,-6.239132,-8.840102,-7.121455,-5.785313,8.547114,6.616901,-8.864627,6.113418,1.891470,5.127536,-0.323417,2.455896,-0.894333,8.657091,7.908426,-6.229988,1.897097,-6.242328,-8.285424,3.265301,2.350209,4.162622,-3.961355,-7.281197,1.963182,7.038647,5.771027,8.522107,7.442198,6.622791,-6.284341,-5.641564,-4.555879,-7.007713,-6.032905,7.421469,-1.504082,3.597319,8.377183,-1.452783,1.973593,-2.487060,2.507713,-6.296964,-2.635795,6.047103,-2.992292,4.358399,-8.736225,9.513858,6.563689,7.046764,9.176325,2.598670,-3.594666,-3.143152,0.367721,-3.347186,-4.717155,0.398884,-1.352351,-5.486951,5.227307,-2.852205,2.616891,-1.286343,-3.265338,-9.951358,-3.627389,-2.953625,1.868790,-1.735030,-7.163008,3.318636,9.695528,0.284470,1.352009,-1.647924,6.243688,0.410903,0.320354,-7.637529,2.205059,5.643554,-8.082615,-5.093707,-8.716092,-4.388541,9.965349,-7.965179,8.186471,0.365130,3.154726,6.540572,-8.357724,0.610474,-6.538184,7.348686,-5.871303,7.694594,-6.346719,-2.299023,-7.000298,4.344264,5.112014,-9.005128,-2.039990,2.313273,-7.962574,-6.320983,7.022861,-6.555375,6.120249,6.204260,-6.192611,-8.079866,-2.856371,-2.462131,-4.442362,1.431195,2.597171,-5.360967,5.983785,4.134523,-1.038417,-4.707056,9.469823,3.524190,-5.085982,-9.766240,3.695347,1.150621,2.063968,-8.786509,-0.022656,4.392426,5.574766,5.224311,-5.850405,-5.264544,-2.182868,-0.724521,5.214814,5.693491,7.518509,7.872052,-4.907081,-1.571119,-0.785649,6.336373,-8.651063,-5.422201,-6.860252,-4.848364,-6.032258,2.902959,6.169166,-7.411970,4.806473,-3.360359,4.649163,0.762077,0.329328,8.336529,-6.915007,-3.722053,-4.502028,3.494863,-3.971642,3.173018,-8.897500,8.299352,-3.181932,2.881218,6.274699,-1.733943,4.580418,9.066539,-6.347313,-4.405063,1.626767,7.981382,9.152550,3.166381,5.467529,-3.893414,0.743408,7.222105,5.293269,-6.320095,1.037579,9.823258,-8.598467,2.803738,1.463416,-9.805451,-7.258326,-9.531540,-9.411911,7.613789,-5.909043,9.254902,7.041716,-6.924078,2.171708,-8.970779,5.196389,4.121057,-1.258957,-1.260299,-6.107943,-6.478002,8.776643,-3.097767,-4.719641,0.542101,4.179329,5.441049,7.201374,4.977879,-9.479341,0.968226,3.925519,3.673194,3.450188,-5.552389,-1.513525,-6.690477,0.546349,7.653021,-8.840631,6.082733,-4.716717,7.640016,-8.915596,-9.472020,-8.429802,-1.548925,4.744952,0.535658,-5.647002,-6.281515,-3.812247,-1.739076,-1.722157,-1.106962,-3.918634,-8.142223,-7.894007,-3.220781,9.227457,9.524859,8.033647,3.270637,0.930018,2.051220,-5.088811,3.521938,5.430410,4.883730,5.095568,-2.061059,9.582229,-2.522135,-4.583705,-0.551383,-4.274779,4.032811,6.455066,-0.982822,4.176840,-3.749616,3.276306,0.924655,8.629580,-2.310799,-2.350780,-5.984337,-9.545562,5.136625,-4.009141,-0.256193,3.542556,-9.687233,-3.598091,6.114152,-6.080422,-5.876355,7.793888,-2.435393,-9.506046,-6.029632,-0.685447,-7.540913,5.336655,0.294117,3.199919,-9.008496,-7.127910,7.411325,8.176074,-2.184629,6.136336,-8.247223,-5.533290,9.631048,6.850973,9.378397,6.849203,-1.659582,2.586881,-3.406171,3.250653,8.357250,9.957163,2.470868,-1.162992,-0.360417,-2.934161,-1.192017,-2.720313,5.265858,1.888521,2.277296,-4.177950,-1.060816,0.576464,-1.694780,2.670500,-2.740891,-7.539290,-0.969873,-4.148564,-3.344087,2.488106,8.304688,6.185403,4.196023,-4.659233,9.584163,6.514897,-3.265284,-1.127352,5.332732,6.498581,-6.330144,3.570488,-0.349272,0.497501], dtype = "float32")#candidate|212|(378,)|const|float32
call_211 = relay.TupleGetItem(func_127_call(relay.reshape(const_212.astype('float32'), [9, 3, 14]), relay.reshape(const_212.astype('float32'), [9, 3, 14]), ), 1)
call_213 = relay.TupleGetItem(func_131_call(relay.reshape(const_212.astype('float32'), [9, 3, 14]), relay.reshape(const_212.astype('float32'), [9, 3, 14]), ), 1)
bop_218 = relay.less(bop_199.astype('bool'), relay.reshape(const_204.astype('bool'), relay.shape_of(bop_199))) # shape=(5, 7, 9)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
call_232 = relay.TupleGetItem(func_127_call(relay.reshape(const_212.astype('float32'), [9, 3, 14]), relay.reshape(call_211.astype('float32'), [9, 3, 14]), ), 0)
call_233 = relay.TupleGetItem(func_131_call(relay.reshape(const_212.astype('float32'), [9, 3, 14]), relay.reshape(call_211.astype('float32'), [9, 3, 14]), ), 0)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
call_236 = relay.TupleGetItem(func_127_call(relay.reshape(call_211.astype('float32'), [9, 3, 14]), relay.reshape(const_212.astype('float32'), [9, 3, 14]), ), 1)
call_237 = relay.TupleGetItem(func_131_call(relay.reshape(call_211.astype('float32'), [9, 3, 14]), relay.reshape(const_212.astype('float32'), [9, 3, 14]), ), 1)
output = relay.Tuple([bop_189,bop_195,bop_205,call_211,const_212,bop_218,call_232,call_236,])
output2 = relay.Tuple([bop_189,bop_195,bop_205,call_213,const_212,bop_218,call_233,call_237,])
func_241 = relay.Function([var_188,var_198,], output)
mod['func_241'] = func_241
mod = relay.transform.InferType()(mod)
var_242 = relay.var("var_242", dtype = "int32", shape = (5, 7, 9))#candidate|242|(5, 7, 9)|var|int32
var_243 = relay.var("var_243", dtype = "int32", shape = (5, 7, 9))#candidate|243|(5, 7, 9)|var|int32
output = func_241(var_242,var_243,)
func_244 = relay.Function([var_242,var_243,], output)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
var_249 = relay.var("var_249", dtype = "float64", shape = (8, 3, 15))#candidate|249|(8, 3, 15)|var|float64
uop_250 = relay.sin(var_249.astype('float64')) # shape=(8, 3, 15)
var_252 = relay.var("var_252", dtype = "float64", shape = (8, 3, 15))#candidate|252|(8, 3, 15)|var|float64
bop_253 = relay.greater(uop_250.astype('bool'), relay.reshape(var_252.astype('bool'), relay.shape_of(uop_250))) # shape=(8, 3, 15)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
var_258 = relay.var("var_258", dtype = "float32", shape = (378,))#candidate|258|(378,)|var|float32
call_257 = relay.TupleGetItem(func_127_call(relay.reshape(var_258.astype('float32'), [9, 3, 14]), relay.reshape(var_258.astype('float32'), [9, 3, 14]), ), 1)
call_259 = relay.TupleGetItem(func_131_call(relay.reshape(var_258.astype('float32'), [9, 3, 14]), relay.reshape(var_258.astype('float32'), [9, 3, 14]), ), 1)
uop_260 = relay.log(bop_253.astype('float64')) # shape=(8, 3, 15)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
call_265 = relay.TupleGetItem(func_127_call(relay.reshape(call_257.astype('float32'), [9, 3, 14]), relay.reshape(var_258.astype('float32'), [9, 3, 14]), ), 1)
call_266 = relay.TupleGetItem(func_131_call(relay.reshape(call_257.astype('float32'), [9, 3, 14]), relay.reshape(var_258.astype('float32'), [9, 3, 14]), ), 1)
uop_268 = relay.tan(uop_260.astype('float64')) # shape=(8, 3, 15)
output = relay.Tuple([call_257,var_258,call_265,uop_268,])
output2 = relay.Tuple([call_259,var_258,call_266,uop_268,])
func_274 = relay.Function([var_249,var_252,var_258,], output)
mod['func_274'] = func_274
mod = relay.transform.InferType()(mod)
mutated_mod['func_274'] = func_274
mutated_mod = relay.transform.InferType()(mutated_mod)
func_274_call = mutated_mod.get_global_var('func_274')
var_276 = relay.var("var_276", dtype = "float64", shape = (8, 3, 15))#candidate|276|(8, 3, 15)|var|float64
var_277 = relay.var("var_277", dtype = "float64", shape = (8, 3, 15))#candidate|277|(8, 3, 15)|var|float64
var_278 = relay.var("var_278", dtype = "float32", shape = (378,))#candidate|278|(378,)|var|float32
call_275 = func_274_call(var_276,var_277,var_278,)
output = call_275
func_279 = relay.Function([var_276,var_277,var_278,], output)
mutated_mod['func_279'] = func_279
mutated_mod = relay.transform.InferType()(mutated_mod)
var_438 = relay.var("var_438", dtype = "int16", shape = ())#candidate|438|()|var|int16
const_439 = relay.const([[[4,-10,-4,-4,-9,-9,10,3,6,-2,2,-7,-6,-9,-4,10],[9,-4,2,-3,3,-9,-3,2,2,8,-6,2,-9,4,5,3],[6,8,7,1,7,-7,-10,10,3,-5,-2,4,7,-7,1,-8],[-9,-7,10,-8,-7,7,-4,-1,-2,-9,-3,-8,-10,-3,-6,10],[-7,-4,6,-6,-1,-2,8,3,8,-6,6,1,10,-6,3,4],[4,9,-1,3,-7,-6,-8,7,-4,9,1,-3,4,-7,-5,6],[-10,7,9,-5,7,-3,4,-5,3,6,5,-4,9,5,6,-1],[-8,10,-10,-8,-8,-6,-9,9,-6,-7,-4,-10,-6,-7,8,-9],[-6,-8,3,5,-4,-2,-1,-8,-5,8,8,4,-4,-6,-7,-3],[-1,-1,-6,4,7,-5,-8,-5,-10,4,-2,-1,9,1,10,1],[6,8,8,3,9,-4,-3,-9,4,8,-2,-4,-8,-7,10,2],[1,7,-1,5,-8,-6,-1,-1,-8,3,-2,1,4,8,7,2],[-9,7,1,1,-4,10,2,2,-4,2,-4,6,2,-3,-4,-6],[8,-3,9,8,-9,-9,-4,2,6,4,6,7,3,-8,-9,-8],[-10,-3,2,-2,-2,5,4,3,-3,1,8,9,9,8,10,2]],[[6,-10,-9,-1,-5,-8,-4,-10,-2,6,6,1,1,-3,5,4],[3,10,-6,4,5,-10,-6,6,9,6,-7,3,6,10,3,1],[-6,7,-1,-3,6,-7,-2,-8,5,10,-3,-10,1,10,-6,10],[-1,-3,-10,1,10,8,-1,6,-2,-7,-9,4,1,-4,7,8],[-2,10,-9,-10,9,2,-4,-5,3,1,-3,4,1,10,3,-3],[10,9,-6,-6,-4,-4,-5,3,3,1,-1,9,-2,-7,8,2],[-5,5,7,-6,-2,-9,1,6,-8,-9,2,10,-10,1,-6,6],[-3,5,-4,2,3,-2,7,-8,7,-5,-9,4,-9,7,-9,9],[2,5,6,2,3,2,2,-7,9,-9,-7,8,-10,-3,6,5],[7,5,-9,1,-1,-2,-4,3,10,-4,6,3,4,7,2,-1],[-1,-2,-3,10,7,10,-10,-7,-6,-7,2,8,-8,10,-8,8],[-5,-10,3,-6,-4,2,-1,4,1,-10,1,-2,5,-8,-1,2],[7,6,-2,10,9,9,2,-7,-3,9,9,3,-1,8,-9,8],[2,6,8,-1,-8,7,4,-7,5,10,-7,-7,-6,-10,5,1],[-4,3,-10,9,5,-10,-3,5,-2,1,10,6,3,8,-6,5]],[[-5,1,5,5,-9,-8,-3,10,9,-3,-1,2,-8,2,-3,-3],[-9,6,-6,3,-6,-10,7,-4,-3,-5,9,8,2,-1,-9,3],[-9,-2,-8,4,8,-4,-6,-1,-9,-3,6,6,-3,10,1,2],[2,9,-8,5,7,-3,2,-1,-2,9,-10,-3,2,-7,-9,-3],[2,1,4,-8,-9,-3,-6,-2,-1,-4,4,3,5,8,-6,6],[-4,-10,5,9,-10,5,-7,2,-7,-8,-8,-6,-9,-8,7,7],[-1,10,-4,9,1,7,2,5,-10,10,-4,-2,1,1,1,9],[-2,7,-10,-7,-2,-1,6,4,9,-3,-8,-7,4,-9,4,7],[1,-2,-4,-3,3,-1,-10,-6,-8,-3,5,-7,7,-4,-6,-7],[-6,-7,7,-7,4,1,4,4,-8,10,8,-10,5,-6,-6,-7],[1,5,6,10,1,-3,-1,-8,-3,-2,-6,-5,-8,-2,-5,7],[7,-4,5,10,-5,4,9,5,-10,4,-9,7,-2,-10,-9,3],[-2,-6,2,4,10,-3,7,10,5,3,5,-1,-10,3,9,-5],[4,-7,-9,6,-8,8,-9,5,4,-9,-8,7,-1,-7,-8,-3],[-6,-1,-3,4,-5,9,3,-9,-1,-8,-7,9,7,2,10,8]],[[-8,-1,-5,4,-9,4,-4,3,-9,6,-7,-8,3,-10,7,-7],[6,-8,6,10,3,-1,5,4,-4,4,5,6,1,-1,3,-3],[-7,-1,-1,10,3,2,-3,5,-7,1,-10,2,-4,-10,-10,-7],[2,-1,8,-8,3,2,6,-5,6,-7,-4,10,1,-4,-2,1],[10,-8,3,1,6,4,-1,-5,7,-7,-2,-9,-6,-5,6,4],[-7,-6,8,8,1,6,-1,10,4,8,-4,-8,-10,1,-8,-3],[7,3,9,9,-3,3,-7,-1,2,-8,4,9,10,1,-4,1],[8,-5,7,-6,-1,4,1,3,-3,-10,-9,-3,8,7,2,1],[3,-6,4,-1,-10,4,-10,-5,-2,-4,-5,-9,2,-4,5,-6],[-3,-9,4,-1,4,-6,-6,9,-7,8,9,2,5,-10,1,-9],[8,-1,-7,-7,-10,7,-5,4,9,-3,-9,-2,2,5,7,10],[-1,8,-5,-10,9,3,-4,-7,6,3,-6,8,-8,3,-6,-1],[-6,-6,5,8,-5,-9,5,-4,-7,-6,4,2,8,7,10,2],[8,-7,-8,-6,-5,2,2,-2,-4,3,4,-10,-1,-8,-9,-7],[10,10,10,-1,-5,-4,8,8,-6,-10,6,-3,-10,-10,-1,6]],[[10,1,1,4,4,3,-5,6,3,2,-7,-8,3,-6,4,6],[-7,5,-9,-4,-8,9,-1,7,-2,-5,3,7,-1,10,6,-3],[-5,1,-7,10,-5,1,5,-9,-6,-9,-8,-4,-1,6,2,-10],[-10,-2,-7,4,-3,-7,-2,9,-1,-6,-5,9,5,-7,7,-4],[10,3,2,9,-2,-6,-5,4,-3,6,-9,6,-10,-8,10,-3],[-10,8,-4,9,-5,2,-10,-7,-5,1,-8,7,-1,5,9,-4],[4,2,10,6,6,7,-4,-9,7,-4,-10,-3,8,-4,5,9],[-1,7,-4,-7,3,-8,-6,8,8,8,-5,-1,-8,1,3,5],[2,-5,7,9,-4,6,6,5,4,-4,-6,-1,9,10,-3,-6],[-6,-5,1,-10,-1,-8,8,2,7,-2,-8,9,-8,-9,-1,10],[-8,-8,-9,5,-5,-8,6,-9,-6,8,-8,3,-4,-9,-9,-2],[-1,7,1,1,-7,-2,-9,-8,4,10,4,7,6,-5,-4,4],[-9,5,-5,7,7,4,10,10,-9,5,-4,1,8,-6,3,8],[-3,2,-1,-9,-6,-4,-7,10,5,-9,-10,8,6,-2,4,-1],[-5,6,-1,6,-9,6,7,6,-3,-2,5,3,-3,-2,-5,5]],[[-5,-7,7,-6,6,-4,9,-6,5,7,7,5,5,9,10,1],[5,4,3,-8,-1,3,-4,-4,6,-3,1,3,3,9,5,1],[-10,-2,4,-6,-6,-6,-10,1,2,-9,2,-3,3,2,-2,8],[7,-6,1,7,1,-10,-4,6,-9,-10,-8,-7,5,-9,-2,8],[8,-2,4,-8,5,-5,-1,6,8,10,-9,5,8,-10,2,2],[9,-8,-5,9,6,-10,5,-8,2,-8,-7,-4,8,7,-1,7],[5,5,6,4,10,-1,-10,9,-5,6,2,6,3,10,-4,-5],[4,1,-6,2,-8,2,-9,3,-9,-7,6,-9,-9,-6,5,2],[-1,-1,-7,-6,5,-5,10,10,5,-8,-1,4,7,-4,9,-4],[-10,-1,8,-4,-5,9,6,-3,-7,4,10,-9,-5,-9,-1,-9],[8,10,9,-8,4,10,-7,1,-5,-3,10,-10,-4,-1,2,8],[2,-5,-3,-5,-4,-6,-7,-4,-1,-5,-8,-8,-6,1,-1,-9],[8,-3,-10,-7,3,-3,10,1,6,-2,5,-9,9,10,-9,-4],[-4,8,-7,-6,8,-3,1,-2,1,1,4,-6,10,-4,4,5],[-6,2,6,6,-9,6,4,-7,2,1,-8,10,3,-6,-3,-5]],[[-2,6,4,5,2,3,7,-9,8,6,8,-5,-3,10,-1,-7],[-1,-4,7,-10,9,3,3,-9,-3,-8,-3,4,3,-2,-8,-1],[3,5,7,9,1,3,5,7,-7,8,8,-7,-5,-9,-9,-3],[-8,-4,-4,-10,-4,-8,-7,4,-10,5,6,10,1,6,2,4],[-6,4,1,6,8,-2,-9,-8,-3,-8,5,2,-7,-5,8,8],[8,-5,-5,4,-6,-6,-5,5,-9,6,-1,9,-7,-9,-4,8],[-9,2,-5,-7,3,6,-10,-5,-4,-1,9,6,-4,10,2,-3],[2,-4,4,-8,5,-6,-4,3,1,-6,-6,8,-7,-10,-8,-9],[10,9,-5,2,10,-4,-10,6,-3,-2,5,8,8,-9,5,-9],[1,1,-5,-5,-5,-3,10,-6,2,-1,-8,8,-9,6,-7,10],[-3,4,-6,-4,2,5,7,-2,-9,-5,-1,5,-2,-5,6,-6],[4,6,1,-10,7,4,7,2,-8,-1,4,1,6,4,9,10],[2,-5,8,-7,-3,1,-6,5,2,6,-3,3,-5,3,8,1],[-9,-4,-4,-2,-1,-8,-5,-2,-1,4,-7,5,-3,8,-2,-1],[5,-8,4,7,-4,-2,-10,-10,7,4,3,6,8,7,-2,10]],[[4,6,9,-5,2,4,6,-2,-9,6,8,8,-5,3,-7,-5],[-3,1,6,-5,-7,-10,-6,-7,1,-7,-5,-2,-3,-10,-9,1],[7,4,-10,-2,1,-2,-4,-1,2,1,-4,10,9,-6,-6,-10],[9,-4,4,-9,-8,1,-2,7,8,-3,-1,10,4,10,-2,-1],[-8,-1,-3,-7,-1,-2,-9,-7,7,1,-3,-9,6,-5,2,9],[-2,-2,9,7,8,-3,10,-7,-9,6,1,7,7,-4,-7,-8],[-3,8,-2,2,-6,-8,-4,-5,5,-3,7,4,1,2,2,10],[4,5,-3,-5,-3,-5,-2,-10,-2,-3,-2,-2,4,-4,5,10],[-10,-5,-6,-1,-7,-4,9,-4,-6,1,-4,4,8,-4,1,-4],[3,-5,-2,-5,-5,6,4,4,-10,9,-9,-2,10,9,4,-2],[3,-5,4,-3,-1,-2,2,-5,2,-6,7,-5,6,8,-9,6],[-8,8,2,-2,4,-8,-1,-9,-8,-3,5,-4,3,-2,4,6],[-10,5,-8,5,-10,-8,-7,-3,4,-2,3,-1,-1,1,-2,7],[-5,-3,9,7,-10,-8,4,4,7,-8,-3,9,3,-9,-10,6],[-9,10,-7,-7,2,6,9,5,-3,8,-1,1,-10,-6,7,8]],[[2,-5,-5,-6,10,-2,-9,6,-9,1,3,3,-2,-6,2,4],[1,1,-2,10,10,-8,10,-2,-8,10,-4,-7,-1,-5,6,-7],[9,-1,-6,10,5,-6,3,6,10,5,9,2,2,3,7,4],[-7,-6,6,-10,5,7,-3,-9,1,8,-3,-9,8,-4,-1,10],[1,-4,1,9,-9,1,-1,6,-4,1,-9,-4,-4,3,-7,7],[8,-7,5,-4,-4,10,-7,9,-3,9,-3,-8,-3,4,5,-8],[-3,3,-2,-6,-3,-4,-3,2,-3,7,-5,-5,-7,8,2,-4],[-10,-7,-3,1,-9,6,7,-7,10,8,-2,-6,4,10,10,5],[10,-4,-4,9,-9,-7,-3,-9,-2,-10,10,3,10,1,9,9],[9,2,-10,9,4,-8,-2,-1,3,-9,9,-8,5,-6,7,-4],[1,-10,-5,-10,6,-7,4,5,1,-10,-5,-7,8,-5,-1,1],[-4,-2,-1,9,10,-8,-6,-4,4,-7,-2,1,10,-7,6,-2],[9,1,-6,-9,-8,3,-9,-1,10,-4,-2,4,-1,-2,9,9],[-10,9,-1,-4,-7,-3,5,3,-1,8,2,-7,3,1,-5,-9],[9,9,-10,8,-2,-6,-4,10,4,1,2,-10,-7,-8,-3,7]],[[6,-3,1,-4,10,-7,8,10,-9,-9,-8,7,-4,-10,-3,9],[3,8,1,4,-1,-8,10,1,10,2,3,-5,-4,1,10,-9],[-2,-6,8,-8,10,-9,3,-10,2,-3,1,1,3,-5,4,5],[-2,2,-10,8,-4,2,7,-2,-6,-5,1,4,-7,-6,-6,6],[-3,-7,-9,-4,-2,-2,-8,9,9,3,-7,5,-10,5,-3,2],[-1,8,-4,5,-9,-8,-1,-3,10,-2,2,4,-8,8,2,-7],[10,5,8,-1,-6,3,7,-10,1,7,7,5,-8,7,4,-5],[9,-8,1,-5,-10,4,6,6,6,-1,3,-10,-1,-3,3,9],[1,1,1,-8,-4,-10,-6,8,-10,-7,-8,-5,6,-6,-9,10],[-4,-1,7,-8,2,3,-6,2,-1,10,7,-6,7,-1,9,-6],[10,-10,5,-10,-9,6,4,-5,8,2,-3,10,2,-6,-1,-6],[10,-10,5,3,-2,1,-7,-1,-2,-2,-1,4,-2,-6,3,-5],[-7,7,6,-8,-6,-6,-7,-10,10,-5,4,6,-2,-10,4,7],[4,-5,-3,-1,-4,-4,6,10,-7,6,-6,9,7,-2,-9,-2],[2,-5,2,4,5,-7,-5,6,3,3,-6,-7,6,-4,8,-7]],[[-5,7,-10,-3,10,-7,-8,7,5,-1,4,6,7,4,7,-8],[-6,2,-6,3,7,-3,3,4,-9,2,3,5,7,-7,-3,-7],[-10,6,-5,3,3,5,-5,-5,-6,-10,-5,10,6,-6,-4,9],[1,-1,-8,-8,-10,6,-6,4,10,-2,10,-8,-1,5,-7,-9],[-2,7,-5,-9,-8,4,10,5,-1,6,2,8,-6,9,7,-2],[-10,-9,3,-4,4,6,-4,-3,-9,6,1,-5,9,10,5,5],[2,2,3,-8,-4,1,5,-4,-1,6,-6,-10,-6,-1,6,-8],[-1,4,6,10,-6,9,5,6,6,1,-2,7,3,8,-5,6],[7,-1,-3,7,2,-1,-9,-4,-6,7,5,-7,-1,6,3,1],[3,5,2,7,7,-2,2,3,10,7,1,4,1,7,3,4],[-9,5,-2,-7,8,-7,-7,2,8,3,-5,3,-8,-8,6,10],[-10,9,5,2,5,-6,8,10,-7,-7,-3,-8,1,-3,-7,-4],[-5,-10,-7,6,10,4,9,-5,-10,-1,-5,3,5,2,-8,1],[-9,4,1,-8,3,10,10,2,-2,-3,2,-4,2,10,-2,3],[-2,-7,-10,-4,1,-5,-1,-2,-10,9,-1,-2,1,-4,3,-6]],[[8,-7,-10,-9,7,5,1,-3,2,-6,7,7,9,10,-1,-6],[10,5,-4,-3,10,-1,5,5,-5,10,6,9,1,7,6,10],[7,-5,2,2,-6,-5,6,-8,-1,-5,6,-4,3,-10,7,4],[7,-6,7,-3,2,6,9,1,10,7,-1,-6,4,-5,6,-6],[-9,4,-10,-1,-4,-10,-8,-1,8,6,-10,-6,-8,5,10,-9],[-4,5,7,-4,-2,3,5,8,-5,5,-9,-10,-3,3,8,4],[-2,-6,6,3,4,8,2,-5,2,-4,10,-8,-3,-8,-2,5],[-3,6,-8,-7,-4,2,-1,-7,-6,2,5,10,-2,8,-1,9],[-3,3,-5,-4,5,7,4,7,-4,-5,-10,6,-1,5,9,-6],[-2,-2,1,-5,3,-10,-5,6,-7,-8,-5,-8,-1,10,10,10],[9,6,1,4,-3,-2,5,-8,8,9,10,-4,2,-1,-2,-4],[-5,10,-2,-10,1,5,-10,8,9,-3,-5,-10,-1,-9,1,5],[-4,10,1,2,9,-10,10,-9,5,8,9,9,10,-6,-9,-5],[-5,-10,-6,-8,10,6,-6,-1,-6,-3,8,6,9,-8,-4,3],[1,-6,8,2,4,9,10,4,-8,3,-8,7,3,-6,-10,6]],[[2,8,8,7,-5,-3,-6,-3,4,4,4,-9,-7,-4,4,10],[-2,8,-6,3,1,8,-4,6,8,-2,-1,1,-8,7,-3,-9],[-5,6,-2,-3,-6,4,4,10,2,-4,4,3,10,-3,-8,9],[-9,-5,5,-6,10,5,-10,8,8,4,-9,4,-8,-2,9,-10],[-10,9,-2,10,-5,-10,9,-4,9,-9,-7,-8,9,-1,-3,7],[5,-1,-1,1,-4,-8,8,-2,-3,5,-5,5,5,2,-7,3],[5,1,3,-1,-4,1,-2,-8,5,5,10,-4,5,5,3,6],[-2,-3,2,3,2,7,-9,10,-10,-4,4,5,3,3,-8,3],[7,-6,-9,-4,-4,-6,-6,7,-1,8,10,-7,5,-9,9,4],[8,-7,-8,-8,2,5,10,7,4,7,1,7,3,3,-7,-10],[-7,-2,7,-7,-3,8,4,-5,1,2,-7,9,2,4,9,4],[2,-10,5,-10,10,4,-3,10,-2,-7,-9,10,10,-1,-7,-6],[-1,10,6,4,7,-7,8,6,2,-10,-10,4,-4,9,-10,-2],[3,-6,-1,-4,10,-1,-1,-8,-8,3,9,-10,-2,6,-5,-4],[-10,7,3,-2,-3,1,7,6,-6,-5,-8,-7,3,-4,-2,7]]], dtype = "int16")#candidate|439|(13, 15, 16)|const|int16
bop_440 = relay.not_equal(var_438.astype('bool'), const_439.astype('bool')) # shape=(13, 15, 16)
output = bop_440
output2 = bop_440
func_450 = relay.Function([var_438,], output)
mod['func_450'] = func_450
mod = relay.transform.InferType()(mod)
var_451 = relay.var("var_451", dtype = "int16", shape = ())#candidate|451|()|var|int16
output = func_450(var_451)
func_452 = relay.Function([var_451], output)
mutated_mod['func_452'] = func_452
mutated_mod = relay.transform.InferType()(mutated_mod)
var_546 = relay.var("var_546", dtype = "uint32", shape = (2, 5, 11))#candidate|546|(2, 5, 11)|var|uint32
var_547 = relay.var("var_547", dtype = "uint32", shape = (2, 5, 11))#candidate|547|(2, 5, 11)|var|uint32
bop_548 = relay.greater_equal(var_546.astype('bool'), relay.reshape(var_547.astype('bool'), relay.shape_of(var_546))) # shape=(2, 5, 11)
uop_562 = relay.sin(var_546.astype('float64')) # shape=(2, 5, 11)
output = relay.Tuple([bop_548,uop_562,])
output2 = relay.Tuple([bop_548,uop_562,])
func_567 = relay.Function([var_546,var_547,], output)
mod['func_567'] = func_567
mod = relay.transform.InferType()(mod)
var_568 = relay.var("var_568", dtype = "uint32", shape = (2, 5, 11))#candidate|568|(2, 5, 11)|var|uint32
var_569 = relay.var("var_569", dtype = "uint32", shape = (2, 5, 11))#candidate|569|(2, 5, 11)|var|uint32
output = func_567(var_568,var_569,)
func_570 = relay.Function([var_568,var_569,], output)
mutated_mod['func_570'] = func_570
mutated_mod = relay.transform.InferType()(mutated_mod)
var_628 = relay.var("var_628", dtype = "float32", shape = (6, 10, 1))#candidate|628|(6, 10, 1)|var|float32
var_629 = relay.var("var_629", dtype = "float32", shape = (6, 10, 13))#candidate|629|(6, 10, 13)|var|float32
bop_630 = relay.divide(var_628.astype('float32'), var_629.astype('float32')) # shape=(6, 10, 13)
uop_641 = relay.acosh(bop_630.astype('float64')) # shape=(6, 10, 13)
var_648 = relay.var("var_648", dtype = "float64", shape = (6, 10, 13))#candidate|648|(6, 10, 13)|var|float64
bop_649 = relay.logical_xor(uop_641.astype('int8'), relay.reshape(var_648.astype('int8'), relay.shape_of(uop_641))) # shape=(6, 10, 13)
uop_654 = relay.log2(bop_649.astype('float64')) # shape=(6, 10, 13)
uop_684 = relay.acos(bop_630.astype('float32')) # shape=(6, 10, 13)
bop_690 = relay.subtract(uop_654.astype('int32'), relay.reshape(var_648.astype('int32'), relay.shape_of(uop_654))) # shape=(6, 10, 13)
bop_705 = relay.bitwise_xor(bop_690.astype('int32'), relay.reshape(uop_684.astype('int32'), relay.shape_of(bop_690))) # shape=(6, 10, 13)
output = relay.Tuple([bop_705,])
output2 = relay.Tuple([bop_705,])
func_710 = relay.Function([var_628,var_629,var_648,], output)
mod['func_710'] = func_710
mod = relay.transform.InferType()(mod)
var_711 = relay.var("var_711", dtype = "float32", shape = (6, 10, 1))#candidate|711|(6, 10, 1)|var|float32
var_712 = relay.var("var_712", dtype = "float32", shape = (6, 10, 13))#candidate|712|(6, 10, 13)|var|float32
var_713 = relay.var("var_713", dtype = "float64", shape = (6, 10, 13))#candidate|713|(6, 10, 13)|var|float64
output = func_710(var_711,var_712,var_713,)
func_714 = relay.Function([var_711,var_712,var_713,], output)
mutated_mod['func_714'] = func_714
mutated_mod = relay.transform.InferType()(mutated_mod)
var_752 = relay.var("var_752", dtype = "float64", shape = (14, 10))#candidate|752|(14, 10)|var|float64
uop_753 = relay.log(var_752.astype('float64')) # shape=(14, 10)
output = uop_753
output2 = uop_753
func_762 = relay.Function([var_752,], output)
mod['func_762'] = func_762
mod = relay.transform.InferType()(mod)
var_763 = relay.var("var_763", dtype = "float64", shape = (14, 10))#candidate|763|(14, 10)|var|float64
output = func_762(var_763)
func_764 = relay.Function([var_763], output)
mutated_mod['func_764'] = func_764
mutated_mod = relay.transform.InferType()(mutated_mod)
var_774 = relay.var("var_774", dtype = "float64", shape = (8, 11, 2))#candidate|774|(8, 11, 2)|var|float64
uop_775 = relay.erf(var_774.astype('float64')) # shape=(8, 11, 2)
uop_787 = relay.asin(var_774.astype('float64')) # shape=(8, 11, 2)
bop_795 = relay.logical_or(uop_787.astype('bool'), relay.reshape(var_774.astype('bool'), relay.shape_of(uop_787))) # shape=(8, 11, 2)
output = relay.Tuple([uop_775,bop_795,])
output2 = relay.Tuple([uop_775,bop_795,])
func_807 = relay.Function([var_774,], output)
mod['func_807'] = func_807
mod = relay.transform.InferType()(mod)
mutated_mod['func_807'] = func_807
mutated_mod = relay.transform.InferType()(mutated_mod)
var_808 = relay.var("var_808", dtype = "float64", shape = (8, 11, 2))#candidate|808|(8, 11, 2)|var|float64
func_807_call = mutated_mod.get_global_var('func_807')
call_809 = func_807_call(var_808)
output = call_809
func_810 = relay.Function([var_808], output)
mutated_mod['func_810'] = func_810
mutated_mod = relay.transform.InferType()(mutated_mod)
var_829 = relay.var("var_829", dtype = "float32", shape = (7, 11, 3))#candidate|829|(7, 11, 3)|var|float32
const_830 = relay.const([[[4.124820,-2.388434,-6.686764],[6.652220,-2.794420,-6.824312],[0.107966,-6.789346,-1.462434],[-0.030016,8.765720,-6.023881],[3.610341,-9.144903,-9.385959],[-0.250376,0.374465,2.909564],[-4.537183,4.730096,5.831426],[7.029616,9.928803,1.819050],[3.234788,-8.609255,-0.724903],[4.776427,-6.542240,1.291571],[-6.228544,-0.440537,2.645209]],[[0.966339,-6.059777,-7.510609],[-7.661764,6.187036,6.212778],[7.903058,-2.528664,2.569684],[-2.526288,-2.729115,3.904760],[7.282011,9.640515,2.386865],[-8.571882,-3.414756,-5.363938],[-8.784390,0.974572,-4.874183],[-1.163816,7.994505,8.871634],[2.123864,0.968299,-5.480075],[-6.417748,-4.918889,8.112655],[-3.709841,-5.523240,-4.163475]],[[1.281704,1.396548,-2.276490],[0.766779,9.957534,-6.173687],[-5.798971,0.676162,1.787879],[-4.217035,3.974753,-0.408254],[-3.538440,-3.168602,0.242894],[3.820086,9.297460,4.178151],[-1.661057,6.440044,1.308552],[-2.646913,8.794892,-1.783694],[3.089168,-1.620709,4.831363],[9.088094,4.321572,9.750337],[4.302090,4.745754,9.846905]],[[-1.716348,-7.650737,-4.789492],[1.369671,6.684535,-5.047829],[-2.277639,-0.066013,-7.710921],[0.875609,0.570376,-7.682781],[8.450564,-5.868044,-3.652293],[-5.315032,3.809305,-9.714893],[-9.684709,4.758923,4.369582],[0.298427,0.089526,-7.738176],[-6.985785,-6.095562,6.414753],[-5.894216,-8.984541,2.143482],[-0.927529,8.729963,-9.578272]],[[1.735428,-5.256082,-2.411206],[-3.142795,0.813482,5.709692],[-8.442186,8.598379,-2.289339],[5.325183,8.662311,3.344432],[-8.497489,-5.613581,1.338103],[-7.232741,-5.266850,8.979982],[-9.567750,-0.907428,-3.911294],[4.215441,3.630244,5.139718],[6.124857,-0.015189,-2.679142],[-9.937197,-4.679594,-2.019796],[6.908477,-6.198335,0.560625]],[[4.098429,-4.558332,6.562342],[2.703429,1.512735,0.223003],[-5.599152,8.254614,-0.049368],[-4.183947,-4.256372,5.270618],[6.697615,5.395153,-8.901429],[3.674192,-5.450955,-3.898416],[-0.767704,0.201559,-4.549063],[-7.992043,5.886428,-0.870681],[9.186637,-6.354566,3.659578],[1.882216,1.148842,-6.178765],[2.348193,7.674185,7.463297]],[[-3.741462,1.226195,8.264533],[-2.188365,2.165280,9.778801],[4.330053,-7.908521,8.903918],[-4.035285,-8.553558,5.498956],[1.983808,6.501905,4.030024],[-1.529298,-6.445050,9.424518],[-8.433661,-2.423463,9.747144],[7.801165,-1.218711,6.664318],[3.664767,-9.436904,7.367126],[-0.404657,7.846542,7.619299],[0.847128,-0.966147,6.478738]]], dtype = "float32")#candidate|830|(7, 11, 3)|const|float32
bop_831 = relay.greater(var_829.astype('bool'), relay.reshape(const_830.astype('bool'), relay.shape_of(var_829))) # shape=(7, 11, 3)
func_762_call = mod.get_global_var('func_762')
func_764_call = mutated_mod.get_global_var('func_764')
var_835 = relay.var("var_835", dtype = "float64", shape = (70, 2))#candidate|835|(70, 2)|var|float64
call_834 = func_762_call(relay.reshape(var_835.astype('float64'), [14, 10]))
call_836 = func_762_call(relay.reshape(var_835.astype('float64'), [14, 10]))
output = relay.Tuple([bop_831,call_834,var_835,])
output2 = relay.Tuple([bop_831,call_836,var_835,])
func_842 = relay.Function([var_829,var_835,], output)
mod['func_842'] = func_842
mod = relay.transform.InferType()(mod)
mutated_mod['func_842'] = func_842
mutated_mod = relay.transform.InferType()(mutated_mod)
func_842_call = mutated_mod.get_global_var('func_842')
var_844 = relay.var("var_844", dtype = "float32", shape = (7, 11, 3))#candidate|844|(7, 11, 3)|var|float32
var_845 = relay.var("var_845", dtype = "float64", shape = (70, 2))#candidate|845|(70, 2)|var|float64
call_843 = func_842_call(var_844,var_845,)
output = call_843
func_846 = relay.Function([var_844,var_845,], output)
mutated_mod['func_846'] = func_846
mutated_mod = relay.transform.InferType()(mutated_mod)
var_873 = relay.var("var_873", dtype = "float64", shape = (6, 15))#candidate|873|(6, 15)|var|float64
uop_874 = relay.tan(var_873.astype('float64')) # shape=(6, 15)
output = uop_874
output2 = uop_874
func_882 = relay.Function([var_873,], output)
mod['func_882'] = func_882
mod = relay.transform.InferType()(mod)
mutated_mod['func_882'] = func_882
mutated_mod = relay.transform.InferType()(mutated_mod)
var_883 = relay.var("var_883", dtype = "float64", shape = (6, 15))#candidate|883|(6, 15)|var|float64
func_882_call = mutated_mod.get_global_var('func_882')
call_884 = func_882_call(var_883)
output = call_884
func_885 = relay.Function([var_883], output)
mutated_mod['func_885'] = func_885
mutated_mod = relay.transform.InferType()(mutated_mod)
var_903 = relay.var("var_903", dtype = "float32", shape = (10, 11))#candidate|903|(10, 11)|var|float32
var_904 = relay.var("var_904", dtype = "float32", shape = (10, 11))#candidate|904|(10, 11)|var|float32
bop_905 = relay.less_equal(var_903.astype('bool'), relay.reshape(var_904.astype('bool'), relay.shape_of(var_903))) # shape=(10, 11)
func_127_call = mod.get_global_var('func_127')
func_131_call = mutated_mod.get_global_var('func_131')
var_920 = relay.var("var_920", dtype = "float32", shape = (378,))#candidate|920|(378,)|var|float32
call_919 = relay.TupleGetItem(func_127_call(relay.reshape(var_920.astype('float32'), [9, 3, 14]), relay.reshape(var_920.astype('float32'), [9, 3, 14]), ), 0)
call_921 = relay.TupleGetItem(func_131_call(relay.reshape(var_920.astype('float32'), [9, 3, 14]), relay.reshape(var_920.astype('float32'), [9, 3, 14]), ), 0)
uop_922 = relay.log(call_919.astype('float64')) # shape=(9, 3, 14)
uop_924 = relay.log(call_921.astype('float64')) # shape=(9, 3, 14)
func_567_call = mod.get_global_var('func_567')
func_570_call = mutated_mod.get_global_var('func_570')
call_928 = relay.TupleGetItem(func_567_call(relay.reshape(var_903.astype('uint32'), [2, 5, 11]), relay.reshape(bop_905.astype('uint32'), [2, 5, 11]), ), 0)
call_929 = relay.TupleGetItem(func_570_call(relay.reshape(var_903.astype('uint32'), [2, 5, 11]), relay.reshape(bop_905.astype('uint32'), [2, 5, 11]), ), 0)
output = relay.Tuple([bop_905,var_920,uop_922,call_928,])
output2 = relay.Tuple([bop_905,var_920,uop_924,call_929,])
func_934 = relay.Function([var_903,var_904,var_920,], output)
mod['func_934'] = func_934
mod = relay.transform.InferType()(mod)
mutated_mod['func_934'] = func_934
mutated_mod = relay.transform.InferType()(mutated_mod)
func_934_call = mutated_mod.get_global_var('func_934')
var_936 = relay.var("var_936", dtype = "float32", shape = (10, 11))#candidate|936|(10, 11)|var|float32
var_937 = relay.var("var_937", dtype = "float32", shape = (10, 11))#candidate|937|(10, 11)|var|float32
var_938 = relay.var("var_938", dtype = "float32", shape = (378,))#candidate|938|(378,)|var|float32
call_935 = func_934_call(var_936,var_937,var_938,)
output = call_935
func_939 = relay.Function([var_936,var_937,var_938,], output)
mutated_mod['func_939'] = func_939
mutated_mod = relay.transform.InferType()(mutated_mod)
var_992 = relay.var("var_992", dtype = "float32", shape = (9, 3, 1))#candidate|992|(9, 3, 1)|var|float32
uop_993 = relay.log(var_992.astype('float32')) # shape=(9, 3, 1)
output = uop_993
output2 = uop_993
func_995 = relay.Function([var_992,], output)
mod['func_995'] = func_995
mod = relay.transform.InferType()(mod)
mutated_mod['func_995'] = func_995
mutated_mod = relay.transform.InferType()(mutated_mod)
var_996 = relay.var("var_996", dtype = "float32", shape = (9, 3, 1))#candidate|996|(9, 3, 1)|var|float32
func_995_call = mutated_mod.get_global_var('func_995')
call_997 = func_995_call(var_996)
output = call_997
func_998 = relay.Function([var_996], output)
mutated_mod['func_998'] = func_998
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1099 = relay.var("var_1099", dtype = "float64", shape = (10, 3, 12))#candidate|1099|(10, 3, 12)|var|float64
uop_1100 = relay.cos(var_1099.astype('float64')) # shape=(10, 3, 12)
var_1102 = relay.var("var_1102", dtype = "float64", shape = (10, 3, 12))#candidate|1102|(10, 3, 12)|var|float64
bop_1103 = relay.subtract(uop_1100.astype('int16'), relay.reshape(var_1102.astype('int16'), relay.shape_of(uop_1100))) # shape=(10, 3, 12)
bop_1111 = relay.mod(uop_1100.astype('float32'), relay.reshape(var_1099.astype('float32'), relay.shape_of(uop_1100))) # shape=(10, 3, 12)
uop_1115 = relay.acosh(bop_1103.astype('float64')) # shape=(10, 3, 12)
output = relay.Tuple([bop_1111,uop_1115,])
output2 = relay.Tuple([bop_1111,uop_1115,])
func_1121 = relay.Function([var_1099,var_1102,], output)
mod['func_1121'] = func_1121
mod = relay.transform.InferType()(mod)
var_1122 = relay.var("var_1122", dtype = "float64", shape = (10, 3, 12))#candidate|1122|(10, 3, 12)|var|float64
var_1123 = relay.var("var_1123", dtype = "float64", shape = (10, 3, 12))#candidate|1123|(10, 3, 12)|var|float64
output = func_1121(var_1122,var_1123,)
func_1124 = relay.Function([var_1122,var_1123,], output)
mutated_mod['func_1124'] = func_1124
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1498 = relay.var("var_1498", dtype = "float32", shape = (14, 9, 4))#candidate|1498|(14, 9, 4)|var|float32
uop_1499 = relay.tan(var_1498.astype('float32')) # shape=(14, 9, 4)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
const_1507 = relay.const([[8.036744,-6.489063,-1.564348,-4.199819,4.488071,-7.556588,4.873872,-6.781946,5.601079,1.255424,-2.614360,-2.183162,-8.188428,5.546136,-9.854354,-1.522229,7.169322,-4.740141,-5.062066,-1.192680,-7.109983,7.727795,-9.754068,0.426100,-5.734199,-1.919518,-5.430785,6.202765,-3.823087,9.824508],[-2.352392,-2.023611,-5.810765,8.742567,3.766722,-3.623369,-2.941649,-3.184515,1.281978,7.615523,-6.808569,-7.507521,-6.029431,3.614411,-1.077365,3.759993,-8.605882,7.012146,8.041220,1.986254,-2.649367,-9.059353,-3.989105,6.008491,-2.487090,4.985865,5.308145,0.909976,1.836953,-9.509606],[2.775437,-0.069441,-1.531208,-7.465884,7.466970,-6.524690,-3.163559,-3.249527,3.298121,-8.661602,-0.509900,0.852082,-6.863097,7.989038,3.633139,2.877996,-7.677790,1.793249,6.186580,-8.009214,6.907571,-9.331332,6.371134,-6.589656,0.558580,-2.848180,1.908501,3.113710,-8.079820,5.150653]], dtype = "float64")#candidate|1507|(3, 30)|const|float64
call_1506 = func_882_call(relay.reshape(const_1507.astype('float64'), [6, 15]))
call_1508 = func_882_call(relay.reshape(const_1507.astype('float64'), [6, 15]))
func_450_call = mod.get_global_var('func_450')
func_452_call = mutated_mod.get_global_var('func_452')
const_1510 = relay.const(1, dtype = "int16")#candidate|1510|()|const|int16
call_1509 = func_450_call(relay.reshape(const_1510.astype('int16'), []))
call_1511 = func_450_call(relay.reshape(const_1510.astype('int16'), []))
var_1520 = relay.var("var_1520", dtype = "int16", shape = (6, 6, 8))#candidate|1520|(6, 6, 8)|var|int16
bop_1521 = relay.bitwise_and(const_1510.astype('int32'), var_1520.astype('int32')) # shape=(6, 6, 8)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
call_1540 = func_882_call(relay.reshape(const_1507.astype('float64'), [6, 15]))
call_1541 = func_882_call(relay.reshape(const_1507.astype('float64'), [6, 15]))
var_1543 = relay.var("var_1543", dtype = "float32", shape = (14, 9, 4))#candidate|1543|(14, 9, 4)|var|float32
bop_1544 = relay.maximum(uop_1499.astype('int64'), relay.reshape(var_1543.astype('int64'), relay.shape_of(uop_1499))) # shape=(14, 9, 4)
output = relay.Tuple([call_1506,const_1507,call_1509,bop_1521,call_1540,bop_1544,])
output2 = relay.Tuple([call_1508,const_1507,call_1511,bop_1521,call_1541,bop_1544,])
func_1547 = relay.Function([var_1498,var_1520,var_1543,], output)
mod['func_1547'] = func_1547
mod = relay.transform.InferType()(mod)
var_1548 = relay.var("var_1548", dtype = "float32", shape = (14, 9, 4))#candidate|1548|(14, 9, 4)|var|float32
var_1549 = relay.var("var_1549", dtype = "int16", shape = (6, 6, 8))#candidate|1549|(6, 6, 8)|var|int16
var_1550 = relay.var("var_1550", dtype = "float32", shape = (14, 9, 4))#candidate|1550|(14, 9, 4)|var|float32
output = func_1547(var_1548,var_1549,var_1550,)
func_1551 = relay.Function([var_1548,var_1549,var_1550,], output)
mutated_mod['func_1551'] = func_1551
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1569 = relay.const([[-3.836891],[7.615705]], dtype = "float32")#candidate|1569|(2, 1)|const|float32
uop_1570 = relay.rsqrt(const_1569.astype('float32')) # shape=(2, 1)
output = uop_1570
output2 = uop_1570
func_1576 = relay.Function([], output)
mod['func_1576'] = func_1576
mod = relay.transform.InferType()(mod)
output = func_1576()
func_1577 = relay.Function([], output)
mutated_mod['func_1577'] = func_1577
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_1619 = func_1576_call()
call_1620 = func_1576_call()
output = call_1619
output2 = call_1620
func_1628 = relay.Function([], output)
mod['func_1628'] = func_1628
mod = relay.transform.InferType()(mod)
output = func_1628()
func_1629 = relay.Function([], output)
mutated_mod['func_1629'] = func_1629
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1628_call = mod.get_global_var('func_1628')
func_1629_call = mutated_mod.get_global_var('func_1629')
call_1643 = func_1628_call()
call_1644 = func_1628_call()
output = relay.Tuple([call_1643,])
output2 = relay.Tuple([call_1644,])
func_1645 = relay.Function([], output)
mod['func_1645'] = func_1645
mod = relay.transform.InferType()(mod)
mutated_mod['func_1645'] = func_1645
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mutated_mod.get_global_var('func_1645')
call_1646 = func_1645_call()
output = call_1646
func_1647 = relay.Function([], output)
mutated_mod['func_1647'] = func_1647
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_1669 = func_1576_call()
call_1670 = func_1576_call()
func_842_call = mod.get_global_var('func_842')
func_846_call = mutated_mod.get_global_var('func_846')
const_1680 = relay.const([-5.097496,9.070335,8.538529,2.617223,9.638187,0.875908,-3.849917,6.000508,3.652339,-8.993509,6.172477,8.873362,-5.802425,-8.237980,4.593745,-7.715615,-8.505166,3.070589,7.020204,0.602430,-3.140122,-3.015783,-1.324826,2.263119,7.106650,4.719564,2.441740,5.679895,5.095767,2.034863,-0.226454,0.327650,6.147421,-1.332924,1.170399,3.715521,-7.815156,0.434602,-0.761047,-8.443156,-1.612010,-1.905927,3.606202,-7.383792,3.712004,7.334915,0.600439,-3.656984,4.426283,9.480936,6.080026,0.955134,6.359800,-5.081373,2.857502,2.021319,5.535864,7.218361,7.457478,7.412027,5.008116,-9.660326,6.146083,4.202515,4.535775,-1.846879,0.110902,0.040520,-6.042250,-2.715864,-5.398705,9.400344,6.733264,-1.693619,-2.823392,4.130091,-0.527970,-2.722104,8.971681,4.937614,8.639187,2.565405,5.308708,3.495296,-4.452048,8.878808,7.502037,-4.220773,1.454212,3.898783,8.610202,4.990221,9.381676,5.465467,5.313548,8.430037,-4.026281,1.040741,7.637156,0.789041,7.251241,2.420111,-7.717911,5.540147,-4.855591,1.371999,-4.356624,-2.939768,3.720884,-2.335291,-2.542930,1.877504,-3.229153,-1.341444,8.787918,7.298624,-3.519736,0.296742,7.012773,-5.635529,0.365921,1.928023,-6.301056,1.235127,1.690549,-1.506478,-8.033423,7.496447,-1.860000,-8.111994,8.219601,-0.631316,-3.994208,-0.709981,7.357191,-8.016290,9.531800,-4.419095,2.027529,9.038684,4.310364,-6.593853,3.230770,-2.987336,2.351051,2.440288,-2.028334,-6.671740,6.167886,-7.121172,3.061945,4.425405,-2.004276,-5.545445,-2.484491,6.094318,0.798883,3.208590,6.395666,-3.669941,-4.884158,0.697418,-2.194569,-9.765775,1.774565,-0.818829,6.197251,7.522795,5.780305,4.768380,2.576252,2.748831,-0.706013,3.857582,-4.210792,-4.375794,6.610858,3.913119,4.366351,1.685637,8.120362,9.790653,5.374384,-3.794338,-9.231891,-0.129831,-1.755264,-1.441351,-7.209726,-6.080864,9.778018,-5.154691,0.767027,-9.998777,1.093802,7.105861,0.286151,4.817618,2.632760,0.482218,9.015626,7.260211,1.020564,-7.300133,8.735842,-0.980733,2.756007,8.803599,3.536665,7.335422,2.526799,9.163600,4.503703,1.490723,-5.434188,-1.825508,0.853134,-1.524152,0.634523,-6.381514,4.991843,7.923537,2.164647,8.331380,4.693532,9.256834,1.502974,2.774313,8.404572,6.131152,-2.085645], dtype = "float32")#candidate|1680|(231,)|const|float32
const_1681 = relay.const([-3.745467,-3.982101,1.001635,-5.464229,-4.889038,8.237346,-8.085041,-9.169535,3.486212,-5.635235,-0.657516,-1.891087,1.210096,0.863285,-1.453907,-5.043749,0.893305,2.114805,3.677448,-0.410892,-3.932108,5.300256,-5.974997,-4.406454,-2.077267,-9.052790,1.335913,-8.744575,3.055495,5.549192,1.382457,7.395524,2.903190,3.311516,-2.311789,-2.382053,-4.770004,7.265043,-2.958681,-1.143033,-2.500186,3.316477,7.564646,-2.071355,-0.557771,-9.658011,-7.426704,-9.707319,6.432399,-1.152795,7.175450,-1.699955,-7.539390,-8.785825,7.996751,2.782843,-8.864261,2.764385,5.481348,-1.496728,-4.337495,7.746519,-3.039481,-5.318827,1.762689,4.046319,6.104171,-5.873749,6.595824,-6.698394,-7.418962,-7.573377,6.189315,-7.519297,-5.730743,-0.174935,-4.615110,5.532325,2.523271,-6.504402,-3.496434,-4.467956,4.910181,0.292052,7.762845,1.318418,-8.558203,-1.652772,-0.625915,-9.868284,-4.821156,0.594831,-7.346177,9.915168,-9.523562,-6.179858,-0.187594,-2.628421,-9.099556,-0.739686,-7.449711,1.598069,-3.861152,-4.154857,-0.019872,9.740298,3.031213,3.968026,7.750316,3.793028,-8.639850,-2.941757,-7.131185,-8.478005,-0.502220,5.784891,6.591171,7.568721,-2.291895,-0.497951,-6.627464,3.926688,-5.353142,2.824398,-5.643933,4.508183,5.044986,0.549133,-3.993913,-2.115136,-0.472412,2.320307,-1.427967,4.733481,-4.795712,5.419242,3.227426,-2.287670,-8.832650,9.654981], dtype = "float64")#candidate|1681|(140,)|const|float64
call_1679 = relay.TupleGetItem(func_842_call(relay.reshape(const_1680.astype('float32'), [7, 11, 3]), relay.reshape(const_1681.astype('float64'), [70, 2]), ), 2)
call_1682 = relay.TupleGetItem(func_846_call(relay.reshape(const_1680.astype('float32'), [7, 11, 3]), relay.reshape(const_1681.astype('float64'), [70, 2]), ), 2)
func_710_call = mod.get_global_var('func_710')
func_714_call = mutated_mod.get_global_var('func_714')
var_1686 = relay.var("var_1686", dtype = "float32", shape = (60,))#candidate|1686|(60,)|var|float32
var_1687 = relay.var("var_1687", dtype = "float32", shape = (780,))#candidate|1687|(780,)|var|float32
call_1685 = relay.TupleGetItem(func_710_call(relay.reshape(var_1686.astype('float32'), [6, 10, 1]), relay.reshape(var_1687.astype('float32'), [6, 10, 13]), relay.reshape(var_1687.astype('float64'), [6, 10, 13]), ), 0)
call_1688 = relay.TupleGetItem(func_714_call(relay.reshape(var_1686.astype('float32'), [6, 10, 1]), relay.reshape(var_1687.astype('float32'), [6, 10, 13]), relay.reshape(var_1687.astype('float64'), [6, 10, 13]), ), 0)
func_842_call = mod.get_global_var('func_842')
func_846_call = mutated_mod.get_global_var('func_846')
call_1700 = relay.TupleGetItem(func_842_call(relay.reshape(const_1680.astype('float32'), [7, 11, 3]), relay.reshape(const_1681.astype('float64'), [70, 2]), ), 0)
call_1701 = relay.TupleGetItem(func_846_call(relay.reshape(const_1680.astype('float32'), [7, 11, 3]), relay.reshape(const_1681.astype('float64'), [70, 2]), ), 0)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_1706 = relay.TupleGetItem(func_1645_call(), 0)
call_1707 = relay.TupleGetItem(func_1647_call(), 0)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_1709 = relay.TupleGetItem(func_1645_call(), 0)
call_1710 = relay.TupleGetItem(func_1647_call(), 0)
output = relay.Tuple([call_1669,call_1679,const_1680,const_1681,call_1685,var_1686,var_1687,call_1700,call_1706,call_1709,])
output2 = relay.Tuple([call_1670,call_1682,const_1680,const_1681,call_1688,var_1686,var_1687,call_1701,call_1707,call_1710,])
func_1711 = relay.Function([var_1686,var_1687,], output)
mod['func_1711'] = func_1711
mod = relay.transform.InferType()(mod)
var_1712 = relay.var("var_1712", dtype = "float32", shape = (60,))#candidate|1712|(60,)|var|float32
var_1713 = relay.var("var_1713", dtype = "float32", shape = (780,))#candidate|1713|(780,)|var|float32
output = func_1711(var_1712,var_1713,)
func_1714 = relay.Function([var_1712,var_1713,], output)
mutated_mod['func_1714'] = func_1714
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_1759 = relay.TupleGetItem(func_1645_call(), 0)
call_1760 = relay.TupleGetItem(func_1647_call(), 0)
output = call_1759
output2 = call_1760
func_1763 = relay.Function([], output)
mod['func_1763'] = func_1763
mod = relay.transform.InferType()(mod)
output = func_1763()
func_1764 = relay.Function([], output)
mutated_mod['func_1764'] = func_1764
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_1776 = func_1576_call()
call_1777 = func_1576_call()
func_1121_call = mod.get_global_var('func_1121')
func_1124_call = mutated_mod.get_global_var('func_1124')
const_1794 = relay.const([-3.209889,-6.602441,-7.328740,-0.416211,-1.185818,-5.262248,-5.333669,-1.207072,2.013276,-5.620045,-8.012311,2.955867,-1.206127,-7.942431,-1.336622,-5.435097,-2.366440,-8.433080,8.810600,4.108719,-7.522763,9.041781,-2.951487,-0.458394,-4.236368,-7.752725,-6.973697,3.517834,-8.944061,-1.469222,9.494715,2.392669,-2.754962,-0.001375,-2.657147,7.102473,0.814489,5.466767,-2.523774,2.174753,-9.818473,6.405301,6.191328,6.896302,2.719312,0.800993,-0.774578,-4.369013,-8.659587,-3.121156,-4.645578,-0.298326,-3.608899,3.808608,9.519181,-0.216992,-3.699915,-2.945380,-1.547975,2.072988,-4.603053,-5.850827,0.581451,9.635862,-0.432942,-1.174447,3.323755,-0.848658,-5.992835,-5.420855,0.699736,2.214505,-5.939927,0.038126,4.132305,-7.900995,1.936093,-9.556145,9.842748,3.942789,6.240167,1.452935,-4.449697,3.578377,1.692197,-4.140820,-1.893687,1.818056,2.327086,0.644950,1.326100,-1.529403,-4.801781,-9.560836,-1.793143,7.768952,1.514419,-0.955151,8.911740,8.489801,6.211169,1.673910,3.783041,6.480221,3.042578,-3.231557,-0.599632,-4.621910,1.173208,6.628578,-7.235046,-1.286378,4.122016,5.688905,-6.989565,0.566548,-7.835031,5.555606,2.536702,-9.028749,1.071908,-7.791931,-6.016587,-3.706821,8.389960,7.949629,7.235283,6.046488,5.077433,5.286125,6.809074,8.252802,3.129649,-7.819341,9.435778,3.784581,4.430338,8.481232,-1.015144,3.992873,0.778899,6.859445,-9.166655,7.104097,5.685696,4.028474,7.481748,-0.595797,3.803018,-0.898758,-2.759724,7.332341,3.329660,2.312087,1.826256,3.302426,0.226739,2.975974,1.602329,2.468085,9.773348,-9.230115,-6.923702,8.132445,0.156664,-9.331823,8.196698,8.999340,-2.974355,3.342364,5.079129,-0.210274,7.536866,2.147180,-4.955257,-5.963811,2.427300,6.659613,2.517443,-3.089184,-4.286373,-5.667297,-4.346198,-6.506355,-2.611059,-3.025831,-7.974885,6.005286,-5.094984,-4.039731,6.194040,-3.715613,5.550364,-7.179618,-3.304869,2.416870,-2.361750,6.231919,4.592788,5.224629,3.260232,-1.904692,4.972536,2.855939,-3.047665,-3.243507,-8.730702,-5.869576,-4.099476,9.918230,2.706853,0.229710,-4.356600,0.233996,-1.370118,-7.653488,-4.145941,0.147800,-0.452866,-4.996729,6.061211,-2.530748,2.398829,5.402156,-8.737236,1.217676,-1.009036,5.463156,0.749260,-3.695991,4.349411,6.835641,2.250363,1.450267,5.137557,-0.741877,7.589633,5.406825,-7.730355,-7.175683,1.352504,-8.769235,0.811655,-4.788545,-2.839285,-6.612553,8.426052,-9.700392,1.876698,-2.768704,7.835560,1.214842,-4.291875,6.486950,8.165028,8.781659,2.698454,-6.249634,3.580260,2.242318,-5.186006,5.260329,6.533138,-7.526687,-0.122122,-4.419240,-1.928198,-7.572700,-9.113136,5.009339,9.805161,-2.324515,-1.597519,2.528160,7.687456,-3.220649,1.971534,-9.569016,2.091701,0.414240,8.752546,6.557816,5.678198,-3.620125,2.572769,9.372666,-7.781832,-2.812651,5.852107,-2.861878,-5.148231,-6.105592,3.481012,4.173360,4.763046,-8.517610,-4.666975,-7.258748,2.143775,4.132264,-1.876021,1.243389,-2.250680,-0.731954,-0.627463,-0.526855,0.881872,-8.982369,8.702637,3.004064,-5.498448,-4.433586,2.903452,-9.062667,4.367550,1.356691,-0.396236,3.457635,-6.577893,-1.331517,-1.565319,-7.325591,3.650454,-7.975304,-5.015847,-1.696835,6.654501,8.220755,8.166354,3.661115,3.089568,1.589826,-2.481002,-0.494117,-0.256680,0.675965,2.092987,-3.254385,-9.320135,-9.341104,-5.911378,2.429379,6.098911,-4.371913,-6.090544,0.288269,-3.259419,5.685553,1.885845,-7.035289,3.030404,5.960779,2.281276,9.606010,-3.875240,-4.826377,-2.359579,2.770902,-0.517279,6.817475], dtype = "float64")#candidate|1794|(360,)|const|float64
call_1793 = relay.TupleGetItem(func_1121_call(relay.reshape(const_1794.astype('float64'), [10, 3, 12]), relay.reshape(const_1794.astype('float64'), [10, 3, 12]), ), 0)
call_1795 = relay.TupleGetItem(func_1124_call(relay.reshape(const_1794.astype('float64'), [10, 3, 12]), relay.reshape(const_1794.astype('float64'), [10, 3, 12]), ), 0)
output = relay.Tuple([call_1776,call_1793,const_1794,])
output2 = relay.Tuple([call_1777,call_1795,const_1794,])
func_1797 = relay.Function([], output)
mod['func_1797'] = func_1797
mod = relay.transform.InferType()(mod)
mutated_mod['func_1797'] = func_1797
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1797_call = mutated_mod.get_global_var('func_1797')
call_1798 = func_1797_call()
output = call_1798
func_1799 = relay.Function([], output)
mutated_mod['func_1799'] = func_1799
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1628_call = mod.get_global_var('func_1628')
func_1629_call = mutated_mod.get_global_var('func_1629')
call_1809 = func_1628_call()
call_1810 = func_1628_call()
output = call_1809
output2 = call_1810
func_1815 = relay.Function([], output)
mod['func_1815'] = func_1815
mod = relay.transform.InferType()(mod)
output = func_1815()
func_1816 = relay.Function([], output)
mutated_mod['func_1816'] = func_1816
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1857 = relay.var("var_1857", dtype = "float32", shape = (1, 1, 6))#candidate|1857|(1, 1, 6)|var|float32
uop_1858 = relay.atanh(var_1857.astype('float32')) # shape=(1, 1, 6)
func_274_call = mod.get_global_var('func_274')
func_279_call = mutated_mod.get_global_var('func_279')
const_1862 = relay.const([-2.482591,3.557151,-1.895444,3.771389,-2.113479,-2.335250,-8.804096,-3.911248,2.631035,1.437506,1.709303,2.282881,8.313333,-4.004543,8.359919,-0.657890,3.127757,-9.394820,1.899949,-4.473582,7.383282,-2.612074,-5.230013,-7.111841,-6.349655,-5.635532,-0.743639,7.372545,6.932927,-6.193219,-1.663575,-3.779325,-0.643727,5.534654,-1.468579,-0.272933,9.196324,4.186750,5.395179,6.328348,-4.438903,7.906943,9.375237,-3.119983,-5.787955,5.525212,5.071149,1.091476,-3.398092,8.762378,-9.033998,-8.590016,-6.676059,9.091071,-5.527426,9.486758,-0.291069,1.718965,-0.448428,-2.898435,-9.325446,5.159439,9.954643,1.971553,6.386680,-9.637392,-1.414987,1.612543,1.326949,-8.610662,9.582257,-4.817913,-3.320740,5.805900,3.101328,-8.664376,-5.911291,2.815533,-3.337698,6.958050,-7.803328,9.921218,8.069596,8.201789,-0.702658,-9.056177,-1.786362,3.987895,-4.540394,3.634627,9.627774,7.027885,1.053584,6.833038,4.755448,8.466894,4.048206,4.240776,-9.282302,0.829259,-0.123702,-6.886884,-8.756509,-2.658872,5.038554,-4.248309,5.750025,-1.808397,-9.280302,-6.356253,-3.256267,1.912030,-8.434922,-6.184299,-6.975410,3.085315,4.855285,5.869715,-9.708579,-8.288660,7.234345,-2.178998,-2.916234,0.783217,-1.706110,6.851189,4.507376,-9.054925,2.598816,-6.400564,-2.176357,-8.137609,-3.078279,6.796003,4.691315,-3.820303,-3.201677,0.379929,5.296231,-3.650571,-6.444745,-3.715041,-3.875628,-2.196571,-6.228330,9.963884,-1.159724,-0.022333,4.827447,0.391380,-0.898809,6.369669,-4.786745,8.842039,2.279045,-7.895347,0.279419,-2.815857,8.780945,1.282850,7.768011,1.417405,0.687189,-4.727498,7.923638,1.907451,9.286883,3.431918,-6.787307,5.343617,-8.612191,1.414559,8.993522,8.881936,-1.829174,-0.366337,-5.385845,-4.391257,5.268918,-2.848272,-2.026485,-0.783718,7.857761,-8.812400,9.951107,2.899881,0.682387,8.802358,-7.130099,5.920711,-0.641113,3.553438,3.487631,-0.827410,-4.767999,-6.800698,-3.382478,4.074737,-8.794123,9.442404,2.310004,5.851320,-0.907816,1.292940,5.093204,5.162625,-7.392006,-6.699335,-3.560264,8.414914,-9.264886,-9.852396,-4.811054,-0.119347,2.046451,-1.428364,-2.674034,4.333085,3.484030,-9.846617,8.286712,-6.267992,-7.868004,-3.419990,0.355298,1.422883,0.343637,-5.082638,9.000775,-2.701305,6.447355,-7.157107,-5.137333,-2.614472,-4.544554,-9.495973,-4.128277,-5.237874,-9.795259,-9.739778,-6.821865,0.385069,9.594919,4.865515,-6.933604,3.766396,1.524627,5.241677,0.353957,2.852906,-4.240240,1.869715,-1.218494,3.905940,-1.382382,-8.173027,-8.452780,-7.638410,9.827743,-8.288090,9.906876,5.917388,9.530465,-0.487759,-7.296758,4.337567,-5.918574,-2.156209,-2.516375,8.794854,1.653643,8.250011,2.973985,4.877554,7.663452,-3.973951,7.809130,5.492068,-2.838772,8.537900,1.865198,-0.353002,3.656642,-0.315687,-0.370115,6.444934,5.436549,7.899650,-1.750305,-0.767475,4.227759,8.182283,-7.763717,4.469497,-4.313372,-3.003860,4.901389,3.189818,2.946499,-0.340929,8.773895,4.907131,-4.340373,9.538733,2.449744,5.055782,-2.897623,8.974815,7.134952,-7.496672,2.514607,-9.886131,5.462715,8.355928,8.184101,-2.540782,5.150056,-0.418030,1.148965,9.520805,5.847617,6.940942,9.665907,-6.783304,0.723218,-4.255299,2.057873,-3.376433,4.208710,7.099713,3.978329,2.354142,4.731620,7.221220,-7.031170,4.695955,5.254334,0.688428,7.891513,6.691746,-1.711854,3.137827,-2.524979,-1.585407,-7.531679,7.200679,-9.956122,-1.984596,-1.385205,-1.107615,-7.439596,-1.529019,-4.892401,-1.895192,7.149610,-4.170207,-9.682114,-6.105568,7.083255,-2.966978], dtype = "float64")#candidate|1862|(360,)|const|float64
var_1863 = relay.var("var_1863", dtype = "float32", shape = (1, 378))#candidate|1863|(1, 378)|var|float32
call_1861 = relay.TupleGetItem(func_274_call(relay.reshape(const_1862.astype('float64'), [8, 3, 15]), relay.reshape(const_1862.astype('float64'), [8, 3, 15]), relay.reshape(var_1863.astype('float32'), [378,]), ), 1)
call_1864 = relay.TupleGetItem(func_279_call(relay.reshape(const_1862.astype('float64'), [8, 3, 15]), relay.reshape(const_1862.astype('float64'), [8, 3, 15]), relay.reshape(var_1863.astype('float32'), [378,]), ), 1)
func_762_call = mod.get_global_var('func_762')
func_764_call = mutated_mod.get_global_var('func_764')
const_1866 = relay.const([[-3.540214],[-7.680422],[5.743759],[-1.544822],[-4.272625],[0.555157],[9.869406],[-8.609458],[-3.068319],[7.365630],[9.466416],[9.291203],[-4.811172],[-0.092455],[-1.870941],[-3.615641],[-8.694064],[9.860039],[8.245758],[0.900776],[-2.065642],[-5.686330],[3.040006],[1.696234],[5.313177],[4.235150],[-0.754158],[-5.733411],[2.980429],[9.983652],[6.560877],[-6.090037],[-5.954772],[9.262880],[-9.906061],[9.866454],[8.208651],[-4.526930],[-9.580628],[-7.317509],[-2.495740],[-3.560170],[-2.215673],[-0.913697],[8.728388],[4.407504],[2.261292],[0.587646],[-7.213146],[2.238960],[7.969967],[5.264280],[-9.452155],[8.670519],[9.486782],[-3.182679],[2.466665],[-2.656080],[7.334063],[-4.009175],[-5.813536],[3.034632],[8.366936],[7.506891],[-6.540106],[-5.442188],[-3.419910],[-3.062311],[-5.443966],[5.341861],[-8.279418],[3.523116],[2.722143],[9.071107],[-1.994478],[0.106342],[-9.799177],[-5.337491],[1.334171],[9.474172],[-6.302572],[-6.568800],[4.889633],[6.703011],[-4.446760],[2.156577],[3.981596],[5.776055],[-4.751603],[7.366664],[-2.667968],[5.758437],[-2.403654],[-6.075655],[1.739082],[7.090105],[-1.985212],[-5.504088],[-2.104365],[2.036448],[8.850172],[4.486092],[1.031656],[4.584266],[-6.678756],[4.509288],[6.958600],[6.337425],[-7.802370],[-3.767873],[7.952061],[-4.022089],[1.517247],[6.613284],[-7.009981],[-7.048767],[5.887506],[-1.979769],[-6.453205],[-2.715882],[9.844430],[1.787337],[-9.811117],[-3.506132],[5.753833],[-8.972646],[-0.082102],[-3.483782],[-7.144299],[7.494353],[-1.931716],[-0.819277],[-6.773634],[-8.492010],[-7.676207],[0.519079],[-2.721617],[-2.786662],[2.709721],[-3.925522]], dtype = "float64")#candidate|1866|(140, 1)|const|float64
call_1865 = func_762_call(relay.reshape(const_1866.astype('float64'), [14, 10]))
call_1867 = func_762_call(relay.reshape(const_1866.astype('float64'), [14, 10]))
output = relay.Tuple([uop_1858,call_1861,const_1862,var_1863,call_1865,const_1866,])
output2 = relay.Tuple([uop_1858,call_1864,const_1862,var_1863,call_1867,const_1866,])
func_1881 = relay.Function([var_1857,var_1863,], output)
mod['func_1881'] = func_1881
mod = relay.transform.InferType()(mod)
mutated_mod['func_1881'] = func_1881
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1881_call = mutated_mod.get_global_var('func_1881')
var_1883 = relay.var("var_1883", dtype = "float32", shape = (1, 1, 6))#candidate|1883|(1, 1, 6)|var|float32
var_1884 = relay.var("var_1884", dtype = "float32", shape = (1, 378))#candidate|1884|(1, 378)|var|float32
call_1882 = func_1881_call(var_1883,var_1884,)
output = call_1882
func_1885 = relay.Function([var_1883,var_1884,], output)
mutated_mod['func_1885'] = func_1885
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_1901 = relay.TupleGetItem(func_1645_call(), 0)
call_1902 = relay.TupleGetItem(func_1647_call(), 0)
func_241_call = mod.get_global_var('func_241')
func_244_call = mutated_mod.get_global_var('func_244')
const_1922 = relay.const([-1,4,-3,7,-10,1,-7,3,-8,-2,-6,-7,-1,-8,9,1,-6,-3,2,6,8,7,4,-5,-7,-5,6,-1,-6,7,2,-5,-1,-7,8,-6,-3,-4,-3,9,5,-2,-7,5,-9,9,10,-2,-4,-5,-4,-1,-4,-4,-5,6,1,1,10,-1,10,-2,4,9,-5,-2,-5,-6,9,7,-4,-4,-10,1,10,-9,-4,9,5,7,4,8,2,10,1,3,-10,-10,-1,10,-5,-10,-2,-8,-10,-8,2,-6,8,-6,-5,6,-6,-8,1,8,1,-6,8,-9,-7,-3,-10,5,-6,9,-10,9,-5,-4,-8,-2,-2,-1,2,-1,6,-2,-2,6,-8,1,2,-7,-8,-1,5,7,-4,10,-8,-5,-2,9,-3,-3,-10,2,-6,-8,2,-5,5,-4,10,-4,-2,1,7,-1,-6,3,1,-7,4,3,7,3,-10,10,9,-6,-5,7,6,-2,4,-8,3,-2,-3,5,3,-3,-2,-10,-8,-6,-3,5,-2,-4,4,-9,-9,4,-6,-5,6,1,-2,-3,-9,-3,1,-6,7,-9,-3,-8,9,-6,2,5,9,5,9,4,-6,-2,5,10,-9,-2,7,-5,-1,-3,-2,8,4,-5,-3,3,-5,8,3,3,9,4,-5,-4,2,7,8,7,3,5,2,7,2,-1,-5,9,-2,-2,4,3,-5,-9,-10,-5,2,-10,9,-4,-5,10,-3,-8,3,-2,-7,-2,7,-7,10,9,3,6,9,5,-7,7,-3,5,-6,3,-3,-10,-8,7,-7,10,8,-4,-8,-4,-1,6,-8,9,7,-9,-1,-1,4,-7,-4,4,2,10,4,-4,-5], dtype = "int32")#candidate|1922|(315,)|const|int32
call_1921 = relay.TupleGetItem(func_241_call(relay.reshape(const_1922.astype('int32'), [5, 7, 9]), relay.reshape(const_1922.astype('int32'), [5, 7, 9]), ), 7)
call_1923 = relay.TupleGetItem(func_244_call(relay.reshape(const_1922.astype('int32'), [5, 7, 9]), relay.reshape(const_1922.astype('int32'), [5, 7, 9]), ), 7)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_1939 = func_1576_call()
call_1940 = func_1576_call()
output = relay.Tuple([call_1901,call_1921,const_1922,call_1939,])
output2 = relay.Tuple([call_1902,call_1923,const_1922,call_1940,])
func_1944 = relay.Function([], output)
mod['func_1944'] = func_1944
mod = relay.transform.InferType()(mod)
output = func_1944()
func_1945 = relay.Function([], output)
mutated_mod['func_1945'] = func_1945
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1944_call = mod.get_global_var('func_1944')
func_1945_call = mutated_mod.get_global_var('func_1945')
call_1951 = relay.TupleGetItem(func_1944_call(), 0)
call_1952 = relay.TupleGetItem(func_1945_call(), 0)
output = call_1951
output2 = call_1952
func_1953 = relay.Function([], output)
mod['func_1953'] = func_1953
mod = relay.transform.InferType()(mod)
mutated_mod['func_1953'] = func_1953
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mutated_mod.get_global_var('func_1953')
call_1954 = func_1953_call()
output = call_1954
func_1955 = relay.Function([], output)
mutated_mod['func_1955'] = func_1955
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1628_call = mod.get_global_var('func_1628')
func_1629_call = mutated_mod.get_global_var('func_1629')
call_1975 = func_1628_call()
call_1976 = func_1628_call()
output = relay.Tuple([call_1975,])
output2 = relay.Tuple([call_1976,])
func_1978 = relay.Function([], output)
mod['func_1978'] = func_1978
mod = relay.transform.InferType()(mod)
output = func_1978()
func_1979 = relay.Function([], output)
mutated_mod['func_1979'] = func_1979
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_1988 = relay.TupleGetItem(func_1645_call(), 0)
call_1989 = relay.TupleGetItem(func_1647_call(), 0)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
var_2014 = relay.var("var_2014", dtype = "float64", shape = (15, 6))#candidate|2014|(15, 6)|var|float64
call_2013 = func_882_call(relay.reshape(var_2014.astype('float64'), [6, 15]))
call_2015 = func_882_call(relay.reshape(var_2014.astype('float64'), [6, 15]))
func_1547_call = mod.get_global_var('func_1547')
func_1551_call = mutated_mod.get_global_var('func_1551')
const_2019 = relay.const([[3.798635,-7.541333,7.602742,-0.901436,-3.194906,7.344062,-5.979825,9.204743,-2.223083,-2.286247,-7.795387,-6.823452,8.186079,-4.358903,-4.480901,-5.706984,-6.119370,5.543539,-2.143045,-5.289374,-8.932052,-5.401815,-0.569831,-1.720954,9.046150,-9.980708,-0.864432,-3.957934,-9.531162,0.060376,-9.717968,-3.561113,-0.085524,-7.456292,4.478639,3.846339,3.450771,-5.883131,6.543354,-1.997534,4.276625,6.119038,5.808208,2.013119,-7.321406,2.142312,-4.606389,9.690494,2.520031,-9.672399,-8.860986,-9.698455,7.145650,-3.991830,3.958543,1.867196,3.772957,-0.326372,-8.748577,-0.166810,3.448177,-6.448206,9.065809,-1.957095,2.109393,5.165624,1.708034,2.013185,-6.698291,0.955544,6.758753,7.727174,1.198498,5.191779,2.469831,-6.401721,6.406229,6.456860,-1.004181,0.899755,-9.204225,-3.586337,2.726752,-2.922808],[2.548351,0.855491,2.868293,4.808007,7.461189,-0.841174,-0.797619,7.843617,5.463026,-2.661447,-1.248684,-6.378385,-9.309358,-3.656117,-2.117415,4.839249,1.180878,-9.985896,4.141277,9.322176,3.520503,-8.777720,1.276149,0.196425,-6.334125,-1.403524,-7.881070,6.617821,8.329663,-4.474576,-1.068427,6.650941,-9.890685,-2.784053,5.740064,-4.272168,-5.669811,-3.805281,-9.154056,-3.509420,3.907807,-3.671443,0.074435,-3.639728,1.579939,3.545725,4.279902,8.824155,-3.262129,4.702437,-5.570454,-1.305447,-9.726376,-5.817421,0.115899,5.113768,6.144575,-7.602688,6.392985,-2.108599,2.664150,7.281497,-9.603900,-9.669552,-4.675780,0.236963,-9.429670,5.419142,-2.697193,8.312107,1.661924,2.746814,-3.025957,7.999178,-5.728845,0.816936,5.056155,-8.820765,1.802666,8.238635,7.782741,-9.233790,-6.666610,-7.329685],[0.589539,-9.046106,0.685687,-2.137505,9.651893,5.622489,7.041771,2.898360,5.152645,-3.249572,5.654643,2.992356,-0.345773,-0.052964,5.318749,-3.687248,-0.408184,9.246894,-3.875049,-3.607521,-0.962089,1.864765,-8.040017,-0.112406,8.112622,-8.001023,-6.491372,3.001435,-2.711791,-6.360213,-7.715604,-2.830714,9.150185,-2.157255,0.214396,-8.682857,2.907938,5.229601,-7.568447,5.673002,9.816674,-7.406591,1.218684,-1.734045,7.467330,-6.784720,2.897919,8.284742,-2.728445,-7.626341,-3.100046,1.181223,3.279447,8.825527,1.087977,5.459418,-3.841739,-6.457147,4.815198,8.597526,4.502837,-6.541627,8.440753,-1.816040,-2.237206,5.277436,8.230346,-4.929148,-2.500789,-8.084189,-2.031472,-8.232338,-5.546287,2.058643,9.516430,-8.997559,-2.793010,-0.376855,-8.944359,4.305964,0.989498,2.676840,-3.585782,0.015728],[2.723140,-9.067355,3.834045,3.226941,9.243821,3.257521,4.366308,-7.653424,7.919462,5.148112,8.273889,-1.745965,-7.331362,3.004588,9.617869,0.612829,-3.531398,-4.932201,-1.506350,1.627398,-2.536905,-9.635724,3.794719,5.946514,4.945720,-6.541846,-8.463549,-0.001732,0.976139,2.294660,-9.547585,8.466877,-2.282299,3.698144,6.834080,-8.004140,8.994312,-9.048321,9.989661,9.735862,-7.928865,7.913790,-6.174692,3.932741,1.240187,-5.332702,6.139239,2.603278,-6.689262,-6.948823,0.088612,5.224586,-7.096098,9.944001,-8.252481,-6.983780,5.565058,9.751655,0.640789,4.404837,6.576367,5.887523,1.423962,-1.582995,1.192539,-5.551568,8.136514,3.106542,6.821483,-2.195931,-4.796061,-8.936065,-4.509613,1.140045,4.328152,0.910047,4.507977,6.244045,6.294911,-3.183629,7.341918,0.993964,-7.487820,-2.794169],[-5.696686,5.916911,2.999372,-9.250725,4.540431,-5.810396,-8.840977,3.820779,-0.084064,-6.964948,-1.826281,-8.582467,1.793900,9.127718,-8.695078,5.226491,-8.382160,8.056726,7.934325,1.731549,-4.226307,1.650494,2.247956,-7.067057,4.350988,0.826158,4.799569,7.973320,-6.791096,-1.717645,-4.112890,1.635569,5.591893,2.601814,-6.026634,6.847278,-0.399290,7.729200,-7.525679,2.173734,-0.224195,5.674116,-0.196655,0.323095,5.134034,8.797930,0.613268,-5.860944,-0.268140,2.726270,3.526420,0.808087,8.581157,6.434749,-5.980358,-8.792179,-9.406932,-1.715663,-3.492518,7.756489,1.127635,-9.192801,8.458090,2.309527,-4.165687,-3.829538,-0.091303,4.309597,-4.258580,4.368885,2.131277,9.662305,7.447009,-3.529049,7.083669,-6.183466,-5.613149,-7.562677,0.544634,4.396218,-8.942930,1.644635,2.897589,3.105844],[-9.128314,-2.523529,-4.171071,3.042353,7.294640,6.810675,9.804448,-7.078676,-6.343705,4.999369,-2.407031,2.615154,-7.606815,1.425012,0.458859,4.569793,6.527490,8.418000,8.182348,-6.784121,-1.642032,-9.187853,2.628987,-0.259740,7.290318,1.017928,-5.984595,-4.871911,-4.087555,7.654591,-1.096754,9.777333,-6.478412,6.121827,1.277645,0.889810,-0.802498,0.256462,-8.108670,5.096636,9.862835,2.338283,-0.134384,-3.165361,-1.498827,-8.116172,-6.998124,-8.913619,7.673010,-3.128739,0.414184,-1.997394,8.400479,3.704124,7.447958,7.605834,-0.945971,2.940764,9.271114,2.495060,4.113011,-0.690637,7.253185,-4.388975,5.540736,2.342227,-0.423047,-9.160184,4.361734,4.980052,-2.867300,9.784547,-9.398044,8.383177,-1.053073,5.024536,1.014895,3.962674,-8.944978,4.271580,-3.846606,8.858748,-4.707780,8.354253]], dtype = "float32")#candidate|2019|(6, 84)|const|float32
const_2020 = relay.const([-7,5,6,-1,2,-4,-6,2,8,8,-7,7,10,1,1,3,9,-4,-3,8,4,6,9,-9,-4,-7,7,-4,-5,7,9,6,8,-6,4,8,6,-7,2,5,5,10,-8,-9,9,9,-5,-9,4,6,7,-5,6,1,-9,7,-10,6,-1,-1,8,-8,4,8,3,7,5,-7,5,10,-4,-8,4,-2,-8,3,2,10,1,3,9,-5,1,7,8,10,6,8,-3,-3,3,6,1,1,-4,-7,-6,9,-4,5,-6,6,4,-2,-4,9,-4,5,3,-7,-4,-9,-9,-10,-5,-5,-4,2,4,-7,8,-8,-1,8,-10,-10,-1,5,-9,3,1,-2,-8,3,8,-2,5,9,-4,-1,-3,-9,2,-4,-10,-5,-4,8,-3,6,2,-8,-8,7,3,6,-1,-2,-8,3,5,1,-7,3,-9,-1,-5,-9,5,4,-4,-6,5,9,4,1,10,-9,8,-6,1,9,-4,10,6,9,-4,6,-5,-3,7,-8,-4,3,-9,9,6,-5,-8,-2,-8,-6,-8,-2,-7,3,8,10,-10,-4,-2,10,-2,-8,1,7,-3,-2,5,-5,10,6,-1,10,1,-2,-8,3,-9,-2,-3,10,2,-8,-7,-1,2,6,6,-5,-9,-2,-1,-7,-7,1,-8,7,-2,-6,7,7,-7,-5,9,9,-8,5,5,-10,-10,4,1,-9,4,1,5,5,2,-4,-1,-9,4,-10,7,-3,2,-7,9,-3,1,6,-3,7,-8,-6,-4,8], dtype = "int16")#candidate|2020|(288,)|const|int16
call_2018 = relay.TupleGetItem(func_1547_call(relay.reshape(const_2019.astype('float32'), [14, 9, 4]), relay.reshape(const_2020.astype('int16'), [6, 6, 8]), relay.reshape(const_2019.astype('float32'), [14, 9, 4]), ), 3)
call_2021 = relay.TupleGetItem(func_1551_call(relay.reshape(const_2019.astype('float32'), [14, 9, 4]), relay.reshape(const_2020.astype('int16'), [6, 6, 8]), relay.reshape(const_2019.astype('float32'), [14, 9, 4]), ), 3)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
call_2045 = func_882_call(relay.reshape(var_2014.astype('float64'), [6, 15]))
call_2046 = func_882_call(relay.reshape(var_2014.astype('float64'), [6, 15]))
func_934_call = mod.get_global_var('func_934')
func_939_call = mutated_mod.get_global_var('func_939')
var_2055 = relay.var("var_2055", dtype = "float32", shape = (110,))#candidate|2055|(110,)|var|float32
var_2056 = relay.var("var_2056", dtype = "float32", shape = (1, 378))#candidate|2056|(1, 378)|var|float32
call_2054 = relay.TupleGetItem(func_934_call(relay.reshape(var_2055.astype('float32'), [10, 11]), relay.reshape(var_2055.astype('float32'), [10, 11]), relay.reshape(var_2056.astype('float32'), [378,]), ), 1)
call_2057 = relay.TupleGetItem(func_939_call(relay.reshape(var_2055.astype('float32'), [10, 11]), relay.reshape(var_2055.astype('float32'), [10, 11]), relay.reshape(var_2056.astype('float32'), [378,]), ), 1)
bop_2086 = relay.add(call_2018.astype('uint8'), relay.reshape(const_2020.astype('uint8'), relay.shape_of(call_2018))) # shape=(6, 6, 8)
bop_2089 = relay.add(call_2021.astype('uint8'), relay.reshape(const_2020.astype('uint8'), relay.shape_of(call_2021))) # shape=(6, 6, 8)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
call_2102 = func_882_call(relay.reshape(call_2013.astype('float64'), [6, 15]))
call_2103 = func_882_call(relay.reshape(call_2013.astype('float64'), [6, 15]))
output = relay.Tuple([call_1988,call_2013,var_2014,const_2019,call_2045,call_2054,var_2055,var_2056,bop_2086,call_2102,])
output2 = relay.Tuple([call_1989,call_2015,var_2014,const_2019,call_2046,call_2057,var_2055,var_2056,bop_2089,call_2103,])
func_2126 = relay.Function([var_2014,var_2055,var_2056,], output)
mod['func_2126'] = func_2126
mod = relay.transform.InferType()(mod)
var_2127 = relay.var("var_2127", dtype = "float64", shape = (15, 6))#candidate|2127|(15, 6)|var|float64
var_2128 = relay.var("var_2128", dtype = "float32", shape = (110,))#candidate|2128|(110,)|var|float32
var_2129 = relay.var("var_2129", dtype = "float32", shape = (1, 378))#candidate|2129|(1, 378)|var|float32
output = func_2126(var_2127,var_2128,var_2129,)
func_2130 = relay.Function([var_2127,var_2128,var_2129,], output)
mutated_mod['func_2130'] = func_2130
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1797_call = mod.get_global_var('func_1797')
func_1799_call = mutated_mod.get_global_var('func_1799')
call_2158 = relay.TupleGetItem(func_1797_call(), 1)
call_2159 = relay.TupleGetItem(func_1799_call(), 1)
func_450_call = mod.get_global_var('func_450')
func_452_call = mutated_mod.get_global_var('func_452')
var_2166 = relay.var("var_2166", dtype = "int16", shape = ())#candidate|2166|()|var|int16
call_2165 = func_450_call(relay.reshape(var_2166.astype('int16'), []))
call_2167 = func_450_call(relay.reshape(var_2166.astype('int16'), []))
var_2174 = relay.var("var_2174", dtype = "float32", shape = (10, 3, 12))#candidate|2174|(10, 3, 12)|var|float32
bop_2175 = relay.less_equal(call_2158.astype('bool'), relay.reshape(var_2174.astype('bool'), relay.shape_of(call_2158))) # shape=(10, 3, 12)
bop_2178 = relay.less_equal(call_2159.astype('bool'), relay.reshape(var_2174.astype('bool'), relay.shape_of(call_2159))) # shape=(10, 3, 12)
func_1711_call = mod.get_global_var('func_1711')
func_1714_call = mutated_mod.get_global_var('func_1714')
const_2181 = relay.const([[-6.245281,9.350962],[-8.578291,-3.932494],[-2.694146,-0.518993],[6.083825,-5.814904],[-0.943741,9.571909],[6.171210,-9.623944],[8.234890,6.756923],[-4.812616,-0.655002],[-5.098278,-5.146501],[4.447983,-4.320282],[-8.611765,7.834454],[-4.415023,-5.152999],[4.240455,8.336101],[9.749881,0.440379],[0.370449,3.419597],[2.858065,-8.830025],[0.122112,1.623945],[7.728196,-8.054039],[-2.124259,-6.474849],[1.172452,1.616116],[-6.513526,-7.232298],[-8.910966,5.046459],[-1.837756,2.045743],[-3.871020,-6.877271],[2.925585,0.384527],[8.318400,-4.194430],[1.490584,-6.991055],[2.690490,6.875726],[-4.812422,-6.545016],[6.050797,-1.887641]], dtype = "float32")#candidate|2181|(30, 2)|const|float32
var_2182 = relay.var("var_2182", dtype = "float32", shape = (780,))#candidate|2182|(780,)|var|float32
call_2180 = relay.TupleGetItem(func_1711_call(relay.reshape(const_2181.astype('float32'), [60,]), relay.reshape(var_2182.astype('float32'), [780,]), ), 0)
call_2183 = relay.TupleGetItem(func_1714_call(relay.reshape(const_2181.astype('float32'), [60,]), relay.reshape(var_2182.astype('float32'), [780,]), ), 0)
output = relay.Tuple([call_2165,var_2166,bop_2175,call_2180,const_2181,var_2182,])
output2 = relay.Tuple([call_2167,var_2166,bop_2178,call_2183,const_2181,var_2182,])
func_2193 = relay.Function([var_2166,var_2174,var_2182,], output)
mod['func_2193'] = func_2193
mod = relay.transform.InferType()(mod)
mutated_mod['func_2193'] = func_2193
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2193_call = mutated_mod.get_global_var('func_2193')
var_2195 = relay.var("var_2195", dtype = "int16", shape = ())#candidate|2195|()|var|int16
var_2196 = relay.var("var_2196", dtype = "float32", shape = (10, 3, 12))#candidate|2196|(10, 3, 12)|var|float32
var_2197 = relay.var("var_2197", dtype = "float32", shape = (780,))#candidate|2197|(780,)|var|float32
call_2194 = func_2193_call(var_2195,var_2196,var_2197,)
output = call_2194
func_2198 = relay.Function([var_2195,var_2196,var_2197,], output)
mutated_mod['func_2198'] = func_2198
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2230 = relay.var("var_2230", dtype = "float32", shape = (6, 14, 7))#candidate|2230|(6, 14, 7)|var|float32
var_2231 = relay.var("var_2231", dtype = "float32", shape = (6, 14, 7))#candidate|2231|(6, 14, 7)|var|float32
bop_2232 = relay.floor_mod(var_2230.astype('float32'), relay.reshape(var_2231.astype('float32'), relay.shape_of(var_2230))) # shape=(6, 14, 7)
output = relay.Tuple([bop_2232,])
output2 = relay.Tuple([bop_2232,])
func_2235 = relay.Function([var_2230,var_2231,], output)
mod['func_2235'] = func_2235
mod = relay.transform.InferType()(mod)
mutated_mod['func_2235'] = func_2235
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2235_call = mutated_mod.get_global_var('func_2235')
var_2237 = relay.var("var_2237", dtype = "float32", shape = (6, 14, 7))#candidate|2237|(6, 14, 7)|var|float32
var_2238 = relay.var("var_2238", dtype = "float32", shape = (6, 14, 7))#candidate|2238|(6, 14, 7)|var|float32
call_2236 = func_2235_call(var_2237,var_2238,)
output = call_2236
func_2239 = relay.Function([var_2237,var_2238,], output)
mutated_mod['func_2239'] = func_2239
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1978_call = mod.get_global_var('func_1978')
func_1979_call = mutated_mod.get_global_var('func_1979')
call_2269 = relay.TupleGetItem(func_1978_call(), 0)
call_2270 = relay.TupleGetItem(func_1979_call(), 0)
func_807_call = mod.get_global_var('func_807')
func_810_call = mutated_mod.get_global_var('func_810')
const_2272 = relay.const([[2.688430,3.588183],[-4.743329,-6.561711],[6.231878,1.675139],[3.513667,-7.721504],[9.723733,4.684067],[-9.917023,-1.358349],[7.205102,9.682620],[-2.560347,-2.726695],[3.894303,3.220208],[-8.293918,6.881290],[-2.954328,-3.144419],[5.277736,-3.690553],[4.145186,-4.429347],[8.735217,-9.175026],[-1.473748,-2.566714],[1.463273,-9.490208],[-8.339774,-2.870429],[4.018393,4.268857],[3.251910,-8.726160],[-0.547504,5.438717],[-7.507411,5.272877],[-7.944500,-9.650248],[-3.803108,-3.324012],[8.753916,-5.359962],[-4.764412,-7.031159],[-6.912457,6.964229],[1.401417,-5.781616],[-1.828141,2.994650],[3.549435,-2.262979],[6.668126,3.914664],[-2.191137,3.045699],[-4.684543,3.273138],[-9.992502,8.067248],[8.961305,-8.988929],[-1.211241,-5.225047],[9.224342,1.402554],[3.921067,-7.725193],[-5.632272,-8.348871],[9.906124,-4.014610],[2.606238,-5.594328],[-8.885691,-0.936630],[-8.225961,-2.722252],[9.375978,5.751559],[5.462825,8.039985],[7.317453,8.381899],[8.726120,-8.282561],[6.930118,6.633052],[-6.224527,-1.578900],[5.314942,8.083255],[-3.405107,-4.787070],[-2.624755,-8.620443],[-8.499593,-7.061488],[2.696422,4.585537],[7.517091,-9.583970],[-9.246085,5.077124],[6.220688,9.089965],[-3.148195,0.180690],[6.951049,-5.856273],[7.512602,8.434080],[7.233399,3.934776],[5.464820,2.568361],[-4.247884,5.852119],[7.066499,-3.347142],[-2.756162,2.037024],[-0.441296,9.570003],[-3.836038,3.756007],[-1.790663,0.889222],[3.980480,6.446459],[1.728044,3.821659],[-6.345982,9.481222],[-1.944040,-0.782524],[9.007602,8.456461],[-3.680776,7.137445],[9.794350,6.396849],[3.584196,2.221733],[9.932200,5.701612],[5.612605,3.957827],[4.117580,-4.691810],[-2.228916,4.343169],[-5.478324,4.141100],[9.356656,-9.632787],[3.369741,9.547352],[-8.519465,1.150777],[3.945507,4.390976],[-4.240563,-0.319627],[5.920767,0.806553],[3.970426,3.888019],[7.172772,6.133347]], dtype = "float64")#candidate|2272|(88, 2)|const|float64
call_2271 = relay.TupleGetItem(func_807_call(relay.reshape(const_2272.astype('float64'), [8, 11, 2])), 1)
call_2273 = relay.TupleGetItem(func_810_call(relay.reshape(const_2272.astype('float64'), [8, 11, 2])), 1)
bop_2281 = relay.mod(const_2272.astype('float32'), relay.reshape(call_2271.astype('float32'), relay.shape_of(const_2272))) # shape=(88, 2)
bop_2284 = relay.mod(const_2272.astype('float32'), relay.reshape(call_2273.astype('float32'), relay.shape_of(const_2272))) # shape=(88, 2)
uop_2285 = relay.sinh(call_2271.astype('float64')) # shape=(8, 11, 2)
uop_2287 = relay.sinh(call_2273.astype('float64')) # shape=(8, 11, 2)
func_1711_call = mod.get_global_var('func_1711')
func_1714_call = mutated_mod.get_global_var('func_1714')
const_2290 = relay.const([[4.370745],[6.664331],[0.983255],[7.683996],[0.385760],[-8.494175],[-7.073377],[-3.360822],[7.271718],[-9.129224],[-0.154237],[-0.562152],[-4.562649],[6.044368],[-5.157281],[-1.499383],[8.503680],[6.643135],[-5.900702],[-9.763360],[4.136891],[-4.402706],[0.902439],[-0.140995],[-8.112693],[3.778925],[7.868659],[-4.375749],[2.660542],[4.056391],[-2.345190],[3.285845],[7.412875],[-3.794252],[-0.744797],[9.764199],[0.239732],[6.782795],[1.216796],[1.839925],[0.114943],[-5.754523],[3.134386],[-1.857682],[-2.174987],[-7.328104],[3.459050],[-1.902617],[8.814738],[-6.119855],[0.578942],[7.921008],[7.495098],[8.577531],[8.775018],[-9.234296],[-3.333386],[9.609939],[6.843869],[-6.708158]], dtype = "float32")#candidate|2290|(60, 1)|const|float32
const_2291 = relay.const([9.123969,6.131690,-0.758600,-0.674462,-5.006417,6.082738,-0.440734,-8.632417,0.428813,-1.926670,3.779462,5.189209,-3.581488,-0.343137,3.086337,-0.718316,5.968299,3.956454,-5.135278,-8.661259,9.549437,-8.958655,-8.522272,8.189197,-6.338684,2.526365,-4.305710,-7.161842,8.274019,-9.536744,-1.214280,3.721095,1.895461,-8.157791,-3.327906,-8.940338,-0.007423,8.005662,6.926297,-3.731955,3.842408,9.675910,8.173578,-3.990114,-6.604697,7.470925,5.145320,-3.117857,2.890437,-9.528043,6.397974,4.838890,-4.535691,3.583539,-0.705004,5.315294,4.083819,9.848598,-5.419250,4.423008,-2.582396,-9.029727,9.970538,-2.735143,6.737605,-3.118516,2.581535,-1.602362,9.036717,2.392151,1.857111,3.904788,3.600834,8.055900,-7.127065,-4.515896,9.862560,-1.962416,-8.515090,3.147891,-7.431563,-9.736399,-4.163979,0.992314,-0.585866,1.974108,8.328217,8.576603,2.819509,-5.775765,2.204401,-0.248706,9.082365,-1.412460,-4.632383,7.060703,3.608650,-7.165114,-8.196134,3.590965,0.770384,3.066576,2.741052,1.711891,7.333629,-2.733510,4.644721,-6.739767,-5.099755,-7.036113,-2.657310,4.573080,8.159245,1.733052,3.086928,8.217424,1.342878,7.508424,0.878577,3.979547,-3.302095,5.871948,9.681369,0.303566,-3.657793,-8.907426,3.549331,4.015125,8.899777,4.072287,4.501767,6.369457,7.408663,-4.816200,-0.840291,2.522256,1.854976,-0.150574,-3.484756,-3.294138,7.637216,-3.615015,-2.928663,0.804550,-6.599604,-7.718930,6.085199,6.324994,-0.148161,4.311918,2.258645,-1.411730,-0.459195,2.065776,-8.995320,8.901169,-3.891154,-2.537843,2.428506,0.764117,-4.475368,-4.698072,7.759678,-9.592408,9.853642,-9.569790,8.553140,5.003319,1.700323,-3.558002,7.991652,2.248915,3.214985,-8.204508,-1.186351,0.419066,-9.196884,-9.095021,5.290257,0.061304,3.665905,7.617032,-1.284837,-3.838321,-3.921371,0.039159,-3.593281,6.538033,3.426059,6.105809,5.193653,-6.698077,-7.571551,-3.515480,-7.375814,-5.570458,1.502756,-6.045223,-2.337253,-4.649192,6.662011,-3.767958,-4.865086,-8.377062,-8.185157,6.794079,-8.916554,7.497734,8.538408,-3.534892,6.238832,9.018064,-3.746489,2.015190,-6.731223,7.169537,8.960438,0.696737,-6.528891,4.785801,3.268719,-6.300593,4.790599,-8.756837,6.279189,-9.348982,0.598334,-6.352170,4.851758,0.031640,2.434021,2.634119,-7.086360,-2.615801,-0.364805,3.090301,5.643549,2.081190,6.384977,1.643286,-9.760901,-2.061814,6.574140,-9.671643,-7.611914,-3.414401,-2.017911,-3.415295,-5.189275,9.162536,8.014344,6.883636,3.130539,2.487610,-9.791320,6.880962,-9.069532,9.258429,-4.394527,-1.639782,0.084607,-3.505974,-0.037968,-9.560873,-0.163963,-3.166295,3.878883,-6.252756,-8.901766,1.661463,-9.591458,-5.705974,-8.816956,-8.416409,1.381677,3.457836,5.535680,-9.159943,4.058800,-0.410689,0.113189,3.609270,-7.214390,8.051847,4.072627,-3.904693,4.491799,-5.232616,-6.506677,6.872263,2.207270,-4.289424,-0.833646,-0.427349,3.196896,7.901199,5.746448,2.630294,-2.524516,0.628881,-8.154065,4.635290,-2.340205,9.966498,-6.670968,-5.916620,-4.620327,-5.278007,7.277383,-8.699662,-4.738824,9.469617,-6.831527,1.487571,5.572660,-2.366526,-2.709488,-1.675700,5.396995,6.809815,-0.505809,8.096594,-9.780223,6.905095,-7.472376,9.115723,-7.181494,4.459403,5.867773,9.281982,6.673609,-0.790144,-8.459459,-2.944733,4.851554,8.614926,7.828845,-9.748952,6.489636,2.143981,9.672946,-7.614702,-8.493746,-8.801572,-0.251348,-7.456186,9.991602,-5.882413,9.884640,0.334982,-4.486443,-9.183500,7.542580,-7.093303,0.482905,-2.230856,-8.419474,-6.575694,-6.687806,-9.376189,-1.812764,-3.483366,-5.325865,2.412400,8.685112,-0.972837,-1.606177,5.679406,0.643686,-0.280422,-4.258867,8.538510,2.335304,0.125405,-7.506972,-8.924756,-3.458495,-5.478064,3.440800,2.554135,-6.633131,9.180234,-8.531231,-1.052987,-9.668466,-6.477957,2.307100,2.341506,0.187522,7.953627,-7.264666,7.400423,-2.547281,3.229038,8.156434,4.905887,-1.383994,-9.538831,4.370649,-7.401096,9.808878,-9.701593,-0.310931,7.063596,-4.194576,4.263690,2.574171,-5.520884,-4.072161,6.485153,-5.169660,3.034757,2.000408,2.262767,-7.879048,-9.611318,4.234221,7.931603,-2.232354,3.301017,1.267024,-2.239520,7.339172,-0.249433,-6.862510,-1.011872,-8.714771,-8.092108,0.764225,6.227955,5.580822,-0.862753,-0.671782,8.298878,-1.550563,-6.574914,0.129531,-8.968896,9.020021,8.815756,9.121158,-6.882903,5.887072,2.349306,-7.669298,0.771796,4.339027,-0.580286,0.699657,-0.997689,4.981401,5.312756,4.581942,1.629811,-3.163471,6.562045,-6.853706,-9.042780,-1.791755,7.680089,-9.528914,-0.797177,-2.573232,7.030973,-3.076172,0.718135,-4.271631,9.002858,9.955189,-3.982711,6.635971,2.550103,8.763664,3.002492,8.555904,9.706993,9.362776,5.181424,-6.342883,8.996061,3.127739,-9.774683,5.018776,0.983627,9.952221,9.701771,2.827309,-4.644358,-9.603321,-3.102694,-6.059729,9.244999,-4.388277,1.484155,6.024267,0.628554,-4.607033,1.556797,6.686339,-4.717562,-9.768800,9.800972,-6.880760,0.636978,-1.025812,-1.933815,0.082615,-1.524646,5.888580,-7.363725,8.496616,1.103360,-3.561965,8.895651,4.809713,-3.973810,7.033802,-8.754828,5.877234,-9.931093,6.826255,9.972840,-2.419121,-6.899191,0.636341,6.368390,7.339182,-5.286892,1.905237,4.634625,2.969332,-1.037938,-3.323925,-5.646639,-2.237994,5.234282,-0.288190,-2.068203,-3.217955,-4.963663,-8.955479,-6.890820,-0.044094,5.976279,2.173952,-1.724245,-3.247405,0.882814,1.783742,-4.424069,5.837775,-3.472106,-4.622951,-5.500840,2.098990,3.519162,-3.174507,2.093416,-5.116210,-4.061814,-4.927126,4.333574,-7.312658,6.434240,-2.472048,8.370463,9.881926,4.298255,6.605400,-1.543294,-1.746730,-1.155972,-1.980539,5.495542,-4.470808,-8.873307,5.247193,6.552372,-0.481771,7.413239,-4.192373,-2.464074,-9.115555,-8.261098,-2.313936,5.844247,-7.774583,0.078321,6.675834,0.464964,9.922580,-5.992183,4.931646,-8.895528,-8.255006,7.873004,-7.475281,-4.423747,3.625789,-3.024160,-8.282141,9.978480,6.816618,6.150536,0.778651,-3.019588,-9.839237,4.363800,-4.232769,6.993104,-2.403248,-9.647743,6.383891,6.661821,2.948330,-1.486047,9.639341,-8.604223,-3.250007,-3.983711,-0.896405,-1.005831,2.319990,4.716522,7.933948,5.271389,6.813623,7.376708,-0.048853,6.435696,-2.906583,0.809912,7.638864,9.268062,2.629584,-0.647332,-3.116857,-0.193136,-3.915459,-4.532782,5.333361,2.357121,4.790365,2.501277,-1.487133,1.992111,-3.576775,-4.987415,-4.463700,1.652606,7.323170,7.971119,5.633042,-7.310784,1.710691,9.355935,-1.952085,0.279861,3.279419,0.664598,-5.703474,1.876593,-6.681980,-1.662678,0.660442,-7.764468,6.847887,-3.702318,-9.736222,6.579368,0.305769,-2.648316,5.066323,9.402053,-8.182120,-0.083167,2.858370,0.218961,-3.356764,-1.221995,2.298324,-4.141237,9.256575,0.577631,6.921138,3.564391,-5.949735,-7.074981,-6.954800,-5.275079,9.474600,3.393745,-5.246185,-3.545905,-6.068205,-6.003047,5.204378,7.203102,-3.692827,-0.448708,4.613329,-3.762119,3.137092,3.829880,7.725171,-3.998343,6.779006,3.092790,-0.714298,-2.137651,7.780275,1.962376,1.360922,9.691974,8.325757,-5.069493,1.910216,-3.978076,7.560222,-8.699962,-0.822189,5.536065,-6.947992,6.800476,-8.192359,-9.568977,2.069447,9.965335,-8.635393,-7.548372,-3.703681,-7.568686,3.525531,1.254150,-2.456840,7.094221,-0.043224,-6.891184,-9.736557,-5.817759,4.840412,-2.299493,-8.632456,-2.842942,-0.367667,-3.371915,3.899951,-8.961966,-7.861164,5.383687,-6.333910,7.388597,-0.958909,1.569029,7.838120,1.653079,-1.215953,5.105099,9.285282,6.140096,-8.739941,0.284749,-6.092300,7.882746,6.718502,-3.022818,1.156342,-7.824890,3.545086,-0.042780,4.596005,-6.649078,9.375129,-1.189427,0.811312,-4.923817,-3.837076,1.419784,-5.236948,1.611797], dtype = "float32")#candidate|2291|(780,)|const|float32
call_2289 = relay.TupleGetItem(func_1711_call(relay.reshape(const_2290.astype('float32'), [60,]), relay.reshape(const_2291.astype('float32'), [780,]), ), 6)
call_2292 = relay.TupleGetItem(func_1714_call(relay.reshape(const_2290.astype('float32'), [60,]), relay.reshape(const_2291.astype('float32'), [780,]), ), 6)
uop_2293 = relay.tan(uop_2285.astype('float64')) # shape=(8, 11, 2)
uop_2295 = relay.tan(uop_2287.astype('float64')) # shape=(8, 11, 2)
bop_2301 = relay.right_shift(uop_2293.astype('int32'), relay.reshape(bop_2281.astype('int32'), relay.shape_of(uop_2293))) # shape=(8, 11, 2)
bop_2304 = relay.right_shift(uop_2295.astype('int32'), relay.reshape(bop_2284.astype('int32'), relay.shape_of(uop_2295))) # shape=(8, 11, 2)
output = relay.Tuple([call_2269,call_2289,const_2290,const_2291,bop_2301,])
output2 = relay.Tuple([call_2270,call_2292,const_2290,const_2291,bop_2304,])
func_2311 = relay.Function([], output)
mod['func_2311'] = func_2311
mod = relay.transform.InferType()(mod)
mutated_mod['func_2311'] = func_2311
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2311_call = mutated_mod.get_global_var('func_2311')
call_2312 = func_2311_call()
output = call_2312
func_2313 = relay.Function([], output)
mutated_mod['func_2313'] = func_2313
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1797_call = mod.get_global_var('func_1797')
func_1799_call = mutated_mod.get_global_var('func_1799')
call_2317 = relay.TupleGetItem(func_1797_call(), 0)
call_2318 = relay.TupleGetItem(func_1799_call(), 0)
func_241_call = mod.get_global_var('func_241')
func_244_call = mutated_mod.get_global_var('func_244')
const_2333 = relay.const([-4,2,10,-2,-7,-6,4,2,-2,9,-3,-4,8,-1,-4,-6,9,-10,5,-5,-6,3,-4,10,-5,3,9,-2,7,-8,10,-9,-4,-7,10,1,4,10,8,2,-8,5,4,-6,1,-4,-10,9,4,4,7,-4,9,6,3,-3,7,-2,-3,4,4,4,-4,9,10,-6,7,7,3,5,5,8,-6,-5,-9,-5,-8,7,4,4,7,1,1,2,-3,-9,-7,-10,-5,1,3,-7,5,-7,-7,-4,-10,-10,5,4,-3,4,-9,-2,-9,-10,-9,3,10,6,10,4,5,10,10,-7,6,2,8,-10,6,-9,-7,-10,4,-4,6,-4,-4,2,-8,-6,2,8,1,-7,9,2,10,-3,-8,9,-5,-4,7,2,4,-10,5,8,-9,10,1,-4,-6,5,3,5,-7,6,6,1,-7,6,-10,-7,9,7,-1,-3,5,8,-2,-6,3,-2,2,-9,-6,-3,1,-7,7,-2,-4,-7,3,-9,-10,3,-8,4,-10,-1,10,-3,3,-3,4,1,2,-9,3,9,2,9,-9,-2,-4,5,8,-6,10,1,5,-3,8,-2,-9,-4,-5,3,-4,2,9,-5,-9,-5,-10,-4,6,2,1,-7,-6,-3,4,4,-2,-9,8,-5,-2,-6,7,-7,-1,-9,4,-5,6,1,6,9,4,7,1,8,4,-6,-8,-6,-4,9,2,10,4,10,-9,-7,7,8,8,1,-2,3,-2,6,5,-7,3,-4,1,8,3,6,1,-9,-9,-1,1,-10,-4,-10,6,-5,-7,6,3,9,-9,-7,8,-9,8,10,5,-2,-10,9,2,-5,-3,6,-4], dtype = "int32")#candidate|2333|(315,)|const|int32
call_2332 = relay.TupleGetItem(func_241_call(relay.reshape(const_2333.astype('int32'), [5, 7, 9]), relay.reshape(const_2333.astype('int32'), [5, 7, 9]), ), 1)
call_2334 = relay.TupleGetItem(func_244_call(relay.reshape(const_2333.astype('int32'), [5, 7, 9]), relay.reshape(const_2333.astype('int32'), [5, 7, 9]), ), 1)
func_842_call = mod.get_global_var('func_842')
func_846_call = mutated_mod.get_global_var('func_846')
var_2337 = relay.var("var_2337", dtype = "float32", shape = (231,))#candidate|2337|(231,)|var|float32
const_2338 = relay.const([-5.996159,3.283505,-0.551626,1.690958,9.085821,3.034924,-1.051673,-2.962523,8.590172,0.152180,-2.308510,-4.744544,-5.331040,-0.443584,4.842219,7.969576,-5.877559,8.386632,2.508743,-9.377749,-8.241024,-2.556570,-4.804261,-9.580062,9.045192,-7.883198,-9.347753,-3.740179,-5.270001,-8.871285,-6.406499,-3.424726,-1.767998,-2.544253,-7.824606,8.094888,-5.294853,-2.043377,-2.914240,-4.760649,-0.842561,5.715050,2.288639,-4.951516,-3.754303,8.254840,-9.007443,4.680347,-4.948612,8.617244,-4.542969,2.778903,-8.612492,4.782657,-7.527900,-4.527924,7.553383,3.942452,-3.883807,-6.518403,7.384161,3.037374,4.766199,-9.172437,-6.035698,7.704446,-5.906011,3.114352,-7.525373,5.794415,-6.679804,1.502224,5.039428,-3.122529,-4.048863,-1.716788,2.760622,-8.210158,-8.719589,-2.102819,-7.497084,3.952207,-0.580563,9.543038,-6.604752,-1.316274,6.722415,-2.068260,-9.524456,-1.338429,4.032786,-5.017663,-0.478043,-1.294271,-8.706890,-9.582410,-6.227628,4.736027,-6.709002,5.448438,-8.560872,-0.340147,5.363765,-2.059605,3.095798,0.858285,-9.927549,6.826572,-4.826533,6.167205,-0.837894,0.090249,-7.203149,-2.658055,0.983760,6.871421,0.168197,1.048986,-8.496314,-6.170220,-9.125841,-6.448317,-3.359423,-6.895929,4.437194,1.936141,0.834981,-5.218489,5.797693,-0.516849,-1.756965,-7.635525,5.156299,6.208575,-5.655568,-3.258282,6.614071,2.849531,-1.023243,-8.030540], dtype = "float64")#candidate|2338|(140,)|const|float64
call_2336 = relay.TupleGetItem(func_842_call(relay.reshape(var_2337.astype('float32'), [7, 11, 3]), relay.reshape(const_2338.astype('float64'), [70, 2]), ), 1)
call_2339 = relay.TupleGetItem(func_846_call(relay.reshape(var_2337.astype('float32'), [7, 11, 3]), relay.reshape(const_2338.astype('float64'), [70, 2]), ), 1)
output = relay.Tuple([call_2317,call_2332,const_2333,call_2336,var_2337,const_2338,])
output2 = relay.Tuple([call_2318,call_2334,const_2333,call_2339,var_2337,const_2338,])
func_2340 = relay.Function([var_2337,], output)
mod['func_2340'] = func_2340
mod = relay.transform.InferType()(mod)
var_2341 = relay.var("var_2341", dtype = "float32", shape = (231,))#candidate|2341|(231,)|var|float32
output = func_2340(var_2341)
func_2342 = relay.Function([var_2341], output)
mutated_mod['func_2342'] = func_2342
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_2355 = relay.TupleGetItem(func_1645_call(), 0)
call_2356 = relay.TupleGetItem(func_1647_call(), 0)
func_1547_call = mod.get_global_var('func_1547')
func_1551_call = mutated_mod.get_global_var('func_1551')
const_2378 = relay.const([-5.779751,-4.555792,6.064527,-2.202587,5.449267,-3.998990,8.599372,4.579969,-2.730678,-0.438765,-9.097043,-6.598693,-3.627729,7.513581,7.265564,-1.616011,0.083521,9.456825,7.171753,1.930733,-7.197432,2.846623,-9.841622,-4.977538,4.456045,2.093643,-4.153017,-9.282900,-2.505766,-7.821283,7.979500,5.008831,-9.199984,1.455257,-8.031791,-9.518884,2.513332,-6.596376,9.390836,-0.387110,8.064884,-4.798112,6.782517,-8.992021,-8.183573,-9.873002,-9.366918,-2.188297,-2.080714,-2.054364,-4.969743,6.123705,1.674874,-7.254337,6.781386,3.980385,-1.622103,-5.349017,-0.518473,3.082122,0.511982,-4.964269,-7.009576,-0.479127,6.072388,1.816127,-9.880738,3.361143,5.634534,-6.681306,-6.193219,9.707155,1.935422,-3.429946,7.406246,-5.714068,8.306667,-5.766164,-1.304576,-6.754117,7.243334,7.219258,3.334007,7.179208,-0.704722,7.917468,2.659526,-0.312469,-8.171028,4.747221,-6.738885,-2.619787,-6.545363,4.425862,-9.083606,-5.774440,-9.856725,2.376401,-2.295018,-6.472720,-6.753702,9.990228,-7.501562,-5.632727,0.387076,-3.396689,0.824094,3.018588,2.676763,-1.379972,0.139446,-3.978857,9.161385,1.569231,-8.328377,-6.258996,6.556088,4.445019,0.395787,-9.890490,8.778548,0.032180,6.528895,-3.802590,2.612473,-9.677577,-5.365389,-7.195727,-0.493941,3.828208,7.313111,2.460591,-4.933644,2.487409,-7.498902,-0.877449,-4.039329,9.651666,-3.063341,3.756477,-9.985795,-9.242691,-1.832696,-0.694771,-2.908818,0.401056,-2.584282,-0.957379,5.067866,-4.913665,5.312846,2.712758,-2.960162,-4.465554,4.279547,2.116468,0.240317,-0.746555,-6.841130,7.505681,-2.114683,7.586582,9.875761,7.996877,0.908645,6.341456,-1.446750,-3.427769,0.908731,-5.679990,-4.526362,-5.781165,9.697901,7.495055,-9.803211,2.760997,-1.290354,9.290784,-5.839225,-5.707498,-5.328175,5.022248,8.471581,-0.790872,-1.008170,0.538565,-7.905186,-9.663883,-9.819658,0.336830,-7.162063,0.420772,9.148582,5.175439,-8.122329,7.251309,5.460483,8.750170,2.184759,9.080179,-2.988150,-9.825844,-7.247477,1.271488,1.406037,3.216880,9.801179,-7.123771,3.391908,-0.673201,1.946992,5.405045,-3.851270,4.168380,-2.686458,5.751553,4.603558,5.840011,1.325924,-6.688351,9.092021,-8.257560,7.393991,-0.878624,-5.280566,-5.573396,-4.258205,-0.619866,-2.880349,-2.951132,-1.431903,-2.930087,-5.618209,-6.447331,-3.511709,-3.585573,-9.694387,-9.838280,7.650651,4.269614,1.639074,9.603301,6.173564,6.667897,-8.203291,-7.943682,5.492284,-7.672901,-9.354647,7.519692,-2.217858,-0.453344,0.996642,-9.689451,1.408948,-5.795543,7.737492,3.630774,-3.245158,-9.392897,-4.634126,6.662747,3.070569,5.103900,-5.458494,-6.433522,-2.787409,-8.003757,9.285069,-4.929027,-2.601275,-0.143761,5.323051,-3.155605,-0.492803,-4.877621,5.848579,-9.576709,2.182905,-1.136119,8.283594,-7.848539,4.527797,0.888565,6.292081,-5.138020,-7.862725,6.662477,-4.789051,9.142633,1.724539,0.523670,-3.413101,9.010862,-0.913330,3.096342,4.984978,8.590442,-9.271289,8.306884,3.892992,0.002895,-8.866497,6.488818,-0.054512,7.399006,-2.138997,-5.107613,6.564612,-1.938037,1.775013,4.361639,-9.682794,-8.451414,5.747163,-7.148597,0.057357,-8.249810,1.179901,-3.536286,8.846894,0.968152,-0.642376,-4.903740,7.570121,1.859774,-2.809315,8.890728,-9.664038,-8.662978,-0.463930,-5.684750,-2.867688,6.998023,-1.180489,-0.033949,-0.350067,8.203603,3.735977,-5.477375,-2.343399,-0.425578,-8.152619,6.058150,5.197227,-5.684423,3.924119,6.516112,3.076841,0.158984,-9.909816,0.959008,0.456720,2.886170,-7.612178,3.145330,-3.458988,-7.083943,-0.500130,9.549617,6.900985,7.327924,-7.307279,-2.546685,1.467562,5.277365,4.991902,-2.299307,9.207762,-5.818080,7.323118,-8.453205,5.293683,-0.119610,6.217174,1.206927,3.345682,5.711322,3.884572,-5.172631,0.607196,-8.385057,-4.452941,9.662118,-3.953281,-7.205064,4.333076,4.802155,0.898514,-3.721645,6.013462,0.646001,-7.522817,-6.885610,-7.632022,4.191230,-9.433629,5.672160,-1.159982,-1.366921,-0.321908,2.555045,-9.545272,-4.550211,-4.218076,-9.925038,-8.103369,3.177720,-5.678751,-9.580166,-1.083846,-7.237329,3.386593,9.423464,-9.139922,-9.817697,-2.504820,-1.496395,5.514221,-3.439013,-5.832787,9.411441,-8.986987,2.760863,8.627917,-6.079686,1.782016,-2.859753,-1.539785,7.898817,-0.939435,9.484287,0.987956,6.936239,-7.862612,-8.020794,-8.522498,3.870074,-5.036975,5.838516,-1.671233,5.987263,-5.872778,-8.305939,-9.623117,0.806387,7.233639,2.033384,-0.441785,6.965083,9.919680,3.860586,-6.000625,-4.507954,-6.426423,-7.237116,8.228118,-0.713275,9.605105,-5.876566,-1.620051,-3.844719,-7.394661,-8.194742,-5.863200,3.051980,-2.180475,3.230397,4.916405,2.583102,8.380131,-8.581995,-1.521759,-9.169778,7.347453,1.942531,-6.850500,-6.370327,7.480355,-6.321040,-3.250690,0.048280,-9.536283,4.367358,6.632706,0.715514,-9.027142,5.090716,7.310363,8.248191,5.826911,-9.323510,-4.059667,-7.053462,-7.884084,-0.361235,3.803163,0.453086,-1.500112,7.076385,8.832376,9.982932,9.646313,-5.260658], dtype = "float32")#candidate|2378|(504,)|const|float32
var_2379 = relay.var("var_2379", dtype = "int16", shape = (288,))#candidate|2379|(288,)|var|int16
call_2377 = relay.TupleGetItem(func_1547_call(relay.reshape(const_2378.astype('float32'), [14, 9, 4]), relay.reshape(var_2379.astype('int16'), [6, 6, 8]), relay.reshape(const_2378.astype('float32'), [14, 9, 4]), ), 4)
call_2380 = relay.TupleGetItem(func_1551_call(relay.reshape(const_2378.astype('float32'), [14, 9, 4]), relay.reshape(var_2379.astype('int16'), [6, 6, 8]), relay.reshape(const_2378.astype('float32'), [14, 9, 4]), ), 4)
output = relay.Tuple([call_2355,call_2377,const_2378,var_2379,])
output2 = relay.Tuple([call_2356,call_2380,const_2378,var_2379,])
func_2382 = relay.Function([var_2379,], output)
mod['func_2382'] = func_2382
mod = relay.transform.InferType()(mod)
var_2383 = relay.var("var_2383", dtype = "int16", shape = (288,))#candidate|2383|(288,)|var|int16
output = func_2382(var_2383)
func_2384 = relay.Function([var_2383], output)
mutated_mod['func_2384'] = func_2384
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_2386 = relay.TupleGetItem(func_1645_call(), 0)
call_2387 = relay.TupleGetItem(func_1647_call(), 0)
func_1547_call = mod.get_global_var('func_1547')
func_1551_call = mutated_mod.get_global_var('func_1551')
var_2396 = relay.var("var_2396", dtype = "float32", shape = (504,))#candidate|2396|(504,)|var|float32
const_2397 = relay.const([-10,4,10,4,4,10,-6,8,-6,-1,7,-9,3,7,2,-3,-4,-10,5,-4,3,-10,-6,3,5,5,3,-6,-2,2,-10,-6,3,9,-1,9,-9,-3,7,7,9,-7,-5,-9,-7,1,-7,4,-3,8,-5,10,-7,-7,6,-8,5,9,10,9,10,-5,-8,-7,-1,9,-3,8,-9,-4,-10,-1,-10,8,-10,-5,10,-7,10,-3,8,-7,-8,-1,-1,-3,-9,-9,-7,-7,6,7,-4,4,8,-5,9,9,6,-4,7,1,4,-3,6,7,-1,-8,9,-9,1,-3,-5,-8,4,-7,3,-7,7,-7,-9,7,-1,7,8,6,1,10,4,6,-3,-9,2,-3,-5,1,7,5,2,-7,-7,2,2,3,-1,4,-3,-2,10,-3,-8,-8,-4,-4,-5,9,-3,2,-4,-9,-8,-10,-7,-8,-10,1,-9,2,4,-2,-6,-4,3,-2,3,-3,-8,3,-5,-2,4,-2,4,6,9,-8,-2,-4,8,-3,1,3,-9,5,-9,-1,-10,-6,7,4,-4,3,-5,-1,4,6,-8,-3,9,-10,-4,-1,-4,-3,-1,-8,-1,8,-7,1,-3,-7,9,-9,-9,-7,2,1,5,6,1,10,3,10,-9,3,1,6,-1,8,-5,-5,3,2,10,-3,-10,-4,2,-6,8,-4,1,-9,-2,4,-2,-4,-5,-2,-2,7,-9,-8,-7,-10,6,-4,-2,9,1,-3,4,-7,-4,5,5,2,5,3,-6,8,6,9,6,4,-3,10], dtype = "int16")#candidate|2397|(288,)|const|int16
call_2395 = relay.TupleGetItem(func_1547_call(relay.reshape(var_2396.astype('float32'), [14, 9, 4]), relay.reshape(const_2397.astype('int16'), [6, 6, 8]), relay.reshape(var_2396.astype('float32'), [14, 9, 4]), ), 3)
call_2398 = relay.TupleGetItem(func_1551_call(relay.reshape(var_2396.astype('float32'), [14, 9, 4]), relay.reshape(const_2397.astype('int16'), [6, 6, 8]), relay.reshape(var_2396.astype('float32'), [14, 9, 4]), ), 3)
output = relay.Tuple([call_2386,call_2395,var_2396,const_2397,])
output2 = relay.Tuple([call_2387,call_2398,var_2396,const_2397,])
func_2406 = relay.Function([var_2396,], output)
mod['func_2406'] = func_2406
mod = relay.transform.InferType()(mod)
var_2407 = relay.var("var_2407", dtype = "float32", shape = (504,))#candidate|2407|(504,)|var|float32
output = func_2406(var_2407)
func_2408 = relay.Function([var_2407], output)
mutated_mod['func_2408'] = func_2408
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2416 = relay.var("var_2416", dtype = "float64", shape = (3, 4, 3))#candidate|2416|(3, 4, 3)|var|float64
uop_2417 = relay.asinh(var_2416.astype('float64')) # shape=(3, 4, 3)
output = relay.Tuple([uop_2417,])
output2 = relay.Tuple([uop_2417,])
func_2422 = relay.Function([var_2416,], output)
mod['func_2422'] = func_2422
mod = relay.transform.InferType()(mod)
mutated_mod['func_2422'] = func_2422
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2423 = relay.var("var_2423", dtype = "float64", shape = (3, 4, 3))#candidate|2423|(3, 4, 3)|var|float64
func_2422_call = mutated_mod.get_global_var('func_2422')
call_2424 = func_2422_call(var_2423)
output = call_2424
func_2425 = relay.Function([var_2423], output)
mutated_mod['func_2425'] = func_2425
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1978_call = mod.get_global_var('func_1978')
func_1979_call = mutated_mod.get_global_var('func_1979')
call_2444 = relay.TupleGetItem(func_1978_call(), 0)
call_2445 = relay.TupleGetItem(func_1979_call(), 0)
func_1547_call = mod.get_global_var('func_1547')
func_1551_call = mutated_mod.get_global_var('func_1551')
const_2450 = relay.const([7.314721,5.549498,-0.063370,-4.074758,-3.999741,-4.488761,4.911811,1.440442,9.626143,-8.147715,8.867699,9.127290,3.810811,-7.664981,6.482718,-5.884927,1.885607,-1.277386,8.128530,7.942602,6.200753,-5.331299,-1.563827,1.513985,4.748513,6.993719,6.199538,-0.684748,0.215516,5.693674,-6.281049,7.240766,0.936031,-2.407225,9.251799,9.514813,8.631072,6.585837,7.947912,5.887222,-5.975991,6.475513,4.063082,-7.887479,-5.870361,-7.773235,-6.359760,2.712531,0.112815,-3.650595,1.456602,4.717946,3.614949,1.344376,-5.840510,3.223717,4.400602,-9.476639,2.908159,1.943874,-9.923739,-8.759632,2.224134,-7.976737,3.670724,8.375792,-8.892942,-9.867745,6.644662,-9.828354,-3.064797,8.917492,1.755592,4.327698,-4.782958,-7.212558,-1.185890,0.804326,-4.115803,-4.267389,9.573031,6.509078,-1.348646,8.049059,0.947426,1.447393,-5.396013,4.540013,-5.598651,-4.984699,2.362081,4.452444,8.504932,6.399043,6.277556,-8.944316,-1.907486,-3.871227,2.938901,-6.483224,-9.608662,-1.992103,-0.355000,-8.426549,-5.461545,-6.768589,4.911171,6.039618,-1.205321,4.058131,-8.385953,-5.595223,1.224229,-4.183613,2.155870,4.445821,-0.008543,-5.228309,0.513119,-2.258037,2.833779,7.852058,-9.277238,6.241848,3.996466,-9.088144,-3.047105,-0.948979,-0.067827,-8.873301,-1.475954,9.960778,-7.908187,5.355731,-2.637625,6.144373,-9.461412,-8.840350,-3.222742,-7.289508,-8.670588,6.901178,3.456328,-8.505149,-1.112587,9.088885,-1.047784,-1.939525,-0.056831,-4.775958,2.768410,-9.310466,-6.681216,-9.212729,8.800332,8.566644,-2.322495,-6.405292,-5.458010,7.951121,-2.748491,5.442143,0.945044,9.295720,-0.327981,-4.485344,6.725763,9.065874,2.843279,-3.781939,-9.424585,5.686167,3.478690,0.760751,-8.015527,-8.151646,2.232360,-4.497026,8.318682,4.800260,2.395742,6.144583,-3.933897,3.831316,0.428258,-5.696577,3.492508,-7.836795,-5.701355,-3.933806,-1.714182,5.120927,5.568600,-0.933909,8.473424,-4.875770,3.287404,4.889274,-0.531837,1.137425,0.562799,-5.808812,8.157938,8.892476,-1.510424,0.537957,-6.498466,-4.503657,-6.479320,9.283444,-8.541170,-8.551263,3.024531,-4.471809,-6.761448,2.093282,5.117729,-0.711183,-7.497979,-2.824387,-3.047972,1.393682,-6.590877,-6.364341,-8.744727,-8.619068,5.308873,-1.986878,8.886992,-7.039141,9.901367,1.068679,6.111817,6.309950,6.844863,-8.384967,6.560145,-6.223977,-0.473914,-9.669964,-1.428847,2.327987,-0.837851,1.676337,-0.180884,0.513839,8.429113,9.648356,4.908243,3.830964,-0.592365,-1.857135,2.927154,-2.520932,5.406546,1.189079,6.908048,2.812207,-6.022400,0.804937,-0.799480,-6.366363,0.119950,8.440818,9.316884,5.649118,-6.768600,9.163602,7.607423,2.474940,0.278809,-9.743634,-2.137234,-5.343327,-2.437299,3.793481,-5.669148,1.377940,2.043316,-8.286988,1.970188,7.550194,2.601713,2.130878,-1.198776,2.393738,2.622789,-6.039806,3.981371,6.001468,6.519090,1.285131,-7.858576,7.619287,5.433484,-0.523514,-9.615582,-8.585799,6.258717,-9.881610,6.099930,6.174409,3.335522,-5.839504,-7.598978,1.625976,-2.779607,6.997427,1.430082,2.621440,1.541677,0.861255,0.363747,8.688780,7.100895,-2.992217,-0.425060,-6.761825,0.191259,-0.393659,-5.576067,2.815715,-8.414838,6.542531,-9.960672,-1.877006,1.415252,1.804766,-4.206402,9.162232,-6.158334,-5.256508,4.954430,-1.973437,9.399552,-2.111223,-4.954408,-3.342096,-0.873189,-9.498929,-6.219388,-2.262957,-4.773779,-4.912906,6.049023,7.252621,5.109854,-4.763269,-9.993932,-1.190588,2.058294,1.606010,-9.078856,0.370602,1.126844,3.869412,-5.884421,5.966116,7.418742,4.427825,6.996928,-7.936539,-7.708429,4.341047,-7.833133,-6.640486,-9.250432,2.686834,-5.117866,-1.900368,8.326000,7.620030,0.046069,-6.009022,2.263734,9.344919,-9.476768,2.968060,-9.167638,1.169739,-9.904094,5.949116,3.807714,-2.736877,4.683326,-2.675107,-7.065368,-8.653597,-0.385242,6.061477,5.711348,8.006333,-9.904217,-5.763367,2.750647,1.842551,-4.145951,7.880123,-5.740980,8.759119,-3.864348,-0.620202,4.886497,-4.917597,1.135315,9.190494,8.976182,6.802711,-0.684319,-3.657504,9.676597,9.597459,-9.880466,-9.413066,-1.456881,-4.700382,9.180759,-9.797572,-6.114199,-6.399866,-3.190754,-0.807550,-8.162691,8.199379,7.672519,5.639504,-4.910166,-5.671356,7.293336,8.630422,-1.502795,-3.336402,-0.941712,4.871869,1.608510,-1.791943,6.044330,8.344998,-6.044879,-4.893148,8.296862,2.044794,6.048184,-2.594688,-9.634947,-0.485603,-6.191225,-1.427867,2.466817,2.204269,8.924161,-2.197262,-6.012308,3.519776,4.738600,-2.957071,5.490971,-0.510990,-0.190766,2.041856,-9.036732,0.469877,-2.641740,8.533023,1.606702,-3.924215,-8.604867,-3.002491,2.322734,-0.759187,2.162535,9.910213,-0.474331,3.370258,-8.551610,9.824878,8.316980,5.707956,-4.940000,4.021588,8.008694,-0.128841,8.796809,-4.719774,-0.348252,7.196860,-7.856975,-9.914003,-5.326739,8.366248,-7.004578,9.642224,5.518849,5.569602,9.570138,-9.857438,0.210496,-7.071156,7.531544,7.242321,8.290526,5.277776,5.575577,-4.262408], dtype = "float32")#candidate|2450|(504,)|const|float32
var_2451 = relay.var("var_2451", dtype = "int16", shape = (72, 4))#candidate|2451|(72, 4)|var|int16
call_2449 = relay.TupleGetItem(func_1547_call(relay.reshape(const_2450.astype('float32'), [14, 9, 4]), relay.reshape(var_2451.astype('int16'), [6, 6, 8]), relay.reshape(const_2450.astype('float32'), [14, 9, 4]), ), 0)
call_2452 = relay.TupleGetItem(func_1551_call(relay.reshape(const_2450.astype('float32'), [14, 9, 4]), relay.reshape(var_2451.astype('int16'), [6, 6, 8]), relay.reshape(const_2450.astype('float32'), [14, 9, 4]), ), 0)
output = relay.Tuple([call_2444,call_2449,const_2450,var_2451,])
output2 = relay.Tuple([call_2445,call_2452,const_2450,var_2451,])
func_2454 = relay.Function([var_2451,], output)
mod['func_2454'] = func_2454
mod = relay.transform.InferType()(mod)
var_2455 = relay.var("var_2455", dtype = "int16", shape = (72, 4))#candidate|2455|(72, 4)|var|int16
output = func_2454(var_2455)
func_2456 = relay.Function([var_2455], output)
mutated_mod['func_2456'] = func_2456
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2311_call = mod.get_global_var('func_2311')
func_2313_call = mutated_mod.get_global_var('func_2313')
call_2510 = relay.TupleGetItem(func_2311_call(), 0)
call_2511 = relay.TupleGetItem(func_2313_call(), 0)
func_2311_call = mod.get_global_var('func_2311')
func_2313_call = mutated_mod.get_global_var('func_2313')
call_2516 = relay.TupleGetItem(func_2311_call(), 4)
call_2517 = relay.TupleGetItem(func_2313_call(), 4)
func_1978_call = mod.get_global_var('func_1978')
func_1979_call = mutated_mod.get_global_var('func_1979')
call_2533 = relay.TupleGetItem(func_1978_call(), 0)
call_2534 = relay.TupleGetItem(func_1979_call(), 0)
func_1944_call = mod.get_global_var('func_1944')
func_1945_call = mutated_mod.get_global_var('func_1945')
call_2539 = relay.TupleGetItem(func_1944_call(), 0)
call_2540 = relay.TupleGetItem(func_1945_call(), 0)
uop_2543 = relay.acosh(call_2539.astype('float64')) # shape=(2, 1)
uop_2545 = relay.acosh(call_2540.astype('float64')) # shape=(2, 1)
bop_2546 = relay.multiply(call_2533.astype('uint32'), relay.reshape(call_2539.astype('uint32'), relay.shape_of(call_2533))) # shape=(2, 1)
bop_2549 = relay.multiply(call_2534.astype('uint32'), relay.reshape(call_2540.astype('uint32'), relay.shape_of(call_2534))) # shape=(2, 1)
func_2382_call = mod.get_global_var('func_2382')
func_2384_call = mutated_mod.get_global_var('func_2384')
var_2559 = relay.var("var_2559", dtype = "int16", shape = (288, 1))#candidate|2559|(288, 1)|var|int16
call_2558 = relay.TupleGetItem(func_2382_call(relay.reshape(var_2559.astype('int16'), [288,])), 0)
call_2560 = relay.TupleGetItem(func_2384_call(relay.reshape(var_2559.astype('int16'), [288,])), 0)
output = relay.Tuple([call_2510,call_2516,uop_2543,bop_2546,call_2558,var_2559,])
output2 = relay.Tuple([call_2511,call_2517,uop_2545,bop_2549,call_2560,var_2559,])
func_2566 = relay.Function([var_2559,], output)
mod['func_2566'] = func_2566
mod = relay.transform.InferType()(mod)
mutated_mod['func_2566'] = func_2566
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2567 = relay.var("var_2567", dtype = "int16", shape = (288, 1))#candidate|2567|(288, 1)|var|int16
func_2566_call = mutated_mod.get_global_var('func_2566')
call_2568 = func_2566_call(var_2567)
output = call_2568
func_2569 = relay.Function([var_2567], output)
mutated_mod['func_2569'] = func_2569
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_2571 = func_1953_call()
call_2572 = func_1953_call()
output = relay.Tuple([call_2571,])
output2 = relay.Tuple([call_2572,])
func_2582 = relay.Function([], output)
mod['func_2582'] = func_2582
mod = relay.transform.InferType()(mod)
output = func_2582()
func_2583 = relay.Function([], output)
mutated_mod['func_2583'] = func_2583
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2606 = relay.var("var_2606", dtype = "float64", shape = (1, 13, 13))#candidate|2606|(1, 13, 13)|var|float64
uop_2607 = relay.rsqrt(var_2606.astype('float64')) # shape=(1, 13, 13)
func_274_call = mod.get_global_var('func_274')
func_279_call = mutated_mod.get_global_var('func_279')
var_2611 = relay.var("var_2611", dtype = "float64", shape = (2, 180))#candidate|2611|(2, 180)|var|float64
var_2612 = relay.var("var_2612", dtype = "float32", shape = (378, 1))#candidate|2612|(378, 1)|var|float32
call_2610 = relay.TupleGetItem(func_274_call(relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2612.astype('float32'), [378,]), ), 0)
call_2613 = relay.TupleGetItem(func_279_call(relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2612.astype('float32'), [378,]), ), 0)
func_274_call = mod.get_global_var('func_274')
func_279_call = mutated_mod.get_global_var('func_279')
call_2632 = relay.TupleGetItem(func_274_call(relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(call_2610.astype('float32'), [378,]), ), 1)
call_2633 = relay.TupleGetItem(func_279_call(relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(var_2611.astype('float64'), [8, 3, 15]), relay.reshape(call_2610.astype('float32'), [378,]), ), 1)
output = relay.Tuple([uop_2607,call_2610,var_2611,var_2612,call_2632,])
output2 = relay.Tuple([uop_2607,call_2613,var_2611,var_2612,call_2633,])
func_2642 = relay.Function([var_2606,var_2611,var_2612,], output)
mod['func_2642'] = func_2642
mod = relay.transform.InferType()(mod)
mutated_mod['func_2642'] = func_2642
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2642_call = mutated_mod.get_global_var('func_2642')
var_2644 = relay.var("var_2644", dtype = "float64", shape = (1, 13, 13))#candidate|2644|(1, 13, 13)|var|float64
var_2645 = relay.var("var_2645", dtype = "float64", shape = (2, 180))#candidate|2645|(2, 180)|var|float64
var_2646 = relay.var("var_2646", dtype = "float32", shape = (378, 1))#candidate|2646|(378, 1)|var|float32
call_2643 = func_2642_call(var_2644,var_2645,var_2646,)
output = call_2643
func_2647 = relay.Function([var_2644,var_2645,var_2646,], output)
mutated_mod['func_2647'] = func_2647
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1944_call = mod.get_global_var('func_1944')
func_1945_call = mutated_mod.get_global_var('func_1945')
call_2697 = relay.TupleGetItem(func_1944_call(), 3)
call_2698 = relay.TupleGetItem(func_1945_call(), 3)
func_1815_call = mod.get_global_var('func_1815')
func_1816_call = mutated_mod.get_global_var('func_1816')
call_2710 = func_1815_call()
call_2711 = func_1815_call()
uop_2716 = relay.log10(call_2697.astype('float32')) # shape=(2, 1)
uop_2718 = relay.log10(call_2698.astype('float32')) # shape=(2, 1)
output = relay.Tuple([call_2710,uop_2716,])
output2 = relay.Tuple([call_2711,uop_2718,])
func_2721 = relay.Function([], output)
mod['func_2721'] = func_2721
mod = relay.transform.InferType()(mod)
output = func_2721()
func_2722 = relay.Function([], output)
mutated_mod['func_2722'] = func_2722
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_2740 = func_1953_call()
call_2741 = func_1953_call()
output = relay.Tuple([call_2740,])
output2 = relay.Tuple([call_2741,])
func_2748 = relay.Function([], output)
mod['func_2748'] = func_2748
mod = relay.transform.InferType()(mod)
mutated_mod['func_2748'] = func_2748
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2748_call = mutated_mod.get_global_var('func_2748')
call_2749 = func_2748_call()
output = call_2749
func_2750 = relay.Function([], output)
mutated_mod['func_2750'] = func_2750
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_2757 = func_1576_call()
call_2758 = func_1576_call()
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_2782 = func_1953_call()
call_2783 = func_1953_call()
func_567_call = mod.get_global_var('func_567')
func_570_call = mutated_mod.get_global_var('func_570')
var_2798 = relay.var("var_2798", dtype = "uint32", shape = (22, 5))#candidate|2798|(22, 5)|var|uint32
call_2797 = relay.TupleGetItem(func_567_call(relay.reshape(var_2798.astype('uint32'), [2, 5, 11]), relay.reshape(var_2798.astype('uint32'), [2, 5, 11]), ), 1)
call_2799 = relay.TupleGetItem(func_570_call(relay.reshape(var_2798.astype('uint32'), [2, 5, 11]), relay.reshape(var_2798.astype('uint32'), [2, 5, 11]), ), 1)
bop_2803 = relay.bitwise_or(call_2757.astype('int8'), relay.reshape(call_2782.astype('int8'), relay.shape_of(call_2757))) # shape=(2, 1)
bop_2806 = relay.bitwise_or(call_2758.astype('int8'), relay.reshape(call_2783.astype('int8'), relay.shape_of(call_2758))) # shape=(2, 1)
bop_2807 = relay.add(bop_2803.astype('uint64'), relay.reshape(call_2782.astype('uint64'), relay.shape_of(bop_2803))) # shape=(2, 1)
bop_2810 = relay.add(bop_2806.astype('uint64'), relay.reshape(call_2783.astype('uint64'), relay.shape_of(bop_2806))) # shape=(2, 1)
output = relay.Tuple([call_2797,var_2798,bop_2807,])
output2 = relay.Tuple([call_2799,var_2798,bop_2810,])
func_2815 = relay.Function([var_2798,], output)
mod['func_2815'] = func_2815
mod = relay.transform.InferType()(mod)
var_2816 = relay.var("var_2816", dtype = "uint32", shape = (22, 5))#candidate|2816|(22, 5)|var|uint32
output = func_2815(var_2816)
func_2817 = relay.Function([var_2816], output)
mutated_mod['func_2817'] = func_2817
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2839 = relay.var("var_2839", dtype = "uint32", shape = (4, 4, 8))#candidate|2839|(4, 4, 8)|var|uint32
var_2840 = relay.var("var_2840", dtype = "uint32", shape = (4, 4, 8))#candidate|2840|(4, 4, 8)|var|uint32
bop_2841 = relay.subtract(var_2839.astype('uint32'), relay.reshape(var_2840.astype('uint32'), relay.shape_of(var_2839))) # shape=(4, 4, 8)
output = bop_2841
output2 = bop_2841
func_2852 = relay.Function([var_2839,var_2840,], output)
mod['func_2852'] = func_2852
mod = relay.transform.InferType()(mod)
var_2853 = relay.var("var_2853", dtype = "uint32", shape = (4, 4, 8))#candidate|2853|(4, 4, 8)|var|uint32
var_2854 = relay.var("var_2854", dtype = "uint32", shape = (4, 4, 8))#candidate|2854|(4, 4, 8)|var|uint32
output = func_2852(var_2853,var_2854,)
func_2855 = relay.Function([var_2853,var_2854,], output)
mutated_mod['func_2855'] = func_2855
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1763_call = mod.get_global_var('func_1763')
func_1764_call = mutated_mod.get_global_var('func_1764')
call_2876 = func_1763_call()
call_2877 = func_1763_call()
output = relay.Tuple([call_2876,])
output2 = relay.Tuple([call_2877,])
func_2884 = relay.Function([], output)
mod['func_2884'] = func_2884
mod = relay.transform.InferType()(mod)
output = func_2884()
func_2885 = relay.Function([], output)
mutated_mod['func_2885'] = func_2885
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_2891 = func_1576_call()
call_2892 = func_1576_call()
output = call_2891
output2 = call_2892
func_2894 = relay.Function([], output)
mod['func_2894'] = func_2894
mod = relay.transform.InferType()(mod)
mutated_mod['func_2894'] = func_2894
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2894_call = mutated_mod.get_global_var('func_2894')
call_2895 = func_2894_call()
output = call_2895
func_2896 = relay.Function([], output)
mutated_mod['func_2896'] = func_2896
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2900 = relay.const([[[7,-6,8,-7,-8,-8,-6],[-2,2,-2,6,-2,-8,-4],[-8,-6,-10,1,-10,-3,-9],[-1,-6,3,-6,6,3,-7],[-8,9,1,-2,10,1,4]],[[-10,-4,9,6,9,-9,6],[-2,4,3,5,1,-4,-1],[-2,-8,-7,7,-2,7,7],[-6,-9,5,6,7,1,-9],[8,2,4,-6,2,-6,-8]],[[-8,1,7,2,-3,9,2],[-5,-5,-10,5,-6,-4,6],[-3,-4,-7,-4,-7,6,5],[-8,-5,7,-8,-8,-2,5],[3,5,-4,-6,6,-7,6]],[[-2,9,5,3,-4,10,3],[6,-5,7,-4,3,-4,-10],[1,-4,-5,3,7,-4,4],[-2,-6,-2,7,3,2,3],[-4,-1,-9,9,10,-7,-5]],[[-9,-7,-3,1,-3,9,-2],[-3,-9,8,-3,-4,2,-4],[2,-2,-8,2,-9,1,3],[-8,-10,-2,10,-8,-8,5],[8,-10,1,-1,-8,-2,-8]],[[-1,-2,-8,1,-1,9,-2],[-6,6,10,-6,8,10,10],[5,7,-1,-4,-1,-6,-4],[4,1,-6,-3,4,7,-7],[-10,-1,7,10,8,4,8]]], dtype = "int16")#candidate|2900|(6, 5, 7)|const|int16
var_2901 = relay.var("var_2901", dtype = "int16", shape = (6, 5, 7))#candidate|2901|(6, 5, 7)|var|int16
bop_2902 = relay.bitwise_and(const_2900.astype('int16'), relay.reshape(var_2901.astype('int16'), relay.shape_of(const_2900))) # shape=(6, 5, 7)
output = relay.Tuple([bop_2902,])
output2 = relay.Tuple([bop_2902,])
func_2906 = relay.Function([var_2901,], output)
mod['func_2906'] = func_2906
mod = relay.transform.InferType()(mod)
var_2907 = relay.var("var_2907", dtype = "int16", shape = (6, 5, 7))#candidate|2907|(6, 5, 7)|var|int16
output = func_2906(var_2907)
func_2908 = relay.Function([var_2907], output)
mutated_mod['func_2908'] = func_2908
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2748_call = mod.get_global_var('func_2748')
func_2750_call = mutated_mod.get_global_var('func_2750')
call_2910 = relay.TupleGetItem(func_2748_call(), 0)
call_2911 = relay.TupleGetItem(func_2750_call(), 0)
func_1815_call = mod.get_global_var('func_1815')
func_1816_call = mutated_mod.get_global_var('func_1816')
call_2913 = func_1815_call()
call_2914 = func_1815_call()
func_1881_call = mod.get_global_var('func_1881')
func_1885_call = mutated_mod.get_global_var('func_1885')
var_2932 = relay.var("var_2932", dtype = "float32", shape = (6,))#candidate|2932|(6,)|var|float32
var_2933 = relay.var("var_2933", dtype = "float32", shape = (378,))#candidate|2933|(378,)|var|float32
call_2931 = relay.TupleGetItem(func_1881_call(relay.reshape(var_2932.astype('float32'), [1, 1, 6]), relay.reshape(var_2933.astype('float32'), [1, 378]), ), 3)
call_2934 = relay.TupleGetItem(func_1885_call(relay.reshape(var_2932.astype('float32'), [1, 1, 6]), relay.reshape(var_2933.astype('float32'), [1, 378]), ), 3)
bop_2936 = relay.greater_equal(call_2910.astype('bool'), call_2931.astype('bool')) # shape=(2, 378)
bop_2939 = relay.greater_equal(call_2911.astype('bool'), call_2934.astype('bool')) # shape=(2, 378)
uop_2941 = relay.asinh(call_2931.astype('float64')) # shape=(1, 378)
uop_2943 = relay.asinh(call_2934.astype('float64')) # shape=(1, 378)
bop_2950 = relay.not_equal(uop_2941.astype('bool'), bop_2936.astype('bool')) # shape=(2, 378)
bop_2953 = relay.not_equal(uop_2943.astype('bool'), bop_2939.astype('bool')) # shape=(2, 378)
func_2340_call = mod.get_global_var('func_2340')
func_2342_call = mutated_mod.get_global_var('func_2342')
const_2959 = relay.const([1.133750,-0.455761,-1.625964,1.453480,1.154813,-0.206800,-0.315465,4.893789,-7.528841,3.752581,3.298045,8.495827,-0.097119,-3.130623,2.744465,-5.091628,-9.309436,-5.382568,-4.515884,-7.865000,4.151195,0.279390,0.490198,-6.547658,-7.407560,-3.456321,9.241297,6.166015,6.782115,-4.913219,8.331396,7.960678,0.210046,5.631477,-0.176917,2.552607,-3.014242,6.114617,4.209268,4.799487,-3.597674,4.367621,1.240603,9.004553,2.108601,1.102797,-7.899339,-9.945486,7.074235,-1.171106,0.061675,0.599856,-3.973152,-3.239020,-4.636588,1.036899,8.096387,-9.728202,-3.412910,0.712482,5.517470,-5.674941,-1.032594,0.554160,-1.234942,7.057277,-2.207468,-1.507138,4.750116,-9.490054,-3.529175,-0.778454,3.287886,0.017829,-1.406579,2.144572,8.749842,-7.045040,-7.337244,3.151887,3.204995,5.505708,2.535718,-9.504170,-3.439631,-5.173475,0.484459,-2.338550,-3.724088,-1.917315,-9.555534,-0.768954,8.549758,1.760731,3.778427,4.889534,0.203640,-8.347208,-9.343571,-4.860569,9.846348,0.130912,7.772694,5.850791,1.998614,7.769771,9.651229,-7.928334,-6.600515,3.190782,-0.457407,-3.227971,7.177442,3.085583,7.360902,-6.365486,-0.537089,2.549960,0.589041,4.916564,-6.929219,-1.832634,4.259460,3.816477,8.156106,2.529054,9.652128,7.154276,-9.942923,-5.929560,-2.639227,-1.545490,-4.486415,2.862577,0.474534,0.660474,6.647917,3.935610,-7.227015,-4.326614,2.010971,5.550950,7.623132,1.801152,0.546363,9.488376,-4.610388,3.702917,-8.907506,-8.609340,5.352742,4.891257,-2.814364,6.918900,4.922541,-8.218393,0.678277,4.186392,-4.725129,-4.648447,-7.712884,-0.441834,-6.891706,-7.890747,-0.168539,-4.930695,-1.357908,-6.235128,-2.868313,6.593612,0.248013,5.080436,2.293639,9.638155,1.187659,-5.474618,4.046651,0.798014,-3.662764,-8.915690,-8.686488,-7.137524,7.221714,8.497563,4.339972,9.295544,-5.494830,9.270500,-8.589567,-5.464795,8.167407,4.610419,9.388799,-1.521256,-8.547373,-3.495770,1.512123,3.367614,1.220764,-9.183784,5.348925,7.606296,-2.646599,-0.832405,9.965679,2.757158,1.956726,4.397404,9.432285,-1.006437,1.156904,-2.118364,6.071915,4.480544,5.542102,1.753321,9.373685,-1.667264,-2.755624,-3.026091,0.807286,-6.856309,3.945203,8.294206,5.659608,-3.826837,-2.324448,5.780538,7.514593,-9.541400,-9.322913], dtype = "float32")#candidate|2959|(231,)|const|float32
call_2958 = relay.TupleGetItem(func_2340_call(relay.reshape(const_2959.astype('float32'), [231,])), 3)
call_2960 = relay.TupleGetItem(func_2342_call(relay.reshape(const_2959.astype('float32'), [231,])), 3)
func_1547_call = mod.get_global_var('func_1547')
func_1551_call = mutated_mod.get_global_var('func_1551')
var_2962 = relay.var("var_2962", dtype = "float32", shape = (2, 252))#candidate|2962|(2, 252)|var|float32
var_2963 = relay.var("var_2963", dtype = "int16", shape = (288,))#candidate|2963|(288,)|var|int16
call_2961 = relay.TupleGetItem(func_1547_call(relay.reshape(var_2962.astype('float32'), [14, 9, 4]), relay.reshape(var_2963.astype('int16'), [6, 6, 8]), relay.reshape(var_2962.astype('float32'), [14, 9, 4]), ), 5)
call_2964 = relay.TupleGetItem(func_1551_call(relay.reshape(var_2962.astype('float32'), [14, 9, 4]), relay.reshape(var_2963.astype('int16'), [6, 6, 8]), relay.reshape(var_2962.astype('float32'), [14, 9, 4]), ), 5)
output = relay.Tuple([call_2913,var_2932,var_2933,bop_2950,call_2958,const_2959,call_2961,var_2962,var_2963,])
output2 = relay.Tuple([call_2914,var_2932,var_2933,bop_2953,call_2960,const_2959,call_2964,var_2962,var_2963,])
func_2968 = relay.Function([var_2932,var_2933,var_2962,var_2963,], output)
mod['func_2968'] = func_2968
mod = relay.transform.InferType()(mod)
var_2969 = relay.var("var_2969", dtype = "float32", shape = (6,))#candidate|2969|(6,)|var|float32
var_2970 = relay.var("var_2970", dtype = "float32", shape = (378,))#candidate|2970|(378,)|var|float32
var_2971 = relay.var("var_2971", dtype = "float32", shape = (2, 252))#candidate|2971|(2, 252)|var|float32
var_2972 = relay.var("var_2972", dtype = "int16", shape = (288,))#candidate|2972|(288,)|var|int16
output = func_2968(var_2969,var_2970,var_2971,var_2972,)
func_2973 = relay.Function([var_2969,var_2970,var_2971,var_2972,], output)
mutated_mod['func_2973'] = func_2973
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_2987 = func_1953_call()
call_2988 = func_1953_call()
output = relay.Tuple([call_2987,])
output2 = relay.Tuple([call_2988,])
func_2990 = relay.Function([], output)
mod['func_2990'] = func_2990
mod = relay.transform.InferType()(mod)
mutated_mod['func_2990'] = func_2990
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2990_call = mutated_mod.get_global_var('func_2990')
call_2991 = func_2990_call()
output = call_2991
func_2992 = relay.Function([], output)
mutated_mod['func_2992'] = func_2992
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1763_call = mod.get_global_var('func_1763')
func_1764_call = mutated_mod.get_global_var('func_1764')
call_3045 = func_1763_call()
call_3046 = func_1763_call()
output = relay.Tuple([call_3045,])
output2 = relay.Tuple([call_3046,])
func_3047 = relay.Function([], output)
mod['func_3047'] = func_3047
mod = relay.transform.InferType()(mod)
mutated_mod['func_3047'] = func_3047
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3047_call = mutated_mod.get_global_var('func_3047')
call_3048 = func_3047_call()
output = call_3048
func_3049 = relay.Function([], output)
mutated_mod['func_3049'] = func_3049
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2748_call = mod.get_global_var('func_2748')
func_2750_call = mutated_mod.get_global_var('func_2750')
call_3147 = relay.TupleGetItem(func_2748_call(), 0)
call_3148 = relay.TupleGetItem(func_2750_call(), 0)
output = call_3147
output2 = call_3148
func_3156 = relay.Function([], output)
mod['func_3156'] = func_3156
mod = relay.transform.InferType()(mod)
mutated_mod['func_3156'] = func_3156
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3156_call = mutated_mod.get_global_var('func_3156')
call_3157 = func_3156_call()
output = call_3157
func_3158 = relay.Function([], output)
mutated_mod['func_3158'] = func_3158
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1645_call = mod.get_global_var('func_1645')
func_1647_call = mutated_mod.get_global_var('func_1647')
call_3193 = relay.TupleGetItem(func_1645_call(), 0)
call_3194 = relay.TupleGetItem(func_1647_call(), 0)
var_3202 = relay.var("var_3202", dtype = "float32", shape = (2, 1))#candidate|3202|(2, 1)|var|float32
bop_3203 = relay.not_equal(call_3193.astype('bool'), relay.reshape(var_3202.astype('bool'), relay.shape_of(call_3193))) # shape=(2, 1)
bop_3206 = relay.not_equal(call_3194.astype('bool'), relay.reshape(var_3202.astype('bool'), relay.shape_of(call_3194))) # shape=(2, 1)
bop_3212 = relay.minimum(var_3202.astype('int16'), relay.reshape(bop_3203.astype('int16'), relay.shape_of(var_3202))) # shape=(2, 1)
bop_3215 = relay.minimum(var_3202.astype('int16'), relay.reshape(bop_3206.astype('int16'), relay.shape_of(var_3202))) # shape=(2, 1)
bop_3217 = relay.greater(bop_3212.astype('bool'), relay.reshape(bop_3203.astype('bool'), relay.shape_of(bop_3212))) # shape=(2, 1)
bop_3220 = relay.greater(bop_3215.astype('bool'), relay.reshape(bop_3206.astype('bool'), relay.shape_of(bop_3215))) # shape=(2, 1)
output = relay.Tuple([bop_3217,])
output2 = relay.Tuple([bop_3220,])
func_3223 = relay.Function([var_3202,], output)
mod['func_3223'] = func_3223
mod = relay.transform.InferType()(mod)
mutated_mod['func_3223'] = func_3223
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3224 = relay.var("var_3224", dtype = "float32", shape = (2, 1))#candidate|3224|(2, 1)|var|float32
func_3223_call = mutated_mod.get_global_var('func_3223')
call_3225 = func_3223_call(var_3224)
output = call_3225
func_3226 = relay.Function([var_3224], output)
mutated_mod['func_3226'] = func_3226
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2311_call = mod.get_global_var('func_2311')
func_2313_call = mutated_mod.get_global_var('func_2313')
call_3242 = relay.TupleGetItem(func_2311_call(), 2)
call_3243 = relay.TupleGetItem(func_2313_call(), 2)
uop_3259 = relay.sigmoid(call_3242.astype('float64')) # shape=(60, 1)
uop_3261 = relay.sigmoid(call_3243.astype('float64')) # shape=(60, 1)
func_2990_call = mod.get_global_var('func_2990')
func_2992_call = mutated_mod.get_global_var('func_2992')
call_3262 = relay.TupleGetItem(func_2990_call(), 0)
call_3263 = relay.TupleGetItem(func_2992_call(), 0)
func_882_call = mod.get_global_var('func_882')
func_885_call = mutated_mod.get_global_var('func_885')
const_3271 = relay.const([-0.837916,-5.840942,5.155411,-9.741142,-2.794858,-4.883571,4.146173,0.291061,-4.162099,-0.859076,-8.433215,-6.543227,1.565500,-2.025485,6.195320,-0.886164,5.994372,-9.072334,7.058919,-6.506470,0.013625,5.962516,9.632818,-2.364159,3.005233,7.667398,6.487677,1.275097,8.621866,9.448056,9.830227,-3.621180,6.414088,6.006286,1.581150,-1.029296,-7.762000,-3.141500,-0.432529,6.538148,5.780775,6.681825,-3.589968,7.712574,8.868726,5.330868,-9.425225,3.555526,-0.684047,-9.537030,-2.413780,-4.210042,-4.966295,-1.523773,4.287275,3.168755,7.521506,-7.132945,7.265524,9.922401,-3.721963,5.517935,0.614911,-9.513417,4.055123,9.629443,0.345533,-9.910359,-2.776203,1.434793,0.884379,-4.535477,0.312548,-3.303074,-5.781993,6.002079,-7.128677,-1.470848,-8.511000,0.773987,6.346617,7.476613,5.780199,4.068271,0.880629,1.661027,5.715000,0.750931,-6.161112,9.069975], dtype = "float64")#candidate|3271|(90,)|const|float64
call_3270 = func_882_call(relay.reshape(const_3271.astype('float64'), [6, 15]))
call_3272 = func_882_call(relay.reshape(const_3271.astype('float64'), [6, 15]))
output = relay.Tuple([uop_3259,call_3262,call_3270,const_3271,])
output2 = relay.Tuple([uop_3261,call_3263,call_3272,const_3271,])
func_3276 = relay.Function([], output)
mod['func_3276'] = func_3276
mod = relay.transform.InferType()(mod)
mutated_mod['func_3276'] = func_3276
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3276_call = mutated_mod.get_global_var('func_3276')
call_3277 = func_3276_call()
output = call_3277
func_3278 = relay.Function([], output)
mutated_mod['func_3278'] = func_3278
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3318 = relay.var("var_3318", dtype = "bool", shape = (16, 12, 9))#candidate|3318|(16, 12, 9)|var|bool
var_3319 = relay.var("var_3319", dtype = "bool", shape = (16, 12, 9))#candidate|3319|(16, 12, 9)|var|bool
bop_3320 = relay.logical_or(var_3318.astype('bool'), relay.reshape(var_3319.astype('bool'), relay.shape_of(var_3318))) # shape=(16, 12, 9)
func_2582_call = mod.get_global_var('func_2582')
func_2583_call = mutated_mod.get_global_var('func_2583')
call_3331 = relay.TupleGetItem(func_2582_call(), 0)
call_3332 = relay.TupleGetItem(func_2583_call(), 0)
func_2852_call = mod.get_global_var('func_2852')
func_2855_call = mutated_mod.get_global_var('func_2855')
const_3337 = relay.const([3,4,-6,-4,5,1,-10,5,-1,-1,-9,2,-1,-4,-3,8,7,-7,2,9,3,-1,9,6,7,8,-6,-1,9,3,-6,-8,-3,-10,-8,4,1,6,-9,9,4,1,-8,-4,10,1,-10,-3,-5,10,1,5,-9,-1,-7,-6,8,10,-2,4,-4,1,3,-1,-4,10,-4,9,-9,-6,1,-7,1,-6,4,8,3,3,1,4,4,-5,2,6,4,-8,-3,-5,-4,7,9,5,-5,9,4,-3,9,9,-2,3,-4,-1,7,1,-7,1,-4,-1,7,-4,5,-9,9,-8,-10,7,1,1,5,1,-10,9,9,4,-10,-6,8,-1], dtype = "uint32")#candidate|3337|(128,)|const|uint32
call_3336 = func_2852_call(relay.reshape(const_3337.astype('uint32'), [4, 4, 8]), relay.reshape(const_3337.astype('uint32'), [4, 4, 8]), )
call_3338 = func_2852_call(relay.reshape(const_3337.astype('uint32'), [4, 4, 8]), relay.reshape(const_3337.astype('uint32'), [4, 4, 8]), )
var_3340 = relay.var("var_3340", dtype = "bool", shape = (16, 12, 9))#candidate|3340|(16, 12, 9)|var|bool
bop_3341 = relay.equal(var_3318.astype('bool'), relay.reshape(var_3340.astype('bool'), relay.shape_of(var_3318))) # shape=(16, 12, 9)
uop_3349 = relay.log2(var_3319.astype('float64')) # shape=(16, 12, 9)
output = relay.Tuple([bop_3320,call_3331,call_3336,const_3337,bop_3341,uop_3349,])
output2 = relay.Tuple([bop_3320,call_3332,call_3338,const_3337,bop_3341,uop_3349,])
func_3353 = relay.Function([var_3318,var_3319,var_3340,], output)
mod['func_3353'] = func_3353
mod = relay.transform.InferType()(mod)
mutated_mod['func_3353'] = func_3353
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3353_call = mutated_mod.get_global_var('func_3353')
var_3355 = relay.var("var_3355", dtype = "bool", shape = (16, 12, 9))#candidate|3355|(16, 12, 9)|var|bool
var_3356 = relay.var("var_3356", dtype = "bool", shape = (16, 12, 9))#candidate|3356|(16, 12, 9)|var|bool
var_3357 = relay.var("var_3357", dtype = "bool", shape = (16, 12, 9))#candidate|3357|(16, 12, 9)|var|bool
call_3354 = func_3353_call(var_3355,var_3356,var_3357,)
output = call_3354
func_3358 = relay.Function([var_3355,var_3356,var_3357,], output)
mutated_mod['func_3358'] = func_3358
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3390 = relay.const([[[-5.653168,-0.092479],[5.175524,1.340184],[7.640453,-0.707095],[4.569718,-8.922007],[4.807759,-7.946595]],[[-6.186298,-7.905017],[-7.949606,1.729977],[-7.523579,-3.931122],[-2.583148,-2.941495],[3.181473,6.218756]],[[-3.760585,4.839868],[4.039692,2.465965],[-1.382481,-6.360544],[0.020265,5.970888],[-8.014236,-0.836663]],[[3.001072,9.401275],[-6.558970,3.285738],[6.070373,-8.354108],[3.024346,6.830179],[-0.992924,1.235546]],[[-4.345350,7.248156],[8.174617,6.797570],[3.734771,-7.658374],[-5.647469,-7.978668],[4.845872,-5.018107]],[[3.627268,8.944203],[-3.015871,7.894251],[3.323537,4.826950],[-2.471360,4.859527],[-7.604076,0.938170]]], dtype = "float64")#candidate|3390|(6, 5, 2)|const|float64
uop_3391 = relay.sqrt(const_3390.astype('float64')) # shape=(6, 5, 2)
func_2235_call = mod.get_global_var('func_2235')
func_2239_call = mutated_mod.get_global_var('func_2239')
const_3394 = relay.const([7.004453,9.525736,6.692002,-4.510954,8.057034,-2.685528,3.380128,8.080319,6.361067,2.057385,9.464921,-5.325515,7.604729,5.816709,-0.736210,9.917486,9.304799,5.301656,9.882529,-9.160256,-4.838596,3.741980,-8.593006,9.900533,3.659296,-7.361283,-7.845600,-3.129595,3.764169,-7.897460,-3.292130,-9.706317,-8.650730,-0.679652,6.330974,3.206788,9.936856,6.632886,7.163486,7.640296,-6.510785,1.474834,4.202067,-7.007880,-9.302023,-9.548127,-7.207727,-5.793592,-9.749259,-4.567583,8.408799,4.327069,-3.650373,7.186419,9.204129,7.278398,-9.085936,4.576697,-0.311676,1.817244,-1.420931,4.465926,-1.257049,-8.600646,-3.328197,-0.563147,-7.185666,4.552711,3.913148,-9.576181,4.521462,-4.305886,2.016435,1.818983,5.728967,-0.689079,-6.162271,-4.042590,5.467145,-3.068125,7.442487,-6.408644,-7.040979,5.097596,-7.919447,1.473896,4.807274,9.161489,2.558142,-9.796154,5.923127,-3.724471,6.082914,-3.164880,0.327768,-9.065454,-3.566965,-9.339780,-1.766281,-8.693932,-9.780504,9.878880,2.596873,-5.193350,2.458606,5.783328,9.924409,1.649788,-1.237847,3.379160,-0.179798,-0.838531,-5.364473,4.624946,-7.428385,3.120988,0.671929,-9.734622,-8.021618,9.168277,-1.764755,-7.198103,4.023404,-9.081102,-9.829219,1.410957,0.426448,0.784594,3.612897,4.888002,-1.211025,-7.147132,3.287250,-8.787332,3.078791,9.904939,-3.129624,6.914693,0.545784,7.806301,7.753955,-1.872556,-2.321111,4.362518,2.214012,-7.531797,8.085253,7.226393,7.294646,-2.648882,-2.719878,-9.744511,-4.110418,-2.654159,8.436124,9.091345,-8.712324,-9.355590,7.722207,-1.380273,-7.044916,-4.362083,2.225882,-0.846615,0.091977,4.620495,-5.232040,-0.376392,1.143804,4.567713,-8.168092,-1.518752,-8.396429,5.579702,-3.786645,-6.578566,-1.119229,-9.424415,4.798362,3.214992,-3.783751,-4.551165,-5.268185,3.994674,0.435223,4.768878,9.060738,-4.074668,-1.177043,4.722771,-9.706260,6.277928,-7.025740,-8.377002,-9.240137,-1.254438,9.232792,4.460245,4.735617,4.806342,3.074406,-0.003158,-0.497959,6.527388,-1.629162,3.171776,-3.621245,-5.938827,2.361400,-9.528362,-9.142989,-0.884604,3.717825,1.409043,-2.690420,5.305102,3.870417,-2.973384,2.649266,-1.979308,-8.411585,-3.406463,-2.544679,9.586592,0.605496,7.499228,-7.589893,1.913269,-1.106453,-5.443215,-8.777132,-1.430455,9.083882,4.969886,-7.993607,-1.374129,-7.986987,-8.544982,0.363093,-4.698435,4.532442,-0.550148,9.636833,-4.345669,7.333992,-6.231045,-1.175319,2.840712,6.177221,-9.562090,2.402897,-1.656484,8.247938,-0.997415,-3.940220,-7.250308,7.263201,7.839663,-0.610513,-1.338405,-8.540922,5.020651,6.233916,-0.434017,-1.825141,0.157300,7.894413,5.249470,-9.812577,-2.079871,-1.457242,-6.210770,9.719669,-9.568537,6.463662,6.928329,1.455750,3.752556,-6.427107,2.822289,-9.705461,-0.713230,-1.566507,1.896814,0.239971,2.069265,-1.638392,8.183068,0.812496,2.010540,-5.400556,-9.813772,-1.487601,0.851049,0.002606,0.985084,5.761708,6.193539,9.784409,-1.314291,-7.985827,-3.691528,5.096397,4.094845,0.183389,-7.858716,4.880584,5.082392,-8.464218,-2.857248,4.141343,4.289387,5.353597,-8.250736,1.060947,-2.741782,-6.459696,-7.481604,-4.567249,-8.409311,1.633532,-7.812206,3.720897,-4.024588,-9.033335,1.773756,-2.443613,-7.736969,3.746043,-4.803098,5.944091,3.411635,0.780876,8.959810,1.032665,8.292712,-2.101210,4.865098,1.907450,5.370906,5.042795,4.003269,-0.480422,6.928273,-4.217242,-0.981533,-4.026467,2.342916,-2.825143,-3.538043,-6.027590,-0.131361,1.477967,-3.315856,6.967311,-8.733194,-2.273622,-4.544398,9.683134,-9.308166,1.420361,8.709246,-3.986660,2.058509,0.982569,-6.952246,1.047208,3.457316,0.262912,6.877586,7.847607,7.173368,8.385031,-3.703790,9.162777,-9.006406,-2.639298,4.150944,-7.050081,0.541607,-1.461448,6.495216,-0.336941,-8.988223,-9.719642,-5.245368,6.938299,9.937839,1.425906,6.152288,-8.979929,7.869393,7.907236,-2.855078,8.208029,0.126382,6.366832,0.957459,9.452194,5.978738,2.014649,-9.263377,8.444231,5.626351,6.258135,-6.639689,-3.126275,-0.940332,-7.627157,5.771648,-6.208451,5.823369,8.479324,9.889243,-0.633389,-7.119031,-2.514818,-7.769289,-5.742441,-2.266506,-7.902587,6.135154,-7.131576,-4.841915,-6.621155,-2.897228,-1.592414,7.380220,-3.285981,7.809831,7.448211,-5.249649,-0.756505,-5.247037,-8.924626,-7.944050,2.243736,7.853937,-9.479550,-4.579133,0.220327,2.574260,-6.491905,4.265502,-8.691975,5.069936,-2.460822,1.648834,-3.147533,5.923910,4.318475,8.005822,-2.789583,-2.263217,-9.768342,8.400217,-0.045846,0.701369,6.494654,7.116906,6.332156,-8.548716,0.019218,-0.938216,6.220442,-4.749651,-6.032229,8.219738,8.330015,9.845531,-8.764072,5.412174,8.526910,0.396034,-9.091886,-1.993767,3.321263,1.950817,7.974667,-3.705396,-5.737093,9.129387,-6.767924,8.420894,-9.005328,6.289115,8.141214,5.073877,9.963890,-4.886075,-9.444516,-2.952056,6.214443,-3.860677,0.724545,8.894363,7.517328,-4.396199,1.337393,9.756118,-3.129811,-9.323279,-5.280695,0.731084,-4.963246,4.491566,-7.294271,-7.621563,5.549151,-6.328747,6.434543,-6.857524,-7.111307,-0.194360,-6.733674,-1.861842,-0.729615,5.226863,-2.401141,-6.894323,-4.399757,6.024710,1.617725,-5.739782,-7.409797,-8.538178,-7.991423,5.762435,8.468373,-2.182103,6.030681,5.885748,0.874087,-3.543349,5.208301,-8.015784,-1.952581,-1.726749,6.271533,5.500959,-8.194037,1.387462,-4.884372,2.419024,-5.217779,-1.820525,-1.686617,8.754457,-9.095126,-1.093571,-3.917789,2.563077,-0.831471,8.363910,-5.666712,-3.017141,-2.664909,6.510739,-4.772375,-4.380980,-3.919587,-0.496581,-7.174654,-2.417161,-2.652830,-0.186900,4.108435,4.397839,-8.854414,6.076429,2.125419,0.654114,9.059305,8.629598,5.652852,-8.839838,-4.672916,1.825141,-8.380347,9.223445,-1.914483,3.388853,7.362369,8.210130,4.479014,-2.790508,-1.371710,-6.933864], dtype = "float32")#candidate|3394|(588,)|const|float32
call_3393 = relay.TupleGetItem(func_2235_call(relay.reshape(const_3394.astype('float32'), [6, 14, 7]), relay.reshape(const_3394.astype('float32'), [6, 14, 7]), ), 0)
call_3395 = relay.TupleGetItem(func_2239_call(relay.reshape(const_3394.astype('float32'), [6, 14, 7]), relay.reshape(const_3394.astype('float32'), [6, 14, 7]), ), 0)
output = relay.Tuple([uop_3391,call_3393,const_3394,])
output2 = relay.Tuple([uop_3391,call_3395,const_3394,])
func_3404 = relay.Function([], output)
mod['func_3404'] = func_3404
mod = relay.transform.InferType()(mod)
mutated_mod['func_3404'] = func_3404
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3404_call = mutated_mod.get_global_var('func_3404')
call_3405 = func_3404_call()
output = call_3405
func_3406 = relay.Function([], output)
mutated_mod['func_3406'] = func_3406
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_3431 = func_1576_call()
call_3432 = func_1576_call()
output = call_3431
output2 = call_3432
func_3436 = relay.Function([], output)
mod['func_3436'] = func_3436
mod = relay.transform.InferType()(mod)
mutated_mod['func_3436'] = func_3436
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3436_call = mutated_mod.get_global_var('func_3436')
call_3437 = func_3436_call()
output = call_3437
func_3438 = relay.Function([], output)
mutated_mod['func_3438'] = func_3438
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2990_call = mod.get_global_var('func_2990')
func_2992_call = mutated_mod.get_global_var('func_2992')
call_3454 = relay.TupleGetItem(func_2990_call(), 0)
call_3455 = relay.TupleGetItem(func_2992_call(), 0)
output = call_3454
output2 = call_3455
func_3458 = relay.Function([], output)
mod['func_3458'] = func_3458
mod = relay.transform.InferType()(mod)
output = func_3458()
func_3459 = relay.Function([], output)
mutated_mod['func_3459'] = func_3459
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1576_call = mod.get_global_var('func_1576')
func_1577_call = mutated_mod.get_global_var('func_1577')
call_3559 = func_1576_call()
call_3560 = func_1576_call()
output = relay.Tuple([call_3559,])
output2 = relay.Tuple([call_3560,])
func_3565 = relay.Function([], output)
mod['func_3565'] = func_3565
mod = relay.transform.InferType()(mod)
mutated_mod['func_3565'] = func_3565
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3565_call = mutated_mod.get_global_var('func_3565')
call_3566 = func_3565_call()
output = call_3566
func_3567 = relay.Function([], output)
mutated_mod['func_3567'] = func_3567
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2721_call = mod.get_global_var('func_2721')
func_2722_call = mutated_mod.get_global_var('func_2722')
call_3584 = relay.TupleGetItem(func_2721_call(), 1)
call_3585 = relay.TupleGetItem(func_2722_call(), 1)
uop_3592 = relay.cos(call_3584.astype('float32')) # shape=(2, 1)
uop_3594 = relay.cos(call_3585.astype('float32')) # shape=(2, 1)
func_1711_call = mod.get_global_var('func_1711')
func_1714_call = mutated_mod.get_global_var('func_1714')
var_3619 = relay.var("var_3619", dtype = "float32", shape = (1, 60))#candidate|3619|(1, 60)|var|float32
const_3620 = relay.const([-2.734425,-3.373920,4.361417,-3.911757,-2.935501,-4.911561,-7.176295,4.816889,-0.678387,0.474063,6.909000,8.546816,-3.344111,-7.864642,-9.496759,-0.242433,-4.236712,5.248464,-6.631784,6.815188,5.854059,7.407264,8.972970,4.621891,0.876568,-1.029749,-1.664031,5.985649,-1.109272,-2.440112,-8.559833,-1.053892,-1.078566,-9.373330,9.212212,8.208382,8.819492,-9.440500,8.291084,9.422348,7.768334,0.854218,3.645936,-6.315449,-5.235109,-7.494407,-9.819439,-1.717201,-1.767996,5.176775,-8.658574,-7.409203,-6.049756,-8.147759,-1.666967,-6.341902,9.385194,-8.705761,-6.046367,3.811057,5.191484,-3.179911,-9.894090,-7.683931,9.460643,-7.465442,-8.560462,-0.602654,-1.562859,-1.151863,-0.130414,-0.389546,-0.397967,-3.206144,2.705158,-0.695411,-1.637641,-5.278536,1.321406,6.456350,6.358917,-8.642319,4.946178,-1.443567,-0.364925,-2.403912,0.838053,-7.482540,2.164058,7.501899,-8.249962,-3.094030,-1.914352,6.438756,2.192009,-3.582008,3.161967,4.858253,8.442484,-2.210893,-8.336152,-3.186180,6.353492,-0.990599,-9.770902,6.595924,7.086540,2.785841,1.217000,0.648322,5.376533,2.806886,-7.631393,7.139336,-6.597149,0.585320,1.397703,-1.631606,2.597020,1.381104,-6.716400,-6.059853,-9.420738,2.074777,7.437259,4.561741,8.689405,9.085111,0.330834,0.507760,-5.518379,-3.302960,4.290097,-8.919232,2.345951,-8.532055,-1.059835,0.876705,-7.796337,-7.476146,1.565633,-9.145780,-3.909516,6.705674,5.737401,8.038284,3.605553,0.158206,7.364222,8.253243,7.025664,-2.560234,3.785697,4.842471,6.501123,1.568320,5.799205,8.146662,1.960976,1.821847,7.569654,-2.084268,-7.752014,1.683353,4.061638,1.552164,-6.988172,-9.780128,9.300706,9.043224,-4.249715,-7.333813,0.130701,4.162819,-0.518558,1.152776,-6.744018,-2.577601,0.568016,-2.067521,5.722772,8.668647,1.627336,6.667958,0.596442,1.947664,7.369182,3.098590,4.125762,7.349322,4.768382,-6.598331,-6.929744,8.527159,-2.760983,-6.059033,4.752607,-1.464031,0.623447,2.295733,-9.261187,-6.219975,-9.535479,-2.891783,3.672424,9.813327,-9.164622,-4.702646,0.567482,-4.882847,0.162060,7.178713,3.833735,1.577496,1.996251,0.759471,-2.323955,6.624150,-1.081988,-9.476592,4.405537,-5.435372,0.160017,-1.372319,-2.579991,6.010005,9.409123,-0.541290,-9.779347,4.292000,8.700191,8.254956,2.300574,-2.745844,1.740957,7.545393,6.893225,-2.921265,-2.200665,6.203785,-4.012338,-6.335441,2.640460,-6.068470,-2.928651,5.555878,-2.704412,0.975640,5.887431,7.837195,3.348106,0.414259,0.631187,8.593312,-5.238510,-6.417586,7.674088,-3.779541,-8.197639,-4.402712,-6.212608,-2.786443,5.026289,-9.316307,3.396422,2.542674,-9.998390,3.896658,-5.754049,5.835484,8.248761,0.682310,-5.584352,3.840686,-4.736950,-1.161578,6.454500,-3.637805,-6.925573,7.468385,-8.181065,-7.609511,4.052540,-7.871440,4.929385,-6.117546,-6.963071,0.991211,4.229510,2.755139,4.120550,8.849293,-0.513973,5.827129,-7.508435,-1.090197,3.241151,-5.312623,1.984026,-5.597945,-7.676334,9.201311,1.894405,-3.768779,-2.763384,-3.988455,5.085202,-1.784317,0.288669,-0.796113,1.969914,9.541825,-2.721799,6.127378,-4.142388,-2.407283,5.979504,9.097990,-2.355388,7.439969,1.560971,-9.258959,5.568981,-8.914326,-7.545471,1.851040,-3.458438,-9.726812,-4.452511,-0.466453,-5.424838,6.747895,-4.801562,-4.848576,-5.740648,3.521131,0.380849,-9.133699,9.570506,-0.591330,-7.153409,2.949993,-6.335782,9.377523,2.709759,9.966708,-9.816150,6.176395,-5.947475,3.938027,-1.904620,-4.927708,5.099822,-2.228481,-6.133197,-5.909100,-3.133650,6.857360,-1.391002,6.734771,-7.688832,9.217941,-6.058426,6.539944,-1.189386,4.093381,-1.296816,-4.962208,7.369456,-4.921747,8.636923,-1.497223,-2.347351,-1.609992,-7.614305,-3.315303,0.358592,6.170787,7.901507,-9.840025,2.165008,-2.484959,2.516406,6.287777,9.097026,-7.666028,0.564084,6.753186,-8.174916,-4.273393,4.137985,7.889548,-6.170336,-1.082397,-0.704881,5.884680,9.557183,6.868450,-1.907950,2.875058,-8.451037,3.734200,-5.632594,2.965098,-5.962974,6.676168,-2.309891,-4.795825,7.535025,-3.627781,6.949385,4.859562,6.172745,-0.925290,-2.918949,-8.741809,4.540951,-3.531327,-3.148484,8.172542,9.411838,-2.336527,6.853569,-9.120564,-3.965953,-1.873322,-0.212972,-1.578753,-2.552828,9.502013,-7.017160,2.915626,8.586660,9.395060,1.220581,5.931421,9.122858,8.725083,-6.331940,2.954998,-7.953357,9.188441,3.184756,7.832284,-8.649000,-9.279920,-1.393133,-1.858601,1.049622,5.396474,-7.834177,9.442804,-0.821845,6.330612,-1.185152,7.933051,-6.660801,0.758994,-2.938140,-2.759961,-5.811343,4.880419,-2.041807,-7.277402,0.012468,8.212888,-2.608734,8.494783,6.901810,-0.773238,2.465884,-3.597941,-7.173732,-2.632069,-1.449039,6.961054,5.103395,6.097164,-6.717425,-4.570760,8.027003,-9.863874,-8.370274,2.121232,-5.945647,-0.913130,2.847833,0.179160,-9.043541,7.802716,-0.072183,0.798655,0.938384,-9.959545,6.919672,-4.301835,-9.632940,-2.291986,-4.839488,-3.321839,5.298425,8.565416,9.965784,-9.837121,-4.981579,6.200170,-7.652966,-9.870129,6.975021,-4.335528,8.923403,-2.575616,9.999827,8.368190,-4.082271,-5.241007,-8.239252,-9.786723,-8.271270,7.644097,-6.539031,-5.561680,-4.158205,6.192914,7.044764,-2.640164,-0.268980,-6.239929,2.478035,0.044872,1.769904,-8.365311,7.685835,8.771879,2.255579,-0.367587,5.060143,6.686508,-1.080458,-7.103309,-4.880031,-6.119433,6.238487,-2.106004,-0.321007,-3.987114,1.433780,6.179977,1.059282,-0.139033,-8.658375,2.735477,6.635623,3.033002,9.441441,3.554685,-0.581850,-3.180839,-8.960672,-6.864402,-3.077220,-7.410469,9.741393,6.760873,0.647085,-3.440460,-3.331728,0.159332,-8.862223,2.683817,-5.243302,6.477499,9.333463,-3.433053,-1.243049,-7.508386,2.105387,-2.661894,2.674453,-6.135601,2.056022,-5.340408,-2.372191,-0.640720,-7.801831,-1.729647,-7.781621,7.243111,8.875028,-1.003561,-1.020323,-5.669132,4.847713,6.057129,-4.154999,3.826934,-7.913301,-7.070388,-4.047244,6.566181,-3.307322,0.181692,0.387896,-6.396614,2.510107,-3.167135,-0.964836,5.632955,1.757744,-6.555464,-8.677044,0.455557,3.033898,7.341042,-3.145690,-0.772998,-0.771396,-6.574826,-2.127578,-4.926161,8.752134,8.384509,-4.644433,7.314994,8.156791,4.584876,-7.984717,-4.839448,-8.893466,4.685824,-9.603713,-3.397569,3.535239,-3.560536,7.897927,2.471886,5.486289,-1.238676,-1.067909,1.396667,2.855482,2.127193,-0.494595,5.940057,-4.221028,2.345896,-7.393908,5.079664,-6.071630,0.980212,5.173403,-5.495516,-0.493786,-2.377320,7.440965,-9.228895,-6.914160,-2.225547,-7.693476,-2.112147,-8.720734,5.358799,7.307856,-9.709793,4.463402,1.411888,3.872397,1.257186,4.685162,-3.136327,-9.232338,4.937341,-9.769897,3.475962,-8.740094,2.110861,6.053381,9.008147,-0.919359,8.920502,-4.583066,-0.658239,-0.125152,2.390774,1.133026,-2.206493,-7.423302,-1.989545,7.798780,4.234703,2.879420,-0.290281,-0.193251,5.019905,-1.014571,-8.469656,-4.704103,-2.209038,-7.409807,-7.763590,1.256633,3.410744,-6.577857,-4.301222,4.115048,0.190058,7.322155,-6.682360,-4.036321,-6.230947,-5.814916,6.087970,1.483496,9.224020,8.566230,9.632105,6.995590,-3.406514,-7.692882,4.440496,0.473378,8.698946,1.926524,-7.129369,8.638192,-0.289013,4.107633,-3.721473,9.812420,-0.180181,4.058126,8.830583,6.769317,4.269822,-6.090169,0.322680,7.684282,-5.428314,-9.916070,-0.486979,0.182031,4.397481,-3.949475,8.634871,-9.593265,-0.266222,8.329164,-2.319680,-2.136673,4.128718,-6.937463,-8.551522,3.252370,-9.594827,3.930330,5.130921,8.407753,8.303767,7.513365,3.980974,-8.987657,9.157113,8.840280,-6.042655,5.585270,6.833557,1.917624,-8.685917,-1.276523,-7.957537,0.642808,4.326581,3.213577,2.583706,1.296487,-3.259085,-2.439513,-4.826628,9.611614,4.260345], dtype = "float32")#candidate|3620|(780,)|const|float32
call_3618 = relay.TupleGetItem(func_1711_call(relay.reshape(var_3619.astype('float32'), [60,]), relay.reshape(const_3620.astype('float32'), [780,]), ), 1)
call_3621 = relay.TupleGetItem(func_1714_call(relay.reshape(var_3619.astype('float32'), [60,]), relay.reshape(const_3620.astype('float32'), [780,]), ), 1)
func_934_call = mod.get_global_var('func_934')
func_939_call = mutated_mod.get_global_var('func_939')
var_3630 = relay.var("var_3630", dtype = "float32", shape = (110,))#candidate|3630|(110,)|var|float32
const_3631 = relay.const([3.448948,2.211591,1.373721,3.956164,7.013030,2.980645,0.879081,0.300397,-8.109907,3.192644,7.094204,3.020149,2.397217,5.537965,-5.210037,-0.932245,6.249989,2.729850,2.362590,-8.799420,-2.536951,-8.878273,-3.984187,4.017691,2.454382,8.093140,1.109347,1.695513,-1.466378,4.864746,-7.676089,-5.906199,7.067319,-0.685259,2.426612,1.152544,5.190805,2.669705,3.877123,-1.393441,6.370063,0.516394,9.788798,2.566795,-2.385000,-0.717669,5.737487,-9.424427,-5.883852,2.190511,6.796453,-1.751473,-2.683900,-5.574031,-1.431215,0.822516,-2.514727,-2.379550,4.285218,2.015615,-3.075508,-3.906138,6.499312,-4.503046,-9.433425,0.400511,-4.603770,-9.009252,8.233174,8.189236,5.229457,-2.304353,5.978853,5.965018,-1.671663,-9.413968,-9.282784,-3.878239,0.123157,-6.179315,4.488830,9.195704,6.967929,-8.560061,0.916993,-7.171745,-5.770464,-1.890318,-1.026258,0.071715,8.572404,9.760189,-8.364419,9.453277,2.721224,1.991736,7.435994,-1.504355,-8.021205,9.433778,-2.171078,8.326919,-8.464049,8.629847,7.177551,-4.452103,-4.406185,7.383955,4.697831,0.210951,-3.717684,9.756799,1.093771,-0.820564,9.550941,5.368779,9.986623,-5.453648,7.883063,-6.892490,-9.240369,9.108233,5.842561,-5.748485,3.033392,8.669314,-0.936350,4.086258,-0.661002,-3.380153,3.016205,-5.029102,-5.035151,4.899808,-0.109053,-0.627255,-0.896024,6.817940,-9.411067,-7.932426,2.119108,0.388520,1.275667,-8.210225,-2.129480,4.308928,-9.171613,6.158017,-7.135069,-6.563471,-7.549334,-7.621317,-0.827581,-4.290248,7.104853,-1.767949,-1.726174,-8.445423,-1.705909,-2.333992,7.025820,0.001726,1.028848,-6.034713,9.974414,-4.472162,-8.136692,-1.583454,-5.032340,9.731158,3.187979,3.326284,-0.632832,0.896372,-9.205519,7.119141,2.146410,8.055910,7.949866,8.205282,2.197663,1.951707,-0.282196,-2.363963,-2.954441,-2.562727,-0.050651,-4.176725,5.865747,6.253809,3.376900,-4.782041,9.772393,8.078182,2.262800,2.903730,-8.654723,-7.404989,-8.888087,1.611217,0.517658,2.946153,2.391192,-3.328704,4.516196,-1.024141,-1.631252,-7.442742,1.786814,1.383707,-3.833389,6.338987,7.294240,1.714722,-3.566091,-7.368521,-2.130331,3.014098,4.337386,-8.293325,-4.379746,5.501597,6.047520,-6.605276,3.801735,-8.059201,1.963857,-9.231979,7.537827,-2.944830,6.224400,7.015650,-6.644471,5.321687,-1.703841,9.558230,9.889298,-7.209011,-7.855531,5.834928,-2.924717,3.214635,5.230655,-7.098780,6.926717,-5.093982,1.882142,-7.744381,3.339410,-9.464597,2.869572,-3.924993,-3.134867,-6.272014,8.046404,1.058323,1.490941,1.334449,-9.933104,1.788164,5.758638,-5.335859,-0.711527,-3.296776,9.883281,9.818635,-3.279427,2.682139,-6.786800,7.200134,7.200105,1.478968,6.287171,5.120264,-5.542592,3.290075,-4.117133,5.420018,-9.459558,-7.763189,-2.303208,6.045129,3.390688,3.722043,-6.535874,-8.056144,3.227698,-7.797772,-6.967167,3.275817,-5.266009,-5.400076,-9.134200,0.999299,-1.800557,-4.420206,-1.162980,7.539606,-1.226743,9.900295,-9.109432,-0.162132,-5.868860,-4.681696,-0.520412,6.034545,2.582613,-1.550887,6.552757,-8.592368,3.730753,-3.605933,-6.681467,-0.926541,2.165145,7.452429,6.616146,9.632366,-9.380797,-6.421479,9.743612,-4.777816,-3.276884,1.525941,-5.364037,0.457404,-1.658141,6.553192,8.562891,-3.298970,-4.271234,-1.519622,9.341772,5.719206,-8.645763,7.744610,4.851719,-2.156040,-7.749566,-8.945519,1.077366,-4.246503,0.722257,4.105290,3.251190,6.619916,3.132612,9.416413,-4.245694,3.787357,0.888623,-0.620913,-3.157828,-8.167091,-6.678854,6.673802,-2.864525,-3.305329,-5.396181,-3.168339,9.399976,1.759202,-8.561379,5.412422,3.138506,3.498168,6.462023,1.568684,-6.935260,-7.146730,-8.801828,-8.345195,-5.554197,9.357928,-4.677151,-8.133038,0.690150,3.402084], dtype = "float32")#candidate|3631|(378,)|const|float32
call_3629 = relay.TupleGetItem(func_934_call(relay.reshape(var_3630.astype('float32'), [10, 11]), relay.reshape(var_3630.astype('float32'), [10, 11]), relay.reshape(const_3631.astype('float32'), [378,]), ), 2)
call_3632 = relay.TupleGetItem(func_939_call(relay.reshape(var_3630.astype('float32'), [10, 11]), relay.reshape(var_3630.astype('float32'), [10, 11]), relay.reshape(const_3631.astype('float32'), [378,]), ), 2)
output = relay.Tuple([uop_3592,call_3618,var_3619,const_3620,call_3629,var_3630,const_3631,])
output2 = relay.Tuple([uop_3594,call_3621,var_3619,const_3620,call_3632,var_3630,const_3631,])
func_3634 = relay.Function([var_3619,var_3630,], output)
mod['func_3634'] = func_3634
mod = relay.transform.InferType()(mod)
var_3635 = relay.var("var_3635", dtype = "float32", shape = (1, 60))#candidate|3635|(1, 60)|var|float32
var_3636 = relay.var("var_3636", dtype = "float32", shape = (110,))#candidate|3636|(110,)|var|float32
output = func_3634(var_3635,var_3636,)
func_3637 = relay.Function([var_3635,var_3636,], output)
mutated_mod['func_3637'] = func_3637
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_3659 = func_1953_call()
call_3660 = func_1953_call()
output = call_3659
output2 = call_3660
func_3665 = relay.Function([], output)
mod['func_3665'] = func_3665
mod = relay.transform.InferType()(mod)
output = func_3665()
func_3666 = relay.Function([], output)
mutated_mod['func_3666'] = func_3666
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3699 = relay.var("var_3699", dtype = "float32", shape = (15, 3, 2))#candidate|3699|(15, 3, 2)|var|float32
var_3700 = relay.var("var_3700", dtype = "float32", shape = (15, 3, 2))#candidate|3700|(15, 3, 2)|var|float32
bop_3701 = relay.floor_divide(var_3699.astype('float32'), relay.reshape(var_3700.astype('float32'), relay.shape_of(var_3699))) # shape=(15, 3, 2)
func_2748_call = mod.get_global_var('func_2748')
func_2750_call = mutated_mod.get_global_var('func_2750')
call_3706 = relay.TupleGetItem(func_2748_call(), 0)
call_3707 = relay.TupleGetItem(func_2750_call(), 0)
var_3710 = relay.var("var_3710", dtype = "float32", shape = (15, 3, 2))#candidate|3710|(15, 3, 2)|var|float32
bop_3711 = relay.maximum(bop_3701.astype('float64'), relay.reshape(var_3710.astype('float64'), relay.shape_of(bop_3701))) # shape=(15, 3, 2)
bop_3719 = relay.greater_equal(bop_3701.astype('bool'), relay.reshape(var_3699.astype('bool'), relay.shape_of(bop_3701))) # shape=(15, 3, 2)
bop_3725 = relay.equal(bop_3711.astype('bool'), relay.reshape(bop_3701.astype('bool'), relay.shape_of(bop_3711))) # shape=(15, 3, 2)
func_2642_call = mod.get_global_var('func_2642')
func_2647_call = mutated_mod.get_global_var('func_2647')
const_3747 = relay.const([-5.414881,-2.778156,-8.920869,3.879676,-2.410989,1.240803,5.499365,9.094305,9.452938,8.987714,0.748000,-2.402502,9.576769,-7.364488,0.972863,4.503000,7.684315,6.708924,-1.662170,-3.901433,7.375279,1.092939,2.182633,-5.165761,-0.427987,-8.501682,6.915106,6.557180,-2.811610,8.429214,2.088459,9.377919,6.348338,-8.694688,6.583306,-5.857355,-0.493233,5.531287,-6.819381,2.609010,3.786010,-9.066969,-1.274459,0.917603,-6.252714,-8.547119,-1.746780,-0.666092,-5.327246,6.028744,-7.734446,8.561279,6.262056,-6.259055,-7.196146,9.095586,9.626392,-7.391621,2.002936,-6.989658,8.624623,6.091393,-2.772860,6.767235,9.297872,3.471808,-0.061558,-3.177859,6.397512,8.125628,-1.671603,-6.334883,-2.662043,0.195337,-3.740269,4.938658,-7.857881,9.050568,5.863010,-9.944697,6.164384,-6.389702,-8.230729,-8.552732,5.584836,5.484820,-9.895813,4.886586,6.579407,-5.085696,1.344680,9.954208,6.066399,-9.411671,7.384867,8.474001,-3.524950,-4.495150,-8.775065,1.908883,-0.445359,-6.844126,-7.257568,8.574024,-6.829401,6.479708,-6.801034,-2.758816,-0.703353,-2.329054,3.507641,-1.323244,0.151713,-6.661343,2.180157,7.999481,-3.520410,-9.068387,7.091037,-9.254140,-0.980833,6.676551,-8.551827,-4.416214,4.285149,-9.818110,-7.549029,-7.905558,4.935218,6.296521,-1.275993,-8.519355,-4.655078,7.352443,2.763545,-2.059794,-4.175665,-1.016596,3.255041,2.832453,9.274675,2.238703,-4.887196,6.858795,-7.852543,3.888108,6.706156,1.380754,-1.894718,2.182234,6.881644,8.760211,-0.309235,-0.045561,-2.667248,-3.313796,-5.375880,5.427035,8.862433,-7.520943,-2.245971,4.919826,-6.224932,4.367254,5.430802,-5.790282,-8.157477,-5.238212,8.852433], dtype = "float64")#candidate|3747|(169,)|const|float64
var_3748 = relay.var("var_3748", dtype = "float64", shape = (12, 30))#candidate|3748|(12, 30)|var|float64
var_3749 = relay.var("var_3749", dtype = "float32", shape = (6, 63))#candidate|3749|(6, 63)|var|float32
call_3746 = relay.TupleGetItem(func_2642_call(relay.reshape(const_3747.astype('float64'), [1, 13, 13]), relay.reshape(var_3748.astype('float64'), [2, 180]), relay.reshape(var_3749.astype('float32'), [378, 1]), ), 2)
call_3750 = relay.TupleGetItem(func_2647_call(relay.reshape(const_3747.astype('float64'), [1, 13, 13]), relay.reshape(var_3748.astype('float64'), [2, 180]), relay.reshape(var_3749.astype('float32'), [378, 1]), ), 2)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_3752 = func_1953_call()
call_3753 = func_1953_call()
output = relay.Tuple([call_3706,bop_3719,bop_3725,call_3746,const_3747,var_3748,var_3749,call_3752,])
output2 = relay.Tuple([call_3707,bop_3719,bop_3725,call_3750,const_3747,var_3748,var_3749,call_3753,])
func_3759 = relay.Function([var_3699,var_3700,var_3710,var_3748,var_3749,], output)
mod['func_3759'] = func_3759
mod = relay.transform.InferType()(mod)
var_3760 = relay.var("var_3760", dtype = "float32", shape = (15, 3, 2))#candidate|3760|(15, 3, 2)|var|float32
var_3761 = relay.var("var_3761", dtype = "float32", shape = (15, 3, 2))#candidate|3761|(15, 3, 2)|var|float32
var_3762 = relay.var("var_3762", dtype = "float32", shape = (15, 3, 2))#candidate|3762|(15, 3, 2)|var|float32
var_3763 = relay.var("var_3763", dtype = "float64", shape = (12, 30))#candidate|3763|(12, 30)|var|float64
var_3764 = relay.var("var_3764", dtype = "float32", shape = (6, 63))#candidate|3764|(6, 63)|var|float32
output = func_3759(var_3760,var_3761,var_3762,var_3763,var_3764,)
func_3765 = relay.Function([var_3760,var_3761,var_3762,var_3763,var_3764,], output)
mutated_mod['func_3765'] = func_3765
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3780 = relay.var("var_3780", dtype = "float64", shape = (9, 3, 8))#candidate|3780|(9, 3, 8)|var|float64
var_3781 = relay.var("var_3781", dtype = "float64", shape = (9, 3, 8))#candidate|3781|(9, 3, 8)|var|float64
bop_3782 = relay.divide(var_3780.astype('float64'), relay.reshape(var_3781.astype('float64'), relay.shape_of(var_3780))) # shape=(9, 3, 8)
func_2894_call = mod.get_global_var('func_2894')
func_2896_call = mutated_mod.get_global_var('func_2896')
call_3787 = func_2894_call()
call_3788 = func_2894_call()
func_2642_call = mod.get_global_var('func_2642')
func_2647_call = mutated_mod.get_global_var('func_2647')
const_3815 = relay.const([4.424649,-4.832213,-1.880414,-2.980959,1.460791,4.106834,-9.121040,6.356126,-7.312711,8.456277,3.477648,-9.262781,0.711313,4.648216,-7.107978,8.632421,4.538439,-7.426835,6.739315,8.393054,2.964702,-0.848637,6.769075,3.025213,7.819280,3.213071,-0.362931,9.696650,9.420284,-7.401054,5.007896,-9.393058,3.049926,-9.299460,8.163046,7.106686,6.691738,-5.976027,2.336299,-8.749773,-6.715894,2.800080,7.602574,-6.084901,5.670298,-3.234536,0.511070,-2.079659,-4.507229,7.799326,-5.115430,-5.148623,6.518223,5.097715,-1.771847,7.184206,9.588024,9.374096,-7.285146,8.283358,-4.384107,-3.054484,-5.508782,-4.494068,8.118476,8.921306,-5.781895,9.187597,-4.591625,6.121309,2.163587,-5.014743,5.706003,3.767452,5.508632,-9.159276,4.589082,8.610640,4.539545,-1.570026,-7.728546,-9.690526,6.904708,-2.055799,-9.222640,4.539591,-2.735550,-9.998973,4.900311,1.348200,4.671064,9.022979,-4.948420,-4.223816,-7.533728,3.963416,-7.560163,-1.141748,2.667525,-6.625701,-4.963691,8.320816,3.054457,-1.455479,-3.942283,1.871999,1.655421,-9.617615,-3.586905,0.514308,6.839488,2.492495,3.165766,6.665569,-6.425484,-0.347964,7.759121,1.866997,-7.218697,-7.641724,-3.282192,-7.144164,-1.795896,6.333897,6.967316,5.477600,0.024421,5.894660,-9.973747,-4.484711,-6.265445,1.117391,-6.791695,-5.211198,7.413641,3.625628,-3.488430,6.345747,-5.173209,9.337613,-6.616829,-9.906216,9.393348,-9.067163,6.289877,-3.731939,9.396237,3.054848,6.031338,-7.306754,-8.961575,-0.437538,4.798088,3.154545,-2.201979,-9.849644,-7.032819,1.609965,-2.725121,4.655213,-4.563663,7.660402,-8.986343,2.276614,-2.085649,-0.733350,-1.367182,8.509814,-7.743368], dtype = "float64")#candidate|3815|(169,)|const|float64
const_3816 = relay.const([[6.778753,-5.844386,-1.036710,4.305505,-9.595673,7.267891,8.341574,-7.469361,-0.606433,3.352835,5.643613,6.978983,7.292804,-3.488784,-1.707250,-4.147273,-5.143579,7.160765,0.194105,2.463030,1.120744,-9.383517,-1.891659,-3.890279,7.133737,9.335445,-4.338732,-9.567856,4.432949,1.643411],[7.820467,4.298847,0.329027,-4.217211,2.892564,-5.667103,-8.208735,-9.745963,7.070557,-5.420799,-6.203765,6.290526,-7.977613,-8.395560,-8.534501,8.628676,-2.727783,-8.835604,6.578653,-9.239686,-8.408914,2.176867,2.845780,4.585093,-0.822894,8.487087,6.736172,2.058630,-8.387969,-9.475311],[-2.257872,-8.009943,-0.207574,-9.968092,-8.156362,2.998655,6.554100,0.626061,-5.386385,2.741332,-7.515990,8.445123,9.536283,8.592987,-3.287598,-4.161856,1.176758,9.472835,3.314201,1.424186,-9.083802,6.877257,2.539678,-2.236472,5.826325,-7.824607,-0.218486,-7.817177,5.809357,-9.247728],[9.675384,-1.870511,7.178944,-6.799930,7.014970,-3.433427,-8.476550,5.386140,2.709539,0.374452,5.030993,-6.179904,7.522168,6.811340,6.755184,0.433970,-9.872430,2.684339,5.340030,0.395664,-8.701210,2.487088,2.515939,-0.095175,-4.166125,2.988908,-7.719889,-9.467118,-3.492032,-0.812080],[6.348863,-0.471331,3.589935,9.107338,-5.096675,9.814893,8.810568,3.528179,8.020667,6.345164,-5.249680,9.223631,-0.893744,-3.352313,4.763381,-9.523465,-1.492696,-5.785845,9.003470,2.441457,-4.268779,-2.604827,6.278513,-1.319773,5.813021,-7.369628,6.182323,-8.945983,0.965860,-9.288876],[7.820246,5.481915,-1.763066,3.759865,-0.059083,6.716489,3.972676,-6.744230,4.415756,8.553975,-7.691829,-3.213857,9.440124,6.698213,-7.942574,3.957270,5.513771,5.109103,4.642511,1.176750,2.008200,-8.437001,-2.378150,8.077330,-6.061092,-5.308944,9.592634,-0.976008,-5.326634,6.355912],[7.374652,-7.298841,-8.722628,-8.322295,7.545544,4.201976,-9.624555,9.701782,-0.643193,8.619814,-1.355926,4.131828,-6.600112,-0.274947,5.008354,7.381505,6.096120,0.804645,-2.639785,0.141920,9.212020,-8.195963,-6.481870,-8.881382,3.465385,4.115166,-9.493383,8.501639,-8.388031,3.028862],[9.907235,9.487989,-7.769447,-2.111206,-5.676956,-1.428444,-9.856351,-3.070580,6.504258,7.225256,-0.105131,-1.970682,0.387208,1.860814,9.680113,-3.561397,-0.599123,4.751704,-9.972170,5.290803,7.123482,7.457817,-9.644397,4.096832,-3.828798,-1.320464,9.731316,7.295713,-8.930013,8.840058],[4.943686,3.180218,5.217140,5.596944,-2.872032,-2.522934,-8.686444,-5.625464,5.211079,-6.670922,-7.754599,-4.639900,7.803503,7.686563,3.732003,-5.603575,-3.792496,-5.369228,-2.088980,8.375287,-6.617191,6.650183,7.302739,0.205156,4.224092,6.175196,8.398176,7.640336,3.218000,-2.402841],[3.630050,-1.964782,-2.797224,9.454581,-8.186645,-2.201964,8.211220,8.403866,1.535818,5.239372,2.402439,-4.794397,3.984934,5.464761,-6.559845,2.502700,-7.427095,3.249665,-7.483919,-4.956474,-1.284901,1.069645,-6.933563,-4.536537,4.265084,-4.913971,-9.271380,2.798722,-0.320805,5.303276],[3.040094,2.917461,-1.420905,-1.479285,8.440505,7.806723,0.273613,7.493041,0.762120,5.872625,-7.605900,2.678186,6.579320,-2.002269,-7.586368,-6.153488,-0.656380,1.917506,4.638977,9.749283,-2.253901,-8.993383,0.757076,1.825287,7.675272,5.937652,-1.088278,4.639020,-4.420671,-5.044273],[9.064938,-4.247269,-1.619945,-6.369480,9.371888,-0.307801,-0.581402,-7.023265,5.255275,-7.440195,-2.769840,-6.632654,0.805591,-5.531720,2.440451,-8.877201,-2.203663,-7.877599,-8.022469,-6.664697,-6.736419,-2.722876,2.729550,-0.834841,-6.139973,2.088631,-5.747180,-8.956730,6.014019,5.247242]], dtype = "float64")#candidate|3816|(12, 30)|const|float64
const_3817 = relay.const([-4.232928,-4.172192,-1.638020,4.760015,4.543734,7.004483,-3.378548,5.322602,9.913333,0.758166,-9.698159,1.790363,3.643706,-4.315833,-9.068004,-0.722932,1.071150,4.201899,2.528201,-1.285502,-6.685399,7.118707,3.028143,-6.413232,1.613313,2.268155,8.941367,-2.680863,3.409769,3.610975,-8.175663,6.567515,-3.733786,-1.050970,9.105339,-9.804497,-2.556833,-9.230536,1.862706,-3.919051,-0.480618,-6.419835,-7.884020,-4.169237,-5.930048,4.920635,0.256518,-6.195120,-2.655553,0.923582,-8.261470,2.145215,5.661120,-0.686035,-6.469185,-6.890881,-1.669728,-4.172965,3.124028,1.214950,-5.754731,-4.745295,9.575690,0.840252,9.253304,5.826903,3.281753,2.525839,-9.773298,3.120417,-0.295442,5.756092,-1.605118,-2.989560,8.219346,5.597770,-9.994085,7.468281,5.945650,7.518420,-3.189565,-6.070040,-7.794745,-0.147691,5.775848,2.821686,7.712068,3.733158,9.999165,8.830677,4.955957,-6.551824,1.239497,0.140452,5.804175,5.548463,5.839564,1.304027,9.523669,-4.816235,8.300186,7.593690,-4.218443,-9.026717,-5.092735,-5.878872,4.147983,8.918926,4.362553,7.283587,-0.538986,-9.730314,-0.499150,7.659697,6.872344,-5.273337,5.762112,-0.911677,-7.954225,1.629776,-9.446393,-3.502577,1.333469,-9.891004,-9.958571,8.040413,6.136136,-8.176536,5.382747,3.620333,2.675913,1.807306,4.449901,9.135823,-2.759852,-9.440786,-4.222494,-7.965830,8.618028,-2.616297,-8.403473,-6.387803,5.422909,8.306263,-8.012588,-7.234073,7.870542,-9.344684,5.316590,-1.501448,8.680422,-6.479707,-7.880544,5.437078,-6.045191,3.646208,9.554568,-1.015291,6.567307,0.968288,-9.478321,9.460638,5.773504,9.209935,7.359950,-5.791302,-6.358965,5.895657,-4.374796,-2.043248,-7.288003,5.488281,9.261335,7.568790,-8.955857,-7.878025,-8.578698,-0.989854,-9.063092,-8.592872,6.015709,-2.762414,0.598060,-5.674031,-4.390159,-3.771570,8.008611,-9.328104,9.564522,8.906653,-6.511994,3.440984,0.464496,-9.688284,7.484783,-4.566449,4.455631,-3.213878,-5.009470,-5.383700,-2.372677,2.941025,8.888032,-1.295319,9.536066,-5.496643,-7.501391,-1.003863,3.839643,0.743391,3.275518,6.155058,-1.732776,-6.435425,8.465576,9.770531,1.900227,9.863079,-7.371570,-2.007233,6.001510,2.632210,-7.920490,9.325156,-6.577636,-0.613601,-1.328513,-7.640331,6.675778,4.567968,-7.234931,-2.730828,-3.697919,-1.462732,-2.608849,-1.041412,6.279346,4.012835,-8.579672,6.895683,3.215345,-0.979731,1.608184,-0.897128,5.686785,4.164191,-9.008883,-3.565124,8.215315,9.917730,0.364471,1.841022,2.075589,-7.436160,-5.784061,-6.296694,6.548723,1.396509,3.064385,6.927729,-3.504287,-7.099504,8.491335,9.550295,3.651473,-4.399306,-7.053916,0.303351,-5.655519,5.368290,7.363019,-6.751806,6.250271,-0.355610,3.202593,4.372339,-5.040104,-7.017571,-4.443552,5.863675,-2.384360,-1.632300,4.060304,-9.269980,4.471331,-5.112160,0.217162,8.636499,1.589109,2.034401,-8.762111,-3.505174,8.734721,-9.101502,-6.357734,0.423287,3.615116,3.418762,-8.301262,-2.269282,-6.266270,-8.648053,1.371826,-3.116944,8.472321,7.641530,-6.642766,4.705309,-3.678175,-6.479369,5.367090,2.181795,0.788888,5.186313,-4.050822,-4.464804,5.161210,-0.156642,-5.968364,3.333521,-7.888096,9.604742,-1.433632,-8.071109,-2.671581,1.710820,7.839959,-3.195995,6.299434,-9.062029,4.913974,-5.386916,9.069728,1.702382,9.547617,-7.467333,5.475262,2.002718,-5.095442,-1.867126,-8.707384,-6.702948,-1.645752,-1.249270,-0.643020,7.056487,0.835624,-3.748588,-4.328345,5.611374,-0.376640,-7.838834,6.766958,3.608240,5.927284,1.333803,2.186110,3.417144,-6.743632,-1.719785,-7.986442,5.180291,0.326417,5.275185,0.283684,-5.584494,-0.805107,-6.711047,-5.308664,0.132810,3.811114,3.503173,-9.594250,8.113166,-9.201000,0.667604,4.790398,4.200413], dtype = "float32")#candidate|3817|(378,)|const|float32
call_3814 = relay.TupleGetItem(func_2642_call(relay.reshape(const_3815.astype('float64'), [1, 13, 13]), relay.reshape(const_3816.astype('float64'), [2, 180]), relay.reshape(const_3817.astype('float32'), [378, 1]), ), 3)
call_3818 = relay.TupleGetItem(func_2647_call(relay.reshape(const_3815.astype('float64'), [1, 13, 13]), relay.reshape(const_3816.astype('float64'), [2, 180]), relay.reshape(const_3817.astype('float32'), [378, 1]), ), 3)
output = relay.Tuple([bop_3782,call_3787,call_3814,const_3815,const_3816,const_3817,])
output2 = relay.Tuple([bop_3782,call_3788,call_3818,const_3815,const_3816,const_3817,])
func_3829 = relay.Function([var_3780,var_3781,], output)
mod['func_3829'] = func_3829
mod = relay.transform.InferType()(mod)
mutated_mod['func_3829'] = func_3829
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3829_call = mutated_mod.get_global_var('func_3829')
var_3831 = relay.var("var_3831", dtype = "float64", shape = (9, 3, 8))#candidate|3831|(9, 3, 8)|var|float64
var_3832 = relay.var("var_3832", dtype = "float64", shape = (9, 3, 8))#candidate|3832|(9, 3, 8)|var|float64
call_3830 = func_3829_call(var_3831,var_3832,)
output = call_3830
func_3833 = relay.Function([var_3831,var_3832,], output)
mutated_mod['func_3833'] = func_3833
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1953_call = mod.get_global_var('func_1953')
func_1955_call = mutated_mod.get_global_var('func_1955')
call_3846 = func_1953_call()
call_3847 = func_1953_call()
output = relay.Tuple([call_3846,])
output2 = relay.Tuple([call_3847,])
func_3861 = relay.Function([], output)
mod['func_3861'] = func_3861
mod = relay.transform.InferType()(mod)
mutated_mod['func_3861'] = func_3861
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3861_call = mutated_mod.get_global_var('func_3861')
call_3862 = func_3861_call()
output = call_3862
func_3863 = relay.Function([], output)
mutated_mod['func_3863'] = func_3863
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3939 = relay.var("var_3939", dtype = "float64", shape = (1, 10, 2))#candidate|3939|(1, 10, 2)|var|float64
var_3940 = relay.var("var_3940", dtype = "float64", shape = (7, 10, 2))#candidate|3940|(7, 10, 2)|var|float64
bop_3941 = relay.power(var_3939.astype('float64'), var_3940.astype('float64')) # shape=(7, 10, 2)
bop_3946 = relay.bitwise_and(bop_3941.astype('uint32'), var_3939.astype('uint32')) # shape=(7, 10, 2)
uop_3955 = relay.asinh(var_3940.astype('float32')) # shape=(7, 10, 2)
bop_3961 = relay.multiply(bop_3941.astype('float32'), relay.reshape(uop_3955.astype('float32'), relay.shape_of(bop_3941))) # shape=(7, 10, 2)
output = relay.Tuple([bop_3946,bop_3961,])
output2 = relay.Tuple([bop_3946,bop_3961,])
func_3971 = relay.Function([var_3939,var_3940,], output)
mod['func_3971'] = func_3971
mod = relay.transform.InferType()(mod)
mutated_mod['func_3971'] = func_3971
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3971_call = mutated_mod.get_global_var('func_3971')
var_3973 = relay.var("var_3973", dtype = "float64", shape = (1, 10, 2))#candidate|3973|(1, 10, 2)|var|float64
var_3974 = relay.var("var_3974", dtype = "float64", shape = (7, 10, 2))#candidate|3974|(7, 10, 2)|var|float64
call_3972 = func_3971_call(var_3973,var_3974,)
output = call_3972
func_3975 = relay.Function([var_3973,var_3974,], output)
mutated_mod['func_3975'] = func_3975
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3047_call = mod.get_global_var('func_3047')
func_3049_call = mutated_mod.get_global_var('func_3049')
call_4037 = relay.TupleGetItem(func_3047_call(), 0)
call_4038 = relay.TupleGetItem(func_3049_call(), 0)
output = relay.Tuple([call_4037,])
output2 = relay.Tuple([call_4038,])
func_4045 = relay.Function([], output)
mod['func_4045'] = func_4045
mod = relay.transform.InferType()(mod)
output = func_4045()
func_4046 = relay.Function([], output)
mutated_mod['func_4046'] = func_4046
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2582_call = mod.get_global_var('func_2582')
func_2583_call = mutated_mod.get_global_var('func_2583')
call_4103 = relay.TupleGetItem(func_2582_call(), 0)
call_4104 = relay.TupleGetItem(func_2583_call(), 0)
output = call_4103
output2 = call_4104
func_4108 = relay.Function([], output)
mod['func_4108'] = func_4108
mod = relay.transform.InferType()(mod)
output = func_4108()
func_4109 = relay.Function([], output)
mutated_mod['func_4109'] = func_4109
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4108_call = mod.get_global_var('func_4108')
func_4109_call = mutated_mod.get_global_var('func_4109')
call_4228 = func_4108_call()
call_4229 = func_4108_call()
output = relay.Tuple([call_4228,])
output2 = relay.Tuple([call_4229,])
func_4239 = relay.Function([], output)
mod['func_4239'] = func_4239
mod = relay.transform.InferType()(mod)
mutated_mod['func_4239'] = func_4239
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4239_call = mutated_mod.get_global_var('func_4239')
call_4240 = func_4239_call()
output = call_4240
func_4241 = relay.Function([], output)
mutated_mod['func_4241'] = func_4241
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2748_call = mod.get_global_var('func_2748')
func_2750_call = mutated_mod.get_global_var('func_2750')
call_4245 = relay.TupleGetItem(func_2748_call(), 0)
call_4246 = relay.TupleGetItem(func_2750_call(), 0)
func_2382_call = mod.get_global_var('func_2382')
func_2384_call = mutated_mod.get_global_var('func_2384')
const_4258 = relay.const([[2,-3,10,8,6,7,-9,-4,-3,1,8,-7],[10,-7,-3,-1,4,1,-3,-7,4,-5,-10,-1],[-3,1,6,2,-3,2,-3,-5,6,-2,-5,-6],[-5,-4,-3,-6,-7,-8,4,-10,-1,6,-7,1],[-7,4,2,-2,2,9,-3,2,3,-5,-3,5],[9,-6,-2,-2,9,-10,-6,5,-10,-5,-1,-3],[10,1,-2,5,-8,10,9,-7,-9,-1,-9,5],[3,2,-1,-6,2,-4,4,8,2,9,1,2],[-5,7,9,5,3,6,2,-5,-9,-1,-8,-4],[8,-4,1,8,-10,10,-8,3,-2,-5,1,3],[9,-5,-2,9,-4,8,10,-1,-3,-2,-8,2],[4,-6,-9,6,-2,5,10,3,-9,-9,-9,1],[-3,-2,-10,1,-4,-10,8,3,5,-4,-3,6],[10,-1,6,-2,7,-8,-2,2,7,-1,5,7],[-1,2,-9,-6,-6,9,-10,-8,7,-5,-6,5],[6,10,-9,1,7,-3,-6,7,-9,1,-1,8],[9,-7,2,-10,-9,10,-9,-7,8,5,3,3],[2,3,5,1,7,-10,-1,7,-3,9,2,8],[-8,-2,-7,-2,1,2,7,1,-3,-2,1,6],[8,4,1,-8,7,9,6,-2,-5,6,-10,-5],[-8,-9,6,-5,-5,4,2,-5,1,2,-7,6],[-3,-3,9,-7,-7,-10,-6,-7,3,-6,-3,-6],[7,4,4,-6,-4,9,-9,2,5,-10,6,-9],[-1,6,6,-10,1,8,6,4,5,3,2,-1]], dtype = "int16")#candidate|4258|(24, 12)|const|int16
call_4257 = relay.TupleGetItem(func_2382_call(relay.reshape(const_4258.astype('int16'), [288,])), 2)
call_4259 = relay.TupleGetItem(func_2384_call(relay.reshape(const_4258.astype('int16'), [288,])), 2)
output = relay.Tuple([call_4245,call_4257,const_4258,])
output2 = relay.Tuple([call_4246,call_4259,const_4258,])
func_4260 = relay.Function([], output)
mod['func_4260'] = func_4260
mod = relay.transform.InferType()(mod)
output = func_4260()
func_4261 = relay.Function([], output)
mutated_mod['func_4261'] = func_4261
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2990_call = mod.get_global_var('func_2990')
func_2992_call = mutated_mod.get_global_var('func_2992')
call_4262 = relay.TupleGetItem(func_2990_call(), 0)
call_4263 = relay.TupleGetItem(func_2992_call(), 0)
func_2311_call = mod.get_global_var('func_2311')
func_2313_call = mutated_mod.get_global_var('func_2313')
call_4271 = relay.TupleGetItem(func_2311_call(), 3)
call_4272 = relay.TupleGetItem(func_2313_call(), 3)
bop_4277 = relay.right_shift(call_4262.astype('uint32'), call_4271.astype('uint32')) # shape=(2, 780)
bop_4280 = relay.right_shift(call_4263.astype('uint32'), call_4272.astype('uint32')) # shape=(2, 780)
output = bop_4277
output2 = bop_4280
func_4281 = relay.Function([], output)
mod['func_4281'] = func_4281
mod = relay.transform.InferType()(mod)
output = func_4281()
func_4282 = relay.Function([], output)
mutated_mod['func_4282'] = func_4282
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3156_call = mod.get_global_var('func_3156')
func_3158_call = mutated_mod.get_global_var('func_3158')
call_4314 = func_3156_call()
call_4315 = func_3156_call()
output = relay.Tuple([call_4314,])
output2 = relay.Tuple([call_4315,])
func_4317 = relay.Function([], output)
mod['func_4317'] = func_4317
mod = relay.transform.InferType()(mod)
mutated_mod['func_4317'] = func_4317
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4317_call = mutated_mod.get_global_var('func_4317')
call_4318 = func_4317_call()
output = call_4318
func_4319 = relay.Function([], output)
mutated_mod['func_4319'] = func_4319
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3436_call = mod.get_global_var('func_3436')
func_3438_call = mutated_mod.get_global_var('func_3438')
call_4320 = func_3436_call()
call_4321 = func_3436_call()
func_4239_call = mod.get_global_var('func_4239')
func_4241_call = mutated_mod.get_global_var('func_4241')
call_4323 = relay.TupleGetItem(func_4239_call(), 0)
call_4324 = relay.TupleGetItem(func_4241_call(), 0)
uop_4345 = relay.asinh(call_4320.astype('float64')) # shape=(2, 1)
uop_4347 = relay.asinh(call_4321.astype('float64')) # shape=(2, 1)
output = relay.Tuple([call_4323,uop_4345,])
output2 = relay.Tuple([call_4324,uop_4347,])
func_4354 = relay.Function([], output)
mod['func_4354'] = func_4354
mod = relay.transform.InferType()(mod)
mutated_mod['func_4354'] = func_4354
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4354_call = mutated_mod.get_global_var('func_4354')
call_4355 = func_4354_call()
output = call_4355
func_4356 = relay.Function([], output)
mutated_mod['func_4356'] = func_4356
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4239_call = mod.get_global_var('func_4239')
func_4241_call = mutated_mod.get_global_var('func_4241')
call_4360 = relay.TupleGetItem(func_4239_call(), 0)
call_4361 = relay.TupleGetItem(func_4241_call(), 0)
output = call_4360
output2 = call_4361
func_4371 = relay.Function([], output)
mod['func_4371'] = func_4371
mod = relay.transform.InferType()(mod)
mutated_mod['func_4371'] = func_4371
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4371_call = mutated_mod.get_global_var('func_4371')
call_4372 = func_4371_call()
output = call_4372
func_4373 = relay.Function([], output)
mutated_mod['func_4373'] = func_4373
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4387 = relay.var("var_4387", dtype = "float32", shape = ())#candidate|4387|()|var|float32
var_4388 = relay.var("var_4388", dtype = "float32", shape = (11, 10, 10))#candidate|4388|(11, 10, 10)|var|float32
bop_4389 = relay.divide(var_4387.astype('float32'), var_4388.astype('float32')) # shape=(11, 10, 10)
uop_4393 = relay.log(bop_4389.astype('float64')) # shape=(11, 10, 10)
bop_4406 = relay.maximum(uop_4393.astype('int64'), relay.reshape(bop_4389.astype('int64'), relay.shape_of(uop_4393))) # shape=(11, 10, 10)
func_2968_call = mod.get_global_var('func_2968')
func_2973_call = mutated_mod.get_global_var('func_2973')
const_4410 = relay.const([3.016117,9.493680,-0.195684,-4.423911,1.526310,-8.278087], dtype = "float32")#candidate|4410|(6,)|const|float32
var_4411 = relay.var("var_4411", dtype = "float32", shape = (378,))#candidate|4411|(378,)|var|float32
var_4412 = relay.var("var_4412", dtype = "float32", shape = (504,))#candidate|4412|(504,)|var|float32
var_4413 = relay.var("var_4413", dtype = "int16", shape = (288,))#candidate|4413|(288,)|var|int16
call_4409 = relay.TupleGetItem(func_2968_call(relay.reshape(const_4410.astype('float32'), [6,]), relay.reshape(var_4411.astype('float32'), [378,]), relay.reshape(var_4412.astype('float32'), [2, 252]), relay.reshape(var_4413.astype('int16'), [288,]), ), 2)
call_4414 = relay.TupleGetItem(func_2973_call(relay.reshape(const_4410.astype('float32'), [6,]), relay.reshape(var_4411.astype('float32'), [378,]), relay.reshape(var_4412.astype('float32'), [2, 252]), relay.reshape(var_4413.astype('int16'), [288,]), ), 2)
func_1881_call = mod.get_global_var('func_1881')
func_1885_call = mutated_mod.get_global_var('func_1885')
call_4424 = relay.TupleGetItem(func_1881_call(relay.reshape(const_4410.astype('float32'), [1, 1, 6]), relay.reshape(call_4409.astype('float32'), [1, 378]), ), 2)
call_4425 = relay.TupleGetItem(func_1885_call(relay.reshape(const_4410.astype('float32'), [1, 1, 6]), relay.reshape(call_4409.astype('float32'), [1, 378]), ), 2)
bop_4426 = relay.add(bop_4406.astype('uint64'), relay.reshape(var_4388.astype('uint64'), relay.shape_of(bop_4406))) # shape=(11, 10, 10)
uop_4429 = relay.sigmoid(uop_4393.astype('float32')) # shape=(11, 10, 10)
func_3861_call = mod.get_global_var('func_3861')
func_3863_call = mutated_mod.get_global_var('func_3863')
call_4433 = relay.TupleGetItem(func_3861_call(), 0)
call_4434 = relay.TupleGetItem(func_3863_call(), 0)
output = relay.Tuple([call_4409,const_4410,var_4411,var_4412,var_4413,call_4424,bop_4426,uop_4429,call_4433,])
output2 = relay.Tuple([call_4414,const_4410,var_4411,var_4412,var_4413,call_4425,bop_4426,uop_4429,call_4434,])
func_4435 = relay.Function([var_4387,var_4388,var_4411,var_4412,var_4413,], output)
mod['func_4435'] = func_4435
mod = relay.transform.InferType()(mod)
mutated_mod['func_4435'] = func_4435
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4435_call = mutated_mod.get_global_var('func_4435')
var_4437 = relay.var("var_4437", dtype = "float32", shape = ())#candidate|4437|()|var|float32
var_4438 = relay.var("var_4438", dtype = "float32", shape = (11, 10, 10))#candidate|4438|(11, 10, 10)|var|float32
var_4439 = relay.var("var_4439", dtype = "float32", shape = (378,))#candidate|4439|(378,)|var|float32
var_4440 = relay.var("var_4440", dtype = "float32", shape = (504,))#candidate|4440|(504,)|var|float32
var_4441 = relay.var("var_4441", dtype = "int16", shape = (288,))#candidate|4441|(288,)|var|int16
call_4436 = func_4435_call(var_4437,var_4438,var_4439,var_4440,var_4441,)
output = call_4436
func_4442 = relay.Function([var_4437,var_4438,var_4439,var_4440,var_4441,], output)
mutated_mod['func_4442'] = func_4442
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3156_call = mod.get_global_var('func_3156')
func_3158_call = mutated_mod.get_global_var('func_3158')
call_4462 = func_3156_call()
call_4463 = func_3156_call()
output = relay.Tuple([call_4462,])
output2 = relay.Tuple([call_4463,])
func_4500 = relay.Function([], output)
mod['func_4500'] = func_4500
mod = relay.transform.InferType()(mod)
mutated_mod['func_4500'] = func_4500
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4500_call = mutated_mod.get_global_var('func_4500')
call_4501 = func_4500_call()
output = call_4501
func_4502 = relay.Function([], output)
mutated_mod['func_4502'] = func_4502
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4508 = relay.var("var_4508", dtype = "uint8", shape = (15, 3, 13))#candidate|4508|(15, 3, 13)|var|uint8
const_4509 = relay.const([[[-9,8,-8,8,3,-1,3,-8,3,5,-2,3,-7],[7,-2,-5,-7,9,-3,-7,5,1,-6,-1,2,-10],[-8,-2,-9,-9,1,-2,-7,6,-7,8,10,2,-6]],[[-4,-10,5,-7,-2,-7,9,-7,6,1,-5,4,-2],[3,-10,-6,1,1,3,2,-6,8,-7,-6,8,5],[-8,9,5,4,4,2,-8,-9,2,-8,-2,-6,1]],[[4,-1,-7,9,-9,-9,-10,4,6,4,5,-1,-6],[1,-2,6,-6,-4,-4,-8,-7,-7,-9,10,5,5],[2,4,8,2,2,5,-2,10,4,-10,-4,-10,8]],[[-2,-4,4,2,-5,-5,-2,2,-4,8,2,-1,-5],[-1,-10,6,3,-1,-7,-5,3,7,7,-3,-6,5],[7,4,3,-8,4,4,-4,-2,-8,-3,-7,8,-2]],[[6,-4,-9,7,2,7,-2,3,-8,-3,-10,-7,3],[3,-9,2,4,6,-6,5,2,9,-4,-2,-1,-3],[5,4,-6,-9,4,10,-9,1,1,5,-9,9,-9]],[[8,-7,3,-2,2,-3,-5,-4,5,3,-1,4,-10],[-2,-8,2,-1,6,-5,-10,-8,-9,10,-8,-9,10],[7,-5,9,-8,8,-1,-2,-2,7,3,-5,-6,6]],[[-9,-9,4,-5,5,-8,-10,-4,5,10,-1,1,10],[-4,10,-2,-6,2,-7,3,-10,4,7,-1,-1,2],[-5,6,9,-4,2,10,9,2,-8,7,-5,-5,8]],[[-5,-6,1,-9,9,-6,7,-10,-10,8,4,-6,-1],[3,-3,2,10,-6,9,9,-5,10,-5,-10,9,6],[3,4,4,2,10,7,1,-10,-3,10,-4,-2,1]],[[5,-10,-3,4,9,-7,-7,-4,-7,-2,-9,8,2],[10,-2,-6,10,9,-10,-1,2,1,-4,-6,4,-10],[2,10,9,-8,10,-9,1,-2,-2,4,-10,5,-10]],[[7,-10,6,-4,-6,6,-5,9,7,-1,-2,3,-4],[1,-8,-2,-6,3,-2,8,-7,3,3,5,3,10],[-6,-3,7,-2,6,2,-6,-4,2,-8,10,-3,5]],[[-4,2,4,-8,-10,1,-2,8,9,-10,4,-6,-1],[6,-9,1,-7,-9,3,-8,-4,7,-5,1,10,-10],[-3,7,-5,7,-1,-9,-1,1,-8,7,7,-10,-2]],[[8,-4,6,-5,3,-5,-3,-9,7,3,-4,-7,4],[8,6,-5,-1,3,8,8,9,2,-2,-4,-2,6],[-7,8,9,-5,7,-9,1,-6,3,10,9,-2,2]],[[-9,5,10,-10,-5,-8,4,-2,-6,9,4,5,-2],[6,5,-1,4,-4,10,-7,-5,-5,2,3,10,7],[-4,6,9,3,3,10,2,2,-10,-6,5,5,7]],[[-3,-2,9,-5,5,10,-7,1,-8,3,-4,-2,7],[10,-3,2,3,8,-3,10,4,3,1,-1,9,-3],[-7,5,-10,-8,3,-3,-5,10,-4,3,10,-1,3]],[[2,-6,-7,7,8,7,2,-5,6,-6,7,7,-9],[-9,-8,2,6,-4,-10,-5,8,-4,1,5,-1,-4],[-10,2,9,-10,6,8,2,7,3,-2,-3,1,-6]]], dtype = "uint8")#candidate|4509|(15, 3, 13)|const|uint8
bop_4510 = relay.right_shift(var_4508.astype('uint8'), relay.reshape(const_4509.astype('uint8'), relay.shape_of(var_4508))) # shape=(15, 3, 13)
output = relay.Tuple([bop_4510,])
output2 = relay.Tuple([bop_4510,])
func_4517 = relay.Function([var_4508,], output)
mod['func_4517'] = func_4517
mod = relay.transform.InferType()(mod)
var_4518 = relay.var("var_4518", dtype = "uint8", shape = (15, 3, 13))#candidate|4518|(15, 3, 13)|var|uint8
output = func_4517(var_4518)
func_4519 = relay.Function([var_4518], output)
mutated_mod['func_4519'] = func_4519
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1978_call = mod.get_global_var('func_1978')
func_1979_call = mutated_mod.get_global_var('func_1979')
call_4540 = relay.TupleGetItem(func_1978_call(), 0)
call_4541 = relay.TupleGetItem(func_1979_call(), 0)
func_842_call = mod.get_global_var('func_842')
func_846_call = mutated_mod.get_global_var('func_846')
var_4557 = relay.var("var_4557", dtype = "float32", shape = (231,))#candidate|4557|(231,)|var|float32
const_4558 = relay.const([-5.803702,-0.677067,7.850063,9.732419,-5.363781,-7.289736,4.374339,-7.255714,-8.670315,2.131852,-5.800392,-3.301176,4.994752,5.862740,4.746220,2.580799,-0.869010,0.499582,-6.677833,3.763599,2.406799,-2.474877,-4.909341,-7.417791,5.544392,4.250143,-3.026210,-8.466874,3.790400,-2.189795,4.202567,-8.858276,1.020434,5.247179,9.422547,-6.667701,-7.453196,0.050198,-4.886728,5.725871,6.385893,5.451464,-0.352778,-7.351641,-0.390566,3.369056,-8.632846,4.282093,4.123788,1.801833,9.994071,-4.598468,-4.219417,3.467994,5.534503,-6.043946,3.132884,-4.615485,-8.501120,4.831789,8.385723,-9.222363,0.799009,2.422768,-3.072604,8.151807,9.030758,-0.462252,7.424832,-8.886683,-6.442388,2.056964,-9.790805,-8.326702,-9.822572,2.516466,-2.698397,-6.928135,4.690278,-1.613418,-7.050294,-4.404603,8.002604,-7.192500,-0.784511,-1.774594,0.232248,-3.287512,3.724544,-6.900566,-0.453531,-5.759425,0.452513,-1.135878,-5.244144,3.078890,-0.558713,-0.008592,-0.923281,-9.117932,9.806579,1.198462,4.630727,7.334086,-0.341898,-2.080521,4.298058,6.705738,3.056651,-1.691829,1.202983,2.416300,5.457633,2.056392,-8.851214,-3.219074,-1.778220,-1.695004,7.844129,5.748498,-6.004473,-0.840558,-3.771645,-4.611788,-5.814045,0.442498,-7.039516,9.737012,4.793174,8.575807,1.079055,0.720752,-9.964833,2.199056,2.580895,8.127835,-0.415166,-6.866948,-9.954147,0.042731], dtype = "float64")#candidate|4558|(140,)|const|float64
call_4556 = relay.TupleGetItem(func_842_call(relay.reshape(var_4557.astype('float32'), [7, 11, 3]), relay.reshape(const_4558.astype('float64'), [70, 2]), ), 2)
call_4559 = relay.TupleGetItem(func_846_call(relay.reshape(var_4557.astype('float32'), [7, 11, 3]), relay.reshape(const_4558.astype('float64'), [70, 2]), ), 2)
uop_4562 = relay.acosh(var_4557.astype('float32')) # shape=(231,)
bop_4564 = relay.floor_divide(uop_4562.astype('float32'), call_4540.astype('float32')) # shape=(2, 231)
bop_4567 = relay.floor_divide(uop_4562.astype('float32'), call_4541.astype('float32')) # shape=(2, 231)
bop_4569 = relay.bitwise_and(uop_4562.astype('int16'), relay.reshape(var_4557.astype('int16'), relay.shape_of(uop_4562))) # shape=(231,)
uop_4573 = relay.cos(var_4557.astype('float32')) # shape=(231,)
const_4581 = relay.const([[-2.051785,2.903005,8.396023,0.689472,2.586769,-1.907213,-0.584874,1.204833,1.491941,-0.187765,2.831031,3.221447,6.557823,-6.028707,2.478269,8.234864,6.361835,2.977327,-6.401004,6.188147,7.501794,2.628313,0.474717,5.667734,-3.323678,9.702199,-7.417766,4.016797,-6.003790,-9.556762,3.297840,5.917133,-6.223394,5.837572,-2.858414,-0.086392,2.007568,6.380378,-6.012490,9.091968,-2.066313,7.294979,1.646237,-8.013716,-3.701100,-6.164821,-5.222824,4.004277,-7.409325,0.659954,8.298312,9.516266,1.495566,2.328942,3.197361,-5.822256,1.040956,4.618551,6.093005,-8.389006,-3.979282,3.811829,0.710832,7.567304,-8.204260,-3.274037,-2.286754,-8.525813,-1.650579,9.450466,-3.093939,9.059070,9.875540,7.072620,2.624536,3.572634,-3.544823,-0.443072,4.454564,0.581429,9.224903,7.982888,8.568968,5.583545,-2.086677,-6.825252,8.426005,-2.780372,-3.036103,0.989719,-3.948012,9.239474,6.974561,-8.933857,9.512652,-6.232450,4.327179,-5.547636,7.693478,-4.148511,-4.126357,4.728586,-4.623957,6.443740,3.907294,-5.324361,8.574875,-7.098391,-1.236531,1.239235,-1.907512,-0.042468,6.186370,6.470562,-8.621016,-3.316260,-1.300371,-0.044407,-5.544049,7.418900,-2.526246,7.823346,7.975584,5.669254,-8.178187,6.100837,-1.385764,-8.056252,-0.159929,7.239621,7.084584,3.541715,2.214936,-6.684253,2.468686,1.589157,-9.104227,0.134439,-3.963436,9.925920,5.701065,2.140852,-4.963096,-6.482454,7.167234,4.802705,8.373320,-2.022961,1.255604,-3.590713,-8.740841,-2.518410,-1.482103,-8.383580,-1.900369,4.480413,-1.000240,-7.148758,-4.165363,2.396542,-3.940200,-7.431719,-9.287622,-5.741494,-7.744280,-3.740285,-1.615005,0.457965,7.634294,9.288548,3.804900,5.611282,4.162537,-6.961829,-9.017930,-8.208812,-6.784002,7.232870,9.675722,-7.568708,-7.097804,6.555737,-5.070859,-7.483303,-0.812388,9.756363,-4.828775,-2.375873,-2.176389,-6.382445,-3.972409,1.328626,8.322325,9.793087,-9.876223,-6.981383,8.538239,8.899288,-8.545048,7.188441,0.786841,-0.731201,3.539807,-0.563174,1.744591,6.125876,8.742055,2.812362,6.947647,1.356409,-5.395142,-3.538511,0.132502,4.165181,0.909709,2.207785,-9.224472,4.487518,-4.854170,2.230047,-3.783407,2.134465,-1.600873,-3.661371,-7.588307,2.877476,-1.391218,-0.566871,-4.879088,7.987041,-2.662381],[-5.749585,8.496503,-1.590401,0.305113,8.571486,-9.497390,2.568194,2.668616,-6.437919,-1.983345,4.873682,-6.024991,-8.823261,7.237995,-5.520124,-0.210569,-6.127961,-3.426936,5.528068,9.325006,8.600667,-4.192023,-6.600580,0.935479,2.902121,-0.139158,-9.497371,-0.453333,5.221794,-0.967702,1.565947,7.329664,-0.993745,-2.430777,0.918625,8.256324,-4.574815,-5.545841,2.137119,2.744729,7.219654,3.167714,-9.907318,-0.870582,-4.135044,-5.135167,-1.893307,3.503723,0.842695,-8.213536,4.649365,-9.681836,8.011857,-7.153959,-7.891227,1.939848,-2.005133,8.116863,-1.892297,-5.331254,-8.927205,-5.594395,-7.088416,-4.356728,-6.948666,0.713514,-7.518571,7.214337,-4.060996,-5.236063,4.091671,-7.866490,6.298442,5.950955,2.853542,-0.321683,-6.526559,-3.315089,-1.717626,-4.878808,-3.000320,7.904384,3.553715,-4.078807,-8.111700,-8.858116,5.942507,0.172621,-1.016382,-2.262636,5.459683,-1.156693,-8.874180,-9.134741,3.634759,3.002285,-0.373172,-0.597505,5.911769,1.035054,1.948182,5.760013,-7.350335,5.447635,5.320835,1.450473,-2.353703,-7.604896,-6.845928,-7.563485,-1.767939,0.164854,-6.934986,-3.041440,8.521124,3.060167,-6.486297,6.714338,-7.106671,8.874382,-0.452222,-6.344982,-5.573648,3.370419,9.206645,-6.490648,7.311548,7.571323,4.612943,7.635197,3.075702,-7.142627,-3.800379,8.892228,-6.310340,-4.550352,8.552285,-4.636317,-1.558544,-5.375506,1.614219,2.694601,4.494487,-0.833880,-5.912166,5.182026,-7.447009,4.966556,3.203078,8.676275,-3.624370,-0.820496,-1.945825,9.951324,2.118859,1.684141,3.743317,4.795370,2.827321,6.454382,-9.787399,4.366897,-4.178375,6.795540,8.432721,-4.231286,-3.522693,7.430209,8.240206,3.862149,0.324714,9.948690,6.292081,6.418280,-2.207653,1.368818,6.641492,3.432592,3.012757,1.270073,-0.286123,-7.486418,-5.955968,6.540327,8.594225,-4.225681,7.251992,0.981870,8.097162,6.744611,0.403914,-4.425057,-5.960039,9.893469,-6.765187,-8.666177,-3.207769,7.370016,9.374638,-8.060301,-7.159687,7.296273,-8.488161,-9.696579,0.868468,1.353613,-7.122387,8.060925,8.977818,4.650845,8.017331,-0.529610,-2.908412,-2.034952,7.409655,-9.978342,-7.619272,7.140588,-8.692220,-4.076275,-3.164430,1.509144,4.890544,7.516022,3.142591,-3.783495,-5.564569,-6.400416,-4.931141,-2.935438,2.641976]], dtype = "float32")#candidate|4581|(2, 231)|const|float32
bop_4582 = relay.left_shift(bop_4564.astype('uint64'), relay.reshape(const_4581.astype('uint64'), relay.shape_of(bop_4564))) # shape=(2, 231)
bop_4585 = relay.left_shift(bop_4567.astype('uint64'), relay.reshape(const_4581.astype('uint64'), relay.shape_of(bop_4567))) # shape=(2, 231)
var_4586 = relay.var("var_4586", dtype = "float64", shape = (70, 2))#candidate|4586|(70, 2)|var|float64
bop_4587 = relay.right_shift(call_4556.astype('uint64'), relay.reshape(var_4586.astype('uint64'), relay.shape_of(call_4556))) # shape=(70, 2)
bop_4590 = relay.right_shift(call_4559.astype('uint64'), relay.reshape(var_4586.astype('uint64'), relay.shape_of(call_4559))) # shape=(70, 2)
bop_4598 = relay.logical_and(uop_4562.astype('bool'), bop_4582.astype('bool')) # shape=(2, 231)
bop_4601 = relay.logical_and(uop_4562.astype('bool'), bop_4585.astype('bool')) # shape=(2, 231)
func_4371_call = mod.get_global_var('func_4371')
func_4373_call = mutated_mod.get_global_var('func_4373')
call_4602 = func_4371_call()
call_4603 = func_4371_call()
output = relay.Tuple([const_4558,bop_4569,uop_4573,bop_4587,bop_4598,call_4602,])
output2 = relay.Tuple([const_4558,bop_4569,uop_4573,bop_4590,bop_4601,call_4603,])
func_4612 = relay.Function([var_4557,var_4586,], output)
mod['func_4612'] = func_4612
mod = relay.transform.InferType()(mod)
mutated_mod['func_4612'] = func_4612
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4612_call = mutated_mod.get_global_var('func_4612')
var_4614 = relay.var("var_4614", dtype = "float32", shape = (231,))#candidate|4614|(231,)|var|float32
var_4615 = relay.var("var_4615", dtype = "float64", shape = (70, 2))#candidate|4615|(70, 2)|var|float64
call_4613 = func_4612_call(var_4614,var_4615,)
output = call_4613
func_4616 = relay.Function([var_4614,var_4615,], output)
mutated_mod['func_4616'] = func_4616
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3156_call = mod.get_global_var('func_3156')
func_3158_call = mutated_mod.get_global_var('func_3158')
call_4642 = func_3156_call()
call_4643 = func_3156_call()
output = relay.Tuple([call_4642,])
output2 = relay.Tuple([call_4643,])
func_4645 = relay.Function([], output)
mod['func_4645'] = func_4645
mod = relay.transform.InferType()(mod)
output = func_4645()
func_4646 = relay.Function([], output)
mutated_mod['func_4646'] = func_4646
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4652 = relay.var("var_4652", dtype = "uint16", shape = (7, 6, 15))#candidate|4652|(7, 6, 15)|var|uint16
const_4653 = relay.const([[[-2,-9,3,-7,5,-5,6,-1,9,-8,2,-8,9,-10,7],[8,-5,8,-3,9,4,-8,-9,-1,-8,5,-9,1,-6,6],[-9,7,-9,3,1,9,-5,-6,-8,1,7,6,-9,-3,9],[2,-2,-3,-3,-5,8,10,1,-6,-4,-6,4,-7,8,10],[9,-3,-1,4,-3,-10,2,-4,7,-3,6,-10,-1,3,-6],[5,5,8,6,-3,4,-10,5,-1,-4,1,-4,-10,9,4]],[[-8,-3,-9,-10,5,-6,9,5,-1,-6,1,-5,-10,3,8],[-5,6,-9,10,-2,5,4,-5,5,7,9,9,-6,8,-6],[-7,-3,10,-10,5,5,-1,1,-10,-1,10,-9,-3,7,4],[-10,-6,-6,8,5,1,1,-9,-7,3,-9,8,-1,5,-6],[3,2,10,-8,-5,2,-4,-7,9,3,-3,-2,-2,-6,1],[6,8,3,-5,8,3,5,5,2,9,6,10,3,-1,7]],[[-10,5,-10,-8,1,-6,-6,-4,6,-6,1,-4,-10,-1,7],[2,-2,5,-7,-10,9,-5,-8,7,-4,6,7,-6,-9,9],[-6,5,-9,-1,-3,9,8,6,-2,9,-10,-3,7,5,6],[-3,-8,-2,1,7,6,-6,7,-10,7,-9,4,-5,9,6],[5,-8,2,-1,-5,4,2,6,2,-2,-3,7,1,8,-3],[5,-6,2,-5,10,1,3,-4,7,-3,6,-8,-3,6,-10]],[[6,8,2,-1,-4,9,4,1,1,5,-4,9,-5,-4,-6],[1,-2,7,10,-1,8,10,-3,2,-8,7,-8,-8,1,-6],[-6,8,-8,10,-4,2,4,-5,-4,9,4,6,2,-6,3],[9,-3,-7,-3,-8,-9,-8,2,-10,-10,8,4,-7,7,-9],[-9,2,-7,-2,-6,-7,1,-10,9,-1,8,5,5,-10,9],[-3,-10,-10,-3,-7,-3,8,2,-8,9,-1,1,10,-2,-6]],[[-2,9,5,-8,2,-1,-2,10,8,-4,-9,5,3,-4,8],[3,-1,1,-2,8,6,8,10,-1,-7,-7,-2,-4,10,9],[10,-9,6,10,-10,-3,-10,6,10,8,4,2,8,2,-6],[5,7,-8,5,-3,8,1,9,-6,7,-4,9,-1,-6,4],[-8,-9,-2,-9,6,-2,-10,-10,4,-7,2,-4,-9,-2,9],[-8,-5,-3,-5,6,6,6,2,-4,4,9,-3,10,9,-9]],[[-6,-7,-4,-1,-3,-10,6,-7,-2,7,4,-1,-9,1,3],[-8,6,3,8,-9,-1,-1,3,-9,3,-10,-2,-4,-3,10],[-6,6,-6,2,-3,4,-1,2,7,10,7,-4,10,9,4],[1,8,10,6,-7,-6,-2,-7,3,-8,-1,3,-4,-5,8],[2,7,-7,-3,2,2,2,5,10,1,-10,-3,-10,-6,7],[2,5,1,-8,-7,-6,-8,-4,-8,-10,-2,-3,-3,-1,-10]],[[5,10,-3,4,-7,2,1,-2,3,-9,8,5,-8,6,9],[8,8,-6,8,2,2,-9,-6,-7,-10,3,-10,1,7,5],[-3,-2,10,-4,9,-2,-2,3,1,-6,-6,8,7,4,-7],[2,4,5,-1,2,6,2,-6,-5,-6,-7,10,-8,5,1],[-1,-3,6,-8,-6,-10,4,-10,8,7,-6,6,-10,-5,6],[-6,-8,1,7,-2,10,5,3,-6,1,-4,-8,5,5,-2]]], dtype = "uint16")#candidate|4653|(7, 6, 15)|const|uint16
bop_4654 = relay.multiply(var_4652.astype('uint16'), relay.reshape(const_4653.astype('uint16'), relay.shape_of(var_4652))) # shape=(7, 6, 15)
uop_4658 = relay.erf(var_4652.astype('float32')) # shape=(7, 6, 15)
func_3436_call = mod.get_global_var('func_3436')
func_3438_call = mutated_mod.get_global_var('func_3438')
call_4663 = func_3436_call()
call_4664 = func_3436_call()
uop_4667 = relay.log(var_4652.astype('float32')) # shape=(7, 6, 15)
output = relay.Tuple([bop_4654,uop_4658,call_4663,uop_4667,])
output2 = relay.Tuple([bop_4654,uop_4658,call_4664,uop_4667,])
func_4669 = relay.Function([var_4652,], output)
mod['func_4669'] = func_4669
mod = relay.transform.InferType()(mod)
var_4670 = relay.var("var_4670", dtype = "uint16", shape = (7, 6, 15))#candidate|4670|(7, 6, 15)|var|uint16
output = func_4669(var_4670)
func_4671 = relay.Function([var_4670], output)
mutated_mod['func_4671'] = func_4671
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3276_call = mod.get_global_var('func_3276')
func_3278_call = mutated_mod.get_global_var('func_3278')
call_4683 = relay.TupleGetItem(func_3276_call(), 3)
call_4684 = relay.TupleGetItem(func_3278_call(), 3)
const_4693 = relay.const([-4.375419,-9.970879,-0.774930,3.124290,5.481572,-2.419783,-3.542093,6.140806,-7.121169,-0.880663,-2.947956,8.940575,-2.800344,9.865112,6.855545,2.522636,9.097095,-3.398004,8.041838,0.493669,1.266195,5.403384,-4.092618,8.420319,-2.933463,-8.283838,3.041583,-9.569544,2.248848,7.735886,-5.771438,-3.673801,-9.734591,-8.724746,-1.574679,-6.380211,1.748774,-2.819970,-7.655018,-3.477873,2.440226,0.877594,-8.906546,7.412146,2.596795,-6.473156,7.010414,5.770588,-9.658296,5.315277,1.732689,-1.274316,-2.909301,-0.610427,5.183265,9.899790,6.965355,8.795545,-4.972687,-8.416140,-7.817800,7.465818,0.910693,9.669261,-1.762154,4.323016,6.413952,1.895549,-3.041903,-0.549373,-1.771906,-3.909076,4.895577,-1.358083,-4.920816,8.923243,6.122294,8.005662,-9.802464,-0.253988,2.018949,-0.454893,-8.579957,-7.568269,-9.746422,-5.156306,-0.576402,-4.013367,-7.312302,6.824015], dtype = "float64")#candidate|4693|(90,)|const|float64
bop_4694 = relay.right_shift(call_4683.astype('int8'), relay.reshape(const_4693.astype('int8'), relay.shape_of(call_4683))) # shape=(90,)
bop_4697 = relay.right_shift(call_4684.astype('int8'), relay.reshape(const_4693.astype('int8'), relay.shape_of(call_4684))) # shape=(90,)
func_2235_call = mod.get_global_var('func_2235')
func_2239_call = mutated_mod.get_global_var('func_2239')
var_4719 = relay.var("var_4719", dtype = "float32", shape = (588, 1))#candidate|4719|(588, 1)|var|float32
call_4718 = relay.TupleGetItem(func_2235_call(relay.reshape(var_4719.astype('float32'), [6, 14, 7]), relay.reshape(var_4719.astype('float32'), [6, 14, 7]), ), 0)
call_4720 = relay.TupleGetItem(func_2239_call(relay.reshape(var_4719.astype('float32'), [6, 14, 7]), relay.reshape(var_4719.astype('float32'), [6, 14, 7]), ), 0)
output = relay.Tuple([bop_4694,call_4718,var_4719,])
output2 = relay.Tuple([bop_4697,call_4720,var_4719,])
func_4741 = relay.Function([var_4719,], output)
mod['func_4741'] = func_4741
mod = relay.transform.InferType()(mod)
var_4742 = relay.var("var_4742", dtype = "float32", shape = (588, 1))#candidate|4742|(588, 1)|var|float32
output = func_4741(var_4742)
func_4743 = relay.Function([var_4742], output)
mutated_mod['func_4743'] = func_4743
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2721_call = mod.get_global_var('func_2721')
func_2722_call = mutated_mod.get_global_var('func_2722')
call_4751 = relay.TupleGetItem(func_2721_call(), 0)
call_4752 = relay.TupleGetItem(func_2722_call(), 0)
output = relay.Tuple([call_4751,])
output2 = relay.Tuple([call_4752,])
func_4766 = relay.Function([], output)
mod['func_4766'] = func_4766
mod = relay.transform.InferType()(mod)
mutated_mod['func_4766'] = func_4766
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4766_call = mutated_mod.get_global_var('func_4766')
call_4767 = func_4766_call()
output = call_4767
func_4768 = relay.Function([], output)
mutated_mod['func_4768'] = func_4768
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3861_call = mod.get_global_var('func_3861')
func_3863_call = mutated_mod.get_global_var('func_3863')
call_4778 = relay.TupleGetItem(func_3861_call(), 0)
call_4779 = relay.TupleGetItem(func_3863_call(), 0)
output = call_4778
output2 = call_4779
func_4819 = relay.Function([], output)
mod['func_4819'] = func_4819
mod = relay.transform.InferType()(mod)
mutated_mod['func_4819'] = func_4819
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4819_call = mutated_mod.get_global_var('func_4819')
call_4820 = func_4819_call()
output = call_4820
func_4821 = relay.Function([], output)
mutated_mod['func_4821'] = func_4821
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4886 = relay.var("var_4886", dtype = "uint8", shape = (1, 9, 8))#candidate|4886|(1, 9, 8)|var|uint8
var_4887 = relay.var("var_4887", dtype = "uint8", shape = (15, 9, 8))#candidate|4887|(15, 9, 8)|var|uint8
bop_4888 = relay.left_shift(var_4886.astype('uint8'), var_4887.astype('uint8')) # shape=(15, 9, 8)
uop_4895 = relay.asinh(bop_4888.astype('float64')) # shape=(15, 9, 8)
output = relay.Tuple([uop_4895,])
output2 = relay.Tuple([uop_4895,])
F = relay.Function([var_4886,var_4887,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_4886,var_4887,], output2)
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
input_4886= np.array([[[-1,7,7,-2,-8,-9,-4,-1],[4,10,1,2,10,2,1,1],[-8,-8,6,4,-8,2,-7,6],[10,1,5,9,-7,3,1,-10],[10,-4,7,-2,-10,-9,-7,10],[1,1,7,1,5,1,-6,4],[10,-4,-5,-4,-1,4,5,7],[5,4,-2,-7,-9,-2,1,2],[-6,-8,3,8,4,3,-5,-7]]], dtype='uint8')
module1.set_input('var_4886', input_4886)
input_4887= np.array([[[-10,2,4,-10,8,-9,-10,10],[-7,8,2,-7,-7,-4,-3,-3],[-2,5,-4,10,-9,-3,-9,-4],[3,8,8,-10,6,-9,6,9],[7,-5,-3,5,3,-10,2,9],[-7,-1,-10,4,-8,4,3,2],[-9,2,-4,-3,-6,-7,8,6],[-3,-2,9,-1,-9,10,8,1],[-1,-9,-10,9,-7,4,6,-1]],[[-6,-2,-1,1,5,1,5,1],[4,10,4,2,10,7,8,3],[-4,-1,9,6,-5,-9,8,-6],[-2,8,8,9,6,1,7,7],[2,-8,-10,-5,9,5,4,10],[8,3,-8,3,10,7,2,9],[-2,5,9,-8,3,-6,5,-4],[5,-8,-10,6,-6,9,9,-2],[-5,6,6,-4,5,8,7,-4]],[[9,-7,4,-1,8,-8,8,-1],[-8,6,-9,3,-10,8,10,2],[-2,10,-4,-6,-1,-1,-6,-3],[-9,8,-1,6,1,1,-3,-3],[-3,3,-4,-10,-5,3,5,-8],[6,6,7,9,5,-1,2,-5],[5,-2,9,3,5,7,-1,-6],[-1,2,-4,5,-10,10,-10,-7],[-5,3,1,9,-3,7,3,1]],[[-6,1,-6,1,-5,-10,-8,-10],[-7,8,-1,5,-5,-8,-6,2],[3,-1,-2,-7,-5,-1,-5,5],[-6,-10,-4,-2,5,-9,8,2],[7,-4,-8,-7,7,6,-1,-4],[8,7,-6,10,7,-4,-1,10],[-10,8,-3,-10,4,-4,7,-5],[9,6,1,9,-5,6,-9,-7],[-9,8,-1,4,-4,-2,6,7]],[[-9,3,1,9,4,4,-2,-4],[-8,-4,-1,-2,-9,-8,-2,-10],[-10,-3,4,-6,9,-4,5,-3],[8,-9,6,-4,6,7,10,2],[-2,-10,-9,7,-10,-1,6,-5],[-9,-3,-6,9,-2,-1,-6,2],[-8,-4,-9,-6,-9,4,-3,2],[-1,-9,6,10,-4,10,-4,8],[3,10,3,2,5,7,-3,1]],[[6,3,-8,9,4,4,10,-1],[-3,6,8,-8,-4,-4,8,2],[-5,-5,10,9,-1,-5,4,-5],[-8,5,-9,-3,-1,-1,-10,10],[-2,2,-9,9,3,-1,9,-8],[-7,4,-7,10,-4,7,-7,-1],[4,-2,-6,2,-4,-8,9,1],[8,-3,-2,3,5,10,-6,5],[-8,-2,-8,4,7,7,10,1]],[[3,5,6,-1,-7,-7,6,5],[-8,6,1,-10,-6,4,4,3],[1,3,5,2,-7,-7,10,-4],[-6,6,4,8,4,-5,-1,-6],[-8,2,4,-7,3,-3,3,1],[2,1,-5,6,-10,8,1,4],[4,3,9,1,4,-3,5,-8],[-3,-3,-7,8,9,-9,-3,-10],[-8,-6,-4,7,3,5,-5,-5]],[[4,-9,-4,-8,-4,-5,-9,-5],[-9,2,-8,-9,2,-8,-1,-7],[10,3,2,9,8,2,-9,4],[8,9,-6,10,6,-7,1,-9],[-10,7,6,8,9,9,9,-6],[-1,6,7,5,3,10,8,7],[4,1,-9,3,-1,6,8,-10],[8,6,9,1,-5,-6,-2,10],[6,-5,-2,-2,3,-4,3,9]],[[7,9,7,8,10,-5,-8,8],[8,-8,4,-5,2,-7,3,-8],[-1,6,8,-10,-9,-1,-3,10],[4,4,-10,4,-8,8,-7,2],[5,6,4,-2,-2,-2,3,8],[10,5,9,-3,-10,-8,7,-7],[-10,8,10,10,4,-8,-2,4],[6,-3,1,-4,8,2,-4,10],[3,-9,-7,-6,9,-7,8,-4]],[[-3,3,-4,6,1,-9,-6,-8],[-6,-1,-3,3,-1,-4,2,-5],[-8,10,7,6,-4,2,4,7],[6,3,5,3,4,4,7,4],[10,-9,-10,3,-8,1,5,-1],[6,-10,8,6,3,2,9,-7],[-10,4,-1,-3,3,10,4,9],[-5,-2,2,-7,-7,-7,2,5],[-2,1,7,-2,-2,8,-6,-7]],[[-6,4,-8,-5,9,-1,-7,5],[5,-1,-2,3,-5,-9,8,2],[6,1,6,-8,7,-7,6,-2],[-8,3,7,-3,9,-5,-9,-8],[-4,8,3,4,5,-4,3,-2],[-2,10,-5,4,-6,-8,8,9],[-4,-2,3,9,-9,-1,-9,-3],[9,10,10,-7,-5,-1,-2,-2],[5,7,2,-8,-9,3,-1,8]],[[-4,6,-4,-10,2,1,10,-2],[9,-7,-5,2,-3,1,5,-5],[-2,3,-8,5,2,3,3,-1],[5,3,-2,-4,-2,-10,-6,6],[-8,4,-9,5,6,7,3,2],[-7,-8,2,5,-6,-10,2,8],[4,7,10,-7,9,-8,4,-10],[4,10,-1,9,2,8,9,3],[8,9,9,6,1,-2,-3,6]],[[2,3,-2,1,-6,-7,4,8],[-7,2,2,8,4,-10,2,-4],[3,10,-2,1,-7,-2,-6,7],[-7,3,-8,5,-9,-9,-3,7],[-10,1,-5,7,1,-5,10,-4],[10,6,-6,-2,-8,-8,-3,6],[2,-5,-2,-1,2,3,7,-8],[-5,-6,9,-8,8,7,10,5],[3,-9,2,9,5,9,1,10]],[[4,-7,-2,2,-10,1,4,-2],[-6,1,10,-6,-4,-8,2,3],[1,-1,-4,5,3,5,-4,4],[9,-9,-9,-3,-8,10,-8,8],[-2,4,-3,3,5,-3,5,8],[-7,3,2,9,-9,-1,-3,-6],[-9,2,-9,8,-10,-2,10,-9],[7,-3,9,10,-2,-3,-2,-9],[-9,-3,-5,-2,-1,2,-4,-10]],[[-4,-3,-6,9,6,4,-2,10],[-3,2,3,-5,-10,-10,9,-9],[-3,-3,5,-9,-7,10,-6,-3],[-3,-7,10,-5,-10,-5,4,-9],[6,8,-8,-4,-9,-4,-7,4],[-3,8,-4,4,-9,2,-4,-2],[6,2,10,9,-2,-4,10,-9],[-2,6,2,-1,-8,-4,-5,3],[-8,-5,2,-8,7,-3,4,9]]], dtype='uint8')
module1.set_input('var_4887', input_4887)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_4886, input_4887, )
res3 = intrp3.evaluate()(input_4886, input_4887, )
res4 = intrp4.evaluate()(input_4886, input_4887, )
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
module5.set_input('var_4886', input_4886)
module5.set_input('var_4887', input_4887)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_4886, input_4887, )
res7 = intrp7.evaluate()(input_4886, input_4887, )
res8 = intrp8.evaluate()(input_4886, input_4887, )
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
module9.set_input('var_4886', input_4886)
module9.set_input('var_4887', input_4887)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_4886, input_4887, )
res11 = intrp11.evaluate()(input_4886, input_4887, )
res12 = intrp12.evaluate()(input_4886, input_4887, )
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
module13.set_input('var_4886', input_4886)
module13.set_input('var_4887', input_4887)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_4886, input_4887, )
res15 = intrp15.evaluate()(input_4886, input_4887, )
res16 = intrp16.evaluate()(input_4886, input_4887, )
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
module17.set_input('var_4886', input_4886)
module17.set_input('var_4887', input_4887)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_4886, input_4887, )
res19 = intrp19.evaluate()(input_4886, input_4887, )
res20 = intrp20.evaluate()(input_4886, input_4887, )
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
module21.set_input('var_4886', input_4886)
module21.set_input('var_4887', input_4887)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_4886, input_4887, )
res23 = intrp23.evaluate()(input_4886, input_4887, )
res24 = intrp24.evaluate()(input_4886, input_4887, )
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

'''13.165883],
0.      ],

'''