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
var_9 = relay.var("var_9", dtype = "bool", shape = (14, 1, 16))#candidate|9|(14, 1, 16)|var|bool
const_10 = relay.const([[[True,True,False,False,False,True,True,True,False,True,False,True,False,True,True,True],[True,True,False,True,True,True,False,False,False,True,False,False,True,True,False,False],[False,True,False,False,False,True,True,True,True,False,False,True,True,False,False,False],[True,True,True,False,False,False,False,False,True,False,False,True,True,True,True,False]],[[False,False,False,False,True,False,True,False,False,False,False,True,False,False,True,True],[True,False,True,True,False,True,False,False,True,False,True,False,True,False,False,True],[False,True,False,False,True,True,False,True,True,False,False,True,True,True,False,False],[True,True,False,False,False,False,False,True,False,True,False,False,True,False,True,False]],[[True,True,False,False,True,False,True,False,True,True,False,False,False,False,False,False],[False,False,False,False,False,False,False,True,True,False,True,False,False,False,False,False],[False,False,False,True,True,False,True,False,True,True,False,False,False,False,False,False],[True,False,False,True,False,False,False,True,False,True,True,True,False,False,True,False]],[[False,True,True,True,True,False,True,True,False,True,True,False,False,True,False,True],[True,False,False,True,True,True,False,True,False,True,False,False,True,True,False,False],[True,True,True,False,False,True,True,False,False,False,False,False,True,False,True,False],[False,False,True,True,True,True,True,True,True,True,False,False,True,False,False,False]],[[False,False,False,False,True,False,False,True,False,False,False,False,False,True,False,False],[True,False,False,False,True,True,False,False,False,False,True,True,False,True,True,False],[True,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True],[True,True,True,False,False,True,True,False,True,False,False,False,True,True,False,True]],[[True,True,True,True,True,True,False,False,True,False,False,True,True,True,False,False],[False,True,True,False,True,False,True,False,False,True,False,True,False,True,False,False],[False,False,True,True,True,True,True,True,False,False,False,True,True,False,False,False],[True,True,False,False,True,True,True,True,False,True,False,True,False,True,True,False]],[[True,False,False,False,False,True,True,False,True,True,True,True,True,True,True,True],[False,True,True,True,True,False,False,True,False,True,False,False,False,True,True,True],[False,True,True,False,False,True,False,False,False,True,True,False,True,False,True,True],[True,False,True,False,True,True,False,True,False,False,True,False,False,False,True,False]],[[True,True,False,False,False,False,False,False,True,True,False,False,True,True,False,False],[False,True,True,True,False,True,False,True,True,True,True,True,False,True,True,True],[False,True,True,False,True,True,False,True,False,True,True,True,False,True,False,False],[False,True,True,True,False,True,False,True,True,True,True,True,False,False,False,False]],[[False,False,False,True,True,True,False,False,False,False,True,False,True,True,True,False],[False,False,True,False,False,True,False,True,False,True,False,True,True,False,True,True],[False,False,True,False,True,True,False,True,True,True,True,True,True,False,True,True],[True,False,False,True,True,False,False,True,True,False,False,False,False,False,False,True]],[[False,True,True,True,False,True,False,False,False,True,True,True,False,False,True,True],[False,True,False,True,True,False,False,False,False,True,False,False,True,False,True,True],[True,False,False,False,True,False,False,False,True,True,True,True,True,False,False,True],[True,False,False,False,False,False,False,False,True,True,True,False,True,False,True,True]],[[True,True,True,False,True,True,False,True,False,False,False,True,False,True,False,False],[True,False,False,False,False,True,False,False,False,True,False,True,False,False,False,True],[True,True,True,True,False,False,False,False,False,False,True,False,True,True,False,True],[True,True,True,False,False,True,False,False,True,False,True,True,False,False,False,False]],[[True,True,True,False,True,True,False,True,True,False,False,True,True,False,False,True],[True,True,True,True,False,True,True,True,True,True,False,False,True,False,False,False],[False,True,False,True,False,True,True,True,True,True,False,False,True,False,True,True],[True,False,False,False,True,False,True,True,True,False,True,False,False,True,False,False]],[[False,True,False,False,False,True,True,True,False,False,True,True,False,True,False,False],[True,True,False,True,True,True,False,False,True,True,False,False,False,False,False,False],[True,False,False,True,True,True,False,True,True,False,True,False,True,True,False,False],[False,False,True,True,True,True,True,True,False,True,True,False,False,True,False,True]],[[False,False,True,True,False,True,True,True,True,False,True,False,True,True,True,False],[True,False,True,True,False,True,False,False,False,True,True,False,False,True,False,False],[False,True,False,False,False,True,True,False,True,True,False,False,False,True,False,False],[False,False,True,False,True,True,False,True,False,True,False,False,True,False,True,True]]], dtype = "bool")#candidate|10|(14, 4, 16)|const|bool
bop_11 = relay.logical_or(var_9.astype('bool'), const_10.astype('bool')) # shape=(14, 4, 16)
output = relay.Tuple([bop_11,])
output2 = relay.Tuple([bop_11,])
func_14 = relay.Function([var_9,], output)
mod['func_14'] = func_14
mod = relay.transform.InferType()(mod)
var_15 = relay.var("var_15", dtype = "bool", shape = (14, 1, 16))#candidate|15|(14, 1, 16)|var|bool
output = func_14(var_15)
func_16 = relay.Function([var_15], output)
mutated_mod['func_16'] = func_16
mutated_mod = relay.transform.InferType()(mutated_mod)
var_184 = relay.var("var_184", dtype = "float32", shape = ())#candidate|184|()|var|float32
var_185 = relay.var("var_185", dtype = "float32", shape = (2, 14, 16))#candidate|185|(2, 14, 16)|var|float32
bop_186 = relay.greater_equal(var_184.astype('bool'), var_185.astype('bool')) # shape=(2, 14, 16)
uop_204 = relay.log(bop_186.astype('float64')) # shape=(2, 14, 16)
func_14_call = mod.get_global_var('func_14')
func_16_call = mutated_mod.get_global_var('func_16')
var_215 = relay.var("var_215", dtype = "bool", shape = (224,))#candidate|215|(224,)|var|bool
call_214 = relay.TupleGetItem(func_14_call(relay.reshape(var_215.astype('bool'), [14, 1, 16])), 0)
call_216 = relay.TupleGetItem(func_16_call(relay.reshape(var_215.astype('bool'), [14, 1, 16])), 0)
func_14_call = mod.get_global_var('func_14')
func_16_call = mutated_mod.get_global_var('func_16')
call_219 = relay.TupleGetItem(func_14_call(relay.reshape(var_215.astype('bool'), [14, 1, 16])), 0)
call_220 = relay.TupleGetItem(func_16_call(relay.reshape(var_215.astype('bool'), [14, 1, 16])), 0)
bop_221 = relay.not_equal(uop_204.astype('bool'), relay.reshape(var_185.astype('bool'), relay.shape_of(uop_204))) # shape=(2, 14, 16)
output = relay.Tuple([call_214,var_215,call_219,bop_221,])
output2 = relay.Tuple([call_216,var_215,call_220,bop_221,])
func_224 = relay.Function([var_184,var_185,var_215,], output)
mod['func_224'] = func_224
mod = relay.transform.InferType()(mod)
mutated_mod['func_224'] = func_224
mutated_mod = relay.transform.InferType()(mutated_mod)
func_224_call = mutated_mod.get_global_var('func_224')
var_226 = relay.var("var_226", dtype = "float32", shape = ())#candidate|226|()|var|float32
var_227 = relay.var("var_227", dtype = "float32", shape = (2, 14, 16))#candidate|227|(2, 14, 16)|var|float32
var_228 = relay.var("var_228", dtype = "bool", shape = (224,))#candidate|228|(224,)|var|bool
call_225 = func_224_call(var_226,var_227,var_228,)
output = call_225
func_229 = relay.Function([var_226,var_227,var_228,], output)
mutated_mod['func_229'] = func_229
mutated_mod = relay.transform.InferType()(mutated_mod)
var_257 = relay.var("var_257", dtype = "uint16", shape = (4, 16, 15))#candidate|257|(4, 16, 15)|var|uint16
var_258 = relay.var("var_258", dtype = "uint16", shape = (4, 16, 15))#candidate|258|(4, 16, 15)|var|uint16
bop_259 = relay.subtract(var_257.astype('uint16'), relay.reshape(var_258.astype('uint16'), relay.shape_of(var_257))) # shape=(4, 16, 15)
uop_273 = relay.log2(var_257.astype('float32')) # shape=(4, 16, 15)
output = relay.Tuple([bop_259,uop_273,])
output2 = relay.Tuple([bop_259,uop_273,])
func_278 = relay.Function([var_257,var_258,], output)
mod['func_278'] = func_278
mod = relay.transform.InferType()(mod)
mutated_mod['func_278'] = func_278
mutated_mod = relay.transform.InferType()(mutated_mod)
func_278_call = mutated_mod.get_global_var('func_278')
var_280 = relay.var("var_280", dtype = "uint16", shape = (4, 16, 15))#candidate|280|(4, 16, 15)|var|uint16
var_281 = relay.var("var_281", dtype = "uint16", shape = (4, 16, 15))#candidate|281|(4, 16, 15)|var|uint16
call_279 = func_278_call(var_280,var_281,)
output = call_279
func_282 = relay.Function([var_280,var_281,], output)
mutated_mod['func_282'] = func_282
mutated_mod = relay.transform.InferType()(mutated_mod)
var_379 = relay.var("var_379", dtype = "float32", shape = (9, 6))#candidate|379|(9, 6)|var|float32
uop_380 = relay.sigmoid(var_379.astype('float32')) # shape=(9, 6)
output = uop_380
output2 = uop_380
func_383 = relay.Function([var_379,], output)
mod['func_383'] = func_383
mod = relay.transform.InferType()(mod)
mutated_mod['func_383'] = func_383
mutated_mod = relay.transform.InferType()(mutated_mod)
var_384 = relay.var("var_384", dtype = "float32", shape = (9, 6))#candidate|384|(9, 6)|var|float32
func_383_call = mutated_mod.get_global_var('func_383')
call_385 = func_383_call(var_384)
output = call_385
func_386 = relay.Function([var_384], output)
mutated_mod['func_386'] = func_386
mutated_mod = relay.transform.InferType()(mutated_mod)
var_405 = relay.var("var_405", dtype = "float64", shape = (3, 2, 10))#candidate|405|(3, 2, 10)|var|float64
uop_406 = relay.asin(var_405.astype('float64')) # shape=(3, 2, 10)
func_278_call = mod.get_global_var('func_278')
func_282_call = mutated_mod.get_global_var('func_282')
var_418 = relay.var("var_418", dtype = "uint16", shape = (960,))#candidate|418|(960,)|var|uint16
call_417 = relay.TupleGetItem(func_278_call(relay.reshape(var_418.astype('uint16'), [4, 16, 15]), relay.reshape(var_418.astype('uint16'), [4, 16, 15]), ), 1)
call_419 = relay.TupleGetItem(func_282_call(relay.reshape(var_418.astype('uint16'), [4, 16, 15]), relay.reshape(var_418.astype('uint16'), [4, 16, 15]), ), 1)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
var_428 = relay.var("var_428", dtype = "float32", shape = ())#candidate|428|()|var|float32
const_429 = relay.const([-4.174737,-4.229088,-2.253737,-6.195447,4.483180,-7.131053,-8.688230,9.890703,1.604170,-7.631551,-3.392700,4.462273,6.808929,1.613628,-2.333781,4.668293,8.805692,7.063869,5.422733,-3.223206,7.618042,5.441828,-8.238584,-4.501943,-5.670047,-6.276176,-0.912302,-4.771331,6.639868,-3.282789,3.153854,-4.427948,6.631678,-2.973341,9.797340,-0.233200,6.004209,5.797295,-6.399747,9.246286,-0.542590,8.608096,8.355709,-2.219618,-0.491979,3.745670,-3.321648,-6.736797,0.811087,-1.263757,1.562190,0.711707,-6.922273,-7.760040,-5.500916,4.115312,1.169239,3.201560,7.983906,8.755804,-6.814705,9.937674,-7.605109,3.607349,-3.271625,6.066762,-0.268866,-0.422853,5.561973,4.442652,-0.321723,-7.082841,-6.002208,-8.141702,2.770076,7.645158,-0.613655,-7.959366,-9.412467,-4.811336,7.614610,-6.642851,-6.794397,-2.873838,8.283389,-2.024991,6.898563,4.789044,-9.149235,1.325820,0.194007,-7.176934,0.965326,-2.444617,-7.280216,-1.322984,-3.552162,4.018327,5.348478,-5.967392,1.659118,-6.788513,6.755356,-2.936062,7.232041,8.493402,-0.619525,2.331758,-1.133698,3.201172,-8.224309,-9.298501,-1.522799,-6.126081,-6.519198,-9.207272,9.874679,-8.264037,-0.503833,-0.673143,-6.882147,-3.685856,3.236050,-7.221977,-3.391100,-3.174422,7.253261,9.256187,-0.650120,3.110740,4.469026,9.238412,-9.125496,1.385664,-6.994794,-7.400961,-0.460412,-4.663504,-7.570830,6.636304,-9.065259,-1.239446,1.912774,-5.232823,5.112016,-4.579076,7.892210,2.245112,5.489191,7.717967,-2.168398,-9.997180,-5.865625,4.638016,0.471724,0.021588,-3.680894,5.418057,-2.631768,-4.898843,2.726038,-7.381834,-2.482690,-9.732879,-5.800701,0.555149,7.402424,1.250101,-7.799629,4.182902,-6.989509,-7.381219,-7.205789,-2.696038,1.077194,-8.556376,7.143970,8.233997,-4.819119,4.383955,2.810075,0.473918,-5.316083,2.712563,5.940056,-8.605999,7.132723,6.068413,-5.932831,-9.437398,-2.158142,-0.959139,4.408243,0.530095,6.066933,0.877484,3.096678,2.232980,8.077970,2.711720,3.289393,-2.170916,-0.463641,-1.881542,2.859068,-7.801701,6.633580,7.867124,-1.829149,-3.449213,6.025004,7.473664,3.267271,-1.001592,4.436301,8.008928,1.781604,5.764485,4.042304,-7.197458,9.127892,-2.138170,5.495194,3.046298,-5.715933,3.281441,7.587742,2.880984,-3.234483,-8.523361,5.988679,-0.275909,3.513205,-4.895477,-7.326428,-8.825827,3.043101,-8.330076,-6.246026,-2.673386,-2.041232,5.834687,-4.841491,-7.414741,-4.636777,-9.093761,-4.978483,3.535396,2.857793,8.949330,-2.129440,5.727099,-0.355415,9.244813,8.489034,-6.542580,9.610477,-4.635881,-4.226989,-4.961434,3.680425,-3.800411,-2.537137,5.871857,5.285504,8.051879,-1.165192,2.293185,-5.126736,-4.534835,-8.697018,9.663768,-8.288692,-9.484261,-6.683823,7.305829,0.397970,-5.982307,7.195955,6.703748,-8.513002,7.654609,1.240661,4.898410,4.007257,-4.944233,-6.566057,3.399833,4.403470,8.549943,3.482270,9.359640,-2.133306,-1.927277,0.414160,-6.243885,-9.980120,1.814883,-0.333670,5.589540,-1.189856,-7.099185,-4.611968,4.436727,6.104762,5.319206,9.162469,9.250707,-5.684949,-6.145095,-2.542685,7.723692,-5.471467,-7.417204,-7.748986,-7.555346,6.429548,2.530745,-2.754054,0.680998,-7.033712,-9.373919,0.557211,5.992604,3.401610,-2.127039,6.849540,-4.876018,-7.088267,-1.384966,0.123605,1.961702,7.305678,-4.734180,8.415000,3.066600,-3.327194,6.532741,-4.058354,-4.673502,-8.417491,4.490717,-9.390606,2.305168,-3.040620,5.315776,-4.063378,4.571273,-2.593461,-7.105939,8.123757,6.915742,4.204439,-4.974495,3.796221,-6.405463,2.222186,2.828404,0.856240,-6.216680,6.204745,5.002506,-1.025410,6.108434,-5.352789,-3.873239,-1.262690,-5.196342,-8.668247,4.784570,6.736431,-6.323725,6.471379,-3.813027,-5.657725,-4.601246,9.639230,-8.873555,-2.472846,-3.819238,4.648689,3.769814,-4.486492,4.545889,-4.274944,-5.109383,-6.229768,-4.912171,5.379680,0.959988,0.824933,9.543161,6.054454,-5.803072,2.062376,3.086714,2.726350,-9.031837,-9.162334,-4.428369,-6.503734,-3.638483,1.500841,-9.480028,5.829998,4.691863,-8.067816,6.293346,-4.934967,2.375400,-1.858286,-3.308356,5.390991,8.194851,-2.108877,-6.806236,0.986633,-2.447112,7.180664,9.812687,-7.615916,2.995787,0.671701,-5.699988,-6.854824,9.522579,-8.707953,-1.672741,3.256485,9.367383,-9.556359,-6.076029,-0.282813,-4.963421,-5.538609,5.504288,-0.473142,9.343120,4.038139,2.992654,1.721928,3.959004,-4.499880,-4.281305,8.612564,-8.281053,3.056315,8.045099], dtype = "float32")#candidate|429|(448,)|const|float32
const_430 = relay.const([[True,True,False,False,True,True,False,False,True,False,True,False,False,True],[True,False,False,True,False,True,False,False,False,True,True,True,True,True],[True,False,False,False,False,False,True,True,True,True,True,False,True,True],[True,True,False,False,True,False,False,False,True,False,False,False,True,True],[True,False,True,True,True,True,True,True,True,False,False,True,True,True],[True,True,False,False,False,False,True,False,True,True,False,False,True,False],[False,False,False,False,False,True,True,True,False,False,False,True,True,False],[False,False,False,False,True,False,False,False,True,True,False,False,False,False],[False,True,False,True,True,False,True,False,True,False,True,True,False,False],[True,True,False,True,False,False,True,True,True,False,True,False,True,True],[False,False,True,False,True,False,True,True,False,False,True,False,False,False],[True,True,False,False,False,True,False,False,True,True,False,False,True,True],[False,True,True,False,True,False,True,False,False,False,True,True,True,False],[True,True,True,False,False,True,True,True,False,True,True,False,False,True],[False,False,True,True,True,False,True,False,False,False,False,True,True,False],[False,False,False,True,True,True,True,False,True,False,True,True,True,False]], dtype = "bool")#candidate|430|(16, 14)|const|bool
call_427 = relay.TupleGetItem(func_224_call(relay.reshape(var_428.astype('float32'), []), relay.reshape(const_429.astype('float32'), [2, 14, 16]), relay.reshape(const_430.astype('bool'), [224,]), ), 0)
call_431 = relay.TupleGetItem(func_229_call(relay.reshape(var_428.astype('float32'), []), relay.reshape(const_429.astype('float32'), [2, 14, 16]), relay.reshape(const_430.astype('bool'), [224,]), ), 0)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
call_434 = relay.TupleGetItem(func_224_call(relay.reshape(var_428.astype('float32'), []), relay.reshape(const_429.astype('float32'), [2, 14, 16]), relay.reshape(const_430.astype('bool'), [224,]), ), 2)
call_435 = relay.TupleGetItem(func_229_call(relay.reshape(var_428.astype('float32'), []), relay.reshape(const_429.astype('float32'), [2, 14, 16]), relay.reshape(const_430.astype('bool'), [224,]), ), 2)
uop_441 = relay.acos(uop_406.astype('float64')) # shape=(3, 2, 10)
bop_443 = relay.floor_divide(uop_441.astype('float32'), var_428.astype('float32')) # shape=(3, 2, 10)
bop_448 = relay.right_shift(uop_441.astype('uint64'), relay.reshape(uop_406.astype('uint64'), relay.shape_of(uop_441))) # shape=(3, 2, 10)
bop_453 = relay.less_equal(uop_406.astype('bool'), relay.reshape(uop_441.astype('bool'), relay.shape_of(uop_406))) # shape=(3, 2, 10)
output = relay.Tuple([call_417,var_418,call_427,const_429,const_430,call_434,bop_443,bop_448,bop_453,])
output2 = relay.Tuple([call_419,var_418,call_431,const_429,const_430,call_435,bop_443,bop_448,bop_453,])
func_458 = relay.Function([var_405,var_418,var_428,], output)
mod['func_458'] = func_458
mod = relay.transform.InferType()(mod)
mutated_mod['func_458'] = func_458
mutated_mod = relay.transform.InferType()(mutated_mod)
func_458_call = mutated_mod.get_global_var('func_458')
var_460 = relay.var("var_460", dtype = "float64", shape = (3, 2, 10))#candidate|460|(3, 2, 10)|var|float64
var_461 = relay.var("var_461", dtype = "uint16", shape = (960,))#candidate|461|(960,)|var|uint16
var_462 = relay.var("var_462", dtype = "float32", shape = ())#candidate|462|()|var|float32
call_459 = func_458_call(var_460,var_461,var_462,)
output = call_459
func_463 = relay.Function([var_460,var_461,var_462,], output)
mutated_mod['func_463'] = func_463
mutated_mod = relay.transform.InferType()(mutated_mod)
const_500 = relay.const([[0.292924,-1.340358,-2.926399,1.336561,6.601811,5.000462],[6.854486,-7.195943,9.287711,5.507039,3.662925,-0.684503],[5.373053,9.410787,1.577741,2.909093,1.857969,9.078500]], dtype = "float32")#candidate|500|(3, 6)|const|float32
uop_501 = relay.cosh(const_500.astype('float32')) # shape=(3, 6)
output = relay.Tuple([uop_501,])
output2 = relay.Tuple([uop_501,])
func_504 = relay.Function([], output)
mod['func_504'] = func_504
mod = relay.transform.InferType()(mod)
output = func_504()
func_505 = relay.Function([], output)
mutated_mod['func_505'] = func_505
mutated_mod = relay.transform.InferType()(mutated_mod)
var_506 = relay.var("var_506", dtype = "bool", shape = (13, 11, 10))#candidate|506|(13, 11, 10)|var|bool
const_507 = relay.const([[[False,False,False,False,True,False,True,True,True,True],[True,True,True,False,False,False,True,True,True,False],[True,True,False,False,True,True,False,True,True,True],[True,True,True,False,True,True,False,False,False,False],[False,False,True,True,False,True,True,True,False,True],[True,True,False,True,True,True,True,True,False,False],[True,False,True,False,False,False,True,False,True,False],[False,True,False,False,False,False,True,False,True,False],[True,False,True,True,False,True,False,True,False,False],[True,True,False,False,False,False,False,True,True,True],[True,True,False,True,True,True,True,True,True,False]],[[True,False,True,False,True,True,True,True,False,False],[True,True,True,False,True,True,False,True,True,True],[True,False,True,True,False,False,False,True,True,True],[False,False,True,True,True,False,False,False,True,False],[False,True,True,False,True,False,True,True,True,False],[True,False,True,False,False,True,False,False,False,False],[False,False,False,True,True,True,False,True,False,True],[True,False,False,False,False,True,False,False,True,False],[False,False,False,True,False,False,False,False,True,True],[False,True,True,True,False,True,False,False,False,False],[False,False,True,False,False,True,False,True,True,True]],[[True,False,True,True,True,True,False,False,True,True],[True,False,False,False,True,False,True,True,True,False],[False,True,False,True,True,False,False,True,True,False],[False,False,False,True,False,True,False,False,True,False],[True,False,False,True,True,True,True,False,False,False],[False,False,True,False,True,True,True,False,False,False],[False,True,True,False,False,True,True,True,True,True],[True,False,True,True,True,False,False,False,True,False],[True,True,True,False,False,False,True,True,False,False],[True,False,True,False,False,True,True,False,False,False],[True,True,False,False,False,True,True,False,False,False]],[[True,True,True,False,True,True,False,True,False,True],[True,False,True,False,False,False,True,False,False,False],[False,True,True,True,True,False,False,False,False,False],[False,True,True,False,True,True,True,False,False,False],[True,True,False,False,True,False,False,False,False,False],[False,True,True,False,False,True,False,False,True,False],[True,False,False,False,False,True,True,True,True,True],[True,False,False,True,True,True,False,True,False,False],[False,False,True,True,False,True,False,False,False,False],[True,True,False,True,True,False,False,True,True,False],[False,True,False,True,False,True,False,False,True,False]],[[True,True,True,False,False,True,False,True,False,False],[True,True,True,True,False,False,True,False,True,False],[False,False,True,True,True,False,False,True,False,True],[False,True,False,True,False,True,False,False,False,False],[False,True,True,True,False,True,True,True,False,True],[True,False,True,True,True,False,True,False,True,True],[True,True,True,False,False,True,True,True,True,True],[True,True,False,True,False,False,False,True,True,False],[False,False,True,True,True,False,True,False,False,True],[False,False,False,True,False,True,False,True,False,True],[False,True,False,False,False,False,False,True,True,True]],[[True,False,True,False,True,True,True,True,True,True],[False,True,True,False,False,True,True,False,False,True],[True,False,True,True,False,True,True,False,False,True],[True,False,True,True,False,False,False,True,True,True],[True,True,True,False,False,True,False,True,False,False],[True,True,True,False,True,True,True,False,False,False],[True,True,False,False,False,False,True,False,False,False],[False,True,False,True,True,False,False,True,True,False],[False,False,False,True,False,True,False,False,True,False],[False,True,False,False,True,False,False,False,True,False],[True,True,True,True,False,True,True,False,False,False]],[[True,False,True,True,True,True,False,False,True,True],[False,True,False,False,True,False,True,False,False,False],[False,True,True,False,False,True,True,True,True,True],[False,False,False,True,True,True,False,True,True,False],[True,False,True,True,False,True,True,True,True,False],[True,True,True,False,True,False,True,False,True,True],[False,True,True,False,False,True,True,True,False,True],[True,True,True,False,True,True,True,False,True,False],[False,False,False,False,True,True,False,False,False,True],[True,False,True,True,False,True,False,True,False,False],[False,True,False,True,False,True,True,True,True,False]],[[False,False,False,False,False,True,True,False,False,True],[True,True,True,False,False,True,False,False,True,False],[True,True,False,True,True,False,False,False,True,True],[False,True,True,False,True,True,False,True,True,False],[False,True,True,False,True,False,True,True,False,False],[False,True,False,False,False,True,False,False,True,True],[False,True,True,True,True,False,True,True,True,False],[True,False,True,True,False,True,True,True,False,True],[False,False,True,False,False,True,True,False,False,False],[False,False,True,True,True,False,True,False,False,True],[True,True,True,False,False,True,True,True,False,False]],[[True,False,False,False,False,True,True,True,True,True],[True,True,True,False,False,True,True,False,True,True],[True,False,False,False,True,True,True,False,False,True],[False,True,False,True,True,False,False,True,False,True],[False,True,True,False,False,True,True,True,True,False],[False,False,True,False,False,False,True,True,False,False],[True,True,True,True,False,True,True,False,False,True],[True,False,True,False,False,True,False,True,False,True],[False,False,False,True,False,False,True,False,False,True],[False,True,False,True,False,False,False,True,False,False],[True,False,True,False,False,True,True,False,True,True]],[[False,True,True,False,False,True,False,True,True,False],[False,True,True,True,True,True,True,True,True,False],[False,False,False,True,False,False,False,True,True,True],[False,True,False,True,True,False,False,True,True,False],[False,False,True,True,True,False,True,False,False,False],[False,False,False,False,True,False,True,True,True,False],[True,True,True,True,False,False,False,False,True,True],[False,True,True,False,True,False,False,False,True,False],[False,True,False,False,False,True,False,True,True,True],[True,False,False,False,True,False,False,True,False,True],[True,True,True,False,True,False,True,True,False,False]],[[False,False,True,False,False,True,False,False,False,True],[True,True,True,True,True,False,True,True,False,True],[True,True,False,False,True,True,False,False,True,False],[False,True,False,False,True,False,True,True,False,False],[False,True,True,True,False,True,False,True,False,False],[False,True,True,True,True,False,False,True,True,True],[True,True,False,True,True,False,True,True,True,True],[True,False,False,False,True,False,True,True,True,False],[True,False,True,False,True,True,True,True,False,False],[True,False,True,True,True,True,True,True,False,True],[False,True,True,True,True,False,True,True,False,True]],[[True,True,True,False,False,False,True,True,True,False],[True,False,False,False,False,True,True,True,False,True],[False,True,False,True,False,False,False,True,True,False],[False,False,True,True,False,True,True,False,False,True],[False,True,True,False,False,True,True,True,True,False],[True,True,True,True,True,True,True,True,False,False],[True,True,False,False,False,True,False,False,True,False],[True,True,False,False,True,False,False,False,True,True],[False,False,False,True,False,True,False,True,False,True],[False,True,False,False,False,False,True,False,False,False],[False,True,True,False,False,False,False,False,True,False]],[[True,True,False,True,True,False,True,True,False,True],[False,False,True,False,False,True,True,False,True,True],[False,True,True,False,False,True,False,False,True,True],[False,False,True,True,True,False,True,False,True,True],[False,False,True,True,False,False,False,True,False,True],[True,False,False,False,False,False,True,True,True,False],[False,True,False,True,False,True,True,False,False,True],[True,False,True,True,True,True,True,True,True,True],[False,False,True,False,False,False,True,True,True,False],[True,True,True,True,True,False,False,False,False,False],[True,True,False,False,False,True,False,True,False,True]]], dtype = "bool")#candidate|507|(13, 11, 10)|const|bool
bop_508 = relay.logical_or(var_506.astype('bool'), relay.reshape(const_507.astype('bool'), relay.shape_of(var_506))) # shape=(13, 11, 10)
output = relay.Tuple([bop_508,])
output2 = relay.Tuple([bop_508,])
func_513 = relay.Function([var_506,], output)
mod['func_513'] = func_513
mod = relay.transform.InferType()(mod)
mutated_mod['func_513'] = func_513
mutated_mod = relay.transform.InferType()(mutated_mod)
var_514 = relay.var("var_514", dtype = "bool", shape = (13, 11, 10))#candidate|514|(13, 11, 10)|var|bool
func_513_call = mutated_mod.get_global_var('func_513')
call_515 = func_513_call(var_514)
output = call_515
func_516 = relay.Function([var_514], output)
mutated_mod['func_516'] = func_516
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_523 = relay.TupleGetItem(func_504_call(), 0)
call_524 = relay.TupleGetItem(func_505_call(), 0)
output = call_523
output2 = call_524
func_525 = relay.Function([], output)
mod['func_525'] = func_525
mod = relay.transform.InferType()(mod)
mutated_mod['func_525'] = func_525
mutated_mod = relay.transform.InferType()(mutated_mod)
func_525_call = mutated_mod.get_global_var('func_525')
call_526 = func_525_call()
output = call_526
func_527 = relay.Function([], output)
mutated_mod['func_527'] = func_527
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_531 = relay.TupleGetItem(func_504_call(), 0)
call_532 = relay.TupleGetItem(func_505_call(), 0)
output = call_531
output2 = call_532
func_546 = relay.Function([], output)
mod['func_546'] = func_546
mod = relay.transform.InferType()(mod)
mutated_mod['func_546'] = func_546
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mutated_mod.get_global_var('func_546')
call_547 = func_546_call()
output = call_547
func_548 = relay.Function([], output)
mutated_mod['func_548'] = func_548
mutated_mod = relay.transform.InferType()(mutated_mod)
var_572 = relay.var("var_572", dtype = "uint32", shape = ())#candidate|572|()|var|uint32
const_573 = relay.const([[[-1,10,4,-10,-5,7,-10,-9,-1,4,5],[6,-6,6,-6,-3,3,-10,7,10,-7,-2]],[[-9,-5,-7,-10,7,-2,5,-6,-8,-7,2],[8,9,1,-4,-4,6,-5,-8,-5,-3,9]],[[-6,8,-2,3,-4,3,-8,1,9,-2,7],[-10,-4,-2,-4,6,8,4,-1,-5,-7,-10]],[[5,4,9,1,-5,10,7,-8,-7,-1,-8],[-6,-1,4,9,10,5,8,-1,-9,1,10]],[[5,9,4,-6,4,-1,-3,2,1,6,7],[10,-4,-6,10,2,-6,7,10,10,7,8]],[[10,-4,1,7,-5,-6,9,3,-10,5,4],[3,7,-10,9,-6,6,-10,-9,5,-5,2]],[[-4,3,8,9,-2,-9,-8,-5,5,-5,-3],[-9,1,4,-6,-10,-1,-6,3,-4,8,5]],[[-10,-2,-4,-9,-2,-9,4,-9,-2,-5,-5],[4,6,2,9,9,-8,8,10,10,8,6]],[[-3,-4,-5,-7,-10,8,-9,10,-8,9,4],[1,-5,-7,-8,-4,-10,-5,-9,-7,3,7]],[[-8,10,-7,-3,8,-4,-4,-7,6,3,-4],[-10,3,4,-5,-10,7,8,-2,-5,-6,7]]], dtype = "uint32")#candidate|573|(10, 2, 11)|const|uint32
bop_574 = relay.right_shift(var_572.astype('uint32'), const_573.astype('uint32')) # shape=(10, 2, 11)
uop_577 = relay.cos(bop_574.astype('float32')) # shape=(10, 2, 11)
var_581 = relay.var("var_581", dtype = "float32", shape = (10, 2, 11))#candidate|581|(10, 2, 11)|var|float32
bop_582 = relay.left_shift(uop_577.astype('int16'), relay.reshape(var_581.astype('int16'), relay.shape_of(uop_577))) # shape=(10, 2, 11)
uop_588 = relay.asin(bop_582.astype('float32')) # shape=(10, 2, 11)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
var_592 = relay.var("var_592", dtype = "float32", shape = (448,))#candidate|592|(448,)|var|float32
const_593 = relay.const([False,True,False,True,False,False,True,False,False,True,True,False,True,False,True,False,True,False,True,False,False,False,False,False,False,True,True,True,True,False,False,True,True,False,False,True,False,False,True,True,True,False,True,False,True,False,False,False,False,False,False,True,False,True,True,False,False,True,True,True,True,True,False,False,False,True,True,False,True,True,True,False,True,False,True,False,False,True,True,True,True,True,False,False,False,True,False,False,False,False,True,True,True,False,True,True,True,True,False,False,False,True,False,True,False,True,False,False,True,True,True,False,False,True,False,True,True,True,True,True,True,True,True,False,True,False,False,False,True,False,False,True,True,False,True,True,False,True,False,True,False,True,True,False,True,False,True,False,True,True,True,False,False,False,False,True,True,False,True,False,False,True,False,False,True,True,True,True,False,True,False,False,True,False,False,False,False,False,False,True,True,True,True,True,False,True,False,True,False,True,True,False,False,True,False,True,False,False,True,False,True,True,False,False,True,True,False,True,True,False,False,False,False,True,True,False,True,True,True,True,False,False,True,False], dtype = "bool")#candidate|593|(224,)|const|bool
call_591 = relay.TupleGetItem(func_224_call(relay.reshape(var_572.astype('float32'), []), relay.reshape(var_592.astype('float32'), [2, 14, 16]), relay.reshape(const_593.astype('bool'), [224,]), ), 0)
call_594 = relay.TupleGetItem(func_229_call(relay.reshape(var_572.astype('float32'), []), relay.reshape(var_592.astype('float32'), [2, 14, 16]), relay.reshape(const_593.astype('bool'), [224,]), ), 0)
func_278_call = mod.get_global_var('func_278')
func_282_call = mutated_mod.get_global_var('func_282')
const_596 = relay.const([-7,6,2,6,-7,-4,-8,-4,-3,-7,-5,2,-3,3,-2,-3,7,-10,-9,9,-3,-6,10,-7,8,-8,2,3,5,-1,-8,-7,2,-4,-4,-6,-5,-9,-2,-10,2,-5,9,10,-8,10,3,-1,-5,-5,9,7,6,9,-2,-8,6,-9,-3,-9,-7,-7,8,-5,-4,-3,4,6,-6,7,3,-4,-1,4,-1,-7,5,9,1,-6,6,9,-9,-1,-10,-5,-6,-2,4,-2,-2,-3,2,-10,-8,1,6,-6,-5,-2,-5,-9,-5,-9,10,4,-2,-7,-1,-9,-5,-2,10,-6,5,1,-4,-5,3,10,7,5,2,-6,-1,1,5,7,-3,2,2,10,-6,-6,8,4,1,7,-1,6,7,-8,-3,8,9,-4,3,-6,10,10,8,-5,1,-4,-4,6,8,-4,10,-8,3,-7,-7,-7,-5,-5,9,3,2,6,-8,7,-8,-8,4,5,-8,-6,-7,10,-10,-6,4,2,-7,-2,-1,-10,2,7,-3,-5,-7,4,4,8,10,4,3,2,2,-4,2,4,-1,-10,-3,5,-8,6,6,5,-9,-2,4,-4,-7,-1,-2,3,-6,-10,-9,-9,-8,8,-10,-10,1,1,7,4,-6,2,-2,5,-7,-4,-2,5,7,-3,-6,6,-6,-5,-4,4,-5,-10,-1,-4,-10,7,3,-1,5,7,-7,-1,4,4,6,-8,6,10,2,4,-4,5,2,8,-2,-9,2,1,-7,-3,-7,-5,-10,-8,3,-6,4,-4,1,-6,-1,-1,-6,-6,-1,5,10,1,-9,4,-1,-8,-4,-3,-6,-8,-5,-9,-5,-3,4,4,-2,-5,4,-2,10,1,-4,-10,5,-9,-9,3,-3,-2,-8,10,1,-10,5,-9,-6,10,7,2,8,-4,10,-2,-4,10,3,2,7,9,8,8,2,-1,6,1,-7,-2,2,9,-10,-8,3,7,3,-4,5,4,10,6,-1,1,-4,4,-10,-2,4,-3,3,8,7,-9,-6,2,8,-1,7,-8,6,4,7,-8,9,9,-1,-8,-1,-3,-9,10,-10,-8,-4,7,8,-7,1,-5,7,10,3,8,-4,10,3,3,-2,2,7,9,3,2,3,2,-3,10,2,-9,-9,2,-8,7,7,-9,9,2,6,-3,3,-9,-10,2,6,-8,10,-7,6,2,3,10,9,-1,-7,-4,-10,9,-1,-1,4,-2,-3,7,-2,3,-2,5,-4,-6,3,-3,2,-4,-5,-2,6,3,4,9,1,-1,-6,4,-7,2,2,-4,9,7,-4,-1,4,10,-8,-4,-2,9,3,7,-6,7,-6,2,10,-9,10,-3,-8,10,1,-2,-9,8,-8,2,-7,2,2,9,3,4,-10,8,1,1,-4,-4,7,6,8,9,-9,-9,5,2,-7,-3,-5,-4,-3,-8,-1,-1,-4,-6,-7,10,-6,-6,8,-8,4,4,4,-6,7,3,6,-8,5,5,-7,-9,-3,-3,-2,-5,-2,-8,7,3,-7,1,2,-6,-1,8,-5,2,-1,6,-4,2,10,-4,3,2,-6,7,-4,5,-3,5,2,-5,-10,8,2,4,1,10,7,-8,-6,-1,8,-6,-2,1,8,-9,-7,-6,9,2,10,-4,7,-10,-6,3,-8,5,2,6,-9,10,-1,6,8,10,3,3,-7,-5,5,-5,7,-4,5,8,-4,2,5,-7,-8,-1,2,-3,8,-10,-5,5,-4,-10,-1,-2,9,10,4,6,-6,-6,-10,7,-6,10,-3,-5,6,-7,6,-1,5,-6,1,-2,-1,-7,-7,9,-5,-5,6,7,-5,-8,-6,-3,-10,-10,5,10,-8,4,-3,-9,4,7,3,-9,-8,1,-5,5,-5,6,6,-3,3,2,-2,6,-4,-5,9,-9,-2,3,5,3,-4,8,-7,6,8,-7,8,8,6,-3,9,2,-2,-9,2,5,-2,-5,3,8,9,4,9,-9,7,-10,4,10,-6,3,2,3,8,-3,-7,-4,8,-4,-2,-4,7,-5,-5,-8,-3,-8,-8,9,8,9,-4,-4,1,-9,-10,-1,-6,-9,3,-1,-10,-3,1,4,-5,5,-6,-2,-4,9,6,5,10,5,3,2,-7,2,7,5,1,-3,-8,5,8,8,-7,9,6,9,-10,-5,-5,4,4,-6,-9,5,6,9,-5,-9,9,8,-2,7,9,10,8,8,3,-9,4,10,-4,-5,8,-1,-4,4,-7,9,-2,2,2,4,-5,-10,8,1,10,3,10,7,-3,1,4,6,8,10,7,5,-8,2,-4,-10,-1,-4,-6,-4,-6,1,-4,9,-6,-1,-2,8,1,10,5,4,4,6,1,-1,6,8,-2,1,6,4,10,2,1,9,-1,-1,-2,-4,-6,9,-10,1,6,1,-4,9,-4,5,2,-1,-8,-7,9,3,-8,-7,-3,-10,10,10,4,-8,-10,-4,-1,-5,-7,-1,-4,2,-9,8,-5,1,-10,9,-10,-2,2,-3,1,1,-10,4,7,4,5,9,2,7,-1,-6,3,-9,4,-7,-2,4,-3,6,6,-7,3,1], dtype = "uint16")#candidate|596|(960,)|const|uint16
call_595 = relay.TupleGetItem(func_278_call(relay.reshape(const_596.astype('uint16'), [4, 16, 15]), relay.reshape(const_596.astype('uint16'), [4, 16, 15]), ), 1)
call_597 = relay.TupleGetItem(func_282_call(relay.reshape(const_596.astype('uint16'), [4, 16, 15]), relay.reshape(const_596.astype('uint16'), [4, 16, 15]), ), 1)
output = relay.Tuple([uop_588,call_591,var_592,const_593,call_595,const_596,])
output2 = relay.Tuple([uop_588,call_594,var_592,const_593,call_597,const_596,])
func_598 = relay.Function([var_572,var_581,var_592,], output)
mod['func_598'] = func_598
mod = relay.transform.InferType()(mod)
mutated_mod['func_598'] = func_598
mutated_mod = relay.transform.InferType()(mutated_mod)
func_598_call = mutated_mod.get_global_var('func_598')
var_600 = relay.var("var_600", dtype = "uint32", shape = ())#candidate|600|()|var|uint32
var_601 = relay.var("var_601", dtype = "float32", shape = (10, 2, 11))#candidate|601|(10, 2, 11)|var|float32
var_602 = relay.var("var_602", dtype = "float32", shape = (448,))#candidate|602|(448,)|var|float32
call_599 = func_598_call(var_600,var_601,var_602,)
output = call_599
func_603 = relay.Function([var_600,var_601,var_602,], output)
mutated_mod['func_603'] = func_603
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_659 = relay.TupleGetItem(func_504_call(), 0)
call_660 = relay.TupleGetItem(func_505_call(), 0)
uop_669 = relay.erf(call_659.astype('float64')) # shape=(3, 6)
uop_671 = relay.erf(call_660.astype('float64')) # shape=(3, 6)
func_383_call = mod.get_global_var('func_383')
func_386_call = mutated_mod.get_global_var('func_386')
const_678 = relay.const([[-7.962133,0.660055,-6.173898,1.312887,-4.270721,4.995156,0.434686,2.646690,-9.244906,-9.970497,0.343025,9.734908,1.565481,-8.440773,-1.133699,5.874188,-4.618075,-8.354199],[-3.213354,6.710119,2.859938,0.171248,-6.571249,-6.575527,4.679408,-4.639199,6.687329,-8.589051,4.045106,-0.035754,-6.734810,-5.189981,-3.450686,2.113480,-2.496521,4.281820],[-1.143362,8.544497,-7.129429,9.727932,-5.317417,-3.568948,6.730102,-8.579877,7.980173,-8.584348,-2.698080,-2.553593,1.441170,2.754277,-0.836458,-6.539148,9.975680,-3.208810]], dtype = "float32")#candidate|678|(3, 18)|const|float32
call_677 = func_383_call(relay.reshape(const_678.astype('float32'), [9, 6]))
call_679 = func_383_call(relay.reshape(const_678.astype('float32'), [9, 6]))
var_688 = relay.var("var_688", dtype = "float64", shape = (3, 6))#candidate|688|(3, 6)|var|float64
bop_689 = relay.greater_equal(uop_669.astype('bool'), relay.reshape(var_688.astype('bool'), relay.shape_of(uop_669))) # shape=(3, 6)
bop_692 = relay.greater_equal(uop_671.astype('bool'), relay.reshape(var_688.astype('bool'), relay.shape_of(uop_671))) # shape=(3, 6)
func_383_call = mod.get_global_var('func_383')
func_386_call = mutated_mod.get_global_var('func_386')
call_696 = func_383_call(relay.reshape(const_678.astype('float32'), [9, 6]))
call_697 = func_383_call(relay.reshape(const_678.astype('float32'), [9, 6]))
uop_698 = relay.asinh(bop_689.astype('float32')) # shape=(3, 6)
uop_700 = relay.asinh(bop_692.astype('float32')) # shape=(3, 6)
output = relay.Tuple([call_677,const_678,call_696,uop_698,])
output2 = relay.Tuple([call_679,const_678,call_697,uop_700,])
func_702 = relay.Function([var_688,], output)
mod['func_702'] = func_702
mod = relay.transform.InferType()(mod)
var_703 = relay.var("var_703", dtype = "float64", shape = (3, 6))#candidate|703|(3, 6)|var|float64
output = func_702(var_703)
func_704 = relay.Function([var_703], output)
mutated_mod['func_704'] = func_704
mutated_mod = relay.transform.InferType()(mutated_mod)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_761 = func_525_call()
call_762 = func_525_call()
func_14_call = mod.get_global_var('func_14')
func_16_call = mutated_mod.get_global_var('func_16')
var_776 = relay.var("var_776", dtype = "bool", shape = (224, 1))#candidate|776|(224, 1)|var|bool
call_775 = relay.TupleGetItem(func_14_call(relay.reshape(var_776.astype('bool'), [14, 1, 16])), 0)
call_777 = relay.TupleGetItem(func_16_call(relay.reshape(var_776.astype('bool'), [14, 1, 16])), 0)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
const_789 = relay.const(9.960323, dtype = "float32")#candidate|789|()|const|float32
var_790 = relay.var("var_790", dtype = "float32", shape = (448,))#candidate|790|(448,)|var|float32
call_788 = relay.TupleGetItem(func_224_call(relay.reshape(const_789.astype('float32'), []), relay.reshape(var_790.astype('float32'), [2, 14, 16]), relay.reshape(var_776.astype('bool'), [224,]), ), 3)
call_791 = relay.TupleGetItem(func_229_call(relay.reshape(const_789.astype('float32'), []), relay.reshape(var_790.astype('float32'), [2, 14, 16]), relay.reshape(var_776.astype('bool'), [224,]), ), 3)
var_796 = relay.var("var_796", dtype = "float32", shape = (448,))#candidate|796|(448,)|var|float32
bop_797 = relay.greater(var_790.astype('bool'), relay.reshape(var_796.astype('bool'), relay.shape_of(var_790))) # shape=(448,)
uop_800 = relay.log(bop_797.astype('float64')) # shape=(448,)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
call_802 = relay.TupleGetItem(func_224_call(relay.reshape(const_789.astype('float32'), []), relay.reshape(var_796.astype('float32'), [2, 14, 16]), relay.reshape(var_776.astype('bool'), [224,]), ), 0)
call_803 = relay.TupleGetItem(func_229_call(relay.reshape(const_789.astype('float32'), []), relay.reshape(var_796.astype('float32'), [2, 14, 16]), relay.reshape(var_776.astype('bool'), [224,]), ), 0)
bop_815 = relay.minimum(call_775.astype('int64'), const_789.astype('int64')) # shape=(14, 4, 16)
bop_818 = relay.minimum(call_777.astype('int64'), const_789.astype('int64')) # shape=(14, 4, 16)
bop_832 = relay.multiply(bop_797.astype('int16'), relay.reshape(uop_800.astype('int16'), relay.shape_of(bop_797))) # shape=(448,)
func_458_call = mod.get_global_var('func_458')
func_463_call = mutated_mod.get_global_var('func_463')
var_837 = relay.var("var_837", dtype = "float64", shape = (60,))#candidate|837|(60,)|var|float64
const_838 = relay.const([7,1,-4,10,-2,-8,8,7,9,8,7,-8,-2,2,-9,6,-6,4,10,9,-2,-2,-4,-2,-8,2,-4,2,5,6,6,10,10,3,1,-3,8,5,4,-5,-8,10,9,-9,-7,-10,8,-6,-7,10,8,3,9,-7,5,-5,-1,6,8,-4,-2,-9,-6,1,-1,-1,6,-6,2,-8,-7,4,10,8,7,5,8,-10,-7,8,-3,1,-5,-6,2,5,-7,-9,-1,8,-9,-5,-5,1,-7,-3,10,-2,3,-3,2,-6,9,9,-10,-7,-10,-6,-9,-8,-6,9,-5,2,-10,2,-2,-2,-4,2,8,5,-4,-2,7,-9,10,-10,10,-4,-6,-4,-7,-1,9,-6,-3,-6,3,-9,-9,7,6,-6,-9,9,-7,6,-9,-4,2,-7,3,1,-1,-8,9,-2,3,3,8,-9,-5,8,4,2,4,-1,1,4,-2,-4,8,-6,-2,4,1,7,-10,9,1,-10,-9,3,-2,-3,9,-3,-1,-9,10,-7,7,-6,-10,-1,-4,-7,1,-2,7,7,-3,7,1,-9,9,1,-9,-3,2,2,-4,-9,-9,1,-8,-5,1,7,8,-5,1,3,5,-2,9,-10,-3,1,2,9,10,-3,6,7,7,8,-9,9,-6,-10,-5,1,1,-9,4,5,1,-7,9,10,-7,1,1,2,2,-6,4,2,-7,-10,-8,7,4,-8,-9,6,-1,3,-10,-1,9,-7,3,10,-10,-3,7,10,3,6,-8,-7,3,1,7,4,-1,10,-2,-6,-1,3,-10,-3,2,-6,-4,-1,4,-8,6,-6,-5,-1,-7,1,7,-9,-8,-1,-2,3,9,-2,-8,4,4,-8,-6,-8,-7,10,3,-2,-2,5,7,9,-4,7,-1,3,-10,-6,2,-9,3,3,-1,-8,6,-6,-1,-1,7,-6,-10,6,2,6,4,-10,-7,4,10,10,5,-10,-2,-3,10,9,-8,10,5,-3,3,7,-10,7,-10,5,4,4,-6,-5,8,-6,7,8,2,-2,-7,-5,7,-5,3,-6,5,3,9,-10,1,5,2,1,-5,8,-1,6,-7,9,-3,6,-5,3,-4,10,9,-3,-9,6,9,-5,-7,10,-4,-5,-2,-6,8,2,2,-9,-7,3,-9,9,3,8,-7,-5,9,-9,-8,9,-4,-10,-3,9,-1,8,-3,8,3,5,8,2,-7,8,7,1,-6,-10,-10,-9,-3,-3,-5,4,8,-7,-9,-2,1,-4,-7,2,8,4,-4,10,-3,-6,3,4,9,-8,4,5,-1,-6,3,-6,1,-8,-5,2,1,-6,-8,-8,4,5,7,3,-8,8,8,-2,-4,-6,3,8,-8,10,5,-2,3,8,7,8,10,-7,-10,7,-3,4,1,-10,6,-4,8,-1,5,-3,-1,-8,-8,1,-5,-3,-1,4,1,-2,4,10,-6,-4,-4,-3,-7,8,-7,-3,-7,-5,-5,4,-5,-5,-6,2,9,3,1,8,1,3,-5,4,-3,2,-8,-4,-5,-9,-3,5,6,-5,7,5,7,6,2,-6,9,-5,-6,-7,7,4,-8,3,-8,-9,10,-3,-1,1,-10,-10,-4,-3,1,3,-5,-9,-7,-7,1,-2,-9,9,5,-3,8,-4,-8,3,7,9,-5,5,1,7,-7,2,-5,7,-3,-4,4,3,3,1,-4,-3,9,10,-1,1,-7,5,-3,1,3,-7,3,10,6,7,-4,-10,10,4,-9,9,-7,10,-7,6,-8,10,2,9,-3,-9,-9,-1,-9,10,9,5,-6,-6,6,4,6,5,-3,-10,-3,3,-10,-2,-9,-8,-2,4,6,10,3,-3,-9,1,-5,8,9,-1,-8,-7,-3,-3,6,2,9,-3,7,3,6,-8,2,8,-9,-8,10,-2,-8,10,4,5,6,-6,-4,9,-5,-1,-4,8,-8,10,2,4,9,-6,-6,8,9,-6,-5,4,-1,6,-8,9,-1,-5,-1,6,3,-4,-3,2,8,-8,3,4,-1,-9,5,4,-7,2,7,9,-10,-10,7,8,2,-10,-7,10,5,6,7,9,-8,3,10,7,5,4,3,-9,7,-9,4,-4,-9,2,-10,-2,-5,-8,-5,5,-4,3,-5,2,2,8,-1,6,5,-5,-10,-7,-10,-7,2,8,-5,5,3,-3,9,2,8,-7,5,-9,-7,-9,-3,-3,-2,-1,1,-8,3,1,3,-1,9,-3,10,-9,-10,5,7,-8,-10,-6,5,5,-3,10,-10,-8,-9,-5,-5,-7,5,8,-1,-8,5,1,-8,-8,5,-1,-6,-3,-10,-3,3,9,1,9,-6,-4,3,-1,-4,-5,6,3,5,-7,4,-9,10,-4,-9,-3,-10,-8,-2,5,7,-4,-6,-4,-1,-5,1,-2,3,-1,9,-9,-3,-3,5,1,-7,-6,-10,-3,3,7,-1,-6,7,9,3,6,-5,3,6,-2,8,6,-8,10,7,7,6,-4,-1,8,3,2,-2,9,-8,5,-9,-4,9,10,3,6,1,2,-5,7,-4,-7,5,-8,-7,9,-3,-4,-8,10,4,-1,-6,-8], dtype = "uint16")#candidate|838|(960,)|const|uint16
call_836 = relay.TupleGetItem(func_458_call(relay.reshape(var_837.astype('float64'), [3, 2, 10]), relay.reshape(const_838.astype('uint16'), [960,]), relay.reshape(const_789.astype('float32'), []), ), 6)
call_839 = relay.TupleGetItem(func_463_call(relay.reshape(var_837.astype('float64'), [3, 2, 10]), relay.reshape(const_838.astype('uint16'), [960,]), relay.reshape(const_789.astype('float32'), []), ), 6)
bop_840 = relay.bitwise_and(bop_832.astype('uint16'), var_776.astype('uint16')) # shape=(224, 448)
var_851 = relay.var("var_851", dtype = "int16", shape = (448,))#candidate|851|(448,)|var|int16
bop_852 = relay.greater_equal(bop_832.astype('bool'), relay.reshape(var_851.astype('bool'), relay.shape_of(bop_832))) # shape=(448,)
output = relay.Tuple([call_761,call_788,call_802,bop_815,call_836,var_837,const_838,bop_840,bop_852,])
output2 = relay.Tuple([call_762,call_791,call_803,bop_818,call_839,var_837,const_838,bop_840,bop_852,])
func_855 = relay.Function([var_776,var_790,var_796,var_837,var_851,], output)
mod['func_855'] = func_855
mod = relay.transform.InferType()(mod)
mutated_mod['func_855'] = func_855
mutated_mod = relay.transform.InferType()(mutated_mod)
func_855_call = mutated_mod.get_global_var('func_855')
var_857 = relay.var("var_857", dtype = "bool", shape = (224, 1))#candidate|857|(224, 1)|var|bool
var_858 = relay.var("var_858", dtype = "float32", shape = (448,))#candidate|858|(448,)|var|float32
var_859 = relay.var("var_859", dtype = "float32", shape = (448,))#candidate|859|(448,)|var|float32
var_860 = relay.var("var_860", dtype = "float64", shape = (60,))#candidate|860|(60,)|var|float64
var_861 = relay.var("var_861", dtype = "int16", shape = (448,))#candidate|861|(448,)|var|int16
call_856 = func_855_call(var_857,var_858,var_859,var_860,var_861,)
output = call_856
func_862 = relay.Function([var_857,var_858,var_859,var_860,var_861,], output)
mutated_mod['func_862'] = func_862
mutated_mod = relay.transform.InferType()(mutated_mod)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_866 = func_525_call()
call_867 = func_525_call()
func_855_call = mod.get_global_var('func_855')
func_862_call = mutated_mod.get_global_var('func_862')
const_883 = relay.const([False,True,True,True,True,False,True,True,True,True,True,False,False,False,True,False,True,False,True,False,False,True,True,True,False,True,False,False,True,True,True,False,False,True,True,True,True,False,True,True,True,False,True,False,False,False,False,True,False,True,False,False,True,True,False,True,False,False,True,False,True,True,False,False,False,True,True,True,True,False,False,True,False,True,True,True,True,True,False,False,True,False,False,False,True,False,True,False,True,True,False,False,False,False,False,False,True,False,True,False,False,False,True,True,True,False,False,True,False,False,True,True,True,True,True,False,False,False,False,True,True,False,True,True,False,False,True,True,False,True,False,False,True,True,True,False,False,True,True,False,False,False,True,True,False,False,True,False,False,False,True,False,False,False,True,True,False,True,False,False,False,False,True,True,False,False,True,False,False,True,False,False,True,True,True,True,True,False,True,True,False,False,True,True,True,True,False,True,False,False,False,False,True,True,True,True,True,False,True,True,True,True,True,True,False,False,False,True,True,False,False,True,False,False,False,True,True,False,True,True,True,True,True,False], dtype = "bool")#candidate|883|(224,)|const|bool
const_884 = relay.const([-7.600759,-5.395121,2.911470,5.252368,-7.769501,7.013726,-1.386843,8.961857,-6.632442,-6.095262,-6.381572,-1.631996,0.956540,6.749331,-1.038388,8.706231,-9.681278,7.054015,3.516963,3.350964,-0.413484,-3.137006,8.176595,9.283894,2.470483,-2.452447,-5.673158,-4.733960,-2.116497,-8.363961,-7.456672,7.411695,-5.306120,2.174615,7.869710,9.560322,2.856275,-2.536337,3.036228,-3.843619,3.110465,-3.095206,1.197480,0.497257,-0.559774,4.812885,-5.685636,-4.004607,0.849897,-9.186774,-6.367945,-3.781651,-6.484135,-0.544408,9.792827,-6.871883,-3.879014,-5.250637,-4.855599,0.520288,-9.081307,5.721307,-5.409015,5.995029,8.848982,-3.181271,0.414991,5.307628,-7.273399,4.348442,-6.592585,-1.638125,2.317389,9.850614,0.875293,-7.843325,6.029025,-1.245733,1.755403,4.770672,6.568607,-8.421149,-0.182825,-2.911292,-7.661038,-3.659812,-5.660190,2.053910,-1.587925,3.583169,8.238797,6.844914,4.824841,-4.517963,0.852465,-1.535936,-2.923801,8.105732,-3.792594,1.682663,2.836847,-5.353756,-0.690537,1.613999,-3.375070,-1.585855,1.546841,-9.258273,8.653651,1.771581,-2.693395,-9.675448,1.788383,6.021083,3.106064,-3.193173,7.232566,-2.472318,-0.401439,-9.402329,-4.101235,-1.943292,-0.902833,4.683216,-3.349428,-1.430466,-7.052596,6.703092,-8.155337,2.885535,3.759508,-3.191082,-0.129910,-3.853561,5.825638,3.339267,-9.523284,0.635483,-9.575874,-9.358474,0.069636,0.045880,1.001143,1.945213,-5.889334,-8.114078,-2.025344,-7.101518,9.619387,4.584784,2.164965,0.693335,4.490028,-7.650450,7.783047,-0.385586,3.197183,-4.505439,-5.353199,-8.595123,-1.645195,-9.153501,2.625822,-1.560513,3.022280,3.696451,-8.635411,-8.485230,9.000562,-7.657881,-1.924770,-4.799151,-5.199709,-2.236965,-0.523902,0.535782,7.894476,-7.063145,-0.659519,-6.611533,-6.078259,-7.298873,-3.278191,-2.447176,8.935593,-2.963142,6.527448,2.173907,8.551318,-7.780550,-8.846546,-6.463804,5.864490,6.078441,0.006014,-6.619878,7.188516,3.855054,8.781036,-3.672048,-1.019826,-7.039638,6.616211,9.422229,-6.222447,-0.395812,-8.314237,-3.389805,7.748286,1.554752,2.692795,6.556362,3.114870,-2.370881,-5.913882,1.834809,7.496792,8.132116,-7.128019,7.826518,-7.686903,9.176521,8.867162,2.705991,9.632808,4.648604,-4.360611,-6.040378,2.826198,-6.673752,7.380860,-1.951307,-0.937002,5.233969,7.173061,5.695200,-3.769055,-1.544888,-9.230690,8.001418,-9.335055,1.226747,8.403539,-4.110149,-2.775815,-2.996167,3.694773,3.291928,-4.774549,-2.177956,5.125241,-2.846641,-6.135566,0.008935,-2.791669,4.613445,-9.785211,7.335312,-4.960557,-8.694131,-8.884539,-6.675056,-5.275244,2.927457,-2.582019,-3.715191,3.927388,5.986447,9.702344,1.337921,6.543323,-6.725733,0.831366,6.275160,-0.680515,-7.259939,8.693325,-1.810408,-3.715961,-8.785540,3.703494,1.696286,6.524725,-8.061020,3.887050,-6.411843,-7.263055,-1.282913,-7.377324,1.146450,-2.123274,-1.442272,-1.568291,0.438594,-7.765743,4.476620,-7.329290,8.377053,-4.950706,-8.692681,9.273331,-5.461811,3.271997,5.553286,1.333362,-6.586174,-9.397642,0.788623,9.135054,-2.823325,-9.614721,7.433031,-1.632205,9.348951,6.672634,-1.396596,-5.598783,-8.560776,-8.082588,-5.340027,7.281950,4.854895,4.171975,-8.044247,4.345530,-6.725021,0.221197,-5.171551,-2.759392,-6.659007,3.787541,-2.155587,5.419035,2.626100,-8.683547,9.465837,8.433718,-4.577289,8.963471,4.562878,7.951745,8.209412,-7.078921,-1.772205,7.130454,-5.649648,7.427111,-4.329377,5.086789,-1.720732,9.985994,1.045021,-0.894602,2.416761,9.062348,-9.017857,5.031763,7.984830,5.581160,-6.416679,-3.564576,4.045373,-9.029280,-8.278159,-6.971735,5.360781,5.352180,-2.166452,-0.219415,1.182981,-2.152451,-5.744264,9.543082,6.756187,9.312257,4.601572,-7.127564,-7.329271,7.646093,6.150665,9.503737,7.094174,4.964390,9.632394,-5.606573,3.505170,-4.763738,-5.065936,2.982429,3.049744,5.116687,-8.246856,0.894296,-8.707891,-7.809924,-3.743620,0.344596,-3.849889,5.018111,-0.409947,-7.544212,6.171005,-8.067227,3.061718,9.566577,-5.527401,3.160418,2.681110,8.451054,4.019388,0.858766,-5.941901,-5.349633,5.557842,-3.240517,3.704912,-4.349196,3.957404,-9.104136,-8.137761,7.176998,-7.658224,-6.837761,-9.945971,-7.718817,5.396127,6.097404,-8.222195,-8.188573,-4.495245,2.581016,5.729682,-6.436182,-8.196473,4.674181,-6.417779,-1.526924,-8.374274,-0.416835,-8.337083,-8.604237,-1.183169,6.022491,1.786758,4.856985,-5.158056,9.948754,-6.066871], dtype = "float32")#candidate|884|(448,)|const|float32
var_885 = relay.var("var_885", dtype = "float64", shape = (60,))#candidate|885|(60,)|var|float64
call_882 = relay.TupleGetItem(func_855_call(relay.reshape(const_883.astype('bool'), [224, 1]), relay.reshape(const_884.astype('float32'), [448,]), relay.reshape(const_884.astype('float32'), [448,]), relay.reshape(var_885.astype('float64'), [60,]), relay.reshape(const_884.astype('int16'), [448,]), ), 2)
call_886 = relay.TupleGetItem(func_862_call(relay.reshape(const_883.astype('bool'), [224, 1]), relay.reshape(const_884.astype('float32'), [448,]), relay.reshape(const_884.astype('float32'), [448,]), relay.reshape(var_885.astype('float64'), [60,]), relay.reshape(const_884.astype('int16'), [448,]), ), 2)
uop_891 = relay.acosh(const_884.astype('float64')) # shape=(448,)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
var_899 = relay.var("var_899", dtype = "float32", shape = ())#candidate|899|()|var|float32
call_898 = relay.TupleGetItem(func_224_call(relay.reshape(var_899.astype('float32'), []), relay.reshape(uop_891.astype('float32'), [2, 14, 16]), relay.reshape(const_883.astype('bool'), [224,]), ), 2)
call_900 = relay.TupleGetItem(func_229_call(relay.reshape(var_899.astype('float32'), []), relay.reshape(uop_891.astype('float32'), [2, 14, 16]), relay.reshape(const_883.astype('bool'), [224,]), ), 2)
output = relay.Tuple([call_866,call_882,const_883,var_885,uop_891,call_898,var_899,])
output2 = relay.Tuple([call_867,call_886,const_883,var_885,uop_891,call_900,var_899,])
func_907 = relay.Function([var_885,var_899,], output)
mod['func_907'] = func_907
mod = relay.transform.InferType()(mod)
mutated_mod['func_907'] = func_907
mutated_mod = relay.transform.InferType()(mutated_mod)
func_907_call = mutated_mod.get_global_var('func_907')
var_909 = relay.var("var_909", dtype = "float64", shape = (60,))#candidate|909|(60,)|var|float64
var_910 = relay.var("var_910", dtype = "float32", shape = ())#candidate|910|()|var|float32
call_908 = func_907_call(var_909,var_910,)
output = call_908
func_911 = relay.Function([var_909,var_910,], output)
mutated_mod['func_911'] = func_911
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_982 = func_546_call()
call_983 = func_546_call()
func_907_call = mod.get_global_var('func_907')
func_911_call = mutated_mod.get_global_var('func_911')
var_994 = relay.var("var_994", dtype = "float64", shape = (60,))#candidate|994|(60,)|var|float64
const_995 = relay.const(-3.880428, dtype = "float32")#candidate|995|()|const|float32
call_993 = relay.TupleGetItem(func_907_call(relay.reshape(var_994.astype('float64'), [60,]), relay.reshape(const_995.astype('float32'), []), ), 5)
call_996 = relay.TupleGetItem(func_911_call(relay.reshape(var_994.astype('float64'), [60,]), relay.reshape(const_995.astype('float32'), []), ), 5)
uop_998 = relay.asinh(call_993.astype('float32')) # shape=(14, 4, 16)
uop_1000 = relay.asinh(call_996.astype('float32')) # shape=(14, 4, 16)
func_907_call = mod.get_global_var('func_907')
func_911_call = mutated_mod.get_global_var('func_911')
call_1002 = relay.TupleGetItem(func_907_call(relay.reshape(var_994.astype('float64'), [60,]), relay.reshape(const_995.astype('float32'), []), ), 6)
call_1003 = relay.TupleGetItem(func_911_call(relay.reshape(var_994.astype('float64'), [60,]), relay.reshape(const_995.astype('float32'), []), ), 6)
output = relay.Tuple([call_982,var_994,const_995,uop_998,call_1002,])
output2 = relay.Tuple([call_983,var_994,const_995,uop_1000,call_1003,])
func_1008 = relay.Function([var_994,], output)
mod['func_1008'] = func_1008
mod = relay.transform.InferType()(mod)
var_1009 = relay.var("var_1009", dtype = "float64", shape = (60,))#candidate|1009|(60,)|var|float64
output = func_1008(var_1009)
func_1010 = relay.Function([var_1009], output)
mutated_mod['func_1010'] = func_1010
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_1052 = relay.TupleGetItem(func_504_call(), 0)
call_1053 = relay.TupleGetItem(func_505_call(), 0)
output = call_1052
output2 = call_1053
func_1059 = relay.Function([], output)
mod['func_1059'] = func_1059
mod = relay.transform.InferType()(mod)
mutated_mod['func_1059'] = func_1059
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1059_call = mutated_mod.get_global_var('func_1059')
call_1060 = func_1059_call()
output = call_1060
func_1061 = relay.Function([], output)
mutated_mod['func_1061'] = func_1061
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1072 = func_546_call()
call_1073 = func_546_call()
output = call_1072
output2 = call_1073
func_1078 = relay.Function([], output)
mod['func_1078'] = func_1078
mod = relay.transform.InferType()(mod)
output = func_1078()
func_1079 = relay.Function([], output)
mutated_mod['func_1079'] = func_1079
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1082 = relay.var("var_1082", dtype = "float32", shape = (16, 3))#candidate|1082|(16, 3)|var|float32
uop_1083 = relay.acos(var_1082.astype('float32')) # shape=(16, 3)
bop_1085 = relay.not_equal(var_1082.astype('bool'), relay.reshape(uop_1083.astype('bool'), relay.shape_of(var_1082))) # shape=(16, 3)
bop_1090 = relay.logical_or(uop_1083.astype('bool'), relay.reshape(bop_1085.astype('bool'), relay.shape_of(uop_1083))) # shape=(16, 3)
uop_1094 = relay.log10(bop_1085.astype('float64')) # shape=(16, 3)
bop_1098 = relay.left_shift(uop_1094.astype('uint32'), relay.reshape(uop_1083.astype('uint32'), relay.shape_of(uop_1094))) # shape=(16, 3)
func_598_call = mod.get_global_var('func_598')
func_603_call = mutated_mod.get_global_var('func_603')
var_1104 = relay.var("var_1104", dtype = "uint32", shape = ())#candidate|1104|()|var|uint32
var_1105 = relay.var("var_1105", dtype = "float32", shape = (220,))#candidate|1105|(220,)|var|float32
const_1106 = relay.const([5.930567,3.298054,-1.867869,-2.403101,5.154309,-7.312575,-7.564805,-5.917256,2.658496,5.216344,-7.735960,1.727003,-2.421546,-4.281719,-9.342667,2.763102,-4.222265,-7.538553,4.774388,-9.176137,-8.483864,9.366102,5.656915,-4.036506,-9.080719,-6.766094,-5.121921,3.514194,5.235848,-2.537961,-3.426630,1.804808,6.021120,-9.684512,-8.064374,1.852307,8.138948,2.203049,8.370081,-7.743756,1.823017,-9.481702,-4.436759,4.167606,1.272778,-9.635604,3.902954,5.508168,9.525869,-3.667764,-7.207101,6.329991,-6.385726,-7.160911,-5.094342,1.548217,1.917709,9.977104,9.921262,1.588913,9.212388,4.305190,2.680957,9.618530,-9.170935,-9.223617,9.341265,-1.820358,6.155470,7.410605,3.207446,4.052528,3.171729,9.824447,6.408556,2.835387,-6.269081,-6.395299,2.001510,-5.274512,4.177656,5.085983,-4.711891,-5.721604,-2.816673,-1.640013,-0.800628,9.723061,-1.279402,-6.864574,9.812635,-5.836898,-6.371630,-1.182266,-7.105341,6.004637,6.727146,7.041787,1.916506,7.614288,5.785217,2.414829,-8.718164,4.551354,0.772205,-1.776434,-9.352872,-2.328045,7.553857,-3.035316,-9.789362,5.772244,-2.537301,1.506536,-2.539172,-3.611873,-4.847891,7.948650,-3.830038,-1.419294,3.645930,-1.891216,9.336460,-5.184882,0.841520,1.438011,6.008581,-3.760882,6.551587,9.764098,-5.373253,0.560531,5.387427,3.421647,-6.077130,-1.986007,8.173799,4.443439,8.704319,5.249894,-5.920795,-8.814335,-6.828413,-2.616292,4.836663,0.120177,9.795267,-7.605865,-3.018216,-1.413911,7.347624,-5.192505,3.124011,-2.504311,2.616457,-6.609048,-3.048414,7.385641,5.489221,-3.975111,2.587287,-4.371443,-4.371245,-1.900122,8.586059,2.431889,1.476146,8.324652,8.422550,5.749864,6.727625,4.961037,3.531764,-9.055987,-2.713736,8.752045,7.390643,-0.177801,-5.023928,-0.671583,0.646550,1.547048,1.203489,4.427127,-2.924974,0.887310,-6.118655,4.821354,-1.621088,-6.423765,-9.396974,-6.701146,-8.379045,2.785274,7.037806,-5.535652,-1.205571,9.446753,-8.271624,-4.566575,-8.614237,-4.673562,-8.232259,-9.710733,-2.869801,9.040355,-1.740419,-0.075383,-2.521695,4.499138,4.796598,-1.752947,7.374304,3.323305,6.580290,-7.998793,1.308303,-1.405114,-7.686745,-2.405326,-1.803135,6.325692,-9.083873,6.564072,-7.328320,-2.485466,-0.111119,6.696434,7.756522,4.583951,-6.718639,-9.095531,-8.703766,-6.621376,9.768847,0.522351,-2.385097,-2.589034,-8.278271,-6.999396,-4.266459,-2.304148,2.751865,4.424145,-2.081898,5.955597,-0.429795,8.110436,9.546853,-7.114773,-3.493718,-2.856958,-4.164854,7.365132,-7.495471,4.287724,3.360886,7.359871,1.357362,6.266349,9.823065,3.845677,-4.245245,-2.323844,-3.514565,-2.830110,9.420297,9.612234,1.407861,-2.246965,-4.854257,-8.240427,-2.200159,9.393041,-3.162513,2.939314,3.357737,-6.901753,4.569180,7.274502,5.938632,9.061647,-6.840072,-4.525226,0.804252,0.961172,9.793468,8.641407,3.749755,4.197384,9.944235,3.266455,9.110547,-0.827833,7.145141,-0.741234,4.993182,9.107880,6.435055,5.685275,-9.817677,-9.248564,9.042614,6.565447,-8.736767,-0.986197,-5.666443,3.081630,3.036710,-1.561935,7.114794,2.030958,-3.584257,3.838143,4.459805,8.039627,4.019976,3.533680,5.541690,-6.799589,3.129144,-2.256483,-2.488156,-4.821232,3.637563,-8.097658,9.801757,-8.671990,-4.518691,4.241322,-8.227671,-9.997383,9.910656,-9.270110,-0.862111,-6.007713,-0.048694,1.822448,-2.186203,-5.885525,6.306148,7.758095,1.152495,-3.125428,1.134412,2.744517,-3.629665,3.100151,0.238161,8.204599,-9.991108,7.046472,-2.693394,-6.629064,-2.645138,-6.750774,4.153216,9.375317,6.428821,4.513671,1.639879,1.740249,-9.173471,5.805797,-9.201219,8.820908,8.468821,3.452544,-2.510908,-2.431231,5.034896,-0.320846,-2.707027,8.585434,6.776763,-5.056549,8.178183,3.302522,0.925023,4.355994,1.645285,-0.970518,8.054915,-2.757181,5.520695,7.041387,1.795693,-1.438500,5.195095,-5.597843,-5.358882,7.026475,-7.967108,5.827344,-2.362194,7.199980,0.758812,-3.250703,-3.583096,-5.555232,1.099293,-1.097599,7.524133,1.555804,-1.788613,-3.083061,-2.425520,-2.035842,-3.861708,-1.350473,-8.506171,-4.002483,-3.785831,-6.943376,6.805054,-1.427909,-5.390708,1.746897,8.712336,-9.126337,-8.014739,-2.335123,3.434833,2.087164,-6.143288,-5.662026,-7.195005,0.026322,6.981420,2.218766,-8.315761,5.331893,-1.783381,9.643002,6.147928,9.219128,-5.197831,7.380307,-4.973069,9.011521,-5.119096,4.485214,-2.399276,3.472925,-7.167275,-3.594751,3.146188,-7.705981], dtype = "float32")#candidate|1106|(448,)|const|float32
call_1103 = relay.TupleGetItem(func_598_call(relay.reshape(var_1104.astype('uint32'), []), relay.reshape(var_1105.astype('float32'), [10, 2, 11]), relay.reshape(const_1106.astype('float32'), [448,]), ), 5)
call_1107 = relay.TupleGetItem(func_603_call(relay.reshape(var_1104.astype('uint32'), []), relay.reshape(var_1105.astype('float32'), [10, 2, 11]), relay.reshape(const_1106.astype('float32'), [448,]), ), 5)
bop_1117 = relay.logical_and(uop_1083.astype('bool'), relay.reshape(bop_1085.astype('bool'), relay.shape_of(uop_1083))) # shape=(16, 3)
output = relay.Tuple([bop_1090,bop_1098,call_1103,var_1104,var_1105,const_1106,bop_1117,])
output2 = relay.Tuple([bop_1090,bop_1098,call_1107,var_1104,var_1105,const_1106,bop_1117,])
func_1120 = relay.Function([var_1082,var_1104,var_1105,], output)
mod['func_1120'] = func_1120
mod = relay.transform.InferType()(mod)
mutated_mod['func_1120'] = func_1120
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1120_call = mutated_mod.get_global_var('func_1120')
var_1122 = relay.var("var_1122", dtype = "float32", shape = (16, 3))#candidate|1122|(16, 3)|var|float32
var_1123 = relay.var("var_1123", dtype = "uint32", shape = ())#candidate|1123|()|var|uint32
var_1124 = relay.var("var_1124", dtype = "float32", shape = (220,))#candidate|1124|(220,)|var|float32
call_1121 = func_1120_call(var_1122,var_1123,var_1124,)
output = call_1121
func_1125 = relay.Function([var_1122,var_1123,var_1124,], output)
mutated_mod['func_1125'] = func_1125
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1161 = func_546_call()
call_1162 = func_546_call()
output = relay.Tuple([call_1161,])
output2 = relay.Tuple([call_1162,])
func_1165 = relay.Function([], output)
mod['func_1165'] = func_1165
mod = relay.transform.InferType()(mod)
output = func_1165()
func_1166 = relay.Function([], output)
mutated_mod['func_1166'] = func_1166
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1166_call = mutated_mod.get_global_var('func_1166')
call_1172 = relay.TupleGetItem(func_1165_call(), 0)
call_1173 = relay.TupleGetItem(func_1166_call(), 0)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
const_1177 = relay.const(-9.042713, dtype = "float32")#candidate|1177|()|const|float32
var_1178 = relay.var("var_1178", dtype = "float32", shape = (448,))#candidate|1178|(448,)|var|float32
var_1179 = relay.var("var_1179", dtype = "bool", shape = (224,))#candidate|1179|(224,)|var|bool
call_1176 = relay.TupleGetItem(func_224_call(relay.reshape(const_1177.astype('float32'), []), relay.reshape(var_1178.astype('float32'), [2, 14, 16]), relay.reshape(var_1179.astype('bool'), [224,]), ), 0)
call_1180 = relay.TupleGetItem(func_229_call(relay.reshape(const_1177.astype('float32'), []), relay.reshape(var_1178.astype('float32'), [2, 14, 16]), relay.reshape(var_1179.astype('bool'), [224,]), ), 0)
uop_1192 = relay.rsqrt(call_1172.astype('float32')) # shape=(3, 6)
uop_1194 = relay.rsqrt(call_1173.astype('float32')) # shape=(3, 6)
func_278_call = mod.get_global_var('func_278')
func_282_call = mutated_mod.get_global_var('func_282')
var_1200 = relay.var("var_1200", dtype = "uint16", shape = (960,))#candidate|1200|(960,)|var|uint16
call_1199 = relay.TupleGetItem(func_278_call(relay.reshape(var_1200.astype('uint16'), [4, 16, 15]), relay.reshape(var_1200.astype('uint16'), [4, 16, 15]), ), 1)
call_1201 = relay.TupleGetItem(func_282_call(relay.reshape(var_1200.astype('uint16'), [4, 16, 15]), relay.reshape(var_1200.astype('uint16'), [4, 16, 15]), ), 1)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_1205 = relay.TupleGetItem(func_504_call(), 0)
call_1206 = relay.TupleGetItem(func_505_call(), 0)
bop_1208 = relay.bitwise_or(call_1199.astype('int64'), const_1177.astype('int64')) # shape=(4, 16, 15)
bop_1211 = relay.bitwise_or(call_1201.astype('int64'), const_1177.astype('int64')) # shape=(4, 16, 15)
output = relay.Tuple([call_1176,var_1178,var_1179,uop_1192,var_1200,call_1205,bop_1208,])
output2 = relay.Tuple([call_1180,var_1178,var_1179,uop_1194,var_1200,call_1206,bop_1211,])
func_1218 = relay.Function([var_1178,var_1179,var_1200,], output)
mod['func_1218'] = func_1218
mod = relay.transform.InferType()(mod)
var_1219 = relay.var("var_1219", dtype = "float32", shape = (448,))#candidate|1219|(448,)|var|float32
var_1220 = relay.var("var_1220", dtype = "bool", shape = (224,))#candidate|1220|(224,)|var|bool
var_1221 = relay.var("var_1221", dtype = "uint16", shape = (960,))#candidate|1221|(960,)|var|uint16
output = func_1218(var_1219,var_1220,var_1221,)
func_1222 = relay.Function([var_1219,var_1220,var_1221,], output)
mutated_mod['func_1222'] = func_1222
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1255 = func_546_call()
call_1256 = func_546_call()
const_1262 = relay.const([[6.273890,9.915907,-0.885875,0.588131,-8.248926,-3.444593],[-9.152165,-1.273284,-3.775998,4.041892,-3.559947,3.918290],[-5.665566,0.116861,7.910065,1.812078,6.484295,7.508712]], dtype = "float32")#candidate|1262|(3, 6)|const|float32
bop_1263 = relay.logical_or(call_1255.astype('bool'), relay.reshape(const_1262.astype('bool'), relay.shape_of(call_1255))) # shape=(3, 6)
bop_1266 = relay.logical_or(call_1256.astype('bool'), relay.reshape(const_1262.astype('bool'), relay.shape_of(call_1256))) # shape=(3, 6)
func_1078_call = mod.get_global_var('func_1078')
func_1079_call = mutated_mod.get_global_var('func_1079')
call_1269 = func_1078_call()
call_1270 = func_1078_call()
var_1277 = relay.var("var_1277", dtype = "bool", shape = (3, 6))#candidate|1277|(3, 6)|var|bool
bop_1278 = relay.bitwise_and(bop_1263.astype('int16'), relay.reshape(var_1277.astype('int16'), relay.shape_of(bop_1263))) # shape=(3, 6)
bop_1281 = relay.bitwise_and(bop_1266.astype('int16'), relay.reshape(var_1277.astype('int16'), relay.shape_of(bop_1266))) # shape=(3, 6)
output = relay.Tuple([call_1269,bop_1278,])
output2 = relay.Tuple([call_1270,bop_1281,])
func_1290 = relay.Function([var_1277,], output)
mod['func_1290'] = func_1290
mod = relay.transform.InferType()(mod)
mutated_mod['func_1290'] = func_1290
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1291 = relay.var("var_1291", dtype = "bool", shape = (3, 6))#candidate|1291|(3, 6)|var|bool
func_1290_call = mutated_mod.get_global_var('func_1290')
call_1292 = func_1290_call(var_1291)
output = call_1292
func_1293 = relay.Function([var_1291], output)
mutated_mod['func_1293'] = func_1293
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1302 = func_546_call()
call_1303 = func_546_call()
output = call_1302
output2 = call_1303
func_1315 = relay.Function([], output)
mod['func_1315'] = func_1315
mod = relay.transform.InferType()(mod)
output = func_1315()
func_1316 = relay.Function([], output)
mutated_mod['func_1316'] = func_1316
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1382 = relay.var("var_1382", dtype = "int64", shape = (10, 1))#candidate|1382|(10, 1)|var|int64
var_1383 = relay.var("var_1383", dtype = "int64", shape = (10, 1))#candidate|1383|(10, 1)|var|int64
bop_1384 = relay.left_shift(var_1382.astype('int64'), relay.reshape(var_1383.astype('int64'), relay.shape_of(var_1382))) # shape=(10, 1)
func_383_call = mod.get_global_var('func_383')
func_386_call = mutated_mod.get_global_var('func_386')
const_1389 = relay.const([5.064364,-3.124038,-4.686830,-9.854946,-9.813094,-7.267580,2.060084,7.973812,-9.140766,-8.645063,7.632622,9.381540,4.861510,-9.291648,-7.236795,0.904263,6.048979,-3.781933,5.483717,-1.740563,-1.837896,6.186195,-8.690670,-9.171424,-7.188156,1.054078,4.170151,-8.399458,7.601302,7.391810,3.607297,-6.274434,-3.363943,-5.323230,9.485779,-4.188380,-8.128115,-0.531580,-8.603604,7.858271,3.757337,6.275859,-5.250410,5.246311,-3.877736,9.696931,-7.369164,-8.095090,-0.908417,-7.798667,8.656156,0.744977,-8.138637,-1.851458], dtype = "float32")#candidate|1389|(54,)|const|float32
call_1388 = func_383_call(relay.reshape(const_1389.astype('float32'), [9, 6]))
call_1390 = func_383_call(relay.reshape(const_1389.astype('float32'), [9, 6]))
func_383_call = mod.get_global_var('func_383')
func_386_call = mutated_mod.get_global_var('func_386')
call_1394 = func_383_call(relay.reshape(call_1388.astype('float32'), [9, 6]))
call_1395 = func_383_call(relay.reshape(call_1388.astype('float32'), [9, 6]))
output = relay.Tuple([bop_1384,call_1388,const_1389,call_1394,])
output2 = relay.Tuple([bop_1384,call_1390,const_1389,call_1395,])
func_1397 = relay.Function([var_1382,var_1383,], output)
mod['func_1397'] = func_1397
mod = relay.transform.InferType()(mod)
var_1398 = relay.var("var_1398", dtype = "int64", shape = (10, 1))#candidate|1398|(10, 1)|var|int64
var_1399 = relay.var("var_1399", dtype = "int64", shape = (10, 1))#candidate|1399|(10, 1)|var|int64
output = func_1397(var_1398,var_1399,)
func_1400 = relay.Function([var_1398,var_1399,], output)
mutated_mod['func_1400'] = func_1400
mutated_mod = relay.transform.InferType()(mutated_mod)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1416 = func_546_call()
call_1417 = func_546_call()
var_1418 = relay.var("var_1418", dtype = "float32", shape = (3, 6))#candidate|1418|(3, 6)|var|float32
bop_1419 = relay.floor_mod(call_1416.astype('float64'), relay.reshape(var_1418.astype('float64'), relay.shape_of(call_1416))) # shape=(3, 6)
bop_1422 = relay.floor_mod(call_1417.astype('float64'), relay.reshape(var_1418.astype('float64'), relay.shape_of(call_1417))) # shape=(3, 6)
bop_1423 = relay.less_equal(var_1418.astype('bool'), relay.reshape(bop_1419.astype('bool'), relay.shape_of(var_1418))) # shape=(3, 6)
bop_1426 = relay.less_equal(var_1418.astype('bool'), relay.reshape(bop_1422.astype('bool'), relay.shape_of(var_1418))) # shape=(3, 6)
output = bop_1423
output2 = bop_1426
func_1434 = relay.Function([var_1418,], output)
mod['func_1434'] = func_1434
mod = relay.transform.InferType()(mod)
mutated_mod['func_1434'] = func_1434
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1435 = relay.var("var_1435", dtype = "float32", shape = (3, 6))#candidate|1435|(3, 6)|var|float32
func_1434_call = mutated_mod.get_global_var('func_1434')
call_1436 = func_1434_call(var_1435)
output = call_1436
func_1437 = relay.Function([var_1435], output)
mutated_mod['func_1437'] = func_1437
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1462 = relay.var("var_1462", dtype = "float32", shape = (15, 13))#candidate|1462|(15, 13)|var|float32
uop_1463 = relay.atan(var_1462.astype('float32')) # shape=(15, 13)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
var_1481 = relay.var("var_1481", dtype = "float32", shape = ())#candidate|1481|()|var|float32
var_1482 = relay.var("var_1482", dtype = "float32", shape = (8, 56))#candidate|1482|(8, 56)|var|float32
const_1483 = relay.const([False,False,True,False,False,False,True,False,False,True,True,False,False,True,True,False,True,True,True,False,False,True,False,False,True,True,True,False,True,False,True,True,False,False,True,False,False,True,False,False,False,True,True,False,True,False,True,False,True,False,False,True,True,True,False,False,False,True,True,True,True,False,False,True,False,False,False,False,True,False,True,True,False,False,True,True,False,False,True,False,False,True,True,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,False,False,False,True,True,False,True,False,True,True,True,False,True,True,False,True,True,False,False,False,True,True,True,False,False,True,True,False,False,False,False,False,False,False,True,True,True,False,True,False,False,False,True,True,False,True,False,True,True,True,True,True,False,True,True,False,False,False,False,False,False,False,True,False,True,False,True,False,True,False,False,True,True,True,False,True,False,True,False,False,False,False,True,False,True,False,True,True,False,True,True,False,False,False,False,True,True,True,True,False,True,True,True,False,True,True,True,True,False,False,True,False,False,False,True,True,False,False,False,False,True,True,False,True,False,False], dtype = "bool")#candidate|1483|(224,)|const|bool
call_1480 = relay.TupleGetItem(func_224_call(relay.reshape(var_1481.astype('float32'), []), relay.reshape(var_1482.astype('float32'), [2, 14, 16]), relay.reshape(const_1483.astype('bool'), [224,]), ), 1)
call_1484 = relay.TupleGetItem(func_229_call(relay.reshape(var_1481.astype('float32'), []), relay.reshape(var_1482.astype('float32'), [2, 14, 16]), relay.reshape(const_1483.astype('bool'), [224,]), ), 1)
output = relay.Tuple([uop_1463,call_1480,var_1481,var_1482,const_1483,])
output2 = relay.Tuple([uop_1463,call_1484,var_1481,var_1482,const_1483,])
func_1492 = relay.Function([var_1462,var_1481,var_1482,], output)
mod['func_1492'] = func_1492
mod = relay.transform.InferType()(mod)
mutated_mod['func_1492'] = func_1492
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1492_call = mutated_mod.get_global_var('func_1492')
var_1494 = relay.var("var_1494", dtype = "float32", shape = (15, 13))#candidate|1494|(15, 13)|var|float32
var_1495 = relay.var("var_1495", dtype = "float32", shape = ())#candidate|1495|()|var|float32
var_1496 = relay.var("var_1496", dtype = "float32", shape = (8, 56))#candidate|1496|(8, 56)|var|float32
call_1493 = func_1492_call(var_1494,var_1495,var_1496,)
output = call_1493
func_1497 = relay.Function([var_1494,var_1495,var_1496,], output)
mutated_mod['func_1497'] = func_1497
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1166_call = mutated_mod.get_global_var('func_1166')
call_1507 = relay.TupleGetItem(func_1165_call(), 0)
call_1508 = relay.TupleGetItem(func_1166_call(), 0)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_1510 = func_525_call()
call_1511 = func_525_call()
output = relay.Tuple([call_1507,call_1510,])
output2 = relay.Tuple([call_1508,call_1511,])
func_1517 = relay.Function([], output)
mod['func_1517'] = func_1517
mod = relay.transform.InferType()(mod)
mutated_mod['func_1517'] = func_1517
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1517_call = mutated_mod.get_global_var('func_1517')
call_1518 = func_1517_call()
output = call_1518
func_1519 = relay.Function([], output)
mutated_mod['func_1519'] = func_1519
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1529 = relay.var("var_1529", dtype = "float32", shape = (15,))#candidate|1529|(15,)|var|float32
var_1530 = relay.var("var_1530", dtype = "float32", shape = (15,))#candidate|1530|(15,)|var|float32
bop_1531 = relay.floor_divide(var_1529.astype('float32'), relay.reshape(var_1530.astype('float32'), relay.shape_of(var_1529))) # shape=(15,)
output = relay.Tuple([bop_1531,])
output2 = relay.Tuple([bop_1531,])
func_1534 = relay.Function([var_1529,var_1530,], output)
mod['func_1534'] = func_1534
mod = relay.transform.InferType()(mod)
var_1535 = relay.var("var_1535", dtype = "float32", shape = (15,))#candidate|1535|(15,)|var|float32
var_1536 = relay.var("var_1536", dtype = "float32", shape = (15,))#candidate|1536|(15,)|var|float32
output = func_1534(var_1535,var_1536,)
func_1537 = relay.Function([var_1535,var_1536,], output)
mutated_mod['func_1537'] = func_1537
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1517_call = mod.get_global_var('func_1517')
func_1519_call = mutated_mod.get_global_var('func_1519')
call_1550 = relay.TupleGetItem(func_1517_call(), 0)
call_1551 = relay.TupleGetItem(func_1519_call(), 0)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_1557 = func_525_call()
call_1558 = func_525_call()
uop_1565 = relay.atan(call_1557.astype('float64')) # shape=(3, 6)
uop_1567 = relay.atan(call_1558.astype('float64')) # shape=(3, 6)
func_907_call = mod.get_global_var('func_907')
func_911_call = mutated_mod.get_global_var('func_911')
const_1570 = relay.const([[9.044547,8.404218,4.551764,9.856013,-6.899521,-4.016587,0.152599,6.488081,1.103472,-7.612015,7.907129,-1.222227,-9.676177,-0.063718,4.358908,-6.531339,-5.732667,-9.481667,5.956787,3.295826],[-1.963706,4.371986,-3.181333,0.826733,3.753943,-5.646466,3.792135,-4.926371,-4.432423,-9.297059,0.733043,-5.617639,-8.994063,1.656466,3.568264,6.976775,0.562821,2.801142,3.089985,7.285446],[-2.555014,-2.150371,-4.132965,6.673964,0.966727,-4.206825,-7.052474,-2.946952,2.779742,8.962123,-8.342291,-1.577378,0.559494,-2.024647,-2.766437,7.865214,8.972483,-2.259949,-0.944667,-0.758292]], dtype = "float64")#candidate|1570|(3, 20)|const|float64
const_1571 = relay.const(-6.441837, dtype = "float32")#candidate|1571|()|const|float32
call_1569 = relay.TupleGetItem(func_907_call(relay.reshape(const_1570.astype('float64'), [60,]), relay.reshape(const_1571.astype('float32'), []), ), 0)
call_1572 = relay.TupleGetItem(func_911_call(relay.reshape(const_1570.astype('float64'), [60,]), relay.reshape(const_1571.astype('float32'), []), ), 0)
bop_1583 = relay.bitwise_xor(uop_1565.astype('uint32'), relay.reshape(call_1569.astype('uint32'), relay.shape_of(uop_1565))) # shape=(3, 6)
bop_1586 = relay.bitwise_xor(uop_1567.astype('uint32'), relay.reshape(call_1572.astype('uint32'), relay.shape_of(uop_1567))) # shape=(3, 6)
uop_1588 = relay.sigmoid(bop_1583.astype('float64')) # shape=(3, 6)
uop_1590 = relay.sigmoid(bop_1586.astype('float64')) # shape=(3, 6)
output = relay.Tuple([call_1550,const_1570,const_1571,uop_1588,])
output2 = relay.Tuple([call_1551,const_1570,const_1571,uop_1590,])
func_1597 = relay.Function([], output)
mod['func_1597'] = func_1597
mod = relay.transform.InferType()(mod)
mutated_mod['func_1597'] = func_1597
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1597_call = mutated_mod.get_global_var('func_1597')
call_1598 = func_1597_call()
output = call_1598
func_1599 = relay.Function([], output)
mutated_mod['func_1599'] = func_1599
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1605 = relay.const([[-0.025981,-9.043029,3.509198,-1.268182,5.969705,-9.829256,8.153836,9.956602,-6.993028],[1.232067,8.606082,7.489495,-7.858752,-5.745914,-6.157245,7.597023,4.640057,5.988534],[-2.128136,7.154395,5.141614,-2.765099,-2.860008,0.971408,3.587005,0.968385,-3.578408],[8.783292,7.005601,-3.314521,1.477982,-1.877701,-8.420323,7.851904,5.229801,-9.885550],[6.577873,-1.398787,-5.632867,-2.653023,-7.024294,9.325135,-5.782943,3.411776,-8.380509],[-8.393559,3.625167,-1.880301,-1.438178,-3.268551,1.345839,-3.692272,6.959158,-1.302659],[8.195792,9.563389,4.497457,-7.153469,1.190306,9.669050,-0.472354,-1.995739,-5.021696],[9.686567,-7.387481,5.856416,-8.979684,-5.317477,-6.156692,-9.590919,-4.779173,2.424546]], dtype = "float32")#candidate|1605|(8, 9)|const|float32
uop_1606 = relay.erf(const_1605.astype('float32')) # shape=(8, 9)
output = uop_1606
output2 = uop_1606
func_1615 = relay.Function([], output)
mod['func_1615'] = func_1615
mod = relay.transform.InferType()(mod)
output = func_1615()
func_1616 = relay.Function([], output)
mutated_mod['func_1616'] = func_1616
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1650 = relay.var("var_1650", dtype = "float64", shape = (2, 5, 4))#candidate|1650|(2, 5, 4)|var|float64
const_1651 = relay.const([[[4.803016,-5.181897,-8.225903,-3.692043],[-6.403137,-5.048523,6.597347,3.923659],[-9.000201,-0.925353,-8.411575,-3.544888],[-3.589566,0.075049,2.368511,-9.605075],[9.394213,4.948175,-2.687766,-4.551930]],[[-1.238646,8.300586,3.512265,4.980697],[3.513973,-7.627665,7.774036,6.310578],[3.559054,-8.866929,-3.797326,-5.553536],[4.174586,9.069637,9.820558,1.008837],[-8.350743,4.095843,7.979083,9.492897]]], dtype = "float64")#candidate|1651|(2, 5, 4)|const|float64
bop_1652 = relay.floor_mod(var_1650.astype('float64'), relay.reshape(const_1651.astype('float64'), relay.shape_of(var_1650))) # shape=(2, 5, 4)
output = bop_1652
output2 = bop_1652
func_1659 = relay.Function([var_1650,], output)
mod['func_1659'] = func_1659
mod = relay.transform.InferType()(mod)
var_1660 = relay.var("var_1660", dtype = "float64", shape = (2, 5, 4))#candidate|1660|(2, 5, 4)|var|float64
output = func_1659(var_1660)
func_1661 = relay.Function([var_1660], output)
mutated_mod['func_1661'] = func_1661
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1615_call = mod.get_global_var('func_1615')
func_1616_call = mutated_mod.get_global_var('func_1616')
call_1675 = func_1615_call()
call_1676 = func_1615_call()
uop_1685 = relay.cos(call_1675.astype('float64')) # shape=(8, 9)
uop_1687 = relay.cos(call_1676.astype('float64')) # shape=(8, 9)
func_1492_call = mod.get_global_var('func_1492')
func_1497_call = mutated_mod.get_global_var('func_1497')
const_1689 = relay.const([-3.904201,-9.992057,6.608651,1.332684,-2.308600,-3.209284,-0.334746,-1.824698,5.140820,1.952148,5.445612,-5.401607,9.068767,-7.329720,0.960368,4.921613,6.307580,9.671434,-0.403370,8.567219,0.616163,5.564014,3.027142,-6.744762,7.687200,-6.065537,5.118510,2.803025,2.565862,8.639411,-6.080456,6.068064,-2.542397,-4.894401,1.239159,-1.903557,2.639700,2.551622,9.965452,7.252275,5.876356,-4.570251,5.642275,1.047869,8.816475,5.646706,-7.330761,1.805023,-5.501046,6.531687,6.368639,-3.032728,8.053986,8.718564,9.319918,-9.170371,1.290242,9.641543,-3.533431,6.004149,8.635538,9.108323,7.587852,-1.171131,5.077537,-4.122001,9.540128,6.733535,-2.532474,-8.310929,-8.633319,-8.742646,2.625850,-8.114107,4.375799,5.669571,-7.626409,-1.281968,8.786481,3.042365,-0.047525,-3.624521,-9.842134,-0.924839,-5.863592,-0.374680,0.305152,-7.665298,-6.770171,-2.840444,9.080432,-8.375245,9.018254,4.933316,-9.966469,6.227971,-3.150461,-1.460107,2.903930,4.602441,7.632795,-4.959892,0.843816,6.414639,2.255805,7.079993,6.966762,5.072326,-6.577849,-3.021371,1.084724,0.493580,0.850661,4.865628,-9.211299,-2.744984,-3.280709,8.420833,-0.629079,-4.882779,-9.485830,-1.759171,7.950701,-2.372248,-3.726195,-3.825332,-8.515898,-0.761104,-5.825766,5.186510,-6.505872,-7.359361,-1.739011,-1.631460,4.831725,6.975286,-5.290406,7.948601,0.190690,1.080964,-5.160194,8.273244,2.155363,-6.055104,5.737932,5.419073,-4.166318,4.121159,2.670298,8.801912,-0.199898,-4.630820,-1.795820,-6.833647,-8.569808,9.751700,-9.781983,6.079094,-4.996249,2.866787,6.654747,-6.399450,-1.040407,0.059132,5.908037,-1.351921,9.361883,5.574091,3.704168,5.906628,8.566232,-9.358314,5.061881,-8.723018,-7.977727,-7.045308,-2.770880,0.073285,-4.047250,-6.545679,4.749507,4.662133,5.627628,-5.424437,-7.844882,-4.564988,-6.941950,3.474470,4.129348,-6.661759,7.591716,6.914996,8.755291,4.040946,8.353488], dtype = "float32")#candidate|1689|(195,)|const|float32
const_1690 = relay.const(-1.385133, dtype = "float32")#candidate|1690|()|const|float32
const_1691 = relay.const([-4.051761,4.608081,-3.494279,-0.527290,-5.688475,7.639002,1.665609,5.330718,-7.927045,9.572503,-2.061312,0.628925,1.400317,-2.023853,5.845249,-0.225360,7.327737,2.304660,6.596540,-4.533932,1.796275,-7.078358,-8.217865,3.398191,-3.640237,-0.176451,-8.269841,4.881207,-2.618133,-0.860264,0.855357,4.247788,7.950927,-2.872431,-4.047168,-9.906864,-8.626775,-9.883004,-6.895945,4.912131,-2.161311,8.925297,5.006958,-4.124046,-2.353317,-1.053534,0.571746,-4.728702,-6.797505,-4.904482,4.817389,-4.992484,7.946589,6.238967,-4.545285,8.305377,-7.183734,9.679564,0.374337,1.961818,5.585442,2.152606,-3.943878,9.580271,3.988186,6.508980,-8.170918,-7.200499,9.509883,5.859542,5.776971,-2.472444,2.366965,1.378490,-8.585618,-1.374218,-6.386496,-6.664130,1.108253,3.087666,0.565525,-3.973974,-4.842188,-1.484046,-1.645113,-1.057956,-1.416348,6.829892,-1.825499,-6.210924,-3.921787,-9.864638,-5.410904,1.280071,-7.798267,-7.755165,1.706985,5.189680,-2.882774,2.978396,3.521854,-1.696162,6.359363,3.398430,-2.614608,-5.959529,-5.836005,-1.242268,5.574883,8.376433,-8.013079,-4.069965,-1.154080,-3.463358,8.685768,3.744389,1.721896,-5.253917,4.417403,-9.932592,9.328993,-9.520650,-3.287910,-0.754794,-4.031490,0.628605,-7.706406,7.037809,-8.997681,3.111224,3.458866,-2.686548,-6.200703,2.946306,9.164044,3.678031,-6.271009,-3.188325,-7.327418,7.215595,-3.783943,-7.025022,-3.265392,4.665189,7.535606,-3.426716,2.875868,4.391103,8.067534,-6.098792,-4.936567,7.128865,1.249660,0.430662,-9.909823,7.376027,-2.129949,6.597311,8.956787,0.424178,8.916015,1.175852,-6.595514,-2.106499,-8.578372,-6.087840,5.498265,0.233267,-9.533992,-2.483744,1.023614,0.937086,7.074563,-5.005692,-0.782955,4.540475,2.300970,-7.592654,1.344228,5.561269,8.037658,1.390939,8.584836,-2.768472,2.143318,2.321686,-5.099760,-2.133088,4.404173,-8.204296,-7.753028,-6.985443,-5.891437,-7.068208,2.957585,4.321756,6.653837,9.839930,-1.710117,-9.175446,7.581538,-4.036366,8.067513,2.837540,-5.787759,-1.487812,-9.807888,-1.746373,7.220031,-2.427158,9.599268,-0.414509,-7.638550,8.733756,1.679608,3.183837,6.040205,-5.480718,-1.539833,-6.947821,5.329849,-3.296801,0.852772,-6.517126,3.057465,0.044288,1.772190,0.837726,-1.507191,-6.010564,6.984829,-7.532612,2.362124,9.766243,-9.786801,-6.247113,-3.405932,4.353797,-1.535434,-4.190518,7.920973,-6.365738,7.793981,-9.675306,-0.596192,3.368896,-2.223615,5.384332,-2.617291,-3.897678,-5.445618,-6.754399,6.740228,-1.193699,-3.334836,3.814927,-3.193914,-3.457858,2.921974,6.894484,8.183958,4.120160,-9.246212,2.141779,3.082067,-4.512699,5.812927,-4.181324,-7.114581,-6.848277,-5.965231,-4.695310,6.876680,-2.493696,-6.408456,-4.442166,7.606353,6.358735,-3.654202,-8.907970,-5.721643,0.164998,-7.240510,-6.588355,-0.528787,1.076514,-4.798659,-0.714779,-7.018920,6.778785,-1.343733,-9.973912,0.588469,-6.005438,-7.816732,-6.198943,1.674548,-9.660031,0.469812,-2.883559,0.384003,-7.959851,-4.930500,6.724274,-6.914685,2.136041,1.469820,-7.163628,9.068978,-7.748402,0.546944,8.529907,2.648328,-0.757882,-3.639620,-2.332275,-2.057918,-3.722114,7.557680,4.983184,-0.289247,8.036678,-1.745209,-6.957330,-1.135446,-0.574593,8.213560,-7.173181,3.502253,-1.551278,4.921520,-8.099893,6.308708,-4.671524,-4.491702,-5.686288,-6.020723,-7.483377,6.202181,8.246566,-5.220840,-3.945318,4.143137,-5.184001,6.035440,-2.581899,3.121520,7.280210,9.673687,-3.057453,0.963132,-5.740201,-4.410434,3.529577,-8.637792,2.720366,-5.688922,-3.458013,-9.094657,-2.860806,4.512469,-1.398024,8.266693,-0.296573,-8.248585,1.562230,1.091783,3.221599,-7.828790,-0.076278,8.581987,4.357854,8.886840,8.945571,-3.141281,4.034898,6.216446,2.471837,0.252751,1.474365,-5.681719,-6.566409,7.160586,9.249768,-5.604528,-5.558940,3.389702,5.212072,1.247497,-6.658716,-9.249066,-8.187387,2.173937,7.210113,5.275745,-0.608873,1.414453,5.338276,6.162646,-9.860215,-8.229700,6.683188,1.327358,-0.448318,-6.538471,-0.173650,8.854993,-4.087934,-2.636829,6.513700,-8.589394,2.824509,-3.043435,-7.565857,7.550103,6.213748,4.386255,2.436518,4.441002,-6.481886,-1.800449,6.395711,0.851148,2.325267,8.753057,2.552266,-0.598081,7.213225,8.489658,-5.378227,-9.565762,7.146954,7.452942,-2.351859,-4.252658,-5.595157,-1.347292,-5.497604,7.396400,0.832624,0.366126,-2.454182,-1.265389,-4.998332,-0.270240,-2.764342,4.498757,-5.606676], dtype = "float32")#candidate|1691|(448,)|const|float32
call_1688 = relay.TupleGetItem(func_1492_call(relay.reshape(const_1689.astype('float32'), [15, 13]), relay.reshape(const_1690.astype('float32'), []), relay.reshape(const_1691.astype('float32'), [8, 56]), ), 4)
call_1692 = relay.TupleGetItem(func_1497_call(relay.reshape(const_1689.astype('float32'), [15, 13]), relay.reshape(const_1690.astype('float32'), []), relay.reshape(const_1691.astype('float32'), [8, 56]), ), 4)
uop_1696 = relay.sqrt(call_1688.astype('float32')) # shape=(224,)
uop_1698 = relay.sqrt(call_1692.astype('float32')) # shape=(224,)
uop_1699 = relay.sqrt(uop_1685.astype('float32')) # shape=(8, 9)
uop_1701 = relay.sqrt(uop_1687.astype('float32')) # shape=(8, 9)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1703 = func_546_call()
call_1704 = func_546_call()
output = relay.Tuple([const_1689,const_1690,const_1691,uop_1696,uop_1699,call_1703,])
output2 = relay.Tuple([const_1689,const_1690,const_1691,uop_1698,uop_1701,call_1704,])
func_1711 = relay.Function([], output)
mod['func_1711'] = func_1711
mod = relay.transform.InferType()(mod)
mutated_mod['func_1711'] = func_1711
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1711_call = mutated_mod.get_global_var('func_1711')
call_1712 = func_1711_call()
output = call_1712
func_1713 = relay.Function([], output)
mutated_mod['func_1713'] = func_1713
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_1761 = relay.TupleGetItem(func_504_call(), 0)
call_1762 = relay.TupleGetItem(func_505_call(), 0)
uop_1778 = relay.sin(call_1761.astype('float32')) # shape=(3, 6)
uop_1780 = relay.sin(call_1762.astype('float32')) # shape=(3, 6)
var_1782 = relay.var("var_1782", dtype = "float32", shape = (3, 6))#candidate|1782|(3, 6)|var|float32
bop_1783 = relay.floor_divide(uop_1778.astype('float32'), relay.reshape(var_1782.astype('float32'), relay.shape_of(uop_1778))) # shape=(3, 6)
bop_1786 = relay.floor_divide(uop_1780.astype('float32'), relay.reshape(var_1782.astype('float32'), relay.shape_of(uop_1780))) # shape=(3, 6)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_1793 = func_546_call()
call_1794 = func_546_call()
output = relay.Tuple([bop_1783,call_1793,])
output2 = relay.Tuple([bop_1786,call_1794,])
func_1796 = relay.Function([var_1782,], output)
mod['func_1796'] = func_1796
mod = relay.transform.InferType()(mod)
var_1797 = relay.var("var_1797", dtype = "float32", shape = (3, 6))#candidate|1797|(3, 6)|var|float32
output = func_1796(var_1797)
func_1798 = relay.Function([var_1797], output)
mutated_mod['func_1798'] = func_1798
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1615_call = mod.get_global_var('func_1615')
func_1616_call = mutated_mod.get_global_var('func_1616')
call_1807 = func_1615_call()
call_1808 = func_1615_call()
var_1813 = relay.var("var_1813", dtype = "float32", shape = (8, 9))#candidate|1813|(8, 9)|var|float32
bop_1814 = relay.subtract(call_1807.astype('uint64'), relay.reshape(var_1813.astype('uint64'), relay.shape_of(call_1807))) # shape=(8, 9)
bop_1817 = relay.subtract(call_1808.astype('uint64'), relay.reshape(var_1813.astype('uint64'), relay.shape_of(call_1808))) # shape=(8, 9)
bop_1826 = relay.logical_and(call_1807.astype('bool'), relay.reshape(bop_1814.astype('bool'), relay.shape_of(call_1807))) # shape=(8, 9)
bop_1829 = relay.logical_and(call_1808.astype('bool'), relay.reshape(bop_1817.astype('bool'), relay.shape_of(call_1808))) # shape=(8, 9)
bop_1830 = relay.maximum(bop_1826.astype('int16'), relay.reshape(bop_1814.astype('int16'), relay.shape_of(bop_1826))) # shape=(8, 9)
bop_1833 = relay.maximum(bop_1829.astype('int16'), relay.reshape(bop_1817.astype('int16'), relay.shape_of(bop_1829))) # shape=(8, 9)
func_1659_call = mod.get_global_var('func_1659')
func_1661_call = mutated_mod.get_global_var('func_1661')
const_1840 = relay.const([-2.479362,-4.759303,6.516618,-8.738483,-2.686716,-9.969201,4.018881,-5.976510,5.042146,-9.772698,-2.411792,2.854532,2.615654,6.960435,4.330936,-1.820356,6.088316,-0.171850,5.607716,4.394960,-1.074913,-7.347089,-1.795709,-9.967034,-5.702191,-0.045882,-5.361890,-4.319064,-3.162189,-4.975653,-1.980323,6.720980,1.327626,-2.137229,-2.434119,-9.804085,-6.779850,-1.718353,3.970425,-8.396790], dtype = "float64")#candidate|1840|(40,)|const|float64
call_1839 = func_1659_call(relay.reshape(const_1840.astype('float64'), [2, 5, 4]))
call_1841 = func_1659_call(relay.reshape(const_1840.astype('float64'), [2, 5, 4]))
func_1534_call = mod.get_global_var('func_1534')
func_1537_call = mutated_mod.get_global_var('func_1537')
const_1864 = relay.const([[3.002110,-8.707232,9.398861,-7.673935,2.553203,2.343104,6.367942,1.457008,5.980898,-2.171566,3.188993,4.136076,-7.679649,8.590087,-7.621707]], dtype = "float32")#candidate|1864|(1, 15)|const|float32
call_1863 = relay.TupleGetItem(func_1534_call(relay.reshape(const_1864.astype('float32'), [15,]), relay.reshape(const_1864.astype('float32'), [15,]), ), 0)
call_1865 = relay.TupleGetItem(func_1537_call(relay.reshape(const_1864.astype('float32'), [15,]), relay.reshape(const_1864.astype('float32'), [15,]), ), 0)
bop_1886 = relay.less(call_1839.astype('bool'), relay.reshape(const_1840.astype('bool'), relay.shape_of(call_1839))) # shape=(2, 5, 4)
bop_1889 = relay.less(call_1841.astype('bool'), relay.reshape(const_1840.astype('bool'), relay.shape_of(call_1841))) # shape=(2, 5, 4)
bop_1890 = relay.logical_and(bop_1826.astype('bool'), relay.reshape(call_1807.astype('bool'), relay.shape_of(bop_1826))) # shape=(8, 9)
bop_1893 = relay.logical_and(bop_1829.astype('bool'), relay.reshape(call_1808.astype('bool'), relay.shape_of(bop_1829))) # shape=(8, 9)
output = relay.Tuple([bop_1830,call_1863,const_1864,bop_1886,bop_1890,])
output2 = relay.Tuple([bop_1833,call_1865,const_1864,bop_1889,bop_1893,])
func_1897 = relay.Function([var_1813,], output)
mod['func_1897'] = func_1897
mod = relay.transform.InferType()(mod)
var_1898 = relay.var("var_1898", dtype = "float32", shape = (8, 9))#candidate|1898|(8, 9)|var|float32
output = func_1897(var_1898)
func_1899 = relay.Function([var_1898], output)
mutated_mod['func_1899'] = func_1899
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1059_call = mod.get_global_var('func_1059')
func_1061_call = mutated_mod.get_global_var('func_1061')
call_1901 = func_1059_call()
call_1902 = func_1059_call()
func_513_call = mod.get_global_var('func_513')
func_516_call = mutated_mod.get_global_var('func_516')
const_1905 = relay.const([True,False,True,True,False,False,True,False,True,True,True,True,False,False,False,False,False,True,False,False,False,True,True,False,True,False,True,True,False,True,False,False,False,True,True,False,True,False,True,True,True,False,False,True,False,False,False,True,True,False,True,False,True,False,False,True,False,True,False,False,True,False,False,True,False,True,True,True,False,False,False,True,True,True,True,True,True,True,False,True,True,True,True,True,True,True,False,True,True,False,False,False,True,False,True,True,False,False,False,False,True,True,True,False,False,False,True,True,True,False,False,True,True,True,False,True,True,False,False,False,False,False,False,True,True,True,False,True,True,True,True,False,False,False,False,False,True,False,True,False,False,False,True,True,True,True,False,False,True,True,False,False,True,False,True,False,True,False,True,True,True,False,True,True,True,False,True,False,False,False,False,False,False,False,True,False,True,False,False,True,True,True,True,False,True,False,True,True,False,False,False,True,True,True,False,False,True,True,False,True,False,False,True,False,False,True,False,False,True,True,True,False,False,False,False,True,False,True,False,True,False,False,False,True,False,True,True,True,False,True,True,False,True,False,True,False,True,True,False,False,False,True,False,False,True,True,False,True,False,False,False,False,True,True,True,True,False,False,False,False,True,True,True,True,False,False,True,True,True,True,False,False,False,False,False,True,True,False,False,False,True,True,False,False,False,False,True,False,False,True,False,False,True,True,True,True,True,False,False,True,True,False,True,True,True,True,False,False,False,False,False,True,True,True,True,True,True,False,True,True,True,False,True,False,True,False,True,True,False,False,False,True,False,True,False,True,False,False,False,False,True,False,True,False,True,False,False,False,False,True,False,False,True,True,False,True,False,False,False,False,False,False,False,False,True,False,False,True,True,False,False,False,False,True,False,False,False,False,False,False,False,False,False,True,False,True,False,False,True,False,False,True,False,False,True,True,True,True,True,False,True,True,False,False,False,False,False,False,True,False,True,True,True,True,False,True,False,True,True,True,True,True,False,False,False,False,True,True,True,False,True,True,True,True,True,False,True,True,False,False,True,True,True,False,True,False,True,True,True,False,True,False,False,True,False,False,True,False,True,True,False,False,False,False,True,True,False,False,False,False,True,True,False,False,False,True,False,True,False,True,False,True,False,False,True,False,False,False,False,True,True,True,True,True,True,False,False,True,False,False,True,True,False,True,False,False,False,False,True,True,False,True,False,False,True,True,False,True,False,True,False,True,False,True,True,True,True,True,False,False,False,True,True,False,True,True,False,True,False,True,False,False,True,True,False,False,False,False,False,False,True,False,False,True,False,True,False,True,False,False,True,False,False,True,False,True,False,False,False,False,False,True,False,True,False,False,True,False,True,True,True,False,False,True,False,False,False,False,True,False,True,True,True,True,False,True,False,False,False,False,True,False,True,True,True,True,False,False,False,True,False,True,True,False,False,True,False,False,False,True,False,True,False,True,False,False,True,False,True,True,False,False,True,False,True,False,True,True,False,True,False,False,False,False,False,False,True,False,False,True,False,True,False,False,False,False,True,True,False,False,False,True,False,True,True,True,True,False,True,False,False,True,False,False,True,True,True,True,True,True,False,True,False,True,False,True,True,True,False,False,True,True,True,True,False,False,False,False,False,True,False,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,True,False,True,True,False,False,False,True,True,True,True,False,True,False,True,True,False,False,True,True,True,True,True,False,False,True,False,False,True,True,True,False,False,False,True,False,True,False,False,False,True,True,False,False,True,False,True,False,False,False,True,True,True,True,False,True,False,False,False,True,False,True,True,False,True,False,False,False,False,False,True,True,False,True,True,True,True,False,True,False,True,True,True,False,False,True,False,False,False,False,True,False,True,False,False,False,False,False,True,True,True,False,False,True,True,False,False,True,False,False,True,True,True,False,False,True,False,False,True,False,False,False,False,True,False,False,True,False,False,False,True,True,False,False,False,False,False,True,True,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,True,True,False,False,True,False,True,True,False,False,False,False,True,False,False,True,True,False,False,True,False,False,False,False,False,False,True,False,True,False,False,False,True,False,False,False,False,True,False,False,True,False,True,False,True,True,True,False,False,True,True,False,True,True,True,True,False,False,True,True,False,False,True,True,False,True,False,True,True,False,True,False,False,False,True,True,False,False,True,False,False,True,False,True,False,True,True,False,False,False,True,False,False,False,False,True,False,False,False,True,False,True,True,False,False,False,False,False,True,True,False,True,False,False,False,True,False,True,True,False,False,True,False,False,True,False,True,True,False,True,False,False,True,False,True,True,False,True,True,True,False,True,False,True,True,True,False,True,False,True,True,False,False,False,True,False,False,False,True,True,False,False,True,True,False,False,False,False,True,True,False,False,False,False,True,True,True,True,True,False,False,False,False,True,False,True,True,True,False,False,False,False,False,True,True,False,False,True,True,True,False,True,True,False,True,False,True,True,True,False,True,False,True,True,True,True,True,False,False,True,False,False,True,True,False,False,True,False,True,False,True,True,True,True,True,True,True,False,False,True,True,True,True,False,False,False,True,True,False,False,False,False,False,True,True,False,True,True,False,False,True,False,True,True,True,False,False,False,True,False,True,False,True,False,False,True,False,True,True,False,True,True,True,False,True,False,False,False,True,True,True,True,True,False,False,False,True,False,False,False,False,False,False,True,False,False,False,True,True,True,True,True,True,False,True,False,True,True,False,False,False,True,True,True,False,True,True,True,True,False,True,True,False,True,False,False,True,True,True,False,False,True,True,True,True,False,True,False,False,False,True,False,True,False,False,True,False,True,False,True,True,True,True,True,False,True,False,True,False,True,False,True,False,True,False,False,False,False,False,False,False,True,False,True,False,False,True,False,False,True,True,True,True,False,True,True,False,True,True,False,False,True,True,True,False,False,True,False,False,True,False,False,True,True,True,True,True,False,True,True,False,False,True,True,True,False,False,True,True,True,True,True,False,True,False,True,True,True,True,True,True,False,True,False,True,False,True,False,True,False,False,True,False,True,False,True,True,False,False,False,False,False,True,False,True,False,True,False,True,True,True,False,True,False,False,False,False,True,False,True,True,False,False,False,True,False,True,False,True,False,False,True,False,False,True,True,False,True,True,True,False,False,False,True,False,False,True,False,False,True,False,True,False,False,True,True,True,True,False,False,True,False,True,True,False,False,False,False,True,True,False,True,True,False,False,True,False,True,True,False,True,True,False,True,False,True,False,True,False,False,True,True,True,False,False], dtype = "bool")#candidate|1905|(1430,)|const|bool
call_1904 = relay.TupleGetItem(func_513_call(relay.reshape(const_1905.astype('bool'), [13, 11, 10])), 0)
call_1906 = relay.TupleGetItem(func_516_call(relay.reshape(const_1905.astype('bool'), [13, 11, 10])), 0)
output = relay.Tuple([call_1901,call_1904,const_1905,])
output2 = relay.Tuple([call_1902,call_1906,const_1905,])
func_1909 = relay.Function([], output)
mod['func_1909'] = func_1909
mod = relay.transform.InferType()(mod)
output = func_1909()
func_1910 = relay.Function([], output)
mutated_mod['func_1910'] = func_1910
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_1934 = relay.TupleGetItem(func_1909_call(), 2)
call_1935 = relay.TupleGetItem(func_1910_call(), 2)
uop_1938 = relay.atanh(call_1934.astype('float32')) # shape=(1430,)
uop_1940 = relay.atanh(call_1935.astype('float32')) # shape=(1430,)
output = relay.Tuple([uop_1938,])
output2 = relay.Tuple([uop_1940,])
func_1941 = relay.Function([], output)
mod['func_1941'] = func_1941
mod = relay.transform.InferType()(mod)
mutated_mod['func_1941'] = func_1941
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1941_call = mutated_mod.get_global_var('func_1941')
call_1942 = func_1941_call()
output = call_1942
func_1943 = relay.Function([], output)
mutated_mod['func_1943'] = func_1943
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1962 = relay.const([[5.985726,-4.001117,-2.611560,3.842140,-5.858476,-3.763765,-8.673850,-6.587163,3.423369,-0.629330,-6.799038,-6.608814],[0.478382,-4.021786,9.831193,-5.311804,7.527767,1.085855,2.533821,4.412092,-6.619537,8.864847,6.889503,8.631892],[-0.584257,0.971275,7.698068,5.551730,4.133937,-3.387128,0.762589,4.135783,4.871050,-3.079125,5.172789,5.396063],[-4.059098,-9.939904,-6.135813,-3.065647,-9.011144,-7.260725,9.402936,-8.289275,2.464618,5.434453,5.230354,-4.570181],[8.088617,-9.924471,-3.801569,0.171338,-8.609942,-3.956242,6.477287,9.947716,5.078337,5.932361,8.740849,6.109897],[-9.132903,-9.933687,-5.089034,6.608997,3.236587,-6.583773,4.220135,-1.044434,-8.967754,-9.381725,-6.593960,6.026756],[9.955712,-0.740727,-4.903995,-5.006984,4.737211,4.504794,6.480335,9.597106,-1.678386,-2.952465,6.697751,6.781813],[9.698372,0.646630,8.169371,3.832519,6.466179,-3.999892,-0.571525,3.728984,1.829167,4.442873,4.651148,5.981261],[-6.221044,7.094331,7.101639,5.952569,-2.014869,-1.069331,9.831309,8.158286,4.615273,1.289184,-0.521952,8.343459],[0.377736,-9.375512,-3.696884,-5.609874,5.921939,-3.298818,-9.292534,7.155192,0.841943,-4.399441,-1.776099,8.382396],[3.999335,2.815652,7.099293,6.007119,5.845061,7.412685,-5.944840,-2.691706,3.540627,-0.708862,8.810807,4.977046]], dtype = "float64")#candidate|1962|(11, 12)|const|float64
uop_1963 = relay.sigmoid(const_1962.astype('float64')) # shape=(11, 12)
bop_1965 = relay.maximum(uop_1963.astype('uint32'), relay.reshape(const_1962.astype('uint32'), relay.shape_of(uop_1963))) # shape=(11, 12)
bop_1970 = relay.divide(const_1962.astype('float64'), relay.reshape(uop_1963.astype('float64'), relay.shape_of(const_1962))) # shape=(11, 12)
bop_1975 = relay.floor_divide(const_1962.astype('float64'), relay.reshape(bop_1965.astype('float64'), relay.shape_of(const_1962))) # shape=(11, 12)
func_224_call = mod.get_global_var('func_224')
func_229_call = mutated_mod.get_global_var('func_229')
const_1979 = relay.const(-3.226422, dtype = "float32")#candidate|1979|()|const|float32
var_1980 = relay.var("var_1980", dtype = "float32", shape = (448,))#candidate|1980|(448,)|var|float32
const_1981 = relay.const([True,True,False,False,False,True,True,False,False,True,True,True,True,False,False,True,False,False,True,True,False,False,False,False,False,False,True,True,False,True,True,False,False,True,False,True,False,False,True,False,True,False,False,False,False,False,True,True,False,True,False,True,True,False,True,True,False,False,True,True,False,False,True,False,False,True,True,False,True,False,True,False,True,True,False,True,True,False,False,True,True,False,False,False,False,True,True,True,False,False,False,False,True,True,False,True,False,False,True,False,False,False,False,True,True,True,True,False,True,True,False,False,False,False,False,False,False,True,True,False,False,True,False,True,False,False,False,True,False,True,True,True,False,True,False,True,False,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,False,False,False,True,True,False,False,True,True,True,False,True,False,True,True,True,False,True,False,True,True,False,False,True,True,False,False,False,True,False,True,True,True,True,False,False,True,False,True,True,True,False,False,False,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,False,True,True,False,False,False,False,False,False,True,False,True,False], dtype = "bool")#candidate|1981|(224,)|const|bool
call_1978 = relay.TupleGetItem(func_224_call(relay.reshape(const_1979.astype('float32'), []), relay.reshape(var_1980.astype('float32'), [2, 14, 16]), relay.reshape(const_1981.astype('bool'), [224,]), ), 0)
call_1982 = relay.TupleGetItem(func_229_call(relay.reshape(const_1979.astype('float32'), []), relay.reshape(var_1980.astype('float32'), [2, 14, 16]), relay.reshape(const_1981.astype('bool'), [224,]), ), 0)
output = relay.Tuple([bop_1970,bop_1975,call_1978,const_1979,var_1980,const_1981,])
output2 = relay.Tuple([bop_1970,bop_1975,call_1982,const_1979,var_1980,const_1981,])
func_1983 = relay.Function([var_1980,], output)
mod['func_1983'] = func_1983
mod = relay.transform.InferType()(mod)
var_1984 = relay.var("var_1984", dtype = "float32", shape = (448,))#candidate|1984|(448,)|var|float32
output = func_1983(var_1984)
func_1985 = relay.Function([var_1984], output)
mutated_mod['func_1985'] = func_1985
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2010 = relay.var("var_2010", dtype = "int16", shape = ())#candidate|2010|()|var|int16
var_2011 = relay.var("var_2011", dtype = "int16", shape = (13, 10, 12))#candidate|2011|(13, 10, 12)|var|int16
bop_2012 = relay.maximum(var_2010.astype('int16'), var_2011.astype('int16')) # shape=(13, 10, 12)
bop_2017 = relay.bitwise_or(bop_2012.astype('uint16'), var_2010.astype('uint16')) # shape=(13, 10, 12)
func_1492_call = mod.get_global_var('func_1492')
func_1497_call = mutated_mod.get_global_var('func_1497')
const_2023 = relay.const([6.924692,-8.817718,3.190450,6.096025,-1.487563,-9.195150,1.244120,-5.526667,2.907095,0.644344,7.530753,-1.401784,-5.285533,4.867913,1.267883,5.327577,4.496293,2.315071,-0.903574,-5.444391,-9.056262,-3.131839,2.904553,-9.671494,0.403944,5.475605,-4.263511,-1.964626,0.081432,-0.398783,3.451465,-7.109072,6.476972,5.068511,-4.247962,8.351227,-8.549265,-2.148750,-0.513601,2.474038,-4.500185,0.474599,-3.971808,-1.369162,1.210565,-9.027105,-1.181264,5.488113,7.632276,2.471996,6.294260,-8.019645,-2.243215,8.815249,-0.491011,-7.418608,-4.934322,1.528358,7.824860,-5.744404,4.529352,0.829318,2.529395,2.018114,-7.886393,-8.000393,6.098817,7.883758,0.046584,4.774958,5.205437,3.111877,3.320337,4.201243,-2.032263,7.331270,-5.084209,2.228909,1.345618,-0.524926,6.633626,7.217128,-1.180194,-9.154155,-8.764801,8.914500,3.724019,4.710156,5.957828,9.774152,-7.396519,0.527431,3.100232,7.204654,-0.571122,-3.143098,-6.679763,-4.899696,9.749015,9.515689,-5.135826,2.013764,5.231828,-3.425166,0.822223,-0.018864,-1.519181,0.080874,-3.821205,4.308311,-5.713007,-5.008367,4.054448,6.047245,2.070569,-2.846091,8.145192,8.724376,-5.904942,-2.621818,6.613681,-6.333173,-5.501735,3.858690,-6.417146,6.802940,1.176373,7.430183,0.306096,6.209175,2.440660,-9.810964,2.225046,8.072267,-2.161595,-3.918831,-8.833214,9.566586,-1.106243,0.849542,1.054410,1.886267,-0.281926,-8.759102,-3.778004,-6.330434,3.965285,-6.872781,-7.864054,7.736659,0.919378,-1.968106,-7.477166,7.850885,1.330460,2.225137,2.946584,3.047335,0.648068,6.233868,3.098069,-7.627386,3.467735,1.448548,-3.486655,-9.559120,-8.594309,-8.439743,-8.793318,4.023982,-1.826521,5.470949,-6.572914,-2.165215,-2.926796,9.394244,6.654795,-2.075162,-1.372611,5.124003,-8.544054,7.935179,-5.847942,7.973966,8.233671,-0.181913,5.953922,9.201814,-8.133344,-8.275994,-1.861904,-3.950285,1.243858,-2.862786,4.723402], dtype = "float32")#candidate|2023|(195,)|const|float32
const_2024 = relay.const([7.419680,-2.118670,9.466674,0.475705,-7.721876,-4.207340,-4.942729,6.665146,-5.517534,-2.158669,0.707381,1.860622,-5.539642,-8.087540,0.411201,-8.365679,7.441516,1.991839,9.683964,7.046996,-6.361562,8.514683,-9.623030,9.285291,-6.073016,-3.415568,-0.710182,-9.073054,-9.649397,-5.255087,-5.426346,-4.522646,4.155316,-1.539325,-1.908851,5.009929,1.442995,2.207913,7.086500,-5.178589,6.395900,6.661692,-1.831676,-6.083793,3.518508,1.955091,1.601900,7.047007,8.442486,-7.014218,0.769396,-9.522141,-7.841523,0.057577,-3.347067,6.051006,-6.712155,9.092471,2.726268,2.242301,-1.650263,-4.493383,9.019093,0.438712,5.781773,-8.843643,5.963681,6.207257,8.016890,5.663561,-1.309021,0.559679,-9.918280,2.932999,-3.393936,-9.926364,2.989732,7.227586,-3.223777,-3.203644,-4.123731,-1.374380,2.178480,0.822684,-4.140099,8.825744,0.818734,-4.117498,3.030291,9.912408,-8.265714,3.118100,-5.119031,-9.364068,3.104617,-5.760514,-2.927782,7.882368,-8.171204,-1.963981,5.661791,9.915003,-6.038667,-0.307699,-9.011032,8.534832,3.245178,-0.113764,0.559468,-3.629119,7.008602,-0.125333,-9.894528,-7.503924,-0.257346,8.253966,-5.431737,2.988896,-3.863405,-8.876625,0.118940,-8.355143,-9.217961,5.223761,4.359526,-4.123506,-5.370719,-7.538881,-1.081839,-6.955005,7.452071,0.140260,-0.574887,-7.117180,-1.875274,1.720351,2.319739,8.042018,5.375869,0.984902,-2.897726,5.247525,8.226057,5.193325,-0.153840,-1.445954,-4.697644,7.413542,-5.563288,6.578159,8.140752,4.269353,-7.894812,-8.440408,6.173583,2.780435,-1.201110,-0.962535,-2.944523,2.495610,5.721677,1.047127,3.428697,5.673179,-5.832690,5.671794,9.729868,-5.655509,2.545128,1.619316,1.382231,-0.855437,-9.173318,8.775816,1.148473,5.561173,-2.195991,-3.300263,-7.446348,-8.101117,-2.198769,-3.829498,-1.135685,-8.278956,1.590042,-1.594606,-5.444030,-7.293566,6.915123,8.162361,7.862548,-4.912576,3.180504,4.401422,-8.144891,2.985842,2.974189,-3.473354,-2.104387,8.088574,0.195659,-1.295632,8.076360,-8.271289,-4.396886,6.622671,-6.361662,-7.895185,-8.251905,2.264123,2.815073,8.442273,-9.226129,-1.939417,-9.158042,2.141523,9.878034,1.771104,-5.970973,7.446864,-9.983371,-9.137408,-9.419865,-7.817568,8.359840,8.098610,-5.103055,9.328291,3.773192,-8.340873,-5.782881,5.998703,5.228439,-8.788206,-4.299816,6.659259,-8.269664,3.976112,-3.751186,6.993378,-9.920826,-3.034249,0.343071,9.237889,4.258947,9.669117,0.898613,4.383697,5.701767,0.841882,4.850473,-1.132215,-8.708155,1.570081,-6.691054,-5.363867,1.261792,4.353666,4.778402,9.301599,-9.141939,-7.296219,1.545314,1.769895,2.077107,-7.468894,1.771799,-7.857247,1.370342,-9.928368,-8.087556,-7.164261,1.246418,-4.535623,-9.088724,3.823367,-7.441012,4.694127,2.645536,9.688869,7.332497,9.894629,-6.642939,-9.535457,4.824266,8.573136,-8.705154,-6.747728,-8.333988,-5.345955,-9.614277,4.383751,5.281808,8.074011,6.726861,4.538771,-7.311301,7.897436,-2.164215,-0.879674,-0.915415,7.713652,3.034549,-8.881139,-7.171169,0.422227,-9.326222,-4.124741,2.100088,-0.396914,-8.115395,4.043320,6.503425,2.161879,-3.861969,8.081149,-8.439671,2.999060,-7.302478,5.079078,-7.646200,-8.501626,9.044799,-0.750355,-7.174766,-8.657751,-8.087456,9.334917,-1.404227,3.838004,3.219585,-1.405482,5.515357,4.158760,7.349416,2.493613,2.856467,5.307434,-2.356521,6.250375,-7.831486,2.605370,5.561014,-9.528575,1.261710,7.891052,5.471797,-6.641316,4.669324,-9.122431,-6.723802,7.484113,1.828995,1.706866,-2.541768,-8.686109,-5.788187,-8.542259,9.786997,1.233465,-4.012614,1.991170,-6.629652,8.436283,8.713810,9.025613,5.039618,-1.441042,1.450748,-5.342473,-4.256583,-9.925953,-3.757874,1.328695,-7.192966,-7.992535,4.601550,-1.173118,-1.655425,7.886569,0.494009,7.929997,5.552088,-0.179322,8.590953,5.341772,8.497288,-1.398462,-4.293453,3.653175,5.887352,9.148203,6.418490,-5.344763,7.656523,-4.412759,-0.165132,6.047342,5.315628,-2.132013,-5.918076,-8.554152,-5.095946,2.668706,-0.299140,4.801394,7.433712,-5.551369,-1.049438,-1.536677,9.922035,-0.743130,-6.171818,9.710293,8.592647,5.176022,7.234213,-8.595978,-7.432028,-4.538618,-5.669300,-7.479939,1.755188,2.135002,-7.046259,7.253140,-7.228723,1.504670,-6.636725,-8.420367,3.314024,1.194680,6.842054,-4.745925,-6.726243,7.624733,-3.615911,-3.824573,-3.168950,0.276863,4.343850,-6.788477,-4.761560,-3.325844,0.788083,7.238545,-4.046892,1.910701], dtype = "float32")#candidate|2024|(448,)|const|float32
call_2022 = relay.TupleGetItem(func_1492_call(relay.reshape(const_2023.astype('float32'), [15, 13]), relay.reshape(var_2010.astype('float32'), []), relay.reshape(const_2024.astype('float32'), [8, 56]), ), 1)
call_2025 = relay.TupleGetItem(func_1497_call(relay.reshape(const_2023.astype('float32'), [15, 13]), relay.reshape(var_2010.astype('float32'), []), relay.reshape(const_2024.astype('float32'), [8, 56]), ), 1)
bop_2026 = relay.not_equal(var_2010.astype('bool'), const_2024.astype('bool')) # shape=(448,)
bop_2031 = relay.logical_xor(const_2023.astype('uint64'), var_2010.astype('uint64')) # shape=(195,)
func_1534_call = mod.get_global_var('func_1534')
func_1537_call = mutated_mod.get_global_var('func_1537')
const_2035 = relay.const([[0.199176,0.195443,7.948617,6.023999,-6.316004,6.787010,0.225798,-3.191829,7.945459,-7.259916,-5.478868,0.786867,3.730398,-0.124171,4.603089]], dtype = "float32")#candidate|2035|(1, 15)|const|float32
call_2034 = relay.TupleGetItem(func_1534_call(relay.reshape(const_2035.astype('float32'), [15,]), relay.reshape(const_2035.astype('float32'), [15,]), ), 0)
call_2036 = relay.TupleGetItem(func_1537_call(relay.reshape(const_2035.astype('float32'), [15,]), relay.reshape(const_2035.astype('float32'), [15,]), ), 0)
uop_2037 = relay.tan(const_2035.astype('float32')) # shape=(1, 15)
output = relay.Tuple([bop_2017,call_2022,bop_2026,bop_2031,call_2034,uop_2037,])
output2 = relay.Tuple([bop_2017,call_2025,bop_2026,bop_2031,call_2036,uop_2037,])
func_2040 = relay.Function([var_2010,var_2011,], output)
mod['func_2040'] = func_2040
mod = relay.transform.InferType()(mod)
mutated_mod['func_2040'] = func_2040
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2040_call = mutated_mod.get_global_var('func_2040')
var_2042 = relay.var("var_2042", dtype = "int16", shape = ())#candidate|2042|()|var|int16
var_2043 = relay.var("var_2043", dtype = "int16", shape = (13, 10, 12))#candidate|2043|(13, 10, 12)|var|int16
call_2041 = func_2040_call(var_2042,var_2043,)
output = call_2041
func_2044 = relay.Function([var_2042,var_2043,], output)
mutated_mod['func_2044'] = func_2044
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1059_call = mod.get_global_var('func_1059')
func_1061_call = mutated_mod.get_global_var('func_1061')
call_2057 = func_1059_call()
call_2058 = func_1059_call()
output = relay.Tuple([call_2057,])
output2 = relay.Tuple([call_2058,])
func_2099 = relay.Function([], output)
mod['func_2099'] = func_2099
mod = relay.transform.InferType()(mod)
mutated_mod['func_2099'] = func_2099
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2099_call = mutated_mod.get_global_var('func_2099')
call_2100 = func_2099_call()
output = call_2100
func_2101 = relay.Function([], output)
mutated_mod['func_2101'] = func_2101
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1059_call = mod.get_global_var('func_1059')
func_1061_call = mutated_mod.get_global_var('func_1061')
call_2136 = func_1059_call()
call_2137 = func_1059_call()
func_1315_call = mod.get_global_var('func_1315')
func_1316_call = mutated_mod.get_global_var('func_1316')
call_2141 = func_1315_call()
call_2142 = func_1315_call()
func_598_call = mod.get_global_var('func_598')
func_603_call = mutated_mod.get_global_var('func_603')
const_2149 = relay.const(-9, dtype = "uint32")#candidate|2149|()|const|uint32
var_2150 = relay.var("var_2150", dtype = "float32", shape = (5, 44))#candidate|2150|(5, 44)|var|float32
const_2151 = relay.const([2.068291,-4.721829,-3.866253,-1.049356,-4.914786,3.815899,-4.283007,6.651379,9.163850,-3.897861,-7.049143,9.048523,4.498127,8.413700,-2.858682,-1.981975,5.916510,-9.974508,-5.797250,1.342935,2.858002,-5.837570,-2.046941,-0.536794,-8.664205,3.807811,9.830346,7.966924,5.168543,0.136867,-4.180111,2.335789,9.292992,1.354315,4.236484,-0.768653,8.475426,-1.314788,-8.818040,-8.135558,5.719153,0.074047,-8.869515,-4.768039,-5.317619,-7.997948,3.135026,-4.632279,9.355452,-5.543876,-8.488029,-2.235267,3.830370,5.244423,8.132341,-2.202694,5.025479,3.594641,5.269441,3.833378,-6.475684,-8.374345,9.342691,-5.744170,-7.730225,5.290551,9.114479,2.845570,3.409226,-8.753550,-8.366357,5.402041,-4.432639,5.588767,-9.954843,5.988884,-4.239577,-5.884203,-9.563174,3.270401,-0.073035,2.443775,-3.262436,-7.519282,4.039917,-7.524957,-0.829077,0.221849,-8.170685,-3.229393,4.424685,1.310308,5.019998,-4.372778,-9.547342,-4.759703,-7.814630,-2.586350,-9.257883,-5.722028,6.071603,3.758753,0.166305,7.975675,-8.238175,-6.784473,0.669067,7.720445,-6.932672,6.159901,8.852133,8.135435,3.860886,-6.473606,-1.373605,-2.207918,9.314716,9.072776,-6.168052,0.101453,-4.344638,0.015297,2.049948,-1.817691,-2.759710,0.085490,-0.854163,4.896608,-6.957546,9.664252,9.188156,4.479812,-6.783460,0.948340,0.631769,5.710231,-7.315896,4.475860,5.579121,7.912178,8.163248,-1.441038,7.393611,-2.964323,5.428711,5.689074,1.919267,-8.373565,6.039286,-3.637634,-4.384266,5.539619,-1.182181,-1.047707,-7.743305,5.704733,-7.616886,9.393431,-7.028687,-6.008737,-4.974556,-2.236810,-9.756698,4.450303,-0.747463,-3.347912,5.334694,3.436297,-1.453165,8.489754,-7.881954,4.875491,-2.150801,-1.888693,-3.211150,4.151656,-4.149876,-6.539632,-0.481030,1.022185,8.761962,1.423297,7.480039,7.058994,-6.849706,-8.372239,2.826343,-2.128539,6.865196,-6.664529,8.285979,-3.236903,0.858247,-0.088146,-8.507776,-5.008266,5.310131,-6.795810,1.004146,-3.305087,-6.525562,-3.165072,-1.258667,-7.777842,5.048094,4.332217,-7.679977,8.666098,-3.196383,2.142773,2.589695,0.754835,6.082500,5.664783,2.688823,-1.825537,-1.331928,2.064644,9.888983,-1.275615,8.769528,4.954358,6.866194,9.449846,-9.487608,-1.716449,9.761643,-1.987295,0.675834,7.847361,-2.423406,0.620159,-5.777713,2.625089,-4.587924,-6.345031,-5.996288,5.759603,-0.768170,7.006599,3.494133,-2.396993,2.893219,4.344637,-2.199833,3.509637,-4.420137,-7.877344,-1.953627,8.646970,2.256993,-8.627551,0.696282,-5.680877,-6.254390,4.183204,7.943779,-8.477577,-1.660493,-3.571945,-6.291966,6.419991,-1.969954,4.026752,-2.829572,1.620590,1.990379,5.597582,7.937001,-9.436006,5.813516,9.464369,-7.557182,-6.679556,2.500771,4.541784,-1.444497,-2.497501,-7.946741,1.265091,-8.356430,3.801931,-1.584353,8.427716,-6.846607,-9.858725,1.656280,-8.097365,0.250396,-9.505050,7.609770,4.378879,7.997579,-1.105016,1.096016,-0.485130,-4.404573,6.009375,-9.152595,-1.082769,0.852521,-9.978169,-1.383135,-9.025468,7.399088,-8.658017,-3.615234,4.912509,-0.163516,-1.595708,3.243168,4.873258,-1.379869,2.298835,0.323931,3.190137,-9.824060,9.640507,-2.424632,-9.250793,2.419163,8.880576,-2.627108,2.467265,-0.498867,7.313396,-2.945507,-5.911389,6.695205,4.430032,8.562275,4.967967,-1.886339,4.531957,-5.218047,7.118663,-6.198294,-3.927623,6.527069,4.430489,1.256917,9.461659,-8.590057,-3.727733,5.620076,-9.838368,-6.954756,-0.816319,0.031696,4.009420,-0.075427,4.339042,4.661869,0.469489,7.993555,5.137343,5.081674,-2.001320,-3.026813,-4.557843,-3.010557,-3.743309,-7.664563,8.050832,-1.774343,-2.488664,0.106978,-1.852804,6.699225,1.339975,-5.932374,6.628425,-2.416591,8.705189,-7.354690,-0.199262,6.920528,9.493378,-4.037616,1.145870,8.849232,8.823007,9.826898,-2.142949,1.512537,6.894656,-2.554732,-8.647193,-0.754254,0.147576,3.152352,0.427955,-7.439370,-8.015857,4.968924,-0.156144,4.592157,-0.328716,3.751170,6.591836,-1.723618,-0.617828,6.865843,6.985130,0.824801,-1.855028,9.752213,6.193614,-0.359362,-3.646566,4.018543,1.131520,-1.229397,3.848898,-3.454699,-4.757916,7.754282,0.527395,7.894417,-8.152502,0.731377,3.526617,-0.475973,-2.263598,7.925345,-6.441945,9.711231,7.570726,1.072427,0.801813,-6.219702,3.120448,4.125485,-4.247066,0.310957,4.097654,-5.136572,1.903966,4.837304,0.116356,-5.866559,6.944958,3.975372,3.221269,1.252615,-3.320273,5.960013,2.293482], dtype = "float32")#candidate|2151|(448,)|const|float32
call_2148 = relay.TupleGetItem(func_598_call(relay.reshape(const_2149.astype('uint32'), []), relay.reshape(var_2150.astype('float32'), [10, 2, 11]), relay.reshape(const_2151.astype('float32'), [448,]), ), 1)
call_2152 = relay.TupleGetItem(func_603_call(relay.reshape(const_2149.astype('uint32'), []), relay.reshape(var_2150.astype('float32'), [10, 2, 11]), relay.reshape(const_2151.astype('float32'), [448,]), ), 1)
output = relay.Tuple([call_2136,call_2141,call_2148,const_2149,var_2150,const_2151,])
output2 = relay.Tuple([call_2137,call_2142,call_2152,const_2149,var_2150,const_2151,])
func_2153 = relay.Function([var_2150,], output)
mod['func_2153'] = func_2153
mod = relay.transform.InferType()(mod)
mutated_mod['func_2153'] = func_2153
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2154 = relay.var("var_2154", dtype = "float32", shape = (5, 44))#candidate|2154|(5, 44)|var|float32
func_2153_call = mutated_mod.get_global_var('func_2153')
call_2155 = func_2153_call(var_2154)
output = call_2155
func_2156 = relay.Function([var_2154], output)
mutated_mod['func_2156'] = func_2156
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2177 = relay.var("var_2177", dtype = "float32", shape = (3, 7, 16))#candidate|2177|(3, 7, 16)|var|float32
var_2178 = relay.var("var_2178", dtype = "float32", shape = (3, 7, 16))#candidate|2178|(3, 7, 16)|var|float32
bop_2179 = relay.power(var_2177.astype('float32'), relay.reshape(var_2178.astype('float32'), relay.shape_of(var_2177))) # shape=(3, 7, 16)
output = bop_2179
output2 = bop_2179
func_2192 = relay.Function([var_2177,var_2178,], output)
mod['func_2192'] = func_2192
mod = relay.transform.InferType()(mod)
var_2193 = relay.var("var_2193", dtype = "float32", shape = (3, 7, 16))#candidate|2193|(3, 7, 16)|var|float32
var_2194 = relay.var("var_2194", dtype = "float32", shape = (3, 7, 16))#candidate|2194|(3, 7, 16)|var|float32
output = func_2192(var_2193,var_2194,)
func_2195 = relay.Function([var_2193,var_2194,], output)
mutated_mod['func_2195'] = func_2195
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_2199 = relay.TupleGetItem(func_1909_call(), 2)
call_2200 = relay.TupleGetItem(func_1910_call(), 2)
func_1315_call = mod.get_global_var('func_1315')
func_1316_call = mutated_mod.get_global_var('func_1316')
call_2211 = func_1315_call()
call_2212 = func_1315_call()
output = relay.Tuple([call_2199,call_2211,])
output2 = relay.Tuple([call_2200,call_2212,])
func_2217 = relay.Function([], output)
mod['func_2217'] = func_2217
mod = relay.transform.InferType()(mod)
mutated_mod['func_2217'] = func_2217
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2217_call = mutated_mod.get_global_var('func_2217')
call_2218 = func_2217_call()
output = call_2218
func_2219 = relay.Function([], output)
mutated_mod['func_2219'] = func_2219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2223 = relay.var("var_2223", dtype = "float32", shape = (7, 2, 15))#candidate|2223|(7, 2, 15)|var|float32
uop_2224 = relay.asinh(var_2223.astype('float32')) # shape=(7, 2, 15)
func_546_call = mod.get_global_var('func_546')
func_548_call = mutated_mod.get_global_var('func_548')
call_2230 = func_546_call()
call_2231 = func_546_call()
func_2217_call = mod.get_global_var('func_2217')
func_2219_call = mutated_mod.get_global_var('func_2219')
call_2235 = relay.TupleGetItem(func_2217_call(), 0)
call_2236 = relay.TupleGetItem(func_2219_call(), 0)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_2240 = relay.TupleGetItem(func_1909_call(), 1)
call_2241 = relay.TupleGetItem(func_1910_call(), 1)
func_855_call = mod.get_global_var('func_855')
func_862_call = mutated_mod.get_global_var('func_862')
var_2245 = relay.var("var_2245", dtype = "bool", shape = (224,))#candidate|2245|(224,)|var|bool
const_2246 = relay.const([-0.373349,-3.726161,-6.763228,-2.797608,-2.033159,9.135334,-2.406138,-2.565357,-7.497706,8.438868,-8.327182,-7.176968,8.327745,8.067908,-2.162041,4.845665,-4.493104,-1.219467,3.440264,6.702533,-4.487400,5.232152,-4.631540,9.101998,7.646780,4.612122,-0.774868,-5.498673,8.379053,-7.440557,-6.040290,5.786506,5.111997,9.870100,-1.314399,7.601217,-6.562826,2.665465,-1.562035,9.163107,-8.790865,1.508186,-0.663627,-4.143347,9.263114,0.493683,-2.217565,-3.230736,7.386522,-5.407647,-9.047269,-6.899078,-7.817590,0.677665,0.181558,-3.270935,1.671459,-0.311257,6.638743,-7.621947,4.271860,7.292161,-6.313118,-9.616535,2.898135,4.330797,-3.129439,-4.384876,5.482088,3.282056,-4.091083,-9.909216,-4.031696,-1.824337,-6.430915,-4.689468,-4.536785,2.079830,-9.328633,-6.564217,4.355076,8.681402,-3.936234,-2.137190,-2.385710,-3.609002,-6.210908,-6.991599,2.863056,-8.722469,3.041810,0.092268,-8.486954,-8.947700,-4.101476,2.536447,-6.489985,5.627516,-0.004321,1.051289,-2.350705,-1.746920,-9.271384,3.993694,5.338626,9.119647,-6.934974,-1.035971,-0.300206,6.903519,-7.779936,-9.143970,-0.769561,0.262759,1.160379,0.659430,-8.218458,-8.933587,-9.122717,-5.986910,-9.220606,9.276408,-4.366181,3.653591,-2.321948,-6.428631,2.699928,-9.086745,-1.445834,9.384110,-9.645767,4.387301,-2.133467,8.113139,7.481411,-6.241991,-8.933055,3.942904,1.371642,4.090984,3.492980,-7.336370,-9.358743,-6.397685,-6.496652,-8.530893,-3.732092,6.164389,-1.579455,-4.214317,-8.096153,6.842749,-7.828738,-7.914464,5.183813,3.323239,5.395667,-8.874965,-4.427213,-6.557485,-2.889594,8.800779,1.103781,7.889457,-4.077209,0.626374,-7.853612,8.834428,0.803755,-9.747809,7.688697,-1.572217,-6.891741,-7.776645,3.262756,9.677286,5.227042,4.080731,-7.564544,-0.517538,-5.991372,6.507370,-9.253627,5.131271,-6.057304,6.547997,6.688608,-3.132454,-1.191881,4.015832,-4.558975,2.971086,6.608803,-5.244533,6.113929,-3.873813,-7.499898,-8.545448,-5.689495,9.780696,7.858366,-6.455741,-6.062209,-0.092592,3.854590,-2.348645,0.804783,-0.956475,-3.501936,-9.200703,3.590889,4.175667,-4.938827,2.144534,5.134399,-8.200781,9.888549,-4.359011,5.719549,-0.741510,-0.772314,-3.135370,-9.737308,-9.124672,8.828607,-7.566991,-1.172291,2.100118,5.271676,3.318329,5.537388,6.887495,0.624745,-0.149568,7.876344,7.208149,2.227574,0.856011,-4.428835,0.610795,-7.005543,2.052688,5.527491,5.323728,0.871471,-3.808627,-5.297608,4.470207,3.568754,8.889806,-9.928954,-3.108189,-3.611759,4.190734,7.275291,4.948939,-3.183531,-2.421694,5.822599,-4.508300,5.060241,5.680405,-7.019828,8.742577,-0.605463,-5.150815,6.517357,-2.983033,-3.853384,5.558448,-4.352403,2.653972,8.961918,1.150897,-3.070567,5.423196,-4.875234,4.466135,6.955763,-4.805413,4.627951,-0.500375,4.885645,7.829890,-6.898747,8.971001,9.505893,9.799724,8.179657,9.798568,-9.265322,0.663701,7.852440,9.857801,5.516297,-3.416947,1.198071,0.260344,4.151565,-9.049831,9.680874,6.814497,0.010945,1.920631,2.360287,-0.814345,7.033182,-8.303209,-6.306716,2.188397,2.446849,2.483630,-6.362262,0.131562,7.007715,9.043812,9.613447,3.772652,7.416235,7.867306,6.458900,-7.859722,4.324267,8.687827,4.508397,7.605063,-2.164606,9.408956,-5.263727,-9.884459,-7.744484,-7.668126,-9.592236,-4.668544,4.188714,-8.474756,4.919560,6.132610,7.568097,5.960561,-2.683745,5.697289,6.026489,0.947075,4.112815,4.440410,-7.740341,-4.786263,5.661828,-7.856168,-3.345097,1.687404,-8.650836,3.459704,7.707420,-1.992645,5.368005,-2.548949,-3.980327,-1.139875,9.138537,7.112370,-3.136404,7.649153,7.598842,-6.570535,-2.834024,1.294593,-6.176257,7.092145,6.235341,8.644230,9.722142,1.929905,-8.981456,-2.494586,5.901316,1.614900,0.344044,2.576383,-5.978621,-5.727720,8.719329,1.447637,-1.504247,6.305555,-5.932371,6.097986,-1.501413,8.303136,-4.258620,5.942665,-2.000566,0.723588,5.038980,-9.490080,-7.271887,-4.801230,-6.046951,-6.839584,-1.372154,0.267534,-7.999977,3.991558,5.915751,8.792176,-0.664832,-8.171795,0.329450,-3.185716,-4.105637,6.949892,1.211620,9.777386,1.734709,9.777047,9.286842,0.742922,6.256536,1.523786,3.323792,-0.385889,3.220755,-5.310535,0.124306,7.256377,4.142262,-0.593690,-0.484490,1.836021,6.509188,-5.191285,8.729242,1.625977,-8.494662,0.203304,6.860641,1.384729,-0.465355,-3.364339,-2.592961,-5.453820,-7.781639,-2.141289,-8.490783,-5.446749,-0.008220,-7.056413], dtype = "float32")#candidate|2246|(448,)|const|float32
var_2247 = relay.var("var_2247", dtype = "float64", shape = (60,))#candidate|2247|(60,)|var|float64
call_2244 = relay.TupleGetItem(func_855_call(relay.reshape(var_2245.astype('bool'), [224, 1]), relay.reshape(const_2246.astype('float32'), [448,]), relay.reshape(const_2246.astype('float32'), [448,]), relay.reshape(var_2247.astype('float64'), [60,]), relay.reshape(const_2246.astype('int16'), [448,]), ), 6)
call_2248 = relay.TupleGetItem(func_862_call(relay.reshape(var_2245.astype('bool'), [224, 1]), relay.reshape(const_2246.astype('float32'), [448,]), relay.reshape(const_2246.astype('float32'), [448,]), relay.reshape(var_2247.astype('float64'), [60,]), relay.reshape(const_2246.astype('int16'), [448,]), ), 6)
output = relay.Tuple([uop_2224,call_2230,call_2235,call_2240,call_2244,var_2245,const_2246,var_2247,])
output2 = relay.Tuple([uop_2224,call_2231,call_2236,call_2241,call_2248,var_2245,const_2246,var_2247,])
func_2250 = relay.Function([var_2223,var_2245,var_2247,], output)
mod['func_2250'] = func_2250
mod = relay.transform.InferType()(mod)
mutated_mod['func_2250'] = func_2250
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2250_call = mutated_mod.get_global_var('func_2250')
var_2252 = relay.var("var_2252", dtype = "float32", shape = (7, 2, 15))#candidate|2252|(7, 2, 15)|var|float32
var_2253 = relay.var("var_2253", dtype = "bool", shape = (224,))#candidate|2253|(224,)|var|bool
var_2254 = relay.var("var_2254", dtype = "float64", shape = (60,))#candidate|2254|(60,)|var|float64
call_2251 = func_2250_call(var_2252,var_2253,var_2254,)
output = call_2251
func_2255 = relay.Function([var_2252,var_2253,var_2254,], output)
mutated_mod['func_2255'] = func_2255
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1711_call = mod.get_global_var('func_1711')
func_1713_call = mutated_mod.get_global_var('func_1713')
call_2286 = relay.TupleGetItem(func_1711_call(), 5)
call_2287 = relay.TupleGetItem(func_1713_call(), 5)
output = call_2286
output2 = call_2287
func_2306 = relay.Function([], output)
mod['func_2306'] = func_2306
mod = relay.transform.InferType()(mod)
output = func_2306()
func_2307 = relay.Function([], output)
mutated_mod['func_2307'] = func_2307
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1166_call = mutated_mod.get_global_var('func_1166')
call_2367 = relay.TupleGetItem(func_1165_call(), 0)
call_2368 = relay.TupleGetItem(func_1166_call(), 0)
output = call_2367
output2 = call_2368
func_2373 = relay.Function([], output)
mod['func_2373'] = func_2373
mod = relay.transform.InferType()(mod)
output = func_2373()
func_2374 = relay.Function([], output)
mutated_mod['func_2374'] = func_2374
mutated_mod = relay.transform.InferType()(mutated_mod)
func_504_call = mod.get_global_var('func_504')
func_505_call = mutated_mod.get_global_var('func_505')
call_2375 = relay.TupleGetItem(func_504_call(), 0)
call_2376 = relay.TupleGetItem(func_505_call(), 0)
output = call_2375
output2 = call_2376
func_2380 = relay.Function([], output)
mod['func_2380'] = func_2380
mod = relay.transform.InferType()(mod)
mutated_mod['func_2380'] = func_2380
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2380_call = mutated_mod.get_global_var('func_2380')
call_2381 = func_2380_call()
output = call_2381
func_2382 = relay.Function([], output)
mutated_mod['func_2382'] = func_2382
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1517_call = mod.get_global_var('func_1517')
func_1519_call = mutated_mod.get_global_var('func_1519')
call_2383 = relay.TupleGetItem(func_1517_call(), 1)
call_2384 = relay.TupleGetItem(func_1519_call(), 1)
output = call_2383
output2 = call_2384
func_2394 = relay.Function([], output)
mod['func_2394'] = func_2394
mod = relay.transform.InferType()(mod)
output = func_2394()
func_2395 = relay.Function([], output)
mutated_mod['func_2395'] = func_2395
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2422 = relay.var("var_2422", dtype = "int8", shape = (3, 16))#candidate|2422|(3, 16)|var|int8
var_2423 = relay.var("var_2423", dtype = "int8", shape = (3, 16))#candidate|2423|(3, 16)|var|int8
bop_2424 = relay.logical_xor(var_2422.astype('int8'), relay.reshape(var_2423.astype('int8'), relay.shape_of(var_2422))) # shape=(3, 16)
output = bop_2424
output2 = bop_2424
func_2433 = relay.Function([var_2422,var_2423,], output)
mod['func_2433'] = func_2433
mod = relay.transform.InferType()(mod)
mutated_mod['func_2433'] = func_2433
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2433_call = mutated_mod.get_global_var('func_2433')
var_2435 = relay.var("var_2435", dtype = "int8", shape = (3, 16))#candidate|2435|(3, 16)|var|int8
var_2436 = relay.var("var_2436", dtype = "int8", shape = (3, 16))#candidate|2436|(3, 16)|var|int8
call_2434 = func_2433_call(var_2435,var_2436,)
output = call_2434
func_2437 = relay.Function([var_2435,var_2436,], output)
mutated_mod['func_2437'] = func_2437
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2448 = relay.var("var_2448", dtype = "float32", shape = (16, 8))#candidate|2448|(16, 8)|var|float32
var_2449 = relay.var("var_2449", dtype = "float32", shape = (16, 8))#candidate|2449|(16, 8)|var|float32
bop_2450 = relay.divide(var_2448.astype('float32'), relay.reshape(var_2449.astype('float32'), relay.shape_of(var_2448))) # shape=(16, 8)
func_1941_call = mod.get_global_var('func_1941')
func_1943_call = mutated_mod.get_global_var('func_1943')
call_2457 = relay.TupleGetItem(func_1941_call(), 0)
call_2458 = relay.TupleGetItem(func_1943_call(), 0)
output = relay.Tuple([bop_2450,call_2457,])
output2 = relay.Tuple([bop_2450,call_2458,])
func_2477 = relay.Function([var_2448,var_2449,], output)
mod['func_2477'] = func_2477
mod = relay.transform.InferType()(mod)
mutated_mod['func_2477'] = func_2477
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2477_call = mutated_mod.get_global_var('func_2477')
var_2479 = relay.var("var_2479", dtype = "float32", shape = (16, 8))#candidate|2479|(16, 8)|var|float32
var_2480 = relay.var("var_2480", dtype = "float32", shape = (16, 8))#candidate|2480|(16, 8)|var|float32
call_2478 = func_2477_call(var_2479,var_2480,)
output = call_2478
func_2481 = relay.Function([var_2479,var_2480,], output)
mutated_mod['func_2481'] = func_2481
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2099_call = mod.get_global_var('func_2099')
func_2101_call = mutated_mod.get_global_var('func_2101')
call_2490 = relay.TupleGetItem(func_2099_call(), 0)
call_2491 = relay.TupleGetItem(func_2101_call(), 0)
func_2373_call = mod.get_global_var('func_2373')
func_2374_call = mutated_mod.get_global_var('func_2374')
call_2506 = func_2373_call()
call_2507 = func_2373_call()
var_2508 = relay.var("var_2508", dtype = "float32", shape = (3, 6))#candidate|2508|(3, 6)|var|float32
bop_2509 = relay.logical_and(call_2506.astype('bool'), relay.reshape(var_2508.astype('bool'), relay.shape_of(call_2506))) # shape=(3, 6)
bop_2512 = relay.logical_and(call_2507.astype('bool'), relay.reshape(var_2508.astype('bool'), relay.shape_of(call_2507))) # shape=(3, 6)
uop_2513 = relay.log2(call_2506.astype('float64')) # shape=(3, 6)
uop_2515 = relay.log2(call_2507.astype('float64')) # shape=(3, 6)
func_1397_call = mod.get_global_var('func_1397')
func_1400_call = mutated_mod.get_global_var('func_1400')
const_2522 = relay.const([[7,10,2,7,-4,-8,-6,2,-2,1]], dtype = "int64")#candidate|2522|(1, 10)|const|int64
call_2521 = relay.TupleGetItem(func_1397_call(relay.reshape(const_2522.astype('int64'), [10, 1]), relay.reshape(const_2522.astype('int64'), [10, 1]), ), 2)
call_2523 = relay.TupleGetItem(func_1400_call(relay.reshape(const_2522.astype('int64'), [10, 1]), relay.reshape(const_2522.astype('int64'), [10, 1]), ), 2)
output = relay.Tuple([call_2490,bop_2509,uop_2513,call_2521,const_2522,])
output2 = relay.Tuple([call_2491,bop_2512,uop_2515,call_2523,const_2522,])
func_2531 = relay.Function([var_2508,], output)
mod['func_2531'] = func_2531
mod = relay.transform.InferType()(mod)
var_2532 = relay.var("var_2532", dtype = "float32", shape = (3, 6))#candidate|2532|(3, 6)|var|float32
output = func_2531(var_2532)
func_2533 = relay.Function([var_2532], output)
mutated_mod['func_2533'] = func_2533
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1315_call = mod.get_global_var('func_1315')
func_1316_call = mutated_mod.get_global_var('func_1316')
call_2537 = func_1315_call()
call_2538 = func_1315_call()
output = relay.Tuple([call_2537,])
output2 = relay.Tuple([call_2538,])
func_2540 = relay.Function([], output)
mod['func_2540'] = func_2540
mod = relay.transform.InferType()(mod)
output = func_2540()
func_2541 = relay.Function([], output)
mutated_mod['func_2541'] = func_2541
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2545 = relay.const(7, dtype = "int16")#candidate|2545|()|const|int16
const_2546 = relay.const([[-6,5,-10,2,-5,-2,8,5,-3,-7,3],[4,-9,9,1,-2,-10,-1,-1,10,6,1],[8,-4,-3,10,5,6,-7,2,9,-6,-10],[-9,2,6,-3,-1,-5,4,6,-1,-9,6],[6,-5,3,-1,-10,3,10,-9,-8,-6,5],[-10,5,1,8,2,-7,-8,2,-3,-10,4],[6,4,6,-9,-9,10,-8,-8,8,2,-8],[-4,-10,3,-3,-9,-2,5,3,9,-10,-6],[9,9,-5,-5,-1,-3,7,-10,-7,-1,4]], dtype = "int16")#candidate|2546|(9, 11)|const|int16
bop_2547 = relay.less(const_2545.astype('bool'), const_2546.astype('bool')) # shape=(9, 11)
var_2555 = relay.var("var_2555", dtype = "int16", shape = (9, 11))#candidate|2555|(9, 11)|var|int16
bop_2556 = relay.less_equal(const_2546.astype('bool'), relay.reshape(var_2555.astype('bool'), relay.shape_of(const_2546))) # shape=(9, 11)
output = relay.Tuple([bop_2547,bop_2556,])
output2 = relay.Tuple([bop_2547,bop_2556,])
func_2564 = relay.Function([var_2555,], output)
mod['func_2564'] = func_2564
mod = relay.transform.InferType()(mod)
mutated_mod['func_2564'] = func_2564
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2565 = relay.var("var_2565", dtype = "int16", shape = (9, 11))#candidate|2565|(9, 11)|var|int16
func_2564_call = mutated_mod.get_global_var('func_2564')
call_2566 = func_2564_call(var_2565)
output = call_2566
func_2567 = relay.Function([var_2565], output)
mutated_mod['func_2567'] = func_2567
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2611 = relay.var("var_2611", dtype = "int16", shape = (5, 8, 7))#candidate|2611|(5, 8, 7)|var|int16
const_2612 = relay.const([[[4,-6,7,6,1,8,-9],[-10,8,5,10,8,-4,5],[8,-7,-2,-1,-4,2,5],[7,-3,-4,-6,-4,9,-4],[-4,-2,7,-5,2,-7,4],[-2,-1,3,-8,4,6,-2],[2,7,-7,8,-6,-2,-10],[-5,-2,3,7,-7,3,-1]],[[-3,-3,9,1,-8,-1,-2],[1,5,8,2,-1,9,6],[-1,-2,-1,9,6,-7,1],[-7,2,6,-5,-5,6,-7],[-3,-7,-1,-1,4,2,2],[-2,-9,7,-8,-1,9,-9],[-9,10,-1,-8,-5,-5,-1],[-1,-6,7,2,9,8,5]],[[5,4,2,7,2,6,-4],[-8,-9,9,10,3,1,5],[3,9,-4,1,7,-9,-3],[6,-7,-2,1,-4,-7,-6],[10,-8,-9,-10,-9,9,10],[3,-3,1,-3,-8,4,5],[2,8,3,6,2,-5,8],[-5,1,2,-8,6,5,6]],[[-8,9,-4,-2,2,9,4],[4,-7,10,8,-2,4,10],[1,-1,6,-9,-6,9,6],[3,-3,2,6,10,6,2],[9,1,7,1,1,-6,-10],[6,-3,8,6,6,3,-7],[7,1,3,-9,2,-4,9],[-2,2,-9,-7,-1,-8,-8]],[[-2,-2,-6,4,-4,8,3],[1,-1,-5,8,-10,-10,1],[-7,-10,4,7,-4,-9,-3],[4,-10,-10,-7,6,6,2],[8,-5,-2,3,9,-8,-10],[-3,6,9,-4,1,-1,-7],[-4,4,-4,-6,-5,2,8],[9,7,-10,-1,10,-7,5]]], dtype = "int16")#candidate|2612|(5, 8, 7)|const|int16
bop_2613 = relay.right_shift(var_2611.astype('int16'), relay.reshape(const_2612.astype('int16'), relay.shape_of(var_2611))) # shape=(5, 8, 7)
func_1078_call = mod.get_global_var('func_1078')
func_1079_call = mutated_mod.get_global_var('func_1079')
call_2617 = func_1078_call()
call_2618 = func_1078_call()
uop_2625 = relay.tan(var_2611.astype('float32')) # shape=(5, 8, 7)
bop_2628 = relay.maximum(uop_2625.astype('int32'), relay.reshape(var_2611.astype('int32'), relay.shape_of(uop_2625))) # shape=(5, 8, 7)
uop_2631 = relay.erf(uop_2625.astype('float64')) # shape=(5, 8, 7)
func_2394_call = mod.get_global_var('func_2394')
func_2395_call = mutated_mod.get_global_var('func_2395')
call_2634 = func_2394_call()
call_2635 = func_2394_call()
func_1492_call = mod.get_global_var('func_1492')
func_1497_call = mutated_mod.get_global_var('func_1497')
var_2637 = relay.var("var_2637", dtype = "float32", shape = (1, 195))#candidate|2637|(1, 195)|var|float32
var_2638 = relay.var("var_2638", dtype = "float32", shape = ())#candidate|2638|()|var|float32
const_2639 = relay.const([-0.387606,0.962134,-8.754485,5.861471,-8.454908,9.062579,-1.759262,1.777270,-0.399104,1.915802,-2.696619,-2.237847,-1.277933,-0.108634,0.207736,-5.332589,-4.389254,-8.742685,0.080808,3.450199,6.245680,9.237119,1.020129,4.343767,-0.132362,-1.586150,-4.726517,-5.434961,6.435020,-8.358436,-4.377401,9.811463,5.862368,8.290200,9.607065,-7.771008,0.046178,0.557235,5.805790,-3.617030,0.740177,-1.746248,4.923308,-5.312139,5.068510,6.272879,7.172348,3.678112,-4.440275,-2.074049,2.635977,4.152365,-0.436854,6.703291,5.896363,-7.746499,-8.957527,-6.793502,9.739208,-8.925142,-2.774776,-1.103550,-1.918387,-0.998755,-1.086663,1.321437,8.868322,-0.111456,9.112749,7.436136,5.589879,8.126363,2.998141,-5.488512,3.716894,7.767578,-4.830461,9.162296,6.605034,-0.155139,-8.894064,9.793051,-2.634563,-5.480703,0.734593,-5.303925,6.648417,-4.298978,7.495897,5.122144,-0.737071,-9.541591,7.715898,8.406224,2.711278,-3.207301,0.002680,-7.537700,-4.195546,8.487688,-8.506616,-6.206158,-2.826958,-7.244350,6.635304,-6.852326,8.250779,4.463701,-8.406311,9.576235,-2.723777,-6.924329,-5.992574,9.228304,6.177210,-4.182200,-4.589184,8.217642,-0.561525,-1.102439,0.759874,5.743488,-5.368049,9.723178,8.983637,-9.053607,7.079292,-3.318863,6.791564,-4.688496,-1.635330,1.991873,5.549120,3.539977,-9.289363,-9.906844,-9.333751,-2.844541,-7.479266,-9.152968,0.005191,5.919400,6.255606,-6.078784,7.434866,0.517767,6.805904,-2.552643,8.014733,-4.510546,0.146337,3.515955,-9.079753,-8.552006,-7.256870,5.338423,-0.834286,-5.927149,-9.302808,-0.306979,5.433152,3.508861,-5.288956,7.356634,-3.946710,-6.524440,-4.570127,4.883513,3.649385,9.175512,-5.142555,9.808872,2.584736,-5.723485,-5.338086,8.086961,-3.300253,4.037051,-5.948970,-9.288451,-5.440325,7.096101,7.824488,-8.756818,-3.258850,-3.408116,-2.879128,4.700420,7.308071,5.552033,8.630958,9.925087,5.822003,-6.446506,5.974338,-9.095187,1.079718,8.392299,3.388269,-1.242904,2.418203,-4.473575,-9.939640,-2.829220,-1.872054,-7.256396,-6.506999,0.795372,-5.651627,-7.022939,3.956701,5.075402,-5.002134,5.649268,-6.424555,9.947804,-9.400865,8.093267,-2.606526,-9.724486,2.567912,-7.765517,-5.934910,-2.125087,5.546487,-4.367629,8.561880,-0.116374,-1.788746,7.155672,7.620021,8.849091,7.565549,-5.136380,-6.740539,-4.519807,5.048342,7.075235,-3.522554,-2.917342,-6.751313,-8.331679,0.204894,0.621668,-7.724132,-0.902493,-9.785826,3.873963,4.509010,6.329035,-1.424714,0.583904,-4.691684,5.343730,-5.482588,-3.326120,2.108784,3.642972,2.883384,-4.847517,4.569038,7.522989,-4.705422,5.288144,1.713393,5.736872,-5.089321,1.577567,-5.574612,-5.085765,4.905411,9.677230,2.951985,7.742266,-3.327204,4.423630,-1.158079,5.810248,5.789217,-3.087650,9.088477,2.907444,0.599716,-3.667754,0.715073,-0.703422,2.900969,6.971988,-2.869975,-0.809454,6.642731,9.012503,-1.411605,-7.736545,3.674825,-8.472288,5.877554,1.934881,1.245405,-6.886968,-2.101918,8.306117,-8.259883,-7.741510,6.660823,6.608074,-9.631530,-8.550437,2.262647,5.784448,-9.854370,0.446143,6.632165,-4.354144,-4.088877,4.563230,3.350335,8.515594,-1.858714,6.026823,-7.276639,-1.888323,1.300677,5.066998,7.319973,4.671424,9.331940,0.603382,9.833094,2.068704,-9.150804,-8.506537,8.930098,-2.894135,-7.637318,6.594627,-2.647781,8.563404,-5.880665,9.781436,8.325143,5.554449,8.893598,9.671527,-6.075619,3.863526,-2.205010,2.943392,2.022370,9.194207,4.414866,-7.710106,-5.008709,7.373852,-2.295077,-4.844081,9.625060,-3.732061,8.106129,-1.449594,-5.165497,-2.665761,6.570911,-3.788236,1.740853,-0.831231,-3.476948,0.073056,4.445286,-4.840147,3.964039,2.263176,-6.845803,5.450207,-0.764411,-6.043682,1.474717,7.226817,-0.744922,8.565898,7.311632,-7.584820,-6.193833,-6.195004,-7.591194,-5.185850,4.900205,-4.099909,2.873753,4.947978,2.162764,6.647888,3.041692,-1.485641,-3.637959,1.887327,6.307993,-3.977202,1.526789,9.917115,2.079616,7.042062,-6.784278,-2.429242,-8.650200,1.755938,4.543923,-2.944974,9.802051,-5.031319,7.831844,6.150388,9.997741,-0.508975,-0.534034,-5.178300,9.124687,-8.710238,-9.618633,-8.083676,4.013698,-8.026189,-7.604011,2.655299,5.210818,7.708070,4.970442,-2.600562,-8.311165,-8.707333,-2.988534,-5.091135,-2.788479,4.020891,-5.942614,8.528275,5.549508,-4.172753,1.515660,2.121047,-6.778792,2.993015,5.412772,-0.742262,-4.444988,-3.425164,-1.490797,8.171022], dtype = "float32")#candidate|2639|(448,)|const|float32
call_2636 = relay.TupleGetItem(func_1492_call(relay.reshape(var_2637.astype('float32'), [15, 13]), relay.reshape(var_2638.astype('float32'), []), relay.reshape(const_2639.astype('float32'), [8, 56]), ), 4)
call_2640 = relay.TupleGetItem(func_1497_call(relay.reshape(var_2637.astype('float32'), [15, 13]), relay.reshape(var_2638.astype('float32'), []), relay.reshape(const_2639.astype('float32'), [8, 56]), ), 4)
func_2099_call = mod.get_global_var('func_2099')
func_2101_call = mutated_mod.get_global_var('func_2101')
call_2651 = relay.TupleGetItem(func_2099_call(), 0)
call_2652 = relay.TupleGetItem(func_2101_call(), 0)
output = relay.Tuple([bop_2613,call_2617,bop_2628,uop_2631,call_2634,call_2636,var_2637,var_2638,const_2639,call_2651,])
output2 = relay.Tuple([bop_2613,call_2618,bop_2628,uop_2631,call_2635,call_2640,var_2637,var_2638,const_2639,call_2652,])
func_2658 = relay.Function([var_2611,var_2637,var_2638,], output)
mod['func_2658'] = func_2658
mod = relay.transform.InferType()(mod)
var_2659 = relay.var("var_2659", dtype = "int16", shape = (5, 8, 7))#candidate|2659|(5, 8, 7)|var|int16
var_2660 = relay.var("var_2660", dtype = "float32", shape = (1, 195))#candidate|2660|(1, 195)|var|float32
var_2661 = relay.var("var_2661", dtype = "float32", shape = ())#candidate|2661|()|var|float32
output = func_2658(var_2659,var_2660,var_2661,)
func_2662 = relay.Function([var_2659,var_2660,var_2661,], output)
mutated_mod['func_2662'] = func_2662
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2217_call = mod.get_global_var('func_2217')
func_2219_call = mutated_mod.get_global_var('func_2219')
call_2672 = relay.TupleGetItem(func_2217_call(), 1)
call_2673 = relay.TupleGetItem(func_2219_call(), 1)
output = call_2672
output2 = call_2673
func_2678 = relay.Function([], output)
mod['func_2678'] = func_2678
mod = relay.transform.InferType()(mod)
output = func_2678()
func_2679 = relay.Function([], output)
mutated_mod['func_2679'] = func_2679
mutated_mod = relay.transform.InferType()(mutated_mod)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_2683 = func_525_call()
call_2684 = func_525_call()
func_2531_call = mod.get_global_var('func_2531')
func_2533_call = mutated_mod.get_global_var('func_2533')
call_2685 = relay.TupleGetItem(func_2531_call(relay.reshape(call_2683.astype('float32'), [3, 6])), 4)
call_2686 = relay.TupleGetItem(func_2533_call(relay.reshape(call_2683.astype('float32'), [3, 6])), 4)
output = relay.Tuple([call_2683,call_2685,])
output2 = relay.Tuple([call_2684,call_2686,])
func_2687 = relay.Function([], output)
mod['func_2687'] = func_2687
mod = relay.transform.InferType()(mod)
output = func_2687()
func_2688 = relay.Function([], output)
mutated_mod['func_2688'] = func_2688
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2691 = relay.var("var_2691", dtype = "uint32", shape = (14, 5))#candidate|2691|(14, 5)|var|uint32
const_2692 = relay.const([[-9,-9,3,4,-5],[-10,8,-3,-9,2],[-6,-7,-4,2,6],[2,-3,8,-4,-6],[9,3,-5,4,3],[1,8,4,8,-5],[-3,2,7,-4,6],[1,-9,1,-9,9],[-2,1,-10,-9,10],[3,-1,1,-6,-9],[6,-8,-2,-8,5],[-3,10,4,-1,-8],[-8,-6,7,9,4],[-8,7,7,6,-2]], dtype = "uint32")#candidate|2692|(14, 5)|const|uint32
bop_2693 = relay.bitwise_or(var_2691.astype('uint32'), relay.reshape(const_2692.astype('uint32'), relay.shape_of(var_2691))) # shape=(14, 5)
output = relay.Tuple([bop_2693,])
output2 = relay.Tuple([bop_2693,])
func_2698 = relay.Function([var_2691,], output)
mod['func_2698'] = func_2698
mod = relay.transform.InferType()(mod)
var_2699 = relay.var("var_2699", dtype = "uint32", shape = (14, 5))#candidate|2699|(14, 5)|var|uint32
output = func_2698(var_2699)
func_2700 = relay.Function([var_2699], output)
mutated_mod['func_2700'] = func_2700
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_2702 = relay.TupleGetItem(func_1909_call(), 0)
call_2703 = relay.TupleGetItem(func_1910_call(), 0)
output = relay.Tuple([call_2702,])
output2 = relay.Tuple([call_2703,])
func_2720 = relay.Function([], output)
mod['func_2720'] = func_2720
mod = relay.transform.InferType()(mod)
output = func_2720()
func_2721 = relay.Function([], output)
mutated_mod['func_2721'] = func_2721
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2727 = relay.const(-7.587945, dtype = "float32")#candidate|2727|()|const|float32
const_2728 = relay.const([[[-4.804576,-1.552636,-8.604205,-3.001751,0.239950,2.607498,1.900703,-8.252002,9.993362,5.324359,-9.048375,-6.903672,3.432256,6.724229],[3.698441,3.660253,-6.970762,7.763080,7.120158,-2.725224,6.517123,4.145372,-8.595240,1.096994,4.733342,3.583251,4.792531,-4.716076],[3.166933,-3.106887,1.632835,5.051792,7.105471,-7.502000,-5.721676,0.646251,-0.326239,9.625877,5.223507,-3.122402,8.987755,-7.800625]],[[-4.530361,5.142229,-6.283628,-0.144372,-8.905408,3.862725,-5.506392,-1.244531,5.411056,1.150451,-9.581320,-7.814008,9.483247,1.936651],[-1.577349,3.573304,-6.903730,7.216385,-6.897063,-3.584887,4.047472,-0.547039,9.774363,-9.157849,1.187274,5.290614,-3.593389,0.273669],[-8.050721,-9.635705,-0.995007,6.388541,3.873540,-9.769971,7.985914,-8.977110,-2.775072,-1.749787,7.541807,-3.239098,-3.892995,2.315647]],[[9.484832,4.133003,0.605336,4.970715,5.380850,-6.371608,2.161638,4.408241,-6.273232,3.597796,-4.931503,-2.565370,-0.778077,-3.075912],[-1.914406,-3.070732,-7.536250,-3.330981,3.541088,-3.112201,-8.210275,3.041358,9.854161,-5.800740,-3.633970,-9.788053,-6.123597,1.570923],[-7.750989,9.605179,5.468375,-8.370841,5.009255,9.930208,-0.205252,-3.669096,-7.974847,-8.840583,-6.433319,8.832049,-5.380326,5.407897]],[[5.718175,-9.551697,1.832298,-7.236825,2.131291,-0.885430,0.532174,2.979429,5.511340,6.891294,-2.561700,7.018720,-1.857530,8.362805],[3.050818,0.190953,0.495779,-0.377883,-0.087401,3.394756,7.621708,-5.450038,8.122329,6.232925,4.743151,0.512540,-0.651432,-2.189117],[6.190137,-6.549791,6.846946,0.759208,3.682711,1.422492,5.446412,9.974471,-0.860940,5.249234,6.635248,2.411279,2.545149,-7.080115]]], dtype = "float32")#candidate|2728|(4, 3, 14)|const|float32
bop_2729 = relay.divide(const_2727.astype('float32'), const_2728.astype('float32')) # shape=(4, 3, 14)
func_278_call = mod.get_global_var('func_278')
func_282_call = mutated_mod.get_global_var('func_282')
const_2736 = relay.const([5,-9,8,8,-2,-3,-7,-8,-1,-3,4,8,-8,4,-10,2,-3,-10,3,-10,7,5,-6,-8,5,-3,-9,-7,-5,2,1,3,-6,-4,-10,-3,-10,2,-1,-4,-2,-5,10,-9,-6,8,-6,1,5,9,2,4,-10,9,-7,-10,-1,-1,9,-4,2,2,5,9,9,-7,2,4,-10,-7,-7,7,2,-7,-9,-7,2,-10,-3,9,7,-6,7,-2,4,-3,6,8,-9,2,5,10,9,6,-9,5,-1,6,10,3,9,4,8,-3,5,7,-5,7,7,7,8,-3,-2,7,8,5,-7,-3,8,-2,-3,3,5,3,5,7,9,-2,-10,-2,-8,8,-3,10,8,4,-8,1,-9,5,-4,9,4,-4,-2,-5,-3,-8,5,-8,8,6,1,1,-4,5,-1,-1,10,1,2,-4,10,-3,-10,8,5,7,-5,-5,7,7,6,2,1,-8,9,2,-6,4,6,-7,8,-6,-4,-2,-2,-6,-4,4,2,-2,8,10,-9,-3,6,7,-5,9,-5,-8,-9,-7,5,-2,-7,-7,8,-10,-2,3,2,4,-4,-5,-3,8,-9,6,-8,10,3,-2,-7,-10,-5,8,3,-5,6,-8,4,2,-2,-7,9,8,5,4,2,2,7,5,2,6,1,-9,-9,10,9,4,7,1,1,3,-9,1,4,3,7,6,-3,-6,-5,-6,8,-4,3,6,-3,9,5,-6,-7,7,-10,5,9,10,10,10,-7,2,4,-3,-8,-6,4,1,5,-7,-2,-10,6,7,-5,5,-5,-1,-4,3,9,-4,-5,4,5,-5,-8,10,-5,1,5,-4,-10,9,-9,7,-4,-1,-7,7,8,1,-6,-9,9,3,-5,-4,1,-10,-6,10,-7,-6,-8,6,-4,-4,-7,1,3,10,8,3,-5,-2,10,-10,-5,-3,9,8,6,4,-5,7,-9,-5,1,8,3,-7,-2,-3,-3,-9,-6,8,1,3,-6,-3,9,2,10,-6,-10,9,1,-10,-8,7,-7,4,-6,3,9,-7,-6,8,8,10,-8,3,-10,-4,-3,-10,7,-8,1,6,-7,-8,-1,2,2,-1,-2,-10,9,-1,-8,-3,-7,9,-2,9,8,-6,-6,10,10,-10,8,-5,5,-9,6,-9,5,-2,3,-8,-1,-9,3,-8,-3,-7,8,5,-7,8,6,-8,-3,10,4,-6,4,-5,-5,2,-1,6,-1,3,-7,1,8,6,2,7,6,2,-7,7,-9,-9,-9,8,-9,-8,-3,4,8,3,8,5,-3,3,3,-3,10,1,-8,9,-3,6,6,7,9,-4,-10,-10,6,-5,3,9,10,5,2,-7,-1,-9,2,10,6,10,-3,-10,-4,-7,-3,-2,-10,-2,1,5,2,-6,2,8,2,-3,-8,3,8,-9,-8,-3,-6,-1,4,-8,10,5,-1,-9,10,-7,6,10,-2,6,-2,10,-8,6,7,-10,-7,10,-9,3,2,10,2,4,1,-5,3,-10,8,-2,-9,5,-9,-7,3,-8,10,5,-7,5,2,-2,7,5,7,2,2,-3,-10,9,-10,10,-4,1,5,1,6,-10,7,-10,-5,-2,8,7,-6,5,3,6,-2,1,3,-3,-7,-3,10,-9,-6,9,-6,5,-3,10,6,3,2,-3,6,-5,-1,-5,4,8,-5,-3,5,5,-8,4,-10,2,-1,-9,5,9,-4,8,10,-7,3,6,-10,6,3,-2,8,-6,-2,-5,-9,-2,-1,-1,6,-4,10,9,-10,-7,-8,7,-6,-3,8,-3,2,-1,6,2,2,3,9,-1,10,3,9,-1,4,3,-2,10,4,7,-5,-3,8,-10,-9,-3,1,6,9,-4,1,-7,-6,9,10,1,6,6,-2,-8,-7,2,8,-5,5,-9,1,-1,5,-8,-5,-4,-2,2,5,7,-10,8,-6,6,6,-10,-3,3,-1,5,-9,7,-9,5,1,-6,10,-1,-8,-10,4,3,-2,-5,9,-3,1,10,-10,-2,7,-4,-3,-7,-3,9,-8,-5,9,5,4,-6,7,2,1,10,3,6,8,9,10,-9,10,-1,5,-7,-4,-3,9,2,-1,-9,-6,-2,-8,-6,-1,-9,8,-10,-7,-4,-1,4,2,1,8,-10,5,-9,-5,5,9,-4,5,-1,-2,3,-4,5,-5,-9,-4,-9,1,4,-10,-9,-3,7,-9,-10,5,-6,4,-5,6,2,-5,1,-9,-1,5,10,-6,-8,7,-8,3,1,5,-6,4,-5,1,4,-2,-8,8,7,-7,1,-4,6,4,5,-5,-10,-1,5,-10,9,-7,-1,7,1,2,7,2,3,1,5,1,5,10,-3,1,4,-10,-7,4,4,1,-4,-5,10,9,1,1,-1,6,-3,5,-8,-4,-6,4,-6,1,9,9,1,-3,7,8,4,1,4,-5,-4,7,9,10,-6,7,7,5,3,2,-3,-9,6,-4,-7,1,2,-1,-9,6,2,-1,1,-8,-5,4,1,-4,-3,-7,9,-2,-8,-2,4,4,-3,-6,2,-9,-4,-4,-8,4,-7], dtype = "uint16")#candidate|2736|(960,)|const|uint16
call_2735 = relay.TupleGetItem(func_278_call(relay.reshape(const_2736.astype('uint16'), [4, 16, 15]), relay.reshape(const_2736.astype('uint16'), [4, 16, 15]), ), 0)
call_2737 = relay.TupleGetItem(func_282_call(relay.reshape(const_2736.astype('uint16'), [4, 16, 15]), relay.reshape(const_2736.astype('uint16'), [4, 16, 15]), ), 0)
output = relay.Tuple([bop_2729,call_2735,const_2736,])
output2 = relay.Tuple([bop_2729,call_2737,const_2736,])
func_2744 = relay.Function([], output)
mod['func_2744'] = func_2744
mod = relay.transform.InferType()(mod)
mutated_mod['func_2744'] = func_2744
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2744_call = mutated_mod.get_global_var('func_2744')
call_2745 = func_2744_call()
output = call_2745
func_2746 = relay.Function([], output)
mutated_mod['func_2746'] = func_2746
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2763 = relay.const([[[6.465520,5.541987,-4.204554,-5.927940,4.499130,6.613810,5.787622,-2.807063,-7.696432,2.017954,-7.430796,-6.761128,-3.270816,5.291087],[2.177654,-0.205064,0.513060,-8.749607,-2.807874,-3.169837,3.689110,6.194674,5.543914,7.424126,-2.455693,-8.319980,2.376633,1.774340],[-9.339311,-5.646712,4.168723,-1.806698,-0.278820,5.483541,4.178798,-5.485633,3.075273,-3.078572,-7.287696,-7.213692,-1.403135,4.156200]],[[0.414474,6.245034,-8.757818,0.194910,-1.902058,-6.050166,-2.859559,-5.218362,5.424635,7.734208,-3.448634,4.208005,9.282612,2.187294],[-5.333215,9.810404,8.413750,0.547327,-5.123833,-5.571574,-6.299117,6.598890,-6.916142,-7.969292,3.050008,-7.495561,0.655251,-9.825981],[9.350783,-8.902163,-5.300922,0.004765,3.629159,8.950672,1.557363,-4.719835,-7.336928,3.859532,9.524702,-7.580962,4.282106,-2.162443]]], dtype = "float32")#candidate|2763|(2, 3, 14)|const|float32
var_2764 = relay.var("var_2764", dtype = "float32", shape = (2, 3, 14))#candidate|2764|(2, 3, 14)|var|float32
bop_2765 = relay.greater(const_2763.astype('bool'), relay.reshape(var_2764.astype('bool'), relay.shape_of(const_2763))) # shape=(2, 3, 14)
var_2771 = relay.var("var_2771", dtype = "bool", shape = (2, 3, 14))#candidate|2771|(2, 3, 14)|var|bool
bop_2772 = relay.bitwise_and(bop_2765.astype('uint16'), relay.reshape(var_2771.astype('uint16'), relay.shape_of(bop_2765))) # shape=(2, 3, 14)
func_1120_call = mod.get_global_var('func_1120')
func_1125_call = mutated_mod.get_global_var('func_1125')
var_2784 = relay.var("var_2784", dtype = "float32", shape = (12, 4))#candidate|2784|(12, 4)|var|float32
const_2785 = relay.const(4, dtype = "uint32")#candidate|2785|()|const|uint32
var_2786 = relay.var("var_2786", dtype = "float32", shape = (220,))#candidate|2786|(220,)|var|float32
call_2783 = relay.TupleGetItem(func_1120_call(relay.reshape(var_2784.astype('float32'), [16, 3]), relay.reshape(const_2785.astype('uint32'), []), relay.reshape(var_2786.astype('float32'), [220,]), ), 1)
call_2787 = relay.TupleGetItem(func_1125_call(relay.reshape(var_2784.astype('float32'), [16, 3]), relay.reshape(const_2785.astype('uint32'), []), relay.reshape(var_2786.astype('float32'), [220,]), ), 1)
func_2564_call = mod.get_global_var('func_2564')
func_2567_call = mutated_mod.get_global_var('func_2567')
const_2792 = relay.const([6,-3,-8,5,-3,10,-7,2,-5,-9,-2,-1,-10,1,-10,-2,-6,-3,4,-10,4,2,6,-3,5,-2,-4,9,-4,9,3,-4,-6,5,-7,-9,-7,-6,-3,4,1,-8,-8,1,9,8,3,1,1,3,1,2,-3,10,5,5,-9,5,1,-1,-7,-2,-3,4,-1,3,-9,9,-1,-9,1,6,-8,-7,-6,1,7,6,10,-3,6,-2,-1,10,-10,-2,-4,2,9,3,8,-9,-9,5,6,10,-6,-9,-5], dtype = "int16")#candidate|2792|(99,)|const|int16
call_2791 = relay.TupleGetItem(func_2564_call(relay.reshape(const_2792.astype('int16'), [9, 11])), 0)
call_2793 = relay.TupleGetItem(func_2567_call(relay.reshape(const_2792.astype('int16'), [9, 11])), 0)
var_2807 = relay.var("var_2807", dtype = "bool", shape = (9, 11))#candidate|2807|(9, 11)|var|bool
bop_2808 = relay.logical_and(call_2791.astype('bool'), relay.reshape(var_2807.astype('bool'), relay.shape_of(call_2791))) # shape=(9, 11)
bop_2811 = relay.logical_and(call_2793.astype('bool'), relay.reshape(var_2807.astype('bool'), relay.shape_of(call_2793))) # shape=(9, 11)
bop_2815 = relay.divide(bop_2765.astype('float64'), const_2785.astype('float64')) # shape=(2, 3, 14)
bop_2820 = relay.divide(const_2792.astype('float32'), relay.reshape(call_2791.astype('float32'), relay.shape_of(const_2792))) # shape=(99,)
bop_2823 = relay.divide(const_2792.astype('float32'), relay.reshape(call_2793.astype('float32'), relay.shape_of(const_2792))) # shape=(99,)
func_2040_call = mod.get_global_var('func_2040')
func_2044_call = mutated_mod.get_global_var('func_2044')
const_2827 = relay.const([[-7],[-10],[6],[6],[-9],[-2],[1],[7],[-2],[-1],[8],[-8],[1],[-4],[5],[3],[-1],[-2],[-7],[6],[-3],[4],[-10],[1],[10],[2],[9],[5],[-9],[1],[1],[5],[9],[10],[4],[-9],[4],[10],[-6],[8],[2],[8],[-2],[-3],[1],[5],[8],[-5],[-5],[-10],[-3],[-5],[-6],[-5],[-5],[10],[2],[9],[10],[5],[4],[5],[-9],[-8],[7],[3],[-8],[-9],[3],[-7],[-6],[8],[-9],[-2],[-8],[2],[8],[-5],[-10],[-3],[-2],[-9],[7],[5],[3],[-10],[2],[-6],[4],[-10],[8],[6],[1],[-5],[4],[-8],[-4],[-7],[4],[-3],[-7],[-5],[-1],[-10],[1],[-9],[3],[-5],[7],[-3],[-5],[9],[7],[-10],[-9],[9],[6],[-7],[-5],[3],[-8],[-4],[-6],[-3],[-8],[5],[-9],[-3],[-10],[9],[-8],[7],[2],[-10],[-2],[4],[-5],[-9],[7],[3],[-8],[9],[5],[-6],[-6],[6],[3],[7],[-9],[6],[-2],[-1],[9],[6],[-2],[1],[1],[9],[-1],[1],[-10],[-1],[-4],[2],[-5],[-7],[9],[2],[6],[3],[-10],[7],[-2],[7],[9],[-9],[9],[-4],[8],[6],[-2],[-10],[1],[-8],[1],[4],[-4],[5],[7],[-9],[2],[-7],[-10],[1],[7],[-1],[2],[-2],[5],[-1],[-9],[-10],[-8],[-9],[4],[3],[-8],[6],[5],[5],[-9],[-3],[-8],[-6],[-2],[9],[-2],[6],[-1],[6],[5],[-3],[2],[-10],[-9],[9],[9],[-10],[6],[10],[-5],[-1],[8],[-8],[4],[-6],[3],[-9],[-9],[2],[-4],[-7],[-2],[2],[5],[5],[-2],[6],[4],[9],[-7],[-5],[4],[-5],[-5],[-2],[-7],[-3],[-8],[-4],[8],[-2],[6],[8],[-3],[9],[2],[10],[10],[-9],[-2],[5],[2],[8],[-5],[-4],[4],[1],[7],[3],[6],[1],[6],[4],[-9],[-2],[-5],[-10],[-6],[-8],[3],[-2],[-1],[8],[-4],[-7],[-2],[10],[2],[-3],[-6],[9],[3],[10],[-10],[-10],[2],[1],[-7],[-3],[-1],[-3],[1],[-1],[7],[1],[1],[7],[-1],[-5],[5],[5],[9],[5],[6],[-9],[8],[-10],[-5],[-1],[4],[-8],[-2],[-1],[1],[-6],[-1],[-8],[-9],[-6],[-8],[-2],[5],[7],[-4],[2],[8],[5],[4],[5],[-9],[-4],[1],[1],[-2],[5],[-2],[-8],[2],[8],[-4],[3],[2],[-4],[-3],[-4],[9],[5],[9],[-2],[-2],[2],[-5],[-2],[-8],[10],[-3],[-5],[-3],[1],[-9],[10],[-2],[-8],[8],[-3],[6],[-6],[-8],[7],[-9],[9],[3],[4],[5],[4],[2],[-10],[7],[-7],[-9],[-7],[8],[7],[2],[-3],[4],[-2],[5],[5],[-3],[2],[2],[-2],[-1],[6],[-1],[-6],[-8],[4],[-10],[2],[8],[2],[-10],[-3],[-5],[10],[8],[-8],[1],[10],[9],[-3],[-10],[9],[-6],[5],[5],[-7],[-6],[7],[5],[1],[-9],[2],[-3],[4],[-8],[-8],[-5],[-8],[8],[-5],[3],[-7],[2],[10],[-1],[-9],[-8],[-2],[3],[2],[4],[8],[9],[-2],[-1],[9],[9],[-1],[-1],[-8],[-2],[5],[-10],[-10],[5],[-4],[1],[10],[-10],[-6],[-1],[5],[4],[3],[-8],[8],[-2],[9],[-9],[6],[-2],[5],[-2],[3],[9],[-5],[8],[-4],[-9],[-8],[10],[7],[8],[-3],[-7],[-5],[7],[2],[6],[3],[9],[-8],[9],[1],[-9],[-7],[4],[10],[-1],[1],[9],[-9],[-5],[-8],[-10],[-6],[1],[10],[-6],[-10],[-8],[5],[-10],[-6],[-5],[1],[2],[-2],[4],[-5],[5],[5],[-7],[-7],[-6],[9],[3],[-7],[-6],[-6],[-7],[-7],[-8],[5],[-3],[6],[2],[-3],[6],[4],[-9],[5],[-5],[-5],[-5],[-4],[-7],[-3],[-1],[8],[-9],[-1],[8],[1],[-4],[-8],[-6],[-4],[-7],[-3],[-9],[7],[9],[-4],[-8],[-2],[9],[-9],[-9],[-2],[5],[-5],[3],[5],[-10],[2],[-7],[-9],[9],[-7],[-6],[-4],[-4],[3],[-5],[-6],[3],[5],[-3],[-5],[1],[4],[4],[-4],[-7],[-10],[10],[9],[8],[7],[6],[-4],[-5],[4],[9],[2],[-7],[-8],[10],[-6],[5],[-9],[9],[-10],[1],[-10],[9],[6],[5],[3],[-9],[4],[1],[-7],[-5],[5],[8],[-4],[-3],[3],[2],[-9],[-9],[6],[-6],[8],[3],[-1],[5],[5],[-2],[10],[8],[6],[10],[5],[7],[2],[10],[-10],[6],[2],[-3],[6],[-3],[-9],[-5],[5],[-5],[-5],[4],[-5],[-6],[7],[-5],[-2],[8],[-2],[7],[8],[-3],[4],[-6],[-1],[2],[3],[8],[7],[5],[9],[4],[10],[-3],[2],[10],[-9],[4],[1],[2],[10],[10],[2],[-3],[9],[6],[9],[9],[-1],[-2],[4],[2],[5],[7],[-8],[9],[-5],[-1],[-6],[-7],[-4],[3],[2],[-1],[-6],[-5],[-4],[-2],[-2],[8],[9],[8],[-6],[-2],[-3],[7],[3],[-1],[4],[7],[6],[8],[10],[-3],[-2],[2],[8],[-7],[9],[7],[10],[1],[6],[8],[5],[-7],[9],[1],[9],[2],[4],[-3],[4],[-7],[1],[4],[9],[-5],[-7],[-2],[10],[3],[9],[1],[-7],[1],[6],[4],[-6],[-2],[-6],[-8],[-4],[-10],[6],[5],[-3],[7],[2],[2],[-8],[-3],[-9],[-6],[-9],[-3],[7],[-1],[-4],[1],[-3],[6],[-5],[-10],[1],[-10],[-4],[-5],[1],[9],[-2],[-6],[4],[9],[8],[-8],[8],[-5],[-8],[-4],[6],[8],[-9],[5],[8],[2],[4],[8],[-7],[6],[1],[-4],[-1],[-6],[3],[-8],[3],[1],[2],[-2],[9],[10],[-3],[-3],[-9],[4],[4],[-9],[-7],[4],[4],[-1],[9],[-1],[-7],[9],[8],[-9],[-10],[-1],[-3],[9],[5],[10],[4],[3],[5],[8],[-5],[5],[3],[-6],[-5],[-2],[-9],[-6],[-10],[-10],[4],[8],[-6],[-7],[-3],[5],[2],[-10],[5],[-6],[4],[5],[-2],[8],[5],[-10],[9],[-6],[-7],[6],[-6],[9],[-2],[-2],[4],[-7],[-7],[4],[5],[-5],[7],[9],[1],[1],[-8],[9],[-1],[-1],[7],[9],[-3],[-2],[1],[4],[-5],[6],[4],[1],[1],[-5],[-10],[-3],[5],[3],[-9],[-5],[-1],[7],[-4],[-5],[7],[4],[7],[9],[7],[6],[-8],[-10],[5],[7],[8],[-6],[-1],[-8],[-3],[-4],[6],[-4],[-3],[-2],[-6],[-8],[9],[9],[1],[-6],[8],[-7],[-7],[3],[-10],[2],[5],[9],[-8],[-8],[1],[-1],[4],[-2],[6],[4],[-4],[-2],[-6],[2],[-2],[-1],[10],[-6],[-3],[5],[10],[4],[-5],[-9],[-6],[6],[3],[-2],[-3],[-8],[9],[-3],[-1],[9],[5],[8],[1],[-1],[8],[-9],[2],[6],[-3],[8],[-3],[10],[-8],[-2],[6],[-5],[-7],[-8],[6],[-9],[9],[1],[-5],[-1],[-5],[4],[-5],[-1],[1],[-2],[-7],[10],[9],[-4],[-9],[-9],[3],[2],[-9],[4],[-3],[-4],[-5],[1],[-10],[-3],[-1],[1],[-2],[7],[10],[-10],[8],[-8],[-6],[-2],[5],[7],[-3],[-7],[5],[-10],[-2],[-8],[-7],[-6],[3],[-9],[9],[-1],[-2],[-3],[-1],[6],[-1],[6],[3],[-2],[-6],[9],[-7],[10],[10],[-4],[-7],[-3],[-8],[-5],[-7],[2],[5],[3],[4],[5],[8],[-3],[1],[-5],[-3],[2],[-4],[10],[-1],[10],[9],[7],[-9],[-2],[-6],[3],[10],[-8],[-1],[4],[4],[5],[-10],[9],[-1],[3],[-10],[-4],[1],[-2],[-2],[10],[2],[5],[-3],[-9],[-4],[4],[-1],[-7],[-9],[-2],[1],[7],[-9],[-10],[7],[5],[9],[5],[8],[8],[8],[7],[-2],[-2],[7],[1],[-10],[-1],[-1],[-2],[-2],[-8],[-6],[5],[-6],[-7],[5],[-6],[10],[-10],[-1],[-6],[-1],[-2],[9],[-9],[-10],[4],[-9],[-1],[1],[9],[-3],[3],[-4],[10],[-3],[-8],[7],[-5],[3],[-9],[-10],[-10],[-3],[-3],[-10],[10],[-9],[7],[-5],[-9],[-5],[-5],[7],[1],[5],[-3],[-5],[5],[6],[2],[-2],[9],[7],[-3],[3],[3],[6],[-2],[-9],[3],[1],[10],[-10],[3],[4],[-4],[-10],[-6],[-9],[1],[8],[1],[5],[7],[4],[-7],[-5],[-2],[3],[3],[-3],[-2],[1],[1],[-6],[3],[-10],[7],[8],[9],[-1],[9],[-5],[-1],[6],[-6],[2],[-3],[-7],[-5],[1],[-8],[-2],[8],[9],[2],[-4],[-8],[-1],[9],[-4],[8],[4],[-10],[-10],[-5],[-1],[-3],[8],[-3],[-9],[-8],[5],[3],[-6],[5],[7],[7],[7],[-2],[2],[-7],[8],[8],[-9],[2],[-7],[10],[8],[-9],[-6],[-6],[8],[-2],[-1],[-1],[2],[8],[5],[-8],[8],[-2],[-2],[-10],[-6],[5],[-5],[-4],[-4],[-7],[1],[-10],[3],[-4],[5],[-3],[-4],[9],[-9],[-6],[-9],[-8],[-2],[7],[-9],[3],[-2],[4],[2],[-6],[-10],[9],[5],[-9],[8],[-3],[8],[1],[-4],[-8],[-3],[-4],[3],[-2],[5],[-1],[6],[6],[1],[-10],[-3],[-7],[-4],[3],[-9],[2],[1],[8],[-5],[-10],[-10],[-3],[2],[-10],[7],[1],[2],[-8],[-8],[-4],[-1],[8],[-6],[7],[-9],[-10],[-9],[-5],[-4],[-3],[4],[-8],[-4],[3],[-9],[-9],[-5],[2],[3],[3],[-8],[9],[7],[8],[9],[-2],[-9],[-9],[9],[5],[1],[8],[-10],[9],[2],[3],[6],[7],[6],[-9],[-9],[7],[-3],[3],[10],[-10],[3],[-9],[10],[6],[2],[2],[-1],[-5],[6],[-9],[-5],[5],[5],[-8],[7],[6],[-7],[2],[-4],[9],[-6],[-6],[-2],[4],[-3],[3],[1],[5],[-7],[1],[-2],[1],[-3],[-9],[9],[-3],[-1],[-4],[9],[10],[-2],[1],[-2],[-1],[9],[10],[-10],[-5],[6],[-8],[-3],[2],[9],[6],[-5],[-10],[-9],[4],[7],[-10],[-9],[2],[-3],[-10],[10],[-6],[10],[-7],[-5],[5],[-9],[-10],[-5],[-2],[2],[-1],[-4],[8],[-9],[-8],[8],[-1],[1],[7],[1],[7],[6],[10],[4],[6],[-6],[4],[-1],[9],[5],[-8],[-1],[-1],[-9],[10],[10],[-1],[7],[-2],[5],[-6],[6],[-1],[6],[-6],[7],[-4],[-6],[-3],[-3],[6],[2],[-4],[1]], dtype = "int16")#candidate|2827|(1560, 1)|const|int16
call_2826 = relay.TupleGetItem(func_2040_call(relay.reshape(const_2785.astype('int16'), []), relay.reshape(const_2827.astype('int16'), [13, 10, 12]), ), 2)
call_2828 = relay.TupleGetItem(func_2044_call(relay.reshape(const_2785.astype('int16'), []), relay.reshape(const_2827.astype('int16'), [13, 10, 12]), ), 2)
output = relay.Tuple([bop_2772,call_2783,var_2784,var_2786,bop_2808,bop_2815,bop_2820,call_2826,const_2827,])
output2 = relay.Tuple([bop_2772,call_2787,var_2784,var_2786,bop_2811,bop_2815,bop_2823,call_2828,const_2827,])
func_2842 = relay.Function([var_2764,var_2771,var_2784,var_2786,var_2807,], output)
mod['func_2842'] = func_2842
mod = relay.transform.InferType()(mod)
var_2843 = relay.var("var_2843", dtype = "float32", shape = (2, 3, 14))#candidate|2843|(2, 3, 14)|var|float32
var_2844 = relay.var("var_2844", dtype = "bool", shape = (2, 3, 14))#candidate|2844|(2, 3, 14)|var|bool
var_2845 = relay.var("var_2845", dtype = "float32", shape = (12, 4))#candidate|2845|(12, 4)|var|float32
var_2846 = relay.var("var_2846", dtype = "float32", shape = (220,))#candidate|2846|(220,)|var|float32
var_2847 = relay.var("var_2847", dtype = "bool", shape = (9, 11))#candidate|2847|(9, 11)|var|bool
output = func_2842(var_2843,var_2844,var_2845,var_2846,var_2847,)
func_2848 = relay.Function([var_2843,var_2844,var_2845,var_2846,var_2847,], output)
mutated_mod['func_2848'] = func_2848
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_2850 = relay.TupleGetItem(func_1909_call(), 0)
call_2851 = relay.TupleGetItem(func_1910_call(), 0)
func_2250_call = mod.get_global_var('func_2250')
func_2255_call = mutated_mod.get_global_var('func_2255')
var_2872 = relay.var("var_2872", dtype = "float32", shape = (210,))#candidate|2872|(210,)|var|float32
const_2873 = relay.const([False,True,True,True,False,True,False,False,False,False,False,False,True,True,True,False,False,False,False,False,True,False,True,False,True,True,False,False,False,False,True,False,True,False,True,True,False,True,False,False,True,False,False,False,True,False,True,False,False,True,False,True,True,True,True,False,True,True,True,True,False,False,True,True,False,False,True,False,True,True,False,False,True,True,True,True,True,False,True,True,True,True,True,True,True,False,True,False,False,False,True,False,False,False,True,True,False,False,True,True,True,False,True,True,True,False,False,False,False,True,True,False,False,False,True,True,True,False,True,True,True,False,True,True,False,False,False,False,True,False,True,False,False,True,True,True,True,True,True,False,False,False,False,True,True,True,False,False,True,False,True,False,False,False,False,True,False,False,True,True,False,True,False,False,False,True,True,True,True,False,True,True,True,True,False,False,False,True,False,False,True,True,False,True,True,False,False,True,True,False,True,True,True,True,False,True,False,True,False,True,False,False,True,True,True,True,True,False,False,True,False,True,False,False,True,True,True,True,False,False,True,True,True,False], dtype = "bool")#candidate|2873|(224,)|const|bool
const_2874 = relay.const([-6.653938,-5.169455,-9.746329,4.774533,6.996206,-9.496870,9.563906,-1.385692,7.789852,-6.612151,-5.191261,-5.046456,-4.213280,-0.743685,8.173889,-9.061597,-1.020325,0.815663,0.094721,6.700601,-8.296546,-8.091982,-4.957248,-1.804902,-1.120882,-2.053495,-7.812748,6.015782,-0.766507,-5.394622,7.560171,-7.232775,8.488471,-5.186469,-8.654784,-4.272047,2.965591,-6.056805,8.116292,6.559193,8.865508,-0.118970,-0.225967,9.599861,-8.939849,-1.932182,-7.879543,5.841402,-6.694363,-3.911073,9.777925,3.726795,6.845692,-5.300989,-8.425451,-7.995212,2.802174,7.534471,3.739857,5.694615], dtype = "float64")#candidate|2874|(60,)|const|float64
call_2871 = relay.TupleGetItem(func_2250_call(relay.reshape(var_2872.astype('float32'), [7, 2, 15]), relay.reshape(const_2873.astype('bool'), [224,]), relay.reshape(const_2874.astype('float64'), [60,]), ), 2)
call_2875 = relay.TupleGetItem(func_2255_call(relay.reshape(var_2872.astype('float32'), [7, 2, 15]), relay.reshape(const_2873.astype('bool'), [224,]), relay.reshape(const_2874.astype('float64'), [60,]), ), 2)
func_525_call = mod.get_global_var('func_525')
func_527_call = mutated_mod.get_global_var('func_527')
call_2879 = func_525_call()
call_2880 = func_525_call()
output = relay.Tuple([call_2850,call_2871,var_2872,const_2873,const_2874,call_2879,])
output2 = relay.Tuple([call_2851,call_2875,var_2872,const_2873,const_2874,call_2880,])
func_2881 = relay.Function([var_2872,], output)
mod['func_2881'] = func_2881
mod = relay.transform.InferType()(mod)
mutated_mod['func_2881'] = func_2881
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2882 = relay.var("var_2882", dtype = "float32", shape = (210,))#candidate|2882|(210,)|var|float32
func_2881_call = mutated_mod.get_global_var('func_2881')
call_2883 = func_2881_call(var_2882)
output = call_2883
func_2884 = relay.Function([var_2882], output)
mutated_mod['func_2884'] = func_2884
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1165_call = mod.get_global_var('func_1165')
func_1166_call = mutated_mod.get_global_var('func_1166')
call_2907 = relay.TupleGetItem(func_1165_call(), 0)
call_2908 = relay.TupleGetItem(func_1166_call(), 0)
var_2913 = relay.var("var_2913", dtype = "float32", shape = (3, 6))#candidate|2913|(3, 6)|var|float32
bop_2914 = relay.maximum(call_2907.astype('int64'), relay.reshape(var_2913.astype('int64'), relay.shape_of(call_2907))) # shape=(3, 6)
bop_2917 = relay.maximum(call_2908.astype('int64'), relay.reshape(var_2913.astype('int64'), relay.shape_of(call_2908))) # shape=(3, 6)
var_2923 = relay.var("var_2923", dtype = "int64", shape = (3, 6))#candidate|2923|(3, 6)|var|int64
bop_2924 = relay.less_equal(bop_2914.astype('bool'), relay.reshape(var_2923.astype('bool'), relay.shape_of(bop_2914))) # shape=(3, 6)
bop_2927 = relay.less_equal(bop_2917.astype('bool'), relay.reshape(var_2923.astype('bool'), relay.shape_of(bop_2917))) # shape=(3, 6)
output = relay.Tuple([bop_2924,])
output2 = relay.Tuple([bop_2927,])
func_2930 = relay.Function([var_2913,var_2923,], output)
mod['func_2930'] = func_2930
mod = relay.transform.InferType()(mod)
var_2931 = relay.var("var_2931", dtype = "float32", shape = (3, 6))#candidate|2931|(3, 6)|var|float32
var_2932 = relay.var("var_2932", dtype = "int64", shape = (3, 6))#candidate|2932|(3, 6)|var|int64
output = func_2930(var_2931,var_2932,)
func_2933 = relay.Function([var_2931,var_2932,], output)
mutated_mod['func_2933'] = func_2933
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2720_call = mod.get_global_var('func_2720')
func_2721_call = mutated_mod.get_global_var('func_2721')
call_2980 = relay.TupleGetItem(func_2720_call(), 0)
call_2981 = relay.TupleGetItem(func_2721_call(), 0)
func_1941_call = mod.get_global_var('func_1941')
func_1943_call = mutated_mod.get_global_var('func_1943')
call_2983 = relay.TupleGetItem(func_1941_call(), 0)
call_2984 = relay.TupleGetItem(func_1943_call(), 0)
uop_3000 = relay.sinh(call_2980.astype('float32')) # shape=(3, 6)
uop_3002 = relay.sinh(call_2981.astype('float32')) # shape=(3, 6)
var_3010 = relay.var("var_3010", dtype = "float32", shape = (3, 6))#candidate|3010|(3, 6)|var|float32
bop_3011 = relay.greater(uop_3000.astype('bool'), relay.reshape(var_3010.astype('bool'), relay.shape_of(uop_3000))) # shape=(3, 6)
bop_3014 = relay.greater(uop_3002.astype('bool'), relay.reshape(var_3010.astype('bool'), relay.shape_of(uop_3002))) # shape=(3, 6)
func_1909_call = mod.get_global_var('func_1909')
func_1910_call = mutated_mod.get_global_var('func_1910')
call_3017 = relay.TupleGetItem(func_1909_call(), 0)
call_3018 = relay.TupleGetItem(func_1910_call(), 0)
bop_3023 = relay.add(bop_3011.astype('float64'), relay.reshape(call_3017.astype('float64'), relay.shape_of(bop_3011))) # shape=(3, 6)
bop_3026 = relay.add(bop_3014.astype('float64'), relay.reshape(call_3018.astype('float64'), relay.shape_of(bop_3014))) # shape=(3, 6)
func_1434_call = mod.get_global_var('func_1434')
func_1437_call = mutated_mod.get_global_var('func_1437')
call_3044 = func_1434_call(relay.reshape(uop_3000.astype('float32'), [3, 6]))
call_3045 = func_1434_call(relay.reshape(uop_3000.astype('float32'), [3, 6]))
bop_3046 = relay.not_equal(uop_3000.astype('bool'), relay.reshape(call_2980.astype('bool'), relay.shape_of(uop_3000))) # shape=(3, 6)
bop_3049 = relay.not_equal(uop_3002.astype('bool'), relay.reshape(call_2981.astype('bool'), relay.shape_of(uop_3002))) # shape=(3, 6)
func_1290_call = mod.get_global_var('func_1290')
func_1293_call = mutated_mod.get_global_var('func_1293')
call_3059 = relay.TupleGetItem(func_1290_call(relay.reshape(call_3044.astype('bool'), [3, 6])), 0)
call_3060 = relay.TupleGetItem(func_1293_call(relay.reshape(call_3044.astype('bool'), [3, 6])), 0)
output = relay.Tuple([call_2983,bop_3023,call_3044,bop_3046,call_3059,])
output2 = relay.Tuple([call_2984,bop_3026,call_3045,bop_3049,call_3060,])
func_3061 = relay.Function([var_3010,], output)
mod['func_3061'] = func_3061
mod = relay.transform.InferType()(mod)
mutated_mod['func_3061'] = func_3061
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3062 = relay.var("var_3062", dtype = "float32", shape = (3, 6))#candidate|3062|(3, 6)|var|float32
func_3061_call = mutated_mod.get_global_var('func_3061')
call_3063 = func_3061_call(var_3062)
output = call_3063
func_3064 = relay.Function([var_3062], output)
mutated_mod['func_3064'] = func_3064
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2306_call = mod.get_global_var('func_2306')
func_2307_call = mutated_mod.get_global_var('func_2307')
call_3078 = func_2306_call()
call_3079 = func_2306_call()
func_2373_call = mod.get_global_var('func_2373')
func_2374_call = mutated_mod.get_global_var('func_2374')
call_3086 = func_2373_call()
call_3087 = func_2373_call()
bop_3090 = relay.multiply(call_3078.astype('int64'), relay.reshape(call_3086.astype('int64'), relay.shape_of(call_3078))) # shape=(3, 6)
bop_3093 = relay.multiply(call_3079.astype('int64'), relay.reshape(call_3087.astype('int64'), relay.shape_of(call_3079))) # shape=(3, 6)
bop_3109 = relay.mod(bop_3090.astype('float64'), relay.reshape(call_3078.astype('float64'), relay.shape_of(bop_3090))) # shape=(3, 6)
bop_3112 = relay.mod(bop_3093.astype('float64'), relay.reshape(call_3079.astype('float64'), relay.shape_of(bop_3093))) # shape=(3, 6)
uop_3123 = relay.asin(bop_3109.astype('float64')) # shape=(3, 6)
uop_3125 = relay.asin(bop_3112.astype('float64')) # shape=(3, 6)
uop_3139 = relay.tan(bop_3109.astype('float32')) # shape=(3, 6)
uop_3141 = relay.tan(bop_3112.astype('float32')) # shape=(3, 6)
output = relay.Tuple([uop_3123,uop_3139,])
output2 = relay.Tuple([uop_3125,uop_3141,])
func_3145 = relay.Function([], output)
mod['func_3145'] = func_3145
mod = relay.transform.InferType()(mod)
output = func_3145()
func_3146 = relay.Function([], output)
mutated_mod['func_3146'] = func_3146
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1711_call = mod.get_global_var('func_1711')
func_1713_call = mutated_mod.get_global_var('func_1713')
call_3147 = relay.TupleGetItem(func_1711_call(), 0)
call_3148 = relay.TupleGetItem(func_1713_call(), 0)
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
var_3172 = relay.var("var_3172", dtype = "float32", shape = (9, 10))#candidate|3172|(9, 10)|var|float32
var_3173 = relay.var("var_3173", dtype = "float32", shape = (9, 10))#candidate|3173|(9, 10)|var|float32
bop_3174 = relay.divide(var_3172.astype('float32'), relay.reshape(var_3173.astype('float32'), relay.shape_of(var_3172))) # shape=(9, 10)
func_907_call = mod.get_global_var('func_907')
func_911_call = mutated_mod.get_global_var('func_911')
const_3187 = relay.const([-0.213868,-5.427031,0.494067,2.265964,-7.476856,-0.932835,-6.748912,9.816912,-2.834845,-9.605794,-2.375990,4.375526,-2.673426,6.843587,-2.310338,-4.442870,-3.665379,-2.601328,1.941258,8.392626,-0.956913,9.487293,-5.015472,-3.106205,-2.052825,-8.574939,8.073478,-8.953496,-0.908700,0.704474,-9.075820,6.799399,-4.957694,-7.751033,0.161721,-3.481410,-0.020303,-1.283958,-6.891559,6.049128,6.422016,6.338823,9.486386,-9.620877,-2.502673,-2.395547,5.277336,3.504708,-6.640872,-8.980501,3.188546,5.314360,-0.598861,1.967521,-8.198058,-0.260933,-1.693371,-1.552633,7.211070,1.075238], dtype = "float64")#candidate|3187|(60,)|const|float64
const_3188 = relay.const(6.103613, dtype = "float32")#candidate|3188|()|const|float32
call_3186 = relay.TupleGetItem(func_907_call(relay.reshape(const_3187.astype('float64'), [60,]), relay.reshape(const_3188.astype('float32'), []), ), 0)
call_3189 = relay.TupleGetItem(func_911_call(relay.reshape(const_3187.astype('float64'), [60,]), relay.reshape(const_3188.astype('float32'), []), ), 0)
output = relay.Tuple([bop_3174,call_3186,const_3187,const_3188,])
output2 = relay.Tuple([bop_3174,call_3189,const_3187,const_3188,])
func_3203 = relay.Function([var_3172,var_3173,], output)
mod['func_3203'] = func_3203
mod = relay.transform.InferType()(mod)
var_3204 = relay.var("var_3204", dtype = "float32", shape = (9, 10))#candidate|3204|(9, 10)|var|float32
var_3205 = relay.var("var_3205", dtype = "float32", shape = (9, 10))#candidate|3205|(9, 10)|var|float32
output = func_3203(var_3204,var_3205,)
func_3206 = relay.Function([var_3204,var_3205,], output)
mutated_mod['func_3206'] = func_3206
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3260 = relay.var("var_3260", dtype = "uint16", shape = (4, 2, 10))#candidate|3260|(4, 2, 10)|var|uint16
var_3261 = relay.var("var_3261", dtype = "uint16", shape = (4, 2, 10))#candidate|3261|(4, 2, 10)|var|uint16
bop_3262 = relay.multiply(var_3260.astype('uint16'), relay.reshape(var_3261.astype('uint16'), relay.shape_of(var_3260))) # shape=(4, 2, 10)
output = relay.Tuple([bop_3262,])
output2 = relay.Tuple([bop_3262,])
F = relay.Function([var_3260,var_3261,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_3260,var_3261,], output2)
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
	relay.transform.SimplifyExpr(),
	relay.transform.InferType(),
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
input_3260= np.array([[[-10,5,2,-8,7,3,7,-4,2,7],[10,-8,2,3,-9,-5,2,8,10,-3]],[[1,-10,-6,1,-6,7,-2,8,4,-10],[-10,6,6,8,-10,-5,-6,9,-9,-10]],[[-9,-3,-4,8,4,-5,-2,-6,2,6],[-4,-5,6,-2,6,-2,-9,-7,3,10]],[[-3,2,3,6,4,9,3,-6,-8,5],[-10,-7,9,-9,1,-2,6,10,5,-7]]], dtype='uint16')
module1.set_input('var_3260', input_3260)
input_3261= np.array([[[9,7,-10,4,3,9,4,10,4,-10],[3,-9,-7,10,-6,1,5,-1,10,10]],[[-3,-7,-7,-4,8,-5,-5,10,3,6],[-9,7,4,6,1,9,6,-3,-3,-5]],[[-1,4,3,8,5,5,2,7,-10,-1],[-10,-7,8,-1,-8,-4,7,5,-3,5]],[[-6,4,-2,3,4,5,6,-2,3,-5],[-2,-10,-8,-8,-7,-9,-3,2,-1,-9]]], dtype='uint16')
module1.set_input('var_3261', input_3261)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_3260, input_3261, )
res3 = intrp3.evaluate()(input_3260, input_3261, )
res4 = intrp4.evaluate()(input_3260, input_3261, )
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
module5.set_input('var_3260', input_3260)
module5.set_input('var_3261', input_3261)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_3260, input_3261, )
res7 = intrp7.evaluate()(input_3260, input_3261, )
res8 = intrp8.evaluate()(input_3260, input_3261, )
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
module9.set_input('var_3260', input_3260)
module9.set_input('var_3261', input_3261)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_3260, input_3261, )
res11 = intrp11.evaluate()(input_3260, input_3261, )
res12 = intrp12.evaluate()(input_3260, input_3261, )
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
module13.set_input('var_3260', input_3260)
module13.set_input('var_3261', input_3261)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_3260, input_3261, )
res15 = intrp15.evaluate()(input_3260, input_3261, )
res16 = intrp16.evaluate()(input_3260, input_3261, )
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
module17.set_input('var_3260', input_3260)
module17.set_input('var_3261', input_3261)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_3260, input_3261, )
res19 = intrp19.evaluate()(input_3260, input_3261, )
res20 = intrp20.evaluate()(input_3260, input_3261, )
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
module21.set_input('var_3260', input_3260)
module21.set_input('var_3261', input_3261)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_3260, input_3261, )
res23 = intrp23.evaluate()(input_3260, input_3261, )
res24 = intrp24.evaluate()(input_3260, input_3261, )
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

'''47: TVMFuncCall
46: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
45: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
44: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
43: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
42: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
41: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
40: tvm::transform::Pass::operator()(tvm::IRModule) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
36: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
35: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
34: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
33: tvm::transform::Pass::operator()(tvm::IRModule) const
32: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
30: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
29: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
28: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
27: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
26: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
25: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
24: _ZN3tvm5relay9transform22Devic
23: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
22: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
21: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
20: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
18: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
17: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
15: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
14: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
8: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
7: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
6: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
5: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
3: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
2: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
1: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
0: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)

'''