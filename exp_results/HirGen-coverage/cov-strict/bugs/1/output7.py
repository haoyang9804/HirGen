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
var_176 = relay.var("var_176", dtype = "bool", shape = (16, 9, 8))#candidate|176|(16, 9, 8)|var|bool
var_177 = relay.var("var_177", dtype = "bool", shape = (16, 9, 8))#candidate|177|(16, 9, 8)|var|bool
bop_178 = relay.logical_and(var_176.astype('bool'), relay.reshape(var_177.astype('bool'), relay.shape_of(var_176))) # shape=(16, 9, 8)
bop_182 = relay.bitwise_xor(var_176.astype('uint64'), relay.reshape(bop_178.astype('uint64'), relay.shape_of(var_176))) # shape=(16, 9, 8)
uop_185 = relay.acosh(var_177.astype('float32')) # shape=(16, 9, 8)
uop_194 = relay.asin(bop_182.astype('float32')) # shape=(16, 9, 8)
output = relay.Tuple([uop_185,uop_194,])
output2 = relay.Tuple([uop_185,uop_194,])
func_196 = relay.Function([var_176,var_177,], output)
mod['func_196'] = func_196
mod = relay.transform.InferType()(mod)
var_197 = relay.var("var_197", dtype = "bool", shape = (16, 9, 8))#candidate|197|(16, 9, 8)|var|bool
var_198 = relay.var("var_198", dtype = "bool", shape = (16, 9, 8))#candidate|198|(16, 9, 8)|var|bool
output = func_196(var_197,var_198,)
func_199 = relay.Function([var_197,var_198,], output)
mutated_mod['func_199'] = func_199
mutated_mod = relay.transform.InferType()(mutated_mod)
var_219 = relay.var("var_219", dtype = "float64", shape = (5, 1, 8))#candidate|219|(5, 1, 8)|var|float64
const_220 = relay.const([[[-3.800154,-5.729894,-4.824332,4.585518,1.174828,5.482292,-0.764108,-5.600356],[-1.646571,-9.851321,3.553247,-4.356364,-5.235324,8.011049,6.457978,-4.387562],[-3.181755,4.016317,3.106385,-6.891822,5.829275,-9.399074,9.350594,2.371383],[-5.018382,1.070991,6.730728,2.788513,-4.676869,2.902153,1.637533,8.679543],[3.567295,1.653693,5.083121,8.318762,5.263255,-2.727900,-6.138429,-0.683763]],[[4.654161,-2.936295,-2.226152,8.839480,3.094919,-6.914389,-6.867883,-7.537304],[4.924217,-5.234320,4.653595,5.491668,-6.303090,8.887404,9.506768,8.287077],[-3.975700,8.836279,5.146183,0.837734,1.471473,-6.367156,3.029818,3.960321],[-2.124519,3.238410,4.664650,-8.617325,2.947795,9.023727,-2.079494,2.730209],[-7.110524,9.049055,6.691705,-0.977433,-1.184189,-0.998128,-5.518713,-9.708011]],[[-9.713003,-0.573656,-6.725536,7.932650,9.959491,-9.255000,7.384003,-4.126981],[-7.585553,2.361962,-8.732931,6.787185,-3.403932,-6.302438,2.644967,7.770886],[-4.503900,-9.783725,1.247578,-0.804447,-9.796074,-9.809641,8.114415,9.741289],[-8.745451,6.293877,6.273954,8.160435,0.371640,-3.293258,3.039548,-8.808549],[9.356664,-9.777312,-6.211351,7.308725,-5.277654,-2.499324,-2.132712,3.323806]],[[-7.570526,4.953991,9.351295,-3.388375,-7.696445,-1.286300,-4.541495,-9.775712],[4.917747,-9.172295,2.115007,-3.184112,5.554549,4.690376,-8.715980,-7.284209],[-0.856137,-9.311668,3.213889,-2.860367,5.857466,8.431331,2.821261,-0.865171],[-8.618992,-9.810667,-9.849597,3.347176,2.119353,-8.820287,-9.101602,-8.218512],[-5.652424,-3.108575,1.530895,-4.865028,1.701074,2.280521,1.289677,2.342053]],[[-3.096162,0.082538,-8.659005,5.599973,-7.994562,-8.968257,-8.516097,4.018381],[3.992745,-8.059042,-2.189920,-3.510488,-7.752780,-9.736978,0.687708,-2.340259],[8.373743,-3.445103,-9.658795,6.007513,2.205923,0.424993,-8.078462,7.273892],[7.345741,9.977984,-9.065176,7.184430,4.694430,-3.627684,-0.049374,-9.645071],[-7.747546,2.540234,-0.644937,5.065557,-9.021448,-1.478196,0.015186,4.634701]]], dtype = "float64")#candidate|220|(5, 5, 8)|const|float64
bop_221 = relay.mod(var_219.astype('float64'), const_220.astype('float64')) # shape=(5, 5, 8)
func_196_call = mod.get_global_var('func_196')
func_199_call = mutated_mod.get_global_var('func_199')
const_226 = relay.const([True,True,False,True,True,False,False,True,False,True,False,True,False,True,True,True,False,False,True,False,False,True,True,False,True,False,True,False,True,False,False,False,True,False,True,False,False,True,True,False,False,False,False,False,True,True,True,True,True,True,False,True,False,True,False,True,False,True,True,True,True,False,True,False,False,False,False,True,True,False,True,True,False,True,True,True,False,True,False,True,False,False,True,False,False,True,True,False,False,True,True,False,True,False,False,True,False,True,False,True,True,False,False,True,True,True,False,False,False,False,True,False,True,False,False,True,True,False,True,False,True,False,False,False,False,False,True,False,True,False,True,False,False,True,True,True,False,True,True,True,False,True,True,True,True,False,False,True,False,True,True,True,True,True,True,True,True,False,True,True,False,False,True,False,True,True,False,True,False,True,False,False,False,False,True,False,False,True,True,False,False,False,True,True,True,False,False,False,False,True,True,True,True,True,True,False,False,True,False,False,True,False,True,True,False,False,True,False,False,False,False,False,False,True,False,True,True,False,False,False,False,True,True,True,False,False,False,False,False,False,True,True,False,False,False,True,False,False,True,False,False,False,True,True,True,True,False,True,True,False,True,True,False,False,True,False,False,True,True,False,True,False,True,True,False,False,False,False,False,False,True,False,False,False,True,True,True,False,False,False,False,True,False,False,True,True,True,False,False,False,False,True,False,False,False,False,False,True,False,False,True,True,False,True,True,False,False,False,False,True,True,False,False,True,True,False,False,False,False,False,False,False,True,False,False,True,False,False,False,False,False,True,False,True,False,True,True,True,False,True,False,True,True,False,False,False,False,False,False,False,False,False,True,True,False,True,False,False,False,True,True,False,False,True,True,True,False,False,False,False,True,False,True,True,False,True,True,True,True,False,True,True,False,False,False,True,False,True,True,False,False,False,False,False,True,False,True,False,False,True,False,False,True,False,True,False,True,False,True,True,False,False,False,True,True,True,False,True,False,True,True,False,False,True,False,True,True,False,True,False,True,False,False,True,False,True,True,True,True,False,False,False,False,True,True,True,False,True,False,False,False,True,False,False,True,False,False,False,False,True,False,False,True,False,True,True,True,False,True,True,False,True,True,False,False,False,False,False,True,False,False,True,False,False,False,True,True,False,True,True,True,False,True,True,False,False,False,False,False,True,True,False,True,False,True,True,False,True,False,True,True,False,False,True,True,False,False,False,False,False,True,False,False,True,True,False,True,True,False,False,True,True,False,False,True,True,True,True,False,True,False,False,False,True,True,True,True,False,True,False,False,False,False,False,True,True,False,True,False,True,True,True,False,True,True,False,True,True,True,True,False,False,True,False,True,True,True,False,True,False,False,True,True,False,True,False,True,True,True,False,False,False,True,True,False,True,True,True,False,False,False,False,False,False,True,True,True,False,False,False,True,False,True,False,True,False,False,False,True,False,False,False,False,False,True,False,True,False,True,True,False,False,True,True,False,False,False,True,True,False,True,False,True,False,False,False,False,False,False,True,False,True,True,True,True,False,True,False,False,True,True,True,True,False,False,True,True,False,False,False,True,True,False,False,True,False,False,True,False,False,False,True,True,False,False,False,False,True,False,True,False,True,False,True,False,False,False,True,False,False,True,True,True,True,True,False,True,True,True,True,False,False,False,True,False,False,False,False,False,False,True,False,False,True,False,False,True,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,True,False,False,True,False,True,True,True,True,False,True,True,True,False,False,False,False,False,False,False,True,True,False,True,True,False,True,False,True,True,True,False,False,True,True,False,False,False,False,False,True,True,True,False,False,True,False,False,True,False,False,False,True,False,True,True,True,False,True,False,False,False,False,False,True,True,False,False,True,False,False,False,False,True,False,False,False,False,False,True,True,True,False,False,True,True,True,False,False,True,False,False,True,False,False,True,True,False,True,True,True,True,True,True,False,False,True,False,False,True,True,True,False,True,False,False,True,True,False,True,False,True,True,False,True,True,True,True,True,False,False,False,True,True,True,True,True,False,True,False,False,False,True,False,False,True,False,True,True,True,False,True,False,True,True,True,False,False,False,True,False,False,False,True,False,True,False,True,False,True,True,False,False,True,False,False,False,True,True,True,False,True,True,False,False,False,True,False,True,False,True,True,False,True,True,False,True,True,False,True,True,True,True,True,False,True,True,True,False,False,False,False,True,True,False,True,False,False,True,True,False,False,False,False,False,True,True,True,True,True,False,False,False,True,True,True,False,False,False,True,False,False,True,True,False,False,False,False,False,True,True,False,True,True,False,True,True,True,False,False,False,False,False,True,True,True,False,False,True,False,True,True,False,False,False,False,False,False,False,False,True,True,False,False,True,True,False,False,False,False,False,True,True,False,False,False,True,False,False,False,False,True,True,False,False,True,True,False,True,True,True,False,True,True,False,False,False,False,False,True,True,False,False,False,False,False,False,True,False,True,True,False,False,False,False,False,True,True,True,False,True,False,False,False,True,False,False,False,True,False,True,False,False,True,False,False,True,False,True,True,True,False,True,False,False,True,False,True,True,True,True,False,True,True,False,True,True,False,True,False,False,False,False,False,True,False,False,False,True,True,True,False,True,False,False,True,True,True,False,False], dtype = "bool")#candidate|226|(1152,)|const|bool
call_225 = relay.TupleGetItem(func_196_call(relay.reshape(const_226.astype('bool'), [16, 9, 8]), relay.reshape(const_226.astype('bool'), [16, 9, 8]), ), 0)
call_227 = relay.TupleGetItem(func_199_call(relay.reshape(const_226.astype('bool'), [16, 9, 8]), relay.reshape(const_226.astype('bool'), [16, 9, 8]), ), 0)
bop_232 = relay.left_shift(const_226.astype('uint8'), relay.reshape(call_225.astype('uint8'), relay.shape_of(const_226))) # shape=(1152,)
bop_235 = relay.left_shift(const_226.astype('uint8'), relay.reshape(call_227.astype('uint8'), relay.shape_of(const_226))) # shape=(1152,)
uop_239 = relay.atan(const_220.astype('float32')) # shape=(5, 5, 8)
uop_242 = relay.log2(uop_239.astype('float32')) # shape=(5, 5, 8)
output = relay.Tuple([bop_221,bop_232,uop_242,])
output2 = relay.Tuple([bop_221,bop_235,uop_242,])
func_245 = relay.Function([var_219,], output)
mod['func_245'] = func_245
mod = relay.transform.InferType()(mod)
var_246 = relay.var("var_246", dtype = "float64", shape = (5, 1, 8))#candidate|246|(5, 1, 8)|var|float64
output = func_245(var_246)
func_247 = relay.Function([var_246], output)
mutated_mod['func_247'] = func_247
mutated_mod = relay.transform.InferType()(mutated_mod)
const_276 = relay.const([[[8.532733,-1.213618,-1.321799],[-3.358663,5.560471,1.748832],[7.043205,5.953162,0.748706],[3.916383,9.860296,2.446201],[-4.275319,-8.907502,-1.788674],[-7.403556,-2.258194,-3.673532],[-1.938268,1.882000,1.690225],[5.664852,-0.697452,3.167683]],[[-2.414685,-6.947402,-2.836683],[0.366708,2.911861,-4.837341],[8.722458,7.135730,-6.837400],[-6.564578,2.198823,-2.952710],[-1.645866,0.361372,-4.563344],[-7.497106,0.094616,-8.676898],[-1.797859,0.674623,6.530860],[1.505594,-5.959544,-8.286656]],[[3.905276,3.911947,-8.798049],[2.693622,3.886929,-7.984793],[0.403321,0.128547,4.144051],[4.261444,6.743412,7.257789],[-9.306306,-9.277920,6.423133],[1.127046,9.209063,2.843153],[-6.164055,8.567873,6.315375],[9.057744,-2.147896,-4.806614]],[[-7.862020,-5.109463,-5.117854],[-0.663932,-0.961270,3.383171],[-3.815772,-6.564290,9.071944],[7.600260,0.053550,4.422923],[-8.555228,4.322355,0.371172],[-9.053126,4.759165,-3.499304],[-9.036289,5.288469,8.151607],[8.216581,2.131610,6.915519]],[[3.090300,0.989615,2.010190],[-9.088914,6.601932,4.962747],[5.799682,3.260334,-3.663044],[-0.361394,-4.488782,8.240941],[-7.604881,9.997454,1.471901],[-8.760754,-2.474368,8.410410],[-8.510286,8.616122,8.698277],[2.201758,9.747628,-2.100566]],[[-6.296488,-6.416191,-3.158030],[-3.292621,-0.988565,-0.493417],[-1.109837,9.825165,-7.539830],[8.273410,-3.789739,2.608884],[0.785749,-1.466371,0.172596],[-0.338444,2.705038,-8.019496],[-8.857473,-2.720954,8.870015],[-8.697919,6.159910,-8.358947]],[[-5.112201,-1.456346,-6.333349],[6.271905,-6.474883,-7.438719],[-4.580072,3.425042,-2.301437],[1.125774,-7.352283,2.326832],[-6.793811,-7.105389,2.135426],[-6.556766,-4.421865,-0.138356],[-5.568773,-2.651547,7.052977],[-0.460855,4.311275,6.382516]],[[8.910722,-4.559254,-4.520541],[9.275602,-8.656850,-3.964718],[-5.126687,3.332993,-4.685967],[-9.995483,3.891769,4.614764],[3.389782,-5.410476,-2.343246],[-4.303191,-6.490570,7.762696],[8.277508,-5.613731,-9.517590],[3.728327,8.802157,2.489702]]], dtype = "float32")#candidate|276|(8, 8, 3)|const|float32
uop_277 = relay.sigmoid(const_276.astype('float32')) # shape=(8, 8, 3)
func_245_call = mod.get_global_var('func_245')
func_247_call = mutated_mod.get_global_var('func_247')
const_280 = relay.const([[5.948348],[-1.558215],[4.126907],[-8.031924],[7.619976],[2.106948],[6.914760],[3.764953],[-3.862144],[-6.610678],[-1.883519],[-4.603687],[4.413638],[4.925193],[9.893217],[-7.933362],[2.479547],[-1.667363],[-0.397428],[-2.033243],[6.031012],[0.214927],[-1.827296],[-0.768681],[5.091033],[1.465080],[3.342204],[0.292692],[1.802879],[2.882981],[-4.852131],[7.070697],[-6.021467],[2.862810],[-4.701067],[-0.873417],[8.097301],[1.345477],[-2.075249],[-9.678293]], dtype = "float64")#candidate|280|(40, 1)|const|float64
call_279 = relay.TupleGetItem(func_245_call(relay.reshape(const_280.astype('float64'), [5, 1, 8])), 0)
call_281 = relay.TupleGetItem(func_247_call(relay.reshape(const_280.astype('float64'), [5, 1, 8])), 0)
uop_285 = relay.rsqrt(uop_277.astype('float64')) # shape=(8, 8, 3)
uop_289 = relay.log(uop_285.astype('float64')) # shape=(8, 8, 3)
output = relay.Tuple([call_279,const_280,uop_289,])
output2 = relay.Tuple([call_281,const_280,uop_289,])
func_293 = relay.Function([], output)
mod['func_293'] = func_293
mod = relay.transform.InferType()(mod)
output = func_293()
func_294 = relay.Function([], output)
mutated_mod['func_294'] = func_294
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_311 = relay.TupleGetItem(func_293_call(), 0)
call_312 = relay.TupleGetItem(func_294_call(), 0)
var_319 = relay.var("var_319", dtype = "float64", shape = (5, 5, 8))#candidate|319|(5, 5, 8)|var|float64
bop_320 = relay.add(call_311.astype('uint16'), relay.reshape(var_319.astype('uint16'), relay.shape_of(call_311))) # shape=(5, 5, 8)
bop_323 = relay.add(call_312.astype('uint16'), relay.reshape(var_319.astype('uint16'), relay.shape_of(call_312))) # shape=(5, 5, 8)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_324 = relay.TupleGetItem(func_293_call(), 2)
call_325 = relay.TupleGetItem(func_294_call(), 2)
func_196_call = mod.get_global_var('func_196')
func_199_call = mutated_mod.get_global_var('func_199')
var_327 = relay.var("var_327", dtype = "bool", shape = (1152,))#candidate|327|(1152,)|var|bool
call_326 = relay.TupleGetItem(func_196_call(relay.reshape(var_327.astype('bool'), [16, 9, 8]), relay.reshape(var_327.astype('bool'), [16, 9, 8]), ), 1)
call_328 = relay.TupleGetItem(func_199_call(relay.reshape(var_327.astype('bool'), [16, 9, 8]), relay.reshape(var_327.astype('bool'), [16, 9, 8]), ), 1)
bop_343 = relay.logical_xor(call_311.astype('uint16'), relay.reshape(bop_320.astype('uint16'), relay.shape_of(call_311))) # shape=(5, 5, 8)
bop_346 = relay.logical_xor(call_312.astype('uint16'), relay.reshape(bop_323.astype('uint16'), relay.shape_of(call_312))) # shape=(5, 5, 8)
bop_353 = relay.greater(call_311.astype('bool'), relay.reshape(bop_343.astype('bool'), relay.shape_of(call_311))) # shape=(5, 5, 8)
bop_356 = relay.greater(call_312.astype('bool'), relay.reshape(bop_346.astype('bool'), relay.shape_of(call_312))) # shape=(5, 5, 8)
const_357 = relay.const([[[True,True,True,False,False,True,False,False],[True,False,False,True,False,False,False,True],[True,False,True,False,False,True,True,False],[False,False,False,False,True,True,False,False],[True,True,True,True,False,True,True,True]],[[True,False,False,False,False,False,True,True],[False,False,True,True,True,False,True,True],[False,True,True,True,False,True,False,True],[False,True,False,False,False,False,True,True],[False,True,True,False,False,False,True,False]],[[True,False,True,False,True,False,False,True],[True,True,True,False,True,True,True,True],[False,False,False,False,False,True,True,False],[True,True,False,True,True,True,True,False],[True,True,True,False,True,True,False,True]],[[False,True,True,True,False,False,True,False],[False,True,False,False,False,True,False,True],[False,False,False,False,True,False,False,True],[True,True,True,False,False,True,True,True],[False,False,False,False,True,True,False,True]],[[False,False,False,True,False,False,False,False],[True,True,False,False,True,True,True,False],[False,True,False,True,False,False,False,True],[False,False,True,True,False,False,True,False],[False,True,True,False,True,False,True,False]]], dtype = "bool")#candidate|357|(5, 5, 8)|const|bool
bop_358 = relay.floor_divide(bop_353.astype('float64'), relay.reshape(const_357.astype('float64'), relay.shape_of(bop_353))) # shape=(5, 5, 8)
bop_361 = relay.floor_divide(bop_356.astype('float64'), relay.reshape(const_357.astype('float64'), relay.shape_of(bop_356))) # shape=(5, 5, 8)
uop_363 = relay.tan(bop_320.astype('float64')) # shape=(5, 5, 8)
uop_365 = relay.tan(bop_323.astype('float64')) # shape=(5, 5, 8)
var_370 = relay.var("var_370", dtype = "float64", shape = (5, 5, 8))#candidate|370|(5, 5, 8)|var|float64
bop_371 = relay.multiply(uop_363.astype('int16'), relay.reshape(var_370.astype('int16'), relay.shape_of(uop_363))) # shape=(5, 5, 8)
bop_374 = relay.multiply(uop_365.astype('int16'), relay.reshape(var_370.astype('int16'), relay.shape_of(uop_365))) # shape=(5, 5, 8)
bop_379 = relay.less(uop_363.astype('bool'), relay.reshape(bop_320.astype('bool'), relay.shape_of(uop_363))) # shape=(5, 5, 8)
bop_382 = relay.less(uop_365.astype('bool'), relay.reshape(bop_323.astype('bool'), relay.shape_of(uop_365))) # shape=(5, 5, 8)
output = relay.Tuple([call_324,call_326,var_327,bop_358,bop_371,bop_379,])
output2 = relay.Tuple([call_325,call_328,var_327,bop_361,bop_374,bop_382,])
func_384 = relay.Function([var_319,var_327,var_370,], output)
mod['func_384'] = func_384
mod = relay.transform.InferType()(mod)
var_385 = relay.var("var_385", dtype = "float64", shape = (5, 5, 8))#candidate|385|(5, 5, 8)|var|float64
var_386 = relay.var("var_386", dtype = "bool", shape = (1152,))#candidate|386|(1152,)|var|bool
var_387 = relay.var("var_387", dtype = "float64", shape = (5, 5, 8))#candidate|387|(5, 5, 8)|var|float64
output = func_384(var_385,var_386,var_387,)
func_388 = relay.Function([var_385,var_386,var_387,], output)
mutated_mod['func_388'] = func_388
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_390 = relay.TupleGetItem(func_293_call(), 0)
call_391 = relay.TupleGetItem(func_294_call(), 0)
output = relay.Tuple([call_390,])
output2 = relay.Tuple([call_391,])
func_392 = relay.Function([], output)
mod['func_392'] = func_392
mod = relay.transform.InferType()(mod)
output = func_392()
func_393 = relay.Function([], output)
mutated_mod['func_393'] = func_393
mutated_mod = relay.transform.InferType()(mutated_mod)
const_431 = relay.const([[[-1,-10,-5,5,2,7],[-5,1,-9,-10,10,-7],[-5,3,-4,-5,-10,-5],[9,2,8,-7,4,-5],[-2,7,-6,-6,5,-6],[8,3,8,3,-2,7],[2,-4,-3,-9,-5,-6],[-3,2,2,7,1,6],[-2,7,3,3,-9,2],[-6,-4,1,2,2,2]],[[-3,-4,10,4,-10,-3],[4,-8,5,-4,3,3],[-9,8,10,5,-8,2],[10,10,-6,3,-1,4],[1,6,-9,9,-6,9],[-4,3,-5,10,10,7],[2,-9,-1,10,2,-7],[7,-9,1,-10,4,9],[-8,-9,10,6,6,-7],[-10,-10,-5,3,10,-4]],[[10,10,-9,5,7,-8],[10,8,-1,4,4,-4],[-10,8,9,-5,-7,-6],[1,6,5,-6,7,-2],[1,-10,1,-3,-2,-3],[4,-3,10,-9,-1,2],[6,3,-1,4,5,8],[-1,-7,1,-6,-7,-6],[-8,-9,-10,7,8,-1],[-6,-3,-2,-7,10,-6]],[[3,6,10,-8,-6,7],[9,-4,-5,5,2,4],[-9,-7,-4,-9,-4,8],[6,-2,4,-10,3,-9],[-9,-2,2,-6,-7,-9],[-5,-10,3,1,-4,6],[2,1,2,-8,4,-3],[-5,9,-2,-3,-7,4],[-10,1,-9,2,-5,-1],[-4,-9,8,-7,10,5]],[[2,-9,-2,1,9,6],[3,1,10,4,-8,2],[-4,9,5,-8,-1,-8],[-9,4,4,-7,-7,-2],[8,-3,-8,4,-6,-8],[6,-7,-2,-9,7,10],[-3,-6,-5,7,6,-2],[5,-10,9,-1,3,-2],[-7,5,-6,3,4,-10],[-10,2,7,2,-10,-4]],[[-10,3,-6,-8,6,5],[10,1,-7,-3,-8,-3],[-9,-10,9,9,10,3],[-7,-7,4,-6,9,-10],[-8,3,10,-2,3,1],[5,-1,1,3,5,-2],[-3,7,8,8,2,4],[9,2,-10,-4,-10,4],[-9,1,6,-6,4,8],[-1,-9,-10,-5,-1,-1]],[[-8,9,-9,1,-7,-5],[8,9,3,3,6,6],[-7,4,-6,10,-9,-3],[-10,-10,7,-10,-7,6],[-6,5,-10,-7,2,4],[6,3,5,-8,10,-1],[-5,1,-10,5,5,8],[-3,-4,-8,-4,-4,-4],[2,-3,-9,-9,-5,-8],[5,-3,-3,8,7,10]],[[2,-2,2,-5,-7,7],[5,2,-4,7,2,-4],[10,-3,-3,4,7,-4],[-3,3,9,6,4,4],[-8,1,-10,9,6,10],[-2,-7,-2,10,-9,-5],[-3,-2,1,-8,6,3],[4,-7,1,-2,-2,4],[1,9,-8,-10,-7,1],[6,-9,9,-10,1,5]],[[-5,-6,7,2,7,8],[8,-8,8,-5,9,9],[-7,10,6,-1,-4,9],[9,9,-7,-1,5,-1],[9,1,3,5,-9,-7],[2,-8,-5,8,-2,-9],[-4,-3,-10,-3,-2,4],[-4,-3,-5,10,8,-6],[6,-8,8,5,-8,2],[3,-4,10,8,8,-9]],[[-2,4,6,5,-1,10],[2,3,4,-5,-1,-1],[8,6,-1,-3,-3,-4],[-5,3,-4,9,-10,6],[-8,-4,4,9,-5,10],[-10,-10,9,6,-5,-2],[-3,6,5,-1,5,-1],[1,10,9,-1,-4,3],[-4,-9,-9,-7,4,-8],[-5,2,-6,-8,-7,2]],[[-10,-4,-5,8,-8,-8],[8,9,-9,8,8,6],[5,1,-4,-5,-6,-1],[-7,-10,-5,-10,-4,-4],[-8,8,4,-9,10,10],[-3,-10,-5,-9,1,-5],[-8,9,6,6,10,9],[10,5,-8,-5,2,-9],[8,7,-9,-4,-7,-10],[8,6,-2,4,-3,-1]],[[3,-3,1,10,-3,-7],[-9,-1,5,9,9,-3],[-10,-2,-5,-5,9,3],[-1,-7,10,-7,6,7],[-1,-9,2,3,1,5],[10,-6,6,-7,2,1],[10,3,4,-9,6,5],[-10,-10,-6,8,-2,5],[2,7,7,-8,-4,3],[10,-4,2,4,10,2]]], dtype = "uint32")#candidate|431|(12, 10, 6)|const|uint32
var_432 = relay.var("var_432", dtype = "uint32", shape = (12, 10, 6))#candidate|432|(12, 10, 6)|var|uint32
bop_433 = relay.bitwise_xor(const_431.astype('uint32'), relay.reshape(var_432.astype('uint32'), relay.shape_of(const_431))) # shape=(12, 10, 6)
bop_436 = relay.equal(bop_433.astype('bool'), relay.reshape(const_431.astype('bool'), relay.shape_of(bop_433))) # shape=(12, 10, 6)
output = bop_436
output2 = bop_436
func_441 = relay.Function([var_432,], output)
mod['func_441'] = func_441
mod = relay.transform.InferType()(mod)
var_442 = relay.var("var_442", dtype = "uint32", shape = (12, 10, 6))#candidate|442|(12, 10, 6)|var|uint32
output = func_441(var_442)
func_443 = relay.Function([var_442], output)
mutated_mod['func_443'] = func_443
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_468 = relay.TupleGetItem(func_293_call(), 1)
call_469 = relay.TupleGetItem(func_294_call(), 1)
output = call_468
output2 = call_469
func_470 = relay.Function([], output)
mod['func_470'] = func_470
mod = relay.transform.InferType()(mod)
output = func_470()
func_471 = relay.Function([], output)
mutated_mod['func_471'] = func_471
mutated_mod = relay.transform.InferType()(mutated_mod)
var_504 = relay.var("var_504", dtype = "uint32", shape = (12, 10, 3))#candidate|504|(12, 10, 3)|var|uint32
const_505 = relay.const([[[-9,-8,1],[-10,-3,6],[2,8,-4],[-8,2,-2],[7,-10,3],[4,-4,-4],[10,9,-10],[-1,-9,-8],[5,-5,-9],[8,6,-6]],[[-7,-4,1],[-9,9,-6],[4,-5,8],[5,1,-4],[8,4,-1],[6,1,2],[-7,2,-8],[-7,8,7],[-5,-1,4],[-7,-7,-8]],[[-1,8,-2],[3,1,7],[8,6,-6],[-1,-5,7],[10,-4,5],[7,4,-2],[-6,-8,-2],[8,9,-5],[-7,7,-1],[-5,-10,-7]],[[-9,-5,-2],[9,-1,6],[-2,-5,-10],[-7,7,3],[10,-6,-5],[9,9,-8],[10,8,-4],[7,3,-8],[-4,-2,-9],[-6,2,-1]],[[-7,2,-4],[-7,-8,8],[-1,10,10],[-9,-3,5],[5,-2,5],[6,6,10],[-10,8,3],[-2,2,-10],[-1,7,9],[7,3,-5]],[[-7,-2,10],[-4,-10,8],[8,-10,3],[-5,9,8],[3,-6,-7],[7,-1,-6],[5,-3,2],[-8,3,-5],[10,-9,-3],[10,4,-4]],[[-4,-2,4],[-8,-10,1],[2,4,-9],[-10,-8,-9],[7,7,-3],[10,-4,4],[2,4,-3],[-6,6,8],[9,-6,8],[10,-5,5]],[[-7,-9,9],[8,5,5],[-7,-1,-1],[10,-9,-10],[4,-10,-6],[10,2,-6],[-1,4,6],[-2,-2,-8],[3,6,-6],[9,-6,-2]],[[-6,10,10],[-10,-10,-6],[8,9,-2],[3,-7,-9],[5,5,-7],[-6,7,-4],[1,7,10],[8,-3,8],[-4,-5,9],[-3,10,-6]],[[5,1,-7],[9,8,-9],[-1,-8,6],[2,-6,2],[-8,8,3],[6,-5,9],[3,9,-6],[-4,7,-1],[-8,2,9],[7,-1,9]],[[8,-3,4],[4,2,3],[-4,-4,6],[4,10,6],[-4,-9,8],[8,-3,9],[-2,7,8],[1,4,5],[-3,-6,8],[2,-4,4]],[[7,10,-10],[-9,1,8],[5,-3,7],[3,-8,-2],[-10,6,-5],[10,10,-4],[-6,-5,-1],[8,4,-7],[1,10,-8],[4,7,5]]], dtype = "uint32")#candidate|505|(12, 10, 3)|const|uint32
bop_506 = relay.less_equal(var_504.astype('bool'), relay.reshape(const_505.astype('bool'), relay.shape_of(var_504))) # shape=(12, 10, 3)
func_384_call = mod.get_global_var('func_384')
func_388_call = mutated_mod.get_global_var('func_388')
const_512 = relay.const([-1.584236,5.131764,6.593221,4.442206,6.565001,9.236913,2.424540,-9.091300,-0.306303,-3.211284,-0.364488,-3.508882,-6.684150,-1.013587,-9.240478,-8.069388,-2.664978,0.864639,6.970455,7.807147,8.387212,7.090287,-2.956406,8.619961,5.611407,-1.813361,-3.793263,-6.962450,3.160664,-8.561479,7.804085,8.680074,-5.804254,-1.363087,2.572432,-2.826509,-4.994676,7.465863,6.030293,2.040404,-1.338554,-1.997430,-7.417718,9.515115,-6.813777,1.273021,-0.994489,5.121312,-7.571034,-8.201324,7.486741,-0.894469,-3.999895,-5.628632,-2.011456,-9.261219,5.069382,-7.662018,-0.347740,-8.837495,5.674373,-0.924502,-0.528642,0.033976,4.918619,-2.773688,-5.121506,-7.147888,-9.733644,6.159587,-0.741136,3.940973,-0.842841,0.272062,0.950023,9.280828,-4.023891,7.387622,6.340259,3.057265,-5.071400,-6.767987,-1.095962,-3.214803,1.513703,-6.238387,8.928844,-3.213826,9.780304,-2.759789,9.247236,4.040770,-9.088164,-5.105790,8.865885,-0.041651,8.543553,4.275668,-3.430658,-8.754292,1.718662,6.635261,-4.285875,-1.174282,-5.193322,-2.432267,3.393026,3.566639,-9.301471,4.672976,7.379464,-1.086714,-8.607113,-5.844837,3.704486,2.588659,8.254819,1.758599,-4.113012,9.096081,2.883646,-7.994846,-4.615493,-3.489019,9.719240,-7.096449,3.834955,0.419970,-9.461195,-7.591616,-2.677782,-5.165341,1.280592,4.189223,-2.945116,-3.116154,-2.662686,-8.182624,-6.566222,-6.122822,-4.286296,-7.607205,1.037962,-9.149747,3.670259,2.069981,-4.847027,-7.432750,3.453209,5.895345,-6.556591,3.731850,-4.834274,7.946965,-4.293388,-1.878378,8.863429,1.427590,-9.580902,-5.206470,-8.597931,7.270223,-8.750115,-8.455892,4.402080,0.038512,5.864993,-0.143195,-9.785547,-7.310599,4.452771,9.506508,4.581624,5.145895,-9.350757,-1.957474,7.798669,2.277586,5.612098,-5.951462,9.722709,-6.532342,-5.848394,-8.083434,6.362980,-3.511159,6.637077,7.174431,-1.490232,-4.251998,-7.540503,4.360178,-6.096491,1.772400,4.441814,-0.348379,6.025620,4.380496,-4.769028,-0.425377], dtype = "float64")#candidate|512|(200,)|const|float64
const_513 = relay.const([[False,False,True,True,False,False,True,True,False,True,True,False,True,False,False,True,True,False,False,True,True,True,True,True,True,False,False,True,False,True,True,False,True,False,False,False,False,True,True,True,False,False,True,True,False,True,False,True,True,True,True,True,False,False,False,True,False,False,False,False,True,True,True,True,False,True,True,False,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,False,False,False,True,True,False,False,True,False,True,False,True,True,True,False,False,False,False,True,True,True,False,True,True,False,False,False,True,False,False,False,False,False,False,False,True,True,False,True,True,True,True,False,True,True,False,True,True,False,False,False,False,False,True,True,False,False,True,False,False,True,False,True,True,False,True,True,True,True,False,False,False,True,False,True,False,False,False,True,True,True,True,True,True,True,False,False,True,True,False,True,False,False,False,True,False,True,False,False,False,False,False,True,False,True,False,False,True,True,False,False,False,True,True,True,False,True,True,True,False,True,True,False,True,True,False,False,True,False,False,True,True,False,False,True,True,True,True,True,False,True,True,False,True,True,True,True,False,True,True,True,False,False,True,False,True,True,False,False,False,False,False,True,False,False,False,False,True,True,True,True,True,False,True,False,True,True,True,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,True,False,True,True,False,False,True,False,False,True,True,True,False,True,True,False,True,False,False,True,False,True,False,True,False,True,False,False,False,False,False,True,True,False,True,True,False,False,True,True,True,False,False,True,True,True,True,True,True,False,False,False,True,True,True,True,False,False,True,False,False,False,False,True,False,True,False,False,True,True,True,False,True,True,False,True,True,True,False,False,True,False,False,False,True,False,True,False,False,True,False,False,True,False,False,True,True,False,True,False,False,True,True,True,False,True,False,True,False,False,False,False,True,False,False,False,False,False,False,False,True,True,True,False,True,True,True,True,True,False,True,True,True,False,True,False,True,True,True,False,False,True,False,True,False,False,True,False,False,False,True,True,True,False,True,False,True,False,True,False,True,True,False,False,True,True,False,True,False,False,True,False,True,True,True,True,True,True,False,False,True,True,True,False,True,True,False,False,True,False,False,False,True,False,True,False,True,True,True,False,True,False,False,True,True,False,False,True,True,False,True,False,True,False,False,False,True,False,False,True,False,True,True,True,True,False,False,True,False,True,True,True,False,True,False,True,True,True,False,False,True,True,False,True,False,False,True,True,True,False,False,True,True,False,True,False,False,True,True,False,False,False,False,False,False,False,False,True,True,False,False,True,False,False,False,False,True,True,True,False,True,False,True,False,False,False,True,False,True,False,True,False,True,True,False,True,True,False,False,True,True,False,False,True,True,False,True,False,True,False,False,True,False,True,True,False,False,False,True,True,True,False,True,False,True,False,True,False,False,True,True,True,False,True,False,True,True,True,True,True,False,True,False,False,False,True,True,False,False,False,False,True,False,True,True,True,True,False,True,False,True,True,True,True,False,False,False,False,True,True,True,True,False,True,False,True,False,True,True,False,True,True,True,True,True,False,False,False,False,True,False,False,False,False,True,True,False,False,True,True,True,False,True,False,True,True,True,True,False,False,False,True,False,True,False,True,False,False,True,False,True,False,False,False,False,False,True,False,False,False,True,True,False,False,True,True,True,False,False,True,True,False,False,True,False,False,False,False,False,True,False,False,True,True,False,True,True,True,True,True,True,True,False,True,True,False,False,True,False,False,False,True,True,True,False,True,True,False,True,False,False,True,False,True,False,False,True,True,True,False,False,False,True,True,True,True,True,True,False,True,True,False,True,False,True,True,True,True,False,False,True,False,False,True,True,False,True,False,False,False,True,False,False,False,True,True,True,False,False,True,False,True,False,True,False,True,False,True,False,False,False,True,False,False,False,False,False,True,False,False,True,True,True,True,False,False,False,True,True,False,True,True,False,True,False,False,False,False,True,True,True,True,False,True,True,True,True,False,False,False,False,False,True,True,True,True,False,False,True,True,False,False,False,False,True,False,False,True,False,False,False,True,True,True,True,True,False,False,True,False,False,True,False,False,True,False,True,True,False,False,False,False,False,False,True,True,False,True,True,False,True,True,False,True,False,True,False,False,True,True,True,False,False,True,False,True,True,True,False,True,False,False,False,False,False,True,False,False,False,True,True,False,False,True,True,True,False,True,True,True,False,False,True,False,False,True,False,True,True,False,True,True,True,True,True,True,False,True,False,False,False,True,False,True,False,True,False,False,False,True,True,False,False,True,True,False,False,True,True,True,True,False,False,False,True,False,False,True,True,False,False,False,True,False,True,True,False,True,True,False,False,False,True,False,True,False,False,False,True,False,True,False,False,False,True,False,False,True,True,True,True,True,True,False,False,False,True,False,True,False,False,False,False,True,False,False,True,True,False,False,True,True,True,True,True,False,True,True,True,True,True,False,False,False,False,False,True,True,False,False,True,True,False,True,False,True,True,False,False,True,False,True,True,True,False,False,True,False,False,False,True,True,False,True,True,False,False,False,True,False,True,False,True,True,False,False,False,True,False,False,True,False,True,False,False,False,False,True,False,False,False,True,True,False,False,True,True,False,True,False,True,False,True,False,False,True,False,False,False,False,True,True,True,False,True,True,False]], dtype = "bool")#candidate|513|(1, 1152)|const|bool
call_511 = relay.TupleGetItem(func_384_call(relay.reshape(const_512.astype('float64'), [5, 5, 8]), relay.reshape(const_513.astype('bool'), [1152,]), relay.reshape(const_512.astype('float64'), [5, 5, 8]), ), 5)
call_514 = relay.TupleGetItem(func_388_call(relay.reshape(const_512.astype('float64'), [5, 5, 8]), relay.reshape(const_513.astype('bool'), [1152,]), relay.reshape(const_512.astype('float64'), [5, 5, 8]), ), 5)
bop_515 = relay.greater(bop_506.astype('bool'), relay.reshape(var_504.astype('bool'), relay.shape_of(bop_506))) # shape=(12, 10, 3)
uop_519 = relay.erf(const_513.astype('float32')) # shape=(1, 1152)
bop_523 = relay.mod(uop_519.astype('float32'), relay.reshape(const_513.astype('float32'), relay.shape_of(uop_519))) # shape=(1, 1152)
uop_529 = relay.sigmoid(uop_519.astype('float64')) # shape=(1, 1152)
bop_531 = relay.bitwise_or(uop_529.astype('uint8'), relay.reshape(bop_523.astype('uint8'), relay.shape_of(uop_529))) # shape=(1, 1152)
bop_535 = relay.right_shift(bop_531.astype('uint16'), relay.reshape(uop_519.astype('uint16'), relay.shape_of(bop_531))) # shape=(1, 1152)
var_538 = relay.var("var_538", dtype = "uint8", shape = (10, 1152))#candidate|538|(10, 1152)|var|uint8
bop_539 = relay.floor_mod(bop_531.astype('float64'), var_538.astype('float64')) # shape=(10, 1152)
var_542 = relay.var("var_542", dtype = "float32", shape = (2, 1152))#candidate|542|(2, 1152)|var|float32
bop_543 = relay.bitwise_or(bop_523.astype('uint64'), var_542.astype('uint64')) # shape=(2, 1152)
uop_546 = relay.acosh(bop_543.astype('float32')) # shape=(2, 1152)
bop_549 = relay.bitwise_and(uop_519.astype('uint16'), bop_543.astype('uint16')) # shape=(2, 1152)
bop_552 = relay.equal(bop_523.astype('bool'), uop_546.astype('bool')) # shape=(2, 1152)
uop_563 = relay.log2(bop_552.astype('float64')) # shape=(2, 1152)
output = relay.Tuple([call_511,const_512,bop_515,bop_535,bop_539,bop_549,uop_563,])
output2 = relay.Tuple([call_514,const_512,bop_515,bop_535,bop_539,bop_549,uop_563,])
func_570 = relay.Function([var_504,var_538,var_542,], output)
mod['func_570'] = func_570
mod = relay.transform.InferType()(mod)
mutated_mod['func_570'] = func_570
mutated_mod = relay.transform.InferType()(mutated_mod)
func_570_call = mutated_mod.get_global_var('func_570')
var_572 = relay.var("var_572", dtype = "uint32", shape = (12, 10, 3))#candidate|572|(12, 10, 3)|var|uint32
var_573 = relay.var("var_573", dtype = "uint8", shape = (10, 1152))#candidate|573|(10, 1152)|var|uint8
var_574 = relay.var("var_574", dtype = "float32", shape = (2, 1152))#candidate|574|(2, 1152)|var|float32
call_571 = func_570_call(var_572,var_573,var_574,)
output = call_571
func_575 = relay.Function([var_572,var_573,var_574,], output)
mutated_mod['func_575'] = func_575
mutated_mod = relay.transform.InferType()(mutated_mod)
var_622 = relay.var("var_622", dtype = "float64", shape = (14, 11))#candidate|622|(14, 11)|var|float64
uop_623 = relay.sinh(var_622.astype('float64')) # shape=(14, 11)
bop_627 = relay.subtract(var_622.astype('int8'), relay.reshape(uop_623.astype('int8'), relay.shape_of(var_622))) # shape=(14, 11)
uop_631 = relay.asinh(bop_627.astype('float64')) # shape=(14, 11)
output = uop_631
output2 = uop_631
func_635 = relay.Function([var_622,], output)
mod['func_635'] = func_635
mod = relay.transform.InferType()(mod)
var_636 = relay.var("var_636", dtype = "float64", shape = (14, 11))#candidate|636|(14, 11)|var|float64
output = func_635(var_636)
func_637 = relay.Function([var_636], output)
mutated_mod['func_637'] = func_637
mutated_mod = relay.transform.InferType()(mutated_mod)
var_658 = relay.var("var_658", dtype = "float64", shape = (7, 2, 8))#candidate|658|(7, 2, 8)|var|float64
var_659 = relay.var("var_659", dtype = "float64", shape = (7, 2, 8))#candidate|659|(7, 2, 8)|var|float64
bop_660 = relay.equal(var_658.astype('bool'), relay.reshape(var_659.astype('bool'), relay.shape_of(var_658))) # shape=(7, 2, 8)
uop_664 = relay.asinh(var_659.astype('float32')) # shape=(7, 2, 8)
uop_666 = relay.erf(uop_664.astype('float32')) # shape=(7, 2, 8)
output = relay.Tuple([bop_660,uop_666,])
output2 = relay.Tuple([bop_660,uop_666,])
func_668 = relay.Function([var_658,var_659,], output)
mod['func_668'] = func_668
mod = relay.transform.InferType()(mod)
var_669 = relay.var("var_669", dtype = "float64", shape = (7, 2, 8))#candidate|669|(7, 2, 8)|var|float64
var_670 = relay.var("var_670", dtype = "float64", shape = (7, 2, 8))#candidate|670|(7, 2, 8)|var|float64
output = func_668(var_669,var_670,)
func_671 = relay.Function([var_669,var_670,], output)
mutated_mod['func_671'] = func_671
mutated_mod = relay.transform.InferType()(mutated_mod)
var_685 = relay.var("var_685", dtype = "float64", shape = (14, 12))#candidate|685|(14, 12)|var|float64
const_686 = relay.const([[-5.928217,-5.849648,5.220845,7.019419,1.020290,-4.329693,8.918952,-4.893226,7.394279,5.991775,9.458826,-8.815851],[0.614574,-4.957273,1.450545,3.847289,8.830493,-9.677087,-7.751280,0.373871,6.696877,-6.385237,-2.196909,-1.035251],[-1.460457,-6.880940,6.674251,5.851643,5.413143,-5.079486,0.097474,-4.182989,6.667349,-1.599220,5.395387,-4.906319],[4.742580,7.016370,2.734076,0.485169,-1.808878,2.945330,-3.251032,-1.452593,-5.320254,1.137517,-1.241713,-7.609946],[-6.988553,3.855979,5.522610,-6.733390,-8.481066,-4.371303,3.936250,6.624994,0.158255,4.204295,-9.134082,1.763492],[8.742138,-0.370405,9.117458,3.224704,3.530179,1.433420,-4.541609,3.878577,8.522706,-6.647024,6.114846,-9.126865],[-2.579288,6.503925,-3.632349,-3.650219,1.307347,0.994658,3.298751,-3.213388,-9.772263,4.881850,5.962054,5.757522],[-6.700655,6.520321,-0.081984,-4.463461,-0.482224,7.099285,2.613057,-5.103392,-2.950834,6.200922,-2.929269,7.668650],[8.618671,-0.382720,6.130094,9.917894,1.333853,-1.581234,2.619647,4.007456,-1.219578,-7.994439,-4.979352,-8.079280],[-8.901458,-5.905887,3.676274,0.532667,8.607458,0.751554,-6.034204,-1.684003,3.435260,2.624731,-2.860936,-2.978949],[5.153901,-3.753021,8.653276,-2.148813,-5.076021,2.582277,-4.887193,-5.394277,-8.127310,1.202964,-1.423974,-4.196441],[-2.510695,-0.194113,-5.172539,6.717480,1.827513,9.968200,-9.635940,7.136202,9.412285,-4.686838,4.878803,9.170602],[-8.781377,-9.855815,-3.411430,-4.711162,-1.865377,5.129538,-8.530696,-1.401848,-2.595456,-3.190028,-3.799193,7.932763],[-9.264514,1.906066,-4.367309,6.030932,8.681043,-5.948802,0.053258,-2.179190,8.123365,-7.243275,-5.734719,-6.455217]], dtype = "float64")#candidate|686|(14, 12)|const|float64
bop_687 = relay.maximum(var_685.astype('float64'), relay.reshape(const_686.astype('float64'), relay.shape_of(var_685))) # shape=(14, 12)
func_196_call = mod.get_global_var('func_196')
func_199_call = mutated_mod.get_global_var('func_199')
var_695 = relay.var("var_695", dtype = "bool", shape = (1152,))#candidate|695|(1152,)|var|bool
call_694 = relay.TupleGetItem(func_196_call(relay.reshape(var_695.astype('bool'), [16, 9, 8]), relay.reshape(var_695.astype('bool'), [16, 9, 8]), ), 1)
call_696 = relay.TupleGetItem(func_199_call(relay.reshape(var_695.astype('bool'), [16, 9, 8]), relay.reshape(var_695.astype('bool'), [16, 9, 8]), ), 1)
func_441_call = mod.get_global_var('func_441')
func_443_call = mutated_mod.get_global_var('func_443')
var_707 = relay.var("var_707", dtype = "uint32", shape = (720, 1))#candidate|707|(720, 1)|var|uint32
call_706 = func_441_call(relay.reshape(var_707.astype('uint32'), [12, 10, 6]))
call_708 = func_441_call(relay.reshape(var_707.astype('uint32'), [12, 10, 6]))
bop_717 = relay.mod(var_685.astype('float64'), relay.reshape(bop_687.astype('float64'), relay.shape_of(var_685))) # shape=(14, 12)
uop_721 = relay.sigmoid(bop_687.astype('float64')) # shape=(14, 12)
output = relay.Tuple([call_694,var_695,call_706,var_707,bop_717,uop_721,])
output2 = relay.Tuple([call_696,var_695,call_708,var_707,bop_717,uop_721,])
func_723 = relay.Function([var_685,var_695,var_707,], output)
mod['func_723'] = func_723
mod = relay.transform.InferType()(mod)
var_724 = relay.var("var_724", dtype = "float64", shape = (14, 12))#candidate|724|(14, 12)|var|float64
var_725 = relay.var("var_725", dtype = "bool", shape = (1152,))#candidate|725|(1152,)|var|bool
var_726 = relay.var("var_726", dtype = "uint32", shape = (720, 1))#candidate|726|(720, 1)|var|uint32
output = func_723(var_724,var_725,var_726,)
func_727 = relay.Function([var_724,var_725,var_726,], output)
mutated_mod['func_727'] = func_727
mutated_mod = relay.transform.InferType()(mutated_mod)
func_470_call = mod.get_global_var('func_470')
func_471_call = mutated_mod.get_global_var('func_471')
call_802 = func_470_call()
call_803 = func_470_call()
uop_808 = relay.sqrt(call_802.astype('float64')) # shape=(40, 1)
uop_810 = relay.sqrt(call_803.astype('float64')) # shape=(40, 1)
uop_811 = relay.acosh(uop_808.astype('float32')) # shape=(40, 1)
uop_813 = relay.acosh(uop_810.astype('float32')) # shape=(40, 1)
var_817 = relay.var("var_817", dtype = "float64", shape = (40, 15))#candidate|817|(40, 15)|var|float64
bop_818 = relay.power(uop_808.astype('float64'), var_817.astype('float64')) # shape=(40, 15)
bop_821 = relay.power(uop_810.astype('float64'), var_817.astype('float64')) # shape=(40, 15)
uop_822 = relay.rsqrt(uop_811.astype('float64')) # shape=(40, 1)
uop_824 = relay.rsqrt(uop_813.astype('float64')) # shape=(40, 1)
bop_825 = relay.right_shift(uop_822.astype('int32'), relay.reshape(uop_811.astype('int32'), relay.shape_of(uop_822))) # shape=(40, 1)
bop_828 = relay.right_shift(uop_824.astype('int32'), relay.reshape(uop_813.astype('int32'), relay.shape_of(uop_824))) # shape=(40, 1)
var_829 = relay.var("var_829", dtype = "float64", shape = (40, 15))#candidate|829|(40, 15)|var|float64
bop_830 = relay.bitwise_or(uop_822.astype('int16'), var_829.astype('int16')) # shape=(40, 15)
bop_833 = relay.bitwise_or(uop_824.astype('int16'), var_829.astype('int16')) # shape=(40, 15)
output = relay.Tuple([bop_818,bop_825,bop_830,])
output2 = relay.Tuple([bop_821,bop_828,bop_833,])
func_836 = relay.Function([var_817,var_829,], output)
mod['func_836'] = func_836
mod = relay.transform.InferType()(mod)
mutated_mod['func_836'] = func_836
mutated_mod = relay.transform.InferType()(mutated_mod)
func_836_call = mutated_mod.get_global_var('func_836')
var_838 = relay.var("var_838", dtype = "float64", shape = (40, 15))#candidate|838|(40, 15)|var|float64
var_839 = relay.var("var_839", dtype = "float64", shape = (40, 15))#candidate|839|(40, 15)|var|float64
call_837 = func_836_call(var_838,var_839,)
output = call_837
func_840 = relay.Function([var_838,var_839,], output)
mutated_mod['func_840'] = func_840
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_848 = relay.TupleGetItem(func_293_call(), 0)
call_849 = relay.TupleGetItem(func_294_call(), 0)
output = relay.Tuple([call_848,])
output2 = relay.Tuple([call_849,])
func_865 = relay.Function([], output)
mod['func_865'] = func_865
mod = relay.transform.InferType()(mod)
output = func_865()
func_866 = relay.Function([], output)
mutated_mod['func_866'] = func_866
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_906 = relay.TupleGetItem(func_865_call(), 0)
call_907 = relay.TupleGetItem(func_866_call(), 0)
func_570_call = mod.get_global_var('func_570')
func_575_call = mutated_mod.get_global_var('func_575')
var_935 = relay.var("var_935", dtype = "uint32", shape = (6, 60))#candidate|935|(6, 60)|var|uint32
var_936 = relay.var("var_936", dtype = "uint8", shape = (720, 16))#candidate|936|(720, 16)|var|uint8
var_937 = relay.var("var_937", dtype = "float32", shape = (2304,))#candidate|937|(2304,)|var|float32
call_934 = relay.TupleGetItem(func_570_call(relay.reshape(var_935.astype('uint32'), [12, 10, 3]), relay.reshape(var_936.astype('uint8'), [10, 1152]), relay.reshape(var_937.astype('float32'), [2, 1152]), ), 6)
call_938 = relay.TupleGetItem(func_575_call(relay.reshape(var_935.astype('uint32'), [12, 10, 3]), relay.reshape(var_936.astype('uint8'), [10, 1152]), relay.reshape(var_937.astype('float32'), [2, 1152]), ), 6)
func_723_call = mod.get_global_var('func_723')
func_727_call = mutated_mod.get_global_var('func_727')
var_947 = relay.var("var_947", dtype = "float64", shape = (168,))#candidate|947|(168,)|var|float64
const_948 = relay.const([False,True,True,False,True,False,True,False,False,True,True,True,False,True,False,True,True,True,True,False,False,True,False,True,True,True,False,True,False,False,True,False,True,False,True,True,False,False,True,False,True,False,True,False,True,False,True,False,True,False,False,True,False,False,False,True,True,True,False,False,True,True,False,True,False,True,False,False,False,True,True,True,True,False,True,False,False,True,False,True,True,True,True,True,True,True,True,True,False,True,True,False,True,True,True,True,True,True,True,True,False,False,False,True,True,False,True,True,True,True,True,False,False,False,False,False,True,True,True,False,False,False,False,True,True,True,False,False,False,False,True,False,False,False,True,True,False,False,True,True,True,False,True,False,False,True,False,True,False,True,True,True,True,True,False,False,False,True,True,False,True,False,False,True,False,True,True,False,True,False,True,True,False,True,True,False,False,True,True,True,False,True,False,True,False,False,True,True,True,False,True,False,True,False,False,True,True,True,False,True,True,True,False,True,False,True,True,True,False,False,False,False,True,False,True,False,False,False,True,False,True,False,False,False,False,False,True,False,True,True,True,False,True,True,True,True,False,False,False,False,True,False,False,False,False,True,False,True,True,True,True,False,False,True,False,False,False,False,False,True,True,True,False,False,False,True,False,False,False,False,False,True,True,False,True,True,True,False,False,True,True,True,True,True,True,False,False,True,False,False,False,True,False,False,False,False,False,False,True,False,False,True,True,True,False,False,True,True,False,True,False,False,True,False,True,False,False,True,True,False,False,True,True,False,False,True,False,False,True,True,False,False,True,True,True,True,True,False,False,False,False,True,False,True,True,True,True,True,True,False,True,True,True,False,True,True,False,True,True,True,True,True,True,False,False,True,True,False,True,True,False,True,False,False,False,True,True,True,False,False,True,True,True,True,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,True,True,False,True,False,False,True,True,True,True,False,False,False,True,True,True,True,False,False,True,False,False,True,True,True,False,False,True,False,False,True,True,True,True,False,True,True,False,True,True,True,True,True,True,False,False,True,False,False,True,True,False,True,False,True,False,False,True,False,False,True,True,False,True,True,False,False,False,False,True,True,True,False,False,True,True,True,False,True,True,True,False,False,False,True,True,False,True,True,False,False,False,False,False,True,True,False,False,True,True,True,True,False,False,True,True,True,False,True,False,False,False,False,False,False,True,True,True,True,False,True,True,True,True,True,False,False,True,False,False,False,False,True,True,False,False,False,True,True,False,True,True,False,True,True,True,True,False,False,False,True,True,True,False,True,False,False,True,True,True,True,False,True,False,True,True,True,True,False,False,True,True,True,False,False,False,True,True,False,True,True,True,False,False,True,True,False,False,True,False,True,False,False,False,True,True,True,False,False,True,False,False,False,True,False,False,True,True,False,True,False,True,True,False,False,False,False,False,False,True,False,True,True,False,True,False,True,False,False,False,True,False,False,True,True,False,False,False,True,False,False,True,True,True,True,True,True,True,False,False,False,False,True,False,True,True,False,False,True,True,False,True,True,False,False,True,False,False,True,True,False,True,False,False,False,False,True,False,True,True,False,False,False,True,False,True,False,False,True,False,True,False,True,True,False,True,False,True,False,True,False,False,True,True,False,True,True,False,True,False,True,True,False,True,True,False,False,True,True,False,True,False,False,False,True,False,False,True,True,False,True,False,False,False,True,True,True,False,True,True,False,False,False,True,False,True,True,False,True,False,False,False,True,False,True,False,True,True,False,False,True,True,False,True,True,True,False,False,True,True,True,False,False,False,True,False,True,False,False,False,True,True,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True,False,True,False,True,False,False,True,False,True,True,False,False,False,False,True,True,True,True,False,True,True,True,False,False,False,False,True,True,False,False,False,False,True,True,True,True,True,True,False,True,False,False,True,False,False,False,True,False,True,False,True,False,True,False,False,True,False,False,False,False,False,True,False,True,False,False,False,True,True,False,False,True,False,True,False,True,True,True,True,False,True,False,True,False,False,True,True,True,True,False,True,True,True,False,False,True,False,True,False,True,True,True,False,False,False,False,True,False,False,False,False,True,False,True,False,True,True,True,False,False,True,True,False,False,True,False,True,True,True,False,False,True,True,True,True,True,True,False,True,True,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,False,True,False,False,True,False,False,False,True,True,True,False,True,True,False,False,True,False,True,True,True,False,True,False,True,False,True,True,True,False,True,False,True,True,False,False,True,False,False,False,False,True,True,True,False,True,True,True,True,True,True,False,True,False,False,False,False,True,True,False,True,True,False,True,False,False,True,False,False,False,False,False,True,True,True,False,False,True,True,True,False,False,False,True,False,False,False,True,False,True,True,True,False,True,False,True,True,False,True,True,False,True,False,True,True,True,True,True,False,True,True,False,True,True,False,False,True,False,True,True,True,False,True,False,True,True,True,False,True,False,True,True,True,True,True,False,True,False,False,True,True,True,False,True,False,False,True,True,False,False,True,True,False,False,True,True,True,False,True,True,False,False,False,False,False,True,False,True,False,False,False,True,True,False,False,True,False,True,True,False,True,False,False,True,False,True,False,True,False,True,False,False,False,True,False,False,False,True,True,False,True], dtype = "bool")#candidate|948|(1152,)|const|bool
var_949 = relay.var("var_949", dtype = "uint32", shape = (720,))#candidate|949|(720,)|var|uint32
call_946 = relay.TupleGetItem(func_723_call(relay.reshape(var_947.astype('float64'), [14, 12]), relay.reshape(const_948.astype('bool'), [1152,]), relay.reshape(var_949.astype('uint32'), [720, 1]), ), 1)
call_950 = relay.TupleGetItem(func_727_call(relay.reshape(var_947.astype('float64'), [14, 12]), relay.reshape(const_948.astype('bool'), [1152,]), relay.reshape(var_949.astype('uint32'), [720, 1]), ), 1)
output = relay.Tuple([call_906,call_934,var_935,var_936,var_937,call_946,var_947,const_948,var_949,])
output2 = relay.Tuple([call_907,call_938,var_935,var_936,var_937,call_950,var_947,const_948,var_949,])
func_963 = relay.Function([var_935,var_936,var_937,var_947,var_949,], output)
mod['func_963'] = func_963
mod = relay.transform.InferType()(mod)
var_964 = relay.var("var_964", dtype = "uint32", shape = (6, 60))#candidate|964|(6, 60)|var|uint32
var_965 = relay.var("var_965", dtype = "uint8", shape = (720, 16))#candidate|965|(720, 16)|var|uint8
var_966 = relay.var("var_966", dtype = "float32", shape = (2304,))#candidate|966|(2304,)|var|float32
var_967 = relay.var("var_967", dtype = "float64", shape = (168,))#candidate|967|(168,)|var|float64
var_968 = relay.var("var_968", dtype = "uint32", shape = (720,))#candidate|968|(720,)|var|uint32
output = func_963(var_964,var_965,var_966,var_967,var_968,)
func_969 = relay.Function([var_964,var_965,var_966,var_967,var_968,], output)
mutated_mod['func_969'] = func_969
mutated_mod = relay.transform.InferType()(mutated_mod)
func_470_call = mod.get_global_var('func_470')
func_471_call = mutated_mod.get_global_var('func_471')
call_1028 = func_470_call()
call_1029 = func_470_call()
func_635_call = mod.get_global_var('func_635')
func_637_call = mutated_mod.get_global_var('func_637')
var_1044 = relay.var("var_1044", dtype = "float64", shape = (154,))#candidate|1044|(154,)|var|float64
call_1043 = func_635_call(relay.reshape(var_1044.astype('float64'), [14, 11]))
call_1045 = func_635_call(relay.reshape(var_1044.astype('float64'), [14, 11]))
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_1047 = relay.TupleGetItem(func_293_call(), 2)
call_1048 = relay.TupleGetItem(func_294_call(), 2)
output = relay.Tuple([call_1028,call_1043,var_1044,call_1047,])
output2 = relay.Tuple([call_1029,call_1045,var_1044,call_1048,])
func_1049 = relay.Function([var_1044,], output)
mod['func_1049'] = func_1049
mod = relay.transform.InferType()(mod)
var_1050 = relay.var("var_1050", dtype = "float64", shape = (154,))#candidate|1050|(154,)|var|float64
output = func_1049(var_1050)
func_1051 = relay.Function([var_1050], output)
mutated_mod['func_1051'] = func_1051
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1053 = relay.var("var_1053", dtype = "uint8", shape = (3,))#candidate|1053|(3,)|var|uint8
var_1054 = relay.var("var_1054", dtype = "uint8", shape = (3,))#candidate|1054|(3,)|var|uint8
bop_1055 = relay.not_equal(var_1053.astype('bool'), relay.reshape(var_1054.astype('bool'), relay.shape_of(var_1053))) # shape=(3,)
var_1067 = relay.var("var_1067", dtype = "bool", shape = (3,))#candidate|1067|(3,)|var|bool
bop_1068 = relay.minimum(bop_1055.astype('float32'), relay.reshape(var_1067.astype('float32'), relay.shape_of(bop_1055))) # shape=(3,)
bop_1073 = relay.add(bop_1068.astype('uint32'), relay.reshape(var_1054.astype('uint32'), relay.shape_of(bop_1068))) # shape=(3,)
uop_1081 = relay.atanh(bop_1073.astype('float32')) # shape=(3,)
func_245_call = mod.get_global_var('func_245')
func_247_call = mutated_mod.get_global_var('func_247')
var_1084 = relay.var("var_1084", dtype = "float64", shape = (40,))#candidate|1084|(40,)|var|float64
call_1083 = relay.TupleGetItem(func_245_call(relay.reshape(var_1084.astype('float64'), [5, 1, 8])), 0)
call_1085 = relay.TupleGetItem(func_247_call(relay.reshape(var_1084.astype('float64'), [5, 1, 8])), 0)
uop_1088 = relay.asinh(var_1053.astype('float64')) # shape=(3,)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_1094 = relay.TupleGetItem(func_293_call(), 0)
call_1095 = relay.TupleGetItem(func_294_call(), 0)
bop_1098 = relay.add(uop_1088.astype('int64'), relay.reshape(var_1067.astype('int64'), relay.shape_of(uop_1088))) # shape=(3,)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_1102 = relay.TupleGetItem(func_293_call(), 2)
call_1103 = relay.TupleGetItem(func_294_call(), 2)
output = relay.Tuple([uop_1081,call_1083,var_1084,call_1094,bop_1098,call_1102,])
output2 = relay.Tuple([uop_1081,call_1085,var_1084,call_1095,bop_1098,call_1103,])
func_1105 = relay.Function([var_1053,var_1054,var_1067,var_1084,], output)
mod['func_1105'] = func_1105
mod = relay.transform.InferType()(mod)
var_1106 = relay.var("var_1106", dtype = "uint8", shape = (3,))#candidate|1106|(3,)|var|uint8
var_1107 = relay.var("var_1107", dtype = "uint8", shape = (3,))#candidate|1107|(3,)|var|uint8
var_1108 = relay.var("var_1108", dtype = "bool", shape = (3,))#candidate|1108|(3,)|var|bool
var_1109 = relay.var("var_1109", dtype = "float64", shape = (40,))#candidate|1109|(40,)|var|float64
output = func_1105(var_1106,var_1107,var_1108,var_1109,)
func_1110 = relay.Function([var_1106,var_1107,var_1108,var_1109,], output)
mutated_mod['func_1110'] = func_1110
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1133 = relay.const([[[6.842344,-3.722217,-3.676844,9.572285,4.994423,-5.780790,6.176977],[9.636196,-0.706357,5.251584,6.749622,-5.400887,8.984257,-1.725618],[-6.826535,-6.198083,-2.159163,-1.954261,-4.328717,9.400061,-8.290043],[9.478620,-8.295412,5.019393,9.803980,-7.250923,8.930003,-1.496884],[-0.381052,6.706713,6.250002,4.707671,-4.133700,-4.429848,0.328272]],[[6.537619,7.267793,-6.749506,9.507062,2.192504,-6.732872,1.478653],[-8.592360,-8.542150,-3.200329,9.774437,7.133672,8.327487,0.969427],[3.600544,6.139047,-7.068342,4.166209,3.674361,8.056718,8.316268],[-0.834725,9.207967,8.164068,7.185631,1.457082,9.495357,7.726195],[2.360509,-5.504074,-5.259432,9.876664,-3.663046,1.633457,0.162072]],[[-9.728722,-4.541547,2.269738,-1.401521,-9.535750,-3.856954,-0.484770],[-9.023222,-5.181932,-4.357782,-2.815845,-1.495560,-5.160381,7.986878],[-1.011387,-6.727864,2.574413,4.922349,-8.955866,6.256098,-4.735665],[-3.544771,3.546372,-5.739410,-1.090779,-1.645173,4.370508,-7.817726],[-2.305517,9.994341,-2.583139,3.995499,7.285981,-2.379461,-8.652522]],[[-4.598833,1.841592,-0.372855,9.518164,-5.436389,3.957900,2.831903],[-0.266643,-0.731271,5.773603,6.114226,1.725716,3.354616,-1.338078],[-9.110598,-2.837348,7.259679,-8.652229,-8.506790,-6.353255,8.278754],[-7.962409,3.785557,5.479425,7.986406,6.964159,-6.013840,2.513042],[1.010737,-7.200966,3.600147,-6.846062,-1.324925,-7.927975,6.840193]]], dtype = "float32")#candidate|1133|(4, 5, 7)|const|float32
uop_1134 = relay.atan(const_1133.astype('float32')) # shape=(4, 5, 7)
bop_1137 = relay.divide(uop_1134.astype('float32'), relay.reshape(const_1133.astype('float32'), relay.shape_of(uop_1134))) # shape=(4, 5, 7)
bop_1141 = relay.logical_and(const_1133.astype('bool'), relay.reshape(uop_1134.astype('bool'), relay.shape_of(const_1133))) # shape=(4, 5, 7)
var_1147 = relay.var("var_1147", dtype = "float32", shape = (4, 5, 7))#candidate|1147|(4, 5, 7)|var|float32
bop_1148 = relay.mod(uop_1134.astype('float64'), relay.reshape(var_1147.astype('float64'), relay.shape_of(uop_1134))) # shape=(4, 5, 7)
uop_1156 = relay.asinh(uop_1134.astype('float64')) # shape=(4, 5, 7)
func_635_call = mod.get_global_var('func_635')
func_637_call = mutated_mod.get_global_var('func_637')
var_1159 = relay.var("var_1159", dtype = "float64", shape = (154,))#candidate|1159|(154,)|var|float64
call_1158 = func_635_call(relay.reshape(var_1159.astype('float64'), [14, 11]))
call_1160 = func_635_call(relay.reshape(var_1159.astype('float64'), [14, 11]))
var_1165 = relay.var("var_1165", dtype = "float32", shape = (4, 5, 7))#candidate|1165|(4, 5, 7)|var|float32
bop_1166 = relay.logical_or(uop_1134.astype('bool'), relay.reshape(var_1165.astype('bool'), relay.shape_of(uop_1134))) # shape=(4, 5, 7)
bop_1171 = relay.right_shift(bop_1148.astype('uint32'), relay.reshape(uop_1156.astype('uint32'), relay.shape_of(bop_1148))) # shape=(4, 5, 7)
output = relay.Tuple([bop_1137,bop_1141,call_1158,var_1159,bop_1166,bop_1171,])
output2 = relay.Tuple([bop_1137,bop_1141,call_1160,var_1159,bop_1166,bop_1171,])
func_1177 = relay.Function([var_1147,var_1159,var_1165,], output)
mod['func_1177'] = func_1177
mod = relay.transform.InferType()(mod)
mutated_mod['func_1177'] = func_1177
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1177_call = mutated_mod.get_global_var('func_1177')
var_1179 = relay.var("var_1179", dtype = "float32", shape = (4, 5, 7))#candidate|1179|(4, 5, 7)|var|float32
var_1180 = relay.var("var_1180", dtype = "float64", shape = (154,))#candidate|1180|(154,)|var|float64
var_1181 = relay.var("var_1181", dtype = "float32", shape = (4, 5, 7))#candidate|1181|(4, 5, 7)|var|float32
call_1178 = func_1177_call(var_1179,var_1180,var_1181,)
output = call_1178
func_1182 = relay.Function([var_1179,var_1180,var_1181,], output)
mutated_mod['func_1182'] = func_1182
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_1184 = relay.TupleGetItem(func_293_call(), 2)
call_1185 = relay.TupleGetItem(func_294_call(), 2)
output = call_1184
output2 = call_1185
func_1196 = relay.Function([], output)
mod['func_1196'] = func_1196
mod = relay.transform.InferType()(mod)
output = func_1196()
func_1197 = relay.Function([], output)
mutated_mod['func_1197'] = func_1197
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1234 = relay.var("var_1234", dtype = "uint64", shape = (14, 8))#candidate|1234|(14, 8)|var|uint64
const_1235 = relay.const([[4,9,-7,4,8,-10,4,5],[-7,-1,-4,-5,1,-10,4,2],[-10,-3,-10,3,-8,2,-10,-9],[9,-5,3,8,5,10,-8,1],[-5,-6,-3,4,1,-5,-4,-9],[4,-6,6,-2,-3,-2,7,10],[-8,-7,-7,-2,7,-3,-2,2],[4,5,-5,1,-9,9,-6,6],[-9,-3,-3,7,4,9,-2,-2],[-10,-6,-1,-2,-2,6,4,-10],[-4,9,1,5,8,9,-6,6],[-7,-2,-3,-9,-7,-3,-7,-4],[-7,8,1,-3,-9,10,-5,-3],[1,9,10,-9,10,-8,4,-8]], dtype = "uint64")#candidate|1235|(14, 8)|const|uint64
bop_1236 = relay.logical_xor(var_1234.astype('uint64'), relay.reshape(const_1235.astype('uint64'), relay.shape_of(var_1234))) # shape=(14, 8)
output = relay.Tuple([bop_1236,])
output2 = relay.Tuple([bop_1236,])
func_1241 = relay.Function([var_1234,], output)
mod['func_1241'] = func_1241
mod = relay.transform.InferType()(mod)
mutated_mod['func_1241'] = func_1241
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1242 = relay.var("var_1242", dtype = "uint64", shape = (14, 8))#candidate|1242|(14, 8)|var|uint64
func_1241_call = mutated_mod.get_global_var('func_1241')
call_1243 = func_1241_call(var_1242)
output = call_1243
func_1244 = relay.Function([var_1242], output)
mutated_mod['func_1244'] = func_1244
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1196_call = mod.get_global_var('func_1196')
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1250 = func_1196_call()
call_1251 = func_1196_call()
func_384_call = mod.get_global_var('func_384')
func_388_call = mutated_mod.get_global_var('func_388')
var_1257 = relay.var("var_1257", dtype = "float64", shape = (200,))#candidate|1257|(200,)|var|float64
const_1258 = relay.const([True,True,False,False,False,True,False,False,True,True,False,True,True,False,False,False,False,False,True,False,False,True,False,True,True,True,True,False,False,True,True,True,True,True,False,True,True,False,True,False,True,True,True,False,False,True,True,False,True,False,True,False,True,True,True,False,False,False,True,True,False,False,False,True,True,False,False,False,True,True,False,False,False,False,True,False,True,False,True,True,False,False,True,True,True,False,True,False,False,False,True,False,False,True,True,False,False,True,False,True,False,True,True,True,True,False,True,False,False,False,True,False,False,False,True,False,False,True,False,True,True,True,True,False,False,True,False,False,False,False,True,True,True,True,False,False,True,True,True,False,False,False,False,False,True,False,False,True,True,False,False,False,True,False,False,False,True,False,False,True,True,False,False,False,True,False,True,False,False,False,False,False,False,True,False,True,True,True,True,False,True,True,False,True,True,True,True,False,True,True,False,False,True,False,True,False,True,False,True,True,False,True,True,False,False,True,False,True,False,True,True,False,False,False,True,False,True,False,False,False,True,False,True,True,True,False,True,False,False,False,True,False,False,False,False,False,True,False,False,False,True,True,False,False,True,True,False,False,True,False,True,False,True,False,True,False,False,True,False,False,True,True,False,True,True,False,False,False,True,False,False,False,True,False,False,True,True,False,True,False,True,False,True,False,False,False,False,False,True,False,False,True,True,False,False,False,True,False,False,False,False,True,False,False,True,True,True,True,True,False,True,False,True,False,False,True,True,False,False,False,False,False,True,True,True,False,True,False,False,False,False,True,True,False,True,False,True,False,True,True,False,True,True,True,True,False,True,False,False,True,True,True,True,False,False,False,False,False,False,True,False,False,False,True,True,True,True,False,True,True,True,True,False,True,True,True,True,False,False,True,True,True,False,False,True,True,True,False,True,True,True,True,False,True,False,True,False,True,True,True,False,True,False,False,False,True,False,True,True,False,False,False,True,True,True,False,False,False,False,True,True,True,False,True,False,False,False,False,True,False,True,False,True,False,False,True,True,False,False,True,False,False,True,True,True,False,False,True,False,False,False,False,False,False,True,False,False,False,True,False,False,False,False,True,False,False,False,False,True,False,True,True,False,False,True,False,True,True,True,True,True,False,True,True,False,True,False,True,True,True,True,True,True,True,False,False,True,False,False,False,False,True,False,False,True,True,False,False,False,False,False,True,False,True,True,False,False,True,True,True,False,False,False,True,True,False,True,True,False,True,True,False,False,True,True,False,False,True,False,False,True,False,False,True,False,True,False,False,False,True,False,False,False,False,True,True,True,True,False,True,False,False,False,True,True,True,True,False,False,True,False,False,False,False,True,False,True,True,False,True,True,False,True,True,True,True,False,False,False,True,True,False,True,True,True,False,False,False,False,True,False,True,True,False,True,False,False,True,False,False,False,True,True,False,False,False,True,False,False,True,True,True,True,False,True,True,True,True,True,True,False,True,False,True,True,False,False,True,True,False,False,True,True,True,True,True,True,False,True,True,True,True,False,True,True,True,False,False,True,True,False,True,True,False,True,False,False,True,False,True,True,False,True,False,True,False,True,False,True,False,True,False,True,True,True,True,True,True,True,False,True,True,True,False,False,False,False,False,True,False,False,False,False,True,False,True,True,True,True,False,False,False,True,True,False,False,False,True,False,False,True,True,True,False,True,True,True,True,False,False,False,False,True,False,True,True,False,False,True,True,True,True,False,False,False,False,False,True,True,False,True,False,True,False,False,False,False,True,False,False,False,False,False,True,False,True,False,False,True,True,False,False,False,False,False,True,False,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,True,True,False,False,True,True,True,False,False,False,False,False,False,True,True,True,True,False,True,False,False,True,True,False,True,True,False,False,True,True,False,False,False,True,False,False,False,False,False,True,False,False,True,False,False,False,False,True,False,True,True,False,False,False,False,True,True,True,True,True,False,False,True,False,True,False,False,True,False,True,False,False,True,True,True,True,True,True,False,True,False,False,True,False,False,False,True,False,True,True,True,True,True,False,True,False,False,False,True,False,True,False,True,False,True,False,False,True,True,False,False,True,False,False,True,True,False,False,True,True,True,False,False,False,False,True,False,True,True,False,True,False,False,False,True,True,False,True,False,True,True,True,False,False,True,True,True,True,False,False,False,True,False,False,False,False,True,False,True,True,False,True,True,False,True,False,False,False,True,False,True,True,True,False,True,False,True,False,True,True,False,True,True,False,True,True,False,True,True,False,False,False,True,True,False,False,False,False,False,True,True,False,False,False,False,True,True,True,True,False,True,True,False,False,True,True,True,False,False,False,False,False,False,True,False,True,True,False,True,False,True,False,False,False,True,False,True,False,True,True,False,False,False,False,False,False,False,True,False,False,False,False,True,False,True,True,True,False,True,True,False,False,True,False,False,False,False,False,False,False,True,True,False,True,True,True,True,True,False,True,False,False,True,True,True,False,False,False,True,True,True,True,True,True,False,False,True,False,False,False,False,True,True,True,False,False,False,False,False,False,True,False,True,True,True,False,True,True,False,False,False,False,False,True,True,False,True,False,False,True,False,True,False,True,False,True,False,False,True,False,False,False,False,True,True,True,True], dtype = "bool")#candidate|1258|(1152,)|const|bool
call_1256 = relay.TupleGetItem(func_384_call(relay.reshape(var_1257.astype('float64'), [5, 5, 8]), relay.reshape(const_1258.astype('bool'), [1152,]), relay.reshape(var_1257.astype('float64'), [5, 5, 8]), ), 4)
call_1259 = relay.TupleGetItem(func_388_call(relay.reshape(var_1257.astype('float64'), [5, 5, 8]), relay.reshape(const_1258.astype('bool'), [1152,]), relay.reshape(var_1257.astype('float64'), [5, 5, 8]), ), 4)
output = relay.Tuple([call_1250,call_1256,var_1257,const_1258,])
output2 = relay.Tuple([call_1251,call_1259,var_1257,const_1258,])
func_1263 = relay.Function([var_1257,], output)
mod['func_1263'] = func_1263
mod = relay.transform.InferType()(mod)
mutated_mod['func_1263'] = func_1263
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1264 = relay.var("var_1264", dtype = "float64", shape = (200,))#candidate|1264|(200,)|var|float64
func_1263_call = mutated_mod.get_global_var('func_1263')
call_1265 = func_1263_call(var_1264)
output = call_1265
func_1266 = relay.Function([var_1264], output)
mutated_mod['func_1266'] = func_1266
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1355 = relay.const([[-3.446663,-6.887388,8.532708,-3.575971,4.732905,-1.004065,6.957649,-5.348218,-0.975404,-8.085677]], dtype = "float64")#candidate|1355|(1, 10)|const|float64
var_1356 = relay.var("var_1356", dtype = "float64", shape = (1, 10))#candidate|1356|(1, 10)|var|float64
bop_1357 = relay.mod(const_1355.astype('float64'), relay.reshape(var_1356.astype('float64'), relay.shape_of(const_1355))) # shape=(1, 10)
uop_1361 = relay.tan(const_1355.astype('float64')) # shape=(1, 10)
uop_1367 = relay.exp(uop_1361.astype('float32')) # shape=(1, 10)
uop_1370 = relay.acosh(const_1355.astype('float32')) # shape=(1, 10)
output = relay.Tuple([bop_1357,uop_1367,uop_1370,])
output2 = relay.Tuple([bop_1357,uop_1367,uop_1370,])
func_1384 = relay.Function([var_1356,], output)
mod['func_1384'] = func_1384
mod = relay.transform.InferType()(mod)
mutated_mod['func_1384'] = func_1384
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1385 = relay.var("var_1385", dtype = "float64", shape = (1, 10))#candidate|1385|(1, 10)|var|float64
func_1384_call = mutated_mod.get_global_var('func_1384')
call_1386 = func_1384_call(var_1385)
output = call_1386
func_1387 = relay.Function([var_1385], output)
mutated_mod['func_1387'] = func_1387
mutated_mod = relay.transform.InferType()(mutated_mod)
func_392_call = mod.get_global_var('func_392')
func_393_call = mutated_mod.get_global_var('func_393')
call_1418 = relay.TupleGetItem(func_392_call(), 0)
call_1419 = relay.TupleGetItem(func_393_call(), 0)
func_1105_call = mod.get_global_var('func_1105')
func_1110_call = mutated_mod.get_global_var('func_1110')
var_1421 = relay.var("var_1421", dtype = "uint8", shape = (1, 3))#candidate|1421|(1, 3)|var|uint8
var_1422 = relay.var("var_1422", dtype = "float64", shape = (40,))#candidate|1422|(40,)|var|float64
call_1420 = relay.TupleGetItem(func_1105_call(relay.reshape(var_1421.astype('uint8'), [3,]), relay.reshape(var_1421.astype('uint8'), [3,]), relay.reshape(var_1421.astype('bool'), [3,]), relay.reshape(var_1422.astype('float64'), [40,]), ), 5)
call_1423 = relay.TupleGetItem(func_1110_call(relay.reshape(var_1421.astype('uint8'), [3,]), relay.reshape(var_1421.astype('uint8'), [3,]), relay.reshape(var_1421.astype('bool'), [3,]), relay.reshape(var_1422.astype('float64'), [40,]), ), 5)
output = relay.Tuple([call_1418,call_1420,var_1421,var_1422,])
output2 = relay.Tuple([call_1419,call_1423,var_1421,var_1422,])
func_1424 = relay.Function([var_1421,var_1422,], output)
mod['func_1424'] = func_1424
mod = relay.transform.InferType()(mod)
var_1425 = relay.var("var_1425", dtype = "uint8", shape = (1, 3))#candidate|1425|(1, 3)|var|uint8
var_1426 = relay.var("var_1426", dtype = "float64", shape = (40,))#candidate|1426|(40,)|var|float64
output = func_1424(var_1425,var_1426,)
func_1427 = relay.Function([var_1425,var_1426,], output)
mutated_mod['func_1427'] = func_1427
mutated_mod = relay.transform.InferType()(mutated_mod)
func_470_call = mod.get_global_var('func_470')
func_471_call = mutated_mod.get_global_var('func_471')
call_1429 = func_470_call()
call_1430 = func_470_call()
var_1441 = relay.var("var_1441", dtype = "float64", shape = (40, 5))#candidate|1441|(40, 5)|var|float64
bop_1442 = relay.minimum(call_1429.astype('float64'), var_1441.astype('float64')) # shape=(40, 5)
bop_1445 = relay.minimum(call_1430.astype('float64'), var_1441.astype('float64')) # shape=(40, 5)
func_196_call = mod.get_global_var('func_196')
func_199_call = mutated_mod.get_global_var('func_199')
const_1452 = relay.const([True,True,False,False,False,True,True,False,True,False,True,False,True,False,False,True,False,False,False,False,True,False,False,True,True,True,True,True,True,False,False,False,True,False,False,True,True,True,True,False,True,True,True,False,True,True,True,True,True,True,False,False,True,False,False,True,False,True,False,True,True,False,True,False,True,True,True,False,False,True,True,True,False,False,True,True,True,False,True,True,True,True,True,False,True,True,True,True,True,False,False,False,False,True,True,True,False,False,False,False,True,True,True,True,True,False,True,False,False,False,True,True,True,True,False,False,False,True,False,True,True,False,False,False,False,True,True,False,True,True,True,True,False,False,False,True,True,True,False,True,True,True,True,False,False,True,True,True,False,True,False,False,True,False,False,True,True,True,False,True,True,True,False,True,True,False,True,False,False,True,False,True,False,True,False,True,False,True,False,False,False,False,False,True,True,False,True,False,False,True,True,True,False,True,False,True,False,True,False,False,False,False,True,True,True,True,False,True,False,False,True,False,False,False,False,True,False,True,False,False,False,True,True,False,True,False,True,True,True,True,True,False,True,False,True,False,False,True,True,False,True,True,True,True,True,True,True,True,False,True,False,False,False,True,False,True,True,True,False,True,True,True,True,False,False,False,True,False,True,False,False,False,True,True,True,False,False,False,False,False,True,False,False,False,True,False,True,True,False,False,False,True,True,True,True,True,True,False,True,False,True,False,False,False,True,True,True,False,False,True,False,True,True,True,True,False,True,True,True,True,True,True,False,False,False,False,False,True,False,True,True,True,True,True,False,True,True,True,True,True,False,True,False,True,False,False,True,False,True,True,True,False,False,False,False,True,False,False,False,False,False,False,False,True,True,False,False,False,True,True,True,True,True,False,False,True,False,True,True,True,False,True,True,True,True,True,False,True,False,False,True,False,False,True,True,False,True,False,False,False,True,False,True,False,False,True,False,False,True,True,True,True,False,False,False,True,True,False,False,True,True,False,True,True,True,True,True,True,True,False,True,False,False,True,True,False,False,True,False,True,False,True,True,True,True,True,False,False,False,True,False,True,True,True,False,False,False,False,True,True,False,True,False,False,False,True,False,False,False,False,False,False,True,True,True,False,False,False,False,False,True,False,True,False,False,False,False,False,False,False,False,False,True,False,False,True,True,False,True,True,False,True,True,True,False,True,True,True,True,True,True,False,False,True,False,False,True,False,False,True,False,False,True,True,False,True,False,True,True,False,False,True,True,False,False,False,True,True,True,False,False,False,False,False,True,False,False,False,False,True,True,True,True,False,False,False,True,True,True,False,True,False,True,False,False,True,False,True,False,True,True,True,False,True,True,True,True,False,False,True,True,True,False,False,False,True,False,True,False,False,False,True,False,True,True,False,True,False,True,True,True,False,False,True,True,False,True,False,False,True,True,True,False,False,True,False,True,False,False,True,False,False,False,False,True,True,False,False,True,True,False,True,True,False,False,False,False,True,False,False,False,True,True,True,True,True,True,False,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,True,False,True,False,False,False,False,False,True,True,True,False,False,False,True,False,True,True,True,False,False,True,False,True,False,False,False,True,False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,False,True,False,False,True,True,True,True,False,True,False,True,False,False,False,True,True,False,False,False,True,False,False,False,True,True,True,True,False,False,False,True,True,True,True,True,False,True,False,False,False,True,True,False,True,True,True,False,True,False,True,False,False,True,True,True,False,False,False,True,False,True,False,True,False,True,False,False,False,True,False,False,False,True,False,True,False,False,True,False,False,False,False,False,True,True,True,False,True,True,True,True,False,True,True,False,False,True,False,True,False,False,True,False,True,False,True,False,False,True,False,False,True,False,False,True,False,True,True,True,False,False,True,True,True,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,True,False,True,False,False,True,False,True,False,True,True,False,True,False,False,True,False,False,True,True,False,True,True,False,True,False,False,True,False,True,True,False,True,False,False,True,False,True,False,False,False,True,False,True,False,False,True,False,True,False,True,True,False,False,False,True,False,False,True,True,True,False,True,True,True,True,False,True,False,True,True,True,False,False,False,False,False,True,True,True,False,False,True,False,True,True,True,True,True,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,False,True,False,True,False,True,False,False,False,True,False,True,False,False,False,False,False,True,True,True,False,True,True,True,True,False,True,True,True,True,True,True,False,True,False,True,True,True,True,True,False,False,False,True,False,False,True,False,True,False,False,False,True,True,True,False,False,False,True,True,False,False,False,False,False,True,True,True,False,True,False,False,True,False,True,True,False,False,True,False,False,True,False,True,True,True,False,True,False,True,False,False,False,False,False,False,True,False,True,True,True,True,False,False,True,True,True,True,False,False,True,False,False,True,False,True,True,False,False,True,True,False,True,True,False,True,True,False,True,False,True,False,True,True,False,False,True,True,False,True,False,True,True,False,True,True,True,False,True,True,True,True,True,False,False,True,True,False,True,True,False,True,True,False,False,False,False,True,True,False,False,True,False,False,True,True], dtype = "bool")#candidate|1452|(1152,)|const|bool
call_1451 = relay.TupleGetItem(func_196_call(relay.reshape(const_1452.astype('bool'), [16, 9, 8]), relay.reshape(const_1452.astype('bool'), [16, 9, 8]), ), 1)
call_1453 = relay.TupleGetItem(func_199_call(relay.reshape(const_1452.astype('bool'), [16, 9, 8]), relay.reshape(const_1452.astype('bool'), [16, 9, 8]), ), 1)
output = relay.Tuple([bop_1442,call_1451,const_1452,])
output2 = relay.Tuple([bop_1445,call_1453,const_1452,])
func_1455 = relay.Function([var_1441,], output)
mod['func_1455'] = func_1455
mod = relay.transform.InferType()(mod)
mutated_mod['func_1455'] = func_1455
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1456 = relay.var("var_1456", dtype = "float64", shape = (40, 5))#candidate|1456|(40, 5)|var|float64
func_1455_call = mutated_mod.get_global_var('func_1455')
call_1457 = func_1455_call(var_1456)
output = call_1457
func_1458 = relay.Function([var_1456], output)
mutated_mod['func_1458'] = func_1458
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1475 = relay.const([[[6.935986,-3.182577,3.247856,5.384817,6.367264,4.526111,-3.878235,3.160720,0.011009,-8.434493,-3.341460,-6.032168,9.771929,8.213178,-1.955023,5.219118],[-3.149091,4.346741,8.742425,-7.008457,-8.363624,4.977417,-1.730601,3.746642,-8.650699,2.643236,-1.637412,-0.942882,5.997979,-1.547099,7.688006,1.628652],[0.370515,0.217092,-7.603398,4.280331,-6.451724,8.967140,5.077969,-0.918201,-1.131976,2.850452,3.915381,-5.025881,2.791521,7.426511,-9.439944,-4.784953],[0.051572,7.505763,4.716246,-2.113968,2.981313,0.480530,-8.240265,-4.087997,9.939270,9.509558,-2.683358,1.734261,-5.212369,-1.773886,-6.058194,9.505842],[-1.223185,6.892450,1.708951,6.335584,7.055305,-7.889101,3.701350,-7.930077,-1.809048,6.941759,9.934521,-9.622898,6.495495,-9.846008,-5.867920,4.197699],[-7.737191,-1.891343,-5.809299,0.028878,-0.929692,1.020699,-1.860795,3.937894,-1.951658,0.866115,9.727972,0.855756,-7.370822,-4.657808,0.311287,-5.976950],[-5.415026,-3.672518,-7.777428,-4.678643,8.072530,-6.160700,0.558603,-7.801474,8.955868,4.605665,0.780132,1.458267,7.503638,-3.028005,-8.922604,-5.872472],[-4.243180,3.384468,-8.211139,4.826548,-0.893587,-8.731232,8.495085,0.392548,-8.133165,-1.189746,6.952635,-8.567523,5.678407,-5.774848,2.723879,7.014989],[7.688055,-6.057040,-4.425167,2.672930,-6.203989,4.848112,-0.474777,-2.866921,5.732851,-6.538111,-6.530400,7.402024,-8.789807,2.739899,0.850309,6.908783],[-2.157116,-9.671114,8.797298,4.792805,2.980270,-7.896804,1.537582,0.331759,3.458824,4.467776,-1.606073,1.313614,-1.824607,-1.989056,5.718634,-4.695053],[2.236927,0.584023,-0.115273,-7.890254,0.501480,9.166104,7.591485,5.878680,4.220466,2.734753,-4.835057,-1.642968,0.504065,-2.287812,-3.019296,-9.273041],[5.119401,-2.691518,-0.477682,-9.999963,2.751825,-3.191965,-1.229946,3.627655,7.510949,5.482257,-7.182351,-6.253009,-7.809317,-3.135567,-1.236620,-3.751608]],[[-9.989493,9.536049,-6.154444,-5.909978,-5.511375,-0.393325,8.457315,-8.725303,-2.002049,-2.127192,3.199330,2.996020,6.888536,-1.282540,-2.886973,1.112891],[9.626485,-7.402971,-4.519774,-2.023691,2.108766,3.468498,-2.360784,5.901942,4.664071,6.500246,1.374900,-7.140517,0.837225,-3.768786,-2.681466,-5.719272],[-3.649842,3.009690,-4.263526,8.042535,-5.997929,-5.105556,-9.926236,-1.097649,4.203374,-7.964316,-2.338583,0.979212,5.347716,9.776296,0.675610,-6.410919],[-5.633751,9.810887,7.473892,0.670500,6.305233,-2.432152,7.707787,-8.533986,5.484980,1.745408,7.136899,5.399343,-9.728406,-3.061559,-4.959535,-2.164935],[-3.139994,0.658404,-0.433200,-4.243540,-8.633404,-2.884871,5.450692,-1.227788,0.323379,-0.504279,-4.384516,-2.786649,-7.635653,2.584941,5.335676,-4.408551],[2.238856,2.415908,-5.022408,9.375902,-8.773429,-1.414570,-8.255190,-9.135174,9.102525,7.841405,-1.002841,-4.382680,-4.849405,-3.187177,4.146873,-1.968031],[-8.037917,9.340092,6.668973,-2.200747,-1.747301,1.368935,1.142038,-2.841371,-3.414388,8.963283,5.045846,7.531730,5.465101,2.673056,-5.035946,2.729501],[3.144691,7.091235,1.300633,2.206527,4.431488,7.786074,-1.493787,9.967354,1.444520,-2.016115,-6.923209,-3.414633,6.425569,9.248886,5.405183,-5.572433],[9.951700,7.520443,7.518684,-9.166205,6.920499,-6.487293,5.385292,-0.133035,-3.978963,-1.914775,9.999949,-8.655702,-1.243060,-0.676556,-3.759726,4.070427],[6.547961,-2.125264,4.116679,9.516501,-0.138991,-3.219897,0.986407,-1.362375,8.507476,-5.631357,-5.008720,3.743574,-5.591089,-7.647413,9.664464,-3.662118],[0.861752,0.274482,8.247789,1.256252,-2.124171,-5.940748,5.804783,1.803038,3.743883,4.723948,-0.408787,-9.003855,-1.065742,6.242033,-8.723941,-5.258442],[9.273216,-2.783502,-6.714049,-7.298876,-1.926268,-1.661694,-7.277312,-9.770051,-9.208657,8.847258,3.943317,4.672467,-7.212472,3.139894,1.212592,-3.260523]],[[6.527023,1.494304,0.481253,9.445895,-6.559897,3.610169,9.766536,-8.400099,6.074131,0.607675,-9.835254,-8.738885,0.339586,0.012687,-4.198592,9.350582],[4.245657,2.195343,6.399943,-7.902612,4.198859,-3.691656,5.409842,-3.423903,-3.906011,2.826169,-8.719725,8.293820,3.530601,-5.825578,5.319169,-6.744361],[7.008012,9.691421,-7.561632,-2.980464,9.491957,-8.987540,-0.606222,0.743557,-2.037426,2.173137,-3.804771,6.159532,2.454355,9.011852,8.648591,0.798824],[-7.782531,6.153803,-6.034622,-4.554938,-2.910008,0.233559,7.901085,4.441431,0.731983,-0.270101,-8.293103,-5.080180,-9.713821,-8.426707,1.108883,-1.219542],[-4.550724,-9.690935,5.990253,-2.499223,2.678517,-1.619593,-2.491051,2.588154,-7.757959,0.323019,6.394049,0.029740,-1.031198,3.188681,7.317234,7.893593],[-0.740114,0.197329,-9.432225,0.903383,1.511715,-2.810121,-9.461690,3.708856,7.050917,8.394895,-7.606051,-7.190049,-8.088268,-6.164547,8.048422,8.926129],[7.146019,-6.241925,-1.883205,-4.219698,9.249122,5.081703,-8.613144,9.457865,8.408743,-2.517534,8.995091,-2.126471,-5.464775,1.175533,-1.220093,4.738652],[1.649502,9.111227,-1.731523,4.168086,-1.142171,2.452467,-8.932335,2.560221,1.720459,8.706631,-9.477326,-0.340323,1.335402,1.326875,-2.458447,7.832427],[-5.088833,6.861450,-9.527996,-1.765512,5.223878,3.115714,-6.659184,5.431748,-6.217789,-0.770321,0.188405,-7.481174,8.840359,-1.190141,-6.774089,8.315317],[4.097920,9.311196,-6.008250,-0.573102,0.269666,7.236880,-1.216000,1.944982,1.592945,-6.260366,-4.794813,2.885402,-4.928096,4.733355,-9.906199,1.732588],[0.274571,6.429510,-5.155478,0.502112,-4.383779,-6.027898,-0.790156,4.285198,5.010877,-9.369126,-6.399011,1.742471,5.599341,-9.815833,-0.316994,9.828171],[-2.641589,4.364338,4.329420,6.925301,0.407991,-2.184046,0.066568,2.872429,1.920416,1.018562,-8.456937,-8.609395,1.244222,5.373450,-1.375813,-2.243733]],[[3.450510,-5.553728,-2.581454,-8.115050,3.785277,2.995770,8.937404,5.274688,-3.192061,-4.598479,-9.133950,5.568888,1.148372,6.433236,7.472524,1.333170],[7.588714,7.917999,9.005844,6.766682,4.341654,6.843006,0.807250,7.225616,3.593215,-4.016204,9.042617,-7.641670,-2.547566,-2.551574,3.584764,1.702804],[0.187982,-3.606834,-5.953581,7.713787,3.387735,-2.178397,-5.361533,-4.606677,-6.145610,5.106609,-9.787803,-8.477490,8.627025,3.912362,-1.830012,5.928774],[7.351642,-4.303251,8.327757,0.463918,7.423725,7.653249,-2.767255,0.210127,3.568763,-5.476025,-6.304076,-1.250520,3.123425,-9.616602,-3.596143,-2.724175],[-5.825136,-0.150644,-0.386935,-2.756562,4.452642,1.155017,2.281235,-0.603903,6.096369,7.811102,-7.363593,3.396569,4.123898,8.425275,7.369121,1.265360],[3.495549,6.608380,-9.054965,8.364863,-7.829503,6.722011,3.814831,-0.082973,-1.336629,6.374313,4.115989,-6.234407,-6.914791,1.810083,2.063119,3.651887],[-0.383278,0.796820,-8.740401,-8.075838,5.947826,0.515958,-6.288299,-3.175740,6.832298,0.005142,-7.068882,9.926411,2.735011,4.944896,7.896845,9.365472],[3.940834,-1.825737,5.027358,-1.957310,-9.305372,-9.982647,9.882438,7.255983,-4.832870,-5.218427,8.351750,8.790009,-0.135504,3.114157,-1.520260,4.108706],[4.081736,0.044603,-7.679852,-8.071992,5.547913,1.544414,9.734662,2.500705,9.350171,-8.127399,4.361450,-2.173748,-6.861145,-7.837065,-7.573593,1.394737],[3.424666,9.154533,-0.823719,5.539867,-4.343232,1.182170,9.645152,3.269534,-3.084010,3.399028,9.415545,5.897642,-1.173410,-9.141156,5.145788,-6.257556],[1.484123,-1.812398,7.095946,7.013351,6.249777,1.613068,-1.727912,7.420986,4.239121,-8.190985,-0.456891,-0.463474,5.980277,-0.930386,-5.500273,-0.981623],[-6.224613,-8.644083,-0.063880,8.801728,-5.033060,-2.138913,5.195422,1.870566,8.089520,9.094171,2.694988,-9.623451,9.083746,7.179355,0.032368,5.443336]],[[-2.060718,-4.596608,6.598349,-0.214467,-5.203490,-5.659488,8.304843,-2.444147,-1.141184,-6.439938,-9.104178,-9.679605,7.895048,1.350713,4.481602,-2.322722],[-7.212087,-9.152988,-8.867829,0.520550,-7.585627,-8.649849,0.136417,-6.188054,4.352903,4.302573,8.343632,1.175645,3.758555,9.425216,2.336276,3.517880],[-7.959977,0.978766,-5.828679,-1.327037,1.112659,-5.649544,1.835706,-2.369683,-0.475878,-1.604215,3.274502,-2.887527,9.997207,-2.735493,-3.126379,-2.719129],[4.619490,0.907396,6.677016,-5.630294,1.642003,0.706043,4.676613,2.425754,-9.124477,5.683991,0.927119,-1.014073,-3.326817,9.014970,-3.755447,0.169918],[5.391051,-1.722033,-2.820506,5.547567,0.075848,8.591022,-3.983586,-4.173741,-3.068563,3.220957,-4.859799,-2.672789,-2.097842,3.482393,-1.030941,-3.013798],[7.809040,-1.214984,-1.010617,6.417812,-6.604479,-6.015239,5.594286,7.489623,3.078971,0.791285,6.949772,-1.765469,6.668330,-0.448401,-4.496259,-1.185029],[6.545182,-4.677221,-7.556527,-1.758405,-7.810565,-6.785002,5.155144,5.184807,-6.322040,-4.598564,3.169066,5.388418,-2.837526,-0.044630,-7.549749,2.686822],[1.041050,0.652786,2.897443,-4.976045,5.030519,-9.799881,4.021794,4.480885,-2.145798,5.673711,-3.130285,0.578859,-1.830254,9.290381,-4.633489,3.800388],[-4.382077,4.444452,-1.431170,2.479913,-5.310094,-6.202427,-1.119272,6.279054,-6.905153,-6.359901,2.615717,3.733649,7.609533,0.927932,9.522833,9.097465],[5.732467,-8.272331,-8.558220,9.733027,9.268452,-0.650233,2.404479,-4.029690,-7.248906,-5.164409,-1.813404,-5.089581,1.145377,4.339812,-0.260011,-4.697578],[-0.660361,8.479320,0.537489,7.819623,-8.524517,-5.265799,-8.352198,1.994839,5.946133,-1.329993,-2.506774,8.508380,-6.617398,-8.193967,3.927066,4.598500],[4.016521,-0.214711,1.280110,6.757057,-4.232250,-5.766386,7.627650,2.370818,9.138444,-9.543050,-7.755878,9.448918,0.002066,9.050789,1.470137,-8.111203]]], dtype = "float32")#candidate|1475|(5, 12, 16)|const|float32
const_1476 = relay.const([[[8.272619,-0.835879,7.836139,-0.001100,7.301861,2.117204,-2.646087,9.069285,-5.402232,-8.738874,-7.780088,-9.244515,-0.873101,-4.083541,-4.350696,5.638888],[2.918675,7.742858,-8.595866,-1.321301,5.704770,2.087195,-9.428424,-9.111858,-9.640413,0.128849,-6.786791,1.204493,4.240497,4.578555,7.615397,4.227875],[7.193790,3.802202,-6.779682,7.095922,-4.831360,4.789020,9.202160,5.622082,-2.295621,0.263499,-4.089245,0.595578,-6.858748,-6.509229,3.427461,-8.837182],[-5.657820,3.637524,5.283610,-6.487708,8.204441,5.622918,7.011563,-6.229677,2.034677,1.296614,-7.925022,1.025050,7.056483,8.770131,6.373916,4.012956],[0.099507,-0.572514,-9.835519,-9.858504,-8.946497,-1.008716,1.477638,7.813523,-6.837974,-0.229542,9.116039,-7.257374,3.706174,3.895834,1.797551,-9.538058],[9.092998,-0.038969,0.360574,6.052297,-4.344945,-6.468480,1.873137,-9.302357,-0.223443,-7.472993,-3.927090,-2.597698,-1.288773,-8.140770,8.447103,4.311403],[4.260127,-6.180042,-5.635158,-1.598457,0.947105,-2.906053,-3.003097,6.243059,-3.419418,-6.231500,-7.659263,-1.356400,-2.148588,-5.955316,-5.962704,6.250644],[3.286229,-5.693708,6.750315,-6.640155,7.314237,9.739682,3.784068,-3.078533,0.878269,-4.017831,-8.611220,-4.638176,9.329992,-9.131943,-6.273682,0.124857],[6.418556,-6.574561,-2.024526,3.515955,0.021647,2.279942,5.220950,-2.597647,-3.587343,-0.901681,-1.765959,-2.449642,-2.538775,3.351628,-2.679271,-1.671364],[-7.789788,-0.299160,-0.050309,6.449604,6.943009,-8.976711,-5.754472,5.146367,-3.047154,2.603575,-7.961140,-5.975886,-0.377708,-8.458767,9.125097,-2.290633],[1.707128,-1.947114,7.270524,2.068318,8.095959,4.422825,1.046959,-3.104449,8.722811,-9.269303,2.067653,-7.514295,-1.485707,-1.070184,6.939343,8.035285],[-2.449455,-3.217950,2.611750,-5.242158,-4.521705,0.389047,-3.120067,6.518666,1.226378,5.930838,-1.421067,6.287496,-0.316342,-8.741701,-4.226747,-7.869626]],[[-6.936551,-8.945918,5.635333,-8.749167,9.429419,6.694268,7.283950,6.938481,-3.488537,-5.035664,-1.351616,0.258892,5.483542,-3.465318,3.657250,-3.301102],[-3.545211,-0.742563,5.427395,7.776847,-7.876785,-3.202686,5.073162,-4.553195,5.598439,-9.771455,-3.243433,-2.890251,-0.877324,-9.201517,-6.144517,6.849131],[0.519438,-7.406732,-6.490283,7.321408,-5.688741,-6.386322,4.842978,8.614818,3.633419,-5.855506,5.763179,-2.995666,-8.848234,0.950327,8.897019,7.952108],[-3.395370,-9.464677,9.745351,1.208814,9.281019,3.323362,0.585340,4.497967,-7.296110,-2.633238,7.993027,4.749970,9.137098,-1.901941,6.842701,-8.542529],[0.014124,-0.987874,-3.083531,-9.868614,-8.312923,8.457690,-0.948487,-6.767171,-8.253821,-6.879778,9.139226,6.425405,-5.505921,-8.799054,9.111474,-3.649668],[3.325650,9.298134,-0.724974,6.766158,2.935971,-4.966879,-7.545294,8.233919,-7.759540,-7.761209,2.600908,-0.443577,5.430673,-1.843777,0.362871,8.569276],[8.005186,0.123295,-9.249134,-9.175439,8.074193,2.425712,-6.037518,-2.974565,-0.074712,-3.904418,1.731046,1.367785,6.041775,4.983433,-8.213254,6.317610],[-2.913394,-2.383044,-7.127973,-8.025332,1.539792,0.212929,-1.037192,-8.365036,-7.659168,2.483444,-4.232033,-9.684390,-9.832303,-8.328788,-4.932960,-5.034737],[-3.783711,0.045302,-7.986136,-0.455079,0.656705,9.180801,-0.030931,-5.604276,-1.355569,-8.635543,-1.773046,4.509127,-5.266516,-7.662057,2.616947,5.823223],[0.498691,-1.635988,0.471841,5.547915,-5.675676,-8.022302,9.636293,0.069390,-5.634919,-4.258826,-8.614562,4.847562,8.250158,0.363311,7.810716,4.796015],[0.659621,-8.652781,-7.227112,7.139990,-5.819263,7.240594,-4.337774,3.840656,-0.202592,-9.230975,3.903476,-3.245485,2.420479,-5.300017,-1.080646,5.497473],[5.542458,-9.544990,6.709209,3.703398,-6.958797,4.539830,5.771784,4.761068,-9.904380,9.144329,-1.349096,4.876017,8.180123,-7.475507,0.967037,-8.466157]],[[8.517232,-6.335870,-9.512874,-0.792931,4.081187,-6.299014,-9.000561,-5.488390,2.765556,1.671637,4.163370,1.080445,-1.416019,2.085180,2.759730,-0.251269],[-6.746261,5.867071,-1.887682,5.056475,-4.305674,2.765166,-0.635645,6.685754,0.608266,7.044875,0.791525,4.408025,-0.291825,8.323129,-9.282308,-8.844219],[2.599916,-2.001629,4.839532,4.147475,7.926442,-2.140513,-6.239815,3.404730,-4.021492,4.248782,3.381695,7.817640,2.076566,2.017029,1.582379,-4.677847],[9.361116,-8.415985,0.264222,-5.574136,-9.058250,-7.406746,9.748016,-7.357898,-5.135435,6.872687,2.145761,0.361988,-6.197644,-9.428997,-7.562933,-9.909182],[4.121460,-0.512894,-5.262919,8.902308,6.745903,6.203139,5.441540,1.855591,2.079851,3.076615,4.442285,2.869115,2.773352,8.853162,3.055172,-7.671501],[3.195663,4.422605,8.495106,-1.525445,-4.907103,6.079119,-6.173410,5.939333,-9.943344,-5.509735,1.805211,3.421537,8.863235,-1.657097,-8.957333,-4.789368],[1.039025,7.760841,-4.468689,6.457561,6.267774,7.714087,0.011978,-6.939552,0.411279,-1.704804,-0.726755,-7.633249,-5.408890,-9.287643,9.064072,-4.818533],[-7.425596,7.407431,-9.517273,6.914469,-6.929993,6.901071,9.717792,0.997018,-4.967792,7.716829,7.665446,-1.759436,-8.501092,-0.861802,7.654464,-8.288892],[-5.809893,2.660422,7.957995,-3.954833,1.409455,3.359494,9.686881,2.014677,-6.405951,8.405803,-5.975864,-4.812468,-3.583150,-8.722293,9.772934,-8.602681],[1.564313,-5.520089,-3.666819,-9.826055,-6.610779,4.097213,-4.499011,2.055319,4.960306,-6.564935,8.740416,0.701729,6.459619,2.912522,-8.230491,-0.288082],[2.907706,-4.136003,8.205532,8.935674,1.319354,-3.749300,-2.905745,-5.007012,-0.874389,7.366643,9.316293,9.234867,4.250621,-9.005847,-9.317825,2.242661],[7.313932,-4.285993,-7.327303,6.264131,-5.270811,3.714814,6.447835,-5.500235,-6.981926,3.925829,-5.097220,-4.545235,-2.707513,-9.257403,2.050038,-3.046375]],[[6.935356,9.352598,9.366375,2.404969,-4.468067,-9.502243,4.484963,7.638829,-6.516921,1.789601,-3.975655,-5.524406,-2.572727,3.082751,-4.742241,0.698998],[-1.697622,8.995473,1.605302,5.298564,8.122741,1.817441,0.169075,-9.300730,-8.418209,-1.892839,2.988038,9.829863,9.554868,7.648163,-8.111565,-8.687317],[-6.807149,-6.282573,-7.185370,0.119157,4.658944,-5.946436,-9.969643,8.760291,7.897162,2.059132,-6.219486,-0.453235,8.523132,-9.266672,-6.840787,1.941600],[-3.168897,-5.710788,3.504695,5.455435,8.085522,8.575393,-6.643625,-1.323575,7.726677,4.228578,-6.330486,6.392206,1.529614,2.645900,-3.751811,-1.742892],[8.199455,-4.274549,6.358663,9.424764,9.137386,-0.913262,2.534587,2.314598,-1.086200,7.448018,0.172918,-1.305492,-1.732922,-4.772629,-9.044608,4.395016],[-5.963079,9.896802,1.411111,-6.354322,9.410344,7.539685,8.793110,0.166999,5.155693,-7.520921,-6.533277,8.049427,-5.077335,4.030414,2.913723,-3.149707],[-2.058509,2.272472,8.718626,5.628913,-9.856608,9.235194,5.642510,1.770600,2.547554,0.863283,-6.542213,-8.662471,5.755749,-1.745035,9.320859,-7.363238],[-4.114221,9.306073,6.968632,-6.632879,0.293458,6.382614,6.887578,3.812313,-7.320900,-1.618193,-1.314874,7.617705,-1.217953,-1.177655,3.522178,-2.205647],[-0.511673,3.244502,-1.338796,3.344804,1.901875,-3.130179,4.449105,-2.124505,5.008072,-9.520745,2.462655,3.044623,6.603517,7.831015,-4.338211,-5.032815],[-0.171409,-2.451052,-1.218898,-1.841854,-6.375245,-7.357739,2.107744,4.541917,7.938531,-5.235971,4.470077,-6.802904,9.984136,0.705394,4.606308,-3.729812],[7.478984,-9.148009,-4.257955,-8.943601,-0.257204,-5.902436,3.964258,-6.813783,3.435635,7.009620,-2.945624,-7.195083,-7.136733,9.137770,6.480386,3.488816],[-1.430549,-3.350634,-0.840717,6.537980,1.384178,-9.479787,4.173201,2.115320,3.421860,3.747457,5.569295,3.605896,-5.388542,1.759833,-6.381911,-6.373258]],[[1.491345,7.810946,4.641868,-6.338259,-1.444514,4.621460,4.236119,8.435537,-7.380733,-7.205075,-7.212032,0.976383,-9.882237,-4.893586,1.449080,9.238923],[-2.580936,-3.806147,8.296150,-4.164791,-2.537002,5.955329,3.656642,-0.523666,6.401433,7.082636,9.255898,-4.648667,0.117181,-2.004380,-3.363419,9.879884],[1.759719,-7.597415,-4.734238,-9.605659,2.217546,-2.625331,-7.979269,-6.989785,0.710722,6.492032,-1.037441,-1.977148,7.524038,-3.001310,-7.661484,-6.115595],[2.591794,-4.817678,8.587730,-8.850495,-8.831513,1.773756,-7.239732,2.273613,0.441426,-2.692707,-6.943306,-9.621585,5.206633,9.710792,-0.468524,3.559409],[8.910861,-8.855044,8.671323,0.664461,3.881599,-0.257962,-8.052191,-9.474445,7.432711,6.409725,6.357690,6.434462,-9.058139,3.463361,-9.969636,-4.490168],[-1.213389,-3.299084,-4.512928,-6.767627,5.731735,9.285220,-8.387513,9.171665,-1.992114,8.782521,-7.753746,-3.602438,-8.478573,-8.345276,7.662094,-1.623646],[0.243976,0.986138,-6.904811,8.718335,-7.796434,5.207868,-0.793741,-9.759102,2.373947,7.767820,-7.738243,-7.330789,4.991482,5.652648,7.821867,-9.492910],[4.847910,-7.481727,0.526683,2.571480,-6.318321,-4.648227,5.982479,3.104119,4.941986,1.295455,5.847516,-8.666248,-9.397229,8.561704,6.700738,-0.777669],[1.049095,-9.275266,-6.912447,-1.049739,-3.640303,-7.604221,-9.156888,-7.962188,-9.889989,3.007087,4.139049,-6.242828,-8.665246,9.618193,3.262453,-9.685070],[5.952179,-2.544231,-7.865653,-3.290665,4.149638,-2.246054,6.228369,9.339411,-0.619131,9.851968,-3.539434,5.503025,2.870902,-7.254436,-5.100367,8.340020],[9.599985,-3.338716,-7.457143,-9.597354,5.007061,9.740112,1.322855,-4.502244,-1.511485,3.943506,0.377066,5.085268,-6.783985,5.103118,-8.697498,-7.011672],[-1.511042,8.187485,7.605168,9.898900,-6.400595,5.082690,2.253497,-3.736119,9.856612,0.210921,-5.459776,1.796032,-3.278842,-0.722913,1.617430,2.440224]]], dtype = "float32")#candidate|1476|(5, 12, 16)|const|float32
bop_1477 = relay.greater_equal(const_1475.astype('bool'), relay.reshape(const_1476.astype('bool'), relay.shape_of(const_1475))) # shape=(5, 12, 16)
bop_1481 = relay.logical_and(bop_1477.astype('bool'), relay.reshape(const_1476.astype('bool'), relay.shape_of(bop_1477))) # shape=(5, 12, 16)
func_1049_call = mod.get_global_var('func_1049')
func_1051_call = mutated_mod.get_global_var('func_1051')
var_1485 = relay.var("var_1485", dtype = "float64", shape = (154,))#candidate|1485|(154,)|var|float64
call_1484 = relay.TupleGetItem(func_1049_call(relay.reshape(var_1485.astype('float64'), [154,])), 0)
call_1486 = relay.TupleGetItem(func_1051_call(relay.reshape(var_1485.astype('float64'), [154,])), 0)
func_570_call = mod.get_global_var('func_570')
func_575_call = mutated_mod.get_global_var('func_575')
var_1490 = relay.var("var_1490", dtype = "uint32", shape = (12, 30))#candidate|1490|(12, 30)|var|uint32
const_1491 = relay.const([[4,-7,-8,-3,-7,-10,1,10,4,6,10,3,-5,-2,9,-1],[10,9,10,-2,-8,4,10,9,-1,-8,-4,-2,-7,9,8,9],[10,7,2,7,-4,9,4,8,2,6,-6,-7,7,-1,8,5],[9,-9,1,-1,8,-9,-8,10,4,8,-8,-2,-2,10,-2,7],[-4,2,-4,-2,-5,-9,-1,-4,6,-4,-8,-9,-1,8,3,5],[-6,5,-2,-6,-10,-9,-3,4,7,-6,-6,5,6,-1,1,-8],[-4,8,-10,2,-9,-10,6,-4,4,7,-6,7,6,-3,-5,5],[-6,5,9,-7,-8,1,-1,-6,4,6,8,6,-3,-2,-1,-10],[-1,-4,-4,-9,8,-8,-4,5,-10,1,-6,1,-7,-2,1,7],[8,-4,-4,5,-4,2,-7,-9,-10,4,-7,-9,-7,5,2,-10],[5,3,7,-1,10,1,10,4,-10,8,-1,-4,-8,4,10,10],[-8,-5,8,-7,9,6,8,1,-6,7,-6,6,9,3,6,7],[4,-8,-9,-3,-5,-10,-8,-2,10,4,2,7,9,-3,3,8],[5,4,-3,4,5,-10,3,-9,-3,8,-10,7,6,6,-1,9],[-10,-9,9,8,7,8,2,-4,8,-10,-8,-2,8,-4,-2,-4],[5,3,9,5,-1,-8,10,-10,6,1,-9,-8,10,-4,4,-5],[3,10,-1,-10,10,7,-3,-10,-7,3,-3,10,3,-6,-6,-6],[8,1,-5,-2,-10,-4,-6,-2,-7,-8,5,6,4,7,-2,8],[-10,-1,-1,-9,4,6,10,-2,4,-9,-9,-7,8,2,-6,2],[-1,-1,6,7,-5,-9,-2,4,-2,-5,8,4,7,-6,8,10],[-10,4,-3,7,6,3,-6,6,-7,-9,-6,-1,-10,-5,-4,10],[8,-2,-6,6,-5,-4,3,-1,2,7,4,10,-6,4,3,-6],[7,9,6,-7,-6,2,-10,2,-2,1,-7,-9,9,-5,-5,-3],[-8,10,-10,3,-10,7,-7,-4,-8,-4,-10,-9,6,1,-2,6],[9,-4,-4,6,3,4,2,10,4,4,8,-3,7,6,10,-9],[-9,-9,1,5,-9,-5,-6,1,-10,-4,-10,1,-2,-7,10,-3],[-1,9,10,-8,-2,1,10,8,9,-8,-10,6,-7,-4,1,3],[-10,5,-8,-8,-6,10,3,7,1,-10,-6,1,-3,10,2,7],[-8,1,-5,-8,-10,-10,6,4,3,-8,-2,2,-4,9,-6,10],[-1,1,4,1,-8,6,8,-3,5,2,10,5,4,10,2,-3],[-2,1,4,9,-2,-9,-8,3,-6,3,7,-10,3,3,3,-2],[-10,1,6,-9,-7,-1,5,-10,-9,8,10,8,-9,3,-7,-7],[8,5,9,4,-4,-3,-9,9,6,5,-5,-1,6,-8,-9,-1],[6,2,-2,3,9,5,9,-7,1,-10,7,-5,-7,7,2,4],[-6,3,-1,3,2,-2,-2,-7,3,3,1,4,-3,-4,2,2],[6,-5,-2,9,-6,7,9,3,1,3,-4,-10,9,1,-3,9],[-8,6,-5,-6,-7,-9,8,10,7,-3,-10,9,8,1,3,-1],[-6,-3,-6,-2,-5,-1,4,-4,5,-4,-10,-4,-3,8,-5,-8],[-9,-2,1,10,-4,9,-4,-8,10,-6,5,9,-7,3,-7,4],[10,-3,-9,-9,3,8,-2,-10,8,-5,8,-8,-10,-10,10,4],[-2,-4,-9,-4,10,-2,8,-9,8,9,7,-4,8,3,-9,-1],[6,-2,8,-7,-7,3,1,-4,-5,8,10,-8,-5,8,9,-5],[-7,-10,-2,-3,6,-6,9,9,-1,-5,-8,-1,5,-6,-1,9],[-9,-2,6,2,5,-2,-8,-10,7,-1,-8,5,-4,-8,-2,-4],[8,2,3,-9,-3,-3,-1,1,-8,-10,-7,-2,5,-5,-10,10],[6,3,7,8,-2,-9,5,8,2,-4,2,9,7,-8,4,-7],[9,10,2,6,-10,-5,-6,-10,1,-9,-9,4,-4,-5,10,1],[-1,-5,-10,7,5,6,-6,-2,-10,4,9,7,10,-2,-3,-2],[-3,-7,10,-2,7,8,4,-4,9,8,-6,1,-7,8,5,6],[7,-2,9,-3,4,2,9,4,4,8,-7,5,2,-7,3,7],[-7,8,7,-2,3,8,-10,2,6,-3,8,8,1,-9,7,8],[-6,-9,8,-3,9,-4,6,2,-7,5,-3,5,8,-2,-1,10],[5,-2,3,5,-7,-1,-5,6,-10,-7,-4,10,-8,7,-3,9],[-6,-2,-6,-3,3,9,10,2,-4,4,-3,10,-9,3,3,9],[-4,-10,-7,7,-8,-10,10,8,-10,6,7,5,2,-7,1,-4],[-6,-7,-9,2,1,-4,5,8,2,-9,9,-6,-3,8,8,10],[8,4,-4,-3,-2,7,-8,-3,1,1,5,3,-2,-1,-2,-2],[-10,6,-7,-10,-5,-6,-4,-5,-8,-6,-6,8,-7,8,-2,8],[9,8,4,-4,-5,-3,-7,3,-5,-8,-8,-8,-7,-2,9,3],[-7,-3,-8,-1,-9,-1,1,4,5,-6,8,5,2,-8,-9,-8],[-3,7,-10,-2,3,5,-7,9,-1,-2,2,-8,-7,5,7,-4],[4,10,2,-5,5,-6,-9,6,-3,-10,-8,-7,-3,2,-6,2],[10,4,-6,-4,8,-5,6,-8,-10,3,-7,4,3,-3,4,8],[9,-5,-10,-8,-10,9,6,6,8,-5,-10,-8,-5,-9,-6,1],[1,-2,7,-1,-5,-3,-10,7,8,4,1,10,-2,-4,-8,-4],[3,9,-6,-4,-3,-1,10,2,2,-6,5,-10,5,-5,-9,9],[5,-1,-8,-3,4,3,-6,1,-10,-2,3,4,-5,10,-10,-5],[8,-7,1,-7,-5,-10,-7,8,10,-4,1,-2,9,-8,2,3],[9,6,1,8,8,-7,-1,-4,-5,3,-8,-2,-8,3,6,-9],[7,7,-1,-4,-3,3,5,4,-8,6,3,-1,10,10,-2,-3],[4,-7,-4,-3,-1,2,6,6,2,-7,1,3,-4,9,2,6],[-1,3,3,4,-4,2,2,-1,-3,7,-8,5,-5,-7,6,7],[6,3,-5,7,-3,9,-8,-10,-9,-5,1,-9,6,-7,1,8],[-1,-9,9,-6,-5,-10,5,4,-8,3,8,-6,-1,-5,-7,10],[-5,4,8,-8,5,8,9,6,-10,7,-2,-2,-7,3,-3,2],[5,4,8,-6,2,-2,5,-8,-6,5,-9,-9,6,-10,9,9],[-8,7,6,8,9,-1,-8,1,6,-4,-3,-1,6,-9,3,-1],[10,-6,3,-6,-4,9,8,7,-8,-10,5,1,5,8,-6,1],[-7,4,-9,-2,-9,-4,-3,-2,-4,-1,6,-10,-9,-10,-5,3],[-8,-2,7,-9,-5,3,5,-8,-8,6,10,-6,4,8,-7,-1],[-7,-1,7,1,2,-1,-1,-4,7,-4,7,7,-6,-1,5,-9],[7,7,7,-7,3,-5,-6,-5,2,9,9,-1,1,1,5,8],[-7,7,4,4,1,-2,-2,-9,6,4,-6,-2,6,-2,-3,-5],[7,6,-6,-9,-10,4,-5,8,3,-9,-6,5,-6,5,1,-10],[1,-8,-4,4,7,10,10,-5,8,1,5,-8,-5,10,4,-4],[6,10,8,-8,1,-9,3,-4,-8,-9,-1,5,1,-3,7,-7],[-9,5,-10,-4,-10,-4,9,-6,-2,-1,1,10,3,-4,-1,3],[8,7,10,3,-4,-3,-2,9,1,9,-8,-3,-4,-1,-7,-3],[2,-7,1,2,-4,8,3,3,9,-5,3,-8,-6,2,-5,10],[-4,2,6,3,-9,9,-8,6,-6,-4,1,-6,1,10,-2,3],[-9,3,9,9,-5,-1,-3,-10,-10,-8,6,-9,3,-3,2,-10],[1,10,9,1,10,-7,-9,-3,-3,3,6,-6,-1,-3,5,8],[7,7,9,-10,-9,3,-7,-8,-7,10,3,10,5,1,-7,5],[6,1,-1,4,9,-2,2,-5,-1,2,-2,-9,6,-6,-8,-8],[-6,-10,10,-9,-1,-3,10,-4,6,-8,8,5,9,8,6,-2],[7,-9,5,-5,-10,-1,-8,-5,1,-3,3,5,-3,9,-1,2],[-2,4,-3,-6,-9,5,-4,-4,-6,-3,6,8,3,9,-10,-1],[-3,2,5,-9,6,2,4,2,4,8,-5,9,6,1,10,8],[8,9,7,8,4,3,7,-10,3,10,6,6,-1,6,1,-7],[4,-2,6,4,1,9,-5,10,-7,-5,-2,-3,-10,-6,-6,9],[9,-3,-3,-6,-6,-6,-4,-6,9,-3,6,3,8,-8,2,-4],[-1,7,-4,2,-8,9,-5,4,-8,-4,-5,-2,-8,-7,10,-1],[-8,6,-1,-2,-7,2,7,7,-5,2,7,4,4,4,-5,-4],[-3,-5,-1,5,7,1,5,1,-8,-1,9,-10,9,7,-1,6],[-5,2,7,5,7,2,-3,-4,-6,-8,4,-7,4,-6,-7,10],[-3,-8,-10,-6,9,10,1,-5,-3,-3,8,8,6,-6,-6,4],[1,-2,5,7,-8,-5,3,9,10,-10,3,-2,-4,3,7,1],[-9,8,-3,-7,-3,1,10,-1,3,1,-1,-7,1,5,9,-7],[-1,2,1,-6,-2,2,8,-4,-5,8,-8,-1,5,9,5,2],[-3,1,-6,10,-5,-1,6,-5,4,8,-8,-3,8,7,-9,9],[-5,-6,7,-10,10,5,-1,-10,-6,7,9,7,3,1,4,4],[2,4,-9,2,-8,-7,9,-3,6,-9,1,4,-1,-8,9,-8],[-5,10,7,-3,-9,3,8,5,10,-2,-9,7,-5,6,-8,-3],[-5,8,5,6,-4,3,6,7,-1,7,7,-9,8,6,-8,-10],[4,-4,9,-2,-8,4,-5,10,-2,-6,-10,3,2,5,-8,2],[2,10,5,-4,-1,7,-1,-1,-5,-6,-4,2,-6,-8,3,-8],[-8,-8,-2,-6,-3,8,3,1,-8,10,3,10,10,-6,7,-9],[4,-8,10,-9,8,-10,-7,-8,7,-7,-3,-2,2,-5,3,4],[1,-4,-5,-7,8,-5,-8,8,-2,-8,9,-7,7,1,-7,-7],[-4,-9,-2,3,4,7,-2,9,-9,8,-6,-2,-8,5,6,4],[-3,-9,1,-6,-10,9,7,6,5,-7,-1,-6,-10,6,2,-5],[5,1,6,3,2,7,-9,7,-9,5,-1,-2,-1,-8,10,-10],[-7,-5,8,-2,7,2,-4,-10,-3,-6,-7,7,9,-10,9,6],[-1,-3,-8,-1,-4,-2,2,-3,8,-1,-8,-10,-6,-5,-2,-2],[-3,-4,-10,5,10,4,9,8,7,-2,-2,1,7,-2,9,2],[3,-3,4,8,-7,-7,2,-6,7,-3,2,-4,9,8,3,10],[-9,-10,-4,7,5,-10,9,5,5,2,5,8,5,-2,9,6],[-8,1,7,3,-3,-6,-10,8,-2,-7,-10,10,3,4,7,-2],[-8,-7,2,-4,5,-9,2,2,-2,2,-4,-7,-8,10,3,-10],[-6,4,1,5,8,2,-3,-8,3,-7,4,-5,5,8,9,8],[7,5,10,3,-6,-9,-9,-6,-9,-1,-6,-5,-4,-8,-1,10],[6,5,7,-5,-2,4,3,6,-4,5,5,-9,-3,6,3,3],[3,-5,-5,-3,-5,2,10,2,-7,2,-6,-5,-7,-4,-6,10],[-4,-10,-7,8,2,4,8,-4,10,8,-8,-7,-1,7,-8,7],[-4,-1,-1,-7,-10,5,-6,4,-6,3,5,-10,3,-6,-4,4],[-1,3,3,5,10,1,10,3,8,-1,-10,-9,-6,-5,-9,10],[-4,-2,-2,10,6,9,9,-3,10,9,-4,-2,-5,-3,-5,-2],[4,6,10,8,-9,3,1,-6,1,8,6,-8,-6,-3,1,-8],[1,3,-1,-7,-9,8,-8,10,10,5,-7,-6,1,4,7,9],[7,9,-4,-8,-1,-3,-3,-6,-1,10,3,3,-7,6,-2,-4],[-10,8,-1,6,-5,1,-4,-10,10,-6,-7,-7,-7,-8,9,-9],[5,2,4,-10,-1,-6,9,-3,-7,2,-8,-6,-1,6,-6,-8],[10,9,-9,2,1,6,-3,1,-6,2,3,10,4,-5,-1,-1],[4,-5,-10,-5,-7,3,10,2,10,-3,-3,6,-9,8,-3,-10],[-1,-1,4,-2,-9,-2,2,6,-9,-4,9,10,10,-7,6,-1],[2,8,8,-8,-8,5,-6,1,-9,8,2,8,-7,-4,-2,-2],[6,2,-1,-2,-3,-8,-9,6,3,-6,3,8,8,-5,-8,7],[-6,-10,1,7,-9,5,-3,6,5,2,5,-8,-3,-7,1,-4],[5,5,6,-2,-6,-4,7,6,1,2,9,3,4,-10,4,-1],[10,4,9,-2,4,5,-8,-5,-8,-7,8,8,7,-6,-3,-7],[4,-10,9,7,10,5,2,6,-9,-6,10,9,-1,-8,2,9],[9,9,1,-3,8,8,-10,-10,5,5,10,-7,-9,-7,-2,-8],[-6,-1,5,-8,2,2,10,2,-10,3,1,10,4,-10,6,9],[7,8,6,6,4,1,-2,9,9,1,-3,7,-10,4,-7,7],[7,-5,-5,-2,-10,3,5,5,8,-2,-9,-8,6,-7,7,6],[-1,-6,3,-5,10,-9,2,8,2,-8,-9,-1,8,-9,-2,-10],[5,9,-5,-2,-2,9,5,-1,7,7,8,10,-8,5,-3,4],[3,7,8,3,2,-2,-9,-5,6,2,-2,10,-3,-9,-9,6],[6,10,-5,-2,6,6,-10,-4,3,-8,2,4,-2,4,5,3],[-3,5,-7,7,-2,-9,-1,9,-10,-8,5,-8,4,10,1,-3],[8,2,-7,-9,1,-1,10,6,-7,2,8,-10,-1,-4,-6,7],[-4,-9,10,-10,-3,-5,3,5,-10,3,-9,8,3,-4,-8,-3],[4,3,-8,-6,8,-6,-7,-10,-1,2,-5,-2,4,-9,9,-7],[2,8,5,-2,7,3,-10,8,-5,1,-3,-10,1,9,1,-4],[6,-2,-7,-3,4,-8,-10,-6,-7,6,-2,-6,8,-4,1,-2],[9,9,10,-10,-9,-9,-5,3,1,-3,-10,5,-4,6,6,-4],[2,3,-7,-6,-1,5,-9,-1,2,3,-3,-7,-5,4,-1,-8],[-4,-6,-4,-7,-7,-3,5,-5,6,-8,-6,7,9,5,-4,4],[-2,8,3,10,-10,-4,-5,-1,-9,-1,5,6,-3,-4,4,4],[-8,-9,-8,9,-2,-4,-4,-9,5,7,9,-1,-4,9,3,-3],[6,10,-3,-3,-3,-1,3,7,-1,7,8,2,5,2,-5,-10],[7,4,-10,4,-5,8,4,-3,10,4,-7,1,-10,-1,6,-5],[-1,4,7,5,-2,7,8,2,1,7,-4,2,-1,-8,-4,1],[-1,7,4,9,4,-3,-7,2,3,-6,8,-1,-2,-8,-2,-4],[-8,3,4,8,-8,-2,-6,-3,-4,5,8,2,5,-8,3,-1],[4,2,4,6,9,-6,-2,5,7,-9,8,-7,-5,9,9,-8],[-4,1,-10,-1,-4,-5,6,-2,10,2,-5,10,-3,8,8,5],[8,-9,2,7,-9,-2,-2,9,-6,-1,-9,-2,1,6,-8,3],[-6,6,4,-4,8,7,-10,6,-7,-10,-8,-1,-2,10,8,10],[6,4,1,-3,3,1,-8,3,3,-2,-4,2,9,-10,5,-3],[-7,-3,6,-9,7,4,8,3,-1,-4,7,4,10,-5,3,-8],[-8,9,10,-2,-10,4,-6,-9,1,3,-1,-6,-9,8,2,4],[-8,1,-3,-4,-3,-10,3,-5,5,4,3,7,-2,5,10,3],[4,-3,-9,7,5,-1,7,5,-4,-7,1,-10,-10,9,4,-9],[8,-5,-9,10,-2,-4,-9,-10,-10,-9,-5,-5,5,7,8,10],[-8,-6,7,-2,-7,7,-9,-8,-8,-9,-7,7,10,3,-10,7],[-6,-6,7,4,-6,1,-10,4,9,10,-2,9,-3,-2,-5,-2],[2,-6,7,4,-2,9,-7,9,-8,3,8,-4,6,5,-4,2],[4,-6,9,4,-8,-3,-8,2,-3,-8,9,-1,7,-4,-2,9],[-6,5,-7,-7,2,10,2,7,-7,6,6,-10,-3,6,-3,-1],[9,4,-5,-1,-1,-8,-4,-10,-10,6,-4,-9,-1,4,9,2],[-1,6,-7,4,9,-3,-7,-6,9,-2,-10,-4,5,6,-1,-2],[7,8,-4,-4,-4,-7,-1,-9,5,-5,-4,6,7,1,-10,-2],[-10,1,-5,-8,3,4,-6,-4,6,-6,8,-10,-3,5,-6,10],[-9,10,5,5,-1,2,1,2,3,8,1,-8,-2,10,2,-4],[1,5,1,-6,6,-4,1,-4,4,-10,3,9,-3,-3,-9,5],[2,9,-1,-8,2,6,5,-10,4,-4,-7,4,2,-6,6,-4],[-3,-2,8,1,-1,3,-9,-5,-4,3,3,-3,5,-9,-1,1],[-4,8,2,-1,-8,-9,7,1,-4,-5,2,-4,-7,9,-6,2],[3,3,9,8,-8,6,2,9,4,5,-1,-2,-3,-6,3,-2],[-10,-4,9,-8,-1,-2,-4,1,2,3,-6,-4,7,-10,10,7],[-5,8,-4,-8,-3,-8,-1,6,-4,9,-7,8,2,4,-8,-1],[3,6,8,-2,6,1,-8,3,7,1,4,7,5,-2,-4,7],[-4,5,-1,-4,-7,-6,-1,9,10,3,-1,-4,2,1,-8,-5],[3,4,-7,1,-5,-2,-6,6,3,2,10,10,9,-9,-2,-1],[6,-9,6,-8,-6,7,3,6,1,-10,3,2,-3,-10,7,2],[-7,-6,-10,5,-7,6,6,-6,1,7,-9,-7,-1,-9,-8,-2],[9,-1,-2,-5,-7,-9,-4,-9,4,-10,-6,-5,-5,5,-5,-2],[5,-5,8,-6,-2,9,1,-2,5,-5,-2,7,-7,3,-9,8],[8,1,-3,2,-1,1,8,-6,-9,3,-3,9,3,2,6,-6],[-3,-10,-3,4,-3,7,-2,6,3,-1,10,-6,1,-4,8,-1],[6,-1,-4,-2,-2,2,-1,-5,10,1,-7,4,5,-7,8,7],[-9,-5,-5,-2,8,-1,4,3,4,-5,-3,1,9,-2,-5,-2],[-7,-7,-4,9,3,-10,8,3,9,4,-7,1,-4,2,2,-4],[-9,-3,-10,-4,4,1,9,-7,10,-10,8,-4,4,2,9,9],[6,2,4,-8,2,-4,3,-9,3,-1,-3,-5,-10,5,4,-1],[5,6,-1,-10,6,-2,-3,4,-2,-9,2,4,-3,-3,-2,-7],[8,-8,10,10,2,-6,6,-3,5,-3,-7,8,5,-5,6,6],[5,-2,2,10,-7,10,-6,-2,-6,5,2,-3,7,-6,10,-8],[5,-2,6,-10,-10,-6,5,7,-5,-7,8,-3,-8,9,-10,6],[10,2,-10,9,-5,6,7,5,6,8,-2,10,-2,1,-9,-7],[5,-5,2,-6,-2,-8,6,10,5,-9,-9,-8,-9,2,6,-8],[-9,-5,-10,-7,-3,-1,10,2,-9,2,-9,8,-9,-7,10,-5],[8,-9,9,3,4,-1,-5,-8,4,-2,-2,3,3,-8,5,3],[3,4,-9,-6,-10,5,-1,-2,5,-8,-6,-8,-10,1,2,-3],[-5,9,-5,-5,1,2,5,7,-6,8,-5,-1,5,-4,-6,-9],[5,5,2,-3,-3,5,9,-9,6,-8,-4,-9,1,-3,10,9],[2,3,5,-7,3,6,3,-6,1,-9,1,3,1,10,-1,-7],[5,7,8,-8,-5,10,10,-6,-9,7,4,5,-7,9,6,-1],[5,-8,-1,3,-1,-5,-2,3,9,-3,-9,10,-3,-3,4,7],[1,4,4,-9,2,-7,6,-6,-9,10,7,-5,9,-9,-2,-2],[-4,5,4,4,-8,10,2,-1,-5,10,-10,-2,10,4,8,7],[2,3,-5,-6,9,-9,4,-3,-2,10,-5,-3,-6,1,2,-7],[7,2,8,-7,4,9,7,-8,4,-10,-2,5,7,1,1,4],[7,-2,-4,1,4,-7,-1,-9,-2,-3,-10,-2,4,4,3,-7],[-4,5,-9,-5,8,5,6,4,1,-2,-6,7,10,-9,-9,-3],[-5,-8,-9,-9,-6,6,-6,-10,8,7,-8,7,-6,-9,1,6],[2,-2,7,-8,4,5,-6,7,6,-1,-8,10,-3,7,-3,-1],[6,-8,5,-8,-1,8,-7,-3,-1,-5,3,3,6,-9,4,8],[3,7,-3,6,-9,1,-1,8,1,1,1,-9,5,-10,7,-10],[-5,10,-3,9,-5,9,9,6,-3,3,10,-2,9,3,-1,8],[-2,-1,-9,7,3,7,-3,3,1,-1,7,7,5,-1,7,-2],[10,-3,4,-2,-8,-2,-4,-9,8,-10,-1,-2,-9,5,-4,4],[-10,4,7,-2,7,-5,8,7,-2,-2,4,-3,7,-8,5,-9],[-6,-1,-1,5,-9,-8,-4,3,-10,1,9,6,4,-10,-7,-5],[2,-6,-1,10,3,4,-10,5,2,-5,-2,8,6,-4,-7,-6],[7,1,-6,7,-8,9,8,10,-10,2,8,7,-5,7,-1,10],[-8,5,3,-3,1,-9,-4,10,3,-7,9,5,4,-8,8,3],[1,7,-10,-5,2,1,-4,-1,2,-10,-9,-8,10,10,-6,-1],[10,2,-10,-10,-2,8,3,-5,-7,-7,1,-1,-1,-6,-3,3],[6,-3,2,-1,-8,10,-10,6,1,4,-7,10,6,-3,4,-4],[3,1,3,-10,10,-2,-7,8,2,-10,6,8,-8,3,-9,6],[-6,-7,-6,-2,1,7,-4,5,-4,8,-7,-6,-7,8,-7,8],[-9,2,6,-2,-10,4,-1,-2,6,-6,4,7,-2,7,-7,-1],[-2,2,7,-4,10,-10,6,7,-4,6,-7,-1,-6,-8,-5,-1],[-4,-4,6,9,-5,6,-3,-5,6,-4,4,1,4,-5,10,6],[9,-6,-8,-1,-4,-6,10,1,-3,-10,-8,9,4,-1,-9,8],[-8,1,9,-1,-8,9,10,-7,-5,-2,-10,7,4,-8,-6,9],[5,4,-1,-7,1,9,6,-4,3,-5,8,8,-3,-5,7,5],[-5,5,3,6,-10,5,1,8,-9,6,7,7,2,-9,5,10],[2,-7,9,-10,6,-4,7,10,7,1,-2,3,10,7,-9,-7],[-4,7,6,10,-3,1,-8,-5,-9,-8,7,-8,-7,-6,2,7],[-8,7,10,-7,4,2,-3,7,-5,4,8,3,-9,-2,-2,6],[1,-3,-3,-3,-2,-10,-4,-8,4,5,6,-7,-6,-3,-6,-8],[4,-10,-6,5,10,3,-2,4,-3,-10,-4,-3,6,9,-6,-10],[10,7,-3,-4,-8,6,9,8,9,-2,-4,5,2,-3,3,-9],[3,9,-8,10,1,8,8,-2,3,-9,-7,-7,10,-7,3,-4],[-4,4,-7,8,2,5,-1,6,1,8,-2,8,5,3,-3,-2],[-1,1,-3,-5,4,-5,-7,10,6,7,-5,10,-10,5,1,2],[-6,-9,1,-3,5,3,-9,-9,1,5,-1,-3,7,6,-7,-1],[1,-3,-3,9,-4,-10,-3,10,-7,9,1,10,-2,-7,5,4],[-7,8,3,9,-10,-4,2,-9,10,-8,2,8,-6,-4,2,-7],[-5,10,4,9,10,9,7,2,5,-8,-10,2,-3,7,4,-6],[-10,-1,8,8,-3,-10,4,10,2,8,-6,7,5,-7,-7,10],[-1,-2,10,7,-9,-4,10,4,-1,5,-3,7,6,10,4,9],[-10,-4,-2,-4,1,6,3,5,-4,-1,3,-8,-1,10,-8,-3],[-3,9,-10,-8,4,-10,6,10,6,-9,-9,-7,5,8,-8,2],[-5,-7,-7,-6,9,-5,-3,-4,-8,9,-2,-10,4,10,6,-3],[4,-8,3,1,1,-8,3,-3,-5,-10,9,-3,7,-2,7,8],[5,4,9,1,-3,4,1,9,-2,-4,-3,1,-10,5,-10,1],[-8,-3,-10,8,-3,1,2,1,-3,2,-10,5,7,9,2,3],[5,-8,-2,10,9,-3,-3,4,7,-9,-10,-1,6,-4,-10,-10],[-7,-1,7,8,3,6,6,-10,-5,-2,-5,1,4,10,6,-8],[10,-4,3,2,1,-1,-9,10,2,-10,6,8,9,-4,8,10],[3,5,-3,-1,-7,1,-5,-2,6,4,3,2,10,9,-9,-5],[10,-8,7,9,-2,-9,-3,4,1,1,-1,-8,-10,10,3,-2],[5,7,-7,-5,-7,6,7,1,-9,4,7,3,-8,-5,-8,5],[6,5,-8,-4,7,7,9,4,10,10,-7,-5,3,7,6,5],[3,7,4,-8,-9,-5,-3,-8,6,-10,-10,9,4,5,7,4],[-1,5,10,6,-8,-2,1,-4,8,1,9,-1,7,5,-7,-6],[-9,6,4,-1,-7,3,-4,7,9,2,-8,-10,-1,9,-3,-8],[7,2,-8,-6,-7,-4,-4,-3,-9,-2,7,1,7,6,8,10],[10,9,-6,-4,3,-4,-7,3,-1,2,-7,6,7,-1,9,-10],[-5,8,5,-4,2,-7,-6,-6,3,7,-2,-3,-8,-6,8,-6],[7,4,-10,10,10,-2,8,-5,5,-3,6,10,10,9,-9,1],[-2,8,7,3,-7,-4,-10,-8,-1,-6,6,-1,9,4,-6,6],[2,-10,-3,-8,8,7,-1,5,-3,10,-6,-6,2,4,7,-6],[5,-9,10,1,9,6,-3,5,1,-5,-6,-1,-3,6,7,8],[-9,10,-6,7,-1,7,-3,9,-8,2,-7,2,-10,-7,-2,6],[-4,5,9,5,-5,1,-7,-4,5,10,-9,-2,-5,-3,-4,10],[-9,-7,-7,8,7,9,-2,-8,9,10,4,-6,-1,-6,1,-4],[-1,9,-3,9,10,1,-3,10,-1,-1,-2,-4,7,1,-5,8],[-6,-3,6,9,2,-8,3,6,-8,-8,3,7,-1,-9,3,-10],[6,10,-7,-8,8,-1,-5,4,-3,6,-6,5,-3,-2,2,7],[-7,-10,5,3,-5,6,-1,-3,-5,-10,4,-6,10,5,-2,-7],[-1,-5,-6,-2,3,9,8,-10,-1,-2,6,3,-8,5,3,10],[1,-2,6,-8,-4,6,-8,-5,-7,-5,-6,6,-9,5,4,-7],[3,-4,7,-1,-8,3,3,7,6,2,2,-10,-10,-2,10,-1],[-3,5,8,1,3,4,-3,-8,4,-2,8,5,7,-2,1,2],[-2,-3,1,10,5,1,-6,-8,-9,-5,4,-4,-1,8,4,5],[-4,-8,-10,10,2,-5,10,-7,4,5,5,-9,-6,6,6,5],[8,7,-2,5,-8,-2,-4,9,-2,5,4,4,10,9,1,-6],[-9,-1,9,5,-6,10,-8,5,-1,-7,-6,-4,-1,4,1,5],[4,3,1,3,-6,5,-9,-4,-3,-7,2,-7,6,-8,-1,7],[8,9,7,-5,4,-7,-2,-5,-9,-6,-6,5,6,9,6,-3],[5,4,-5,-4,-10,-5,-9,-9,-5,-1,-10,2,-10,3,6,-10],[1,2,-2,-8,7,-5,4,-9,9,-9,-6,4,2,4,7,-9],[-8,4,-9,-4,-6,7,3,8,-5,2,9,-9,-5,-9,9,-1],[-2,10,-7,-10,-3,2,2,-9,-6,10,7,6,-4,-4,-4,-3],[10,-10,6,-10,8,7,2,8,-5,3,8,3,6,-7,10,9],[-3,-7,-10,10,-8,9,-2,-10,-5,-8,2,-7,9,-1,5,-9],[8,9,-6,7,5,6,2,4,-6,-10,-4,-9,4,1,9,-8],[-7,-6,-8,-7,-6,2,-6,5,9,-2,-6,-3,2,-10,4,-7],[-6,4,-5,-9,-5,-1,7,4,-5,8,-9,2,5,-10,1,-9],[-2,3,5,2,-2,10,-1,5,3,7,-4,-1,2,-5,-3,-2],[-9,-4,-8,6,-9,1,-3,8,-2,7,-6,5,4,-5,-6,3],[3,-4,6,-4,8,-7,-4,8,9,-1,8,7,8,8,-8,-9],[-10,3,-5,-3,2,6,2,-3,-3,-10,-2,2,8,4,-3,-6],[-8,-1,7,5,-3,-9,-9,7,-9,-3,-3,-3,-7,6,1,-4],[-4,1,-8,-3,9,5,-6,-2,-1,-8,-1,7,7,7,7,9],[8,-3,8,7,-9,7,-7,10,5,-4,8,-2,3,5,3,-1],[1,7,-1,-7,9,4,5,3,2,4,-7,5,10,-1,-1,5],[-6,3,9,8,8,-5,7,-10,-7,9,5,-5,-3,9,9,-9],[-8,-4,4,6,-3,-4,-9,7,3,10,-1,4,-3,9,-8,-1],[-3,-5,5,1,-2,-6,1,9,-1,-1,-7,-6,9,4,-6,-4],[-8,-7,2,-10,-5,7,2,-10,-6,5,8,-4,7,-4,3,-3],[7,-5,-2,-3,-3,-2,8,-8,-7,-8,2,-9,1,3,1,6],[-8,9,6,-6,6,-4,4,-1,-4,-3,2,-7,-7,-5,-8,10],[-2,7,-3,-8,3,-5,5,7,4,6,-5,-7,3,3,-2,7],[-2,-3,7,5,-5,-10,-6,-8,8,4,-6,6,-2,-1,3,-10],[2,1,-7,3,9,-2,10,-3,5,2,8,-9,6,-5,-7,4],[-10,-3,5,9,-9,9,3,6,1,7,5,-10,-7,-1,-1,8],[-2,-8,-4,1,10,-3,-3,-8,8,-2,5,-5,6,-1,-1,-8],[-9,-1,3,-8,-3,-5,9,-1,-3,9,6,2,-10,-10,-3,3],[5,5,-8,-3,6,-7,3,4,-2,4,7,-1,3,7,10,-8],[-5,-6,-7,-4,-9,8,9,-8,4,-1,1,9,3,-8,7,-9],[10,-5,-6,-10,10,-9,-1,-1,10,-8,-6,-3,-5,4,-10,4],[10,-6,6,1,7,6,8,3,-1,-9,7,6,6,4,4,-5],[-7,1,9,1,4,4,-7,8,-5,7,-10,6,4,9,7,-2],[-4,-8,-5,-1,-8,-10,9,5,6,9,-6,9,4,9,1,6],[2,-10,-4,-7,5,9,5,1,-5,-7,-5,2,4,1,-1,-3],[-2,2,-6,-10,6,7,3,-3,8,1,4,-3,5,6,-2,8],[1,-1,9,-8,-1,6,-2,-2,9,-4,-5,3,-10,5,-7,-5],[5,3,-7,10,-5,2,7,10,5,1,2,1,5,5,1,10],[7,-1,-6,4,9,-1,-10,-2,-8,9,7,5,4,-5,1,10],[-9,-4,-3,-6,-6,7,-1,-5,2,-8,-1,-4,-2,-1,-4,-3],[-3,-8,9,9,5,3,4,3,-3,-6,6,1,-9,-3,7,-6],[4,7,9,1,1,-10,-4,-3,-5,-1,-1,1,9,2,3,-6],[-3,-2,-8,-6,7,-6,-8,7,-1,6,-3,3,4,-5,5,-6],[4,-1,-1,1,-3,-8,-7,-4,-3,2,9,5,-5,10,2,8],[-4,-6,8,-4,6,1,6,4,8,-3,-3,10,-9,-8,8,4],[-7,-8,10,5,3,1,-3,-9,9,8,10,-10,2,-9,3,4],[-8,4,-6,5,-5,1,9,9,-9,2,6,-8,9,9,-3,-4],[9,8,-7,5,7,-9,5,4,2,-5,9,9,3,10,2,4],[1,5,4,9,-7,8,6,-7,-2,-9,6,-7,-5,-2,2,-6],[-1,-3,4,-6,2,-6,10,9,6,10,-5,10,7,-4,-1,4],[-5,4,-2,-4,-8,5,2,9,1,9,-10,3,-10,-6,1,-2],[10,-8,8,-9,10,6,7,1,7,-3,8,2,-4,10,10,-2],[8,3,1,4,-5,5,-6,-1,4,-7,9,-2,-2,8,2,2],[-9,-8,8,-4,-10,6,-3,8,8,-2,9,-10,6,-2,-8,-10],[3,9,-10,4,-10,-6,5,2,9,10,8,6,-6,2,-7,10],[10,9,2,10,2,-7,-7,-10,-3,-4,-10,4,-3,1,9,-7],[-3,-9,-2,9,4,-2,-6,-9,3,-1,6,-1,2,-8,-3,3],[5,-1,10,10,-7,-5,5,2,-9,-2,-10,5,-3,4,2,-2],[-9,4,4,-5,6,-3,-6,-10,-10,2,-5,2,2,6,5,-10],[2,6,-10,6,-3,-9,-6,3,-3,-7,-4,6,7,1,-1,-10],[7,-8,9,2,-5,-10,-2,-8,-2,3,6,-3,3,-6,-6,-5],[4,-9,9,4,2,6,9,-9,1,-6,-1,-9,-4,9,3,3],[9,-10,-3,2,-10,-9,4,-7,-10,-9,1,-5,3,5,8,-3],[9,-3,10,10,-6,9,7,-3,9,7,-10,-7,6,10,1,1],[9,-6,1,-8,-5,7,3,10,-1,-5,1,5,5,3,9,10],[8,-8,10,-8,-3,6,4,-9,2,8,-6,-3,8,-1,-8,6],[6,8,10,-10,2,8,-3,3,-10,1,5,9,9,-2,-2,4],[-4,4,10,-4,4,-8,-8,-4,-5,-4,-10,6,4,8,-8,-4],[7,-6,-3,-9,4,-6,2,3,-9,-8,2,10,-1,2,1,2],[5,4,10,-9,-3,-9,6,-10,8,6,9,-6,2,-2,5,10],[-3,1,8,-8,-4,-5,8,5,-3,-4,5,-5,-9,4,10,-5],[10,8,6,10,2,8,1,6,4,-8,-9,10,9,-1,7,4],[3,-5,10,9,-9,4,4,-5,9,7,-6,5,8,6,5,3],[4,9,3,7,-8,-3,-10,-1,10,6,-5,-1,10,-2,-9,-5],[3,8,5,-8,-4,-5,7,-3,4,-5,2,-5,5,-9,2,-9],[9,9,-1,-7,5,-9,-8,1,2,4,-4,-8,8,-3,-3,-5],[8,-6,-10,3,-7,-10,-5,-8,9,-4,-4,-1,-2,-3,9,-7],[2,7,8,-7,-2,-9,-10,-7,2,-1,4,9,8,-10,9,-5],[1,-3,10,2,-7,-3,10,5,3,-8,-1,2,-8,8,8,-8],[-3,-7,-1,9,2,6,-3,-3,8,-3,8,-8,-6,2,10,-4],[-1,8,-10,1,-8,-5,7,-7,-9,3,4,4,2,5,-5,4],[9,-9,8,2,-6,-7,5,9,-4,3,-8,9,2,9,-2,5],[-7,-2,-6,-1,1,5,-3,4,-7,-5,-6,4,-8,-7,-7,4],[9,-9,10,9,-7,9,-1,7,-10,4,9,-3,-10,-1,-5,1],[1,-10,9,2,6,-5,-4,4,9,1,-5,-6,5,1,2,-3],[-1,4,4,5,-9,6,5,9,-9,3,-5,8,6,-9,1,-1],[-8,-2,1,2,-8,-3,-10,8,-5,1,9,9,9,4,-1,-2],[-10,-1,10,10,2,2,8,-1,-10,-2,-4,-6,8,-10,-5,-5],[-7,-5,-10,8,10,5,-1,-10,3,-1,4,7,-9,10,1,10],[-2,8,10,7,7,8,10,-8,2,7,9,8,3,2,3,-6],[7,-10,-2,-4,9,-7,-2,2,-4,-1,-5,1,-7,3,-4,-7],[-5,-5,2,-5,3,-9,-5,-2,10,-4,-2,-2,-7,-9,6,-10],[-2,7,6,6,2,-8,-6,7,-4,2,5,5,1,-5,-9,-7],[1,7,-3,-1,-8,-3,-9,-2,-1,-7,8,-10,-5,7,9,-6],[-7,-8,-8,7,-7,-2,-1,9,9,-10,-9,8,-5,-9,9,7],[-10,8,5,4,6,-8,10,-8,-5,-6,9,10,8,-10,-5,-5],[-1,5,-8,8,4,-10,-5,-3,-9,5,-2,-3,4,-9,10,6],[-5,-9,-9,-9,-5,7,-9,2,-9,5,-4,4,-4,7,-7,7],[-3,6,-7,4,4,-4,2,-3,5,-4,6,6,-1,6,-7,1],[-1,-3,2,2,9,-8,-2,-3,9,-5,-5,7,-6,-5,7,-5],[3,-9,-5,10,7,-6,1,5,7,6,-2,10,-9,-1,3,5],[5,-2,4,-6,-1,9,2,-1,3,2,4,3,9,10,-1,-1],[-8,9,-4,-5,9,-7,8,-1,9,8,5,7,-7,-9,1,-7],[2,4,-2,3,3,-2,1,-3,7,6,-5,10,-1,10,-9,-4],[4,-9,5,8,1,8,-3,2,4,-2,10,-9,-10,-2,8,2],[-4,-8,2,-8,6,-6,-8,-7,-6,-9,2,9,9,-1,-7,-10],[-3,-1,-6,7,-6,-1,1,7,8,4,-8,-1,4,5,-4,-6],[5,9,6,-5,9,9,7,-7,-5,-4,6,7,7,5,9,8],[9,-7,-5,9,2,4,-5,-4,-4,1,-4,-1,-4,-6,-8,-7],[2,-1,-8,-1,-8,8,5,-4,-10,8,7,9,-8,2,-9,-2],[-5,-8,-3,3,-3,7,-5,-7,-7,-1,6,6,6,9,-7,4],[4,-6,2,-10,-2,-1,9,-4,-2,-1,-8,4,6,-7,2,6],[-5,3,-2,10,-10,-3,-10,-5,3,-3,-4,-2,10,-5,1,10],[-2,5,6,9,8,-6,6,-3,9,9,2,-6,-1,5,-10,9],[-8,7,-2,1,-8,-10,-10,5,-8,-7,-4,-3,3,6,-2,9],[-1,-5,3,-3,-5,-5,-6,6,-5,-4,-5,-3,-4,-4,-7,-8],[9,1,1,5,9,-3,-2,-8,-2,-5,9,5,-10,2,2,-4],[-3,-3,1,10,-7,6,2,6,6,10,1,9,-6,7,-10,9],[7,10,9,9,-5,-7,9,-10,5,2,-7,7,3,9,6,9],[-9,3,-10,-6,-1,1,-8,-7,-1,1,10,-1,-2,6,-10,3],[-3,9,-3,8,10,4,-10,-1,9,-9,9,-2,2,-3,5,2],[-6,-1,6,3,-9,-3,9,10,-7,-9,-2,-9,10,-7,-10,3],[9,-5,5,2,8,-5,-1,-6,-7,-5,7,3,8,2,-10,3],[1,6,-5,-2,-6,-1,3,5,4,6,8,1,-9,3,10,-9],[-4,-6,5,-4,-6,-8,-2,-8,-7,-8,-6,-10,-1,-6,-10,1],[2,-7,-7,8,5,2,-10,-6,7,-1,-2,-7,5,6,-8,1],[6,7,-8,1,-2,3,-5,1,2,2,5,8,1,-7,-1,-9],[6,-4,-2,10,-8,-7,9,-5,-4,-4,6,-10,6,-8,-10,5],[2,10,6,9,2,-9,6,3,-10,-10,9,-5,-9,-1,1,-5],[-9,8,-7,-9,3,-10,-9,10,2,-8,-9,-5,8,-9,5,-4],[1,8,-9,6,3,2,1,-5,-9,3,8,6,7,8,1,6],[4,-1,1,4,5,9,-5,10,-7,-3,2,-4,-7,1,-2,10],[-3,8,8,9,7,1,10,7,-4,-4,-2,-7,1,-6,7,1],[-7,-3,3,8,9,4,-9,-9,-3,-1,-5,1,8,-1,3,4],[-2,-5,-2,5,-2,-3,1,-6,-8,7,-5,9,6,-8,-3,9],[3,2,-6,-5,7,-9,-1,-3,2,-2,10,10,-3,-7,3,4],[9,-9,7,3,-5,2,-5,-5,-5,-6,10,3,6,-10,5,7],[1,8,4,-2,2,-5,9,8,8,-10,-1,10,2,-8,-3,10],[-3,-2,2,5,9,2,1,3,3,-8,8,-4,9,7,9,7],[-8,8,-4,3,-3,1,-7,1,-10,4,8,-6,10,-10,3,-10],[8,-10,-10,-6,-7,5,1,-4,7,-9,-8,4,8,-8,1,1],[-5,-2,3,-8,-6,4,8,-10,-10,4,4,9,-3,-3,4,-6],[7,5,-7,-10,-3,10,-6,10,9,-5,3,-5,10,3,-3,4],[6,-10,2,5,8,-9,-8,-2,3,8,-3,2,-10,-9,3,-5],[-5,-3,-8,-2,-6,1,-10,-10,10,10,-10,9,9,-5,3,3],[-6,5,-7,-10,-8,6,4,4,2,9,8,2,-8,-10,-8,9],[1,9,10,8,10,7,-10,8,-5,-5,-7,-10,5,-7,5,8],[9,4,7,-9,-8,1,-7,1,7,5,-5,10,7,5,-3,-7],[-2,-4,-5,-6,-9,-10,-5,10,-8,-10,10,-3,9,-8,-2,-7],[-8,-1,-5,-5,-3,3,-7,-2,8,6,-1,-6,8,5,10,2],[4,-10,-10,5,9,1,-1,-1,-4,8,5,-6,-9,-2,-4,9],[-8,6,6,-1,-9,3,5,10,8,-5,-3,7,-3,-3,1,-4],[2,7,1,-3,2,10,-9,-7,8,1,4,-3,6,3,-7,8],[-8,7,5,-10,7,4,-2,-2,-4,2,-10,-4,-1,6,7,9],[-9,5,-8,10,-6,-3,5,7,-8,-7,3,7,-10,2,-6,-3],[7,8,-2,3,1,6,-3,5,-3,-2,-4,10,3,-3,1,4],[5,-5,8,-7,-8,-7,8,-10,6,9,2,-8,-9,4,9,2],[-7,1,9,-8,-2,5,1,-3,9,9,-9,-6,7,2,7,-10],[-1,2,10,6,-9,2,-6,4,-6,7,-9,-8,5,4,4,-6],[6,6,2,-8,10,-8,8,4,-7,1,-5,4,-9,-2,2,-9],[-10,6,6,10,7,-4,1,9,10,6,1,7,-8,-2,-1,-1],[-5,-2,3,6,-1,5,8,-2,3,6,-9,3,-4,-10,-5,3],[-9,10,-6,-2,-9,-10,10,-7,-9,-9,6,6,7,4,-8,-4],[3,-9,-2,-7,-7,-8,-9,9,8,8,10,-5,-1,-10,7,8],[-2,10,-7,5,-5,8,-9,-4,5,-2,-10,-1,-7,2,9,-3],[-6,-2,-1,-6,1,-10,2,-9,10,1,10,2,5,1,-5,-4],[1,-5,-5,9,-4,-8,5,-1,-1,-6,6,-8,4,-4,8,-2],[-6,-1,8,8,-9,8,6,-10,4,7,3,-1,-9,-8,5,-5],[-6,3,-6,-8,9,-8,5,-3,-1,-7,8,-1,-2,-5,10,-7],[6,-2,3,7,4,-9,3,-8,-2,1,4,8,-3,-3,3,-1],[2,-1,-1,-8,-6,-6,9,-7,10,1,-2,-9,2,-9,-6,3],[-2,-4,5,-10,7,1,-7,-6,-4,10,-8,2,3,-2,-2,6],[9,6,1,4,1,5,2,6,10,4,5,5,-9,-8,2,-2],[-5,7,-6,4,-6,5,10,-8,7,6,9,-1,-10,7,4,-5],[10,5,-4,-3,-7,10,10,-8,9,-9,5,7,9,-9,5,-2],[-7,1,5,4,2,9,-2,-9,1,1,7,3,-4,-1,-6,4],[-9,7,-10,-7,-4,6,-3,1,2,10,-2,5,5,6,8,-7],[6,-2,4,3,-6,-6,-4,3,-10,5,-7,-9,6,-4,-5,3],[4,10,-5,-5,-3,-1,9,3,8,3,-8,-8,6,-1,2,4],[1,3,8,8,1,8,-3,1,-7,-9,-7,-9,10,5,6,-9],[5,5,-1,7,-2,4,-6,-1,9,9,-6,8,2,7,5,5],[9,5,2,-2,9,-4,-10,1,2,3,2,-3,7,10,-7,3],[-6,-3,-2,-5,-10,-9,1,-8,-10,2,-4,-6,-4,-2,3,4],[-4,8,-2,9,1,2,6,6,1,-9,10,6,2,-4,3,-4],[6,3,4,-2,9,-8,6,-2,9,-1,-9,-4,7,7,7,7],[5,3,-1,6,-1,-5,-8,3,2,-6,-5,3,-10,1,4,-3],[9,10,-5,2,-3,-4,-7,6,-4,-10,7,-10,4,6,-3,-10],[-9,4,9,1,-4,10,-1,-9,-3,4,2,6,-9,-7,3,5],[-8,-5,7,-6,10,-2,-4,-6,4,6,-5,-4,-5,7,8,10],[-2,-2,1,-2,6,2,9,-9,-1,8,7,2,-5,-2,2,-1],[-8,-3,-5,-2,1,2,-3,10,8,-1,2,-9,-4,-1,4,1],[9,3,-1,4,8,-1,-9,-6,-10,3,10,-10,4,1,5,2],[-5,5,-7,-9,7,7,-2,-4,-7,6,-1,-4,-8,-5,-6,-7],[1,9,1,4,-8,3,-1,4,-3,9,-7,-9,7,-9,2,-4],[-10,-2,8,8,-8,-6,9,8,10,-6,-2,3,-2,10,10,8],[6,-10,7,-10,2,-3,-8,3,-2,-5,7,-2,4,-2,8,6],[-1,10,-8,-4,-8,1,9,4,-6,-4,10,-1,-9,-3,-4,-5],[-7,4,7,6,-4,3,-3,-5,-7,2,-6,-4,-8,6,-5,-9],[-5,-10,3,-8,-2,-2,-8,-6,7,-4,10,2,4,2,-8,-4],[5,5,-9,-5,-2,10,4,-6,-3,-8,8,-1,-7,-6,3,9],[-3,4,-7,-4,5,-10,-9,2,7,-3,9,9,-6,-4,-4,-6],[7,-10,-10,5,8,4,2,-9,5,5,-9,-9,-9,5,6,-10],[-3,9,-5,-7,-10,4,7,10,-7,-6,-9,-1,-10,2,8,-10],[2,-2,-4,4,2,-5,-1,-10,1,-3,1,-9,-2,8,2,-1],[-7,-3,-4,2,-5,4,3,-2,-10,-5,3,-3,-4,5,8,2],[10,6,-4,-1,-7,-3,-1,-5,4,-3,3,-1,8,6,-3,3],[4,9,-10,-7,7,7,6,10,-1,6,8,2,-3,-4,-7,-10],[-8,10,-1,-5,8,-8,-4,-4,-1,9,-8,-7,-1,7,-10,2],[-6,-6,-1,9,2,2,8,-1,8,10,4,7,8,5,1,-6],[10,-7,-2,8,6,10,-8,2,-2,2,-10,1,-2,-8,4,3],[-7,-4,-7,-4,4,6,6,10,2,4,-2,2,3,-9,7,2],[-2,-4,9,-9,-4,7,-5,-8,8,9,-10,5,2,-2,-6,4],[9,5,7,-10,-1,1,-3,-4,-4,2,5,-7,1,-5,-2,-3],[6,8,3,4,-7,3,5,-3,-10,3,2,2,9,-10,-6,-3],[-2,6,1,3,-9,1,4,5,1,-9,9,-8,-7,-1,-3,5],[-5,3,5,7,-2,9,2,-10,-1,-8,10,-1,4,10,-4,3],[10,-9,-1,10,-5,-1,-3,2,2,6,-7,9,-7,-1,7,-4],[1,3,-8,-7,-4,-9,-1,-10,-2,3,3,3,2,-3,-7,6],[-10,9,-4,2,7,4,4,-3,2,6,-9,-5,2,-9,-7,-6],[-6,5,-5,9,-10,-2,-8,-9,-10,-6,-5,-9,1,3,-5,-6],[-5,-2,-10,2,3,-4,-8,1,-9,8,-10,-8,-7,10,3,-2],[9,-9,-4,-10,-7,-9,-7,-7,-7,-10,-2,-7,-1,-10,5,3],[3,-4,4,-5,8,8,-4,10,9,-8,10,1,-4,-6,-2,-4],[2,3,4,9,-2,5,-3,7,7,-5,3,-10,9,-6,6,6],[-10,4,-7,-3,-1,6,5,5,1,1,-8,-10,9,-5,-6,3],[-2,-9,9,1,-2,3,10,-3,8,2,-5,2,-8,8,-2,1],[7,4,-4,7,1,5,-8,2,-2,-3,-9,-2,-3,-5,1,-8],[8,6,8,9,-3,-3,-3,-6,-1,4,3,-9,4,9,1,-4],[1,2,3,-9,8,-5,8,7,-2,9,5,-2,-3,-6,10,5],[2,-3,3,-9,-5,-3,10,10,9,7,-1,-4,-4,1,2,7],[7,7,3,2,-2,9,-3,4,-2,7,-4,10,-3,1,-8,9],[10,7,4,1,2,-8,3,-8,-4,8,6,-5,-10,-2,7,-6],[-9,-10,4,-1,-3,3,2,1,-6,6,-9,-9,-7,9,-4,3],[-10,8,-6,2,7,-5,7,-7,-7,3,1,-4,4,-4,-8,4],[10,2,1,7,-9,7,8,-9,4,-10,-7,-5,8,1,-4,4],[9,2,-1,-7,1,8,-9,-4,-2,-10,-1,-10,-4,-4,8,-4],[5,-8,-5,9,-3,5,8,5,-6,-7,-9,5,-9,-2,7,7],[-2,-10,-2,-2,8,8,-9,-5,7,-1,10,10,2,-8,7,-1],[-2,5,10,10,-5,5,1,-8,-2,5,-4,3,6,-7,9,-2],[-8,7,-6,-9,5,-6,4,5,2,9,-4,4,-6,-4,8,7],[7,-6,-1,6,9,-2,-8,8,-6,8,1,-3,2,3,-5,7],[-6,-7,-10,-3,-6,1,-4,6,7,-4,-5,-5,-6,10,-1,-5],[2,6,-4,-2,-5,-4,3,1,-1,8,1,6,4,-1,5,-3],[-8,10,9,-4,-8,-3,6,10,-10,7,4,1,1,4,-9,-2],[10,-1,-5,-3,-3,3,2,-9,4,4,-1,-7,-8,-6,-1,-4],[-10,8,-9,-8,9,4,-5,-4,8,8,-4,2,5,8,8,-7],[-6,4,7,-10,3,8,6,6,3,-5,-6,-8,5,-9,8,1],[-2,-1,-9,5,-6,10,8,-8,-10,7,-10,-9,1,-7,-5,-2],[7,7,-8,-8,9,8,5,-8,9,-1,10,-1,9,9,10,2],[-7,-3,-3,-1,2,10,5,10,-1,10,-10,-2,3,-2,5,10],[3,8,8,-8,9,6,6,-6,7,-8,-9,4,4,-2,-2,-2],[8,2,-9,6,10,10,8,-3,-1,9,-3,-1,1,-7,-8,7],[-7,3,1,4,3,-5,1,3,-1,4,-7,1,10,2,-7,7],[-10,-1,-2,-2,8,-4,-7,1,-3,-2,-9,10,-10,-9,-3,-2],[7,-4,8,-3,-3,-5,6,-8,2,9,2,-3,4,-1,10,-10],[7,-5,-1,-1,-5,-7,-4,-10,-5,-2,-2,9,-8,-3,-6,-5],[-8,-9,-9,3,9,-5,-5,3,-5,4,7,3,8,4,-10,8],[-9,7,7,2,-8,-1,3,2,7,7,-1,-1,-8,-2,4,-4],[-5,1,-1,-10,-1,3,-6,-3,-10,9,3,-8,-7,-7,10,-1],[4,-9,-4,-5,3,-10,10,-2,-4,-2,2,-7,-2,-7,-8,7],[2,-7,6,-3,8,8,7,-10,10,9,2,1,10,4,10,-8],[-9,3,4,6,9,2,3,-8,-4,7,3,-9,9,9,-7,4],[-3,-5,2,1,-7,-9,4,5,4,-3,-3,-10,-6,-10,9,10],[10,8,-7,5,-3,4,2,5,-3,-5,-1,-7,-7,9,-2,7],[-4,-5,-8,-3,-5,-8,6,2,-4,-6,-5,1,3,-7,10,-8],[-3,-5,3,1,5,-4,-5,-10,5,-3,2,-2,7,-9,-7,1],[-2,-8,4,-5,-9,10,5,-2,-10,3,-10,-7,6,3,-5,-3],[1,-5,3,4,7,5,-5,-4,-1,-7,4,7,9,5,-4,-2],[2,-10,7,4,-7,8,5,9,3,-8,-9,8,6,-4,2,5],[-5,-6,8,-6,10,-9,10,-4,5,9,-8,3,-8,6,5,5],[-8,-10,-2,5,2,10,3,4,-7,-1,-8,1,-1,8,-10,-6],[4,7,4,-2,6,-10,-6,-7,5,-10,-4,3,-7,1,-3,-6],[-3,-7,-4,3,10,7,8,-1,-3,10,-1,3,3,7,-1,5],[7,8,8,9,7,-4,-1,-9,-5,2,-9,5,-8,-8,-5,-10],[-4,2,-4,-1,-9,5,1,4,3,1,-10,-6,9,7,8,6],[8,5,1,2,-3,1,2,-10,-3,9,-7,4,-10,10,8,7],[8,-9,-9,10,-1,4,8,10,-4,7,-5,-9,-9,10,-9,-7],[-10,7,-9,-9,-9,4,8,-1,3,8,-6,-3,-8,-6,2,3],[10,-9,-8,-1,-5,6,-1,8,2,3,4,3,-9,-6,-2,2],[8,5,4,-3,-3,2,6,7,7,4,-6,-3,7,7,-8,-3],[4,4,-4,-7,-8,10,-1,9,-6,1,-1,-8,-4,8,-8,2],[-7,-10,-5,-7,-5,6,3,-6,-8,1,-3,-10,8,1,-8,-7],[-9,-5,3,-5,10,-6,7,-2,-3,8,5,-3,5,8,4,8],[2,8,-7,6,6,-2,6,-10,10,4,-6,-6,-8,-2,-10,10],[4,-7,-10,-10,-5,-1,-8,-9,-9,-4,8,3,6,2,8,6],[-9,4,-6,-8,-7,-1,10,-3,-10,1,7,-8,1,-9,-4,-10],[-9,-4,9,3,3,-10,3,9,2,-5,-5,-2,-5,-7,5,3],[2,-9,1,6,6,-6,-1,4,6,4,2,-10,-6,-7,-4,5],[10,-6,5,-7,-10,-2,8,6,-5,5,-6,-6,4,3,6,6],[1,-10,-6,9,-10,10,-7,1,-7,-9,3,-3,-9,-7,-4,2],[8,2,-9,-1,2,-4,-8,-10,3,-7,-7,7,9,-2,8,-8],[7,10,3,-2,3,-5,-5,5,8,-6,4,-10,9,6,7,-4],[6,-10,-9,5,2,-4,-6,-5,-1,-9,4,8,3,-1,-7,-7],[-2,5,9,3,1,8,-6,-2,-4,4,2,3,5,6,-10,-9],[9,-4,-4,-2,4,-2,3,-9,-7,-6,6,-4,3,5,2,8],[5,-4,10,-4,5,7,10,-6,-1,-5,2,-8,-9,10,6,-5],[-3,2,4,-6,-1,-9,-8,-9,-5,-8,4,-2,10,4,-4,-8],[7,-8,5,-10,-8,-9,-1,-7,-3,-3,-2,7,9,-7,2,7],[7,-8,1,1,6,8,-6,-6,-2,5,-4,-6,-6,5,2,7],[9,-2,-5,2,2,-9,-4,9,-3,-10,7,-3,10,-10,2,-2],[3,3,-8,9,4,-8,4,6,2,-6,3,3,-7,-8,10,-2],[-5,-3,2,2,-2,6,-10,1,10,10,-7,9,-5,3,-8,-10],[-3,-5,-7,-3,-9,-2,3,10,3,7,2,3,2,3,-2,2],[-10,-2,-5,-5,-6,-7,10,-2,-7,2,6,7,3,9,-6,3],[-9,-2,-10,3,4,4,-5,-9,-5,4,-6,1,-2,7,1,-6],[4,5,7,-9,7,-6,-8,-7,-6,6,1,-3,8,-4,9,9],[10,-7,-3,-7,-8,3,10,2,-10,-9,-6,7,-4,4,1,-9],[9,-7,2,3,10,7,5,-5,-8,7,1,-9,-4,-10,-6,7],[7,-8,-9,1,6,-9,-9,-5,3,2,-5,-10,-6,6,-1,9],[-7,7,9,-7,-7,-7,5,3,5,3,-8,9,-7,-6,-5,8],[-4,-10,-2,2,-5,6,-1,7,4,-6,-5,4,1,10,-7,-10],[3,7,-6,7,-8,-9,-1,6,8,-6,8,-5,-2,-5,-3,-3],[4,6,-9,-1,-7,5,-9,-5,-3,-10,-3,-2,-9,-5,3,-1],[-7,8,1,10,-2,-7,-8,-7,-3,3,-1,-2,-1,1,4,2],[-10,-7,10,2,-4,4,-1,-2,10,-4,-10,9,3,5,3,-3],[10,5,-9,3,-6,-5,9,-5,5,-8,-6,-6,7,8,-1,-5],[1,10,-7,-5,-9,1,-10,-5,-6,9,9,10,-9,6,-1,10],[10,-4,-2,-1,-9,-10,-1,-5,-7,-10,10,-2,2,10,10,-7],[-9,-9,-10,-3,-5,10,-10,-6,9,-3,7,9,-7,3,8,-8],[-1,-9,1,1,-7,-3,-6,-3,-6,6,-6,-3,4,10,5,1],[7,8,2,-4,-10,5,8,1,-6,7,1,-3,-3,-9,1,6],[-7,-7,-2,1,10,6,-6,3,-3,2,-8,1,9,2,9,9],[-2,7,7,-6,6,4,10,-3,-8,8,-1,-9,-10,-10,1,-5],[6,-7,-1,-10,4,8,3,-6,6,7,7,-8,1,4,-9,-6],[-2,-8,3,-10,5,-10,9,9,4,-3,-6,-5,-10,8,7,-4],[7,2,3,-9,-1,-5,9,-8,-4,5,3,-4,1,2,-7,7],[-10,6,-2,-2,-7,5,-2,10,-3,5,-5,9,-2,1,1,7],[-10,7,-9,-7,5,-2,-3,5,5,2,-9,-3,-5,-2,6,-8],[-2,10,-1,8,7,6,-2,7,-9,9,-1,-8,-1,-1,-2,5],[3,7,-2,-1,-7,-10,8,-8,8,4,2,-4,-6,1,-5,4],[-2,3,8,-5,-2,5,4,-6,3,-2,4,7,2,-10,4,4],[-8,10,-2,2,9,2,1,6,10,8,-5,-5,-2,-5,-2,-2],[7,-6,-6,10,6,-1,-2,10,4,-7,4,7,1,-4,-2,-3],[10,-5,-8,4,-1,-2,1,-6,-1,-5,6,-2,-3,-2,-5,-3],[4,3,-6,-7,-1,9,-10,-6,-7,5,2,-2,4,-1,9,-5],[-5,9,7,-8,5,-8,-2,8,-7,7,-10,6,5,4,5,-6],[-10,-2,1,3,7,4,10,-3,3,-6,-8,-7,-4,-6,-2,7],[-9,9,9,7,5,-5,1,2,7,9,2,10,-8,-5,1,2],[-3,10,-5,7,-10,7,-2,9,6,-9,-3,6,-5,4,3,6],[9,7,4,-5,6,-8,-10,-10,-5,9,-7,-1,-4,-8,3,2],[-9,5,2,-8,2,9,-2,-2,-9,5,4,8,-1,6,10,5],[6,3,-9,-6,-6,5,-5,5,8,10,5,3,-6,-9,6,6],[-3,-9,9,-3,2,2,-7,-3,-8,8,-3,-9,-9,-5,-2,4],[-9,8,-8,-2,-1,8,-3,4,4,-5,2,-5,-4,-3,-3,-4],[1,8,-6,-10,5,6,10,9,1,-2,10,5,10,-2,6,5],[9,7,-4,-4,-6,8,10,2,3,-6,1,-9,-9,6,-9,3],[2,5,8,8,-8,1,-6,-1,5,4,-7,9,-5,1,3,2],[-5,-5,4,8,10,2,-10,-8,8,-3,6,3,6,-3,-10,-6],[2,8,5,-2,5,2,6,-2,-4,1,6,1,-2,3,-1,-8],[10,-1,-4,-1,-4,4,-5,-6,3,8,3,-1,4,-8,-3,-8],[-4,10,-1,-7,-4,-6,-9,-7,-10,-2,-4,10,9,5,-4,-6],[1,8,-7,-8,8,-9,4,1,-2,-7,-10,-6,-4,-2,5,9],[-6,-9,-2,3,1,-10,9,-1,6,9,4,4,8,8,-8,9],[-4,6,-7,9,-10,-5,10,-9,-5,-2,-4,6,-4,5,-1,-5],[-5,2,-9,-2,-10,9,-4,-6,-9,-7,-5,-6,7,-1,6,-5],[5,4,10,5,2,10,2,-4,-4,-5,2,-8,-1,-9,-2,-9],[-4,3,9,-4,9,9,-1,4,2,4,9,4,8,-8,-10,7],[1,9,-7,-5,8,10,6,7,-3,3,3,-3,-3,-1,-9,-8],[-10,2,-2,6,10,-3,9,-4,1,6,5,2,10,2,4,2],[10,1,-3,-4,-2,1,7,6,5,9,-1,9,-4,1,-2,2],[-6,-4,8,4,-4,-1,-10,-6,-9,4,-6,1,3,-8,4,-9],[-10,1,-10,10,-8,7,-8,-5,-10,8,2,-5,-1,2,-2,-8],[-3,-2,4,10,-3,-1,2,9,-5,-7,4,1,-7,-3,-5,1],[9,-4,-5,4,3,-10,-1,6,-6,2,9,7,8,4,2,5],[3,4,-4,-8,8,-2,7,-5,-3,4,2,-7,-5,8,-10,-7],[-10,1,-3,-1,-6,-10,-10,-9,3,6,-7,8,6,9,8,-1],[9,2,-2,3,-5,3,9,8,-7,7,-7,-8,-10,-1,3,7],[-5,-7,4,-4,-2,-4,-1,1,9,4,5,-1,8,-5,5,4],[3,-6,1,5,9,4,-4,-7,-5,8,-2,-6,10,-1,3,6],[7,2,2,-3,9,-8,-2,-3,-6,8,-8,6,8,-7,-5,9],[-5,-1,-10,-5,-2,9,9,-3,-3,-3,-7,-2,-8,-6,-2,-6],[7,1,10,6,10,-5,9,6,-1,4,-1,3,-3,-4,-3,1],[6,7,9,-9,9,10,4,-10,4,-2,-10,6,-5,10,9,-10],[-3,-2,-2,-1,-6,1,-6,6,-4,-7,-5,-10,1,8,1,-4],[7,-7,6,-10,-9,-2,-1,-7,7,1,-4,2,-9,7,10,-4],[-9,4,1,10,4,-4,-2,-1,-4,-3,-3,6,-9,4,6,6],[9,3,-5,-9,-6,-5,-4,7,9,6,7,4,5,-5,5,6],[-3,-3,3,9,-7,-8,-1,10,-7,1,-10,-1,-4,8,8,-5],[-3,9,-10,-5,-3,7,8,-5,6,10,-3,-1,-7,-8,2,-6],[-2,3,3,9,1,-1,5,-7,-2,-7,7,-3,6,1,-5,-5],[2,-6,-7,3,5,10,-8,3,3,-6,-9,-5,10,-4,-3,-9],[8,3,4,-5,10,9,2,9,-1,6,-3,6,8,-8,1,-7],[-5,-6,6,6,-9,-3,8,-6,8,10,4,-5,-5,3,-10,-5],[-9,-4,4,-1,6,6,8,2,6,7,7,2,7,-8,-6,-2],[5,1,7,-4,8,6,2,1,-1,9,4,-9,-9,-8,1,-1],[7,6,4,8,10,7,3,1,10,-3,1,5,-9,4,-4,-6],[9,-4,2,6,-5,5,4,-3,5,-7,8,-8,-5,-6,-7,10],[2,10,8,-7,8,-3,7,-7,2,3,-7,6,3,-1,-2,-8],[-6,2,4,-9,1,-5,9,-5,7,8,6,6,-6,2,9,-4],[-8,-1,6,-5,5,9,3,3,-1,3,-7,-9,-7,2,-10,-1],[3,-1,5,9,-7,1,2,9,9,1,3,10,10,4,-10,-10],[-10,-3,-8,-4,-1,-4,-1,4,-9,7,-6,1,-10,-8,1,-3],[-9,-7,-1,2,6,-4,-3,8,-5,1,1,-7,-7,1,-3,3],[1,8,9,-1,-8,-3,-3,-10,-5,10,-4,-6,2,-3,10,-6],[3,2,-4,8,-9,-2,1,5,-6,1,3,-3,7,-9,-1,9],[-9,-10,7,6,1,2,-1,-10,-9,-6,-10,-7,-6,-3,3,3],[-4,-6,10,-9,-9,5,-6,-6,-7,9,-3,6,5,8,-3,2],[3,8,-2,-8,-8,1,-9,-8,1,-4,6,-6,9,-10,5,2],[2,5,7,-4,-6,8,4,-2,5,-1,-5,-1,1,9,-5,-5],[-10,-8,-4,3,-7,-5,-9,10,-2,7,10,-4,-9,-8,-5,-9],[3,-6,-2,-7,2,-2,7,-6,-2,8,1,2,6,-6,5,8],[5,-5,2,5,-3,6,-10,-1,-7,6,-3,-10,10,-2,1,1],[3,-4,9,9,-8,-4,-9,-3,-5,2,7,7,-8,-10,-8,-7],[-10,-3,10,-3,-9,9,3,2,-10,6,-8,3,-1,-7,8,7]], dtype = "uint8")#candidate|1491|(720, 16)|const|uint8
var_1492 = relay.var("var_1492", dtype = "float32", shape = (2304,))#candidate|1492|(2304,)|var|float32
call_1489 = relay.TupleGetItem(func_570_call(relay.reshape(var_1490.astype('uint32'), [12, 10, 3]), relay.reshape(const_1491.astype('uint8'), [10, 1152]), relay.reshape(var_1492.astype('float32'), [2, 1152]), ), 4)
call_1493 = relay.TupleGetItem(func_575_call(relay.reshape(var_1490.astype('uint32'), [12, 10, 3]), relay.reshape(const_1491.astype('uint8'), [10, 1152]), relay.reshape(var_1492.astype('float32'), [2, 1152]), ), 4)
bop_1510 = relay.not_equal(const_1475.astype('bool'), relay.reshape(bop_1481.astype('bool'), relay.shape_of(const_1475))) # shape=(5, 12, 16)
uop_1519 = relay.cosh(const_1475.astype('float32')) # shape=(5, 12, 16)
bop_1529 = relay.greater(uop_1519.astype('bool'), relay.reshape(const_1476.astype('bool'), relay.shape_of(uop_1519))) # shape=(5, 12, 16)
uop_1535 = relay.acos(bop_1529.astype('float32')) # shape=(5, 12, 16)
func_836_call = mod.get_global_var('func_836')
func_840_call = mutated_mod.get_global_var('func_840')
const_1539 = relay.const([-4.948015,2.071379,0.988224,-3.535994,1.434334,-6.031014,0.917728,-3.901662,1.227225,5.677134,-2.650051,-1.849621,5.344419,-9.118511,3.444933,6.453122,5.098859,-6.492559,-7.088933,8.150706,-7.066182,-5.982269,-6.123545,6.439285,3.795048,-8.468375,-6.181259,7.884025,-7.704480,-6.943903,-4.075462,8.735594,-3.812858,-8.793354,-7.464317,-6.360522,8.532637,-7.636569,7.551946,-8.359458,4.618764,-1.199264,6.329832,4.375709,-1.161981,7.182822,2.815673,-5.969003,2.949238,-5.378259,-6.206148,-9.728883,-2.957258,-0.505870,-7.768566,-7.291188,7.043427,1.001870,-5.388445,-0.634624,-0.079551,0.465418,8.058397,2.737957,-2.825165,-6.707171,-6.745802,0.773466,-2.281644,4.996056,-3.351095,-2.245918,-1.384847,-2.022359,-6.248284,6.871757,1.333195,-5.933207,7.227692,1.800717,5.512943,-4.770439,-3.995085,-9.373761,1.003274,-8.254323,-9.596141,-3.729517,-6.583679,1.410462,2.269042,-9.094509,-3.477777,0.067192,-7.137214,9.276619,1.194987,-4.499778,7.536872,-5.062256,-3.495692,-9.376212,0.364980,1.038008,-8.418141,-8.178920,-5.437394,-9.499086,4.170843,-3.675106,-2.492416,3.552286,2.060638,7.855229,5.605201,-7.429413,-6.927732,-0.076564,8.933880,-6.559295,-7.320546,3.404548,2.950400,7.594062,-7.055825,-0.454416,-9.115846,-0.043121,2.247802,-1.298975,2.507770,-0.887051,-0.188197,3.662528,8.687975,2.358708,-0.278121,-0.988410,-3.080381,-9.122814,5.070559,0.236208,-4.044856,1.849994,5.342259,-2.592498,-9.441162,7.810283,7.870110,-6.341715,-2.496831,7.682748,8.785432,-4.028418,9.688226,-6.357954,7.563018,-7.218638,6.084139,-5.629540,-3.621127,8.618073,4.419474,3.650540,0.412294,3.561970,3.624346,-2.661380,-0.786662,-9.726373,7.598740,-9.279158,-6.806621,-5.758200,4.672071,-3.488127,-8.499166,-5.182871,-4.277536,-3.964342,-1.584050,6.311947,-2.086745,-1.212788,-9.553935,4.637963,7.606706,3.276774,-6.532983,2.411693,-4.068484,0.024653,9.418702,-3.203379,4.965644,2.384360,6.021350,7.419479,-7.889250,-1.773747,-7.565752,-4.955585,-2.213410,3.572595,2.502307,-3.713907,5.345944,-6.577404,4.105741,-7.721448,-7.302933,-9.165290,6.950068,3.738428,0.084745,-0.720935,4.308838,2.494901,-1.593842,3.959176,3.713739,4.435814,8.901677,8.561929,9.091221,-0.179540,7.991124,1.746452,-9.788935,-0.289778,-4.484880,-2.040245,7.517391,2.104028,-5.523725,-2.297742,-0.306459,-9.511481,5.518470,9.147277,-5.835866,-5.822119,-1.967402,-8.824379,5.356534,1.405979,5.245738,9.292117,9.644351,2.740169,9.542678,-7.784570,8.932558,-8.687857,-4.359145,2.251379,4.809582,2.117072,8.527297,-8.591782,-3.147175,3.765933,3.947025,1.073854,2.735473,3.095171,-8.557693,2.353207,-0.626777,-7.367114,7.071324,8.398116,-8.788669,-1.617021,-8.702161,-5.749102,-2.768191,8.756801,-4.126611,3.878084,3.361543,3.930051,-7.381386,2.342942,-7.831138,6.566954,-1.531310,-0.544588,5.889426,-6.014430,2.822627,9.006759,-6.310632,-2.827413,2.811151,9.045902,-7.714445,-1.499228,1.948103,4.362773,5.265750,1.908622,-1.984915,-7.369320,-3.982278,1.235502,0.114465,-0.559655,2.097919,2.861907,-9.792861,5.671700,0.571742,5.965752,-6.790512,-6.287154,9.406097,8.868191,2.315848,-2.928031,-8.077147,1.060376,-4.017562,-8.696071,-5.550872,-3.524276,-4.902425,0.787644,4.954253,-0.890653,-4.643251,4.489009,-1.220829,9.871757,6.090550,7.842639,8.047167,0.278214,-8.141970,-2.951646,9.380208,3.944433,-3.992462,4.329643,1.574594,1.729625,-9.619870,-9.684210,7.788696,-7.753788,2.378547,5.587066,1.676612,-0.982570,-6.892577,-7.411258,4.253213,9.890325,9.626100,6.545209,-2.672131,-4.138410,2.329582,6.131486,-2.769730,6.848757,2.949503,-6.565315,-3.523868,3.152130,4.544540,-7.479659,4.206421,7.537142,0.217327,-9.638917,1.253247,-4.566984,2.990878,-7.900711,-1.243249,4.056122,-0.646603,-0.277329,-0.165839,2.678633,8.731686,-4.188526,5.323197,0.340433,-2.613161,-5.321016,-5.494075,-1.285982,-5.024960,7.949365,0.322546,-7.289958,2.636300,-1.669501,-2.195417,8.447640,9.331546,3.426930,5.594842,8.952547,-3.021097,-4.841265,2.283214,-1.823644,2.064662,-7.101098,-1.980519,-0.996109,1.426627,4.584508,-6.333595,7.596807,-6.763962,-8.526212,-3.137747,7.327657,9.182205,5.677442,-4.315121,-5.268648,-1.340927,-8.140311,5.020033,-3.060008,0.941923,5.685252,-2.951430,-2.658937,6.629832,9.139434,-8.940369,-2.242295,-4.245682,-0.792596,3.957165,-8.311847,5.804515,-9.958342,3.337782,0.826591,-1.371522,-8.850832,-1.974171,3.459774,2.341302,3.246419,4.302984,5.367865,9.696286,-2.466555,-7.493737,-5.387891,-3.909508,-6.438505,3.282083,-2.217552,-3.661194,4.304765,9.879436,-8.048868,9.342338,5.863840,4.035742,-2.455242,-7.969507,2.698730,1.368167,7.892628,-6.815110,7.100857,7.784962,4.205767,-3.222065,-6.001460,0.920336,-5.757189,-2.921852,9.246420,-7.040755,5.931522,8.135688,5.624370,-0.425357,-6.632317,-0.016553,6.965106,6.134089,-3.174651,8.244650,-8.393721,5.046775,6.161811,-4.441048,-2.740469,-6.901334,5.766397,-9.423029,-5.846093,8.158158,-8.888677,-8.651342,-2.091814,-9.163352,0.054469,6.677804,9.228743,-9.280346,5.262218,9.639583,8.072810,7.614826,9.727308,9.204232,-3.767819,-8.100893,-5.103178,-9.762040,7.228494,4.050397,-0.549213,-8.731359,-7.813309,-3.917823,7.207493,1.616521,9.117619,6.910184,-8.512705,-1.912697,-4.931919,7.198171,2.952991,7.780623,5.331955,5.924754,5.644527,9.412076,-8.599346,9.942477,6.616033,-3.657529,1.645128,-9.555998,-7.644238,2.155764,2.179886,-9.854619,-2.141561,5.702961,-2.478644,-3.145163,-2.812701,0.061117,1.245041,-6.339465,8.542412,-3.547025,7.994920,9.293019,6.036311,-9.128190,6.104147,-1.064321,-2.162482,2.480414,-0.631083,-5.151234,7.799098,-6.540580,7.190473,-9.951016,4.748043,2.377720,1.212589,6.444602,8.678524,1.549828,-8.257933,5.311085,-8.839879,2.702569,-6.156138,7.923909,-2.900687,-2.156989,9.874590,-5.658699,5.364103,5.028697,-5.736373,-9.778713,-8.578336,1.482460,4.642096], dtype = "float64")#candidate|1539|(600,)|const|float64
call_1538 = relay.TupleGetItem(func_836_call(relay.reshape(const_1539.astype('float64'), [40, 15]), relay.reshape(const_1539.astype('float64'), [40, 15]), ), 0)
call_1540 = relay.TupleGetItem(func_840_call(relay.reshape(const_1539.astype('float64'), [40, 15]), relay.reshape(const_1539.astype('float64'), [40, 15]), ), 0)
output = relay.Tuple([call_1484,var_1485,call_1489,var_1490,const_1491,var_1492,bop_1510,uop_1535,call_1538,const_1539,])
output2 = relay.Tuple([call_1486,var_1485,call_1493,var_1490,const_1491,var_1492,bop_1510,uop_1535,call_1540,const_1539,])
func_1543 = relay.Function([var_1485,var_1490,var_1492,], output)
mod['func_1543'] = func_1543
mod = relay.transform.InferType()(mod)
mutated_mod['func_1543'] = func_1543
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1543_call = mutated_mod.get_global_var('func_1543')
var_1545 = relay.var("var_1545", dtype = "float64", shape = (154,))#candidate|1545|(154,)|var|float64
var_1546 = relay.var("var_1546", dtype = "uint32", shape = (12, 30))#candidate|1546|(12, 30)|var|uint32
var_1547 = relay.var("var_1547", dtype = "float32", shape = (2304,))#candidate|1547|(2304,)|var|float32
call_1544 = func_1543_call(var_1545,var_1546,var_1547,)
output = call_1544
func_1548 = relay.Function([var_1545,var_1546,var_1547,], output)
mutated_mod['func_1548'] = func_1548
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1196_call = mod.get_global_var('func_1196')
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1561 = func_1196_call()
call_1562 = func_1196_call()
uop_1565 = relay.log2(call_1561.astype('float64')) # shape=(8, 8, 3)
uop_1567 = relay.log2(call_1562.astype('float64')) # shape=(8, 8, 3)
uop_1571 = relay.exp(uop_1565.astype('float32')) # shape=(8, 8, 3)
uop_1573 = relay.exp(uop_1567.astype('float32')) # shape=(8, 8, 3)
func_293_call = mod.get_global_var('func_293')
func_294_call = mutated_mod.get_global_var('func_294')
call_1574 = relay.TupleGetItem(func_293_call(), 0)
call_1575 = relay.TupleGetItem(func_294_call(), 0)
uop_1583 = relay.erf(uop_1565.astype('float64')) # shape=(8, 8, 3)
uop_1585 = relay.erf(uop_1567.astype('float64')) # shape=(8, 8, 3)
output = relay.Tuple([uop_1571,call_1574,uop_1583,])
output2 = relay.Tuple([uop_1573,call_1575,uop_1585,])
func_1587 = relay.Function([], output)
mod['func_1587'] = func_1587
mod = relay.transform.InferType()(mod)
output = func_1587()
func_1588 = relay.Function([], output)
mutated_mod['func_1588'] = func_1588
mutated_mod = relay.transform.InferType()(mutated_mod)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1596 = relay.TupleGetItem(func_865_call(), 0)
call_1597 = relay.TupleGetItem(func_866_call(), 0)
func_668_call = mod.get_global_var('func_668')
func_671_call = mutated_mod.get_global_var('func_671')
var_1602 = relay.var("var_1602", dtype = "float64", shape = (112,))#candidate|1602|(112,)|var|float64
call_1601 = relay.TupleGetItem(func_668_call(relay.reshape(var_1602.astype('float64'), [7, 2, 8]), relay.reshape(var_1602.astype('float64'), [7, 2, 8]), ), 0)
call_1603 = relay.TupleGetItem(func_671_call(relay.reshape(var_1602.astype('float64'), [7, 2, 8]), relay.reshape(var_1602.astype('float64'), [7, 2, 8]), ), 0)
output = relay.Tuple([call_1596,call_1601,var_1602,])
output2 = relay.Tuple([call_1597,call_1603,var_1602,])
func_1604 = relay.Function([var_1602,], output)
mod['func_1604'] = func_1604
mod = relay.transform.InferType()(mod)
var_1605 = relay.var("var_1605", dtype = "float64", shape = (112,))#candidate|1605|(112,)|var|float64
output = func_1604(var_1605)
func_1606 = relay.Function([var_1605], output)
mutated_mod['func_1606'] = func_1606
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1196_call = mod.get_global_var('func_1196')
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1644 = func_1196_call()
call_1645 = func_1196_call()
output = call_1644
output2 = call_1645
func_1654 = relay.Function([], output)
mod['func_1654'] = func_1654
mod = relay.transform.InferType()(mod)
mutated_mod['func_1654'] = func_1654
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1654_call = mutated_mod.get_global_var('func_1654')
call_1655 = func_1654_call()
output = call_1655
func_1656 = relay.Function([], output)
mutated_mod['func_1656'] = func_1656
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1726 = relay.const(6, dtype = "uint32")#candidate|1726|()|const|uint32
var_1727 = relay.var("var_1727", dtype = "uint32", shape = (12, 3, 3))#candidate|1727|(12, 3, 3)|var|uint32
bop_1728 = relay.greater_equal(const_1726.astype('bool'), var_1727.astype('bool')) # shape=(12, 3, 3)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1731 = relay.TupleGetItem(func_865_call(), 0)
call_1732 = relay.TupleGetItem(func_866_call(), 0)
func_1105_call = mod.get_global_var('func_1105')
func_1110_call = mutated_mod.get_global_var('func_1110')
var_1734 = relay.var("var_1734", dtype = "uint8", shape = (1, 3))#candidate|1734|(1, 3)|var|uint8
const_1735 = relay.const([-6.807736,3.913903,1.950468,3.708470,7.928666,2.663060,-3.273349,7.812733,7.429538,2.857893,-2.287123,2.883362,-0.606101,5.605518,5.640772,7.774370,-8.903427,-6.779249,-3.651001,6.051689,-3.395507,-0.738652,-3.850991,-0.391893,5.863624,-8.917989,9.171472,7.908199,4.629333,2.546456,-6.207135,1.315404,0.378956,-7.961841,4.680341,8.864017,6.473733,1.466485,2.784837,9.356014], dtype = "float64")#candidate|1735|(40,)|const|float64
call_1733 = relay.TupleGetItem(func_1105_call(relay.reshape(var_1734.astype('uint8'), [3,]), relay.reshape(var_1734.astype('uint8'), [3,]), relay.reshape(var_1734.astype('bool'), [3,]), relay.reshape(const_1735.astype('float64'), [40,]), ), 2)
call_1736 = relay.TupleGetItem(func_1110_call(relay.reshape(var_1734.astype('uint8'), [3,]), relay.reshape(var_1734.astype('uint8'), [3,]), relay.reshape(var_1734.astype('bool'), [3,]), relay.reshape(const_1735.astype('float64'), [40,]), ), 2)
uop_1739 = relay.log2(const_1735.astype('float64')) # shape=(40,)
var_1741 = relay.var("var_1741", dtype = "float64", shape = (40,))#candidate|1741|(40,)|var|float64
bop_1742 = relay.mod(uop_1739.astype('float32'), relay.reshape(var_1741.astype('float32'), relay.shape_of(uop_1739))) # shape=(40,)
uop_1745 = relay.exp(bop_1742.astype('float32')) # shape=(40,)
func_1604_call = mod.get_global_var('func_1604')
func_1606_call = mutated_mod.get_global_var('func_1606')
var_1750 = relay.var("var_1750", dtype = "float64", shape = (112,))#candidate|1750|(112,)|var|float64
call_1749 = relay.TupleGetItem(func_1604_call(relay.reshape(var_1750.astype('float64'), [112,])), 2)
call_1751 = relay.TupleGetItem(func_1606_call(relay.reshape(var_1750.astype('float64'), [112,])), 2)
bop_1752 = relay.not_equal(uop_1745.astype('bool'), relay.reshape(call_1733.astype('bool'), relay.shape_of(uop_1745))) # shape=(40,)
bop_1755 = relay.not_equal(uop_1745.astype('bool'), relay.reshape(call_1736.astype('bool'), relay.shape_of(uop_1745))) # shape=(40,)
uop_1757 = relay.asinh(bop_1752.astype('float64')) # shape=(40,)
uop_1759 = relay.asinh(bop_1755.astype('float64')) # shape=(40,)
output = relay.Tuple([bop_1728,call_1731,var_1734,call_1749,var_1750,uop_1757,])
output2 = relay.Tuple([bop_1728,call_1732,var_1734,call_1751,var_1750,uop_1759,])
func_1761 = relay.Function([var_1727,var_1734,var_1741,var_1750,], output)
mod['func_1761'] = func_1761
mod = relay.transform.InferType()(mod)
var_1762 = relay.var("var_1762", dtype = "uint32", shape = (12, 3, 3))#candidate|1762|(12, 3, 3)|var|uint32
var_1763 = relay.var("var_1763", dtype = "uint8", shape = (1, 3))#candidate|1763|(1, 3)|var|uint8
var_1764 = relay.var("var_1764", dtype = "float64", shape = (40,))#candidate|1764|(40,)|var|float64
var_1765 = relay.var("var_1765", dtype = "float64", shape = (112,))#candidate|1765|(112,)|var|float64
output = func_1761(var_1762,var_1763,var_1764,var_1765,)
func_1766 = relay.Function([var_1762,var_1763,var_1764,var_1765,], output)
mutated_mod['func_1766'] = func_1766
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1196_call = mod.get_global_var('func_1196')
func_1197_call = mutated_mod.get_global_var('func_1197')
call_1790 = func_1196_call()
call_1791 = func_1196_call()
output = call_1790
output2 = call_1791
func_1809 = relay.Function([], output)
mod['func_1809'] = func_1809
mod = relay.transform.InferType()(mod)
mutated_mod['func_1809'] = func_1809
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1809_call = mutated_mod.get_global_var('func_1809')
call_1810 = func_1809_call()
output = call_1810
func_1811 = relay.Function([], output)
mutated_mod['func_1811'] = func_1811
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1654_call = mod.get_global_var('func_1654')
func_1656_call = mutated_mod.get_global_var('func_1656')
call_1818 = func_1654_call()
call_1819 = func_1654_call()
func_470_call = mod.get_global_var('func_470')
func_471_call = mutated_mod.get_global_var('func_471')
call_1836 = func_470_call()
call_1837 = func_470_call()
output = relay.Tuple([call_1818,call_1836,])
output2 = relay.Tuple([call_1819,call_1837,])
func_1852 = relay.Function([], output)
mod['func_1852'] = func_1852
mod = relay.transform.InferType()(mod)
mutated_mod['func_1852'] = func_1852
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1852_call = mutated_mod.get_global_var('func_1852')
call_1853 = func_1852_call()
output = call_1853
func_1854 = relay.Function([], output)
mutated_mod['func_1854'] = func_1854
mutated_mod = relay.transform.InferType()(mutated_mod)
func_392_call = mod.get_global_var('func_392')
func_393_call = mutated_mod.get_global_var('func_393')
call_1860 = relay.TupleGetItem(func_392_call(), 0)
call_1861 = relay.TupleGetItem(func_393_call(), 0)
func_865_call = mod.get_global_var('func_865')
func_866_call = mutated_mod.get_global_var('func_866')
call_1862 = relay.TupleGetItem(func_865_call(), 0)
call_1863 = relay.TupleGetItem(func_866_call(), 0)
func_1543_call = mod.get_global_var('func_1543')
func_1548_call = mutated_mod.get_global_var('func_1548')
var_1865 = relay.var("var_1865", dtype = "float64", shape = (154,))#candidate|1865|(154,)|var|float64
var_1866 = relay.var("var_1866", dtype = "uint32", shape = (360, 1))#candidate|1866|(360, 1)|var|uint32
var_1867 = relay.var("var_1867", dtype = "float32", shape = (2304,))#candidate|1867|(2304,)|var|float32
call_1864 = relay.TupleGetItem(func_1543_call(relay.reshape(var_1865.astype('float64'), [154,]), relay.reshape(var_1866.astype('uint32'), [12, 30]), relay.reshape(var_1867.astype('float32'), [2304,]), ), 8)
call_1868 = relay.TupleGetItem(func_1548_call(relay.reshape(var_1865.astype('float64'), [154,]), relay.reshape(var_1866.astype('uint32'), [12, 30]), relay.reshape(var_1867.astype('float32'), [2304,]), ), 8)
uop_1869 = relay.log(var_1867.astype('float32')) # shape=(2304,)
uop_1880 = relay.atan(uop_1869.astype('float64')) # shape=(2304,)
bop_1882 = relay.add(uop_1880.astype('uint64'), relay.reshape(var_1867.astype('uint64'), relay.shape_of(uop_1880))) # shape=(2304,)
const_1885 = relay.const([8.189465,-4.942164,6.001317,-8.642670,8.070928,3.007082,5.702455,-2.528266,-4.715974,-9.106119,-2.379617,-3.284734,-9.718194,9.398540,-1.546658,-7.621321,-2.021198,-6.350113,-5.190374,6.723380,8.672855,-9.021799,6.893438,7.960838,9.800051,7.255500,5.966032,-8.836285,7.665341,-7.414521,-5.457963,7.714433,-7.143198,8.251928,2.621042,-9.756774,1.998346,3.338120,-4.728336,-1.566063,6.877459,6.618949,6.521976,-9.947564,3.074513,-9.997637,7.805658,-7.240884,-3.125328,-0.451817,9.033373,-4.618629,-3.410296,8.756294,3.182068,-6.017690,5.773522,0.371799,3.862014,5.846505,0.784185,-3.390667,0.829330,-9.789342,-4.089013,8.163387,3.929634,3.203746,-9.397971,-1.559561,-5.396367,-4.423739,-9.909527,9.233333,7.608226,-2.778633,-8.296904,9.830492,-2.629845,8.352466,9.811296,-1.361016,1.225083,-3.426472,-8.524732,9.445349,4.108906,-2.566414,9.686123,6.665107,-6.364261,2.044300,-8.197175,-0.326852,-7.190866,-1.421427,8.705369,4.129585,-8.819336,-0.382827,-2.884139,5.280075,2.800387,-8.096916,6.189671,5.266388,4.242065,5.577348,1.734572,-5.022497,8.000826,1.986223,0.343075,4.842296,-0.464762,-6.005311,9.240380,-1.690325,4.134979,-9.701434,-4.648309,-0.901251,-6.052811,-5.757741,-8.682491,-6.547739,4.917920,-2.873869,-1.358848,-6.489962,7.541518,3.413560,-7.901350,2.862749,-0.646215,4.454396,6.101093,7.870150,1.323740,7.534200,-3.382651,4.863301,-1.231970,2.175082,-3.006185,1.522719,-8.901017,-5.090656,8.978258,4.859008,1.176398,-6.320474,-3.186733,0.598024,7.867227,2.190906,-2.470502,7.167272,-9.498647,8.168600,-4.736220,-5.250467,5.003731,-3.554837,-9.355137,7.578205,-9.261204,-4.060032,8.701030,-9.057195,2.719715,-1.875558,-1.014612,-4.172700,-8.577646,1.547819,-3.196819,-4.237194,9.052633,-1.140644,-1.799764,4.609676,-0.911563,1.766915,8.402046,-6.518575,-2.668797,-8.989865,3.126389,7.786137,-8.275805,-2.513054,7.691991,5.560185,5.434399,-5.972529,4.121624,-3.604618,-0.369783,7.965079,2.001448,-3.397445,-3.018016,6.219551,8.946277,-7.588634,4.172706,-9.434291,7.900790,0.623151,-9.668366,-4.278321,-5.542576,4.446320,-3.681251,-5.920091,-9.233212,7.295190,-3.000872,-6.810361,-9.547352,7.805724,-3.443645,-6.681132,5.840719,-5.041228,8.058275,2.400325,-5.343704,-5.923241,7.690757,-1.457311,6.176603,1.544580,-4.430096,3.280809,-6.381199,3.173868,8.342850,-7.474746,9.899076,-8.469312,4.508951,-3.024020,-8.553477,-8.374619,-8.612799,9.289839,2.434536,8.254676,4.359921,-5.398734,4.921145,-7.567375,8.135255,-2.966158,8.702157,8.483551,-5.719429,-8.337022,9.055237,6.143708,-2.428207,4.841262,3.224680,9.346419,-0.596640,-2.702033,6.091361,-3.654858,1.313324,-6.934866,2.537453,4.774059,0.133269,-0.157188,-5.346974,-9.637065,-5.274125,-9.392406,-2.863413,-9.021024,2.399278,4.763982,3.257688,0.984588,-0.952173,-8.592582,-3.900692,7.090694,7.359535,-7.735642,1.319786,-9.490703,-9.160825,4.329989,0.019857,6.955775,-8.268476,5.060423,-6.928856,-4.274994,-0.523938,-8.645749,-0.914721,2.446415,6.315456,-6.717835,-8.984156,9.877955,5.494246,-8.743200,-9.946006,-1.299254,-5.349613,4.766351,-8.472617,7.932961,-7.210960,4.997927,2.540267,-6.431226,-6.304738,-3.161614,-1.624705,-6.473515,-6.014278,-5.494312,-2.236894,-5.794309,-9.130278,-2.253175,0.620834,-0.441031,8.024304,7.930696,-8.513358,-6.799877,-5.778570,6.230921,-8.792479,-2.269028,-9.765055,5.662950,9.881422,2.375388,4.462478,8.874497,7.113674,-1.429762,2.583274,-0.290172,-3.498769,6.517445,1.013437,2.958831,-9.577164,-1.390050,-7.699668,3.617578,6.596758,7.008036,-1.021602,-2.198036,-8.099039,1.059604,-7.010985,4.931503,-8.128272,8.550270,-1.410498,7.060652,-6.847579,1.390314,2.150810,4.580538,-6.261272,-3.772430,8.790704,-1.351848,6.290808,6.911408,1.564084,-0.506833,1.735017,0.369821,-0.553539,-6.752097,9.429303,0.538512,7.493618,7.888934,-6.139606,-9.495978,-8.539958,0.611814,-1.600926,9.055882,-8.149838,0.795404,7.535445,0.494374,4.912901,1.728362,9.781914,-5.607137,2.484958,-8.482312,5.468134,3.694971,-3.739317,4.960389,4.964746,-8.137915,4.541965,-3.445972,5.835768,-9.255511,9.445031,0.628193,9.862798,6.653849,0.724232,-1.912999,6.352975,0.717852,-6.583750,-0.783702,3.539955,0.013223,2.559538,-4.150998,3.899236,8.901501,-8.514726,-6.654426,-8.848354,-6.135795,9.289220,-8.028771,-7.200336,-1.113964,4.721819,-2.165152,-1.774898,2.724305,1.301112,-9.588300,-5.171297,-1.401754,-3.627272,-7.093449,-7.823212,-7.870661,5.715206,-3.377753,-7.901424,1.108181,-6.394565,7.867229,0.707770,-6.729754,-9.589444,-2.874381,-9.401004,7.998426,3.212289,-3.119581,-7.963187,1.328933,5.356526,0.076355,7.351377,7.383939,-8.001093,-3.133022,4.567152,5.967843,6.251498,7.102355,-2.457226,-5.460302,-7.545463,8.158042,4.452258,8.662992,2.997045,6.034570,6.341127,-1.005745,8.560872,3.320706,8.952035,-2.319184,-2.303924,-5.900744,-6.604915,-8.391490,4.992149,7.381608,3.485985,-8.134533,3.740826,5.285283,-1.224147,-6.683583,-5.180343,8.187147,-4.940397,-9.210946,7.745404,7.382700,6.708097,-5.856931,-7.263532,6.200928,-0.945571,-2.370473,3.079322,-0.039655,4.824974,6.637039,3.743199,6.846350,-3.829055,-1.742132,-5.024628,0.015653,-7.428396,4.308138,2.393041,-0.820826,-3.693237,8.243471,-4.919692,8.911613,3.040002,6.728223,1.927990,-8.032067,-6.683444,6.487601,3.894967,-3.759227,2.371102,-1.481787,-2.396137,-3.194953,1.145320,6.828027,-1.746024,1.880699,3.290526,-6.070170,6.043141,0.476053,3.968610,-3.443905,4.874048,-2.829893,-7.753515,-9.080262,3.128350,0.360523,-2.557729,-1.439476,-6.797212,-8.859189,-0.869631,-8.272815,-5.189107,9.577092,1.147407,-9.282512,-7.115342,4.676830,1.469806,-9.788543,-9.442566,1.188764,-0.643127,9.940370,4.051854,-4.581373,-3.551460,8.142356,-2.294246,0.490518,-9.975165,4.932752,-9.247507,-1.391824,-9.976815,1.517922,-5.636500,9.325674,-0.328775,0.646537,4.437981,-7.064678,5.245659,-2.080234,-5.287825,0.593063,6.030476,-8.521478,4.949249,7.511283,7.394286,-7.610544,6.559856,-4.783067,6.020176,8.989523,8.846313,-3.076957,1.896013,1.009176,-7.840799,-5.102527,2.913167,-9.394781,0.057870,-9.052170,-6.516405,3.153951,-1.160116,8.814460,7.020138,-7.561341,-9.829739,3.168273,4.127975,-9.073158,7.055616,1.348818,7.935179,7.479867,7.158091,6.862968,1.922283,8.100910,-9.987802,8.365413,3.116817,-5.734593,-7.474655,6.956545,-9.358743,-4.628513,3.106194,-8.825588,5.741292,8.872492,1.151721,-2.031917,8.527085,3.322860,-5.217170,-6.831122,-5.657291,-0.727883,-7.407551,6.854755,7.584953,6.224821,7.321312,6.735621,3.078889,-9.826936,1.248999,9.918614,-3.528409,-9.596659,1.413292,9.989841,6.638923,1.412110,3.105532,9.996611,-7.468104,5.039024,-5.857873,-4.275447,-0.915473,5.417035,3.851823,-9.158776,8.105812,-8.546186,2.276513,-6.756649,8.508445,4.553884,-5.319357,6.529004,6.821887,8.236021,0.094150,5.129943,-6.195045,4.972552,-6.793308,3.647816,-2.514245,-7.050640,5.419260,8.667194,-9.469931,-9.779987,9.133120,8.929248,-3.069924,-9.843587,-8.769904,3.830691,-4.878503,-8.253980,3.884745,-2.870215,-3.274488,2.828441,-5.897887,-9.792423,2.175852,4.289819,9.375803,9.939436,1.478865,7.370700,-4.690627,-5.703564,-8.387038,3.088820,8.916148,1.745682,-0.514684,8.225842,9.303439,-6.950013,-5.730009,7.364769,-6.497306,-2.952566,4.613204,-9.927621,-1.923899,-4.605045,3.250013,-9.315660,-2.934395,-8.938539,1.872424,8.804895,-9.755189,0.737754,3.097899,-2.900809,-1.901177,-1.719341,-2.434800,3.673043,7.810259,5.040427,3.117119,0.656477,-7.047628,-9.309957,-0.402857,-8.255692,0.730241,3.047307,6.124602,2.438809,-2.450356,7.686640,-3.984335,-0.694036,-4.656305,-9.888894,0.299358,7.698514,-6.907626,4.739953,6.582035,0.870852,-6.631926,-7.308070,-2.851660,9.270453,1.149484,4.728147,7.541128,3.663886,-2.000675,4.476481,-3.622621,5.594961,-2.897835,3.074134,-9.374798,4.565673,3.539127,-3.220389,6.426232,4.125959,-0.146349,-4.242041,-5.015900,-6.076571,1.992451,-2.186650,-2.188570,-7.809653,6.614186,3.695389,-9.054185,-2.870130,-0.280008,2.930857,-7.371979,-0.238180,-6.583960,-8.649065,-3.023989,6.414164,-1.492243,3.958454,5.865805,3.962143,6.636412,-4.843706,4.313045,-3.052508,-5.194621,7.325022,-6.980401,9.863525,-6.055353,2.711728,-3.847445,7.406885,-0.009236,-1.460372,-2.202421,-2.156307,-7.081603,-9.117169,4.308019,7.871629,-5.087413,-7.535994,-8.423738,5.988002,6.899025,1.454665,0.806801,-1.187567,-3.302607,4.556428,3.146469,-1.848488,4.384437,7.858016,7.274156,-1.968663,-8.014838,-9.757366,-8.780330,8.178889,-9.431150,-1.251124,7.883388,-5.034899,9.749120,6.604751,3.907220,-2.494958,-5.761841,-8.883325,-2.749272,8.321356,9.268632,5.550608,-5.350060,-6.917650,9.300207,-4.462366,3.758240,-2.897943,2.323503,9.818266,-8.603189,9.097373,-9.428312,-6.356693,-8.640152,5.822907,-9.372956,6.191773,8.685151,0.842914,-1.697136,-4.773441,-8.510705,-3.264817,-9.854737,6.386281,0.922992,0.054990,-7.594156,-9.243869,8.100277,-4.083833,-9.341968,-5.125715,7.089170,4.033257,-6.066893,-9.064604,-6.380964,-0.802030,-2.907674,0.058238,-5.573151,4.402080,9.047101,0.041433,3.270008,-3.224936,0.493355,-5.267510,-2.574147,2.686739,5.869241,-1.251254,-8.024044,-0.246726,-8.060512,2.835563,-7.055335,7.864675,-4.043596,-2.209675,-4.744965,-2.855420,-6.755454,4.737901,-0.138174,6.936963,-1.065486,4.950350,-1.696125,4.575335,-0.737545,6.931520,-2.343484,-9.962054,7.373907,-8.741071,2.303588,1.889720,7.723484,-0.926834,6.621750,9.424861,5.308213,9.233426,-8.623398,-5.293036,5.839131,3.884483,-8.886733,-5.550161,8.088453,-6.485568,5.968332,-9.268932,-4.943584,-7.740107,-3.663192,7.591372,7.644430,4.100661,5.399780,-1.367474,4.814682,-8.304095,9.487484,6.250932,-4.978590,8.421918,8.440387,-2.248009,-8.241857,-2.095512,2.044653,2.252827,-2.344439,-0.161032,-1.833592,-6.738199,9.181007,-4.459975,1.070279,1.213660,2.183202,-1.124048,-6.073831,-4.028839,8.753013,7.975313,-2.469350,9.139085,8.966421,-4.575186,-0.386223,0.346850,-7.522064,-3.807371,7.977376,5.607738,-1.036231,0.960128,-9.633642,-2.895725,-2.087562,-5.089840,-4.180725,9.337931,-3.935386,3.134885,-6.739165,-9.718646,-9.210331,8.089215,-6.671360,-1.691779,-2.363129,6.129688,7.914261,-2.787126,9.508190,-8.734996,2.460561,3.672777,-6.457344,-5.939857,9.849034,6.507077,-7.505956,5.667161,1.676765,1.426367,0.052260,-9.216002,-8.015402,7.823368,-9.927708,9.437059,9.706153,8.547148,-2.890020,6.931474,-4.556895,4.581161,3.754619,6.827625,1.964229,-5.946486,2.797715,8.105962,6.669438,0.040166,9.945774,3.626658,2.713936,-1.588317,8.674563,5.350036,-8.025094,-6.493248,6.443042,-9.947440,-5.025732,0.007792,9.440901,-4.533911,0.263550,-3.304000,-6.511595,-2.196442,-1.931900,0.393502,-8.990344,0.852792,-0.522848,-0.712432,0.672885,-1.213725,6.184488,8.363515,-7.267696,6.801355,4.183354,-6.071307,-6.622361,4.934374,0.449255,-2.408892,6.257537,-2.844858,7.783066,3.868641,2.802330,-6.156228,-9.512990,-1.926159,5.054811,-8.319303,2.385803,9.640818,6.113718,2.593515,8.455238,-5.713101,9.004394,-2.586227,8.487007,-7.417316,7.975731,0.166272,-4.253750,2.377491,-8.693285,3.985882,-0.707246,9.627773,1.090695,0.052133,-0.823240,-1.648310,-4.168608,-6.651937,-4.822626,5.385228,9.237535,2.134873,-9.684026,-5.734999,7.573963,-5.618531,-7.503782,7.569405,-2.378337,-2.280944,8.907938,0.541940,-0.126989,2.612736,-9.480910,-6.732630,-9.999943,2.991413,-0.061678,4.312353,6.324341,-2.547015,7.381933,-1.320373,-7.821489,-7.041362,5.410987,2.144686,3.398111,-6.417320,6.343553,1.511845,2.385405,-0.532891,-6.062226,1.681605,-5.423964,9.367989,6.298768,5.548609,7.011428,8.733958,6.183021,-5.986308,-7.751483,-3.541512,5.724343,3.073847,5.177529,4.746159,8.660517,9.239384,-3.100999,9.728655,-9.554353,6.119060,-7.559529,-2.217299,7.748326,-9.022431,-9.389118,-6.818655,-6.515201,-4.352655,5.735812,-7.511533,-9.239031,-1.267179,-9.962566,-8.232450,7.507681,-4.317620,6.150424,2.109842,9.392352,-6.584874,6.687033,-7.392987,-2.630410,0.580402,-9.514523,-7.532602,4.732659,9.562402,-3.317153,5.835180,9.250513,4.785707,-5.553302,7.872995,-0.674541,9.771713,-1.005029,9.430362,-4.434521,5.368514,-3.671352,2.443089,-1.562198,-4.595110,8.134568,-2.252079,3.695340,8.201226,1.512784,-8.619220,-7.658390,9.755746,6.563947,6.432157,4.472378,-7.994672,-7.814735,9.801217,-1.961910,0.318604,3.182690,-3.038494,6.667854,3.280728,-1.305929,-2.624681,3.957968,-6.714051,-2.096780,4.921464,-6.571155,-5.947891,7.601288,1.176125,6.381764,0.950988,0.782636,8.733728,2.387405,1.144850,-0.296906,-7.110281,4.215448,-0.594269,-8.169152,6.447049,8.257445,9.210102,3.984372,-1.324652,6.743400,-9.605699,9.444976,9.056350,-3.482384,9.047050,-1.211205,-3.027677,-8.875841,-5.324790,-6.864090,2.181245,0.839670,0.111765,6.549000,1.781876,2.914530,2.066531,3.731046,-2.345070,8.377001,2.963538,7.809506,8.791763,-5.742550,1.889603,0.600946,3.922070,3.339337,-4.706617,-5.059552,-4.436163,6.061244,4.558188,-8.744308,-5.946276,3.795247,-6.897760,3.333424,-0.070409,2.049356,5.707300,-5.840597,2.764403,6.179136,-7.233353,0.928807,6.208949,9.220466,8.692821,-2.677513,8.140441,6.751372,2.416006,-0.305956,1.225347,5.206318,-7.058452,6.333698,5.514322,9.833727,-0.860006,-1.254284,8.043521,-8.937524,-9.224623,-1.224666,-1.414923,2.634784,8.077327,7.864021,7.888692,6.980947,-9.656713,4.317354,3.117511,-6.101443,7.060307,5.310772,-5.717631,7.549501,9.639660,-3.087170,-0.335487,8.052602,5.599559,-8.050831,8.743370,2.771548,-0.971041,0.959202,8.184375,9.240382,2.851698,-5.361722,4.338291,-7.087427,5.402755,-1.954443,-0.773956,0.295805,5.397811,-1.783456,8.851476,-9.342101,-0.072902,7.586824,5.578395,-3.889008,-3.308051,-2.444549,-1.623467,-6.676520,-6.712585,1.852382,-3.760927,-2.756524,-5.412687,-4.552645,-7.411392,0.972045,-5.065819,1.629335,-0.042826,-4.279324,-5.196895,-6.626678,8.112619,-3.001998,4.960317,-3.095296,5.434893,3.098781,2.120911,6.077523,2.205183,-4.595571,-3.090097,8.772637,1.738985,5.354435,3.958491,9.230920,-2.956891,-4.125853,6.804495,3.086866,8.415462,9.745000,-8.849568,8.393591,0.993839,5.010581,6.374355,-2.290287,2.307163,-9.643052,7.124525,5.998911,0.627922,-5.915047,3.911302,8.809272,-1.929357,1.190047,-9.068003,-3.410184,0.941503,2.735341,3.922419,7.966220,6.357495,9.816656,1.201205,-5.205884,-9.677023,-7.110059,-2.650108,-2.834398,-0.520916,9.473985,-7.792383,-8.979961,7.033079,-2.013494,-0.966206,1.931734,3.142931,-2.975869,-7.374822,-6.815402,9.484334,-3.842715,-9.966795,2.224985,-4.199904,5.398143,-6.070528,-7.314426,5.545613,-1.623027,2.522851,1.438959,-8.609040,3.148044,-7.405311,2.913683,-8.718971,7.540442,0.417087,2.691448,7.541333,-5.799300,3.478416,-8.497913,-9.128770,-7.717037,-9.636011,-6.378701,4.435269,5.014821,8.545429,-0.502876,-1.462276,0.731965,-3.443208,-0.455450,2.945548,6.570481,2.427938,-1.992794,5.127119,6.510544,-1.701611,-4.758546,-1.937052,-9.021219,1.708412,2.455231,-9.451531,6.539119,-6.130934,6.467137,-0.140059,-6.305669,-0.522635,1.536929,6.099771,4.139130,6.930350,3.942812,-5.682511,4.400312,-1.260405,1.108033,7.886563,6.803840,-3.830858,-5.231522,-3.783366,3.236212,7.888221,-1.899423,0.465078,-0.637979,5.982698,-2.003075,9.837308,9.161998,6.784846,7.922095,0.615985,-5.220081,9.895308,8.778197,0.618339,-9.389600,5.097723,8.699626,-8.833804,3.837153,6.412546,5.404970,-6.119042,2.003997,-3.234627,-9.377891,4.209703,8.495773,2.518914,-4.899265,6.056559,6.986712,6.735692,-3.671457,3.229378,-7.911485,7.645680,5.501060,-2.563426,5.352292,1.338809,3.979311,-2.324795,4.095700,4.855523,-9.452594,-0.306943,8.749320,-6.950709,-8.661133,-8.338210,0.456116,-2.970139,5.265926,4.016689,1.798055,-9.359603,9.573936,-5.395267,1.860171,3.631361,-9.396067,0.114714,7.771418,9.220130,8.543536,-8.568866,-6.619778,-9.550630,-6.720302,-2.030047,-4.172833,-1.509748,-8.734546,8.007877,-8.308490,8.541187,-4.484029,-7.501228,5.838463,6.937744,4.852720,4.902787,9.847901,9.084797,2.695058,5.126295,2.088796,7.934838,-3.912824,3.478966,8.023385,-6.557376,1.496711,-7.987043,-9.149399,-4.819923,5.715918,9.190530,4.580130,-3.678785,0.063125,-4.200995,9.059159,-9.980443,-6.404825,-9.657349,-5.107879,-9.676039,-8.614807,-9.970720,-1.983939,-9.217538,-0.733053,-6.856718,0.253614,-8.761166,3.072316,3.385478,2.854465,-8.829561,4.427901,2.774663,7.744834,4.227279,-8.014954,2.634316,-0.022540,3.128355,8.921075,-0.257245,-4.668652,0.645875,-1.257293,6.411328,-7.254353,9.222262,-0.626846,-2.359853,-1.919514,1.317834,1.816175,-6.657648,6.549882,9.003704,9.994270,1.531082,-9.922005,9.463021,3.556789,-1.661771,2.615801,4.490603,8.381142,3.524591,6.445919,-3.909787,8.177428,0.793147,-4.749674,-3.712973,-2.791374,-1.412058,-5.597551,1.144196,-4.436830,-1.627518,-2.344229,3.894441,2.945125,0.734655,-9.146672,4.438271,-7.885029,-5.800743,4.808094,8.110572,-8.992351,1.540163,6.880043,-6.694153,-2.070908,7.642912,-7.422343,-8.284635,-1.099629,-5.058568,-0.484286,-9.175844,-4.730725,7.788411,9.081490,-0.835557,-7.855227,8.179349,-1.447505,5.267826,-6.120086,7.178985,-8.303793,-9.059867,-6.155410,0.540597,-3.412658,8.950154,-2.178112,8.522103,5.794567,7.534119,2.993784,-8.586285,-1.214493,-7.761628,-2.416601,9.648474,-4.482692,8.349423,-2.690983,3.848081,8.019804,-1.416531,3.524327,1.359715,1.957723,7.293048,9.603857,-0.733914,5.736891,-7.579737,6.979481,-1.077544,5.508445,-2.910014,-9.462659,4.370290,-8.271181,4.475708,4.220454,1.243241,6.647948,3.428830,-4.730934,-1.407026,-9.201912,2.231389,-2.691960,2.067021,-8.645776,-0.215996,2.979367,-6.476995,8.926724,-9.923557,-1.744559,3.495970,-0.293546,6.958417,-1.197814,-8.399833,-4.508347,-9.431272,-5.160543,5.902758,9.989616,9.048694,3.267924,1.785586,-9.339415,9.517027,4.242998,-7.412747,2.416047,-5.843766,-3.976669,-5.811447,-0.239899,8.482043,0.748611,3.791539,-6.247821,6.520460,-1.707904,-8.040272,-8.807136,-9.270283,-3.693616,2.822436,-1.260576,-2.951898,2.874769,1.918160,0.366824,-6.975385,0.422840,-4.297171,-8.989948,6.248667,-3.531121,0.509505,9.419683,-9.344518,9.472209,1.762657,8.079285,-3.439888,1.745872,-6.080068,-6.286393,4.781209,4.118049,-5.647863,3.199776,-7.941922,8.168275,5.355498,6.756858,2.583571,7.687887,-8.057930,-5.357721,2.517091,7.948922,-6.497546,-7.282954,6.362195,7.967033,-7.658904,-1.862636,-2.144825,-9.484671,-9.827273,2.236534,-6.118982,0.465994,9.548408,9.034972,2.751664,4.853601,7.354235,6.535630,8.057034,-5.425772,1.059060,-0.594205,-9.796715,7.576588,1.761619,-9.600072,-8.564190,-3.301474,8.795254,-9.808352,-1.714180,6.550215,-5.843597,1.915404,0.939783,4.205850,-0.768198,6.959955,-8.713779,5.654095,8.348721,-6.132956,4.364225,-3.665215,-6.787798,4.422013,-8.756560,-1.897907,2.274872,-5.012697,3.519872,-8.345738,2.504944,2.485098,-1.803390,7.455150,-6.041859,0.177334,4.975017,8.884751,7.834923,-1.455869,9.866728,-7.692661,-9.578186,8.871713,-8.887925,9.108072,2.687010,9.020539,-0.051868,-6.224028,-3.370462,-3.157902,1.732837,-5.897504,0.751570,-3.783308,8.057066,3.729572,0.756669,-7.694751,-5.909514,3.503477,-0.068646,-8.852761,-1.213347,0.968374,-8.560387,-7.414144,1.541361,-1.634611,5.198251,0.039376,1.363931,4.287841,-0.070179,-3.649603,8.521271,-2.526407,9.312574,-9.172496,5.563874,-8.809467,-1.801544,9.954893,3.362756,-6.794070,-6.489147,-9.536810,-0.421012,-5.923365,1.467173,-8.631120,1.381232,-3.481377,4.534572,-6.086122,-0.126330,-4.793713,8.807523,9.974410,-2.182853,1.698407,8.075777,-3.793594,8.909360,-5.617239,-1.876140,-9.423274,-3.952474,-6.201737,8.192826,-5.807054,5.465419,-6.788247,8.032330,-2.622299,6.144136,-3.727478,8.287024,1.874751,-9.299885,6.248593,-6.701829,-6.249955,-2.247895,-7.592363,2.201425,8.705172,4.089755,8.168343,2.081859,-8.180763,0.514079,-9.752845,-9.647431,-8.000414,9.277486,3.398259,7.018487,-8.670045,-7.558471,3.658726,-4.812699,-5.622316,6.830258,-4.875146,9.526649,6.161240,4.064611,8.932997,2.315633,-8.769690,6.357914,-1.092119,2.819572,-2.858793,6.429085,7.245996,0.133197,-5.619059,1.690782,1.129725,1.923562,-0.852843,-9.120055,-1.908014,1.107007,5.902756,-4.000725,9.899845,-1.772920,4.072842,7.413547,8.836701,0.155812,8.658328,-8.194162,1.548896,-7.069990,2.145552,-0.299453,-1.711078,-9.670982,-3.509218,-4.600199,5.038219,-6.563272,-9.988819,9.696359,-1.841281,0.641007,-1.219129,-7.220064,-1.867676,5.448031,-8.146453,8.532118,0.502067,-5.183810,0.818148,5.257565,9.659966,6.228926,8.547305,-6.378974,6.409555,-7.563716,0.181660,-6.230343,5.381806,5.825535,3.814340,-2.980880,3.129869,-4.272590,1.323291,2.584407,0.550553,2.961460,8.968581,-1.678488,0.014402,-3.704940,-3.939921,-8.960474,-3.932266,6.900602,5.307405,-3.696466,-9.512542,9.717973,-2.740126,-7.125964,1.041239,3.918172,0.546662,-8.180147,-3.306073,-5.884789,4.178847,-8.915686,-2.012037,5.778900,5.636906,-8.405068,-1.249632,-5.226508,-4.564664,8.709658,-1.352439,0.182115,-3.598973,-1.975720,-1.499081,1.416249,-3.288953,-4.372538,2.443645,-8.120592,3.703281,-1.432459,-2.393323,7.055089,-4.873064,-1.530616,2.983476,-6.401112,9.948418,-4.391690,4.053025,2.604724,2.886371,2.674266,2.917602,6.682119,-0.871824,3.614410,-5.448297,-7.726617,9.557069,5.985936,-1.143081,-1.117288,9.582965,4.099937,9.820030,-0.889910,-6.199364,7.872992,5.739221,4.378542,1.496607,-9.145415,1.777917,-1.011563,9.443061,0.585746,7.240985,-5.399008,-1.625839,-5.391968,-9.641656,-4.076597,-2.516702,1.210203,-9.819461,8.188411,1.706278,-8.683196,-8.867853,5.283481,-7.579635,5.543832,-2.504377,-9.636255,-7.636664,-7.184889,7.126355,2.320985,0.747590,1.690347,9.842280,4.116897,-0.567886,4.772319,-9.722477,1.679812,7.771493,-2.869621,-6.092227,-4.125837,8.971602,9.485805,6.684041,-0.512719,5.833957,1.414049,4.833647,7.255830,-3.131626,-9.187468,2.708219,7.027213,8.900349,5.281945,6.348615,-9.037920,-9.570073,7.865992,-1.632446,-4.901511,-6.702872,7.229376,-1.035852,0.817117,6.219815,0.024546,3.027494,-2.097118,1.691425,-9.598441,5.411774,0.978862,0.810007,-1.746454,8.560397,2.884148,-8.853446,0.612095,-2.903904,7.854983,-7.618582,-9.697841,2.977483,-3.692918,-8.501487,1.028867,3.648145,-7.088989,-8.774114,1.700907,8.327463,5.928103,8.701969,2.464484,7.752692,9.177494,1.610843,-9.278012,-2.802615,1.857361,1.838896,6.669100,-5.828455,8.105908,-6.757971,-9.476211,-5.055139,6.607164,-5.043712,3.202226,3.322906,3.753689,-5.354278,3.845061,7.777911,-0.893266], dtype = "float64")#candidate|1885|(2304,)|const|float64
bop_1886 = relay.equal(uop_1880.astype('bool'), relay.reshape(const_1885.astype('bool'), relay.shape_of(uop_1880))) # shape=(2304,)
output = relay.Tuple([call_1860,call_1862,call_1864,var_1865,var_1866,bop_1882,bop_1886,])
output2 = relay.Tuple([call_1861,call_1863,call_1868,var_1865,var_1866,bop_1882,bop_1886,])
func_1891 = relay.Function([var_1865,var_1866,var_1867,], output)
mod['func_1891'] = func_1891
mod = relay.transform.InferType()(mod)
var_1892 = relay.var("var_1892", dtype = "float64", shape = (154,))#candidate|1892|(154,)|var|float64
var_1893 = relay.var("var_1893", dtype = "uint32", shape = (360, 1))#candidate|1893|(360, 1)|var|uint32
var_1894 = relay.var("var_1894", dtype = "float32", shape = (2304,))#candidate|1894|(2304,)|var|float32
output = func_1891(var_1892,var_1893,var_1894,)
func_1895 = relay.Function([var_1892,var_1893,var_1894,], output)
mutated_mod['func_1895'] = func_1895
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1897 = relay.var("var_1897", dtype = "float32", shape = (15, 1))#candidate|1897|(15, 1)|var|float32
var_1898 = relay.var("var_1898", dtype = "float32", shape = (15, 4))#candidate|1898|(15, 4)|var|float32
bop_1899 = relay.power(var_1897.astype('float32'), var_1898.astype('float32')) # shape=(15, 4)
bop_1904 = relay.greater_equal(bop_1899.astype('bool'), relay.reshape(var_1898.astype('bool'), relay.shape_of(bop_1899))) # shape=(15, 4)
uop_1909 = relay.cos(bop_1899.astype('float32')) # shape=(15, 4)
bop_1912 = relay.multiply(uop_1909.astype('int32'), var_1897.astype('int32')) # shape=(15, 4)
bop_1915 = relay.add(bop_1912.astype('uint64'), relay.reshape(bop_1904.astype('uint64'), relay.shape_of(bop_1912))) # shape=(15, 4)
uop_1918 = relay.acos(uop_1909.astype('float32')) # shape=(15, 4)
const_1920 = relay.const([[3.455799,-3.313814,1.526195,-6.124534],[-6.059136,-7.062998,-7.242669,5.669953],[-5.898947,7.604498,7.131257,-8.721520],[0.119007,7.589964,7.481122,4.062525],[-6.587305,-4.233598,2.776584,-9.051276],[-7.296756,3.577209,-7.105412,3.717266],[-0.179984,-1.482827,9.179336,-8.286101],[6.536277,8.809080,-5.346958,-3.105173],[-7.374128,-7.451320,3.547737,1.639614],[-9.509565,0.199581,3.981041,9.025880],[5.599309,0.222622,-9.661690,-3.312899],[7.242054,-7.527232,-0.892531,8.965652],[8.025760,-3.107607,-5.522299,-8.776492],[-7.672625,7.023356,-2.126882,2.734942],[-6.160399,-5.697883,-3.659802,-5.713705]], dtype = "float32")#candidate|1920|(15, 4)|const|float32
bop_1921 = relay.right_shift(uop_1918.astype('int64'), relay.reshape(const_1920.astype('int64'), relay.shape_of(uop_1918))) # shape=(15, 4)
bop_1924 = relay.bitwise_and(uop_1918.astype('int32'), relay.reshape(bop_1915.astype('int32'), relay.shape_of(uop_1918))) # shape=(15, 4)
uop_1927 = relay.rsqrt(bop_1912.astype('float32')) # shape=(15, 4)
uop_1929 = relay.sqrt(bop_1921.astype('float64')) # shape=(15, 4)
uop_1931 = relay.asin(uop_1929.astype('float32')) # shape=(15, 4)
func_1654_call = mod.get_global_var('func_1654')
func_1656_call = mutated_mod.get_global_var('func_1656')
call_1933 = func_1654_call()
call_1934 = func_1654_call()
output = relay.Tuple([bop_1924,uop_1927,uop_1931,call_1933,])
output2 = relay.Tuple([bop_1924,uop_1927,uop_1931,call_1934,])
func_1935 = relay.Function([var_1897,var_1898,], output)
mod['func_1935'] = func_1935
mod = relay.transform.InferType()(mod)
var_1936 = relay.var("var_1936", dtype = "float32", shape = (15, 1))#candidate|1936|(15, 1)|var|float32
var_1937 = relay.var("var_1937", dtype = "float32", shape = (15, 4))#candidate|1937|(15, 4)|var|float32
output = func_1935(var_1936,var_1937,)
func_1938 = relay.Function([var_1936,var_1937,], output)
mutated_mod['func_1938'] = func_1938
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1957 = relay.var("var_1957", dtype = "uint32", shape = (16, 3, 13))#candidate|1957|(16, 3, 13)|var|uint32
const_1958 = relay.const([[[-2,10,1,-1,-9,-10,-10,-3,2,-2,-4,-7,-10],[1,-6,-7,-6,-4,-5,4,2,2,-2,-10,6,7],[8,-3,2,-4,-1,-1,-1,5,-9,-10,3,-6,9]],[[-1,9,5,1,6,1,6,-2,-10,1,1,-4,-3],[-4,-4,-5,10,-3,9,-2,-1,-4,5,-3,1,10],[-4,-4,1,-5,1,-10,-2,-5,-8,8,9,6,10]],[[-1,-3,-4,3,-3,-5,9,-8,6,-3,3,-7,-1],[-2,4,-6,6,4,-6,7,-1,-10,8,-1,-8,-5],[-8,-1,-1,-9,9,-10,5,-5,10,10,4,-4,-8]],[[9,3,-6,-8,-1,-3,6,6,-5,9,-1,1,2],[1,-2,-3,4,6,-7,10,5,-10,3,8,-6,9],[-8,9,10,8,-9,-2,6,-5,-2,3,-8,10,-9]],[[-1,-7,-2,-3,-5,-9,4,3,-2,-1,4,-2,3],[-1,3,-1,-5,-7,3,6,-4,2,-4,3,4,5],[-9,6,-6,3,5,4,-4,8,2,7,10,3,4]],[[2,-1,-7,-3,-10,-8,-10,9,-6,6,2,8,3],[1,-4,3,-5,2,-3,-7,-5,8,-9,-9,1,8],[3,-5,4,4,2,-3,-2,-1,1,9,-2,-1,-3]],[[-7,-6,-10,-9,-9,4,10,-3,-3,9,1,9,-5],[-7,3,9,-6,4,-5,-4,7,-8,-2,-1,5,8],[-1,-5,9,5,2,3,-8,-6,4,-9,-8,10,-9]],[[6,3,9,-2,-10,-6,4,-10,6,5,-3,3,-10],[1,2,2,-4,-3,-9,-7,5,-6,1,-7,-4,-10],[1,-2,-6,-2,-5,-8,7,3,-8,7,-6,-4,10]],[[4,6,-2,-9,10,9,6,4,2,7,9,6,1],[-10,-8,8,6,-8,-1,-5,4,10,-6,4,-8,-3],[-6,-5,-10,-2,5,4,2,-2,-1,-3,-8,5,-7]],[[-4,-9,4,-6,2,3,-8,-4,10,4,-8,-8,2],[-3,-2,3,7,-2,9,9,-6,7,-1,-6,-6,4],[7,-4,7,-3,2,-2,-10,-2,-2,2,3,2,8]],[[-6,8,8,-1,-2,4,7,3,-4,6,4,5,8],[3,2,9,6,8,-5,-10,7,-4,7,8,2,7],[3,-2,-2,2,8,1,-1,10,8,-4,-4,-3,-10]],[[2,4,8,3,-8,-6,-8,4,-8,8,2,10,5],[8,2,-9,-8,8,-10,-7,-6,1,-5,3,1,-3],[-9,10,-4,-2,-6,5,-3,9,3,-6,-9,3,3]],[[7,-9,-8,3,4,8,-6,-6,8,3,5,-5,7],[7,6,-7,-5,-5,-7,-2,-8,9,-3,-6,-6,-8],[6,-3,7,-2,1,8,-8,-10,-3,-7,1,4,1]],[[-6,-10,-10,-8,-9,3,8,3,-2,10,5,2,-4],[-8,-10,-8,-7,-1,-5,10,-5,-5,8,2,-8,-2],[8,7,-1,-10,-7,7,7,3,-10,5,-7,-2,-10]],[[-5,-7,-3,10,-6,-6,-2,-5,7,9,-5,2,-6],[-2,-1,3,3,2,-4,6,3,-6,-3,4,-6,-9],[-3,1,-7,8,-7,6,10,8,2,3,5,10,4]],[[-7,10,1,6,-3,-7,3,-1,-6,8,-6,-5,4],[10,-8,-2,5,3,-5,8,-2,-10,-8,5,-3,-7],[1,-8,4,1,-4,-8,-4,7,-10,7,9,-8,-2]]], dtype = "uint32")#candidate|1958|(16, 3, 13)|const|uint32
bop_1959 = relay.subtract(var_1957.astype('uint32'), relay.reshape(const_1958.astype('uint32'), relay.shape_of(var_1957))) # shape=(16, 3, 13)
bop_1973 = relay.logical_or(bop_1959.astype('bool'), relay.reshape(const_1958.astype('bool'), relay.shape_of(bop_1959))) # shape=(16, 3, 13)
bop_1976 = relay.less(bop_1973.astype('bool'), relay.reshape(bop_1959.astype('bool'), relay.shape_of(bop_1973))) # shape=(16, 3, 13)
output = bop_1976
output2 = bop_1976
F = relay.Function([var_1957,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1957,], output2)
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
input_1957= np.array([[[7,2,-7,10,-1,-4,-1,-1,-10,10,1,-9,9],[2,8,-2,5,8,6,-6,10,8,-6,8,-9,6],[-4,-8,-8,-5,-6,-1,10,3,-5,5,-2,7,-3]],[[6,5,1,6,-2,-2,3,5,-10,-1,-1,5,-4],[-2,-5,-9,-6,-4,6,8,8,-7,9,3,7,2],[4,5,-4,-10,1,-3,-6,5,1,-1,5,-8,-10]],[[-2,6,-6,-3,8,-7,8,8,4,-8,-5,4,6],[6,4,6,4,3,-10,-10,-4,7,-4,3,8,-10],[-10,10,6,1,-1,-9,9,-4,8,7,2,4,-10]],[[5,8,-4,-2,4,2,7,4,-10,-7,10,-10,-7],[-5,-10,6,7,-3,-2,-7,2,-4,3,-6,-9,-9],[8,9,4,6,1,3,6,10,10,4,7,2,8]],[[-2,3,-7,3,-6,-10,-2,4,10,1,10,7,5],[4,3,7,8,-9,1,-2,8,9,4,7,-5,-6],[-4,-1,-7,8,6,5,7,2,3,-5,6,5,1]],[[-6,2,3,4,-7,-8,9,2,9,-8,-7,5,-6],[-8,-2,2,-7,-9,10,-2,-4,-2,6,9,-3,-8],[-9,-4,3,3,3,3,-10,10,-8,7,-5,-7,4]],[[8,-4,-10,9,-6,-10,-8,-6,2,-5,-8,-7,5],[5,-2,-9,-8,9,3,-4,10,2,6,1,6,2],[-9,10,-10,-7,5,5,8,-1,-1,4,10,1,3]],[[-4,10,-2,5,-4,-6,7,8,3,5,-8,3,-10],[-9,-9,-7,-1,10,-5,-5,10,-9,-3,1,2,3],[4,4,-9,9,5,-2,10,-2,-7,-3,10,-5,6]],[[10,2,9,-7,10,-8,-3,-4,1,-9,10,-2,4],[-2,5,-3,-7,6,10,1,-1,-10,1,-1,9,-7],[-5,-9,-3,8,-10,-4,-7,-5,6,-9,-4,-3,-5]],[[9,-6,1,7,10,3,-9,10,-2,-4,4,-4,8],[3,2,8,2,-8,-5,6,-5,5,4,7,-2,4],[2,-8,7,-4,1,-10,5,-7,5,6,-2,5,-6]],[[-3,-5,6,-6,-8,8,-2,5,2,-4,-10,2,-10],[-9,-2,1,-5,2,-4,6,9,-5,4,5,-7,7],[-7,-5,5,5,-9,3,-1,-5,-1,-8,-8,-3,2]],[[-8,-6,-6,4,-3,-5,8,-3,8,-4,-10,6,9],[-10,6,-2,-10,-9,-5,-10,-4,5,7,-1,-1,-9],[-3,-8,1,-5,-7,-1,-7,-3,-5,-9,8,-1,1]],[[1,8,-4,9,3,9,6,5,4,10,-4,-10,4],[-8,-4,-8,10,7,6,5,5,-7,1,10,6,5],[-7,-5,-5,-7,-2,4,-10,6,-6,-7,-9,1,10]],[[7,1,6,-7,8,-5,-10,9,-7,-4,9,7,8],[2,-6,-1,8,7,2,6,5,-7,8,-9,-5,9],[-3,-9,2,2,-6,5,9,-10,5,2,-9,6,-3]],[[2,10,-3,10,5,7,-4,-4,-7,-8,-3,2,4],[4,-7,-5,-8,-2,7,5,6,1,3,8,8,4],[10,9,-7,1,10,5,-10,-9,1,-1,5,6,7]],[[9,-9,-7,1,2,-6,2,-1,-6,-5,10,-10,-10],[2,5,9,5,7,6,5,-5,-3,9,-5,-8,3],[-4,-10,-9,10,4,-8,-7,8,-2,-3,5,4,-3]]], dtype='uint32')
module1.set_input('var_1957', input_1957)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1957, )
res3 = intrp3.evaluate()(input_1957, )
res4 = intrp4.evaluate()(input_1957, )
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
module5.set_input('var_1957', input_1957)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1957, )
res7 = intrp7.evaluate()(input_1957, )
res8 = intrp8.evaluate()(input_1957, )
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
module9.set_input('var_1957', input_1957)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1957, )
res11 = intrp11.evaluate()(input_1957, )
res12 = intrp12.evaluate()(input_1957, )
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
module13.set_input('var_1957', input_1957)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1957, )
res15 = intrp15.evaluate()(input_1957, )
res16 = intrp16.evaluate()(input_1957, )
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
module17.set_input('var_1957', input_1957)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1957, )
res19 = intrp19.evaluate()(input_1957, )
res20 = intrp20.evaluate()(input_1957, )
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
module21.set_input('var_1957', input_1957)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1957, )
res23 = intrp23.evaluate()(input_1957, )
res24 = intrp24.evaluate()(input_1957, )
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

'''19: TVMFuncCall
18: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
17: tvm::transform::Pass::operator()(tvm::IRModule) const
16: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
14: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
13: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
12: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
11: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
10: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
9: _ZNSt17_Function_handlerIFSt10sha
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''