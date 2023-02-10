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
var_50 = relay.var("var_50", dtype = "float32", shape = (7, 1, 9))#candidate|50|(7, 1, 9)|var|float32
uop_51 = relay.sinh(var_50.astype('float32')) # shape=(7, 1, 9)
bop_54 = relay.less_equal(uop_51.astype('bool'), relay.reshape(var_50.astype('bool'), relay.shape_of(uop_51))) # shape=(7, 1, 9)
bop_58 = relay.bitwise_xor(var_50.astype('int64'), relay.reshape(uop_51.astype('int64'), relay.shape_of(var_50))) # shape=(7, 1, 9)
bop_62 = relay.subtract(uop_51.astype('uint16'), relay.reshape(bop_58.astype('uint16'), relay.shape_of(uop_51))) # shape=(7, 1, 9)
uop_67 = relay.asin(bop_62.astype('float64')) # shape=(7, 1, 9)
bop_71 = relay.logical_xor(bop_54.astype('uint8'), relay.reshape(bop_58.astype('uint8'), relay.shape_of(bop_54))) # shape=(7, 1, 9)
uop_75 = relay.erf(uop_67.astype('float32')) # shape=(7, 1, 9)
var_78 = relay.var("var_78", dtype = "float32", shape = (7, 2, 9))#candidate|78|(7, 2, 9)|var|float32
bop_79 = relay.greater_equal(uop_51.astype('bool'), var_78.astype('bool')) # shape=(7, 2, 9)
output = relay.Tuple([bop_71,uop_75,bop_79,])
output2 = relay.Tuple([bop_71,uop_75,bop_79,])
func_82 = relay.Function([var_50,var_78,], output)
mod['func_82'] = func_82
mod = relay.transform.InferType()(mod)
var_83 = relay.var("var_83", dtype = "float32", shape = (7, 1, 9))#candidate|83|(7, 1, 9)|var|float32
var_84 = relay.var("var_84", dtype = "float32", shape = (7, 2, 9))#candidate|84|(7, 2, 9)|var|float32
output = func_82(var_83,var_84,)
func_85 = relay.Function([var_83,var_84,], output)
mutated_mod['func_85'] = func_85
mutated_mod = relay.transform.InferType()(mutated_mod)
var_126 = relay.var("var_126", dtype = "float64", shape = (3, 3))#candidate|126|(3, 3)|var|float64
const_127 = relay.const([[-3.040588,-0.586851,-9.144047],[6.662859,-0.453949,-4.954258],[-8.700707,6.108468,-3.918701]], dtype = "float64")#candidate|127|(3, 3)|const|float64
bop_128 = relay.multiply(var_126.astype('float64'), relay.reshape(const_127.astype('float64'), relay.shape_of(var_126))) # shape=(3, 3)
output = relay.Tuple([bop_128,])
output2 = relay.Tuple([bop_128,])
func_133 = relay.Function([var_126,], output)
mod['func_133'] = func_133
mod = relay.transform.InferType()(mod)
var_134 = relay.var("var_134", dtype = "float64", shape = (3, 3))#candidate|134|(3, 3)|var|float64
output = func_133(var_134)
func_135 = relay.Function([var_134], output)
mutated_mod['func_135'] = func_135
mutated_mod = relay.transform.InferType()(mutated_mod)
const_150 = relay.const([[-4.241765,5.093996,-8.066145,4.940968,-5.262238,-7.150974,-7.400309,-3.827131,4.067242,1.912638,9.696650,0.167486,-1.277947,2.347762,-2.387049,3.087805]], dtype = "float32")#candidate|150|(1, 16)|const|float32
uop_151 = relay.asin(const_150.astype('float32')) # shape=(1, 16)
uop_163 = relay.acos(uop_151.astype('float32')) # shape=(1, 16)
bop_165 = relay.subtract(const_150.astype('float64'), relay.reshape(uop_163.astype('float64'), relay.shape_of(const_150))) # shape=(1, 16)
output = relay.Tuple([bop_165,])
output2 = relay.Tuple([bop_165,])
func_171 = relay.Function([], output)
mod['func_171'] = func_171
mod = relay.transform.InferType()(mod)
output = func_171()
func_172 = relay.Function([], output)
mutated_mod['func_172'] = func_172
mutated_mod = relay.transform.InferType()(mutated_mod)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_176 = relay.TupleGetItem(func_171_call(), 0)
call_177 = relay.TupleGetItem(func_172_call(), 0)
uop_178 = relay.tan(call_176.astype('float64')) # shape=(1, 16)
uop_180 = relay.tan(call_177.astype('float64')) # shape=(1, 16)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_181 = relay.TupleGetItem(func_171_call(), 0)
call_182 = relay.TupleGetItem(func_172_call(), 0)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_188 = relay.TupleGetItem(func_171_call(), 0)
call_189 = relay.TupleGetItem(func_172_call(), 0)
var_196 = relay.var("var_196", dtype = "float64", shape = (8, 16))#candidate|196|(8, 16)|var|float64
bop_197 = relay.bitwise_and(uop_178.astype('uint32'), var_196.astype('uint32')) # shape=(8, 16)
bop_200 = relay.bitwise_and(uop_180.astype('uint32'), var_196.astype('uint32')) # shape=(8, 16)
const_202 = relay.const([[4.659227,-2.936068,8.967782,2.155438,-9.721515,-1.770231,6.717534,0.504943,6.737151,-5.318044,5.016302,6.988410,9.512574,4.022307,8.183671,-9.766366],[-4.267871,-8.139684,5.473207,-5.864416,6.037738,7.392139,4.982927,-9.679825,1.986318,-5.931752,-9.958782,3.971077,8.845496,-2.045607,0.343953,0.891115],[3.667326,-7.021973,-3.850150,-1.280206,3.879454,-9.051739,-3.587690,6.994205,3.247769,-5.456715,-0.799649,8.930447,-1.873511,-5.412750,-3.937868,-5.204111],[5.407009,-5.058293,4.385601,9.915998,-6.995523,8.816348,8.476352,-7.040935,-3.763255,-6.479347,-3.024011,-7.960658,6.078599,4.338134,4.053148,3.465965],[-6.518950,-3.143536,-7.412445,-4.164934,3.036586,-0.255073,-5.312181,9.700035,8.744824,5.547395,7.771468,0.226861,-4.690884,-9.278013,-0.617996,-8.389941],[-1.344945,-6.259290,0.604447,0.788498,-2.494727,7.252028,3.406136,-1.949524,-4.645722,-8.846564,0.848000,5.626377,-4.571508,5.414321,9.121457,2.031968],[-5.789660,2.988446,-5.674667,-7.734134,-4.555409,-5.237582,-2.436197,-6.957553,-0.072439,-2.920086,-1.344313,-2.869302,-1.905566,9.040257,0.830350,5.830207],[-6.353276,2.201588,0.864155,-1.529624,7.628816,8.312794,6.779294,6.814801,6.268168,4.660690,6.378408,-0.967688,-3.710215,2.641000,5.201377,5.556134]], dtype = "float64")#candidate|202|(8, 16)|const|float64
bop_203 = relay.equal(uop_178.astype('bool'), const_202.astype('bool')) # shape=(8, 16)
bop_206 = relay.equal(uop_180.astype('bool'), const_202.astype('bool')) # shape=(8, 16)
func_82_call = mod.get_global_var('func_82')
func_85_call = mutated_mod.get_global_var('func_85')
const_210 = relay.const([3.900131,1.450763,-3.014796,7.667346,-8.617981,-3.599641,5.482300,-1.624749,5.733324,7.185217,7.468756,-3.572729,-2.210310,1.997725,-4.336727,-0.019753,3.271313,-3.211081,-3.803568,4.718140,-8.096562,4.595731,5.957684,0.465967,0.178097,-8.909625,3.791698,-3.374655,-1.627843,7.462813,8.865444,-3.087779,2.235413,-9.912827,-0.852837,-9.937035,-8.124390,7.796026,6.195034,-5.938237,7.023750,-3.745435,9.386934,-7.171529,-3.709836,-6.177389,-5.729415,-1.209784,-0.190646,2.897879,7.555724,4.899840,4.302377,0.345757,9.549348,9.721199,5.561515,6.945754,0.819789,-6.269211,7.745670,6.635718,5.594055], dtype = "float32")#candidate|210|(63,)|const|float32
const_211 = relay.const([-4.987471,0.801925,6.200694,-5.285996,1.769645,-8.343294,6.144044,-5.563765,-0.661586,-1.357583,-5.740547,-6.208248,1.984813,2.154041,5.995227,0.786828,1.030844,8.303982,5.536599,-1.442276,9.065677,-8.961885,-4.165421,7.993979,-8.082071,0.316686,6.616785,-1.316532,8.102558,6.157264,1.780641,-4.439648,4.315021,-6.390605,-7.673791,-2.923245,4.832182,-9.358955,9.872559,6.100829,9.632403,-4.783641,-4.150979,-0.335302,5.460750,-4.505489,-3.926330,7.915554,-1.010686,1.648132,-3.653069,-3.909223,-6.578628,8.100055,-2.712324,9.558070,5.699152,-6.700227,-5.547895,5.524254,-8.075180,-1.605410,9.016430,3.138132,9.396540,-5.343720,-8.816320,-1.071110,7.989638,-2.462781,-7.267845,9.100237,-7.511706,5.501219,-0.355242,-6.094653,-8.059773,-8.356803,-7.372236,6.347164,-1.467066,-1.896982,-5.419879,3.224380,-0.064477,6.165768,9.597368,0.237554,-4.460654,-1.011993,3.983306,1.250855,0.400620,6.274303,2.653125,-5.629821,-9.582213,-8.850516,1.351936,-1.523835,-2.222637,-0.982619,7.030118,1.854154,-1.647070,-1.110501,-4.933576,7.463564,2.189265,3.826241,5.100729,-2.457066,-7.959388,-9.552132,-3.933672,-2.751386,-1.543037,-9.812599,-5.327495,6.145174,1.823638,-1.305579,6.425673,-2.668439,3.785977,-7.752071], dtype = "float32")#candidate|211|(126,)|const|float32
call_209 = relay.TupleGetItem(func_82_call(relay.reshape(const_210.astype('float32'), [7, 1, 9]), relay.reshape(const_211.astype('float32'), [7, 2, 9]), ), 2)
call_212 = relay.TupleGetItem(func_85_call(relay.reshape(const_210.astype('float32'), [7, 1, 9]), relay.reshape(const_211.astype('float32'), [7, 2, 9]), ), 2)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
var_215 = relay.var("var_215", dtype = "float64", shape = (9,))#candidate|215|(9,)|var|float64
call_214 = relay.TupleGetItem(func_133_call(relay.reshape(var_215.astype('float64'), [3, 3])), 0)
call_216 = relay.TupleGetItem(func_135_call(relay.reshape(var_215.astype('float64'), [3, 3])), 0)
const_217 = relay.const([[[False,False,True,False,True,False,True,False,False],[False,False,False,True,False,True,True,True,False]],[[True,False,True,False,True,True,False,False,True],[False,True,False,False,True,True,True,True,False]],[[True,True,True,True,True,True,True,False,True],[True,True,False,True,False,True,False,False,False]],[[True,False,False,True,False,True,True,False,True],[False,True,False,True,False,True,False,True,False]],[[True,False,False,False,True,True,False,False,True],[True,True,True,False,False,True,False,True,True]],[[False,True,True,True,True,False,True,False,False],[True,False,True,True,True,False,False,True,True]],[[True,False,False,True,True,True,False,True,True],[False,False,True,True,False,False,True,True,True]]], dtype = "bool")#candidate|217|(7, 2, 9)|const|bool
bop_218 = relay.floor_divide(call_209.astype('float64'), relay.reshape(const_217.astype('float64'), relay.shape_of(call_209))) # shape=(7, 2, 9)
bop_221 = relay.floor_divide(call_212.astype('float64'), relay.reshape(const_217.astype('float64'), relay.shape_of(call_212))) # shape=(7, 2, 9)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
call_233 = relay.TupleGetItem(func_133_call(relay.reshape(call_214.astype('float64'), [3, 3])), 0)
call_234 = relay.TupleGetItem(func_135_call(relay.reshape(call_214.astype('float64'), [3, 3])), 0)
bop_240 = relay.floor_mod(bop_203.astype('float32'), call_181.astype('float32')) # shape=(8, 16)
bop_243 = relay.floor_mod(bop_206.astype('float32'), call_182.astype('float32')) # shape=(8, 16)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
call_244 = relay.TupleGetItem(func_133_call(relay.reshape(var_215.astype('float64'), [3, 3])), 0)
call_245 = relay.TupleGetItem(func_135_call(relay.reshape(var_215.astype('float64'), [3, 3])), 0)
output = relay.Tuple([call_188,bop_197,const_210,const_211,call_214,var_215,bop_218,call_233,bop_240,call_244,])
output2 = relay.Tuple([call_189,bop_200,const_210,const_211,call_216,var_215,bop_221,call_234,bop_243,call_245,])
func_251 = relay.Function([var_196,var_215,], output)
mod['func_251'] = func_251
mod = relay.transform.InferType()(mod)
mutated_mod['func_251'] = func_251
mutated_mod = relay.transform.InferType()(mutated_mod)
func_251_call = mutated_mod.get_global_var('func_251')
var_253 = relay.var("var_253", dtype = "float64", shape = (8, 16))#candidate|253|(8, 16)|var|float64
var_254 = relay.var("var_254", dtype = "float64", shape = (9,))#candidate|254|(9,)|var|float64
call_252 = func_251_call(var_253,var_254,)
output = call_252
func_255 = relay.Function([var_253,var_254,], output)
mutated_mod['func_255'] = func_255
mutated_mod = relay.transform.InferType()(mutated_mod)
var_274 = relay.var("var_274", dtype = "float32", shape = (10, 15))#candidate|274|(10, 15)|var|float32
uop_275 = relay.rsqrt(var_274.astype('float32')) # shape=(10, 15)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
var_279 = relay.var("var_279", dtype = "float64", shape = (9,))#candidate|279|(9,)|var|float64
call_278 = relay.TupleGetItem(func_133_call(relay.reshape(var_279.astype('float64'), [3, 3])), 0)
call_280 = relay.TupleGetItem(func_135_call(relay.reshape(var_279.astype('float64'), [3, 3])), 0)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
call_282 = relay.TupleGetItem(func_133_call(relay.reshape(var_279.astype('float64'), [3, 3])), 0)
call_283 = relay.TupleGetItem(func_135_call(relay.reshape(var_279.astype('float64'), [3, 3])), 0)
var_284 = relay.var("var_284", dtype = "float64", shape = (3, 3))#candidate|284|(3, 3)|var|float64
bop_285 = relay.less_equal(call_282.astype('bool'), relay.reshape(var_284.astype('bool'), relay.shape_of(call_282))) # shape=(3, 3)
bop_288 = relay.less_equal(call_283.astype('bool'), relay.reshape(var_284.astype('bool'), relay.shape_of(call_283))) # shape=(3, 3)
const_290 = relay.const([[0.240170,1.404972,-2.905344,2.398791,-3.781430,-9.270613,3.364414,4.907134,8.706979,-8.008943,-3.109732,2.367565,0.150756,-7.976697,-0.537597],[-6.707613,5.692107,7.521668,-9.752236,1.916825,4.044202,0.153931,-7.735068,9.844137,9.330230,-2.801833,0.806379,-9.641148,2.918071,6.895488],[-0.075409,5.089339,8.432140,4.272188,3.538389,-4.551872,6.004184,9.294920,7.637043,9.693547,-2.371609,-8.387322,7.568781,7.024604,-2.519672],[-5.963703,-6.064382,-5.008017,3.495551,7.104497,-8.285834,1.672876,-3.191731,3.352037,9.220962,-2.477849,-2.401626,-4.100103,2.620865,-4.972629],[-6.159381,-6.751863,8.409388,-8.012297,-7.862250,9.189773,7.272876,-8.525159,-8.588190,9.195943,-1.262252,8.185297,6.827575,-5.534475,5.716029],[-1.056495,-1.289619,3.761993,8.667067,-9.507470,2.619144,-6.827474,-4.737189,-3.267823,-8.345277,3.782591,4.828018,-1.471128,-6.479667,2.439322],[7.818190,8.548164,-4.798060,9.669342,-7.769981,-2.315942,-3.527781,-5.539918,2.879764,1.042494,5.529354,8.245449,1.844321,1.364364,-3.740317],[-2.622818,-2.259162,9.472832,-1.759496,-8.039357,-5.450038,-3.874703,8.016883,-4.171314,-0.874432,-9.247251,-0.991579,-3.008663,1.780941,3.777963],[4.633220,-8.741041,0.317883,-1.035027,-4.169040,-6.788529,7.328460,4.507544,5.922863,-6.152034,4.809596,7.851142,-8.593327,-7.312514,2.782392],[5.665805,7.692728,-2.568692,-5.449958,-1.507489,4.728031,6.455445,7.527062,3.659497,-0.436526,-9.392833,4.745850,0.792687,-8.325246,1.776821]], dtype = "float32")#candidate|290|(10, 15)|const|float32
bop_291 = relay.floor_mod(uop_275.astype('float64'), relay.reshape(const_290.astype('float64'), relay.shape_of(uop_275))) # shape=(10, 15)
uop_296 = relay.exp(uop_275.astype('float64')) # shape=(10, 15)
bop_299 = relay.floor_divide(uop_296.astype('float64'), relay.reshape(bop_291.astype('float64'), relay.shape_of(uop_296))) # shape=(10, 15)
uop_307 = relay.acosh(bop_299.astype('float64')) # shape=(10, 15)
uop_309 = relay.tan(uop_307.astype('float64')) # shape=(10, 15)
func_82_call = mod.get_global_var('func_82')
func_85_call = mutated_mod.get_global_var('func_85')
var_312 = relay.var("var_312", dtype = "float32", shape = (63,))#candidate|312|(63,)|var|float32
const_313 = relay.const([-0.067429,1.572463,-4.463155,-0.008625,8.441315,-4.763384,3.342369,8.839712,-8.154344,3.523906,-0.878937,-3.084134,4.171324,-7.563511,-0.754901,9.868334,-7.455505,5.123038,8.955561,-9.572099,5.554919,9.539215,4.926038,-8.249779,-8.471078,-0.445481,-2.743932,6.638705,-0.556622,-8.825930,2.209328,7.821985,9.028965,-7.869475,0.690429,2.223664,3.684170,3.895442,7.663605,5.444310,6.793011,8.903392,-9.260922,-6.937792,-5.012501,2.444045,-5.750937,-1.912391,-9.134004,9.594693,2.957329,-9.575077,5.461102,-0.698694,-4.118515,-9.009311,2.333635,6.422527,-2.138235,-7.570322,2.564673,0.217301,3.762543,6.024317,-7.265721,9.175102,4.956372,5.641730,7.840658,0.961205,1.958945,1.760987,2.273227,-9.475473,-6.509190,2.913799,3.473659,-4.413430,-3.876118,-6.909709,-0.047649,-0.188706,4.238463,-7.547526,4.028878,-4.221927,-5.505090,9.900251,8.337527,4.073220,4.024142,7.439815,7.322154,6.791414,8.489640,4.316093,1.801102,4.435302,1.821413,0.372388,3.263079,4.533320,0.302830,-2.018570,-5.467726,3.648196,-8.838639,-1.626904,-5.055540,5.740418,1.426082,8.903010,-7.996599,-7.514880,-2.625526,0.747194,1.020558,3.902989,2.401330,5.555876,-5.919846,1.826190,3.899466,-6.270554,8.156223,2.067284], dtype = "float32")#candidate|313|(126,)|const|float32
call_311 = relay.TupleGetItem(func_82_call(relay.reshape(var_312.astype('float32'), [7, 1, 9]), relay.reshape(const_313.astype('float32'), [7, 2, 9]), ), 0)
call_314 = relay.TupleGetItem(func_85_call(relay.reshape(var_312.astype('float32'), [7, 1, 9]), relay.reshape(const_313.astype('float32'), [7, 2, 9]), ), 0)
output = relay.Tuple([call_278,var_279,bop_285,uop_309,call_311,var_312,const_313,])
output2 = relay.Tuple([call_280,var_279,bop_288,uop_309,call_314,var_312,const_313,])
func_315 = relay.Function([var_274,var_279,var_284,var_312,], output)
mod['func_315'] = func_315
mod = relay.transform.InferType()(mod)
mutated_mod['func_315'] = func_315
mutated_mod = relay.transform.InferType()(mutated_mod)
func_315_call = mutated_mod.get_global_var('func_315')
var_317 = relay.var("var_317", dtype = "float32", shape = (10, 15))#candidate|317|(10, 15)|var|float32
var_318 = relay.var("var_318", dtype = "float64", shape = (9,))#candidate|318|(9,)|var|float64
var_319 = relay.var("var_319", dtype = "float64", shape = (3, 3))#candidate|319|(3, 3)|var|float64
var_320 = relay.var("var_320", dtype = "float32", shape = (63,))#candidate|320|(63,)|var|float32
call_316 = func_315_call(var_317,var_318,var_319,var_320,)
output = call_316
func_321 = relay.Function([var_317,var_318,var_319,var_320,], output)
mutated_mod['func_321'] = func_321
mutated_mod = relay.transform.InferType()(mutated_mod)
const_331 = relay.const([[[-4.824906,3.396734,-0.336125,3.119959,6.883668,-3.917641,-0.962667,-0.193139],[-6.705004,6.591940,-6.165350,9.431129,7.101568,8.217889,2.872491,-1.372251],[-1.040480,-1.274757,-3.377814,4.948671,6.815677,-1.502974,7.202393,3.834494],[-8.341686,-5.821490,-9.192302,4.769302,2.374595,4.159459,4.153390,8.380859],[4.733524,-5.255795,3.938661,-4.118801,9.831244,3.719899,0.664414,-1.746660],[1.456436,0.820017,3.709169,3.342422,5.707475,6.877793,0.839511,-5.575473],[2.133094,-5.297437,-2.103294,1.837544,0.628000,3.250088,8.736159,-4.345657],[-1.410415,-5.225733,9.950250,-1.983151,1.698138,4.326675,-3.459855,7.059611],[7.032339,1.330269,-9.741501,6.635959,2.580449,8.511620,0.196639,5.969408],[4.692809,7.332074,6.127037,3.059451,8.617566,9.602946,-8.719174,8.412928]],[[-3.853959,5.314426,5.203040,-5.689506,1.441583,8.095748,-0.147952,-7.379130],[5.284039,-0.801991,-2.594565,5.155530,-6.934261,1.748534,4.412230,-8.309447],[0.410366,2.834514,0.452134,4.099438,3.941468,6.326230,8.700738,0.952723],[5.289974,-3.513970,6.769500,6.913945,-0.030058,-5.605562,-8.581513,-6.669066],[-8.740485,-2.971948,3.621269,1.169246,9.762932,-7.889428,-5.827271,4.851704],[3.007620,7.377506,6.545405,8.507803,-6.767385,9.834283,-7.937207,-1.968455],[9.196466,-7.793725,9.823767,0.834779,-0.435698,8.154258,5.183441,-7.499825],[-6.647645,1.863554,2.018141,-1.356959,-7.069157,-0.893304,2.705466,-1.122705],[5.664510,6.784027,-5.994387,2.191007,1.824280,0.199245,9.849664,0.769991],[-6.841585,5.416834,8.310214,2.667254,-4.036817,1.215858,-9.127797,-4.036561]],[[-9.234585,-3.647081,2.447018,-6.940902,2.742190,8.616658,4.498853,-7.481059],[-7.418249,4.172754,0.319512,7.142658,-2.657312,3.107790,-3.625510,6.079791],[-5.634780,7.146804,2.099501,-6.112085,-1.615578,-2.431568,-3.810206,4.342583],[-3.451335,4.703888,0.608017,5.380906,9.016783,-2.989662,1.487308,-9.752160],[-4.967568,9.049345,3.755206,-2.427193,2.100583,-3.904177,9.648266,-2.769557],[-8.875387,-9.068978,7.220467,-6.956035,-8.420158,-7.249130,-7.603642,0.820568],[5.632152,-7.933478,-6.165077,-8.731666,5.611283,-6.075781,-5.160188,1.092202],[-0.145549,-3.889415,0.147693,-2.148108,5.429837,9.786557,5.582530,-1.389935],[8.067360,-1.269672,5.945160,-2.358702,9.278948,-9.823569,5.188569,-5.775656],[9.643164,1.962592,6.825868,-2.986666,2.533623,-8.234227,1.081258,7.369045]],[[-2.009767,-1.104902,-7.183673,-5.676798,5.058822,5.490078,-3.447148,-8.310876],[-1.737233,4.904636,4.312812,-8.298618,6.416071,-7.974821,-6.975487,-6.450410],[3.965568,-3.505526,1.827681,4.170038,4.953725,4.480295,-2.079114,9.085644],[-3.141424,1.285501,0.638308,-3.970867,-6.469998,9.186378,-3.724816,0.470728],[-6.571819,2.968245,8.571128,4.292612,3.272787,-8.766714,-6.118004,-1.604277],[-6.072776,8.085148,1.866168,6.381524,-7.530824,-2.570460,-4.367161,-3.173038],[-2.097958,-9.524215,-7.858326,-0.204900,-4.975550,2.555363,9.088001,8.055882],[5.981104,2.937111,-7.371292,7.671303,8.183046,4.950634,1.806916,-9.178816],[1.634440,8.783655,-2.202283,6.002137,6.300641,-9.853021,-4.580469,-2.046595],[-2.248769,7.108699,-2.818479,9.634190,-4.978873,8.242468,-3.128774,9.179401]]], dtype = "float64")#candidate|331|(4, 10, 8)|const|float64
uop_332 = relay.asinh(const_331.astype('float64')) # shape=(4, 10, 8)
uop_337 = relay.atanh(const_331.astype('float64')) # shape=(4, 10, 8)
uop_340 = relay.acosh(const_331.astype('float32')) # shape=(4, 10, 8)
bop_342 = relay.multiply(const_331.astype('float64'), relay.reshape(uop_332.astype('float64'), relay.shape_of(const_331))) # shape=(4, 10, 8)
func_315_call = mod.get_global_var('func_315')
func_321_call = mutated_mod.get_global_var('func_321')
var_346 = relay.var("var_346", dtype = "float32", shape = (150,))#candidate|346|(150,)|var|float32
const_347 = relay.const([9.919983,2.459944,3.541709,6.571154,-6.078471,-4.506476,0.904175,1.194615,0.238657], dtype = "float64")#candidate|347|(9,)|const|float64
const_348 = relay.const([2.232442,-0.476175,9.200242,6.588780,-1.894354,-8.867536,5.899338,0.264673,5.432935,6.855279,-0.505010,-5.172642,7.131560,-2.988984,-8.667154,-7.839464,-4.526255,-2.923885,3.821126,-5.854052,-0.151103,3.689901,8.335565,1.787751,4.990797,9.228059,-5.460708,0.800769,4.786247,-3.084570,-1.380727,9.748746,4.818626,-3.160390,3.536551,-1.966559,3.277410,7.530430,-8.375422,-2.015279,0.846317,6.045350,4.460400,2.218510,-3.711391,-1.867434,6.897738,7.710027,5.926321,9.846955,6.959891,-3.546211,7.486609,8.789201,-5.450230,8.283763,4.230726,7.326014,-7.341131,-1.269339,-6.480756,6.530864,-9.411385], dtype = "float32")#candidate|348|(63,)|const|float32
call_345 = relay.TupleGetItem(func_315_call(relay.reshape(var_346.astype('float32'), [10, 15]), relay.reshape(const_347.astype('float64'), [9,]), relay.reshape(const_347.astype('float64'), [3, 3]), relay.reshape(const_348.astype('float32'), [63,]), ), 0)
call_349 = relay.TupleGetItem(func_321_call(relay.reshape(var_346.astype('float32'), [10, 15]), relay.reshape(const_347.astype('float64'), [9,]), relay.reshape(const_347.astype('float64'), [3, 3]), relay.reshape(const_348.astype('float32'), [63,]), ), 0)
uop_361 = relay.sqrt(var_346.astype('float32')) # shape=(150,)
uop_372 = relay.log2(uop_332.astype('float64')) # shape=(4, 10, 8)
func_82_call = mod.get_global_var('func_82')
func_85_call = mutated_mod.get_global_var('func_85')
var_381 = relay.var("var_381", dtype = "float32", shape = (126,))#candidate|381|(126,)|var|float32
call_380 = relay.TupleGetItem(func_82_call(relay.reshape(const_348.astype('float32'), [7, 1, 9]), relay.reshape(var_381.astype('float32'), [7, 2, 9]), ), 0)
call_382 = relay.TupleGetItem(func_85_call(relay.reshape(const_348.astype('float32'), [7, 1, 9]), relay.reshape(var_381.astype('float32'), [7, 2, 9]), ), 0)
bop_389 = relay.logical_or(uop_372.astype('bool'), relay.reshape(uop_337.astype('bool'), relay.shape_of(uop_372))) # shape=(4, 10, 8)
bop_396 = relay.greater(const_331.astype('bool'), relay.reshape(uop_372.astype('bool'), relay.shape_of(const_331))) # shape=(4, 10, 8)
output = relay.Tuple([uop_340,bop_342,call_345,const_347,const_348,uop_361,call_380,var_381,bop_389,bop_396,])
output2 = relay.Tuple([uop_340,bop_342,call_349,const_347,const_348,uop_361,call_382,var_381,bop_389,bop_396,])
func_405 = relay.Function([var_346,var_381,], output)
mod['func_405'] = func_405
mod = relay.transform.InferType()(mod)
mutated_mod['func_405'] = func_405
mutated_mod = relay.transform.InferType()(mutated_mod)
func_405_call = mutated_mod.get_global_var('func_405')
var_407 = relay.var("var_407", dtype = "float32", shape = (150,))#candidate|407|(150,)|var|float32
var_408 = relay.var("var_408", dtype = "float32", shape = (126,))#candidate|408|(126,)|var|float32
call_406 = func_405_call(var_407,var_408,)
output = call_406
func_409 = relay.Function([var_407,var_408,], output)
mutated_mod['func_409'] = func_409
mutated_mod = relay.transform.InferType()(mutated_mod)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_427 = relay.TupleGetItem(func_171_call(), 0)
call_428 = relay.TupleGetItem(func_172_call(), 0)
var_430 = relay.var("var_430", dtype = "float64", shape = (7, 16))#candidate|430|(7, 16)|var|float64
bop_431 = relay.less_equal(call_427.astype('bool'), var_430.astype('bool')) # shape=(7, 16)
bop_434 = relay.less_equal(call_428.astype('bool'), var_430.astype('bool')) # shape=(7, 16)
func_315_call = mod.get_global_var('func_315')
func_321_call = mutated_mod.get_global_var('func_321')
var_438 = relay.var("var_438", dtype = "float32", shape = (5, 30))#candidate|438|(5, 30)|var|float32
var_439 = relay.var("var_439", dtype = "float64", shape = (9,))#candidate|439|(9,)|var|float64
var_440 = relay.var("var_440", dtype = "float32", shape = (63, 1))#candidate|440|(63, 1)|var|float32
call_437 = relay.TupleGetItem(func_315_call(relay.reshape(var_438.astype('float32'), [10, 15]), relay.reshape(var_439.astype('float64'), [9,]), relay.reshape(var_439.astype('float64'), [3, 3]), relay.reshape(var_440.astype('float32'), [63,]), ), 2)
call_441 = relay.TupleGetItem(func_321_call(relay.reshape(var_438.astype('float32'), [10, 15]), relay.reshape(var_439.astype('float64'), [9,]), relay.reshape(var_439.astype('float64'), [3, 3]), relay.reshape(var_440.astype('float32'), [63,]), ), 2)
uop_446 = relay.exp(bop_431.astype('float64')) # shape=(7, 16)
uop_448 = relay.exp(bop_434.astype('float64')) # shape=(7, 16)
output = relay.Tuple([call_437,var_438,var_439,var_440,uop_446,])
output2 = relay.Tuple([call_441,var_438,var_439,var_440,uop_448,])
func_472 = relay.Function([var_430,var_438,var_439,var_440,], output)
mod['func_472'] = func_472
mod = relay.transform.InferType()(mod)
mutated_mod['func_472'] = func_472
mutated_mod = relay.transform.InferType()(mutated_mod)
func_472_call = mutated_mod.get_global_var('func_472')
var_474 = relay.var("var_474", dtype = "float64", shape = (7, 16))#candidate|474|(7, 16)|var|float64
var_475 = relay.var("var_475", dtype = "float32", shape = (5, 30))#candidate|475|(5, 30)|var|float32
var_476 = relay.var("var_476", dtype = "float64", shape = (9,))#candidate|476|(9,)|var|float64
var_477 = relay.var("var_477", dtype = "float32", shape = (63, 1))#candidate|477|(63, 1)|var|float32
call_473 = func_472_call(var_474,var_475,var_476,var_477,)
output = call_473
func_478 = relay.Function([var_474,var_475,var_476,var_477,], output)
mutated_mod['func_478'] = func_478
mutated_mod = relay.transform.InferType()(mutated_mod)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_497 = relay.TupleGetItem(func_171_call(), 0)
call_498 = relay.TupleGetItem(func_172_call(), 0)
output = call_497
output2 = call_498
func_499 = relay.Function([], output)
mod['func_499'] = func_499
mod = relay.transform.InferType()(mod)
mutated_mod['func_499'] = func_499
mutated_mod = relay.transform.InferType()(mutated_mod)
func_499_call = mutated_mod.get_global_var('func_499')
call_500 = func_499_call()
output = call_500
func_501 = relay.Function([], output)
mutated_mod['func_501'] = func_501
mutated_mod = relay.transform.InferType()(mutated_mod)
func_499_call = mod.get_global_var('func_499')
func_501_call = mutated_mod.get_global_var('func_501')
call_518 = func_499_call()
call_519 = func_499_call()
var_521 = relay.var("var_521", dtype = "float64", shape = (3, 16))#candidate|521|(3, 16)|var|float64
bop_522 = relay.less_equal(call_518.astype('bool'), var_521.astype('bool')) # shape=(3, 16)
bop_525 = relay.less_equal(call_519.astype('bool'), var_521.astype('bool')) # shape=(3, 16)
bop_535 = relay.power(var_521.astype('float32'), relay.reshape(bop_522.astype('float32'), relay.shape_of(var_521))) # shape=(3, 16)
bop_538 = relay.power(var_521.astype('float32'), relay.reshape(bop_525.astype('float32'), relay.shape_of(var_521))) # shape=(3, 16)
output = bop_535
output2 = bop_538
func_544 = relay.Function([var_521,], output)
mod['func_544'] = func_544
mod = relay.transform.InferType()(mod)
mutated_mod['func_544'] = func_544
mutated_mod = relay.transform.InferType()(mutated_mod)
var_545 = relay.var("var_545", dtype = "float64", shape = (3, 16))#candidate|545|(3, 16)|var|float64
func_544_call = mutated_mod.get_global_var('func_544')
call_546 = func_544_call(var_545)
output = call_546
func_547 = relay.Function([var_545], output)
mutated_mod['func_547'] = func_547
mutated_mod = relay.transform.InferType()(mutated_mod)
const_562 = relay.const([[2.024055,0.321770,-7.085303,-6.658718,-5.039685,8.190879,-2.049343,6.815107,1.512709,-5.792945],[7.140558,-6.070526,-5.074935,-1.742787,6.820344,4.940780,8.753274,3.323257,4.661366,8.586612],[3.582408,-2.494114,-7.523358,-3.194416,4.457495,-6.184521,9.838819,6.483865,-4.485540,-8.930283],[0.723655,7.099862,1.546043,-0.059386,-0.633265,-5.032674,0.816535,5.595674,-4.719069,0.388539],[3.535220,-9.616992,-3.301458,-5.691650,0.685441,4.520242,-4.006619,4.364262,2.816847,-2.989574],[6.518528,2.353459,-3.440477,-9.303691,-6.068898,-2.757037,3.768344,-5.197486,3.043181,4.557821],[0.859308,6.793520,9.899462,4.372364,8.915586,-9.584147,-0.390384,1.134287,-0.370203,5.532660],[6.928958,-7.289874,-1.582446,0.440647,5.355130,-9.015775,8.761539,4.921198,-7.438177,5.308936],[-8.163118,-2.588506,9.024845,7.465100,-7.337883,-5.743547,3.737177,-9.479693,6.508184,1.502700]], dtype = "float32")#candidate|562|(9, 10)|const|float32
uop_563 = relay.acosh(const_562.astype('float32')) # shape=(9, 10)
uop_565 = relay.log(uop_563.astype('float64')) # shape=(9, 10)
output = uop_565
output2 = uop_565
func_573 = relay.Function([], output)
mod['func_573'] = func_573
mod = relay.transform.InferType()(mod)
output = func_573()
func_574 = relay.Function([], output)
mutated_mod['func_574'] = func_574
mutated_mod = relay.transform.InferType()(mutated_mod)
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_581 = func_573_call()
call_582 = func_573_call()
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_588 = relay.TupleGetItem(func_171_call(), 0)
call_589 = relay.TupleGetItem(func_172_call(), 0)
output = relay.Tuple([call_581,call_588,])
output2 = relay.Tuple([call_582,call_589,])
func_597 = relay.Function([], output)
mod['func_597'] = func_597
mod = relay.transform.InferType()(mod)
mutated_mod['func_597'] = func_597
mutated_mod = relay.transform.InferType()(mutated_mod)
func_597_call = mutated_mod.get_global_var('func_597')
call_598 = func_597_call()
output = call_598
func_599 = relay.Function([], output)
mutated_mod['func_599'] = func_599
mutated_mod = relay.transform.InferType()(mutated_mod)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_628 = relay.TupleGetItem(func_171_call(), 0)
call_629 = relay.TupleGetItem(func_172_call(), 0)
uop_634 = relay.atanh(call_628.astype('float64')) # shape=(1, 16)
uop_636 = relay.atanh(call_629.astype('float64')) # shape=(1, 16)
bop_641 = relay.bitwise_or(uop_634.astype('int64'), relay.reshape(call_628.astype('int64'), relay.shape_of(uop_634))) # shape=(1, 16)
bop_644 = relay.bitwise_or(uop_636.astype('int64'), relay.reshape(call_629.astype('int64'), relay.shape_of(uop_636))) # shape=(1, 16)
bop_654 = relay.multiply(bop_641.astype('uint64'), relay.reshape(uop_634.astype('uint64'), relay.shape_of(bop_641))) # shape=(1, 16)
bop_657 = relay.multiply(bop_644.astype('uint64'), relay.reshape(uop_636.astype('uint64'), relay.shape_of(bop_644))) # shape=(1, 16)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_658 = relay.TupleGetItem(func_171_call(), 0)
call_659 = relay.TupleGetItem(func_172_call(), 0)
func_315_call = mod.get_global_var('func_315')
func_321_call = mutated_mod.get_global_var('func_321')
const_661 = relay.const([7.468047,-5.052224,0.750739,9.091545,7.926061,9.559094,-2.583303,2.125168,7.708744,-3.901263,6.909869,7.282449,1.036938,1.153323,9.318080,-9.183233,6.596043,-3.342319,4.121191,8.554391,2.337110,-4.507583,5.367611,-0.838521,-7.895697,5.836878,3.008050,4.724455,5.763392,1.729005,5.276431,0.675362,4.753798,4.015824,-5.197476,5.250392,1.213257,6.870922,7.669559,-5.210075,-3.511335,5.020263,-1.585817,-3.337755,4.582872,-1.658188,-6.920695,-8.797878,4.304200,-7.409885,-7.404868,-0.261610,6.107816,-6.033476,5.917331,-7.950064,9.687048,9.653790,1.751698,7.904146,2.570593,9.368726,-6.104254,6.980234,5.823505,5.977323,7.144914,2.010663,4.296224,1.494329,-8.898635,-6.082983,-2.640642,1.186032,-8.908562,-4.497069,-1.735175,7.563941,5.926799,5.051333,-6.238480,-3.432805,-6.655628,-0.083102,-5.314888,-0.250301,-2.150020,-9.901610,-9.671511,-8.850444,-1.286644,-0.073963,-7.084017,3.359077,6.853516,-2.297008,7.064224,8.706370,-9.822046,8.207737,-9.377474,-6.493257,4.109892,9.886685,-9.806381,-9.169828,-8.692377,-3.973989,-8.843162,8.007993,-4.175161,5.137608,-5.340369,0.800638,-2.272092,2.054934,-1.881279,4.708534,-6.898918,1.546990,-4.421993,-5.200902,-2.232642,7.991009,-3.924089,-4.203360,-2.226169,2.678399,9.474854,-9.484112,5.884361,-2.019454,3.734479,3.440631,-3.907594,-8.374399,-6.093927,8.529094,3.540664,7.763840,3.000104,-6.992536,2.121849,9.815477,-5.668755,8.887582,5.926582,5.908355,4.279883,0.942391], dtype = "float32")#candidate|661|(150,)|const|float32
const_662 = relay.const([-1.747784,4.648876,0.544393,2.945385,-9.519506,8.934181,9.769983,-9.158535,-6.036767], dtype = "float64")#candidate|662|(9,)|const|float64
const_663 = relay.const([-7.105485,-9.149473,1.589875,-0.773885,5.842074,-0.530588,-2.730669,0.139716,9.037200,0.193341,6.754937,8.668999,9.749084,-0.221196,4.047728,4.876845,-2.157964,-3.740123,0.485563,3.460254,-9.301663,-4.702484,1.427960,4.287776,-2.728892,-5.148026,5.288876,0.206353,0.741916,5.614739,-8.469449,-6.324571,2.785442,-7.899489,5.908014,1.895714,2.487651,4.506338,9.222415,-8.231467,2.876145,-3.577793,-5.938537,0.151828,2.680692,-8.411804,4.024321,-0.320431,0.221248,-2.666252,-1.622065,-5.402964,8.666496,-1.475388,2.138494,-8.439415,6.588678,5.804249,-3.672272,6.032061,2.856239,6.223245,7.469298], dtype = "float32")#candidate|663|(63,)|const|float32
call_660 = relay.TupleGetItem(func_315_call(relay.reshape(const_661.astype('float32'), [10, 15]), relay.reshape(const_662.astype('float64'), [9,]), relay.reshape(const_662.astype('float64'), [3, 3]), relay.reshape(const_663.astype('float32'), [63,]), ), 3)
call_664 = relay.TupleGetItem(func_321_call(relay.reshape(const_661.astype('float32'), [10, 15]), relay.reshape(const_662.astype('float64'), [9,]), relay.reshape(const_662.astype('float64'), [3, 3]), relay.reshape(const_663.astype('float32'), [63,]), ), 3)
func_315_call = mod.get_global_var('func_315')
func_321_call = mutated_mod.get_global_var('func_321')
call_676 = relay.TupleGetItem(func_315_call(relay.reshape(const_661.astype('float32'), [10, 15]), relay.reshape(const_662.astype('float64'), [9,]), relay.reshape(const_662.astype('float64'), [3, 3]), relay.reshape(const_663.astype('float32'), [63,]), ), 5)
call_677 = relay.TupleGetItem(func_321_call(relay.reshape(const_661.astype('float32'), [10, 15]), relay.reshape(const_662.astype('float64'), [9,]), relay.reshape(const_662.astype('float64'), [3, 3]), relay.reshape(const_663.astype('float32'), [63,]), ), 5)
bop_683 = relay.bitwise_xor(bop_641.astype('uint64'), relay.reshape(uop_634.astype('uint64'), relay.shape_of(bop_641))) # shape=(1, 16)
bop_686 = relay.bitwise_xor(bop_644.astype('uint64'), relay.reshape(uop_636.astype('uint64'), relay.shape_of(bop_644))) # shape=(1, 16)
output = relay.Tuple([bop_654,call_658,call_660,const_661,const_662,const_663,call_676,bop_683,])
output2 = relay.Tuple([bop_657,call_659,call_664,const_661,const_662,const_663,call_677,bop_686,])
func_691 = relay.Function([], output)
mod['func_691'] = func_691
mod = relay.transform.InferType()(mod)
output = func_691()
func_692 = relay.Function([], output)
mutated_mod['func_692'] = func_692
mutated_mod = relay.transform.InferType()(mutated_mod)
var_706 = relay.var("var_706", dtype = "float64", shape = (2, 4))#candidate|706|(2, 4)|var|float64
uop_707 = relay.sin(var_706.astype('float64')) # shape=(2, 4)
func_691_call = mod.get_global_var('func_691')
func_692_call = mutated_mod.get_global_var('func_692')
call_710 = relay.TupleGetItem(func_691_call(), 2)
call_711 = relay.TupleGetItem(func_692_call(), 2)
uop_715 = relay.asin(uop_707.astype('float32')) # shape=(2, 4)
bop_721 = relay.floor_divide(uop_707.astype('float64'), relay.reshape(uop_715.astype('float64'), relay.shape_of(uop_707))) # shape=(2, 4)
output = relay.Tuple([call_710,bop_721,])
output2 = relay.Tuple([call_711,bop_721,])
func_730 = relay.Function([var_706,], output)
mod['func_730'] = func_730
mod = relay.transform.InferType()(mod)
mutated_mod['func_730'] = func_730
mutated_mod = relay.transform.InferType()(mutated_mod)
var_731 = relay.var("var_731", dtype = "float64", shape = (2, 4))#candidate|731|(2, 4)|var|float64
func_730_call = mutated_mod.get_global_var('func_730')
call_732 = func_730_call(var_731)
output = call_732
func_733 = relay.Function([var_731], output)
mutated_mod['func_733'] = func_733
mutated_mod = relay.transform.InferType()(mutated_mod)
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_738 = func_573_call()
call_739 = func_573_call()
const_742 = relay.const([[-0.673946,1.833042,-0.583010,8.541647,-0.835536,6.110034,-6.250019,-5.459091,-9.489072,6.122175],[-3.378144,4.754113,-4.985269,3.658350,0.860494,3.278158,7.030065,-7.911549,2.960362,-1.544898],[-5.693872,-0.897472,-3.576530,0.677346,-2.637075,2.078435,2.039211,3.416810,-9.134386,-3.878007],[2.760348,2.644966,-3.612883,-8.649902,1.334991,-3.720192,-7.693188,4.223708,1.513248,1.619884],[8.240301,8.245578,3.640937,-5.877126,2.278261,7.306232,1.317074,5.440868,-0.664464,6.571609],[8.060347,-5.695252,-4.027017,-2.077649,-8.322610,2.488638,-4.955627,-6.658879,2.360122,2.641329],[6.428144,1.620319,-2.461900,0.567117,-5.406566,-5.482750,-7.457366,7.736015,-8.608248,-2.406307],[4.796052,-3.195260,-8.352512,8.492530,-0.260689,9.964702,-4.481577,-3.464407,2.481633,6.591645],[-7.129845,4.254337,-5.731387,1.668491,-6.828350,-4.043542,-2.364423,-1.772043,0.561078,-4.429044]], dtype = "float64")#candidate|742|(9, 10)|const|float64
bop_743 = relay.floor_mod(call_738.astype('float32'), relay.reshape(const_742.astype('float32'), relay.shape_of(call_738))) # shape=(9, 10)
bop_746 = relay.floor_mod(call_739.astype('float32'), relay.reshape(const_742.astype('float32'), relay.shape_of(call_739))) # shape=(9, 10)
var_750 = relay.var("var_750", dtype = "float64", shape = (9, 10))#candidate|750|(9, 10)|var|float64
bop_751 = relay.logical_and(call_738.astype('bool'), relay.reshape(var_750.astype('bool'), relay.shape_of(call_738))) # shape=(9, 10)
bop_754 = relay.logical_and(call_739.astype('bool'), relay.reshape(var_750.astype('bool'), relay.shape_of(call_739))) # shape=(9, 10)
output = relay.Tuple([bop_743,bop_751,])
output2 = relay.Tuple([bop_746,bop_754,])
func_760 = relay.Function([var_750,], output)
mod['func_760'] = func_760
mod = relay.transform.InferType()(mod)
mutated_mod['func_760'] = func_760
mutated_mod = relay.transform.InferType()(mutated_mod)
var_761 = relay.var("var_761", dtype = "float64", shape = (9, 10))#candidate|761|(9, 10)|var|float64
func_760_call = mutated_mod.get_global_var('func_760')
call_762 = func_760_call(var_761)
output = call_762
func_763 = relay.Function([var_761], output)
mutated_mod['func_763'] = func_763
mutated_mod = relay.transform.InferType()(mutated_mod)
var_777 = relay.var("var_777", dtype = "float64", shape = (7, 5))#candidate|777|(7, 5)|var|float64
uop_778 = relay.asinh(var_777.astype('float64')) # shape=(7, 5)
output = relay.Tuple([uop_778,])
output2 = relay.Tuple([uop_778,])
func_786 = relay.Function([var_777,], output)
mod['func_786'] = func_786
mod = relay.transform.InferType()(mod)
var_787 = relay.var("var_787", dtype = "float64", shape = (7, 5))#candidate|787|(7, 5)|var|float64
output = func_786(var_787)
func_788 = relay.Function([var_787], output)
mutated_mod['func_788'] = func_788
mutated_mod = relay.transform.InferType()(mutated_mod)
func_597_call = mod.get_global_var('func_597')
func_599_call = mutated_mod.get_global_var('func_599')
call_796 = relay.TupleGetItem(func_597_call(), 0)
call_797 = relay.TupleGetItem(func_599_call(), 0)
func_499_call = mod.get_global_var('func_499')
func_501_call = mutated_mod.get_global_var('func_501')
call_801 = func_499_call()
call_802 = func_499_call()
uop_803 = relay.sin(call_801.astype('float32')) # shape=(1, 16)
uop_805 = relay.sin(call_802.astype('float32')) # shape=(1, 16)
bop_809 = relay.add(uop_803.astype('float32'), relay.reshape(call_801.astype('float32'), relay.shape_of(uop_803))) # shape=(1, 16)
bop_812 = relay.add(uop_805.astype('float32'), relay.reshape(call_802.astype('float32'), relay.shape_of(uop_805))) # shape=(1, 16)
output = relay.Tuple([call_796,bop_809,])
output2 = relay.Tuple([call_797,bop_812,])
func_817 = relay.Function([], output)
mod['func_817'] = func_817
mod = relay.transform.InferType()(mod)
output = func_817()
func_818 = relay.Function([], output)
mutated_mod['func_818'] = func_818
mutated_mod = relay.transform.InferType()(mutated_mod)
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_950 = func_573_call()
call_951 = func_573_call()
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_963 = func_573_call()
call_964 = func_573_call()
func_786_call = mod.get_global_var('func_786')
func_788_call = mutated_mod.get_global_var('func_788')
const_979 = relay.const([[-4.872780,7.546795,-3.711623,1.431122,0.867394,-5.441595,-8.022073,-1.281794,-3.915818,5.455430,-0.331268,-3.471670,8.316198,8.713566,6.359745,-4.437647,-4.275423,7.343620,-7.758812,4.021293,8.202766,-1.983435,-0.952028,0.441796,2.643320,-3.273867,0.682161,9.252479,-2.034674,-2.525094,3.504455,2.007601,-5.204676,-6.854682,-0.046041]], dtype = "float64")#candidate|979|(1, 35)|const|float64
call_978 = relay.TupleGetItem(func_786_call(relay.reshape(const_979.astype('float64'), [7, 5])), 0)
call_980 = relay.TupleGetItem(func_788_call(relay.reshape(const_979.astype('float64'), [7, 5])), 0)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
var_988 = relay.var("var_988", dtype = "float64", shape = (3, 3))#candidate|988|(3, 3)|var|float64
call_987 = relay.TupleGetItem(func_133_call(relay.reshape(var_988.astype('float64'), [3, 3])), 0)
call_989 = relay.TupleGetItem(func_135_call(relay.reshape(var_988.astype('float64'), [3, 3])), 0)
output = relay.Tuple([call_950,call_963,call_978,const_979,call_987,var_988,])
output2 = relay.Tuple([call_951,call_964,call_980,const_979,call_989,var_988,])
func_995 = relay.Function([var_988,], output)
mod['func_995'] = func_995
mod = relay.transform.InferType()(mod)
mutated_mod['func_995'] = func_995
mutated_mod = relay.transform.InferType()(mutated_mod)
var_996 = relay.var("var_996", dtype = "float64", shape = (3, 3))#candidate|996|(3, 3)|var|float64
func_995_call = mutated_mod.get_global_var('func_995')
call_997 = func_995_call(var_996)
output = call_997
func_998 = relay.Function([var_996], output)
mutated_mod['func_998'] = func_998
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1071 = relay.var("var_1071", dtype = "int32", shape = ())#candidate|1071|()|var|int32
var_1072 = relay.var("var_1072", dtype = "int32", shape = (13, 4, 8))#candidate|1072|(13, 4, 8)|var|int32
bop_1073 = relay.right_shift(var_1071.astype('int32'), var_1072.astype('int32')) # shape=(13, 4, 8)
output = bop_1073
output2 = bop_1073
func_1076 = relay.Function([var_1071,var_1072,], output)
mod['func_1076'] = func_1076
mod = relay.transform.InferType()(mod)
mutated_mod['func_1076'] = func_1076
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1076_call = mutated_mod.get_global_var('func_1076')
var_1078 = relay.var("var_1078", dtype = "int32", shape = ())#candidate|1078|()|var|int32
var_1079 = relay.var("var_1079", dtype = "int32", shape = (13, 4, 8))#candidate|1079|(13, 4, 8)|var|int32
call_1077 = func_1076_call(var_1078,var_1079,)
output = call_1077
func_1080 = relay.Function([var_1078,var_1079,], output)
mutated_mod['func_1080'] = func_1080
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1098 = relay.const([[[9.105740,-9.737271,8.383434,9.671738,-6.055291,7.127719,7.053246,8.989109,8.957220,2.532651,5.475403,7.701460,4.287713],[-2.559077,6.942595,7.467261,1.852925,7.651668,-0.603549,-1.666745,9.081005,-4.029648,-9.099929,-2.867189,5.089457,-6.744708],[1.211157,-7.652887,3.860191,-5.114103,-1.689476,-7.044194,5.977732,-6.348766,-2.868935,-9.117057,-8.986389,-0.285355,-7.164298],[-2.385202,-6.328286,-0.948468,-7.740355,5.354983,5.296003,6.026966,9.792417,0.929258,1.909923,-4.577976,7.174120,3.537781],[2.498240,-1.384979,-8.502537,-0.078709,1.294798,5.787559,0.583032,-4.462157,-9.264788,5.402332,6.865071,-1.240750,-8.772796],[-9.075422,-1.526026,-2.190669,3.001366,5.966483,3.190072,5.648900,3.104098,7.429410,2.098857,7.246115,3.942185,9.476507],[-0.399391,2.801627,-9.994337,-3.284097,3.959134,6.132455,3.837405,9.518741,-1.710205,-3.412320,8.213708,-6.956656,5.254121],[-3.157467,-9.715801,2.276258,5.484045,-6.911416,-3.728031,-3.443037,5.193257,-1.381082,3.097078,3.335905,-3.221901,6.323178]],[[4.629935,-4.621572,1.190029,4.286992,-1.870014,-9.699098,1.023838,-2.403371,8.803780,-1.274726,-9.445092,-5.769807,-3.725587],[2.683955,6.152906,-4.866418,-5.014263,0.670049,9.006328,-0.729573,4.266858,8.597451,3.568140,-1.393585,1.271764,9.798900],[9.110436,1.678370,6.618939,8.589324,7.827602,-4.294288,-8.284883,3.140623,-7.936336,-2.280247,3.747595,4.527720,8.383208],[4.636419,-8.172703,-2.071616,-5.715818,2.761460,-2.261179,-3.469198,-1.684778,1.989723,-5.881610,-7.647415,-3.350342,-5.831052],[-8.826205,-2.652662,7.119764,-7.964965,-6.187564,-9.742478,-1.506769,3.753033,-6.008648,-4.924984,-9.135588,2.238055,1.956663],[6.870681,-9.255544,4.980151,2.653987,-9.513757,1.812179,-9.292298,7.337096,4.089538,-9.732960,-5.755414,-0.815861,0.722027],[6.997859,-9.964353,-4.901295,-5.665908,7.025034,-7.906846,-4.620241,-3.036878,1.035725,8.425773,6.021405,3.001909,5.323816],[-9.573582,-6.003217,-7.971247,-4.117338,-1.096666,8.786835,-7.550647,-5.240444,3.727535,9.567940,0.039808,2.384429,-2.427735]],[[5.607104,-3.830834,9.046987,-2.558977,3.703050,1.764124,1.415104,-5.164889,1.535964,-1.307501,2.398430,-1.855280,4.428189],[-4.848735,-1.558100,0.302462,-3.865998,5.797599,-1.645619,4.392041,-4.523801,7.194542,5.194059,-7.475750,-5.404667,-3.031555],[3.336868,9.593226,-5.310588,2.729796,-9.154912,-0.165226,-9.253773,-4.972472,-3.515291,-6.412355,6.911039,-5.568294,-4.310869],[7.419880,3.135416,-6.834082,-2.083198,6.232525,0.614294,6.740965,2.619609,-9.408721,0.373510,-3.645491,7.854877,3.691661],[5.273546,0.300267,-6.627427,-8.624056,-2.256528,-3.549644,-9.304987,-0.809573,-0.023857,8.094976,4.647089,-4.342084,9.709657],[-3.726059,5.224886,6.357103,-5.065135,-8.964989,0.481161,-2.439009,-5.628900,-5.363093,0.162335,-0.193554,-3.663207,6.505550],[-5.979695,0.483885,5.769102,-9.677980,9.777152,1.787439,-3.465844,-1.908474,-2.205605,2.076860,-8.556313,-1.507624,-6.015056],[3.679851,-2.617012,1.186855,-1.968384,-4.188929,-3.930571,6.687825,-0.093040,6.538708,-3.274096,-2.060794,8.578805,8.095177]],[[3.896231,7.466338,9.012820,2.343253,4.591735,-8.522180,5.631232,6.507097,3.201729,9.547813,9.721740,-5.166705,-6.395841],[9.703802,-9.701331,7.894858,-7.886656,-3.606078,-9.005954,8.225431,-3.727615,0.943055,5.022417,7.100825,1.627852,-1.103040],[-9.040300,9.358850,-2.244108,-1.453698,7.890494,-1.276723,-3.931832,2.973972,-4.008084,-7.202471,-3.757965,-7.098927,1.568199],[-3.755554,-1.338104,9.406727,3.684538,-1.784012,5.898431,2.073870,-5.282704,-2.205775,-8.893883,9.135446,9.616122,0.951470],[-8.095700,6.110453,3.077170,-4.891475,-8.861286,-3.398973,-4.894690,1.674001,9.276919,6.317897,6.695972,-8.358827,8.381549],[-2.358470,-2.109475,2.645460,-1.990340,-1.267915,-2.386763,7.639096,-9.068688,7.511241,-6.583484,4.027865,4.119012,-5.801670],[-8.152370,-0.771219,-8.995391,-8.489401,-3.564115,6.604723,-8.449242,-6.812703,5.060785,-8.973633,-3.529932,-9.901156,-4.758977],[-1.340932,-4.994686,-0.284269,-9.294098,7.622243,-2.478288,5.976655,2.303375,1.427021,-1.519216,-2.440287,8.303671,5.701452]],[[-6.585702,1.418387,5.114307,-7.216268,-4.749112,9.331966,2.803949,3.211511,-1.453242,-5.411794,-5.454159,-3.385735,-9.318158],[-7.611639,6.024764,5.649938,6.767310,-7.699574,-8.684126,8.278354,-1.433419,9.357346,5.566374,7.708476,8.538228,4.725334],[9.037101,-7.845584,-2.743379,7.781909,5.272330,-7.825116,6.023699,4.298716,6.832388,-2.430195,4.328782,-3.390780,8.818917],[7.164146,6.415939,-1.859107,-2.767379,-3.018908,6.265639,-9.915195,-9.544332,8.514016,9.352057,-4.242675,8.969050,-7.272411],[-7.622993,1.276541,-8.662710,3.299335,2.188854,-8.269149,-6.779692,1.989116,8.664331,-9.450636,-7.795207,6.917526,2.350095],[-4.484220,4.682084,1.402868,9.133485,7.694470,3.649542,-7.438170,-8.415066,-3.180896,5.417090,-4.203541,2.264510,0.114914],[-6.764119,-5.813576,-8.167017,3.693602,7.947381,9.904500,2.110802,0.299242,4.009497,-6.617172,-3.081913,7.403893,-7.542822],[-8.899487,-3.230323,6.325475,7.628366,-3.784061,-8.256555,1.879063,7.733188,2.220379,7.850649,6.089960,-6.627960,-6.067339]],[[9.731070,-5.803911,5.697852,9.478329,-8.089297,-5.132075,-4.121148,5.843318,-5.660723,8.651260,-1.877899,0.850729,3.953136],[-1.030699,-0.092784,-5.434387,-7.797314,0.012510,-2.147866,-0.420189,5.966585,-1.067234,6.950055,-6.143141,-8.075099,2.564515],[-3.852636,-3.598878,-1.070026,7.191863,1.495939,-4.131569,9.473866,2.188917,-3.595840,-7.484251,-1.577912,-1.614804,3.777409],[-1.167771,-3.466423,-8.822248,-7.151226,-4.173519,3.756976,0.896119,3.370682,1.342037,6.333263,-3.794601,-2.563716,1.901753],[0.187759,-0.886753,-1.116452,-8.136624,2.006047,-7.739093,2.115084,-1.102451,-9.617014,0.627904,3.988487,-6.520304,4.280063],[-5.484578,-6.885677,0.165249,2.852464,-3.883991,-1.930323,-1.427933,7.329251,7.945364,3.639002,4.266342,6.303090,-0.331891],[7.303430,8.920561,1.822729,-1.257375,0.167178,-0.668115,-2.753097,-4.701028,2.038551,-1.273939,1.534482,-4.888688,9.996753],[8.440045,-5.020905,0.551672,-3.837061,4.197173,-1.802279,5.116488,9.553441,4.762575,-7.901667,-6.574558,-7.200518,-6.005780]],[[-2.357200,1.496591,6.091272,-0.597455,-1.202995,1.049243,-1.759850,-2.396881,0.767603,-9.028776,1.135539,-8.375112,8.076691],[-8.751221,4.235357,-7.282123,8.094350,6.734041,-8.861442,-1.377323,3.366018,3.981508,-5.583752,-1.069778,-6.522675,-1.018698],[-6.856501,1.732547,9.748957,-0.248974,0.155022,-2.828694,-6.635669,2.212400,8.943015,7.702081,4.527571,-2.246480,-8.682325],[0.466657,1.536911,6.219845,-4.432844,-1.289113,-6.470807,-5.800244,4.230768,-3.532251,-6.874326,-4.938530,3.790861,-6.853652],[-8.163465,0.367271,-7.441665,4.162929,-2.429383,1.011641,-8.719739,9.660622,-9.467035,-4.394401,-1.881317,8.795263,-6.611551],[5.752669,9.778202,-4.763189,-9.762967,5.778152,5.641031,7.322217,6.071802,-5.898547,-2.052763,-4.458993,-4.919415,-8.476285],[6.994697,3.724583,4.409776,0.540398,4.483479,-9.471474,2.082880,-4.911565,2.221873,4.742043,-1.358093,-3.065605,8.940028],[-9.509873,2.988262,-3.557051,-2.143497,-3.446524,-4.506188,1.504607,3.790287,-4.014070,4.224142,6.180178,-3.524084,9.563265]],[[2.840782,4.821946,-2.854099,-2.350770,-9.606671,-1.728175,-2.843630,-9.345188,-6.837767,-7.246943,1.137032,8.976563,8.211324],[3.484706,9.021354,2.061222,-2.618851,7.460472,-3.765177,-5.138692,-2.593504,2.694434,4.536120,-5.553058,0.744428,-1.386405],[1.950931,0.643553,-5.047663,-5.408574,0.118831,-8.821909,-0.948194,-5.517225,5.524796,8.194998,9.150348,9.549560,-4.462180],[5.734001,-4.069133,-6.037779,7.657125,3.182455,9.710175,0.572233,4.846302,-0.584386,8.625564,-0.516268,-6.429877,-5.668718],[-2.718154,-1.207401,9.493942,9.157990,3.731286,2.120015,6.136027,5.884723,-9.215383,-3.551201,-8.983022,8.763829,0.968949],[7.638591,0.341982,-5.641394,1.921215,-5.618055,-6.759347,-4.784279,-8.223264,-6.107944,-4.484069,-7.328839,-0.599832,-7.668096],[5.720477,-2.275232,4.729264,-6.476703,0.535639,7.558341,-6.340488,-0.917019,6.477976,-6.667745,-0.004200,0.492051,-4.087293],[-3.992122,-3.547162,-8.572628,-8.841450,6.050565,-7.656956,7.330524,-7.966101,-5.351920,-2.862120,0.210249,0.626180,2.595489]],[[-7.701020,2.383622,-5.501926,9.290864,-4.239024,4.371370,-9.947258,8.344557,-9.269501,-6.188324,4.716366,8.520358,6.763920],[2.552671,9.871324,6.363403,7.576111,-3.182913,-0.860580,8.433192,7.613087,3.129808,2.801863,-2.885068,2.879796,-3.796710],[0.896301,1.813181,-8.719549,-8.406693,0.735611,5.697006,-2.245697,2.447996,9.293153,5.380198,0.496473,-8.960209,9.324138],[-9.679833,2.986577,8.638345,3.497603,-3.571053,2.011872,3.574520,-1.995280,-1.049582,-5.139026,6.201396,-1.790534,6.565934],[2.519568,-5.585890,-8.799418,-2.223448,3.222179,8.965104,-8.907571,-2.322414,-0.537057,-7.271399,-4.859279,5.304017,-5.597172],[-6.714950,0.434106,-1.958376,5.510893,-4.653778,-9.326861,1.801391,-1.339947,-8.940582,-2.201370,3.366347,-7.549292,4.955333],[-6.312482,3.389924,0.882009,1.996542,0.316220,-5.260976,-3.725634,7.423612,5.005295,-9.101950,4.548343,-8.215374,4.301700],[-2.568601,3.073074,-3.128729,-5.839191,2.599681,2.922785,-7.909287,-9.430568,2.825670,4.090272,0.473876,7.046666,1.604307]],[[-5.396803,4.580700,4.262483,6.037973,9.250231,-2.365015,-9.461380,-6.607823,-3.468185,-2.221039,4.331001,5.324276,-7.401346],[0.984780,-1.317342,-6.443831,-3.572701,-1.480720,3.998966,-9.658281,4.020853,-2.368580,-8.247715,3.648283,-0.091183,-5.246173],[0.648585,-1.784784,-9.769260,9.542834,-4.730784,-7.439795,4.724226,-8.029053,-2.624040,-9.851169,7.660335,1.562568,4.709760],[-9.693643,9.855446,6.815029,-6.351373,-1.545460,4.066861,5.152561,-2.509358,-8.444742,-2.875989,-9.438320,1.325030,-0.295208],[-6.659155,4.977719,-4.414243,8.814216,-0.360438,-9.406403,-6.627296,9.977046,6.215998,-8.870177,-5.323001,8.857674,0.634313],[-5.326607,-9.915767,-2.966703,-8.648075,-0.305663,4.479493,-2.830641,-9.723962,7.941922,-6.628327,-2.546144,9.324176,2.012501],[6.816850,8.311749,-4.944175,-8.225390,-9.559310,5.490466,-8.930179,-6.690193,4.130126,0.643355,1.241296,1.966541,0.789257],[9.164896,-0.670182,8.414143,-5.180498,-6.057018,-2.356792,3.664986,5.947298,6.891167,4.152241,-1.685181,4.549673,-1.757515]],[[-8.186557,-2.508571,-8.391046,-1.998280,-1.751304,-9.499985,-3.760230,2.204847,7.288396,-6.998641,-6.279746,-3.717412,4.798451],[-6.985109,-3.184414,8.271261,-1.758236,4.240085,-3.440777,6.834807,8.039315,0.114256,9.847311,-6.002255,2.746749,5.920405],[-5.184245,0.148415,-9.670514,-3.693427,7.665009,2.393014,1.350731,-9.673415,6.828759,-7.596357,6.334821,-3.075204,4.831094],[-1.250604,3.651203,-9.955367,8.187540,5.144924,1.232164,2.353593,-2.468211,6.278831,3.674890,6.142786,3.916666,4.000488],[3.722263,6.644372,-6.436059,0.836439,-8.013630,8.265336,-9.208521,0.485415,2.877949,-0.982823,-4.417868,0.268125,5.866124],[-5.459429,7.440329,6.724040,2.925366,0.216339,-0.161926,-6.454525,-8.829764,1.627169,-7.728330,-2.136766,6.294634,3.457966],[-0.931663,2.123986,2.683349,6.015699,4.940246,1.150044,4.078029,-6.330228,-6.853066,-3.601979,-5.001591,2.050060,4.213893],[-8.702596,-8.016749,-2.556100,2.598909,-1.564012,7.491719,1.273050,-5.257111,7.141790,1.476889,-7.820629,-3.981934,-4.294585]],[[-6.715987,-2.408595,2.257125,-2.778143,-8.846764,6.654070,-4.862518,4.319505,1.363553,-8.709875,9.006589,7.279263,0.585364],[-1.994668,9.584878,-8.801858,-1.831156,5.569690,-3.253497,2.800993,-1.761878,-0.531029,-4.491488,-7.784747,-8.801405,-3.509302],[-6.624703,-4.104159,9.052767,-0.672637,7.771263,-9.649856,9.026004,2.022387,5.380166,-3.456198,6.515351,-9.008736,-5.580537],[7.718291,0.114939,2.688674,-0.760491,7.646392,-6.827448,-1.734594,-8.716421,-9.957176,-6.870712,8.593638,-1.840700,0.193673],[-1.616183,-2.273086,-1.467798,4.516680,0.959919,2.760486,-3.830330,9.251675,-6.612499,-1.741740,-0.520608,-3.264398,4.277055],[-8.620221,-8.948532,3.852855,0.758443,9.455246,-5.453919,7.309389,7.583503,4.874079,7.962890,4.511287,-4.853517,-5.683734],[-3.076086,8.195584,5.088098,7.256634,5.000094,-5.804184,-9.398793,-0.183204,4.638207,-8.931189,-8.393102,-6.957661,-3.047773],[-9.217689,-2.532026,6.189104,1.124805,-4.702210,-8.051082,-6.210678,-7.328432,-0.255279,5.471174,5.669943,-8.775867,6.085472]]], dtype = "float32")#candidate|1098|(12, 8, 13)|const|float32
uop_1099 = relay.acos(const_1098.astype('float32')) # shape=(12, 8, 13)
func_251_call = mod.get_global_var('func_251')
func_255_call = mutated_mod.get_global_var('func_255')
var_1103 = relay.var("var_1103", dtype = "float64", shape = (128,))#candidate|1103|(128,)|var|float64
var_1104 = relay.var("var_1104", dtype = "float64", shape = (9,))#candidate|1104|(9,)|var|float64
call_1102 = relay.TupleGetItem(func_251_call(relay.reshape(var_1103.astype('float64'), [8, 16]), relay.reshape(var_1104.astype('float64'), [9,]), ), 6)
call_1105 = relay.TupleGetItem(func_255_call(relay.reshape(var_1103.astype('float64'), [8, 16]), relay.reshape(var_1104.astype('float64'), [9,]), ), 6)
uop_1106 = relay.log2(uop_1099.astype('float32')) # shape=(12, 8, 13)
uop_1111 = relay.rsqrt(uop_1106.astype('float64')) # shape=(12, 8, 13)
bop_1117 = relay.maximum(uop_1106.astype('float32'), relay.reshape(const_1098.astype('float32'), relay.shape_of(uop_1106))) # shape=(12, 8, 13)
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_1133 = func_573_call()
call_1134 = func_573_call()
func_405_call = mod.get_global_var('func_405')
func_409_call = mutated_mod.get_global_var('func_409')
const_1139 = relay.const([[-1.766461,-2.135201],[-3.084170,-2.088388],[0.260101,5.730325],[4.753083,-9.324311],[-5.583539,-6.786143],[2.267758,-6.147052],[6.812755,2.782720],[-3.306325,-7.469053],[4.881031,4.523443],[4.218889,8.151115],[0.814717,2.316243],[6.668535,-8.996733],[1.898720,-2.781617],[0.907748,5.085476],[-0.377306,-5.963047],[-8.483130,7.691009],[4.556397,4.084081],[8.717313,-6.953715],[3.280597,3.554823],[5.485087,-0.637495],[-5.360479,0.211424],[-9.122104,4.311826],[-4.292128,3.998792],[-1.386324,-4.824486],[2.556418,2.217395],[-9.644532,0.914457],[7.160563,-8.300169],[-9.023999,8.588403],[2.541417,-1.086606],[7.012911,1.350457],[-0.148070,-3.689734],[5.684402,-6.743765],[-4.469135,9.764524],[1.883287,-5.537056],[-1.397034,-9.712485],[-1.539742,2.224632],[-9.308940,8.747777],[8.244729,6.026598],[4.883776,8.959024],[-1.429042,5.988551],[5.201779,3.817312],[8.005745,2.228413],[3.701847,-6.191235],[7.458549,3.289018],[2.917269,6.709411],[1.373019,-1.305454],[7.671087,-0.435426],[-6.482641,-6.138785],[-5.507183,-0.862025],[-6.822762,6.301398],[4.966807,6.657894],[-3.814471,5.485464],[5.370245,-3.504119],[3.527003,6.351147],[-5.747459,9.947844],[4.363793,-0.696645],[2.481347,6.533225],[-3.448054,2.883655],[-9.804258,-7.820141],[-2.891479,-2.657350],[4.347396,-6.434017],[-3.599866,7.819182],[0.822251,2.864394],[-9.072491,4.871900],[-7.711307,4.499015],[6.494382,8.358090],[4.923528,-7.932756],[3.485422,5.608962],[-6.532239,8.302907],[5.345013,3.535912],[4.410505,4.878293],[7.065070,-1.150569],[9.542028,-2.303371],[-7.706630,-4.864917],[-7.821974,1.249444]], dtype = "float32")#candidate|1139|(75, 2)|const|float32
call_1138 = relay.TupleGetItem(func_405_call(relay.reshape(const_1139.astype('float32'), [150,]), relay.reshape(call_1102.astype('float32'), [126,]), ), 4)
call_1140 = relay.TupleGetItem(func_409_call(relay.reshape(const_1139.astype('float32'), [150,]), relay.reshape(call_1102.astype('float32'), [126,]), ), 4)
func_315_call = mod.get_global_var('func_315')
func_321_call = mutated_mod.get_global_var('func_321')
call_1142 = relay.TupleGetItem(func_315_call(relay.reshape(const_1139.astype('float32'), [10, 15]), relay.reshape(var_1104.astype('float64'), [9,]), relay.reshape(var_1104.astype('float64'), [3, 3]), relay.reshape(call_1138.astype('float32'), [63,]), ), 1)
call_1143 = relay.TupleGetItem(func_321_call(relay.reshape(const_1139.astype('float32'), [10, 15]), relay.reshape(var_1104.astype('float64'), [9,]), relay.reshape(var_1104.astype('float64'), [3, 3]), relay.reshape(call_1138.astype('float32'), [63,]), ), 1)
bop_1147 = relay.logical_xor(bop_1117.astype('uint32'), relay.reshape(uop_1099.astype('uint32'), relay.shape_of(bop_1117))) # shape=(12, 8, 13)
bop_1156 = relay.bitwise_xor(uop_1099.astype('int64'), relay.reshape(bop_1147.astype('int64'), relay.shape_of(uop_1099))) # shape=(12, 8, 13)
func_82_call = mod.get_global_var('func_82')
func_85_call = mutated_mod.get_global_var('func_85')
call_1163 = relay.TupleGetItem(func_82_call(relay.reshape(call_1138.astype('float32'), [7, 1, 9]), relay.reshape(call_1102.astype('float32'), [7, 2, 9]), ), 0)
call_1164 = relay.TupleGetItem(func_85_call(relay.reshape(call_1138.astype('float32'), [7, 1, 9]), relay.reshape(call_1102.astype('float32'), [7, 2, 9]), ), 0)
bop_1170 = relay.right_shift(bop_1117.astype('uint8'), relay.reshape(uop_1099.astype('uint8'), relay.shape_of(bop_1117))) # shape=(12, 8, 13)
func_817_call = mod.get_global_var('func_817')
func_818_call = mutated_mod.get_global_var('func_818')
call_1176 = relay.TupleGetItem(func_817_call(), 0)
call_1177 = relay.TupleGetItem(func_818_call(), 0)
func_251_call = mod.get_global_var('func_251')
func_255_call = mutated_mod.get_global_var('func_255')
call_1190 = relay.TupleGetItem(func_251_call(relay.reshape(var_1103.astype('float64'), [8, 16]), relay.reshape(call_1142.astype('float64'), [9,]), ), 9)
call_1191 = relay.TupleGetItem(func_255_call(relay.reshape(var_1103.astype('float64'), [8, 16]), relay.reshape(call_1142.astype('float64'), [9,]), ), 9)
output = relay.Tuple([call_1102,var_1103,var_1104,uop_1111,call_1133,call_1138,const_1139,call_1142,bop_1156,call_1163,bop_1170,call_1176,call_1190,])
output2 = relay.Tuple([call_1105,var_1103,var_1104,uop_1111,call_1134,call_1140,const_1139,call_1143,bop_1156,call_1164,bop_1170,call_1177,call_1191,])
func_1192 = relay.Function([var_1103,var_1104,], output)
mod['func_1192'] = func_1192
mod = relay.transform.InferType()(mod)
mutated_mod['func_1192'] = func_1192
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1192_call = mutated_mod.get_global_var('func_1192')
var_1194 = relay.var("var_1194", dtype = "float64", shape = (128,))#candidate|1194|(128,)|var|float64
var_1195 = relay.var("var_1195", dtype = "float64", shape = (9,))#candidate|1195|(9,)|var|float64
call_1193 = func_1192_call(var_1194,var_1195,)
output = call_1193
func_1196 = relay.Function([var_1194,var_1195,], output)
mutated_mod['func_1196'] = func_1196
mutated_mod = relay.transform.InferType()(mutated_mod)
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_1202 = func_573_call()
call_1203 = func_573_call()
output = relay.Tuple([call_1202,])
output2 = relay.Tuple([call_1203,])
func_1215 = relay.Function([], output)
mod['func_1215'] = func_1215
mod = relay.transform.InferType()(mod)
mutated_mod['func_1215'] = func_1215
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1215_call = mutated_mod.get_global_var('func_1215')
call_1216 = func_1215_call()
output = call_1216
func_1217 = relay.Function([], output)
mutated_mod['func_1217'] = func_1217
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1221 = relay.var("var_1221", dtype = "float64", shape = (15, 11))#candidate|1221|(15, 11)|var|float64
var_1222 = relay.var("var_1222", dtype = "float64", shape = (15, 11))#candidate|1222|(15, 11)|var|float64
bop_1223 = relay.floor_mod(var_1221.astype('float64'), relay.reshape(var_1222.astype('float64'), relay.shape_of(var_1221))) # shape=(15, 11)
func_405_call = mod.get_global_var('func_405')
func_409_call = mutated_mod.get_global_var('func_409')
const_1240 = relay.const([-0.967697,8.815524,-9.359965,-4.739086,4.119468,4.734418,-7.369447,-7.047401,8.344783,3.417574,7.051102,1.379944,2.734391,7.518431,-7.961034,4.979748,-9.251462,-7.153315,-8.544938,-2.021392,-1.194831,-4.058238,-4.735350,-4.782630,5.288727,0.301733,9.549105,-0.997022,4.914570,2.310733,-7.760065,-8.019363,-5.190238,-3.725794,-8.343630,-3.419201,-0.459828,4.953522,5.324479,-7.153117,4.040182,1.790522,-3.719048,-1.788660,5.471646,2.133320,-4.291654,-0.678246,-6.050946,4.297199,9.612909,-2.743451,-5.479557,2.998680,8.243042,-4.725659,3.637824,8.467369,5.820973,1.432354,-0.658282,0.148234,1.379816,5.028687,-0.557181,-6.768567,-9.337397,6.169845,-7.730190,-1.997486,-0.182308,4.555844,-9.451540,0.133413,-8.260120,9.573039,-4.161760,5.049629,-3.603885,0.072552,-0.453439,6.898243,-0.144306,8.070359,0.112214,3.311629,7.815934,6.402809,6.592422,3.067248,8.481857,-3.601431,6.162383,6.749911,-4.882187,-3.511644,-6.917399,2.855253,-7.329232,-9.479414,-9.433903,2.786658,-5.196875,-2.661230,-4.130292,6.413738,-7.315932,1.877426,9.683551,8.576739,7.395614,-5.874330,6.741871,7.555339,5.893582,0.188528,7.193932,8.317501,4.682538,7.033821,7.175417,-2.888265,4.846083,-3.944462,-7.643334,-5.633199,9.494459,-8.666017,3.134955,4.372788,9.515642,-0.851503,1.012420,-9.973196,7.335099,-3.016581,-8.101518,-2.390877,-6.092976,-7.410954,0.937002,2.941087,-9.978561,1.590465,4.183985,-5.407952,-5.865552,8.944496,-0.797265,-7.725904], dtype = "float32")#candidate|1240|(150,)|const|float32
const_1241 = relay.const([[2.633068,8.495662,5.233703,9.974242,-0.968321,-7.739447,7.926954,-7.648002,-4.917049,-9.106348,-2.295965,4.585048,7.923269,-1.100933,-2.121002,-6.244519,4.039578,8.266938,-6.818617,6.645499,-7.444902,3.200278,6.559702,8.894668,-3.490570,9.334448,-9.340391,6.598305,-1.984102,-1.930582,-2.093101,-6.648324,-5.640961,-3.316261,-0.407130,2.950778,0.348378,-0.644508,-6.943444,-9.387817,-4.908476,-7.416096,-0.011595,-7.456026,6.016128,2.712574,6.503236,9.905937,0.643224,0.521158,-4.373505,1.231785,7.613559,-6.323002,5.655877,-4.496102,9.589967,4.050828,-1.614101,-5.173727,-4.873064,1.024582,9.693874,8.349273,-6.037641,9.984241,-8.277286,-5.840125,-1.829494,1.679710,7.088007,3.538778,-5.113786,-6.924726,-5.652199,-6.176855,-8.665766,5.043816,6.134037,6.312365,8.598769,-8.457367,5.987548,6.701611,1.137756,9.947041,-3.618191,0.389649,3.668164,5.319077,-8.491817,-0.294343,-8.286053,9.544248,6.114048,0.179497,4.496165,-0.693647,9.135498,-4.409957,-3.420940,0.897083,2.427567,-5.671921,0.300632,-5.559383,9.728166,-9.167330,-2.723281,7.091435,-3.043298,-7.576706,-8.385419,-3.278159,-5.348638,2.575081,7.573982,3.112522,1.301674,8.064994,-4.971005,7.832627,4.369183,5.123011,8.745738,3.821504]], dtype = "float32")#candidate|1241|(1, 126)|const|float32
call_1239 = relay.TupleGetItem(func_405_call(relay.reshape(const_1240.astype('float32'), [150,]), relay.reshape(const_1241.astype('float32'), [126,]), ), 0)
call_1242 = relay.TupleGetItem(func_409_call(relay.reshape(const_1240.astype('float32'), [150,]), relay.reshape(const_1241.astype('float32'), [126,]), ), 0)
func_597_call = mod.get_global_var('func_597')
func_599_call = mutated_mod.get_global_var('func_599')
call_1246 = relay.TupleGetItem(func_597_call(), 1)
call_1247 = relay.TupleGetItem(func_599_call(), 1)
uop_1252 = relay.log2(call_1246.astype('float32')) # shape=(1, 16)
uop_1254 = relay.log2(call_1247.astype('float32')) # shape=(1, 16)
uop_1255 = relay.asin(call_1239.astype('float64')) # shape=(4, 10, 8)
uop_1257 = relay.asin(call_1242.astype('float64')) # shape=(4, 10, 8)
uop_1277 = relay.sigmoid(uop_1255.astype('float32')) # shape=(4, 10, 8)
uop_1279 = relay.sigmoid(uop_1257.astype('float32')) # shape=(4, 10, 8)
bop_1280 = relay.less_equal(uop_1277.astype('bool'), relay.reshape(uop_1255.astype('bool'), relay.shape_of(uop_1277))) # shape=(4, 10, 8)
bop_1283 = relay.less_equal(uop_1279.astype('bool'), relay.reshape(uop_1257.astype('bool'), relay.shape_of(uop_1279))) # shape=(4, 10, 8)
output = relay.Tuple([bop_1223,const_1240,const_1241,uop_1252,bop_1280,])
output2 = relay.Tuple([bop_1223,const_1240,const_1241,uop_1254,bop_1283,])
func_1288 = relay.Function([var_1221,var_1222,], output)
mod['func_1288'] = func_1288
mod = relay.transform.InferType()(mod)
mutated_mod['func_1288'] = func_1288
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1288_call = mutated_mod.get_global_var('func_1288')
var_1290 = relay.var("var_1290", dtype = "float64", shape = (15, 11))#candidate|1290|(15, 11)|var|float64
var_1291 = relay.var("var_1291", dtype = "float64", shape = (15, 11))#candidate|1291|(15, 11)|var|float64
call_1289 = func_1288_call(var_1290,var_1291,)
output = call_1289
func_1292 = relay.Function([var_1290,var_1291,], output)
mutated_mod['func_1292'] = func_1292
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1294 = relay.var("var_1294", dtype = "float32", shape = (12,))#candidate|1294|(12,)|var|float32
var_1295 = relay.var("var_1295", dtype = "float32", shape = (12,))#candidate|1295|(12,)|var|float32
bop_1296 = relay.floor_mod(var_1294.astype('float32'), relay.reshape(var_1295.astype('float32'), relay.shape_of(var_1294))) # shape=(12,)
output = bop_1296
output2 = bop_1296
func_1310 = relay.Function([var_1294,var_1295,], output)
mod['func_1310'] = func_1310
mod = relay.transform.InferType()(mod)
var_1311 = relay.var("var_1311", dtype = "float32", shape = (12,))#candidate|1311|(12,)|var|float32
var_1312 = relay.var("var_1312", dtype = "float32", shape = (12,))#candidate|1312|(12,)|var|float32
output = func_1310(var_1311,var_1312,)
func_1313 = relay.Function([var_1311,var_1312,], output)
mutated_mod['func_1313'] = func_1313
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1330 = relay.const([[-1.651844,3.743279,2.582636,-4.401344,-3.928975,0.489594,-3.312317],[-0.963622,5.256823,-9.746553,9.217920,2.553878,0.117426,-0.793778],[-0.347700,9.554236,4.589288,-3.877514,5.446844,9.693692,-9.581702],[-3.352196,-6.861785,6.249125,-1.086737,-2.457912,-4.083542,7.597410],[-9.944690,-4.598929,2.748102,8.757766,-1.629170,5.115636,-2.684179],[-9.326443,-8.545704,2.983688,4.708388,-6.525233,-9.281186,-5.743734],[-5.070995,-9.776564,6.311753,0.758303,-9.450315,5.783621,-2.908651],[-0.024084,7.575132,-1.882413,9.855820,-6.329953,-5.414306,2.270117],[3.325777,6.964991,2.360704,-2.510148,9.315073,4.770701,-5.895515],[-8.222828,3.824907,-8.931091,3.734338,-3.664751,-7.422665,-6.398065],[8.169395,-2.601222,3.130686,3.043274,2.957355,2.136301,1.986007]], dtype = "float32")#candidate|1330|(11, 7)|const|float32
const_1331 = relay.const([[3.777719,6.033651,-9.436816,4.203711,-4.868740,1.960861,-3.923090],[6.360782,2.669498,3.808244,1.202908,0.070292,-8.582385,4.027805],[7.935653,-5.226046,6.074356,8.078145,3.394875,0.934548,1.866617],[-3.087274,-6.827675,4.070245,8.443506,-8.599310,2.569371,-3.850132],[-6.268168,-2.439727,-0.948826,-5.195897,-6.031860,9.014476,-1.935982],[9.373346,3.130085,-9.507972,8.989648,1.647406,-8.076839,7.431478],[2.781577,3.132558,-8.038197,3.378849,-9.319310,-6.131221,-7.046175],[-5.604829,-6.218403,-7.105158,-2.936962,-2.212528,-4.511237,-1.465589],[9.723828,1.700607,5.007912,9.931618,-2.783604,-6.983391,-7.288398],[4.822982,4.077818,-9.081392,5.840744,7.763386,4.190570,5.081670],[-5.088804,-8.234498,9.247029,-4.272648,-7.929046,-3.796486,5.405980]], dtype = "float32")#candidate|1331|(11, 7)|const|float32
bop_1332 = relay.equal(const_1330.astype('bool'), relay.reshape(const_1331.astype('bool'), relay.shape_of(const_1330))) # shape=(11, 7)
bop_1335 = relay.bitwise_and(const_1330.astype('uint16'), relay.reshape(bop_1332.astype('uint16'), relay.shape_of(const_1330))) # shape=(11, 7)
uop_1339 = relay.atan(bop_1335.astype('float64')) # shape=(11, 7)
bop_1351 = relay.logical_and(uop_1339.astype('bool'), relay.reshape(const_1331.astype('bool'), relay.shape_of(uop_1339))) # shape=(11, 7)
uop_1354 = relay.sqrt(uop_1339.astype('float32')) # shape=(11, 7)
uop_1359 = relay.tan(bop_1351.astype('float32')) # shape=(11, 7)
bop_1362 = relay.subtract(uop_1354.astype('uint16'), relay.reshape(uop_1359.astype('uint16'), relay.shape_of(uop_1354))) # shape=(11, 7)
const_1370 = relay.const([[4,2,4,10,-7,7,6],[-4,-6,6,-10,1,7,4],[2,-3,7,-4,7,8,-8],[5,2,5,-4,8,-5,7],[10,-5,-3,9,-6,-2,6],[4,-4,10,-2,-4,5,-4],[10,4,3,2,-8,7,-3],[-4,6,5,-10,-7,-9,1],[-2,6,-6,-4,10,4,-7],[3,-7,-9,8,4,10,-2],[-4,-2,-10,10,1,-5,-8]], dtype = "uint16")#candidate|1370|(11, 7)|const|uint16
bop_1371 = relay.power(bop_1362.astype('float32'), relay.reshape(const_1370.astype('float32'), relay.shape_of(bop_1362))) # shape=(11, 7)
output = relay.Tuple([bop_1371,])
output2 = relay.Tuple([bop_1371,])
func_1376 = relay.Function([], output)
mod['func_1376'] = func_1376
mod = relay.transform.InferType()(mod)
output = func_1376()
func_1377 = relay.Function([], output)
mutated_mod['func_1377'] = func_1377
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1408 = relay.const([[5.934711,9.495752],[9.695706,1.982852],[8.447850,-5.835341],[6.179651,2.515107],[8.750220,0.065197],[-9.346522,-7.152368],[6.592347,6.106468],[-1.227839,-7.386813],[-8.452668,4.787321],[-5.971243,8.517956],[-1.733206,3.549607],[-2.431696,5.568362],[-3.635539,-1.449224],[-4.756131,-7.055474]], dtype = "float32")#candidate|1408|(14, 2)|const|float32
var_1409 = relay.var("var_1409", dtype = "float32", shape = (14, 2))#candidate|1409|(14, 2)|var|float32
bop_1410 = relay.divide(const_1408.astype('float32'), relay.reshape(var_1409.astype('float32'), relay.shape_of(const_1408))) # shape=(14, 2)
func_1192_call = mod.get_global_var('func_1192')
func_1196_call = mutated_mod.get_global_var('func_1196')
const_1419 = relay.const([1.703326,8.656541,1.228708,1.624513,-4.090533,1.055622,-1.060569,0.377515,-6.905061,7.684628,-3.661456,-9.889292,-4.321892,1.819851,-2.803032,-4.571136,-3.677249,1.490151,-0.137697,-7.326760,-7.418722,0.355133,6.439414,8.546672,-5.543240,-2.129109,7.551441,1.986999,-7.488583,6.751805,-2.155510,5.588916,8.642834,-7.208044,7.321655,-8.906165,-9.313255,-8.308366,-7.079761,-4.138588,-0.429489,-9.498001,8.702974,5.264339,-2.847001,-1.356335,-8.346912,1.837593,-2.042480,-3.229250,-9.452582,4.993187,-4.370352,9.395979,6.363120,-4.130828,-3.417452,2.946481,-6.196373,9.409226,2.321963,4.777497,2.909682,0.212003,-6.515389,1.420800,5.350447,-9.715986,-6.978115,-0.339625,6.704529,6.470936,9.984777,2.496409,7.401277,4.779281,-9.264239,1.084579,3.111090,9.311516,-7.025959,5.020819,2.349769,6.157549,3.075345,5.868274,-3.186322,-9.754289,-6.712073,-5.034210,-0.226831,-8.981499,-0.364256,8.534371,1.450002,-6.415143,0.527781,9.998554,9.374828,0.243101,-8.687601,8.460720,6.673944,7.879491,3.764230,-3.411269,1.764864,-8.748812,-8.784311,5.672914,-3.154969,0.130259,2.285736,-3.208001,1.207397,0.602644,-8.576605,6.640975,-8.491366,7.385289,-7.014555,-4.212508,5.678003,5.885724,-8.051967,3.468009,-6.684569,-5.811311], dtype = "float64")#candidate|1419|(128,)|const|float64
const_1420 = relay.const([-9.988090,-8.515118,-9.917131,1.190802,-5.787270,-3.169237,-3.590692,-5.637488,5.406459], dtype = "float64")#candidate|1420|(9,)|const|float64
call_1418 = relay.TupleGetItem(func_1192_call(relay.reshape(const_1419.astype('float64'), [128,]), relay.reshape(const_1420.astype('float64'), [9,]), ), 8)
call_1421 = relay.TupleGetItem(func_1196_call(relay.reshape(const_1419.astype('float64'), [128,]), relay.reshape(const_1420.astype('float64'), [9,]), ), 8)
uop_1424 = relay.sigmoid(bop_1410.astype('float64')) # shape=(14, 2)
bop_1433 = relay.less(uop_1424.astype('bool'), relay.reshape(const_1408.astype('bool'), relay.shape_of(uop_1424))) # shape=(14, 2)
func_544_call = mod.get_global_var('func_544')
func_547_call = mutated_mod.get_global_var('func_547')
const_1442 = relay.const([7.226566,9.532740,0.885793,-9.408877,6.011635,1.461031,2.105033,0.888384,3.075093,1.403996,0.223714,-9.140055,2.228579,5.952655,4.748863,5.626512,7.010329,-7.172211,1.183655,9.134965,3.366103,4.185979,6.934200,-8.754657,-0.435167,-9.428485,9.844839,-8.909196,-4.591045,-2.642604,8.527687,-0.689103,-9.576331,9.602907,-5.298017,-1.590478,9.927041,-1.469748,-5.094441,-0.304418,-8.489260,-1.968148,4.941174,-3.166940,-9.805206,5.159231,2.009940,9.182623], dtype = "float64")#candidate|1442|(48,)|const|float64
call_1441 = func_544_call(relay.reshape(const_1442.astype('float64'), [3, 16]))
call_1443 = func_544_call(relay.reshape(const_1442.astype('float64'), [3, 16]))
func_573_call = mod.get_global_var('func_573')
func_574_call = mutated_mod.get_global_var('func_574')
call_1456 = func_573_call()
call_1457 = func_573_call()
bop_1468 = relay.logical_xor(bop_1433.astype('uint64'), relay.reshape(bop_1410.astype('uint64'), relay.shape_of(bop_1433))) # shape=(14, 2)
bop_1471 = relay.logical_and(uop_1424.astype('bool'), relay.reshape(bop_1468.astype('bool'), relay.shape_of(uop_1424))) # shape=(14, 2)
uop_1478 = relay.log2(call_1456.astype('float64')) # shape=(9, 10)
uop_1480 = relay.log2(call_1457.astype('float64')) # shape=(9, 10)
func_133_call = mod.get_global_var('func_133')
func_135_call = mutated_mod.get_global_var('func_135')
call_1487 = relay.TupleGetItem(func_133_call(relay.reshape(const_1420.astype('float64'), [3, 3])), 0)
call_1488 = relay.TupleGetItem(func_135_call(relay.reshape(const_1420.astype('float64'), [3, 3])), 0)
bop_1491 = relay.divide(uop_1478.astype('float32'), relay.reshape(call_1456.astype('float32'), relay.shape_of(uop_1478))) # shape=(9, 10)
bop_1494 = relay.divide(uop_1480.astype('float32'), relay.reshape(call_1457.astype('float32'), relay.shape_of(uop_1480))) # shape=(9, 10)
uop_1495 = relay.erf(bop_1471.astype('float64')) # shape=(14, 2)
uop_1500 = relay.exp(const_1419.astype('float64')) # shape=(128,)
output = relay.Tuple([call_1418,const_1420,call_1441,const_1442,call_1487,bop_1491,uop_1495,uop_1500,])
output2 = relay.Tuple([call_1421,const_1420,call_1443,const_1442,call_1488,bop_1494,uop_1495,uop_1500,])
func_1503 = relay.Function([var_1409,], output)
mod['func_1503'] = func_1503
mod = relay.transform.InferType()(mod)
mutated_mod['func_1503'] = func_1503
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1504 = relay.var("var_1504", dtype = "float32", shape = (14, 2))#candidate|1504|(14, 2)|var|float32
func_1503_call = mutated_mod.get_global_var('func_1503')
call_1505 = func_1503_call(var_1504)
output = call_1505
func_1506 = relay.Function([var_1504], output)
mutated_mod['func_1506'] = func_1506
mutated_mod = relay.transform.InferType()(mutated_mod)
func_817_call = mod.get_global_var('func_817')
func_818_call = mutated_mod.get_global_var('func_818')
call_1552 = relay.TupleGetItem(func_817_call(), 0)
call_1553 = relay.TupleGetItem(func_818_call(), 0)
output = relay.Tuple([call_1552,])
output2 = relay.Tuple([call_1553,])
func_1557 = relay.Function([], output)
mod['func_1557'] = func_1557
mod = relay.transform.InferType()(mod)
mutated_mod['func_1557'] = func_1557
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1557_call = mutated_mod.get_global_var('func_1557')
call_1558 = func_1557_call()
output = call_1558
func_1559 = relay.Function([], output)
mutated_mod['func_1559'] = func_1559
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1567 = relay.var("var_1567", dtype = "float64", shape = (10, 5, 1))#candidate|1567|(10, 5, 1)|var|float64
uop_1568 = relay.sin(var_1567.astype('float64')) # shape=(10, 5, 1)
uop_1576 = relay.tan(uop_1568.astype('float32')) # shape=(10, 5, 1)
bop_1579 = relay.multiply(uop_1576.astype('int8'), relay.reshape(uop_1568.astype('int8'), relay.shape_of(uop_1576))) # shape=(10, 5, 1)
output = bop_1579
output2 = bop_1579
func_1582 = relay.Function([var_1567,], output)
mod['func_1582'] = func_1582
mod = relay.transform.InferType()(mod)
mutated_mod['func_1582'] = func_1582
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1583 = relay.var("var_1583", dtype = "float64", shape = (10, 5, 1))#candidate|1583|(10, 5, 1)|var|float64
func_1582_call = mutated_mod.get_global_var('func_1582')
call_1584 = func_1582_call(var_1583)
output = call_1584
func_1585 = relay.Function([var_1583], output)
mutated_mod['func_1585'] = func_1585
mutated_mod = relay.transform.InferType()(mutated_mod)
func_171_call = mod.get_global_var('func_171')
func_172_call = mutated_mod.get_global_var('func_172')
call_1604 = relay.TupleGetItem(func_171_call(), 0)
call_1605 = relay.TupleGetItem(func_172_call(), 0)
uop_1617 = relay.log10(call_1604.astype('float64')) # shape=(1, 16)
uop_1619 = relay.log10(call_1605.astype('float64')) # shape=(1, 16)
bop_1621 = relay.logical_xor(uop_1617.astype('int16'), relay.reshape(call_1604.astype('int16'), relay.shape_of(uop_1617))) # shape=(1, 16)
bop_1624 = relay.logical_xor(uop_1619.astype('int16'), relay.reshape(call_1605.astype('int16'), relay.shape_of(uop_1619))) # shape=(1, 16)
uop_1625 = relay.sqrt(call_1604.astype('float64')) # shape=(1, 16)
uop_1627 = relay.sqrt(call_1605.astype('float64')) # shape=(1, 16)
bop_1629 = relay.mod(bop_1621.astype('float32'), relay.reshape(uop_1625.astype('float32'), relay.shape_of(bop_1621))) # shape=(1, 16)
bop_1632 = relay.mod(bop_1624.astype('float32'), relay.reshape(uop_1627.astype('float32'), relay.shape_of(bop_1624))) # shape=(1, 16)
output = relay.Tuple([bop_1629,])
output2 = relay.Tuple([bop_1632,])
func_1635 = relay.Function([], output)
mod['func_1635'] = func_1635
mod = relay.transform.InferType()(mod)
mutated_mod['func_1635'] = func_1635
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1635_call = mutated_mod.get_global_var('func_1635')
call_1636 = func_1635_call()
output = call_1636
func_1637 = relay.Function([], output)
mutated_mod['func_1637'] = func_1637
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1646 = relay.const([[-5.950440,4.997307,-0.262916],[4.618013,-5.850672,7.277417],[-1.354750,-9.641650,-4.593712],[1.011209,-0.880595,-1.599422]], dtype = "float32")#candidate|1646|(4, 3)|const|float32
uop_1647 = relay.sinh(const_1646.astype('float32')) # shape=(4, 3)
output = uop_1647
output2 = uop_1647
func_1657 = relay.Function([], output)
mod['func_1657'] = func_1657
mod = relay.transform.InferType()(mod)
mutated_mod['func_1657'] = func_1657
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1657_call = mutated_mod.get_global_var('func_1657')
call_1658 = func_1657_call()
output = call_1658
func_1659 = relay.Function([], output)
mutated_mod['func_1659'] = func_1659
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1660 = relay.var("var_1660", dtype = "bool", shape = (8, 1, 5))#candidate|1660|(8, 1, 5)|var|bool
var_1661 = relay.var("var_1661", dtype = "bool", shape = (8, 8, 5))#candidate|1661|(8, 8, 5)|var|bool
bop_1662 = relay.logical_or(var_1660.astype('bool'), var_1661.astype('bool')) # shape=(8, 8, 5)
func_251_call = mod.get_global_var('func_251')
func_255_call = mutated_mod.get_global_var('func_255')
const_1670 = relay.const([8.560818,-8.792939,3.165119,7.857548,-3.535356,3.404036,9.339095,-7.100399,7.883300,-1.370858,5.083784,-9.857038,-1.604025,4.758914,9.207952,1.618738,8.487555,4.178078,-8.120493,1.890790,-2.986650,-9.645053,-4.336104,7.676124,-1.831559,-0.101059,-3.451983,3.745941,7.136710,-2.599490,5.478851,3.408721,-3.029397,8.445192,-0.011365,3.381490,-0.849459,-6.812425,-1.506677,-7.149698,4.600090,-9.690173,-1.841802,-3.593259,3.970993,-3.730305,9.134381,5.365665,-1.889079,-1.702110,5.415645,4.972590,9.084063,-2.702378,7.424090,3.808559,1.577596,2.048926,1.105981,-8.965431,-3.043240,8.296137,-4.992972,3.290233,-3.449222,1.138947,-4.167374,4.856067,-0.640900,3.834347,-1.259264,1.633284,2.378444,1.253748,-6.588732,9.572109,1.612292,-6.381922,3.666211,-0.787688,2.506448,8.610492,6.670927,-1.986288,5.069803,3.549067,-1.676385,4.548559,1.434574,-0.807611,-9.125481,9.859902,1.044423,-2.906811,9.625360,-3.612574,-2.513297,-2.621274,-3.747361,-1.509465,5.341597,5.656080,-3.749217,5.342608,-4.392078,8.088883,4.140642,-1.432633,0.995219,9.771884,7.670615,6.777094,3.004673,4.357200,-5.946527,2.598332,-8.140408,5.321408,-4.471022,-8.842934,3.106295,2.175241,-9.097705,2.794622,-5.789389,9.780780,-8.060301,-6.787155], dtype = "float64")#candidate|1670|(128,)|const|float64
var_1671 = relay.var("var_1671", dtype = "float64", shape = (9,))#candidate|1671|(9,)|var|float64
call_1669 = relay.TupleGetItem(func_251_call(relay.reshape(const_1670.astype('float64'), [8, 16]), relay.reshape(var_1671.astype('float64'), [9,]), ), 9)
call_1672 = relay.TupleGetItem(func_255_call(relay.reshape(const_1670.astype('float64'), [8, 16]), relay.reshape(var_1671.astype('float64'), [9,]), ), 9)
uop_1678 = relay.acos(const_1670.astype('float64')) # shape=(128,)
bop_1680 = relay.less(uop_1678.astype('bool'), relay.reshape(const_1670.astype('bool'), relay.shape_of(uop_1678))) # shape=(128,)
bop_1683 = relay.power(bop_1662.astype('float64'), relay.reshape(var_1661.astype('float64'), relay.shape_of(bop_1662))) # shape=(8, 8, 5)
uop_1688 = relay.sigmoid(uop_1678.astype('float32')) # shape=(128,)
output = relay.Tuple([call_1669,var_1671,bop_1680,bop_1683,uop_1688,])
output2 = relay.Tuple([call_1672,var_1671,bop_1680,bop_1683,uop_1688,])
func_1690 = relay.Function([var_1660,var_1661,var_1671,], output)
mod['func_1690'] = func_1690
mod = relay.transform.InferType()(mod)
var_1691 = relay.var("var_1691", dtype = "bool", shape = (8, 1, 5))#candidate|1691|(8, 1, 5)|var|bool
var_1692 = relay.var("var_1692", dtype = "bool", shape = (8, 8, 5))#candidate|1692|(8, 8, 5)|var|bool
var_1693 = relay.var("var_1693", dtype = "float64", shape = (9,))#candidate|1693|(9,)|var|float64
output = func_1690(var_1691,var_1692,var_1693,)
func_1694 = relay.Function([var_1691,var_1692,var_1693,], output)
mutated_mod['func_1694'] = func_1694
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1714 = relay.var("var_1714", dtype = "float32", shape = (6, 2, 14))#candidate|1714|(6, 2, 14)|var|float32
uop_1715 = relay.sin(var_1714.astype('float32')) # shape=(6, 2, 14)
const_1717 = relay.const([[[0.198529,-2.835512,6.237020,-6.411345,-2.926256,5.897949,-7.092766,5.949160,9.109354,4.333989,9.455178,9.421271,8.620958,-0.673488],[8.599482,2.332117,6.210206,8.324849,-0.162382,-4.822700,-4.252464,-6.766300,4.828941,-8.166506,-0.009147,3.164509,7.929247,6.555219]],[[-7.682510,7.003333,-1.327648,-5.867467,-8.282537,-8.526240,-8.589381,4.692518,-1.736367,2.084218,6.695582,-5.465747,3.999355,2.644679],[-1.497150,1.527910,1.455901,-5.456796,-7.238396,5.153922,-2.795680,3.532258,5.099749,-8.089093,-2.666936,1.219616,2.954581,-9.268811]],[[1.305878,-0.626811,-8.349545,1.819077,-4.444087,9.898874,-4.020234,2.332082,-3.992021,9.469463,2.422995,2.661023,0.983459,-0.039182],[-2.286205,5.161621,-9.295212,5.228295,-1.227078,-0.810912,6.198251,3.600164,9.515685,-8.645682,6.467138,3.741474,5.535335,7.120729]],[[9.327281,-1.276073,-3.815373,-6.255747,6.691579,5.195815,-9.806978,-1.365893,-8.243389,5.921393,-9.565065,4.361810,-1.898147,-5.732591],[-2.736722,-3.681169,0.244143,9.923650,5.371200,0.865007,-4.700700,-6.704103,-0.258320,-2.875754,-0.602882,-8.685627,4.679594,-0.644479]],[[-5.977178,-8.808741,-1.829568,7.312733,4.430361,9.948221,-8.287517,3.078986,-4.326117,6.342095,-0.342276,0.567498,1.487902,2.156793],[4.187598,-3.038742,-4.828008,-1.388863,-7.335246,1.762522,7.025879,-6.352074,5.422637,-4.130524,7.855170,-1.276837,9.244124,-0.044724]],[[-8.129988,3.269282,4.132582,6.707252,-7.894517,0.079958,7.052321,-6.948247,3.721057,-7.009602,-8.496386,-8.427703,-9.529671,2.142232],[-4.696762,5.521658,2.392032,0.887088,-9.142229,5.159759,-1.285078,0.702487,0.178790,-0.780924,-2.474303,9.327311,4.103765,-9.680432]]], dtype = "float32")#candidate|1717|(6, 2, 14)|const|float32
bop_1718 = relay.maximum(uop_1715.astype('uint8'), relay.reshape(const_1717.astype('uint8'), relay.shape_of(uop_1715))) # shape=(6, 2, 14)
bop_1722 = relay.less_equal(bop_1718.astype('bool'), relay.reshape(var_1714.astype('bool'), relay.shape_of(bop_1718))) # shape=(6, 2, 14)
func_544_call = mod.get_global_var('func_544')
func_547_call = mutated_mod.get_global_var('func_547')
const_1728 = relay.const([-2.161802,1.096663,7.731728,-5.739933,1.160028,6.858012,5.154697,8.583001,-8.614684,8.384899,2.079276,9.149761,1.786350,-4.224161,-2.605117,1.560075,4.566890,-7.405925,0.237416,1.983848,8.848236,8.754189,7.940382,3.164089,5.689556,-1.741077,-7.842355,-6.797558,5.518896,-8.681989,-4.374674,-9.377772,3.572405,-2.762722,3.583518,6.229023,-4.902500,-9.903629,4.769288,6.342798,-4.891700,3.770874,8.538881,6.482770,-3.724388,-9.873845,2.129627,-5.790491], dtype = "float64")#candidate|1728|(48,)|const|float64
call_1727 = func_544_call(relay.reshape(const_1728.astype('float64'), [3, 16]))
call_1729 = func_544_call(relay.reshape(const_1728.astype('float64'), [3, 16]))
bop_1731 = relay.floor_mod(bop_1722.astype('float32'), relay.reshape(uop_1715.astype('float32'), relay.shape_of(bop_1722))) # shape=(6, 2, 14)
output = relay.Tuple([call_1727,const_1728,bop_1731,])
output2 = relay.Tuple([call_1729,const_1728,bop_1731,])
func_1736 = relay.Function([var_1714,], output)
mod['func_1736'] = func_1736
mod = relay.transform.InferType()(mod)
var_1737 = relay.var("var_1737", dtype = "float32", shape = (6, 2, 14))#candidate|1737|(6, 2, 14)|var|float32
output = func_1736(var_1737)
func_1738 = relay.Function([var_1737], output)
mutated_mod['func_1738'] = func_1738
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1759 = relay.var("var_1759", dtype = "float32", shape = (11, 12, 7))#candidate|1759|(11, 12, 7)|var|float32
var_1760 = relay.var("var_1760", dtype = "float32", shape = (11, 12, 7))#candidate|1760|(11, 12, 7)|var|float32
bop_1761 = relay.divide(var_1759.astype('float32'), relay.reshape(var_1760.astype('float32'), relay.shape_of(var_1759))) # shape=(11, 12, 7)
output = bop_1761
output2 = bop_1761
func_1766 = relay.Function([var_1759,var_1760,], output)
mod['func_1766'] = func_1766
mod = relay.transform.InferType()(mod)
var_1767 = relay.var("var_1767", dtype = "float32", shape = (11, 12, 7))#candidate|1767|(11, 12, 7)|var|float32
var_1768 = relay.var("var_1768", dtype = "float32", shape = (11, 12, 7))#candidate|1768|(11, 12, 7)|var|float32
output = func_1766(var_1767,var_1768,)
func_1769 = relay.Function([var_1767,var_1768,], output)
mutated_mod['func_1769'] = func_1769
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1775 = relay.var("var_1775", dtype = "float32", shape = (12, 6, 6))#candidate|1775|(12, 6, 6)|var|float32
uop_1776 = relay.atan(var_1775.astype('float32')) # shape=(12, 6, 6)
output = relay.Tuple([uop_1776,])
output2 = relay.Tuple([uop_1776,])
F = relay.Function([var_1775,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1775,], output2)
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
input_1775= np.array([[[8.978292,1.442033,-7.434887,-5.126408,8.497456,-1.548165],[8.091482,-5.647164,-3.892121,-4.525860,9.343953,-8.283697],[-8.806394,-5.310651,-5.710373,6.186982,-2.385126,-1.239227],[-1.763352,6.849477,-5.049969,-4.651952,2.423591,4.157071],[7.002622,-7.065572,-6.467194,-3.901491,8.308927,-8.013736],[-6.339126,1.145164,-3.201243,9.426591,6.415961,-1.754008]],[[-1.989836,6.948316,0.875181,-1.170481,-4.029219,4.208697],[1.345329,9.510637,2.537383,-7.635848,1.117959,-9.861090],[5.222726,-1.611041,-0.300830,-0.863625,-5.387915,-4.455093],[2.174021,-1.925579,-8.141105,-5.446994,-3.874732,-7.445086],[3.731464,2.111528,-3.778586,-3.398823,1.220567,-7.650013],[1.709191,-0.576584,-0.277847,1.552103,8.956072,-9.125981]],[[-2.963505,-4.096739,-2.512461,-2.493126,5.915682,8.928295],[2.044737,-4.770036,-5.686282,-6.354137,9.846850,6.598940],[-9.971282,3.580472,4.901797,5.523916,6.987629,-8.018639],[-6.839128,3.683934,-3.856871,6.008876,2.550924,-7.139738],[7.177295,5.120202,7.410409,-1.810690,-6.924893,-0.302226],[9.501673,3.839565,0.011075,2.373828,-8.229085,8.038197]],[[1.525064,-5.679039,-3.551459,-2.147568,8.272905,7.947042],[5.427878,-0.048685,4.496917,7.059272,-0.680468,5.061029],[-0.647267,0.062644,-8.469642,2.142860,-1.204301,-5.209431],[-1.631215,-5.601658,-9.106059,-7.941466,8.626162,9.343131],[5.315688,-5.486435,1.239370,5.851266,-7.480829,-6.302296],[-8.053269,-3.552507,5.743749,2.453560,0.364238,-3.444021]],[[8.269723,0.509538,7.804171,-2.363055,1.290329,5.390117],[-9.513133,-6.487810,2.118722,8.738621,-1.666819,-6.267372],[-2.217353,7.495473,9.779409,-0.679669,1.801023,-1.962625],[-2.451339,5.049332,5.389044,6.381384,-1.654004,-7.267960],[-9.167570,1.287841,8.209330,9.187198,5.427891,2.479576],[-3.038320,-1.710954,-4.494086,-2.096937,1.618848,-1.683747]],[[-1.130318,-0.034577,-4.024315,2.105473,0.042501,6.210927],[-4.736178,-2.495343,-5.367635,5.684474,6.007042,-8.061119],[-6.194305,4.613978,-8.272799,-4.785905,6.771539,9.460162],[4.859268,-2.722937,4.985237,5.209803,-4.664928,-8.054619],[7.301302,7.494504,7.836482,-8.719162,6.701837,1.996981],[-2.085598,-7.799068,-6.227521,6.611044,-0.669408,-3.356158]],[[0.972178,-6.208122,-1.584503,-0.831050,-0.695295,-9.569357],[5.511132,4.305169,-1.769271,5.369944,-5.237237,-6.916044],[4.123982,-8.392220,-2.732410,-1.414225,-5.313345,7.656688],[2.282943,-8.724206,-3.721644,8.502049,-7.413294,-6.355003],[6.786489,8.232556,1.104436,-3.798922,2.912210,2.551069],[-4.279964,5.461979,-4.005762,3.105395,8.062780,4.382142]],[[-9.629397,7.741477,4.654321,-3.395674,4.817361,-3.267519],[9.065663,-5.241153,3.083155,-7.789592,5.879539,-1.320292],[7.702586,8.546947,1.875602,-0.015466,3.876790,-1.429671],[1.390906,8.190585,-5.448741,3.022168,-0.681938,-4.732214],[0.564777,3.699211,7.086072,-2.592317,6.188197,3.680801],[-5.801708,8.855241,9.273569,-0.119204,-8.008465,-2.613612]],[[2.210557,-4.214521,9.165232,0.733151,-2.528093,3.818725],[-8.862043,-1.726907,5.386554,4.867346,2.086938,9.090279],[-3.098904,8.648864,8.675153,-8.069878,1.468220,-6.389644],[9.916643,-7.383473,1.445332,2.028347,6.432228,0.097263],[0.644018,3.029137,7.220611,-5.073450,-6.647305,-6.300126],[7.322935,1.073707,-3.208295,8.141242,7.390367,8.984368]],[[3.235816,1.069484,-7.066960,7.741816,1.620818,-0.529999],[1.808195,-1.944511,-9.843746,6.581571,7.034978,-5.485842],[-7.786779,-3.444153,8.400100,-4.292354,-7.391600,-4.481513],[6.252047,3.374162,-4.951720,9.819327,-4.747952,4.492504],[7.893343,-0.634251,1.402966,-7.484832,-9.609982,-5.391793],[-0.985135,-2.212052,-4.026936,3.089428,-8.356104,-2.413322]],[[-9.149172,-1.212596,5.594587,0.809518,-5.661283,-7.586438],[-7.870484,0.089740,8.962650,9.546840,0.655436,-6.967929],[-0.424115,6.783046,-0.925018,4.054146,2.891094,2.497593],[8.714166,6.319505,-0.643684,-8.580971,-3.504458,1.101492],[8.175089,2.010232,-1.181791,-7.567518,2.390260,3.992707],[4.231901,-3.573783,-4.623100,3.801048,-1.720458,1.570360]],[[-0.581298,-8.626640,-3.983061,-7.028519,-6.600098,-6.658501],[-1.913921,5.346811,-3.045887,1.940928,0.062565,-9.228202],[-3.342185,3.351565,7.383369,1.239338,-8.570790,-3.875062],[-2.501467,-4.711606,6.489470,6.974463,5.016299,8.756630],[-1.343789,0.293282,0.611735,4.181605,-5.457101,-1.591242],[-2.851484,2.690712,3.965477,3.420096,4.589757,-3.403213]]], dtype='float32')
module1.set_input('var_1775', input_1775)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1775, )
res3 = intrp3.evaluate()(input_1775, )
res4 = intrp4.evaluate()(input_1775, )
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
module5.set_input('var_1775', input_1775)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1775, )
res7 = intrp7.evaluate()(input_1775, )
res8 = intrp8.evaluate()(input_1775, )
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
module9.set_input('var_1775', input_1775)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1775, )
res11 = intrp11.evaluate()(input_1775, )
res12 = intrp12.evaluate()(input_1775, )
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
module13.set_input('var_1775', input_1775)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1775, )
res15 = intrp15.evaluate()(input_1775, )
res16 = intrp16.evaluate()(input_1775, )
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
module17.set_input('var_1775', input_1775)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1775, )
res19 = intrp19.evaluate()(input_1775, )
res20 = intrp20.evaluate()(input_1775, )
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
module21.set_input('var_1775', input_1775)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1775, )
res23 = intrp23.evaluate()(input_1775, )
res24 = intrp24.evaluate()(input_1775, )
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