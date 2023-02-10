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
const_46 = relay.const([[-3.997598,7.575467,-8.243411,2.425016,-9.011590,-6.168144,8.182537,-4.074703,5.246165,-8.624217,7.443425,-6.783352,6.603527,3.728286,-7.257629,-1.949969],[6.546779,-4.935660,8.019516,2.508540,-5.406810,-0.708853,-0.850486,-2.196306,7.969381,-9.442255,3.397478,3.632787,-9.570895,-0.367047,9.085636,6.558391],[7.294306,-0.928685,2.781546,-5.517292,2.895013,6.183633,1.484938,1.461563,-5.720993,7.607142,4.592772,-5.957510,-7.373047,-4.080756,4.363743,-2.025720],[-4.474429,0.256770,8.059888,3.263017,5.779682,-1.458107,4.244374,9.872052,7.449113,1.805056,2.749560,-0.468453,-3.066764,-1.008287,-3.354917,-9.288230],[8.434911,-3.563702,7.594721,-7.909292,-4.799671,-4.346256,-1.796500,-9.237747,-7.512285,-4.308418,8.545780,9.628554,-8.351595,-2.548032,9.828111,1.743439],[-4.809430,3.112375,6.190438,-3.075248,-5.850809,3.579245,-5.480421,-0.898035,7.594304,9.305378,-8.169540,2.130651,-4.571181,7.113931,-5.536883,-9.636080],[1.162358,7.068527,2.720210,-7.469833,-8.528090,7.051177,3.559891,5.957343,-6.765690,-7.289302,-1.408544,4.107195,1.676382,-9.891818,5.725762,-8.766075],[-0.959145,-8.287255,-1.468676,1.364425,-5.595194,3.658784,-7.469970,4.396947,4.503229,1.812827,9.598393,-1.762295,-4.844755,-2.269304,-4.306362,-4.022191],[-4.257696,-7.633721,7.531163,0.787019,-5.806915,-9.921274,-3.457447,-6.681425,1.473845,6.217258,-4.180603,3.786406,-5.359309,2.080299,-9.154873,1.656974],[-4.083024,4.520771,0.465517,5.513240,-1.784699,-6.453633,6.380132,-2.511962,2.630024,3.220778,-1.242510,-1.410414,6.043222,8.352019,1.606577,-9.903985],[-0.330528,0.667690,8.716528,-4.137122,-6.433589,-6.068052,4.853234,-7.395199,-0.431406,-3.038095,-5.738154,7.510108,-5.198466,-2.573313,7.640456,0.290375],[-0.227639,5.239957,-9.224970,2.215852,-4.645645,-3.393634,4.156202,-8.413884,9.663439,-4.133744,9.292653,6.368098,-7.074085,2.273622,-2.735153,5.148568]], dtype = "float64")#candidate|46|(12, 16)|const|float64
uop_47 = relay.atan(const_46.astype('float64')) # shape=(12, 16)
uop_49 = relay.sinh(uop_47.astype('float32')) # shape=(12, 16)
var_53 = relay.var("var_53", dtype = "float32", shape = (12, 16))#candidate|53|(12, 16)|var|float32
bop_54 = relay.less_equal(uop_49.astype('bool'), relay.reshape(var_53.astype('bool'), relay.shape_of(uop_49))) # shape=(12, 16)
output = bop_54
output2 = bop_54
func_59 = relay.Function([var_53,], output)
mod['func_59'] = func_59
mod = relay.transform.InferType()(mod)
mutated_mod['func_59'] = func_59
mutated_mod = relay.transform.InferType()(mutated_mod)
var_60 = relay.var("var_60", dtype = "float32", shape = (12, 16))#candidate|60|(12, 16)|var|float32
func_59_call = mutated_mod.get_global_var('func_59')
call_61 = func_59_call(var_60)
output = call_61
func_62 = relay.Function([var_60], output)
mutated_mod['func_62'] = func_62
mutated_mod = relay.transform.InferType()(mutated_mod)
var_67 = relay.var("var_67", dtype = "float32", shape = (1, 4))#candidate|67|(1, 4)|var|float32
uop_68 = relay.erf(var_67.astype('float32')) # shape=(1, 4)
uop_70 = relay.sigmoid(uop_68.astype('float64')) # shape=(1, 4)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
const_73 = relay.const([-6.497131,-1.760981,7.045564,6.036763,5.678040,6.432540,8.753900,7.331134,9.769418,-2.665590,-1.929143,-4.182665,1.588242,-7.480831,-5.192354,3.422679,-0.665476,-7.167496,3.893157,2.720950,-9.560911,-4.958560,6.301551,0.397165,7.439365,8.416456,-2.240913,0.972631,-9.078021,-0.088175,0.311483,7.513832,7.370696,-0.116465,8.604325,-9.123866,-8.128262,3.017820,-0.477108,-0.802759,-4.640607,6.102588,-7.787486,-0.933075,1.836070,6.016915,-9.255718,-6.145252,0.163578,7.506368,-4.416052,-3.999359,-0.217892,-1.790164,-6.743333,0.320748,-7.531694,6.936198,7.683934,0.693150,0.097131,-7.912711,5.662181,7.680184,4.418249,0.498033,-9.903008,-6.034744,3.715559,-8.620481,-8.037114,-6.288401,-1.073173,8.269573,5.044379,-3.466969,5.638094,6.824355,5.285616,7.592419,2.559015,-0.740241,5.547108,-4.333213,0.412704,-4.758063,-2.207604,9.307217,6.119836,1.616990,8.306014,0.822400,-6.878174,-4.433853,8.260985,-4.547908,-0.373274,2.204502,1.672920,5.051157,-2.887732,-9.667108,-5.310708,7.623617,-8.215569,9.948326,6.531646,4.883279,1.895174,5.967551,4.741427,4.205400,7.778925,0.950591,-0.202196,2.932247,6.101798,-7.079438,6.745422,3.603375,-5.259163,-6.232518,3.448678,0.679363,8.157571,4.999853,-1.855313,-3.152559,9.898965,-7.126900,-3.858236,-6.269755,-4.601308,-4.728277,0.130070,4.027064,-3.907334,2.823845,4.251901,-6.101915,2.159216,-0.716599,-6.990535,2.529091,-4.659046,6.986117,-3.550264,-2.758768,-0.229929,-1.090500,-3.484987,2.958578,8.064440,-0.235592,7.743025,5.813591,2.465453,-5.398689,5.221535,7.619510,-6.927716,-4.437803,0.158319,4.871972,4.020369,0.256429,7.827013,-6.272777,6.135549,-1.213458,7.273578,5.277243,7.668908,-4.180127,-0.161839,-5.057526,2.669087,2.814583,-2.225815,-2.253904,-5.015660,9.919193,1.159051,9.903996,4.278054,5.322890,5.459387,2.830044,3.285047,4.148559,4.177596,-5.712219], dtype = "float32")#candidate|73|(192,)|const|float32
call_72 = func_59_call(relay.reshape(const_73.astype('float32'), [12, 16]))
call_74 = func_59_call(relay.reshape(const_73.astype('float32'), [12, 16]))
bop_76 = relay.mod(uop_68.astype('float64'), relay.reshape(uop_70.astype('float64'), relay.shape_of(uop_68))) # shape=(1, 4)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_81 = func_59_call(relay.reshape(const_73.astype('float32'), [12, 16]))
call_82 = func_59_call(relay.reshape(const_73.astype('float32'), [12, 16]))
var_83 = relay.var("var_83", dtype = "float32", shape = (9, 4))#candidate|83|(9, 4)|var|float32
bop_84 = relay.divide(uop_68.astype('float64'), var_83.astype('float64')) # shape=(9, 4)
const_87 = relay.const([[-7.734030,9.220965,5.538353,-2.405071],[7.290283,-3.602960,-1.168196,2.829737],[-5.209086,8.949843,-2.402626,2.156562],[2.205469,-4.767658,8.788820,6.084797],[2.820589,8.457891,-1.376435,2.257138],[2.163701,-7.054450,-5.998918,-7.729579],[-0.245547,1.910919,-2.721189,-7.155725],[7.810289,-3.742652,-7.181529,-8.874289],[3.828583,4.747926,-8.667717,4.938756],[-6.797011,0.521933,2.022406,0.971368],[3.814059,0.360458,-9.217715,-9.480943],[-1.751650,-2.663244,-3.774244,-5.214988],[-0.617912,-4.697094,6.906937,-3.883393],[5.917297,-7.811544,-2.094905,-7.132104],[9.667315,4.102247,4.317471,4.618093]], dtype = "float32")#candidate|87|(15, 4)|const|float32
bop_88 = relay.left_shift(uop_68.astype('uint64'), const_87.astype('uint64')) # shape=(15, 4)
uop_91 = relay.log10(uop_70.astype('float32')) # shape=(1, 4)
uop_96 = relay.atanh(uop_91.astype('float32')) # shape=(1, 4)
bop_99 = relay.add(uop_68.astype('int64'), var_83.astype('int64')) # shape=(9, 4)
output = relay.Tuple([call_72,const_73,bop_76,call_81,bop_84,bop_88,uop_96,bop_99,])
output2 = relay.Tuple([call_74,const_73,bop_76,call_82,bop_84,bop_88,uop_96,bop_99,])
func_102 = relay.Function([var_67,var_83,], output)
mod['func_102'] = func_102
mod = relay.transform.InferType()(mod)
var_103 = relay.var("var_103", dtype = "float32", shape = (1, 4))#candidate|103|(1, 4)|var|float32
var_104 = relay.var("var_104", dtype = "float32", shape = (9, 4))#candidate|104|(9, 4)|var|float32
output = func_102(var_103,var_104,)
func_105 = relay.Function([var_103,var_104,], output)
mutated_mod['func_105'] = func_105
mutated_mod = relay.transform.InferType()(mutated_mod)
var_111 = relay.var("var_111", dtype = "float32", shape = (8, 3))#candidate|111|(8, 3)|var|float32
uop_112 = relay.atan(var_111.astype('float32')) # shape=(8, 3)
bop_114 = relay.add(var_111.astype('int8'), relay.reshape(uop_112.astype('int8'), relay.shape_of(var_111))) # shape=(8, 3)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
var_119 = relay.var("var_119", dtype = "float32", shape = (192,))#candidate|119|(192,)|var|float32
call_118 = func_59_call(relay.reshape(var_119.astype('float32'), [12, 16]))
call_120 = func_59_call(relay.reshape(var_119.astype('float32'), [12, 16]))
uop_123 = relay.log10(bop_114.astype('float32')) # shape=(8, 3)
uop_128 = relay.acosh(uop_123.astype('float64')) # shape=(8, 3)
bop_132 = relay.subtract(uop_128.astype('int32'), relay.reshape(uop_112.astype('int32'), relay.shape_of(uop_128))) # shape=(8, 3)
bop_135 = relay.bitwise_and(uop_128.astype('int8'), relay.reshape(bop_132.astype('int8'), relay.shape_of(uop_128))) # shape=(8, 3)
bop_140 = relay.logical_xor(bop_135.astype('uint32'), relay.reshape(bop_114.astype('uint32'), relay.shape_of(bop_135))) # shape=(8, 3)
uop_144 = relay.acos(bop_132.astype('float64')) # shape=(8, 3)
bop_146 = relay.greater_equal(uop_123.astype('bool'), relay.reshape(bop_140.astype('bool'), relay.shape_of(uop_123))) # shape=(8, 3)
uop_153 = relay.exp(uop_144.astype('float32')) # shape=(8, 3)
var_157 = relay.var("var_157", dtype = "float32", shape = (8, 3))#candidate|157|(8, 3)|var|float32
bop_158 = relay.bitwise_xor(uop_153.astype('int32'), relay.reshape(var_157.astype('int32'), relay.shape_of(uop_153))) # shape=(8, 3)
uop_161 = relay.atanh(bop_158.astype('float64')) # shape=(8, 3)
uop_164 = relay.sin(bop_158.astype('float32')) # shape=(8, 3)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_170 = func_59_call(relay.reshape(call_118.astype('float32'), [12, 16]))
call_171 = func_59_call(relay.reshape(call_118.astype('float32'), [12, 16]))
bop_175 = relay.less_equal(uop_164.astype('bool'), relay.reshape(bop_140.astype('bool'), relay.shape_of(uop_164))) # shape=(8, 3)
func_102_call = mod.get_global_var('func_102')
func_105_call = mutated_mod.get_global_var('func_105')
const_189 = relay.const([1.626216,3.659109,7.759751,-2.240268], dtype = "float32")#candidate|189|(4,)|const|float32
const_190 = relay.const([6.788115,-4.727765,9.347691,1.965845,-4.513394,-5.642782,-5.863351,5.639954,3.414916,1.579458,-4.310220,-1.620154,-7.876988,8.117539,3.444616,3.104986,7.252570,-4.703419,3.712419,4.338157,1.322859,-3.184589,-1.621995,0.293871,7.582104,6.769026,-2.153268,-7.301787,-0.039032,-5.453722,-1.234956,-0.385968,1.137028,-5.546883,-5.501776,-2.267233], dtype = "float32")#candidate|190|(36,)|const|float32
call_188 = relay.TupleGetItem(func_102_call(relay.reshape(const_189.astype('float32'), [1, 4]), relay.reshape(const_190.astype('float32'), [9, 4]), ), 3)
call_191 = relay.TupleGetItem(func_105_call(relay.reshape(const_189.astype('float32'), [1, 4]), relay.reshape(const_190.astype('float32'), [9, 4]), ), 3)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_192 = func_59_call(relay.reshape(call_188.astype('float32'), [12, 16]))
call_193 = func_59_call(relay.reshape(call_188.astype('float32'), [12, 16]))
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_194 = func_59_call(relay.reshape(var_119.astype('float32'), [12, 16]))
call_195 = func_59_call(relay.reshape(var_119.astype('float32'), [12, 16]))
uop_196 = relay.cos(uop_144.astype('float32')) # shape=(8, 3)
uop_209 = relay.sigmoid(uop_196.astype('float32')) # shape=(8, 3)
bop_218 = relay.right_shift(bop_175.astype('uint16'), relay.reshape(uop_112.astype('uint16'), relay.shape_of(bop_175))) # shape=(8, 3)
bop_227 = relay.minimum(uop_209.astype('int32'), relay.reshape(bop_218.astype('int32'), relay.shape_of(uop_209))) # shape=(8, 3)
var_231 = relay.var("var_231", dtype = "int32", shape = (8, 3))#candidate|231|(8, 3)|var|int32
bop_232 = relay.logical_and(bop_227.astype('bool'), relay.reshape(var_231.astype('bool'), relay.shape_of(bop_227))) # shape=(8, 3)
bop_235 = relay.bitwise_xor(bop_175.astype('int8'), relay.reshape(uop_164.astype('int8'), relay.shape_of(bop_175))) # shape=(8, 3)
func_102_call = mod.get_global_var('func_102')
func_105_call = mutated_mod.get_global_var('func_105')
call_238 = relay.TupleGetItem(func_102_call(relay.reshape(const_189.astype('float32'), [1, 4]), relay.reshape(const_190.astype('float32'), [9, 4]), ), 0)
call_239 = relay.TupleGetItem(func_105_call(relay.reshape(const_189.astype('float32'), [1, 4]), relay.reshape(const_190.astype('float32'), [9, 4]), ), 0)
bop_241 = relay.power(bop_232.astype('float64'), relay.reshape(uop_123.astype('float64'), relay.shape_of(bop_232))) # shape=(8, 3)
bop_247 = relay.divide(bop_132.astype('float64'), relay.reshape(var_111.astype('float64'), relay.shape_of(bop_132))) # shape=(8, 3)
output = relay.Tuple([call_118,var_119,bop_146,uop_161,call_170,call_188,const_189,const_190,call_192,call_194,bop_235,call_238,bop_241,bop_247,])
output2 = relay.Tuple([call_120,var_119,bop_146,uop_161,call_171,call_191,const_189,const_190,call_193,call_195,bop_235,call_239,bop_241,bop_247,])
func_270 = relay.Function([var_111,var_119,var_157,var_231,], output)
mod['func_270'] = func_270
mod = relay.transform.InferType()(mod)
var_271 = relay.var("var_271", dtype = "float32", shape = (8, 3))#candidate|271|(8, 3)|var|float32
var_272 = relay.var("var_272", dtype = "float32", shape = (192,))#candidate|272|(192,)|var|float32
var_273 = relay.var("var_273", dtype = "float32", shape = (8, 3))#candidate|273|(8, 3)|var|float32
var_274 = relay.var("var_274", dtype = "int32", shape = (8, 3))#candidate|274|(8, 3)|var|int32
output = func_270(var_271,var_272,var_273,var_274,)
func_275 = relay.Function([var_271,var_272,var_273,var_274,], output)
mutated_mod['func_275'] = func_275
mutated_mod = relay.transform.InferType()(mutated_mod)
var_292 = relay.var("var_292", dtype = "float64", shape = (13, 11))#candidate|292|(13, 11)|var|float64
const_293 = relay.const([[-3.904953,-9.086287,9.630434,-7.937284,-2.267305,-9.654279,-4.891612,-5.366591,-1.975194,8.109252,-7.332038],[-1.425258,-9.406639,-5.281396,-9.517223,9.345033,3.889301,6.337117,2.051344,-2.167626,2.737295,-2.987527],[5.731391,1.750881,9.766676,8.128236,2.055175,-1.090851,6.149386,7.260147,-8.728053,-4.458197,-1.760370],[-3.183429,1.205129,7.897552,-4.391986,-6.812239,-2.394154,5.129825,-7.591034,2.819706,-9.382073,-6.630434],[-1.917771,-5.879795,-0.620548,-9.886319,-0.724369,4.796520,-3.496747,0.895617,-8.810991,-5.530434,4.675367],[1.841599,1.665462,9.757209,-0.020058,7.326177,9.129343,8.220039,8.490615,9.455916,7.723014,0.372301],[0.971149,-7.320568,2.701133,9.368694,-4.051239,-5.203270,3.557667,4.244141,1.998670,3.945363,-3.362674],[4.959425,-2.631965,-6.797608,5.638812,5.073789,-6.979472,-4.333643,7.474305,-7.557768,-8.821754,3.334975],[-3.018264,1.381131,2.824908,3.717127,-8.581967,0.261706,4.055268,-5.157821,-1.190131,5.512435,4.305761],[-5.613578,3.842451,-6.704213,4.045435,-8.724605,-8.101427,-0.948740,-8.869794,-4.618867,-9.135211,3.060586],[8.305198,-8.138607,-3.981817,-5.904956,0.697664,-2.956759,9.014379,8.720321,-4.627537,-7.507951,6.085464],[-8.050149,-1.773617,-0.845984,9.830950,-7.229703,6.772868,-1.668851,7.803080,4.361408,5.687508,9.131410],[-0.331653,4.783830,-9.206520,-1.625396,-3.143059,7.062264,9.702422,9.362704,-4.865414,0.744816,-9.112724]], dtype = "float64")#candidate|293|(13, 11)|const|float64
bop_294 = relay.add(var_292.astype('float64'), relay.reshape(const_293.astype('float64'), relay.shape_of(var_292))) # shape=(13, 11)
output = bop_294
output2 = bop_294
func_299 = relay.Function([var_292,], output)
mod['func_299'] = func_299
mod = relay.transform.InferType()(mod)
var_300 = relay.var("var_300", dtype = "float64", shape = (13, 11))#candidate|300|(13, 11)|var|float64
output = func_299(var_300)
func_301 = relay.Function([var_300], output)
mutated_mod['func_301'] = func_301
mutated_mod = relay.transform.InferType()(mutated_mod)
var_303 = relay.var("var_303", dtype = "uint32", shape = (14, 9, 11))#candidate|303|(14, 9, 11)|var|uint32
var_304 = relay.var("var_304", dtype = "uint32", shape = (14, 9, 11))#candidate|304|(14, 9, 11)|var|uint32
bop_305 = relay.equal(var_303.astype('bool'), relay.reshape(var_304.astype('bool'), relay.shape_of(var_303))) # shape=(14, 9, 11)
uop_314 = relay.cosh(var_304.astype('float32')) # shape=(14, 9, 11)
bop_318 = relay.bitwise_xor(uop_314.astype('int8'), relay.reshape(var_304.astype('int8'), relay.shape_of(uop_314))) # shape=(14, 9, 11)
func_270_call = mod.get_global_var('func_270')
func_275_call = mutated_mod.get_global_var('func_275')
const_322 = relay.const([1.535814,8.306532,-3.643241,-5.790231,5.680814,-4.549828,0.994491,-5.631664,-5.653380,-6.199923,8.454475,4.957264,1.174483,9.042480,-9.201671,-2.536793,-8.841489,-2.751670,-7.466986,-2.110173,9.782966,-4.124011,-4.914412,-6.783383], dtype = "float32")#candidate|322|(24,)|const|float32
var_323 = relay.var("var_323", dtype = "float32", shape = (192,))#candidate|323|(192,)|var|float32
call_321 = relay.TupleGetItem(func_270_call(relay.reshape(const_322.astype('float32'), [8, 3]), relay.reshape(var_323.astype('float32'), [192,]), relay.reshape(const_322.astype('float32'), [8, 3]), relay.reshape(const_322.astype('int32'), [8, 3]), ), 1)
call_324 = relay.TupleGetItem(func_275_call(relay.reshape(const_322.astype('float32'), [8, 3]), relay.reshape(var_323.astype('float32'), [192,]), relay.reshape(const_322.astype('float32'), [8, 3]), relay.reshape(const_322.astype('int32'), [8, 3]), ), 1)
bop_329 = relay.logical_or(bop_318.astype('bool'), relay.reshape(uop_314.astype('bool'), relay.shape_of(bop_318))) # shape=(14, 9, 11)
uop_332 = relay.log2(bop_329.astype('float32')) # shape=(14, 9, 11)
uop_343 = relay.atan(bop_329.astype('float32')) # shape=(14, 9, 11)
output = relay.Tuple([bop_305,call_321,const_322,var_323,uop_332,uop_343,])
output2 = relay.Tuple([bop_305,call_324,const_322,var_323,uop_332,uop_343,])
func_345 = relay.Function([var_303,var_304,var_323,], output)
mod['func_345'] = func_345
mod = relay.transform.InferType()(mod)
mutated_mod['func_345'] = func_345
mutated_mod = relay.transform.InferType()(mutated_mod)
func_345_call = mutated_mod.get_global_var('func_345')
var_347 = relay.var("var_347", dtype = "uint32", shape = (14, 9, 11))#candidate|347|(14, 9, 11)|var|uint32
var_348 = relay.var("var_348", dtype = "uint32", shape = (14, 9, 11))#candidate|348|(14, 9, 11)|var|uint32
var_349 = relay.var("var_349", dtype = "float32", shape = (192,))#candidate|349|(192,)|var|float32
call_346 = func_345_call(var_347,var_348,var_349,)
output = call_346
func_350 = relay.Function([var_347,var_348,var_349,], output)
mutated_mod['func_350'] = func_350
mutated_mod = relay.transform.InferType()(mutated_mod)
var_352 = relay.var("var_352", dtype = "uint64", shape = (7, 5))#candidate|352|(7, 5)|var|uint64
const_353 = relay.const([[-4,8,8,6,-7],[-5,1,-10,7,-3],[3,-1,-1,8,2],[-9,-8,-10,2,-10],[-9,1,6,-10,5],[-1,-7,1,-9,8],[-8,8,8,2,5]], dtype = "uint64")#candidate|353|(7, 5)|const|uint64
bop_354 = relay.subtract(var_352.astype('uint64'), relay.reshape(const_353.astype('uint64'), relay.shape_of(var_352))) # shape=(7, 5)
output = relay.Tuple([bop_354,])
output2 = relay.Tuple([bop_354,])
func_366 = relay.Function([var_352,], output)
mod['func_366'] = func_366
mod = relay.transform.InferType()(mod)
var_367 = relay.var("var_367", dtype = "uint64", shape = (7, 5))#candidate|367|(7, 5)|var|uint64
output = func_366(var_367)
func_368 = relay.Function([var_367], output)
mutated_mod['func_368'] = func_368
mutated_mod = relay.transform.InferType()(mutated_mod)
var_382 = relay.var("var_382", dtype = "uint16", shape = (8, 11, 16))#candidate|382|(8, 11, 16)|var|uint16
var_383 = relay.var("var_383", dtype = "uint16", shape = (8, 11, 16))#candidate|383|(8, 11, 16)|var|uint16
bop_384 = relay.maximum(var_382.astype('uint16'), relay.reshape(var_383.astype('uint16'), relay.shape_of(var_382))) # shape=(8, 11, 16)
bop_393 = relay.multiply(bop_384.astype('int64'), relay.reshape(var_382.astype('int64'), relay.shape_of(bop_384))) # shape=(8, 11, 16)
var_398 = relay.var("var_398", dtype = "int64", shape = (8, 11, 16))#candidate|398|(8, 11, 16)|var|int64
bop_399 = relay.bitwise_and(bop_393.astype('uint16'), relay.reshape(var_398.astype('uint16'), relay.shape_of(bop_393))) # shape=(8, 11, 16)
uop_404 = relay.sinh(bop_393.astype('float64')) # shape=(8, 11, 16)
output = relay.Tuple([bop_399,uop_404,])
output2 = relay.Tuple([bop_399,uop_404,])
func_406 = relay.Function([var_382,var_383,var_398,], output)
mod['func_406'] = func_406
mod = relay.transform.InferType()(mod)
mutated_mod['func_406'] = func_406
mutated_mod = relay.transform.InferType()(mutated_mod)
func_406_call = mutated_mod.get_global_var('func_406')
var_408 = relay.var("var_408", dtype = "uint16", shape = (8, 11, 16))#candidate|408|(8, 11, 16)|var|uint16
var_409 = relay.var("var_409", dtype = "uint16", shape = (8, 11, 16))#candidate|409|(8, 11, 16)|var|uint16
var_410 = relay.var("var_410", dtype = "int64", shape = (8, 11, 16))#candidate|410|(8, 11, 16)|var|int64
call_407 = func_406_call(var_408,var_409,var_410,)
output = call_407
func_411 = relay.Function([var_408,var_409,var_410,], output)
mutated_mod['func_411'] = func_411
mutated_mod = relay.transform.InferType()(mutated_mod)
const_452 = relay.const([[[-8,8,-5,3,4,9],[-6,10,-1,6,3,-7],[-4,-3,-9,-8,4,-7],[9,-2,10,7,-4,-2],[7,10,-1,6,5,-10],[4,3,6,-9,-10,5],[2,7,-8,-10,8,4],[-8,5,-10,10,-4,4]],[[-3,-2,5,2,-1,3],[8,6,-7,5,1,-1],[-6,-6,3,10,-2,8],[2,-7,1,-4,-5,-6],[3,8,-3,6,1,10],[2,2,2,3,2,-7],[-2,-5,-3,9,-4,9],[-7,7,-7,-2,6,-4]],[[3,-10,-7,-9,10,-6],[-5,-9,4,-6,-3,9],[-1,8,6,-6,9,-6],[-5,-7,-9,7,10,4],[-4,-4,-5,-6,5,-6],[5,-2,7,5,10,-8],[7,-4,-3,-10,9,3],[-1,2,2,9,-7,-6]],[[-6,8,-2,-7,-4,-5],[-9,4,-3,4,8,9],[4,-9,-8,7,5,4],[-4,6,3,-4,8,-2],[7,-4,4,4,3,8],[10,7,-3,-5,-1,-4],[4,-9,-8,-3,9,-9],[-1,10,8,-5,-5,6]]], dtype = "uint8")#candidate|452|(4, 8, 6)|const|uint8
var_453 = relay.var("var_453", dtype = "uint8", shape = (4, 8, 6))#candidate|453|(4, 8, 6)|var|uint8
bop_454 = relay.less_equal(const_452.astype('bool'), relay.reshape(var_453.astype('bool'), relay.shape_of(const_452))) # shape=(4, 8, 6)
bop_466 = relay.less(const_452.astype('bool'), relay.reshape(bop_454.astype('bool'), relay.shape_of(const_452))) # shape=(4, 8, 6)
bop_469 = relay.logical_or(bop_466.astype('bool'), relay.reshape(var_453.astype('bool'), relay.shape_of(bop_466))) # shape=(4, 8, 6)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_472 = func_59_call(relay.reshape(var_453.astype('float32'), [12, 16]))
call_473 = func_59_call(relay.reshape(var_453.astype('float32'), [12, 16]))
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
call_474 = func_59_call(relay.reshape(bop_469.astype('float32'), [12, 16]))
call_475 = func_59_call(relay.reshape(bop_469.astype('float32'), [12, 16]))
bop_477 = relay.logical_and(call_474.astype('bool'), relay.reshape(bop_469.astype('bool'), relay.shape_of(call_474))) # shape=(12, 16)
bop_480 = relay.logical_and(call_475.astype('bool'), relay.reshape(bop_469.astype('bool'), relay.shape_of(call_475))) # shape=(12, 16)
func_270_call = mod.get_global_var('func_270')
func_275_call = mutated_mod.get_global_var('func_275')
var_483 = relay.var("var_483", dtype = "float32", shape = (12, 2))#candidate|483|(12, 2)|var|float32
call_482 = relay.TupleGetItem(func_270_call(relay.reshape(var_483.astype('float32'), [8, 3]), relay.reshape(bop_466.astype('float32'), [192,]), relay.reshape(var_483.astype('float32'), [8, 3]), relay.reshape(var_483.astype('int32'), [8, 3]), ), 1)
call_484 = relay.TupleGetItem(func_275_call(relay.reshape(var_483.astype('float32'), [8, 3]), relay.reshape(bop_466.astype('float32'), [192,]), relay.reshape(var_483.astype('float32'), [8, 3]), relay.reshape(var_483.astype('int32'), [8, 3]), ), 1)
func_345_call = mod.get_global_var('func_345')
func_350_call = mutated_mod.get_global_var('func_350')
var_486 = relay.var("var_486", dtype = "uint32", shape = (1, 1386))#candidate|486|(1, 1386)|var|uint32
call_485 = relay.TupleGetItem(func_345_call(relay.reshape(var_486.astype('uint32'), [14, 9, 11]), relay.reshape(var_486.astype('uint32'), [14, 9, 11]), relay.reshape(var_453.astype('float32'), [192,]), ), 4)
call_487 = relay.TupleGetItem(func_350_call(relay.reshape(var_486.astype('uint32'), [14, 9, 11]), relay.reshape(var_486.astype('uint32'), [14, 9, 11]), relay.reshape(var_453.astype('float32'), [192,]), ), 4)
bop_488 = relay.floor_divide(bop_469.astype('float32'), relay.reshape(const_452.astype('float32'), relay.shape_of(bop_469))) # shape=(4, 8, 6)
output = relay.Tuple([call_472,bop_477,call_482,var_483,call_485,var_486,bop_488,])
output2 = relay.Tuple([call_473,bop_480,call_484,var_483,call_487,var_486,bop_488,])
func_492 = relay.Function([var_453,var_483,var_486,], output)
mod['func_492'] = func_492
mod = relay.transform.InferType()(mod)
mutated_mod['func_492'] = func_492
mutated_mod = relay.transform.InferType()(mutated_mod)
func_492_call = mutated_mod.get_global_var('func_492')
var_494 = relay.var("var_494", dtype = "uint8", shape = (4, 8, 6))#candidate|494|(4, 8, 6)|var|uint8
var_495 = relay.var("var_495", dtype = "float32", shape = (12, 2))#candidate|495|(12, 2)|var|float32
var_496 = relay.var("var_496", dtype = "uint32", shape = (1, 1386))#candidate|496|(1, 1386)|var|uint32
call_493 = func_492_call(var_494,var_495,var_496,)
output = call_493
func_497 = relay.Function([var_494,var_495,var_496,], output)
mutated_mod['func_497'] = func_497
mutated_mod = relay.transform.InferType()(mutated_mod)
var_513 = relay.var("var_513", dtype = "float64", shape = (8, 13))#candidate|513|(8, 13)|var|float64
uop_514 = relay.sinh(var_513.astype('float64')) # shape=(8, 13)
uop_516 = relay.rsqrt(uop_514.astype('float32')) # shape=(8, 13)
var_518 = relay.var("var_518", dtype = "float64", shape = (8, 13))#candidate|518|(8, 13)|var|float64
bop_519 = relay.mod(uop_514.astype('float32'), relay.reshape(var_518.astype('float32'), relay.shape_of(uop_514))) # shape=(8, 13)
var_524 = relay.var("var_524", dtype = "float32", shape = (8, 13))#candidate|524|(8, 13)|var|float32
bop_525 = relay.logical_and(uop_516.astype('bool'), relay.reshape(var_524.astype('bool'), relay.shape_of(uop_516))) # shape=(8, 13)
uop_529 = relay.asinh(bop_525.astype('float64')) # shape=(8, 13)
uop_536 = relay.atanh(uop_529.astype('float64')) # shape=(8, 13)
uop_538 = relay.cos(uop_536.astype('float32')) # shape=(8, 13)
bop_542 = relay.greater_equal(uop_538.astype('bool'), relay.reshape(uop_536.astype('bool'), relay.shape_of(uop_538))) # shape=(8, 13)
bop_545 = relay.bitwise_and(uop_536.astype('uint16'), relay.reshape(uop_529.astype('uint16'), relay.shape_of(uop_536))) # shape=(8, 13)
uop_548 = relay.erf(uop_538.astype('float32')) # shape=(8, 13)
bop_555 = relay.power(uop_548.astype('float32'), relay.reshape(var_513.astype('float32'), relay.shape_of(uop_548))) # shape=(8, 13)
uop_559 = relay.log(bop_555.astype('float64')) # shape=(8, 13)
const_562 = relay.const([[9.434655,9.319393,7.928519,-7.898306,-8.854318,-9.035266,-8.582690,8.703551,3.829405,-2.217545,2.129761,3.752798,2.069533],[-4.424305,6.683757,-4.047875,5.385158,1.498750,0.051502,-5.511624,8.836668,2.403450,-4.951524,4.537175,2.749712,-7.783470],[-5.475612,-2.295122,-9.924906,-2.238578,0.994383,5.249395,-5.214246,3.776427,-1.028533,-5.974652,-1.508419,9.978496,-0.461363],[1.686134,4.775773,-1.387931,6.767989,3.987734,4.891374,-6.267519,-5.089324,6.419613,-0.751647,3.610056,-4.786611,-4.646328],[3.043241,-7.836300,-4.620153,3.225398,2.031240,4.663962,-2.110851,-7.398074,-6.505464,-1.269916,-7.479297,-6.830962,-0.502719],[6.949334,1.415983,-7.910624,-3.519086,-3.333227,-1.145528,-6.122750,-3.188997,-4.059303,-6.044022,-3.051196,3.898371,4.332517],[2.956702,-9.185528,8.792349,-0.257859,-6.067470,5.952865,8.893360,4.492652,3.951918,5.649844,-7.754909,8.835202,0.132473],[8.414988,-3.781255,-1.586084,-2.105192,-0.613558,3.016129,-4.161537,4.964482,-9.813948,-2.677453,-0.142163,4.565173,4.859278]], dtype = "float64")#candidate|562|(8, 13)|const|float64
bop_563 = relay.not_equal(uop_559.astype('bool'), relay.reshape(const_562.astype('bool'), relay.shape_of(uop_559))) # shape=(8, 13)
output = relay.Tuple([bop_519,bop_542,bop_545,bop_563,])
output2 = relay.Tuple([bop_519,bop_542,bop_545,bop_563,])
func_570 = relay.Function([var_513,var_518,var_524,], output)
mod['func_570'] = func_570
mod = relay.transform.InferType()(mod)
mutated_mod['func_570'] = func_570
mutated_mod = relay.transform.InferType()(mutated_mod)
func_570_call = mutated_mod.get_global_var('func_570')
var_572 = relay.var("var_572", dtype = "float64", shape = (8, 13))#candidate|572|(8, 13)|var|float64
var_573 = relay.var("var_573", dtype = "float64", shape = (8, 13))#candidate|573|(8, 13)|var|float64
var_574 = relay.var("var_574", dtype = "float32", shape = (8, 13))#candidate|574|(8, 13)|var|float32
call_571 = func_570_call(var_572,var_573,var_574,)
output = call_571
func_575 = relay.Function([var_572,var_573,var_574,], output)
mutated_mod['func_575'] = func_575
mutated_mod = relay.transform.InferType()(mutated_mod)
var_605 = relay.var("var_605", dtype = "int8", shape = ())#candidate|605|()|var|int8
var_606 = relay.var("var_606", dtype = "int8", shape = (4, 13))#candidate|606|(4, 13)|var|int8
bop_607 = relay.not_equal(var_605.astype('bool'), var_606.astype('bool')) # shape=(4, 13)
output = relay.Tuple([bop_607,])
output2 = relay.Tuple([bop_607,])
func_615 = relay.Function([var_605,var_606,], output)
mod['func_615'] = func_615
mod = relay.transform.InferType()(mod)
var_616 = relay.var("var_616", dtype = "int8", shape = ())#candidate|616|()|var|int8
var_617 = relay.var("var_617", dtype = "int8", shape = (4, 13))#candidate|617|(4, 13)|var|int8
output = func_615(var_616,var_617,)
func_618 = relay.Function([var_616,var_617,], output)
mutated_mod['func_618'] = func_618
mutated_mod = relay.transform.InferType()(mutated_mod)
var_631 = relay.var("var_631", dtype = "float32", shape = (12, 11))#candidate|631|(12, 11)|var|float32
var_632 = relay.var("var_632", dtype = "float32", shape = (12, 11))#candidate|632|(12, 11)|var|float32
bop_633 = relay.floor_mod(var_631.astype('float32'), relay.reshape(var_632.astype('float32'), relay.shape_of(var_631))) # shape=(12, 11)
uop_638 = relay.sqrt(var_631.astype('float64')) # shape=(12, 11)
var_645 = relay.var("var_645", dtype = "float64", shape = (12, 11))#candidate|645|(12, 11)|var|float64
bop_646 = relay.logical_or(uop_638.astype('bool'), relay.reshape(var_645.astype('bool'), relay.shape_of(uop_638))) # shape=(12, 11)
uop_649 = relay.acosh(uop_638.astype('float64')) # shape=(12, 11)
func_102_call = mod.get_global_var('func_102')
func_105_call = mutated_mod.get_global_var('func_105')
const_653 = relay.const([-6.500772,-2.624482,-7.788753,3.243243], dtype = "float32")#candidate|653|(4,)|const|float32
const_654 = relay.const([[6.966345],[-6.554709],[0.737255],[-9.059317],[4.092924],[9.952604],[-9.122991],[5.502344],[9.247444],[3.773217],[3.467929],[-5.263629],[6.844634],[4.701204],[-6.251145],[-8.064344],[-7.283983],[4.243096],[-8.724986],[5.490692],[-4.860590],[3.091529],[4.254752],[-2.650674],[-7.189726],[8.051248],[-3.715749],[-9.622681],[6.741853],[6.498909],[7.805884],[-4.893959],[-8.011984],[6.444443],[5.092280],[-2.233146]], dtype = "float32")#candidate|654|(36, 1)|const|float32
call_652 = relay.TupleGetItem(func_102_call(relay.reshape(const_653.astype('float32'), [1, 4]), relay.reshape(const_654.astype('float32'), [9, 4]), ), 2)
call_655 = relay.TupleGetItem(func_105_call(relay.reshape(const_653.astype('float32'), [1, 4]), relay.reshape(const_654.astype('float32'), [9, 4]), ), 2)
uop_659 = relay.cosh(uop_649.astype('float32')) # shape=(12, 11)
output = relay.Tuple([bop_633,bop_646,call_652,const_653,const_654,uop_659,])
output2 = relay.Tuple([bop_633,bop_646,call_655,const_653,const_654,uop_659,])
func_661 = relay.Function([var_631,var_632,var_645,], output)
mod['func_661'] = func_661
mod = relay.transform.InferType()(mod)
mutated_mod['func_661'] = func_661
mutated_mod = relay.transform.InferType()(mutated_mod)
func_661_call = mutated_mod.get_global_var('func_661')
var_663 = relay.var("var_663", dtype = "float32", shape = (12, 11))#candidate|663|(12, 11)|var|float32
var_664 = relay.var("var_664", dtype = "float32", shape = (12, 11))#candidate|664|(12, 11)|var|float32
var_665 = relay.var("var_665", dtype = "float64", shape = (12, 11))#candidate|665|(12, 11)|var|float64
call_662 = func_661_call(var_663,var_664,var_665,)
output = call_662
func_666 = relay.Function([var_663,var_664,var_665,], output)
mutated_mod['func_666'] = func_666
mutated_mod = relay.transform.InferType()(mutated_mod)
var_776 = relay.var("var_776", dtype = "int16", shape = (1, 11, 2))#candidate|776|(1, 11, 2)|var|int16
var_777 = relay.var("var_777", dtype = "int16", shape = (6, 11, 2))#candidate|777|(6, 11, 2)|var|int16
bop_778 = relay.bitwise_or(var_776.astype('int16'), var_777.astype('int16')) # shape=(6, 11, 2)
bop_788 = relay.bitwise_and(var_776.astype('uint8'), bop_778.astype('uint8')) # shape=(6, 11, 2)
var_798 = relay.var("var_798", dtype = "int16", shape = (6, 11, 2))#candidate|798|(6, 11, 2)|var|int16
bop_799 = relay.logical_and(bop_778.astype('bool'), relay.reshape(var_798.astype('bool'), relay.shape_of(bop_778))) # shape=(6, 11, 2)
func_270_call = mod.get_global_var('func_270')
func_275_call = mutated_mod.get_global_var('func_275')
const_803 = relay.const([[-8.219820,0.667770,0.427587,-5.629541],[-1.575606,4.717708,-5.472669,-5.037325],[0.702621,-6.156836,-4.805896,7.749193],[-1.623729,-2.427942,1.945560,-2.648437],[-6.943673,-9.465669,-3.420081,-5.692598],[9.383229,-0.453775,-7.101432,1.944161]], dtype = "float32")#candidate|803|(6, 4)|const|float32
var_804 = relay.var("var_804", dtype = "float32", shape = (4, 48))#candidate|804|(4, 48)|var|float32
call_802 = relay.TupleGetItem(func_270_call(relay.reshape(const_803.astype('float32'), [8, 3]), relay.reshape(var_804.astype('float32'), [192,]), relay.reshape(const_803.astype('float32'), [8, 3]), relay.reshape(const_803.astype('int32'), [8, 3]), ), 9)
call_805 = relay.TupleGetItem(func_275_call(relay.reshape(const_803.astype('float32'), [8, 3]), relay.reshape(var_804.astype('float32'), [192,]), relay.reshape(const_803.astype('float32'), [8, 3]), relay.reshape(const_803.astype('int32'), [8, 3]), ), 9)
uop_807 = relay.sinh(bop_799.astype('float64')) # shape=(6, 11, 2)
uop_813 = relay.acosh(uop_807.astype('float32')) # shape=(6, 11, 2)
uop_815 = relay.asin(uop_813.astype('float64')) # shape=(6, 11, 2)
uop_817 = relay.sin(uop_815.astype('float64')) # shape=(6, 11, 2)
uop_819 = relay.cos(uop_817.astype('float32')) # shape=(6, 11, 2)
bop_821 = relay.power(uop_815.astype('float64'), relay.reshape(uop_819.astype('float64'), relay.shape_of(uop_815))) # shape=(6, 11, 2)
const_825 = relay.const([[[0.856100,3.189434],[2.094209,-1.782661],[-7.614693,-3.229744],[-2.931726,0.912528],[9.148490,2.376591],[-0.632500,-6.664690],[9.981393,9.670190],[-3.331267,-4.941461],[-7.687101,-8.570481],[-5.434977,7.252164],[2.770847,-4.695676]],[[3.481884,0.817145],[-1.911385,7.446251],[2.993717,-2.486741],[3.969143,4.051994],[-7.700735,6.997334],[-3.640172,-3.921187],[-7.345603,7.114310],[2.323755,9.996091],[-0.876259,-8.353556],[5.996082,2.534557],[6.498725,4.146428]],[[6.362657,-9.989890],[8.834757,-4.326418],[0.220237,-8.281637],[-2.944079,1.921688],[0.730303,-5.975751],[7.171860,-7.597128],[7.533450,3.307843],[4.539683,7.915063],[-9.052596,-5.922920],[2.131223,2.541877],[6.367237,-2.230900]],[[3.482038,-6.648052],[7.207039,-5.654751],[6.813269,7.462404],[-3.643264,2.181472],[-0.577084,-4.169753],[7.227136,-8.577573],[9.743636,8.928621],[1.891266,0.766634],[-7.637339,-6.111732],[-2.940522,-3.236501],[0.452533,-3.354858]],[[8.950844,-8.417604],[-5.411616,-4.395786],[0.433306,-9.044539],[-7.375839,-5.649344],[-2.190096,-1.652665],[7.428428,-4.840012],[-3.902477,9.024752],[-1.603036,1.991640],[8.193637,2.621707],[5.098205,3.044620],[-4.949398,6.082117]],[[-5.739830,0.937177],[8.864567,5.466694],[-3.130349,1.483015],[-7.977750,-7.547451],[-0.112430,-1.291163],[4.237764,-5.580318],[4.333456,-8.277264],[6.696769,-7.957307],[7.599025,-0.245200],[-3.848474,3.362304],[2.056627,6.761238]]], dtype = "float64")#candidate|825|(6, 11, 2)|const|float64
bop_826 = relay.mod(bop_821.astype('float64'), relay.reshape(const_825.astype('float64'), relay.shape_of(bop_821))) # shape=(6, 11, 2)
uop_840 = relay.rsqrt(uop_815.astype('float64')) # shape=(6, 11, 2)
func_492_call = mod.get_global_var('func_492')
func_497_call = mutated_mod.get_global_var('func_497')
const_843 = relay.const([-6,-9,7,2,-4,-8,-9,3,1,2,-7,1,-6,-1,-8,-2,9,1,-3,-1,4,-10,-6,-2,-2,5,3,1,8,6,8,-10,1,-3,4,3,-5,-10,-4,10,-9,7,4,9,-10,5,3,7,9,-10,5,-1,-4,3,-3,6,-9,-6,-3,-4,2,-2,-7,10,-2,8,-4,10,-2,-8,8,-9,-8,-2,9,-5,7,7,8,-9,8,10,8,9,-4,7,4,-3,-10,-3,-2,2,-6,3,-4,6,4,3,-7,4,-1,7,4,8,-4,-4,1,-6,-5,-3,-3,-3,5,1,-8,5,-3,-7,-2,-2,5,-7,-2,-3,3,-5,5,5,4,3,-8,-3,3,-7,-4,5,5,-6,-4,1,5,4,-4,1,3,-9,-4,-2,-4,-4,-8,-8,9,-9,4,-8,7,-10,-1,8,5,-6,2,8,7,-4,8,-1,9,2,1,4,-4,-7,-7,10,4,3,-4,-2,2,9,2,-9,-7,6,-2,5,9,10,9,-10,-8,9,-5,-4,-8,6,-10,-3,-7,6,-1,5,3,6,7,-4,-10,-6,3,1,-10,1,4,-8,3,10,3,6,1,9,-7,-3,-6,-3,9,-8,10,1,-5,7,-1,4,10,-1,1,9,2,6,-10,-5,-2,10,-4,8,-7,7,-1,3,8,-2,-3,6,4,-1,10,5,-9,-4,5,-2,8,-2,8,-10,-9,-5,5,8,-5,4,-5,7,-4,-8,-6,1,10,3,-9,6,7,-2,7,6,9,7,-8,9,-7,1,9,-7,7,9,-3,6,10,-6,8,-7,1,10,-10,3,5,-3,8,-5,8,-3,-9,-4,-8,6,-2,4,8,-5,2,8,-4,4,-8,8,-3,8,-1,2,3,-7,10,-10,-5,9,-5,-7,-5,-2,9,1,-1,3,3,5,-3,5,-6,-2,-2,-8,-7,5,-5,-6,-6,-6,8,-1,7,9,8,3,-8,5,-7,6,-10,-5,7,1,10,2,-9,1,1,-7,-3,-5,7,-3,-4,10,2,-9,9,-2,-9,7,9,5,-7,1,3,2,-5,-6,-6,3,2,-6,-3,-7,5,2,1,8,-10,-9,4,-4,-8,7,5,-1,1,-7,8,-9,-10,-10,10,1,6,4,-3,8,9,7,-3,-2,-10,9,4,2,10,-7,10,10,-9,-6,3,6,-9,-7,-3,-9,7,-10,7,-5,4,7,1,1,-2,-2,5,8,-1,8,-5,5,-6,-2,-7,-2,-10,-6,-6,9,-8,-4,9,6,-8,-1,3,4,-3,4,5,2,7,-4,-6,1,-4,-8,-9,-10,-1,10,9,5,2,1,-7,1,-10,7,8,4,4,-6,6,-2,8,-3,7,10,-7,-7,5,-10,-4,5,-6,-5,-10,-7,-6,-5,2,3,6,-1,8,1,7,-10,-10,-3,3,1,9,-2,-7,-1,-5,-8,-10,5,-6,-1,8,6,-8,4,-8,-7,-3,4,-5,6,-9,-2,3,7,4,4,1,9,8,-5,10,-5,-2,-10,-4,-9,-3,2,-8,7,8,1,-3,3,6,10,-9,-7,-4,7,5,10,7,-6,-2,-5,6,2,-8,3,-2,-10,-7,-1,7,-8,-4,-2,2,-5,1,-5,-8,8,-3,-2,5,-5,10,-5,3,-9,4,7,-9,10,-2,9,-3,6,10,-10,-1,-7,-10,7,-6,2,6,-5,2,5,10,4,2,9,1,9,2,-10,-10,8,3,-5,-7,-4,-8,5,-4,3,-2,-2,-2,2,8,-8,10,4,4,-2,2,4,-6,9,-9,-10,7,-3,-2,2,-10,8,-6,1,2,5,-9,5,9,7,-8,-10,-8,3,9,-4,8,6,4,6,-3,1,1,9,-4,7,-5,9,9,7,-7,-6,3,-1,-1,5,9,10,-4,9,7,-9,2,-10,-10,-8,10,-4,-7,-1,3,4,2,1,3,5,5,-6,2,9,-5,-10,2,-7,10,-7,4,4,-7,5,4,-2,3,-7,6,-8,-2,5,6,-3,1,-1,3,-2,-10,-2,6,5,-4,4,-10,-10,-3,-6,-10,-7,-1,8,4,4,-7,-10,2,10,-10,5,-9,2,-3,-4,9,4,10,4,1,7,6,-9,-3,-4,7,2,-8,9,3,6,5,-10,1,-8,-1,10,-2,-8,2,-7,5,5,2,-7,-7,6,3,6,1,-8,-9,-2,1,-4,-3,-2,-8,10,-10,6,-9,1,10,8,1,6,8,1,-9,4,5,8,2,-5,6,4,-4,7,-4,5,-3,-7,10,-2,-10,9,-2,-1,5,-9,-7,-6,5,5,2,-7,3,-8,10,10,-6,-7,5,3,4,-1,-2,-2,-2,-8,-10,-1,-7,-6,9,-6,6,-1,1,-3,5,-6,2,-6,6,1,9,9,-9,2,4,2,3,-10,-7,-1,-6,-9,-4,5,9,8,-5,3,-5,-3,6,8,9,6,-3,-8,4,7,10,-10,7,-6,-6,-10,5,9,-9,3,-6,-3,5,-8,-9,-7,-9,8,6,10,-5,4,-6,-3,1,-2,-6,6,8,-10,-8,-6,5,-5,-7,4,5,-4,-2,-5,-5,-2,6,10,-1,-1,-9,-8,-2,-5,7,-1,-7,5,-3,2,9,-7,-1,1,9,-1,-10,1,9,10,-5,-10,7,-2,6,-8,-8,-2,-4,-1,6,1,10,-9,-10,-4,-2,-5,-10,1,3,-10,4,1,4,1,3,-9,4,3,7,-8,4,9,-2,-7,-2,5,-9,-4,-10,7,6,8,4,-5,1,-9,9,-6,-2,4,-1,10,7,7,2,3,-9,-10,-2,-1,8,-7,-2,8,5,-8,4,8,-5,-1,6,4,4,-8,1,-1,-1,-8,1,-9,-9,-5,-4,-4,-6,5,-6,-9,5,9,7,6,8,3,9,6,-8,9,-1,-2,-6,1,9,6,-7,9,7,-6,-9,-4,10,6,-10,1,-8,2,-3,9,5,4,-10,-8,10,2,2,2,8,1,3,-3,6,-8,-8,-10,10,-5,-9,-7,-6,2,7,-9,2,-4,3,-7,-1,-9,-1,2,-3,4,-9,-2,-10,-2,-8,3,-4,-4,-2,-7,5,-2,4,-10,10,-10,-4,5,-9,-2,-1,-7,-10,1,2,-5,-3,-7,-10,6,10,-2,-10,-10,-7,8,-7,-10,-5,-1,8,-6,-4,7,-10,5,-1,-4,-5,2,9,9,-4,-5,9,-8,3,-7,8,10,-1,8,3,4,8,-1,7,-3,8,-2,5,5,5,-4,9,1,-8,-3,10,-10,-5,-5,10,5,-5,4,-10,-9,-2,-6,-6,3,5,-6,3,-5,-8,3,8,3,-9,-7,-9,-2,7,4,-6,-9,2,7,-3,6,-3,-9,-8,-8,1,-4,10,-1,-7,-4,3,5,1,3,-4,-7,-7,-2,9,4,3,-10,5,4,8,4,-1,9,10,-8,-10,-3,-1,3,-5,10,4,-3,-3,10,2,5,4,-7,-5,-1,8,-1,9,9,-3,-1,8,8,8,-5,6,8,-1,4,-8,10,-9,6,-8,-1,3,10,6,3,7,-8,5,3,-9,-3,-4,-4,5,-10,-3,4,4,7,1,4,9,3,8,-1,4,3,-8,-10,4,-4,2,-7,1,6,6,-3,4,1,3,8,-8,5,7,-9,-7,2,-2,-6,2,6,-4,5,5,9,-7,-7,-7,3,10,-6,6,6], dtype = "uint32")#candidate|843|(1386,)|const|uint32
call_842 = relay.TupleGetItem(func_492_call(relay.reshape(call_802.astype('uint8'), [4, 8, 6]), relay.reshape(const_803.astype('float32'), [12, 2]), relay.reshape(const_843.astype('uint32'), [1, 1386]), ), 4)
call_844 = relay.TupleGetItem(func_497_call(relay.reshape(call_802.astype('uint8'), [4, 8, 6]), relay.reshape(const_803.astype('float32'), [12, 2]), relay.reshape(const_843.astype('uint32'), [1, 1386]), ), 4)
output = relay.Tuple([bop_788,call_802,const_803,var_804,bop_826,uop_840,call_842,const_843,])
output2 = relay.Tuple([bop_788,call_805,const_803,var_804,bop_826,uop_840,call_844,const_843,])
func_845 = relay.Function([var_776,var_777,var_798,var_804,], output)
mod['func_845'] = func_845
mod = relay.transform.InferType()(mod)
mutated_mod['func_845'] = func_845
mutated_mod = relay.transform.InferType()(mutated_mod)
func_845_call = mutated_mod.get_global_var('func_845')
var_847 = relay.var("var_847", dtype = "int16", shape = (1, 11, 2))#candidate|847|(1, 11, 2)|var|int16
var_848 = relay.var("var_848", dtype = "int16", shape = (6, 11, 2))#candidate|848|(6, 11, 2)|var|int16
var_849 = relay.var("var_849", dtype = "int16", shape = (6, 11, 2))#candidate|849|(6, 11, 2)|var|int16
var_850 = relay.var("var_850", dtype = "float32", shape = (4, 48))#candidate|850|(4, 48)|var|float32
call_846 = func_845_call(var_847,var_848,var_849,var_850,)
output = call_846
func_851 = relay.Function([var_847,var_848,var_849,var_850,], output)
mutated_mod['func_851'] = func_851
mutated_mod = relay.transform.InferType()(mutated_mod)
const_905 = relay.const([-0.969462,6.245373,4.402703,-3.601033,-9.067130,2.808256,-7.442955,1.598484,3.033426,4.008847,-6.162088,4.236412,1.781000,1.126079,7.912543], dtype = "float64")#candidate|905|(15,)|const|float64
uop_906 = relay.asinh(const_905.astype('float64')) # shape=(15,)
uop_908 = relay.atan(uop_906.astype('float64')) # shape=(15,)
uop_917 = relay.sigmoid(uop_906.astype('float32')) # shape=(15,)
bop_921 = relay.equal(uop_917.astype('bool'), relay.reshape(const_905.astype('bool'), relay.shape_of(uop_917))) # shape=(15,)
uop_928 = relay.cos(uop_908.astype('float32')) # shape=(15,)
uop_933 = relay.log(uop_928.astype('float64')) # shape=(15,)
func_366_call = mod.get_global_var('func_366')
func_368_call = mutated_mod.get_global_var('func_368')
const_939 = relay.const([2,5,-1,-3,9,6,1,5,-9,9,-2,4,-2,-4,4,9,1,1,7,8,-8,-10,-4,-2,-4,2,3,7,6,10,-7,3,-2,-9,-8], dtype = "uint64")#candidate|939|(35,)|const|uint64
call_938 = relay.TupleGetItem(func_366_call(relay.reshape(const_939.astype('uint64'), [7, 5])), 0)
call_940 = relay.TupleGetItem(func_368_call(relay.reshape(const_939.astype('uint64'), [7, 5])), 0)
func_270_call = mod.get_global_var('func_270')
func_275_call = mutated_mod.get_global_var('func_275')
var_942 = relay.var("var_942", dtype = "float32", shape = (24,))#candidate|942|(24,)|var|float32
var_943 = relay.var("var_943", dtype = "float32", shape = (192,))#candidate|943|(192,)|var|float32
call_941 = relay.TupleGetItem(func_270_call(relay.reshape(var_942.astype('float32'), [8, 3]), relay.reshape(var_943.astype('float32'), [192,]), relay.reshape(var_942.astype('float32'), [8, 3]), relay.reshape(var_942.astype('int32'), [8, 3]), ), 6)
call_944 = relay.TupleGetItem(func_275_call(relay.reshape(var_942.astype('float32'), [8, 3]), relay.reshape(var_943.astype('float32'), [192,]), relay.reshape(var_942.astype('float32'), [8, 3]), relay.reshape(var_942.astype('int32'), [8, 3]), ), 6)
output = relay.Tuple([bop_921,uop_933,call_938,const_939,call_941,var_942,var_943,])
output2 = relay.Tuple([bop_921,uop_933,call_940,const_939,call_944,var_942,var_943,])
func_945 = relay.Function([var_942,var_943,], output)
mod['func_945'] = func_945
mod = relay.transform.InferType()(mod)
mutated_mod['func_945'] = func_945
mutated_mod = relay.transform.InferType()(mutated_mod)
func_945_call = mutated_mod.get_global_var('func_945')
var_947 = relay.var("var_947", dtype = "float32", shape = (24,))#candidate|947|(24,)|var|float32
var_948 = relay.var("var_948", dtype = "float32", shape = (192,))#candidate|948|(192,)|var|float32
call_946 = func_945_call(var_947,var_948,)
output = call_946
func_949 = relay.Function([var_947,var_948,], output)
mutated_mod['func_949'] = func_949
mutated_mod = relay.transform.InferType()(mutated_mod)
var_957 = relay.var("var_957", dtype = "float32", shape = (8, 3))#candidate|957|(8, 3)|var|float32
uop_958 = relay.log2(var_957.astype('float32')) # shape=(8, 3)
func_59_call = mod.get_global_var('func_59')
func_62_call = mutated_mod.get_global_var('func_62')
var_962 = relay.var("var_962", dtype = "float32", shape = (192,))#candidate|962|(192,)|var|float32
call_961 = func_59_call(relay.reshape(var_962.astype('float32'), [12, 16]))
call_963 = func_59_call(relay.reshape(var_962.astype('float32'), [12, 16]))
bop_968 = relay.bitwise_or(call_961.astype('int64'), relay.reshape(var_962.astype('int64'), relay.shape_of(call_961))) # shape=(12, 16)
bop_971 = relay.bitwise_or(call_963.astype('int64'), relay.reshape(var_962.astype('int64'), relay.shape_of(call_963))) # shape=(12, 16)
output = relay.Tuple([uop_958,bop_968,])
output2 = relay.Tuple([uop_958,bop_971,])
func_980 = relay.Function([var_957,var_962,], output)
mod['func_980'] = func_980
mod = relay.transform.InferType()(mod)
mutated_mod['func_980'] = func_980
mutated_mod = relay.transform.InferType()(mutated_mod)
func_980_call = mutated_mod.get_global_var('func_980')
var_982 = relay.var("var_982", dtype = "float32", shape = (8, 3))#candidate|982|(8, 3)|var|float32
var_983 = relay.var("var_983", dtype = "float32", shape = (192,))#candidate|983|(192,)|var|float32
call_981 = func_980_call(var_982,var_983,)
output = call_981
func_984 = relay.Function([var_982,var_983,], output)
mutated_mod['func_984'] = func_984
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1004 = relay.var("var_1004", dtype = "uint64", shape = (4, 7, 15))#candidate|1004|(4, 7, 15)|var|uint64
var_1005 = relay.var("var_1005", dtype = "uint64", shape = (4, 7, 15))#candidate|1005|(4, 7, 15)|var|uint64
bop_1006 = relay.logical_xor(var_1004.astype('uint64'), relay.reshape(var_1005.astype('uint64'), relay.shape_of(var_1004))) # shape=(4, 7, 15)
output = bop_1006
output2 = bop_1006
func_1011 = relay.Function([var_1004,var_1005,], output)
mod['func_1011'] = func_1011
mod = relay.transform.InferType()(mod)
var_1012 = relay.var("var_1012", dtype = "uint64", shape = (4, 7, 15))#candidate|1012|(4, 7, 15)|var|uint64
var_1013 = relay.var("var_1013", dtype = "uint64", shape = (4, 7, 15))#candidate|1013|(4, 7, 15)|var|uint64
output = func_1011(var_1012,var_1013,)
func_1014 = relay.Function([var_1012,var_1013,], output)
mutated_mod['func_1014'] = func_1014
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1058 = relay.var("var_1058", dtype = "float32", shape = (10, 10, 15))#candidate|1058|(10, 10, 15)|var|float32
const_1059 = relay.const([[[-0.741768,7.950694,8.368710,5.208448,3.875792,-1.413846,8.015150,-3.800967,2.198380,7.988621,5.186125,-5.457402,-7.347869,3.281015,3.079381],[-6.727147,2.981967,7.530419,3.831082,7.232738,5.844454,5.684844,3.535205,4.451251,4.600483,-9.656176,-3.305015,-4.689783,-7.189330,-2.991084],[-5.149492,6.912166,-5.826457,-3.441279,9.588052,-7.203152,-8.737992,-7.367536,5.339086,-0.685428,-1.847177,-7.428898,5.736538,5.932772,-1.506477],[6.098355,-7.642620,-0.897838,7.431155,5.358483,6.212796,-8.730092,7.497318,8.227843,-8.404343,8.844182,-9.815281,7.020119,6.008552,6.984585],[-5.270832,-9.234422,3.493930,4.275424,-4.803074,-3.028778,-7.001744,0.722890,5.643170,7.083810,-9.805154,-3.714911,-7.455029,-7.860667,-5.756900],[1.719371,-1.749137,3.898039,1.387493,-7.744938,9.591010,5.119989,6.604322,-3.180813,9.725034,-7.416122,-1.516084,-8.819086,1.262761,-8.276008],[-1.209157,-7.238075,-8.359429,-9.088974,7.282257,8.335174,4.027281,-7.751081,0.794550,3.931738,-3.443088,-8.444395,-4.832805,8.955108,5.791716],[6.721432,-1.787370,-5.854779,-7.956014,-9.323960,8.457886,-2.968022,-9.002293,-2.818691,3.760893,-7.567266,4.989885,0.057201,5.268032,4.065843],[-2.941957,-6.203366,1.830629,9.074822,0.822298,7.857820,7.709490,-2.568641,-4.242051,5.826086,-8.822619,-2.997860,-9.227063,-4.363935,-4.850892],[-2.848143,-7.371396,0.791428,2.140567,3.503710,-9.092009,-0.688316,8.597921,7.695794,1.500608,7.542920,-5.099181,8.923389,4.909465,-4.723889]],[[-3.562043,5.732374,4.492716,-5.901713,2.950613,-9.858552,-0.358164,-4.580877,3.435896,8.044136,-6.055723,-7.559765,-7.403608,-9.350748,0.449537],[-2.961739,1.779975,6.350780,-0.767185,-8.753630,2.770824,-7.715413,9.226248,7.923826,-1.398075,8.779614,-6.069681,-9.444676,2.767019,-5.926785],[2.295418,-0.737796,-5.551526,-4.529430,8.544196,0.851364,0.067082,-2.972158,-7.473537,9.014522,9.581688,9.975951,-3.181806,-4.981745,-0.983018],[6.138553,5.802551,6.008867,7.490496,4.397745,-6.567609,5.776148,-5.853041,-7.955070,-9.135738,6.419036,-3.956583,7.780304,-9.790987,-1.438318],[-8.386459,9.540492,-7.135669,-3.763343,-9.003353,-0.843577,-9.250425,-3.579027,-4.086403,4.164416,-6.874383,-7.318076,7.573671,3.842451,0.121235],[8.925727,3.958235,-1.671522,9.927391,2.023247,-4.174132,-9.150319,-2.204805,8.720577,8.929450,4.727529,4.530115,-5.177804,8.825686,7.150050],[8.475099,-7.919206,8.429129,-4.459042,5.816097,-8.632629,-8.365801,-2.663415,1.607880,1.739719,-4.663646,-2.147647,-6.666459,6.144339,-4.447644],[-3.938153,2.728990,9.738945,3.778819,1.942688,1.176729,-2.891879,-0.016970,-1.578315,-0.909842,-6.533141,-4.082508,3.079097,7.639640,-8.885176],[3.549778,2.801896,-2.696673,3.760356,5.540649,-5.245317,-0.867603,-0.881337,-8.113764,-9.215032,-6.356126,2.627868,0.710212,5.752205,5.097514],[-9.327596,-4.929348,7.102479,-5.544566,3.664885,7.323132,3.210858,-2.931367,-8.505567,-6.109471,-5.127769,2.569583,-1.752075,-2.651255,4.547066]],[[3.534099,-5.140819,-6.473576,-4.792781,-0.479035,-6.330537,-2.114611,5.171252,9.430011,3.311093,-7.241611,-4.586766,9.067959,-7.188436,-6.239598],[-4.913874,6.459974,2.154798,-5.403087,4.364164,2.977712,6.442892,3.688339,3.193311,-9.778771,-2.857864,-9.173059,4.754790,5.746273,6.906571],[7.085875,-5.431706,-2.555601,-8.062612,8.169871,0.410446,5.458711,-7.097534,3.143922,5.750907,-8.930179,-4.606549,-1.510321,4.737138,1.985393],[8.525273,-4.985913,7.654329,-6.630765,6.946439,2.124581,-7.657616,7.044079,8.746820,-6.511506,0.206621,7.566128,8.269291,7.255073,7.189635],[6.979740,6.597697,-7.609402,-4.478999,5.630947,1.045802,-5.871757,2.357676,-2.751352,-1.328726,-4.895402,-7.641658,6.495964,9.309922,-6.688212],[4.235517,-9.787494,9.834954,-8.594228,8.305635,1.925058,-7.647295,-0.134138,-8.022807,0.878176,-1.135522,-0.920387,9.311419,-0.564315,-0.602839],[7.379643,3.147012,-2.861412,7.268763,-8.095843,-6.966299,1.870697,-7.925306,8.922850,-7.903594,0.287281,-0.330575,1.835477,-3.914282,-1.712333],[-8.055151,-5.589699,7.464953,-5.131709,6.842883,9.641319,5.660535,8.749269,9.662630,-6.316470,8.201702,-0.458022,2.300308,4.843019,2.188712],[-1.715499,3.659915,6.041726,2.055829,0.891274,9.038163,5.012454,6.079429,-7.412029,2.702616,2.615535,-1.205613,-3.183543,8.264388,9.390325],[3.810628,9.059153,9.314482,-2.142124,-2.436552,4.195048,-1.001113,-4.867500,3.413373,-4.539013,1.994558,4.128895,5.030197,9.848516,1.694599]],[[-5.863439,-3.249035,-7.134701,-6.589607,3.096010,-0.745522,-8.178401,8.876802,8.250557,2.886225,1.857364,6.581679,7.734687,-7.119917,3.579714],[8.149886,-0.750484,1.986321,8.004065,2.776139,2.313055,-8.953183,9.014250,-0.140468,-8.551461,-5.591999,2.360365,2.981560,5.591392,8.584348],[0.251078,-8.528773,2.493495,-5.297243,1.768436,0.962738,-5.750703,3.398100,9.294637,-7.048610,8.294376,0.968472,-9.542667,7.156993,-5.422639],[-3.714572,-2.708200,9.385376,6.302647,-7.764365,-8.752059,4.927357,6.934274,-3.794684,-9.293055,7.493255,5.571375,-4.706821,7.782344,5.014906],[2.807761,7.506755,-1.945299,4.225793,1.092800,9.085004,8.024626,-8.054679,6.175315,9.952108,8.150906,4.192200,2.213595,-4.074045,-7.960002],[-8.402262,2.763412,3.689221,-7.734310,1.904858,-2.045587,1.728673,1.548706,3.214917,-3.314911,8.530412,9.696456,4.650779,-1.704714,9.055013],[7.106655,-3.678963,0.627682,2.678786,-4.324090,8.010778,1.240576,-8.382462,-0.775388,0.950759,-6.951974,-7.368979,2.388592,-5.814608,0.120587],[-5.684378,7.314539,-6.250047,-7.810821,7.826228,-9.233357,-6.582590,7.772314,2.738580,-2.689956,9.749794,3.430601,-6.057533,4.034044,-1.646898],[-9.134902,6.141129,-6.145913,-1.270069,9.399322,5.801707,8.904227,5.453761,7.183155,-7.319183,7.908526,-8.939631,-8.030432,-5.591665,-2.543385],[-4.121611,1.964943,-3.370072,7.740023,2.177009,-2.370497,5.953185,-9.045811,5.391279,6.236885,5.343920,6.881529,5.229210,-1.210530,5.672033]],[[-1.322380,2.707254,0.063694,7.380700,-0.751735,-4.131090,-3.072060,-5.901112,-8.188260,-6.199953,-1.450046,0.003255,7.201921,0.663902,-7.080312],[-1.515571,-8.702452,-3.809616,-5.803163,0.382083,-6.467860,7.343055,8.388067,6.337494,3.449732,1.923144,9.837080,7.432779,0.301211,-5.216562],[9.855317,-1.168056,-3.529307,-8.918671,3.705037,-1.752094,7.747416,-6.871736,6.952813,-5.041936,1.478298,-3.563184,-3.033069,0.868706,-9.103678],[9.358024,8.844720,-3.037448,4.269664,-1.465215,-5.699193,-0.922417,-7.058182,-6.171839,2.142213,1.052970,-0.071614,6.915536,6.536874,-0.969800],[2.588982,-1.745308,2.225888,-7.766954,2.793488,0.731805,2.216560,7.847939,-7.842287,-6.274061,-0.409744,-1.969206,8.527781,-3.747437,6.659399],[0.020115,-2.294481,8.299304,-1.548467,-3.096736,-2.286423,5.748949,-3.374127,0.044126,-2.417655,-5.948442,4.939790,2.484642,-1.883510,-9.067200],[-7.257579,-8.290466,3.065051,-4.054095,-1.321187,5.629796,4.513105,-2.793135,5.115876,-0.080459,2.081248,-6.810096,-5.682643,-3.255301,5.800255],[-7.097267,-2.333483,5.490772,-6.060438,8.300350,-4.483844,8.303159,7.694529,4.504670,1.622187,-9.157316,-7.971591,-9.358759,-5.948831,-0.154539],[6.931479,3.211210,5.445241,0.201928,-0.133579,-9.345265,2.509464,-0.411527,-0.805577,5.594850,-2.360252,-2.549238,-7.878795,4.571264,0.148614],[5.078466,-0.634139,-7.451976,6.151384,4.840129,-0.161679,-6.487398,0.300853,-8.962994,5.428646,1.325911,8.481987,-8.007026,-4.023849,-1.105097]],[[3.970593,-2.672590,-1.890321,-9.301414,-9.633957,-5.189554,2.723993,-2.694082,3.549123,-3.502858,-8.819405,-9.221823,-4.947627,-0.332646,-9.016619],[-6.894189,-9.187626,4.834389,-0.116835,-6.855723,6.924649,5.031055,5.252608,5.008056,8.289695,5.591286,-4.145787,0.629958,4.075004,4.478694],[5.894052,9.085628,2.407776,-7.649422,-4.263141,-3.424721,0.566371,3.494340,-6.580678,-2.020711,-5.132319,3.663623,5.329439,8.044342,-1.059368],[9.483934,2.694564,4.734977,7.283511,-2.144494,-5.417774,-6.411119,-3.463396,8.056708,-7.809594,-2.610534,8.221894,8.293217,7.317399,0.493289],[-6.809671,1.131188,-7.534168,-5.210207,-4.894822,9.126101,4.729075,6.248588,9.723721,-3.911247,-7.244413,6.601511,7.129047,-1.481034,6.185793],[-9.599272,-8.340339,9.389376,7.611008,2.066858,-1.101161,-6.692839,-8.798606,-8.967087,-6.256764,9.129359,6.682331,0.453312,-4.650800,5.882024],[3.047226,2.102271,5.853727,-7.030317,-7.232397,8.505726,3.611158,-0.206545,6.611458,9.343604,-9.215087,-4.186146,9.479472,-5.139048,8.532054],[-3.989209,2.824933,7.287009,-2.690580,6.880892,8.858305,-6.517601,9.357016,-8.665478,6.937712,-1.073569,-4.779521,2.297701,-0.513254,1.738597],[-6.471788,4.888455,4.240085,-7.074418,-2.372508,-2.919643,-4.313344,-8.009143,-9.890139,-9.865463,7.250834,5.715212,9.163228,1.889843,-0.959942],[-3.307542,-8.180150,-6.612926,-4.523687,-6.136699,1.614774,-2.392063,2.026095,0.624221,9.635196,0.348374,-9.645845,-7.506888,1.609096,-5.912971]],[[8.713715,-1.634841,-0.738644,0.557562,-0.672656,2.117519,3.806781,4.139247,1.486828,-8.674348,6.036436,8.905084,-4.802346,7.356277,0.476949],[6.998657,-8.400370,5.031807,-3.439722,6.604573,-6.488768,-6.913018,-8.224184,-0.565174,-4.612846,-0.694704,-6.013142,-3.814559,-8.787371,-5.944562],[2.481931,-0.202820,-4.855071,-7.981915,-3.035009,5.723901,-9.662356,-1.333802,1.535374,-7.723639,6.297131,2.204217,-9.358182,-8.413950,-1.828615],[3.228270,-7.828343,0.537107,1.997530,-6.699640,-6.265884,-0.969134,1.681972,0.270075,4.123730,2.135957,0.747622,-4.686542,-7.079945,-6.876272],[8.947250,-3.086414,2.028500,-2.656366,3.996068,-8.147308,-0.212587,-9.166885,-3.061054,6.846540,-4.062751,-9.120688,7.493628,2.638171,-2.456454],[-2.893137,3.395874,1.116460,7.628239,6.595274,-3.858855,-3.146991,-9.425200,9.758386,-0.656168,-1.146337,4.587936,-8.548490,2.464295,-9.949005],[0.558601,-4.702032,3.921905,5.011895,1.745234,0.539643,2.947983,5.393746,-3.734679,-5.538586,1.141662,6.465302,6.454453,2.799356,-8.153442],[-6.291386,0.591704,-3.261945,2.126108,6.469361,3.630120,-7.026477,-5.521470,-0.382038,-2.066221,5.589036,-2.738158,-9.574000,3.288255,-1.354710],[2.201706,1.345703,9.311593,5.096624,-3.614410,-7.815498,6.340613,-6.664126,-0.474773,4.172848,2.640069,-0.191133,9.381354,4.583821,-4.276458],[-8.699251,0.211861,-9.108281,-6.054643,7.157819,-0.095522,-7.787291,-5.685560,2.718131,6.008285,-2.810405,-4.430183,-6.111020,8.318147,-0.091894]],[[-0.998452,-7.146723,-3.246282,-0.711332,-6.790130,9.310721,8.930885,8.840934,-3.414338,-7.644961,2.158921,-7.517058,6.609663,3.968789,3.138408],[1.255748,3.658960,5.947517,-6.721366,-9.642162,7.421614,-0.762943,1.436216,-4.937606,-8.662882,6.894652,4.698326,6.052783,9.021183,-3.728833],[-0.379452,-3.561427,0.739490,-1.189535,6.606415,-6.338282,-0.267082,-7.285080,-7.977706,-4.596120,8.296791,-6.062575,9.482919,-8.492010,2.278937],[3.236012,-6.118319,3.059234,5.919570,6.418782,-9.138832,2.330060,-8.018738,-3.188016,1.033209,-7.608206,-4.167271,9.031245,-4.557412,-0.884553],[-6.150304,-0.029533,0.765886,-8.646821,4.561194,-9.985257,-8.428495,-6.384559,1.836171,3.482410,5.544052,6.040414,8.276318,-2.324401,-5.576056],[9.225928,-5.257759,7.155811,4.776787,4.517572,-2.293188,-0.166165,6.107466,5.414622,6.142611,7.537466,-3.099533,-5.764739,-6.235936,-0.836764],[-0.143015,7.732081,3.542904,-7.440530,4.091296,9.473391,8.391023,-6.500684,-8.774229,6.971132,-5.115372,2.444071,-0.476030,-7.637249,-1.858109],[6.540835,-1.734077,8.146347,-3.735161,-2.719097,4.189301,4.152749,5.623637,-4.624100,0.977881,8.818933,-0.796029,7.803731,2.505762,5.894243],[-6.933847,-9.077767,-8.843303,-6.811559,8.958658,-6.920509,-9.020625,-3.183716,5.331127,4.827205,-9.994267,-8.887957,1.220674,4.876317,-2.996507],[-8.527295,-9.737923,3.482547,4.092488,2.769491,2.422388,5.855704,3.883590,8.154598,-7.599809,2.590037,-5.965384,-5.129880,4.072155,-2.240532]],[[-9.242149,-3.957570,8.891910,8.350477,1.346781,7.515331,-2.606888,-2.029214,-5.606484,-0.549896,1.257349,-0.609028,-3.761629,0.588707,-6.933376],[2.956568,3.771534,6.278715,4.439303,2.521951,7.733410,5.073758,8.464031,1.745548,-5.255579,9.358065,-2.533530,8.439970,-7.385664,-1.419952],[-7.717865,5.741064,-9.005452,-1.745720,-2.723515,-1.413786,-0.897383,9.629940,-7.469819,-9.530524,4.260347,-0.359584,5.469226,-2.047659,-1.360405],[-4.957705,-4.206753,-4.419332,7.344721,-3.050874,-3.778509,9.412864,4.552642,1.449858,-2.330412,-2.876971,5.677912,8.803378,-2.714773,-2.547161],[-2.887365,2.760819,-8.481792,2.672183,-1.557746,6.745331,9.100004,2.410344,-3.898938,-6.647473,-8.390436,-4.488669,9.778232,-6.025040,8.288003],[-6.778298,-7.702199,-9.674578,-9.851107,-5.443840,5.911921,6.064334,-2.767912,3.394078,5.882597,5.275692,-9.462568,9.338274,6.798573,8.709407],[-4.845084,2.220637,4.703069,-6.224676,9.397577,-6.109042,-8.113727,6.579832,0.283153,8.841592,-6.145878,-2.822998,-6.344862,-3.195428,-8.796846],[-7.850125,-0.589812,5.240921,3.513496,9.394903,-2.331305,-3.534657,4.754013,8.769379,-9.954753,-6.861509,-9.106051,-0.303157,-5.081579,-0.730617],[-7.214953,-4.220642,-3.306926,1.551812,7.257456,1.557908,-0.894874,-2.550811,8.005266,-7.824145,6.676861,2.082958,-3.460866,-7.065178,-7.069392],[8.219508,-2.012219,7.767087,6.116100,-9.758675,3.541505,-6.765747,7.185804,-8.135828,-2.125902,4.000839,1.661835,4.407648,-1.536091,-7.570841]],[[1.996815,1.835508,1.112276,-0.430356,0.162984,7.442242,-8.003565,7.527310,0.452347,0.317730,-2.016271,9.237706,0.358234,9.804064,-1.742667],[-1.162372,2.663367,6.822096,6.914730,9.948115,5.876049,7.659642,-4.711879,5.531273,-7.663994,-3.991937,8.071510,-1.406457,-0.562624,-5.281559],[-1.910830,-9.068746,3.173904,-0.102510,-7.534344,1.047834,8.984468,7.466574,8.625595,2.800796,2.447302,9.240661,9.328508,2.953495,2.719273],[-2.924600,4.691410,-3.292722,8.960861,4.091724,-6.565537,8.065898,7.443337,-1.630492,0.584521,1.002421,1.990661,-1.812406,-7.405758,-2.021449],[4.679168,3.096324,-8.103920,3.659801,5.166689,8.688453,-2.817831,5.654264,9.695227,-0.610107,3.307632,-0.486635,-6.861589,7.713805,-0.535120],[9.429739,-6.992473,-0.174237,-2.341312,-5.884928,-2.344648,1.560509,-5.561536,3.788450,-6.578085,9.869237,4.393576,-8.650369,-9.318572,-0.320613],[-0.713524,-0.021433,7.983056,4.617953,0.640265,3.379512,1.360661,-7.944229,6.297723,0.724892,2.366013,5.273083,-9.074290,1.270383,4.161876],[-2.447343,9.265423,-6.738794,7.911600,-3.284111,-3.923762,0.829403,-3.547689,3.991691,1.173580,-8.216733,1.065089,5.277804,5.399383,-6.711482],[-7.752842,-6.425866,-3.782092,-9.194480,-7.002749,-3.839693,-5.086841,-5.828888,2.192033,-2.962828,9.388102,1.671425,-9.582745,-1.087393,1.429293],[-4.810588,0.742858,9.662515,-5.252727,-2.761120,-2.627430,7.539161,-4.458645,-4.869076,6.892582,-2.722593,1.710324,-8.283802,-5.192386,5.960883]]], dtype = "float32")#candidate|1059|(10, 10, 15)|const|float32
bop_1060 = relay.less(var_1058.astype('bool'), relay.reshape(const_1059.astype('bool'), relay.shape_of(var_1058))) # shape=(10, 10, 15)
const_1069 = relay.const([[[-2.898266,6.880485,-2.738540,-5.789620,-3.099020,0.593526,5.542655,-6.045877,1.554128,-2.698813,3.750634,4.930071,9.082918,1.021772,5.296935],[2.501720,-0.076505,9.233044,-8.581902,-1.305112,2.212353,-9.216533,-7.466856,-3.134833,-9.055939,8.252694,-0.202677,-9.209040,8.062173,2.607669],[-9.421028,-2.445846,9.202943,3.821867,6.028995,-6.591246,2.031038,-7.658323,8.665476,3.924209,3.725229,-5.147239,7.242658,-6.512210,-7.049115],[-2.649885,-1.005319,3.482481,-4.696965,-9.175107,-2.618129,2.030785,8.010731,-1.040073,-8.911545,-8.254979,-1.234743,8.317665,-5.417515,-3.560754],[-3.788048,3.701315,3.084574,-3.901316,-5.493004,-5.615849,-0.417284,-1.009298,8.025506,-0.829227,6.512919,7.072284,5.873242,3.054144,-2.669772],[-1.010983,-9.517282,-7.497170,-9.872559,-0.089054,0.969419,-9.828982,-8.773762,1.928841,9.364108,5.849637,-3.128826,-5.383999,0.876214,0.622042],[5.885700,-0.310901,-7.218300,3.383526,-1.590442,1.468685,-7.613999,0.773615,-3.063887,-5.131843,-8.346705,-7.995502,7.929110,3.263241,5.418805],[-9.614239,-9.354582,-5.216057,5.694057,-2.077641,9.648817,2.047342,7.122087,-1.971191,2.677520,3.009466,-9.618169,9.012848,-7.749630,9.384930],[4.537036,-9.967695,-1.671770,-1.463755,-3.669471,7.200470,-6.066739,-5.475319,0.935721,-1.723998,0.044230,3.634823,8.520814,0.097053,7.579810],[5.790427,6.668326,-4.246813,1.429050,2.915687,0.564283,0.193056,-5.744102,5.073872,8.016993,-6.203311,8.933288,-7.081269,-9.200595,-9.981421]],[[-0.675650,9.486611,4.425337,-1.379776,-1.254808,3.293184,-5.728140,-1.831117,-4.979152,7.667396,7.379985,-7.520681,9.465284,0.610750,2.304883],[6.957935,-5.273780,7.644041,-9.587995,-7.776687,3.717672,5.845728,-5.143186,-9.153111,5.478925,-2.901678,-4.982695,-4.627951,-6.238862,9.944348],[9.787374,8.712835,0.912620,-1.350865,-7.775718,2.821706,-4.434693,-3.671383,-8.694838,2.400447,-2.999357,-1.160668,-5.889994,-7.428671,-1.898481],[-4.223601,-7.828398,4.958242,-9.371281,9.501897,-8.094474,4.741295,6.353327,-3.187875,-7.748551,-2.170918,2.605852,5.415509,5.682648,-7.680621],[-3.355124,2.110036,4.579865,-5.481974,-9.084004,-7.218619,-2.569490,2.506423,3.384398,3.592498,1.269195,-8.584959,-7.202352,4.520751,1.056781],[-6.328052,-6.662907,-6.907429,-5.692703,-4.783355,4.787495,-4.651426,-1.980628,4.982936,2.655028,8.621974,-8.934264,2.425140,-4.276871,-1.505523],[3.717598,4.898109,3.500170,1.376378,-9.920914,-5.787654,-0.395409,-8.135997,6.484316,9.883755,-0.715782,1.225847,-5.710113,-4.037904,5.607913],[7.786507,7.515310,2.103636,-4.172200,-2.040451,-5.249555,-0.701318,8.396834,7.359959,8.928768,8.846485,0.117791,-1.678243,-7.777658,-4.934692],[-6.016650,-1.194600,-6.084003,-0.397849,-8.536179,-1.260960,2.413162,4.061354,3.792982,2.759876,1.218528,-4.451606,-9.326339,5.931734,6.484679],[-6.198560,8.975585,7.030305,-3.828747,-5.609764,2.254443,2.123939,-7.918438,9.717545,1.713118,-0.620746,-6.857152,-4.924352,3.647902,4.421541]],[[7.506809,-2.015773,-9.137936,2.640617,7.044997,4.015551,-0.129303,-5.405793,3.611796,-3.835530,-4.695390,-4.252774,7.380719,4.986833,-5.812723],[-7.998827,9.398531,-2.295071,-7.779673,-5.401037,9.136641,-6.783179,4.862889,-1.898621,8.051339,8.324179,9.542805,-0.155866,-7.317417,-3.060650],[-4.250853,9.162967,9.633501,6.268605,3.539425,-4.099160,0.492033,-7.617579,8.660566,3.049824,7.974073,-1.350923,-0.657396,7.259523,6.275453],[3.979589,-3.246498,3.523967,-1.869967,-8.299136,-8.253502,4.565157,9.942460,-7.239421,-3.800027,6.904719,2.839183,2.597021,-5.422056,-9.603622],[-5.854355,5.542593,-7.827923,-2.180328,4.894472,-4.190515,1.722014,5.106557,6.904189,0.692167,-3.025902,1.474163,4.219809,8.029979,-5.310124],[2.401303,-9.288460,3.662358,0.289596,0.791006,-3.354457,-1.959487,9.991753,5.343485,-6.472050,4.147617,8.591031,-8.469431,1.669930,7.937984],[7.949728,-0.797095,2.220799,5.892600,8.890272,-1.313128,-2.473031,-8.647027,-8.017031,9.771281,-2.181317,4.307540,6.753751,-4.986058,-7.568287],[-8.000339,-4.909662,5.398688,6.629138,-7.599174,4.761290,-2.738435,-6.665145,-9.678176,-0.850595,5.619065,-4.408358,1.428310,-8.824673,-0.998954],[5.291259,-9.540126,6.336007,9.976418,-6.188595,6.558288,-5.522177,-7.730751,-8.485031,-4.879132,-3.071228,-7.796434,-0.675607,-3.597660,3.431626],[2.976913,-5.849927,5.198390,6.891286,-9.463597,-8.846315,-5.505475,3.836083,6.377798,1.504770,0.585162,2.260539,-5.345653,3.301818,-0.173573]],[[0.986959,6.914756,-5.255638,1.737413,7.711195,6.746717,1.714414,0.916533,-2.668718,0.349309,-8.276809,0.074615,-4.434516,2.657646,8.592351],[-0.282652,-5.183148,-3.813744,-1.133262,-4.527820,2.422629,-2.707172,6.916428,1.841496,-7.335557,1.405843,2.443783,9.049772,-2.158578,2.191461],[-3.859048,-3.405084,3.216239,6.963794,-6.496091,-5.905225,-7.378087,2.838145,-6.808850,6.337508,8.623198,2.105715,-7.869306,-0.486366,4.627474],[-1.757938,3.011309,-2.982532,9.683292,5.307482,-6.187540,0.709779,-7.385323,-7.828250,2.194719,6.956975,3.456222,-4.201982,8.150123,-8.291599],[-7.466063,5.290260,-8.632248,0.955526,3.634889,2.682920,-4.641869,-1.356991,2.345602,0.128327,8.179920,3.696821,2.033346,-3.352870,-7.437984],[4.886294,3.808521,9.397474,5.302077,0.075697,2.299078,-8.203818,8.010233,-1.938609,4.357550,9.970560,-0.687684,3.894826,-3.940412,6.148747],[0.488991,-1.412696,9.674976,-1.952769,-8.729137,2.131564,-0.705265,5.520572,6.780372,8.883507,5.644422,0.619595,8.712594,7.664219,-2.968867],[1.695007,3.247370,-3.561694,4.448521,6.746852,-6.193628,3.669624,-7.728034,-4.763046,-8.455082,5.633607,-6.441107,9.041511,0.229179,-6.421375],[5.517874,2.703486,9.139791,4.209951,-3.494423,-7.068805,-3.110784,-2.266640,-9.975122,-1.223334,-4.556324,-0.499312,-4.999466,-5.710299,-4.364378],[2.910334,-7.063430,-0.909390,-6.762611,-4.713750,1.698708,3.779764,-0.205577,-7.296700,-6.744516,4.956823,-7.348040,0.567524,-3.952732,3.408221]],[[-5.564382,-2.513817,9.200669,-1.804456,3.496191,1.106988,9.081944,-4.661033,-7.006659,-2.763336,1.759482,4.889075,0.176466,-6.118122,-6.671376],[-6.883267,-1.920701,-6.109829,-0.216662,8.319977,4.607462,-1.515951,2.371773,8.669935,9.198045,-5.349949,4.149305,7.142005,-5.055677,-7.064494],[1.454340,-5.708285,-7.363749,7.507813,1.950781,-8.647955,0.785589,-1.697259,9.116889,-7.160531,5.370221,5.815035,-7.656451,-8.119585,-7.223555],[-0.645589,8.111131,7.638473,0.354009,-0.363097,8.578494,0.374967,9.042301,3.049135,1.537916,-9.132851,-8.537938,-1.831663,-9.701976,6.929267],[2.446922,-6.692517,-7.485897,-3.283550,-7.696258,5.804229,5.801046,8.269789,-8.375365,9.858043,-3.878891,-1.812573,6.600863,-6.270273,2.156479],[-2.786055,3.029612,-1.693119,-3.721196,6.115005,9.597645,-7.593150,5.118324,-3.610557,-2.743844,-6.167848,0.223742,3.355307,4.852574,-5.164065],[-5.785883,1.220227,3.652730,0.361848,3.371702,-3.755781,4.260363,5.367632,-1.035319,-8.637952,-7.721110,-2.654787,-4.730011,-6.786029,2.505064],[-7.690828,-7.226808,9.554429,2.336384,3.502786,8.795651,-1.280706,5.784572,0.740238,9.117726,-5.248597,-5.335813,9.676007,1.527721,-3.866254],[-7.423702,-5.896481,7.029352,-4.027121,-6.972102,0.958607,0.511928,4.049892,4.982085,7.498860,-8.833309,-2.049774,-8.506924,0.958517,-1.759026],[-9.461664,3.320292,9.262557,-7.693872,4.069877,3.779041,-9.901027,-9.874110,5.118818,-5.492084,-7.810435,0.506397,-6.479254,-0.589748,8.930063]],[[5.965008,9.504141,-7.207593,0.647605,-8.318858,5.305529,0.149853,4.417172,9.822532,-1.555718,8.583584,1.027501,1.339460,-6.550505,7.235975],[-4.369121,1.147848,3.121175,-1.298644,-7.016964,-8.072983,5.281859,0.131737,2.195125,1.390815,-1.504347,9.861483,8.863442,9.411884,-2.655601],[2.619022,-4.061045,2.230240,-1.104821,-8.106913,-5.584760,0.116447,0.690718,-8.212094,8.913828,-6.239614,-6.456603,0.630715,-8.478046,2.145107],[-3.015678,-7.384168,2.631330,-7.458589,9.353069,-5.359499,4.279290,8.840306,-4.578569,-0.100196,-8.929416,-6.689954,-7.347132,-7.105455,-2.729976],[8.646469,3.737376,3.224907,9.982689,-2.113021,2.679358,3.388465,-6.573109,6.708138,-8.617633,1.692650,-0.587991,-6.811511,5.494525,4.395131],[-2.106819,-8.391294,-8.500863,1.692490,-5.068006,7.680623,-5.410549,-8.187014,0.417919,-4.385805,8.398772,-2.310959,0.871242,-2.519291,-0.539548],[-9.952092,5.416167,-3.093748,6.751682,7.559325,3.733569,0.686337,8.901201,2.252654,8.336526,6.716298,4.919458,-4.677105,-9.838937,3.910378],[-7.552048,7.284317,8.449668,-8.615346,8.747837,-4.280344,-7.701584,4.113593,8.323566,-7.655364,-2.912865,-6.370924,-0.293125,-6.897000,6.757492],[5.539542,8.789230,7.268948,-1.397651,5.906783,-6.345548,-5.751385,5.582113,-2.289118,-7.884663,0.992805,-9.745372,-1.677971,-9.143478,1.121079],[4.131854,-4.251353,-3.357749,5.951995,0.840157,-6.637593,-8.625884,-1.150512,-0.697943,-6.491059,4.404609,9.715038,6.807191,-6.404803,-3.113525]],[[-1.678609,-8.937389,0.647403,-7.632722,6.370913,1.266673,9.990737,3.826535,-3.360302,5.166336,-6.133474,-5.343541,-5.385493,-9.729034,-2.282672],[0.696132,-8.482325,4.780594,-5.473846,-8.031430,8.244222,1.311960,8.776400,-7.323085,8.275741,-4.715676,-3.415441,3.972582,5.733567,-9.015062],[5.899969,-0.118409,6.303835,7.151994,9.824386,-9.058540,2.961449,-6.113810,-4.274821,-8.394232,-7.220773,5.246799,-8.315604,1.539446,-9.890659],[9.397717,8.803398,6.256043,6.827006,-4.984900,-5.309615,7.856714,0.876026,-7.962570,-8.860392,4.295484,-1.113171,6.297055,5.351181,-3.520495],[-2.672303,5.636070,1.142598,7.922785,4.301294,4.811102,-1.306696,-8.339082,-9.028105,2.824559,3.699421,5.465063,6.769569,-1.629871,-0.151022],[3.713057,-7.783596,-0.985925,6.527204,-7.515491,-1.669655,7.669474,-2.875081,-0.166443,-4.439653,1.877770,-4.512880,2.350202,-8.362712,-4.547499],[-4.535006,6.146127,6.725579,-0.667252,3.034029,2.199348,0.470734,-6.853199,-4.073724,-4.899259,-2.167794,3.798300,2.683350,4.972613,-8.788981],[-5.912964,6.531516,-5.559067,5.711214,3.474258,4.969934,-5.215991,9.603140,1.208522,0.937986,4.867172,0.570941,-4.426731,2.568069,-3.481197],[-8.582986,6.739916,4.929124,-5.040031,-6.307714,0.397690,-7.015512,-8.911291,8.027804,-3.527842,-6.609755,-4.734835,-7.324195,-7.246948,-8.740933],[6.734683,6.057420,-9.288592,-1.334946,8.212485,3.657883,2.883238,8.068970,7.751519,9.253306,2.019861,5.274089,7.430235,3.026104,-1.695399]],[[1.979948,-2.127458,-4.359241,9.266111,8.832596,0.914393,1.438531,-7.126730,4.780898,7.002499,0.759133,-3.275724,3.431016,-5.378005,7.930762],[-8.075117,7.689712,2.703340,-1.515608,-0.272258,9.811665,-7.641945,-8.523633,9.199513,-7.292227,-5.220349,-2.599213,7.860120,9.417343,-9.499533],[-6.955945,6.073075,0.886723,8.854452,1.801946,9.974352,7.278899,3.787313,-7.823643,6.445730,-6.361155,2.793537,-2.222133,-4.572268,-8.478948],[-3.332379,3.710404,-3.979603,-4.540507,-2.492085,-6.896410,7.754779,8.772091,2.644482,5.392722,3.683807,4.181613,-6.830949,9.115585,4.838890],[5.122680,-1.771357,5.580067,2.519673,-6.991563,9.680440,1.766937,-0.297017,1.951549,3.526063,3.786772,-8.888589,2.311714,-1.058085,5.534381],[-1.641860,-6.574214,-3.454529,-6.971081,9.512385,8.170943,4.733130,2.841440,-0.776862,6.845262,-2.985879,-7.723486,2.184627,3.283166,-9.803346],[7.416200,-7.831503,4.536117,2.570509,-5.420057,-8.214956,5.128725,9.826456,2.514710,-2.260935,8.317443,8.650798,3.245052,4.958515,9.811564],[-6.055375,-8.476965,-9.445225,5.376693,3.874402,-4.729051,2.496772,6.983960,8.505842,-3.872304,-8.160979,-3.067874,2.643788,-0.315924,-8.470296],[-2.575108,5.514642,4.648009,-6.366658,8.539609,0.161222,4.397017,-7.921237,9.544899,-1.968255,6.590293,7.580907,-8.264347,8.530766,1.967781],[-9.264880,-9.997776,-3.769572,8.487171,-0.020208,2.111276,4.513390,8.117515,7.576046,3.138937,-0.390146,-8.565818,-0.635050,8.351989,-0.271707]],[[0.335007,-0.346755,-6.948704,-1.355394,-7.754609,-6.466884,-1.945626,-2.635840,1.452041,-3.603181,7.831518,3.330257,-1.872470,9.149457,-1.309814],[-7.768344,-4.110353,6.141127,5.164326,6.610383,6.528229,1.400260,8.456426,7.626189,-4.423668,5.040520,-4.726851,6.824835,-2.373248,-2.573012],[-9.590921,-1.326555,-0.585223,5.017729,6.385132,5.909793,0.828255,-6.772592,-4.277357,5.333604,0.783783,6.519875,-9.835697,-6.749942,3.603453],[2.768107,1.488183,-0.780333,5.137019,-2.501760,-4.557895,0.138101,-0.147337,-1.958965,9.958308,1.858818,-6.423471,8.734666,-4.920663,4.396969],[-9.716067,2.798294,2.358389,4.817765,-4.141420,5.243998,9.512398,-6.883086,0.643516,4.074155,-9.397862,1.209909,9.348526,8.278262,1.355450],[2.666489,9.079563,6.545969,-7.676293,-3.383514,-7.672872,-5.870157,-6.105555,-1.992518,1.549678,0.346766,-2.827387,2.971576,3.490466,8.128791],[2.338499,-2.305212,-0.039229,8.340435,6.861196,3.886109,-6.282314,7.620938,-6.886048,-1.479860,2.513797,8.312837,-2.979356,2.080305,-2.638124],[-3.576316,8.200873,-5.900185,9.450755,6.685434,3.925822,-7.501177,-8.748007,8.394791,9.820488,-8.170292,8.558876,2.448040,-0.918216,8.534691],[6.286647,1.800880,-5.994069,-2.073258,3.801303,-8.392492,-3.519112,-1.059574,4.026048,5.890332,3.822498,4.293372,-3.010400,9.817637,-5.066252],[9.169173,-3.908596,-7.720736,-9.709939,-8.164428,-9.857785,6.255973,-8.138305,3.834217,7.969712,-6.295497,0.828776,3.360143,-8.839778,-0.004783]],[[3.770601,7.526623,-8.345589,-6.110331,-6.801625,9.737704,-4.900028,-2.752445,-6.143300,-0.933328,-5.756828,3.228412,-9.393296,7.935169,9.630439],[0.438125,3.770380,-3.689692,8.897353,7.988571,-5.284354,-6.530107,9.658944,6.705495,5.706185,6.444151,-1.452871,0.506858,0.914794,-6.740823],[-8.990004,1.160469,-5.454874,-5.648249,-9.432519,8.164968,-3.839703,2.977703,-9.817127,9.364249,-3.138545,-1.962402,0.008128,-1.668691,-0.984621],[4.307859,-7.821744,6.906510,5.552098,-8.188624,3.992105,-4.092022,-3.048320,8.405120,-5.479641,-0.142049,8.775998,7.085350,-1.698148,3.574255],[-8.223325,9.741016,3.026785,7.193222,-7.556962,-2.381888,-9.096473,4.622819,8.645398,9.737599,-5.157674,-5.308808,-8.000512,-8.997371,5.426692],[-5.510358,4.004722,8.960396,-7.327974,5.947887,-1.161770,-5.700485,-1.892010,6.529578,-7.172653,9.837560,-7.692822,-2.335725,2.721228,9.669163],[-7.945849,1.918016,-7.721201,-5.672482,5.865428,0.415018,6.523668,-4.875883,-4.537389,3.777359,7.076891,3.013113,6.456046,-6.931562,-5.675625],[7.210580,1.492829,-0.716177,9.868347,-4.468733,-7.481668,7.086183,7.140814,2.728556,5.339087,-1.300702,7.320683,6.889046,4.240934,-0.205032],[3.071229,-3.468726,-0.683170,-5.605005,4.018485,-7.244547,8.343521,-2.128501,-0.911954,-7.469266,-3.691768,-3.735331,-6.131755,-6.929803,2.611982],[8.830087,2.780455,-7.505185,4.375865,4.048893,4.376094,-6.018746,3.690133,-8.173830,8.082619,-3.650114,6.584716,6.144421,-9.032612,-1.131489]]], dtype = "float32")#candidate|1069|(10, 10, 15)|const|float32
bop_1070 = relay.right_shift(const_1059.astype('int8'), relay.reshape(const_1069.astype('int8'), relay.shape_of(const_1059))) # shape=(10, 10, 15)
var_1087 = relay.var("var_1087", dtype = "float32", shape = (10, 10, 15))#candidate|1087|(10, 10, 15)|var|float32
bop_1088 = relay.add(const_1059.astype('int32'), relay.reshape(var_1087.astype('int32'), relay.shape_of(const_1059))) # shape=(10, 10, 15)
bop_1100 = relay.logical_or(bop_1070.astype('bool'), relay.reshape(var_1058.astype('bool'), relay.shape_of(bop_1070))) # shape=(10, 10, 15)
output = relay.Tuple([bop_1060,bop_1088,bop_1100,])
output2 = relay.Tuple([bop_1060,bop_1088,bop_1100,])
func_1104 = relay.Function([var_1058,var_1087,], output)
mod['func_1104'] = func_1104
mod = relay.transform.InferType()(mod)
mutated_mod['func_1104'] = func_1104
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1104_call = mutated_mod.get_global_var('func_1104')
var_1106 = relay.var("var_1106", dtype = "float32", shape = (10, 10, 15))#candidate|1106|(10, 10, 15)|var|float32
var_1107 = relay.var("var_1107", dtype = "float32", shape = (10, 10, 15))#candidate|1107|(10, 10, 15)|var|float32
call_1105 = func_1104_call(var_1106,var_1107,)
output = call_1105
func_1108 = relay.Function([var_1106,var_1107,], output)
mutated_mod['func_1108'] = func_1108
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1132 = relay.var("var_1132", dtype = "int16", shape = (8, 12, 4))#candidate|1132|(8, 12, 4)|var|int16
var_1133 = relay.var("var_1133", dtype = "int16", shape = (8, 12, 4))#candidate|1133|(8, 12, 4)|var|int16
bop_1134 = relay.logical_xor(var_1132.astype('int16'), relay.reshape(var_1133.astype('int16'), relay.shape_of(var_1132))) # shape=(8, 12, 4)
bop_1140 = relay.mod(bop_1134.astype('float32'), relay.reshape(var_1133.astype('float32'), relay.shape_of(bop_1134))) # shape=(8, 12, 4)
func_1104_call = mod.get_global_var('func_1104')
func_1108_call = mutated_mod.get_global_var('func_1108')
const_1149 = relay.const([0.916426,5.445190,-9.443723,9.433399,4.860201,9.691564,8.575046,-2.963702,-6.320449,8.666417,7.505966,-8.852289,6.397965,2.887322,5.977526,-0.716937,-3.083789,3.470299,1.873755,7.168512,5.363433,-9.602614,-4.474411,-9.485868,-4.201832,8.404989,-6.305892,7.326961,4.998424,9.146237,0.955188,9.451268,4.872587,6.010455,4.429064,-7.282723,8.204321,-1.184131,1.675238,-5.619132,-9.884799,1.725199,-0.579472,5.014167,-1.949913,-1.208766,-7.938561,8.991339,-5.876755,-7.304956,9.452132,-5.497642,-9.179380,-3.952737,-5.663043,9.881611,-8.988371,1.095103,5.733659,3.648399,1.732337,1.688920,9.641616,-3.813273,1.113539,-0.012106,7.594824,5.043752,0.257360,7.789074,5.399204,-4.489269,-7.030419,-2.779426,-7.097734,4.134657,1.253487,7.896716,3.134437,9.517578,4.194071,6.201700,-0.980467,-9.622463,4.552169,5.401804,-7.087248,-5.627743,-9.942089,-5.809233,-5.282768,-8.479660,-0.028071,9.896523,6.064580,5.799000,-8.043715,1.388748,0.951290,-3.721608,-9.633787,1.349346,4.883386,9.974976,-3.089047,3.448534,1.789069,3.569903,1.916055,4.249558,2.037052,2.722861,7.271056,-3.670001,-7.713490,1.873159,8.874591,7.660697,7.221064,-4.009361,-1.541479,-7.532386,5.407957,-5.985081,-2.165041,-8.500605,-6.494167,8.753294,-7.140688,-3.989772,-9.715096,5.794716,3.637202,3.745780,6.395926,-8.070913,8.455166,8.387503,-4.614714,0.947156,1.911868,-7.869409,-5.130069,-3.158875,-4.642221,5.461814,-9.178656,-2.902747,-7.852549,-3.862362,2.286658,4.109118,-0.688403,3.481401,-8.894254,3.580834,-6.314359,9.398639,-6.222641,-9.834093,5.341979,9.912417,-0.104750,5.698002,1.120543,2.874707,-2.764297,3.102224,2.386621,-0.858043,-7.942043,-4.115461,7.218228,9.418543,1.270779,-0.935189,1.353983,-8.990493,2.187060,-9.030963,-8.186118,1.813398,0.125464,6.284914,-8.632980,-4.779179,4.023718,5.368596,3.073776,-9.261883,-0.235436,5.292500,7.021213,-8.037220,4.758951,5.738629,-4.105775,-7.350708,-5.203991,-5.942589,0.599349,4.422977,5.666558,-9.419609,0.503391,5.694871,5.198259,9.688101,9.125706,-4.632842,0.585601,-6.969236,-0.064735,0.255521,-0.335249,0.202895,-3.014864,-6.146161,-8.905926,-6.029939,-1.676250,3.849383,-4.145015,-8.633599,-9.047097,-3.891628,-6.511029,1.983231,-8.856306,6.042801,8.904205,-6.843493,3.998881,3.626899,7.846837,1.679845,-8.938888,-2.992970,-3.403548,-1.468667,9.433984,-4.955407,1.647970,-2.164214,-2.534315,-9.324782,2.744219,8.729301,8.404813,-1.964192,1.136245,-6.279841,-6.413705,-6.719808,0.764165,-4.820520,-5.828345,3.272560,-3.399915,2.655690,7.185281,-8.801508,-1.569644,-8.051173,8.738146,0.914052,-8.778982,-8.714190,-5.928569,0.052608,2.258248,-7.450796,-4.186825,1.121675,6.375326,5.263278,-0.697045,0.066816,-5.263356,-9.569304,-7.221354,-0.597924,-8.972193,-3.011577,5.746150,-2.968173,7.713134,-9.955941,-8.880414,-9.075319,-4.262988,1.425203,-0.991409,-1.553444,-8.058339,-6.531142,-2.437378,2.061486,6.013683,7.648713,-7.522683,9.391515,-8.321033,0.797832,6.186093,-6.744099,-5.516782,-7.729479,5.461546,2.704988,7.557535,-8.778299,-5.744404,-3.765061,9.691569,-6.307439,-4.084507,3.869940,-8.983945,7.392880,2.791730,4.275531,6.817414,-7.458720,-7.771325,-0.315634,5.851544,9.499878,3.099340,6.621253,0.570695,-4.144084,3.935513,-8.706370,-7.493528,5.127534,-6.071725,0.128581,-5.239181,-5.957660,7.600191,-7.729537,0.450518,-7.189614,-2.360612,-3.366840,-0.276840,5.726449,-6.068056,0.444097,3.753236,-7.429394,3.708875,-7.670512,6.977546,7.893309,6.975314,7.735832,4.230433,9.493788,3.684552,-0.911374,-0.535987,4.552663,6.431142,-7.074060,4.332274,-0.472531,-6.027948,-3.496421,5.065411,5.790474,-8.973215,-8.442972,-6.430931,7.373727,5.059095,6.878400,1.464253,-6.567260,1.660510,5.792267,-1.465416,-8.743774,-1.299031,-4.933073,-4.914517,1.771484,2.366227,-5.010518,1.476411,9.133325,4.797136,3.079352,-2.599755,-6.117873,-4.427846,-5.101796,6.196882,6.398504,-2.923510,-4.973583,9.442719,5.382255,1.558223,4.426670,7.910091,7.499673,1.533627,4.582163,-2.633527,-9.419553,8.595330,3.538611,5.707147,9.003246,3.879254,-8.596083,1.134209,9.437958,6.365109,7.055768,4.109334,-8.242761,0.170353,-1.481350,2.599537,-1.044457,-1.787234,0.215078,6.865543,5.016878,-2.464171,3.423847,1.032606,-4.399646,0.035071,3.136179,-9.061471,-0.791993,8.724128,-9.929216,9.825928,1.712582,-4.609395,5.881936,1.427219,-7.186433,-1.655736,0.653411,-3.291958,-2.734609,-6.910860,0.741819,8.993909,-5.711774,-1.607356,8.193147,-7.449170,7.793085,-4.834746,-2.145119,5.574027,4.682050,-0.340663,3.088285,0.113470,3.808217,6.588838,-7.677394,-6.617810,6.126997,5.803741,-4.471441,-3.458197,9.131945,-2.723144,-5.541206,-8.552876,2.301098,9.713246,-0.688032,-9.412385,-9.771652,-1.696454,0.254078,1.647372,7.685370,3.400243,-6.799803,-8.364336,-0.597652,7.808408,-4.048007,0.239844,2.279558,1.878786,8.953991,9.551342,5.796563,-2.193001,-4.001866,-2.852346,0.132365,7.646495,9.398135,0.383050,-7.489831,-2.883050,-4.698346,-2.948085,-1.824707,0.804543,9.880318,3.523670,-7.302041,6.573341,-3.919184,3.486284,-4.957645,2.541042,-2.477001,8.596098,-6.895971,7.010770,9.115223,-6.480023,5.878626,3.151933,-8.647519,-8.633016,9.256940,6.427577,8.235745,5.212528,6.075321,1.122374,7.861612,-7.585910,-4.925510,-6.004664,4.805282,-7.328308,1.339119,8.794124,-9.646861,-2.904186,-0.652473,9.507127,-3.238051,3.420932,-1.970975,6.307312,5.536883,9.101497,-0.265050,-2.212282,-1.144121,-6.490598,-2.190376,2.310896,0.894044,-3.847248,4.246598,8.182255,-6.353305,8.166982,1.484588,-1.778552,-0.799955,8.317691,-0.648467,-6.420465,-8.898724,-3.990693,-1.134309,0.939546,4.435923,1.609890,-5.291296,7.743084,5.173160,0.095940,8.278581,-3.687005,5.374222,-3.271909,1.491698,-6.195485,-3.423522,-8.281465,-8.310625,-3.897099,-2.100649,-9.971387,0.849954,1.603695,-4.001256,2.082341,0.342919,4.266523,3.601583,-3.843181,-0.162445,-8.289186,9.320177,1.422501,2.081769,5.313901,-6.490028,-8.639412,0.037224,-6.716006,-3.554921,-6.703324,9.295033,-8.222743,3.545115,5.541443,3.919542,-2.176041,6.369008,-9.214178,-6.921957,-3.874490,7.307233,-0.420141,-2.371868,-0.078314,-0.871473,-7.630545,-9.855694,8.020807,9.251459,9.315417,-2.522100,2.119286,6.720857,-8.783071,-2.351178,-1.394286,8.017464,-3.993246,-2.464278,-9.765129,1.191044,-1.824596,5.005935,-6.302207,7.097932,7.045830,1.267088,7.932401,-2.043988,-4.583463,-7.909833,-5.569156,6.503303,9.551788,6.084352,6.213887,9.280299,-0.547345,3.663778,7.898366,-2.716070,-6.996677,-5.200478,0.884634,-2.915908,7.592402,-9.205909,-6.043645,-0.464704,4.116155,-4.084135,5.117526,7.155464,-3.900771,-9.409632,7.429785,5.620348,-8.593919,2.602572,-6.597229,-1.477135,-4.876103,-1.624781,-6.529211,-5.881120,-8.212131,-1.168258,-3.719963,-9.479476,-0.438148,2.234816,7.243152,-6.108826,-3.940658,2.998585,3.982145,-5.801785,2.683806,-1.584236,9.564267,2.733615,5.389570,0.309705,3.189417,7.041666,-3.224902,0.544628,8.250284,-1.526686,3.175687,-9.658605,5.968057,-2.371178,1.170782,-0.562477,-4.529874,1.090331,-3.038014,5.927045,-2.239353,0.085782,2.212857,-5.843730,-4.457834,-4.778696,3.310324,-3.020167,5.652363,5.434864,-8.690556,-6.133178,7.908276,-2.442818,-5.416433,-4.425285,-4.201798,-2.247995,-2.792641,-2.804771,3.566871,9.377283,-2.741577,6.808515,-3.087689,-8.208867,-4.667523,0.909390,-8.518206,8.938552,-1.981257,5.363818,-9.469780,-2.502796,6.228720,3.503796,-8.694586,0.375380,3.764816,-4.020220,2.578392,-8.552487,7.535016,8.169901,1.276558,7.554534,-4.567975,0.917742,-4.536625,-7.538802,-9.893895,-6.616626,-5.418267,4.375505,-7.917198,4.308965,3.744783,0.685106,-9.200854,-8.944519,-7.785975,7.053816,6.573268,2.359726,-5.766175,8.519853,2.768857,2.959347,9.546396,4.734475,7.004805,4.594703,-2.174972,4.398059,2.366791,3.809944,8.052239,3.506866,4.188975,-2.363163,-4.773749,-4.787506,0.989772,4.698148,1.403668,-4.108614,4.187249,1.693214,5.891269,-4.783512,-5.380780,-9.110545,-3.839243,-8.243312,-4.190679,-8.337259,-5.618066,-3.956090,-5.826872,-7.415575,3.893287,-5.139520,-9.931280,-7.822925,3.641536,0.591106,-1.653946,6.725934,6.119774,-2.487027,0.403020,-7.725584,-0.852196,0.963771,-2.178812,-3.486302,1.786523,-6.398408,-7.666657,6.846140,-7.962619,0.912700,-0.954228,4.127506,7.018103,6.794691,-8.225680,-4.649843,-6.261692,-1.374683,-1.801783,-5.186238,4.121839,6.512108,-6.082657,7.566986,1.105989,6.161501,-7.024888,9.889618,5.358685,-3.897405,-3.829182,-5.426730,-0.573822,7.087880,5.116095,-9.605443,2.294096,-9.145006,-0.085314,-1.167944,-1.509889,1.824946,-1.117210,-9.179311,2.950650,-0.764937,-6.424610,3.987531,1.361115,3.065937,-2.141877,3.750005,7.275434,5.288851,-7.731875,-2.033671,-9.180688,7.760653,-3.537419,-2.317602,-4.630151,-3.131537,3.964527,0.651141,-4.114183,4.034613,8.241770,-3.522936,-8.766477,-6.451503,8.704784,0.258535,0.587438,9.350533,3.166939,2.911212,-3.012399,-4.004175,0.859047,-3.585942,-2.932999,3.657379,-1.745197,-6.452850,0.277729,0.664018,4.301985,-2.235175,0.542728,9.820278,6.076452,-9.400913,-0.771077,9.819725,6.886577,5.408419,4.810349,6.219424,0.347729,-3.060132,-1.501964,0.132349,7.436318,-3.081830,0.724461,8.769624,2.199986,-6.317382,-0.377587,3.855524,3.380629,0.993126,1.258205,4.006550,-8.772787,-9.564527,8.516318,5.816072,6.559843,-3.834187,3.450162,-4.487564,9.792117,-4.288421,-5.285068,8.650660,-6.738267,7.232172,0.282056,-2.377879,-5.124607,6.603458,6.185282,-0.910828,-6.357255,9.941451,-5.108324,-1.766344,6.781311,5.557262,0.983849,4.903215,6.185643,0.930980,9.532774,0.782369,-3.834556,-4.776842,4.472108,6.565623,9.679007,1.838669,-5.448142,1.077594,-0.602937,4.176050,-3.607951,8.052096,9.175117,-3.969829,5.051848,7.051362,-2.651955,3.809848,-1.656765,-3.094858,5.086174,-8.182141,5.965369,6.831168,-7.118532,6.928641,-7.863179,4.624413,-5.774631,-0.629821,3.315804,5.463209,9.913496,1.418333,0.562093,6.165858,5.812959,-0.245010,-3.697005,-9.593249,0.542741,9.669234,-3.800445,8.752141,-8.819327,-6.522938,7.092156,-8.874140,-9.594788,5.447376,4.957288,7.713108,-6.850133,9.547044,5.993982,9.973216,-4.947698,-5.202427,9.479419,8.734492,-1.922241,5.946229,4.616082,-6.255030,0.627208,-6.743301,4.560700,3.455192,0.946921,-8.833559,-2.198792,3.586416,-0.971978,-8.593670,-2.620892,4.681317,8.728033,-2.752930,-8.784573,1.004589,-1.097118,-9.977043,5.714087,2.293554,4.036136,6.670096,-3.512939,-5.435370,1.203517,-4.897024,-0.822648,-5.208495,4.420823,-5.228934,9.699968,7.386784,2.517048,9.070368,-2.383305,-5.009591,-8.519804,6.071906,-0.443649,5.298776,-0.873761,8.482406,-5.058150,3.234450,0.736186,6.257752,2.632883,-8.784408,1.764929,7.428512,6.757629,3.354287,4.320782,-9.376956,-3.203627,1.243954,7.116804,9.788137,-3.429052,4.943557,-1.633876,6.622078,-2.343839,7.376059,9.760277,2.322180,-6.688878,1.520362,8.908000,9.518541,9.672171,6.575780,-1.814860,-6.345012,3.299673,-0.959664,9.174445,7.767266,9.379207,1.170163,8.162826,5.583224,3.167009,1.979346,-3.038435,-3.382051,-6.642014,3.030481,-7.460542,8.417878,7.400890,5.755625,3.660925,-8.345645,6.630027,-1.223118,-4.704710,-5.124386,-4.102112,6.074849,3.229802,-2.889886,4.698211,-4.417478,-3.448884,6.282243,1.047283,-3.707957,4.376773,3.135717,-9.101248,9.927090,-8.592504,-0.313847,0.799627,7.295450,8.462269,-2.720941,6.300987,-3.413665,3.430886,-8.304893,-4.146797,3.852762,0.043662,0.574446,-9.545808,8.379387,-7.290243,9.589896,4.763162,-4.670449,7.048307,9.937227,7.577591,-6.128495,8.107352,0.434091,2.867506,8.070940,-6.594041,-6.830396,7.834928,2.598816,3.464916,9.199069,2.186784,0.275933,2.050072,-0.793761,-9.985357,4.211871,4.856406,9.904594,7.046375,8.597954,9.540145,7.450469,3.893607,9.929544,0.709753,6.367702,-2.906395,0.608631,8.752201,8.650318,-1.655562,4.672978,3.491556,-6.788108,9.368472,-0.132917,9.983867,3.845477,-1.575025,1.561927,-2.338641,8.376531,1.700414,-2.199562,-0.543995,-6.265332,-5.884088,-7.091125,6.284355,-4.391520,-4.027242,8.738991,-6.452500,7.220093,4.804714,3.451349,5.480721,0.354389,6.541398,-9.708440,7.428290,7.591846,-7.269051,6.656297,-4.929137,3.717312,0.073540,6.919613,2.733832,-0.054361,1.015235,-5.308635,-9.552586,-3.362492,0.524609,0.450553,6.508868,-0.419297,-6.647658,5.669813,-2.274297,-1.166036,-0.465590,7.729226,-1.940407,-4.304695,8.663439,-1.232052,-1.804769,-6.317061,8.714698,-9.077242,-1.668493,1.260410,6.370966,4.050670,0.132879,2.550919,0.589267,8.778392,-1.708615,-6.663481,3.863794,-8.211157,7.661612,2.270135,-2.358264,3.792198,4.895632,-4.309150,-9.684066,-9.496643,2.153673,5.941671,-1.958277,-0.590279,-8.205456,8.813533,4.166676,-3.593524,-7.936691,-4.071530,8.143766,6.955617,-5.869223,4.761903,2.337360,-3.957512,5.608679,-7.594502,-6.233345,7.025804,1.735261,-0.679505,-7.880397,3.742724,1.246393,6.434306,1.313290,1.481072,-6.314459,-8.812062,-8.584937,-9.881252,-6.971773,8.845881,2.572405,-1.187125,8.888237,-1.719143,2.709527,-9.495981,1.566010,9.910043,-0.978251,-1.087508,-5.311763,1.739799,1.920585,-0.640184,-7.101297,-9.840236,-4.523995,-5.462252,-4.691062,5.635205,-6.736752,-0.103680,-8.593865,-3.222938,-2.680257,0.567667,-7.274083,0.288919,1.372503,6.113776,1.623479,8.920876,-2.292567,-7.184742,-4.697658,-8.352607,2.786618,-7.792931,9.215986,4.736673,6.505600,-0.605692,2.016921,5.462896,3.807322,-6.729609,4.397018,8.488663,8.881618,-3.148252,8.022615,-6.801000,4.395426,-5.074918,2.449155,-1.797541,8.293517,-8.780705,-0.829921,1.885333,7.744762,-3.796721,4.269119,-9.197788,2.317103,2.217928,-1.241007,-3.019043,-1.202340,-5.139656,-1.956068,5.977884,3.079571,-3.589617,5.466292,0.216332,6.345793,1.210309,4.077001,5.924929,-9.841845,-6.536853,-8.581339,-2.926215,0.248562,-1.575513,1.172138,7.304220,-9.718667,-3.886680,8.216366,-9.437305,-4.400273,-4.071709,3.505912,-1.336484,-0.008474,-6.438275,0.981089,7.065306,6.755384,-2.198294,4.976993,4.811105,0.681035,-6.141460,-7.692292,7.532407,-9.899110,-2.066356,9.099012,-3.823218,6.342661,-1.192793,1.074187,-8.961992,-6.172603,-3.433237,-6.865659,1.952750,6.127567,3.901881,-4.222576,-4.125935,-0.194478,-1.867099,-9.865180,1.602363,-7.307789,-7.884516,3.280655,-7.436927,-5.080460,-0.092534,8.585607,-5.744899,3.091619,-2.678333,8.515842,-9.640795,2.407473,6.335282,8.584581,-7.651385,2.863248,-3.802458,0.542952,-2.731823,7.928160,-7.825003,-8.741349,3.704764,-1.970451,7.900414,9.572029,6.507432,-2.773258,-5.620234,5.926986,4.212234,5.603472,-0.523887,4.101762,-0.362424,9.605630,9.336681,-6.031006,1.277250,-2.923576,-1.297873,-3.661289,-3.524213], dtype = "float32")#candidate|1149|(1500,)|const|float32
call_1148 = relay.TupleGetItem(func_1104_call(relay.reshape(const_1149.astype('float32'), [10, 10, 15]), relay.reshape(const_1149.astype('float32'), [10, 10, 15]), ), 2)
call_1150 = relay.TupleGetItem(func_1108_call(relay.reshape(const_1149.astype('float32'), [10, 10, 15]), relay.reshape(const_1149.astype('float32'), [10, 10, 15]), ), 2)
uop_1156 = relay.acosh(var_1133.astype('float64')) # shape=(8, 12, 4)
output = relay.Tuple([bop_1140,call_1148,const_1149,uop_1156,])
output2 = relay.Tuple([bop_1140,call_1150,const_1149,uop_1156,])
func_1159 = relay.Function([var_1132,var_1133,], output)
mod['func_1159'] = func_1159
mod = relay.transform.InferType()(mod)
mutated_mod['func_1159'] = func_1159
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1159_call = mutated_mod.get_global_var('func_1159')
var_1161 = relay.var("var_1161", dtype = "int16", shape = (8, 12, 4))#candidate|1161|(8, 12, 4)|var|int16
var_1162 = relay.var("var_1162", dtype = "int16", shape = (8, 12, 4))#candidate|1162|(8, 12, 4)|var|int16
call_1160 = func_1159_call(var_1161,var_1162,)
output = call_1160
func_1163 = relay.Function([var_1161,var_1162,], output)
mutated_mod['func_1163'] = func_1163
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1196 = relay.var("var_1196", dtype = "uint16", shape = (13, 15))#candidate|1196|(13, 15)|var|uint16
var_1197 = relay.var("var_1197", dtype = "uint16", shape = (13, 15))#candidate|1197|(13, 15)|var|uint16
bop_1198 = relay.greater_equal(var_1196.astype('bool'), relay.reshape(var_1197.astype('bool'), relay.shape_of(var_1196))) # shape=(13, 15)
func_102_call = mod.get_global_var('func_102')
func_105_call = mutated_mod.get_global_var('func_105')
var_1202 = relay.var("var_1202", dtype = "float32", shape = (4,))#candidate|1202|(4,)|var|float32
var_1203 = relay.var("var_1203", dtype = "float32", shape = (36,))#candidate|1203|(36,)|var|float32
call_1201 = relay.TupleGetItem(func_102_call(relay.reshape(var_1202.astype('float32'), [1, 4]), relay.reshape(var_1203.astype('float32'), [9, 4]), ), 1)
call_1204 = relay.TupleGetItem(func_105_call(relay.reshape(var_1202.astype('float32'), [1, 4]), relay.reshape(var_1203.astype('float32'), [9, 4]), ), 1)
bop_1209 = relay.divide(var_1197.astype('float64'), relay.reshape(var_1196.astype('float64'), relay.shape_of(var_1197))) # shape=(13, 15)
output = relay.Tuple([bop_1198,call_1201,var_1202,var_1203,bop_1209,])
output2 = relay.Tuple([bop_1198,call_1204,var_1202,var_1203,bop_1209,])
func_1216 = relay.Function([var_1196,var_1197,var_1202,var_1203,], output)
mod['func_1216'] = func_1216
mod = relay.transform.InferType()(mod)
mutated_mod['func_1216'] = func_1216
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1216_call = mutated_mod.get_global_var('func_1216')
var_1218 = relay.var("var_1218", dtype = "uint16", shape = (13, 15))#candidate|1218|(13, 15)|var|uint16
var_1219 = relay.var("var_1219", dtype = "uint16", shape = (13, 15))#candidate|1219|(13, 15)|var|uint16
var_1220 = relay.var("var_1220", dtype = "float32", shape = (4,))#candidate|1220|(4,)|var|float32
var_1221 = relay.var("var_1221", dtype = "float32", shape = (36,))#candidate|1221|(36,)|var|float32
call_1217 = func_1216_call(var_1218,var_1219,var_1220,var_1221,)
output = call_1217
func_1222 = relay.Function([var_1218,var_1219,var_1220,var_1221,], output)
mutated_mod['func_1222'] = func_1222
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1234 = relay.var("var_1234", dtype = "float32", shape = (16, 16))#candidate|1234|(16, 16)|var|float32
uop_1235 = relay.atanh(var_1234.astype('float32')) # shape=(16, 16)
func_270_call = mod.get_global_var('func_270')
func_275_call = mutated_mod.get_global_var('func_275')
const_1240 = relay.const([9.916797,1.492650,-6.534755,-8.272647,-3.229531,4.708872,-6.575730,8.245073,5.352169,-2.760100,-8.823057,-8.389239,-6.410880,9.707318,-0.557747,-0.018466,-7.918499,9.261001,3.270976,-9.904537,-3.956090,-3.677590,8.010364,-5.076082], dtype = "float32")#candidate|1240|(24,)|const|float32
const_1241 = relay.const([-0.192332,-9.109350,-0.460922,-8.348287,-1.861099,-5.440071,-9.112144,-4.751939,3.340986,-2.922144,-2.903305,-7.563761,-5.292140,-3.788401,8.291612,1.318472,-3.251105,1.069636,-9.544332,6.688628,-2.882068,7.168771,-1.771389,-4.968415,-1.315142,6.007788,7.572030,-2.141841,-9.551125,7.196397,6.172708,3.788744,-7.484173,-8.799618,-0.556761,-3.813053,3.670288,7.281221,5.393916,6.229005,-5.409507,-4.981518,-2.961814,-4.598227,-3.169184,8.614509,-4.195071,3.638832,6.149402,9.995428,0.452203,-1.080534,9.964971,-6.740325,3.002384,-4.523303,-9.331458,9.684975,0.789039,6.726195,9.031854,-1.095927,2.146812,8.157481,6.098955,-9.347854,-9.139656,-8.917200,4.086991,-1.558842,-9.914771,-8.849159,0.659672,-3.983295,4.003230,-5.391861,-0.013596,6.683191,-3.075551,-3.220761,5.110651,6.031751,8.425014,-8.959960,-8.429157,7.986112,3.277300,-9.207075,0.230668,-8.467312,-3.412465,2.486207,4.278524,3.013417,1.514054,3.547106,-8.933301,-3.190494,-0.797572,-0.230152,3.778594,-5.611711,-1.192865,-9.425510,5.429629,-7.149244,7.674608,-2.879379,-2.632187,8.574099,5.949433,8.900942,1.880691,7.049038,-6.221184,-2.168428,1.241110,-2.312244,2.121427,7.665919,0.390454,3.504169,3.422432,8.690387,9.766972,-7.733736,-9.373351,-1.498051,9.559219,8.383008,-4.817489,-3.581337,-2.891898,5.385586,2.215520,-8.297125,-2.185308,7.454192,3.262040,-1.925896,4.997266,8.039117,-8.741804,0.938163,5.207043,2.366170,5.244404,1.899428,-4.979718,6.838415,1.414320,4.411185,2.159805,-1.279428,8.000681,3.791273,7.332289,-0.756017,3.681280,1.743269,6.099231,-3.283265,-1.700436,-8.872992,-1.939408,-5.241725,-0.667708,5.417433,5.168183,-9.444981,-7.814536,9.596004,0.774257,5.471190,-3.917959,2.235481,3.476677,-3.782711,-7.688994,6.282753,-4.439265,-2.681909,5.165741,7.568565,4.177080,0.712000,3.455868,3.002474,3.558714,7.580617,-2.214365,6.465594], dtype = "float32")#candidate|1241|(192,)|const|float32
call_1239 = relay.TupleGetItem(func_270_call(relay.reshape(const_1240.astype('float32'), [8, 3]), relay.reshape(const_1241.astype('float32'), [192,]), relay.reshape(const_1240.astype('float32'), [8, 3]), relay.reshape(const_1240.astype('int32'), [8, 3]), ), 1)
call_1242 = relay.TupleGetItem(func_275_call(relay.reshape(const_1240.astype('float32'), [8, 3]), relay.reshape(const_1241.astype('float32'), [192,]), relay.reshape(const_1240.astype('float32'), [8, 3]), relay.reshape(const_1240.astype('int32'), [8, 3]), ), 1)
func_661_call = mod.get_global_var('func_661')
func_666_call = mutated_mod.get_global_var('func_666')
var_1246 = relay.var("var_1246", dtype = "float32", shape = (132,))#candidate|1246|(132,)|var|float32
call_1245 = relay.TupleGetItem(func_661_call(relay.reshape(var_1246.astype('float32'), [12, 11]), relay.reshape(var_1246.astype('float32'), [12, 11]), relay.reshape(var_1246.astype('float64'), [12, 11]), ), 0)
call_1247 = relay.TupleGetItem(func_666_call(relay.reshape(var_1246.astype('float32'), [12, 11]), relay.reshape(var_1246.astype('float32'), [12, 11]), relay.reshape(var_1246.astype('float64'), [12, 11]), ), 0)
output = relay.Tuple([uop_1235,call_1239,const_1240,const_1241,call_1245,var_1246,])
output2 = relay.Tuple([uop_1235,call_1242,const_1240,const_1241,call_1247,var_1246,])
func_1249 = relay.Function([var_1234,var_1246,], output)
mod['func_1249'] = func_1249
mod = relay.transform.InferType()(mod)
mutated_mod['func_1249'] = func_1249
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1249_call = mutated_mod.get_global_var('func_1249')
var_1251 = relay.var("var_1251", dtype = "float32", shape = (16, 16))#candidate|1251|(16, 16)|var|float32
var_1252 = relay.var("var_1252", dtype = "float32", shape = (132,))#candidate|1252|(132,)|var|float32
call_1250 = func_1249_call(var_1251,var_1252,)
output = call_1250
func_1253 = relay.Function([var_1251,var_1252,], output)
mutated_mod['func_1253'] = func_1253
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1271 = relay.var("var_1271", dtype = "float64", shape = (13, 3, 3))#candidate|1271|(13, 3, 3)|var|float64
uop_1272 = relay.exp(var_1271.astype('float64')) # shape=(13, 3, 3)
output = relay.Tuple([uop_1272,])
output2 = relay.Tuple([uop_1272,])
F = relay.Function([var_1271,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1271,], output2)
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
input_1271= np.array([[[-4.660921,4.518790,-3.377620],[2.177389,-7.688503,0.386249],[7.184169,5.441267,4.122552]],[[0.293770,-2.503673,-5.216271],[-5.017926,8.293598,8.296867],[7.011474,-0.454001,9.622375]],[[-8.991548,-7.872295,8.265409],[0.459188,9.925938,-4.107998],[9.056556,3.017291,-4.393177]],[[-9.926358,-6.133342,6.878917],[3.027492,2.945253,2.889864],[-3.923452,3.148640,4.053923]],[[-3.281044,4.268522,8.393757],[4.317231,-6.545412,-9.622811],[-0.650607,2.103208,-0.709449]],[[-2.738294,4.433365,-7.210834],[-3.747457,0.757335,1.962361],[-4.113889,8.637397,4.414863]],[[5.215819,-3.305194,-7.814037],[2.834360,8.741812,9.880314],[9.689855,-9.747553,-3.070937]],[[-8.515699,7.189525,-1.835921],[-2.807785,6.918311,2.771066],[-6.456457,8.175319,-2.672536]],[[3.473860,-6.989544,-8.552890],[1.703420,0.857279,-0.968171],[-4.308552,7.244585,-4.930837]],[[-2.441128,-1.689011,3.222405],[0.837790,7.458828,-4.919230],[-2.555721,9.036234,-4.395704]],[[-9.927284,7.402931,2.264619],[8.541344,-7.869778,-5.020863],[8.233756,7.889647,-2.745319]],[[5.331272,-6.619993,-1.574256],[-3.506258,1.232591,0.464570],[9.047953,9.677569,5.400287]],[[-6.878360,-0.644511,3.340225],[-2.328648,5.767031,2.149514],[-9.903288,7.186458,5.563940]]], dtype='float64')
module1.set_input('var_1271', input_1271)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1271, )
res3 = intrp3.evaluate()(input_1271, )
res4 = intrp4.evaluate()(input_1271, )
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
module5.set_input('var_1271', input_1271)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1271, )
res7 = intrp7.evaluate()(input_1271, )
res8 = intrp8.evaluate()(input_1271, )
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
module9.set_input('var_1271', input_1271)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1271, )
res11 = intrp11.evaluate()(input_1271, )
res12 = intrp12.evaluate()(input_1271, )
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
module13.set_input('var_1271', input_1271)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1271, )
res15 = intrp15.evaluate()(input_1271, )
res16 = intrp16.evaluate()(input_1271, )
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
module17.set_input('var_1271', input_1271)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1271, )
res19 = intrp19.evaluate()(input_1271, )
res20 = intrp20.evaluate()(input_1271, )
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
module21.set_input('var_1271', input_1271)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1271, )
res23 = intrp23.evaluate()(input_1271, )
res24 = intrp24.evaluate()(input_1271, )
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