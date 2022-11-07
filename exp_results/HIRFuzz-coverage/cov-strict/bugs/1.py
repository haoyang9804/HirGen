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
var_5 = relay.var("var_5", dtype = "float64", shape = (11, 2, 14))#candidate|5|(11, 2, 14)|var|float64
uop_6 = relay.sqrt(var_5.astype('float64')) # shape=(11, 2, 14)
bop_8 = relay.bitwise_xor(uop_6.astype('uint64'), relay.reshape(var_5.astype('uint64'), relay.shape_of(uop_6))) # shape=(11, 2, 14)
bop_14 = relay.bitwise_and(bop_8.astype('uint16'), relay.reshape(var_5.astype('uint16'), relay.shape_of(bop_8))) # shape=(11, 2, 14)
var_17 = relay.var("var_17", dtype = "float64", shape = (11, 2, 14))#candidate|17|(11, 2, 14)|var|float64
bop_18 = relay.equal(uop_6.astype('bool'), relay.reshape(var_17.astype('bool'), relay.shape_of(uop_6))) # shape=(11, 2, 14)
bop_21 = relay.add(var_17.astype('uint8'), relay.reshape(bop_18.astype('uint8'), relay.shape_of(var_17))) # shape=(11, 2, 14)
uop_24 = relay.asinh(bop_8.astype('float32')) # shape=(11, 2, 14)
bop_26 = relay.right_shift(uop_24.astype('int64'), relay.reshape(uop_6.astype('int64'), relay.shape_of(uop_24))) # shape=(11, 2, 14)
var_29 = relay.var("var_29", dtype = "uint8", shape = (11, 2, 14))#candidate|29|(11, 2, 14)|var|uint8
bop_30 = relay.subtract(bop_21.astype('float32'), relay.reshape(var_29.astype('float32'), relay.shape_of(bop_21))) # shape=(11, 2, 14)
var_33 = relay.var("var_33", dtype = "int64", shape = (11, 2, 14))#candidate|33|(11, 2, 14)|var|int64
bop_34 = relay.subtract(bop_26.astype('int64'), relay.reshape(var_33.astype('int64'), relay.shape_of(bop_26))) # shape=(11, 2, 14)
bop_37 = relay.subtract(uop_24.astype('uint64'), relay.reshape(bop_18.astype('uint64'), relay.shape_of(uop_24))) # shape=(11, 2, 14)
uop_40 = relay.erf(bop_21.astype('float64')) # shape=(11, 2, 14)
bop_42 = relay.logical_or(bop_14.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(bop_14))) # shape=(11, 2, 14)
uop_45 = relay.cos(bop_37.astype('float32')) # shape=(11, 2, 14)
bop_47 = relay.minimum(bop_26.astype('uint16'), relay.reshape(bop_37.astype('uint16'), relay.shape_of(bop_26))) # shape=(11, 2, 14)
uop_52 = relay.cos(bop_30.astype('float64')) # shape=(11, 2, 14)
bop_56 = relay.greater_equal(uop_45.astype('bool'), relay.reshape(bop_47.astype('bool'), relay.shape_of(uop_45))) # shape=(11, 2, 14)
uop_59 = relay.tan(uop_52.astype('float32')) # shape=(11, 2, 14)
uop_61 = relay.log(bop_37.astype('float32')) # shape=(11, 2, 14)
var_63 = relay.var("var_63", dtype = "bool", shape = (11, 2, 14))#candidate|63|(11, 2, 14)|var|bool
bop_64 = relay.greater_equal(bop_56.astype('bool'), relay.reshape(var_63.astype('bool'), relay.shape_of(bop_56))) # shape=(11, 2, 14)
bop_67 = relay.equal(uop_45.astype('bool'), relay.reshape(bop_34.astype('bool'), relay.shape_of(uop_45))) # shape=(11, 2, 14)
bop_70 = relay.add(bop_14.astype('uint32'), relay.reshape(uop_40.astype('uint32'), relay.shape_of(bop_14))) # shape=(11, 2, 14)
uop_75 = relay.cosh(bop_64.astype('float32')) # shape=(11, 2, 14)
bop_77 = relay.not_equal(uop_75.astype('bool'), relay.reshape(uop_52.astype('bool'), relay.shape_of(uop_75))) # shape=(11, 2, 14)
output = relay.Tuple([bop_42,uop_59,uop_61,bop_67,bop_70,bop_77,])
output2 = relay.Tuple([bop_42,uop_59,uop_61,bop_67,bop_70,bop_77,])
func_80 = relay.Function([var_5,var_17,var_29,var_33,var_63,], output)
mod['func_80'] = func_80
mod = relay.transform.InferType()(mod)
var_81 = relay.var("var_81", dtype = "float64", shape = (11, 2, 14))#candidate|81|(11, 2, 14)|var|float64
var_82 = relay.var("var_82", dtype = "float64", shape = (11, 2, 14))#candidate|82|(11, 2, 14)|var|float64
var_83 = relay.var("var_83", dtype = "uint8", shape = (11, 2, 14))#candidate|83|(11, 2, 14)|var|uint8
var_84 = relay.var("var_84", dtype = "int64", shape = (11, 2, 14))#candidate|84|(11, 2, 14)|var|int64
var_85 = relay.var("var_85", dtype = "bool", shape = (11, 2, 14))#candidate|85|(11, 2, 14)|var|bool
output = func_80(var_81,var_82,var_83,var_84,var_85,)
func_86 = relay.Function([var_81,var_82,var_83,var_84,var_85,], output)
mutated_mod['func_86'] = func_86
mutated_mod = relay.transform.InferType()(mutated_mod)
var_96 = relay.var("var_96", dtype = "float64", shape = (3, 13, 16))#candidate|96|(3, 13, 16)|var|float64
uop_97 = relay.tan(var_96.astype('float64')) # shape=(3, 13, 16)
var_102 = relay.var("var_102", dtype = "float64", shape = (3, 13, 16))#candidate|102|(3, 13, 16)|var|float64
bop_103 = relay.minimum(uop_97.astype('int8'), relay.reshape(var_102.astype('int8'), relay.shape_of(uop_97))) # shape=(3, 13, 16)
func_80_call = mod.get_global_var('func_80')
func_86_call = mutated_mod.get_global_var('func_86')
const_109 = relay.const([-9.938919,-3.458701,8.497252,0.164169,8.333897,0.535558,-9.647393,-2.857354,-9.434975,-2.726861,6.321434,4.124607,2.551468,1.324227,5.184952,5.998011,-0.428692,-5.115948,-1.865149,-5.034542,-3.519963,7.970099,-5.345267,2.159922,-3.610845,3.783942,-8.820141,8.432112,0.621876,-7.928964,2.341871,9.239771,-6.944143,2.822344,6.463359,-2.326339,-4.715769,-5.796503,2.027243,1.791914,8.195936,5.456800,-4.287364,7.785678,3.826815,-6.776168,8.659346,-8.213901,0.051404,-9.870518,2.328502,8.180816,-3.286067,-9.462004,2.544656,4.621757,-1.279311,2.806036,3.600393,1.050378,-8.558669,2.918347,2.129914,7.315746,-7.417783,8.558999,-6.757535,7.311318,-7.220758,-3.644749,8.096112,-1.166101,-0.852175,-6.873597,-8.057918,8.521442,4.369971,-5.445366,-9.907208,-1.601700,-3.556476,3.176774,-9.635569,1.904381,-7.123567,-6.562049,-8.676600,-3.831462,-7.147017,-6.665662,-0.381676,-1.828128,6.493779,-2.108483,-9.713395,-2.271074,-5.132740,-0.897890,-6.224499,3.253799,7.174266,-8.599153,-3.980093,9.449187,-7.632363,6.238318,5.866066,-8.558792,-5.613547,-4.578979,7.928592,3.497105,-7.996037,-8.946761,-2.495226,-1.946306,-7.608115,3.365185,4.163594,0.846381,-3.781597,-0.101340,-9.375485,-8.488935,3.031745,-7.772948,8.479876,-3.776713,8.573551,-7.770563,-4.617609,2.186555,-5.882312,5.505459,5.406166,8.128125,7.407937,6.033121,-8.794618,-7.235356,8.776450,1.096694,6.357905,-3.431819,-5.323076,-6.669157,9.763842,2.255085,-7.534857,6.337235,-4.950083,1.728364,-3.378093,6.950810,1.141856,-4.672964,-8.540062,7.960375,3.226376,-9.979122,3.250047,-6.363834,3.923868,-4.266567,-2.715003,4.459645,-2.307135,-2.144374,-6.743134,7.799263,-1.659329,-2.569500,5.450104,-5.231587,4.342731,3.261624,5.296583,-1.545813,0.465503,-8.753743,-0.489838,-8.729446,-1.498570,-7.848499,5.611248,-6.191129,6.186224,-6.457175,4.795362,-7.805958,-1.857019,3.355382,5.690248,-7.882626,-1.705124,7.946493,-1.194278,9.349971,-4.475374,2.127221,8.576368,9.354693,-1.588548,-9.983603,-1.113020,4.776571,-5.972611,-6.457963,-1.660985,-7.143726,2.865972,8.632340,-8.518255,6.438362,-2.461206,-9.116999,-7.570628,9.331708,4.787400,-8.740061,-6.623822,-5.938288,-5.088841,-1.860867,7.731908,4.001925,-6.204557,9.750897,0.184683,-0.002056,3.935283,-5.404680,-3.758787,2.149608,6.576213,-5.703036,3.416426,-0.356851,2.727680,-9.226552,4.246608,-3.969744,-0.257495,-5.080590,-4.965921,-4.240808,6.742728,5.412633,-3.202578,-1.376094,3.435175,2.478729,6.563517,-7.166062,-6.544666,-0.533672,2.315444,-7.667787,-5.507825,3.665092,2.567913,-6.640125,-9.687792,-2.831865,-0.924806,-3.128336,4.497945,-4.012231,9.697084,-5.477436,0.810284,-8.264980,1.910844,6.588258,5.211695,-0.641684,-9.192340,-0.254971,-8.861953,7.175645,5.591356,-8.831007,1.612473,8.499915,-8.513076,-4.315552,-4.466263,-4.074837,9.537358,5.269066,-9.431807,4.229451,0.106645,-7.745028,0.675537,-3.880163,6.909201,6.615394,-9.844015,-3.689115,-7.190933,-3.001447,4.448781,-2.039267,-2.867640,3.812836,4.967959,7.917699], dtype = "float64")#candidate|109|(308,)|const|float64
call_108 = relay.TupleGetItem(func_80_call(relay.reshape(const_109.astype('float64'), [11, 2, 14]), relay.reshape(const_109.astype('float64'), [11, 2, 14]), relay.reshape(const_109.astype('uint8'), [11, 2, 14]), relay.reshape(const_109.astype('int64'), [11, 2, 14]), relay.reshape(const_109.astype('bool'), [11, 2, 14]), ), 4)
call_110 = relay.TupleGetItem(func_86_call(relay.reshape(const_109.astype('float64'), [11, 2, 14]), relay.reshape(const_109.astype('float64'), [11, 2, 14]), relay.reshape(const_109.astype('uint8'), [11, 2, 14]), relay.reshape(const_109.astype('int64'), [11, 2, 14]), relay.reshape(const_109.astype('bool'), [11, 2, 14]), ), 4)
bop_111 = relay.multiply(uop_97.astype('int16'), relay.reshape(var_102.astype('int16'), relay.shape_of(uop_97))) # shape=(3, 13, 16)
uop_117 = relay.log10(bop_111.astype('float64')) # shape=(3, 13, 16)
var_119 = relay.var("var_119", dtype = "float64", shape = (3, 13, 16))#candidate|119|(3, 13, 16)|var|float64
bop_120 = relay.greater(uop_117.astype('bool'), relay.reshape(var_119.astype('bool'), relay.shape_of(uop_117))) # shape=(3, 13, 16)
bop_123 = relay.maximum(bop_120.astype('uint8'), relay.reshape(var_96.astype('uint8'), relay.shape_of(bop_120))) # shape=(3, 13, 16)
uop_128 = relay.cosh(bop_123.astype('float64')) # shape=(3, 13, 16)
bop_130 = relay.floor_divide(uop_117.astype('float64'), relay.reshape(uop_97.astype('float64'), relay.shape_of(uop_117))) # shape=(3, 13, 16)
uop_133 = relay.log10(bop_123.astype('float32')) # shape=(3, 13, 16)
bop_137 = relay.greater_equal(uop_133.astype('bool'), relay.reshape(bop_111.astype('bool'), relay.shape_of(uop_133))) # shape=(3, 13, 16)
output = relay.Tuple([bop_103,call_108,const_109,uop_128,bop_130,bop_137,])
output2 = relay.Tuple([bop_103,call_110,const_109,uop_128,bop_130,bop_137,])
func_143 = relay.Function([var_96,var_102,var_119,], output)
mod['func_143'] = func_143
mod = relay.transform.InferType()(mod)
mutated_mod['func_143'] = func_143
mutated_mod = relay.transform.InferType()(mutated_mod)
func_143_call = mutated_mod.get_global_var('func_143')
var_145 = relay.var("var_145", dtype = "float64", shape = (3, 13, 16))#candidate|145|(3, 13, 16)|var|float64
var_146 = relay.var("var_146", dtype = "float64", shape = (3, 13, 16))#candidate|146|(3, 13, 16)|var|float64
var_147 = relay.var("var_147", dtype = "float64", shape = (3, 13, 16))#candidate|147|(3, 13, 16)|var|float64
call_144 = func_143_call(var_145,var_146,var_147,)
output = call_144
func_148 = relay.Function([var_145,var_146,var_147,], output)
mutated_mod['func_148'] = func_148
mutated_mod = relay.transform.InferType()(mutated_mod)
const_154 = relay.const([-9.554640,-0.417769,-9.709899,-4.956550,-4.296769,-6.718803,9.973050,-2.561166,-9.881760,1.931252,4.449987], dtype = "float64")#candidate|154|(11,)|const|float64
var_155 = relay.var("var_155", dtype = "float64", shape = (11,))#candidate|155|(11,)|var|float64
bop_156 = relay.greater(const_154.astype('bool'), relay.reshape(var_155.astype('bool'), relay.shape_of(const_154))) # shape=(11,)
bop_159 = relay.minimum(const_154.astype('float32'), relay.reshape(bop_156.astype('float32'), relay.shape_of(const_154))) # shape=(11,)
var_162 = relay.var("var_162", dtype = "float32", shape = (11,))#candidate|162|(11,)|var|float32
bop_163 = relay.subtract(bop_159.astype('int16'), relay.reshape(var_162.astype('int16'), relay.shape_of(bop_159))) # shape=(11,)
bop_166 = relay.greater(bop_156.astype('bool'), relay.reshape(bop_159.astype('bool'), relay.shape_of(bop_156))) # shape=(11,)
bop_169 = relay.add(var_155.astype('uint16'), relay.reshape(bop_156.astype('uint16'), relay.shape_of(var_155))) # shape=(11,)
bop_173 = relay.greater_equal(bop_159.astype('bool'), relay.reshape(var_162.astype('bool'), relay.shape_of(bop_159))) # shape=(11,)
uop_176 = relay.acosh(bop_163.astype('float64')) # shape=(11,)
func_143_call = mod.get_global_var('func_143')
func_148_call = mutated_mod.get_global_var('func_148')
var_179 = relay.var("var_179", dtype = "float64", shape = (624,))#candidate|179|(624,)|var|float64
call_178 = relay.TupleGetItem(func_143_call(relay.reshape(var_179.astype('float64'), [3, 13, 16]), relay.reshape(var_179.astype('float64'), [3, 13, 16]), relay.reshape(var_179.astype('float64'), [3, 13, 16]), ), 1)
call_180 = relay.TupleGetItem(func_148_call(relay.reshape(var_179.astype('float64'), [3, 13, 16]), relay.reshape(var_179.astype('float64'), [3, 13, 16]), relay.reshape(var_179.astype('float64'), [3, 13, 16]), ), 1)
bop_181 = relay.right_shift(uop_176.astype('uint32'), relay.reshape(bop_159.astype('uint32'), relay.shape_of(uop_176))) # shape=(11,)
uop_184 = relay.sqrt(bop_173.astype('float64')) # shape=(11,)
bop_186 = relay.maximum(uop_184.astype('float32'), relay.reshape(var_162.astype('float32'), relay.shape_of(uop_184))) # shape=(11,)
var_189 = relay.var("var_189", dtype = "float64", shape = (11,))#candidate|189|(11,)|var|float64
bop_190 = relay.minimum(uop_176.astype('int32'), relay.reshape(var_189.astype('int32'), relay.shape_of(uop_176))) # shape=(11,)
output = relay.Tuple([bop_166,bop_169,call_178,var_179,bop_181,bop_186,bop_190,])
output2 = relay.Tuple([bop_166,bop_169,call_180,var_179,bop_181,bop_186,bop_190,])
func_193 = relay.Function([var_155,var_162,var_179,var_189,], output)
mod['func_193'] = func_193
mod = relay.transform.InferType()(mod)
mutated_mod['func_193'] = func_193
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mutated_mod.get_global_var('func_193')
var_195 = relay.var("var_195", dtype = "float64", shape = (11,))#candidate|195|(11,)|var|float64
var_196 = relay.var("var_196", dtype = "float32", shape = (11,))#candidate|196|(11,)|var|float32
var_197 = relay.var("var_197", dtype = "float64", shape = (624,))#candidate|197|(624,)|var|float64
var_198 = relay.var("var_198", dtype = "float64", shape = (11,))#candidate|198|(11,)|var|float64
call_194 = func_193_call(var_195,var_196,var_197,var_198,)
output = call_194
func_199 = relay.Function([var_195,var_196,var_197,var_198,], output)
mutated_mod['func_199'] = func_199
mutated_mod = relay.transform.InferType()(mutated_mod)
var_206 = relay.var("var_206", dtype = "int32", shape = (1,))#candidate|206|(1,)|var|int32
var_207 = relay.var("var_207", dtype = "int32", shape = (15,))#candidate|207|(15,)|var|int32
bop_208 = relay.bitwise_or(var_206.astype('int32'), var_207.astype('int32')) # shape=(15,)
uop_211 = relay.atan(bop_208.astype('float64')) # shape=(15,)
var_213 = relay.var("var_213", dtype = "float64", shape = (15,))#candidate|213|(15,)|var|float64
bop_214 = relay.subtract(uop_211.astype('int16'), relay.reshape(var_213.astype('int16'), relay.shape_of(uop_211))) # shape=(15,)
bop_221 = relay.floor_divide(uop_211.astype('float32'), relay.reshape(var_207.astype('float32'), relay.shape_of(uop_211))) # shape=(15,)
bop_224 = relay.logical_or(bop_214.astype('bool'), var_206.astype('bool')) # shape=(15,)
output = relay.Tuple([bop_221,bop_224,])
output2 = relay.Tuple([bop_221,bop_224,])
func_227 = relay.Function([var_206,var_207,var_213,], output)
mod['func_227'] = func_227
mod = relay.transform.InferType()(mod)
var_228 = relay.var("var_228", dtype = "int32", shape = (1,))#candidate|228|(1,)|var|int32
var_229 = relay.var("var_229", dtype = "int32", shape = (15,))#candidate|229|(15,)|var|int32
var_230 = relay.var("var_230", dtype = "float64", shape = (15,))#candidate|230|(15,)|var|float64
output = func_227(var_228,var_229,var_230,)
func_231 = relay.Function([var_228,var_229,var_230,], output)
mutated_mod['func_231'] = func_231
mutated_mod = relay.transform.InferType()(mutated_mod)
const_241 = relay.const([False], dtype = "bool")#candidate|241|(1,)|const|bool
const_242 = relay.const([True,True], dtype = "bool")#candidate|242|(2,)|const|bool
bop_243 = relay.logical_or(const_241.astype('bool'), const_242.astype('bool')) # shape=(2,)
bop_246 = relay.less(const_241.astype('bool'), bop_243.astype('bool')) # shape=(2,)
bop_250 = relay.maximum(bop_246.astype('int64'), relay.reshape(const_242.astype('int64'), relay.shape_of(bop_246))) # shape=(2,)
func_143_call = mod.get_global_var('func_143')
func_148_call = mutated_mod.get_global_var('func_148')
const_257 = relay.const([3.529124,5.442411,-7.743535,2.752207,2.719672,8.827296,-7.036717,9.107638,-0.560609,-7.246276,-8.729600,9.315609,-3.455045,-9.730782,2.518619,0.708821,-3.564556,8.951637,-4.814447,5.729581,4.169994,-1.924791,-2.970535,-3.030425,4.856298,-1.691203,-0.695778,3.504154,6.509098,6.525857,-0.308668,6.755441,8.408779,3.329437,2.348668,9.153993,-5.224824,-8.733093,-6.870579,2.604644,-7.465026,5.811066,3.759944,6.370097,-7.684150,2.542050,2.968530,3.563115,1.235642,2.833071,7.676762,2.787685,-7.923477,-6.227309,-4.803955,-3.377742,0.215680,3.929681,-5.586353,-9.965284,4.660189,-6.876287,-6.026563,3.692492,-3.358301,4.520174,-8.214513,0.195163,-8.239435,2.495195,7.475652,9.025464,1.724491,0.207546,-5.152381,6.377194,2.775287,-7.868013,9.570255,8.236660,-3.548831,-2.785444,0.922971,3.606693,9.426688,-4.665402,6.861347,-5.859942,1.419110,-1.090046,5.794995,3.631885,2.139804,-3.042288,-7.098033,9.362146,-4.130567,8.476830,8.101301,-4.384452,0.361477,-8.139093,-7.448698,3.242465,-8.501683,4.343927,1.262417,4.152810,0.145840,-6.109436,4.082043,-9.128875,9.960085,-6.329231,7.295074,7.508236,3.748448,3.560348,-4.621784,4.238892,-5.476733,-6.573402,-6.747101,9.454545,-9.960783,2.621156,1.857857,2.584002,9.345806,9.393594,-5.767873,8.140249,-8.851371,-0.267509,9.298893,-9.115963,-4.897102,-0.183468,-4.756264,-5.225295,2.785365,-4.095138,-6.910013,4.701568,3.172577,-1.522663,9.325403,-0.221545,-8.246086,8.729556,-6.598529,5.786009,-5.275333,2.256053,1.158907,3.078104,5.413393,7.074109,-8.878984,2.682679,-8.938269,1.208192,9.587851,9.739861,-4.502403,-8.319836,-2.640218,-4.862810,4.477486,8.077650,7.271338,0.193971,4.154037,-2.765473,8.046220,9.602175,0.114138,9.423376,-9.703324,-6.786630,4.087335,-4.078984,4.011815,0.234215,-4.636781,6.773815,7.597027,8.882685,-2.534047,-8.811197,0.551674,9.655844,2.054579,-2.427792,5.158036,-4.575836,5.900279,6.994548,8.519637,5.433083,0.150755,-5.136239,8.814034,9.347147,3.951090,7.893453,-1.959648,-6.560436,9.013494,1.025677,0.725274,-4.743170,-9.039717,7.550692,-0.720921,-7.900585,-7.029592,-4.563198,-5.106159,3.300846,-6.651983,-9.175366,6.891556,-0.877711,3.703039,8.364588,4.555183,6.376271,3.773830,-7.541724,8.718437,-2.367184,0.441566,7.487667,-8.433751,9.089136,9.021495,9.216057,-9.741087,-4.818016,-2.200909,0.505560,7.191141,9.885380,8.991838,-2.597782,-4.921538,-1.331529,2.931149,5.122683,-0.124506,-4.299498,5.810130,-4.182528,5.692325,7.500599,9.556239,-2.402406,-7.935287,6.627631,9.774071,-3.619342,-3.882129,6.165297,9.565097,-7.522540,8.846019,4.626960,-2.535658,7.994581,-2.924356,9.209280,5.722706,-1.654195,-9.374102,7.755272,4.008244,1.054944,4.720852,-5.593999,-5.165480,-0.090585,7.891339,-8.753166,-8.473502,-5.103472,-0.281943,-1.103050,9.363819,7.088995,-1.112010,9.735443,5.723801,-0.866491,4.800045,0.194063,6.261425,5.865999,8.747101,-0.038828,2.393074,5.814424,2.745092,5.050561,-9.733117,-7.019877,-6.994387,0.699120,-3.235313,-9.071712,-4.060377,8.479433,5.655590,-8.492311,-8.378880,-5.602935,5.518248,-7.857610,0.199388,9.805194,-2.109678,5.001839,6.091889,-4.108000,-5.356647,5.865746,-1.184104,-4.354788,-1.382303,9.556946,1.072873,-5.500531,0.899167,5.063783,3.763131,-0.870471,-8.569112,-7.741094,-0.985289,-8.634396,5.443803,2.504779,-8.517669,1.384883,4.727627,-1.257110,7.951023,9.695240,0.004933,8.432731,-6.767977,-2.254759,0.819149,-6.850197,-4.787587,-0.988697,6.691332,2.106617,0.666540,-0.100437,1.852588,-2.038610,8.885945,2.448245,-3.222992,0.831625,2.442212,-5.939953,-0.968819,-9.616668,5.076785,-8.310076,8.256028,-3.345709,5.166243,8.531651,5.835747,2.373929,3.716970,4.888822,2.297666,-6.763362,1.374164,-3.948179,-7.656566,-3.169488,0.918851,-7.126056,-1.188577,-2.704431,8.182711,8.453948,3.925273,5.017294,5.607707,-7.145508,-5.904077,3.851146,2.097687,0.421560,8.146565,4.625120,5.661282,6.360608,5.286394,6.160187,2.799445,2.504969,2.740450,7.817178,-8.915670,7.093676,6.615593,-0.975223,-8.017804,-5.160053,0.543000,7.665397,9.735993,-5.699360,-2.972564,8.639438,-1.183268,-9.943980,2.258409,4.911726,1.691149,-1.432693,6.173133,-4.170698,4.401423,5.075960,-8.522481,9.788424,5.256592,-5.085699,0.708181,9.336597,0.496222,7.757563,-9.201559,-1.777386,5.504884,-0.630028,0.155186,5.307672,1.026461,2.666605,5.672873,6.044614,-9.393309,-2.974238,4.838453,-5.974500,6.702695,1.086450,-8.389562,4.549058,2.936031,-0.371840,-4.140143,7.178878,3.203613,-7.277245,2.098869,8.103289,3.162355,9.991189,-0.393097,5.262111,3.534327,9.752633,-5.666551,5.918302,-4.402872,-2.307083,4.061933,-3.463296,-7.468420,5.416311,-1.494491,-2.171698,-1.398538,-3.092732,4.897556,-2.133568,-8.662598,-3.782106,4.095970,-3.964821,6.012040,-2.982028,-9.725298,4.981443,1.621119,-5.498385,-8.835978,-7.727728,-3.112108,8.450029,-2.818159,-8.093532,9.098361,-5.745395,9.937708,-5.803028,1.368102,4.908191,1.879601,2.463457,-7.104781,-8.637529,-9.249802,1.825595,1.915702,-4.818302,1.042275,-3.192806,9.330696,5.892468,-2.735481,4.719285,-8.101666,3.655265,-7.070537,2.328469,4.121666,4.230978,-9.737507,-9.268140,7.561674,9.765721,-6.796432,8.022885,-5.363758,-8.110237,-4.618971,8.548505,-0.798836,-3.818562,2.588427,9.812208,5.310289,-6.639586,-9.657080,4.926869,7.860390,-4.061128,-9.440793,6.659699,-8.624935,9.172987,7.017612,7.503786,-7.532944,-4.552312,0.633334,-0.850723,-0.473568,-4.880455,4.365599,-5.547167,7.455478,-3.269417,-4.123363,-3.882922,2.564402,4.738678,-8.879472,9.405690,-6.109650,-9.941809,6.730013,-1.680692,7.500390,-6.751896,-0.830984,6.079511,5.355480,-1.576064,5.215683,-3.156510,-0.898849,-7.920643,3.136818,7.489253,0.088467,3.515915,-6.830172,1.169089,3.375731,0.368995,-5.381620,1.132682,-7.390559,-2.662738,-7.037643,0.494253,-7.554545,-6.958202,4.852388,-8.980293,5.544478,0.189642,5.512129,-2.544696,3.514112,0.146395,-9.832786,4.898440,-2.837227,1.894608,3.441046,9.512047,-9.441508,6.458721,-5.076347,-2.102366,6.860469,7.664677,-3.362773,5.189668,7.589995,-0.634257], dtype = "float64")#candidate|257|(624,)|const|float64
call_256 = relay.TupleGetItem(func_143_call(relay.reshape(const_257.astype('float64'), [3, 13, 16]), relay.reshape(const_257.astype('float64'), [3, 13, 16]), relay.reshape(const_257.astype('float64'), [3, 13, 16]), ), 5)
call_258 = relay.TupleGetItem(func_148_call(relay.reshape(const_257.astype('float64'), [3, 13, 16]), relay.reshape(const_257.astype('float64'), [3, 13, 16]), relay.reshape(const_257.astype('float64'), [3, 13, 16]), ), 5)
bop_260 = relay.equal(const_242.astype('bool'), relay.reshape(bop_250.astype('bool'), relay.shape_of(const_242))) # shape=(2,)
bop_263 = relay.mod(call_256.astype('float32'), const_241.astype('float32')) # shape=(3, 13, 16)
bop_266 = relay.mod(call_258.astype('float32'), const_241.astype('float32')) # shape=(3, 13, 16)
bop_267 = relay.power(const_241.astype('float32'), const_242.astype('float32')) # shape=(2,)
uop_270 = relay.sin(const_241.astype('float32')) # shape=(1,)
var_273 = relay.var("var_273", dtype = "float32", shape = (4,))#candidate|273|(4,)|var|float32
bop_274 = relay.maximum(uop_270.astype('int64'), var_273.astype('int64')) # shape=(4,)
uop_277 = relay.atanh(uop_270.astype('float32')) # shape=(1,)
uop_279 = relay.log(uop_277.astype('float32')) # shape=(1,)
bop_281 = relay.floor_divide(uop_279.astype('float64'), relay.reshape(uop_270.astype('float64'), relay.shape_of(uop_279))) # shape=(1,)
bop_284 = relay.logical_xor(uop_277.astype('uint16'), const_257.astype('uint16')) # shape=(624,)
uop_287 = relay.atan(bop_274.astype('float64')) # shape=(4,)
bop_290 = relay.divide(bop_281.astype('float64'), bop_250.astype('float64')) # shape=(2,)
uop_293 = relay.atan(bop_284.astype('float64')) # shape=(624,)
uop_295 = relay.exp(bop_281.astype('float32')) # shape=(1,)
var_297 = relay.var("var_297", dtype = "float32", shape = (6,))#candidate|297|(6,)|var|float32
bop_298 = relay.add(uop_277.astype('float32'), var_297.astype('float32')) # shape=(6,)
uop_303 = relay.asin(uop_295.astype('float32')) # shape=(1,)
bop_305 = relay.greater_equal(uop_303.astype('bool'), relay.reshape(bop_281.astype('bool'), relay.shape_of(uop_303))) # shape=(1,)
uop_308 = relay.atan(bop_305.astype('float32')) # shape=(1,)
var_310 = relay.var("var_310", dtype = "float32", shape = (11,))#candidate|310|(11,)|var|float32
bop_311 = relay.equal(uop_308.astype('bool'), var_310.astype('bool')) # shape=(11,)
bop_314 = relay.bitwise_and(uop_295.astype('uint8'), var_273.astype('uint8')) # shape=(4,)
func_80_call = mod.get_global_var('func_80')
func_86_call = mutated_mod.get_global_var('func_86')
const_318 = relay.const([[1.560224,-7.638841,-3.546203,-0.016122,1.596054,-5.953232,-4.380070,-3.600926,7.999486,2.975491,-9.029248,0.542311,-8.987101,-2.161319,2.674971,3.532898,7.103585,4.504236,-3.112390,1.371868,-8.399770,-9.700173,-7.510686,7.532731,1.449454,-6.181013,9.105036,0.826656,-3.697472,-9.719377,-6.675126,9.382442,4.095586,9.202492,-7.822732,5.270731,4.127411,9.250630,1.069328,1.867184,4.835706,-2.770580,2.525894,-1.252975,4.668295,9.892543,8.255507,9.040836,8.865037,-6.490555,-0.354526,6.285643,3.954722,7.532268,5.367667,-2.664141,-9.262617,-7.021316,-1.210057,-6.283506,6.752663,9.180104,-0.396117,-1.722516,0.176324,7.042940,-8.944607,8.741281,-1.387793,-4.046537,-5.825619,-9.345005,-4.321272,-3.729146,-5.632511,-0.345543,3.277615,-4.596799,7.977369,0.782163,2.269263,-4.962473,0.471117,-1.645164,-5.572004,-8.967632,9.978179,-1.829044,8.098670,-4.935574,7.684938,3.427882,-8.878404,-4.801876,-0.054475,-5.109882,-0.038497,-9.383184,2.789995,3.645120,3.946095,-0.634240,3.113633,-0.082893,3.369162,-9.028787,-1.483724,6.799110,9.895536,-7.161104,-1.425815,1.324406,5.015492,-9.085520,-7.623669,-5.386170,3.294509,-8.151212,3.116914,5.232367,1.167627,9.906938,-0.606078,-7.451833,-8.819773,6.266839,-0.643762,-0.029473,-4.107300,-9.634779,-8.971292,9.372970,0.694129,-3.230294,-2.491686,9.497122,9.975453,8.186758,-2.761396,5.477741,-5.617183,1.602324,-5.467292,-8.751304,-3.631728,5.932592,-9.647583,9.675057,6.276237,-9.067279,5.007957,8.854212,5.328977,-6.793531,0.056042,-2.214260,7.788302,-8.480010,9.225503,-2.929048,5.475602,2.186147,-6.378684,1.823049,-1.794954,-0.748226,2.089993,7.110712,-7.759373,-4.832342,4.609817,-0.262268,-0.141883,-9.279814,6.903640,-6.484270,4.550952,-4.422084,-5.748787,0.921625,-6.239875,-8.868513,0.270683,-3.155853,-5.658409,1.085700,7.218230,-7.958136,-1.786318,-8.232545,4.914884,-4.672663,-3.609098,7.780503,-8.042700,4.572746,-3.136701,3.297908,9.082052,1.532388,-1.046294,-4.582354,-0.228166,-2.578597,9.462065,0.986824,8.773780,9.785186,-7.709559,-5.708986,4.357326,7.780157,6.040165,-1.963313,-5.676091,-7.971162,-5.864931,7.047032,5.756670,-0.179121,-8.017807,-5.439881,2.969081,1.184876,-0.765182,2.339228,3.927637,1.504054,7.071697,-6.114672,3.022560,-2.090741,7.783551,6.415846,4.144982,6.381723,4.769750,7.042696,-9.951357,-7.526520,-9.596315,-5.242990,6.457572,5.394981,1.752056,-8.062490,-2.800440,8.405351,-9.944203,4.125627,-6.088549,3.014333,-2.378398,-5.209352,8.174975,2.732160,-0.819855,-6.210868,7.062497,-6.491100,6.633812,1.463520,-2.378062,6.064657,1.612550,-7.718573,2.316198,9.058121,2.176674,-3.004935,3.013781,-9.509870,-6.918651,8.063122,-0.500759,-2.269099,-7.431680,-4.760526,-2.372629,-4.388268,-4.919062,5.357203,-9.697463,3.614550,3.105119,0.184230,3.710469,-4.277351,-9.227560,8.933243,-5.474960,-7.562207,0.835508,9.691506,-7.603122,-9.208930,-0.043047,-6.898686,-6.601155,-3.254615,-2.419939,-2.104402,-9.191769,2.827617,-4.668019,1.852248,-7.003739,-5.327601]], dtype = "float64")#candidate|318|(1, 308)|const|float64
call_317 = relay.TupleGetItem(func_80_call(relay.reshape(const_318.astype('float64'), [11, 2, 14]), relay.reshape(const_318.astype('float64'), [11, 2, 14]), relay.reshape(const_318.astype('uint8'), [11, 2, 14]), relay.reshape(const_318.astype('int64'), [11, 2, 14]), relay.reshape(const_318.astype('bool'), [11, 2, 14]), ), 5)
call_319 = relay.TupleGetItem(func_86_call(relay.reshape(const_318.astype('float64'), [11, 2, 14]), relay.reshape(const_318.astype('float64'), [11, 2, 14]), relay.reshape(const_318.astype('uint8'), [11, 2, 14]), relay.reshape(const_318.astype('int64'), [11, 2, 14]), relay.reshape(const_318.astype('bool'), [11, 2, 14]), ), 5)
bop_322 = relay.not_equal(bop_314.astype('bool'), relay.reshape(var_273.astype('bool'), relay.shape_of(bop_314))) # shape=(4,)
uop_325 = relay.log2(uop_279.astype('float64')) # shape=(1,)
var_331 = relay.var("var_331", dtype = "float64", shape = (1,))#candidate|331|(1,)|var|float64
bop_332 = relay.logical_and(uop_325.astype('bool'), relay.reshape(var_331.astype('bool'), relay.shape_of(uop_325))) # shape=(1,)
bop_335 = relay.less(bop_305.astype('bool'), uop_293.astype('bool')) # shape=(624,)
output = relay.Tuple([bop_260,bop_263,bop_267,uop_287,bop_290,bop_298,bop_311,call_317,const_318,bop_322,bop_332,bop_335,])
output2 = relay.Tuple([bop_260,bop_266,bop_267,uop_287,bop_290,bop_298,bop_311,call_319,const_318,bop_322,bop_332,bop_335,])
func_338 = relay.Function([var_273,var_297,var_310,var_331,], output)
mod['func_338'] = func_338
mod = relay.transform.InferType()(mod)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
func_338_call = mutated_mod.get_global_var('func_338')
var_340 = relay.var("var_340", dtype = "float32", shape = (4,))#candidate|340|(4,)|var|float32
var_341 = relay.var("var_341", dtype = "float32", shape = (6,))#candidate|341|(6,)|var|float32
var_342 = relay.var("var_342", dtype = "float32", shape = (11,))#candidate|342|(11,)|var|float32
var_343 = relay.var("var_343", dtype = "float64", shape = (1,))#candidate|343|(1,)|var|float64
call_339 = func_338_call(var_340,var_341,var_342,var_343,)
output = call_339
func_344 = relay.Function([var_340,var_341,var_342,var_343,], output)
mutated_mod['func_344'] = func_344
mutated_mod = relay.transform.InferType()(mutated_mod)
const_354 = relay.const([5.942541,-6.626306,-7.922827,-2.985662], dtype = "float32")#candidate|354|(4,)|const|float32
uop_355 = relay.log2(const_354.astype('float32')) # shape=(4,)
uop_357 = relay.tan(uop_355.astype('float64')) # shape=(4,)
bop_359 = relay.less(uop_355.astype('bool'), relay.reshape(uop_357.astype('bool'), relay.shape_of(uop_355))) # shape=(4,)
var_362 = relay.var("var_362", dtype = "float64", shape = (4,))#candidate|362|(4,)|var|float64
bop_363 = relay.equal(uop_357.astype('bool'), relay.reshape(var_362.astype('bool'), relay.shape_of(uop_357))) # shape=(4,)
uop_366 = relay.rsqrt(uop_355.astype('float64')) # shape=(4,)
bop_368 = relay.equal(uop_366.astype('bool'), relay.reshape(bop_359.astype('bool'), relay.shape_of(uop_366))) # shape=(4,)
var_371 = relay.var("var_371", dtype = "float64", shape = (4,))#candidate|371|(4,)|var|float64
bop_372 = relay.subtract(var_362.astype('uint8'), relay.reshape(var_371.astype('uint8'), relay.shape_of(var_362))) # shape=(4,)
bop_375 = relay.multiply(bop_368.astype('int8'), relay.reshape(uop_355.astype('int8'), relay.shape_of(bop_368))) # shape=(4,)
func_338_call = mod.get_global_var('func_338')
func_344_call = mutated_mod.get_global_var('func_344')
var_380 = relay.var("var_380", dtype = "float32", shape = (6,))#candidate|380|(6,)|var|float32
var_381 = relay.var("var_381", dtype = "float32", shape = (1, 11))#candidate|381|(1, 11)|var|float32
var_382 = relay.var("var_382", dtype = "float64", shape = (1,))#candidate|382|(1,)|var|float64
call_379 = relay.TupleGetItem(func_338_call(relay.reshape(bop_359.astype('float32'), [4,]), relay.reshape(var_380.astype('float32'), [6,]), relay.reshape(var_381.astype('float32'), [11,]), relay.reshape(var_382.astype('float64'), [1,]), ), 11)
call_383 = relay.TupleGetItem(func_344_call(relay.reshape(bop_359.astype('float32'), [4,]), relay.reshape(var_380.astype('float32'), [6,]), relay.reshape(var_381.astype('float32'), [11,]), relay.reshape(var_382.astype('float64'), [1,]), ), 11)
var_386 = relay.var("var_386", dtype = "float32", shape = (4,))#candidate|386|(4,)|var|float32
bop_387 = relay.multiply(uop_355.astype('uint16'), relay.reshape(var_386.astype('uint16'), relay.shape_of(uop_355))) # shape=(4,)
var_390 = relay.var("var_390", dtype = "float64", shape = (4,))#candidate|390|(4,)|var|float64
bop_391 = relay.floor_mod(uop_357.astype('float32'), relay.reshape(var_390.astype('float32'), relay.shape_of(uop_357))) # shape=(4,)
bop_394 = relay.add(uop_355.astype('uint32'), relay.reshape(bop_372.astype('uint32'), relay.shape_of(uop_355))) # shape=(4,)
bop_398 = relay.add(bop_363.astype('float64'), relay.reshape(uop_357.astype('float64'), relay.shape_of(bop_363))) # shape=(4,)
uop_401 = relay.acos(uop_357.astype('float32')) # shape=(4,)
uop_403 = relay.erf(uop_355.astype('float32')) # shape=(4,)
bop_405 = relay.left_shift(uop_401.astype('int8'), relay.reshape(bop_391.astype('int8'), relay.shape_of(uop_401))) # shape=(4,)
output = relay.Tuple([bop_375,call_379,var_380,var_381,var_382,bop_387,bop_394,bop_398,uop_403,bop_405,])
output2 = relay.Tuple([bop_375,call_383,var_380,var_381,var_382,bop_387,bop_394,bop_398,uop_403,bop_405,])
func_408 = relay.Function([var_362,var_371,var_380,var_381,var_382,var_386,var_390,], output)
mod['func_408'] = func_408
mod = relay.transform.InferType()(mod)
var_409 = relay.var("var_409", dtype = "float64", shape = (4,))#candidate|409|(4,)|var|float64
var_410 = relay.var("var_410", dtype = "float64", shape = (4,))#candidate|410|(4,)|var|float64
var_411 = relay.var("var_411", dtype = "float32", shape = (6,))#candidate|411|(6,)|var|float32
var_412 = relay.var("var_412", dtype = "float32", shape = (1, 11))#candidate|412|(1, 11)|var|float32
var_413 = relay.var("var_413", dtype = "float64", shape = (1,))#candidate|413|(1,)|var|float64
var_414 = relay.var("var_414", dtype = "float32", shape = (4,))#candidate|414|(4,)|var|float32
var_415 = relay.var("var_415", dtype = "float64", shape = (4,))#candidate|415|(4,)|var|float64
output = func_408(var_409,var_410,var_411,var_412,var_413,var_414,var_415,)
func_416 = relay.Function([var_409,var_410,var_411,var_412,var_413,var_414,var_415,], output)
mutated_mod['func_416'] = func_416
mutated_mod = relay.transform.InferType()(mutated_mod)
const_421 = relay.const([[-8,3,5,-3,-10,10,3,-4,2,-9,6,-7,4,8,1,-7],[4,-6,-7,9,-10,-9,-4,-10,-10,1,5,-6,4,-2,3,-5],[-9,-9,-2,6,-5,4,9,-6,-3,6,-4,8,-8,4,-10,-10],[2,5,-6,-5,-7,10,3,4,-4,-3,-3,-8,-1,-4,-5,-8],[-6,10,-8,-4,-6,-1,-2,-2,-5,7,-5,-3,-9,-7,-5,-1],[2,2,-8,2,-3,-2,10,-3,10,7,7,4,-8,10,8,2],[-7,-7,-3,1,-6,10,9,-2,-5,6,6,-8,1,-8,3,8],[-4,-5,5,-2,-8,-6,-8,-6,5,3,-10,-8,10,1,-5,1],[5,6,-1,1,-1,-6,5,-9,4,4,4,-6,-7,6,-7,1],[5,1,4,-7,9,3,5,-4,4,4,9,2,8,-8,-9,6],[6,9,-9,6,-10,-2,9,-6,2,-4,-10,9,-6,-5,-7,8],[10,5,6,7,-8,-4,5,-3,10,9,4,9,8,5,-5,-1],[4,4,9,7,10,-6,1,-3,-7,-10,2,-2,-5,8,-2,7],[-4,8,-2,-1,-4,1,-8,-6,-9,-9,8,-3,-8,3,9,10]], dtype = "int16")#candidate|421|(14, 16)|const|int16
var_422 = relay.var("var_422", dtype = "int16", shape = (14, 16))#candidate|422|(14, 16)|var|int16
bop_423 = relay.equal(const_421.astype('bool'), relay.reshape(var_422.astype('bool'), relay.shape_of(const_421))) # shape=(14, 16)
bop_426 = relay.add(bop_423.astype('int32'), relay.reshape(var_422.astype('int32'), relay.shape_of(bop_423))) # shape=(14, 16)
bop_429 = relay.greater_equal(bop_426.astype('bool'), relay.reshape(var_422.astype('bool'), relay.shape_of(bop_426))) # shape=(14, 16)
var_432 = relay.var("var_432", dtype = "bool", shape = (14, 16))#candidate|432|(14, 16)|var|bool
bop_433 = relay.floor_divide(bop_423.astype('float32'), relay.reshape(var_432.astype('float32'), relay.shape_of(bop_423))) # shape=(14, 16)
uop_438 = relay.atanh(bop_426.astype('float64')) # shape=(14, 16)
uop_440 = relay.sinh(var_432.astype('float64')) # shape=(14, 16)
func_227_call = mod.get_global_var('func_227')
func_231_call = mutated_mod.get_global_var('func_231')
var_443 = relay.var("var_443", dtype = "int32", shape = (1,))#candidate|443|(1,)|var|int32
const_444 = relay.const([-1,2,-1,7,-7,9,-7,10,1,-8,9,-8,-9,-1,-6], dtype = "int32")#candidate|444|(15,)|const|int32
call_442 = relay.TupleGetItem(func_227_call(relay.reshape(var_443.astype('int32'), [1,]), relay.reshape(const_444.astype('int32'), [15,]), relay.reshape(const_444.astype('float64'), [15,]), ), 1)
call_445 = relay.TupleGetItem(func_231_call(relay.reshape(var_443.astype('int32'), [1,]), relay.reshape(const_444.astype('int32'), [15,]), relay.reshape(const_444.astype('float64'), [15,]), ), 1)
bop_446 = relay.divide(bop_426.astype('float64'), relay.reshape(bop_429.astype('float64'), relay.shape_of(bop_426))) # shape=(14, 16)
var_449 = relay.var("var_449", dtype = "float64", shape = (14, 16))#candidate|449|(14, 16)|var|float64
bop_450 = relay.multiply(uop_438.astype('float64'), relay.reshape(var_449.astype('float64'), relay.shape_of(uop_438))) # shape=(14, 16)
uop_453 = relay.sin(var_422.astype('float64')) # shape=(14, 16)
output = relay.Tuple([bop_433,uop_440,call_442,var_443,const_444,bop_446,bop_450,uop_453,])
output2 = relay.Tuple([bop_433,uop_440,call_445,var_443,const_444,bop_446,bop_450,uop_453,])
func_455 = relay.Function([var_422,var_432,var_443,var_449,], output)
mod['func_455'] = func_455
mod = relay.transform.InferType()(mod)
var_456 = relay.var("var_456", dtype = "int16", shape = (14, 16))#candidate|456|(14, 16)|var|int16
var_457 = relay.var("var_457", dtype = "bool", shape = (14, 16))#candidate|457|(14, 16)|var|bool
var_458 = relay.var("var_458", dtype = "int32", shape = (1,))#candidate|458|(1,)|var|int32
var_459 = relay.var("var_459", dtype = "float64", shape = (14, 16))#candidate|459|(14, 16)|var|float64
output = func_455(var_456,var_457,var_458,var_459,)
func_460 = relay.Function([var_456,var_457,var_458,var_459,], output)
mutated_mod['func_460'] = func_460
mutated_mod = relay.transform.InferType()(mutated_mod)
var_462 = relay.var("var_462", dtype = "float64", shape = (11, 9))#candidate|462|(11, 9)|var|float64
uop_463 = relay.rsqrt(var_462.astype('float64')) # shape=(11, 9)
output = relay.Tuple([uop_463,])
output2 = relay.Tuple([uop_463,])
F = relay.Function([var_462,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_462,], output2)
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
input_462= np.array([[-8.923781,-7.940352,-8.822570,-3.956300,4.496418,1.095972,-6.379929,4.748952,-1.137892],[-5.529132,0.954361,0.086757,-3.524956,-9.186604,0.869795,-2.312774,-2.403793,-6.907985],[2.440647,-1.486740,0.677688,-2.946345,7.257505,4.732264,1.615950,5.596098,-3.468914],[-6.062522,-5.713746,6.821890,4.622823,0.242312,-7.494774,-8.768909,5.014044,-6.357145],[2.933037,-3.960031,-4.772677,-9.535114,-7.392674,5.424987,-9.396952,4.861228,-0.615931],[3.237708,-0.581628,3.635252,-8.056775,2.977594,-0.430179,-3.391593,-4.039948,4.241824],[6.715197,-8.653812,8.007501,-1.542553,-2.485616,-5.042562,-5.978498,-5.430389,3.974902],[-0.450745,9.372875,8.048872,0.263687,9.087626,-6.074727,4.004018,3.796947,6.391005],[5.416929,-3.854582,-1.874704,-2.593485,-8.945697,9.546687,-8.451092,-4.003093,9.726576],[5.373797,9.846952,1.921004,1.645235,8.953495,-9.606721,-5.265619,-5.704903,0.020441],[3.238399,-1.486368,-0.193779,9.848906,9.138816,2.370249,0.985604,1.596576,8.436522]], dtype='float64')
module1.set_input('var_462', input_462)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_462, )
res3 = intrp3.evaluate()(input_462, )
res4 = intrp4.evaluate()(input_462, )
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
module5.set_input('var_462', input_462)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_462, )
res7 = intrp7.evaluate()(input_462, )
res8 = intrp8.evaluate()(input_462, )
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
module9.set_input('var_462', input_462)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_462, )
res11 = intrp11.evaluate()(input_462, )
res12 = intrp12.evaluate()(input_462, )
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
module13.set_input('var_462', input_462)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_462, )
res15 = intrp15.evaluate()(input_462, )
res16 = intrp16.evaluate()(input_462, )
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
module17.set_input('var_462', input_462)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_462, )
res19 = intrp19.evaluate()(input_462, )
res20 = intrp20.evaluate()(input_462, )
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
module21.set_input('var_462', input_462)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_462, )
res23 = intrp23.evaluate()(input_462, )
res24 = intrp24.evaluate()(input_462, )
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

'''22: TVMFuncCall
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
16: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
14: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
13: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
12: _ZNSt17_Function_handlerIFSt10sha
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''