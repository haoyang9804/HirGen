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
var_0 = relay.var("var_0", dtype = "uint32", shape = ())#candidate|0|()|var|uint32
var_1 = relay.var("var_1", dtype = "uint32", shape = ())#candidate|1|()|var|uint32
bop_2 = relay.greater_equal(var_0.astype('bool'), var_1.astype('bool')) # shape=()
var_5 = relay.var("var_5", dtype = "uint32", shape = ())#candidate|5|()|var|uint32
bop_6 = relay.greater_equal(var_1.astype('bool'), var_5.astype('bool')) # shape=()
var_9 = relay.var("var_9", dtype = "bool", shape = (11,))#candidate|9|(11,)|var|bool
bop_10 = relay.floor_divide(bop_6.astype('float32'), var_9.astype('float32')) # shape=(11,)
output = relay.Tuple([bop_2,bop_10,])
output2 = relay.Tuple([bop_2,bop_10,])
func_13 = relay.Function([var_0,var_1,var_5,var_9,], output)
mod['func_13'] = func_13
mod = relay.transform.InferType()(mod)
mutated_mod['func_13'] = func_13
mutated_mod = relay.transform.InferType()(mutated_mod)
func_13_call = mutated_mod.get_global_var('func_13')
var_15 = relay.var("var_15", dtype = "uint32", shape = ())#candidate|15|()|var|uint32
var_16 = relay.var("var_16", dtype = "uint32", shape = ())#candidate|16|()|var|uint32
var_17 = relay.var("var_17", dtype = "uint32", shape = ())#candidate|17|()|var|uint32
var_18 = relay.var("var_18", dtype = "bool", shape = (11,))#candidate|18|(11,)|var|bool
call_14 = func_13_call(var_15,var_16,var_17,var_18,)
output = call_14
func_19 = relay.Function([var_15,var_16,var_17,var_18,], output)
mutated_mod['func_19'] = func_19
mutated_mod = relay.transform.InferType()(mutated_mod)
var_21 = relay.var("var_21", dtype = "uint16", shape = (15, 2))#candidate|21|(15, 2)|var|uint16
const_22 = relay.const([[-8,2],[-5,1],[1,1],[-6,-6],[9,-1],[-9,-6],[6,-6],[6,-7],[-5,8],[-1,7],[8,1],[8,8],[-1,-8],[9,-1],[-10,4]], dtype = "uint16")#candidate|22|(15, 2)|const|uint16
bop_23 = relay.right_shift(var_21.astype('uint16'), relay.reshape(const_22.astype('uint16'), relay.shape_of(var_21))) # shape=(15, 2)
const_26 = relay.const([[-1,-1],[9,-4],[-4,-3],[3,-1],[-1,6],[7,2],[1,5],[3,9],[3,6],[-7,6],[1,4],[3,8],[6,-9],[-8,-6],[-4,-7]], dtype = "uint16")#candidate|26|(15, 2)|const|uint16
bop_27 = relay.equal(bop_23.astype('bool'), relay.reshape(const_26.astype('bool'), relay.shape_of(bop_23))) # shape=(15, 2)
const_30 = relay.const([[2,-7],[10,-8],[-7,4],[-1,-2],[5,9],[9,-3],[5,8],[-4,-3],[6,9],[-3,1],[-3,2],[1,-8],[7,4],[-7,3],[-6,-5]], dtype = "uint16")#candidate|30|(15, 2)|const|uint16
bop_31 = relay.left_shift(var_21.astype('int16'), relay.reshape(const_30.astype('int16'), relay.shape_of(var_21))) # shape=(15, 2)
uop_34 = relay.cosh(const_30.astype('float64')) # shape=(15, 2)
uop_36 = relay.erf(const_26.astype('float32')) # shape=(15, 2)
var_38 = relay.var("var_38", dtype = "float32", shape = (15, 2))#candidate|38|(15, 2)|var|float32
bop_39 = relay.bitwise_xor(uop_36.astype('uint16'), relay.reshape(var_38.astype('uint16'), relay.shape_of(uop_36))) # shape=(15, 2)
func_13_call = mod.get_global_var('func_13')
func_19_call = mutated_mod.get_global_var('func_19')
const_43 = relay.const(1, dtype = "uint32")#candidate|43|()|const|uint32
const_44 = relay.const([True,False,True,False,True,True,False,False,False,False,True], dtype = "bool")#candidate|44|(11,)|const|bool
call_42 = relay.TupleGetItem(func_13_call(relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_44.astype('bool'), [11,]), ), 0)
call_45 = relay.TupleGetItem(func_19_call(relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_43.astype('uint32'), []), relay.reshape(const_44.astype('bool'), [11,]), ), 0)
uop_46 = relay.cos(bop_39.astype('float32')) # shape=(15, 2)
bop_48 = relay.equal(uop_46.astype('bool'), const_43.astype('bool')) # shape=(15, 2)
uop_51 = relay.tan(bop_48.astype('float32')) # shape=(15, 2)
bop_53 = relay.bitwise_and(uop_46.astype('uint32'), relay.reshape(bop_31.astype('uint32'), relay.shape_of(uop_46))) # shape=(15, 2)
bop_56 = relay.minimum(bop_53.astype('int8'), relay.reshape(bop_39.astype('int8'), relay.shape_of(bop_53))) # shape=(15, 2)
bop_59 = relay.not_equal(uop_46.astype('bool'), relay.reshape(uop_34.astype('bool'), relay.shape_of(uop_46))) # shape=(15, 2)
uop_62 = relay.sqrt(bop_56.astype('float64')) # shape=(15, 2)
bop_64 = relay.maximum(uop_51.astype('int64'), relay.reshape(const_30.astype('int64'), relay.shape_of(uop_51))) # shape=(15, 2)
uop_67 = relay.sin(bop_56.astype('float32')) # shape=(15, 2)
uop_69 = relay.sin(uop_51.astype('float64')) # shape=(15, 2)
const_71 = relay.const([[9.399274,0.962730],[-8.765877,7.872548],[3.898504,-4.570022],[-2.222108,-8.265877],[-6.293243,-4.222361],[7.833932,8.315621],[2.851684,-6.129830],[-9.945730,-8.308691],[0.848749,-3.534504],[-6.599576,9.939287],[-3.434305,3.771325],[-9.019480,-7.191119],[-2.144806,-9.986456],[-5.091797,-8.659154],[0.540138,5.120998]], dtype = "float32")#candidate|71|(15, 2)|const|float32
bop_72 = relay.multiply(uop_67.astype('int64'), relay.reshape(const_71.astype('int64'), relay.shape_of(uop_67))) # shape=(15, 2)
output = relay.Tuple([bop_27,call_42,const_44,bop_59,uop_62,bop_64,uop_69,bop_72,])
output2 = relay.Tuple([bop_27,call_45,const_44,bop_59,uop_62,bop_64,uop_69,bop_72,])
func_75 = relay.Function([var_21,var_38,], output)
mod['func_75'] = func_75
mod = relay.transform.InferType()(mod)
mutated_mod['func_75'] = func_75
mutated_mod = relay.transform.InferType()(mutated_mod)
func_75_call = mutated_mod.get_global_var('func_75')
var_77 = relay.var("var_77", dtype = "uint16", shape = (15, 2))#candidate|77|(15, 2)|var|uint16
var_78 = relay.var("var_78", dtype = "float32", shape = (15, 2))#candidate|78|(15, 2)|var|float32
call_76 = func_75_call(var_77,var_78,)
output = call_76
func_79 = relay.Function([var_77,var_78,], output)
mutated_mod['func_79'] = func_79
mutated_mod = relay.transform.InferType()(mutated_mod)
var_81 = relay.var("var_81", dtype = "float64", shape = (14,))#candidate|81|(14,)|var|float64
uop_82 = relay.erf(var_81.astype('float64')) # shape=(14,)
const_84 = relay.const([8.523996,3.177295,8.925059,1.524934,-8.661345,-7.175051,-8.825320,-2.236033,7.328648,-5.601930,-7.914190,-8.800712,4.580611,2.649156], dtype = "float64")#candidate|84|(14,)|const|float64
bop_85 = relay.floor_divide(var_81.astype('float64'), relay.reshape(const_84.astype('float64'), relay.shape_of(var_81))) # shape=(14,)
var_88 = relay.var("var_88", dtype = "float64", shape = (14,))#candidate|88|(14,)|var|float64
bop_89 = relay.mod(uop_82.astype('float64'), relay.reshape(var_88.astype('float64'), relay.shape_of(uop_82))) # shape=(14,)
func_13_call = mod.get_global_var('func_13')
func_19_call = mutated_mod.get_global_var('func_19')
const_93 = relay.const(-8, dtype = "uint32")#candidate|93|()|const|uint32
var_94 = relay.var("var_94", dtype = "bool", shape = (11,))#candidate|94|(11,)|var|bool
call_92 = relay.TupleGetItem(func_13_call(relay.reshape(const_93.astype('uint32'), []), relay.reshape(const_93.astype('uint32'), []), relay.reshape(const_93.astype('uint32'), []), relay.reshape(var_94.astype('bool'), [11,]), ), 0)
call_95 = relay.TupleGetItem(func_19_call(relay.reshape(const_93.astype('uint32'), []), relay.reshape(const_93.astype('uint32'), []), relay.reshape(const_93.astype('uint32'), []), relay.reshape(var_94.astype('bool'), [11,]), ), 0)
bop_96 = relay.equal(bop_89.astype('bool'), relay.reshape(var_81.astype('bool'), relay.shape_of(bop_89))) # shape=(14,)
uop_99 = relay.erf(uop_82.astype('float64')) # shape=(14,)
bop_101 = relay.bitwise_and(uop_82.astype('uint32'), const_93.astype('uint32')) # shape=(14,)
var_104 = relay.var("var_104", dtype = "float64", shape = (14,))#candidate|104|(14,)|var|float64
bop_105 = relay.greater(uop_99.astype('bool'), relay.reshape(var_104.astype('bool'), relay.shape_of(uop_99))) # shape=(14,)
uop_108 = relay.cos(call_92.astype('float64')) # shape=()
uop_110 = relay.cos(call_95.astype('float64')) # shape=()
var_111 = relay.var("var_111", dtype = "bool", shape = (14,))#candidate|111|(14,)|var|bool
bop_112 = relay.bitwise_or(bop_105.astype('uint16'), relay.reshape(var_111.astype('uint16'), relay.shape_of(bop_105))) # shape=(14,)
uop_115 = relay.exp(uop_99.astype('float64')) # shape=(14,)
uop_117 = relay.log(uop_99.astype('float64')) # shape=(14,)
uop_119 = relay.cosh(bop_96.astype('float32')) # shape=(14,)
bop_121 = relay.power(var_81.astype('float32'), uop_108.astype('float32')) # shape=(14,)
bop_124 = relay.power(var_81.astype('float32'), uop_110.astype('float32')) # shape=(14,)
bop_125 = relay.bitwise_or(uop_117.astype('int32'), relay.reshape(var_111.astype('int32'), relay.shape_of(uop_117))) # shape=(14,)
uop_128 = relay.acos(bop_125.astype('float64')) # shape=(14,)
uop_130 = relay.sqrt(uop_128.astype('float32')) # shape=(14,)
bop_132 = relay.subtract(uop_117.astype('int64'), relay.reshape(const_84.astype('int64'), relay.shape_of(uop_117))) # shape=(14,)
uop_135 = relay.log2(bop_132.astype('float32')) # shape=(14,)
bop_137 = relay.logical_or(uop_135.astype('bool'), relay.reshape(bop_132.astype('bool'), relay.shape_of(uop_135))) # shape=(14,)
const_140 = relay.const([-5.672573,-1.037984,-5.679471,1.750796,9.517907,-8.188585,8.833024,2.863520,5.262062,4.043728,2.682235,-0.189607,3.835831,-4.032365], dtype = "float32")#candidate|140|(14,)|const|float32
bop_141 = relay.logical_or(uop_130.astype('bool'), relay.reshape(const_140.astype('bool'), relay.shape_of(uop_130))) # shape=(14,)
bop_144 = relay.bitwise_xor(bop_125.astype('uint16'), call_92.astype('uint16')) # shape=(14,)
bop_147 = relay.bitwise_xor(bop_125.astype('uint16'), call_95.astype('uint16')) # shape=(14,)
bop_148 = relay.right_shift(uop_130.astype('int8'), relay.reshape(const_140.astype('int8'), relay.shape_of(uop_130))) # shape=(14,)
output = relay.Tuple([bop_85,var_94,bop_101,bop_112,uop_115,uop_119,bop_121,bop_137,bop_141,bop_144,bop_148,])
output2 = relay.Tuple([bop_85,var_94,bop_101,bop_112,uop_115,uop_119,bop_124,bop_137,bop_141,bop_147,bop_148,])
func_151 = relay.Function([var_81,var_88,var_94,var_104,var_111,], output)
mod['func_151'] = func_151
mod = relay.transform.InferType()(mod)
var_152 = relay.var("var_152", dtype = "float64", shape = (14,))#candidate|152|(14,)|var|float64
var_153 = relay.var("var_153", dtype = "float64", shape = (14,))#candidate|153|(14,)|var|float64
var_154 = relay.var("var_154", dtype = "bool", shape = (11,))#candidate|154|(11,)|var|bool
var_155 = relay.var("var_155", dtype = "float64", shape = (14,))#candidate|155|(14,)|var|float64
var_156 = relay.var("var_156", dtype = "bool", shape = (14,))#candidate|156|(14,)|var|bool
output = func_151(var_152,var_153,var_154,var_155,var_156,)
func_157 = relay.Function([var_152,var_153,var_154,var_155,var_156,], output)
mutated_mod['func_157'] = func_157
mutated_mod = relay.transform.InferType()(mutated_mod)
var_159 = relay.var("var_159", dtype = "float64", shape = ())#candidate|159|()|var|float64
uop_160 = relay.asinh(var_159.astype('float64')) # shape=()
uop_162 = relay.asinh(uop_160.astype('float32')) # shape=()
uop_164 = relay.atanh(uop_160.astype('float32')) # shape=()
bop_166 = relay.logical_and(uop_164.astype('bool'), uop_162.astype('bool')) # shape=()
uop_169 = relay.rsqrt(var_159.astype('float64')) # shape=()
output = relay.Tuple([bop_166,uop_169,])
output2 = relay.Tuple([bop_166,uop_169,])
func_171 = relay.Function([var_159,], output)
mod['func_171'] = func_171
mod = relay.transform.InferType()(mod)
var_172 = relay.var("var_172", dtype = "float64", shape = ())#candidate|172|()|var|float64
output = func_171(var_172)
func_173 = relay.Function([var_172], output)
mutated_mod['func_173'] = func_173
mutated_mod = relay.transform.InferType()(mutated_mod)
const_175 = relay.const([[[-9.877575,-6.519422,1.433854,-1.411467,5.808295,-8.696087,3.504242,-2.714336,4.360447,7.768021,9.321705,2.330062,-4.183061],[1.089552,1.837725,8.430013,-6.027736,8.487734,7.581712,-4.926745,-1.846647,-6.742943,4.462802,7.000741,-3.931855,-9.047193],[-7.828545,-5.807979,-8.041697,-7.363020,0.633677,-3.745099,7.513181,3.201166,8.372902,3.021712,-8.925959,5.811233,4.951449],[-7.858082,5.016416,1.007957,1.935966,-6.158675,-7.861908,5.970228,-1.641571,8.771027,9.619379,4.184166,-4.345233,7.979678],[-9.021754,7.445806,1.517355,1.659362,-7.085569,-2.013253,8.061742,-0.031903,-9.073322,6.900927,7.052040,-3.853214,6.386210],[0.705115,2.484951,-6.047241,7.243940,5.826403,-8.818873,-3.267633,8.140062,-9.224691,4.372960,-4.740091,-8.605346,4.709173],[-5.160800,-1.423730,5.836556,-9.889928,-0.762176,-9.725092,-3.283496,-8.219762,4.012282,0.847142,-0.313813,4.527662,-4.896818]],[[2.832426,8.658247,-4.800435,-8.826609,9.145391,-8.701348,-7.819802,-5.715956,7.742125,-3.444170,0.093844,-7.731690,9.164679],[-2.183029,-7.395579,-6.791249,5.042022,-2.404992,-8.942965,8.550893,-3.995452,1.058450,-5.888800,-9.248121,-9.509388,-6.262410],[-9.988321,-2.234406,6.559505,-5.622008,-4.639961,9.974376,-6.797725,3.584117,-8.949407,-5.296693,6.687874,-8.808372,4.361236],[8.802517,6.522946,-0.777297,-7.273793,7.432795,4.811840,7.852556,4.208038,-8.458152,4.260406,-7.773633,-8.972219,-1.975643],[-9.673073,9.114240,8.184436,8.744169,-2.464950,-4.037034,-5.334008,-9.866718,9.132201,8.914162,-2.643358,-5.897611,-7.817519],[-6.378635,2.149630,-5.863587,5.464874,-6.407427,-7.475024,7.108649,8.214054,9.799457,-5.962317,3.433916,0.566276,-1.177779],[-3.474412,1.666294,1.230743,2.485170,6.262181,-5.918285,-1.779266,1.500015,-1.147174,5.286845,7.564582,1.083543,5.629448]],[[0.664658,-8.275278,-6.873861,2.395006,6.841540,6.498590,5.575845,-1.059179,7.568707,-2.347505,4.603515,-2.620881,-0.148343],[0.380910,7.457045,8.682955,5.241221,-9.256049,-8.840430,-0.213261,-3.970556,0.402130,-9.304242,7.925602,-2.100140,-2.068459],[-1.463519,-6.538770,-4.203229,-2.366209,-4.985393,2.398730,8.953172,-2.020014,4.156890,8.795495,-0.809633,3.891339,-8.498282],[1.821347,4.927575,-1.651750,-3.624726,-7.470325,-1.402644,-6.800652,-8.251915,6.288679,2.915942,-9.190831,-6.407481,-5.321041],[-2.352721,5.993233,-1.417392,8.796630,-9.566632,2.047859,-1.500082,-0.447682,-0.876034,-9.997236,-0.957607,8.713005,5.078127],[-3.254306,3.772536,-0.399159,-4.338826,7.836433,-5.180296,-2.456309,-8.216016,8.327479,-7.248948,-5.591977,-4.103859,9.732082],[-9.601920,-0.966404,-5.312468,7.513156,6.456968,8.204694,8.577882,-7.733404,2.932335,-8.714546,-8.303207,2.400149,4.179075]],[[-9.758019,-1.945061,-8.569678,-5.636053,-5.473042,3.421422,-1.207576,4.241478,1.196692,-0.905790,5.674532,-7.257794,-5.414891],[5.854745,4.753383,1.798156,5.939714,-2.816393,-1.666577,-4.334847,6.791779,-4.028272,7.349839,-5.993263,-1.752699,-2.685392],[-7.045851,4.614682,-1.288484,-1.262450,-1.649628,2.568675,3.277364,7.378513,-2.803913,-4.521884,-2.586669,9.323843,2.073857],[-8.265980,1.645397,4.767880,8.754406,-3.011195,6.844969,-3.952370,-8.303878,5.480186,7.252984,6.562519,8.041554,9.353673],[-2.110450,8.671876,-5.502793,8.849159,-3.877867,-7.099172,2.450464,0.328544,9.372259,-0.003328,4.545737,-4.741909,8.210566],[-8.055604,8.877769,-7.234266,6.495006,0.708762,4.250663,8.863429,2.712184,4.685619,-0.901159,-1.332206,-3.053302,2.481669],[-5.718170,-5.950786,8.160989,-0.333049,7.709620,-5.434478,-0.368500,-7.258870,-1.641273,-0.707057,8.585347,-3.878721,-6.258695]],[[-6.933034,-8.531529,7.049859,4.711250,-5.994021,-1.535747,2.512341,1.850455,-8.240359,-6.304739,0.539198,-9.600024,-3.083222],[6.277836,6.415506,3.394749,5.099902,0.405196,2.601188,-2.506706,9.056851,-0.796262,-9.807888,-2.037894,-9.535763,-2.291658],[6.023706,6.823813,-6.255736,6.510220,-5.116607,1.006811,-1.206789,-5.605227,9.476063,8.419622,-1.651461,-4.812263,1.067152],[2.616101,8.326516,1.855206,8.576362,-8.671164,2.421969,3.517432,3.190158,9.146154,-1.280572,7.788597,0.154993,-4.809765],[-4.835336,-8.508788,-2.096499,0.872377,5.175596,-3.063808,2.897932,0.646065,-7.814744,4.595948,8.773323,7.992027,-0.688605],[-1.548471,2.788324,9.882292,1.327936,3.741135,-6.503398,0.507007,1.325575,-8.419215,-1.134257,-0.188858,-4.581535,-4.892318],[-9.619264,8.964129,-3.371652,-9.419749,3.139820,-4.862128,-4.026815,-0.634535,-5.338940,-8.580005,2.023542,4.570502,-3.235128]]], dtype = "float32")#candidate|175|(5, 7, 13)|const|float32
uop_176 = relay.erf(const_175.astype('float32')) # shape=(5, 7, 13)
uop_178 = relay.rsqrt(const_175.astype('float64')) # shape=(5, 7, 13)
uop_180 = relay.acosh(uop_178.astype('float64')) # shape=(5, 7, 13)
uop_182 = relay.sqrt(uop_176.astype('float64')) # shape=(5, 7, 13)
uop_184 = relay.sin(uop_180.astype('float64')) # shape=(5, 7, 13)
const_186 = relay.const([[[3.674590,-1.720225,9.702100,6.327465,-0.367548,0.361310,9.513866,3.687777,-2.591826,-5.884826,-4.623420,-5.823713,7.901715],[5.069955,2.662099,-3.418727,0.616005,5.974735,-2.770986,-0.306606,3.280910,-8.643485,1.463880,3.390036,-4.878550,5.898128],[2.943918,-6.688908,-8.448264,-3.428222,-6.462167,-0.328473,7.291132,-4.568738,-1.581678,4.842211,-8.513995,-7.342877,2.748276],[-7.436739,5.304346,-0.757979,1.302604,0.448916,-3.565040,-7.890796,-9.370612,3.764807,-3.086906,-9.188372,-0.091887,-4.273599],[2.385202,-0.233537,-1.167722,-1.123835,-7.558974,-6.332745,-6.840192,9.699215,8.834128,9.896634,8.392451,-1.786691,4.905232],[-6.242012,-9.277483,-2.800849,-4.435977,-3.943325,7.880465,-3.713277,7.381647,-0.705503,9.422184,-3.646296,6.940434,-9.595901],[-6.151373,-6.845695,4.737231,-8.405409,-2.822956,-0.323119,4.406407,6.075694,6.182750,1.190062,8.703633,-0.340063,5.347023]],[[6.746822,7.257276,5.465958,-2.971318,8.270556,8.059248,6.933038,-3.398090,-0.436264,-5.330506,1.048369,3.456366,-2.844673],[-3.299610,2.402846,-9.245867,-8.003196,0.509737,0.509785,9.362984,1.513747,-4.892438,-9.838963,-1.491381,7.661063,4.438819],[7.528257,2.623921,7.306800,-0.153234,1.332472,9.156913,-7.573689,-5.997774,4.067381,-8.628537,-1.480886,4.688327,-8.524563],[2.553365,4.094149,-7.317420,1.532947,-5.941488,9.630704,-5.383744,5.760110,-1.199254,-5.996520,-7.263737,-8.175676,-5.226441],[2.896836,-4.889592,-6.490766,8.642463,-1.229384,-0.863409,2.716632,-3.282026,7.037790,-3.293932,-4.388768,5.724795,2.234155],[-1.614419,-4.721489,-9.316186,2.521001,9.652544,-5.902454,9.363752,-8.582368,-3.280349,-2.200704,-4.814325,2.263495,1.244061],[4.911829,-0.973238,-1.590449,4.102943,-8.342092,-2.199700,-8.662253,1.799714,4.346445,6.942169,-4.838181,0.096125,-2.696325]],[[-7.237959,-7.401731,8.425763,-0.091664,-6.547238,0.496467,-6.439255,9.972069,8.426178,-5.470360,-2.027588,3.850735,-3.368396],[-2.918339,-2.703162,5.537160,-9.327751,-0.289991,-6.254557,-7.017462,9.729179,-2.727478,3.795673,3.166364,8.362322,5.814046],[3.550089,8.114763,-4.134111,5.210079,-1.051374,5.084726,7.715532,-7.142024,2.804237,-0.735722,-3.153384,-5.953281,7.450090],[-1.437843,4.564853,-5.287712,-7.131459,-6.882096,9.718414,-1.394453,9.826002,2.063048,3.631311,-3.403688,-7.572271,-5.642960],[5.021163,-7.168535,-0.187392,8.124651,-5.336238,-2.101509,8.572374,5.731938,-0.439758,-7.297882,2.023012,-6.174655,6.976134],[1.019288,-9.155591,5.191315,-8.523842,-1.996742,-0.897367,0.851676,0.051093,-1.167329,-6.849012,-2.169003,7.879581,-2.940722],[-5.489604,7.951098,9.678711,8.042503,-2.346712,4.269403,-8.227730,4.561373,9.317813,1.284328,-2.015718,1.377476,1.484115]],[[-3.940049,-6.027521,0.917161,5.237152,9.533321,-8.259615,-8.732047,3.545320,1.738943,9.311393,-3.999391,4.268034,-1.972747],[9.223971,6.308978,3.822257,8.960875,-0.489120,-0.188546,-4.399500,0.321430,-2.639198,0.787046,8.142251,2.747134,-9.615967],[-3.441408,5.286872,1.504125,-2.995169,1.218614,-7.549460,-6.304811,3.256865,4.222015,-2.839089,-8.476930,-4.002550,-4.828456],[1.774333,6.158367,2.818022,4.330992,-5.793389,-1.889203,4.012726,-8.032868,-1.374370,7.711780,6.770792,8.039740,2.428622],[-4.497936,-1.513779,5.645007,-2.746926,4.698253,-1.989322,-1.415065,4.378882,-3.303871,-0.680360,7.521962,1.003352,-4.402084],[7.526316,-4.799513,-5.820999,-6.295369,1.054846,-0.759925,-0.134510,1.284119,-8.995273,5.506671,-0.642009,6.903194,-3.356534],[-2.058861,9.896862,-8.335420,-7.300682,4.707134,6.253358,0.061136,1.801844,-2.597390,0.640982,5.678413,3.122414,-3.347070]],[[-5.229845,-3.016402,-8.457837,-9.392477,-1.779770,-8.610113,0.053150,9.663287,5.286550,0.155438,-5.169953,9.071300,4.793519],[5.755753,1.710595,8.562488,7.689367,1.547976,-8.481501,0.356671,-3.366469,5.765469,7.855878,5.420252,-1.656260,-0.348488],[-4.691455,-6.653935,-5.842251,5.769191,9.367912,2.797217,-1.287291,-9.571655,-4.808628,1.256068,4.531271,-5.764418,5.813801],[-8.343291,-9.258136,-4.944664,-6.432202,-6.679523,0.764543,-7.539871,-8.865912,5.297730,-5.364732,5.321356,-1.895983,-2.253066],[-9.836157,-8.721145,-0.114644,8.269101,7.612221,8.471503,-0.186333,-7.378442,9.024020,8.497152,-1.903130,2.183593,-5.141047],[8.426027,9.269297,4.229307,-2.670229,-6.604340,-8.174364,1.283722,6.370517,0.485136,0.900985,2.488621,-5.261901,-6.325758],[-8.160235,-9.231560,8.624518,2.682751,3.302471,-4.739290,-2.433070,-0.186988,-1.550799,4.768782,-3.805134,1.887562,-6.533324]]], dtype = "float64")#candidate|186|(5, 7, 13)|const|float64
bop_187 = relay.not_equal(uop_182.astype('bool'), relay.reshape(const_186.astype('bool'), relay.shape_of(uop_182))) # shape=(5, 7, 13)
bop_190 = relay.bitwise_and(uop_180.astype('uint64'), relay.reshape(const_175.astype('uint64'), relay.shape_of(uop_180))) # shape=(5, 7, 13)
output = relay.Tuple([uop_184,bop_187,bop_190,])
output2 = relay.Tuple([uop_184,bop_187,bop_190,])
func_193 = relay.Function([], output)
mod['func_193'] = func_193
mod = relay.transform.InferType()(mod)
output = func_193()
func_194 = relay.Function([], output)
mutated_mod['func_194'] = func_194
mutated_mod = relay.transform.InferType()(mutated_mod)
var_195 = relay.var("var_195", dtype = "float32", shape = (12, 10))#candidate|195|(12, 10)|var|float32
var_196 = relay.var("var_196", dtype = "float32", shape = (12, 10))#candidate|196|(12, 10)|var|float32
bop_197 = relay.floor_divide(var_195.astype('float32'), relay.reshape(var_196.astype('float32'), relay.shape_of(var_195))) # shape=(12, 10)
uop_200 = relay.acosh(bop_197.astype('float32')) # shape=(12, 10)
bop_202 = relay.less_equal(uop_200.astype('bool'), relay.reshape(bop_197.astype('bool'), relay.shape_of(uop_200))) # shape=(12, 10)
uop_205 = relay.atan(bop_202.astype('float64')) # shape=(12, 10)
bop_207 = relay.bitwise_xor(uop_205.astype('uint8'), relay.reshape(bop_202.astype('uint8'), relay.shape_of(uop_205))) # shape=(12, 10)
uop_210 = relay.sin(bop_202.astype('float64')) # shape=(12, 10)
output = relay.Tuple([bop_207,uop_210,])
output2 = relay.Tuple([bop_207,uop_210,])
func_212 = relay.Function([var_195,var_196,], output)
mod['func_212'] = func_212
mod = relay.transform.InferType()(mod)
mutated_mod['func_212'] = func_212
mutated_mod = relay.transform.InferType()(mutated_mod)
func_212_call = mutated_mod.get_global_var('func_212')
var_214 = relay.var("var_214", dtype = "float32", shape = (12, 10))#candidate|214|(12, 10)|var|float32
var_215 = relay.var("var_215", dtype = "float32", shape = (12, 10))#candidate|215|(12, 10)|var|float32
call_213 = func_212_call(var_214,var_215,)
output = call_213
func_216 = relay.Function([var_214,var_215,], output)
mutated_mod['func_216'] = func_216
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_218 = relay.TupleGetItem(func_193_call(), 0)
call_219 = relay.TupleGetItem(func_194_call(), 0)
var_220 = relay.var("var_220", dtype = "float64", shape = (5, 7, 13))#candidate|220|(5, 7, 13)|var|float64
bop_221 = relay.right_shift(call_218.astype('uint64'), relay.reshape(var_220.astype('uint64'), relay.shape_of(call_218))) # shape=(5, 7, 13)
bop_224 = relay.right_shift(call_219.astype('uint64'), relay.reshape(var_220.astype('uint64'), relay.shape_of(call_219))) # shape=(5, 7, 13)
bop_225 = relay.floor_mod(bop_221.astype('float32'), relay.reshape(var_220.astype('float32'), relay.shape_of(bop_221))) # shape=(5, 7, 13)
bop_228 = relay.floor_mod(bop_224.astype('float32'), relay.reshape(var_220.astype('float32'), relay.shape_of(bop_224))) # shape=(5, 7, 13)
output = bop_225
output2 = bop_228
func_229 = relay.Function([var_220,], output)
mod['func_229'] = func_229
mod = relay.transform.InferType()(mod)
mutated_mod['func_229'] = func_229
mutated_mod = relay.transform.InferType()(mutated_mod)
var_230 = relay.var("var_230", dtype = "float64", shape = (5, 7, 13))#candidate|230|(5, 7, 13)|var|float64
func_229_call = mutated_mod.get_global_var('func_229')
call_231 = func_229_call(var_230)
output = call_231
func_232 = relay.Function([var_230], output)
mutated_mod['func_232'] = func_232
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_234 = relay.TupleGetItem(func_193_call(), 2)
call_235 = relay.TupleGetItem(func_194_call(), 2)
var_236 = relay.var("var_236", dtype = "uint64", shape = (5, 7, 13))#candidate|236|(5, 7, 13)|var|uint64
bop_237 = relay.not_equal(call_234.astype('bool'), relay.reshape(var_236.astype('bool'), relay.shape_of(call_234))) # shape=(5, 7, 13)
bop_240 = relay.not_equal(call_235.astype('bool'), relay.reshape(var_236.astype('bool'), relay.shape_of(call_235))) # shape=(5, 7, 13)
uop_241 = relay.tan(call_234.astype('float32')) # shape=(5, 7, 13)
uop_243 = relay.tan(call_235.astype('float32')) # shape=(5, 7, 13)
output = relay.Tuple([bop_237,uop_241,])
output2 = relay.Tuple([bop_240,uop_243,])
func_244 = relay.Function([var_236,], output)
mod['func_244'] = func_244
mod = relay.transform.InferType()(mod)
mutated_mod['func_244'] = func_244
mutated_mod = relay.transform.InferType()(mutated_mod)
var_245 = relay.var("var_245", dtype = "uint64", shape = (5, 7, 13))#candidate|245|(5, 7, 13)|var|uint64
func_244_call = mutated_mod.get_global_var('func_244')
call_246 = func_244_call(var_245)
output = call_246
func_247 = relay.Function([var_245], output)
mutated_mod['func_247'] = func_247
mutated_mod = relay.transform.InferType()(mutated_mod)
var_249 = relay.var("var_249", dtype = "float32", shape = ())#candidate|249|()|var|float32
const_250 = relay.const([[5.976019,8.835744,7.098546],[7.063362,-2.283626,5.619975],[-8.516249,-1.728904,-2.465076],[-2.042616,-2.940559,-5.757656],[7.036228,5.438903,2.985291],[-9.701809,-7.981897,-3.467068],[0.579786,1.946061,-6.748491],[0.252418,0.525808,2.612946],[-6.826422,-4.661900,-3.642578],[6.649252,-4.154745,-6.728502],[6.817481,4.512812,-3.975416],[1.117565,5.383405,-4.414093],[-8.264786,-1.468437,8.981968],[5.847699,-7.059116,-1.224735],[-6.327292,-2.650920,-3.601266]], dtype = "float32")#candidate|250|(15, 3)|const|float32
bop_251 = relay.subtract(var_249.astype('float32'), const_250.astype('float32')) # shape=(15, 3)
uop_254 = relay.sqrt(bop_251.astype('float64')) # shape=(15, 3)
uop_256 = relay.sigmoid(bop_251.astype('float64')) # shape=(15, 3)
uop_258 = relay.acos(uop_256.astype('float64')) # shape=(15, 3)
bop_260 = relay.right_shift(uop_258.astype('uint8'), relay.reshape(bop_251.astype('uint8'), relay.shape_of(uop_258))) # shape=(15, 3)
bop_263 = relay.equal(bop_260.astype('bool'), relay.reshape(uop_254.astype('bool'), relay.shape_of(bop_260))) # shape=(15, 3)
var_266 = relay.var("var_266", dtype = "uint8", shape = (15, 3))#candidate|266|(15, 3)|var|uint8
bop_267 = relay.less(bop_260.astype('bool'), relay.reshape(var_266.astype('bool'), relay.shape_of(bop_260))) # shape=(15, 3)
bop_270 = relay.greater_equal(bop_260.astype('bool'), relay.reshape(uop_256.astype('bool'), relay.shape_of(bop_260))) # shape=(15, 3)
uop_273 = relay.sinh(bop_270.astype('float64')) # shape=(15, 3)
output = relay.Tuple([bop_263,bop_267,uop_273,])
output2 = relay.Tuple([bop_263,bop_267,uop_273,])
func_275 = relay.Function([var_249,var_266,], output)
mod['func_275'] = func_275
mod = relay.transform.InferType()(mod)
mutated_mod['func_275'] = func_275
mutated_mod = relay.transform.InferType()(mutated_mod)
func_275_call = mutated_mod.get_global_var('func_275')
var_277 = relay.var("var_277", dtype = "float32", shape = ())#candidate|277|()|var|float32
var_278 = relay.var("var_278", dtype = "uint8", shape = (15, 3))#candidate|278|(15, 3)|var|uint8
call_276 = func_275_call(var_277,var_278,)
output = call_276
func_279 = relay.Function([var_277,var_278,], output)
mutated_mod['func_279'] = func_279
mutated_mod = relay.transform.InferType()(mutated_mod)
func_193_call = mod.get_global_var('func_193')
func_194_call = mutated_mod.get_global_var('func_194')
call_281 = relay.TupleGetItem(func_193_call(), 1)
call_282 = relay.TupleGetItem(func_194_call(), 1)
var_283 = relay.var("var_283", dtype = "bool", shape = (5, 7, 13))#candidate|283|(5, 7, 13)|var|bool
bop_284 = relay.subtract(call_281.astype('int64'), relay.reshape(var_283.astype('int64'), relay.shape_of(call_281))) # shape=(5, 7, 13)
bop_287 = relay.subtract(call_282.astype('int64'), relay.reshape(var_283.astype('int64'), relay.shape_of(call_282))) # shape=(5, 7, 13)
bop_288 = relay.floor_divide(bop_284.astype('float64'), relay.reshape(var_283.astype('float64'), relay.shape_of(bop_284))) # shape=(5, 7, 13)
bop_291 = relay.floor_divide(bop_287.astype('float64'), relay.reshape(var_283.astype('float64'), relay.shape_of(bop_287))) # shape=(5, 7, 13)
uop_292 = relay.cos(bop_284.astype('float32')) # shape=(5, 7, 13)
uop_294 = relay.cos(bop_287.astype('float32')) # shape=(5, 7, 13)
uop_295 = relay.acos(uop_292.astype('float32')) # shape=(5, 7, 13)
uop_297 = relay.acos(uop_294.astype('float32')) # shape=(5, 7, 13)
uop_298 = relay.log2(uop_295.astype('float64')) # shape=(5, 7, 13)
uop_300 = relay.log2(uop_297.astype('float64')) # shape=(5, 7, 13)
bop_301 = relay.equal(uop_295.astype('bool'), relay.reshape(bop_284.astype('bool'), relay.shape_of(uop_295))) # shape=(5, 7, 13)
bop_304 = relay.equal(uop_297.astype('bool'), relay.reshape(bop_287.astype('bool'), relay.shape_of(uop_297))) # shape=(5, 7, 13)
uop_305 = relay.cosh(bop_301.astype('float32')) # shape=(5, 7, 13)
uop_307 = relay.cosh(bop_304.astype('float32')) # shape=(5, 7, 13)
uop_308 = relay.atanh(uop_292.astype('float64')) # shape=(5, 7, 13)
uop_310 = relay.atanh(uop_294.astype('float64')) # shape=(5, 7, 13)
func_212_call = mod.get_global_var('func_212')
func_216_call = mutated_mod.get_global_var('func_216')
var_312 = relay.var("var_312", dtype = "float32", shape = (120,))#candidate|312|(120,)|var|float32
call_311 = relay.TupleGetItem(func_212_call(relay.reshape(var_312.astype('float32'), [12, 10]), relay.reshape(var_312.astype('float32'), [12, 10]), ), 1)
call_313 = relay.TupleGetItem(func_216_call(relay.reshape(var_312.astype('float32'), [12, 10]), relay.reshape(var_312.astype('float32'), [12, 10]), ), 1)
const_314 = relay.const([[[8.307813,8.409128,9.677729,8.746770,3.248456,-8.031886,8.680576,9.198772,1.683878,9.414885,-1.505919,5.150804,-9.348185],[-7.826897,-8.887125,1.040225,-5.583094,-6.904811,4.166041,-1.322085,-6.494184,-9.075693,9.317501,-2.833877,7.527677,2.532561],[-9.969492,9.464849,-2.724729,-0.487843,1.667319,-4.553981,-2.531743,8.386893,-5.830837,-9.739408,-4.913680,-2.312075,9.976770],[1.636550,5.349024,-5.814454,3.586445,-0.141832,3.599722,-2.054225,-7.224320,-5.317090,6.258993,-8.881225,-8.655955,3.423597],[-9.296455,-9.605768,-4.277390,-8.159459,3.033991,2.021386,-3.543664,-4.049995,-7.921889,-5.374166,4.129801,3.762779,-4.007036],[6.146668,0.481916,-3.391191,-0.389887,9.538571,-7.764956,-6.656399,-1.152549,-0.564510,-9.521939,6.520884,-5.494279,9.498722],[-2.777504,-0.593015,9.581414,4.044459,-0.694650,-6.045654,2.553181,1.340699,4.544352,9.323867,-7.013273,-6.020947,-0.289077]],[[1.544618,4.789367,-4.014445,-8.105314,-4.199824,-4.934252,-8.842469,3.636338,-2.844550,-3.311546,-6.751213,6.142162,9.578037],[-8.541718,-3.513687,3.926409,6.099873,9.005603,6.999152,-6.039787,6.478150,1.423221,-3.996024,-7.009705,5.010162,-6.987035],[-1.040679,1.935355,9.339085,8.334735,-0.498143,-1.456703,4.869617,-7.068890,4.401363,-5.305454,0.325593,-1.312349,0.185098],[-5.436290,6.025507,-2.876863,3.134630,1.025985,0.382030,-9.957535,-0.406127,7.720271,-8.317193,-7.954506,8.025820,-5.733517],[-1.933793,-5.876634,8.729347,6.472752,6.291622,-8.194245,9.809917,-7.966175,-6.724544,0.646281,7.810169,-4.165653,-5.112092],[7.893034,-9.687220,-9.314741,-0.850746,-2.350929,-4.484765,-3.716650,5.005191,-5.498444,-3.993980,-6.314670,8.654614,2.518958],[-4.153647,3.959723,-6.183315,-9.220195,-4.851915,-5.372559,1.599398,1.425821,-4.735366,-2.108704,-2.199329,-9.127342,1.986450]],[[-2.825870,-3.722327,9.288539,0.450802,1.254319,7.508924,-2.247689,2.904118,-8.590008,0.490243,-3.455498,3.246902,-5.711352],[8.621829,-3.638848,7.843219,4.171325,8.362901,-3.848730,6.776612,-0.959374,-1.258917,4.615075,0.130628,-7.288650,1.056487],[-0.834015,9.400114,-3.885660,-6.527624,0.023512,6.644502,3.780013,3.387546,-1.985130,-1.879578,8.698628,4.676096,-4.221432],[-7.201669,-9.912575,0.331868,-3.136408,7.439338,1.289908,-9.217985,4.591013,7.259425,-9.565309,-8.727428,-3.685106,1.828338],[4.145558,-2.200006,-4.037938,7.321719,-8.409718,1.567658,6.468940,0.414886,-9.273657,-2.149789,-9.066546,3.890009,-6.277666],[6.530491,-9.390208,9.403280,0.078133,5.257300,-4.157153,-4.526894,4.729792,-0.756635,-8.260231,7.685922,0.584105,-5.254576],[2.671907,-5.074815,-9.779656,-4.605575,-0.590511,-0.336973,0.470919,1.187589,-6.380987,-1.576834,-3.214563,8.080921,6.569635]],[[4.873718,-8.173338,-1.297614,4.111722,9.190812,-6.536234,-5.260282,5.245220,-2.062716,-5.023000,5.863952,-5.596253,-2.928383],[-2.330065,6.293809,-2.430460,-8.249268,6.755669,4.630523,3.210348,-1.751189,-4.073250,-1.964480,7.890249,9.433164,0.673326],[9.027680,-1.846883,-3.009299,4.450480,-7.323079,-7.878851,-5.725184,3.639743,3.041505,-0.267758,-5.170748,-5.791288,7.642441],[-4.029774,-6.561894,1.107159,8.383867,2.023266,-9.746121,8.472868,7.034830,6.250258,3.414520,9.805702,7.020253,-7.143041],[-9.822098,-3.659786,8.612367,-2.664455,-0.435755,2.656745,-8.281941,9.277721,2.701137,3.391592,6.547102,-0.657419,-3.291354],[9.130889,4.323915,4.665629,-7.679780,1.256185,-2.313758,5.571159,6.724458,-2.901548,-6.944294,-8.236911,6.124334,-5.063508],[-8.612205,3.192443,-7.794982,-1.290853,9.165149,7.721940,-9.935330,7.839443,5.328774,1.252987,9.801579,-5.160193,-6.362675]],[[8.333070,-6.976081,-4.997940,-8.169407,2.783230,-4.047762,-2.132087,-7.371592,9.245029,4.278261,6.304878,-2.419554,4.598904],[8.357155,9.912997,8.063745,5.443218,-4.394128,-3.985768,6.384308,3.664330,-5.200825,9.131314,0.944939,9.285519,0.217619],[2.336920,5.743302,2.029250,-7.929510,-8.051110,7.837423,-5.826901,3.677981,-2.458639,-1.469791,-1.803382,-6.011692,8.365330],[9.531724,-0.835783,-4.295298,-0.247302,-5.607267,-1.581777,-2.910138,-4.670534,-5.483935,1.012678,-7.249400,-3.583662,2.037094],[8.977877,8.268647,3.863366,7.462979,1.223618,-5.752152,2.555402,-1.345556,1.174020,7.500804,-7.256649,-9.413116,5.483295],[-6.360294,-2.760295,2.412459,7.759958,-7.116360,7.523711,-0.836597,4.862127,-3.365492,1.527200,-2.435297,9.108801,5.650676],[7.706327,8.461672,-2.010067,8.811174,5.144192,-8.653767,9.848682,7.110394,-9.713002,4.115387,-7.798711,-2.439456,-1.233133]]], dtype = "float64")#candidate|314|(5, 7, 13)|const|float64
bop_315 = relay.power(uop_298.astype('float64'), relay.reshape(const_314.astype('float64'), relay.shape_of(uop_298))) # shape=(5, 7, 13)
bop_318 = relay.power(uop_300.astype('float64'), relay.reshape(const_314.astype('float64'), relay.shape_of(uop_300))) # shape=(5, 7, 13)
uop_319 = relay.sigmoid(uop_305.astype('float32')) # shape=(5, 7, 13)
uop_321 = relay.sigmoid(uop_307.astype('float32')) # shape=(5, 7, 13)
uop_322 = relay.sigmoid(uop_319.astype('float64')) # shape=(5, 7, 13)
uop_324 = relay.sigmoid(uop_321.astype('float64')) # shape=(5, 7, 13)
bop_325 = relay.logical_and(uop_322.astype('bool'), relay.reshape(uop_298.astype('bool'), relay.shape_of(uop_322))) # shape=(5, 7, 13)
bop_328 = relay.logical_and(uop_324.astype('bool'), relay.reshape(uop_300.astype('bool'), relay.shape_of(uop_324))) # shape=(5, 7, 13)
bop_329 = relay.greater(uop_298.astype('bool'), relay.reshape(const_314.astype('bool'), relay.shape_of(uop_298))) # shape=(5, 7, 13)
bop_332 = relay.greater(uop_300.astype('bool'), relay.reshape(const_314.astype('bool'), relay.shape_of(uop_300))) # shape=(5, 7, 13)
uop_333 = relay.cosh(bop_325.astype('float32')) # shape=(5, 7, 13)
uop_335 = relay.cosh(bop_328.astype('float32')) # shape=(5, 7, 13)
uop_336 = relay.atanh(uop_333.astype('float64')) # shape=(5, 7, 13)
uop_338 = relay.atanh(uop_335.astype('float64')) # shape=(5, 7, 13)
bop_339 = relay.logical_or(bop_325.astype('bool'), relay.reshape(uop_308.astype('bool'), relay.shape_of(bop_325))) # shape=(5, 7, 13)
bop_342 = relay.logical_or(bop_328.astype('bool'), relay.reshape(uop_310.astype('bool'), relay.shape_of(bop_328))) # shape=(5, 7, 13)
bop_343 = relay.right_shift(uop_336.astype('int16'), relay.reshape(bop_288.astype('int16'), relay.shape_of(uop_336))) # shape=(5, 7, 13)
bop_346 = relay.right_shift(uop_338.astype('int16'), relay.reshape(bop_291.astype('int16'), relay.shape_of(uop_338))) # shape=(5, 7, 13)
bop_347 = relay.floor_divide(uop_322.astype('float64'), relay.reshape(bop_325.astype('float64'), relay.shape_of(uop_322))) # shape=(5, 7, 13)
bop_350 = relay.floor_divide(uop_324.astype('float64'), relay.reshape(bop_328.astype('float64'), relay.shape_of(uop_324))) # shape=(5, 7, 13)
bop_351 = relay.equal(uop_333.astype('bool'), relay.reshape(uop_298.astype('bool'), relay.shape_of(uop_333))) # shape=(5, 7, 13)
bop_354 = relay.equal(uop_335.astype('bool'), relay.reshape(uop_300.astype('bool'), relay.shape_of(uop_335))) # shape=(5, 7, 13)
const_355 = relay.const([[[-2,-7,8,2,-3,5,-2,5,-3,3,4,9,9],[-2,9,-4,-6,5,1,10,-8,3,9,9,-4,3],[-4,-5,8,-9,5,-7,-4,-5,-2,2,9,-9,3],[10,10,-7,-8,9,-2,-3,4,7,2,7,-6,3],[4,6,2,7,1,-8,8,5,6,6,9,9,5],[7,-3,-1,-4,6,-6,8,9,-6,-1,-9,-10,10],[10,1,5,1,10,-7,-10,10,6,-3,4,-5,10]],[[-4,3,-8,-5,-2,-10,-10,-4,5,-10,10,-5,-4],[-10,-3,-5,1,-3,10,-1,-8,-5,1,5,-3,-3],[4,6,6,-7,6,-6,-1,-5,6,-3,-7,5,8],[-4,-7,-6,6,9,7,10,1,8,6,-2,1,9],[4,4,3,6,6,1,3,-3,-7,2,9,-9,-7],[-9,4,4,1,-3,5,-6,-1,-4,-8,-5,-10,-4],[-8,3,-2,4,10,8,-6,-8,2,4,1,-10,-5]],[[-6,3,-3,3,-8,8,-4,8,10,3,-6,1,8],[10,-3,-2,-1,10,-8,9,9,6,8,3,9,-5],[1,-7,-2,-9,-1,3,-6,5,8,-9,-3,-10,10],[-8,-1,6,-3,-3,9,7,2,1,4,3,-1,-9],[-9,5,2,10,8,-7,10,-5,3,10,-3,5,-8],[1,3,4,-1,-1,4,-8,9,3,-5,-10,-2,6],[-3,7,1,3,-10,-4,-7,5,6,4,-9,5,-7]],[[4,7,-5,1,4,-9,-4,5,1,-4,-3,-10,-6],[-4,-4,-7,-8,-8,-5,-4,1,-9,4,6,5,-2],[2,5,-2,10,2,6,-2,-5,-5,7,4,-2,9],[6,10,4,1,-10,2,10,-7,-4,-1,5,9,4],[1,4,6,6,7,5,-3,2,5,9,-8,-1,1],[-7,-2,-6,6,3,-7,7,-10,3,-9,1,6,-6],[-3,3,6,6,10,-9,1,10,2,3,10,-7,2]],[[-3,-8,-3,7,-4,-6,9,2,2,-3,-1,-10,8],[4,-8,-5,-5,9,3,5,-9,10,7,-9,7,-3],[-6,5,-4,-1,-5,3,-9,-9,-10,-2,5,8,-5],[-5,8,1,3,-8,-4,7,-5,-1,-5,-9,-5,4],[2,-4,-9,-7,8,-1,-5,8,-9,-1,7,3,2],[-7,-2,7,-4,-10,3,-10,-7,-2,1,4,-4,6],[-6,-3,8,5,2,-5,6,-9,-3,10,1,4,1]]], dtype = "int16")#candidate|355|(5, 7, 13)|const|int16
bop_356 = relay.minimum(bop_343.astype('float32'), relay.reshape(const_355.astype('float32'), relay.shape_of(bop_343))) # shape=(5, 7, 13)
bop_359 = relay.minimum(bop_346.astype('float32'), relay.reshape(const_355.astype('float32'), relay.shape_of(bop_346))) # shape=(5, 7, 13)
bop_360 = relay.equal(bop_347.astype('bool'), relay.reshape(bop_301.astype('bool'), relay.shape_of(bop_347))) # shape=(5, 7, 13)
bop_363 = relay.equal(bop_350.astype('bool'), relay.reshape(bop_304.astype('bool'), relay.shape_of(bop_350))) # shape=(5, 7, 13)
const_364 = relay.const([[[-8.649545,6.376114,-7.110308,7.107728,9.997043,-1.812768,1.289107,1.018644,-7.770396,-1.145866,-3.980563,-7.376900,-9.326917],[1.139524,2.026910,-9.969649,-4.342880,-2.813954,7.985479,2.878928,-0.137974,2.676895,-0.689131,-8.543901,6.304482,-9.898985],[-9.603143,6.810881,1.971482,-1.616039,0.894645,-4.044418,-8.347375,-4.969645,-9.383555,4.819534,-1.333204,-5.452113,9.177155],[2.790088,9.249718,-6.512978,-1.664759,-7.130954,7.601603,-4.866990,7.677157,-6.418005,6.390680,-9.745178,-0.116781,3.994147],[-8.285674,-6.184406,-3.560198,-0.705992,2.320529,8.703842,-8.228442,1.610469,-8.673302,6.119475,-3.523809,-7.220094,-1.860243],[-9.818810,-7.993617,6.641266,7.856554,-4.223835,0.927601,-0.399785,-8.185666,0.374354,8.019228,-4.245140,-6.706992,-4.137110],[7.634930,0.716595,3.499114,-4.392854,3.570885,-7.982787,5.415302,-9.372529,0.595608,-4.709946,3.434777,-5.620027,-7.842443]],[[-8.792913,-4.398926,-3.643282,5.971284,-7.913565,9.597195,-4.178340,-2.130679,2.771502,-2.796699,8.408729,-7.412370,7.990754],[9.745092,7.083278,-0.990698,-6.578009,-0.725286,-2.180431,-4.391452,-4.613916,7.946831,-8.021373,5.866277,-8.937281,-4.049882],[7.799825,1.160919,-6.007274,-7.045488,-2.054197,5.699923,-6.923681,2.243046,-7.403458,7.543859,3.336211,4.052811,-3.400164],[-6.483443,-7.243764,-7.917819,5.446953,-5.933608,-7.742138,2.134133,8.369502,-3.762591,-5.899582,-1.797847,9.676924,3.888071],[9.664967,6.815444,6.984630,-8.728690,-9.429189,5.661715,-2.567284,2.647809,4.393083,0.513810,4.179304,-0.633037,-2.193742],[-9.219379,-6.085689,-7.663367,4.904058,4.292210,5.433626,-0.479997,1.325614,-4.542144,-6.098108,-7.614121,-5.961593,-2.344233],[8.591835,-7.576286,-1.688181,-1.885864,-9.564016,-6.409242,-2.182159,3.536129,7.180507,3.915041,6.377337,-6.397857,5.845585]],[[9.305974,-7.759825,6.304632,-3.662935,8.330655,1.038913,5.787891,2.168923,5.773095,-0.222194,4.094589,-3.633744,-3.812259],[-4.827938,9.885093,8.002933,-5.780435,4.368918,-3.554785,6.948838,-0.265039,9.961901,-8.110728,3.554328,-6.625741,-7.479385],[4.974828,-3.948080,0.262865,-4.553219,-0.193205,5.415262,9.726660,-5.184995,3.304005,-0.491041,-4.799962,9.487497,-2.191260],[-6.238989,-0.516740,-2.913220,-7.554327,8.981028,-7.096284,-5.597880,-8.354832,4.069171,-4.331166,4.215209,2.918112,9.568290],[3.890218,-9.600922,5.786489,-8.330005,-4.656334,9.657880,6.122077,-8.521220,-8.718251,-2.652734,-9.306813,0.427828,5.332167],[-0.498590,-4.677578,3.675458,2.959867,0.549048,-3.875846,-3.909048,-6.208485,9.906574,4.128253,5.622847,3.450043,-6.314382],[3.300319,-8.543683,-1.585951,8.573863,8.542746,-4.424850,3.424027,0.158972,6.732014,-2.082071,-0.825544,3.831339,0.750518]],[[0.853008,-9.023561,-0.668918,7.245784,-3.407183,-9.519039,-9.637189,-7.764589,-0.805142,-5.698485,-7.483498,3.286131,-9.137077],[-7.685146,5.532804,-6.305196,-8.729989,-2.498837,8.313080,-7.440325,7.222456,1.225343,-6.628832,-5.864714,2.911306,-3.889269],[-1.938156,-7.707575,0.786981,-6.093135,-9.314839,-7.659242,3.580573,-6.368178,-0.739111,1.987838,0.032434,-9.657255,-5.979107],[-6.966102,4.926015,9.477822,-1.538680,-6.170257,5.159153,9.647220,-9.783547,-3.988359,-0.061252,-1.701526,3.715414,1.862876],[-7.851165,-6.450237,-5.785320,0.143105,-3.138405,-1.145199,-0.006297,-4.313991,6.539754,-7.370890,5.792672,-2.707991,8.486864],[9.829701,-5.024777,2.545693,-3.044231,1.690258,-6.818606,-5.833409,-5.208351,1.436559,-5.333613,-5.184008,6.534233,9.476020],[0.300972,0.076745,-8.027530,-4.318544,4.955131,2.459660,5.820081,-1.751054,7.534227,-7.885088,5.014037,0.340766,-3.955323]],[[-4.637212,-0.903359,-0.475905,4.962130,-1.400748,-7.560378,-6.947750,-4.833358,2.502578,-8.477289,3.140511,5.531702,0.064647],[7.136059,-6.273946,-9.685778,-3.835359,-3.107073,3.210826,-6.469998,5.237908,-6.536004,-0.523497,6.429645,-1.265927,7.477818],[-9.898451,-0.879624,-7.615894,-4.619727,8.425943,3.141140,9.474541,-0.390550,5.141124,-1.934170,-2.068322,-5.829477,1.420155],[-2.486791,2.915210,0.533403,-5.975641,-2.307189,0.069665,6.316749,9.461939,-3.075583,-8.620687,5.109101,5.444727,-2.362648],[-7.513881,-8.152037,9.851295,-0.840895,4.486372,-3.233032,9.928530,-1.047717,2.596083,1.447871,-7.290310,6.360776,-5.723489],[-3.593748,-6.901147,7.218352,3.705833,-4.936226,5.532728,2.321394,-7.159850,-3.747041,-7.576802,-7.901528,1.247284,6.749789],[-9.709095,-5.465368,8.876963,-0.123879,-3.370363,5.096425,-7.479923,-3.923255,-3.256966,-8.124271,1.281425,-2.093204,1.345643]]], dtype = "float32")#candidate|364|(5, 7, 13)|const|float32
bop_365 = relay.less_equal(uop_292.astype('bool'), relay.reshape(const_364.astype('bool'), relay.shape_of(uop_292))) # shape=(5, 7, 13)
bop_368 = relay.less_equal(uop_294.astype('bool'), relay.reshape(const_364.astype('bool'), relay.shape_of(uop_294))) # shape=(5, 7, 13)
var_369 = relay.var("var_369", dtype = "float32", shape = (5, 7, 13))#candidate|369|(5, 7, 13)|var|float32
bop_370 = relay.not_equal(uop_333.astype('bool'), relay.reshape(var_369.astype('bool'), relay.shape_of(uop_333))) # shape=(5, 7, 13)
bop_373 = relay.not_equal(uop_335.astype('bool'), relay.reshape(var_369.astype('bool'), relay.shape_of(uop_335))) # shape=(5, 7, 13)
output = relay.Tuple([call_311,var_312,bop_315,bop_329,bop_339,bop_351,bop_356,bop_360,bop_365,bop_370,])
output2 = relay.Tuple([call_313,var_312,bop_318,bop_332,bop_342,bop_354,bop_359,bop_363,bop_368,bop_373,])
func_374 = relay.Function([var_283,var_312,var_369,], output)
mod['func_374'] = func_374
mod = relay.transform.InferType()(mod)
mutated_mod['func_374'] = func_374
mutated_mod = relay.transform.InferType()(mutated_mod)
func_374_call = mutated_mod.get_global_var('func_374')
var_376 = relay.var("var_376", dtype = "bool", shape = (5, 7, 13))#candidate|376|(5, 7, 13)|var|bool
var_377 = relay.var("var_377", dtype = "float32", shape = (120,))#candidate|377|(120,)|var|float32
var_378 = relay.var("var_378", dtype = "float32", shape = (5, 7, 13))#candidate|378|(5, 7, 13)|var|float32
call_375 = func_374_call(var_376,var_377,var_378,)
output = call_375
func_379 = relay.Function([var_376,var_377,var_378,], output)
mutated_mod['func_379'] = func_379
mutated_mod = relay.transform.InferType()(mutated_mod)
const_381 = relay.const([[[2.062367,-4.982207,3.170825,4.074404,-8.408552,-1.726381,1.603491,5.269710,8.793407],[-1.910939,-7.162735,7.497086,6.267741,-3.253172,6.992425,-0.225769,8.162413,9.243663],[5.746670,3.083897,-6.729501,-0.943141,7.812757,-8.680583,3.508550,0.997830,4.147794],[2.873362,0.196286,7.717158,-7.843891,0.247011,2.887358,-2.320824,-3.450668,-1.176104],[7.165432,-2.569122,-2.943060,5.563886,-8.575860,0.172317,8.316500,-3.735691,-8.194060],[5.185583,-2.876250,-7.933305,0.441919,-7.457977,-2.150894,-0.224834,1.297466,-2.398774],[3.489765,1.040364,-4.491609,8.484573,0.118386,1.867625,6.433608,-2.726446,9.097775]],[[6.904837,5.360372,-2.564575,-1.154767,4.178523,-7.023127,2.194078,7.139043,5.249646],[3.983094,9.458291,-9.647185,-6.921498,9.977593,8.088493,9.452745,0.584784,-1.175386],[-6.282270,-1.839644,-3.708923,7.872198,-4.583690,-5.208169,3.144226,-0.496983,9.539814],[6.802133,8.056532,-3.723139,4.097510,3.001535,2.194619,2.239337,8.717508,3.899080],[-9.879057,2.789921,-3.599412,-4.716098,-5.843407,-3.331722,2.683519,2.827808,5.583740],[-3.193332,-4.637547,1.864531,2.474517,1.342884,1.393321,3.577626,-8.259625,-1.574062],[-8.544976,-4.970273,-6.081183,-3.054269,-0.785411,-0.278870,-7.949507,-4.496462,-2.700909]],[[-9.807127,-1.078933,-5.157124,-2.312045,2.039128,1.270509,7.687848,-8.204523,-5.299592],[1.789524,6.806545,-0.993962,3.923034,-4.685448,7.844796,1.800136,-7.902091,-0.934365],[1.183531,-3.337462,6.762339,3.989524,-3.846988,-5.962309,-1.988510,7.195142,2.114485],[0.308268,-0.993463,-3.146888,-0.285160,4.866690,-7.592796,-1.157958,0.609990,-3.048010],[-9.831007,5.790985,-8.556119,3.131035,8.667730,-1.826078,-0.934686,6.511494,5.627059],[2.607601,-3.722275,-9.975426,-6.943316,-9.855508,-3.315324,-4.792446,-6.584449,8.623508],[8.265813,1.937775,9.321855,7.214956,0.002986,4.671542,-3.415107,5.068099,7.071822]]], dtype = "float64")#candidate|381|(3, 7, 9)|const|float64
uop_382 = relay.cos(const_381.astype('float64')) # shape=(3, 7, 9)
bop_384 = relay.logical_and(uop_382.astype('bool'), relay.reshape(const_381.astype('bool'), relay.shape_of(uop_382))) # shape=(3, 7, 9)
const_387 = relay.const([[[True,True,True,False,False,False,True,False,False],[True,False,True,True,True,True,True,True,True],[True,False,False,True,False,True,False,False,False],[False,True,False,False,True,True,False,True,False],[False,True,False,True,False,False,False,False,True],[True,True,False,False,False,False,False,False,False],[True,False,False,False,True,False,False,True,True]],[[False,True,False,False,False,True,False,True,False],[False,True,False,False,False,True,False,False,True],[True,True,True,True,False,False,False,False,True],[False,True,False,True,True,False,True,True,False],[True,True,True,True,True,False,True,True,False],[False,False,False,True,True,True,True,False,False],[True,False,False,False,False,True,False,True,False]],[[False,True,True,False,False,False,True,True,False],[True,False,True,True,False,True,False,True,False],[True,False,True,True,True,True,False,True,False],[True,False,True,False,False,False,True,True,False],[False,False,True,False,False,True,True,True,True],[True,True,True,True,True,True,False,False,True],[False,False,False,False,False,False,True,False,True]]], dtype = "bool")#candidate|387|(3, 7, 9)|const|bool
bop_388 = relay.greater_equal(bop_384.astype('bool'), relay.reshape(const_387.astype('bool'), relay.shape_of(bop_384))) # shape=(3, 7, 9)
uop_391 = relay.cos(uop_382.astype('float64')) # shape=(3, 7, 9)
output = relay.Tuple([bop_388,uop_391,])
output2 = relay.Tuple([bop_388,uop_391,])
F = relay.Function([], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([], output2)
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
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()()
res3 = intrp3.evaluate()()
res4 = intrp4.evaluate()()
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
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()()
res7 = intrp7.evaluate()()
res8 = intrp8.evaluate()()
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
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()()
res11 = intrp11.evaluate()()
res12 = intrp12.evaluate()()
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
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()()
res15 = intrp15.evaluate()()
res16 = intrp16.evaluate()()
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
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()()
res19 = intrp19.evaluate()()
res20 = intrp20.evaluate()()
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
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()()
res23 = intrp23.evaluate()()
res24 = intrp24.evaluate()()
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