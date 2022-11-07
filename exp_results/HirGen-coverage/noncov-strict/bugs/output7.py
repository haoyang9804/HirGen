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
var_0 = relay.var("var_0", dtype = "float32", shape = (4,))#candidate|0|(4,)|var|float32
var_1 = relay.var("var_1", dtype = "float32", shape = (4,))#candidate|1|(4,)|var|float32
bop_2 = relay.add(var_0.astype('float32'), relay.reshape(var_1.astype('float32'), relay.shape_of(var_0))) # shape=(4,)
var_5 = relay.var("var_5", dtype = "float32", shape = (4,))#candidate|5|(4,)|var|float32
bop_6 = relay.power(var_0.astype('float64'), relay.reshape(var_5.astype('float64'), relay.shape_of(var_0))) # shape=(4,)
bop_9 = relay.add(var_0.astype('int16'), relay.reshape(bop_2.astype('int16'), relay.shape_of(var_0))) # shape=(4,)
bop_12 = relay.less(bop_9.astype('bool'), relay.reshape(bop_2.astype('bool'), relay.shape_of(bop_9))) # shape=(4,)
uop_15 = relay.tan(var_1.astype('float64')) # shape=(4,)
var_17 = relay.var("var_17", dtype = "float64", shape = (4,))#candidate|17|(4,)|var|float64
bop_18 = relay.add(uop_15.astype('uint16'), relay.reshape(var_17.astype('uint16'), relay.shape_of(uop_15))) # shape=(4,)
uop_21 = relay.asin(bop_18.astype('float32')) # shape=(4,)
bop_23 = relay.floor_divide(uop_21.astype('float64'), relay.reshape(uop_15.astype('float64'), relay.shape_of(uop_21))) # shape=(4,)
uop_26 = relay.atan(bop_23.astype('float64')) # shape=(4,)
uop_28 = relay.tan(var_1.astype('float32')) # shape=(4,)
bop_30 = relay.greater(uop_28.astype('bool'), relay.reshape(bop_18.astype('bool'), relay.shape_of(uop_28))) # shape=(4,)
bop_33 = relay.add(bop_23.astype('uint32'), relay.reshape(uop_15.astype('uint32'), relay.shape_of(bop_23))) # shape=(4,)
bop_36 = relay.logical_xor(uop_26.astype('int64'), relay.reshape(bop_18.astype('int64'), relay.shape_of(uop_26))) # shape=(4,)
uop_39 = relay.sqrt(uop_26.astype('float32')) # shape=(4,)
uop_41 = relay.asin(uop_26.astype('float64')) # shape=(4,)
bop_43 = relay.mod(uop_41.astype('float64'), relay.reshape(uop_39.astype('float64'), relay.shape_of(uop_41))) # shape=(4,)
const_46 = relay.const([9.197949,1.836000,-6.190715,-0.118410], dtype = "float64")#candidate|46|(4,)|const|float64
bop_47 = relay.divide(uop_26.astype('float64'), relay.reshape(const_46.astype('float64'), relay.shape_of(uop_26))) # shape=(4,)
uop_50 = relay.log10(uop_26.astype('float64')) # shape=(4,)
uop_52 = relay.erf(bop_18.astype('float64')) # shape=(4,)
bop_54 = relay.multiply(bop_33.astype('uint16'), relay.reshape(bop_36.astype('uint16'), relay.shape_of(bop_33))) # shape=(4,)
bop_57 = relay.logical_and(uop_50.astype('bool'), relay.reshape(bop_47.astype('bool'), relay.shape_of(uop_50))) # shape=(4,)
const_60 = relay.const([8.900553,-6.643442,-3.890035,9.408337], dtype = "float64")#candidate|60|(4,)|const|float64
bop_61 = relay.logical_and(uop_41.astype('bool'), relay.reshape(const_60.astype('bool'), relay.shape_of(uop_41))) # shape=(4,)
uop_64 = relay.asinh(bop_36.astype('float64')) # shape=(4,)
uop_66 = relay.acos(uop_21.astype('float64')) # shape=(4,)
bop_68 = relay.left_shift(bop_57.astype('int32'), relay.reshape(uop_66.astype('int32'), relay.shape_of(bop_57))) # shape=(4,)
uop_71 = relay.cos(bop_47.astype('float32')) # shape=(4,)
var_73 = relay.var("var_73", dtype = "float64", shape = (4,))#candidate|73|(4,)|var|float64
bop_74 = relay.greater(const_46.astype('bool'), relay.reshape(var_73.astype('bool'), relay.shape_of(const_46))) # shape=(4,)
uop_77 = relay.asinh(bop_23.astype('float64')) # shape=(4,)
bop_79 = relay.power(bop_57.astype('float64'), relay.reshape(const_46.astype('float64'), relay.shape_of(bop_57))) # shape=(4,)
bop_82 = relay.greater_equal(bop_79.astype('bool'), relay.reshape(bop_23.astype('bool'), relay.shape_of(bop_79))) # shape=(4,)
bop_85 = relay.divide(bop_23.astype('float64'), relay.reshape(bop_54.astype('float64'), relay.shape_of(bop_23))) # shape=(4,)
output = relay.Tuple([bop_6,bop_12,bop_30,bop_43,uop_52,bop_61,uop_64,bop_68,uop_71,bop_74,uop_77,bop_82,bop_85,])
output2 = relay.Tuple([bop_6,bop_12,bop_30,bop_43,uop_52,bop_61,uop_64,bop_68,uop_71,bop_74,uop_77,bop_82,bop_85,])
func_88 = relay.Function([var_0,var_1,var_5,var_17,var_73,], output)
mod['func_88'] = func_88
mod = relay.transform.InferType()(mod)
var_89 = relay.var("var_89", dtype = "float32", shape = (4,))#candidate|89|(4,)|var|float32
var_90 = relay.var("var_90", dtype = "float32", shape = (4,))#candidate|90|(4,)|var|float32
var_91 = relay.var("var_91", dtype = "float32", shape = (4,))#candidate|91|(4,)|var|float32
var_92 = relay.var("var_92", dtype = "float64", shape = (4,))#candidate|92|(4,)|var|float64
var_93 = relay.var("var_93", dtype = "float64", shape = (4,))#candidate|93|(4,)|var|float64
output = func_88(var_89,var_90,var_91,var_92,var_93,)
func_94 = relay.Function([var_89,var_90,var_91,var_92,var_93,], output)
mutated_mod['func_94'] = func_94
mutated_mod = relay.transform.InferType()(mutated_mod)
var_96 = relay.var("var_96", dtype = "float64", shape = (5, 8, 1))#candidate|96|(5, 8, 1)|var|float64
var_97 = relay.var("var_97", dtype = "float64", shape = (5, 8, 13))#candidate|97|(5, 8, 13)|var|float64
bop_98 = relay.floor_divide(var_96.astype('float64'), var_97.astype('float64')) # shape=(5, 8, 13)
bop_101 = relay.not_equal(var_96.astype('bool'), bop_98.astype('bool')) # shape=(5, 8, 13)
uop_104 = relay.atan(bop_101.astype('float64')) # shape=(5, 8, 13)
bop_106 = relay.add(uop_104.astype('uint8'), relay.reshape(bop_98.astype('uint8'), relay.shape_of(uop_104))) # shape=(5, 8, 13)
bop_109 = relay.greater_equal(var_96.astype('bool'), var_97.astype('bool')) # shape=(5, 8, 13)
output = relay.Tuple([bop_106,bop_109,])
output2 = relay.Tuple([bop_106,bop_109,])
func_112 = relay.Function([var_96,var_97,], output)
mod['func_112'] = func_112
mod = relay.transform.InferType()(mod)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
func_112_call = mutated_mod.get_global_var('func_112')
var_114 = relay.var("var_114", dtype = "float64", shape = (5, 8, 1))#candidate|114|(5, 8, 1)|var|float64
var_115 = relay.var("var_115", dtype = "float64", shape = (5, 8, 13))#candidate|115|(5, 8, 13)|var|float64
call_113 = func_112_call(var_114,var_115,)
output = call_113
func_116 = relay.Function([var_114,var_115,], output)
mutated_mod['func_116'] = func_116
mutated_mod = relay.transform.InferType()(mutated_mod)
const_118 = relay.const([[-8.606027,-5.894368,2.756710,2.072244,-2.660540,5.355227,-0.891992,8.821701,-6.964597,2.339765,-0.108845,0.312414,-7.141844],[-3.747269,3.600057,-7.070334,-4.843103,5.425396,-6.279539,4.503839,-2.643760,-5.974727,7.235946,-9.382805,-4.925700,3.290428],[-5.090918,-5.337409,4.666190,-2.935484,0.753933,1.885873,5.624242,9.897022,1.081397,1.621464,7.271520,-3.860256,4.892980],[-1.454095,3.191944,0.806288,-2.298083,-3.126143,-3.190795,6.785736,3.893633,8.662721,6.289887,4.866648,5.395326,3.997874],[-1.552213,2.933127,-2.904622,-5.512544,0.467892,9.786767,3.418568,-9.984641,8.859418,7.976691,-7.231036,1.443734,8.438164],[-5.615653,2.685162,-5.402275,0.790271,-4.854813,8.063048,8.748078,-5.611658,5.230722,-4.217570,5.276202,2.632542,7.334472],[7.081286,7.795995,0.568055,-2.679354,3.666969,-4.088957,-4.682601,-0.417387,7.019364,0.324487,5.448841,-8.930699,-7.271758]], dtype = "float64")#candidate|118|(7, 13)|const|float64
var_119 = relay.var("var_119", dtype = "float64", shape = (7, 13))#candidate|119|(7, 13)|var|float64
bop_120 = relay.floor_divide(const_118.astype('float64'), relay.reshape(var_119.astype('float64'), relay.shape_of(const_118))) # shape=(7, 13)
uop_123 = relay.atan(const_118.astype('float64')) # shape=(7, 13)
uop_125 = relay.sqrt(const_118.astype('float32')) # shape=(7, 13)
uop_127 = relay.sinh(uop_125.astype('float32')) # shape=(7, 13)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
var_130 = relay.var("var_130", dtype = "float64", shape = (40,))#candidate|130|(40,)|var|float64
const_131 = relay.const([[8.390581,-6.669221,1.726861,3.741136,7.817030,-0.824438,0.191116,3.159182,-5.816497,7.970552,6.925019,4.834993,-1.422254,-9.487399,-8.811532,-6.119202,0.948620,-7.510960,6.962015,3.748853,-5.839459,-3.662898,-0.747331,1.842803,-2.027929,-4.892690,2.657638,-5.263974,5.206794,-5.360812,-5.381880,1.252754,0.699153,-5.546942,0.230101,-5.526565,7.700468,9.228608,-2.189050,-2.608827,3.460911,-4.576032,-7.230329,7.475784,2.928658,-3.367683,-2.693435,3.808227,2.115371,1.710806,3.423215,-7.758884],[-0.257544,0.273626,5.640669,6.593755,-3.037806,5.590886,2.454451,-3.227344,-0.513351,-2.227273,-4.357658,-1.450327,-2.479334,-6.526246,6.105448,-5.884717,-6.994827,-8.125784,-1.635648,0.411584,1.934301,5.963845,2.021902,1.383888,-8.722212,-5.402494,-3.712177,8.407967,-1.536714,-5.164276,-2.974804,-5.994255,-7.083236,-7.442829,-6.203810,-6.358432,-0.030836,-4.665523,1.049135,0.416691,-9.757301,3.298743,5.412012,6.640819,-0.415654,0.589298,-4.991448,-0.324922,7.679777,0.460510,-5.974006,4.526690],[-1.679814,-2.223868,-4.573512,4.080652,8.139445,1.429755,2.992077,1.624216,-4.015667,3.284439,-8.395003,-1.916793,-6.148743,7.264450,6.187848,-8.472089,8.109176,6.344933,-4.936320,-8.040870,9.924048,-4.196914,-0.254444,0.325634,0.925671,6.534987,-8.645006,7.523498,-6.769662,5.879307,7.286249,1.081097,-5.074347,-3.835892,8.989510,-9.326398,-4.820991,-0.491046,-6.513022,-7.426738,-7.713443,7.262427,2.138855,5.516323,-8.273239,4.859232,-0.803928,-2.361712,6.205423,-8.239358,0.692204,-9.115411],[8.918980,-9.471783,-0.539542,4.689743,-4.996481,5.098849,3.718719,5.539932,6.116742,-6.027720,7.864950,-8.937716,8.805904,-0.975403,0.257796,8.674384,1.888154,8.765641,-0.266198,-9.263917,2.099126,-6.476537,-8.675663,-2.943631,-7.601725,-2.506976,6.409808,-7.368187,7.102268,1.458931,5.654422,7.241736,-0.387758,-2.026069,-5.193103,-5.109389,-4.416695,-2.325710,-0.832603,-2.953497,9.226552,1.272846,9.020337,-1.881288,-0.803471,-7.470640,-1.418941,-9.543898,-8.339560,9.662528,7.149906,-9.868477],[-4.750791,-6.034624,-5.577604,8.735583,8.458381,-7.575895,-8.433258,8.577876,7.642096,1.662636,-7.117069,-4.730872,-2.183518,5.026226,4.639640,-1.700044,-6.659026,1.824961,5.525060,2.805608,3.124522,-0.748642,-7.298588,-6.684906,2.306727,6.975372,8.725807,-4.181642,4.381873,8.382930,1.451480,-3.936946,-2.311939,-3.652262,-6.765185,3.712952,-4.487004,5.851892,8.684983,-3.366403,8.614057,-2.588294,5.891621,8.793434,-0.716153,-7.965081,-7.414141,-6.143197,-0.107151,-2.658686,-0.213319,-9.732544],[6.980637,2.697251,9.263978,6.437302,3.535827,2.662116,-2.147373,5.189749,4.881085,2.224311,-5.352162,-6.973878,-4.401751,9.947677,-8.863843,4.812151,4.049998,3.067699,-8.090993,-9.111273,-0.534587,4.758632,-3.680556,-6.702561,-2.083001,1.681365,6.120165,1.197325,-6.550569,-5.698861,-1.520122,8.299330,-2.400976,-4.631865,4.254383,-8.441022,-5.843857,1.923658,-6.335938,-0.866235,1.570137,2.038327,7.107169,-6.334798,0.820121,9.441181,9.347120,7.422504,8.659085,-8.398977,-6.576142,-0.709628],[-3.297515,4.899691,4.635974,4.974318,2.837356,1.430331,5.644122,0.868023,-0.316572,-8.824460,5.688479,4.252485,2.713718,2.616573,-8.624200,-6.732835,6.011522,2.494405,8.944469,3.312937,8.209841,-5.864701,1.979530,6.549662,-6.650666,7.317571,-9.126095,-9.206459,-6.349097,9.582070,3.840809,-0.938227,-8.814776,-7.665338,-1.498362,-1.378137,-8.927013,1.523789,-7.771486,-3.712185,-5.957782,-8.444106,-2.650314,3.873379,7.899333,-2.511582,-4.872280,8.685243,-0.281075,0.047139,-9.105530,1.306148],[-3.088122,-9.476182,1.960657,2.062835,-7.400253,-8.229661,-9.708093,9.299095,1.673027,-1.453185,-9.725944,-7.562825,6.538052,-3.958744,4.055078,0.006689,-0.424144,4.085916,-3.394541,2.532398,-8.411766,8.527760,1.300137,-3.479889,-2.821579,1.390740,2.974576,-5.260718,8.581955,-0.377801,-4.965873,-2.096938,7.625195,-8.862127,-4.566685,-9.868225,-2.415926,7.082734,-2.278915,-6.912682,-1.744992,-6.607753,8.646538,1.614424,3.479490,8.770355,2.338396,-4.791702,9.426970,-8.023225,-2.647786,0.416879],[-8.379664,8.132670,-9.642899,-1.979064,3.066606,-4.749486,5.758526,2.568351,1.889046,-7.370480,-8.304591,9.055175,6.453430,8.408943,-1.037207,-6.150855,1.583276,-5.451541,0.431063,-6.460438,3.180324,3.308641,-5.395565,2.486007,-7.422783,-9.999273,-5.291570,-6.341217,3.064526,1.969054,-6.321455,0.578600,6.101242,-3.020846,-5.454457,3.807431,5.117441,-4.140736,8.375804,-8.470584,9.684002,-9.274944,-5.233605,6.636303,-5.480940,-8.035768,4.172955,-7.196251,9.847357,2.348386,-6.437952,7.082276],[-2.743178,1.602947,6.101580,0.195673,2.736460,-3.038313,-0.714947,6.922319,-1.847272,-9.724578,6.216127,4.791876,9.143484,-7.191006,4.380709,2.927396,3.148551,-1.492930,-2.957840,1.878788,6.784763,-0.676843,-3.998777,3.528072,6.229562,-7.619048,-4.727359,7.580136,-8.166091,2.063114,2.509323,8.730771,-4.171590,7.941092,2.071090,-0.443511,-6.578431,5.706694,-4.738114,0.875358,-3.078202,0.603729,0.867031,-2.706335,-8.879385,-6.054537,-1.146888,5.428032,4.185427,-8.658977,4.609028,4.994929]], dtype = "float64")#candidate|131|(10, 52)|const|float64
call_129 = relay.TupleGetItem(func_112_call(relay.reshape(var_130.astype('float64'), [5, 8, 1]), relay.reshape(const_131.astype('float64'), [5, 8, 13]), ), 0)
call_132 = relay.TupleGetItem(func_116_call(relay.reshape(var_130.astype('float64'), [5, 8, 1]), relay.reshape(const_131.astype('float64'), [5, 8, 13]), ), 0)
bop_133 = relay.logical_or(uop_127.astype('bool'), relay.reshape(uop_125.astype('bool'), relay.shape_of(uop_127))) # shape=(7, 13)
uop_136 = relay.cosh(uop_127.astype('float64')) # shape=(7, 13)
uop_138 = relay.sin(uop_136.astype('float64')) # shape=(7, 13)
bop_140 = relay.logical_and(uop_138.astype('bool'), relay.reshape(bop_120.astype('bool'), relay.shape_of(uop_138))) # shape=(7, 13)
bop_143 = relay.subtract(bop_140.astype('int32'), relay.reshape(bop_133.astype('int32'), relay.shape_of(bop_140))) # shape=(7, 13)
bop_146 = relay.greater_equal(uop_138.astype('bool'), relay.reshape(bop_120.astype('bool'), relay.shape_of(uop_138))) # shape=(7, 13)
bop_149 = relay.less(uop_136.astype('bool'), relay.reshape(uop_125.astype('bool'), relay.shape_of(uop_136))) # shape=(7, 13)
func_88_call = mod.get_global_var('func_88')
func_94_call = mutated_mod.get_global_var('func_94')
const_153 = relay.const([8.130547,0.659481,1.849406,-7.326906], dtype = "float32")#candidate|153|(4,)|const|float32
call_152 = relay.TupleGetItem(func_88_call(relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float64'), [4,]), relay.reshape(const_153.astype('float64'), [4,]), ), 8)
call_154 = relay.TupleGetItem(func_94_call(relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float32'), [4,]), relay.reshape(const_153.astype('float64'), [4,]), relay.reshape(const_153.astype('float64'), [4,]), ), 8)
bop_155 = relay.divide(bop_140.astype('float32'), relay.reshape(uop_127.astype('float32'), relay.shape_of(bop_140))) # shape=(7, 13)
uop_158 = relay.sin(bop_140.astype('float32')) # shape=(7, 13)
bop_160 = relay.greater(uop_158.astype('bool'), relay.reshape(uop_123.astype('bool'), relay.shape_of(uop_158))) # shape=(7, 13)
bop_163 = relay.minimum(uop_138.astype('int32'), relay.reshape(bop_133.astype('int32'), relay.shape_of(uop_138))) # shape=(7, 13)
uop_166 = relay.asinh(bop_160.astype('float64')) # shape=(7, 13)
bop_168 = relay.greater_equal(uop_166.astype('bool'), relay.reshape(uop_125.astype('bool'), relay.shape_of(uop_166))) # shape=(7, 13)
uop_171 = relay.sinh(bop_140.astype('float64')) # shape=(7, 13)
var_173 = relay.var("var_173", dtype = "bool", shape = (7, 13))#candidate|173|(7, 13)|var|bool
bop_174 = relay.not_equal(bop_160.astype('bool'), relay.reshape(var_173.astype('bool'), relay.shape_of(bop_160))) # shape=(7, 13)
var_177 = relay.var("var_177", dtype = "float64", shape = (7, 13))#candidate|177|(7, 13)|var|float64
bop_178 = relay.bitwise_or(uop_166.astype('int32'), relay.reshape(var_177.astype('int32'), relay.shape_of(uop_166))) # shape=(7, 13)
uop_181 = relay.atanh(uop_158.astype('float64')) # shape=(7, 13)
var_183 = relay.var("var_183", dtype = "float64", shape = (7, 13))#candidate|183|(7, 13)|var|float64
bop_184 = relay.subtract(uop_136.astype('float64'), relay.reshape(var_183.astype('float64'), relay.shape_of(uop_136))) # shape=(7, 13)
uop_187 = relay.acos(bop_178.astype('float64')) # shape=(7, 13)
uop_189 = relay.tan(uop_187.astype('float32')) # shape=(7, 13)
uop_191 = relay.atanh(uop_189.astype('float32')) # shape=(7, 13)
bop_193 = relay.logical_or(uop_191.astype('bool'), relay.reshape(uop_138.astype('bool'), relay.shape_of(uop_191))) # shape=(7, 13)
uop_196 = relay.sinh(uop_189.astype('float32')) # shape=(7, 13)
bop_198 = relay.right_shift(uop_189.astype('int8'), relay.reshape(bop_155.astype('int8'), relay.shape_of(uop_189))) # shape=(7, 13)
var_201 = relay.var("var_201", dtype = "float32", shape = (7, 13))#candidate|201|(7, 13)|var|float32
bop_202 = relay.logical_xor(uop_191.astype('int32'), relay.reshape(var_201.astype('int32'), relay.shape_of(uop_191))) # shape=(7, 13)
uop_205 = relay.log10(bop_193.astype('float64')) # shape=(7, 13)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
call_207 = relay.TupleGetItem(func_112_call(relay.reshape(var_130.astype('float64'), [5, 8, 1]), relay.reshape(const_131.astype('float64'), [5, 8, 13]), ), 0)
call_208 = relay.TupleGetItem(func_116_call(relay.reshape(var_130.astype('float64'), [5, 8, 1]), relay.reshape(const_131.astype('float64'), [5, 8, 13]), ), 0)
uop_209 = relay.tan(uop_205.astype('float64')) # shape=(7, 13)
uop_211 = relay.atanh(uop_209.astype('float32')) # shape=(7, 13)
var_213 = relay.var("var_213", dtype = "float32", shape = (7, 13))#candidate|213|(7, 13)|var|float32
bop_214 = relay.logical_xor(uop_211.astype('uint64'), relay.reshape(var_213.astype('uint64'), relay.shape_of(uop_211))) # shape=(7, 13)
bop_217 = relay.less_equal(bop_193.astype('bool'), relay.reshape(uop_166.astype('bool'), relay.shape_of(bop_193))) # shape=(7, 13)
uop_220 = relay.asin(uop_209.astype('float32')) # shape=(7, 13)
uop_222 = relay.log2(uop_205.astype('float64')) # shape=(7, 13)
output = relay.Tuple([call_129,var_130,const_131,bop_143,bop_146,bop_149,call_152,const_153,bop_163,bop_168,uop_171,bop_174,uop_181,bop_184,uop_196,bop_198,bop_202,call_207,bop_214,bop_217,uop_220,uop_222,])
output2 = relay.Tuple([call_132,var_130,const_131,bop_143,bop_146,bop_149,call_154,const_153,bop_163,bop_168,uop_171,bop_174,uop_181,bop_184,uop_196,bop_198,bop_202,call_208,bop_214,bop_217,uop_220,uop_222,])
func_224 = relay.Function([var_119,var_130,var_173,var_177,var_183,var_201,var_213,], output)
mod['func_224'] = func_224
mod = relay.transform.InferType()(mod)
var_225 = relay.var("var_225", dtype = "float64", shape = (7, 13))#candidate|225|(7, 13)|var|float64
var_226 = relay.var("var_226", dtype = "float64", shape = (40,))#candidate|226|(40,)|var|float64
var_227 = relay.var("var_227", dtype = "bool", shape = (7, 13))#candidate|227|(7, 13)|var|bool
var_228 = relay.var("var_228", dtype = "float64", shape = (7, 13))#candidate|228|(7, 13)|var|float64
var_229 = relay.var("var_229", dtype = "float64", shape = (7, 13))#candidate|229|(7, 13)|var|float64
var_230 = relay.var("var_230", dtype = "float32", shape = (7, 13))#candidate|230|(7, 13)|var|float32
var_231 = relay.var("var_231", dtype = "float32", shape = (7, 13))#candidate|231|(7, 13)|var|float32
output = func_224(var_225,var_226,var_227,var_228,var_229,var_230,var_231,)
func_232 = relay.Function([var_225,var_226,var_227,var_228,var_229,var_230,var_231,], output)
mutated_mod['func_232'] = func_232
mutated_mod = relay.transform.InferType()(mutated_mod)
const_234 = relay.const([[[-10,8,-2,-2,-7,3,-4,-9,-6,6,1,-7,-5],[5,-7,-9,-1,-4,4,-4,-9,-1,10,7,3,1],[1,10,-2,2,-9,4,-2,1,-6,-6,-1,-8,-5],[6,8,3,-7,7,-5,-7,-2,2,-1,9,-9,-2],[10,-1,5,-4,-7,4,9,1,2,6,-1,7,3],[-9,1,-3,1,7,-1,1,-9,2,-8,-9,3,4],[6,-5,-9,3,6,-2,-2,-3,3,3,-10,-7,-9],[9,-4,5,-6,2,7,-7,3,-9,2,2,-7,5],[-9,6,-3,-10,-9,8,7,3,7,-6,1,7,-9],[-9,3,6,8,-10,6,-9,-4,10,-9,5,5,6],[9,9,2,6,2,7,-3,-5,6,-4,3,10,-1],[9,6,-1,-1,-5,-8,10,3,4,-6,-5,-2,-3]],[[-6,-8,9,2,7,2,-6,2,-8,-8,4,6,-10],[2,-3,7,-9,-6,-6,-4,4,10,-3,-7,-7,-9],[-4,10,-8,-6,-7,-10,-1,3,-9,-8,-4,-4,-1],[5,4,10,1,-4,2,9,4,-6,-8,-2,-9,-9],[-1,8,-3,-1,-7,7,-10,-6,-2,4,6,10,-10],[-8,-5,9,-9,-7,10,-9,8,6,6,10,8,-3],[3,-8,-8,-1,-5,-8,-9,-7,10,7,-4,5,4],[-10,-9,-9,-6,5,-3,-8,7,3,8,5,9,-7],[-5,-5,-10,-3,7,1,-9,9,5,10,-1,9,6],[9,-8,-8,-9,-3,-9,-7,-10,-2,-7,3,-8,5],[8,1,10,9,6,-2,-4,3,-5,10,1,8,-10],[-3,4,2,-3,-7,-2,-2,8,5,-3,5,-1,5]],[[4,-3,2,5,-3,9,-9,-1,-3,-3,-9,2,7],[9,-10,-1,8,1,9,8,-2,-7,3,5,-9,8],[-5,9,8,1,4,2,-7,2,8,5,4,4,10],[-3,6,-8,-2,4,-6,-7,10,-10,4,-10,-9,4],[-8,3,6,6,9,-4,2,-10,4,9,-9,-8,1],[5,4,7,6,10,8,-5,-10,6,8,10,-3,5],[-8,-5,4,2,9,-1,-9,-8,-9,-4,3,-10,-5],[-2,-1,-2,-3,-9,7,6,1,1,-10,-7,-4,-7],[10,-8,10,-7,1,10,-6,-1,-10,-9,4,-3,8],[3,-2,9,-8,8,8,-10,5,-3,1,-4,-2,-4],[-2,3,8,2,-6,1,-9,-3,4,-8,-3,10,6],[7,-8,6,-7,-1,-5,-4,-5,3,3,3,-9,-4]],[[3,-8,-9,-10,-3,8,5,6,2,9,4,1,9],[-5,-8,-3,-5,1,6,-5,-8,-3,-10,-3,1,-10],[-5,4,-9,-4,-10,-8,-2,10,-2,6,-8,5,8],[-2,1,5,5,1,8,10,6,2,-1,-5,7,-5],[2,5,-9,10,9,5,5,-9,-3,1,-1,7,-5],[-4,-4,9,8,-6,1,4,-1,5,7,9,4,-9],[-2,6,-1,-10,-7,10,3,10,-4,-3,5,1,3],[-2,-1,-2,1,10,8,6,-1,3,-2,7,1,-3],[9,10,5,8,5,10,-10,8,10,-7,10,-4,-5],[10,-4,-2,-6,-4,-4,-10,-2,7,-1,-2,-10,-2],[9,7,-1,-7,2,8,2,1,-4,-5,1,10,8],[3,-1,10,-2,-3,6,8,1,-5,7,7,7,8]],[[5,-10,-2,-2,-7,-2,-1,-4,6,2,7,8,-5],[5,-10,-2,5,-9,4,-9,-8,-3,-7,-6,8,-8],[-9,-1,5,6,1,10,9,-2,-3,-5,-3,1,-7],[2,-7,2,-1,10,-9,9,2,5,-9,4,7,4],[-10,-1,-2,-2,-5,6,-5,10,-1,-5,8,-4,-6],[-8,8,6,-7,-4,-8,-7,-1,9,-9,2,-9,3],[-7,-6,-3,-10,8,-10,-1,-3,5,5,5,3,-4],[4,-4,-3,-5,-5,-5,1,1,8,10,-4,-3,10],[-3,-2,10,2,-4,-6,-6,-9,-8,-6,-4,2,8],[-6,2,-9,-10,2,8,5,-3,-2,2,1,2,8],[4,7,-3,-10,-10,2,9,-10,-9,8,-4,10,2],[5,-1,-7,2,-6,8,1,-5,7,-6,1,-4,7]],[[1,-5,1,6,-4,6,10,2,-9,7,-6,10,-3],[-9,-1,-4,-6,-9,10,8,10,-9,-7,10,6,2],[4,-5,-2,-4,3,-9,-9,-5,-4,5,8,-6,-7],[-2,4,4,-5,-4,5,2,-8,-4,-7,-7,1,-9],[-6,-5,1,-5,-3,-2,-7,-3,-4,8,-6,-10,1],[-6,-1,-1,-9,-1,6,-5,4,5,-4,4,5,-2],[4,6,3,-8,-6,-3,-7,-5,-2,-2,-10,8,9],[-3,-3,7,-9,7,-6,6,5,-7,6,5,-6,4],[6,-9,-5,-1,-7,8,-1,6,9,9,3,-6,2],[9,-8,4,-7,9,7,-10,8,10,4,-7,10,-7],[-6,6,-2,-5,-9,4,-4,-8,1,-7,8,8,-7],[-5,3,-2,4,-6,-4,-9,-2,9,-6,-8,6,5]],[[5,6,-10,-9,-4,3,-10,6,-10,6,-8,7,-1],[10,-2,-3,-10,-1,3,-3,-2,5,-5,-5,-8,6],[-6,-1,-2,6,5,3,2,6,-3,-10,-5,-8,-10],[7,3,-5,10,8,6,-6,-5,8,2,-1,-3,1],[-10,10,3,9,1,-5,-1,-3,-8,9,7,-3,1],[5,6,2,6,-7,9,4,-1,3,5,10,6,-9],[9,3,5,-4,5,-9,9,-1,-4,-10,-8,-7,-10],[6,-8,-8,-4,-7,-10,-2,-8,-9,6,7,7,-7],[4,-5,2,-10,-6,-7,3,1,-4,10,3,1,1],[1,9,10,3,8,1,-6,-8,9,-5,-5,-5,-3],[6,10,-3,-5,1,6,5,9,-1,2,6,-3,4],[-1,4,-9,-9,-10,-4,-5,-2,-10,6,-2,-4,8]],[[-8,2,8,-4,9,-5,-4,1,1,4,-8,10,-4],[10,-9,-10,9,2,7,-10,-9,5,1,3,1,-5],[2,2,2,-3,-2,9,-5,9,-4,8,-5,9,8],[-7,-9,-7,-10,8,5,-2,-4,-8,-1,-3,-7,10],[-4,7,-3,1,9,-5,-1,2,4,-4,-6,-9,1],[-3,3,-8,4,-8,-2,-9,9,9,-6,-5,-4,3],[1,-8,-4,-1,-6,-5,-6,7,4,-6,-1,4,-7],[-10,-9,-2,4,3,1,-6,-2,-9,-6,6,-4,2],[9,-5,-6,-7,7,-5,1,1,-1,1,10,4,8],[-5,-4,8,-4,2,-7,2,10,6,-5,-5,8,4],[8,3,9,8,6,9,-8,-9,-9,10,-8,3,7],[-2,6,2,-3,1,2,7,7,9,8,2,-1,-9]],[[3,4,1,-5,-3,2,10,10,-3,2,-5,-7,9],[2,-4,-10,-4,6,-10,2,-6,9,-8,-1,-8,-2],[4,-9,9,-4,2,-9,-8,-10,6,4,-8,9,4],[8,-3,-8,-2,-3,-6,-3,2,-5,-6,1,-8,-3],[7,-4,-4,-6,5,-5,1,-7,6,7,1,-8,-6],[-9,5,-3,-4,5,9,4,-5,10,-8,3,8,-3],[10,6,-2,4,-6,10,-4,-9,-6,7,-3,-7,8],[6,10,3,-1,-7,3,-10,8,1,7,-5,3,6],[2,9,-3,-4,-4,-2,7,8,3,-8,-8,-3,9],[7,1,-7,7,3,-1,-5,-5,10,10,-1,-5,-4],[-10,-7,-6,-8,-8,-9,3,8,-10,-8,2,6,-5],[-5,-8,3,-5,-4,5,-7,10,-9,-10,4,-10,5]],[[-10,3,8,6,-6,-1,-9,1,10,6,-6,1,-2],[7,9,7,1,-7,-8,4,-3,-7,3,-4,-8,2],[-7,-1,2,-2,-10,-2,-9,-9,-3,-2,-9,9,-9],[-4,5,-10,-1,-1,8,3,4,-8,-6,1,1,8],[-4,-1,-1,-9,-1,1,1,2,-9,9,10,-6,8],[6,5,-9,-10,-2,-3,9,2,-5,-4,-4,2,6],[-3,-10,2,-10,9,-5,-10,-4,-2,9,5,-6,-1],[-1,4,-8,-10,7,4,8,-7,-6,4,1,3,9],[-4,-4,-7,-1,-2,9,-7,3,-4,-5,-6,9,1],[5,2,9,-8,4,-9,8,1,-1,-4,10,-5,-10],[-7,10,-10,9,10,-9,-2,6,7,-3,9,-9,-8],[10,3,6,-2,-8,-3,6,9,-4,8,2,-3,-6]],[[-9,-5,1,2,-9,5,-7,4,4,-5,-5,3,10],[1,-10,-6,-1,6,10,9,6,-9,-9,-5,4,7],[1,-5,5,4,6,-2,3,-10,-6,2,-2,-7,5],[5,3,9,4,-7,-3,-4,-3,5,-6,10,-3,-4],[-3,-6,-2,-8,-5,-9,4,-7,8,-5,8,8,-6],[-6,1,1,-10,4,2,-6,-10,7,2,-2,-5,-3],[8,9,-2,-2,4,8,-2,5,-4,-10,-6,10,-1],[10,9,1,5,-1,2,1,-7,-9,1,3,-4,2],[2,-2,4,10,-3,1,8,8,-7,5,10,-5,9],[6,2,7,-10,8,-9,7,7,4,2,-7,-3,-3],[8,1,7,-3,7,-5,-9,3,1,-1,-10,6,-9],[10,-5,2,-2,-3,2,6,-5,8,-3,-1,-3,-5]]], dtype = "uint32")#candidate|234|(11, 12, 13)|const|uint32
var_235 = relay.var("var_235", dtype = "uint32", shape = (11, 12, 13))#candidate|235|(11, 12, 13)|var|uint32
bop_236 = relay.right_shift(const_234.astype('uint32'), relay.reshape(var_235.astype('uint32'), relay.shape_of(const_234))) # shape=(11, 12, 13)
uop_239 = relay.log(var_235.astype('float64')) # shape=(11, 12, 13)
var_241 = relay.var("var_241", dtype = "uint32", shape = (11, 12, 13))#candidate|241|(11, 12, 13)|var|uint32
bop_242 = relay.minimum(var_235.astype('int16'), relay.reshape(var_241.astype('int16'), relay.shape_of(var_235))) # shape=(11, 12, 13)
var_245 = relay.var("var_245", dtype = "float64", shape = (11, 12, 13))#candidate|245|(11, 12, 13)|var|float64
bop_246 = relay.equal(uop_239.astype('bool'), relay.reshape(var_245.astype('bool'), relay.shape_of(uop_239))) # shape=(11, 12, 13)
uop_249 = relay.tan(var_241.astype('float64')) # shape=(11, 12, 13)
uop_251 = relay.log2(uop_239.astype('float32')) # shape=(11, 12, 13)
uop_253 = relay.atan(uop_251.astype('float32')) # shape=(11, 12, 13)
bop_255 = relay.left_shift(uop_239.astype('uint16'), relay.reshape(var_235.astype('uint16'), relay.shape_of(uop_239))) # shape=(11, 12, 13)
uop_258 = relay.exp(uop_253.astype('float32')) # shape=(11, 12, 13)
func_88_call = mod.get_global_var('func_88')
func_94_call = mutated_mod.get_global_var('func_94')
var_261 = relay.var("var_261", dtype = "float32", shape = (4,))#candidate|261|(4,)|var|float32
call_260 = relay.TupleGetItem(func_88_call(relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float64'), [4,]), relay.reshape(var_261.astype('float64'), [4,]), ), 5)
call_262 = relay.TupleGetItem(func_94_call(relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float32'), [4,]), relay.reshape(var_261.astype('float64'), [4,]), relay.reshape(var_261.astype('float64'), [4,]), ), 5)
output = relay.Tuple([bop_236,bop_242,bop_246,uop_249,bop_255,uop_258,call_260,var_261,])
output2 = relay.Tuple([bop_236,bop_242,bop_246,uop_249,bop_255,uop_258,call_262,var_261,])
func_263 = relay.Function([var_235,var_241,var_245,var_261,], output)
mod['func_263'] = func_263
mod = relay.transform.InferType()(mod)
var_264 = relay.var("var_264", dtype = "uint32", shape = (11, 12, 13))#candidate|264|(11, 12, 13)|var|uint32
var_265 = relay.var("var_265", dtype = "uint32", shape = (11, 12, 13))#candidate|265|(11, 12, 13)|var|uint32
var_266 = relay.var("var_266", dtype = "float64", shape = (11, 12, 13))#candidate|266|(11, 12, 13)|var|float64
var_267 = relay.var("var_267", dtype = "float32", shape = (4,))#candidate|267|(4,)|var|float32
output = func_263(var_264,var_265,var_266,var_267,)
func_268 = relay.Function([var_264,var_265,var_266,var_267,], output)
mutated_mod['func_268'] = func_268
mutated_mod = relay.transform.InferType()(mutated_mod)
var_270 = relay.var("var_270", dtype = "int16", shape = (6, 2))#candidate|270|(6, 2)|var|int16
var_271 = relay.var("var_271", dtype = "int16", shape = (6, 2))#candidate|271|(6, 2)|var|int16
bop_272 = relay.minimum(var_270.astype('int16'), relay.reshape(var_271.astype('int16'), relay.shape_of(var_270))) # shape=(6, 2)
bop_275 = relay.greater_equal(var_270.astype('bool'), relay.reshape(var_271.astype('bool'), relay.shape_of(var_270))) # shape=(6, 2)
bop_278 = relay.left_shift(var_271.astype('uint64'), relay.reshape(bop_272.astype('uint64'), relay.shape_of(var_271))) # shape=(6, 2)
func_112_call = mod.get_global_var('func_112')
func_116_call = mutated_mod.get_global_var('func_116')
const_282 = relay.const([7.763675,4.111905,5.495288,-8.573433,0.043238,1.139732,-3.797060,-3.966521,-9.953806,-5.543832,-6.866331,-9.152892,7.852329,1.830833,-0.345632,4.952833,1.962911,1.634042,-0.262680,-9.254058,-1.585406,3.962762,9.083038,-6.808512,9.203977,-7.464951,5.345092,4.918118,-5.145715,9.745723,2.941957,-7.621069,-4.830200,1.560938,-8.256441,-1.580906,2.295200,6.552203,7.964514,7.872116], dtype = "float64")#candidate|282|(40,)|const|float64
var_283 = relay.var("var_283", dtype = "float64", shape = (520,))#candidate|283|(520,)|var|float64
call_281 = relay.TupleGetItem(func_112_call(relay.reshape(const_282.astype('float64'), [5, 8, 1]), relay.reshape(var_283.astype('float64'), [5, 8, 13]), ), 0)
call_284 = relay.TupleGetItem(func_116_call(relay.reshape(const_282.astype('float64'), [5, 8, 1]), relay.reshape(var_283.astype('float64'), [5, 8, 13]), ), 0)
uop_285 = relay.sinh(var_283.astype('float64')) # shape=(520,)
uop_287 = relay.tan(call_281.astype('float64')) # shape=(5, 8, 13)
uop_289 = relay.tan(call_284.astype('float64')) # shape=(5, 8, 13)
uop_290 = relay.sigmoid(uop_285.astype('float32')) # shape=(520,)
uop_292 = relay.tan(uop_290.astype('float64')) # shape=(520,)
func_88_call = mod.get_global_var('func_88')
func_94_call = mutated_mod.get_global_var('func_94')
var_295 = relay.var("var_295", dtype = "float32", shape = (4,))#candidate|295|(4,)|var|float32
call_294 = relay.TupleGetItem(func_88_call(relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float64'), [4,]), relay.reshape(var_295.astype('float64'), [4,]), ), 4)
call_296 = relay.TupleGetItem(func_94_call(relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float32'), [4,]), relay.reshape(var_295.astype('float64'), [4,]), relay.reshape(var_295.astype('float64'), [4,]), ), 4)
var_297 = relay.var("var_297", dtype = "float64", shape = (520,))#candidate|297|(520,)|var|float64
bop_298 = relay.logical_and(uop_292.astype('bool'), relay.reshape(var_297.astype('bool'), relay.shape_of(uop_292))) # shape=(520,)
output = relay.Tuple([bop_275,bop_278,const_282,uop_287,call_294,var_295,bop_298,])
output2 = relay.Tuple([bop_275,bop_278,const_282,uop_289,call_296,var_295,bop_298,])
func_301 = relay.Function([var_270,var_271,var_283,var_295,var_297,], output)
mod['func_301'] = func_301
mod = relay.transform.InferType()(mod)
var_302 = relay.var("var_302", dtype = "int16", shape = (6, 2))#candidate|302|(6, 2)|var|int16
var_303 = relay.var("var_303", dtype = "int16", shape = (6, 2))#candidate|303|(6, 2)|var|int16
var_304 = relay.var("var_304", dtype = "float64", shape = (520,))#candidate|304|(520,)|var|float64
var_305 = relay.var("var_305", dtype = "float32", shape = (4,))#candidate|305|(4,)|var|float32
var_306 = relay.var("var_306", dtype = "float64", shape = (520,))#candidate|306|(520,)|var|float64
output = func_301(var_302,var_303,var_304,var_305,var_306,)
func_307 = relay.Function([var_302,var_303,var_304,var_305,var_306,], output)
mutated_mod['func_307'] = func_307
mutated_mod = relay.transform.InferType()(mutated_mod)
var_309 = relay.var("var_309", dtype = "float64", shape = (10, 4))#candidate|309|(10, 4)|var|float64
uop_310 = relay.sin(var_309.astype('float64')) # shape=(10, 4)
uop_312 = relay.cos(var_309.astype('float32')) # shape=(10, 4)
bop_314 = relay.multiply(uop_312.astype('int8'), relay.reshape(var_309.astype('int8'), relay.shape_of(uop_312))) # shape=(10, 4)
bop_317 = relay.left_shift(var_309.astype('int32'), relay.reshape(bop_314.astype('int32'), relay.shape_of(var_309))) # shape=(10, 4)
uop_320 = relay.asinh(bop_317.astype('float32')) # shape=(10, 4)
uop_322 = relay.erf(uop_320.astype('float32')) # shape=(10, 4)
uop_324 = relay.sqrt(uop_322.astype('float32')) # shape=(10, 4)
uop_326 = relay.erf(uop_324.astype('float32')) # shape=(10, 4)
var_328 = relay.var("var_328", dtype = "float32", shape = (10, 4))#candidate|328|(10, 4)|var|float32
bop_329 = relay.equal(uop_322.astype('bool'), relay.reshape(var_328.astype('bool'), relay.shape_of(uop_322))) # shape=(10, 4)
output = relay.Tuple([uop_310,uop_326,bop_329,])
output2 = relay.Tuple([uop_310,uop_326,bop_329,])
func_332 = relay.Function([var_309,var_328,], output)
mod['func_332'] = func_332
mod = relay.transform.InferType()(mod)
mutated_mod['func_332'] = func_332
mutated_mod = relay.transform.InferType()(mutated_mod)
func_332_call = mutated_mod.get_global_var('func_332')
var_334 = relay.var("var_334", dtype = "float64", shape = (10, 4))#candidate|334|(10, 4)|var|float64
var_335 = relay.var("var_335", dtype = "float32", shape = (10, 4))#candidate|335|(10, 4)|var|float32
call_333 = func_332_call(var_334,var_335,)
output = call_333
func_336 = relay.Function([var_334,var_335,], output)
mutated_mod['func_336'] = func_336
mutated_mod = relay.transform.InferType()(mutated_mod)
var_338 = relay.var("var_338", dtype = "float32", shape = ())#candidate|338|()|var|float32
uop_339 = relay.sqrt(var_338.astype('float32')) # shape=()
uop_341 = relay.acos(uop_339.astype('float32')) # shape=()
output = uop_341
output2 = uop_341
F = relay.Function([var_338,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_338,], output2)
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
input_338= np.array(7.609648, dtype='float32')
module1.set_input('var_338', input_338)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_338, )
res3 = intrp3.evaluate()(input_338, )
res4 = intrp4.evaluate()(input_338, )
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
module5.set_input('var_338', input_338)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_338, )
res7 = intrp7.evaluate()(input_338, )
res8 = intrp8.evaluate()(input_338, )
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
module9.set_input('var_338', input_338)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_338, )
res11 = intrp11.evaluate()(input_338, )
res12 = intrp12.evaluate()(input_338, )
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
module13.set_input('var_338', input_338)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_338, )
res15 = intrp15.evaluate()(input_338, )
res16 = intrp16.evaluate()(input_338, )
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
module17.set_input('var_338', input_338)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_338, )
res19 = intrp19.evaluate()(input_338, )
res20 = intrp20.evaluate()(input_338, )
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
module21.set_input('var_338', input_338)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_338, )
res23 = intrp23.evaluate()(input_338, )
res24 = intrp24.evaluate()(input_338, )
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