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
var_0 = relay.var("var_0", dtype = "float64", shape = (13, 8, 5))#candidate|0|(13, 8, 5)|var|float64
uop_1 = relay.cos(var_0.astype('float64')) # shape=(13, 8, 5)
uop_3 = relay.log(var_0.astype('float32')) # shape=(13, 8, 5)
uop_5 = relay.rsqrt(var_0.astype('float64')) # shape=(13, 8, 5)
bop_7 = relay.multiply(uop_3.astype('float64'), relay.reshape(uop_1.astype('float64'), relay.shape_of(uop_3))) # shape=(13, 8, 5)
uop_10 = relay.atanh(uop_5.astype('float64')) # shape=(13, 8, 5)
uop_12 = relay.sigmoid(var_0.astype('float64')) # shape=(13, 8, 5)
bop_14 = relay.bitwise_xor(uop_10.astype('uint64'), relay.reshape(uop_12.astype('uint64'), relay.shape_of(uop_10))) # shape=(13, 8, 5)
uop_17 = relay.asin(uop_10.astype('float64')) # shape=(13, 8, 5)
bop_19 = relay.add(uop_10.astype('uint16'), relay.reshape(uop_12.astype('uint16'), relay.shape_of(uop_10))) # shape=(13, 8, 5)
bop_22 = relay.equal(uop_17.astype('bool'), relay.reshape(uop_12.astype('bool'), relay.shape_of(uop_17))) # shape=(13, 8, 5)
bop_25 = relay.floor_divide(uop_1.astype('float64'), relay.reshape(uop_10.astype('float64'), relay.shape_of(uop_1))) # shape=(13, 8, 5)
uop_28 = relay.rsqrt(uop_3.astype('float64')) # shape=(13, 8, 5)
bop_30 = relay.power(bop_22.astype('float32'), relay.reshape(uop_1.astype('float32'), relay.shape_of(bop_22))) # shape=(13, 8, 5)
var_33 = relay.var("var_33", dtype = "uint64", shape = (13, 8, 5))#candidate|33|(13, 8, 5)|var|uint64
bop_34 = relay.logical_and(bop_14.astype('bool'), relay.reshape(var_33.astype('bool'), relay.shape_of(bop_14))) # shape=(13, 8, 5)
uop_37 = relay.rsqrt(bop_34.astype('float32')) # shape=(13, 8, 5)
bop_39 = relay.add(bop_22.astype('int32'), relay.reshape(uop_12.astype('int32'), relay.shape_of(bop_22))) # shape=(13, 8, 5)
uop_42 = relay.sqrt(uop_28.astype('float64')) # shape=(13, 8, 5)
bop_44 = relay.subtract(bop_14.astype('float64'), relay.reshape(var_33.astype('float64'), relay.shape_of(bop_14))) # shape=(13, 8, 5)
bop_47 = relay.floor_divide(bop_7.astype('float64'), relay.reshape(uop_3.astype('float64'), relay.shape_of(bop_7))) # shape=(13, 8, 5)
output = relay.Tuple([bop_19,bop_25,bop_30,uop_37,bop_39,uop_42,bop_44,bop_47,])
output2 = relay.Tuple([bop_19,bop_25,bop_30,uop_37,bop_39,uop_42,bop_44,bop_47,])
func_50 = relay.Function([var_0,var_33,], output)
mod['func_50'] = func_50
mod = relay.transform.InferType()(mod)
mutated_mod['func_50'] = func_50
mutated_mod = relay.transform.InferType()(mutated_mod)
func_50_call = mutated_mod.get_global_var('func_50')
var_52 = relay.var("var_52", dtype = "float64", shape = (13, 8, 5))#candidate|52|(13, 8, 5)|var|float64
var_53 = relay.var("var_53", dtype = "uint64", shape = (13, 8, 5))#candidate|53|(13, 8, 5)|var|uint64
call_51 = func_50_call(var_52,var_53,)
output = call_51
func_54 = relay.Function([var_52,var_53,], output)
mutated_mod['func_54'] = func_54
mutated_mod = relay.transform.InferType()(mutated_mod)
var_56 = relay.var("var_56", dtype = "uint16", shape = (13, 13, 13))#candidate|56|(13, 13, 13)|var|uint16
var_57 = relay.var("var_57", dtype = "uint16", shape = (13, 13, 13))#candidate|57|(13, 13, 13)|var|uint16
bop_58 = relay.add(var_56.astype('uint16'), relay.reshape(var_57.astype('uint16'), relay.shape_of(var_56))) # shape=(13, 13, 13)
uop_61 = relay.acos(var_57.astype('float64')) # shape=(13, 13, 13)
uop_63 = relay.acos(uop_61.astype('float64')) # shape=(13, 13, 13)
var_65 = relay.var("var_65", dtype = "float64", shape = (13, 13, 13))#candidate|65|(13, 13, 13)|var|float64
bop_66 = relay.equal(uop_63.astype('bool'), relay.reshape(var_65.astype('bool'), relay.shape_of(uop_63))) # shape=(13, 13, 13)
uop_69 = relay.cos(uop_63.astype('float64')) # shape=(13, 13, 13)
bop_71 = relay.floor_divide(uop_69.astype('float32'), relay.reshape(bop_58.astype('float32'), relay.shape_of(uop_69))) # shape=(13, 13, 13)
var_74 = relay.var("var_74", dtype = "float64", shape = (13, 13, 13))#candidate|74|(13, 13, 13)|var|float64
bop_75 = relay.floor_mod(uop_63.astype('float32'), relay.reshape(var_74.astype('float32'), relay.shape_of(uop_63))) # shape=(13, 13, 13)
output = relay.Tuple([bop_66,bop_71,bop_75,])
output2 = relay.Tuple([bop_66,bop_71,bop_75,])
func_78 = relay.Function([var_56,var_57,var_65,var_74,], output)
mod['func_78'] = func_78
mod = relay.transform.InferType()(mod)
var_79 = relay.var("var_79", dtype = "uint16", shape = (13, 13, 13))#candidate|79|(13, 13, 13)|var|uint16
var_80 = relay.var("var_80", dtype = "uint16", shape = (13, 13, 13))#candidate|80|(13, 13, 13)|var|uint16
var_81 = relay.var("var_81", dtype = "float64", shape = (13, 13, 13))#candidate|81|(13, 13, 13)|var|float64
var_82 = relay.var("var_82", dtype = "float64", shape = (13, 13, 13))#candidate|82|(13, 13, 13)|var|float64
output = func_78(var_79,var_80,var_81,var_82,)
func_83 = relay.Function([var_79,var_80,var_81,var_82,], output)
mutated_mod['func_83'] = func_83
mutated_mod = relay.transform.InferType()(mutated_mod)
var_85 = relay.var("var_85", dtype = "float32", shape = (9, 3))#candidate|85|(9, 3)|var|float32
uop_86 = relay.log(var_85.astype('float32')) # shape=(9, 3)
bop_88 = relay.bitwise_or(uop_86.astype('uint16'), relay.reshape(var_85.astype('uint16'), relay.shape_of(uop_86))) # shape=(9, 3)
func_78_call = mod.get_global_var('func_78')
func_83_call = mutated_mod.get_global_var('func_83')
var_92 = relay.var("var_92", dtype = "uint16", shape = (2197,))#candidate|92|(2197,)|var|uint16
call_91 = relay.TupleGetItem(func_78_call(relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), ), 2)
call_93 = relay.TupleGetItem(func_83_call(relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), ), 2)
bop_94 = relay.greater_equal(uop_86.astype('bool'), relay.reshape(var_85.astype('bool'), relay.shape_of(uop_86))) # shape=(9, 3)
bop_97 = relay.add(bop_94.astype('uint64'), relay.reshape(var_85.astype('uint64'), relay.shape_of(bop_94))) # shape=(9, 3)
bop_100 = relay.left_shift(bop_88.astype('int32'), relay.reshape(bop_94.astype('int32'), relay.shape_of(bop_88))) # shape=(9, 3)
bop_103 = relay.less_equal(var_85.astype('bool'), relay.reshape(uop_86.astype('bool'), relay.shape_of(var_85))) # shape=(9, 3)
var_106 = relay.var("var_106", dtype = "uint64", shape = (9, 3))#candidate|106|(9, 3)|var|uint64
bop_107 = relay.power(bop_97.astype('float32'), relay.reshape(var_106.astype('float32'), relay.shape_of(bop_97))) # shape=(9, 3)
uop_110 = relay.acos(var_106.astype('float32')) # shape=(9, 3)
uop_112 = relay.asin(bop_97.astype('float64')) # shape=(9, 3)
func_50_call = mod.get_global_var('func_50')
func_54_call = mutated_mod.get_global_var('func_54')
var_115 = relay.var("var_115", dtype = "float64", shape = (520,))#candidate|115|(520,)|var|float64
call_114 = relay.TupleGetItem(func_50_call(relay.reshape(var_115.astype('float64'), [13, 8, 5]), relay.reshape(var_115.astype('uint64'), [13, 8, 5]), ), 6)
call_116 = relay.TupleGetItem(func_54_call(relay.reshape(var_115.astype('float64'), [13, 8, 5]), relay.reshape(var_115.astype('uint64'), [13, 8, 5]), ), 6)
const_117 = relay.const([-7.964494,0.757397,-9.863435,-0.499094,-9.955209,2.387128,6.498125,5.213761,5.098717,-4.082907,-2.387036,-2.650334,-8.487054,6.751856,1.899856,8.337058,1.358138,2.591590,-9.664684,-4.342346,2.929096,2.961922,3.555272,2.614984,1.935750,-9.376603,-7.996716,9.810650,-3.284846,1.190241,-2.926013,-6.756077,5.837039,9.165647,0.948733,-3.223621,5.689815,2.252098,6.359255,6.935347,-4.219193,-8.743237,-6.628719,-7.562447,6.567444,-4.266725,-3.025693,8.099051,-3.205912,-2.501467,-8.401743,-6.003672,-8.963136,1.323152,-1.376052,-9.238700,-1.450788,1.768231,-5.335930,9.261705,6.246682,5.982981,-9.704801,-5.301922,0.848834,2.738309,6.151683,-6.564755,3.379348,1.744882,-7.212945,1.247462,2.670719,-0.992004,-2.247574,7.409259,0.219101,2.577073,7.747514,9.056677,-7.841146,3.453278,-9.098651,6.319376,6.007563,-5.359609,-8.012654,-6.933868,-6.150699,-2.389678,-9.835485,7.752229,-1.014190,-0.379911,-1.899816,4.525496,-9.932600,-7.449702,-8.501889,-1.714329,9.831056,5.559879,-0.487653,3.564702,-1.637231,3.055388,3.287473,1.283245,-2.123482,2.719439,3.803409,9.260633,1.242066,-0.997758,-4.291288,1.121562,1.440637,8.162169,-3.099552,1.868645,-1.477550,-6.749041,8.245762,-5.909132,1.016113,-8.829168,-5.064170,-4.763128,3.366758,9.188654,5.913705,7.515135,0.032771,3.562460,4.615437,4.844348,5.019393,5.553879,7.067553,-7.918378,-1.554943,4.680168,7.837866,-9.578974,-4.250844,-5.941549,2.443303,4.576912,-7.684712,-6.976186,2.672637,-1.862324,9.791243,5.696122,2.491471,3.932868,-1.749030,-3.343553,-7.101274,-8.197223,-0.642162,6.221946,4.462311,-9.058473,7.348716,0.971060,-0.918284,1.106773,-9.678202,-4.237488,8.087911,0.765551,-3.947041,4.213890,-7.705470,-4.960980,-6.602529,-2.908428,-4.854239,3.944394,-7.858108,-9.061891,-1.512439,-9.756552,-6.616025,-9.915717,3.997933,9.160960,1.016123,-4.135679,-4.293308,-3.256765,-3.483515,-7.292648,-6.698776,-4.726512,1.756780,5.911876,5.605680,-2.922998,0.460468,5.509731,-0.690363,-2.683708,8.591831,0.752313,0.881941,-0.783035,6.658882,-6.695021,-1.274529,-9.123062,0.283151,-8.102957,0.201870,-7.557955,9.034831,0.347002,-6.672604,-7.316022,9.238266,-6.718584,1.474955,-4.473812,6.552946,-8.849233,-0.211209,2.525594,1.205580,-6.342273,-7.998328,4.184735,-1.725889,3.613442,-2.845538,-0.905622,-1.416331,0.909694,1.793381,-5.745038,-2.979661,-9.150974,9.042383,0.539373,8.775348,5.654984,-0.885748,-9.374458,-8.425367,-9.816854,4.165250,-7.161203,2.910627,3.390860,0.752286,7.486705,5.365793,-9.728089,-5.338884,1.031960,5.662298,5.227497,2.318604,-9.490971,-0.981235,-0.279001,4.616925,5.235851,-1.117245,3.010006,-2.741935,-3.523251,1.483303,-2.243390,-9.462150,7.856709,-7.247779,2.046404,-7.792080,7.012042,-4.853372,-0.060776,-6.845574,-3.704362,7.860989,3.844741,9.467213,1.741156,-3.016689,-7.836507,8.604793,9.794895,-0.364091,0.172194,-4.553880,5.351221,-7.424178,0.379856,2.106674,-1.471570,7.478370,-5.418091,1.925917,1.686479,1.903543,-2.027925,-9.316087,8.406841,-1.473780,3.511214,7.238000,-2.909038,0.440197,8.015261,1.700040,7.514371,6.112774,7.766243,8.507393,-6.635365,-3.392897,8.821633,-4.693509,-7.326117,7.373514,-9.781624,-7.802599,-3.383331,-0.072865,5.977941,0.235560,-7.042658,-7.640519,5.147167,4.736387,6.605374,-4.082420,-6.858870,1.605952,-5.357406,-3.280013,9.944420,7.187908,-7.574168,7.759782,5.082578,-9.105551,0.796095,-2.769274,-5.537005,9.143093,7.752166,-7.301100,-0.943996,-8.363049,-5.837565,9.533631,-7.321162,-3.013628,-9.852092,-5.009606,-7.578657,-6.314932,3.794169,4.419848,-8.133871,0.519721,-5.746442,6.886261,2.652560,6.044370,-3.872087,-7.274618,1.973614,-5.451554,-5.024795,-0.260556,1.960509,7.283307,8.163936,8.501021,3.339508,2.452045,-7.356745,-5.403697,2.085370,-6.536536,5.787608,-6.977691,1.143655,-1.006877,3.937832,-6.197385,0.628672,7.539219,-0.819662,-4.680601,-0.828123,-2.346422,-3.868372,-9.372171,-6.370069,-1.580903,3.620853,9.406006,1.483994,7.498677,-2.716875,0.316073,3.929472,-6.239325,0.924733,1.939004,-5.762553,5.982682,-1.055526,-1.659290,-5.979057,-2.315624,5.485381,-7.483196,-4.084970,0.137126,0.248254,-6.124226,-6.167037,9.815822,-7.818233,0.027788,-3.843337,6.774717,-7.878035,0.390203,-9.909023,9.904559,7.811674,-1.001068,7.800819,-6.570340,6.334239,2.736010,6.195725,2.152722,5.571803,2.130858,7.114672,-6.889032,-1.634493,-4.248359,-3.819427,7.591858,0.839583,5.516435,6.509195,-2.641483,-9.803139,-6.479000,-1.105768,1.054220,-0.435246,0.124287,-5.913374,-1.293247,6.530382,3.386709,-2.761307,-8.307408,2.480285,-7.753656,4.054316,4.316853,-7.839331,-5.724053,8.345232,-0.042907,-7.283922,-5.840971,2.659882,6.631623,6.540608,-9.754982,4.747726,-3.739846,-2.253020,-0.935707,8.537557,-7.558131,3.710684,-6.281458,-7.142457,2.669049,-4.306466,1.025068,-4.953387,-0.927585,-8.948102,1.006217,0.455873,-8.589862,-0.452123,-7.795707,-1.723745,7.382471,5.844019,-5.765174,-3.736892,-3.334587,4.211721,-7.534993,-0.252591,-0.349684,-0.229292,6.408862,9.549059,7.037007,-0.226078,0.240853,-3.999517,-5.219613,-8.915674], dtype = "float64")#candidate|117|(520,)|const|float64
bop_118 = relay.bitwise_and(var_115.astype('uint32'), relay.reshape(const_117.astype('uint32'), relay.shape_of(var_115))) # shape=(520,)
bop_121 = relay.maximum(var_115.astype('uint32'), relay.reshape(const_117.astype('uint32'), relay.shape_of(var_115))) # shape=(520,)
uop_124 = relay.acos(var_106.astype('float32')) # shape=(9, 3)
const_126 = relay.const([[7.759135,-4.799216,-0.245134],[-8.890469,-0.864596,-2.976367],[-5.552617,4.098777,-7.415172],[-3.252488,1.022086,3.812106],[5.605676,-9.638735,6.211898],[-5.941538,5.805486,2.926631],[2.520628,0.085535,0.350844],[-1.302439,7.720510,-7.782421],[6.813696,-5.809219,0.985957]], dtype = "float64")#candidate|126|(9, 3)|const|float64
bop_127 = relay.power(uop_112.astype('float32'), relay.reshape(const_126.astype('float32'), relay.shape_of(uop_112))) # shape=(9, 3)
bop_130 = relay.maximum(bop_97.astype('int64'), relay.reshape(bop_103.astype('int64'), relay.shape_of(bop_97))) # shape=(9, 3)
uop_133 = relay.asinh(var_106.astype('float32')) # shape=(9, 3)
bop_135 = relay.equal(call_114.astype('bool'), relay.reshape(const_117.astype('bool'), relay.shape_of(call_114))) # shape=(13, 8, 5)
bop_138 = relay.equal(call_116.astype('bool'), relay.reshape(const_117.astype('bool'), relay.shape_of(call_116))) # shape=(13, 8, 5)
uop_139 = relay.sinh(bop_130.astype('float32')) # shape=(9, 3)
bop_141 = relay.maximum(bop_127.astype('int8'), relay.reshape(var_85.astype('int8'), relay.shape_of(bop_127))) # shape=(9, 3)
uop_144 = relay.atanh(bop_127.astype('float32')) # shape=(9, 3)
uop_146 = relay.atanh(uop_144.astype('float64')) # shape=(9, 3)
uop_148 = relay.tan(bop_141.astype('float32')) # shape=(9, 3)
uop_150 = relay.cosh(uop_146.astype('float32')) # shape=(9, 3)
bop_152 = relay.greater_equal(uop_150.astype('bool'), relay.reshape(uop_124.astype('bool'), relay.shape_of(uop_150))) # shape=(9, 3)
var_155 = relay.var("var_155", dtype = "float32", shape = (9, 3))#candidate|155|(9, 3)|var|float32
bop_156 = relay.less(uop_150.astype('bool'), relay.reshape(var_155.astype('bool'), relay.shape_of(uop_150))) # shape=(9, 3)
const_159 = relay.const([[-1,-6,5],[7,4,-7],[-10,-10,10],[-7,-8,6],[-1,4,-10],[-8,5,10],[-7,-9,1],[-3,-4,-5],[2,1,-8]], dtype = "int32")#candidate|159|(9, 3)|const|int32
bop_160 = relay.add(bop_100.astype('uint8'), relay.reshape(const_159.astype('uint8'), relay.shape_of(bop_100))) # shape=(9, 3)
uop_163 = relay.sinh(bop_156.astype('float64')) # shape=(9, 3)
var_165 = relay.var("var_165", dtype = "float64", shape = (9, 3))#candidate|165|(9, 3)|var|float64
bop_166 = relay.bitwise_and(uop_163.astype('int32'), relay.reshape(var_165.astype('int32'), relay.shape_of(uop_163))) # shape=(9, 3)
uop_169 = relay.rsqrt(bop_166.astype('float64')) # shape=(9, 3)
uop_171 = relay.sigmoid(uop_144.astype('float32')) # shape=(9, 3)
uop_173 = relay.sinh(bop_166.astype('float32')) # shape=(9, 3)
uop_175 = relay.sqrt(uop_163.astype('float32')) # shape=(9, 3)
uop_177 = relay.atanh(uop_173.astype('float64')) # shape=(9, 3)
bop_179 = relay.minimum(uop_177.astype('uint8'), relay.reshape(var_85.astype('uint8'), relay.shape_of(uop_177))) # shape=(9, 3)
func_78_call = mod.get_global_var('func_78')
func_83_call = mutated_mod.get_global_var('func_83')
call_182 = relay.TupleGetItem(func_78_call(relay.reshape(call_91.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), ), 1)
call_183 = relay.TupleGetItem(func_83_call(relay.reshape(call_91.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('uint16'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), relay.reshape(var_92.astype('float64'), [13, 13, 13]), ), 1)
var_184 = relay.var("var_184", dtype = "float32", shape = (9, 3))#candidate|184|(9, 3)|var|float32
bop_185 = relay.maximum(uop_150.astype('uint32'), relay.reshape(var_184.astype('uint32'), relay.shape_of(uop_150))) # shape=(9, 3)
bop_188 = relay.logical_and(uop_171.astype('bool'), relay.reshape(uop_148.astype('bool'), relay.shape_of(uop_171))) # shape=(9, 3)
uop_191 = relay.rsqrt(uop_175.astype('float64')) # shape=(9, 3)
bop_193 = relay.logical_and(bop_179.astype('bool'), relay.reshape(uop_169.astype('bool'), relay.shape_of(bop_179))) # shape=(9, 3)
output = relay.Tuple([call_91,var_92,bop_107,uop_110,bop_118,bop_121,uop_133,bop_135,uop_139,bop_152,bop_160,call_182,bop_185,bop_188,uop_191,bop_193,])
output2 = relay.Tuple([call_93,var_92,bop_107,uop_110,bop_118,bop_121,uop_133,bop_138,uop_139,bop_152,bop_160,call_183,bop_185,bop_188,uop_191,bop_193,])
func_196 = relay.Function([var_85,var_92,var_106,var_115,var_155,var_165,var_184,], output)
mod['func_196'] = func_196
mod = relay.transform.InferType()(mod)
mutated_mod['func_196'] = func_196
mutated_mod = relay.transform.InferType()(mutated_mod)
func_196_call = mutated_mod.get_global_var('func_196')
var_198 = relay.var("var_198", dtype = "float32", shape = (9, 3))#candidate|198|(9, 3)|var|float32
var_199 = relay.var("var_199", dtype = "uint16", shape = (2197,))#candidate|199|(2197,)|var|uint16
var_200 = relay.var("var_200", dtype = "uint64", shape = (9, 3))#candidate|200|(9, 3)|var|uint64
var_201 = relay.var("var_201", dtype = "float64", shape = (520,))#candidate|201|(520,)|var|float64
var_202 = relay.var("var_202", dtype = "float32", shape = (9, 3))#candidate|202|(9, 3)|var|float32
var_203 = relay.var("var_203", dtype = "float64", shape = (9, 3))#candidate|203|(9, 3)|var|float64
var_204 = relay.var("var_204", dtype = "float32", shape = (9, 3))#candidate|204|(9, 3)|var|float32
call_197 = func_196_call(var_198,var_199,var_200,var_201,var_202,var_203,var_204,)
output = call_197
func_205 = relay.Function([var_198,var_199,var_200,var_201,var_202,var_203,var_204,], output)
mutated_mod['func_205'] = func_205
mutated_mod = relay.transform.InferType()(mutated_mod)
const_207 = relay.const([5.046819,-1.374567,1.464657,4.032510,-9.549930], dtype = "float64")#candidate|207|(5,)|const|float64
var_208 = relay.var("var_208", dtype = "float64", shape = (5,))#candidate|208|(5,)|var|float64
bop_209 = relay.divide(const_207.astype('float64'), relay.reshape(var_208.astype('float64'), relay.shape_of(const_207))) # shape=(5,)
bop_212 = relay.divide(var_208.astype('float32'), relay.reshape(bop_209.astype('float32'), relay.shape_of(var_208))) # shape=(5,)
uop_215 = relay.atanh(var_208.astype('float32')) # shape=(5,)
bop_217 = relay.mod(uop_215.astype('float32'), relay.reshape(var_208.astype('float32'), relay.shape_of(uop_215))) # shape=(5,)
bop_220 = relay.floor_divide(var_208.astype('float64'), relay.reshape(bop_217.astype('float64'), relay.shape_of(var_208))) # shape=(5,)
bop_223 = relay.equal(uop_215.astype('bool'), relay.reshape(var_208.astype('bool'), relay.shape_of(uop_215))) # shape=(5,)
output = relay.Tuple([bop_212,bop_220,bop_223,])
output2 = relay.Tuple([bop_212,bop_220,bop_223,])
func_226 = relay.Function([var_208,], output)
mod['func_226'] = func_226
mod = relay.transform.InferType()(mod)
mutated_mod['func_226'] = func_226
mutated_mod = relay.transform.InferType()(mutated_mod)
var_227 = relay.var("var_227", dtype = "float64", shape = (5,))#candidate|227|(5,)|var|float64
func_226_call = mutated_mod.get_global_var('func_226')
call_228 = func_226_call(var_227)
output = call_228
func_229 = relay.Function([var_227], output)
mutated_mod['func_229'] = func_229
mutated_mod = relay.transform.InferType()(mutated_mod)
var_231 = relay.var("var_231", dtype = "float64", shape = (5, 9))#candidate|231|(5, 9)|var|float64
uop_232 = relay.asinh(var_231.astype('float64')) # shape=(5, 9)
func_50_call = mod.get_global_var('func_50')
func_54_call = mutated_mod.get_global_var('func_54')
const_235 = relay.const([-9.486689,9.972875,-0.914550,9.923140,6.242341,-4.827481,-3.918337,-8.273540,5.279026,2.224428,-7.843769,-8.416015,8.215970,-9.776294,7.322935,-6.853883,-9.387374,-2.420733,6.636079,-3.525601,1.759940,0.949146,-6.574189,7.652491,-8.243769,8.776163,-6.213648,-0.435013,6.006694,-0.383249,8.007156,-8.169996,-1.684485,-9.186956,1.949643,8.382713,4.293190,-3.758114,-8.924365,6.251789,5.228926,2.898983,-9.465343,4.373644,-4.518816,-4.328059,-7.101841,0.742997,-5.215967,-9.492458,6.190518,-1.690117,7.487440,-9.805138,-6.580714,-7.405765,3.221600,4.133920,3.909660,7.297207,-0.425244,0.545228,6.645165,-0.964228,-3.165952,9.706239,-6.797715,7.998418,-2.458260,4.703996,1.123798,-2.854200,9.871423,6.690058,2.988360,-3.666886,-0.124024,-3.537305,5.705510,3.427449,-1.225971,6.910786,-2.369679,-5.818028,-0.125930,6.874437,3.864227,-9.114578,5.428210,7.523564,7.175851,5.672815,9.441677,4.676174,-0.017656,-2.248601,7.942397,4.879856,-0.520098,-7.086627,3.097503,5.251745,-5.102829,8.215159,8.190517,-9.886311,-5.634522,-6.906766,1.778254,-7.390479,3.395469,-9.847362,-6.149051,-9.704216,7.636108,-6.628971,8.182113,8.154898,2.832204,8.127150,8.806709,7.122536,3.472012,6.713044,8.886365,-7.006165,6.655500,1.590828,-1.447644,-6.230862,3.319620,-4.661869,-8.089693,-7.383887,-9.797521,-6.183419,0.679027,-0.839114,-7.684175,2.140849,9.462445,-4.518926,3.201988,9.756938,0.113228,3.842250,-5.142989,9.994094,-6.161197,-0.189436,-7.626104,-8.778249,0.936465,-0.001542,7.223312,0.453022,-3.761941,8.174833,-5.828835,3.849907,-5.265583,-8.976728,9.409288,-4.423357,8.116291,-9.084051,1.479781,7.688025,1.650126,-8.245471,3.313400,-4.322099,7.952291,0.468926,-5.498436,-3.744293,-9.733845,9.907923,-8.091838,4.367102,-6.874867,5.104606,-5.964981,-3.954175,3.270552,-2.651716,1.858362,-5.973887,-8.658498,-7.187198,-0.912264,-3.898282,5.458296,-3.713535,9.746334,1.885643,-8.471248,7.511104,5.128132,-9.508461,7.193246,2.764161,7.073545,5.498595,-5.465658,-2.967971,-4.207921,4.994728,-0.676282,-4.019791,0.005517,-4.455992,-3.924950,-8.576424,4.879686,-3.386264,-7.561226,0.651467,5.315938,8.715538,-0.368680,0.432322,-9.187381,-0.984079,-1.991812,-0.931115,-8.158786,8.169802,4.637911,-0.180374,7.382246,9.535192,6.840008,5.571659,0.154825,-1.228397,8.226719,5.423895,-0.823711,-9.400826,-6.264100,-5.520368,9.054521,5.315139,6.829389,-4.620718,2.592139,-1.018558,-6.171896,9.985455,0.106018,-3.996910,6.391548,-5.943516,5.158646,5.248304,-1.019464,6.533038,4.568635,-5.671163,-4.923261,0.773241,-0.416337,-7.770546,-5.884060,-5.566234,2.964314,-3.826029,-9.050646,-1.518197,8.630121,8.570219,4.516319,-3.268182,-8.377388,-7.395728,8.602333,-8.625011,7.517910,-6.204965,-1.157458,6.957109,3.207350,3.995125,1.558345,-6.451430,6.105172,-2.972395,-5.195179,1.290254,3.984671,2.334558,-0.364156,-9.542464,9.911168,8.650909,3.530026,4.365034,-2.419585,5.796666,1.972477,-9.405905,3.649661,9.701765,2.876811,-7.611985,-3.954051,8.624006,0.448769,2.209388,1.199137,8.655932,2.430540,3.817064,3.073567,2.592930,-0.672421,1.113004,-2.217658,8.605298,6.447450,-7.354826,-4.887165,-0.803502,0.120126,-1.780674,-9.759998,-6.518330,-6.064000,2.670977,-2.901712,-6.094005,1.389298,-8.153841,5.528817,0.907230,-9.883231,-3.339234,-9.244881,-2.148489,-9.607634,-7.956197,-6.205531,3.382324,-3.431450,-2.978700,5.448602,-2.818771,-1.584464,9.371292,-2.498621,4.440745,-7.381285,7.240767,-0.032551,3.904131,-5.430916,-9.470659,-6.766988,-3.462362,-3.413932,9.835011,0.777630,-3.778171,-4.646760,5.410039,0.106048,0.117254,6.410459,-0.760677,5.617081,-6.892123,4.728962,-3.165256,-5.305715,-1.537512,-6.958493,7.895141,2.196415,6.440502,5.427516,7.423988,5.789189,-9.063489,8.074002,2.492139,-6.442015,9.356303,-0.650163,4.695321,-9.975477,6.232343,9.496644,-2.009018,2.786635,3.641133,5.901940,7.247045,9.979001,1.952025,-4.767282,3.837127,-7.270902,-7.218911,6.384408,-7.161459,2.531501,-9.381877,-2.446191,-2.030565,-0.829504,-7.081427,-9.487399,9.390237,7.141052,-6.587606,2.990122,9.130064,-6.656833,-8.710384,8.466167,4.080557,-0.593891,-4.166511,8.947039,-2.714940,-5.980992,8.539711,-9.677154,-8.362532,0.049936,2.456176,7.163935,-4.759344,-9.933906,4.345024,3.944641,1.717578,-0.028190,-3.699871,-2.331990,3.635545,6.324495,6.918901,2.213501,-4.100340,-6.646695,7.974175,-3.383814,-8.390571,-6.403957,-9.650002,-0.407708,-4.537647,6.461739,-6.640890,5.034808,-1.650167,-4.278663,0.207128,-7.445266,-6.637752,2.727593,-4.729835,-2.933808,5.387779,-9.675773,-6.564418,8.298664,8.352488,-2.831853,-5.402966,8.042976,-2.566220,-5.384915,-6.463623,5.202971,5.478264,-2.232104,-6.432738,-8.301063,-7.968969,-0.936057,-0.631012,3.770414,-9.342015,6.167440,4.568274,-2.844396,6.067107,0.955140,9.176176,-6.039222,-8.790638,-4.953474,8.514083,-6.890415,3.261865,0.954550,-7.820764,4.760377,-3.151087,0.342449,8.385832,0.131672,-2.442534,-3.731046,-1.063489,-2.056850,-7.766489,0.049531,-1.220460,4.036462,-5.163573,8.459176,6.738692,-2.411634,6.874373,4.168497,3.627655], dtype = "float64")#candidate|235|(520,)|const|float64
call_234 = relay.TupleGetItem(func_50_call(relay.reshape(const_235.astype('float64'), [13, 8, 5]), relay.reshape(const_235.astype('uint64'), [13, 8, 5]), ), 0)
call_236 = relay.TupleGetItem(func_54_call(relay.reshape(const_235.astype('float64'), [13, 8, 5]), relay.reshape(const_235.astype('uint64'), [13, 8, 5]), ), 0)
bop_237 = relay.bitwise_and(uop_232.astype('int8'), relay.reshape(var_231.astype('int8'), relay.shape_of(uop_232))) # shape=(5, 9)
uop_240 = relay.sinh(call_234.astype('float32')) # shape=(13, 8, 5)
uop_242 = relay.sinh(call_236.astype('float32')) # shape=(13, 8, 5)
bop_243 = relay.divide(const_235.astype('float64'), relay.reshape(uop_240.astype('float64'), relay.shape_of(const_235))) # shape=(520,)
bop_246 = relay.divide(const_235.astype('float64'), relay.reshape(uop_242.astype('float64'), relay.shape_of(const_235))) # shape=(520,)
uop_247 = relay.sinh(var_231.astype('float64')) # shape=(5, 9)
var_249 = relay.var("var_249", dtype = "float64", shape = (520,))#candidate|249|(520,)|var|float64
bop_250 = relay.mod(const_235.astype('float32'), relay.reshape(var_249.astype('float32'), relay.shape_of(const_235))) # shape=(520,)
uop_253 = relay.acos(uop_247.astype('float32')) # shape=(5, 9)
uop_255 = relay.cos(bop_243.astype('float32')) # shape=(520,)
uop_257 = relay.cos(bop_246.astype('float32')) # shape=(520,)
func_50_call = mod.get_global_var('func_50')
func_54_call = mutated_mod.get_global_var('func_54')
call_258 = relay.TupleGetItem(func_50_call(relay.reshape(bop_250.astype('float64'), [13, 8, 5]), relay.reshape(call_234.astype('uint64'), [13, 8, 5]), ), 7)
call_259 = relay.TupleGetItem(func_54_call(relay.reshape(bop_250.astype('float64'), [13, 8, 5]), relay.reshape(call_234.astype('uint64'), [13, 8, 5]), ), 7)
uop_260 = relay.erf(uop_240.astype('float32')) # shape=(13, 8, 5)
uop_262 = relay.erf(uop_242.astype('float32')) # shape=(13, 8, 5)
output = relay.Tuple([bop_237,bop_250,uop_253,uop_255,call_258,uop_260,])
output2 = relay.Tuple([bop_237,bop_250,uop_253,uop_257,call_259,uop_262,])
func_263 = relay.Function([var_231,var_249,], output)
mod['func_263'] = func_263
mod = relay.transform.InferType()(mod)
var_264 = relay.var("var_264", dtype = "float64", shape = (5, 9))#candidate|264|(5, 9)|var|float64
var_265 = relay.var("var_265", dtype = "float64", shape = (520,))#candidate|265|(520,)|var|float64
output = func_263(var_264,var_265,)
func_266 = relay.Function([var_264,var_265,], output)
mutated_mod['func_266'] = func_266
mutated_mod = relay.transform.InferType()(mutated_mod)
var_268 = relay.var("var_268", dtype = "bool", shape = (15,))#candidate|268|(15,)|var|bool
var_269 = relay.var("var_269", dtype = "bool", shape = (15,))#candidate|269|(15,)|var|bool
bop_270 = relay.logical_or(var_268.astype('bool'), relay.reshape(var_269.astype('bool'), relay.shape_of(var_268))) # shape=(15,)
uop_273 = relay.log2(var_268.astype('float32')) # shape=(15,)
bop_275 = relay.minimum(bop_270.astype('uint16'), relay.reshape(var_269.astype('uint16'), relay.shape_of(bop_270))) # shape=(15,)
func_263_call = mod.get_global_var('func_263')
func_266_call = mutated_mod.get_global_var('func_266')
const_279 = relay.const([4.391833,-2.759165,8.980139,-2.579370,0.489417,9.669516,7.421912,9.865031,4.746821,0.063502,8.100200,-2.574525,-8.272290,3.771714,0.641198,-1.076612,-1.416527,-3.064433,-2.050545,-4.655104,-5.296987,6.100824,9.645450,-4.277891,-5.396977,-2.782079,0.542722,-0.590570,-2.784753,9.703289,-9.530447,-7.325449,0.309962,-3.304621,7.952420,5.960588,-8.516754,-8.765975,-3.395843,-5.007227,-5.993054,-4.397554,-9.431688,8.948632,-8.422646], dtype = "float64")#candidate|279|(45,)|const|float64
var_280 = relay.var("var_280", dtype = "float64", shape = (1, 520))#candidate|280|(1, 520)|var|float64
call_278 = relay.TupleGetItem(func_263_call(relay.reshape(const_279.astype('float64'), [5, 9]), relay.reshape(var_280.astype('float64'), [520,]), ), 3)
call_281 = relay.TupleGetItem(func_266_call(relay.reshape(const_279.astype('float64'), [5, 9]), relay.reshape(var_280.astype('float64'), [520,]), ), 3)
bop_282 = relay.less_equal(uop_273.astype('bool'), relay.reshape(bop_270.astype('bool'), relay.shape_of(uop_273))) # shape=(15,)
func_226_call = mod.get_global_var('func_226')
func_229_call = mutated_mod.get_global_var('func_229')
var_286 = relay.var("var_286", dtype = "float64", shape = (1, 5))#candidate|286|(1, 5)|var|float64
call_285 = relay.TupleGetItem(func_226_call(relay.reshape(var_286.astype('float64'), [5,])), 2)
call_287 = relay.TupleGetItem(func_229_call(relay.reshape(var_286.astype('float64'), [5,])), 2)
uop_288 = relay.asinh(bop_282.astype('float64')) # shape=(15,)
bop_290 = relay.divide(uop_288.astype('float64'), relay.reshape(uop_273.astype('float64'), relay.shape_of(uop_288))) # shape=(15,)
uop_293 = relay.atan(uop_273.astype('float64')) # shape=(15,)
uop_295 = relay.exp(uop_288.astype('float32')) # shape=(15,)
uop_297 = relay.cos(uop_295.astype('float64')) # shape=(15,)
var_299 = relay.var("var_299", dtype = "bool", shape = (15,))#candidate|299|(15,)|var|bool
bop_300 = relay.floor_divide(bop_282.astype('float64'), relay.reshape(var_299.astype('float64'), relay.shape_of(bop_282))) # shape=(15,)
uop_303 = relay.asin(bop_270.astype('float64')) # shape=(15,)
bop_305 = relay.minimum(uop_297.astype('int16'), relay.reshape(bop_290.astype('int16'), relay.shape_of(uop_297))) # shape=(15,)
output = relay.Tuple([bop_275,call_278,const_279,var_280,call_285,var_286,uop_293,bop_300,uop_303,bop_305,])
output2 = relay.Tuple([bop_275,call_281,const_279,var_280,call_287,var_286,uop_293,bop_300,uop_303,bop_305,])
func_308 = relay.Function([var_268,var_269,var_280,var_286,var_299,], output)
mod['func_308'] = func_308
mod = relay.transform.InferType()(mod)
mutated_mod['func_308'] = func_308
mutated_mod = relay.transform.InferType()(mutated_mod)
func_308_call = mutated_mod.get_global_var('func_308')
var_310 = relay.var("var_310", dtype = "bool", shape = (15,))#candidate|310|(15,)|var|bool
var_311 = relay.var("var_311", dtype = "bool", shape = (15,))#candidate|311|(15,)|var|bool
var_312 = relay.var("var_312", dtype = "float64", shape = (1, 520))#candidate|312|(1, 520)|var|float64
var_313 = relay.var("var_313", dtype = "float64", shape = (1, 5))#candidate|313|(1, 5)|var|float64
var_314 = relay.var("var_314", dtype = "bool", shape = (15,))#candidate|314|(15,)|var|bool
call_309 = func_308_call(var_310,var_311,var_312,var_313,var_314,)
output = call_309
func_315 = relay.Function([var_310,var_311,var_312,var_313,var_314,], output)
mutated_mod['func_315'] = func_315
mutated_mod = relay.transform.InferType()(mutated_mod)
var_317 = relay.var("var_317", dtype = "float32", shape = ())#candidate|317|()|var|float32
var_318 = relay.var("var_318", dtype = "float32", shape = (8, 11, 5))#candidate|318|(8, 11, 5)|var|float32
bop_319 = relay.maximum(var_317.astype('float32'), var_318.astype('float32')) # shape=(8, 11, 5)
uop_322 = relay.log10(var_318.astype('float32')) # shape=(8, 11, 5)
uop_324 = relay.atanh(var_318.astype('float32')) # shape=(8, 11, 5)
func_78_call = mod.get_global_var('func_78')
func_83_call = mutated_mod.get_global_var('func_83')
const_327 = relay.const([10,9,1,-6,-5,9,-8,-7,-5,-6,-9,-9,10,8,-8,9,2,8,3,10,10,-5,2,-1,6,-1,-10,4,8,2,-8,4,8,-3,5,-10,1,-3,-7,1,4,10,2,6,5,-4,10,5,1,8,1,10,3,-6,-8,4,1,-9,-8,-3,-4,6,-8,-3,-8,4,1,3,7,-4,8,-10,2,2,-4,-2,3,-4,10,-9,-8,-10,-6,3,-2,-10,-5,-9,-10,4,7,2,-4,10,-4,2,1,1,5,-1,-8,-7,-1,-9,-7,-7,-1,-4,-7,-8,-3,-3,4,-2,-7,-4,-3,-8,-7,-7,2,-1,-3,7,-1,-1,7,1,-7,8,3,10,5,2,7,-10,-5,1,8,-8,-2,5,2,-9,-6,-4,3,7,-2,-1,2,7,2,5,-3,-9,3,-4,-10,7,-4,-7,3,-4,-4,-3,6,-4,-7,10,-4,-6,-3,-6,-5,-8,-2,-5,-10,-1,4,1,-5,2,-5,-1,-9,10,9,-6,-6,3,-9,4,-5,-6,-10,-3,-3,8,-7,-9,-8,-10,-2,1,-1,10,1,1,-1,-9,3,8,6,4,-5,-2,1,-9,-8,-2,5,9,6,-3,7,9,-5,6,5,7,-2,-4,-8,-5,-5,5,1,-5,-6,-5,2,4,3,1,9,-7,-6,3,5,-10,9,8,1,-8,-4,1,6,-3,5,-6,-1,-5,-5,-1,-8,-1,-8,6,2,8,6,-2,-5,3,1,4,4,9,8,10,-2,-5,-9,2,7,-7,-9,5,-8,-4,8,2,-7,-1,-10,-5,1,-5,-2,-2,-8,2,-10,3,-5,-8,7,5,1,-1,7,5,5,-8,8,-4,9,-4,-2,8,4,-2,-5,6,-3,7,2,-9,4,2,7,6,3,-6,7,-2,-9,-3,-1,10,-6,5,-1,-5,-2,-4,-4,-5,-2,-1,5,1,10,-10,-5,-7,7,8,-8,2,-9,-1,7,-9,-6,-10,-8,5,-7,10,-5,-4,3,-2,-5,3,1,9,9,-8,10,4,-1,7,5,-1,-6,-3,-6,-10,-9,7,4,-7,3,7,-3,5,-1,-3,10,5,8,-3,-6,5,3,-3,-7,-1,8,-6,-9,7,3,-6,-4,7,2,-6,2,-2,-3,-3,7,-1,-3,8,-9,-2,4,5,9,-10,3,-7,-7,-1,6,-1,9,-10,10,-4,2,1,-4,-4,10,3,-1,-10,9,-9,7,-2,1,-9,-1,-9,-9,-10,9,-5,9,-4,-8,9,4,-4,9,7,7,2,-1,10,-3,4,-4,3,-6,-6,-1,-2,4,-8,-10,-4,3,3,-8,7,-1,7,9,-1,-9,9,-1,-5,-5,-9,-7,1,10,7,7,3,-4,-5,10,-10,-4,-9,10,-1,6,-1,2,5,6,-10,-8,10,3,7,-3,4,-7,-2,6,3,3,7,6,7,-5,-2,10,-5,1,-6,-10,-4,5,6,-1,-3,3,-10,-3,-8,2,1,5,-6,-9,-3,-5,8,2,1,-4,-8,6,7,-4,8,-6,9,5,10,-8,-5,-9,-2,8,6,8,1,1,1,-3,-3,-8,-4,8,8,-6,-10,7,-6,5,-9,6,-7,7,4,9,-1,-2,1,-9,-9,-4,-10,10,6,-7,-8,-4,-2,5,1,5,-10,-6,-10,-3,-2,9,10,7,7,-10,9,-2,-4,-8,10,8,-2,-2,-1,-3,-4,2,-2,-9,-5,3,-2,4,1,-3,-5,10,3,-2,-9,-5,7,-8,-8,-9,-5,-4,9,-4,-8,-1,6,-2,7,5,-9,10,10,7,4,-1,-8,5,3,-10,-6,5,4,1,1,9,6,8,-9,6,-2,-2,1,10,9,-6,10,-7,9,8,-5,2,8,9,-6,1,-1,-10,10,-6,-1,4,-4,-8,10,-2,4,-8,5,-10,-4,9,-3,8,3,2,4,3,-8,10,-8,-5,-8,5,-4,-4,-5,4,6,-1,3,10,1,7,6,-9,4,7,9,-5,6,-1,-4,10,6,8,-9,1,2,1,-7,-2,-8,-5,10,10,-1,8,-8,7,4,-5,2,2,4,-1,8,2,-2,-3,-8,8,-5,-7,7,-5,9,-3,-6,6,10,10,3,-4,-10,-9,9,8,-6,-5,8,3,1,-10,-10,-6,-8,-10,8,-7,-8,-7,-6,1,9,-6,-7,6,-6,5,-5,-8,-7,5,10,-4,-10,9,5,9,-7,10,-7,-4,-3,9,-1,-2,-3,4,10,-6,-10,-6,5,6,4,8,2,-4,-4,-4,10,-3,-4,-10,6,5,-1,8,10,-7,-10,6,8,7,-3,-7,9,-9,9,-7,5,1,-3,-1,-5,-7,-3,6,-3,3,6,-2,-10,-9,-10,3,8,-1,-5,6,-1,4,8,-7,-5,-8,10,7,2,-8,2,-10,-3,5,-5,7,-6,2,-5,6,1,2,-5,-3,4,-9,-1,1,10,-4,-9,-1,-4,-10,5,-8,1,8,-8,2,2,-2,-10,7,-1,5,1,-6,2,-1,-3,3,-8,9,-4,10,4,-9,5,4,-8,-7,-8,4,-10,-7,6,6,-3,-10,9,2,-1,8,8,1,5,-7,-10,5,7,-2,3,5,-6,2,-4,-9,-7,1,10,-1,2,8,-8,2,4,-8,-6,3,-9,-7,9,-8,-2,-2,-2,7,-3,-6,5,-8,-3,-6,-9,-3,3,-3,-8,9,-1,4,5,7,-6,1,-2,2,5,-6,-6,7,3,-2,-3,4,-1,7,-10,6,-9,7,7,-7,-7,9,4,-3,-5,9,-3,2,-9,-1,-5,-5,3,-8,-3,-5,7,-4,-9,9,9,2,9,5,4,9,3,6,7,-8,-8,-1,-6,-8,-9,1,-5,9,6,3,1,9,-7,-2,8,7,3,-3,-10,-4,-9,8,-7,-9,-2,-2,4,1,-8,-8,-2,-4,-2,4,7,-2,2,-2,6,1,-2,4,-4,-9,-5,9,-3,-4,10,-8,5,-5,9,-5,9,-2,-9,-8,1,-7,8,6,1,-8,-8,-9,-6,2,-4,-10,-5,5,-1,8,2,-8,1,4,6,4,10,-8,10,10,9,8,-4,-9,-5,5,-2,2,9,-9,9,8,7,-4,6,-4,5,8,-4,-4,-3,10,4,-3,7,-7,-3,4,10,10,5,10,5,-1,6,2,4,7,4,-8,-1,-7,-9,8,-2,-5,2,7,5,10,1,1,4,1,7,8,6,4,-8,-5,1,-5,7,-2,-10,-2,-4,6,10,-5,1,-4,-9,-2,-1,5,-1,4,-4,10,7,-5,-4,9,10,-2,-7,8,5,3,9,7,-10,-7,5,8,-9,7,4,9,1,-5,-8,-6,-4,-10,-9,-2,7,-3,7,8,-4,1,-10,4,-10,10,3,-2,5,10,8,-4,3,1,1,-8,-2,-7,-7,-4,1,6,1,10,9,-1,4,-4,-5,9,-4,-2,-5,-1,-9,1,10,-3,5,-9,5,-1,-7,8,-10,7,7,10,-8,6,7,-5,-5,10,-4,-4,10,8,2,1,-6,-8,8,4,-1,3,-8,-5,-7,-4,-5,-9,-6,4,4,10,6,2,-7,1,-9,-6,-1,1,-7,8,-5,-8,-4,9,-10,10,8,-1,10,-3,-4,-7,9,4,5,9,-1,-3,1,-4,2,8,-8,4,-9,-9,5,10,6,-1,8,-2,-9,3,-6,3,-1,-7,10,-6,-10,5,-1,-1,-9,-7,-4,-3,4,3,4,-6,-8,3,4,-10,6,-10,-10,-3,5,3,9,-4,-1,2,6,-10,3,-4,-7,-3,-10,10,-6,-9,3,4,-8,-6,7,4,-6,-8,2,-5,-8,1,-1,-10,-4,-10,8,2,2,5,-3,-5,-7,-7,-4,-7,-1,2,9,-9,-10,-3,-4,-6,-10,8,-4,-2,5,-10,-2,-3,5,-4,-9,9,-5,10,5,8,9,-9,-10,-5,-2,5,-4,-6,-6,-5,-5,-4,7,10,2,-9,5,-5,3,-1,5,-9,-4,7,-7,-3,-7,5,-3,-9,-4,-5,8,-5,1,6,-1,6,9,-3,8,-9,5,10,-10,2,-7,4,-8,-8,4,-9,-6,10,8,2,-1,10,8,5,-1,1,-5,-3,3,-6,-5,3,7,7,-7,3,3,-6,-7,6,-2,-4,-3,2,10,4,-2,-7,8,-8,4,-5,-7,7,-3,-10,1,4,4,-9,-5,-10,-9,7,2,2,-9,4,-7,-9,-3,7,-6,-7,6,-5,-8,-7,-3,-6,-2,-2,5,-6,-6,-7,6,2,-5,-2,-4,4,-3,10,-7,-2,5,9,-1,-6,-2,5,-9,1,7,-6,2,4,-1,3,3,9,9,-1,5,-7,-2,-8,-9,-6,-4,-9,4,-2,-10,-5,10,4,-1,2,-4,-4,1,-9,1,-9,10,1,-3,-9,10,7,-5,1,-6,4,2,-6,10,-5,3,-5,-2,-4,-3,-8,-6,-5,6,1,-9,-3,4,5,-5,4,4,10,10,-1,-9,-6,-3,-10,-6,-4,5,-7,-7,3,-3,-9,-10,1,4,-3,-10,-10,9,-6,10,-10,-7,3,9,5,7,-9,4,-9,1,-1,-10,-5,-10,3,-1,4,2,-2,-5,5,3,-1,3,3,-9,-6,-5,7,4,-5,3,-4,5,5,3,4,-1,2,2,-7,-10,-9,6,2,5,4,5,10,4,-3,-5,-2,9,-2,9,-3,6,6,7,-1,-7,10,-3,-3,-8,-3,-3,-1,8,5,-7,-8,10,3,4,-10,9,5,2,10,-6,3,7,-4,-1,4,-7,9,8,7,-7,5,8,-9,-7,10,-9,3,-9,5,-5,-6,-3,2,-8,-5,7,-1,9,-5,7,10,-8,-4,-1,7,-9,-10,9,-4,-10,-1,9,4,-10,-4,6,-4,8,-3,8,5,-2,-2,7,-5,-6,-6,8,1,-10,-6,-1,-3,10,-8,-4,10,9,7,7,-3,-10,-8,-10,8,-3,6,-7,-3,-5,-7,-7,6,-8,-2,-9,-10,5,-9,6,-4,-2,-5,1,5,-10,-6,-1,2,7,6,10,-5,1,-9,9,-6,-8,-1,-10,7,-3,8,5,-3,4,8,-5,-9,7,3,-5,-9,1,-4,-8,-10,1,-5,8,9,-9,-7,10,-1,6,-2,-3,4,3,-5,-6,-4,-5,-3,8,6,-1,-2,-9,-7,-6,-3,-4,-5,4,9,-6,-9,5,5,-6,-3,-6,-9,5,-9,6,10,-1,-8,6,7,4,5,7,-3,3,-1,8,-5,5,5,-7,5,1,-7,3,-9,9,4,-3,1,4,6,-2,-6,6,10,8,5,9,4,-5,6,-6,10,-1,7,10,7,4,-9,-7,-6,-6,-4,-3,8,-3,10,7,9,10,3,-9,2,-6,-6,3,7,2,-8,-2,-3,-1,4,-4,7,-9,3,-1,-1,-9,-9,8,-4,7,-3,8,-4,10,6,1,5,-4,-7,-7,-3,5,1,-7,-2,-6,-5,-5,-3,9,7,5,-1,4,5,8,8,-1,7,-3,-4,7,3,-10,-5,8,-10,-5,1,1,-9,-7,2,-5,5,9,-6,9,-6,-8,8,4,8,-9,7,8,-10,-4,-2,-3,-6,4,-9,-1,-2,8,4,7,1,4,7,5,-6,2,-9,-3,-9,5,2,5,-5,-5,6,3,-8,-5,-7,-6,3,-9,5,5,10,4,-6,-7,5,-9,4,8,9,10,-8,-10,-4,-1,-2,-5,-6,8,9,2,9,-4,-5,6,-8,1,3,-6,2,-8,-4,9,-5,4,2,9,-1,-4,-5,-10,7,10,-3,-8,4,1,2,-1,-7,-2,-5,2,-4,-10,-9,4,4,-7,-2,3,-5,-5,-4,-6,3,2,9,-4,2,-3,-8,8,-3,-10,2,-3,8], dtype = "uint16")#candidate|327|(2197,)|const|uint16
call_326 = relay.TupleGetItem(func_78_call(relay.reshape(const_327.astype('uint16'), [13, 13, 13]), relay.reshape(const_327.astype('uint16'), [13, 13, 13]), relay.reshape(const_327.astype('float64'), [13, 13, 13]), relay.reshape(const_327.astype('float64'), [13, 13, 13]), ), 1)
call_328 = relay.TupleGetItem(func_83_call(relay.reshape(const_327.astype('uint16'), [13, 13, 13]), relay.reshape(const_327.astype('uint16'), [13, 13, 13]), relay.reshape(const_327.astype('float64'), [13, 13, 13]), relay.reshape(const_327.astype('float64'), [13, 13, 13]), ), 1)
output = relay.Tuple([bop_319,uop_322,uop_324,call_326,const_327,])
output2 = relay.Tuple([bop_319,uop_322,uop_324,call_328,const_327,])
func_329 = relay.Function([var_317,var_318,], output)
mod['func_329'] = func_329
mod = relay.transform.InferType()(mod)
mutated_mod['func_329'] = func_329
mutated_mod = relay.transform.InferType()(mutated_mod)
func_329_call = mutated_mod.get_global_var('func_329')
var_331 = relay.var("var_331", dtype = "float32", shape = ())#candidate|331|()|var|float32
var_332 = relay.var("var_332", dtype = "float32", shape = (8, 11, 5))#candidate|332|(8, 11, 5)|var|float32
call_330 = func_329_call(var_331,var_332,)
output = call_330
func_333 = relay.Function([var_331,var_332,], output)
mutated_mod['func_333'] = func_333
mutated_mod = relay.transform.InferType()(mutated_mod)
const_335 = relay.const([-3.268410,-3.800820,-1.905207,-8.046501,9.379041,-4.180931,2.938083], dtype = "float64")#candidate|335|(7,)|const|float64
var_336 = relay.var("var_336", dtype = "float64", shape = (7,))#candidate|336|(7,)|var|float64
bop_337 = relay.floor_divide(const_335.astype('float64'), relay.reshape(var_336.astype('float64'), relay.shape_of(const_335))) # shape=(7,)
bop_340 = relay.equal(var_336.astype('bool'), relay.reshape(bop_337.astype('bool'), relay.shape_of(var_336))) # shape=(7,)
uop_343 = relay.acosh(const_335.astype('float64')) # shape=(7,)
uop_345 = relay.log2(uop_343.astype('float64')) # shape=(7,)
output = relay.Tuple([bop_340,uop_345,])
output2 = relay.Tuple([bop_340,uop_345,])
func_347 = relay.Function([var_336,], output)
mod['func_347'] = func_347
mod = relay.transform.InferType()(mod)
mutated_mod['func_347'] = func_347
mutated_mod = relay.transform.InferType()(mutated_mod)
var_348 = relay.var("var_348", dtype = "float64", shape = (7,))#candidate|348|(7,)|var|float64
func_347_call = mutated_mod.get_global_var('func_347')
call_349 = func_347_call(var_348)
output = call_349
func_350 = relay.Function([var_348], output)
mutated_mod['func_350'] = func_350
mutated_mod = relay.transform.InferType()(mutated_mod)
var_352 = relay.var("var_352", dtype = "float32", shape = ())#candidate|352|()|var|float32
uop_353 = relay.exp(var_352.astype('float32')) # shape=()
bop_355 = relay.add(uop_353.astype('uint16'), var_352.astype('uint16')) # shape=()
bop_358 = relay.add(bop_355.astype('int8'), uop_353.astype('int8')) # shape=()
output = relay.Tuple([bop_358,])
output2 = relay.Tuple([bop_358,])
F = relay.Function([var_352,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_352,], output2)
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
input_352= np.array(9.817886, dtype='float32')
module1.set_input('var_352', input_352)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_352, )
res3 = intrp3.evaluate()(input_352, )
res4 = intrp4.evaluate()(input_352, )
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
module5.set_input('var_352', input_352)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_352, )
res7 = intrp7.evaluate()(input_352, )
res8 = intrp8.evaluate()(input_352, )
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
module9.set_input('var_352', input_352)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_352, )
res11 = intrp11.evaluate()(input_352, )
res12 = intrp12.evaluate()(input_352, )
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
module13.set_input('var_352', input_352)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_352, )
res15 = intrp15.evaluate()(input_352, )
res16 = intrp16.evaluate()(input_352, )
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
module17.set_input('var_352', input_352)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_352, )
res19 = intrp19.evaluate()(input_352, )
res20 = intrp20.evaluate()(input_352, )
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
module21.set_input('var_352', input_352)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_352, )
res23 = intrp23.evaluate()(input_352, )
res24 = intrp24.evaluate()(input_352, )
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

'''193: TVMFuncCall
192: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::vm::VMCompiler::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
191: tvm::relay::vm::VMCompiler::Lower(tvm::IRModule, tvm::runtime::Map<tvm::Integer, tvm::Target, void, void>, tvm::Target)
190: tvm::relay::vm::VMFunctionCompiler::Compile(tvm::GlobalVar const&, tvm::relay::Function const&)
189: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
188: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
187: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::FunctionNode const*)
186: tvm::relay::vm::VMFunctionCompiler::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
185: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
184: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
183: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
182: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
181: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
180: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
179: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
178: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
177: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
176: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
175: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
174: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
173: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
172: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
171: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
170: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
169: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
168: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
167: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
166: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
165: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
164: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
163: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
162: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
161: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
160: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
159: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
158: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
157: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
156: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
155: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
154: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
153: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
152: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
151: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
150: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
149: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
148: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
147: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
146: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
145: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
144: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
143: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
142: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
141: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
140: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
139: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
138: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
137: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
136: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
135: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
134: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
133: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
132: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
131: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
130: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
129: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
128: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
127: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
126: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
125: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
124: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
123: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
122: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
121: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
120: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
119: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
118: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
117: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
116: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
115: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
114: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
113: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
112: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
111: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
110: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
109: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
108: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
107: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
106: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
105: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
104: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
103: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
102: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
101: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
100: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
99: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
98: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
97: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
96: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
95: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
94: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
93: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
92: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
91: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
90: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
89: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
88: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
87: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
86: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
85: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
84: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
83: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
82: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
81: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
80: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
79: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
78: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
77: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
76: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
75: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
74: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
73: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
72: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
71: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
70: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
69: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
68: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
67: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
66: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
65: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
64: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
63: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
62: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
61: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
60: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
59: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
58: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
57: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
56: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
55: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
54: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
53: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
52: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
51: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
50: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
49: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
48: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
47: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
46: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
45: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
44: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
43: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
42: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
41: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
40: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
39: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
38: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
37: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
36: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
35: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
34: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
33: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
32: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
31: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
30: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
29: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
28: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
27: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
26: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
25: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
24: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
22: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
21: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
20: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
19: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
18: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
17: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
15: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
14: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
11: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
8: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
7: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
6: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
5: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
4: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::CallNode const*)
3: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
1: tvm::relay::transform::DeviceAwareExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr_(tvm::relay::LetNode const*)
0: tvm::relay::vm::VMFunctionCompiler::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)

'''