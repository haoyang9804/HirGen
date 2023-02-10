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
var_0 = relay.var("var_0", dtype = "float32", shape = (15, 13))#candidate|0|(15, 13)|var|float32
uop_1 = relay.log10(var_0.astype('float32')) # shape=(15, 13)
uop_3 = relay.log2(uop_1.astype('float32')) # shape=(15, 13)
bop_5 = relay.mod(uop_3.astype('float64'), relay.reshape(uop_1.astype('float64'), relay.shape_of(uop_3))) # shape=(15, 13)
uop_8 = relay.acosh(uop_1.astype('float32')) # shape=(15, 13)
bop_10 = relay.logical_or(var_0.astype('bool'), relay.reshape(uop_8.astype('bool'), relay.shape_of(var_0))) # shape=(15, 13)
bop_13 = relay.less(bop_5.astype('bool'), relay.reshape(bop_10.astype('bool'), relay.shape_of(bop_5))) # shape=(15, 13)
var_16 = relay.var("var_16", dtype = "float32", shape = (15, 13))#candidate|16|(15, 13)|var|float32
bop_17 = relay.minimum(uop_1.astype('float64'), relay.reshape(var_16.astype('float64'), relay.shape_of(uop_1))) # shape=(15, 13)
bop_20 = relay.bitwise_and(uop_1.astype('int8'), relay.reshape(var_16.astype('int8'), relay.shape_of(uop_1))) # shape=(15, 13)
bop_23 = relay.divide(bop_17.astype('float32'), relay.reshape(uop_3.astype('float32'), relay.shape_of(bop_17))) # shape=(15, 13)
uop_26 = relay.log2(bop_13.astype('float64')) # shape=(15, 13)
uop_28 = relay.exp(bop_17.astype('float32')) # shape=(15, 13)
bop_30 = relay.less_equal(uop_3.astype('bool'), relay.reshape(uop_8.astype('bool'), relay.shape_of(uop_3))) # shape=(15, 13)
bop_35 = relay.equal(uop_26.astype('bool'), relay.reshape(uop_28.astype('bool'), relay.shape_of(uop_26))) # shape=(15, 13)
uop_38 = relay.erf(uop_26.astype('float64')) # shape=(15, 13)
bop_40 = relay.greater(bop_23.astype('bool'), relay.reshape(uop_28.astype('bool'), relay.shape_of(bop_23))) # shape=(15, 13)
bop_43 = relay.greater_equal(bop_10.astype('bool'), relay.reshape(var_0.astype('bool'), relay.shape_of(bop_10))) # shape=(15, 13)
const_46 = relay.const([[-1.182167,-5.354485,7.228103,1.479627,4.750661,8.185097,3.202468,6.479198,-4.140841,6.904187,4.877710,-3.229185,1.284491],[-7.698194,-0.449460,-6.220391,1.505868,9.563586,-5.404209,-9.735090,-1.700205,-4.075314,-5.117068,-3.747332,1.874788,0.994003],[-3.619818,0.640224,6.297364,-7.290850,1.563533,-4.318435,0.675416,0.004860,-2.297856,4.665121,-4.169142,8.728516,-8.813693],[6.291452,5.832718,-2.097707,-1.474949,6.697756,5.988382,0.081796,-8.533049,0.811484,-5.145261,4.825354,-1.771810,8.523693],[-7.773595,1.839389,-6.651698,-9.634029,-2.560799,1.305845,-4.210169,-7.645114,-3.584102,-4.522476,-6.926385,-5.612098,6.210202],[2.107422,6.966271,-8.560268,6.459176,-5.051597,-2.893064,3.751185,3.168075,-0.087147,5.327477,-6.668764,-2.713050,-0.835452],[-1.550905,7.982794,-9.152780,-3.973177,7.017868,-8.854398,-3.366521,0.371746,1.942054,0.298070,5.618676,8.521251,7.358522],[9.260880,-0.032538,-0.057155,-4.992526,-9.081727,-0.507916,-9.606038,3.983632,3.280122,1.173131,7.405113,3.138074,-8.675939],[-2.474647,-8.779282,4.770517,0.574054,3.529322,-5.726685,-4.615363,6.227693,5.998426,-5.062180,-5.517902,2.615329,-3.214896],[-3.624361,0.810103,1.785914,-8.714347,6.418222,8.141822,-1.675312,-4.896592,-0.222650,9.391594,-1.656637,-4.122984,-0.223217],[-4.801137,-8.965026,-8.911152,-7.866386,0.219607,1.332426,4.810731,9.662028,9.657630,-0.208057,2.410031,-9.094347,-3.543209],[8.462271,-2.746022,9.923220,1.824966,8.190944,-3.045564,-2.627671,3.898447,-9.501665,-4.330956,7.710410,4.816773,-6.053768],[3.416342,2.175079,8.512092,-1.567602,-1.379057,-4.515530,5.632439,4.386808,6.901627,-5.374078,7.435191,-6.110656,-4.830300],[-6.674367,9.484686,-7.636615,1.045238,-7.461684,6.955014,-0.059683,-4.067699,0.458568,7.144198,4.311966,-0.244494,3.900839],[-0.793902,-6.523682,-0.509061,-2.341269,8.584224,-2.269604,0.181134,7.423365,4.171106,-6.896026,-6.644463,8.517100,2.716543]], dtype = "float32")#candidate|46|(15, 13)|const|float32
bop_47 = relay.right_shift(uop_3.astype('uint8'), relay.reshape(const_46.astype('uint8'), relay.shape_of(uop_3))) # shape=(15, 13)
bop_50 = relay.add(uop_38.astype('float32'), relay.reshape(bop_20.astype('float32'), relay.shape_of(uop_38))) # shape=(15, 13)
bop_53 = relay.add(bop_50.astype('int16'), relay.reshape(uop_26.astype('int16'), relay.shape_of(bop_50))) # shape=(15, 13)
uop_56 = relay.sinh(uop_38.astype('float32')) # shape=(15, 13)
bop_58 = relay.floor_divide(uop_56.astype('float32'), relay.reshape(bop_13.astype('float32'), relay.shape_of(uop_56))) # shape=(15, 13)
uop_61 = relay.asinh(uop_3.astype('float32')) # shape=(15, 13)
uop_63 = relay.tan(uop_56.astype('float64')) # shape=(15, 13)
var_65 = relay.var("var_65", dtype = "float64", shape = (15, 13))#candidate|65|(15, 13)|var|float64
bop_66 = relay.maximum(uop_63.astype('uint32'), relay.reshape(var_65.astype('uint32'), relay.shape_of(uop_63))) # shape=(15, 13)
bop_72 = relay.equal(bop_13.astype('bool'), relay.reshape(uop_61.astype('bool'), relay.shape_of(bop_13))) # shape=(15, 13)
bop_75 = relay.logical_xor(uop_63.astype('uint64'), relay.reshape(bop_10.astype('uint64'), relay.shape_of(uop_63))) # shape=(15, 13)
uop_79 = relay.asin(bop_75.astype('float32')) # shape=(15, 13)
const_81 = relay.const([[0.283960,-0.350449,-0.928000,1.834333,-9.142370,-8.070004,-6.583833,-1.060093,9.626647,-6.853197,-1.085304,3.126883,5.897723],[1.809955,-6.028472,2.195370,7.586439,-5.816456,-9.536473,-0.895387,-7.917493,0.594189,2.054546,-4.100120,4.420289,-2.169733],[-3.162898,0.152726,-4.917730,-7.646552,8.644250,8.382298,-7.507686,-9.460536,0.007668,8.538821,8.163014,-1.665848,-8.434806],[-7.291288,-1.336804,2.733590,-1.922564,-4.505353,0.439913,5.341654,2.724324,6.432008,-3.112421,-2.339495,-1.861945,7.828519],[-7.635564,7.779741,-1.253839,-4.570020,-9.924961,-1.915271,-0.089471,-8.541059,7.138250,3.807051,-6.488027,-4.595755,-6.822310],[-0.207174,-2.490238,9.206594,-7.889253,3.435151,-5.056886,2.292228,8.401621,5.806847,-9.078762,3.113289,0.294750,-0.341104],[-0.096451,-6.931178,5.398766,9.929116,1.058918,0.587559,0.649812,-0.897555,-8.866432,2.369843,1.868170,-6.559140,-5.721382],[-8.235608,-7.442466,2.627903,1.089022,4.070662,-1.184920,6.317723,0.045601,3.395586,2.888902,-5.838369,1.221394,-5.961266],[-7.362290,3.937971,-2.883416,-8.289489,-2.112749,7.248467,-6.369267,2.702178,-6.488869,-7.386411,7.975984,-6.709085,2.721762],[-8.954479,-6.304181,-1.656717,0.057152,1.616779,9.143183,-5.623474,7.017178,5.164611,3.128824,7.089747,4.335863,7.657377],[8.898491,-3.521137,3.338839,-6.801234,5.429474,-2.129299,-9.763779,-7.158166,-2.043910,4.459979,0.874265,6.797819,2.645689],[-9.168700,-0.768402,7.879538,6.312696,-4.648989,0.983436,6.011358,7.790345,6.789157,2.443876,7.546221,6.613512,-0.263890],[-1.028543,9.760423,-0.996661,-5.082109,4.114792,-9.101963,0.559167,8.906827,6.869447,8.069829,1.960095,-2.192523,8.882904],[-2.435151,2.771419,2.492081,-1.845322,5.688051,-1.872188,3.139167,-9.150657,-0.525079,-6.734089,-4.815881,-6.876188,5.138666],[-6.435035,4.062534,2.093561,-1.266794,-7.271245,-4.391281,-1.095739,4.411151,-5.774800,0.114964,-3.059456,-0.988279,2.021482]], dtype = "float32")#candidate|81|(15, 13)|const|float32
bop_82 = relay.right_shift(uop_79.astype('int64'), relay.reshape(const_81.astype('int64'), relay.shape_of(uop_79))) # shape=(15, 13)
bop_85 = relay.bitwise_xor(uop_56.astype('int16'), relay.reshape(uop_26.astype('int16'), relay.shape_of(uop_56))) # shape=(15, 13)
uop_88 = relay.atanh(bop_82.astype('float64')) # shape=(15, 13)
const_90 = relay.const([[9.591643,-4.615007,7.221327,0.777670,-1.631999,-5.701898,4.203995,2.020957,-1.201883,-3.656695,6.403478,7.764980,8.122395],[-4.531513,1.180502,1.073677,-2.223128,8.581211,9.273961,-7.489929,-6.467057,7.821444,-3.514119,-0.564716,-3.667618,7.329345],[5.097295,5.173720,3.286554,9.033827,7.298420,1.597338,4.991755,2.812537,3.939080,-0.791664,-3.858904,5.745085,-3.808343],[1.338334,-4.455314,3.340110,4.928863,2.339713,-0.418590,-8.534140,4.225325,1.997552,-8.031935,-5.071190,-3.435813,7.465566],[-2.367336,-2.801088,8.900210,-0.070671,-5.240509,2.060181,-4.682186,-3.722346,-0.027948,-8.051814,-0.245802,-9.649459,-4.725790],[-8.947697,4.378862,2.798372,6.735297,-9.115445,9.269664,-4.492970,-1.521738,1.238768,-9.391457,1.965148,-1.807690,9.245555],[-3.235492,3.328307,-6.256424,1.814474,-4.095583,4.662461,-4.791758,-6.590758,-0.428514,8.094989,-7.954703,3.135413,-3.399398],[2.306541,-9.326924,4.588641,-0.580314,2.875701,9.413315,-9.163554,8.750850,0.434513,-7.737999,8.782115,-6.729510,6.990026],[-7.327109,-3.171588,-2.269518,0.952880,-7.336598,2.005157,-9.622161,1.781098,6.110700,7.043910,5.755320,4.049896,-2.779424],[8.329116,1.550920,8.815511,2.591220,6.246741,-5.431253,6.736535,8.446056,6.284396,2.894643,0.012723,3.209954,-8.166928],[4.783206,-1.602040,-4.925246,3.947895,-6.768228,5.468064,-3.962289,5.089930,-5.353812,8.712258,1.965192,9.419404,9.593232],[-9.270688,9.036839,0.563370,-6.597536,-8.437684,2.121154,-9.209875,-1.560581,-4.617361,-4.170980,8.122957,-1.749775,2.348831],[-9.821331,-4.315038,4.584143,-1.625737,1.296148,-7.594445,4.974546,3.421667,-8.908506,-8.018299,0.135784,6.370550,9.408463],[-1.480820,2.604194,9.722553,-4.216438,0.366080,-7.512167,-1.319894,8.763875,8.330523,6.173859,-1.047446,9.933748,8.690536],[3.691616,-5.609544,8.251535,-0.825844,-2.498642,-0.544883,2.736541,7.832842,9.542209,-9.552228,-2.723776,0.854539,-3.560248]], dtype = "float32")#candidate|90|(15, 13)|const|float32
bop_91 = relay.bitwise_xor(uop_79.astype('uint32'), relay.reshape(const_90.astype('uint32'), relay.shape_of(uop_79))) # shape=(15, 13)
output = relay.Tuple([bop_30,bop_35,bop_40,bop_43,bop_47,bop_53,bop_58,bop_66,bop_72,bop_85,uop_88,bop_91,])
output2 = relay.Tuple([bop_30,bop_35,bop_40,bop_43,bop_47,bop_53,bop_58,bop_66,bop_72,bop_85,uop_88,bop_91,])
func_95 = relay.Function([var_0,var_16,var_65,], output)
mod['func_95'] = func_95
mod = relay.transform.InferType()(mod)
mutated_mod['func_95'] = func_95
mutated_mod = relay.transform.InferType()(mutated_mod)
func_95_call = mutated_mod.get_global_var('func_95')
var_97 = relay.var("var_97", dtype = "float32", shape = (15, 13))#candidate|97|(15, 13)|var|float32
var_98 = relay.var("var_98", dtype = "float32", shape = (15, 13))#candidate|98|(15, 13)|var|float32
var_99 = relay.var("var_99", dtype = "float64", shape = (15, 13))#candidate|99|(15, 13)|var|float64
call_96 = func_95_call(var_97,var_98,var_99,)
output = call_96
func_100 = relay.Function([var_97,var_98,var_99,], output)
mutated_mod['func_100'] = func_100
mutated_mod = relay.transform.InferType()(mutated_mod)
const_104 = relay.const([-2.653606,4.174785,-6.558088,2.812826,-7.108760,0.983939,-5.224353,-3.759464,-5.551480,1.973768,2.800535,-5.548566,-9.822716], dtype = "float64")#candidate|104|(13,)|const|float64
uop_105 = relay.asin(const_104.astype('float64')) # shape=(13,)
uop_107 = relay.acos(uop_105.astype('float64')) # shape=(13,)
uop_109 = relay.cosh(const_104.astype('float64')) # shape=(13,)
uop_111 = relay.sqrt(uop_107.astype('float64')) # shape=(13,)
bop_113 = relay.maximum(uop_107.astype('uint32'), relay.reshape(uop_111.astype('uint32'), relay.shape_of(uop_107))) # shape=(13,)
bop_116 = relay.maximum(uop_109.astype('float64'), relay.reshape(uop_107.astype('float64'), relay.shape_of(uop_109))) # shape=(13,)
uop_119 = relay.rsqrt(uop_111.astype('float64')) # shape=(13,)
bop_122 = relay.floor_divide(uop_119.astype('float64'), relay.reshape(uop_107.astype('float64'), relay.shape_of(uop_119))) # shape=(13,)
bop_125 = relay.logical_xor(bop_122.astype('uint64'), relay.reshape(bop_116.astype('uint64'), relay.shape_of(bop_122))) # shape=(13,)
uop_131 = relay.atanh(bop_113.astype('float32')) # shape=(13,)
uop_133 = relay.atan(bop_125.astype('float64')) # shape=(13,)
output = relay.Tuple([uop_131,uop_133,])
output2 = relay.Tuple([uop_131,uop_133,])
func_135 = relay.Function([], output)
mod['func_135'] = func_135
mod = relay.transform.InferType()(mod)
mutated_mod['func_135'] = func_135
mutated_mod = relay.transform.InferType()(mutated_mod)
func_135_call = mutated_mod.get_global_var('func_135')
call_136 = func_135_call()
output = call_136
func_137 = relay.Function([], output)
mutated_mod['func_137'] = func_137
mutated_mod = relay.transform.InferType()(mutated_mod)
const_141 = relay.const(-7, dtype = "int64")#candidate|141|()|const|int64
const_142 = relay.const([-5,7,1,6,-3,6,8,1,3,-9,-6,-4,-2], dtype = "int64")#candidate|142|(13,)|const|int64
bop_143 = relay.bitwise_or(const_141.astype('int64'), const_142.astype('int64')) # shape=(13,)
var_146 = relay.var("var_146", dtype = "int64", shape = (13,))#candidate|146|(13,)|var|int64
bop_147 = relay.logical_or(bop_143.astype('bool'), relay.reshape(var_146.astype('bool'), relay.shape_of(bop_143))) # shape=(13,)
uop_150 = relay.log2(bop_143.astype('float32')) # shape=(13,)
bop_152 = relay.greater_equal(uop_150.astype('bool'), relay.reshape(const_142.astype('bool'), relay.shape_of(uop_150))) # shape=(13,)
var_155 = relay.var("var_155", dtype = "bool", shape = (13,))#candidate|155|(13,)|var|bool
bop_156 = relay.divide(bop_152.astype('float64'), relay.reshape(var_155.astype('float64'), relay.shape_of(bop_152))) # shape=(13,)
bop_159 = relay.equal(bop_152.astype('bool'), relay.reshape(var_155.astype('bool'), relay.shape_of(bop_152))) # shape=(13,)
bop_163 = relay.minimum(bop_152.astype('uint64'), relay.reshape(bop_143.astype('uint64'), relay.shape_of(bop_152))) # shape=(13,)
bop_166 = relay.floor_mod(bop_156.astype('float64'), relay.reshape(bop_152.astype('float64'), relay.shape_of(bop_156))) # shape=(13,)
bop_170 = relay.greater_equal(bop_156.astype('bool'), relay.reshape(uop_150.astype('bool'), relay.shape_of(bop_156))) # shape=(13,)
uop_173 = relay.acosh(bop_143.astype('float64')) # shape=(13,)
var_175 = relay.var("var_175", dtype = "uint64", shape = (13,))#candidate|175|(13,)|var|uint64
bop_176 = relay.less(bop_163.astype('bool'), relay.reshape(var_175.astype('bool'), relay.shape_of(bop_163))) # shape=(13,)
bop_179 = relay.mod(uop_150.astype('float32'), relay.reshape(bop_163.astype('float32'), relay.shape_of(uop_150))) # shape=(13,)
uop_182 = relay.erf(bop_156.astype('float32')) # shape=(13,)
func_95_call = mod.get_global_var('func_95')
func_100_call = mutated_mod.get_global_var('func_100')
var_186 = relay.var("var_186", dtype = "float32", shape = (195,))#candidate|186|(195,)|var|float32
call_185 = relay.TupleGetItem(func_95_call(relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float64'), [15, 13]), ), 11)
call_187 = relay.TupleGetItem(func_100_call(relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float64'), [15, 13]), ), 11)
bop_188 = relay.right_shift(uop_182.astype('uint64'), relay.reshape(var_146.astype('uint64'), relay.shape_of(uop_182))) # shape=(13,)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_191 = relay.TupleGetItem(func_135_call(), 0)
call_192 = relay.TupleGetItem(func_137_call(), 0)
func_95_call = mod.get_global_var('func_95')
func_100_call = mutated_mod.get_global_var('func_100')
call_193 = relay.TupleGetItem(func_95_call(relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float64'), [15, 13]), ), 1)
call_194 = relay.TupleGetItem(func_100_call(relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float32'), [15, 13]), relay.reshape(var_186.astype('float64'), [15, 13]), ), 1)
uop_195 = relay.erf(bop_163.astype('float64')) # shape=(13,)
uop_198 = relay.log10(uop_182.astype('float32')) # shape=(13,)
bop_204 = relay.maximum(uop_198.astype('int64'), relay.reshape(var_146.astype('int64'), relay.shape_of(uop_198))) # shape=(13,)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_207 = relay.TupleGetItem(func_135_call(), 1)
call_208 = relay.TupleGetItem(func_137_call(), 1)
uop_209 = relay.acos(uop_198.astype('float64')) # shape=(13,)
uop_212 = relay.log(bop_176.astype('float32')) # shape=(13,)
bop_214 = relay.not_equal(bop_188.astype('bool'), relay.reshape(uop_150.astype('bool'), relay.shape_of(bop_188))) # shape=(13,)
bop_217 = relay.multiply(uop_209.astype('uint32'), relay.reshape(var_146.astype('uint32'), relay.shape_of(uop_209))) # shape=(13,)
bop_220 = relay.subtract(bop_217.astype('uint8'), relay.reshape(bop_179.astype('uint8'), relay.shape_of(bop_217))) # shape=(13,)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_223 = relay.TupleGetItem(func_135_call(), 1)
call_224 = relay.TupleGetItem(func_137_call(), 1)
uop_225 = relay.tan(bop_220.astype('float64')) # shape=(13,)
bop_228 = relay.logical_and(uop_225.astype('bool'), relay.reshape(bop_159.astype('bool'), relay.shape_of(uop_225))) # shape=(13,)
bop_231 = relay.power(bop_220.astype('float32'), relay.reshape(bop_143.astype('float32'), relay.shape_of(bop_220))) # shape=(13,)
uop_235 = relay.acos(uop_209.astype('float32')) # shape=(13,)
uop_237 = relay.rsqrt(uop_209.astype('float32')) # shape=(13,)
const_240 = relay.const([-8.158665,9.391525,-5.624291,5.621038,-2.543465,-3.956704,-4.681395,-1.465554,-6.643174,-7.294370,-4.232707,-1.681311,-0.863781], dtype = "float64")#candidate|240|(13,)|const|float64
bop_241 = relay.bitwise_xor(uop_225.astype('uint64'), relay.reshape(const_240.astype('uint64'), relay.shape_of(uop_225))) # shape=(13,)
uop_244 = relay.tan(bop_228.astype('float32')) # shape=(13,)
bop_246 = relay.less(bop_228.astype('bool'), relay.reshape(uop_195.astype('bool'), relay.shape_of(bop_228))) # shape=(13,)
output = relay.Tuple([bop_147,bop_166,bop_170,uop_173,call_185,var_186,call_191,call_193,bop_204,call_207,uop_212,bop_214,call_223,bop_231,uop_235,uop_237,bop_241,uop_244,bop_246,])
output2 = relay.Tuple([bop_147,bop_166,bop_170,uop_173,call_187,var_186,call_192,call_194,bop_204,call_208,uop_212,bop_214,call_224,bop_231,uop_235,uop_237,bop_241,uop_244,bop_246,])
func_251 = relay.Function([var_146,var_155,var_175,var_186,], output)
mod['func_251'] = func_251
mod = relay.transform.InferType()(mod)
var_252 = relay.var("var_252", dtype = "int64", shape = (13,))#candidate|252|(13,)|var|int64
var_253 = relay.var("var_253", dtype = "bool", shape = (13,))#candidate|253|(13,)|var|bool
var_254 = relay.var("var_254", dtype = "uint64", shape = (13,))#candidate|254|(13,)|var|uint64
var_255 = relay.var("var_255", dtype = "float32", shape = (195,))#candidate|255|(195,)|var|float32
output = func_251(var_252,var_253,var_254,var_255,)
func_256 = relay.Function([var_252,var_253,var_254,var_255,], output)
mutated_mod['func_256'] = func_256
mutated_mod = relay.transform.InferType()(mutated_mod)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_258 = relay.TupleGetItem(func_135_call(), 0)
call_259 = relay.TupleGetItem(func_137_call(), 0)
output = relay.Tuple([call_258,])
output2 = relay.Tuple([call_259,])
func_264 = relay.Function([], output)
mod['func_264'] = func_264
mod = relay.transform.InferType()(mod)
mutated_mod['func_264'] = func_264
mutated_mod = relay.transform.InferType()(mutated_mod)
func_264_call = mutated_mod.get_global_var('func_264')
call_265 = func_264_call()
output = call_265
func_266 = relay.Function([], output)
mutated_mod['func_266'] = func_266
mutated_mod = relay.transform.InferType()(mutated_mod)
const_272 = relay.const([[2,-6,2,8],[10,6,5,3],[2,-9,-8,4],[2,-10,-7,-4],[2,5,-8,2],[1,7,-9,-3],[6,-1,7,-4],[-7,7,-3,-6]], dtype = "uint16")#candidate|272|(8, 4)|const|uint16
var_273 = relay.var("var_273", dtype = "uint16", shape = (8, 4))#candidate|273|(8, 4)|var|uint16
bop_274 = relay.bitwise_or(const_272.astype('uint16'), relay.reshape(var_273.astype('uint16'), relay.shape_of(const_272))) # shape=(8, 4)
uop_282 = relay.sin(const_272.astype('float64')) # shape=(8, 4)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_284 = relay.TupleGetItem(func_135_call(), 0)
call_285 = relay.TupleGetItem(func_137_call(), 0)
func_251_call = mod.get_global_var('func_251')
func_256_call = mutated_mod.get_global_var('func_256')
const_287 = relay.const([-1.182373,3.663848,2.160967,-2.118216,-0.319387,3.908471,3.555107,5.077618,2.260195,-8.617908,5.627869,6.905196,3.878344,0.240902,6.871134,8.989106,-4.640728,-5.264676,0.950049,2.369498,1.917337,-2.993719,3.650505,7.864161,-0.115688,-3.922837,0.213001,7.871642,-2.110201,7.136294,2.138429,7.914787,1.450529,4.379229,8.622033,-5.997076,-8.978805,7.196755,8.500130,-3.639631,-6.331467,5.934207,-7.432801,-8.836630,-0.099508,4.570875,-3.948532,5.234696,-8.698709,-8.677944,7.588715,9.410112,3.895526,0.665104,4.967612,-8.864214,-3.267957,9.938356,6.351735,0.088704,8.803644,8.724359,-0.255562,-7.388413,3.771234,-4.436515,6.370678,-9.046018,-2.620051,1.985580,2.718185,9.696879,-5.280191,3.735810,-3.273106,9.235465,4.291481,1.207728,2.819843,-0.620994,-9.329195,6.762259,-2.067931,-7.828673,0.065000,4.819515,0.935520,5.857016,0.577590,7.521871,9.486581,-4.530706,4.386553,-2.199536,5.797625,-8.654309,8.749746,8.576818,8.506035,-0.641279,-2.640347,-3.383196,-1.894001,0.734766,-7.694527,0.388988,7.764297,-2.852195,5.021388,5.262400,6.769440,-2.774918,5.067240,4.159053,7.568183,1.819611,0.373880,-9.287753,5.463827,6.399343,-2.041484,-3.404413,-6.870557,8.111508,9.181938,-3.272931,-9.804953,-5.342569,8.091543,-6.481271,8.663100,-8.467996,2.730969,5.304506,-1.180012,-8.620525,1.547845,-1.272320,6.928866,-8.948420,-1.323859,-6.629860,5.374992,-1.604174,-1.121652,-7.712676,-2.930147,-0.857072,5.982136,9.281573,-9.658191,-4.870595,-7.427189,-0.830114,-8.306982,-9.412355,5.111455,5.986874,6.994286,4.003568,1.606301,-5.511295,-3.577036,4.200000,-9.742345,1.589923,7.719937,-4.607422,-1.514471,6.277626,9.029325,1.206898,-0.143715,-6.440036,-3.236793,3.887976,9.517958,-0.700228,-4.296126,2.178866,-4.900764,-0.537005,-4.110384,7.541693,5.995249,-4.293772,-4.146262,4.791324,-7.806187,3.623281,-4.622754,-5.292285,-7.903603,8.767872,6.757017], dtype = "float32")#candidate|287|(195,)|const|float32
call_286 = relay.TupleGetItem(func_251_call(relay.reshape(call_284.astype('int64'), [13,]), relay.reshape(call_284.astype('bool'), [13,]), relay.reshape(call_284.astype('uint64'), [13,]), relay.reshape(const_287.astype('float32'), [195,]), ), 7)
call_288 = relay.TupleGetItem(func_256_call(relay.reshape(call_284.astype('int64'), [13,]), relay.reshape(call_284.astype('bool'), [13,]), relay.reshape(call_284.astype('uint64'), [13,]), relay.reshape(const_287.astype('float32'), [195,]), ), 7)
bop_289 = relay.less_equal(uop_282.astype('bool'), relay.reshape(bop_274.astype('bool'), relay.shape_of(uop_282))) # shape=(8, 4)
bop_292 = relay.mod(bop_289.astype('float32'), relay.reshape(bop_274.astype('float32'), relay.shape_of(bop_289))) # shape=(8, 4)
bop_295 = relay.less(uop_282.astype('bool'), relay.reshape(bop_274.astype('bool'), relay.shape_of(uop_282))) # shape=(8, 4)
bop_298 = relay.right_shift(call_284.astype('int32'), call_286.astype('int32')) # shape=(15, 13)
bop_301 = relay.right_shift(call_285.astype('int32'), call_288.astype('int32')) # shape=(15, 13)
uop_302 = relay.log10(bop_289.astype('float32')) # shape=(8, 4)
uop_306 = relay.erf(uop_282.astype('float64')) # shape=(8, 4)
output = relay.Tuple([const_287,bop_292,bop_295,bop_298,uop_302,uop_306,])
output2 = relay.Tuple([const_287,bop_292,bop_295,bop_301,uop_302,uop_306,])
func_308 = relay.Function([var_273,], output)
mod['func_308'] = func_308
mod = relay.transform.InferType()(mod)
var_309 = relay.var("var_309", dtype = "uint16", shape = (8, 4))#candidate|309|(8, 4)|var|uint16
output = func_308(var_309)
func_310 = relay.Function([var_309], output)
mutated_mod['func_310'] = func_310
mutated_mod = relay.transform.InferType()(mutated_mod)
var_314 = relay.var("var_314", dtype = "uint32", shape = (4, 2))#candidate|314|(4, 2)|var|uint32
var_315 = relay.var("var_315", dtype = "uint32", shape = (4, 2))#candidate|315|(4, 2)|var|uint32
bop_316 = relay.subtract(var_314.astype('uint32'), relay.reshape(var_315.astype('uint32'), relay.shape_of(var_314))) # shape=(4, 2)
var_319 = relay.var("var_319", dtype = "uint32", shape = (4, 2))#candidate|319|(4, 2)|var|uint32
bop_320 = relay.greater(var_314.astype('bool'), relay.reshape(var_319.astype('bool'), relay.shape_of(var_314))) # shape=(4, 2)
bop_323 = relay.left_shift(bop_320.astype('int64'), relay.reshape(var_315.astype('int64'), relay.shape_of(bop_320))) # shape=(4, 2)
bop_326 = relay.bitwise_xor(bop_320.astype('uint64'), relay.reshape(var_315.astype('uint64'), relay.shape_of(bop_320))) # shape=(4, 2)
uop_330 = relay.asin(bop_326.astype('float32')) # shape=(4, 2)
bop_332 = relay.power(uop_330.astype('float32'), relay.reshape(var_315.astype('float32'), relay.shape_of(uop_330))) # shape=(4, 2)
uop_336 = relay.asinh(uop_330.astype('float32')) # shape=(4, 2)
uop_338 = relay.rsqrt(bop_332.astype('float32')) # shape=(4, 2)
bop_340 = relay.floor_mod(bop_332.astype('float64'), relay.reshape(bop_326.astype('float64'), relay.shape_of(bop_332))) # shape=(4, 2)
uop_343 = relay.acos(uop_338.astype('float64')) # shape=(4, 2)
func_251_call = mod.get_global_var('func_251')
func_256_call = mutated_mod.get_global_var('func_256')
var_346 = relay.var("var_346", dtype = "int64", shape = (13,))#candidate|346|(13,)|var|int64
const_347 = relay.const([4.998749,-0.663573,-0.454207,4.367507,-7.158240,5.408926,-8.074084,-5.564838,-9.818903,-5.530268,5.232537,-2.685962,-5.715793,-2.509261,-1.950480,-0.421445,8.220438,3.744575,-2.042985,7.702396,-5.962738,5.633824,-2.948581,-3.459147,2.986367,2.462816,-7.593935,2.772531,-2.683107,-9.371753,9.403358,-4.054687,7.871083,-6.345330,7.341925,-1.736179,-3.253621,-4.124846,6.628707,5.336577,-6.695930,-9.460834,-2.429385,9.807181,7.491805,1.456259,-3.986855,-6.226548,-2.097552,-4.256754,-2.315911,2.315160,-1.435808,3.547226,-0.823257,-5.865474,9.428296,4.340303,-0.104759,-7.142807,-8.807646,-0.154863,-6.481696,-5.973201,-9.876669,-1.804751,-5.133312,3.210299,3.326248,-9.893024,-5.752716,-6.270938,9.630447,5.550461,-4.574451,-2.349102,-0.983772,-1.524393,1.472336,8.552086,7.507772,-8.906251,8.238498,-1.506096,-1.041691,6.787298,1.743166,-8.673402,8.611231,5.703610,6.379112,4.324874,-5.634623,-5.151204,-6.406534,1.520633,7.553404,8.110235,-5.742610,-3.367752,6.603200,6.005877,7.913900,1.825735,6.343339,1.234873,-0.202182,-0.739639,-9.017208,-7.173831,-9.319469,-9.948189,5.080859,6.205148,3.195759,-8.432415,0.990162,6.970848,7.224291,5.939457,5.352444,-9.861501,8.645385,9.153447,8.629830,4.862022,1.825000,-7.086952,5.334337,-7.939149,7.252144,6.087182,-1.166038,7.722658,-1.305495,1.625579,2.601575,6.281889,-0.443727,-2.872896,4.167014,-9.214352,0.435807,0.570565,-1.751554,2.313001,9.548396,-2.702908,-6.043625,8.315086,-1.630147,-4.171432,-6.663141,7.600554,-8.110654,7.654838,8.102914,6.445027,-0.769037,4.772953,-8.894327,-4.368008,-7.780434,-9.908961,-9.809174,3.694906,-8.154636,3.099114,5.663129,3.607521,-8.490286,4.121133,-7.681580,8.548806,-2.635364,3.364783,5.573201,-2.508997,-2.637167,-4.259422,-0.350609,-3.574812,-9.712335,-2.068985,-7.076070,7.709794,-6.902336,-3.260717,3.950130,2.116042,-5.392826,1.253611,-1.657081,6.484009,-0.472033], dtype = "float32")#candidate|347|(195,)|const|float32
call_345 = relay.TupleGetItem(func_251_call(relay.reshape(var_346.astype('int64'), [13,]), relay.reshape(var_346.astype('bool'), [13,]), relay.reshape(var_346.astype('uint64'), [13,]), relay.reshape(const_347.astype('float32'), [195,]), ), 16)
call_348 = relay.TupleGetItem(func_256_call(relay.reshape(var_346.astype('int64'), [13,]), relay.reshape(var_346.astype('bool'), [13,]), relay.reshape(var_346.astype('uint64'), [13,]), relay.reshape(const_347.astype('float32'), [195,]), ), 16)
var_350 = relay.var("var_350", dtype = "float32", shape = (4, 2))#candidate|350|(4, 2)|var|float32
bop_351 = relay.multiply(uop_336.astype('int16'), relay.reshape(var_350.astype('int16'), relay.shape_of(uop_336))) # shape=(4, 2)
var_359 = relay.var("var_359", dtype = "float32", shape = (4, 2))#candidate|359|(4, 2)|var|float32
bop_360 = relay.not_equal(uop_336.astype('bool'), relay.reshape(var_359.astype('bool'), relay.shape_of(uop_336))) # shape=(4, 2)
bop_363 = relay.bitwise_and(uop_336.astype('uint16'), relay.reshape(bop_323.astype('uint16'), relay.shape_of(uop_336))) # shape=(4, 2)
bop_367 = relay.less_equal(uop_343.astype('bool'), relay.reshape(var_359.astype('bool'), relay.shape_of(uop_343))) # shape=(4, 2)
uop_373 = relay.sinh(uop_338.astype('float64')) # shape=(4, 2)
func_135_call = mod.get_global_var('func_135')
func_137_call = mutated_mod.get_global_var('func_137')
call_375 = relay.TupleGetItem(func_135_call(), 0)
call_376 = relay.TupleGetItem(func_137_call(), 0)
uop_377 = relay.exp(const_347.astype('float64')) # shape=(195,)
bop_380 = relay.add(const_347.astype('float64'), relay.reshape(uop_377.astype('float64'), relay.shape_of(const_347))) # shape=(195,)
var_385 = relay.var("var_385", dtype = "uint16", shape = (4, 2))#candidate|385|(4, 2)|var|uint16
bop_386 = relay.equal(bop_363.astype('bool'), relay.reshape(var_385.astype('bool'), relay.shape_of(bop_363))) # shape=(4, 2)
var_389 = relay.var("var_389", dtype = "float64", shape = (195,))#candidate|389|(195,)|var|float64
bop_390 = relay.equal(uop_377.astype('bool'), relay.reshape(var_389.astype('bool'), relay.shape_of(uop_377))) # shape=(195,)
output = relay.Tuple([bop_316,bop_340,call_345,var_346,bop_351,bop_360,bop_367,uop_373,call_375,bop_380,bop_386,bop_390,])
output2 = relay.Tuple([bop_316,bop_340,call_348,var_346,bop_351,bop_360,bop_367,uop_373,call_376,bop_380,bop_386,bop_390,])
func_393 = relay.Function([var_314,var_315,var_319,var_346,var_350,var_359,var_385,var_389,], output)
mod['func_393'] = func_393
mod = relay.transform.InferType()(mod)
mutated_mod['func_393'] = func_393
mutated_mod = relay.transform.InferType()(mutated_mod)
func_393_call = mutated_mod.get_global_var('func_393')
var_395 = relay.var("var_395", dtype = "uint32", shape = (4, 2))#candidate|395|(4, 2)|var|uint32
var_396 = relay.var("var_396", dtype = "uint32", shape = (4, 2))#candidate|396|(4, 2)|var|uint32
var_397 = relay.var("var_397", dtype = "uint32", shape = (4, 2))#candidate|397|(4, 2)|var|uint32
var_398 = relay.var("var_398", dtype = "int64", shape = (13,))#candidate|398|(13,)|var|int64
var_399 = relay.var("var_399", dtype = "float32", shape = (4, 2))#candidate|399|(4, 2)|var|float32
var_400 = relay.var("var_400", dtype = "float32", shape = (4, 2))#candidate|400|(4, 2)|var|float32
var_401 = relay.var("var_401", dtype = "uint16", shape = (4, 2))#candidate|401|(4, 2)|var|uint16
var_402 = relay.var("var_402", dtype = "float64", shape = (195,))#candidate|402|(195,)|var|float64
call_394 = func_393_call(var_395,var_396,var_397,var_398,var_399,var_400,var_401,var_402,)
output = call_394
func_403 = relay.Function([var_395,var_396,var_397,var_398,var_399,var_400,var_401,var_402,], output)
mutated_mod['func_403'] = func_403
mutated_mod = relay.transform.InferType()(mutated_mod)
var_420 = relay.var("var_420", dtype = "int16", shape = (3, 7, 2))#candidate|420|(3, 7, 2)|var|int16
var_421 = relay.var("var_421", dtype = "int16", shape = (3, 7, 2))#candidate|421|(3, 7, 2)|var|int16
bop_422 = relay.right_shift(var_420.astype('int16'), relay.reshape(var_421.astype('int16'), relay.shape_of(var_420))) # shape=(3, 7, 2)
output = bop_422
output2 = bop_422
F = relay.Function([var_420,var_421,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_420,var_421,], output2)
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
input_420= np.array([[[10,4],[-10,8],[1,8],[7,2],[-7,6],[-6,-6],[5,-5]],[[6,5],[-8,1],[8,-9],[-5,1],[-3,4],[10,3],[-9,7]],[[10,2],[-2,1],[8,9],[1,6],[-4,-8],[10,6],[-3,3]]], dtype='int16')
module1.set_input('var_420', input_420)
input_421= np.array([[[7,-7],[6,2],[-6,7],[5,9],[3,-4],[4,4],[5,3]],[[-1,2],[-10,7],[-8,-3],[4,-10],[7,6],[7,-7],[-1,5]],[[2,-6],[6,-1],[6,-2],[8,1],[-7,-10],[-10,-4],[-4,8]]], dtype='int16')
module1.set_input('var_421', input_421)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_420, input_421, )
res3 = intrp3.evaluate()(input_420, input_421, )
res4 = intrp4.evaluate()(input_420, input_421, )
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
module5.set_input('var_420', input_420)
module5.set_input('var_421', input_421)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_420, input_421, )
res7 = intrp7.evaluate()(input_420, input_421, )
res8 = intrp8.evaluate()(input_420, input_421, )
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
module9.set_input('var_420', input_420)
module9.set_input('var_421', input_421)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_420, input_421, )
res11 = intrp11.evaluate()(input_420, input_421, )
res12 = intrp12.evaluate()(input_420, input_421, )
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
module13.set_input('var_420', input_420)
module13.set_input('var_421', input_421)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_420, input_421, )
res15 = intrp15.evaluate()(input_420, input_421, )
res16 = intrp16.evaluate()(input_420, input_421, )
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
module17.set_input('var_420', input_420)
module17.set_input('var_421', input_421)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_420, input_421, )
res19 = intrp19.evaluate()(input_420, input_421, )
res20 = intrp20.evaluate()(input_420, input_421, )
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
module21.set_input('var_420', input_420)
module21.set_input('var_421', input_421)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_420, input_421, )
res23 = intrp23.evaluate()(input_420, input_421, )
res24 = intrp24.evaluate()(input_420, input_421, )
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