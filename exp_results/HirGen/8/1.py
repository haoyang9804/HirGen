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
var_14 = relay.var("var_14", dtype = "int16", shape = (16,))#candidate|14|(16,)|var|int16
var_15 = relay.var("var_15", dtype = "int16", shape = (16,))#candidate|15|(16,)|var|int16
bop_16 = relay.maximum(var_14.astype('int16'), relay.reshape(var_15.astype('int16'), relay.shape_of(var_14))) # shape=(16,)
uop_21 = relay.cos(var_15.astype('float64')) # shape=(16,)
uop_23 = relay.log(uop_21.astype('float32')) # shape=(16,)
output = relay.Tuple([bop_16,uop_23,])
output2 = relay.Tuple([bop_16,uop_23,])
func_25 = relay.Function([var_14,var_15,], output)
mod['func_25'] = func_25
mod = relay.transform.InferType()(mod)
mutated_mod['func_25'] = func_25
mutated_mod = relay.transform.InferType()(mutated_mod)
func_25_call = mutated_mod.get_global_var('func_25')
var_27 = relay.var("var_27", dtype = "int16", shape = (16,))#candidate|27|(16,)|var|int16
var_28 = relay.var("var_28", dtype = "int16", shape = (16,))#candidate|28|(16,)|var|int16
call_26 = func_25_call(var_27,var_28,)
output = call_26
func_29 = relay.Function([var_27,var_28,], output)
mutated_mod['func_29'] = func_29
mutated_mod = relay.transform.InferType()(mutated_mod)
var_31 = relay.var("var_31", dtype = "float64", shape = (16, 5))#candidate|31|(16, 5)|var|float64
uop_32 = relay.asinh(var_31.astype('float64')) # shape=(16, 5)
bop_34 = relay.left_shift(var_31.astype('int64'), relay.reshape(uop_32.astype('int64'), relay.shape_of(var_31))) # shape=(16, 5)
uop_37 = relay.tan(var_31.astype('float64')) # shape=(16, 5)
uop_39 = relay.acos(uop_37.astype('float64')) # shape=(16, 5)
bop_42 = relay.right_shift(uop_39.astype('uint8'), relay.reshape(uop_32.astype('uint8'), relay.shape_of(uop_39))) # shape=(16, 5)
uop_45 = relay.acos(bop_42.astype('float32')) # shape=(16, 5)
uop_47 = relay.asinh(uop_37.astype('float32')) # shape=(16, 5)
var_49 = relay.var("var_49", dtype = "float32", shape = (16, 5))#candidate|49|(16, 5)|var|float32
bop_50 = relay.subtract(uop_45.astype('uint32'), relay.reshape(var_49.astype('uint32'), relay.shape_of(uop_45))) # shape=(16, 5)
var_53 = relay.var("var_53", dtype = "float64", shape = (16, 5))#candidate|53|(16, 5)|var|float64
bop_54 = relay.logical_and(uop_39.astype('bool'), relay.reshape(var_53.astype('bool'), relay.shape_of(uop_39))) # shape=(16, 5)
uop_57 = relay.exp(uop_45.astype('float32')) # shape=(16, 5)
var_59 = relay.var("var_59", dtype = "float32", shape = (16, 5))#candidate|59|(16, 5)|var|float32
bop_60 = relay.less_equal(uop_57.astype('bool'), relay.reshape(var_59.astype('bool'), relay.shape_of(uop_57))) # shape=(16, 5)
output = relay.Tuple([bop_34,uop_47,bop_50,bop_54,bop_60,])
output2 = relay.Tuple([bop_34,uop_47,bop_50,bop_54,bop_60,])
func_63 = relay.Function([var_31,var_49,var_53,var_59,], output)
mod['func_63'] = func_63
mod = relay.transform.InferType()(mod)
var_64 = relay.var("var_64", dtype = "float64", shape = (16, 5))#candidate|64|(16, 5)|var|float64
var_65 = relay.var("var_65", dtype = "float32", shape = (16, 5))#candidate|65|(16, 5)|var|float32
var_66 = relay.var("var_66", dtype = "float64", shape = (16, 5))#candidate|66|(16, 5)|var|float64
var_67 = relay.var("var_67", dtype = "float32", shape = (16, 5))#candidate|67|(16, 5)|var|float32
output = func_63(var_64,var_65,var_66,var_67,)
func_68 = relay.Function([var_64,var_65,var_66,var_67,], output)
mutated_mod['func_68'] = func_68
mutated_mod = relay.transform.InferType()(mutated_mod)
var_79 = relay.var("var_79", dtype = "float32", shape = (1,))#candidate|79|(1,)|var|float32
var_80 = relay.var("var_80", dtype = "float32", shape = (12,))#candidate|80|(12,)|var|float32
bop_81 = relay.less_equal(var_79.astype('bool'), var_80.astype('bool')) # shape=(12,)
bop_84 = relay.equal(var_79.astype('bool'), bop_81.astype('bool')) # shape=(12,)
uop_87 = relay.log2(bop_81.astype('float32')) # shape=(12,)
const_91 = relay.const([-3.999188,-6.508537,-8.094229,5.379844,-4.249499,-2.896659,6.410108,1.955897,-4.866044,-2.819998,4.245260,-4.185659], dtype = "float32")#candidate|91|(12,)|const|float32
bop_92 = relay.greater(uop_87.astype('bool'), relay.reshape(const_91.astype('bool'), relay.shape_of(uop_87))) # shape=(12,)
const_96 = relay.const([True,True,False,True,False,False,True,False,True,False,True,False], dtype = "bool")#candidate|96|(12,)|const|bool
bop_97 = relay.divide(bop_92.astype('float32'), relay.reshape(const_96.astype('float32'), relay.shape_of(bop_92))) # shape=(12,)
uop_101 = relay.atan(bop_97.astype('float32')) # shape=(12,)
output = relay.Tuple([bop_84,uop_101,])
output2 = relay.Tuple([bop_84,uop_101,])
func_105 = relay.Function([var_79,var_80,], output)
mod['func_105'] = func_105
mod = relay.transform.InferType()(mod)
mutated_mod['func_105'] = func_105
mutated_mod = relay.transform.InferType()(mutated_mod)
func_105_call = mutated_mod.get_global_var('func_105')
var_107 = relay.var("var_107", dtype = "float32", shape = (1,))#candidate|107|(1,)|var|float32
var_108 = relay.var("var_108", dtype = "float32", shape = (12,))#candidate|108|(12,)|var|float32
call_106 = func_105_call(var_107,var_108,)
output = call_106
func_109 = relay.Function([var_107,var_108,], output)
mutated_mod['func_109'] = func_109
mutated_mod = relay.transform.InferType()(mutated_mod)
const_111 = relay.const([2,-8,1,9,7,-6,8,4,-10], dtype = "uint16")#candidate|111|(9,)|const|uint16
var_112 = relay.var("var_112", dtype = "uint16", shape = (9,))#candidate|112|(9,)|var|uint16
bop_113 = relay.bitwise_or(const_111.astype('uint16'), relay.reshape(var_112.astype('uint16'), relay.shape_of(const_111))) # shape=(9,)
uop_116 = relay.cosh(var_112.astype('float32')) # shape=(9,)
bop_118 = relay.logical_or(uop_116.astype('bool'), relay.reshape(bop_113.astype('bool'), relay.shape_of(uop_116))) # shape=(9,)
uop_121 = relay.acosh(bop_118.astype('float64')) # shape=(9,)
output = relay.Tuple([uop_121,])
output2 = relay.Tuple([uop_121,])
func_123 = relay.Function([var_112,], output)
mod['func_123'] = func_123
mod = relay.transform.InferType()(mod)
var_124 = relay.var("var_124", dtype = "uint16", shape = (9,))#candidate|124|(9,)|var|uint16
output = func_123(var_124)
func_125 = relay.Function([var_124], output)
mutated_mod['func_125'] = func_125
mutated_mod = relay.transform.InferType()(mutated_mod)
const_135 = relay.const([[[4.207157,-8.329207,-3.695992,8.249967,-8.158273,-8.152388,-1.721368,-7.753320,-7.126299,4.436973],[-9.492127,-1.643166,-6.716894,1.062523,-4.889007,7.114669,8.582606,5.007383,1.557979,4.331483],[-3.328498,-8.022019,7.086312,-9.914336,-1.890909,4.166130,0.595617,-5.806939,1.344559,-1.557708]],[[-6.729769,-5.900028,7.469794,0.397688,-7.007857,-1.687044,-8.004373,-4.906021,-1.682636,7.721893],[-2.081302,4.478918,7.383193,7.996726,3.863874,-2.362649,4.070778,9.249085,6.185927,2.757411],[-2.075699,-8.265618,-1.856470,-5.627740,3.339168,6.293567,-1.590128,3.916147,-4.221665,-3.804921]],[[-5.490101,3.195747,4.261306,9.943586,5.625118,0.123765,-2.717406,-9.856890,-6.572260,7.209887],[-7.732545,4.555765,5.676995,-3.259459,3.127915,6.633270,-5.547859,5.393446,-7.472109,-4.047453],[6.785493,0.713976,0.842281,-1.992922,-1.067278,5.566144,-0.001832,5.090155,4.100874,3.341044]],[[-1.542900,-1.418744,3.042773,0.831800,-7.185106,3.093650,7.207118,-8.148109,-8.433673,-9.546532],[-8.640866,1.841187,-5.840616,6.819679,8.687816,-7.490523,9.490030,-7.068670,-8.720781,-1.092046],[7.001995,-8.598253,-3.520731,6.573293,1.034043,6.991446,6.052519,-8.175432,5.414647,6.560507]],[[-3.200213,9.240929,-4.220522,-3.769197,6.461109,-5.728344,-2.507946,0.789605,4.239483,0.268958],[-2.726592,-0.441119,0.227388,7.099671,-3.905791,5.288539,-9.018555,6.430568,-8.238061,4.364389],[-0.406776,-2.876984,-9.327507,-8.641962,-8.660994,-1.598895,-7.130610,-1.274968,0.584098,3.322998]],[[-3.720126,-1.276207,0.771773,-8.327313,5.246095,9.384132,9.567256,-3.298219,8.497463,9.847902],[-7.074971,-8.389041,-5.553061,-0.947773,-2.404050,-5.629825,5.816610,1.973619,-4.242140,-3.300942],[9.527457,5.947379,-4.672062,-6.409971,-3.931002,6.775465,1.106978,-9.642618,9.065897,0.351748]],[[-6.964426,-3.321636,-9.344906,-2.701328,3.908196,-7.657584,-9.884253,8.927208,3.366948,-8.684897],[7.210940,-4.408176,4.097775,4.214308,2.563708,-3.242272,-4.037048,6.448717,-0.921911,5.001931],[1.516710,-1.138558,6.260854,-3.296556,3.361381,7.807670,-5.097715,2.253025,-9.166235,-6.336833]]], dtype = "float64")#candidate|135|(7, 3, 10)|const|float64
uop_136 = relay.erf(const_135.astype('float64')) # shape=(7, 3, 10)
var_138 = relay.var("var_138", dtype = "float64", shape = (7, 3, 10))#candidate|138|(7, 3, 10)|var|float64
bop_139 = relay.minimum(uop_136.astype('float32'), relay.reshape(var_138.astype('float32'), relay.shape_of(uop_136))) # shape=(7, 3, 10)
var_144 = relay.var("var_144", dtype = "float32", shape = (7, 3, 10))#candidate|144|(7, 3, 10)|var|float32
bop_145 = relay.bitwise_xor(bop_139.astype('uint32'), relay.reshape(var_144.astype('uint32'), relay.shape_of(bop_139))) # shape=(7, 3, 10)
var_151 = relay.var("var_151", dtype = "float64", shape = (7, 3, 10))#candidate|151|(7, 3, 10)|var|float64
bop_152 = relay.equal(uop_136.astype('bool'), relay.reshape(var_151.astype('bool'), relay.shape_of(uop_136))) # shape=(7, 3, 10)
bop_156 = relay.left_shift(var_138.astype('uint16'), relay.reshape(bop_145.astype('uint16'), relay.shape_of(var_138))) # shape=(7, 3, 10)
uop_159 = relay.erf(uop_136.astype('float32')) # shape=(7, 3, 10)
uop_161 = relay.atan(uop_159.astype('float64')) # shape=(7, 3, 10)
output = relay.Tuple([bop_152,bop_156,uop_161,])
output2 = relay.Tuple([bop_152,bop_156,uop_161,])
func_163 = relay.Function([var_138,var_144,var_151,], output)
mod['func_163'] = func_163
mod = relay.transform.InferType()(mod)
var_164 = relay.var("var_164", dtype = "float64", shape = (7, 3, 10))#candidate|164|(7, 3, 10)|var|float64
var_165 = relay.var("var_165", dtype = "float32", shape = (7, 3, 10))#candidate|165|(7, 3, 10)|var|float32
var_166 = relay.var("var_166", dtype = "float64", shape = (7, 3, 10))#candidate|166|(7, 3, 10)|var|float64
output = func_163(var_164,var_165,var_166,)
func_167 = relay.Function([var_164,var_165,var_166,], output)
mutated_mod['func_167'] = func_167
mutated_mod = relay.transform.InferType()(mutated_mod)
var_177 = relay.var("var_177", dtype = "int32", shape = (1, 13))#candidate|177|(1, 13)|var|int32
var_178 = relay.var("var_178", dtype = "int32", shape = (5, 13))#candidate|178|(5, 13)|var|int32
bop_179 = relay.logical_xor(var_177.astype('int32'), var_178.astype('int32')) # shape=(5, 13)
uop_183 = relay.log10(bop_179.astype('float32')) # shape=(5, 13)
uop_185 = relay.sinh(uop_183.astype('float64')) # shape=(5, 13)
bop_188 = relay.multiply(uop_185.astype('float32'), relay.reshape(bop_179.astype('float32'), relay.shape_of(uop_185))) # shape=(5, 13)
const_191 = relay.const([[8.647378,-2.280254,3.944790,7.239165,-6.481527,7.145281,9.289249,4.796316,-9.842395,9.709413,5.496353,-5.083157,1.343869],[9.899549,-1.281917,4.370005,8.433322,0.399958,-9.728708,4.104393,2.011660,-0.993901,5.703529,-4.755653,1.916538,1.898288],[7.218292,-2.232088,-3.144132,-5.171298,-0.242623,8.444256,-8.045039,-0.353441,1.819618,9.731946,-6.444755,0.924038,7.126545],[7.265641,0.275315,2.463470,-9.195413,4.736926,4.343454,1.458810,-4.512718,4.526122,-7.361885,3.236302,-7.914741,-0.152019],[9.363477,-9.948100,9.100633,2.637600,-6.822671,1.755930,-5.918883,9.161684,-5.353676,3.495926,9.847255,9.688928,-6.200710]], dtype = "float32")#candidate|191|(5, 13)|const|float32
bop_192 = relay.maximum(uop_183.astype('float32'), relay.reshape(const_191.astype('float32'), relay.shape_of(uop_183))) # shape=(5, 13)
var_195 = relay.var("var_195", dtype = "int32", shape = (5, 13))#candidate|195|(5, 13)|var|int32
bop_196 = relay.floor_divide(bop_179.astype('float64'), relay.reshape(var_195.astype('float64'), relay.shape_of(bop_179))) # shape=(5, 13)
uop_199 = relay.log(bop_196.astype('float64')) # shape=(5, 13)
var_201 = relay.var("var_201", dtype = "float64", shape = (5, 13))#candidate|201|(5, 13)|var|float64
bop_202 = relay.right_shift(uop_199.astype('uint8'), relay.reshape(var_201.astype('uint8'), relay.shape_of(uop_199))) # shape=(5, 13)
uop_205 = relay.acos(bop_188.astype('float64')) # shape=(5, 13)
bop_207 = relay.floor_mod(uop_199.astype('float32'), relay.reshape(var_201.astype('float32'), relay.shape_of(uop_199))) # shape=(5, 13)
var_210 = relay.var("var_210", dtype = "float64", shape = (5, 13))#candidate|210|(5, 13)|var|float64
bop_211 = relay.subtract(uop_205.astype('uint16'), relay.reshape(var_210.astype('uint16'), relay.shape_of(uop_205))) # shape=(5, 13)
output = relay.Tuple([bop_192,bop_202,bop_207,bop_211,])
output2 = relay.Tuple([bop_192,bop_202,bop_207,bop_211,])
func_214 = relay.Function([var_177,var_178,var_195,var_201,var_210,], output)
mod['func_214'] = func_214
mod = relay.transform.InferType()(mod)
mutated_mod['func_214'] = func_214
mutated_mod = relay.transform.InferType()(mutated_mod)
func_214_call = mutated_mod.get_global_var('func_214')
var_216 = relay.var("var_216", dtype = "int32", shape = (1, 13))#candidate|216|(1, 13)|var|int32
var_217 = relay.var("var_217", dtype = "int32", shape = (5, 13))#candidate|217|(5, 13)|var|int32
var_218 = relay.var("var_218", dtype = "int32", shape = (5, 13))#candidate|218|(5, 13)|var|int32
var_219 = relay.var("var_219", dtype = "float64", shape = (5, 13))#candidate|219|(5, 13)|var|float64
var_220 = relay.var("var_220", dtype = "float64", shape = (5, 13))#candidate|220|(5, 13)|var|float64
call_215 = func_214_call(var_216,var_217,var_218,var_219,var_220,)
output = call_215
func_221 = relay.Function([var_216,var_217,var_218,var_219,var_220,], output)
mutated_mod['func_221'] = func_221
mutated_mod = relay.transform.InferType()(mutated_mod)
var_228 = relay.var("var_228", dtype = "float64", shape = (7, 14, 13))#candidate|228|(7, 14, 13)|var|float64
var_229 = relay.var("var_229", dtype = "float64", shape = (7, 14, 13))#candidate|229|(7, 14, 13)|var|float64
bop_230 = relay.mod(var_228.astype('float64'), relay.reshape(var_229.astype('float64'), relay.shape_of(var_228))) # shape=(7, 14, 13)
var_234 = relay.var("var_234", dtype = "float64", shape = (7, 14, 13))#candidate|234|(7, 14, 13)|var|float64
bop_235 = relay.logical_and(bop_230.astype('bool'), relay.reshape(var_234.astype('bool'), relay.shape_of(bop_230))) # shape=(7, 14, 13)
bop_240 = relay.left_shift(bop_230.astype('uint8'), relay.reshape(var_229.astype('uint8'), relay.shape_of(bop_230))) # shape=(7, 14, 13)
bop_246 = relay.power(bop_240.astype('float32'), relay.reshape(bop_235.astype('float32'), relay.shape_of(bop_240))) # shape=(7, 14, 13)
bop_249 = relay.maximum(var_234.astype('int64'), relay.reshape(bop_240.astype('int64'), relay.shape_of(var_234))) # shape=(7, 14, 13)
output = relay.Tuple([bop_246,bop_249,])
output2 = relay.Tuple([bop_246,bop_249,])
func_253 = relay.Function([var_228,var_229,var_234,], output)
mod['func_253'] = func_253
mod = relay.transform.InferType()(mod)
var_254 = relay.var("var_254", dtype = "float64", shape = (7, 14, 13))#candidate|254|(7, 14, 13)|var|float64
var_255 = relay.var("var_255", dtype = "float64", shape = (7, 14, 13))#candidate|255|(7, 14, 13)|var|float64
var_256 = relay.var("var_256", dtype = "float64", shape = (7, 14, 13))#candidate|256|(7, 14, 13)|var|float64
output = func_253(var_254,var_255,var_256,)
func_257 = relay.Function([var_254,var_255,var_256,], output)
mutated_mod['func_257'] = func_257
mutated_mod = relay.transform.InferType()(mutated_mod)
var_259 = relay.var("var_259", dtype = "float64", shape = (13, 14, 5))#candidate|259|(13, 14, 5)|var|float64
var_260 = relay.var("var_260", dtype = "float64", shape = (13, 14, 5))#candidate|260|(13, 14, 5)|var|float64
bop_261 = relay.divide(var_259.astype('float64'), relay.reshape(var_260.astype('float64'), relay.shape_of(var_259))) # shape=(13, 14, 5)
func_123_call = mod.get_global_var('func_123')
func_125_call = mutated_mod.get_global_var('func_125')
const_265 = relay.const([[-2,10,6],[8,-9,5],[2,4,-8]], dtype = "uint16")#candidate|265|(3, 3)|const|uint16
call_264 = relay.TupleGetItem(func_123_call(relay.reshape(const_265.astype('uint16'), [9,])), 0)
call_266 = relay.TupleGetItem(func_125_call(relay.reshape(const_265.astype('uint16'), [9,])), 0)
uop_267 = relay.cos(call_264.astype('float64')) # shape=(9,)
uop_269 = relay.cos(call_266.astype('float64')) # shape=(9,)
uop_270 = relay.cosh(uop_267.astype('float64')) # shape=(9,)
uop_272 = relay.cosh(uop_269.astype('float64')) # shape=(9,)
bop_273 = relay.greater(uop_270.astype('bool'), relay.reshape(const_265.astype('bool'), relay.shape_of(uop_270))) # shape=(9,)
bop_276 = relay.greater(uop_272.astype('bool'), relay.reshape(const_265.astype('bool'), relay.shape_of(uop_272))) # shape=(9,)
bop_277 = relay.not_equal(uop_267.astype('bool'), relay.reshape(const_265.astype('bool'), relay.shape_of(uop_267))) # shape=(9,)
bop_280 = relay.not_equal(uop_269.astype('bool'), relay.reshape(const_265.astype('bool'), relay.shape_of(uop_269))) # shape=(9,)
uop_282 = relay.log10(bop_273.astype('float64')) # shape=(9,)
uop_284 = relay.log10(bop_276.astype('float64')) # shape=(9,)
output = relay.Tuple([bop_261,bop_277,uop_282,])
output2 = relay.Tuple([bop_261,bop_280,uop_284,])
func_285 = relay.Function([var_259,var_260,], output)
mod['func_285'] = func_285
mod = relay.transform.InferType()(mod)
mutated_mod['func_285'] = func_285
mutated_mod = relay.transform.InferType()(mutated_mod)
func_285_call = mutated_mod.get_global_var('func_285')
var_287 = relay.var("var_287", dtype = "float64", shape = (13, 14, 5))#candidate|287|(13, 14, 5)|var|float64
var_288 = relay.var("var_288", dtype = "float64", shape = (13, 14, 5))#candidate|288|(13, 14, 5)|var|float64
call_286 = func_285_call(var_287,var_288,)
output = call_286
func_289 = relay.Function([var_287,var_288,], output)
mutated_mod['func_289'] = func_289
mutated_mod = relay.transform.InferType()(mutated_mod)
const_293 = relay.const([[-4.112936,-7.302182,8.083617,-3.871422,-2.510955,1.074636,6.649808,-3.576057,0.038276,2.208698,-6.227526,-4.422870],[3.426286,2.881653,2.854903,7.884387,-4.754139,-4.536832,-1.541536,-7.218536,-7.240740,-1.042280,-5.278116,1.857963],[2.649017,0.003204,-0.554144,0.082338,-5.588952,-5.474860,-8.559171,1.003777,8.028745,-5.822591,-6.969024,-0.508361],[-2.532795,-7.275434,6.926746,-5.805193,3.591616,4.396017,1.142404,5.366359,1.188209,-2.965477,1.770018,-9.289136],[4.674130,-8.977615,-1.421874,-7.035542,4.437195,1.133334,5.734008,-8.600307,-0.761459,-2.221612,-7.256396,9.385721]], dtype = "float32")#candidate|293|(5, 12)|const|float32
uop_294 = relay.rsqrt(const_293.astype('float32')) # shape=(5, 12)
output = uop_294
output2 = uop_294
func_296 = relay.Function([], output)
mod['func_296'] = func_296
mod = relay.transform.InferType()(mod)
output = func_296()
func_297 = relay.Function([], output)
mutated_mod['func_297'] = func_297
mutated_mod = relay.transform.InferType()(mutated_mod)
var_320 = relay.var("var_320", dtype = "float32", shape = ())#candidate|320|()|var|float32
uop_321 = relay.acosh(var_320.astype('float32')) # shape=()
var_324 = relay.var("var_324", dtype = "float32", shape = (11,))#candidate|324|(11,)|var|float32
bop_325 = relay.equal(uop_321.astype('bool'), var_324.astype('bool')) # shape=(11,)
bop_329 = relay.floor_mod(uop_321.astype('float32'), bop_325.astype('float32')) # shape=(11,)
uop_332 = relay.sqrt(var_324.astype('float32')) # shape=(11,)
bop_335 = relay.less(var_320.astype('bool'), uop_332.astype('bool')) # shape=(11,)
bop_338 = relay.logical_xor(uop_321.astype('int64'), bop_329.astype('int64')) # shape=(11,)
var_341 = relay.var("var_341", dtype = "float32", shape = (5,))#candidate|341|(5,)|var|float32
bop_342 = relay.less_equal(var_320.astype('bool'), var_341.astype('bool')) # shape=(5,)
output = relay.Tuple([bop_335,bop_338,bop_342,])
output2 = relay.Tuple([bop_335,bop_338,bop_342,])
func_346 = relay.Function([var_320,var_324,var_341,], output)
mod['func_346'] = func_346
mod = relay.transform.InferType()(mod)
var_347 = relay.var("var_347", dtype = "float32", shape = ())#candidate|347|()|var|float32
var_348 = relay.var("var_348", dtype = "float32", shape = (11,))#candidate|348|(11,)|var|float32
var_349 = relay.var("var_349", dtype = "float32", shape = (5,))#candidate|349|(5,)|var|float32
output = func_346(var_347,var_348,var_349,)
func_350 = relay.Function([var_347,var_348,var_349,], output)
mutated_mod['func_350'] = func_350
mutated_mod = relay.transform.InferType()(mutated_mod)
var_361 = relay.var("var_361", dtype = "uint32", shape = (11,))#candidate|361|(11,)|var|uint32
var_362 = relay.var("var_362", dtype = "uint32", shape = (11,))#candidate|362|(11,)|var|uint32
bop_363 = relay.left_shift(var_361.astype('uint32'), relay.reshape(var_362.astype('uint32'), relay.shape_of(var_361))) # shape=(11,)
bop_366 = relay.less_equal(var_362.astype('bool'), relay.reshape(bop_363.astype('bool'), relay.shape_of(var_362))) # shape=(11,)
func_105_call = mod.get_global_var('func_105')
func_109_call = mutated_mod.get_global_var('func_109')
const_370 = relay.const([-5.421980], dtype = "float32")#candidate|370|(1,)|const|float32
const_371 = relay.const([-9.681104,-0.004830,-6.530025,-6.233198,7.852347,-8.142830,-1.448727,5.951376,-6.258090,9.448215,0.459057,6.433600], dtype = "float32")#candidate|371|(12,)|const|float32
call_369 = relay.TupleGetItem(func_105_call(relay.reshape(const_370.astype('float32'), [1,]), relay.reshape(const_371.astype('float32'), [12,]), ), 1)
call_372 = relay.TupleGetItem(func_109_call(relay.reshape(const_370.astype('float32'), [1,]), relay.reshape(const_371.astype('float32'), [12,]), ), 1)
uop_373 = relay.sigmoid(const_371.astype('float64')) # shape=(12,)
bop_376 = relay.power(uop_373.astype('float32'), relay.reshape(const_371.astype('float32'), relay.shape_of(uop_373))) # shape=(12,)
uop_379 = relay.sin(bop_363.astype('float32')) # shape=(11,)
bop_381 = relay.bitwise_or(uop_379.astype('int32'), const_370.astype('int32')) # shape=(11,)
uop_384 = relay.rsqrt(call_369.astype('float64')) # shape=(12,)
uop_386 = relay.rsqrt(call_372.astype('float64')) # shape=(12,)
bop_387 = relay.subtract(bop_381.astype('float32'), relay.reshape(bop_366.astype('float32'), relay.shape_of(bop_381))) # shape=(11,)
bop_390 = relay.add(bop_387.astype('int32'), relay.reshape(bop_381.astype('int32'), relay.shape_of(bop_387))) # shape=(11,)
bop_395 = relay.right_shift(var_362.astype('int16'), relay.reshape(bop_363.astype('int16'), relay.shape_of(var_362))) # shape=(11,)
output = relay.Tuple([bop_376,uop_384,bop_390,bop_395,])
output2 = relay.Tuple([bop_376,uop_386,bop_390,bop_395,])
func_398 = relay.Function([var_361,var_362,], output)
mod['func_398'] = func_398
mod = relay.transform.InferType()(mod)
mutated_mod['func_398'] = func_398
mutated_mod = relay.transform.InferType()(mutated_mod)
func_398_call = mutated_mod.get_global_var('func_398')
var_400 = relay.var("var_400", dtype = "uint32", shape = (11,))#candidate|400|(11,)|var|uint32
var_401 = relay.var("var_401", dtype = "uint32", shape = (11,))#candidate|401|(11,)|var|uint32
call_399 = func_398_call(var_400,var_401,)
output = call_399
func_402 = relay.Function([var_400,var_401,], output)
mutated_mod['func_402'] = func_402
mutated_mod = relay.transform.InferType()(mutated_mod)
var_413 = relay.var("var_413", dtype = "float64", shape = (11,))#candidate|413|(11,)|var|float64
var_414 = relay.var("var_414", dtype = "float64", shape = (11,))#candidate|414|(11,)|var|float64
bop_415 = relay.mod(var_413.astype('float64'), relay.reshape(var_414.astype('float64'), relay.shape_of(var_413))) # shape=(11,)
bop_418 = relay.bitwise_xor(var_413.astype('uint64'), relay.reshape(bop_415.astype('uint64'), relay.shape_of(var_413))) # shape=(11,)
output = bop_418
output2 = bop_418
func_422 = relay.Function([var_413,var_414,], output)
mod['func_422'] = func_422
mod = relay.transform.InferType()(mod)
mutated_mod['func_422'] = func_422
mutated_mod = relay.transform.InferType()(mutated_mod)
func_422_call = mutated_mod.get_global_var('func_422')
var_424 = relay.var("var_424", dtype = "float64", shape = (11,))#candidate|424|(11,)|var|float64
var_425 = relay.var("var_425", dtype = "float64", shape = (11,))#candidate|425|(11,)|var|float64
call_423 = func_422_call(var_424,var_425,)
output = call_423
func_426 = relay.Function([var_424,var_425,], output)
mutated_mod['func_426'] = func_426
mutated_mod = relay.transform.InferType()(mutated_mod)
var_430 = relay.var("var_430", dtype = "float64", shape = (12,))#candidate|430|(12,)|var|float64
uop_431 = relay.exp(var_430.astype('float64')) # shape=(12,)
var_433 = relay.var("var_433", dtype = "float64", shape = (12,))#candidate|433|(12,)|var|float64
bop_434 = relay.floor_mod(uop_431.astype('float64'), relay.reshape(var_433.astype('float64'), relay.shape_of(uop_431))) # shape=(12,)
uop_438 = relay.sqrt(uop_431.astype('float32')) # shape=(12,)
uop_440 = relay.atanh(bop_434.astype('float32')) # shape=(12,)
bop_442 = relay.greater_equal(var_433.astype('bool'), relay.reshape(uop_431.astype('bool'), relay.shape_of(var_433))) # shape=(12,)
uop_445 = relay.erf(uop_440.astype('float32')) # shape=(12,)
bop_447 = relay.logical_xor(uop_445.astype('uint32'), relay.reshape(uop_431.astype('uint32'), relay.shape_of(uop_445))) # shape=(12,)
var_452 = relay.var("var_452", dtype = "uint32", shape = (12,))#candidate|452|(12,)|var|uint32
bop_453 = relay.left_shift(bop_447.astype('int16'), relay.reshape(var_452.astype('int16'), relay.shape_of(bop_447))) # shape=(12,)
var_456 = relay.var("var_456", dtype = "int16", shape = (12,))#candidate|456|(12,)|var|int16
bop_457 = relay.subtract(bop_453.astype('uint8'), relay.reshape(var_456.astype('uint8'), relay.shape_of(bop_453))) # shape=(12,)
output = relay.Tuple([uop_438,bop_442,bop_457,])
output2 = relay.Tuple([uop_438,bop_442,bop_457,])
func_460 = relay.Function([var_430,var_433,var_452,var_456,], output)
mod['func_460'] = func_460
mod = relay.transform.InferType()(mod)
mutated_mod['func_460'] = func_460
mutated_mod = relay.transform.InferType()(mutated_mod)
func_460_call = mutated_mod.get_global_var('func_460')
var_462 = relay.var("var_462", dtype = "float64", shape = (12,))#candidate|462|(12,)|var|float64
var_463 = relay.var("var_463", dtype = "float64", shape = (12,))#candidate|463|(12,)|var|float64
var_464 = relay.var("var_464", dtype = "uint32", shape = (12,))#candidate|464|(12,)|var|uint32
var_465 = relay.var("var_465", dtype = "int16", shape = (12,))#candidate|465|(12,)|var|int16
call_461 = func_460_call(var_462,var_463,var_464,var_465,)
output = call_461
func_466 = relay.Function([var_462,var_463,var_464,var_465,], output)
mutated_mod['func_466'] = func_466
mutated_mod = relay.transform.InferType()(mutated_mod)
func_296_call = mod.get_global_var('func_296')
func_297_call = mutated_mod.get_global_var('func_297')
call_473 = func_296_call()
call_474 = func_296_call()
func_398_call = mod.get_global_var('func_398')
func_402_call = mutated_mod.get_global_var('func_402')
const_477 = relay.const([-1,9,6,1,10,-8,-1,-4,9,1,1], dtype = "uint32")#candidate|477|(11,)|const|uint32
call_476 = relay.TupleGetItem(func_398_call(relay.reshape(const_477.astype('uint32'), [11,]), relay.reshape(const_477.astype('uint32'), [11,]), ), 1)
call_478 = relay.TupleGetItem(func_402_call(relay.reshape(const_477.astype('uint32'), [11,]), relay.reshape(const_477.astype('uint32'), [11,]), ), 1)
func_422_call = mod.get_global_var('func_422')
func_426_call = mutated_mod.get_global_var('func_426')
call_484 = func_422_call(relay.reshape(const_477.astype('float64'), [11,]), relay.reshape(const_477.astype('float64'), [11,]), )
call_485 = func_422_call(relay.reshape(const_477.astype('float64'), [11,]), relay.reshape(const_477.astype('float64'), [11,]), )
func_253_call = mod.get_global_var('func_253')
func_257_call = mutated_mod.get_global_var('func_257')
var_489 = relay.var("var_489", dtype = "float64", shape = (1274,))#candidate|489|(1274,)|var|float64
call_488 = relay.TupleGetItem(func_253_call(relay.reshape(var_489.astype('float64'), [7, 14, 13]), relay.reshape(var_489.astype('float64'), [7, 14, 13]), relay.reshape(var_489.astype('float64'), [7, 14, 13]), ), 0)
call_490 = relay.TupleGetItem(func_257_call(relay.reshape(var_489.astype('float64'), [7, 14, 13]), relay.reshape(var_489.astype('float64'), [7, 14, 13]), relay.reshape(var_489.astype('float64'), [7, 14, 13]), ), 0)
func_163_call = mod.get_global_var('func_163')
func_167_call = mutated_mod.get_global_var('func_167')
const_500 = relay.const([3.669994,3.550133,8.751428,0.873074,1.922092,-0.630002,-2.630331,-6.837443,-7.294400,8.405172,-9.763475,-2.878347,6.695357,-0.476434,-6.845311,-3.452247,-9.391427,-7.043440,-8.245840,6.228366,1.132111,6.729327,-5.104026,6.776417,5.768840,3.983059,6.294847,-1.067815,-9.840800,-0.988864,1.186653,-8.851887,-7.752873,0.002374,7.159533,-6.364887,-6.707883,-8.303356,1.994506,2.042915,-4.758214,7.393270,6.284958,-6.952960,-8.636217,-2.118461,-7.673158,4.469785,2.753635,5.869563,-1.676160,-7.465189,6.540165,5.774215,-2.661232,-8.535740,2.880478,3.521422,3.679805,3.715412,-3.849386,2.008879,-6.725949,9.226271,3.795074,-5.754762,4.401974,5.952830,5.619452,0.602805,2.364453,7.512454,6.990214,8.746148,-5.650682,-5.297554,1.076782,8.182297,4.477001,2.617466,2.660311,-9.702257,-6.347286,-5.500835,-1.824307,-2.332422,3.764719,-0.844530,-0.555454,8.459917,-5.896391,1.236035,-4.081348,-1.161762,6.778331,-3.199861,6.934006,2.545024,-9.871740,-4.338360,-8.596067,0.314635,-1.671571,-7.046149,7.975894,5.157058,-2.468826,6.750787,-0.006796,-1.683407,-1.378705,-5.961399,1.317276,1.158678,3.609753,-0.463889,-4.268221,1.129600,0.597117,7.728384,-1.068643,8.449300,8.014001,5.206012,-4.375326,-6.721080,-8.461493,-1.267749,-3.184307,-5.877345,8.431288,1.380284,-0.208239,-2.382020,7.121961,-7.185152,7.841444,-2.423476,8.981996,-0.831351,3.554662,-1.845218,0.891798,7.992044,-2.299334,-5.011543,-4.369767,-8.499861,-9.169043,-1.410996,-6.986109,3.542461,-9.776284,1.273348,8.695775,2.550774,-6.070900,9.403746,6.251394,2.451924,-7.218824,4.808603,-7.352663,9.108122,2.543224,-0.755327,-8.264430,0.710062,6.416506,-7.760711,-8.748844,6.741603,1.545791,-3.267521,4.846381,9.668525,3.214792,-4.763813,-1.600813,4.176513,3.051565,5.448680,5.620794,9.824671,5.079119,-6.398276,0.051349,-9.835776,-3.488410,8.621069,6.411744,4.670102,-9.024413,-4.631905,2.773009,-5.324774,3.865613,-2.922757,-8.488846,7.189389,-1.925431,1.879667,8.074531,-2.398210,-3.392397,3.958803,-2.623313,3.373518,-6.046511,0.145299], dtype = "float64")#candidate|500|(210,)|const|float64
call_499 = relay.TupleGetItem(func_163_call(relay.reshape(const_500.astype('float64'), [7, 3, 10]), relay.reshape(const_500.astype('float32'), [7, 3, 10]), relay.reshape(const_500.astype('float64'), [7, 3, 10]), ), 1)
call_501 = relay.TupleGetItem(func_167_call(relay.reshape(const_500.astype('float64'), [7, 3, 10]), relay.reshape(const_500.astype('float32'), [7, 3, 10]), relay.reshape(const_500.astype('float64'), [7, 3, 10]), ), 1)
uop_503 = relay.sigmoid(var_489.astype('float64')) # shape=(1274,)
uop_505 = relay.acos(uop_503.astype('float64')) # shape=(1274,)
uop_507 = relay.atan(uop_505.astype('float64')) # shape=(1274,)
uop_509 = relay.cos(uop_507.astype('float32')) # shape=(1274,)
bop_511 = relay.floor_divide(uop_503.astype('float32'), relay.reshape(uop_505.astype('float32'), relay.shape_of(uop_503))) # shape=(1274,)
uop_514 = relay.asin(uop_509.astype('float64')) # shape=(1274,)
func_123_call = mod.get_global_var('func_123')
func_125_call = mutated_mod.get_global_var('func_125')
const_520 = relay.const([1,2,-7,10,10,-1,3,-2,-1], dtype = "uint16")#candidate|520|(9,)|const|uint16
call_519 = relay.TupleGetItem(func_123_call(relay.reshape(const_520.astype('uint16'), [9,])), 0)
call_521 = relay.TupleGetItem(func_125_call(relay.reshape(const_520.astype('uint16'), [9,])), 0)
var_525 = relay.var("var_525", dtype = "float64", shape = (1274,))#candidate|525|(1274,)|var|float64
bop_526 = relay.power(uop_505.astype('float32'), relay.reshape(var_525.astype('float32'), relay.shape_of(uop_505))) # shape=(1274,)
bop_530 = relay.floor_mod(uop_509.astype('float64'), relay.reshape(bop_526.astype('float64'), relay.shape_of(uop_509))) # shape=(1274,)
uop_533 = relay.cosh(uop_514.astype('float32')) # shape=(1274,)
bop_535 = relay.add(uop_533.astype('uint16'), relay.reshape(uop_503.astype('uint16'), relay.shape_of(uop_533))) # shape=(1274,)
var_539 = relay.var("var_539", dtype = "float64", shape = (1274,))#candidate|539|(1274,)|var|float64
bop_540 = relay.less_equal(uop_514.astype('bool'), relay.reshape(var_539.astype('bool'), relay.shape_of(uop_514))) # shape=(1274,)
bop_545 = relay.logical_or(uop_503.astype('bool'), relay.reshape(uop_533.astype('bool'), relay.shape_of(uop_503))) # shape=(1274,)
bop_548 = relay.maximum(uop_509.astype('float64'), relay.reshape(uop_503.astype('float64'), relay.shape_of(uop_509))) # shape=(1274,)
output = relay.Tuple([call_473,call_476,const_477,call_484,call_488,call_499,const_500,bop_511,call_519,const_520,bop_530,bop_535,bop_540,bop_545,bop_548,])
output2 = relay.Tuple([call_474,call_478,const_477,call_485,call_490,call_501,const_500,bop_511,call_521,const_520,bop_530,bop_535,bop_540,bop_545,bop_548,])
func_554 = relay.Function([var_489,var_525,var_539,], output)
mod['func_554'] = func_554
mod = relay.transform.InferType()(mod)
var_555 = relay.var("var_555", dtype = "float64", shape = (1274,))#candidate|555|(1274,)|var|float64
var_556 = relay.var("var_556", dtype = "float64", shape = (1274,))#candidate|556|(1274,)|var|float64
var_557 = relay.var("var_557", dtype = "float64", shape = (1274,))#candidate|557|(1274,)|var|float64
output = func_554(var_555,var_556,var_557,)
func_558 = relay.Function([var_555,var_556,var_557,], output)
mutated_mod['func_558'] = func_558
mutated_mod = relay.transform.InferType()(mutated_mod)
func_296_call = mod.get_global_var('func_296')
func_297_call = mutated_mod.get_global_var('func_297')
call_562 = func_296_call()
call_563 = func_296_call()
func_398_call = mod.get_global_var('func_398')
func_402_call = mutated_mod.get_global_var('func_402')
var_573 = relay.var("var_573", dtype = "uint32", shape = (11,))#candidate|573|(11,)|var|uint32
call_572 = relay.TupleGetItem(func_398_call(relay.reshape(var_573.astype('uint32'), [11,]), relay.reshape(var_573.astype('uint32'), [11,]), ), 2)
call_574 = relay.TupleGetItem(func_402_call(relay.reshape(var_573.astype('uint32'), [11,]), relay.reshape(var_573.astype('uint32'), [11,]), ), 2)
output = relay.Tuple([call_562,call_572,var_573,])
output2 = relay.Tuple([call_563,call_574,var_573,])
func_579 = relay.Function([var_573,], output)
mod['func_579'] = func_579
mod = relay.transform.InferType()(mod)
var_580 = relay.var("var_580", dtype = "uint32", shape = (11,))#candidate|580|(11,)|var|uint32
output = func_579(var_580)
func_581 = relay.Function([var_580], output)
mutated_mod['func_581'] = func_581
mutated_mod = relay.transform.InferType()(mutated_mod)
func_296_call = mod.get_global_var('func_296')
func_297_call = mutated_mod.get_global_var('func_297')
call_587 = func_296_call()
call_588 = func_296_call()
func_214_call = mod.get_global_var('func_214')
func_221_call = mutated_mod.get_global_var('func_221')
var_590 = relay.var("var_590", dtype = "int32", shape = (13, 1))#candidate|590|(13, 1)|var|int32
var_591 = relay.var("var_591", dtype = "int32", shape = (65,))#candidate|591|(65,)|var|int32
call_589 = relay.TupleGetItem(func_214_call(relay.reshape(var_590.astype('int32'), [1, 13]), relay.reshape(var_591.astype('int32'), [5, 13]), relay.reshape(var_591.astype('int32'), [5, 13]), relay.reshape(var_591.astype('float64'), [5, 13]), relay.reshape(var_591.astype('float64'), [5, 13]), ), 0)
call_592 = relay.TupleGetItem(func_221_call(relay.reshape(var_590.astype('int32'), [1, 13]), relay.reshape(var_591.astype('int32'), [5, 13]), relay.reshape(var_591.astype('int32'), [5, 13]), relay.reshape(var_591.astype('float64'), [5, 13]), relay.reshape(var_591.astype('float64'), [5, 13]), ), 0)
uop_594 = relay.log2(call_587.astype('float64')) # shape=(5, 12)
uop_596 = relay.log2(call_588.astype('float64')) # shape=(5, 12)
bop_597 = relay.power(uop_594.astype('float64'), relay.reshape(call_587.astype('float64'), relay.shape_of(uop_594))) # shape=(5, 12)
bop_600 = relay.power(uop_596.astype('float64'), relay.reshape(call_588.astype('float64'), relay.shape_of(uop_596))) # shape=(5, 12)
bop_602 = relay.bitwise_or(bop_597.astype('int64'), relay.reshape(call_587.astype('int64'), relay.shape_of(bop_597))) # shape=(5, 12)
bop_605 = relay.bitwise_or(bop_600.astype('int64'), relay.reshape(call_588.astype('int64'), relay.shape_of(bop_600))) # shape=(5, 12)
uop_606 = relay.asinh(bop_597.astype('float32')) # shape=(5, 12)
uop_608 = relay.asinh(bop_600.astype('float32')) # shape=(5, 12)
output = relay.Tuple([call_589,var_590,var_591,bop_602,uop_606,])
output2 = relay.Tuple([call_592,var_590,var_591,bop_605,uop_608,])
func_609 = relay.Function([var_590,var_591,], output)
mod['func_609'] = func_609
mod = relay.transform.InferType()(mod)
var_610 = relay.var("var_610", dtype = "int32", shape = (13, 1))#candidate|610|(13, 1)|var|int32
var_611 = relay.var("var_611", dtype = "int32", shape = (65,))#candidate|611|(65,)|var|int32
output = func_609(var_610,var_611,)
func_612 = relay.Function([var_610,var_611,], output)
mutated_mod['func_612'] = func_612
mutated_mod = relay.transform.InferType()(mutated_mod)
func_296_call = mod.get_global_var('func_296')
func_297_call = mutated_mod.get_global_var('func_297')
call_614 = func_296_call()
call_615 = func_296_call()
var_620 = relay.var("var_620", dtype = "float32", shape = (5, 12))#candidate|620|(5, 12)|var|float32
bop_621 = relay.minimum(call_614.astype('int64'), relay.reshape(var_620.astype('int64'), relay.shape_of(call_614))) # shape=(5, 12)
bop_624 = relay.minimum(call_615.astype('int64'), relay.reshape(var_620.astype('int64'), relay.shape_of(call_615))) # shape=(5, 12)
output = bop_621
output2 = bop_624
F = relay.Function([var_620,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_620,], output2)
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
input_620= np.array([[-3.061253,-3.312839,3.381169,-2.782146,2.563521,-2.373198,7.510379,-0.862318,-9.516279,-4.486374,9.595771,-9.980349],[-3.724801,4.680290,7.128653,-2.075455,-3.588070,-6.042347,-8.156051,4.573953,-4.103999,-4.213520,3.731388,-2.401701],[6.287030,-3.240354,0.480348,1.576877,6.458075,1.123932,-8.842433,-1.472178,2.470394,3.087808,3.583205,9.239637],[-1.334878,-0.131564,3.718268,-8.769733,3.201242,3.821590,2.826026,-6.713859,2.475643,-5.784526,3.794447,-7.284561],[4.478722,4.665724,9.723194,-0.755030,-8.661458,4.978259,9.489278,-4.917343,-5.428992,-0.949772,-3.203793,-2.576037]], dtype='float32')
module1.set_input('var_620', input_620)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_620, )
res3 = intrp3.evaluate()(input_620, )
res4 = intrp4.evaluate()(input_620, )
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
module5.set_input('var_620', input_620)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_620, )
res7 = intrp7.evaluate()(input_620, )
res8 = intrp8.evaluate()(input_620, )
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
module9.set_input('var_620', input_620)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_620, )
res11 = intrp11.evaluate()(input_620, )
res12 = intrp12.evaluate()(input_620, )
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
module13.set_input('var_620', input_620)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_620, )
res15 = intrp15.evaluate()(input_620, )
res16 = intrp16.evaluate()(input_620, )
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
module17.set_input('var_620', input_620)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_620, )
res19 = intrp19.evaluate()(input_620, )
res20 = intrp20.evaluate()(input_620, )
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
module21.set_input('var_620', input_620)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_620, )
res23 = intrp23.evaluate()(input_620, )
res24 = intrp24.evaluate()(input_620, )
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

'''21: TVMFuncCall
20: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
19: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
18: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
17: tvm::relay::backend::RelayBuildModule::OptimizeImpl(tvm::IRModule)
16: tvm::transform::Pass::operator()(tvm::IRModule) const
15: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
14: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
13: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
12: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
11: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::DynamicToStatic()::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
10: tvm::relay::DynamicToStatic(tvm::relay::Function, tvm::IRModule)
9: tvm::relay::DynamicToStaticMutator::PrepareInput(tvm::RelayExpr const&)
8: tvm::transform::Pass::operator()(tvm::IRModule) const
7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
6: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
5: tvm::transform::Pass::operator()(tvm::IRModule) const
4: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
3: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
2: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
1: tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}::operator()(tvm::IRModule, tvm::transform::PassContext const&) const [clone .isra.813]
0: tvm::DiagnosticContext::Render()

'''