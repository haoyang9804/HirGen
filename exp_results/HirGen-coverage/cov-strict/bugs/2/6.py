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
const_25 = relay.const([[9.041374,3.214490,2.049258,-7.429464,-7.086818,-1.416420,1.646030,3.268613,-6.174374,3.792293,9.824265,-8.596101,1.580582,-3.296573,9.307226],[-9.751527,-1.495164,-6.357956,-0.162551,-2.965628,-2.290341,6.874660,-2.595610,5.612287,-4.259933,-6.581338,-9.112045,-7.028923,5.793539,3.883961],[7.067739,4.694774,5.607953,2.032420,8.421368,-6.426444,-8.097704,0.073558,-4.848082,5.470262,2.557832,-9.032365,-9.023877,3.688896,9.460029],[1.795210,5.423641,-4.100180,-2.845347,-3.442773,7.815365,4.442134,3.220693,-8.596762,9.830485,-9.216484,2.098480,-2.348268,-9.449035,2.126470],[-0.507900,-9.962806,-9.538010,-4.217984,-7.184324,-3.151421,3.827518,6.322384,-8.702118,-9.783225,1.033738,-5.352894,5.896982,5.592107,7.138704],[-5.181354,1.831258,5.030524,4.730244,-2.536684,-4.994402,-4.323555,-4.474302,-3.798138,3.446825,-6.104724,-0.217761,-0.421949,6.491054,6.621164],[-1.138530,6.052216,-7.397331,9.337460,-0.591366,8.586237,-6.507310,4.454512,9.996435,2.888128,-4.007815,6.492740,3.683078,5.808908,7.255375],[3.264918,-1.005498,4.583521,6.859779,-2.371253,-5.457847,-9.393178,5.939393,0.910131,-4.692330,-7.995734,-5.568040,1.145916,0.558538,4.682665],[4.616647,-1.343910,7.709604,-1.263800,2.652078,-5.535381,-3.172102,-3.633289,-9.826314,-7.090096,9.566731,-0.982386,-4.019694,5.117027,-1.702013],[0.221690,8.990519,1.299737,8.820281,8.161280,-7.033473,4.583724,-9.642740,5.286351,3.170200,5.637675,-7.725059,4.256688,6.597410,-5.293522],[4.644606,4.381211,4.757555,3.694279,-9.521005,5.946136,-0.361168,1.420932,7.717845,-0.659323,-0.703503,5.217791,9.891435,-0.100147,2.234542],[-3.226684,-1.304262,1.736697,-2.965904,-4.710052,6.349784,9.437131,-3.583906,-0.275969,0.276433,7.065709,0.758825,-4.617023,0.014755,3.646440],[9.637930,6.059663,-2.740365,-8.486218,6.094771,2.523947,-9.840019,-3.793405,-7.980096,-6.329286,6.969827,5.922571,-8.216115,8.909479,-1.121802],[-6.161489,-9.320511,1.568731,-7.145569,6.809154,1.823997,-9.776378,-9.357238,-6.211601,-8.414241,-3.590952,-5.750666,-1.023372,-0.767264,-2.094301]], dtype = "float32")#candidate|25|(14, 15)|const|float32
uop_26 = relay.sin(const_25.astype('float32')) # shape=(14, 15)
bop_34 = relay.floor_mod(uop_26.astype('float64'), relay.reshape(const_25.astype('float64'), relay.shape_of(uop_26))) # shape=(14, 15)
bop_37 = relay.greater_equal(const_25.astype('bool'), relay.reshape(uop_26.astype('bool'), relay.shape_of(const_25))) # shape=(14, 15)
bop_41 = relay.minimum(uop_26.astype('int8'), relay.reshape(bop_37.astype('int8'), relay.shape_of(uop_26))) # shape=(14, 15)
bop_54 = relay.bitwise_xor(uop_26.astype('int32'), relay.reshape(bop_34.astype('int32'), relay.shape_of(uop_26))) # shape=(14, 15)
bop_57 = relay.floor_divide(bop_34.astype('float64'), relay.reshape(bop_37.astype('float64'), relay.shape_of(bop_34))) # shape=(14, 15)
uop_63 = relay.sinh(const_25.astype('float32')) # shape=(14, 15)
var_67 = relay.var("var_67", dtype = "float32", shape = (14, 15))#candidate|67|(14, 15)|var|float32
bop_68 = relay.greater(uop_26.astype('bool'), relay.reshape(var_67.astype('bool'), relay.shape_of(uop_26))) # shape=(14, 15)
var_72 = relay.var("var_72", dtype = "int8", shape = (14, 15))#candidate|72|(14, 15)|var|int8
bop_73 = relay.divide(bop_41.astype('float64'), relay.reshape(var_72.astype('float64'), relay.shape_of(bop_41))) # shape=(14, 15)
uop_81 = relay.tan(bop_41.astype('float64')) # shape=(14, 15)
uop_85 = relay.cos(bop_54.astype('float64')) # shape=(14, 15)
bop_87 = relay.subtract(uop_85.astype('uint8'), relay.reshape(var_72.astype('uint8'), relay.shape_of(uop_85))) # shape=(14, 15)
var_93 = relay.var("var_93", dtype = "float32", shape = (14, 15))#candidate|93|(14, 15)|var|float32
bop_94 = relay.less_equal(uop_63.astype('bool'), relay.reshape(var_93.astype('bool'), relay.shape_of(uop_63))) # shape=(14, 15)
const_105 = relay.const([[-8,-7,-8,-9,5,7,-1,7,10,-10,8,-4,-2,10,-5],[-3,8,5,-8,3,4,-3,10,2,1,-5,-2,1,-9,7],[4,8,9,9,10,3,1,-9,-8,-3,5,-8,-8,2,-10],[3,10,-7,7,-8,-8,-2,-9,-10,-2,-7,-2,3,-4,7],[-5,-7,4,-3,9,-1,-4,-4,2,1,-8,10,-3,-1,-9],[10,-1,-8,-2,9,4,-6,10,3,-7,-7,-3,6,1,5],[-6,-5,-4,3,6,5,9,-10,3,-3,6,-5,7,6,-9],[-8,-6,3,8,5,-7,7,4,-6,-9,-7,-9,7,-6,-3],[-4,3,6,1,3,9,8,9,-3,9,4,-10,-2,8,-10],[4,-2,5,4,-3,1,8,-9,-1,2,4,-8,8,1,-10],[6,4,-3,-7,4,4,1,-3,9,-3,-6,6,1,-3,-1],[-3,7,7,4,-6,2,-4,-2,-3,2,9,-3,1,-10,-8],[7,9,9,-5,8,1,-3,-4,3,5,5,3,-10,4,1],[10,3,4,-5,-5,2,6,-3,10,8,-1,-9,1,-1,7]], dtype = "uint8")#candidate|105|(14, 15)|const|uint8
bop_106 = relay.right_shift(bop_87.astype('int8'), relay.reshape(const_105.astype('int8'), relay.shape_of(bop_87))) # shape=(14, 15)
bop_113 = relay.bitwise_and(bop_106.astype('uint32'), relay.reshape(bop_41.astype('uint32'), relay.shape_of(bop_106))) # shape=(14, 15)
uop_116 = relay.cosh(bop_106.astype('float32')) # shape=(14, 15)
var_124 = relay.var("var_124", dtype = "float32", shape = (14, 15))#candidate|124|(14, 15)|var|float32
bop_125 = relay.logical_and(uop_116.astype('bool'), relay.reshape(var_124.astype('bool'), relay.shape_of(uop_116))) # shape=(14, 15)
uop_131 = relay.acosh(bop_113.astype('float32')) # shape=(14, 15)
const_133 = relay.const([[True,True,False,False,False,False,True,False,False,True,False,False,False,True,True],[False,True,True,False,False,False,True,False,True,False,True,True,True,False,True],[True,False,False,True,False,True,True,True,True,False,False,True,False,False,False],[True,True,True,True,True,True,True,False,True,False,False,False,True,False,True],[True,True,True,True,True,True,False,False,True,True,False,True,False,True,False],[False,False,True,True,True,False,False,False,False,False,True,True,False,False,True],[True,True,False,False,True,True,False,True,False,True,True,False,False,True,True],[False,False,False,True,True,True,True,False,False,False,False,True,True,False,True],[False,False,True,False,False,False,False,False,True,False,True,False,False,False,False],[False,False,False,False,False,True,True,True,True,True,True,True,False,False,False],[False,False,False,True,True,False,True,True,True,False,True,False,True,True,False],[True,True,True,True,True,True,False,True,False,False,False,False,True,True,False],[True,True,True,True,False,False,False,True,True,True,True,False,True,False,True],[False,True,True,True,False,False,False,True,True,False,True,False,False,False,True]], dtype = "bool")#candidate|133|(14, 15)|const|bool
bop_134 = relay.bitwise_or(bop_94.astype('uint16'), relay.reshape(const_133.astype('uint16'), relay.shape_of(bop_94))) # shape=(14, 15)
uop_142 = relay.asin(uop_131.astype('float64')) # shape=(14, 15)
output = relay.Tuple([bop_57,bop_68,bop_73,uop_81,bop_125,bop_134,uop_142,])
output2 = relay.Tuple([bop_57,bop_68,bop_73,uop_81,bop_125,bop_134,uop_142,])
func_149 = relay.Function([var_67,var_72,var_93,var_124,], output)
mod['func_149'] = func_149
mod = relay.transform.InferType()(mod)
var_150 = relay.var("var_150", dtype = "float32", shape = (14, 15))#candidate|150|(14, 15)|var|float32
var_151 = relay.var("var_151", dtype = "int8", shape = (14, 15))#candidate|151|(14, 15)|var|int8
var_152 = relay.var("var_152", dtype = "float32", shape = (14, 15))#candidate|152|(14, 15)|var|float32
var_153 = relay.var("var_153", dtype = "float32", shape = (14, 15))#candidate|153|(14, 15)|var|float32
output = func_149(var_150,var_151,var_152,var_153,)
func_154 = relay.Function([var_150,var_151,var_152,var_153,], output)
mutated_mod['func_154'] = func_154
mutated_mod = relay.transform.InferType()(mutated_mod)
var_156 = relay.var("var_156", dtype = "int64", shape = (15, 9))#candidate|156|(15, 9)|var|int64
var_157 = relay.var("var_157", dtype = "int64", shape = (15, 9))#candidate|157|(15, 9)|var|int64
bop_158 = relay.maximum(var_156.astype('int64'), relay.reshape(var_157.astype('int64'), relay.shape_of(var_156))) # shape=(15, 9)
output = bop_158
output2 = bop_158
func_166 = relay.Function([var_156,var_157,], output)
mod['func_166'] = func_166
mod = relay.transform.InferType()(mod)
var_167 = relay.var("var_167", dtype = "int64", shape = (15, 9))#candidate|167|(15, 9)|var|int64
var_168 = relay.var("var_168", dtype = "int64", shape = (15, 9))#candidate|168|(15, 9)|var|int64
output = func_166(var_167,var_168,)
func_169 = relay.Function([var_167,var_168,], output)
mutated_mod['func_169'] = func_169
mutated_mod = relay.transform.InferType()(mutated_mod)
const_202 = relay.const([[[0.869605,3.638228],[-7.969167,-2.866128],[-9.919866,8.967958]],[[5.338990,-6.783321],[-6.488202,5.920079],[-8.417941,-9.544608]],[[4.577065,-3.039780],[-2.246262,-0.186264],[-5.912327,-0.204471]]], dtype = "float64")#candidate|202|(3, 3, 2)|const|float64
uop_203 = relay.exp(const_202.astype('float64')) # shape=(3, 3, 2)
output = uop_203
output2 = uop_203
func_205 = relay.Function([], output)
mod['func_205'] = func_205
mod = relay.transform.InferType()(mod)
output = func_205()
func_206 = relay.Function([], output)
mutated_mod['func_206'] = func_206
mutated_mod = relay.transform.InferType()(mutated_mod)
var_209 = relay.var("var_209", dtype = "float64", shape = (6, 6, 14))#candidate|209|(6, 6, 14)|var|float64
uop_210 = relay.exp(var_209.astype('float64')) # shape=(6, 6, 14)
bop_214 = relay.add(var_209.astype('int64'), relay.reshape(uop_210.astype('int64'), relay.shape_of(var_209))) # shape=(6, 6, 14)
uop_217 = relay.rsqrt(uop_210.astype('float64')) # shape=(6, 6, 14)
bop_219 = relay.less(uop_217.astype('bool'), relay.reshape(bop_214.astype('bool'), relay.shape_of(uop_217))) # shape=(6, 6, 14)
bop_222 = relay.right_shift(uop_217.astype('int64'), relay.reshape(bop_219.astype('int64'), relay.shape_of(uop_217))) # shape=(6, 6, 14)
uop_228 = relay.asin(bop_219.astype('float32')) # shape=(6, 6, 14)
bop_234 = relay.multiply(uop_228.astype('uint16'), relay.reshape(var_209.astype('uint16'), relay.shape_of(uop_228))) # shape=(6, 6, 14)
output = relay.Tuple([bop_222,bop_234,])
output2 = relay.Tuple([bop_222,bop_234,])
func_237 = relay.Function([var_209,], output)
mod['func_237'] = func_237
mod = relay.transform.InferType()(mod)
var_238 = relay.var("var_238", dtype = "float64", shape = (6, 6, 14))#candidate|238|(6, 6, 14)|var|float64
output = func_237(var_238)
func_239 = relay.Function([var_238], output)
mutated_mod['func_239'] = func_239
mutated_mod = relay.transform.InferType()(mutated_mod)
var_267 = relay.var("var_267", dtype = "uint64", shape = (15, 15))#candidate|267|(15, 15)|var|uint64
var_268 = relay.var("var_268", dtype = "uint64", shape = (15, 15))#candidate|268|(15, 15)|var|uint64
bop_269 = relay.minimum(var_267.astype('uint64'), relay.reshape(var_268.astype('uint64'), relay.shape_of(var_267))) # shape=(15, 15)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_275 = func_205_call()
call_276 = func_205_call()
output = relay.Tuple([bop_269,call_275,])
output2 = relay.Tuple([bop_269,call_276,])
func_277 = relay.Function([var_267,var_268,], output)
mod['func_277'] = func_277
mod = relay.transform.InferType()(mod)
var_278 = relay.var("var_278", dtype = "uint64", shape = (15, 15))#candidate|278|(15, 15)|var|uint64
var_279 = relay.var("var_279", dtype = "uint64", shape = (15, 15))#candidate|279|(15, 15)|var|uint64
output = func_277(var_278,var_279,)
func_280 = relay.Function([var_278,var_279,], output)
mutated_mod['func_280'] = func_280
mutated_mod = relay.transform.InferType()(mutated_mod)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_284 = func_205_call()
call_285 = func_205_call()
output = call_284
output2 = call_285
func_295 = relay.Function([], output)
mod['func_295'] = func_295
mod = relay.transform.InferType()(mod)
mutated_mod['func_295'] = func_295
mutated_mod = relay.transform.InferType()(mutated_mod)
func_295_call = mutated_mod.get_global_var('func_295')
call_296 = func_295_call()
output = call_296
func_297 = relay.Function([], output)
mutated_mod['func_297'] = func_297
mutated_mod = relay.transform.InferType()(mutated_mod)
const_300 = relay.const([[5.857215,2.860936,9.959576,-3.040595,8.605686,-7.686699,5.845732,-7.110552],[8.854719,7.095859,5.452676,6.086600,-6.115706,1.382067,2.893170,8.034909],[-9.338660,-7.775446,1.672730,6.171696,-0.163877,-0.613436,3.181173,5.508810],[-4.657546,-2.834909,-7.109930,3.833461,-3.336572,4.439300,5.424020,0.486466],[-3.901375,-5.815623,-8.224241,-8.833644,0.945262,-4.230820,4.191024,7.340068]], dtype = "float32")#candidate|300|(5, 8)|const|float32
uop_301 = relay.acos(const_300.astype('float32')) # shape=(5, 8)
uop_303 = relay.sigmoid(uop_301.astype('float32')) # shape=(5, 8)
uop_306 = relay.asin(uop_301.astype('float32')) # shape=(5, 8)
bop_308 = relay.logical_xor(uop_301.astype('uint8'), relay.reshape(const_300.astype('uint8'), relay.shape_of(uop_301))) # shape=(5, 8)
func_277_call = mod.get_global_var('func_277')
func_280_call = mutated_mod.get_global_var('func_280')
const_312 = relay.const([[-3],[-2],[-1],[-7],[7],[5],[-5],[1],[-6],[-1],[-9],[-4],[5],[3],[-3],[8],[-9],[-2],[-7],[1],[6],[-2],[-3],[-3],[5],[9],[-6],[-2],[-2],[-8],[3],[-1],[-4],[-5],[2],[7],[-4],[-1],[1],[3],[-9],[-1],[-7],[5],[-10],[-4],[-8],[-7],[8],[3],[6],[9],[8],[-6],[-5],[8],[-1],[-10],[-4],[10],[7],[6],[9],[-8],[-3],[-10],[-3],[-10],[1],[2],[5],[2],[9],[-8],[7],[6],[2],[-1],[-5],[-1],[-10],[-1],[9],[3],[-9],[6],[5],[2],[9],[8],[8],[4],[-5],[-10],[-4],[3],[1],[6],[-7],[-8],[6],[-10],[1],[6],[4],[10],[5],[1],[-8],[4],[-1],[5],[-1],[1],[-2],[4],[-2],[-8],[10],[5],[-5],[5],[8],[3],[6],[-6],[-4],[-10],[2],[3],[-2],[-1],[-3],[7],[4],[-7],[-8],[10],[-6],[-10],[10],[-1],[6],[9],[5],[-3],[9],[1],[-6],[5],[-2],[-8],[3],[-4],[3],[-3],[-1],[7],[-6],[-7],[-8],[-7],[-8],[-1],[7],[-2],[-8],[-9],[10],[-2],[-9],[-6],[-9],[-10],[-9],[7],[-9],[4],[-3],[2],[1],[-10],[6],[-2],[5],[9],[-8],[6],[8],[-2],[3],[2],[-4],[-9],[-3],[8],[5],[-5],[-9],[9],[-8],[10],[3],[-3],[4],[-6],[7],[-8],[6],[9],[-8],[-3],[4],[6],[-4],[7],[-3],[-10],[-7],[-5],[-8],[-7],[-9],[-5],[2]], dtype = "uint64")#candidate|312|(225, 1)|const|uint64
call_311 = relay.TupleGetItem(func_277_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(const_312.astype('uint64'), [15, 15]), ), 0)
call_313 = relay.TupleGetItem(func_280_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(const_312.astype('uint64'), [15, 15]), ), 0)
bop_314 = relay.floor_divide(uop_303.astype('float32'), relay.reshape(uop_306.astype('float32'), relay.shape_of(uop_303))) # shape=(5, 8)
uop_317 = relay.sqrt(uop_306.astype('float64')) # shape=(5, 8)
bop_320 = relay.left_shift(bop_314.astype('uint32'), relay.reshape(uop_303.astype('uint32'), relay.shape_of(bop_314))) # shape=(5, 8)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
const_324 = relay.const([-1.866230,-6.787657,4.031843,4.478053,-7.246545,7.449859,-6.199323,4.481433,8.544308,1.756952,7.708696,-2.412901,0.382003,4.328811,6.537854,-3.507031,3.318788,-5.515544,1.552227,-5.950600,9.108196,7.653145,-6.324946,5.779689,2.541542,-6.218777,-3.642145,-2.018083,7.644318,-6.045347,6.024877,-8.437238,3.065440,6.533739,-4.140639,4.373344,-5.774727,-2.693724,1.933330,-4.096443,-5.426352,8.866971,7.028953,-4.682186,4.632725,8.772059,8.880961,6.335490,-1.796474,3.204653,-3.767323,-9.394086,9.996356,-1.759737,6.985455,8.249354,4.327783,-9.766764,6.046961,4.956072,0.923628,9.855774,-6.019569,1.799420,-0.983338,-6.242280,9.234405,0.499249,3.112592,8.197326,-0.036873,-1.047450,3.068769,-5.141340,-3.845291,-8.032481,-4.562859,-6.602708,-3.996999,-7.895942,5.086707,-6.780742,5.053779,4.879203,6.188170,8.710464,-7.608726,1.810877,-4.257876,-0.969110,-7.366746,-5.951991,-6.378015,4.185500,7.551167,0.787262,-0.136010,2.185334,8.942333,-5.152009,-2.019115,-0.771491,-6.365554,-4.005030,6.265267,0.998638,5.704484,3.722828,-9.273119,8.982269,-7.762630,-6.479574,-7.235046,-9.014620,6.658669,-9.237248,0.331814,-8.388193,-5.732521,-9.043333,0.105802,8.468777,-6.874147,-5.789590,-2.273241,-5.782693,2.850886,2.050083,-0.099043,8.221926,3.727782,4.476093,9.728783,-7.758089,9.626388,0.800228,-9.132349,5.091782,4.658859,7.789029,5.403631,-1.151798,4.374131,-2.557315,-3.817903,8.015090,9.179463,-1.112744,-5.488449,8.962780,5.795198,-6.374168,4.759100,-2.835047,4.891096,-3.744236,9.364822,2.314095,6.703260,5.014279,-1.576547,2.448493,6.280333,-8.427152,-4.076835,2.772679,-0.804321,4.805733,-6.791647,1.451645,0.087187,-2.211167,-0.711863,9.330519,-2.747970,7.304638,6.743089,-6.317171,-6.251416,-2.719472,-8.418307,4.776609,7.265028,9.220216,-3.139581,-2.370959,-2.867726,6.925339,-9.808270,-1.779447,0.124537,2.598999,-1.036440,0.478642,-4.118416,5.352915,-2.648988,-7.232272,2.002817,3.643989,-3.867647,-5.029528,-6.763633,0.645174,-8.991119,1.848076,-5.936318,9.042857,4.282755,4.070138,-7.178727,-8.313074,4.175159,9.190501,-0.999865,8.868467,-9.141657,-9.219166,5.698641,5.320679,2.442726,6.522155,-0.336662,5.041426,7.392801,0.140963,9.986288,-6.826369,-9.357502,-9.862047,-1.396103,3.199650,2.656201,-1.228059,-7.895718,-8.045834,-6.546700,2.102810,-0.865693,-0.005798,1.907594,-2.420204,-1.359840,-4.353053,-5.219433,8.441225,-6.162887,4.367362,-8.514018,-1.650090,9.895207,5.786792,-3.488783,-0.768149,9.399115,-1.841802,1.154945,7.109950,-6.034191,-9.375166,-2.759105,7.388608,8.476367,-8.775963,-0.031243,-8.003617,-3.136094,-1.639009,-6.916567,7.383385,6.740159,-9.101031,-8.430301,-1.237837,8.961716,8.112847,3.412862,-3.500943,6.916824,-3.951568,8.850931,-5.880896,-3.529420,-5.628885,2.666289,-9.659739,-2.787407,6.516778,7.056960,5.961255,2.307454,-4.300784,9.107135,4.165568,-0.679473,4.134921,-4.155888,5.654072,-6.213607,9.633548,0.684373,-3.628479,-7.222449,5.806181,-2.136157,9.380892,8.279972,-3.213826,-7.139363,-6.032296,8.986956,8.454585,6.323764,7.141116,-7.032528,6.461402,-8.907423,3.279773,-8.981050,-9.736878,7.480920,-5.829039,-8.187389,-7.152192,-5.651524,7.585393,-9.140306,9.195373,7.759650,6.699839,1.097725,6.469896,-5.802728,-9.672063,9.125557,0.648699,-0.075711,-6.214867,-6.133711,-9.730362,0.660434,4.527700,-3.533315,5.820978,7.270679,8.518312,7.567316,-7.142522,-7.668013,-5.655152,-9.575575,-9.034862,5.192666,5.884825,1.035295,-3.939274,1.448747,-6.108085,-2.426444,4.091288,-6.527310,-8.222498,0.223363,-4.524645,-9.385446,-9.494061,-6.587072,2.706559,8.366890,-9.457899,-6.574557,0.196051,-3.319924,0.210149,9.103818,-1.939196,-6.419340,9.929125,1.529531,8.517280,0.084377,-9.098871,-7.672057,2.871257,-6.207923,3.603827,4.920598,-1.663146,-0.169019,-2.919211,-6.289705,-3.132059,-6.606260,5.899315,8.505274,8.396427,-6.422565,-2.447277,5.092174,-4.369868,5.269021,0.744674,-5.243273,-5.726368,9.791541,-5.254153,3.859098,-1.703692,-8.867161,-3.627413,-3.608485,-7.505740,7.435888,-6.155197,5.939751,9.963186,-9.926554,6.990454,-0.377766,8.703893,-1.698969,-8.489432,-1.050209,8.372272,-8.329761,-2.256033,0.823581,3.490253,-7.211363,-4.737466,-1.378893,-5.933146,-8.932605,9.180514,-5.199119,-7.378735,-2.879308,-3.935549,-6.812548,4.656378,2.008510,4.294488,1.161816,-2.287029,-0.054722,3.813038,-9.193287,8.591487,3.055876,2.461157,-2.237973,-2.432050,-6.561261,4.885779,2.393075,5.082469,1.270510,4.821656,0.563846,-4.903422,-5.254068,-7.830414,2.208687,2.950896,-6.071232,-7.756184,0.257970,0.079997,-3.239909,-2.484080,-2.358129,-8.657072,-3.718225,-7.329140,3.991038,-5.494963,-0.393815,-1.617293,-8.411657,-1.556738,0.862965,3.237956,9.278960,-9.036284,-3.788387,8.081922,7.243652,-0.740206,-9.324295,9.044008,7.074759,3.973437,9.232387,1.005445,-3.722668,-6.736585,9.330435,-4.292389,5.435873,-1.809116,5.345755,-7.342867,3.939080,-6.530126], dtype = "float64")#candidate|324|(504,)|const|float64
call_323 = relay.TupleGetItem(func_237_call(relay.reshape(const_324.astype('float64'), [6, 6, 14])), 0)
call_325 = relay.TupleGetItem(func_239_call(relay.reshape(const_324.astype('float64'), [6, 6, 14])), 0)
bop_329 = relay.not_equal(uop_303.astype('bool'), relay.reshape(bop_314.astype('bool'), relay.shape_of(uop_303))) # shape=(5, 8)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
call_334 = relay.TupleGetItem(func_237_call(relay.reshape(const_324.astype('float64'), [6, 6, 14])), 1)
call_335 = relay.TupleGetItem(func_239_call(relay.reshape(const_324.astype('float64'), [6, 6, 14])), 1)
func_149_call = mod.get_global_var('func_149')
func_154_call = mutated_mod.get_global_var('func_154')
const_337 = relay.const([-5.584628,5.465739,9.504726,4.912944,-3.682573,3.250862,-5.834933,-3.264227,-9.560437,-7.361897,6.131556,-0.884135,9.733964,-0.356901,-2.835751,-0.090011,7.113148,-0.522226,-6.206529,6.035849,9.581843,1.583063,5.374860,-6.734458,-3.683899,-3.908683,-8.087791,-8.602551,-9.496006,-8.694905,2.656733,9.991422,3.930595,0.200294,-2.233951,-0.367325,-3.380352,8.897635,-1.089114,2.030791,-0.276930,-0.722844,-9.694607,4.912765,7.317881,-6.043852,-1.426088,8.434738,5.513168,-0.623204,-5.715102,-4.141726,-0.111921,-8.864746,9.262367,8.656175,6.400694,9.647746,7.609849,7.881895,-9.165207,2.944498,1.730854,-1.481649,2.512384,-0.682619,-0.294686,-6.846578,-3.240270,2.192577,-5.327115,-6.820113,-4.321033,-2.772591,-2.399092,-0.685834,0.716390,4.666804,7.565962,0.174531,-6.677711,-0.921479,-5.578865,9.479908,3.704830,4.655048,-9.868066,-6.612637,0.060029,-8.621280,-3.996701,-5.173497,-0.868272,-8.438789,-9.391359,-3.405681,-8.433997,1.605279,-9.400669,7.767473,5.739490,4.646801,-8.144109,-2.749732,-6.355367,5.889522,4.861845,-7.713416,-3.522636,8.376714,2.445890,2.611981,-7.197654,-2.271486,-5.583227,1.728129,-5.111418,7.548929,-0.625309,-8.251546,-3.813150,9.469103,-4.520773,3.190718,-9.475650,5.199797,-4.125150,7.871668,0.802016,0.395146,-4.278200,6.074304,-8.762309,6.672349,-3.843137,7.347425,-6.482136,-8.658644,2.719250,4.206499,8.926469,0.476563,7.833059,-2.949172,-7.193465,-5.988234,-4.313375,-2.111125,-6.209073,8.757383,0.893475,0.926017,6.654291,-0.399880,4.170439,7.953951,-4.549599,8.491294,1.281031,-0.389045,8.531686,-4.487319,-3.905365,-6.995249,3.235110,4.271589,4.477415,-7.435853,2.376930,0.372085,-9.961854,7.404339,2.165139,-2.401412,-0.441431,-3.184689,-9.491730,-2.925635,0.609993,6.654945,-5.674485,-7.748370,-5.453710,3.743324,8.351977,-8.872912,8.152840,8.983166,-0.408581,9.363008,-1.429370,3.507641,7.933931,-8.015858,-8.482534,0.591693,-9.444981,-1.047027,-2.031100,-1.799248,-9.988524,5.751623,-4.517358,-4.440971,-8.262541,-3.695156,-2.108935,9.413892,8.202684,-1.063024], dtype = "float32")#candidate|337|(210,)|const|float32
call_336 = relay.TupleGetItem(func_149_call(relay.reshape(const_337.astype('float32'), [14, 15]), relay.reshape(const_337.astype('int8'), [14, 15]), relay.reshape(const_337.astype('float32'), [14, 15]), relay.reshape(const_337.astype('float32'), [14, 15]), ), 6)
call_338 = relay.TupleGetItem(func_154_call(relay.reshape(const_337.astype('float32'), [14, 15]), relay.reshape(const_337.astype('int8'), [14, 15]), relay.reshape(const_337.astype('float32'), [14, 15]), relay.reshape(const_337.astype('float32'), [14, 15]), ), 6)
func_277_call = mod.get_global_var('func_277')
func_280_call = mutated_mod.get_global_var('func_280')
call_340 = relay.TupleGetItem(func_277_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(call_311.astype('uint64'), [15, 15]), ), 1)
call_341 = relay.TupleGetItem(func_280_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(call_311.astype('uint64'), [15, 15]), ), 1)
func_277_call = mod.get_global_var('func_277')
func_280_call = mutated_mod.get_global_var('func_280')
call_349 = relay.TupleGetItem(func_277_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(const_312.astype('uint64'), [15, 15]), ), 0)
call_350 = relay.TupleGetItem(func_280_call(relay.reshape(const_312.astype('uint64'), [15, 15]), relay.reshape(const_312.astype('uint64'), [15, 15]), ), 0)
bop_352 = relay.power(bop_329.astype('float32'), relay.reshape(bop_320.astype('float32'), relay.shape_of(bop_329))) # shape=(5, 8)
output = relay.Tuple([bop_308,call_311,const_312,uop_317,call_323,const_324,call_334,call_336,const_337,call_340,call_349,bop_352,])
output2 = relay.Tuple([bop_308,call_313,const_312,uop_317,call_325,const_324,call_335,call_338,const_337,call_341,call_350,bop_352,])
func_355 = relay.Function([], output)
mod['func_355'] = func_355
mod = relay.transform.InferType()(mod)
mutated_mod['func_355'] = func_355
mutated_mod = relay.transform.InferType()(mutated_mod)
func_355_call = mutated_mod.get_global_var('func_355')
call_356 = func_355_call()
output = call_356
func_357 = relay.Function([], output)
mutated_mod['func_357'] = func_357
mutated_mod = relay.transform.InferType()(mutated_mod)
const_358 = relay.const([[1.342056,9.681692,9.689271,0.344582,-3.681257,5.272375,9.314637],[5.113909,-0.237008,-1.809165,-6.930339,-0.745122,8.730091,-0.708982],[-1.102081,-5.758809,-6.351068,7.271444,-3.832861,-7.806395,-0.782199],[6.255025,7.842016,6.713914,-3.964967,0.736669,5.879007,-8.434091],[1.836399,3.914974,3.525464,-9.742526,0.038126,3.650363,-8.043789],[9.796482,0.117273,-7.911214,2.226825,-9.358779,1.725807,-8.120435],[6.067862,2.519383,-6.347587,-6.517474,-4.699546,5.513100,-4.390832]], dtype = "float32")#candidate|358|(7, 7)|const|float32
uop_359 = relay.rsqrt(const_358.astype('float32')) # shape=(7, 7)
bop_361 = relay.minimum(uop_359.astype('int8'), relay.reshape(const_358.astype('int8'), relay.shape_of(uop_359))) # shape=(7, 7)
uop_365 = relay.cos(uop_359.astype('float64')) # shape=(7, 7)
bop_367 = relay.power(uop_365.astype('float32'), relay.reshape(uop_359.astype('float32'), relay.shape_of(uop_365))) # shape=(7, 7)
uop_374 = relay.asin(uop_365.astype('float64')) # shape=(7, 7)
var_376 = relay.var("var_376", dtype = "int8", shape = (7, 7))#candidate|376|(7, 7)|var|int8
bop_377 = relay.bitwise_or(bop_361.astype('int8'), relay.reshape(var_376.astype('int8'), relay.shape_of(bop_361))) # shape=(7, 7)
bop_383 = relay.left_shift(uop_374.astype('uint64'), relay.reshape(bop_377.astype('uint64'), relay.shape_of(uop_374))) # shape=(7, 7)
output = relay.Tuple([bop_367,bop_383,])
output2 = relay.Tuple([bop_367,bop_383,])
func_387 = relay.Function([var_376,], output)
mod['func_387'] = func_387
mod = relay.transform.InferType()(mod)
var_388 = relay.var("var_388", dtype = "int8", shape = (7, 7))#candidate|388|(7, 7)|var|int8
output = func_387(var_388)
func_389 = relay.Function([var_388], output)
mutated_mod['func_389'] = func_389
mutated_mod = relay.transform.InferType()(mutated_mod)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_410 = func_205_call()
call_411 = func_205_call()
func_277_call = mod.get_global_var('func_277')
func_280_call = mutated_mod.get_global_var('func_280')
const_417 = relay.const([10,4,-1,4,-4,5,-10,8,-9,1,2,1,-4,-4,1,-10,5,7,-6,10,-3,-3,-3,10,-10,5,8,2,8,4,-7,10,-8,-10,9,4,4,-4,8,6,-6,4,9,-5,1,-6,-6,2,5,-10,-1,-4,-4,1,-4,-4,10,-4,3,7,1,-4,-10,-4,-8,7,1,-6,2,8,4,6,-9,6,2,-8,5,-3,-6,-9,-1,4,10,8,-5,9,10,7,8,3,-7,4,-1,-6,-3,-9,-3,-6,-3,6,3,-1,-1,-4,2,-10,6,-10,10,-4,-2,7,8,-10,-10,-10,2,-9,10,1,-1,-2,6,6,9,-10,6,6,-3,-4,-3,-9,6,7,1,9,-10,9,-10,-1,-6,9,8,-8,4,-5,8,1,2,-9,7,-3,8,10,-8,8,9,9,-5,-8,9,-9,9,-3,9,4,-2,10,10,8,10,-5,3,6,-1,10,-10,-8,-8,-10,3,-10,2,4,9,-1,6,-8,1,9,-10,3,6,1,-2,-9,-6,9,9,9,3,-7,-9,10,-10,-2,-7,-2,-6,-9,5,-1,-6,-8,-4,-7,8,-10,10,7,-10,-10,-3,2,-5], dtype = "uint64")#candidate|417|(225,)|const|uint64
call_416 = relay.TupleGetItem(func_277_call(relay.reshape(const_417.astype('uint64'), [15, 15]), relay.reshape(const_417.astype('uint64'), [15, 15]), ), 1)
call_418 = relay.TupleGetItem(func_280_call(relay.reshape(const_417.astype('uint64'), [15, 15]), relay.reshape(const_417.astype('uint64'), [15, 15]), ), 1)
uop_423 = relay.atan(const_417.astype('float32')) # shape=(225,)
func_149_call = mod.get_global_var('func_149')
func_154_call = mutated_mod.get_global_var('func_154')
var_427 = relay.var("var_427", dtype = "float32", shape = (210,))#candidate|427|(210,)|var|float32
call_426 = relay.TupleGetItem(func_149_call(relay.reshape(var_427.astype('float32'), [14, 15]), relay.reshape(var_427.astype('int8'), [14, 15]), relay.reshape(var_427.astype('float32'), [14, 15]), relay.reshape(var_427.astype('float32'), [14, 15]), ), 5)
call_428 = relay.TupleGetItem(func_154_call(relay.reshape(var_427.astype('float32'), [14, 15]), relay.reshape(var_427.astype('int8'), [14, 15]), relay.reshape(var_427.astype('float32'), [14, 15]), relay.reshape(var_427.astype('float32'), [14, 15]), ), 5)
uop_429 = relay.tan(uop_423.astype('float32')) # shape=(225,)
bop_431 = relay.mod(uop_429.astype('float32'), relay.reshape(const_417.astype('float32'), relay.shape_of(uop_429))) # shape=(225,)
uop_434 = relay.erf(bop_431.astype('float32')) # shape=(225,)
bop_436 = relay.bitwise_or(uop_434.astype('int64'), relay.reshape(uop_429.astype('int64'), relay.shape_of(uop_434))) # shape=(225,)
bop_443 = relay.floor_mod(uop_434.astype('float32'), relay.reshape(uop_429.astype('float32'), relay.shape_of(uop_434))) # shape=(225,)
bop_447 = relay.left_shift(var_427.astype('int64'), relay.reshape(call_426.astype('int64'), relay.shape_of(var_427))) # shape=(210,)
bop_450 = relay.left_shift(var_427.astype('int64'), relay.reshape(call_428.astype('int64'), relay.shape_of(var_427))) # shape=(210,)
bop_451 = relay.minimum(bop_436.astype('uint8'), relay.reshape(uop_423.astype('uint8'), relay.shape_of(bop_436))) # shape=(225,)
bop_454 = relay.divide(bop_431.astype('float64'), relay.reshape(uop_429.astype('float64'), relay.shape_of(bop_431))) # shape=(225,)
var_457 = relay.var("var_457", dtype = "float32", shape = (225,))#candidate|457|(225,)|var|float32
bop_458 = relay.power(uop_429.astype('float32'), relay.reshape(var_457.astype('float32'), relay.shape_of(uop_429))) # shape=(225,)
bop_461 = relay.not_equal(bop_451.astype('bool'), relay.reshape(uop_423.astype('bool'), relay.shape_of(bop_451))) # shape=(225,)
output = relay.Tuple([call_410,call_416,bop_443,bop_447,bop_454,bop_458,bop_461,])
output2 = relay.Tuple([call_411,call_418,bop_443,bop_450,bop_454,bop_458,bop_461,])
func_464 = relay.Function([var_427,var_457,], output)
mod['func_464'] = func_464
mod = relay.transform.InferType()(mod)
var_465 = relay.var("var_465", dtype = "float32", shape = (210,))#candidate|465|(210,)|var|float32
var_466 = relay.var("var_466", dtype = "float32", shape = (225,))#candidate|466|(225,)|var|float32
output = func_464(var_465,var_466,)
func_467 = relay.Function([var_465,var_466,], output)
mutated_mod['func_467'] = func_467
mutated_mod = relay.transform.InferType()(mutated_mod)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_477 = func_205_call()
call_478 = func_205_call()
output = call_477
output2 = call_478
func_480 = relay.Function([], output)
mod['func_480'] = func_480
mod = relay.transform.InferType()(mod)
mutated_mod['func_480'] = func_480
mutated_mod = relay.transform.InferType()(mutated_mod)
func_480_call = mutated_mod.get_global_var('func_480')
call_481 = func_480_call()
output = call_481
func_482 = relay.Function([], output)
mutated_mod['func_482'] = func_482
mutated_mod = relay.transform.InferType()(mutated_mod)
var_507 = relay.var("var_507", dtype = "int64", shape = (12, 4))#candidate|507|(12, 4)|var|int64
const_508 = relay.const([[-9,-3,10,-7],[7,-6,8,5],[8,3,1,2],[-9,-3,-2,-5],[3,-3,9,4],[6,5,9,1],[-3,1,-6,-5],[4,4,-4,-8],[-6,10,10,1],[3,-3,-9,8],[-7,-3,5,-7],[-8,4,1,9]], dtype = "int64")#candidate|508|(12, 4)|const|int64
bop_509 = relay.bitwise_or(var_507.astype('int64'), relay.reshape(const_508.astype('int64'), relay.shape_of(var_507))) # shape=(12, 4)
func_355_call = mod.get_global_var('func_355')
func_357_call = mutated_mod.get_global_var('func_357')
call_513 = relay.TupleGetItem(func_355_call(), 8)
call_514 = relay.TupleGetItem(func_357_call(), 8)
bop_518 = relay.floor_divide(const_508.astype('float64'), relay.reshape(var_507.astype('float64'), relay.shape_of(const_508))) # shape=(12, 4)
bop_521 = relay.logical_or(var_507.astype('bool'), relay.reshape(bop_518.astype('bool'), relay.shape_of(var_507))) # shape=(12, 4)
uop_528 = relay.asin(bop_518.astype('float32')) # shape=(12, 4)
bop_532 = relay.divide(uop_528.astype('float64'), relay.reshape(bop_509.astype('float64'), relay.shape_of(uop_528))) # shape=(12, 4)
uop_540 = relay.cosh(bop_532.astype('float64')) # shape=(12, 4)
uop_543 = relay.cos(uop_540.astype('float32')) # shape=(12, 4)
func_277_call = mod.get_global_var('func_277')
func_280_call = mutated_mod.get_global_var('func_280')
var_546 = relay.var("var_546", dtype = "uint64", shape = (225,))#candidate|546|(225,)|var|uint64
call_545 = relay.TupleGetItem(func_277_call(relay.reshape(var_546.astype('uint64'), [15, 15]), relay.reshape(var_546.astype('uint64'), [15, 15]), ), 0)
call_547 = relay.TupleGetItem(func_280_call(relay.reshape(var_546.astype('uint64'), [15, 15]), relay.reshape(var_546.astype('uint64'), [15, 15]), ), 0)
output = relay.Tuple([call_513,bop_521,uop_543,call_545,var_546,])
output2 = relay.Tuple([call_514,bop_521,uop_543,call_547,var_546,])
func_548 = relay.Function([var_507,var_546,], output)
mod['func_548'] = func_548
mod = relay.transform.InferType()(mod)
var_549 = relay.var("var_549", dtype = "int64", shape = (12, 4))#candidate|549|(12, 4)|var|int64
var_550 = relay.var("var_550", dtype = "uint64", shape = (225,))#candidate|550|(225,)|var|uint64
output = func_548(var_549,var_550,)
func_551 = relay.Function([var_549,var_550,], output)
mutated_mod['func_551'] = func_551
mutated_mod = relay.transform.InferType()(mutated_mod)
const_577 = relay.const([[[True,False,True,False,True,True,True],[False,True,False,False,True,False,True],[False,True,False,False,False,True,False],[True,False,True,True,True,True,True],[False,True,True,True,True,True,True],[True,False,True,True,True,True,False],[True,True,True,True,False,True,False],[True,False,False,False,True,False,False],[False,True,True,False,False,True,False]],[[False,False,True,True,False,False,False],[True,True,False,False,False,False,False],[True,True,False,False,False,False,False],[True,False,False,True,True,False,False],[False,True,False,False,True,True,True],[True,False,True,False,True,False,True],[False,False,True,True,True,True,True],[True,True,True,False,False,False,False],[True,False,False,True,True,False,True]],[[False,True,False,True,True,False,True],[True,False,False,True,False,True,False],[True,False,True,True,False,False,True],[False,False,True,True,False,True,True],[True,True,False,True,True,True,False],[False,True,False,True,True,False,False],[True,False,False,False,False,True,True],[False,False,True,False,False,False,False],[True,False,True,False,True,True,False]],[[False,False,False,True,True,False,False],[False,True,True,True,True,True,False],[True,True,True,False,True,False,False],[True,True,False,False,True,True,True],[False,True,True,True,True,True,False],[True,False,False,True,True,True,True],[False,True,True,True,False,False,True],[True,True,False,False,False,False,True],[True,False,False,True,True,True,False]],[[False,False,False,True,False,True,True],[True,False,False,True,True,True,True],[True,True,False,False,False,False,True],[False,True,False,True,True,False,True],[False,True,True,False,True,False,False],[False,True,True,True,True,True,True],[True,False,False,False,True,False,True],[False,True,False,False,False,False,False],[True,False,True,True,True,False,True]]], dtype = "bool")#candidate|577|(5, 9, 7)|const|bool
const_578 = relay.const([[[False,True,False,True,False,False,True],[True,True,False,True,True,False,True],[True,True,True,False,True,True,False],[True,True,True,True,False,False,False],[True,False,True,True,True,True,True],[False,True,False,True,False,False,True],[True,False,False,True,False,True,True],[True,False,False,True,False,True,False],[False,False,True,True,False,False,True]],[[True,True,False,True,True,False,True],[True,False,False,True,False,False,False],[False,True,True,False,True,True,True],[True,True,True,False,True,False,True],[True,False,False,False,False,False,False],[True,False,True,False,False,True,True],[True,True,True,True,False,True,True],[True,False,False,True,True,False,True],[False,False,False,True,True,True,False]],[[True,True,False,False,False,True,False],[False,False,False,True,True,True,True],[True,False,False,False,True,True,True],[False,True,False,True,True,True,False],[False,False,False,True,True,False,True],[True,True,False,False,True,False,True],[False,True,False,True,False,True,False],[True,False,True,True,True,False,False],[False,True,True,True,True,True,False]],[[False,False,False,False,True,False,False],[True,False,True,True,True,False,True],[True,True,True,False,True,False,False],[False,False,False,False,True,True,True],[False,True,False,True,True,False,True],[False,False,True,True,False,False,True],[True,False,False,True,True,True,True],[False,True,True,False,False,False,True],[True,True,False,False,False,False,True]],[[True,False,False,False,False,True,True],[False,True,False,False,False,False,True],[True,True,False,False,True,False,False],[True,False,True,False,True,False,False],[False,False,True,True,True,True,True],[True,False,True,False,False,True,False],[False,False,True,True,True,True,True],[False,True,False,True,True,True,False],[True,True,False,True,False,False,False]]], dtype = "bool")#candidate|578|(5, 9, 7)|const|bool
bop_579 = relay.logical_and(const_577.astype('bool'), relay.reshape(const_578.astype('bool'), relay.shape_of(const_577))) # shape=(5, 9, 7)
bop_582 = relay.less(const_577.astype('bool'), relay.reshape(const_578.astype('bool'), relay.shape_of(const_577))) # shape=(5, 9, 7)
func_480_call = mod.get_global_var('func_480')
func_482_call = mutated_mod.get_global_var('func_482')
call_588 = func_480_call()
call_589 = func_480_call()
bop_591 = relay.logical_xor(bop_579.astype('int8'), relay.reshape(bop_582.astype('int8'), relay.shape_of(bop_579))) # shape=(5, 9, 7)
bop_597 = relay.bitwise_xor(bop_582.astype('int32'), relay.reshape(bop_579.astype('int32'), relay.shape_of(bop_582))) # shape=(5, 9, 7)
uop_600 = relay.cos(const_578.astype('float32')) # shape=(5, 9, 7)
bop_602 = relay.floor_mod(uop_600.astype('float32'), relay.reshape(bop_597.astype('float32'), relay.shape_of(uop_600))) # shape=(5, 9, 7)
bop_606 = relay.bitwise_or(bop_602.astype('uint8'), relay.reshape(bop_582.astype('uint8'), relay.shape_of(bop_602))) # shape=(5, 9, 7)
bop_609 = relay.minimum(const_577.astype('int16'), relay.reshape(bop_579.astype('int16'), relay.shape_of(const_577))) # shape=(5, 9, 7)
var_612 = relay.var("var_612", dtype = "bool", shape = (5, 9, 7))#candidate|612|(5, 9, 7)|var|bool
bop_613 = relay.floor_divide(bop_579.astype('float64'), relay.reshape(var_612.astype('float64'), relay.shape_of(bop_579))) # shape=(5, 9, 7)
output = relay.Tuple([call_588,bop_591,bop_606,bop_609,bop_613,])
output2 = relay.Tuple([call_589,bop_591,bop_606,bop_609,bop_613,])
func_617 = relay.Function([var_612,], output)
mod['func_617'] = func_617
mod = relay.transform.InferType()(mod)
mutated_mod['func_617'] = func_617
mutated_mod = relay.transform.InferType()(mutated_mod)
var_618 = relay.var("var_618", dtype = "bool", shape = (5, 9, 7))#candidate|618|(5, 9, 7)|var|bool
func_617_call = mutated_mod.get_global_var('func_617')
call_619 = func_617_call(var_618)
output = call_619
func_620 = relay.Function([var_618], output)
mutated_mod['func_620'] = func_620
mutated_mod = relay.transform.InferType()(mutated_mod)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_622 = func_205_call()
call_623 = func_205_call()
const_655 = relay.const([[[4.881171,1.755281],[-4.752070,-6.373776],[-6.743134,8.918720]],[[9.130263,-7.536225],[3.222453,-5.529118],[-9.955759,8.216920]],[[9.384796,0.166978],[-1.782747,-2.324036],[-9.693377,0.058185]]], dtype = "float64")#candidate|655|(3, 3, 2)|const|float64
bop_656 = relay.equal(call_622.astype('bool'), relay.reshape(const_655.astype('bool'), relay.shape_of(call_622))) # shape=(3, 3, 2)
bop_659 = relay.equal(call_623.astype('bool'), relay.reshape(const_655.astype('bool'), relay.shape_of(call_623))) # shape=(3, 3, 2)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
const_663 = relay.const([[-2.924683,-6.936366],[-3.098845,5.699013],[-5.666380,2.016978],[-6.526595,3.372895],[6.724693,5.605090],[-5.888793,-4.009617],[-3.050258,-5.931819],[-6.679006,-8.290500],[-5.121483,2.639729],[-4.509922,3.739751],[1.531908,-0.182367],[0.773741,-9.279039],[5.373150,5.075033],[4.735370,2.079155],[8.618589,-5.135152],[-4.281124,8.114834],[6.574068,0.522523],[9.063261,9.559549],[2.932319,2.850461],[6.760391,-2.490630],[2.070763,-8.608280],[4.895991,-3.526700],[-7.231505,-8.625158],[4.275450,3.649762],[1.851238,-3.617155],[2.918939,2.295855],[-8.518347,5.003180],[-3.634699,-6.284694],[8.062775,-8.793928],[-4.373302,8.231760],[-6.244100,0.618787],[4.569685,-0.291614],[7.920564,-4.038011],[-4.247900,-8.460584],[-5.422007,-2.937335],[4.672905,-6.021987],[-8.672285,2.675075],[1.894397,5.032721],[1.716853,-3.683575],[1.054492,2.359537],[0.079488,7.493567],[1.750389,-3.275063],[9.032216,-8.370048],[-3.010269,4.549176],[7.507599,6.212913],[1.277483,-7.904141],[-7.694125,-2.470071],[-4.813240,-8.974205],[9.903161,8.225579],[3.537364,-0.692468],[3.612910,9.874935],[0.179499,-9.551588],[-1.913011,6.501705],[-6.138485,-8.114129],[-7.597671,-2.663981],[-0.304410,8.549059],[-7.559383,-4.106294],[-3.790857,-6.703818],[3.527844,3.311493],[-6.045194,8.034658],[-4.009363,-4.791909],[8.451320,8.002730],[7.282149,-8.900047],[-9.898906,0.354376],[-6.019439,1.029380],[-2.170649,-5.364891],[2.474558,-8.704820],[0.195514,-7.367536],[5.888516,-0.979884],[5.326842,-3.078070],[2.677334,-2.675597],[-9.680397,-8.625432],[6.753160,-9.686640],[2.296859,9.233636],[-1.286786,-6.094119],[-1.888540,9.388529],[-1.334772,-7.039908],[1.728829,-0.645051],[8.199807,0.792817],[3.613775,-5.345571],[9.120730,-2.392322],[-4.523842,5.453250],[-0.386858,-4.792005],[7.431797,-7.642886],[-3.094563,3.100917],[2.051671,-3.761425],[2.831357,8.435986],[-1.978967,2.092298],[-6.880300,-7.340536],[-5.740218,2.382100],[-2.545113,1.460034],[-9.545838,9.154337],[-9.949216,-5.690005],[-5.827915,-0.231006],[-9.594881,-8.286671],[8.001312,-8.119904],[-0.440049,-1.496312],[0.946567,1.045569],[-2.224755,9.288481],[-1.056525,1.221427],[3.356674,-7.401883],[-3.398155,6.306343],[-3.802527,6.168565],[-7.231258,5.127263],[8.254059,0.612776],[-6.562833,-9.727025],[8.888072,1.404155],[-5.270386,-9.123846],[-8.590861,9.015077],[7.816581,-2.912553],[7.730696,4.403112],[1.475737,-1.495683],[7.845234,0.584835],[-6.160139,-5.044309],[2.964082,3.879214],[-4.845050,-4.487909],[-7.785507,1.135318],[-4.065613,-1.818104],[-6.634787,7.926423],[-6.618478,-2.476317],[5.920285,3.342528],[6.044983,9.540556],[6.677073,-9.547955],[-5.213226,9.196939],[-1.745532,-9.502280],[-7.905792,-6.654433],[1.844100,-5.319761],[3.046800,9.202350],[2.949729,-5.295586],[-3.524971,2.698467],[-3.515438,4.795566],[-8.060274,-6.385816],[5.828695,-8.548442],[7.344022,-9.436977],[-1.416324,-1.861492],[2.871432,9.291511],[9.298419,7.442981],[9.227996,4.947797],[-9.501600,3.784894],[9.775696,-4.974998],[-5.901348,-1.790686],[1.736488,-9.783820],[-5.990435,-7.398826],[-8.203492,-7.591712],[5.458588,2.403025],[-9.719593,1.198717],[7.771194,-4.083783],[-1.330406,-4.123582],[4.654884,-3.674792],[3.585230,-7.903620],[8.514676,-8.974274],[-9.771868,3.579823],[-5.012293,1.230614],[-0.005557,-2.417482],[8.027087,-7.956926],[7.795119,-8.330395],[-7.488048,6.511690],[-0.336214,7.597554],[-7.344700,5.859702],[-6.043959,1.506873],[-8.457237,7.333999],[4.316226,-8.904146],[-4.340861,-3.158972],[5.462352,-8.024989],[6.159268,-3.176891],[-9.970099,8.232990],[1.509235,1.984770],[-5.191773,-8.323367],[7.990168,4.597075],[1.595252,3.290011],[-4.091658,6.171079],[1.175088,4.219256],[-1.019345,-3.787048],[5.845883,3.015787],[-0.455801,-3.889215],[4.088017,0.976012],[5.662997,3.438949],[7.142539,-6.146280],[-2.235659,0.448683],[-6.501751,-6.049247],[-9.167109,5.141513],[0.621578,-1.459856],[3.569567,-8.885687],[-3.217361,-2.927223],[-4.200846,-4.487225],[-2.700149,3.902234],[7.441081,-9.278912],[-9.243048,-0.797714],[8.017717,-8.698430],[-4.926573,9.086085],[-2.548700,9.058512],[6.740930,-7.171006],[3.222018,6.734995],[8.921905,7.081788],[-6.274921,-6.862380],[-6.824327,0.821812],[-7.490084,-6.297137],[8.356571,0.224475],[6.775604,-9.101312],[-6.176086,-1.912600],[3.105437,2.802596],[3.415280,4.525300],[8.534074,-8.343502],[-0.982962,0.628593],[7.816097,-2.113939],[1.589324,-3.397373],[-0.255806,-0.110766],[7.007351,-9.286324],[-2.978098,-4.989699],[-4.885550,7.265176],[1.310119,-1.346881],[-4.526768,1.117018],[6.414786,-2.741406],[0.283834,-6.039364],[9.108133,-2.573910],[5.581080,0.192815],[-0.204379,4.048748],[-4.361594,-3.645549],[-2.775806,5.450297],[4.812887,2.039111],[-5.105295,2.525835],[-6.585690,2.903512],[-0.231219,4.517399],[1.133618,-2.326548],[-1.606340,-6.784602],[-9.220704,-9.425999],[5.795007,1.621993],[-7.690469,-8.902929],[-2.235737,-8.014453],[8.151349,-7.948040],[9.721228,7.607173],[2.255050,9.468995],[-2.831549,-3.139914],[-5.006300,0.876226],[-8.992769,-1.963894],[4.787497,-8.147264],[4.124120,3.898759],[-7.460770,-2.960784],[1.990240,-1.025829],[6.322293,-0.807068],[-8.391727,6.290499],[2.923860,-9.815562],[3.058099,5.481981],[4.571966,7.766594],[-8.655632,3.046758],[-5.255905,4.011430],[-5.598054,-0.699862],[8.709359,4.085513],[8.953732,-5.477683],[2.463809,7.911957],[-3.940637,-5.365174],[-1.913759,-9.873153]], dtype = "float64")#candidate|663|(252, 2)|const|float64
call_662 = relay.TupleGetItem(func_237_call(relay.reshape(const_663.astype('float64'), [6, 6, 14])), 1)
call_664 = relay.TupleGetItem(func_239_call(relay.reshape(const_663.astype('float64'), [6, 6, 14])), 1)
func_355_call = mod.get_global_var('func_355')
func_357_call = mutated_mod.get_global_var('func_357')
call_665 = relay.TupleGetItem(func_355_call(), 10)
call_666 = relay.TupleGetItem(func_357_call(), 10)
output = relay.Tuple([bop_656,call_662,const_663,call_665,])
output2 = relay.Tuple([bop_659,call_664,const_663,call_666,])
func_669 = relay.Function([], output)
mod['func_669'] = func_669
mod = relay.transform.InferType()(mod)
output = func_669()
func_670 = relay.Function([], output)
mutated_mod['func_670'] = func_670
mutated_mod = relay.transform.InferType()(mutated_mod)
func_669_call = mod.get_global_var('func_669')
func_670_call = mutated_mod.get_global_var('func_670')
call_673 = relay.TupleGetItem(func_669_call(), 2)
call_674 = relay.TupleGetItem(func_670_call(), 2)
func_617_call = mod.get_global_var('func_617')
func_620_call = mutated_mod.get_global_var('func_620')
const_677 = relay.const([False,False,True,True,False,True,False,True,True,False,False,False,False,True,True,True,False,True,True,True,True,False,False,False,False,True,True,False,True,False,True,False,False,False,True,False,False,True,True,True,True,True,True,True,True,True,False,True,False,False,True,False,False,True,False,False,False,False,False,False,False,True,False,False,False,True,False,False,False,False,True,True,True,True,True,False,False,True,False,False,True,True,False,True,False,True,True,True,True,False,True,True,True,True,True,True,False,True,True,False,True,True,True,True,False,False,True,False,False,True,False,True,False,True,True,True,False,False,False,True,False,True,False,False,False,True,True,False,False,True,False,False,False,True,True,False,False,False,False,False,False,False,True,False,True,False,True,True,True,True,False,True,False,False,True,False,True,True,False,False,False,False,False,False,False,True,False,False,True,False,False,True,False,True,False,False,False,True,True,True,True,False,False,True,False,False,False,False,True,False,False,True,True,False,True,True,True,True,True,False,True,True,False,True,False,False,True,False,True,True,True,False,True,False,False,True,False,False,True,True,False,True,False,True,True,True,False,False,False,True,True,True,False,True,False,True,True,False,True,False,True,True,True,False,True,True,True,True,True,True,False,True,False,False,True,False,True,True,False,True,True,True,False,True,False,False,False,True,False,False,False,True,True,True,True,False,False,True,True,True,False,True,False,False,True,True,False,False,True,True,True,False,False,True,True,True,True,False,False,False,False,False,True,True,True,True,True,True,False,False,False,False,True,True,False], dtype = "bool")#candidate|677|(315,)|const|bool
call_676 = relay.TupleGetItem(func_617_call(relay.reshape(const_677.astype('bool'), [5, 9, 7])), 1)
call_678 = relay.TupleGetItem(func_620_call(relay.reshape(const_677.astype('bool'), [5, 9, 7])), 1)
bop_697 = relay.bitwise_and(call_676.astype('uint8'), relay.reshape(const_677.astype('uint8'), relay.shape_of(call_676))) # shape=(5, 9, 7)
bop_700 = relay.bitwise_and(call_678.astype('uint8'), relay.reshape(const_677.astype('uint8'), relay.shape_of(call_678))) # shape=(5, 9, 7)
uop_706 = relay.exp(bop_697.astype('float64')) # shape=(5, 9, 7)
uop_708 = relay.exp(bop_700.astype('float64')) # shape=(5, 9, 7)
uop_711 = relay.sinh(uop_706.astype('float32')) # shape=(5, 9, 7)
uop_713 = relay.sinh(uop_708.astype('float32')) # shape=(5, 9, 7)
uop_719 = relay.log10(uop_706.astype('float64')) # shape=(5, 9, 7)
uop_721 = relay.log10(uop_708.astype('float64')) # shape=(5, 9, 7)
output = relay.Tuple([call_673,uop_711,uop_719,])
output2 = relay.Tuple([call_674,uop_713,uop_721,])
func_727 = relay.Function([], output)
mod['func_727'] = func_727
mod = relay.transform.InferType()(mod)
output = func_727()
func_728 = relay.Function([], output)
mutated_mod['func_728'] = func_728
mutated_mod = relay.transform.InferType()(mutated_mod)
var_742 = relay.var("var_742", dtype = "float32", shape = (11, 6))#candidate|742|(11, 6)|var|float32
uop_743 = relay.log2(var_742.astype('float32')) # shape=(11, 6)
func_166_call = mod.get_global_var('func_166')
func_169_call = mutated_mod.get_global_var('func_169')
var_747 = relay.var("var_747", dtype = "int64", shape = (45, 3))#candidate|747|(45, 3)|var|int64
call_746 = func_166_call(relay.reshape(var_747.astype('int64'), [15, 9]), relay.reshape(var_747.astype('int64'), [15, 9]), )
call_748 = func_166_call(relay.reshape(var_747.astype('int64'), [15, 9]), relay.reshape(var_747.astype('int64'), [15, 9]), )
uop_749 = relay.acos(uop_743.astype('float64')) # shape=(11, 6)
var_751 = relay.var("var_751", dtype = "float32", shape = (11, 6))#candidate|751|(11, 6)|var|float32
bop_752 = relay.maximum(uop_743.astype('uint16'), relay.reshape(var_751.astype('uint16'), relay.shape_of(uop_743))) # shape=(11, 6)
output = relay.Tuple([call_746,var_747,uop_749,bop_752,])
output2 = relay.Tuple([call_748,var_747,uop_749,bop_752,])
func_756 = relay.Function([var_742,var_747,var_751,], output)
mod['func_756'] = func_756
mod = relay.transform.InferType()(mod)
mutated_mod['func_756'] = func_756
mutated_mod = relay.transform.InferType()(mutated_mod)
func_756_call = mutated_mod.get_global_var('func_756')
var_758 = relay.var("var_758", dtype = "float32", shape = (11, 6))#candidate|758|(11, 6)|var|float32
var_759 = relay.var("var_759", dtype = "int64", shape = (45, 3))#candidate|759|(45, 3)|var|int64
var_760 = relay.var("var_760", dtype = "float32", shape = (11, 6))#candidate|760|(11, 6)|var|float32
call_757 = func_756_call(var_758,var_759,var_760,)
output = call_757
func_761 = relay.Function([var_758,var_759,var_760,], output)
mutated_mod['func_761'] = func_761
mutated_mod = relay.transform.InferType()(mutated_mod)
func_205_call = mod.get_global_var('func_205')
func_206_call = mutated_mod.get_global_var('func_206')
call_787 = func_205_call()
call_788 = func_205_call()
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
var_793 = relay.var("var_793", dtype = "float64", shape = (504,))#candidate|793|(504,)|var|float64
call_792 = relay.TupleGetItem(func_237_call(relay.reshape(var_793.astype('float64'), [6, 6, 14])), 1)
call_794 = relay.TupleGetItem(func_239_call(relay.reshape(var_793.astype('float64'), [6, 6, 14])), 1)
output = relay.Tuple([call_787,call_792,var_793,])
output2 = relay.Tuple([call_788,call_794,var_793,])
func_796 = relay.Function([var_793,], output)
mod['func_796'] = func_796
mod = relay.transform.InferType()(mod)
var_797 = relay.var("var_797", dtype = "float64", shape = (504,))#candidate|797|(504,)|var|float64
output = func_796(var_797)
func_798 = relay.Function([var_797], output)
mutated_mod['func_798'] = func_798
mutated_mod = relay.transform.InferType()(mutated_mod)
func_480_call = mod.get_global_var('func_480')
func_482_call = mutated_mod.get_global_var('func_482')
call_830 = func_480_call()
call_831 = func_480_call()
uop_836 = relay.cosh(call_830.astype('float32')) # shape=(3, 3, 2)
uop_838 = relay.cosh(call_831.astype('float32')) # shape=(3, 3, 2)
output = uop_836
output2 = uop_838
func_839 = relay.Function([], output)
mod['func_839'] = func_839
mod = relay.transform.InferType()(mod)
output = func_839()
func_840 = relay.Function([], output)
mutated_mod['func_840'] = func_840
mutated_mod = relay.transform.InferType()(mutated_mod)
var_852 = relay.var("var_852", dtype = "uint8", shape = (4,))#candidate|852|(4,)|var|uint8
var_853 = relay.var("var_853", dtype = "uint8", shape = (4,))#candidate|853|(4,)|var|uint8
bop_854 = relay.right_shift(var_852.astype('uint8'), relay.reshape(var_853.astype('uint8'), relay.shape_of(var_852))) # shape=(4,)
func_166_call = mod.get_global_var('func_166')
func_169_call = mutated_mod.get_global_var('func_169')
const_858 = relay.const([-3,8,-10,-9,8,7,3,-5,-5,-5,-10,-4,10,-5,8,-4,8,-9,4,-1,7,-2,10,-5,8,-3,-6,-5,3,-3,-5,-2,-6,-1,-8,3,-10,10,-4,-4,4,1,6,-6,-7,10,3,4,6,10,-8,1,-7,8,3,-4,-1,7,-10,-6,2,-8,1,-4,7,-1,8,1,-3,9,-4,5,7,-3,-6,3,-9,3,-10,9,1,-2,-3,6,-3,4,-9,3,2,2,-8,-4,-5,9,3,5,-7,7,-10,-6,10,-10,-8,5,-2,-7,-4,-7,9,-8,-5,8,-10,2,8,2,-7,-8,-9,4,-2,10,-4,-8,-7,7,9,-2,10,6,-1,7,3,2,8], dtype = "int64")#candidate|858|(135,)|const|int64
call_857 = func_166_call(relay.reshape(const_858.astype('int64'), [15, 9]), relay.reshape(const_858.astype('int64'), [15, 9]), )
call_859 = func_166_call(relay.reshape(const_858.astype('int64'), [15, 9]), relay.reshape(const_858.astype('int64'), [15, 9]), )
func_295_call = mod.get_global_var('func_295')
func_297_call = mutated_mod.get_global_var('func_297')
call_860 = func_295_call()
call_861 = func_295_call()
bop_874 = relay.floor_divide(bop_854.astype('float64'), relay.reshape(var_852.astype('float64'), relay.shape_of(bop_854))) # shape=(4,)
func_617_call = mod.get_global_var('func_617')
func_620_call = mutated_mod.get_global_var('func_620')
const_878 = relay.const([False,True,True,False,False,False,True,True,False,True,False,True,True,True,False,True,False,False,True,True,False,True,True,False,True,False,True,False,False,True,False,False,False,False,True,True,False,False,False,False,True,False,False,True,False,False,False,False,True,True,True,True,False,True,True,True,True,True,True,True,False,False,False,True,False,True,False,False,True,False,False,True,True,False,False,True,True,False,True,False,False,True,True,False,False,True,False,True,False,True,True,False,True,True,True,True,False,True,True,True,False,False,False,True,False,False,False,True,True,True,True,True,False,True,True,False,False,True,False,False,True,True,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,False,True,False,False,False,False,True,False,True,True,False,False,False,False,True,False,False,True,False,False,False,True,True,False,True,False,False,False,False,True,True,False,False,False,True,False,False,False,False,True,True,False,False,True,True,True,True,True,False,True,True,False,False,True,False,False,True,False,False,True,False,True,False,False,True,True,True,True,True,True,True,False,False,True,True,True,False,False,False,False,True,False,False,True,True,True,True,False,True,True,True,True,True,True,False,False,False,True,False,True,False,True,True,False,False,False,True,False,False,False,False,True,False,False,True,True,True,False,True,True,False,False,False,True,False,False,True,False,True,True,False,False,False,True,False,False,False,False,False,False,False,False,False,False,True,True,True,False,True,False,True,True,False,False,False,False,False,False,True,False,True,True,False,False,False,False,False,False,False,True,True,False,True,True,False], dtype = "bool")#candidate|878|(315,)|const|bool
call_877 = relay.TupleGetItem(func_617_call(relay.reshape(const_878.astype('bool'), [5, 9, 7])), 4)
call_879 = relay.TupleGetItem(func_620_call(relay.reshape(const_878.astype('bool'), [5, 9, 7])), 4)
output = relay.Tuple([call_857,const_858,call_860,bop_874,call_877,const_878,])
output2 = relay.Tuple([call_859,const_858,call_861,bop_874,call_879,const_878,])
func_880 = relay.Function([var_852,var_853,], output)
mod['func_880'] = func_880
mod = relay.transform.InferType()(mod)
mutated_mod['func_880'] = func_880
mutated_mod = relay.transform.InferType()(mutated_mod)
func_880_call = mutated_mod.get_global_var('func_880')
var_882 = relay.var("var_882", dtype = "uint8", shape = (4,))#candidate|882|(4,)|var|uint8
var_883 = relay.var("var_883", dtype = "uint8", shape = (4,))#candidate|883|(4,)|var|uint8
call_881 = func_880_call(var_882,var_883,)
output = call_881
func_884 = relay.Function([var_882,var_883,], output)
mutated_mod['func_884'] = func_884
mutated_mod = relay.transform.InferType()(mutated_mod)
var_889 = relay.var("var_889", dtype = "float32", shape = (14, 3))#candidate|889|(14, 3)|var|float32
const_890 = relay.const([[-7.967928,6.411370,9.790255],[8.510805,4.621484,8.543011],[0.771308,-7.245668,-0.304782],[8.703680,-9.224677,-9.311450],[3.409903,6.154019,-5.708836],[7.182166,0.121901,5.604480],[3.726484,-6.862452,-1.588023],[-0.580105,-7.452246,4.986867],[-6.037931,-9.192343,9.590009],[-4.535244,-0.667329,4.472907],[-0.132484,2.405475,9.807460],[-5.427392,-0.257819,6.821652],[-4.846893,-1.575109,2.493596],[7.220988,-2.315974,-0.725280]], dtype = "float32")#candidate|890|(14, 3)|const|float32
bop_891 = relay.divide(var_889.astype('float32'), relay.reshape(const_890.astype('float32'), relay.shape_of(var_889))) # shape=(14, 3)
output = relay.Tuple([bop_891,])
output2 = relay.Tuple([bop_891,])
func_896 = relay.Function([var_889,], output)
mod['func_896'] = func_896
mod = relay.transform.InferType()(mod)
mutated_mod['func_896'] = func_896
mutated_mod = relay.transform.InferType()(mutated_mod)
var_897 = relay.var("var_897", dtype = "float32", shape = (14, 3))#candidate|897|(14, 3)|var|float32
func_896_call = mutated_mod.get_global_var('func_896')
call_898 = func_896_call(var_897)
output = call_898
func_899 = relay.Function([var_897], output)
mutated_mod['func_899'] = func_899
mutated_mod = relay.transform.InferType()(mutated_mod)
func_355_call = mod.get_global_var('func_355')
func_357_call = mutated_mod.get_global_var('func_357')
call_916 = relay.TupleGetItem(func_355_call(), 1)
call_917 = relay.TupleGetItem(func_357_call(), 1)
var_932 = relay.var("var_932", dtype = "uint64", shape = (15, 15))#candidate|932|(15, 15)|var|uint64
bop_933 = relay.left_shift(call_916.astype('int8'), relay.reshape(var_932.astype('int8'), relay.shape_of(call_916))) # shape=(15, 15)
bop_936 = relay.left_shift(call_917.astype('int8'), relay.reshape(var_932.astype('int8'), relay.shape_of(call_917))) # shape=(15, 15)
bop_938 = relay.mod(var_932.astype('float32'), relay.reshape(bop_933.astype('float32'), relay.shape_of(var_932))) # shape=(15, 15)
bop_941 = relay.mod(var_932.astype('float32'), relay.reshape(bop_936.astype('float32'), relay.shape_of(var_932))) # shape=(15, 15)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
const_946 = relay.const([1.506699,1.891124,-1.965304,-5.224236,-6.773153,6.097383,8.056124,9.448505,-4.179242,2.475259,-2.458769,2.729156,8.630886,-3.431224,4.023412,-9.163556,-9.529386,-6.938130,-3.204035,5.024509,7.343498,-0.749716,-6.182249,5.618302,8.938981,1.153999,-2.447666,-3.234851,4.960823,-9.680495,2.249680,9.214801,9.406886,-5.629262,8.482415,-6.169424,4.123021,0.216888,3.243852,-0.378360,-2.120777,-6.119880,7.119146,-4.839998,5.963564,6.262137,5.974968,-8.177486,-1.587647,8.296000,-3.191265,-0.635939,-2.952731,7.682098,-6.967724,-5.690815,-6.048385,8.721438,3.701840,-8.130184,1.549552,-0.196555,-2.405501,2.411055,3.983419,-1.987497,-0.953881,7.194501,0.742557,-3.966422,5.228487,-5.414154,2.370136,6.422828,8.631683,3.404480,5.803597,8.779935,-4.557475,-9.290420,9.239350,3.450821,0.778579,-6.173043,-0.003671,8.072714,-4.894780,0.610665,7.472144,-8.776130,-4.237681,2.001200,6.168988,6.778000,0.568434,-6.676146,1.417157,9.263605,-2.138383,3.496896,-3.318657,-5.632251,-0.131158,5.569730,-3.695376,-3.954692,2.608004,-6.575233,-6.457412,4.139179,9.336807,8.187387,6.980323,-1.654812,-3.481922,-4.262439,-2.511267,7.138857,-1.578744,-3.512763,3.105899,4.254300,6.765350,1.545304,-9.297922,8.870729,-5.559573,-6.535499,0.833488,-5.154835,-0.290292,2.897476,9.189678,1.338353,9.677161,4.107145,-4.264247,3.940897,-1.530216,6.748383,3.252880,5.036539,3.284174,-7.256254,-5.247902,3.343671,9.161935,0.229512,2.346972,5.679849,-9.895595,8.980077,7.225864,-3.171124,-6.771860,-0.801277,7.388832,4.565957,3.700928,7.777137,-6.891549,-2.204720,5.725199,8.118379,-0.542387,6.932472,7.624190,0.345355,-2.377488,-3.597678,-1.542573,-2.710029,-7.282935,-6.060555,7.693720,6.948402,-4.419009,-4.080263,3.523652,6.814289,-9.528278,9.990910,0.166608,-8.955062,-5.815690,-9.068790,-9.751317,-0.442429,2.046758,-2.075983,3.164583,-9.506120,-6.503147,4.782244,-7.324256,6.197498,-3.426865,-9.544068,-5.189686,1.678909,-6.711274,6.914280,-5.287479,1.380904,-1.797519,9.192879,-7.624503,-2.648726,-1.366286,-2.644618,-8.397614,7.309075,-0.324168,-8.399469,-5.651720,-8.779225,-2.941917,6.904623,6.523834,-5.931129,5.117022,3.607062,4.445274,-8.692988,-5.605907,9.005139,-3.978462,0.300310,-8.378810,5.277047,1.962454,2.038138,3.458133,-1.213959,-4.701886,9.124522,-4.285564,1.993694,-4.104915,7.843752,-9.902863,2.729372,1.657810,-2.758170,-2.541966,9.396546,8.715840,9.847863,4.132580,4.219639,-5.495081,6.758096,-5.549000,2.696188,-9.336890,4.970722,7.053090,-7.881925,4.650173,-5.517883,-2.615819,7.383111,3.727365,9.937772,-5.367925,-7.765415,7.283328,-8.742992,-6.070932,-7.799853,-9.769319,8.505568,-9.813025,7.212236,-2.504683,2.159629,6.498838,-6.691880,-7.072188,5.450276,6.620006,5.543049,-0.898373,-6.765061,4.010529,3.276513,-3.517239,9.902525,-7.662007,-9.145888,1.723494,-2.704686,-5.441965,5.606768,-0.322991,4.033089,-4.775839,-0.717447,-8.024416,0.573872,7.211567,-2.154095,0.580530,-5.344561,-9.319979,1.031253,-7.338624,-5.648368,-4.312964,2.965790,-2.126300,-4.190338,-2.655603,2.088752,7.842149,-1.061674,-6.642671,-7.395352,5.929506,4.873185,6.224444,5.297997,-4.808966,-0.734596,0.210363,2.500284,0.732168,-2.548650,-5.087740,-9.146881,-4.271189,-3.657327,6.448958,-9.049674,5.798101,-5.724319,-4.206554,3.202737,4.502312,-5.834077,1.923655,5.456731,6.811969,-3.370012,8.439989,-2.701706,-5.673654,-9.731940,8.684414,-6.624156,7.449403,-5.069125,-8.362594,-6.571693,1.118400,-1.577699,0.560475,8.875416,3.173864,-0.486837,2.362970,-8.932548,-8.405553,7.558627,1.091710,-4.213048,9.510170,0.803262,-2.418970,9.267031,-8.662100,4.509405,7.473328,-0.639818,-8.032314,-6.148424,7.541713,8.104545,4.050156,-4.983816,-1.803168,7.673833,4.025224,4.345900,8.995223,2.081730,-5.566145,7.471863,-3.459371,-7.761825,-2.008997,-3.900714,6.678847,4.962752,-9.364620,-3.157248,8.235301,-3.853232,-5.049765,-6.897402,7.982191,9.862803,-0.440754,8.532168,-4.136967,-8.160660,-0.783791,9.328894,-3.817890,6.122129,-0.597783,-9.644953,8.870659,-4.097713,-8.223348,1.336483,-9.902332,-5.720462,3.911947,1.620231,5.118853,0.387877,-2.484212,-3.360056,2.221806,-9.419445,5.198244,2.121183,9.469273,-8.620869,6.112547,9.857035,-4.022323,-0.207155,-6.710950,4.888548,7.532208,3.678101,7.871906,8.579990,0.429426,-3.952168,-0.449307,-3.326341,7.323381,5.422447,-6.843990,4.820502,-6.957976,1.784978,5.722339,1.039310,5.175489,7.070996,-5.955220,3.369035,-4.383214,-7.131066,-4.090091,-4.817838,5.004919,-4.493606,1.119472,-6.071123,4.704383,-1.357911,-6.604201,-4.610821,-5.730102,5.397256,-1.150544,6.387094,-6.806493,2.925158,-5.184104,6.172709,7.925548,6.842131,0.523040,8.516414,1.619217,9.486002,3.544695,4.027394,7.550718,-9.473619,-1.571669,-1.649817,2.674184,3.062461,-4.596327,5.304453,-7.391325,-9.033885,-8.537319,-3.045656,9.831337,7.451149,2.149308,-5.211621,0.633226,-9.892449,-6.566264,9.022762], dtype = "float64")#candidate|946|(504,)|const|float64
call_945 = relay.TupleGetItem(func_237_call(relay.reshape(const_946.astype('float64'), [6, 6, 14])), 1)
call_947 = relay.TupleGetItem(func_239_call(relay.reshape(const_946.astype('float64'), [6, 6, 14])), 1)
uop_950 = relay.cosh(bop_938.astype('float32')) # shape=(15, 15)
uop_952 = relay.cosh(bop_941.astype('float32')) # shape=(15, 15)
uop_953 = relay.acosh(var_932.astype('float32')) # shape=(15, 15)
output = relay.Tuple([call_945,const_946,uop_950,uop_953,])
output2 = relay.Tuple([call_947,const_946,uop_952,uop_953,])
func_960 = relay.Function([var_932,], output)
mod['func_960'] = func_960
mod = relay.transform.InferType()(mod)
mutated_mod['func_960'] = func_960
mutated_mod = relay.transform.InferType()(mutated_mod)
var_961 = relay.var("var_961", dtype = "uint64", shape = (15, 15))#candidate|961|(15, 15)|var|uint64
func_960_call = mutated_mod.get_global_var('func_960')
call_962 = func_960_call(var_961)
output = call_962
func_963 = relay.Function([var_961], output)
mutated_mod['func_963'] = func_963
mutated_mod = relay.transform.InferType()(mutated_mod)
func_727_call = mod.get_global_var('func_727')
func_728_call = mutated_mod.get_global_var('func_728')
call_999 = relay.TupleGetItem(func_727_call(), 1)
call_1000 = relay.TupleGetItem(func_728_call(), 1)
uop_1007 = relay.acos(call_999.astype('float64')) # shape=(5, 9, 7)
uop_1009 = relay.acos(call_1000.astype('float64')) # shape=(5, 9, 7)
uop_1014 = relay.asinh(call_999.astype('float64')) # shape=(5, 9, 7)
uop_1016 = relay.asinh(call_1000.astype('float64')) # shape=(5, 9, 7)
uop_1020 = relay.rsqrt(uop_1007.astype('float32')) # shape=(5, 9, 7)
uop_1022 = relay.rsqrt(uop_1009.astype('float32')) # shape=(5, 9, 7)
const_1023 = relay.const([[[2.073254,6.740050,5.446646,-3.939455,-6.599807,8.498512,3.133311],[5.096673,1.561361,6.738742,6.414222,-8.661983,2.124993,3.581937],[-8.174311,6.191761,-3.298509,7.830570,-2.092821,8.840369,3.296127],[-8.929324,4.149920,0.331422,1.967077,-9.080266,1.247600,4.332784],[-0.494301,9.346202,8.721401,-9.969541,7.477865,0.351382,1.151648],[9.383992,-3.000664,5.853521,8.864168,-8.038083,4.219151,6.646607],[8.029032,-8.875019,5.624956,-9.871011,-4.558451,0.860944,-6.322611],[8.230828,1.190971,-2.731638,-7.522789,-9.312888,6.224727,-2.242209],[8.180769,4.581232,-9.589897,-4.533362,0.248554,-2.965345,2.385641]],[[0.629953,5.082552,-6.686350,1.977564,-0.311259,-8.289127,-4.792826],[0.209783,3.185350,-4.388902,7.282543,-9.621505,9.478703,9.927844],[1.716865,1.198206,-0.149331,4.530940,2.469703,-1.227429,3.468882],[-8.944587,-3.569828,-3.911415,1.648229,-7.802432,-5.872627,-4.004316],[-9.575022,7.887433,6.637432,-3.316220,7.668553,-9.160963,7.112287],[-3.691999,8.449337,-0.169986,-9.908487,1.252101,0.360336,1.858836],[-2.607541,4.243181,6.275239,-1.488064,0.930806,-2.927510,2.056087],[-6.588255,8.488241,4.407169,9.671820,-9.102215,-6.288039,6.384983],[8.631499,0.421007,4.155745,-7.689023,-0.752878,2.726894,-1.873702]],[[-9.518169,-6.048075,-9.720019,-1.596953,-8.841266,9.008776,2.033287],[2.164814,9.316778,0.804867,-4.179403,-1.408127,8.970728,-2.264113],[-5.464665,5.600273,-4.808394,9.536035,1.051014,-8.688979,-5.585983],[0.657158,3.610679,-9.076068,8.288597,-8.990241,-0.018850,8.818395],[3.062214,-0.216277,-8.262635,-7.789797,1.435635,1.060314,5.924517],[6.715853,4.321160,-8.415292,3.619429,1.267311,-9.038880,-6.767441],[-6.326683,3.311224,1.392379,-2.205883,3.050751,-8.832349,-6.504762],[3.354323,0.030538,6.859095,3.335623,-4.914287,9.171541,-2.427129],[3.705740,-4.530903,-9.126259,8.798930,-0.106443,3.680984,-4.036434]],[[-7.550019,8.785529,5.620726,-5.229049,7.630783,-1.333696,-8.295007],[1.354858,7.281431,-8.001597,3.790287,7.444243,-7.074960,-1.617547],[0.688241,-2.949932,0.756943,-4.558732,-3.332192,-3.399782,-2.973448],[0.033549,-0.340356,0.189348,0.654683,3.134127,-5.644789,5.113951],[-3.220354,-4.260727,1.004928,-0.941303,-4.523027,7.379573,1.904186],[-6.391039,-6.428342,4.886336,8.040256,-3.763808,-4.566246,1.484244],[7.345146,9.016366,-4.973745,0.028790,-9.712248,-0.967889,-8.883823],[-5.491876,1.963243,-2.216252,-7.514643,-7.991516,-4.560498,1.626205],[9.403599,-9.367071,9.868925,-6.963939,2.694043,4.880598,2.013153]],[[5.189707,3.793408,-0.971273,-6.369433,-5.378171,-2.521324,-3.813533],[-5.608065,0.699119,-1.804604,-8.560065,4.361241,5.385595,-3.986044],[0.593688,4.078684,-9.913307,-6.934610,3.285749,9.876965,6.024971],[7.375518,-5.766940,4.029326,0.984729,3.787711,-5.626406,-9.707057],[-9.437870,-7.187115,6.885125,-1.100689,4.911315,-5.026835,2.966061],[-6.525490,3.818623,4.848542,8.342193,1.167542,-6.799519,-0.826601],[2.883094,1.049999,-9.375728,9.194050,3.050248,7.247155,-1.257790],[9.821635,1.330324,5.069674,2.590638,-0.435581,7.717050,7.529047],[9.139398,-7.281104,5.929233,-0.971709,-2.475239,6.701469,9.450402]]], dtype = "float32")#candidate|1023|(5, 9, 7)|const|float32
bop_1024 = relay.not_equal(uop_1020.astype('bool'), relay.reshape(const_1023.astype('bool'), relay.shape_of(uop_1020))) # shape=(5, 9, 7)
bop_1027 = relay.not_equal(uop_1022.astype('bool'), relay.reshape(const_1023.astype('bool'), relay.shape_of(uop_1022))) # shape=(5, 9, 7)
func_796_call = mod.get_global_var('func_796')
func_798_call = mutated_mod.get_global_var('func_798')
var_1030 = relay.var("var_1030", dtype = "float64", shape = (504,))#candidate|1030|(504,)|var|float64
call_1029 = relay.TupleGetItem(func_796_call(relay.reshape(var_1030.astype('float64'), [504,])), 1)
call_1031 = relay.TupleGetItem(func_798_call(relay.reshape(var_1030.astype('float64'), [504,])), 1)
func_548_call = mod.get_global_var('func_548')
func_551_call = mutated_mod.get_global_var('func_551')
var_1034 = relay.var("var_1034", dtype = "int64", shape = (48,))#candidate|1034|(48,)|var|int64
var_1035 = relay.var("var_1035", dtype = "uint64", shape = (225,))#candidate|1035|(225,)|var|uint64
call_1033 = relay.TupleGetItem(func_548_call(relay.reshape(var_1034.astype('int64'), [12, 4]), relay.reshape(var_1035.astype('uint64'), [225,]), ), 4)
call_1036 = relay.TupleGetItem(func_551_call(relay.reshape(var_1034.astype('int64'), [12, 4]), relay.reshape(var_1035.astype('uint64'), [225,]), ), 4)
func_727_call = mod.get_global_var('func_727')
func_728_call = mutated_mod.get_global_var('func_728')
call_1037 = relay.TupleGetItem(func_727_call(), 1)
call_1038 = relay.TupleGetItem(func_728_call(), 1)
const_1040 = relay.const([[[True,False,True,True,False,True,False],[True,True,False,False,False,True,False],[True,True,False,False,True,False,False],[True,True,True,True,False,True,True],[True,False,True,False,False,True,False],[False,False,False,True,True,False,True],[True,False,True,False,True,True,True],[False,False,True,True,True,True,True],[False,False,False,True,False,False,False]],[[False,True,False,True,True,False,False],[False,True,False,True,True,True,False],[False,True,True,False,True,False,False],[False,True,True,False,True,True,False],[True,True,False,False,False,False,True],[True,False,True,True,True,True,True],[False,True,True,False,False,False,True],[True,False,True,True,False,False,False],[True,True,False,True,True,False,True]],[[True,False,False,True,False,True,False],[False,True,True,False,False,False,True],[False,False,False,True,True,True,False],[True,True,False,False,False,False,True],[True,False,False,True,False,False,False],[True,False,False,True,True,False,True],[True,False,False,True,True,False,False],[False,True,False,True,False,True,True],[True,True,True,False,False,True,True]],[[False,False,True,True,False,False,False],[True,False,False,False,False,False,True],[True,True,True,True,False,True,False],[True,False,False,False,False,True,False],[False,False,False,False,False,True,False],[False,True,False,True,True,False,True],[False,True,False,True,False,True,True],[False,True,True,True,True,True,True],[True,False,False,True,True,False,False]],[[True,True,False,False,True,False,True],[False,True,True,False,False,True,False],[False,True,True,False,False,False,False],[True,False,True,True,False,True,False],[True,True,False,True,True,False,True],[False,True,False,False,False,True,True],[False,True,True,False,False,False,False],[False,False,False,True,False,False,False],[True,True,True,False,True,True,True]]], dtype = "bool")#candidate|1040|(5, 9, 7)|const|bool
bop_1041 = relay.less_equal(bop_1024.astype('bool'), relay.reshape(const_1040.astype('bool'), relay.shape_of(bop_1024))) # shape=(5, 9, 7)
bop_1044 = relay.less_equal(bop_1027.astype('bool'), relay.reshape(const_1040.astype('bool'), relay.shape_of(bop_1027))) # shape=(5, 9, 7)
bop_1050 = relay.maximum(uop_1014.astype('float32'), relay.reshape(bop_1024.astype('float32'), relay.shape_of(uop_1014))) # shape=(5, 9, 7)
bop_1053 = relay.maximum(uop_1016.astype('float32'), relay.reshape(bop_1027.astype('float32'), relay.shape_of(uop_1016))) # shape=(5, 9, 7)
uop_1058 = relay.cosh(uop_1014.astype('float64')) # shape=(5, 9, 7)
uop_1060 = relay.cosh(uop_1016.astype('float64')) # shape=(5, 9, 7)
func_237_call = mod.get_global_var('func_237')
func_239_call = mutated_mod.get_global_var('func_239')
call_1061 = relay.TupleGetItem(func_237_call(relay.reshape(call_1029.astype('float64'), [6, 6, 14])), 0)
call_1062 = relay.TupleGetItem(func_239_call(relay.reshape(call_1029.astype('float64'), [6, 6, 14])), 0)
var_1068 = relay.var("var_1068", dtype = "bool", shape = (5, 9, 7))#candidate|1068|(5, 9, 7)|var|bool
bop_1069 = relay.greater_equal(bop_1041.astype('bool'), relay.reshape(var_1068.astype('bool'), relay.shape_of(bop_1041))) # shape=(5, 9, 7)
bop_1072 = relay.greater_equal(bop_1044.astype('bool'), relay.reshape(var_1068.astype('bool'), relay.shape_of(bop_1044))) # shape=(5, 9, 7)
var_1074 = relay.var("var_1074", dtype = "bool", shape = (5, 9, 7))#candidate|1074|(5, 9, 7)|var|bool
bop_1075 = relay.mod(bop_1024.astype('float64'), relay.reshape(var_1074.astype('float64'), relay.shape_of(bop_1024))) # shape=(5, 9, 7)
bop_1078 = relay.mod(bop_1027.astype('float64'), relay.reshape(var_1074.astype('float64'), relay.shape_of(bop_1027))) # shape=(5, 9, 7)
uop_1079 = relay.sigmoid(bop_1075.astype('float64')) # shape=(5, 9, 7)
uop_1081 = relay.sigmoid(bop_1078.astype('float64')) # shape=(5, 9, 7)
output = relay.Tuple([call_1029,var_1030,call_1033,var_1034,var_1035,call_1037,bop_1050,uop_1058,call_1061,bop_1069,uop_1079,])
output2 = relay.Tuple([call_1031,var_1030,call_1036,var_1034,var_1035,call_1038,bop_1053,uop_1060,call_1062,bop_1072,uop_1081,])
func_1085 = relay.Function([var_1030,var_1034,var_1035,var_1068,var_1074,], output)
mod['func_1085'] = func_1085
mod = relay.transform.InferType()(mod)
mutated_mod['func_1085'] = func_1085
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1085_call = mutated_mod.get_global_var('func_1085')
var_1087 = relay.var("var_1087", dtype = "float64", shape = (504,))#candidate|1087|(504,)|var|float64
var_1088 = relay.var("var_1088", dtype = "int64", shape = (48,))#candidate|1088|(48,)|var|int64
var_1089 = relay.var("var_1089", dtype = "uint64", shape = (225,))#candidate|1089|(225,)|var|uint64
var_1090 = relay.var("var_1090", dtype = "bool", shape = (5, 9, 7))#candidate|1090|(5, 9, 7)|var|bool
var_1091 = relay.var("var_1091", dtype = "bool", shape = (5, 9, 7))#candidate|1091|(5, 9, 7)|var|bool
call_1086 = func_1085_call(var_1087,var_1088,var_1089,var_1090,var_1091,)
output = call_1086
func_1092 = relay.Function([var_1087,var_1088,var_1089,var_1090,var_1091,], output)
mutated_mod['func_1092'] = func_1092
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1111 = relay.var("var_1111", dtype = "float32", shape = (13, 15, 10))#candidate|1111|(13, 15, 10)|var|float32
uop_1112 = relay.tan(var_1111.astype('float32')) # shape=(13, 15, 10)
output = uop_1112
output2 = uop_1112
F = relay.Function([var_1111,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_1111,], output2)
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
input_1111= np.array([[[5.874049,-1.666931,6.114563,-3.270139,3.920259,9.846293,7.178607,5.264083,-3.749992,0.787621],[-4.447322,-1.306759,-1.955372,8.677103,0.853397,-7.060253,7.508612,4.726920,-4.214575,-1.956722],[-3.153329,-6.774207,4.084709,8.483637,-9.576743,-9.724673,-2.650074,-0.094151,8.540192,-0.032126],[-7.176963,9.982237,8.311068,-1.760193,6.094380,5.987947,9.055008,3.347460,1.738225,-6.731416],[-6.116443,-3.428214,7.551694,5.050325,7.882411,2.556165,-8.094778,0.387616,0.872647,3.140986],[-3.866875,-1.589844,9.230877,-4.881970,-5.107329,-2.187603,0.516143,9.899940,9.567049,-7.397325],[-4.744804,-9.334974,-7.745525,-9.154594,-1.965158,-8.839678,-6.035824,-5.400500,6.792407,-8.440171],[9.100419,-2.275493,-9.806374,8.259551,-8.128449,-3.428948,0.121406,0.665403,-2.907721,-1.925213],[-9.785617,5.077954,5.801655,-5.532750,-6.481231,7.501141,5.874135,7.988989,7.390211,-1.875616],[-2.428829,5.075902,1.885585,6.631002,-5.980879,-8.476378,5.929541,-7.231348,9.644693,8.295574],[8.884024,7.973610,-6.500862,-5.020268,9.586909,6.833903,8.145807,-0.337507,5.367967,-9.592369],[4.978713,-3.002189,-1.720589,9.178829,6.789226,-5.269168,8.556445,-0.966882,-2.947810,-8.896722],[2.239953,5.617226,-2.749568,5.483540,8.244844,9.866183,4.430029,1.522726,-2.352848,2.171706],[6.849383,4.805918,-0.118710,-4.560366,-3.667970,-0.917957,-3.419245,3.745026,5.045726,8.890901],[-0.669734,4.435317,-2.141617,-5.585812,-6.477962,2.104527,4.224844,-5.374716,-7.418525,-8.800456]],[[8.484581,-9.321279,-0.062078,-1.433116,-9.367770,-6.403822,5.452990,-4.929999,8.473561,-5.900934],[8.879173,6.076496,-7.329886,-0.922389,9.966018,-5.711675,5.170631,-9.394263,-8.803601,-7.020720],[-3.912184,6.350113,-4.523443,-5.954265,-1.796930,7.310487,-6.855988,-1.433300,8.016009,1.613952],[1.234046,2.071157,-5.253987,-6.102256,5.756585,7.014914,-0.007717,2.118471,3.916912,-5.909818],[-2.846418,3.316090,-1.468712,-1.430043,-1.952763,-7.254272,1.854634,-0.965474,-6.818794,-9.196475],[4.921003,8.479190,-3.461031,8.704975,-9.652173,-8.216766,8.066118,7.865264,3.056794,5.997997],[9.858756,-6.164313,-7.937418,-8.022724,1.233175,-8.068953,3.851823,-5.740878,-3.877569,7.504240],[2.563994,0.319317,8.965297,7.756658,-1.736680,-7.460135,-7.791438,-6.511257,2.684009,6.639157],[5.184939,-7.112572,-3.801144,9.754178,-6.766159,6.283607,-2.415353,-6.434264,-2.998572,-8.234470],[3.148913,-3.910209,8.566790,-5.812847,-0.680607,-9.438036,4.837855,-1.915290,6.816283,-7.313348],[-1.598737,-2.641353,-7.284775,-9.985875,-5.219522,-0.586397,4.248489,7.571820,6.128442,8.443156],[-0.909285,-0.452190,-4.155728,4.993864,-1.419178,4.585586,5.938623,3.876970,2.973775,-5.493183],[-2.311639,9.016584,-6.762697,0.422825,-3.899475,2.451423,-8.028731,-0.110766,-5.675500,-6.517935],[8.094840,3.096965,9.764264,-2.517531,-7.542640,-1.328611,1.875728,-9.815588,6.432678,-1.072176],[-4.961352,-2.573081,0.491990,-5.648274,-6.275026,-3.539731,-4.359588,-2.228856,-6.008170,-9.208509]],[[-1.137039,3.183704,-4.963941,2.280580,7.357153,2.075918,8.496440,-1.037803,-6.875163,-3.162858],[-0.879256,-9.212763,-2.240581,-5.602426,-7.224515,8.229492,-3.986130,-2.462707,-8.179874,-7.904258],[6.843024,1.102381,1.648517,9.159423,-2.329230,-3.950375,-0.851151,-9.499378,5.940671,2.311587],[4.387692,-7.454492,-3.626297,6.714250,-7.575871,-0.834682,3.599670,-9.578939,-4.858448,7.892492],[7.532978,6.484041,-3.408355,-4.483580,-7.457935,1.312828,-2.853136,2.252070,8.151487,1.372186],[-8.424033,9.720005,-1.854835,-9.531800,-4.880360,3.848496,9.381657,3.256586,-8.253433,-9.265516],[-1.347624,6.938904,6.462299,6.490557,1.720801,1.852505,-5.778764,5.505787,-9.528606,8.903727],[1.208274,8.613294,7.966669,2.743196,-4.719351,3.951289,5.853230,-4.210729,0.988946,3.949555],[-3.041186,6.574230,-9.263055,9.357160,-9.412440,1.513236,-5.720629,1.611798,8.273954,0.255584],[-4.654289,-4.361593,4.438837,-8.967164,-6.379684,7.065229,-8.234688,-6.015920,-7.715734,-5.010311],[-4.931908,-3.448723,-2.638323,8.201513,-3.008649,5.337803,9.096142,1.279800,-2.156666,4.526480],[1.382674,-5.966215,8.096412,-4.659801,-7.779845,8.788466,-8.044065,0.518720,9.384888,4.012918],[-7.274932,-0.463846,-1.230407,-5.215670,3.998822,-4.903349,4.538719,7.957591,8.004613,-3.655729],[-3.522213,-3.194149,0.124729,7.290294,-3.783498,5.356275,1.042925,-6.874948,3.803603,1.758413],[4.199382,8.163300,-1.316728,9.780220,7.763081,1.657129,7.120834,-1.144465,-2.266473,7.617729]],[[-3.432121,5.356060,-1.547714,-2.504521,-8.291696,5.888520,1.165648,-9.472777,6.909766,5.945889],[8.930814,-8.891243,-8.694688,-7.969113,-8.605928,-2.982239,-6.563277,-3.972917,8.987275,-5.030855],[-3.846253,0.410122,-5.752948,4.920623,-1.876509,-1.276284,-1.588941,-1.198801,-7.311957,0.609881],[3.750315,6.600600,7.090294,-4.174442,-3.265780,-0.630600,-8.181517,-4.153602,-4.357176,-6.732008],[-9.884469,-5.065659,9.010137,-2.665012,0.268504,-3.399749,3.096961,-8.873782,-9.901284,8.157785],[2.145054,4.681618,-3.923219,7.291488,-3.701245,0.222441,-8.305386,2.417314,-3.146599,0.255237],[6.775807,-4.871801,-2.021146,2.013184,-9.929287,0.335170,-5.082719,9.299106,7.001654,5.636321],[7.633164,-3.773484,1.036247,7.722406,4.417228,4.608690,2.277073,3.209986,5.422007,-6.539592],[-3.617177,6.211243,-4.823527,2.994132,6.027163,3.074947,0.091833,0.052174,1.986114,-6.693208],[1.788306,6.771894,-6.345889,-8.208790,7.062662,5.934487,4.745542,8.206904,2.335301,8.644103],[4.732728,3.365323,-2.007670,2.874834,-3.755865,0.819231,-1.081744,-0.008454,2.016615,-3.843515],[3.842541,-3.020168,-2.120327,-6.054605,-6.763136,4.660980,2.521529,-6.856585,9.542907,5.134425],[-0.098242,3.841255,-3.967884,2.490411,1.702563,-3.197363,0.420284,2.911046,-5.864947,9.664175],[5.150990,5.429129,-6.547594,-1.078991,8.735314,2.705209,-1.059236,-7.470626,-8.165513,9.696397],[5.245321,-9.390713,-0.108935,7.322330,4.127288,-7.536741,1.325033,8.851074,6.275092,4.558596]],[[-0.158613,-9.919480,3.942151,2.661417,-5.130387,9.558430,-1.136293,-0.787176,3.395151,-3.488837],[3.534033,5.164971,-6.955540,-0.980701,0.831497,-3.322400,-9.125426,-2.227948,-6.129144,0.491689],[-5.816488,-9.755393,-0.573610,-6.670701,-0.240103,-5.345189,8.779723,-8.659665,6.583666,5.676143],[-4.901863,-3.256668,2.976533,5.000125,-8.493305,-6.189393,6.996434,0.702468,-6.632756,-0.638046],[-9.551300,-3.068036,-2.222108,2.631568,3.991794,-8.052522,1.754965,-8.719398,-4.791815,-4.410138],[-1.340239,7.708577,1.930170,1.682073,-8.535543,8.332759,-8.071396,-4.914608,-1.575932,-7.085139],[6.091125,-6.663300,9.373368,2.083977,-0.019704,-9.763527,-5.493343,-5.447338,8.043831,2.244947],[8.798009,-5.499254,-9.739639,4.520311,3.214495,-4.636906,-8.421861,0.730348,0.569624,1.795274],[-8.621757,-4.331771,-5.261833,-2.588191,-7.494103,5.980241,-0.744766,-7.659552,3.608400,-8.853000],[6.257579,2.360783,2.641848,-4.043636,0.555208,-3.095700,5.514735,-1.914153,0.212980,1.198953],[8.361546,-6.441503,8.005771,-1.102881,-9.627624,6.202659,-8.296096,0.196487,9.639657,-3.233216],[-9.373449,-5.179054,-0.379868,9.070066,5.199142,9.655099,6.789703,4.035049,-5.079959,1.101289],[-9.823184,3.560244,9.310795,-2.207638,-3.971858,4.707143,9.115255,-4.461306,5.037570,0.249118],[2.728652,8.552776,-0.224715,5.891555,-7.630314,9.132330,6.671316,2.215345,-3.017933,5.063749],[0.683762,4.222364,-7.886704,-2.641529,9.103644,-9.625281,-5.674413,7.114652,-9.574704,-5.260063]],[[-8.569274,6.461045,-2.641677,-2.570258,-5.471106,3.217869,2.371778,-9.464510,-4.335634,9.230588],[5.016167,-2.333916,2.884566,-0.133939,-5.006365,-8.092882,4.272236,-2.338026,4.414327,-9.211735],[-3.853341,6.791786,-4.803029,1.566450,-8.683160,-7.407466,2.412637,1.713993,-5.789875,-2.573220],[-9.762893,4.710191,0.523148,2.204168,1.069381,-7.116163,9.174062,3.943651,-9.998972,5.189024],[1.321222,-8.791601,-6.649091,0.564001,9.788117,-5.015871,-3.116369,2.396729,1.567494,-7.573391],[-3.005960,4.161087,-3.373054,2.291274,7.563243,9.357872,-2.217370,4.260789,-2.255438,-8.609681],[8.374341,8.711106,-1.726026,4.953963,-5.808151,-6.945683,-9.037486,-2.020729,-2.335702,6.663882],[-1.471297,-4.589492,-4.859354,-7.902759,-5.852913,5.130598,2.747135,-9.124168,0.099058,7.430809],[9.829668,9.307645,0.831198,6.650472,7.840714,-4.902528,0.603313,9.320842,-7.808196,-6.604362],[-8.033947,-4.821370,7.545074,9.491423,-6.637412,-4.345827,8.848648,-1.548814,0.566019,-6.604079],[-4.023679,6.228288,-1.545461,-1.493090,-0.463667,9.756902,-9.809421,-3.478711,-0.704623,-3.410293],[-0.338647,9.257610,-7.201850,7.224818,3.734188,3.181149,-7.600810,2.302088,3.799220,9.834874],[5.660800,-3.435789,1.599575,8.760772,-2.155172,-1.575055,-4.731199,-9.276739,3.624353,-7.916155],[-1.432586,2.878445,-3.012404,4.407523,-9.627322,4.769710,3.057716,2.178938,-7.668003,7.168165],[-0.651391,-2.027043,6.859781,-4.734494,-0.772981,2.432668,-2.758862,9.070596,6.149718,-3.152268]],[[0.228083,1.831094,6.677778,4.081256,3.346014,-3.667228,5.774720,-2.780122,-8.795251,6.798809],[-0.478703,8.903283,2.417190,3.177303,1.502649,-1.213856,-6.587541,4.869309,-1.009857,3.174523],[8.771805,-2.197223,-4.333321,1.862534,5.969005,6.268340,1.581603,-3.347877,3.695185,-0.172363],[7.102140,-9.172234,-0.259387,3.637195,-3.736973,6.405767,-9.042581,-3.260137,-5.351827,4.836208],[0.709684,4.821070,2.860137,9.240777,-9.864413,5.849977,9.746295,8.537046,-9.318905,-7.085144],[0.762257,-2.531463,8.559057,8.541852,7.425928,1.610230,9.061221,2.966235,-0.729121,-2.764740],[-8.738050,-5.666484,3.003459,-4.738012,-2.036720,-9.293337,-6.216879,-2.705158,-6.335501,6.960727],[-4.810719,-9.507761,-4.961680,8.905798,-5.570133,0.802511,4.225826,-3.510262,7.317507,-1.051965],[-4.336649,6.540459,-1.719882,-0.962051,4.265442,-7.307505,5.855254,-4.079599,-3.855289,-5.347205],[5.038286,-4.675605,-1.726698,7.055346,6.652827,9.055976,-2.895558,-9.146693,-8.284361,8.325446],[-9.314631,-2.428793,-0.028831,0.111652,-7.845965,-2.307266,1.600808,-9.860517,2.821258,4.994664],[-2.586712,-3.554169,-6.848306,1.914961,-3.112951,-7.922641,-8.811440,-4.878850,7.886615,0.161331],[9.725411,2.637932,-4.995751,-1.324357,7.850797,-4.186347,7.553088,0.033770,7.014799,-5.810034],[5.736609,8.019133,7.688728,5.632979,1.553499,-9.529263,-3.109315,-3.117957,-5.215691,-3.731413],[-0.555136,5.874646,-0.926302,0.450494,6.690822,6.635268,5.233272,-8.459917,-9.576804,9.640333]],[[-5.400659,-4.560733,-9.892359,-3.531667,-3.529707,-0.040627,9.127254,-7.270690,-0.827744,-5.073828],[6.110178,3.819509,-5.311096,-6.619556,5.585780,-6.967405,5.779115,-4.449126,5.776207,-6.088478],[6.402933,9.788703,-9.887293,-0.142506,5.233639,-5.219197,-2.854573,8.475123,8.403931,9.437683],[-4.073403,5.020853,5.198735,6.369731,-1.109641,-8.147038,-2.882130,-5.686833,9.953617,-5.892481],[4.152425,3.322962,-9.136112,-4.522077,-3.557930,-9.319736,-7.298302,5.283227,7.733414,-8.320344],[-9.019995,2.542632,-5.155735,-3.833754,6.460574,3.299580,5.965289,-9.401769,8.661857,-2.875717],[4.329830,-7.727743,-6.360732,0.622310,-1.822021,1.584550,-6.231172,-4.242626,-5.825289,-9.926870],[8.518958,2.710451,-8.175563,-2.261903,-9.310940,-6.097329,0.744232,4.979379,-6.746050,3.878625],[6.753048,8.105762,7.659231,-6.527074,-0.294812,0.645338,-4.436677,4.419197,5.118619,9.373861],[-0.373430,7.058903,-6.870676,9.932076,1.591144,-1.917397,8.292878,-8.734255,1.661848,-7.689652],[0.713353,9.937285,-2.288249,-5.753662,-7.900491,-4.128111,2.297289,-3.410793,-7.673957,-2.860954],[-4.613874,-8.415214,3.546197,-9.384966,-7.101064,-5.820797,7.507835,7.040152,-6.829665,5.194590],[6.950842,5.264526,3.693094,9.677660,5.188513,2.254552,-3.337880,6.176140,-5.559394,-3.798721],[2.529834,-3.979269,-8.554372,-5.408621,1.738216,4.954425,3.745541,-5.925086,-2.188053,9.658431],[2.328603,6.942421,-1.975234,5.116655,-2.900736,8.600945,1.611979,-2.164816,-0.573988,-1.560682]],[[-0.699689,-1.875092,7.215268,-6.338797,-0.066347,9.894693,-3.055844,4.506001,-5.714828,7.403562],[-9.707868,-8.070975,-6.587387,5.987690,3.733975,0.220434,2.108960,-1.149643,-9.405067,-1.487187],[-1.428369,-8.841179,-1.119604,9.717006,2.033336,-5.587918,-7.208466,0.664929,9.443994,9.118204],[-2.673269,2.899950,-5.517440,-4.773287,-0.639068,-3.527274,0.941821,-8.560763,-3.311832,-1.184923],[-5.671972,-0.451675,0.911636,-5.948661,-5.890669,5.453877,-3.336166,8.761830,-5.064796,2.201470],[-9.221453,5.965819,-3.215181,-0.649734,-1.955938,-8.752286,3.050033,8.898206,-5.110094,5.847724],[4.878800,-4.035929,9.605516,-1.202486,2.814181,4.476549,7.773564,5.672149,-2.582179,4.732128],[-2.889042,0.062312,-0.428113,-2.317381,-6.660768,8.101485,6.454585,3.433137,8.675720,-3.712927],[3.255653,2.026566,0.549089,7.938113,-6.515089,-3.505677,-0.272519,-9.835614,-5.934431,-9.832384],[-3.874052,-4.790175,5.393794,7.896626,9.642843,0.588507,6.065556,-4.134792,-3.845023,-3.523398],[-3.715038,8.625211,-5.070091,0.636920,6.860625,-5.371381,1.504179,6.483786,-0.761253,-6.857429],[-3.592344,2.421995,-6.366121,-0.225974,3.245642,1.289763,4.434084,-4.577994,-2.851742,-8.510273],[8.755426,4.062916,9.619459,1.875154,0.542688,-1.075317,-5.917061,1.399139,5.621049,2.282239],[-8.626632,-3.626692,5.396067,-1.324283,-5.163066,1.150599,6.348334,-1.689762,0.479876,-8.399263],[9.490694,0.562764,-3.396298,-0.380055,-9.694106,-9.929449,-0.847677,9.236584,-6.503401,0.970873]],[[-2.057607,-4.546411,-0.321998,8.962409,3.158224,-0.900736,-6.081716,-6.660192,7.017763,-4.677305],[-6.920256,-3.968806,4.286246,-3.943778,4.925078,-7.022710,3.576130,7.447622,4.171030,0.421035],[7.225051,-5.419988,6.254191,-4.761766,5.962581,0.724597,-4.836270,9.855388,1.497427,9.293737],[1.270095,-5.830542,-0.997482,-1.358895,-5.538881,-1.035863,-4.442591,0.620740,2.109164,-4.786801],[-2.161482,-0.541897,3.722580,-4.925558,-7.543760,-3.396561,9.238036,-7.629286,-3.265042,-5.948593],[-8.433402,-9.397365,8.454863,4.683985,-4.440791,2.285291,5.921110,0.252926,5.006766,0.569081],[0.014402,-0.588758,-2.419857,-9.205551,9.170542,-2.320641,-1.140102,-3.548616,-8.477688,-1.891779],[5.255722,-1.744488,-9.978433,-6.640775,2.065354,4.462473,5.175780,-1.743345,-1.172004,-0.180038],[1.748729,4.538598,-8.088318,2.630338,6.201594,6.332402,3.553312,6.321095,1.492645,-9.888552],[2.807199,2.542535,3.643845,5.296916,-5.803355,0.175617,-9.966964,6.520454,4.134131,9.117273],[5.474099,9.223207,-0.882478,-4.922107,-7.321046,2.680658,-2.407587,-2.426802,-0.259021,3.270355],[-0.925428,-9.292092,5.540309,0.341322,-3.908887,-2.746900,4.870331,-4.939357,-3.851297,3.051535],[3.264318,9.277970,4.597703,7.754249,1.062347,-2.066581,-9.667987,-8.842706,8.667849,-8.467990],[-1.839983,6.413896,-7.069123,-1.627703,-5.573889,-2.198658,-3.926892,4.879247,3.066266,1.719034],[7.747175,-0.059522,-8.279983,4.572257,-3.501625,2.705256,-8.815774,-9.907494,-4.971802,-9.717158]],[[-3.358384,5.521436,-2.549192,-7.939926,7.419341,-7.820496,-1.781650,-4.547347,6.636504,-0.805992],[0.264218,-4.355274,-3.520113,7.708769,2.111250,0.134443,-0.916591,-5.932862,5.407541,-5.947913],[-0.450221,2.043987,-3.308415,8.436739,9.637433,5.501333,0.733900,2.187623,6.786148,8.114459],[-8.887462,-2.710993,9.009540,8.447912,-4.849334,0.950296,-4.459475,2.550132,1.050881,-5.203707],[-1.415277,5.983977,0.467804,8.030230,9.422984,-1.302225,9.718988,-5.195281,7.730562,8.334400],[-8.328545,9.082861,1.634610,3.606684,7.734731,4.966597,9.975706,-1.947290,9.829645,0.624710],[-4.739394,-0.311419,4.243175,-9.586019,-1.802987,-4.816336,-4.099708,4.866205,-4.070627,-5.012473],[-3.205406,2.185429,0.933342,5.825156,8.290274,4.402564,-2.534756,3.292335,8.564010,-4.261883],[1.761550,-2.185244,-8.104710,9.782701,-4.134107,4.427897,6.384074,-7.875745,-4.938331,-4.961061],[7.245270,3.700777,5.831885,2.400132,-1.432534,3.029899,-7.182663,7.617531,0.744891,8.878961],[4.889617,7.807222,9.018520,9.021782,5.229420,4.048569,7.652931,-9.391332,8.753302,3.287453],[2.983276,7.553974,-7.651426,5.472008,3.905316,0.254976,7.157759,1.180510,4.952075,-8.042711],[-2.257526,8.340405,-6.280598,0.164527,6.551552,2.692578,5.112770,-7.708322,5.012750,0.645268],[-6.065557,5.449754,9.641481,-5.041226,9.005248,7.070951,-3.844303,-4.483928,-2.867823,-7.402841],[-1.193415,3.209937,-9.496213,-2.770699,1.143389,-6.964587,-3.682907,1.685303,-7.809774,-2.492066]],[[-0.490701,-6.235602,-0.010582,9.639369,-2.091489,-3.062142,5.395417,-3.639510,8.040633,1.345930],[-3.672182,-5.892116,-2.679040,0.340045,5.071726,-1.418988,4.112297,-2.049725,9.468562,3.571272],[2.362350,2.948318,5.875726,-5.586565,-2.258582,4.876852,8.289853,0.051950,3.937107,0.186453],[7.179784,-1.681381,-9.707971,-1.530352,-3.429142,9.128880,-8.360618,-4.444062,-0.278891,-1.060123],[-9.886416,6.091109,-8.134843,6.893999,5.367325,-2.322640,0.290200,-2.456922,-2.667829,6.471243],[-0.745431,-4.209010,2.693878,-0.564018,-9.932215,-4.101840,-8.648344,-4.338058,9.186073,-7.482877],[-7.869651,7.726566,9.025434,7.241603,-7.755300,-9.728379,8.820086,8.953874,5.061115,-4.297924],[5.462092,-4.309230,3.949405,-2.208222,5.458722,2.729532,0.191350,1.401752,-3.007755,-5.085940],[5.833052,2.948551,-2.571997,-6.665259,-2.044763,9.448730,8.897169,4.388263,-4.626827,6.109499],[-2.198037,-2.347910,7.151742,-3.839696,-0.025639,8.079351,0.943176,5.598546,1.189327,3.162795],[-7.900421,7.635712,0.738523,-0.991054,-0.068510,9.478689,8.554635,9.611048,2.511965,-8.905820],[-7.850818,7.010606,5.844890,8.175378,-1.635799,-0.136355,5.994599,6.839284,4.325436,0.275722],[9.610004,-6.264197,-5.157756,-4.748424,-4.014851,3.957040,-4.949318,-9.389105,2.700550,-7.957109],[1.952441,4.088485,4.645730,5.020585,-1.654878,-9.867777,9.245438,-4.313478,-4.203207,3.528903],[-2.006466,-0.601626,5.646708,-3.541225,-3.342152,8.548534,3.829740,5.174252,6.948697,3.581864]],[[9.172302,1.616974,3.691139,-5.913739,-5.514116,-8.116521,-3.239497,6.210158,5.483489,4.255245],[8.335608,3.069282,7.912608,9.052202,4.111001,-6.192561,-1.893002,4.355149,-6.337106,-9.361729],[5.086119,5.206809,5.918688,-8.251177,-1.991482,-5.998238,-2.406362,-1.710714,-1.990495,-4.224292],[8.626071,-1.100252,-3.650951,5.088526,6.031594,-5.414810,-2.998764,2.812469,-3.803022,-4.009278],[8.220620,-6.039215,4.780031,8.206662,-2.979088,8.725883,-5.806902,-1.551900,-2.834173,2.108200],[7.976608,-5.354837,1.117384,4.881529,9.021610,-3.375238,-4.419572,-5.025857,-2.296750,-9.222979],[9.891759,-6.907205,9.913860,8.840510,-7.210140,-4.671281,-1.269379,3.983299,-9.712127,9.786711],[-8.107718,-5.597880,8.853312,-9.874198,-9.855305,-3.881331,2.230946,9.213604,7.264623,-9.806714],[-5.407454,9.112491,8.824476,-4.009869,4.924984,-3.502855,-9.351005,7.420573,4.680008,2.430854],[-4.866827,-2.355636,-6.875307,-6.177235,-1.752398,5.178958,0.938980,6.692801,1.616909,-9.158003],[5.035882,-0.731156,6.256922,8.569971,-9.329076,-3.664588,5.690558,-0.421807,8.517943,6.162452],[7.726547,-1.256688,-1.402347,1.201371,6.886646,-5.360351,-3.278231,-4.657837,-1.353250,0.135887],[-7.442558,0.142629,-8.986463,1.669509,6.324997,3.258609,3.217397,6.070692,5.343614,6.972162],[7.826892,4.737024,0.147167,6.525545,-5.598147,-1.991450,8.302870,-6.355756,5.283940,2.660292],[-7.957720,0.459442,-4.516422,9.760764,8.732058,-7.278079,7.243796,9.984696,-3.958266,-5.287445]]], dtype='float32')
module1.set_input('var_1111', input_1111)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_1111, )
res3 = intrp3.evaluate()(input_1111, )
res4 = intrp4.evaluate()(input_1111, )
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
module5.set_input('var_1111', input_1111)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_1111, )
res7 = intrp7.evaluate()(input_1111, )
res8 = intrp8.evaluate()(input_1111, )
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
module9.set_input('var_1111', input_1111)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_1111, )
res11 = intrp11.evaluate()(input_1111, )
res12 = intrp12.evaluate()(input_1111, )
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
module13.set_input('var_1111', input_1111)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_1111, )
res15 = intrp15.evaluate()(input_1111, )
res16 = intrp16.evaluate()(input_1111, )
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
module17.set_input('var_1111', input_1111)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_1111, )
res19 = intrp19.evaluate()(input_1111, )
res20 = intrp20.evaluate()(input_1111, )
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
module21.set_input('var_1111', input_1111)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_1111, )
res23 = intrp23.evaluate()(input_1111, )
res24 = intrp24.evaluate()(input_1111, )
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