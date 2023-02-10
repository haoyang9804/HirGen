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
var_19 = relay.var("var_19", dtype = "float32", shape = (15,))#candidate|19|(15,)|var|float32
uop_20 = relay.acosh(var_19.astype('float32')) # shape=(15,)
const_24 = relay.const([0.512973,-9.842062,-9.940993,-5.801219,-8.953344,-4.891634,9.691471,-4.313909,-1.351265,-1.200787,5.820199,6.137812,-7.228003,6.064286,3.015474], dtype = "float32")#candidate|24|(15,)|const|float32
bop_25 = relay.bitwise_or(uop_20.astype('int32'), relay.reshape(const_24.astype('int32'), relay.shape_of(uop_20))) # shape=(15,)
output = bop_25
output2 = bop_25
func_30 = relay.Function([var_19,], output)
mod['func_30'] = func_30
mod = relay.transform.InferType()(mod)
var_31 = relay.var("var_31", dtype = "float32", shape = (15,))#candidate|31|(15,)|var|float32
output = func_30(var_31)
func_32 = relay.Function([var_31], output)
mutated_mod['func_32'] = func_32
mutated_mod = relay.transform.InferType()(mutated_mod)
var_71 = relay.var("var_71", dtype = "int32", shape = (1, 14))#candidate|71|(1, 14)|var|int32
var_72 = relay.var("var_72", dtype = "int32", shape = (7, 14))#candidate|72|(7, 14)|var|int32
bop_73 = relay.maximum(var_71.astype('int32'), var_72.astype('int32')) # shape=(7, 14)
output = bop_73
output2 = bop_73
func_76 = relay.Function([var_71,var_72,], output)
mod['func_76'] = func_76
mod = relay.transform.InferType()(mod)
var_77 = relay.var("var_77", dtype = "int32", shape = (1, 14))#candidate|77|(1, 14)|var|int32
var_78 = relay.var("var_78", dtype = "int32", shape = (7, 14))#candidate|78|(7, 14)|var|int32
output = func_76(var_77,var_78,)
func_79 = relay.Function([var_77,var_78,], output)
mutated_mod['func_79'] = func_79
mutated_mod = relay.transform.InferType()(mutated_mod)
const_83 = relay.const([[7.706458,-7.396509,-7.092052,4.741504,0.304458,8.163950,6.933653,-1.591171,-1.885161,-4.150672,-0.786972,-9.011706],[4.933565,-9.233201,7.195094,6.192203,0.598090,2.997342,6.026371,-0.498220,3.930759,-3.515819,-1.412558,-8.647113],[-8.471605,1.739832,4.447303,-2.163194,-0.574279,-8.945596,-4.627172,-1.186810,7.909637,-6.566360,0.570761,-4.744879],[-0.077599,4.787679,0.768519,8.533369,-9.598667,-1.119638,-9.569075,-4.193102,-4.950448,5.986582,-7.127552,-4.562992],[5.440873,-4.172058,-0.198604,8.764800,2.276437,4.623097,-0.060563,-4.189529,1.289783,-0.213631,-3.064062,7.019312],[3.329262,1.496588,-5.630233,-4.251085,-0.945330,-7.282195,4.663481,-3.875743,-6.493292,2.572685,-4.770107,7.716908]], dtype = "float32")#candidate|83|(6, 12)|const|float32
uop_84 = relay.cosh(const_83.astype('float32')) # shape=(6, 12)
bop_89 = relay.bitwise_xor(const_83.astype('uint16'), relay.reshape(uop_84.astype('uint16'), relay.shape_of(const_83))) # shape=(6, 12)
uop_92 = relay.sigmoid(bop_89.astype('float32')) # shape=(6, 12)
func_76_call = mod.get_global_var('func_76')
func_79_call = mutated_mod.get_global_var('func_79')
var_95 = relay.var("var_95", dtype = "int32", shape = (14,))#candidate|95|(14,)|var|int32
var_96 = relay.var("var_96", dtype = "int32", shape = (98,))#candidate|96|(98,)|var|int32
call_94 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(var_96.astype('int32'), [7, 14]), )
call_97 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(var_96.astype('int32'), [7, 14]), )
uop_99 = relay.tan(uop_92.astype('float32')) # shape=(6, 12)
bop_101 = relay.floor_mod(uop_92.astype('float32'), relay.reshape(bop_89.astype('float32'), relay.shape_of(uop_92))) # shape=(6, 12)
bop_104 = relay.left_shift(uop_92.astype('uint64'), relay.reshape(const_83.astype('uint64'), relay.shape_of(uop_92))) # shape=(6, 12)
func_76_call = mod.get_global_var('func_76')
func_79_call = mutated_mod.get_global_var('func_79')
call_107 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(call_94.astype('int32'), [7, 14]), )
call_108 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(call_94.astype('int32'), [7, 14]), )
uop_110 = relay.sin(uop_99.astype('float32')) # shape=(6, 12)
uop_113 = relay.erf(uop_99.astype('float64')) # shape=(6, 12)
bop_118 = relay.equal(uop_92.astype('bool'), relay.reshape(bop_101.astype('bool'), relay.shape_of(uop_92))) # shape=(6, 12)
bop_122 = relay.power(uop_99.astype('float32'), relay.reshape(uop_84.astype('float32'), relay.shape_of(uop_99))) # shape=(6, 12)
bop_126 = relay.minimum(uop_99.astype('float64'), relay.reshape(bop_118.astype('float64'), relay.shape_of(uop_99))) # shape=(6, 12)
uop_129 = relay.log2(uop_110.astype('float64')) # shape=(6, 12)
var_133 = relay.var("var_133", dtype = "float32", shape = (6, 12))#candidate|133|(6, 12)|var|float32
bop_134 = relay.subtract(uop_110.astype('float32'), relay.reshape(var_133.astype('float32'), relay.shape_of(uop_110))) # shape=(6, 12)
bop_138 = relay.not_equal(uop_129.astype('bool'), relay.reshape(bop_89.astype('bool'), relay.shape_of(uop_129))) # shape=(6, 12)
func_76_call = mod.get_global_var('func_76')
func_79_call = mutated_mod.get_global_var('func_79')
call_141 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(var_96.astype('int32'), [7, 14]), )
call_142 = func_76_call(relay.reshape(var_95.astype('int32'), [1, 14]), relay.reshape(var_96.astype('int32'), [7, 14]), )
uop_145 = relay.acosh(uop_129.astype('float64')) # shape=(6, 12)
bop_148 = relay.bitwise_or(bop_138.astype('int64'), relay.reshape(uop_145.astype('int64'), relay.shape_of(bop_138))) # shape=(6, 12)
uop_151 = relay.log(uop_129.astype('float64')) # shape=(6, 12)
bop_153 = relay.logical_or(uop_145.astype('bool'), relay.reshape(bop_138.astype('bool'), relay.shape_of(uop_145))) # shape=(6, 12)
uop_157 = relay.log10(uop_151.astype('float32')) # shape=(6, 12)
output = relay.Tuple([call_94,var_95,var_96,bop_104,call_107,uop_113,bop_122,bop_126,bop_134,call_141,bop_148,bop_153,uop_157,])
output2 = relay.Tuple([call_97,var_95,var_96,bop_104,call_108,uop_113,bop_122,bop_126,bop_134,call_142,bop_148,bop_153,uop_157,])
func_159 = relay.Function([var_95,var_96,var_133,], output)
mod['func_159'] = func_159
mod = relay.transform.InferType()(mod)
var_160 = relay.var("var_160", dtype = "int32", shape = (14,))#candidate|160|(14,)|var|int32
var_161 = relay.var("var_161", dtype = "int32", shape = (98,))#candidate|161|(98,)|var|int32
var_162 = relay.var("var_162", dtype = "float32", shape = (6, 12))#candidate|162|(6, 12)|var|float32
output = func_159(var_160,var_161,var_162,)
func_163 = relay.Function([var_160,var_161,var_162,], output)
mutated_mod['func_163'] = func_163
mutated_mod = relay.transform.InferType()(mutated_mod)
const_168 = relay.const(-1.628894, dtype = "float64")#candidate|168|()|const|float64
const_169 = relay.const([8.309667,7.730290,0.757471,-9.434026,-8.914844,0.905676,2.298643,-4.438741], dtype = "float64")#candidate|169|(8,)|const|float64
bop_170 = relay.not_equal(const_168.astype('bool'), const_169.astype('bool')) # shape=(8,)
output = relay.Tuple([bop_170,])
output2 = relay.Tuple([bop_170,])
func_173 = relay.Function([], output)
mod['func_173'] = func_173
mod = relay.transform.InferType()(mod)
output = func_173()
func_174 = relay.Function([], output)
mutated_mod['func_174'] = func_174
mutated_mod = relay.transform.InferType()(mutated_mod)
var_183 = relay.var("var_183", dtype = "int8", shape = (12, 11, 1))#candidate|183|(12, 11, 1)|var|int8
var_184 = relay.var("var_184", dtype = "int8", shape = (12, 11, 8))#candidate|184|(12, 11, 8)|var|int8
bop_185 = relay.not_equal(var_183.astype('bool'), var_184.astype('bool')) # shape=(12, 11, 8)
func_30_call = mod.get_global_var('func_30')
func_32_call = mutated_mod.get_global_var('func_32')
const_192 = relay.const([3.396051,-3.969992,5.037864,-0.269631,7.623029,3.447149,3.863939,-0.017855,-6.806014,7.433774,9.400973,7.081134,-0.775831,5.219700,1.154623], dtype = "float32")#candidate|192|(15,)|const|float32
call_191 = func_30_call(relay.reshape(const_192.astype('float32'), [15,]))
call_193 = func_30_call(relay.reshape(const_192.astype('float32'), [15,]))
bop_196 = relay.multiply(var_184.astype('uint8'), var_183.astype('uint8')) # shape=(12, 11, 8)
var_200 = relay.var("var_200", dtype = "int8", shape = (12, 11, 8))#candidate|200|(12, 11, 8)|var|int8
bop_201 = relay.greater(var_184.astype('bool'), relay.reshape(var_200.astype('bool'), relay.shape_of(var_184))) # shape=(12, 11, 8)
func_76_call = mod.get_global_var('func_76')
func_79_call = mutated_mod.get_global_var('func_79')
const_209 = relay.const([-4,-8,6,10,3,-8,-3,10,-10,2,-10,-9,-3,2], dtype = "int32")#candidate|209|(14,)|const|int32
const_210 = relay.const([[-4],[-5],[7],[7],[10],[8],[10],[-4],[4],[-1],[-7],[-1],[-1],[6],[5],[-2],[2],[-7],[-5],[-1],[-2],[10],[7],[-10],[-10],[-2],[-10],[-9],[-5],[4],[-5],[-1],[6],[-6],[-2],[-6],[7],[6],[-8],[5],[-9],[-4],[-9],[9],[10],[-4],[-10],[1],[-2],[-6],[3],[-1],[-8],[2],[-3],[-3],[-3],[3],[-1],[-6],[6],[5],[-4],[5],[-5],[-2],[-3],[-9],[-10],[-6],[-10],[-5],[-7],[5],[-1],[-9],[2],[6],[-2],[3],[1],[6],[-8],[4],[9],[-1],[2],[-2],[-4],[-7],[9],[-5],[-1],[8],[-1],[-3],[-8],[-6]], dtype = "int32")#candidate|210|(98, 1)|const|int32
call_208 = func_76_call(relay.reshape(const_209.astype('int32'), [1, 14]), relay.reshape(const_210.astype('int32'), [7, 14]), )
call_211 = func_76_call(relay.reshape(const_209.astype('int32'), [1, 14]), relay.reshape(const_210.astype('int32'), [7, 14]), )
func_76_call = mod.get_global_var('func_76')
func_79_call = mutated_mod.get_global_var('func_79')
call_212 = func_76_call(relay.reshape(const_209.astype('int32'), [1, 14]), relay.reshape(const_210.astype('int32'), [7, 14]), )
call_213 = func_76_call(relay.reshape(const_209.astype('int32'), [1, 14]), relay.reshape(const_210.astype('int32'), [7, 14]), )
output = relay.Tuple([bop_185,call_191,const_192,bop_196,bop_201,call_208,const_209,const_210,call_212,])
output2 = relay.Tuple([bop_185,call_193,const_192,bop_196,bop_201,call_211,const_209,const_210,call_213,])
func_220 = relay.Function([var_183,var_184,var_200,], output)
mod['func_220'] = func_220
mod = relay.transform.InferType()(mod)
var_221 = relay.var("var_221", dtype = "int8", shape = (12, 11, 1))#candidate|221|(12, 11, 1)|var|int8
var_222 = relay.var("var_222", dtype = "int8", shape = (12, 11, 8))#candidate|222|(12, 11, 8)|var|int8
var_223 = relay.var("var_223", dtype = "int8", shape = (12, 11, 8))#candidate|223|(12, 11, 8)|var|int8
output = func_220(var_221,var_222,var_223,)
func_224 = relay.Function([var_221,var_222,var_223,], output)
mutated_mod['func_224'] = func_224
mutated_mod = relay.transform.InferType()(mutated_mod)
const_232 = relay.const([-2.165730,2.350912,-9.714172,6.733450,2.926300], dtype = "float64")#candidate|232|(5,)|const|float64
uop_233 = relay.sinh(const_232.astype('float64')) # shape=(5,)
uop_239 = relay.acosh(uop_233.astype('float32')) # shape=(5,)
bop_242 = relay.greater(uop_239.astype('bool'), relay.reshape(const_232.astype('bool'), relay.shape_of(uop_239))) # shape=(5,)
uop_246 = relay.tan(bop_242.astype('float32')) # shape=(5,)
func_30_call = mod.get_global_var('func_30')
func_32_call = mutated_mod.get_global_var('func_32')
const_249 = relay.const([[-8.533468],[8.519437],[7.200233],[0.278133],[0.633661],[5.500103],[-7.843668],[5.651687],[7.283937],[-4.229603],[-8.562735],[9.368916],[6.995114],[5.249286],[9.763994]], dtype = "float32")#candidate|249|(15, 1)|const|float32
call_248 = func_30_call(relay.reshape(const_249.astype('float32'), [15,]))
call_250 = func_30_call(relay.reshape(const_249.astype('float32'), [15,]))
func_173_call = mod.get_global_var('func_173')
func_174_call = mutated_mod.get_global_var('func_174')
call_251 = relay.TupleGetItem(func_173_call(), 0)
call_252 = relay.TupleGetItem(func_174_call(), 0)
var_257 = relay.var("var_257", dtype = "float32", shape = (5,))#candidate|257|(5,)|var|float32
bop_258 = relay.floor_mod(uop_246.astype('float32'), relay.reshape(var_257.astype('float32'), relay.shape_of(uop_246))) # shape=(5,)
output = relay.Tuple([call_248,const_249,call_251,bop_258,])
output2 = relay.Tuple([call_250,const_249,call_252,bop_258,])
func_264 = relay.Function([var_257,], output)
mod['func_264'] = func_264
mod = relay.transform.InferType()(mod)
mutated_mod['func_264'] = func_264
mutated_mod = relay.transform.InferType()(mutated_mod)
var_265 = relay.var("var_265", dtype = "float32", shape = (5,))#candidate|265|(5,)|var|float32
func_264_call = mutated_mod.get_global_var('func_264')
call_266 = func_264_call(var_265)
output = call_266
func_267 = relay.Function([var_265], output)
mutated_mod['func_267'] = func_267
mutated_mod = relay.transform.InferType()(mutated_mod)
var_272 = relay.var("var_272", dtype = "bool", shape = (11, 11))#candidate|272|(11, 11)|var|bool
var_273 = relay.var("var_273", dtype = "bool", shape = (11, 11))#candidate|273|(11, 11)|var|bool
bop_274 = relay.logical_and(var_272.astype('bool'), relay.reshape(var_273.astype('bool'), relay.shape_of(var_272))) # shape=(11, 11)
bop_277 = relay.bitwise_and(var_273.astype('uint64'), relay.reshape(bop_274.astype('uint64'), relay.shape_of(var_273))) # shape=(11, 11)
bop_281 = relay.logical_xor(var_272.astype('uint32'), relay.reshape(var_273.astype('uint32'), relay.shape_of(var_272))) # shape=(11, 11)
uop_285 = relay.exp(bop_277.astype('float32')) # shape=(11, 11)
bop_288 = relay.greater_equal(var_272.astype('bool'), relay.reshape(bop_277.astype('bool'), relay.shape_of(var_272))) # shape=(11, 11)
bop_293 = relay.divide(uop_285.astype('float32'), relay.reshape(var_272.astype('float32'), relay.shape_of(uop_285))) # shape=(11, 11)
func_30_call = mod.get_global_var('func_30')
func_32_call = mutated_mod.get_global_var('func_32')
const_298 = relay.const([6.064224,-3.426178,1.184936,-7.376918,4.894882,5.627220,-8.404824,2.886331,-4.785757,-1.518273,-3.680031,-7.529739,6.652526,-5.169608,-6.260980], dtype = "float32")#candidate|298|(15,)|const|float32
call_297 = func_30_call(relay.reshape(const_298.astype('float32'), [15,]))
call_299 = func_30_call(relay.reshape(const_298.astype('float32'), [15,]))
var_300 = relay.var("var_300", dtype = "float32", shape = (11, 11))#candidate|300|(11, 11)|var|float32
bop_301 = relay.left_shift(bop_293.astype('int64'), relay.reshape(var_300.astype('int64'), relay.shape_of(bop_293))) # shape=(11, 11)
uop_305 = relay.cosh(bop_293.astype('float64')) # shape=(11, 11)
bop_307 = relay.power(bop_293.astype('float64'), relay.reshape(bop_274.astype('float64'), relay.shape_of(bop_293))) # shape=(11, 11)
uop_312 = relay.tan(uop_305.astype('float32')) # shape=(11, 11)
uop_314 = relay.log10(uop_312.astype('float64')) # shape=(11, 11)
bop_317 = relay.not_equal(uop_314.astype('bool'), relay.reshape(bop_277.astype('bool'), relay.shape_of(uop_314))) # shape=(11, 11)
bop_321 = relay.add(bop_317.astype('uint8'), relay.reshape(bop_277.astype('uint8'), relay.shape_of(bop_317))) # shape=(11, 11)
output = relay.Tuple([bop_281,bop_288,call_297,const_298,bop_301,bop_307,bop_321,])
output2 = relay.Tuple([bop_281,bop_288,call_299,const_298,bop_301,bop_307,bop_321,])
func_324 = relay.Function([var_272,var_273,var_300,], output)
mod['func_324'] = func_324
mod = relay.transform.InferType()(mod)
var_325 = relay.var("var_325", dtype = "bool", shape = (11, 11))#candidate|325|(11, 11)|var|bool
var_326 = relay.var("var_326", dtype = "bool", shape = (11, 11))#candidate|326|(11, 11)|var|bool
var_327 = relay.var("var_327", dtype = "float32", shape = (11, 11))#candidate|327|(11, 11)|var|float32
output = func_324(var_325,var_326,var_327,)
func_328 = relay.Function([var_325,var_326,var_327,], output)
mutated_mod['func_328'] = func_328
mutated_mod = relay.transform.InferType()(mutated_mod)
const_353 = relay.const([[-8.172285,-3.491051,2.364356,-8.809464,5.900590,-6.002306,7.420664,0.302562,-3.371437,7.268429,9.654008,-3.777742,2.881740,0.297212,4.062135],[-3.378922,-6.129182,3.958426,-9.494450,-1.978492,5.951680,2.518546,1.742108,0.907888,-8.741617,-7.794993,-1.453517,-9.475417,-1.090737,5.024934],[5.584927,7.355987,-9.004070,2.096616,-6.243703,-7.311304,8.576812,9.707171,-9.513366,-2.724047,4.362430,-8.908893,9.314819,7.830873,-8.747957],[7.513816,-1.741892,-5.003845,-9.798987,-9.958358,3.543218,-3.758486,-3.928601,5.637296,-2.918160,0.989072,-8.370026,-8.475393,-9.527618,1.826338],[-3.898826,-0.661228,-9.918461,2.261553,1.648206,1.637707,4.588084,2.243204,8.715714,-8.910794,-4.374629,7.531440,-1.962361,-5.243678,5.168145],[0.406487,-7.426653,5.140576,-8.012694,7.179353,-4.817796,-5.700584,3.697643,4.792886,-0.961049,-6.768174,7.102996,5.803764,3.178703,-5.501315],[-4.410224,-9.640676,-4.567031,6.653961,9.729006,-5.132747,9.990111,0.871278,-7.056292,-2.849928,-6.195407,1.048845,-5.736142,0.526012,-0.659710],[7.743215,-5.566139,2.793014,9.708081,7.292753,-3.546844,-1.422041,7.074348,-6.986096,-9.291360,-2.243170,6.061210,-0.947713,8.528413,-2.657524],[5.757276,-9.422856,0.730286,-9.166634,-4.161616,0.318630,4.791103,-9.590541,1.336193,-7.809645,5.527215,5.009122,-0.728936,1.222455,1.861641]], dtype = "float32")#candidate|353|(9, 15)|const|float32
var_354 = relay.var("var_354", dtype = "float32", shape = (9, 15))#candidate|354|(9, 15)|var|float32
bop_355 = relay.power(const_353.astype('float32'), relay.reshape(var_354.astype('float32'), relay.shape_of(const_353))) # shape=(9, 15)
var_360 = relay.var("var_360", dtype = "float32", shape = (9, 15))#candidate|360|(9, 15)|var|float32
bop_361 = relay.bitwise_xor(bop_355.astype('uint8'), relay.reshape(var_360.astype('uint8'), relay.shape_of(bop_355))) # shape=(9, 15)
output = relay.Tuple([bop_361,])
output2 = relay.Tuple([bop_361,])
func_366 = relay.Function([var_354,var_360,], output)
mod['func_366'] = func_366
mod = relay.transform.InferType()(mod)
mutated_mod['func_366'] = func_366
mutated_mod = relay.transform.InferType()(mutated_mod)
func_366_call = mutated_mod.get_global_var('func_366')
var_368 = relay.var("var_368", dtype = "float32", shape = (9, 15))#candidate|368|(9, 15)|var|float32
var_369 = relay.var("var_369", dtype = "float32", shape = (9, 15))#candidate|369|(9, 15)|var|float32
call_367 = func_366_call(var_368,var_369,)
output = call_367
func_370 = relay.Function([var_368,var_369,], output)
mutated_mod['func_370'] = func_370
mutated_mod = relay.transform.InferType()(mutated_mod)
func_173_call = mod.get_global_var('func_173')
func_174_call = mutated_mod.get_global_var('func_174')
call_385 = relay.TupleGetItem(func_173_call(), 0)
call_386 = relay.TupleGetItem(func_174_call(), 0)
var_394 = relay.var("var_394", dtype = "bool", shape = (8,))#candidate|394|(8,)|var|bool
bop_395 = relay.less(call_385.astype('bool'), relay.reshape(var_394.astype('bool'), relay.shape_of(call_385))) # shape=(8,)
bop_398 = relay.less(call_386.astype('bool'), relay.reshape(var_394.astype('bool'), relay.shape_of(call_386))) # shape=(8,)
uop_400 = relay.atanh(call_385.astype('float32')) # shape=(8,)
uop_402 = relay.atanh(call_386.astype('float32')) # shape=(8,)
bop_403 = relay.less_equal(var_394.astype('bool'), relay.reshape(bop_395.astype('bool'), relay.shape_of(var_394))) # shape=(8,)
bop_406 = relay.less_equal(var_394.astype('bool'), relay.reshape(bop_398.astype('bool'), relay.shape_of(var_394))) # shape=(8,)
bop_409 = relay.equal(uop_400.astype('bool'), relay.reshape(var_394.astype('bool'), relay.shape_of(uop_400))) # shape=(8,)
bop_412 = relay.equal(uop_402.astype('bool'), relay.reshape(var_394.astype('bool'), relay.shape_of(uop_402))) # shape=(8,)
uop_416 = relay.asinh(bop_403.astype('float32')) # shape=(8,)
uop_418 = relay.asinh(bop_406.astype('float32')) # shape=(8,)
output = relay.Tuple([bop_409,uop_416,])
output2 = relay.Tuple([bop_412,uop_418,])
func_419 = relay.Function([var_394,], output)
mod['func_419'] = func_419
mod = relay.transform.InferType()(mod)
var_420 = relay.var("var_420", dtype = "bool", shape = (8,))#candidate|420|(8,)|var|bool
output = func_419(var_420)
func_421 = relay.Function([var_420], output)
mutated_mod['func_421'] = func_421
mutated_mod = relay.transform.InferType()(mutated_mod)
var_428 = relay.var("var_428", dtype = "int32", shape = (4, 9))#candidate|428|(4, 9)|var|int32
const_429 = relay.const([[-5,1,5,-4,4,-3,8,-3,9],[10,-10,-10,4,5,1,9,6,9],[10,10,-5,10,-8,6,10,7,10],[-5,3,-8,1,-2,7,-6,-5,10]], dtype = "int32")#candidate|429|(4, 9)|const|int32
bop_430 = relay.less(var_428.astype('bool'), relay.reshape(const_429.astype('bool'), relay.shape_of(var_428))) # shape=(4, 9)
uop_433 = relay.acos(bop_430.astype('float64')) # shape=(4, 9)
uop_436 = relay.asinh(uop_433.astype('float64')) # shape=(4, 9)
bop_439 = relay.less_equal(uop_436.astype('bool'), relay.reshape(uop_433.astype('bool'), relay.shape_of(uop_436))) # shape=(4, 9)
bop_442 = relay.bitwise_or(uop_433.astype('int64'), relay.reshape(var_428.astype('int64'), relay.shape_of(uop_433))) # shape=(4, 9)
output = relay.Tuple([bop_439,bop_442,])
output2 = relay.Tuple([bop_439,bop_442,])
func_445 = relay.Function([var_428,], output)
mod['func_445'] = func_445
mod = relay.transform.InferType()(mod)
mutated_mod['func_445'] = func_445
mutated_mod = relay.transform.InferType()(mutated_mod)
var_446 = relay.var("var_446", dtype = "int32", shape = (4, 9))#candidate|446|(4, 9)|var|int32
func_445_call = mutated_mod.get_global_var('func_445')
call_447 = func_445_call(var_446)
output = call_447
func_448 = relay.Function([var_446], output)
mutated_mod['func_448'] = func_448
mutated_mod = relay.transform.InferType()(mutated_mod)
var_462 = relay.var("var_462", dtype = "uint64", shape = ())#candidate|462|()|var|uint64
const_463 = relay.const([-7,-8,-6,-9,7,-1,4,-4,-6,-6,-10,2], dtype = "uint64")#candidate|463|(12,)|const|uint64
bop_464 = relay.greater_equal(var_462.astype('bool'), const_463.astype('bool')) # shape=(12,)
bop_471 = relay.divide(bop_464.astype('float32'), var_462.astype('float32')) # shape=(12,)
uop_474 = relay.sqrt(const_463.astype('float32')) # shape=(12,)
const_480 = relay.const([0.199554,-6.292062,-7.640344,-2.810724,6.611265,0.097830,2.129589,6.490090,-5.691449,-2.239754,3.208224,-7.473594], dtype = "float32")#candidate|480|(12,)|const|float32
bop_481 = relay.bitwise_xor(uop_474.astype('uint32'), relay.reshape(const_480.astype('uint32'), relay.shape_of(uop_474))) # shape=(12,)
uop_485 = relay.atanh(bop_481.astype('float32')) # shape=(12,)
bop_488 = relay.equal(uop_485.astype('bool'), relay.reshape(const_480.astype('bool'), relay.shape_of(uop_485))) # shape=(12,)
bop_491 = relay.minimum(bop_488.astype('int16'), relay.reshape(const_480.astype('int16'), relay.shape_of(bop_488))) # shape=(12,)
const_494 = relay.const([-0.712896,2.360393,6.814092,9.355587,0.506862,5.526487,0.067711,-7.149100,-7.609574,-7.544442,1.167883,6.301981], dtype = "float32")#candidate|494|(12,)|const|float32
bop_495 = relay.logical_or(uop_485.astype('bool'), relay.reshape(const_494.astype('bool'), relay.shape_of(uop_485))) # shape=(12,)
output = relay.Tuple([bop_471,bop_491,bop_495,])
output2 = relay.Tuple([bop_471,bop_491,bop_495,])
func_499 = relay.Function([var_462,], output)
mod['func_499'] = func_499
mod = relay.transform.InferType()(mod)
mutated_mod['func_499'] = func_499
mutated_mod = relay.transform.InferType()(mutated_mod)
var_500 = relay.var("var_500", dtype = "uint64", shape = ())#candidate|500|()|var|uint64
func_499_call = mutated_mod.get_global_var('func_499')
call_501 = func_499_call(var_500)
output = call_501
func_502 = relay.Function([var_500], output)
mutated_mod['func_502'] = func_502
mutated_mod = relay.transform.InferType()(mutated_mod)
func_173_call = mod.get_global_var('func_173')
func_174_call = mutated_mod.get_global_var('func_174')
call_516 = relay.TupleGetItem(func_173_call(), 0)
call_517 = relay.TupleGetItem(func_174_call(), 0)
output = relay.Tuple([call_516,])
output2 = relay.Tuple([call_517,])
func_521 = relay.Function([], output)
mod['func_521'] = func_521
mod = relay.transform.InferType()(mod)
output = func_521()
func_522 = relay.Function([], output)
mutated_mod['func_522'] = func_522
mutated_mod = relay.transform.InferType()(mutated_mod)
func_521_call = mod.get_global_var('func_521')
func_522_call = mutated_mod.get_global_var('func_522')
call_531 = relay.TupleGetItem(func_521_call(), 0)
call_532 = relay.TupleGetItem(func_522_call(), 0)
var_544 = relay.var("var_544", dtype = "bool", shape = (8,))#candidate|544|(8,)|var|bool
bop_545 = relay.logical_and(call_531.astype('bool'), relay.reshape(var_544.astype('bool'), relay.shape_of(call_531))) # shape=(8,)
bop_548 = relay.logical_and(call_532.astype('bool'), relay.reshape(var_544.astype('bool'), relay.shape_of(call_532))) # shape=(8,)
bop_550 = relay.greater_equal(call_531.astype('bool'), relay.reshape(var_544.astype('bool'), relay.shape_of(call_531))) # shape=(8,)
bop_553 = relay.greater_equal(call_532.astype('bool'), relay.reshape(var_544.astype('bool'), relay.shape_of(call_532))) # shape=(8,)
bop_557 = relay.subtract(var_544.astype('uint64'), relay.reshape(bop_545.astype('uint64'), relay.shape_of(var_544))) # shape=(8,)
bop_560 = relay.subtract(var_544.astype('uint64'), relay.reshape(bop_548.astype('uint64'), relay.shape_of(var_544))) # shape=(8,)
output = relay.Tuple([bop_550,bop_557,])
output2 = relay.Tuple([bop_553,bop_560,])
func_561 = relay.Function([var_544,], output)
mod['func_561'] = func_561
mod = relay.transform.InferType()(mod)
mutated_mod['func_561'] = func_561
mutated_mod = relay.transform.InferType()(mutated_mod)
var_562 = relay.var("var_562", dtype = "bool", shape = (8,))#candidate|562|(8,)|var|bool
func_561_call = mutated_mod.get_global_var('func_561')
call_563 = func_561_call(var_562)
output = call_563
func_564 = relay.Function([var_562], output)
mutated_mod['func_564'] = func_564
mutated_mod = relay.transform.InferType()(mutated_mod)
var_571 = relay.var("var_571", dtype = "bool", shape = (7, 14))#candidate|571|(7, 14)|var|bool
const_572 = relay.const([[False,True,False,True,False,False,True,False,True,True,False,False,True,True],[True,False,False,False,True,False,False,True,False,True,False,True,False,False],[False,True,True,True,False,True,False,False,True,True,True,False,False,True],[False,False,False,False,False,False,False,True,False,True,False,False,False,False],[False,False,True,False,True,False,True,True,True,True,False,False,False,True],[True,True,False,True,True,False,True,True,False,False,True,False,True,True],[True,True,False,True,False,True,True,True,True,True,True,False,False,True]], dtype = "bool")#candidate|572|(7, 14)|const|bool
bop_573 = relay.logical_and(var_571.astype('bool'), relay.reshape(const_572.astype('bool'), relay.shape_of(var_571))) # shape=(7, 14)
uop_580 = relay.atan(bop_573.astype('float32')) # shape=(7, 14)
var_582 = relay.var("var_582", dtype = "float32", shape = (7, 14))#candidate|582|(7, 14)|var|float32
bop_583 = relay.not_equal(uop_580.astype('bool'), relay.reshape(var_582.astype('bool'), relay.shape_of(uop_580))) # shape=(7, 14)
bop_589 = relay.right_shift(bop_583.astype('uint8'), relay.reshape(uop_580.astype('uint8'), relay.shape_of(bop_583))) # shape=(7, 14)
func_366_call = mod.get_global_var('func_366')
func_370_call = mutated_mod.get_global_var('func_370')
var_593 = relay.var("var_593", dtype = "float32", shape = (1, 135))#candidate|593|(1, 135)|var|float32
call_592 = relay.TupleGetItem(func_366_call(relay.reshape(var_593.astype('float32'), [9, 15]), relay.reshape(var_593.astype('float32'), [9, 15]), ), 0)
call_594 = relay.TupleGetItem(func_370_call(relay.reshape(var_593.astype('float32'), [9, 15]), relay.reshape(var_593.astype('float32'), [9, 15]), ), 0)
uop_595 = relay.erf(bop_583.astype('float64')) # shape=(7, 14)
bop_601 = relay.floor_divide(uop_595.astype('float64'), relay.reshape(bop_573.astype('float64'), relay.shape_of(uop_595))) # shape=(7, 14)
bop_604 = relay.floor_divide(uop_595.astype('float64'), relay.reshape(bop_601.astype('float64'), relay.shape_of(uop_595))) # shape=(7, 14)
bop_608 = relay.mod(var_571.astype('float64'), relay.reshape(bop_589.astype('float64'), relay.shape_of(var_571))) # shape=(7, 14)
bop_611 = relay.bitwise_and(bop_583.astype('int32'), relay.reshape(bop_601.astype('int32'), relay.shape_of(bop_583))) # shape=(7, 14)
bop_619 = relay.power(bop_601.astype('float32'), relay.reshape(bop_573.astype('float32'), relay.shape_of(bop_601))) # shape=(7, 14)
uop_623 = relay.atanh(bop_604.astype('float32')) # shape=(7, 14)
uop_626 = relay.rsqrt(uop_623.astype('float32')) # shape=(7, 14)
bop_630 = relay.greater(bop_583.astype('bool'), relay.reshape(bop_573.astype('bool'), relay.shape_of(bop_583))) # shape=(7, 14)
bop_635 = relay.floor_mod(uop_626.astype('float64'), relay.reshape(bop_601.astype('float64'), relay.shape_of(uop_626))) # shape=(7, 14)
uop_642 = relay.sqrt(bop_608.astype('float64')) # shape=(7, 14)
bop_650 = relay.mod(bop_635.astype('float64'), relay.reshape(bop_608.astype('float64'), relay.shape_of(bop_635))) # shape=(7, 14)
uop_656 = relay.tan(bop_611.astype('float32')) # shape=(7, 14)
bop_658 = relay.not_equal(bop_635.astype('bool'), relay.reshape(uop_626.astype('bool'), relay.shape_of(bop_635))) # shape=(7, 14)
uop_665 = relay.asinh(bop_635.astype('float32')) # shape=(7, 14)
uop_668 = relay.log(uop_665.astype('float64')) # shape=(7, 14)
var_673 = relay.var("var_673", dtype = "float64", shape = (7, 14))#candidate|673|(7, 14)|var|float64
bop_674 = relay.logical_or(uop_668.astype('bool'), relay.reshape(var_673.astype('bool'), relay.shape_of(uop_668))) # shape=(7, 14)
func_264_call = mod.get_global_var('func_264')
func_267_call = mutated_mod.get_global_var('func_267')
var_678 = relay.var("var_678", dtype = "float32", shape = (5, 1))#candidate|678|(5, 1)|var|float32
call_677 = relay.TupleGetItem(func_264_call(relay.reshape(var_678.astype('float32'), [5,])), 2)
call_679 = relay.TupleGetItem(func_267_call(relay.reshape(var_678.astype('float32'), [5,])), 2)
bop_680 = relay.bitwise_xor(bop_674.astype('int64'), relay.reshape(uop_668.astype('int64'), relay.shape_of(bop_674))) # shape=(7, 14)
uop_683 = relay.cosh(bop_680.astype('float32')) # shape=(7, 14)
output = relay.Tuple([call_592,var_593,bop_619,bop_630,uop_642,bop_650,uop_656,bop_658,call_677,var_678,uop_683,])
output2 = relay.Tuple([call_594,var_593,bop_619,bop_630,uop_642,bop_650,uop_656,bop_658,call_679,var_678,uop_683,])
func_685 = relay.Function([var_571,var_582,var_593,var_673,var_678,], output)
mod['func_685'] = func_685
mod = relay.transform.InferType()(mod)
mutated_mod['func_685'] = func_685
mutated_mod = relay.transform.InferType()(mutated_mod)
func_685_call = mutated_mod.get_global_var('func_685')
var_687 = relay.var("var_687", dtype = "bool", shape = (7, 14))#candidate|687|(7, 14)|var|bool
var_688 = relay.var("var_688", dtype = "float32", shape = (7, 14))#candidate|688|(7, 14)|var|float32
var_689 = relay.var("var_689", dtype = "float32", shape = (1, 135))#candidate|689|(1, 135)|var|float32
var_690 = relay.var("var_690", dtype = "float64", shape = (7, 14))#candidate|690|(7, 14)|var|float64
var_691 = relay.var("var_691", dtype = "float32", shape = (5, 1))#candidate|691|(5, 1)|var|float32
call_686 = func_685_call(var_687,var_688,var_689,var_690,var_691,)
output = call_686
func_692 = relay.Function([var_687,var_688,var_689,var_690,var_691,], output)
mutated_mod['func_692'] = func_692
mutated_mod = relay.transform.InferType()(mutated_mod)
func_173_call = mod.get_global_var('func_173')
func_174_call = mutated_mod.get_global_var('func_174')
call_697 = relay.TupleGetItem(func_173_call(), 0)
call_698 = relay.TupleGetItem(func_174_call(), 0)
var_705 = relay.var("var_705", dtype = "bool", shape = (8,))#candidate|705|(8,)|var|bool
bop_706 = relay.logical_xor(call_697.astype('uint16'), relay.reshape(var_705.astype('uint16'), relay.shape_of(call_697))) # shape=(8,)
bop_709 = relay.logical_xor(call_698.astype('uint16'), relay.reshape(var_705.astype('uint16'), relay.shape_of(call_698))) # shape=(8,)
var_711 = relay.var("var_711", dtype = "uint16", shape = (8,))#candidate|711|(8,)|var|uint16
bop_712 = relay.multiply(bop_706.astype('uint8'), relay.reshape(var_711.astype('uint8'), relay.shape_of(bop_706))) # shape=(8,)
bop_715 = relay.multiply(bop_709.astype('uint8'), relay.reshape(var_711.astype('uint8'), relay.shape_of(bop_709))) # shape=(8,)
bop_716 = relay.right_shift(call_697.astype('int16'), relay.reshape(bop_712.astype('int16'), relay.shape_of(call_697))) # shape=(8,)
bop_719 = relay.right_shift(call_698.astype('int16'), relay.reshape(bop_715.astype('int16'), relay.shape_of(call_698))) # shape=(8,)
uop_723 = relay.cos(bop_712.astype('float32')) # shape=(8,)
uop_725 = relay.cos(bop_715.astype('float32')) # shape=(8,)
uop_727 = relay.atan(uop_723.astype('float32')) # shape=(8,)
uop_729 = relay.atan(uop_725.astype('float32')) # shape=(8,)
bop_733 = relay.multiply(uop_723.astype('int64'), relay.reshape(bop_712.astype('int64'), relay.shape_of(uop_723))) # shape=(8,)
bop_736 = relay.multiply(uop_725.astype('int64'), relay.reshape(bop_715.astype('int64'), relay.shape_of(uop_725))) # shape=(8,)
func_419_call = mod.get_global_var('func_419')
func_421_call = mutated_mod.get_global_var('func_421')
call_738 = relay.TupleGetItem(func_419_call(relay.reshape(bop_716.astype('bool'), [8,])), 0)
call_739 = relay.TupleGetItem(func_421_call(relay.reshape(bop_716.astype('bool'), [8,])), 0)
output = relay.Tuple([bop_716,uop_727,bop_733,call_738,])
output2 = relay.Tuple([bop_719,uop_729,bop_736,call_739,])
func_742 = relay.Function([var_705,var_711,], output)
mod['func_742'] = func_742
mod = relay.transform.InferType()(mod)
var_743 = relay.var("var_743", dtype = "bool", shape = (8,))#candidate|743|(8,)|var|bool
var_744 = relay.var("var_744", dtype = "uint16", shape = (8,))#candidate|744|(8,)|var|uint16
output = func_742(var_743,var_744,)
func_745 = relay.Function([var_743,var_744,], output)
mutated_mod['func_745'] = func_745
mutated_mod = relay.transform.InferType()(mutated_mod)
func_521_call = mod.get_global_var('func_521')
func_522_call = mutated_mod.get_global_var('func_522')
call_765 = relay.TupleGetItem(func_521_call(), 0)
call_766 = relay.TupleGetItem(func_522_call(), 0)
const_768 = relay.const([True,False,False,False,True,False,True,True], dtype = "bool")#candidate|768|(8,)|const|bool
bop_769 = relay.not_equal(call_765.astype('bool'), relay.reshape(const_768.astype('bool'), relay.shape_of(call_765))) # shape=(8,)
bop_772 = relay.not_equal(call_766.astype('bool'), relay.reshape(const_768.astype('bool'), relay.shape_of(call_766))) # shape=(8,)
bop_774 = relay.floor_divide(bop_769.astype('float64'), relay.reshape(const_768.astype('float64'), relay.shape_of(bop_769))) # shape=(8,)
bop_777 = relay.floor_divide(bop_772.astype('float64'), relay.reshape(const_768.astype('float64'), relay.shape_of(bop_772))) # shape=(8,)
uop_780 = relay.rsqrt(bop_774.astype('float64')) # shape=(8,)
uop_782 = relay.rsqrt(bop_777.astype('float64')) # shape=(8,)
bop_783 = relay.less(uop_780.astype('bool'), relay.reshape(call_765.astype('bool'), relay.shape_of(uop_780))) # shape=(8,)
bop_786 = relay.less(uop_782.astype('bool'), relay.reshape(call_766.astype('bool'), relay.shape_of(uop_782))) # shape=(8,)
output = bop_783
output2 = bop_786
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