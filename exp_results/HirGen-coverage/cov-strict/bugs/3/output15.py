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
var_14 = relay.var("var_14", dtype = "float32", shape = (10, 16))#candidate|14|(10, 16)|var|float32
uop_15 = relay.log(var_14.astype('float32')) # shape=(10, 16)
output = uop_15
output2 = uop_15
func_20 = relay.Function([var_14,], output)
mod['func_20'] = func_20
mod = relay.transform.InferType()(mod)
var_21 = relay.var("var_21", dtype = "float32", shape = (10, 16))#candidate|21|(10, 16)|var|float32
output = func_20(var_21)
func_22 = relay.Function([var_21], output)
mutated_mod['func_22'] = func_22
mutated_mod = relay.transform.InferType()(mutated_mod)
const_30 = relay.const([[6,3,7,8,3,4,8,-1,-8,-3,6,-9,-3],[8,6,8,-1,-2,9,10,10,6,-6,4,-6,2],[4,-6,2,1,-3,-1,-9,3,-1,-10,9,-10,2],[10,-2,3,-6,5,3,5,4,-2,-9,1,-2,-10],[8,7,-5,-10,-1,-2,-2,-4,-4,2,-7,3,1],[-3,-6,5,4,-3,-1,5,-6,1,-4,2,8,-8],[-10,8,-4,6,5,-3,5,7,9,-10,-8,4,9]], dtype = "int32")#candidate|30|(7, 13)|const|int32
var_31 = relay.var("var_31", dtype = "int32", shape = (7, 13))#candidate|31|(7, 13)|var|int32
bop_32 = relay.bitwise_or(const_30.astype('int32'), relay.reshape(var_31.astype('int32'), relay.shape_of(const_30))) # shape=(7, 13)
bop_41 = relay.add(const_30.astype('int16'), relay.reshape(bop_32.astype('int16'), relay.shape_of(const_30))) # shape=(7, 13)
output = bop_41
output2 = bop_41
func_44 = relay.Function([var_31,], output)
mod['func_44'] = func_44
mod = relay.transform.InferType()(mod)
var_45 = relay.var("var_45", dtype = "int32", shape = (7, 13))#candidate|45|(7, 13)|var|int32
output = func_44(var_45)
func_46 = relay.Function([var_45], output)
mutated_mod['func_46'] = func_46
mutated_mod = relay.transform.InferType()(mutated_mod)
var_94 = relay.var("var_94", dtype = "uint8", shape = (12, 3))#candidate|94|(12, 3)|var|uint8
var_95 = relay.var("var_95", dtype = "uint8", shape = (12, 3))#candidate|95|(12, 3)|var|uint8
bop_96 = relay.bitwise_or(var_94.astype('uint8'), relay.reshape(var_95.astype('uint8'), relay.shape_of(var_94))) # shape=(12, 3)
bop_101 = relay.multiply(var_94.astype('uint8'), relay.reshape(bop_96.astype('uint8'), relay.shape_of(var_94))) # shape=(12, 3)
uop_112 = relay.sqrt(bop_96.astype('float32')) # shape=(12, 3)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
var_120 = relay.var("var_120", dtype = "int32", shape = (91,))#candidate|120|(91,)|var|int32
call_119 = func_44_call(relay.reshape(var_120.astype('int32'), [7, 13]))
call_121 = func_44_call(relay.reshape(var_120.astype('int32'), [7, 13]))
bop_122 = relay.minimum(var_95.astype('uint8'), relay.reshape(uop_112.astype('uint8'), relay.shape_of(var_95))) # shape=(12, 3)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
call_127 = func_44_call(relay.reshape(var_120.astype('int32'), [7, 13]))
call_128 = func_44_call(relay.reshape(var_120.astype('int32'), [7, 13]))
bop_130 = relay.divide(uop_112.astype('float64'), relay.reshape(bop_122.astype('float64'), relay.shape_of(uop_112))) # shape=(12, 3)
bop_135 = relay.logical_xor(call_127.astype('uint8'), relay.reshape(var_120.astype('uint8'), relay.shape_of(call_127))) # shape=(7, 13)
bop_138 = relay.logical_xor(call_128.astype('uint8'), relay.reshape(var_120.astype('uint8'), relay.shape_of(call_128))) # shape=(7, 13)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
var_140 = relay.var("var_140", dtype = "float32", shape = (2, 80))#candidate|140|(2, 80)|var|float32
call_139 = func_20_call(relay.reshape(var_140.astype('float32'), [10, 16]))
call_141 = func_20_call(relay.reshape(var_140.astype('float32'), [10, 16]))
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_146 = func_20_call(relay.reshape(var_140.astype('float32'), [10, 16]))
call_147 = func_20_call(relay.reshape(var_140.astype('float32'), [10, 16]))
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_151 = func_20_call(relay.reshape(call_139.astype('float32'), [10, 16]))
call_152 = func_20_call(relay.reshape(call_139.astype('float32'), [10, 16]))
output = relay.Tuple([bop_101,call_119,bop_130,bop_135,call_139,var_140,call_146,call_151,])
output2 = relay.Tuple([bop_101,call_121,bop_130,bop_138,call_141,var_140,call_147,call_152,])
func_158 = relay.Function([var_94,var_95,var_120,var_140,], output)
mod['func_158'] = func_158
mod = relay.transform.InferType()(mod)
var_159 = relay.var("var_159", dtype = "uint8", shape = (12, 3))#candidate|159|(12, 3)|var|uint8
var_160 = relay.var("var_160", dtype = "uint8", shape = (12, 3))#candidate|160|(12, 3)|var|uint8
var_161 = relay.var("var_161", dtype = "int32", shape = (91,))#candidate|161|(91,)|var|int32
var_162 = relay.var("var_162", dtype = "float32", shape = (2, 80))#candidate|162|(2, 80)|var|float32
output = func_158(var_159,var_160,var_161,var_162,)
func_163 = relay.Function([var_159,var_160,var_161,var_162,], output)
mutated_mod['func_163'] = func_163
mutated_mod = relay.transform.InferType()(mutated_mod)
var_193 = relay.var("var_193", dtype = "float32", shape = (6, 13, 4))#candidate|193|(6, 13, 4)|var|float32
uop_194 = relay.acos(var_193.astype('float32')) # shape=(6, 13, 4)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
const_198 = relay.const([[6.212312],[-7.754475],[-9.013621],[4.097305],[-9.239899],[9.927140],[-6.039869],[6.575813],[5.774025],[4.630207],[-8.834431],[-1.470708],[0.264142],[9.281407],[1.101090],[6.659704],[-9.211099],[-7.140951],[-4.605304],[-6.961433],[1.318213],[-3.479654],[2.976352],[-6.854904],[9.025783],[-8.246245],[9.118900],[-6.303382],[-6.307960],[-1.338978],[-2.593580],[7.231218],[4.375702],[6.980672],[6.124741],[7.245341],[1.679978],[-4.734513],[7.451118],[7.977979],[-1.015513],[-7.840930],[9.552629],[-1.031082],[5.497648],[9.412824],[4.200343],[-1.319479],[7.246049],[-3.569695],[3.752042],[6.030473],[-4.954167],[-9.257545],[-3.853705],[-2.186584],[1.192491],[-8.902332],[0.304997],[-6.781926],[4.606162],[-1.337173],[-4.315631],[8.070479],[-6.275686],[-7.478507],[8.955214],[-2.830706],[9.391725],[1.705455],[-3.639414],[8.071214],[3.807098],[-9.591586],[-2.700286],[0.377184],[8.553258],[-2.340232],[-9.917816],[6.763960],[-3.419741],[0.818334],[5.202198],[0.288687],[-3.486326],[-0.143162],[-4.074579],[-0.126478],[1.562646],[-6.096989],[6.849350],[8.958273],[5.056529],[4.638684],[3.606505],[-5.049053],[-2.204972],[-3.953488],[5.971811],[8.572143],[-4.554229],[-3.870194],[2.504031],[-8.298028],[-8.750159],[-0.327144],[-5.185607],[-9.717179],[-8.087016],[-1.783524],[-7.959959],[0.079531],[-6.029156],[-8.598846],[4.273276],[6.934637],[1.129341],[-2.333834],[-0.885893],[4.267432],[1.795814],[-9.422691],[1.805093],[2.963548],[-7.134947],[0.645848],[-4.989193],[-3.936285],[-9.937097],[-3.870997],[-9.805694],[-6.834329],[7.270052],[-0.069067],[7.903873],[-3.395626],[8.109498],[2.480855],[4.680096],[-4.829864],[-8.593421],[-6.044330],[-3.070297],[7.785751],[5.396633],[-8.932613],[9.340221],[4.485041],[2.127418],[-7.290932],[2.923925],[9.303586],[5.667588],[5.562243],[5.309493],[2.354278],[-3.388351],[-6.385674],[-8.329825],[2.323616]], dtype = "float32")#candidate|198|(160, 1)|const|float32
call_197 = func_20_call(relay.reshape(const_198.astype('float32'), [10, 16]))
call_199 = func_20_call(relay.reshape(const_198.astype('float32'), [10, 16]))
uop_206 = relay.acos(const_198.astype('float32')) # shape=(160, 1)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_218 = func_20_call(relay.reshape(call_197.astype('float32'), [10, 16]))
call_219 = func_20_call(relay.reshape(call_197.astype('float32'), [10, 16]))
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
const_224 = relay.const([-1,-5,6,1,7,-1,2,9,-10,-2,-5,-4,6,9,9,9,8,6,10,-7,9,-4,-6,9,-1,4,3,6,-9,3,-1,-7,8,1,3,5,-6,3,3,9,-2,-9,10,-4,2,-9,2,10,-1,10,-10,6,-7,7,10,3,10,6,-9,1,10,2,7,-3,-6,-4,1,2,10,7,7,1,4,-1,10,4,-3,-3,10,-1,-1,1,-7,-8,-4,5,7,-3,3,-5,6], dtype = "int32")#candidate|224|(91,)|const|int32
call_223 = func_44_call(relay.reshape(const_224.astype('int32'), [7, 13]))
call_225 = func_44_call(relay.reshape(const_224.astype('int32'), [7, 13]))
output = relay.Tuple([uop_194,call_197,uop_206,call_218,call_223,const_224,])
output2 = relay.Tuple([uop_194,call_199,uop_206,call_219,call_225,const_224,])
func_232 = relay.Function([var_193,], output)
mod['func_232'] = func_232
mod = relay.transform.InferType()(mod)
var_233 = relay.var("var_233", dtype = "float32", shape = (6, 13, 4))#candidate|233|(6, 13, 4)|var|float32
output = func_232(var_233)
func_234 = relay.Function([var_233], output)
mutated_mod['func_234'] = func_234
mutated_mod = relay.transform.InferType()(mutated_mod)
var_329 = relay.var("var_329", dtype = "uint16", shape = (13, 2, 3))#candidate|329|(13, 2, 3)|var|uint16
var_330 = relay.var("var_330", dtype = "uint16", shape = (13, 2, 3))#candidate|330|(13, 2, 3)|var|uint16
bop_331 = relay.less(var_329.astype('bool'), relay.reshape(var_330.astype('bool'), relay.shape_of(var_329))) # shape=(13, 2, 3)
bop_334 = relay.equal(var_329.astype('bool'), relay.reshape(var_330.astype('bool'), relay.shape_of(var_329))) # shape=(13, 2, 3)
var_340 = relay.var("var_340", dtype = "bool", shape = (13, 2, 3))#candidate|340|(13, 2, 3)|var|bool
bop_341 = relay.logical_and(bop_334.astype('bool'), relay.reshape(var_340.astype('bool'), relay.shape_of(bop_334))) # shape=(13, 2, 3)
bop_345 = relay.multiply(var_330.astype('int32'), relay.reshape(bop_341.astype('int32'), relay.shape_of(var_330))) # shape=(13, 2, 3)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
const_349 = relay.const([8,-1,-9,-9,-10,-3,1,-10,-3,6,5,-6,5,-3,1,-4,-9,-1,1,3,1,6,4,-6,10,10,3,-8,-6,-3,-8,3,-2,-6,-6,-6,6,8,10,8,7,-5,6,1,6,-6,-1,-6,5,7,2,10,-5,-6,4,5,2,2,-5,9,-6,-10,3,-1,-7,-4,2,-4,4,-4,-6,8,8,7,4,9,-8,-9,4,-1,-10,7,-1,-1,-3,-1,-3,-8,-10,6,-8], dtype = "int32")#candidate|349|(91,)|const|int32
call_348 = func_44_call(relay.reshape(const_349.astype('int32'), [7, 13]))
call_350 = func_44_call(relay.reshape(const_349.astype('int32'), [7, 13]))
bop_380 = relay.not_equal(bop_345.astype('bool'), relay.reshape(var_330.astype('bool'), relay.shape_of(bop_345))) # shape=(13, 2, 3)
uop_390 = relay.cos(bop_345.astype('float32')) # shape=(13, 2, 3)
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
const_393 = relay.const([4.783493,-1.187819,4.518800,-1.521998,2.924649,-7.705236,-4.216536,4.819445,7.765975,7.498095,-6.246536,-2.329892,-8.461577,-9.683690,-3.062280,4.044659,-9.635442,6.578391,5.817796,-5.494306,-6.674610,8.920490,9.710472,-9.726510,3.021547,-1.148140,5.354308,-2.679192,5.814308,5.386777,9.243869,-3.173516,0.854636,-2.132221,7.277650,7.900663,4.272815,4.702533,9.623614,9.813318,8.030272,-7.454072,9.276298,-3.969812,-7.087957,-5.215376,-6.079166,5.468087,2.463461,-9.719327,-9.795375,-3.490356,-9.789504,-3.925742,6.315677,1.624877,9.918244,-4.987053,-0.902123,-2.067099,-0.975705,2.403404,4.616720,-3.270378,-4.336075,8.136451,-6.976044,-9.168361,0.208222,-0.511132,-7.875691,5.219632,-7.539890,-2.516037,-4.421147,9.797135,-3.773315,2.468667,2.068364,-3.744282,-5.117596,3.757188,0.552364,-6.511865,-2.833354,-3.301652,-6.464956,0.301747,5.561874,8.972599,-1.212306,-8.740541,7.113799,-5.995358,-5.512901,-4.230586,-0.418429,5.467991,0.179250,2.406023,3.310787,6.546039,2.332239,-6.078103,-4.780185,7.258020,-6.006915,7.172248,5.128119,-0.802305,-1.243320,1.206582,-6.151819,7.787122,-1.470901,-1.130215,-6.783439,6.890094,-1.348773,-8.033894,1.681255,2.066490,2.582353,-2.656707,-6.787790,-9.705294,8.757776,2.507263,6.914571,-0.299562,4.433303,1.026458,-7.189247,-6.572252,-9.308273,2.687777,-5.902655,0.267022,-1.600573,-9.910587,-8.375601,3.519624,-3.153984,-9.083766,7.753093,5.147565,1.853270,-1.732962,6.095648,-4.923657,4.451200,-2.419548,-7.926460,-9.229671,8.047871,6.540765,5.606806,2.215590,8.731104,9.324647,-1.337067,-6.394239,8.657782,6.929112,6.757340,-7.957352,-1.038518,-9.916657,-6.774244,8.887568,3.027573,-6.962502,5.245731,6.428661,9.069382,-3.340015,5.756025,0.048919,3.516278,-6.045498,-4.098203,-9.937024,9.135023,0.079419,0.565966,2.291912,9.349035,2.420857,6.723083,3.027267,-5.212035,5.611526,-5.188367,-9.261450,6.749968,-5.125847,-6.731502,-4.081241,0.883568,9.128493,-0.973299,-4.191607,-4.231955,4.429942,-6.473059,-1.605716,-3.151211,-6.146380,-7.984419,-2.116398,1.723216,6.345160,8.654671,0.841293,0.301698,0.356831,3.755709,2.037650,-4.658490,-2.642344,-9.238969,7.402385,-1.771709,6.710498,-6.798252,-4.001397,8.268426,3.628356,-0.069651,4.928739,-1.323538,0.074374,-3.909559,2.352586,3.640225,6.645885,1.626082,-2.702591,0.607913,-5.774940,-4.713797,4.453050,3.894392,-2.229211,-7.403153,5.617706,7.414026,-3.466270,-8.981945,-9.134467,8.246650,7.574974,-3.593867,-5.762250,1.328559,7.005158,-7.878460,2.712710,-3.631384,2.860265,-2.440920,9.137603,5.228308,2.160994,0.404669,-0.893500,-9.247906,2.314080,0.685895,-9.666514,1.639976,-0.702621,-3.419622,4.418901,-3.101923,0.231431,9.269193,-5.532731,0.165079,-4.541520,6.206229,9.389359,-1.808429,-3.485006,6.476477,5.481721,-8.271249,-5.617683,2.066152,1.921832,-8.340315,-3.672057,2.648841,-6.348133,4.619626,-0.465736,6.421949,-9.858793,-4.256425,-1.308436,6.961081,-2.130391,1.418709,-8.297956,-7.395336,-4.317118,0.107700,3.144504,0.898110,3.382648,-1.277761,7.696546], dtype = "float32")#candidate|393|(312,)|const|float32
call_392 = relay.TupleGetItem(func_232_call(relay.reshape(const_393.astype('float32'), [6, 13, 4])), 1)
call_394 = relay.TupleGetItem(func_234_call(relay.reshape(const_393.astype('float32'), [6, 13, 4])), 1)
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
call_400 = relay.TupleGetItem(func_232_call(relay.reshape(const_393.astype('float32'), [6, 13, 4])), 0)
call_401 = relay.TupleGetItem(func_234_call(relay.reshape(const_393.astype('float32'), [6, 13, 4])), 0)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
call_406 = func_44_call(relay.reshape(const_349.astype('int32'), [7, 13]))
call_407 = func_44_call(relay.reshape(const_349.astype('int32'), [7, 13]))
bop_419 = relay.bitwise_or(uop_390.astype('int8'), relay.reshape(bop_334.astype('int8'), relay.shape_of(uop_390))) # shape=(13, 2, 3)
output = relay.Tuple([bop_331,call_348,const_349,bop_380,call_392,const_393,call_400,call_406,bop_419,])
output2 = relay.Tuple([bop_331,call_350,const_349,bop_380,call_394,const_393,call_401,call_407,bop_419,])
func_422 = relay.Function([var_329,var_330,var_340,], output)
mod['func_422'] = func_422
mod = relay.transform.InferType()(mod)
var_423 = relay.var("var_423", dtype = "uint16", shape = (13, 2, 3))#candidate|423|(13, 2, 3)|var|uint16
var_424 = relay.var("var_424", dtype = "uint16", shape = (13, 2, 3))#candidate|424|(13, 2, 3)|var|uint16
var_425 = relay.var("var_425", dtype = "bool", shape = (13, 2, 3))#candidate|425|(13, 2, 3)|var|bool
output = func_422(var_423,var_424,var_425,)
func_426 = relay.Function([var_423,var_424,var_425,], output)
mutated_mod['func_426'] = func_426
mutated_mod = relay.transform.InferType()(mutated_mod)
var_525 = relay.var("var_525", dtype = "uint32", shape = (13, 8))#candidate|525|(13, 8)|var|uint32
var_526 = relay.var("var_526", dtype = "uint32", shape = (13, 8))#candidate|526|(13, 8)|var|uint32
bop_527 = relay.bitwise_xor(var_525.astype('uint32'), relay.reshape(var_526.astype('uint32'), relay.shape_of(var_525))) # shape=(13, 8)
func_158_call = mod.get_global_var('func_158')
func_163_call = mutated_mod.get_global_var('func_163')
var_533 = relay.var("var_533", dtype = "uint8", shape = (36,))#candidate|533|(36,)|var|uint8
const_534 = relay.const([[4,-9,10,-2,9,-5,-2],[-9,5,5,2,-5,-8,-10],[-7,-8,4,-10,9,5,10],[9,7,5,2,9,7,-10],[3,2,4,-10,9,-5,2],[6,-9,3,5,-9,7,2],[4,-7,-3,8,10,10,-10],[8,4,-3,9,3,4,-10],[2,5,5,4,2,7,-1],[-3,-3,-4,-5,2,-3,-9],[-8,9,2,-7,-7,-4,2],[3,10,6,-7,10,7,9],[8,-10,6,8,5,-2,6]], dtype = "int32")#candidate|534|(13, 7)|const|int32
const_535 = relay.const([[3.561836,-8.539224],[3.162542,-0.776185],[1.417489,3.956892],[-4.824253,-3.565887],[-6.667155,-1.938080],[5.927193,6.948374],[-1.990337,0.224645],[-5.454891,-1.869256],[8.730340,1.175450],[5.510120,5.905821],[9.320381,0.176473],[2.278287,2.481703],[-4.304001,-0.460700],[-5.595215,-5.153368],[4.784343,9.248188],[9.405015,3.588112],[-5.870247,1.132271],[-8.495957,1.141483],[6.879550,6.872868],[-3.103097,-3.631358],[-5.371451,-7.110207],[-0.593017,1.405132],[4.058203,-8.657009],[-3.449974,1.299790],[-6.900117,-3.432018],[2.953832,-7.234717],[-7.924107,4.192387],[-1.743662,6.057313],[2.368657,-8.286223],[3.779419,4.912186],[-5.549895,7.939826],[5.100780,-0.820296],[0.990086,2.784347],[-8.389453,3.056780],[5.890843,9.639983],[-7.602049,0.151392],[5.633172,-1.108256],[5.499802,1.070616],[2.153289,8.483450],[4.902159,-4.216160],[-0.205149,0.704629],[-1.058440,-6.915266],[6.503253,8.000981],[-4.516081,0.114349],[-8.829093,-8.915917],[5.371558,-1.576898],[-8.254898,-7.730511],[1.299087,-8.509947],[-3.239885,-2.853452],[-9.662988,-5.387998],[-4.299716,-5.408993],[-0.446726,-8.678908],[0.843876,6.586963],[-0.957895,2.866567],[6.881904,-3.748895],[-3.635508,-6.588514],[-6.410361,4.698917],[2.806193,9.542723],[-5.574687,8.773499],[9.423951,1.957808],[-9.913063,-1.127395],[-4.203282,-2.537973],[7.605966,-9.691590],[4.423227,-8.803638],[7.830018,0.684165],[4.005557,2.872197],[-6.725813,1.081575],[-3.046737,-1.647440],[8.380216,-9.142641],[-6.931295,5.955345],[6.125487,4.697942],[-5.292403,2.347838],[-5.938845,7.042214],[8.984884,6.398841],[6.739922,2.452296],[2.124696,-2.443576],[-5.278945,9.113681],[2.630191,-2.675114],[-8.815236,-8.385357],[5.608674,5.545841]], dtype = "float32")#candidate|535|(80, 2)|const|float32
call_532 = relay.TupleGetItem(func_158_call(relay.reshape(var_533.astype('uint8'), [12, 3]), relay.reshape(var_533.astype('uint8'), [12, 3]), relay.reshape(const_534.astype('int32'), [91,]), relay.reshape(const_535.astype('float32'), [2, 80]), ), 3)
call_536 = relay.TupleGetItem(func_163_call(relay.reshape(var_533.astype('uint8'), [12, 3]), relay.reshape(var_533.astype('uint8'), [12, 3]), relay.reshape(const_534.astype('int32'), [91,]), relay.reshape(const_535.astype('float32'), [2, 80]), ), 3)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
call_558 = func_20_call(relay.reshape(const_535.astype('float32'), [10, 16]))
call_559 = func_20_call(relay.reshape(const_535.astype('float32'), [10, 16]))
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
var_566 = relay.var("var_566", dtype = "float32", shape = (312,))#candidate|566|(312,)|var|float32
call_565 = relay.TupleGetItem(func_232_call(relay.reshape(var_566.astype('float32'), [6, 13, 4])), 5)
call_567 = relay.TupleGetItem(func_234_call(relay.reshape(var_566.astype('float32'), [6, 13, 4])), 5)
var_570 = relay.var("var_570", dtype = "float32", shape = (312,))#candidate|570|(312,)|var|float32
bop_571 = relay.not_equal(var_566.astype('bool'), relay.reshape(var_570.astype('bool'), relay.shape_of(var_566))) # shape=(312,)
func_422_call = mod.get_global_var('func_422')
func_426_call = mutated_mod.get_global_var('func_426')
const_576 = relay.const([-4,6,6,-10,2,4,2,-8,-2,-3,-7,-5,1,-9,7,8,3,-5,8,9,-5,-6,2,2,-10,-5,1,5,1,-2,-2,-2,-6,3,10,4,9,-6,5,5,2,-2,7,4,-3,-6,7,-9,7,-9,-6,-4,-4,-5,8,-2,-9,3,6,-6,4,-4,-2,-5,7,-5,8,7,10,9,-2,-8,9,-7,-9,-2,6,3], dtype = "uint16")#candidate|576|(78,)|const|uint16
call_575 = relay.TupleGetItem(func_422_call(relay.reshape(const_576.astype('uint16'), [13, 2, 3]), relay.reshape(const_576.astype('uint16'), [13, 2, 3]), relay.reshape(const_576.astype('bool'), [13, 2, 3]), ), 5)
call_577 = relay.TupleGetItem(func_426_call(relay.reshape(const_576.astype('uint16'), [13, 2, 3]), relay.reshape(const_576.astype('uint16'), [13, 2, 3]), relay.reshape(const_576.astype('bool'), [13, 2, 3]), ), 5)
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
call_587 = relay.TupleGetItem(func_232_call(relay.reshape(bop_571.astype('float32'), [6, 13, 4])), 4)
call_588 = relay.TupleGetItem(func_234_call(relay.reshape(bop_571.astype('float32'), [6, 13, 4])), 4)
uop_595 = relay.asinh(var_525.astype('float64')) # shape=(13, 8)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
call_597 = func_44_call(relay.reshape(call_565.astype('int32'), [7, 13]))
call_598 = func_44_call(relay.reshape(call_565.astype('int32'), [7, 13]))
output = relay.Tuple([bop_527,call_532,var_533,const_534,const_535,call_558,call_565,bop_571,call_575,const_576,call_587,uop_595,call_597,])
output2 = relay.Tuple([bop_527,call_536,var_533,const_534,const_535,call_559,call_567,bop_571,call_577,const_576,call_588,uop_595,call_598,])
func_601 = relay.Function([var_525,var_526,var_533,var_566,var_570,], output)
mod['func_601'] = func_601
mod = relay.transform.InferType()(mod)
var_602 = relay.var("var_602", dtype = "uint32", shape = (13, 8))#candidate|602|(13, 8)|var|uint32
var_603 = relay.var("var_603", dtype = "uint32", shape = (13, 8))#candidate|603|(13, 8)|var|uint32
var_604 = relay.var("var_604", dtype = "uint8", shape = (36,))#candidate|604|(36,)|var|uint8
var_605 = relay.var("var_605", dtype = "float32", shape = (312,))#candidate|605|(312,)|var|float32
var_606 = relay.var("var_606", dtype = "float32", shape = (312,))#candidate|606|(312,)|var|float32
output = func_601(var_602,var_603,var_604,var_605,var_606,)
func_607 = relay.Function([var_602,var_603,var_604,var_605,var_606,], output)
mutated_mod['func_607'] = func_607
mutated_mod = relay.transform.InferType()(mutated_mod)
var_631 = relay.var("var_631", dtype = "float32", shape = (8, 12))#candidate|631|(8, 12)|var|float32
uop_632 = relay.asinh(var_631.astype('float32')) # shape=(8, 12)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
var_637 = relay.var("var_637", dtype = "int32", shape = (91, 1))#candidate|637|(91, 1)|var|int32
call_636 = func_44_call(relay.reshape(var_637.astype('int32'), [7, 13]))
call_638 = func_44_call(relay.reshape(var_637.astype('int32'), [7, 13]))
uop_639 = relay.rsqrt(uop_632.astype('float64')) # shape=(8, 12)
func_601_call = mod.get_global_var('func_601')
func_607_call = mutated_mod.get_global_var('func_607')
const_645 = relay.const([-8,-8,10,4,-7,10,5,10,-4,2,-8,6,-8,7,10,9,-4,-8,4,-4,5,4,2,9,6,5,-8,1,-4,5,8,-1,2,9,-1,9,-6,5,-7,-3,-8,5,9,-9,4,-8,9,-5,9,8,-8,6,-10,-2,-7,-9,5,-8,-2,7,2,-8,-5,3,8,3,-10,2,10,-6,10,2,1,-10,-7,-9,-2,-10,-10,8,4,-6,10,4,-3,-9,-8,-10,8,-2,-7,-4,7,-4,6,-9,-3,7,-4,-5,3,-3,2,-10], dtype = "uint32")#candidate|645|(104,)|const|uint32
const_646 = relay.const([-9,10,-2,-2,-2,10,9,1,-8,5,-5,-7,1,-5,6,-1,-10,-4,6,5,-1,3,8,1,6,-3,6,6,-2,2,-9,-4,-4,3,4,-7], dtype = "uint8")#candidate|646|(36,)|const|uint8
var_647 = relay.var("var_647", dtype = "float32", shape = (312,))#candidate|647|(312,)|var|float32
call_644 = relay.TupleGetItem(func_601_call(relay.reshape(const_645.astype('uint32'), [13, 8]), relay.reshape(const_645.astype('uint32'), [13, 8]), relay.reshape(const_646.astype('uint8'), [36,]), relay.reshape(var_647.astype('float32'), [312,]), relay.reshape(var_647.astype('float32'), [312,]), ), 2)
call_648 = relay.TupleGetItem(func_607_call(relay.reshape(const_645.astype('uint32'), [13, 8]), relay.reshape(const_645.astype('uint32'), [13, 8]), relay.reshape(const_646.astype('uint8'), [36,]), relay.reshape(var_647.astype('float32'), [312,]), relay.reshape(var_647.astype('float32'), [312,]), ), 2)
func_20_call = mod.get_global_var('func_20')
func_22_call = mutated_mod.get_global_var('func_22')
const_656 = relay.const([[-5.992549,-7.457201,-3.758250,5.869910,-2.426943,6.213975,1.407533,-5.139477,6.852408,-4.828074,2.626022,2.850592,8.281123,1.807932,-1.592679,3.110932,-0.206578,6.016491,3.337646,-8.817781],[-3.081293,-7.665943,5.814133,0.944686,9.110340,-0.213931,1.224721,3.603839,-0.535886,-1.638209,-9.421682,-9.207634,-6.090215,1.826875,-8.940178,5.780192,-5.689544,-7.999494,-9.235806,-4.584184],[8.304165,-6.518084,-8.253730,4.589859,-1.013796,-4.320265,6.134659,-0.509215,-5.346492,9.621421,7.990535,3.457899,3.915703,2.118269,-3.862164,7.284467,-6.389325,-7.287547,1.653449,-1.461104],[6.867966,-8.987548,8.427260,-4.805001,5.016128,-8.067914,7.167470,-9.787126,2.220644,-2.750415,8.501620,-9.892689,-2.679495,0.989791,-9.322041,9.013630,-2.303085,4.238170,-1.799789,5.332596],[8.635269,-7.590603,-2.224957,9.809748,-8.905215,4.500198,9.633195,-8.568321,-7.115985,6.892522,0.498537,-9.046714,6.920808,8.266131,-3.561069,9.891770,-3.182983,-9.363073,-1.496255,-0.854427],[4.711510,-8.688512,6.509085,-8.038520,-1.661248,5.893427,3.034101,0.819103,0.043898,-0.975553,-2.067368,1.955304,-5.707989,1.636586,-4.353351,5.940250,5.815957,-2.506268,-4.450242,-0.221717],[9.799106,-2.117765,6.747463,4.840172,-2.069318,-6.227489,5.226668,5.267726,6.369951,-5.443094,1.661531,-5.277576,2.513251,5.594464,3.289411,9.780158,5.547745,2.770390,8.691349,8.984213],[8.290501,-9.600359,-6.623514,-1.947484,-5.327957,5.230662,-8.075763,-0.618861,-1.768102,-9.264571,6.437905,-2.360276,4.865646,-7.128896,-7.870358,8.945138,8.171782,-5.462418,0.738851,-5.943398]], dtype = "float32")#candidate|656|(8, 20)|const|float32
call_655 = func_20_call(relay.reshape(const_656.astype('float32'), [10, 16]))
call_657 = func_20_call(relay.reshape(const_656.astype('float32'), [10, 16]))
uop_658 = relay.sin(uop_639.astype('float64')) # shape=(8, 12)
output = relay.Tuple([call_636,var_637,call_644,const_645,const_646,var_647,call_655,const_656,uop_658,])
output2 = relay.Tuple([call_638,var_637,call_648,const_645,const_646,var_647,call_657,const_656,uop_658,])
func_662 = relay.Function([var_631,var_637,var_647,], output)
mod['func_662'] = func_662
mod = relay.transform.InferType()(mod)
var_663 = relay.var("var_663", dtype = "float32", shape = (8, 12))#candidate|663|(8, 12)|var|float32
var_664 = relay.var("var_664", dtype = "int32", shape = (91, 1))#candidate|664|(91, 1)|var|int32
var_665 = relay.var("var_665", dtype = "float32", shape = (312,))#candidate|665|(312,)|var|float32
output = func_662(var_663,var_664,var_665,)
func_666 = relay.Function([var_663,var_664,var_665,], output)
mutated_mod['func_666'] = func_666
mutated_mod = relay.transform.InferType()(mutated_mod)
var_731 = relay.var("var_731", dtype = "int32", shape = (5, 9, 12))#candidate|731|(5, 9, 12)|var|int32
var_732 = relay.var("var_732", dtype = "int32", shape = (5, 9, 12))#candidate|732|(5, 9, 12)|var|int32
bop_733 = relay.minimum(var_731.astype('int32'), relay.reshape(var_732.astype('int32'), relay.shape_of(var_731))) # shape=(5, 9, 12)
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
const_739 = relay.const([[-1.352929,-6.840312],[2.448301,-9.266618],[7.743869,-0.675848],[7.340372,7.771519],[7.434343,-3.244446],[7.554848,-3.942768],[-6.109347,-8.072104],[9.590089,-9.296909],[-4.825677,2.710003],[-7.016889,7.053177],[8.798408,-7.962810],[5.106382,-6.460297],[-8.503629,5.041031],[2.429861,-9.099098],[4.944051,9.170363],[-3.682411,-2.215618],[-8.326580,-2.400499],[6.128000,6.775839],[5.648418,0.873583],[8.949238,9.424321],[8.492609,-7.226004],[-8.224853,-2.449357],[5.067723,2.355376],[6.641127,-5.048792],[4.460468,4.929586],[-8.670989,0.865024],[-6.598934,5.313700],[4.423941,7.166345],[3.200093,3.090025],[-4.560103,2.458067],[0.816119,8.916630],[-1.185217,6.742569],[-1.810087,-2.729567],[3.762026,7.355655],[-5.312393,8.697516],[3.349410,-2.482863],[-9.995416,3.335524],[7.152686,-1.984552],[9.178584,-8.951338],[-6.842752,9.947899],[-1.851098,5.039082],[-8.328449,9.196785],[1.091671,-6.317133],[-0.475106,6.715517],[-4.342453,4.565389],[2.846003,9.250660],[-7.402648,-3.196106],[1.722344,4.819875],[6.436640,9.436235],[1.110412,2.346215],[2.646338,-2.940333],[-5.302427,2.927979],[-7.401577,8.836295],[-3.025883,-0.878765],[-9.252407,1.678820],[-9.647056,8.744500],[-1.721475,8.690747],[7.997948,-4.931647],[-3.228393,-9.664823],[9.458709,-4.254090],[2.576889,5.622932],[-5.121939,-4.107731],[8.553262,-9.700129],[-7.035908,3.168684],[-1.210449,5.313676],[9.486821,-1.779100],[-3.762712,6.931770],[-1.095322,3.286773],[9.606183,-7.386904],[-2.743665,-4.950444],[-9.505929,8.665484],[-1.680276,6.027017],[-8.238056,-4.507423],[5.004740,1.640479],[-6.880324,6.746840],[-1.312656,-2.293689],[4.680346,8.796377],[0.935493,8.241468],[-5.404658,-4.518513],[-8.707592,0.816082],[-7.404773,-9.503800],[5.797601,8.211264],[-4.490678,-1.012533],[8.150400,-1.147569],[6.432170,-7.031452],[4.161833,3.487458],[1.276606,-6.321004],[4.839212,-4.175520],[-6.293333,8.565379],[-7.373697,7.973669],[8.671302,-8.368363],[-4.036349,-7.703981],[-7.532128,-4.856026],[-3.334144,-5.663810],[8.199110,2.014722],[-2.173340,6.554007],[1.087148,-8.025490],[8.011794,-2.201835],[-0.483670,-3.870094],[-4.953583,-3.530939],[1.975940,-0.635490],[-0.350444,3.817412],[-7.285902,1.909509],[-0.720496,2.909617],[2.357916,-7.923859],[4.453059,-9.190523],[7.441474,-7.416358],[6.509598,-5.401273],[3.292789,-6.485460],[2.347819,2.678234],[5.222290,1.945441],[1.703767,6.202743],[-3.785722,-7.247750],[-3.529627,6.804589],[2.649049,8.021439],[3.202935,3.319708],[0.078599,-6.407760],[-9.112197,-1.041550],[-4.913817,3.462603],[-8.826266,3.502557],[-0.023374,-2.239118],[-9.879558,-7.268266],[4.172662,-0.518356],[-8.743742,2.758541],[9.168298,-3.666245],[9.955411,1.729154],[-3.678978,-9.541296],[9.533485,-9.628562],[0.122300,5.370904],[-1.918320,5.663602],[7.088785,-1.381295],[1.882460,1.561006],[-2.147348,-2.385273],[-1.108276,7.480200],[1.885750,7.735144],[3.786215,-0.995310],[-3.758733,-7.121863],[-5.767882,-7.259482],[6.463040,3.101422],[-9.738650,-9.330719],[-4.634458,4.741286],[-4.661824,-8.091553],[-3.877942,0.297093],[1.163270,-6.819260],[4.370493,-5.631978],[-0.286740,9.156084],[-9.477537,-1.835931],[-9.844096,3.598965],[-6.142330,6.267899],[4.183906,-7.926211],[-4.370524,-9.646433],[1.349220,0.754227],[2.121575,-0.935795],[8.108639,-9.503208],[-6.807435,-8.779817],[4.388176,-9.818162]], dtype = "float32")#candidate|739|(156, 2)|const|float32
call_738 = relay.TupleGetItem(func_232_call(relay.reshape(const_739.astype('float32'), [6, 13, 4])), 0)
call_740 = relay.TupleGetItem(func_234_call(relay.reshape(const_739.astype('float32'), [6, 13, 4])), 0)
func_158_call = mod.get_global_var('func_158')
func_163_call = mutated_mod.get_global_var('func_163')
var_745 = relay.var("var_745", dtype = "uint8", shape = (36,))#candidate|745|(36,)|var|uint8
var_746 = relay.var("var_746", dtype = "int32", shape = (91, 1))#candidate|746|(91, 1)|var|int32
const_747 = relay.const([[3.421928,-3.383322,8.000253,1.279152,4.750747,4.183699,7.804852,8.957222,9.331721,2.661507,3.070934,5.555531,7.441164,-3.213028,-3.345083,-5.802575,8.819042,7.103812,-0.693011,-0.378253],[2.392625,5.234602,8.780210,5.095285,3.484225,-5.518259,6.117838,-2.898032,-6.590907,-7.609505,1.243199,-9.780242,2.283213,8.485084,2.726443,-0.144193,-0.292217,0.472581,-4.613893,-6.773378],[8.677119,-4.050288,-2.049709,-8.513306,1.825627,1.886848,7.291070,5.652632,8.995240,7.425343,-7.609012,5.269828,3.083478,-7.567873,-1.758860,5.415820,8.572969,-8.598993,-5.654657,7.339647],[4.461084,-8.461931,-8.611076,-5.199020,1.869537,-2.148985,3.715693,-3.921336,-6.648114,-8.119814,4.746434,3.971521,8.685081,-8.401884,-9.659653,1.693145,7.849246,-3.090854,8.000721,4.997798],[-0.523317,5.089955,-8.296983,-3.192911,-8.758791,8.951055,-0.815565,6.155662,-5.801144,-6.188285,8.662483,-3.804835,5.239712,-3.172895,8.993078,0.174985,-0.845163,5.748190,-3.409476,6.957735],[-0.240308,-6.750750,8.742496,-9.635338,-5.651416,-0.191526,-4.826274,-4.311581,-2.867136,4.490357,-1.271507,5.774732,3.739120,3.391169,3.252249,-3.864453,3.413543,-6.974676,-7.676805,-7.296075],[7.989231,-5.166888,-4.673825,-3.068362,-2.888419,-8.179519,-6.534966,5.845908,2.593464,-2.163292,1.593130,-1.798374,8.151043,-0.364147,-4.250002,-0.499887,2.667767,6.360546,8.284867,-5.014346],[9.288718,9.729338,6.440004,2.861554,8.040119,8.193872,6.936060,-5.782623,-0.101151,-5.527765,5.077815,9.635601,-2.027717,-3.139122,-7.815576,1.967740,-7.460244,7.442203,5.502360,-5.217453]], dtype = "float32")#candidate|747|(8, 20)|const|float32
call_744 = relay.TupleGetItem(func_158_call(relay.reshape(var_745.astype('uint8'), [12, 3]), relay.reshape(var_745.astype('uint8'), [12, 3]), relay.reshape(var_746.astype('int32'), [91,]), relay.reshape(const_747.astype('float32'), [2, 80]), ), 4)
call_748 = relay.TupleGetItem(func_163_call(relay.reshape(var_745.astype('uint8'), [12, 3]), relay.reshape(var_745.astype('uint8'), [12, 3]), relay.reshape(var_746.astype('int32'), [91,]), relay.reshape(const_747.astype('float32'), [2, 80]), ), 4)
bop_760 = relay.greater(const_739.astype('bool'), relay.reshape(call_738.astype('bool'), relay.shape_of(const_739))) # shape=(156, 2)
bop_763 = relay.greater(const_739.astype('bool'), relay.reshape(call_740.astype('bool'), relay.shape_of(const_739))) # shape=(156, 2)
uop_764 = relay.acosh(bop_760.astype('float32')) # shape=(156, 2)
uop_766 = relay.acosh(bop_763.astype('float32')) # shape=(156, 2)
bop_767 = relay.less_equal(uop_764.astype('bool'), relay.reshape(bop_760.astype('bool'), relay.shape_of(uop_764))) # shape=(156, 2)
bop_770 = relay.less_equal(uop_766.astype('bool'), relay.reshape(bop_763.astype('bool'), relay.shape_of(uop_766))) # shape=(156, 2)
output = relay.Tuple([bop_733,call_744,var_745,var_746,const_747,bop_767,])
output2 = relay.Tuple([bop_733,call_748,var_745,var_746,const_747,bop_770,])
func_774 = relay.Function([var_731,var_732,var_745,var_746,], output)
mod['func_774'] = func_774
mod = relay.transform.InferType()(mod)
var_775 = relay.var("var_775", dtype = "int32", shape = (5, 9, 12))#candidate|775|(5, 9, 12)|var|int32
var_776 = relay.var("var_776", dtype = "int32", shape = (5, 9, 12))#candidate|776|(5, 9, 12)|var|int32
var_777 = relay.var("var_777", dtype = "uint8", shape = (36,))#candidate|777|(36,)|var|uint8
var_778 = relay.var("var_778", dtype = "int32", shape = (91, 1))#candidate|778|(91, 1)|var|int32
output = func_774(var_775,var_776,var_777,var_778,)
func_779 = relay.Function([var_775,var_776,var_777,var_778,], output)
mutated_mod['func_779'] = func_779
mutated_mod = relay.transform.InferType()(mutated_mod)
var_806 = relay.var("var_806", dtype = "int32", shape = (12, 2))#candidate|806|(12, 2)|var|int32
const_807 = relay.const([[-9,3],[7,-4],[3,4],[-7,8],[3,9],[5,-1],[3,6],[-10,1],[7,1],[-6,-8],[-6,-8],[3,-1]], dtype = "int32")#candidate|807|(12, 2)|const|int32
bop_808 = relay.bitwise_and(var_806.astype('int32'), relay.reshape(const_807.astype('int32'), relay.shape_of(var_806))) # shape=(12, 2)
uop_818 = relay.erf(const_807.astype('float32')) # shape=(12, 2)
const_827 = relay.const([[7.204971,-9.841561],[7.040898,-2.909831],[-5.945694,7.010935],[-1.348136,2.843562],[-5.394417,1.387017],[-5.153298,-9.338979],[-7.422856,-2.556371],[9.598568,8.191541],[9.195207,-2.716142],[-7.335151,-2.000135],[-7.769235,5.572432],[8.048385,3.540442]], dtype = "float32")#candidate|827|(12, 2)|const|float32
bop_828 = relay.bitwise_or(uop_818.astype('int64'), relay.reshape(const_827.astype('int64'), relay.shape_of(uop_818))) # shape=(12, 2)
bop_839 = relay.power(uop_818.astype('float64'), relay.reshape(bop_828.astype('float64'), relay.shape_of(uop_818))) # shape=(12, 2)
var_850 = relay.var("var_850", dtype = "float32", shape = (12, 2))#candidate|850|(12, 2)|var|float32
bop_851 = relay.minimum(uop_818.astype('uint16'), relay.reshape(var_850.astype('uint16'), relay.shape_of(uop_818))) # shape=(12, 2)
bop_855 = relay.maximum(uop_818.astype('uint8'), relay.reshape(bop_828.astype('uint8'), relay.shape_of(uop_818))) # shape=(12, 2)
func_44_call = mod.get_global_var('func_44')
func_46_call = mutated_mod.get_global_var('func_46')
const_861 = relay.const([[4,10,-6,1,-3,-5,-1],[-5,7,5,3,-10,-1,8],[-9,-5,-4,8,-3,3,-10],[-6,3,5,-5,2,2,-9],[-4,-8,-7,-10,1,10,5],[-9,10,-7,7,2,2,2],[-5,1,-9,-6,4,-9,2],[-9,5,-3,5,9,-1,6],[7,1,10,-5,1,-9,-7],[-9,10,-7,8,-9,1,-5],[-8,-10,-8,9,1,8,5],[-6,-8,1,-3,-2,-10,1],[10,6,6,5,6,-1,7]], dtype = "int32")#candidate|861|(13, 7)|const|int32
call_860 = func_44_call(relay.reshape(const_861.astype('int32'), [7, 13]))
call_862 = func_44_call(relay.reshape(const_861.astype('int32'), [7, 13]))
bop_868 = relay.subtract(bop_855.astype('int64'), relay.reshape(bop_828.astype('int64'), relay.shape_of(bop_855))) # shape=(12, 2)
func_774_call = mod.get_global_var('func_774')
func_779_call = mutated_mod.get_global_var('func_779')
const_874 = relay.const([3,-5,-2,-9,6,-3,7,-2,-4,1,-7,9,1,-3,5,5,2,8,-10,-3,4,-3,-7,-6,-5,4,-7,4,-10,2,-4,5,6,-2,-2,-3,2,-7,-10,-8,4,-7,-5,-7,-9,-2,8,9,-1,-10,-7,-3,-8,7,-5,-5,7,8,-2,-8,-5,-5,8,-9,-4,-5,-4,1,-9,2,-3,3,-8,3,7,2,-4,7,1,-4,-6,8,-7,2,7,-2,-10,2,7,9,-9,-7,9,8,1,7,-10,5,-5,-5,-10,5,-1,9,4,-1,-10,7,-2,9,6,-1,10,6,-3,-10,-8,-3,1,9,10,8,8,-7,-2,-6,-4,-9,-2,-2,7,-6,-6,-6,-2,2,-4,-10,-10,5,-7,4,-7,-6,7,-2,9,-9,-8,1,7,2,8,4,-10,9,9,9,-6,2,2,-6,-3,-5,10,-1,-3,-10,6,6,2,-3,1,-4,-8,8,2,10,-1,-8,-3,1,3,-7,-5,5,10,-6,-4,8,1,-2,2,-2,-9,7,1,-9,7,-5,9,-10,10,10,10,-3,-3,-6,9,9,8,1,4,-6,-6,-3,-9,-5,-1,-2,-2,2,2,-2,3,8,-7,-1,-10,2,-7,10,10,-4,2,-1,-3,-10,10,3,-5,-7,1,-5,-6,-5,-4,7,-4,-4,-9,9,-8,4,-4,1,9,2,-5,-10,-4,2,-4,1,-3,2,-9,-3,-1,5,7,5,-4,-7,3,-8,2,6,-10,-4,3,-5,9,9,9,7,10,2,-8,2,6,-10,-1,-6,5,-6,-6,-1,10,-6,6,3,-3,1,-5,5,-3,10,3,-7,-8,7,5,-8,1,-4,-7,-4,-3,2,-8,-9,-3,-8,-7,-7,9,8,3,9,-1,-1,-1,9,-10,3,-9,2,3,10,-7,6,10,-10,-3,-7,3,5,4,9,6,-3,-6,10,-7,3,7,1,-5,-2,-1,3,6,-9,1,-6,2,3,-6,-1,-3,-8,4,-8,10,5,9,-3,-3,7,10,2,-3,-2,-8,-4,5,-3,-1,6,-1,1,9,6,8,-6,7,2,3,-10,7,3,-6,-3,-7,8,-1,7,5,-4,9,3,9,5,-1,4,9,-2,-6,-1,7,3,8,-10,5,8,-3,7,6,6,9,7,-9,-2,-5,10,-2,-10,2,3,-3,9,-2,1,-10,-5,-7,6,1,-5,-1,-10,7,-5,-1,6,9,-3,-4,1,1,5,-10,6,-1,10,5,6,10,-2,-10,-7,4,2,-4,1,10,-3,-9,2,1,3,-1,-9,10,4,-10,1,-2,-3,5,-2,-2,-5,-6,-2,-10,5,3,2,7,-3,10,-2,-6,-9,-9,-10,-7,-3,-4,2,-9,5,4,7,-5,10,-6,-8,-9,5,-9,10,-10,1,-9,9,-1,-2,-2,-6,-9,4,1,8,-10,-10,8,5], dtype = "int32")#candidate|874|(540,)|const|int32
const_875 = relay.const([-3,-1,-8,3,-7,7,-1,2,4,-4,6,-3,-6,-10,-10,-8,1,-5,4,10,-9,1,-9,-7,-8,-8,-8,-9,-7,-7,1,8,-5,-2,1,4], dtype = "uint8")#candidate|875|(36,)|const|uint8
call_873 = relay.TupleGetItem(func_774_call(relay.reshape(const_874.astype('int32'), [5, 9, 12]), relay.reshape(const_874.astype('int32'), [5, 9, 12]), relay.reshape(const_875.astype('uint8'), [36,]), relay.reshape(const_861.astype('int32'), [91, 1]), ), 2)
call_876 = relay.TupleGetItem(func_779_call(relay.reshape(const_874.astype('int32'), [5, 9, 12]), relay.reshape(const_874.astype('int32'), [5, 9, 12]), relay.reshape(const_875.astype('uint8'), [36,]), relay.reshape(const_861.astype('int32'), [91, 1]), ), 2)
uop_877 = relay.tan(bop_868.astype('float64')) # shape=(12, 2)
output = relay.Tuple([bop_808,bop_839,bop_851,call_860,const_861,call_873,const_874,const_875,uop_877,])
output2 = relay.Tuple([bop_808,bop_839,bop_851,call_862,const_861,call_876,const_874,const_875,uop_877,])
func_890 = relay.Function([var_806,var_850,], output)
mod['func_890'] = func_890
mod = relay.transform.InferType()(mod)
mutated_mod['func_890'] = func_890
mutated_mod = relay.transform.InferType()(mutated_mod)
func_890_call = mutated_mod.get_global_var('func_890')
var_892 = relay.var("var_892", dtype = "int32", shape = (12, 2))#candidate|892|(12, 2)|var|int32
var_893 = relay.var("var_893", dtype = "float32", shape = (12, 2))#candidate|893|(12, 2)|var|float32
call_891 = func_890_call(var_892,var_893,)
output = call_891
func_894 = relay.Function([var_892,var_893,], output)
mutated_mod['func_894'] = func_894
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1117 = relay.var("var_1117", dtype = "float64", shape = (8, 15, 1))#candidate|1117|(8, 15, 1)|var|float64
var_1118 = relay.var("var_1118", dtype = "float64", shape = (8, 15, 6))#candidate|1118|(8, 15, 6)|var|float64
bop_1119 = relay.divide(var_1117.astype('float64'), var_1118.astype('float64')) # shape=(8, 15, 6)
output = relay.Tuple([bop_1119,])
output2 = relay.Tuple([bop_1119,])
func_1126 = relay.Function([var_1117,var_1118,], output)
mod['func_1126'] = func_1126
mod = relay.transform.InferType()(mod)
var_1127 = relay.var("var_1127", dtype = "float64", shape = (8, 15, 1))#candidate|1127|(8, 15, 1)|var|float64
var_1128 = relay.var("var_1128", dtype = "float64", shape = (8, 15, 6))#candidate|1128|(8, 15, 6)|var|float64
output = func_1126(var_1127,var_1128,)
func_1129 = relay.Function([var_1127,var_1128,], output)
mutated_mod['func_1129'] = func_1129
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1241 = relay.const([[[-4.485425,-1.930661,4.151111,6.753188,-7.980057,-2.946551,7.173663,-9.291057,4.685962],[-8.701228,-7.789671,4.361121,-4.124752,-5.294800,-9.371279,-2.642645,2.771722,-7.297125],[8.204416,3.184602,-0.383904,-0.538272,-9.549412,-0.857234,-0.455894,4.358495,9.763972],[-9.286634,-4.920690,-9.799347,-4.843012,-1.967057,9.048704,-9.691774,-9.462166,-8.253782],[1.506114,5.608700,0.470202,1.753076,-7.159400,-6.955182,9.233052,8.670591,9.409871],[-1.928336,4.086154,-5.030595,2.445909,9.341403,-7.601793,6.150429,2.901076,5.964069],[-8.459410,-9.122927,7.664687,-8.876537,-4.555330,3.330480,6.053432,2.019614,-0.246366],[6.481068,6.535790,0.447458,-1.203325,9.240922,0.608214,-2.714542,-0.201286,-8.234373],[-5.919773,-1.864387,3.232447,-8.931998,-4.040772,8.428486,8.855416,-9.578608,8.196481],[-5.707071,-9.606010,0.391491,2.726244,0.907716,8.958173,1.169422,6.549988,-6.803757],[5.703611,6.247331,-2.967462,-1.690592,-5.296928,-9.665691,5.399871,7.902233,1.595456],[6.685747,6.512453,-3.909733,8.490011,-9.551414,3.723614,6.027511,3.159457,-1.170962],[3.200888,-8.380375,-9.541795,1.024429,2.677391,1.954767,-3.053496,4.381798,-3.650211],[-9.649357,-8.578192,-4.450140,0.051840,-7.420527,-7.871241,-6.967984,-0.932558,5.836869]],[[-0.859193,-2.902511,-9.888024,-0.914416,-0.279573,-9.975836,3.199549,-8.178429,1.753139],[1.437537,-2.495759,5.520333,0.878968,-4.331425,-0.267639,-1.970646,5.980312,8.972149],[-5.285138,-8.220144,0.449891,4.493939,0.016841,-9.435028,-5.454273,1.383529,-3.230036],[-0.595391,-6.531273,9.694835,-3.956375,2.154782,-2.333081,3.898266,-4.272369,8.871882],[-2.596395,-2.037437,5.452493,-7.847988,0.656413,-7.547776,0.049463,8.432419,-7.430070],[-3.850130,-9.081020,-8.583450,-6.826373,4.563248,6.210898,6.667134,-2.542910,1.343861],[-2.081253,3.242338,5.485003,9.243802,-2.151073,-1.804326,6.943341,-2.631876,9.235701],[-0.617505,-1.319480,-8.871546,4.160311,-5.089878,-0.251277,9.675240,0.957954,-5.716080],[-6.377349,0.900228,-7.080528,4.293479,4.817714,1.415134,-3.893013,-0.102680,7.912120],[-8.502128,-1.403189,-4.490091,-9.170124,-6.700066,9.613389,-9.592606,-2.125762,-9.015802],[-2.441499,9.651378,5.201836,-0.122879,-6.506860,0.343794,2.750760,-9.228947,-4.517976],[-4.135061,-9.249950,-8.148338,9.295308,-4.862027,9.419482,-4.355226,-0.581643,1.202678],[-9.760197,-1.933214,-6.735226,6.541544,0.444487,-1.593967,-0.768473,8.815544,8.078699],[-2.385193,-4.052326,-5.361129,-3.374774,4.533610,-6.561227,-9.271094,-7.066137,-1.309564]],[[-1.516142,6.500221,8.232553,6.775447,-5.532685,0.598769,1.907997,3.420727,-1.444328],[1.699795,-8.410122,8.125218,8.110508,-3.926756,-0.222816,-4.273625,-9.761116,2.156222],[0.152783,8.966222,8.872146,5.197373,9.970912,-3.841013,7.433601,-4.358245,8.442270],[-9.949767,1.178818,-9.816025,3.734376,3.332197,-9.882942,-4.578211,5.258277,-7.286679],[-7.774085,1.099510,-6.427046,1.778488,7.557768,0.318726,2.424423,2.034671,-3.309528],[2.015924,7.218578,5.066140,0.859744,3.352654,-7.772465,3.295654,-6.686625,-3.071247],[-5.034189,-6.981160,6.609411,-4.598759,5.371778,7.382219,-0.867881,-6.342397,5.077663],[3.032551,3.353809,-1.637003,-1.547546,-7.782442,-8.786546,0.749507,-5.920383,7.887296],[-4.278542,-8.448126,9.608534,4.868756,-9.960318,3.196617,-7.170777,-8.227813,-7.374549],[3.834239,-4.591708,4.749510,-4.700171,-7.906684,-3.732033,-4.522004,6.270915,-1.490810],[-7.440328,-2.975534,0.267318,8.129789,8.093215,6.789686,-1.191116,-4.941220,-6.245111],[6.742269,2.359813,6.333147,0.708543,-0.985411,6.459859,-9.568584,-5.209956,-1.983176],[-3.310510,7.377610,-5.326211,9.313213,2.076844,8.345387,2.581966,-3.539314,-8.394804],[-5.657963,6.077536,0.923145,-8.932563,7.978620,-1.662273,-3.961987,0.322917,-9.797510]]], dtype = "float64")#candidate|1241|(3, 14, 9)|const|float64
const_1242 = relay.const([[[-8.890928,-0.762247,-6.175639,-1.757841,-9.545322,2.337704,8.878317,6.605421,9.878847],[-8.294861,-3.731594,-8.828641,1.397727,3.874791,0.994413,-0.833706,6.747727,-2.729576],[-2.533624,9.829588,9.737827,-8.992284,2.805157,9.789868,8.800082,5.810516,7.274176],[-7.422516,0.287729,-4.606901,-2.001954,-0.521444,7.911810,-2.470050,-9.832725,-5.905763],[-4.948613,-1.710803,-9.820308,-4.097586,4.689800,-1.279587,6.972690,9.678607,-5.846233],[-2.414028,-3.684246,-5.977681,9.436728,-1.939913,-6.577589,1.958608,5.536106,3.971034],[-6.113070,3.538301,5.077481,-1.833114,-1.570685,-7.099724,7.945846,-2.009966,7.727327],[-3.230520,1.687227,-8.285086,3.177651,-7.925967,6.317700,7.145575,5.939844,9.325500],[-9.662782,2.989630,-2.799666,4.595898,-0.933322,4.293672,-3.243495,-3.827239,-5.326148],[-2.885812,9.201380,-1.234207,-9.803677,-9.180709,-4.273867,-3.998388,1.092111,-1.918234],[6.851933,8.440741,3.950148,-3.729040,-9.256351,1.996683,9.180985,1.489359,-7.953464],[-2.086617,3.083619,-6.689747,-1.368169,-4.112774,-2.482180,0.305983,-3.999581,-6.641062],[-6.774822,8.915445,0.443774,7.612066,-1.410429,7.332230,1.273510,4.000607,-6.329401],[-0.189501,6.363226,-6.833098,2.071200,2.821504,1.080990,3.437111,-6.510268,-7.944540]],[[4.366652,3.389682,-3.273309,-0.821883,-9.251768,-1.801712,3.824105,7.321121,7.972253],[2.332032,-7.902116,-9.691409,-5.636221,2.989221,5.376155,-2.361720,0.531837,-8.390216],[8.139966,3.299603,5.400928,-4.200641,2.723327,8.737813,-8.800088,8.656452,1.031639],[-8.677901,-8.325100,9.338632,2.443435,-6.591785,3.355438,1.926139,-7.716746,-0.418169],[-7.609160,-3.388271,-7.212079,2.447902,-1.547112,4.483990,-8.833287,4.086187,3.521389],[-4.315664,7.317663,-5.050149,-6.733778,9.546729,-3.473624,-9.239682,-4.011912,-6.142095],[-1.926338,0.604815,3.890328,-0.887170,-4.090991,-0.580328,-6.848591,3.383732,3.433474],[-1.983369,5.798017,-5.642045,3.723497,-7.663837,-0.463595,4.224700,-9.974170,2.710894],[8.794603,-3.379153,-0.676607,-6.050955,3.623654,-9.875869,4.731518,-9.479789,-0.725000],[-6.741045,8.618810,0.001726,-3.137622,-2.147012,-9.485487,-1.254873,-2.152796,7.024876],[2.619443,1.696357,9.262356,2.659846,0.741713,4.971818,6.114459,1.683088,8.128833],[-1.611019,7.968249,-7.929774,-0.552094,-5.977948,2.636076,1.497327,5.937153,-5.838922],[-8.091903,3.694246,0.577208,2.050431,-0.502131,-2.236575,-7.221222,8.291777,5.183271],[2.423784,-1.660786,-4.646316,0.971472,-7.868840,0.623341,-2.319847,1.443605,7.304378]],[[-1.175299,1.169425,4.506323,-9.083623,-3.140392,-2.377934,-6.751322,0.206897,3.295007],[-8.704875,-4.027136,7.687842,7.619932,5.366500,4.998923,-1.343548,-8.319985,-6.672799],[2.821387,5.347764,6.275536,2.039327,2.199614,-9.495027,1.223944,9.270117,-2.677076],[2.742696,-8.810839,-5.666760,3.613870,-1.100023,-8.952380,7.498264,5.785088,0.467472],[-7.932267,-7.436192,2.053271,-2.204204,-5.845053,-6.449726,-3.579357,-3.947293,9.932675],[-8.330840,9.681842,-3.960648,-1.200024,8.308310,-0.120827,-0.808492,-7.945108,-8.525700],[1.026453,-0.793995,-5.313273,-0.442637,-8.134934,-1.383198,4.175114,-2.192603,9.618497],[-6.402503,-9.680458,2.617213,-5.636049,-6.007187,9.554998,2.520966,3.277380,-8.122601],[-8.367103,9.474187,0.107618,-5.979916,-0.790393,-6.750045,5.148914,0.568447,-9.708175],[-9.034966,-6.991031,1.364815,9.938148,7.886931,-5.964062,1.221550,-8.043141,6.290085],[-4.218409,2.060424,-7.525177,-9.404515,-0.203483,-8.461844,-8.536639,-1.607077,6.710590],[-2.583026,-8.052752,9.000787,4.168330,2.644524,9.504644,-5.303334,9.084762,6.585200],[8.190961,4.329806,1.146360,-7.369258,0.638204,4.221744,0.753593,4.584726,1.203727],[-8.783497,5.340379,-1.099605,0.186066,-0.578618,-3.038505,-4.980768,9.118840,-9.805412]]], dtype = "float64")#candidate|1242|(3, 14, 9)|const|float64
bop_1243 = relay.power(const_1241.astype('float64'), relay.reshape(const_1242.astype('float64'), relay.shape_of(const_1241))) # shape=(3, 14, 9)
bop_1249 = relay.bitwise_or(bop_1243.astype('int8'), relay.reshape(const_1241.astype('int8'), relay.shape_of(bop_1243))) # shape=(3, 14, 9)
output = bop_1249
output2 = bop_1249
func_1256 = relay.Function([], output)
mod['func_1256'] = func_1256
mod = relay.transform.InferType()(mod)
mutated_mod['func_1256'] = func_1256
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mutated_mod.get_global_var('func_1256')
call_1257 = func_1256_call()
output = call_1257
func_1258 = relay.Function([], output)
mutated_mod['func_1258'] = func_1258
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_1283 = func_1256_call()
call_1284 = func_1256_call()
func_601_call = mod.get_global_var('func_601')
func_607_call = mutated_mod.get_global_var('func_607')
var_1302 = relay.var("var_1302", dtype = "uint32", shape = (104,))#candidate|1302|(104,)|var|uint32
var_1303 = relay.var("var_1303", dtype = "uint8", shape = (36,))#candidate|1303|(36,)|var|uint8
const_1304 = relay.const([[-3.466403,-7.969891],[2.595740,7.658449],[-5.280442,-6.163134],[5.718040,-5.430699],[3.297062,-5.577764],[0.192621,5.670000],[-6.304096,-9.974024],[8.491567,8.427014],[-6.458723,0.162605],[-0.696265,0.658003],[-5.686459,1.356120],[1.299114,0.244007],[1.473240,7.460687],[3.549495,8.880245],[8.060325,-1.252920],[5.121761,-8.406304],[4.264292,5.943740],[-9.241955,-2.119127],[7.186309,9.107963],[9.743726,0.929582],[-9.174399,-4.998247],[5.363458,9.322920],[8.279603,1.406909],[8.634844,2.687044],[-6.376190,5.355302],[6.870391,9.471990],[6.708694,5.472590],[7.694255,3.571992],[0.565602,2.791505],[-5.175592,3.983006],[3.496747,-0.584772],[3.493632,9.098640],[8.515690,-2.620990],[4.313180,9.409151],[-1.525513,5.254804],[7.135260,9.531371],[-1.182391,-6.503315],[-0.082263,2.057095],[-9.836573,-8.247878],[0.370589,9.984903],[-8.770767,5.435185],[6.961381,-5.582335],[7.760867,-7.917319],[4.770717,-0.872391],[-2.766837,-0.117465],[-9.034600,-0.087156],[4.400955,-9.690862],[1.056663,2.567702],[-1.302804,6.584735],[8.772115,4.513741],[6.118544,-4.995334],[-8.309408,7.075934],[4.358652,9.661472],[4.680298,-8.269888],[-3.358415,6.238449],[8.702107,-2.888997],[8.057159,5.921876],[3.359838,1.315545],[-2.143275,2.391074],[-6.846468,-4.177857],[-9.528644,-9.267426],[7.571454,6.679358],[7.901935,1.781367],[6.309544,-1.061808],[0.092524,-2.034326],[5.664236,-3.693261],[2.684382,-9.829454],[-4.259316,0.482447],[-7.835810,-3.177874],[-5.162122,7.534180],[-8.460871,-3.761479],[-8.524200,-9.347650],[4.653915,5.909922],[-6.286601,-3.701351],[8.955840,-4.857769],[3.884541,0.888158],[4.265535,-8.386045],[1.624714,9.919348],[-0.506257,6.616846],[1.895764,6.420643],[-6.124061,9.584672],[-5.765349,-1.709011],[-6.358571,0.385921],[-9.339831,5.255956],[-3.741135,-7.314070],[4.099858,3.755295],[2.521655,5.660016],[-2.934838,9.842851],[6.573857,5.648532],[7.981404,4.090525],[-5.446425,0.431547],[2.094141,-1.406660],[-0.070981,7.657934],[6.449585,-1.749157],[-7.917794,3.012259],[-9.554405,2.210423],[6.973912,3.822610],[3.610784,-3.197739],[9.428088,-8.130361],[-2.973661,-7.982259],[9.727476,-6.712060],[-4.722735,-9.415863],[-0.836462,8.581859],[-1.723514,-3.525950],[2.522387,-2.182695],[0.798813,8.440261],[7.531538,3.100850],[-8.799477,-3.959205],[-9.111092,-9.797721],[6.340860,3.677648],[-8.868568,-7.518629],[4.164226,-4.381109],[-0.208744,4.487803],[6.379291,-7.639720],[4.286547,-3.726498],[1.516120,6.005778],[-1.218270,-4.085490],[-3.068240,9.140724],[-0.658466,-5.873174],[0.559356,0.715735],[9.749602,9.850810],[9.989694,1.395113],[1.210440,-1.606209],[6.479525,-4.342749],[-8.086681,0.393159],[-9.282907,9.961456],[6.276590,3.239723],[0.694679,-1.678886],[9.082668,9.887927],[-9.223008,2.680878],[0.413630,-2.108513],[-0.795119,-9.917724],[-9.917024,1.101329],[-7.236131,3.490897],[-6.976448,-0.870810],[4.921174,1.154014],[-9.036022,1.840284],[0.331632,6.896296],[4.822480,7.869777],[-5.070002,-5.277966],[-5.908716,-4.097905],[-2.212856,-4.134195],[-3.148244,-7.158711],[3.599798,-6.945736],[2.735148,-3.423815],[3.052257,0.640177],[-9.130399,-4.322412],[-4.262406,3.179029],[-8.695729,2.981399],[1.166417,-3.620541],[2.187646,-3.170091],[-7.639472,-8.120656],[-6.904964,-3.037697],[-3.242431,-6.120652],[6.056136,-7.716982],[6.910318,-2.111379]], dtype = "float32")#candidate|1304|(156, 2)|const|float32
call_1301 = relay.TupleGetItem(func_601_call(relay.reshape(var_1302.astype('uint32'), [13, 8]), relay.reshape(var_1302.astype('uint32'), [13, 8]), relay.reshape(var_1303.astype('uint8'), [36,]), relay.reshape(const_1304.astype('float32'), [312,]), relay.reshape(const_1304.astype('float32'), [312,]), ), 11)
call_1305 = relay.TupleGetItem(func_607_call(relay.reshape(var_1302.astype('uint32'), [13, 8]), relay.reshape(var_1302.astype('uint32'), [13, 8]), relay.reshape(var_1303.astype('uint8'), [36,]), relay.reshape(const_1304.astype('float32'), [312,]), relay.reshape(const_1304.astype('float32'), [312,]), ), 11)
bop_1308 = relay.greater_equal(call_1301.astype('bool'), relay.reshape(var_1302.astype('bool'), relay.shape_of(call_1301))) # shape=(13, 8)
bop_1311 = relay.greater_equal(call_1305.astype('bool'), relay.reshape(var_1302.astype('bool'), relay.shape_of(call_1305))) # shape=(13, 8)
output = relay.Tuple([call_1283,var_1303,const_1304,bop_1308,])
output2 = relay.Tuple([call_1284,var_1303,const_1304,bop_1311,])
func_1315 = relay.Function([var_1302,var_1303,], output)
mod['func_1315'] = func_1315
mod = relay.transform.InferType()(mod)
var_1316 = relay.var("var_1316", dtype = "uint32", shape = (104,))#candidate|1316|(104,)|var|uint32
var_1317 = relay.var("var_1317", dtype = "uint8", shape = (36,))#candidate|1317|(36,)|var|uint8
output = func_1315(var_1316,var_1317,)
func_1318 = relay.Function([var_1316,var_1317,], output)
mutated_mod['func_1318'] = func_1318
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_1336 = func_1256_call()
call_1337 = func_1256_call()
output = call_1336
output2 = call_1337
func_1351 = relay.Function([], output)
mod['func_1351'] = func_1351
mod = relay.transform.InferType()(mod)
mutated_mod['func_1351'] = func_1351
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1351_call = mutated_mod.get_global_var('func_1351')
call_1352 = func_1351_call()
output = call_1352
func_1353 = relay.Function([], output)
mutated_mod['func_1353'] = func_1353
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_1374 = func_1256_call()
call_1375 = func_1256_call()
output = call_1374
output2 = call_1375
func_1380 = relay.Function([], output)
mod['func_1380'] = func_1380
mod = relay.transform.InferType()(mod)
output = func_1380()
func_1381 = relay.Function([], output)
mutated_mod['func_1381'] = func_1381
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_1418 = func_1256_call()
call_1419 = func_1256_call()
func_158_call = mod.get_global_var('func_158')
func_163_call = mutated_mod.get_global_var('func_163')
var_1428 = relay.var("var_1428", dtype = "uint8", shape = (36,))#candidate|1428|(36,)|var|uint8
const_1429 = relay.const([-2,6,1,3,-9,4,8,-9,-2,-10,4,9,7,-4,-7,-10,6,-10,9,-8,8,10,-4,9,-1,-9,8,-3,-4,-3,-8,9,-2,2,4,8,6,8,-6,-5,5,-1,2,1,-1,-6,-9,4,5,3,-7,-1,-8,2,5,10,-3,-4,5,10,2,-4,2,4,-3,9,7,-3,-9,8,2,6,9,10,-7,-9,-6,-2,10,-8,-3,2,-8,-5,-8,3,-3,5,8,5,6], dtype = "int32")#candidate|1429|(91,)|const|int32
var_1430 = relay.var("var_1430", dtype = "float32", shape = (160,))#candidate|1430|(160,)|var|float32
call_1427 = relay.TupleGetItem(func_158_call(relay.reshape(var_1428.astype('uint8'), [12, 3]), relay.reshape(var_1428.astype('uint8'), [12, 3]), relay.reshape(const_1429.astype('int32'), [91,]), relay.reshape(var_1430.astype('float32'), [2, 80]), ), 0)
call_1431 = relay.TupleGetItem(func_163_call(relay.reshape(var_1428.astype('uint8'), [12, 3]), relay.reshape(var_1428.astype('uint8'), [12, 3]), relay.reshape(const_1429.astype('int32'), [91,]), relay.reshape(var_1430.astype('float32'), [2, 80]), ), 0)
output = relay.Tuple([call_1418,call_1427,var_1428,const_1429,var_1430,])
output2 = relay.Tuple([call_1419,call_1431,var_1428,const_1429,var_1430,])
func_1449 = relay.Function([var_1428,var_1430,], output)
mod['func_1449'] = func_1449
mod = relay.transform.InferType()(mod)
mutated_mod['func_1449'] = func_1449
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1449_call = mutated_mod.get_global_var('func_1449')
var_1451 = relay.var("var_1451", dtype = "uint8", shape = (36,))#candidate|1451|(36,)|var|uint8
var_1452 = relay.var("var_1452", dtype = "float32", shape = (160,))#candidate|1452|(160,)|var|float32
call_1450 = func_1449_call(var_1451,var_1452,)
output = call_1450
func_1453 = relay.Function([var_1451,var_1452,], output)
mutated_mod['func_1453'] = func_1453
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1475 = relay.var("var_1475", dtype = "float32", shape = (5, 7))#candidate|1475|(5, 7)|var|float32
uop_1476 = relay.erf(var_1475.astype('float32')) # shape=(5, 7)
output = relay.Tuple([uop_1476,])
output2 = relay.Tuple([uop_1476,])
func_1482 = relay.Function([var_1475,], output)
mod['func_1482'] = func_1482
mod = relay.transform.InferType()(mod)
mutated_mod['func_1482'] = func_1482
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1483 = relay.var("var_1483", dtype = "float32", shape = (5, 7))#candidate|1483|(5, 7)|var|float32
func_1482_call = mutated_mod.get_global_var('func_1482')
call_1484 = func_1482_call(var_1483)
output = call_1484
func_1485 = relay.Function([var_1483], output)
mutated_mod['func_1485'] = func_1485
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1351_call = mod.get_global_var('func_1351')
func_1353_call = mutated_mod.get_global_var('func_1353')
call_1520 = func_1351_call()
call_1521 = func_1351_call()
func_1449_call = mod.get_global_var('func_1449')
func_1453_call = mutated_mod.get_global_var('func_1453')
var_1530 = relay.var("var_1530", dtype = "uint8", shape = (36,))#candidate|1530|(36,)|var|uint8
const_1531 = relay.const([[3.148732,-2.543966],[3.017877,2.158975],[-2.014557,-3.446216],[-2.225734,-9.253393],[3.498662,6.942223],[0.267496,4.002615],[9.858131,-6.795091],[-5.305945,-6.669933],[2.698805,-6.731288],[-4.626624,-6.271840],[9.388951,6.593668],[-2.190736,4.772364],[7.435403,-5.069501],[-4.171541,2.659904],[-3.525380,-6.172631],[5.848551,-7.366213],[8.615280,6.525888],[1.538090,-7.992635],[0.202260,-0.557954],[-1.014269,1.209794],[-3.615858,-4.286395],[4.515453,-1.326652],[0.332753,-8.438886],[3.184505,8.045311],[0.987138,-7.098625],[-7.930997,5.128788],[-3.286277,1.042164],[-4.003435,8.551006],[-6.411102,6.430972],[1.323618,-5.926915],[-6.351650,7.313664],[8.014049,3.550488],[6.617145,9.251715],[-2.665214,4.442049],[-5.673879,8.241595],[0.896231,8.686066],[-6.054721,-9.824447],[5.026458,-8.168408],[4.369307,7.463917],[-2.163855,-3.488730],[1.159127,2.184697],[-8.892715,-7.856636],[3.396201,4.525060],[-2.647315,7.251767],[-6.378386,-8.904129],[-4.695911,8.991538],[-0.329209,-8.632366],[8.306708,5.134368],[0.414785,8.691044],[3.187160,-8.519709],[4.448822,-5.234814],[9.846009,7.831249],[3.853360,7.580683],[-3.226814,0.409493],[6.294912,-9.196039],[-5.323243,0.833536],[-0.603294,-4.749609],[4.711479,0.433615],[-8.661281,-2.567939],[2.421277,1.146858],[-4.581635,-8.096850],[-2.611922,5.078351],[7.888632,-6.169148],[5.807511,0.835792],[-5.375179,-2.063033],[9.437263,6.816535],[3.910631,4.547583],[6.234209,7.060414],[7.910906,-4.891313],[7.512632,3.747182],[4.301418,8.538625],[-3.720707,7.729596],[-1.656479,-0.784567],[5.702561,4.600388],[2.316112,4.348586],[-3.136876,-2.141268],[-1.743868,4.219267],[9.159493,-7.304392],[-8.396044,8.767143],[6.772604,-5.090967]], dtype = "float32")#candidate|1531|(80, 2)|const|float32
call_1529 = relay.TupleGetItem(func_1449_call(relay.reshape(var_1530.astype('uint8'), [36,]), relay.reshape(const_1531.astype('float32'), [160,]), ), 1)
call_1532 = relay.TupleGetItem(func_1453_call(relay.reshape(var_1530.astype('uint8'), [36,]), relay.reshape(const_1531.astype('float32'), [160,]), ), 1)
func_662_call = mod.get_global_var('func_662')
func_666_call = mutated_mod.get_global_var('func_666')
const_1536 = relay.const([-0.747123,8.131532,-7.123583,-1.351466,-4.524170,3.302255,-3.160603,-2.642860,-1.132048,-6.995346,4.469446,3.429428,6.234695,4.927020,4.067323,-6.346149,8.337323,-8.037254,-5.720696,-8.986695,2.000177,-4.032933,5.412643,4.264547,-3.278065,-0.932255,-4.433180,-1.835162,-6.203950,-7.529965,2.047652,3.807300,-1.964088,-6.658383,-8.445711,7.650265,4.522701,1.791331,0.895910,-3.089182,-0.723553,1.329927,-5.301901,-2.248464,-8.550887,4.399393,-1.685177,-3.150405,-3.420787,2.580920,0.208953,-9.857225,6.354707,6.543381,-1.513516,-2.794196,-4.562259,-0.104071,-5.259885,6.904812,4.254781,-2.084358,-5.624056,2.030569,7.546173,9.780483,8.314321,9.798102,-5.542167,8.068392,-2.208274,3.244816,-8.488420,-5.698010,2.128524,1.731680,-6.492653,-4.076828,-4.565726,2.009186,-1.785864,-4.841216,-7.661215,-5.962452,-8.720546,-6.126513,-4.096817,0.628425,5.830842,6.848320,5.522822,5.260364,-1.188070,-6.562775,5.331087,-6.751968], dtype = "float32")#candidate|1536|(96,)|const|float32
const_1537 = relay.const([[-8,-8,9,4,10,-6,3,-2,2,-1,1,5,-2,-4,2,4,6,1,-10,5,3,7,-3,2,6,-4,-9,8,10,2,9,2,-9,-3,7,1,1,-5,6,2,3,10,5,-3,-10,-4,2,8,-8,2,-1,3,-2,6,8,-3,-4,-2,1,-3,-2,-1,5,-8,10,-8,7,-3,-2,1,7,-10,5,7,9,8,-8,-6,10,-6,5,1,5,-3,3,6,5,10,-6,-7,-8]], dtype = "int32")#candidate|1537|(1, 91)|const|int32
var_1538 = relay.var("var_1538", dtype = "float32", shape = (312,))#candidate|1538|(312,)|var|float32
call_1535 = relay.TupleGetItem(func_662_call(relay.reshape(const_1536.astype('float32'), [8, 12]), relay.reshape(const_1537.astype('int32'), [91, 1]), relay.reshape(var_1538.astype('float32'), [312,]), ), 3)
call_1539 = relay.TupleGetItem(func_666_call(relay.reshape(const_1536.astype('float32'), [8, 12]), relay.reshape(const_1537.astype('int32'), [91, 1]), relay.reshape(var_1538.astype('float32'), [312,]), ), 3)
func_601_call = mod.get_global_var('func_601')
func_607_call = mutated_mod.get_global_var('func_607')
call_1543 = relay.TupleGetItem(func_601_call(relay.reshape(call_1535.astype('uint32'), [13, 8]), relay.reshape(call_1535.astype('uint32'), [13, 8]), relay.reshape(call_1529.astype('uint8'), [36,]), relay.reshape(var_1538.astype('float32'), [312,]), relay.reshape(var_1538.astype('float32'), [312,]), ), 8)
call_1544 = relay.TupleGetItem(func_607_call(relay.reshape(call_1535.astype('uint32'), [13, 8]), relay.reshape(call_1535.astype('uint32'), [13, 8]), relay.reshape(call_1529.astype('uint8'), [36,]), relay.reshape(var_1538.astype('float32'), [312,]), relay.reshape(var_1538.astype('float32'), [312,]), ), 8)
func_422_call = mod.get_global_var('func_422')
func_426_call = mutated_mod.get_global_var('func_426')
const_1547 = relay.const([5,-10,2,-4,9,7,5,-4,5,9,-2,-8,3,3,8,-5,-4,8,1,4,9,-1,-1,9,1,-5,4,7,3,-9,-4,8,9,5,7,-2,-3,5,8,-7,5,1,7,7,7,-5,-8,10,8,-9,-5,5,4,1,3,-2,-1,-8,-2,-10,-10,-10,5,-8,-3,10,-8,5,5,5,-3,-6,9,-3,-8,-9,-6,-7], dtype = "uint16")#candidate|1547|(78,)|const|uint16
call_1546 = relay.TupleGetItem(func_422_call(relay.reshape(const_1547.astype('uint16'), [13, 2, 3]), relay.reshape(const_1547.astype('uint16'), [13, 2, 3]), relay.reshape(const_1547.astype('bool'), [13, 2, 3]), ), 4)
call_1548 = relay.TupleGetItem(func_426_call(relay.reshape(const_1547.astype('uint16'), [13, 2, 3]), relay.reshape(const_1547.astype('uint16'), [13, 2, 3]), relay.reshape(const_1547.astype('bool'), [13, 2, 3]), ), 4)
output = relay.Tuple([call_1520,call_1529,var_1530,const_1531,call_1535,const_1536,const_1537,var_1538,call_1543,call_1546,const_1547,])
output2 = relay.Tuple([call_1521,call_1532,var_1530,const_1531,call_1539,const_1536,const_1537,var_1538,call_1544,call_1548,const_1547,])
func_1549 = relay.Function([var_1530,var_1538,], output)
mod['func_1549'] = func_1549
mod = relay.transform.InferType()(mod)
mutated_mod['func_1549'] = func_1549
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1549_call = mutated_mod.get_global_var('func_1549')
var_1551 = relay.var("var_1551", dtype = "uint8", shape = (36,))#candidate|1551|(36,)|var|uint8
var_1552 = relay.var("var_1552", dtype = "float32", shape = (312,))#candidate|1552|(312,)|var|float32
call_1550 = func_1549_call(var_1551,var_1552,)
output = call_1550
func_1553 = relay.Function([var_1551,var_1552,], output)
mutated_mod['func_1553'] = func_1553
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1590 = relay.var("var_1590", dtype = "bool", shape = ())#candidate|1590|()|var|bool
var_1591 = relay.var("var_1591", dtype = "bool", shape = (11, 1, 13))#candidate|1591|(11, 1, 13)|var|bool
bop_1592 = relay.logical_or(var_1590.astype('bool'), var_1591.astype('bool')) # shape=(11, 1, 13)
bop_1595 = relay.bitwise_xor(var_1591.astype('uint64'), var_1590.astype('uint64')) # shape=(11, 1, 13)
bop_1599 = relay.add(var_1590.astype('uint64'), bop_1592.astype('uint64')) # shape=(11, 1, 13)
bop_1610 = relay.greater_equal(bop_1595.astype('bool'), relay.reshape(bop_1599.astype('bool'), relay.shape_of(bop_1595))) # shape=(11, 1, 13)
var_1625 = relay.var("var_1625", dtype = "bool", shape = (11, 11, 13))#candidate|1625|(11, 11, 13)|var|bool
bop_1626 = relay.logical_or(bop_1610.astype('bool'), var_1625.astype('bool')) # shape=(11, 11, 13)
uop_1636 = relay.sigmoid(bop_1592.astype('float32')) # shape=(11, 1, 13)
var_1639 = relay.var("var_1639", dtype = "float32", shape = (11, 12, 13))#candidate|1639|(11, 12, 13)|var|float32
bop_1640 = relay.logical_and(uop_1636.astype('bool'), var_1639.astype('bool')) # shape=(11, 12, 13)
output = relay.Tuple([bop_1626,bop_1640,])
output2 = relay.Tuple([bop_1626,bop_1640,])
func_1653 = relay.Function([var_1590,var_1591,var_1625,var_1639,], output)
mod['func_1653'] = func_1653
mod = relay.transform.InferType()(mod)
mutated_mod['func_1653'] = func_1653
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1653_call = mutated_mod.get_global_var('func_1653')
var_1655 = relay.var("var_1655", dtype = "bool", shape = ())#candidate|1655|()|var|bool
var_1656 = relay.var("var_1656", dtype = "bool", shape = (11, 1, 13))#candidate|1656|(11, 1, 13)|var|bool
var_1657 = relay.var("var_1657", dtype = "bool", shape = (11, 11, 13))#candidate|1657|(11, 11, 13)|var|bool
var_1658 = relay.var("var_1658", dtype = "float32", shape = (11, 12, 13))#candidate|1658|(11, 12, 13)|var|float32
call_1654 = func_1653_call(var_1655,var_1656,var_1657,var_1658,)
output = call_1654
func_1659 = relay.Function([var_1655,var_1656,var_1657,var_1658,], output)
mutated_mod['func_1659'] = func_1659
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1666 = relay.const([[7,-10,-5,1,-6,6,1,2,-2,-3,-6,3,9,-10],[5,4,-6,-6,1,8,10,-1,-3,-8,-5,1,10,-7],[-9,1,9,-8,10,-9,-1,-10,-10,-4,1,-9,-3,-10],[-2,3,-4,-9,-10,9,1,2,-4,5,-1,-6,-10,-1],[-5,5,-7,-1,3,-5,7,4,3,-3,8,2,8,-4],[-5,1,-7,9,5,7,2,-4,8,-8,-3,10,1,-3],[-2,-9,-3,6,9,7,7,-3,-8,7,-4,-8,8,-10]], dtype = "int16")#candidate|1666|(7, 14)|const|int16
var_1667 = relay.var("var_1667", dtype = "int16", shape = (7, 14))#candidate|1667|(7, 14)|var|int16
bop_1668 = relay.bitwise_xor(const_1666.astype('int16'), relay.reshape(var_1667.astype('int16'), relay.shape_of(const_1666))) # shape=(7, 14)
output = relay.Tuple([bop_1668,])
output2 = relay.Tuple([bop_1668,])
func_1687 = relay.Function([var_1667,], output)
mod['func_1687'] = func_1687
mod = relay.transform.InferType()(mod)
var_1688 = relay.var("var_1688", dtype = "int16", shape = (7, 14))#candidate|1688|(7, 14)|var|int16
output = func_1687(var_1688)
func_1689 = relay.Function([var_1688], output)
mutated_mod['func_1689'] = func_1689
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1380_call = mod.get_global_var('func_1380')
func_1381_call = mutated_mod.get_global_var('func_1381')
call_1696 = func_1380_call()
call_1697 = func_1380_call()
output = call_1696
output2 = call_1697
func_1701 = relay.Function([], output)
mod['func_1701'] = func_1701
mod = relay.transform.InferType()(mod)
output = func_1701()
func_1702 = relay.Function([], output)
mutated_mod['func_1702'] = func_1702
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1762 = relay.const([[[5.769016,-5.711431,7.834058,-6.682572,-6.933321,3.835232,-4.544695,-0.056349],[-5.579279,1.956373,-3.487866,-7.527339,-9.090562,6.380567,9.575089,5.489777],[-8.269820,2.652904,-9.895673,-0.375658,-9.404371,7.707668,1.633789,-0.434479],[-4.113317,9.582518,-1.709094,-0.004368,-1.375079,3.622841,7.969040,0.999112],[0.604656,7.179927,-6.278159,-8.787303,4.330367,1.278054,9.468456,0.361641],[-8.899417,-6.498759,-9.533892,5.622128,-7.485919,-0.424177,-6.763272,5.520024],[-4.123440,4.812785,9.674666,-7.169522,4.781381,5.393708,3.832962,-9.450432],[6.076394,-2.631113,-7.744403,6.726249,6.861800,-1.099893,4.494989,7.800604],[-6.379279,-4.133728,-9.462976,-7.418675,-1.749383,-0.206125,-9.458226,9.741399],[0.412778,9.315956,-2.199905,-4.025201,0.714603,7.112841,3.786945,-1.235209]],[[2.832700,7.408903,5.810048,1.341599,3.257364,-9.270273,9.922154,8.655142],[2.712493,-9.028016,0.360929,9.686114,6.617838,2.411712,5.543942,-2.401617],[-2.710065,-3.071521,4.584361,-9.795905,2.749042,4.476615,-3.091893,6.710594],[6.119464,4.982050,7.474841,2.127838,4.543761,8.299478,7.927194,5.553855],[3.894996,9.901083,-7.281692,8.278817,-5.876305,-9.555179,3.831052,2.643290],[3.416387,-1.981294,8.341207,9.198747,-3.351798,-9.755666,-5.682284,-9.404066],[9.791034,-2.700239,3.598918,-7.926069,-9.761048,-1.911274,-8.426282,3.664787],[0.529463,-9.810247,3.848717,3.631622,-0.642163,7.747793,-9.996273,-2.855477],[-6.717440,7.768071,-1.181351,4.420662,-8.455902,6.050979,7.483685,-4.225045],[-8.764676,-1.053063,-8.360754,7.135036,2.173839,6.872294,-8.464555,4.014237]],[[-2.360144,-4.460280,-1.278998,-9.883355,-4.565107,-8.366008,-8.921954,-9.104752],[1.360185,-0.272306,-0.805760,2.145060,3.895428,5.772271,-9.256710,-6.163427],[1.290592,-7.509374,-3.998713,-9.821543,-5.881779,8.326139,7.191042,3.654967],[-4.451193,-0.503918,7.882003,-8.111185,-0.685285,9.389235,-7.533860,-0.243495],[-3.896395,-7.178368,6.275887,1.874009,1.199683,-9.941578,-8.325111,0.047751],[7.094306,-2.723314,5.123678,5.277163,-2.547094,8.402660,-3.712842,6.497709],[-9.382052,-0.843214,-5.123424,2.889821,-1.624439,5.390077,9.496224,9.075308],[-2.689363,2.826182,-3.352960,-4.125665,2.485884,8.624739,1.405965,9.321633],[-6.115959,1.317533,8.442114,-1.481713,-1.840923,6.003838,2.772101,-1.193855],[-3.525355,0.975815,7.470766,5.813501,7.168789,-5.085363,7.012098,-5.978910]],[[0.322696,5.110710,-1.589620,-7.857701,5.577369,-6.148374,-5.305729,-4.507161],[6.880984,1.301133,4.789025,2.164462,-8.985766,9.795020,3.841698,7.080174],[9.534368,-5.394853,1.127445,-1.199956,-1.827959,8.391718,-3.658565,1.767745],[7.643835,-6.603920,7.593206,-8.820298,1.827800,-9.528707,9.970890,-3.260843],[-7.136017,2.387720,3.366941,-0.392905,-5.147979,6.340734,1.971786,-7.449339],[0.547114,-8.878001,5.717342,-5.594458,-6.610414,6.628410,0.844805,1.498269],[-3.974639,9.424207,-7.091908,2.812024,4.865777,1.332057,1.160341,0.475921],[-3.820565,-7.803877,-9.505010,7.261836,5.616176,5.003940,-6.727915,-0.021226],[4.523937,-1.872877,9.062428,-9.424129,-6.795562,1.835749,-1.520361,3.508870],[-9.208100,-6.481509,-1.928276,-8.873522,-9.155640,-2.151026,-6.694999,-0.915435]],[[-6.487469,5.490385,-8.865118,-8.525931,-7.081150,4.371215,3.562637,1.714151],[-5.885706,-0.769156,3.177032,-3.448747,-4.157072,-6.921702,5.750962,-6.550957],[-1.968033,8.406506,-8.395192,1.109356,3.052618,9.740737,2.382810,8.371710],[-9.570732,3.689580,-5.674628,8.509298,-2.260914,1.174912,-7.419787,-3.835588],[3.285418,-0.038154,7.207891,-3.668978,-0.574847,7.282443,5.847546,5.720833],[3.071520,1.950759,-0.087285,-8.440609,1.536296,-7.132415,-6.195043,-0.133322],[-2.671865,9.724642,-1.905572,-7.525203,9.028831,4.433928,-0.343770,-6.904327],[2.292513,4.305902,-9.632488,-2.989609,8.176999,-8.898066,0.725681,-5.897364],[1.962643,-8.176370,5.640624,-2.575162,-0.554266,7.967788,-5.329298,-6.478461],[-2.017529,1.719980,-9.329103,-4.653090,-7.007967,0.160768,8.687557,8.190853]],[[7.448894,-4.766487,-6.970768,4.494040,-6.433739,-2.222901,-3.225437,-7.634575],[-0.447125,8.391754,-3.774890,4.134189,-1.760304,-8.190569,-1.965724,-4.358107],[3.831947,-8.049977,5.308025,4.516373,1.044863,-6.968889,-9.340547,-6.163019],[-7.787167,8.400896,4.318583,0.006392,2.977674,4.931987,5.585796,0.372864],[-6.973425,5.275074,8.914001,-7.311836,-8.857171,-4.119669,0.805963,0.010178],[9.799579,-5.842746,3.905521,-7.713321,-2.868273,-0.187402,-2.705796,-4.386199],[-8.033320,-9.370734,-5.326281,-0.405485,-7.120305,-1.256434,8.215643,8.652231],[-3.935870,7.961251,-2.138025,4.216721,5.175050,-6.391710,0.943029,-0.461338],[-9.070642,-2.027801,1.197104,8.401895,1.700980,-6.233715,-1.407475,5.268144],[7.547084,-4.348861,0.138776,-5.515922,-8.432156,-5.486372,-4.044678,5.962579]],[[4.407259,8.986070,-4.178249,-7.086449,-2.000290,-0.760282,8.274193,-0.373262],[-9.132059,-8.552858,-5.451281,-2.453323,4.660686,-8.265508,6.489421,7.586857],[2.527286,-9.686767,5.857543,-2.714730,5.118531,-3.245939,-2.814309,3.032562],[-8.203281,-6.212802,8.288982,4.828200,3.810848,-5.716309,5.346103,-7.707614],[-5.364850,-0.650718,3.584618,-6.507637,3.202035,8.809014,8.422028,-2.199285],[-0.634921,-6.044379,5.083064,5.438670,6.792771,8.453082,1.179353,-3.949684],[-1.653914,7.808998,8.752837,-9.925955,0.157067,-5.695396,9.551709,-3.397376],[-6.478688,0.893057,-4.028309,-9.302354,-7.076767,-7.277414,3.445851,4.098729],[8.203191,-8.366559,-0.116498,6.291631,-3.297597,-7.220595,-7.418460,-5.019735],[3.928628,-9.115409,-4.111359,-0.775165,-9.675387,-7.759618,4.664282,-4.927437]],[[-3.604930,2.926770,6.906406,-4.959453,-6.110463,8.582492,3.492043,8.617897],[4.259402,-5.125949,6.251955,-2.985935,8.186009,2.080165,8.873649,3.531653],[-1.233661,-1.587516,5.211598,-7.869385,-7.881854,-6.652375,0.063023,-9.635194],[3.879212,-7.547435,6.071827,2.102114,2.498242,8.315015,-9.451829,-8.109957],[7.741816,-7.906644,-2.793679,4.879546,1.222565,-8.781394,5.390669,0.366027],[-5.121571,4.667295,0.050963,-5.071830,-4.610693,5.908768,1.252742,1.500012],[-3.797024,3.199161,-9.787946,8.154740,8.908808,-1.233940,7.321997,9.737936],[3.697453,2.877277,6.787471,5.211036,-6.888744,-6.840853,6.566849,-8.862301],[-9.962762,-7.370639,-1.286292,-2.608945,4.474525,0.122604,-2.677007,-4.129448],[1.439428,-0.913291,-2.698487,-3.810265,-1.657131,-4.990546,4.461136,-3.304437]],[[-5.911136,-2.734983,-4.806622,-5.749166,-7.113277,-1.450188,6.278460,-6.197350],[6.272480,2.837917,1.493862,-1.312014,6.046459,9.141184,7.228845,-0.094143],[-9.023924,0.987412,2.505096,-4.310549,8.004867,-0.778080,-1.346461,8.102013],[-3.444128,3.791513,0.719368,6.261637,-4.635687,1.997064,-0.467047,9.598729],[-4.997049,-9.423664,7.853446,3.653099,6.159140,2.225292,2.592004,7.656311],[6.931513,5.482183,-7.953783,5.082284,2.530691,4.165265,-9.279074,-1.766882],[3.639188,-2.928605,4.511874,0.205719,-1.590337,-6.417480,8.655483,3.840653],[-5.627257,9.972487,2.719220,8.660709,2.265813,1.796709,-0.950275,0.504102],[-1.933879,-0.261770,5.359516,-9.126408,-9.812163,9.127084,5.134971,-1.204138],[0.826536,-3.060190,1.883463,2.142219,-3.479723,-8.679489,-7.978539,-4.443161]],[[-0.097606,2.783517,-8.660029,1.915552,-5.806427,1.579319,8.513427,-9.073439],[5.640320,2.937637,-3.199130,-8.473322,3.230299,5.442072,-1.986037,-6.959710],[-9.251459,-8.760726,-6.779889,4.418320,5.693149,-9.291785,9.999909,6.350600],[2.798183,1.531046,-4.922578,0.683257,-5.376352,8.166337,-7.433699,0.444353],[3.295624,-2.375670,0.302677,8.645178,-5.350285,9.016316,-3.898790,-6.529510],[-1.569199,0.827883,9.055900,-2.358680,-2.572895,-3.654391,-0.660817,0.465003],[-0.467919,3.844521,-5.047230,-7.801148,-0.779421,-3.592705,-7.852876,-3.888466],[-4.236289,8.385876,-6.641556,5.828863,-0.533390,-1.603040,-3.118334,-7.138891],[9.164304,-4.081271,-1.359006,-0.010617,6.023351,4.652095,3.215932,-8.528543],[-4.962817,-5.785150,9.713986,0.269355,2.464894,-0.168957,6.655819,-5.984715]],[[-5.822136,-9.305552,3.079042,4.226132,8.988835,0.244747,4.123230,-7.881241],[6.242323,-8.848830,9.430183,-4.046093,0.925247,-6.644001,-2.620162,5.506100],[2.513052,5.749382,-8.113541,2.748946,-9.314221,0.543328,-9.845755,5.420019],[-5.887894,8.116233,6.500929,-9.644400,-2.904345,2.488465,-0.183846,-6.870126],[5.819274,-4.151863,-0.444122,7.044277,-5.902980,-3.243692,0.085137,6.539449],[-1.159355,8.641213,-2.486743,-9.931933,-0.510488,7.499512,1.056769,-6.630815],[-4.135358,-7.716275,4.536604,7.536593,-7.272466,-8.471482,5.579651,9.654066],[-2.487663,1.020014,5.466961,-5.306719,0.582161,-4.177876,8.533242,0.096384],[4.532052,-8.204191,-5.144395,6.844296,-9.163251,-8.827901,4.238012,0.305326],[-6.865229,-8.821280,-0.158365,3.578223,1.584128,-5.587962,-0.580790,-2.919983]],[[-4.135484,-8.864394,-4.483302,6.394527,-1.557916,-8.062100,-9.853774,1.225588],[-5.663424,-9.446730,-7.350407,-6.967315,6.317345,-9.829935,9.887429,5.887982],[-6.875847,-2.424563,8.581070,1.011514,0.348591,7.223505,8.887399,-5.170180],[6.157059,-2.611855,-0.949996,6.640773,-3.113161,4.710498,-0.626737,8.549614],[7.098067,9.137144,9.969640,-6.396110,-8.146080,-0.794444,-8.206565,0.418233],[7.781953,-0.664419,-4.328118,6.666657,8.961100,-0.259571,-6.506056,-6.238755],[2.615335,-5.666992,-2.384540,7.864862,6.968548,-5.294795,-2.604531,0.248103],[-8.262423,3.827151,3.290011,1.599939,-8.182744,-3.243400,4.991233,-5.439394],[-0.465960,5.539628,8.996650,2.272030,4.599263,-1.815877,3.188171,7.776413],[1.362687,-7.417344,-3.532108,-6.323551,-4.942722,5.094776,8.508626,-3.976123]],[[-7.518805,1.863413,2.769789,-4.035924,-7.943891,-6.172261,-5.669256,4.566070],[9.819071,-5.565851,5.379997,-1.310066,7.745306,3.129462,5.308248,-5.629798],[-9.884709,4.757533,-1.825955,-6.033405,-2.742575,-9.954418,-1.620218,4.592513],[-2.177002,9.109500,-3.449545,4.095553,3.879916,-5.882294,-4.981137,2.769341],[-8.858703,-4.398138,-5.446922,-1.668766,-6.172700,-1.993643,-1.182167,-2.801458],[0.075183,9.461010,9.678336,-0.831574,2.471907,9.779915,7.482260,-3.868691],[5.200885,9.175624,5.852913,-0.851717,-5.924468,7.662980,-8.161135,5.126199],[3.849613,0.953526,4.481810,-1.561228,5.763478,7.495219,-5.725436,-4.826561],[-9.659983,7.188353,6.540031,6.349564,-7.807137,9.886808,-7.806947,0.986091],[8.574708,-6.111307,-6.457976,2.994427,-3.045312,-9.197805,0.623511,-3.216013]],[[-7.690424,4.833139,-5.545473,-6.365039,8.770193,2.491043,-4.728325,-2.401309],[-1.043236,3.949134,8.880622,-9.895455,8.534296,-2.903379,2.933958,-5.015470],[-4.102304,-1.452111,5.133548,-3.102208,-6.833069,-2.871584,-7.809738,-0.063538],[-2.444986,2.141564,-1.147147,7.137929,-8.573512,-6.201275,2.753586,-8.376617],[6.491224,-9.277263,1.278005,9.484609,-2.745631,-5.671698,-4.050576,3.568937],[-4.970190,-4.148011,-7.322539,-1.063784,6.309745,7.949034,-5.802869,2.339350],[-5.275977,-8.190223,-6.717715,0.809589,6.440728,6.328139,-1.221319,-8.584705],[3.729487,3.971726,5.921231,-8.160130,2.099907,7.789395,6.489146,8.368795],[-0.648823,1.608639,2.951995,0.736469,7.358680,9.470849,-8.880555,-7.443660],[-6.126048,5.464672,-4.405773,9.689665,4.470568,5.483048,6.950804,-3.539636]],[[-1.126932,-6.282114,-4.265735,-1.394050,-6.641384,9.412938,-2.570205,-7.791525],[-4.847891,-3.703144,5.314963,-7.418566,-5.642096,-6.763159,-2.741745,3.703171],[-6.306121,-0.510746,3.010040,-9.892548,-5.968685,-2.485891,-2.030706,-7.574945],[5.860270,6.438803,-7.474396,3.204624,3.453723,-6.434676,5.984965,-4.349642],[1.284988,-4.268017,9.668850,-6.186055,0.742602,-7.202309,8.365523,-0.299925],[1.236505,2.807683,0.918205,-2.870096,-8.791393,-5.244615,-2.895272,-1.349656],[3.563265,2.596533,4.744842,6.158361,-6.076858,-5.159853,-4.352611,-0.988307],[-1.221840,7.549224,9.793544,-5.604022,-3.642353,3.329813,5.217800,-1.459976],[-9.947460,-6.219902,5.031098,-5.384345,3.398895,6.746023,7.049820,-9.944320],[9.232985,1.295196,2.204923,1.310293,-1.980206,2.072168,-7.751887,6.503643]]], dtype = "float64")#candidate|1762|(15, 10, 8)|const|float64
uop_1763 = relay.sqrt(const_1762.astype('float64')) # shape=(15, 10, 8)
uop_1765 = relay.acosh(uop_1763.astype('float64')) # shape=(15, 10, 8)
output = uop_1765
output2 = uop_1765
func_1770 = relay.Function([], output)
mod['func_1770'] = func_1770
mod = relay.transform.InferType()(mod)
output = func_1770()
func_1771 = relay.Function([], output)
mutated_mod['func_1771'] = func_1771
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1779 = relay.var("var_1779", dtype = "float64", shape = (1, 6))#candidate|1779|(1, 6)|var|float64
uop_1780 = relay.atan(var_1779.astype('float64')) # shape=(1, 6)
bop_1792 = relay.bitwise_xor(uop_1780.astype('uint16'), relay.reshape(var_1779.astype('uint16'), relay.shape_of(uop_1780))) # shape=(1, 6)
output = relay.Tuple([bop_1792,])
output2 = relay.Tuple([bop_1792,])
func_1801 = relay.Function([var_1779,], output)
mod['func_1801'] = func_1801
mod = relay.transform.InferType()(mod)
mutated_mod['func_1801'] = func_1801
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1802 = relay.var("var_1802", dtype = "float64", shape = (1, 6))#candidate|1802|(1, 6)|var|float64
func_1801_call = mutated_mod.get_global_var('func_1801')
call_1803 = func_1801_call(var_1802)
output = call_1803
func_1804 = relay.Function([var_1802], output)
mutated_mod['func_1804'] = func_1804
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1814 = relay.var("var_1814", dtype = "float64", shape = (12, 6))#candidate|1814|(12, 6)|var|float64
var_1815 = relay.var("var_1815", dtype = "float64", shape = (12, 6))#candidate|1815|(12, 6)|var|float64
bop_1816 = relay.maximum(var_1814.astype('float64'), relay.reshape(var_1815.astype('float64'), relay.shape_of(var_1814))) # shape=(12, 6)
uop_1826 = relay.asin(bop_1816.astype('float64')) # shape=(12, 6)
uop_1872 = relay.atan(bop_1816.astype('float32')) # shape=(12, 6)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_1875 = func_1256_call()
call_1876 = func_1256_call()
output = relay.Tuple([uop_1826,uop_1872,call_1875,])
output2 = relay.Tuple([uop_1826,uop_1872,call_1876,])
func_1877 = relay.Function([var_1814,var_1815,], output)
mod['func_1877'] = func_1877
mod = relay.transform.InferType()(mod)
mutated_mod['func_1877'] = func_1877
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1877_call = mutated_mod.get_global_var('func_1877')
var_1879 = relay.var("var_1879", dtype = "float64", shape = (12, 6))#candidate|1879|(12, 6)|var|float64
var_1880 = relay.var("var_1880", dtype = "float64", shape = (12, 6))#candidate|1880|(12, 6)|var|float64
call_1878 = func_1877_call(var_1879,var_1880,)
output = call_1878
func_1881 = relay.Function([var_1879,var_1880,], output)
mutated_mod['func_1881'] = func_1881
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1894 = relay.const([[[-4.642422,7.213426,-3.129718,-7.552589,-8.713954,2.765445,-4.275323,3.114923,-9.726151,9.223632,-3.877780],[0.374691,0.650404,-2.721558,-1.120960,-0.329631,-4.727385,-9.453112,-6.757834,-9.778483,-2.212566,-8.429193],[-0.302087,-6.627596,6.052699,6.387382,9.338413,9.170702,-1.125095,-0.220903,-9.638186,-1.328880,-0.875409],[4.167615,6.172547,3.857572,-4.657001,-3.654603,-1.985444,8.440462,-1.078529,6.486505,3.664974,9.995655],[-0.673774,-0.992615,-9.302279,4.856239,9.605840,2.362584,-8.191784,3.158842,-2.625713,8.389450,-7.583173],[-6.848186,-8.100360,3.539827,0.310970,2.591127,-0.987524,-2.337382,-8.530563,2.473992,-8.354207,-3.631531],[-6.441079,6.260080,0.654670,5.949372,6.934442,5.309174,2.120185,0.707331,5.002870,-1.298044,-0.781530],[-5.377774,-1.032392,6.812159,-9.172984,-5.893821,5.180256,-5.399976,-0.039581,-6.490747,2.927929,5.058550],[-3.789726,9.920671,-5.553154,8.643074,-0.569187,5.037040,-3.993377,-9.044849,7.362788,8.352380,-8.855049],[3.615150,3.713347,8.420657,4.538935,5.655916,3.465063,2.090327,-4.190876,3.279109,-4.673949,5.771444],[0.965578,-4.926239,-8.947326,2.662480,1.579265,-4.324306,-8.190545,7.090690,-9.682194,-3.622675,-8.754136],[3.381510,-1.019165,4.900192,-5.869081,8.846492,-1.222292,-9.009370,3.971967,-8.730939,-7.705560,6.027530],[-5.677459,5.006608,0.124540,-3.463070,7.036203,-2.228198,-9.453819,2.949086,9.799928,-0.473858,-2.214555],[-5.567893,-6.010695,-2.773213,9.164482,-0.907801,-6.613163,6.156606,1.490278,5.874448,1.098483,-9.996258],[-8.676676,-6.362126,1.823556,6.312315,4.863227,8.401574,-8.763662,1.326813,-3.054794,4.898835,8.457507],[-7.581635,1.944900,8.616776,-8.147978,-2.354212,4.300350,-1.975417,-8.903692,-3.355109,2.545594,1.330599]],[[4.124538,-5.126239,-7.641245,5.025162,-5.157538,-1.356147,3.427574,-7.670476,-5.459272,-3.538397,-6.185878],[-5.720428,4.093773,-7.582154,-2.643399,-5.413366,-7.969494,3.645684,8.924146,-0.715810,8.137247,3.629216],[6.734290,-0.281680,-9.086720,3.115615,2.354288,0.583362,3.158016,5.376075,-1.727164,6.841798,-4.411563],[4.263746,6.106925,3.808272,9.038907,-9.635582,3.563067,-4.822558,9.327960,-5.306177,8.109558,9.950533],[2.225947,-0.825750,0.067335,9.044762,5.735280,5.958759,-8.630488,3.891287,2.018920,-7.456925,-4.763313],[-9.631434,1.439924,-6.294097,6.717020,0.682687,-2.905265,-1.002019,3.338629,-1.876846,-5.572301,-1.500810],[-5.725905,-0.900686,1.320037,-3.163985,3.293554,7.507475,6.656495,4.689633,-6.140322,-1.845445,7.038933],[-3.860189,-6.425817,1.459631,-5.324090,4.274765,6.668787,-8.570479,2.938403,-4.280030,9.759281,0.906753],[9.435033,0.321519,7.371192,-0.009872,-0.910135,5.424747,6.804832,-2.004231,-3.583673,7.833477,7.416262],[-1.260745,-5.054510,-1.462465,9.334987,7.072922,-3.626702,-8.363637,-8.641479,-9.048324,2.872666,3.023006],[-9.316426,-7.314274,8.571126,-6.046195,4.071955,7.455210,5.160182,2.396946,7.262058,1.855048,-0.167088],[1.110342,9.892318,2.288785,-9.387478,4.161147,-8.641419,3.761132,3.685896,1.046232,5.224493,4.983721],[-1.966616,6.697553,0.192241,9.321318,-2.412488,-5.549067,7.820002,-7.180206,-1.887629,-3.484623,5.729687],[-4.132378,-9.762812,2.866491,8.396640,-8.819314,1.657539,2.760280,6.458798,-5.008922,-9.139524,1.263247],[9.031643,8.429796,9.428889,-6.249493,7.187127,-8.969079,-8.324199,6.708103,-7.908653,8.227376,7.412066],[-7.873641,-8.931691,-2.460263,3.668091,-7.376888,8.601519,-2.414740,8.568866,0.354493,2.226484,-9.979558]],[[2.575518,1.487585,-1.037759,-2.482606,-8.318750,-8.075908,5.594949,6.684452,-8.295808,8.299085,0.316263],[-2.993679,6.119695,8.843387,2.677400,-4.369570,-8.455522,-1.672259,3.668618,-6.222587,3.561062,-6.334452],[-6.849384,7.750916,1.668692,-6.675264,-0.303317,1.209257,9.978928,1.997088,7.160283,3.531914,6.783736],[-0.934723,-3.517047,-4.078464,7.985723,1.510678,-5.284090,-0.324408,9.140012,-2.668777,2.510707,2.625215],[0.394799,-3.965695,2.268251,-3.756335,1.206371,6.469517,8.144796,-9.597831,-8.104229,-8.576948,-6.069394],[-4.864938,-5.567445,-6.276757,-8.944545,-2.633736,-8.322150,-5.296772,-4.478945,-9.429723,8.295245,-5.387406],[-6.120092,4.558735,-1.605874,8.313757,-0.294839,-2.149418,0.700796,6.489393,0.520135,2.731346,-3.004109],[9.807034,-4.667280,-7.356603,6.395929,-8.574145,0.515465,-9.560574,-4.880465,4.398123,9.448706,0.288923],[-1.907566,-0.598172,-1.405353,9.827701,-7.351774,3.786833,-3.252128,-5.225451,-5.045829,9.915058,1.932761],[4.293380,-8.094906,-4.431152,8.156164,-2.845727,-9.948682,4.283415,-3.572798,4.202646,4.841348,0.791444],[5.153134,0.055182,-7.808643,7.241163,-1.906689,8.697091,9.161574,-4.216433,7.753971,-8.810977,6.115137],[-7.023705,-0.666084,-0.605094,1.997035,-5.865302,-2.851991,-4.696386,-6.774099,-0.134036,3.191216,9.800069],[5.321391,3.371169,-8.520932,7.099448,3.451293,0.026714,5.894803,3.672155,6.295303,3.692486,1.754512],[7.352135,-4.122923,-2.107748,2.142167,-9.994664,4.468198,-3.161443,-8.013585,7.580960,1.396835,5.245442],[8.285820,2.880111,-0.179842,-8.464412,3.059102,-2.092942,4.417073,5.816379,-8.119618,7.734238,7.732968],[-3.061368,7.945824,0.854101,-7.497210,-6.462648,7.125946,9.283115,-7.034056,7.516328,3.842394,7.374290]],[[-2.320352,9.203586,0.441608,-9.454550,-2.601611,-7.039166,-1.903075,3.698788,-7.296962,-9.681442,6.694989],[-0.046192,1.548656,2.440797,0.221944,3.455011,7.480311,-3.241883,-0.733636,1.478623,8.221221,-4.388285],[7.345419,5.473847,6.010527,8.138889,3.741643,8.514202,3.327083,0.700166,1.668134,-1.776736,8.210015],[1.594873,0.422801,5.008878,-8.715958,8.458806,1.311726,-0.087389,-8.034571,8.899529,6.418691,8.778526],[-0.657240,6.615758,5.763068,-8.807402,-3.160547,-6.665432,7.226587,5.210099,-1.435456,5.092534,3.706674],[-6.394030,-0.673661,0.472419,-3.674553,-6.868938,6.463394,3.069195,-3.665829,8.074473,5.085669,-2.252141],[5.694443,6.704739,-4.334822,6.412450,-2.668034,-9.774140,-9.613028,-9.618390,-1.265873,8.184564,-1.120770],[-2.029006,-9.607185,7.863185,-9.122981,-7.754851,8.145462,9.480436,9.545612,4.871736,-5.117899,-6.400345],[3.802912,-0.217679,1.611009,5.835821,-1.251168,-1.138862,-0.908681,-2.785705,-5.906755,-6.047323,-4.868380],[8.721413,0.417110,1.028405,-9.323828,-2.031904,-3.424590,8.886290,-1.658017,9.735563,9.364086,4.434821],[-7.994090,-4.085206,0.173256,-0.490945,-0.587699,5.629294,0.285218,7.865362,-2.052019,2.754177,-5.229786],[-8.476249,-3.918425,4.593385,-8.384467,8.025155,7.725206,5.486349,-3.980790,-0.818300,5.674141,-4.406615],[9.750693,0.174692,0.120503,-9.249317,8.022248,-9.959933,7.034417,9.623977,6.561921,-8.275540,-2.643681],[7.515670,1.971870,1.297937,6.007349,2.568830,-6.861585,-8.288971,-8.301251,4.960399,8.938259,4.814051],[-3.555835,-2.351722,4.478731,-5.123871,-4.688071,-9.083378,5.056869,8.120571,-3.636372,4.098446,1.015628],[-8.132508,1.143012,8.123879,-1.727275,9.223329,-2.059945,-6.307243,-7.065962,3.795239,-2.850657,-2.920522]],[[-7.775291,-5.813514,3.329575,9.220660,1.099015,-3.374776,9.336140,2.022287,-1.588930,-2.915731,3.649592],[-1.906289,9.082299,-5.735539,0.612414,-8.339579,-8.711470,-2.514954,-0.208333,-3.675209,-6.087201,-5.551389],[2.374229,1.874545,-4.294251,-7.353782,-1.641409,1.593373,-3.451958,9.202360,8.150850,8.009593,-2.299383],[7.801269,5.254803,-7.619142,-7.647864,-1.065281,-8.209199,8.839194,2.717097,-2.166923,-3.844426,-2.597279],[-0.263621,-5.028847,1.245435,9.249885,-3.913261,-8.570271,0.538955,9.411591,-1.540044,7.550403,3.764508],[6.122512,-9.937137,-1.704311,-7.686020,6.791059,-8.873252,-5.802061,-6.265674,2.153146,-2.786319,-2.713983],[-6.059993,8.117642,3.351838,9.180551,9.257709,-3.769302,4.839468,-1.226309,5.598335,-4.817869,6.389393],[9.760304,-3.043115,-1.504764,-0.084323,-6.381423,-8.367495,3.911589,-9.863962,-1.281489,-3.236442,5.844206],[5.951968,4.616544,-9.799836,-9.672985,-3.206439,1.529047,-2.473636,8.314413,-3.863681,7.986377,5.875409],[5.708245,-0.363572,9.445962,-4.467699,-6.826526,-5.760555,-2.534432,3.942938,-6.995033,-8.174438,-7.856812],[-4.271373,6.261808,-5.185008,7.960607,-7.753819,8.469670,5.565464,0.649201,-6.453622,7.592109,6.378257],[2.538974,1.304746,-6.705771,5.044405,-1.216286,-6.624488,-9.948855,8.417347,-0.143941,-1.759011,-6.369382],[-0.921998,3.686976,0.471568,9.877536,-4.583514,2.333727,-1.789241,5.933459,-8.901022,6.878409,-9.291577],[4.107598,-9.343420,0.429983,-8.431222,-6.562893,1.648450,0.171546,-6.501436,3.060780,0.643644,-7.055181],[-9.986648,1.512989,-4.030082,7.029295,-2.295404,-9.385005,-8.740253,-1.061350,2.343942,8.282707,-1.827716],[0.236544,3.298017,-1.323711,6.610004,-1.457051,-1.445109,-9.575204,1.814016,2.834666,4.846061,-6.649639]],[[7.642466,-9.717819,3.263181,9.229033,-1.443697,-7.537516,5.249560,-7.183954,-3.986952,-2.146237,-4.340179],[2.921209,-4.048649,6.153383,-2.099341,5.033725,-8.825537,-6.080694,1.635578,-7.683663,-8.321764,9.608931],[6.958829,-7.857562,3.170114,-7.606627,6.276197,3.777119,1.560411,1.940306,-9.178580,8.989614,-4.137746],[-2.509387,-1.322998,1.206232,0.478682,-6.062805,-2.533813,-6.489938,-2.691921,4.442801,-4.021706,3.300982],[-4.073225,7.129220,-9.242199,0.528781,-7.179352,-2.546271,-4.956865,4.693295,4.667950,-2.561675,-2.030333],[1.750905,-0.549894,4.117720,6.973373,-2.129858,6.646456,-7.677214,-6.496414,7.706017,8.452945,9.278766],[2.906946,9.653672,1.069081,9.239905,5.665216,-2.347066,0.982449,-2.366215,-4.062299,-6.111725,1.602774],[6.149996,8.108708,-6.778839,9.753855,-0.041429,9.221745,2.969608,-1.102885,-8.156805,3.614944,9.317891],[-1.271067,1.116269,-7.101002,9.761598,-7.855256,1.355532,6.023936,-3.230672,-0.937077,1.012658,-6.012965],[-6.445799,5.359887,-6.356587,0.315156,-1.425117,-2.896095,-3.717575,1.223646,-5.849846,3.206388,3.979821],[6.339830,2.252647,2.640687,4.741315,7.401399,-9.275976,9.447023,0.302250,-9.455250,-8.210996,1.305109],[6.381667,-1.571577,-6.878670,7.683137,-3.449754,9.862514,-3.874052,7.836227,-7.448804,8.111375,9.587553],[-0.120099,5.100663,-5.062874,3.100903,8.807397,2.715104,-1.637283,9.152355,6.337024,-2.630679,8.854156],[9.822082,6.761631,6.443971,-1.377195,7.678298,0.501420,2.253208,-3.743314,5.552001,4.055726,-1.017695],[-2.679340,-9.043142,2.419672,-2.591866,4.382353,8.460046,0.625845,-3.629892,-7.821330,-7.260673,-0.243571],[0.389732,-4.834990,-4.200582,7.980459,3.294976,-3.611137,-6.914049,7.856049,-3.937145,-4.331092,1.943870]],[[-3.802932,-4.211084,6.127089,-6.808594,-4.041920,-6.307935,-6.868004,2.618993,0.658103,6.644256,-7.507487],[-6.968638,-4.963024,-5.208564,-7.450699,7.080841,4.790186,5.466556,-8.631411,5.000933,-1.022803,-8.734675],[-5.731220,-1.608683,-4.093860,-1.302613,6.965897,-4.178684,8.506844,4.373541,-8.493129,3.708332,-1.441511],[-7.897629,-9.050614,0.796305,-6.733639,-7.866877,3.996372,-6.604754,-3.446101,-2.681147,5.748832,-8.644906],[-2.276625,3.716623,-9.912039,-5.128920,7.681058,8.029984,-8.806266,-8.282822,-6.899457,-3.300160,-0.415281],[-4.156717,1.359747,-5.741507,1.160573,5.501738,-1.956697,1.867129,4.573877,5.514864,-7.997695,6.679518],[-4.037061,7.405278,-4.992957,-1.367277,8.201302,8.629226,-7.535971,0.876083,-7.746709,4.798251,5.966662],[5.209565,-3.982084,-7.963849,-3.831936,-5.652075,2.744239,0.063903,6.788472,-9.734326,-8.305132,2.356959],[-9.361421,2.263434,3.720445,6.756652,0.387172,-7.665428,-4.029949,-5.026934,-4.056490,7.731992,7.969640],[6.228556,-7.166468,2.842646,1.078014,1.354643,9.553079,9.865361,1.736345,-8.611281,3.541547,1.819769],[3.455539,-4.535986,-0.015866,1.569334,3.084700,-5.749463,9.538202,-3.685726,-2.457906,-3.822746,-7.314269],[-5.522791,-7.745513,-4.926809,6.977133,5.348637,-9.750179,-5.673656,0.667597,-8.077965,-7.204149,9.777742],[-8.379605,9.066251,4.098869,-2.260498,4.167072,-1.406088,2.692235,0.312044,-4.754481,3.656392,5.660705],[6.811462,-6.979138,1.158469,-4.194541,-6.452285,-0.457463,7.311004,3.425118,-2.531598,-4.235298,-2.161810],[7.660041,5.804605,5.959635,-7.743508,-1.377685,0.441091,4.578204,-1.053267,1.335507,-2.598793,-2.809527],[-1.839375,-2.052666,-5.402877,7.237198,-6.978323,-8.961516,-0.952412,4.806829,-5.134339,-4.641994,-5.676219]],[[-5.162704,0.416935,8.134107,-7.107545,1.256793,4.088445,9.116415,-7.059369,-0.194610,3.549338,-7.186114],[-4.858512,7.251365,-3.878913,9.713876,2.274570,0.144080,0.321325,9.141966,-1.902029,-4.965462,-8.754702],[8.316215,-1.503713,4.925678,-9.729632,4.455751,4.294774,-5.149954,-3.358085,-7.778569,8.891537,-3.783290],[-5.903490,-3.016330,-2.087530,-2.928507,-0.850866,-1.304161,-8.458979,3.778208,9.972451,9.553222,9.303312],[-2.945568,5.008280,4.599981,-5.698146,-3.509055,1.244923,8.456763,4.809540,-4.967423,6.958796,0.036556],[1.013728,2.106877,4.797675,8.675161,-6.385840,9.936921,2.962410,-4.367579,-7.053435,-0.217047,-0.643765],[-5.767830,-6.519434,0.462660,8.820607,9.640292,6.926928,2.851611,5.505651,4.353550,8.221134,-9.230972],[-3.895283,-8.488403,6.899298,-3.463365,4.299256,9.517407,-2.761118,5.444713,6.074660,-8.444651,4.756900],[-1.740614,-6.738403,-1.660101,-0.067114,-7.613726,-3.799091,0.852431,-7.768410,1.405403,-2.871987,9.489910],[5.970873,-6.813221,4.431368,-4.091798,3.634744,5.957812,-4.389404,3.148526,-2.633896,5.705520,-9.106999],[-1.904442,1.475829,-2.115623,4.132309,-8.591231,9.170857,3.162100,4.443796,3.943024,-1.458441,-2.443161],[7.232901,1.166244,-6.328529,-1.138547,-3.589333,4.689849,-6.225624,0.798876,2.104482,-4.146768,8.995290],[0.111239,2.251858,5.607015,3.139858,9.945755,9.844221,9.578625,9.105756,-6.797282,-7.904547,9.269931],[-7.025776,-5.950839,-0.364847,-4.799628,1.835165,-8.567377,-2.680993,-4.448946,-1.645222,-7.648083,6.743344],[6.014703,0.968206,-7.489089,7.516359,-0.848249,-9.612798,0.774551,-4.422581,-3.610010,5.812933,-4.450708],[-1.542794,1.945255,-7.351670,-3.690686,1.055163,7.394374,-1.827851,7.866651,7.177040,7.324261,-3.990477]],[[4.305181,-0.940957,-7.953024,-5.533810,4.032334,-1.681223,0.913908,0.813222,-3.983923,2.790400,1.564953],[3.642206,7.266453,0.703381,-5.759019,9.203191,-5.798347,-0.471504,6.357334,7.674596,7.179087,-1.917744],[1.821537,5.690923,-3.569636,8.724522,-0.120543,-4.361822,6.056605,-6.724643,-5.564158,9.327021,7.270052],[4.758041,1.121016,0.146383,-3.874700,2.034365,7.845157,6.344316,-2.416320,-2.324144,7.455602,-9.700953],[-2.896036,5.376612,8.430373,0.660370,-4.552526,9.739223,9.463621,-0.557790,-1.528511,9.582440,-5.124958],[-0.652246,8.983128,7.960424,4.977370,-8.589097,-7.747049,-7.643650,-3.837413,-2.656011,7.435529,-6.218094],[7.171276,-9.016850,6.116788,6.887408,0.987956,-6.407269,-1.519504,0.274414,1.800995,0.264356,9.409412],[4.961894,6.344211,-9.312529,-0.367017,-8.006788,-4.690424,5.888756,-0.125307,5.248832,-9.837041,4.886558],[-9.032207,7.159507,-7.363126,8.036185,-4.639192,8.801201,4.094729,-2.228975,6.770903,2.143514,-8.445089],[6.038638,2.183116,-3.821724,-2.412885,5.860004,-0.229626,-4.453137,5.590307,7.287338,5.351740,-2.209663],[-6.320894,-6.372284,6.598320,6.875853,-9.942744,-7.110915,-9.771173,-9.398856,-2.085725,1.700490,-9.652645],[6.412830,6.335005,2.385508,-0.304911,0.371961,2.677012,-2.920987,9.182624,-1.747124,-4.647992,-7.268381],[1.938693,8.325137,-8.863506,-5.723741,-4.278242,-9.902740,-8.492461,-1.025030,5.149054,-9.550783,2.083256],[3.650257,-8.353129,5.015929,0.910382,8.670700,-8.859038,-9.419076,1.176962,1.201251,2.128464,-4.009490],[8.591892,-1.558133,-0.016575,1.943216,2.558393,8.297106,-9.219460,-9.182201,-7.281398,-6.753983,-8.568787],[8.136238,-6.739174,7.447657,-1.789657,1.748726,8.603342,6.335231,-8.046277,5.488814,4.238996,-9.398268]],[[5.689062,9.474695,-3.348911,-2.823201,1.919975,6.079398,-1.746126,9.013304,-4.257703,9.587202,-1.126475],[1.763765,1.144796,-5.225426,2.019966,0.390807,4.127343,-4.307306,-0.757068,-2.240280,-3.328472,4.571389],[-6.971675,-8.622695,-7.334365,-8.274013,-3.872039,4.694754,1.442703,-6.545000,-1.154935,2.978896,8.890984],[-8.918939,3.750170,9.000918,-9.449190,-5.540097,8.118490,7.982921,-7.638297,-9.912283,3.745226,8.072425],[0.573760,-5.563505,3.795268,4.171730,9.642924,-6.809228,-0.839086,6.715958,8.340732,3.138773,-9.859122],[-8.815107,7.264319,5.876313,-6.813240,7.610001,-3.723413,-5.108191,9.936804,-7.287632,4.721311,0.393482],[-5.258535,-1.544124,7.467438,-6.152018,-2.756839,-7.480784,9.396731,-0.585694,-3.139714,5.706344,1.530038],[-1.060568,3.945085,-7.473447,-7.850466,-2.839932,-4.283457,-1.818910,7.966053,1.489630,6.717093,4.080329],[1.072597,-1.263254,6.841255,-1.215776,3.712866,8.119014,-2.875176,7.228849,-6.755319,0.205193,-7.659475],[-3.890678,5.466160,9.708923,6.704167,7.066106,2.379127,8.350501,-4.768772,1.681702,-6.626107,-9.217375],[-3.037028,0.399699,-5.624737,7.194084,-8.543271,-3.849299,9.838733,-5.445245,-1.367918,4.310103,-6.588231],[-7.037754,1.453157,-8.569067,-8.426608,-1.509484,9.087448,6.273958,-2.281704,-7.045977,-2.795961,-0.064324],[2.526716,3.193869,8.878343,9.078051,6.301436,-8.756052,2.924643,-7.262624,4.662998,1.411155,6.259161],[-0.502201,3.472172,-9.990026,2.227193,-3.988855,-3.961391,-1.045504,-7.508558,6.776736,3.946333,-2.898021],[-2.796972,2.758357,1.655084,-6.653186,3.384479,5.490653,-5.280969,1.687689,-9.396155,-5.704389,2.072464],[-6.481810,-3.517474,6.482408,8.755349,-3.710924,-1.735955,-2.616684,-5.608367,-2.121148,7.917559,5.596116]],[[-6.708703,-9.314476,-4.849661,4.415784,7.482130,5.211477,2.804861,-0.716167,7.035839,-1.529134,-1.122804],[1.033639,-6.359628,-3.402729,3.920860,1.302610,-2.906563,6.953206,-7.678564,-2.537199,-4.593677,9.442469],[-7.262391,3.574036,-6.391314,-6.234664,-5.414106,-7.058422,5.248356,-0.315333,-0.184584,-2.166587,3.968447],[9.559918,-1.310713,-0.196536,1.647748,-5.601050,3.238784,8.202792,-3.693725,-0.406815,-4.806725,8.666362],[8.666311,-8.759040,1.756508,4.161354,-7.495399,-6.417752,-8.747825,-8.729901,-1.733567,-6.430047,-2.101577],[8.424119,3.760112,6.347044,-1.059384,6.129615,-9.321613,4.020834,5.380869,4.419843,5.064179,-1.267408],[-1.007478,-0.772537,-1.700850,-9.024244,-7.708388,4.883072,5.712707,4.396723,7.700938,7.453779,0.366412],[6.963578,5.859869,-5.395870,-0.891097,0.164762,9.992841,1.005589,-9.949361,-9.887758,4.048824,-8.088850],[-2.578477,4.861866,-1.495486,-6.903083,2.608193,-6.166264,-5.383438,-4.284223,-6.597315,4.784807,4.657848],[8.555835,-0.793352,-3.450317,6.738579,-1.548422,9.304687,6.753702,-5.990786,3.461697,5.817902,5.635837],[-7.987124,-8.755666,8.809956,5.407244,-5.539642,-7.290454,-3.294659,-9.957689,9.354158,-0.224033,-0.941858],[-8.200172,1.085655,4.035527,-4.733884,-9.343675,-8.174077,4.119751,-9.276229,-8.083755,1.803973,4.801949],[3.820247,3.661890,-2.156200,8.160108,0.054169,0.230390,6.734750,1.241233,2.158519,7.865584,-8.790295],[3.141364,7.631574,6.895423,-7.220539,-3.419259,-6.935017,-1.309822,4.101705,-0.250091,-9.805448,5.972215],[-6.345384,4.876087,-9.202472,6.796436,7.892017,1.298244,-3.312484,1.628500,8.575632,-1.206462,7.013965],[-6.936601,1.775063,8.343271,6.805721,3.487560,4.259120,4.518231,3.879023,-9.511042,-3.279094,0.265851]],[[-3.787475,3.520506,1.482330,3.212067,0.499949,6.502253,-2.329720,-3.506696,-4.245891,7.774274,-3.230487],[4.097638,7.554984,-4.402889,-7.935215,-4.763062,-2.933673,-8.126398,-1.410962,-5.314988,-2.185567,-6.994720],[-1.298604,-3.915932,-5.812073,-0.099929,-5.166472,9.346868,9.977451,2.765793,7.440921,-8.253010,-0.602718],[8.132919,-7.115418,-3.121907,-4.438307,-0.376896,7.487964,5.627784,-8.834179,-0.184458,4.416597,4.112338],[9.328647,-2.934385,-6.417471,-2.375321,-0.627824,8.489252,7.547627,8.932468,-3.945748,-0.993171,-0.294557],[-1.202749,0.875038,3.510409,-3.196694,9.777560,-7.188671,-3.885471,-2.270560,9.045402,2.643950,6.677685],[-2.424117,-7.006895,-5.110377,-7.345861,-8.454237,-8.831283,2.676863,-7.019901,-4.307267,8.710423,-8.188997],[2.380807,-1.495863,3.327442,-7.346748,-0.627486,4.490359,0.607710,0.024591,-2.430508,-0.761774,6.085914],[-1.987471,8.346665,9.892595,0.989723,7.963120,-6.007208,1.934116,2.799645,-8.917942,-0.769045,-4.609170],[-7.832700,5.893071,7.864760,5.644877,2.032874,-5.991407,-9.469305,5.534127,2.793541,-9.554218,-7.021331],[8.524245,8.138771,9.941909,9.859132,-1.894259,1.600956,-8.335007,2.017562,4.657459,-0.874383,-3.970032],[9.784065,-4.421453,-5.616987,-1.549467,2.223404,-1.158961,-7.046704,-3.680415,7.210651,1.624313,8.726938],[8.037111,1.676897,0.444851,-6.189081,-4.082503,-3.092553,3.047652,6.949798,8.514462,1.882719,-5.542103],[-6.920343,-6.702984,1.794337,-3.620812,5.281439,3.345800,8.885611,-7.833713,-2.690647,3.421178,-4.857409],[5.305530,1.438970,-2.085723,-2.949286,-5.694648,2.861418,-4.020167,-4.923214,2.261747,-5.800905,-0.480016],[-6.123448,-6.951533,0.834940,4.184528,-8.814392,7.849702,1.903768,-3.150017,-6.899738,-2.764398,1.383859]],[[-2.006110,-4.402051,-9.528660,5.827579,-6.938684,0.905997,-4.008751,-6.382245,-4.113509,6.666574,-8.904273],[6.701225,-7.641594,4.353609,-5.146912,-8.292390,-5.246096,-8.728040,1.443648,9.917229,0.377932,-6.483794],[3.515397,5.682339,9.294123,-9.535943,-6.527281,7.107919,3.003519,0.688972,0.470232,9.984973,-3.168850],[-9.092986,4.668712,0.002024,1.463383,-8.374877,1.045035,-5.756122,-4.833082,-9.441694,9.629696,-6.498103],[4.236446,-5.319094,9.933830,-4.348342,8.719866,0.204713,2.664621,-8.788348,2.905556,1.662033,-2.342878],[1.823863,-0.141455,9.244335,6.792110,9.038649,-3.535233,-1.174088,1.480990,7.118520,3.430320,5.761475],[-5.049696,5.833587,-8.312673,-7.200545,1.983243,-9.957221,7.941589,0.011593,9.187956,4.069307,-9.527498],[6.464327,5.308873,-3.689370,9.094810,4.262575,7.251390,1.723116,5.166560,3.037097,-7.344837,2.309258],[5.451453,-4.680919,7.471029,-1.008289,4.554094,4.498474,5.257793,-6.088291,4.642924,-3.681349,-2.396748],[-6.385153,2.551088,-5.392285,6.894559,2.132215,-7.850949,6.330098,-7.032453,0.079949,7.359597,-2.882945],[8.333996,-5.864816,-6.187316,2.479761,-5.692169,-6.843520,7.508262,9.169464,-2.342317,-2.723815,2.261590],[-4.616618,5.778877,3.956433,-0.120677,7.954356,-6.182691,-5.240815,-4.846817,-0.830198,-4.975101,-6.142188],[9.680176,-0.031913,6.796390,8.896660,3.014929,-2.488557,7.651404,2.687946,3.274614,7.720628,-5.901346],[-0.152760,-3.670929,-0.672724,6.993481,-4.767144,-9.402947,1.197297,1.206526,-8.670906,-8.952592,-2.685862],[5.793389,-6.914249,-9.824957,-6.602132,2.567100,-1.604231,0.389468,-2.686569,7.230088,-1.163632,-9.940277],[9.075816,-1.500192,-4.445174,-6.224012,-5.061011,-8.135281,-2.550079,4.146243,-3.649864,4.957975,-8.428311]],[[4.538535,-1.296981,3.851273,-9.483217,-7.004027,-3.462183,-1.091630,8.744213,0.336585,-6.731507,-4.482958],[-1.578574,4.564310,9.273172,6.379198,-1.427888,-7.175250,-2.276001,3.713252,8.684741,-3.449565,-6.461107],[-2.896319,-1.489780,1.765130,-7.943384,5.399858,8.147920,4.727847,-2.075556,-5.995192,8.557217,5.975089],[2.310058,-3.372477,-5.701208,-2.722632,4.572062,-4.016702,-3.209773,-1.442478,-6.117182,4.718801,4.977434],[-6.513681,-5.823788,-2.604571,9.274883,8.626189,-4.946992,-0.812854,-9.029063,4.734992,9.394065,7.907827],[6.758216,-7.822549,-6.387016,-6.954829,-6.276107,8.272503,1.034918,-8.473807,5.094687,2.346974,-1.064908],[-8.688483,-7.645393,-0.359478,4.943306,4.435236,-2.866488,-3.612894,-8.407458,-2.052250,5.478162,9.850655],[3.554151,-7.279688,1.439034,-1.247134,0.738135,-7.052812,-9.049144,3.182960,0.045913,-1.656179,4.938397],[-4.943002,5.307347,-3.603372,-3.208125,-8.295866,-9.289076,-0.753723,-0.220297,0.231505,3.014968,-3.683918],[-5.371628,1.798005,-8.747800,7.037906,6.514503,9.116899,-0.673088,7.566683,1.939970,-0.564376,-5.415064],[3.096700,5.665269,-6.111045,9.656775,-9.611170,6.450079,0.330547,-1.617422,4.852980,-6.815755,-0.793194],[-7.303418,8.097532,6.264318,8.300502,-5.051470,7.610545,8.948381,-6.510474,-9.134588,6.152395,-5.198867],[9.439134,5.153102,-6.685538,-1.982853,2.243583,6.560289,-6.839184,-3.222663,-5.948466,6.184915,2.975385],[-5.199053,9.452300,-4.654380,-2.269720,6.419992,3.029755,-3.354806,-5.069309,9.937410,0.058930,-6.895227],[0.306227,-3.681808,2.152065,3.438440,-2.999742,-8.931002,-2.846264,-2.690758,5.189905,2.583302,9.433214],[4.424853,-9.695482,5.674651,5.528649,9.998914,-8.560549,-7.831201,8.520609,-3.397440,1.840291,5.713890]],[[-6.693596,-9.864303,1.387463,-0.029116,3.933335,-7.100939,-1.701553,-6.862207,7.511682,1.081316,8.294077],[6.607649,0.972857,-3.283949,-4.405062,2.848703,-2.661896,1.599461,3.070151,-0.168680,6.986729,-9.373121],[-9.287486,1.208189,-0.172931,5.660262,-0.413589,4.290561,4.876293,2.369721,5.369484,-3.985180,-9.019011],[-1.132189,2.222160,9.272930,5.978031,3.270543,-9.129950,-2.967054,5.413166,-3.355200,-6.372500,1.569654],[5.431249,-5.942745,2.487447,-8.715687,-7.666693,2.542203,3.137196,-8.926883,-2.059293,-4.550014,-8.890638],[-3.800886,8.092737,8.671468,-0.221953,2.714746,8.765538,-3.067636,-2.952102,7.424342,6.813487,-8.701082],[2.869267,-6.137210,-7.491741,-8.775138,4.899289,-3.658671,1.757348,-7.346787,8.617116,9.548633,8.988445],[2.740971,3.572999,-0.422316,-9.606346,9.386377,0.562250,8.709317,9.099248,-1.149268,-4.448701,2.745699],[4.819259,9.233574,6.090828,2.741392,8.499605,-5.142315,8.802387,4.259278,8.547059,2.206919,-9.680410],[2.291487,-8.949790,-5.664950,-7.055461,-0.012317,-7.051503,-2.226620,0.896177,-1.385251,6.863303,6.438947],[-2.916041,-9.636968,8.835347,-1.619628,-2.107178,2.195387,-9.120282,-8.566978,2.479745,-0.276318,-6.167206],[5.866792,-9.792465,-9.424038,-4.336125,3.439219,0.085235,5.947865,0.538922,-7.180381,3.150547,9.697085],[-9.953082,-6.103583,-8.519839,2.265782,-4.075965,-1.338486,-5.346824,-4.136275,-4.597166,3.299922,6.306710],[9.196955,3.623582,-5.768164,6.722326,-6.000231,3.652336,-4.266160,0.650762,5.168469,-7.018403,-5.927181],[-1.317274,-5.008359,-3.419172,-1.176717,-1.660367,8.180450,-4.745768,-1.717808,6.392892,8.166573,9.200607],[-0.049576,9.328068,-0.371249,-2.368019,-3.620728,1.018981,5.115421,8.860298,8.139068,8.280541,2.542965]]], dtype = "float32")#candidate|1894|(15, 16, 11)|const|float32
uop_1895 = relay.tan(const_1894.astype('float32')) # shape=(15, 16, 11)
func_1351_call = mod.get_global_var('func_1351')
func_1353_call = mutated_mod.get_global_var('func_1353')
call_1897 = func_1351_call()
call_1898 = func_1351_call()
const_1899 = relay.const([[[9.518510,8.165972,1.595054,-4.907236,-8.422986,4.177884,7.233530,7.908708,3.608464,2.093983,-7.755250],[-5.661151,5.953992,8.700830,4.516838,5.737647,-3.285186,-4.016561,9.115644,-1.577790,-0.036444,6.049869],[-3.533017,3.837621,-6.623242,9.379558,0.205009,9.586066,4.219931,9.447348,5.368051,7.179621,7.950663],[9.545178,5.860763,0.522555,1.658333,8.030746,3.082582,8.227484,-1.426361,-7.113628,2.909307,-3.034201],[3.144817,7.431387,-9.230021,-9.868118,-1.172169,-2.612732,5.877675,-8.226144,-4.308248,7.789674,-4.213985],[3.354991,9.982213,3.638288,1.890755,1.796063,1.626607,-4.814137,-3.956705,1.275510,-2.824119,-2.389891],[2.826806,7.665878,-3.031854,-7.407494,7.358864,9.324787,3.450557,8.932230,6.754940,6.017718,-3.805447],[-6.371928,-1.621242,8.673244,-7.983266,2.545032,-5.238952,3.803617,2.163689,-4.732870,-2.151526,0.795052],[-3.543409,-7.397361,5.503872,-8.712713,9.306786,-1.232238,-2.498529,3.531312,0.733683,5.365245,1.055389],[4.055705,-3.863041,7.906718,3.279529,-1.110590,-0.313453,5.578906,4.786082,9.752923,8.904893,-0.771088],[0.091987,-6.282625,-1.928976,5.101310,-0.645325,-7.799069,-9.340465,5.152231,-0.618111,8.608759,5.162388],[0.702668,-8.947422,0.286321,-2.407402,0.973045,8.650317,8.809613,9.087897,9.718183,0.867665,-0.343527],[-1.407027,7.066919,-1.142073,8.378216,-2.478516,-9.364835,8.844467,-7.954799,6.147657,1.455483,1.318607],[2.784964,8.685286,-3.611718,-9.611299,-6.347662,2.835788,-6.851073,-9.131409,2.818659,-5.431719,2.393187],[7.338269,1.184843,-5.454846,-4.680216,2.512757,-3.620085,-9.005534,8.059710,-2.892307,8.630434,-3.951650],[4.999129,3.992060,5.133937,-9.707512,3.672199,-4.865799,-3.252932,-6.734462,-9.488881,-0.302493,-7.575340]],[[8.037659,-6.820914,-9.534271,9.181087,0.564524,1.648063,4.958213,-2.993692,-9.123428,-2.913802,-0.044073],[5.838348,1.195965,4.057891,-1.475631,-6.116583,9.167428,0.287052,4.110311,-2.532684,6.977583,6.737057],[-4.311170,-6.118459,-4.912012,0.966180,-7.173759,1.670221,-5.225121,1.633814,-4.008616,3.784713,-9.325368],[1.371630,6.620584,-3.403545,-9.105525,-1.840417,8.277192,4.934957,3.482682,8.869542,9.460222,1.175432],[0.083441,-5.128775,-4.510110,5.164545,4.602568,-5.866096,7.532548,1.089780,5.789274,-1.509436,5.526411],[-7.674089,3.850468,-9.444725,-4.674766,-5.162347,-8.367415,6.629554,-3.115304,-6.204914,-2.638984,3.000013],[2.773378,0.662684,-9.117640,-6.555230,6.213162,-9.822521,-0.623156,2.180314,-8.615458,-7.421386,-0.798500],[0.834304,-8.305690,-7.085930,4.499536,-6.619465,0.722075,7.889952,-1.538765,0.432095,-3.900052,-1.619310],[-5.091828,7.235569,6.027965,1.749978,3.394365,4.082532,-8.156577,-2.349801,6.222551,8.865170,-0.732665],[-1.263060,-0.215198,-4.123041,3.772879,5.323412,-1.798746,-6.490811,-4.288213,-2.671073,-2.029358,8.285789],[3.461343,-6.686540,9.291793,9.357892,-4.067557,-4.778065,4.043896,6.601213,6.792019,-6.232100,5.946018],[4.314404,3.690410,-0.447956,3.076085,-1.165246,4.590580,0.550532,-2.565755,5.449047,-0.743929,8.751901],[-1.402505,-9.450320,1.728405,-6.948130,7.653049,-8.161262,9.736029,4.537466,1.455477,7.225240,-2.301905],[7.897161,-6.695545,-2.750633,-9.038148,-7.617017,-1.511854,-5.776705,-9.287940,4.073417,-7.784909,-1.694748],[-5.340357,1.933354,4.517748,-7.050741,9.918030,2.481105,-0.906617,8.429932,0.679970,-6.713495,-2.883809],[-3.559048,6.387824,3.725728,3.449799,0.869743,1.435644,9.419421,6.555990,-1.783409,5.208262,-8.573398]],[[2.941724,-9.955430,-6.271376,7.747441,7.578044,8.516101,6.708418,8.960425,-3.689978,-8.779518,5.666490],[-4.697918,6.518751,4.641573,-8.517114,1.250702,8.417530,9.051487,-0.995736,8.044083,-5.038766,-2.755984],[4.043781,-3.717352,-5.916277,2.286480,3.946673,-4.921374,-7.906510,0.769136,-0.830815,7.513597,8.483055],[2.724182,9.718581,4.756208,1.411759,9.575307,4.704630,-0.619878,-1.281879,5.429731,7.058494,-1.149962],[-0.088291,2.536281,-8.348593,-1.701648,4.751247,0.412740,-0.336027,-5.133836,-1.079676,-5.736194,6.020064],[6.831697,-8.723631,6.571898,0.228094,-1.256121,-6.475062,-6.989179,-7.033524,-2.801617,-2.898089,-1.731590],[-3.286784,-8.869332,-2.682771,7.575382,8.861413,1.815519,0.849052,3.265832,-1.470780,-5.726045,0.015515],[-3.403584,-1.893171,4.284879,-7.472722,5.083089,-2.813185,-3.790070,7.418588,9.104088,-6.862052,-5.979997],[3.053962,-3.681271,9.295761,-9.041963,5.476447,-4.802449,5.533769,-0.813065,8.308873,4.008938,7.620170],[-5.257237,4.573562,0.071476,-3.537951,5.743173,-9.136066,-5.749703,7.016210,-5.366333,0.412570,-3.279518],[0.406956,-7.286619,-5.685733,-6.040587,7.909646,-5.739970,-7.822958,6.213697,-1.019203,3.056261,-0.662539],[-9.209140,-8.211456,8.418756,-4.174389,6.088381,-6.888698,-5.854302,1.654123,2.958738,-9.559565,0.865691],[7.742976,8.409310,2.489619,-4.553955,-1.538561,5.349811,-9.396241,-8.451265,-3.785262,-2.836162,-7.098062],[-1.985994,8.068805,-9.676059,7.431982,7.432951,-0.479722,-2.451539,7.085908,-4.394398,-8.048774,-1.242282],[-0.159244,6.955945,-5.003635,7.940110,-2.349958,-4.394869,6.939342,3.026195,-5.961399,-9.996909,-8.344158],[5.944370,4.716660,9.128695,6.239725,-7.978628,8.430327,2.078803,0.457845,-6.409838,-4.609191,-6.205671]],[[0.028121,-7.874733,-3.776553,1.310396,-8.689129,-0.738125,-5.241340,5.539909,2.515006,3.346253,9.955182],[3.046610,5.818465,-3.844134,-6.859804,-4.961623,5.677856,-8.099583,9.007801,-2.353715,-5.189896,2.508017],[-4.964158,-9.478174,3.637764,-1.458856,-9.019251,4.531953,0.295532,4.197361,-2.152802,-3.821916,3.181677],[6.322239,-4.170038,-9.259349,-3.246810,6.971582,-7.353021,1.810303,8.165116,-1.690163,4.767731,2.699631],[8.821001,-5.329728,-2.572307,1.367321,1.345791,-4.345075,-6.947262,-8.325025,4.304001,-5.215366,-1.959638],[5.965677,5.451397,6.436841,2.756073,-8.683043,-2.727388,6.805459,9.100481,-5.672722,0.341076,-3.389688],[3.685185,1.967770,-1.949801,-8.127692,7.384405,-5.197523,0.084754,6.309939,-3.597421,4.791557,-3.807626],[-0.356432,-6.835283,7.958802,0.841123,4.301360,-1.573727,1.268012,-3.088716,8.831483,9.475201,3.756391],[-9.375730,-2.738695,-2.758962,9.639446,-7.311332,-6.480637,-6.560811,-1.709704,2.470169,-1.683847,-8.287596],[-3.339644,5.831125,-4.009568,6.137889,9.268697,-5.295027,-6.861000,4.324069,-5.504106,1.180357,-6.390229],[-7.294310,-6.612907,5.374773,9.870597,-0.779813,-3.868954,-3.166354,5.971581,-8.049507,8.246068,-4.814755],[0.587013,2.087516,2.519491,-4.735510,-4.760772,-2.422392,-7.020081,7.626624,-7.490215,5.689704,7.191278],[8.016308,9.264435,2.610477,1.257693,5.823809,2.130116,-8.253474,5.748268,8.569152,3.461291,-1.346660],[5.313499,9.524392,-0.963180,-7.893103,-2.131475,-5.692631,-6.818625,-7.047091,7.010795,-3.646255,-8.697672],[1.107227,-2.065114,1.170815,-7.606610,9.931344,9.810530,2.060855,-3.223689,1.355638,-0.904665,-0.117319],[6.142368,-0.748614,-6.024472,-0.202183,1.016551,0.373759,-8.996637,0.982124,-6.719149,-7.285917,6.413894]],[[4.240406,9.747027,4.030816,-1.146093,-0.447493,-8.855404,-9.790962,-3.844694,-7.811642,-0.174611,3.820356],[-3.160602,-5.695279,6.489869,6.105155,-6.429306,-5.581019,1.415329,3.303496,-9.158432,-8.279898,-5.162162],[4.049167,6.792212,-6.940238,6.906788,-1.715497,-5.064751,9.169083,0.345405,9.797565,7.680305,9.036372],[0.791589,5.255264,8.195212,-4.085008,-9.467628,-9.622179,3.532865,-4.192747,-3.095604,5.458712,2.136918],[-3.190820,1.822198,-3.182710,-9.963868,5.839227,4.471776,3.302992,-1.925237,-3.152866,9.723867,-7.136099],[-8.344151,-1.585262,1.820633,2.679155,-7.411968,2.018293,4.687456,-3.307062,-0.742116,7.263289,-8.410399],[-1.708533,2.116339,-1.524258,7.493313,9.456019,9.632154,-9.617952,1.255204,6.413050,8.701909,6.625607],[-3.540163,-6.721452,-3.175545,-3.428697,-3.992936,4.856654,-1.577986,2.290622,1.436114,5.686597,-3.533934],[7.290609,5.154405,7.770280,-6.304064,-2.729167,-6.072706,3.565347,-6.757521,-9.875397,-4.796433,-9.824361],[5.611850,-1.687169,6.869927,5.268286,8.622934,5.019152,5.488411,2.281745,-6.967302,-0.974887,8.184521],[1.194152,-9.062887,-1.470515,0.621578,4.182527,-2.293115,7.248699,9.259999,3.159354,-3.249362,0.846783],[3.555893,8.950588,-3.977741,4.151987,0.548162,-7.048816,-9.064512,-0.692141,1.304896,-2.013625,5.575323],[4.675746,-0.921299,9.261485,-8.843487,-9.005925,3.855586,8.068071,2.290242,4.077047,8.296513,-7.067886],[4.731695,5.285405,-5.539581,-1.748785,-4.453339,-7.127244,0.014135,8.926945,1.243263,4.750715,4.679924],[-3.582927,1.485302,6.695301,9.916103,-8.747694,-8.071647,-0.283447,8.645793,-3.913433,-3.225628,-0.552658],[-1.657618,3.328351,9.616151,-1.462023,0.309515,-8.150929,3.002537,9.388152,3.348694,3.042090,-1.266181]],[[9.048186,-6.320697,3.351353,-3.830669,-0.346359,7.016635,-8.225671,2.042112,7.202178,-7.333466,9.080108],[5.709129,-7.824220,0.745050,-9.933204,0.223670,0.062481,7.735803,2.054579,7.430330,-3.851823,6.830011],[5.752052,5.536753,-8.831680,-3.897156,5.137808,-5.724871,6.070643,3.381890,0.907118,6.933136,0.645160],[5.677297,-6.084568,-4.642132,9.546270,-2.130513,-2.171621,5.780112,0.638422,8.240582,1.528699,-4.427523],[3.445831,-8.034974,0.364183,7.455574,-7.292785,2.678253,5.232255,4.182624,0.568957,-7.134114,-4.153569],[-3.135180,3.771541,-8.443845,-7.927151,8.884812,8.892835,-1.014933,4.479700,4.625983,9.381569,4.067907],[-8.564268,7.523858,-7.707633,-9.004040,-9.030774,-6.854464,-4.678183,-8.616801,1.313181,-1.437107,-6.647233],[2.745162,-1.017121,7.101989,0.652585,0.472422,-8.723981,-5.842712,-1.108648,7.714883,2.889553,-8.858252],[-2.879566,-9.499474,2.344204,-2.602807,-8.790839,4.770006,-2.638182,-0.701648,4.406150,-8.035774,-4.722106],[-0.160436,2.605218,6.323538,1.158290,-8.163142,7.194031,-0.690824,0.783670,-8.575399,-2.951118,-8.307687],[-3.725358,3.904126,-3.651072,5.174067,-5.652660,1.840503,0.878153,-1.220483,5.531969,3.119261,-0.744908],[4.963058,-4.150467,7.099965,6.806644,-6.079862,-9.992234,-2.323464,6.474797,3.522203,6.994441,2.017064],[-0.942631,-2.231858,-2.173058,-8.232455,0.798191,-3.987434,4.744267,-4.067889,6.733667,-5.789804,-8.743933],[0.441388,5.830645,4.494810,4.383759,5.935741,6.186146,1.375924,5.781203,-2.331293,3.076890,5.738570],[-8.813048,-6.446070,-6.189742,7.990707,5.377193,-4.560254,6.978615,8.541381,-9.975006,-8.023152,-6.661986],[0.513501,0.392791,-3.559203,-9.406136,9.626690,-4.817234,-3.425741,6.284474,5.006171,-9.857752,7.377799]],[[-3.238178,2.615110,6.921466,-3.934780,3.270994,-1.973130,5.064786,-7.533759,-7.131576,-8.453624,7.400845],[0.055806,-0.622175,1.377500,-3.457981,-4.569397,0.849399,1.798181,-9.060896,8.017239,4.653849,-9.741773],[-4.966496,-5.129853,0.347210,9.052444,3.672258,5.973758,-1.079187,5.477581,6.381551,9.322343,0.881338],[9.695339,-2.372770,5.465207,-1.306494,-1.330045,-7.554736,2.766180,4.159067,-9.092368,0.274401,-7.223723],[-8.742903,-6.257481,0.193055,2.956226,-4.713831,-7.136199,5.205745,-4.305710,-3.115552,-0.146894,1.813634],[-2.529959,8.562510,-7.487952,-1.075920,-4.979335,-6.376968,-7.797600,-8.426699,-9.010222,-9.233896,2.200150],[-3.182963,-7.402637,-0.047260,-2.482204,-2.556125,-7.577639,6.273093,3.900456,2.942465,8.138243,-5.112108],[2.536253,2.943147,-3.796320,0.228580,1.445637,3.336118,-2.672872,-5.787322,-8.042887,6.316917,-5.716765],[5.835798,-9.621629,2.739037,-8.554702,7.350625,-6.733319,9.010146,-2.148020,-6.686013,-0.818524,0.325733],[8.734226,3.873968,8.637833,-9.953585,-4.907893,5.401491,-5.999249,-0.596760,-8.902104,7.949461,-9.899702],[3.122334,9.918144,-1.201750,-5.055698,-9.183043,1.557999,2.084865,-6.043234,5.575182,4.811747,3.750807],[2.269407,-9.740961,6.750792,9.149255,-2.921469,-0.331094,-3.816117,8.767238,5.619332,5.969030,-3.544822],[9.416881,-2.716963,8.744579,1.639697,-0.320752,-2.221125,-2.439212,-8.412903,3.540844,2.669820,-5.553955],[-8.765256,-3.162076,0.115885,-4.676642,-7.557735,-1.112193,-4.136151,-8.041241,-2.847400,3.447568,6.831362],[-7.904521,2.912257,-1.162737,0.692852,-0.047765,-4.553937,4.779366,-8.143596,2.653178,6.642315,-7.892545],[1.972247,1.753691,4.853256,-9.042786,5.629644,-0.002349,-4.209316,2.765630,9.805677,4.511204,-7.291271]],[[7.137278,9.458476,5.355172,-0.264979,-8.371321,-6.522937,9.557575,-0.576469,-0.112084,-0.498313,-0.053002],[2.729339,-6.846242,-5.056823,-9.966863,4.822255,6.191353,6.122889,9.708669,-3.761929,7.507063,-6.835132],[-9.633400,9.892709,-3.593827,3.656299,-5.989578,-4.785264,5.166304,1.888389,9.258697,4.199872,5.451214],[1.345794,-9.662374,-4.158503,-6.443586,-2.184791,-9.978978,0.911559,4.020666,4.198603,7.653489,3.826732],[-7.949468,-0.540593,6.879959,8.143660,-8.940479,1.685362,-3.960480,8.809204,-9.481009,-4.643537,1.605823],[6.855278,-1.620918,-6.763187,-9.941606,1.692628,-9.757754,-4.118831,2.761845,-3.686906,-1.089258,3.709582],[-9.002604,1.053498,-5.076149,1.415104,7.951573,6.738452,8.197970,3.168261,7.973751,8.666631,0.447013],[4.200889,7.424579,-4.549035,2.100276,-3.762565,-4.693772,8.587884,9.680860,-7.061784,4.390715,2.812938],[-8.680794,5.329662,0.953723,6.955498,-1.977807,3.367148,5.284937,-8.780745,3.562928,-8.668840,-5.936189],[4.982866,-3.037239,6.477549,0.114364,-8.163747,7.755440,5.528246,-3.459157,2.449015,1.596171,-4.638872],[2.547295,2.791325,-0.761087,-7.232393,8.706004,-9.545782,-9.780004,-4.014297,-8.029590,1.758455,-0.523233],[-0.789792,1.812784,2.528264,2.845790,3.601793,-5.497365,2.485974,-5.011652,-3.474211,7.484289,9.632228],[-8.595175,1.121243,8.972710,-0.437229,4.478655,-7.088653,1.082608,0.436508,-5.048382,4.893034,-2.835005],[6.443304,-1.703662,-7.157835,-7.266448,0.110904,5.682134,-6.401705,9.375428,-5.330952,5.543355,-7.841506],[1.384412,4.218728,-2.034567,-2.251120,-8.419261,-6.659307,8.982064,-1.752836,-3.242540,-3.627763,7.289821],[-2.421723,-2.635298,8.632524,-1.487140,-7.365418,-8.000807,5.785925,-4.157873,9.519697,2.377385,5.859164]],[[-0.344022,-3.441581,-3.181291,6.947847,7.180594,-8.986107,-0.112197,-9.097374,1.378260,-1.372235,-4.258467],[1.058461,-8.464707,4.605249,-5.312926,-4.321032,4.988724,2.364185,-8.733238,7.023307,7.491237,8.155797],[9.611208,-0.004636,-3.985646,-3.885372,-0.329963,7.211631,-8.239049,-4.682496,8.584328,-8.643749,9.665208],[-0.865226,-1.321343,-9.704613,-9.816106,8.825558,3.897491,-4.343020,9.247574,-0.647066,-4.690050,-4.905178],[8.595312,3.738924,0.408321,3.155869,-1.191888,-2.762371,6.527282,-6.321763,-8.790153,7.627420,2.837078],[4.262612,0.890668,-6.147196,7.619018,-1.311254,3.324690,-4.576994,4.220091,3.969794,6.189631,-2.898503],[-3.917081,-4.480048,4.334469,-4.036062,-3.008475,9.312951,-9.194889,1.181551,-9.242708,-8.065888,-9.175042],[-8.895974,-9.213249,5.584822,-8.372403,-6.025673,-3.261986,1.579454,-9.076244,-7.038573,-6.200886,-9.553570],[1.806762,-6.423063,-1.073243,0.897371,2.146056,-3.389510,3.556651,-6.984595,-4.493623,7.611423,7.272861],[-1.258885,-2.295613,-1.527321,-9.275545,8.898988,1.569768,0.329496,-3.239700,-1.530027,-1.489315,1.852237],[-3.702043,0.467083,-8.546909,7.839998,-7.408093,-8.994780,-1.489264,0.734986,5.824907,-9.294239,-7.488480],[-3.141775,-1.332201,5.922007,-8.569935,-1.871546,-9.800825,4.543708,-3.610078,-3.083516,-9.965563,-8.655414],[-5.156413,7.622810,-7.894760,-8.033253,4.483348,-3.907231,0.458516,-7.460608,-7.922482,-0.066089,8.232298],[-1.389926,-5.359003,-2.669692,6.347008,0.905114,-9.966410,5.298842,-9.996881,4.566955,8.005562,8.767915],[0.976695,-7.228858,-2.898485,-4.875669,-4.859088,-1.462016,-6.898508,5.621571,3.242639,9.206325,7.826640],[-9.202510,-3.044471,8.862636,5.115786,-9.406839,-8.635468,5.610240,2.100556,-1.577995,-7.651031,-4.379982]],[[1.197956,-8.209952,-6.072018,-7.798462,-9.770388,6.713510,8.672315,-0.454584,2.664043,8.217821,0.786391],[-9.565505,1.178962,-7.027845,2.004401,-0.439048,-3.735379,6.845380,-4.788735,-6.965111,0.435080,5.714161],[-5.485635,9.119179,-0.355496,-8.667405,2.581872,-2.635375,5.832255,-8.889255,9.513653,-6.377707,1.046831],[-8.244836,7.948177,9.001325,-3.541617,4.972225,-3.082947,-3.619990,-7.453186,3.309889,8.621322,-3.383865],[-3.469345,-8.541722,7.219596,-5.812962,-4.667649,-9.948272,5.665843,-8.520332,-7.567778,3.801761,5.222809],[7.471990,-1.886022,4.091928,-9.394586,-3.622833,1.817794,-9.686505,5.808511,8.774522,3.525915,-9.414174],[5.150789,-8.865183,-9.801238,4.030285,0.974860,2.223543,3.705486,0.772926,-2.389100,-0.474049,-6.968680],[-9.288197,8.468615,-1.237367,1.771594,-9.386521,8.053166,-3.276004,-0.237038,6.700175,-4.069266,9.497305],[-8.465407,1.578442,-4.797169,-9.954830,-0.743828,6.738746,-8.984898,-5.754043,-8.104481,4.412140,3.288264],[2.566693,1.729294,7.993214,-5.675723,1.772637,-9.547418,2.414996,-0.260570,8.548585,7.763457,6.059124],[-7.332285,0.418662,-1.274606,1.968219,6.043803,1.179665,3.864362,-8.416665,7.460670,-1.998659,4.288653],[-0.260914,9.165132,-9.860370,-3.612740,6.372268,-8.485752,1.621758,2.106388,-0.066583,-5.383004,8.738016],[5.870374,0.847332,3.104503,-4.024809,0.906417,-8.781596,-1.509615,4.948035,-5.356980,-0.273351,-7.207379],[3.620463,-0.248019,-7.686778,3.578454,-4.390501,9.357255,9.710402,-8.524643,1.323914,4.831288,-9.122325],[2.938433,-2.899049,-9.923804,-1.506658,4.382570,-8.759933,9.189319,-5.253055,-7.206212,5.704934,-5.926683],[-1.177467,-1.892823,-9.926640,-8.042003,-7.855429,8.298159,0.727544,8.301257,-2.753444,-3.590862,-7.941238]],[[1.076382,-8.852851,7.009955,2.820518,5.605392,-4.378406,-2.191703,6.069692,-3.256454,-5.713949,-0.107607],[1.862942,0.589336,-4.519098,2.334200,6.135663,4.436037,-0.620684,-7.033220,-3.295315,-4.958130,5.569316],[-2.251143,5.974584,-6.583567,-5.172088,1.363264,-4.303095,1.882898,-6.821582,8.788456,2.968158,-0.755964],[-4.325855,2.676047,-6.119261,2.871574,-7.743191,-7.423554,-7.095560,0.658089,8.219192,0.937146,6.252340],[5.581792,4.220559,-4.814702,2.764833,-5.317765,4.954942,0.130537,0.962358,8.960801,-1.674433,5.117984],[4.388711,5.724500,-2.106248,-0.813633,1.549391,2.026410,2.396090,1.747813,-3.836476,-9.691061,7.086294],[0.384529,-6.036410,0.292283,-9.479859,-5.776445,8.017293,4.946527,-8.081969,-7.823459,9.192708,-6.676306],[3.985551,2.563184,7.098470,8.046945,6.984424,-0.574948,6.142656,-7.795071,-7.688685,9.174490,-1.883765],[0.454404,7.914962,-4.780694,-6.692875,8.261973,-1.159341,0.460387,-9.342357,2.969663,-2.678309,4.569229],[4.965629,2.969554,-2.452859,4.902617,0.394038,7.376886,-5.985828,8.088029,-4.699599,-9.413559,-7.522574],[0.295993,-3.702948,-3.897377,-5.617280,2.382767,-1.608654,7.225939,0.273284,8.770811,-4.991835,9.906629],[8.056988,2.349453,2.867270,-7.523657,3.050666,-6.649405,-6.838669,-4.082643,2.244635,-2.811656,4.379342],[-6.657190,9.753823,7.308945,2.949450,-0.997160,-6.257825,-5.513240,-9.932557,1.255570,-7.907200,1.730062],[-6.870927,0.695293,4.649129,7.803053,-2.455374,-6.290740,-7.425924,0.685137,-6.199568,-1.224288,6.744971],[9.093120,4.366793,2.947026,-2.189747,5.569003,8.082255,-6.260381,-9.573785,0.851147,-5.272579,-0.890827],[5.077971,-4.772713,-7.787173,-8.879946,4.569395,0.709941,-2.458075,7.416760,7.114685,-0.202657,7.308750]],[[-4.523009,-0.818690,-4.577594,5.719090,4.038100,-8.470433,-9.569137,-0.511302,-8.018547,-2.734616,6.075025],[2.519198,-3.574602,6.988095,5.205382,1.097163,-9.993065,5.827628,-3.019964,-0.066923,2.112211,9.251555],[0.571910,7.183506,-9.342983,9.788546,-1.261119,-3.779708,3.213866,1.597096,-8.057268,-1.980307,2.021160],[-8.168403,-9.082431,0.034843,3.764173,-8.679487,4.151771,6.757581,5.134520,2.142433,2.869368,-4.711660],[4.290988,1.936406,3.432970,-1.320532,8.519053,-6.516254,6.732934,3.709777,7.310208,6.300295,3.189939],[-1.458925,8.481127,9.346924,6.268593,-9.644223,1.430350,-5.103410,9.927328,4.400181,9.295655,2.030247],[-3.758684,-7.408718,-6.540438,-7.263701,-0.716850,-0.218282,-9.776990,-0.442480,-2.871955,-6.311656,-9.827870],[-5.172847,6.813616,-1.935921,8.430164,-1.878992,-7.924781,8.805840,-6.872019,8.272930,-0.194314,6.578397],[1.436796,-5.431435,-4.862100,5.126022,-2.703306,-2.871785,-9.639321,-7.201268,-9.088551,5.881877,7.180780],[2.423867,-4.645004,-9.079870,3.021337,0.652972,5.626854,-8.992866,-9.133392,-8.914198,4.836373,-6.315524],[2.180993,5.674153,0.269337,6.932577,-1.821699,5.531246,1.771125,4.508108,2.198169,2.060096,6.928486],[-5.443594,4.585053,-0.635073,1.404250,-5.456874,9.824179,4.726506,-2.013076,-8.058851,6.365764,2.067293],[1.615861,1.345729,9.991942,-2.386243,-4.115400,-6.416217,9.356816,-4.360811,-5.709664,-0.523866,2.948226],[-6.069879,-7.747604,-4.475053,-7.676147,-1.980387,-6.187775,-8.054734,-6.835895,-1.634317,0.675370,-4.301848],[4.007815,-5.419091,-9.134963,-2.881925,-3.058246,8.323274,6.412928,-8.764633,-4.195027,4.647063,7.193339],[-5.013177,-7.737026,-3.762976,-5.132962,6.319134,-6.162794,3.576889,-3.060723,-5.144171,-9.496508,1.142734]],[[1.262702,-0.752396,-5.086231,-5.807605,6.830890,1.789122,-7.588364,-4.512404,-3.060111,-2.128163,2.067612],[7.347591,4.853150,-6.008327,-3.595848,6.601081,1.558813,2.872174,-4.982864,1.009256,-1.423119,2.246324],[3.267998,9.155011,2.219397,-3.775155,1.274140,-9.605219,1.405239,6.119028,-0.656030,4.419579,-4.344678],[6.172927,-7.348446,-9.457394,0.573544,5.757805,1.642894,1.071718,9.047294,4.665039,9.840645,7.623084],[-1.680504,-6.411358,-6.651602,8.956118,-3.536604,6.401358,8.320712,-5.431450,2.013048,9.126945,-0.192666],[-6.510743,-2.906798,4.399333,5.531549,-8.847602,3.015765,0.210833,-2.203292,-0.568299,3.679297,-6.560023],[4.756082,-4.279488,-6.405630,-1.564697,-2.733307,-3.637274,-9.297606,-1.814738,7.472968,-9.545825,-6.998198],[-8.970006,-3.452851,3.807898,-2.806927,-7.356693,-4.236526,-8.274930,2.956083,-1.180690,0.568215,-3.398056],[-0.254716,-4.137020,-6.740948,-9.498352,-4.520491,8.880598,9.367968,-6.257964,-0.396942,9.231506,2.793600],[-0.312532,-7.476190,-3.048455,9.013483,-7.372382,-1.518484,1.725024,0.828929,7.695790,-8.650287,-0.476844],[5.495684,9.392579,3.522964,5.859890,7.912616,9.395396,-3.186404,3.381090,0.551531,9.154230,-6.093474],[-1.889835,-3.380004,5.982115,-8.379726,-3.273063,9.597669,-2.644441,-0.720209,4.542106,9.437136,-9.293081],[9.953237,9.681661,8.791229,-7.201744,-1.622205,3.950113,-1.777732,1.219645,-6.994239,-4.931893,-3.552627],[-6.464000,-2.397160,-4.312054,9.651750,2.570615,-0.047859,5.443479,6.158758,-4.607060,-1.722055,-1.444459],[-6.379179,1.072525,5.818990,-6.265254,-8.873841,7.391560,3.599466,-0.370911,-7.806778,4.722343,3.976859],[-2.964642,-7.746976,-2.023819,-0.948349,3.423923,-3.784162,7.989072,-5.997099,4.657228,-3.407411,-8.779385]],[[-8.821507,-6.824487,-8.926386,-1.573100,-4.084862,7.641969,9.322215,6.898845,-6.906132,-0.078887,-2.413999],[4.353340,-3.698697,4.519320,8.424500,-2.099919,1.953361,-5.528821,-8.291321,4.194876,8.762239,0.318789],[-5.252207,-0.492178,9.560942,5.130662,-7.502703,-6.470056,-9.804316,4.640568,8.813129,2.676058,9.922026],[2.324546,-7.769748,-6.232008,5.256906,-9.557770,8.521507,-7.080393,-4.618494,-3.453493,-2.430288,5.547532],[-9.720610,-7.124563,-3.463494,-1.395555,0.152128,3.991746,-7.879116,1.209078,7.298614,2.374808,-1.740698],[9.792283,-0.408642,0.893159,-1.180850,4.258520,-5.686081,0.258456,8.331310,-8.081007,7.696481,8.424458],[3.117773,7.859824,-8.855088,8.151962,-3.600400,4.833286,-3.276610,-4.287435,6.638774,-4.305346,2.653526],[-1.561271,3.238581,-6.634759,-6.197519,4.969073,-1.406678,-9.754970,7.429526,-9.985200,-3.685883,-6.598369],[-0.735794,-3.981911,-5.348562,8.131780,-9.853235,5.289374,-8.039638,-2.962832,7.994020,9.795857,-2.462397],[-5.264293,8.416066,3.744395,3.780746,9.664575,-0.607420,8.558151,-7.319211,3.515768,8.192720,-5.515688],[-4.682873,-8.636509,-9.666344,1.953858,-9.604092,7.004129,-3.350082,4.776349,-7.895144,-0.156122,-1.024970],[-5.604431,-6.407217,0.704392,-8.842313,-6.245818,-8.160146,-5.598757,-9.005516,2.876323,1.250765,-8.433233],[-5.305643,2.872520,-9.816552,7.189560,-9.935035,6.180159,3.670947,-0.565232,9.600012,-2.876579,6.695355],[-3.688452,9.725376,-9.524911,-7.278316,-8.688903,3.402906,-2.885620,8.343444,-7.206314,0.977405,4.710222],[8.149710,-6.370144,-2.612430,-2.835422,3.470353,1.974849,1.612462,8.487405,-7.081325,-2.730172,-8.527378],[-7.072121,-8.619528,3.615135,-4.187604,7.452036,-0.445859,-0.389178,-3.934450,-8.633860,-1.011410,-9.680497]],[[-3.183192,7.037122,-7.439377,-0.782970,-4.182294,-8.161515,9.068492,-0.757157,-1.560484,9.692126,3.456048],[3.948036,8.173014,1.613048,8.651359,5.910517,-5.848358,9.319132,-2.301342,4.226594,-3.880593,0.453626],[6.102822,2.372001,6.037139,-3.968622,0.582308,3.266521,2.662316,-5.876576,-5.051680,-1.525857,-5.253144],[7.342729,9.374723,-8.867666,5.279373,-4.397179,5.329672,-5.640258,-9.782566,-7.965842,4.337403,-7.191694],[-3.878152,3.319094,-9.503449,8.771957,0.836604,1.275733,2.725460,-9.067718,-2.721064,0.587839,5.388809],[-9.971700,-1.044195,6.953890,-6.207593,0.974695,-7.472168,-1.207140,-7.261576,5.733853,-4.977101,2.131939],[-2.652935,-6.641994,-4.429433,3.887162,-9.338519,0.271612,-7.886421,-3.202152,2.789260,5.715734,-0.155404],[-8.951112,-8.820329,3.331531,2.395957,-3.552799,-0.564860,-8.659809,0.701146,-7.564192,0.830684,-0.731362],[-6.366449,3.643660,-6.494875,4.944198,-3.014931,-1.842534,-8.661986,3.220489,-2.953590,3.423263,7.911190],[8.129391,2.812249,-6.628326,0.799154,-4.265830,5.338278,8.950306,8.350181,-7.862770,5.942454,-4.753606],[-6.103336,0.023289,3.162899,-3.191210,-4.975758,4.440564,3.248557,-2.391955,1.959273,2.453427,-1.926343],[-8.666199,4.862016,1.420648,1.840324,-0.943125,-5.934945,-3.606651,2.554289,-1.699343,-3.228128,5.948188],[9.766425,0.917677,9.154003,5.344409,3.130235,0.733989,-1.917377,1.421378,-1.737009,-9.232876,9.312476],[-3.501592,1.142703,-1.974360,0.638344,2.357057,-0.268865,3.139596,5.878772,5.069241,-1.493540,-3.037490],[-7.920609,6.692725,1.822683,6.843298,-2.019397,6.563756,-0.454286,-1.580888,-1.587552,-8.662182,-7.696672],[-5.877464,-7.884196,-2.565814,-2.262674,0.665253,6.849597,5.792427,-9.917098,3.600555,-4.454954,-5.425047]]], dtype = "float32")#candidate|1899|(15, 16, 11)|const|float32
bop_1900 = relay.bitwise_and(uop_1895.astype('int16'), relay.reshape(const_1899.astype('int16'), relay.shape_of(uop_1895))) # shape=(15, 16, 11)
output = relay.Tuple([call_1897,bop_1900,])
output2 = relay.Tuple([call_1898,bop_1900,])
func_1904 = relay.Function([], output)
mod['func_1904'] = func_1904
mod = relay.transform.InferType()(mod)
output = func_1904()
func_1905 = relay.Function([], output)
mutated_mod['func_1905'] = func_1905
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1380_call = mod.get_global_var('func_1380')
func_1381_call = mutated_mod.get_global_var('func_1381')
call_1906 = func_1380_call()
call_1907 = func_1380_call()
func_1315_call = mod.get_global_var('func_1315')
func_1318_call = mutated_mod.get_global_var('func_1318')
var_1917 = relay.var("var_1917", dtype = "uint32", shape = (104,))#candidate|1917|(104,)|var|uint32
const_1918 = relay.const([-10,6,-8,-9,9,-9,2,8,3,3,-6,1,5,-9,10,10,-10,5,-1,9,6,1,-5,-10,-10,-2,-7,8,8,8,-1,6,-5,1,9,-6], dtype = "uint8")#candidate|1918|(36,)|const|uint8
call_1916 = relay.TupleGetItem(func_1315_call(relay.reshape(var_1917.astype('uint32'), [104,]), relay.reshape(const_1918.astype('uint8'), [36,]), ), 0)
call_1919 = relay.TupleGetItem(func_1318_call(relay.reshape(var_1917.astype('uint32'), [104,]), relay.reshape(const_1918.astype('uint8'), [36,]), ), 0)
const_1922 = relay.const([[[-3,7,5,-2,-10,-8,4,4,-10],[3,-8,3,10,1,8,4,-2,3],[-6,-10,-9,1,-2,9,7,-1,9],[-6,-8,3,-1,-7,-9,-8,-5,-1],[7,-8,-9,4,-6,-1,-10,4,-4],[-4,-7,7,1,-10,6,-9,-7,5],[-3,-2,-3,-6,9,-3,-10,-7,-9],[-4,4,7,7,-4,-4,-10,9,7],[-6,6,-4,8,5,-2,4,5,10],[-9,-9,-3,3,9,7,3,-9,-9],[-8,-7,-4,9,5,-5,7,1,-8],[-8,-10,-10,-3,10,-4,4,-8,8],[2,-2,-3,-9,3,-6,-8,10,1],[5,-7,-7,6,4,4,5,-9,-2]],[[-6,-3,-3,-7,-3,-9,9,10,10],[8,9,-9,-3,-9,-3,-7,2,-7],[-8,7,3,6,9,9,-1,5,-8],[3,9,6,-2,3,2,2,4,9],[-3,-1,-1,2,-9,2,-1,1,-8],[4,-9,-8,6,-10,7,-8,-4,-5],[-1,-9,-3,-6,-7,-6,3,6,6],[-2,2,-1,-3,3,9,-4,-4,9],[-3,4,3,2,-10,7,-2,6,-2],[5,9,-3,-6,9,10,-9,-4,-1],[-1,-2,-3,6,-9,-1,1,-8,-6],[9,10,1,-5,-4,-3,3,-10,-6],[-3,2,5,6,-1,-8,-7,-8,6],[-8,-10,-9,7,-9,-8,-4,7,4]],[[-9,-10,5,5,7,-1,-3,-4,-1],[-7,7,-7,-1,6,7,2,2,-6],[1,9,5,3,-8,2,-1,8,-5],[3,-8,-3,-8,-10,-6,7,8,-5],[7,2,4,5,-2,-3,9,-2,4],[-9,2,-5,8,8,-2,-8,1,-6],[-3,-2,5,-2,3,9,-1,2,-2],[2,10,5,-2,-3,-2,-10,10,-9],[-9,-9,-5,10,1,-7,-8,2,-10],[-2,-9,-9,-5,-7,3,2,-7,4],[1,7,-5,-4,7,5,-2,-2,-3],[1,10,-3,-6,-4,-5,-6,-4,8],[-9,-4,-8,-4,5,1,-6,6,-2],[8,-5,2,3,-8,-1,6,-1,-6]]], dtype = "int8")#candidate|1922|(3, 14, 9)|const|int8
bop_1923 = relay.maximum(call_1916.astype('int16'), relay.reshape(const_1922.astype('int16'), relay.shape_of(call_1916))) # shape=(3, 14, 9)
bop_1926 = relay.maximum(call_1919.astype('int16'), relay.reshape(const_1922.astype('int16'), relay.shape_of(call_1919))) # shape=(3, 14, 9)
func_1687_call = mod.get_global_var('func_1687')
func_1689_call = mutated_mod.get_global_var('func_1689')
const_1929 = relay.const([-6,5,-4,6,10,4,7,3,-3,5,10,3,-2,-7,8,-10,-5,-2,3,-6,-3,1,4,10,-6,9,2,10,-9,-8,-10,-3,-3,-6,3,9,4,-10,3,-6,-6,5,-8,-2,1,-4,-8,8,4,-8,-7,6,7,10,5,-1,2,1,-10,-7,-4,-1,-1,-4,4,1,8,-5,-3,-7,2,1,3,-3,6,-5,2,5,-5,-8,2,10,-7,9,9,-10,3,4,5,10,-7,4,-2,-5,-1,-8,2,-4], dtype = "int16")#candidate|1929|(98,)|const|int16
call_1928 = relay.TupleGetItem(func_1687_call(relay.reshape(const_1929.astype('int16'), [7, 14])), 0)
call_1930 = relay.TupleGetItem(func_1689_call(relay.reshape(const_1929.astype('int16'), [7, 14])), 0)
var_1931 = relay.var("var_1931", dtype = "int8", shape = (3, 14, 9))#candidate|1931|(3, 14, 9)|var|int8
bop_1932 = relay.less_equal(const_1922.astype('bool'), relay.reshape(var_1931.astype('bool'), relay.shape_of(const_1922))) # shape=(3, 14, 9)
bop_1942 = relay.add(const_1929.astype('float32'), relay.reshape(call_1928.astype('float32'), relay.shape_of(const_1929))) # shape=(98,)
bop_1945 = relay.add(const_1929.astype('float32'), relay.reshape(call_1930.astype('float32'), relay.shape_of(const_1929))) # shape=(98,)
func_1126_call = mod.get_global_var('func_1126')
func_1129_call = mutated_mod.get_global_var('func_1129')
const_1954 = relay.const([-8.070444,1.700474,-0.224331,-2.404695,-5.032908,9.800493,8.570307,-8.124048,-6.040098,8.136937,-3.552009,-0.184956,8.134149,-8.202757,-6.416812,-7.115248,-1.505041,-2.330383,-8.629338,-8.055406,0.226223,0.997413,9.731456,-4.642565,6.452126,-1.511226,7.992613,-0.753668,7.757021,5.000112,-4.169413,-7.867478,5.551547,-4.682634,1.182925,2.222657,-0.676465,-7.402589,-4.102354,-3.446485,-8.692318,1.923697,9.206371,-0.947695,5.844261,3.266932,1.261275,-6.461841,-9.269335,6.636869,9.221090,-9.358808,4.056854,0.839735,8.397294,5.996346,-3.783239,0.329022,-3.817579,-4.302152,7.891727,-1.715420,-4.082558,-8.964748,5.966762,-4.982348,-6.929876,-6.967150,0.161535,6.533179,3.502640,-6.418527,9.789010,-3.191391,-3.428043,7.669908,-0.336432,-2.968894,2.043788,5.906511,-2.197271,7.282534,2.731127,9.003456,-7.867881,5.875348,5.351660,8.116758,-2.998382,5.540733,2.879899,-8.446062,-5.086854,-6.229597,8.314768,-0.522296,8.579423,-6.608037,9.593959,-7.072249,-8.559862,-7.160219,6.167324,9.054196,9.025072,-1.181663,4.548919,-1.176376,-4.232321,4.478324,-7.833835,5.630110,-2.193673,-2.203415,9.060019,4.531797,7.860326,-0.611531,8.079809,-1.306974], dtype = "float64")#candidate|1954|(120,)|const|float64
var_1955 = relay.var("var_1955", dtype = "float64", shape = (720,))#candidate|1955|(720,)|var|float64
call_1953 = relay.TupleGetItem(func_1126_call(relay.reshape(const_1954.astype('float64'), [8, 15, 1]), relay.reshape(var_1955.astype('float64'), [8, 15, 6]), ), 0)
call_1956 = relay.TupleGetItem(func_1129_call(relay.reshape(const_1954.astype('float64'), [8, 15, 1]), relay.reshape(var_1955.astype('float64'), [8, 15, 6]), ), 0)
output = relay.Tuple([call_1906,var_1917,const_1918,bop_1923,bop_1932,bop_1942,call_1953,const_1954,var_1955,])
output2 = relay.Tuple([call_1907,var_1917,const_1918,bop_1926,bop_1932,bop_1945,call_1956,const_1954,var_1955,])
func_1964 = relay.Function([var_1917,var_1931,var_1955,], output)
mod['func_1964'] = func_1964
mod = relay.transform.InferType()(mod)
mutated_mod['func_1964'] = func_1964
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1964_call = mutated_mod.get_global_var('func_1964')
var_1966 = relay.var("var_1966", dtype = "uint32", shape = (104,))#candidate|1966|(104,)|var|uint32
var_1967 = relay.var("var_1967", dtype = "int8", shape = (3, 14, 9))#candidate|1967|(3, 14, 9)|var|int8
var_1968 = relay.var("var_1968", dtype = "float64", shape = (720,))#candidate|1968|(720,)|var|float64
call_1965 = func_1964_call(var_1966,var_1967,var_1968,)
output = call_1965
func_1969 = relay.Function([var_1966,var_1967,var_1968,], output)
mutated_mod['func_1969'] = func_1969
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1904_call = mod.get_global_var('func_1904')
func_1905_call = mutated_mod.get_global_var('func_1905')
call_1974 = relay.TupleGetItem(func_1904_call(), 1)
call_1975 = relay.TupleGetItem(func_1905_call(), 1)
var_1980 = relay.var("var_1980", dtype = "int16", shape = (15, 16, 11))#candidate|1980|(15, 16, 11)|var|int16
bop_1981 = relay.greater(call_1974.astype('bool'), relay.reshape(var_1980.astype('bool'), relay.shape_of(call_1974))) # shape=(15, 16, 11)
bop_1984 = relay.greater(call_1975.astype('bool'), relay.reshape(var_1980.astype('bool'), relay.shape_of(call_1975))) # shape=(15, 16, 11)
uop_1990 = relay.erf(call_1974.astype('float64')) # shape=(15, 16, 11)
uop_1992 = relay.erf(call_1975.astype('float64')) # shape=(15, 16, 11)
var_1994 = relay.var("var_1994", dtype = "int16", shape = (15, 16, 11))#candidate|1994|(15, 16, 11)|var|int16
bop_1995 = relay.minimum(call_1974.astype('uint8'), relay.reshape(var_1994.astype('uint8'), relay.shape_of(call_1974))) # shape=(15, 16, 11)
bop_1998 = relay.minimum(call_1975.astype('uint8'), relay.reshape(var_1994.astype('uint8'), relay.shape_of(call_1975))) # shape=(15, 16, 11)
func_1964_call = mod.get_global_var('func_1964')
func_1969_call = mutated_mod.get_global_var('func_1969')
var_2006 = relay.var("var_2006", dtype = "uint32", shape = (2, 52))#candidate|2006|(2, 52)|var|uint32
var_2007 = relay.var("var_2007", dtype = "int8", shape = (378,))#candidate|2007|(378,)|var|int8
var_2008 = relay.var("var_2008", dtype = "float64", shape = (720,))#candidate|2008|(720,)|var|float64
call_2005 = relay.TupleGetItem(func_1964_call(relay.reshape(var_2006.astype('uint32'), [104,]), relay.reshape(var_2007.astype('int8'), [3, 14, 9]), relay.reshape(var_2008.astype('float64'), [720,]), ), 2)
call_2009 = relay.TupleGetItem(func_1969_call(relay.reshape(var_2006.astype('uint32'), [104,]), relay.reshape(var_2007.astype('int8'), [3, 14, 9]), relay.reshape(var_2008.astype('float64'), [720,]), ), 2)
var_2010 = relay.var("var_2010", dtype = "float64", shape = (15, 16, 11))#candidate|2010|(15, 16, 11)|var|float64
bop_2011 = relay.bitwise_or(uop_1990.astype('uint16'), relay.reshape(var_2010.astype('uint16'), relay.shape_of(uop_1990))) # shape=(15, 16, 11)
bop_2014 = relay.bitwise_or(uop_1992.astype('uint16'), relay.reshape(var_2010.astype('uint16'), relay.shape_of(uop_1992))) # shape=(15, 16, 11)
output = relay.Tuple([bop_1981,bop_1995,call_2005,var_2006,var_2007,var_2008,bop_2011,])
output2 = relay.Tuple([bop_1984,bop_1998,call_2009,var_2006,var_2007,var_2008,bop_2014,])
func_2016 = relay.Function([var_1980,var_1994,var_2006,var_2007,var_2008,var_2010,], output)
mod['func_2016'] = func_2016
mod = relay.transform.InferType()(mod)
mutated_mod['func_2016'] = func_2016
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2016_call = mutated_mod.get_global_var('func_2016')
var_2018 = relay.var("var_2018", dtype = "int16", shape = (15, 16, 11))#candidate|2018|(15, 16, 11)|var|int16
var_2019 = relay.var("var_2019", dtype = "int16", shape = (15, 16, 11))#candidate|2019|(15, 16, 11)|var|int16
var_2020 = relay.var("var_2020", dtype = "uint32", shape = (2, 52))#candidate|2020|(2, 52)|var|uint32
var_2021 = relay.var("var_2021", dtype = "int8", shape = (378,))#candidate|2021|(378,)|var|int8
var_2022 = relay.var("var_2022", dtype = "float64", shape = (720,))#candidate|2022|(720,)|var|float64
var_2023 = relay.var("var_2023", dtype = "float64", shape = (15, 16, 11))#candidate|2023|(15, 16, 11)|var|float64
call_2017 = func_2016_call(var_2018,var_2019,var_2020,var_2021,var_2022,var_2023,)
output = call_2017
func_2024 = relay.Function([var_2018,var_2019,var_2020,var_2021,var_2022,var_2023,], output)
mutated_mod['func_2024'] = func_2024
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_2034 = func_1256_call()
call_2035 = func_1256_call()
output = relay.Tuple([call_2034,])
output2 = relay.Tuple([call_2035,])
func_2043 = relay.Function([], output)
mod['func_2043'] = func_2043
mod = relay.transform.InferType()(mod)
output = func_2043()
func_2044 = relay.Function([], output)
mutated_mod['func_2044'] = func_2044
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1351_call = mod.get_global_var('func_1351')
func_1353_call = mutated_mod.get_global_var('func_1353')
call_2045 = func_1351_call()
call_2046 = func_1351_call()
output = relay.Tuple([call_2045,])
output2 = relay.Tuple([call_2046,])
func_2047 = relay.Function([], output)
mod['func_2047'] = func_2047
mod = relay.transform.InferType()(mod)
output = func_2047()
func_2048 = relay.Function([], output)
mutated_mod['func_2048'] = func_2048
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1904_call = mod.get_global_var('func_1904')
func_1905_call = mutated_mod.get_global_var('func_1905')
call_2081 = relay.TupleGetItem(func_1904_call(), 1)
call_2082 = relay.TupleGetItem(func_1905_call(), 1)
var_2087 = relay.var("var_2087", dtype = "int16", shape = (15, 16, 11))#candidate|2087|(15, 16, 11)|var|int16
bop_2088 = relay.equal(call_2081.astype('bool'), relay.reshape(var_2087.astype('bool'), relay.shape_of(call_2081))) # shape=(15, 16, 11)
bop_2091 = relay.equal(call_2082.astype('bool'), relay.reshape(var_2087.astype('bool'), relay.shape_of(call_2082))) # shape=(15, 16, 11)
bop_2092 = relay.multiply(bop_2088.astype('int16'), relay.reshape(var_2087.astype('int16'), relay.shape_of(bop_2088))) # shape=(15, 16, 11)
bop_2095 = relay.multiply(bop_2091.astype('int16'), relay.reshape(var_2087.astype('int16'), relay.shape_of(bop_2091))) # shape=(15, 16, 11)
uop_2105 = relay.cosh(bop_2088.astype('float64')) # shape=(15, 16, 11)
uop_2107 = relay.cosh(bop_2091.astype('float64')) # shape=(15, 16, 11)
output = relay.Tuple([bop_2092,uop_2105,])
output2 = relay.Tuple([bop_2095,uop_2107,])
func_2122 = relay.Function([var_2087,], output)
mod['func_2122'] = func_2122
mod = relay.transform.InferType()(mod)
mutated_mod['func_2122'] = func_2122
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2123 = relay.var("var_2123", dtype = "int16", shape = (15, 16, 11))#candidate|2123|(15, 16, 11)|var|int16
func_2122_call = mutated_mod.get_global_var('func_2122')
call_2124 = func_2122_call(var_2123)
output = call_2124
func_2125 = relay.Function([var_2123], output)
mutated_mod['func_2125'] = func_2125
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2205 = relay.var("var_2205", dtype = "uint32", shape = (2, 8, 6))#candidate|2205|(2, 8, 6)|var|uint32
var_2206 = relay.var("var_2206", dtype = "uint32", shape = (2, 8, 6))#candidate|2206|(2, 8, 6)|var|uint32
bop_2207 = relay.minimum(var_2205.astype('uint32'), relay.reshape(var_2206.astype('uint32'), relay.shape_of(var_2205))) # shape=(2, 8, 6)
output = relay.Tuple([bop_2207,])
output2 = relay.Tuple([bop_2207,])
func_2212 = relay.Function([var_2205,var_2206,], output)
mod['func_2212'] = func_2212
mod = relay.transform.InferType()(mod)
mutated_mod['func_2212'] = func_2212
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2212_call = mutated_mod.get_global_var('func_2212')
var_2214 = relay.var("var_2214", dtype = "uint32", shape = (2, 8, 6))#candidate|2214|(2, 8, 6)|var|uint32
var_2215 = relay.var("var_2215", dtype = "uint32", shape = (2, 8, 6))#candidate|2215|(2, 8, 6)|var|uint32
call_2213 = func_2212_call(var_2214,var_2215,)
output = call_2213
func_2216 = relay.Function([var_2214,var_2215,], output)
mutated_mod['func_2216'] = func_2216
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2047_call = mod.get_global_var('func_2047')
func_2048_call = mutated_mod.get_global_var('func_2048')
call_2242 = relay.TupleGetItem(func_2047_call(), 0)
call_2243 = relay.TupleGetItem(func_2048_call(), 0)
var_2259 = relay.var("var_2259", dtype = "int8", shape = (3, 14, 9))#candidate|2259|(3, 14, 9)|var|int8
bop_2260 = relay.greater(call_2242.astype('bool'), relay.reshape(var_2259.astype('bool'), relay.shape_of(call_2242))) # shape=(3, 14, 9)
bop_2263 = relay.greater(call_2243.astype('bool'), relay.reshape(var_2259.astype('bool'), relay.shape_of(call_2243))) # shape=(3, 14, 9)
bop_2265 = relay.greater_equal(var_2259.astype('bool'), relay.reshape(call_2242.astype('bool'), relay.shape_of(var_2259))) # shape=(3, 14, 9)
bop_2268 = relay.greater_equal(var_2259.astype('bool'), relay.reshape(call_2243.astype('bool'), relay.shape_of(var_2259))) # shape=(3, 14, 9)
bop_2271 = relay.floor_mod(bop_2260.astype('float64'), relay.reshape(bop_2265.astype('float64'), relay.shape_of(bop_2260))) # shape=(3, 14, 9)
bop_2274 = relay.floor_mod(bop_2263.astype('float64'), relay.reshape(bop_2268.astype('float64'), relay.shape_of(bop_2263))) # shape=(3, 14, 9)
uop_2277 = relay.sinh(bop_2265.astype('float32')) # shape=(3, 14, 9)
uop_2279 = relay.sinh(bop_2268.astype('float32')) # shape=(3, 14, 9)
func_1315_call = mod.get_global_var('func_1315')
func_1318_call = mutated_mod.get_global_var('func_1318')
const_2281 = relay.const([5,-3,-8,-3,4,-3,-5,3,-2,-3,9,-4,-8,9,4,-6,-3,-4,-6,10,-7,8,-5,3,9,6,9,-2,-10,-8,-1,3,-5,-7,-6,-4,1,4,5,5,3,-2,3,9,5,10,-2,-3,2,8,8,-10,4,-5,1,-2,3,-5,10,-2,9,-2,-6,8,5,-3,-7,-2,-7,9,8,2,-7,-8,1,1,-5,-1,-10,-9,4,-7,-7,4,-7,1,6,9,-2,-10,10,-5,6,3,4,10,4,-10,8,4,-7,-5,2,-6], dtype = "uint32")#candidate|2281|(104,)|const|uint32
var_2282 = relay.var("var_2282", dtype = "uint8", shape = (3, 12))#candidate|2282|(3, 12)|var|uint8
call_2280 = relay.TupleGetItem(func_1315_call(relay.reshape(const_2281.astype('uint32'), [104,]), relay.reshape(var_2282.astype('uint8'), [36,]), ), 0)
call_2283 = relay.TupleGetItem(func_1318_call(relay.reshape(const_2281.astype('uint32'), [104,]), relay.reshape(var_2282.astype('uint8'), [36,]), ), 0)
output = relay.Tuple([bop_2271,uop_2277,call_2280,const_2281,var_2282,])
output2 = relay.Tuple([bop_2274,uop_2279,call_2283,const_2281,var_2282,])
func_2296 = relay.Function([var_2259,var_2282,], output)
mod['func_2296'] = func_2296
mod = relay.transform.InferType()(mod)
var_2297 = relay.var("var_2297", dtype = "int8", shape = (3, 14, 9))#candidate|2297|(3, 14, 9)|var|int8
var_2298 = relay.var("var_2298", dtype = "uint8", shape = (3, 12))#candidate|2298|(3, 12)|var|uint8
output = func_2296(var_2297,var_2298,)
func_2299 = relay.Function([var_2297,var_2298,], output)
mutated_mod['func_2299'] = func_2299
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1380_call = mod.get_global_var('func_1380')
func_1381_call = mutated_mod.get_global_var('func_1381')
call_2325 = func_1380_call()
call_2326 = func_1380_call()
output = call_2325
output2 = call_2326
func_2336 = relay.Function([], output)
mod['func_2336'] = func_2336
mod = relay.transform.InferType()(mod)
output = func_2336()
func_2337 = relay.Function([], output)
mutated_mod['func_2337'] = func_2337
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2347 = relay.var("var_2347", dtype = "float32", shape = (7, 7))#candidate|2347|(7, 7)|var|float32
uop_2348 = relay.asinh(var_2347.astype('float32')) # shape=(7, 7)
bop_2360 = relay.greater_equal(uop_2348.astype('bool'), relay.reshape(var_2347.astype('bool'), relay.shape_of(uop_2348))) # shape=(7, 7)
func_1315_call = mod.get_global_var('func_1315')
func_1318_call = mutated_mod.get_global_var('func_1318')
const_2376 = relay.const([[-1],[7],[-5],[8],[6],[-5],[-3],[9],[4],[-5],[4],[-9],[-10],[-3],[-2],[-4],[-6],[-4],[-3],[-8],[-6],[-7],[3],[10],[-3],[2],[-3],[-6],[-7],[7],[-5],[-10],[-9],[3],[-9],[-4],[-4],[-3],[8],[4],[-10],[-9],[-5],[-5],[-5],[10],[3],[-1],[6],[-8],[-7],[-9],[-3],[9],[8],[1],[8],[-9],[4],[-7],[-8],[10],[-9],[-6],[7],[10],[-6],[6],[-1],[-9],[8],[5],[1],[-3],[9],[5],[-9],[1],[-8],[-2],[10],[-10],[9],[6],[-2],[-7],[6],[-10],[-7],[3],[5],[3],[2],[1],[-10],[-4],[5],[7],[-1],[5],[9],[-4],[5],[1]], dtype = "uint32")#candidate|2376|(104, 1)|const|uint32
const_2377 = relay.const([10,-1,-5,7,-3,-10,-7,1,2,8,-5,-1,-1,5,10,7,5,3,1,9,-6,2,-6,9,5,5,-4,1,5,-7,-9,7,2,-5,9,-8], dtype = "uint8")#candidate|2377|(36,)|const|uint8
call_2375 = relay.TupleGetItem(func_1315_call(relay.reshape(const_2376.astype('uint32'), [104,]), relay.reshape(const_2377.astype('uint8'), [36,]), ), 0)
call_2378 = relay.TupleGetItem(func_1318_call(relay.reshape(const_2376.astype('uint32'), [104,]), relay.reshape(const_2377.astype('uint8'), [36,]), ), 0)
bop_2403 = relay.bitwise_or(const_2377.astype('uint64'), const_2376.astype('uint64')) # shape=(104, 36)
func_890_call = mod.get_global_var('func_890')
func_894_call = mutated_mod.get_global_var('func_894')
var_2409 = relay.var("var_2409", dtype = "int32", shape = (24,))#candidate|2409|(24,)|var|int32
call_2408 = relay.TupleGetItem(func_890_call(relay.reshape(var_2409.astype('int32'), [12, 2]), relay.reshape(var_2409.astype('float32'), [12, 2]), ), 3)
call_2410 = relay.TupleGetItem(func_894_call(relay.reshape(var_2409.astype('int32'), [12, 2]), relay.reshape(var_2409.astype('float32'), [12, 2]), ), 3)
uop_2415 = relay.acosh(uop_2348.astype('float64')) # shape=(7, 7)
var_2427 = relay.var("var_2427", dtype = "float64", shape = (7, 7))#candidate|2427|(7, 7)|var|float64
bop_2428 = relay.logical_xor(uop_2415.astype('int16'), relay.reshape(var_2427.astype('int16'), relay.shape_of(uop_2415))) # shape=(7, 7)
const_2431 = relay.const([[0.089706,8.507229,-6.177307,-0.827366,-9.541509,9.025603,6.130935],[-5.697272,6.876153,-6.274230,9.835345,-3.313912,-5.250448,1.387243],[6.785384,4.829061,4.111160,-6.371455,9.237148,3.073754,-8.769514],[9.091894,5.998286,-4.304392,5.225967,-4.429041,9.009503,-0.061271],[-1.352825,8.578604,-6.996277,-3.136093,7.973698,7.225087,-8.514861],[9.042204,-9.822807,-7.771516,-7.661757,-2.304011,7.016616,-7.704428],[-3.025423,5.629487,2.771303,-8.272689,4.746132,5.188662,7.660803]], dtype = "float64")#candidate|2431|(7, 7)|const|float64
bop_2432 = relay.mod(uop_2415.astype('float32'), relay.reshape(const_2431.astype('float32'), relay.shape_of(uop_2415))) # shape=(7, 7)
output = relay.Tuple([bop_2360,call_2375,bop_2403,call_2408,var_2409,bop_2428,bop_2432,])
output2 = relay.Tuple([bop_2360,call_2378,bop_2403,call_2410,var_2409,bop_2428,bop_2432,])
func_2437 = relay.Function([var_2347,var_2409,var_2427,], output)
mod['func_2437'] = func_2437
mod = relay.transform.InferType()(mod)
mutated_mod['func_2437'] = func_2437
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2437_call = mutated_mod.get_global_var('func_2437')
var_2439 = relay.var("var_2439", dtype = "float32", shape = (7, 7))#candidate|2439|(7, 7)|var|float32
var_2440 = relay.var("var_2440", dtype = "int32", shape = (24,))#candidate|2440|(24,)|var|int32
var_2441 = relay.var("var_2441", dtype = "float64", shape = (7, 7))#candidate|2441|(7, 7)|var|float64
call_2438 = func_2437_call(var_2439,var_2440,var_2441,)
output = call_2438
func_2442 = relay.Function([var_2439,var_2440,var_2441,], output)
mutated_mod['func_2442'] = func_2442
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2452 = relay.var("var_2452", dtype = "uint16", shape = (9, 11, 7))#candidate|2452|(9, 11, 7)|var|uint16
var_2453 = relay.var("var_2453", dtype = "uint16", shape = (9, 11, 7))#candidate|2453|(9, 11, 7)|var|uint16
bop_2454 = relay.less_equal(var_2452.astype('bool'), relay.reshape(var_2453.astype('bool'), relay.shape_of(var_2452))) # shape=(9, 11, 7)
bop_2460 = relay.equal(var_2452.astype('bool'), relay.reshape(bop_2454.astype('bool'), relay.shape_of(var_2452))) # shape=(9, 11, 7)
func_232_call = mod.get_global_var('func_232')
func_234_call = mutated_mod.get_global_var('func_234')
var_2469 = relay.var("var_2469", dtype = "float32", shape = (312,))#candidate|2469|(312,)|var|float32
call_2468 = relay.TupleGetItem(func_232_call(relay.reshape(var_2469.astype('float32'), [6, 13, 4])), 0)
call_2470 = relay.TupleGetItem(func_234_call(relay.reshape(var_2469.astype('float32'), [6, 13, 4])), 0)
bop_2471 = relay.logical_or(call_2468.astype('bool'), relay.reshape(var_2469.astype('bool'), relay.shape_of(call_2468))) # shape=(6, 13, 4)
bop_2474 = relay.logical_or(call_2470.astype('bool'), relay.reshape(var_2469.astype('bool'), relay.shape_of(call_2470))) # shape=(6, 13, 4)
output = relay.Tuple([bop_2460,bop_2471,])
output2 = relay.Tuple([bop_2460,bop_2474,])
func_2478 = relay.Function([var_2452,var_2453,var_2469,], output)
mod['func_2478'] = func_2478
mod = relay.transform.InferType()(mod)
var_2479 = relay.var("var_2479", dtype = "uint16", shape = (9, 11, 7))#candidate|2479|(9, 11, 7)|var|uint16
var_2480 = relay.var("var_2480", dtype = "uint16", shape = (9, 11, 7))#candidate|2480|(9, 11, 7)|var|uint16
var_2481 = relay.var("var_2481", dtype = "float32", shape = (312,))#candidate|2481|(312,)|var|float32
output = func_2478(var_2479,var_2480,var_2481,)
func_2482 = relay.Function([var_2479,var_2480,var_2481,], output)
mutated_mod['func_2482'] = func_2482
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1351_call = mod.get_global_var('func_1351')
func_1353_call = mutated_mod.get_global_var('func_1353')
call_2508 = func_1351_call()
call_2509 = func_1351_call()
output = relay.Tuple([call_2508,])
output2 = relay.Tuple([call_2509,])
func_2536 = relay.Function([], output)
mod['func_2536'] = func_2536
mod = relay.transform.InferType()(mod)
output = func_2536()
func_2537 = relay.Function([], output)
mutated_mod['func_2537'] = func_2537
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1701_call = mod.get_global_var('func_1701')
func_1702_call = mutated_mod.get_global_var('func_1702')
call_2552 = func_1701_call()
call_2553 = func_1701_call()
output = call_2552
output2 = call_2553
func_2559 = relay.Function([], output)
mod['func_2559'] = func_2559
mod = relay.transform.InferType()(mod)
mutated_mod['func_2559'] = func_2559
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2559_call = mutated_mod.get_global_var('func_2559')
call_2560 = func_2559_call()
output = call_2560
func_2561 = relay.Function([], output)
mutated_mod['func_2561'] = func_2561
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2559_call = mod.get_global_var('func_2559')
func_2561_call = mutated_mod.get_global_var('func_2561')
call_2592 = func_2559_call()
call_2593 = func_2559_call()
func_1801_call = mod.get_global_var('func_1801')
func_1804_call = mutated_mod.get_global_var('func_1804')
var_2601 = relay.var("var_2601", dtype = "float64", shape = (6,))#candidate|2601|(6,)|var|float64
call_2600 = relay.TupleGetItem(func_1801_call(relay.reshape(var_2601.astype('float64'), [1, 6])), 0)
call_2602 = relay.TupleGetItem(func_1804_call(relay.reshape(var_2601.astype('float64'), [1, 6])), 0)
output = relay.Tuple([call_2592,call_2600,var_2601,])
output2 = relay.Tuple([call_2593,call_2602,var_2601,])
func_2605 = relay.Function([var_2601,], output)
mod['func_2605'] = func_2605
mod = relay.transform.InferType()(mod)
var_2606 = relay.var("var_2606", dtype = "float64", shape = (6,))#candidate|2606|(6,)|var|float64
output = func_2605(var_2606)
func_2607 = relay.Function([var_2606], output)
mutated_mod['func_2607'] = func_2607
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2631 = relay.var("var_2631", dtype = "float32", shape = (5, 10))#candidate|2631|(5, 10)|var|float32
var_2632 = relay.var("var_2632", dtype = "float32", shape = (5, 10))#candidate|2632|(5, 10)|var|float32
bop_2633 = relay.less(var_2631.astype('bool'), relay.reshape(var_2632.astype('bool'), relay.shape_of(var_2631))) # shape=(5, 10)
func_158_call = mod.get_global_var('func_158')
func_163_call = mutated_mod.get_global_var('func_163')
const_2646 = relay.const([-1,-10,8,-3,9,9,-2,1,2,-7,-5,-4,-7,-10,-1,4,-9,9,8,-10,-6,8,1,-9,3,8,8,-2,8,-2,2,10,-10,1,-1,-4], dtype = "uint8")#candidate|2646|(36,)|const|uint8
var_2647 = relay.var("var_2647", dtype = "int32", shape = (13, 7))#candidate|2647|(13, 7)|var|int32
var_2648 = relay.var("var_2648", dtype = "float32", shape = (160,))#candidate|2648|(160,)|var|float32
call_2645 = relay.TupleGetItem(func_158_call(relay.reshape(const_2646.astype('uint8'), [12, 3]), relay.reshape(const_2646.astype('uint8'), [12, 3]), relay.reshape(var_2647.astype('int32'), [91,]), relay.reshape(var_2648.astype('float32'), [2, 80]), ), 3)
call_2649 = relay.TupleGetItem(func_163_call(relay.reshape(const_2646.astype('uint8'), [12, 3]), relay.reshape(const_2646.astype('uint8'), [12, 3]), relay.reshape(var_2647.astype('int32'), [91,]), relay.reshape(var_2648.astype('float32'), [2, 80]), ), 3)
func_662_call = mod.get_global_var('func_662')
func_666_call = mutated_mod.get_global_var('func_666')
const_2662 = relay.const([-5.904395,-2.612074,9.653005,-8.383463,2.245820,5.866502,5.350925,-6.039529,3.512027,8.934014,-6.048563,7.486962,-0.481064,-7.375745,-4.437213,8.363756,9.776403,4.733835,-9.944769,-1.076495,6.719640,1.206854,-6.255006,-8.912184,2.576559,9.534548,-4.084343,-9.776890,-4.539443,-7.215244,-6.738299,-8.640209,3.988580,-0.646090,1.529524,-6.608950,5.737350,-7.953276,4.127056,6.901891,-5.049801,-7.939360,1.857412,-4.357777,9.757554,-3.434079,-6.122108,4.388230,6.134688,1.997547,-0.870949,0.814234,-3.357862,-3.308066,-8.509285,-6.620390,-8.975049,0.779199,-2.245465,-5.732952,-9.286468,1.039666,4.653647,2.806745,2.445298,2.035982,-6.131979,3.405935,-6.664444,-5.896724,5.741681,-2.104544,5.965565,7.969217,3.365760,9.387089,-1.439006,-3.611118,0.341340,-3.559277,-7.702585,1.722422,-8.405794,1.230585,-9.421009,9.848258,9.913037,1.309103,7.113878,-8.558885,-4.356376,4.146759,-1.983136,-2.323531,8.368311,-1.664700], dtype = "float32")#candidate|2662|(96,)|const|float32
var_2663 = relay.var("var_2663", dtype = "float32", shape = (312,))#candidate|2663|(312,)|var|float32
call_2661 = relay.TupleGetItem(func_662_call(relay.reshape(const_2662.astype('float32'), [8, 12]), relay.reshape(var_2647.astype('int32'), [91, 1]), relay.reshape(var_2663.astype('float32'), [312,]), ), 6)
call_2664 = relay.TupleGetItem(func_666_call(relay.reshape(const_2662.astype('float32'), [8, 12]), relay.reshape(var_2647.astype('int32'), [91, 1]), relay.reshape(var_2663.astype('float32'), [312,]), ), 6)
uop_2681 = relay.sigmoid(const_2662.astype('float32')) # shape=(96,)
output = relay.Tuple([bop_2633,call_2645,const_2646,var_2647,var_2648,call_2661,var_2663,uop_2681,])
output2 = relay.Tuple([bop_2633,call_2649,const_2646,var_2647,var_2648,call_2664,var_2663,uop_2681,])
func_2684 = relay.Function([var_2631,var_2632,var_2647,var_2648,var_2663,], output)
mod['func_2684'] = func_2684
mod = relay.transform.InferType()(mod)
var_2685 = relay.var("var_2685", dtype = "float32", shape = (5, 10))#candidate|2685|(5, 10)|var|float32
var_2686 = relay.var("var_2686", dtype = "float32", shape = (5, 10))#candidate|2686|(5, 10)|var|float32
var_2687 = relay.var("var_2687", dtype = "int32", shape = (13, 7))#candidate|2687|(13, 7)|var|int32
var_2688 = relay.var("var_2688", dtype = "float32", shape = (160,))#candidate|2688|(160,)|var|float32
var_2689 = relay.var("var_2689", dtype = "float32", shape = (312,))#candidate|2689|(312,)|var|float32
output = func_2684(var_2685,var_2686,var_2687,var_2688,var_2689,)
func_2690 = relay.Function([var_2685,var_2686,var_2687,var_2688,var_2689,], output)
mutated_mod['func_2690'] = func_2690
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2536_call = mod.get_global_var('func_2536')
func_2537_call = mutated_mod.get_global_var('func_2537')
call_2695 = relay.TupleGetItem(func_2536_call(), 0)
call_2696 = relay.TupleGetItem(func_2537_call(), 0)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_2699 = func_1256_call()
call_2700 = func_1256_call()
output = relay.Tuple([call_2695,call_2699,])
output2 = relay.Tuple([call_2696,call_2700,])
func_2704 = relay.Function([], output)
mod['func_2704'] = func_2704
mod = relay.transform.InferType()(mod)
output = func_2704()
func_2705 = relay.Function([], output)
mutated_mod['func_2705'] = func_2705
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2708 = relay.var("var_2708", dtype = "int32", shape = (1, 13))#candidate|2708|(1, 13)|var|int32
var_2709 = relay.var("var_2709", dtype = "int32", shape = (8, 13))#candidate|2709|(8, 13)|var|int32
bop_2710 = relay.greater(var_2708.astype('bool'), var_2709.astype('bool')) # shape=(8, 13)
func_601_call = mod.get_global_var('func_601')
func_607_call = mutated_mod.get_global_var('func_607')
const_2719 = relay.const([3,-5,4,-3,5,-10,-9,6,6,10,5,10,6,5,7,3,-3,4,-2,-2,8,-1,-4,-7,-8,6,9,-9,6,-8,7,-5,-2,10,10,10], dtype = "uint8")#candidate|2719|(36,)|const|uint8
const_2720 = relay.const([[-9.953566,-3.906606],[7.896719,1.366665],[5.819003,-1.438299],[-4.852631,9.535254],[-3.209277,9.560600],[-8.352006,1.459438],[-1.771297,-6.449717],[4.871503,9.309117],[8.762944,0.359427],[8.276173,-4.614378],[-3.972316,3.763253],[4.946459,-8.537191],[-8.256732,8.447826],[4.693703,-6.661810],[5.001968,-2.318801],[-4.209612,-1.181629],[0.438034,-8.937523],[7.084814,-0.349659],[-7.473147,-9.817255],[-3.914275,8.399712],[-3.047053,7.418488],[-1.366634,4.195841],[3.402959,-9.194587],[-9.282491,7.956859],[-0.396496,-6.596716],[-1.092443,4.219936],[-4.640613,-7.625764],[-5.738611,5.220835],[-6.418398,-5.939331],[8.776713,-3.394080],[1.125875,-7.589100],[5.771358,-4.902258],[-1.129544,8.677451],[9.979357,7.035828],[5.323986,-7.188267],[0.358697,-3.889716],[3.674789,-7.155540],[6.503453,0.973151],[1.272380,-5.638786],[8.170219,-2.492431],[5.334736,-8.116541],[-7.330760,3.765768],[7.738674,7.803721],[4.270409,-4.279130],[-5.095475,4.796415],[0.659738,9.026198],[6.496456,4.832333],[-2.707834,-6.175301],[-5.051675,-8.759213],[-7.223875,-1.799187],[9.801727,-8.945270],[3.758644,3.919862],[-1.911432,8.819068],[4.758734,-6.258291],[-7.476615,2.721537],[-1.730290,3.452783],[-4.917279,8.902128],[2.815116,-9.390986],[-5.080627,5.347242],[2.764080,8.963210],[-9.253926,7.732676],[4.369076,-6.538319],[-9.233078,-8.489261],[2.350170,9.251972],[8.963858,-4.674492],[-6.260671,-6.402935],[9.293207,-7.573383],[-1.084495,-4.603829],[9.662283,5.292748],[-5.294667,-0.066106],[4.087577,4.762187],[-2.273264,6.845752],[7.565548,-0.451908],[-6.313425,-4.115517],[5.352334,8.713456],[9.376821,-3.030910],[3.268816,0.719523],[1.307651,-8.452645],[0.723819,9.006034],[-8.737291,9.045421],[-1.829678,3.851823],[-1.529713,-4.964321],[4.355436,-9.748822],[-0.778086,2.610444],[-2.644354,9.066656],[-9.518188,-3.604112],[6.439140,-0.909484],[-6.297050,5.587126],[6.761444,-7.051755],[3.121929,9.282124],[-0.118638,5.991020],[-5.335993,-0.181894],[5.938627,-6.233971],[-4.052938,4.262413],[-9.559792,1.041024],[-3.605170,6.573280],[-9.173599,2.452251],[2.813707,-6.336403],[-4.148247,-9.662894],[-7.742474,7.520991],[-0.139859,7.876924],[5.758355,-0.119911],[-1.464230,-6.989419],[-5.116666,6.813511],[3.217982,-8.917056],[9.234786,-6.572527],[-0.613492,2.632039],[-0.621481,4.968526],[-1.101069,7.097800],[4.154857,-5.181687],[4.091655,-3.700476],[-5.908994,-7.263909],[3.554536,-7.436828],[0.637170,-8.505423],[2.473482,-4.233820],[1.930221,-2.319508],[2.009623,5.593915],[0.882829,-9.843226],[-9.345026,9.719421],[-0.286452,1.873770],[-2.574489,2.123368],[-5.174852,-3.280915],[0.368899,4.159238],[-4.550998,-8.038298],[4.580899,-4.555717],[7.524906,-1.089273],[-5.712865,6.708520],[5.230169,-2.101040],[-2.854433,-0.841495],[0.607433,5.783148],[3.890205,9.858522],[-6.613356,9.892805],[5.394011,0.711547],[5.589124,7.551054],[8.903554,-4.118045],[3.422900,-8.044046],[-5.975270,-3.397780],[9.688749,-2.130995],[4.781939,-6.730593],[-0.706724,7.707701],[-8.372540,-7.309125],[-0.330916,1.718517],[-0.112469,2.482734],[-6.685931,-7.371923],[-7.240005,-5.975209],[-6.290912,-9.063349],[4.998024,7.930769],[3.751776,2.268025],[0.491200,0.287063],[6.872810,7.922627],[-2.973336,1.968978],[8.327368,-3.980076],[5.996970,5.302410],[8.661513,4.554232],[9.407434,-2.607609],[7.929043,-0.062858]], dtype = "float32")#candidate|2720|(156, 2)|const|float32
call_2718 = relay.TupleGetItem(func_601_call(relay.reshape(bop_2710.astype('uint32'), [13, 8]), relay.reshape(var_2709.astype('uint32'), [13, 8]), relay.reshape(const_2719.astype('uint8'), [36,]), relay.reshape(const_2720.astype('float32'), [312,]), relay.reshape(const_2720.astype('float32'), [312,]), ), 12)
call_2721 = relay.TupleGetItem(func_607_call(relay.reshape(bop_2710.astype('uint32'), [13, 8]), relay.reshape(var_2709.astype('uint32'), [13, 8]), relay.reshape(const_2719.astype('uint8'), [36,]), relay.reshape(const_2720.astype('float32'), [312,]), relay.reshape(const_2720.astype('float32'), [312,]), ), 12)
output = relay.Tuple([bop_2710,call_2718,const_2719,const_2720,])
output2 = relay.Tuple([bop_2710,call_2721,const_2719,const_2720,])
func_2739 = relay.Function([var_2708,var_2709,], output)
mod['func_2739'] = func_2739
mod = relay.transform.InferType()(mod)
mutated_mod['func_2739'] = func_2739
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2739_call = mutated_mod.get_global_var('func_2739')
var_2741 = relay.var("var_2741", dtype = "int32", shape = (1, 13))#candidate|2741|(1, 13)|var|int32
var_2742 = relay.var("var_2742", dtype = "int32", shape = (8, 13))#candidate|2742|(8, 13)|var|int32
call_2740 = func_2739_call(var_2741,var_2742,)
output = call_2740
func_2743 = relay.Function([var_2741,var_2742,], output)
mutated_mod['func_2743'] = func_2743
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2767 = relay.var("var_2767", dtype = "bool", shape = (8, 14, 2))#candidate|2767|(8, 14, 2)|var|bool
var_2768 = relay.var("var_2768", dtype = "bool", shape = (8, 14, 2))#candidate|2768|(8, 14, 2)|var|bool
bop_2769 = relay.logical_and(var_2767.astype('bool'), relay.reshape(var_2768.astype('bool'), relay.shape_of(var_2767))) # shape=(8, 14, 2)
output = bop_2769
output2 = bop_2769
func_2780 = relay.Function([var_2767,var_2768,], output)
mod['func_2780'] = func_2780
mod = relay.transform.InferType()(mod)
var_2781 = relay.var("var_2781", dtype = "bool", shape = (8, 14, 2))#candidate|2781|(8, 14, 2)|var|bool
var_2782 = relay.var("var_2782", dtype = "bool", shape = (8, 14, 2))#candidate|2782|(8, 14, 2)|var|bool
output = func_2780(var_2781,var_2782,)
func_2783 = relay.Function([var_2781,var_2782,], output)
mutated_mod['func_2783'] = func_2783
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2047_call = mod.get_global_var('func_2047')
func_2048_call = mutated_mod.get_global_var('func_2048')
call_2791 = relay.TupleGetItem(func_2047_call(), 0)
call_2792 = relay.TupleGetItem(func_2048_call(), 0)
output = relay.Tuple([call_2791,])
output2 = relay.Tuple([call_2792,])
func_2793 = relay.Function([], output)
mod['func_2793'] = func_2793
mod = relay.transform.InferType()(mod)
output = func_2793()
func_2794 = relay.Function([], output)
mutated_mod['func_2794'] = func_2794
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1701_call = mod.get_global_var('func_1701')
func_1702_call = mutated_mod.get_global_var('func_1702')
call_2816 = func_1701_call()
call_2817 = func_1701_call()
uop_2821 = relay.erf(call_2816.astype('float32')) # shape=(3, 14, 9)
uop_2823 = relay.erf(call_2817.astype('float32')) # shape=(3, 14, 9)
func_2559_call = mod.get_global_var('func_2559')
func_2561_call = mutated_mod.get_global_var('func_2561')
call_2825 = func_2559_call()
call_2826 = func_2559_call()
output = relay.Tuple([uop_2821,call_2825,])
output2 = relay.Tuple([uop_2823,call_2826,])
func_2833 = relay.Function([], output)
mod['func_2833'] = func_2833
mod = relay.transform.InferType()(mod)
mutated_mod['func_2833'] = func_2833
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2833_call = mutated_mod.get_global_var('func_2833')
call_2834 = func_2833_call()
output = call_2834
func_2835 = relay.Function([], output)
mutated_mod['func_2835'] = func_2835
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2868 = relay.var("var_2868", dtype = "float32", shape = ())#candidate|2868|()|var|float32
var_2869 = relay.var("var_2869", dtype = "float32", shape = (10, 12, 8))#candidate|2869|(10, 12, 8)|var|float32
bop_2870 = relay.greater(var_2868.astype('bool'), var_2869.astype('bool')) # shape=(10, 12, 8)
output = bop_2870
output2 = bop_2870
func_2874 = relay.Function([var_2868,var_2869,], output)
mod['func_2874'] = func_2874
mod = relay.transform.InferType()(mod)
var_2875 = relay.var("var_2875", dtype = "float32", shape = ())#candidate|2875|()|var|float32
var_2876 = relay.var("var_2876", dtype = "float32", shape = (10, 12, 8))#candidate|2876|(10, 12, 8)|var|float32
output = func_2874(var_2875,var_2876,)
func_2877 = relay.Function([var_2875,var_2876,], output)
mutated_mod['func_2877'] = func_2877
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2336_call = mod.get_global_var('func_2336')
func_2337_call = mutated_mod.get_global_var('func_2337')
call_2886 = func_2336_call()
call_2887 = func_2336_call()
func_2043_call = mod.get_global_var('func_2043')
func_2044_call = mutated_mod.get_global_var('func_2044')
call_2924 = relay.TupleGetItem(func_2043_call(), 0)
call_2925 = relay.TupleGetItem(func_2044_call(), 0)
output = relay.Tuple([call_2886,call_2924,])
output2 = relay.Tuple([call_2887,call_2925,])
func_2933 = relay.Function([], output)
mod['func_2933'] = func_2933
mod = relay.transform.InferType()(mod)
mutated_mod['func_2933'] = func_2933
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2933_call = mutated_mod.get_global_var('func_2933')
call_2934 = func_2933_call()
output = call_2934
func_2935 = relay.Function([], output)
mutated_mod['func_2935'] = func_2935
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1701_call = mod.get_global_var('func_1701')
func_1702_call = mutated_mod.get_global_var('func_1702')
call_2941 = func_1701_call()
call_2942 = func_1701_call()
func_2043_call = mod.get_global_var('func_2043')
func_2044_call = mutated_mod.get_global_var('func_2044')
call_2946 = relay.TupleGetItem(func_2043_call(), 0)
call_2947 = relay.TupleGetItem(func_2044_call(), 0)
uop_2949 = relay.cos(call_2946.astype('float64')) # shape=(3, 14, 9)
uop_2951 = relay.cos(call_2947.astype('float64')) # shape=(3, 14, 9)
func_2739_call = mod.get_global_var('func_2739')
func_2743_call = mutated_mod.get_global_var('func_2743')
const_2953 = relay.const([2,8,1,4,10,-4,-6,9,-4,4,-1,1,7], dtype = "int32")#candidate|2953|(13,)|const|int32
var_2954 = relay.var("var_2954", dtype = "int32", shape = (104,))#candidate|2954|(104,)|var|int32
call_2952 = relay.TupleGetItem(func_2739_call(relay.reshape(const_2953.astype('int32'), [1, 13]), relay.reshape(var_2954.astype('int32'), [8, 13]), ), 2)
call_2955 = relay.TupleGetItem(func_2743_call(relay.reshape(const_2953.astype('int32'), [1, 13]), relay.reshape(var_2954.astype('int32'), [8, 13]), ), 2)
output = relay.Tuple([call_2941,uop_2949,call_2952,const_2953,var_2954,])
output2 = relay.Tuple([call_2942,uop_2951,call_2955,const_2953,var_2954,])
func_2992 = relay.Function([var_2954,], output)
mod['func_2992'] = func_2992
mod = relay.transform.InferType()(mod)
mutated_mod['func_2992'] = func_2992
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2993 = relay.var("var_2993", dtype = "int32", shape = (104,))#candidate|2993|(104,)|var|int32
func_2992_call = mutated_mod.get_global_var('func_2992')
call_2994 = func_2992_call(var_2993)
output = call_2994
func_2995 = relay.Function([var_2993], output)
mutated_mod['func_2995'] = func_2995
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1380_call = mod.get_global_var('func_1380')
func_1381_call = mutated_mod.get_global_var('func_1381')
call_2999 = func_1380_call()
call_3000 = func_1380_call()
output = relay.Tuple([call_2999,])
output2 = relay.Tuple([call_3000,])
func_3007 = relay.Function([], output)
mod['func_3007'] = func_3007
mod = relay.transform.InferType()(mod)
mutated_mod['func_3007'] = func_3007
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3007_call = mutated_mod.get_global_var('func_3007')
call_3008 = func_3007_call()
output = call_3008
func_3009 = relay.Function([], output)
mutated_mod['func_3009'] = func_3009
mutated_mod = relay.transform.InferType()(mutated_mod)
func_1256_call = mod.get_global_var('func_1256')
func_1258_call = mutated_mod.get_global_var('func_1258')
call_3017 = func_1256_call()
call_3018 = func_1256_call()
var_3030 = relay.var("var_3030", dtype = "int8", shape = (3, 14, 9))#candidate|3030|(3, 14, 9)|var|int8
bop_3031 = relay.bitwise_xor(call_3017.astype('int8'), relay.reshape(var_3030.astype('int8'), relay.shape_of(call_3017))) # shape=(3, 14, 9)
bop_3034 = relay.bitwise_xor(call_3018.astype('int8'), relay.reshape(var_3030.astype('int8'), relay.shape_of(call_3018))) # shape=(3, 14, 9)
bop_3035 = relay.logical_xor(var_3030.astype('uint64'), relay.reshape(bop_3031.astype('uint64'), relay.shape_of(var_3030))) # shape=(3, 14, 9)
bop_3038 = relay.logical_xor(var_3030.astype('uint64'), relay.reshape(bop_3034.astype('uint64'), relay.shape_of(var_3030))) # shape=(3, 14, 9)
uop_3052 = relay.sin(var_3030.astype('float64')) # shape=(3, 14, 9)
uop_3056 = relay.log2(bop_3035.astype('float32')) # shape=(3, 14, 9)
uop_3058 = relay.log2(bop_3038.astype('float32')) # shape=(3, 14, 9)
func_2122_call = mod.get_global_var('func_2122')
func_2125_call = mutated_mod.get_global_var('func_2125')
const_3064 = relay.const([[-4,3,-3,1,-10,8,4,3,-8,7,-8,-3,9,6,3,-5,4,-8,-9,-8,2,2,6,9,-6,-1,-1,3,-6,-1,1,-9,-6,-4,4,3,10,5,1,-7,2,-1,4,2,-1,10,-1,-8,4,9,4,7,9,9,5,7,7,10,-2,-1,5,6,10,2,-6,5,5,3,3,-6,-1,8,2,7,-10,6,2,1,6,6,-4,-1,3,7,-7,6,5,-5,-2,3,-3,-7,-3,-6,2,10,-5,9,6,-3,6,-2,-3,-1,1,7,3,10,-2,4,8,-4,-6,10,6,6,5,6,-6,1,1,3,1,9,-10,9,-8,3,-5,-7,-8,-6,2,9,1,5,-9,5,4,5,-8,1,9,-1,7,-7,2,6,-5,9,-8,-9,7,-6,9,-2,-6,-4,-5,9,-6,6,10,-9,-8,3,5,2,-6,-10,2,4,4,-4,7,-10,-10,9,8,-6,-8,-4,-3,-4,5,-5,-9,6,9,2,-3,-8,-1,-4,-7,-7,1,8,8,-4,10,-3,-8,-10,-6,4,-9,2,-1,10,7,3,-1,-2,4,5,5,3,-4,4,-7,9,3,5,8,-4,-3,4,9,-2,-2,-9,4,2,-4,7,-7,7,-10,3,-8,-4,5,2,4,1,2,4,-3,-1,5,10,6,1,2,3,-3,8,5,-4,-7,-4,6,-4,-6,-7,6,-2,-7,-8,7,6,8,-7,-7,1,-7,-4,3,-3,-4,2,8,-10,-7,3,-10,1,4,-1,1,6,9,5,7,-2,10,10,-6,-4,-1,-6,-5,-8,-6,-1,-5,9,6,-5,-3,3,9,-1,4,-9,-2,5,-1,-8,6,-6,-8,-2,-8,-7,-3,2,4,6,-1,-2,-9,2,-3,-9,-2,1,-9,-2,1,-7,10,4,-7,1,-7,-2,-3,6,-5,-7,9,4,-1,4,8,6,-5,3,3,9,-8,-6,-3,8,-9,-4,-1,6,-7,-5,6,-4,-2,-10,-10,4,-9,-10,7,10,-1,4,-5,7,3,-8,1,-7,-8,7,5,9,-4,-6,-4,-6,-3,10,-1,1,-7,2,-10,1,5,9,1,-3,6,7,6,-7,-4,-8,-4,-10,7,-8,-10,-6,-9,4,-2,7,6,7,7,-2,3,-9,-10,-7,-8,3,-8,-2,5,-8,-3,-4,9,3,-9,1,7,-8,-7,-4,-2,5,-8,-6,3,2,9,6,-1,-7,4,-2,-6,-5,-5,-1,8,8,3,10,10,-1,-10,-10,-6,3,5,-8,-6,-5,4,10,2,-9,-4,10,-3,-4,9,-5,-10,-2,4,5,5,-2,6,2,-2,2,1,1,-4,-8,8,6,4,1,10,-7,-8,5,-1,-3,-9,8,-6,4,3,-2,7,-8,-8,8,-9,-2,-2,-10,-5,5,-5,-3,-3,2,1,8,2,1,-7,7,-4,2,7,10,-2,-5,1,1,-7,9,-4,7,2,-7,7,-6,2,-10,9,-2,-6,-9,5,2,-10,-9,7,-4,4,-2,3,3,-8,-8,-2,8,-7,8,7,-5,10,-5,10,9,-1,4,-1,-3,1,-7,-2,-4,4,-3,1,-9,-2,9,6,-5,-9,10,9,10,-3,-2,2,5,7,8,-6,8,-3,-1,4,-5,6,-4,9,4,1,-8,-5,10,1,-9,-6,4,-3,-8,10,10,9,2,10,-9,-5,5,-7,1,4,4,6,-9,2,5,-1,1,-9,-5,10,-9,10,-6,-10,5,-6,3,-9,-10,10,-5,10,-10,-9,-7,10,6,1,-6,-9,-6,2,-9,-2,1,4,-4,2,-3,-8,9,-6,7,-7,8,-4,2,-2,4,-2,7,-1,-2,6,3,6,3,7,9,2,3,-10,1,-8,9,10,-7,8,3,5,-7,7,2,7,-9,7,-5,7,-10,-1,-6,-6,1,2,-5,1,7,1,2,-1,8,-3,8,9,6,8,10,-2,-7,8,-1,-5,6,10,-5,-9,10,8,-3,-5,-6,8,6,-8,6,6,-4,8,6,-8,-8,5,-8,-7,1,-7,-6,4,8,-6,9,-4,10,5,-1,-8,-7,-8,-8,-7,6,6,10,5,-4,-4,10,8,-3,7,-5,2,9,-8,-6,8,8,5,5,-7,10,6,1,-1,8,-8,-4,4,2,-2,8,10,5,-5,7,3,-5,-7,-2,1,-8,-2,1,8,9,-2,2,-2,-9,-9,-1,4,-3,4,-1,7,6,-2,5,-10,1,-3,4,-2,-4,-9,-5,-3,-6,-4,-6,4,10,10,-7,-4,-2,3,-10,-1,10,-1,5,-5,-3,-5,2,10,9,-1,-8,-7,10,8,-9,-6,1,3,-1,6,-9,-10,2,3,8,10,-7,-8,-8,-8,-5,5,-2,-1,-7,5,-10,7,7,-5,-3,2,-3,8,-9,8,1,-2,-4,-5,2,6,-7,5,-4,8,5,-7,2,-10,2,-7,-3,-5,3,-7,10,-8,8,-2,4,-10,-1,9,-2,6,-7,8,-3,-7,3,-5,1,1,-9,-2,2,-3,-10,-6,-9,8,-5,-8,-9,7,2,4,-5,3,-7,-6,8,2,2,-10,-8,1,-5,2,-7,-6,2,10,-7,3,9,-3,-1,-5,-2,-5,5,4,6,-4,4,2,9,3,-7,10,-6,-3,9,-1,4,9,-4,7,-8,5,-10,8,-8,-5,-9,-1,7,3,3,-5,10,4,7,-5,4,-7,-6,-4,-7,-1,4,-1,-7,7,8,7,-7,5,-4,-9,10,-5,-2,-3,9,-5,-6,-6,-3,-3,8,-3,2,-9,8,-7,5,-5,4,-3,-1,-8,7,2,-8,4,-6,10,-5,-4,-8,-9,4,9,-3,-3,-3,5,-4,1,-3,-7,-2,-7,-6,9,-4,7,6,8,-8,-6,-1,-2,-10,-6,-10,2,-9,-8,-1,-9,7,5,-1,3,-5,2,-1,-10,7,-2,8,-8,4,-2,-7,10,-8,-7,9,10,-8,9,-9,10,4,5,-7,-8,3,6,10,1,-2,-9,-8,-4,-3,6,3,-10,-10,-10,7,8,8,7,-1,-8,-9,-3,-5,-3,-1,2,7,-10,6,3,-9,-3,-8,-7,-1,9,1,6,8,-8,-10,-2,2,-5,-5,-2,-1,1,5,8,2,-7,1,3,-3,-8,-1,2,-6,2,5,-6,-10,8,5,-3,-5,-4,-2,-4,7,-9,-6,-4,8,-4,1,-1,-6,-6,7,8,10,8,8,-10,-6,5,-10,6,6,-10,9,10,2,-6,7,-3,-5,-10,3,-10,-7,-2,-4,4,10,9,-7,7,-4,10,-7,8,-2,-5,-6,-5,-9,-1,-7,5,1,9,2,7,6,-7,8,5,2,6,-8,-5,10,-2,-3,4,-2,1,-8,-9,-4,7,10,6,5,-3,7,-10,6,-5,10,4,-2,-3,3,-7,8,3,-8,-3,-2,-8,4,5,-9,-10,10,8,-1,6,-6,-3,5,-4,4,9,-4,5,4,-8,2,3,-4,-8,3,8,-5,6,4,6,-7,-6,-10,9,-2,-9,6,8,5,4,-10,-9,-1,4,5,5,-4,3,1,10,-2,2,2,-10,8,5,6,2,9,7,7,-5,3,5,-1,-9,-7,-1,-3,-6,-6,-8,6,-4,-5,6,10,-2,-9,-7,5,5,9,1,-8,-10,9,-9,-7,10,1,-5,-7,-6,-6,-1,10,-5,2,10,4,-6,-9,9,-5,9,5,9,-1,-2,-7,-9,9,-6,1,-2,-10,-9,-3,-5,7,4,-8,-6,5,9,3,9,-4,-2,4,10,-4,3,7,-4,3,-6,1,3,4,-10,-10,6,7,-6,5,5,5,5,-7,4,-8,-3,6,9,-8,4,7,1,-9,-4,-3,4,-4,-7,9,-3,6,8,-1,-2,5,8,-8,-7,-6,-3,-8,8,-7,5,10,5,-1,8,9,5,-7,-7,2,4,-2,10,6,-4,-4,-9,4,-10,6,-10,-1,-10,-8,5,-10,7,-8,4,4,6,-6,2,-6,8,4,8,3,-4,-4,10,-8,8,3,-1,-9,-4,-10,6,3,-10,1,10,6,4,5,-4,2,-10,-3,1,5,-8,6,-5,-7,5,2,2,1,8,2,8,5,9,8,1,-9,-2,-2,-3,-3,4,6,-5,-4,3,-6,-1,3,-10,-2,-6,-9,-6,-7,-9,9,-6,-1,-4,-4,4,1,3,-7,-3,-4,-10,9,-6,-8,7,-10,2,3,-8,-4,9,8,-7,6,4,3,5,-1,-8,-5,6,-9,-7,-6,-1,7,-6,-3,2,5,9,9,9,4,8,-2,5,5,-4,-6,3,-9,2,2,-1,-9,4,-4,-1,2,-8,-6,-8,6,1,-8,-7,3,-9,3,-1,-10,-4,2,5,-5,-8,7,-1,7,6,8,2,-3,6,3,-2,-6,9,2,-10,8,-7,6,-1,3,10,-1,5,-5,1,10,-1,8,4,6,8,-4,3,4,-2,-1,10,-7,3,-6,10,-8,4,-2,7,-6,4,-1,-1,2,3,8,-3,-8,6,3,-2,6,-8,-6,-8,1,2,2,-6,-7,-6,-2,-9,-7,6,9,-10,8,7,4,-6,-9,6,4,-4,1,-5,10,7,8,-5,5,-4,-5,7,-6,6,-3,-7,-6,-7,2,2,-7,9,-4,4,2,-10,-1,-10,-10,-2,-9,-8,-3,-2,9,7,4,-1,2,8,3,3,-5,-1,-7,7,-6,-3,-2,-4,-10,6,8,-4,5,4,-10,-2,-8,-1,7,-5,3,-2,2,5,-6,7,10,-4,-1,-3,-3,-7,7,8,-6,-9,-10,3,8,5,8,-9,1,-3,2,-5,-8,-8,10,3,10,2,-6,3,4,-7,9,-1,9,-5,10,8,2,-5,3,-8,-1,-4,-7,10,-6,-4,-2,7,8,4,2,6,-9,8,-10,-4,7,9,-6,5,-5,-7,3,10,-6,9,-8,3,7,-9,8,-1,3,8,10,-1,2,4,2,7,2,-5,3,10,-9,-1,4,1,2,1,5,-9,-2,2,-8,-1,9,3,-7,-5,1,5,-8,7,8,-3,9,-1,4,-4,-4,-3,2,6,8,-6,-7,-6,-7,-4,2,9,-7,2,-4,6,-3,3,-6,5,-10,4,-3,-10,-5,7,-2,-9,-5,-2,7,-6,-8,6,-8,7,5,7,2,-10,4,4,-4,-4,-4,-9,-6,-3,-6,10,1,3,-9,-4,10,9,3,-9,-8,9,1,5,3,-4,7,2,-1,4,7,-2,5,-8,-7,-7,-10,-9,4,-10,1,7,-10,-6,4,-1,-9,8,2,-2,3,-4,-8,7,6,-3,3,-5,-9,-2,-10,-5,-5,2,-9,8,6,6,-4,9,2,7,2,-8,4,5,4,4,3,7,-2,5,-7,-5,-5,3,8,3,-3,-2,-3,-10,9,6,-1,3,8,-3,-8,-7,3,-10,-9,3,-6,-6,4,-5,1,-4,-4,6,-2,8,7,4,-5,-3,9,-4,-10,1,-10,6,8,4,6,4,10,9,7,-7,8,-7,-3,10,-3,1,7,-7,-7,7,-2,3,-4,9,-6,4,-4,2,1,-5,5,8,-3,-9,7,-6,-6,10,1,-3,-4,10,-7,-9,6,-5,-8,-3,9,1,-7,-2,6,-2,1,-4,2,5,8,-10,-6,3,7,6,7,9,-6,6,-4,9,-6,10,-8,-9,-1,-9,3,2,10,-7,7,9,-8,-7,5,-8,10,-9,7,5,9,4,7,-6,-5,-4,-1,-8,-9,-3,-3,-5,1,1,6,-5,-1,-8,8,-7,-9,7,-5,-8,10,-3,2,-5,-4,2,-2,4,7,-2,-10,-8,-10,-6,10,3,7,2,-4,4,-1,9,-5,-7,-6,7,-9,-4,2,8,-6,-1,-7,-2,-9,-10,-5,6,-1,-2,3,5,-6,1,-9,-9,9,7,2,8,8,-2,6,-2,-10,-8,-4,4,-9,8,-3,-2,7,-10,-1,-10,6,-1,-9,-3,-7,-1,1,-1,-10,8,-5,8,2,-1,-4,-1,1,5,4,1,9,2,-7,2,-3,-3,-5,-10,-10,-7,2,-7,9,-4,-1,6,-7,2,3,-3,8,-7,-10,-9,10,5,3,8,5,9,-7,8,5,-9,-5,7,10,-8,2,1,1,-5,-7,10,-5,7,-2,-7,6,5,10,-7,-5,3,10,9,-7,10,7,7,7,-5,1,-7,-10,8,-4,2,-10,9,8,-3,-2,10,-1,3,-1,8,-2,6,-7,2,9,-3,9,-1,9,4,7,3,1,7,-6,-5,-7,-8,7,-6,-6,-7,10,-3,-10,-2,-1,6,-3,-5,5,5,10,8,5,-4,-4,7,-7,-3,-5,5,-2,-9,3,3,-5,3,4,-3,-5,2,-9,3,7,-6,-3,-2,3,9,7,-5,-2,10,-10,-5,-1,2,5,-8,3,7,-7,10,-10,6,-3,-10,1,-1,4,-6,4,-6,5,-3,8,-4,-9,-4,7,6,5,7,7,-7,-5,-5,2,1,-7,-6,4,-1,6,2,7,3,6,-8,-7,6,-7,-4,4,6,10,6,2,7,-3,1,-8,4,6,9,-6,-7,7,3,9,-3,-3,-2,-6,-5,9,1,-2,-1,6,2,-6,8,-2,-7,4,1,8,-2,-4,7,3,-9,2,5,-2,3,-8,3,4,3,7,5,7,2,-4,-5,9,-4,-7,-1,6,7,7,-5,8,-8,2,-2,1,-4,-7,4,4,6,-6,10,-1,7,-7,-10,10,3,1,5,-3,7,-8,3,10,5,6,-10,-2,-6,1,7,-2,-7,6,-7,1,-7,6,2,9,3,-3,-2,-2,-6,-2,6,1,-1,3,6,-3,-4,-2,4,-6,-5,-2,3,-5,6,-4,-3,4,-4,1,5,-2,2,9,-7,5,-3,-1,10,7,-5,-8,8,2,3,9,-7,8,8,7,10,7,8,2,-4,-6,2,7,10,-1,7,10,2,5,-3,5,-4,-9,-2,4,3,-5,6,6,-5,6,6,6,-3,-10,-3,2,-9,4,5,7,8,3,-8,-9,-4,3]], dtype = "int16")#candidate|3064|(1, 2640)|const|int16
call_3063 = relay.TupleGetItem(func_2122_call(relay.reshape(const_3064.astype('int16'), [15, 16, 11])), 0)
call_3065 = relay.TupleGetItem(func_2125_call(relay.reshape(const_3064.astype('int16'), [15, 16, 11])), 0)
func_2933_call = mod.get_global_var('func_2933')
func_2935_call = mutated_mod.get_global_var('func_2935')
call_3072 = relay.TupleGetItem(func_2933_call(), 0)
call_3073 = relay.TupleGetItem(func_2935_call(), 0)
func_2437_call = mod.get_global_var('func_2437')
func_2442_call = mutated_mod.get_global_var('func_2442')
var_3079 = relay.var("var_3079", dtype = "float32", shape = (49,))#candidate|3079|(49,)|var|float32
const_3080 = relay.const([1,-7,7,6,-1,-4,5,1,-7,-6,2,-10,1,-1,9,2,-1,7,-9,9,-7,-10,-10,10], dtype = "int32")#candidate|3080|(24,)|const|int32
call_3078 = relay.TupleGetItem(func_2437_call(relay.reshape(var_3079.astype('float32'), [7, 7]), relay.reshape(const_3080.astype('int32'), [24,]), relay.reshape(var_3079.astype('float64'), [7, 7]), ), 5)
call_3081 = relay.TupleGetItem(func_2442_call(relay.reshape(var_3079.astype('float32'), [7, 7]), relay.reshape(const_3080.astype('int32'), [24,]), relay.reshape(var_3079.astype('float64'), [7, 7]), ), 5)
func_2780_call = mod.get_global_var('func_2780')
func_2783_call = mutated_mod.get_global_var('func_2783')
var_3084 = relay.var("var_3084", dtype = "bool", shape = (2, 112))#candidate|3084|(2, 112)|var|bool
call_3083 = func_2780_call(relay.reshape(var_3084.astype('bool'), [8, 14, 2]), relay.reshape(var_3084.astype('bool'), [8, 14, 2]), )
call_3085 = func_2780_call(relay.reshape(var_3084.astype('bool'), [8, 14, 2]), relay.reshape(var_3084.astype('bool'), [8, 14, 2]), )
func_2212_call = mod.get_global_var('func_2212')
func_2216_call = mutated_mod.get_global_var('func_2216')
const_3088 = relay.const([[7,-3],[3,-4],[2,2],[10,9],[10,10],[-3,4],[-10,10],[-5,3],[6,-10],[-1,-1],[-6,3],[4,7],[9,-1],[8,-2],[8,-10],[-6,-6],[-1,1],[-9,-3],[5,5],[7,3],[-9,2],[4,6],[6,5],[-3,4],[7,3],[6,-3],[3,10],[9,-8],[8,6],[4,3],[1,3],[1,5],[-3,-4],[-6,1],[6,-9],[-5,-3],[-7,-6],[-2,2],[-9,7],[-4,-7],[-3,-7],[-9,-6],[10,-10],[-2,-7],[-2,9],[-1,9],[-6,9],[-3,7]], dtype = "uint32")#candidate|3088|(48, 2)|const|uint32
call_3087 = relay.TupleGetItem(func_2212_call(relay.reshape(const_3088.astype('uint32'), [2, 8, 6]), relay.reshape(const_3088.astype('uint32'), [2, 8, 6]), ), 0)
call_3089 = relay.TupleGetItem(func_2216_call(relay.reshape(const_3088.astype('uint32'), [2, 8, 6]), relay.reshape(const_3088.astype('uint32'), [2, 8, 6]), ), 0)
output = relay.Tuple([uop_3052,uop_3056,call_3063,const_3064,call_3072,call_3078,var_3079,const_3080,call_3083,var_3084,call_3087,const_3088,])
output2 = relay.Tuple([uop_3052,uop_3058,call_3065,const_3064,call_3073,call_3081,var_3079,const_3080,call_3085,var_3084,call_3089,const_3088,])
func_3090 = relay.Function([var_3030,var_3079,var_3084,], output)
mod['func_3090'] = func_3090
mod = relay.transform.InferType()(mod)
var_3091 = relay.var("var_3091", dtype = "int8", shape = (3, 14, 9))#candidate|3091|(3, 14, 9)|var|int8
var_3092 = relay.var("var_3092", dtype = "float32", shape = (49,))#candidate|3092|(49,)|var|float32
var_3093 = relay.var("var_3093", dtype = "bool", shape = (2, 112))#candidate|3093|(2, 112)|var|bool
output = func_3090(var_3091,var_3092,var_3093,)
func_3094 = relay.Function([var_3091,var_3092,var_3093,], output)
mutated_mod['func_3094'] = func_3094
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2704_call = mod.get_global_var('func_2704')
func_2705_call = mutated_mod.get_global_var('func_2705')
call_3140 = relay.TupleGetItem(func_2704_call(), 1)
call_3141 = relay.TupleGetItem(func_2705_call(), 1)
func_2704_call = mod.get_global_var('func_2704')
func_2705_call = mutated_mod.get_global_var('func_2705')
call_3157 = relay.TupleGetItem(func_2704_call(), 1)
call_3158 = relay.TupleGetItem(func_2705_call(), 1)
func_2874_call = mod.get_global_var('func_2874')
func_2877_call = mutated_mod.get_global_var('func_2877')
const_3169 = relay.const(-1.903094, dtype = "float32")#candidate|3169|()|const|float32
var_3170 = relay.var("var_3170", dtype = "float32", shape = (960,))#candidate|3170|(960,)|var|float32
call_3168 = func_2874_call(relay.reshape(const_3169.astype('float32'), []), relay.reshape(var_3170.astype('float32'), [10, 12, 8]), )
call_3171 = func_2874_call(relay.reshape(const_3169.astype('float32'), []), relay.reshape(var_3170.astype('float32'), [10, 12, 8]), )
output = relay.Tuple([call_3140,call_3157,call_3168,const_3169,var_3170,])
output2 = relay.Tuple([call_3141,call_3158,call_3171,const_3169,var_3170,])
func_3173 = relay.Function([var_3170,], output)
mod['func_3173'] = func_3173
mod = relay.transform.InferType()(mod)
var_3174 = relay.var("var_3174", dtype = "float32", shape = (960,))#candidate|3174|(960,)|var|float32
output = func_3173(var_3174)
func_3175 = relay.Function([var_3174], output)
mutated_mod['func_3175'] = func_3175
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3177 = relay.const(-1, dtype = "uint64")#candidate|3177|()|const|uint64
var_3178 = relay.var("var_3178", dtype = "uint64", shape = (15, 4))#candidate|3178|(15, 4)|var|uint64
bop_3179 = relay.less_equal(const_3177.astype('bool'), var_3178.astype('bool')) # shape=(15, 4)
uop_3184 = relay.log(var_3178.astype('float32')) # shape=(15, 4)
func_2559_call = mod.get_global_var('func_2559')
func_2561_call = mutated_mod.get_global_var('func_2561')
call_3204 = func_2559_call()
call_3205 = func_2559_call()
output = relay.Tuple([bop_3179,uop_3184,call_3204,])
output2 = relay.Tuple([bop_3179,uop_3184,call_3205,])
func_3221 = relay.Function([var_3178,], output)
mod['func_3221'] = func_3221
mod = relay.transform.InferType()(mod)
var_3222 = relay.var("var_3222", dtype = "uint64", shape = (15, 4))#candidate|3222|(15, 4)|var|uint64
output = func_3221(var_3222)
func_3223 = relay.Function([var_3222], output)
mutated_mod['func_3223'] = func_3223
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2704_call = mod.get_global_var('func_2704')
func_2705_call = mutated_mod.get_global_var('func_2705')
call_3228 = relay.TupleGetItem(func_2704_call(), 0)
call_3229 = relay.TupleGetItem(func_2705_call(), 0)
func_2793_call = mod.get_global_var('func_2793')
func_2794_call = mutated_mod.get_global_var('func_2794')
call_3240 = relay.TupleGetItem(func_2793_call(), 0)
call_3241 = relay.TupleGetItem(func_2794_call(), 0)
output = relay.Tuple([call_3228,call_3240,])
output2 = relay.Tuple([call_3229,call_3241,])
func_3245 = relay.Function([], output)
mod['func_3245'] = func_3245
mod = relay.transform.InferType()(mod)
mutated_mod['func_3245'] = func_3245
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3245_call = mutated_mod.get_global_var('func_3245')
call_3246 = func_3245_call()
output = call_3246
func_3247 = relay.Function([], output)
mutated_mod['func_3247'] = func_3247
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3263 = relay.const([[[True,False,True,False,True,True,False,False,False,False,True,False]],[[False,True,False,False,False,False,False,True,False,False,True,True]],[[True,False,True,False,True,False,False,False,True,False,False,False]],[[True,False,False,True,True,True,True,True,True,True,True,True]],[[False,True,False,False,True,False,True,True,False,False,True,True]],[[True,True,False,False,True,False,False,True,True,False,False,False]],[[False,False,True,True,True,False,False,True,False,False,True,True]],[[False,True,False,True,True,True,False,False,True,False,False,False]],[[True,False,True,False,True,False,False,True,False,True,False,True]],[[True,False,True,True,False,False,True,True,True,True,False,True]],[[True,False,True,False,True,False,False,False,False,False,False,True]]], dtype = "bool")#candidate|3263|(11, 1, 12)|const|bool
const_3264 = relay.const([[[False,False,True,False,True,False,False,False,False,True,True,True]],[[False,False,False,False,True,True,False,True,True,True,True,False]],[[True,True,False,True,True,False,True,True,False,False,False,True]],[[True,False,False,True,True,True,False,True,True,False,True,True]],[[False,False,False,True,True,True,True,True,False,True,False,True]],[[True,False,True,True,False,True,True,True,True,True,False,True]],[[False,False,False,True,True,False,False,True,False,False,False,True]],[[True,False,False,False,True,True,True,True,True,False,False,True]],[[True,True,True,True,False,True,False,False,False,False,False,True]],[[False,False,False,False,True,False,False,False,False,False,False,False]],[[True,False,True,False,False,True,False,False,True,True,True,True]]], dtype = "bool")#candidate|3264|(11, 1, 12)|const|bool
bop_3265 = relay.logical_and(const_3263.astype('bool'), relay.reshape(const_3264.astype('bool'), relay.shape_of(const_3263))) # shape=(11, 1, 12)
output = relay.Tuple([bop_3265,])
output2 = relay.Tuple([bop_3265,])
func_3269 = relay.Function([], output)
mod['func_3269'] = func_3269
mod = relay.transform.InferType()(mod)
mutated_mod['func_3269'] = func_3269
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3269_call = mutated_mod.get_global_var('func_3269')
call_3270 = func_3269_call()
output = call_3270
func_3271 = relay.Function([], output)
mutated_mod['func_3271'] = func_3271
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3296 = relay.const([[[9.629483,-8.191432,-2.456367,-7.805662,-2.033701,-2.900687,6.164410,8.239308,-2.274237,4.324369,4.627190],[-5.583584,3.902200,-6.915655,-8.197415,1.863847,9.185879,0.927976,-2.876206,-8.595750,-4.362133,-3.205862],[-1.829329,0.593602,4.480360,7.258576,-7.606425,-4.959276,-1.497762,-1.997841,-3.232590,-0.387113,-2.395658],[-1.764597,-1.048242,8.603103,5.908222,-2.254745,-1.352819,-6.968331,-0.989619,-9.836561,-3.118549,-6.040514],[4.257721,3.434374,-4.895543,6.534827,6.679031,-5.722497,2.121839,-8.738624,4.053919,-2.093839,0.977634]],[[2.008445,7.140998,-3.747541,-6.365065,-5.940605,-5.022807,-1.723935,9.201757,2.246905,8.553886,-2.538041],[0.701277,2.038601,4.218575,-0.855758,-6.913291,-0.286363,-2.038651,6.424855,7.074805,-8.023232,1.184107],[7.116779,8.675341,7.727236,-2.595004,5.232098,7.831067,-2.071854,-1.446210,1.069369,-2.500276,-9.523002],[-1.090416,-9.773401,7.602720,8.649194,3.430686,-1.904141,9.356422,3.578751,3.205494,-0.235286,9.006296],[-9.267640,4.872371,-3.925618,8.858997,-8.171342,1.907111,-1.942966,-5.467476,8.212187,-8.014141,-3.523941]],[[-4.063596,4.840660,-7.951424,-8.372032,-2.658342,6.179532,-5.237680,7.748216,4.235098,0.800880,-2.515156],[8.918194,-6.658182,-5.151591,-4.203963,-8.440054,-7.555076,0.986353,2.616097,-6.352620,-2.514477,-9.994059],[-0.835167,-1.870381,7.160448,-7.104543,0.683012,-2.835454,-9.458853,-6.395329,1.082368,7.149056,-1.794170],[-3.439702,-9.379676,5.287299,2.522008,6.652011,-8.442625,9.118704,-4.396262,5.389827,8.779242,6.284486],[2.332889,1.817961,-4.082036,-8.264155,7.954913,5.568924,-3.877668,-5.542571,-7.041558,8.233324,4.192546]],[[-1.929856,-9.418437,-3.609099,-3.538669,9.456957,9.022575,-8.268314,-9.919798,5.709915,-8.004210,2.632088],[7.671965,-2.183078,-9.149208,-0.897016,-7.369212,4.738188,9.058425,-1.851268,5.634521,8.913938,-4.219636],[-7.448270,-2.610140,-1.539043,-8.662063,7.219183,8.317886,0.005474,6.733907,-4.412940,5.360138,-5.216473],[0.896356,-7.036956,5.929783,2.060897,9.261890,-4.982322,4.259018,-9.587572,-1.615463,-3.011017,6.919053],[6.020175,3.199538,-0.811104,0.922725,3.585402,-8.084062,2.273182,4.312258,6.374265,5.695674,2.215171]],[[8.399306,-4.909290,9.556217,7.210608,3.835135,-6.220072,-1.312808,3.535954,7.543175,-7.290129,-5.230250],[0.817849,9.660842,-8.689525,-0.921357,1.626004,-2.456418,-7.436344,-9.225244,-3.046378,-8.591457,9.933023],[-4.002016,7.521616,4.605086,9.973385,3.598793,-6.281599,2.234522,0.117455,-5.305307,9.528462,5.704539],[-7.409955,-6.457642,0.238826,0.443727,5.282690,4.504784,-7.760856,5.582176,-3.168865,-5.627591,3.763205],[4.905403,-6.553694,1.609597,-8.237593,9.193253,0.050716,-3.356852,-0.873306,3.446049,8.638282,-9.817829]],[[9.195262,8.224564,3.914051,-1.570787,2.252333,2.804351,5.404444,-8.862872,4.173583,5.518541,3.570456],[3.759555,7.338568,9.698708,7.001565,-2.506072,1.676376,-3.840468,-2.358615,9.122254,7.249085,0.522356],[-0.485101,-2.169542,9.208243,-2.973327,3.341171,-1.677825,-3.562148,6.594602,-8.116663,-7.716246,7.697410],[-5.866544,-9.541976,-1.302104,-5.538390,-4.357411,5.562095,0.813226,-2.846783,0.399529,0.861108,3.639413],[7.030836,-9.792406,-8.864110,-4.555483,5.987364,0.833219,-6.878795,1.243249,4.063505,8.107637,5.969081]],[[3.871049,-4.545111,6.568851,5.186075,6.477166,-9.714862,-0.957393,8.149798,-9.953843,8.301423,-4.051602],[2.295262,-2.573677,7.167783,4.460127,-4.730065,2.733092,8.313757,8.507480,9.097730,-3.116244,-3.292329],[-6.874890,9.975559,-9.574988,6.453187,-6.313132,2.055363,0.626910,1.091602,1.861979,-0.043448,-7.327075],[3.520637,-5.383225,9.262487,-9.925812,6.521997,7.108978,-6.526727,5.439681,-2.292605,-7.018160,-9.588340],[-2.002113,8.632968,-9.478578,-4.453208,7.711006,-3.034950,-9.625971,-2.021145,0.814380,-8.279171,-9.380320]],[[-8.091942,-8.879028,-0.660989,-0.068036,-3.458032,-4.057200,-1.879989,1.973321,8.635601,4.298024,-9.212995],[7.413000,-9.559217,-1.462576,0.955181,3.249347,-0.406665,-6.124376,6.462027,-7.852885,-9.055384,9.720107],[2.819025,9.478796,7.044279,-8.098925,4.223017,4.864627,-6.157061,-5.114770,-0.852181,4.244674,0.178900],[8.369599,6.361936,-0.459113,-8.395566,-6.738791,0.354558,2.216688,-8.966181,-3.633190,2.506746,-9.562180],[-0.325591,-2.299638,-5.255808,9.727808,0.985662,-4.500686,-8.547005,-8.217218,-4.413024,-5.084585,6.321919]],[[5.711903,8.448769,8.807326,0.028297,-8.220416,-1.658809,-1.890861,9.518378,-3.153860,7.801210,8.662615],[5.937359,-1.269048,-5.317787,1.183314,-3.358248,-8.784011,0.115909,8.412843,4.480826,-1.006234,-5.935253],[-4.330425,-1.465437,-7.284068,-4.884392,-3.152395,-6.626151,6.915242,3.792468,-3.618223,3.000071,-1.796193],[-4.532919,7.760446,5.675041,6.216496,3.612933,5.631538,-0.257800,8.385459,1.843748,-2.887012,-0.645370],[8.509446,6.315819,-6.910494,-4.260648,0.063202,-5.529548,-7.544218,-0.201918,9.716640,8.200585,2.373053]],[[-4.340228,-1.176358,-3.481999,7.127943,6.246010,4.403530,-3.745219,9.320069,-2.860542,9.639844,-3.294708],[-8.235618,8.931390,-7.474455,0.071604,-9.743975,-4.911296,4.668101,-8.595491,1.760575,3.424242,-4.418042],[-7.052598,-7.792772,-9.378354,7.892892,3.210238,4.800304,5.597607,-8.399207,2.647336,5.614133,4.013948],[1.099858,1.926126,9.216778,6.074584,1.984882,-3.489348,-8.082535,0.776342,3.198478,8.850577,-4.670788],[-3.727681,1.880207,1.900573,9.636641,-3.617088,-0.979731,0.798255,-5.280531,1.298343,8.402927,-2.046502]],[[-9.997018,4.989669,-0.386568,-2.664127,-9.776140,4.601115,3.809043,-8.796500,-7.761208,7.868599,-0.584706],[-3.454819,8.816800,-6.647424,5.906042,-9.714416,9.442806,6.087167,-4.551293,-2.274600,-9.979863,4.073935],[-9.620061,-0.151927,-7.510036,-6.737487,2.391565,8.009784,-1.574198,9.496530,-8.810135,-5.209134,5.154614],[-1.545655,3.977018,-4.946535,-5.902939,8.629372,0.969617,8.686742,6.115130,0.985554,7.245861,6.679426],[-2.861008,2.074584,-2.357952,-7.352073,-9.963473,-2.396027,6.089997,3.201561,-7.649281,5.515995,0.608435]],[[2.981294,-3.454841,2.582601,4.678458,3.482134,2.295718,6.303334,-4.094317,1.819288,-5.240276,2.452175],[3.057221,3.421054,6.931130,4.892091,-4.766971,-8.350425,9.017210,-1.219865,-9.084671,7.585240,-1.537384],[-0.526935,-3.019414,2.463529,9.420844,2.709739,-3.200499,-2.442956,7.222982,-4.156221,-6.559952,-8.636583],[-5.118501,-7.595062,-5.689055,-4.741209,-9.023957,-8.989689,-1.169699,8.178621,-5.894248,-5.240635,-1.506651],[-2.939271,1.629354,1.155328,4.687305,4.362095,-5.296235,3.045281,9.441895,0.122700,-6.956789,4.586362]],[[-2.903427,-1.117075,9.483831,6.359605,7.376968,7.989363,-0.059811,-2.295375,-9.459211,2.173840,3.659886],[8.494696,-7.869389,9.583138,-1.966001,0.152210,0.576749,2.731733,0.248265,-1.537309,7.201094,1.182734],[-8.748388,6.087791,3.827979,0.735670,6.199545,-0.857869,-4.270432,-8.947876,-1.055252,1.988011,0.682543],[-9.887845,1.417797,7.543681,-2.031950,-6.798176,-4.427411,5.145202,-2.872007,4.138219,7.080741,2.631419],[6.009069,3.355906,7.650424,4.330948,9.872845,9.337775,7.451498,4.160803,8.068264,-8.340479,-1.876933]],[[-8.887146,0.070277,-2.414165,2.329291,-0.799452,-7.409185,-8.156103,-2.061444,-8.545968,9.187876,-1.989093],[3.121345,-2.001586,0.211964,-6.812794,-2.398968,-0.406289,-4.450882,-3.694073,-1.431922,-3.668307,-7.661872],[-4.838976,7.632240,4.867569,8.278697,8.411928,7.058762,6.044991,-5.722676,-1.153299,-5.529402,-1.271110],[-1.318146,-3.816502,1.355997,7.200064,2.495281,-4.014803,-7.318927,8.609047,6.164600,-5.845322,-9.337151],[4.953639,-2.132141,7.447024,-4.386253,4.036792,3.556472,3.252988,9.123235,-0.004765,3.549491,5.122574]],[[-6.234033,1.821337,1.044504,0.269862,2.050365,-0.977332,-9.919035,3.408181,-8.782143,0.974557,-6.066325],[-6.010197,2.501316,4.255601,-7.265832,2.323636,6.032931,-1.076285,-9.322592,8.930955,5.434479,5.280048],[8.375172,-0.762746,-0.856425,7.032738,5.316453,-2.490266,6.773933,5.008414,-8.307667,-5.213796,4.076836],[-2.268938,-1.635575,4.094900,9.546292,-0.529792,3.239633,-6.434458,-7.041093,1.275974,-9.959369,4.513640],[0.755539,9.185314,4.763590,5.145186,-6.727016,-4.919449,9.057588,-2.421550,2.751093,-1.421210,-9.860549]]], dtype = "float32")#candidate|3296|(15, 5, 11)|const|float32
var_3297 = relay.var("var_3297", dtype = "float32", shape = (15, 5, 11))#candidate|3297|(15, 5, 11)|var|float32
bop_3298 = relay.floor_divide(const_3296.astype('float32'), relay.reshape(var_3297.astype('float32'), relay.shape_of(const_3296))) # shape=(15, 5, 11)
bop_3302 = relay.add(const_3296.astype('uint8'), relay.reshape(var_3297.astype('uint8'), relay.shape_of(const_3296))) # shape=(15, 5, 11)
func_2704_call = mod.get_global_var('func_2704')
func_2705_call = mutated_mod.get_global_var('func_2705')
call_3310 = relay.TupleGetItem(func_2704_call(), 0)
call_3311 = relay.TupleGetItem(func_2705_call(), 0)
func_1653_call = mod.get_global_var('func_1653')
func_1659_call = mutated_mod.get_global_var('func_1659')
const_3314 = relay.const(True, dtype = "bool")#candidate|3314|()|const|bool
var_3315 = relay.var("var_3315", dtype = "bool", shape = (13, 11))#candidate|3315|(13, 11)|var|bool
var_3316 = relay.var("var_3316", dtype = "bool", shape = (1573,))#candidate|3316|(1573,)|var|bool
var_3317 = relay.var("var_3317", dtype = "float32", shape = (22, 78))#candidate|3317|(22, 78)|var|float32
call_3313 = relay.TupleGetItem(func_1653_call(relay.reshape(const_3314.astype('bool'), []), relay.reshape(var_3315.astype('bool'), [11, 1, 13]), relay.reshape(var_3316.astype('bool'), [11, 11, 13]), relay.reshape(var_3317.astype('float32'), [11, 12, 13]), ), 0)
call_3318 = relay.TupleGetItem(func_1659_call(relay.reshape(const_3314.astype('bool'), []), relay.reshape(var_3315.astype('bool'), [11, 1, 13]), relay.reshape(var_3316.astype('bool'), [11, 11, 13]), relay.reshape(var_3317.astype('float32'), [11, 12, 13]), ), 0)
output = relay.Tuple([bop_3298,bop_3302,call_3310,call_3313,const_3314,var_3315,var_3316,var_3317,])
output2 = relay.Tuple([bop_3298,bop_3302,call_3311,call_3318,const_3314,var_3315,var_3316,var_3317,])
func_3320 = relay.Function([var_3297,var_3315,var_3316,var_3317,], output)
mod['func_3320'] = func_3320
mod = relay.transform.InferType()(mod)
var_3321 = relay.var("var_3321", dtype = "float32", shape = (15, 5, 11))#candidate|3321|(15, 5, 11)|var|float32
var_3322 = relay.var("var_3322", dtype = "bool", shape = (13, 11))#candidate|3322|(13, 11)|var|bool
var_3323 = relay.var("var_3323", dtype = "bool", shape = (1573,))#candidate|3323|(1573,)|var|bool
var_3324 = relay.var("var_3324", dtype = "float32", shape = (22, 78))#candidate|3324|(22, 78)|var|float32
output = func_3320(var_3321,var_3322,var_3323,var_3324,)
func_3325 = relay.Function([var_3321,var_3322,var_3323,var_3324,], output)
mutated_mod['func_3325'] = func_3325
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3245_call = mod.get_global_var('func_3245')
func_3247_call = mutated_mod.get_global_var('func_3247')
call_3329 = relay.TupleGetItem(func_3245_call(), 0)
call_3330 = relay.TupleGetItem(func_3247_call(), 0)
output = relay.Tuple([call_3329,])
output2 = relay.Tuple([call_3330,])
func_3353 = relay.Function([], output)
mod['func_3353'] = func_3353
mod = relay.transform.InferType()(mod)
mutated_mod['func_3353'] = func_3353
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3353_call = mutated_mod.get_global_var('func_3353')
call_3354 = func_3353_call()
output = call_3354
func_3355 = relay.Function([], output)
mutated_mod['func_3355'] = func_3355
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2047_call = mod.get_global_var('func_2047')
func_2048_call = mutated_mod.get_global_var('func_2048')
call_3407 = relay.TupleGetItem(func_2047_call(), 0)
call_3408 = relay.TupleGetItem(func_2048_call(), 0)
var_3413 = relay.var("var_3413", dtype = "int8", shape = (3, 14, 9))#candidate|3413|(3, 14, 9)|var|int8
bop_3414 = relay.add(call_3407.astype('uint32'), relay.reshape(var_3413.astype('uint32'), relay.shape_of(call_3407))) # shape=(3, 14, 9)
bop_3417 = relay.add(call_3408.astype('uint32'), relay.reshape(var_3413.astype('uint32'), relay.shape_of(call_3408))) # shape=(3, 14, 9)
func_2605_call = mod.get_global_var('func_2605')
func_2607_call = mutated_mod.get_global_var('func_2607')
const_3422 = relay.const([-2.196445,3.365579,6.241670,-9.923281,-5.473911,6.442951], dtype = "float64")#candidate|3422|(6,)|const|float64
call_3421 = relay.TupleGetItem(func_2605_call(relay.reshape(const_3422.astype('float64'), [6,])), 1)
call_3423 = relay.TupleGetItem(func_2607_call(relay.reshape(const_3422.astype('float64'), [6,])), 1)
func_2212_call = mod.get_global_var('func_2212')
func_2216_call = mutated_mod.get_global_var('func_2216')
const_3425 = relay.const([-6,4,10,-8,6,10,5,-3,-3,-1,4,1,6,-4,3,-2,4,10,5,-4,3,-8,-7,9,1,-5,5,8,-3,6,2,2,-3,9,-10,10,8,4,-10,-5,3,8,9,1,-10,5,6,-7,10,8,10,-5,-9,-6,-2,-10,3,4,3,2,-4,-3,-4,1,8,-6,-1,-7,-2,7,-3,8,4,6,7,1,6,-4,-6,-6,8,3,5,-10,-4,-2,6,3,-1,-9,8,2,4,4,-2,-4], dtype = "uint32")#candidate|3425|(96,)|const|uint32
call_3424 = relay.TupleGetItem(func_2212_call(relay.reshape(const_3425.astype('uint32'), [2, 8, 6]), relay.reshape(const_3425.astype('uint32'), [2, 8, 6]), ), 0)
call_3426 = relay.TupleGetItem(func_2216_call(relay.reshape(const_3425.astype('uint32'), [2, 8, 6]), relay.reshape(const_3425.astype('uint32'), [2, 8, 6]), ), 0)
output = relay.Tuple([bop_3414,call_3421,const_3422,call_3424,const_3425,])
output2 = relay.Tuple([bop_3417,call_3423,const_3422,call_3426,const_3425,])
func_3433 = relay.Function([var_3413,], output)
mod['func_3433'] = func_3433
mod = relay.transform.InferType()(mod)
mutated_mod['func_3433'] = func_3433
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3434 = relay.var("var_3434", dtype = "int8", shape = (3, 14, 9))#candidate|3434|(3, 14, 9)|var|int8
func_3433_call = mutated_mod.get_global_var('func_3433')
call_3435 = func_3433_call(var_3434)
output = call_3435
func_3436 = relay.Function([var_3434], output)
mutated_mod['func_3436'] = func_3436
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2704_call = mod.get_global_var('func_2704')
func_2705_call = mutated_mod.get_global_var('func_2705')
call_3443 = relay.TupleGetItem(func_2704_call(), 0)
call_3444 = relay.TupleGetItem(func_2705_call(), 0)
output = call_3443
output2 = call_3444
func_3454 = relay.Function([], output)
mod['func_3454'] = func_3454
mod = relay.transform.InferType()(mod)
mutated_mod['func_3454'] = func_3454
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3454_call = mutated_mod.get_global_var('func_3454')
call_3455 = func_3454_call()
output = call_3455
func_3456 = relay.Function([], output)
mutated_mod['func_3456'] = func_3456
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3457 = relay.var("var_3457", dtype = "float32", shape = (15, 9, 11))#candidate|3457|(15, 9, 11)|var|float32
uop_3458 = relay.tan(var_3457.astype('float32')) # shape=(15, 9, 11)
output = uop_3458
output2 = uop_3458
F = relay.Function([var_3457,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_3457,], output2)
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
input_3457= np.array([[[-3.729464,-7.348530,4.937984,-1.428222,0.708231,-7.246475,5.136957,-5.885883,-5.194609,-1.802866,7.644468],[5.750598,4.715892,9.524629,9.565713,-9.176631,-9.149264,-0.971777,9.739029,-4.907766,5.868443,-0.864876],[3.556600,-9.383836,-4.313872,3.544923,-8.620099,5.524986,-9.907074,6.731708,0.188850,2.886372,-1.875978],[3.365225,-7.506358,-5.248381,-9.068275,1.456979,-0.023941,2.496501,-2.568447,-2.115125,-3.144229,-6.146747],[-8.819587,-2.306374,3.055826,5.142246,-6.823655,9.026767,-9.256956,-5.055030,8.547254,-2.831841,1.861746],[-0.678358,6.548813,8.258336,-1.524557,-0.839238,5.234582,-0.845323,-7.461921,-5.408287,-3.883063,4.327589],[7.047444,-6.531649,8.987309,-8.399759,-5.491134,0.933054,0.212970,1.256284,-9.408910,0.988878,7.262020],[9.937137,2.460656,-1.423077,8.028476,9.613493,1.173147,0.903619,-6.605085,2.923451,4.573967,-2.005508],[3.202589,-9.685165,8.603638,1.881685,-0.689412,-6.603540,-8.839695,2.405171,8.457023,2.712058,-5.110948]],[[3.565153,-4.033811,-6.079644,2.768352,5.135710,-4.185116,5.953901,2.162913,-6.507064,8.670767,3.802674],[9.230970,9.199237,9.344916,7.822093,-2.982794,3.336795,-3.579381,-8.881188,-8.774568,-8.921594,9.416784],[3.646657,4.882265,-9.599283,-3.586605,-2.405468,2.958840,-8.794650,-4.572674,-2.894024,6.322474,-6.791303],[1.730022,3.302267,-6.612861,1.670814,4.656710,-3.727269,-1.064830,7.969264,-2.941511,0.991910,-9.179030],[5.283811,4.600398,5.517000,2.574519,-5.915200,-1.749203,-3.414725,-8.001715,-4.486252,-4.285954,-6.802044],[-8.487537,-4.093412,9.686875,-7.196476,5.642246,5.022372,8.978890,1.475704,-4.703991,-2.861569,-7.389018],[4.470136,-1.161794,-8.157264,-0.871895,6.429597,4.828364,1.797693,1.657240,-6.031703,7.806088,-2.510406],[-2.765176,2.641125,2.351390,9.757249,9.722847,-3.217182,-8.613029,-3.623247,9.299948,4.137589,2.023891],[-2.087294,-9.183130,-6.368347,-2.746197,0.146747,-9.166366,4.350939,-9.377400,2.078553,0.227282,3.948523]],[[4.504476,-3.784060,3.083665,6.928791,-5.596494,-0.991097,-4.933471,7.633032,4.160033,4.697205,6.367965],[2.662996,0.572669,1.996708,4.594959,-3.757272,5.986552,4.501688,2.845586,4.371742,-5.136261,-8.526602],[-7.118015,5.136812,-3.006551,-8.060746,7.533320,0.428576,-1.578334,0.017121,4.349499,-6.098838,-3.965283],[-7.944736,-7.922561,-4.602274,8.155567,9.612846,-2.399589,0.452872,5.601367,-5.269669,8.879959,2.927331],[5.714001,2.117751,8.764010,4.979972,-8.037614,2.490949,9.392773,7.737521,-5.289742,3.980173,-8.249223],[-4.997153,-5.206679,-0.159762,-1.078600,2.420538,2.067234,2.988908,7.375249,-5.248519,-5.072377,-7.148559],[8.055768,-9.998606,1.162444,-3.050339,-8.642142,5.873978,-2.512736,-0.519661,-6.508617,-6.953712,9.418901],[-5.296202,-9.572893,-3.007526,-7.525686,-2.068192,-7.563706,0.666210,-7.729558,7.698375,-8.563620,2.074050],[5.547930,-4.443352,-3.281363,7.629877,-7.327102,-3.051968,-5.360161,4.363919,7.732402,8.446799,-0.685984]],[[6.839345,-9.500248,8.284594,-8.289614,2.491472,-8.076131,6.863214,-6.046739,-8.853467,-2.408737,2.139445],[5.055653,8.998748,9.537838,0.052449,5.780924,-7.623647,5.509030,7.692207,5.695454,5.992994,-9.874241],[-1.972488,8.012184,0.114046,8.855954,5.464952,0.125991,8.738331,9.176393,-1.839560,-1.315426,-9.415660],[-3.263133,6.239454,4.629233,6.291466,3.067637,-7.141276,8.106180,-1.329076,2.403618,0.172494,-7.103940],[2.448535,2.537538,-8.752496,-3.706945,-6.592942,-6.228781,-0.939309,-0.525905,2.218552,2.052926,2.738216],[4.098976,0.629189,7.823344,0.642335,-0.309559,0.508134,-2.306969,1.817773,-6.552912,0.999846,0.417752],[-4.791340,-8.382747,3.281925,-8.342448,-1.199300,-7.395307,-9.325488,-8.546419,4.012991,-1.277953,8.240642],[-1.230268,3.755274,-2.020545,-6.329494,2.834737,7.556953,-4.638610,1.623968,6.850893,1.118359,-2.824378],[3.730931,7.288310,1.546749,1.136251,4.501780,0.937615,5.981602,8.458863,3.169696,-9.057623,-4.831871]],[[0.009189,2.912097,2.689304,-2.600262,-3.427171,8.406253,-3.632566,4.331959,2.409526,4.582117,-6.800164],[-3.597289,-3.708926,-5.894553,-1.874498,-0.523580,-7.247040,3.952461,-1.458993,8.431355,3.356486,-9.053773],[3.262907,-4.225507,-2.071825,7.020304,-5.502012,7.900525,5.773461,0.306420,-8.418094,1.242695,-0.565281],[9.729902,-7.305403,9.142253,0.875320,9.408040,-6.263877,5.547321,-9.539193,7.266445,-7.073051,6.914535],[-2.119669,5.979275,-1.492585,4.631058,1.106199,2.284897,-2.665444,-8.427069,8.414453,-6.058637,9.234148],[-0.012363,7.702144,-5.292149,-1.749325,8.824244,-8.651840,-7.115623,-2.570679,7.123660,-6.317341,1.494063],[-0.326073,-8.398418,-0.130501,6.245174,-2.626444,0.933895,9.036192,3.657124,6.346080,-1.577042,4.590193],[0.851451,-0.364305,-1.067617,4.428395,6.736265,0.749392,7.305383,3.228210,6.948760,5.206926,2.817517],[-3.891354,-2.238685,8.433113,9.168236,0.572506,-1.207964,-5.613565,-9.568147,2.877412,0.139383,4.915473]],[[-7.751300,0.603388,-3.917752,-5.324743,0.697741,-8.948762,-1.020936,-0.548050,5.396833,-8.328597,1.283272],[-5.404219,5.801770,-1.349687,-9.140085,0.631609,-8.904093,-7.018727,9.650521,-5.290185,4.913488,9.471207],[-9.819803,7.303786,1.350879,-8.572683,-0.240773,3.309601,-2.831637,-9.184673,8.709312,6.317050,-5.013050],[1.227259,-3.618462,-7.477636,-0.812432,8.216802,5.605900,-0.229744,9.471246,-4.840967,1.193217,7.828254],[6.100217,-6.908066,7.836805,-9.010214,4.673726,-7.738611,8.668260,7.285753,-7.474155,-6.363937,-4.054884],[-4.049438,5.761869,1.251835,-9.233343,-4.106620,4.287378,3.922598,-6.911435,-3.586132,8.933610,6.977412],[-8.094542,-5.899183,-8.702157,-0.168523,-5.847427,-9.185412,4.129693,-4.970947,-6.257677,4.407148,-7.818499],[-5.824272,-4.737903,-0.674791,8.681769,-7.480683,9.138398,-4.219652,-0.669838,-4.461960,-3.045636,-3.691263],[-1.223611,8.055940,-6.572126,-1.385093,5.020158,4.875307,-1.057088,3.822038,-2.192231,1.991065,3.026671]],[[5.992151,-7.280023,-1.026041,-1.242469,6.632237,4.090730,9.962576,1.602059,-0.561447,-1.062990,3.219834],[-0.534449,1.930204,-9.453898,-1.887500,-0.608753,0.196231,6.692118,9.055192,8.428134,-2.631826,-8.217013],[-2.107921,-1.323775,-4.162516,-8.048201,6.969713,2.868539,8.743482,6.370992,5.757091,1.950477,-3.253525],[-0.143132,-0.557837,-4.926853,-9.391469,-0.921562,5.137098,-9.241770,-5.133466,6.415284,9.644804,-3.365501],[-9.141032,-4.492430,2.060745,3.855157,-5.512509,-4.264762,3.846568,3.707298,0.566507,-1.878873,-9.220526],[-0.615679,6.572885,-3.425113,-4.682310,-5.310283,-5.708362,-6.232388,-9.014723,-3.107443,4.094745,-0.933686],[4.639583,4.812941,8.755591,0.207609,-2.478523,-9.849692,9.818310,-1.137632,1.949902,2.049860,3.901165],[-5.044783,1.089069,-3.660193,1.103909,5.786461,-6.691584,4.617972,-0.371390,-2.043436,-4.494600,-3.153132],[0.600969,-4.931098,0.812691,1.811297,8.155568,7.063141,-8.038632,-1.039613,-4.236440,-2.436382,-4.549602]],[[-0.915121,3.818035,7.988782,-6.273687,0.975740,-1.762073,-1.974554,7.127930,-6.165914,0.738332,1.774066],[9.868876,6.990960,0.370845,-2.929293,8.095238,-9.776238,4.919370,6.096169,-2.305201,-9.131458,-0.429921],[-0.082183,-9.813960,-9.695736,-1.832794,1.939307,9.736034,4.901459,-6.691896,-5.686604,4.732475,-7.355289],[-8.295098,-3.313562,9.435870,8.389069,5.112847,4.296797,-7.010748,-9.259752,-2.431974,-2.423279,-6.559911],[-0.228519,-6.627165,6.943394,-4.256716,-4.086102,0.197166,1.017953,2.459877,-7.299311,-3.983269,-2.939954],[8.324730,-6.941222,-7.865565,-9.670552,-4.456692,-4.017340,1.148183,-6.746885,4.793461,2.637529,7.390213],[8.554909,4.156538,-9.738271,-7.140886,9.661589,4.604571,8.897498,-4.175640,-5.106626,-1.969921,4.171859],[-0.163246,0.728451,8.631397,-7.006484,2.461040,0.849747,0.961207,8.197133,5.496329,-2.070465,-5.480558],[-2.341065,5.609893,9.713639,8.631672,2.995050,4.845627,-9.623539,8.196714,5.719957,-0.650828,-0.383413]],[[-5.307998,8.760720,-8.218581,-9.145698,4.088431,0.417533,-9.970107,1.218826,-7.026205,3.114326,1.082236],[-2.810442,8.085343,-5.246823,-7.196220,4.831249,-2.342253,-0.952714,-8.543432,0.776539,-2.892442,-9.389809],[5.670965,-6.344805,3.291838,2.928286,0.188382,9.078782,-4.669671,4.653854,-7.084350,5.714933,0.499489],[8.999915,1.251513,-2.266647,-4.176038,-7.996403,-5.602952,8.243441,-7.015464,-9.291892,-2.632742,-5.587314],[2.819894,6.705255,4.995787,-4.186549,7.847873,-3.595443,-6.157282,-3.772447,-5.446777,7.178238,4.547706],[-9.430214,4.229093,-8.074064,-3.485076,-3.275142,-4.309738,-7.606207,9.973311,-7.804313,-1.301250,5.046737],[9.801683,1.050038,-7.551387,-4.063601,0.508296,5.265331,-0.303599,-8.262660,8.247368,-1.686182,7.475265],[8.449523,-8.671463,3.333650,-1.426724,2.810201,-6.120558,8.367904,9.792949,-2.180854,-5.145765,-5.446684],[0.594365,-7.998971,9.634095,-3.761593,3.309029,1.271832,-3.513483,-8.650510,6.665595,-8.016377,3.153330]],[[-7.030303,-6.701899,2.899167,3.862745,5.105795,3.642204,9.498634,-1.883638,7.124359,5.709655,6.589581],[0.370340,-4.308313,-6.262168,-9.024541,4.868275,-1.203406,0.215557,-3.047030,4.206112,-7.311652,5.053302],[-3.179057,3.267549,5.940515,-8.282093,-3.160142,5.482734,7.883901,-7.668950,9.955057,-7.720415,9.974400],[7.385103,-3.097702,7.456491,5.443642,-3.258618,-6.019026,0.727658,-5.757518,7.821698,3.946287,-9.005062],[-1.043661,-3.853575,-6.645649,2.791019,3.162153,3.569415,-1.502559,-4.292316,-8.888802,1.422696,-5.526945],[-3.792324,-4.516836,-6.828032,-4.759587,-7.623428,5.873276,-6.330006,-4.141385,4.259540,6.790862,0.159792],[9.035216,7.243438,2.751166,-1.530805,-1.897836,-7.028009,-0.350627,-3.135430,2.701019,6.206186,-2.121004],[-9.658094,-4.432738,-1.605560,7.888355,7.593588,6.874665,-5.593110,6.941872,-8.388252,5.073677,-2.995112],[4.393895,8.832937,-9.937007,0.998554,-8.042497,0.830633,7.793705,-6.273550,8.398525,8.816918,0.854945]],[[4.012927,8.155017,-4.605640,6.256171,-7.303307,4.257087,9.903070,0.812587,-0.405267,8.063308,-5.794854],[5.448076,0.930353,-7.567442,9.738485,-5.581078,-9.500228,3.857770,-0.835025,7.405106,0.154708,-3.366034],[-3.961844,-4.279902,-3.227686,-6.686265,-1.204755,4.645254,6.302238,0.571312,4.450157,1.162996,7.900246],[-7.523145,4.840455,-6.018001,9.805149,-8.651621,-4.716537,-1.327027,-3.849770,2.642225,6.602531,-6.501369],[-3.187300,8.408372,6.791344,8.449628,9.816900,-5.970621,-3.641787,-6.305619,-5.266343,4.010076,6.487984],[7.272389,4.016403,-8.122201,9.268427,5.989399,-2.855818,-8.432772,4.945771,-9.903338,2.721538,-6.827951],[-8.110886,7.643673,0.457153,0.167878,5.889829,-3.048021,-7.154694,2.188258,9.143053,7.380237,2.903349],[7.531073,-3.094975,-6.158647,-0.975175,2.190029,6.995875,8.905184,5.022046,3.776284,9.680531,0.635951],[-0.685157,-0.185792,5.566917,-5.803168,0.739352,-5.470355,0.437198,3.742918,7.845811,-8.776850,-1.304416]],[[1.498786,9.559674,-9.742048,-1.053377,-6.119979,-7.313748,-8.318608,-2.188755,-3.395546,-9.986381,3.601246],[8.186089,2.166277,2.671083,7.603148,-8.742832,-7.359814,4.381843,-8.815691,-6.686250,7.737429,9.690486],[1.852002,-6.280756,5.465005,-5.075440,-4.137899,-0.089790,-6.728996,-4.402200,0.912187,-4.216345,-2.828589],[-8.475479,4.558779,5.738187,9.241459,-9.984838,-0.878127,-9.438099,9.202663,9.387340,1.133357,8.304868],[3.771588,2.291650,-8.682008,-1.711290,-6.875339,-6.655132,-7.301965,-0.999603,-0.710037,-8.887499,0.472362],[6.317880,9.201421,-7.257030,-1.268727,-4.043807,2.583724,-0.559268,2.890760,-8.361307,0.549021,2.605064],[-8.948670,7.540306,-0.825226,-8.498891,-8.419531,-1.980781,-5.348251,-4.741162,-6.599073,-6.275664,2.734589],[-2.160324,-6.585112,-4.776157,4.091495,8.144805,6.410513,-1.716849,-2.292363,5.229603,-8.841648,-1.291001],[-3.582695,8.261511,0.127551,2.248075,-0.920587,3.488285,-1.354848,-1.255120,7.686604,-5.099032,-7.787013]],[[-1.756385,6.617060,-4.931107,8.848671,-3.727291,8.879789,-5.132962,7.127427,0.084541,1.133174,-9.470898],[-7.965969,5.853479,-4.424070,-9.400904,0.115132,7.455300,-3.176332,-8.721354,-7.812760,-9.264678,7.804964],[-7.456147,-3.242049,-1.439497,-9.248006,8.452551,-9.124581,-5.310567,-8.340261,5.406416,1.126934,-7.892557],[-5.606661,1.732380,7.331213,9.787286,0.158544,-9.476664,0.268703,9.731027,9.350978,2.956550,-8.663189],[6.344759,-8.749669,-9.524832,2.483868,-7.757763,0.964048,9.119658,7.028951,4.832992,-2.018795,-9.295704],[8.557599,-0.816153,-1.931991,-4.549043,8.462488,-2.242020,-6.383126,-1.375200,7.607769,-9.601641,-9.983324],[-7.620248,-4.643325,-2.833812,3.683136,-1.442887,0.647472,3.586459,-5.686763,-9.644121,5.670790,-4.486691],[2.226806,-0.665193,8.038814,6.510483,-1.626426,6.466762,-9.519774,7.510690,-3.697245,-2.466415,-1.410162],[2.453951,5.514183,2.897061,-0.945814,-4.048819,-3.619753,3.609990,0.899957,0.283679,-9.827836,-6.278137]],[[0.059802,-8.996670,-5.211935,5.065067,-3.062989,8.796755,-0.418074,9.224892,-1.667936,-2.300585,-9.140098],[-4.064127,-1.887725,7.848895,-1.503605,4.048795,7.256490,9.596089,-6.854711,3.051887,-0.260919,-2.674457],[2.609096,0.371297,-6.715388,-0.864944,6.458738,-8.455935,-6.792575,-0.350129,-4.951148,-4.073253,0.608671],[0.988813,-5.052008,8.868604,6.765457,3.529876,4.611655,7.362182,-9.946233,-5.313006,-0.323538,-2.572020],[-8.503093,-7.930744,-3.894845,-1.210660,-1.995793,-6.193080,-8.607852,1.084345,4.629877,-8.704087,-2.452099],[-0.629969,6.575223,-3.296260,-8.068304,3.230029,-8.688740,9.524853,-7.914361,-0.433783,9.149004,-7.554632],[-1.732226,6.708421,-6.209193,-4.900080,9.156757,-8.170088,9.677048,8.782486,6.514599,-7.003877,-4.011196],[-5.931441,-2.345466,2.231373,3.904363,1.654009,-2.731938,-0.935304,5.309945,3.734648,1.252958,9.708332],[-9.671257,-2.987762,7.246469,6.142155,-7.920492,7.578593,6.931661,-1.552929,-8.380164,-3.853950,5.817454]],[[5.275124,-3.236858,1.052552,-3.341025,-4.149250,0.186838,-9.678007,-8.418021,-8.145562,-5.445285,9.410730],[0.319696,-1.404590,4.715650,-9.562441,8.824757,1.743254,-6.020441,-4.942586,-3.307135,0.979605,9.505072],[-7.482951,-3.953925,0.982597,-6.866770,-1.480494,-3.186554,3.592953,1.552087,-1.088665,-2.104835,-6.375660],[3.336472,4.874726,-4.660469,3.217548,9.390915,2.220584,1.694082,-7.547995,9.357243,0.221090,8.732823],[2.418380,-0.852243,-6.463968,-1.758760,1.160333,4.563511,7.426680,8.590052,9.430628,-2.598536,-8.813667],[2.206677,-0.134379,1.599070,7.790901,-8.057294,-3.115751,-3.212728,5.210443,3.384243,-3.555403,-3.273691],[-1.066423,5.899375,-9.445265,-5.976964,2.414656,2.417912,-2.299602,6.039752,9.650383,1.467334,-9.130276],[-2.583182,-6.626881,-2.065295,-3.883627,-0.644944,-9.370491,6.527161,5.671192,-2.461989,1.357757,-9.903627],[-6.406936,8.767121,-7.294095,4.201149,8.455942,-0.620993,-8.464034,-4.727767,6.003241,0.318016,-7.303835]]], dtype='float32')
module1.set_input('var_3457', input_3457)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_3457, )
res3 = intrp3.evaluate()(input_3457, )
res4 = intrp4.evaluate()(input_3457, )
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
module5.set_input('var_3457', input_3457)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_3457, )
res7 = intrp7.evaluate()(input_3457, )
res8 = intrp8.evaluate()(input_3457, )
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
module9.set_input('var_3457', input_3457)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_3457, )
res11 = intrp11.evaluate()(input_3457, )
res12 = intrp12.evaluate()(input_3457, )
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
module13.set_input('var_3457', input_3457)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_3457, )
res15 = intrp15.evaluate()(input_3457, )
res16 = intrp16.evaluate()(input_3457, )
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
module17.set_input('var_3457', input_3457)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_3457, )
res19 = intrp19.evaluate()(input_3457, )
res20 = intrp20.evaluate()(input_3457, )
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
module21.set_input('var_3457', input_3457)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_3457, )
res23 = intrp23.evaluate()(input_3457, )
res24 = intrp24.evaluate()(input_3457, )
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

'''43: TVMFuncCall
42: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
41: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
40: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
39: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
38: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
37: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
36: tvm::transform::Pass::operator()(tvm::IRModule) const
35: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
34: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
31: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
30: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
29: tvm::transform::Pass::operator()(tvm::IRModule) const
28: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
27: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
26: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
25: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
22: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
20: _ZN3tvm5relay9transform22Devic
19: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
18: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
17: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
16: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
15: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::TupleNode const*)
14: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
11: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
8: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
7: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
6: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
5: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
3: tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String)
2: tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::function<tvm::runtime::String (tvm::runtime::String)>)
1: tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
0: tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)

'''