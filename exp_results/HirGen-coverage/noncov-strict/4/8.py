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
var_0 = relay.var("var_0", dtype = "int8", shape = (2, 7))#candidate|0|(2, 7)|var|int8
var_1 = relay.var("var_1", dtype = "int8", shape = (2, 7))#candidate|1|(2, 7)|var|int8
bop_2 = relay.bitwise_xor(var_0.astype('int8'), relay.reshape(var_1.astype('int8'), relay.shape_of(var_0))) # shape=(2, 7)
var_5 = relay.var("var_5", dtype = "int8", shape = (2, 7))#candidate|5|(2, 7)|var|int8
bop_6 = relay.greater_equal(var_0.astype('bool'), relay.reshape(var_5.astype('bool'), relay.shape_of(var_0))) # shape=(2, 7)
uop_9 = relay.cos(var_1.astype('float64')) # shape=(2, 7)
var_11 = relay.var("var_11", dtype = "float64", shape = (2, 7))#candidate|11|(2, 7)|var|float64
bop_12 = relay.mod(uop_9.astype('float64'), relay.reshape(var_11.astype('float64'), relay.shape_of(uop_9))) # shape=(2, 7)
bop_15 = relay.bitwise_or(var_5.astype('int8'), relay.reshape(bop_2.astype('int8'), relay.shape_of(var_5))) # shape=(2, 7)
uop_18 = relay.acos(bop_6.astype('float64')) # shape=(2, 7)
uop_20 = relay.rsqrt(var_0.astype('float64')) # shape=(2, 7)
output = relay.Tuple([bop_12,bop_15,uop_18,uop_20,])
output2 = relay.Tuple([bop_12,bop_15,uop_18,uop_20,])
func_22 = relay.Function([var_0,var_1,var_5,var_11,], output)
mod['func_22'] = func_22
mod = relay.transform.InferType()(mod)
mutated_mod['func_22'] = func_22
mutated_mod = relay.transform.InferType()(mutated_mod)
func_22_call = mutated_mod.get_global_var('func_22')
var_24 = relay.var("var_24", dtype = "int8", shape = (2, 7))#candidate|24|(2, 7)|var|int8
var_25 = relay.var("var_25", dtype = "int8", shape = (2, 7))#candidate|25|(2, 7)|var|int8
var_26 = relay.var("var_26", dtype = "int8", shape = (2, 7))#candidate|26|(2, 7)|var|int8
var_27 = relay.var("var_27", dtype = "float64", shape = (2, 7))#candidate|27|(2, 7)|var|float64
call_23 = func_22_call(var_24,var_25,var_26,var_27,)
output = call_23
func_28 = relay.Function([var_24,var_25,var_26,var_27,], output)
mutated_mod['func_28'] = func_28
mutated_mod = relay.transform.InferType()(mutated_mod)
const_30 = relay.const([[[-8,-4,-3,-7,2,9,5,3,3,1]],[[-9,8,9,4,8,1,-6,10,-4,2]],[[-4,-5,7,6,-6,-8,5,-10,-6,8]],[[6,-1,-4,4,9,-3,5,3,-6,3]],[[-7,-6,7,-9,-10,-8,-6,-7,-3,8]]], dtype = "uint64")#candidate|30|(5, 1, 10)|const|uint64
const_31 = relay.const([[[-4,-9,-4,9,6,8,-10,-8,9,-4],[4,-7,2,-3,1,-2,-10,-5,6,2],[8,6,7,7,-3,3,2,-8,2,-5],[-1,-2,-7,-7,-9,-9,-1,6,-10,-3],[3,10,-10,10,-4,-6,1,7,-5,3],[3,2,6,3,-3,10,-4,3,-1,-7]],[[9,1,1,8,-2,1,-2,1,-5,3],[-1,-5,-3,1,3,-2,-1,-2,1,-10],[2,10,4,-9,8,-6,7,5,2,-8],[4,-9,1,7,5,4,-9,-9,-6,6],[4,-2,-5,-1,-3,-4,-8,-1,-8,8],[-8,-9,7,-3,-10,-8,1,5,6,-3]],[[-4,-7,-3,1,9,-3,4,-7,1,7],[-1,6,-2,9,7,-9,10,5,-4,-3],[8,1,9,4,-10,-3,-3,-2,-8,-6],[-3,-5,-1,-6,-7,-10,-6,-6,9,-7],[10,-9,-9,-1,-3,9,2,-9,-4,-9],[-8,5,-10,-1,10,9,9,4,-10,-4]],[[-2,9,9,-6,-4,4,-2,6,-10,2],[-8,-4,8,5,-3,-10,-4,-8,8,-5],[-6,6,10,9,-2,1,-1,-7,5,8],[-2,2,1,8,3,-4,4,7,6,3],[-9,8,-1,1,-8,5,-1,-6,8,-4],[9,8,-1,-5,-10,-3,-7,1,5,-6]],[[-4,2,8,10,4,-7,-3,6,9,-9],[4,-10,-4,-8,2,3,-5,-1,-5,-8],[-6,-3,9,3,3,-9,-5,-7,-7,-1],[9,6,10,-6,10,-5,-1,5,10,-10],[-9,1,5,4,-2,-2,6,-4,7,1],[-4,-4,-9,-6,7,5,4,-10,3,-4]]], dtype = "uint64")#candidate|31|(5, 6, 10)|const|uint64
bop_32 = relay.greater(const_30.astype('bool'), const_31.astype('bool')) # shape=(5, 6, 10)
bop_35 = relay.bitwise_or(bop_32.astype('int64'), const_30.astype('int64')) # shape=(5, 6, 10)
uop_38 = relay.log(const_31.astype('float32')) # shape=(5, 6, 10)
uop_40 = relay.acos(uop_38.astype('float64')) # shape=(5, 6, 10)
func_22_call = mod.get_global_var('func_22')
func_28_call = mutated_mod.get_global_var('func_28')
var_43 = relay.var("var_43", dtype = "int8", shape = (14,))#candidate|43|(14,)|var|int8
call_42 = relay.TupleGetItem(func_22_call(relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('float64'), [2, 7]), ), 2)
call_44 = relay.TupleGetItem(func_28_call(relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('int8'), [2, 7]), relay.reshape(var_43.astype('float64'), [2, 7]), ), 2)
output = relay.Tuple([bop_35,uop_40,call_42,var_43,])
output2 = relay.Tuple([bop_35,uop_40,call_44,var_43,])
func_45 = relay.Function([var_43,], output)
mod['func_45'] = func_45
mod = relay.transform.InferType()(mod)
mutated_mod['func_45'] = func_45
mutated_mod = relay.transform.InferType()(mutated_mod)
var_46 = relay.var("var_46", dtype = "int8", shape = (14,))#candidate|46|(14,)|var|int8
func_45_call = mutated_mod.get_global_var('func_45')
call_47 = func_45_call(var_46)
output = call_47
func_48 = relay.Function([var_46], output)
mutated_mod['func_48'] = func_48
mutated_mod = relay.transform.InferType()(mutated_mod)
var_50 = relay.var("var_50", dtype = "float64", shape = (3,))#candidate|50|(3,)|var|float64
uop_51 = relay.sinh(var_50.astype('float64')) # shape=(3,)
uop_53 = relay.sigmoid(uop_51.astype('float32')) # shape=(3,)
bop_55 = relay.mod(uop_53.astype('float64'), relay.reshape(var_50.astype('float64'), relay.shape_of(uop_53))) # shape=(3,)
bop_58 = relay.maximum(uop_53.astype('int32'), relay.reshape(bop_55.astype('int32'), relay.shape_of(uop_53))) # shape=(3,)
uop_61 = relay.asin(uop_51.astype('float64')) # shape=(3,)
uop_63 = relay.sinh(var_50.astype('float32')) # shape=(3,)
var_65 = relay.var("var_65", dtype = "float64", shape = (3,))#candidate|65|(3,)|var|float64
bop_66 = relay.bitwise_and(uop_51.astype('int8'), relay.reshape(var_65.astype('int8'), relay.shape_of(uop_51))) # shape=(3,)
bop_69 = relay.maximum(var_50.astype('uint16'), relay.reshape(bop_66.astype('uint16'), relay.shape_of(var_50))) # shape=(3,)
uop_72 = relay.sqrt(uop_61.astype('float32')) # shape=(3,)
output = relay.Tuple([bop_58,uop_63,bop_69,uop_72,])
output2 = relay.Tuple([bop_58,uop_63,bop_69,uop_72,])
func_74 = relay.Function([var_50,var_65,], output)
mod['func_74'] = func_74
mod = relay.transform.InferType()(mod)
var_75 = relay.var("var_75", dtype = "float64", shape = (3,))#candidate|75|(3,)|var|float64
var_76 = relay.var("var_76", dtype = "float64", shape = (3,))#candidate|76|(3,)|var|float64
output = func_74(var_75,var_76,)
func_77 = relay.Function([var_75,var_76,], output)
mutated_mod['func_77'] = func_77
mutated_mod = relay.transform.InferType()(mutated_mod)
var_79 = relay.var("var_79", dtype = "int32", shape = (10, 3, 16))#candidate|79|(10, 3, 16)|var|int32
const_80 = relay.const([[[4,-2,1,-10,1,-3,-1,6,-1,6,9,8,9,8,1,-3],[5,3,-3,6,-4,-6,6,-9,-10,-7,-9,2,3,3,7,10],[-3,7,-5,-10,2,8,-10,4,-9,9,-8,-9,1,6,10,-2]],[[-8,-9,-2,3,-8,1,-5,-10,-8,6,1,8,9,-4,1,-4],[-4,1,5,-1,7,1,-1,-8,-9,6,4,-4,1,4,3,-10],[-2,-5,6,-9,2,-7,-8,-1,-1,-9,-10,-8,10,-2,9,-9]],[[2,9,4,3,6,6,-4,3,1,-10,4,3,-9,6,-6,10],[-1,-7,7,-8,7,6,-10,-7,-10,5,5,9,-2,4,10,-9],[5,7,-6,-4,10,6,-1,-2,-2,-7,4,4,1,-5,4,-3]],[[9,-2,4,2,7,3,-5,-1,9,-7,-5,8,-9,-8,-8,-3],[-8,6,10,-5,7,5,2,-10,-7,5,-6,-8,7,-7,7,-6],[4,9,9,10,-4,10,2,-9,7,-1,-9,7,6,-3,-5,-10]],[[-2,-2,2,8,9,-2,-8,-2,-6,-7,3,-3,9,4,6,4],[2,8,9,2,4,1,-6,6,-3,9,6,-4,7,-10,1,-10],[9,9,-3,-6,-4,10,7,3,10,7,-1,2,1,2,5,7]],[[1,-2,6,5,4,-1,3,2,-6,-1,3,9,-7,-8,-5,6],[10,2,5,-7,3,9,-1,-9,-3,-8,-8,-8,7,7,10,8],[3,-7,7,7,-3,-8,-1,-3,-9,-9,-5,-2,-5,-4,4,7]],[[4,-8,-10,-3,-3,-7,-9,-9,1,-7,-6,-9,-10,8,-7,-3],[-10,1,8,-3,6,5,2,4,-3,10,-10,2,3,5,-6,-8],[1,-8,-5,10,-3,-5,-9,8,9,-3,8,2,5,1,8,-10]],[[1,10,1,5,4,-10,-10,7,8,-2,-5,-8,-2,2,8,-2],[9,5,9,5,2,10,-4,3,7,8,2,2,5,-5,-7,-2],[-7,8,8,3,1,-4,-10,-8,-4,9,-2,-9,7,2,1,2]],[[-2,5,-4,-2,2,9,-4,-9,3,4,10,-4,9,3,-9,-9],[-5,-1,-3,-3,8,6,-1,10,-4,-1,2,-3,9,-3,1,1],[6,-5,6,5,2,-6,6,-6,2,7,1,7,8,-2,1,6]],[[-7,6,1,-7,-3,-3,-1,-9,6,-9,8,1,-5,4,6,3],[1,7,1,4,-9,10,7,4,-9,-2,-10,-10,8,3,6,-9],[5,8,2,-7,8,-2,1,10,-1,-7,-7,-8,-3,-1,8,7]]], dtype = "int32")#candidate|80|(10, 3, 16)|const|int32
bop_81 = relay.multiply(var_79.astype('int32'), relay.reshape(const_80.astype('int32'), relay.shape_of(var_79))) # shape=(10, 3, 16)
bop_84 = relay.floor_mod(bop_81.astype('float64'), relay.reshape(const_80.astype('float64'), relay.shape_of(bop_81))) # shape=(10, 3, 16)
var_87 = relay.var("var_87", dtype = "int32", shape = (10, 3, 16))#candidate|87|(10, 3, 16)|var|int32
bop_88 = relay.bitwise_and(const_80.astype('int64'), relay.reshape(var_87.astype('int64'), relay.shape_of(const_80))) # shape=(10, 3, 16)
uop_91 = relay.atanh(var_87.astype('float64')) # shape=(10, 3, 16)
const_93 = relay.const([[[4.128322,-5.195828,7.216423,0.756057,-6.985706,-1.023560,-8.256302,1.933872,-1.076744,0.920668,-5.269364,3.125472,8.753241,9.201745,6.046491,-6.846682],[-1.090796,-7.469248,7.646319,7.707251,-1.361693,3.745306,2.577627,-4.928438,-4.987979,-7.776875,0.396917,-7.638057,2.430220,8.853999,-0.131071,6.549206],[0.814001,-6.819678,1.520170,-8.797435,-5.174075,-6.309293,-0.480874,3.135307,-9.440329,4.097707,-3.513933,-4.276257,4.474380,-8.111396,2.599684,-3.438668]],[[5.084469,-3.019835,7.677092,3.618037,-8.424538,9.805172,-8.220230,9.526693,-2.012393,9.512522,-4.651237,-1.298839,8.768482,2.271442,-9.905414,2.541664],[7.271253,-7.438633,1.930323,-9.472246,8.932322,-7.664985,-1.408309,-8.369495,6.745031,0.363004,4.072696,7.105924,7.904566,4.864307,6.022882,2.872664],[-4.216145,9.869837,9.330751,-4.013372,5.506473,3.723302,-8.119766,-0.102857,3.673935,-8.443294,1.151698,0.608847,-1.492027,-0.758187,-7.721808,-3.696887]],[[7.780727,-5.578790,7.807886,7.996765,1.843019,4.959004,-8.291546,3.843003,1.036968,-5.882382,-6.965156,0.692720,-4.463090,-1.036356,-8.509996,-5.531365],[-0.347683,-7.612781,8.035962,-3.945121,-5.963779,-3.395371,6.682402,-7.782936,3.706342,7.433953,0.447515,5.171652,-6.492973,-9.537451,-1.455860,-7.612552],[0.317143,0.532627,-7.322559,9.841766,4.110552,4.422248,-0.850026,7.574804,-9.216871,5.117108,-8.620534,0.824155,-2.952284,5.046848,-7.049570,6.157403]],[[-2.937016,-7.259043,-9.846422,-5.364408,-2.911353,-3.302529,6.342488,-3.505064,-3.738690,5.741418,-5.737948,4.871261,-1.579556,-4.935038,-0.367930,-2.193132],[-4.135723,0.566078,2.634124,-7.714236,3.674407,-3.712391,-4.314068,-0.320567,-5.851352,3.390081,-3.131355,7.166477,9.859986,-7.492916,8.946155,8.931041],[4.379700,-4.602076,-3.918530,-0.686773,7.131130,-2.152439,-6.196540,-6.161867,8.075048,0.506180,-2.648136,-2.235859,-9.365598,3.559544,-9.237913,1.179029]],[[6.822172,-9.225856,-1.770271,2.322502,8.304242,-4.283262,-6.092259,-0.668838,-2.460995,1.332125,-7.806158,9.062635,-5.579290,-7.108576,2.396625,4.543244],[1.242255,-7.997131,-4.150198,2.776189,2.055826,0.468810,-4.567598,8.767511,-0.593827,4.757150,7.466735,9.324607,3.101025,-2.637145,-2.069155,5.742633],[5.892764,4.506629,2.682531,1.353115,-7.848051,-8.579593,-0.491083,-6.165522,-4.966995,5.915934,-5.203766,-0.280256,-9.867332,-7.264403,-5.218464,-2.189380]],[[4.774261,8.518552,-2.681814,2.246685,-2.744160,2.415316,-5.930529,6.706690,9.954958,-7.426095,2.429242,0.416970,1.848954,-0.256454,5.090097,8.344300],[1.088510,0.880775,9.378005,-0.581455,-9.675402,-2.079864,8.764213,9.498529,-3.570743,-8.003391,-0.999445,6.605157,9.497167,1.962130,-1.092738,4.423122],[-5.100235,2.815600,-8.989955,3.206785,5.988128,4.642418,4.027631,3.589256,-8.462761,9.375610,2.967498,-4.959019,1.779110,3.675277,-4.891918,2.365160]],[[6.464451,-9.849996,-0.159700,-9.611360,4.520717,7.776735,-6.881712,-7.634469,-6.917572,-3.572392,-7.775100,8.219396,-2.259068,2.771742,5.018092,6.061878],[1.264627,-8.775631,9.010804,9.001078,-4.214134,8.748640,8.728531,6.452255,-6.942553,9.091589,-9.055276,-4.249396,9.920763,4.156960,-4.159111,7.994231],[-9.265413,4.230670,-6.725532,-4.560649,9.408728,2.955490,3.120388,5.663712,-8.917585,-6.834311,7.233165,-4.254352,-9.512578,1.667145,-3.416256,-1.522176]],[[2.111030,9.867095,-1.308628,-4.986341,7.170622,-6.097298,5.494885,-3.233273,-0.334667,-5.468909,3.927585,-2.532821,7.397624,-7.799099,-0.917472,-6.274041],[2.729658,1.173267,-7.313912,2.368645,-9.237358,-8.229110,3.062782,9.768763,-5.004837,6.996638,-2.408281,6.951156,-4.206967,-4.842782,4.740367,-2.218333],[-2.702832,-8.571762,3.325757,-5.531849,-1.289946,3.850700,-2.735079,0.718722,4.849970,-4.091479,9.032211,4.991946,-6.977202,-5.480720,-1.160693,-0.342820]],[[7.769383,4.636313,-0.197805,-3.746441,1.410113,9.190071,-5.039922,-1.280863,5.324908,-1.060638,-0.269276,-9.594197,5.906701,-9.459962,4.295557,-7.131050],[-6.895143,-2.451412,1.613933,4.264506,-5.404986,-6.881658,3.600318,-9.436400,-1.070565,2.483064,-2.665044,8.643648,0.517849,-9.802153,-7.950806,2.619968],[-5.098940,6.147422,-3.698361,-6.557409,8.372454,7.911306,5.415441,-3.577241,-8.308312,5.082871,4.807272,-5.484858,-9.870544,-2.194925,8.642982,0.183803]],[[8.551919,-2.955848,9.586945,-8.063021,4.772494,-1.266415,-6.900542,-4.830432,-5.164249,7.730453,-8.601155,-4.225473,-7.131933,0.940481,-2.550486,-4.123064],[-7.226764,4.565837,9.369902,-9.470733,-1.313697,-2.631151,3.209809,2.964243,-1.568889,-8.038995,0.497983,6.056124,1.179974,3.094425,-0.407868,4.642870],[-9.035262,-3.466796,-5.309799,9.457409,1.126947,1.699362,-7.573517,9.568894,-0.309709,-3.212637,-7.338553,8.685324,-7.211461,-9.224664,-4.555600,4.348716]]], dtype = "float64")#candidate|93|(10, 3, 16)|const|float64
bop_94 = relay.minimum(bop_84.astype('uint16'), relay.reshape(const_93.astype('uint16'), relay.shape_of(bop_84))) # shape=(10, 3, 16)
uop_97 = relay.log2(bop_84.astype('float64')) # shape=(10, 3, 16)
uop_99 = relay.cosh(uop_97.astype('float64')) # shape=(10, 3, 16)
bop_101 = relay.bitwise_or(uop_99.astype('uint8'), relay.reshape(const_93.astype('uint8'), relay.shape_of(uop_99))) # shape=(10, 3, 16)
uop_104 = relay.rsqrt(uop_91.astype('float32')) # shape=(10, 3, 16)
uop_106 = relay.sinh(uop_104.astype('float32')) # shape=(10, 3, 16)
output = relay.Tuple([bop_88,bop_94,bop_101,uop_106,])
output2 = relay.Tuple([bop_88,bop_94,bop_101,uop_106,])
func_108 = relay.Function([var_79,var_87,], output)
mod['func_108'] = func_108
mod = relay.transform.InferType()(mod)
mutated_mod['func_108'] = func_108
mutated_mod = relay.transform.InferType()(mutated_mod)
func_108_call = mutated_mod.get_global_var('func_108')
var_110 = relay.var("var_110", dtype = "int32", shape = (10, 3, 16))#candidate|110|(10, 3, 16)|var|int32
var_111 = relay.var("var_111", dtype = "int32", shape = (10, 3, 16))#candidate|111|(10, 3, 16)|var|int32
call_109 = func_108_call(var_110,var_111,)
output = call_109
func_112 = relay.Function([var_110,var_111,], output)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
var_114 = relay.var("var_114", dtype = "bool", shape = (9, 16, 9))#candidate|114|(9, 16, 9)|var|bool
const_115 = relay.const([[[True,True,True,False,False,True,True,True,False],[False,False,True,True,False,True,False,True,True],[True,False,False,False,False,True,False,False,True],[True,False,True,True,True,False,True,True,False],[False,False,False,False,True,False,False,False,False],[True,False,False,False,False,False,True,False,True],[False,False,True,True,False,True,False,True,False],[True,False,True,True,False,True,True,True,False],[False,True,True,False,False,True,False,False,True],[True,True,False,False,True,False,True,True,False],[False,True,False,False,False,False,True,False,True],[False,True,False,True,True,True,False,False,True],[True,False,True,True,True,True,True,True,False],[True,False,True,False,False,True,False,False,True],[False,True,True,True,False,True,True,True,False],[False,True,False,True,False,True,True,True,False]],[[False,False,False,False,False,False,False,False,True],[True,False,True,False,False,True,False,False,True],[True,True,False,True,False,True,False,True,True],[True,False,True,True,False,True,True,True,True],[False,True,True,True,False,True,False,False,False],[True,False,False,False,True,True,False,True,True],[True,True,True,True,False,True,False,True,False],[True,True,True,True,True,False,False,False,False],[False,False,False,False,False,True,False,False,False],[True,True,True,True,False,False,False,True,False],[True,True,False,True,True,True,False,False,False],[False,False,False,False,False,False,True,False,False],[False,False,True,False,False,False,False,True,False],[False,True,False,True,False,True,True,False,False],[False,False,False,False,True,True,False,True,True],[False,False,False,True,False,False,False,True,False]],[[False,True,True,False,True,False,False,False,True],[False,True,True,False,True,True,True,True,False],[False,True,False,True,False,False,True,True,True],[False,True,False,False,True,True,False,True,False],[False,False,True,True,False,False,False,False,False],[False,True,True,False,True,True,False,True,True],[True,False,False,False,False,True,False,True,False],[True,True,True,True,True,True,False,True,True],[True,True,False,True,True,True,False,False,True],[True,False,False,False,True,False,False,True,True],[True,True,False,True,False,True,True,False,False],[False,False,True,False,True,True,False,False,False],[True,True,False,False,False,True,False,True,False],[True,True,False,False,True,True,False,False,False],[True,True,False,True,False,False,True,False,False],[False,False,False,False,True,True,True,False,False]],[[False,False,True,False,True,False,False,True,True],[False,True,False,False,False,True,False,False,True],[False,True,True,False,True,True,True,True,True],[False,False,True,False,False,True,True,True,True],[False,True,False,True,True,False,True,True,False],[True,True,False,False,False,True,False,False,False],[True,True,False,False,False,False,True,False,True],[True,False,False,False,False,True,False,True,True],[False,True,False,True,False,False,True,False,False],[True,False,False,True,False,False,True,False,False],[False,False,False,True,True,False,True,True,False],[False,True,False,True,False,True,False,True,True],[False,False,True,False,True,False,False,True,False],[False,False,False,False,False,False,True,True,True],[True,False,False,False,True,False,False,False,False],[True,False,True,False,False,True,True,False,True]],[[True,True,False,True,True,False,False,False,True],[False,True,False,False,False,True,False,False,False],[False,False,False,False,True,True,True,True,True],[True,True,False,False,False,True,False,False,False],[False,False,False,True,False,True,False,False,False],[True,True,False,True,True,True,True,False,False],[False,True,False,False,False,True,False,False,True],[True,False,True,True,True,True,False,False,False],[True,False,False,True,True,True,False,False,True],[True,False,True,True,False,False,True,False,True],[False,False,True,False,True,False,True,True,True],[True,True,True,True,False,False,True,False,True],[True,False,False,False,True,False,True,False,False],[True,False,True,False,False,True,False,False,True],[False,False,False,True,True,True,False,False,True],[False,True,True,False,False,True,False,False,False]],[[False,True,True,False,True,True,True,True,True],[True,True,False,False,True,False,False,False,True],[True,True,True,False,True,False,False,True,True],[True,True,True,False,True,True,True,False,False],[False,True,True,True,False,True,True,False,False],[True,False,True,False,True,False,True,False,True],[False,False,True,True,True,False,False,True,False],[True,False,False,True,False,True,True,False,False],[False,True,False,False,True,True,True,True,False],[True,True,False,True,True,False,False,False,False],[False,False,True,False,False,False,False,True,False],[False,False,False,False,False,False,False,True,True],[True,False,False,False,False,True,False,True,False],[True,True,False,True,True,True,False,False,True],[False,False,False,True,False,True,True,False,True],[True,True,False,False,False,True,True,False,True]],[[False,True,False,True,False,True,True,True,False],[False,True,False,True,False,True,False,True,True],[True,False,False,False,False,True,True,False,True],[False,True,False,True,False,True,True,True,True],[False,False,False,False,True,True,True,False,True],[False,False,False,True,True,True,True,False,True],[False,True,True,False,True,True,False,False,True],[True,True,False,False,True,False,False,True,True],[True,False,False,True,False,False,True,False,False],[False,True,False,True,False,True,True,False,False],[False,False,False,True,True,True,True,True,False],[True,True,True,True,False,False,True,True,False],[True,True,False,True,True,False,True,True,False],[False,False,False,False,False,False,False,True,True],[True,False,False,True,True,True,True,False,True],[True,True,True,True,True,False,False,False,True]],[[False,False,False,False,False,False,False,True,False],[False,True,True,True,True,True,True,False,True],[False,True,True,True,False,True,False,False,False],[False,False,False,False,False,False,False,False,True],[True,False,False,True,False,True,True,True,False],[False,False,True,True,False,False,True,True,True],[False,False,True,False,False,True,False,False,True],[True,True,True,False,False,True,False,True,True],[True,False,False,False,True,False,True,False,False],[True,True,True,False,True,True,True,True,False],[False,False,False,True,True,True,False,True,True],[True,True,True,False,False,True,False,False,False],[False,True,True,False,True,False,False,True,False],[True,True,True,True,True,True,False,True,False],[True,True,True,True,True,False,False,True,True],[True,False,True,False,False,True,True,True,False]],[[True,True,True,True,False,False,True,False,False],[False,False,True,True,True,False,False,False,True],[True,False,True,False,False,True,True,False,True],[False,True,False,False,False,True,False,False,False],[False,True,False,False,True,False,True,False,True],[False,True,False,True,False,False,False,False,False],[True,True,False,True,False,True,True,False,False],[True,False,False,True,True,True,True,True,False],[True,True,True,False,True,False,False,False,False],[True,True,False,True,False,True,False,True,True],[True,True,False,True,False,False,True,True,True],[False,False,True,True,True,False,False,True,True],[False,False,True,False,True,False,False,False,True],[True,False,False,True,False,True,True,True,True],[True,True,False,True,True,False,False,False,True],[False,False,True,True,False,True,False,False,False]]], dtype = "bool")#candidate|115|(9, 16, 9)|const|bool
bop_116 = relay.logical_or(var_114.astype('bool'), relay.reshape(const_115.astype('bool'), relay.shape_of(var_114))) # shape=(9, 16, 9)
output = bop_116
output2 = bop_116
func_119 = relay.Function([var_114,], output)
mod['func_119'] = func_119
mod = relay.transform.InferType()(mod)
mutated_mod['func_119'] = func_119
mutated_mod = relay.transform.InferType()(mutated_mod)
var_120 = relay.var("var_120", dtype = "bool", shape = (9, 16, 9))#candidate|120|(9, 16, 9)|var|bool
func_119_call = mutated_mod.get_global_var('func_119')
call_121 = func_119_call(var_120)
output = call_121
func_122 = relay.Function([var_120], output)
mutated_mod['func_122'] = func_122
mutated_mod = relay.transform.InferType()(mutated_mod)
const_124 = relay.const([[9,-10,4,-1,1,4,3,-2,9,-3],[-5,4,-6,-2,5,10,-6,3,2,-10],[-6,-2,4,-10,-3,9,1,-2,4,3],[6,5,6,1,2,2,-8,8,8,6],[4,-1,-1,-1,-7,-6,8,4,-9,-1],[-8,10,-7,9,8,-7,5,5,-1,2],[-3,-2,2,-4,-10,-4,9,-4,2,-2],[-3,3,8,-3,-4,-8,7,-7,-5,2],[9,-4,9,5,-5,4,-7,-5,4,-4],[5,5,-4,-1,-7,-4,8,-5,10,-10],[6,-3,8,-7,-6,-1,5,-4,-2,7],[9,-8,-1,-4,2,6,-8,2,-8,8]], dtype = "uint8")#candidate|124|(12, 10)|const|uint8
var_125 = relay.var("var_125", dtype = "uint8", shape = (12, 10))#candidate|125|(12, 10)|var|uint8
bop_126 = relay.add(const_124.astype('uint8'), relay.reshape(var_125.astype('uint8'), relay.shape_of(const_124))) # shape=(12, 10)
output = relay.Tuple([bop_126,])
output2 = relay.Tuple([bop_126,])
func_129 = relay.Function([var_125,], output)
mod['func_129'] = func_129
mod = relay.transform.InferType()(mod)
var_130 = relay.var("var_130", dtype = "uint8", shape = (12, 10))#candidate|130|(12, 10)|var|uint8
output = func_129(var_130)
func_131 = relay.Function([var_130], output)
mutated_mod['func_131'] = func_131
mutated_mod = relay.transform.InferType()(mutated_mod)
var_133 = relay.var("var_133", dtype = "uint64", shape = ())#candidate|133|()|var|uint64
var_134 = relay.var("var_134", dtype = "uint64", shape = ())#candidate|134|()|var|uint64
bop_135 = relay.greater_equal(var_133.astype('bool'), var_134.astype('bool')) # shape=()
bop_138 = relay.logical_xor(bop_135.astype('uint32'), var_134.astype('uint32')) # shape=()
func_108_call = mod.get_global_var('func_108')
func_112_call = mutated_mod.get_global_var('func_112')
const_142 = relay.const([10,10,-6,-7,6,3,-9,-3,8,-6,6,7,-6,6,2,-3,-3,4,-10,4,-5,1,-5,-4,10,1,4,9,6,-1,-5,5,-7,-7,-2,-2,9,10,10,8,5,-3,-5,6,8,-2,-2,-2,5,-8,2,5,-5,-5,10,2,1,2,-10,7,-6,2,7,-6,5,3,-6,-7,5,4,5,1,-3,-10,-5,-4,10,-8,-2,-3,-10,4,-5,7,-5,7,6,4,5,4,-4,1,-4,3,5,5,-2,-3,10,-9,8,-4,4,-6,5,6,-8,-3,6,-10,-4,8,4,5,1,10,7,-8,4,-9,3,8,4,6,9,-5,-1,-9,6,8,5,1,-3,-5,10,-6,1,6,3,-9,8,-5,7,-3,-7,2,-10,-6,-9,-10,4,6,7,-3,4,5,6,-2,-6,-7,7,-7,-2,9,-5,-4,4,6,-2,4,-3,-7,-8,3,5,9,9,8,-8,-5,-8,3,9,-3,-10,-6,-3,3,3,7,6,-9,-1,-8,-5,-5,-2,-3,8,3,-3,3,3,1,-4,5,6,6,-10,-8,7,-8,-2,1,8,-7,4,-9,8,6,-6,-1,-4,2,1,-1,-3,-7,-9,-4,4,-8,-3,-9,4,-2,-6,-5,-10,-6,-6,9,-3,6,-5,-1,-4,-9,-5,-8,4,1,3,-2,-2,-5,-7,-10,9,4,-6,2,3,5,-10,7,2,-7,4,9,-10,-4,7,7,8,1,-6,4,-6,7,3,-1,-2,10,-8,1,3,4,7,-9,5,-1,-3,-2,3,7,3,-2,8,10,1,5,5,8,-3,9,7,10,-4,-10,-2,-4,7,-2,8,-3,5,4,9,4,-9,1,2,7,-6,2,4,-6,1,-5,2,-7,6,-9,-3,7,5,-7,1,-4,1,-6,-1,7,10,1,2,1,9,-2,-10,-6,9,3,-7,9,-2,-4,-5,6,-5,-9,5,4,7,-6,-5,3,6,-5,4,3,-9,10,-10,-1,-5,9,-10,6,8,-6,4,8,7,5,9,9,6,4,-8,-10,-5,5,-1,1,-4,-4,-6,10,-8,-7,4,-7,-5,-8,-3,-8,-7,-4,-7,-10,3,9,6,5,4,-3,7,-1,-8,-7,-9,-1,-1,9,-9,-7,2,-2,-3,4,-8,-8,-6,7,-1,-9,1,-9,3,10,-6,4,1,8,-2,3,-9,7,1,-3,10,1,2,-4,9,-8,-1,2,-8,-7,-3,2,-1,-9,-5,-5,4,6,7,10,-6,-10,3,-8,-4,-3,7,-1], dtype = "int32")#candidate|142|(480,)|const|int32
call_141 = relay.TupleGetItem(func_108_call(relay.reshape(const_142.astype('int32'), [10, 3, 16]), relay.reshape(const_142.astype('int32'), [10, 3, 16]), ), 0)
call_143 = relay.TupleGetItem(func_112_call(relay.reshape(const_142.astype('int32'), [10, 3, 16]), relay.reshape(const_142.astype('int32'), [10, 3, 16]), ), 0)
func_22_call = mod.get_global_var('func_22')
func_28_call = mutated_mod.get_global_var('func_28')
var_145 = relay.var("var_145", dtype = "int8", shape = (14,))#candidate|145|(14,)|var|int8
call_144 = relay.TupleGetItem(func_22_call(relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('float64'), [2, 7]), ), 3)
call_146 = relay.TupleGetItem(func_28_call(relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('int8'), [2, 7]), relay.reshape(var_145.astype('float64'), [2, 7]), ), 3)
uop_147 = relay.sqrt(call_144.astype('float32')) # shape=(2, 7)
uop_149 = relay.sqrt(call_146.astype('float32')) # shape=(2, 7)
var_150 = relay.var("var_150", dtype = "uint64", shape = (1,))#candidate|150|(1,)|var|uint64
bop_151 = relay.bitwise_xor(var_133.astype('int16'), var_150.astype('int16')) # shape=()
uop_154 = relay.exp(uop_147.astype('float32')) # shape=(2, 7)
uop_156 = relay.exp(uop_149.astype('float32')) # shape=(2, 7)
uop_157 = relay.sqrt(uop_147.astype('float64')) # shape=(2, 7)
uop_159 = relay.sqrt(uop_149.astype('float64')) # shape=(2, 7)
bop_160 = relay.bitwise_and(uop_157.astype('uint16'), relay.reshape(uop_154.astype('uint16'), relay.shape_of(uop_157))) # shape=(2, 7)
bop_163 = relay.bitwise_and(uop_159.astype('uint16'), relay.reshape(uop_156.astype('uint16'), relay.shape_of(uop_159))) # shape=(2, 7)
uop_164 = relay.acosh(uop_154.astype('float32')) # shape=(2, 7)
uop_166 = relay.acosh(uop_156.astype('float32')) # shape=(2, 7)
var_167 = relay.var("var_167", dtype = "float32", shape = (2, 7))#candidate|167|(2, 7)|var|float32
bop_168 = relay.logical_or(uop_164.astype('bool'), relay.reshape(var_167.astype('bool'), relay.shape_of(uop_164))) # shape=(2, 7)
bop_171 = relay.logical_or(uop_166.astype('bool'), relay.reshape(var_167.astype('bool'), relay.shape_of(uop_166))) # shape=(2, 7)
func_129_call = mod.get_global_var('func_129')
func_131_call = mutated_mod.get_global_var('func_131')
const_173 = relay.const([9,8,8,-5,8,-6,-8,5,-5,8,1,-1,4,2,-6,1,-10,-9,9,-3,7,-3,-4,-8,9,2,-7,-9,-6,-10,-4,10,-8,-4,5,3,7,-5,-8,-3,-1,9,7,9,-2,10,8,2,8,-4,-3,5,4,-2,-4,-1,7,-3,1,2,10,1,-1,-3,-7,2,-5,10,8,4,-5,7,9,10,1,-5,-3,9,7,3,-1,4,3,-5,-7,2,-6,-8,8,-6,10,-2,-8,-9,2,-9,-10,-9,10,-5,-5,-9,-6,9,-4,7,-8,1,7,-1,3,-4,-4,7,8,-10,-8,7,7,2], dtype = "uint8")#candidate|173|(120,)|const|uint8
call_172 = relay.TupleGetItem(func_129_call(relay.reshape(const_173.astype('uint8'), [12, 10])), 0)
call_174 = relay.TupleGetItem(func_131_call(relay.reshape(const_173.astype('uint8'), [12, 10])), 0)
bop_175 = relay.maximum(uop_157.astype('uint16'), relay.reshape(var_167.astype('uint16'), relay.shape_of(uop_157))) # shape=(2, 7)
bop_178 = relay.maximum(uop_159.astype('uint16'), relay.reshape(var_167.astype('uint16'), relay.shape_of(uop_159))) # shape=(2, 7)
uop_179 = relay.sqrt(bop_160.astype('float64')) # shape=(2, 7)
uop_181 = relay.sqrt(bop_163.astype('float64')) # shape=(2, 7)
uop_182 = relay.cos(uop_179.astype('float64')) # shape=(2, 7)
uop_184 = relay.cos(uop_181.astype('float64')) # shape=(2, 7)
uop_185 = relay.log10(uop_179.astype('float32')) # shape=(2, 7)
uop_187 = relay.log10(uop_181.astype('float32')) # shape=(2, 7)
bop_188 = relay.logical_or(uop_185.astype('bool'), relay.reshape(var_145.astype('bool'), relay.shape_of(uop_185))) # shape=(2, 7)
bop_191 = relay.logical_or(uop_187.astype('bool'), relay.reshape(var_145.astype('bool'), relay.shape_of(uop_187))) # shape=(2, 7)
bop_192 = relay.multiply(bop_188.astype('float64'), relay.reshape(uop_182.astype('float64'), relay.shape_of(bop_188))) # shape=(2, 7)
bop_195 = relay.multiply(bop_191.astype('float64'), relay.reshape(uop_184.astype('float64'), relay.shape_of(bop_191))) # shape=(2, 7)
uop_196 = relay.sinh(bop_175.astype('float64')) # shape=(2, 7)
uop_198 = relay.sinh(bop_178.astype('float64')) # shape=(2, 7)
var_199 = relay.var("var_199", dtype = "float64", shape = (2, 7))#candidate|199|(2, 7)|var|float64
bop_200 = relay.divide(uop_196.astype('float32'), relay.reshape(var_199.astype('float32'), relay.shape_of(uop_196))) # shape=(2, 7)
bop_203 = relay.divide(uop_198.astype('float32'), relay.reshape(var_199.astype('float32'), relay.shape_of(uop_198))) # shape=(2, 7)
bop_204 = relay.multiply(bop_200.astype('uint8'), relay.reshape(uop_157.astype('uint8'), relay.shape_of(bop_200))) # shape=(2, 7)
bop_207 = relay.multiply(bop_203.astype('uint8'), relay.reshape(uop_159.astype('uint8'), relay.shape_of(bop_203))) # shape=(2, 7)
uop_208 = relay.acosh(bop_192.astype('float64')) # shape=(2, 7)
uop_210 = relay.acosh(bop_195.astype('float64')) # shape=(2, 7)
func_22_call = mod.get_global_var('func_22')
func_28_call = mutated_mod.get_global_var('func_28')
call_211 = relay.TupleGetItem(func_22_call(relay.reshape(uop_208.astype('int8'), [2, 7]), relay.reshape(bop_192.astype('int8'), [2, 7]), relay.reshape(bop_204.astype('int8'), [2, 7]), relay.reshape(bop_188.astype('float64'), [2, 7]), ), 1)
call_212 = relay.TupleGetItem(func_28_call(relay.reshape(uop_208.astype('int8'), [2, 7]), relay.reshape(bop_192.astype('int8'), [2, 7]), relay.reshape(bop_204.astype('int8'), [2, 7]), relay.reshape(bop_188.astype('float64'), [2, 7]), ), 1)
uop_213 = relay.sinh(uop_147.astype('float32')) # shape=(2, 7)
uop_215 = relay.sinh(uop_149.astype('float32')) # shape=(2, 7)
uop_216 = relay.log10(uop_208.astype('float64')) # shape=(2, 7)
uop_218 = relay.log10(uop_210.astype('float64')) # shape=(2, 7)
bop_219 = relay.subtract(uop_208.astype('uint64'), relay.reshape(var_145.astype('uint64'), relay.shape_of(uop_208))) # shape=(2, 7)
bop_222 = relay.subtract(uop_210.astype('uint64'), relay.reshape(var_145.astype('uint64'), relay.shape_of(uop_210))) # shape=(2, 7)
bop_223 = relay.power(bop_219.astype('float32'), bop_138.astype('float32')) # shape=(2, 7)
bop_226 = relay.power(bop_222.astype('float32'), bop_138.astype('float32')) # shape=(2, 7)
output = relay.Tuple([call_141,const_142,bop_151,bop_168,call_172,const_173,bop_204,call_211,uop_213,uop_216,bop_223,])
output2 = relay.Tuple([call_143,const_142,bop_151,bop_171,call_174,const_173,bop_207,call_212,uop_215,uop_218,bop_226,])
func_227 = relay.Function([var_133,var_134,var_145,var_150,var_167,var_199,], output)
mod['func_227'] = func_227
mod = relay.transform.InferType()(mod)
mutated_mod['func_227'] = func_227
mutated_mod = relay.transform.InferType()(mutated_mod)
func_227_call = mutated_mod.get_global_var('func_227')
var_229 = relay.var("var_229", dtype = "uint64", shape = ())#candidate|229|()|var|uint64
var_230 = relay.var("var_230", dtype = "uint64", shape = ())#candidate|230|()|var|uint64
var_231 = relay.var("var_231", dtype = "int8", shape = (14,))#candidate|231|(14,)|var|int8
var_232 = relay.var("var_232", dtype = "uint64", shape = (1,))#candidate|232|(1,)|var|uint64
var_233 = relay.var("var_233", dtype = "float32", shape = (2, 7))#candidate|233|(2, 7)|var|float32
var_234 = relay.var("var_234", dtype = "float64", shape = (2, 7))#candidate|234|(2, 7)|var|float64
call_228 = func_227_call(var_229,var_230,var_231,var_232,var_233,var_234,)
output = call_228
func_235 = relay.Function([var_229,var_230,var_231,var_232,var_233,var_234,], output)
mutated_mod['func_235'] = func_235
mutated_mod = relay.transform.InferType()(mutated_mod)
var_237 = relay.var("var_237", dtype = "float32", shape = (13, 5, 4))#candidate|237|(13, 5, 4)|var|float32
uop_238 = relay.atan(var_237.astype('float32')) # shape=(13, 5, 4)
output = uop_238
output2 = uop_238
func_240 = relay.Function([var_237,], output)
mod['func_240'] = func_240
mod = relay.transform.InferType()(mod)
var_241 = relay.var("var_241", dtype = "float32", shape = (13, 5, 4))#candidate|241|(13, 5, 4)|var|float32
output = func_240(var_241)
func_242 = relay.Function([var_241], output)
mutated_mod['func_242'] = func_242
mutated_mod = relay.transform.InferType()(mutated_mod)
var_244 = relay.var("var_244", dtype = "float32", shape = (6, 2, 14))#candidate|244|(6, 2, 14)|var|float32
uop_245 = relay.tan(var_244.astype('float32')) # shape=(6, 2, 14)
bop_247 = relay.logical_and(uop_245.astype('bool'), relay.reshape(var_244.astype('bool'), relay.shape_of(uop_245))) # shape=(6, 2, 14)
uop_250 = relay.tan(bop_247.astype('float64')) # shape=(6, 2, 14)
output = relay.Tuple([uop_250,])
output2 = relay.Tuple([uop_250,])
func_252 = relay.Function([var_244,], output)
mod['func_252'] = func_252
mod = relay.transform.InferType()(mod)
mutated_mod['func_252'] = func_252
mutated_mod = relay.transform.InferType()(mutated_mod)
var_253 = relay.var("var_253", dtype = "float32", shape = (6, 2, 14))#candidate|253|(6, 2, 14)|var|float32
func_252_call = mutated_mod.get_global_var('func_252')
call_254 = func_252_call(var_253)
output = call_254
func_255 = relay.Function([var_253], output)
mutated_mod['func_255'] = func_255
mutated_mod = relay.transform.InferType()(mutated_mod)
var_257 = relay.var("var_257", dtype = "float32", shape = ())#candidate|257|()|var|float32
uop_258 = relay.log10(var_257.astype('float32')) # shape=()
output = uop_258
output2 = uop_258
func_260 = relay.Function([var_257,], output)
mod['func_260'] = func_260
mod = relay.transform.InferType()(mod)
var_261 = relay.var("var_261", dtype = "float32", shape = ())#candidate|261|()|var|float32
output = func_260(var_261)
func_262 = relay.Function([var_261], output)
mutated_mod['func_262'] = func_262
mutated_mod = relay.transform.InferType()(mutated_mod)
const_264 = relay.const([[[-7,2,1,-2,-6],[-8,7,-4,4,-1],[-8,7,8,-2,10],[4,4,-7,-10,3],[-1,2,3,-6,-6],[9,4,5,-9,10],[-10,7,8,1,3],[4,7,4,4,4],[-7,-5,9,2,8],[-7,6,-6,-10,-4],[1,6,6,7,-10],[-7,5,7,-1,-2]],[[-2,-9,5,-3,-2],[-7,-5,-2,-3,-10],[-1,-9,8,-10,6],[9,-5,8,-4,6],[-10,-1,-5,7,-4],[-2,-3,-1,8,1],[5,6,8,9,-7],[-10,-7,-1,5,-4],[-4,-4,-9,7,6],[5,-7,3,-5,7],[-9,-8,-9,-1,-8],[3,-3,-10,-2,2]],[[-4,-1,-8,-8,9],[4,4,4,-10,8],[-10,-1,-10,5,-5],[-1,-2,3,7,4],[7,-3,9,-4,10],[6,-5,-5,5,6],[-5,-2,-9,8,6],[-3,-9,4,2,6],[2,-6,-10,-1,-8],[-8,3,-3,-6,-10],[4,-5,-8,10,9],[-3,-1,-9,9,7]],[[-8,-2,6,-5,8],[-5,7,-5,9,6],[-5,-10,-8,-7,-10],[4,-9,2,4,6],[-1,4,10,-4,9],[8,8,8,8,5],[-8,-7,9,-2,7],[5,-8,-7,-2,-9],[10,3,-5,6,-6],[3,-1,-3,-10,-3],[5,-4,-1,-3,-1],[-2,-10,-8,7,10]],[[4,9,6,8,-9],[4,1,6,8,-4],[2,9,-2,6,-9],[-5,2,2,8,-3],[-1,8,4,3,-3],[-2,-5,-4,-5,10],[-1,-1,-1,-8,-1],[3,-7,-5,6,4],[6,-2,-6,-2,9],[9,7,2,2,-4],[-2,-6,-1,-1,-10],[-4,-2,9,-5,10]],[[-3,5,-8,-7,-6],[3,1,8,-2,-6],[-5,-9,9,4,1],[9,3,-5,2,9],[-2,-7,-2,4,5],[6,-2,-6,7,6],[-3,3,10,5,7],[-9,8,3,-2,-6],[8,-6,9,3,9],[-5,6,-8,-3,8],[9,-10,-7,9,-1],[-6,-6,-1,-2,2]],[[-4,8,4,-6,-1],[6,6,-8,-10,-10],[-1,2,-9,-4,-7],[-6,5,1,-10,-6],[-5,-10,10,-2,3],[3,-10,-3,8,-4],[8,-8,-4,2,-7],[-4,1,-1,-9,1],[-7,-8,-1,-10,10],[6,3,-1,-2,-10],[1,1,-7,-3,9],[10,10,5,-1,-2]],[[6,-9,1,-6,2],[2,-10,-5,7,-3],[4,2,-3,7,1],[-7,9,-8,3,-1],[-1,-8,6,-6,-1],[-9,4,3,8,3],[1,-8,3,9,1],[-3,-3,-3,-6,3],[-10,4,-2,1,-7],[1,4,8,-7,-5],[9,3,1,-2,-7],[-6,-5,-10,7,9]],[[8,4,1,-9,-3],[9,6,-5,-9,6],[3,-2,-10,7,6],[3,-8,-5,-1,9],[1,-5,1,-4,8],[-6,-1,9,-3,-9],[6,2,-1,3,3],[5,2,-9,-1,-3],[4,-1,-10,-6,-9],[-3,-9,9,3,-3],[5,-10,-9,10,-3],[-10,-4,4,1,10]],[[-5,5,9,-7,5],[1,-7,9,-2,-7],[-3,4,9,-3,5],[2,-8,7,2,-2],[-3,9,5,-5,6],[-2,10,6,1,2],[5,-10,-3,-10,1],[2,-6,4,-6,8],[1,-10,-3,10,6],[-4,6,-3,10,-3],[-7,7,-3,-7,-1],[2,-5,3,-9,-4]],[[7,-5,1,-3,1],[-6,-1,4,-5,5],[-7,8,-1,10,7],[2,-1,10,-3,4],[9,4,4,-5,8],[5,-9,7,-5,8],[10,-8,7,4,-1],[7,8,-3,7,3],[8,-3,10,-6,5],[-1,-8,-6,6,-6],[-10,-10,-4,3,1],[-9,4,7,-1,1]]], dtype = "uint32")#candidate|264|(11, 12, 5)|const|uint32
var_265 = relay.var("var_265", dtype = "uint32", shape = (11, 12, 5))#candidate|265|(11, 12, 5)|var|uint32
bop_266 = relay.greater_equal(const_264.astype('bool'), relay.reshape(var_265.astype('bool'), relay.shape_of(const_264))) # shape=(11, 12, 5)
bop_269 = relay.bitwise_and(const_264.astype('uint16'), relay.reshape(var_265.astype('uint16'), relay.shape_of(const_264))) # shape=(11, 12, 5)
uop_272 = relay.atanh(const_264.astype('float64')) # shape=(11, 12, 5)
bop_274 = relay.logical_or(uop_272.astype('bool'), relay.reshape(const_264.astype('bool'), relay.shape_of(uop_272))) # shape=(11, 12, 5)
bop_277 = relay.logical_xor(var_265.astype('int32'), relay.reshape(bop_274.astype('int32'), relay.shape_of(var_265))) # shape=(11, 12, 5)
uop_280 = relay.acos(bop_274.astype('float32')) # shape=(11, 12, 5)
func_240_call = mod.get_global_var('func_240')
func_242_call = mutated_mod.get_global_var('func_242')
const_283 = relay.const([3.089455,0.835626,-0.352092,-8.014015,-7.279376,-9.838528,3.167391,3.547504,-4.288240,8.582345,0.981546,2.223512,6.684296,2.616023,-6.805791,-6.749446,7.001816,-5.147626,1.246391,0.842893,-5.697162,9.265770,9.909315,6.337888,5.897610,-5.594565,8.284330,6.880700,0.814387,5.926386,-9.385018,3.392855,-4.871544,-0.391701,6.074550,1.401386,7.460172,8.811556,3.051333,-6.772045,-9.636215,9.132096,-3.593957,-2.812407,1.879217,5.056442,1.303567,9.791961,-5.103142,-6.785675,3.914531,0.093185,-5.698842,-4.611429,-6.795552,0.868617,4.754647,-0.092680,9.348669,-6.601225,-0.224337,4.764683,-6.318294,6.926488,-5.670943,-7.602103,-3.760741,-2.049293,7.128280,-3.204958,-1.542357,-8.993441,2.308805,9.200137,4.043837,3.969289,-8.784551,-3.960202,-1.125202,2.056150,3.099051,-9.363816,-1.113199,-1.185503,4.504864,7.742611,-2.492680,-0.848631,-4.130957,8.974127,8.415544,6.425406,9.917472,-5.180509,-2.919228,-4.180028,-9.623613,0.670333,3.995000,-0.803063,-4.384553,-4.484236,9.431535,-3.020502,6.235881,1.745480,2.626316,-6.814239,-1.057488,-6.265482,-2.742109,-9.467649,2.219288,8.068797,-6.708542,-0.266226,9.210932,7.026248,6.075934,-8.545906,1.319378,7.080451,5.250249,0.051849,-7.047032,-8.525364,-1.470973,6.300378,-3.173227,6.793014,-6.956576,7.711566,-2.804031,-4.729232,5.641168,-6.503897,4.944421,1.708538,9.848270,-4.874458,-5.943140,9.710647,-6.727555,-8.622633,3.585168,9.730728,8.792986,-0.606669,5.360658,9.124818,7.645511,-4.430962,0.253842,4.781574,-1.257660,2.779106,-3.987164,-0.674441,4.886169,7.028432,-0.400630,8.777593,-8.795870,1.952667,-5.178760,-3.371411,8.201503,-1.886415,-3.895044,6.534446,-8.242713,9.675713,0.668436,2.471662,-1.875373,-3.505301,0.350356,9.627613,2.085840,5.262594,7.679007,9.115452,-6.863417,-4.284361,-4.958117,-4.434823,-7.830572,4.456300,6.017087,-4.677324,-2.902459,-9.960130,-3.736477,3.549575,-7.804363,-3.940281,-5.644037,-8.238991,0.544740,8.621690,7.787737,-4.147985,6.858561,4.776854,-0.878112,-4.156402,-3.790129,-5.086323,-9.790695,-1.613317,7.822399,0.086845,0.162327,-3.114770,-3.343916,-3.982042,2.912031,1.599472,-7.787227,5.050358,-7.687054,2.766592,-7.885016,-5.505476,4.291579,-8.205090,9.723118,-6.119483,3.757325,7.665359,-7.433820,3.925433,7.476276,-1.082208,-0.403096,-9.420368,-8.773642,4.356468,-6.191506,9.828155,2.769308,-6.963639,6.873318,8.680351,6.218941,-7.686151,-4.981247,-0.642582,-7.381625,7.169583,-5.216861,-8.318162,-5.546430,-6.109611,9.542388,-2.270931,5.689772,3.126358,-6.470717,3.203510], dtype = "float32")#candidate|283|(260,)|const|float32
call_282 = func_240_call(relay.reshape(const_283.astype('float32'), [13, 5, 4]))
call_284 = func_240_call(relay.reshape(const_283.astype('float32'), [13, 5, 4]))
bop_285 = relay.less(bop_266.astype('bool'), relay.reshape(uop_272.astype('bool'), relay.shape_of(bop_266))) # shape=(11, 12, 5)
bop_288 = relay.add(const_283.astype('int32'), relay.reshape(call_282.astype('int32'), relay.shape_of(const_283))) # shape=(260,)
bop_291 = relay.add(const_283.astype('int32'), relay.reshape(call_284.astype('int32'), relay.shape_of(const_283))) # shape=(260,)
uop_292 = relay.exp(bop_274.astype('float32')) # shape=(11, 12, 5)
func_260_call = mod.get_global_var('func_260')
func_262_call = mutated_mod.get_global_var('func_262')
const_295 = relay.const(-3.757977, dtype = "float32")#candidate|295|()|const|float32
call_294 = func_260_call(relay.reshape(const_295.astype('float32'), []))
call_296 = func_260_call(relay.reshape(const_295.astype('float32'), []))
uop_297 = relay.sigmoid(uop_280.astype('float64')) # shape=(11, 12, 5)
func_119_call = mod.get_global_var('func_119')
func_122_call = mutated_mod.get_global_var('func_122')
const_300 = relay.const([True,True,False,True,True,True,False,False,False,False,False,False,True,False,False,False,False,True,False,True,True,True,True,True,False,True,True,True,True,True,True,True,False,True,False,True,False,False,True,True,True,True,True,False,False,True,False,False,False,True,True,True,False,False,False,False,True,True,False,False,True,True,True,True,True,False,True,True,False,False,False,True,False,False,True,False,True,False,False,False,True,True,True,True,True,False,False,False,True,False,False,False,True,False,False,False,False,True,False,False,True,False,False,True,False,True,True,False,True,True,False,False,False,True,False,True,True,False,True,True,False,False,True,True,False,True,False,False,False,False,False,False,False,False,True,True,False,True,True,True,False,True,False,True,False,False,False,False,False,False,True,False,False,False,True,False,False,True,False,False,True,False,False,False,True,False,True,True,True,False,False,True,True,False,False,True,False,True,True,False,True,False,False,True,True,False,True,True,True,True,True,True,True,False,True,False,False,False,True,True,False,False,False,True,False,True,False,True,False,False,True,True,False,False,False,True,False,True,False,True,False,False,False,True,False,True,False,False,True,True,True,True,True,True,False,False,False,True,True,False,True,False,True,True,False,True,True,False,False,True,False,False,True,False,False,True,False,False,True,True,True,False,True,True,False,True,True,False,False,False,True,True,False,False,True,True,False,False,True,False,True,True,True,True,False,True,False,False,True,False,True,False,False,False,True,False,False,False,True,False,False,False,False,True,False,True,False,False,True,True,True,False,True,False,True,True,True,False,True,False,False,False,False,False,True,False,True,True,False,False,True,True,False,True,False,False,False,False,True,True,True,False,False,False,False,True,True,True,True,False,True,True,True,True,False,False,True,True,True,False,True,False,True,True,False,True,True,False,True,False,False,False,False,False,True,False,True,False,True,True,True,False,False,False,False,False,False,True,True,True,True,False,True,False,True,True,True,True,False,False,True,False,True,False,False,False,False,True,False,False,False,True,False,True,True,False,True,True,False,True,False,True,True,False,False,True,True,True,False,True,False,True,True,True,True,True,True,False,True,True,False,True,True,False,False,False,True,False,False,True,True,False,False,False,False,False,True,False,False,True,True,False,True,True,True,False,False,False,False,True,True,False,True,False,True,True,True,False,True,True,True,False,True,True,True,False,False,False,False,False,False,True,False,True,False,True,True,True,True,False,False,False,False,True,True,True,True,False,True,False,True,False,True,False,False,False,False,False,False,False,False,False,False,False,True,False,True,True,True,False,True,False,False,True,True,True,True,False,True,False,True,False,True,False,True,True,False,True,True,False,False,True,True,False,True,False,False,False,True,False,False,False,False,False,False,True,False,True,False,True,True,True,False,False,True,True,True,True,False,False,True,False,True,False,False,False,True,True,False,False,True,False,True,True,True,True,False,True,False,False,False,True,True,False,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,False,True,True,True,False,False,False,True,False,True,True,True,True,False,False,False,False,True,True,False,False,True,False,True,False,True,True,True,True,True,True,False,False,False,True,False,False,True,True,False,False,False,True,True,True,False,True,True,True,True,True,True,False,True,True,False,True,False,True,False,True,False,False,False,False,True,False,False,False,False,True,False,False,False,False,True,False,True,True,True,False,False,True,True,False,False,True,True,False,True,True,True,True,False,True,False,True,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,True,True,True,True,True,True,False,True,False,False,False,True,False,False,True,True,False,False,True,False,True,False,False,True,False,False,False,False,True,False,True,True,True,False,False,False,True,True,True,True,True,True,True,True,True,False,False,True,False,False,True,False,False,False,True,False,False,True,True,False,True,False,True,False,False,True,False,False,False,True,True,True,True,False,False,False,True,True,False,True,True,True,True,True,True,True,True,False,False,False,False,False,False,True,False,True,False,True,True,True,False,False,False,True,True,True,False,False,False,False,True,True,True,True,False,True,False,True,True,False,False,True,False,False,False,True,True,False,False,False,True,False,True,False,False,False,True,False,False,True,False,True,False,True,False,False,False,False,True,True,True,True,False,True,False,False,False,True,True,False,False,False,True,True,False,True,True,True,True,True,False,True,False,False,False,True,False,True,True,False,False,False,True,True,False,True,True,False,True,False,True,True,True,False,False,True,True,True,True,False,False,True,True,False,False,True,True,False,False,True,False,True,True,False,False,True,True,True,False,False,False,True,True,True,True,True,False,False,False,True,False,False,True,True,True,True,False,False,True,True,True,False,False,True,False,False,False,True,True,False,False,True,True,True,False,False,True,True,False,True,False,False,False,True,True,True,False,False,True,False,True,True,False,True,False,False,True,False,True,False,True,True,False,False,True,False,True,False,True,True,True,False,False,True,True,True,False,False,True,False,False,False,True,False,True,True,False,False,True,True,True,False,True,True,True,False,True,False,False,True,True,True,True,True,True,False,True,True,False,False,True,False,True,False,False,False,True,False,True,True,False,False,True,True,True,False,True,False,False,True,True,False,False,False,True,True,True,False,True,True,True,False,False,False,True,False,False,False,True,True,True,True,True,True,False,False,True,True,True,False,False,False,False,False,True,True,False,False,False,True,True,True,True,True,True,False,False,True,True,True,True,False,False,False,True,False,True,True,True,False,True,True,False,True,True,True,False,True,True,False,False,True,True,False,False,False,False,False,False,True,True,True,False,True,True,True,True,False,False,False,False,True,True,True,False,True,False,True,False,False,True,True,True,True,True,True,True,True,False,True,True,True,False,True,True,False,False,False,False,True,True,True,False,False,False,True,True,False,False,False,False,True,True,True,False,False,True,False,True,True,True,False,False,False,True,True,False,False,False,False,True,True,True,True,True,True,False,True,False,False,True,False,False,False,False,False,False,True,False,True,False,False,False,False,False,True,False,False,True,False,False,False,True,False,False,False,True,False,True,True,True,False,False,True,False,False,True,False,True,False,False,True,False,False,True,False,True,True,False,True,True,False,True,False,False], dtype = "bool")#candidate|300|(1296,)|const|bool
call_299 = func_119_call(relay.reshape(const_300.astype('bool'), [9, 16, 9]))
call_301 = func_119_call(relay.reshape(const_300.astype('bool'), [9, 16, 9]))
uop_302 = relay.tan(uop_280.astype('float32')) # shape=(11, 12, 5)
bop_304 = relay.logical_xor(uop_302.astype('uint64'), relay.reshape(bop_277.astype('uint64'), relay.shape_of(uop_302))) # shape=(11, 12, 5)
uop_307 = relay.log10(uop_302.astype('float64')) # shape=(11, 12, 5)
uop_309 = relay.log(uop_307.astype('float32')) # shape=(11, 12, 5)
bop_311 = relay.logical_and(uop_292.astype('bool'), relay.reshape(const_264.astype('bool'), relay.shape_of(uop_292))) # shape=(11, 12, 5)
bop_314 = relay.floor_mod(uop_307.astype('float64'), relay.reshape(uop_309.astype('float64'), relay.shape_of(uop_307))) # shape=(11, 12, 5)
uop_317 = relay.acosh(uop_309.astype('float64')) # shape=(11, 12, 5)
uop_319 = relay.sigmoid(uop_302.astype('float64')) # shape=(11, 12, 5)
var_321 = relay.var("var_321", dtype = "float32", shape = (11, 12, 5))#candidate|321|(11, 12, 5)|var|float32
bop_322 = relay.mod(uop_309.astype('float32'), relay.reshape(var_321.astype('float32'), relay.shape_of(uop_309))) # shape=(11, 12, 5)
bop_325 = relay.bitwise_and(uop_309.astype('int32'), relay.reshape(bop_269.astype('int32'), relay.shape_of(uop_309))) # shape=(11, 12, 5)
uop_328 = relay.acosh(uop_309.astype('float32')) # shape=(11, 12, 5)
func_240_call = mod.get_global_var('func_240')
func_242_call = mutated_mod.get_global_var('func_242')
call_330 = func_240_call(relay.reshape(const_283.astype('float32'), [13, 5, 4]))
call_331 = func_240_call(relay.reshape(const_283.astype('float32'), [13, 5, 4]))
uop_332 = relay.acosh(bop_322.astype('float64')) # shape=(11, 12, 5)
bop_334 = relay.mod(uop_317.astype('float32'), relay.reshape(var_265.astype('float32'), relay.shape_of(uop_317))) # shape=(11, 12, 5)
uop_337 = relay.tan(bop_311.astype('float32')) # shape=(11, 12, 5)
var_339 = relay.var("var_339", dtype = "float32", shape = (11, 12, 5))#candidate|339|(11, 12, 5)|var|float32
bop_340 = relay.divide(bop_334.astype('float32'), relay.reshape(var_339.astype('float32'), relay.shape_of(bop_334))) # shape=(11, 12, 5)
var_343 = relay.var("var_343", dtype = "float64", shape = (11, 12, 5))#candidate|343|(11, 12, 5)|var|float64
bop_344 = relay.subtract(uop_307.astype('uint32'), relay.reshape(var_343.astype('uint32'), relay.shape_of(uop_307))) # shape=(11, 12, 5)
bop_347 = relay.logical_and(bop_322.astype('bool'), relay.reshape(uop_292.astype('bool'), relay.shape_of(bop_322))) # shape=(11, 12, 5)
uop_350 = relay.log(uop_302.astype('float32')) # shape=(11, 12, 5)
bop_352 = relay.multiply(uop_332.astype('int16'), relay.reshape(uop_317.astype('int16'), relay.shape_of(uop_332))) # shape=(11, 12, 5)
uop_355 = relay.acos(bop_347.astype('float32')) # shape=(11, 12, 5)
uop_357 = relay.cosh(bop_325.astype('float32')) # shape=(11, 12, 5)
const_359 = relay.const([[[5.289289,6.331448,8.053083,-2.788475,9.595591],[-1.450270,8.704109,-3.668481,5.241773,4.181100],[7.404982,9.199445,-5.091899,-9.198478,5.700676],[3.084439,8.104270,-2.435574,-7.532786,-0.400416],[-0.095893,-3.101426,-9.360361,5.838718,-1.162836],[7.228861,-7.388002,-5.561835,0.613944,-8.687033],[-7.532223,-6.585898,-3.095132,-2.519939,5.353578],[-7.698458,-0.485320,4.302103,1.280959,-6.432245],[3.224350,6.699971,4.920007,-1.715912,-9.404315],[5.494721,6.997235,-3.876021,-1.818771,-6.351961],[1.683030,5.453537,3.849910,0.826143,1.509210],[-9.976927,6.143808,7.957787,1.549537,-8.506508]],[[-8.125627,-7.692276,6.560975,0.479255,-4.498326],[5.242712,-3.514886,7.529770,9.522211,-5.005016],[-5.792539,0.284142,-9.056991,-5.277368,-4.517376],[1.695008,-6.690294,-0.097275,-2.959963,7.090514],[-5.624107,0.993173,-1.447547,-7.444394,7.249737],[-2.652385,-9.405918,7.032533,-3.757043,-7.117311],[-8.978802,5.351667,-1.342861,-8.479602,6.876767],[9.250416,-7.266229,-8.638694,1.587484,-6.499884],[1.071402,-8.536811,-8.431609,-1.667820,-6.120548],[-8.030547,-3.196145,7.233361,4.344377,-9.851452],[-8.291458,-2.710689,-7.401983,9.865259,-7.229022],[-0.864455,-4.067160,3.252462,-7.209802,-7.555494]],[[-1.397443,-8.047048,-5.775105,9.901025,9.766878],[8.124876,1.309191,4.704721,-9.819861,6.347192],[-9.960309,8.146506,9.468900,-8.203347,-2.778055],[-7.392916,5.342010,-4.288890,-5.777792,0.830869],[8.849568,8.625736,2.618791,-2.027642,7.730975],[-1.030209,6.236615,0.436160,-3.657545,-4.108585],[-1.567223,1.615819,8.540619,9.302594,-8.835141],[-4.794251,2.654782,-8.441209,-2.208730,5.201587],[0.841803,1.972591,2.688266,4.312942,-9.357962],[-3.339767,-1.573911,-5.628453,8.200312,0.288617],[9.913662,7.010970,-0.544664,5.937814,-4.220742],[-6.709747,-9.095574,-5.267888,-6.325983,0.896646]],[[-6.742684,1.996021,1.889826,-0.198593,-8.278071],[-6.424305,9.124203,-6.038344,-2.751095,-1.147469],[-4.475937,-9.205169,8.701176,-6.015157,-7.122553],[-0.912198,-5.260274,1.844118,-3.522430,-6.535537],[4.853425,-4.768825,6.153492,-8.181298,-0.599228],[-5.296867,-0.854544,3.899704,-6.962151,8.574585],[0.592214,8.631341,-2.406129,6.408369,-6.734552],[0.765737,7.223062,0.019060,-0.099869,-7.174832],[-3.271136,-2.640285,7.692176,-6.726494,9.123008],[-6.339344,-0.165653,-6.662921,-4.868963,3.000453],[-2.646386,-0.690208,-8.922129,-1.310236,-0.454655],[1.424604,9.833198,1.216059,1.458533,-7.412771]],[[0.928886,3.729932,-8.782952,-9.772190,5.270934],[-2.018630,-3.737560,3.100454,-4.225280,-2.619755],[-9.891836,3.746571,9.387951,-4.617049,6.758796],[6.963118,6.962452,-9.466672,3.940255,-5.135004],[-9.858987,-1.783362,-5.397643,1.082820,3.840341],[6.750760,2.233729,1.238015,8.453089,8.251640],[2.587846,1.659534,5.652547,-7.239915,9.809607],[4.508686,2.416547,-6.706533,-6.562138,-0.320635],[0.327455,2.274436,0.159231,3.236528,8.781279],[9.983733,-7.281922,4.979822,-0.226155,-7.060600],[1.921574,0.937541,2.595216,-0.519132,6.554606],[5.586554,-5.256609,6.168430,8.101210,-8.270365]],[[0.950802,0.172202,-2.022091,9.902706,1.841982],[-2.820970,7.738444,9.585785,-2.120581,-2.834974],[5.165247,-5.833841,6.764770,1.700598,8.414444],[2.390354,0.877141,-9.704849,7.417930,-1.235629],[0.014277,1.366593,-5.280495,2.664105,-7.804114],[-2.310207,1.882700,6.552546,-8.915995,2.380230],[-1.275813,-4.996609,-9.540690,-8.431439,-4.257045],[-1.855425,7.115655,-9.282392,-7.472900,-4.689215],[-9.558315,-0.606448,-6.727915,-4.883666,-7.034358],[-2.034940,-4.778960,-2.281076,2.582171,-6.228746],[1.737024,6.032837,2.122710,-7.754801,0.206320],[-4.071653,6.846573,4.619870,6.724833,-4.756270]],[[3.072489,-6.939564,5.544612,-9.101536,9.472118],[-9.382617,4.833289,-2.377119,-7.757700,2.623664],[5.709220,4.313891,-9.777834,-1.222917,-0.235394],[-5.325922,4.975510,7.091519,9.952339,1.501672],[-2.046055,-3.985447,7.831102,-8.048448,4.779991],[9.609450,-2.248317,-5.449061,4.160891,-3.431437],[4.964874,-3.816909,4.516112,-7.371940,1.857696],[-8.420550,8.235904,-8.831348,3.239093,-2.430021],[-9.754390,9.289833,-8.226296,4.647864,4.313445],[9.683267,-6.138441,0.255945,0.631685,4.303229],[-4.877741,2.820147,-1.628643,-9.096586,-3.792831],[-4.274407,2.985481,7.953842,5.721076,8.578385]],[[-4.684220,-5.874741,-8.864932,1.729878,6.211173],[9.820290,-9.814000,2.377265,-1.341592,-9.303291],[4.569637,4.608732,-8.879648,-2.542797,8.349920],[-7.712649,6.467860,7.669793,4.575935,5.961393],[9.589400,-9.159313,7.443013,-4.796968,8.340011],[4.451419,3.276842,2.903827,7.002140,-2.986789],[-6.870395,5.156565,2.857167,1.349971,1.522810],[4.703189,2.948110,-1.308127,3.745801,-1.397148],[4.183021,5.942716,-1.430694,-1.189221,9.608815],[3.569333,-8.177617,3.205796,1.652234,-2.440306],[9.375258,7.682210,6.085166,-9.140573,-5.849561],[9.084091,-9.842010,-4.752967,-1.442594,-2.272202]],[[8.767446,2.516690,-3.643479,2.341479,1.894163],[-9.469345,5.048293,-1.669983,4.341962,-4.877822],[2.681467,1.450444,-6.396942,9.572696,2.077788],[-5.856599,1.431513,-7.509921,-2.064161,7.209327],[7.879680,4.840051,9.737795,1.573285,-8.766456],[-8.093131,3.528251,3.203869,4.258469,-5.685327],[-4.748593,-4.821296,-1.370637,-0.558233,-5.875749],[6.677230,2.915828,1.775122,-4.290685,-6.081424],[-1.048347,-5.333836,-4.105692,4.795954,-0.707622],[-9.335634,4.345272,-0.836808,8.641128,-2.798132],[-6.221659,9.996622,-9.997236,-5.389815,0.844423],[3.976090,8.834018,-8.540039,1.646633,-9.919973]],[[1.767268,9.390747,9.633948,2.752388,-6.854906],[-6.932579,4.970461,-3.589061,-7.529868,-6.086200],[3.617484,4.139073,3.755419,7.307123,-4.160000],[0.040486,-1.676519,-1.707913,-6.297146,4.036654],[8.654390,-6.893043,5.257292,9.244115,1.613322],[-7.907036,7.354879,-5.282640,6.472698,-7.612044],[4.342581,4.660540,-1.041501,6.749578,6.845159],[9.296559,-3.808327,2.497593,-4.326856,-3.686478],[-2.350194,8.251525,-4.621023,5.669317,5.205932],[8.559713,4.024212,-0.204493,-6.405924,-2.658906],[-0.047145,5.060073,8.766456,0.969773,4.010547],[-9.820009,-9.921074,7.575092,0.964300,-1.136016]],[[-0.514366,9.296099,-3.340603,8.315065,5.421941],[3.463932,2.226217,-3.393578,6.838104,0.770853],[5.874980,8.111806,4.752253,4.390476,1.042374],[7.081154,0.328378,4.963562,-3.435840,2.958945],[-5.849072,-9.581440,2.784659,8.979821,-0.342223],[2.596619,-5.834800,7.989150,6.895351,-6.350426],[-8.282540,6.375092,-2.780041,6.946236,6.221551],[-4.812150,0.233823,-4.248658,9.465511,-5.892335],[-4.084860,-0.619325,-9.099256,-3.084025,-6.568359],[-7.761430,-4.234669,-6.502839,-2.806372,3.113862],[-7.073030,9.171706,-1.789315,-5.104706,4.065382],[6.136119,-0.832218,-1.580841,0.745643,0.102222]]], dtype = "float64")#candidate|359|(11, 12, 5)|const|float64
bop_360 = relay.left_shift(uop_317.astype('uint8'), relay.reshape(const_359.astype('uint8'), relay.shape_of(uop_317))) # shape=(11, 12, 5)
uop_363 = relay.asin(bop_340.astype('float64')) # shape=(11, 12, 5)
uop_365 = relay.exp(uop_363.astype('float64')) # shape=(11, 12, 5)
output = relay.Tuple([bop_285,bop_288,call_294,const_295,uop_297,call_299,const_300,bop_304,bop_314,uop_319,uop_328,call_330,uop_337,bop_344,uop_350,bop_352,uop_355,uop_357,bop_360,uop_365,])
output2 = relay.Tuple([bop_285,bop_291,call_296,const_295,uop_297,call_301,const_300,bop_304,bop_314,uop_319,uop_328,call_331,uop_337,bop_344,uop_350,bop_352,uop_355,uop_357,bop_360,uop_365,])
func_367 = relay.Function([var_265,var_321,var_339,var_343,], output)
mod['func_367'] = func_367
mod = relay.transform.InferType()(mod)
mutated_mod['func_367'] = func_367
mutated_mod = relay.transform.InferType()(mutated_mod)
func_367_call = mutated_mod.get_global_var('func_367')
var_369 = relay.var("var_369", dtype = "uint32", shape = (11, 12, 5))#candidate|369|(11, 12, 5)|var|uint32
var_370 = relay.var("var_370", dtype = "float32", shape = (11, 12, 5))#candidate|370|(11, 12, 5)|var|float32
var_371 = relay.var("var_371", dtype = "float32", shape = (11, 12, 5))#candidate|371|(11, 12, 5)|var|float32
var_372 = relay.var("var_372", dtype = "float64", shape = (11, 12, 5))#candidate|372|(11, 12, 5)|var|float64
call_368 = func_367_call(var_369,var_370,var_371,var_372,)
output = call_368
func_373 = relay.Function([var_369,var_370,var_371,var_372,], output)
mutated_mod['func_373'] = func_373
mutated_mod = relay.transform.InferType()(mutated_mod)
const_375 = relay.const([[[7.291679,-4.173475,8.750738],[-1.446314,0.496211,-3.969942],[-1.487148,-9.038458,-6.623142],[-9.333614,-5.931219,-6.587154],[-6.254840,-6.914571,-4.485831],[-8.157535,-9.517115,-2.556028],[-8.287703,-1.011288,-9.347953],[8.567307,-2.840931,-3.760035],[0.303466,-9.268420,-2.315901],[-5.452254,-1.380633,-0.213039]],[[5.422325,-4.074041,9.734437],[7.087446,5.812744,-3.996399],[-6.393329,5.993087,5.506969],[4.016065,-5.456506,-4.851929],[-5.960870,1.629721,7.405894],[9.624052,-7.901343,-9.400882],[5.612546,-5.840718,-8.100183],[-7.303088,-8.201057,7.834822],[1.064269,0.723616,-6.514687],[-5.007879,-8.474488,0.301512]],[[-7.162569,-9.452412,5.349050],[-7.100877,-6.378251,8.776958],[6.502164,-4.313493,6.966543],[5.666261,-3.227201,-2.715370],[8.474388,-9.531554,6.018315],[1.036078,-7.616695,-9.379963],[2.245354,-9.291144,2.548018],[-4.149415,9.235190,7.830237],[-7.587472,-4.926587,9.140327],[7.841458,5.397841,-8.009401]],[[-3.467334,-4.130028,-1.522176],[-3.029504,-9.134489,5.992254],[-8.000816,-1.251853,-5.687324],[7.782601,7.420314,-2.379303],[-9.801947,4.744682,7.452749],[-4.036418,6.439911,-4.574184],[-2.946574,1.707271,-9.783061],[-2.313770,-0.657944,-8.778225],[-7.311108,2.925266,7.028554],[-4.858682,-1.959150,-3.305857]],[[-9.107131,7.680443,1.730499],[8.924339,-8.576753,-8.034407],[1.424926,3.063708,-2.965642],[5.549492,0.096613,-8.153771],[-4.100308,9.801609,-3.316627],[-6.303085,-1.679492,-9.890090],[3.666877,9.097390,3.256267],[-6.879682,7.745344,-4.185385],[-7.150359,-7.221071,3.325226],[-5.744160,-0.620633,-2.262117]],[[8.186773,8.912570,-6.789328],[-4.797640,4.275753,8.688962],[9.803428,6.039089,3.312564],[-3.950091,3.637104,-9.948451],[4.546796,-2.187078,-6.871775],[-2.252173,7.913952,6.407597],[-6.182418,-8.712926,-8.164866],[-7.469480,5.586935,-4.679356],[0.905036,-5.476941,8.156487],[3.851563,-5.845052,-9.298579]],[[-5.378429,-2.911772,-0.654480],[9.336637,5.909122,-6.366102],[4.886472,-3.262385,1.003826],[-6.755347,9.819789,3.165953],[0.724934,-8.303194,2.394379],[-0.315816,4.274991,8.023017],[-7.982408,-4.314392,-2.142366],[1.274299,8.549336,2.018772],[-6.653983,-5.391710,-7.005997],[2.427728,-5.012465,7.721918]],[[-7.038345,-6.239708,2.789023],[-3.505383,-1.863692,1.198050],[1.489547,6.958646,-8.600098],[6.648372,-8.298050,1.854520],[-4.440096,-9.438951,-8.465527],[-0.239845,3.158178,-2.377531],[-9.076143,-9.898536,-2.686665],[-1.387957,9.652772,-3.881315],[-6.760942,-7.872736,3.802990],[9.724313,-2.768951,0.102313]],[[-8.011767,-3.108554,-2.605492],[-2.204603,6.608759,0.229495],[-3.298055,6.544933,8.277472],[-0.449855,-2.391990,-7.069470],[-9.171475,4.068606,2.753053],[4.314804,-4.264069,3.702312],[-9.473821,9.716780,7.528849],[1.589118,-1.171658,3.584476],[7.976494,5.391547,-4.519393],[1.662884,-2.139804,-8.779955]],[[0.677363,2.518568,2.391263],[-1.338420,-5.341415,2.703612],[7.956271,9.810531,9.410189],[4.075500,-2.477644,-9.272341],[7.229130,8.003328,-4.748146],[5.011836,5.077987,2.417275],[-0.455259,2.177736,-8.070968],[-6.868205,2.704848,4.655204],[-5.896493,1.792525,-7.789372],[8.800544,-4.378478,3.420721]],[[-9.977344,-2.958601,6.119107],[-4.385159,-0.680020,-7.517479],[-5.977678,-9.920845,-5.930944],[6.628770,-7.036877,-1.099053],[-9.748051,-1.053627,-4.832183],[-1.423479,2.491434,2.986975],[-4.154249,-6.495359,6.113774],[9.549155,9.417117,1.466994],[-9.813150,-4.076482,-5.544031],[9.644506,-7.462687,3.471677]],[[8.906479,-9.259235,-7.735817],[4.944071,5.159621,2.587186],[-5.213010,1.144411,-9.682146],[2.219450,4.128509,-8.402162],[-4.600200,-8.557651,2.944433],[-8.329231,9.947233,0.401863],[-4.612320,2.817523,-6.439867],[-7.732407,1.024818,-4.982178],[-0.656900,-5.484236,7.063144],[7.479531,-2.793770,4.532448]],[[2.149453,-6.033400,-0.901946],[5.433428,-9.204905,-6.321841],[7.773762,1.396392,-0.509306],[6.753500,-2.760119,-6.174750],[-5.538347,1.564578,-7.771662],[-6.117651,-5.159365,-7.699421],[5.192119,2.847577,-2.152857],[-5.632723,-1.227832,0.553841],[-1.280169,-5.571179,-9.601973],[-8.691977,-5.768522,-1.205513]],[[-7.503124,7.125670,1.087400],[7.086269,3.180501,3.777262],[3.016867,6.052066,0.296343],[-0.129231,4.644026,-9.170624],[-5.004219,-2.993433,9.416441],[3.871533,-1.703475,-7.437700],[-7.963181,-2.493823,3.704788],[-5.357929,0.900299,-9.720211],[-7.410841,-5.829344,1.722387],[2.074134,-0.725173,3.247733]]], dtype = "float64")#candidate|375|(14, 10, 3)|const|float64
var_376 = relay.var("var_376", dtype = "float64", shape = (14, 10, 3))#candidate|376|(14, 10, 3)|var|float64
bop_377 = relay.less_equal(const_375.astype('bool'), relay.reshape(var_376.astype('bool'), relay.shape_of(const_375))) # shape=(14, 10, 3)
uop_380 = relay.sqrt(const_375.astype('float64')) # shape=(14, 10, 3)
uop_382 = relay.sinh(uop_380.astype('float64')) # shape=(14, 10, 3)
func_119_call = mod.get_global_var('func_119')
func_122_call = mutated_mod.get_global_var('func_122')
const_385 = relay.const([True,True,True,True,False,True,False,True,True,True,False,True,True,True,True,True,False,True,True,True,True,True,False,False,True,False,True,True,False,True,True,True,True,True,False,True,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,True,False,True,True,True,False,False,True,False,False,False,True,True,True,False,False,False,False,True,True,False,False,False,True,False,True,True,True,True,False,False,False,False,False,False,True,False,False,False,False,False,True,True,True,False,True,False,False,False,True,False,False,True,False,True,True,True,False,False,False,False,True,True,False,True,True,True,True,True,False,True,False,True,False,True,True,False,True,False,False,False,False,False,True,False,True,True,False,True,True,False,True,False,True,True,True,False,False,False,False,False,True,False,True,False,True,True,False,True,True,False,True,True,False,True,True,True,False,True,False,True,False,True,False,True,False,True,False,False,False,False,True,True,False,False,True,True,True,True,False,False,True,False,True,True,True,True,False,True,False,False,False,False,True,False,False,True,False,False,False,False,False,True,True,False,True,True,True,True,False,False,True,False,False,True,True,True,False,False,False,False,False,False,True,False,True,True,True,True,True,True,True,True,False,False,True,False,True,False,True,False,False,False,False,False,True,True,True,True,True,True,False,False,False,True,False,True,False,True,False,True,True,True,False,True,True,True,True,True,True,False,True,False,True,True,False,False,False,False,False,False,True,False,False,True,True,False,False,True,True,False,False,False,True,False,False,True,True,True,False,False,False,True,False,True,False,True,True,False,True,True,False,False,True,False,False,False,False,False,True,False,True,True,False,False,True,False,True,False,False,True,True,False,False,True,True,False,False,False,True,True,False,True,False,True,False,False,False,False,False,True,False,True,True,True,False,False,True,True,True,True,True,False,True,True,True,False,False,False,True,True,True,True,False,True,False,False,True,False,True,False,False,True,True,True,False,True,True,False,True,False,True,False,False,True,True,False,True,True,False,False,False,True,True,True,True,False,True,False,False,False,False,False,False,False,True,False,True,True,False,False,True,False,False,False,True,False,False,False,True,False,True,False,True,False,True,False,False,False,True,True,True,True,True,True,True,True,True,True,False,False,True,True,False,False,True,True,False,True,True,True,True,False,True,True,True,False,True,True,True,False,False,False,False,False,True,True,True,False,False,True,False,False,False,False,False,False,True,False,True,True,True,True,True,True,False,False,True,True,False,False,False,False,False,False,False,True,True,True,False,False,False,False,False,True,True,False,True,False,False,False,True,True,True,True,False,True,True,False,True,True,False,True,False,True,True,False,False,False,False,False,False,False,True,False,True,False,False,False,False,False,True,False,False,False,True,False,False,False,False,True,False,True,False,False,False,True,False,False,True,False,True,False,True,False,False,False,False,True,True,False,True,False,False,True,False,True,False,False,False,False,True,False,True,True,False,True,False,False,False,False,True,True,False,False,True,False,False,True,True,True,True,True,True,False,False,False,True,False,False,True,True,False,True,False,True,True,False,False,False,False,False,True,True,False,True,False,False,True,True,False,True,False,True,False,False,True,False,False,False,True,True,True,True,True,True,False,False,True,False,False,True,False,True,False,False,False,False,True,False,True,True,True,False,False,True,False,True,False,False,True,True,False,False,False,True,False,False,True,True,True,False,True,True,True,True,False,False,False,True,False,True,False,True,True,False,False,False,True,False,False,True,True,False,True,True,True,True,False,True,True,True,True,False,False,False,True,False,False,True,True,False,True,True,True,False,True,False,False,True,False,True,False,False,True,True,True,True,True,True,False,False,False,True,False,True,True,True,True,False,True,True,False,False,False,False,False,False,False,True,True,False,False,True,False,True,False,False,False,True,True,False,True,False,True,True,True,True,True,False,True,False,True,True,False,False,True,False,False,True,True,True,False,True,False,False,True,False,False,True,False,False,False,True,False,True,True,True,False,False,True,True,False,True,True,False,True,False,False,True,False,False,False,False,True,False,False,False,False,True,False,False,True,False,False,True,True,True,False,False,True,True,True,True,False,False,True,True,True,True,False,True,True,False,True,True,False,True,True,True,False,True,True,True,True,True,False,True,False,False,True,True,False,False,False,False,True,True,False,False,True,False,True,False,True,False,True,True,True,True,False,False,False,False,True,False,True,False,True,False,False,False,True,False,False,False,True,True,True,True,True,False,True,False,True,False,False,False,False,False,True,False,False,False,False,True,False,False,True,True,False,False,True,True,False,True,True,True,True,True,False,False,True,False,True,False,False,True,True,False,True,False,True,True,False,True,True,False,True,False,True,True,False,False,True,True,False,False,False,True,True,True,True,True,True,False,True,True,False,False,False,True,True,True,True,True,False,False,False,False,False,True,True,True,False,False,False,False,True,False,True,False,True,False,True,False,True,True,False,True,True,False,False,False,True,True,False,True,True,False,True,False,True,True,True,True,True,True,True,False,True,False,True,True,True,False,True,False,True,True,True,True,True,True,True,False,True,True,False,False,True,True,False,True,False,True,False,False,False,False,False,False,False,True,True,True,False,False,True,True,False,False,False,True,False,False,False,True,True,False,True,True,True,False,False,False,True,False,False,False,False,False,False,True,False,True,False,False,True,False,True,True,False,False,True,False,False,True,True,True,True,True,False,False,True,False,False,False,True,False,False,True,True,False,False,True,True,True,True,True,True,False,False,True,False,True,False,False,False,True,False,True,False,False,False,True,True,False,False,False,True,False,True,False,True,False,True,False,True,False,True,False,False,False,True,True,True,True,True,False,True,True,True,True,False,True,True,True,False,True,True,True,True,False,True,False,False,False,True,True,False,False,True,False,False,True,True,False,False,True,False,True,False,True,True,False,True,False,True,True,True,False,False,False,True,True,True,True,True,False,True,True,False,False,True,True,True,True,True,False,False,True,True,False,False,False,True,True,False,False,False,True,True,False,False,False,True,True,True,False,True,False,True,True,True,True,False,False,False,True,False,False,False,False,False,True,False,True,False,True,False,True,False,True,True], dtype = "bool")#candidate|385|(1296,)|const|bool
call_384 = func_119_call(relay.reshape(const_385.astype('bool'), [9, 16, 9]))
call_386 = func_119_call(relay.reshape(const_385.astype('bool'), [9, 16, 9]))
output = relay.Tuple([bop_377,uop_382,call_384,const_385,])
output2 = relay.Tuple([bop_377,uop_382,call_386,const_385,])
func_387 = relay.Function([var_376,], output)
mod['func_387'] = func_387
mod = relay.transform.InferType()(mod)
var_388 = relay.var("var_388", dtype = "float64", shape = (14, 10, 3))#candidate|388|(14, 10, 3)|var|float64
output = func_387(var_388)
func_389 = relay.Function([var_388], output)
mutated_mod['func_389'] = func_389
mutated_mod = relay.transform.InferType()(mutated_mod)
var_391 = relay.var("var_391", dtype = "int8", shape = (13,))#candidate|391|(13,)|var|int8
var_392 = relay.var("var_392", dtype = "int8", shape = (13,))#candidate|392|(13,)|var|int8
bop_393 = relay.bitwise_xor(var_391.astype('int8'), relay.reshape(var_392.astype('int8'), relay.shape_of(var_391))) # shape=(13,)
uop_396 = relay.sigmoid(var_392.astype('float64')) # shape=(13,)
output = relay.Tuple([bop_393,uop_396,])
output2 = relay.Tuple([bop_393,uop_396,])
func_398 = relay.Function([var_391,var_392,], output)
mod['func_398'] = func_398
mod = relay.transform.InferType()(mod)
mutated_mod['func_398'] = func_398
mutated_mod = relay.transform.InferType()(mutated_mod)
func_398_call = mutated_mod.get_global_var('func_398')
var_400 = relay.var("var_400", dtype = "int8", shape = (13,))#candidate|400|(13,)|var|int8
var_401 = relay.var("var_401", dtype = "int8", shape = (13,))#candidate|401|(13,)|var|int8
call_399 = func_398_call(var_400,var_401,)
output = call_399
func_402 = relay.Function([var_400,var_401,], output)
mutated_mod['func_402'] = func_402
mutated_mod = relay.transform.InferType()(mutated_mod)
var_404 = relay.var("var_404", dtype = "float64", shape = ())#candidate|404|()|var|float64
uop_405 = relay.log(var_404.astype('float64')) # shape=()
output = uop_405
output2 = uop_405
F = relay.Function([var_404,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_404,], output2)
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
	relay.transform.DefuseOps(),
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
input_404= np.array(-3.233692, dtype='float64')
module1.set_input('var_404', input_404)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_404, )
res3 = intrp3.evaluate()(input_404, )
res4 = intrp4.evaluate()(input_404, )
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
module5.set_input('var_404', input_404)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_404, )
res7 = intrp7.evaluate()(input_404, )
res8 = intrp8.evaluate()(input_404, )
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
module9.set_input('var_404', input_404)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_404, )
res11 = intrp11.evaluate()(input_404, )
res12 = intrp12.evaluate()(input_404, )
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
module13.set_input('var_404', input_404)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_404, )
res15 = intrp15.evaluate()(input_404, )
res16 = intrp16.evaluate()(input_404, )
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
module17.set_input('var_404', input_404)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_404, )
res19 = intrp19.evaluate()(input_404, )
res20 = intrp20.evaluate()(input_404, )
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
module21.set_input('var_404', input_404)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_404, )
res23 = intrp23.evaluate()(input_404, )
res24 = intrp24.evaluate()(input_404, )
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