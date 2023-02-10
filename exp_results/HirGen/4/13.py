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
var_0 = relay.var("var_0", dtype = "float64", shape = (4, 11))#candidate|0|(4, 11)|var|float64
uop_1 = relay.sin(var_0.astype('float64')) # shape=(4, 11)
uop_3 = relay.sqrt(uop_1.astype('float64')) # shape=(4, 11)
uop_5 = relay.asinh(uop_3.astype('float32')) # shape=(4, 11)
uop_7 = relay.log10(uop_3.astype('float32')) # shape=(4, 11)
uop_9 = relay.tan(uop_3.astype('float64')) # shape=(4, 11)
output = relay.Tuple([uop_5,uop_7,uop_9,])
output2 = relay.Tuple([uop_5,uop_7,uop_9,])
func_11 = relay.Function([var_0,], output)
mod['func_11'] = func_11
mod = relay.transform.InferType()(mod)
var_12 = relay.var("var_12", dtype = "float64", shape = (4, 11))#candidate|12|(4, 11)|var|float64
output = func_11(var_12)
func_13 = relay.Function([var_12], output)
mutated_mod['func_13'] = func_13
mutated_mod = relay.transform.InferType()(mutated_mod)
var_15 = relay.var("var_15", dtype = "float32", shape = (10, 7, 16))#candidate|15|(10, 7, 16)|var|float32
uop_16 = relay.cos(var_15.astype('float32')) # shape=(10, 7, 16)
var_18 = relay.var("var_18", dtype = "float32", shape = (10, 7, 16))#candidate|18|(10, 7, 16)|var|float32
bop_19 = relay.add(var_15.astype('int8'), relay.reshape(var_18.astype('int8'), relay.shape_of(var_15))) # shape=(10, 7, 16)
uop_22 = relay.cos(var_15.astype('float32')) # shape=(10, 7, 16)
uop_24 = relay.asin(uop_22.astype('float32')) # shape=(10, 7, 16)
func_11_call = mod.get_global_var('func_11')
func_13_call = mutated_mod.get_global_var('func_13')
var_27 = relay.var("var_27", dtype = "float64", shape = (44,))#candidate|27|(44,)|var|float64
call_26 = relay.TupleGetItem(func_11_call(relay.reshape(var_27.astype('float64'), [4, 11])), 1)
call_28 = relay.TupleGetItem(func_13_call(relay.reshape(var_27.astype('float64'), [4, 11])), 1)
output = relay.Tuple([uop_16,bop_19,uop_24,call_26,var_27,])
output2 = relay.Tuple([uop_16,bop_19,uop_24,call_28,var_27,])
func_29 = relay.Function([var_15,var_18,var_27,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
mutated_mod['func_29'] = func_29
mutated_mod = relay.transform.InferType()(mutated_mod)
func_29_call = mutated_mod.get_global_var('func_29')
var_31 = relay.var("var_31", dtype = "float32", shape = (10, 7, 16))#candidate|31|(10, 7, 16)|var|float32
var_32 = relay.var("var_32", dtype = "float32", shape = (10, 7, 16))#candidate|32|(10, 7, 16)|var|float32
var_33 = relay.var("var_33", dtype = "float64", shape = (44,))#candidate|33|(44,)|var|float64
call_30 = func_29_call(var_31,var_32,var_33,)
output = call_30
func_34 = relay.Function([var_31,var_32,var_33,], output)
mutated_mod['func_34'] = func_34
mutated_mod = relay.transform.InferType()(mutated_mod)
var_36 = relay.var("var_36", dtype = "float32", shape = (4, 14, 11))#candidate|36|(4, 14, 11)|var|float32
uop_37 = relay.log10(var_36.astype('float32')) # shape=(4, 14, 11)
bop_39 = relay.bitwise_xor(uop_37.astype('uint64'), relay.reshape(var_36.astype('uint64'), relay.shape_of(uop_37))) # shape=(4, 14, 11)
func_11_call = mod.get_global_var('func_11')
func_13_call = mutated_mod.get_global_var('func_13')
var_43 = relay.var("var_43", dtype = "float64", shape = (44,))#candidate|43|(44,)|var|float64
call_42 = relay.TupleGetItem(func_11_call(relay.reshape(var_43.astype('float64'), [4, 11])), 0)
call_44 = relay.TupleGetItem(func_13_call(relay.reshape(var_43.astype('float64'), [4, 11])), 0)
bop_45 = relay.bitwise_xor(uop_37.astype('int8'), relay.reshape(var_36.astype('int8'), relay.shape_of(uop_37))) # shape=(4, 14, 11)
uop_48 = relay.acosh(uop_37.astype('float32')) # shape=(4, 14, 11)
var_50 = relay.var("var_50", dtype = "float32", shape = (4, 14, 11))#candidate|50|(4, 14, 11)|var|float32
bop_51 = relay.greater(uop_48.astype('bool'), relay.reshape(var_50.astype('bool'), relay.shape_of(uop_48))) # shape=(4, 14, 11)
uop_54 = relay.tan(bop_45.astype('float32')) # shape=(4, 14, 11)
uop_56 = relay.erf(uop_54.astype('float32')) # shape=(4, 14, 11)
uop_58 = relay.atan(uop_56.astype('float32')) # shape=(4, 14, 11)
func_29_call = mod.get_global_var('func_29')
func_34_call = mutated_mod.get_global_var('func_34')
var_61 = relay.var("var_61", dtype = "float32", shape = (1120,))#candidate|61|(1120,)|var|float32
call_60 = relay.TupleGetItem(func_29_call(relay.reshape(var_61.astype('float32'), [10, 7, 16]), relay.reshape(var_61.astype('float32'), [10, 7, 16]), relay.reshape(var_43.astype('float64'), [44,]), ), 3)
call_62 = relay.TupleGetItem(func_34_call(relay.reshape(var_61.astype('float32'), [10, 7, 16]), relay.reshape(var_61.astype('float32'), [10, 7, 16]), relay.reshape(var_43.astype('float64'), [44,]), ), 3)
output = relay.Tuple([bop_39,call_42,var_43,bop_51,uop_58,call_60,var_61,])
output2 = relay.Tuple([bop_39,call_44,var_43,bop_51,uop_58,call_62,var_61,])
func_63 = relay.Function([var_36,var_43,var_50,var_61,], output)
mod['func_63'] = func_63
mod = relay.transform.InferType()(mod)
var_64 = relay.var("var_64", dtype = "float32", shape = (4, 14, 11))#candidate|64|(4, 14, 11)|var|float32
var_65 = relay.var("var_65", dtype = "float64", shape = (44,))#candidate|65|(44,)|var|float64
var_66 = relay.var("var_66", dtype = "float32", shape = (4, 14, 11))#candidate|66|(4, 14, 11)|var|float32
var_67 = relay.var("var_67", dtype = "float32", shape = (1120,))#candidate|67|(1120,)|var|float32
output = func_63(var_64,var_65,var_66,var_67,)
func_68 = relay.Function([var_64,var_65,var_66,var_67,], output)
mutated_mod['func_68'] = func_68
mutated_mod = relay.transform.InferType()(mutated_mod)
var_70 = relay.var("var_70", dtype = "int16", shape = (1, 5, 6))#candidate|70|(1, 5, 6)|var|int16
var_71 = relay.var("var_71", dtype = "int16", shape = (14, 5, 6))#candidate|71|(14, 5, 6)|var|int16
bop_72 = relay.bitwise_and(var_70.astype('int16'), var_71.astype('int16')) # shape=(14, 5, 6)
bop_75 = relay.greater(var_70.astype('bool'), bop_72.astype('bool')) # shape=(14, 5, 6)
func_29_call = mod.get_global_var('func_29')
func_34_call = mutated_mod.get_global_var('func_34')
const_79 = relay.const([-2.911221,-0.985677,-8.906926,7.808875,-5.969990,8.736631,-2.733108,-3.299985,-7.839816,-6.698586,-1.487017,4.749083,-8.609080,0.587195,4.810218,-8.888290,-1.557258,-9.419517,-8.541767,-1.506301,7.494750,4.987091,-7.438498,-6.543219,-4.022584,-0.412886,-7.606941,-7.861303,4.868122,-1.017433,0.084402,-9.877985,-9.945143,-6.818580,0.888655,0.173527,-7.062389,-6.836951,-2.796229,4.529870,7.938835,9.042145,-9.179370,2.110094,6.540863,-4.552605,-2.304929,6.649375,2.624097,2.828272,-4.245054,3.827486,8.125440,0.256272,-8.200722,-9.263407,-4.626136,6.001884,-0.215952,-7.156763,8.833281,-2.980993,-2.428418,-1.885261,3.704340,-7.359993,-8.085880,-3.006795,1.986254,6.750698,-2.036952,8.512685,2.235141,9.983434,0.168977,-4.982026,-9.052679,7.094543,-9.609339,-3.582863,5.475801,2.158783,1.922472,-4.187127,0.630915,-1.590670,4.858493,9.984541,1.497806,-4.538738,-6.446134,6.726272,3.748818,7.227729,-4.991308,9.196342,-8.210278,4.161764,-2.257535,-4.668831,6.822722,-4.110062,6.243877,3.627133,-2.980739,5.351408,-6.584255,-9.078320,0.135945,2.120533,2.044391,-9.580975,0.047453,6.759719,8.506158,-1.759988,4.322163,8.502399,-9.525828,4.298991,9.578641,1.368341,7.560748,7.176089,-9.942827,-4.316396,-4.852823,6.626245,-7.244805,2.381059,4.550908,-8.566382,-5.882550,0.543795,-3.460754,-3.418505,9.587534,-6.189559,0.616988,-3.566295,-4.019991,3.476045,9.478298,4.191422,-6.414461,3.338673,-6.147413,-4.840512,1.460494,4.207149,-1.302908,0.316791,4.173158,-1.249207,-9.997103,-7.005135,7.234203,6.506020,5.632407,-9.178461,6.345090,9.326288,7.461069,0.238071,8.636410,-4.482133,-6.647748,3.291206,4.460390,-6.484885,-2.046021,8.493214,-0.269932,-9.763057,6.885267,2.363323,-0.195219,-6.250463,-0.473287,1.515714,-8.372171,8.836514,0.167567,-5.467958,-5.926137,-4.156899,-0.740352,-5.519228,-1.702311,-7.202434,2.303439,3.368954,-8.771939,5.708794,5.549694,-1.561244,7.953484,-6.626517,-6.511860,2.085351,1.972554,-9.080926,4.743381,-9.166737,9.500314,8.270288,7.717151,2.159509,-8.778187,-4.113464,3.749420,6.998183,8.448062,5.127910,3.856104,-5.313702,0.853390,-1.669803,-1.834059,-8.682722,7.440731,-2.515983,-8.816398,-9.373549,9.719293,-4.242167,5.682264,9.151758,7.032503,-9.388079,8.169316,4.074212,6.880267,-7.144877,4.518498,-6.217986,-1.714564,5.516623,-1.390414,-0.082603,-8.718095,-2.639227,-8.062782,-2.144969,-4.601997,-1.277510,-5.184330,-6.474503,6.295921,2.666743,-1.984224,1.195235,-7.566636,7.292733,-2.643219,-2.344106,3.420730,-2.386818,-0.574830,7.234643,-2.571908,9.344401,-6.377194,-8.008649,-2.356603,-8.960821,6.612259,9.043985,-1.553856,-9.422841,-3.274970,9.517034,6.952137,-6.597035,8.953029,2.376545,3.424242,-6.741768,-9.090176,-8.214347,-0.157725,9.894535,-0.080530,1.564512,-9.729683,-2.019308,5.869381,-5.030899,8.919882,-0.482989,-0.160944,-4.204062,-2.421630,8.702418,-1.265639,6.201638,-6.521032,-5.610710,-5.953249,1.785852,-3.776724,5.587985,-5.858861,-5.072995,-5.681980,-0.898322,-8.166992,7.829297,5.205327,-3.303591,-7.045842,-1.896236,-3.933941,1.351966,-2.428176,8.667830,-7.929545,-4.973670,-1.023298,3.914120,-4.441175,-0.507556,-1.519868,8.488014,-1.645398,5.850129,-4.290073,-8.264570,4.599949,7.007812,-7.254902,-9.533697,4.788943,6.003966,8.404000,2.681998,-6.702465,1.258765,3.047554,-4.542400,-5.017076,-6.782026,2.222216,5.190703,-9.484069,8.017689,-3.726891,-7.706237,-7.036089,2.041427,-6.170567,6.113092,-9.456882,4.625528,-8.480610,-7.400218,-9.466328,9.197160,9.343392,-3.344550,3.000185,-8.919761,1.154240,0.749052,4.892361,9.936727,3.083887,-4.963695,9.712441,4.212090,-3.545334,-4.585360,-3.737692,-6.892804,5.806589,-1.190640,-0.476247,-2.535321,4.802542,-9.732229,-7.977550,-1.471594,-6.397939,-0.627037,-6.263005,3.106767,2.060390,0.314581,0.633281,2.673123,-3.193734,1.591501,-5.672405,-1.209434,-9.963620,-2.710929,6.335461,-5.236201,-1.363984,3.664828,0.532648,3.560458,1.180715,-9.643992,-5.348310,7.023536,1.954675,-4.566081,-2.762316,1.136738,6.437663,5.074554,-2.819506,-5.533505,-4.173948,-3.808347,2.505190,-8.902860,-6.559269,5.730153,-0.847628,-9.958176,-3.966966,-6.255467,6.725467,5.039596,0.736042,0.678002,-5.364250,-5.167725,-9.454309,-5.456404,-6.582081,-2.410040,-6.741535,-5.783826,-8.509418,-5.479863,6.207884,2.603106,3.563817,9.297574,9.379606,-5.590893,-4.583674,9.107262,2.099147,-2.214699,-1.191033,6.414401,4.882956,-2.422472,-9.392955,-3.197186,3.608269,-6.841012,-2.258394,1.403410,7.741916,2.758166,8.978709,-6.296721,1.880812,-1.974765,9.780814,-6.182424,4.436791,-3.030650,6.103529,4.196507,-2.680378,0.094063,3.612417,-2.563050,1.327671,4.572072,-0.572036,7.365423,-9.846806,-4.052814,2.850222,-3.575243,3.952509,-8.290918,6.974477,9.020048,-5.320112,-9.461438,4.877020,-9.820856,0.439140,6.321421,3.412722,5.750651,8.107843,8.163102,-0.685435,8.774226,-4.422764,-3.692712,-7.604236,6.062271,7.479000,-8.956537,-2.002096,6.471459,-7.485046,-9.428564,8.121669,-3.903251,1.388948,-5.982885,5.558059,6.216857,0.884110,-4.467399,3.859942,-1.717278,5.074484,2.833719,8.966401,2.161806,-8.453665,2.316207,8.270430,-4.392717,2.631807,-3.263058,-4.169549,-8.936961,-1.009138,8.810981,-6.675257,-6.274453,7.559547,-0.013630,-0.713712,-0.740212,2.459257,-9.995614,2.628926,5.013463,8.837772,-9.501014,6.662596,-0.712439,0.231357,9.833665,1.024081,2.442539,8.691960,-4.821715,-2.323459,-4.300877,-5.158009,-2.560046,8.441256,8.250336,-6.033786,1.324087,-6.910495,1.950241,1.326731,-2.963391,-6.227347,9.209919,9.072404,2.282546,2.516173,2.134359,-4.910715,2.961930,3.252765,2.277390,-7.547682,1.325781,-7.083138,-9.048073,0.375644,-4.701782,3.509710,0.973433,2.609436,-6.764553,5.739219,6.372236,8.044911,4.790008,1.828691,-5.496327,0.283593,-7.221085,-0.556585,-9.021825,0.482287,7.635678,-0.639293,8.462045,8.734374,-7.711316,4.757121,0.486707,0.267415,6.045615,3.103886,-8.011711,3.678847,-9.142892,-1.503471,9.244097,4.439534,-1.385075,-1.905602,-7.005256,9.369758,-3.863636,-8.793697,5.172488,-8.268028,-9.649935,1.941992,4.633718,9.564056,-4.838128,-5.226594,-1.896732,3.606673,1.759882,-6.553698,7.922879,7.438422,2.621649,1.574088,3.433141,5.702789,8.311763,-3.987451,-8.970661,-0.194475,0.976985,-7.103110,7.770532,-3.238773,8.342931,-9.108767,-1.149347,-6.948866,2.320573,4.732350,-0.199676,-4.110521,9.649781,-3.049674,-6.028585,-4.413412,3.959222,9.483672,-4.087680,0.379584,3.204453,4.654562,3.797475,0.530494,8.350579,0.702896,4.174502,-3.330901,1.046336,-4.785958,1.634475,3.167452,-5.989427,4.291867,3.834767,-6.724900,-9.913809,-0.977475,-5.695082,-1.938465,-9.522576,4.515118,6.954158,-5.741833,-4.297960,1.737862,-4.284787,-5.594539,4.720894,8.412839,-2.255460,4.778129,1.683973,5.010581,1.556151,-4.110659,0.546599,-7.091533,9.821105,6.221809,0.320344,1.722047,-5.463353,-6.779263,-2.815516,-2.713288,-5.142164,9.072300,9.611173,9.257473,2.176604,7.789010,2.041262,0.453656,-2.220628,-5.865196,9.568104,8.888665,4.307902,-1.379685,-4.662323,-2.936329,-6.913409,-7.795374,-4.274453,3.037457,2.086912,7.370146,-8.504869,1.036129,9.524298,-2.215397,-3.520709,-6.771319,-2.294339,1.433416,6.069209,9.250419,6.011748,-9.462917,7.119038,4.356208,-6.878853,1.696776,-6.784233,9.732474,-4.682524,-2.806322,-7.874913,-9.763002,5.321876,1.467580,1.132801,2.139359,-6.722853,9.604648,2.000589,6.944004,6.951592,-5.856492,6.049526,8.490297,0.905220,-2.401885,3.739601,-3.091126,-3.498477,5.157907,-3.443192,3.010399,-7.389453,-6.705065,9.962948,9.136289,-0.709097,-8.272678,-8.125716,2.817734,-7.540749,3.330792,-7.866957,-2.702956,4.197166,8.873191,-5.491589,-6.366106,6.553365,-6.027078,-2.483284,6.757871,5.585663,1.330173,6.233587,-4.057652,-7.591305,0.300657,1.156244,-8.902776,-6.167670,8.047586,-1.097297,0.007994,-6.776976,-0.439605,-9.232131,-7.591341,-6.549736,3.198503,-2.808127,9.243199,7.214203,-8.978940,-8.272817,6.907124,-3.956542,-6.978508,-2.254588,-5.819804,9.111226,-4.299967,-2.088764,-2.882013,-5.787971,-2.938271,9.355902,3.373323,1.404971,2.355787,-7.287448,-1.796752,5.108380,8.204676,-6.415107,6.889023,2.758179,0.943162,-6.967710,-9.880363,3.305642,6.040338,-1.892903,8.186625,-8.359592,5.654274,6.412920,2.249627,-7.810303,9.931438,-8.250564,6.146311,-1.106790,8.157907,-5.023520,3.481623,6.017331,-8.703753,8.353106,8.345549,-9.018849,0.189665,-8.100502,9.477234,1.778450,5.264985,9.524780,5.153987,-9.112815,-7.223200,-7.558544,-1.732177,-4.415528,6.799338,0.792284,-8.837674,-7.636052,0.542794,-4.957668,6.620326,3.173681,-3.047094,8.719586,-1.687686,-1.028322,-1.597104,-4.467654,-0.942676,4.967460,7.450344,-8.469498,9.612182,9.538408,6.382367,7.943187,6.455750,-1.096748,6.071275,-8.283037,2.635182,-6.498364,7.773905,-3.552243,8.148816,-8.162525,-4.144256,-8.312372,2.174148,9.324008,-8.944711,8.391251,5.753087,8.901850,-1.399581,5.373677,-4.814657,4.329674,-1.382075,-5.506034,-3.213376,-7.800137,0.055372,-2.752236,-8.136779,-8.390042,-0.833220,5.397228,7.055421,-3.835687,-7.631926,1.619750,0.392960,-1.520893,-9.887459,5.074866,7.446902,-6.886454,9.766803,1.853148,-5.719462,-2.618656,-1.265298,-9.742071,-1.132662,8.294398,-5.165246,1.618640,6.548376,-4.135206,-3.908680,-8.613519,2.967298,1.626094,1.470526,8.554609,-9.236977,-2.389329,6.946069,-0.449247,1.805527,-4.342990,-5.507369,6.021780,2.574930,4.625084,-5.851553,-3.406426,-9.932908,-2.112258,-3.759185,7.699330,5.277634,5.010852,1.045485,-7.569193,6.570159,-9.825184,-9.380201,-8.989507,5.971384,-5.626761,2.759615,-8.589974,-4.443842,-2.979690,-4.855838,3.640502,2.126574,-3.812290,-0.038710,-4.933119,1.313893,2.824950,-3.154340,0.971764,3.568409,2.198712,-8.746884,6.963743,-0.343894,-4.593813,9.464737,-6.649078,-3.810806,1.632290,9.409573,2.063682,0.600104,2.934866,-0.635226,6.550431,-1.009918,-6.655117,-3.819531,0.417281,7.054270,1.426535,3.509098,-7.573339,-7.975074,-0.041138,1.178553,-2.695101,-8.297425,-9.452938,7.098138,-6.454300,5.387005,-9.237502,-0.751210,-1.287341,6.062987,5.281663,-2.459570,9.120636,5.120529,-1.009607,7.905801,3.782120,-2.859682,-4.455876,-7.025224,1.102542,2.172208,7.496682,5.055086,-5.459206,-8.787510,4.227775,5.432896,7.347470,-2.825291,7.972580,7.220291,7.032562,-6.566004,9.600672,1.008428,-7.939974,-3.459539,-0.626115,-3.970412,8.571969,-7.709831,0.354800,4.433475,5.399667,-1.517454,4.056987,3.321525,-7.292345,-9.956243,-0.427476,-4.179138,-5.207933,8.848902,1.044669,6.833378,-3.349230,1.541551,0.672077,-9.638414,1.725251,2.759519,-9.030362,4.805671,2.036626,-0.158036,8.768773,-7.345879,-3.059966,9.316712,4.683903,-0.350619,-1.205362,1.921431,-9.707761,-1.126957,8.030346,-0.218917,4.718481,-5.035672,6.384242,-1.168769,6.476901,-4.316629,-1.456259,1.223725,2.182389,0.771071,-9.992218,7.033953,8.089665,-6.889918,-8.867301,-3.466145,-1.993771,-8.246672,-7.547396,8.560898,0.625110,6.786597,7.781437], dtype = "float32")#candidate|79|(1120,)|const|float32
var_80 = relay.var("var_80", dtype = "float64", shape = (44,))#candidate|80|(44,)|var|float64
call_78 = relay.TupleGetItem(func_29_call(relay.reshape(const_79.astype('float32'), [10, 7, 16]), relay.reshape(const_79.astype('float32'), [10, 7, 16]), relay.reshape(var_80.astype('float64'), [44,]), ), 1)
call_81 = relay.TupleGetItem(func_34_call(relay.reshape(const_79.astype('float32'), [10, 7, 16]), relay.reshape(const_79.astype('float32'), [10, 7, 16]), relay.reshape(var_80.astype('float64'), [44,]), ), 1)
func_63_call = mod.get_global_var('func_63')
func_68_call = mutated_mod.get_global_var('func_68')
var_83 = relay.var("var_83", dtype = "float32", shape = (616,))#candidate|83|(616,)|var|float32
call_82 = relay.TupleGetItem(func_63_call(relay.reshape(var_83.astype('float32'), [4, 14, 11]), relay.reshape(var_80.astype('float64'), [44,]), relay.reshape(var_83.astype('float32'), [4, 14, 11]), relay.reshape(call_78.astype('float32'), [1120,]), ), 4)
call_84 = relay.TupleGetItem(func_68_call(relay.reshape(var_83.astype('float32'), [4, 14, 11]), relay.reshape(var_80.astype('float64'), [44,]), relay.reshape(var_83.astype('float32'), [4, 14, 11]), relay.reshape(call_78.astype('float32'), [1120,]), ), 4)
bop_85 = relay.floor_divide(const_79.astype('float32'), relay.reshape(call_78.astype('float32'), relay.shape_of(const_79))) # shape=(1120,)
bop_88 = relay.floor_divide(const_79.astype('float32'), relay.reshape(call_81.astype('float32'), relay.shape_of(const_79))) # shape=(1120,)
bop_89 = relay.mod(var_83.astype('float32'), relay.reshape(call_82.astype('float32'), relay.shape_of(var_83))) # shape=(616,)
bop_92 = relay.mod(var_83.astype('float32'), relay.reshape(call_84.astype('float32'), relay.shape_of(var_83))) # shape=(616,)
const_93 = relay.const([[[5,5,-2,6,1,-2],[-9,-9,-10,7,5,-7],[-6,9,-5,6,-9,-5],[-7,5,-10,-6,1,-1],[2,-10,-3,-2,-7,7]],[[-10,-3,5,9,8,2],[10,-7,3,-2,8,-2],[9,2,-4,8,-3,2],[9,-9,-5,5,9,6],[1,-7,-2,-10,2,8]],[[2,-10,9,2,-3,-6],[-2,-4,2,2,-10,2],[7,-4,3,-5,-6,-9],[-8,4,2,10,1,-6],[5,-8,-7,6,10,3]],[[10,8,4,4,8,3],[-5,5,-6,1,-9,8],[-7,1,4,3,-3,-3],[-10,-9,4,8,7,-5],[4,-6,-5,10,9,-1]]], dtype = "int16")#candidate|93|(4, 5, 6)|const|int16
bop_94 = relay.floor_mod(var_70.astype('float32'), const_93.astype('float32')) # shape=(4, 5, 6)
uop_97 = relay.acos(bop_72.astype('float64')) # shape=(14, 5, 6)
var_99 = relay.var("var_99", dtype = "float64", shape = (14, 5, 6))#candidate|99|(14, 5, 6)|var|float64
bop_100 = relay.minimum(uop_97.astype('uint16'), relay.reshape(var_99.astype('uint16'), relay.shape_of(uop_97))) # shape=(14, 5, 6)
var_103 = relay.var("var_103", dtype = "float64", shape = (14, 5, 6))#candidate|103|(14, 5, 6)|var|float64
bop_104 = relay.subtract(uop_97.astype('int32'), relay.reshape(var_103.astype('int32'), relay.shape_of(uop_97))) # shape=(14, 5, 6)
bop_107 = relay.logical_or(bop_104.astype('bool'), relay.reshape(bop_75.astype('bool'), relay.shape_of(bop_104))) # shape=(14, 5, 6)
bop_110 = relay.power(var_71.astype('float64'), var_70.astype('float64')) # shape=(14, 5, 6)
uop_113 = relay.acos(uop_97.astype('float64')) # shape=(14, 5, 6)
uop_115 = relay.sinh(uop_113.astype('float32')) # shape=(14, 5, 6)
uop_117 = relay.sqrt(bop_104.astype('float64')) # shape=(14, 5, 6)
output = relay.Tuple([var_80,bop_85,bop_89,bop_94,bop_100,bop_107,bop_110,uop_115,uop_117,])
output2 = relay.Tuple([var_80,bop_88,bop_92,bop_94,bop_100,bop_107,bop_110,uop_115,uop_117,])
func_119 = relay.Function([var_70,var_71,var_80,var_83,var_99,var_103,], output)
mod['func_119'] = func_119
mod = relay.transform.InferType()(mod)
var_120 = relay.var("var_120", dtype = "int16", shape = (1, 5, 6))#candidate|120|(1, 5, 6)|var|int16
var_121 = relay.var("var_121", dtype = "int16", shape = (14, 5, 6))#candidate|121|(14, 5, 6)|var|int16
var_122 = relay.var("var_122", dtype = "float64", shape = (44,))#candidate|122|(44,)|var|float64
var_123 = relay.var("var_123", dtype = "float32", shape = (616,))#candidate|123|(616,)|var|float32
var_124 = relay.var("var_124", dtype = "float64", shape = (14, 5, 6))#candidate|124|(14, 5, 6)|var|float64
var_125 = relay.var("var_125", dtype = "float64", shape = (14, 5, 6))#candidate|125|(14, 5, 6)|var|float64
output = func_119(var_120,var_121,var_122,var_123,var_124,var_125,)
func_126 = relay.Function([var_120,var_121,var_122,var_123,var_124,var_125,], output)
mutated_mod['func_126'] = func_126
mutated_mod = relay.transform.InferType()(mutated_mod)
var_128 = relay.var("var_128", dtype = "float32", shape = (10, 5))#candidate|128|(10, 5)|var|float32
uop_129 = relay.atanh(var_128.astype('float32')) # shape=(10, 5)
uop_131 = relay.acos(uop_129.astype('float32')) # shape=(10, 5)
uop_133 = relay.asinh(uop_129.astype('float32')) # shape=(10, 5)
bop_135 = relay.right_shift(uop_133.astype('uint64'), relay.reshape(uop_131.astype('uint64'), relay.shape_of(uop_133))) # shape=(10, 5)
uop_138 = relay.atanh(var_128.astype('float32')) # shape=(10, 5)
bop_140 = relay.floor_mod(uop_138.astype('float32'), relay.reshape(var_128.astype('float32'), relay.shape_of(uop_138))) # shape=(10, 5)
var_143 = relay.var("var_143", dtype = "float32", shape = (10, 5))#candidate|143|(10, 5)|var|float32
bop_144 = relay.logical_or(uop_133.astype('bool'), relay.reshape(var_143.astype('bool'), relay.shape_of(uop_133))) # shape=(10, 5)
var_147 = relay.var("var_147", dtype = "bool", shape = (10, 5))#candidate|147|(10, 5)|var|bool
bop_148 = relay.bitwise_and(bop_144.astype('int8'), relay.reshape(var_147.astype('int8'), relay.shape_of(bop_144))) # shape=(10, 5)
uop_151 = relay.log2(uop_138.astype('float32')) # shape=(10, 5)
output = relay.Tuple([bop_135,bop_140,bop_148,uop_151,])
output2 = relay.Tuple([bop_135,bop_140,bop_148,uop_151,])
func_153 = relay.Function([var_128,var_143,var_147,], output)
mod['func_153'] = func_153
mod = relay.transform.InferType()(mod)
mutated_mod['func_153'] = func_153
mutated_mod = relay.transform.InferType()(mutated_mod)
func_153_call = mutated_mod.get_global_var('func_153')
var_155 = relay.var("var_155", dtype = "float32", shape = (10, 5))#candidate|155|(10, 5)|var|float32
var_156 = relay.var("var_156", dtype = "float32", shape = (10, 5))#candidate|156|(10, 5)|var|float32
var_157 = relay.var("var_157", dtype = "bool", shape = (10, 5))#candidate|157|(10, 5)|var|bool
call_154 = func_153_call(var_155,var_156,var_157,)
output = call_154
func_158 = relay.Function([var_155,var_156,var_157,], output)
mutated_mod['func_158'] = func_158
mutated_mod = relay.transform.InferType()(mutated_mod)
var_160 = relay.var("var_160", dtype = "float64", shape = ())#candidate|160|()|var|float64
uop_161 = relay.log(var_160.astype('float64')) # shape=()
var_163 = relay.var("var_163", dtype = "float64", shape = ())#candidate|163|()|var|float64
bop_164 = relay.bitwise_or(uop_161.astype('uint32'), var_163.astype('uint32')) # shape=()
uop_167 = relay.log(bop_164.astype('float32')) # shape=()
uop_169 = relay.sqrt(uop_167.astype('float64')) # shape=()
bop_171 = relay.floor_mod(uop_169.astype('float64'), uop_167.astype('float64')) # shape=()
uop_174 = relay.sqrt(uop_169.astype('float32')) # shape=()
uop_176 = relay.cos(uop_167.astype('float64')) # shape=()
uop_178 = relay.sin(uop_161.astype('float32')) # shape=()
bop_180 = relay.right_shift(bop_171.astype('uint64'), uop_178.astype('uint64')) # shape=()
uop_183 = relay.log10(uop_174.astype('float64')) # shape=()
var_185 = relay.var("var_185", dtype = "float64", shape = ())#candidate|185|()|var|float64
bop_186 = relay.mod(uop_183.astype('float64'), var_185.astype('float64')) # shape=()
var_189 = relay.var("var_189", dtype = "float64", shape = (9,))#candidate|189|(9,)|var|float64
bop_190 = relay.greater(uop_183.astype('bool'), var_189.astype('bool')) # shape=(9,)
bop_193 = relay.greater(uop_183.astype('bool'), var_160.astype('bool')) # shape=()
uop_196 = relay.sqrt(uop_174.astype('float64')) # shape=()
const_198 = relay.const([5.561466,2.289035,-7.738166,-1.427386,-3.729385,-3.834928], dtype = "float32")#candidate|198|(6,)|const|float32
bop_199 = relay.logical_xor(uop_174.astype('uint16'), const_198.astype('uint16')) # shape=(6,)
var_202 = relay.var("var_202", dtype = "uint16", shape = (6,))#candidate|202|(6,)|var|uint16
bop_203 = relay.greater(bop_199.astype('bool'), relay.reshape(var_202.astype('bool'), relay.shape_of(bop_199))) # shape=(6,)
var_206 = relay.var("var_206", dtype = "bool", shape = (6,))#candidate|206|(6,)|var|bool
bop_207 = relay.multiply(bop_203.astype('uint8'), relay.reshape(var_206.astype('uint8'), relay.shape_of(bop_203))) # shape=(6,)
bop_210 = relay.floor_mod(uop_178.astype('float64'), var_206.astype('float64')) # shape=(6,)
uop_213 = relay.sin(uop_169.astype('float32')) # shape=()
output = relay.Tuple([uop_176,bop_180,bop_186,bop_190,bop_193,uop_196,bop_207,bop_210,uop_213,])
output2 = relay.Tuple([uop_176,bop_180,bop_186,bop_190,bop_193,uop_196,bop_207,bop_210,uop_213,])
func_215 = relay.Function([var_160,var_163,var_185,var_189,var_202,var_206,], output)
mod['func_215'] = func_215
mod = relay.transform.InferType()(mod)
var_216 = relay.var("var_216", dtype = "float64", shape = ())#candidate|216|()|var|float64
var_217 = relay.var("var_217", dtype = "float64", shape = ())#candidate|217|()|var|float64
var_218 = relay.var("var_218", dtype = "float64", shape = ())#candidate|218|()|var|float64
var_219 = relay.var("var_219", dtype = "float64", shape = (9,))#candidate|219|(9,)|var|float64
var_220 = relay.var("var_220", dtype = "uint16", shape = (6,))#candidate|220|(6,)|var|uint16
var_221 = relay.var("var_221", dtype = "bool", shape = (6,))#candidate|221|(6,)|var|bool
output = func_215(var_216,var_217,var_218,var_219,var_220,var_221,)
func_222 = relay.Function([var_216,var_217,var_218,var_219,var_220,var_221,], output)
mutated_mod['func_222'] = func_222
mutated_mod = relay.transform.InferType()(mutated_mod)
var_224 = relay.var("var_224", dtype = "uint64", shape = (13, 9, 6))#candidate|224|(13, 9, 6)|var|uint64
var_225 = relay.var("var_225", dtype = "uint64", shape = (13, 9, 6))#candidate|225|(13, 9, 6)|var|uint64
bop_226 = relay.less(var_224.astype('bool'), relay.reshape(var_225.astype('bool'), relay.shape_of(var_224))) # shape=(13, 9, 6)
uop_229 = relay.sinh(bop_226.astype('float64')) # shape=(13, 9, 6)
uop_231 = relay.exp(uop_229.astype('float32')) # shape=(13, 9, 6)
output = uop_231
output2 = uop_231
func_233 = relay.Function([var_224,var_225,], output)
mod['func_233'] = func_233
mod = relay.transform.InferType()(mod)
mutated_mod['func_233'] = func_233
mutated_mod = relay.transform.InferType()(mutated_mod)
func_233_call = mutated_mod.get_global_var('func_233')
var_235 = relay.var("var_235", dtype = "uint64", shape = (13, 9, 6))#candidate|235|(13, 9, 6)|var|uint64
var_236 = relay.var("var_236", dtype = "uint64", shape = (13, 9, 6))#candidate|236|(13, 9, 6)|var|uint64
call_234 = func_233_call(var_235,var_236,)
output = call_234
func_237 = relay.Function([var_235,var_236,], output)
mutated_mod['func_237'] = func_237
mutated_mod = relay.transform.InferType()(mutated_mod)
var_239 = relay.var("var_239", dtype = "float64", shape = (11, 12))#candidate|239|(11, 12)|var|float64
uop_240 = relay.log10(var_239.astype('float64')) # shape=(11, 12)
var_242 = relay.var("var_242", dtype = "float64", shape = (11, 12))#candidate|242|(11, 12)|var|float64
bop_243 = relay.bitwise_and(uop_240.astype('int32'), relay.reshape(var_242.astype('int32'), relay.shape_of(uop_240))) # shape=(11, 12)
bop_246 = relay.greater(var_242.astype('bool'), relay.reshape(uop_240.astype('bool'), relay.shape_of(var_242))) # shape=(11, 12)
bop_249 = relay.logical_xor(var_239.astype('int8'), relay.reshape(bop_246.astype('int8'), relay.shape_of(var_239))) # shape=(11, 12)
var_252 = relay.var("var_252", dtype = "bool", shape = (11, 12))#candidate|252|(11, 12)|var|bool
bop_253 = relay.equal(bop_246.astype('bool'), relay.reshape(var_252.astype('bool'), relay.shape_of(bop_246))) # shape=(11, 12)
bop_256 = relay.not_equal(bop_243.astype('bool'), relay.reshape(bop_249.astype('bool'), relay.shape_of(bop_243))) # shape=(11, 12)
bop_259 = relay.mod(uop_240.astype('float32'), relay.reshape(bop_253.astype('float32'), relay.shape_of(uop_240))) # shape=(11, 12)
uop_262 = relay.acos(bop_246.astype('float32')) # shape=(11, 12)
uop_264 = relay.asinh(bop_249.astype('float32')) # shape=(11, 12)
uop_266 = relay.log2(var_252.astype('float32')) # shape=(11, 12)
bop_268 = relay.floor_mod(uop_264.astype('float64'), relay.reshape(bop_246.astype('float64'), relay.shape_of(uop_264))) # shape=(11, 12)
bop_271 = relay.right_shift(uop_264.astype('uint8'), relay.reshape(bop_243.astype('uint8'), relay.shape_of(uop_264))) # shape=(11, 12)
uop_274 = relay.sigmoid(bop_271.astype('float32')) # shape=(11, 12)
const_276 = relay.const([[-4.402629,4.508845,0.605105,4.526095,1.221900,-0.570118,-2.602698,-2.086821,0.008257,-5.222169,4.567838,0.091820],[2.374294,5.380142,7.816815,7.664713,-3.169814,0.744219,3.430155,-4.277141,-0.769176,4.109672,5.019292,1.985118],[4.734799,-8.341102,-4.132943,7.474987,-6.942036,-2.549763,-0.860680,3.966516,-7.704122,-7.990243,8.413177,-1.629006],[5.893278,-5.308633,-8.950360,-9.873148,-8.182399,-3.327137,-2.749509,-5.068574,-7.003593,-8.935105,-6.557788,-9.272480],[7.004733,-6.509704,6.714810,-4.770956,7.330811,9.421689,-7.618815,3.590894,-2.426851,9.589998,9.472568,0.624456],[-7.019090,-2.045952,1.840401,-1.657846,6.335147,-7.826724,5.523468,-8.903644,3.514936,-5.165712,7.044768,-1.638250],[0.163766,-5.357083,-4.603203,5.123253,1.627416,8.367419,-9.058177,5.329068,7.728387,-0.998222,-5.382455,3.589599],[-2.591798,-1.774378,6.996498,3.475143,-9.975939,2.316530,9.680791,-6.231419,8.300270,6.635984,9.105873,2.984838],[-1.589858,1.259630,-8.416982,0.828966,5.816165,-2.001878,-0.322179,8.304063,6.122553,2.862230,3.783012,-5.788562],[-5.821469,-1.498841,-4.430335,-3.445741,4.031888,-6.707195,-1.103029,-5.849770,4.748767,-5.949289,-2.337382,-3.124033],[-2.137944,-2.515865,6.514494,-7.702611,6.422322,4.211273,2.717255,-6.295827,-8.179841,-7.378789,-1.989921,-8.953127]], dtype = "float32")#candidate|276|(11, 12)|const|float32
bop_277 = relay.mod(uop_274.astype('float64'), relay.reshape(const_276.astype('float64'), relay.shape_of(uop_274))) # shape=(11, 12)
uop_280 = relay.acos(uop_264.astype('float32')) # shape=(11, 12)
func_215_call = mod.get_global_var('func_215')
func_222_call = mutated_mod.get_global_var('func_222')
const_283 = relay.const(-0.659007, dtype = "float64")#candidate|283|()|const|float64
const_284 = relay.const([[-6.019880,-5.344331,-2.005321,-2.568883,2.282554,-4.452038,7.538003,-1.896676,3.032114]], dtype = "float64")#candidate|284|(1, 9)|const|float64
const_285 = relay.const([7,5,-6,10,4,-5], dtype = "uint16")#candidate|285|(6,)|const|uint16
call_282 = relay.TupleGetItem(func_215_call(relay.reshape(const_283.astype('float64'), []), relay.reshape(const_283.astype('float64'), []), relay.reshape(const_283.astype('float64'), []), relay.reshape(const_284.astype('float64'), [9,]), relay.reshape(const_285.astype('uint16'), [6,]), relay.reshape(const_285.astype('bool'), [6,]), ), 3)
call_286 = relay.TupleGetItem(func_222_call(relay.reshape(const_283.astype('float64'), []), relay.reshape(const_283.astype('float64'), []), relay.reshape(const_283.astype('float64'), []), relay.reshape(const_284.astype('float64'), [9,]), relay.reshape(const_285.astype('uint16'), [6,]), relay.reshape(const_285.astype('bool'), [6,]), ), 3)
bop_287 = relay.equal(bop_277.astype('bool'), relay.reshape(bop_259.astype('bool'), relay.shape_of(bop_277))) # shape=(11, 12)
uop_290 = relay.acosh(uop_262.astype('float64')) # shape=(11, 12)
uop_292 = relay.exp(uop_274.astype('float32')) # shape=(11, 12)
const_294 = relay.const([[3.318315,6.204168,-3.323815,5.822535,-9.862117,1.969434,-5.231607,-0.994492,8.103271,7.110134,9.240097,7.183204],[5.724084,-0.759699,9.253961,-7.729612,-8.368833,-5.250766,-0.691993,3.170492,-8.606103,6.113230,3.205472,5.605508],[-0.899032,2.896043,-0.832072,-1.730300,0.198461,-7.152388,-5.494577,-7.127044,-6.857636,-0.013984,-9.066335,1.227969],[-2.820739,2.938571,1.442255,-4.106636,2.488278,6.033959,-1.126608,0.221808,3.659291,8.130569,-2.214216,-5.600957],[-1.903726,8.254445,4.698836,0.307795,8.338385,1.340475,2.722347,4.965582,8.648532,-6.300369,-8.121771,-5.239521],[-8.259127,-4.240359,4.993211,8.491713,6.940573,-0.689709,-3.633616,-0.558131,-7.575186,-5.104479,9.919530,-9.259739],[-9.402926,-3.144118,-2.518683,-1.646281,-5.897376,-1.941618,-5.895704,5.087702,-4.344786,-7.995205,3.327175,1.833521],[-3.056658,5.677081,5.352345,-7.075620,5.586108,-2.570673,1.607605,-8.811080,2.346377,6.123729,-5.760679,5.193934],[-2.413088,0.506688,-1.647609,2.377131,-1.584522,-5.154621,-9.174918,-6.434311,-9.596978,-8.388072,0.721398,7.709671],[0.922892,-9.408631,-1.436312,-7.972347,5.230176,-7.457368,-3.454345,-5.933184,4.035001,-2.804922,-3.123258,9.261055],[-9.670263,-3.323707,-9.054088,-8.422030,7.411835,7.074410,8.223058,-5.742559,-8.421942,-8.134179,8.372497,7.416859]], dtype = "float32")#candidate|294|(11, 12)|const|float32
bop_295 = relay.floor_mod(uop_292.astype('float32'), relay.reshape(const_294.astype('float32'), relay.shape_of(uop_292))) # shape=(11, 12)
output = relay.Tuple([bop_256,uop_266,bop_268,uop_280,call_282,const_283,const_284,const_285,bop_287,uop_290,bop_295,])
output2 = relay.Tuple([bop_256,uop_266,bop_268,uop_280,call_286,const_283,const_284,const_285,bop_287,uop_290,bop_295,])
func_298 = relay.Function([var_239,var_242,var_252,], output)
mod['func_298'] = func_298
mod = relay.transform.InferType()(mod)
var_299 = relay.var("var_299", dtype = "float64", shape = (11, 12))#candidate|299|(11, 12)|var|float64
var_300 = relay.var("var_300", dtype = "float64", shape = (11, 12))#candidate|300|(11, 12)|var|float64
var_301 = relay.var("var_301", dtype = "bool", shape = (11, 12))#candidate|301|(11, 12)|var|bool
output = func_298(var_299,var_300,var_301,)
func_302 = relay.Function([var_299,var_300,var_301,], output)
mutated_mod['func_302'] = func_302
mutated_mod = relay.transform.InferType()(mutated_mod)
var_304 = relay.var("var_304", dtype = "int16", shape = ())#candidate|304|()|var|int16
const_305 = relay.const(-8, dtype = "int16")#candidate|305|()|const|int16
bop_306 = relay.greater(var_304.astype('bool'), const_305.astype('bool')) # shape=()
uop_309 = relay.acos(bop_306.astype('float64')) # shape=()
uop_311 = relay.acos(uop_309.astype('float32')) # shape=()
uop_313 = relay.atan(uop_309.astype('float64')) # shape=()
uop_315 = relay.cosh(uop_309.astype('float64')) # shape=()
const_317 = relay.const([[-5.702752,-7.263673,3.046721,1.496849,-4.178713,6.090951,8.232874,-7.679940,-8.902749,-0.482760,7.212383,3.427001],[9.721774,-0.862812,-0.088117,7.390015,-4.376394,-8.169794,8.639771,-9.212723,-4.245448,-2.340730,6.149297,-5.334653]], dtype = "float64")#candidate|317|(2, 12)|const|float64
bop_318 = relay.subtract(uop_315.astype('uint64'), const_317.astype('uint64')) # shape=(2, 12)
bop_321 = relay.right_shift(uop_309.astype('int32'), uop_313.astype('int32')) # shape=()
output = relay.Tuple([uop_311,bop_318,bop_321,])
output2 = relay.Tuple([uop_311,bop_318,bop_321,])
func_324 = relay.Function([var_304,], output)
mod['func_324'] = func_324
mod = relay.transform.InferType()(mod)
mutated_mod['func_324'] = func_324
mutated_mod = relay.transform.InferType()(mutated_mod)
var_325 = relay.var("var_325", dtype = "int16", shape = ())#candidate|325|()|var|int16
func_324_call = mutated_mod.get_global_var('func_324')
call_326 = func_324_call(var_325)
output = call_326
func_327 = relay.Function([var_325], output)
mutated_mod['func_327'] = func_327
mutated_mod = relay.transform.InferType()(mutated_mod)
const_329 = relay.const([[7,5,9,-4,6,-1,-10,9,-5,-7,-4,3,6,2],[4,-3,1,4,-6,8,-8,-6,-7,-2,-7,3,9,-5],[-7,-4,10,-5,7,-8,8,4,8,-4,5,5,-4,-3],[6,10,8,6,3,-6,9,-3,3,-2,-7,4,-4,-6],[-9,3,-1,-9,-9,7,-6,-2,7,3,3,1,-4,4],[5,-6,10,-10,-2,-10,-5,3,-9,6,10,-8,-5,-5]], dtype = "uint16")#candidate|329|(6, 14)|const|uint16
const_330 = relay.const([[-8,-9,-10,-9,5,-1,2,1,7,-7,-8,-4,1,-2],[3,-2,-7,-1,-8,-9,2,10,-3,3,2,3,-2,-9],[-6,1,-5,2,4,10,-4,6,8,1,6,3,1,10],[-4,-6,1,2,10,7,-5,-7,3,8,-7,3,2,3],[-2,9,-1,-10,2,-8,4,-8,2,10,7,9,7,-9],[-7,-4,-5,-9,-10,1,-9,-6,-4,-3,-10,-2,-3,2]], dtype = "uint16")#candidate|330|(6, 14)|const|uint16
bop_331 = relay.not_equal(const_329.astype('bool'), relay.reshape(const_330.astype('bool'), relay.shape_of(const_329))) # shape=(6, 14)
bop_334 = relay.logical_or(const_330.astype('bool'), relay.reshape(bop_331.astype('bool'), relay.shape_of(const_330))) # shape=(6, 14)
output = relay.Tuple([bop_334,])
output2 = relay.Tuple([bop_334,])
func_337 = relay.Function([], output)
mod['func_337'] = func_337
mod = relay.transform.InferType()(mod)
output = func_337()
func_338 = relay.Function([], output)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
const_339 = relay.const([[[False,False,False],[True,True,True],[False,False,False],[True,True,True],[False,True,False],[False,True,True],[False,True,True],[True,False,False],[False,False,True],[True,False,True]],[[True,False,False],[True,True,True],[True,False,True],[True,True,True],[True,False,False],[True,False,False],[True,False,True],[False,True,True],[False,True,True],[True,False,True]],[[False,False,True],[False,True,True],[False,False,True],[True,False,False],[False,True,False],[True,False,False],[True,True,False],[False,True,True],[True,True,True],[False,False,True]],[[False,True,True],[True,True,True],[False,True,True],[True,True,True],[False,True,False],[False,False,True],[True,True,False],[True,True,False],[True,False,True],[False,True,False]],[[True,True,True],[True,False,False],[False,True,False],[True,False,True],[True,False,False],[True,True,True],[False,False,False],[True,True,False],[True,False,False],[False,False,True]],[[False,False,False],[True,True,True],[False,True,False],[False,False,False],[True,True,True],[True,True,False],[False,True,False],[True,False,True],[True,True,False],[True,False,False]],[[True,False,False],[True,False,True],[False,False,False],[False,False,True],[True,True,False],[False,False,True],[False,True,True],[False,False,True],[True,True,True],[True,False,True]],[[False,True,True],[False,True,True],[False,True,True],[False,False,True],[True,True,False],[False,True,True],[True,True,False],[False,True,False],[False,False,True],[True,False,True]],[[False,False,True],[True,False,False],[True,False,True],[False,True,True],[False,False,False],[False,False,True],[True,False,False],[True,False,True],[True,False,False],[False,True,False]],[[False,True,False],[True,True,False],[True,False,True],[False,False,False],[False,False,False],[False,True,True],[False,False,True],[False,False,True],[False,True,False],[False,False,True]],[[False,False,True],[False,True,False],[False,False,False],[True,False,False],[True,False,True],[False,True,False],[True,True,False],[False,True,False],[True,True,False],[True,True,False]],[[True,True,False],[False,True,True],[False,False,True],[False,True,True],[False,True,False],[True,True,True],[True,False,False],[False,False,True],[False,True,True],[False,True,False]],[[False,False,False],[False,False,True],[True,False,True],[False,False,True],[False,False,False],[False,False,True],[True,True,True],[True,True,True],[False,False,False],[True,False,True]]], dtype = "bool")#candidate|339|(13, 10, 3)|const|bool
const_340 = relay.const([[[True,True,False],[True,True,True],[True,True,True],[True,True,False],[True,False,False],[True,False,True],[False,True,False],[True,True,True],[False,True,True],[True,True,True]],[[True,False,False],[True,False,False],[True,True,True],[False,False,True],[False,False,True],[False,True,True],[True,False,False],[True,True,True],[False,True,False],[True,True,False]],[[False,False,False],[True,False,False],[True,True,True],[False,True,False],[True,True,False],[False,True,True],[True,False,True],[False,True,True],[True,True,False],[False,False,True]],[[False,True,False],[False,False,False],[True,True,True],[False,True,False],[False,True,True],[False,True,False],[False,True,False],[True,True,True],[False,False,False],[True,False,False]],[[False,False,True],[False,True,True],[False,False,False],[True,False,False],[True,True,True],[False,True,False],[False,True,True],[False,True,False],[True,True,True],[True,False,True]],[[False,True,False],[True,True,True],[True,False,True],[True,True,True],[True,True,False],[False,True,False],[True,False,True],[False,False,False],[True,False,False],[False,True,False]],[[True,True,True],[True,True,True],[False,False,True],[True,True,False],[True,True,True],[True,True,True],[True,False,True],[True,True,True],[True,False,True],[True,False,True]],[[False,True,False],[True,True,True],[False,True,True],[True,True,True],[False,False,False],[True,True,True],[False,True,False],[True,False,True],[True,True,True],[False,True,True]],[[True,True,False],[False,False,True],[True,True,True],[True,False,False],[False,False,False],[False,False,True],[True,False,False],[True,False,False],[False,True,False],[True,True,True]],[[False,True,False],[True,True,False],[False,False,True],[True,True,False],[True,True,False],[True,True,False],[True,True,True],[True,False,True],[False,True,False],[False,False,True]],[[True,True,False],[True,False,True],[True,False,False],[True,False,True],[True,True,True],[True,True,False],[False,False,True],[True,True,False],[False,True,True],[False,True,True]],[[True,False,False],[False,True,False],[True,True,True],[True,False,True],[True,True,False],[False,False,True],[False,False,True],[False,True,True],[False,True,False],[True,True,False]],[[False,True,False],[True,True,False],[True,False,True],[False,False,True],[True,True,False],[False,True,False],[True,True,True],[True,True,False],[False,True,False],[False,False,True]]], dtype = "bool")#candidate|340|(13, 10, 3)|const|bool
bop_341 = relay.logical_or(const_339.astype('bool'), relay.reshape(const_340.astype('bool'), relay.shape_of(const_339))) # shape=(13, 10, 3)
bop_344 = relay.greater(bop_341.astype('bool'), relay.reshape(const_340.astype('bool'), relay.shape_of(bop_341))) # shape=(13, 10, 3)
uop_347 = relay.log(bop_341.astype('float64')) # shape=(13, 10, 3)
uop_349 = relay.sin(uop_347.astype('float32')) # shape=(13, 10, 3)
bop_351 = relay.logical_or(uop_349.astype('bool'), relay.reshape(bop_344.astype('bool'), relay.shape_of(uop_349))) # shape=(13, 10, 3)
func_63_call = mod.get_global_var('func_63')
func_68_call = mutated_mod.get_global_var('func_68')
const_355 = relay.const([4.165182,-0.151948,6.891869,-3.736979,8.380819,0.737936,6.382466,-0.605735,-1.955032,5.481424,-9.537921,-2.891030,5.804600,4.738384,-6.951049,-9.111835,-1.049433,1.699728,-0.400442,3.886432,-9.147512,-6.583772,-5.343894,-7.487829,6.816457,3.738001,-7.776062,1.343338,-7.499768,2.521056,-5.066738,-2.886012,-8.745137,-2.784291,2.380907,-8.762995,-6.783772,-2.322653,4.608227,-8.236437,-7.480573,-5.889995,6.544722,1.046038,-1.471048,-3.020131,-2.301493,0.682625,0.554907,-5.630632,6.612522,-0.343427,6.020595,-6.810600,-7.270518,0.143910,-3.407316,5.936731,2.339532,2.364621,7.598125,6.625438,5.101173,-9.956732,-6.372261,5.888245,3.623716,-5.156971,-7.186496,9.646254,6.728832,-1.819976,-3.309379,1.765324,-7.565698,7.034402,4.895313,4.673844,7.511667,2.028402,-3.064238,-7.998590,-6.877182,-5.318684,2.236260,2.915876,-3.184520,5.597119,-5.361931,3.154787,5.312453,-3.154821,6.072832,9.151367,-7.122458,-3.851402,-5.895959,-9.018976,3.529830,8.721494,5.581017,9.378730,-8.035276,-3.803534,-5.728224,0.883696,8.030868,-0.377526,-9.150949,0.422287,-0.627009,0.217833,-5.850266,7.437054,6.693400,9.274337,-9.516563,-3.708835,5.206629,8.497754,-2.303433,2.583352,-5.097676,-2.612206,-3.715067,3.420335,-3.924924,4.255964,6.617975,-6.101828,9.513191,4.852472,9.161443,7.524756,-2.892554,-5.488956,0.920413,-1.272579,7.333162,-8.920512,5.196994,-5.931751,5.469173,-7.355905,-5.958987,5.578833,-4.419645,7.683935,-7.171851,-5.598330,5.544902,-3.096694,2.798454,7.170918,2.439437,-9.959651,0.877955,-3.222185,-4.515125,2.917733,-1.250273,-4.880379,8.286264,-0.385019,-6.130449,8.258571,9.021502,7.726726,-8.202713,6.180226,-0.998425,-3.424320,-7.446145,-0.265672,-6.473163,8.814621,-1.344584,4.197851,-2.547418,-4.007408,2.279115,-8.091474,2.752416,4.831205,-4.597910,7.152145,-8.802584,9.290546,7.956993,3.171168,9.589941,-7.866222,-5.529683,-7.452255,3.715550,-0.200313,-5.759862,5.082862,-5.586435,4.167896,3.121744,4.514411,3.423735,-5.849476,6.746944,-7.860189,-8.720989,-4.283187,-0.789383,3.100000,0.708112,-4.212025,6.556883,5.823832,-2.892467,-5.146797,-7.215981,4.070341,-0.742377,-1.806748,3.614733,7.366265,2.860823,3.113737,-7.359032,5.879107,-7.658127,5.656782,4.912210,5.463989,4.854125,4.283271,2.845533,0.190121,-8.390919,4.838007,-2.481404,4.048126,0.903046,-4.463953,-8.665490,8.969919,1.666044,-5.794747,-4.061567,8.244287,9.960037,5.648993,8.870756,0.037052,-0.714818,-5.700942,4.741815,-4.696985,-4.818323,-2.402467,-1.003692,3.739769,-1.972643,-8.070681,5.271654,-5.285016,-5.693205,6.958730,9.260295,4.477438,-8.760635,-8.255384,4.453870,-3.222026,4.592311,-4.404480,-1.258111,-8.165513,8.421351,6.157490,-7.073456,-4.182589,0.644570,-3.111984,-5.620416,-3.753429,4.910543,-2.859423,4.981198,-0.802401,2.966143,-2.591358,-0.523518,-5.743692,8.544307,-1.528145,8.062270,-2.700773,3.784974,3.594351,-8.558899,-5.890644,3.017415,2.544550,4.228889,3.979314,-7.115351,-0.305557,-8.962555,1.374692,5.519128,0.750497,-1.849262,9.312035,1.079010,0.378633,2.457053,2.501765,-7.605254,6.069901,6.440765,-1.947767,0.199726,-2.279528,7.955463,1.771337,-1.761274,6.854654,8.160235,-2.434359,-1.832615,-3.671501,8.236010,-7.123566,-7.818841,5.109610,-3.229961,-9.407099,-8.067722,-5.824658,-5.500356,-3.927602,-7.031438,8.910276,1.060576,-9.586645,1.864620,0.051317,7.635685,-0.617800,6.891383,9.651653,1.738547,2.605039,-0.267768,0.549265,-2.423024,5.398997,5.981774,-0.349551,-7.588675,-5.859526,3.183027,-2.293724,5.120473,0.469314,-3.137860,8.501158,-4.754165,-8.681497,-8.070219,4.496839,3.787179,9.809434,2.061761,3.205412,1.141045,9.745489,4.678280,7.945146,-1.336120,-1.613766,3.318124,-1.041433,-8.143005,-0.191349,-7.419329,8.953478,9.074592,-4.076713,-2.133953,-1.930859,-0.951902,5.416135,-7.801485,-5.100896,8.921348,1.031951,-0.545510,-5.479923,-1.245571,1.405372,-7.115923,0.949847,-6.816733,-4.419777,3.287220,-0.833080,-7.023174,-6.628751,-7.682953,8.181481,9.013277,-0.091553,-3.377505,-7.496472,9.911337,9.621401,9.719829,4.852005,-4.255636,-9.487007,0.563274,7.953182,-7.915017,-9.501528,3.170745,-9.970971,-1.086399,-2.293603,4.579205,-8.916311,5.022129,-2.083184,-1.350377,-1.563463,3.724103,-4.438059,9.417792,-6.740219,2.687506,-5.942303,-2.307366,5.004691,9.249708,9.359955,6.998799,2.806332,3.438012,2.126027,-4.229604,5.394015,-1.391159,-0.815517,7.428983,-9.241565,1.478738,0.295311,6.646639,-1.431921,3.611243,-4.527893,4.075807,9.331797,8.234277,-8.934668,-6.725728,-3.081474,-4.133028,-8.915514,3.556951,-9.827586,6.944298,1.176359,8.905041,0.838296,2.118265,3.680257,7.099211,-4.348016,4.566398,-3.783022,-3.465990,-2.604866,4.716105,-7.694576,7.385627,7.408424,2.125612,-9.203967,2.408049,7.556286,9.073734,2.854707,6.494322,7.806167,0.479699,-6.259337,-3.766092,-0.816124,1.650338,5.161331,5.424249,4.848276,1.870387,1.025622,7.189799,8.765067,8.189442,7.604185,-3.599452,8.488397,8.840840,3.855081,1.506058,7.089019,1.925770,4.163464,-2.235784,8.500198,-6.300549,-1.211364,9.459054,-3.888048,-1.285383,7.694426,9.594472,-1.273106,-2.457790,-3.440999,9.324124,-8.296294,-5.910572,-8.721260,-1.332696,-6.387609,-1.906432,-3.869629,4.136545,7.123767,-1.912647,-8.262169,9.640161,-0.814668,3.446024,5.674735,8.733913,7.258388,-9.433809,-6.486046,1.253800,1.944419,1.112976,4.195093,6.858392,-4.446742,7.566883,0.917331,-7.446367,-2.925447,-5.783529,-4.219892,6.191016,-7.230169,1.845193,-2.612786,4.162985,9.773317,4.705466,4.657664,5.327137,-3.482229,6.284525,6.780719,-1.709202,4.547976,-7.446807,-1.426272,2.424738,1.490823,5.716249,4.045256,7.153849,8.896545,0.244667,5.459144,-2.754794,8.374699,0.471682,9.034037,-9.769781,0.685879,7.503885,-9.503078,-8.831294,1.782855,6.840670,-2.543384,1.178507,2.390999,4.663665,4.033323,-8.666019,1.547962,-5.315731,4.754169,2.421775,0.033038,-3.040473,9.871629,9.260271,-2.390028,9.881606,5.117562,-1.467425,4.643620,-2.075952,-2.163659,0.731186,-9.954977], dtype = "float32")#candidate|355|(616,)|const|float32
var_356 = relay.var("var_356", dtype = "float64", shape = (44,))#candidate|356|(44,)|var|float64
var_357 = relay.var("var_357", dtype = "float32", shape = (1120,))#candidate|357|(1120,)|var|float32
call_354 = relay.TupleGetItem(func_63_call(relay.reshape(const_355.astype('float32'), [4, 14, 11]), relay.reshape(var_356.astype('float64'), [44,]), relay.reshape(const_355.astype('float32'), [4, 14, 11]), relay.reshape(var_357.astype('float32'), [1120,]), ), 6)
call_358 = relay.TupleGetItem(func_68_call(relay.reshape(const_355.astype('float32'), [4, 14, 11]), relay.reshape(var_356.astype('float64'), [44,]), relay.reshape(const_355.astype('float32'), [4, 14, 11]), relay.reshape(var_357.astype('float32'), [1120,]), ), 6)
output = relay.Tuple([bop_351,call_354,const_355,var_356,var_357,])
output2 = relay.Tuple([bop_351,call_358,const_355,var_356,var_357,])
func_359 = relay.Function([var_356,var_357,], output)
mod['func_359'] = func_359
mod = relay.transform.InferType()(mod)
mutated_mod['func_359'] = func_359
mutated_mod = relay.transform.InferType()(mutated_mod)
func_359_call = mutated_mod.get_global_var('func_359')
var_361 = relay.var("var_361", dtype = "float64", shape = (44,))#candidate|361|(44,)|var|float64
var_362 = relay.var("var_362", dtype = "float32", shape = (1120,))#candidate|362|(1120,)|var|float32
call_360 = func_359_call(var_361,var_362,)
output = call_360
func_363 = relay.Function([var_361,var_362,], output)
mutated_mod['func_363'] = func_363
mutated_mod = relay.transform.InferType()(mutated_mod)
const_365 = relay.const(3.317686, dtype = "float64")#candidate|365|()|const|float64
uop_366 = relay.tan(const_365.astype('float64')) # shape=()
output = relay.Tuple([uop_366,])
output2 = relay.Tuple([uop_366,])
func_368 = relay.Function([], output)
mod['func_368'] = func_368
mod = relay.transform.InferType()(mod)
mutated_mod['func_368'] = func_368
mutated_mod = relay.transform.InferType()(mutated_mod)
func_368_call = mutated_mod.get_global_var('func_368')
call_369 = func_368_call()
output = call_369
func_370 = relay.Function([], output)
mutated_mod['func_370'] = func_370
mutated_mod = relay.transform.InferType()(mutated_mod)
var_371 = relay.var("var_371", dtype = "float64", shape = ())#candidate|371|()|var|float64
uop_372 = relay.acosh(var_371.astype('float64')) # shape=()
var_374 = relay.var("var_374", dtype = "float64", shape = ())#candidate|374|()|var|float64
bop_375 = relay.logical_and(var_371.astype('bool'), var_374.astype('bool')) # shape=()
uop_378 = relay.atan(uop_372.astype('float64')) # shape=()
const_380 = relay.const([[-7.211771,-4.556265,-9.239922,-1.895099,-5.316366,6.107259,9.167699,-0.282577],[9.682178,3.342022,-9.196110,1.121993,4.683519,-6.192944,0.322991,-0.330676],[-5.658413,2.856032,-0.077078,-5.966116,6.792630,-1.210026,-1.886327,-2.427157],[0.590453,6.086470,-5.035858,0.775996,-3.007864,0.706245,-1.712810,5.547440],[-8.104523,3.063584,-3.840437,3.197698,8.630328,-5.407606,-0.204895,7.484119],[8.434331,9.836538,1.686194,5.145531,-8.705841,0.733879,-5.931825,5.935804]], dtype = "float64")#candidate|380|(6, 8)|const|float64
bop_381 = relay.maximum(uop_372.astype('uint64'), const_380.astype('uint64')) # shape=(6, 8)
bop_384 = relay.greater(uop_378.astype('bool'), const_380.astype('bool')) # shape=(6, 8)
output = relay.Tuple([bop_375,bop_381,bop_384,])
output2 = relay.Tuple([bop_375,bop_381,bop_384,])
F = relay.Function([var_371,var_374,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_371,var_374,], output2)
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
input_371= np.array(-7.111000, dtype='float64')
module1.set_input('var_371', input_371)
input_374= np.array(9.210038, dtype='float64')
module1.set_input('var_374', input_374)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_371, input_374, )
res3 = intrp3.evaluate()(input_371, input_374, )
res4 = intrp4.evaluate()(input_371, input_374, )
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
res1_2 = module1.get_output(2).asnumpy()
res2_2 = res2[2].asnumpy()
res3_2 = res3[2].asnumpy()
res4_2 = res4[2].asnumpy()
np.testing.assert_allclose(res1_2 ,res2_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res3_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res1_2 ,res4_2, atol=1e-3, rtol=1e-3)
(res1_2 == res2_2).all()
(res1_2 == res3_2).all()
(res1_2 == res4_2).all()
module5.set_input('var_371', input_371)
module5.set_input('var_374', input_374)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_371, input_374, )
res7 = intrp7.evaluate()(input_371, input_374, )
res8 = intrp8.evaluate()(input_371, input_374, )
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
res5_2 = module5.get_output(2).asnumpy()
res6_2 = res6[2].asnumpy()
res7_2 = res7[2].asnumpy()
res8_2 = res8[2].asnumpy()
np.testing.assert_allclose(res5_2 ,res6_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res7_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res5_2 ,res8_2, atol=1e-3, rtol=1e-3)
(res5_2 == res6_2).all()
(res5_2 == res7_2).all()
(res5_2 == res8_2).all()
module9.set_input('var_371', input_371)
module9.set_input('var_374', input_374)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_371, input_374, )
res11 = intrp11.evaluate()(input_371, input_374, )
res12 = intrp12.evaluate()(input_371, input_374, )
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
res9_2 = module9.get_output(2).asnumpy()
res10_2 = res10[2].asnumpy()
res11_2 = res11[2].asnumpy()
res12_2 = res12[2].asnumpy()
np.testing.assert_allclose(res9_2 ,res10_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res11_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res9_2 ,res12_2, atol=1e-3, rtol=1e-3)
(res9_2 == res10_2).all()
(res9_2 == res11_2).all()
(res9_2 == res12_2).all()
module13.set_input('var_371', input_371)
module13.set_input('var_374', input_374)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_371, input_374, )
res15 = intrp15.evaluate()(input_371, input_374, )
res16 = intrp16.evaluate()(input_371, input_374, )
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
res13_2 = module13.get_output(2).asnumpy()
res14_2 = res14[2].asnumpy()
res15_2 = res15[2].asnumpy()
res16_2 = res16[2].asnumpy()
np.testing.assert_allclose(res13_2 ,res14_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res15_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res13_2 ,res16_2, atol=1e-3, rtol=1e-3)
(res13_2 == res14_2).all()
(res13_2 == res15_2).all()
(res13_2 == res16_2).all()
module17.set_input('var_371', input_371)
module17.set_input('var_374', input_374)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_371, input_374, )
res19 = intrp19.evaluate()(input_371, input_374, )
res20 = intrp20.evaluate()(input_371, input_374, )
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
res17_2 = module17.get_output(2).asnumpy()
res18_2 = res18[2].asnumpy()
res19_2 = res19[2].asnumpy()
res20_2 = res20[2].asnumpy()
np.testing.assert_allclose(res17_2 ,res18_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res19_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res17_2 ,res20_2, atol=1e-3, rtol=1e-3)
(res17_2 == res18_2).all()
(res17_2 == res19_2).all()
(res17_2 == res20_2).all()
module21.set_input('var_371', input_371)
module21.set_input('var_374', input_374)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_371, input_374, )
res23 = intrp23.evaluate()(input_371, input_374, )
res24 = intrp24.evaluate()(input_371, input_374, )
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
res21_2 = module21.get_output(2).asnumpy()
res22_2 = res22[2].asnumpy()
res23_2 = res23[2].asnumpy()
res24_2 = res24[2].asnumpy()
np.testing.assert_allclose(res21_2 ,res22_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res23_2, atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(res21_2 ,res24_2, atol=1e-3, rtol=1e-3)
(res21_2 == res22_2).all()
(res21_2 == res23_2).all()
(res21_2 == res24_2).all()

'''18446744073709551615, 18446744073709551611,  9223372036854775808,
9223372036854775808,  9223372036854775808],...
9223372036854775808, 9223372036854775808, 9223372036854775808,
9223372036854775808, 9223372036854775808],...

'''