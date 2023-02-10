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
var_0 = relay.var("var_0", dtype = "float64", shape = ())#candidate|0|()|var|float64
uop_1 = relay.acos(var_0.astype('float64')) # shape=()
output = relay.Tuple([uop_1,])
output2 = relay.Tuple([uop_1,])
func_3 = relay.Function([var_0,], output)
mod['func_3'] = func_3
mod = relay.transform.InferType()(mod)
var_4 = relay.var("var_4", dtype = "float64", shape = ())#candidate|4|()|var|float64
output = func_3(var_4)
func_5 = relay.Function([var_4], output)
mutated_mod['func_5'] = func_5
mutated_mod = relay.transform.InferType()(mutated_mod)
const_7 = relay.const([[[-4.199000,7.353567,-2.159838,-3.832731,5.221656,8.700099,-5.259569,-2.417969,-9.289893]]], dtype = "float32")#candidate|7|(1, 1, 9)|const|float32
uop_8 = relay.acosh(const_7.astype('float32')) # shape=(1, 1, 9)
uop_10 = relay.sinh(uop_8.astype('float32')) # shape=(1, 1, 9)
bop_12 = relay.less_equal(uop_10.astype('bool'), relay.reshape(const_7.astype('bool'), relay.shape_of(uop_10))) # shape=(1, 1, 9)
var_15 = relay.var("var_15", dtype = "float32", shape = (11, 14, 9))#candidate|15|(11, 14, 9)|var|float32
bop_16 = relay.right_shift(uop_8.astype('uint16'), var_15.astype('uint16')) # shape=(11, 14, 9)
const_19 = relay.const([[[9.714719,0.047991,-7.030643,-5.068758,6.247059,-4.783721,-4.552212,9.313152,3.261726],[0.478580,6.730358,2.690648,1.174817,-2.775289,2.294515,0.306538,-4.168281,-6.447960],[-6.495324,3.744385,-4.415086,9.972635,-2.338055,3.658152,1.610096,-3.247788,3.584523],[-5.580900,0.900444,-2.721234,1.650627,5.482328,-0.273702,-3.394937,-4.454534,-6.618070]],[[7.754496,-9.687612,0.208534,4.144375,-9.256176,1.098491,-2.122202,-2.080249,-3.875788],[-9.412511,-0.037869,-2.198217,-6.858564,3.700377,-2.480418,8.693614,-5.042712,0.558729],[4.319438,-4.839287,-9.332514,6.314532,2.231868,5.337990,-9.763376,9.866231,5.825515],[4.360932,-3.672788,9.801508,1.011632,2.532282,-3.014259,3.153709,6.951000,3.472319]],[[3.524743,-9.831822,-9.535190,-8.942522,-5.258852,-7.014997,2.718892,8.424881,-5.018282],[-9.919356,6.871514,9.225949,-8.136657,2.863045,-3.444723,-0.225826,1.704340,8.859204],[-8.739958,-3.039354,0.294588,-3.289806,-9.499286,3.834309,-0.343800,-0.764777,9.283871],[9.041683,-7.288734,4.131202,2.591334,-7.568555,8.447246,0.601337,-9.694770,-5.275990]],[[-7.088343,-8.953734,-5.375934,9.366264,-3.973161,-5.641083,-7.635598,0.914718,-6.905573],[9.054429,-9.414467,-6.866168,2.065665,6.042135,1.666593,-1.448221,4.077399,-2.510168],[-4.360475,-5.441781,-4.782413,-6.047094,9.702189,4.845792,-2.351540,0.272360,0.761305],[-6.846488,6.309495,-6.832061,3.588009,-1.363522,6.167523,-8.312268,0.181205,-7.939018]],[[0.553860,-1.784581,6.555922,-8.513256,-1.775772,-6.643412,5.830225,0.881803,6.462120],[3.297665,-0.492131,8.471937,1.016887,8.877742,-4.800348,-4.039674,-1.320633,-1.417246],[-7.016479,8.377129,9.544878,-9.044051,4.291250,0.431328,2.071893,1.931890,7.395035],[-4.551469,1.088015,2.238319,2.066561,9.147797,-4.107737,8.917599,2.074660,9.807526]],[[1.781723,9.978130,6.689411,0.663050,-6.585775,-5.996704,6.111974,-7.709330,-0.945321],[-2.860527,9.382957,0.168694,-2.958551,-2.638401,-9.767555,-9.324649,-0.209308,-3.760576],[-5.189580,-3.996324,-4.031096,-7.424187,2.570554,-1.558747,-1.329149,-9.811624,6.205492],[7.713192,-5.724682,6.333267,-1.427471,-3.356619,7.552387,-0.325818,-5.312375,-2.078422]],[[-2.054477,-8.812037,6.232159,6.702526,-2.686605,0.922391,-5.156339,4.641390,8.194642],[0.601232,-8.395487,-0.815652,-3.330488,3.620877,0.695754,8.294344,6.668194,-9.977439],[4.440645,-5.359564,4.872319,7.273011,5.075421,-5.662374,8.519301,-0.562909,-3.207155],[2.203505,2.084142,7.100210,2.573049,-9.749406,4.340465,4.165569,1.777729,-6.930663]],[[-8.075832,-1.852057,-7.013441,-4.466819,-7.847018,8.465257,3.363574,3.053610,9.487850],[1.978746,3.780871,2.549172,-6.260406,7.847227,9.657501,-9.173071,8.911140,1.473347],[-8.206479,2.028793,-3.559763,2.811468,-4.537436,-3.185959,0.676793,-4.203480,-6.674907],[0.368674,-8.515196,-3.165534,-3.707286,1.951996,3.600572,3.440374,-3.991088,-4.362861]],[[3.774952,-6.500610,-2.592698,-3.558494,9.200999,-4.934788,-4.851865,-3.928488,9.301986],[-0.347093,-9.396550,9.214974,-6.981406,4.363116,-1.987586,5.423018,5.305482,0.286025],[2.865526,9.238771,-8.394130,-9.379268,-8.486229,5.218443,5.033176,3.759172,-9.713555],[2.455155,-8.660671,8.738326,-0.632485,-6.923907,6.676415,7.056472,8.371885,5.897031]],[[-0.585868,5.962947,-3.276675,-5.833317,-1.505089,7.422748,3.728651,-7.719430,6.232791],[-7.736328,-5.979010,5.010687,-8.882961,-1.110436,-4.715440,8.910132,2.244734,5.157073],[1.689022,4.745800,2.704707,-2.558266,4.791226,-8.646907,9.449063,-9.538484,-5.253009],[5.234718,-5.715915,-5.123965,8.207400,-8.130123,-6.743628,-4.398241,9.368276,-1.988748]],[[5.349725,4.262195,4.446313,8.344638,-5.830705,-1.178850,-9.068616,3.123407,-9.513216],[3.954613,8.171664,-4.365851,9.620081,-1.987725,2.269307,4.486245,4.750617,4.411439],[8.670436,7.039314,2.803933,-3.370428,4.254349,-3.904114,8.031361,8.547362,3.499759],[3.992123,9.015827,-8.312122,-5.288326,3.506322,-3.137532,-2.008585,1.169431,-6.993306]],[[6.390013,-2.443306,-1.471980,-1.362139,-1.965661,2.103991,-4.628609,-1.813697,-1.556959],[-2.004137,2.118944,-5.995090,-3.515191,1.323504,2.276133,-6.930215,-6.461265,6.886466],[-8.652575,-0.233133,-8.112375,-1.661401,-2.000485,-1.043340,9.636788,9.097535,-4.434297],[7.718077,-4.167210,-5.268738,-2.401458,2.063332,-7.787883,1.561622,-9.019299,9.744780]],[[7.908694,9.483232,6.206647,-1.204814,-8.656447,0.312204,-5.475876,-8.886982,4.261942],[-1.922760,9.189145,-8.109585,-8.366166,-2.889631,8.207304,3.701242,-8.281828,6.072005],[-4.010048,-8.310562,6.664411,0.740490,-0.893391,7.184925,1.575199,5.828384,3.681359],[-6.082238,-2.020854,-7.418936,-2.863925,-9.692642,3.773121,-9.885058,-2.584148,-8.555236]],[[-4.593016,8.836359,1.564348,-2.108131,-9.056583,-2.443828,5.853160,8.118950,4.116810],[-2.262492,-0.077324,-6.595196,-8.861416,8.214270,-7.112006,-5.929680,-5.347723,4.182045],[-3.017568,-3.266666,-4.097204,-2.745528,5.836185,2.198555,-8.183842,-4.981104,-4.388432],[-6.900037,-0.576245,2.441886,9.736647,5.227838,9.542965,-4.520276,9.456072,-9.927338]],[[6.003159,6.245798,0.230686,8.459680,-7.848602,-9.861837,3.506991,9.831090,-0.279391],[-5.317718,8.869810,5.644598,-2.415524,-3.474461,-6.930438,5.122533,-0.667555,-3.252716],[6.964620,4.205716,-9.406489,1.492185,-4.118175,4.415408,3.019625,-3.102697,0.992600],[-9.135143,9.559503,5.851490,0.341239,3.422471,-9.530176,9.622342,-9.986634,8.542270]],[[-5.075441,-0.080351,-9.927659,-3.770412,6.997127,-1.807237,-3.205198,-3.364337,9.937073],[-4.425621,2.336400,-4.916492,2.466548,8.307752,-0.603647,-9.131617,8.408812,6.108576],[-9.323831,-4.040442,3.443907,2.412849,3.230018,0.355096,3.988860,5.980543,-4.807039],[-4.111988,-0.018967,6.140286,8.550834,7.477180,-2.162243,-3.266083,-1.170169,-5.238877]]], dtype = "float32")#candidate|19|(16, 4, 9)|const|float32
bop_20 = relay.bitwise_xor(uop_8.astype('uint64'), const_19.astype('uint64')) # shape=(16, 4, 9)
uop_23 = relay.log10(uop_10.astype('float64')) # shape=(1, 1, 9)
uop_25 = relay.log10(uop_23.astype('float64')) # shape=(1, 1, 9)
bop_27 = relay.greater_equal(uop_10.astype('bool'), relay.reshape(uop_23.astype('bool'), relay.shape_of(uop_10))) # shape=(1, 1, 9)
uop_30 = relay.sqrt(bop_16.astype('float64')) # shape=(11, 14, 9)
uop_32 = relay.tan(bop_27.astype('float64')) # shape=(1, 1, 9)
var_34 = relay.var("var_34", dtype = "float64", shape = (12, 11, 9))#candidate|34|(12, 11, 9)|var|float64
bop_35 = relay.less_equal(uop_32.astype('bool'), var_34.astype('bool')) # shape=(12, 11, 9)
bop_38 = relay.less_equal(bop_35.astype('bool'), bop_27.astype('bool')) # shape=(12, 11, 9)
uop_41 = relay.acos(uop_8.astype('float32')) # shape=(1, 1, 9)
var_43 = relay.var("var_43", dtype = "bool", shape = (12, 11, 9))#candidate|43|(12, 11, 9)|var|bool
bop_44 = relay.add(bop_38.astype('int32'), relay.reshape(var_43.astype('int32'), relay.shape_of(bop_38))) # shape=(12, 11, 9)
bop_47 = relay.mod(bop_38.astype('float64'), uop_23.astype('float64')) # shape=(12, 11, 9)
output = relay.Tuple([bop_12,bop_20,uop_25,uop_30,uop_41,bop_44,bop_47,])
output2 = relay.Tuple([bop_12,bop_20,uop_25,uop_30,uop_41,bop_44,bop_47,])
func_50 = relay.Function([var_15,var_34,var_43,], output)
mod['func_50'] = func_50
mod = relay.transform.InferType()(mod)
mutated_mod['func_50'] = func_50
mutated_mod = relay.transform.InferType()(mutated_mod)
func_50_call = mutated_mod.get_global_var('func_50')
var_52 = relay.var("var_52", dtype = "float32", shape = (11, 14, 9))#candidate|52|(11, 14, 9)|var|float32
var_53 = relay.var("var_53", dtype = "float64", shape = (12, 11, 9))#candidate|53|(12, 11, 9)|var|float64
var_54 = relay.var("var_54", dtype = "bool", shape = (12, 11, 9))#candidate|54|(12, 11, 9)|var|bool
call_51 = func_50_call(var_52,var_53,var_54,)
output = call_51
func_55 = relay.Function([var_52,var_53,var_54,], output)
mutated_mod['func_55'] = func_55
mutated_mod = relay.transform.InferType()(mutated_mod)
const_57 = relay.const([[[10,-10,4,3,-7,-3,8,-3,3,9,2,5,5,4],[1,-1,-6,8,-5,9,-7,-5,6,-10,-5,6,8,-7],[-9,2,3,-1,-6,-10,1,5,2,6,5,-9,-4,8]],[[-6,-4,-9,5,10,2,-8,-9,2,6,3,3,8,10],[9,-4,4,10,-4,7,2,2,-2,8,6,-7,7,-6],[-6,-7,-4,-9,-10,-4,-7,-10,10,10,2,5,1,-4]],[[-4,3,9,-6,-4,-3,-6,-1,-10,4,3,-4,-10,-1],[-1,-1,-9,6,8,-3,8,7,5,-7,-10,-6,4,6],[-7,7,5,-6,4,-2,-3,-3,1,5,1,-2,4,2]],[[8,-2,-6,-6,6,-5,3,3,2,-8,-9,8,6,10],[8,10,1,-2,10,10,-9,8,-2,-9,-6,-7,2,4],[-5,-10,4,10,1,-4,1,8,-1,-1,5,7,-9,-6]],[[-10,-8,3,-10,9,4,-6,-2,-4,-7,5,8,-10,-1],[1,-9,-8,-6,7,1,9,-8,6,10,-8,-3,-6,3],[7,-6,-3,-9,-4,-5,-2,6,2,6,4,5,4,-3]],[[10,10,-1,-6,-1,-7,-10,7,9,-5,7,-8,-7,3],[3,-4,3,2,10,8,3,-8,10,-6,9,9,-1,9],[9,9,-7,7,-7,-3,9,8,-9,-4,2,-9,9,6]],[[-6,-8,-4,-1,2,-10,8,-3,5,8,8,6,-10,-10],[4,1,-3,-8,-7,-10,8,-7,-3,1,2,3,-1,-3],[-3,-2,-10,-8,-1,2,7,9,1,-10,-6,2,2,9]],[[2,6,4,-4,-8,-9,-1,9,2,3,-3,8,8,7],[-5,5,8,-5,-7,1,1,8,-1,-2,-6,6,10,7],[4,-7,-10,-6,-4,-5,10,-8,-5,3,-5,9,-9,-9]],[[1,-4,9,5,4,10,-4,8,-10,-5,5,1,2,-1],[6,-10,7,-3,-5,-8,2,6,-4,10,-3,6,-2,9],[-6,-6,10,9,-10,-6,3,-4,3,-2,6,-10,8,8]],[[5,-5,8,3,-7,-9,-9,2,10,-6,7,1,-2,4],[-7,10,-2,4,8,1,-2,-2,7,2,1,-10,9,-4],[2,-1,-2,2,-7,7,-4,1,-1,6,9,5,9,8]],[[7,-7,10,8,8,-8,-4,2,-5,5,-8,-6,-10,-8],[-6,-7,8,2,3,-2,9,7,8,3,-4,1,-9,10],[-8,10,-7,-7,-2,-7,-6,-7,-2,10,4,5,-10,-2]],[[3,-10,-2,-5,7,6,-8,2,3,-8,-4,2,2,7],[-4,-2,-1,-10,-1,2,6,5,10,-9,-9,-7,-7,-4],[-8,10,-10,-1,-3,3,-7,-3,-4,6,-10,-8,-1,-3]],[[-4,7,9,-4,-10,-1,-5,8,-1,4,1,-9,4,-9],[10,5,-10,6,2,10,4,-6,6,-2,6,2,7,-6],[9,-7,-9,5,-9,-1,-10,2,2,4,1,-5,-3,4]]], dtype = "uint8")#candidate|57|(13, 3, 14)|const|uint8
var_58 = relay.var("var_58", dtype = "uint8", shape = (13, 3, 14))#candidate|58|(13, 3, 14)|var|uint8
bop_59 = relay.bitwise_xor(const_57.astype('uint8'), relay.reshape(var_58.astype('uint8'), relay.shape_of(const_57))) # shape=(13, 3, 14)
bop_62 = relay.left_shift(var_58.astype('uint16'), relay.reshape(bop_59.astype('uint16'), relay.shape_of(var_58))) # shape=(13, 3, 14)
output = relay.Tuple([bop_62,])
output2 = relay.Tuple([bop_62,])
func_65 = relay.Function([var_58,], output)
mod['func_65'] = func_65
mod = relay.transform.InferType()(mod)
mutated_mod['func_65'] = func_65
mutated_mod = relay.transform.InferType()(mutated_mod)
var_66 = relay.var("var_66", dtype = "uint8", shape = (13, 3, 14))#candidate|66|(13, 3, 14)|var|uint8
func_65_call = mutated_mod.get_global_var('func_65')
call_67 = func_65_call(var_66)
output = call_67
func_68 = relay.Function([var_66], output)
mutated_mod['func_68'] = func_68
mutated_mod = relay.transform.InferType()(mutated_mod)
var_70 = relay.var("var_70", dtype = "float32", shape = (10,))#candidate|70|(10,)|var|float32
uop_71 = relay.tan(var_70.astype('float32')) # shape=(10,)
bop_73 = relay.left_shift(var_70.astype('uint32'), relay.reshape(uop_71.astype('uint32'), relay.shape_of(var_70))) # shape=(10,)
uop_76 = relay.atan(uop_71.astype('float64')) # shape=(10,)
bop_78 = relay.left_shift(uop_76.astype('int8'), relay.reshape(bop_73.astype('int8'), relay.shape_of(uop_76))) # shape=(10,)
bop_81 = relay.floor_mod(uop_76.astype('float32'), relay.reshape(uop_71.astype('float32'), relay.shape_of(uop_76))) # shape=(10,)
uop_84 = relay.atanh(var_70.astype('float32')) # shape=(10,)
bop_86 = relay.floor_divide(bop_81.astype('float64'), relay.reshape(uop_71.astype('float64'), relay.shape_of(bop_81))) # shape=(10,)
bop_89 = relay.multiply(bop_86.astype('uint16'), relay.reshape(uop_71.astype('uint16'), relay.shape_of(bop_86))) # shape=(10,)
var_92 = relay.var("var_92", dtype = "float64", shape = (10,))#candidate|92|(10,)|var|float64
bop_93 = relay.bitwise_xor(bop_86.astype('int64'), relay.reshape(var_92.astype('int64'), relay.shape_of(bop_86))) # shape=(10,)
uop_96 = relay.atan(bop_78.astype('float64')) # shape=(10,)
bop_98 = relay.add(uop_96.astype('uint16'), relay.reshape(bop_86.astype('uint16'), relay.shape_of(uop_96))) # shape=(10,)
uop_101 = relay.acosh(bop_98.astype('float32')) # shape=(10,)
bop_103 = relay.mod(uop_101.astype('float32'), relay.reshape(uop_96.astype('float32'), relay.shape_of(uop_101))) # shape=(10,)
bop_106 = relay.mod(uop_101.astype('float64'), relay.reshape(uop_84.astype('float64'), relay.shape_of(uop_101))) # shape=(10,)
var_109 = relay.var("var_109", dtype = "float64", shape = (10,))#candidate|109|(10,)|var|float64
bop_110 = relay.logical_and(bop_106.astype('bool'), relay.reshape(var_109.astype('bool'), relay.shape_of(bop_106))) # shape=(10,)
const_113 = relay.const([True,False,False,True,False,False,False,True,False,True], dtype = "bool")#candidate|113|(10,)|const|bool
bop_114 = relay.minimum(bop_110.astype('uint16'), relay.reshape(const_113.astype('uint16'), relay.shape_of(bop_110))) # shape=(10,)
output = relay.Tuple([bop_89,bop_93,bop_103,bop_114,])
output2 = relay.Tuple([bop_89,bop_93,bop_103,bop_114,])
func_117 = relay.Function([var_70,var_92,var_109,], output)
mod['func_117'] = func_117
mod = relay.transform.InferType()(mod)
mutated_mod['func_117'] = func_117
mutated_mod = relay.transform.InferType()(mutated_mod)
func_117_call = mutated_mod.get_global_var('func_117')
var_119 = relay.var("var_119", dtype = "float32", shape = (10,))#candidate|119|(10,)|var|float32
var_120 = relay.var("var_120", dtype = "float64", shape = (10,))#candidate|120|(10,)|var|float64
var_121 = relay.var("var_121", dtype = "float64", shape = (10,))#candidate|121|(10,)|var|float64
call_118 = func_117_call(var_119,var_120,var_121,)
output = call_118
func_122 = relay.Function([var_119,var_120,var_121,], output)
mutated_mod['func_122'] = func_122
mutated_mod = relay.transform.InferType()(mutated_mod)
var_124 = relay.var("var_124", dtype = "float64", shape = (14, 5))#candidate|124|(14, 5)|var|float64
var_125 = relay.var("var_125", dtype = "float64", shape = (14, 5))#candidate|125|(14, 5)|var|float64
bop_126 = relay.greater(var_124.astype('bool'), relay.reshape(var_125.astype('bool'), relay.shape_of(var_124))) # shape=(14, 5)
uop_129 = relay.tan(bop_126.astype('float64')) # shape=(14, 5)
uop_131 = relay.asinh(var_124.astype('float32')) # shape=(14, 5)
var_133 = relay.var("var_133", dtype = "float32", shape = (14, 5))#candidate|133|(14, 5)|var|float32
bop_134 = relay.power(uop_131.astype('float64'), relay.reshape(var_133.astype('float64'), relay.shape_of(uop_131))) # shape=(14, 5)
uop_137 = relay.cosh(uop_131.astype('float64')) # shape=(14, 5)
output = relay.Tuple([uop_129,bop_134,uop_137,])
output2 = relay.Tuple([uop_129,bop_134,uop_137,])
func_139 = relay.Function([var_124,var_125,var_133,], output)
mod['func_139'] = func_139
mod = relay.transform.InferType()(mod)
mutated_mod['func_139'] = func_139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_139_call = mutated_mod.get_global_var('func_139')
var_141 = relay.var("var_141", dtype = "float64", shape = (14, 5))#candidate|141|(14, 5)|var|float64
var_142 = relay.var("var_142", dtype = "float64", shape = (14, 5))#candidate|142|(14, 5)|var|float64
var_143 = relay.var("var_143", dtype = "float32", shape = (14, 5))#candidate|143|(14, 5)|var|float32
call_140 = func_139_call(var_141,var_142,var_143,)
output = call_140
func_144 = relay.Function([var_141,var_142,var_143,], output)
mutated_mod['func_144'] = func_144
mutated_mod = relay.transform.InferType()(mutated_mod)
var_146 = relay.var("var_146", dtype = "int64", shape = (3, 1, 14))#candidate|146|(3, 1, 14)|var|int64
const_147 = relay.const([[[-3,-1,-10,9,-7,3,-10,-7,-9,8,-8,-1,-3,4],[-7,-9,-1,1,5,3,1,-1,-10,-1,-9,-7,3,-7],[-4,6,-1,-4,-5,-6,-2,-4,6,5,4,-7,6,9],[1,10,8,-7,-2,3,-5,-10,7,-2,-9,-1,5,-5],[3,2,3,-9,1,2,-8,-3,-6,-10,6,8,-10,-10]],[[8,-6,1,-6,2,-5,-5,8,-6,6,1,6,9,5],[-6,4,2,6,-8,10,-10,-4,-6,6,-1,-5,-10,4],[-7,-1,-5,7,-3,-5,-5,-2,1,-8,8,8,-8,-4],[4,-10,-6,-3,3,10,-3,8,9,-3,-6,9,8,-3],[-2,7,-6,6,7,5,-9,8,10,-2,-6,3,-7,-5]],[[-10,1,7,8,4,-5,3,-3,-5,-4,8,7,-8,-5],[-1,-10,-9,4,9,-7,-6,-6,-2,-9,1,-3,-7,5],[1,2,10,-4,7,5,10,10,6,-2,6,-7,-5,8],[-1,9,4,-3,-5,5,6,7,4,9,3,10,5,6],[8,-1,-8,-10,-2,-9,3,-9,6,3,5,-3,5,8]]], dtype = "int64")#candidate|147|(3, 5, 14)|const|int64
bop_148 = relay.subtract(var_146.astype('int64'), const_147.astype('int64')) # shape=(3, 5, 14)
uop_151 = relay.acos(const_147.astype('float64')) # shape=(3, 5, 14)
uop_153 = relay.log(const_147.astype('float64')) # shape=(3, 5, 14)
bop_155 = relay.greater_equal(const_147.astype('bool'), var_146.astype('bool')) # shape=(3, 5, 14)
uop_158 = relay.acosh(uop_153.astype('float32')) # shape=(3, 5, 14)
uop_160 = relay.asin(uop_151.astype('float64')) # shape=(3, 5, 14)
output = relay.Tuple([bop_148,bop_155,uop_158,uop_160,])
output2 = relay.Tuple([bop_148,bop_155,uop_158,uop_160,])
func_162 = relay.Function([var_146,], output)
mod['func_162'] = func_162
mod = relay.transform.InferType()(mod)
var_163 = relay.var("var_163", dtype = "int64", shape = (3, 1, 14))#candidate|163|(3, 1, 14)|var|int64
output = func_162(var_163)
func_164 = relay.Function([var_163], output)
mutated_mod['func_164'] = func_164
mutated_mod = relay.transform.InferType()(mutated_mod)
var_166 = relay.var("var_166", dtype = "float32", shape = (2, 7))#candidate|166|(2, 7)|var|float32
uop_167 = relay.log2(var_166.astype('float32')) # shape=(2, 7)
uop_169 = relay.erf(uop_167.astype('float32')) # shape=(2, 7)
output = relay.Tuple([uop_169,])
output2 = relay.Tuple([uop_169,])
func_171 = relay.Function([var_166,], output)
mod['func_171'] = func_171
mod = relay.transform.InferType()(mod)
mutated_mod['func_171'] = func_171
mutated_mod = relay.transform.InferType()(mutated_mod)
var_172 = relay.var("var_172", dtype = "float32", shape = (2, 7))#candidate|172|(2, 7)|var|float32
func_171_call = mutated_mod.get_global_var('func_171')
call_173 = func_171_call(var_172)
output = call_173
func_174 = relay.Function([var_172], output)
mutated_mod['func_174'] = func_174
mutated_mod = relay.transform.InferType()(mutated_mod)
const_176 = relay.const(-6.266781, dtype = "float32")#candidate|176|()|const|float32
uop_177 = relay.rsqrt(const_176.astype('float32')) # shape=()
bop_179 = relay.add(const_176.astype('int32'), uop_177.astype('int32')) # shape=()
uop_182 = relay.sin(bop_179.astype('float32')) # shape=()
var_184 = relay.var("var_184", dtype = "float32", shape = (15, 13, 15))#candidate|184|(15, 13, 15)|var|float32
bop_185 = relay.bitwise_xor(uop_182.astype('int8'), var_184.astype('int8')) # shape=(15, 13, 15)
uop_188 = relay.asin(bop_179.astype('float64')) # shape=()
bop_190 = relay.bitwise_or(uop_182.astype('int16'), uop_188.astype('int16')) # shape=()
bop_193 = relay.divide(bop_190.astype('float64'), bop_185.astype('float64')) # shape=(15, 13, 15)
uop_196 = relay.cosh(var_184.astype('float32')) # shape=(15, 13, 15)
bop_198 = relay.minimum(bop_179.astype('uint64'), uop_177.astype('uint64')) # shape=()
uop_201 = relay.asinh(uop_177.astype('float32')) # shape=()
uop_203 = relay.cos(uop_201.astype('float64')) # shape=()
var_205 = relay.var("var_205", dtype = "float64", shape = (15, 13, 15))#candidate|205|(15, 13, 15)|var|float64
bop_206 = relay.equal(bop_193.astype('bool'), relay.reshape(var_205.astype('bool'), relay.shape_of(bop_193))) # shape=(15, 13, 15)
bop_209 = relay.less(uop_188.astype('bool'), uop_201.astype('bool')) # shape=()
uop_212 = relay.rsqrt(bop_206.astype('float32')) # shape=(15, 13, 15)
uop_214 = relay.cosh(bop_190.astype('float32')) # shape=()
bop_216 = relay.floor_divide(uop_212.astype('float64'), uop_182.astype('float64')) # shape=(15, 13, 15)
bop_219 = relay.add(uop_203.astype('float64'), uop_201.astype('float64')) # shape=()
bop_222 = relay.add(bop_179.astype('int64'), uop_201.astype('int64')) # shape=()
uop_225 = relay.tan(uop_214.astype('float32')) # shape=()
bop_227 = relay.floor_divide(uop_212.astype('float32'), uop_225.astype('float32')) # shape=(15, 13, 15)
uop_230 = relay.asin(bop_219.astype('float32')) # shape=()
bop_232 = relay.greater(uop_230.astype('bool'), bop_190.astype('bool')) # shape=()
bop_235 = relay.floor_mod(uop_230.astype('float64'), uop_203.astype('float64')) # shape=()
uop_238 = relay.exp(bop_193.astype('float32')) # shape=(15, 13, 15)
uop_240 = relay.cosh(uop_230.astype('float64')) # shape=()
uop_242 = relay.tan(bop_232.astype('float32')) # shape=()
output = relay.Tuple([uop_196,bop_198,bop_209,bop_216,bop_222,bop_227,bop_235,uop_238,uop_240,uop_242,])
output2 = relay.Tuple([uop_196,bop_198,bop_209,bop_216,bop_222,bop_227,bop_235,uop_238,uop_240,uop_242,])
func_244 = relay.Function([var_184,var_205,], output)
mod['func_244'] = func_244
mod = relay.transform.InferType()(mod)
var_245 = relay.var("var_245", dtype = "float32", shape = (15, 13, 15))#candidate|245|(15, 13, 15)|var|float32
var_246 = relay.var("var_246", dtype = "float64", shape = (15, 13, 15))#candidate|246|(15, 13, 15)|var|float64
output = func_244(var_245,var_246,)
func_247 = relay.Function([var_245,var_246,], output)
mutated_mod['func_247'] = func_247
mutated_mod = relay.transform.InferType()(mutated_mod)
var_249 = relay.var("var_249", dtype = "float32", shape = (14,))#candidate|249|(14,)|var|float32
uop_250 = relay.asin(var_249.astype('float32')) # shape=(14,)
uop_252 = relay.cos(uop_250.astype('float64')) # shape=(14,)
bop_254 = relay.left_shift(uop_252.astype('uint64'), relay.reshape(var_249.astype('uint64'), relay.shape_of(uop_252))) # shape=(14,)
uop_257 = relay.sigmoid(uop_252.astype('float64')) # shape=(14,)
uop_259 = relay.cos(uop_257.astype('float64')) # shape=(14,)
var_261 = relay.var("var_261", dtype = "float32", shape = (14,))#candidate|261|(14,)|var|float32
bop_262 = relay.greater_equal(uop_250.astype('bool'), relay.reshape(var_261.astype('bool'), relay.shape_of(uop_250))) # shape=(14,)
bop_265 = relay.less(uop_257.astype('bool'), relay.reshape(var_261.astype('bool'), relay.shape_of(uop_257))) # shape=(14,)
var_268 = relay.var("var_268", dtype = "float64", shape = (14,))#candidate|268|(14,)|var|float64
bop_269 = relay.right_shift(uop_259.astype('int16'), relay.reshape(var_268.astype('int16'), relay.shape_of(uop_259))) # shape=(14,)
bop_272 = relay.bitwise_and(bop_265.astype('int32'), relay.reshape(var_268.astype('int32'), relay.shape_of(bop_265))) # shape=(14,)
bop_275 = relay.equal(bop_272.astype('bool'), relay.reshape(var_249.astype('bool'), relay.shape_of(bop_272))) # shape=(14,)
bop_278 = relay.bitwise_and(bop_269.astype('int32'), relay.reshape(uop_257.astype('int32'), relay.shape_of(bop_269))) # shape=(14,)
bop_281 = relay.mod(uop_259.astype('float32'), relay.reshape(bop_278.astype('float32'), relay.shape_of(uop_259))) # shape=(14,)
bop_284 = relay.less_equal(uop_257.astype('bool'), relay.reshape(bop_278.astype('bool'), relay.shape_of(uop_257))) # shape=(14,)
uop_287 = relay.log(bop_281.astype('float64')) # shape=(14,)
uop_289 = relay.sqrt(uop_252.astype('float64')) # shape=(14,)
bop_291 = relay.bitwise_and(bop_272.astype('int64'), relay.reshape(bop_275.astype('int64'), relay.shape_of(bop_272))) # shape=(14,)
bop_294 = relay.maximum(bop_272.astype('uint16'), relay.reshape(var_249.astype('uint16'), relay.shape_of(bop_272))) # shape=(14,)
uop_297 = relay.acos(bop_278.astype('float32')) # shape=(14,)
bop_299 = relay.greater(bop_262.astype('bool'), relay.reshape(bop_278.astype('bool'), relay.shape_of(bop_262))) # shape=(14,)
output = relay.Tuple([bop_254,bop_284,uop_287,uop_289,bop_291,bop_294,uop_297,bop_299,])
output2 = relay.Tuple([bop_254,bop_284,uop_287,uop_289,bop_291,bop_294,uop_297,bop_299,])
func_302 = relay.Function([var_249,var_261,var_268,], output)
mod['func_302'] = func_302
mod = relay.transform.InferType()(mod)
var_303 = relay.var("var_303", dtype = "float32", shape = (14,))#candidate|303|(14,)|var|float32
var_304 = relay.var("var_304", dtype = "float32", shape = (14,))#candidate|304|(14,)|var|float32
var_305 = relay.var("var_305", dtype = "float64", shape = (14,))#candidate|305|(14,)|var|float64
output = func_302(var_303,var_304,var_305,)
func_306 = relay.Function([var_303,var_304,var_305,], output)
mutated_mod['func_306'] = func_306
mutated_mod = relay.transform.InferType()(mutated_mod)
var_308 = relay.var("var_308", dtype = "float32", shape = (14, 10))#candidate|308|(14, 10)|var|float32
uop_309 = relay.log10(var_308.astype('float32')) # shape=(14, 10)
var_311 = relay.var("var_311", dtype = "float32", shape = (14, 10))#candidate|311|(14, 10)|var|float32
bop_312 = relay.subtract(var_308.astype('uint32'), relay.reshape(var_311.astype('uint32'), relay.shape_of(var_308))) # shape=(14, 10)
var_315 = relay.var("var_315", dtype = "float32", shape = (14, 10))#candidate|315|(14, 10)|var|float32
bop_316 = relay.bitwise_or(uop_309.astype('uint16'), relay.reshape(var_315.astype('uint16'), relay.shape_of(uop_309))) # shape=(14, 10)
uop_319 = relay.sin(bop_316.astype('float32')) # shape=(14, 10)
bop_321 = relay.floor_mod(uop_309.astype('float64'), relay.reshape(var_308.astype('float64'), relay.shape_of(uop_309))) # shape=(14, 10)
var_324 = relay.var("var_324", dtype = "float32", shape = (14, 10))#candidate|324|(14, 10)|var|float32
bop_325 = relay.logical_and(uop_319.astype('bool'), relay.reshape(var_324.astype('bool'), relay.shape_of(uop_319))) # shape=(14, 10)
var_328 = relay.var("var_328", dtype = "float64", shape = (14, 10))#candidate|328|(14, 10)|var|float64
bop_329 = relay.divide(bop_321.astype('float32'), relay.reshape(var_328.astype('float32'), relay.shape_of(bop_321))) # shape=(14, 10)
output = relay.Tuple([bop_312,bop_325,bop_329,])
output2 = relay.Tuple([bop_312,bop_325,bop_329,])
func_332 = relay.Function([var_308,var_311,var_315,var_324,var_328,], output)
mod['func_332'] = func_332
mod = relay.transform.InferType()(mod)
var_333 = relay.var("var_333", dtype = "float32", shape = (14, 10))#candidate|333|(14, 10)|var|float32
var_334 = relay.var("var_334", dtype = "float32", shape = (14, 10))#candidate|334|(14, 10)|var|float32
var_335 = relay.var("var_335", dtype = "float32", shape = (14, 10))#candidate|335|(14, 10)|var|float32
var_336 = relay.var("var_336", dtype = "float32", shape = (14, 10))#candidate|336|(14, 10)|var|float32
var_337 = relay.var("var_337", dtype = "float64", shape = (14, 10))#candidate|337|(14, 10)|var|float64
output = func_332(var_333,var_334,var_335,var_336,var_337,)
func_338 = relay.Function([var_333,var_334,var_335,var_336,var_337,], output)
mutated_mod['func_338'] = func_338
mutated_mod = relay.transform.InferType()(mutated_mod)
var_340 = relay.var("var_340", dtype = "uint32", shape = (4,))#candidate|340|(4,)|var|uint32
var_341 = relay.var("var_341", dtype = "uint32", shape = (4,))#candidate|341|(4,)|var|uint32
bop_342 = relay.multiply(var_340.astype('uint32'), relay.reshape(var_341.astype('uint32'), relay.shape_of(var_340))) # shape=(4,)
output = relay.Tuple([bop_342,])
output2 = relay.Tuple([bop_342,])
F = relay.Function([var_340,var_341,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_340,var_341,], output2)
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
input_340= np.array([3,-1,3,-5], dtype='uint32')
module1.set_input('var_340', input_340)
input_341= np.array([-4,-3,-2,3], dtype='uint32')
module1.set_input('var_341', input_341)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_340, input_341, )
res3 = intrp3.evaluate()(input_340, input_341, )
res4 = intrp4.evaluate()(input_340, input_341, )
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
module5.set_input('var_340', input_340)
module5.set_input('var_341', input_341)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_340, input_341, )
res7 = intrp7.evaluate()(input_340, input_341, )
res8 = intrp8.evaluate()(input_340, input_341, )
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
module9.set_input('var_340', input_340)
module9.set_input('var_341', input_341)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_340, input_341, )
res11 = intrp11.evaluate()(input_340, input_341, )
res12 = intrp12.evaluate()(input_340, input_341, )
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
module13.set_input('var_340', input_340)
module13.set_input('var_341', input_341)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_340, input_341, )
res15 = intrp15.evaluate()(input_340, input_341, )
res16 = intrp16.evaluate()(input_340, input_341, )
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
module17.set_input('var_340', input_340)
module17.set_input('var_341', input_341)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_340, input_341, )
res19 = intrp19.evaluate()(input_340, input_341, )
res20 = intrp20.evaluate()(input_340, input_341, )
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
module21.set_input('var_340', input_340)
module21.set_input('var_341', input_341)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_340, input_341, )
res23 = intrp23.evaluate()(input_340, input_341, )
res24 = intrp24.evaluate()(input_340, input_341, )
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