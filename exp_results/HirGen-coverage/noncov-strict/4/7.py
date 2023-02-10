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
var_0 = relay.var("var_0", dtype = "int32", shape = (12, 8))#candidate|0|(12, 8)|var|int32
const_1 = relay.const([[-9,-8,9,-2,2,8,4,-6],[1,-9,-5,-6,4,6,2,1],[-1,6,8,-10,-3,2,-8,2],[2,-2,8,5,8,3,9,1],[3,-1,5,-3,8,6,6,5],[1,-4,9,-1,1,5,6,-2],[-3,-1,-4,9,-10,10,-2,3],[-6,2,-6,-10,-7,-9,-3,3],[5,6,-6,3,-9,3,2,7],[-3,-1,2,-8,-8,-1,7,7],[-7,1,-8,-1,-9,-8,-10,-7],[6,2,1,6,-4,-2,-10,-2]], dtype = "int32")#candidate|1|(12, 8)|const|int32
bop_2 = relay.bitwise_and(var_0.astype('int32'), relay.reshape(const_1.astype('int32'), relay.shape_of(var_0))) # shape=(12, 8)
uop_5 = relay.log(bop_2.astype('float64')) # shape=(12, 8)
uop_7 = relay.cosh(uop_5.astype('float64')) # shape=(12, 8)
bop_9 = relay.greater(uop_7.astype('bool'), relay.reshape(uop_5.astype('bool'), relay.shape_of(uop_7))) # shape=(12, 8)
output = relay.Tuple([bop_9,])
output2 = relay.Tuple([bop_9,])
func_12 = relay.Function([var_0,], output)
mod['func_12'] = func_12
mod = relay.transform.InferType()(mod)
var_13 = relay.var("var_13", dtype = "int32", shape = (12, 8))#candidate|13|(12, 8)|var|int32
output = func_12(var_13)
func_14 = relay.Function([var_13], output)
mutated_mod['func_14'] = func_14
mutated_mod = relay.transform.InferType()(mutated_mod)
var_16 = relay.var("var_16", dtype = "float64", shape = (9,))#candidate|16|(9,)|var|float64
uop_17 = relay.cos(var_16.astype('float64')) # shape=(9,)
func_12_call = mod.get_global_var('func_12')
func_14_call = mutated_mod.get_global_var('func_14')
var_20 = relay.var("var_20", dtype = "int32", shape = (96,))#candidate|20|(96,)|var|int32
call_19 = relay.TupleGetItem(func_12_call(relay.reshape(var_20.astype('int32'), [12, 8])), 0)
call_21 = relay.TupleGetItem(func_14_call(relay.reshape(var_20.astype('int32'), [12, 8])), 0)
uop_22 = relay.log10(uop_17.astype('float64')) # shape=(9,)
bop_24 = relay.minimum(uop_22.astype('int32'), relay.reshape(uop_17.astype('int32'), relay.shape_of(uop_22))) # shape=(9,)
bop_27 = relay.floor_divide(uop_22.astype('float64'), relay.reshape(uop_17.astype('float64'), relay.shape_of(uop_22))) # shape=(9,)
uop_30 = relay.log2(bop_24.astype('float32')) # shape=(9,)
uop_32 = relay.rsqrt(uop_30.astype('float64')) # shape=(9,)
uop_34 = relay.acos(uop_30.astype('float64')) # shape=(9,)
uop_36 = relay.log10(uop_34.astype('float64')) # shape=(9,)
func_12_call = mod.get_global_var('func_12')
func_14_call = mutated_mod.get_global_var('func_14')
call_38 = relay.TupleGetItem(func_12_call(relay.reshape(call_19.astype('int32'), [12, 8])), 0)
call_39 = relay.TupleGetItem(func_14_call(relay.reshape(call_19.astype('int32'), [12, 8])), 0)
uop_40 = relay.asinh(uop_36.astype('float32')) # shape=(9,)
output = relay.Tuple([call_19,var_20,bop_27,uop_32,call_38,uop_40,])
output2 = relay.Tuple([call_21,var_20,bop_27,uop_32,call_39,uop_40,])
func_42 = relay.Function([var_16,var_20,], output)
mod['func_42'] = func_42
mod = relay.transform.InferType()(mod)
var_43 = relay.var("var_43", dtype = "float64", shape = (9,))#candidate|43|(9,)|var|float64
var_44 = relay.var("var_44", dtype = "int32", shape = (96,))#candidate|44|(96,)|var|int32
output = func_42(var_43,var_44,)
func_45 = relay.Function([var_43,var_44,], output)
mutated_mod['func_45'] = func_45
mutated_mod = relay.transform.InferType()(mutated_mod)
const_47 = relay.const([2.627559,6.858113,7.470286,-6.223510,6.773559,2.107799,-0.660149,-1.404151,-0.672473,9.967617,6.694489,-9.922006,4.620913,8.661399], dtype = "float32")#candidate|47|(14,)|const|float32
uop_48 = relay.sqrt(const_47.astype('float32')) # shape=(14,)
uop_50 = relay.log10(uop_48.astype('float32')) # shape=(14,)
output = relay.Tuple([uop_50,])
output2 = relay.Tuple([uop_50,])
func_52 = relay.Function([], output)
mod['func_52'] = func_52
mod = relay.transform.InferType()(mod)
mutated_mod['func_52'] = func_52
mutated_mod = relay.transform.InferType()(mutated_mod)
func_52_call = mutated_mod.get_global_var('func_52')
call_53 = func_52_call()
output = call_53
func_54 = relay.Function([], output)
mutated_mod['func_54'] = func_54
mutated_mod = relay.transform.InferType()(mutated_mod)
var_55 = relay.var("var_55", dtype = "float64", shape = (12,))#candidate|55|(12,)|var|float64
uop_56 = relay.asin(var_55.astype('float64')) # shape=(12,)
output = uop_56
output2 = uop_56
func_58 = relay.Function([var_55,], output)
mod['func_58'] = func_58
mod = relay.transform.InferType()(mod)
var_59 = relay.var("var_59", dtype = "float64", shape = (12,))#candidate|59|(12,)|var|float64
output = func_58(var_59)
func_60 = relay.Function([var_59], output)
mutated_mod['func_60'] = func_60
mutated_mod = relay.transform.InferType()(mutated_mod)
const_62 = relay.const([[[-0.135371,8.785321,-8.793013,7.394418,3.396526,-7.528373,-8.400119,4.287456,-0.477462,-6.729909,-9.891415,0.247867,-1.987654],[-4.185578,-6.856663,-6.058481,0.300178,-4.698834,3.669321,-3.904712,-7.446201,5.613840,6.134211,3.601762,7.339643,6.109009],[-2.027870,-7.505728,-4.182490,7.956106,-8.120932,-7.865805,7.398051,-4.009181,-9.718524,5.732502,1.042818,-3.615957,6.058330],[5.935795,2.771652,-0.334633,6.078187,-0.616943,1.470274,-6.559957,5.414612,1.077149,-2.676553,3.007541,-0.080752,1.586191],[1.629207,-7.403512,1.945968,-8.889329,3.843280,-1.175056,6.471807,-4.836607,-9.010981,-2.560883,6.217914,-4.079708,3.125188],[2.336724,-1.495022,-0.672420,-5.153121,-5.882890,-9.831250,-2.781403,3.190459,-3.606492,6.499676,8.353489,3.634167,2.649100],[1.238215,-1.644291,6.784206,6.770478,-7.191763,-9.254262,-3.692953,-8.498077,8.444454,-4.215505,-4.203603,8.425251,-4.840181],[6.627933,-9.836762,-2.214779,9.688952,0.798651,-4.388615,-8.700082,-8.914845,7.633074,-8.418234,8.368225,-7.577855,-5.563528],[-9.936148,-7.419332,6.749190,-4.593632,-6.903240,9.373029,-7.122902,4.741087,3.405784,0.913615,-4.942490,0.805641,8.291790],[-6.567491,-2.148509,-4.027071,-7.250527,9.938787,-3.718897,5.626949,7.358912,-7.507105,4.277036,-6.530549,-9.871854,-4.991118],[9.404396,8.287341,-9.365291,-8.244134,-9.694027,4.270963,-2.895672,-0.801242,4.341578,3.318786,4.611483,-9.356835,9.566528],[4.212327,-9.519727,-3.524419,8.637334,-7.524824,7.426301,4.380949,5.004465,-2.115271,8.563648,8.615569,0.021996,3.658658],[-4.136554,6.459504,9.694223,4.833564,2.068774,8.409635,8.682847,-8.859156,-8.440357,2.931758,2.201522,0.514536,-2.567340],[6.507456,-5.258303,-5.415787,-0.258072,-4.744832,0.938373,8.136041,-6.958978,4.081933,7.345833,-2.094415,2.272241,7.587666]],[[-2.399201,-7.660816,6.336488,-6.695497,-9.680865,5.682192,-4.960889,-8.980515,3.473237,-0.366886,-7.782200,-7.033406,-0.165923],[-9.019530,-1.835285,-7.393228,-5.833300,8.056627,-5.500812,5.532689,6.489885,9.527350,5.305794,7.833190,1.648383,-1.318343],[9.766264,5.803117,7.865077,-1.244209,-6.019460,-9.157712,0.308470,-0.383274,0.242041,-0.344289,-0.016808,3.768352,-1.461779],[5.258374,-9.916554,2.373777,-5.948630,5.107386,-0.549374,5.446148,-9.468268,-4.146819,1.341934,3.682354,-8.281952,8.501348],[2.045000,-6.947709,3.339956,-4.437843,9.072599,-6.746613,-2.600637,-6.024477,1.287462,7.551949,9.055268,4.213399,4.536703],[-6.132242,-9.340866,-2.677868,-3.920497,2.728829,-2.946529,7.924444,-5.831695,6.010015,-3.147112,1.502130,2.430915,6.688013],[-2.322253,-4.716121,-1.757611,-3.967767,-5.980064,-7.861201,3.194770,5.513258,4.734931,-4.164014,-0.724409,7.204730,8.565900],[-9.560462,0.829456,8.132990,-2.787937,5.581720,-3.181935,2.063983,5.322227,5.832512,-8.009889,-4.657731,-4.986788,5.381859],[7.480045,3.766731,5.326897,2.393073,8.078458,9.282066,-8.885592,-9.958304,-4.263886,-9.656309,-0.081673,-6.187254,-5.880830],[-5.993703,2.023971,0.336662,-3.359818,1.718665,-9.176738,9.802776,-1.615504,0.825195,0.380601,3.602802,1.648089,8.068210],[-1.438632,-5.093049,4.922444,6.813442,-0.544109,5.306303,4.581516,2.305022,7.840978,2.765664,8.585250,4.068276,2.106594],[-1.257426,-2.914241,-0.467698,8.555225,0.368544,8.560172,-2.707997,5.137406,6.097771,-5.647386,-7.258624,6.049136,1.277959],[-0.773049,4.935981,-2.371115,-8.639936,-8.302227,6.104932,-7.571904,-2.043595,5.925111,3.125664,-5.475359,4.507144,1.119002],[-9.968875,8.438926,8.060932,-7.066005,0.098265,-2.282860,2.922282,-7.228973,4.213168,3.934476,3.156028,-7.214426,9.208050]],[[8.674122,1.687536,-1.664740,0.210770,-7.961353,3.138543,0.673266,1.957916,7.048335,-6.308798,6.790627,-3.910950,1.403923],[6.277286,-5.946418,-5.346340,0.017119,2.532566,-8.025903,7.314149,1.948276,6.455267,-9.616254,9.931196,-6.488504,2.917472],[5.997511,-4.019509,8.927765,-0.333904,2.797106,8.598181,-2.830003,-9.817540,4.951105,2.044547,1.087468,-1.712470,2.677728],[9.157968,8.928690,1.798539,-6.676923,-2.345851,7.042574,6.012815,6.744380,7.775504,3.292390,-1.577791,-4.379600,-2.021061],[3.841511,3.173013,-7.971734,-3.581138,9.784947,-6.707468,8.927866,-0.683061,4.126150,5.377983,-5.837782,-0.467275,0.478832],[0.176898,3.179440,0.961585,-1.253660,4.046038,7.324939,-8.073695,-5.083699,-0.329638,4.022626,-9.659744,-8.071885,-5.717539],[5.607265,4.679814,8.683466,-3.014437,-1.525865,4.198613,-1.937136,7.095071,-8.101748,-1.591957,-4.244288,9.715949,2.042296],[-3.735199,8.061921,4.103663,1.968328,2.821816,0.217005,9.199343,7.623692,-9.519141,-7.687910,0.330474,6.153309,8.969437],[-1.459628,-2.165495,-4.902089,-3.370993,1.362839,-6.360761,-7.942784,3.073590,5.774308,0.959926,-6.997947,-3.345852,0.115597],[3.071978,-4.874906,-9.502381,2.852211,-9.083212,4.664190,-7.670532,-2.194492,6.933434,-3.021260,-5.674806,-8.023449,6.834917],[0.369933,4.736127,2.060668,-1.977681,2.557131,5.982402,5.418243,-8.406013,-4.237404,1.791843,-1.676109,0.185557,3.146911],[-6.540821,4.999799,-9.944971,-4.634811,4.952833,9.871492,9.186013,-4.949235,0.612879,-8.550661,-0.702679,-1.431307,-5.298632],[-8.225922,-9.743660,9.013426,7.599467,-2.542983,-4.813270,5.533429,-2.259199,8.513792,2.851340,-9.932298,1.021276,5.269141],[0.577531,4.068512,-9.620345,8.489968,3.932999,9.121660,9.593170,9.312267,-8.173348,6.735798,9.632095,-7.535730,-9.797106]],[[9.389749,-1.391034,5.223017,7.231260,6.347853,3.213766,-5.396788,9.936127,3.399853,3.727230,-3.016591,-6.613006,-7.536813],[-2.918060,-7.208742,-0.099615,-4.690063,-3.076588,-6.668716,-7.160965,9.681670,5.510427,3.417301,-2.388511,9.593489,0.211165],[8.879877,-2.188909,9.890095,9.136770,4.663095,1.244021,-7.263630,-4.649974,6.833256,-7.976283,-9.626492,-7.847419,-0.041762],[-2.386287,-0.990847,-4.988715,1.976791,-8.248387,7.915420,6.128831,5.794278,2.250706,-8.892545,-6.085716,-1.877280,5.691993],[-7.692383,-9.780307,-5.755494,0.043161,-8.964481,-9.632987,-5.712370,-3.411905,-4.774539,-4.229347,1.099758,0.884722,2.512442],[2.165916,-4.088011,-6.248808,7.627778,-4.063202,-1.283757,-4.390430,7.474506,-7.480476,9.888752,5.329862,-3.779519,7.697747],[0.456148,-4.466282,4.787829,5.616195,-8.794030,-2.100332,-3.682939,1.016545,-5.649291,-9.921267,-4.995776,2.794517,2.802819],[-9.829005,3.737265,7.506702,-4.913238,5.779173,-0.003909,-8.734367,8.366957,-0.805326,4.122400,-7.105977,9.070041,4.120096],[8.497306,-8.344092,8.467152,2.094708,-0.758261,-4.339419,-7.085091,-8.018476,-8.053221,0.226064,-1.358154,8.932522,6.765882],[3.536686,-5.536237,-4.715967,-5.361688,1.193087,4.776972,-4.023934,6.717885,9.324437,-8.744327,-7.356272,-9.711361,-0.560736],[7.429226,-9.942022,5.973144,-7.734557,-9.619781,-4.674073,0.337937,-0.152758,-5.841405,-3.744883,5.613508,1.835319,-3.496407],[-3.808252,3.041138,0.071546,0.466292,9.051039,1.510927,-1.007846,-4.058874,-8.035782,-6.411881,6.037827,0.711166,4.265829],[0.752281,1.922492,-6.994888,-7.734134,-6.144190,3.503318,3.884243,9.315496,-2.295633,-4.222136,-3.343319,-1.500020,5.398289],[7.563786,9.871912,-8.008038,9.330364,-6.759650,2.554395,2.438763,-0.405579,8.194463,5.284344,4.603233,3.829609,-0.789660]],[[-0.867504,-6.107293,-9.326300,7.029438,-7.809658,-5.333571,5.075244,-8.533197,6.334255,-6.669575,-8.479709,1.918476,-8.701240],[0.598882,-0.394669,2.630904,-0.438233,7.717419,9.544998,-0.549434,-1.051078,2.010632,-3.834423,-1.787130,9.945909,0.506899],[8.354812,6.915176,2.602204,4.520448,7.213445,2.496607,-3.158509,-1.395774,8.547069,-3.659396,-8.612566,1.128350,0.990311],[-6.397076,-4.472328,-7.615642,5.344786,-5.938326,-4.286344,-8.000865,-3.053059,3.532691,-9.388231,1.245099,-1.968367,3.016497],[0.683840,1.622819,4.673575,-1.267446,-9.028001,2.695117,-5.570251,-7.180499,0.234275,-7.428161,-5.084877,-7.124557,-7.210598],[-5.598412,7.656279,8.639204,-6.380281,5.638603,-5.063984,-2.690122,-3.543353,-8.760001,3.668872,0.133564,-8.180060,-0.340506],[4.026878,2.627375,-9.718699,9.940051,-8.256160,3.496062,2.551637,-3.152689,-6.673834,-0.040750,-7.944652,1.561827,-4.809805],[-5.002808,5.735721,0.208765,8.784977,-3.784056,6.266197,-3.073853,-2.877701,-3.546203,-5.155829,8.085900,1.281136,-6.054670],[1.215144,5.110609,-7.061156,-2.872716,7.066060,6.824914,-3.130465,-9.308925,0.184314,4.110844,-3.247021,0.198587,-1.662765],[3.258886,6.563723,6.499406,-7.930573,-4.470550,-2.336781,-2.614527,-3.998428,2.433300,6.060104,-9.109248,-8.153469,-5.350577],[-5.810461,2.199828,-9.649779,-0.523569,-6.373789,-9.390904,8.484855,-1.474688,2.432565,0.743052,-0.196501,1.501894,-3.608073],[-6.029530,4.665023,-1.615977,-7.855226,0.942689,-4.552515,-0.414025,0.716072,0.002877,3.510658,6.941275,-0.329605,6.457626],[8.318744,-7.954525,-1.231989,-5.014732,2.476939,9.850024,-5.823473,1.399576,9.439086,-4.655734,-2.282402,7.088753,6.286241],[1.570695,2.394631,-9.239610,8.664891,-6.792896,0.810633,1.642167,1.080870,-3.839029,1.353590,2.298346,-6.281728,-5.865030]]], dtype = "float64")#candidate|62|(5, 14, 13)|const|float64
uop_63 = relay.erf(const_62.astype('float64')) # shape=(5, 14, 13)
uop_65 = relay.log10(const_62.astype('float32')) # shape=(5, 14, 13)
bop_67 = relay.right_shift(uop_63.astype('uint16'), relay.reshape(uop_65.astype('uint16'), relay.shape_of(uop_63))) # shape=(5, 14, 13)
uop_70 = relay.asin(uop_65.astype('float32')) # shape=(5, 14, 13)
bop_72 = relay.maximum(uop_65.astype('int32'), relay.reshape(bop_67.astype('int32'), relay.shape_of(uop_65))) # shape=(5, 14, 13)
bop_75 = relay.less(const_62.astype('bool'), relay.reshape(bop_67.astype('bool'), relay.shape_of(const_62))) # shape=(5, 14, 13)
uop_78 = relay.tan(uop_63.astype('float32')) # shape=(5, 14, 13)
func_58_call = mod.get_global_var('func_58')
func_60_call = mutated_mod.get_global_var('func_60')
var_81 = relay.var("var_81", dtype = "float64", shape = (12, 1))#candidate|81|(12, 1)|var|float64
call_80 = func_58_call(relay.reshape(var_81.astype('float64'), [12,]))
call_82 = func_58_call(relay.reshape(var_81.astype('float64'), [12,]))
bop_83 = relay.less_equal(bop_67.astype('bool'), relay.reshape(uop_65.astype('bool'), relay.shape_of(bop_67))) # shape=(5, 14, 13)
uop_86 = relay.log10(bop_75.astype('float32')) # shape=(5, 14, 13)
const_88 = relay.const([[[-8.707291,6.782280,4.135341,7.546454,3.447466,-1.448310,6.846969,-8.195358,-7.897122,3.241486,4.826968,9.758301,-9.466745],[3.171183,-3.148484,-4.858230,3.141869,9.999626,-8.224246,-8.704703,5.663708,0.867566,8.728146,0.442368,-0.479544,4.583201],[1.727907,-6.231851,4.012593,8.296523,-5.060188,8.029703,-2.362088,-6.361056,1.124585,-5.032311,9.121687,-8.670778,8.872614],[-4.550360,-9.319382,-3.729297,5.247124,7.043589,-2.228664,-6.783083,-1.589877,-3.040720,-5.702508,-7.625605,8.073003,-9.987257],[-8.070408,-1.166303,-1.665350,6.607277,-1.943276,3.916591,-8.148615,4.057706,3.680167,2.824422,8.229588,-0.053613,-3.243447],[5.403282,3.322573,5.658134,-2.223274,-7.332402,-7.418180,3.379508,5.503676,-9.509887,4.218030,-4.005227,-3.110520,1.129212],[7.113046,4.977449,-9.623568,5.942790,9.927802,1.979949,-2.025981,-2.440396,-6.474284,-3.086894,-7.457499,6.940070,-1.704130],[0.080805,-8.180514,-9.451197,-1.196105,4.547860,-3.868829,5.866505,1.656982,-6.410290,-0.783597,1.770852,-1.152441,-7.861478],[-7.926080,2.926202,4.162325,4.642661,4.485542,6.930828,-9.070213,-7.490953,4.794480,-7.165324,-1.186243,1.264457,7.813850],[5.754892,-6.467061,6.377074,-2.206415,8.364550,9.342578,2.936502,-7.871626,-0.434453,1.481436,1.988698,-9.159007,-1.758387],[-5.972722,4.571968,-9.632872,-6.924386,-1.902284,6.073063,-4.554366,-0.533577,-4.683442,-2.168533,-6.666918,5.783337,3.183045],[9.198664,-9.281814,-3.086569,3.380254,-2.714088,2.582601,-5.540265,-9.491952,-8.393258,-4.465201,-2.263567,-7.045841,7.757896],[-4.653160,-3.262561,1.925795,4.117521,9.453682,1.260308,-4.140479,-7.054821,-8.177657,1.858327,9.028303,5.145966,3.034059],[-3.152497,-0.911638,-5.768838,-3.041169,-6.120933,-6.126127,7.674576,2.757310,-5.011128,-6.185586,4.573785,3.071805,-1.178887]],[[-6.330917,-4.034427,2.526138,5.032889,-6.122149,-0.006359,7.107169,6.692977,5.158671,8.586899,5.199274,-3.681648,4.600727],[-0.920517,-9.476343,2.163211,-2.912879,-3.695066,-3.306572,8.995877,-5.371976,9.866546,-3.511770,2.060919,5.537578,-7.656738],[0.837681,-3.764106,7.642058,6.968364,1.405337,-5.308874,6.607579,-0.999387,9.691859,-4.733289,4.770407,4.912782,-3.363502],[6.780579,6.408556,-7.628358,-4.396395,5.234545,2.767866,-0.012727,2.518743,-9.921662,3.061147,-5.646672,-5.117788,-1.487811],[-7.995046,8.443439,8.444065,-5.918718,0.431512,9.415908,-4.407052,5.877843,4.607958,-9.624956,-1.472836,5.766381,1.694152],[-5.158911,1.488937,-5.755507,-7.576375,-5.645507,-0.185847,6.222801,-1.454760,-8.439797,1.832749,-9.449475,5.346194,-1.652191],[-5.862993,9.463001,9.649546,8.628628,4.921238,4.101614,-2.462339,6.264775,-9.714461,-4.451120,4.580291,5.836950,-1.313521],[4.241574,1.572877,9.253686,-8.045198,3.752865,3.449352,-0.917986,-5.541094,-3.884274,0.848719,-9.146256,-8.562162,-3.246079],[3.183054,5.054369,-4.127743,5.139612,-3.662343,7.029743,7.793785,-7.593744,8.024151,2.681573,-0.089540,-7.208801,2.523338],[-5.464520,6.608992,5.783112,-1.151004,-0.601656,-9.763883,-3.384423,2.646961,4.098635,2.941178,0.429313,-6.298082,-2.436596],[-5.183056,-9.128851,9.703293,-3.804973,-1.997532,1.135971,-7.774211,5.498160,1.020073,8.630130,1.212440,-7.615351,-8.601177],[1.781419,-2.239466,-8.101908,5.106616,0.861868,4.393269,5.521916,0.697670,-6.824034,1.114598,-4.742342,-0.154266,1.624762],[-4.642175,-4.796552,3.686150,4.404258,3.271271,-0.635136,7.892385,-3.565418,0.812690,-7.143058,-8.001304,-5.175710,-7.497744],[6.658759,-8.436839,-3.492647,-8.796660,-3.588879,-2.294164,7.641772,7.235774,-8.338813,7.249438,1.127129,7.917872,-6.600192]],[[-4.978821,6.397687,3.942804,9.573628,3.627894,0.509891,-2.773209,9.763609,2.334866,4.454377,-8.571588,9.402368,-4.500894],[4.397506,-8.937312,6.875323,-2.565618,5.470170,5.029779,-9.234396,-5.741613,-2.808870,7.157926,-9.953986,2.195671,3.691921],[0.043526,6.337314,8.462106,7.553965,-3.728703,-1.017222,2.629684,0.338400,2.266734,4.897445,2.883661,-3.743724,5.032694],[7.648541,-3.630140,1.663045,-6.825406,4.465272,2.865940,5.347013,-1.460754,-9.897677,-4.820462,-7.421980,6.374373,9.809821],[-8.300508,8.630799,-0.399075,-3.469457,5.168652,4.370621,-7.300615,7.717138,0.014599,-3.270299,6.054062,-1.835612,-3.897998],[9.067771,-0.801984,-2.544360,-8.857369,1.713041,-7.791154,-2.219869,1.180059,4.188639,-9.717255,5.311155,-8.450697,2.867710],[-3.014707,3.807242,4.797347,1.706130,3.285436,-2.955262,5.280754,0.009245,3.937081,-9.023481,5.335964,-5.181475,-4.961099],[3.228717,2.030406,-1.764493,-8.021786,-9.611967,-0.273439,-3.958516,6.608384,-3.416573,-5.341738,0.665960,9.365311,-0.561682],[5.574826,-8.759893,-6.935196,-6.446890,7.510677,3.148621,-4.276364,9.465698,7.410025,9.697867,-7.872248,-5.041810,9.655224],[-6.654069,9.471068,7.669350,9.355546,-9.297215,3.295693,-9.769148,-1.063054,9.257359,0.451205,2.242566,-8.040560,-6.772584],[-3.797124,9.533767,-0.224181,-9.773528,-6.829860,-3.492648,-6.547959,6.805460,-0.749356,-7.055574,0.190782,5.774944,-5.573201],[-5.739101,4.720564,4.573449,7.991129,7.073672,-4.380552,0.645901,0.526098,-5.389574,-6.738365,-6.214521,5.743221,7.681965],[-9.666257,-0.739472,-1.027003,2.030021,6.402140,1.555986,5.357423,-5.365794,8.817308,-0.982854,1.988160,-5.310888,-5.734437],[-8.752036,4.106926,-7.568261,5.715059,-4.317023,7.172750,4.060650,-5.320597,3.408873,-7.773246,3.625118,1.910726,-5.209690]],[[-1.009081,7.097535,7.739922,8.573631,8.793694,-7.245450,-2.339761,-5.186186,-4.734543,-3.777711,9.440654,-7.924551,4.757747],[0.334956,6.530020,8.034410,-4.396580,-1.242971,7.767445,-6.734168,-5.903020,9.918112,8.746938,-5.999150,0.543560,-4.147238],[4.085603,-4.368082,-6.174476,2.661458,-8.850220,6.556387,2.377817,-2.441574,1.015148,-9.236311,-7.074182,-9.203401,1.537856],[2.859627,8.204656,4.755745,8.966697,2.091303,4.469744,0.764276,-3.873961,-2.906085,6.741914,-3.796715,2.229887,-1.960578],[-7.319373,4.332042,5.941816,-6.009169,-1.842890,3.922024,-9.966685,3.926064,-1.426366,5.412425,-1.745812,-5.008898,-6.402036],[-8.976586,4.415185,1.091501,-1.761491,-3.767824,-7.874022,-0.276437,-1.557783,-6.175974,6.291839,1.280072,-4.406497,-7.646028],[-0.954679,-0.073548,1.640417,0.538587,3.122047,-9.259146,4.620002,-6.388998,7.812196,3.651834,-6.896955,-2.681012,1.699741],[4.606170,-6.082262,-3.990596,5.372963,-2.688753,-7.448737,2.129941,-2.505847,5.858047,-7.752179,0.639181,-7.636778,-5.232568],[-8.075289,-3.991476,-1.775278,-9.592498,1.648999,2.968760,3.874758,2.724427,9.145375,-4.424480,-4.559642,-1.080217,5.819890],[4.986623,-4.849376,-4.185585,-7.903211,-3.589985,0.027286,-6.005537,-4.634059,2.872019,9.260103,8.129538,2.125274,-1.076223],[4.557056,6.323639,6.540878,-4.086557,-4.386877,-6.610056,-5.238882,5.653940,4.499501,-8.383212,0.539317,-6.464409,4.245530],[-3.967029,1.137195,-6.894352,7.325113,3.152434,-8.844205,-0.711793,-6.321338,-3.316759,8.539666,-3.750162,9.420858,-9.492851],[-4.136724,1.431771,-3.201840,6.865774,-9.904567,-3.834729,4.429088,-8.129925,-3.434970,-0.135268,-7.862250,8.317180,-2.078841],[-9.491915,-3.008470,-6.799016,7.911608,-8.627761,5.320438,1.957515,-2.451220,8.496278,5.510760,5.048773,-6.765415,9.668652]],[[-1.770405,5.535675,-0.212373,-6.551919,-8.143674,-6.438137,-3.414065,-2.990273,-6.615371,-5.823001,9.595829,-1.630076,5.836669],[-4.001459,5.788827,-0.190911,-4.879515,-1.873999,-8.744455,-3.657753,2.954881,-2.105970,4.311125,-8.608065,2.086265,-1.137702],[-3.368932,1.012441,3.682650,-3.703662,3.979142,-6.091197,-3.117431,3.625854,-9.301071,1.491553,-6.475634,6.175784,-8.719328],[2.648849,-0.013418,8.068986,-5.007992,8.064211,-1.824921,-2.957002,-8.048923,-7.766447,3.491480,0.358875,1.911435,1.681658],[6.452713,7.725638,-8.975029,0.098273,5.453930,-3.722658,9.263517,9.578022,9.621057,-3.586171,-2.356747,-5.258522,-6.269399],[6.108849,-3.807151,9.999836,-1.815924,-0.620869,8.087863,9.819714,-5.167172,-8.580486,6.188014,3.782196,6.761105,5.595103],[-9.453660,4.176550,1.370837,-4.143218,0.033653,9.954998,1.750375,5.941054,7.075751,2.277115,-0.556756,-5.234733,0.425341],[-3.642576,1.565451,-3.900811,3.689579,-7.482677,6.813521,-3.062446,-1.889460,-7.777417,1.981145,-6.532604,-5.680021,-3.321751],[0.054725,7.458382,4.195411,5.376974,6.744053,-3.107160,5.545298,8.062339,-8.549654,8.337905,-0.873653,-7.692834,0.550825],[-5.517893,-5.669352,-1.706911,9.671476,-5.476132,3.202986,7.437115,3.674262,7.144621,-6.356295,1.110947,8.683796,4.745102],[-3.139377,0.585867,-4.443575,-7.808305,9.373302,-7.815453,-9.039854,0.207810,-3.375894,-6.331200,-1.823368,-9.362265,-7.673232],[2.909177,5.068248,6.202847,-7.946222,2.340160,4.422850,8.145851,0.420009,1.592520,-3.233217,5.119936,-5.393567,2.402933],[-9.447724,5.964872,7.120629,-7.356616,-8.454029,6.794747,-1.956795,5.205570,-0.321128,-7.515321,3.248771,9.791972,-4.508730],[-6.466925,-1.275460,-0.370169,-5.894750,5.444977,-6.839249,-0.543678,-5.412965,-2.008562,-3.736037,-1.142215,-9.128997,-9.511999]]], dtype = "float64")#candidate|88|(5, 14, 13)|const|float64
bop_89 = relay.equal(uop_63.astype('bool'), relay.reshape(const_88.astype('bool'), relay.shape_of(uop_63))) # shape=(5, 14, 13)
bop_92 = relay.bitwise_and(uop_63.astype('uint16'), relay.reshape(uop_70.astype('uint16'), relay.shape_of(uop_63))) # shape=(5, 14, 13)
uop_95 = relay.rsqrt(bop_67.astype('float64')) # shape=(5, 14, 13)
bop_97 = relay.divide(uop_95.astype('float32'), relay.reshape(const_88.astype('float32'), relay.shape_of(uop_95))) # shape=(5, 14, 13)
uop_100 = relay.rsqrt(uop_86.astype('float32')) # shape=(5, 14, 13)
var_102 = relay.var("var_102", dtype = "bool", shape = (5, 14, 13))#candidate|102|(5, 14, 13)|var|bool
bop_103 = relay.maximum(bop_83.astype('uint8'), relay.reshape(var_102.astype('uint8'), relay.shape_of(bop_83))) # shape=(5, 14, 13)
bop_106 = relay.minimum(uop_95.astype('float64'), relay.reshape(bop_67.astype('float64'), relay.shape_of(uop_95))) # shape=(5, 14, 13)
output = relay.Tuple([bop_72,uop_78,call_80,var_81,bop_89,bop_92,bop_97,uop_100,bop_103,bop_106,])
output2 = relay.Tuple([bop_72,uop_78,call_82,var_81,bop_89,bop_92,bop_97,uop_100,bop_103,bop_106,])
func_109 = relay.Function([var_81,var_102,], output)
mod['func_109'] = func_109
mod = relay.transform.InferType()(mod)
var_110 = relay.var("var_110", dtype = "float64", shape = (12, 1))#candidate|110|(12, 1)|var|float64
var_111 = relay.var("var_111", dtype = "bool", shape = (5, 14, 13))#candidate|111|(5, 14, 13)|var|bool
output = func_109(var_110,var_111,)
func_112 = relay.Function([var_110,var_111,], output)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
const_114 = relay.const([9.800729,-0.046369,-6.934669,3.570321,-5.683361,1.400503,-0.179130,-4.832495,-2.966263,-5.367741,9.817194,-9.761984], dtype = "float32")#candidate|114|(12,)|const|float32
const_115 = relay.const([6.760043,-5.421108,-7.678184,-8.987342,-7.134853,9.939750,-4.072764,-0.018589,9.283888,3.984887,4.763210,7.316696], dtype = "float32")#candidate|115|(12,)|const|float32
bop_116 = relay.power(const_114.astype('float32'), relay.reshape(const_115.astype('float32'), relay.shape_of(const_114))) # shape=(12,)
bop_119 = relay.greater(const_115.astype('bool'), relay.reshape(const_114.astype('bool'), relay.shape_of(const_115))) # shape=(12,)
uop_122 = relay.cosh(bop_119.astype('float32')) # shape=(12,)
uop_124 = relay.log(const_115.astype('float64')) # shape=(12,)
func_12_call = mod.get_global_var('func_12')
func_14_call = mutated_mod.get_global_var('func_14')
var_127 = relay.var("var_127", dtype = "int32", shape = (8, 12))#candidate|127|(8, 12)|var|int32
call_126 = relay.TupleGetItem(func_12_call(relay.reshape(var_127.astype('int32'), [12, 8])), 0)
call_128 = relay.TupleGetItem(func_14_call(relay.reshape(var_127.astype('int32'), [12, 8])), 0)
bop_129 = relay.greater_equal(const_115.astype('bool'), relay.reshape(const_114.astype('bool'), relay.shape_of(const_115))) # shape=(12,)
uop_132 = relay.acos(uop_122.astype('float64')) # shape=(12,)
uop_134 = relay.asin(uop_132.astype('float64')) # shape=(12,)
bop_136 = relay.greater_equal(uop_134.astype('bool'), relay.reshape(uop_132.astype('bool'), relay.shape_of(uop_134))) # shape=(12,)
uop_139 = relay.sinh(uop_132.astype('float32')) # shape=(12,)
uop_141 = relay.sin(uop_132.astype('float64')) # shape=(12,)
func_42_call = mod.get_global_var('func_42')
func_45_call = mutated_mod.get_global_var('func_45')
var_144 = relay.var("var_144", dtype = "float64", shape = (9,))#candidate|144|(9,)|var|float64
call_143 = relay.TupleGetItem(func_42_call(relay.reshape(var_144.astype('float64'), [9,]), relay.reshape(call_126.astype('int32'), [96,]), ), 0)
call_145 = relay.TupleGetItem(func_45_call(relay.reshape(var_144.astype('float64'), [9,]), relay.reshape(call_126.astype('int32'), [96,]), ), 0)
bop_146 = relay.greater_equal(uop_132.astype('bool'), relay.reshape(bop_136.astype('bool'), relay.shape_of(uop_132))) # shape=(12,)
bop_149 = relay.power(bop_136.astype('float64'), relay.reshape(bop_119.astype('float64'), relay.shape_of(bop_136))) # shape=(12,)
var_152 = relay.var("var_152", dtype = "float64", shape = (12,))#candidate|152|(12,)|var|float64
bop_153 = relay.right_shift(uop_132.astype('int64'), relay.reshape(var_152.astype('int64'), relay.shape_of(uop_132))) # shape=(12,)
bop_156 = relay.logical_or(uop_122.astype('bool'), relay.reshape(const_115.astype('bool'), relay.shape_of(uop_122))) # shape=(12,)
const_159 = relay.const([True,False,True,True,False,False,False,False,False,False,True,True], dtype = "bool")#candidate|159|(12,)|const|bool
bop_160 = relay.logical_and(bop_129.astype('bool'), relay.reshape(const_159.astype('bool'), relay.shape_of(bop_129))) # shape=(12,)
uop_163 = relay.cosh(bop_149.astype('float64')) # shape=(12,)
bop_165 = relay.floor_divide(bop_146.astype('float64'), relay.reshape(var_152.astype('float64'), relay.shape_of(bop_146))) # shape=(12,)
uop_168 = relay.acos(uop_163.astype('float64')) # shape=(12,)
uop_170 = relay.acos(bop_153.astype('float64')) # shape=(12,)
bop_172 = relay.power(uop_163.astype('float32'), var_127.astype('float32')) # shape=(8, 12)
var_175 = relay.var("var_175", dtype = "float64", shape = (12,))#candidate|175|(12,)|var|float64
bop_176 = relay.left_shift(uop_168.astype('uint16'), relay.reshape(var_175.astype('uint16'), relay.shape_of(uop_168))) # shape=(12,)
var_179 = relay.var("var_179", dtype = "float64", shape = (12,))#candidate|179|(12,)|var|float64
bop_180 = relay.mod(uop_134.astype('float64'), relay.reshape(var_179.astype('float64'), relay.shape_of(uop_134))) # shape=(12,)
bop_183 = relay.subtract(bop_176.astype('float32'), relay.reshape(bop_129.astype('float32'), relay.shape_of(bop_176))) # shape=(12,)
uop_186 = relay.sinh(bop_176.astype('float64')) # shape=(12,)
uop_188 = relay.log10(uop_186.astype('float64')) # shape=(12,)
bop_190 = relay.power(uop_186.astype('float32'), relay.reshape(bop_165.astype('float32'), relay.shape_of(uop_186))) # shape=(12,)
uop_193 = relay.cos(uop_188.astype('float32')) # shape=(12,)
uop_195 = relay.exp(uop_188.astype('float64')) # shape=(12,)
bop_197 = relay.multiply(uop_193.astype('uint8'), relay.reshape(bop_180.astype('uint8'), relay.shape_of(uop_193))) # shape=(12,)
bop_200 = relay.mod(uop_193.astype('float64'), relay.reshape(bop_153.astype('float64'), relay.shape_of(uop_193))) # shape=(12,)
uop_203 = relay.tan(uop_188.astype('float32')) # shape=(12,)
uop_205 = relay.cosh(uop_188.astype('float64')) # shape=(12,)
uop_207 = relay.sin(uop_188.astype('float64')) # shape=(12,)
uop_209 = relay.cos(bop_190.astype('float32')) # shape=(12,)
uop_211 = relay.log2(bop_190.astype('float32')) # shape=(12,)
uop_213 = relay.erf(uop_203.astype('float32')) # shape=(12,)
bop_215 = relay.logical_xor(uop_193.astype('int8'), relay.reshape(bop_165.astype('int8'), relay.shape_of(uop_193))) # shape=(12,)
const_218 = relay.const([1.051189,-7.361109,-0.105130,-1.695631,7.242858,0.910412,-5.859486,8.853074,5.455570,0.898098,-5.408462,9.374326], dtype = "float64")#candidate|218|(12,)|const|float64
bop_219 = relay.less(uop_207.astype('bool'), relay.reshape(const_218.astype('bool'), relay.shape_of(uop_207))) # shape=(12,)
bop_222 = relay.multiply(bop_190.astype('uint32'), relay.reshape(bop_180.astype('uint32'), relay.shape_of(bop_190))) # shape=(12,)
uop_225 = relay.rsqrt(uop_207.astype('float32')) # shape=(12,)
uop_227 = relay.exp(uop_225.astype('float64')) # shape=(12,)
bop_229 = relay.logical_xor(uop_227.astype('int32'), relay.reshape(uop_213.astype('int32'), relay.shape_of(uop_227))) # shape=(12,)
uop_232 = relay.sigmoid(bop_229.astype('float64')) # shape=(12,)
const_234 = relay.const([4.426472,-8.837201,-5.199251,-7.761861,4.443476,-4.238053,-7.404779,-4.219379,6.581910,-0.701145,1.704168,-9.495302], dtype = "float64")#candidate|234|(12,)|const|float64
bop_235 = relay.subtract(uop_227.astype('int8'), relay.reshape(const_234.astype('int8'), relay.shape_of(uop_227))) # shape=(12,)
uop_238 = relay.atan(bop_235.astype('float32')) # shape=(12,)
uop_240 = relay.sigmoid(uop_232.astype('float32')) # shape=(12,)
uop_242 = relay.log(uop_240.astype('float32')) # shape=(12,)
output = relay.Tuple([bop_116,uop_124,call_126,uop_139,uop_141,call_143,var_144,bop_156,bop_160,uop_170,bop_172,bop_183,uop_195,bop_197,bop_200,uop_205,uop_209,uop_211,bop_215,bop_219,bop_222,uop_238,uop_242,])
output2 = relay.Tuple([bop_116,uop_124,call_128,uop_139,uop_141,call_145,var_144,bop_156,bop_160,uop_170,bop_172,bop_183,uop_195,bop_197,bop_200,uop_205,uop_209,uop_211,bop_215,bop_219,bop_222,uop_238,uop_242,])
func_244 = relay.Function([var_127,var_144,var_152,var_175,var_179,], output)
mod['func_244'] = func_244
mod = relay.transform.InferType()(mod)
var_245 = relay.var("var_245", dtype = "int32", shape = (8, 12))#candidate|245|(8, 12)|var|int32
var_246 = relay.var("var_246", dtype = "float64", shape = (9,))#candidate|246|(9,)|var|float64
var_247 = relay.var("var_247", dtype = "float64", shape = (12,))#candidate|247|(12,)|var|float64
var_248 = relay.var("var_248", dtype = "float64", shape = (12,))#candidate|248|(12,)|var|float64
var_249 = relay.var("var_249", dtype = "float64", shape = (12,))#candidate|249|(12,)|var|float64
output = func_244(var_245,var_246,var_247,var_248,var_249,)
func_250 = relay.Function([var_245,var_246,var_247,var_248,var_249,], output)
mutated_mod['func_250'] = func_250
mutated_mod = relay.transform.InferType()(mutated_mod)
var_252 = relay.var("var_252", dtype = "int16", shape = (11, 14, 3))#candidate|252|(11, 14, 3)|var|int16
const_253 = relay.const([[[-2,-2,-7],[-1,3,3],[-9,7,-9],[1,-3,8],[-4,1,4],[9,-8,2],[4,-8,-5],[-3,7,8],[-8,8,7],[7,2,1],[-3,4,1],[-1,-1,-4],[-9,3,-8],[4,10,-7]],[[7,8,4],[-7,-8,-2],[-9,4,3],[8,-6,10],[4,1,-3],[2,-4,1],[-10,9,-1],[8,6,-4],[1,-1,-5],[-2,2,-4],[4,8,6],[8,-10,-5],[1,-5,-5],[-6,-2,-7]],[[7,-5,-5],[-7,-5,-6],[6,7,4],[1,3,-10],[4,9,4],[-3,7,3],[9,10,-5],[-4,-2,-1],[1,2,7],[-4,-10,8],[-9,5,1],[6,8,-3],[-2,-7,8],[-3,-7,2]],[[-6,6,-1],[10,-7,6],[4,9,7],[8,1,-8],[-1,-7,-5],[-4,-9,-1],[-5,10,7],[-2,-5,-7],[-7,6,8],[6,-6,-8],[3,-5,2],[-3,-6,-3],[-7,8,-7],[8,7,1]],[[-9,-7,-1],[-7,-8,-7],[8,-9,10],[-9,-6,9],[-1,-9,-1],[6,-7,-7],[9,-3,6],[-7,-7,-9],[8,-4,-5],[10,-9,4],[9,-10,-2],[-5,10,-8],[-9,3,1],[4,9,6]],[[10,8,6],[1,-2,4],[-9,7,-8],[-8,-1,-2],[2,-4,4],[5,-9,-5],[-6,10,7],[-6,-4,8],[9,8,6],[2,6,8],[-9,5,-7],[3,-7,1],[-9,-2,9],[3,-7,-8]],[[1,-10,-8],[-8,-3,-5],[3,4,-2],[-6,4,-2],[-10,-1,-7],[8,8,6],[10,3,5],[6,7,-8],[6,8,1],[-3,8,-8],[2,3,-5],[8,-3,1],[-10,-7,-8],[5,-10,4]],[[-4,2,8],[7,5,-1],[-10,-10,-9],[-2,9,-7],[-6,-10,7],[8,10,5],[2,-9,-3],[1,-7,-6],[-3,7,-10],[8,-9,2],[4,-7,-4],[2,4,6],[10,-9,-10],[5,-7,-5]],[[9,3,-4],[1,-9,-2],[-1,-10,4],[2,-5,2],[8,-7,2],[-2,-7,-8],[-1,1,-9],[4,7,-7],[-2,-6,-10],[2,-6,4],[3,8,10],[3,6,5],[6,7,5],[-4,-1,-4]],[[7,-10,5],[10,8,9],[3,-3,-3],[6,10,-3],[-7,2,5],[10,3,6],[4,7,-5],[3,-6,9],[-5,-9,10],[9,-8,7],[8,9,9],[-1,-3,-4],[5,-8,-4],[-2,-4,5]],[[4,-3,-7],[-8,-10,-6],[2,10,6],[-6,3,-1],[1,-9,4],[-10,10,-1],[-4,1,5],[10,3,-5],[-8,-9,4],[-10,-3,1],[5,-8,-3],[-1,9,10],[9,3,2],[2,-10,-10]]], dtype = "int16")#candidate|253|(11, 14, 3)|const|int16
bop_254 = relay.greater_equal(var_252.astype('bool'), relay.reshape(const_253.astype('bool'), relay.shape_of(var_252))) # shape=(11, 14, 3)
func_109_call = mod.get_global_var('func_109')
func_112_call = mutated_mod.get_global_var('func_112')
const_258 = relay.const([[3.942624,-8.858733],[-8.734007,-7.340216],[-9.195738,5.224218],[-3.117786,7.744213],[9.232985,-8.936107],[-6.138560,-0.751226]], dtype = "float64")#candidate|258|(6, 2)|const|float64
const_259 = relay.const([[False,False,False,False,False,True,True,False,False,True],[False,True,False,False,False,False,False,True,False,True],[False,False,False,False,True,False,True,False,False,False],[True,False,True,True,False,True,True,False,True,True],[True,False,False,True,False,False,False,True,True,False],[False,True,False,True,True,True,True,False,True,True],[True,True,True,False,False,False,True,True,False,True],[True,True,True,True,True,True,False,True,False,True],[True,True,True,True,False,False,True,True,True,False],[False,False,True,True,False,False,True,True,True,True],[False,False,True,True,False,False,True,False,True,True],[True,False,False,False,True,False,True,False,True,False],[True,True,False,False,True,False,False,False,True,False],[False,False,False,True,True,False,True,False,False,False],[False,False,False,False,False,True,True,True,False,False],[True,True,False,True,True,True,True,False,True,True],[False,True,True,False,False,False,True,True,True,True],[True,True,True,True,True,False,True,False,True,True],[True,True,False,True,False,True,False,False,True,True],[True,True,True,False,False,True,False,True,True,True],[False,False,False,False,False,False,False,True,False,True],[False,True,False,False,False,True,True,False,True,True],[False,False,False,True,False,False,False,False,True,True],[False,False,False,False,False,False,False,False,True,True],[True,True,False,False,True,True,True,False,True,False],[True,True,False,False,False,False,False,True,False,False],[False,False,False,False,True,False,False,True,False,True],[False,True,False,True,True,True,False,False,False,True],[False,True,True,False,True,True,False,False,False,True],[False,True,True,False,True,False,False,False,False,False],[True,False,True,False,True,True,True,True,True,True],[True,False,True,False,False,False,True,True,False,False],[False,False,True,True,False,False,False,False,False,False],[False,False,False,False,False,False,True,True,True,False],[True,False,False,False,False,True,False,False,False,True],[False,False,True,True,True,False,True,True,False,False],[True,True,False,False,True,False,False,False,True,True],[False,False,False,True,False,False,False,True,False,False],[False,False,False,True,True,True,True,True,True,False],[True,False,True,True,False,False,True,False,False,False],[False,False,True,False,True,True,False,True,False,True],[True,False,True,True,False,True,True,True,False,False],[True,True,False,False,False,True,False,True,True,False],[True,True,True,False,True,False,False,False,False,False],[True,True,True,False,True,True,True,False,False,True],[False,False,False,False,False,False,True,True,True,True],[True,True,False,False,True,False,True,True,False,True],[False,True,False,True,True,True,False,True,True,False],[False,True,False,True,False,True,True,True,False,True],[False,True,False,True,False,True,True,True,True,True],[False,True,False,False,False,True,False,False,False,True],[False,True,True,True,False,True,False,True,False,False],[False,True,True,False,False,True,False,True,False,True],[False,False,False,False,True,False,True,True,False,False],[False,False,True,True,True,True,False,True,False,True],[True,True,False,True,True,False,False,True,True,True],[False,True,True,False,True,False,False,False,True,False],[False,False,True,True,True,False,False,False,False,True],[True,True,False,True,False,True,True,True,True,False],[False,True,True,True,False,False,False,False,False,True],[True,True,True,False,False,True,False,True,True,False],[False,False,False,False,True,False,True,False,True,False],[False,True,False,True,True,False,True,True,False,True],[False,True,False,False,True,True,True,False,False,False],[False,False,False,False,False,True,True,True,True,False],[False,True,False,False,False,True,False,True,False,False],[False,False,False,True,False,True,False,True,True,False],[True,False,False,True,False,False,False,True,True,True],[False,True,False,False,True,False,True,True,True,True],[False,False,True,False,True,False,True,True,True,True],[True,True,True,True,False,True,True,True,True,False],[False,True,False,True,True,True,True,False,True,True],[True,True,True,False,True,False,False,False,True,False],[True,False,True,False,True,True,False,False,False,True],[False,True,False,False,False,True,False,True,True,True],[False,False,False,True,False,True,True,True,True,False],[True,True,False,False,True,True,True,True,True,False],[False,False,True,False,False,True,True,False,False,False],[True,True,False,True,False,True,False,False,False,False],[False,True,True,True,True,False,False,False,False,False],[False,False,False,True,True,False,False,False,True,True],[True,False,False,True,False,False,False,False,False,False],[False,True,True,True,False,False,False,False,True,False],[False,True,False,False,True,True,True,True,False,False],[False,True,False,False,False,False,True,True,True,True],[True,True,False,True,True,False,True,True,True,False],[True,True,False,True,False,True,True,True,False,True],[True,True,False,True,True,False,False,False,True,True],[False,True,False,False,False,True,True,True,False,False],[False,True,True,False,True,True,True,False,False,True],[True,True,False,True,True,False,True,True,False,True]], dtype = "bool")#candidate|259|(91, 10)|const|bool
call_257 = relay.TupleGetItem(func_109_call(relay.reshape(const_258.astype('float64'), [12, 1]), relay.reshape(const_259.astype('bool'), [5, 14, 13]), ), 2)
call_260 = relay.TupleGetItem(func_112_call(relay.reshape(const_258.astype('float64'), [12, 1]), relay.reshape(const_259.astype('bool'), [5, 14, 13]), ), 2)
bop_261 = relay.left_shift(bop_254.astype('int8'), relay.reshape(var_252.astype('int8'), relay.shape_of(bop_254))) # shape=(11, 14, 3)
bop_264 = relay.power(const_258.astype('float64'), relay.reshape(call_257.astype('float64'), relay.shape_of(const_258))) # shape=(6, 2)
bop_267 = relay.power(const_258.astype('float64'), relay.reshape(call_260.astype('float64'), relay.shape_of(const_258))) # shape=(6, 2)
bop_268 = relay.mod(bop_264.astype('float32'), relay.reshape(const_258.astype('float32'), relay.shape_of(bop_264))) # shape=(6, 2)
bop_271 = relay.mod(bop_267.astype('float32'), relay.reshape(const_258.astype('float32'), relay.shape_of(bop_267))) # shape=(6, 2)
uop_272 = relay.erf(bop_254.astype('float32')) # shape=(11, 14, 3)
bop_274 = relay.power(const_258.astype('float32'), relay.reshape(bop_268.astype('float32'), relay.shape_of(const_258))) # shape=(6, 2)
bop_277 = relay.power(const_258.astype('float32'), relay.reshape(bop_271.astype('float32'), relay.shape_of(const_258))) # shape=(6, 2)
uop_278 = relay.cos(uop_272.astype('float32')) # shape=(11, 14, 3)
func_52_call = mod.get_global_var('func_52')
func_54_call = mutated_mod.get_global_var('func_54')
call_280 = relay.TupleGetItem(func_52_call(), 0)
call_281 = relay.TupleGetItem(func_54_call(), 0)
bop_282 = relay.multiply(uop_278.astype('uint32'), relay.reshape(bop_261.astype('uint32'), relay.shape_of(uop_278))) # shape=(11, 14, 3)
uop_285 = relay.sigmoid(uop_278.astype('float32')) # shape=(11, 14, 3)
uop_287 = relay.exp(uop_272.astype('float64')) # shape=(11, 14, 3)
bop_289 = relay.subtract(call_257.astype('uint16'), relay.reshape(const_258.astype('uint16'), relay.shape_of(call_257))) # shape=(12,)
bop_292 = relay.subtract(call_260.astype('uint16'), relay.reshape(const_258.astype('uint16'), relay.shape_of(call_260))) # shape=(12,)
uop_293 = relay.cosh(uop_285.astype('float32')) # shape=(11, 14, 3)
uop_295 = relay.log10(uop_293.astype('float64')) # shape=(11, 14, 3)
uop_297 = relay.asinh(uop_295.astype('float32')) # shape=(11, 14, 3)
var_299 = relay.var("var_299", dtype = "float64", shape = (11, 14, 3))#candidate|299|(11, 14, 3)|var|float64
bop_300 = relay.less_equal(uop_295.astype('bool'), relay.reshape(var_299.astype('bool'), relay.shape_of(uop_295))) # shape=(11, 14, 3)
bop_303 = relay.not_equal(uop_295.astype('bool'), relay.reshape(bop_261.astype('bool'), relay.shape_of(uop_295))) # shape=(11, 14, 3)
var_306 = relay.var("var_306", dtype = "bool", shape = (11, 14, 3))#candidate|306|(11, 14, 3)|var|bool
bop_307 = relay.bitwise_and(bop_303.astype('uint8'), relay.reshape(var_306.astype('uint8'), relay.shape_of(bop_303))) # shape=(11, 14, 3)
uop_310 = relay.asin(uop_297.astype('float32')) # shape=(11, 14, 3)
bop_312 = relay.greater_equal(uop_297.astype('bool'), relay.reshape(uop_310.astype('bool'), relay.shape_of(uop_297))) # shape=(11, 14, 3)
bop_315 = relay.multiply(bop_312.astype('uint16'), relay.reshape(uop_278.astype('uint16'), relay.shape_of(bop_312))) # shape=(11, 14, 3)
output = relay.Tuple([const_259,bop_274,call_280,bop_282,uop_287,bop_289,bop_300,bop_307,bop_315,])
output2 = relay.Tuple([const_259,bop_277,call_281,bop_282,uop_287,bop_292,bop_300,bop_307,bop_315,])
func_318 = relay.Function([var_252,var_299,var_306,], output)
mod['func_318'] = func_318
mod = relay.transform.InferType()(mod)
var_319 = relay.var("var_319", dtype = "int16", shape = (11, 14, 3))#candidate|319|(11, 14, 3)|var|int16
var_320 = relay.var("var_320", dtype = "float64", shape = (11, 14, 3))#candidate|320|(11, 14, 3)|var|float64
var_321 = relay.var("var_321", dtype = "bool", shape = (11, 14, 3))#candidate|321|(11, 14, 3)|var|bool
output = func_318(var_319,var_320,var_321,)
func_322 = relay.Function([var_319,var_320,var_321,], output)
mutated_mod['func_322'] = func_322
mutated_mod = relay.transform.InferType()(mutated_mod)
const_324 = relay.const(-2.204745, dtype = "float64")#candidate|324|()|const|float64
uop_325 = relay.asin(const_324.astype('float64')) # shape=()
output = uop_325
output2 = uop_325
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