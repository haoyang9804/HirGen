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
var_0 = relay.var("var_0", dtype = "float32", shape = (16,))#candidate|0|(16,)|var|float32
var_1 = relay.var("var_1", dtype = "float32", shape = (16,))#candidate|1|(16,)|var|float32
bop_2 = relay.greater_equal(var_0.astype('bool'), relay.reshape(var_1.astype('bool'), relay.shape_of(var_0))) # shape=(16,)
uop_5 = relay.log(bop_2.astype('float32')) # shape=(16,)
bop_7 = relay.left_shift(uop_5.astype('uint16'), relay.reshape(bop_2.astype('uint16'), relay.shape_of(uop_5))) # shape=(16,)
uop_10 = relay.asin(bop_7.astype('float32')) # shape=(16,)
uop_12 = relay.cos(uop_10.astype('float64')) # shape=(16,)
uop_14 = relay.log10(uop_12.astype('float64')) # shape=(16,)
bop_16 = relay.bitwise_and(uop_14.astype('uint16'), relay.reshape(bop_7.astype('uint16'), relay.shape_of(uop_14))) # shape=(16,)
var_19 = relay.var("var_19", dtype = "uint16", shape = (16,))#candidate|19|(16,)|var|uint16
bop_20 = relay.bitwise_and(bop_16.astype('uint32'), relay.reshape(var_19.astype('uint32'), relay.shape_of(bop_16))) # shape=(16,)
uop_23 = relay.log(uop_12.astype('float64')) # shape=(16,)
uop_25 = relay.rsqrt(bop_20.astype('float32')) # shape=(16,)
uop_27 = relay.sinh(uop_25.astype('float64')) # shape=(16,)
output = relay.Tuple([uop_23,uop_27,])
output2 = relay.Tuple([uop_23,uop_27,])
func_29 = relay.Function([var_0,var_1,var_19,], output)
mod['func_29'] = func_29
mod = relay.transform.InferType()(mod)
var_30 = relay.var("var_30", dtype = "float32", shape = (16,))#candidate|30|(16,)|var|float32
var_31 = relay.var("var_31", dtype = "float32", shape = (16,))#candidate|31|(16,)|var|float32
var_32 = relay.var("var_32", dtype = "uint16", shape = (16,))#candidate|32|(16,)|var|uint16
output = func_29(var_30,var_31,var_32,)
func_33 = relay.Function([var_30,var_31,var_32,], output)
mutated_mod['func_33'] = func_33
mutated_mod = relay.transform.InferType()(mutated_mod)
var_35 = relay.var("var_35", dtype = "uint16", shape = (2, 16, 12))#candidate|35|(2, 16, 12)|var|uint16
var_36 = relay.var("var_36", dtype = "uint16", shape = (2, 16, 12))#candidate|36|(2, 16, 12)|var|uint16
bop_37 = relay.bitwise_xor(var_35.astype('uint16'), relay.reshape(var_36.astype('uint16'), relay.shape_of(var_35))) # shape=(2, 16, 12)
var_40 = relay.var("var_40", dtype = "uint16", shape = (2, 16, 12))#candidate|40|(2, 16, 12)|var|uint16
bop_41 = relay.not_equal(bop_37.astype('bool'), relay.reshape(var_40.astype('bool'), relay.shape_of(bop_37))) # shape=(2, 16, 12)
var_44 = relay.var("var_44", dtype = "uint16", shape = (2, 16, 12))#candidate|44|(2, 16, 12)|var|uint16
bop_45 = relay.mod(var_36.astype('float64'), relay.reshape(var_44.astype('float64'), relay.shape_of(var_36))) # shape=(2, 16, 12)
var_48 = relay.var("var_48", dtype = "uint16", shape = (2, 16, 12))#candidate|48|(2, 16, 12)|var|uint16
bop_49 = relay.power(var_40.astype('float32'), relay.reshape(var_48.astype('float32'), relay.shape_of(var_40))) # shape=(2, 16, 12)
uop_52 = relay.sqrt(bop_41.astype('float64')) # shape=(2, 16, 12)
uop_54 = relay.tan(uop_52.astype('float32')) # shape=(2, 16, 12)
bop_56 = relay.floor_divide(uop_54.astype('float32'), relay.reshape(bop_41.astype('float32'), relay.shape_of(uop_54))) # shape=(2, 16, 12)
bop_59 = relay.equal(var_40.astype('bool'), relay.reshape(var_35.astype('bool'), relay.shape_of(var_40))) # shape=(2, 16, 12)
uop_62 = relay.log10(uop_54.astype('float32')) # shape=(2, 16, 12)
uop_64 = relay.asin(uop_54.astype('float64')) # shape=(2, 16, 12)
uop_66 = relay.sqrt(uop_54.astype('float64')) # shape=(2, 16, 12)
func_29_call = mod.get_global_var('func_29')
func_33_call = mutated_mod.get_global_var('func_33')
var_69 = relay.var("var_69", dtype = "float32", shape = (16,))#candidate|69|(16,)|var|float32
call_68 = relay.TupleGetItem(func_29_call(relay.reshape(var_69.astype('float32'), [16,]), relay.reshape(var_69.astype('float32'), [16,]), relay.reshape(var_69.astype('uint16'), [16,]), ), 1)
call_70 = relay.TupleGetItem(func_33_call(relay.reshape(var_69.astype('float32'), [16,]), relay.reshape(var_69.astype('float32'), [16,]), relay.reshape(var_69.astype('uint16'), [16,]), ), 1)
bop_71 = relay.divide(bop_56.astype('float32'), relay.reshape(var_36.astype('float32'), relay.shape_of(bop_56))) # shape=(2, 16, 12)
output = relay.Tuple([bop_45,bop_49,bop_59,uop_62,uop_64,uop_66,call_68,var_69,bop_71,])
output2 = relay.Tuple([bop_45,bop_49,bop_59,uop_62,uop_64,uop_66,call_70,var_69,bop_71,])
func_74 = relay.Function([var_35,var_36,var_40,var_44,var_48,var_69,], output)
mod['func_74'] = func_74
mod = relay.transform.InferType()(mod)
mutated_mod['func_74'] = func_74
mutated_mod = relay.transform.InferType()(mutated_mod)
func_74_call = mutated_mod.get_global_var('func_74')
var_76 = relay.var("var_76", dtype = "uint16", shape = (2, 16, 12))#candidate|76|(2, 16, 12)|var|uint16
var_77 = relay.var("var_77", dtype = "uint16", shape = (2, 16, 12))#candidate|77|(2, 16, 12)|var|uint16
var_78 = relay.var("var_78", dtype = "uint16", shape = (2, 16, 12))#candidate|78|(2, 16, 12)|var|uint16
var_79 = relay.var("var_79", dtype = "uint16", shape = (2, 16, 12))#candidate|79|(2, 16, 12)|var|uint16
var_80 = relay.var("var_80", dtype = "uint16", shape = (2, 16, 12))#candidate|80|(2, 16, 12)|var|uint16
var_81 = relay.var("var_81", dtype = "float32", shape = (16,))#candidate|81|(16,)|var|float32
call_75 = func_74_call(var_76,var_77,var_78,var_79,var_80,var_81,)
output = call_75
func_82 = relay.Function([var_76,var_77,var_78,var_79,var_80,var_81,], output)
mutated_mod['func_82'] = func_82
mutated_mod = relay.transform.InferType()(mutated_mod)
var_84 = relay.var("var_84", dtype = "float64", shape = (6, 14, 7))#candidate|84|(6, 14, 7)|var|float64
uop_85 = relay.sin(var_84.astype('float64')) # shape=(6, 14, 7)
bop_87 = relay.logical_and(uop_85.astype('bool'), relay.reshape(var_84.astype('bool'), relay.shape_of(uop_85))) # shape=(6, 14, 7)
uop_90 = relay.acos(var_84.astype('float64')) # shape=(6, 14, 7)
uop_92 = relay.atanh(uop_85.astype('float32')) # shape=(6, 14, 7)
uop_94 = relay.acosh(uop_92.astype('float64')) # shape=(6, 14, 7)
uop_96 = relay.erf(uop_94.astype('float64')) # shape=(6, 14, 7)
uop_98 = relay.cosh(uop_92.astype('float32')) # shape=(6, 14, 7)
uop_100 = relay.asin(uop_92.astype('float32')) # shape=(6, 14, 7)
bop_102 = relay.bitwise_xor(uop_92.astype('int32'), relay.reshape(bop_87.astype('int32'), relay.shape_of(uop_92))) # shape=(6, 14, 7)
uop_105 = relay.log2(uop_96.astype('float32')) # shape=(6, 14, 7)
uop_107 = relay.log10(uop_96.astype('float64')) # shape=(6, 14, 7)
const_109 = relay.const([[[2.866637,-0.078274,-2.345748,-1.646769,0.319017,5.165689,-4.407072],[-6.091132,7.528055,-1.544677,4.040466,-0.350679,-1.707915,3.077563],[2.117930,-1.878949,-8.811692,-4.319676,0.154682,-2.349774,-5.860882],[4.049729,-2.480771,-8.325874,-0.628207,-5.904708,-5.781245,5.684280],[3.897868,0.630548,7.321410,9.147819,-5.614772,-9.737490,8.114708],[-7.464654,-0.222174,-7.565588,-7.726763,-3.081717,0.537356,-2.495693],[0.391168,9.570962,3.974967,2.824272,-4.105237,-6.433768,-1.228785],[-4.304501,8.600821,8.933819,2.879041,4.019149,-1.099197,-4.321735],[-7.460192,0.960905,-7.732324,4.088074,0.249017,-7.081127,9.847347],[-0.731224,-2.842950,1.543226,-2.089459,2.047529,3.890751,3.869323],[3.487974,7.302207,-0.774072,3.155604,1.977331,-9.199017,-9.474287],[2.562983,-5.530895,8.774147,-8.886534,-8.567936,-3.499161,-6.101626],[-9.174195,2.897934,6.962306,-2.827742,-7.494355,-2.113688,-3.123579],[-6.630627,-6.528762,-2.375090,-7.287909,3.014875,5.870579,2.099191]],[[9.563240,3.282491,2.064380,7.186538,8.271428,-1.216361,-5.424406],[-8.934051,-4.644049,0.965433,-2.575099,0.862866,8.068597,5.044577],[-6.060774,-9.575224,1.726303,-6.669138,3.743244,4.293356,0.914528],[-9.473012,9.513621,-1.384018,1.228682,-7.342448,9.998911,-7.739171],[-6.747408,4.838367,-0.898550,-3.650122,5.983424,-9.813011,0.533690],[-8.304586,-1.692122,-0.389107,7.708843,1.377783,-8.412464,-9.440829],[2.236746,-3.379419,-6.788136,-6.746499,-6.635609,-7.078955,8.862430],[-5.253058,9.556548,3.706516,9.685728,3.178110,-8.196148,7.654080],[-0.087915,5.072149,9.389461,4.676251,4.478138,-3.106447,-5.499681],[-4.266914,-6.730261,-9.378649,8.517865,1.300094,3.811173,3.405271],[1.665909,-3.619556,-8.927097,2.853229,-0.102726,8.470033,-7.125327],[4.021548,9.089806,0.535674,-5.504122,6.089673,4.615482,-3.114564],[5.522443,9.528008,-7.783036,-0.390325,2.192096,-6.039914,-9.133586],[0.749145,-2.126217,3.423587,-8.027340,1.094761,4.201771,-9.649707]],[[2.324220,1.932946,-8.603745,9.222107,0.411331,4.604388,3.617670],[-1.961347,-1.137464,-3.292504,-4.295809,-6.342645,-6.046898,3.636688],[-9.394704,2.244530,-4.106175,-6.418999,8.132154,9.203980,4.669003],[1.769579,3.931575,-7.396808,1.787680,-7.851640,-4.155549,7.651732],[-8.752475,-0.640229,-6.516927,-1.357082,6.557933,-0.466748,-1.566373],[2.417146,-8.508027,-9.320546,-7.109261,8.373217,6.817451,0.466670],[-8.428660,8.892759,8.615455,2.791957,7.366052,-8.407635,-0.614194],[2.888494,-0.915825,-3.676079,5.568332,-8.544631,6.769876,6.208023],[6.256731,0.964749,2.168321,-6.336456,-9.175798,9.374585,7.113388],[-9.433825,-3.743517,-6.145726,-7.109173,9.606537,2.589319,3.653913],[-9.092450,-9.827174,5.829865,5.266367,-8.957761,0.396004,-5.635354],[-4.456428,-9.431021,7.125129,-5.871531,-1.793433,4.108495,-1.510819],[3.721175,5.368852,-1.661492,-8.638295,7.396431,7.819432,8.134816],[5.615043,3.291102,-4.513773,-1.316596,-4.355396,7.023142,-1.848885]],[[-3.024851,-3.798120,-2.295893,7.699701,4.961446,7.587555,-2.219819],[-1.332113,7.117728,5.828178,4.377870,7.386278,4.660575,-3.145276],[-0.126435,7.397286,3.570403,-4.845047,-3.225785,-1.024621,1.723375],[-0.299625,-0.325995,-9.938268,-4.931035,-8.436003,3.443751,-5.451702],[9.934972,6.201161,4.312510,9.440806,-3.019375,-0.410929,2.524921],[-7.850045,-1.853774,-0.868429,4.978519,0.619061,-7.913150,1.710155],[1.335827,0.467096,-1.364466,-3.751788,4.079780,-8.261073,3.191813],[-6.754726,-4.830322,-8.630912,-8.003438,9.036941,5.629994,4.129491],[6.636471,8.168963,-3.204169,9.450896,7.475590,-4.551430,9.816650],[5.623890,-4.104877,-7.434599,-3.345477,5.673836,6.863944,7.515342],[8.037469,2.679743,7.455183,4.539785,2.585968,4.701565,-2.411316],[8.234647,-2.741755,1.752638,5.259576,4.643458,-8.086124,-5.558984],[-2.870540,-0.602833,-0.941734,-2.267030,-3.220404,-6.061711,-4.004307],[-5.460335,4.297830,8.638205,-5.626256,-8.803884,-1.573685,-6.933260]],[[3.175599,-5.707434,-5.743643,7.696052,-8.918367,-3.514085,-5.238872],[-1.006247,-5.860353,8.721832,8.486231,-3.100417,6.879025,9.371844],[-0.922754,-8.418722,-9.863572,9.555319,7.059736,1.973652,-7.854114],[-4.659785,-2.965974,-7.057236,-1.689877,5.994284,-7.957912,0.374270],[-8.505320,5.311900,-1.271401,-4.177845,-0.271262,6.964383,1.103300],[-1.794960,8.931425,-2.777112,7.958600,-5.407547,-1.935435,9.567590],[3.676969,8.338026,3.485275,7.815014,-1.726112,-9.762620,4.803613],[5.724626,9.938069,3.020638,-1.468388,-5.590932,-5.575623,5.812166],[-9.209287,4.184041,-2.407831,-3.950205,7.102313,1.427054,2.072027],[-0.035801,-0.400147,5.116374,2.599534,-5.772263,-7.951367,-3.890351],[1.758761,-1.064975,-7.431623,-1.054818,-9.317976,9.704891,-3.024673],[9.805494,-6.625743,-0.764439,4.761376,-9.931958,-4.924387,7.710152],[-2.211756,0.201818,3.127221,3.527611,0.725839,-4.334909,-7.407589],[0.229427,9.930214,5.054817,1.338310,-9.609938,7.642773,4.270533]],[[3.139257,9.300628,-2.538485,6.575117,-1.671791,-6.795690,-6.330517],[5.483467,-4.303475,1.280602,4.146363,3.302741,4.503190,-3.465772],[6.251933,-1.784687,-9.388130,-2.927541,5.201987,-8.150213,-4.486900],[4.106551,-9.408984,-2.567305,-3.000877,-6.978812,-8.607059,-4.686174],[0.284873,-3.533258,-5.615287,6.613235,-3.162002,-3.474626,5.625784],[4.113831,5.467358,5.157883,8.898678,-7.843041,5.934998,-5.712700],[-8.639584,8.632750,4.310304,-6.003983,5.811805,-5.308662,9.946044],[1.748938,7.083572,3.146389,4.505725,-6.118209,-3.041083,-3.258540],[7.415672,5.643736,6.913033,6.228711,-6.489902,8.915326,7.023748],[-3.279536,-7.684898,-6.570210,-0.409524,-4.102810,-4.874936,8.683680],[6.026165,-3.796229,-1.645135,-9.077060,3.653398,-7.902482,-5.365792],[0.389028,9.776736,-4.052215,6.541816,2.435736,-9.032002,6.208926],[7.861389,2.583301,0.635287,1.468195,0.362680,7.738452,-2.484698],[8.126290,9.846292,-1.720341,-5.318118,-5.420643,5.629228,6.058008]]], dtype = "float64")#candidate|109|(6, 14, 7)|const|float64
bop_110 = relay.right_shift(uop_107.astype('int16'), relay.reshape(const_109.astype('int16'), relay.shape_of(uop_107))) # shape=(6, 14, 7)
uop_113 = relay.acosh(bop_110.astype('float64')) # shape=(6, 14, 7)
var_115 = relay.var("var_115", dtype = "float64", shape = (6, 14, 7))#candidate|115|(6, 14, 7)|var|float64
bop_116 = relay.divide(uop_107.astype('float64'), relay.reshape(var_115.astype('float64'), relay.shape_of(uop_107))) # shape=(6, 14, 7)
var_119 = relay.var("var_119", dtype = "float64", shape = (6, 14, 7))#candidate|119|(6, 14, 7)|var|float64
bop_120 = relay.right_shift(uop_113.astype('uint8'), relay.reshape(var_119.astype('uint8'), relay.shape_of(uop_113))) # shape=(6, 14, 7)
uop_123 = relay.acos(bop_110.astype('float32')) # shape=(6, 14, 7)
uop_125 = relay.rsqrt(bop_110.astype('float64')) # shape=(6, 14, 7)
bop_127 = relay.maximum(uop_94.astype('int64'), relay.reshape(uop_100.astype('int64'), relay.shape_of(uop_94))) # shape=(6, 14, 7)
bop_130 = relay.floor_divide(uop_113.astype('float64'), relay.reshape(uop_96.astype('float64'), relay.shape_of(uop_113))) # shape=(6, 14, 7)
bop_133 = relay.multiply(bop_120.astype('uint32'), relay.reshape(uop_90.astype('uint32'), relay.shape_of(bop_120))) # shape=(6, 14, 7)
bop_136 = relay.floor_divide(uop_92.astype('float32'), relay.reshape(uop_96.astype('float32'), relay.shape_of(uop_92))) # shape=(6, 14, 7)
uop_139 = relay.exp(uop_125.astype('float32')) # shape=(6, 14, 7)
var_141 = relay.var("var_141", dtype = "float64", shape = (6, 14, 7))#candidate|141|(6, 14, 7)|var|float64
bop_142 = relay.minimum(bop_130.astype('int64'), relay.reshape(var_141.astype('int64'), relay.shape_of(bop_130))) # shape=(6, 14, 7)
uop_145 = relay.sinh(uop_139.astype('float64')) # shape=(6, 14, 7)
uop_147 = relay.erf(bop_110.astype('float32')) # shape=(6, 14, 7)
uop_149 = relay.rsqrt(uop_139.astype('float32')) # shape=(6, 14, 7)
bop_151 = relay.less(uop_123.astype('bool'), relay.reshape(var_119.astype('bool'), relay.shape_of(uop_123))) # shape=(6, 14, 7)
bop_154 = relay.greater_equal(uop_149.astype('bool'), relay.reshape(bop_102.astype('bool'), relay.shape_of(uop_149))) # shape=(6, 14, 7)
bop_157 = relay.less(bop_154.astype('bool'), relay.reshape(uop_90.astype('bool'), relay.shape_of(bop_154))) # shape=(6, 14, 7)
bop_160 = relay.minimum(uop_145.astype('uint64'), relay.reshape(bop_102.astype('uint64'), relay.shape_of(uop_145))) # shape=(6, 14, 7)
uop_163 = relay.atan(uop_123.astype('float64')) # shape=(6, 14, 7)
bop_165 = relay.right_shift(bop_130.astype('uint8'), relay.reshape(bop_87.astype('uint8'), relay.shape_of(bop_130))) # shape=(6, 14, 7)
bop_168 = relay.less_equal(bop_154.astype('bool'), relay.reshape(uop_100.astype('bool'), relay.shape_of(bop_154))) # shape=(6, 14, 7)
const_171 = relay.const([[[-2.886723,3.429163,1.200013,8.060995,-4.852226,5.240637,-7.936284],[0.815148,-6.291400,8.852096,-1.722396,-6.951564,5.199288,-6.537569],[-1.237989,5.825816,9.584229,-2.451950,5.518087,-9.343071,5.893849],[-0.875701,-5.864780,-2.958762,9.455116,-6.847873,9.491260,-5.058846],[-4.170674,-1.356123,-1.570209,9.481592,-4.864752,3.110827,-3.744480],[1.733244,1.367195,0.324587,5.615913,-1.944131,-8.825983,9.225812],[-9.294550,-7.548382,-8.575306,9.523805,0.229548,-8.063408,-2.852128],[9.947297,1.881266,-4.653750,-9.233404,1.126024,-5.143848,-6.041831],[-0.300912,-0.246583,-0.356265,1.054997,0.853058,6.701529,-1.849219],[-6.797253,2.805824,0.552769,-6.348129,-9.120543,2.231173,0.022299],[2.805441,-2.848761,5.425913,-1.240583,-7.812642,-6.758260,8.153647],[-9.572015,-5.555817,-9.396172,-2.037498,-9.264664,-2.436470,8.167707],[-8.696095,9.247802,2.656205,-6.006436,-3.416794,-2.023264,-3.130826],[4.368276,-8.353504,6.917407,-3.652033,6.597499,1.487914,-9.550367]],[[-4.283187,-9.509086,-0.540395,-2.488120,7.103261,-4.434465,-2.057281],[-8.354524,1.477920,-1.043961,1.495064,-2.766885,6.943586,-0.643343],[-9.257054,-0.112956,9.891902,0.663526,-5.827719,3.433173,-0.035779],[-1.330893,-5.842405,-9.348735,-5.293940,5.305817,-7.957031,1.497413],[9.985878,7.668781,-1.909071,-8.966673,-2.335866,3.692095,-2.586537],[-4.279442,-1.778972,3.000290,-9.422014,7.530387,-8.363873,-6.199259],[-2.243287,-6.020932,-3.111502,7.727586,-1.935706,-7.140584,-2.202685],[2.973262,-8.298073,7.618935,-6.113449,9.915509,-9.559687,4.232211],[8.488377,-1.249868,-9.870473,-0.397952,-5.186589,7.835708,4.970601],[9.683205,-4.648075,-0.665865,-9.166509,9.943082,-3.512392,6.695198],[5.186824,8.453759,3.973209,0.450252,8.320908,-3.724119,0.825339],[-4.926408,7.067270,-0.706759,-2.534024,-9.310795,0.389958,-9.455559],[-1.092429,-5.774107,-8.616404,-2.438326,1.715386,7.052615,8.976843],[-5.747539,-9.499532,-0.961402,-6.909534,1.747511,2.649348,-0.834697]],[[4.901374,1.749567,-5.243076,-1.769326,7.842382,-3.057099,-7.964478],[1.166745,4.279064,-4.119405,-9.390092,6.900112,-3.478106,-6.001085],[-7.393588,-2.302573,9.195131,7.165832,7.488429,-7.691083,1.447513],[-2.884817,6.232920,3.685365,4.934741,-2.040374,-3.801962,9.197199],[-3.394689,1.134848,-6.052396,-1.761458,2.189249,3.390921,5.417039],[0.065229,9.362618,-9.153505,-6.581737,7.410102,-3.891564,3.612926],[-8.255480,2.617314,-2.879539,0.024409,-9.899224,-7.142368,-7.953208],[-7.293032,-4.472191,-8.176823,9.768508,-0.853249,-1.399874,-6.197215],[4.285447,-5.283063,-2.374435,-8.850958,-7.937725,7.534234,6.249864],[-3.436562,3.177215,-7.238129,-6.778715,1.738991,1.745008,6.963386],[-5.902608,-1.798380,-9.647187,7.141791,-1.916671,7.886544,7.653608],[6.656615,1.859459,-8.692477,9.136016,-1.264316,4.713404,3.465561],[6.440509,1.099920,-1.036610,7.328719,-0.560425,0.357266,-1.421401],[7.271955,6.992647,-2.172583,-4.496221,-7.956188,0.464642,2.346508]],[[-6.041927,-6.702467,-6.789761,7.870670,0.080565,9.903054,6.661243],[7.300511,4.891791,-3.880589,1.711297,6.742820,3.296074,6.732054],[-9.515824,-7.222882,-1.459069,-4.828984,-5.165033,1.585013,-2.433983],[8.770375,-4.536031,0.161638,7.941850,-1.654413,7.325441,-8.085219],[9.172883,8.598119,-3.781325,5.765657,-0.133492,1.811511,3.028045],[7.573300,6.442053,-0.426314,4.739175,-8.191852,0.762690,4.977888],[-5.990700,1.811804,-7.339121,-3.912347,-4.370388,7.867155,3.059627],[6.393282,8.826279,3.844352,6.615617,-3.443056,1.314373,5.799270],[-0.078111,0.031931,-4.121301,-7.711888,-2.584453,-3.752468,-4.820917],[4.776784,-5.375439,4.326206,1.803048,5.013477,-7.909085,2.321483],[-1.794603,-4.008596,-7.987393,0.929460,-2.004745,7.797962,-2.463249],[-9.285125,1.066987,0.679635,-6.813708,2.004734,4.613600,0.725573],[3.051910,9.616132,9.433944,-2.234530,-6.577862,8.053461,6.129952],[7.809817,-9.108147,1.722303,9.386416,2.383753,2.810010,-1.174286]],[[1.929553,-7.205348,-0.150813,-6.625916,3.660437,1.711974,8.078065],[-7.996568,0.273492,-4.060511,0.956860,2.254545,-8.429035,9.446539],[-5.142676,-0.742252,9.491303,-8.993400,1.887595,4.041201,2.006610],[-9.097389,4.704824,6.863147,-9.502124,-2.568950,4.841714,-4.364604],[6.163456,4.715574,1.088114,8.027653,-1.479644,4.705967,6.160993],[-9.642459,8.698715,9.200568,-1.472055,-0.383934,-6.481521,-5.581122],[-0.674412,-5.052338,-3.794849,9.191697,1.714459,-6.429206,-7.723179],[1.276826,5.234251,-5.730111,-5.135629,2.186245,6.733371,9.982505],[4.866551,9.308350,5.432156,8.059277,-6.375699,1.392101,6.311818],[2.001914,7.674444,5.201572,1.903607,-1.891703,9.613597,-7.138556],[0.249494,2.991222,5.889178,-2.666756,-0.807313,5.318750,3.744764],[-8.275043,-4.764993,-0.117263,8.701203,-1.638509,-3.747821,-8.030248],[7.699698,-8.661051,6.971166,5.146791,-9.251645,9.283030,6.613549],[4.788431,0.048591,6.742109,-4.797735,-9.008703,-0.477012,-7.566905]],[[2.899963,-0.004324,1.670693,8.772126,-1.247389,0.608378,-8.257906],[3.775328,6.551064,-0.976198,0.523285,3.108783,-2.673631,9.703744],[-8.690636,9.777108,-0.452998,-0.101609,-0.735790,-8.731761,-9.963889],[2.629931,-2.285859,-5.430889,-8.042064,4.988285,0.793380,3.397236],[-0.993141,-2.440011,-8.589838,-4.512688,1.144429,-5.267574,-3.598815],[0.946590,-4.641684,8.998363,1.881699,1.786878,3.337068,1.017635],[5.020970,-9.550963,9.300891,-2.388903,-3.776246,0.077463,-8.293694],[6.560912,-7.763248,5.038003,3.293362,-3.727885,8.315187,-2.060517],[-7.082649,2.508444,-0.014110,5.307591,-2.096873,9.857140,-2.856213],[-2.742619,-4.946643,0.571716,-4.443111,-9.393359,3.100886,5.754726],[-4.074538,1.325360,5.022379,-8.161758,-7.830589,-9.821557,0.649757],[8.290273,-2.479156,-9.644860,8.086602,8.090126,2.850757,-1.924144],[-1.221097,2.009365,6.651838,3.444491,4.519420,4.463302,3.454958],[-2.693772,1.917446,-0.553875,2.903515,6.771542,-9.274995,3.761420]]], dtype = "float64")#candidate|171|(6, 14, 7)|const|float64
bop_172 = relay.subtract(uop_163.astype('float32'), relay.reshape(const_171.astype('float32'), relay.shape_of(uop_163))) # shape=(6, 14, 7)
bop_175 = relay.not_equal(bop_160.astype('bool'), relay.reshape(uop_139.astype('bool'), relay.shape_of(bop_160))) # shape=(6, 14, 7)
bop_178 = relay.less(bop_168.astype('bool'), relay.reshape(bop_127.astype('bool'), relay.shape_of(bop_168))) # shape=(6, 14, 7)
uop_181 = relay.sin(bop_172.astype('float64')) # shape=(6, 14, 7)
uop_183 = relay.rsqrt(uop_139.astype('float32')) # shape=(6, 14, 7)
uop_185 = relay.sinh(bop_168.astype('float32')) # shape=(6, 14, 7)
uop_187 = relay.log(uop_185.astype('float64')) # shape=(6, 14, 7)
bop_189 = relay.equal(uop_139.astype('bool'), relay.reshape(bop_102.astype('bool'), relay.shape_of(uop_139))) # shape=(6, 14, 7)
bop_192 = relay.less_equal(uop_187.astype('bool'), relay.reshape(bop_136.astype('bool'), relay.shape_of(uop_187))) # shape=(6, 14, 7)
func_74_call = mod.get_global_var('func_74')
func_82_call = mutated_mod.get_global_var('func_82')
var_196 = relay.var("var_196", dtype = "uint16", shape = (384,))#candidate|196|(384,)|var|uint16
const_197 = relay.const([[3.312723,-1.588678,-5.864581,-4.782249,0.069063,-1.696750,7.658741,-5.399462],[8.285357,-9.639248,-7.246829,-9.835211,0.447101,0.908902,2.997907,-4.875312]], dtype = "float32")#candidate|197|(2, 8)|const|float32
call_195 = relay.TupleGetItem(func_74_call(relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(const_197.astype('float32'), [16,]), ), 4)
call_198 = relay.TupleGetItem(func_82_call(relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(var_196.astype('uint16'), [2, 16, 12]), relay.reshape(const_197.astype('float32'), [16,]), ), 4)
uop_199 = relay.acosh(bop_192.astype('float32')) # shape=(6, 14, 7)
uop_201 = relay.cos(uop_199.astype('float64')) # shape=(6, 14, 7)
uop_203 = relay.cos(uop_199.astype('float64')) # shape=(6, 14, 7)
bop_205 = relay.equal(uop_185.astype('bool'), relay.reshape(uop_123.astype('bool'), relay.shape_of(uop_185))) # shape=(6, 14, 7)
uop_208 = relay.cos(uop_187.astype('float64')) # shape=(6, 14, 7)
uop_210 = relay.log(uop_203.astype('float32')) # shape=(6, 14, 7)
bop_212 = relay.left_shift(uop_203.astype('int8'), relay.reshape(bop_102.astype('int8'), relay.shape_of(uop_203))) # shape=(6, 14, 7)
uop_215 = relay.cosh(uop_201.astype('float64')) # shape=(6, 14, 7)
uop_217 = relay.log10(uop_215.astype('float64')) # shape=(6, 14, 7)
uop_219 = relay.cos(uop_203.astype('float32')) # shape=(6, 14, 7)
func_29_call = mod.get_global_var('func_29')
func_33_call = mutated_mod.get_global_var('func_33')
call_221 = relay.TupleGetItem(func_29_call(relay.reshape(const_197.astype('float32'), [16,]), relay.reshape(const_197.astype('float32'), [16,]), relay.reshape(const_197.astype('uint16'), [16,]), ), 1)
call_222 = relay.TupleGetItem(func_33_call(relay.reshape(const_197.astype('float32'), [16,]), relay.reshape(const_197.astype('float32'), [16,]), relay.reshape(const_197.astype('uint16'), [16,]), ), 1)
var_223 = relay.var("var_223", dtype = "float64", shape = (6, 14, 7))#candidate|223|(6, 14, 7)|var|float64
bop_224 = relay.logical_xor(uop_217.astype('int8'), relay.reshape(var_223.astype('int8'), relay.shape_of(uop_217))) # shape=(6, 14, 7)
uop_227 = relay.exp(uop_203.astype('float64')) # shape=(6, 14, 7)
uop_229 = relay.tan(bop_224.astype('float64')) # shape=(6, 14, 7)
bop_231 = relay.logical_and(bop_192.astype('bool'), relay.reshape(var_223.astype('bool'), relay.shape_of(bop_192))) # shape=(6, 14, 7)
uop_234 = relay.asin(uop_217.astype('float64')) # shape=(6, 14, 7)
bop_236 = relay.equal(uop_229.astype('bool'), relay.reshape(bop_205.astype('bool'), relay.shape_of(uop_229))) # shape=(6, 14, 7)
output = relay.Tuple([uop_98,uop_105,bop_116,bop_133,bop_142,uop_147,bop_151,bop_157,bop_165,bop_175,bop_178,uop_181,uop_183,bop_189,call_195,var_196,const_197,uop_208,uop_210,bop_212,uop_219,call_221,uop_227,bop_231,uop_234,bop_236,])
output2 = relay.Tuple([uop_98,uop_105,bop_116,bop_133,bop_142,uop_147,bop_151,bop_157,bop_165,bop_175,bop_178,uop_181,uop_183,bop_189,call_198,var_196,const_197,uop_208,uop_210,bop_212,uop_219,call_222,uop_227,bop_231,uop_234,bop_236,])
func_239 = relay.Function([var_84,var_115,var_119,var_141,var_196,var_223,], output)
mod['func_239'] = func_239
mod = relay.transform.InferType()(mod)
mutated_mod['func_239'] = func_239
mutated_mod = relay.transform.InferType()(mutated_mod)
func_239_call = mutated_mod.get_global_var('func_239')
var_241 = relay.var("var_241", dtype = "float64", shape = (6, 14, 7))#candidate|241|(6, 14, 7)|var|float64
var_242 = relay.var("var_242", dtype = "float64", shape = (6, 14, 7))#candidate|242|(6, 14, 7)|var|float64
var_243 = relay.var("var_243", dtype = "float64", shape = (6, 14, 7))#candidate|243|(6, 14, 7)|var|float64
var_244 = relay.var("var_244", dtype = "float64", shape = (6, 14, 7))#candidate|244|(6, 14, 7)|var|float64
var_245 = relay.var("var_245", dtype = "uint16", shape = (384,))#candidate|245|(384,)|var|uint16
var_246 = relay.var("var_246", dtype = "float64", shape = (6, 14, 7))#candidate|246|(6, 14, 7)|var|float64
call_240 = func_239_call(var_241,var_242,var_243,var_244,var_245,var_246,)
output = call_240
func_247 = relay.Function([var_241,var_242,var_243,var_244,var_245,var_246,], output)
mutated_mod['func_247'] = func_247
mutated_mod = relay.transform.InferType()(mutated_mod)
var_249 = relay.var("var_249", dtype = "float64", shape = (12, 9, 3))#candidate|249|(12, 9, 3)|var|float64
uop_250 = relay.erf(var_249.astype('float64')) # shape=(12, 9, 3)
const_252 = relay.const([[[2.505271,0.852868,8.345685],[0.485590,-7.942536,7.290519],[-4.040269,2.876112,7.110902],[8.696171,7.685779,6.484083],[-9.895100,6.846288,2.849694],[-0.089282,0.966573,-9.724278],[-4.138117,-6.673209,3.734501],[6.828282,9.440520,2.939080],[5.942277,4.806163,-0.802650]],[[-6.721055,-8.906458,7.070709],[-0.056314,-4.631071,2.078288],[-4.290354,5.976884,9.086353],[-1.612984,-9.141283,5.996624],[-8.652755,-5.780599,2.283400],[7.169884,4.381587,-4.512260],[0.594562,-2.846273,-5.252406],[-2.876320,5.379347,-9.955016],[4.398770,3.792256,-2.908949]],[[5.482699,-2.085265,-2.763859],[-7.159528,8.613344,9.964536],[7.530897,-1.995020,5.695628],[7.377854,8.356465,4.701345],[5.461904,-1.098281,-9.954229],[0.928636,3.307768,9.518693],[-0.052157,6.457211,2.564553],[9.802776,1.431759,-0.046170],[-5.075200,9.985653,-8.819739]],[[0.008340,9.865641,3.426094],[-8.761857,7.884643,-6.059803],[8.494186,-5.951206,-9.107258],[-6.132996,-8.344327,4.562470],[-4.697378,7.471661,5.570650],[-4.794021,7.171811,-8.205674],[0.750087,-6.350500,-3.915760],[3.201062,8.756219,6.188200],[3.306052,2.871423,7.820762]],[[-9.345891,-4.687598,8.944932],[9.491106,1.543064,3.157954],[7.835761,-3.084089,0.514467],[-4.142707,2.629228,4.503365],[7.867974,-1.029474,-0.093481],[7.326462,3.296055,-3.503285],[4.181577,-2.469223,4.905023],[-5.336542,-2.417849,-5.435166],[-7.046645,-4.901569,-8.910571]],[[-6.914030,-3.391643,-2.743754],[-4.377366,5.815801,9.973074],[4.978403,3.190791,7.159823],[7.776034,-6.267351,6.055322],[-0.780162,7.558485,1.678178],[-9.585096,-6.363966,-8.788121],[7.396294,-2.953841,-8.018109],[9.710441,1.917135,2.603128],[9.832459,-7.970478,-8.872067]],[[-8.652123,7.739119,7.449920],[4.727723,-7.636011,-5.159056],[-9.912318,-9.863953,-9.896016],[-3.919968,-1.065769,-6.254914],[4.859842,-7.900265,6.219310],[8.807918,7.442837,-4.700191],[2.806321,-8.156368,6.249145],[-7.474781,2.835814,-2.889148],[4.396306,4.973318,-2.690506]],[[6.431355,8.743050,-5.309465],[-3.198036,-8.185445,-9.098348],[-7.700633,-0.646888,-4.176207],[9.678474,4.088473,6.010260],[-1.227938,-5.872408,-3.956757],[0.298838,-3.104994,0.251819],[-6.795170,-2.112487,-3.081065],[7.815743,-5.813961,1.610477],[4.962945,4.834314,9.348989]],[[-8.696023,1.689705,8.407172],[-6.612152,7.447137,5.747206],[3.699865,-7.213775,-9.000386],[-8.193156,3.094407,7.066800],[5.043693,1.142002,8.870053],[9.313395,-4.101415,3.192082],[-5.097426,9.987668,-6.477453],[5.134164,-2.296083,-6.525501],[-4.499768,-8.074355,1.431332]],[[-5.925163,-2.400437,-9.595615],[3.129780,1.628978,-1.453649],[1.975662,-6.318360,5.998816],[-3.824604,-9.776471,-7.173488],[5.563114,-4.242994,2.368948],[-8.158574,-2.578015,5.359862],[-0.259388,4.835955,7.323559],[-7.439833,-8.810789,0.580456],[-5.338304,-0.447551,-7.975751]],[[8.843435,5.172301,2.599339],[9.792398,6.170397,-5.715943],[-5.672491,4.717049,-2.291314],[-6.348654,6.122419,-5.965750],[-1.668730,-7.485356,4.850476],[0.759284,3.182558,-8.291525],[5.536834,-5.709329,-1.090531],[-3.560573,8.412865,-2.228701],[-0.386254,0.399498,3.769963]],[[-7.931400,-1.494334,8.622381],[9.569384,-0.592317,0.339543],[9.594199,-5.029351,4.766997],[-0.708185,-2.380855,4.568046],[5.612059,6.021329,-4.776005],[-6.316406,-9.398546,3.997658],[6.819148,-2.626164,1.272649],[-5.042832,7.406961,-4.302801],[5.707443,-5.688948,0.600072]]], dtype = "float64")#candidate|252|(12, 9, 3)|const|float64
bop_253 = relay.maximum(uop_250.astype('float32'), relay.reshape(const_252.astype('float32'), relay.shape_of(uop_250))) # shape=(12, 9, 3)
uop_256 = relay.acos(uop_250.astype('float64')) # shape=(12, 9, 3)
bop_258 = relay.right_shift(uop_250.astype('int16'), relay.reshape(const_252.astype('int16'), relay.shape_of(uop_250))) # shape=(12, 9, 3)
uop_261 = relay.sqrt(uop_256.astype('float64')) # shape=(12, 9, 3)
func_239_call = mod.get_global_var('func_239')
func_247_call = mutated_mod.get_global_var('func_247')
var_264 = relay.var("var_264", dtype = "float64", shape = (14, 42))#candidate|264|(14, 42)|var|float64
var_265 = relay.var("var_265", dtype = "uint16", shape = (6, 64))#candidate|265|(6, 64)|var|uint16
call_263 = relay.TupleGetItem(func_239_call(relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_265.astype('uint16'), [384,]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), ), 8)
call_266 = relay.TupleGetItem(func_247_call(relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), relay.reshape(var_265.astype('uint16'), [384,]), relay.reshape(var_264.astype('float64'), [6, 14, 7]), ), 8)
uop_267 = relay.rsqrt(uop_261.astype('float32')) # shape=(12, 9, 3)
uop_269 = relay.asin(uop_267.astype('float64')) # shape=(12, 9, 3)
var_271 = relay.var("var_271", dtype = "float64", shape = (12, 9, 3))#candidate|271|(12, 9, 3)|var|float64
bop_272 = relay.right_shift(uop_269.astype('uint16'), relay.reshape(var_271.astype('uint16'), relay.shape_of(uop_269))) # shape=(12, 9, 3)
uop_275 = relay.tan(bop_272.astype('float64')) # shape=(12, 9, 3)
func_29_call = mod.get_global_var('func_29')
func_33_call = mutated_mod.get_global_var('func_33')
var_278 = relay.var("var_278", dtype = "float32", shape = (16,))#candidate|278|(16,)|var|float32
call_277 = relay.TupleGetItem(func_29_call(relay.reshape(var_278.astype('float32'), [16,]), relay.reshape(var_278.astype('float32'), [16,]), relay.reshape(var_278.astype('uint16'), [16,]), ), 1)
call_279 = relay.TupleGetItem(func_33_call(relay.reshape(var_278.astype('float32'), [16,]), relay.reshape(var_278.astype('float32'), [16,]), relay.reshape(var_278.astype('uint16'), [16,]), ), 1)
bop_280 = relay.not_equal(uop_267.astype('bool'), relay.reshape(bop_258.astype('bool'), relay.shape_of(uop_267))) # shape=(12, 9, 3)
var_283 = relay.var("var_283", dtype = "float64", shape = (12, 9, 3))#candidate|283|(12, 9, 3)|var|float64
bop_284 = relay.multiply(uop_275.astype('int8'), relay.reshape(var_283.astype('int8'), relay.shape_of(uop_275))) # shape=(12, 9, 3)
uop_287 = relay.log2(uop_256.astype('float64')) # shape=(12, 9, 3)
uop_289 = relay.tan(uop_275.astype('float64')) # shape=(12, 9, 3)
bop_291 = relay.bitwise_and(bop_280.astype('int32'), relay.reshape(var_249.astype('int32'), relay.shape_of(bop_280))) # shape=(12, 9, 3)
output = relay.Tuple([bop_253,call_263,var_264,var_265,call_277,var_278,bop_284,uop_287,uop_289,bop_291,])
output2 = relay.Tuple([bop_253,call_266,var_264,var_265,call_279,var_278,bop_284,uop_287,uop_289,bop_291,])
func_294 = relay.Function([var_249,var_264,var_265,var_271,var_278,var_283,], output)
mod['func_294'] = func_294
mod = relay.transform.InferType()(mod)
mutated_mod['func_294'] = func_294
mutated_mod = relay.transform.InferType()(mutated_mod)
func_294_call = mutated_mod.get_global_var('func_294')
var_296 = relay.var("var_296", dtype = "float64", shape = (12, 9, 3))#candidate|296|(12, 9, 3)|var|float64
var_297 = relay.var("var_297", dtype = "float64", shape = (14, 42))#candidate|297|(14, 42)|var|float64
var_298 = relay.var("var_298", dtype = "uint16", shape = (6, 64))#candidate|298|(6, 64)|var|uint16
var_299 = relay.var("var_299", dtype = "float64", shape = (12, 9, 3))#candidate|299|(12, 9, 3)|var|float64
var_300 = relay.var("var_300", dtype = "float32", shape = (16,))#candidate|300|(16,)|var|float32
var_301 = relay.var("var_301", dtype = "float64", shape = (12, 9, 3))#candidate|301|(12, 9, 3)|var|float64
call_295 = func_294_call(var_296,var_297,var_298,var_299,var_300,var_301,)
output = call_295
func_302 = relay.Function([var_296,var_297,var_298,var_299,var_300,var_301,], output)
mutated_mod['func_302'] = func_302
mutated_mod = relay.transform.InferType()(mutated_mod)
const_304 = relay.const([[[5.733848,3.989467,-5.933769,-5.385272,-2.301422,-6.012902,-3.832075,-9.282674,-2.837059,4.701210,-1.487074],[0.106351,-5.218504,-6.063045,3.921889,-7.615932,1.084610,9.625378,4.539004,-0.193127,0.898373,-5.164708],[-2.916785,0.853391,3.033994,5.991400,-9.935853,2.886021,5.089749,-0.421727,7.168151,4.407469,3.005628],[4.364286,8.567561,-0.664944,-1.560896,-7.221522,0.249640,8.977292,2.009762,-7.078904,-1.143654,-4.103358]],[[3.733563,-3.814811,4.092925,6.393685,8.473407,2.562024,6.304170,5.396178,-0.198630,-4.737507,7.663167],[7.427130,-1.886044,-4.064883,1.420519,-4.905769,-2.214442,-9.404162,2.830584,-8.536107,-8.226674,1.316009],[4.447939,5.056196,-1.654053,4.349813,1.190230,-1.675954,-2.456341,-5.210440,4.087647,-3.114714,-8.459845],[5.983457,3.984923,-7.463624,-7.866767,0.929623,-6.673624,1.289579,-2.600069,-5.930964,8.167148,9.232379]],[[4.557223,-3.991860,7.268196,-3.637249,7.352602,6.447049,5.141101,-8.213246,7.716148,1.221837,-0.454409],[-3.056552,5.143963,-6.019094,-1.910825,-1.634291,-1.112886,-9.307485,-4.350501,0.099150,5.979016,-6.631579],[-1.563449,-5.860796,-7.644917,8.902140,-8.779385,1.202493,-4.169419,9.352179,0.570843,-5.642139,-4.596950],[3.562492,-2.077353,-1.164397,6.044938,-0.540615,2.628699,8.893859,9.355531,-6.770356,3.682450,0.418598]]], dtype = "float32")#candidate|304|(3, 4, 11)|const|float32
var_305 = relay.var("var_305", dtype = "float32", shape = (3, 4, 11))#candidate|305|(3, 4, 11)|var|float32
bop_306 = relay.floor_divide(const_304.astype('float32'), relay.reshape(var_305.astype('float32'), relay.shape_of(const_304))) # shape=(3, 4, 11)
bop_309 = relay.power(bop_306.astype('float32'), relay.reshape(const_304.astype('float32'), relay.shape_of(bop_306))) # shape=(3, 4, 11)
uop_312 = relay.cosh(bop_306.astype('float32')) # shape=(3, 4, 11)
bop_314 = relay.multiply(bop_309.astype('uint8'), relay.reshape(bop_306.astype('uint8'), relay.shape_of(bop_309))) # shape=(3, 4, 11)
bop_317 = relay.minimum(uop_312.astype('float64'), relay.reshape(var_305.astype('float64'), relay.shape_of(uop_312))) # shape=(3, 4, 11)
output = relay.Tuple([bop_314,bop_317,])
output2 = relay.Tuple([bop_314,bop_317,])
F = relay.Function([var_305,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_305,], output2)
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
input_305= np.array([[[4.864013,-7.465279,1.043310,3.156037,1.605255,3.270799,9.405293,9.473787,-4.564599,-6.085510,-3.708225],[-0.354984,5.001284,6.338284,1.301002,0.401967,-2.125706,-7.595184,1.964428,2.183221,9.058855,-9.284120],[2.776824,1.721631,-5.888309,-6.687819,-6.683601,1.255683,1.990422,1.377338,5.905601,4.360624,-6.785054],[2.304416,-2.823657,-8.913715,2.544392,-7.046099,-8.869897,-0.598121,1.530759,0.966042,5.568068,3.011281]],[[-2.922793,1.976837,-0.884965,2.146082,-1.594640,-4.323558,-1.670327,-7.191355,7.188726,-6.498409,4.625643],[-6.655832,-4.150712,-5.920290,-7.496944,-9.342868,-9.296257,-2.771024,8.519270,5.613068,-9.263914,-7.626514],[-5.701667,-6.777408,3.473209,7.480305,9.640016,8.169494,-7.183222,9.795699,-0.792800,5.359019,1.488561],[3.948488,7.519261,-6.207351,-1.313686,-5.665898,-9.128224,-0.211741,-6.326832,-1.714271,8.225796,8.467891]],[[6.430390,-1.046957,-2.783296,-4.223340,-5.122021,-3.727257,-7.794717,4.517946,7.890939,6.957199,5.719082],[6.381409,4.516013,3.016484,-3.430516,-7.270028,-5.255493,6.030600,-5.089405,-2.986201,0.670115,8.178008],[-9.282580,-6.844722,-8.339284,8.599352,9.510101,8.157191,-8.394610,0.175665,3.443868,-6.877403,-0.717939],[9.142979,5.003227,9.535651,-2.015481,-7.553769,-1.681917,5.233454,-1.209614,-2.519770,2.883157,-5.328031]]], dtype='float32')
module1.set_input('var_305', input_305)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_305, )
res3 = intrp3.evaluate()(input_305, )
res4 = intrp4.evaluate()(input_305, )
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
module5.set_input('var_305', input_305)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_305, )
res7 = intrp7.evaluate()(input_305, )
res8 = intrp8.evaluate()(input_305, )
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
module9.set_input('var_305', input_305)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_305, )
res11 = intrp11.evaluate()(input_305, )
res12 = intrp12.evaluate()(input_305, )
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
module13.set_input('var_305', input_305)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_305, )
res15 = intrp15.evaluate()(input_305, )
res16 = intrp16.evaluate()(input_305, )
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
module17.set_input('var_305', input_305)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_305, )
res19 = intrp19.evaluate()(input_305, )
res20 = intrp20.evaluate()(input_305, )
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
module21.set_input('var_305', input_305)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_305, )
res23 = intrp23.evaluate()(input_305, )
res24 = intrp24.evaluate()(input_305, )
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

'''49: TVMFuncCall
48: _ZNSt17_Function_handlerIFvN3tvm7run
47: tvm::runtime::TypedPackedFunc<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>::AssignTypedLambda<tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)>(tvm::runtime::TypedPackedFunc<tvm::runtime::ObjectRef (tvm::runtime::Array<tvm::RelayExpr, void>)> (*)(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
46: tvm::relay::EvalFunction(tvm::IRModule, tvm::RelayExpr, DLDevice, tvm::Target)
45: tvm::relay::Prepare(tvm::IRModule, tvm::CompilationConfig)
44: tvm::transform::Pass::operator()(tvm::IRModule) const
43: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
42: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
41: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
40: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
39: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
38: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
37: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTEPass(tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
36: tvm::relay::tec::LowerTE(tvm::IRModule const&, tvm::runtime::String const&, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)
35: tvm::transform::Pass::operator()(tvm::IRModule) const
34: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
33: tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
32: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::relay::Function (tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::tec::LowerTensorExpr(tvm::runtime::String const&, tvm::relay::tec::TECompiler, std::function<void (tvm::BaseFunc)>, tvm::VirtualDevice)::{lambda(tvm::relay::Function, tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
31: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
30: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
29: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
28: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
27: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
26: _ZN3tvm5relay9transform22Devic
25: tvm::relay::ExprMutator::VisitExpr_(tvm::relay::FunctionNode const*)
24: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
23: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
22: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
21: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::LetNode const*)
20: tvm::relay::tec::LowerTensorExprMutator::PreVisitLetBinding_(tvm::relay::Var const&, tvm::RelayExpr const&)
19: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
16: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
15: tvm::relay::ExprMutator::VisitExpr(tvm::RelayExpr const&)
14: tvm::relay::ExprFunctor<tvm::RelayExpr (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
13: _ZZN3tvm5relay11ExprFunctorIFNS_9RelayEx
12: tvm::relay::transform::DeviceAwareExprMutator::VisitExpr_(tvm::relay::CallNode const*)
11: tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::tec::LowerTensorExprMutator::MakeLoweredCall(tvm::relay::Function, tvm::runtime::Array<tvm::RelayExpr, void>, tvm::Span, tvm::Target)
9: tvm::relay::tec::TECompilerImpl::LowerShapeFunc(tvm::relay::tec::CCacheKey const&)
8: tvm::relay::tec::TECompilerImpl::LowerShapeFuncInternal(tvm::relay::tec::CCacheKey const&)
7: tvm::relay::tec::ShapeFuncFor(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
6: tvm::relay::tec::MakeShapeFunc::Create(tvm::relay::Function const&, tvm::Target const&, std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > (std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>)
5: tvm::relay::tec::MakeShapeFunc::VisitExpr(tvm::RelayExpr const&)
4: tvm::relay::backend::MemoizedExprTranslator<tvm::runtime::Array<tvm::te::Tensor, void> >::VisitExpr(tvm::RelayExpr const&)
3: tvm::relay::ExprFunctor<tvm::runtime::Array<tvm::te::Tensor, void> (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
2: _ZZN3tvm5relay11ExprFunctorIFNS_7runtime
1: tvm::relay::tec::MakeShapeFunc::VisitExpr_(tvm::relay::CallNode const*)
0: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), TVMFuncCreateFromCFunc::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
3: TVMFuncCall
2: _ZNSt17_Function_handlerIFvN3tvm7run
1: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
0: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
4: TVMFuncCall
3: _ZNSt17_Function_handlerIFvN3tvm7run
2: tvm::runtime::TypedPackedFunc<tvm::tir::ProducerLoad (tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)>::AssignTypedLambda<tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}>(tvm::tir::{lambda(tvm::tir::DataProducer, tvm::runtime::Array<tvm::PrimExpr, void>, tvm::Span)#103}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const, tvm::runtime::TVMRetValue) const
1: tvm::runtime::TVMMovableArgValueWithContext_::operator tvm::runtime::Array<tvm::PrimExpr, void><tvm::runtime::Array<tvm::PrimExpr, void> >() const
0: tvm::runtime::Array<tvm::PrimExpr, void> tvm::runtime::TVMPODValue_::AsObjectRef<tvm::runtime::Array<tvm::PrimExpr, void> >() const

'''