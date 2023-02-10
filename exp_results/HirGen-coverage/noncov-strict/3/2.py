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
var_0 = relay.var("var_0", dtype = "float32", shape = (9, 1, 7))#candidate|0|(9, 1, 7)|var|float32
var_1 = relay.var("var_1", dtype = "float32", shape = (9, 2, 7))#candidate|1|(9, 2, 7)|var|float32
bop_2 = relay.add(var_0.astype('float32'), var_1.astype('float32')) # shape=(9, 2, 7)
bop_5 = relay.logical_xor(var_0.astype('uint32'), var_1.astype('uint32')) # shape=(9, 2, 7)
const_8 = relay.const([[[8,2,1,3,-3,6,10],[-1,4,8,1,4,-5,-10]],[[9,-3,-5,7,3,6,8],[-5,3,-9,-8,-6,-8,-2]],[[-9,-1,6,7,3,9,-3],[1,-9,2,-4,1,-9,-4]],[[-9,-9,7,-1,6,-6,4],[-4,-4,-10,3,7,4,-1]],[[6,9,10,-6,-2,3,4],[6,7,-5,-4,4,-7,-5]],[[8,6,-9,-1,-5,-7,-3],[6,2,-7,3,-9,-9,6]],[[-5,-8,7,9,1,7,7],[4,6,1,-1,6,2,-2]],[[-10,8,-9,8,-2,-2,-7],[8,6,-5,-9,2,-5,-3]],[[6,10,2,-1,8,-7,-6],[-1,1,-6,1,-1,6,-2]]], dtype = "uint32")#candidate|8|(9, 2, 7)|const|uint32
bop_9 = relay.divide(bop_5.astype('float32'), relay.reshape(const_8.astype('float32'), relay.shape_of(bop_5))) # shape=(9, 2, 7)
uop_12 = relay.erf(var_0.astype('float64')) # shape=(9, 1, 7)
uop_14 = relay.cosh(bop_2.astype('float64')) # shape=(9, 2, 7)
bop_16 = relay.power(uop_14.astype('float64'), relay.reshape(bop_5.astype('float64'), relay.shape_of(uop_14))) # shape=(9, 2, 7)
bop_19 = relay.bitwise_or(var_1.astype('uint64'), relay.reshape(bop_2.astype('uint64'), relay.shape_of(var_1))) # shape=(9, 2, 7)
uop_22 = relay.cos(bop_16.astype('float32')) # shape=(9, 2, 7)
uop_24 = relay.log(uop_14.astype('float64')) # shape=(9, 2, 7)
uop_26 = relay.cos(uop_22.astype('float64')) # shape=(9, 2, 7)
bop_28 = relay.equal(uop_24.astype('bool'), relay.reshape(uop_14.astype('bool'), relay.shape_of(uop_24))) # shape=(9, 2, 7)
bop_31 = relay.minimum(uop_26.astype('uint8'), relay.reshape(bop_9.astype('uint8'), relay.shape_of(uop_26))) # shape=(9, 2, 7)
bop_34 = relay.floor_mod(bop_31.astype('float32'), relay.reshape(var_1.astype('float32'), relay.shape_of(bop_31))) # shape=(9, 2, 7)
uop_37 = relay.acos(bop_31.astype('float32')) # shape=(9, 2, 7)
uop_39 = relay.cos(uop_37.astype('float32')) # shape=(9, 2, 7)
uop_41 = relay.log10(uop_39.astype('float64')) # shape=(9, 2, 7)
var_43 = relay.var("var_43", dtype = "float64", shape = (9, 2, 7))#candidate|43|(9, 2, 7)|var|float64
bop_44 = relay.not_equal(uop_41.astype('bool'), relay.reshape(var_43.astype('bool'), relay.shape_of(uop_41))) # shape=(9, 2, 7)
var_47 = relay.var("var_47", dtype = "float32", shape = (9, 2, 7))#candidate|47|(9, 2, 7)|var|float32
bop_48 = relay.greater(uop_37.astype('bool'), relay.reshape(var_47.astype('bool'), relay.shape_of(uop_37))) # shape=(9, 2, 7)
uop_51 = relay.atanh(uop_39.astype('float64')) # shape=(9, 2, 7)
uop_53 = relay.atan(bop_9.astype('float64')) # shape=(9, 2, 7)
var_55 = relay.var("var_55", dtype = "float64", shape = (9, 2, 7))#candidate|55|(9, 2, 7)|var|float64
bop_56 = relay.less_equal(uop_41.astype('bool'), relay.reshape(var_55.astype('bool'), relay.shape_of(uop_41))) # shape=(9, 2, 7)
bop_59 = relay.subtract(bop_56.astype('float64'), relay.reshape(uop_26.astype('float64'), relay.shape_of(bop_56))) # shape=(9, 2, 7)
bop_62 = relay.floor_divide(uop_41.astype('float64'), relay.reshape(var_47.astype('float64'), relay.shape_of(uop_41))) # shape=(9, 2, 7)
var_65 = relay.var("var_65", dtype = "float32", shape = (9, 2, 7))#candidate|65|(9, 2, 7)|var|float32
bop_66 = relay.logical_xor(uop_37.astype('uint32'), relay.reshape(var_65.astype('uint32'), relay.shape_of(uop_37))) # shape=(9, 2, 7)
uop_69 = relay.exp(uop_39.astype('float32')) # shape=(9, 2, 7)
const_71 = relay.const([[[-3.124170,3.237198,8.859323,8.048868,7.261147,7.025398,2.853491],[-0.151416,-5.366542,2.822628,6.449180,4.108272,-1.079807,2.310109]],[[-3.886981,0.205283,6.548773,-6.423014,7.982400,7.708947,-8.499075],[-8.008479,-7.313512,9.241325,3.370963,5.211999,6.315200,-6.731858]],[[1.859317,-2.125150,-4.159807,1.838997,-9.553661,-2.790186,2.989250],[-1.428473,5.823516,7.166840,-0.683921,-6.060193,-1.457722,-7.859731]],[[-9.446008,2.350306,-4.727982,7.691245,-1.033972,-1.635072,-7.960296],[-6.321171,-8.609156,2.049466,8.910452,5.549056,5.862839,7.823481]],[[-1.440537,9.841043,-8.288400,-7.719075,-5.286257,-3.671190,-1.588656],[-9.770588,6.092516,-7.232085,8.427604,4.313824,2.821209,-3.436238]],[[-6.781835,-0.733320,-5.651633,3.522925,-2.901382,5.150997,9.953904],[0.667074,3.298072,-2.814953,-7.392653,-3.518951,0.764534,-9.353186]],[[7.322459,3.431365,-0.074154,-8.980956,-4.233261,-8.991486,5.804190],[6.909922,7.896933,2.927473,-2.419487,3.230328,-1.002174,2.249465]],[[-1.420785,-8.644105,-9.426136,2.896007,7.756150,6.632369,-0.389217],[1.352607,-7.917258,-0.633425,7.740377,-3.397904,-4.048891,7.534410]],[[9.877619,-4.386100,-3.144113,0.081475,-5.914321,-1.112364,-0.121169],[4.398087,2.714362,8.030680,-1.951935,0.177902,3.544992,9.765827]]], dtype = "float32")#candidate|71|(9, 2, 7)|const|float32
bop_72 = relay.floor_divide(uop_39.astype('float64'), relay.reshape(const_71.astype('float64'), relay.shape_of(uop_39))) # shape=(9, 2, 7)
output = relay.Tuple([uop_12,bop_19,bop_28,bop_34,bop_44,bop_48,uop_51,uop_53,bop_59,bop_62,bop_66,uop_69,bop_72,])
output2 = relay.Tuple([uop_12,bop_19,bop_28,bop_34,bop_44,bop_48,uop_51,uop_53,bop_59,bop_62,bop_66,uop_69,bop_72,])
func_75 = relay.Function([var_0,var_1,var_43,var_47,var_55,var_65,], output)
mod['func_75'] = func_75
mod = relay.transform.InferType()(mod)
mutated_mod['func_75'] = func_75
mutated_mod = relay.transform.InferType()(mutated_mod)
func_75_call = mutated_mod.get_global_var('func_75')
var_77 = relay.var("var_77", dtype = "float32", shape = (9, 1, 7))#candidate|77|(9, 1, 7)|var|float32
var_78 = relay.var("var_78", dtype = "float32", shape = (9, 2, 7))#candidate|78|(9, 2, 7)|var|float32
var_79 = relay.var("var_79", dtype = "float64", shape = (9, 2, 7))#candidate|79|(9, 2, 7)|var|float64
var_80 = relay.var("var_80", dtype = "float32", shape = (9, 2, 7))#candidate|80|(9, 2, 7)|var|float32
var_81 = relay.var("var_81", dtype = "float64", shape = (9, 2, 7))#candidate|81|(9, 2, 7)|var|float64
var_82 = relay.var("var_82", dtype = "float32", shape = (9, 2, 7))#candidate|82|(9, 2, 7)|var|float32
call_76 = func_75_call(var_77,var_78,var_79,var_80,var_81,var_82,)
output = call_76
func_83 = relay.Function([var_77,var_78,var_79,var_80,var_81,var_82,], output)
mutated_mod['func_83'] = func_83
mutated_mod = relay.transform.InferType()(mutated_mod)
var_85 = relay.var("var_85", dtype = "float32", shape = (5, 14))#candidate|85|(5, 14)|var|float32
var_86 = relay.var("var_86", dtype = "float32", shape = (5, 14))#candidate|86|(5, 14)|var|float32
bop_87 = relay.power(var_85.astype('float32'), relay.reshape(var_86.astype('float32'), relay.shape_of(var_85))) # shape=(5, 14)
uop_90 = relay.sqrt(var_86.astype('float64')) # shape=(5, 14)
uop_92 = relay.sinh(uop_90.astype('float32')) # shape=(5, 14)
uop_94 = relay.asinh(uop_90.astype('float64')) # shape=(5, 14)
bop_96 = relay.multiply(uop_94.astype('uint8'), relay.reshape(uop_90.astype('uint8'), relay.shape_of(uop_94))) # shape=(5, 14)
uop_99 = relay.acos(uop_90.astype('float64')) # shape=(5, 14)
uop_101 = relay.erf(uop_94.astype('float64')) # shape=(5, 14)
var_103 = relay.var("var_103", dtype = "float32", shape = (5, 14))#candidate|103|(5, 14)|var|float32
bop_104 = relay.minimum(uop_92.astype('float64'), relay.reshape(var_103.astype('float64'), relay.shape_of(uop_92))) # shape=(5, 14)
uop_107 = relay.sigmoid(bop_96.astype('float64')) # shape=(5, 14)
uop_109 = relay.atan(var_85.astype('float64')) # shape=(5, 14)
bop_111 = relay.minimum(bop_96.astype('float64'), relay.reshape(uop_92.astype('float64'), relay.shape_of(bop_96))) # shape=(5, 14)
uop_114 = relay.sqrt(uop_107.astype('float32')) # shape=(5, 14)
uop_116 = relay.cosh(uop_114.astype('float64')) # shape=(5, 14)
bop_118 = relay.logical_xor(uop_116.astype('int16'), relay.reshape(uop_99.astype('int16'), relay.shape_of(uop_116))) # shape=(5, 14)
func_75_call = mod.get_global_var('func_75')
func_83_call = mutated_mod.get_global_var('func_83')
const_122 = relay.const([[-4.935833],[5.396381],[-4.959878],[-1.134614],[1.560967],[-5.310617],[-3.399529],[0.584159],[4.874881],[-9.519528],[-5.900125],[-5.918051],[5.840560],[-9.341062],[3.393644],[1.941198],[3.281114],[1.470949],[-1.391826],[-6.749911],[-3.087557],[2.710934],[-9.625666],[2.533911],[5.968408],[0.043985],[-1.654166],[-4.429622],[8.759033],[8.867430],[9.944593],[-5.111388],[-0.208963],[-2.222241],[7.792687],[-9.548086],[-1.517702],[-9.426905],[-6.429823],[-7.753842],[-8.365422],[-5.584728],[8.505639],[-0.733278],[9.409787],[-7.737305],[1.688649],[4.321416],[-8.400664],[-4.986458],[-6.083933],[9.917841],[-2.615490],[2.151701],[7.476852],[-9.515084],[2.294924],[-7.871821],[7.856585],[-0.531777],[0.689135],[-4.507843],[5.579243]], dtype = "float32")#candidate|122|(63, 1)|const|float32
var_123 = relay.var("var_123", dtype = "float32", shape = (126,))#candidate|123|(126,)|var|float32
call_121 = relay.TupleGetItem(func_75_call(relay.reshape(const_122.astype('float32'), [9, 1, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), relay.reshape(var_123.astype('float64'), [9, 2, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), relay.reshape(var_123.astype('float64'), [9, 2, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), ), 10)
call_124 = relay.TupleGetItem(func_83_call(relay.reshape(const_122.astype('float32'), [9, 1, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), relay.reshape(var_123.astype('float64'), [9, 2, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), relay.reshape(var_123.astype('float64'), [9, 2, 7]), relay.reshape(var_123.astype('float32'), [9, 2, 7]), ), 10)
uop_125 = relay.sinh(uop_90.astype('float32')) # shape=(5, 14)
bop_127 = relay.less_equal(bop_118.astype('bool'), relay.reshape(uop_116.astype('bool'), relay.shape_of(bop_118))) # shape=(5, 14)
bop_130 = relay.multiply(uop_94.astype('int64'), relay.reshape(uop_114.astype('int64'), relay.shape_of(uop_94))) # shape=(5, 14)
bop_133 = relay.bitwise_or(bop_118.astype('uint8'), relay.reshape(uop_94.astype('uint8'), relay.shape_of(bop_118))) # shape=(5, 14)
bop_136 = relay.logical_xor(uop_101.astype('uint32'), relay.reshape(uop_99.astype('uint32'), relay.shape_of(uop_101))) # shape=(5, 14)
bop_139 = relay.multiply(bop_127.astype('float32'), relay.reshape(uop_125.astype('float32'), relay.shape_of(bop_127))) # shape=(5, 14)
bop_142 = relay.logical_and(uop_114.astype('bool'), relay.reshape(bop_136.astype('bool'), relay.shape_of(uop_114))) # shape=(5, 14)
bop_145 = relay.logical_xor(uop_109.astype('int8'), relay.reshape(uop_94.astype('int8'), relay.shape_of(uop_109))) # shape=(5, 14)
uop_148 = relay.cos(uop_109.astype('float64')) # shape=(5, 14)
uop_150 = relay.sqrt(bop_139.astype('float64')) # shape=(5, 14)
output = relay.Tuple([bop_87,bop_104,bop_111,call_121,const_122,var_123,bop_130,bop_133,bop_142,bop_145,uop_148,uop_150,])
output2 = relay.Tuple([bop_87,bop_104,bop_111,call_124,const_122,var_123,bop_130,bop_133,bop_142,bop_145,uop_148,uop_150,])
func_152 = relay.Function([var_85,var_86,var_103,var_123,], output)
mod['func_152'] = func_152
mod = relay.transform.InferType()(mod)
var_153 = relay.var("var_153", dtype = "float32", shape = (5, 14))#candidate|153|(5, 14)|var|float32
var_154 = relay.var("var_154", dtype = "float32", shape = (5, 14))#candidate|154|(5, 14)|var|float32
var_155 = relay.var("var_155", dtype = "float32", shape = (5, 14))#candidate|155|(5, 14)|var|float32
var_156 = relay.var("var_156", dtype = "float32", shape = (126,))#candidate|156|(126,)|var|float32
output = func_152(var_153,var_154,var_155,var_156,)
func_157 = relay.Function([var_153,var_154,var_155,var_156,], output)
mutated_mod['func_157'] = func_157
mutated_mod = relay.transform.InferType()(mutated_mod)
var_159 = relay.var("var_159", dtype = "float32", shape = ())#candidate|159|()|var|float32
uop_160 = relay.sigmoid(var_159.astype('float32')) # shape=()
uop_162 = relay.rsqrt(var_159.astype('float64')) # shape=()
bop_164 = relay.subtract(uop_160.astype('uint8'), var_159.astype('uint8')) # shape=()
bop_167 = relay.maximum(var_159.astype('int64'), uop_162.astype('int64')) # shape=()
uop_170 = relay.log10(bop_164.astype('float64')) # shape=()
bop_172 = relay.subtract(bop_164.astype('int32'), uop_160.astype('int32')) # shape=()
bop_175 = relay.logical_xor(uop_170.astype('uint8'), uop_162.astype('uint8')) # shape=()
uop_178 = relay.atan(bop_175.astype('float64')) # shape=()
bop_180 = relay.floor_mod(var_159.astype('float64'), uop_162.astype('float64')) # shape=()
bop_183 = relay.floor_mod(uop_162.astype('float64'), uop_170.astype('float64')) # shape=()
var_186 = relay.var("var_186", dtype = "uint8", shape = ())#candidate|186|()|var|uint8
bop_187 = relay.left_shift(bop_175.astype('uint64'), var_186.astype('uint64')) # shape=()
func_152_call = mod.get_global_var('func_152')
func_157_call = mutated_mod.get_global_var('func_157')
const_191 = relay.const([6.337135,-5.023424,0.964378,-5.257639,6.930302,3.562731,-1.813469,-6.951913,-1.425807,5.042816,3.957247,-1.399971,9.663670,-9.894126,3.928004,1.230579,3.624550,7.466694,9.055764,-9.796240,-1.294475,7.744138,-3.909040,8.283217,-6.177565,5.333541,-0.750068,-9.775758,-0.462858,-3.571957,-7.635160,-9.945465,5.536325,-1.794918,-0.902277,8.406417,-5.351126,9.860094,7.511249,-4.969548,9.841048,5.241899,4.297977,9.380274,2.465571,7.469365,9.937020,9.877625,7.787839,9.022166,-3.196401,-9.127638,-0.895937,4.587868,-7.510510,4.353473,9.157109,-9.753523,-5.484978,-4.128735,8.026418,-0.985044,-2.811833,-4.819080,8.952072,4.438151,-9.400758,3.751475,8.593373,-1.253210], dtype = "float32")#candidate|191|(70,)|const|float32
const_192 = relay.const([[5.183266,-4.860730,-2.399131,3.078113,-5.822660,-7.327724,-4.254587,7.449412,2.357999,-1.994351,5.343070,-5.648785,-8.919945,-6.784571,6.430807,-3.024806,-8.138462,-0.862498,3.609790,5.507361,3.490735,2.635220,7.991555,6.127913,5.260769,-2.550321,-9.320820,6.056855,-4.644996,1.306778,5.657063,-6.575806,-3.977649,-6.595762,-0.928314,2.336583,-4.450427,4.809590,5.439940,-4.674913,-0.586478,7.536263],[1.600233,-2.281428,-3.380040,-5.263345,2.478738,-9.690970,5.126054,8.820667,-2.238942,0.777320,0.813577,-0.305407,5.088475,4.521740,-0.012066,-3.608129,7.204038,5.050280,-3.162087,6.813095,4.463668,3.013759,5.147196,-5.886149,1.883971,-5.186866,9.749496,-4.929010,8.957628,-4.261786,-9.778054,-7.200660,4.764645,-0.217703,4.594732,1.559637,7.077897,8.769799,4.664897,-3.789152,-5.101133,4.315703],[-4.236986,-3.512546,1.778404,4.053940,-1.678946,6.415565,9.721072,1.201403,-0.382965,5.304446,-0.020679,-2.373322,0.585825,-2.246028,3.229619,6.604259,6.718951,8.254874,-2.909192,2.826281,-2.894689,7.936380,-8.877182,-5.430786,6.593074,-9.386640,1.654849,-5.044825,0.007104,-5.743166,-2.843726,-0.458640,3.211652,1.951705,-4.280828,-8.822494,2.059004,5.674929,6.660707,-8.687551,0.363719,-7.504347]], dtype = "float32")#candidate|192|(3, 42)|const|float32
call_190 = relay.TupleGetItem(func_152_call(relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_192.astype('float32'), [126,]), ), 10)
call_193 = relay.TupleGetItem(func_157_call(relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_191.astype('float32'), [5, 14]), relay.reshape(const_192.astype('float32'), [126,]), ), 10)
const_194 = relay.const([-7.553489,3.842049,9.148646,0.573828,9.882597,8.423893,-6.988660,9.547953,-3.190495,-3.735069,-0.061547,-2.734429,-3.112828,-0.646792], dtype = "float64")#candidate|194|(14,)|const|float64
bop_195 = relay.maximum(uop_178.astype('int16'), const_194.astype('int16')) # shape=(14,)
uop_198 = relay.asin(bop_195.astype('float64')) # shape=(14,)
uop_200 = relay.asinh(uop_198.astype('float32')) # shape=(14,)
output = relay.Tuple([bop_167,bop_172,bop_180,bop_183,bop_187,call_190,const_191,const_192,uop_200,])
output2 = relay.Tuple([bop_167,bop_172,bop_180,bop_183,bop_187,call_193,const_191,const_192,uop_200,])
func_202 = relay.Function([var_159,var_186,], output)
mod['func_202'] = func_202
mod = relay.transform.InferType()(mod)
var_203 = relay.var("var_203", dtype = "float32", shape = ())#candidate|203|()|var|float32
var_204 = relay.var("var_204", dtype = "uint8", shape = ())#candidate|204|()|var|uint8
output = func_202(var_203,var_204,)
func_205 = relay.Function([var_203,var_204,], output)
mutated_mod['func_205'] = func_205
mutated_mod = relay.transform.InferType()(mutated_mod)
var_207 = relay.var("var_207", dtype = "float64", shape = (10, 2))#candidate|207|(10, 2)|var|float64
uop_208 = relay.acosh(var_207.astype('float64')) # shape=(10, 2)
bop_210 = relay.subtract(uop_208.astype('int64'), relay.reshape(var_207.astype('int64'), relay.shape_of(uop_208))) # shape=(10, 2)
uop_213 = relay.log10(var_207.astype('float64')) # shape=(10, 2)
uop_215 = relay.rsqrt(bop_210.astype('float64')) # shape=(10, 2)
uop_217 = relay.acos(uop_208.astype('float32')) # shape=(10, 2)
output = relay.Tuple([uop_213,uop_215,uop_217,])
output2 = relay.Tuple([uop_213,uop_215,uop_217,])
func_219 = relay.Function([var_207,], output)
mod['func_219'] = func_219
mod = relay.transform.InferType()(mod)
mutated_mod['func_219'] = func_219
mutated_mod = relay.transform.InferType()(mutated_mod)
var_220 = relay.var("var_220", dtype = "float64", shape = (10, 2))#candidate|220|(10, 2)|var|float64
func_219_call = mutated_mod.get_global_var('func_219')
call_221 = func_219_call(var_220)
output = call_221
func_222 = relay.Function([var_220], output)
mutated_mod['func_222'] = func_222
mutated_mod = relay.transform.InferType()(mutated_mod)
var_224 = relay.var("var_224", dtype = "int64", shape = ())#candidate|224|()|var|int64
const_225 = relay.const([[-3,2,9,6],[2,-6,-7,-3],[-9,-8,-2,4],[-4,-3,-9,1],[-4,-1,2,-4],[9,5,8,-10]], dtype = "int64")#candidate|225|(6, 4)|const|int64
bop_226 = relay.left_shift(var_224.astype('int64'), const_225.astype('int64')) # shape=(6, 4)
var_229 = relay.var("var_229", dtype = "int64", shape = (7,))#candidate|229|(7,)|var|int64
bop_230 = relay.power(var_224.astype('float64'), var_229.astype('float64')) # shape=(7,)
uop_233 = relay.asin(var_229.astype('float64')) # shape=(7,)
bop_235 = relay.bitwise_and(uop_233.astype('uint16'), relay.reshape(var_229.astype('uint16'), relay.shape_of(uop_233))) # shape=(7,)
uop_238 = relay.acos(bop_235.astype('float32')) # shape=(7,)
uop_240 = relay.acos(var_229.astype('float64')) # shape=(7,)
uop_242 = relay.acos(bop_226.astype('float64')) # shape=(6, 4)
bop_244 = relay.greater_equal(var_224.astype('bool'), bop_235.astype('bool')) # shape=(7,)
var_247 = relay.var("var_247", dtype = "float64", shape = (7,))#candidate|247|(7,)|var|float64
bop_248 = relay.bitwise_xor(uop_240.astype('uint8'), relay.reshape(var_247.astype('uint8'), relay.shape_of(uop_240))) # shape=(7,)
bop_251 = relay.mod(uop_238.astype('float32'), relay.reshape(bop_235.astype('float32'), relay.shape_of(uop_238))) # shape=(7,)
var_254 = relay.var("var_254", dtype = "uint16", shape = (7,))#candidate|254|(7,)|var|uint16
bop_255 = relay.logical_xor(bop_235.astype('int64'), relay.reshape(var_254.astype('int64'), relay.shape_of(bop_235))) # shape=(7,)
var_258 = relay.var("var_258", dtype = "float32", shape = (7,))#candidate|258|(7,)|var|float32
bop_259 = relay.power(uop_238.astype('float32'), relay.reshape(var_258.astype('float32'), relay.shape_of(uop_238))) # shape=(7,)
output = relay.Tuple([bop_230,uop_242,bop_244,bop_248,bop_251,bop_255,bop_259,])
output2 = relay.Tuple([bop_230,uop_242,bop_244,bop_248,bop_251,bop_255,bop_259,])
func_262 = relay.Function([var_224,var_229,var_247,var_254,var_258,], output)
mod['func_262'] = func_262
mod = relay.transform.InferType()(mod)
var_263 = relay.var("var_263", dtype = "int64", shape = ())#candidate|263|()|var|int64
var_264 = relay.var("var_264", dtype = "int64", shape = (7,))#candidate|264|(7,)|var|int64
var_265 = relay.var("var_265", dtype = "float64", shape = (7,))#candidate|265|(7,)|var|float64
var_266 = relay.var("var_266", dtype = "uint16", shape = (7,))#candidate|266|(7,)|var|uint16
var_267 = relay.var("var_267", dtype = "float32", shape = (7,))#candidate|267|(7,)|var|float32
output = func_262(var_263,var_264,var_265,var_266,var_267,)
func_268 = relay.Function([var_263,var_264,var_265,var_266,var_267,], output)
mutated_mod['func_268'] = func_268
mutated_mod = relay.transform.InferType()(mutated_mod)
var_270 = relay.var("var_270", dtype = "int64", shape = (4, 5))#candidate|270|(4, 5)|var|int64
var_271 = relay.var("var_271", dtype = "int64", shape = (4, 5))#candidate|271|(4, 5)|var|int64
bop_272 = relay.less_equal(var_270.astype('bool'), relay.reshape(var_271.astype('bool'), relay.shape_of(var_270))) # shape=(4, 5)
uop_275 = relay.asinh(var_270.astype('float64')) # shape=(4, 5)
output = relay.Tuple([bop_272,uop_275,])
output2 = relay.Tuple([bop_272,uop_275,])
func_277 = relay.Function([var_270,var_271,], output)
mod['func_277'] = func_277
mod = relay.transform.InferType()(mod)
var_278 = relay.var("var_278", dtype = "int64", shape = (4, 5))#candidate|278|(4, 5)|var|int64
var_279 = relay.var("var_279", dtype = "int64", shape = (4, 5))#candidate|279|(4, 5)|var|int64
output = func_277(var_278,var_279,)
func_280 = relay.Function([var_278,var_279,], output)
mutated_mod['func_280'] = func_280
mutated_mod = relay.transform.InferType()(mutated_mod)
var_282 = relay.var("var_282", dtype = "uint32", shape = (16, 1, 11))#candidate|282|(16, 1, 11)|var|uint32
var_283 = relay.var("var_283", dtype = "uint32", shape = (16, 9, 11))#candidate|283|(16, 9, 11)|var|uint32
bop_284 = relay.bitwise_or(var_282.astype('uint32'), var_283.astype('uint32')) # shape=(16, 9, 11)
uop_287 = relay.acos(var_283.astype('float32')) # shape=(16, 9, 11)
uop_289 = relay.log(uop_287.astype('float32')) # shape=(16, 9, 11)
bop_291 = relay.multiply(uop_287.astype('uint8'), relay.reshape(bop_284.astype('uint8'), relay.shape_of(uop_287))) # shape=(16, 9, 11)
bop_294 = relay.mod(uop_289.astype('float32'), var_282.astype('float32')) # shape=(16, 9, 11)
uop_297 = relay.atan(uop_289.astype('float64')) # shape=(16, 9, 11)
bop_299 = relay.subtract(bop_294.astype('uint64'), relay.reshape(bop_284.astype('uint64'), relay.shape_of(bop_294))) # shape=(16, 9, 11)
bop_302 = relay.logical_or(uop_297.astype('bool'), relay.reshape(uop_287.astype('bool'), relay.shape_of(uop_297))) # shape=(16, 9, 11)
uop_305 = relay.cos(bop_302.astype('float64')) # shape=(16, 9, 11)
output = relay.Tuple([bop_291,bop_299,uop_305,])
output2 = relay.Tuple([bop_291,bop_299,uop_305,])
func_307 = relay.Function([var_282,var_283,], output)
mod['func_307'] = func_307
mod = relay.transform.InferType()(mod)
var_308 = relay.var("var_308", dtype = "uint32", shape = (16, 1, 11))#candidate|308|(16, 1, 11)|var|uint32
var_309 = relay.var("var_309", dtype = "uint32", shape = (16, 9, 11))#candidate|309|(16, 9, 11)|var|uint32
output = func_307(var_308,var_309,)
func_310 = relay.Function([var_308,var_309,], output)
mutated_mod['func_310'] = func_310
mutated_mod = relay.transform.InferType()(mutated_mod)
var_312 = relay.var("var_312", dtype = "int16", shape = (9, 1, 15))#candidate|312|(9, 1, 15)|var|int16
const_313 = relay.const([[[-4,-6,3,-3,-4,9,2,3,1,-4,6,3,6,-7,2],[3,-7,2,-8,-1,9,2,-9,9,-3,-2,-6,-4,-9,-3],[9,5,-8,-7,4,-1,-7,4,-10,-6,3,1,5,4,-6],[-4,-7,-10,8,3,7,-6,-9,-2,1,-1,8,-6,7,6],[10,-8,-9,-2,-10,-10,9,3,6,8,-5,-4,10,7,-3],[-1,-5,8,9,-2,4,-8,-7,2,8,1,-8,-6,8,-8],[-6,-3,-1,-4,2,-1,-7,-1,10,8,-3,9,7,-4,9],[-7,2,-8,3,-5,-8,-1,-3,-1,-6,-1,1,2,7,2],[-7,-3,-9,4,-2,4,3,-7,10,1,1,4,-2,-8,2]],[[5,9,8,8,7,-1,10,-10,2,8,-5,5,-1,-6,4],[-8,-5,8,4,-7,3,-4,2,3,4,-7,-2,-6,7,-9],[10,-3,-6,-2,4,1,-5,-4,9,-9,1,-9,8,4,-1],[-10,4,-8,2,-1,9,-4,7,1,5,10,-8,9,-6,3],[9,8,5,-4,-8,-3,-6,-3,-7,2,6,8,-1,-2,-1],[4,-6,-3,-10,10,-3,9,2,-4,-6,-10,8,-8,6,-4],[3,-9,2,-4,3,2,-8,5,1,-8,-6,1,8,-3,-9],[3,6,-3,-6,-1,-10,2,2,-8,7,-3,-8,-10,-6,-1],[1,7,6,6,-6,2,9,-3,-3,7,-1,-4,-7,-5,7]],[[-7,10,6,1,-1,-8,-4,6,10,-2,-9,7,2,-10,-6],[3,-5,-5,-2,-10,5,3,-7,-9,-8,1,-4,1,-2,-2],[-5,-10,-9,-1,5,-1,1,2,-3,-2,9,-8,-2,6,-7],[-2,5,-5,9,3,-9,-4,10,-8,-6,4,4,6,-1,1],[5,4,-7,2,7,10,4,6,-2,7,-5,-1,4,-8,-4],[-5,-2,8,-10,-5,-8,2,-3,-5,5,10,6,-10,3,-7],[4,-5,-3,-8,-5,-9,-10,-4,1,7,-3,5,-10,-2,-10],[-1,3,5,-1,8,-7,-9,-9,-1,4,6,5,-9,-6,-10],[-9,5,1,7,-2,-7,1,-8,-9,-1,-7,2,-10,-8,3]],[[-5,3,6,7,-5,8,9,-8,-2,-8,-3,1,6,6,8],[-6,-2,2,-2,-1,1,10,1,-9,-6,7,9,-5,-9,-2],[2,1,-8,4,-5,8,-6,-4,-6,-1,9,4,-6,-6,1],[4,-8,9,-2,-8,8,1,-4,-3,9,5,-2,7,-9,4],[10,5,-7,-6,5,-5,9,9,2,-4,-5,-8,-9,8,-7],[9,-4,-5,-9,9,-4,9,2,-5,-9,5,9,5,-10,5],[7,5,8,-10,3,-1,6,8,-7,2,1,1,3,-7,9],[10,-6,5,5,9,9,-4,-1,4,-5,-5,6,-8,2,-1],[7,-6,6,-5,9,10,-5,2,7,5,4,10,-3,-10,4]],[[-7,5,-2,-5,4,2,-3,-8,-4,-2,-7,-6,-6,-5,-6],[-8,-8,6,-3,-6,6,-3,-4,3,8,-6,2,1,-2,4],[5,-3,-6,4,2,8,-7,-9,2,10,9,-4,-7,3,-3],[-9,1,-3,3,-6,-7,-6,-3,-7,1,6,5,-3,7,2],[-2,-10,5,10,1,6,8,4,9,8,-1,9,4,-8,9],[1,-3,6,2,-7,-2,3,-4,-1,8,-10,4,5,-3,-9],[5,6,8,-3,8,5,2,9,-3,-9,-2,2,-10,3,3],[8,1,3,3,-2,-7,-10,5,5,8,2,4,-5,4,8],[2,-3,-7,4,2,3,-5,8,-10,9,-8,-9,7,-7,-6]],[[8,10,-2,-6,10,-2,-9,4,6,-2,-6,-8,-3,1,-7],[4,6,5,-2,2,-2,5,-1,1,-4,2,3,5,6,7],[-4,7,6,7,4,2,-4,-7,5,-2,-1,9,4,-4,-3],[-2,-4,8,-2,10,4,3,-6,-5,-10,-1,3,7,-5,8],[1,-6,10,7,-9,-9,8,5,4,-4,-8,5,5,9,-4],[2,8,2,-4,-1,5,-7,3,6,3,2,-7,3,1,-3],[-8,10,-3,-5,1,-1,7,9,2,-7,-7,6,-8,-5,-10],[3,5,7,8,-6,-7,5,-5,-4,-10,-1,2,1,6,-10],[-6,6,8,1,5,-3,4,-8,4,5,-5,-1,4,6,9]],[[6,-10,-7,6,-2,-3,5,5,-8,5,3,10,8,7,1],[1,1,10,8,6,-8,7,8,-7,10,-10,4,6,-3,8],[-9,1,-10,10,8,-8,-4,-10,-6,9,-8,9,-8,-4,8],[-4,10,-8,8,-8,3,3,-2,2,-7,-6,-1,-7,-9,-9],[-4,3,8,-4,-1,-5,10,-2,10,-2,4,-1,2,5,2],[8,6,-8,5,3,6,-8,5,-1,-3,-3,5,-3,4,8],[-1,2,10,-4,-3,2,-6,-8,-2,7,5,4,3,-7,-7],[-3,-3,-8,4,-9,10,8,2,8,-4,9,2,-9,5,4],[-3,2,1,-9,3,-3,7,-10,7,3,2,3,7,1,-10]],[[-6,1,8,-10,-8,-5,8,2,3,8,1,9,-6,-5,-10],[3,7,8,5,-4,-2,5,3,2,5,1,9,-1,8,-5],[3,8,1,7,3,10,10,-7,-3,6,-8,5,9,-4,1],[7,7,-4,-1,-4,-2,-10,-10,-3,8,2,-10,-3,10,-8],[-9,5,-2,4,-9,8,5,-4,-6,9,-8,-10,-5,-7,-2],[-7,2,9,-2,7,-5,3,-4,-6,-5,4,5,10,-7,-6],[4,2,-7,7,-7,-1,1,2,-2,-4,3,-10,2,-8,-9],[-9,-2,-6,-3,-8,-9,-3,-6,6,3,-5,-7,-5,-7,9],[-1,-8,1,-2,10,5,1,-10,2,5,5,4,-6,-5,3]],[[-10,4,7,-3,10,-9,3,-6,6,-10,-5,-10,-8,-7,-1],[7,-5,3,-6,1,10,-4,9,-9,-8,-6,3,-2,-4,4],[-7,1,3,4,-10,7,4,-4,-1,-7,-1,2,1,-5,-10],[2,-9,1,-7,-3,-9,-7,-4,-5,-6,-4,7,9,9,-3],[-2,-8,10,-4,4,3,8,6,3,-6,4,-3,5,9,7],[-3,-8,1,10,-8,3,-10,3,2,5,-4,8,-10,-1,-7],[10,9,-10,-10,6,-7,-9,-9,-2,8,-8,6,10,-9,4],[1,-7,-4,1,-9,1,-5,4,-3,-6,4,1,-10,-6,6],[-8,-6,9,1,-5,8,-10,-8,-8,1,-10,-8,4,4,10]]], dtype = "int16")#candidate|313|(9, 9, 15)|const|int16
bop_314 = relay.greater_equal(var_312.astype('bool'), const_313.astype('bool')) # shape=(9, 9, 15)
uop_317 = relay.sigmoid(var_312.astype('float32')) # shape=(9, 1, 15)
output = relay.Tuple([bop_314,uop_317,])
output2 = relay.Tuple([bop_314,uop_317,])
func_319 = relay.Function([var_312,], output)
mod['func_319'] = func_319
mod = relay.transform.InferType()(mod)
var_320 = relay.var("var_320", dtype = "int16", shape = (9, 1, 15))#candidate|320|(9, 1, 15)|var|int16
output = func_319(var_320)
func_321 = relay.Function([var_320], output)
mutated_mod['func_321'] = func_321
mutated_mod = relay.transform.InferType()(mutated_mod)
var_323 = relay.var("var_323", dtype = "float64", shape = (3, 13))#candidate|323|(3, 13)|var|float64
uop_324 = relay.sin(var_323.astype('float64')) # shape=(3, 13)
var_326 = relay.var("var_326", dtype = "float64", shape = (3, 13))#candidate|326|(3, 13)|var|float64
bop_327 = relay.subtract(var_323.astype('uint64'), relay.reshape(var_326.astype('uint64'), relay.shape_of(var_323))) # shape=(3, 13)
var_330 = relay.var("var_330", dtype = "float64", shape = (3, 13))#candidate|330|(3, 13)|var|float64
bop_331 = relay.bitwise_xor(uop_324.astype('int8'), relay.reshape(var_330.astype('int8'), relay.shape_of(uop_324))) # shape=(3, 13)
bop_334 = relay.logical_or(var_330.astype('bool'), relay.reshape(bop_331.astype('bool'), relay.shape_of(var_330))) # shape=(3, 13)
uop_337 = relay.sinh(bop_334.astype('float64')) # shape=(3, 13)
var_339 = relay.var("var_339", dtype = "int8", shape = (3, 13))#candidate|339|(3, 13)|var|int8
bop_340 = relay.minimum(bop_331.astype('uint8'), relay.reshape(var_339.astype('uint8'), relay.shape_of(bop_331))) # shape=(3, 13)
output = relay.Tuple([bop_327,uop_337,bop_340,])
output2 = relay.Tuple([bop_327,uop_337,bop_340,])
F = relay.Function([var_323,var_326,var_330,var_339,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_323,var_326,var_330,var_339,], output2)
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
input_323= np.array([[-5.172608,-5.987722,-2.557159,6.011787,-8.430540,-4.660037,2.734777,-3.771136,-1.276025,7.544160,8.156082,1.262925,4.755121],[-0.860334,-1.032491,-4.841710,-3.687423,2.328703,9.776047,-0.486366,1.632803,-4.393550,-6.193407,-4.895575,1.519939,4.505227],[6.348690,-0.766652,2.392331,-0.129790,-5.550671,7.143018,1.828402,9.628661,3.255409,-2.301842,-2.404994,-4.839740,-8.228094]], dtype='float64')
module1.set_input('var_323', input_323)
input_326= np.array([[-7.991323,-4.758909,-6.577038,-4.448205,-5.093295,1.553984,-4.939570,0.557005,-6.886478,-5.036065,0.917150,-9.760762,-9.668613],[-5.765320,8.368524,-8.448507,-6.882592,8.976770,-6.486088,0.159183,-0.852383,3.121675,-4.128025,-1.291657,-8.228167,-5.073362],[-6.662073,9.410709,-0.591028,9.752199,-0.927004,3.291838,-9.140844,0.908557,-1.779490,5.331366,-4.934584,8.862254,6.584272]], dtype='float64')
module1.set_input('var_326', input_326)
input_330= np.array([[8.971725,-6.763797,2.286964,-8.344316,-4.022847,-2.915000,8.950844,3.756798,3.451513,3.696371,-0.461404,-5.766305,-0.926992],[6.716968,2.998843,3.830937,-0.641761,4.464638,8.464168,-9.981941,-6.560879,2.884923,-2.690608,-8.440656,2.953920,-2.014985],[8.238319,5.161225,7.838775,-8.415352,-4.665170,-3.633515,8.394148,-1.636418,7.810020,-3.174935,-3.298704,8.870871,-5.945087]], dtype='float64')
module1.set_input('var_330', input_330)
input_339= np.array([[-4,6,-9,-2,-5,-2,-9,-1,8,2,1,-6,2],[9,8,-9,-10,10,7,2,10,-2,10,-1,5,6],[7,-1,5,3,5,8,8,-9,-2,6,6,1,1]], dtype='int8')
module1.set_input('var_339', input_339)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_323, input_326, input_330, input_339, )
res3 = intrp3.evaluate()(input_323, input_326, input_330, input_339, )
res4 = intrp4.evaluate()(input_323, input_326, input_330, input_339, )
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
module5.set_input('var_323', input_323)
module5.set_input('var_326', input_326)
module5.set_input('var_330', input_330)
module5.set_input('var_339', input_339)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_323, input_326, input_330, input_339, )
res7 = intrp7.evaluate()(input_323, input_326, input_330, input_339, )
res8 = intrp8.evaluate()(input_323, input_326, input_330, input_339, )
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
module9.set_input('var_323', input_323)
module9.set_input('var_326', input_326)
module9.set_input('var_330', input_330)
module9.set_input('var_339', input_339)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_323, input_326, input_330, input_339, )
res11 = intrp11.evaluate()(input_323, input_326, input_330, input_339, )
res12 = intrp12.evaluate()(input_323, input_326, input_330, input_339, )
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
module13.set_input('var_323', input_323)
module13.set_input('var_326', input_326)
module13.set_input('var_330', input_330)
module13.set_input('var_339', input_339)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_323, input_326, input_330, input_339, )
res15 = intrp15.evaluate()(input_323, input_326, input_330, input_339, )
res16 = intrp16.evaluate()(input_323, input_326, input_330, input_339, )
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
module17.set_input('var_323', input_323)
module17.set_input('var_326', input_326)
module17.set_input('var_330', input_330)
module17.set_input('var_339', input_339)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_323, input_326, input_330, input_339, )
res19 = intrp19.evaluate()(input_323, input_326, input_330, input_339, )
res20 = intrp20.evaluate()(input_323, input_326, input_330, input_339, )
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
module21.set_input('var_323', input_323)
module21.set_input('var_326', input_326)
module21.set_input('var_330', input_330)
module21.set_input('var_339', input_339)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_323, input_326, input_330, input_339, )
res23 = intrp23.evaluate()(input_323, input_326, input_330, input_339, )
res24 = intrp24.evaluate()(input_323, input_326, input_330, input_339, )
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