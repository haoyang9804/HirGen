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
var_0 = relay.var("var_0", dtype = "uint8", shape = ())#candidate|0|()|var|uint8
var_1 = relay.var("var_1", dtype = "uint8", shape = ())#candidate|1|()|var|uint8
bop_2 = relay.subtract(var_0.astype('uint8'), var_1.astype('uint8')) # shape=()
uop_5 = relay.cos(bop_2.astype('float32')) # shape=()
uop_7 = relay.sin(uop_5.astype('float64')) # shape=()
var_9 = relay.var("var_9", dtype = "float64", shape = (8, 13, 5))#candidate|9|(8, 13, 5)|var|float64
bop_10 = relay.bitwise_and(uop_7.astype('int16'), var_9.astype('int16')) # shape=(8, 13, 5)
uop_13 = relay.sin(var_1.astype('float64')) # shape=()
bop_15 = relay.left_shift(bop_10.astype('uint64'), bop_2.astype('uint64')) # shape=(8, 13, 5)
bop_18 = relay.bitwise_or(uop_5.astype('uint64'), var_1.astype('uint64')) # shape=()
const_21 = relay.const([[[-2,7,7,-7,6],[6,-6,9,8,5],[10,5,8,-7,-5],[-1,2,-3,5,-4],[10,1,2,-5,1],[-2,-4,2,6,-8],[-6,8,-8,9,-2],[6,-4,-1,6,-1],[-6,9,10,1,-3],[-9,-10,7,9,-3],[8,3,8,-2,-10],[3,-4,5,6,-3],[4,7,10,-2,-9]],[[-10,-6,-1,-8,1],[6,3,8,7,3],[2,2,1,-5,-1],[-8,1,5,-8,-7],[-1,3,-4,-4,-7],[-10,6,10,4,-2],[8,-7,6,3,4],[-5,2,1,-8,-4],[5,4,6,9,-9],[-6,-3,-6,-4,2],[-8,3,4,-5,-2],[-9,-4,6,-9,1],[-10,-5,-9,-1,-1]],[[-9,10,8,-7,-10],[1,-3,6,-5,8],[5,1,10,-2,-3],[4,3,-3,1,3],[8,-1,-8,-2,-2],[-7,8,-8,9,9],[-4,-3,6,7,5],[-9,3,-6,8,2],[-10,-8,8,-9,5],[-9,1,-5,6,-6],[-3,6,6,-7,-1],[6,-9,-10,-10,5],[8,-4,4,8,5]],[[-4,9,-4,-3,-9],[3,4,10,10,4],[5,9,-1,-1,4],[-2,-4,-10,8,-1],[-6,4,6,-5,-4],[-2,-4,-10,-6,3],[7,-8,-8,9,-2],[5,-5,-9,3,-7],[-6,7,2,5,8],[-7,5,6,-6,1],[5,5,3,1,9],[-4,-9,-2,-10,-10],[-5,5,-2,5,-8]],[[-5,2,-5,5,4],[5,8,-9,-6,7],[7,-3,-4,-5,-5],[9,6,-2,-8,-8],[-1,-10,-7,-9,-8],[-1,10,9,-10,-5],[7,-3,-7,5,-1],[8,4,-3,9,-7],[8,-3,-1,4,-7],[6,-5,7,5,-7],[-5,-10,-5,1,5],[-10,-6,-3,3,10],[-10,-6,-5,-7,6]],[[-9,-1,9,2,-9],[-8,9,-1,-2,-4],[6,9,7,4,-6],[-9,1,-2,5,2],[10,-4,3,5,-10],[10,-1,8,2,-3],[4,3,-2,-10,10],[5,-2,-1,2,2],[5,8,8,-3,1],[1,9,-5,-1,-10],[-4,1,1,1,-7],[7,-5,-5,4,5],[5,-1,2,-5,-8]],[[3,9,-10,-5,-8],[-3,4,2,3,1],[-1,6,1,7,-2],[-3,1,-9,6,1],[8,-9,-4,-4,-6],[4,3,1,-9,-2],[-5,6,-10,-6,-2],[-10,-1,-5,-1,1],[-5,-5,7,2,6],[7,-2,-7,9,-9],[-10,-1,10,-7,6],[6,-5,10,1,-5],[8,6,4,-4,-7]],[[-4,-2,10,-3,-10],[-5,8,1,1,10],[9,-10,9,8,-9],[-6,4,9,9,10],[-1,-10,-4,-7,-5],[3,-6,-6,-6,-6],[-3,-7,-10,2,-10],[-1,-7,8,-3,8],[-3,-9,1,3,4],[-2,2,6,8,3],[2,7,-6,3,-1],[5,-2,-2,-2,2],[-4,4,6,2,-8]]], dtype = "uint64")#candidate|21|(8, 13, 5)|const|uint64
bop_22 = relay.floor_mod(bop_15.astype('float64'), relay.reshape(const_21.astype('float64'), relay.shape_of(bop_15))) # shape=(8, 13, 5)
const_25 = relay.const([[[8.437980,-8.076735,-5.530025,-6.493413,1.056202,1.840674],[6.674937,3.625043,8.640057,4.397676,-9.779969,-1.233188],[-4.402479,5.250682,5.920702,9.229359,-9.211693,0.603248],[-6.148417,4.760133,0.271428,-1.279245,6.606941,-8.463625]],[[8.708738,8.895207,6.295849,6.759541,-9.131079,2.960360],[4.567051,-9.024602,-1.829353,-3.016409,8.405019,7.691863],[7.980312,4.921678,6.987383,4.371616,0.875397,-2.456563],[-0.466313,0.227661,-4.484861,-5.796131,3.896130,7.986775]],[[-0.862599,-4.645805,-6.473195,-0.397207,8.272156,-0.938204],[3.870912,4.331815,-0.780289,-9.274535,-4.756649,-2.109793],[-6.900498,-2.482283,7.480782,2.085473,-4.674821,-3.183533],[5.685486,-8.089084,-6.447974,4.838362,-4.594770,1.011305]],[[1.724467,9.670785,2.134590,-9.929847,-2.355668,1.651384],[2.472304,2.560954,-5.123496,-8.009780,-7.817604,-0.137967],[5.996962,0.227703,2.153431,5.151571,-9.727032,-4.953815],[-8.008971,9.894333,1.578794,-2.682391,9.249763,-2.414178]],[[0.730557,-2.951028,1.773876,9.096208,-0.615028,-5.110309],[1.626393,-3.342559,-9.228535,5.680660,-1.686069,5.398249],[9.516566,-5.116152,9.568437,-3.936072,1.030148,6.250332],[7.483932,5.103128,6.549525,7.336012,8.649209,-3.683772]],[[-2.322448,-7.754392,-0.077147,2.737029,-3.916363,-8.381256],[-8.514378,-2.276212,7.409456,7.548752,-9.153622,8.097846],[-5.219421,-9.341658,-6.967032,-8.767333,-8.392675,8.586984],[-1.431160,-5.975805,-9.214075,8.321123,4.440055,-7.759891]],[[9.600763,-0.787744,5.070188,6.414834,-4.008302,0.514311],[-2.865694,-5.407381,0.466794,0.749503,-6.816875,-3.992556],[-0.880111,6.065388,4.076224,4.627795,0.998533,4.425700],[1.302444,9.200345,-3.452019,-0.426226,-0.262339,-2.444391]],[[2.738378,3.067429,0.843392,-3.823329,9.170774,-8.203233],[-3.816696,8.404568,0.234848,-3.818998,0.578348,-4.780077],[-5.735328,-0.312235,1.816428,-8.615844,-3.911719,-2.989090],[3.437985,-4.211208,0.581844,-0.316124,-1.665138,1.988427]],[[-5.781868,-2.213786,6.310055,5.422916,2.385896,-0.954671],[-9.763021,8.549592,0.435927,5.273889,1.669354,8.169404],[-0.098641,-8.419210,-8.489554,-8.042918,1.241855,-7.977714],[2.257525,-0.690991,-2.411777,-8.532103,-5.400473,5.270380]],[[5.139245,-2.729102,-3.291227,9.320999,3.745569,3.622995],[-2.624799,-0.663245,-9.395786,-5.748409,7.619590,1.846280],[-9.602903,-5.969868,6.095341,-2.263794,-2.069056,6.800933],[-3.200022,2.401898,-9.537129,6.936608,-1.829178,-7.524038]],[[5.253908,9.478122,-3.928354,-8.589975,-6.142618,6.883116],[-3.866570,-8.239923,-8.158838,-7.744680,6.608538,3.950979],[-0.459377,-0.998748,-3.085815,-2.031742,3.734829,-4.923178],[-1.195621,-5.749370,-0.137731,-9.002635,4.346902,5.013878]]], dtype = "float32")#candidate|25|(11, 4, 6)|const|float32
bop_26 = relay.greater(uop_5.astype('bool'), const_25.astype('bool')) # shape=(11, 4, 6)
bop_29 = relay.right_shift(uop_7.astype('uint32'), uop_13.astype('uint32')) # shape=()
uop_32 = relay.cosh(const_21.astype('float32')) # shape=(8, 13, 5)
bop_34 = relay.minimum(bop_26.astype('uint8'), uop_13.astype('uint8')) # shape=(11, 4, 6)
bop_37 = relay.subtract(bop_26.astype('float64'), uop_13.astype('float64')) # shape=(11, 4, 6)
uop_40 = relay.acos(bop_15.astype('float32')) # shape=(8, 13, 5)
var_42 = relay.var("var_42", dtype = "float32", shape = (8, 13, 5))#candidate|42|(8, 13, 5)|var|float32
bop_43 = relay.less(uop_40.astype('bool'), relay.reshape(var_42.astype('bool'), relay.shape_of(uop_40))) # shape=(8, 13, 5)
bop_46 = relay.equal(uop_40.astype('bool'), relay.reshape(bop_22.astype('bool'), relay.shape_of(uop_40))) # shape=(8, 13, 5)
bop_49 = relay.less(bop_46.astype('bool'), relay.reshape(bop_43.astype('bool'), relay.shape_of(bop_46))) # shape=(8, 13, 5)
bop_52 = relay.multiply(bop_49.astype('float32'), relay.reshape(bop_15.astype('float32'), relay.shape_of(bop_49))) # shape=(8, 13, 5)
var_55 = relay.var("var_55", dtype = "float32", shape = (8, 13, 5))#candidate|55|(8, 13, 5)|var|float32
bop_56 = relay.bitwise_xor(bop_52.astype('int8'), relay.reshape(var_55.astype('int8'), relay.shape_of(bop_52))) # shape=(8, 13, 5)
uop_59 = relay.sinh(bop_46.astype('float64')) # shape=(8, 13, 5)
bop_61 = relay.logical_xor(bop_22.astype('int32'), var_1.astype('int32')) # shape=(8, 13, 5)
bop_64 = relay.add(uop_7.astype('uint8'), bop_18.astype('uint8')) # shape=()
bop_67 = relay.add(uop_59.astype('int8'), var_0.astype('int8')) # shape=(8, 13, 5)
var_70 = relay.var("var_70", dtype = "int8", shape = (8, 13, 5))#candidate|70|(8, 13, 5)|var|int8
bop_71 = relay.divide(bop_67.astype('float64'), relay.reshape(var_70.astype('float64'), relay.shape_of(bop_67))) # shape=(8, 13, 5)
var_74 = relay.var("var_74", dtype = "bool", shape = (8, 13, 5))#candidate|74|(8, 13, 5)|var|bool
bop_75 = relay.maximum(bop_49.astype('int8'), relay.reshape(var_74.astype('int8'), relay.shape_of(bop_49))) # shape=(8, 13, 5)
uop_78 = relay.cos(uop_59.astype('float64')) # shape=(8, 13, 5)
uop_80 = relay.asinh(uop_78.astype('float64')) # shape=(8, 13, 5)
output = relay.Tuple([bop_29,uop_32,bop_34,bop_37,bop_56,bop_61,bop_64,bop_71,bop_75,uop_80,])
output2 = relay.Tuple([bop_29,uop_32,bop_34,bop_37,bop_56,bop_61,bop_64,bop_71,bop_75,uop_80,])
func_82 = relay.Function([var_0,var_1,var_9,var_42,var_55,var_70,var_74,], output)
mod['func_82'] = func_82
mod = relay.transform.InferType()(mod)
var_83 = relay.var("var_83", dtype = "uint8", shape = ())#candidate|83|()|var|uint8
var_84 = relay.var("var_84", dtype = "uint8", shape = ())#candidate|84|()|var|uint8
var_85 = relay.var("var_85", dtype = "float64", shape = (8, 13, 5))#candidate|85|(8, 13, 5)|var|float64
var_86 = relay.var("var_86", dtype = "float32", shape = (8, 13, 5))#candidate|86|(8, 13, 5)|var|float32
var_87 = relay.var("var_87", dtype = "float32", shape = (8, 13, 5))#candidate|87|(8, 13, 5)|var|float32
var_88 = relay.var("var_88", dtype = "int8", shape = (8, 13, 5))#candidate|88|(8, 13, 5)|var|int8
var_89 = relay.var("var_89", dtype = "bool", shape = (8, 13, 5))#candidate|89|(8, 13, 5)|var|bool
output = func_82(var_83,var_84,var_85,var_86,var_87,var_88,var_89,)
func_90 = relay.Function([var_83,var_84,var_85,var_86,var_87,var_88,var_89,], output)
mutated_mod['func_90'] = func_90
mutated_mod = relay.transform.InferType()(mutated_mod)
var_92 = relay.var("var_92", dtype = "int64", shape = (2, 11, 8))#candidate|92|(2, 11, 8)|var|int64
var_93 = relay.var("var_93", dtype = "int64", shape = (2, 11, 8))#candidate|93|(2, 11, 8)|var|int64
bop_94 = relay.less(var_92.astype('bool'), relay.reshape(var_93.astype('bool'), relay.shape_of(var_92))) # shape=(2, 11, 8)
bop_97 = relay.power(var_93.astype('float32'), relay.reshape(bop_94.astype('float32'), relay.shape_of(var_93))) # shape=(2, 11, 8)
uop_100 = relay.cosh(bop_97.astype('float64')) # shape=(2, 11, 8)
bop_102 = relay.power(uop_100.astype('float32'), relay.reshape(bop_94.astype('float32'), relay.shape_of(uop_100))) # shape=(2, 11, 8)
var_105 = relay.var("var_105", dtype = "int64", shape = (2, 11, 8))#candidate|105|(2, 11, 8)|var|int64
bop_106 = relay.subtract(var_92.astype('uint8'), relay.reshape(var_105.astype('uint8'), relay.shape_of(var_92))) # shape=(2, 11, 8)
bop_109 = relay.greater_equal(bop_102.astype('bool'), relay.reshape(uop_100.astype('bool'), relay.shape_of(bop_102))) # shape=(2, 11, 8)
output = relay.Tuple([bop_106,bop_109,])
output2 = relay.Tuple([bop_106,bop_109,])
func_112 = relay.Function([var_92,var_93,var_105,], output)
mod['func_112'] = func_112
mod = relay.transform.InferType()(mod)
mutated_mod['func_112'] = func_112
mutated_mod = relay.transform.InferType()(mutated_mod)
func_112_call = mutated_mod.get_global_var('func_112')
var_114 = relay.var("var_114", dtype = "int64", shape = (2, 11, 8))#candidate|114|(2, 11, 8)|var|int64
var_115 = relay.var("var_115", dtype = "int64", shape = (2, 11, 8))#candidate|115|(2, 11, 8)|var|int64
var_116 = relay.var("var_116", dtype = "int64", shape = (2, 11, 8))#candidate|116|(2, 11, 8)|var|int64
call_113 = func_112_call(var_114,var_115,var_116,)
output = call_113
func_117 = relay.Function([var_114,var_115,var_116,], output)
mutated_mod['func_117'] = func_117
mutated_mod = relay.transform.InferType()(mutated_mod)
var_119 = relay.var("var_119", dtype = "float64", shape = (16, 9))#candidate|119|(16, 9)|var|float64
var_120 = relay.var("var_120", dtype = "float64", shape = (16, 9))#candidate|120|(16, 9)|var|float64
bop_121 = relay.floor_divide(var_119.astype('float64'), relay.reshape(var_120.astype('float64'), relay.shape_of(var_119))) # shape=(16, 9)
bop_124 = relay.divide(bop_121.astype('float32'), relay.reshape(var_119.astype('float32'), relay.shape_of(bop_121))) # shape=(16, 9)
uop_127 = relay.sigmoid(var_119.astype('float64')) # shape=(16, 9)
bop_129 = relay.bitwise_or(uop_127.astype('int8'), relay.reshape(bop_124.astype('int8'), relay.shape_of(uop_127))) # shape=(16, 9)
uop_132 = relay.erf(uop_127.astype('float32')) # shape=(16, 9)
output = relay.Tuple([bop_129,uop_132,])
output2 = relay.Tuple([bop_129,uop_132,])
func_134 = relay.Function([var_119,var_120,], output)
mod['func_134'] = func_134
mod = relay.transform.InferType()(mod)
var_135 = relay.var("var_135", dtype = "float64", shape = (16, 9))#candidate|135|(16, 9)|var|float64
var_136 = relay.var("var_136", dtype = "float64", shape = (16, 9))#candidate|136|(16, 9)|var|float64
output = func_134(var_135,var_136,)
func_137 = relay.Function([var_135,var_136,], output)
mutated_mod['func_137'] = func_137
mutated_mod = relay.transform.InferType()(mutated_mod)
var_139 = relay.var("var_139", dtype = "uint32", shape = (13, 11))#candidate|139|(13, 11)|var|uint32
const_140 = relay.const([[-5,-1,9,1,1,9,-8,-10,2,5,-8],[8,-7,5,6,5,-4,-10,-10,9,2,9],[6,-4,-9,-4,9,-5,4,8,7,-3,-4],[8,-6,5,-4,-8,6,-1,-1,2,-8,6],[-8,3,-2,10,1,-9,-4,-7,-9,-3,8],[-1,-8,9,-3,9,1,-9,-10,8,9,-10],[-2,-5,1,7,1,-8,-3,6,-4,-7,9],[-10,9,2,-7,-4,9,-9,7,-2,4,-7],[9,-2,8,-9,-7,8,1,5,2,7,7],[2,2,2,5,4,3,-8,7,-10,8,-2],[9,7,5,3,-7,-7,10,-6,9,9,-8],[-4,2,-1,3,-1,1,2,-9,-9,1,-7],[5,3,7,-9,7,-10,-7,-1,-4,-3,-10]], dtype = "uint32")#candidate|140|(13, 11)|const|uint32
bop_141 = relay.subtract(var_139.astype('uint32'), relay.reshape(const_140.astype('uint32'), relay.shape_of(var_139))) # shape=(13, 11)
output = bop_141
output2 = bop_141
func_144 = relay.Function([var_139,], output)
mod['func_144'] = func_144
mod = relay.transform.InferType()(mod)
var_145 = relay.var("var_145", dtype = "uint32", shape = (13, 11))#candidate|145|(13, 11)|var|uint32
output = func_144(var_145)
func_146 = relay.Function([var_145], output)
mutated_mod['func_146'] = func_146
mutated_mod = relay.transform.InferType()(mutated_mod)
const_148 = relay.const([[-2,8,-5,2,10,10,3,-8,-6,2,7,5,-9,9],[-10,5,-1,9,8,7,5,5,-1,-7,-5,4,10,-1]], dtype = "int64")#candidate|148|(2, 14)|const|int64
var_149 = relay.var("var_149", dtype = "int64", shape = (2, 14))#candidate|149|(2, 14)|var|int64
bop_150 = relay.maximum(const_148.astype('int64'), relay.reshape(var_149.astype('int64'), relay.shape_of(const_148))) # shape=(2, 14)
uop_153 = relay.acosh(bop_150.astype('float64')) # shape=(2, 14)
uop_155 = relay.log2(bop_150.astype('float32')) # shape=(2, 14)
bop_157 = relay.floor_mod(uop_153.astype('float32'), relay.reshape(var_149.astype('float32'), relay.shape_of(uop_153))) # shape=(2, 14)
uop_160 = relay.cos(const_148.astype('float32')) # shape=(2, 14)
bop_162 = relay.divide(uop_155.astype('float32'), relay.reshape(bop_157.astype('float32'), relay.shape_of(uop_155))) # shape=(2, 14)
uop_165 = relay.sin(uop_155.astype('float64')) # shape=(2, 14)
uop_167 = relay.sinh(uop_153.astype('float64')) # shape=(2, 14)
bop_169 = relay.minimum(uop_167.astype('uint8'), relay.reshape(bop_150.astype('uint8'), relay.shape_of(uop_167))) # shape=(2, 14)
bop_172 = relay.logical_or(uop_167.astype('bool'), relay.reshape(bop_157.astype('bool'), relay.shape_of(uop_167))) # shape=(2, 14)
output = relay.Tuple([uop_160,bop_162,uop_165,bop_169,bop_172,])
output2 = relay.Tuple([uop_160,bop_162,uop_165,bop_169,bop_172,])
func_175 = relay.Function([var_149,], output)
mod['func_175'] = func_175
mod = relay.transform.InferType()(mod)
var_176 = relay.var("var_176", dtype = "int64", shape = (2, 14))#candidate|176|(2, 14)|var|int64
output = func_175(var_176)
func_177 = relay.Function([var_176], output)
mutated_mod['func_177'] = func_177
mutated_mod = relay.transform.InferType()(mutated_mod)
const_179 = relay.const([5.733805,-1.441287,-8.132551,-8.780463,-7.552567,0.964838,-8.557931,-2.430178,8.955928,6.783285,-3.538277,-6.019776,1.128290,2.035985,-3.468615,-7.988861], dtype = "float32")#candidate|179|(16,)|const|float32
uop_180 = relay.log10(const_179.astype('float32')) # shape=(16,)
uop_182 = relay.sqrt(const_179.astype('float32')) # shape=(16,)
func_134_call = mod.get_global_var('func_134')
func_137_call = mutated_mod.get_global_var('func_137')
const_185 = relay.const([4.571358,-7.499261,8.057190,8.996447,4.298815,-6.748276,-1.737871,-5.319573,-5.125178,-9.802267,6.764592,0.854640,-6.028353,-8.367187,-2.351134,-9.340960,-9.138820,-7.426079,5.714090,-4.678409,-7.097213,-5.223320,9.131537,-8.283306,2.773789,6.881417,1.165182,4.467457,-6.664279,8.499820,6.579889,-4.712656,7.872967,1.391167,-2.254889,1.364231,9.465210,6.876845,-0.411653,-5.728636,6.860681,-2.932578,-1.757967,-8.429998,0.945599,1.901773,7.660271,9.157000,-9.540417,8.444464,-6.580126,0.501511,-1.218105,-9.689037,6.946704,6.778735,-7.967540,4.150677,9.724025,5.937659,-4.754072,4.395093,-1.165565,-8.696448,-4.164626,8.214225,-5.636348,-5.860321,-9.923582,8.790543,6.101422,-1.024884,-4.760678,-7.768812,2.990521,4.943009,-7.742306,-3.210293,3.044882,1.261842,-2.328062,-6.499706,8.352390,-9.021734,0.146912,6.802488,-3.728947,2.707442,-7.274206,-0.747096,9.990913,5.846975,-9.001364,7.095017,-8.914651,6.206102,0.699997,0.589312,-2.151304,-0.088980,7.929813,-7.174042,7.878580,-9.477848,8.187155,-7.127360,-8.618097,-7.394389,0.627863,-9.666882,-5.979312,-7.518782,7.005306,7.022017,4.120783,-4.799275,5.140028,-3.390025,-2.759346,-2.329965,9.094532,-3.090534,-6.364511,2.680779,-4.689975,9.925328,8.790848,3.039034,0.145863,7.817125,-6.468094,8.314133,-9.874589,1.553967,4.274854,6.437893,-0.535677,-9.228970,-2.218968,-9.999156,-9.461317,-7.886087,4.947064,-1.523564], dtype = "float64")#candidate|185|(144,)|const|float64
call_184 = relay.TupleGetItem(func_134_call(relay.reshape(const_185.astype('float64'), [16, 9]), relay.reshape(const_185.astype('float64'), [16, 9]), ), 1)
call_186 = relay.TupleGetItem(func_137_call(relay.reshape(const_185.astype('float64'), [16, 9]), relay.reshape(const_185.astype('float64'), [16, 9]), ), 1)
uop_187 = relay.acos(uop_180.astype('float64')) # shape=(16,)
bop_189 = relay.less_equal(uop_187.astype('bool'), relay.reshape(uop_180.astype('bool'), relay.shape_of(uop_187))) # shape=(16,)
uop_192 = relay.atanh(uop_182.astype('float32')) # shape=(16,)
uop_194 = relay.acosh(uop_187.astype('float64')) # shape=(16,)
bop_196 = relay.add(uop_180.astype('uint64'), relay.reshape(uop_187.astype('uint64'), relay.shape_of(uop_180))) # shape=(16,)
bop_199 = relay.mod(uop_192.astype('float64'), relay.reshape(uop_182.astype('float64'), relay.shape_of(uop_192))) # shape=(16,)
bop_202 = relay.floor_divide(uop_194.astype('float64'), relay.reshape(uop_180.astype('float64'), relay.shape_of(uop_194))) # shape=(16,)
uop_205 = relay.atan(uop_194.astype('float32')) # shape=(16,)
uop_207 = relay.cos(uop_205.astype('float64')) # shape=(16,)
uop_209 = relay.atanh(uop_207.astype('float32')) # shape=(16,)
uop_211 = relay.exp(uop_209.astype('float32')) # shape=(16,)
bop_213 = relay.greater_equal(uop_211.astype('bool'), relay.reshape(uop_209.astype('bool'), relay.shape_of(uop_211))) # shape=(16,)
uop_216 = relay.asin(uop_209.astype('float32')) # shape=(16,)
bop_218 = relay.logical_and(uop_209.astype('bool'), relay.reshape(bop_196.astype('bool'), relay.shape_of(uop_209))) # shape=(16,)
bop_221 = relay.logical_or(uop_207.astype('bool'), relay.reshape(uop_205.astype('bool'), relay.shape_of(uop_207))) # shape=(16,)
bop_224 = relay.floor_divide(uop_209.astype('float32'), relay.reshape(uop_216.astype('float32'), relay.shape_of(uop_209))) # shape=(16,)
bop_227 = relay.logical_xor(bop_224.astype('uint16'), relay.reshape(bop_199.astype('uint16'), relay.shape_of(bop_224))) # shape=(16,)
bop_230 = relay.greater_equal(uop_211.astype('bool'), relay.reshape(bop_227.astype('bool'), relay.shape_of(uop_211))) # shape=(16,)
uop_233 = relay.exp(uop_205.astype('float32')) # shape=(16,)
uop_235 = relay.sigmoid(uop_233.astype('float64')) # shape=(16,)
uop_237 = relay.asin(uop_207.astype('float32')) # shape=(16,)
bop_239 = relay.less(bop_221.astype('bool'), relay.reshape(uop_209.astype('bool'), relay.shape_of(bop_221))) # shape=(16,)
uop_242 = relay.atanh(uop_209.astype('float32')) # shape=(16,)
bop_244 = relay.logical_and(uop_216.astype('bool'), relay.reshape(uop_182.astype('bool'), relay.shape_of(uop_216))) # shape=(16,)
const_247 = relay.const([2,-10,-7,-7,6,-2,4,-10,10,1,-8,-3,1,8,3,3], dtype = "uint16")#candidate|247|(16,)|const|uint16
bop_248 = relay.left_shift(bop_227.astype('int16'), relay.reshape(const_247.astype('int16'), relay.shape_of(bop_227))) # shape=(16,)
bop_251 = relay.logical_and(bop_239.astype('bool'), relay.reshape(uop_180.astype('bool'), relay.shape_of(bop_239))) # shape=(16,)
func_82_call = mod.get_global_var('func_82')
func_90_call = mutated_mod.get_global_var('func_90')
var_255 = relay.var("var_255", dtype = "uint8", shape = ())#candidate|255|()|var|uint8
var_256 = relay.var("var_256", dtype = "float64", shape = (520,))#candidate|256|(520,)|var|float64
call_254 = relay.TupleGetItem(func_82_call(relay.reshape(var_255.astype('uint8'), []), relay.reshape(var_255.astype('uint8'), []), relay.reshape(var_256.astype('float64'), [8, 13, 5]), relay.reshape(var_256.astype('float32'), [8, 13, 5]), relay.reshape(var_256.astype('float32'), [8, 13, 5]), relay.reshape(var_256.astype('int8'), [8, 13, 5]), relay.reshape(var_256.astype('bool'), [8, 13, 5]), ), 5)
call_257 = relay.TupleGetItem(func_90_call(relay.reshape(var_255.astype('uint8'), []), relay.reshape(var_255.astype('uint8'), []), relay.reshape(var_256.astype('float64'), [8, 13, 5]), relay.reshape(var_256.astype('float32'), [8, 13, 5]), relay.reshape(var_256.astype('float32'), [8, 13, 5]), relay.reshape(var_256.astype('int8'), [8, 13, 5]), relay.reshape(var_256.astype('bool'), [8, 13, 5]), ), 5)
bop_258 = relay.mod(bop_202.astype('float64'), relay.reshape(uop_207.astype('float64'), relay.shape_of(bop_202))) # shape=(16,)
bop_261 = relay.logical_and(bop_258.astype('bool'), relay.reshape(bop_189.astype('bool'), relay.shape_of(bop_258))) # shape=(16,)
output = relay.Tuple([call_184,const_185,bop_213,bop_218,bop_230,uop_235,uop_237,uop_242,bop_244,bop_248,bop_251,call_254,var_255,var_256,bop_261,])
output2 = relay.Tuple([call_186,const_185,bop_213,bop_218,bop_230,uop_235,uop_237,uop_242,bop_244,bop_248,bop_251,call_257,var_255,var_256,bop_261,])
func_264 = relay.Function([var_255,var_256,], output)
mod['func_264'] = func_264
mod = relay.transform.InferType()(mod)
var_265 = relay.var("var_265", dtype = "uint8", shape = ())#candidate|265|()|var|uint8
var_266 = relay.var("var_266", dtype = "float64", shape = (520,))#candidate|266|(520,)|var|float64
output = func_264(var_265,var_266,)
func_267 = relay.Function([var_265,var_266,], output)
mutated_mod['func_267'] = func_267
mutated_mod = relay.transform.InferType()(mutated_mod)
var_269 = relay.var("var_269", dtype = "float32", shape = (8,))#candidate|269|(8,)|var|float32
var_270 = relay.var("var_270", dtype = "float32", shape = (8,))#candidate|270|(8,)|var|float32
bop_271 = relay.divide(var_269.astype('float32'), relay.reshape(var_270.astype('float32'), relay.shape_of(var_269))) # shape=(8,)
output = bop_271
output2 = bop_271
func_274 = relay.Function([var_269,var_270,], output)
mod['func_274'] = func_274
mod = relay.transform.InferType()(mod)
mutated_mod['func_274'] = func_274
mutated_mod = relay.transform.InferType()(mutated_mod)
func_274_call = mutated_mod.get_global_var('func_274')
var_276 = relay.var("var_276", dtype = "float32", shape = (8,))#candidate|276|(8,)|var|float32
var_277 = relay.var("var_277", dtype = "float32", shape = (8,))#candidate|277|(8,)|var|float32
call_275 = func_274_call(var_276,var_277,)
output = call_275
func_278 = relay.Function([var_276,var_277,], output)
mutated_mod['func_278'] = func_278
mutated_mod = relay.transform.InferType()(mutated_mod)
var_280 = relay.var("var_280", dtype = "float32", shape = (6,))#candidate|280|(6,)|var|float32
var_281 = relay.var("var_281", dtype = "float32", shape = (6,))#candidate|281|(6,)|var|float32
bop_282 = relay.not_equal(var_280.astype('bool'), relay.reshape(var_281.astype('bool'), relay.shape_of(var_280))) # shape=(6,)
uop_285 = relay.exp(var_280.astype('float64')) # shape=(6,)
var_287 = relay.var("var_287", dtype = "bool", shape = (6,))#candidate|287|(6,)|var|bool
bop_288 = relay.logical_xor(bop_282.astype('int32'), relay.reshape(var_287.astype('int32'), relay.shape_of(bop_282))) # shape=(6,)
uop_291 = relay.asinh(uop_285.astype('float64')) # shape=(6,)
uop_293 = relay.cosh(var_281.astype('float32')) # shape=(6,)
uop_295 = relay.asin(uop_291.astype('float64')) # shape=(6,)
var_297 = relay.var("var_297", dtype = "float64", shape = (6,))#candidate|297|(6,)|var|float64
bop_298 = relay.logical_and(uop_291.astype('bool'), relay.reshape(var_297.astype('bool'), relay.shape_of(uop_291))) # shape=(6,)
uop_301 = relay.tan(uop_295.astype('float32')) # shape=(6,)
uop_303 = relay.asinh(uop_293.astype('float32')) # shape=(6,)
uop_305 = relay.asinh(uop_301.astype('float32')) # shape=(6,)
uop_307 = relay.sigmoid(uop_301.astype('float32')) # shape=(6,)
uop_309 = relay.log(uop_301.astype('float64')) # shape=(6,)
uop_311 = relay.acosh(uop_307.astype('float64')) # shape=(6,)
uop_313 = relay.log10(uop_311.astype('float32')) # shape=(6,)
uop_315 = relay.tan(uop_313.astype('float32')) # shape=(6,)
uop_317 = relay.tan(uop_313.astype('float32')) # shape=(6,)
bop_319 = relay.bitwise_or(uop_317.astype('uint32'), relay.reshape(var_297.astype('uint32'), relay.shape_of(uop_317))) # shape=(6,)
uop_322 = relay.cos(uop_313.astype('float64')) # shape=(6,)
output = relay.Tuple([bop_288,bop_298,uop_303,uop_305,uop_309,uop_315,bop_319,uop_322,])
output2 = relay.Tuple([bop_288,bop_298,uop_303,uop_305,uop_309,uop_315,bop_319,uop_322,])
func_324 = relay.Function([var_280,var_281,var_287,var_297,], output)
mod['func_324'] = func_324
mod = relay.transform.InferType()(mod)
mutated_mod['func_324'] = func_324
mutated_mod = relay.transform.InferType()(mutated_mod)
func_324_call = mutated_mod.get_global_var('func_324')
var_326 = relay.var("var_326", dtype = "float32", shape = (6,))#candidate|326|(6,)|var|float32
var_327 = relay.var("var_327", dtype = "float32", shape = (6,))#candidate|327|(6,)|var|float32
var_328 = relay.var("var_328", dtype = "bool", shape = (6,))#candidate|328|(6,)|var|bool
var_329 = relay.var("var_329", dtype = "float64", shape = (6,))#candidate|329|(6,)|var|float64
call_325 = func_324_call(var_326,var_327,var_328,var_329,)
output = call_325
func_330 = relay.Function([var_326,var_327,var_328,var_329,], output)
mutated_mod['func_330'] = func_330
mutated_mod = relay.transform.InferType()(mutated_mod)
var_332 = relay.var("var_332", dtype = "uint64", shape = (10,))#candidate|332|(10,)|var|uint64
var_333 = relay.var("var_333", dtype = "uint64", shape = (10,))#candidate|333|(10,)|var|uint64
bop_334 = relay.bitwise_or(var_332.astype('uint64'), relay.reshape(var_333.astype('uint64'), relay.shape_of(var_332))) # shape=(10,)
bop_337 = relay.left_shift(var_333.astype('uint8'), relay.reshape(bop_334.astype('uint8'), relay.shape_of(var_333))) # shape=(10,)
output = relay.Tuple([bop_337,])
output2 = relay.Tuple([bop_337,])
F = relay.Function([var_332,var_333,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_332,var_333,], output2)
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
input_332= np.array([6,-2,-8,4,-3,10,-4,7,-3,10], dtype='uint64')
module1.set_input('var_332', input_332)
input_333= np.array([-8,7,-3,-8,5,-2,-7,2,7,5], dtype='uint64')
module1.set_input('var_333', input_333)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_332, input_333, )
res3 = intrp3.evaluate()(input_332, input_333, )
res4 = intrp4.evaluate()(input_332, input_333, )
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
module5.set_input('var_332', input_332)
module5.set_input('var_333', input_333)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_332, input_333, )
res7 = intrp7.evaluate()(input_332, input_333, )
res8 = intrp8.evaluate()(input_332, input_333, )
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
module9.set_input('var_332', input_332)
module9.set_input('var_333', input_333)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_332, input_333, )
res11 = intrp11.evaluate()(input_332, input_333, )
res12 = intrp12.evaluate()(input_332, input_333, )
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
module13.set_input('var_332', input_332)
module13.set_input('var_333', input_333)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_332, input_333, )
res15 = intrp15.evaluate()(input_332, input_333, )
res16 = intrp16.evaluate()(input_332, input_333, )
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
module17.set_input('var_332', input_332)
module17.set_input('var_333', input_333)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_332, input_333, )
res19 = intrp19.evaluate()(input_332, input_333, )
res20 = intrp20.evaluate()(input_332, input_333, )
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
module21.set_input('var_332', input_332)
module21.set_input('var_333', input_333)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_332, input_333, )
res23 = intrp23.evaluate()(input_332, input_333, )
res24 = intrp24.evaluate()(input_332, input_333, )
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

'''22: TVMFuncCall
21: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}>(tvm::transform::{lambda(tvm::transform::Pass, tvm::IRModule)#7}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
20: tvm::transform::Pass::operator()(tvm::IRModule) const
19: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
18: tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
17: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
16: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
15: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}>(tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
14: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const [clone .isra.405]
13: tvm::relay::transform::FirstOrderGradient()::{lambda(tvm::IRModule, tvm::transform::PassContext)#1}::operator()(tvm::IRModule, tvm::transform::PassContext) const::{lambda(tvm::relay::LetList*)#1}::operator()(tvm::relay::LetList) const [clone .constprop.436]
12: _ZNSt17_Function_handlerIFSt10sha
11: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::FunctionNode const*)::{lambda(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&)#1}::operator()(std::vector<std::shared_ptr<tvm::relay::ADValueNode>, std::allocator<std::shared_ptr<tvm::relay::ADValueNode> > > const&, tvm::relay::Call const&) const
10: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
9: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
8: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::TupleNode const*)
7: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
5: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::FirstOrderReverseAD::VisitExpr(tvm::RelayExpr const&)
3: _ZZN3tvm5relay11ExprFunctorIFSt10shared_
2: tvm::relay::FirstOrderReverseAD::VisitExpr_(tvm::OpNode const*)
1: _ZN3tvm17DiagnosticContext9EmitFatalERKNS_1
0: tvm::DiagnosticContext::Render()

'''