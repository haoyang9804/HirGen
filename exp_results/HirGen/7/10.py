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
var_389 = relay.var("var_389", dtype = "float64", shape = (13, 5, 5))#candidate|389|(13, 5, 5)|var|float64
uop_390 = relay.sqrt(var_389.astype('float64')) # shape=(13, 5, 5)
bop_393 = relay.floor_divide(var_389.astype('float64'), relay.reshape(uop_390.astype('float64'), relay.shape_of(var_389))) # shape=(13, 5, 5)
output = bop_393
output2 = bop_393
func_418 = relay.Function([var_389,], output)
mod['func_418'] = func_418
mod = relay.transform.InferType()(mod)
mutated_mod['func_418'] = func_418
mutated_mod = relay.transform.InferType()(mutated_mod)
var_419 = relay.var("var_419", dtype = "float64", shape = (13, 5, 5))#candidate|419|(13, 5, 5)|var|float64
func_418_call = mutated_mod.get_global_var('func_418')
call_420 = func_418_call(var_419)
output = call_420
func_421 = relay.Function([var_419], output)
mutated_mod['func_421'] = func_421
mutated_mod = relay.transform.InferType()(mutated_mod)
var_558 = relay.var("var_558", dtype = "float32", shape = (16, 6, 6))#candidate|558|(16, 6, 6)|var|float32
uop_559 = relay.asinh(var_558.astype('float32')) # shape=(16, 6, 6)
bop_564 = relay.add(var_558.astype('uint16'), relay.reshape(uop_559.astype('uint16'), relay.shape_of(var_558))) # shape=(16, 6, 6)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
var_568 = relay.var("var_568", dtype = "float64", shape = (325,))#candidate|568|(325,)|var|float64
call_567 = func_418_call(relay.reshape(var_568.astype('float64'), [13, 5, 5]))
call_569 = func_418_call(relay.reshape(var_568.astype('float64'), [13, 5, 5]))
output = relay.Tuple([bop_564,call_567,var_568,])
output2 = relay.Tuple([bop_564,call_569,var_568,])
func_571 = relay.Function([var_558,var_568,], output)
mod['func_571'] = func_571
mod = relay.transform.InferType()(mod)
mutated_mod['func_571'] = func_571
mutated_mod = relay.transform.InferType()(mutated_mod)
func_571_call = mutated_mod.get_global_var('func_571')
var_573 = relay.var("var_573", dtype = "float32", shape = (16, 6, 6))#candidate|573|(16, 6, 6)|var|float32
var_574 = relay.var("var_574", dtype = "float64", shape = (325,))#candidate|574|(325,)|var|float64
call_572 = func_571_call(var_573,var_574,)
output = call_572
func_575 = relay.Function([var_573,var_574,], output)
mutated_mod['func_575'] = func_575
mutated_mod = relay.transform.InferType()(mutated_mod)
var_815 = relay.var("var_815", dtype = "float64", shape = (14, 10))#candidate|815|(14, 10)|var|float64
uop_816 = relay.sinh(var_815.astype('float64')) # shape=(14, 10)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
var_834 = relay.var("var_834", dtype = "float64", shape = (325,))#candidate|834|(325,)|var|float64
call_833 = func_418_call(relay.reshape(var_834.astype('float64'), [13, 5, 5]))
call_835 = func_418_call(relay.reshape(var_834.astype('float64'), [13, 5, 5]))
output = relay.Tuple([uop_816,call_833,var_834,])
output2 = relay.Tuple([uop_816,call_835,var_834,])
func_838 = relay.Function([var_815,var_834,], output)
mod['func_838'] = func_838
mod = relay.transform.InferType()(mod)
mutated_mod['func_838'] = func_838
mutated_mod = relay.transform.InferType()(mutated_mod)
func_838_call = mutated_mod.get_global_var('func_838')
var_840 = relay.var("var_840", dtype = "float64", shape = (14, 10))#candidate|840|(14, 10)|var|float64
var_841 = relay.var("var_841", dtype = "float64", shape = (325,))#candidate|841|(325,)|var|float64
call_839 = func_838_call(var_840,var_841,)
output = call_839
func_842 = relay.Function([var_840,var_841,], output)
mutated_mod['func_842'] = func_842
mutated_mod = relay.transform.InferType()(mutated_mod)
var_844 = relay.var("var_844", dtype = "float32", shape = (10, 3, 11))#candidate|844|(10, 3, 11)|var|float32
uop_845 = relay.atan(var_844.astype('float32')) # shape=(10, 3, 11)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
var_849 = relay.var("var_849", dtype = "float64", shape = (325,))#candidate|849|(325,)|var|float64
call_848 = func_418_call(relay.reshape(var_849.astype('float64'), [13, 5, 5]))
call_850 = func_418_call(relay.reshape(var_849.astype('float64'), [13, 5, 5]))
output = relay.Tuple([uop_845,call_848,var_849,])
output2 = relay.Tuple([uop_845,call_850,var_849,])
func_851 = relay.Function([var_844,var_849,], output)
mod['func_851'] = func_851
mod = relay.transform.InferType()(mod)
mutated_mod['func_851'] = func_851
mutated_mod = relay.transform.InferType()(mutated_mod)
func_851_call = mutated_mod.get_global_var('func_851')
var_853 = relay.var("var_853", dtype = "float32", shape = (10, 3, 11))#candidate|853|(10, 3, 11)|var|float32
var_854 = relay.var("var_854", dtype = "float64", shape = (325,))#candidate|854|(325,)|var|float64
call_852 = func_851_call(var_853,var_854,)
output = call_852
func_855 = relay.Function([var_853,var_854,], output)
mutated_mod['func_855'] = func_855
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1096 = relay.var("var_1096", dtype = "float64", shape = (12, 6, 11))#candidate|1096|(12, 6, 11)|var|float64
var_1097 = relay.var("var_1097", dtype = "float64", shape = (12, 6, 11))#candidate|1097|(12, 6, 11)|var|float64
bop_1098 = relay.floor_mod(var_1096.astype('float64'), relay.reshape(var_1097.astype('float64'), relay.shape_of(var_1096))) # shape=(12, 6, 11)
bop_1105 = relay.less_equal(var_1096.astype('bool'), relay.reshape(bop_1098.astype('bool'), relay.shape_of(var_1096))) # shape=(12, 6, 11)
uop_1109 = relay.erf(bop_1098.astype('float32')) # shape=(12, 6, 11)
func_838_call = mod.get_global_var('func_838')
func_842_call = mutated_mod.get_global_var('func_842')
const_1126 = relay.const([[-4.712095,6.922910],[-3.860178,-5.308136],[-2.538265,5.581789],[-3.197827,-7.496300],[7.601065,2.972009],[-0.654466,-2.424938],[5.410275,3.120718],[2.323149,2.065349],[-6.038184,1.526269],[1.239853,1.864718],[-2.816378,5.527380],[1.489344,1.722155],[5.128982,1.347372],[-6.608994,-8.772162],[-7.482106,9.256932],[9.708209,2.965198],[5.427835,0.069671],[-4.544659,-4.420839],[5.764647,-1.406701],[-3.453365,1.563514],[-4.606867,1.540171],[3.569981,-1.131902],[-8.305864,2.193594],[5.605203,8.984429],[0.902952,6.106493],[7.649450,-8.402346],[0.572528,6.653199],[-5.310903,1.645325],[0.610820,6.270208],[-4.207211,-2.490274],[-2.733838,-8.273766],[2.062974,-0.040582],[-8.419313,-7.829936],[-9.144288,-3.882308],[7.742718,5.414473],[5.530865,0.563773],[3.635027,6.571249],[-0.521349,-9.184134],[0.439946,-6.241793],[-3.389310,7.440171],[-9.552434,-0.494733],[-8.423300,-5.445064],[-9.844199,2.420007],[-5.355784,0.335273],[-3.372001,0.367433],[-4.578672,8.657069],[5.430287,-5.001248],[-4.952156,-3.097002],[-1.443509,5.742914],[-5.944011,-7.474606],[1.466105,-7.810043],[-6.846050,-5.469077],[-7.178953,-6.886877],[3.487496,-0.172660],[4.372228,5.514298],[-4.030298,1.265151],[-8.009035,3.556207],[-4.667056,-3.864526],[0.917593,-9.117995],[-9.812425,-8.932243],[-1.760089,2.294545],[-3.183621,-5.422607],[1.937686,-8.831573],[-9.997503,-5.865818],[-2.801086,8.710240],[9.380581,1.185137],[-5.558523,0.030815],[-5.242095,5.873701],[-5.588679,-5.472319],[-5.981714,-0.999597]], dtype = "float64")#candidate|1126|(70, 2)|const|float64
var_1127 = relay.var("var_1127", dtype = "float64", shape = (325,))#candidate|1127|(325,)|var|float64
call_1125 = relay.TupleGetItem(func_838_call(relay.reshape(const_1126.astype('float64'), [14, 10]), relay.reshape(var_1127.astype('float64'), [325,]), ), 0)
call_1128 = relay.TupleGetItem(func_842_call(relay.reshape(const_1126.astype('float64'), [14, 10]), relay.reshape(var_1127.astype('float64'), [325,]), ), 0)
bop_1140 = relay.less(var_1097.astype('bool'), relay.reshape(bop_1098.astype('bool'), relay.shape_of(var_1097))) # shape=(12, 6, 11)
output = relay.Tuple([bop_1105,uop_1109,call_1125,const_1126,var_1127,bop_1140,])
output2 = relay.Tuple([bop_1105,uop_1109,call_1128,const_1126,var_1127,bop_1140,])
func_1156 = relay.Function([var_1096,var_1097,var_1127,], output)
mod['func_1156'] = func_1156
mod = relay.transform.InferType()(mod)
var_1157 = relay.var("var_1157", dtype = "float64", shape = (12, 6, 11))#candidate|1157|(12, 6, 11)|var|float64
var_1158 = relay.var("var_1158", dtype = "float64", shape = (12, 6, 11))#candidate|1158|(12, 6, 11)|var|float64
var_1159 = relay.var("var_1159", dtype = "float64", shape = (325,))#candidate|1159|(325,)|var|float64
output = func_1156(var_1157,var_1158,var_1159,)
func_1160 = relay.Function([var_1157,var_1158,var_1159,], output)
mutated_mod['func_1160'] = func_1160
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1335 = relay.const([[[True,True,False,False,False,True,True,True,False,True],[False,True,False,True,False,False,False,True,True,True],[False,True,True,False,True,True,True,True,True,True],[True,True,False,True,True,False,False,False,True,True],[False,True,False,False,True,False,True,True,False,False],[True,False,True,False,False,False,False,True,True,True],[False,False,False,False,True,False,True,True,False,False],[False,False,False,False,True,True,True,False,False,True],[False,True,True,True,False,True,True,False,False,False],[True,True,False,False,True,True,False,False,False,False],[True,True,True,True,True,False,False,False,False,False]],[[True,False,False,False,True,False,True,False,False,False],[False,True,True,False,True,False,True,True,True,True],[False,False,False,True,True,False,True,True,False,True],[True,False,True,True,False,False,True,False,False,True],[False,False,True,True,False,False,True,True,False,False],[False,False,False,True,True,True,True,False,False,True],[True,False,True,False,True,False,False,True,False,False],[False,False,False,True,True,False,False,False,True,False],[True,True,False,True,False,True,True,True,True,True],[True,False,True,False,False,True,False,False,False,False],[False,False,False,False,False,True,False,False,False,True]],[[False,True,False,False,False,True,True,True,False,False],[True,True,False,False,False,False,True,False,False,True],[True,False,False,True,False,False,True,False,False,True],[True,False,False,True,False,False,False,True,False,True],[True,True,False,True,True,False,True,True,True,True],[False,False,True,False,True,True,False,False,True,False],[True,False,False,True,True,False,False,False,True,False],[True,False,True,True,True,False,False,False,True,True],[True,False,True,False,False,False,True,True,True,False],[True,False,False,False,False,False,False,False,False,False],[False,True,False,True,False,False,True,False,False,True]],[[True,False,True,False,False,True,True,False,False,False],[False,False,False,True,False,False,True,False,False,True],[False,False,False,True,True,False,True,True,True,True],[False,False,True,True,True,False,False,False,False,True],[False,False,True,False,True,True,True,False,True,True],[True,False,False,True,True,True,False,False,False,True],[True,False,True,True,True,False,True,False,False,True],[True,False,True,False,True,True,True,False,True,False],[True,True,False,True,False,True,True,False,True,True],[True,True,False,True,False,True,True,True,True,False],[False,False,False,True,False,True,False,True,True,False]],[[False,True,True,False,False,True,False,True,False,True],[True,True,False,True,False,False,False,False,True,False],[False,True,False,False,True,True,False,True,False,True],[True,False,False,False,True,True,False,True,False,False],[False,True,True,True,False,False,True,True,False,True],[True,False,False,True,False,True,False,False,True,True],[False,False,True,False,True,False,True,True,True,False],[True,False,True,False,True,False,False,False,True,False],[True,False,False,False,True,True,True,False,True,False],[True,True,True,False,False,False,True,True,True,False],[True,False,False,True,False,True,True,True,False,False]]], dtype = "bool")#candidate|1335|(5, 11, 10)|const|bool
var_1336 = relay.var("var_1336", dtype = "bool", shape = (5, 11, 10))#candidate|1336|(5, 11, 10)|var|bool
bop_1337 = relay.logical_and(const_1335.astype('bool'), relay.reshape(var_1336.astype('bool'), relay.shape_of(const_1335))) # shape=(5, 11, 10)
output = relay.Tuple([bop_1337,])
output2 = relay.Tuple([bop_1337,])
func_1345 = relay.Function([var_1336,], output)
mod['func_1345'] = func_1345
mod = relay.transform.InferType()(mod)
var_1346 = relay.var("var_1346", dtype = "bool", shape = (5, 11, 10))#candidate|1346|(5, 11, 10)|var|bool
output = func_1345(var_1346)
func_1347 = relay.Function([var_1346], output)
mutated_mod['func_1347'] = func_1347
mutated_mod = relay.transform.InferType()(mutated_mod)
const_1472 = relay.const([[[1],[7],[-3],[-1],[-1],[-5],[-10],[-7],[-7],[-3],[-5],[6],[7],[4]],[[-9],[1],[2],[-2],[-9],[5],[1],[2],[10],[10],[-8],[10],[5],[-5]],[[-5],[4],[-9],[-4],[4],[-5],[10],[5],[-9],[-5],[-8],[-8],[9],[-2]],[[-3],[-5],[-3],[2],[3],[-4],[-5],[2],[-10],[-9],[5],[9],[10],[-2]],[[5],[-3],[8],[-7],[-8],[-6],[9],[10],[2],[1],[8],[-2],[-5],[-6]],[[8],[6],[-1],[-4],[-7],[-3],[-1],[8],[10],[6],[-1],[-3],[-9],[3]],[[6],[1],[-9],[-9],[-5],[9],[-3],[-9],[-1],[8],[6],[-1],[-3],[5]],[[9],[-5],[7],[-1],[2],[7],[5],[1],[8],[4],[-8],[3],[6],[-2]],[[-10],[10],[5],[4],[-3],[10],[-9],[5],[3],[-1],[6],[2],[-8],[10]],[[9],[-5],[-9],[8],[-10],[7],[1],[9],[4],[-1],[-9],[-10],[8],[9]]], dtype = "int32")#candidate|1472|(10, 14, 1)|const|int32
var_1473 = relay.var("var_1473", dtype = "int32", shape = (10, 14, 1))#candidate|1473|(10, 14, 1)|var|int32
bop_1474 = relay.greater_equal(const_1472.astype('bool'), relay.reshape(var_1473.astype('bool'), relay.shape_of(const_1472))) # shape=(10, 14, 1)
output = bop_1474
output2 = bop_1474
func_1498 = relay.Function([var_1473,], output)
mod['func_1498'] = func_1498
mod = relay.transform.InferType()(mod)
mutated_mod['func_1498'] = func_1498
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1499 = relay.var("var_1499", dtype = "int32", shape = (10, 14, 1))#candidate|1499|(10, 14, 1)|var|int32
func_1498_call = mutated_mod.get_global_var('func_1498')
call_1500 = func_1498_call(var_1499)
output = call_1500
func_1501 = relay.Function([var_1499], output)
mutated_mod['func_1501'] = func_1501
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1566 = relay.var("var_1566", dtype = "float32", shape = (3, 16, 10))#candidate|1566|(3, 16, 10)|var|float32
uop_1567 = relay.acos(var_1566.astype('float32')) # shape=(3, 16, 10)
func_851_call = mod.get_global_var('func_851')
func_855_call = mutated_mod.get_global_var('func_855')
const_1570 = relay.const([[0.802741],[5.352397],[-9.749826],[8.071016],[-5.863215],[-7.155929],[-9.284804],[7.320820],[1.525322],[-0.693304],[-2.732883],[5.239864],[5.305182],[0.580258],[-5.741600],[-1.447403],[7.739644],[-8.319048],[9.047015],[-8.126106],[-7.217176],[5.938932],[-3.062013],[8.072277],[-4.930739],[-0.521145],[-9.071406],[9.795852],[-1.359624],[4.127191],[8.087009],[-1.948823],[5.659312],[-1.885440],[-1.002927],[4.121519],[9.828517],[8.513987],[3.578523],[0.715901],[1.024784],[-8.982795],[-3.479212],[3.670022],[9.516273],[3.180696],[-5.098222],[1.529346],[-9.373856],[6.859776],[0.218667],[7.062526],[6.153002],[-7.781140],[2.964742],[-2.488712],[6.386018],[9.465871],[-6.863377],[6.342279],[0.525035],[-6.126078],[-7.546058],[-6.775824],[5.703118],[-0.336234],[-9.631629],[-0.388761],[-6.326903],[-6.383356],[-9.705520],[-0.138544],[5.076174],[2.128805],[-1.915362],[3.718180],[-0.689193],[-1.766615],[-0.397018],[6.921624],[-8.253804],[-4.308142],[-9.887659],[-9.800198],[6.060068],[-8.915085],[8.836746],[3.817074],[-0.426221],[-4.091690],[5.953699],[5.193672],[-3.164957],[2.655720],[-2.354998],[-5.356607],[-8.451708],[4.673030],[8.418866],[0.655371],[-8.763240],[6.694502],[9.004667],[-8.704863],[3.851167],[-8.671922],[9.893339],[-1.989674],[4.526804],[-5.104916],[9.539833],[-3.423537],[4.745412],[-0.741125],[0.838090],[2.376564],[7.513883],[5.496049],[7.288189],[5.421273],[4.921268],[-9.684262],[8.936163],[-5.067935],[-8.644144],[-0.785749],[5.006352],[0.652968],[-2.089604],[-3.265864],[3.597475],[3.210495],[-1.862273],[7.785986],[-7.947741],[-1.712417],[1.723678],[-1.251397],[-5.049423],[6.592918],[3.447864],[-0.597036],[7.979500],[8.528750],[1.838155],[3.419316],[3.022433],[0.731681],[-8.574350],[-0.170354],[0.989075],[-2.546194],[9.727844],[2.183676],[5.152087],[-7.393801],[-4.328583],[5.389184],[-6.816272],[2.284431],[-0.699550],[-1.326340],[8.945643],[1.256651],[9.225465],[3.906731],[6.334455],[8.950685],[-4.021087],[-7.092726],[5.327995],[7.376223],[4.732340],[4.058391],[6.329261],[6.851130],[-8.229348],[-1.341732],[7.382790],[7.073733],[6.651888],[-5.988034],[-7.174498],[5.262744],[-5.686919],[-9.695350],[-2.554698],[-2.732460],[-3.195413],[5.460708],[-4.959769],[-9.825120],[-8.118257],[1.401196],[-8.131175],[-2.855003],[8.260823],[-6.123198],[-8.482654],[-3.356452],[7.120117],[4.868691],[-8.010978],[1.736013],[2.924451],[8.945925],[9.341843],[-9.478750],[-1.254928],[-2.381395],[-9.548304],[-3.777086],[2.500495],[6.805129],[-9.805964],[-3.650673],[-0.758521],[1.723335],[7.398669],[-5.947513],[2.337554],[3.544415],[9.518131],[-1.963810],[9.959044],[-0.854305],[-3.439145],[7.622811],[-9.020325],[4.208571],[-0.247788],[-3.258640],[1.575679],[-8.767104],[8.736466],[-7.551162],[-6.711457],[0.291523],[7.396577],[6.671747],[-4.960609],[9.580078],[-1.030426],[-5.258050],[6.696781],[1.624294],[-6.528378],[1.799182],[8.680651],[-0.389354],[-3.603700],[7.435354],[-9.479649],[-2.037159],[-3.918471],[-0.654811],[9.196375],[3.649133],[-0.598003],[-4.272170],[-2.065687],[-5.578250],[-1.610861],[-8.409115],[7.309678],[-7.086467],[6.189769],[-9.514936],[-0.468904],[-8.310573],[-8.090351],[-4.100683],[-9.773090],[5.594869],[9.353961],[-1.835502],[1.074365],[-2.858925],[-9.007339],[0.985584],[-2.089171],[4.296166],[1.444723],[7.924488],[-5.687403],[-9.038455],[3.295750],[8.139885],[9.450839],[-6.238573],[-2.913220],[-6.423644],[2.421648],[1.039439],[4.676573],[3.928077],[-5.450434],[-6.633937],[-2.202985],[-3.982025],[3.182821],[8.540917],[2.860393],[-7.459716],[2.317821],[2.263840],[9.699863],[6.405089],[7.813891],[4.828382],[3.128308],[-3.314002],[7.459489],[7.091833],[-1.146086],[4.156274],[9.798790],[-6.184342],[8.327263],[7.457602],[-9.469254],[-9.078020],[-7.402556],[-2.543586],[9.101512],[-0.627881],[4.769060],[-3.880350],[0.470211],[-4.316849]], dtype = "float32")#candidate|1570|(330, 1)|const|float32
const_1571 = relay.const([0.002411,2.829199,-3.770592,9.699487,-0.404070,3.883336,-6.538722,-9.265531,1.439916,-0.469742,1.809077,3.463620,3.679551,-8.119528,6.370325,3.040135,-5.612729,8.616957,0.469962,7.840007,8.099746,-0.933717,-2.407810,5.542929,0.816490,-3.659458,-7.632112,-2.794922,-3.496643,9.060139,-6.413868,-0.727996,3.039342,-4.983674,-9.985662,-8.863356,-6.994317,6.308937,5.387690,-5.035934,6.530084,8.079523,-4.651330,5.078131,9.149015,7.823797,3.394780,-1.136821,1.062857,9.188558,-4.805167,8.496040,-0.131406,6.505072,4.261813,-5.837659,8.955289,5.703300,4.832936,-9.184097,0.839539,0.280531,9.941626,5.196373,-3.267413,-2.705698,2.628966,3.944987,1.868176,5.715829,-1.525847,1.259070,-8.952840,5.200735,7.805044,-5.017025,9.763298,8.381113,0.184599,-5.143001,6.025486,-6.942598,-4.086129,-3.670704,-9.887442,4.841504,9.225176,-8.154659,1.441818,-4.440595,6.474136,2.920614,-8.226489,1.696444,6.115188,8.990649,0.418684,1.976970,5.368689,7.395025,-6.733431,-9.193468,2.519234,7.666712,-9.829222,4.645177,-7.415011,4.779691,1.652602,-3.110948,-2.686199,9.202424,3.230633,0.829587,-9.344149,-8.463958,3.179533,-2.384248,-9.638207,2.576554,6.859203,-4.884764,0.785805,-3.411467,3.662344,-8.848197,-9.756078,5.893890,-5.740894,-7.751043,9.124184,7.024843,1.816918,7.241051,5.711557,9.080564,5.675605,-6.548521,9.985047,4.688070,1.945491,-5.160155,8.374781,2.312824,-1.342151,-1.346020,1.722236,9.841048,4.815251,-7.159001,5.666882,-0.385398,6.025881,-3.720840,4.585379,6.721366,5.112496,1.914312,4.801637,-4.690205,-6.412678,9.833676,3.971463,7.331557,2.927854,1.025221,1.111777,7.888562,2.614749,7.933546,3.366008,0.733573,-9.295921,2.816522,-5.803163,6.727164,-4.089812,-0.710056,0.399101,1.566398,6.027268,-0.753637,-8.647591,8.030708,4.837228,-1.102067,-6.999424,0.282399,-7.981838,-3.194379,-8.679683,-0.455185,3.853273,-1.646995,-8.350214,2.399828,9.243019,-2.010538,-4.411334,5.715057,-4.693127,-3.536738,9.440536,0.922845,6.174900,-0.475429,0.001988,-2.618310,7.648804,-4.048623,-7.918163,2.661921,5.868585,4.113700,-7.126447,1.167559,0.679528,8.731017,8.711280,-7.898223,-1.828699,-2.123979,5.028529,-5.549758,-6.037386,-3.507284,4.270228,-0.038327,3.397732,-0.935255,-8.380290,-0.198226,0.181740,-6.200865,-3.529424,5.416227,-4.298169,4.660511,3.266172,-9.591357,-3.877346,-9.806032,7.025773,-1.650558,6.135667,-4.491556,-8.977459,5.374975,-2.792434,-1.182195,-3.915154,3.902498,9.600998,3.118452,0.669687,-8.755156,-3.808135,0.496564,7.854564,4.347943,-4.543425,-3.453385,-3.368776,-9.365267,-7.595894,-5.332972,8.684627,8.467453,-4.230243,7.850109,-5.640826,8.044934,-0.745571,-1.734372,-9.361178,-3.927482,2.307561,1.712988,-5.476406,-1.682345,-8.877939,-8.442950,5.871450,-0.841962,-2.577782,2.688279,-4.596573,6.014422,7.799556,-1.812698,8.054070,-4.941356,-2.648034,9.825133,8.024742,2.507897,-5.982838,1.198226,6.274131,3.376047,4.234076,-5.629746,1.038457,5.034626,-8.661296,-3.396255,-8.692555,-8.630341,-9.146678,-1.055888,-3.940511,6.354446,6.861660,-4.782409,-0.343440,-2.237341,-3.745784,8.546745,-1.026499,-7.171783,2.596348,7.316617,0.523403,-6.641109,3.541491], dtype = "float64")#candidate|1571|(325,)|const|float64
call_1569 = relay.TupleGetItem(func_851_call(relay.reshape(const_1570.astype('float32'), [10, 3, 11]), relay.reshape(const_1571.astype('float64'), [325,]), ), 2)
call_1572 = relay.TupleGetItem(func_855_call(relay.reshape(const_1570.astype('float32'), [10, 3, 11]), relay.reshape(const_1571.astype('float64'), [325,]), ), 2)
output = relay.Tuple([uop_1567,call_1569,const_1570,const_1571,])
output2 = relay.Tuple([uop_1567,call_1572,const_1570,const_1571,])
func_1574 = relay.Function([var_1566,], output)
mod['func_1574'] = func_1574
mod = relay.transform.InferType()(mod)
mutated_mod['func_1574'] = func_1574
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1575 = relay.var("var_1575", dtype = "float32", shape = (3, 16, 10))#candidate|1575|(3, 16, 10)|var|float32
func_1574_call = mutated_mod.get_global_var('func_1574')
call_1576 = func_1574_call(var_1575)
output = call_1576
func_1577 = relay.Function([var_1575], output)
mutated_mod['func_1577'] = func_1577
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1688 = relay.var("var_1688", dtype = "float64", shape = (7, 10, 12))#candidate|1688|(7, 10, 12)|var|float64
var_1689 = relay.var("var_1689", dtype = "float64", shape = (7, 10, 12))#candidate|1689|(7, 10, 12)|var|float64
bop_1690 = relay.floor_divide(var_1688.astype('float64'), relay.reshape(var_1689.astype('float64'), relay.shape_of(var_1688))) # shape=(7, 10, 12)
bop_1697 = relay.maximum(var_1689.astype('int32'), relay.reshape(bop_1690.astype('int32'), relay.shape_of(var_1689))) # shape=(7, 10, 12)
func_838_call = mod.get_global_var('func_838')
func_842_call = mutated_mod.get_global_var('func_842')
const_1703 = relay.const([-0.633271,-0.330529,9.589561,-3.323895,6.025514,5.909090,-3.411801,9.144907,7.800823,8.228510,4.493828,3.727107,-6.675906,-2.614327,-1.249516,3.512179,5.504030,9.963076,-4.319220,4.808040,-3.789222,9.885114,2.678117,7.293190,-9.595028,-9.935272,3.704602,6.072345,3.783385,4.136757,4.694180,-9.474693,-6.940748,4.829241,6.973613,-5.422247,-6.494206,-1.226838,-0.929660,-8.501962,3.930950,-3.032328,-0.221028,-9.196352,6.626818,6.745848,-0.166923,-4.450999,5.518310,7.451535,-9.512314,-0.058493,-6.779520,-6.359127,9.152256,9.491839,-3.389058,1.340276,6.874343,-5.946660,9.204347,6.383316,4.204069,-7.172440,-6.363604,4.742918,7.825059,3.708891,-0.766535,-4.895440,-3.943851,-4.520432,-8.000274,8.275367,-9.788313,-3.634355,6.018613,6.706386,6.918318,1.655891,-8.392562,3.684772,-0.696262,8.246132,-7.615656,-3.122748,7.795111,2.714428,-8.516218,-1.749480,-7.767666,-4.798050,-6.987690,-1.544167,9.243874,1.698061,-9.663083,3.765188,4.525393,-8.020292,3.092317,-3.908469,2.573144,-9.617525,-7.067674,-8.199606,3.480411,-7.578850,0.486144,-0.043204,-5.965005,-1.912456,-3.600916,-7.463278,-3.599232,4.657717,9.814901,-9.929360,-7.325456,2.427518,6.127221,5.653085,-9.718073,3.844950,4.165785,2.621851,0.221958,-1.160823,1.459727,-3.858691,7.565772,0.302626,9.325691,-4.975667,-1.456551,2.394794,6.724461,6.913536,-1.059232,7.184957], dtype = "float64")#candidate|1703|(140,)|const|float64
var_1704 = relay.var("var_1704", dtype = "float64", shape = (325, 1))#candidate|1704|(325, 1)|var|float64
call_1702 = relay.TupleGetItem(func_838_call(relay.reshape(const_1703.astype('float64'), [14, 10]), relay.reshape(var_1704.astype('float64'), [325,]), ), 2)
call_1705 = relay.TupleGetItem(func_842_call(relay.reshape(const_1703.astype('float64'), [14, 10]), relay.reshape(var_1704.astype('float64'), [325,]), ), 2)
output = relay.Tuple([bop_1697,call_1702,const_1703,var_1704,])
output2 = relay.Tuple([bop_1697,call_1705,const_1703,var_1704,])
func_1711 = relay.Function([var_1688,var_1689,var_1704,], output)
mod['func_1711'] = func_1711
mod = relay.transform.InferType()(mod)
var_1712 = relay.var("var_1712", dtype = "float64", shape = (7, 10, 12))#candidate|1712|(7, 10, 12)|var|float64
var_1713 = relay.var("var_1713", dtype = "float64", shape = (7, 10, 12))#candidate|1713|(7, 10, 12)|var|float64
var_1714 = relay.var("var_1714", dtype = "float64", shape = (325, 1))#candidate|1714|(325, 1)|var|float64
output = func_1711(var_1712,var_1713,var_1714,)
func_1715 = relay.Function([var_1712,var_1713,var_1714,], output)
mutated_mod['func_1715'] = func_1715
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1801 = relay.var("var_1801", dtype = "int16", shape = (1, 5, 3))#candidate|1801|(1, 5, 3)|var|int16
const_1802 = relay.const([[[6,-1,2],[-9,-9,-2],[4,7,-5],[3,-4,-2],[-4,-8,9]],[[-7,1,9],[-9,2,1],[4,-3,7],[4,7,-2],[1,-9,-3]],[[3,3,-9],[-9,-3,6],[-5,3,1],[-6,-9,3],[4,-9,2]],[[-1,-1,-1],[9,7,-1],[9,-5,-3],[-2,3,2],[3,9,8]],[[2,-8,-6],[-1,9,10],[-3,7,1],[6,1,-5],[5,-10,7]],[[-8,9,-2],[-5,1,-5],[5,2,-7],[3,6,-3],[6,8,2]]], dtype = "int16")#candidate|1802|(6, 5, 3)|const|int16
bop_1803 = relay.bitwise_and(var_1801.astype('int16'), const_1802.astype('int16')) # shape=(6, 5, 3)
output = relay.Tuple([bop_1803,])
output2 = relay.Tuple([bop_1803,])
func_1809 = relay.Function([var_1801,], output)
mod['func_1809'] = func_1809
mod = relay.transform.InferType()(mod)
var_1810 = relay.var("var_1810", dtype = "int16", shape = (1, 5, 3))#candidate|1810|(1, 5, 3)|var|int16
output = func_1809(var_1810)
func_1811 = relay.Function([var_1810], output)
mutated_mod['func_1811'] = func_1811
mutated_mod = relay.transform.InferType()(mutated_mod)
var_1988 = relay.var("var_1988", dtype = "float32", shape = (14, 9, 15))#candidate|1988|(14, 9, 15)|var|float32
uop_1989 = relay.erf(var_1988.astype('float32')) # shape=(14, 9, 15)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
var_1992 = relay.var("var_1992", dtype = "float64", shape = (325,))#candidate|1992|(325,)|var|float64
call_1991 = func_418_call(relay.reshape(var_1992.astype('float64'), [13, 5, 5]))
call_1993 = func_418_call(relay.reshape(var_1992.astype('float64'), [13, 5, 5]))
uop_2000 = relay.cos(uop_1989.astype('float32')) # shape=(14, 9, 15)
output = relay.Tuple([call_1991,var_1992,uop_2000,])
output2 = relay.Tuple([call_1993,var_1992,uop_2000,])
func_2018 = relay.Function([var_1988,var_1992,], output)
mod['func_2018'] = func_2018
mod = relay.transform.InferType()(mod)
var_2019 = relay.var("var_2019", dtype = "float32", shape = (14, 9, 15))#candidate|2019|(14, 9, 15)|var|float32
var_2020 = relay.var("var_2020", dtype = "float64", shape = (325,))#candidate|2020|(325,)|var|float64
output = func_2018(var_2019,var_2020,)
func_2021 = relay.Function([var_2019,var_2020,], output)
mutated_mod['func_2021'] = func_2021
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2121 = relay.var("var_2121", dtype = "float64", shape = (16, 16, 8))#candidate|2121|(16, 16, 8)|var|float64
uop_2122 = relay.sqrt(var_2121.astype('float64')) # shape=(16, 16, 8)
output = relay.Tuple([uop_2122,])
output2 = relay.Tuple([uop_2122,])
func_2133 = relay.Function([var_2121,], output)
mod['func_2133'] = func_2133
mod = relay.transform.InferType()(mod)
var_2134 = relay.var("var_2134", dtype = "float64", shape = (16, 16, 8))#candidate|2134|(16, 16, 8)|var|float64
output = func_2133(var_2134)
func_2135 = relay.Function([var_2134], output)
mutated_mod['func_2135'] = func_2135
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2177 = relay.var("var_2177", dtype = "float32", shape = (6, 15, 2))#candidate|2177|(6, 15, 2)|var|float32
uop_2178 = relay.cosh(var_2177.astype('float32')) # shape=(6, 15, 2)
uop_2180 = relay.sigmoid(uop_2178.astype('float32')) # shape=(6, 15, 2)
func_1711_call = mod.get_global_var('func_1711')
func_1715_call = mutated_mod.get_global_var('func_1715')
const_2183 = relay.const([-6.239410,-0.277120,2.726576,-1.650052,-7.666277,-0.738661,6.330553,-6.298097,0.147951,8.061570,-8.246950,0.986260,0.569992,9.299211,-3.468337,-8.404356,-1.790643,7.186874,4.385131,7.825600,6.540643,9.584563,3.771067,9.855555,-3.055527,1.211197,0.631697,-7.754909,-3.533669,-0.299180,-2.318675,0.879843,5.388958,-0.076837,4.433903,-5.846776,9.682270,1.700942,3.549155,8.360812,9.043330,-5.891457,7.170021,8.591293,8.676015,5.724160,-3.790720,-9.451865,5.473771,2.343763,1.523952,4.657526,8.987582,4.950010,4.128586,-8.469048,7.631353,-7.847729,-6.107491,5.450313,7.745441,2.103589,9.576694,3.558572,-3.008202,5.724628,-0.832202,-2.322830,-4.248891,4.316274,2.574563,-7.340659,-7.730933,5.810726,3.157625,-3.763575,6.205939,-8.997689,-1.318027,8.255664,-2.774630,-5.681573,7.443060,-6.244069,-6.408039,-2.384688,-9.750581,-1.477078,3.191001,1.881166,-5.147093,9.958562,0.400507,3.639685,9.168995,-0.040443,7.064453,-8.501878,8.800300,-2.431530,-4.755104,6.617597,-9.155059,-0.320226,-8.660570,-1.738866,3.729581,-7.561576,4.462730,-9.749757,0.029245,4.009361,5.733043,1.907405,-9.023712,-1.988341,-3.351461,7.111628,-3.787458,-8.750919,9.017609,9.262394,-3.293402,0.129612,6.244093,-2.329939,-5.438147,-7.506184,-9.404479,2.483431,4.093287,9.783613,3.789900,-5.397454,-0.172521,-3.692544,5.166698,7.041096,-5.744984,-1.875295,-4.306529,-9.992805,-7.843741,7.801932,-0.631312,1.827333,-3.058294,-3.325055,-2.759440,8.445252,6.066700,-2.453917,-4.594828,0.763802,5.381306,-1.864279,-4.508734,-6.592665,4.718788,7.847732,-6.579769,-3.405405,-1.244145,5.728603,1.282141,4.897390,-8.286032,-9.408849,4.830924,5.003861,8.886952,5.422241,-5.321460,-3.039405,4.733196,-2.908911,-8.174557,-3.655070,4.717150,7.870875,6.513825,1.447674,9.006031,-4.461024,3.931304,8.773832,0.811054,3.242065,-8.717982,-2.670610,2.452582,-4.336160,5.118747,-4.539188,-3.724076,6.134841,5.266284,4.535401,-2.211271,0.511976,-3.376363,-1.451930,4.544977,4.211921,4.813379,3.542748,-3.573757,5.091125,4.895349,-0.533514,7.452112,0.451834,3.589418,9.165564,-3.707057,8.301975,7.136885,-9.690761,3.682446,8.366976,9.167522,-9.729899,-9.952650,8.992840,4.873128,8.292858,-7.072874,1.503994,1.497434,8.902358,-3.741584,-3.914184,7.273867,-9.347101,0.026693,4.373099,4.786721,-6.146409,6.124477,5.865420,-0.133303,-8.312494,6.954142,-5.639143,-1.096098,-0.449054,-5.175576,8.332124,5.197696,5.863910,2.127215,-4.344984,-1.879254,-7.614331,-7.663582,-9.412050,-3.118996,6.279249,1.558762,-6.284033,-8.367884,-7.604289,-2.704608,-7.104516,3.988382,-2.685486,-5.243312,2.320754,7.315496,7.197825,9.639972,-2.470836,-1.745940,5.529166,7.722122,4.295506,-0.307492,8.172199,6.892277,1.366369,9.794228,7.477295,-0.414684,-1.277285,-8.127386,8.034542,8.545841,2.495393,-5.294028,6.128156,0.219787,8.669752,-3.632247,-5.613525,-5.902133,3.672554,2.962651,9.784967,3.126663,9.971285,6.517962,-6.761112,1.712505,-2.574764,-6.764199,7.593438,9.640558,-0.411532,4.870063,4.310604,-4.080411,8.532946,-1.811343,9.425950,-2.481242,7.329371,6.947876,4.312466,-8.728266,-3.050412,5.665293,-1.858400,-9.143745,7.097775,0.534430,5.456230,1.360665,3.521957,-2.987047,5.034399,3.532278,-9.783702,-7.489511,9.891552,-3.457545,-0.737653,-6.113122,-2.169984,0.678047,-5.695831,-3.435458,-0.586376,9.490989,3.264693,-3.425101,-4.790219,-3.673539,-1.928002,-3.883865,0.996499,-9.173738,2.588641,-6.707205,5.216969,0.250853,-0.558421,-0.541135,8.144548,-2.205710,9.464216,3.539678,-1.722532,-9.177657,5.595808,9.187458,4.068159,-6.326834,-8.913712,-6.059013,-8.651980,2.018898,-6.623697,3.991873,3.139347,-6.630539,-9.461444,7.718399,9.979885,4.754970,-7.224059,0.650608,-3.503840,-8.095039,-4.838351,8.932136,-3.787212,8.549767,-0.017665,-1.922279,-7.915962,-6.247524,3.820162,-1.532938,4.733516,6.853178,-7.819216,-3.902460,-2.624241,8.573767,1.801203,0.960510,6.435676,-6.067514,6.576173,-3.017375,-8.927049,-8.608671,-7.964672,-8.621410,-8.279536,1.840947,9.742273,-3.198162,1.577359,-6.149705,-5.746879,-5.498053,0.337953,1.268917,-5.582966,-1.044233,-1.084469,-8.503400,5.472913,-7.921770,-4.844638,-7.740130,5.213150,1.759847,-6.343970,-5.666181,1.615702,6.574330,4.113829,7.555883,-8.727174,-4.421280,-1.227983,8.422985,-5.804882,9.010690,5.186444,-0.298207,-4.811519,3.125039,3.030918,7.116226,8.601911,7.283129,1.143159,-2.172858,7.908536,-4.954539,0.727053,7.801891,-0.181126,-8.474622,8.820110,-3.043209,-6.277770,-4.796374,9.514338,-5.841459,-4.189939,-4.865043,-1.363468,0.548764,6.375563,8.154134,3.918650,-0.486416,6.062382,-8.407958,4.195588,7.934230,-1.000177,-2.297789,1.056089,-2.663436,-6.053055,-3.900724,8.585519,-7.844743,2.311676,8.327146,8.164948,-6.134305,-9.107666,7.218610,-7.986915,5.947754,-5.488796,-9.137967,-0.366998,-7.556154,6.122616,3.502114,-9.447200,-8.605115,0.140085,4.396989,1.815415,-1.554807,-7.389345,-2.285744,-4.747433,2.881726,3.864012,-7.143042,6.050743,-6.757271,2.746818,2.962276,-1.593372,-2.274147,-8.794564,-7.878966,5.163441,7.333694,-2.197820,-0.517332,-1.496855,1.670175,-3.175968,0.445526,-6.274406,-2.736061,-6.003478,-9.336858,1.896764,-9.708830,-5.508597,-5.433828,1.030337,9.948512,2.008256,-9.147250,2.950914,2.328135,-1.720950,7.812574,-8.171368,-9.835043,-0.607601,8.791910,0.024350,7.716343,-9.530845,-0.424994,8.754105,3.544436,1.426458,-8.029100,-1.723609,-5.486206,2.874518,9.005092,-4.187617,-4.062841,-8.388500,-8.022160,1.717485,2.658623,7.066652,-6.457788,5.522201,0.857321,6.969979,-5.009958,9.010637,-7.092047,-8.752945,7.921212,-1.030286,4.220061,4.652152,-5.397247,3.211463,5.084448,7.145725,-3.267769,4.488951,-0.812770,2.028372,-1.125966,6.050828,0.816315,6.253624,-6.782128,-1.180988,-6.107756,-9.568156,3.785657,4.463451,-5.656989,-9.140016,8.098735,2.864684,-1.372215,-4.157273,8.225263,-0.521946,-9.415449,0.409510,1.163995,-8.463190,-4.094207,-2.924107,0.910968,-7.030873,5.952753,-3.362095,-1.305554,-0.233808,-3.094123,-7.793659,-1.153050,4.713952,-0.640351,-5.098143,-8.073709,2.195300,5.263063,-8.186064,0.664669,-8.169678,2.775140,-0.628880,-9.857306,1.841136,2.657935,2.594429,-5.069873,-1.574245,9.361846,-4.486707,-7.023665,-9.398274,5.117766,-5.885278,0.918808,8.106160,7.186122,8.956469,-0.475854,-8.309765,-7.195497,-1.124019,-7.180293,3.651610,0.816903,-2.936847,1.655016,1.500114,1.121043,-9.466693,-4.854161,-5.523596,5.294563,-2.025154,-8.872413,6.587021,-9.862744,8.061816,5.695404,-2.801935,-4.886170,-1.973878,-9.138721,-7.114025,9.028380,-5.876096,0.036247,0.096102,-3.871132,-0.851239,-9.537284,7.555685,8.151236,-3.566292,4.854567,3.032109,4.858229,-9.928775,7.899682,9.110107,7.785535,8.930926,-3.895367,1.220045,-6.813286,5.862552,2.122254,4.692534,5.245576,2.586070,-9.790146,-5.627195,-3.076254,9.710247,-3.545846,-3.801025,8.667235,-9.402147,-0.358566,-2.660719,7.662499,-1.071184,4.195968,-7.125719,6.862189,0.956034,-3.030588,5.740520,-4.875270,7.449487,9.087211,-0.137581,8.652319,4.474019,-4.651054,-3.589740,8.594346,8.941116,-4.433426,-9.904300,5.887966,4.420646,5.391959,6.073337,-2.321659,-9.141185,-8.418807,-7.979768,6.632268,-5.897158,2.558153,-7.544868,0.744612,5.041243,5.532772,8.247763,-0.457051,-0.397969,7.904235,8.108199,-6.099622,5.986529,2.311113,-2.463281,5.615898,-5.529175,8.911474,-9.277955,-8.757463,-5.796556,8.913413,9.460566,0.756628,7.628141,7.119809,-3.602027,9.674280,-1.354005,8.767368,5.201104,8.313758,5.073883,-4.936519,3.321128,-9.620630,-0.175433,0.852354,4.614535,-1.538552,1.265634,8.586752,9.404671,-5.271036,0.367283,9.329890,-9.115444,1.308394,3.259897,5.588538,-4.349869,3.684727,9.434999,-3.566433,6.062308,-1.171585,-2.634267,-0.343130,-2.543055,-5.061323,-2.355677,-9.937973,-8.800902,-7.095974,-3.987714,5.454003,4.326341,7.613825,4.305866,6.129804,-3.913992,-6.301764,-4.499001,2.013746,7.275540,-8.532934,0.504706,-6.825426,-4.246315,4.545823,2.482764,-3.369704,1.155932,7.385381,7.048022,1.602403,-8.084606,7.505593,-5.291569,2.981492,-6.773829,7.364283,3.367755,5.670893,0.338860,-6.327311,2.870749,2.311213,-5.399402,-3.495424,-1.118826,-7.435730,9.433302,9.802591,-5.778361], dtype = "float64")#candidate|2183|(840,)|const|float64
var_2184 = relay.var("var_2184", dtype = "float64", shape = (325, 1))#candidate|2184|(325, 1)|var|float64
call_2182 = relay.TupleGetItem(func_1711_call(relay.reshape(const_2183.astype('float64'), [7, 10, 12]), relay.reshape(const_2183.astype('float64'), [7, 10, 12]), relay.reshape(var_2184.astype('float64'), [325, 1]), ), 1)
call_2185 = relay.TupleGetItem(func_1715_call(relay.reshape(const_2183.astype('float64'), [7, 10, 12]), relay.reshape(const_2183.astype('float64'), [7, 10, 12]), relay.reshape(var_2184.astype('float64'), [325, 1]), ), 1)
output = relay.Tuple([uop_2180,call_2182,const_2183,var_2184,])
output2 = relay.Tuple([uop_2180,call_2185,const_2183,var_2184,])
func_2186 = relay.Function([var_2177,var_2184,], output)
mod['func_2186'] = func_2186
mod = relay.transform.InferType()(mod)
var_2187 = relay.var("var_2187", dtype = "float32", shape = (6, 15, 2))#candidate|2187|(6, 15, 2)|var|float32
var_2188 = relay.var("var_2188", dtype = "float64", shape = (325, 1))#candidate|2188|(325, 1)|var|float64
output = func_2186(var_2187,var_2188,)
func_2189 = relay.Function([var_2187,var_2188,], output)
mutated_mod['func_2189'] = func_2189
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2444 = relay.const([[[9,-2,-4,-7,-2],[-7,7,9,-3,2],[-4,1,-8,-3,3],[2,-9,-10,5,-9],[-2,-9,6,-8,-3],[-2,-3,-3,3,4]]], dtype = "uint16")#candidate|2444|(1, 6, 5)|const|uint16
const_2445 = relay.const([[[-10,4,-7,-7,-6],[-3,-10,5,10,8],[4,-7,10,2,6],[4,4,-7,-7,-9],[3,-9,5,-3,-3],[8,-2,-6,-5,-10]],[[8,8,-2,-2,9],[-3,3,7,6,-2],[-1,-8,-3,7,3],[-8,9,-10,1,1],[3,-2,6,2,-9],[2,-5,3,5,6]],[[7,-10,2,-9,-3],[-2,-4,1,-4,-1],[9,3,4,-2,7],[-10,5,-1,9,1],[2,-8,-10,6,4],[-6,-1,-7,-8,7]]], dtype = "uint16")#candidate|2445|(3, 6, 5)|const|uint16
bop_2446 = relay.multiply(const_2444.astype('uint16'), const_2445.astype('uint16')) # shape=(3, 6, 5)
func_1498_call = mod.get_global_var('func_1498')
func_1501_call = mutated_mod.get_global_var('func_1501')
var_2465 = relay.var("var_2465", dtype = "int32", shape = (140,))#candidate|2465|(140,)|var|int32
call_2464 = func_1498_call(relay.reshape(var_2465.astype('int32'), [10, 14, 1]))
call_2466 = func_1498_call(relay.reshape(var_2465.astype('int32'), [10, 14, 1]))
output = relay.Tuple([bop_2446,call_2464,var_2465,])
output2 = relay.Tuple([bop_2446,call_2466,var_2465,])
func_2469 = relay.Function([var_2465,], output)
mod['func_2469'] = func_2469
mod = relay.transform.InferType()(mod)
var_2470 = relay.var("var_2470", dtype = "int32", shape = (140,))#candidate|2470|(140,)|var|int32
output = func_2469(var_2470)
func_2471 = relay.Function([var_2470], output)
mutated_mod['func_2471'] = func_2471
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2671 = relay.var("var_2671", dtype = "float64", shape = (12, 3, 6))#candidate|2671|(12, 3, 6)|var|float64
const_2672 = relay.const([[[3.438905,6.215181,9.495261,-4.999171,-3.052857,1.138556],[-1.371701,-9.210059,-8.582764,-5.316473,-5.966020,7.012478],[5.878411,-1.274033,7.108220,-3.326320,-0.567330,-1.439695]],[[-5.106893,-1.772766,-9.096735,-6.177951,-6.403881,9.600960],[-5.950744,5.029380,-7.334343,-2.587530,-8.361368,-7.984157],[9.970274,2.075869,-6.088417,7.412849,-7.438154,7.343607]],[[0.657826,5.076103,-5.142851,7.400392,4.430395,4.819889],[5.656060,6.471124,-8.959358,3.373577,-9.145836,-7.110232],[-2.552119,5.029736,3.025837,-6.498589,-7.005882,-4.826279]],[[-4.473154,-0.961075,-2.937803,3.713447,2.551579,-9.971804],[-4.540362,3.705969,-0.892009,7.389897,-0.858447,9.246756],[1.066688,-8.455193,9.215338,-6.203934,-6.856175,3.668343]],[[7.827050,-2.826551,2.041249,-1.027068,3.751738,5.538484],[2.335397,-5.204482,8.251986,-7.289213,0.433780,5.597112],[0.051426,-0.735965,-4.963017,3.862990,-5.867176,-6.298766]],[[-8.298864,9.208687,4.778548,7.792292,1.128439,-2.861556],[3.572834,4.983401,6.032268,8.912226,9.195163,9.165853],[-9.064009,8.822070,-3.897216,-9.394158,-7.661739,-0.539037]],[[-5.819023,0.948226,-5.223782,8.719684,-8.772093,-3.518745],[0.081989,-7.541703,-7.515894,-7.894039,-5.157612,7.700960],[4.292325,8.258442,-7.420988,9.764650,3.727280,-7.442153]],[[-9.496328,1.955283,4.707455,9.181681,-3.266053,5.028919],[0.879891,8.099070,4.557079,-7.670611,-6.568166,-5.476442],[-7.740539,-5.071320,-0.134640,1.111236,-7.445667,0.943610]],[[-4.839212,-1.618141,5.260284,1.012242,6.674440,-7.026244],[-4.680340,7.430763,0.221336,0.109551,-7.101642,2.100267],[-1.803740,6.477618,-2.148933,-4.892714,2.905847,6.263119]],[[-6.328290,7.735614,9.691270,-4.438260,8.197528,-0.256298],[-5.753948,-1.345295,7.944676,0.509653,-2.530359,-3.971269],[-3.195742,3.066604,7.167719,-5.783725,-1.432039,1.677889]],[[6.658672,6.208404,-7.029614,4.202227,2.540636,9.153347],[-7.031052,-8.274501,2.413248,9.124121,-9.867033,-0.392468],[2.902298,7.671146,-6.071760,5.629564,3.805900,-7.125192]],[[-4.174695,-8.549519,5.520741,8.415806,-9.148591,-9.161274],[-8.504930,9.937861,5.647713,9.139117,-8.668275,-9.561053],[3.153334,6.895687,-2.955219,-6.842008,2.181158,-5.457604]]], dtype = "float64")#candidate|2672|(12, 3, 6)|const|float64
bop_2673 = relay.maximum(var_2671.astype('float64'), relay.reshape(const_2672.astype('float64'), relay.shape_of(var_2671))) # shape=(12, 3, 6)
output = bop_2673
output2 = bop_2673
func_2676 = relay.Function([var_2671,], output)
mod['func_2676'] = func_2676
mod = relay.transform.InferType()(mod)
mutated_mod['func_2676'] = func_2676
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2677 = relay.var("var_2677", dtype = "float64", shape = (12, 3, 6))#candidate|2677|(12, 3, 6)|var|float64
func_2676_call = mutated_mod.get_global_var('func_2676')
call_2678 = func_2676_call(var_2677)
output = call_2678
func_2679 = relay.Function([var_2677], output)
mutated_mod['func_2679'] = func_2679
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2681 = relay.var("var_2681", dtype = "uint8", shape = (14, 1, 15))#candidate|2681|(14, 1, 15)|var|uint8
var_2682 = relay.var("var_2682", dtype = "uint8", shape = (14, 1, 15))#candidate|2682|(14, 1, 15)|var|uint8
bop_2683 = relay.equal(var_2681.astype('bool'), relay.reshape(var_2682.astype('bool'), relay.shape_of(var_2681))) # shape=(14, 1, 15)
output = relay.Tuple([bop_2683,])
output2 = relay.Tuple([bop_2683,])
func_2687 = relay.Function([var_2681,var_2682,], output)
mod['func_2687'] = func_2687
mod = relay.transform.InferType()(mod)
mutated_mod['func_2687'] = func_2687
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2687_call = mutated_mod.get_global_var('func_2687')
var_2689 = relay.var("var_2689", dtype = "uint8", shape = (14, 1, 15))#candidate|2689|(14, 1, 15)|var|uint8
var_2690 = relay.var("var_2690", dtype = "uint8", shape = (14, 1, 15))#candidate|2690|(14, 1, 15)|var|uint8
call_2688 = func_2687_call(var_2689,var_2690,)
output = call_2688
func_2691 = relay.Function([var_2689,var_2690,], output)
mutated_mod['func_2691'] = func_2691
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2756 = relay.var("var_2756", dtype = "int16", shape = (11, 15, 5))#candidate|2756|(11, 15, 5)|var|int16
var_2757 = relay.var("var_2757", dtype = "int16", shape = (11, 15, 5))#candidate|2757|(11, 15, 5)|var|int16
bop_2758 = relay.right_shift(var_2756.astype('int16'), relay.reshape(var_2757.astype('int16'), relay.shape_of(var_2756))) # shape=(11, 15, 5)
func_2186_call = mod.get_global_var('func_2186')
func_2189_call = mutated_mod.get_global_var('func_2189')
var_2764 = relay.var("var_2764", dtype = "float32", shape = (180, 1))#candidate|2764|(180, 1)|var|float32
var_2765 = relay.var("var_2765", dtype = "float64", shape = (13, 25))#candidate|2765|(13, 25)|var|float64
call_2763 = relay.TupleGetItem(func_2186_call(relay.reshape(var_2764.astype('float32'), [6, 15, 2]), relay.reshape(var_2765.astype('float64'), [325, 1]), ), 2)
call_2766 = relay.TupleGetItem(func_2189_call(relay.reshape(var_2764.astype('float32'), [6, 15, 2]), relay.reshape(var_2765.astype('float64'), [325, 1]), ), 2)
func_1345_call = mod.get_global_var('func_1345')
func_1347_call = mutated_mod.get_global_var('func_1347')
var_2775 = relay.var("var_2775", dtype = "bool", shape = (550,))#candidate|2775|(550,)|var|bool
call_2774 = relay.TupleGetItem(func_1345_call(relay.reshape(var_2775.astype('bool'), [5, 11, 10])), 0)
call_2776 = relay.TupleGetItem(func_1347_call(relay.reshape(var_2775.astype('bool'), [5, 11, 10])), 0)
output = relay.Tuple([bop_2758,call_2763,var_2764,var_2765,call_2774,var_2775,])
output2 = relay.Tuple([bop_2758,call_2766,var_2764,var_2765,call_2776,var_2775,])
func_2779 = relay.Function([var_2756,var_2757,var_2764,var_2765,var_2775,], output)
mod['func_2779'] = func_2779
mod = relay.transform.InferType()(mod)
mutated_mod['func_2779'] = func_2779
mutated_mod = relay.transform.InferType()(mutated_mod)
func_2779_call = mutated_mod.get_global_var('func_2779')
var_2781 = relay.var("var_2781", dtype = "int16", shape = (11, 15, 5))#candidate|2781|(11, 15, 5)|var|int16
var_2782 = relay.var("var_2782", dtype = "int16", shape = (11, 15, 5))#candidate|2782|(11, 15, 5)|var|int16
var_2783 = relay.var("var_2783", dtype = "float32", shape = (180, 1))#candidate|2783|(180, 1)|var|float32
var_2784 = relay.var("var_2784", dtype = "float64", shape = (13, 25))#candidate|2784|(13, 25)|var|float64
var_2785 = relay.var("var_2785", dtype = "bool", shape = (550,))#candidate|2785|(550,)|var|bool
call_2780 = func_2779_call(var_2781,var_2782,var_2783,var_2784,var_2785,)
output = call_2780
func_2786 = relay.Function([var_2781,var_2782,var_2783,var_2784,var_2785,], output)
mutated_mod['func_2786'] = func_2786
mutated_mod = relay.transform.InferType()(mutated_mod)
const_2947 = relay.const([[[8,-6,1],[7,9,-5],[4,-5,-2],[9,-1,-1],[-4,-1,1],[-3,-6,-3],[2,-6,8],[9,-9,-9],[1,-6,8],[-1,10,1],[1,-10,2],[5,5,-7],[5,5,8],[5,-5,-2],[-10,-5,-6],[-10,2,-6]],[[-5,-8,-4],[-3,1,-3],[5,8,8],[-3,4,9],[-6,-7,-2],[3,-5,10],[-8,-10,-3],[7,10,-8],[-7,9,8],[-1,8,5],[2,7,-10],[9,-10,-6],[2,10,1],[2,-8,-9],[-10,5,8],[-8,-7,-8]],[[-10,9,-5],[-2,10,-1],[-2,9,8],[8,-8,-6],[2,-5,-6],[10,-9,7],[-3,3,3],[-4,-8,4],[-1,-2,7],[5,1,-2],[2,-5,10],[7,1,9],[-3,-6,8],[-7,-4,7],[-8,-3,-2],[-5,7,-7]]], dtype = "uint8")#candidate|2947|(3, 16, 3)|const|uint8
var_2948 = relay.var("var_2948", dtype = "uint8", shape = (3, 16, 3))#candidate|2948|(3, 16, 3)|var|uint8
bop_2949 = relay.add(const_2947.astype('uint8'), relay.reshape(var_2948.astype('uint8'), relay.shape_of(const_2947))) # shape=(3, 16, 3)
output = bop_2949
output2 = bop_2949
func_2954 = relay.Function([var_2948,], output)
mod['func_2954'] = func_2954
mod = relay.transform.InferType()(mod)
var_2955 = relay.var("var_2955", dtype = "uint8", shape = (3, 16, 3))#candidate|2955|(3, 16, 3)|var|uint8
output = func_2954(var_2955)
func_2956 = relay.Function([var_2955], output)
mutated_mod['func_2956'] = func_2956
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2975 = relay.var("var_2975", dtype = "float32", shape = (15, 5, 1))#candidate|2975|(15, 5, 1)|var|float32
uop_2976 = relay.log2(var_2975.astype('float32')) # shape=(15, 5, 1)
output = uop_2976
output2 = uop_2976
func_2978 = relay.Function([var_2975,], output)
mod['func_2978'] = func_2978
mod = relay.transform.InferType()(mod)
var_2979 = relay.var("var_2979", dtype = "float32", shape = (15, 5, 1))#candidate|2979|(15, 5, 1)|var|float32
output = func_2978(var_2979)
func_2980 = relay.Function([var_2979], output)
mutated_mod['func_2980'] = func_2980
mutated_mod = relay.transform.InferType()(mutated_mod)
var_2989 = relay.var("var_2989", dtype = "float64", shape = (14, 13, 1))#candidate|2989|(14, 13, 1)|var|float64
uop_2990 = relay.atanh(var_2989.astype('float64')) # shape=(14, 13, 1)
bop_2997 = relay.floor_mod(var_2989.astype('float32'), relay.reshape(uop_2990.astype('float32'), relay.shape_of(var_2989))) # shape=(14, 13, 1)
func_2978_call = mod.get_global_var('func_2978')
func_2980_call = mutated_mod.get_global_var('func_2980')
const_3001 = relay.const([-2.526724,-9.136027,5.783387,-9.899177,2.863229,-3.023718,5.187316,-8.986275,-4.217509,7.750392,-3.381339,8.192408,-2.049042,-0.734467,-4.450647,-8.684868,9.906093,6.505358,0.168785,-1.945452,-6.496451,-3.933898,3.186539,0.459758,1.858762,-2.189108,4.654296,4.361965,-1.869073,-6.799877,2.121395,-2.778892,0.520481,7.715642,5.932997,0.371318,-6.745246,5.116477,-9.116368,4.771105,1.733444,-1.522026,-8.341432,-5.198325,7.936570,1.103874,1.351627,6.242847,-1.334805,9.761656,-6.297704,3.945600,-8.874525,9.116150,-3.929158,-0.784564,-3.898589,-6.423218,-6.795861,3.778393,1.275205,-7.017150,-9.490129,-4.115024,-4.214357,-9.589319,-6.769326,-6.329308,8.039516,-4.449098,-9.271786,7.142084,6.582509,6.448984,5.358662], dtype = "float32")#candidate|3001|(75,)|const|float32
call_3000 = func_2978_call(relay.reshape(const_3001.astype('float32'), [15, 5, 1]))
call_3002 = func_2978_call(relay.reshape(const_3001.astype('float32'), [15, 5, 1]))
bop_3007 = relay.logical_and(uop_2990.astype('bool'), relay.reshape(bop_2997.astype('bool'), relay.shape_of(uop_2990))) # shape=(14, 13, 1)
func_2018_call = mod.get_global_var('func_2018')
func_2021_call = mutated_mod.get_global_var('func_2021')
var_3015 = relay.var("var_3015", dtype = "float32", shape = (9, 210))#candidate|3015|(9, 210)|var|float32
const_3016 = relay.const([-2.343768,-0.500712,6.960020,8.556557,8.298145,-1.373059,-8.632611,9.668868,2.193613,6.626969,8.249645,-6.980020,5.585877,-0.897076,2.251026,-9.312646,4.729248,-6.143788,-4.620498,8.255831,1.580829,-3.457829,5.964497,1.540912,-8.150473,-2.306644,4.334080,9.909366,2.207758,3.470029,6.352538,6.796026,-0.590784,8.322081,3.577399,1.404840,-0.964228,9.752814,0.209645,-4.542374,6.982109,5.608804,5.410320,6.553368,-9.880432,0.927210,4.900450,-3.365466,-5.755085,-1.927505,3.209304,-1.631334,9.456756,1.985114,8.941245,2.905614,-6.759344,2.842424,8.596116,-3.359177,5.206418,1.374322,-1.648263,-8.734369,6.200940,4.365408,-8.064908,-9.440148,8.968158,-9.605736,4.549133,9.722689,2.725900,-2.230030,-4.915409,-9.761007,-5.250968,-3.349920,-0.878747,-0.890975,6.141577,-3.860337,4.431961,0.105942,-9.362533,4.819368,4.439914,3.165521,8.488967,-3.489320,-2.137995,7.096208,-1.080536,-2.891681,1.508254,2.091309,-1.617418,-3.845662,-7.586523,-4.169157,-2.320724,4.834224,-7.230957,-6.010295,-4.037766,2.992716,0.412684,6.930898,-3.038490,-8.113649,-7.672434,-8.280216,8.454892,6.373696,5.680327,-4.096464,1.313012,3.588349,-6.511496,-4.758786,-1.273669,-7.229122,-6.699317,-9.573757,4.496677,-4.366734,-7.974245,-4.133434,-2.089620,5.165962,6.379171,1.296104,8.905666,-9.958110,-1.850750,-8.748840,6.750871,-8.616138,1.847819,-8.464720,-2.571558,-1.872993,-1.194048,-1.627092,-2.589191,-7.969135,-1.040103,2.136411,-8.805612,3.456160,-3.618895,-1.556428,-9.618541,0.579453,4.824928,-7.968897,5.643681,-6.789187,1.684696,-9.561855,-6.686764,6.100538,-4.842325,0.705951,3.098594,7.666781,-7.111295,4.959852,-8.077921,-6.271151,-4.812127,5.579708,-3.256437,4.588910,2.952754,9.343396,-3.482193,-6.996263,3.661777,-3.786921,-1.640559,-5.812249,8.346690,-9.216576,5.071739,9.898923,4.580873,4.539043,-2.602053,-5.923621,-6.340388,6.433965,-0.619020,8.905132,3.804879,-4.662037,9.265988,0.281732,-8.436501,-1.493391,-6.156141,-8.809351,0.336408,7.061953,-6.312773,9.730970,-2.283739,-8.258027,-3.501509,-4.748381,8.583013,9.819643,-4.677759,2.864703,-2.201734,-4.161239,2.063436,8.873106,-2.208508,-3.668586,-7.863622,2.820112,-1.755498,6.682699,1.411791,3.732863,4.537363,2.905355,8.461812,-0.254125,-7.993865,-0.411435,7.247671,-4.430930,8.047209,2.323829,-2.176629,-4.229460,-9.402262,6.467643,3.424132,-2.394208,-4.205682,-9.285695,-9.242525,0.839167,5.725517,-4.620549,1.269586,-6.728827,2.982597,-0.031487,-9.303442,-0.191556,8.319045,-0.553492,-0.183322,1.276473,-8.951016,-3.033373,-8.893343,-6.753009,-8.486680,4.521099,5.471856,-8.595256,6.196584,3.834244,2.022590,-7.461523,5.095903,7.932447,4.838040,-8.396220,5.377862,-8.906571,4.796007,2.967660,0.091949,3.187352,1.445805,1.918221,2.113945,-1.218094,-4.785133,9.772490,-4.757688,-6.614818,1.724174,7.287838,6.167506,-6.102339,-0.078627,-4.630275,-7.908303,2.140395,2.494093,-9.889362,8.361785,0.230443,7.763068,7.947690,-3.780470,3.108676,7.755674,-9.115718,-1.794254,0.593958,-4.390702,-7.660973,4.576605,-4.671380,-6.765876,0.782683,-2.597773,-1.754751,8.280138,4.665002,3.922466,0.139830,4.367530,1.147096,1.299834,2.672905,0.754851], dtype = "float64")#candidate|3016|(325,)|const|float64
call_3014 = relay.TupleGetItem(func_2018_call(relay.reshape(var_3015.astype('float32'), [14, 9, 15]), relay.reshape(const_3016.astype('float64'), [325,]), ), 0)
call_3017 = relay.TupleGetItem(func_2021_call(relay.reshape(var_3015.astype('float32'), [14, 9, 15]), relay.reshape(const_3016.astype('float64'), [325,]), ), 0)
func_2687_call = mod.get_global_var('func_2687')
func_2691_call = mutated_mod.get_global_var('func_2691')
const_3027 = relay.const([[-8],[2],[-10],[-8],[6],[-4],[8],[1],[-8],[-10],[-4],[-6],[8],[8],[1],[-7],[-2],[9],[-9],[9],[-2],[8],[-6],[5],[1],[9],[-2],[4],[2],[7],[5],[-2],[-5],[1],[7],[-1],[-4],[-8],[9],[-4],[-1],[-6],[1],[4],[-4],[-7],[-7],[10],[-2],[-5],[-10],[10],[7],[-6],[-3],[7],[-7],[-8],[4],[2],[-1],[9],[-1],[-9],[9],[-4],[2],[4],[-3],[-6],[-3],[4],[-5],[-8],[2],[-7],[10],[2],[-6],[-4],[7],[-3],[-3],[-6],[5],[7],[-6],[-8],[3],[2],[-6],[10],[-6],[7],[5],[4],[-1],[-9],[-3],[-3],[1],[-8],[10],[10],[4],[9],[-2],[5],[-3],[-1],[-6],[-10],[-3],[3],[8],[-4],[-7],[7],[-8],[3],[10],[1],[-2],[1],[3],[-4],[8],[-5],[-7],[-7],[-6],[5],[3],[3],[9],[-5],[-7],[-3],[7],[6],[8],[8],[-6],[-9],[5],[7],[6],[5],[3],[5],[3],[3],[8],[-2],[-1],[-6],[8],[9],[1],[6],[-3],[-5],[6],[2],[-9],[3],[7],[-5],[3],[-10],[4],[7],[2],[-6],[5],[6],[7],[-9],[-3],[-4],[6],[-6],[7],[-7],[-9],[-7],[5],[-4],[4],[-9],[-2],[-7],[8],[1],[6],[6],[5],[8],[5],[2],[-4],[-6],[-7],[-5],[10],[9],[1],[-8],[3],[2]], dtype = "uint8")#candidate|3027|(210, 1)|const|uint8
call_3026 = relay.TupleGetItem(func_2687_call(relay.reshape(const_3027.astype('uint8'), [14, 1, 15]), relay.reshape(const_3027.astype('uint8'), [14, 1, 15]), ), 0)
call_3028 = relay.TupleGetItem(func_2691_call(relay.reshape(const_3027.astype('uint8'), [14, 1, 15]), relay.reshape(const_3027.astype('uint8'), [14, 1, 15]), ), 0)
func_2676_call = mod.get_global_var('func_2676')
func_2679_call = mutated_mod.get_global_var('func_2679')
const_3038 = relay.const([6.213894,-2.611973,-0.160455,0.863086,4.951608,7.148764,1.419117,2.189362,1.600837,7.071895,-9.342505,-9.935535,1.160932,3.055864,2.553024,1.181007,0.568730,-4.060882,-6.210787,5.852698,-0.840810,7.520668,-6.536627,9.368040,-2.347801,3.050370,-7.218451,-7.857150,-7.291134,-1.727324,8.729565,-3.364755,-8.965093,-6.314681,-9.248507,-3.346451,-0.293128,2.778603,-8.025169,9.613503,8.975206,-0.892864,-3.269829,-9.445450,-0.939042,-9.734004,-2.421772,3.413824,5.919091,3.667978,4.833795,-3.725306,4.558811,-1.844405,-2.512838,-6.467511,7.244763,-1.750270,-1.809786,-9.138968,1.011796,-7.487907,-4.261475,-8.165673,-8.915443,9.744424,8.914233,9.481233,4.499065,5.716053,2.766941,-9.347688,-7.355061,-7.908940,1.078792,8.376091,1.488796,-8.192950,1.687130,7.081410,-8.596855,-0.841037,6.771388,-9.697941,1.884765,-2.895433,9.047486,9.720436,5.867474,-1.432050,6.770882,-1.572573,-3.083539,-2.907901,3.248218,6.587583,-5.194380,-3.859777,2.059366,3.577065,-0.527442,-3.800122,-2.246810,1.464254,8.378923,-2.686623,1.011254,-5.420777,5.136759,-7.564031,-7.531624,-4.912808,1.750221,-4.083142,7.855651,5.784731,-9.730521,0.567711,-8.029253,-5.610556,6.707224,-1.901549,1.272686,-7.749284,2.164788,-3.533024,-5.048535,-7.157134,3.605481,-9.600155,-4.536710,-7.522795,0.897863,3.807201,-8.757983,-8.765939,-8.179463,-7.908129,7.283294,-7.316430,1.117945,-1.453605,8.255843,5.707455,-5.754960,8.345640,-0.489625,8.378847,-1.522462,5.687574,-7.231681,-9.835262,-9.946466,1.807480,5.457535,-4.438901,-4.244563,-6.500915,2.330417,-8.951441,4.607639,4.202058,3.865919,-9.509362,3.543612,-2.302058,7.474950,7.017657,6.533975,0.463380,-9.211088,-6.612340,-7.045035,1.420421,-8.980715,3.652419,8.330347,9.994354,7.759308,-2.611350,9.206765,7.459027,5.359578,-3.855679,6.577274,7.615496,6.243679,-3.801980,-4.457657,-1.669756,5.902558,2.065487,-4.632883,0.947865,-1.188457,-0.185833,-7.473185,-1.716537,-8.559784,-6.940342,-4.058991,-7.630462,3.088471,6.255272,-7.863476,4.828570,-8.382574,-9.993780,6.523689,-6.394081,-7.303470,-5.508991,-0.898728,-8.074773,9.273251,-6.025122], dtype = "float64")#candidate|3038|(216,)|const|float64
call_3037 = func_2676_call(relay.reshape(const_3038.astype('float64'), [12, 3, 6]))
call_3039 = func_2676_call(relay.reshape(const_3038.astype('float64'), [12, 3, 6]))
output = relay.Tuple([call_3000,const_3001,bop_3007,call_3014,var_3015,const_3016,call_3026,const_3027,call_3037,const_3038,])
output2 = relay.Tuple([call_3002,const_3001,bop_3007,call_3017,var_3015,const_3016,call_3028,const_3027,call_3039,const_3038,])
func_3042 = relay.Function([var_2989,var_3015,], output)
mod['func_3042'] = func_3042
mod = relay.transform.InferType()(mod)
mutated_mod['func_3042'] = func_3042
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3042_call = mutated_mod.get_global_var('func_3042')
var_3044 = relay.var("var_3044", dtype = "float64", shape = (14, 13, 1))#candidate|3044|(14, 13, 1)|var|float64
var_3045 = relay.var("var_3045", dtype = "float32", shape = (9, 210))#candidate|3045|(9, 210)|var|float32
call_3043 = func_3042_call(var_3044,var_3045,)
output = call_3043
func_3046 = relay.Function([var_3044,var_3045,], output)
mutated_mod['func_3046'] = func_3046
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3072 = relay.var("var_3072", dtype = "uint32", shape = (14, 14))#candidate|3072|(14, 14)|var|uint32
const_3073 = relay.const([[-3,-10,-1,-2,-7,-9,10,-1,-10,-4,6,8,-10,4],[6,-1,4,-8,8,6,-4,-2,4,9,-8,7,-10,8],[7,-7,-3,-6,4,2,5,-9,4,7,5,-1,9,-6],[2,9,7,6,6,5,-5,2,-4,3,4,-5,6,-1],[4,3,5,-7,10,10,-5,-8,3,-10,-2,-6,-3,-5],[-7,-3,6,-10,2,1,1,9,-8,6,-4,-8,-1,3],[5,2,-6,4,-3,-3,2,-6,-7,10,-10,9,-7,4],[-3,-2,7,-10,-1,5,1,-8,3,-9,-8,10,-9,6],[5,-4,7,7,-1,8,-5,7,-2,10,-10,-10,10,-10],[6,-3,-6,-1,2,-8,9,7,9,-10,1,-8,9,-1],[5,-7,9,-6,-6,9,-9,3,3,9,-5,4,-9,9],[10,6,-3,-10,3,-7,-5,9,5,1,3,1,9,4],[-10,5,-4,9,3,8,-5,3,3,-5,5,8,6,-5],[-7,-4,-3,10,1,3,-9,8,-9,-6,-4,-1,-7,-6]], dtype = "uint32")#candidate|3073|(14, 14)|const|uint32
bop_3074 = relay.greater_equal(var_3072.astype('bool'), relay.reshape(const_3073.astype('bool'), relay.shape_of(var_3072))) # shape=(14, 14)
func_851_call = mod.get_global_var('func_851')
func_855_call = mutated_mod.get_global_var('func_855')
const_3079 = relay.const([1.675271,0.411716,-4.246195,9.761568,9.002723,8.537095,-4.178205,5.783128,-3.020475,9.654821,-4.818656,1.721050,-3.778157,-0.232062,5.143363,4.755274,-4.370747,-3.102409,-8.571120,2.590049,-8.815155,-3.304665,2.865694,4.077720,7.716154,-0.374484,-3.968229,-6.497915,-1.607074,-0.809832,-3.007055,-8.927680,-7.145808,6.645843,-7.485108,-9.216174,-5.708033,-5.508686,-3.392523,4.640859,-3.595153,-0.890869,1.023450,-4.215761,9.074412,9.841596,-4.210292,-5.222360,-2.317262,-1.193851,-6.378853,5.719720,8.689335,-2.078149,-2.743020,2.118447,6.947848,9.882534,-2.221437,-9.728293,-9.414548,7.689136,2.803858,7.098348,-2.735467,-4.741024,-5.279374,2.895313,8.378274,2.976188,1.485112,4.314389,-1.879946,5.676120,-5.822340,-5.042509,0.846873,4.624450,6.772151,-8.567108,-9.225702,-6.694742,9.432760,-1.860268,5.276285,-6.046041,-0.657497,0.160207,-8.794625,-5.698077,-4.017604,8.183039,-0.958612,-8.483805,2.967548,8.988602,-7.894664,-9.058666,2.972801,-6.620316,-5.465986,8.778175,-2.884801,-4.102451,7.168454,2.179866,0.750060,7.864600,-3.139978,-4.083401,3.569759,8.767904,9.001329,9.981766,2.050943,-1.732519,-6.005109,-1.362572,-6.494210,2.556448,8.252142,-6.325908,6.155960,8.482183,6.317308,8.490666,4.491330,-0.786308,-0.726699,-6.642092,-4.805977,4.863348,-8.483277,3.104712,-1.115988,9.864733,-2.548600,8.643267,4.781523,1.185931,-0.455133,7.068087,8.884062,-7.957267,4.499686,3.743345,4.866220,-0.887969,0.128158,9.640882,-6.044133,9.090661,-7.711785,-7.462832,0.406466,-0.445418,-0.715169,-6.524982,-1.197066,7.402690,7.283828,1.059449,-8.445368,-9.974826,2.517666,7.066385,-6.331368,-1.035059,3.811072,7.898356,-5.826757,-7.967934,6.671244,9.321789,-4.158350,-7.151466,7.113933,-2.964469,6.497071,1.108079,-2.758165,6.497851,-1.188526,2.060749,-3.159799,-4.128068,1.043298,-5.025505,2.266355,-7.848017,-8.835830,9.154748,-3.042018,9.265973,7.230862,-7.976109,-6.165478,-1.972744,-2.546089,-8.380027,-2.579779,2.492788,-4.459699,8.030514,7.898753,0.207687,-2.371925,-2.069886,-7.990503,-7.630964,-2.931238,-0.847735,-1.493095,9.670405,-5.517746,-0.064126,3.787665,-6.736502,4.626900,6.165080,2.336018,-0.479393,-2.159564,-4.969254,3.140677,-4.017612,-6.965018,-3.685751,7.804943,9.117572,7.698314,7.656378,3.790901,-3.058313,8.508662,-6.603365,-7.773908,-4.253556,7.493311,-5.904733,2.186137,-2.880891,2.932874,2.635141,6.268735,-1.108297,-8.315278,1.303362,-8.994957,-3.932573,9.551356,-4.681083,4.111904,-5.420467,-4.435202,9.280909,2.379097,0.076910,-4.840483,-8.854632,0.491761,0.243060,-5.599091,3.764862,-9.855049,-4.436061,-0.414809,-0.676131,1.742194,-9.865639,1.684621,5.896596,-0.723955,1.773587,4.644638,-8.326700,-5.238060,-9.247154,-1.936562,1.946475,6.197303,-1.136536,5.794093,-4.624925,-5.427754,-3.245773,-2.477893,-1.705850,-4.568878,-3.898103,3.916334,-8.169675,-8.794734,-7.857128,-5.853382,9.779781,1.159243,-7.355125,-9.343920,2.133584,0.129879,-7.340061,-7.632632,-5.792255,-3.137720,-5.982672,6.942537,-0.325376,-1.754377,-1.365772,-1.738851,-8.331583,-5.620465,-4.987261,9.939563,0.103589,9.653454,7.661212,-3.106884,-0.937564,-0.771495,3.959239,5.599948,9.278238,-7.682851,5.250832,7.610723,7.794841,-6.435801,6.057052], dtype = "float32")#candidate|3079|(330,)|const|float32
const_3080 = relay.const([[-3.539584,-8.467696,-0.146342,-0.532299,-5.803171,0.544731,3.493775,-9.482683,-8.398961,-0.240037,9.430138,-9.803246,6.943426,3.440028,-7.473688,5.802665,-0.187542,-0.305473,-6.828282,-8.330628,-5.764760,-8.363241,-8.539348,-9.352015,-8.816077],[-9.242189,1.478355,-1.545243,-2.358014,2.058121,-3.511807,-5.411834,-2.141792,8.860681,9.692577,5.209514,2.904959,4.365965,-1.252635,-9.103758,-9.367597,-1.243452,0.701211,-4.383039,-4.866938,1.748099,-2.085322,-3.671176,-0.392652,5.028612],[3.234904,0.551648,-6.528401,-1.295543,3.177423,-0.912747,1.107448,0.148479,2.914117,9.933746,6.604219,-5.485060,-6.596247,-0.493213,-9.196435,-6.896811,1.373695,-5.748525,-1.272277,-3.086019,-9.817640,8.982857,-4.666658,-1.117558,1.381609],[2.226089,-7.422320,-4.215593,-6.741811,4.254954,7.361699,1.835368,-0.610706,-4.246789,-8.717046,-5.898340,2.936055,1.573100,-1.629590,-8.818137,5.687765,1.608760,6.650312,-1.917496,-3.437684,2.736429,0.926465,-1.718431,-9.719213,-6.964779],[-3.942750,4.362902,-5.000705,5.024910,7.946515,-6.274690,9.914368,-1.801766,-8.331111,1.873287,4.632664,-0.927452,1.870496,-2.357457,6.008736,-0.036993,2.432645,-4.372603,-3.448709,-3.502986,7.621290,-3.710801,-7.469510,8.551400,9.303799],[-0.916169,5.979622,9.865703,3.021173,4.040072,-1.152349,-8.682222,6.529689,-9.992117,-6.036936,1.150578,4.291139,5.902396,8.630193,-9.923094,2.341555,8.140904,8.178445,2.803760,-1.630819,-7.421593,-0.772461,-7.746046,5.940116,-8.778302],[2.236705,6.287404,4.147238,0.260670,0.276474,-7.190482,-0.062328,-1.306226,-0.823667,-4.475725,-5.447414,-3.099698,-3.987802,-4.094879,2.620169,6.144513,1.321018,5.270337,6.142937,0.973953,8.036256,0.674311,7.717741,2.720812,0.867201],[-5.555050,-7.158988,0.631209,3.621563,-6.174195,-9.663050,6.854363,-6.207699,-0.777302,9.050905,4.412394,7.007545,-7.617051,-0.410639,5.745078,-5.185776,2.309691,-1.502037,9.392923,-8.443536,-1.808626,1.239226,-6.005144,-6.788445,-5.673704],[-0.972852,-5.015623,-8.545903,-5.277677,4.532765,4.579372,3.575274,-6.235211,3.918638,0.141158,-1.292085,-3.261302,-5.919512,6.679201,6.747008,-6.192607,5.763582,-5.524975,-8.905736,-5.130028,-5.871855,9.770567,-8.317555,6.623871,7.829295],[-2.529050,-6.790827,7.031201,6.151154,2.768134,-3.969492,4.046556,0.667160,-4.593552,-9.444765,9.170129,-9.293628,8.504629,-4.635986,-1.422172,4.920490,7.649185,2.571844,-1.511075,-3.899670,1.694811,-8.136694,-0.222193,-5.728535,8.136931],[-2.244167,7.711470,5.427082,3.459117,7.081648,2.756356,-9.124299,-4.085178,2.882525,0.976948,1.585305,2.701372,3.799345,2.644175,-9.039931,-7.305636,8.414518,5.387216,-9.334347,-8.514103,-6.872389,0.630467,9.339984,-6.832837,-5.192459],[-2.771244,-8.250114,2.191149,3.134354,-3.595034,7.138223,-4.138124,-4.217578,7.017898,6.823501,2.012931,2.463463,-1.168212,-2.270912,-1.476212,-4.463853,-1.074093,-8.245027,-3.402864,-5.459476,5.134612,-1.197870,-6.232861,2.369522,6.217574],[-4.684688,-0.605739,5.718385,-3.393317,6.621720,6.891214,5.434432,-7.828099,4.740290,8.682439,7.938161,-1.259941,6.477802,8.533972,-3.881566,8.614514,7.855247,4.656600,-7.532039,-6.177695,-0.188082,-5.328584,-8.733703,9.752707,4.325651]], dtype = "float64")#candidate|3080|(13, 25)|const|float64
call_3078 = relay.TupleGetItem(func_851_call(relay.reshape(const_3079.astype('float32'), [10, 3, 11]), relay.reshape(const_3080.astype('float64'), [325,]), ), 0)
call_3081 = relay.TupleGetItem(func_855_call(relay.reshape(const_3079.astype('float32'), [10, 3, 11]), relay.reshape(const_3080.astype('float64'), [325,]), ), 0)
output = relay.Tuple([bop_3074,call_3078,const_3079,const_3080,])
output2 = relay.Tuple([bop_3074,call_3081,const_3079,const_3080,])
func_3102 = relay.Function([var_3072,], output)
mod['func_3102'] = func_3102
mod = relay.transform.InferType()(mod)
mutated_mod['func_3102'] = func_3102
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3103 = relay.var("var_3103", dtype = "uint32", shape = (14, 14))#candidate|3103|(14, 14)|var|uint32
func_3102_call = mutated_mod.get_global_var('func_3102')
call_3104 = func_3102_call(var_3103)
output = call_3104
func_3105 = relay.Function([var_3103], output)
mutated_mod['func_3105'] = func_3105
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3263 = relay.var("var_3263", dtype = "uint16", shape = (14, 9, 10))#candidate|3263|(14, 9, 10)|var|uint16
var_3264 = relay.var("var_3264", dtype = "uint16", shape = (14, 9, 10))#candidate|3264|(14, 9, 10)|var|uint16
bop_3265 = relay.equal(var_3263.astype('bool'), relay.reshape(var_3264.astype('bool'), relay.shape_of(var_3263))) # shape=(14, 9, 10)
output = relay.Tuple([bop_3265,])
output2 = relay.Tuple([bop_3265,])
func_3273 = relay.Function([var_3263,var_3264,], output)
mod['func_3273'] = func_3273
mod = relay.transform.InferType()(mod)
mutated_mod['func_3273'] = func_3273
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3273_call = mutated_mod.get_global_var('func_3273')
var_3275 = relay.var("var_3275", dtype = "uint16", shape = (14, 9, 10))#candidate|3275|(14, 9, 10)|var|uint16
var_3276 = relay.var("var_3276", dtype = "uint16", shape = (14, 9, 10))#candidate|3276|(14, 9, 10)|var|uint16
call_3274 = func_3273_call(var_3275,var_3276,)
output = call_3274
func_3277 = relay.Function([var_3275,var_3276,], output)
mutated_mod['func_3277'] = func_3277
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3413 = relay.var("var_3413", dtype = "float32", shape = (1, 6, 1))#candidate|3413|(1, 6, 1)|var|float32
uop_3414 = relay.rsqrt(var_3413.astype('float32')) # shape=(1, 6, 1)
bop_3416 = relay.floor_divide(uop_3414.astype('float64'), relay.reshape(var_3413.astype('float64'), relay.shape_of(uop_3414))) # shape=(1, 6, 1)
bop_3419 = relay.add(var_3413.astype('float32'), relay.reshape(uop_3414.astype('float32'), relay.shape_of(var_3413))) # shape=(1, 6, 1)
output = relay.Tuple([bop_3416,bop_3419,])
output2 = relay.Tuple([bop_3416,bop_3419,])
func_3422 = relay.Function([var_3413,], output)
mod['func_3422'] = func_3422
mod = relay.transform.InferType()(mod)
mutated_mod['func_3422'] = func_3422
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3423 = relay.var("var_3423", dtype = "float32", shape = (1, 6, 1))#candidate|3423|(1, 6, 1)|var|float32
func_3422_call = mutated_mod.get_global_var('func_3422')
call_3424 = func_3422_call(var_3423)
output = call_3424
func_3425 = relay.Function([var_3423], output)
mutated_mod['func_3425'] = func_3425
mutated_mod = relay.transform.InferType()(mutated_mod)
const_3697 = relay.const([[[-8.634027,-6.791419],[-1.438821,2.216556],[3.663159,-5.250292],[8.541711,-4.348596],[2.766384,8.330903],[-8.312362,-1.383266],[-7.285390,-7.357379],[2.949270,-4.655898],[4.267926,8.026066],[0.231452,-2.981952],[8.647659,-4.449281],[-3.676155,-9.497653],[-1.773772,-5.255019],[3.387797,6.906952]],[[5.040002,-7.020327],[2.403799,-5.698512],[4.294761,-9.798190],[-5.796541,5.260633],[-3.971472,-9.309036],[1.240996,-8.770637],[4.679752,-8.731637],[1.641426,-1.922496],[8.297792,6.063808],[0.885114,6.984464],[4.382148,1.427760],[-5.457385,5.472139],[8.613103,-4.624285],[-3.031367,-6.255740]],[[9.748762,2.216643],[1.548840,6.945059],[-5.841091,3.627585],[0.165630,7.540110],[3.394064,-0.053117],[1.540523,-1.061729],[-8.036029,0.334539],[8.497317,-9.318688],[-4.070499,6.887906],[4.456950,6.394762],[5.103849,6.370646],[9.971399,5.693242],[7.996947,0.641872],[5.154453,0.856337]],[[-7.693289,8.244634],[4.477605,-3.082982],[5.812080,3.358745],[-6.220940,-3.164694],[0.300242,-1.177974],[-4.715872,8.811679],[-9.049527,0.663666],[-6.893474,9.348744],[4.791428,-9.019283],[-1.635897,6.249107],[-5.531443,-1.293350],[9.303029,-1.570730],[-0.351011,4.481564],[-9.779194,-3.191233]],[[4.005897,0.909568],[3.383618,-3.200353],[1.901925,4.885161],[5.991873,0.170736],[6.430640,4.848853],[-6.992195,2.491013],[-7.992038,-1.281174],[-7.226053,1.340550],[-2.254470,3.988006],[5.092677,-1.621981],[1.608510,-1.053212],[-5.344403,-4.928035],[5.904018,-6.650780],[-6.509423,-1.181524]],[[2.274982,0.535506],[-1.048838,0.245939],[1.390133,-6.155498],[7.780663,1.846390],[-5.012672,0.647459],[4.129208,-5.929920],[7.308181,1.667359],[-2.076958,9.326869],[-0.815570,9.750979],[5.004312,2.245331],[-5.118861,-9.526258],[-8.316493,-0.239449],[2.172694,-3.255653],[7.420068,-3.735077]],[[1.808954,-3.211152],[6.019202,4.716506],[1.778544,-7.550942],[-5.128970,0.228777],[2.926193,-4.380483],[-9.229081,-3.203830],[-2.794689,-8.736974],[5.902625,2.640894],[-1.650685,1.761731],[7.379309,4.904770],[5.515695,9.517398],[4.912136,-3.212862],[-0.474654,2.148206],[2.875004,5.241956]],[[-4.234165,-7.237795],[-6.128428,-8.580572],[1.260094,-8.787902],[1.311561,3.214452],[-9.730735,-0.710863],[-1.187628,-5.645878],[8.173235,7.937312],[-0.717340,-5.900393],[-2.783833,-7.277706],[-4.480152,-4.892388],[-4.397751,-2.435584],[-8.879212,6.510705],[6.291347,3.191828],[-6.685748,-6.120168]],[[9.641540,9.291788],[-9.068950,-5.464821],[-6.865978,3.631377],[5.227928,-5.622237],[5.501088,0.569170],[5.786026,-4.107566],[-6.403680,-2.289633],[2.790777,-6.014537],[-4.953764,-5.301959],[-7.426393,4.798008],[-9.302722,4.752378],[-8.345732,3.655188],[2.811410,-7.790114],[1.258573,6.256253]],[[-5.843105,4.361392],[-5.474647,-0.153025],[7.860157,-8.863886],[8.061402,-0.161157],[-0.330767,0.652150],[8.401544,-9.951718],[-7.972226,5.751468],[1.042156,4.427401],[-3.489139,-0.138562],[6.338015,6.378281],[2.383323,-9.487513],[-5.093833,-0.832462],[-0.328020,0.028392],[1.817474,-7.628579]],[[-6.129522,1.231300],[-5.907057,2.774898],[0.553041,-0.875972],[-8.757723,9.049311],[1.070904,-2.426563],[-3.143491,-0.048536],[5.908057,4.985581],[-3.422378,-0.103379],[6.285547,-4.253329],[-6.496176,0.235347],[7.535406,-6.861198],[3.672915,-3.072722],[3.725917,-8.438327],[0.079001,4.057761]],[[3.351427,9.070458],[-4.652207,-3.396544],[-9.437950,-1.526969],[7.854472,-2.491159],[4.899215,-5.294248],[4.683500,3.207288],[-0.458654,-3.631281],[-5.528210,-6.332108],[-9.434712,2.302772],[8.167505,3.433338],[2.582214,-9.498146],[8.621341,4.067840],[-7.140036,5.040760],[-2.415438,-6.333315]],[[3.500764,0.398574],[-2.119385,1.550416],[5.689445,-0.510592],[-6.210182,-0.493338],[6.619683,-9.312423],[-9.523244,4.306372],[-1.300997,-2.215860],[9.535392,3.386770],[9.585570,-5.297515],[-0.392712,9.981032],[4.051539,-2.769016],[5.855966,8.067097],[7.326302,3.301476],[8.756424,-1.728444]],[[7.739670,4.122824],[-9.259483,1.989521],[4.286518,5.629401],[1.371736,0.457225],[-6.266501,8.701654],[7.366820,-2.553961],[-3.574209,-2.561117],[-2.503259,2.066116],[9.014876,-8.265705],[3.727700,6.243483],[-7.845961,-3.452399],[2.207200,-5.270709],[3.109952,4.848365],[-6.619614,-1.540905]]], dtype = "float32")#candidate|3697|(14, 14, 2)|const|float32
uop_3698 = relay.asinh(const_3697.astype('float32')) # shape=(14, 14, 2)
output = uop_3698
output2 = uop_3698
func_3706 = relay.Function([], output)
mod['func_3706'] = func_3706
mod = relay.transform.InferType()(mod)
mutated_mod['func_3706'] = func_3706
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mutated_mod.get_global_var('func_3706')
call_3707 = func_3706_call()
output = call_3707
func_3708 = relay.Function([], output)
mutated_mod['func_3708'] = func_3708
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_3742 = func_3706_call()
call_3743 = func_3706_call()
var_3745 = relay.var("var_3745", dtype = "float32", shape = (14, 14, 2))#candidate|3745|(14, 14, 2)|var|float32
bop_3746 = relay.logical_and(call_3742.astype('bool'), relay.reshape(var_3745.astype('bool'), relay.shape_of(call_3742))) # shape=(14, 14, 2)
bop_3749 = relay.logical_and(call_3743.astype('bool'), relay.reshape(var_3745.astype('bool'), relay.shape_of(call_3743))) # shape=(14, 14, 2)
func_2133_call = mod.get_global_var('func_2133')
func_2135_call = mutated_mod.get_global_var('func_2135')
var_3757 = relay.var("var_3757", dtype = "float64", shape = (128, 16))#candidate|3757|(128, 16)|var|float64
call_3756 = relay.TupleGetItem(func_2133_call(relay.reshape(var_3757.astype('float64'), [16, 16, 8])), 0)
call_3758 = relay.TupleGetItem(func_2135_call(relay.reshape(var_3757.astype('float64'), [16, 16, 8])), 0)
func_1711_call = mod.get_global_var('func_1711')
func_1715_call = mutated_mod.get_global_var('func_1715')
const_3760 = relay.const([-0.067721,6.132604,-7.572824,-3.355177,-7.881867,-7.847613,5.052085,-6.291352,8.194347,-7.392862,0.507752,-8.813729,8.551299,-6.563944,3.005147,-5.859032,-3.413346,-0.391883,9.227204,-2.387244,-4.261634,-9.559644,-5.726913,8.247610,0.522318,5.249322,-8.341938,-8.666770,7.118144,-8.810936,-5.743858,-6.767485,-5.886983,7.114040,-0.556223,6.032651,-3.516212,-7.481624,-8.093044,-1.790450,6.144047,2.860106,-0.110069,-5.619538,-4.916327,-5.213460,1.543191,-3.387837,8.585761,-2.807817,5.656923,1.907988,1.723291,-2.746524,2.737679,-8.259676,5.936711,-2.666885,3.790084,8.470986,-2.809622,-3.221717,3.149678,6.369496,4.222362,-7.990333,-4.837903,-1.169931,7.319992,5.609246,-6.967482,-0.091065,5.017152,5.720963,0.424489,6.325366,-8.329687,0.249580,-4.331080,-6.570937,-8.183098,-2.810342,-7.518108,0.051777,-9.907920,4.745516,-6.954349,-5.218176,-1.753559,-3.037107,-2.980367,9.834954,7.737883,-8.811856,-1.404971,1.338907,9.426459,6.337551,9.849083,-0.603994,6.565376,-4.590362,7.701642,-6.863666,-6.280746,6.316762,-7.864925,9.262638,9.310140,9.606108,-3.597096,9.174859,-1.068357,-6.214139,0.974300,0.603411,-8.445895,-2.454681,-5.497512,9.103399,3.349108,-9.998849,-2.770855,5.022642,3.756843,3.790264,-2.304763,-5.558393,9.591591,-1.536635,4.292696,0.948518,2.709275,7.020949,-9.431783,-1.922406,6.697221,2.291718,-8.582011,8.329377,2.274642,6.517600,7.976524,1.681013,-7.608035,5.778466,7.310605,6.271153,-5.712975,4.674328,-3.982999,-7.551076,-5.099081,0.751720,0.747360,-8.268145,3.339769,-3.313032,3.334151,6.538934,6.484561,7.868176,1.358614,0.649603,5.050668,8.464867,-1.669747,5.487910,7.924784,5.189225,-5.916738,-7.548398,0.264966,-5.157941,-3.727521,4.733401,4.654541,-5.611117,9.402862,-8.745274,-4.386676,-3.825881,-3.176387,8.206799,-5.763063,-2.886461,4.873676,-1.164826,3.605342,5.490470,-3.775739,9.810487,-6.225789,7.638238,2.404941,-8.208351,9.997295,-8.695189,9.821489,7.291401,-0.326379,-1.355516,-1.536959,-4.899330,9.345931,1.006730,-2.403618,4.617928,3.563739,3.671412,6.292090,-6.256627,-6.466234,-0.157928,9.697133,-1.193280,5.295733,1.995277,9.555760,6.865496,2.429085,2.446781,7.331930,-3.963206,1.666280,4.895263,-7.384917,9.901537,-4.562330,-1.136407,3.772620,8.920809,5.705067,0.899424,-0.736379,3.004534,0.182442,-4.583082,-0.773426,8.479213,-0.408893,-7.203308,5.127263,1.522420,6.760523,6.972269,-0.070802,8.737894,2.854039,6.459580,2.270665,3.989782,5.252557,5.035280,1.511497,6.197524,1.400810,-5.482502,1.033503,-1.278244,9.412323,8.590995,-7.551705,9.646879,6.745080,3.253215,4.824118,-3.845415,-5.996502,-9.364171,-9.583855,2.822054,0.908818,-7.570288,-1.037587,5.627576,-7.490673,6.273472,6.559411,-4.766307,3.753149,8.801888,9.160085,5.287927,1.994560,7.845063,2.370647,0.797739,-7.923129,8.333834,-1.490117,-8.044031,-0.312518,8.905787,-5.607810,-3.356234,-9.917352,-9.593998,7.290262,6.207432,-8.883026,9.114209,-9.241117,-8.098699,-6.441188,1.975179,9.045471,-9.421161,3.810374,1.742373,9.267484,-7.124632,-3.619241,1.788275,-7.329309,-6.584089,5.631826,6.356628,8.345230,-1.537937,5.391276,-7.185683,9.192922,-6.388003,3.921118,-8.462495,-8.963968,-8.064416,-2.230051,-4.251383,-0.957069,8.580003,-8.315665,2.432141,-7.417179,-9.931799,0.676557,-6.952177,4.196818,-2.971613,2.143065,8.723746,0.144214,-9.879949,-8.489376,-2.004683,1.887906,-8.790884,-2.155548,7.374755,-9.144075,7.386199,3.832771,-7.263949,-4.186311,5.027283,-2.491620,7.708935,-1.266645,3.652274,1.889034,6.916352,0.408108,-8.672957,-6.026827,-3.633998,3.427673,-0.795530,-4.067243,4.123857,3.123634,-0.388124,-5.197144,4.164636,9.105560,0.397425,3.032701,2.343512,-5.951144,-0.865771,-8.343647,1.544338,-8.515057,-7.245338,3.632381,-0.852860,-0.605731,-6.491755,-8.988688,1.381193,-5.932404,-7.889445,6.165954,-0.323051,-4.759745,-4.320110,2.802119,-1.604099,-4.145984,-5.412666,1.749579,-6.989742,7.224853,-6.849394,0.619028,6.280789,-8.448803,4.623713,-8.725861,9.429755,-7.543440,6.436399,-6.508023,-6.764192,-7.459053,-4.404854,2.646006,4.498891,0.595570,-6.224744,5.139807,-3.160665,-5.601477,4.921200,5.024003,3.054235,3.375877,6.533937,-7.295934,-9.439875,7.261006,-5.395117,6.013598,6.652388,7.601407,5.778340,-0.096336,-5.229082,5.602784,1.479141,0.048917,1.763444,6.617593,-2.472180,-2.818616,-3.980388,4.553709,-4.018279,-9.899256,9.270128,1.047243,-7.850461,-7.100198,-1.261865,-5.917984,-2.471773,-4.115346,-1.222072,-1.367141,-1.202492,8.205237,-4.871102,5.796880,-6.684708,8.847924,-6.912100,-9.902695,-9.200409,8.560745,1.021065,-1.473442,7.459837,6.121365,-2.852304,-2.805371,-2.022558,7.672697,-9.941441,1.325069,-7.425450,-9.705755,8.130340,3.560072,6.941574,0.564160,-8.855082,-5.310263,-5.479813,-3.447616,-7.582957,3.238951,-4.736855,6.115213,-6.371374,6.994390,3.208925,7.562838,2.897362,-6.578140,6.340797,9.481444,8.280773,-0.113227,6.311435,-9.390216,8.713070,-6.905967,-5.691534,5.244915,-7.758192,5.595794,-1.442175,-7.063121,9.989899,9.616059,-3.534986,1.037400,-7.694891,-7.803045,-5.973266,1.280290,-4.776542,7.622530,-5.586143,1.171045,-3.359059,-9.251953,2.830980,-4.112497,-8.333119,-2.966909,-8.720167,-4.997602,9.649685,-0.653457,-4.974026,0.442293,2.480264,7.011396,0.850116,-4.213669,-2.381476,7.283108,2.900150,-1.437854,4.592822,6.126666,-9.275454,-7.262819,-8.487168,4.027073,4.668623,4.732705,-4.049217,-8.685041,-4.639671,4.478686,7.870774,-2.182121,-8.948510,-2.225863,2.640248,1.385055,-9.546484,1.670954,-9.379872,7.850067,-6.812240,-5.840362,3.367418,-9.199547,5.586435,-1.162045,-0.514628,6.477352,-3.617570,-6.553438,1.469800,6.383376,-7.996524,-7.153291,-8.686279,-8.153382,7.071857,4.016533,8.705462,-2.666221,0.503005,9.277184,1.913252,-0.212568,2.476751,-1.701821,-5.101357,-3.278882,-3.589021,9.677202,-1.575710,5.329392,-4.873941,-5.944662,1.504267,-8.918176,-0.246332,-1.979849,-4.441577,7.150725,-6.227389,9.135732,7.466637,4.230713,-0.742003,-9.527815,-8.520210,-8.419001,-0.482442,-7.841116,5.603483,-7.140040,-5.423556,-3.718948,4.426148,-9.494022,9.418482,-6.701216,5.642282,-3.763083,-4.456430,-8.336652,9.848417,0.539733,-9.698406,9.775121,-4.672711,9.619536,-2.239238,-1.277598,3.077839,5.647750,-7.620532,2.390505,5.605338,9.262458,9.403871,-5.785874,1.155523,-9.967399,6.418357,4.547893,-3.431678,0.585280,-8.468921,-9.814033,-2.421718,-3.763738,-2.932762,6.708422,-3.340849,5.686375,-2.796433,-2.272401,-9.597260,1.450562,-9.822809,6.709124,9.624714,5.938721,-6.487032,3.923658,-6.344021,1.666547,-7.085425,-9.337837,-6.749539,-0.180280,-0.632208,-2.456250,5.464329,-1.958250,-2.100207,-0.364685,-4.548910,-2.233545,-2.434590,-5.162527,9.788404,6.217530,-6.685602,-7.612125,9.055906,3.424762,4.241820,-9.279607,-5.755180,-6.100210,7.069347,-8.308109,7.740213,-7.487346,4.565449,-6.971947,9.150576,-8.551067,6.489254,-8.022831,1.042047,-3.498871,2.941504,7.855637,-3.732345,-4.612798,0.957910,7.075564,7.593438,-8.380061,-0.054492,-3.949053,-7.953134,5.700524,5.760944,-4.725802,-2.564775,-5.098340,-3.413825,7.256746,1.121242,-9.063642,-9.781675,2.993496,4.088416,-3.278684,-9.213012,7.792133,3.464885,9.771680,-0.322861,0.213502,-4.772521,-4.495260,-3.561977,8.859025,-9.826089,-6.566390,-9.960848,-5.023936,9.669753,0.465988,0.063252,-1.298269,-2.561571,-7.043765,2.536671,3.259890,7.461991,-3.193140,9.819249,3.359004,6.058983,-2.146667,-1.485070,-5.810490,-3.530912,2.494285,-1.953886,-0.730469,9.117395,-6.887669,0.060275,7.548569,2.340669,-1.678960,4.934460,-1.757849,3.032498,-2.666948,-5.396517,3.703024,2.610311,-1.435725,-0.915646,-3.481074,4.609839,-7.032772,-7.261003,2.398994,-7.298814,-5.298815,-1.176758,-1.599673,7.092648,9.663164,2.509083,-8.945004,1.715089,2.528233,-0.908182,-8.284096,-4.937098,-3.214511,-5.171941,-2.585136,9.568749,-7.883964,-9.867851,4.395069,-6.362477,3.248333,5.510307,-7.752801,-5.598028,-6.542937,1.282793,9.089413,9.072359,-1.138188,3.472631,-9.678720,9.566359,-3.250474,-6.004435,3.705878,-1.626142,-4.651932,2.330118,6.361386,-2.368129,-7.829186,-4.659257,3.456446,0.478540,4.972337,5.835198,4.861892,-8.238505,-1.972288,0.849828,7.072819,-4.914000,8.774856,-7.172789], dtype = "float64")#candidate|3760|(840,)|const|float64
var_3761 = relay.var("var_3761", dtype = "float64", shape = (325,))#candidate|3761|(325,)|var|float64
call_3759 = relay.TupleGetItem(func_1711_call(relay.reshape(const_3760.astype('float64'), [7, 10, 12]), relay.reshape(const_3760.astype('float64'), [7, 10, 12]), relay.reshape(var_3761.astype('float64'), [325, 1]), ), 2)
call_3762 = relay.TupleGetItem(func_1715_call(relay.reshape(const_3760.astype('float64'), [7, 10, 12]), relay.reshape(const_3760.astype('float64'), [7, 10, 12]), relay.reshape(var_3761.astype('float64'), [325, 1]), ), 2)
output = relay.Tuple([bop_3746,call_3756,var_3757,call_3759,const_3760,var_3761,])
output2 = relay.Tuple([bop_3749,call_3758,var_3757,call_3762,const_3760,var_3761,])
func_3767 = relay.Function([var_3745,var_3757,var_3761,], output)
mod['func_3767'] = func_3767
mod = relay.transform.InferType()(mod)
mutated_mod['func_3767'] = func_3767
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3767_call = mutated_mod.get_global_var('func_3767')
var_3769 = relay.var("var_3769", dtype = "float32", shape = (14, 14, 2))#candidate|3769|(14, 14, 2)|var|float32
var_3770 = relay.var("var_3770", dtype = "float64", shape = (128, 16))#candidate|3770|(128, 16)|var|float64
var_3771 = relay.var("var_3771", dtype = "float64", shape = (325,))#candidate|3771|(325,)|var|float64
call_3768 = func_3767_call(var_3769,var_3770,var_3771,)
output = call_3768
func_3772 = relay.Function([var_3769,var_3770,var_3771,], output)
mutated_mod['func_3772'] = func_3772
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_3779 = func_3706_call()
call_3780 = func_3706_call()
output = call_3779
output2 = call_3780
func_3786 = relay.Function([], output)
mod['func_3786'] = func_3786
mod = relay.transform.InferType()(mod)
output = func_3786()
func_3787 = relay.Function([], output)
mutated_mod['func_3787'] = func_3787
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_3791 = func_3786_call()
call_3792 = func_3786_call()
output = relay.Tuple([call_3791,])
output2 = relay.Tuple([call_3792,])
func_3796 = relay.Function([], output)
mod['func_3796'] = func_3796
mod = relay.transform.InferType()(mod)
mutated_mod['func_3796'] = func_3796
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3796_call = mutated_mod.get_global_var('func_3796')
call_3797 = func_3796_call()
output = call_3797
func_3798 = relay.Function([], output)
mutated_mod['func_3798'] = func_3798
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_3799 = func_3706_call()
call_3800 = func_3706_call()
uop_3834 = relay.cosh(call_3799.astype('float32')) # shape=(14, 14, 2)
uop_3836 = relay.cosh(call_3800.astype('float32')) # shape=(14, 14, 2)
func_3422_call = mod.get_global_var('func_3422')
func_3425_call = mutated_mod.get_global_var('func_3425')
const_3848 = relay.const([-5.860554,-0.467907,-9.029097,-3.707638,9.493589,-9.178893], dtype = "float32")#candidate|3848|(6,)|const|float32
call_3847 = relay.TupleGetItem(func_3422_call(relay.reshape(const_3848.astype('float32'), [1, 6, 1])), 0)
call_3849 = relay.TupleGetItem(func_3425_call(relay.reshape(const_3848.astype('float32'), [1, 6, 1])), 0)
output = relay.Tuple([uop_3834,call_3847,const_3848,])
output2 = relay.Tuple([uop_3836,call_3849,const_3848,])
func_3866 = relay.Function([], output)
mod['func_3866'] = func_3866
mod = relay.transform.InferType()(mod)
output = func_3866()
func_3867 = relay.Function([], output)
mutated_mod['func_3867'] = func_3867
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_3868 = func_3786_call()
call_3869 = func_3786_call()
output = call_3868
output2 = call_3869
func_3870 = relay.Function([], output)
mod['func_3870'] = func_3870
mod = relay.transform.InferType()(mod)
mutated_mod['func_3870'] = func_3870
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3870_call = mutated_mod.get_global_var('func_3870')
call_3871 = func_3870_call()
output = call_3871
func_3872 = relay.Function([], output)
mutated_mod['func_3872'] = func_3872
mutated_mod = relay.transform.InferType()(mutated_mod)
var_3946 = relay.var("var_3946", dtype = "float64", shape = (7, 15, 6))#candidate|3946|(7, 15, 6)|var|float64
uop_3947 = relay.atan(var_3946.astype('float64')) # shape=(7, 15, 6)
bop_3952 = relay.subtract(uop_3947.astype('int8'), relay.reshape(var_3946.astype('int8'), relay.shape_of(uop_3947))) # shape=(7, 15, 6)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_3958 = func_3706_call()
call_3959 = func_3706_call()
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_3962 = func_3786_call()
call_3963 = func_3786_call()
output = relay.Tuple([bop_3952,call_3958,call_3962,])
output2 = relay.Tuple([bop_3952,call_3959,call_3963,])
func_3967 = relay.Function([var_3946,], output)
mod['func_3967'] = func_3967
mod = relay.transform.InferType()(mod)
var_3968 = relay.var("var_3968", dtype = "float64", shape = (7, 15, 6))#candidate|3968|(7, 15, 6)|var|float64
output = func_3967(var_3968)
func_3969 = relay.Function([var_3968], output)
mutated_mod['func_3969'] = func_3969
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_3978 = func_3706_call()
call_3979 = func_3706_call()
func_2018_call = mod.get_global_var('func_2018')
func_2021_call = mutated_mod.get_global_var('func_2021')
var_4000 = relay.var("var_4000", dtype = "float32", shape = (1890,))#candidate|4000|(1890,)|var|float32
const_4001 = relay.const([2.684237,4.409085,6.000452,-6.357953,1.545882,-1.231655,-7.410952,4.823147,0.893270,8.183729,-7.719593,-5.069432,-5.345034,7.789039,-3.668218,3.679843,-4.115461,-0.605220,-1.983579,8.847171,2.350781,5.160513,8.603067,9.678251,5.160128,-0.880669,6.040337,3.423993,4.320599,7.130359,5.029920,9.192621,0.874523,3.132556,-4.957463,-8.384210,-6.659200,-6.211107,-5.277794,-2.915946,-1.695809,0.886060,3.237691,3.517341,5.489085,0.121549,9.968433,-8.828832,-2.779024,-2.727460,-4.299050,-8.103526,-6.596079,9.398428,-4.781923,4.806891,-5.613799,-7.556201,7.065183,-5.404380,9.679675,7.287885,-7.935032,7.491485,0.472428,-7.842428,-1.923235,-7.189123,2.248346,3.306194,-7.451804,-6.008514,-8.099201,-7.437949,-4.924170,-0.598567,-2.752656,7.351049,-8.109280,9.784798,-3.499411,2.664566,-0.574127,-1.919413,3.638098,7.271857,-9.510523,-9.003884,2.328103,-2.875240,7.755226,0.527082,-1.405540,1.846350,1.250473,-1.929121,-6.730591,6.416098,6.398132,-5.201347,6.256234,-4.090105,2.982733,-5.716595,-5.387302,-0.035514,-4.051035,1.191475,-5.179786,9.645225,-7.187034,-1.190304,7.658767,8.821539,7.500896,-2.542266,-6.309673,-2.178440,2.202281,-4.779504,0.510877,-5.058601,-2.530512,-2.475146,-3.913579,-6.094170,5.126765,5.401123,-2.344350,6.629225,-9.097030,0.817748,5.312804,-8.810725,-4.525772,-7.951081,8.283132,7.617031,-6.693618,3.497824,7.559734,8.785235,-0.900089,-6.262806,-4.858603,-7.804493,1.405717,3.156521,5.860920,0.317754,-4.463515,1.356332,-6.597224,0.226687,-9.511102,8.136258,-1.496885,-6.675915,6.228613,-9.654259,-5.229139,-8.598640,-8.500000,-7.395428,7.413957,2.752947,5.520536,-5.951021,-9.681907,5.638066,-0.754427,-9.097532,8.764610,9.178297,3.969109,-5.780967,-7.829534,0.907455,-3.294497,-2.072984,-9.085782,2.356381,-4.141464,2.575421,3.965599,1.393737,-8.802432,4.069991,-6.012917,-4.299781,-2.001659,7.211520,-8.107990,3.210615,1.448874,-4.673397,0.198972,8.586753,2.517125,-6.473673,6.659513,-0.861094,-0.271558,-2.855546,-9.803478,-3.199958,2.773059,2.604972,3.310678,8.252305,-3.958388,-8.909341,2.742498,8.890255,-8.470532,-1.491847,2.568486,3.938466,-5.123186,-8.085151,3.904684,0.899342,8.626117,-0.143501,1.951189,9.230910,4.622142,9.646411,2.060628,-0.122798,4.208556,-4.397525,4.409223,-0.051427,9.875932,3.890922,8.546640,5.291151,3.271588,4.118458,2.454680,5.143602,-0.947188,-7.203264,6.203191,-0.331781,3.671631,-9.331766,-8.328217,-4.820517,5.608731,-6.080515,-4.589655,6.088906,5.878447,-2.182086,3.481476,-7.744582,-0.135020,-7.450001,0.792487,-8.255967,2.074410,2.505115,2.627145,9.968027,7.041555,-4.586841,2.114156,-2.627322,-5.171647,-4.947097,-6.529602,-5.757851,0.293027,-0.948654,5.991037,8.490570,2.147783,-7.804830,-8.377951,-3.547065,-3.271639,2.702371,-7.769777,-5.971667,-1.907444,1.726420,6.464472,4.602509,-5.653290,2.597106,0.196269,-0.587078,-5.297184,2.705120,-1.544179,-3.819997,-3.977681,-4.266907,8.032828,-4.701138,2.960198,-8.426374,2.541926,4.055087,-1.478643,3.622452,6.259524,-8.079632,-2.362122,-0.707503,7.237860,0.377406,-5.921877,0.126456,4.658260,2.087052,2.598246,5.267585,-1.615539,-8.506008,-5.046202,-5.118188,3.151855], dtype = "float64")#candidate|4001|(325,)|const|float64
call_3999 = relay.TupleGetItem(func_2018_call(relay.reshape(var_4000.astype('float32'), [14, 9, 15]), relay.reshape(const_4001.astype('float64'), [325,]), ), 0)
call_4002 = relay.TupleGetItem(func_2021_call(relay.reshape(var_4000.astype('float32'), [14, 9, 15]), relay.reshape(const_4001.astype('float64'), [325,]), ), 0)
uop_4019 = relay.exp(call_3978.astype('float64')) # shape=(14, 14, 2)
uop_4021 = relay.exp(call_3979.astype('float64')) # shape=(14, 14, 2)
func_571_call = mod.get_global_var('func_571')
func_575_call = mutated_mod.get_global_var('func_575')
var_4034 = relay.var("var_4034", dtype = "float32", shape = (576,))#candidate|4034|(576,)|var|float32
call_4033 = relay.TupleGetItem(func_571_call(relay.reshape(var_4034.astype('float32'), [16, 6, 6]), relay.reshape(const_4001.astype('float64'), [325,]), ), 2)
call_4035 = relay.TupleGetItem(func_575_call(relay.reshape(var_4034.astype('float32'), [16, 6, 6]), relay.reshape(const_4001.astype('float64'), [325,]), ), 2)
var_4037 = relay.var("var_4037", dtype = "float64", shape = (13, 5, 5))#candidate|4037|(13, 5, 5)|var|float64
bop_4038 = relay.greater_equal(call_3999.astype('bool'), relay.reshape(var_4037.astype('bool'), relay.shape_of(call_3999))) # shape=(13, 5, 5)
bop_4041 = relay.greater_equal(call_4002.astype('bool'), relay.reshape(var_4037.astype('bool'), relay.shape_of(call_4002))) # shape=(13, 5, 5)
output = relay.Tuple([var_4000,const_4001,uop_4019,call_4033,var_4034,bop_4038,])
output2 = relay.Tuple([var_4000,const_4001,uop_4021,call_4035,var_4034,bop_4041,])
func_4050 = relay.Function([var_4000,var_4034,var_4037,], output)
mod['func_4050'] = func_4050
mod = relay.transform.InferType()(mod)
mutated_mod['func_4050'] = func_4050
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4050_call = mutated_mod.get_global_var('func_4050')
var_4052 = relay.var("var_4052", dtype = "float32", shape = (1890,))#candidate|4052|(1890,)|var|float32
var_4053 = relay.var("var_4053", dtype = "float32", shape = (576,))#candidate|4053|(576,)|var|float32
var_4054 = relay.var("var_4054", dtype = "float64", shape = (13, 5, 5))#candidate|4054|(13, 5, 5)|var|float64
call_4051 = func_4050_call(var_4052,var_4053,var_4054,)
output = call_4051
func_4055 = relay.Function([var_4052,var_4053,var_4054,], output)
mutated_mod['func_4055'] = func_4055
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_4108 = func_3706_call()
call_4109 = func_3706_call()
output = relay.Tuple([call_4108,])
output2 = relay.Tuple([call_4109,])
func_4118 = relay.Function([], output)
mod['func_4118'] = func_4118
mod = relay.transform.InferType()(mod)
output = func_4118()
func_4119 = relay.Function([], output)
mutated_mod['func_4119'] = func_4119
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4118_call = mod.get_global_var('func_4118')
func_4119_call = mutated_mod.get_global_var('func_4119')
call_4177 = relay.TupleGetItem(func_4118_call(), 0)
call_4178 = relay.TupleGetItem(func_4119_call(), 0)
output = relay.Tuple([call_4177,])
output2 = relay.Tuple([call_4178,])
func_4186 = relay.Function([], output)
mod['func_4186'] = func_4186
mod = relay.transform.InferType()(mod)
mutated_mod['func_4186'] = func_4186
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4186_call = mutated_mod.get_global_var('func_4186')
call_4187 = func_4186_call()
output = call_4187
func_4188 = relay.Function([], output)
mutated_mod['func_4188'] = func_4188
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4189 = relay.var("var_4189", dtype = "float64", shape = (15, 5, 8))#candidate|4189|(15, 5, 8)|var|float64
uop_4190 = relay.sigmoid(var_4189.astype('float64')) # shape=(15, 5, 8)
uop_4198 = relay.sqrt(var_4189.astype('float32')) # shape=(15, 5, 8)
bop_4200 = relay.equal(var_4189.astype('bool'), relay.reshape(uop_4190.astype('bool'), relay.shape_of(var_4189))) # shape=(15, 5, 8)
output = relay.Tuple([uop_4198,bop_4200,])
output2 = relay.Tuple([uop_4198,bop_4200,])
func_4207 = relay.Function([var_4189,], output)
mod['func_4207'] = func_4207
mod = relay.transform.InferType()(mod)
var_4208 = relay.var("var_4208", dtype = "float64", shape = (15, 5, 8))#candidate|4208|(15, 5, 8)|var|float64
output = func_4207(var_4208)
func_4209 = relay.Function([var_4208], output)
mutated_mod['func_4209'] = func_4209
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_4218 = func_3786_call()
call_4219 = func_3786_call()
output = call_4218
output2 = call_4219
func_4230 = relay.Function([], output)
mod['func_4230'] = func_4230
mod = relay.transform.InferType()(mod)
mutated_mod['func_4230'] = func_4230
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4230_call = mutated_mod.get_global_var('func_4230')
call_4231 = func_4230_call()
output = call_4231
func_4232 = relay.Function([], output)
mutated_mod['func_4232'] = func_4232
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3796_call = mod.get_global_var('func_3796')
func_3798_call = mutated_mod.get_global_var('func_3798')
call_4254 = relay.TupleGetItem(func_3796_call(), 0)
call_4255 = relay.TupleGetItem(func_3798_call(), 0)
output = call_4254
output2 = call_4255
func_4265 = relay.Function([], output)
mod['func_4265'] = func_4265
mod = relay.transform.InferType()(mod)
mutated_mod['func_4265'] = func_4265
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4265_call = mutated_mod.get_global_var('func_4265')
call_4266 = func_4265_call()
output = call_4266
func_4267 = relay.Function([], output)
mutated_mod['func_4267'] = func_4267
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_4291 = func_3706_call()
call_4292 = func_3706_call()
var_4296 = relay.var("var_4296", dtype = "float32", shape = (14, 14, 2))#candidate|4296|(14, 14, 2)|var|float32
bop_4297 = relay.floor_divide(call_4291.astype('float32'), relay.reshape(var_4296.astype('float32'), relay.shape_of(call_4291))) # shape=(14, 14, 2)
bop_4300 = relay.floor_divide(call_4292.astype('float32'), relay.reshape(var_4296.astype('float32'), relay.shape_of(call_4292))) # shape=(14, 14, 2)
func_3422_call = mod.get_global_var('func_3422')
func_3425_call = mutated_mod.get_global_var('func_3425')
var_4313 = relay.var("var_4313", dtype = "float32", shape = (6,))#candidate|4313|(6,)|var|float32
call_4312 = relay.TupleGetItem(func_3422_call(relay.reshape(var_4313.astype('float32'), [1, 6, 1])), 1)
call_4314 = relay.TupleGetItem(func_3425_call(relay.reshape(var_4313.astype('float32'), [1, 6, 1])), 1)
uop_4319 = relay.log10(var_4296.astype('float64')) # shape=(14, 14, 2)
func_2018_call = mod.get_global_var('func_2018')
func_2021_call = mutated_mod.get_global_var('func_2021')
const_4332 = relay.const([[-2.830171,4.635632,0.618460,-3.293499,8.521070,-4.507728,-7.891693,4.818329,-1.081187,-2.924936,-8.706587,2.279790,-0.094097,8.417425,3.541604,-4.988098,5.246131,-7.473614,5.454506,-4.808680,-2.284045,-4.104063,1.208739,4.678115,1.678384,4.611899,7.134608,1.719494,7.137431,1.633600,-8.389301,-5.432519,1.527624,-8.812986,-5.618022,-7.650461,2.633966,9.157254,-6.717398,8.983672,-9.250965,0.306661,5.240063,8.332316,9.570547,6.555059,7.114180,6.305501,-8.339069,8.900370,-0.969844,-0.003693,7.299893,2.449955,-9.142625,5.886824,0.794237,4.686701,9.992016,5.246602,3.722731,2.213868,-7.273635,0.269807,-0.462613,3.247018,-4.052151,0.641367,0.116140,6.243225,5.449639,7.616825,7.041484,1.904691,1.550660,-0.804212,5.521800,0.594561,-2.515461,8.153863,-0.730738,-7.633723,-1.185239,-4.144319,-7.473311,9.524467,-9.722042,-5.565970,5.506280,2.898709,-7.094539,3.986503,3.779682,1.864364,-8.754878,-2.674222,-6.078748,3.854396,-8.858108,-4.393593,-2.060160,-0.172096,-5.512512,-8.277695,-7.617250,6.329759,1.321846,8.102693,5.510461,9.503069,2.511177,-8.031319,-8.163422,-4.176973,3.646194,3.604105,-8.366255,4.941106,-6.865576,2.847003,8.846036,7.209241,2.795893,3.422880,8.949816,-1.137213,-2.722403,-7.372191,5.589508,-9.870734,-6.307893,-3.263725,0.549889,8.992412,3.143567,7.733140,0.331289,-3.902947,9.593358,-5.402307,-5.277642,1.634748,2.774602,0.733032,-5.076411,-5.656144,5.914450,2.137378,-9.843363,-0.354934,5.157354,-9.539734,-0.537466,-7.151439,-6.777939,-6.982205,9.386382,6.060892,-7.039948,9.789088,-5.960521,7.396953,-0.566181,4.036755,8.820456,8.507811,5.351077,-5.751411,-9.299850,6.716894,3.453945,9.510255,3.686092,8.666793,5.160427,-4.589292,-1.852284,6.865695,-0.100407,6.932367,-3.665868,6.427441,-3.013641,2.061012,-2.092188,1.053064,-4.638366,-6.725654,-0.223560,4.987744,1.820898,5.237872,9.915002,-8.309934,3.603765,-3.250965,-2.399817,9.285326,1.415105,1.119334,6.444714,0.326852,3.393158,0.971033,1.038193,2.370301,-2.860722,4.821014,3.002160,-5.286893,3.252409,-3.957969,-3.707493,0.198716,-4.519903,6.549158,4.245595,-5.933062,-6.048832,-8.184297,-2.058464,-9.540619,6.066174,7.986693,-6.832592,7.341411,-8.535370,9.018167,-7.022910,-9.269501,-2.497722,-8.698026,-4.896799,-8.195251,1.203317,-0.842789,9.552133,-4.779738,-8.420743,7.604523,0.799134,3.887407,0.269058,-4.117238,9.059190,-0.796188,-0.542858,4.826621,3.855914,-8.742610,0.113805,6.650996,0.643830,3.542542,4.156717,-0.033955,1.050878,-4.768233,5.639717,-9.975905,4.981768,-7.876179,-5.901071,-4.722277,5.253767,-8.331816,6.995383,6.254919,6.658023,1.501211,-5.994921,-3.881425,7.437619,-7.213906,5.827241,8.855270,7.805037,0.352938,0.942467,-3.994127,-0.717812,8.836661,-6.401015,-9.986427,0.428583,-9.916642,-4.630795,-9.193381,9.304634,-0.369647,6.461883,4.402730,2.005180,-4.216767,-8.017178,0.018130,3.101033,-8.555738,-0.416685,-8.824179,-3.003109,8.646290,7.957950,-2.171691,-9.778219,0.467965,-7.982751,-3.564124,2.562556,4.312989,7.172638,-8.981870,5.658923,-1.161033,-1.676225,-5.966515,8.440322,5.141256,3.436260,5.008051,-5.542239,9.644700,3.115110,-0.983155,6.303688,-8.535431,-5.420245,4.756891,-7.303639,6.469306,-2.790966,1.763879,7.517354,-5.555700,8.950484,-0.427847,3.067923,-0.974886,-0.538359,4.010496,-6.723938,-6.520691,9.678119,7.271227,-7.678408,-5.357806,-4.842328,-9.371609,-7.980791,-9.152398,7.756146,2.459555,-1.718971,4.562042,-2.028676,-7.319214,4.370668,8.184797,2.408481,-6.292040,-4.450942,-5.538429,-9.215794,-4.688419,-0.223701,-7.438278,-4.529797,-1.248098,-1.232898,5.372544,-3.188607,6.139253,-6.780454,-1.835430,8.918493,1.850691,0.582042,-4.194007,2.261115,-3.340695,5.481214,0.170272,-5.054656,6.593064,3.623516,6.754713,-0.474089,4.199307,-2.519239,-6.408373,-8.544875,5.213911,-1.568727,-9.783312,-0.258208,2.996390,4.322900,9.028420,-4.501022,-5.117509,0.199789,0.012098,2.698781,4.886212,4.603350,6.488175,5.788389,3.498430,2.510492,9.804644,6.395677,1.378775,4.510648,-0.990507,0.266596,-6.695700,9.652798,8.652556,-7.497509,-8.486577,-8.072743,-9.098056,-3.419863,6.630111,5.022779,7.136170,9.854935,4.161248,-5.431132,-8.908769,2.214880,-4.792857,0.299231,6.853220,-4.837185,4.211076,8.864020,0.485283,5.013377,5.581234,1.235096,9.879449,-2.064980,-4.727774,9.501477,-3.449204,7.664045,4.140327,-4.043092,-6.922626,-3.293123,-3.921331,7.380576,4.662492,-3.738135,4.370256,9.674139,7.697951,-3.245706,-7.190815,-3.994271,-1.719611,-0.556916,-0.871952,-4.505892,7.538953,-9.820854,-1.489166,5.594517,1.843740,4.572588,-8.129669,7.870686,-7.936214,6.419689,0.198044,7.802275,2.068109,6.754190,9.358008,-7.829745,-2.730816,-7.988977,7.158184,-1.875673,0.869300,6.924612,1.606026,7.136008,-9.085806,-0.035810,5.363641,2.245985,6.792856,-1.084736,-7.559228,-1.840048,-4.322901,-3.624812,-7.354705,-6.729890,-8.093168,2.116336,-2.906848,0.238052,3.976586,6.710062,6.179348,2.023685,4.094206,9.261489,7.624749,1.630033,-3.216618,9.774477,-4.919919,7.514357,7.152476,-9.540734,7.891239,7.073494,-8.408714,0.031845,8.647647,-5.744062,6.379065,9.696204,1.341411,-3.177559,-9.969754,4.331123,7.401408,7.491891,7.015944,-4.428596,6.586336,-2.554968,-6.785297,2.533421,-4.767507,5.107650,-5.319841,-0.064317,5.990787,3.450104,-5.373918,-3.339603,0.546220,1.811192,-5.878967,-9.298425,4.241670,3.806082,-8.851780,5.621565,-8.140298,-5.871181,3.844144,4.026141,8.056169,-3.587590,2.589747,7.934198,-8.021780,9.527821,5.421588,-6.490043,-7.729161,-6.435222,1.653383,-0.151409,-8.551940,-0.074371,1.111654,-6.715894,-9.278501,5.767376,-5.564651,2.687288,9.283396,2.106287,-0.294449,2.817885,3.039095,9.063233,2.485922,-9.550561,2.678065,-5.219239,-3.528910,9.281400,4.087284,-0.441954,-3.859216,3.336586,6.574453,-2.599404,-0.469072,-9.222298,2.806049,-8.951782,4.401816,7.396561,8.472860,-3.399430,-7.114879,0.606911,-5.302619,7.369673,4.042022,-3.723886,1.875618,-6.126071,-8.984070,7.306868,-1.781408,-4.457985,2.995206,-2.132284,4.571512,-7.202001,1.200515,-2.991605,-2.879538,-4.254031,-0.033978,-6.047053,5.706964,8.547845,8.444623,0.093478,5.216796,-0.395064,8.357657,7.802721,2.943203,9.281244,2.459882,-5.862059,0.695395,9.474500,7.109465,-9.981723,-7.562630,-5.729460,-2.722012,-5.591547,4.117092,-5.620474,4.420208,-3.763216,-0.478777,3.327256,8.303788,2.054869,-8.537644,-1.877419,-0.621313,-9.561102,-5.282166,-5.072606,4.805280,-7.994225,6.378837,3.587087,1.026056,5.067851,0.229917,2.676730,-6.664445,-5.863305,-3.154857,-7.030776,7.860435,9.648421,0.667408,-4.220434,-8.408700,-9.675959,-4.140781,0.214656,8.162834,2.648849,3.974019,1.055777,4.292831,5.549740,-5.042335,-5.098147,-2.970773,1.483058,2.342175,-1.308924,5.413434,-9.100132,2.942117,8.246915,-3.333500,8.650190,2.611692,-9.868671,-1.839349,-1.174848,9.960397,-2.648791,8.771251,-6.186270,-5.122166,-7.469337,4.169167,0.683311,-9.902708,5.395997,8.502175,2.973992,9.995762,9.976117,8.373102,-4.330004,-3.603087,-3.395425,-4.336772,-7.546706,-3.342128,-0.512589,3.627371,-8.917200,-4.703422,6.557993,-9.982163,0.475793,-8.996132,-8.492388,-3.739624,-4.888974,3.752251,-4.526498,0.903339,-6.004318,-4.458168,2.619223,-8.754148,5.138999,0.155091,-3.562250,5.566373,3.825262,-6.305202,8.121483,-7.652781,7.329388,9.665648,-2.518787,2.648930,-8.124562,4.217951,3.528126,6.965566,5.018993,1.338707,-1.723003,2.006652,-5.966496,1.389518,-3.315499,-9.964818,-9.859180,-8.998520,-7.724060,8.266511,1.525232,9.886896,6.302838,8.695517,8.950735,-5.047840,-7.862617,-6.565510,-8.333470,-0.082213,4.483465,-0.142897,-9.165940,-4.140231,-3.618521,0.524579,6.634095,8.523997,2.278983,6.330051,7.329394,2.743126,6.247639,9.491631,1.418531,-0.743030,-8.142489,-1.683196,2.686713,3.091233,2.855226,1.572861,1.952913,4.610198,-8.849211,-8.868909,-2.540686,6.436879,0.072782,9.037909,-4.298928,4.938664,5.364733,-3.709784,1.991673,-6.721266,-2.498451,2.509879,-1.102035,5.230692,4.098758,5.744100,3.170003,1.748938,4.196685,1.492150,-1.105557,-3.850602,3.684210,0.607643,-6.967652,0.287188,7.392201,2.499637,8.467578,-3.360660,-7.182241,-6.999401,-6.357659,-8.088141,4.604155,-4.562505,-4.982893,2.619718,9.375463,-6.740637,-4.898705,-7.830028,-1.634264,-8.257080,-7.868635,-4.254372,-5.531738,-2.742654,-7.600504,-8.850468,3.238665,-8.783222,0.947818,4.630526,-8.162085,-1.289533,0.026636,-9.843026,-4.694150,2.469505,-8.993332,2.809703,-2.501694,6.323050,-3.443630,-5.216141,-9.687447,-2.197584,-9.716486,-9.917436,1.648949,-5.000441,-8.241679,7.207894,0.454790,0.416988,-8.563655,-0.966495,0.674958,-5.597386,7.683191,1.652030,9.701501,2.913105,5.348356,-2.298932,-3.232736,3.936490,8.495787,-2.280054,3.653170,-5.724218,-6.444944,-6.110625,5.194442,-4.034319,-7.469668,-7.985282,2.906102,1.226968,-1.985092,0.123606,-8.136866,4.426972,2.410222,-9.746136,2.582742,0.512884,6.894808,4.551484,-0.004963,4.606495,8.476567,-6.999260,6.826754,3.364766,4.240911,7.164008,3.994328,6.386877,-8.377489,6.452568,0.697248,1.070575,2.677369,7.457079,-3.470251,-1.950514,-7.235363,-9.656945,-6.769121,8.350711,1.968762,-7.577404,6.128568,5.644802,0.792776,-0.666655,9.467053,-7.839921,-1.453823,-1.680819,-5.054580,2.532398,7.413083,9.701370,0.586936,1.977522,5.539761,-5.573645,-6.393288,-4.106942,-6.330709,-9.306185,-2.034808,-8.905491,-5.687356,-2.347668,5.557005,7.124184,6.489108,1.269323,-9.222713,9.610592,-0.507402,-0.427595,-9.921278,-7.536783,9.394594,-6.029027,3.866772,3.280759,-0.564406,4.731953,9.592634,-5.576052,8.496989,4.099713,2.234087,7.448355,-1.623092,-1.065619,-5.644144,-6.049036,6.778886,1.591261,0.894243,-1.524425,-5.618325,-9.548643,-6.689046,9.938273,-7.672221,9.714898,5.892645,0.719188,7.786206,-4.454953,-8.808879,4.762695,-4.685019,2.951472,6.140175,3.087966,1.894342,7.220024,-7.211551,1.811495,-8.223876,-4.982340,9.489280,8.599995,8.254409,-2.496206,-7.186738,4.304664,-8.076067,3.779606,-0.975498,-0.729698,-2.569972,-3.442325,-7.301052,7.956986,-9.726313,-3.830129,7.844009,-2.168296,-3.030370,-2.566526,9.983935,2.748153,-5.811454,7.365637,1.718494,-2.069864,-9.301719,-9.362845,-4.103382,8.873425,-7.080756,-1.415672,0.849560,-3.744793,6.912271,1.254450,7.095684,6.164440,0.613616,-1.968151,-9.081128,-1.443118,1.035950,0.757916,-4.787082,0.173560,-1.055228,-4.714319,-9.915388,1.466017,4.001948,-4.010900,3.183433,6.712543,6.536252,5.282902,-6.853444,-0.121463,1.412793,-1.093471,-8.543977,-3.783867,6.035891,-6.816907,1.267796,-7.986037,-3.490924,0.942531,-7.822675,-7.713160,7.134909,8.264980,8.689850,6.504342,3.385098,-2.591073,5.025266,-7.480447,6.570939,4.117017,2.762430,-8.415299,-4.313950,3.111316,-7.392942,-4.092653,-5.652788,1.920367,2.940094,-7.981643,-2.401885,-1.721218,5.804386,6.416796,-0.466405,9.753045,-8.714755,0.405925,-0.952871,0.852249,3.471159,-5.950578,5.335366,7.911306,-5.744919,0.444366,4.606594,2.744221,-4.571496,2.105125,4.387693,-0.787490,-4.594305,-5.063297,6.865210,9.034860,-6.605805,-6.726070,-8.590376,-6.770812,4.551430,6.593379,-0.502571,5.726025,4.955412,9.611966,3.152591,-1.657331,6.135925,-8.697770,-4.103521,3.806804,9.629197,-4.294503,-5.369392,2.128283,-7.012210,-6.895013,8.871366,-2.339779,-5.109335,0.828019,5.178862,-8.718740,-1.720504,3.690024,-0.087944,5.080876,-6.095498,7.680391,6.392351,-7.793601,-2.679042,-3.678462,-4.787027,-1.669641,-1.821244,6.096911,-8.000038,-8.579062,-0.507446,4.285245,4.160162,4.782037,7.141073,-5.351059,5.088263,-0.497265,3.126442,3.112011,8.630599,-2.581580,9.521426,-2.256139,2.299739,-5.339947,-5.548435,4.152898,3.988510,-0.727436,7.138471,3.325529,1.478478,-5.796954,-2.731136,3.449793,-4.521800,9.722765,2.417846,8.904013,-1.070263,-0.739541,2.424618,-3.207071,-9.149342,0.418719,7.215266,-6.325126,2.582353,0.999770,-8.198517,7.918452,8.945267,-7.643572,-8.149538,3.883419,-9.879942,-3.271339,3.365103,-0.793145,3.645639,-7.241588,-7.834271,7.536762,7.413805,4.386198,6.955637,-8.110812,8.453880,5.772883,-5.693734,-4.842677,7.960758,7.894040,-7.171793,-6.182716,6.814766,-1.335243,-6.313868,6.139614,2.587514,9.011132,-6.539247,1.194561,9.230474,1.880909,7.344995,-5.074419,-2.221587,-8.321534,9.118994,-0.015639,3.566598,-9.772184,7.594479,2.840590,-0.398923,5.183873,3.177317,-4.139788,-8.016318,-9.378454,4.140053,-5.361627,-9.051534,-8.899670,6.906038,4.411617,-9.256117,-0.849594,0.946310,-9.697514,7.953319,-0.255263,-3.700949,4.987300,5.578073,-0.149804,-8.794196,-1.589544,5.428297,-1.297742,-4.420749,4.659383,4.477136,-8.494761,-3.490463,-5.644716,-6.752559,-7.068709,0.377337,7.739912,6.237650,0.633181,-5.022122,1.196448,2.598775,1.775961,2.965119,-6.712142,-5.932699,8.172308,4.904978,-5.233708,6.446305,-7.489017,-7.888572,2.743910,2.564242,3.974176,-7.588218,-5.132787,-0.483345,1.388302,-8.226479,-3.956446,-7.956628,9.481815,4.554245,-5.114679,-2.312689,0.568842,-5.011369,2.727619,-1.828712,-4.544896,3.233935,-9.364158,-9.271307,6.494202,6.795871,5.358638,7.899582,8.586465,2.967642,1.170199,-9.287071,-2.778984,6.794880,-5.094222,8.111417,1.189127,3.039095,-0.391386,9.604217,7.039233,-9.147362,3.551213,1.293521,-1.946967,-7.773664,8.028880,1.554334,5.998286,1.359739,-3.294090,-6.986308,3.906023,5.287615,3.716027,5.798345,9.605746,-6.011447,5.507775,3.591281,-9.084258,-0.136206,-2.756241,0.057419,-7.985320,1.107017,2.200997,-8.482745,-1.618565,9.164431,1.625288,-2.915009,-2.969062,9.879660,-4.800026,5.298340,-0.681891,-0.830660,7.091073,-2.439592,-6.159365,-3.161842,-5.713990,-6.320140,-3.148447,-3.809146,-0.270648,-3.576711,1.294182,4.006181,-0.307271,-1.398290,-5.402758,6.660396,0.727668,2.185473,-9.133041,7.037612,2.757084,0.619850,8.364012,8.836250,-8.829299,-7.044816,1.432530,6.526889,4.744609,-5.603730,1.078043,7.093084,7.628244,-4.354801,1.691439,-0.782719,-4.396263,-6.504497,-0.703588,8.290168,-4.508507,-4.194370,6.484757,1.474571,-1.878828,-9.575132,-6.491276,-5.587397,8.005884,3.582703,-4.914937,5.140260,9.499413,9.326416,0.162655,-8.786770,-1.428393,0.266301,-3.144571,-0.162862,-3.008848,-9.701544,8.244192,-0.660589,-2.083268,5.958779,-2.213582,2.982758,-6.788460,3.613053,5.208738,8.365127,-2.098029,-4.433523,-7.480169,8.069701,0.054350,8.080878,-6.065739,7.031102,1.803464,-8.717727,9.253382,3.940148,-9.902485,2.725948,-6.938597,9.181549,3.040666,4.165675,7.290856,-3.752260,6.369872,5.615523,-7.279944,-2.094179,8.655332,7.362001,-9.531699,-6.467105,5.876850,-8.651459,0.044232,-8.142163,6.091609,-1.117618,4.523524,-0.181349,-5.936044,-7.261113,-5.983440,-8.960904,7.771153,6.619942,-9.968777,-6.319523,3.757317,9.996417,6.206829,-5.867489,7.605665,7.770248,3.446473,1.033401,1.049796,-4.238282,-9.077243,6.859403,-8.713335,9.933420,-6.177226,-2.927926,2.120829,-1.861119,-2.283460,5.030054,9.595510,-8.167364,-3.332246,-2.902230,5.819794,-3.227435,7.370314,-7.733365,-2.630297,-8.229515,0.342427,-6.268626,-4.179493,2.872246,0.989972,5.479999,3.283027,-8.649022,3.732875,4.313828,5.427902,8.926370,4.138417,6.483116,1.361981,8.684387,-5.299364,3.401927,-9.006038,2.875517,-8.101565,9.701368,-0.889575,6.455972,-6.515124,6.612094,-5.659786,4.124894,-9.332441,0.352948,-8.554818,-2.798032,5.741126,-4.923255,3.600510,1.613990,-4.826568,0.301639,-8.970735,-3.737336,-1.222308,1.314577,-8.633741,-3.869659,2.028284,1.101196,-6.801932,-7.043305,6.738087,-1.715500,1.535143,-7.064918,-7.204159,-0.833224,9.230001,3.483988,-4.332960,-1.904318,6.218771,7.182997,6.591141,0.291757,6.669421,1.327955,-4.002306,0.060297,8.304556,0.576419,3.922899,0.685237,-4.080423,-3.244203,8.694102,-1.247193,-4.968213,5.519519,-3.761128,3.482221,-6.722007,-8.411941,-2.663325,-7.828712,-5.949914,0.031202,-9.424262,-4.603957,-9.422055,7.419377,5.606245,-4.733898,-9.536578,0.867099,-4.857613,-1.920047,6.533879,3.185114,-6.383549,-3.185485,7.109208,-7.688259,-9.022808,2.875324,-7.742134,-1.490898,-0.341398,3.493468,-4.747008,7.149882,4.941682,1.310197,-7.219036,8.304045,9.264386,7.694012,6.181662,-3.374887,-7.784149,-6.415558,-1.563106,6.870481,-3.330074,2.243567,2.497301,7.971185,-2.863109,-6.552350,-7.178644,5.625621,7.376805,-8.178895,2.295511,-6.066623,-6.176859,-8.430189,5.880415,7.474400,-1.810114,-4.960685,1.528633,-2.893546,-7.291791,9.124021,-6.415066,-6.175811,4.989551,7.080504,3.397941,0.276410,-6.599072,7.376147,5.274362,1.423929,1.671927,9.205059,-9.430970,3.053298,-0.929667,2.845285,0.583952,-7.522047,-1.964718,4.223714,0.422023,-0.248032,-9.458382,1.128004,-3.525266,2.158942,-2.387331,2.663586,8.019517,-9.679260,3.903965,-6.221024,2.596365,1.588282,-9.972722,-8.435038,5.881603,-9.453827,-4.915591,8.782508,-5.675523,-2.988936,-5.438913,7.366704,-9.493399,9.166183,0.282865,5.012734,3.513667,9.165507,5.719004,1.001242,-1.459628,1.665606,-7.975275,4.224406,2.475439,1.619243,0.218128,1.404183,-1.497340,3.331051,1.507304,9.994274,-4.762421,-4.537616,-4.618146,-4.521409,6.429411,9.447271,1.683673,2.514962,-4.680259,-5.548473,5.246840,-7.574604,-8.534975,-5.575292,-0.765610,-6.192900,-7.802856,-5.664411,3.077240,8.954724,-2.920164,-2.526764,1.609059,6.435490,-4.427589,-8.928812,3.402752,-3.563468,-7.049823,6.994586,-8.353722,1.355127,7.767853,3.516866,8.685125,8.012910,5.498081,0.840631,4.823815,-5.169619,-1.317145,-2.342692,0.569301,6.049794,7.273415,-7.671038,9.714094,-9.954237,-6.572355,8.833016,-0.047399,3.780082,-1.072020,-0.104462,8.734363,5.985256,6.455651,-3.928548,-4.416522,3.631588,-0.378104,2.102752,4.838292,-2.716511,7.634940,4.277410,8.267517,5.375689,3.474618,2.852695,6.570331,-0.251969,-7.560031,3.651022,1.037289,-1.611226,9.134038,7.375426,4.403244,-7.343867,1.810380,-1.073694,-6.672935,-9.027748,-9.021305,1.754872,-4.186243,-0.086497,-5.493446,-2.096351,-1.504587,7.893995,-7.927503,0.952216,0.039920,0.919273,9.304186,0.265618,8.536871,-8.675296,4.186833,8.823581,7.915873,5.490599,-9.449498,4.203410,-3.514605,-1.101008,1.780571,3.283357,5.751777,7.575000,-4.467365,-9.024473,-8.192746,1.785734,-1.614067,-5.812494,7.486029,0.867793,4.593539,-2.809677,-5.051877,-6.963510,-4.822896,-1.286558,-7.484283,3.290454,-7.050360,5.810058,-1.313051,-8.079588,6.955200,-7.198503,-5.951230,-4.558443,5.746320,-6.593151,2.420363,5.703534,-8.115217,7.944864,-6.487723,4.061375,7.971786,5.090060,-6.510326,-9.036505,-3.162533]], dtype = "float32")#candidate|4332|(1, 1890)|const|float32
var_4333 = relay.var("var_4333", dtype = "float64", shape = (325,))#candidate|4333|(325,)|var|float64
call_4331 = relay.TupleGetItem(func_2018_call(relay.reshape(const_4332.astype('float32'), [14, 9, 15]), relay.reshape(var_4333.astype('float64'), [325,]), ), 2)
call_4334 = relay.TupleGetItem(func_2021_call(relay.reshape(const_4332.astype('float32'), [14, 9, 15]), relay.reshape(var_4333.astype('float64'), [325,]), ), 2)
bop_4336 = relay.left_shift(bop_4297.astype('int16'), relay.reshape(var_4296.astype('int16'), relay.shape_of(bop_4297))) # shape=(14, 14, 2)
bop_4339 = relay.left_shift(bop_4300.astype('int16'), relay.reshape(var_4296.astype('int16'), relay.shape_of(bop_4300))) # shape=(14, 14, 2)
output = relay.Tuple([call_4312,var_4313,uop_4319,call_4331,const_4332,var_4333,bop_4336,])
output2 = relay.Tuple([call_4314,var_4313,uop_4319,call_4334,const_4332,var_4333,bop_4339,])
func_4342 = relay.Function([var_4296,var_4313,var_4333,], output)
mod['func_4342'] = func_4342
mod = relay.transform.InferType()(mod)
var_4343 = relay.var("var_4343", dtype = "float32", shape = (14, 14, 2))#candidate|4343|(14, 14, 2)|var|float32
var_4344 = relay.var("var_4344", dtype = "float32", shape = (6,))#candidate|4344|(6,)|var|float32
var_4345 = relay.var("var_4345", dtype = "float64", shape = (325,))#candidate|4345|(325,)|var|float64
output = func_4342(var_4343,var_4344,var_4345,)
func_4346 = relay.Function([var_4343,var_4344,var_4345,], output)
mutated_mod['func_4346'] = func_4346
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_4351 = relay.TupleGetItem(func_3866_call(), 1)
call_4352 = relay.TupleGetItem(func_3867_call(), 1)
func_851_call = mod.get_global_var('func_851')
func_855_call = mutated_mod.get_global_var('func_855')
var_4361 = relay.var("var_4361", dtype = "float32", shape = (330,))#candidate|4361|(330,)|var|float32
const_4362 = relay.const([4.718863,5.520212,4.914705,5.041170,-1.246473,-9.717413,-2.739338,5.291669,0.011128,-5.928868,-2.147004,-1.142586,-5.547669,-9.222838,9.973452,8.686572,-0.867234,9.643468,0.944964,-5.129354,4.821567,-9.370866,9.419261,-7.580175,-2.371068,-8.961031,9.047033,0.378448,-3.690557,2.719351,9.736717,-8.664997,6.724399,5.245675,4.556070,-5.125729,-4.647753,-7.990247,7.290584,-2.472129,6.450631,-7.253054,5.723804,-4.339199,1.610186,-7.348573,3.130917,-5.542197,-1.850471,6.581804,-3.202531,-9.956694,-9.330640,-2.473275,2.269985,7.064117,9.037090,9.084437,-9.812975,-6.118683,-9.987957,-3.694116,-7.568780,-8.030604,-8.501855,-4.849416,-3.860717,4.700835,1.804341,-8.143739,-0.222522,9.114882,8.441530,-9.570385,-8.567663,-6.909325,8.518604,-0.964472,5.483651,9.845008,1.592746,-3.905008,-0.089827,-8.357028,-4.880620,1.825063,7.895216,-0.554991,6.857401,-7.818552,-3.422286,-5.681868,5.032857,6.028242,-1.753823,-1.104623,-8.986747,1.357690,-7.786190,-4.590842,-5.990570,2.391185,-4.122339,-0.581021,-7.573268,6.381635,9.493809,7.970715,7.071803,-9.446296,8.265911,8.319155,7.986448,-7.346366,-5.679502,-0.899868,-0.032298,-2.709408,-6.135138,9.021234,6.163535,0.161907,-6.298434,4.346019,6.166055,2.036799,-9.431883,-4.938420,1.070051,4.591939,5.315206,-2.733299,9.286200,-1.123049,-9.044173,5.146487,-7.112595,-4.664483,7.055771,-2.855031,6.911089,7.941454,-2.169135,-1.637830,-1.001779,-6.083289,6.903168,-4.352931,3.986163,-8.691991,-0.898494,-5.466291,-8.519196,7.815112,-1.236949,-1.254739,-7.943906,-4.332349,6.437496,-4.646099,-7.826516,6.399914,-8.909397,3.939487,-6.348399,5.253336,2.025381,-3.366797,-6.302769,8.783229,6.245141,7.223989,1.061432,0.926372,-6.427133,-0.038863,-4.988633,-6.998537,2.065972,-4.793552,-6.515452,-8.068727,7.610748,3.871780,7.339454,7.790949,2.800038,-3.471672,-1.938457,8.703543,8.151989,1.947340,-8.608067,-7.689731,9.574921,4.216658,-1.848601,3.209136,-1.294567,-5.329312,5.594765,7.385767,-7.080744,8.688270,7.089300,9.107529,9.255863,-6.986288,-8.084779,7.759465,4.856120,3.249738,6.289454,-8.848362,-0.436890,0.550509,-4.072081,-5.898956,-9.908326,5.889634,-1.526413,-5.873880,7.655077,4.263768,-1.295496,-4.443867,-4.844100,2.373358,-1.670572,1.226680,-9.333223,6.472559,6.269200,8.034910,3.734293,5.605947,4.079896,3.845008,-8.941456,2.588732,-9.479785,4.502090,6.760636,-9.423259,-2.747263,-6.839749,-2.596648,8.466268,8.361273,0.557665,3.281345,2.607997,-3.419963,0.285818,3.794844,7.143447,-0.079036,-5.934670,-0.357595,-6.207290,0.988635,6.808291,-1.387186,3.991808,-0.271797,-1.940909,-5.046647,-1.213812,2.557333,3.558819,-3.017327,-4.456480,1.220345,8.002737,7.708970,-8.750188,-3.288190,-6.510000,-3.664616,6.598275,-3.177208,-9.357372,5.194162,5.479242,6.054731,4.748421,-3.315572,-9.653981,3.400782,6.795244,3.910482,5.976589,8.714735,-4.327284,5.578161,-0.348735,-7.842204,-0.313156,-5.610041,-3.494243,1.647590,0.169156,5.513638,9.607856,9.689779,9.411314,7.974838,1.019983,-1.629005,-9.212351,4.286552,4.848236,-8.454873,4.403410,8.913539,-1.115842,6.648060,-3.779662,7.368935,9.818796,-3.568376,-2.579415,9.795353,7.821828,2.146244], dtype = "float64")#candidate|4362|(325,)|const|float64
call_4360 = relay.TupleGetItem(func_851_call(relay.reshape(var_4361.astype('float32'), [10, 3, 11]), relay.reshape(const_4362.astype('float64'), [325,]), ), 1)
call_4363 = relay.TupleGetItem(func_855_call(relay.reshape(var_4361.astype('float32'), [10, 3, 11]), relay.reshape(const_4362.astype('float64'), [325,]), ), 1)
output = relay.Tuple([call_4351,call_4360,var_4361,const_4362,])
output2 = relay.Tuple([call_4352,call_4363,var_4361,const_4362,])
func_4369 = relay.Function([var_4361,], output)
mod['func_4369'] = func_4369
mod = relay.transform.InferType()(mod)
mutated_mod['func_4369'] = func_4369
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4370 = relay.var("var_4370", dtype = "float32", shape = (330,))#candidate|4370|(330,)|var|float32
func_4369_call = mutated_mod.get_global_var('func_4369')
call_4371 = func_4369_call(var_4370)
output = call_4371
func_4372 = relay.Function([var_4370], output)
mutated_mod['func_4372'] = func_4372
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4405 = relay.var("var_4405", dtype = "uint16", shape = (15, 1, 9))#candidate|4405|(15, 1, 9)|var|uint16
var_4406 = relay.var("var_4406", dtype = "uint16", shape = (15, 3, 9))#candidate|4406|(15, 3, 9)|var|uint16
bop_4407 = relay.bitwise_or(var_4405.astype('uint16'), var_4406.astype('uint16')) # shape=(15, 3, 9)
bop_4410 = relay.add(var_4405.astype('uint64'), bop_4407.astype('uint64')) # shape=(15, 3, 9)
output = bop_4410
output2 = bop_4410
func_4417 = relay.Function([var_4405,var_4406,], output)
mod['func_4417'] = func_4417
mod = relay.transform.InferType()(mod)
mutated_mod['func_4417'] = func_4417
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4417_call = mutated_mod.get_global_var('func_4417')
var_4419 = relay.var("var_4419", dtype = "uint16", shape = (15, 1, 9))#candidate|4419|(15, 1, 9)|var|uint16
var_4420 = relay.var("var_4420", dtype = "uint16", shape = (15, 3, 9))#candidate|4420|(15, 3, 9)|var|uint16
call_4418 = func_4417_call(var_4419,var_4420,)
output = call_4418
func_4421 = relay.Function([var_4419,var_4420,], output)
mutated_mod['func_4421'] = func_4421
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4186_call = mod.get_global_var('func_4186')
func_4188_call = mutated_mod.get_global_var('func_4188')
call_4456 = relay.TupleGetItem(func_4186_call(), 0)
call_4457 = relay.TupleGetItem(func_4188_call(), 0)
output = call_4456
output2 = call_4457
func_4462 = relay.Function([], output)
mod['func_4462'] = func_4462
mod = relay.transform.InferType()(mod)
mutated_mod['func_4462'] = func_4462
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4462_call = mutated_mod.get_global_var('func_4462')
call_4463 = func_4462_call()
output = call_4463
func_4464 = relay.Function([], output)
mutated_mod['func_4464'] = func_4464
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_4492 = relay.TupleGetItem(func_3866_call(), 2)
call_4493 = relay.TupleGetItem(func_3867_call(), 2)
func_3767_call = mod.get_global_var('func_3767')
func_3772_call = mutated_mod.get_global_var('func_3772')
const_4531 = relay.const([3.729830,0.916319,-6.686894,8.997837,3.131290,-2.032671,1.981089,5.571331,-3.965365,2.180255,-7.382474,1.482042,-8.688006,9.208825,-0.018468,6.018882,-1.624493,-1.919945,-0.555852,-9.641147,-2.828850,-4.325321,3.774878,-9.881042,2.396664,8.019428,-6.508329,7.620017,4.520609,8.603514,7.310568,-6.113626,9.685003,2.991920,-3.430850,4.058002,-7.432070,7.884001,3.496304,-7.269437,6.851865,-6.385432,-5.641378,-6.214435,7.499752,2.179886,-5.103601,-1.631930,8.910476,4.246983,-7.664748,-1.240586,-1.762275,-6.123428,6.076585,-5.593400,-8.557524,4.922469,-1.488037,-6.063886,5.929472,7.971225,2.053935,8.642400,9.082069,2.058391,-3.539763,-5.895861,5.472217,-4.658368,9.894509,8.713494,-6.868911,-9.189110,-1.616043,-7.149927,3.300734,3.254320,1.286735,4.699683,1.021872,-7.487584,-3.567341,3.164881,8.690518,-2.036551,-7.442358,-6.941336,-3.623746,-5.180958,-4.127837,-4.877382,-5.073067,-0.464913,9.568498,-4.657763,-4.972181,-3.760865,6.691422,-2.782208,6.935796,-2.390580,-1.447018,-1.796467,1.819161,7.534918,4.957081,7.682573,3.647928,-8.015078,7.132159,-9.985696,7.385513,-3.309023,-3.604322,-0.780202,-0.961925,0.629357,2.761299,3.054003,-0.836753,-5.735433,5.633051,7.765092,-5.873275,1.029736,-4.264045,9.516209,-3.757405,9.713061,8.779229,-9.497151,-7.942082,-5.074567,-4.024366,-6.407542,8.918260,-5.345554,4.312039,1.129286,5.387589,8.742629,-1.183607,-9.991965,7.554240,1.994657,9.904289,-7.377047,-3.201982,4.426125,-2.318570,1.804252,2.119624,-5.992602,-9.222101,-7.913301,-9.889219,2.139246,-0.892750,6.204015,0.981977,-7.980079,6.145564,-4.715733,-2.598266,-3.446873,-3.950109,-3.140659,4.386037,-8.613238,-7.693922,-6.276548,-0.342131,-5.456755,8.078421,-4.034176,-5.734084,8.350405,6.743855,4.137682,4.908717,8.526537,-9.216098,3.625213,-6.166212,3.179819,-2.147773,-0.209803,5.454791,-5.558596,-1.138246,-0.269948,2.602226,-1.268284,9.124247,-3.448976,9.989207,-1.023186,7.597057,-8.791003,6.003551,0.956218,0.771671,2.215553,-8.845522,4.162619,-2.346160,3.565617,1.496924,7.821342,-5.221306,4.444072,9.181300,4.943519,-6.375921,-2.339273,9.564143,5.140634,4.166219,5.445543,-4.394381,2.594252,-2.066847,8.285228,-4.074076,5.846630,-1.237267,3.311955,-6.972619,0.607892,-5.916012,6.536019,7.540378,-5.318359,5.464359,-3.256489,-9.441832,-3.070320,-7.028255,-8.667270,-5.269820,8.167658,-4.753012,-9.224123,8.591773,0.525198,-6.495237,6.891568,-0.049850,-1.400884,-3.594150,1.435099,-7.993174,6.887070,7.719122,-6.431828,9.747274,1.526030,-0.901371,8.531327,3.860335,1.979979,2.681725,0.656747,-1.659057,8.511039,-5.859550,8.610242,-8.460481,6.398243,-7.006998,-8.751829,-2.941893,5.420376,1.347418,-1.637839,-5.580217,-3.668049,1.016657,4.972162,-0.449278,-6.942977,5.465252,6.162485,-2.467992,-9.786793,-3.201048,3.699728,-5.397149,8.229347,0.616798,-2.589663,5.486709,2.964024,-8.170063,-3.975971,4.534955,5.426551,-7.142734,-5.567120,9.660828,6.837716,8.776841,-1.476954,-5.696649,-0.331944,-0.421433,0.772290,9.020244,-7.660350,1.834657,-4.692082,9.670590,9.181424,2.334758,-1.928761,-2.227768,-2.152251,-4.746580,5.544126,1.246194,-5.192072,-3.773415,8.837315,-8.673172,0.075657,5.499868,-0.097078,1.667434,-7.611520,-6.388162,-5.645872,7.625527,-1.668033,4.954553,-3.112695,6.239375,-5.432746,-4.380619,-3.603270,9.726145,-0.187295,1.700285,1.566098,-1.971098,1.978807,8.611565,-0.801041,3.460721,-7.009515,-2.379695,5.751425,-3.644992,-3.652790,5.723779,7.895107,-2.428979,0.149215,7.534918,9.732477,0.861344,-2.807182,-9.706293,5.552898,2.202370,-3.887511,-8.416403,-9.524417,-3.719232,1.041913,7.624009,-8.332804,-7.640290,1.007151,3.469805,4.715403,-8.723323,0.649199,-7.805589,-6.112253,-7.500804,-9.176672,-0.295792,1.585882,2.229761,-3.577860,-4.699360,-6.526477,8.913486,-0.983973,-2.350960,3.118253], dtype = "float32")#candidate|4531|(392,)|const|float32
const_4532 = relay.const([2.181363,3.978666,7.843813,-4.922765,-3.496297,-1.651863,-2.446687,-2.131386,7.437698,3.569896,9.191523,2.316428,3.760360,-6.324153,3.248948,0.170210,5.629020,9.176936,-8.559704,-9.807884,4.885632,-3.520560,3.799864,-0.155179,5.310189,5.960482,1.366979,5.116228,7.752944,1.236107,4.581686,3.300865,-1.156266,9.751523,-8.119223,4.232863,-5.193833,9.366630,5.977647,8.163018,0.968936,8.291584,5.293089,-9.914245,-8.833674,-5.641744,-4.791549,-5.734125,-7.916404,0.489542,9.094692,-2.142626,-8.170015,9.299692,-9.175473,-9.927050,-8.900588,-9.472691,-1.577294,-0.374076,3.852922,3.089030,0.441290,-2.767224,7.285256,-6.705863,-2.487353,0.639742,6.756167,-3.832762,1.439255,6.623246,-3.202007,5.986330,-7.108851,6.191620,-2.606617,3.645618,-4.553057,3.355817,9.163133,-7.101075,8.889962,-1.733988,0.711654,-6.656686,-0.698717,-8.923743,0.505462,5.694005,6.429122,7.632718,0.286845,2.221626,-6.587172,-7.420911,5.113143,-1.518159,-6.802153,8.036029,3.001932,-1.552400,0.841191,4.187902,-0.532153,5.785175,4.929955,-0.918804,8.516926,6.997555,-7.113726,-6.066351,-5.563974,-8.736031,1.981902,-2.381426,3.506558,-1.390418,3.141310,-2.162459,9.989448,-0.215534,-4.783473,1.976293,-8.280935,-1.450337,-8.220932,-4.890707,0.201310,6.750301,-7.128305,7.237058,-7.557449,0.664672,3.032407,-8.843324,6.620410,-4.181706,0.286755,3.387591,-2.760085,-8.653167,-3.822357,3.526459,-6.539385,1.761005,7.886622,-7.007326,-2.923835,6.396817,6.304655,-6.230211,-1.211487,8.542175,2.753159,-6.714418,-5.887295,-5.239539,3.027205,-2.717203,1.970595,9.796595,-3.344750,-7.206962,0.046111,8.752081,1.689237,6.724173,-4.300113,4.449435,5.569973,8.658950,7.923211,2.441324,0.835574,-1.812867,5.924739,7.023274,9.157889,-5.671360,-4.830191,-5.784216,-8.596915,3.620661,-5.960323,-1.915232,-7.696599,4.036967,8.109387,9.938461,-7.808879,3.767823,5.136239,7.135989,8.438282,5.038492,7.537556,6.787871,7.984603,-2.609639,-1.336450,1.034859,-6.104173,-3.398455,1.620266,1.744260,6.337132,1.186981,6.929775,2.361270,3.427375,-1.703803,-1.090734,-7.254959,9.998797,5.185334,4.087875,-1.323585,0.278038,5.135579,-5.393159,-9.429391,-1.758702,-7.928624,7.324243,-5.873453,-5.359843,4.714502,-5.358146,-3.718343,2.245189,5.647790,0.172764,-9.482995,-1.761791,2.128568,-8.106876,-9.794333,-6.869994,9.515239,2.985154,-5.912965,-2.389624,-8.349083,8.403962,-3.879967,8.555197,4.197342,-9.390843,9.170808,8.277897,3.701180,7.899237,-3.787388,4.808392,8.357844,7.966000,-9.950101,2.327700,8.045205,-0.283643,2.930303,3.743821,-9.579298,2.088474,4.194601,-1.585693,-7.604027,-0.958693,0.324043,-8.448274,-7.529498,-7.149390,-4.615679,-9.329565,-9.241068,9.758642,-0.162433,-9.173239,-5.268587,-1.587145,2.465672,0.809470,-8.714816,-9.958517,9.880783,-0.840599,7.558746,-5.833837,8.893364,3.528869,3.711878,0.289755,8.212648,-8.904117,6.980626,5.211980,-2.631720,4.840241,5.862054,-7.433507,-5.203443,-2.428405,-5.289634,4.628225,1.911728,8.412762,-9.467207,-5.756552,4.288956,4.305771,-6.816217,2.419185,2.788176,0.854235,-2.468642,6.849183,8.045869,-1.140931,0.638236,3.105621,-6.315625,9.186347,4.915882,8.749619,7.545119,-7.501641,9.865292,-6.149823,-5.655997,7.334592,2.963359,7.404987,6.493828,-5.506910,-8.571604,9.387038,0.288840,1.440196,9.404264,9.526442,6.641427,-6.797983,9.443513,-3.519998,-1.610373,-9.376528,-5.573254,-2.853854,-0.819262,-5.457773,-0.421601,1.824733,3.298232,6.389115,8.380614,0.132872,-3.044413,-1.221911,-8.900768,2.099796,-5.565387,-6.897928,7.357483,-1.768753,-2.361955,-0.094770,4.383261,-6.525755,-8.399165,2.607123,-1.223596,8.678696,-8.487388,-7.661381,-6.597927,4.493132,2.867797,0.747468,3.704056,-3.656618,3.240057,-2.674215,1.382887,6.708818,-6.826168,3.721561,1.873650,9.138593,-4.042306,-0.522519,7.052447,-3.666228,5.119199,-3.500701,3.406403,-8.748620,2.214655,2.742334,6.231023,5.364938,7.834033,-8.982276,3.860769,-5.060565,7.043502,-8.662894,3.393053,-4.443030,-9.410562,0.817020,7.467151,-9.143703,-4.076014,7.177166,-6.971259,-3.554746,-4.870532,-4.599761,-5.419643,0.961711,7.680146,3.821852,-3.496736,-9.620298,-9.463866,8.760401,8.595912,-2.717909,1.703965,-2.908848,-6.499771,-0.706056,2.035857,-2.304174,8.961213,-5.075639,8.859856,3.512174,5.113958,9.380567,7.355982,5.338181,-8.913725,2.415821,-5.878666,-8.613958,-6.001957,6.309209,7.546145,9.265102,5.594219,-1.105370,8.287911,1.046562,8.106919,-0.230567,0.158453,4.607502,-7.527590,6.976200,-6.823704,6.805714,-2.652949,7.565799,8.281161,7.730522,-7.837218,0.019054,3.359208,-5.514724,2.587670,-9.680087,-0.169411,2.880838,-3.189549,8.112462,-6.752405,8.066246,3.953543,-2.906473,9.872454,-8.494455,0.623060,-3.377357,-0.511760,8.813157,8.294922,-2.321864,-3.967453,-2.911107,-3.329882,1.972067,6.706678,-8.752806,8.110580,0.981704,-7.675879,-2.123776,4.562930,-2.069610,0.713277,-9.948249,7.737713,9.534126,-2.396389,-1.054121,0.296815,1.395180,-9.190796,-6.309762,-3.287231,6.174623,1.325443,7.528113,-6.232976,-9.253350,3.271227,-4.968170,0.786368,-1.976962,-0.743446,-0.086614,2.106467,-5.445953,8.528990,4.882138,-2.464747,-8.204317,4.473330,-5.564151,-4.936639,6.195261,7.884364,-2.160350,6.158424,-4.692005,-3.004487,-0.391391,-2.796340,0.196609,4.347187,-8.618914,0.135754,8.049203,-9.788660,-5.111179,-3.172418,-8.300190,-4.488951,-2.141379,-7.478514,1.572857,3.650863,-9.283807,-7.857605,-4.208117,-0.066674,9.397274,7.997369,-1.873489,-3.663002,-9.219549,4.249097,0.274931,-6.257508,4.298088,-0.415359,8.429387,-1.393590,-2.634761,1.301542,-9.542787,4.364498,-3.456181,-5.703711,-9.725163,3.194309,7.590648,3.953041,-5.976035,-6.098045,-1.738053,-5.606288,7.667237,8.737694,0.068398,-1.006637,7.147044,2.768952,-0.954223,-3.034426,6.952851,-1.587661,-9.069584,2.961086,4.976131,2.252047,-6.129858,4.033702,-8.414096,3.528521,1.661306,-6.202489,6.700393,-9.391501,7.957427,-5.144981,5.963820,-9.997940,3.639260,-8.156026,2.985370,-9.467724,-7.068617,0.972265,5.494448,-0.159330,-0.129411,3.538779,7.677835,-6.930587,-2.381546,4.156913,5.235915,-1.834550,4.757110,5.815839,-4.309922,-8.823688,5.281421,1.703190,4.361449,1.296510,-6.808148,4.619228,9.355803,2.013224,1.971142,0.639451,-9.363417,5.698029,-0.487245,-9.041812,-5.449912,9.222895,7.024953,-0.907653,1.003722,-3.681664,-8.181284,3.442456,3.306319,-7.677784,2.428759,7.360164,8.387189,7.673113,7.510478,-0.415006,-0.026533,8.664038,-7.460751,-9.965655,-4.671351,-0.488917,0.247202,-0.476803,-8.205915,4.373581,-0.458969,3.176688,5.755212,8.196637,-5.512970,-8.011159,-7.253727,-6.124592,-6.557742,-5.396846,-7.129312,5.882829,-1.038962,-4.844662,-9.296918,-1.842181,5.802260,-1.931465,-5.513492,9.078299,4.301985,9.274045,-3.921095,9.222328,5.131919,8.260573,-1.952775,0.960740,-6.716393,0.530690,8.705368,2.160329,2.221194,2.422260,6.004103,-3.448259,-7.151823,-0.312495,-6.696485,5.563447,0.875945,2.574142,-2.563034,-1.148568,-9.250711,9.593412,9.351758,-4.833757,3.050603,-0.341799,4.914535,6.161386,2.211543,-9.340493,9.623697,0.483805,-7.788638,-9.751972,7.847427,0.494463,-6.763899,4.608384,7.816288,-6.994097,5.092312,3.355601,-3.350011,-6.527579,3.186253,9.277225,7.567141,5.121102,4.898095,-4.512045,4.847391,-6.140537,-7.466300,9.152318,-8.916923,-8.882783,3.507460,-5.936698,5.622846,7.532302,0.452740,6.740111,5.038325,-4.023125,0.720951,8.874789,-7.572037,-9.326167,1.418447,-0.300944,-1.429416,8.188650,-2.182696,9.637958,-9.949127,-0.982443,0.492738,-9.237467,5.899100,2.139980,3.762931,7.575387,-7.238685,5.179617,8.335068,7.366038,-2.711698,4.561474,-5.607170,-7.217763,7.027776,-1.984884,5.936951,8.725159,9.392948,8.829136,1.993982,-0.142600,-6.701841,7.705524,4.144712,2.055077,7.459258,4.346700,9.264206,4.517730,-0.095924,1.090207,-9.220571,-8.059736,3.206369,9.710917,-9.668580,1.050003,7.929698,9.165187,6.342448,-2.481099,9.260197,9.819480,-8.457606,-8.330486,-0.656261,-3.706760,-5.919046,5.484843,-2.579208,3.133037,-0.511408,0.822591,5.403718,0.616094,-3.221852,-5.595721,8.649196,-0.943978,-5.817981,-1.820652,3.414937,6.979434,-5.106379,4.520526,7.105642,3.608313,0.960626,8.839779,8.930635,-4.206685,8.712817,-6.266891,-5.855265,-4.678359,7.772343,5.931619,0.217710,-6.824414,6.763016,-4.017440,-2.310886,2.193182,-0.391432,0.613290,0.373035,4.711133,-0.532898,2.042062,3.445423,8.414668,-8.807985,9.000131,2.259595,9.685732,5.319883,-1.630070,-5.390053,-4.828843,-4.605878,-9.782792,0.943285,-3.797357,0.439606,-9.939996,4.888120,3.013056,-9.742502,9.705526,9.057586,4.841268,-4.280640,-0.029275,9.954711,-7.000170,-1.096463,2.135988,-6.186429,7.420708,7.536955,-3.221662,9.492154,-4.776739,-3.497560,-8.924227,1.727518,3.237527,7.973936,-1.290637,-2.250653,-9.945614,6.262286,-4.941248,-5.279420,-7.121402,0.799311,-2.675273,-4.251338,-0.517629,3.882572,-9.838255,-0.759916,-7.639113,-1.235703,-8.183898,9.948693,6.428190,-6.064168,9.534415,-7.878678,-9.872014,-6.894026,-7.282526,-2.198646,-2.415903,6.305667,-0.485620,-1.054684,8.994229,1.415755,9.243377,0.911121,9.874972,-4.376074,-9.702759,-9.484037,-5.948668,6.264218,-5.891951,8.730643,9.595901,5.605929,5.707215,5.972667,-3.703709,-6.172044,4.147883,9.290859,-7.026487,7.449972,-0.194836,5.247444,5.505009,4.317415,-2.891576,-7.891535,-2.592140,2.753797,9.688567,-0.632874,7.036572,0.156064,-6.611957,-2.363789,-0.192716,9.005055,4.175171,9.255294,-6.971405,-9.360435,3.820867,-5.925028,0.416396,0.045489,-8.373911,-0.255506,5.106578,3.065339,-4.020887,-2.941186,5.956963,1.598872,-4.453996,1.687286,-4.204369,-3.081084,-5.425227,8.738161,-9.160463,-7.480258,-2.054072,8.591140,-3.597128,-9.301361,-2.130897,8.312234,-5.039470,-1.538604,1.420031,4.597652,-1.811237,-2.910207,3.321597,-6.725087,-0.551058,-6.539155,7.030759,-0.278316,-8.361153,9.782329,9.291660,5.551710,8.269684,9.133190,0.528406,-0.126091,-6.647029,9.512923,-3.653788,8.265409,-1.526232,-4.483366,0.919999,-6.894299,0.386043,-7.458502,-9.829970,-0.781724,-3.761866,9.502673,3.791123,0.618187,-1.371653,5.833971,-7.829927,1.815643,-6.316080,-0.577930,9.343087,-7.967912,-8.539360,6.461903,-9.042697,6.189337,5.421681,-6.113483,-0.474106,-6.158641,-5.128789,-2.627191,0.106534,-3.585476,5.094844,4.915675,4.599679,6.949010,0.738899,-1.850887,-5.282540,-6.566596,-9.640568,-3.563227,-9.698310,8.625407,-1.276905,7.761202,-1.651629,-2.048243,-9.504574,-3.651680,2.685126,-8.636339,-3.423850,-3.101082,-3.552268,7.503527,1.458879,1.603038,-8.694786,1.797131,-6.708466,-7.166391,-9.718442,-2.299380,8.599077,-4.354394,-6.952230,5.976615,0.147506,-1.540775,-3.847634,3.746285,8.875361,0.138611,3.464154,4.452206,9.900736,-3.456496,-9.550455,-6.334951,7.768015,-9.506418,7.970203,7.030957,8.168265,-5.363686,6.248476,7.859063,1.244003,-2.973989,9.370385,-6.716756,-9.332746,-7.219738,-4.975764,-8.192406,6.592006,3.757383,-8.853324,-8.288098,-9.372674,2.860676,-9.402906,9.907469,7.321307,-8.223585,8.069021,6.996174,-8.768518,-7.112091,-6.897225,-2.934912,0.994240,4.990842,-1.550655,1.356744,-4.946832,-8.910323,7.262174,-5.182745,-9.215529,-5.713533,-9.446684,2.510445,5.389670,5.983311,-4.535660,-7.999428,1.354438,8.660888,4.395069,4.213048,9.192452,-3.186351,3.524320,6.461972,9.105255,7.138918,9.824353,-0.679273,-5.775272,-1.688962,-8.552361,-7.498838,-7.576820,1.832608,-8.738672,-5.461866,9.428736,-6.912360,4.479950,-5.040427,3.621394,-8.764463,-4.540890,-3.374521,-3.856090,2.869917,6.212678,6.635180,0.415770,1.078628,6.440361,5.850654,-6.832590,0.636701,7.951399,-2.199484,8.852545,7.901941,-9.397551,-9.386077,-9.422687,4.287736,0.378160,-8.188964,6.301737,-9.062179,5.246949,8.742929,6.508173,0.092918,-9.139620,-3.182095,1.616036,4.597991,5.531358,-7.118072,8.718851,1.702365,-9.770900,7.340709,4.075819,-6.181438,3.745005,-8.476057,-1.422698,7.027027,2.693763,-6.935097,5.576025,-5.292927,3.829276,4.191646,-6.774408,0.139310,-6.192493,2.984748,-5.311505,-3.019300,6.338411,4.811854,-6.860437,5.292448,-5.606250,-1.109949,5.892147,-1.054152,-9.479977,-5.196544,-0.716356,5.968868,-8.122233,-5.130125,8.471284,7.968543,6.735833,-6.997827,9.209962,4.795189,-2.437169,-5.153498,7.401035,-9.768519,-0.094136,-1.250052,4.021259,4.539822,-5.728871,-8.416689,-4.757298,7.878531,-1.573269,3.172609,0.673362,8.599581,1.074004,3.295296,-6.673069,-3.034601,-0.409156,7.524293,-1.217493,-0.394304,-6.995851,-8.604196,-3.092769,-4.623828,-5.814667,-8.147924,3.528026,9.812460,6.251242,8.474131,-0.660813,-5.234088,-2.356697,-9.415780,-7.955066,-1.280560,-6.281287,5.583152,-3.795728,9.737334,-8.534487,-9.772820,-5.092850,-7.386704,-6.219806,4.474726,3.917802,-5.079443,-6.264398,-7.397528,8.153083,4.377646,-7.752900,-6.354871,8.061571,6.999289,3.201686,5.383687,9.944053,-4.659870,1.197066,-7.749286,1.722388,-6.658674,-4.200106,3.266412,6.677444,4.546435,8.207085,-2.250288,0.351566,-7.053580,4.037845,-6.660402,8.594086,8.025134,4.453823,2.813244,-2.803613,-4.968060,6.945146,-0.610773,-5.318815,-0.977083,8.972943,-7.397343,-5.227104,1.504391,-4.640674,-7.710769,-3.074956,-1.618317,1.890972,2.701491,-3.710554,-6.759577,7.456323,-3.541873,-2.551068,7.667406,-4.060768,8.232933,5.042779,-3.249553,-1.458344,9.257595,-4.827651,-2.885519,-3.501225,5.152704,7.487623,-2.925027,-0.935451,7.395291,9.399794,-6.452165,-2.191868,-9.517936,2.492654,-6.963836,1.912499,-2.311066,5.501497,7.733355,-1.868787,-8.839824,7.664554,-9.669694,-2.059431,-6.867355,8.148650,9.249836,-4.701830,7.876828,-8.392586,-1.078985,6.802775,-8.802412,2.649878,-7.952895,-6.671247,-2.539214,9.861741,-3.037277,-7.295627,7.312730,9.523242,7.380961,-9.688749,2.923472,-3.632653,-8.190670,9.437929,8.043596,4.228922,-3.192276,-8.544690,5.757448,-0.785139,8.036861,-2.786235,-2.741772,1.214394,-1.141739,-0.513994,1.569456,0.399587,-2.642165,-6.123104,4.546163,-7.512427,4.634224,-9.398771,-6.142768,-6.045780,0.314093,6.264797,6.246889,-9.401919,-2.390957,5.345843,9.077064,-6.711963,9.888890,4.280192,-6.652485,3.227545,1.666388,-6.899417,-2.939684,5.366456,7.209588,4.354556,-2.611011,-6.418047,-7.069353,-6.971883,-8.033033,3.933986,3.006356,4.384568,-4.034663,8.220077,-9.058276,-2.724491,5.223421,3.872329,-2.051763,8.642960,-7.221259,6.960921,0.106949,-6.149603,-6.419154,-3.813379,6.764144,0.903917,-7.361240,6.235584,-4.590542,4.176650,5.361918,8.705066,2.691928,-3.188867,-4.175622,-9.616914,1.605701,-3.476337,-4.954169,5.359946,-6.161259,-0.022771,-0.356842,8.366902,3.903174,-1.728519,3.565949,-0.286202,-0.655052,-5.229526,-4.654299,-0.816103,-5.628990,-0.025218,7.072700,-3.778412,-2.032616,-7.041169,-7.435320,-9.192754,-2.871673,-3.368023,-2.066835,-8.470936,-0.739232,7.097476,0.488565,-1.965138,-2.976586,7.573441,6.860343,7.238091,-3.738435,-3.257559,-1.664133,-7.236727,7.742324,-5.939120,8.459511,-4.099335,-6.396836,-6.909941,8.219343,2.537265,5.930780,-9.017280,-0.201274,6.306320,6.157638,3.138148,-0.630078,7.742128,-5.131537,2.943898,-4.662423,1.765552,1.015006,1.703995,4.782009,4.592359,9.961845,1.602474,5.383947,-7.533140,-9.468470,-9.517862,7.062594,-7.967809,-0.989038,4.879194,4.198889,0.816546,-3.970082,7.930906,3.039439,7.815858,-8.846097,-8.047585,3.389467,6.216708,-9.305035,-5.439440,6.290702,6.311569,-0.376289,-9.252353,-4.866675,-0.686618,-5.227731,-8.305223,-1.980135,-6.201445,-1.798652,9.443284,4.579119,0.201067,-8.122267,1.739005,-3.355953,3.315147,2.736300,-8.757520,2.068710,3.402587,-2.534527,1.235253,0.726382,-8.490492,-8.966859,4.782790,-2.002089,6.628893,-1.839175,-8.802852,-3.854898,8.832132,7.134025,7.242148,-3.948807,3.989964,-3.980327,-6.740638,-2.468593,3.915528,-6.207458,0.309951,5.224118,6.876133,3.750785,-0.853686,-0.882741,-0.771868,9.225898,-3.364214,-7.084275,-5.274405,3.208322,-6.154763,0.787758,-6.721406,-1.223795,-1.586917,2.288204,4.004454,4.045969,-7.038677,4.088388,-6.375068,-6.704389,-5.928607,-5.799445,1.880727,5.234864,6.663016,6.389613,-2.144671,2.270524,6.967453,7.176195,-0.710945,-7.824157,-6.973313,8.929107,-4.061056,2.336833,-1.689099,9.545603,0.948603,8.133839,-6.882754,7.454065,-2.780015,6.746960,0.899612,-3.100617,-5.680918,-0.085451,-4.339823,5.438022,4.302069,5.038844,-2.883191,3.308801,-3.548519,-9.420864,2.252596,5.329485,2.293165,7.918767,6.333279,-1.984514,-4.887912,-2.802052,-0.349302,2.285378,2.708147,3.006255,8.910703,8.512141,8.552724,-7.028874,3.822095,-9.638073,8.037153,9.103908,1.444463,-5.424585,-9.155145,-7.956508,-4.563574,-3.974158,-7.618015,3.227727,-4.118415,7.220289,-7.644157,-4.837685,-2.033175,-3.997891,-6.117091,-9.683443,7.395115,1.943117,-3.267081,-1.764393,0.036369,-2.591964,-8.986960,-0.193205,-2.561321,-6.642048,0.804023,1.781505,4.676969,5.260823,0.395526,9.720354,-4.686033,7.262516,-2.302164,2.983918,-7.266768,-4.233468,6.820646,-5.065135,-6.365650,6.385990,2.737428,3.180628,-9.026400,1.943971,2.636242,4.331328,1.309677,-2.159460,-1.675250,-0.369931,1.372808,-8.895947,5.814884,-5.666837,6.700922,6.244814,-8.088959,-6.113218,-1.948506,3.188386,-5.047256,2.555926,0.224032,-5.534991,-7.224518,-3.484048,9.999887,-0.363009,2.089156,-6.517686,-2.194093,-4.865323,2.090926,-1.371949,-3.824606,6.150530,-5.269267,-8.232859,-0.115222,9.390214,0.594614,-5.921444,-9.705724,4.389312,0.160997,5.999986,-4.438606,0.863807,-2.021764,5.202170,8.215152,7.443307,4.490159,2.461151,-2.772540,-8.650833,-3.276843,-7.744610,-8.823116,9.956106,-5.006484,4.127120,-7.023329,-6.623533,-8.815696,1.554368,-9.289716,8.124232,-6.998933,6.668830,-7.078741,5.713290,-5.840838,8.803066,6.103383,-3.735795,0.564504,-5.448628,-4.547757,-2.493674,-6.876024,-8.527642,2.407931,6.264072,-0.867868,-0.593752,-2.920804,-5.439811,-1.871170,5.114498,-2.932290,3.445048,-5.082402,1.858619,-0.094419,7.749731,4.265737,0.467645,-0.171952,-7.735177,7.855404,7.574905,-2.489363,-7.494958,9.666523,7.039904,3.263141,-1.073673,-3.015979,3.884219,3.636687,-5.210095,-2.429702,0.281415,-6.867237,-2.649818,8.562655,5.748706,6.670111,-7.205616,-2.426626,-0.292851,-5.959034,9.788961,-7.016844,9.272849,5.404893,-2.800929,3.843159,-9.817756,8.933898,9.013888,-0.859847,1.325979,-2.778877,8.155710,9.115702,-8.961246,-2.016408,-9.663737,-1.480589,-5.740654,3.903527,-5.397772,4.023578,-8.833895,0.165014,2.564160,2.364662,3.341278,-1.437226,-7.020318,-0.058967,1.890597,-8.472157,5.185888,0.768115,-1.006373,-9.050064,2.687854,-0.310620,-5.536095,9.529599,9.735910,-9.278549,-6.347788,4.261554,-9.516350,4.799608,-0.003969,-0.541390,5.700644,-5.472255,-2.519181,-4.751370,-5.437788,-3.292333,-4.177735,-7.984278,-5.803803,-2.615972,6.872057,7.929659,1.029258,9.762134,-0.692060,5.023279,2.909271,4.569524,-3.853479,8.792042,-0.810358,3.303423,7.037608,1.227049,7.540142,-5.341590,8.873059,-8.468717,-3.366235,0.924847,0.764393,-4.936102,-4.876698,8.391202,2.538764,9.742692,-7.826837,-5.743160,-5.541181,2.739303,8.362022,-6.145366,5.237190,0.895167,-6.754325,-0.455674,5.111458,-5.632863,-8.561300,-9.115146,-1.536695,-4.668418,1.788123,-1.046795,-2.779959,4.405801,5.520172,-4.029813,-8.892796,-1.100041,-0.714812,1.620678,-9.545818,2.197687,7.232370,-8.354091,6.215505,-3.414889,-6.577281,-8.740802,-7.063088,-5.680437,0.962557,-7.125809,5.536368,-8.775681,7.215037,0.614932,3.157669,-8.787222,7.709999,-7.156712,0.119658,8.553016,0.510835,-6.086465,0.933201,0.377805,-8.592439,-3.027473,9.630260,7.922073,-4.845738,-0.306438,-5.978759,-4.489561,6.516555,-4.635051,7.066557,-0.521944,-8.111582,-6.542244,-5.711831,-2.551405,3.795104,-3.215903,-3.585069,2.440712,-2.784985,-8.497535,3.840516,-5.386378,-5.636431,8.583057,-3.321650,-1.512952,0.884578,-7.817400,7.133455,-5.303696,-1.313281,6.515458,-8.921341,8.054584,-2.627867,0.408802,8.016927,-7.185889,8.771812,6.277785,-5.045911,-2.926481,5.440043,6.344645,8.087329,7.208981,-0.698731,-2.726196,-2.903227,0.880991,-9.433099,-9.487585,1.359017,0.709587], dtype = "float64")#candidate|4532|(2048,)|const|float64
var_4533 = relay.var("var_4533", dtype = "float64", shape = (325, 1))#candidate|4533|(325, 1)|var|float64
call_4530 = relay.TupleGetItem(func_3767_call(relay.reshape(const_4531.astype('float32'), [14, 14, 2]), relay.reshape(const_4532.astype('float64'), [128, 16]), relay.reshape(var_4533.astype('float64'), [325,]), ), 2)
call_4534 = relay.TupleGetItem(func_3772_call(relay.reshape(const_4531.astype('float32'), [14, 14, 2]), relay.reshape(const_4532.astype('float64'), [128, 16]), relay.reshape(var_4533.astype('float64'), [325,]), ), 2)
output = relay.Tuple([call_4492,call_4530,const_4531,const_4532,var_4533,])
output2 = relay.Tuple([call_4493,call_4534,const_4531,const_4532,var_4533,])
func_4535 = relay.Function([var_4533,], output)
mod['func_4535'] = func_4535
mod = relay.transform.InferType()(mod)
var_4536 = relay.var("var_4536", dtype = "float64", shape = (325, 1))#candidate|4536|(325, 1)|var|float64
output = func_4535(var_4536)
func_4537 = relay.Function([var_4536], output)
mutated_mod['func_4537'] = func_4537
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4554 = relay.var("var_4554", dtype = "int16", shape = (7, 5, 12))#candidate|4554|(7, 5, 12)|var|int16
var_4555 = relay.var("var_4555", dtype = "int16", shape = (7, 5, 12))#candidate|4555|(7, 5, 12)|var|int16
bop_4556 = relay.greater(var_4554.astype('bool'), relay.reshape(var_4555.astype('bool'), relay.shape_of(var_4554))) # shape=(7, 5, 12)
output = relay.Tuple([bop_4556,])
output2 = relay.Tuple([bop_4556,])
func_4559 = relay.Function([var_4554,var_4555,], output)
mod['func_4559'] = func_4559
mod = relay.transform.InferType()(mod)
var_4560 = relay.var("var_4560", dtype = "int16", shape = (7, 5, 12))#candidate|4560|(7, 5, 12)|var|int16
var_4561 = relay.var("var_4561", dtype = "int16", shape = (7, 5, 12))#candidate|4561|(7, 5, 12)|var|int16
output = func_4559(var_4560,var_4561,)
func_4562 = relay.Function([var_4560,var_4561,], output)
mutated_mod['func_4562'] = func_4562
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4265_call = mod.get_global_var('func_4265')
func_4267_call = mutated_mod.get_global_var('func_4267')
call_4578 = func_4265_call()
call_4579 = func_4265_call()
const_4597 = relay.const([[[5.231361,9.477472],[1.386135,-6.380120],[1.012527,-9.244393],[-3.470835,-8.839425],[1.027896,-8.363120],[-1.123105,-5.884466],[3.352733,5.044699],[-5.036318,-7.239084],[-6.815660,6.708529],[1.318151,-4.584955],[-6.100200,-9.133112],[-1.139595,-9.635489],[5.850939,-3.511888],[5.806152,-4.561896]],[[5.911792,6.621298],[0.900216,0.371200],[9.463831,3.167170],[-3.415727,9.744648],[-4.365064,-4.765350],[-0.883645,-7.662827],[7.191635,-6.940751],[3.932523,0.677760],[-8.254733,-7.531710],[7.324307,-3.785266],[-1.729580,7.726348],[5.676707,-1.053554],[6.288761,-4.171354],[9.791255,9.586297]],[[8.908821,6.224175],[1.212063,3.871652],[3.239432,8.638834],[-7.765863,-7.926887],[2.801591,1.246464],[-0.494212,-8.094560],[-6.003092,-4.086416],[2.633190,0.924027],[8.023306,1.119107],[3.073603,-6.511620],[5.129946,4.274604],[5.920085,-2.388556],[-9.124524,2.851193],[9.667883,-7.473224]],[[-5.526711,-1.636446],[-6.031143,4.742183],[-5.764349,-6.202288],[0.039592,6.646688],[-1.909443,-5.610032],[-1.724971,-9.200618],[6.496424,0.533784],[9.454313,-4.305592],[-5.860565,4.756771],[-3.426601,-0.637873],[-4.133218,7.835329],[-7.375070,-7.714405],[2.513432,-0.140349],[2.315305,-9.582056]],[[6.300683,-0.108784],[6.732070,-7.279350],[7.092573,3.700620],[-1.485839,-4.626423],[-0.363107,3.509408],[-7.236838,-9.163874],[-6.644840,1.755999],[0.398708,-2.710773],[-5.595565,-5.881498],[-1.567702,2.960129],[-9.701697,8.524419],[1.355885,2.964961],[4.256358,-2.865317],[-9.262831,-6.890957]],[[2.393581,8.567265],[4.150809,8.075912],[-6.750536,-1.793336],[0.213171,-5.774672],[-8.804119,-8.696159],[2.058306,2.470074],[-5.523486,5.840205],[-6.847379,-0.731226],[0.071689,5.847058],[8.865298,-9.127783],[-7.425725,-3.363743],[5.546641,4.973164],[2.389427,2.852606],[-7.490854,8.268690]],[[-6.914103,8.177586],[-8.168004,-0.867923],[1.476158,5.875213],[0.317175,-7.788203],[-5.262595,7.127792],[4.216066,0.568619],[-1.077901,-6.809660],[4.479415,-5.634788],[-2.997022,7.465524],[5.250254,-5.906699],[1.852641,1.996624],[-1.749743,-5.611895],[7.035105,-9.130349],[-7.072286,7.873184]],[[-0.812664,1.082851],[6.669361,8.728515],[-2.989716,6.328177],[-0.180495,-9.429412],[0.914871,-5.328088],[-2.841046,-7.278769],[-1.682628,-1.951070],[-9.073584,-7.333543],[9.761492,-0.821699],[1.759458,-4.198670],[-4.247486,5.045039],[-2.591652,-2.330315],[4.399212,5.627635],[1.262467,3.204080]],[[-8.541093,1.674691],[2.699465,0.804510],[1.706028,-7.395428],[2.120989,-3.435798],[-2.480768,-7.885902],[-1.407178,-6.418361],[-6.656309,3.442020],[4.887315,5.376457],[-7.255003,-4.614225],[-5.913887,7.023579],[8.344900,-2.347170],[-6.154817,3.475445],[-2.301458,9.291354],[-1.010738,1.695266]],[[0.897105,9.537437],[8.169761,-9.837763],[5.817515,-0.718687],[-9.453631,2.037295],[7.978912,-2.112136],[5.427269,-7.964054],[-3.464428,-5.578231],[1.864475,9.255094],[-7.898678,3.318177],[3.733840,7.621908],[-7.774371,6.328864],[8.627679,-2.332578],[-9.710149,6.095012],[-0.407324,-0.573830]],[[4.209932,-0.452386],[-9.762342,-9.971124],[-2.019407,-3.811701],[8.594953,-8.573440],[3.620815,5.247991],[-6.434012,6.370933],[-1.092574,-9.942733],[-2.781709,6.638935],[9.728122,-8.845782],[9.417330,-2.154231],[-0.797828,-6.271378],[6.462636,6.455303],[-0.172374,9.169663],[7.016825,2.191811]],[[5.064654,-4.201654],[-0.216583,-3.446973],[3.257264,8.548284],[4.687868,-0.692921],[4.568298,4.924700],[8.573642,3.264298],[1.223894,7.540460],[-4.693111,5.971524],[-8.232674,7.056011],[0.196893,9.045418],[0.271065,-9.327432],[-7.366791,-5.193481],[5.950439,-8.976005],[6.277372,-3.367550]],[[0.504113,6.263375],[-9.399811,3.211723],[-4.988975,-3.359829],[-9.880212,0.641959],[-2.790962,-6.460564],[-1.741800,-3.385133],[9.602829,1.672559],[3.397215,2.573130],[9.182289,-6.054434],[1.548098,-2.588141],[-4.878050,-0.888803],[-3.424337,-6.771320],[8.457189,-0.011248],[-9.799699,2.368341]],[[2.774947,-5.927036],[-5.261287,4.376700],[-5.797559,6.168357],[-2.500174,8.281493],[-4.954491,0.731579],[7.623791,-0.976565],[2.872580,8.934591],[7.707786,5.423022],[1.621182,7.373651],[4.270639,8.956937],[-3.216748,-2.477397],[-3.361219,-9.908932],[7.602766,-1.739009],[7.938877,7.961144]]], dtype = "float32")#candidate|4597|(14, 14, 2)|const|float32
bop_4598 = relay.bitwise_and(call_4578.astype('int8'), relay.reshape(const_4597.astype('int8'), relay.shape_of(call_4578))) # shape=(14, 14, 2)
bop_4601 = relay.bitwise_and(call_4579.astype('int8'), relay.reshape(const_4597.astype('int8'), relay.shape_of(call_4579))) # shape=(14, 14, 2)
output = relay.Tuple([bop_4598,])
output2 = relay.Tuple([bop_4601,])
func_4610 = relay.Function([], output)
mod['func_4610'] = func_4610
mod = relay.transform.InferType()(mod)
mutated_mod['func_4610'] = func_4610
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4610_call = mutated_mod.get_global_var('func_4610')
call_4611 = func_4610_call()
output = call_4611
func_4612 = relay.Function([], output)
mutated_mod['func_4612'] = func_4612
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4230_call = mod.get_global_var('func_4230')
func_4232_call = mutated_mod.get_global_var('func_4232')
call_4639 = func_4230_call()
call_4640 = func_4230_call()
output = relay.Tuple([call_4639,])
output2 = relay.Tuple([call_4640,])
func_4652 = relay.Function([], output)
mod['func_4652'] = func_4652
mod = relay.transform.InferType()(mod)
output = func_4652()
func_4653 = relay.Function([], output)
mutated_mod['func_4653'] = func_4653
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4670 = relay.var("var_4670", dtype = "float32", shape = (14, 7, 13))#candidate|4670|(14, 7, 13)|var|float32
uop_4671 = relay.sqrt(var_4670.astype('float32')) # shape=(14, 7, 13)
uop_4674 = relay.sin(var_4670.astype('float32')) # shape=(14, 7, 13)
uop_4689 = relay.erf(uop_4674.astype('float64')) # shape=(14, 7, 13)
bop_4693 = relay.bitwise_or(uop_4689.astype('int64'), relay.reshape(var_4670.astype('int64'), relay.shape_of(uop_4689))) # shape=(14, 7, 13)
func_4462_call = mod.get_global_var('func_4462')
func_4464_call = mutated_mod.get_global_var('func_4464')
call_4696 = func_4462_call()
call_4697 = func_4462_call()
output = relay.Tuple([uop_4671,bop_4693,call_4696,])
output2 = relay.Tuple([uop_4671,bop_4693,call_4697,])
func_4704 = relay.Function([var_4670,], output)
mod['func_4704'] = func_4704
mod = relay.transform.InferType()(mod)
var_4705 = relay.var("var_4705", dtype = "float32", shape = (14, 7, 13))#candidate|4705|(14, 7, 13)|var|float32
output = func_4704(var_4705)
func_4706 = relay.Function([var_4705], output)
mutated_mod['func_4706'] = func_4706
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3870_call = mod.get_global_var('func_3870')
func_3872_call = mutated_mod.get_global_var('func_3872')
call_4720 = func_3870_call()
call_4721 = func_3870_call()
var_4726 = relay.var("var_4726", dtype = "float32", shape = (14, 14, 2))#candidate|4726|(14, 14, 2)|var|float32
bop_4727 = relay.floor_mod(call_4720.astype('float64'), relay.reshape(var_4726.astype('float64'), relay.shape_of(call_4720))) # shape=(14, 14, 2)
bop_4730 = relay.floor_mod(call_4721.astype('float64'), relay.reshape(var_4726.astype('float64'), relay.shape_of(call_4721))) # shape=(14, 14, 2)
func_4186_call = mod.get_global_var('func_4186')
func_4188_call = mutated_mod.get_global_var('func_4188')
call_4732 = relay.TupleGetItem(func_4186_call(), 0)
call_4733 = relay.TupleGetItem(func_4188_call(), 0)
bop_4735 = relay.less(call_4732.astype('bool'), relay.reshape(var_4726.astype('bool'), relay.shape_of(call_4732))) # shape=(14, 14, 2)
bop_4738 = relay.less(call_4733.astype('bool'), relay.reshape(var_4726.astype('bool'), relay.shape_of(call_4733))) # shape=(14, 14, 2)
func_2779_call = mod.get_global_var('func_2779')
func_2786_call = mutated_mod.get_global_var('func_2786')
var_4740 = relay.var("var_4740", dtype = "int16", shape = (825,))#candidate|4740|(825,)|var|int16
const_4741 = relay.const([-6.731278,2.149986,5.098423,6.292453,-8.738104,-7.074824,-6.651715,-5.128816,2.274127,-4.983733,6.732132,-4.486093,0.679754,7.101978,8.377088,-6.724112,8.451464,-7.728142,-4.384536,0.276689,-0.072170,5.337367,9.851965,-0.875778,6.700470,-2.130409,8.243254,-1.325816,-5.816345,6.556211,6.242406,-4.206570,3.664063,4.880895,-7.050825,-4.391481,-1.474330,6.049279,-9.123451,1.842585,-1.379093,-1.928693,-6.780941,5.695893,-6.784235,-4.667052,-6.370659,0.205778,3.511281,-5.349423,8.075889,-8.080623,-2.629367,8.700975,1.869771,-0.753377,-4.843144,4.670194,2.901369,-9.933057,4.691483,0.241103,6.141200,-3.895254,6.574370,-5.239053,1.296643,8.907953,-7.041846,-0.699929,5.820486,-7.209218,-2.902138,-4.306024,-7.813498,9.034120,9.356347,7.881852,8.258044,2.599572,-0.904458,1.626022,6.887185,9.063980,-7.549554,4.565731,-8.361448,7.967212,-5.897827,-5.254934,-9.795654,-6.742629,-1.083786,-7.119026,2.887175,7.506979,2.501744,-2.641359,4.816928,3.090006,6.177919,9.822920,-1.460168,-3.433324,-8.430589,0.762029,0.233117,-5.192597,2.284078,9.097860,-0.390641,7.999639,-4.236292,-7.687050,0.852374,1.556036,8.880681,0.359087,2.607302,-6.590329,-0.899088,2.215849,1.560043,-1.852647,-1.050072,-0.184208,-7.334297,3.876475,5.894159,-5.433067,7.057338,1.285556,-5.325359,-8.754434,9.232325,8.279609,-6.958196,8.056536,-7.980853,-3.864265,5.004064,2.684488,-3.173494,-0.383221,7.278633,1.219596,-2.210362,4.471974,5.507777,8.060752,-3.210391,-0.691285,5.684688,-1.308089,0.417601,-6.405283,5.865285,-2.039227,-1.683647,5.513800,8.844930,-4.136480,-3.831240,-1.730811,3.614914,-1.884224,-4.593998,-8.700233,-4.672846,5.889045,-5.260107,-1.320591,-2.349676,7.388645,-0.254151,-1.947461,5.072774,3.425300,-8.455368,2.870932], dtype = "float32")#candidate|4741|(180,)|const|float32
const_4742 = relay.const([0.493269,2.546044,-5.966212,7.741412,9.689663,9.098292,4.433892,-5.236149,9.290769,6.795076,-3.669997,-6.469939,5.873668,-6.314485,5.577006,-5.684663,-8.811758,-5.804567,7.948013,6.569571,-8.814032,-6.496018,2.022532,3.922830,1.927438,-3.616215,3.207124,-0.002967,-1.333734,-5.824886,-6.861453,-8.875754,-2.333375,8.025787,7.399597,-1.343286,2.894322,6.069981,-9.695149,-7.438738,0.937563,9.334574,6.065195,8.381342,-7.202768,-3.111933,5.383983,0.334802,9.982638,8.123248,7.425041,-7.855888,-8.272673,-1.958002,-7.411288,-6.050633,0.005350,9.459371,4.702740,-6.521901,-6.525727,-3.588511,-6.338520,5.437438,-7.483393,-9.082184,3.100613,9.484007,-9.149606,-4.893432,-0.898659,-5.173852,-4.035113,-7.189457,0.086211,4.403677,0.294948,8.937724,-0.494556,-1.886489,2.683887,-5.088818,7.976964,-1.575089,9.887322,7.378999,4.062297,7.970942,3.657982,4.480025,6.080533,-1.373964,-8.893807,-9.419328,-1.339411,7.037391,0.666553,-9.085164,8.279385,-0.343974,2.735426,-9.766536,6.535226,-9.308924,-0.762362,0.347446,9.690453,-2.824540,0.563050,-6.423774,5.472478,6.323515,2.957947,-1.290168,-5.124151,-0.506829,-4.437335,-8.906300,-1.104512,-7.410769,-4.918042,-6.553254,6.851717,6.876439,0.121101,-6.616513,-5.556481,4.054930,-0.031624,-2.652717,-4.786240,-3.039052,-8.826616,6.205728,-4.672016,8.581445,5.063234,-7.410568,-0.553478,-1.069122,0.916278,9.261373,-9.942366,4.262059,-4.072672,-8.204209,-2.094749,4.897859,8.940079,-3.416584,-6.519142,6.127222,9.821197,-2.634506,2.076879,-8.308910,6.210325,-8.231692,-9.404281,-5.488151,-6.625797,-8.866699,-9.455644,6.935222,-8.695918,8.372519,9.625804,-0.129921,-3.257529,-4.126567,2.757010,-0.137294,-2.928849,8.807901,-2.348427,4.522726,-0.272310,-2.704456,5.356217,-1.896973,5.312772,-5.937181,-8.352766,1.708924,7.088375,9.349590,-3.608947,-5.827900,-1.411272,-4.100387,7.183838,5.651914,8.281247,-3.018630,5.077135,-9.891225,-6.641600,2.368234,7.908846,2.537420,2.463566,-7.109695,-2.814295,-0.568279,-2.041120,-7.856458,3.393767,9.374340,7.850064,5.052561,-6.300815,6.669748,3.697335,-6.813563,6.073782,5.954791,-7.638576,6.277614,-0.001863,5.622396,-1.994289,2.268766,-8.454109,-5.866078,-4.921207,-9.390097,0.448214,-9.164074,6.485012,6.730351,-5.491250,1.476677,-3.524238,-1.571216,-8.974502,9.920079,2.249751,-6.701283,5.537109,7.070545,5.455671,5.618758,3.201967,6.563332,-9.953967,5.926497,3.406262,3.342484,1.816369,1.516489,8.087022,-9.713059,3.843893,-0.304461,1.163626,-0.663889,-6.478771,3.603459,9.979703,-2.584777,-5.426882,3.474248,-8.007712,0.572406,-1.807337,0.220822,-0.578479,-2.897811,1.138404,-4.259282,-8.755541,7.997869,-5.445378,-2.507126,5.420910,2.885878,-8.032852,5.481750,3.984518,9.099798,1.999969,-4.092714,-6.048733,2.253975,8.807971,3.432886,4.864962,-4.994758,7.036039,0.947994,9.383142,-2.646916,7.376950,-8.939272,9.966391,-1.324395,1.191453,-1.983368,5.633730,4.298494,9.786867,-8.012531,2.374429,1.374305,2.905521,5.956591,4.255472,5.572161,5.656238,-5.705319,6.217099,-4.343074,6.505225,-8.377453,-3.705384,6.747121,-2.429959,-9.840584,5.375000,1.685241,2.043746,7.784143,-4.212700,1.216293,1.758479], dtype = "float64")#candidate|4742|(325,)|const|float64
var_4743 = relay.var("var_4743", dtype = "bool", shape = (550,))#candidate|4743|(550,)|var|bool
call_4739 = relay.TupleGetItem(func_2779_call(relay.reshape(var_4740.astype('int16'), [11, 15, 5]), relay.reshape(var_4740.astype('int16'), [11, 15, 5]), relay.reshape(const_4741.astype('float32'), [180, 1]), relay.reshape(const_4742.astype('float64'), [13, 25]), relay.reshape(var_4743.astype('bool'), [550,]), ), 0)
call_4744 = relay.TupleGetItem(func_2786_call(relay.reshape(var_4740.astype('int16'), [11, 15, 5]), relay.reshape(var_4740.astype('int16'), [11, 15, 5]), relay.reshape(const_4741.astype('float32'), [180, 1]), relay.reshape(const_4742.astype('float64'), [13, 25]), relay.reshape(var_4743.astype('bool'), [550,]), ), 0)
uop_4745 = relay.log10(const_4741.astype('float64')) # shape=(180,)
output = relay.Tuple([bop_4727,bop_4735,call_4739,var_4740,const_4742,var_4743,uop_4745,])
output2 = relay.Tuple([bop_4730,bop_4738,call_4744,var_4740,const_4742,var_4743,uop_4745,])
func_4750 = relay.Function([var_4726,var_4740,var_4743,], output)
mod['func_4750'] = func_4750
mod = relay.transform.InferType()(mod)
var_4751 = relay.var("var_4751", dtype = "float32", shape = (14, 14, 2))#candidate|4751|(14, 14, 2)|var|float32
var_4752 = relay.var("var_4752", dtype = "int16", shape = (825,))#candidate|4752|(825,)|var|int16
var_4753 = relay.var("var_4753", dtype = "bool", shape = (550,))#candidate|4753|(550,)|var|bool
output = func_4750(var_4751,var_4752,var_4753,)
func_4754 = relay.Function([var_4751,var_4752,var_4753,], output)
mutated_mod['func_4754'] = func_4754
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_4816 = func_3786_call()
call_4817 = func_3786_call()
func_571_call = mod.get_global_var('func_571')
func_575_call = mutated_mod.get_global_var('func_575')
var_4827 = relay.var("var_4827", dtype = "float32", shape = (576,))#candidate|4827|(576,)|var|float32
var_4828 = relay.var("var_4828", dtype = "float64", shape = (325,))#candidate|4828|(325,)|var|float64
call_4826 = relay.TupleGetItem(func_571_call(relay.reshape(var_4827.astype('float32'), [16, 6, 6]), relay.reshape(var_4828.astype('float64'), [325,]), ), 2)
call_4829 = relay.TupleGetItem(func_575_call(relay.reshape(var_4827.astype('float32'), [16, 6, 6]), relay.reshape(var_4828.astype('float64'), [325,]), ), 2)
func_3767_call = mod.get_global_var('func_3767')
func_3772_call = mutated_mod.get_global_var('func_3772')
var_4835 = relay.var("var_4835", dtype = "float64", shape = (2048,))#candidate|4835|(2048,)|var|float64
call_4834 = relay.TupleGetItem(func_3767_call(relay.reshape(call_4816.astype('float32'), [14, 14, 2]), relay.reshape(var_4835.astype('float64'), [128, 16]), relay.reshape(call_4826.astype('float64'), [325,]), ), 0)
call_4836 = relay.TupleGetItem(func_3772_call(relay.reshape(call_4816.astype('float32'), [14, 14, 2]), relay.reshape(var_4835.astype('float64'), [128, 16]), relay.reshape(call_4826.astype('float64'), [325,]), ), 0)
const_4840 = relay.const([-4.509500,-7.423983,1.167201,5.867250,-5.523829,4.605099,6.685448,-2.012004,9.439964,4.996504,-4.758885,-4.394332,-1.804842,-6.307006,0.130250,7.471787,7.244903,8.027383,-9.375297,-4.106571,3.018851,5.947216,-6.512967,-1.738116,-0.408980,-0.506546,0.821810,9.710261,4.620766,-6.613597,3.653927,-2.384350,9.923827,4.950698,-6.394689,-7.708068,8.031723,6.273688,5.451009,2.231093,8.710271,9.082259,8.325642,-8.013919,-1.934755,3.183061,8.548793,9.834245,7.524002,-2.756245,-9.164582,2.576125,-4.282126,-3.757488,-3.459143,-3.022131,2.519612,2.101842,-9.299207,3.205364,-2.944747,4.379001,-5.923048,0.507951,7.939934,-8.222162,-3.280509,-1.897311,-9.778162,0.031011,-3.381752,-6.193957,2.269182,-9.480071,8.349847,-0.474848,-2.742741,4.057869,-3.165192,-0.033045,-2.714515,-4.579802,-4.296140,-8.485356,3.777052,-9.465010,0.120625,9.111739,-4.975042,-9.307618,-7.380492,1.118744,-1.527797,5.434766,1.269863,5.414067,-3.501469,4.781178,-6.361365,8.095466,6.166283,5.814848,-0.360569,8.030519,1.017004,8.883108,-3.816087,-7.628874,4.124154,-9.523847,-3.188763,9.597131,6.267585,8.352798,-0.661778,9.797921,-9.435841,-6.151878,5.450141,3.005832,3.328714,-1.955699,-0.782069,9.408663,2.542553,-5.063341,6.396786,-5.436350,-2.032776,-8.189841,8.175847,-7.207001,-1.129915,-0.409821,-8.516771,-0.383346,-1.344478,-6.679195,-3.741929,1.073939,7.499122,2.142278,1.974588,0.200916,7.462763,-7.894522,7.067413,6.364694,9.788783,4.120635,6.054309,8.032525,6.478914,-5.625495,1.206222,0.169156,-7.879558,4.025570,-1.453750,9.868421,-6.207302,-8.109668,2.056684,-8.573170,-2.727624,-1.837738,2.443553,6.343634,8.285601,-1.828833,8.346909,6.999503,-8.104057,8.160029,8.755803,8.574289,-7.478809,0.824306,9.451523,-1.341992,-9.029324,6.189425,6.937502,-2.826527,8.714173,0.119159,-8.284755,-6.701401,-7.026382,3.227466,-0.408361,6.549654,1.744321,-9.941406,-2.116521,-2.544361,-9.535423,2.063787,-4.091795,-7.107054,-3.738638,6.071152,6.199184,-7.670424,2.804430,7.500708,-5.031820,8.546941,-9.760557,-2.456997,-6.594588,8.646146,-6.545420,-9.235402,-0.175838,-7.117676,4.409943,1.279687,9.450141,-6.070122,9.299941,7.311020,-7.692709,8.256213,-0.272853,-7.520756,-5.602066,1.774285,-6.778711,-6.192992,7.954556,9.573286,-4.604680,1.490780,-6.387058,-1.127629,9.686379,6.796598,-9.795016,-9.947023,-1.903370,-5.162146,-8.894183,-5.579702,4.305799,6.797964,4.638563,0.516539,2.139409,-4.758225,-3.789397,-4.903169,-4.691536,-5.266242,-8.508747,0.646631,8.756807,-3.232404,-5.331880,-4.588122,3.478550,-2.522516,-7.772283,-9.353920,-7.955692,-3.870878,4.103299,-7.722023,-8.764446,8.929119,-3.915589,-2.369683,-6.129020,3.518269,9.729548,-5.659886,-1.143906,4.072691,4.787936,-9.900389,-1.826620,-6.105951,8.147052,4.874285,6.103677,0.904344,-7.908600,7.062514,-1.097617,2.643307,-2.657809,-5.434425,-2.259966,-9.512069,9.251176,2.046449,0.178270,2.174822,-1.563212,-3.277663,2.050781,0.724681,0.978748,-5.010040,-3.061065,3.332331,-3.387705,8.191444,-9.838042,-3.821504,-2.861690,8.552536,4.312050,4.025092,-3.675592,-7.647981,7.288353,0.286601,-1.829728,-3.733814,1.143170,-7.831640,8.265057,-0.258973,7.507858], dtype = "float64")#candidate|4840|(325,)|const|float64
bop_4841 = relay.less(var_4828.astype('bool'), relay.reshape(const_4840.astype('bool'), relay.shape_of(var_4828))) # shape=(325,)
func_4230_call = mod.get_global_var('func_4230')
func_4232_call = mutated_mod.get_global_var('func_4232')
call_4851 = func_4230_call()
call_4852 = func_4230_call()
func_2954_call = mod.get_global_var('func_2954')
func_2956_call = mutated_mod.get_global_var('func_2956')
var_4873 = relay.var("var_4873", dtype = "uint8", shape = (144,))#candidate|4873|(144,)|var|uint8
call_4872 = func_2954_call(relay.reshape(var_4873.astype('uint8'), [3, 16, 3]))
call_4874 = func_2954_call(relay.reshape(var_4873.astype('uint8'), [3, 16, 3]))
uop_4875 = relay.tan(var_4828.astype('float64')) # shape=(325,)
bop_4877 = relay.greater_equal(uop_4875.astype('bool'), relay.reshape(bop_4841.astype('bool'), relay.shape_of(uop_4875))) # shape=(325,)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_4880 = relay.TupleGetItem(func_3866_call(), 2)
call_4881 = relay.TupleGetItem(func_3867_call(), 2)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_4882 = func_3786_call()
call_4883 = func_3786_call()
output = relay.Tuple([call_4816,call_4826,var_4827,call_4834,var_4835,call_4851,call_4872,var_4873,bop_4877,call_4880,call_4882,])
output2 = relay.Tuple([call_4817,call_4829,var_4827,call_4836,var_4835,call_4852,call_4874,var_4873,bop_4877,call_4881,call_4883,])
func_4899 = relay.Function([var_4827,var_4828,var_4835,var_4873,], output)
mod['func_4899'] = func_4899
mod = relay.transform.InferType()(mod)
mutated_mod['func_4899'] = func_4899
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4899_call = mutated_mod.get_global_var('func_4899')
var_4901 = relay.var("var_4901", dtype = "float32", shape = (576,))#candidate|4901|(576,)|var|float32
var_4902 = relay.var("var_4902", dtype = "float64", shape = (325,))#candidate|4902|(325,)|var|float64
var_4903 = relay.var("var_4903", dtype = "float64", shape = (2048,))#candidate|4903|(2048,)|var|float64
var_4904 = relay.var("var_4904", dtype = "uint8", shape = (144,))#candidate|4904|(144,)|var|uint8
call_4900 = func_4899_call(var_4901,var_4902,var_4903,var_4904,)
output = call_4900
func_4905 = relay.Function([var_4901,var_4902,var_4903,var_4904,], output)
mutated_mod['func_4905'] = func_4905
mutated_mod = relay.transform.InferType()(mutated_mod)
var_4932 = relay.var("var_4932", dtype = "float32", shape = (8, 6, 4))#candidate|4932|(8, 6, 4)|var|float32
uop_4933 = relay.asin(var_4932.astype('float32')) # shape=(8, 6, 4)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_4937 = func_3706_call()
call_4938 = func_3706_call()
func_2954_call = mod.get_global_var('func_2954')
func_2956_call = mutated_mod.get_global_var('func_2956')
var_4941 = relay.var("var_4941", dtype = "uint8", shape = (144, 1))#candidate|4941|(144, 1)|var|uint8
call_4940 = func_2954_call(relay.reshape(var_4941.astype('uint8'), [3, 16, 3]))
call_4942 = func_2954_call(relay.reshape(var_4941.astype('uint8'), [3, 16, 3]))
func_3273_call = mod.get_global_var('func_3273')
func_3277_call = mutated_mod.get_global_var('func_3277')
var_4962 = relay.var("var_4962", dtype = "uint16", shape = (1260,))#candidate|4962|(1260,)|var|uint16
call_4961 = relay.TupleGetItem(func_3273_call(relay.reshape(var_4962.astype('uint16'), [14, 9, 10]), relay.reshape(var_4962.astype('uint16'), [14, 9, 10]), ), 0)
call_4963 = relay.TupleGetItem(func_3277_call(relay.reshape(var_4962.astype('uint16'), [14, 9, 10]), relay.reshape(var_4962.astype('uint16'), [14, 9, 10]), ), 0)
func_3706_call = mod.get_global_var('func_3706')
func_3708_call = mutated_mod.get_global_var('func_3708')
call_4967 = func_3706_call()
call_4968 = func_3706_call()
func_2018_call = mod.get_global_var('func_2018')
func_2021_call = mutated_mod.get_global_var('func_2021')
const_4977 = relay.const([-6.068116,-9.852860,8.009465,-7.380298,0.178646,2.983778,3.962437,7.937296,1.188648,-1.561990,7.222944,-2.386127,4.532619,-3.465201,-4.823403,2.848685,5.719156,-5.963076,8.220819,-8.324338,4.472395,0.032722,2.075878,6.472026,-7.990351,-0.852072,-6.418652,-4.485791,-4.699879,-9.103574,6.801328,3.815722,-1.237847,1.978997,6.761473,-6.030736,3.488469,3.904903,-1.326654,-1.669010,-7.574123,-8.055566,7.823869,-5.958040,-5.391672,6.412005,9.000825,-3.944717,-3.043357,9.965928,9.393959,-5.002107,-7.673478,2.318313,-2.868781,4.599690,-6.638074,-3.615197,1.403271,4.373634,8.227258,8.732742,-5.934120,-4.149528,7.100421,-9.789129,8.334564,7.802398,-2.696934,-8.712634,-5.649276,-7.134573,-2.166914,0.084182,8.094655,1.430355,5.453497,2.585691,-7.669543,-1.447977,-7.139402,-0.019862,1.769994,5.993958,-1.916189,-1.152785,2.647488,-6.329623,1.322813,7.633230,-4.053618,-8.945624,0.727751,1.685990,4.284674,6.704259,-4.085165,-9.734569,-1.928262,-5.853961,5.687381,7.685973,6.266746,-8.393585,5.839059,3.755445,-8.851316,6.980627,-3.016200,4.252465,-5.745381,1.043839,-7.003146,-0.491143,8.968812,-2.780816,-3.119707,0.952024,8.914955,-4.304396,-3.606993,-7.673556,5.225980,-2.142166,2.887437,-8.426675,1.297349,-3.226816,5.148144,-4.494057,0.625465,7.228830,-2.066205,3.068629,6.498008,0.701831,-8.547188,-2.510776,9.396180,9.446035,3.794734,2.507561,6.543849,-4.393457,-4.605719,6.320447,8.969957,3.911911,4.435739,1.117629,6.122281,-6.070130,5.785393,1.693936,-2.541736,0.212111,-0.102415,-1.262855,-3.032274,-0.314794,6.877717,6.127229,4.539400,-9.769779,6.292579,9.912155,-4.638739,-6.717127,-0.653743,5.221476,-9.566625,-2.951283,6.535393,-1.717162,-7.091859,-2.757699,-5.284879,-8.979791,6.094561,9.730796,9.451817,4.077175,-7.153254,4.268832,5.550891,6.257445,4.150977,5.945591,-5.938268,-9.412503,-5.215486,-1.069627,-9.188906,-4.410632,-4.166661,3.699664,-7.055671,1.017488,-4.020203,-8.670097,4.977285,-7.003992,9.529568,6.196398,0.016521,-3.244527,1.767517,4.070031,-2.234131,4.594781,8.164114,-6.646082,4.223237,-8.458303,-1.163700,1.456133,4.061141,-0.436297,0.020509,-2.883109,6.489416,9.212843,-8.590161,9.529118,-9.035097,1.310283,0.463222,-2.592647,3.709927,-7.012167,5.927599,-7.005775,5.519510,2.219952,-1.959476,-2.162012,6.693604,-1.343531,-1.994700,-9.764971,-7.030489,-0.281148,5.381055,-1.143262,1.699802,-7.977045,1.343419,0.962577,2.509367,6.199558,5.579813,5.449843,2.813802,-0.682416,3.421578,4.266843,-1.764888,6.677382,-3.416143,-2.391071,5.050233,-4.435045,8.610172,4.113507,-1.529692,6.893725,-0.582617,-2.156758,-8.069417,3.079660,6.663612,-2.682065,-8.429809,4.029011,5.172352,9.732888,-2.857191,5.685647,4.984281,-0.710559,-9.846356,3.911035,1.253931,-4.625688,-4.218830,1.945086,-1.895571,4.438448,-3.576884,0.786518,1.240707,-9.698433,-8.531468,5.454472,-8.380168,-0.793791,5.781553,-3.479959,1.450204,9.943594,-8.499833,-2.268639,-8.687639,-3.154895,-4.604153,-4.269723,-2.460179,-5.360535,4.293854,-2.125837,9.520513,-2.626504,9.587432,9.719822,2.553404,-1.459176,-0.946954,3.554230,1.002001,3.212591,5.173541,5.007977,-5.802632,2.022327,0.610336,9.779301,-2.488197,-4.570211,-2.349776,-5.932246,-9.726085,-6.697847,9.322164,0.264561,5.386920,4.256476,-2.976058,-6.218480,-2.888283,-4.015071,4.689798,-2.371295,1.013607,4.224195,6.869987,-3.970803,5.912589,-9.020889,-9.836249,1.596439,-8.059785,-6.220028,2.487782,5.570771,-5.081841,2.050090,9.915405,0.764247,-8.738569,9.120199,0.241337,8.156623,-1.907290,5.623410,9.725284,-2.763204,4.177064,1.348371,1.815434,-4.275403,-3.922715,0.606670,-8.154611,8.809839,-6.188828,-7.828387,-1.966618,-7.507324,1.482382,8.014766,-6.069006,-3.929038,-8.376112,-0.067316,-8.439935,-1.964608,-9.463165,-7.643183,-8.614930,-1.164772,-9.445049,-4.402354,-8.788209,6.505588,6.774082,0.926238,-8.532479,8.105244,8.353094,-3.311199,2.069457,-8.271247,2.250305,-3.582158,-7.041810,-7.854996,4.654638,9.970362,6.075688,2.060760,8.906380,-2.769681,-7.191809,3.115936,5.816023,5.487207,-1.743376,-0.709037,-3.309781,4.055249,5.411167,-4.759620,9.173624,9.094792,-2.805479,5.613660,4.318569,-2.105272,4.567149,8.284570,7.416734,8.656685,-5.068321,0.654556,-3.926250,-6.359479,-1.046189,-0.747575,-7.993113,-4.685744,-0.368957,3.487944,4.478092,3.716686,9.498908,6.835263,0.806986,-4.849716,7.534740,4.326450,-0.691773,0.605954,-7.361675,-1.992630,3.219645,8.207414,-7.085052,0.761661,-2.531683,1.352107,4.190285,4.789842,6.545486,3.724004,5.867030,5.655950,-2.351826,0.845679,-5.892203,8.808066,-7.289565,-4.793372,0.231468,1.522467,-6.665551,8.952531,2.158998,0.015674,-6.520666,8.653739,7.880579,-9.423076,-1.295384,-8.539105,5.157100,-4.968489,-4.986572,7.573878,4.192707,-4.417378,8.647430,6.988277,-9.353332,7.070830,9.443474,-6.757583,7.210778,-6.033782,-7.237796,0.902075,4.391682,-8.410140,-8.870996,0.846713,-9.123606,5.393673,-6.395295,-7.453591,3.981459,6.791628,6.848050,4.077739,-1.286849,-3.312384,-3.387325,1.810733,6.344941,-7.019085,4.443032,0.769081,4.962595,5.250453,-0.576487,-7.704101,9.144858,-9.797740,8.262053,-3.060200,-9.604306,6.330591,-2.053204,-5.793646,-1.051424,0.456314,-7.277330,1.187885,0.497435,1.006890,-4.607624,-1.647972,-6.878404,-5.413266,-7.180117,6.979864,4.384594,-7.260046,2.711486,-4.614221,6.849499,5.556439,-9.313648,9.758760,4.998351,2.739161,-3.773743,-5.924850,7.146258,-6.059012,6.244100,-7.612031,-0.971798,-2.417718,3.670816,8.218608,-2.101254,-2.671137,4.807707,-2.738348,4.966293,1.541455,7.013939,-2.955110,-3.013546,7.701302,-1.114967,6.868544,-9.072461,6.422844,-8.293260,-0.636684,0.451268,6.816190,-6.413312,-9.066390,-2.595500,9.850600,8.627696,-1.705748,-2.418090,0.673656,6.332984,-0.338175,-4.596727,9.686860,-8.779518,1.038884,-9.842326,5.872818,-5.865894,-3.143843,7.710617,5.850526,0.488292,-7.584343,7.125294,4.739008,-8.104137,-5.135863,4.406916,4.801123,1.848193,7.121902,-3.228006,-5.187443,-1.544780,9.996552,-5.050588,-3.334904,-8.153560,4.170559,-5.658879,-6.549058,-3.796753,1.308888,-7.300273,-9.616906,9.801436,0.469655,-2.110358,8.487635,1.261636,-1.234611,2.846496,3.098727,2.501551,9.319520,5.299970,4.158484,-1.357509,3.263664,9.572936,-1.495987,-7.258171,-7.535564,6.936557,5.871026,-1.805166,-9.649137,4.622745,5.713844,-5.678313,-1.100108,-0.599797,-5.191345,2.543288,3.299280,4.335709,7.593072,-8.849320,-3.396485,8.723494,-5.002885,3.561959,8.951135,-4.748138,4.054837,5.521238,8.246700,9.985132,-2.629839,2.542346,-7.829809,1.650776,-6.773256,7.917680,0.299856,6.882934,1.966398,-9.261571,4.551521,8.167962,-4.636402,-3.509473,0.297584,1.663317,-8.649574,0.797797,2.527870,3.186610,-2.339801,-2.281742,3.324745,0.546589,-0.292564,-6.032243,6.309257,-7.458012,-3.271974,1.701131,3.468697,-5.447367,3.614419,-0.861164,-1.330540,8.145543,-2.340471,4.467464,1.389066,-8.972971,-6.054414,9.998832,6.877512,6.602407,-3.066252,2.259106,0.214486,-0.623246,7.757996,8.629231,-5.305769,1.402002,-2.159337,-8.052365,5.305269,4.071679,1.516860,8.162513,6.368522,0.813480,5.981078,0.171135,-1.461230,-5.028837,-6.388198,3.264236,-9.974885,1.238744,5.002141,-5.698488,5.193261,6.790248,-7.682204,1.630082,4.940083,3.927831,8.192683,-4.242883,-6.602174,3.167298,-7.704690,9.376742,3.254932,-8.591943,7.146337,-0.968158,-2.678910,4.526244,-6.891754,-7.588932,3.227496,3.592281,-9.282098,-0.446262,-8.700755,4.412429,7.832033,1.228648,-7.930311,8.741627,-1.004242,-3.880697,0.932070,1.726535,-2.739935,-2.922589,7.212548,-0.371106,6.399277,3.277438,-7.500985,-2.236134,1.562228,0.291606,8.612805,6.178047,-5.539498,6.232967,-2.998423,7.875669,-6.515336,-7.878139,6.767714,-9.211208,-2.019783,-1.005132,-5.069617,-5.400540,-7.690464,0.892795,-6.579783,5.102038,-7.663038,0.593834,9.531408,5.466923,-7.165518,-8.607463,-1.857615,-5.903982,-0.980981,-5.990437,8.615947,-2.547758,-8.405663,-7.701087,0.256050,0.198694,-0.668191,-5.468717,1.058423,9.369983,5.762305,-9.525949,-7.025898,-1.875099,7.476599,-8.182361,3.793335,4.901532,-9.867185,7.238839,3.049997,-1.111812,4.419775,-5.398281,-0.599393,-8.533524,2.832327,-1.753117,4.933588,-0.197416,9.072058,1.706493,0.278248,-4.105879,-9.261409,1.639439,-9.171514,1.400537,-0.660249,-0.836985,-2.492111,3.873347,8.187145,0.751046,-6.058755,-7.772864,-5.774269,1.988203,-8.700499,-5.789748,2.500827,-5.620998,7.099500,3.395569,1.308721,-8.132457,-2.949317,9.741563,9.418962,-1.961403,-9.547326,1.453630,-8.342700,4.354484,-2.991569,-5.185725,6.740664,9.608134,-3.730747,7.743033,4.259047,5.690028,8.647113,6.076692,-0.366693,-2.841659,-0.857464,0.929122,-2.721243,2.660797,-3.888471,-7.339675,6.269961,-8.586434,-6.152751,6.719417,-8.331499,5.142282,-2.524010,-1.389044,-7.096328,9.454561,1.031630,-7.450693,0.748043,-4.019906,8.794213,-2.258942,9.587305,-2.884679,-9.410397,7.277796,9.497363,-5.278764,2.575356,-0.839981,8.723307,3.059909,-6.035787,4.237576,3.964758,-3.664578,-6.180341,5.475488,4.909220,-9.700386,-8.657179,-4.253224,-3.274514,0.100399,-2.877321,-2.407401,1.035359,-1.208922,6.092754,-7.176046,2.642408,1.560950,3.639052,4.835959,-4.392389,-2.017048,5.390670,-6.167309,1.610173,-6.919887,3.819450,2.869330,-9.796368,7.509154,8.020289,-7.460422,-3.072520,1.470184,7.296544,-5.985158,-4.838488,-8.108851,-8.088794,4.521141,-8.283458,2.309453,-8.009154,-5.286298,-8.345751,-9.738256,1.168214,-0.032639,-5.839570,6.822363,8.293179,-6.513623,-2.581762,5.535975,8.424006,-0.843215,-5.855928,-3.393474,2.094398,9.513600,-0.126747,5.631527,6.159284,8.537586,1.518890,2.587969,-2.770478,-9.549528,-4.677153,5.950502,7.167978,3.158209,7.525144,-8.390048,-7.977396,-1.383362,-0.127601,3.418512,-9.978490,0.530717,9.598151,-5.229067,-7.690341,6.237758,-5.918976,7.382282,-2.162764,-0.801304,-0.659863,-3.715696,-3.817099,2.076594,5.706143,-1.431454,6.434868,4.744107,-7.718795,-6.574955,7.182497,9.465145,8.107600,2.027627,8.715036,6.728165,-9.379753,1.689400,3.985886,4.237436,0.354379,-0.961028,-6.324637,6.870544,8.467749,2.389198,0.419170,-2.285492,9.566300,4.745691,3.774837,-1.032677,-4.440183,0.147668,-6.361513,-8.253065,-7.557548,-9.953099,4.233026,-3.920845,-7.114054,5.507636,-5.247944,-2.745771,7.004661,0.337239,-8.104039,-5.182359,7.196086,-2.180211,7.384493,7.994759,1.026824,-2.060312,5.512621,6.794527,-7.628064,-0.187995,8.100601,7.134966,-0.168618,9.909938,-2.926498,1.088462,7.209293,-1.701309,5.049455,9.306429,8.808646,7.563550,-1.426136,-4.436905,4.773817,7.312075,2.996937,2.341277,3.724874,8.885265,-8.358406,6.159314,-8.861140,-1.445306,9.461308,8.683560,8.784361,-3.372873,2.374828,6.056872,9.992726,2.042406,8.117488,-7.670234,-3.236205,-9.503774,-0.108558,-1.971606,0.357040,-5.921922,-7.927305,-0.874396,-5.757133,2.694342,3.294681,-8.520068,-6.489738,-7.993517,-7.172864,-5.911141,-8.937314,-0.070228,1.637825,7.410575,9.191597,1.235392,2.485411,-2.737050,-2.708117,-7.130946,-1.173012,-0.679188,-0.913371,-4.531556,7.195439,1.876038,0.909095,0.091813,2.020862,-6.505395,-6.024033,-5.326055,7.549314,-3.790545,5.027293,6.828594,5.557641,-5.308407,-4.442710,5.182688,3.286719,9.095396,-5.888961,9.403520,-0.285797,-3.269127,-1.178362,-1.069165,4.596646,6.421669,9.501895,-1.687285,1.523325,5.428894,4.969568,3.011948,3.855659,-2.269133,8.540261,7.571231,4.145062,9.398786,3.181896,2.073742,-1.552268,5.550057,9.165143,-6.349517,-8.197127,6.904419,-7.906351,-7.049014,6.972868,1.371811,3.495416,-6.777207,6.116611,-1.728703,-2.518301,4.122138,-5.292481,-3.036591,7.519874,-9.827856,-7.382838,8.344758,4.310101,0.139231,-0.261478,0.962550,1.743756,-5.345685,7.539326,8.033103,-8.708935,-0.639657,7.127383,1.914123,-5.392713,-2.109360,0.824315,6.481365,-1.454584,0.173771,-0.910739,-2.367722,5.513013,-8.283294,5.759117,-6.704540,-4.193722,-0.215954,7.234321,-6.585075,-3.658162,4.578801,8.698741,-3.198850,2.400949,-4.466024,-5.515002,8.404473,-3.112793,6.243499,-2.963530,8.350984,-7.594393,-6.186038,4.184882,5.336326,-9.047687,7.285278,-4.629345,2.567578,-0.177992,9.704367,-9.961725,-6.455254,-1.118715,-4.589486,9.074754,6.541693,2.628945,8.321085,4.990219,-4.452791,9.392931,3.560812,4.306389,2.272013,-9.186483,6.818445,-1.849951,0.453155,5.285591,9.685456,-2.137718,-9.689708,-7.060188,-8.329580,5.599915,3.490479,0.214386,-0.771152,9.788124,-8.796758,-9.982781,-1.419378,2.090476,9.956976,5.241135,4.523588,9.226425,-2.976749,8.551674,1.427955,9.056942,-3.180034,2.309553,-8.449626,-5.586366,0.953772,-8.900306,1.859044,7.029460,-3.896223,4.937553,-4.856230,8.262062,0.438878,-5.460557,9.823055,8.553063,-5.915747,-3.783006,-1.977965,0.240239,1.558068,3.586766,2.141993,9.009499,-1.912982,6.652474,-6.280796,8.186861,2.493900,-8.762333,0.192757,-8.835204,7.733657,-3.435768,-1.645863,5.944996,1.634131,-1.056247,7.283019,-0.917701,0.644764,2.326854,2.278189,1.406959,-3.432917,8.561662,5.781878,2.359108,9.435789,-1.042101,6.083132,5.190362,6.960291,3.485840,-6.936139,-5.936901,7.792700,4.271710,-3.398506,6.907507,-9.553663,-7.148599,2.160194,-0.966889,-8.417499,-0.697489,-8.635463,-9.881441,1.109959,8.254185,-5.526785,6.011564,5.376422,-2.165458,1.768438,-6.006346,-5.676564,-5.957638,-6.992866,-1.383327,8.068178,-0.913286,6.309444,3.178614,-8.138445,-7.915620,-0.482118,-9.753021,-3.545817,5.577447,-4.215547,0.649174,4.134357,-1.292019,6.180692,-4.652465,2.972227,-7.821167,0.605589,-3.826371,-4.465404,-0.771809,-4.045514,-6.746903,5.684943,4.544621,3.434485,6.768003,-7.757500,-7.278517,-3.635597,-7.111356,9.927056,-5.213133,3.057014,-4.752442,1.534443,-3.351187,1.625007,-7.685770,0.614766,3.840642,4.368451,9.317421,-2.467456,1.747895,-6.845427,2.386731,-5.918427,-5.100903,-2.627020,-8.882684,-3.804314,1.514029,1.651389,-3.750883,1.250409,-7.199975,-6.232924,-6.283036,8.835501,-7.149689,-2.757710,-5.252998,4.847180,-1.096200,1.114567,-7.551480,5.095701,-1.428021,4.785356,8.349001,-7.007620,0.996658,8.222591,6.975547,0.997429,-3.446916,5.227707,0.451164,-9.409761,-3.519918,-8.908194,1.748097,9.205263,-6.674583,-6.405568,9.336497,4.350725,-7.349887,5.732078,-0.981615,8.709554,-8.613766,8.693203,3.595301,9.568267,5.152848,-7.653171,0.225264,-3.346758,5.047423,-8.658597,8.038791,3.539065,-8.489624,-4.340558,-1.576679,-1.494098,-8.295938,-6.581390,5.243565,-2.601649,9.491944,6.530522,-2.732842,-4.410946,-4.621782,-5.562504,6.886412,4.077087,-0.483304,-9.615701,6.440825,9.246877,-9.846064,-4.272151,4.602343,-1.301488,3.537754,-4.702315,-8.736549,8.883237,6.182831,-6.967253,-2.884508,-5.567491,5.624663,1.331227,0.034589,5.587134,-6.978118,5.603705,-6.768900,-7.341423,5.061796,7.695911,0.332966,6.432040,-6.508496,7.665685,3.240085,-8.922226,-0.297301,-5.660112,-4.855708,3.223635,5.103540,7.891723,1.868340,1.191381,-0.916445,-4.164771,-0.803796,-9.131059,8.020580,2.094458,8.676038,-4.984931,8.758580,-6.256628,-9.841055,-3.916207,-0.421137,8.129936,0.906782,2.902170,6.733499,9.295655,0.460490,-8.336306,-8.444842,1.635206,6.102578,4.024388,-6.711680,2.303975,-5.609964,-6.475662,0.469691,-5.673624,6.531937,-9.186349,4.086390,-0.804843,0.247005,4.960352,-2.702198,9.294026,-0.050068,-7.575793,-8.250712,6.341879,2.967851,-0.670979,7.884488,1.694554,-4.559847,6.289581,0.090151,-8.748791,8.799684,-3.940561,-1.145891,2.856541,8.439038,-6.975396,2.817417,-8.183656,6.448945,1.646500,7.841036,8.576533,-3.477729,7.947565,-1.642205,-4.226321,8.851897,2.993178,1.034289,8.928637,-0.645429,-6.234755,3.669878,-7.769531,6.983876,-7.857004,-7.443701,-9.309396,-1.177107,3.643405,0.800811,-2.187107,-9.189846,8.648247,0.437916,-1.615546,-1.316632,7.835421,4.487781,-0.449406,-7.141225,0.870144,4.342706,0.659553,4.217387,-7.792838,9.852658,-2.862812,0.014454,9.175215,-2.876946,-0.988857,-3.145058,-4.732848,-7.110868,8.697850,7.988699,6.276533,-7.987811,7.913176,0.081890,2.566860,-3.234075,9.697341,9.225442,-0.199167,-5.109400,-9.281532,6.861674,1.493623,4.883678,-7.113097,2.534085,7.832610,-4.273272,-6.234620,5.545825,-6.885927,1.523407,4.107447,-4.068111,-7.365746,-4.228529,-5.915375,3.805835,2.842607,-3.807852,-2.573295,-2.770842,3.374247,-3.121384,-0.643816,7.818570,1.916315,-4.854778,5.837656,3.434779,-3.910483,-5.494899,9.916017,-5.170052,3.828455,-0.075941,-4.895645,9.760905,3.593786,4.409822,9.719694,7.436484,0.515467,6.256661,-6.395484,-6.553196,0.887355,1.050898,0.759484,2.236125,-1.495748,-2.082040,-0.269085,4.030999,4.484401,-0.230412,6.423595,6.648874,2.901228,-6.754457,9.812157,-7.888433,7.247852,1.483957,4.377187,2.985472,-9.228547,8.760462,-8.093630,1.214229,1.605365,-0.040217,9.551767,-1.822644,6.260573,-3.356522,4.938103,4.099413,3.967933,0.412688,6.356450,-9.262296,-8.590844,5.372458,-6.351052,-5.700929,-7.795051,7.420319,2.957107,-7.603742,-4.550002,6.684141,-9.043078,-4.224743,-8.159176,-3.526705,0.583731,6.937062,-2.070985,-6.350538,4.910334,6.833405,0.672331,3.174595,-5.977812,8.230222,9.988279,-9.079395,5.693539,4.290352,4.532756,-8.900785,-7.097141,-8.742876,-7.002425,-7.813922,-6.955720,7.245174,-2.567277,-2.013569,-8.383444,0.696258,-2.923671,9.501722,-9.956687,0.537077,7.798693,-1.569509,-7.659916,-3.209946,1.138888,-7.298032,-4.565150,-8.523587,4.809419,6.133922,-8.416946,1.761930,5.712844,8.455241,6.939174,6.658898,-1.899278,-9.984884,-3.960296,9.697605,3.994977,9.875843,0.300655,-6.445882,-7.721820,-9.932004,-0.678419,-5.269082,2.443956,-4.127685,-5.694207,7.259276,9.341568,1.927858,-2.679763,-6.071383,0.959412,-8.643685,8.928601,3.879626,-9.115375,-7.529548,1.066965,-4.669791,-3.786666,2.549804,-3.200986,8.242052,-3.165820,-2.702841,4.773302,5.666787,-8.696498,0.003241,-9.047225,-1.657594,4.817117,-9.781733,7.614507,0.175478,0.326096,-6.951642,9.015085,1.759098,0.202631,-4.031088,-7.195961,7.344864,7.496699,-2.978664,9.217461,0.279884,-5.128000,7.756834,-5.934855,4.448050,6.754632,-1.605042,-1.724677,-5.420677,9.945733,-3.400662,-5.408477,-8.679367,1.124759,-6.515373,6.642886,9.573271,8.812912,9.542254,8.746719,2.545120,6.872318,1.708606,8.870802,0.685993,-4.918847,9.582763,0.930674,1.150603,-7.732569,8.690274,6.215350,1.121800,0.481319,-9.771633,-7.561110,4.104130,7.836903,3.744837,-7.866963,8.641418,-1.320518,-1.366788,4.516080,-8.974335,8.847072,9.959552,-0.124577,4.923115,-6.725928,-4.378203], dtype = "float32")#candidate|4977|(1890,)|const|float32
const_4978 = relay.const([[3.471079,-4.877738,-0.468920,2.823942,-3.435869,-4.631890,9.221404,-2.775355,-0.377287,4.276715,-9.991007,1.198925,-2.717997,-7.008966,-0.067504,-2.872767,5.113867,-5.921315,-0.361552,-4.064355,-0.004934,5.216035,-8.844534,2.890235,0.030899],[-8.164726,-9.270847,-0.258177,-3.669327,8.694007,-1.721551,5.732691,-0.307286,1.265190,6.671026,1.935357,-3.936280,6.675304,-6.150014,5.295508,-9.791626,5.814012,8.987745,0.611673,-3.682447,-8.450551,7.052698,-8.880548,-6.905574,-5.293574],[-9.831969,6.593989,-5.504512,7.254611,1.564798,-4.331119,2.487601,-1.202712,-6.035703,-1.877821,-5.560704,8.776108,-3.477166,8.252724,5.599235,0.897050,5.635126,-4.424427,0.743883,-8.297758,-9.510379,-2.462503,-7.444392,-4.902720,5.259562],[4.012832,-0.782797,4.312640,1.073261,2.426152,-0.888848,-0.548897,5.646948,5.238092,-6.640341,-7.734042,7.182008,-8.898783,9.447650,-6.286106,-6.318995,1.455610,-4.443669,0.163532,3.222802,-3.593222,3.958815,-5.779828,0.766644,-8.258998],[7.783669,-0.066919,3.579201,8.580842,7.180294,6.253800,1.743390,-9.881188,1.300034,1.047829,-2.457234,-9.473148,-9.788349,3.548836,1.312102,-2.400682,-6.375248,6.690382,1.948986,-6.320346,-9.998498,-5.322810,-0.007019,5.720364,7.778635],[2.488817,-3.893712,3.322439,-6.777709,-3.638199,-6.485840,-3.719822,0.930413,5.353948,-3.211657,7.993724,0.272176,5.297394,9.418659,-2.040843,3.447988,-0.151405,1.739635,9.152063,-2.589711,0.396177,-7.784536,1.850286,6.750019,-8.935291],[7.870677,-7.506212,-2.800282,-2.637699,-3.810488,-8.512619,-8.660852,1.762072,-9.972413,9.916352,-2.249290,-1.651488,4.144028,-7.999303,-1.119875,-0.276418,3.224836,-5.151461,-0.258450,1.049222,8.482343,4.712404,-0.189316,-9.817859,3.696263],[-2.216219,-3.343349,-6.787995,-5.133387,0.125786,5.665893,-3.790143,-0.731005,-3.344468,-5.632698,-2.358835,4.071156,5.956409,-6.490677,6.840661,-7.541183,-0.514503,-9.617272,-2.410441,2.741394,6.302485,9.673084,3.045360,-6.933922,0.927474],[9.087992,3.259333,4.197391,2.713815,-2.333928,9.277579,-6.259501,1.420475,-9.700346,0.945809,4.443151,-0.603708,9.690699,4.046544,-4.424022,-6.042709,-3.430499,-5.086647,-4.326994,9.722249,-7.868002,5.168155,-1.965189,5.828657,-0.381930],[2.528256,2.262806,1.553987,8.068013,5.293789,9.685615,-8.300605,0.316046,4.738160,-2.283317,-8.130296,6.406148,6.402875,-1.123596,-3.646972,6.784809,1.443510,4.976145,0.919336,7.733981,-8.182700,-2.294747,-5.706428,-2.579491,-7.625313],[-1.126731,1.244548,-1.537395,-3.111036,-2.539135,-7.188595,-2.673106,-6.662319,-7.359075,-3.000343,-9.759681,-1.792239,5.045493,-3.311716,-2.830485,-9.560089,2.543599,-9.956682,-7.957851,8.055876,3.785604,5.499332,5.661104,7.037879,9.855343],[9.670549,1.109587,8.310060,-0.581472,-2.119625,-1.817542,5.363093,-7.052768,2.081614,-5.011649,7.221636,-5.577369,7.566936,5.308824,2.383131,6.218514,-9.730869,-4.093926,5.714106,-2.225448,7.816345,8.813811,-0.612632,-4.471519,-6.988914],[-9.554386,-0.893284,-5.737785,-9.201994,8.676727,0.411268,-0.691080,3.143872,3.025779,1.045314,-5.248727,-8.703566,-4.939793,2.757651,-5.178730,4.232042,-3.471275,-4.368086,-1.047955,-8.493415,-1.231856,-9.992432,8.436856,-8.959240,-4.013711]], dtype = "float64")#candidate|4978|(13, 25)|const|float64
call_4976 = relay.TupleGetItem(func_2018_call(relay.reshape(const_4977.astype('float32'), [14, 9, 15]), relay.reshape(const_4978.astype('float64'), [325,]), ), 0)
call_4979 = relay.TupleGetItem(func_2021_call(relay.reshape(const_4977.astype('float32'), [14, 9, 15]), relay.reshape(const_4978.astype('float64'), [325,]), ), 0)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
call_4984 = func_418_call(relay.reshape(call_4976.astype('float64'), [13, 5, 5]))
call_4985 = func_418_call(relay.reshape(call_4976.astype('float64'), [13, 5, 5]))
uop_4988 = relay.acosh(uop_4933.astype('float64')) # shape=(8, 6, 4)
func_571_call = mod.get_global_var('func_571')
func_575_call = mutated_mod.get_global_var('func_575')
const_4992 = relay.const([[-7.462866],[-1.648486],[-8.202738],[4.502509],[-6.307802],[-9.825552],[-9.414298],[8.630787],[2.536041],[4.891982],[2.069574],[1.480362],[2.450590],[-2.057601],[7.697174],[-6.926365],[1.462098],[0.523003],[2.119112],[-2.924718],[3.056241],[3.231701],[-7.622178],[2.797472],[0.750890],[-3.018404],[-3.584009],[-2.976024],[-7.421872],[2.477279],[8.787726],[1.858681],[0.304220],[5.562554],[-4.852266],[-6.672151],[-0.004424],[3.613136],[2.310173],[6.511088],[4.727492],[4.874053],[9.782275],[-1.761598],[-2.666154],[0.181734],[1.013685],[5.547920],[2.788104],[6.893475],[-3.777629],[-2.541679],[2.317651],[-5.805588],[-6.248785],[2.658715],[2.239307],[-9.635399],[8.034943],[-7.705965],[-2.647077],[4.774475],[-2.234397],[-1.001562],[0.409738],[0.178884],[-6.374658],[3.098479],[3.661364],[-8.371321],[0.884141],[-9.901085],[-9.088454],[5.264525],[9.279325],[1.427776],[7.986897],[5.847437],[6.779747],[3.648963],[1.355424],[2.365821],[-8.135242],[3.457886],[-3.885281],[-9.882996],[-4.773553],[-8.699065],[8.542938],[-3.057134],[2.019309],[-1.814805],[-7.127473],[-3.564069],[-0.154897],[0.522525],[0.398305],[9.564157],[3.218113],[2.107080],[-0.741734],[2.959538],[-6.498118],[2.251652],[0.280237],[-0.342704],[1.879948],[2.405937],[-1.000959],[-0.387783],[3.896656],[-5.795520],[-6.761008],[4.250648],[0.395161],[6.509376],[-5.317480],[-7.486256],[-4.048418],[-3.239919],[2.674604],[-0.753283],[0.591541],[-2.970423],[6.676721],[-6.588375],[-8.685700],[9.066420],[-5.803882],[-5.993528],[-6.120405],[0.453952],[-8.895760],[3.031333],[9.496416],[-4.256299],[8.654889],[-4.941364],[-2.043283],[-6.554360],[7.349404],[0.319081],[-0.401388],[-1.940492],[6.065895],[-5.947360],[-8.219266],[-0.751675],[8.805252],[3.515259],[1.355499],[9.269273],[-7.715409],[1.852171],[-0.553174],[-2.418559],[5.499679],[-7.310367],[3.078305],[-8.374136],[7.903249],[-6.133541],[4.657881],[5.012223],[-7.178534],[7.898507],[-3.034447],[-8.451249],[-8.293599],[-5.444629],[-4.227380],[-6.545527],[-6.787056],[9.266848],[7.463493],[-4.939842],[-8.437692],[-4.037884],[3.910608],[-6.395349],[-2.487140],[8.233456],[-9.189284],[7.409743],[7.706775],[9.614378],[0.548588],[2.063405],[-0.219654],[1.535939],[8.517865],[6.253105],[-9.643401],[3.215169],[9.780417],[3.761573],[0.063706],[-6.459715],[4.978298],[8.191296],[5.263974],[-1.699687],[-2.421208],[-8.180696],[-5.382233],[6.338274],[0.180319],[0.223897],[-0.146659],[1.583697],[-9.872398],[-7.718349],[0.412919],[3.240736],[-4.900900],[-9.659879],[4.803786],[-4.990612],[-8.180859],[6.113711],[-5.920107],[-1.185232],[4.550291],[9.526569],[-3.368588],[4.794951],[-4.414824],[6.907132],[-0.860048],[-5.704599],[1.614155],[9.762618],[0.672558],[-0.397748],[-0.189474],[-0.212775],[-4.877327],[2.567068],[-9.485159],[-0.571270],[0.387445],[7.195123],[8.854694],[-7.145227],[0.809759],[-2.110662],[2.270635],[9.753777],[-1.740814],[-5.343275],[2.941917],[9.373479],[-5.976683],[2.261852],[-7.063090],[0.346472],[9.869354],[2.966625],[2.615912],[3.604039],[-4.870283],[3.034350],[1.209851],[-0.212636],[3.251861],[-2.452455],[1.927212],[-2.511371],[-8.475298],[-8.804011],[8.172592],[0.971901],[-4.839479],[0.424104],[-2.483810],[-9.524613],[5.703305],[-2.827003],[-5.364984],[-4.854378],[9.043238],[9.249282],[-4.515646],[-7.801483],[6.724437],[3.396522],[8.165204],[-5.448191],[8.104817],[6.842771],[-6.847732],[-7.867047],[-4.668122],[-6.853548],[6.931282],[-0.846289],[0.022453],[2.125572],[1.034174],[5.351391],[-0.746746],[-9.111459],[3.912783],[1.626716],[-7.114636],[-0.530823],[-9.481594],[-1.205755],[9.439702],[-4.674940],[1.392610],[0.660063],[-5.247590],[2.993337],[0.633049],[-7.428825],[4.459911],[0.377689],[9.290978],[-6.725911],[8.355571],[-3.675313],[8.163909],[-8.681736],[0.803899],[-3.669723],[9.748790],[-5.978487],[-0.404341],[-5.277136],[-9.661275],[-5.833412],[4.991076],[-6.882277],[-0.118783],[-5.455892],[8.805907],[-2.211826],[-9.099691],[1.623863],[6.818700],[-1.068268],[7.744230],[-2.415040],[-4.900433],[8.170383],[1.072972],[8.135865],[5.041606],[-1.634425],[-8.612771],[-9.842816],[4.495550],[-4.467663],[1.404030],[-5.714223],[5.778392],[-5.387422],[0.255212],[8.816026],[9.104558],[-2.647555],[-8.302731],[9.918939],[4.937576],[3.260061],[-1.093550],[1.843833],[-5.592169],[0.767916],[8.032670],[-8.413382],[0.752809],[7.522396],[5.163002],[-2.534766],[-4.893081],[-0.242035],[-6.239930],[-4.008276],[5.283121],[5.559540],[7.336613],[3.807189],[6.929716],[-4.053521],[-4.865008],[8.009740],[2.770417],[-2.317980],[3.799232],[6.080103],[6.492528],[-9.038756],[2.376791],[4.424762],[8.364178],[-7.318316],[3.843094],[-0.118359],[8.026611],[3.166687],[-1.732919],[3.918796],[-8.522511],[-9.309579],[-7.806139],[8.552229],[5.940040],[-7.948240],[-4.163376],[6.842248],[-4.532306],[5.673985],[-9.240736],[-4.365174],[-7.022200],[2.933117],[7.803142],[-7.646705],[8.691284],[0.739054],[-6.944583],[3.852596],[9.648400],[-7.285593],[-1.853812],[7.014355],[9.455711],[-3.271233],[-3.269311],[2.254378],[2.246538],[6.878997],[8.484804],[-2.664465],[8.990623],[8.147870],[-2.997331],[3.696239],[-4.169889],[0.066407],[-1.030682],[3.632940],[0.946428],[0.316834],[8.806169],[-3.351254],[0.735330],[8.343806],[8.302056],[7.828354],[4.312387],[-1.009420],[-0.189201],[9.314967],[0.165952],[3.375286],[4.883016],[-9.688675],[-0.967444],[9.971436],[-2.270821],[-8.623698],[8.168628],[0.509822],[1.896153],[3.413863],[1.539939],[-7.954223],[-1.876625],[-1.895361],[4.816387],[7.576286],[-2.055574],[5.805174],[1.405991],[-9.196938],[6.984211],[3.840532],[-4.309769],[9.521437],[-2.181069],[1.889521],[3.783630],[7.230167],[4.678387],[0.013055],[3.521999],[-6.783827],[-7.664652],[-7.770529],[-2.109039],[-9.397102],[-9.120586],[-5.920275],[-4.298594],[-3.134413],[2.816731],[-0.780072],[-1.629524],[-4.777239],[5.072937],[-1.182485],[-8.357689],[-5.247861],[-4.127730],[-9.624610],[-4.622255],[-1.910448],[-5.280760],[4.829152],[4.234064],[0.636752],[9.429717],[1.187373],[3.095561],[-0.120944],[2.603725],[9.551692],[6.362212],[1.608139],[2.420197],[-5.961825],[-6.883755],[-3.320983],[0.961104],[5.041046],[7.367705],[4.562402],[-1.940651],[-9.091499],[-2.841557],[-4.651346],[7.805606],[4.173619],[9.797022],[2.148849],[-3.217802],[-7.176024],[-7.442847],[-1.328341],[8.690123],[-3.644444],[-3.271816],[4.805530],[-3.760882],[0.588282],[-0.617318],[1.624068],[7.293281],[-1.477650],[3.014213],[3.331076],[7.899351],[-4.753425],[-3.567772],[-7.558164],[-3.584296],[3.265352],[-8.435388],[6.085615],[-6.125933],[-0.620784],[-3.934515],[4.304700],[-6.278658],[-1.553353],[6.183993],[-9.338796],[-8.195836],[-2.202501],[9.387728],[-3.939759],[-2.301217],[-0.056512]], dtype = "float32")#candidate|4992|(576, 1)|const|float32
call_4991 = relay.TupleGetItem(func_571_call(relay.reshape(const_4992.astype('float32'), [16, 6, 6]), relay.reshape(const_4978.astype('float64'), [325,]), ), 2)
call_4993 = relay.TupleGetItem(func_575_call(relay.reshape(const_4992.astype('float32'), [16, 6, 6]), relay.reshape(const_4978.astype('float64'), [325,]), ), 2)
bop_4995 = relay.bitwise_and(uop_4933.astype('int16'), relay.reshape(var_4932.astype('int16'), relay.shape_of(uop_4933))) # shape=(8, 6, 4)
uop_4999 = relay.rsqrt(uop_4988.astype('float64')) # shape=(8, 6, 4)
func_838_call = mod.get_global_var('func_838')
func_842_call = mutated_mod.get_global_var('func_842')
var_5002 = relay.var("var_5002", dtype = "float64", shape = (140,))#candidate|5002|(140,)|var|float64
call_5001 = relay.TupleGetItem(func_838_call(relay.reshape(var_5002.astype('float64'), [14, 10]), relay.reshape(const_4978.astype('float64'), [325,]), ), 2)
call_5003 = relay.TupleGetItem(func_842_call(relay.reshape(var_5002.astype('float64'), [14, 10]), relay.reshape(const_4978.astype('float64'), [325,]), ), 2)
func_2676_call = mod.get_global_var('func_2676')
func_2679_call = mutated_mod.get_global_var('func_2679')
const_5005 = relay.const([[7.641218,7.955939],[-8.496088,7.015902],[-8.775063,-7.317428],[7.574964,-8.225131],[-6.875892,0.694417],[6.557062,6.714213],[1.116985,-0.777386],[-2.191983,-3.028687],[-4.160451,3.744323],[-5.617372,6.302972],[3.663196,1.348895],[-9.861387,-5.515633],[0.021889,-6.234775],[1.470662,7.715039],[-8.840598,-9.484481],[-9.993096,-8.548506],[-3.562872,-2.179465],[3.167761,9.887943],[9.810774,4.450878],[-9.383445,-6.423449],[-0.412452,9.391136],[-9.783087,-9.172587],[-8.335979,-5.239019],[3.151061,3.692573],[-6.085247,-8.967010],[0.408232,6.773228],[-4.688149,-3.924968],[-8.877183,-4.824582],[7.684270,-1.954503],[2.250753,7.821225],[-8.033224,-3.585626],[-9.090340,6.116204],[1.710468,-6.329750],[-7.058876,9.566386],[3.560291,3.343891],[4.824702,-8.422228],[9.906842,-9.325841],[8.797862,-2.162496],[8.082681,5.574345],[-1.079138,0.649107],[7.178898,-8.363858],[3.700221,-8.470060],[2.548182,-6.956911],[-5.240126,-2.146242],[-2.141948,-0.616502],[-3.834835,5.279157],[8.366304,3.400275],[7.125470,-8.854391],[5.409646,-3.053361],[-5.090837,7.727295],[5.491554,-8.218231],[-7.593129,1.136625],[-6.966756,-0.439135],[3.206762,-7.554045],[-2.645144,-6.174378],[-3.052709,-3.640399],[-0.798664,-5.764968],[-7.033213,-2.466626],[-7.354647,4.363528],[7.964749,2.551573],[-6.042890,-7.202001],[5.653139,-8.176061],[0.841336,4.164538],[6.871590,6.206333],[6.156914,3.670982],[9.034408,4.394734],[-6.630740,-9.421933],[8.043266,0.566812],[2.329851,4.292548],[0.852286,-9.849935],[9.974711,-3.442936],[-0.027518,-4.734108],[-5.595936,4.433332],[8.751304,9.371349],[1.197170,-0.997188],[9.660027,-2.687621],[0.791322,8.671609],[0.500626,7.782142],[7.123809,-8.971027],[4.869104,1.839822],[-3.151817,6.183235],[8.419553,4.167992],[-7.379553,-2.916954],[-2.879166,-8.263426],[-2.403890,-2.509143],[-6.130416,-7.078596],[9.577508,7.088613],[0.753363,-6.349423],[8.796519,-1.652999],[-5.538079,2.020136],[0.229634,-5.706978],[-4.632951,2.122064],[-4.947306,-7.760554],[-7.933636,2.174592],[1.246072,-9.701073],[5.614398,-5.279928],[5.901975,-0.319377],[1.729483,-3.554111],[0.522481,-2.965729],[-0.768222,-8.804845],[9.338742,6.504255],[-8.666529,7.218267],[6.682547,-4.049599],[0.819035,8.009486],[0.946536,-2.555558],[2.693516,4.529769],[-7.096204,-7.004067],[-0.261847,3.913249]], dtype = "float64")#candidate|5005|(108, 2)|const|float64
call_5004 = func_2676_call(relay.reshape(const_5005.astype('float64'), [12, 3, 6]))
call_5006 = func_2676_call(relay.reshape(const_5005.astype('float64'), [12, 3, 6]))
const_5010 = relay.const([[[-5.824409,6.086227,-6.351858,5.387501],[-7.080607,-5.450462,1.951604,-3.563847],[1.834481,-7.818497,5.644170,-3.553226],[-2.261986,6.356920,-5.745650,1.628962],[4.293935,-8.901101,3.982518,9.803527],[1.422146,2.434536,0.475510,1.185414]],[[0.364360,8.609805,9.970032,1.090857],[-9.892103,1.827690,4.058694,1.465326],[2.561289,-9.135756,3.211092,-1.063894],[1.030209,0.435383,-0.196868,-1.882520],[-9.430015,-7.457517,9.403615,5.045531],[8.100966,-8.619117,6.134775,5.129566]],[[8.224355,0.642598,7.981797,-0.500916],[2.939826,-4.506099,-0.582340,9.315943],[-9.291020,2.333338,3.696304,1.555018],[5.644340,1.913663,3.597648,0.932635],[-8.543964,5.816091,7.272051,3.964427],[6.219995,-7.712683,-8.115457,0.506757]],[[2.791792,-8.555758,5.127127,2.471575],[0.600997,-8.887688,-8.154257,-5.106337],[-0.657574,-7.278647,0.062953,2.112300],[4.012426,-9.393726,5.121692,0.964597],[-9.174950,-6.385770,0.305556,-7.444791],[-1.684687,-5.601503,7.875154,-8.218019]],[[-3.287945,7.339007,-3.166574,2.403671],[-1.997337,2.834818,7.609258,-9.847939],[-2.363246,4.965395,1.033174,-6.320743],[-3.123566,-2.980424,5.395995,-0.208443],[1.278592,5.516051,-5.708381,-2.170059],[9.322181,-1.746860,-5.293840,-9.589294]],[[3.564747,-4.660752,-2.130003,2.603219],[4.461320,-9.344400,-6.528324,9.230793],[-6.036839,8.848265,1.368334,8.032819],[7.930139,9.233205,-6.626814,-6.859654],[8.281690,-3.582892,0.557724,2.517862],[5.652152,-8.330844,-3.572336,-9.375456]],[[-7.934208,5.702242,6.726278,-6.932839],[-8.631415,-3.292482,-2.087906,4.296416],[8.098763,5.742401,0.995459,-0.978109],[-1.554643,-6.447878,-3.472489,7.820810],[-0.411219,-5.007466,8.176630,-5.787715],[-5.105425,-4.518604,0.834785,-5.741168]],[[-3.155994,6.914012,-4.987989,-3.506851],[-6.713995,3.034664,2.850166,-0.203976],[2.372394,8.163559,5.141207,4.456050],[-2.025766,-3.893263,2.656443,5.078734],[-8.416202,0.703979,-8.439946,8.074187],[-0.442319,8.830126,2.483429,5.909814]]], dtype = "float64")#candidate|5010|(8, 6, 4)|const|float64
bop_5011 = relay.left_shift(uop_4999.astype('int8'), relay.reshape(const_5010.astype('int8'), relay.shape_of(uop_4999))) # shape=(8, 6, 4)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_5014 = relay.TupleGetItem(func_3866_call(), 0)
call_5015 = relay.TupleGetItem(func_3867_call(), 0)
uop_5018 = relay.tan(bop_5011.astype('float64')) # shape=(8, 6, 4)
output = relay.Tuple([call_4937,call_4940,var_4941,call_4961,var_4962,call_4967,call_4976,const_4977,const_4978,call_4984,call_4991,const_4992,bop_4995,call_5001,var_5002,call_5004,const_5005,call_5014,uop_5018,])
output2 = relay.Tuple([call_4938,call_4942,var_4941,call_4963,var_4962,call_4968,call_4979,const_4977,const_4978,call_4985,call_4993,const_4992,bop_4995,call_5003,var_5002,call_5006,const_5005,call_5015,uop_5018,])
func_5021 = relay.Function([var_4932,var_4941,var_4962,var_5002,], output)
mod['func_5021'] = func_5021
mod = relay.transform.InferType()(mod)
var_5022 = relay.var("var_5022", dtype = "float32", shape = (8, 6, 4))#candidate|5022|(8, 6, 4)|var|float32
var_5023 = relay.var("var_5023", dtype = "uint8", shape = (144, 1))#candidate|5023|(144, 1)|var|uint8
var_5024 = relay.var("var_5024", dtype = "uint16", shape = (1260,))#candidate|5024|(1260,)|var|uint16
var_5025 = relay.var("var_5025", dtype = "float64", shape = (140,))#candidate|5025|(140,)|var|float64
output = func_5021(var_5022,var_5023,var_5024,var_5025,)
func_5026 = relay.Function([var_5022,var_5023,var_5024,var_5025,], output)
mutated_mod['func_5026'] = func_5026
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4265_call = mod.get_global_var('func_4265')
func_4267_call = mutated_mod.get_global_var('func_4267')
call_5077 = func_4265_call()
call_5078 = func_4265_call()
func_4342_call = mod.get_global_var('func_4342')
func_4346_call = mutated_mod.get_global_var('func_4346')
const_5082 = relay.const([8.715287,-2.768699,-1.363680,8.644413,5.043794,-2.800824], dtype = "float32")#candidate|5082|(6,)|const|float32
const_5083 = relay.const([6.318219,-9.880335,4.530870,-8.581113,5.382253,-3.949220,1.414698,7.452979,-3.118904,-7.729292,-6.945845,-1.700925,-2.013992,9.596642,7.823000,2.473104,-2.313410,-8.173395,3.765372,7.339995,7.422376,-4.368835,3.616232,5.267294,1.851834,-2.002852,9.713713,8.363910,-9.169580,-2.083191,9.707591,-3.217791,3.581761,5.539871,4.774321,3.673698,-0.514751,8.811222,-2.046066,1.782738,-4.512927,9.111081,-2.159590,2.784219,8.783529,-1.039775,3.866026,-8.424123,-8.344381,-2.738566,-9.702206,-6.481038,4.241606,-4.446694,-3.095324,6.166922,-4.826955,3.348257,1.863536,7.748854,-8.754631,4.319359,-3.981182,-5.356759,2.738722,-4.482951,0.799351,-9.008565,-4.480383,-5.785231,2.357433,-2.935686,-2.776901,-2.574406,-8.491213,-0.909897,-3.879021,8.334584,9.233612,5.897133,-6.818012,-8.791740,-0.263698,-8.160176,9.611132,5.804230,-8.113760,-7.736435,4.364244,-5.200939,-3.486485,-2.251415,-4.533766,1.371906,-7.220032,-4.858100,-1.922535,-6.861295,1.636325,-6.294374,2.284638,0.497266,8.582310,-9.672887,0.123049,-1.744427,5.553731,-5.870394,-9.566191,9.045784,-1.462028,4.082233,-3.971166,0.816670,-6.104374,-9.962464,5.362423,2.061159,-2.064462,-8.951526,-0.234875,7.957683,4.675577,0.300494,-1.282231,-9.528651,-9.729550,8.353536,7.008980,8.433544,-9.119509,1.749150,9.026588,-7.369804,-7.641038,1.030613,-1.685156,-4.762854,-2.904848,3.907521,-0.450827,9.781366,-5.505781,7.340408,2.975520,5.279529,4.414946,-6.123382,-4.809577,-2.123692,1.709705,9.960309,-9.427169,6.173549,-6.017056,3.813791,2.060971,2.251250,-3.917513,1.678634,2.167248,7.528525,-7.295096,-6.985165,5.842549,5.277682,-4.195832,-0.033081,6.207966,-3.384038,-8.698423,-6.665268,-6.150196,6.737914,0.629167,9.596676,-4.792785,2.713535,-5.184544,8.120106,-7.131636,-8.765165,4.340438,0.252786,-2.892023,-1.441477,-7.542115,-9.963332,9.823667,0.678975,4.605807,9.464453,9.905915,-0.931994,-8.730953,6.359880,-3.824451,-3.523039,-8.407464,-2.649550,3.724259,-4.102180,-7.169504,-3.211772,-9.360277,-4.303923,-8.906625,0.746888,-9.432731,9.635593,7.554849,4.898269,-8.474470,-7.763766,1.578715,-9.402639,9.866573,-5.695539,6.942975,6.245510,-4.731776,0.379322,-3.132988,8.824989,-2.684236,4.246573,5.870592,-3.338431,-2.599694,6.116562,2.887181,-3.613510,-4.607591,-1.175065,-3.211598,4.807266,1.383617,6.385171,-1.303988,-6.867428,-4.134309,-7.429525,-2.643940,3.677888,0.367287,8.943643,-6.908956,-9.512549,-8.013091,4.231376,-6.317617,-0.542269,4.946966,8.680364,-5.839080,8.030191,-1.096314,-8.687777,-8.328631,5.750370,-1.855976,-3.104408,2.203938,7.745151,-7.871809,-0.516692,-5.069234,-8.828501,-4.785045,5.675126,-0.712491,0.996021,-7.712109,3.110841,-8.451221,2.729235,0.425558,9.461771,-3.236744,9.508227,5.888335,5.828588,-6.104237,-4.814290,-3.221970,6.672823,-1.380324,4.110289,-5.031230,3.878822,3.951845,-3.868609,-8.457990,1.965829,-4.397628,4.057730,-3.395228,0.493151,7.114757,-6.461595,2.463379,-8.286252,-9.854084,3.789650,-1.657999,8.924554,2.813949,1.423020,-7.734188,-3.546440,2.904324,-7.307016,-1.184878,-7.762603,7.403838,2.031902,-7.374333,-6.971714,2.246452,0.532681,-8.245212,-3.661511,-9.927191,-8.571345,-9.418640], dtype = "float64")#candidate|5083|(325,)|const|float64
call_5081 = relay.TupleGetItem(func_4342_call(relay.reshape(call_5077.astype('float32'), [14, 14, 2]), relay.reshape(const_5082.astype('float32'), [6,]), relay.reshape(const_5083.astype('float64'), [325,]), ), 5)
call_5084 = relay.TupleGetItem(func_4346_call(relay.reshape(call_5077.astype('float32'), [14, 14, 2]), relay.reshape(const_5082.astype('float32'), [6,]), relay.reshape(const_5083.astype('float64'), [325,]), ), 5)
func_1809_call = mod.get_global_var('func_1809')
func_1811_call = mutated_mod.get_global_var('func_1811')
const_5090 = relay.const([-7,-1,-7,5,2,9,1,3,-1,4,9,-2,-8,4,-5], dtype = "int16")#candidate|5090|(15,)|const|int16
call_5089 = relay.TupleGetItem(func_1809_call(relay.reshape(const_5090.astype('int16'), [1, 5, 3])), 0)
call_5091 = relay.TupleGetItem(func_1811_call(relay.reshape(const_5090.astype('int16'), [1, 5, 3])), 0)
const_5105 = relay.const([[[2.631850,-1.291965],[-6.486926,8.518886],[-0.077234,-3.957651],[9.480480,-7.412409],[4.211461,9.727921],[-9.263063,3.101559],[7.163755,-3.032723],[7.930257,1.728916],[7.346480,-0.552121],[9.027527,9.647197],[-4.329788,1.499388],[-9.549165,-9.809952],[-2.407229,4.492134],[-7.717636,6.795102]],[[1.153600,7.236984],[2.305442,-9.516180],[-4.872857,-3.945488],[-8.112436,9.141783],[3.736098,-7.265299],[3.503329,-1.127929],[-7.020507,-6.703515],[-0.550363,-3.045677],[4.980740,-7.789571],[4.187986,-8.409062],[6.761633,4.002920],[-8.743008,0.769483],[3.851879,-6.579416],[-5.037231,-6.877311]],[[-6.371278,6.569548],[7.638960,-7.579111],[7.935400,6.933847],[6.686174,9.874778],[6.084190,7.634473],[9.500534,5.678086],[-0.876806,0.840446],[-4.081662,-0.640215],[7.664201,-0.755814],[0.215774,3.898170],[6.918226,9.157422],[-0.025415,-4.393700],[-8.421608,-0.353303],[-6.226780,-7.507820]],[[7.165892,1.387048],[-9.136533,-5.602679],[7.196776,-3.762763],[4.565664,-1.742854],[-2.964300,-1.983461],[-1.505792,-7.685162],[-1.447221,-7.646136],[5.241918,8.091454],[1.652484,-5.272065],[-5.814519,1.529223],[7.273454,7.329679],[7.303192,-8.018183],[2.508183,5.722303],[3.914842,-8.003189]],[[-1.617919,-9.331336],[-7.651306,-4.180175],[-1.733257,7.308955],[4.435531,-4.398549],[-4.946215,7.949625],[5.032511,5.782675],[-1.103351,-6.161552],[0.493247,9.252111],[-0.676849,6.275224],[-6.091816,-1.707390],[-4.275759,0.803944],[-0.890535,9.084099],[7.349837,-4.058409],[-2.337927,-1.830215]],[[-1.350397,-7.528198],[6.893257,9.171234],[1.174539,-6.343781],[-4.706373,-9.399600],[-8.072114,-4.952849],[6.206542,0.543344],[5.111098,-8.927196],[-2.379323,-9.284455],[6.605731,-8.115257],[9.272523,-0.766728],[-6.321607,-5.767331],[5.275624,4.140887],[-5.486069,-0.798635],[8.416685,3.377423]],[[-0.782767,-5.483584],[-5.786800,7.743867],[3.092247,-4.102405],[-1.523199,9.102229],[-8.252340,3.102226],[-1.578315,6.536832],[8.317012,3.015085],[-2.093364,8.947878],[-2.146749,6.214288],[-1.579118,-5.485167],[-5.743132,-3.704747],[-8.795906,8.606856],[9.425696,-2.396208],[5.025430,9.826817]],[[-3.802865,-3.806701],[9.417929,5.874731],[5.492642,-1.043811],[6.930370,-2.332122],[5.137008,-9.261857],[-5.301836,-7.553535],[3.657339,-8.948597],[1.599514,8.519369],[-4.705579,-6.161647],[0.489541,5.224295],[-8.720784,-0.022379],[-9.726907,9.146573],[-6.816425,8.279335],[2.125743,-8.879991]],[[9.123905,8.962893],[-4.176836,-7.835993],[-7.626402,-4.676995],[-2.264866,1.615842],[3.598729,-3.002671],[-5.380558,-5.647955],[-9.761705,2.970410],[9.073988,-6.660282],[-0.429404,0.202627],[-4.518286,7.266760],[-6.206574,-5.296557],[9.159292,6.943413],[1.067049,1.151575],[7.676471,-2.557645]],[[7.790718,0.276745],[-0.348923,8.479750],[-7.987826,6.916760],[-7.861796,2.734136],[0.889232,3.646717],[0.685864,9.548271],[8.508362,-0.037320],[-2.822595,-4.910627],[0.980413,7.288876],[2.480223,-5.237315],[2.404648,-9.905121],[5.099136,4.353369],[9.299899,2.911362],[1.722817,6.358148]],[[6.180748,-1.588089],[3.886190,-7.788295],[-6.981700,-8.159259],[-8.246227,9.315376],[-6.404215,-0.088248],[-9.807712,-0.183162],[-1.872305,-6.818664],[0.098196,-7.281247],[-2.828586,-7.059105],[3.312496,-9.533380],[-9.193181,1.853375],[1.004808,0.322954],[-0.254619,5.895403],[5.772219,-8.244714]],[[-4.536955,5.643375],[3.752933,-3.674645],[-6.686596,1.590338],[6.726529,-7.321018],[-4.621827,7.915841],[-4.449769,8.719688],[-7.828964,3.753064],[6.743005,4.254058],[5.991618,2.223999],[4.222238,-8.302331],[-9.593882,7.169361],[5.912426,-0.724997],[-4.178944,4.168980],[-3.601586,2.738151]],[[-3.929536,-3.556768],[7.047716,4.336547],[-1.273994,8.052188],[-2.219582,-3.909604],[7.437148,6.051478],[1.613369,-4.522135],[5.385872,-5.579319],[-0.574578,-6.337580],[3.786483,-0.715200],[-0.843286,2.272922],[-8.913651,-3.905241],[-7.230218,-4.628339],[3.370827,-9.169330],[1.428835,-6.313739]],[[-4.562537,-7.874259],[-2.683664,6.622557],[8.066147,-6.475281],[1.447011,-9.271112],[-8.212371,8.756369],[9.185889,-8.757375],[9.707315,1.945898],[1.577614,7.332243],[4.263948,3.759750],[-0.980141,2.342651],[-8.214559,1.497073],[2.541683,-8.191146],[3.321682,-9.162881],[5.284333,6.625366]]], dtype = "float32")#candidate|5105|(14, 14, 2)|const|float32
bop_5106 = relay.mod(call_5077.astype('float64'), relay.reshape(const_5105.astype('float64'), relay.shape_of(call_5077))) # shape=(14, 14, 2)
bop_5109 = relay.mod(call_5078.astype('float64'), relay.reshape(const_5105.astype('float64'), relay.shape_of(call_5078))) # shape=(14, 14, 2)
output = relay.Tuple([call_5081,const_5082,const_5083,call_5089,const_5090,bop_5106,])
output2 = relay.Tuple([call_5084,const_5082,const_5083,call_5091,const_5090,bop_5109,])
func_5115 = relay.Function([], output)
mod['func_5115'] = func_5115
mod = relay.transform.InferType()(mod)
output = func_5115()
func_5116 = relay.Function([], output)
mutated_mod['func_5116'] = func_5116
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_5132 = func_3786_call()
call_5133 = func_3786_call()
output = relay.Tuple([call_5132,])
output2 = relay.Tuple([call_5133,])
func_5139 = relay.Function([], output)
mod['func_5139'] = func_5139
mod = relay.transform.InferType()(mod)
mutated_mod['func_5139'] = func_5139
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5139_call = mutated_mod.get_global_var('func_5139')
call_5140 = func_5139_call()
output = call_5140
func_5141 = relay.Function([], output)
mutated_mod['func_5141'] = func_5141
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5160 = relay.var("var_5160", dtype = "float64", shape = (3, 2, 10))#candidate|5160|(3, 2, 10)|var|float64
uop_5161 = relay.cos(var_5160.astype('float64')) # shape=(3, 2, 10)
bop_5170 = relay.bitwise_and(uop_5161.astype('uint16'), relay.reshape(var_5160.astype('uint16'), relay.shape_of(uop_5161))) # shape=(3, 2, 10)
func_1498_call = mod.get_global_var('func_1498')
func_1501_call = mutated_mod.get_global_var('func_1501')
var_5188 = relay.var("var_5188", dtype = "int32", shape = (140,))#candidate|5188|(140,)|var|int32
call_5187 = func_1498_call(relay.reshape(var_5188.astype('int32'), [10, 14, 1]))
call_5189 = func_1498_call(relay.reshape(var_5188.astype('int32'), [10, 14, 1]))
func_2779_call = mod.get_global_var('func_2779')
func_2786_call = mutated_mod.get_global_var('func_2786')
var_5197 = relay.var("var_5197", dtype = "int16", shape = (825,))#candidate|5197|(825,)|var|int16
var_5198 = relay.var("var_5198", dtype = "float32", shape = (180,))#candidate|5198|(180,)|var|float32
var_5199 = relay.var("var_5199", dtype = "float64", shape = (325,))#candidate|5199|(325,)|var|float64
var_5200 = relay.var("var_5200", dtype = "bool", shape = (550,))#candidate|5200|(550,)|var|bool
call_5196 = relay.TupleGetItem(func_2779_call(relay.reshape(var_5197.astype('int16'), [11, 15, 5]), relay.reshape(var_5197.astype('int16'), [11, 15, 5]), relay.reshape(var_5198.astype('float32'), [180, 1]), relay.reshape(var_5199.astype('float64'), [13, 25]), relay.reshape(var_5200.astype('bool'), [550,]), ), 2)
call_5201 = relay.TupleGetItem(func_2786_call(relay.reshape(var_5197.astype('int16'), [11, 15, 5]), relay.reshape(var_5197.astype('int16'), [11, 15, 5]), relay.reshape(var_5198.astype('float32'), [180, 1]), relay.reshape(var_5199.astype('float64'), [13, 25]), relay.reshape(var_5200.astype('bool'), [550,]), ), 2)
output = relay.Tuple([bop_5170,call_5187,var_5188,call_5196,var_5197,var_5198,var_5199,var_5200,])
output2 = relay.Tuple([bop_5170,call_5189,var_5188,call_5201,var_5197,var_5198,var_5199,var_5200,])
func_5208 = relay.Function([var_5160,var_5188,var_5197,var_5198,var_5199,var_5200,], output)
mod['func_5208'] = func_5208
mod = relay.transform.InferType()(mod)
var_5209 = relay.var("var_5209", dtype = "float64", shape = (3, 2, 10))#candidate|5209|(3, 2, 10)|var|float64
var_5210 = relay.var("var_5210", dtype = "int32", shape = (140,))#candidate|5210|(140,)|var|int32
var_5211 = relay.var("var_5211", dtype = "int16", shape = (825,))#candidate|5211|(825,)|var|int16
var_5212 = relay.var("var_5212", dtype = "float32", shape = (180,))#candidate|5212|(180,)|var|float32
var_5213 = relay.var("var_5213", dtype = "float64", shape = (325,))#candidate|5213|(325,)|var|float64
var_5214 = relay.var("var_5214", dtype = "bool", shape = (550,))#candidate|5214|(550,)|var|bool
output = func_5208(var_5209,var_5210,var_5211,var_5212,var_5213,var_5214,)
func_5215 = relay.Function([var_5209,var_5210,var_5211,var_5212,var_5213,var_5214,], output)
mutated_mod['func_5215'] = func_5215
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_5229 = relay.TupleGetItem(func_3866_call(), 2)
call_5230 = relay.TupleGetItem(func_3867_call(), 2)
func_4230_call = mod.get_global_var('func_4230')
func_4232_call = mutated_mod.get_global_var('func_4232')
call_5262 = func_4230_call()
call_5263 = func_4230_call()
func_5021_call = mod.get_global_var('func_5021')
func_5026_call = mutated_mod.get_global_var('func_5026')
const_5268 = relay.const([[7.751464,-4.820877,0.560347,3.090497,4.281845,-8.723913,-4.258944,1.528797,2.016624,-9.377053,-8.175289,7.139432,9.041227,9.150348,-6.686365,-0.413294,-6.394749,-2.909455,-2.581054,8.289325,3.259642,-2.897636,5.234631,0.884921,-7.236714,-4.728166,8.677887,3.427128,-2.661070,-7.844935,-3.204842,7.704036,-2.230956,-3.688524,9.261991,-7.091922,-6.779574,-8.417657,-8.356270,-0.794812,6.312639,-0.127436,-9.925682,0.493558,-2.246020,-7.477755,5.455733,-3.042117,7.197140,-0.323946,-2.105126,-9.219111,-3.315516,-1.113365,-6.097071,6.481940,0.103036,-0.895055,6.823396,-3.379719,3.153535,-9.715358,-2.644643,-1.448007,-6.095965,7.514258,8.584106,9.875760,-5.124077,3.470919,-8.930329,3.205885,-1.358294,-4.256349,1.935134,9.727535,9.085381,8.407787,-9.635502,9.589832,5.643064,4.598227,-9.085126,-1.513511,-2.636037,7.036414,-7.330047,-3.321189,-6.218403,-6.006030,-0.817521,9.201984,-1.256840,5.939006,-9.420096,8.798295,6.530279,8.583041,4.688558,9.532835,6.014806,-1.239188,2.381353,-4.667843,3.048548,1.567261,-7.534329,4.288643,-6.483042,-0.391478,-9.096349,-7.211821,-7.105234,0.416807,5.322854,-4.607443,-2.291798,-3.269205,0.808945,1.939253,3.106197,-4.997470,3.159090,-1.103357,6.050370,7.567722,-3.658049,2.251071,8.758598,-4.535929,3.421295,7.979636,-6.356859,7.296096,-8.323353,-0.274177,-7.191356,0.366816,4.602471,8.806541,-9.938920,-8.963786,-0.547893,-1.174910,-6.136840,-9.458393,7.702360,1.138619,0.109561,2.320031,-2.373213,2.273551,6.875116,-8.872315,7.669736,9.889326,-9.117494,-3.963242,3.741292,-2.795424,-1.028252,-8.847443,-3.730463,8.177617,-2.297816,4.741108,-7.106375,4.590870,-8.133576,-8.831443,2.710667,6.597762,7.554245,-9.311820,-2.851109,4.450165,2.444555,-1.882152,4.364649,5.799490,1.701587,-1.183130,-9.131835,9.648152,-0.602455,-0.415007,4.619821,4.637612,9.036511,3.636117,1.271726,-4.497972]], dtype = "float32")#candidate|5268|(1, 192)|const|float32
var_5269 = relay.var("var_5269", dtype = "uint8", shape = (72, 2))#candidate|5269|(72, 2)|var|uint8
var_5270 = relay.var("var_5270", dtype = "uint16", shape = (1260,))#candidate|5270|(1260,)|var|uint16
var_5271 = relay.var("var_5271", dtype = "float64", shape = (140,))#candidate|5271|(140,)|var|float64
call_5267 = relay.TupleGetItem(func_5021_call(relay.reshape(const_5268.astype('float32'), [8, 6, 4]), relay.reshape(var_5269.astype('uint8'), [144, 1]), relay.reshape(var_5270.astype('uint16'), [1260,]), relay.reshape(var_5271.astype('float64'), [140,]), ), 1)
call_5272 = relay.TupleGetItem(func_5026_call(relay.reshape(const_5268.astype('float32'), [8, 6, 4]), relay.reshape(var_5269.astype('uint8'), [144, 1]), relay.reshape(var_5270.astype('uint16'), [1260,]), relay.reshape(var_5271.astype('float64'), [140,]), ), 1)
output = relay.Tuple([call_5229,call_5262,call_5267,const_5268,var_5269,var_5270,var_5271,])
output2 = relay.Tuple([call_5230,call_5263,call_5272,const_5268,var_5269,var_5270,var_5271,])
func_5281 = relay.Function([var_5269,var_5270,var_5271,], output)
mod['func_5281'] = func_5281
mod = relay.transform.InferType()(mod)
mutated_mod['func_5281'] = func_5281
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5281_call = mutated_mod.get_global_var('func_5281')
var_5283 = relay.var("var_5283", dtype = "uint8", shape = (72, 2))#candidate|5283|(72, 2)|var|uint8
var_5284 = relay.var("var_5284", dtype = "uint16", shape = (1260,))#candidate|5284|(1260,)|var|uint16
var_5285 = relay.var("var_5285", dtype = "float64", shape = (140,))#candidate|5285|(140,)|var|float64
call_5282 = func_5281_call(var_5283,var_5284,var_5285,)
output = call_5282
func_5286 = relay.Function([var_5283,var_5284,var_5285,], output)
mutated_mod['func_5286'] = func_5286
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5139_call = mod.get_global_var('func_5139')
func_5141_call = mutated_mod.get_global_var('func_5141')
call_5368 = relay.TupleGetItem(func_5139_call(), 0)
call_5369 = relay.TupleGetItem(func_5141_call(), 0)
const_5370 = relay.const([[[3.562079,1.791852],[7.047413,4.648229],[7.795233,-1.943976],[-8.532107,8.543865],[3.732148,-3.921716],[-8.320569,-8.887176],[-7.740629,3.234766],[-6.784988,4.413280],[-5.239052,-4.038326],[8.212026,3.053283],[-9.724510,-1.927472],[0.248568,7.351397],[8.106414,2.556247],[7.462110,2.256703]],[[4.374110,7.058064],[1.106459,0.648853],[7.152673,0.173352],[9.621754,0.034628],[-7.894222,-4.303888],[1.711860,0.810125],[-1.654059,4.345120],[-6.970734,8.628726],[6.892974,6.052530],[-8.130260,-1.462328],[-1.563822,-9.961484],[6.906518,8.967342],[9.427893,7.471668],[-3.268659,8.863242]],[[6.286303,-4.824999],[4.625119,9.026513],[-5.652183,5.106716],[4.189135,-1.885128],[0.703183,-4.341167],[5.848600,3.366853],[9.406906,0.122523],[6.821078,-8.434561],[3.318025,-0.374692],[0.717142,9.789217],[-9.404076,-4.104745],[2.215528,2.980180],[7.368592,1.003096],[2.481975,-8.810973]],[[4.952932,-0.245857],[-2.536839,1.398310],[-0.207966,1.550936],[-7.528009,3.287432],[5.102977,-0.874374],[6.838241,-3.688294],[0.304907,8.977940],[-4.347940,-3.018230],[0.105440,-1.810047],[7.462725,-8.828047],[2.476693,8.011376],[-5.400950,-3.666168],[-5.984510,-7.225286],[-4.405595,7.928827]],[[-7.582560,-1.321823],[-8.528297,-5.627487],[-3.991958,4.135367],[3.411536,8.819751],[2.179397,-9.747919],[5.880702,-0.753926],[3.196298,3.350320],[0.712625,-1.161104],[0.900007,-4.143517],[8.839840,-8.276609],[4.088042,7.715192],[4.373054,9.556408],[-7.427564,-5.195990],[-9.845046,9.716052]],[[-7.990478,4.185736],[8.527926,5.531568],[-7.977058,-1.681239],[2.758323,2.341277],[2.487202,3.983127],[2.591393,1.876248],[-3.891237,-4.535130],[-5.333397,-4.305508],[-3.672729,-0.176149],[1.056677,1.019314],[9.610322,1.203295],[-8.300387,0.190878],[-8.310182,-7.331829],[1.574516,2.550107]],[[5.248241,-2.864244],[4.592229,-9.688915],[-7.550058,-4.781278],[3.901929,-8.528918],[6.799237,9.339606],[-8.203602,3.110724],[4.758126,7.893739],[-5.930471,-3.901189],[-9.055725,0.542987],[0.603852,3.423490],[7.754883,-1.185548],[9.379795,0.580259],[-2.613022,-7.895781],[-6.598396,8.918635]],[[6.333339,-3.111139],[8.433421,-9.939726],[-1.226167,-4.929251],[4.042453,8.021900],[0.944422,2.291634],[-1.287221,2.287018],[5.395414,-0.719142],[-1.593117,0.879799],[-9.013286,7.698345],[-0.383861,6.385663],[3.188664,3.266321],[2.885064,3.252445],[-7.259907,-9.451389],[-8.174280,0.241697]],[[3.093640,9.170707],[-7.973646,-2.940840],[-6.946784,-9.753860],[7.335682,4.720426],[4.268162,5.952405],[-9.444268,7.864463],[-3.495910,5.708745],[-3.785685,-2.342974],[-8.007938,4.419140],[2.530511,6.133488],[2.611848,-6.471423],[5.737237,0.955298],[-7.974869,-8.807999],[-4.794617,7.911452]],[[4.848694,8.706355],[-5.452425,9.712423],[7.457202,7.081645],[-4.840856,-5.569947],[-4.226397,-0.657966],[6.288427,2.906612],[8.499300,-1.466975],[6.703553,-4.336247],[-8.345334,-5.794545],[0.414436,-9.155077],[-8.067499,1.872064],[-7.632925,-6.447646],[3.008219,7.842955],[6.686984,-5.553592]],[[-6.321839,-3.685894],[7.272511,-1.457809],[-8.523228,-9.743365],[-0.436017,-6.917297],[-1.260716,-2.904463],[0.180579,9.168481],[2.698855,0.859413],[4.799703,1.747292],[-9.140473,-6.397613],[-4.866533,-2.009694],[0.279242,5.055173],[8.395835,8.543982],[5.197658,-4.520591],[-6.596401,3.034131]],[[-9.518148,-4.402424],[6.086344,2.858898],[5.854104,9.238554],[-0.142642,-2.142733],[-8.843935,-2.176378],[0.014452,6.635651],[5.193218,-8.830911],[6.681245,1.756143],[-8.153126,-8.081802],[-5.348926,-6.768930],[9.083076,4.282483],[6.742370,8.897605],[2.289879,9.479677],[7.179241,-2.151450]],[[-2.054151,4.180769],[-7.239500,-1.317427],[-3.574079,-3.595046],[-2.243960,8.813030],[-4.523108,-4.134344],[-5.858659,7.001988],[8.047697,0.877408],[0.645091,0.292545],[-5.278182,-7.991009],[2.559683,9.187876],[2.896817,-2.773115],[-6.654135,7.363967],[4.592451,-4.729792],[5.506508,3.970779]],[[-6.838616,-4.798791],[-4.432297,-9.084157],[-3.530247,-0.598354],[4.880520,6.219557],[0.692079,-8.616835],[8.671719,0.398848],[6.942822,6.018624],[0.936280,5.386725],[9.757476,-0.758368],[8.678679,-7.645003],[8.565635,-3.205157],[-2.635917,2.644142],[1.463919,-5.663335],[5.046023,8.122218]]], dtype = "float32")#candidate|5370|(14, 14, 2)|const|float32
bop_5371 = relay.maximum(call_5368.astype('int8'), relay.reshape(const_5370.astype('int8'), relay.shape_of(call_5368))) # shape=(14, 14, 2)
bop_5374 = relay.maximum(call_5369.astype('int8'), relay.reshape(const_5370.astype('int8'), relay.shape_of(call_5369))) # shape=(14, 14, 2)
func_5021_call = mod.get_global_var('func_5021')
func_5026_call = mutated_mod.get_global_var('func_5026')
var_5390 = relay.var("var_5390", dtype = "float32", shape = (192,))#candidate|5390|(192,)|var|float32
var_5391 = relay.var("var_5391", dtype = "uint8", shape = (144,))#candidate|5391|(144,)|var|uint8
var_5392 = relay.var("var_5392", dtype = "uint16", shape = (18, 70))#candidate|5392|(18, 70)|var|uint16
var_5393 = relay.var("var_5393", dtype = "float64", shape = (140,))#candidate|5393|(140,)|var|float64
call_5389 = relay.TupleGetItem(func_5021_call(relay.reshape(var_5390.astype('float32'), [8, 6, 4]), relay.reshape(var_5391.astype('uint8'), [144, 1]), relay.reshape(var_5392.astype('uint16'), [1260,]), relay.reshape(var_5393.astype('float64'), [140,]), ), 13)
call_5394 = relay.TupleGetItem(func_5026_call(relay.reshape(var_5390.astype('float32'), [8, 6, 4]), relay.reshape(var_5391.astype('uint8'), [144, 1]), relay.reshape(var_5392.astype('uint16'), [1260,]), relay.reshape(var_5393.astype('float64'), [140,]), ), 13)
output = relay.Tuple([bop_5371,call_5389,var_5390,var_5391,var_5392,var_5393,])
output2 = relay.Tuple([bop_5374,call_5394,var_5390,var_5391,var_5392,var_5393,])
func_5401 = relay.Function([var_5390,var_5391,var_5392,var_5393,], output)
mod['func_5401'] = func_5401
mod = relay.transform.InferType()(mod)
var_5402 = relay.var("var_5402", dtype = "float32", shape = (192,))#candidate|5402|(192,)|var|float32
var_5403 = relay.var("var_5403", dtype = "uint8", shape = (144,))#candidate|5403|(144,)|var|uint8
var_5404 = relay.var("var_5404", dtype = "uint16", shape = (18, 70))#candidate|5404|(18, 70)|var|uint16
var_5405 = relay.var("var_5405", dtype = "float64", shape = (140,))#candidate|5405|(140,)|var|float64
output = func_5401(var_5402,var_5403,var_5404,var_5405,)
func_5406 = relay.Function([var_5402,var_5403,var_5404,var_5405,], output)
mutated_mod['func_5406'] = func_5406
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_5438 = relay.TupleGetItem(func_3866_call(), 0)
call_5439 = relay.TupleGetItem(func_3867_call(), 0)
const_5455 = relay.const([[[-1.596407,5.691324],[-8.596676,3.782580],[2.208268,-1.495317],[-2.662027,-6.771657],[7.663430,-6.497396],[1.680186,7.699798],[-3.557798,1.611506],[3.818840,3.063605],[3.273462,-5.439133],[8.174907,-4.254524],[-5.715092,8.083054],[-3.170942,-3.376890],[6.788105,-7.130953],[4.057524,5.072255]],[[9.254724,5.869503],[-8.927461,-8.143521],[9.462320,8.754724],[-6.195571,-6.483093],[1.744252,-5.467068],[1.782063,2.864809],[6.540142,0.061131],[-7.845439,-3.397306],[2.424029,-4.789773],[4.917660,6.879887],[5.355038,7.676723],[1.413054,-0.254852],[8.455519,-8.371508],[-4.689427,4.945304]],[[0.478659,-5.478376],[-3.754036,7.333503],[7.631810,3.378283],[-4.769272,6.571941],[4.488154,8.670999],[-5.010967,6.676156],[-2.546928,-2.336062],[-7.669343,-2.476395],[-3.746192,5.847632],[-9.996028,8.004667],[-9.647800,-9.705913],[5.232663,-4.492835],[-1.158677,9.073522],[-1.729688,-1.982461]],[[-3.110419,1.120111],[9.578293,-6.722514],[-5.483832,-6.050882],[-4.445328,-2.201236],[-2.002851,3.774687],[0.628205,-1.120155],[-3.357004,-6.365176],[0.068996,8.130056],[-3.068030,-1.677699],[1.664924,1.865896],[-5.889177,-2.924447],[3.045211,3.749749],[5.559811,-7.525357],[-2.649263,-9.733323]],[[-9.911579,2.810938],[9.585550,-0.618383],[9.970150,2.987200],[9.028871,3.943068],[-8.722733,5.333853],[-2.539402,6.790845],[9.180572,-8.406330],[8.207085,-2.088208],[-3.109325,0.564376],[9.713485,1.632622],[-6.124866,5.998159],[2.584790,2.966204],[-5.624561,-3.018754],[7.821300,-8.407567]],[[9.717198,-3.209611],[9.747867,-1.947540],[8.846406,2.961558],[6.930067,7.850865],[-3.382278,-9.552147],[8.457845,0.249490],[-2.971083,6.859022],[-8.690905,-7.292991],[-3.918042,-4.860455],[-6.211696,6.366634],[-6.959758,-8.598229],[3.696765,3.236488],[-9.474878,-9.606526],[6.918330,8.534459]],[[1.400224,3.735450],[-2.597009,0.407318],[-2.809933,-5.330887],[-5.415619,-6.425765],[-4.883456,4.393214],[5.831937,3.633231],[-8.098608,6.937516],[-0.049615,9.003911],[9.968623,6.328668],[-8.723391,5.440391],[-6.992648,9.309515],[1.503060,9.451305],[-7.937835,-1.773399],[5.450254,7.483572]],[[0.186057,8.773863],[4.412541,0.809520],[8.329643,-3.130294],[9.399003,-4.340685],[2.065575,-8.654199],[-1.927293,7.442072],[-2.414928,-6.850900],[8.692019,3.858166],[-6.851062,-7.155673],[0.704705,6.394608],[-0.830662,2.640198],[3.524043,8.181811],[3.093991,2.267528],[9.360295,2.989516]],[[-3.743731,2.933821],[0.449499,-6.339422],[5.475358,-3.479556],[-7.658264,3.812827],[5.643649,-7.720666],[5.042353,-5.480921],[-9.706707,-0.709811],[-8.626032,1.731871],[2.045401,-3.506042],[-3.950148,-6.160149],[-7.016694,-4.300913],[-4.842721,8.368958],[-4.568796,-8.928442],[-0.373831,9.916295]],[[9.774632,-7.993338],[7.099387,-8.060817],[0.375401,0.362153],[-0.010270,2.904541],[-9.625317,5.833749],[-2.105909,0.766895],[8.929216,-7.022125],[-6.025935,-9.195715],[6.898653,5.735773],[-5.650583,6.617951],[-3.822314,3.171350],[-3.272310,2.043189],[5.426509,-0.274248],[-0.580340,-1.232585]],[[4.352750,6.089830],[-4.890160,1.578191],[-4.128797,-4.717310],[9.893981,-4.974721],[9.333454,-6.010864],[1.689649,-9.200365],[-5.701498,-1.295918],[-2.245629,-7.186676],[6.364303,9.668837],[5.015962,3.483812],[6.423982,3.441986],[0.936078,3.092701],[1.381789,-7.580051],[3.891295,3.110234]],[[7.632958,-9.118293],[8.222350,-0.548987],[-7.073265,-6.388880],[-0.914166,-7.724939],[-9.111751,-0.457696],[8.502531,-7.848560],[9.641539,9.989915],[0.187983,2.287107],[-6.740070,3.090563],[-9.846679,-3.680459],[5.086156,3.906924],[-7.267201,-2.127382],[5.022853,-2.574815],[-2.105223,0.228502]],[[-4.517087,-8.982395],[-4.744218,-9.781003],[-7.934531,-7.566232],[0.460482,3.536844],[1.568176,6.506614],[-1.551160,-9.093445],[-9.563461,2.103461],[5.139946,6.954665],[3.418753,-0.501484],[5.349823,4.749328],[-6.720875,-7.679741],[-7.099593,-6.985386],[-8.727504,-6.733169],[4.339169,-7.358055]],[[1.500998,-6.799059],[2.425536,-6.364892],[-1.064629,-6.159233],[-5.923874,5.966745],[6.678321,-6.269582],[2.311433,0.454226],[-7.430406,8.629154],[9.035167,-4.479524],[0.678949,8.211765],[-7.154848,5.192564],[-6.982946,-7.367128],[6.620100,2.087392],[1.515487,7.376391],[6.923498,0.118651]]], dtype = "float32")#candidate|5455|(14, 14, 2)|const|float32
bop_5456 = relay.multiply(call_5438.astype('int32'), relay.reshape(const_5455.astype('int32'), relay.shape_of(call_5438))) # shape=(14, 14, 2)
bop_5459 = relay.multiply(call_5439.astype('int32'), relay.reshape(const_5455.astype('int32'), relay.shape_of(call_5439))) # shape=(14, 14, 2)
func_1574_call = mod.get_global_var('func_1574')
func_1577_call = mutated_mod.get_global_var('func_1577')
var_5463 = relay.var("var_5463", dtype = "float32", shape = (120, 4))#candidate|5463|(120, 4)|var|float32
call_5462 = relay.TupleGetItem(func_1574_call(relay.reshape(var_5463.astype('float32'), [3, 16, 10])), 3)
call_5464 = relay.TupleGetItem(func_1577_call(relay.reshape(var_5463.astype('float32'), [3, 16, 10])), 3)
uop_5467 = relay.atan(var_5463.astype('float32')) # shape=(120, 4)
output = relay.Tuple([bop_5456,call_5462,uop_5467,])
output2 = relay.Tuple([bop_5459,call_5464,uop_5467,])
func_5471 = relay.Function([var_5463,], output)
mod['func_5471'] = func_5471
mod = relay.transform.InferType()(mod)
mutated_mod['func_5471'] = func_5471
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5472 = relay.var("var_5472", dtype = "float32", shape = (120, 4))#candidate|5472|(120, 4)|var|float32
func_5471_call = mutated_mod.get_global_var('func_5471')
call_5473 = func_5471_call(var_5472)
output = call_5473
func_5474 = relay.Function([var_5472], output)
mutated_mod['func_5474'] = func_5474
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4265_call = mod.get_global_var('func_4265')
func_4267_call = mutated_mod.get_global_var('func_4267')
call_5530 = func_4265_call()
call_5531 = func_4265_call()
output = call_5530
output2 = call_5531
func_5533 = relay.Function([], output)
mod['func_5533'] = func_5533
mod = relay.transform.InferType()(mod)
mutated_mod['func_5533'] = func_5533
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5533_call = mutated_mod.get_global_var('func_5533')
call_5534 = func_5533_call()
output = call_5534
func_5535 = relay.Function([], output)
mutated_mod['func_5535'] = func_5535
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5553 = relay.var("var_5553", dtype = "int32", shape = (6, 7, 1))#candidate|5553|(6, 7, 1)|var|int32
var_5554 = relay.var("var_5554", dtype = "int32", shape = (6, 7, 4))#candidate|5554|(6, 7, 4)|var|int32
bop_5555 = relay.subtract(var_5553.astype('int32'), var_5554.astype('int32')) # shape=(6, 7, 4)
uop_5558 = relay.tan(var_5553.astype('float32')) # shape=(6, 7, 1)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
const_5562 = relay.const([4.949912,-9.352482,-6.332423,-1.533848,-8.371396,-2.903672,6.953217,6.810046,-9.790804,8.180855,8.261217,-3.878928,2.906265,-1.114257,2.255492,-4.368535,-1.559450,-9.723023,5.054920,6.268793,3.162915,2.867574,3.379462,-2.119285,8.539744,0.895391,6.756432,0.837831,-8.456666,-5.453748,6.935978,-7.303012,5.860035,-2.340482,6.164974,-1.362628,4.513082,4.981894,3.789944,-7.482923,-8.744804,1.946355,-6.240904,-5.377323,1.082656,-4.630731,2.792968,-0.167072,8.426229,5.834332,-2.315288,5.578827,2.117963,6.378654,4.903502,4.899620,8.514682,1.206904,8.101975,-9.141894,-0.120083,0.552921,-0.033674,-1.182789,-2.671777,-5.743542,-9.578132,0.049692,6.639227,-5.854821,-7.106323,-1.915068,-0.645386,8.650096,2.856554,-3.836490,-1.297734,2.423281,2.366748,6.842898,0.660973,5.256637,9.197978,2.622953,4.761999,9.142606,4.860918,5.040919,-0.687321,9.736461,0.140065,5.089181,-0.749152,-2.066889,8.866574,3.515940,2.688404,-2.679621,0.245063,0.114025,7.941089,0.601953,-1.341486,-4.530869,5.107113,-5.185369,-6.380301,6.265615,-8.355207,-2.094576,8.567764,-6.116035,-9.389372,6.405671,-0.260585,-8.744183,-9.176891,-8.365977,9.724176,4.121624,1.051331,3.362342,6.785706,-6.522226,-4.578237,2.547280,-6.266115,-6.637120,-9.062932,0.309138,5.643478,-0.098517,8.889277,-0.957259,-3.412532,5.199879,-5.318633,2.683091,5.138252,4.251383,-2.503769,0.475149,4.987172,-9.635774,-2.252009,-7.704237,3.724902,2.257699,-7.365659,6.971801,3.718905,-5.993793,7.881728,1.754865,5.337203,-6.954424,-1.797158,-4.601302,-2.513694,5.674056,-3.971791,-7.844897,5.069074,-5.984737,5.697349,-3.982817,3.531739,-8.145062,-2.818665,1.294063,4.714206,1.217725,-1.912063,8.473533,-7.806614,2.729278,-6.427907,2.521016,3.042758,1.997778,2.326226,5.632982,-7.778095,-6.694043,2.362264,-3.916375,8.361508,-4.854331,-2.128782,-0.620230,8.703554,6.155103,4.277027,6.646319,4.512267,-1.037924,-2.256658,0.242017,8.226086,2.315434,6.343523,1.741164,3.174555,-8.143968,7.375720,5.286054,-9.525533,0.330265,7.451735,-2.084064,8.711125,1.410540,-0.217187,3.463800,-9.851157,0.949969,-1.226182,9.123722,-7.822772,2.733281,-6.420569,-9.221187,-1.037671,0.025967,-6.914726,-5.264893,-4.308533,2.446414,-7.005570,-1.213274,3.010688,-7.086430,5.534913,-9.204593,-0.916834,3.092117,-3.910391,1.720115,0.623424,-0.285173,0.416149,-4.848126,-2.622634,-3.575444,-9.316871,-7.455485,-5.513091,6.803826,-6.113216,-7.955404,-4.630386,6.132973,8.835459,4.912117,9.002780,4.028597,-9.449312,-0.085197,8.235130,1.222543,-2.775526,-0.472832,-3.450182,-9.052955,3.219788,7.961629,-0.031882,-2.023949,-4.042339,9.147510,3.690023,-2.989140,-9.068971,-0.014180,1.690536,1.694686,7.597125,0.635363,3.261844,-2.907996,-6.642511,-0.409365,5.142845,-7.709806,-2.848068,0.853727,-7.440192,4.752531,3.326792,9.832589,4.464031,-2.482938,-4.084814,8.629959,7.557180,-9.608361,9.507324,-7.269685,5.013719,-7.114427,-2.368634,7.471924,0.997733,6.274087,-3.292985,-8.886179,4.628097,7.316130,-8.168793,-7.767218,-8.933108,1.123739,8.390272,-7.054275,-0.165011,9.843784,3.104345,-0.973355,3.447677,1.086640,-1.473251,-4.725023,-9.789575,0.393432,-8.286859], dtype = "float64")#candidate|5562|(325,)|const|float64
call_5561 = func_418_call(relay.reshape(const_5562.astype('float64'), [13, 5, 5]))
call_5563 = func_418_call(relay.reshape(const_5562.astype('float64'), [13, 5, 5]))
const_5567 = relay.const([2.051282,2.926867,-6.595733,-2.404713,-9.977197,7.864815,-7.741210,-0.009186,-1.655304,-4.918095,-8.370044,6.516306,-7.620991,3.784860,-0.693587,-2.276887,7.693587,-9.213582,1.023688,4.749961,-1.516166,0.528369,-7.735387,6.861114,-7.393314,9.841109,-4.303720,3.773984,-2.988761,2.728337,-2.068637,-5.471972,3.899277,2.822429,-7.017224,-1.407318,7.219925,8.062037,7.688554,6.651905,-2.145654,3.586093,9.290518,9.732423,0.616072,-0.433058,8.704768,5.571092,4.090984,0.783336,2.615398,-4.261919,8.815306,-5.907131,-7.332850,4.403854,2.615820,5.811721,6.805778,-8.574862,2.299121,-3.076128,-4.536078,-3.798620,-7.503377,9.952429,4.674095,2.682596,4.088396,-7.986081,3.963198,3.699988,-9.329085,1.003402,4.983458,-2.240923,-1.646121,6.343587,-5.937644,-7.766958,-9.761188,-7.868108,-2.414604,-2.305106,2.905281,1.342809,-0.520525,6.121410,-7.895490,0.307651,1.334773,-2.821334,-4.685527,-2.396434,1.067047,-3.601139,-1.923417,-0.791049,-6.499081,-7.279221,1.228584,-3.180370,-0.595503,-0.534088,-5.037981,5.838042,6.008348,1.061328,3.543531,-5.369921,-8.815893,-8.454673,6.472837,9.691467,-4.990798,0.550140,-6.731012,9.120369,-5.642829,7.913705,3.484116,-0.609881,-1.596447,6.657661,9.431541,2.357678,-5.635712,-6.819047,-9.984508,3.760135,6.820604,-5.957021,-7.449382,5.256537,0.935216,1.292515,7.631175,2.991086,0.899989,4.572776,-2.085814,-6.113974,9.287429,-3.568910,5.466367,-6.374665,-0.902986,-3.843962,-0.594397,5.300049,-7.593407,2.582843,-7.832998,7.469649,8.323993,-7.447298,3.831229,-1.758655,9.701757,-0.941322,-4.908221,-8.422042,2.452222,2.955044,-9.382855,6.129080,7.864049,-9.117711,9.858607,9.417723,-4.212680,-3.298336,-0.442401,-5.125535,4.407901,-6.241122,2.972947,7.407029,3.690211,5.772673,7.109226,-4.905071,5.312871,9.483427,1.156531,1.505727,-3.651760,7.180377,-0.455898,6.251652,8.578867,4.565848,-1.629202,2.383878,6.422083,3.779036,-3.318581,2.237517,-5.755778,-4.531853,1.766942,7.759238,-1.966300,-8.687405,5.634545,-5.883188,-1.387846,6.617250,-6.377229,-8.008943,4.986772,0.110555,-2.472864,6.251211,6.211321,-7.336656,4.911017,-5.811757,-5.616527,-5.226577,-9.691799,-9.140873,-6.007227,1.245401,7.064011,-5.536307,6.671395,-2.630022,-5.930664,-1.797366,3.294230,-1.597890,4.162933,-2.417702,-9.579808,9.634751,-4.186044,7.341884,1.269588,6.448072,-6.355971,-9.238665,1.921707,1.287395,-2.128682,7.257352,-9.888936,8.664686,8.806517,-9.110130,-6.904193,-7.418774,4.509462,-9.811979,7.612534,5.942319,4.663720,2.605465,-1.938699,3.696979,1.525837,4.786294,-4.644084,6.946846,-6.696290,3.860003,9.989072,7.042509,4.283979,7.222231,-3.188508,4.345813,-4.071549,-9.910799,9.569112,-7.481069,9.705514,7.667418,-1.066377,7.489921,-1.717752,6.475972,-9.512455,-6.788145,-3.333698,-5.671608,1.729956,9.584890,4.980084,-1.375502,0.820854,9.120190,-5.492476,-4.851850,6.688100,-1.079114,9.247084,0.821710,-5.395923,9.924358,9.472249,5.953193,7.535841,7.597492,0.612156,-1.094732,-5.458976,0.405199,-3.966257,9.012256,7.388961,0.270686,5.886023,-1.484982,-8.627232,5.980782,-6.223997,-0.213412,0.129018,-0.688240,-8.313466,4.191271,-7.980993,9.548767,-8.495232], dtype = "float64")#candidate|5567|(325,)|const|float64
bop_5568 = relay.divide(const_5562.astype('float32'), relay.reshape(const_5567.astype('float32'), relay.shape_of(const_5562))) # shape=(325,)
func_418_call = mod.get_global_var('func_418')
func_421_call = mutated_mod.get_global_var('func_421')
call_5578 = func_418_call(relay.reshape(bop_5568.astype('float64'), [13, 5, 5]))
call_5579 = func_418_call(relay.reshape(bop_5568.astype('float64'), [13, 5, 5]))
uop_5586 = relay.cos(uop_5558.astype('float64')) # shape=(6, 7, 1)
output = relay.Tuple([bop_5555,call_5561,bop_5568,call_5578,uop_5586,])
output2 = relay.Tuple([bop_5555,call_5563,bop_5568,call_5579,uop_5586,])
func_5594 = relay.Function([var_5553,var_5554,], output)
mod['func_5594'] = func_5594
mod = relay.transform.InferType()(mod)
mutated_mod['func_5594'] = func_5594
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5594_call = mutated_mod.get_global_var('func_5594')
var_5596 = relay.var("var_5596", dtype = "int32", shape = (6, 7, 1))#candidate|5596|(6, 7, 1)|var|int32
var_5597 = relay.var("var_5597", dtype = "int32", shape = (6, 7, 4))#candidate|5597|(6, 7, 4)|var|int32
call_5595 = func_5594_call(var_5596,var_5597,)
output = call_5595
func_5598 = relay.Function([var_5596,var_5597,], output)
mutated_mod['func_5598'] = func_5598
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5533_call = mod.get_global_var('func_5533')
func_5535_call = mutated_mod.get_global_var('func_5535')
call_5613 = func_5533_call()
call_5614 = func_5533_call()
output = relay.Tuple([call_5613,])
output2 = relay.Tuple([call_5614,])
func_5634 = relay.Function([], output)
mod['func_5634'] = func_5634
mod = relay.transform.InferType()(mod)
output = func_5634()
func_5635 = relay.Function([], output)
mutated_mod['func_5635'] = func_5635
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5115_call = mod.get_global_var('func_5115')
func_5116_call = mutated_mod.get_global_var('func_5116')
call_5642 = relay.TupleGetItem(func_5115_call(), 4)
call_5643 = relay.TupleGetItem(func_5116_call(), 4)
func_4559_call = mod.get_global_var('func_4559')
func_4562_call = mutated_mod.get_global_var('func_4562')
var_5649 = relay.var("var_5649", dtype = "int16", shape = (1, 420))#candidate|5649|(1, 420)|var|int16
call_5648 = relay.TupleGetItem(func_4559_call(relay.reshape(var_5649.astype('int16'), [7, 5, 12]), relay.reshape(var_5649.astype('int16'), [7, 5, 12]), ), 0)
call_5650 = relay.TupleGetItem(func_4562_call(relay.reshape(var_5649.astype('int16'), [7, 5, 12]), relay.reshape(var_5649.astype('int16'), [7, 5, 12]), ), 0)
uop_5655 = relay.acosh(var_5649.astype('float32')) # shape=(1, 420)
output = relay.Tuple([call_5642,call_5648,uop_5655,])
output2 = relay.Tuple([call_5643,call_5650,uop_5655,])
func_5657 = relay.Function([var_5649,], output)
mod['func_5657'] = func_5657
mod = relay.transform.InferType()(mod)
mutated_mod['func_5657'] = func_5657
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5658 = relay.var("var_5658", dtype = "int16", shape = (1, 420))#candidate|5658|(1, 420)|var|int16
func_5657_call = mutated_mod.get_global_var('func_5657')
call_5659 = func_5657_call(var_5658)
output = call_5659
func_5660 = relay.Function([var_5658], output)
mutated_mod['func_5660'] = func_5660
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5668 = relay.var("var_5668", dtype = "uint32", shape = (13, 13, 6))#candidate|5668|(13, 13, 6)|var|uint32
var_5669 = relay.var("var_5669", dtype = "uint32", shape = (13, 13, 6))#candidate|5669|(13, 13, 6)|var|uint32
bop_5670 = relay.add(var_5668.astype('uint32'), relay.reshape(var_5669.astype('uint32'), relay.shape_of(var_5668))) # shape=(13, 13, 6)
bop_5673 = relay.maximum(var_5669.astype('int8'), relay.reshape(var_5668.astype('int8'), relay.shape_of(var_5669))) # shape=(13, 13, 6)
output = relay.Tuple([bop_5670,bop_5673,])
output2 = relay.Tuple([bop_5670,bop_5673,])
func_5676 = relay.Function([var_5668,var_5669,], output)
mod['func_5676'] = func_5676
mod = relay.transform.InferType()(mod)
mutated_mod['func_5676'] = func_5676
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5676_call = mutated_mod.get_global_var('func_5676')
var_5678 = relay.var("var_5678", dtype = "uint32", shape = (13, 13, 6))#candidate|5678|(13, 13, 6)|var|uint32
var_5679 = relay.var("var_5679", dtype = "uint32", shape = (13, 13, 6))#candidate|5679|(13, 13, 6)|var|uint32
call_5677 = func_5676_call(var_5678,var_5679,)
output = call_5677
func_5680 = relay.Function([var_5678,var_5679,], output)
mutated_mod['func_5680'] = func_5680
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4186_call = mod.get_global_var('func_4186')
func_4188_call = mutated_mod.get_global_var('func_4188')
call_5682 = relay.TupleGetItem(func_4186_call(), 0)
call_5683 = relay.TupleGetItem(func_4188_call(), 0)
output = call_5682
output2 = call_5683
func_5713 = relay.Function([], output)
mod['func_5713'] = func_5713
mod = relay.transform.InferType()(mod)
mutated_mod['func_5713'] = func_5713
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5713_call = mutated_mod.get_global_var('func_5713')
call_5714 = func_5713_call()
output = call_5714
func_5715 = relay.Function([], output)
mutated_mod['func_5715'] = func_5715
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4652_call = mod.get_global_var('func_4652')
func_4653_call = mutated_mod.get_global_var('func_4653')
call_5725 = relay.TupleGetItem(func_4652_call(), 0)
call_5726 = relay.TupleGetItem(func_4653_call(), 0)
func_3422_call = mod.get_global_var('func_3422')
func_3425_call = mutated_mod.get_global_var('func_3425')
var_5728 = relay.var("var_5728", dtype = "float32", shape = (1, 6))#candidate|5728|(1, 6)|var|float32
call_5727 = relay.TupleGetItem(func_3422_call(relay.reshape(var_5728.astype('float32'), [1, 6, 1])), 1)
call_5729 = relay.TupleGetItem(func_3425_call(relay.reshape(var_5728.astype('float32'), [1, 6, 1])), 1)
bop_5734 = relay.maximum(call_5727.astype('uint64'), relay.reshape(var_5728.astype('uint64'), relay.shape_of(call_5727))) # shape=(1, 6, 1)
bop_5737 = relay.maximum(call_5729.astype('uint64'), relay.reshape(var_5728.astype('uint64'), relay.shape_of(call_5729))) # shape=(1, 6, 1)
const_5743 = relay.const([[[-3,-4,5,-1,5],[1,2,-6,-6,6],[-9,2,6,5,2],[-2,6,-3,-4,-2],[-2,7,-2,4,4],[-1,-7,-9,5,-3]],[[-7,-2,-2,1,-1],[-7,8,4,6,8],[-10,-5,2,-7,9],[3,-6,-6,-2,-4],[-10,4,6,-7,-10],[4,-4,6,-4,-10]],[[-5,-6,8,9,1],[-2,6,-8,-6,7],[-2,5,-2,1,-5],[7,-5,-1,9,2],[-3,10,2,4,-4],[5,3,-3,10,1]],[[6,5,4,-4,-4],[-10,5,-6,7,5],[2,9,-10,2,-8],[-7,-2,-6,8,7],[8,-9,-7,-10,-5],[-8,-4,4,-1,6]],[[-9,4,5,6,-9],[5,4,2,9,8],[6,-10,-6,4,1],[7,-3,9,-2,-3],[-2,10,-2,-7,-2],[-2,-4,-9,-9,8]],[[-3,9,-7,-7,-7],[-9,9,-5,2,2],[9,-2,-8,-1,-4],[-6,5,6,-4,-2],[3,10,-1,-5,-6],[-6,-10,-10,-1,-3]],[[2,-5,-5,-1,-4],[-1,-2,-3,5,5],[4,8,2,-10,9],[-10,-5,9,6,10],[-1,-10,-2,-8,-10],[10,-6,8,8,-3]],[[-3,9,-7,-9,-6],[-6,-2,4,1,2],[-8,-7,-3,-4,-1],[10,-9,-2,-7,1],[10,2,6,-5,-9],[10,1,2,2,-7]],[[-1,9,-5,-6,1],[-6,7,-4,-6,-1],[8,-2,1,-5,10],[-6,-10,-10,-7,7],[-8,-1,-3,-7,4],[3,1,10,-9,-5]],[[-10,9,7,7,10],[1,9,2,-5,-10],[-8,10,8,3,7],[-6,8,-9,9,-8],[1,6,9,1,-8],[10,4,5,-2,-9]]], dtype = "uint64")#candidate|5743|(10, 6, 5)|const|uint64
bop_5744 = relay.add(bop_5734.astype('int8'), const_5743.astype('int8')) # shape=(10, 6, 5)
bop_5747 = relay.add(bop_5737.astype('int8'), const_5743.astype('int8')) # shape=(10, 6, 5)
func_1574_call = mod.get_global_var('func_1574')
func_1577_call = mutated_mod.get_global_var('func_1577')
var_5755 = relay.var("var_5755", dtype = "float32", shape = (480,))#candidate|5755|(480,)|var|float32
call_5754 = relay.TupleGetItem(func_1574_call(relay.reshape(var_5755.astype('float32'), [3, 16, 10])), 1)
call_5756 = relay.TupleGetItem(func_1577_call(relay.reshape(var_5755.astype('float32'), [3, 16, 10])), 1)
uop_5768 = relay.acosh(bop_5734.astype('float32')) # shape=(1, 6, 1)
uop_5770 = relay.acosh(bop_5737.astype('float32')) # shape=(1, 6, 1)
output = relay.Tuple([call_5725,bop_5744,call_5754,var_5755,uop_5768,])
output2 = relay.Tuple([call_5726,bop_5747,call_5756,var_5755,uop_5770,])
func_5772 = relay.Function([var_5728,var_5755,], output)
mod['func_5772'] = func_5772
mod = relay.transform.InferType()(mod)
var_5773 = relay.var("var_5773", dtype = "float32", shape = (1, 6))#candidate|5773|(1, 6)|var|float32
var_5774 = relay.var("var_5774", dtype = "float32", shape = (480,))#candidate|5774|(480,)|var|float32
output = func_5772(var_5773,var_5774,)
func_5775 = relay.Function([var_5773,var_5774,], output)
mutated_mod['func_5775'] = func_5775
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4186_call = mod.get_global_var('func_4186')
func_4188_call = mutated_mod.get_global_var('func_4188')
call_5801 = relay.TupleGetItem(func_4186_call(), 0)
call_5802 = relay.TupleGetItem(func_4188_call(), 0)
func_4462_call = mod.get_global_var('func_4462')
func_4464_call = mutated_mod.get_global_var('func_4464')
call_5811 = func_4462_call()
call_5812 = func_4462_call()
func_2687_call = mod.get_global_var('func_2687')
func_2691_call = mutated_mod.get_global_var('func_2691')
const_5820 = relay.const([1,-10,7,-9,-1,-9,2,7,-6,-5,8,1,-5,5,10,-1,-1,5,-6,-4,9,6,-3,-9,9,-5,9,6,-8,-8,-4,-3,-7,-6,5,1,-1,-3,-1,5,-10,6,8,-3,-4,9,7,4,10,-6,2,-9,6,9,-8,-9,5,-6,-9,3,10,-1,4,1,-7,-9,6,-2,-1,4,-7,8,10,7,4,1,2,1,9,-9,8,-5,-7,-1,-10,-10,7,1,1,-7,9,-9,1,-6,1,-7,3,-5,1,-1,-8,9,8,-5,7,-5,3,7,3,4,-9,3,3,6,-4,4,-9,-9,3,7,-7,8,6,4,1,3,9,5,4,5,-8,9,-1,4,2,4,4,1,-9,1,-3,1,-10,3,6,1,-4,-1,9,-9,-9,-4,6,4,8,-7,-2,2,-5,-8,-10,6,3,9,5,5,-10,6,4,-3,-7,-5,7,5,-2,-4,3,-8,-7,10,-4,7,-4,2,-5,2,5,3,7,-5,-10,-3,9,1,8,-10,-6,10,8,-10,9,6,-3,8,-6,1,2,-1,5,6], dtype = "uint8")#candidate|5820|(210,)|const|uint8
call_5819 = relay.TupleGetItem(func_2687_call(relay.reshape(const_5820.astype('uint8'), [14, 1, 15]), relay.reshape(const_5820.astype('uint8'), [14, 1, 15]), ), 0)
call_5821 = relay.TupleGetItem(func_2691_call(relay.reshape(const_5820.astype('uint8'), [14, 1, 15]), relay.reshape(const_5820.astype('uint8'), [14, 1, 15]), ), 0)
uop_5825 = relay.rsqrt(call_5819.astype('float64')) # shape=(14, 1, 15)
uop_5827 = relay.rsqrt(call_5821.astype('float64')) # shape=(14, 1, 15)
output = relay.Tuple([call_5801,call_5811,const_5820,uop_5825,])
output2 = relay.Tuple([call_5802,call_5812,const_5820,uop_5827,])
func_5831 = relay.Function([], output)
mod['func_5831'] = func_5831
mod = relay.transform.InferType()(mod)
output = func_5831()
func_5832 = relay.Function([], output)
mutated_mod['func_5832'] = func_5832
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5533_call = mod.get_global_var('func_5533')
func_5535_call = mutated_mod.get_global_var('func_5535')
call_5841 = func_5533_call()
call_5842 = func_5533_call()
output = call_5841
output2 = call_5842
func_5856 = relay.Function([], output)
mod['func_5856'] = func_5856
mod = relay.transform.InferType()(mod)
mutated_mod['func_5856'] = func_5856
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5856_call = mutated_mod.get_global_var('func_5856')
call_5857 = func_5856_call()
output = call_5857
func_5858 = relay.Function([], output)
mutated_mod['func_5858'] = func_5858
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5929 = relay.var("var_5929", dtype = "float32", shape = (4, 6))#candidate|5929|(4, 6)|var|float32
uop_5930 = relay.sqrt(var_5929.astype('float32')) # shape=(4, 6)
func_5594_call = mod.get_global_var('func_5594')
func_5598_call = mutated_mod.get_global_var('func_5598')
var_5952 = relay.var("var_5952", dtype = "int32", shape = (42,))#candidate|5952|(42,)|var|int32
var_5953 = relay.var("var_5953", dtype = "int32", shape = (168,))#candidate|5953|(168,)|var|int32
call_5951 = relay.TupleGetItem(func_5594_call(relay.reshape(var_5952.astype('int32'), [6, 7, 1]), relay.reshape(var_5953.astype('int32'), [6, 7, 4]), ), 2)
call_5954 = relay.TupleGetItem(func_5598_call(relay.reshape(var_5952.astype('int32'), [6, 7, 1]), relay.reshape(var_5953.astype('int32'), [6, 7, 4]), ), 2)
func_3866_call = mod.get_global_var('func_3866')
func_3867_call = mutated_mod.get_global_var('func_3867')
call_5959 = relay.TupleGetItem(func_3866_call(), 1)
call_5960 = relay.TupleGetItem(func_3867_call(), 1)
uop_5968 = relay.acos(uop_5930.astype('float32')) # shape=(4, 6)
output = relay.Tuple([call_5951,var_5952,var_5953,call_5959,uop_5968,])
output2 = relay.Tuple([call_5954,var_5952,var_5953,call_5960,uop_5968,])
func_5972 = relay.Function([var_5929,var_5952,var_5953,], output)
mod['func_5972'] = func_5972
mod = relay.transform.InferType()(mod)
mutated_mod['func_5972'] = func_5972
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5972_call = mutated_mod.get_global_var('func_5972')
var_5974 = relay.var("var_5974", dtype = "float32", shape = (4, 6))#candidate|5974|(4, 6)|var|float32
var_5975 = relay.var("var_5975", dtype = "int32", shape = (42,))#candidate|5975|(42,)|var|int32
var_5976 = relay.var("var_5976", dtype = "int32", shape = (168,))#candidate|5976|(168,)|var|int32
call_5973 = func_5972_call(var_5974,var_5975,var_5976,)
output = call_5973
func_5977 = relay.Function([var_5974,var_5975,var_5976,], output)
mutated_mod['func_5977'] = func_5977
mutated_mod = relay.transform.InferType()(mutated_mod)
var_5995 = relay.var("var_5995", dtype = "uint32", shape = (15, 12, 4))#candidate|5995|(15, 12, 4)|var|uint32
const_5996 = relay.const([[[-10,6,-7,-5],[-6,-6,8,1],[8,10,-8,6],[6,9,-6,-5],[1,-6,-8,3],[-3,-10,10,-4],[-9,-4,5,-4],[8,10,-10,10],[6,10,-8,9],[6,2,-3,2],[1,1,-6,8],[2,-9,2,-4]],[[9,4,8,9],[-3,10,7,9],[10,5,-7,-1],[10,-1,-1,-1],[-2,-8,-8,-6],[5,-8,10,2],[9,5,10,-7],[7,-1,5,-1],[-6,-7,3,7],[9,-4,-3,6],[-4,8,-8,2],[-2,-6,-6,-4]],[[-3,7,4,7],[9,2,-6,2],[1,3,2,7],[-10,-3,-9,9],[9,9,1,-9],[-7,-2,1,-4],[3,-6,-3,-1],[5,1,10,-7],[7,-4,-3,-1],[-6,-7,7,1],[-1,9,1,8],[5,-9,9,-5]],[[3,-7,2,7],[2,-5,5,1],[5,1,9,2],[-9,8,-9,6],[-2,-6,-2,-2],[-9,3,5,2],[-3,5,-2,-7],[-10,-7,-8,10],[-8,4,-1,6],[5,6,5,-9],[-2,-2,3,-3],[4,9,5,-4]],[[6,-4,-6,1],[-10,-3,-6,5],[9,2,9,-3],[5,7,-3,-4],[-2,-10,-1,-2],[-7,-7,-7,10],[-10,-8,-8,-1],[4,7,-2,-8],[1,3,-6,-4],[9,-7,2,2],[10,2,-2,-2],[-3,-4,2,-9]],[[10,5,-4,1],[10,10,5,-5],[-4,4,-10,-2],[7,-9,-10,2],[-8,-7,7,6],[6,-9,-1,-10],[9,-3,-2,3],[-1,-10,1,-10],[-10,1,-5,-3],[6,5,-4,2],[-9,-4,10,-2],[-2,-5,-2,-9]],[[-1,10,-9,10],[4,-3,-8,-4],[2,8,7,-6],[-2,-9,-5,7],[-1,6,9,-7],[1,3,-9,-6],[7,4,6,5],[2,-2,-4,-4],[5,9,8,-9],[6,-8,-1,5],[-2,3,2,4],[4,2,5,7]],[[8,8,-1,-4],[-10,-3,1,-10],[8,9,-3,-3],[-8,-7,1,-3],[-9,6,7,-7],[4,-8,8,6],[-1,-8,-9,-1],[8,-10,-1,9],[-3,10,-4,-8],[4,1,-6,-10],[3,3,-10,9],[-4,6,-1,7]],[[-7,5,-1,10],[-3,-1,-10,-5],[-10,-9,5,-8],[-6,-6,9,-2],[-10,8,1,9],[10,2,3,-7],[10,-3,3,-5],[6,-5,-5,-1],[-3,8,-8,-10],[9,10,-1,-6],[2,-5,-1,-3],[2,-3,-4,-3]],[[5,3,-8,-6],[6,10,-3,-8],[6,-10,-6,7],[1,-4,-6,2],[-8,-7,-9,-5],[8,-2,-2,8],[6,6,-2,-9],[-2,-8,1,10],[-6,-10,-1,-7],[2,2,2,3],[-1,-1,3,8],[2,-6,7,-6]],[[6,-7,1,-7],[-3,-7,10,-4],[-3,6,4,5],[-1,1,-8,2],[-1,10,2,4],[-6,-2,-3,7],[-1,5,-6,-1],[1,6,9,5],[6,-3,-5,-6],[8,-7,-6,10],[-9,3,3,-2],[-10,-8,-6,-10]],[[-8,-1,-9,-4],[-6,-1,-9,-2],[-4,-4,2,5],[-4,-10,4,3],[2,7,-2,2],[5,-2,4,3],[-10,-7,4,-2],[-5,-1,2,-4],[-1,-10,5,-2],[-5,7,10,-7],[8,3,-8,-10],[-9,3,-6,-10]],[[10,-9,9,-5],[8,4,-1,-10],[-7,-1,-2,-1],[4,4,2,9],[1,-7,3,-7],[-4,-1,-3,4],[-4,8,-5,-6],[-2,-3,4,4],[-1,-10,-3,-5],[9,9,-5,-4],[7,-6,9,-3],[2,-5,7,-4]],[[2,-3,10,-9],[-4,-10,3,-9],[-6,-3,6,-7],[-2,-7,-4,7],[6,9,10,4],[10,2,-8,6],[-7,-10,-6,5],[9,-9,4,9],[8,5,-9,1],[-3,-5,-5,6],[1,-6,4,7],[3,6,-3,-7]],[[-4,10,-1,3],[3,8,10,-9],[1,7,7,1],[-9,-7,-4,3],[7,1,6,6],[3,-7,-10,-1],[-4,-8,-10,10],[6,2,-4,-5],[-10,-10,2,6],[-2,6,1,-2],[-6,7,7,-7],[-9,-2,6,3]]], dtype = "uint32")#candidate|5996|(15, 12, 4)|const|uint32
bop_5997 = relay.not_equal(var_5995.astype('bool'), relay.reshape(const_5996.astype('bool'), relay.shape_of(var_5995))) # shape=(15, 12, 4)
output = relay.Tuple([bop_5997,])
output2 = relay.Tuple([bop_5997,])
func_6010 = relay.Function([var_5995,], output)
mod['func_6010'] = func_6010
mod = relay.transform.InferType()(mod)
var_6011 = relay.var("var_6011", dtype = "uint32", shape = (15, 12, 4))#candidate|6011|(15, 12, 4)|var|uint32
output = func_6010(var_6011)
func_6012 = relay.Function([var_6011], output)
mutated_mod['func_6012'] = func_6012
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_6020 = func_3786_call()
call_6021 = func_3786_call()
func_3767_call = mod.get_global_var('func_3767')
func_3772_call = mutated_mod.get_global_var('func_3772')
const_6027 = relay.const([[-1.468787,8.721818,5.816075,-7.531758,-0.998795,-4.001382,7.070217,3.886991,3.322282,-7.643958,-8.531631,1.346070,8.157805,-0.268444,-8.512242,-7.369953,0.473549,-7.983741,-8.598298,9.996280,-0.998202,1.034238,-9.137466,4.905396,5.223166,-9.915764,9.289974,8.460710,-7.990871,3.783082,6.073339,-4.804520,7.929600,-8.586010,9.625865,-8.271676,4.045721,4.885377,5.984617,0.427137,-1.642581,-8.084116,9.145597,-9.807688,1.288534,-5.170266,9.364568,-7.512529,6.659337,-1.366101,-6.076337,-7.368995,-3.693347,-5.662485,-6.590210,6.484882,9.677296,7.745264,9.522870,4.736912,9.147238,-8.363971,-0.010551,3.895770,9.907886,-5.734129,6.773171,0.409001,-8.981556,8.359306,-8.100855,3.900253,-1.108593,-6.973490,0.507240,-2.018635,-3.168617,4.406834,1.333847,-4.436356,8.689039,2.488785,-5.831240,-5.848355,-1.505869,2.635218,-2.843395,-7.356676,0.368408,0.691764,-4.496505,0.736215,-9.033381,7.799668,-9.414692,-7.942118,-2.776241,-3.077719,-8.895244,6.596210,-7.056988,2.077659,-6.520977,4.208554,5.550161,-1.449297,-3.257045,-6.101907,-8.955686,7.914656,0.208125,-0.086518,4.039561,-9.346764,-1.346823,-1.309374,1.963282,2.246648,1.809535,3.362797,5.911906,8.787286,-9.748425,-0.071527,2.385112,9.923232,4.899899,5.215600,-6.164005,0.550503,3.657615,0.044911,4.494404,-4.989204,3.036906,0.887378,3.751542,-3.296906,-0.732159,0.886861,-0.991251,9.337003,-6.896495,-3.390025,-2.507021,1.426494,8.750146,-2.877703,-6.483998,-3.145759,6.812092,-0.342039,-5.913192,-2.035693,-6.928044,-3.714966,-2.168209,-7.862736,-1.848664,5.605792,-4.295606,5.601120,7.706677,4.695807,-5.148371,5.459860,2.192734,4.145911,-9.464087,8.798869,7.772187,-2.297006,3.857494,-9.147728,9.118573,-4.721725,4.809062,-2.310617,-7.186141,-3.759659,8.582439,-9.073630,1.414052,-7.322186,7.475027,-5.514810,5.628991,-9.137802,-0.860887,7.989009,-4.163640,7.401712,0.741455,-4.588411,6.717815,-3.275271,-3.138522,-0.465280,-5.580286,4.864397,-7.861917,-3.501662,-8.623905,-6.093807,-2.637415,2.804209,3.522997,-3.153969,-5.723926,-8.047208,7.795472,-5.992647,3.339908,5.439935,2.930941,9.948426,5.146552,0.629323,-9.622463,-1.986741,3.745915,8.136495,-4.246927,0.127775,-3.177357,0.184828,3.639691,6.568564,-1.338175,-3.451682,-7.151683,6.112911,-9.325808,8.640394,-5.185121,8.435415,6.346694,-1.805324,-4.070648,-0.669134,-8.223709,-8.639606,5.381296,-4.503228,-7.608310,2.362036,1.615219,-5.172093,-6.328200,-6.838988,2.363086,-8.736810,3.293426,8.223700,1.592798,-0.788710,0.360117,5.922791,9.075954,-0.328468,-6.971150,-5.378214,-5.491443,1.098527,-1.279882,2.085158,5.882335,4.617164,-1.335764,-1.993665,-6.196308,7.671527,3.461202,3.074886,3.572556,7.089734,7.238868,4.589697,-0.977845,-4.729989,-4.501392,5.505496,-1.318686,-1.433109,8.885887,-2.839640,-2.582027,7.079799,-8.578588,-0.356053,-5.881201,3.883894,5.498596,4.506599,-2.484610,-0.385596,0.785200,-4.467129,-1.315390,-0.971283,-0.958815,-2.220437,7.652348,-2.398529,-2.704881,8.318396,-5.562882,-6.794124,1.345334,-6.803324,-1.197747,-7.075194,-6.990346,4.319263,-9.605575,-9.347002,-7.505237,0.111887,-3.980265,1.826810,-1.887964,7.658019,5.205558,-2.428528,8.919905,-1.988717,7.044780,1.381833,-6.618847,4.684631,-4.518042,4.456654,-1.711306,-7.632318,-5.411311,8.615524,-9.502959,-9.113432,-9.822973,-7.750545,6.037659,7.666629,-5.895527,-5.158099,5.544395,0.826158,5.387480,-0.399769,1.459495,-0.636374,-2.122620,-9.590478,-8.438558,9.188140,5.558937,-8.214142,-4.091772,5.552986,-0.350993,4.341026,4.602700,5.291391,1.527194,-2.505338,3.019412,5.773228,-3.797293,-7.696467,-7.609053,-7.962137,-2.052340,-6.776146,-7.511037,6.138816,8.859440,7.574002,9.200065,4.064101,-0.318060,5.287858,-5.132121,-8.674820,-8.900480,5.414309,-7.041733,-6.373736,-7.019502,6.427626,-1.371984,6.265989,-3.716036,0.389870,1.298555,-8.824932,-6.833138,1.223167,-0.462313,-1.884203,7.792992,3.265555,4.980380,-2.792247,-1.763855,-2.582537,-6.831841,-7.385718,-3.144768,-9.402785,-5.500461,-9.639514,-6.608767,9.086566,1.539994,0.319250,1.833433,-4.474673,-5.373991,-7.135497,-2.227680,0.301262,-8.998733,7.920191,9.990039,9.795352,8.866999,6.476039,6.866859,-3.301971,-9.415347,9.397325,5.487244,-4.644373,-7.721016,-3.680462,1.539300,3.362532,-3.599587,9.072069,-8.769872,4.286770,-9.023393,1.845872,-7.442432,-6.310403,0.109272,-4.601478,1.825370,-9.874201,-9.889607,-1.704096,9.448829,7.163287,1.015916,9.918414,7.885417,5.819328,3.087321,-8.418895,-6.460969,-3.522866,-3.867391,-0.963634,8.704985,-1.282067,-5.495960,-8.212592,-7.347387,7.384209,-9.844772,-5.764270,2.193279,3.915458,-5.627130,-0.199049,-0.125184,8.502472,-5.338778,-3.909476,-5.688640,2.863489,9.097370,-6.783311,1.780747,1.789911,4.997029,-6.603261,0.199932,7.956644,0.845310,8.327206,1.067020,-9.359428,-6.313694,0.022224,3.151358,1.089950,-3.487350,6.955972,6.392168,4.707310,-2.661592,-4.443259,8.451636,3.011909,1.998367,-2.466887,4.722531,-7.416413,7.699208,4.865182,-9.165825,-8.742759,-6.840692,0.915284,-2.150833,-3.834953,-0.443508,3.075538,3.161443,-7.001021,9.614052,5.369537,2.968378,-8.694913,4.968755,-9.074895,5.899339,8.627663,-1.345179,9.500932,0.452711,6.669203,-7.961859,-5.569436,8.188576,5.323958,7.310788,-0.673791,-9.991229,-6.697975,-8.340497,-2.568254,4.428184,3.985913,-3.665422,-8.041163,2.119195,2.705226,-6.228838,5.351221,-3.651117,-0.263822,6.569356,-3.731891,-4.074051,3.478015,7.564616,-0.052173,2.712621,-6.577517,-5.458444,-2.185974,-6.478312,-2.565825,8.186255,6.439570,-5.637240,3.169909,-7.007518,3.069618,-2.041154,1.827258,-7.953910,-7.522921,3.891984,8.681483,-3.683221,4.732960,4.896740,3.214673,1.592251,-7.424052,9.170184,7.055156,-4.770243,-5.802706,-5.008798,-9.307560,0.316358,-0.850344,-6.758356,-4.161801,5.694490,6.587847,3.746626,8.206276,7.309060,-5.144343,-7.294648,-1.117914,4.642583,7.788774,-2.184616,-8.412746,2.296975,0.370939,0.088649,-3.827506,9.369267,1.670720,-1.175866,-4.160140,2.868751,-7.048478,-4.734320,-4.659477,6.330612,7.834765,4.879563,8.804318,5.752151,-7.189557,7.210940,2.502376,4.818898,1.489205,-3.467719,2.723945,3.382090,2.480504,0.816490,3.476883,6.659558,-8.622576,-1.635649,4.254287,-0.336221,-0.414931,-5.928549,3.245819,8.743988,-0.264761,-6.832190,2.930664,4.260975,2.094689,-8.134878,-7.735664,-9.803786,-6.005174,-2.810491,-4.192648,-0.411590,-4.246823,-0.209285,-5.077002,8.047367,7.700122,-1.225940,7.504715,2.904343,-4.069087,6.829451,-6.467189,-0.559127,-6.988912,-9.211743,3.833986,-9.944355,-4.845445,8.948988,-8.751831,6.096037,-9.240712,-4.974419,8.625155,6.898575,6.425741,-0.697165,-7.269315,-3.623361,-1.952266,8.813538,4.478360,1.911187,-8.879726,-8.518203,9.892492,4.229040,-3.124306,7.322846,3.508833,5.475911,8.494059,-7.921577,-6.681188,0.652158,-4.764917,9.251853,-3.019444,7.882292,2.374869,8.970925,2.156711,9.319989,-5.091655,1.247968,7.072408,-2.789149,9.595495,-8.302182,9.131243,-3.261500,-0.853558,-7.980183,-5.403702,5.488285,9.503050,9.537546,7.941922,-9.481260,0.304365,-1.774462,9.120494,-6.315354,7.373315,-9.257414,-0.063939,-6.688044,-9.599754,0.129249,-1.424528,7.223503,5.073996,-2.797248,2.699262,-7.406116,3.095143,0.444775,6.997835,3.747899,-4.939859,4.683809,-9.071897,1.143567,-3.231340,2.986961,6.997887,8.345241,1.282807,-6.316604,-1.089527,-9.827358,7.359916,-7.878883,1.252740,3.407254,-3.155245,-3.536649,9.634713,8.413868,-1.264971,-6.442048,3.264470,-1.321581,-8.855540,4.012771,4.716555,-2.787598,8.396061,2.162362,-6.331925,-0.099180,0.873232,2.413726,-8.025241,6.507232,8.079308,2.924430,8.542011,-2.095878,-3.948791,-1.720474,3.777525,-8.149167,-2.233150,1.440099,-5.060860,0.094365,-4.465832,-2.322198,0.532724,1.299932,-2.510106,-0.227476,-2.795953,9.954214,-8.355615,-9.435305,-6.275382,4.172966,-5.371558,4.512960,-0.070804,-5.102096,1.081507,-7.499447,-4.335072,9.489303,-8.889745,-5.310011,-1.398674,9.945185,-6.179350,-4.180638,-7.961622,4.703423,-8.229554,9.219288,3.882267,9.411653,9.922326,5.757752,5.650007,-1.672518,-5.449338,-8.512044,-0.591428,-8.867584,0.729106,8.790732,-8.875811,-2.709637,6.582003,2.532943,-8.665944,9.631705,-6.311133,6.268348,8.853180,7.968438,9.317335,4.909913,7.425735,-0.876548,-9.060207,-0.053698,-7.425057,-5.750143,7.276411,-1.626374,-2.044381,1.324338,-2.354008,2.375511,-7.824437,-0.709517,-5.105765,2.333902,2.691387,1.276662,7.196323,-7.562171,-3.785346,1.659939,-1.878360,6.535699,4.260628,-2.828087,-9.934233,6.403412,-5.045796,-2.354871,4.891802,5.222135,3.136373,6.998542,-1.530408,-3.521489,-4.031348,-8.180119,3.810058,2.107022,-9.476727,-3.236906,3.344122,1.818606,0.097924,0.279188,3.257525,5.705297,4.152798,4.919241,-3.133000,-2.556513,6.126326,-7.976479,2.152479,2.059917,1.249877,-6.463835,-8.786239,-5.170406,9.572775,-4.443329,-9.586274,2.852848,-1.867312,-5.541509,-2.518577,4.512343,3.390066,-1.778284,-2.870708,3.171655,7.305743,-5.091434,9.630350,-0.255923,-1.006346,-8.758695,-2.094828,-2.855298,-7.105900,-4.948215,-5.631802,6.318826,2.736628,-5.693134,1.229609,5.541632,6.372532,-4.769097,-6.757518,2.891079,-2.033916,-4.627872,5.111065,4.639068,2.415606,-0.242897,-1.642064,0.553224,7.718331,3.572897,2.171336,1.508591,-9.924480,-2.225390,-4.949827,-8.748397,9.706438,-7.040875,1.246009,-1.261701,8.475491,-5.436918,-6.689753,3.643778,8.011559,1.945042,8.738121,5.955199,-7.446344,-0.814885,4.706350,6.610699,5.756483,9.056603,-9.222297,8.512971,-9.737566,1.719736,-2.002738,-1.504451,1.534597,-7.574295,5.423566,-0.135379,0.701935,-1.832708,6.720141,2.951189,5.052234,-9.074212,5.800313,2.804437,5.780290,-4.108053,5.560399,-4.794466,-6.343076,8.442349,-0.543838,-7.116426,1.028916,4.337625,-3.976692,1.675995,0.070370,0.182716,-1.954451,9.309222,2.345481,8.685571,-6.002852,7.979934,-3.122623,-9.233675,3.477980,5.043212,-6.683219,6.765352,5.074143,-5.939167,-2.560537,3.521954,6.854447,5.681020,8.809509,3.377516,-6.996673,-0.045902,-2.340190,4.555583,0.883268,-2.863451,4.760788,3.945744,-1.821876,-6.897566,8.540135,4.999407,-3.633271,-6.223130,8.248116,0.032790,-1.796638,3.829693,0.031901,-3.917629,0.965200,-1.809192,5.430746,0.389732,3.392413,-5.050417,-0.618491,2.441083,-0.950609,-9.543962,-0.208332,6.168122,7.210435,2.647975,-2.832492,-2.911118,-1.320683,-2.727098,-4.532287,1.900467,2.861515,-9.749223,-1.800730,-1.962051,5.242633,-0.981595,4.638123,6.980101,-2.037999,0.506280,-7.483251,-6.195304,-4.835498,-1.656652,3.572721,-3.264270,-4.515361,-5.448379,5.721069,8.705184,8.169609,-2.710070,-2.772368,9.628171,-2.726121,3.008120,-9.517216,2.557659,-2.204878,8.136823,1.845197,9.371357,6.820833,-5.779506,-4.525428,0.488642,-6.437078,-4.184297,2.126869,1.556137,-1.919088,5.553649,1.866591,2.896311,-7.543613,5.277237,9.985123,9.582699,-1.566861,1.280349,-0.812082,-9.533655,-3.076702,7.714751,-4.939492,8.993209,1.907709,-5.803944,6.084959,-8.283347,-4.546005,9.371553,-5.869636,-8.912881,0.821834,-4.331969,9.207699,-5.820689,2.237761,3.838330,-4.055415,4.904385,8.855577,-0.979136,5.606646,5.908783,-9.679976,4.071373,2.455135,5.176465,7.998973,-9.265498,-1.582521,-7.517734,-1.948800,-8.068575,-0.592229,-9.076767,2.248634,-3.549559,-9.280591,-4.792546,4.825267,2.696893,-6.972006,0.323896,-7.716061,-0.304477,-3.856914,5.291788,-1.739030,3.538601,7.965183,4.372151,8.763827,1.600307,0.337265,-4.141498,-1.799241,9.120652,4.703141,-9.052010,5.574803,-0.612849,-9.994353,-6.061633,8.966895,2.472473,-3.493613,1.094962,9.354071,7.471381,4.727654,5.138503,5.041418,-7.874807,8.617913,4.181456,5.336495,-0.708248,5.747204,-9.024428,-1.605825,6.052657,-4.970751,-5.046804,6.172793,-8.775016,-5.021291,-2.567863,5.579496,-9.794421,-6.716334,8.316437,4.681511,3.297446,-2.882140,-1.773303,2.019812,5.593249,-0.113955,4.355497,9.339523,-7.084659,3.161964,-2.121398,3.005157,6.032940,-7.594853,6.609884,-0.757180,-5.825222,4.651712,-1.281215,-5.305320,-8.640939,7.990990,-9.392778,-5.615486,-6.028235,-0.396804,-3.826322,9.966121,5.868572,-5.839120,-5.163788,5.446213,5.589482,3.774475,1.716025,-5.198491,-6.274514,-9.073359,-3.536077,-9.815074,-3.687819,2.583026,-8.016947,1.175392,-1.832233,7.186486,-2.437021,8.587774,2.935438,5.660642,-4.407793,-8.331630,-8.735948,-1.738821,1.894997,6.455896,5.347403,-8.190594,-0.297956,0.709671,3.792298,0.245630,3.684998,-5.258203,8.867993,7.919356,7.027996,-4.424645,-0.254199,-6.633877,4.054116,-9.415068,1.884652,9.061062,-3.649953,-0.045923,6.341149,2.995786,3.746805,0.356390,-5.605689,-4.044297,-5.694711,3.771087,7.294196,2.306948,9.173638,4.017806,-2.165122,-0.581301,6.970395,8.286248,-9.038487,3.402269,-6.984781,-2.186793,-2.809920,-1.238000,-2.037148,6.050109,4.452749,-6.763755,-5.674575,4.132820,-9.780058,7.699387,1.845423,-7.655647,-3.562746,6.568466,-0.941089,8.028806,-6.372850,-8.223621,9.332159,4.639974,-7.083613,6.925065,4.395360,0.504182,6.669646,-3.372259,9.989690,-6.520814,-0.789870,-5.794179,-5.308659,6.076632,2.560766,7.293431,-1.474834,-3.060507,-3.499050,4.428979,-1.042998,-3.944791,-0.393776,-7.444412,5.237450,9.736727,-7.006008,8.979167,6.763696,-7.529432,-0.954660,5.508648,-0.591322,7.553280,9.210321,-4.712224,6.106102,2.774173,1.359347,-1.402681,0.274600,1.886247,6.687894,-3.820220,9.430475,-5.120540,-0.097922,-6.390591,6.536887,3.345491,1.374255,-0.024354,-4.929008,-6.836833,-7.508093,5.171592,-8.122524,0.726881,7.937550,-7.576184,-9.475217,-0.643667,-9.833542,0.757446,-2.628449,-1.454762,3.436558,-8.450151,5.002644,9.802118,6.574527,6.869087,0.603236,-4.934335,-0.324558,-0.628607,6.935940,0.240214,0.819048,5.830439,8.301900,3.745907,-5.009253,0.977471,-3.251253,7.761234,0.217328,6.585629,-3.656519,2.718687,-3.871848,7.496970,-0.247281,2.020188,-4.742323,5.996037,6.933607,-7.858949,-9.992242,8.000043,-8.467636,3.614531,6.957637,9.420806,-8.701565,-5.651292,-8.084390,-1.473041,-1.029630,-4.071666,-3.408084,-0.947638,3.799891,7.781219,-7.584660,-3.680964,-1.928118,-1.974537,1.315894,2.296518,-3.193476,-3.079721,8.953482,3.987757,8.156599,-4.319908,9.344312,6.584242,7.348230,4.495061,4.811230,6.163172,7.161902,1.668134,6.994972,-1.196993,-2.772905,-0.473321,8.793965,0.381759,4.085821,-6.424311,-6.672483,7.930208,-3.549940,-3.040679,8.732826,0.909473,5.504151,9.471139,5.589377,0.423488,-6.266035,-1.107906,5.573481,-5.644704,1.829126,-2.179346,-5.478664,-0.675417,-1.922730,-1.737164,-9.228539,-0.986056,-8.916371,-3.731413,4.346834,3.645703,4.905568,2.728065,2.301125,2.287883,-6.506748,6.573789,-9.904406,0.136104,-7.080541,-5.876136,-6.327749,-6.945011,-5.611496,-3.225620,-8.242215,1.304732,1.665566,6.019699,5.303717,-6.775804,-6.979993,-5.533394,5.887089,8.937818,-1.683375,-8.920761,-9.659834,3.391045,4.198808,-5.120153,-2.889053,3.502152,-4.205836,-5.779171,9.919676,9.104299,-6.231538,-7.678302,5.685493,5.288691,2.406834,-5.328212,1.097314,-4.559303,-7.661885,-4.518259,0.705564,-7.873684,-1.925709,6.909539,1.226961,5.368946,-7.210685,9.697134,7.479764,4.806099,-2.544309,-3.252800,-4.557958,-6.700348,5.100493,9.072784,-6.536422,-2.953078,-8.378885,4.148387,-0.965640,-9.045463,-7.020471,4.045069,-0.082217,4.842179,-7.104548,-6.949583,-8.464770,4.501468,-7.441688,-6.527092,2.876771,0.996049,-5.130578,0.627759,-1.702372,-5.151293,-6.582763,-0.759097,-2.970675,5.176708,-1.045019,7.931794,5.939519,-1.721838,0.558712,-6.999542,-3.258633,0.034858,-8.225686,1.528710,-7.131562,-8.874314,5.126954,1.886703,-3.606428,8.672703,1.117126,8.841289,5.872086,-0.510879,8.397806,1.766737,-5.817819,-7.300486,-4.649931,2.373359,6.897666,4.535912,9.691902,7.981982,2.926223,-7.338848,6.551370,4.553795,9.833581,-1.172454,6.550090,-7.049091,-9.677460,-6.474848,7.470045,4.080876,0.896657,0.631237,4.649955,-0.276381,6.572344,1.049974,3.709436,-3.244914,-6.329976,6.264764,-1.829599,-6.771693,-4.380172,6.448756,-1.237910,9.905407,9.806461,-1.297049,1.336804,-4.290518,4.127154,6.692557,-8.029436,-5.659963,-0.881634,-1.944798,9.154982,7.052163,-0.607356,6.034866,5.734610,-7.923287,-8.372444,-0.166175,-0.789458,-3.248041,-0.988017,-8.346571,8.147739,-9.863027,-7.883374,0.291223,4.803762,4.506241,4.271117,-7.511114,3.374983,-3.777656,-3.272917,3.021464,0.940821,-4.050355,-8.074337,-5.589048,5.341086,9.543925,-7.699069,-1.314842,-3.108899,8.904914,-1.945734,-9.507982,-2.044904,8.516840,5.748840,5.526897,8.950591,8.060621,-5.907251,3.256200,-4.506014,4.089049,0.902643,-9.264221,-9.262780,-5.607187,2.449698,-1.246460,5.134187,-4.696146,7.470732,5.902503,-4.326941,-8.646602,-0.758467,-9.784474,-1.821085,1.225594,0.494948,3.719850,4.237951,8.781107,6.698242,1.407199,4.317558,8.323876,7.977864,9.940639,3.402534,8.317903,-6.762625,-7.769984,7.612824,-4.184996,3.238389,-3.590851,1.948843,-1.699588,-4.428517,2.271827,8.064340,-6.573046,1.452468,-7.482948,2.634179,1.514631,-7.436593,-8.762009,0.880497,-9.456965,2.253263,-1.407295,-1.254784,5.261410,-0.187759,7.668711,1.186301,-3.513831,9.318969,-1.820730,-3.485584,-8.130644,2.506542,8.959475,-2.442714,-4.151657,-4.543652,-7.907891,3.453073,6.647503,1.784645,1.968190,-6.583971,6.521262,6.236345,-9.191360,5.224009,-7.167160,-2.385391,-2.862150,3.000351,-2.766378,8.393335,7.518501,-2.706666,-2.017418,4.068562,-9.654635,5.112947,2.683103,4.952829,1.940737,8.149550,-4.892316,4.203317,-0.948952,0.734577,-6.017744,2.391022,3.890649,3.650882,5.968616,3.474601,-9.045852,-8.690532,7.341536,-9.688256,-4.659736,-8.825261,9.688808,6.368032,8.111771,-8.191150,3.969515,2.328288,1.869233,-4.811736,3.604349,-8.326973,6.452551,6.284071,4.608406,8.106781,6.445948,-2.132271,0.689789,-5.556117,-1.031939,-9.324460,0.179805,-6.020122,-1.457938,0.881552,-7.426015,-2.442990,-1.496300,9.111149,3.731154,-1.548446,-2.604476,4.831701,-8.289582,-0.221969,-1.945805,7.725508,-6.903382,-5.583451,-2.122646,-1.149872,1.493530,-9.584569,8.602276,6.403613,-4.292671,0.515582,-0.179333,-2.472849,5.222213,5.619176,-0.340362,1.404521,-8.338022,6.174243,2.940851,-0.643145,-5.124692,8.755749,-2.467208,-3.353307,-3.197660,9.014275,-4.225971,8.877960,7.227208,7.130036,5.319488,-4.698187,-8.456303,4.731009,-8.528003,6.168900,-3.262208,-9.163001,3.063995,-4.869180,-5.537330,4.366242,4.920520,-7.799851,4.237260,-8.199361,-6.587722,1.307945,0.767511,2.886795,-4.377607,-6.347219,8.852437,8.368639,-1.455275,-4.390188,9.112039,2.237591,-4.604389,1.245349,-5.597364,5.722909,0.241645,5.647274,3.545564,-9.908051,1.081821,-5.498648,3.234251,-7.859562,3.432695,-9.739803,-4.646881,-2.631565,-7.855280,-9.403960,7.088069,1.225706,-2.277345,6.572133,8.975981,9.147858,7.354596,6.182935,2.739909,-2.229208,-6.917655,-2.614691,-8.828841,-1.453997,-4.137649,5.164756,-2.929464,-6.601455,-5.975876,-5.030457,6.525708,-9.765068,7.778441,6.808126,4.422626,-6.684323,2.227797,-4.485686,0.747131,-0.937900,-1.431841,0.008538,-4.059320,1.063770,-8.374474,0.719950,-3.369467,-5.189204,2.297111,-0.271309,3.430719,8.724085,9.321895,-9.072380,-7.567471,8.974118,1.399380,3.583312,-3.315511,-8.578049,-8.068137,2.537937,8.067531,-2.066837,6.394900,-9.805203,3.123839,6.258865,8.067687,8.347557,-8.209351,-6.823977,-6.501506,-3.120795,8.594940,8.157807,-1.416274,-8.965557,-3.800750,-0.127825,3.153121,-7.511037,-6.435582,-2.974207,5.190160,7.938367,-2.598151,6.938298,8.546772,2.736161,-6.481695,-5.921414,8.366072,0.468262,-6.658541,-8.470608,-5.433059,-5.813838,-7.451857,-9.330989,0.127213,-9.591649,-5.164664,-7.959360,6.781356,-9.489741,-1.405071,-8.920946,2.909858,2.111220,0.530013,8.157975,-6.726280,4.350430,-6.610806,-9.315557,9.983730,0.043082,3.398394,-0.190598,-8.186400,-3.611138,4.561576,-1.009472,-0.205087,-1.794192,1.185971,6.201522,-4.407935,4.808823,7.530194,-9.148747,-0.729125,-2.020122,7.497298,-9.894115,0.918097,-6.262294,0.775444,4.819661,-7.040542,6.103658,7.893280,5.088864,6.369994,1.240063,-8.378129,-6.671584,6.395290]], dtype = "float64")#candidate|6027|(1, 2048)|const|float64
var_6028 = relay.var("var_6028", dtype = "float64", shape = (325,))#candidate|6028|(325,)|var|float64
call_6026 = relay.TupleGetItem(func_3767_call(relay.reshape(call_6020.astype('float32'), [14, 14, 2]), relay.reshape(const_6027.astype('float64'), [128, 16]), relay.reshape(var_6028.astype('float64'), [325,]), ), 0)
call_6029 = relay.TupleGetItem(func_3772_call(relay.reshape(call_6020.astype('float32'), [14, 14, 2]), relay.reshape(const_6027.astype('float64'), [128, 16]), relay.reshape(var_6028.astype('float64'), [325,]), ), 0)
output = relay.Tuple([call_6020,call_6026,const_6027,var_6028,])
output2 = relay.Tuple([call_6021,call_6029,const_6027,var_6028,])
func_6030 = relay.Function([var_6028,], output)
mod['func_6030'] = func_6030
mod = relay.transform.InferType()(mod)
var_6031 = relay.var("var_6031", dtype = "float64", shape = (325,))#candidate|6031|(325,)|var|float64
output = func_6030(var_6031)
func_6032 = relay.Function([var_6031], output)
mutated_mod['func_6032'] = func_6032
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_6039 = func_3786_call()
call_6040 = func_3786_call()
uop_6043 = relay.log2(call_6039.astype('float32')) # shape=(14, 14, 2)
uop_6045 = relay.log2(call_6040.astype('float32')) # shape=(14, 14, 2)
bop_6061 = relay.power(call_6039.astype('float32'), relay.reshape(uop_6043.astype('float32'), relay.shape_of(call_6039))) # shape=(14, 14, 2)
bop_6064 = relay.power(call_6040.astype('float32'), relay.reshape(uop_6045.astype('float32'), relay.shape_of(call_6040))) # shape=(14, 14, 2)
output = bop_6061
output2 = bop_6064
func_6073 = relay.Function([], output)
mod['func_6073'] = func_6073
mod = relay.transform.InferType()(mod)
mutated_mod['func_6073'] = func_6073
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6073_call = mutated_mod.get_global_var('func_6073')
call_6074 = func_6073_call()
output = call_6074
func_6075 = relay.Function([], output)
mutated_mod['func_6075'] = func_6075
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3796_call = mod.get_global_var('func_3796')
func_3798_call = mutated_mod.get_global_var('func_3798')
call_6129 = relay.TupleGetItem(func_3796_call(), 0)
call_6130 = relay.TupleGetItem(func_3798_call(), 0)
func_571_call = mod.get_global_var('func_571')
func_575_call = mutated_mod.get_global_var('func_575')
const_6144 = relay.const([[-2.149271,-9.954625,6.485565,6.364089,5.216464,7.443515,-0.404967,-5.745962,0.983867,-4.923001,-0.685797,7.634579,-1.558865,-0.475841,-3.864906,-9.208749,2.928838,-5.041475,1.737131,1.126884,1.387755,-7.661800,6.593008,-2.530260,7.432632,-6.373149,4.333314,-4.225410,-8.559173,-2.283764,4.362342,0.199965,-0.281616,-8.915013,3.252888,-7.172323,-9.820626,-3.389061,4.731417,2.884764,-3.640211,-9.583625,5.108577,6.381293,7.677553,-7.666847,-2.486393,9.613829,8.885773,0.800441,-5.396100,-3.483169,-8.846639,-8.515361,7.944736,-1.618285,8.574890,3.117138,1.354778,-5.630662,2.494186,-8.871942,-1.710351,-6.960191,2.696492,6.535714,7.904009,-8.277244,-3.876714,2.173972,-2.856351,5.757425,-6.282139,-3.250137,5.609366,7.174696,-2.296137,7.027036,5.571447,-8.360386,6.019575,-4.744703,5.693749,-7.611331,-3.255493,-8.998446,-6.331508,-7.458753,-2.854153,-4.256197,-4.133640,-4.112782,0.895632,7.076957,8.629322,8.078973],[6.097944,5.985272,3.398913,9.744674,5.841141,-6.191894,4.754785,5.324070,-2.390595,-2.119338,4.729667,0.130266,-8.479701,4.460368,-5.441441,-6.475296,2.211759,3.646153,1.960368,-4.783842,-1.065790,-4.999043,7.336987,0.679623,-5.710333,-0.913174,0.666385,-9.073244,2.134265,5.029959,0.921456,2.646491,-5.450027,8.862998,6.947158,-5.684004,-5.503058,-8.984444,-5.669714,0.977798,2.658354,-8.600673,3.759735,-0.098142,9.587182,7.901307,9.344025,9.094542,1.490055,5.858284,-2.570135,-5.735394,8.417582,-2.015022,-4.482215,-7.890316,-7.864160,-4.472202,3.690118,-9.766641,9.926517,2.440218,-4.357912,-5.529464,5.849944,5.738006,6.758700,6.589530,-5.041670,5.269058,1.240559,-8.875710,-1.827993,-0.299328,9.608826,-1.671144,-2.492006,8.005019,-6.256115,-5.389969,1.268902,5.956699,9.005681,2.826186,6.186231,-6.966242,-7.068516,-5.578560,-5.462538,9.380258,6.241716,-2.323751,5.346191,-0.161141,2.280683,-4.203356],[-5.032469,7.290288,-6.380629,2.351050,3.577506,-5.716704,-7.250527,2.168317,2.036327,-1.139323,-5.451566,-2.461666,-1.409067,-9.635572,7.230916,0.402774,-6.698171,8.516658,2.337846,-4.318933,1.759149,9.141052,6.872660,-7.136072,1.710698,6.262361,4.714588,9.424130,0.239460,5.736640,-2.323595,-6.752462,2.247210,0.573001,3.220659,-7.526177,7.763842,-7.467421,-5.282852,3.288645,-4.171027,4.132973,7.656844,7.201038,-1.282612,6.007139,4.203182,-4.636747,-1.995446,-7.514322,5.113676,-1.873113,-8.630237,-7.104297,9.892506,-2.616046,-2.961773,0.334543,-6.195583,-8.847091,0.540549,-2.326989,-0.234717,0.298258,-9.483651,-8.431949,-6.237201,-1.537775,6.608663,3.892940,6.516355,-5.440337,-6.912927,8.041276,7.949370,7.823087,6.661868,1.876126,-2.386081,-4.430228,-8.266333,2.463899,5.083063,8.777484,-9.795896,6.313613,-1.356219,9.498749,9.375385,-1.720599,3.450661,5.992365,9.886071,-0.167049,-6.151861,-1.632057],[2.442309,4.630821,-9.830101,0.170880,-5.541466,-6.000229,1.104083,0.666560,-7.101000,8.718129,0.535440,-9.913483,0.051867,4.692358,-6.691696,9.270737,5.163381,-1.469579,0.536938,-9.605668,7.955206,8.085943,-6.763587,4.000138,-5.874708,-0.409202,-0.709037,-3.811416,1.976642,5.150225,7.288764,8.179962,8.139741,-4.207978,-8.912131,-9.842116,8.147032,-7.776438,-7.712461,-8.010921,-2.331874,9.489692,-9.723513,-2.981404,5.200885,1.074516,-9.222960,4.340342,-2.758026,0.203903,-8.989133,-3.212788,4.152634,1.935801,3.957420,-5.691099,6.591212,-2.903651,9.570088,9.770171,-7.627165,3.048661,-8.965819,-7.967188,5.691932,9.736464,-4.041848,2.151709,-3.401604,7.949248,8.615897,7.059582,1.360979,-2.082330,-7.537321,7.690365,-1.670156,-9.966546,4.884173,8.766399,-9.048530,-1.550521,8.189270,-6.780087,3.555801,0.891081,8.494332,0.452732,-5.002295,4.342250,6.596565,6.134093,1.795359,-2.399197,-3.888253,9.512297],[-9.485660,1.928502,-9.832232,3.852975,-8.151138,-1.493580,7.223271,-1.068722,8.604721,7.066815,9.976073,-8.175546,-5.711991,3.290391,-6.660036,8.719764,9.676404,-7.186749,-4.154635,1.682918,-9.343716,-5.585277,2.626542,5.443592,-0.118138,1.125882,8.788740,-0.490113,0.736859,9.998146,3.679925,-3.211576,1.392194,-1.992688,4.302836,-0.711346,8.856116,0.758507,-4.979890,4.933214,6.940069,-8.906160,-2.242419,-9.128079,-0.812668,5.580166,-6.323284,-0.019620,6.156763,3.983915,-7.065686,-2.980696,-2.315766,-4.250940,5.713443,1.105409,8.683003,0.194443,7.558271,8.935388,2.085560,4.010399,-6.633221,-6.061141,-6.265675,-4.069808,0.458655,-7.905918,3.994949,-3.010441,-9.625882,6.326996,6.504309,9.619478,1.602976,2.259973,2.898454,-5.101007,-7.780929,6.571749,5.325949,-0.262476,3.977219,-0.409345,1.562984,4.682080,1.629223,7.673995,6.198327,0.837397,7.210186,-3.608160,2.136901,0.982107,7.782743,-3.392714],[-1.735114,1.914434,0.945643,-1.029100,3.205982,-1.029274,6.103375,2.277087,-6.507826,3.880696,4.694188,-3.973706,-6.846066,4.418054,6.333783,-3.497643,-1.428547,2.184523,-6.464851,-4.511077,8.888063,-0.447641,6.472336,7.643344,-7.293783,8.337208,-0.487421,-7.732675,1.234815,-4.566657,7.254888,9.808093,3.431397,7.422917,-4.653863,1.128736,-3.726449,-0.966296,3.737412,3.603318,-2.704591,-1.531552,-6.231363,-7.166273,-5.873921,-4.748377,0.787509,5.440409,-6.841906,-8.390386,-7.416187,-3.019931,3.328908,-4.033214,5.624052,-7.405051,9.060421,-5.831491,0.565547,1.487672,-0.465861,6.942237,5.457296,-9.065472,5.086444,7.250062,-6.008033,-8.438404,-9.758736,-1.316056,-5.070694,-7.233108,9.314088,-6.333896,1.083581,-8.649553,9.230709,5.078640,-9.872490,0.901072,7.379046,-4.614209,-0.498392,4.621539,2.001066,-3.834759,6.675087,6.326871,-4.365110,2.073037,-5.445026,6.901475,-0.624564,-3.989791,-5.941663,-2.207183]], dtype = "float32")#candidate|6144|(6, 96)|const|float32
var_6145 = relay.var("var_6145", dtype = "float64", shape = (325,))#candidate|6145|(325,)|var|float64
call_6143 = relay.TupleGetItem(func_571_call(relay.reshape(const_6144.astype('float32'), [16, 6, 6]), relay.reshape(var_6145.astype('float64'), [325,]), ), 1)
call_6146 = relay.TupleGetItem(func_575_call(relay.reshape(const_6144.astype('float32'), [16, 6, 6]), relay.reshape(var_6145.astype('float64'), [325,]), ), 1)
uop_6175 = relay.sinh(call_6129.astype('float32')) # shape=(14, 14, 2)
uop_6177 = relay.sinh(call_6130.astype('float32')) # shape=(14, 14, 2)
output = relay.Tuple([call_6143,const_6144,var_6145,uop_6175,])
output2 = relay.Tuple([call_6146,const_6144,var_6145,uop_6177,])
func_6193 = relay.Function([var_6145,], output)
mod['func_6193'] = func_6193
mod = relay.transform.InferType()(mod)
var_6194 = relay.var("var_6194", dtype = "float64", shape = (325,))#candidate|6194|(325,)|var|float64
output = func_6193(var_6194)
func_6195 = relay.Function([var_6194], output)
mutated_mod['func_6195'] = func_6195
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5634_call = mod.get_global_var('func_5634')
func_5635_call = mutated_mod.get_global_var('func_5635')
call_6213 = relay.TupleGetItem(func_5634_call(), 0)
call_6214 = relay.TupleGetItem(func_5635_call(), 0)
func_2954_call = mod.get_global_var('func_2954')
func_2956_call = mutated_mod.get_global_var('func_2956')
const_6221 = relay.const([-4,2,-10,-10,3,-3,-8,2,-4,-8,-3,6,-6,10,-1,-1,-5,-8,-4,-1,-6,5,-1,-5,-3,2,4,3,-9,5,4,-5,-9,-6,-1,-1,3,1,3,9,-9,-4,-10,-2,-10,1,4,-2,8,9,7,8,-8,-6,-3,8,7,-6,7,-8,-6,10,6,5,2,-5,6,10,10,-4,2,-6,4,-8,9,-9,-8,7,2,9,-5,-6,-8,4,2,-5,-2,5,-3,8,-10,9,5,-3,-7,5,-7,-5,-5,4,-1,-4,-2,-8,2,6,-2,-2,-10,-5,5,9,1,6,-4,-3,-1,-7,-6,7,2,9,-10,-10,6,-7,5,3,-1,2,-3,-2,4,9,-6,-7,2,4,-7,-2,2,-3,7,-3], dtype = "uint8")#candidate|6221|(144,)|const|uint8
call_6220 = func_2954_call(relay.reshape(const_6221.astype('uint8'), [3, 16, 3]))
call_6222 = func_2954_call(relay.reshape(const_6221.astype('uint8'), [3, 16, 3]))
uop_6235 = relay.acos(const_6221.astype('float64')) # shape=(144,)
func_1156_call = mod.get_global_var('func_1156')
func_1160_call = mutated_mod.get_global_var('func_1160')
const_6250 = relay.const([4.119756,-5.721292,6.264302,0.267882,4.343197,8.834752,-6.324747,-2.566494,2.066268,5.334306,7.258149,0.384840,7.939692,8.144389,-8.594990,7.528389,3.813557,9.420307,0.491145,4.511279,-4.321830,-6.749703,8.943830,-9.827666,-4.950977,5.075300,-4.669628,-2.245133,8.579780,-5.893067,0.075287,1.524437,0.409654,-6.217839,9.208940,6.701749,-1.942614,-1.569152,-8.344834,-2.657025,-6.987452,-9.793244,2.131295,-9.473341,8.394339,-7.752220,-7.850112,6.795902,2.250115,-6.103598,6.193144,-8.756932,-9.305971,3.362503,-6.786223,7.868915,5.665438,5.987820,-4.480151,5.769473,3.814322,3.126019,2.752687,0.073712,-8.579713,5.344558,-8.945126,8.492689,-8.423352,1.936708,-2.893961,-6.165253,7.821761,-1.162287,9.136011,5.845992,-9.161053,9.744800,1.457378,-0.550409,-6.424636,8.608730,1.917993,9.439422,2.342504,-2.826628,1.322954,-7.846236,2.289247,4.358283,-7.624981,5.056577,5.027810,5.504141,8.695909,-8.406992,8.376273,-4.370923,-0.371951,4.107324,8.784576,5.184375,-9.337753,8.887639,1.107080,-6.296690,7.505390,-1.640792,-2.831058,0.287341,1.183513,-9.808551,8.649700,7.817512,-4.651625,7.076387,5.725093,8.817525,-4.598112,5.545373,2.846816,-2.170183,6.671697,1.295479,-0.282380,8.671672,-3.390732,-4.771251,4.491658,5.101009,-7.720779,6.862595,3.698416,0.813786,5.989310,0.336593,7.794662,7.846458,-3.964885,-5.835812,5.051314,-7.552645,3.547472,1.463966,8.481879,6.964571,6.232359,5.853977,0.002206,1.332289,4.412778,7.826848,-4.665596,6.329942,0.325386,9.085497,9.076568,-8.628205,-6.265046,5.364216,-0.011813,-9.746897,8.323681,-3.749136,-6.642250,8.053693,9.295285,-5.534502,5.394346,-1.954847,-0.393848,7.442739,-5.620064,-1.072583,3.145969,-6.579833,-3.664798,-3.661803,-3.300765,-6.495543,-5.885098,-8.767099,-8.711718,3.703095,-5.604515,-9.808516,4.436356,-1.931069,4.824906,-9.073789,0.974620,-6.683845,0.396743,5.128927,-0.967020,9.040905,-6.730775,-2.383677,-6.594103,-0.443560,9.748588,2.684313,4.104133,-4.909262,4.362086,4.358665,0.279014,-5.545110,-3.789583,-3.672449,-1.051834,7.960944,0.485326,2.678403,2.000478,-6.068199,4.674752,9.267818,-8.269628,2.659511,6.814442,-8.119796,5.801986,8.364841,-1.731303,-2.478622,9.876382,5.299816,9.903179,9.877826,8.676337,0.864161,6.897042,0.800997,4.514901,9.241392,9.869610,4.877338,8.118909,1.917258,5.924240,-6.273418,9.800284,1.078992,-0.070301,-7.477584,4.942013,-8.340756,-9.075314,-3.476768,2.277135,-9.516809,0.772093,6.570945,6.132219,-2.642971,-8.540848,0.599547,7.215625,-4.933962,5.956953,2.012088,-7.618853,-0.693851,3.472187,-2.714519,-7.758625,7.524231,3.719106,-5.093646,6.657364,5.376122,5.090980,-3.861135,5.837995,9.822659,9.643333,8.794181,-2.248195,-1.514722,-8.928868,-7.391416,-8.671403,-1.644411,-3.747864,-2.245777,-8.936394,2.080922,-7.290114,-7.251565,-5.263732,2.577009,-4.246471,-0.241366,-5.019793,-1.307497,7.454662,1.952964,-8.138295,8.736449,3.331736,8.653862,7.861667,6.603380,-1.905180,-3.429108,8.323472,-9.233354,-0.858947,1.239277,6.294372,-7.445499,-2.965129,-1.476807,2.184797,-0.218849,0.190355,-8.382495,-7.649185,8.025722,3.507591,-6.169195,2.576008,-8.267212,5.842349,-2.564383,5.835100,0.632290,-7.788772,5.878687,-6.534302,-9.148702,6.127632,-9.857644,9.732947,-4.288592,4.349126,7.872208,2.147546,2.145925,1.314873,7.714200,-0.873430,3.280164,2.796102,-5.530414,-0.606701,1.156525,-5.674523,3.842280,8.375422,-2.538673,-6.064962,2.766680,-4.726752,-7.296483,-1.865188,-8.645277,3.387944,-0.443234,-1.965652,-1.999567,2.199142,-4.500995,8.703795,7.567728,3.839891,-6.523527,-1.459497,3.822779,-3.829571,-0.489938,-7.221633,3.498807,-0.395100,-7.922974,-5.649237,0.154388,-8.440907,0.863637,-1.239567,8.613918,2.803503,7.605863,1.769411,-0.442223,1.925647,7.781599,2.572573,5.104738,-0.752008,6.011941,-2.388066,-9.029132,-4.157795,-2.516946,0.196735,3.906923,9.093375,-0.556753,3.590172,-9.942676,3.832468,-8.896989,-4.879957,5.345900,9.502525,-6.985819,0.986714,2.222684,-7.878139,-8.981678,1.578719,-6.757127,3.236664,8.121435,5.509524,8.516608,7.557410,-2.698787,8.744001,-0.705195,-2.951471,-2.304388,-0.355318,6.015296,1.075287,-1.781352,5.087209,-7.452477,1.346719,-9.178940,1.639749,1.443838,3.154001,7.407761,7.894562,-0.003673,-8.039614,-1.839635,-9.873024,3.339199,9.521623,-9.310057,3.305699,9.586214,-7.295600,9.415657,0.667589,8.695601,-9.747988,4.523948,-9.807924,5.993359,-9.859005,4.941708,-0.075711,6.567451,-1.988770,-7.638914,-1.922709,-5.793042,8.967231,-7.720697,2.143668,-5.022030,-7.198999,6.069248,9.131480,2.460073,-1.534788,-1.550920,-6.782673,3.226356,1.754583,-6.148772,2.828367,-4.468661,-0.010781,-1.854060,-1.030692,-4.175442,-7.426628,-9.701762,-2.555892,-6.605493,-2.830558,5.252025,-9.912179,-0.192568,2.502759,2.455807,9.315566,2.075879,-8.978551,6.141489,-4.839592,-9.218928,-5.780344,-4.567102,-8.280698,8.677375,8.033134,9.581999,5.807513,5.823696,-9.344661,-4.557975,1.895515,-0.704094,6.015459,-3.825885,1.203520,-4.528270,8.275008,-9.570239,7.515346,3.321119,8.453626,-5.543731,-6.809533,0.959157,-3.486133,-3.864621,1.280463,-5.818904,4.708191,7.253197,0.127582,-7.215832,-5.505180,-6.696598,-3.203177,5.947563,5.004248,3.458133,1.808095,-6.386623,-0.998480,0.923575,1.191165,-9.312804,8.622394,0.216638,3.391711,9.513966,7.475467,-9.886544,7.351932,4.650405,0.789061,-0.699486,9.170280,5.966169,-7.962997,4.066865,-5.495483,-7.975136,7.371595,-5.157492,1.995665,6.414846,7.415294,-9.871910,0.617427,6.837801,-5.083854,4.338490,4.060016,8.021248,-9.929356,5.176360,-0.356783,-1.003912,-7.087396,5.851657,-0.797185,-9.404885,-8.596651,6.998583,-2.954524,-7.200388,-2.289423,-4.908337,-2.930676,9.046282,-2.238294,7.582143,-5.309461,0.662259,-4.078038,-5.476105,8.897188,4.966944,-3.207321,2.836151,8.060037,6.902429,0.827570,6.669544,-9.614794,-2.744919,-2.762711,2.528448,1.052451,5.338939,5.080600,7.222128,-9.040083,-2.127390,3.794175,3.355960,-0.089595,-0.004539,-2.432155,9.999390,0.320158,7.101867,-8.767157,-1.936737,3.499884,8.971379,-6.546605,1.906299,2.523292,2.452577,6.314108,-3.818263,7.261662,-6.732700,-5.973944,-6.618746,6.799775,1.618035,4.091102,-4.968956,8.589467,2.878616,-6.001345,-9.989614,4.036299,-3.676281,-1.162757,-5.415721,7.976291,8.368736,-7.556498,1.906815,-5.749333,8.965080,6.208564,-0.779021,-4.985397,-8.710642,0.163067,-8.722663,-7.647957,3.281230,1.037905,-9.162704,-5.154908,6.731364,-7.556714,5.909875,1.394678,-5.015712,-7.901375,-5.805079,-7.071238,7.081359,1.941088,-0.462356,-6.566535,8.461607,9.294610,-6.522392,3.832798,4.131823,3.477880,-5.824910,1.157602,7.057601,-8.931212,-9.884576,-9.599195,-8.123288,-1.140271,6.678239,-0.062157,-1.193642,-8.189806,-8.027811,-7.732203,3.713982,-5.906941,9.751456,-1.541211,8.804491,-7.045648,-5.406419,2.494905,-0.235687,4.348435,-7.254128,-7.240640,-7.338727,7.814380,9.516192,-0.201375,3.977366,-2.775955,-9.553759,5.291492,9.567421,-6.374603,-6.939099,-2.231309,9.094363,-3.393300,-1.830104,-5.546489,-6.354068,3.093780,-7.263374,-7.575326,-6.751417,-0.951468,-7.059185,-6.184175,1.284296,7.148449,-3.206201,-8.226045,8.935301,7.543084,-6.435711,-1.317365,6.350616,-8.594638,2.057285,-7.683943,3.233323,2.610471,-5.101992,-2.671042,-2.925464,-8.380987,0.353696,-7.397839,8.901178,-3.181573,9.234451,-7.232585,-6.434121,-5.685751,-0.673772,8.072538,6.441316,9.927220,-6.173827,9.600211,0.354409,4.563656,0.444624,7.088096,1.566971,-2.529057,7.206582,-0.848616,4.293459,5.995751,1.829061,0.221505,8.342066,-4.750267,-2.111962,-2.812520,3.634539,2.698459,2.899725,-3.880374,-7.552919,-9.186079,6.464381,-5.819549,3.166752,-9.408347,-7.721824,-4.211519,-4.232703,-4.326082,-0.487969], dtype = "float64")#candidate|6250|(792,)|const|float64
var_6251 = relay.var("var_6251", dtype = "float64", shape = (65, 5))#candidate|6251|(65, 5)|var|float64
call_6249 = relay.TupleGetItem(func_1156_call(relay.reshape(const_6250.astype('float64'), [12, 6, 11]), relay.reshape(const_6250.astype('float64'), [12, 6, 11]), relay.reshape(var_6251.astype('float64'), [325,]), ), 0)
call_6252 = relay.TupleGetItem(func_1160_call(relay.reshape(const_6250.astype('float64'), [12, 6, 11]), relay.reshape(const_6250.astype('float64'), [12, 6, 11]), relay.reshape(var_6251.astype('float64'), [325,]), ), 0)
output = relay.Tuple([call_6213,call_6220,uop_6235,call_6249,const_6250,var_6251,])
output2 = relay.Tuple([call_6214,call_6222,uop_6235,call_6252,const_6250,var_6251,])
func_6253 = relay.Function([var_6251,], output)
mod['func_6253'] = func_6253
mod = relay.transform.InferType()(mod)
mutated_mod['func_6253'] = func_6253
mutated_mod = relay.transform.InferType()(mutated_mod)
var_6254 = relay.var("var_6254", dtype = "float64", shape = (65, 5))#candidate|6254|(65, 5)|var|float64
func_6253_call = mutated_mod.get_global_var('func_6253')
call_6255 = func_6253_call(var_6254)
output = call_6255
func_6256 = relay.Function([var_6254], output)
mutated_mod['func_6256'] = func_6256
mutated_mod = relay.transform.InferType()(mutated_mod)
var_6310 = relay.var("var_6310", dtype = "float64", shape = (1, 16, 3))#candidate|6310|(1, 16, 3)|var|float64
var_6311 = relay.var("var_6311", dtype = "float64", shape = (3, 16, 3))#candidate|6311|(3, 16, 3)|var|float64
bop_6312 = relay.divide(var_6310.astype('float64'), var_6311.astype('float64')) # shape=(3, 16, 3)
uop_6321 = relay.tan(bop_6312.astype('float64')) # shape=(3, 16, 3)
bop_6331 = relay.minimum(uop_6321.astype('int64'), var_6310.astype('int64')) # shape=(3, 16, 3)
uop_6345 = relay.log10(bop_6331.astype('float32')) # shape=(3, 16, 3)
output = relay.Tuple([uop_6345,])
output2 = relay.Tuple([uop_6345,])
func_6353 = relay.Function([var_6310,var_6311,], output)
mod['func_6353'] = func_6353
mod = relay.transform.InferType()(mod)
mutated_mod['func_6353'] = func_6353
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6353_call = mutated_mod.get_global_var('func_6353')
var_6355 = relay.var("var_6355", dtype = "float64", shape = (1, 16, 3))#candidate|6355|(1, 16, 3)|var|float64
var_6356 = relay.var("var_6356", dtype = "float64", shape = (3, 16, 3))#candidate|6356|(3, 16, 3)|var|float64
call_6354 = func_6353_call(var_6355,var_6356,)
output = call_6354
func_6357 = relay.Function([var_6355,var_6356,], output)
mutated_mod['func_6357'] = func_6357
mutated_mod = relay.transform.InferType()(mutated_mod)
func_4186_call = mod.get_global_var('func_4186')
func_4188_call = mutated_mod.get_global_var('func_4188')
call_6359 = relay.TupleGetItem(func_4186_call(), 0)
call_6360 = relay.TupleGetItem(func_4188_call(), 0)
func_3042_call = mod.get_global_var('func_3042')
func_3046_call = mutated_mod.get_global_var('func_3046')
var_6364 = relay.var("var_6364", dtype = "float64", shape = (182,))#candidate|6364|(182,)|var|float64
var_6365 = relay.var("var_6365", dtype = "float32", shape = (1890,))#candidate|6365|(1890,)|var|float32
call_6363 = relay.TupleGetItem(func_3042_call(relay.reshape(var_6364.astype('float64'), [14, 13, 1]), relay.reshape(var_6365.astype('float32'), [9, 210]), ), 8)
call_6366 = relay.TupleGetItem(func_3046_call(relay.reshape(var_6364.astype('float64'), [14, 13, 1]), relay.reshape(var_6365.astype('float32'), [9, 210]), ), 8)
func_4652_call = mod.get_global_var('func_4652')
func_4653_call = mutated_mod.get_global_var('func_4653')
call_6367 = relay.TupleGetItem(func_4652_call(), 0)
call_6368 = relay.TupleGetItem(func_4653_call(), 0)
output = relay.Tuple([call_6359,call_6363,var_6364,var_6365,call_6367,])
output2 = relay.Tuple([call_6360,call_6366,var_6364,var_6365,call_6368,])
func_6372 = relay.Function([var_6364,var_6365,], output)
mod['func_6372'] = func_6372
mod = relay.transform.InferType()(mod)
mutated_mod['func_6372'] = func_6372
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6372_call = mutated_mod.get_global_var('func_6372')
var_6374 = relay.var("var_6374", dtype = "float64", shape = (182,))#candidate|6374|(182,)|var|float64
var_6375 = relay.var("var_6375", dtype = "float32", shape = (1890,))#candidate|6375|(1890,)|var|float32
call_6373 = func_6372_call(var_6374,var_6375,)
output = call_6373
func_6376 = relay.Function([var_6374,var_6375,], output)
mutated_mod['func_6376'] = func_6376
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3870_call = mod.get_global_var('func_3870')
func_3872_call = mutated_mod.get_global_var('func_3872')
call_6380 = func_3870_call()
call_6381 = func_3870_call()
output = call_6380
output2 = call_6381
func_6391 = relay.Function([], output)
mod['func_6391'] = func_6391
mod = relay.transform.InferType()(mod)
output = func_6391()
func_6392 = relay.Function([], output)
mutated_mod['func_6392'] = func_6392
mutated_mod = relay.transform.InferType()(mutated_mod)
var_6484 = relay.var("var_6484", dtype = "float64", shape = (15, 10, 2))#candidate|6484|(15, 10, 2)|var|float64
uop_6485 = relay.log(var_6484.astype('float64')) # shape=(15, 10, 2)
output = relay.Tuple([uop_6485,])
output2 = relay.Tuple([uop_6485,])
func_6496 = relay.Function([var_6484,], output)
mod['func_6496'] = func_6496
mod = relay.transform.InferType()(mod)
var_6497 = relay.var("var_6497", dtype = "float64", shape = (15, 10, 2))#candidate|6497|(15, 10, 2)|var|float64
output = func_6496(var_6497)
func_6498 = relay.Function([var_6497], output)
mutated_mod['func_6498'] = func_6498
mutated_mod = relay.transform.InferType()(mutated_mod)
func_5856_call = mod.get_global_var('func_5856')
func_5858_call = mutated_mod.get_global_var('func_5858')
call_6593 = func_5856_call()
call_6594 = func_5856_call()
output = relay.Tuple([call_6593,])
output2 = relay.Tuple([call_6594,])
func_6600 = relay.Function([], output)
mod['func_6600'] = func_6600
mod = relay.transform.InferType()(mod)
mutated_mod['func_6600'] = func_6600
mutated_mod = relay.transform.InferType()(mutated_mod)
func_6600_call = mutated_mod.get_global_var('func_6600')
call_6601 = func_6600_call()
output = call_6601
func_6602 = relay.Function([], output)
mutated_mod['func_6602'] = func_6602
mutated_mod = relay.transform.InferType()(mutated_mod)
func_3786_call = mod.get_global_var('func_3786')
func_3787_call = mutated_mod.get_global_var('func_3787')
call_6603 = func_3786_call()
call_6604 = func_3786_call()
output = relay.Tuple([call_6603,])
output2 = relay.Tuple([call_6604,])
func_6610 = relay.Function([], output)
mod['func_6610'] = func_6610
mod = relay.transform.InferType()(mod)
output = func_6610()
func_6611 = relay.Function([], output)
mutated_mod['func_6611'] = func_6611
mutated_mod = relay.transform.InferType()(mutated_mod)
const_6612 = relay.const([[[2.411759,2.742129,4.285142,-4.949236,7.782262,-3.968025,-6.583733,9.176056,6.072910],[7.896397,-7.059114,-2.440717,4.386313,-0.476876,4.623633,-5.517389,2.742624,-1.453158],[4.959968,8.096033,-2.486238,-7.705734,-3.565099,-8.191378,5.698827,8.857783,4.434569],[7.320592,6.299971,3.291957,1.091027,0.345182,4.864853,1.997730,6.209891,7.457725],[-8.015307,-5.972703,2.786615,-5.540563,-2.995639,9.085397,6.227478,-6.694860,-3.415257],[5.680426,-2.006347,-3.664502,1.624619,-7.518023,-7.405566,-8.267989,-1.769213,1.534924],[1.856880,-7.806900,0.350373,-2.580109,-5.775355,-3.769268,-6.171364,-3.476206,-8.707467],[8.164945,-3.600979,-6.671956,2.291043,8.377245,4.873326,-0.620320,-3.644171,-1.114527],[-0.010232,-1.073876,-9.656855,5.899119,7.401744,3.330366,-3.216542,7.608935,-1.441037],[-4.716187,-8.994915,5.432043,-7.197573,6.769743,-7.467097,3.207931,-6.526979,7.505338],[-7.440940,4.665241,-7.325832,2.497301,8.594086,5.782694,1.499579,-8.982794,6.949596]],[[0.198481,9.748886,-8.878147,7.449879,-4.823473,-7.091060,-8.205264,-8.650594,-8.069766],[-6.183218,4.687290,-7.983487,4.881805,-3.255801,7.346264,-2.816043,-0.148100,4.253768],[6.679959,-2.279846,-4.953457,-4.983945,5.640233,8.072837,1.261348,5.666762,7.919334],[-4.359641,7.296153,-9.943377,-1.949522,-2.323748,-8.367019,-4.035270,-1.445355,5.967946],[7.853019,4.064620,-9.156107,-0.383644,-9.234604,8.031391,6.613730,-8.917705,-4.298537],[9.335418,2.636224,9.371808,-2.389508,-8.559911,-9.062581,2.821365,6.359596,-7.198433],[0.288272,8.202705,3.613836,-7.610216,-3.745226,8.605237,7.440980,-0.946426,-6.418114],[5.395842,6.470568,9.725889,2.568955,8.577111,8.496451,7.363264,-0.504978,-7.746308],[1.626605,-6.319901,-9.748997,-9.067844,-1.402408,-8.705048,2.341300,2.127167,8.246945],[-1.337651,-1.169140,1.051305,-4.117399,-8.363874,-5.241870,-3.334979,8.549332,7.876866],[-4.223049,-3.745660,-7.951906,-6.117881,-0.169909,-4.184373,-1.854246,-0.743813,-4.728548]],[[6.516516,-5.564715,8.489191,-6.436247,-7.171188,-5.589246,2.189995,-0.052080,-0.643916],[1.748982,-2.825859,-2.564530,2.882174,5.363663,7.738982,-7.887214,-8.902221,6.580992],[1.212872,0.860553,3.379279,5.153234,7.666974,-7.947779,6.009372,-1.537552,1.135881],[-8.604001,7.675430,3.880894,-0.918988,9.104692,-2.902145,-4.295209,0.812144,-3.528936],[-6.726593,3.681006,0.952574,4.852841,4.655277,-5.454831,-5.253048,-6.238632,-9.881510],[0.715408,2.930315,-8.702563,7.467313,5.946214,-9.649999,7.817705,5.808404,-1.797693],[8.162776,1.992166,6.814514,3.035716,0.502835,-2.834682,3.969481,-6.395772,-4.787840],[5.841103,8.678734,-8.891731,-0.705647,-5.816672,2.164537,-8.086066,-9.838565,3.834072],[7.830382,-1.098591,-4.635741,-8.061261,-3.132921,-8.544177,9.764939,-8.567691,-3.170514],[0.046208,-4.929454,0.168174,-6.777598,-0.024026,7.198141,3.458220,9.890747,6.893900],[-9.100086,3.396836,-7.948100,6.116152,9.643628,3.654829,1.832852,-8.668553,-6.150680]]], dtype = "float32")#candidate|6612|(3, 11, 9)|const|float32
uop_6613 = relay.asinh(const_6612.astype('float32')) # shape=(3, 11, 9)
func_1498_call = mod.get_global_var('func_1498')
func_1501_call = mutated_mod.get_global_var('func_1501')
var_6624 = relay.var("var_6624", dtype = "int32", shape = (140,))#candidate|6624|(140,)|var|int32
call_6623 = func_1498_call(relay.reshape(var_6624.astype('int32'), [10, 14, 1]))
call_6625 = func_1498_call(relay.reshape(var_6624.astype('int32'), [10, 14, 1]))
bop_6631 = relay.floor_divide(call_6623.astype('float64'), relay.reshape(var_6624.astype('float64'), relay.shape_of(call_6623))) # shape=(10, 14, 1)
bop_6634 = relay.floor_divide(call_6625.astype('float64'), relay.reshape(var_6624.astype('float64'), relay.shape_of(call_6625))) # shape=(10, 14, 1)
uop_6650 = relay.log10(uop_6613.astype('float64')) # shape=(3, 11, 9)
output = relay.Tuple([bop_6631,uop_6650,])
output2 = relay.Tuple([bop_6634,uop_6650,])
F = relay.Function([var_6624,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_6624,], output2)
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
input_6624= np.array([-2,-10,-10,4,-4,-6,-7,5,-6,10,-2,-2,-6,-2,4,6,6,6,-2,-1,-7,1,9,-1,-3,-7,-4,4,7,1,-4,5,4,-8,-10,-3,6,-2,-1,10,-6,9,-9,-9,10,6,-8,-4,10,9,2,10,7,3,-7,-4,-1,2,-7,-4,5,-1,-1,-7,10,5,-9,8,-9,6,5,10,4,-7,6,-9,3,4,-8,5,7,8,6,-2,8,-2,9,-10,-5,7,10,10,8,5,7,4,1,7,-9,-4,-8,-1,-8,1,9,5,-6,1,1,9,-8,-9,4,-8,-9,-1,1,-4,-4,2,-5,-3,2,-4,-9,-10,-8,10,8,-7,2,-4,10,2,4,-5,1,-9,-1,-9], dtype='int32')
module1.set_input('var_6624', input_6624)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_6624, )
res3 = intrp3.evaluate()(input_6624, )
res4 = intrp4.evaluate()(input_6624, )
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
module5.set_input('var_6624', input_6624)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_6624, )
res7 = intrp7.evaluate()(input_6624, )
res8 = intrp8.evaluate()(input_6624, )
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
module9.set_input('var_6624', input_6624)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_6624, )
res11 = intrp11.evaluate()(input_6624, )
res12 = intrp12.evaluate()(input_6624, )
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
module13.set_input('var_6624', input_6624)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_6624, )
res15 = intrp15.evaluate()(input_6624, )
res16 = intrp16.evaluate()(input_6624, )
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
module17.set_input('var_6624', input_6624)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_6624, )
res19 = intrp19.evaluate()(input_6624, )
res20 = intrp20.evaluate()(input_6624, )
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
module21.set_input('var_6624', input_6624)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_6624, )
res23 = intrp23.evaluate()(input_6624, )
res24 = intrp24.evaluate()(input_6624, )
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

'''34: TVMFuncCall
33: _ZNSt17_Function_handlerIFvN3tvm7runtime7TVM
32: tvm::relay::backend::RelayBuildModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#3}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
31: tvm::relay::backend::RelayBuildModule::BuildRelay(tvm::IRModule, tvm::runtime::String const&)
30: tvm::relay::backend::ExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function const&, tvm::runtime::String)
29: std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), tvm::relay::backend::GraphExecutorCodegenModule::GetFunction(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#2}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)
28: tvm::relay::backend::GraphExecutorCodegen::Codegen(tvm::IRModule, tvm::relay::Function, tvm::runtime::String)
27: tvm::relay::GraphPlanMemory(tvm::relay::Function const&)
26: tvm::relay::StorageAllocator::Plan(tvm::relay::Function const&)
25: tvm::relay::ExprVisitor::VisitExpr(tvm::RelayExpr const&)
24: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
23: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
22: tvm::relay::transform::DeviceAwareExprVisitor::VisitExpr_(tvm::relay::FunctionNode const*)
21: tvm::relay::StorageAllocaBaseVisitor::DeviceAwareVisitExpr_(tvm::relay::FunctionNode const*)
20: tvm::relay::StorageAllocaBaseVisitor::GetToken(tvm::RelayExpr const&)
19: tvm::relay::ExprVisitor::VisitExpr(tvm::RelayExpr const&)
18: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
17: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
16: tvm::relay::StorageAllocaBaseVisitor::VisitExpr_(tvm::relay::TupleNode const*)
15: tvm::relay::StorageAllocaBaseVisitor::GetToken(tvm::RelayExpr const&)
14: tvm::relay::ExprVisitor::VisitExpr(tvm::RelayExpr const&)
13: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
12: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
11: tvm::relay::transform::DeviceAwareExprVisitor::VisitExpr_(tvm::relay::CallNode const*)
10: tvm::relay::StorageAllocator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
9: tvm::relay::StorageAllocaBaseVisitor::GetToken(tvm::RelayExpr const&)
8: tvm::relay::ExprVisitor::VisitExpr(tvm::RelayExpr const&)
7: tvm::relay::ExprFunctor<void (tvm::RelayExpr const&)>::VisitExpr(tvm::RelayExpr const&)
6: _ZZN3tvm5relay11ExprFunctorIFvRKNS_9Rela
5: tvm::relay::transform::DeviceAwareExprVisitor::VisitExpr_(tvm::relay::CallNode const*)
4: tvm::relay::StorageAllocator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*)
3: tvm::relay::StorageAllocaBaseVisitor::CreateToken(tvm::RelayExprNode const*, bool)
2: tvm::relay::StorageAllocator::CreateTokenOnDevice(tvm::RelayExprNode const*, tvm::VirtualDevice const&, bool)
1: tvm::relay::StorageAllocator::Request(tvm::relay::StorageToken*)
0: tvm::relay::StorageAllocator::GetMemorySize(tvm::relay::StorageToken*)

'''