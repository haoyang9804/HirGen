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
var_1 = relay.var("var_1", dtype = "float64", shape = (10, 1))#candidate|1|(10, 1)|var|float64
bop_2 = relay.power(var_0.astype('float64'), var_1.astype('float64')) # shape=(10, 1)
bop_5 = relay.maximum(var_0.astype('uint8'), bop_2.astype('uint8')) # shape=(10, 1)
uop_8 = relay.acos(bop_5.astype('float64')) # shape=(10, 1)
output = uop_8
output2 = uop_8
func_10 = relay.Function([var_0,var_1,], output)
mod['func_10'] = func_10
mod = relay.transform.InferType()(mod)
mutated_mod['func_10'] = func_10
mutated_mod = relay.transform.InferType()(mutated_mod)
func_10_call = mutated_mod.get_global_var('func_10')
var_12 = relay.var("var_12", dtype = "float64", shape = ())#candidate|12|()|var|float64
var_13 = relay.var("var_13", dtype = "float64", shape = (10, 1))#candidate|13|(10, 1)|var|float64
call_11 = func_10_call(var_12,var_13,)
output = call_11
func_14 = relay.Function([var_12,var_13,], output)
mutated_mod['func_14'] = func_14
mutated_mod = relay.transform.InferType()(mutated_mod)
const_16 = relay.const([[[2.891375,-3.549195,7.547135,2.254525,-0.879899,-0.413404],[-1.303046,-6.358864,-9.251862,0.239280,-3.767408,0.482017],[-3.823981,5.056794,-6.456119,-6.491246,-3.784225,9.308167],[2.495902,6.663017,6.854472,2.467863,-2.640230,5.511036],[-2.547067,-4.696703,-5.158563,6.691087,7.197113,1.095273],[6.954188,1.952706,-4.096564,7.351436,5.139357,6.535187],[-6.132374,-8.264257,2.035157,-5.938697,-7.722781,-2.258341],[-7.217802,-2.875724,-7.466214,-5.121059,1.669139,9.625886],[-5.225897,8.901657,-8.404958,-3.242789,6.859770,0.862019],[-5.995288,7.289222,-6.249080,6.796094,7.007427,-7.283413],[-4.179928,8.970961,-2.435204,0.114099,1.552617,2.797854]],[[-5.382590,-4.587945,-6.334373,-9.181475,2.613406,-1.800696],[0.807316,1.182637,2.568572,-0.391370,-6.781887,4.407929],[2.705626,3.988089,3.330319,-2.736386,-1.135943,5.471257],[-2.084669,1.764986,3.167312,2.813830,6.779345,6.375063],[5.284323,-5.938312,0.925641,-4.437342,7.820712,7.673946],[-6.278255,-2.277996,2.652625,-1.205560,5.375449,2.052143],[9.372912,3.754188,9.072930,4.584492,-4.896319,-4.797489],[3.477005,-5.932300,-2.862418,9.687453,2.066508,3.414024],[7.976264,-4.753626,6.539373,-9.252661,7.705876,-3.116503],[-9.344743,0.896295,0.170520,5.839416,-8.397582,6.137469],[4.300292,-1.484627,-7.040965,6.126340,2.640875,-9.338886]],[[-4.320660,7.055383,-2.513862,-6.994565,4.798034,-3.562305],[-8.967912,7.726476,7.745779,7.609952,8.474223,-0.273958],[-7.455287,7.873606,1.004824,-3.859524,-7.238450,6.857298],[2.563463,-4.174560,4.720776,-8.318586,-3.998752,4.122926],[-4.388114,-1.148346,-6.307372,-1.949990,0.686381,-2.075128],[9.628097,-5.897181,-5.146535,1.847174,8.297042,-7.723450],[-1.365957,-8.769107,-3.071097,-9.403260,3.728900,-7.461184],[3.634801,-2.477780,-6.462740,-6.353366,6.001835,-9.564340],[-7.979681,-5.637450,5.043846,2.431984,7.460607,-5.778293],[2.936918,-1.882284,4.316506,-6.128134,4.947608,3.398155],[9.122941,-0.424876,-4.142420,-4.200444,-7.709122,6.687573]],[[0.061833,9.116898,-4.789626,-9.159548,-4.055687,-6.924244],[8.032969,5.656695,5.321700,-1.151961,-0.637175,-0.957183],[3.326060,5.234763,4.350632,7.786908,5.114686,7.429220],[-4.473797,2.190610,-2.299981,3.091292,-1.121502,3.873725],[8.657795,-9.296949,-4.371274,6.484627,2.655559,6.478264],[9.107448,3.014964,1.404600,-2.765944,-6.733397,-9.572111],[-0.373568,-2.016591,-7.462843,-0.655525,-7.411443,2.949778],[5.173334,-6.040739,-6.767996,6.021244,5.176443,-1.800938],[0.949421,-4.502496,2.718014,8.945759,-1.413752,-7.391798],[6.404697,2.189092,-6.802040,-7.236167,1.291400,-6.755405],[-3.681291,2.624642,7.229548,-4.928206,-1.931697,-4.919320]],[[6.899376,4.639336,-4.835395,1.074590,9.079605,1.681390],[2.448304,1.066791,-8.698640,-1.473770,-3.841339,-6.646583],[-9.812592,-7.823480,-1.061585,-6.721589,-9.953010,1.611619],[3.237355,-9.679036,5.813660,-9.072196,2.534689,3.369023],[-8.843085,-1.178858,4.192385,-9.854554,2.078954,3.865050],[6.519491,-3.870663,2.243746,-9.656874,-3.083093,2.567155],[-0.166299,-2.995341,-6.345232,-8.529985,-8.363544,1.035364],[8.724598,9.221314,-9.068114,-7.205698,-0.130404,-6.196470],[5.793160,-2.236124,6.269724,1.678778,-2.160609,-9.189226],[-4.641195,9.649296,4.405397,-7.425774,-4.460262,-9.883690],[-0.898615,-3.649793,6.649900,0.165256,-2.219211,-5.654911]],[[1.864451,4.975507,4.744041,-2.644950,-8.641355,-9.701156],[-4.802372,3.300635,-8.646365,-7.273967,-8.179123,0.768089],[7.452366,-0.344558,5.561341,6.645121,6.951356,-6.598782],[-8.547622,-3.368430,-9.937365,7.381193,1.213845,6.108420],[-4.802307,4.838451,-8.673667,-6.241158,4.995700,3.944586],[2.537389,2.947778,-4.817913,8.309032,7.049668,6.340772],[6.113873,-2.565647,0.391480,-5.179439,3.736620,0.983723],[0.703295,5.233176,-4.082513,7.661955,-6.747733,-2.176314],[-7.105707,2.163666,0.875687,8.776195,7.701002,-6.148812],[1.600750,-1.667113,-2.323347,-0.400443,0.153869,-9.457151],[1.486195,1.732998,-7.857152,-8.878512,-9.509167,-6.113450]],[[-4.021900,-7.974214,-0.560149,3.915740,7.580657,-2.092127],[6.661125,3.607201,4.344596,-2.374868,4.639814,-5.928713],[-8.798790,-5.053671,-4.873848,-1.438529,-1.912114,4.679208],[2.748970,-3.377986,-6.694146,-7.547139,-1.438839,5.500042],[1.943349,-9.715128,4.885783,8.389694,2.464307,-4.734544],[-4.096277,-4.851652,7.220363,3.854090,-7.476495,-2.668780],[2.702149,-0.572470,-5.093935,0.781672,-1.269304,4.455453],[-3.007746,9.387059,-1.962063,1.808948,-1.119981,-1.410341],[1.547919,4.259908,3.824642,8.484280,-3.856730,-4.424011],[9.581806,0.653544,9.213870,4.229395,1.778381,1.505393],[3.090752,-5.226639,1.687871,4.888050,3.816055,-2.015855]],[[-0.855126,0.802228,-1.743089,-3.769641,7.327245,6.238167],[-3.315887,-4.821249,3.627244,9.856328,-0.379937,-8.991682],[-3.912266,7.944288,-2.570572,9.548866,5.440215,-0.817657],[-8.700797,3.604845,-8.656679,1.028667,4.690834,-4.630766],[0.739710,0.093364,2.960254,4.561322,-0.878915,4.696344],[0.388888,7.891567,2.987654,-7.344090,-6.653863,8.178549],[-5.936328,2.776739,-9.683717,-7.541450,-0.844270,-8.988230],[8.075717,7.806566,-3.189433,5.102488,8.330050,-1.639295],[-7.367008,6.081769,-9.282991,4.071114,0.280458,-5.010425],[-0.153909,7.643086,3.152080,-0.586737,-3.704194,-9.681670],[0.582577,-9.193657,-1.921081,0.969384,5.076567,-8.259112]],[[-7.207616,-2.892355,-2.541366,7.525465,6.422601,7.823444],[-8.080213,5.849131,-0.830089,7.608502,2.472834,-6.841696],[-4.300428,-5.528801,-5.109869,4.721901,-0.174914,-1.609622],[7.843061,8.342091,8.721726,4.643594,1.967127,-5.652225],[-2.315391,3.367505,2.326907,9.775540,0.036024,7.857942],[-8.111031,-1.835636,-7.601788,-1.698010,2.245599,-0.465091],[-2.227413,-6.950700,-0.839537,3.963915,8.552304,1.328951],[7.666309,-8.336421,-0.993212,-7.516748,5.253747,-5.847963],[-6.422371,9.947685,-1.925943,-7.005321,1.509719,-8.422112],[6.489333,-5.736725,-6.339426,3.926943,-2.644315,8.899579],[6.999028,-1.254908,-9.473961,-4.083072,8.707535,-8.187584]],[[9.224882,-9.282786,-4.160488,-1.902594,0.500463,-5.421260],[-9.455091,4.687749,3.483947,-7.673389,-3.228879,2.558001],[6.537126,3.405824,2.919438,-7.003469,-3.957176,9.841041],[-4.242074,-4.783099,-6.819913,3.457927,-1.942590,-2.868453],[0.722940,-4.885373,8.490921,3.184833,3.798043,8.628197],[-7.256225,9.450606,-2.628707,5.011281,4.902832,-3.030351],[-9.979350,-8.762448,-3.727571,4.102941,4.321647,-0.774411],[2.914639,6.270194,-8.017133,-7.088467,4.451647,-4.555414],[-9.020627,-9.954298,8.180986,-8.420230,-0.364979,9.981715],[0.916585,3.386718,-9.540052,8.189521,6.654592,5.544528],[0.247240,-1.646823,4.154172,-1.466199,9.366001,3.723604]],[[-5.456501,-5.625415,6.429347,4.848583,-4.071875,2.905357],[-9.573192,0.984901,-4.371399,-0.964441,-6.184233,-6.145498],[2.542875,7.235454,1.456682,5.762598,-1.868890,-8.937632],[5.790292,-3.649438,-2.381300,6.744470,-3.424241,-6.233205],[3.264346,-7.344560,-6.102177,1.316401,-6.809529,-3.390726],[-3.626593,4.732894,3.216149,-2.169133,1.323247,6.932096],[-6.082485,5.224261,1.072735,3.714674,-6.185897,-2.370181],[-5.232650,-5.967924,-4.249915,-1.402762,-4.045968,-5.167880],[-4.496513,6.125251,7.805458,7.168748,4.252128,3.746508],[8.593802,0.307657,-4.215378,-0.107118,5.032078,4.006420],[0.492066,-0.779171,8.574478,3.681183,3.484312,-7.773606]],[[0.944936,2.221209,-3.442270,1.496561,4.739481,6.731781],[-3.252202,3.028708,-9.141865,-6.748356,-6.058631,8.641683],[2.176559,4.903629,9.656554,5.919362,-2.335595,5.221841],[2.194196,-4.025583,9.470199,3.770370,8.497857,-2.705726],[4.390525,5.842283,7.129864,-7.725135,2.178654,-7.215982],[5.799141,-1.490575,-6.307219,-3.355900,-1.797991,-6.841393],[-7.896197,-3.713779,-8.913035,9.628717,6.957517,-1.802100],[1.463126,1.039833,-9.206017,-3.559599,2.170343,9.547905],[-1.853700,-1.454341,-6.827876,2.638897,-2.030710,-6.294568],[-3.795492,-3.584473,-0.993850,-4.306243,8.187552,7.411200],[5.442241,4.306836,-5.939915,-1.265156,-0.582120,6.308979]],[[-1.157045,5.268202,-3.952246,5.591664,8.236175,-0.853576],[-3.235024,-3.303615,-7.766969,-1.712442,3.486291,5.718001],[-7.491661,-3.943238,-0.151591,-6.592941,-9.669104,7.978743],[0.794464,1.623222,-7.451556,1.182810,3.250865,0.787994],[2.425200,9.588270,5.105308,-0.994435,8.412292,0.898137],[-7.631638,-9.952069,2.166054,-8.002109,-4.681898,-6.555529],[6.059152,-7.502634,0.597944,5.108299,0.764346,2.335224],[1.568878,-3.163837,-0.088759,-7.204485,1.326205,-1.530068],[-8.252280,-0.042837,-4.610441,6.299039,-6.249656,-8.365182],[8.829297,-7.630581,5.896950,-1.881759,5.699984,7.533931],[-0.397610,-5.097521,-5.029975,-7.603393,1.782868,5.770559]]], dtype = "float64")#candidate|16|(13, 11, 6)|const|float64
var_17 = relay.var("var_17", dtype = "float64", shape = (13, 11, 6))#candidate|17|(13, 11, 6)|var|float64
bop_18 = relay.greater_equal(const_16.astype('bool'), relay.reshape(var_17.astype('bool'), relay.shape_of(const_16))) # shape=(13, 11, 6)
var_21 = relay.var("var_21", dtype = "float64", shape = (13, 11, 6))#candidate|21|(13, 11, 6)|var|float64
bop_22 = relay.bitwise_xor(const_16.astype('int8'), relay.reshape(var_21.astype('int8'), relay.shape_of(const_16))) # shape=(13, 11, 6)
var_25 = relay.var("var_25", dtype = "bool", shape = (13, 11, 6))#candidate|25|(13, 11, 6)|var|bool
bop_26 = relay.subtract(bop_18.astype('int16'), relay.reshape(var_25.astype('int16'), relay.shape_of(bop_18))) # shape=(13, 11, 6)
uop_29 = relay.sqrt(bop_22.astype('float32')) # shape=(13, 11, 6)
uop_31 = relay.cosh(uop_29.astype('float32')) # shape=(13, 11, 6)
bop_33 = relay.power(uop_31.astype('float32'), relay.reshape(const_16.astype('float32'), relay.shape_of(uop_31))) # shape=(13, 11, 6)
bop_36 = relay.greater(uop_31.astype('bool'), relay.reshape(bop_33.astype('bool'), relay.shape_of(uop_31))) # shape=(13, 11, 6)
bop_39 = relay.less(bop_18.astype('bool'), relay.reshape(bop_36.astype('bool'), relay.shape_of(bop_18))) # shape=(13, 11, 6)
bop_42 = relay.logical_xor(bop_18.astype('int32'), relay.reshape(bop_39.astype('int32'), relay.shape_of(bop_18))) # shape=(13, 11, 6)
bop_45 = relay.left_shift(uop_29.astype('int16'), relay.reshape(uop_31.astype('int16'), relay.shape_of(uop_29))) # shape=(13, 11, 6)
bop_48 = relay.less(uop_29.astype('bool'), relay.reshape(bop_39.astype('bool'), relay.shape_of(uop_29))) # shape=(13, 11, 6)
uop_51 = relay.atan(uop_29.astype('float32')) # shape=(13, 11, 6)
bop_53 = relay.add(bop_36.astype('float64'), relay.reshape(uop_51.astype('float64'), relay.shape_of(bop_36))) # shape=(13, 11, 6)
bop_56 = relay.logical_or(var_17.astype('bool'), relay.reshape(var_25.astype('bool'), relay.shape_of(var_17))) # shape=(13, 11, 6)
uop_59 = relay.atanh(var_25.astype('float64')) # shape=(13, 11, 6)
const_61 = relay.const([[[1.486173,3.923912,-3.822212,-8.403295,-4.076466,6.886502],[-7.891483,-5.684024,-1.668616,-2.767794,-3.726035,-3.817272],[-5.761165,7.674193,1.333708,6.290492,-6.448785,-3.391369],[8.954538,6.749926,-3.123071,-1.401447,-7.992031,-1.459551],[6.274002,-3.682670,6.825061,5.791912,-9.771488,2.815133],[3.822207,-8.305877,1.013334,-9.497785,7.215832,-1.634880],[-9.402849,9.541825,-5.493417,3.986209,-0.310875,4.312938],[8.052679,-8.058904,8.007310,0.436928,-0.402611,-0.892445],[-9.685601,3.645779,-0.455637,1.940917,-5.896841,7.841941],[-7.277745,-2.015078,-3.964946,8.151355,-3.081087,1.701150],[5.894200,-6.802181,-8.926507,-6.389567,0.476622,2.647839]],[[1.611923,-2.786228,4.998245,-6.927046,-2.103153,-4.667465],[-5.637622,-2.265353,8.123243,-4.936293,-8.328811,7.369646],[6.561147,5.886369,6.361133,-0.252804,-9.898683,-2.726506],[8.110263,1.656619,5.214722,-5.259351,0.963967,-6.005307],[9.008580,-7.102185,9.737619,-0.448075,4.899404,0.886533],[-8.203409,9.081904,-4.477388,-7.608020,9.373233,3.091661],[-9.921979,-3.046419,8.598482,-0.895072,-0.455468,-3.361382],[-2.270406,-4.910448,3.963694,-1.436553,-5.969765,-3.258450],[4.465019,-1.858739,-0.428144,0.021469,0.113767,0.240749],[-1.369343,8.940520,-0.662249,-8.980537,0.733311,-4.123459],[-4.540871,-1.076261,-8.855223,0.335462,-5.799496,-2.155600]],[[-6.267511,2.298465,-1.001571,-5.094486,-3.932577,3.167331],[4.430012,-1.647051,-4.571116,9.371221,6.618724,8.073160],[-4.037715,-8.853514,-1.828901,-7.412017,-0.909704,5.190521],[-7.118680,-4.596752,1.055808,4.696187,2.851978,8.047523],[-6.649222,9.683894,0.589473,4.639595,6.360811,-3.517472],[3.611542,-5.256003,-0.605711,-6.356475,8.382898,5.141275],[-9.340047,-9.839972,-8.473625,2.511784,-8.703768,3.151638],[-6.735516,-7.613763,-5.656296,6.540411,5.028322,4.672928],[2.126626,-9.481509,-1.159054,-6.229073,-6.112447,4.693851],[-0.576731,1.979280,2.576926,-9.169229,0.503413,2.197647],[6.571234,7.465350,3.227202,-5.500822,-2.306723,1.450185]],[[6.477301,7.961942,-2.967265,-8.166764,-3.991023,-6.376270],[-0.557772,7.593748,9.666245,1.381947,9.558541,3.791527],[-3.350868,-1.300467,-1.148801,1.501370,1.368977,0.025536],[5.757941,-6.448702,4.899036,2.119131,-5.390949,4.075757],[-3.372983,-5.037228,-4.596173,-3.224109,-5.241527,8.170332],[-6.327753,-1.713316,-7.889535,-6.260692,1.340985,5.395265],[9.566840,4.289539,5.648459,0.532783,6.152265,2.791634],[0.374638,-9.094163,-7.628572,-3.334328,6.306250,-4.427876],[-9.215182,7.258789,-4.603901,-0.542591,-3.943783,1.955938],[-5.111918,3.705138,-5.859748,7.849711,-8.036138,0.741707],[-3.345419,2.796889,-9.126842,5.241004,-8.718608,6.877938]],[[4.955979,-3.604216,5.384252,4.059577,-5.361793,0.210258],[2.376785,-1.407295,3.189857,2.443134,4.613538,9.312679],[-2.534808,-6.488620,5.235249,7.349250,0.543061,-3.272428],[5.190848,7.280309,-4.930601,3.132970,8.621026,4.565051],[2.256082,-7.123351,-7.088268,4.774691,-9.162407,8.748929],[5.490990,-4.076950,4.970552,6.855250,-2.223037,-6.734786],[5.384911,6.604945,-1.075992,0.759411,4.092450,-0.303424],[1.520551,-0.324954,-9.979714,-7.928129,3.681705,-8.697419],[6.292471,9.034687,-4.717425,4.231894,8.820373,0.792329],[4.226444,-4.723489,-6.286285,-3.319044,-9.165095,-0.738759],[3.711980,-3.232488,1.994382,3.114312,-5.005909,0.533560]],[[-2.986411,-5.216787,1.274414,0.384004,4.386549,9.661645],[6.750328,2.457996,-7.314108,-2.077579,-3.076997,-4.880286],[7.806956,2.236325,6.599874,5.990405,0.429384,2.347477],[5.448599,6.442333,-4.825336,-3.635711,0.361743,1.575291],[0.494357,8.893911,-4.331271,-5.816426,4.562249,2.986004],[-4.824305,-6.367675,8.388138,-4.948981,0.980004,0.064466],[-6.181655,7.450202,4.476299,6.205362,-7.998000,7.878079],[8.211057,-5.749242,2.867052,2.929008,-5.873845,1.348253],[7.577443,-8.495309,7.960343,8.481910,-0.079592,1.243892],[6.155512,-9.754406,8.790368,5.873209,1.595269,2.252770],[9.638161,-1.253612,1.450495,0.006419,0.180471,7.232501]],[[1.644744,-9.296721,6.865606,-7.779234,7.625031,-8.188020],[-6.038620,7.622558,7.794743,-2.153613,3.427145,2.302927],[-2.112917,2.443180,-4.907975,-1.814264,0.616126,5.821979],[8.636720,-9.889801,2.322299,6.397970,-5.748008,1.013295],[-4.303947,-4.416421,9.121201,-0.399122,9.351789,0.058534],[-0.839168,6.228018,8.837260,-3.095580,-8.990477,0.787439],[6.054982,1.086807,0.359780,2.613501,-5.717169,6.776526],[8.652691,4.108202,0.240305,0.778132,7.373539,-9.763451],[6.506341,8.107390,0.909166,6.634469,0.942161,-7.061545],[-7.567102,-4.079893,8.416828,-6.372450,9.128494,8.933916],[-7.307889,-2.633274,-9.348332,-5.585267,9.602759,2.100738]],[[-0.845037,0.248925,-3.818032,-3.158836,-8.302816,1.317717],[-8.207566,-5.386165,-2.682613,4.654874,-9.656427,-9.003746],[-8.089465,9.020488,-9.795656,-6.294689,5.399691,-4.063489],[2.783883,-6.946560,-2.049184,8.487822,1.593058,-1.603061],[-2.869457,4.784229,-2.331087,2.228611,-2.431213,-4.089926],[0.262692,-7.014058,5.666293,-0.317491,2.333965,-8.767325],[6.067765,6.654449,0.309182,-7.370463,4.836432,1.538633],[7.987268,-6.859497,7.190529,-8.768005,1.119739,0.258470],[7.555989,0.409154,0.510081,-0.184061,2.889260,6.271061],[-7.544106,-6.101187,0.672809,7.822344,-7.773974,2.960918],[-7.938759,3.838290,-8.740459,-0.564452,-2.715454,-4.633930]],[[7.543639,-7.629524,-5.668765,-9.295169,0.347826,-4.851650],[-9.008872,1.565947,9.301469,-2.712974,-2.752082,9.017437],[-0.465523,1.403162,5.393424,-7.614218,-3.556526,-0.361953],[1.213665,4.472812,0.564827,4.441170,7.667458,4.114468],[6.082034,3.469354,2.885522,-8.646185,-3.123035,-8.099985],[-0.652869,9.433469,-5.861750,7.622779,-6.255422,-2.278526],[9.501968,5.552906,4.002452,6.902412,-4.610208,-9.920822],[-9.178063,4.193486,-3.964220,-2.183965,3.332873,-1.048000],[5.061974,8.906932,-7.236117,-2.034997,0.925270,8.862053],[-4.994842,-9.515756,7.719981,-3.588924,3.926475,-1.484844],[6.228349,-7.113882,-2.676469,-5.202308,8.221766,2.590824]],[[-1.487895,3.479982,2.810833,-3.927706,1.660361,-0.277260],[9.602273,-5.211027,7.302644,7.429419,-6.852614,2.638527],[-7.003112,-7.256096,5.040856,9.728024,8.346460,5.298916],[3.941145,-3.828086,-3.005742,4.855673,8.822956,7.862097],[4.491953,-7.809336,-5.084193,-7.691888,2.565733,-0.602288],[6.595049,-5.879209,-0.921996,8.259988,8.414343,8.777156],[1.356674,-6.062220,-8.793234,3.983826,-9.886889,5.850127],[9.527708,5.298142,7.165535,6.437630,7.107043,7.716803],[5.294965,-4.220537,0.672788,2.343515,-1.967956,3.835920],[8.083870,-4.781092,-0.407008,-4.363070,-9.085222,2.387125],[-3.084344,-9.927543,-8.897296,2.087274,-3.684916,0.982855]],[[3.549424,-0.251106,-4.561794,4.124241,-3.273379,4.976715],[8.719020,-6.131414,8.445819,0.819529,-8.186561,8.380875],[-5.672102,7.579209,-9.813509,-4.072546,1.679845,-1.021289],[-1.841611,5.651332,3.867902,-8.158317,-5.219396,-0.242329],[-5.220575,8.416337,2.727824,5.100791,-5.053916,5.453586],[5.493545,-0.897912,-4.184446,-9.968751,-8.869123,-3.687680],[-6.394099,2.236937,0.967286,-5.558907,4.290900,7.567654],[9.357599,-5.229748,3.803345,-8.060203,3.282962,7.027260],[4.027114,9.229748,1.488764,-3.191290,-0.021368,-6.344833],[2.557935,0.102880,2.687600,-9.959912,7.324316,-0.140055],[-3.166100,-4.761877,3.445312,3.857739,3.422422,2.394577]],[[-4.168608,7.890057,8.973540,8.927761,-1.119660,5.743513],[2.890927,-9.265825,8.433513,3.016057,-4.155395,8.954606],[9.358552,-0.124218,3.313732,6.583467,5.996462,-6.542739],[-9.007566,-0.351437,-7.323470,-1.132786,4.597952,3.145801],[1.016228,-2.754534,-5.247893,-7.748102,-1.947603,-3.874258],[-9.324118,-1.749546,-2.590858,-7.816084,-0.640780,-3.537046],[-1.134745,-5.882431,5.942335,-3.720555,-2.495009,-4.875522],[0.696825,-5.022623,-8.426037,1.224402,-2.774621,-0.945057],[5.302698,6.923703,3.503147,-4.340038,5.587445,9.707895],[-6.533802,3.078358,-8.457936,9.218278,-6.244536,-5.976269],[6.791809,1.820243,5.871069,-6.882855,-0.016157,-1.108927]],[[1.709160,-1.472068,7.981487,-2.655823,4.910727,-9.664625],[7.442359,2.486012,3.095485,-6.586090,-8.970680,-6.678931],[-5.844995,-8.975986,-5.305605,-7.342284,5.303906,-2.269777],[3.747316,4.688207,5.131451,-6.576524,6.428186,-7.403861],[1.822408,1.017992,8.333142,5.149027,1.345822,-5.969583],[-2.300012,4.673521,2.839612,-2.897399,-1.089179,-1.373843],[5.461293,9.078781,-8.078690,4.898711,-8.647673,6.246477],[-5.154415,9.931401,-3.773873,-7.410451,4.129016,6.034284],[6.624001,1.254622,9.147586,3.888925,-5.986968,2.648182],[2.124863,2.637564,-1.142138,7.951741,4.935178,-1.616452],[-0.387408,0.991273,-8.166901,5.580732,-5.843479,-0.131842]]], dtype = "float32")#candidate|61|(13, 11, 6)|const|float32
bop_62 = relay.minimum(uop_29.astype('float32'), relay.reshape(const_61.astype('float32'), relay.shape_of(uop_29))) # shape=(13, 11, 6)
func_10_call = mod.get_global_var('func_10')
func_14_call = mutated_mod.get_global_var('func_14')
const_66 = relay.const(6.376810, dtype = "float64")#candidate|66|()|const|float64
const_67 = relay.const([1.555688,-6.161274,4.491655,-7.309962,-6.995948,8.929175,-7.094562,-4.817379,7.970006,1.281463], dtype = "float64")#candidate|67|(10,)|const|float64
call_65 = func_10_call(relay.reshape(const_66.astype('float64'), []), relay.reshape(const_67.astype('float64'), [10, 1]), )
call_68 = func_10_call(relay.reshape(const_66.astype('float64'), []), relay.reshape(const_67.astype('float64'), [10, 1]), )
uop_69 = relay.cos(bop_36.astype('float32')) # shape=(13, 11, 6)
bop_71 = relay.greater_equal(uop_69.astype('bool'), relay.reshape(bop_42.astype('bool'), relay.shape_of(uop_69))) # shape=(13, 11, 6)
var_74 = relay.var("var_74", dtype = "bool", shape = (13, 11, 6))#candidate|74|(13, 11, 6)|var|bool
bop_75 = relay.right_shift(bop_48.astype('int8'), relay.reshape(var_74.astype('int8'), relay.shape_of(bop_48))) # shape=(13, 11, 6)
bop_78 = relay.bitwise_or(uop_69.astype('uint16'), relay.reshape(bop_56.astype('uint16'), relay.shape_of(uop_69))) # shape=(13, 11, 6)
var_81 = relay.var("var_81", dtype = "float32", shape = (13, 11, 6))#candidate|81|(13, 11, 6)|var|float32
bop_82 = relay.right_shift(uop_69.astype('int32'), relay.reshape(var_81.astype('int32'), relay.shape_of(uop_69))) # shape=(13, 11, 6)
bop_85 = relay.bitwise_and(uop_31.astype('int8'), relay.reshape(bop_82.astype('int8'), relay.shape_of(uop_31))) # shape=(13, 11, 6)
var_88 = relay.var("var_88", dtype = "uint16", shape = (13, 11, 6))#candidate|88|(13, 11, 6)|var|uint16
bop_89 = relay.logical_or(bop_78.astype('bool'), relay.reshape(var_88.astype('bool'), relay.shape_of(bop_78))) # shape=(13, 11, 6)
var_92 = relay.var("var_92", dtype = "bool", shape = (13, 11, 6))#candidate|92|(13, 11, 6)|var|bool
bop_93 = relay.logical_xor(bop_89.astype('int64'), relay.reshape(var_92.astype('int64'), relay.shape_of(bop_89))) # shape=(13, 11, 6)
bop_96 = relay.subtract(bop_45.astype('int64'), relay.reshape(uop_51.astype('int64'), relay.shape_of(bop_45))) # shape=(13, 11, 6)
uop_99 = relay.log(bop_96.astype('float64')) # shape=(13, 11, 6)
output = relay.Tuple([bop_26,bop_53,uop_59,bop_62,call_65,const_66,const_67,bop_71,bop_75,bop_85,bop_93,uop_99,])
output2 = relay.Tuple([bop_26,bop_53,uop_59,bop_62,call_68,const_66,const_67,bop_71,bop_75,bop_85,bop_93,uop_99,])
func_101 = relay.Function([var_17,var_21,var_25,var_74,var_81,var_88,var_92,], output)
mod['func_101'] = func_101
mod = relay.transform.InferType()(mod)
mutated_mod['func_101'] = func_101
mutated_mod = relay.transform.InferType()(mutated_mod)
func_101_call = mutated_mod.get_global_var('func_101')
var_103 = relay.var("var_103", dtype = "float64", shape = (13, 11, 6))#candidate|103|(13, 11, 6)|var|float64
var_104 = relay.var("var_104", dtype = "float64", shape = (13, 11, 6))#candidate|104|(13, 11, 6)|var|float64
var_105 = relay.var("var_105", dtype = "bool", shape = (13, 11, 6))#candidate|105|(13, 11, 6)|var|bool
var_106 = relay.var("var_106", dtype = "bool", shape = (13, 11, 6))#candidate|106|(13, 11, 6)|var|bool
var_107 = relay.var("var_107", dtype = "float32", shape = (13, 11, 6))#candidate|107|(13, 11, 6)|var|float32
var_108 = relay.var("var_108", dtype = "uint16", shape = (13, 11, 6))#candidate|108|(13, 11, 6)|var|uint16
var_109 = relay.var("var_109", dtype = "bool", shape = (13, 11, 6))#candidate|109|(13, 11, 6)|var|bool
call_102 = func_101_call(var_103,var_104,var_105,var_106,var_107,var_108,var_109,)
output = call_102
func_110 = relay.Function([var_103,var_104,var_105,var_106,var_107,var_108,var_109,], output)
mutated_mod['func_110'] = func_110
mutated_mod = relay.transform.InferType()(mutated_mod)
var_112 = relay.var("var_112", dtype = "float32", shape = (5,))#candidate|112|(5,)|var|float32
uop_113 = relay.acosh(var_112.astype('float32')) # shape=(5,)
bop_115 = relay.floor_mod(uop_113.astype('float64'), relay.reshape(var_112.astype('float64'), relay.shape_of(uop_113))) # shape=(5,)
uop_118 = relay.atan(var_112.astype('float32')) # shape=(5,)
var_120 = relay.var("var_120", dtype = "float64", shape = (5,))#candidate|120|(5,)|var|float64
bop_121 = relay.divide(bop_115.astype('float32'), relay.reshape(var_120.astype('float32'), relay.shape_of(bop_115))) # shape=(5,)
uop_124 = relay.asin(bop_115.astype('float32')) # shape=(5,)
bop_126 = relay.bitwise_xor(uop_124.astype('int16'), relay.reshape(bop_115.astype('int16'), relay.shape_of(uop_124))) # shape=(5,)
var_129 = relay.var("var_129", dtype = "int16", shape = (5,))#candidate|129|(5,)|var|int16
bop_130 = relay.right_shift(bop_126.astype('int32'), relay.reshape(var_129.astype('int32'), relay.shape_of(bop_126))) # shape=(5,)
var_133 = relay.var("var_133", dtype = "float32", shape = (5,))#candidate|133|(5,)|var|float32
bop_134 = relay.divide(uop_118.astype('float32'), relay.reshape(var_133.astype('float32'), relay.shape_of(uop_118))) # shape=(5,)
var_137 = relay.var("var_137", dtype = "int16", shape = (5,))#candidate|137|(5,)|var|int16
bop_138 = relay.equal(bop_126.astype('bool'), relay.reshape(var_137.astype('bool'), relay.shape_of(bop_126))) # shape=(5,)
var_141 = relay.var("var_141", dtype = "float32", shape = (5,))#candidate|141|(5,)|var|float32
bop_142 = relay.greater_equal(uop_124.astype('bool'), relay.reshape(var_141.astype('bool'), relay.shape_of(uop_124))) # shape=(5,)
uop_145 = relay.sqrt(uop_124.astype('float64')) # shape=(5,)
bop_147 = relay.floor_divide(uop_145.astype('float32'), relay.reshape(uop_118.astype('float32'), relay.shape_of(uop_145))) # shape=(5,)
uop_150 = relay.cosh(uop_145.astype('float64')) # shape=(5,)
const_152 = relay.const([6.345298,-3.123314,4.671322,-7.510415,-2.025887], dtype = "float64")#candidate|152|(5,)|const|float64
bop_153 = relay.bitwise_and(uop_150.astype('uint32'), relay.reshape(const_152.astype('uint32'), relay.shape_of(uop_150))) # shape=(5,)
uop_156 = relay.sigmoid(uop_145.astype('float64')) # shape=(5,)
func_10_call = mod.get_global_var('func_10')
func_14_call = mutated_mod.get_global_var('func_14')
var_159 = relay.var("var_159", dtype = "float64", shape = ())#candidate|159|()|var|float64
var_160 = relay.var("var_160", dtype = "float64", shape = (10, 1))#candidate|160|(10, 1)|var|float64
call_158 = func_10_call(relay.reshape(var_159.astype('float64'), []), relay.reshape(var_160.astype('float64'), [10, 1]), )
call_161 = func_10_call(relay.reshape(var_159.astype('float64'), []), relay.reshape(var_160.astype('float64'), [10, 1]), )
var_162 = relay.var("var_162", dtype = "float64", shape = (5,))#candidate|162|(5,)|var|float64
bop_163 = relay.subtract(uop_156.astype('uint32'), relay.reshape(var_162.astype('uint32'), relay.shape_of(uop_156))) # shape=(5,)
output = relay.Tuple([bop_121,bop_130,bop_134,bop_138,bop_142,bop_147,bop_153,call_158,var_159,var_160,bop_163,])
output2 = relay.Tuple([bop_121,bop_130,bop_134,bop_138,bop_142,bop_147,bop_153,call_161,var_159,var_160,bop_163,])
func_166 = relay.Function([var_112,var_120,var_129,var_133,var_137,var_141,var_159,var_160,var_162,], output)
mod['func_166'] = func_166
mod = relay.transform.InferType()(mod)
var_167 = relay.var("var_167", dtype = "float32", shape = (5,))#candidate|167|(5,)|var|float32
var_168 = relay.var("var_168", dtype = "float64", shape = (5,))#candidate|168|(5,)|var|float64
var_169 = relay.var("var_169", dtype = "int16", shape = (5,))#candidate|169|(5,)|var|int16
var_170 = relay.var("var_170", dtype = "float32", shape = (5,))#candidate|170|(5,)|var|float32
var_171 = relay.var("var_171", dtype = "int16", shape = (5,))#candidate|171|(5,)|var|int16
var_172 = relay.var("var_172", dtype = "float32", shape = (5,))#candidate|172|(5,)|var|float32
var_173 = relay.var("var_173", dtype = "float64", shape = ())#candidate|173|()|var|float64
var_174 = relay.var("var_174", dtype = "float64", shape = (10, 1))#candidate|174|(10, 1)|var|float64
var_175 = relay.var("var_175", dtype = "float64", shape = (5,))#candidate|175|(5,)|var|float64
output = func_166(var_167,var_168,var_169,var_170,var_171,var_172,var_173,var_174,var_175,)
func_176 = relay.Function([var_167,var_168,var_169,var_170,var_171,var_172,var_173,var_174,var_175,], output)
mutated_mod['func_176'] = func_176
mutated_mod = relay.transform.InferType()(mutated_mod)
var_178 = relay.var("var_178", dtype = "float64", shape = (2, 5))#candidate|178|(2, 5)|var|float64
uop_179 = relay.atanh(var_178.astype('float64')) # shape=(2, 5)
bop_181 = relay.left_shift(uop_179.astype('int64'), relay.reshape(var_178.astype('int64'), relay.shape_of(uop_179))) # shape=(2, 5)
bop_184 = relay.subtract(var_178.astype('int8'), relay.reshape(bop_181.astype('int8'), relay.shape_of(var_178))) # shape=(2, 5)
bop_187 = relay.equal(var_178.astype('bool'), relay.reshape(bop_181.astype('bool'), relay.shape_of(var_178))) # shape=(2, 5)
uop_190 = relay.asinh(var_178.astype('float32')) # shape=(2, 5)
bop_192 = relay.power(bop_181.astype('float32'), relay.reshape(uop_179.astype('float32'), relay.shape_of(bop_181))) # shape=(2, 5)
uop_195 = relay.sqrt(uop_179.astype('float64')) # shape=(2, 5)
const_197 = relay.const([[-4.377279,2.699235,0.863101,-1.849398,2.378964],[5.947093,0.399812,-2.431601,-7.949162,-7.928596]], dtype = "float64")#candidate|197|(2, 5)|const|float64
bop_198 = relay.multiply(uop_195.astype('uint16'), relay.reshape(const_197.astype('uint16'), relay.shape_of(uop_195))) # shape=(2, 5)
bop_201 = relay.multiply(bop_181.astype('uint8'), relay.reshape(const_197.astype('uint8'), relay.shape_of(bop_181))) # shape=(2, 5)
bop_204 = relay.divide(uop_195.astype('float32'), relay.reshape(uop_190.astype('float32'), relay.shape_of(uop_195))) # shape=(2, 5)
bop_207 = relay.not_equal(var_178.astype('bool'), relay.reshape(uop_190.astype('bool'), relay.shape_of(var_178))) # shape=(2, 5)
bop_210 = relay.floor_mod(bop_181.astype('float32'), relay.reshape(uop_179.astype('float32'), relay.shape_of(bop_181))) # shape=(2, 5)
uop_213 = relay.erf(var_178.astype('float32')) # shape=(2, 5)
bop_215 = relay.divide(bop_192.astype('float32'), relay.reshape(uop_179.astype('float32'), relay.shape_of(bop_192))) # shape=(2, 5)
uop_218 = relay.asin(uop_190.astype('float32')) # shape=(2, 5)
func_166_call = mod.get_global_var('func_166')
func_176_call = mutated_mod.get_global_var('func_176')
var_221 = relay.var("var_221", dtype = "float32", shape = (5, 1))#candidate|221|(5, 1)|var|float32
var_222 = relay.var("var_222", dtype = "float64", shape = ())#candidate|222|()|var|float64
call_220 = relay.TupleGetItem(func_166_call(relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_221.astype('float64'), [5,]), relay.reshape(var_221.astype('int16'), [5,]), relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_221.astype('int16'), [5,]), relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_222.astype('float64'), []), relay.reshape(bop_187.astype('float64'), [10, 1]), relay.reshape(var_221.astype('float64'), [5,]), ), 3)
call_223 = relay.TupleGetItem(func_176_call(relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_221.astype('float64'), [5,]), relay.reshape(var_221.astype('int16'), [5,]), relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_221.astype('int16'), [5,]), relay.reshape(var_221.astype('float32'), [5,]), relay.reshape(var_222.astype('float64'), []), relay.reshape(bop_187.astype('float64'), [10, 1]), relay.reshape(var_221.astype('float64'), [5,]), ), 3)
bop_224 = relay.right_shift(bop_204.astype('uint64'), var_222.astype('uint64')) # shape=(2, 5)
uop_227 = relay.acosh(uop_190.astype('float32')) # shape=(2, 5)
bop_229 = relay.bitwise_or(bop_192.astype('int32'), relay.reshape(uop_179.astype('int32'), relay.shape_of(bop_192))) # shape=(2, 5)
output = relay.Tuple([bop_184,bop_187,bop_198,bop_201,bop_207,bop_210,uop_213,bop_215,uop_218,call_220,var_221,bop_224,uop_227,bop_229,])
output2 = relay.Tuple([bop_184,bop_187,bop_198,bop_201,bop_207,bop_210,uop_213,bop_215,uop_218,call_223,var_221,bop_224,uop_227,bop_229,])
func_232 = relay.Function([var_178,var_221,var_222,], output)
mod['func_232'] = func_232
mod = relay.transform.InferType()(mod)
mutated_mod['func_232'] = func_232
mutated_mod = relay.transform.InferType()(mutated_mod)
func_232_call = mutated_mod.get_global_var('func_232')
var_234 = relay.var("var_234", dtype = "float64", shape = (2, 5))#candidate|234|(2, 5)|var|float64
var_235 = relay.var("var_235", dtype = "float32", shape = (5, 1))#candidate|235|(5, 1)|var|float32
var_236 = relay.var("var_236", dtype = "float64", shape = ())#candidate|236|()|var|float64
call_233 = func_232_call(var_234,var_235,var_236,)
output = call_233
func_237 = relay.Function([var_234,var_235,var_236,], output)
mutated_mod['func_237'] = func_237
mutated_mod = relay.transform.InferType()(mutated_mod)
var_239 = relay.var("var_239", dtype = "float32", shape = ())#candidate|239|()|var|float32
uop_240 = relay.exp(var_239.astype('float32')) # shape=()
uop_242 = relay.acos(var_239.astype('float32')) # shape=()
bop_244 = relay.not_equal(var_239.astype('bool'), uop_242.astype('bool')) # shape=()
var_247 = relay.var("var_247", dtype = "float32", shape = (5, 6))#candidate|247|(5, 6)|var|float32
bop_248 = relay.maximum(uop_242.astype('int64'), var_247.astype('int64')) # shape=(5, 6)
bop_251 = relay.maximum(bop_244.astype('uint8'), uop_240.astype('uint8')) # shape=()
uop_254 = relay.sigmoid(bop_244.astype('float32')) # shape=()
bop_256 = relay.greater_equal(uop_254.astype('bool'), uop_242.astype('bool')) # shape=()
func_10_call = mod.get_global_var('func_10')
func_14_call = mutated_mod.get_global_var('func_14')
var_260 = relay.var("var_260", dtype = "float64", shape = (10,))#candidate|260|(10,)|var|float64
call_259 = func_10_call(relay.reshape(var_239.astype('float64'), []), relay.reshape(var_260.astype('float64'), [10, 1]), )
call_261 = func_10_call(relay.reshape(var_239.astype('float64'), []), relay.reshape(var_260.astype('float64'), [10, 1]), )
bop_262 = relay.floor_mod(bop_256.astype('float64'), var_247.astype('float64')) # shape=(5, 6)
uop_265 = relay.acosh(bop_244.astype('float32')) # shape=()
uop_267 = relay.log(bop_262.astype('float64')) # shape=(5, 6)
bop_269 = relay.logical_and(uop_265.astype('bool'), var_260.astype('bool')) # shape=(10,)
uop_272 = relay.log10(uop_267.astype('float64')) # shape=(5, 6)
bop_274 = relay.left_shift(bop_256.astype('uint64'), uop_265.astype('uint64')) # shape=()
uop_277 = relay.sigmoid(uop_254.astype('float32')) # shape=()
const_279 = relay.const([[-3.655790,9.991669,-7.323766,4.384995,7.404684,8.954527],[-4.771310,-1.816967,-6.354439,3.562603,-2.649248,-3.300823],[-7.244065,-4.201528,7.947140,-4.703228,-4.724230,4.761198],[-8.932124,5.032847,-7.154396,9.726221,-1.735885,8.072079],[4.383652,-3.209514,-1.899896,6.461664,-5.452493,-8.079127]], dtype = "float64")#candidate|279|(5, 6)|const|float64
bop_280 = relay.less_equal(uop_272.astype('bool'), relay.reshape(const_279.astype('bool'), relay.shape_of(uop_272))) # shape=(5, 6)
bop_283 = relay.minimum(uop_272.astype('uint8'), relay.reshape(const_279.astype('uint8'), relay.shape_of(uop_272))) # shape=(5, 6)
var_286 = relay.var("var_286", dtype = "float32", shape = (1, 6, 9))#candidate|286|(1, 6, 9)|var|float32
bop_287 = relay.bitwise_xor(uop_277.astype('uint64'), var_286.astype('uint64')) # shape=(1, 6, 9)
bop_290 = relay.right_shift(uop_277.astype('int8'), uop_242.astype('int8')) # shape=()
output = relay.Tuple([bop_248,bop_251,call_259,bop_269,bop_274,bop_280,bop_283,bop_287,bop_290,])
output2 = relay.Tuple([bop_248,bop_251,call_261,bop_269,bop_274,bop_280,bop_283,bop_287,bop_290,])
func_293 = relay.Function([var_239,var_247,var_260,var_286,], output)
mod['func_293'] = func_293
mod = relay.transform.InferType()(mod)
mutated_mod['func_293'] = func_293
mutated_mod = relay.transform.InferType()(mutated_mod)
func_293_call = mutated_mod.get_global_var('func_293')
var_295 = relay.var("var_295", dtype = "float32", shape = ())#candidate|295|()|var|float32
var_296 = relay.var("var_296", dtype = "float32", shape = (5, 6))#candidate|296|(5, 6)|var|float32
var_297 = relay.var("var_297", dtype = "float64", shape = (10,))#candidate|297|(10,)|var|float64
var_298 = relay.var("var_298", dtype = "float32", shape = (1, 6, 9))#candidate|298|(1, 6, 9)|var|float32
call_294 = func_293_call(var_295,var_296,var_297,var_298,)
output = call_294
func_299 = relay.Function([var_295,var_296,var_297,var_298,], output)
mutated_mod['func_299'] = func_299
mutated_mod = relay.transform.InferType()(mutated_mod)
var_301 = relay.var("var_301", dtype = "bool", shape = (3, 15))#candidate|301|(3, 15)|var|bool
const_302 = relay.const([[False,False,False,False,True,True,False,False,False,True,True,False,True,True,False],[False,True,True,True,False,True,True,False,False,True,False,False,True,False,False],[True,False,False,True,True,False,True,True,False,True,True,True,True,False,False]], dtype = "bool")#candidate|302|(3, 15)|const|bool
bop_303 = relay.logical_or(var_301.astype('bool'), relay.reshape(const_302.astype('bool'), relay.shape_of(var_301))) # shape=(3, 15)
bop_306 = relay.less(var_301.astype('bool'), relay.reshape(const_302.astype('bool'), relay.shape_of(var_301))) # shape=(3, 15)
func_166_call = mod.get_global_var('func_166')
func_176_call = mutated_mod.get_global_var('func_176')
const_310 = relay.const([5.351840,1.457364,6.252844,-8.009731,-8.417947], dtype = "float32")#candidate|310|(5,)|const|float32
const_311 = relay.const(-8.610603, dtype = "float64")#candidate|311|()|const|float64
const_312 = relay.const([[-5.073506],[-4.975779],[0.545009],[5.563187],[4.384039],[-8.759317],[-6.487280],[-3.175711],[8.586217],[-2.289574]], dtype = "float64")#candidate|312|(10, 1)|const|float64
call_309 = relay.TupleGetItem(func_166_call(relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_310.astype('float64'), [5,]), relay.reshape(const_310.astype('int16'), [5,]), relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_310.astype('int16'), [5,]), relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_311.astype('float64'), []), relay.reshape(const_312.astype('float64'), [10, 1]), relay.reshape(const_310.astype('float64'), [5,]), ), 4)
call_313 = relay.TupleGetItem(func_176_call(relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_310.astype('float64'), [5,]), relay.reshape(const_310.astype('int16'), [5,]), relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_310.astype('int16'), [5,]), relay.reshape(const_310.astype('float32'), [5,]), relay.reshape(const_311.astype('float64'), []), relay.reshape(const_312.astype('float64'), [10, 1]), relay.reshape(const_310.astype('float64'), [5,]), ), 4)
const_314 = relay.const([[-4.321221,-6.321773,-4.986635,-3.246095,3.984993,1.970957,-4.998402,3.211207,4.772012,8.770782,9.766244,7.445799,6.374295],[-2.620740,-6.038029,0.783248,-6.886248,-8.631547,5.887286,-7.553934,-9.798328,6.833374,7.353699,1.754321,-2.548328,-0.749356],[2.786829,-7.734845,3.709944,9.970440,-2.231030,-1.102238,-0.109246,-9.732912,2.667686,-5.397930,-0.777568,-6.014721,-7.873582],[7.149772,-6.999366,-1.394940,3.164084,-0.117028,-5.137935,-4.619722,3.447530,8.870926,2.145979,9.621906,-9.255967,-8.374035],[7.229436,-0.502443,8.412832,-3.733300,8.791684,8.779631,-9.805227,8.079401,9.025779,6.786849,-2.911253,-6.110331,-9.637225],[2.290258,1.779854,-7.859169,-4.491799,1.470158,-1.612815,-2.820222,-7.495239,-7.371704,6.981106,4.622404,4.118354,2.152268],[-3.871691,8.520553,6.883469,9.878320,7.174130,0.746129,9.147110,9.518196,4.473547,3.979577,9.027439,-9.979553,-0.451697],[0.040332,-0.978668,9.845491,-3.196551,-3.897352,1.541950,-2.992263,8.322118,1.180663,-3.995018,-9.826862,-0.832731,-1.826523],[9.193819,-7.637077,-4.223639,6.358502,9.389703,2.381909,1.393275,5.409370,-1.995224,5.220041,-9.944427,-9.355422,0.576438],[-7.224757,-1.999015,6.792885,8.895662,8.448118,5.124336,-2.634668,-1.168773,-1.027832,0.244643,-8.876580,-5.795087,1.055837]], dtype = "float64")#candidate|314|(10, 13)|const|float64
bop_315 = relay.multiply(const_312.astype('uint16'), const_314.astype('uint16')) # shape=(10, 13)
var_318 = relay.var("var_318", dtype = "bool", shape = (3, 15))#candidate|318|(3, 15)|var|bool
bop_319 = relay.logical_or(var_301.astype('bool'), relay.reshape(var_318.astype('bool'), relay.shape_of(var_301))) # shape=(3, 15)
bop_322 = relay.bitwise_and(const_311.astype('uint16'), bop_306.astype('uint16')) # shape=(3, 15)
uop_325 = relay.exp(const_310.astype('float32')) # shape=(5,)
func_293_call = mod.get_global_var('func_293')
func_299_call = mutated_mod.get_global_var('func_299')
const_328 = relay.const([0.535424,-2.008581,4.936445,9.743957,3.198538,-2.240030,-4.223131,-5.535521,0.737786,0.388712,3.803350,8.800879,3.667329,-1.369452,-1.179046,3.235258,-6.682645,-9.222785,0.177680,-4.782387,8.613531,7.531296,-2.427924,-3.397044,0.314797,3.001067,2.724289,-0.161622,-4.841409,4.019154], dtype = "float32")#candidate|328|(30,)|const|float32
const_329 = relay.const([-4.203811,0.717112,5.832445,8.243786,-3.687072,-5.567811,-3.295384,-8.552058,6.221515,-4.458452,-8.470679,-4.287116,9.299238,-9.755968,8.991752,4.498849,3.092207,7.596108,0.249245,6.767390,-8.732786,-6.087758,7.041845,-9.380396,-6.234999,-2.919502,7.553960,5.513542,4.361228,4.144430,6.717453,7.427336,2.937274,5.025993,0.743763,9.555627,1.281972,-4.387831,-6.126087,5.835889,6.573002,-4.038657,-3.661162,-3.301781,5.899843,6.550684,-3.786220,5.436272,5.215485,-1.032860,2.103399,-4.267709,1.531892,-3.270002], dtype = "float32")#candidate|329|(54,)|const|float32
call_327 = relay.TupleGetItem(func_293_call(relay.reshape(const_311.astype('float32'), []), relay.reshape(const_328.astype('float32'), [5, 6]), relay.reshape(const_312.astype('float64'), [10,]), relay.reshape(const_329.astype('float32'), [1, 6, 9]), ), 7)
call_330 = relay.TupleGetItem(func_299_call(relay.reshape(const_311.astype('float32'), []), relay.reshape(const_328.astype('float32'), [5, 6]), relay.reshape(const_312.astype('float64'), [10,]), relay.reshape(const_329.astype('float32'), [1, 6, 9]), ), 7)
bop_331 = relay.logical_and(const_314.astype('bool'), relay.reshape(bop_315.astype('bool'), relay.shape_of(const_314))) # shape=(10, 13)
uop_334 = relay.asinh(bop_331.astype('float32')) # shape=(10, 13)
var_336 = relay.var("var_336", dtype = "float32", shape = (10, 13))#candidate|336|(10, 13)|var|float32
bop_337 = relay.divide(uop_334.astype('float32'), relay.reshape(var_336.astype('float32'), relay.shape_of(uop_334))) # shape=(10, 13)
uop_340 = relay.log10(bop_337.astype('float32')) # shape=(10, 13)
const_342 = relay.const([[9.081535,7.013893,1.602437,1.325531,-8.324618,-8.614465,-1.639564,8.197301,2.498360,-2.086835,-3.888994,-6.093728,-1.747334],[3.194141,-4.740979,-2.198043,-1.987177,6.308202,-7.611339,-6.647044,3.829032,6.433848,-9.563358,0.658506,4.144006,-6.034218],[-3.753056,-2.004399,-7.295307,-4.329553,2.583974,0.680905,-7.791192,1.228961,5.570683,1.562621,-2.486715,-6.406773,5.087363],[-4.868747,0.979728,-4.583186,-3.594613,-8.324221,6.926414,4.681724,4.543663,-6.725037,3.198848,-2.246282,7.324477,0.657891],[6.462351,1.355472,9.070393,2.152973,-8.217656,3.652831,-8.070011,3.307057,-7.831600,-2.791418,6.473379,4.338684,2.815404],[0.811126,-0.879556,-3.407120,4.997732,-9.681041,-4.988771,7.796392,-5.294061,8.048168,2.031093,-6.424566,8.500044,6.783390],[9.830550,6.356117,1.991395,-3.689825,1.093852,8.822819,-9.360702,0.248981,1.493637,4.011528,3.866709,-4.037899,5.640473],[-5.441972,-9.487673,-4.968614,0.315323,-5.805531,-1.212882,-5.166626,8.992464,-7.965386,1.935587,-7.467669,7.621302,-3.681805],[-3.984468,9.743152,2.418079,-2.162355,5.874130,-1.277325,-3.583319,7.080518,1.372207,-3.535263,1.754605,-5.176286,4.838849],[3.493111,-3.135416,7.230285,6.014990,-4.510533,3.297177,-2.251890,3.702503,1.018315,7.146603,4.593579,-1.866337,1.855307]], dtype = "float32")#candidate|342|(10, 13)|const|float32
bop_343 = relay.floor_divide(uop_340.astype('float32'), relay.reshape(const_342.astype('float32'), relay.shape_of(uop_340))) # shape=(10, 13)
var_346 = relay.var("var_346", dtype = "float32", shape = (10, 13))#candidate|346|(10, 13)|var|float32
bop_347 = relay.greater(bop_343.astype('bool'), relay.reshape(var_346.astype('bool'), relay.shape_of(bop_343))) # shape=(10, 13)
const_350 = relay.const([9.797641,-5.331911,-9.702291,2.783938,-8.392966], dtype = "float32")#candidate|350|(5,)|const|float32
bop_351 = relay.floor_divide(uop_325.astype('float64'), relay.reshape(const_350.astype('float64'), relay.shape_of(uop_325))) # shape=(5,)
var_354 = relay.var("var_354", dtype = "float32", shape = (10, 13))#candidate|354|(10, 13)|var|float32
bop_355 = relay.not_equal(bop_343.astype('bool'), relay.reshape(var_354.astype('bool'), relay.shape_of(bop_343))) # shape=(10, 13)
uop_358 = relay.erf(bop_355.astype('float64')) # shape=(10, 13)
bop_360 = relay.maximum(bop_343.astype('float32'), const_311.astype('float32')) # shape=(10, 13)
uop_363 = relay.asinh(uop_358.astype('float32')) # shape=(10, 13)
uop_365 = relay.atanh(uop_363.astype('float64')) # shape=(10, 13)
output = relay.Tuple([bop_303,call_309,bop_319,bop_322,call_327,const_328,const_329,bop_347,bop_351,bop_360,uop_365,])
output2 = relay.Tuple([bop_303,call_313,bop_319,bop_322,call_330,const_328,const_329,bop_347,bop_351,bop_360,uop_365,])
func_367 = relay.Function([var_301,var_318,var_336,var_346,var_354,], output)
mod['func_367'] = func_367
mod = relay.transform.InferType()(mod)
mutated_mod['func_367'] = func_367
mutated_mod = relay.transform.InferType()(mutated_mod)
func_367_call = mutated_mod.get_global_var('func_367')
var_369 = relay.var("var_369", dtype = "bool", shape = (3, 15))#candidate|369|(3, 15)|var|bool
var_370 = relay.var("var_370", dtype = "bool", shape = (3, 15))#candidate|370|(3, 15)|var|bool
var_371 = relay.var("var_371", dtype = "float32", shape = (10, 13))#candidate|371|(10, 13)|var|float32
var_372 = relay.var("var_372", dtype = "float32", shape = (10, 13))#candidate|372|(10, 13)|var|float32
var_373 = relay.var("var_373", dtype = "float32", shape = (10, 13))#candidate|373|(10, 13)|var|float32
call_368 = func_367_call(var_369,var_370,var_371,var_372,var_373,)
output = call_368
func_374 = relay.Function([var_369,var_370,var_371,var_372,var_373,], output)
mutated_mod['func_374'] = func_374
mutated_mod = relay.transform.InferType()(mutated_mod)
var_376 = relay.var("var_376", dtype = "int16", shape = (11, 2, 7))#candidate|376|(11, 2, 7)|var|int16
var_377 = relay.var("var_377", dtype = "int16", shape = (11, 2, 7))#candidate|377|(11, 2, 7)|var|int16
bop_378 = relay.add(var_376.astype('int16'), relay.reshape(var_377.astype('int16'), relay.shape_of(var_376))) # shape=(11, 2, 7)
output = relay.Tuple([bop_378,])
output2 = relay.Tuple([bop_378,])
F = relay.Function([var_376,var_377,], output)
mod['main'] = F
mod = relay.transform.InferType()(mod)
print('==========mod==========')
print(mod.astext(show_meta_data=False))
print('===================================')
F = relay.Function([var_376,var_377,], output2)
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
input_376= np.array([[[-2,1,6,8,4,-2,3],[9,10,-7,-5,-5,9,-3]],[[-6,7,8,8,1,5,-1],[-6,-4,10,8,10,4,-2]],[[-1,-3,-5,-7,-8,4,-2],[-5,-7,5,10,5,5,-3]],[[-3,1,9,-3,2,-7,6],[10,-4,8,9,-3,9,9]],[[7,4,-9,6,-8,-4,2],[3,-7,-5,-1,10,-6,-1]],[[3,10,3,10,10,-10,6],[9,7,5,8,-3,-2,-8]],[[-8,-5,2,10,1,7,-7],[-10,-3,10,-2,4,-4,-8]],[[1,-7,7,-9,7,-4,1],[-10,10,-1,10,5,-8,-1]],[[-5,1,2,-3,9,-8,3],[-7,-3,1,8,8,-5,7]],[[1,10,-6,10,-10,-9,-6],[5,8,4,-10,7,-8,3]],[[10,-9,-9,-7,-1,-3,1],[-9,-8,8,6,-7,6,-4]]], dtype='int16')
module1.set_input('var_376', input_376)
input_377= np.array([[[3,-10,-9,-5,9,3,2],[-10,-7,-2,3,9,4,8]],[[4,-6,1,-3,4,-1,2],[4,2,-9,-3,2,-2,5]],[[-10,6,7,4,-8,10,-5],[-7,-7,-2,-5,7,-1,3]],[[-1,1,-3,9,9,-2,10],[-10,-5,3,10,-10,-6,2]],[[-6,-3,5,-2,1,10,5],[-8,4,-2,-7,-10,10,-1]],[[-4,4,-9,-9,-2,-3,-10],[-4,7,-4,10,-9,-2,-1]],[[6,9,-6,-2,-1,-9,9],[8,7,8,2,4,-7,8]],[[-9,-2,-2,6,5,4,8],[10,5,9,-8,5,-7,3]],[[-1,5,-1,-6,-7,2,2],[-7,9,-6,-3,-8,1,4]],[[-2,4,-7,3,6,-10,4],[5,8,-6,9,-1,8,7]],[[3,-9,-6,10,2,3,7],[-2,-5,-8,-9,10,-8,-10]]], dtype='int16')
module1.set_input('var_377', input_377)
module1.set_input(**params)
module1.run()
res2 = intrp2.evaluate()(input_376, input_377, )
res3 = intrp3.evaluate()(input_376, input_377, )
res4 = intrp4.evaluate()(input_376, input_377, )
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
module5.set_input('var_376', input_376)
module5.set_input('var_377', input_377)
module5.set_input(**params)
module5.run()
res6 = intrp6.evaluate()(input_376, input_377, )
res7 = intrp7.evaluate()(input_376, input_377, )
res8 = intrp8.evaluate()(input_376, input_377, )
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
module9.set_input('var_376', input_376)
module9.set_input('var_377', input_377)
module9.set_input(**params)
module9.run()
res10 = intrp10.evaluate()(input_376, input_377, )
res11 = intrp11.evaluate()(input_376, input_377, )
res12 = intrp12.evaluate()(input_376, input_377, )
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
module13.set_input('var_376', input_376)
module13.set_input('var_377', input_377)
module13.set_input(**params)
module13.run()
res14 = intrp14.evaluate()(input_376, input_377, )
res15 = intrp15.evaluate()(input_376, input_377, )
res16 = intrp16.evaluate()(input_376, input_377, )
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
module17.set_input('var_376', input_376)
module17.set_input('var_377', input_377)
module17.set_input(**params)
module17.run()
res18 = intrp18.evaluate()(input_376, input_377, )
res19 = intrp19.evaluate()(input_376, input_377, )
res20 = intrp20.evaluate()(input_376, input_377, )
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
module21.set_input('var_376', input_376)
module21.set_input('var_377', input_377)
module21.set_input(**params)
module21.run()
res22 = intrp22.evaluate()(input_376, input_377, )
res23 = intrp23.evaluate()(input_376, input_377, )
res24 = intrp24.evaluate()(input_376, input_377, )
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