from tvm.relay.testing import count
from tvm.relay.analysis import get_calibration_data
from tvm import runtime
import sys
from tvm import autotvm
from tvm.relay.testing import rand
from tvm.relay.analysis import well_formed
from tvm.relay import TypeMutator
from tvm.ir import structural_equal
from tvm.relay import TypeVisitor
from tvm.relay.transform import FastMath
import tvm.topi.testing
from tvm.relay.transform import to_cps
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import FuncType
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay.transform
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import Any
from tvm.autotvm.tuner import RandomTuner
import tvm.relay.testing
from tvm.relay.testing import run_infer_type
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.prelude import Prelude
import numpy as np
from tvm.relay import testing
from tvm.relay import ExprVisitor
import logging
from tvm.relay.analysis import check_basic_block_normal_form
import tvm.relay.transform as _transform
from tvm.relay.testing import Prelude
from tvm.relay.ty import RefType
from tvm.relay.testing import run_opt_pass
from scipy import special
from tvm.relay.analysis import check_kind
from tvm import relay
import random
from tvm.relay.adt import TypeData
from tvm.relay.ty import TupleType
from tvm.ir import IRModule
import scipy
from tvm import topi
from tvm.relay.ty import IncompleteType
import os
from tvm.relay import op
import tvm.relay as relay
from tvm.testing import assert_allclose
from tvm.relay import TypeFunctor
from tvm.contrib import graph_runtime
from tvm.relay.ty import TypeCall
from tvm.relay.backend.interpreter import RefValue
from tvm import te
import math
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.op.annotation import compiler_begin
import time
from typing import Union
from tvm.relay.ty import TensorType
from tvm.relay import analysis
from functools import wraps
import pytest
from tvm.relay.op.annotation import compiler_end
import tvm
from tvm.relay.testing import enabled_targets
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import json
from tvm import nd
from tvm.relay.testing import check_grad
from tvm.runtime import container
from tvm.relay.transform import un_cps
from tvm.relay.analysis import Feature
from tvm.relay.analysis import detect_feature
from tvm.relay import create_executor
from tvm.relay.ty import TypeRelation
import scipy.sparse as sp
from numpy import isclose
from tvm.relay.testing import create_workload
from tvm.relay import expr as _expr
import tvm.testing
from tvm import relay as rly
import itertools
from tvm.relay.testing import make_nat_expr
from tvm.relay.transform import SimplifyInference
from tvm.contrib.nvcc import have_fp16
from tvm.relay.ty import TypeVar

ac2FO=relay.var('''x1''',shape=(1,64,56,56,))
v0bz3=relay.nn.conv2d(ac2FO,ac2FO,kernel_size=(3,3,),padding=(1,1,),groups=32)
L2xns=relay.const(1)
h3Qca=relay.add(v0bz3,L2xns)
P45o2=relay.op.add(ac2FO,ac2FO)
fttPf=relay.Function([],P45o2)
EnEn5=tvm.IRModule()
S4Ddb=relay.GlobalVar('''g1''')
aGo6X=fttPf.with_attr('''Compiler''','''test_graph''')
EnEn5[S4Ddb]=aGo6X
EnEn5['''main''']=fttPf
mSAfV=tvm.IRModule({})
mSAfV[S4Ddb]=fttPf
mSAfV[S4Ddb]=fttPf
mSAfV['''main''']=fttPf


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)
qGX8U=transform.InferType()
onDWR=run_opt_pass(fttPf,qGX8U)
XPCiK=analysis.get_total_mac_number(onDWR)
cPcJt=fttPf.with_attr('''Compiler''','''test_graph''')
q2rEC=tvm.runtime.convert([v0bz3,v0bz3,h3Qca,])
QEtA8=relay.TupleType(q2rEC)
OqLDp=relay.TypeVar('''tp1''',relay.TypeKind.ShapeVar)
Do8lm=relay.FuncType(q2rEC,OqLDp,q2rEC,q2rEC)
check_kind(Do8lm)
fLIfO=te.reduce_axis((0,q2rEC,))
XpCPA=te.placeholder((q2rEC,q2rEC,),name='''A''')
CE5GD=relay.var('''i''',shape=[],dtype='''uint16''')
WFcuf=te.compute((q2rEC,q2rEC,),lambda i,j: te.sum((XpCPA[CE5GD,fLIfO] * XpCPA[fLIfO,CE5GD]),axis=fLIfO,init=XpCPA[CE5GD,CE5GD]),name='''B''')
nmMzo=te.create_schedule(WFcuf.op)
wIj0n=nmMzo[WFcuf].split(fLIfO,nparts=4)
l7UbL=nmMzo.rfactor(WFcuf,wIj0n[0],1)
nmMzo[l7UbL].parallel(l7UbL.op.axis[0])
