from tvm.relay.ty import GlobalTypeVar
import tvm.testing
from tvm.relay import testing
from tvm.relay.ty import RefType
from tvm import nd
import tvm
import tvm.relay as relay
from tvm.relay.ty import FuncType
from tvm.contrib.nvcc import have_fp16
from tvm.testing import assert_allclose
import os
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_end
from tvm.relay import op
from tvm import relay as rly
from tvm.relay.analysis import check_kind
import pytest
import itertools
from tvm.relay.prelude import Prelude
import json
from tvm.relay.ty import TypeRelation
from tvm.relay.ty import TypeVar
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.contrib import graph_runtime
from tvm.relay.ty import TypeCall
from tvm.relay.testing import enabled_targets
from tvm.relay.analysis import get_calibration_data
from tvm.relay import TypeFunctor
from tvm.relay.testing import Prelude
from tvm.relay import create_executor
import numpy as np
import math
from tvm.relay.testing import run_opt_pass
import time
from tvm.relay.transform import SimplifyInference
from tvm.relay.transform import FastMath
from tvm.relay import analysis
import tvm.relay.transform
from scipy import special
from tvm.relay import expr as _expr
import logging
from tvm import topi
from tvm.relay.testing import check_grad
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import IncompleteType
from tvm.relay import ExprVisitor
from tvm.relay.analysis import Feature
from tvm.relay.analysis import detect_feature
from tvm.relay.ty import TensorType
from tvm.relay import TypeVisitor
from tvm.relay import TypeMutator
from tvm.relay.transform import un_cps
from functools import wraps
from numpy import isclose
import scipy.sparse as sp
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.backend.interpreter import RefValue
import tvm.relay.testing
from tvm.ir import IRModule
from tvm.relay import Any
from tvm.relay.testing.synthetic import get_workload
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.runtime import container
import tvm.topi.testing
from typing import Union
from tvm.relay.transform import to_cps
from tvm.relay.ty import TupleType
import tvm.relay.transform as _transform
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.testing import rand
from tvm.autotvm.tuner import RandomTuner
import random
from tvm.relay.adt import TypeData
from tvm.relay.testing import create_workload
from tvm.relay.analysis import well_formed
from tvm.relay.testing import run_infer_type
import sys
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.testing import make_nat_expr
from tvm import runtime
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing import count
import scipy
from tvm.ir import structural_equal

l8uc9=relay.var('''cond''',shape=(1,),dtype='''int64''')
I5NDb=relay.var('''x0''',shape=(1,16,))
KxEn6=relay.const(1,dtype='''int16''')
PlMrc=relay.add(I5NDb,KxEn6)
WMX7P=relay.scalar_type('''uint64''')
zZEcJ=relay.Var('''b''',WMX7P)
nDjyG=relay.Var('''y''')
o85n8=relay.Let(zZEcJ,PlMrc,nDjyG)


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)
wOYSE=relay.const(1.0)
F9z4r=relay.If(zZEcJ,wOYSE,o85n8)
biD7y=transform.InferType()
s2To4=run_opt_pass(F9z4r,biD7y)
A1JhA=s2To4()


def run_opt_pass(expr, passes):
    passes = (passes if isinstance(passes, list) else [passes])
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod['main']
    return (entry if isinstance(expr, relay.Function) else entry.body)


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context('llvm', 0)
    intrp = create_executor(mod=mod, ctx=ctx, target='llvm')
    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)
UziXO=relay.multiply(I5NDb,KxEn6)
O9Tcy=relay.Function([I5NDb,I5NDb,],UziXO)
pRNo9=tvm.IRModule({})
pRNo9['''g0''']=O9Tcy
zeq9L=tvm.IRModule()
zeq9L['''main''']=O9Tcy
zeq9L['''main''']=O9Tcy
zeq9L['''main''']=O9Tcy
TuyT3=O9Tcy()
check_eval(TuyT3,8.0)
ddBPZ=relay.subtract(l8uc9,KxEn6)
yC0vy=relay.TensorType([],'''int64''')
jKfNZ=relay.Function([l8uc9,],ddBPZ,ret_type=yC0vy)
mT1ut=relay.split(l8uc9,2,axis=0)
xhvLA=relay.qnn.op.concatenate(mT1ut,input_scales=(KxEn6,KxEn6,),input_zero_points=(KxEn6,KxEn6,),output_scale=KxEn6,output_zero_point=KxEn6,axis=0)
L5K8w=relay.equal(l8uc9,KxEn6)
qyze1=relay.ScopeBuilder()
lFR1L=qyze1.if_scope(L5K8w)
with lFR1L:
	qyze1.ret(I5NDb)

