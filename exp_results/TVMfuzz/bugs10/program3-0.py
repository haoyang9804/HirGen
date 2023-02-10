from tvm.relay.analysis import get_calibration_data
from tvm.relay import op
from tvm.relay import expr as _expr
from tvm.relay.ty import FuncType
from functools import wraps
import tvm.topi.testing
from tvm.relay.testing import enabled_targets
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
from tvm import nd
from tvm.testing import assert_allclose
from tvm.relay.ty import TupleType
import logging
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.adt import TypeData
import pytest
from tvm.relay.analysis import detect_feature
from typing import Union
from tvm.relay import Any
import tvm.relay.transform as _transform
import time
from tvm.relay.testing import make_nat_expr
import scipy
from tvm.relay.backend.interpreter import RefValue
from tvm.ir import IRModule
from tvm.ir import structural_equal
import scipy.sparse as sp
from tvm import te
from tvm.relay.ty import TensorType
import sys
from tvm.relay import TypeVisitor
from tvm.relay.transform import to_cps
from tvm.relay import ExprVisitor
from tvm.relay.testing import Prelude
from tvm.relay.testing import rand
from tvm.relay.ty import GlobalTypeVar
import tvm.relay.testing
from tvm.relay.testing import create_workload
from tvm.autotvm.tuner import RandomTuner
import math
import tvm
from tvm.relay.testing import check_grad
from tvm.relay.op.annotation import compiler_end
from tvm.relay.testing.synthetic import get_workload
from tvm.runtime import container
from numpy import isclose
from tvm.relay.scope_builder import ScopeBuilder
import tvm.relay as relay
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import count
from tvm import relay
from tvm import topi
from tvm.relay.transform import SimplifyInference
from tvm.relay.ty import TypeCall
from tvm.relay.analysis import well_formed
from tvm.relay import testing
from tvm.relay.ty import TypeVar
from tvm.relay.build_module import bind_params_by_name
import os
from tvm import relay as rly
from tvm.relay import create_executor
import random
from tvm.relay import TypeMutator
from tvm.relay.ty import IncompleteType
from tvm import runtime
from tvm.relay.analysis import check_kind
from tvm import autotvm
from tvm.relay.testing import run_infer_type
import tvm.testing
import json
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.backend.interpreter import ConstructorValue
import tvm.relay.transform
from tvm.relay import analysis
from scipy import special
from tvm.relay.transform import FastMath
from tvm.relay import TypeFunctor
from tvm.relay.prelude import Prelude
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.ty import TypeRelation
from tvm.relay.ty import RefType
import numpy as np
from tvm.relay.transform import un_cps
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.contrib import graph_runtime
import itertools
from tvm.relay.analysis import Feature

j74lt=FastMath()
t0UY0=relay.TensorType((10,10,),'''float16''')
loRtB=relay.var('''x''',t0UY0)
VgOb7=relay.Function([loRtB,],(loRtB + loRtB))
Ob15W=tvm.IRModule()
bB6kr=relay.GlobalTypeVar('''Ayy''')
o4pCe=relay.TypeData(bB6kr,[],[])
Ob15W[bB6kr]=o4pCe
Ob15W['''main''']=VgOb7
IkfLc=relay.GlobalVar('''f''')


def check_memory_plan(func, check_fn):
    mod = tvm.IRModule().from_expr(func)
    args = []
    for param in func.params:
        param = param.type_annotation
        sh = [int(sh) for sh in param.shape]
        data = np.random.rand(*sh).astype(param.dtype)
        args.append(tvm.nd.array(data))
    ex = relay.create_executor('vm', mod)
    no_plan_result = ex.evaluate(mod['main'])(*args)
    with tvm.transform.PassContext(opt_level=1, disabled_pass=['MemoryPlan']):
        plan_result = ex.evaluate(mod['main'])(*args)
    py_res = check_fn(*[arg.asnumpy() for arg in args])
    np.testing.assert_allclose(no_plan_result.asnumpy(), plan_result.asnumpy())
    np.testing.assert_allclose(plan_result.asnumpy(), py_res)


def storage_type(mod):
    return relay.TypeCall(mod.get_global_type_var('Storage'), [])
hxlal=storage_type(Ob15W)
hvU4H=relay.Var('''x''',hxlal)
hGhln=np.ones((2,10,),'''float32''')
D7Uoc=relay.const(hGhln)
EMFqN=relay.const(2,'''uint64''')
Bqt4O=IkfLc((hvU4H - EMFqN))
f9UwA=relay.If(D7Uoc,(hvU4H * EMFqN),((hvU4H * EMFqN) + Bqt4O))
xhjXt=relay.Function([hvU4H,],f9UwA,t0UY0,[])
Ob15W[IkfLc]=xhjXt
Ob15W['''main''']=VgOb7
Ob15W['''main''']=VgOb7
Ob15W['''main''']=VgOb7
AktM9=tvm.IRModule.from_expr(VgOb7)
L5b9e=j74lt(AktM9)
