from tvm.relay.testing import check_grad
from tvm import te
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import enabled_targets
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.op.annotation import compiler_end
from tvm.relay import op
from tvm.relay.transform import un_cps
import numpy as np
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm import nd
from tvm.relay.adt import TypeData
import scipy.sparse as sp
from tvm.relay import Any
import tvm.relay.transform
from tvm.relay.transform import SimplifyInference
from scipy import special
from numpy import isclose
from tvm.relay import TypeMutator
from tvm.relay import expr as _expr
from tvm.relay.transform import to_cps
from tvm.relay.ty import TupleType
from tvm.relay import analysis
from tvm.relay.testing import create_workload
from tvm.relay.testing.synthetic import get_workload
from typing import Union
from tvm.relay.analysis import well_formed
from tvm.testing import assert_allclose
import time
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
from tvm import relay
from tvm import autotvm
from tvm.relay.analysis import check_basic_block_normal_form
import random
import tvm.relay as relay
from tvm.relay.testing import rand
from tvm.relay import transform
from tvm.ir import structural_equal
import tvm.testing
from tvm.relay.analysis import Feature
from tvm import topi
from tvm.relay import ExprVisitor
from tvm.relay import create_executor
from tvm.relay.analysis import get_calibration_data
from tvm.relay import testing
from tvm.relay.ty import FuncType
from tvm.autotvm.tuner import RandomTuner
import tvm.relay.testing
from tvm.runtime import container
from tvm.relay.ty import IncompleteType
from tvm.relay import TypeFunctor
from tvm.relay.ty import RefType
import sys
from tvm.relay.ty import TypeVar
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.ty import TypeRelation
from tvm.relay import TypeVisitor
from tvm.contrib.nvcc import have_fp16
from tvm.relay.analysis import check_kind
from tvm.relay.testing import Prelude
from tvm.relay.testing import run_infer_type
import tvm.relay.transform as _transform
from tvm.relay.ty import TensorType
from functools import wraps
from tvm.relay.testing import count
from tvm.relay.analysis import detect_feature
from tvm.contrib import graph_runtime
import pytest
from tvm.ir import IRModule
from tvm.relay.testing import run_opt_pass
from tvm import runtime
from tvm.relay.ty import GlobalTypeVar
import scipy
import os
import itertools
from tvm import relay as rly
from tvm.relay.build_module import bind_params_by_name
import logging
from tvm.relay.transform import FastMath
import tvm.topi.testing
from tvm.relay.testing import make_nat_expr
from tvm.relay.op.annotation import compiler_begin
import tvm
import json
import math
from tvm.relay.ty import TypeCall

UMzfM=relay.const(0)
C2j7O=relay.var('''cond''',shape=(1,4,),dtype='''uint1''')
V12QU=relay.Function([C2j7O,C2j7O,],UMzfM)
xMjJn=tvm.IRModule.from_expr(V12QU)
OBGDz=relay.transform.ToANormalForm()
spY7X=OBGDz(xMjJn)


def run_infer_type(expr, mod=None):
    if (not mod):
        mod = tvm.IRModule.from_expr(expr)
        mod = transform.InferType()(mod)
        entry = mod['main']
        return (entry if isinstance(expr, relay.Function) else entry.body)
    else:
        if isinstance(expr, relay.GlobalVar):
            gv = expr.name_hint
        else:
            func = expr
            if (not isinstance(expr, relay.Function)):
                func = relay.Function(analysis.free_vars(expr), expr)
            mod['main'] = func
            gv = 'main'
        mod = transform.InferType()(mod)
        if isinstance(expr, (relay.GlobalVar, relay.Function)):
            return mod[gv]
        return mod[gv].body
LFVmZ=relay.var('''data''',shape=(0,16,))
PxPVc=tvm.IRModule()
PxPVc['''main''']=V12QU
vjqtO=PxPVc.get_global_type_var('''Storage''')
BRoDh=relay.TypeCall(vjqtO,[])
zpMfC=relay.Var('''scale_w''',BRoDh)
LjKQx=relay.nn.upsampling(LFVmZ,1.0,zpMfC)
Ocig3=run_infer_type(LjKQx)
