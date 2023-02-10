from tvm.runtime import container
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.ty import RefType
from tvm.relay.analysis import well_formed
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import enabled_targets
from tvm.relay import analysis
import tvm.relay.transform
from tvm.relay.ty import IncompleteType
import tvm.topi.testing
import scipy
from tvm.relay import transform
from tvm.relay.testing import rand
from tvm.relay.testing import Prelude
from tvm.relay import TypeMutator
from tvm.relay.ty import TupleType
from numpy import isclose
from functools import wraps
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import FuncType
from tvm.relay.testing import create_workload
from tvm.relay.testing import check_grad
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.testing import run_opt_pass
from tvm.relay.analysis import Feature
import tvm.relay.transform as _transform
from tvm.testing import assert_allclose
from tvm.relay.ty import TensorType
import sys
from tvm.relay.transform import SimplifyInference
from tvm.contrib import graph_runtime
import time
import tvm.relay as relay
from tvm.relay.transform import un_cps
import numpy as np
from tvm.relay import op
from tvm.relay.transform import FastMath
from tvm import topi
from tvm.relay.build_module import bind_params_by_name
from tvm.ir import IRModule
from tvm.relay.adt import TypeData
from tvm import runtime
import math
from tvm.relay.analysis import detect_feature
from tvm import autotvm
from tvm import relay as rly
from tvm.relay.testing import make_nat_expr
from tvm.relay import ExprVisitor
import tvm.testing
import tvm
from tvm import nd
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.contrib.nvcc import have_fp16
from tvm.relay.scope_builder import ScopeBuilder
import itertools
from tvm.relay import TypeFunctor
from typing import Union
from tvm.relay.ty import TypeVar
import json
import pytest
from tvm import te
from tvm.relay import Any
from tvm.relay.testing import run_infer_type
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import TypeRelation
from tvm.relay.backend.interpreter import RefValue
from tvm.ir import structural_equal
from tvm.relay import testing
from tvm import relay
from tvm.relay.op.annotation import compiler_end
from tvm.autotvm.tuner import RandomTuner
import logging
from scipy import special
from tvm.relay import expr as _expr
from tvm.relay.testing import count
import scipy.sparse as sp
import os
import tvm.relay.testing
from tvm.relay import create_executor
from tvm.relay import TypeVisitor
from tvm.relay.transform import to_cps
from tvm.relay.ty import TypeCall
from tvm.relay.prelude import Prelude
from tvm.relay.backend.interpreter import ConstructorValue
import random
from tvm.relay.analysis import check_kind

fyqWI=relay.TensorType((),'''int64''')
PFECu=relay.Var('''alignment''',fyqWI)


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


def assert_has_type(expr, typ, mod=tvm.IRModule({})):
    checked_expr = run_infer_type(expr, mod)
    checked_type = checked_expr.checked_type
    if (checked_type != typ):
        raise RuntimeError(('Type mismatch %s vs %s' % (checked_type, typ)))


def initialize_box_adt(mod):
    box = relay.GlobalTypeVar('box')
    tv = relay.TypeVar('tv')
    constructor = relay.Constructor('constructor', [tv], box)
    data = relay.TypeData(box, [tv], [constructor])
    mod[box] = data
    return (box, constructor)
MOdvT=tvm.IRModule()
QDLvN=relay.var('''x0''',shape=(8,8,))
kwse6=relay.const(0)
pJvjQ=relay.Function([QDLvN,QDLvN,],kwse6)
MOdvT['''main''']=pJvjQ
snUwf=pJvjQ.with_attr('''Compiler''','''test_graph''')
qwzrJ=relay.GlobalVar('''g1''')
MOdvT[qwzrJ]=snUwf
t7YJr=pJvjQ.with_attr('''Compiler''','''test_graph''')
MOdvT[qwzrJ]=t7YJr
o8v71=initialize_box_adt(MOdvT)
YzHrr=o8v71[1](PFECu)
OZpYm=relay.TypeVar('''a''')
a8kTj=o8v71[0](OZpYm)
kgU9M=relay.Function([PFECu,],YzHrr,a8kTj,[OZpYm,])


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
Ev08S=run_infer_type(kgU9M,MOdvT)
GpQFX=tvm.gpu()
G7TTN=relay.op.memory.alloc_storage(PFECu,PFECu,GpQFX)
