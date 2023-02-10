from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.ty import TypeCall
import sys
import tvm.relay.transform as _transform
from tvm.relay.ty import TypeVar
from tvm import relay
import pytest
from tvm.relay.testing.synthetic import get_workload
import tvm.relay.transform
from tvm.relay.backend.interpreter import RefValue
from functools import wraps
from tvm import autotvm
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import TypeVisitor
from tvm.relay import op
from tvm.relay.testing import run_opt_pass
from tvm.ir import IRModule
import os
import math
from numpy import isclose
from tvm import topi
from tvm.relay.testing import rand
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import count
from tvm import nd
from tvm.relay.transform import SimplifyInference
from scipy import special
from tvm.relay.ty import TensorType
from tvm.relay.testing import run_infer_type
from tvm.relay.transform import to_cps
import numpy as np
import itertools
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.ty import TypeRelation
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import enabled_targets
from tvm.relay.transform import FastMath
from tvm.relay.backend.interpreter import ConstructorValue
import json
from tvm.relay.adt import TypeData
from tvm.relay.ty import FuncType
from tvm.relay.analysis import check_kind
import tvm.relay.testing
from tvm.relay.testing import check_grad
import logging
from tvm.relay.testing import create_workload
from tvm.relay import expr as _expr
from tvm.relay.ty import TupleType
import scipy.sparse as sp
from tvm.testing import assert_allclose
from tvm.contrib import graph_runtime
import tvm.testing
import tvm.relay as relay
from tvm.relay.op.annotation import compiler_end
import time
from tvm.relay.analysis import well_formed
from tvm.relay.transform import un_cps
from tvm.relay.analysis import detect_feature
from tvm.contrib.nvcc import have_fp16
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import testing
from tvm import te
import random
from tvm.relay import analysis
from tvm.relay.ty import IncompleteType
from tvm import runtime
from tvm.relay import transform
from tvm.relay import Any
import tvm
from tvm.relay import ExprVisitor
from tvm.relay import TypeMutator
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import RefType
from tvm import relay as rly
from typing import Union
from tvm.relay.op.annotation import compiler_begin
from tvm.autotvm.tuner import RandomTuner
from tvm.relay import TypeFunctor
from tvm.relay.prelude import Prelude
from tvm.relay.testing import Prelude
from tvm.ir import structural_equal
from tvm.relay import create_executor
import scipy
import tvm.topi.testing
from tvm.runtime import container
from tvm.relay.analysis import Feature

mgOVs=relay.TypeVar('''A''')
dzDl2=relay.var('''y1''',mgOVs)
yigjB=relay.nn.dense(dzDl2,dzDl2,units=1,out_dtype='''uint8''')


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
cuBtL=run_infer_type(yigjB)
mTVM8=tvm.IRModule()
mTVM8['''main''']=cuBtL
N5iUM=relay.GlobalTypeVar('''gtv1''')
PXT5P=relay.Constructor('''Nil''',[],N5iUM)
jE0Zc=relay.TypeData(N5iUM,[],[PXT5P,])
mTVM8[N5iUM]=jE0Zc
mTVM8[N5iUM]=jE0Zc
mTVM8['''main''']=cuBtL
WNuSY=relay.GlobalVar('''g0''')
cfY1f=relay.Function([dzDl2,dzDl2,],(dzDl2 + dzDl2))
mTVM8['''main''']=cfY1f
Nv4Hn=cfY1f.with_attr('''Compiler''','''test_graph''')
mTVM8[WNuSY]=Nv4Hn
sSX2E=cfY1f.with_attr('''Compiler''','''test_graph''')
mTVM8[WNuSY]=sSX2E
mTVM8['''main''']=cuBtL
mTVM8['''main''']=cuBtL
itZ1L=relay.transform.gradient(mTVM8['''main'''],mode='''higher_order''')
