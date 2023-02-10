import numpy as np
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.analysis import check_basic_block_normal_form
from typing import Union
from tvm.contrib.nvcc import have_fp16
import sys
from tvm.runtime import container
import scipy
from numpy import isclose
from tvm import relay as rly
from tvm.relay.ty import FuncType
from tvm.relay.testing import check_grad
from tvm.relay.ty import RefType
from tvm.relay.testing import Prelude
from tvm.ir import IRModule
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_end
from tvm.relay.transform import to_cps
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.analysis import detect_feature
from tvm.relay.prelude import Prelude
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm import relay
import math
from tvm.relay.testing import run_infer_type
from tvm import te
from tvm.relay.testing import enabled_targets
from tvm.relay.ty import TypeCall
from tvm.relay import create_executor
from tvm.relay.testing import make_nat_expr
from tvm.relay.analysis import check_kind
from tvm.relay.ty import TensorType
from tvm.relay.testing import count
import tvm.relay.testing
import os
from scipy import special
import json
from tvm.ir import structural_equal
import tvm
from tvm.relay import expr as _expr
from tvm import nd
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.ty import TypeVar
from functools import wraps
from tvm.relay.transform import SimplifyInference
from tvm import topi
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import Any
import itertools
from tvm.relay.analysis import get_calibration_data
from tvm import runtime
from tvm.relay.ty import TupleType
from tvm.relay.ty import IncompleteType
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import create_workload
from tvm.relay.backend.interpreter import ConstructorValue
import scipy.sparse as sp
from tvm.relay.transform import FastMath
from tvm.relay import TypeMutator
from tvm.relay import testing
from tvm import autotvm
from tvm.testing import assert_allclose
import tvm.relay as relay
from tvm.relay.testing import run_opt_pass
import time
from tvm.relay import TypeVisitor
from tvm.relay.analysis import well_formed
import pytest
import logging
from tvm.relay.ty import TypeRelation
import tvm.relay.transform
import tvm.relay.transform as _transform
from tvm.relay.testing import rand
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import op
from tvm.relay.adt import TypeData
import tvm.testing
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.analysis import Feature
from tvm.contrib import graph_runtime
import random
from tvm.relay import ExprVisitor
import tvm.topi.testing
from tvm.relay import TypeFunctor
from tvm.relay import analysis
from tvm.relay.transform import un_cps

sJOpT=tvm.transform.PassContext(opt_level=3,required_pass=['''FastMath''',])
IDzLw=relay.var('''x''',shape=(2,))
ZYvmX=tvm.cpu()
Tdogq=tvm.gpu()
yKQwX=relay.op.device_copy(IDzLw,ZYvmX,Tdogq)
rjFxY=np.random.rand(0,1)
pr72O=relay.const(rjFxY)
CKSYB=relay.Function([IDzLw,],(yKQwX + pr72O))
sxOwk=tvm.IRModule()
CLhoY=relay.GlobalTypeVar('''gtv2''')
An6aQ=relay.TypeVar('''tv''')
SCzAV=relay.TypeData(CLhoY,[An6aQ,],[])
sxOwk[CLhoY]=SCzAV
B02L5=relay.GlobalVar('''main''')
sxOwk[B02L5]=CKSYB
sxOwk['''main''']=CKSYB
ZOEC8=tvm.IRModule.from_expr(CKSYB)
with sJOpT:
	KJL73=relay.optimize(ZOEC8,target='''llvm''',params=None)

