from tvm.contrib import graph_runtime
from typing import Union
from tvm.relay.analysis import well_formed
from tvm.relay.analysis import check_basic_block_normal_form
import tvm.testing
from tvm.relay.backend.interpreter import RefValue
import tvm.relay.transform
from tvm.relay.op.annotation import compiler_end
import numpy as np
import time
from tvm.relay.build_module import bind_params_by_name
from tvm.autotvm.tuner import RandomTuner
import sys
from tvm.contrib.nvcc import have_fp16
from tvm import relay
import pytest
import scipy
from tvm.relay import ExprVisitor
import random
from tvm.relay.testing import enabled_targets
from tvm.relay.ty import FuncType
from tvm.relay.op.annotation import compiler_begin
import tvm.topi.testing
from tvm.relay.ty import RefType
from tvm.relay.analysis import Feature
from tvm.relay import transform
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.analysis import detect_feature
from tvm.relay.transform import un_cps
import json
from tvm.relay.testing import make_nat_expr
from tvm.testing import assert_allclose
import os
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import TupleType
from tvm.relay.transform import to_cps
from tvm.relay import testing
import tvm.relay.transform as _transform
import logging
from tvm.ir import IRModule
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.ty import GlobalTypeVar
import itertools
import math
from tvm import nd
from scipy import special
from tvm.relay import Any
from tvm.relay import TypeMutator
from tvm.relay.ty import TypeRelation
from tvm.runtime import container
from tvm.relay.analysis import get_calibration_data
from numpy import isclose
from tvm.relay.adt import TypeData
from tvm.relay.ty import TypeCall
from tvm.relay.testing import check_grad
from tvm.relay.ty import IncompleteType
from tvm import topi
import scipy.sparse as sp
from tvm.relay import op
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import run_opt_pass
from tvm.relay import analysis
import tvm.relay as relay
from tvm import te
from tvm.relay import expr as _expr
from tvm import autotvm
from tvm.relay import TypeFunctor
from tvm import runtime
from tvm.relay.testing import create_workload
from tvm.relay.testing import run_infer_type
from tvm.relay.transform import SimplifyInference
from tvm.relay.analysis import check_kind
from tvm.relay import TypeVisitor
from tvm.relay.prelude import Prelude
import tvm
from tvm.ir import structural_equal
from tvm.relay.testing import count
import tvm.relay.testing
from functools import wraps
from tvm.relay.ty import TensorType
from tvm.relay.transform import FastMath
from tvm.relay import create_executor
from tvm.relay.testing import rand
from tvm.relay.ty import TypeVar
from tvm.relay.testing import Prelude
from tvm import relay as rly

o6Rz7=relay.TensorType((),'''uint16''')
Ut2Vt=relay.TypeVar('''a''')
wa6Br=relay.GlobalTypeVar('''id''')
EQScj=relay.TypeData(wa6Br,[Ut2Vt,],[])
VHSZS=tvm.IRModule()
vYZAY=relay.var('''x''',shape=(2,))
mA3Vu=relay.Function([vYZAY,],(vYZAY + vYZAY))
VHSZS['''main''']=mA3Vu
p8CiG=relay.GlobalVar('''f''')
VHSZS[p8CiG]=mA3Vu
VHSZS['''main''']=mA3Vu
VHSZS[wa6Br]=EQScj
NaU4u=wa6Br(Ut2Vt)
RCU2x=relay.FuncType([Ut2Vt,],NaU4u,[Ut2Vt,])
JAmnK=relay.Var('''b''',RCU2x)
m1ntg=tvm.gpu()
ZumtS=relay.op.memory.alloc_storage(JAmnK,JAmnK,m1ntg)
XZz0f=JAmnK(JAmnK)
