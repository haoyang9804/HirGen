from tvm.relay.ty import TypeVar
import json
from tvm.testing import assert_allclose
import tvm.relay.transform
from tvm.relay.testing import Prelude
from tvm.relay.backend.interpreter import RefValue
from functools import wraps
from tvm.relay import Any
from tvm.relay import create_executor
from tvm.relay.analysis import check_kind
from tvm import te
from tvm.relay import analysis
from tvm.relay.testing import check_grad
import os
import sys
from tvm import runtime
import tvm.topi.testing
from tvm.relay.testing import rand
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import TensorType
from tvm.relay.ty import IncompleteType
from tvm.relay.prelude import Prelude
from tvm.relay import TypeFunctor
from tvm import relay
from numpy import isclose
from tvm import autotvm
from tvm.relay.transform import SimplifyInference
from tvm.ir import structural_equal
import numpy as np
import pytest
from tvm.runtime import container
from tvm.relay.testing import run_opt_pass
from tvm.relay import TypeMutator
from tvm.relay.testing import count
from tvm.relay.analysis import Feature
from tvm.relay.ty import TupleType
from tvm import nd
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.ir import IRModule
from tvm.contrib.nvcc import have_fp16
import time
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.analysis import well_formed
from tvm.contrib import graph_runtime
from tvm.relay.transform import to_cps
import random
from tvm.relay.testing import make_nat_expr
import tvm.relay.transform as _transform
import tvm.testing
from tvm import topi
from tvm.relay import expr as _expr
import itertools
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import FuncType
from tvm.relay.adt import TypeData
from scipy import special
from tvm.relay.analysis import get_calibration_data
import tvm
import math
from tvm.relay.transform import un_cps
import tvm.relay as relay
from tvm.relay.op.annotation import compiler_end
import scipy.sparse as sp
from tvm.relay import op
import scipy
import logging
from tvm.relay.ty import TypeRelation
from tvm.relay import transform
from tvm.relay.testing import create_workload
from tvm.relay.transform import FastMath
import tvm.relay.testing
from tvm.relay.testing import enabled_targets
from tvm.relay import TypeVisitor
from typing import Union
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm import relay as rly
from tvm.relay import testing
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.analysis import detect_feature
from tvm.relay.testing import run_infer_type
from tvm.relay import ExprVisitor
from tvm.relay.ty import RefType
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import TypeCall

xr7Ou=transform.LazyGradientInit()
CPTkc=tvm.IRModule()
Wf7AJ=xr7Ou(CPTkc)
p7odk=create_executor(mod=Wf7AJ)
Q4tqG=relay.Var('''_''')
o0xVU=relay.Function([Q4tqG,],Q4tqG)
CPTkc['''main''']=o0xVU
D91Iz=p7odk.evaluate(CPTkc['''main'''])
EYNpV=rand('''uint16''',*(10,10,))
AeBut=D91Iz(EYNpV)
BkKUo=AeBut.asnumpy()
iNRWX=EYNpV.asnumpy()
assert_allclose(BkKUo,iNRWX)
