from tvm.relay import transform
import sys
from tvm.relay.ty import GlobalTypeVar
import numpy as np
from tvm.testing import assert_allclose
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.prelude import Prelude
from tvm.relay.analysis import Feature
from tvm.relay import expr as _expr
import tvm.topi.testing
from tvm.relay import create_executor
from tvm import runtime
from tvm.relay import TypeVisitor
from tvm.relay.backend.interpreter import RefValue
import tvm.relay.testing
from tvm.relay.ty import TypeCall
from tvm import relay
from tvm import relay as rly
import math
from functools import wraps
from tvm.relay.analysis import check_kind
from tvm.runtime import container
import json
from tvm.relay.transform import un_cps
from tvm.relay.ty import TypeVar
from tvm.relay.testing import Prelude
from tvm.relay.transform import SimplifyInference
from tvm.relay import TypeFunctor
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.annotation import compiler_end
import tvm.relay.transform
from tvm.relay import ExprVisitor
import tvm.relay.transform as _transform
import tvm.relay as relay
import logging
import tvm
from tvm.ir import IRModule
import random
from tvm.relay.testing import check_grad
from tvm.relay.testing import run_infer_type
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import run_opt_pass
from scipy import special
from tvm.relay.ty import TupleType
from tvm.relay import analysis
from tvm.relay.adt import TypeData
from tvm.relay.analysis import get_calibration_data
import itertools
from tvm.relay.testing import count
from tvm import te
from tvm.contrib.nvcc import have_fp16
from tvm.relay import TypeMutator
import pytest
from tvm.relay.testing import enabled_targets
from tvm.contrib import graph_runtime
from tvm.relay.scope_builder import ScopeBuilder
from numpy import isclose
from tvm.relay.transform import FastMath
from tvm.relay import Any
from tvm.relay.testing import rand
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.ty import IncompleteType
from tvm.relay import op
from tvm import autotvm
from tvm import topi
from tvm.relay.ty import TypeRelation
from tvm.relay.analysis import detect_feature
import tvm.testing
from tvm.autotvm.tuner import RandomTuner
import scipy
from tvm.relay.ty import FuncType
from tvm.ir import structural_equal
from tvm.relay.testing import create_workload
from tvm.relay.testing.synthetic import get_workload
import time
import os
from tvm.relay.ty import TensorType
import scipy.sparse as sp
from tvm.relay.transform import to_cps
from tvm.relay.ty import RefType
from tvm.relay import testing
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay.analysis import well_formed
from tvm import nd
from typing import Union

Dw2ho=tvm.transform.PassContext(opt_level=2,required_pass=['''FastMath''',])
ekiun=relay.var('''x''',shape=(2,3,))
PS3a1=tvm.cpu()
ZTYfm=tvm.gpu()
kLlEp=relay.op.device_copy(ekiun,PS3a1,ZTYfm)
iiok4=np.random.rand(2,3)
Z3BPq=relay.const(iiok4)
Gf6z2=relay.Function([ekiun,],(kLlEp + Z3BPq))
jYdBT=tvm.IRModule.from_expr(Gf6z2)
with Dw2ho:
	vXWdi=relay.optimize(jYdBT,target='''llvm''',params=None)

