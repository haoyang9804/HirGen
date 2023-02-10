from tvm.relay.testing import enabled_targets
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.analysis import check_kind
import tvm.relay.testing
from tvm.relay import op
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay as relay
from tvm.relay.testing import check_grad
from tvm.testing import assert_allclose
from tvm.relay.ty import TypeRelation
from tvm import te
import json
from tvm.relay.testing import make_nat_expr
from typing import Union
from tvm.relay.ty import RefType
from functools import wraps
from tvm.relay import testing
from tvm.relay import create_executor
from tvm import runtime
from tvm.ir import IRModule
from tvm.autotvm.tuner import RandomTuner
import scipy.sparse as sp
from tvm.relay import ExprVisitor
from tvm.relay.prelude import Prelude
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.ty import IncompleteType
import pytest
import sys
from numpy import isclose
from tvm import relay as rly
from tvm.relay import TypeMutator
from tvm.contrib.nvcc import have_fp16
from tvm.ir import structural_equal
from scipy import special
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay import expr as _expr
import itertools
from tvm import autotvm
from tvm.relay.ty import TupleType
from tvm import nd
from tvm.relay.ty import TensorType
import logging
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.op.annotation import compiler_end
import math
import os
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.adt import TypeData
from tvm.relay.analysis import get_calibration_data
from tvm.relay import Any
from tvm.relay.testing import create_workload
from tvm.relay import TypeVisitor
from tvm.relay.testing import run_infer_type
from tvm.relay.ty import TypeCall
from tvm.relay.transform import SimplifyInference
import tvm.relay.transform as _transform
from tvm.relay.testing import count
from tvm.contrib import graph_runtime
from tvm.relay.op.annotation import compiler_begin
from tvm import relay
from tvm.relay.transform import to_cps
from tvm.relay.analysis import well_formed
from tvm.relay.testing import run_opt_pass
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import TypeFunctor
import tvm.topi.testing
from tvm.relay import transform
from tvm.relay.testing import Prelude
from tvm import topi
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.testing import rand
from tvm.runtime import container
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay import analysis
from tvm.relay.ty import TypeVar
from tvm.relay.transform import un_cps
from tvm.relay.analysis import detect_feature
from tvm.relay.ty import FuncType
from tvm.relay.analysis import Feature
import tvm
import random
import time
import numpy as np
import tvm.testing
from tvm.relay.transform import FastMath
import scipy
import tvm.relay.transform

yEqWj=relay.scalar_type('''int64''')
xkDVc=relay.Var('''_''',yEqWj)
lOe1H=xkDVc()
uAM3u=relay.var('''i''',shape=(1,16,))
yChBC=relay.If(lOe1H,uAM3u,xkDVc)
BLSDo=relay.add(uAM3u,yChBC)
check_basic_block_normal_form(BLSDo)
