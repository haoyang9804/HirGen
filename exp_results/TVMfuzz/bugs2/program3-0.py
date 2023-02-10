from tvm.ir import IRModule
import tvm.relay.testing
import tvm
from tvm import topi
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import TypeVar
from tvm.relay import expr as _expr
from tvm.ir import structural_equal
import os
from tvm.relay.analysis import check_kind
from tvm.relay.testing import check_grad
import tvm.testing
from tvm import te
from tvm.relay import ExprVisitor
from tvm.relay import testing
from tvm import relay
from tvm.relay.testing import count
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.op.annotation import compiler_end
from tvm.relay.transform import un_cps
import tvm.relay.transform as _transform
import random
from tvm.relay.testing import make_nat_expr
from tvm.relay.analysis import well_formed
from tvm.relay.analysis import Feature
import itertools
from tvm import relay as rly
import json
from tvm.relay.op.annotation import compiler_begin
from tvm.runtime import container
from tvm.autotvm.tuner import RandomTuner
from tvm.relay import create_executor
from tvm.relay.ty import RefType
from tvm.relay.prelude import Prelude
import tvm.relay.transform
from tvm.relay.ty import TypeRelation
from tvm.relay.testing import enabled_targets
import scipy
from numpy import isclose
from tvm.relay.ty import TupleType
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from functools import wraps
from tvm.relay.transform import FastMath
from tvm.relay.analysis import detect_feature
from tvm.relay.ty import GlobalTypeVar
from tvm.contrib import graph_runtime
import pytest
from tvm.relay.ty import TensorType
from tvm.relay import op
from tvm.relay import transform
from tvm.relay import analysis
from tvm.relay import TypeMutator
from tvm.relay.backend.interpreter import RefValue
import scipy.sparse as sp
import math
from tvm.relay.testing import Prelude
from tvm.relay import Any
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.contrib.nvcc import have_fp16
from scipy import special
from tvm import runtime
from tvm.relay.transform import to_cps
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.ty import TypeCall
from tvm.relay.scope_builder import ScopeBuilder
import sys
from tvm.relay.testing import rand
from tvm.relay.testing import run_opt_pass
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import IncompleteType
import tvm.topi.testing
from tvm import autotvm
from tvm.relay.adt import TypeData
from tvm.testing import assert_allclose
from tvm.relay import TypeVisitor
import tvm.relay as relay
import time
from tvm.relay.testing import run_infer_type
from typing import Union
import logging
from tvm.relay.testing import create_workload
from tvm.relay.ty import FuncType
from tvm.relay.transform import SimplifyInference
import numpy as np
from tvm.relay.analysis import check_basic_block_normal_form
from tvm import nd
from tvm.relay import TypeFunctor

VsJTm=tvm.runtime.convert([0,0,3,])
Mfven=relay.TypeRelation(None,VsJTm,1,None)
YSEEI=Any()
a2HKQ=relay.TensorType([YSEEI,0,],dtype='''int32''')
jCxj3=relay.FuncType(VsJTm,a2HKQ,VsJTm,VsJTm)
