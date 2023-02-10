from tvm.relay.ty import GlobalTypeVar
from tvm.relay import TypeFunctor
from tvm.relay.transform import to_cps
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.testing import count
import math
from tvm import te
from tvm.relay.op.annotation import compiler_end
from tvm.relay.transform import FastMath
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.contrib import graph_runtime
from tvm.relay.testing import run_infer_type
from tvm import runtime
import sys
from tvm.relay.analysis import Feature
import scipy
from tvm.relay.ty import TypeVar
from tvm.testing import assert_allclose
from tvm import nd
from tvm.relay import transform
import tvm.relay.testing
from tvm.relay.ty import RefType
from tvm.relay.ty import TupleType
from tvm.relay.analysis import detect_feature
from tvm.relay.testing.temp_op_attr import TempOpAttr
from numpy import isclose
from tvm import relay as rly
import itertools
from tvm.relay import Any
from tvm import relay
import pytest
from tvm.relay.prelude import Prelude
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.adt import TypeData
from tvm.relay.testing import Prelude
from tvm.relay import create_executor
import tvm.relay.transform as _transform
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.interpreter import RefValue
import scipy.sparse as sp
import os
from tvm.relay import TypeMutator
import numpy as np
from tvm.relay.testing import rand
from tvm.ir import IRModule
from tvm.relay.testing import make_nat_expr
from tvm.relay.transform import SimplifyInference
from tvm.relay import op
from tvm.ir import structural_equal
from tvm.relay.testing import check_grad
import tvm.relay as relay
from tvm.relay.data_dep_optimization import simplify_fc_transpose
import tvm.topi.testing
from tvm.relay.analysis import well_formed
from tvm import autotvm
from tvm.relay.transform import un_cps
import random
import tvm.relay.transform
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.ty import FuncType
from tvm.runtime import container
from functools import wraps
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import TypeCall
from tvm.relay import expr as _expr
import logging
from tvm import topi
import tvm.testing
from tvm.contrib.nvcc import have_fp16
from tvm.relay.ty import TypeRelation
from tvm.relay import analysis
from tvm.relay import testing
import tvm
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.testing import create_workload
import time
from tvm.relay import TypeVisitor
from tvm.relay.testing import enabled_targets
from scipy import special
from tvm.relay import ExprVisitor
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import TensorType
from typing import Union
import json
from tvm.relay.analysis import check_kind

LMLlW=tvm.gpu()
doJen=relay.scalar_type('''uint32''')
BoZJC=relay.Var('''alignment''',doJen)
ZvHXQ=relay.op.memory.alloc_storage(BoZJC,BoZJC,LMLlW)
