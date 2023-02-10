from tvm.autotvm.tuner import RandomTuner
from tvm.relay import TypeVisitor
from tvm.relay.op.annotation import compiler_end
from tvm.relay.ty import RefType
from tvm.relay.analysis import Feature
from tvm import relay as rly
import tvm.testing
from tvm.relay.ty import TypeRelation
from tvm.relay.adt import TypeData
from tvm.relay import Any
from tvm.relay.testing.synthetic import get_workload
from tvm.runtime import container
from tvm.relay import create_executor
from tvm.relay import TypeFunctor
from tvm.relay import expr as _expr
from tvm.relay.op.annotation import compiler_begin
from tvm.relay import op
from tvm.testing import assert_allclose
import tvm.relay.transform
from tvm.relay import analysis
from tvm.relay.testing import count
from tvm.relay import TypeMutator
from tvm.relay.transform import to_cps
from tvm.relay.testing import Prelude
from tvm import runtime
from tvm.relay.analysis import check_kind
from tvm.relay.analysis import check_basic_block_normal_form
import math
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.relay.testing
import os
from tvm import relay
from tvm.relay.ty import TensorType
from tvm.relay.analysis import detect_feature
from tvm.relay import ExprVisitor
from tvm.relay.build_module import bind_params_by_name
from tvm import te
import sys
import random
from tvm import nd
from tvm.relay.testing import make_nat_expr
from tvm.relay.ty import FuncType
from tvm.relay.transform import FastMath
from tvm.relay.testing import run_opt_pass
from tvm.contrib.nvcc import have_fp16
from tvm.relay import testing
from tvm import autotvm
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.analysis import get_calibration_data
from tvm.relay.testing import check_grad
import tvm.relay.transform as _transform
from functools import wraps
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.analysis import well_formed
from tvm.relay.ty import IncompleteType
from tvm.relay.transform import un_cps
import json
import tvm.relay as relay
from tvm.relay.ty import TupleType
from tvm.relay.ty import TypeVar
import logging
from numpy import isclose
from tvm.relay import transform
from tvm.ir import structural_equal
import pytest
from tvm.contrib import graph_runtime
from tvm.relay.transform import SimplifyInference
from tvm.relay.prelude import Prelude
from tvm.relay.testing import run_infer_type
from tvm import topi
from tvm.relay.ty import GlobalTypeVar
import time
from tvm.relay.testing import rand
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.testing import create_workload
import numpy as np
from scipy import special
import scipy.sparse as sp
from typing import Union
from tvm.ir import IRModule
import scipy
import itertools
from tvm.relay.ty import TypeCall
import tvm
import tvm.topi.testing
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import enabled_targets

VoCdv=relay.TensorType((),'''uint64''')
Qh41b=relay.Var('''size''',VoCdv)
QeoOK=tvm.gpu()
kfrJN=relay.op.memory.alloc_storage(Qh41b,Qh41b,QeoOK)
