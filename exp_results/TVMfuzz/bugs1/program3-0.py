from tvm.relay import TypeFunctor
from tvm import te
import sys
from tvm.autotvm.tuner import RandomTuner
from tvm.contrib import graph_runtime
from tvm import runtime
from tvm import nd
import scipy
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
from tvm.relay import testing
from tvm.relay.transform import SimplifyInference
from tvm.relay.ty import TensorType
from typing import Union
from tvm.relay.ty import RefType
from tvm import relay
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import analysis
import tvm.testing
from tvm.relay.analysis import check_kind
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.ty import TupleType
from tvm import topi
from tvm.ir import IRModule
from tvm.relay.testing import enabled_targets
import scipy.sparse as sp
from tvm.relay import ExprVisitor
from tvm.relay.testing import count
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.op.annotation import compiler_end
from tvm.relay.testing import check_grad
import tvm.topi.testing
from tvm.testing import assert_allclose
from tvm.relay.analysis import well_formed
from tvm.relay import TypeMutator
import json
import tvm
from functools import wraps
import itertools
from tvm.relay.testing.temp_op_attr import TempOpAttr
import pytest
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.analysis import detect_feature
from tvm.relay.transform import un_cps
from tvm.relay import Any
from tvm.relay.testing import run_opt_pass
import time
from tvm import relay as rly
from scipy import special
import random
from tvm.relay.analysis import Feature
from tvm.relay import TypeVisitor
import tvm.relay.testing
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.testing import Prelude
from tvm.relay import op
import tvm.relay.transform
from numpy import isclose
from tvm.relay.ty import TypeRelation
from tvm.relay import create_executor
from tvm.relay.ty import TypeVar
from tvm.ir import structural_equal
from tvm.relay import expr as _expr
import tvm.relay.transform as _transform
import math
from tvm.relay.testing import create_workload
from tvm.relay.testing import rand
from tvm.relay.analysis import get_calibration_data
from tvm.relay import transform
from tvm.relay.ty import TypeCall
from tvm.relay.backend.interpreter import RefValue
import tvm.relay as relay
from tvm.runtime import container
import os
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import make_nat_expr
from tvm.relay.ty import IncompleteType
from tvm.relay.transform import to_cps
from tvm import autotvm
from tvm.relay.transform import FastMath
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type
from tvm.relay.adt import TypeData
from tvm.relay.ty import FuncType
from tvm.relay.ty import GlobalTypeVar
import numpy as np
import logging

PhdZv=relay.TypeVar('''tp1''',relay.TypeKind.ShapeVar)
FjQQJ=tvm.runtime.convert([PhdZv,PhdZv,PhdZv,])
dl66x=relay.TupleType(FjQQJ)
KUnBA=relay.FuncType(FjQQJ,dl66x,FjQQJ,FjQQJ)
check_kind(KUnBA)
