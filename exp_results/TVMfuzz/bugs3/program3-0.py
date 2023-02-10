import math
from tvm.relay.testing import rand
import tvm.relay.transform
from tvm.relay.transform import to_cps
from tvm import relay as rly
from tvm.relay import TypeVisitor
from typing import Union
from tvm.relay.transform import FastMath
from tvm.relay.analysis import check_basic_block_normal_form
from numpy import isclose
from tvm import topi
from tvm.autotvm.tuner import RandomTuner
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.analysis import Feature
from tvm import te
from tvm.relay.op.annotation import compiler_begin
from tvm.relay import Any
from tvm.relay.ty import TypeVar
from tvm.relay.analysis import get_calibration_data
import os
from tvm.relay.ty import RefType
from tvm.relay import create_executor
from tvm.testing import assert_allclose
from tvm.relay.testing import check_grad
from tvm import autotvm
from tvm.relay.ty import TensorType
from tvm.relay.testing import create_workload
import sys
from tvm.relay.adt import TypeData
import tvm.relay as relay
import tvm.relay.testing
import random
import numpy as np
from tvm.relay.testing import count
from tvm.relay.testing import run_infer_type
from tvm.relay import ExprVisitor
from tvm.ir import IRModule
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.ty import TupleType
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import enabled_targets
import tvm.topi.testing
import pytest
import scipy
from tvm import runtime
from tvm.relay.testing import make_nat_expr
from tvm.relay.ty import FuncType
from functools import wraps
from tvm.relay.transform import un_cps
from tvm import nd
from tvm.contrib import graph_runtime
from tvm.relay import expr as _expr
import tvm.relay.transform as _transform
import json
from tvm.relay.testing import run_opt_pass
from tvm.relay.testing.synthetic import get_workload
from tvm.relay.ty import TypeRelation
from tvm.relay import transform
from tvm.relay.transform import SimplifyInference
import itertools
from tvm.runtime import container
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.analysis import well_formed
from tvm import relay
from tvm.relay import analysis
import logging
from tvm.relay.ty import TypeCall
from tvm.relay import TypeMutator
import tvm
import time
from tvm.relay.scope_builder import ScopeBuilder
import scipy.sparse as sp
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.ir import structural_equal
import tvm.testing
from tvm.relay.analysis import check_kind
from tvm.relay import TypeFunctor
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay import testing
from tvm.relay.testing import Prelude
from tvm.relay.op.annotation import compiler_end
from tvm.relay.prelude import Prelude
from tvm.relay.ty import IncompleteType
from tvm.relay import op
from scipy import special
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.analysis import detect_feature

dnWF9=transform.ToBasicBlockNormalForm()
gJ3aB=tvm.IRModule()
c23mR=relay.var('''shared_bound''',shape=[],dtype='''float16''')
zKSt6=relay.Function([c23mR,c23mR,],(c23mR + c23mR))
nU3GF=zKSt6.with_attr('''Compiler''','''test_graph''')
M9JGT=relay.GlobalVar('''g0''')
gJ3aB[M9JGT]=nU3GF
gJ3aB['''main''']=zKSt6
g32tV=dnWF9(gJ3aB)
