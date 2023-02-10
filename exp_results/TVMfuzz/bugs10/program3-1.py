from tvm.relay.analysis import check_kind
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.analysis import well_formed
from tvm.relay.ty import TypeCall
from tvm.relay.testing import make_nat_expr
from tvm.runtime import container
from functools import wraps
from tvm.ir import structural_equal
from tvm.relay.transform import SimplifyInference
import random
from tvm.relay.backend.interpreter import RefValue
import os
from tvm.relay.testing.synthetic import get_workload
from tvm.relay import create_executor
from tvm import runtime
from tvm import relay as rly
from numpy import isclose
from tvm.relay.transform import un_cps
from tvm.relay.prelude import Prelude
from tvm.relay import testing
from tvm.relay.transform import FastMath
from tvm import te
from tvm.relay.ty import FuncType
from tvm.ir import IRModule
import tvm.topi.testing
from tvm.relay.testing import create_workload
from tvm.relay.ty import TypeRelation
from tvm.relay.analysis import get_calibration_data
from tvm.contrib.nvcc import have_fp16
import tvm.relay as relay
from tvm.relay import TypeVisitor
from tvm.relay.transform import to_cps
from tvm.relay.testing import Prelude
from tvm.relay import Any
from tvm import topi
import tvm.relay.transform as _transform
import tvm.relay.transform
from tvm.relay.testing import run_opt_pass
import sys
from tvm.relay.testing import check_grad
from tvm.relay.ty import TupleType
import tvm.relay.testing
from tvm.relay import op
from tvm.relay import expr as _expr
from tvm.relay import TypeMutator
import tvm
from tvm.relay.testing import enabled_targets
from tvm.relay.testing import run_infer_type
from tvm.relay.op.annotation import compiler_end
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import ExprVisitor
from tvm.contrib import graph_runtime
from tvm import autotvm
from tvm import relay
from tvm.relay.ty import RefType
import math
from tvm.relay.analysis import Feature
import pytest
import scipy
from tvm.relay.analysis import check_basic_block_normal_form
from tvm.relay import analysis
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.ty import IncompleteType
from tvm.relay.ty import TensorType
from tvm.relay.analysis import detect_feature
import json
from tvm.relay.op.annotation import compiler_begin
from tvm.relay.testing.temp_op_attr import TempOpAttr
import time
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from tvm.relay.ty import TypeVar
import tvm.testing
import scipy.sparse as sp
from tvm.autotvm.tuner import RandomTuner
import numpy as np
from tvm.relay import transform
from typing import Union
from tvm import nd
from tvm.relay.testing import count
from tvm.relay.adt import TypeData
from tvm.testing import assert_allclose
import itertools
from tvm.relay.ty import GlobalTypeVar
from scipy import special
import logging
from tvm.relay import TypeFunctor
from tvm.relay.testing import rand

j7pJr=tvm.IRModule()
VYpL6=relay.var('''y''',shape=(1,4,),dtype='''uint1''')
UJtXY=relay.const(0.91768,'''int32''')
evMiT=relay.Function([VYpL6,VYpL6,],UJtXY)
j7pJr['''main''']=evMiT
j7pJr['''main''']=evMiT
AZ7f8=Prelude(j7pJr)
g2MaV=make_nat_expr(AZ7f8,3)
HWBNS=AZ7f8.nat_iterate(evMiT,g2MaV)
urMg5=relay.TensorType([],'''uint32''')
QHkBG=relay.var('''y''',urMg5)
PY5nC=HWBNS(QHkBG)
