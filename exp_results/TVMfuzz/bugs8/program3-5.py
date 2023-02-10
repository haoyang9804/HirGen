import tvm.relay as relay
from tvm.relay.ty import TupleType
from tvm.relay.analysis import check_basic_block_normal_form
from functools import wraps
from tvm import te
from tvm.relay.data_dep_optimization import simplify_fc_transpose
from scipy import special
from tvm.relay.ty import GlobalTypeVar
from tvm.relay.ty import FuncType
import tvm.topi.testing
import itertools
from tvm.relay.ty import TypeVar
from tvm.testing import assert_allclose
from tvm.relay.ty import TypeRelation
from tvm.relay.build_module import bind_params_by_name
import json
from tvm.ir import structural_equal
import scipy
from tvm.relay.ty import TensorType
from tvm.relay.analysis import check_kind
from tvm import autotvm
from tvm import relay as rly
from numpy import isclose
import sys
from tvm.relay import analysis
from tvm.relay import transform
import logging
from tvm.relay import op
import tvm.relay.transform as _transform
from tvm.ir import IRModule
from tvm.relay.ty import IncompleteType
from tvm.relay.testing import Prelude
from tvm.relay import TypeVisitor
import tvm.relay.transform
import time
from tvm.relay.transform import FastMath
from tvm.relay.testing import count
from tvm.relay.analysis import Feature
import tvm
from tvm.relay.analysis import detect_feature
from tvm.contrib.nvcc import have_fp16
from tvm.relay.testing import make_nat_expr
from tvm.relay.testing import rand
from typing import Union
import pytest
import numpy as np
import math
from tvm.relay.testing import check_grad
from tvm.relay.testing import run_opt_pass
from tvm.relay.ty import RefType
from tvm.relay import TypeFunctor
from tvm.relay import expr as _expr
from tvm import relay
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.op.annotation import compiler_end
from tvm.relay.prelude import Prelude
from tvm.runtime import container
from tvm.contrib import graph_runtime
import os
from tvm import runtime
from tvm.relay.transform import SimplifyInference
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing.temp_op_attr import TempOpAttr
import random
from tvm.relay.testing import create_workload
from tvm.relay.testing.synthetic import get_workload
import scipy.sparse as sp
from tvm import nd
import tvm.relay.testing
from tvm.relay import TypeMutator
from tvm.relay.analysis import well_formed
from tvm.relay import testing
from tvm.relay.analysis import get_calibration_data
from tvm.relay.ty import TypeCall
from tvm.autotvm.tuner import RandomTuner
from tvm.relay import create_executor
from tvm.relay.transform import to_cps
from tvm.relay.testing import run_infer_type
from tvm.relay.transform import un_cps
from tvm.relay import ExprVisitor
from tvm import topi
from tvm.relay import Any
import tvm.testing
from tvm.relay.testing import enabled_targets
from tvm.relay.backend.interpreter import RefValue
from tvm.relay.adt import TypeData
from tvm.relay.op.annotation import compiler_begin



def run_infer_type(expr, mod=None):
    if (not mod):
        mod = tvm.IRModule.from_expr(expr)
        mod = transform.InferType()(mod)
        entry = mod['main']
        return (entry if isinstance(expr, relay.Function) else entry.body)
    else:
        if isinstance(expr, relay.GlobalVar):
            gv = expr.name_hint
        else:
            func = expr
            if (not isinstance(expr, relay.Function)):
                func = relay.Function(analysis.free_vars(expr), expr)
            mod['main'] = func
            gv = 'main'
        mod = transform.InferType()(mod)
        if isinstance(expr, (relay.GlobalVar, relay.Function)):
            return mod[gv]
        return mod[gv].body
JyiGj=relay.Var('''_''')
Ecoot=relay.const(2.0,'''float32''')
SfMwr=relay.Let(JyiGj,Ecoot,(JyiGj + JyiGj))
qqlbb=run_infer_type(SfMwr)

SEMVER = '#[version = "0.0.5"]\n'

BINARY_OPS = {'*': relay.multiply, '/': relay.divide, '+': relay.add, '-': relay.subtract, '<': relay.less, '>': relay.greater, '<=': relay.less_equal, '>=': relay.greater_equal, '==': relay.equal, '!=': relay.not_equal}

TYPES = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'bool', 'int8x4', 'uint1x4', 'float16x4'}

LIST_DEFN = '\ntype List[A] {\n    Cons(A, List[A]),\n    Nil,\n}\n'

int32 = relay.scalar_type('int32')

_ = relay.Var('_')

X = relay.Var('x')

Y = relay.Var('y')

X_ANNO = relay.Var('x', int32)

Y_ANNO = relay.Var('y', int32)

UNIT = relay.Tuple([])


def assert_graph_equal(lhs, rhs):
    tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)


def graph_equal(lhs, rhs):
    return tvm.ir.structural_equal(lhs, rhs, map_free_vars=True)


def roundtrip_expr(expr):
    text = tvm.relay.Expr.astext(expr, show_meta_data=False)
    x = tvm.parser.parse_expr(text)
    assert_graph_equal(x, expr)


def roundtrip(expr):
    x = tvm.parser.fromtext(expr.astext())
    assert_graph_equal(x, expr)


def parse_text(code):
    expr = tvm.parser.parse_expr(code)
    roundtrip_expr(expr)
    return expr


def parses_as(code, expr):
    parsed = parse_text(code)
    result = graph_equal(parsed, expr)
    return result


def parse_module(code):
    mod = tvm.parser.parse((SEMVER + code))
    roundtrip(mod)
    return mod


def assert_parses_as(code, expr):
    parsed = parse_text(code)
    assert_graph_equal(parsed, expr)


def assert_parse_module_as(code, mod):
    parsed = parse_module(code)
    assert_graph_equal(parsed, mod)


def get_scalar(x):
    return x.data.asnumpy().item()


def test_comments():
    assert_parses_as('\n        // This is a line comment!\n        ()\n        ', UNIT)
    assert_parses_as('\n        /* This is a block comment!\n            This is still a block comment!\n        */\n        ()\n        ', UNIT)
    assert_parses_as('\n        /* This is a block comment!\n           /*Block comment is recursive!*/\n        */\n        ()\n        ', UNIT)


def test_int_literal():
    assert isinstance(parse_text('1'), relay.Constant)
    assert isinstance(parse_text('1').data, tvm.nd.NDArray)
    assert (get_scalar(parse_text('1')) == 1)
    assert (get_scalar(parse_text('10')) == 10)
    assert (get_scalar(parse_text('0')) == 0)
    assert (get_scalar(parse_text('-100')) == (- 100))
    assert (get_scalar(parse_text('-05')) == (- 5))


def test_float_literal():
    assert (get_scalar(parse_text('1.0f')) == 1.0)
    assert isclose(get_scalar(parse_text('1.56667f')), 1.56667)
    assert (get_scalar(parse_text('0.0f')) == 0.0)
    assert (get_scalar(parse_text('-10.0f')) == (- 10.0))
    assert isclose(get_scalar(parse_text('1e-1f')), 0.1)
    assert (get_scalar(parse_text('1e+1f')) == 10.0)
    assert isclose(get_scalar(parse_text('1E-1f')), 0.1)
    assert (get_scalar(parse_text('1E+1f')) == 10.0)
    assert isclose(get_scalar(parse_text('1.0e-1f')), 0.1)
    assert (get_scalar(parse_text('1.0e+1f')) == 10.0)
    assert isclose(get_scalar(parse_text('1.0E-1f')), 0.1)
    assert (get_scalar(parse_text('1.0E+1f')) == 10.0)


def test_bool_literal():
    assert (get_scalar(parse_text('True')) == True)
    assert (get_scalar(parse_text('False')) == False)


def test_negative():
    assert (get_scalar(parse_text('--10')) == 10)
    assert (get_scalar(parse_text('---10')) == (- 10))


def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert_parses_as('1 {} 1'.format(bin_op), BINARY_OPS.get(bin_op)(relay.const(1), relay.const(1)))


def test_parens():
    assert graph_equal(parse_text('1 * 1 + 1'), parse_text('(1 * 1) + 1'))
    assert (not graph_equal(parse_text('1 * 1 + 1'), parse_text('1 * (1 + 1)')))


def test_op_assoc():
    assert graph_equal(parse_text('1 * 1 + 1 < 1 == 1'), parse_text('(((1 * 1) + 1) < 1) == 1'))
    assert graph_equal(parse_text('1 == 1 < 1 + 1 * 1'), parse_text('1 == (1 < (1 + (1 * 1)))'))


def test_vars():
    var = parse_text('let %foo = (); %foo')
    assert isinstance(var.body, relay.Var)
    assert (var.body.name_hint == 'foo')
    global_var = parse_text('@foo')
    assert isinstance(global_var, relay.GlobalVar)
    assert (global_var.name_hint == 'foo')
    op = parse_text('add')
    assert isinstance(op, tvm.ir.Op)
    assert (op.name == 'add')
    op = parse_text('nn.global_avg_pool2d')
    assert isinstance(op, tvm.ir.Op)
    assert (op.name == 'nn.global_avg_pool2d')


def test_meta_ref():
    with pytest.raises(tvm.error.DiagnosticError):
        meta_op = parse_text('meta[type_key][1337]')
        assert (meta_op.attrs.node_type_key == 'type_key')
        assert (meta_op.attrs.node_index == 1337)


def test_let():
    assert_parses_as('let %x = 1; ()', relay.Let(X, relay.const(1), UNIT))
    assert_parses_as('\n        let %x = 1;\n        let %y = 2;\n        ()\n        ', relay.Let(X, relay.const(1), relay.Let(Y, relay.const(2), UNIT)))


def test_seq():
    assert_parses_as('(); ()', relay.Let(_, UNIT, UNIT))
    assert_parses_as('let %_ = 1; ()', relay.Let(X, relay.const(1), UNIT))


def test_graph():
    code = '%0 = (); %1 = 1; (%0, %0, %1)'
    assert_parses_as(code, relay.Tuple([UNIT, UNIT, relay.const(1)]))


def test_graph_single():
    assert_parses_as('%1 = (); %1', relay.Tuple([]))


def test_let_global_var():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('let @x = 1; ()')


def test_let_op():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('let x = 1; ()')


def test_tuple():
    assert_parses_as('()', relay.Tuple([]))
    assert_parses_as('(0,)', relay.Tuple([relay.const(0)]))
    assert_parses_as('(0, 1)', relay.Tuple([relay.const(0), relay.const(1)]))
    assert_parses_as('(0, 1, 2)', relay.Tuple([relay.const(0), relay.const(1), relay.const(2)]))


def test_func():
    assert_parses_as('fn () { 0 }', relay.Function([], relay.const(0), None, []))
    assert_parses_as('fn (%x) { %x }', relay.Function([X], X, None, []))
    assert_parses_as('fn (%x, %y) { %x + %y }', relay.Function([X, Y], relay.add(X, Y), None, []))
    assert_parses_as('fn (%x: int32) -> int32 { %x }', relay.Function([X_ANNO], X_ANNO, int32, []))


def test_defn():
    id_defn = parse_module('\n        def @id(%x: int32) -> int32 {\n            %x\n        }\n        ')
    assert isinstance(id_defn, tvm.IRModule)


def test_recursive_call():
    id_defn = parse_module('\n        def @id(%x: int32) -> int32 {\n            @id(%x)\n        }\n        ')
    assert isinstance(id_defn, tvm.IRModule)


def test_ifelse():
    assert_parses_as('\n        if (True) {\n            0\n        } else {\n            1\n        }\n        ', relay.If(relay.const(True), relay.const(0), relay.const(1)))


def test_ifelse_scope():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('\n            if (True) {\n                let %x = ();\n                ()\n            } else {\n                %x\n            }\n            ')


def test_call():
    id_func = relay.Var('id')
    assert_parses_as('\n        let %id = fn (%x) { %x };\n        10 * %id(10)\n        ', relay.Let(id_func, relay.Function([X], X, None, []), relay.multiply(relay.const(10), relay.Call(id_func, [relay.const(10)]))))
    constant = relay.Var('constant')
    assert_parses_as('\n        let %constant = fn () { 0 };\n        %constant()\n        ', relay.Let(constant, relay.Function([], relay.const(0), None, []), relay.Call(constant, [], None, None)))
    id_var = relay.Var('id')
    assert_parses_as('\n        let %id = fn (%x) { %x };\n        %id(1)\n        ', relay.Let(id_var, relay.Function([X], X, None, []), relay.Call(id_var, [relay.const(1)], None, None)))
    multiply = relay.Var('multiply')
    assert_parses_as('\n        let %multiply = fn (%x, %y) { %x * %y };\n        %multiply(0, 0)\n        ', relay.Let(multiply, relay.Function([X, Y], relay.multiply(X, Y), None, []), relay.Call(multiply, [relay.const(0), relay.const(0)], None, None)))
    assert_parses_as('\n        (fn (%x) { %x })(0)\n        ', relay.Call(relay.Function([X], X, None, []), [relay.const(0)], None, None))
    curried_mult = relay.Var('curried_mult')
    assert_parses_as('\n        let %curried_mult =\n            fn (%x) {\n            fn (%y) {\n                %x * %y\n            }\n            };\n            %curried_mult(0);\n            %curried_mult(0)(0)\n        ', relay.Let(curried_mult, relay.Function([X], relay.Function([Y], relay.multiply(X, Y), None, []), None, []), relay.Let(_, relay.Call(curried_mult, [relay.const(0)], None, None), relay.Call(relay.Call(curried_mult, [relay.const(0)], None, None), [relay.const(0)], None, None))))
    assert_parses_as('abs(1)', relay.Call(relay.op.get('abs'), [relay.const(1)], None, None))


def test_incomplete_type():
    assert_parses_as('let %_ : _ = (); ()', relay.Let(_, UNIT, UNIT))


def test_builtin_types():
    for builtin_type in TYPES:
        parse_text('let %_ : {} = (); ()'.format(builtin_type))


def test_tensor_type():
    assert_parses_as('let %_ : Tensor[(), float32] = (); ()', relay.Let(relay.Var('_', relay.TensorType((), 'float32')), UNIT, UNIT))
    assert_parses_as('let %_ : Tensor[(1), float32] = (); ()', relay.Let(relay.Var('_', relay.TensorType((1,), 'float32')), UNIT, UNIT))
    assert_parses_as('let %_ : Tensor[(1, 1), float32] = (); ()', relay.Let(relay.Var('_', relay.TensorType((1, 1), 'float32')), UNIT, UNIT))
    assert_parses_as('let %_ : Tensor[(?, 1), float32] = (); ()', relay.Let(relay.Var('_', relay.TensorType((tvm.tir.Any(), 1), 'float32')), UNIT, UNIT))


def test_function_type():
    assert_parses_as('\n        let %_: fn () -> int32 = fn () -> int32 { 0 }; ()\n        ', relay.Let(relay.Var('_', relay.FuncType([], int32, [], [])), relay.Function([], relay.const(0), int32, []), UNIT))
    assert_parses_as('\n        let %_: fn (int32) -> int32 = fn (%x: int32) -> int32 { 0 }; ()\n        ', relay.Let(relay.Var('_', relay.FuncType([int32], int32, [], [])), relay.Function([relay.Var('x', int32)], relay.const(0), int32, []), UNIT))
    assert_parses_as('\n        let %_: fn (int32, int32) -> int32 = fn (%x: int32, %y: int32) -> int32 { 0 }; ()\n        ', relay.Let(relay.Var('_', relay.FuncType([int32, int32], int32, [], [])), relay.Function([relay.Var('x', int32), relay.Var('y', int32)], relay.const(0), int32, []), UNIT))


def test_tuple_type():
    assert_parses_as('\n        let %_: () = (); ()\n        ', relay.Let(relay.Var('_', relay.TupleType([])), UNIT, UNIT))
    assert_parses_as('\n        let %_: (int32,) = (0,); ()\n        ', relay.Let(relay.Var('_', relay.TupleType([int32])), relay.Tuple([relay.const(0)]), UNIT))
    assert_parses_as('\n        let %_: (int32, int32) = (0, 1); ()\n        ', relay.Let(relay.Var('_', relay.TupleType([int32, int32])), relay.Tuple([relay.const(0), relay.const(1)]), UNIT))


def test_adt_defn():
    mod = tvm.IRModule()
    glob_typ_var = relay.GlobalTypeVar('Ayy')
    prog = relay.TypeData(glob_typ_var, [], [relay.Constructor('Nil', [], glob_typ_var)])
    mod[glob_typ_var] = prog
    assert_parse_module_as('\n        type Ayy { Nil }\n        ', mod)


def test_adt_any():
    code = '\n    type my_dtype {\n        my_cons(Tensor[(?, 1), uint16]),\n    }\n    '
    mod = parse_module(code)
    items = mod.type_definitions.items()
    (global_type_var, type_data) = items[0]
    assert (global_type_var.name_hint == 'my_dtype')
    ctors = type_data.constructors
    assert (len(ctors) == 1)
    my_cons = ctors[0]
    assert (my_cons.name_hint == 'my_cons')
    ty_shape = my_cons.inputs[0].shape
    assert isinstance(ty_shape[0], tvm.tir.Any)
    assert (ty_shape[1] == 1)


def test_empty_adt_defn():
    mod = tvm.IRModule()
    glob_typ_var = relay.GlobalTypeVar('Ayy')
    prog = relay.TypeData(glob_typ_var, [], [])
    mod[glob_typ_var] = prog
    assert_parse_module_as('\n        type Ayy { }\n        ', mod)


def test_multiple_cons_defn():
    mod = tvm.IRModule()
    list_var = relay.GlobalTypeVar('List')
    typ_var = relay.TypeVar('A')
    prog = relay.TypeData(list_var, [typ_var], [relay.Constructor('Cons', [typ_var, list_var(typ_var)], list_var), relay.Constructor('Nil', [], list_var)])
    mod[list_var] = prog
    assert_parse_module_as(LIST_DEFN, mod)


def test_multiple_type_param_defn():
    glob_typ_var = relay.GlobalTypeVar('Either')
    typ_var_a = relay.TypeVar('A')
    typ_var_b = relay.TypeVar('B')
    prog = relay.TypeData(glob_typ_var, [typ_var_a, typ_var_b], [relay.Constructor('Left', [typ_var_a], glob_typ_var), relay.Constructor('Right', [typ_var_b], glob_typ_var)])
    mod = tvm.IRModule()
    mod[glob_typ_var] = prog
    assert_parse_module_as('\n        type Either[A, B] {\n          Left(A),\n          Right(B),\n        }\n        ', mod)


def test_match():
    match_keywords = [('match', True), ('match?', False)]
    for (match_keyword, is_complete) in match_keywords:
        mod = tvm.IRModule()
        list_var = relay.GlobalTypeVar('List')
        typ_var = relay.TypeVar('A')
        cons_constructor = relay.Constructor('Cons', [typ_var, list_var(typ_var)], list_var)
        nil_constructor = relay.Constructor('Nil', [], list_var)
        list_def = relay.TypeData(list_var, [typ_var], [cons_constructor, nil_constructor])
        mod[list_var] = list_def
        length_var = relay.GlobalVar('length')
        typ_var = relay.TypeVar('A')
        input_type = list_var(typ_var)
        input_var = relay.Var('xs', input_type)
        rest_var = relay.Var('rest')
        cons_case = relay.Let(relay.var('', type_annotation=None), UNIT, relay.add(relay.const(1), relay.Call(length_var, [rest_var])))
        body = relay.Match(input_var, [relay.Clause(relay.PatternConstructor(cons_constructor, [relay.PatternWildcard(), relay.PatternVar(rest_var)]), cons_case), relay.Clause(relay.PatternConstructor(nil_constructor, []), relay.const(0))], complete=is_complete)
        length_func = relay.Function([input_var], body, int32, [typ_var])
        mod[length_var] = length_func
        assert_parse_module_as(('\n            %s\n\n            def @length[A](%%xs: List[A]) -> int32 {\n              %s (%%xs) {\n                Cons(_, %%rest : List[A]) => {\n                  ();\n                  1 + @length(%%rest)\n                },\n                Nil => 0,\n              }\n            }\n            ' % (LIST_DEFN, match_keyword)), mod)


def test_adt_cons_expr():
    mod = tvm.IRModule()
    list_var = relay.GlobalTypeVar('List')
    typ_var = relay.TypeVar('A')
    cons_constructor = relay.Constructor('Cons', [typ_var, list_var(typ_var)], list_var)
    nil_constructor = relay.Constructor('Nil', [], list_var)
    list_def = relay.TypeData(list_var, [typ_var], [cons_constructor, nil_constructor])
    mod[list_var] = list_def
    make_singleton_var = relay.GlobalVar('make_singleton')
    input_var = relay.Var('x', int32)
    make_singleton_func = relay.Function([input_var], cons_constructor(input_var, nil_constructor()), list_var(int32))
    mod[make_singleton_var] = make_singleton_func
    assert_parse_module_as(('\n        %s\n\n        def @make_singleton(%%x: int32) -> List[int32] {\n          Cons(%%x, Nil)\n        }\n        ' % LIST_DEFN), mod)


def test_duplicate_adt_defn():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_module(('\n            %s\n\n            type List[A] {\n            Cons(A, List[A]),\n            Nil,\n            }\n            ' % LIST_DEFN))


def test_duplicate_adt_cons():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('\n            type Ayy { Lmao }\n            type Haha { Lmao }\n            ')


def test_duplicate_adt_cons_defn():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('\n            type Ayy { Lmao }\n            type Lmao { Ayy }\n            ')


def test_duplicate_global_var():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text('\n            def @id[A](%x: A) -> A { x }\n            def @id[A](%x: A) -> A { x }\n            ')


def test_extern_adt_defn():
    mod = tvm.IRModule()
    extern_var = relay.GlobalTypeVar('T')
    typ_var = relay.TypeVar('A')
    extern_def = relay.TypeData(extern_var, [typ_var], [])
    mod[extern_var] = extern_def
    assert_parse_module_as('\n        extern type T[A]\n        ', mod)


def test_import_grad():
    mod = tvm.IRModule()
    mod.import_from_std('gradient.rly')


def test_resnet():
    (mod, _) = relay.testing.resnet.get_workload()
    text = mod.astext()
    parsed_mod = tvm.parser.parse(text)
    tvm.ir.assert_structural_equal(mod, parsed_mod)


def inline_params(mod, params):
    main_fn = mod['main']
    str_to_var = {}
    for param in main_fn.params:
        str_to_var[param.name_hint] = param
    bind_map = {}
    for param in params:
        bind_map[str_to_var[param]] = relay.const(params[param])
    body = relay.bind(main_fn.body, bind_map)
    main_fn = relay.Function(relay.analysis.free_vars(body), body)
    mod['main_fn'] = main_fn
    return mod
rOOBz=relay.testing.resnet.get_workload()
zvKW8=inline_params(rOOBz[0],rOOBz[1])
tbplk=relay.TypeVar('''tv''')
Igd6o=relay.TupleType([tbplk,tbplk,])
tvm.ir.assert_structural_equal(zvKW8,Igd6o)
