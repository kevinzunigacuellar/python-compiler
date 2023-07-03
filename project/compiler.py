from typing import Set, Dict, Tuple
import sys
import traceback
from dataclasses import dataclass

from cs202_support.python import *
import cs202_support.x86 as x86
import constants
import cfun
import print_x86defs
from cs202_support import ast_pretty_printer as ast_printer

from interference_graph import InterferenceGraph

comparisons = ['eq', 'gt', 'gte', 'lt', 'lte']
gensym_num = 0
global_logging = False

global_values = ['free_ptr', 'fromspace_end']

tuple_var_types = {}
function_names = set()


def log(label, value):
    if global_logging:
        print()
        print(f'--------------------------------------------------')
        print(f'Logging: {label}')
        print(value)
        print(f'--------------------------------------------------')


def log_ast(label, value):
    log(label, print_ast(value))


def gensym(x):
    """
    Constructs a new variable name guaranteed to be unique.
    :param x: A "base" variable name (e.g. "x")
    :return: A unique variable name (e.g. "x_1")
    """

    global gensym_num
    gensym_num = gensym_num + 1
    return f'{x}_{gensym_num}'


##################################################
# typecheck
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr]) | Begin(Stmts, Expr)
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

@dataclass
class Callable:
    args: List[type]
    output_type: type


Type_Env = Dict[str, Callable | Tuple | type]


def typecheck(prog: Program) -> Program:
    """
    Typechecks the input program; throws an error if the program is not well-typed.
    :param prog: The Ltup program to typecheck
    :return: The program, if it is well-typed
    """

    def tc_expr(e: Expr, env: Type_Env) -> type | tuple:
        """
        Supports
            Expr ::= Var(x) | Constant(n) | Prim(op, List[Expr]) | Begin(Stmts, Expr)
            op   ::= "add" | "sub" | "mult" |
                     "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte" |
                     "tuple" | "subscript"
        :param e: Expression to typecheck
        :param env: Dictionary of previously initialized variables and their types
        :return: The type of result of the given expression
        """
        match e:
            case Call(func_name, args):
                function_type = tc_expr(func_name, env)
                assert (isinstance(function_type, Callable))

                param_types = function_type.args
                for i, arg in enumerate(args):
                    assert (param_types[i] == tc_expr(arg, env))

                return function_type.output_type

            case Var(var):
                if var in global_values:
                    return int
                else:
                    return env[var]

            case Constant(n):
                if isinstance(n, bool) or isinstance(n, int):
                    return type(n)
                else:
                    raise Exception('tc_expr, Constant', e)

            case Begin(stmts, expr_args):
                tc_stmts(stmts, env)
                return tc_expr(expr_args, env)

            case Prim(op, expr_args):
                match op:
                    case "add" | "sub" | "mult":
                        assert tc_expr(expr_args[0], env) == int
                        assert tc_expr(expr_args[1], env) == int
                        return int

                    case "not":
                        assert tc_expr(expr_args[0], env) == bool
                        return bool

                    case "or" | "and":
                        assert tc_expr(expr_args[0], env) == bool
                        assert tc_expr(expr_args[1], env) == bool
                        return bool

                    case "eq" | "gt" | "gte" | "lt" | "lte":
                        assert tc_expr(expr_args[0], env) == tc_expr(
                            expr_args[1], env)
                        return bool

                    case "tuple":
                        types = []
                        for arg in expr_args:
                            types.append(tc_expr(arg, env))

                        return tuple(types)

                    case "subscript":  # TODO: args are [e1, Constant(idx)]
                        tuple_to_subscript = expr_args[0]
                        subscript = expr_args[1]

                        assert (type(subscript) == Constant)

                        tuple_types = tc_expr(tuple_to_subscript, env)
                        assert (isinstance(tuple_types, tuple))

                        return tuple_types[subscript.val]

                    case _:
                        raise Exception("tc_exp, prim", e)

            case _:
                raise Exception("tc_expr", e)

    def tc_stmt(stmt: Stmt, env: Type_Env):
        """
        Typechecks a statement
        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
        :param stmt: The statement to typecheck
        :param env: Dictionary of previously initialized variables and their types
        """
        match stmt:
            case FunctionDef(name, params, body_stmts, return_type):
                param_types = []
                for param_name, param_type in params:
                    param_types.append(param_type)

                env[name] = Callable(param_types, return_type)

                function_names.add(name)

                new_env = env.copy()

                for param_name, param_type in params:
                    new_env[param_name] = param_type

                new_env["retval"] = return_type

                tc_stmts(body_stmts, new_env)

                # TODO
                for name, varType in new_env.items():
                    tuple_var_types[name] = varType

            case Return(e):
                assert (tc_expr(e, env) == env["retval"])

            case Assign(x, expr):
                if x in env:
                    assert env[x] == tc_expr(expr, env)
                else:
                    env[x] = tc_expr(expr, env)

            case Print(expr):
                tc_expr(expr, env)

            case If(condition_expr, if_stmts, else_stmts):
                assert (tc_expr(condition_expr, env) == bool)
                for s in if_stmts:
                    tc_stmt(s, env)
                for s in else_stmts:
                    tc_stmt(s, env)

            case While(expr, stmts):
                assert (tc_expr(expr, env) == bool)
                for stmt in stmts:
                    tc_stmt(stmt, env)

            case _:
                raise Exception("tc_stmt", stmt)

    def tc_stmts(stmts: List[Stmt], env: Type_Env):
        """
        Typechecks a list of statements
        :param stmts: The list of statements to typecheck
        :param env:
        """

        for stmt in stmts:
            tc_stmt(stmt, env)

        for var in env:
            if isinstance(env[var], tuple):
                tuple_var_types[var] = env[var]

    environment = {}
    tc_stmts(prog.stmts, environment)
    return prog

##################################################
# compress-parameters
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr]) | Begin(Stmts, Expr)
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)


def compress_parameters(prog: Program) -> Program:
    compress_length = len(constants.argument_registers)

    def cp_function_def(function: FunctionDef) -> FunctionDef:
        compressed_param_name = "rest"

        params = function.params

        non_compressed_params = tuple(params[0:compress_length-1])
        compressed_params = tuple(params[compress_length-1:])

        # Add non-compressed parameters unchanged
        new_params = []
        for param in non_compressed_params:
            new_params.append(param)

        # Create the tuple of types for the compressed parameters
        compressed_type = []
        for param_name, param_type in compressed_params:
            compressed_type.append(param_type)
        new_params.append((compressed_param_name, tuple(compressed_type)))

        # Create the new body stmts
        new_body_stmts = function.body
        for index, param in enumerate(compressed_params):
            param_name = param[0]
            param_type = param[1]

            decompress_stmt = Assign(param_name, Prim(
                "subscript", [Var(compressed_param_name), Constant(index)]))
            new_body_stmts.insert(0, decompress_stmt)

        return FunctionDef(function.name, new_params, new_body_stmts, function.return_type)

    def cp_function_call(call: Call):
        args = call.args

        uncompressed_args = args[0:compress_length-1]
        compressed_args = args[compress_length-1:]

        new_args = []
        for arg in uncompressed_args:
            new_args.append(arg)

        new_args.append(Prim("tuple", compressed_args))

        return new_args

    def cp_expr(expr: Expr) -> Expr:
        match expr:
            case Call(func_var, args):
                call = Call(func_var, args)
                if len(args) > compress_length:
                    new_args = cp_function_call(call)
                else:
                    new_args = args

                return Call(func_var, new_args)

            case Prim(op, args):
                new_args = []
                for arg in args:
                    new_args.append(cp_expr(arg))
                return Prim(op, new_args)
            case Begin(stmts, expr):
                return Begin(stmts, cp_expr(expr))
            case _:
                return expr

    def cp_stmt(stmt: Stmt) -> Stmt:
        """

        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) |
                     If(Expr, Stmts, Stmts) | While(Expr, Stmts) | Return(Expr) |
                     FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
        :param stmt:
        :return:
        """
        match stmt:
            # Compress the parameters in the function definition
            case FunctionDef(name, params, body_stmts, return_type):
                # Prevents IDE type error
                func_def = FunctionDef(name, params, body_stmts, return_type)

                if len(params) > compress_length:
                    new_func_def = cp_function_def(func_def)
                else:
                    new_func_def = func_def

                return new_func_def

            # Compress parameters in any calls to functions within these statements
            case Assign(var, expr):
                return Assign(var, cp_expr(expr))

            case Print(expr):
                return Print(cp_expr(expr))

            case If(condition_expr, if_stmts, else_stmts):
                new_if_stmts = cp_stmts(if_stmts)
                new_else_stmts = cp_stmts(else_stmts)
                return If(cp_expr(condition_expr), new_if_stmts, new_else_stmts)

            case While(condition_expr, body_stmts):
                new_body_stmts = cp_stmts(body_stmts)
                return While(cp_expr(condition_expr), new_body_stmts)

            case Return(expr):
                return Return(cp_expr(expr))

    def cp_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []
        for stmt in stmts:
            new_stmts.append(cp_stmt(stmt))

        return new_stmts

    return Program(cp_stmts(prog.stmts))


##################################################
# remove-complex-opera*
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)


def rco(prog: Program) -> Program:
    """
    Removes complex operands. After this pass, the arguments to operators (unary and binary
    operators, and function calls like "print") will be atomic.
    :param prog: An Ltup program
    :return: An Ltup program with atomic operator arguments.
    """

    def rco_expr(e: Expr, bindings: Dict[str, Expr]) -> Expr:
        """
        Converts non-atomic expressions into atomic ones
        Supports
            Expr ::= Var(x) | Constant(n) | Prim(op, List[Expr])
            op   ::= "add" | "sub" | "mult" |
                     "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte" |
                     "tuple" | "subscript"
        :param e: Expression to be converted
        :param bindings: List of variable assignments necessary to make all expressions atomic
        :return: An atomic version of e. If e is atomic already, return unchanged.
        """
        match e:
            case Constant(n):
                return Constant(n)

            case Var(x):
                if x in function_names:
                    new_tmp_var_name = gensym("tmp")
                    bindings[new_tmp_var_name] = Var(x)
                    return Var(new_tmp_var_name)

                return Var(x)

            case Call(func_var, args):
                new_args = []
                for arg in args:
                    new_args.append(rco_expr(arg, bindings))

                new_function = rco_expr(func_var, bindings)

                new_call = Call(new_function, new_args)

                new_label = gensym("tmp")
                bindings[new_label] = new_call

                return Var(new_label)

            case Prim(op, args):
                match op:
                    # case "tuple":
                    #     pass
                    # case "subscript":
                    #     pass
                    case _:
                        # The recursive call to rco_exp should make the arguments atomic
                        new_args = []
                        for arg in args:
                            new_args.append(rco_expr(arg, bindings))

                        tmp = gensym("tmp")

                        # Bind tmp to Prim(op, new_args)
                        bindings[tmp] = Prim(op, new_args)

                        # Return the variable
                        return Var(tmp)

            case _:
                raise Exception("rco_exr", e)

    def rco_stmt(s: Stmt, bindings: Dict[str, Expr]) -> Stmt:
        """
        Converts all non-atomic expressions in a statement to atomic ones
        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
        :param s: Statement to be converted
        :param bindings: List of variable assignments necessary to make all expressions atomic
        :return: An atomic version of s
        """
        match s:
            case FunctionDef(name, params, body_stmts, return_type):
                return FunctionDef(name, params, rco_stmts(body_stmts), return_type)

            case Return(e):
                return Return(rco_expr(e, bindings))

            case Assign(x, expr):
                return Assign(x, rco_expr(expr, bindings))

            case Print(expr):
                return Print(rco_expr(expr, bindings))

            case If(condition_expr, then_stmts, else_stmts):
                return If(rco_expr(condition_expr, bindings),
                          rco_stmts(then_stmts),
                          rco_stmts(else_stmts))

            case While(condition, body_stmts):
                new_body_stmts = rco_stmts(body_stmts)

                condition_bindings = {}
                new_condition_expr = rco_expr(condition, condition_bindings)

                condition_stmts = []
                for var, binding in condition_bindings.items():
                    condition_stmts.append(Assign(var, binding))

                return While(Begin(condition_stmts, new_condition_expr), new_body_stmts)

            case _:
                raise Exception("rco_stmt", s)

    def rco_stmts(stmts: List[Stmt]) -> List[Stmt]:
        """
        Converts a list of statements into ones with no non-atomic expressions
        :param stmts: Statements to be converted
        :return: A list of statements that are equivalent to stmts, but with no non-atomic expressions
        """
        new_stmts = []

        for stmt in stmts:
            bindings = {}
            new_stmt = rco_stmt(stmt, bindings)

            # Turn each binding into an assignment statement (x -> e => Assign(x, e))
            # Add each binding assignment statement to new_stmts
            for name, value in bindings.items():
                new_stmts.append(Assign(name, value))

            new_stmts.append(new_stmt)

        return new_stmts

    return Program(rco_stmts(prog.stmts))

##################################################
# expose-allocation
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)


def expose_alloc(prog: Program) -> Program:
    """
    Exposes allocations in an Ltup program. Replaces tuple(...) with explicit
    allocation.
    :param prog: An Ltup program
    :return: An Ltup program, without Tuple constructors
    """

    def create_tag(types: tuple):
        """
        Create the tag for a tuple
        :param types: A tuple containing the types of every element in the tuple
        """
        tag = 0
        # Construct pointer mask
        for element_type in reversed(types):
            tag = tag << 1
            if isinstance(element_type, tuple):
                tag += 1
            else:
                tag += 0

        # Construct the length
        tag = tag << 6
        tag += len(types)  # Can't handle lengths greater than 50

        # Add the forwarding pointer indicator
        tag = tag << 1
        tag += 1

        return tag

    def ea_stmt(stmt: Stmt) -> List[Stmt]:
        match stmt:
            case FunctionDef(name, params, body_stmts, return_type):
                return [FunctionDef(name, params, ea_stmts(body_stmts), return_type)]

            case Assign(var, Prim("tuple", args)):
                # Collect #
                # Check if the free_pointer plus the needed number of bytes is less than the end of fromspace
                #   If it isn't, call the garbage collector

                bytes_needed = (len(args) * 8) + 8

                new_freeptr_name = gensym('tmp')
                less_than_var_name = gensym("tmp")

                all_stmts = [
                    Assign(new_freeptr_name, Prim(
                        "add", [Var("free_ptr"), Constant(bytes_needed)])),
                    Assign(less_than_var_name, Prim(
                        "lt", [Var(new_freeptr_name), Var("fromspace_end")])),
                    If(Var(less_than_var_name),
                       [],
                       [Assign("_", Prim("collect", [Constant(bytes_needed)]))])
                ]

                # Allocate #
                tag = create_tag(tuple_var_types[var])
                all_stmts.append(
                    Assign(var, Prim("allocate", [
                           Constant(bytes_needed), Constant(tag)]))
                )

                # Set contents #
                for i, arg in enumerate(args):
                    all_stmts.append(
                        Assign("_", Prim("tuple_set", [Var(var), Constant(i), arg])))

                return all_stmts

            case If(condition_expr, then_stmts, else_stmts):
                return [If(condition_expr, ea_stmts(then_stmts), ea_stmts(else_stmts))]

            case While(Begin(condition_stmts, condition_expr), body_stmts):
                return [While(Begin(ea_stmts(condition_stmts), condition_expr), ea_stmts(body_stmts))]

            case _:
                return [stmt]

    def ea_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []
        for stmt in stmts:
            new_stmts.extend(ea_stmt(stmt))

        return new_stmts

    # print(ast_printer.print_ast(Program(ea_stmts(prog.stmts))))
    return Program(ea_stmts(prog.stmts))


##################################################
# explicate-control
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "subscript" | "allocate" | "collect" | "tuple_set"
# Atm    ::= Var(x) | Constant(n)
# Expr   ::= Atm | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

def explicate_control(prog: Program) -> cfun.CProgram:
    """
    Transforms an Ltup Expression into a cfun program.
    :param prog: An Ltup Expression
    :return: A cfun Program
    """

    # the basic blocks of the program
    basic_blocks: Dict[str, List[cfun.Stmt]] = {}
    functions: List[cfun.CFunctionDef] = []
    current_function = 'main'

    def create_block(stmts: List[cfun.Stmt]) -> str:
        """
        Creates a new block with a generic, unique label
        :param stmts: The stmts to be added to the block
        :return: The created block's label
        """
        new_label = (current_function + gensym("label"))
        basic_blocks[new_label] = stmts
        return new_label

    def ec_atm(e: Expr) -> cfun.Atm:
        """
        Converts an Lif atomic expression into a Cif atomic
        Supports
            Atm ::= Constant(n) | Var(x)
        :param e: The atomic expression to be converted
        :return: The Cif version of e
        """
        match e:
            case Constant(n):
                return cfun.Constant(n)

            case Var(x):
                return cfun.Var(x)

            case _:
                raise Exception("ec_atm", e)

    def ec_expr(e: Expr) -> cfun.Expr:
        """
        Converts an Lfun expression into a Cfun expression
        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
            Expr ::= Atm | Prim(op, List[Expr])
            op   ::= "add" | "sub" | "mult" |
                     "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte" |
                     "subscript" | "allocate" | "collect" | "tuple_set"
            Atm  ::= Constant(n) | Var(x)
        :param e: The expression to be converted
        :return: The Cif version of e
        """
        match e:
            case Call(function, args):
                # TODO: similar to Prim, use cfun.Call
                new_args = []
                for arg in args:
                    new_args.append(ec_atm(arg))

                new_function = ec_atm(function)

                return cfun.Call(new_function, new_args)

            case Prim(op, expr_args):
                new_args = []
                for arg in expr_args:
                    new_args.append(ec_atm(arg))

                return cfun.Prim(op, new_args)

            case Constant() | Var():
                return ec_atm(e)

            case _:
                raise Exception("ec_expr", e)

    def ec_stmt(s: Stmt, continuation: List[cfun.Stmt]) -> List[cfun.Stmt]:
        """
        Converts a statement and its continuation into an equivalent list of CIf statements
        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
        :param s: The statement to be converted
        :param continuation: The statements that are done after s
        :return: A list of Cif statements equivalent to s and its continuation
        """

        match s:
            case Return(e):
                new_expr = ec_expr(e)
                new_return = [cfun.Return(new_expr)]
                return new_return

            # TODO?
            case FunctionDef(name, params, body_stmts, return_type):
                ec_function(name, params, body_stmts)
                return continuation

            case Assign(x, e):
                new_stmts: List[cfun.Stmt] = [cfun.Assign(x, ec_expr(e))]
                return new_stmts + continuation

            case Print(e):
                new_stmts: List[cfun.Stmt] = [cfun.Print(ec_expr(e))]
                return new_stmts + continuation

            case If(condition, then_stmts, else_stmts):
                continuation_label = create_block(continuation)

                then_label = create_block(
                    ec_stmts(then_stmts, [cfun.Goto(continuation_label)]))

                else_label = create_block(
                    ec_stmts(else_stmts, [cfun.Goto(continuation_label)]))

                return [cfun.If(ec_expr(condition), cfun.Goto(then_label), cfun.Goto(else_label))]

            case While(Begin(condition_stmts, condition_expr), body_stmts):
                continuation_label = create_block(continuation)

                test_label = gensym("loop_label")

                body_label = create_block(
                    ec_stmts(body_stmts, [cfun.Goto(test_label)]))

                new_continuation_stmts = [cfun.If(ec_expr(condition_expr),
                                                  cfun.Goto(body_label),
                                                  cfun.Goto(continuation_label))]

                new_condition_stmts = ec_stmts(
                    condition_stmts, new_continuation_stmts)

                basic_blocks[test_label] = new_condition_stmts

                return [cfun.Goto(test_label)]

            case _:
                raise Exception("ec_stmt", s)

    def ec_stmts(stmts: List[Stmt], continuation: List[cfun.Stmt]) -> List[cfun.Stmt]:
        """
        Converts a list of Lif statements into an equivalent list of Cif statements
        :param stmts: The statements to be converted
        :param continuation: The code to be run after the current set of statements
        :return: A list of Cif statements equivalent to stmts
        """
        for s in reversed(stmts):
            continuation = ec_stmt(s, continuation)

        return continuation

    # TODO
    def ec_function(name: str, params: List[Tuple[str, type]], body_stmts: List[Stmt]):
        # Prevents python from creating new shadowed variables
        nonlocal basic_blocks
        nonlocal current_function

        # Save `basic_blocks` and `current_function` so that we can restore them at the end
        old_basic_blocks = basic_blocks
        old_current_function = current_function

        # Set the new basic_blocks and name
        basic_blocks = {}
        current_function = name

        # TODO: adding return 0 where it shouldn't be
        new_stmts = ec_stmts(body_stmts, continuation=[
                             cfun.Return(cfun.Constant(0))])

        basic_blocks[(name + "start")] = new_stmts

        # Create the new function
        param_names = []
        for param_name, param_type in params:
            param_names.append(param_name)

        new_function = cfun.CFunctionDef(name, param_names, basic_blocks)

        functions.append(new_function)

        # Restore `basic_blocks` and `current_function`
        basic_blocks = old_basic_blocks
        current_function = old_current_function

    new_body = ec_stmts(prog.stmts, continuation=[
                        cfun.Return(cfun.Constant(0))])

    basic_blocks["mainstart"] = new_body

    functions.append(cfun.CFunctionDef("main", [], basic_blocks))

    # print(ast_printer.print_ast(cfun.CProgram(functions)))
    return cfun.CProgram(functions)


##################################################
# select-instructions
##################################################


# Expr         ::= Atm | Prim(op, List[Expr])

# Stmts        ::= List[Stmt]
# CFunctionDef ::= CFunctionDef(name, List[str], Dict[label, Stmts])
# Cfun         ::= CProgram(List[CFunctionDef])

@dataclass(frozen=True, eq=True)
class X86FunctionDef(AST):
    label: str
    blocks: Dict[str, List[x86.Instr]]
    stack_space: Tuple[int, int]


@dataclass(frozen=True, eq=True)
class X86ProgramDefs(AST):
    defs: List[X86FunctionDef]


def select_instructions(prog: cfun.CProgram) -> X86ProgramDefs:
    """
    Transforms a Ltup program into a pseudo-x86 assembly program.
    :param prog: a Ltup program
    :return: a pseudo-x86 program
    """

    current_function_name = 'main'

    op_instructions = {
        "add": "addq",
        "sub": "subq",
        "mult": "imulq",
        "or": "orq",
        "and": "andq"
    }

    cmp_codes = {
        "eq": "e",
        "gt": "g",
        "gte": "ge",
        "lt": "l",
        "lte": "le"
    }

    def si_atm(atm: cfun.Atm) -> x86.Arg:
        """
        Converts an LIf atomic in an x86 atomic
        Supports
            Atm ::= Constant(n) | Var(x)
        :param atm: The atomic to be converted
        :return: The x86 equivalent to atm
        """
        match atm:
            case cfun.Constant(i):
                return x86.Immediate(int(i))

            case cfun.Var(x):
                # TODO: change to match statement if possible
                if x in global_values:
                    return x86.GlobalVal(x)
                elif x in function_names:
                    pass
                else:
                    return x86.Var(x)

            case _:
                raise Exception("si_atm", atm)

    def si_stmt(stmt: cfun.Stmt) -> List[x86.Instr]:
        """
        Converts a Cif statement into one or more x86 instructions
        Supports
            Stmt ::= Assign(x, Expr) | Print(Expr) |
                     If(Expr, Goto(label), Goto(label)) | Goto(label) | Return(Expr)
            Expr ::= Atm | Prim(op, List[Expr])
            op   ::= "add" | "sub" | "mult" |
                     "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte" |
                     "subscript" | "allocate" | "collect" | "tuple_set"
        :param stmt: The statement to be converted
        :return: A list of pseudo-x86 statements equivalent to stmt
        """
        match stmt:
            case cfun.Assign(var, expr):
                match expr:
                    case cfun.Call(function_var, args):
                        new_instrs = []

                        available_arg_registers = constants.argument_registers.copy()

                        # Moves the arguments back into the parameter registers
                        arg_instrs: List[x86.Instr] = []
                        for arg in args:
                            arg_register = x86.Reg(
                                available_arg_registers.pop(0))
                            arg_instrs.extend(
                                [x86.NamedInstr("movq", [si_atm(arg), arg_register])])

                        new_instrs.extend(arg_instrs)

                        # Calls the function and moves its return value into the variable being assigned
                        new_instrs.extend([x86.IndirectCallq(si_atm(function_var), 0),
                                           x86.NamedInstr("movq", [x86.Reg("rax"), x86.Var(var)])])

                        return new_instrs

                    case cfun.Prim(op, expr_args):
                        match op:
                            case "add" | "sub" | "mult" | "or" | "and":
                                return [x86.NamedInstr("movq", [si_atm(expr_args[0]), x86.Reg("rax")]),
                                        x86.NamedInstr(op_instructions[op], [
                                                       si_atm(expr_args[1]), x86.Reg("rax")]),
                                        x86.NamedInstr("movq", [x86.Reg("rax"), x86.Var(var)])]

                            case "not":
                                return [x86.NamedInstr("movq", [si_atm(expr_args[0]), x86.Var(var)]),
                                        x86.NamedInstr("xorq", [x86.Immediate(1), x86.Var(var)])]

                            case "eq" | "gt" | "gte" | "lt" | "lte":
                                return [x86.NamedInstr("cmpq", [si_atm(expr_args[1]), si_atm(expr_args[0])]),
                                        x86.Set(cmp_codes[op],
                                                x86.ByteReg("al")),
                                        x86.NamedInstr("movzbq", [x86.ByteReg("al"), si_atm(cfun.Var(var))])]

                            case "subscript":
                                tuple_var = expr_args[0].var
                                offset_bytes = (expr_args[1].val * 8) + 8
                                return [x86.NamedInstr("movq", [x86.Var(tuple_var), x86.Reg("r11")]),
                                        x86.NamedInstr("movq", [x86.Deref("r11", offset_bytes), x86.Var(var)])]

                            case "allocate":
                                bytes_to_allocate = expr_args[0].val
                                tag = expr_args[1].val
                                return [x86.NamedInstr("movq", [x86.GlobalVal("free_ptr"), x86.Var(var)]),
                                        x86.NamedInstr("addq", [x86.Immediate(bytes_to_allocate),
                                                                x86.GlobalVal("free_ptr")]),
                                        x86.NamedInstr(
                                            "movq", [x86.Var(var), x86.Reg("r11")]),
                                        x86.NamedInstr("movq", [x86.Immediate(tag), x86.Deref("r11", 0)])]

                            case "collect":
                                bytes_to_collect = expr_args[0].val
                                return [x86.NamedInstr("movq", [x86.Reg("r15"), x86.Reg("rdi")]),
                                        x86.NamedInstr("movq", [x86.Immediate(
                                            bytes_to_collect), x86.Reg("rsi")]),
                                        x86.Callq("collect")]

                            case "tuple_set":
                                tuple_var = expr_args[0].var
                                offset_bytes = (expr_args[1].val * 8) + 8
                                value_to_add = expr_args[2]
                                return [x86.NamedInstr("movq", [x86.Var(tuple_var), x86.Reg("r11")]),
                                        x86.NamedInstr("movq", [si_atm(value_to_add), x86.Deref("r11", offset_bytes)])]

                            case _:
                                raise Exception("si_stmt, assign", stmt)

                    case atm:
                        # If it is an assignment to a function
                        if (type(atm) == cfun.Var) and (atm.var in function_names):
                            return [x86.NamedInstr("leaq", [x86.GlobalVal(atm.var), x86.Var(var)])]
                        else:
                            return [x86.NamedInstr("movq", [si_atm(atm), x86.Var(var)])]

            case cfun.If(condition, cfun.Goto(label1), cfun.Goto(label2)):
                return [x86.NamedInstr("cmpq", [si_atm(condition), x86.Immediate(1)]),
                        x86.JmpIf("e", label1),
                        x86.Jmp(label2)]

            case cfun.Print(a1):
                return [x86.NamedInstr("movq", [si_atm(a1), x86.Reg("rdi")]),
                        x86.Callq("print_int")]

            case cfun.Goto(label):
                return [x86.Jmp(label)]

            case cfun.Return(a1):
                return [x86.NamedInstr("movq", [si_atm(a1), x86.Reg("rax")]),
                        x86.Jmp(current_function_name+"conclusion")]

            case _:
                raise Exception("si_stmt", stmt)

    def si_stmts(stmts: List[cfun.Stmt]) -> List[x86.Instr]:
        """
        Converts a list of Cif statements into equivalent pseudo-x86 statements
        :param stmts: The statements to be converted
        :return: A list of pseudo-x86 statements equivalent to stmts
        """
        instrs = []
        for stmt in stmts:
            instrs.extend(si_stmt(stmt))

        return instrs

    # TODO: test
    def si_def(func_def: cfun.CFunctionDef) -> X86FunctionDef:
        nonlocal current_function_name

        current_function_name = func_def.name
        available_arg_registers = constants.argument_registers.copy()

        new_blocks: Dict[str, List[x86.Instr]] = {}
        for name, stmts in func_def.blocks.items():
            new_blocks[name] = si_stmts(stmts)

        # Instructions for setting up the args for each function
        arg_instrs: List[x86.Instr] = []
        for arg_name in func_def.args:
            arg_register = x86.Reg(available_arg_registers.pop(0))
            arg_instrs.extend(
                [x86.NamedInstr("movq", [arg_register, x86.Var(arg_name)])])

        # Add setup instructions to start of block
        new_blocks[current_function_name + "start"] = arg_instrs + \
            new_blocks[current_function_name + "start"]

        return X86FunctionDef(func_def.name, new_blocks, (None, None))

    x86_functions = []
    for function_def in prog.defs:
        x86_functions.append(si_def(function_def))

    # print(ast_printer.print_ast(X86ProgramDefs(x86_functions)))
    return X86ProgramDefs(x86_functions)


##################################################
# allocate-registers
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

Color = x86.Arg
Coloring = Dict[x86.Var, x86.Arg]
Saturation = Set[x86.Arg]


def allocate_registers(prog: X86ProgramDefs) -> X86ProgramDefs:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param prog: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """

    # call _allocate_registers for each function definition
    new_defs = []
    for function_def in prog.defs:
        mini_prog: x86.X86Program = x86.X86Program(blocks=function_def.blocks)
        result_mini_prog: x86.X86Program = _allocate_registers(
            name=function_def.label, prog=mini_prog)

        new_function_def = X86FunctionDef(
            function_def.label, result_mini_prog.blocks, result_mini_prog.stack_space)
        new_defs.append(new_function_def)

    # print(ast_printer.print_ast(X86ProgramDefs(new_defs)))
    return X86ProgramDefs(new_defs)


def _allocate_registers(name: str, prog: x86.X86Program) -> x86.X86Program:
    # --------------------------------------------------
    # utilities
    # --------------------------------------------------
    all_vars: Set[x86.Var] = set()  # every variable in the program

    # variable homes; arg can be a register or a deref
    homes: Dict[x86.Var, x86.Arg] = {}
    basic_blocks = prog.blocks

    live_after_sets: Dict[str, List[Set[x86.Var]]] = {}
    live_before_sets: Dict[str, Set[x86.Var]] = {name+"conclusion": set()}
    for label in basic_blocks.keys():
        live_before_sets[label] = set()

    def align(num_bytes: int) -> int:
        """
        Aligns a given number of bytes to the next largest multiple of 16
        :param num_bytes: The number of bytes to align, must be a multiple of 8
        :return: num_bytes if it's a multiple of 16, nmu_bytes + 8 otherwise
        """
        if num_bytes % 16 == 0:
            return num_bytes
        else:
            return num_bytes + 8

    def vars_of_arg(a: x86.Arg) -> Set[x86.Var] | Set[x86.Reg]:
        """
        Returns variable in an argument if there is any
        Supports
            Arg ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
        :param a: Argument to check
        :return: x86 variable if variable present, empty set otherwise
        """
        match a:
            case x86.Immediate() | x86.ByteReg() | x86.Deref() | x86.GlobalVal():
                return set()

            case x86.Var(x):
                all_vars.add(x86.Var(x))
                return {x86.Var(x)}

            case x86.Reg(r):
                return {x86.Reg(r)}

            case _:
                raise Exception("allocate_registers, vars_of", a)

    def reads_of(instr: x86.Instr) -> Set[x86.Var]:
        """
        Returns a set of variables read by a given instruction
        Supports
            Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq() |
                       Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
            op     ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
        :param instr: The instruction to check
        :return: The variables read by instruction i
        """

        match instr:
            case x86.NamedInstr(op, [e1, e2]):
                match op:
                    case "movq" | "movzbq" | "leaq":
                        return vars_of_arg(e1)

                    case "addq" | "subq" | "imulq" | "cmpq" | "andq" | "orq" | "xorq":
                        return vars_of_arg(e1).union(vars_of_arg(e2))

            case x86.Jmp(label) | x86.JmpIf(_, label):
                # Return all variables in the destination's live-before set
                return live_before_sets[label]

            # TODO: Don't understand this
            case x86.IndirectCallq(e1):
                return vars_of_arg(e1)

            case x86.Callq() | x86.Retq | x86.Set():  # TODO: retq might not be in right place
                return set()

            case _:
                raise Exception("allocate_registers, reads_of", instr)

    def writes_of(instr: x86.Instr) -> Set[x86.Var]:
        """
        Returns a set of variables written by a given instruction
        Supports
            Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) |
                       Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
            op     ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
        :param instr: The instruction to check
        :return: The variables written by instruction i
        """

        match instr:
            case x86.NamedInstr(i, [e1, e2]):
                match i:
                    case 'movq' | 'movzbq' | "leaq" | 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq':
                        return vars_of_arg(e2)

                    case _:
                        raise Exception(
                            "allocate_registers, writes_of, NamedInstr", i)

            case x86.Jmp() | x86.JmpIf() | x86.Callq() | x86.IndirectCallq() | x86.Set():
                return set()

            case _:
                raise Exception("allocate_registers, writes_of", instr)

    def get_stack_locations_used():
        stack_locations = 0
        root_stack_locations = 0

        for location in homes.values():
            match location:
                case x86.Reg:
                    pass
                case x86.Deref("rbp", _):
                    stack_locations += 1
                case x86.Deref("r15", _):
                    root_stack_locations += 1

        return stack_locations, root_stack_locations

    # --------------------------------------------------
    # liveness analysis
    # --------------------------------------------------

    def ul_instr(instr: x86.Instr, live_after: Set[x86.Var]) -> Set[x86.Var]:
        """
        Given an instruction `k+1` and the live-after set for instruction `k+1`,
            produce the live-after set for instruction `k`
            by removing any variables read and adding any variables written in instruction 'k+1'
        :param instr: The instruction 'k+1' that the live-after set is calculated from
        :param live_after: The live-after set of instruction 'k+1'
        :return: The live-after set for instruction 'k'
        """

        return live_after.difference(writes_of(instr)).union(reads_of(instr))

    def ul_block(label: str):
        """
        Given a label, calculate and save the live-after sets and live-before set of the block with that label
        :param label: The label of the block to be checked
        """
        instrs = basic_blocks[label]

        # Start with empty list of live-after sets
        block_live_after_sets: List[set] = []

        current_live_after = set()  # Start with empty current live-after set

        # Loop over instructions in reverse order
        for instr in reversed(instrs):
            # Add live after set of k+1 to the list
            block_live_after_sets.append(current_live_after)

            # Get the live-after set for instruction k
            current_live_after = ul_instr(instr, current_live_after)

        live_before_sets[label] = current_live_after

        # At the end, reverse the list of live-after sets and save it
        live_after_sets[label] = list(reversed(block_live_after_sets))

    def ul_fixpoint(labels: List[str]):
        old_live_afters = live_after_sets.copy()

        while True:
            for label in labels:
                ul_block(label)

            if old_live_afters == live_after_sets:
                return
            else:
                old_live_afters = live_after_sets.copy()

    # --------------------------------------------------
    # interference graph
    # --------------------------------------------------
    def bi_instr(instr: x86.Instr, live_after: Set[x86.Var], graph: InterferenceGraph):
        match instr:
            case x86.Callq("collect"):
                for live_var in live_after:
                    # Add an edge between all caller-saved registers and the current live_var
                    for reg in constants.caller_saved_registers:
                        graph.add_edge(live_var, x86.Reg(reg))

                    # Forces compiler to move all tuple vars to the root stack
                    # TODO: Don't understand how
                    if isinstance(live_var, x86.Var) and live_var.var in tuple_var_types:  # TODO
                        for reg in constants.callee_saved_registers:
                            graph.add_edge(x86.Reg(reg), live_var)

            case x86.Callq() | x86.IndirectCallq():
                # Add an edge between all caller-saved registers and the current live_var
                for live_var in live_after:
                    for register in constants.caller_saved_registers:
                        graph.add_edge(live_var, x86.Reg(register))

            case _:
                for written_var in writes_of(instr):
                    for live_after_var in live_after:
                        graph.add_edge(written_var, live_after_var)

    def bi_block(instrs: List[x86.Instr], live_afters: List[Set[x86.Var]], graph: InterferenceGraph):
        for i in range(0, len(instrs)):
            bi_instr(instrs[i], live_afters[i], graph)

    # --------------------------------------------------
    # graph coloring
    # --------------------------------------------------
    def color_graph(local_vars: Set[x86.Var],
                    interference_graph: InterferenceGraph,
                    regular_locations: List[x86.Arg],
                    tuple_locations: List[x86.Arg]) -> Coloring:

        # Fill each saturation set with all registers that are neighbors of each variable
        saturation_sets: Dict[x86.Var, Saturation] = {}

        for local_var in local_vars:
            saturation_sets[local_var] = set()
            for neighbor in interference_graph.neighbors(local_var):
                if isinstance(neighbor, x86.Reg):
                    saturation_sets[local_var].add(neighbor)

        coloring: Coloring = {}

        vars_to_color = local_vars.copy()
        non_colored_sat_sets = saturation_sets.copy()

        # Loop until there are no more vars to color
        while len(vars_to_color) > 0:
            # Find the variable with the largest saturation set
            var_to_color = max(non_colored_sat_sets,
                               key=lambda var: len(saturation_sets[var]))

            if var_to_color.var in tuple_var_types:
                possible_locations = tuple_locations
            else:
                possible_locations = regular_locations

            # Pick the lowest color not in its saturation set
            location_to_use = next(
                i for i in possible_locations if i not in saturation_sets[var_to_color])
            coloring[var_to_color] = location_to_use

            # Update the saturation sets
            for neighbor in interference_graph.neighbors(var_to_color):
                if isinstance(neighbor, x86.Var):
                    saturation_sets[neighbor].add(location_to_use)

            non_colored_sat_sets.pop(var_to_color)
            vars_to_color.discard(var_to_color)

        return coloring

    # --------------------------------------------------
    # assigning homes
    # --------------------------------------------------
    # Match for each Arg: registers, variables, immediates
    # For variable case:
    #   if 'a' in homes return homes[a],
    #   else create new home (Deref node), save it to homes, return homes[a]
    def ah_arg(arg: x86.Arg) -> x86.Arg:
        match arg:
            case x86.Immediate() | x86.Reg() | x86.ByteReg() | x86.GlobalVal() | x86.Deref():
                return arg

            case x86.Var(x):
                return homes[x86.Var(x)]

            case _:
                raise Exception("ah_arg", arg)

    # Match with one case per instruction type
    #   (only need to worry about instruction w/ variables; named instructions(op, arg)).
    # Calls ah_arg on each arg
    def ah_instr(instr: x86.Instr) -> x86.Instr:
        match instr:
            case x86.NamedInstr(op, args):
                new_args = []
                for arg in args:
                    new_args.append(ah_arg(arg))

                return x86.NamedInstr(op, new_args)

            case x86.Set(cc, arg):
                return x86.Set(cc, ah_arg(arg))

            case x86.IndirectCallq(func, num_args):
                return x86.IndirectCallq(ah_arg(func), num_args)

            case x86.Callq() | x86.Retq() | x86.Jmp() | x86.JmpIf():
                return instr

            case _:
                raise Exception("ah_instr", instr)

    def ah_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = []
        for instr in instrs:
            new_instrs.append(ah_instr(instr))

        return new_instrs

    # --------------------------------------------------
    # main body of the pass
    # --------------------------------------------------

    # Step 1: Perform liveness analysis
    # Run the liveness analysis
    blocks = prog.blocks
    ul_fixpoint(list(blocks.keys()))

    # prints live_after sets
    log_ast('live-after sets', live_after_sets)

    # Step 2: Build the interference graph
    interference_graph = InterferenceGraph()

    for label in blocks.keys():
        bi_block(blocks[label], live_after_sets[label], interference_graph)

    log_ast('interference graph', interference_graph)

    # Step 3: Color the graph
    # coloring = color_graph(all_vars, interference_graph)
    # log('coloring', coloring)
    #
    # # Defines the set of registers to use

    available_registers = constants.caller_saved_registers + \
        constants.callee_saved_registers

    regular_locations = []
    tuple_locations = []

    for reg in available_registers:
        regular_locations.append(x86.Reg(reg))
        tuple_locations.append(x86.Reg(reg))

    for offset in range(0, 200):
        regular_locations.append(x86.Deref("rbp", -(offset * 8)))
        tuple_locations.append(x86.Deref("r15", -(offset * 8)))

    homes = color_graph(all_vars, interference_graph,
                        regular_locations, tuple_locations)

    stack_locations_used, root_stack_locations_used = get_stack_locations_used()

    # Step 5: replace variables with their homes
    blocks = prog.blocks
    new_blocks = {}
    for label, instrs in blocks.items():
        new_blocks[label] = ah_block(instrs)

    stack_size = align(stack_locations_used * 8)
    return x86.X86Program(new_blocks, (stack_size, root_stack_locations_used))


##################################################
# patch-instructions
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

def patch_instructions(prog: X86ProgramDefs) -> X86ProgramDefs:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param prog: An x86 program.
    :return: A patched x86 program.
    """

    # call _patch_instructions for each function definition
    new_defs = []
    for function_def in prog.defs:
        mini_prog: x86.X86Program = x86.X86Program(
            blocks=function_def.blocks, stack_space=(0, 0))
        result_mini_prog: x86.X86Program = _patch_instructions(mini_prog)

        new_function_def = X86FunctionDef(
            function_def.label, result_mini_prog.blocks, result_mini_prog.stack_space)
        new_defs.append(new_function_def)

    # print(ast_printer.print_ast(X86ProgramDefs(new_defs)))
    return X86ProgramDefs(new_defs)


def _patch_instructions(prog: x86.X86Program) -> x86.X86Program:

    def needs_patching(args: List) -> bool:
        """
        Determines if an instruction with the given args needs to be patched
        :param args: The args to check
        :return: True if the instruction needs to be patched; False otherwise
        """
        # If both of an instruction's arguments are of a type in this list, it must be patched
        memory_location_types = [x86.Deref, x86.GlobalVal]

        if type(args[0]) in memory_location_types and type(args[1]) in memory_location_types:
            return True
        else:
            return False

    def pi_instr(instr: x86.Instr) -> List[x86.Instr]:
        """
        Patches an instruction with multiple memory accesses into multiple instructions with single memory access
        :param instr: The instruction to patch
        :return: The original instruction if not patches were needed; A list of patched instructions otherwise
        """
        match instr:
            case x86.NamedInstr("cmpq", [a1, x86.Immediate(instr)]):
                # cmpq cannot have an immediate as a second arg
                return [x86.NamedInstr("movq", [x86.Immediate(instr), x86.Reg("rax")]),
                        x86.NamedInstr("cmpq", [a1, x86.Reg("rax")])]

            case x86.NamedInstr("movq", args):
                if needs_patching(args):
                    return [x86.NamedInstr("movq", [args[0], x86.Reg("rax")]),
                            x86.NamedInstr("movq", [x86.Reg("rax"), args[1]])]
                else:
                    return [instr]

            case x86.NamedInstr("addq", args):
                if needs_patching(args):
                    return [x86.NamedInstr("movq", [args[0], x86.Reg("rax")]),
                            x86.NamedInstr("addq", [x86.Reg("rax"), args[1]])]
                else:
                    return [instr]

            case _:
                return [instr]

    def pi_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        """
        Patches a block of the program
        :param instrs: A list of instructions to patch
        :return: A list of patched instructions equivalent to `instrs`
        """
        new_instrs = []
        for instr in instrs:
            new_instrs.extend(pi_instr(instr))

        return new_instrs

    blocks = prog.blocks
    new_blocks = {}
    for block_label, block_instrs in blocks.items():
        new_blocks[block_label] = pi_block(block_instrs)

    return x86.X86Program(new_blocks, stack_space=prog.stack_space)


##################################################
# prelude-and-conclusion
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Deref(r, offset) | GlobalVal(x)
# op     ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#          | 'leaq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#          | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#          | IndirectCallq(Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

def prelude_and_conclusion(prog: X86ProgramDefs) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param prog: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """

    new_blocks = {}
    for function_def in prog.defs:
        mini_prog = x86.X86Program(
            blocks=function_def.blocks, stack_space=function_def.stack_space)
        result_mini_prog = _prelude_and_conclusion(
            name=function_def.label, prog=mini_prog)
        # TODO: what is happening here?
        new_blocks.update(result_mini_prog.blocks)

    # print(ast_printer.print_ast(x86.X86Program(new_blocks)))
    return x86.X86Program(new_blocks)


def _prelude_and_conclusion(name: str, prog: x86.X86Program) -> x86.X86Program:
    """
    The following registers are used for the following purposes
        rbp: the base of the current stack frame
        rsp: the 'top' of the current stack frame
        rdi: the root-stack size
        rsi: the heap size
        r15: the current position in the root stack
    """
    stack_bytes, root_stack_locations = prog.stack_space

    def create_prelude() -> List[x86.Instr]:
        prelude: List[x86.Instr] = []

        # Set up the stack frame (rsp points to the 'top' of stack from and rbp points to the 'base')
        #   1. Push the old rbp (stack base) onto the stack
        #   2. Store current 'top' of stack in rbp
        prelude += [x86.NamedInstr("pushq", [x86.Reg("rbp")]),
                    x86.NamedInstr("movq", [x86.Reg("rsp"), x86.Reg("rbp")])]

        # TODO: last instr might need to go after next section

        # Save the callee-saved registers
        #   Push each onto the stack
        for reg in constants.callee_saved_registers:
            prelude += [x86.NamedInstr("pushq", [x86.Reg(reg)])]

        # Continue setting up the stack frame
        #   Move rsp to the new 'top' of the stack
        prelude += [x86.NamedInstr('subq',
                                   [x86.Immediate(stack_bytes), x86.Reg('rsp')])]

        # Initialize the heap if this is the `main` function
        #   1. Store the root-stack size in rdi
        #   2. Store the heap size in rsi
        #   3. Initialize the heap
        #   4. Store the value of where the root-stack begins in r15
        if name == "main":
            prelude += [x86.NamedInstr("movq", [x86.Immediate(constants.root_stack_size), x86.Reg("rdi")]),
                        x86.NamedInstr("movq", [x86.Immediate(
                            constants.heap_size), x86.Reg("rsi")]),
                        x86.Callq("initialize"),
                        x86.NamedInstr("movq", [x86.GlobalVal("rootstack_begin"), x86.Reg("r15")])]

        # Set up the root stack
        #   1. Put the value `0` (8 bytes) into the memory address where r15 is currently pointing
        #   2. Move r15 to point to the next 8-byte section of memory
        for i in range(root_stack_locations):
            prelude += [[x86.NamedInstr("movq", [x86.Immediate(0), x86.Deref("r15", 0)]),
                         x86.NamedInstr("addq", [x86.Immediate(8), x86.Reg("r15")])]]

        # Jump to the start of the program
        prelude += [x86.Jmp(name+"start")]

        return prelude

    def create_conclusion() -> List[x86.Instr]:
        conclusion: List[x86.Instr] = []

        # Restore rsp and r15
        #   1. Move rsp back to the 'top' of the previous stack
        #   2. Move r15 back to the base of the root-stack
        conclusion += [x86.NamedInstr("addq", [x86.Immediate(stack_bytes), x86.Reg("rsp")]),
                       x86.NamedInstr("subq", [x86.Immediate(root_stack_locations * 8), x86.Reg("r15")])]

        # Restore the callee-saved registers
        #   Pop each off of the stack
        for reg in reversed(constants.callee_saved_registers):
            conclusion += [x86.NamedInstr("popq", [x86.Reg(reg)])]

        # Restore rbp
        #   Pop the old rbp off of the stack
        conclusion += [x86.NamedInstr("popq", [x86.Reg("rbp")])]

        # Return
        conclusion += [x86.Retq()]

        return conclusion

    new_blocks = prog.blocks.copy()
    new_blocks[name] = create_prelude()
    new_blocks[name + "conclusion"] = create_conclusion()

    return x86.X86Program(new_blocks, stack_space=prog.stack_space)


##################################################
# Compiler definition
##################################################

compiler_passes = {
    'typecheck': typecheck,
    'compress parameters': compress_parameters,
    'remove complex opera*': rco,
    'typecheck2': typecheck,
    'expose allocation': expose_alloc,
    'explicate control': explicate_control,
    'select instructions': select_instructions,
    'allocate registers': allocate_registers,
    'patch instructions': patch_instructions,
    'prelude & conclusion': prelude_and_conclusion,
    'print x86': x86.print_x86
}


def run_compiler(s, logging=False):
    global global_logging

    old_logging = global_logging
    global_logging = logging

    def print_prog(current_program):
        print('Concrete syntax:')
        if isinstance(current_program, x86.X86Program):
            print(x86.print_x86(current_program))
        elif isinstance(current_program, X86ProgramDefs):
            print(print_x86defs.print_x86_defs(current_program))
        elif isinstance(current_program, Program):
            print(print_program(current_program))
        elif isinstance(current_program, cfun.CProgram):
            print(cfun.print_program(current_program))

        print()
        print('Abstract syntax:')
        print(print_ast(current_program))

    current_program = parse(s)

    if logging == True:
        print()
        print('==================================================')
        print(' Input program')
        print('==================================================')
        print()
        print_prog(current_program)

    for pass_name, pass_fn in compiler_passes.items():
        current_program = pass_fn(current_program)

        if logging == True:
            print()
            print('==================================================')
            print(f' Output of pass: {pass_name}')
            print('==================================================')
            print()
            print_prog(current_program)

    global_logging = old_logging
    return current_program


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python compiler.py <source filename>')
    else:
        file_name = sys.argv[1]
        with open(file_name) as f:
            print(f'Compiling program {file_name}...')

            try:
                program = f.read()
                x86_program = run_compiler(program, logging=False)

                with open(file_name + '.s', 'w') as output_file:
                    output_file.write(x86_program)

            except:
                print(
                    'Error during compilation! **************************************************')
                traceback.print_exception(*sys.exc_info())
