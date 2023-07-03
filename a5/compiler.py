from typing import Set, Dict
import itertools
import sys
import traceback

from cs202_support.python import *
import cs202_support.x86 as x86
import constants
import cif
from interference_graph import InterferenceGraph

comparisons = ['eq', 'gt', 'gte', 'lt', 'lte']
gensym_num = 0
global_logging = False


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
# op     ::= "add" | "sub" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
# Stmts  ::= List[Stmt]
# LWhile ::= Program(Stmts)

TEnv = Dict[str, type]


def typecheck(program: Program) -> Program:
    """
    Typechecks the input program; throws an error if the program is not well-typed.
    :param program: The Lif program to typecheck
    :return: The program, if it is well-typed
    """

    prim_arg_types = {
        'add':   [int, int],
        'sub':   [int, int],
        'mult':  [int, int],
        'not': [bool],
        'or':  [bool, bool],
        'and':  [bool, bool],
        'gt':   [int, int],
        'gte':  [int, int],
        'lt':   [int, int],
        'lte':  [int, int],
    }

    prim_output_types = {
        'add':   int,
        'sub':   int,
        'mult':  int,
        'not': bool,
        'or':  bool,
        'and':  bool,
        'gt':   bool,
        'gte':  bool,
        'lt':   bool,
        'lte':  bool,
    }

    def tc_exp(e: Expr, env: TEnv) -> type:
        match e:
            case Var(x):
                return env[x]
            case Constant(i):
                if isinstance(i, bool):
                    return bool
                elif isinstance(i, int):
                    return int
                else:
                    raise Exception('tc_exp', e)
            case Prim('eq', [e1, e2]):
                assert tc_exp(e1, env) == tc_exp(e2, env)
                return bool
            case Prim(op, args):
                arg_types = [tc_exp(a, env) for a in args]
                assert arg_types == prim_arg_types[op]
                return prim_output_types[op]
            case _:
                raise Exception('tc_exp', e)

    def tc_stmt(s: Stmt, env: TEnv):
        match s:
            case If(condition, then_stmts, else_stmts):
                assert tc_exp(condition, env) == bool
                for s in then_stmts:
                    tc_stmt(s, env)
                for s in else_stmts:
                    tc_stmt(s, env)
            # Add typechecking for While
            case While(condition, body):
                assert tc_exp(condition, env) == bool
                for s in body:
                    tc_stmt(s, env)
            case Print(e):
                tc_exp(e, env)
            case Assign(x, e):
                t_e = tc_exp(e, env)
                if x in env:
                    assert t_e == env[x]
                else:
                    env[x] = t_e
            case _:
                raise Exception('tc_stmt', s)

    def tc_stmts(stmts: List[Stmt], env: TEnv):
        for s in stmts:
            tc_stmt(s, env)

    env = {}
    tc_stmts(program.stmts, env)
    return program



##################################################
# remove-complex-opera*
##################################################
# op     ::= "add" | "sub" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
# Stmts  ::= List[Stmt]
# LWhile ::= Program(Stmts)

def rco(prog: Program) -> Program:
    """
    Removes complex operands. After this pass, the arguments to operators (unary and binary
    operators, and function calls like "print") will be atomic.
    :param prog: An Lif program
    :return: An Lif program with atomic operator arguments.
    """

    def rco_stmt(stmt: Stmt, bindings: Dict[str, Expr]) -> Stmt:
        match stmt:
            case Assign(x, e1):
                new_e1 = rco_exp(e1, bindings)
                return Assign(x, new_e1)
            case Print(e1):
                new_e1 = rco_exp(e1, bindings)
                return Print(new_e1)
            case If(condition, then_stmts, else_stmts):
                new_condition = rco_exp(condition, bindings)
                new_then_stmts = rco_stmts(then_stmts)
                new_else_stmts = rco_stmts(else_stmts)

                return If(new_condition,
                          new_then_stmts,
                          new_else_stmts)
            case While(condition, body):
                # (1) compile the body
                new_body_stmts = rco_stmts(body)
                condition_bindings = {}
                new_condition_exp = rco_exp(condition, condition_bindings)
                begin_stmts = [Assign(x, e) for x, e in condition_bindings.items()]
                return While(Begin(begin_stmts, new_condition_exp), new_body_stmts)
            case _:
                raise Exception('rco_stmt', stmt)

    def rco_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []

        for stmt in stmts:
            bindings = {}
            # (1) compile the statement
            new_stmt = rco_stmt(stmt, bindings)
            # (2) add the new bindings created by rco_exp
            new_stmts.extend([Assign(x, e) for x, e in bindings.items()])
            # (3) add the compiled statement itself
            new_stmts.append(new_stmt)

        return new_stmts

    def rco_exp(e: Expr, bindings: Dict[str, Expr]) -> Expr:
        match e:
            case Var(x):
                return Var(x)
            case Constant(i):
                return Constant(i)
            case Prim(op, args):
                new_args = [rco_exp(e, bindings) for e in args]
                new_e = Prim(op, new_args)
                new_v = gensym('tmp')
                bindings[new_v] = new_e
                return Var(new_v)
            case _:
                raise Exception('rco_exp', e)

    return Program(rco_stmts(prog.stmts))


##################################################
# explicate-control
##################################################
# op     ::= "add" | "sub" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
# Atm    ::= Var(x) | Constant(n)
# Expr   ::= Atm | Prim(op, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
# Stmts  ::= List[Stmt]
# LWhile ::= Program(Stmts)

def explicate_control(prog: Program) -> cif.CProgram:
    """
    Transforms an Lif Expression into a Cif program
    :param prog: An Lif Expression
    :return: A Cif Program
    """
    basic_blocks: Dict[str, List[cif.Stmt]] = {}

    def create_block(stmts: List[cif.Stmt]) -> str:
        label = gensym('label')
        basic_blocks[label] = stmts
        return label

    def explicate_atm(e: Expr) -> cif.Atm:
        match e:
            case Var(x):
                return cif.Var(x)
            case Constant(c):
                return cif.Constant(c)
            case _:
                raise RuntimeError(e)

    def explicate_exp(e: Expr) -> cif.Expr:
        match e:
            case Prim(op, args):
                new_args = [explicate_atm(a) for a in args]
                return cif.Prim(op, new_args)
            case _:
                return explicate_atm(e)

    def explicate_stmt(stmt: Stmt, cont: List[cif.Stmt]) -> List[cif.Stmt]:
        match stmt:
            case Assign(x, exp):
                new_exp = explicate_exp(exp)
                new_stmt: List[cif.Stmt] = [cif.Assign(x, new_exp)]
                return new_stmt + cont
            case Print(exp):
                new_exp = explicate_atm(exp)
                new_stmt: List[cif.Stmt] = [cif.Print(new_exp)]
                return new_stmt + cont
            case If(condition, then_stmts, else_stmts):
                cont_label = create_block(cont)
                e2_label = create_block(explicate_stmts(then_stmts, [cif.Goto(cont_label)]))
                e3_label = create_block(explicate_stmts(else_stmts, [cif.Goto(cont_label)]))
                return [cif.If(explicate_exp(condition),
                               cif.Goto(e2_label),
                               cif.Goto(e3_label))]
            
            case While(Begin(condition_stmts, condition_exp), body_stmts):
                continuation_label = create_block(cont)
                test_label = gensym('loop_label')
                body_label = create_block(explicate_stmts(body_stmts, [cif.Goto(test_label)]))

                new_continuation_stmts = [cif.If(explicate_exp(condition_exp), cif.Goto(body_label), cif.Goto(continuation_label))]
                new_condition_stmts = explicate_stmts(condition_stmts, new_continuation_stmts)
                basic_blocks[test_label] = new_condition_stmts

                return [cif.Goto(test_label)]
                
            case _:
                raise RuntimeError(stmt)

    def explicate_stmts(stmts: List[Stmt], cont: List[cif.Stmt]) -> List[cif.Stmt]:
        for s in reversed(stmts):
            cont = explicate_stmt(s, cont)
        return cont

    new_body = [cif.Return(cif.Constant(0))]
    new_body = explicate_stmts(prog.stmts, new_body)

    basic_blocks['start'] = new_body
    return cif.CProgram(basic_blocks)


##################################################
# select-instructions
##################################################
# op    ::= "add" | "sub" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
# Atm   ::= Var(x) | Constant(n)
# Expr  ::= Atm | Prim(op, List[Expr])
# Stmt  ::= Assign(x, Expr) | Print(Expr)
#        | If(Expr, Goto(label), Goto(label)) | Goto(label) | Return(Expr)
# Stmts ::= List[Stmt]
# Cif   ::= CProgram(Dict[label, Stmts])

def select_instructions(prog: cif.CProgram) -> x86.X86Program:
    """
    Transforms a Lif program into a pseudo-x86 assembly program.
    :param prog: a Lif program
    :return: a pseudo-x86 program
    """

    def si_atm(a: cif.Expr) -> x86.Arg:
        match a:
            case cif.Constant(i):
                return x86.Immediate(int(i))
            case cif.Var(x):
                return x86.Var(x)
            case _:
                raise Exception('si_atm', a)

    def si_stmts(stmts: List[cif.Stmt]) -> List[x86.Instr]:
        instrs = []

        for stmt in stmts:
            instrs.extend(si_stmt(stmt))

        return instrs

    op_cc = {'eq': 'e', 'gt': 'g', 'gte': 'ge', 'lt': 'l', 'lte': 'le'}

    binop_instrs = {'add': 'addq', 'sub': 'subq', 'mult': 'imulq', 'and': 'andq', 'or': 'orq'}

    def si_stmt(stmt: cif.Stmt) -> List[x86.Instr]:
        match stmt:
            case cif.Assign(x, cif.Prim(op, [atm1, atm2])):
                if op in binop_instrs:
                    return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                            x86.NamedInstr(binop_instrs[op], [si_atm(atm2), x86.Reg('rax')]),
                            x86.NamedInstr('movq', [x86.Reg('rax'), x86.Var(x)])]
                elif op in op_cc:
                    return [x86.NamedInstr('cmpq', [si_atm(atm2), si_atm(atm1)]),
                            x86.Set(op_cc[op], x86.ByteReg('al')),
                            x86.NamedInstr('movzbq', [x86.ByteReg('al'), x86.Var(x)])]

                else:
                    raise Exception('si_stmt failed op', op)
            case cif.Assign(x, cif.Prim('not', [atm1])):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Var(x)]),
                        x86.NamedInstr('xorq', [x86.Immediate(1), x86.Var(x)])]
            case cif.Assign(x, atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Var(x)])]
            case cif.Print(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rdi')]),
                        x86.Callq('print_int')]
            case cif.Return(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                        x86.Jmp('conclusion')]
            case cif.Goto(label):
                return [x86.Jmp(label)]
            case cif.If(a, cif.Goto(then_label), cif.Goto(else_label)):
                return [x86.NamedInstr('cmpq', [si_atm(a), x86.Immediate(1)]),
                        x86.JmpIf('e', then_label),
                        x86.Jmp(else_label)]
            case _:
                raise Exception('si_stmt', stmt)

    basic_blocks = {label: si_stmts(block) for (label, block) in prog.blocks.items()}
    return x86.X86Program(basic_blocks)



##################################################
# allocate-registers
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset)
# op     ::= 'addq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#         | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

Color = int
Coloring = Dict[x86.Var, Color]
Saturation = Set[Color]

def allocate_registers(program: x86.X86Program) -> x86.X86Program:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param program: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """


    all_vars: Set[x86.Var] = set()
    live_before_sets = {'conclusion': set()}
    live_after_sets = {}
    homes: Dict[x86.Var, x86.Arg] = {}

    # --------------------------------------------------
    # utilities
    # --------------------------------------------------
    # initilize live_before_sets every basic block to be empty

    basic_blocks = program.blocks
    for label in basic_blocks.keys():
        live_before_sets[label] = set()

    def vars_arg(a: x86.Arg) -> Set[x86.Var]:
        match a:
            case x86.Immediate(i):
                return set()
            case x86.Reg(r):
                return set()
            case x86.ByteReg(r):
                return set()
            case x86.Var(x):
                all_vars.add(x86.Var(x))
                return {x86.Var(x)}
            case _:
                raise Exception('ul_arg', a)

    def reads_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, [e1, e2]) if i in ['movq', 'movzbq']:
                return vars_arg(e1)
            case x86.NamedInstr(i, [e1, e2]) if i in ['addq', 'cmpq', 'imulq', 'subq', 'andq', 'orq', 'xorq']:
                return vars_arg(e1).union(vars_arg(e2))
            case x86.Jmp(label) | x86.JmpIf(_, label):
                return live_before_sets[label]
            case _:
                if isinstance(i, (x86.Callq, x86.Set)):
                    return set()
                else:
                    raise Exception(i)

    def writes_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, [e1, e2]) \
                if i in ['movq', 'movzbq', 'addq', 'subq', 'imulq', 'cmpq', 'andq', 'orq', 'xorq']:
                return vars_arg(e2)
            case _:
                if isinstance(i, (x86.Jmp, x86.JmpIf, x86.Callq, x86.Set)):
                    return set()
                else:
                    raise Exception(i)

    # --------------------------------------------------
    # liveness analysis
    # --------------------------------------------------
    def ul_instr(i: x86.Instr, live_after: Set[x86.Var]) -> Set[x86.Var]:
        return live_after.difference(writes_of(i)).union(reads_of(i))

    def ul_block(label: str):
        instrs = basic_blocks[label]
        current_live_after: Set[x86.Var] = set()

        block_live_after_sets = []
        for i in reversed(instrs):
            block_live_after_sets.append(current_live_after)
            current_live_after = ul_instr(i, current_live_after)

        live_before_sets[label] = current_live_after
        live_after_sets[label] = list(reversed(block_live_after_sets))
    
    def ul_fixpoint(labels: List[str]):
        prev_live_after = live_after_sets.copy()
        while (True):
            for label in labels:
                ul_block(label)
            if prev_live_after == live_after_sets:
                return
            prev_live_after = live_after_sets.copy()

    # --------------------------------------------------
    # interference graph
    # --------------------------------------------------
    def bi_instr(e: x86.Instr, live_after: Set[x86.Var], graph: InterferenceGraph):
        for v1 in writes_of(e):
            for v2 in live_after:
                graph.add_edge(v1, v2)

    def bi_block(instrs: List[x86.Instr], live_afters: List[Set[x86.Var]], graph: InterferenceGraph):
        for instr, live_after in zip(instrs, live_afters):
            bi_instr(instr, live_after, graph)

    # --------------------------------------------------
    # graph coloring
    # --------------------------------------------------
    def color_graph(local_vars: Set[x86.Var], interference_graph: InterferenceGraph) -> Coloring:
        coloring: Coloring = {}

        to_color = local_vars.copy()

        # Saturation sets start out empty
        saturation_sets = {x: set() for x in local_vars}

        # Loop until we are finished coloring
        while to_color:
            # Find the variable x with the largest saturation set
            x = max(to_color, key=lambda x: len(saturation_sets[x]))

            # Remove x from the variables to color
            to_color.remove(x)

            # Find the smallest color not in x's saturation set
            x_color = next(i for i in itertools.count() if i not in saturation_sets[x])

            # Assign x's color
            coloring[x] = x_color

            # Add x's color to the saturation sets of its neighbors
            for y in interference_graph.neighbors(x):
                saturation_sets[y].add(x_color)

        return coloring

    # --------------------------------------------------
    # assigning homes
    # --------------------------------------------------
    def align(num_bytes: int) -> int:
        if num_bytes % 16 == 0:
            return num_bytes
        else:
            return num_bytes + (16 - (num_bytes % 16))

    def ah_arg(a: x86.Arg) -> x86.Arg:
        match a:
            case x86.Immediate(i):
                return a
            case x86.Reg(r):
                return a
            case x86.ByteReg(r):
                return a
            case x86.Var(x):
                return homes[x86.Var(x)]
            case _:
                raise Exception('ah_arg', a)

    def ah_instr(e: x86.Instr) -> x86.Instr:
        match e:
            case x86.NamedInstr(i, args):
                return x86.NamedInstr(i, [ah_arg(a) for a in args])
            case x86.Set(cc, a1):
                return x86.Set(cc, ah_arg(a1))
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf)):
                    return e
                else:
                    raise Exception('ah_instr', e)

    def ah_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        return [ah_instr(i) for i in instrs]

    # --------------------------------------------------
    # main body of the pass
    # --------------------------------------------------

    # Step 1: Perform liveness analysis
    blocks_labels = basic_blocks.keys()
    ul_fixpoint(blocks_labels)
    log_ast('live-after sets', live_after_sets)

    # Step 2: Build the interference graph
    interference_graph = InterferenceGraph()

    for label in basic_blocks.keys():
        bi_block(program.blocks[label], live_after_sets[label], interference_graph)

    log_ast('interference graph', interference_graph)

    # Step 3: Color the graph
    coloring = color_graph(all_vars, interference_graph)
    colors_used = set(coloring.values())
    log('coloring', coloring)

    # Defines the set of registers to use
    available_registers = constants.caller_saved_registers + constants.callee_saved_registers

    # Step 4: map variables to homes
    color_map = {}
    stack_locations_used = 0

    # Step 4.1: Map colors to locations (the "color map")
    for color in sorted(colors_used):
        if available_registers != []:
            r = available_registers.pop()
            color_map[color] = x86.Reg(r)
        else:
            offset = stack_locations_used+1
            color_map[color] = x86.Deref('rbp', -(offset * 8))
            stack_locations_used += 1

    # Step 4.2: Compose the "coloring" with the "color map" to get "homes"
    for v in all_vars:
        homes[v] = color_map[coloring[v]]
    log('homes', homes)

    # Step 5: replace variables with their homes
    blocks = program.blocks
    new_blocks = {label: ah_block(block) for label, block in blocks.items()}
    return x86.X86Program(new_blocks, stack_space = align(8 * stack_locations_used))

##################################################
# patch-instructions
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Deref(r, offset)
# op     ::= 'addq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#         | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

def patch_instructions(program: x86.X86Program) -> x86.X86Program:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param program: An x86 program.
    :return: A patched x86 program.
    """

    def pi_instr(e: x86.Instr) -> List[x86.Instr]:
        match e:
            case x86.NamedInstr(i, [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                return [x86.NamedInstr('movq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                        x86.NamedInstr(i, [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                    return [x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                            x86.NamedInstr('movq', [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case x86.NamedInstr('cmpq', [a1, x86.Immediate(i)]):
                return [x86.NamedInstr('movq', [x86.Immediate(i), x86.Reg('rax')]),
                        x86.NamedInstr('cmpq', [a1, x86.Reg('rax')])]
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf, x86.NamedInstr, x86.Set)):
                    return [e]
                else:
                    raise Exception('pi_instr', e)

    def pi_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = [pi_instr(i) for i in instrs]
        flattened = [val for sublist in new_instrs for val in sublist]
        return flattened

    blocks = program.blocks
    new_blocks = {label: pi_block(block) for label, block in blocks.items()}
    return x86.X86Program(new_blocks, stack_space = program.stack_space)



##################################################
# prelude-and-conclusion
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Deref(r, offset)
# op     ::= 'addq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#         | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

def prelude_and_conclusion(program: x86.X86Program) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param program: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """

    prelude = [x86.NamedInstr('pushq', [x86.Reg('rbp')]),
               x86.NamedInstr('movq',  [x86.Reg('rsp'), x86.Reg('rbp')]),
               x86.NamedInstr('subq',  [x86.Immediate(program.stack_space),
                                        x86.Reg('rsp')]),
               x86.Jmp('start')]

    conclusion = [x86.NamedInstr('addq', [x86.Immediate(program.stack_space),
                                          x86.Reg('rsp')]),
                  x86.NamedInstr('popq', [x86.Reg('rbp')]),
                  x86.Retq()]

    new_blocks = program.blocks.copy()
    new_blocks['main'] = prelude
    new_blocks['conclusion'] = conclusion
    return x86.X86Program(new_blocks, stack_space = program.stack_space)




##################################################
# Compiler definition
##################################################

compiler_passes = {
    'typecheck': typecheck,
    'remove complex opera*': rco,
    'explicate control': explicate_control,
    'select instructions': select_instructions,
    'allocate registers': allocate_registers,
    'patch instructions': patch_instructions,
    'prelude & conclusion': prelude_and_conclusion,
    'print x86': x86.print_x86
}


def run_compiler(s, logging=False):
    def print_prog(current_program):
        print('Concrete syntax:')
        if isinstance(current_program, x86.X86Program):
            print(x86.print_x86(current_program))
        elif isinstance(current_program, Program):
            print(print_program(current_program))
        elif isinstance(current_program, cif.CProgram):
            print(cif.print_program(current_program))

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
                print('Error during compilation! **************************************************')
                traceback.print_exception(*sys.exc_info())

