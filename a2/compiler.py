from typing import List, Set, Dict, Tuple
import sys
import traceback

from cs202_support.python import *
from cs202_support.ast_pretty_printer import *
from cs202_support.python_pretty_printer import *
import cs202_support.x86 as x86

gensym_num = 0

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
# remove-complex-opera*
##################################################

def rco(prog: Program) -> Program:
    """
    Removes complex operands. After this pass, the arguments to operators (unary and binary
    operators, and function calls like "print") will be atomic.
    :param prog: An Lvar program
    :return: An Lvar program with atomic operator arguments.
    """
    # Should always return an atomic expression
    def rco_expr(e: Expr, bindings: Dict[str, Expr]) -> Expr:
        match e:
            case Constant(n):
                return Constant(n)
            case Var(x):
                return Var(x)
            case Prim(op, args):
                # Recursive call to rco_exp should make the arguments atomic
                new_args = [rco_expr(a, bindings) for a in args]
                tmp = gensym("tmp")
                # Bind tmp to Prim(op, new_args)
                bindings[tmp] = Prim(op, new_args)
                # Return the temp variable
                return Var(tmp)
            case _:
                raise Exception(f'Unexpected expression: {e}')
    
    def rco_stmt(s: Stmt, bindings: Dict[str, Expr]) -> Stmt:
        match s:
            case Assign(x, e):
                return Assign(x, rco_expr(e, bindings))
            case Print(e):
                return Print(rco_expr(e, bindings))
            case _:
                raise Exception(f'Unexpected statement: {s}')
    
    def rco_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []
        
        for stmt in stmts:
            # every statement gets its own set of bindings
            bindings = {}
            new_stmt = rco_stmt(stmt, bindings)
            # add bindings to new_stmts
            for name, value in bindings.items():
                new_stmts.append(Assign(name, value))

            new_stmts.append(new_stmt)

        return new_stmts
    
    return Program(rco_stmts(prog.stmts))


##################################################
# select-instructions
##################################################

# Output language: pseudo-x86
# ATM ::= Immediate(n) | Var(x) | Reg(str)
# instr_name ::= "movq" | "addq"
# Instr ::= NamedInstr(instr_name, [Atm]) | Callq(str) | Retq()
# X86 ::= X86Program(Dict[str, [Instr]])

def select_instructions(prog: Program) -> x86.X86Program:
    """
    Transforms a Lvar program into a pseudo-x86 assembly program.
    :param prog: a Lvar program
    :return: a pseudo-x86 program
    """

    def si_atm(atm: Expr) -> x86.Arg:
        match atm:
            case Var(x):
                return x86.Var(x)
            case Constant(n):
                return x86.Immediate(n)
            case _:
                raise Exception(f'Unexpected expression: {atm}')

    # Converts an LVar statement into one or more x86 instructions
    def si_stmt(stmt: Stmt) -> List[x86.Instr]:
        instrs = []
        match stmt:
            case Assign(x, Prim('add', [atm1, atm2])):
                instrs.append(x86.NamedInstr("movq", [si_atm(atm1), x86.Reg("rax")]))
                instrs.append(x86.NamedInstr("addq", [si_atm(atm2), x86.Reg("rax")]))
                instrs.append(x86.NamedInstr("movq", [x86.Reg("rax"), x86.Var(x)]))
            case Assign(x, atm1):
                instrs.append(x86.NamedInstr("movq", [si_atm(atm1), x86.Var(x)]))
            case Print(atm1):
                instrs.append(x86.NamedInstr("movq", [si_atm(atm1), x86.Reg("rdi")]))
                instrs.append(x86.Callq("print_int"))
            case _:
                raise Exception(f'Unexpected statement: {stmt}')

        return instrs

    # si_stmts compiles a list of statements
    def si_stmts(stmts: List[Stmt]) -> List[x86.Instr]:
        instrs = []

        for stmt in stmts:
            # extend is used because si_stmt returns a list
            instrs.extend(si_stmt(stmt))
        return instrs

    new_instructions = si_stmts(prog.stmts)
    return x86.X86Program({"main": new_instructions})


##################################################
# assign-homes
##################################################

def assign_homes(prog: x86.X86Program) -> x86.X86Program:
    """
    Assigns homes to variables in the input program. Allocates a stack location for each
    variable in the program.
    :param prog: A pseudo-x86 program.
    :return: An x86 program, annotated with the amount of stack space used
    """

    def ah_arg(a: x86.Arg) -> x86.Arg:
        match a:
            case x86.Immediate(i):
                return x86.Immediate(i)
            case x86.Reg(r):
                return x86.Reg(r)
            case x86.Var(x):
                if x in homes:
                    return homes[x]
                else:
                    # calculate offset
                    offset = -8 * (len(homes) + 1)
                    deref = x86.Deref("rbp", offset)
                    homes[x] = deref
                    return homes[x]
            case _:
                raise Exception(f'Unexpected argument: {a}')

    
    def ah_instr(instr: x86.Instr) -> x86.Instr:
        match instr:
            case x86.NamedInstr(op, args):
                new_args = [ah_arg(arg) for arg in args]
                return x86.NamedInstr(op, new_args)

            case x86.Callq(op):
                return x86.Callq(op)

            # this may need to be changed
            case _:
                raise Exception(f'Unexpected instruction: {instr}')

    
    def ah_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = [ah_instr(instr) for instr in instrs]
        return new_instrs

    def align(num_bytes: int) -> int:
        if num_bytes % 16 == 0:
            return num_bytes
        else:
            return num_bytes + 8

    homes: Dict[str, x86.Deref] = {}

    blocks = prog.blocks
    new_blocks = {}
    # for most examples, there is only one block label 'main'
    for label, instrs in blocks.items():
        new_blocks[label] = ah_block(instrs)

    # now that we have homes, we can calculate the stack space
    stack_size = align(len(homes) * 8)

    return x86.X86Program(new_blocks, stack_space=stack_size)


##################################################
# patch-instructions
##################################################

def patch_instructions(prog: x86.X86Program) -> x86.X86Program:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param prog: An x86 program.
    :return: A patched x86 program.
    """
    
    def pi_instr(i: x86.Instr) -> List[x86.Instr]:
        match i:
            case x86.NamedInstr("movq", [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                new_instrs = [x86.NamedInstr("movq", [x86.Deref(r1, o1), x86.Reg("rax")]),
                              x86.NamedInstr("movq", [x86.Reg("rax"), x86.Deref(r2, o2)])]

                return new_instrs
            case x86.NamedInstr("addq", [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                new_instrs = [x86.NamedInstr("movq", [x86.Deref(r1, o1), x86.Reg("rax")]),
                              x86.NamedInstr("addq", [x86.Reg("rax"), x86.Deref(r2, o2)])]

                return new_instrs
            case _:
                # if the instruction is not one of the two above, just return it wrapped in a list
                return [i]

    def pi_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = []
        for instr in instrs:
            new_instrs.extend(pi_instr(instr))

        return new_instrs

    blocks = prog.blocks
    new_blocks = {}
    for label, instrs in blocks.items():
        new_blocks[label] = pi_block(instrs)

    # pass the stack space through
    return x86.X86Program(new_blocks, stack_space=prog.stack_space)


##################################################
# prelude-and-conclusion
##################################################

def prelude_and_conclusion(prog: x86.X86Program) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param prog: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """
    stack_size = prog.stack_space

    # push the stack frame
    prelude_instrs = [x86.NamedInstr("pushq", [x86.Reg("rbp")]),
                      x86.NamedInstr("movq", [x86.Reg("rsp"), x86.Reg("rbp")]),
                      x86.NamedInstr("subq", [x86.Immediate(stack_size), x86.Reg("rsp")])]
    prelude_instrs = list[x86.Instr](prelude_instrs)

    # pop the stack frame
    conclusion_instrs = [x86.NamedInstr("addq", [x86.Immediate(stack_size), x86.Reg("rsp")]),
                         x86.NamedInstr("popq", [x86.Reg("rbp")]),
                         x86.Retq()]

    main_instrs = prog.blocks["main"]

    # add the prelude and conclusion to the main block
    new_main_block = prelude_instrs + main_instrs + conclusion_instrs
    new_program = x86.X86Program({"main": new_main_block}, stack_space=stack_size)
    return new_program


##################################################
# Compiler definition
##################################################

compiler_passes = {
    'remove complex opera*': rco,
    'select instructions': select_instructions,
    'assign homes': assign_homes,
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
                x86_program = run_compiler(program, logging=True)

                with open(file_name + '.s', 'w') as output_file:
                    output_file.write(x86_program)

            except:
                print('Error during compilation! **************************************************')
                traceback.print_exception(*sys.exc_info())
