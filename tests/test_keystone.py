import os
import logging
import sys

import unittest

from common import skip_if_not_linux

import angr

l = logging.getLogger("angr.tests")
test_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'binaries', 'tests')

insn_texts = {
    'i386': b"add eax, 0xf",
    'x86_64': b"add rax, 0xf",
    'ppc': b"addi %r1, %r1, 0xf",
    'armel': b"add r1, r1, 0xf",
    'armel_thumb': b"add.w r1, r1, #0xf",
    'mips': b"addi $1, $1, 0xf"
}


class TestKeystone(unittest.TestCase):
    def run_keystone(self, arch):
        proj_arch = arch
        is_thumb = False
        if arch == "armel_thumb":
            is_thumb = True
            proj_arch = "armel"
        p = angr.Project(os.path.join(test_location, proj_arch, "fauxware"), auto_load_libs=False)
        addr = p.loader.main_object.get_symbol('authenticate').rebased_addr

        sm = p.factory.simulation_manager()
        if arch in ['i386', 'x86_64']:
            sm.one_active.regs.eax = 3
        else:
            sm.one_active.regs.r1 = 3

        if is_thumb:
            addr |= 1
        block = p.factory.block(addr, insn_text=insn_texts[arch], thumb=is_thumb).vex

        assert block.instructions == 1

        sm.step(force_addr=addr, insn_text=insn_texts[arch], thumb=is_thumb)

        if arch in ['i386', 'x86_64']:
            assert sm.one_active.solver.eval(sm.one_active.regs.eax) == 0x12
        else:
            assert sm.one_active.solver.eval(sm.one_active.regs.r1) == 0x12

    @skip_if_not_linux
    def test_keystone(self):

        # Installing keystone on Windows is currently a pain. Fix the installation first (may it pip installable) before
        # re-enabling this test on Windows.
        if not sys.platform.startswith('linux'):
            return

        for arch_name in insn_texts:
            yield self.run_keystone, arch_name

if __name__ == "__main__":
    for arch_name in insn_texts:
        print(arch_name)
    unittest.main()

