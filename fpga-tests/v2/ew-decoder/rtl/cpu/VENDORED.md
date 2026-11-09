# Vendored CPU

These files are riscv-core's `rtl/` at commit `48a8279` (sibling repo
`../riscv-core`), copied so this project can extend the core without
touching the flagship repo or its formal proofs.

The one change: a coprocessor port for custom-0 instructions.
- `rv32i_pkg.sv`: `OP_CUSTOM0`, and a `cop` bit in `ctrl_t`
- `decoder.sv`: custom-0 with funct7 = 0 decodes as `cop` (reads rs1/rs2, writes rd)
- `core.sv`: `cop_*` ports; the instruction is issued from MEM, freezes the
  pipe like `dmem_wait` until `cop_ready`, writes `cop_rdata` to rd, and is in
  the load-use stall. See the "Coprocessor" paragraph in core.sv's header.

`diff -r ../riscv-core/rtl rtl/cpu` shows exactly this. The `ifdef FORMAL`
properties in core.sv are upstream's and have not been re-proved with the
change.
