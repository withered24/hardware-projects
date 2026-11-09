OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
    . = 0x0;
    .text : { *(.text.init) *(.text*) }
    .rodata : { *(.rodata*) *(.srodata*) }
    .data : { *(.data*) *(.sdata*) }
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
    ASSERT(. < 0x8000, "program overlaps the job table at 0x8000")
}
