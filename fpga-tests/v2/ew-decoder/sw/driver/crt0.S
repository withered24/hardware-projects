    .section .text.init
    .globl _start
_start:
    li   sp, 0xF000
    call main
1:  j    1b
