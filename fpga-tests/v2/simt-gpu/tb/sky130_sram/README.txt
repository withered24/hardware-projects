OpenRAM simulation models of the SkyWater SKY130 SRAM macros, copied from the
PDK (sky130A/libs.ref/sky130_sram_macros/verilog, Apache-2.0). Two edits:
VERBOSE defaults to 0 (otherwise every access prints), and the `mem`
declaration is moved above the $display that reads it, which Icarus requires.
