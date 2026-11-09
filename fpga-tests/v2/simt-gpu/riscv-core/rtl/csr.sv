// csr.sv — machine-mode control and status registers + trap state
//
// The minimum a machine-mode-only RV32I core needs to run the official
// riscv-tests: a trap vector (mtvec), the registers a trap fills in (mepc,
// mcause, mtval), a scratch register, the interrupt-enable bits of mstatus,
// and the read-only ID registers. Everything is from the privileged spec
// (RISC-V ISA manual Vol. II, "Machine-Level ISA").
//
// Accessing a CSR that isn't implemented, or writing a read-only one, is an
// illegal instruction. The riscv-tests startup code probes CSRs this core
// doesn't have (satp, pmpaddr0, medeleg, ...) and is written to survive
// exactly that: it points mtvec at the next instruction first.

import rv32i_pkg::*;

module csr (
    input  logic        clk,
    input  logic        rst,

    // CSR instruction (CSRRW/S/C and their immediate forms)
    input  logic        csr_en,      // this instruction is a CSR access
    input  logic [2:0]  funct3,
    input  logic [11:0] addr,
    input  logic [31:0] wsrc,        // rs1 value, or zero-extended uimm
    input  logic [4:0]  rs1_field,   // CSRRS/C with rs1 = x0 don't write
    output logic [31:0] rdata,
    output logic        csr_illegal, // unimplemented CSR, or write to read-only

    // traps and returns (from the core)
    input  logic        trap,        // take a trap this cycle
    input  logic [31:0] trap_cause,
    input  logic [31:0] trap_pc,
    input  logic [31:0] trap_val,
    input  logic        mret,

    output logic [31:0] mtvec_base,  // where traps go
    output logic [31:0] mepc_q       // where MRET returns to
);

    localparam logic [11:0] CSR_MSTATUS  = 12'h300;
    localparam logic [11:0] CSR_MISA     = 12'h301;
    localparam logic [11:0] CSR_MIE      = 12'h304;
    localparam logic [11:0] CSR_MTVEC    = 12'h305;
    localparam logic [11:0] CSR_MSCRATCH = 12'h340;
    localparam logic [11:0] CSR_MEPC     = 12'h341;
    localparam logic [11:0] CSR_MCAUSE   = 12'h342;
    localparam logic [11:0] CSR_MTVAL    = 12'h343;
    localparam logic [11:0] CSR_MIP      = 12'h344;
    localparam logic [11:0] CSR_MVENDOR  = 12'hF11;
    localparam logic [11:0] CSR_MARCHID  = 12'hF12;
    localparam logic [11:0] CSR_MIMPID   = 12'hF13;
    localparam logic [11:0] CSR_MHARTID  = 12'hF14;

    localparam logic [31:0] MISA_RV32I = 32'h4000_0100;  // MXL=1 (32-bit), extension I

    // mstatus: only MIE (bit 3) and MPIE (bit 7) are stored. MPP (bits 12:11)
    // always reads 2'b11 because machine mode is the only privilege level.
    logic        mie_q, mpie_q;
    logic [31:0] mtvec_q, mscratch_q, mcause_q, mtval_q;

    logic [31:0] mstatus;
    assign mstatus = {19'd0, 2'b11, 3'd0, mpie_q, 3'd0, mie_q, 3'd0};

    // ---- read ----
    logic known;
    always_comb begin
        known = 1'b1;
        case (addr)
            CSR_MSTATUS:  rdata = mstatus;
            CSR_MISA:     rdata = MISA_RV32I;
            CSR_MIE:      rdata = 32'd0;      // no interrupts implemented
            CSR_MTVEC:    rdata = mtvec_q;
            CSR_MSCRATCH: rdata = mscratch_q;
            CSR_MEPC:     rdata = mepc_q;
            CSR_MCAUSE:   rdata = mcause_q;
            CSR_MTVAL:    rdata = mtval_q;
            CSR_MIP:      rdata = 32'd0;
            CSR_MVENDOR, CSR_MARCHID, CSR_MIMPID, CSR_MHARTID: rdata = 32'd0;
            default: begin
                rdata = 32'd0;
                known = 1'b0;
            end
        endcase
    end

    // ---- write ----
    // CSRRW always writes. CSRRS/CSRRC only write when the source field is
    // nonzero, so "csrr rd, csr" (= csrrs rd, csr, x0) is a pure read, which
    // matters for read-only CSRs.
    logic        wants_write;
    logic [31:0] wdata;

    always_comb begin
        case (funct3[1:0])
            2'b01:   begin wants_write = 1'b1;               wdata = wsrc;          end  // RW
            2'b10:   begin wants_write = (rs1_field != 5'd0); wdata = rdata | wsrc;  end  // RS
            default: begin wants_write = (rs1_field != 5'd0); wdata = rdata & ~wsrc; end  // RC
        endcase
    end

    assign csr_illegal = csr_en && (!known || (wants_write && addr[11:10] == 2'b11));

    logic do_write;
    assign do_write = csr_en && wants_write && !trap;

    always_ff @(posedge clk) begin
        if (rst) begin
            mie_q      <= 1'b0;
            mpie_q     <= 1'b0;
            mtvec_q    <= 32'd0;
            mscratch_q <= 32'd0;
            mepc_q     <= 32'd0;
            mcause_q   <= 32'd0;
            mtval_q    <= 32'd0;
        end else if (trap) begin
            mepc_q   <= trap_pc;
            mcause_q <= trap_cause;
            mtval_q  <= trap_val;
            mpie_q   <= mie_q;
            mie_q    <= 1'b0;
        end else if (mret) begin
            mie_q  <= mpie_q;
            mpie_q <= 1'b1;
        end else if (do_write) begin
            case (addr)
                CSR_MSTATUS:  begin mie_q <= wdata[3]; mpie_q <= wdata[7]; end
                CSR_MTVEC:    mtvec_q    <= {wdata[31:2], 2'b00};  // direct mode only
                CSR_MSCRATCH: mscratch_q <= wdata;
                CSR_MEPC:     mepc_q     <= {wdata[31:2], 2'b00};  // IALIGN = 32
                CSR_MCAUSE:   mcause_q   <= wdata;
                CSR_MTVAL:    mtval_q    <= wdata;
                default: ;                                        // misa, mie, mip: WARL, read-only zero/const
            endcase
        end
    end

    assign mtvec_base = mtvec_q;

endmodule
