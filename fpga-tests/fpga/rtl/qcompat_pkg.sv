// qcompat_pkg.sv: stand-ins for SystemVerilog built-ins that Quartus Prime
// Standard does not synthesize. tools/qcompat.py rewrites $countones(x) into
// qcompat_pkg::qc_popcount(x); x is zero-extended to 256 bits on the call,
// which doesn't change its count.
package qcompat_pkg;
    function automatic int qc_popcount(input logic [255:0] x);
        int n;
        n = 0;
        for (int i = 0; i < 256; i++) n = n + x[i];
        return n;
    endfunction
endpackage
