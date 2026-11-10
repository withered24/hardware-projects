# mock_stp.tcl: stand-in for quartus_stp's JTAG memory commands, to test
# replay.tcl without a board:  quartus_sh -t mock_stp.tcl <plan> [order]
# Emulates the replay bitstream: RAMs by name, content strings in the given
# order (descending by default, to exercise calibration), STAT words, and a
# "DUT" whose response word k at entry i is stimulus word (k mod NIW) at i
# XOR k, run when a new token lands in CTRL[0].
set plan [lindex $quartus(args) 0]
set MOCK_ORDER [expr {[llength $quartus(args)] > 1 ? [lindex $quartus(args) 1] : "descending"}]
set fh [open $plan r]; set pd [read $fh]; close $fh
set names {CTRL STAT}
set sid 0
foreach line [split $pd "\n"] {
    set f [regexp -all -inline {\S+} $line]
    if {[lindex $f 0] in {S R}} { set names [concat $names [lrange $f 1 end]] }
    if {[lindex $f 0] eq "SUITE_ID"} { set sid [lindex $f 1] }
    if {[lindex $f 0] eq "S"} { set ::SN [lrange $f 1 end] }
    if {[lindex $f 0] eq "R"} { set ::RN [lrange $f 1 end] }
}
array set M {}
set i 0
foreach n $names { set ::IDX($i) $n; set ::NAMEIDX($n) $i; incr i }
proc mget {n a} { if {[info exists ::M($n,$a)]} { return $::M($n,$a) } else { return 0 } }
set ::M(STAT,2) [expr 0x5EB1A701]
set ::M(STAT,3) [expr 0x$sid]
set ::LASTTOK 0
proc get_hardware_names {} { return [list "USB-Blaster \[USB-0\]"] }
proc get_device_names {args} { return [list "@1: EP3C120/EP4CE115 (0x020F70DD)"] }
proc begin_memory_edit {args} {}
proc end_memory_edit {args} {}
proc get_editable_mem_instances {args} {
    set r {}
    foreach i [lsort -integer [array names ::IDX]] { lappend r [list $i 2048 32 Read/Write RAM/ROM $::IDX($i)] }
    return $r
}
proc opt {args key} { set k [lsearch $args $key]; return [lindex $args [expr {$k + 1}]] }
proc write_content_to_memory {args} {
    set n $::IDX([opt $args -instance_index]); set a [opt $args -start_address]
    set cnt [opt $args -word_count]; set c [opt $args -content]
    for {set k 0} {$k < $cnt} {incr k} {
        set w [expr 0x[string range $c [expr {8*$k}] [expr {8*$k+7}]]]
        set addr [expr {$::MOCK_ORDER eq "ascending" ? $a + $k : $a + $cnt - 1 - $k}]
        set ::M($n,$addr) $w
    }
    if {$n eq "CTRL" && $a == 0 && [mget CTRL 0] != $::LASTTOK && [mget CTRL 1] != 0} {
        set ::LASTTOK [mget CTRL 0]
        set cnt [mget CTRL 1]
        for {set i 0} {$i < $cnt} {incr i} {
            set k 0
            foreach r $::RN {
                set s [lindex $::SN [expr {$k % [llength $::SN]}]]
                set ::M($r,$i) [expr {[mget $s $i] ^ $k}]
                incr k
            }
        }
        set ::M(STAT,1) [expr {[mget STAT 1] + $cnt}]
        set ::M(STAT,0) $::LASTTOK
    }
}
proc read_content_from_memory {args} {
    set n $::IDX([opt $args -instance_index]); set a [opt $args -start_address]
    set cnt [opt $args -word_count]
    set s ""
    for {set k 0} {$k < $cnt} {incr k} {
        set addr [expr {$::MOCK_ORDER eq "ascending" ? $a + $k : $a + $cnt - 1 - $k}]
        append s [format %08X [mget $n $addr]]
    }
    return $s
}
source [file join [file dirname [info script]] .. replay.tcl]
