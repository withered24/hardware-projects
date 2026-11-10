# replay.tcl: drive a programmed replay bitstream over JTAG (quartus_stp -t)
#
#   quartus_stp -t replay.tcl <plan file>
#
# The plan file (written by replay.py) has one item per line:
#   SUITE_ID <8 hex digits>     the id the bitstream must report in STAT[3]
#   S <S00> <S01> ...           stimulus RAM names, word 0 first
#   R <R00> <R01> ...           response RAM names
#   CHUNK <in dir> <out dir> <count>
# For each chunk: write <in dir>/s00.hex.. (one 32-bit hex word per line) into
# the stimulus RAMs, write the count to CTRL[1] and a new token to CTRL[0],
# wait until STAT[0] echoes the token, and read <count> words of every
# response RAM into <out dir>/r00.hex... With no CHUNK lines it only checks
# the JTAG path (the selftest). Prints REPLAY-OK at the end, or REPLAY-FAIL.
#
# The RAMs are reached through the In-System Memory Content Editor by their
# INSTANCE_NAME. Its multi-word content strings put one end of the address
# range first; which end is calibrated at start-up on S00 entries 0 and 1.

set MAGIC 5EB1A701

proc fail {msg} {
    puts "REPLAY-FAIL $msg"
    catch {end_memory_edit}
    exit 1
}

proc read_file {path} {
    set fh [open $path r]
    set d [read $fh]
    close $fh
    return $d
}

# ------------------------------------------------------------------ plan
if {[llength $quartus(args)] < 1} { fail "usage: quartus_stp -t replay.tcl <plan>" }
set plan [lindex $quartus(args) 0]
set suite_id ""
set s_names {}
set r_names {}
set chunks {}
foreach line [split [read_file $plan] "\n"] {
    set f [regexp -all -inline {\S+} $line]
    if {[llength $f] == 0} continue
    switch -- [lindex $f 0] {
        SUITE_ID { set suite_id [string toupper [lindex $f 1]] }
        S        { set s_names [lrange $f 1 end] }
        R        { set r_names [lrange $f 1 end] }
        CHUNK    { lappend chunks [lrange $f 1 3] }
    }
}

# ------------------------------------------------------------------ connect
set hw ""
foreach h [get_hardware_names] {
    if {[string match -nocase "*USB-Blaster*" $h]} { set hw $h; break }
}
if {$hw eq ""} { fail "no USB-Blaster found (get_hardware_names: [get_hardware_names])" }
set dev ""
foreach d [get_device_names -hardware_name $hw] {
    if {[string match -nocase "*EP4CE115*" $d]} { set dev $d; break }
}
if {$dev eq ""} { fail "no EP4CE115 on $hw (devices: [get_device_names -hardware_name $hw])" }
puts "jtag: $hw / $dev"

if {[catch {begin_memory_edit -hardware_name $hw -device_name $dev} err]} {
    fail "begin_memory_edit: $err"
}

# name -> instance index
set wanted [concat CTRL STAT $s_names $r_names]
array set idx {}
set seen {}
foreach inst [get_editable_mem_instances -hardware_name $hw -device_name $dev] {
    set i [lindex $inst 0]
    lappend seen [lindex $inst end]
    foreach field [lrange $inst 1 end] {
        if {[lsearch -exact $wanted $field] >= 0} { set idx($field) $i }
    }
}
puts "jtag: [llength $seen] memory instances: $seen"
foreach n $wanted {
    if {![info exists idx($n)]} {
        fail "memory instance $n not found (have: $seen); wrong bitstream?"
    }
}

proc wr {name addr words} {
    # words: list of 8-digit hex strings for addr, addr+1, ...
    global idx ASCENDING
    if {$ASCENDING} { set c [join $words ""] } else { set c [join [lreverse $words] ""] }
    write_content_to_memory -instance_index $idx($name) -start_address $addr \
        -word_count [llength $words] -content $c -content_in_hex
}

proc rd {name addr count} {
    # returns a list of 8-digit hex strings for addr, addr+1, ...
    global idx ASCENDING
    set c [read_content_from_memory -instance_index $idx($name) -start_address $addr \
        -word_count $count -content_in_hex]
    set c [string toupper [string trim $c]]
    if {[string length $c] != 8 * $count} {
        fail "read $name: got [string length $c] hex digits for $count words"
    }
    set w {}
    for {set k 0} {$k < $count} {incr k} { lappend w [string range $c [expr {8*$k}] [expr {8*$k+7}]] }
    if {$ASCENDING} { return $w } else { return [lreverse $w] }
}

proc rd1 {name addr} {
    # one word; STAT is rewritten by the design every 4 base cycles, so read
    # until two reads agree
    for {set t 0} {$t < 8} {incr t} {
        set a [lindex [rd $name $addr 1] 0]
        set b [lindex [rd $name $addr 1] 0]
        if {$a eq $b} { return $a }
    }
    fail "unstable read of $name\[$addr\]"
}

proc stat {k} {
    # STAT[k] as an integer
    scan [rd1 STAT $k] %x v
    return $v
}

# ------------------------------------------------------------------ calibrate
set ASCENDING 1
set cal [lindex $s_names 0]
wr $cal 0 {A5A5F00D}
wr $cal 1 {0123BEEF}
set both [string toupper [string trim [read_content_from_memory -instance_index $idx($cal) \
    -start_address 0 -word_count 2 -content_in_hex]]]
if {$both eq "A5A5F00D0123BEEF"} {
    set ASCENDING 1
} elseif {$both eq "0123BEEFA5A5F00D"} {
    set ASCENDING 0
} else {
    fail "calibration: wrote A5A5F00D, 0123BEEF to $cal\[0..1\], read back $both"
}
puts "jtag: content order [expr {$ASCENDING ? {ascending} : {descending}}]"

set magic [rd1 STAT 2]
set sid   [rd1 STAT 3]
if {$magic ne $MAGIC} { fail "STAT\[2\] = $magic, want $MAGIC: not a replay bitstream" }
if {$suite_id ne "" && $sid ne $suite_id} {
    fail "STAT\[3\] = $sid, plan wants $suite_id: the bitstream is for another suite or build"
}
puts "jtag: replay bitstream ok, suite id $sid, DUT edges so far [stat 1]"

# ------------------------------------------------------------------ chunks
set tok [stat 0]
set n 0
foreach ch $chunks {
    lassign $ch cdir odir cnt
    set t0 [clock milliseconds]
    foreach name $s_names {
        set words {}
        foreach line [split [read_file [file join $cdir "[string tolower $name].hex"]] "\n"] {
            set line [string trim $line]
            if {$line ne ""} { lappend words [string toupper $line] }
        }
        if {[llength $words] != $cnt} { fail "$cdir/$name: [llength $words] words, want $cnt" }
        wr $name 0 $words
    }
    set tok [expr {($tok + 1) & 0xFFFFFFFF}]
    if {$tok == 0} { set tok 1 }
    set edges0 [stat 1]
    wr CTRL 1 [list [format %08X $cnt]]
    wr CTRL 0 [list [format %08X $tok]]
    set waited 0
    while {[stat 0] != $tok} {
        after 5
        incr waited 5
        if {$waited > 60000} { fail "chunk $cdir: no done token after 60 s" }
    }
    set edges [expr {[stat 1] - $edges0}]
    if {$edges != $cnt} { fail "chunk $cdir: the DUT got $edges clock edges, want $cnt" }
    file mkdir $odir
    foreach name $r_names {
        set fh [open [file join $odir "[string tolower $name].hex"] w]
        fconfigure $fh -translation lf
        puts -nonewline $fh "[join [rd $name 0 $cnt] "\n"]\n"
        close $fh
    }
    incr n
    puts "chunk $n/[llength $chunks]: $cnt edges, [expr {[clock milliseconds] - $t0}] ms"
}

end_memory_edit
puts "REPLAY-OK [llength $chunks] chunks"
exit 0
