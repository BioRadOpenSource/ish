import sys
from time.time import perf_counter
from ishlib.vendor.kseq import FastxReader, BufferedReader, KRead


# ── thin wrapper so FileHandle implements KRead ─────────────────────────
struct FileReader(KRead):
    var fh: FileHandle

    fn __init__(out self, owned fh: FileHandle):
        self.fh = fh^

    fn __moveinit__(out self, owned other: Self):
        self.fh = other.fh^

    fn unbuffered_read[
        o: MutableOrigin
    ](mut self, buffer: Span[UInt8, o]) raises -> Int:
        return Int(self.fh.read(buffer.unsafe_ptr(), len(buffer)))


# ────────────────────────────────────────────────────────────────────────


fn bench_original(path: String) raises -> (Int, Int, Float64):
    var fh = open(path, "r")
    var rdr = FastxReader[read_comment=False](BufferedReader(FileReader(fh^)))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while rdr.read() > 0:
        rec += 1
        seq += len(rdr.seq)
    return (rec, seq, perf_counter() - t0)


fn bench_fastxpp(path: String) raises -> (Int, Int, Float64):
    var fh = open(path, "r")
    var rdr = FastxReader[read_comment=False](BufferedReader(FileReader(fh^)))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while True:
        var n = rdr.read_fastxpp()
        if n < 0:
            break
        rec += 1
        seq += n
    return (rec, seq, perf_counter() - t0)


fn main() raises:
    var argv = sys.argv()
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: mojo run fastxpp_bench.mojo <file> [fastxpp]")
        return

    var path = String(argv[1])
    var use_fast = (len(argv) == 3) and (String(argv[2]) == "fastxpp")

    if use_fast:
        var tup = bench_fastxpp(path)  # (Int, Int, Float64)
        var r = tup[0]  # records
        var s = tup[1]  # bases
        var t = tup[2]  # seconds
        print("mode=fastxpp  records=", r, "  bases=", s, "  time=", t, "s")
    else:
        var tup = bench_original(path)
        var r = tup[0]
        var s = tup[1]
        var t = tup[2]
        print("mode=orig     records=", r, "  bases=", s, "  time=", t, "s")
