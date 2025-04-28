import sys
from time.time import perf_counter
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile
from ExtraMojo.utils.ir import dump_ir


# ── thin wrapper so FileHandle implements KRead ─────────────────────────
# struct FileReader(KRead):
#     var fh: FileHandle

#     fn __init__(out self, owned fh: FileHandle):
#         self.fh = fh^

#     fn __moveinit__(out self, owned other: Self):
#         self.fh = other.fh^

#     fn unbuffered_read[
#         o: MutableOrigin
#     ](mut self, buffer: Span[UInt8, o]) raises -> Int:
#         return Int(self.fh.read(buffer.unsafe_ptr(), len(buffer)))


# ────────────────────────────────────────────────────────────────────────


fn bench_original(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while rdr.read() > 0:
        rec += 1
        seq += len(rdr.seq)
    return (rec, seq, perf_counter() - t0)


fn bench_fastxpp(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
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


fn bench_fastxpp_bpl(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while True:
        var n = rdr.read_fastxpp_bpl()
        if n < 0:
            break
        rec += 1
        seq += n
    return (rec, seq, perf_counter() - t0)


fn bench_fastxpp_swar(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while True:
        var n = rdr.read_fastxpp_swar()
        if n < 0:
            break
        rec += 1
        seq += n
    return (rec, seq, perf_counter() - t0)


fn bench_fastxpp_read_once(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while True:
        var n = rdr.read_fastxpp_read_once()
        if n < 0:
            break
        rec += 1
        seq += n
    return (rec, seq, perf_counter() - t0)


fn bench_fastxpp_bpl2(path: String) raises -> (Int, Int, Float64):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    var rec = 0
    var seq = 0
    var t0 = perf_counter()
    while True:
        var n = rdr.read_fastxpp_bpl()
        if n < 0:
            break
        rec += 1
        seq += n
    return (rec, seq, perf_counter() - t0)


fn main() raises:
    var argv = sys.argv()
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: mojo run fastxpp_bench.mojo <file> [orig|fastxpp|bpl]")
        return

    var path = String(argv[1])
    var mode: String = "orig"  # default when no flag given
    if len(argv) == 3:
        mode = String(argv[2])

    if mode == "orig":
        r, s, t = bench_original(path)
        print(
            "mode=orig          records=", r, "  bases=", s, "  time=", t, "s"
        )
    elif mode == "bpl":
        r, s, t = bench_fastxpp_bpl(path)
        print(
            "mode=fastxpp_bpl   records=", r, "  bases=", s, "  time=", t, "s"
        )
    elif mode == "fastxpp":
        r, s, t = bench_fastxpp(path)
        print(
            "mode=fastxpp       records=", r, "  bases=", s, "  time=", t, "s"
        )
    elif mode == "swar":
        r, s, t = bench_fastxpp_swar(path)
        print(
            "mode=fastxpp_swar   records=", r, "  bases=", s, "  time=", t, "s"
        )
    elif mode == "read_once":
        r, s, t = bench_fastxpp_read_once(path)
        print("mode=read_once   records=", r, "  bases=", s, "  time=", t, "s")
    elif mode == "filler":
        r, s, t = bench_fastxpp_bpl2(path)
        print(
            "mode=fastxpp_bpl   records=", r, "  bases=", s, "  time=", t, "s"
        )
    else:
        print("Unknown mode:", mode)
