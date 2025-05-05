import sys
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile


@no_inline
fn bench_fastxpp_swar(path: String) raises -> (Int, Int, Int):
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    while True:
        var n = rdr.read_fastxpp_swar()
        if n < 0:
            break
    return (0, 0, 0)


fn main() raises:
    var argv = sys.argv()
    if len(argv) < 2:
        print("Usage: bench_fastxpp_swar <file>")
        return

    var path = String(argv[1])
    r, s, t = bench_fastxpp_swar(path)
