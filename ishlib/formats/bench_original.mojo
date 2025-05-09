import sys
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile


@no_inline
fn bench_original(path: String) raises -> (Int, Int, Int):
    # Read the file using 'read' calls, do nothing with the data
    var rdr = FastxReader[read_comment=False](BufferedReader(GZFile(path, "r")))
    while rdr.read() > 0:
        pass
    return (0, 0, 0)


fn main() raises:
    var argv = sys.argv()
    if len(argv) < 2:
        print("Usage: bench_original <file>")
        return

    var path = String(argv[1])

    r, s, t = bench_original(path)
