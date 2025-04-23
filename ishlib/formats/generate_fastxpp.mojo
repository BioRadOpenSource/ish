import sys
from collections import Optional
from ExtraMojo.io.buffered import BufferedReader, BufferedWriter

# ---------- helpers -------------------------------------------------


fn string_count(s: String) -> Int:
    var n: Int = 0
    for _ in s.codepoints():
        n = n + 1
    return n


fn read_line(mut rdr: BufferedReader) raises -> String:
    var buf = List[UInt8]()
    var n = rdr.read_until(buf, ord("\n"))
    if n == 0:
        return ""
    var s = String()
    s.write_bytes(Span(buf))
    return s


# ---------- FASTX++ builder -----------------------------------------


fn generate_fastxpp(
    marker: String,
    header: String,
    seq_lines: List[String],
    qualities: Optional[List[String]] = None,
) -> String:
    var bpl = string_count(seq_lines[0]) + 1  # bases + LF
    var seq_len: Int = 0
    for i in range(len(seq_lines)):
        seq_len = seq_len + string_count(seq_lines[i])

    var meta = String(string_count(header)) + ":" + String(
        seq_len
    ) + ":" + String(len(seq_lines))

    var rec = marker + "`" + meta + "`" + header + "\n"

    for i in range(len(seq_lines)):
        rec.write(seq_lines[i], "\n")

    if qualities:
        var q = qualities.value()
        rec += "+\n"
        for i in range(len(q)):
            rec.write(q[i], "\n")

    return rec


fn generate_fastxpp_bpl(
    marker: String,
    header: String,
    seq_lines: List[String],
    qualities: Optional[List[String]] = None,
) -> String:
    var bpl = string_count(seq_lines[0]) + 1          # bases + LF
    var slen = (bpl - 1) * (len(seq_lines) - 1) +     # (bases per full line)
               string_count(seq_lines[-1])            # + last (ragged) line
    var meta = String(string_count(header)) + ":" +
               String(slen) + ":" +
               String(len(seq_lines)) + ":" +
               String(bpl)
    var rec = marker + "`" + meta + "`" + header + "\n"
    for i in range(len(seq_lines)):
        rec.write(seq_lines[i], "\n")
    if qualities:
        var q = qualities.value()
        for i in range(len(q)):
            rec.write(q[i], "\n")
    return rec


# ---------- main ----------------------------------------------------


fn main() raises:
    var argv = sys.argv()
    if len(argv) != 3:
        print(
            "Usage: mojo run generate_fastxpp.mojo <input.fastx>"
            " <output.fastxpp>"
        )
        return

    var reader = BufferedReader(
        open(String(argv[1]), "r"), buffer_capacity=128 * 1024
    )
    var writer = BufferedWriter(
        open(String(argv[2]), "w"), buffer_capacity=128 * 1024
    )

    var pending_header = String()  # carries a header we already read

    while True:
        var header_line = pending_header
        if header_line == "":
            header_line = read_line(reader)
        pending_header = String()

        if header_line == "":
            break

        var marker = String(header_line[0:1])
        var header = String(header_line[1:])

        var seq = List[String]()
        var line: String

        while True:
            line = read_line(reader)
            if line == "":
                break
            if (
                line.startswith(">")
                or line.startswith("@")
                or (marker == "@" and line.startswith("+"))
            ):
                pending_header = line  # save for the next record
                break
            seq.append(line)

        var qual: Optional[List[String]] = None
        if marker == "@" and line.startswith("+"):
            var qlines = List[String]()
            for _ in range(len(seq)):
                qlines.append(read_line(reader))
            qual = Optional[List[String]](qlines)

        writer.write(generate_fastxpp_bpl(marker, header, seq, qual))

    writer.flush()
    writer.close()
