"""Peek at a file to check for:
1. Whether or not it is binary.
2. Get some stats on the sequence lengths if we have a gpu attached.
"""

from bit import next_power_of_two
from pathlib import Path

from ExtraMojo.bstr.memchr import memchr_wide, memchr

from ishlib.vendor.zlib import GZFile
from ishlib.vendor.kseq import KRead, BufferedReader, FastxReader
from ishlib.vendor.running_stats import RunningStats

alias PEEK_SIZE: Int = 8192  # This is the initial read size done by zlib
alias DEFAULT_MAX_LENGTH = 1024


@value
struct PeekFindings:
    var is_binary: Bool
    var suggested_max_length: UInt

    fn __init__(out self):
        self.is_binary = False
        self.suggested_max_length = DEFAULT_MAX_LENGTH


fn peek_file[
    *,
    record_type: RecordType,
    peek_size: Int = PEEK_SIZE,
    check_record_size: Bool = True,
](file: Path) raises -> PeekFindings:
    var fh = GZFile(String(file), "rb")
    var buffer = InlineArray[UInt8, size=peek_size](fill=0)

    var bytes_read = fh.unbuffered_read(buffer)
    if bytes_read < 0:
        raise "Error reading file."
    if bytes_read == 0:
        return PeekFindings(
            is_binary=False, suggested_max_length=DEFAULT_MAX_LENGTH
        )
    var filled_buffer = Span(buffer)[0:bytes_read]

    # Check if is_binary
    var pos = memchr_wide(filled_buffer, 0)
    var is_binary = pos >= 0

    @parameter
    if not check_record_size:
        return PeekFindings(
            is_binary=is_binary, suggested_max_length=DEFAULT_MAX_LENGTH
        )

    var suggested_max_length = DEFAULT_MAX_LENGTH

    @parameter
    if record_type == RecordType.LINE:
        var stats = RunningStats[DType.uint32]()

        var start = 0
        while start <= len(filled_buffer):
            var newline = memchr[do_alignment=False](
                filled_buffer, ord("\n"), start=start
            )
            if newline < 0:
                break
            stats.push(newline - start)
            start += newline + 1
        suggested_max_length = next_power_of_two(
            Int(stats.mean() + stats.standard_deviation())
        )
    elif record_type == RecordType.FASTA:
        var stats = RunningStats[DType.uint32]()
        var buffer_span = filled_buffer.get_immutable()
        var reader = FastxReader(
            BufferedReader(SpanReader(buffer_span, index=0))
        )
        while reader.read() > 0:
            stats.push(len(reader.seq))
        suggested_max_length = next_power_of_two(
            Int(stats.mean() + stats.standard_deviation())
        )
    else:
        raise "Unknown record type"

    return PeekFindings(
        is_binary=is_binary, suggested_max_length=suggested_max_length
    )


@value
struct SpanReader[origin: ImmutableOrigin](KRead):
    var data: Span[UInt8, origin]
    var index: Int

    fn unbuffered_read[
        o: MutableOrigin
    ](mut self, buffer: Span[UInt8, o]) raises -> Int:
        var bytes_read = 0
        for i in range(0, min(len(buffer), len(self.data[self.index :]))):
            buffer[i] = self.data[self.index]
            self.index += 1
            bytes_read += 1
        return bytes_read
