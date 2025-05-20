from collections import Optional
from sys.info import sizeof

from ExtraMojo.io.buffered import BufferedReader
from ExtraMojo.bstr.memchr import memchr
from ExtraMojo.bstr.bstr import to_ascii_uppercase

from ishlib.matcher import Matcher
from ishlib.gpu.searcher_device import Searchable


@value
struct ByteFastaRecord(Searchable):
    var name: List[UInt8]
    var seq: List[UInt8]

    fn size_in_bytes(read self) -> UInt:
        return sizeof[Self]() + len(self.name) + len(self.seq)

    fn buffer_to_search(ref self) -> Span[UInt8, __origin_of(self)]:
        # This lifetime of seq is at least as long as self
        return rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))
        # This also works
        # return Span[UInt8, __origin_of(self)](
        #     ptr=self.seq.unsafe_ptr(), length=len(self.seq)
        # )


@value
struct FastaRecord:
    var name: String
    var seq: String

    fn __str__(read self) -> String:
        var out = String()
        out += ">"
        self.name.write_to(out)
        out += "\n"
        self.seq.write_to(out)
        out += "\n"
        return out

    fn seq_contains_n(read self) -> Bool:
        return memchr(self.seq.as_bytes(), ord("N")) != -1

    fn uppercase_seq(mut self):
        to_ascii_uppercase(self.seq._buffer)

    @staticmethod
    fn slurp_fasta(file: String) raises -> List[FastaRecord]:
        var reader = BufferedReader(open(file, "r"))
        var records = List[FastaRecord]()

        var buffer = List[UInt8]()
        var bytes_read = reader.read_until(buffer, ord(">"))
        if bytes_read != 1:
            print(bytes_read)
            raise String.write("File should start with '>': ", len(buffer))

        while True:
            buffer.clear()
            while reader.read_until(buffer, ord(">")) > 0:
                if buffer[-1] != ord("\n"):
                    continue
                else:
                    break
            if len(buffer) == 0:
                break

            # Find the first newline in the buffer
            var header_line_end = memchr(buffer, ord("\n"))
            if header_line_end == -1:
                print(StringSlice(unsafe_from_utf8=buffer))
                raise "No newline found for FASTA record header"

            var name = String()
            name.write_bytes(buffer[0:header_line_end])

            var seq = String()
            # Now copy in the seq bytes, removing newlines
            var start = header_line_end + 1
            var end = memchr(buffer, ord("\n"), start)
            while end != -1:
                seq.write_bytes(buffer[start:end])
                start = end + 1
                end = memchr(buffer, ord("\n"), start)
            records.append(FastaRecord(name, seq))

        return records


@value
struct BorrowedFastaRecord[
    mut: Bool, //,
    name_origin: Origin[mut],
    seq_origin: Origin[mut],
    original_origin: Origin[mut],
]:
    var name: Span[UInt8, name_origin]
    var seq: Span[UInt8, seq_origin]
    var original_seq: Span[UInt8, original_origin]
    var original_seq_start: Int


struct FastaReader:
    var reader: BufferedReader
    var buffer: List[UInt8]
    var header_buffer: List[UInt8]
    var seq_buffer: List[UInt8]

    fn __init__(out self, owned reader: BufferedReader) raises:
        self.reader = reader^
        self.buffer = List[UInt8]()
        self.header_buffer = List[UInt8]()
        self.seq_buffer = List[UInt8]()

        # get the first header delim out of the way
        var bytes_read = self.reader.read_until(self.buffer, ord(">"))
        if bytes_read != 1:
            raise String.write("File should start with '>': ", len(self.buffer))

    fn __moveinit__(out self, owned existing: Self):
        self.reader = existing.reader^
        self.buffer = existing.buffer^
        self.seq_buffer = existing.seq_buffer^
        self.header_buffer = existing.header_buffer^

    fn read_borrowed(
        mut self,
    ) raises -> Optional[
        BorrowedFastaRecord[
            __origin_of(self.header_buffer),
            __origin_of(self.seq_buffer),
            __origin_of(self.buffer),
        ]
    ]:
        self.buffer.clear()
        self.header_buffer.clear()
        self.seq_buffer.clear()
        while self.reader.read_until(self.buffer, ord(">")) > 0:
            # If the next char isn't a newline, this isn't actually the end of the header
            if self.buffer[-1] != ord("\n"):
                continue
            else:
                break

        if len(self.buffer) == 0:
            return None

        # Find the first newline in the buffer
        var header_line_end = memchr(self.buffer, ord("\n"))
        if header_line_end == -1:
            raise "No newline found for FASTA record header"

        # Should be able to retrurn a slice of self.buffer, but that isn't working for some reason
        for i in range(0, header_line_end):
            self.header_buffer.append(self.buffer[i])

        # Now copy in the seq bytes, removing newlines
        var start = header_line_end + 1
        var end = memchr(self.buffer, ord("\n"), start)
        while end != -1:
            # TODO: switch to memcpy
            for i in range(start, end):
                self.seq_buffer.append(self.buffer[i])
            start = end + 1
            end = memchr(self.buffer, ord("\n"), start)

        return BorrowedFastaRecord(
            Span(self.header_buffer),
            Span(self.seq_buffer),
            Span(self.buffer),
            header_line_end + 1,
        )

    # TODO: move the encoding from Matcher to its own encoder trait

    fn read_owned[
        M: Matcher
    ](mut self, read encoder: M) raises -> Optional[ByteFastaRecord]:
        self.buffer.clear()

        # TODO: reserve cap based on last seen sequence?
        var header = List[UInt8]()
        var seq = List[UInt8]()

        while self.reader.read_until(self.buffer, ord(">")) > 0:
            # If the next char isn't a newline, this isn't actually the end of the header
            if self.buffer[-1] != ord("\n"):
                continue
            else:
                break

        if len(self.buffer) == 0:
            return None

        # Find the first newline in the buffer
        var header_line_end = memchr(self.buffer, ord("\n"))
        if header_line_end == -1:
            raise "No newline found for FASTA record header"

        # Should be able to retrurn a slice of self.buffer, but that isn't working for some reason
        for i in range(0, header_line_end):
            header.append(self.buffer[i])

        # Now copy in the seq bytes, removing newlines
        var start = header_line_end + 1
        var end = memchr(self.buffer, ord("\n"), start)
        while end != -1:
            # TODO: switch to memcpy
            for i in range(start, end):
                seq.append(encoder.convert_ascii_to_encoding(self.buffer[i]))
                # seq.append(self.buffer[i])
            start = end + 1
            end = memchr(self.buffer, ord("\n"), start)

        return ByteFastaRecord(header, seq)
