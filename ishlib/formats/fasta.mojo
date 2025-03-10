from utils import StringSlice

from ExtraMojo.io.buffered import BufferedReader
from ExtraMojo.bstr.memchr import memchr_wide
from ExtraMojo.bstr.bstr import to_ascii_uppercase


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
        return memchr_wide(self.seq.as_bytes(), ord("N")) != -1

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
            var header_line_end = memchr_wide(buffer, ord("\n"))
            if header_line_end == -1:
                print(StringSlice(unsafe_from_utf8=buffer))
                raise "No newline found for FASTA record header"

            var name = String()
            name.write_bytes(buffer[0:header_line_end])

            var seq = String()
            # Now copy in the seq bytes, removing newlines
            var start = header_line_end + 1
            var end = memchr_wide(buffer, ord("\n"), start)
            while end != -1:
                seq.write_bytes(buffer[start:end])
                start = end + 1
                end = memchr_wide(buffer, ord("\n"), start)
            records.append(FastaRecord(name, seq))

        return records
