from ExtraMojo.io.buffered import BufferedWriter

from ishlib.vendor.kseq import BufferedReader, SearchChar, ByteString
from ishlib.vendor.zlib import GZFile
from ishlib.searcher_settings import SearcherSettings
from ishlib.matcher import Matcher
from ishlib import ByteSpanWriter

from utils import StringSlice
from sys import stdout


@value
struct LineSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        var reader = BufferedReader(GZFile(file, "r"))
        # var buffer = List[UInt8]()
        var buffer = ByteString()

        var writer = BufferedWriter(stdout)

        var line_number = 1
        while True:
            buffer.clear()
            if reader.read_until[SearchChar.Newline](buffer) == 0:
                break
            var m = self.matcher.first_match(
                buffer.as_span(), self.settings.pattern
            )
            if m:
                var b = buffer.as_span()
                writer.write_bytes(file.as_bytes())
                writer.write_bytes(":".as_bytes())
                writer.write(line_number)
                writer.write(" ")
                writer.write_bytes(b[0 : m.value().start])
                writer.write("\033[1;31m")
                writer.write_bytes(b[m.value().start : m.value().end])
                writer.write("\033[0m")
                writer.write_bytes(b[m.value().end :])
                writer.write("\n")
            line_number += 1
        writer.flush()
        writer.close()
