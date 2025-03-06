from ExtraMojo.io.buffered import BufferedReader, BufferedWriter

from ishlib.searcher_settings import SearcherSettings
from ishlib.matcher import Matcher
from ishlib import ByteSpanWriter

from utils import StringSlice
from sys import stdout


@value
struct SearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        var reader = BufferedReader(open(file, "r"))
        var buffer = List[UInt8]()

        var writer = BufferedWriter(stdout)

        var line_number = 1
        while True:
            if reader.read_until(buffer) == 0:
                break
            print("CHECKING", ByteSpanWriter(buffer[:]))
            var m = self.matcher.first_match(buffer, self.settings.pattern)
            if m:
                print("FOUND:", m.value().start, "-", m.value().end)
                writer.write(
                    file,
                    ":",
                    line_number,
                    " ",
                    ByteSpanWriter(buffer[0 : m.value().start]),
                    "\033[1;31m",
                    ByteSpanWriter(buffer[m.value().start : m.value().end]),
                    "\033[0m",
                    ByteSpanWriter(buffer[m.value().end :]),
                    "\n",
                )
            line_number += 1
        writer.flush()
        writer.close()
