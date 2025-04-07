from ExtraMojo.io.buffered import BufferedReader, BufferedWriter

from ishlib.formats.fasta import (
    FastaReader,
    BorrowedFastaRecord,
    ByteFastaRecord,
)
from ishlib.searcher_settings import SearcherSettings
from ishlib.matcher import Matcher, MatchResult
from ishlib import ByteSpanWriter

from algorithm.functional import parallelize
from utils import StringSlice
from sys import stdout, info


@value
struct ParallelLineSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        # TODO: pass an enocoder to the FastaReader
        var reader = BufferedReader(open(file, "r"))
        var writer = BufferedWriter(stdout)

        var lines = List[List[UInt8]]()
        var bytes_saved = 0
        var buffer = List[UInt8]()
        var line_number = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var do_work = True
        while do_work:
            buffer.clear()
            if reader.read_until(buffer) == 0:
                do_work = False
            else:
                var line = List[UInt8](capacity=len(buffer))
                for i in range(0, len(buffer)):
                    line.append(
                        self.matcher.convert_ascii_to_encoding(buffer[i])
                    )
                bytes_saved += len(line)
                lines.append(line)

            if bytes_saved >= self.settings.batch_size or not do_work:
                var outputs = self.par(lines)
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer.address_of(lines[i])
                    writer.write(
                        file,
                        ":",
                        line_number,
                        " ",
                        # ByteSpanWriter(buffer[:]),
                        ByteSpanWriter(r[][0 : m.value().start]),
                        "\033[1;31m",
                        ByteSpanWriter(r[][m.value().start : m.value().end]),
                        "\033[0m",
                        ByteSpanWriter(r[][m.value().end :]),
                        "\n",
                    )
                    line_number += 1
                lines.clear()
                bytes_saved = 0
        writer.flush()
        writer.close()

    fn par(
        read self, read seqs: Span[List[UInt8]]
    ) -> List[Optional[MatchResult]]:
        # TODO: reusable output buffer
        var output = List[Optional[MatchResult]](capacity=len(seqs))
        for _ in range(0, len(seqs)):
            output.append(None)

        fn do_matching(index: Int) capturing:
            var target = Pointer.address_of(seqs[index])
            output[index] = self.matcher.first_match(
                target[], self.settings.pattern
            )

        parallelize[do_matching](len(seqs), self.settings.threads)
        return output
