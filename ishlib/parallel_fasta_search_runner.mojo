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
struct ParallelFastaSearchRunner[M: Matcher]:
    var settings: SearcherSettings
    var matcher: M

    fn run_search(mut self) raises:
        # Simple thing first?
        for file in self.settings.files:
            var f = file[]  # force copy
            self.run_search_on_file(f)

    fn run_search_on_file(mut self, file: String) raises:
        # TODO: pass an enocoder to the FastaReader
        var reader = FastaReader(BufferedReader(open(file, "r")))
        var writer = BufferedWriter(stdout)

        var sequences = List[ByteFastaRecord]()
        var bytes_saved = 0

        # TODO: hold onto the non-newline stripped sequence as well for outputting the match color

        var do_work = True
        while do_work:
            var record = reader.read_owned(self.matcher)

            if not record:
                do_work = False
            else:
                bytes_saved += record.value().size_in_bytes()
                sequences.append(record.value())

            if bytes_saved >= self.settings.batch_size or not do_work:
                var outputs = self.par(sequences)
                for i in range(0, len(outputs)):
                    var m = outputs[i]
                    if not m:
                        continue
                    var r = Pointer.address_of(sequences[i])
                    writer.write(">")
                    writer.write_bytes(r[].name)
                    writer.write("\n")
                    writer.write_bytes(r[].seq[0 : m.value().start])
                    writer.write("\033[1;31m")
                    writer.write_bytes(r[].seq[m.value().start : m.value().end])
                    writer.write()
                    writer.write("\033[0m")
                    writer.write_bytes(r[].seq[m.value().end :])
                    writer.write("\n")
                sequences.clear()
                bytes_saved = 0
        writer.flush()
        writer.close()

    fn par(
        read self, read seqs: Span[ByteFastaRecord]
    ) -> List[Optional[MatchResult]]:
        # TODO: reusable output buffer
        var output = List[Optional[MatchResult]](capacity=len(seqs))
        for _ in range(0, len(seqs)):
            output.append(None)

        fn do_matching(index: Int) capturing:
            var target = Pointer.address_of(seqs[index])
            output[index] = self.matcher.first_match(
                target[].seq, self.settings.pattern
            )

        parallelize[do_matching](len(seqs), self.settings.threads)
        return output
