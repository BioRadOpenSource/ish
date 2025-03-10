"""Test program for SSW."""
from sys import stdout, stderr
from time.time import perf_counter

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind, ParsedOpts
from ExtraMojo.io.buffered import BufferedWriter
from ishlib.matcher.alignment.ssw_align import (
    ScoringMatrix,
    ssw_align,
    aa_to_num,
    Profile,
    ScoreSize,
)
from ishlib.formats.fasta import FastaRecord

"""
fprintf(stderr, "\n");
fprintf(stderr, "Usage: ssw_test [options] ... <target.fasta> <query.fasta>(or <query.fastq>)\n");
fprintf(stderr, "Options:\n");
fprintf(stderr, "\t-m N\tN is a positive integer for weight match in genome sequence alignment. [default: 2]\n");
fprintf(stderr, "\t-x N\tN is a positive integer. -N will be used as weight mismatch in genome sequence alignment. [default: 2]\n");
fprintf(stderr, "\t-o N\tN is a positive integer. -N will be used as the weight for the gap opening. [default: 3]\n");
fprintf(stderr, "\t-e N\tN is a positive integer. -N will be used as the weight for the gap extension. [default: 1]\n");
fprintf(stderr, "\t-p\tDo protein sequence alignment. Without this option, the ssw_test will do genome sequence alignment.\n");
fprintf(stderr, "\t-a FILE\tFILE is either the Blosum or Pam weight matrix. [default: Blosum50]\n");
fprintf(stderr, "\t-c\tReturn the alignment path.\n");
fprintf(stderr, "\t-f N\tN is a positive integer. Only output the alignments with the Smith-Waterman score >= N.\n");
fprintf(stderr, "\t-r\tThe best alignment will be picked between the original read alignment and the reverse complement read alignment.\n");
fprintf(stderr, "\t-s\tOutput in SAM format. [default: no header]\n");
fprintf(stderr, "\t-h\tIf -s is used, include header in SAM output.\n\n");
"""


fn parse_args() raises -> ParsedOpts:
    var parser = OptParser(name="ssw", description="Test SSW.")
    parser.add_opt(
        OptConfig(
            "match-score",
            OptKind.IntLike,
            default_value=String("2"),
            description="Score for a match.",
        )
    )
    parser.add_opt(
        OptConfig(
            "mismatch-score",
            OptKind.IntLike,
            default_value=String("2"),
            description=(
                "Score for a mismatch, as a positive number that will be"
                " subtracted."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "gap-open-score",
            OptKind.IntLike,
            default_value=String("3"),
            description=(
                "Score for a opening a gap, as a positive number that will be"
                " subtracted."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "gap-ext-score",
            OptKind.IntLike,
            default_value=String("1"),
            description=(
                "Score for a extending a gap, as a positive number that will be"
                " subtracted."
            ),
        )
    )
    parser.add_opt(
        OptConfig(
            "scoring-matrix",
            OptKind.StringLike,
            default_value=String("Blosum50"),
            description="Scoring matrix to use. Currently supports: [Blosum50]",
        )
    )
    parser.add_opt(
        OptConfig(
            "target-fasta",
            OptKind.StringLike,
            description="File containing the target sequences.",
        )
    )
    parser.add_opt(
        OptConfig(
            "query-fasta",
            OptKind.StringLike,
            description="File containing the query sequences.",
        )
    )
    return parser.parse_sys_args()


@value
struct ByteFastaRecord:
    var name: String
    var seq: List[UInt8]
    var rev: List[UInt8]

    fn __init__(
        out self,
        owned name: String,
        owned seq: String,
        convert_to_aa: Bool = True,
    ):
        var seq_bytes = List(seq.as_bytes())
        if convert_to_aa:
            seq_bytes = aa_to_num(seq_bytes)
        var rev = List[UInt8](capacity=len(seq_bytes))
        for s in reversed(seq_bytes):
            rev.append(s[])
        self.name = name^
        self.seq = seq_bytes^
        self.rev = rev^


@value
struct Profiles:
    var fwd: Profile
    var rev: Profile

    fn __init__(
        out self, read record: ByteFastaRecord, read matrix: ScoringMatrix
    ):
        self.fwd = Profile(record.seq, matrix, ScoreSize.Adaptive)
        self.rev = Profile(record.rev, matrix, ScoreSize.Adaptive)


fn main() raises:
    var opts = parse_args()
    if opts.get_bool("help"):
        print(opts.get_help_message()[])
        return None

    # Create the score matrix
    var matrix_name = opts.get_string("scoring-matrix")
    var matrix: ScoringMatrix
    if matrix_name == "Blosum50":
        matrix = ScoringMatrix.blosm50()
    else:
        raise "Unknown matrix " + matrix_name

    var match_score = opts.get_int("match-score")
    var mismatch_score = opts.get_int("mismatch-score")
    var gap_open_score = opts.get_int("gap-open-score")
    var gap_extension_score = opts.get_int("gap-ext-score")

    ## Assuming we are using Blosum50 AA matrix for everything below this for now.

    # Read the fastas and encode the sequences
    var target_file = opts.get_string("target-fasta")
    var target_seqs = FastaRecord.slurp_fasta(target_file)
    var targets = List[ByteFastaRecord](capacity=len(target_seqs))
    while len(target_seqs) > 0:
        var t = target_seqs.pop()
        # I should be able to pass with ^ here, not sure why I can't
        targets.append(ByteFastaRecord(t.name, t.seq))
    targets.reverse()

    var query_file = opts.get_string("query-fasta")
    var query_seqs = FastaRecord.slurp_fasta(query_file)
    var queries = List[ByteFastaRecord](capacity=len(query_seqs))
    while len(query_seqs) > 0:
        var q = query_seqs.pop()
        queries.append(ByteFastaRecord(q.name, q.seq))
    queries.reverse()

    # Create query profiles
    var profiles = List[Profiles](capacity=len(queries))
    for q in queries:
        profiles.append(Profiles(q[], matrix))

    var writer = BufferedWriter(stdout)
    # Align
    print("Total query seqs:", len(queries), file=stderr)
    print("Total target seqs:", len(targets), file=stderr)
    var start = perf_counter()
    var work: UInt64 = 0
    for i in range(0, len(queries)):
        var query = Pointer.address_of(queries[i])
        var profiles = Pointer.address_of(profiles[i])
        for j in range(0, len(targets)):
            var target = Pointer.address_of(targets[j])
            var result = ssw_align(
                profile=profiles[].fwd,
                matrix=matrix,
                reference=target[].seq,
                query=query[].seq,
                reverse_profile=profiles[].rev,
                gap_open_penalty=gap_open_score,
                gap_extension_penalty=gap_extension_score,
                return_only_alignment_end=True,
                mask_length=15,
            )
            if result:
                writer.write(
                    # target[].name,
                    # ",",
                    # query[].name,
                    # ",",
                    i,
                    ",",
                    j,
                    ",",
                    len(query[].seq),
                    ",",
                    len(target[].seq),
                    ",",
                    result.value().score1,
                    ",",
                    result.value().read_end1,
                    ",",
                    result.value().ref_end1,
                    "\n",
                )
            work += len(target[].seq) * len(query[].seq)
    var end = perf_counter()

    var elapsed = end - start
    var cells_per_second = work.cast[DType.float64]() / elapsed
    writer.flush()
    print("Ran in", elapsed, "seconds", file=stderr)
    print("Total work:", work)
    print(cells_per_second, "cells per second", file=stderr)
    print(cells_per_second / 1000000000, "GCUPs", file=stderr)
