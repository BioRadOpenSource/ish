import csv
import io
import sys
import subprocess as sp
from dataclasses import dataclass
from typing import Optional, List

# Requries ish-aligner to have been compiled for the 3 widths of interest: (128, 256, 512)
ISH_128 = "/home/ubuntu/dev/ish/ish-aligner-128"
ISH_256 = "/home/ubuntu/dev/ish/ish-aligner-256"
ISH_512 = "/home/ubuntu/dev/ish/ish-aligner-512"

PARASAIL_ALIGNER = "/home/ubuntu/dev/parasail/apps/parasail_aligner"

# From parasail data dir
# https://github.com/jeffdaily/parasail/tree/600fb26151ff19899ee39a214972dcf2b9b11ed7/data
QUERY_SEQS = {
    "/home/ubuntu/dev/parasail/data/P56980.fasta": 24,
    "/home/ubuntu/dev/parasail/data/O29181.fasta": 63,
    "/home/ubuntu/dev/parasail/data/O60341.fasta": 852,
    "/home/ubuntu/dev/parasail/data/P00762.fasta": 246,
    "/home/ubuntu/dev/parasail/data/P01008.fasta": 464,
    "/home/ubuntu/dev/parasail/data/P01111.fasta": 189,
    "/home/ubuntu/dev/parasail/data/P02232.fasta": 144,
    "/home/ubuntu/dev/parasail/data/P03435.fasta": 567,
    "/home/ubuntu/dev/parasail/data/P03630.fasta": 127,
    "/home/ubuntu/dev/parasail/data/P03989.fasta": 362,
    "/home/ubuntu/dev/parasail/data/P04775.fasta": 2005,
    "/home/ubuntu/dev/parasail/data/P05013.fasta": 189,
    "/home/ubuntu/dev/parasail/data/P07327.fasta": 375,
    "/home/ubuntu/dev/parasail/data/P07756.fasta": 1500,
    "/home/ubuntu/dev/parasail/data/P08519.fasta": 4548,
    "/home/ubuntu/dev/parasail/data/P0C6B8.fasta": 3564,
    "/home/ubuntu/dev/parasail/data/P10635.fasta": 497,
    "/home/ubuntu/dev/parasail/data/P14942.fasta": 222,
    "/home/ubuntu/dev/parasail/data/P19096.fasta": 2504,
    "/home/ubuntu/dev/parasail/data/P20930.fasta": 4061,
    "/home/ubuntu/dev/parasail/data/P21177.fasta": 729,
    "/home/ubuntu/dev/parasail/data/P25705.fasta": 553,
    "/home/ubuntu/dev/parasail/data/P27895.fasta": 1000,
    "/home/ubuntu/dev/parasail/data/P28167.fasta": 3005,
    "/home/ubuntu/dev/parasail/data/P33450.fasta": 5147,
    "/home/ubuntu/dev/parasail/data/P42357.fasta": 657,
    "/home/ubuntu/dev/parasail/data/P53765.fasta": 255,
    "/home/ubuntu/dev/parasail/data/P58229.fasta": 511,
    "/home/ubuntu/dev/parasail/data/Q7TMA5.fasta": 4743,
    "/home/ubuntu/dev/parasail/data/Q8ZGB4.fasta": 361,
    "/home/ubuntu/dev/parasail/data/Q9UKN1.fasta": 5478,
}

MATRIX = ["blosum62", "blosum50"]

# curl https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2015_11/knowledgebase/uniprot_sprot-only2015_11.tar.gz --output uniprot_sprot-only2015_11.tar.gz
REF_DB = "/home/ubuntu/data/uniprot_sprot-only/uniprot_sprot.fasta"


@dataclass
class BenchmarkResults:
    aligner: str
    total_query_seqs: int
    total_target_seqs: int
    query_len: int
    matrix: str
    gap_open: int
    gap_extend: int
    u8_width: int
    u16_width: int
    score_size: str
    runtime_secs: float
    cells_updated: float
    gcups: float

    HEADERS = [
        "aligner",
        "total_query_seqs",
        "total_target_seqs",
        "query_len",
        "matrix",
        "gap_open",
        "gap_extend",
        "u8_width",
        "u16_width",
        "score_size",
        "runtime_secs",
        "cells_updated",
        "gcups",
    ]

    @staticmethod
    def to_csv(results: List["BenchmarkResults"]):
        writer = csv.DictWriter(sys.stdout, fieldnames=BenchmarkResults.HEADERS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "aligner": r.aligner,
                    "total_query_seqs": r.total_query_seqs,
                    "total_target_seqs": r.total_target_seqs,
                    "query_len": r.query_len,
                    "matrix": r.matrix,
                    "gap_open": r.gap_open,
                    "gap_extend": r.gap_extend,
                    "u8_width": r.u8_width,
                    "u16_width": r.u16_width,
                    "score_size": r.score_size,
                    "runtime_secs": r.runtime_secs,
                    "cells_updated": r.cells_updated,
                    "gcups": r.gcups,
                }
            )

    @staticmethod
    def from_ish_csv_str(csv_str: str, aligner: str) -> List["BenchmarkResults"]:
        csv_file = io.StringIO(csv_str)
        reader = csv.DictReader(
            csv_file,
            delimiter=",",
            fieldnames=BenchmarkResults.HEADERS[1:],  # skip "aligner"
        )
        results = []
        x = next(reader)  # Skip headers
        for row in reader:
            results.append(
                BenchmarkResults(
                    aligner,
                    int(row["total_query_seqs"]),
                    int(row["total_target_seqs"]),
                    int(row["query_len"]),
                    row["matrix"],
                    int(row["gap_open"]),
                    int(row["gap_extend"]),
                    int(row["u8_width"]),
                    int(row["u16_width"]),
                    row["score_size"],
                    float(row["runtime_secs"]),
                    float(row["cells_updated"]),
                    float(row["gcups"]),
                )
            )
        return results

    @staticmethod
    def from_parasail_blob_str(
        blob_str: str,
        query_len: int,
        instruction_set: str,
        score_size: int,
        aligner: str,
    ) -> "BenchmarkResults":
        file = io.StringIO(blob_str)

        u8_width = 0
        u16_width = 0
        if "128" in instruction_set:
            u8_width = 16
            u16_width = 8
        elif "256" in instruction_set:
            u8_width = 32
            u16_width = 16

        kv_pairs = {}
        for row in file:
            key, value = row.split(":")
            kv_pairs[key.strip()] = value.strip()

        return BenchmarkResults(
            aligner,
            int(kv_pairs["number of queries"]),
            int(kv_pairs["number of db seqs"]),
            int(query_len),
            kv_pairs["matrix"],
            int(kv_pairs["gap_open"]),
            int(kv_pairs["gap_extend"]),
            int(u8_width),
            int(u16_width),
            score_size,
            float(kv_pairs["alignment time"].split(" ")[0]),
            float(kv_pairs["work"].split(" ")[0]),
            float(kv_pairs["gcups"]),
        )


def run_parasail_aligner(
    parasail_path,
    target_fasta,
    query_fasta,
    query_len,
    output_file,
    instruction_set=None,  # available are "sse2_128,sse41_128,avx2_256,altivec_128,neon_128"
    score_size="adaptive",
    scoring_matrix="Blosum62",
    gap_open_score=3,
    gap_ext_score=1,
    *,
    algo="sg"
):

    scoring_matrix = scoring_matrix.lower()
    if score_size == "adaptive":
        score_size = "sat"
    elif score_size == "byte":
        score_size = "8"
    elif score_size == "word":
        score_size = "16"
    else:
        raise ValueError("Invalid score size")

    algorithm = (
        f"{algo}_striped_" + (instruction_set if instruction_set else "") + f"_{score_size}"
    )

    # fmt: off
    args = [
        parasail_path,
        "-t", "1", # force single threaded
        "-v", # output stats
        "-x", # Don't use suffix array filter
        "-o", str(gap_open_score), # gap open
        "-e", str(gap_ext_score), # gap extend
        "-m", scoring_matrix, # score matrix
        "-a", algorithm, # algorithm to use, selecting from `vectorized` from here: https://github.com/jeffdaily/parasail/blob/600fb26151ff19899ee39a214972dcf2b9b11ed7/README.md
        "-f", target_fasta,
        "-q", query_fasta,
        "-g", output_file
    ]
    # fmt: on

    result = None
    try:
        out = sp.run(
            " ".join(args), shell=True, check=True, text=True, capture_output=True
        )
        result = BenchmarkResults.from_parasail_blob_str(
            out.stdout,
            query_len=query_len,
            instruction_set=instruction_set,
            score_size=score_size,
            aligner="parasail_aligner",
        )
    except sp.CalledProcessError as e:
        print(f"Failed to run command: {e}", file=sys.stderr)
        print(" ".join(args), file=sys.stderr)
    return result


def run_ish_aligner(
    aligner_path,
    query_fasta,
    target_fasta,
    output_file,
    score_size="adaptive",
    scoring_matrix="Blosum62",
    gap_open_score=3,
    gap_ext_score=1,
    iterations=3,
    *,
    algo="striped-local"
) -> Optional[BenchmarkResults]:
    # fmt: off
    args = [
        aligner_path,
        "--algo", algo,
        "--query-fasta", query_fasta,
        "--target-fasta", target_fasta,
        "--output-file", output_file,
        "--gap-open-score", str(gap_open_score),
        "--gap-ext-score", str(gap_ext_score),
        "--scoring-matrix", scoring_matrix,
        "--iterations", str(iterations),
        "--score-size", score_size
    ]
    # fmt: on
    result = None
    try:
        out = sp.run(
            " ".join(args), shell=True, check=True, text=True, capture_output=True
        )
        if "overflow" in out.stdout:
            print("Overflow, no result for: ", " ".join(args), file=sys.stderr)
            return None
        result = BenchmarkResults.from_ish_csv_str(out.stdout, aligner="ish-aligner")[
            0
        ]  # Only take the first item since we're running this in such a way that only one will be there anyways
    except sp.CalledProcessError as e:
        print(f"Failed to run command: {e}", file=sys.stderr)
        print(" ".join(args), file=sys.stderr)
    return result


def main():

    # score_sizes = ["byte", "word", "adaptive"]
    score_sizes = ["word"]

    writer = csv.DictWriter(sys.stdout, fieldnames=BenchmarkResults.HEADERS)
    writer.writeheader()

    results: List[BenchmarkResults] = []
    for ish in [ISH_128]: #, ISH_256, ISH_512]:
        for score_size in score_sizes:
            for query in QUERY_SEQS.keys():
                print(f"Running {ish} on {query} with {score_size}", file=sys.stderr)
                r = run_ish_aligner(
                    ish,
                    query,
                    REF_DB,
                    score_size=score_size,
                    scoring_matrix="Blosum62",
                    output_file="/home/ubuntu/outputs/ish-aligner.csv",
                    iterations=1,
                    algo="striped-semi-global"
                )
                if r:
                    writer.writerow(
                        {
                            "aligner": r.aligner,
                            "total_query_seqs": r.total_query_seqs,
                            "total_target_seqs": r.total_target_seqs,
                            "query_len": r.query_len,
                            "matrix": r.matrix,
                            "gap_open": r.gap_open,
                            "gap_extend": r.gap_extend,
                            "u8_width": r.u8_width,
                            "u16_width": r.u16_width,
                            "score_size": r.score_size,
                            "runtime_secs": r.runtime_secs,
                            "cells_updated": r.cells_updated,
                            "gcups": r.gcups,
                        }
                    )
                    results.append(r)

    # for inst in ["sse41_128", "avx2_256"]:
    for inst in ["neon_128"]:
        for score_size in score_sizes:
            for query, query_len in QUERY_SEQS.items():
                print(f"Running {PARASAIL_ALIGNER} on {query}", file=sys.stderr)
                r = run_parasail_aligner(
                    PARASAIL_ALIGNER,
                    REF_DB,
                    query,
                    query_len,
                    instruction_set=inst,
                    score_size=score_size,
                    scoring_matrix="Blosum62",
                    output_file="/home/ubuntu/outputs/parasail-aligner.csv",
                    algo="sg"
                )
                if r:
                    writer.writerow(
                        {
                            "aligner": r.aligner,
                            "total_query_seqs": r.total_query_seqs,
                            "total_target_seqs": r.total_target_seqs,
                            "query_len": r.query_len,
                            "matrix": r.matrix,
                            "gap_open": r.gap_open,
                            "gap_extend": r.gap_extend,
                            "u8_width": r.u8_width,
                            "u16_width": r.u16_width,
                            "score_size": r.score_size,
                            "runtime_secs": r.runtime_secs,
                            "cells_updated": r.cells_updated,
                            "gcups": r.gcups,
                        }
                    )
                    results.append(r)

    #BenchmarkResults.to_csv(results)


if __name__ == "__main__":
    main()
