from collections import Optional
from collections.string import StringSlice


@value
@register_passable("trivial")
struct TargetSpan:
    var start: Int
    var end: Int


@value
struct AlignmentResult:
    var score: Int32
    var alignment1: Optional[String]
    var alignment2: Optional[String]
    var coords: Optional[TargetSpan]

    fn display_alignments(
        read self, cols: Int = 120
    ) raises -> Optional[String]:
        if not self.alignment1:
            return None
        var ret = String()
        var aln1 = StringSlice(self.alignment1.value())
        var aln2 = StringSlice(self.alignment2.value())
        var i = 0
        ret.write("Score: ", self.score, "\n")
        while i < len(aln1):
            ret.write(aln1[i : min(i + cols, len(aln1))], "\n")
            ret.write("|" * (min(i + cols, len(aln1)) - i), "\n")
            ret.write(aln2[i : min(i + cols, len(aln1))], "\n")
            ret.write("-" * (min(i + cols, len(aln1)) - i), "\n")
            i += cols
        return ret
