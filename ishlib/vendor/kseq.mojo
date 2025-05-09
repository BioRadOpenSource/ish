"""
Mojo port of some of kseq based on the Crystal implementation in the biofast repo.

```mojo
from ishlib.vendor.kseq import FastxReader, BufferedReader
from ishlib.vendor.zlib import GZFile

def main():
    var reader = FastxReader[read_comment=False](
        BufferedReader(GZFile("./M_abscessus_HiSeq.fq", "r"))
    )

    var count = 0
    var slen = 0
    var qlen = 0
    while reader.read() > 0:
        count += 1
        slen += len(reader.seq)
        qlen += len(reader.qual)
    print(count, slen, qlen, sep="\t")
```
"""
import sys
from memory import UnsafePointer, memcpy
from utils import StringSlice
from ishlib.vendor.swar_decode import decode

from time.time import perf_counter

from ExtraMojo.bstr.memchr import memchr

alias ASCII_NEWLINE = ord("\n")
alias ASCII_CARRIAGE_RETURN = ord("\r")
alias ASCII_TAB = ord("\t")
alias ASCII_SPACE = ord(" ")
alias ASCII_FASTA_RECORD_START = ord(">")
alias ASCII_FASTQ_RECORD_START = ord("@")
alias ASCII_FASTQ_SEPARATOR = ord("+")
alias ASCII_ZERO = UInt8(ord("0"))


# ──────────────────────────────────────────────────────────────
#  Helpers for reading in fastx++ files
# ──────────────────────────────────────────────────────────────


@always_inline
fn strip_newlines_in_place(
    mut bs: ByteString, disk: Int, expected: Int
) -> Bool:
    """Compact `bs` by removing every `\n` byte in‑place; return True if the
    resulting length equals `expected`.
    SIMD search for newline, shifts the bytes to the left, and resizes the buffer.
    Avoids allocating a new buffer and copying the data.

    bs:     Mutable buffer that already holds the raw FASTQ/FASTA chunk just read from disk
    disk:   The number of bytes that were actually read into bs
    expected:   how many bases/quality bytes should remain after stripping newlines;
        used as a quick integrity check.

    Returns:
        True if the resulting buffer's length equals `expected`, False otherwise.
    """
    # read_pos always starts the loop at the first byte that has not yet been examined.
    var read_pos: Int = 0
    # write_pos always starts at the first byte that has not yet been written into its final position
    var write_pos: Int = 0
    # Before the first newline, every byte is kept, so the pointers march together (no gap)
    # After the first newline, the pointers may diverge, and we will need to copy bytes

    while read_pos < disk:
        var span_rel = memchr[do_alignment=False](
            Span[UInt8, __origin_of(bs.ptr)](
                ptr=bs.ptr.offset(read_pos), length=disk - read_pos
            ),
            UInt8(ASCII_NEWLINE),
        )
        # If there are no new lines we dont have to adjust buffer
        # If there was newlines, compute the contiguous span without newlines
        var end_pos = disk if span_rel == -1 else read_pos + span_rel
        var span_len = end_pos - read_pos
        # We only need to copy if there are newlines that would made gaps resulting in write_pos != read_pos
        # See read_pos and write_pos comments above
        if span_len > 0 and write_pos != read_pos:
            memcpy(bs.ptr.offset(write_pos), bs.ptr.offset(read_pos), span_len)
        write_pos += span_len
        read_pos = end_pos + 1  # skip the '\n' (or exit loop if none)
    bs.resize(write_pos)
    return write_pos == expected


# ──────────────────────────────────────────────────────────────


@value
@register_passable("trivial")
struct SearchChar:
    var value: Int32
    alias Newline = Self(-1)
    alias Whitespace = Self(-2)
    # All other values are just the literal char to search

    @always_inline
    fn __eq__(read self, read other: Self) -> Bool:
        return self.value == other.value

    @always_inline
    fn as_char(read self) -> UInt8:
        return UInt8(self.value)


@value
struct ByteString(Sized):
    # TODO: add address_space
    var size: UInt32
    var cap: UInt32
    var ptr: UnsafePointer[UInt8]

    fn __init__(out self, capacity: UInt = 0):
        self.ptr = UnsafePointer[UInt8].alloc(0)
        self.size = 0
        self.cap = 0
        if capacity > 0:
            self.resize(capacity)

    fn __del__(owned self):
        self.ptr.free()

    fn __copyinit__(out self, read other: Self):
        self.ptr = UnsafePointer[UInt8].alloc(Int(other.size))
        self.size = other.size
        self.cap = other.size

    @always_inline
    fn __getitem__[I: Indexer](read self, idx: I) -> UInt8:
        return self.ptr[idx]

    @always_inline
    fn __setitem__[I: Indexer](mut self, idx: I, val: UInt8):
        self.ptr[idx] = val

    @always_inline
    fn __len__(read self) -> Int:
        return Int(self.size)

    # TODO: rename offset
    @always_inline
    fn addr[I: Indexer](mut self, i: I) -> UnsafePointer[UInt8]:
        return self.ptr.offset(i)

    @staticmethod
    @always_inline
    fn _roundup32(val: UInt32) -> UInt32:
        var x = val
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        return x + 1

    @always_inline
    fn clear(mut self):
        self.size = 0

    @always_inline
    fn reserve(mut self, cap: UInt32):
        if cap < self.cap:
            return
        self.cap = cap
        var new_data = UnsafePointer[UInt8].alloc(Int(self.cap))
        memcpy(new_data, self.ptr, len(self))
        if self.ptr:
            self.ptr.free()
        self.ptr = new_data

    @always_inline
    fn resize(mut self, size: UInt32):
        self.size = size
        if self.size <= self.cap:
            return
        self.cap = Self._roundup32(self.size)
        var new_data = UnsafePointer[UInt8].alloc(Int(self.cap))
        memcpy(new_data, self.ptr, len(self))

        if self.ptr:
            self.ptr.free()
        self.ptr = new_data

    # TODO: rename append
    @always_inline
    fn push(mut self, c: UInt8):
        self.resize(len(self) + 1)
        self.ptr[len(self) - 1] = c

    # TODO: rename extend
    @always_inline
    fn append(mut self, ptr: UnsafePointer[UInt8], length: Int):
        if length <= 0:
            return

        var old_size = len(self)
        self.resize(len(self) + length)
        memcpy(self.ptr.offset(old_size), ptr, Int(length))

    fn find_chr[c: UInt8](read self, start: Int, end: Int) -> Int:
        var p = memchr[do_alignment=False](
            Span[UInt8, __origin_of(self.ptr)](
                ptr=self.ptr.offset(start), length=end - start
            ),
            c,
        )

        return end if p == -1 else p + start

    fn find_chr(read self, c: UInt8, start: Int, end: Int) -> Int:
        var p = memchr[do_alignment=False](
            Span[UInt8, __origin_of(self.ptr)](
                ptr=self.ptr.offset(start), length=end - start
            ),
            c,
        )

        return end if p == -1 else p + start

    fn find(read self, c: SearchChar, start: Int, end: Int) -> Int:
        var r = end

        if c == SearchChar.Newline:
            r = self.find_chr[ASCII_NEWLINE](start, end)
        elif c == SearchChar.Whitespace:
            for i in range(start, end):
                var x = self[i]
                if x == ASCII_TAB or x == ASCII_SPACE or x == ASCII_NEWLINE:
                    r = i
                    break
        else:
            r = self.find_chr(c.as_char(), start, end)

        return r

    @always_inline
    fn as_span[mut: Bool, //, o: Origin[mut]](ref [o]self) -> Span[UInt8, o]:
        return Span[UInt8, o](ptr=self.ptr, length=len(self))

    fn to_string(self) -> String:
        return String(StringSlice(unsafe_from_utf8=self.as_span()))


trait KRead(Movable):
    fn unbuffered_read[
        o: MutableOrigin
    ](mut self, buffer: Span[UInt8, o]) raises -> Int:
        ...


struct BufferedReader[R: KRead](Movable):
    var buffer: ByteString
    var start: Int
    var end: Int
    var end_of_file: Bool
    var reader: R

    fn __init__(out self, owned reader: R):
        self.buffer = ByteString(1024 * 128)
        self.start = 0
        self.end = 0
        self.end_of_file = False
        self.reader = reader^

    fn __moveinit__(out self, owned other: Self):
        self.buffer = other.buffer^
        self.start = other.start
        self.end = other.end
        self.end_of_file = other.end_of_file
        self.reader = other.reader^

    @always_inline
    fn read_bytes(
        mut self, mut buffer: ByteString, mut rest: Int
    ) raises -> Int32:
        if self.end_of_file and self.start >= self.end:
            return 0

        while rest > self.end - self.start:
            buffer.append(self.buffer.addr(self.start), self.end - self.start)
            rest -= self.end - self.start
            self.start = 0
            self.end = self.reader.unbuffered_read(self.buffer.as_span())
            if self.end < len(self.buffer):
                self.end_of_file = True
            if self.end < 0:
                return -2
            if self.end == 0:
                return len(buffer)
        buffer.append(self.buffer.addr(self.start), rest)
        self.start += rest
        return len(buffer)

    @always_inline
    fn read_byte(mut self) raises -> Int32:
        if self.end_of_file and self.start >= self.end:
            return -1
        if self.start >= self.end:
            self.start = 0
            self.end = self.reader.unbuffered_read(self.buffer.as_span())
            if self.end < len(self.buffer):
                self.end_of_file = True
            if self.end == 0:
                return -1
            elif self.end < 0:
                return -2
        var c = self.buffer[self.start]
        self.start += 1
        return Int32(c)

    @always_inline
    fn eof(read self) -> Bool:
        return self.end_of_file and self.start >= self.end

    # TODO: make keep a parameter
    # TODO: document, keep is whether or not to keep the char read up until
    @always_inline
    fn read_until[
        delim: SearchChar
    ](
        mut self,
        mut buffer: ByteString,
        # delim: SearchChar = SearchChar.Newline,
        offset: Int = 0,
        keep: Bool = False,
    ) raises -> Int32:
        var got_any = False
        buffer.resize(offset)
        while True:
            if self.start >= self.end:
                if self.end_of_file:
                    break
                self.start = 0
                self.end = self.reader.unbuffered_read(self.buffer.as_span())
                if self.end < len(self.buffer):
                    self.end_of_file = True
                if self.end < 0:
                    return -2
                elif self.end == 0:
                    break
            got_any = True
            # this is technically two passes, what if we just use a for loop?
            var r: Int

            @parameter
            if delim == SearchChar.Newline:
                r = self.buffer.find_chr[ASCII_NEWLINE](self.start, self.end)
            elif delim == SearchChar.Whitespace:
                r = self.buffer.find(delim, self.start, self.end)
            else:
                r = self.buffer.find_chr[UInt8(delim.value)](
                    self.start, self.end
                )

            if r < self.end and keep:
                buffer.append(self.buffer.addr(self.start), r - self.start + 1)
            else:
                buffer.append(self.buffer.addr(self.start), r - self.start)

            self.start = r + 1
            if r < self.end:
                break
        if not got_any and self.end_of_file:
            return -1
        # Handle \r
        if (
            delim == SearchChar.Newline
            and len(buffer) > 1
            and buffer[len(buffer) - 1] == ASCII_CARRIAGE_RETURN
        ):
            buffer.resize(len(buffer) - 1)
        return len(buffer)


@value
struct FastxRecord:
    var name: ByteString
    var seq: ByteString
    var qual: Optional[ByteString]
    var comment: Optional[ByteString]


struct FastxReader[R: KRead, read_comment: Bool = True](Movable):
    var name: ByteString
    var seq: ByteString
    var qual: ByteString
    var comment: ByteString
    var reader: BufferedReader[R]
    var last_char: Int32

    fn __init__(out self, owned reader: BufferedReader[R]):
        self.reader = reader^
        self.name = ByteString()
        self.seq = ByteString(256)
        self.qual = ByteString()
        self.comment = ByteString()
        # Special comment field is 26 bytes long +1 from backtick
        self.last_char = 0

    fn __moveinit__(out self, owned other: Self):
        self.name = other.name^
        self.seq = other.seq^
        self.qual = other.qual^
        self.comment = other.comment^
        self.reader = other.reader^
        self.last_char = other.last_char

    fn to_owned(read self) -> FastxRecord:
        var qual: Optional[ByteString] = None
        var comment: Optional[ByteString] = None
        if self.qual.size > 0:
            qual = self.qual
        if self.comment.size > 0:
            comment = self.comment

        return FastxRecord(self.name, self.seq, qual, comment)

    fn read(mut self) raises -> Int:
        """Read a single record into the reused FastxReader buffers.

        Returns:
            >=0  length of the sequence (normal)
            -1   end-of-file
            -2   truncated quality string
            -3   error reading stream.
        """
        # Jump to next header line, or if false, the first header char was read by the last call
        if self.last_char == 0:
            var c = self.reader.read_byte()
            while (
                c >= 0
                and c != ASCII_FASTA_RECORD_START
                and c != ASCII_FASTQ_RECORD_START
            ):
                c = self.reader.read_byte()
            if c < 0:
                return Int(c)  # EOF Error

        # Reset all buffers for reuse
        self.seq.clear()
        self.qual.clear()
        self.comment.clear()
        self.name.clear()

        @parameter
        if read_comment:
            var r = self.reader.read_until[SearchChar.Whitespace](
                self.name, 0, True
            )
            if r < 0:
                return Int(r)  # normal exit: EOF or error
            if self.name[len(self.name) - 1] != ASCII_NEWLINE:
                r = self.reader.read_until[SearchChar.Newline](
                    self.comment
                )  # read FASTA/Q comment
            self.name.resize(len(self.name) - 1)
        else:
            var r = self.reader.read_until[SearchChar.Newline](
                self.name, 0, False
            )
            if r < 0:
                return Int(r)  # normal exit: EOF or error

        var c = self.reader.read_byte()
        while (
            c >= 0
            and c != ASCII_FASTA_RECORD_START
            and c != ASCII_FASTQ_SEPARATOR
            and c != ASCII_FASTQ_RECORD_START
        ):
            if c == ASCII_CARRIAGE_RETURN:  # Skip empty lines
                c = self.reader.read_byte()
                continue

            self.seq.push(UInt8(c))  # always enough room for 1 char
            r = self.reader.read_until[SearchChar.Newline](
                self.seq, len(self.seq)
            )  # read the rest of the line
            c = self.reader.read_byte()
        if c == ASCII_FASTA_RECORD_START or c == ASCII_FASTQ_RECORD_START:
            self.last_char = c  # the first header char has been read

        if c != ASCII_FASTQ_SEPARATOR:  # FASTA
            return len(self.seq)

        # FASTQ
        # Skip the rest of the '+' line
        # r = self.reader.read_until(self.tmp)
        c = self.reader.read_byte()
        while c >= 0 and c != ASCII_NEWLINE:
            c = self.reader.read_byte()
        if c < 0:
            return -2  # error: no quality string

        # TODO: reserve cap for qual
        self.qual.reserve(len(self.seq))
        r = self.reader.read_until[SearchChar.Newline](
            self.qual, len(self.qual)
        )
        while r >= 0 and len(self.qual) < len(self.seq):
            r = self.reader.read_until[SearchChar.Newline](
                self.qual, len(self.qual)
            )
        if r < 0:
            return -3  # stream error
        self.last_char = 0  # we have not come to the next header line
        if len(self.qual) != len(self.seq):
            return -2  # error: qual string is of different length
        return len(self.seq)

    fn read_fastxpp_strip_newline(mut self) raises -> Int:
        # ── 0 Locate the next header byte ('>' or '@') ──────────────────────
        var marker: UInt8
        if self.last_char == 0:
            var c = self.reader.read_byte()
            while (
                c >= 0
                and c != ASCII_FASTA_RECORD_START
                and c != ASCII_FASTQ_RECORD_START
            ):
                c = self.reader.read_byte()
            if c < 0:
                # EOF or stream error
                return Int(c)
            marker = UInt8(c)
        else:
            marker = UInt8(self.last_char)
            self.last_char = 0

        # ── 1 Reset buffers reused across records ───────────────────────────
        self.seq.clear()
        self.qual.clear()
        self.comment.clear()
        self.name.clear()

        # ── 2 Read the back‑tick header line --------------------------------
        var r = self.reader.read_until[SearchChar.Newline](self.name, 0, False)
        if r < 0:
            return Int(r)

        var hdr = self.name.as_span()
        if len(hdr) == 0 or hdr[0] != UInt8(ord("`")):
            # We need at least 21 bytes: 1 backtick + 9 + 7 + 3 + 1 backtick
            return -3  # Not a proper FASTX++ header

        # ── 3 Decode slen:lcnt:bpl from the fixed fields --------------------
        var slen = decode[9](hdr[1:10])  # 9‑digit field at positions [1..9]
        var lcnt = decode[7](hdr[10:17])  # 7‑digit field at positions [10..16]
        var bpl = decode[3](hdr[17:20])  # 3‑digit field at positions [17..19]

        # Confirm the second backtick is at hdr[20]
        if hdr[20] != UInt8(ord("`")):
            return -3

        # ── 4 Read the sequence block (slen + lcnt bytes on disk) -----------
        var disk_seq = slen + lcnt
        self.seq.reserve(disk_seq)
        var _disk_seq = disk_seq
        var got_seq = self.reader.read_bytes(self.seq, _disk_seq)
        if got_seq != disk_seq:
            return -3  # truncated record

        # ── 5  Remove newline characters in‑place using the helper -----------
        var ok = strip_newlines_in_place(self.seq, disk_seq, slen)
        if not ok:
            return -2  # mismatch: not the expected base count

        return len(self.seq)

    fn read_fastxpp_swar(mut self) raises -> Int:
        # ── 0 Locate the next header byte ('>' or '@') ──────────────────────
        var marker: UInt8  # remember which flavour we’re on
        if self.last_char == 0:
            var c = self.reader.read_byte()
            while (
                c >= 0
                and c != ASCII_FASTA_RECORD_START
                and c != ASCII_FASTQ_RECORD_START
            ):
                c = self.reader.read_byte()
            if c < 0:
                return Int(c)  # EOF / stream error (-1 / -2)
            marker = UInt8(c)
        else:
            marker = UInt8(self.last_char)
            var c = self.last_char
            self.last_char = 0

        # ── 1 Reset buffers reused across records ───────────────────────────
        self.seq.clear()
        self.qual.clear()
        self.comment.clear()
        self.name.clear()

        # ── 2 Read the back‑tick header line --------------------------------
        var r = self.reader.read_until[SearchChar.Newline](self.name, 0, False)
        if r < 0:
            return Int(r)

        var hdr = self.name.as_span()
        if len(hdr) == 0 or hdr[0] != UInt8(ord("`")):
            print("ERROR: Opening backtick check failed. hdr[0]")
            return -3  # not a FASTX++ BPL header

        # useful debugging
        # for i in range(len(hdr)):
        #     var code = Int(hdr[i])
        #     print(i, hdr[i], code, chr(code))

        # ── 3 Find closing back‑tick and parse slen:lcnt:bpl -----------
        var slen = decode[9](hdr[1:10])  # bytes 1–9
        var lcnt = decode[7](hdr[10:17])  # bytes 10–16
        var bpl = decode[3](hdr[17:20])  # bytes 17–19

        if hdr[20] != UInt8(ord("`")):
            print("ERROR: Closing backtick check failed. base[25]")
            return -3

        # ── 4 SEQUENCE block ---------------------------------------
        var disk_seq = slen + lcnt  # immutable reference
        var rest_seq = disk_seq  # mutable copy for read_bytes

        self.seq.reserve(disk_seq)
        var got_seq = self.reader.read_bytes(self.seq, rest_seq)
        if got_seq != disk_seq:
            print("ERROR: Sequence read failed. got_seq != disk_seq")
            return -3  # truncated record

        # compact in‑place: copy (bpl‑1) bases, skip the LF, repeat
        var write_pos: Int = 0
        var read_pos: Int = 0
        while read_pos < disk_seq:
            memcpy(
                self.seq.addr(write_pos),  # destination
                self.seq.addr(read_pos),  # source
                bpl - 1,
            )  # copy only the bases
            write_pos += bpl - 1
            read_pos += bpl  # jump over the LF
        self.seq.resize(write_pos)  # write_pos == slen

        return len(self.seq)

    fn read_fastxpp_read_once(mut self) raises -> Int:
        # ── 0 Locate the next header byte ('>' or '@') ──────────────────────
        var marker: UInt8
        if self.last_char == 0:
            var c = self.reader.read_byte()
            while (
                c >= 0
                and c != ASCII_FASTA_RECORD_START
                and c != ASCII_FASTQ_RECORD_START
            ):
                c = self.reader.read_byte()
            if c < 0:
                return Int(c)
            marker = UInt8(c)
        else:
            marker = UInt8(self.last_char)
            self.last_char = 0

        # ── 1 Reset buffers reused across records ───────────────────────────
        self.seq.clear()
        self.qual.clear()
        self.comment.clear()
        self.name.clear()

        # ── 2 Read the back‑tick header line --------------------------------
        var r = self.reader.read_until[SearchChar.Newline](self.name, 0, False)
        if r < 0:
            return Int(r)

        var hdr = self.name.as_span()
        # ── 3 Find closing back‑tick and parse slen:lcnt:bpl -----------
        if len(hdr) == 0 or hdr[0] != UInt8(ord("`")):
            print("ERROR: header lacks opening back-tick")
            return -3

        var slen = decode[9](hdr[1:10])
        var lcnt = decode[7](hdr[10:17])  # not needed in this approach
        var bpl = decode[3](hdr[17:20])

        if hdr[20] != UInt8(ord("`")):
            print("ERROR: header lacks closing back-tick")
            return -3

        # ── 4 SEQUENCE block ---------------------------------------
        self.seq.reserve(UInt32(slen))
        var copied: Int = 0
        while copied < slen:
            var want = min(bpl - 1, slen - copied)

            # track length before and after
            var before = len(self.seq)
            var _want = want
            var _total = self.reader.read_bytes(self.seq, _want)
            if _total < 0:
                print("ERROR: read_bytes returned error", _total)
                return -3
            var got = Int(_total) - before  # true delta

            if got != want:
                return -3

            copied += got

            # consume newline
            var nl = self.reader.read_byte()
            if nl != ASCII_NEWLINE:
                print("ERROR: expected newline after sequence chunk, found", nl)
                return -3
        return slen


struct FileReader(KRead):
    var fh: FileHandle

    fn __init__(out self, owned fh: FileHandle):
        self.fh = fh^

    fn __moveinit__(out self, owned other: Self):
        self.fh = other.fh^

    fn unbuffered_read[
        o: MutableOrigin
    ](mut self, buffer: Span[UInt8, o]) raises -> Int:
        return Int(self.fh.read(buffer.unsafe_ptr(), len(buffer)))


# ──────────────────────────────────────────────────────────────
# Main for debugging
# ──────────────────────────────────────────────────────────────
#
fn main() raises:
    var argv = sys.argv()
    if len(argv) != 2:
        print("Usage: mojo run kseq.mojo <file.fastxpp_bpl>")
        return

    var fh = open(String(argv[1]), "r")
    var reader = FastxReader[read_comment=False](
        BufferedReader(FileReader(fh^))
    )

    var first = reader.read_fastxpp()
    print("first‑read returned", first)

    var count = 0
    while True:
        var n = reader.read_fastxpp()
        if n < 0:
            break
        count += 1
        print("rec#", count, "seq_len", n, "hdr_len", len(reader.name))
