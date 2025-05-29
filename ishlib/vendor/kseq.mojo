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
from memory import UnsafePointer, memcpy

from time.time import perf_counter

from ExtraMojo.bstr.memchr import memchr


alias ASCII_NEWLINE = ord("\n")
alias ASCII_CARRIAGE_RETURN = ord("\r")
alias ASCII_TAB = ord("\t")
alias ASCII_SPACE = ord(" ")
alias ASCII_FASTA_RECORD_START = ord(">")
alias ASCII_FASTQ_RECORD_START = ord("@")
alias ASCII_FASTQ_SEPARATOR = ord("+")


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

    fn __moveinit__(out self, owned other: Self):
        self.ptr = other.ptr
        self.size = other.size
        self.cap = other.cap

    fn __copyinit__(out self, read other: Self):
        self.cap = other.cap
        self.size = other.size
        self.ptr = UnsafePointer[UInt8].alloc(Int(self.cap))
        memcpy(self.ptr, other.ptr, Int(other.size))

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
        self.ptr.free()
        self.ptr = new_data

    @always_inline
    fn resize(mut self, size: UInt32):
        var old_size = self.size
        self.size = size
        if self.size <= self.cap:
            return
        self.cap = Self._roundup32(self.size)
        var new_data = UnsafePointer[UInt8].alloc(Int(self.cap))
        memcpy(new_data, self.ptr, Int(old_size))

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
        self.seq = ByteString()
        self.seq.reserve(256)
        self.qual = ByteString()
        self.comment = ByteString()
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

        # Reset all members
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
