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
    while reader.read_fastxpp_strip_newline() > 0:
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

from algorithm.functional import vectorize # vectorize might need a compile-time width.
from sys.intrinsics import compressed_store
from bit import pop_count

from ExtraMojo.bstr.memchr import memchr

alias ASCII_NEWLINE = ord("\n")
alias ASCII_CARRIAGE_RETURN = ord("\r")
alias ASCII_TAB = ord("\t")
alias ASCII_SPACE = ord(" ")
alias ASCII_FASTA_RECORD_START = ord(">")
alias ASCII_FASTQ_RECORD_START = ord("@")
alias ASCII_FASTQ_SEPARATOR = ord("+")
alias ASCII_ZERO = UInt8(ord("0"))

alias NL = UInt8(ASCII_NEWLINE)

# ──────────────────────────────────────────────────────────────
#  Helpers for reading in fastx++ files
# ──────────────────────────────────────────────────────────────

@always_inline
fn strip_lf(buf_ptr: UnsafePointer[UInt8], n: Int) -> Int:
    """Remove all LF bytes from `buf_ptr[0:n)` in-place, return new length.
    Uses vectorized operations with compressed_store.
    The SIMD vector width is determined by VEC_WIDTH parameter for vectorize.
    """
    var wr = 0  # write cursor

    alias VEC_WIDTH = sys.intrinsics.simdwidthof[DType.uint8]()
                         # This means width parameter in step will be VEC_WIDTH.

    @parameter
    fn step[width: Int](i: Int):
        # Load SIMD vector from UnsafePointer. `width` will be VEC_WIDTH (e.g., 32).
        # The result `v` is of type SIMD[DType.uint8, width].
        var v = buf_ptr.offset(i).load[width=width]()
        var keep_mask = (v != NL)  # Mask is true for elements to KEEP (non-newlines)
        
        sys.intrinsics.compressed_store(v,
                         buf_ptr.offset(wr),
                         keep_mask) # Store elements where keep_mask is true
        wr += keep_mask.reduce_bit_count() # Increment wr by the number of elements kept

    vectorize[step, VEC_WIDTH](n)
    return wr

# @always_inline
# fn strip_newlines_in_place(
#     mut bs: ByteString, disk_len_to_process: Int
# ) -> Int:
#     """Compact `bs` by removing every `\n` byte in‑place from the first
#     `disk_len_to_process` bytes. `bs` is resized to the new compacted length.
#     Returns the new length of the compacted content.
#     """
#     var read_pos: Int = 0
#     var write_pos: Int = 0
# 
#     if disk_len_to_process == 0:
#         bs.resize(0)
#         return 0
# 
#     while read_pos < disk_len_to_process:
#         var search_len = disk_len_to_process - read_pos
#         var span_rel = -1
#         if search_len > 0:
#             span_rel = memchr[do_alignment=False](
#                 Span[UInt8, __origin_of(bs.ptr)](
#                     ptr=bs.ptr.offset(read_pos), length=search_len
#                 ),
#                 UInt8(ASCII_NEWLINE),
#             )
#         
#         var current_chunk_end_pos: Int
#         if span_rel == -1:
#             current_chunk_end_pos = disk_len_to_process
#         else:
#             current_chunk_end_pos = read_pos + span_rel
#             
#         var chunk_len_to_copy = current_chunk_end_pos - read_pos
#         
#         if chunk_len_to_copy > 0:
#             if write_pos != read_pos:
#                 memcpy(bs.ptr.offset(write_pos), bs.ptr.offset(read_pos), chunk_len_to_copy)
#             write_pos += chunk_len_to_copy
#         
#         if span_rel == -1:
#             read_pos = disk_len_to_process 
#         else:
#             read_pos = current_chunk_end_pos + 1
#             
#     bs.resize(UInt32(write_pos))
#     return write_pos

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

    fn _refill_and_compact_buffer(mut self) raises -> Bool:
        """Tries to refill `self.buffer`, compacts it by stripping newlines,
        and updates `self.start` and `self.end`.
        Returns True if new data (or EOF) was processed, False on stream error.
        Sets self.end_of_file appropriately.
        """
        self.start = 0
        self.buffer.resize(self.buffer.cap) 

        var raw_bytes_read = self.reader.unbuffered_read(self.buffer.as_span())

        if raw_bytes_read < 0:
            self.end = 0
            return False

        if raw_bytes_read == 0:
            self.end_of_file = True
            self.end = 0
            self.buffer.resize(0)
            return True

        if UInt32(raw_bytes_read) < self.buffer.cap:
            self.end_of_file = True
        
        var new_compacted_length = strip_lf(self.buffer.ptr, raw_bytes_read)
        self.buffer.resize(UInt32(new_compacted_length)) # Resize buffer to new compacted length
        self.end = new_compacted_length

        return True

    @always_inline
    fn read_bytes[
        o: MutableOrigin
    ](
        mut self, target_span: Span[UInt8, o], num_bytes_to_read: Int
    ) raises -> Int32:
        if num_bytes_to_read <= 0 or len(target_span) == 0:
            return 0
        
        # Ensure we don't try to read more than the target span can hold
        var effective_num_bytes_to_read = min(num_bytes_to_read, len(target_span))
        if effective_num_bytes_to_read == 0:
            return 0
            
        var total_bytes_copied_to_target = 0
        var target_write_offset = 0

        while effective_num_bytes_to_read > 0:
            var available_in_internal_buffer = self.end - self.start
            if available_in_internal_buffer == 0:
                if self.end_of_file:
                    break 
                
                if not self._refill_and_compact_buffer():
                    return -2
                
                if self.end == 0 and self.end_of_file:
                    break
                if self.end == 0 and not self.end_of_file:
                    continue 
                available_in_internal_buffer = self.end - self.start
                if available_in_internal_buffer == 0 and self.end_of_file:
                    break
            
            var bytes_to_copy_now = min(effective_num_bytes_to_read, available_in_internal_buffer)
            if bytes_to_copy_now <= 0:
                 break

            # Copy from self.buffer to target_span
            memcpy(target_span.unsafe_ptr().offset(target_write_offset), 
                   self.buffer.addr(self.start), 
                   bytes_to_copy_now)
            
            self.start += bytes_to_copy_now
            target_write_offset += bytes_to_copy_now
            total_bytes_copied_to_target += bytes_to_copy_now
            effective_num_bytes_to_read -= bytes_to_copy_now

        return total_bytes_copied_to_target

    @always_inline
    fn read_byte(mut self) raises -> Int32:
        if self.start >= self.end:
            if self.end_of_file:
                return -1 # EOF
            
            if not self._refill_and_compact_buffer():
                return -2

            if self.end == 0:
                return -1 # Treat as EOF for this byte read
        
        var c = self.buffer[self.start]
        self.start += 1
        return Int32(c)

    @always_inline
    fn eof(read self) -> Bool:
        return self.end_of_file and self.start >= self.end

    @always_inline
    fn read_until[
        delim: SearchChar
    ](
        mut self,
        mut target_buffer: ByteString,
        offset: Int = 0,
        keep: Bool = False,
    ) raises -> Int32:
        target_buffer.resize(offset)
        var initial_target_len = len(target_buffer)

        while True:
            if self.start >= self.end:
                if self.end_of_file:
                    break
                if not self._refill_and_compact_buffer():
                    return -2
                if self.end == 0:
                    break 
            
            var search_end_in_buffer = self.end
            var r: Int
            
            @parameter
            if delim == SearchChar.Newline:
                r = self.buffer.find_chr[ASCII_NEWLINE](self.start, search_end_in_buffer)
            elif delim == SearchChar.Whitespace:
                r = self.buffer.find(delim, self.start, search_end_in_buffer)
            else:
                r = self.buffer.find_chr[UInt8(delim.value)](self.start, search_end_in_buffer)

            var found_delimiter = (r < search_end_in_buffer)
            var bytes_to_append_len = r - self.start
            if found_delimiter and keep:
                bytes_to_append_len += 1
            
            if bytes_to_append_len > 0:
                target_buffer.append(self.buffer.addr(self.start), bytes_to_append_len)
            
            self.start += bytes_to_append_len
            if found_delimiter:
                 break
            
            if not found_delimiter and self.start >= self.end:
                if self.end_of_file:
                    break

            elif not found_delimiter:
                 break

        var final_len = len(target_buffer)
        if final_len == initial_target_len and self.eof():
             # No new bytes were added and we are at EOF
             if initial_target_len == 0 : return -1 # Truly nothing read, EOF
             # else, initial_target_len bytes were already there, but nothing new added

        # Handle \r
        if (
            delim == SearchChar.Newline
            and len(target_buffer) > 1
            and target_buffer[len(target_buffer) - 1] == ASCII_CARRIAGE_RETURN
        ):
            target_buffer.resize(len(target_buffer) - 1)
            
        return len(target_buffer) # Return total length of target_buffer


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


    fn read_fastxpp_strip_newline(mut self) raises -> Int:
        # ── 0 Locate the next header byte ('>' or '@') ──────────────────────
        var marker_char_code: Int32
        if self.last_char == 0:
            var c = self.reader.read_byte()
            while (
                c >= 0
                and c != ASCII_FASTA_RECORD_START
                and c != ASCII_FASTQ_RECORD_START
            ):
                c = self.reader.read_byte()
            if c < 0:
                return Int(c) # EOF or stream error
            marker_char_code = c
        else:
            marker_char_code = self.last_char
            self.last_char = 0

        # ── 1 Reset buffers reused across records ───────────────────────────
        self.seq.clear()
        self.qual.clear()
        self.comment.clear()
        self.name.clear()

        # ── 2 Parse Record ID (up to first backtick) ───────────────────────
        self.name.push(UInt8(marker_char_code)) # Start name with @ or >
        
        while True:
            var c = self.reader.read_byte()
            if c < 0:
                return -3 if c == -2 else -1 
            
            if UInt8(c) == UInt8(ord('`')):
                break
            self.name.push(UInt8(c))
            #if len(self.name) > 4096: 
            #    return -3
        
        # ── 3 Read and Decode SWAR Block (`hlen:slen`) ─────────────────────
        var swar_buffer = ByteString() 
        var swar_block_len = 16 # hlen(6) + slen(9) + closing_tick(1)
        swar_buffer.resize(UInt32(swar_block_len))

        var swar_bytes_read = self.reader.read_bytes(swar_buffer.as_span[mut=True](), swar_block_len)

        if swar_bytes_read != swar_block_len:
            return -3

        if swar_buffer[15] != UInt8(ord('`')):
            return -3

        var swar_span = swar_buffer.as_span()
        var hlen_val = decode[6](swar_span[0:6])
        var slen_val = decode[9](swar_span[6:15]) 

        # ── 4 Read "Rest of Header" and append to self.name (if hlen_val > 0) ----
        self.comment.clear()
        if hlen_val > 0:
            self.comment.resize(UInt32(hlen_val))
            var rest_header_bytes_read = self.reader.read_bytes(self.comment.as_span[mut=True](), hlen_val)
            if rest_header_bytes_read != hlen_val:
                return -3
            self.name.append(self.comment.addr(0), len(self.comment))

        # ── 5 Read Sequence ---------------------------------------------------
        self.seq.clear()
        self.seq.resize(UInt32(slen_val)) # Allocate space
        var seq_bytes_read = self.reader.read_bytes(self.seq.as_span[mut=True](), slen_val)

        if seq_bytes_read != slen_val:
            return -3
          
        self.last_char = 0
        return len(self.seq)


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

    var first = reader.read_fastxpp_strip_newline()
    print("first‑read returned", first)

    var count = 0
    while True:
        var n = reader.read_fastxpp_strip_newline()
        if n < 0:
            break
        count += 1
        #print("rec#", count, "seq_len", n, "hdr_len", len(reader.name), "hdr_raw", reader.name.to_string())
        #print("rec#", count, "seq_len", n, "hdr_len", len(reader.name), "seq raw", reader.seq.to_string())
        print("rec#", count, "seq_len", n, "hdr_len", len(reader.name))
