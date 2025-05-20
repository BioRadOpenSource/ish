# This code is MIT Licensed by Seth Stadick

from runtime.asyncrt import DeviceContextPtr, TaskGroup, parallelism_level
from os.atomic import Atomic
from time import sleep


# Example: https://github.com/modular/modular/blob/ce57a36bdc97c1fe2419f0070fa7d8469e4bcfdf/mojo/stdlib/src/algorithm/functional.mojo#L354
fn run[
    thread_a_part1: fn () capturing -> Int,
    thread_a_part2: fn () capturing -> Int,
    thread_b_part1: fn () capturing -> Int,
    thread_b_part2: fn () capturing -> Int,
]():
    """Alternate work on two threads.

    While Thread a is doing part1, Thread b is doing part2,
    and then flip.

    It is expected that part2 depends on part1, and that
    both part1 and part2 have checks to make sure they need to run.

    Any fn can trigger a shutdown by returning -1.
    """
    var a_1_done = Atomic[DType.uint8](0)
    var a_2_done = Atomic[DType.uint8](0)
    var b_1_done = Atomic[DType.uint8](1)
    var b_2_done = Atomic[DType.uint8](1)
    var shutdown = Atomic[DType.uint8](0)
    alias SLEEP_TIME = 1.0

    @parameter
    @always_inline
    async fn thread_a():
        while True:
            if b_2_done.load() == 1:
                _ = b_2_done.fetch_sub(1)
                if thread_a_part2() < 0:
                    _ = shutdown.fetch_add(1)
                _ = a_2_done.fetch_add(1)

            if shutdown.load() > 0:
                print("shutting down a")
                break

            if b_1_done.load() == 1:
                _ = b_1_done.fetch_sub(1)
                if thread_a_part1() < 0:
                    _ = shutdown.fetch_add(1)
                _ = a_1_done.fetch_add(1)

            sleep(SLEEP_TIME)

    @parameter
    @always_inline
    async fn thread_b():
        while True:
            if a_2_done.load() == 1:
                _ = a_2_done.fetch_sub(1)
                if thread_b_part2() < 0:
                    _ = shutdown.fetch_add(1)
                _ = b_2_done.fetch_add(1)

            if shutdown.load() > 0:
                print("shutting down b")
                break

            if a_1_done.load() == 1:
                _ = a_1_done.fetch_sub(1)
                if thread_b_part1() < 0:
                    _ = shutdown.fetch_add(1)
                _ = b_1_done.fetch_add(1)

            sleep(SLEEP_TIME)

    var tg = TaskGroup()
    tg.create_task(thread_a())
    tg.create_task(thread_b())
    tg.wait()


def main():
    var from_values = List[Int](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    var a_to_values = List[Int]()
    var b_to_values = List[Int]()

    @parameter
    fn a_part1() capturing -> Int:
        if len(from_values) > 0:
            a_to_values.append(from_values.pop())
            return 0
        else:
            return -1

    @parameter
    fn b_part1() capturing -> Int:
        if len(from_values) > 0:
            b_to_values.append(from_values.pop())
            return 0
        else:
            return -1

    @parameter
    fn a_part2() capturing -> Int:
        if len(a_to_values) > 0:
            print("a", a_to_values[0])
            a_to_values.clear()
        return 0

    @parameter
    fn b_part2() capturing -> Int:
        if len(b_to_values) > 0:
            print("b", b_to_values[0])
            b_to_values.clear()
        return 0

    run[a_part1, a_part2, b_part1, b_part2]()

    _ = a_to_values
    _ = b_to_values
    _ = from_values
