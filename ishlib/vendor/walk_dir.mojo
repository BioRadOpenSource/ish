"""Walk directories and gather paths.

```mojo
from pathlib.path import cwd

from ishlib.vendor.walk_dir import walk_dir

def main():
    var paths = walk_dir[ignore_dot_files=True](cwd())
    for path in paths:
        print(path)
```
"""
from pathlib.path import Path, cwd
from collections.deque import Deque


fn walk_dir[
    *, ignore_dot_files: Bool
](path: Path,) raises -> List[Path]:
    """Walk dirs and collect all files.

    Note that this uses a heap allocated queue instead of recursion.

    Args:
        path: The path to begin the search.

    Returns:
        A list of files in all dirs.

    Paramaters:
        ignore_dot_files: If True, skip all dot files and dot dirs.

    """
    var out = List[Path]()
    var to_examine = Deque[Path](path)

    while len(to_examine) > 0:
        var check = to_examine.pop()
        for path in check.listdir():
            var child = check / path

            @parameter
            if ignore_dot_files:
                if String(path).startswith("."):
                    continue
            if child.is_file():
                out.append(child)
            elif child.is_dir():
                to_examine.append(child)
    return out
