import fnmatch
import os
from typing import List, Tuple, Union


def match_files(path: str,
                pattern: Union[str, List],
                fuzzy: bool = False) -> List[Tuple[str, str]]:
    """
    Find files in a directory that match the given pattern(s).

    Args:
        path (str): The directory path to search for files.
        pattern (Union[str, List]): The pattern(s) to match against file names.
        fuzzy (bool, optional): Whether to perform fuzzy matching. If True,
            the pattern(s) will be treated as substrings to match against file names.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the matched file names
        (without the file extension) and their corresponding absolute paths.
        The list is sorted alphabetically by the file names.

    """
    if isinstance(pattern, str):
        pattern = [pattern]
    if fuzzy:
        pattern = [f'*{p}*' for p in pattern]
    files_list = []
    for root, _, files in os.walk(path):
        for name in files:
            for p in pattern:
                if fnmatch.fnmatch(name.lower(), p.lower()):
                    files_list.append((name[:-3], os.path.join(root, name)))
                    break

    return sorted(files_list, key=lambda x: x[0])
