import typing as tp


class CustomLogger:
    def __init__(self, path_to_file: str):
        self._path_to_file = path_to_file

    def write(self, content: list[tp.Any], mode: str):
        with open(self._path_to_file, mode) as logs:
            row = ",".join([str(elem) for elem in content])
            logs.write(row + "\n")
