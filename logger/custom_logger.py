import typing as tp


class CustomLogger:
    """
    Lil custom logger. It'll be useful in the case you want to simulate swarm behavior multiple times.
    """
    def __init__(self, path_to_file: str, columns_name: list[str], buffer_size: int = 20) -> None:
        """
        Args:
            path_to_file: path to file where you want to store simulation's results;
            columns_name: names of your data columns;
            buffer_size: size of interim buffer;
        """
        self._path_to_file: str = path_to_file
        self._buffer_size: int = buffer_size
        self._buffer: list[list[tp.Any]] = []

        with open(path_to_file, "w") as logs:
            row = ",".join(columns_name)
            logs.write(row + "\n")

    def write(self, content: list[tp.Any]) -> None:
        """
        Adds your content to buffer; If it's large enough buffer's content will be read into the file;
        Args:
            content: data you want to store;
        Returns:
            Nothing;
        """
        self._buffer.append(content)
        if len(self._buffer) == self._buffer_size:
            with open(self._path_to_file, "a") as logs:
                for stored_content in self._buffer:
                    row = ",".join([str(elem) for elem in stored_content])
                    logs.write(row + "\n")
            self._buffer = []

    def check_buffer(self) -> None:
        """
        If you've already pass all the data you want to store, but the buffer still contains your data,
        this method will flush it;
        Returns:
            Nothing;
        """
        if len(self._buffer) != 0:
            with open(self._path_to_file, "a") as logs:
                for stored_content in self._buffer:
                    row = ",".join([str(elem) for elem in stored_content])
                    logs.write(row + "\n")
            self._buffer = []
