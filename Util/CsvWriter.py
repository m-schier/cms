class CsvWriter:
    """
    Wrapper for csv.DictWriter which
        a) Ensures all values are always present
        b) Removes the necessity to specify keys twice (initialization and row writing)
    """
    def __init__(self, path, sep=',', lineterminator='\n', flush_lines=True):
        self.__path = path
        self.__fp = None
        self.__writer = None
        self.__keys = None
        self.__delimiter = sep
        self.__lineterminator = lineterminator
        self.__flush_lines = flush_lines

    def __enter__(self):
        self.__fp = open(self.__path, 'w')
        return self

    def __exit__(self, *args):
        if self.__fp is not None:
            self.__fp.close()
            self.__fp = None

    def write_row(self, **kwargs):
        if self.__fp is None:
            raise RuntimeError("Must enter context first")

        if self.__writer is None:
            import csv

            self.__writer = csv.DictWriter(self.__fp, fieldnames=kwargs.keys(), delimiter=self.__delimiter,
                                           lineterminator=self.__lineterminator)
            self.__keys = set(kwargs.keys())
            self.__writer.writeheader()
        else:
            if self.__keys != set(kwargs.keys()):
                raise ValueError("Mismatching keys from first row")

        self.__writer.writerow(kwargs)

        if self.__flush_lines:
            self.__fp.flush()
