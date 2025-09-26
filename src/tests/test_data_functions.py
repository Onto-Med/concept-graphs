import pathlib
from unittest import TestCase

from data_functions import DataProcessing


class TestDataProcessing(TestCase):
    def test_data_path(self):
        dp = DataProcessing(base_data_path=pathlib.Path("."), file_ext="py")
        print(list(dp.data_entries))
        dp = DataProcessing(
            base_data_path=pathlib.Path(".."), sub_paths=["tests"], file_ext="py"
        )
        print(list(dp.data_entries))
        dp = DataProcessing(base_data_path=pathlib.Path(".."), file_ext="py")
        print(list(dp.data_entries))
