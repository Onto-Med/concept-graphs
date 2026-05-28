from unittest import TestCase

from src.core.data_functions import DataProcessingFactory


class TestDataProcessing(TestCase):
    def test_data_processing_class_is_exposed_by_factory(self):
        self.assertTrue(hasattr(DataProcessingFactory, "DataProcessing"))
