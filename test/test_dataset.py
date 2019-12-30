import unittest
from dataset import Dataset, Dataloader
import json
import numpy as np


class TestDataset(unittest.TestCase):
    testfile = 'test/testfile.txt'
    first_content_parsed = np.array([
        "495906 b21 hp x5560 2 80ghz ml350 g6 , null new wholesale price",
        "null , 495906 b21 hp x5560 2 80ghz ml350 g6",
        1])

    def setUp(self):
        self.file_content = list(map(lambda x: json.loads(x), open(self.testfile, "r").readlines()))

    def test_new(self):
        dataset = Dataset(self.testfile)
        self.assertIsNotNone(dataset)

    def test_getitem_from_new(self):
        test_num_words = 10
        test_max_len = 20
        dataset = Dataset(self.testfile, num_words=test_num_words)
        assert dataset is not None

        item = dataset[0]

        self.assertEqual(len(item), len(self.first_content_parsed))
        self.assertEqual(len(item[0]), test_max_len)
        self.assertEqual(len(item[1]), test_max_len)


class TestDataloader(unittest.TestCase):
    testfile = 'test/testfile.txt'

    def setUp(self):
        self.max_len = 20

        self.dataset = Dataset(self.testfile, 20, max_len=self.max_len)

    def test_new(self):
        dataloader = Dataloader(self.dataset)
        self.assertIsNotNone(dataloader)

    def test_getitem(self):
        batch_size = 2
        dataloader = Dataloader(self.dataset, batch_size=batch_size)

        item = dataloader[0]
        self.assertEqual(len(item[0]), 2)
        self.assertEqual(len(item[0][0]), batch_size)
        self.assertEqual(len(item[0][1]), batch_size)

        self.assertEqual(len(item[0][0][0]), self.max_len)
        self.assertEqual(len(item[0][0][1]), self.max_len)
