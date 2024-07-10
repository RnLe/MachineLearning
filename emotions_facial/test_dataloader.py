import dataloader


def test_load():
    dataloader.load("test/sample_dataset_1/train", "test/sample_dataset_1/test", 10)
