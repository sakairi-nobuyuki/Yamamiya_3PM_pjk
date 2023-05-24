# coding: utf-8

from recognizer.components.region_extractor import ObjectFilter


class TestObjectFilter:
    def test_init(self):
        filter = ObjectFilter()
        assert isinstance(filter, ObjectFilter)

    def test_detect_filter(self) -> None:
        filter = ObjectFilter()
        filtered = filter.run([(1, 1, 1, 1), (2, 1, 3, 4)])

        print(filtered)

        assert isinstance(filtered, list)

    def test_detect_filter_identity(self) -> None:

        filter = ObjectFilter(filtering_flag=False)
        filtered = filter.run([(1, 1, 1, 1), (2, 1, 3, 4)])
        assert isinstance(filtered, list)
