from typing import List, Iterable, Optional
from trackit.datasets.common.factory import DatasetFactory
from trackit.datasets.base.video.dataset import VideoDataset
from trackit.datasets.base.video.filter.func import apply_filters_on_video_dataset_unaligned_
from .constructor import UnalignedMultiModalObjectTrackingDatasetConstructorGenerator
from .specialization.memory_mapped.dataset import UnalignedMultiModalObjectTrackingDataset_MemoryMapped
from .specialization.memory_mapped.constructor import \
    construct_unalgined_multi_modal_object_tracking_dataset_memory_mapped_from_base_video_dataset

__all__ = ['UnalignedMultiModalObjectTrackingDatasetFactory']


class UnalignedMultiModalObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: Iterable):
        super(UnalignedMultiModalObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                     UnalignedMultiModalObjectTrackingDatasetConstructorGenerator,
                                                                     apply_filters_on_video_dataset_unaligned_,
                                                                     UnalignedMultiModalObjectTrackingDataset_MemoryMapped,
                                                                     construct_unalgined_multi_modal_object_tracking_dataset_memory_mapped_from_base_video_dataset)

    def construct(self, filters: Optional[Iterable] = None, cache_base_format: bool = True,
                  dump_human_readable: bool = False) -> List[UnalignedMultiModalObjectTrackingDataset_MemoryMapped]:
        return super(UnalignedMultiModalObjectTrackingDatasetFactory, self).construct(filters, cache_base_format,
                                                                             dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(UnalignedMultiModalObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache,
                                                                                            dump_human_readable)
