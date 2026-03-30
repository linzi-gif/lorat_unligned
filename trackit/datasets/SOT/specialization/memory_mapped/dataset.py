import numpy as np
from typing import Optional
from trackit.miscellanies.operating_system.path import join_paths, join_mmot_paths
from trackit.datasets.common.specialization.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset
from trackit.datasets.common.specialization.memory_mapped.engine import ListMemoryMapped

__all__ = ['SingleObjectTrackingDataset_MemoryMapped', 'SingleObjectTrackingDataset_MemoryMapped_MM']
__version__ = 4


class SingleObjectTrackingDatasetFrame_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 frame_index: int, image_size,
                 bounding_box: Optional[np.ndarray], bounding_box_validity_flag: Optional[np.ndarray],
                 sequence_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.frame_attributes = sequence_attributes['frames'][frame_index]
        self.frame_index = frame_index
        self.image_size = image_size
        self.bounding_box = bounding_box
        self.bounding_box_validity_flag = bounding_box_validity_flag
        self.sequence_additional_attributes_loader = sequence_additional_attributes_loader

    def get_frame_index(self):
        return self.frame_index

    def get_bounding_box(self):
        return self.bounding_box

    def get_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag

    def get_image_size(self):
        return self.image_size

    def get_image_path(self):
        return join_paths(self.root_path, self.sequence_attributes['path'], self.frame_attributes['path'])

    def has_bounding_box(self):
        return self.bounding_box is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag is not None

    def get_all_frame_attribute_names(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(('frames', self.frame_index))

    def get_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(('frames', self.frame_index, name))

    def has_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(('frames', self.frame_index, name))

    def get_all_object_attribute_names(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(('frames', self.frame_index, 'object'))

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(('frames', self.frame_index, name, 'object'))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(('frames', self.frame_index, name, 'object'))


class SingleObjectTrackingDatasetSequence_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 image_size_matrix: np.ndarray,
                 bounding_box_matrix: np.ndarray, bounding_box_validity_flag_matrix: np.ndarray,
                 sequence_additional_attributes_loader:LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.image_size_matrix = image_size_matrix
        self.bounding_box_matrix = bounding_box_matrix
        self.bounding_box_validity_flag_vector = bounding_box_validity_flag_matrix
        self.sequence_additional_attributes = sequence_additional_attributes_loader

    def get_name(self):
        return self.sequence_attributes['name']

    def has_fps(self):
        return 'fps' in self.sequence_attributes

    def get_fps(self):
        return self.sequence_attributes['fps']

    def has_bounding_box(self):
        return self.bounding_box_matrix is not None

    def has_category_id(self):
        return 'category_id' in self.sequence_attributes

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector is not None

    def has_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute(name)

    def get_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute(name)

    def get_all_frame_sizes(self):
        if self.image_size_matrix is None:
            return self.sequence_attributes['frame_size'][None, :].repeat(len(self), axis=0)
        return self.image_size_matrix

    def get_all_sequence_attribute_names(self):
        return self.sequence_additional_attributes.get_all_attribute_name()

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(('object', name))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(('object', name))

    def get_all_object_attribute_names(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(('object',))

    def get_all_bounding_boxes(self):
        return self.bounding_box_matrix

    def get_all_bounding_box_validity_flags(self):
        return self.bounding_box_validity_flag_vector

    def get_category_id(self):
        return self.sequence_attributes['category_id']

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        if self.image_size_matrix is not None:
            image_size = self.image_size_matrix[index, :]
        else:
            image_size = self.sequence_attributes['frame_size']

        bounding_box = self.bounding_box_matrix[index, :] if self.bounding_box_matrix is not None else None

        bounding_box_validity_flag = self.bounding_box_validity_flag_vector[index] if self.bounding_box_validity_flag_vector is not None else None

        return SingleObjectTrackingDatasetFrame_MemoryMapped(self.root_path,
                                                             self.sequence_attributes, index, image_size,
                                                             bounding_box, bounding_box_validity_flag,
                                                             self.sequence_additional_attributes)

    def __len__(self):
        return len(self.sequence_attributes['frames'])


class SingleObjectTrackingDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(SingleObjectTrackingDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'SingleObjectTracking')

    @staticmethod
    def load(root_path: str, path: str):
        return SingleObjectTrackingDataset_MemoryMapped(root_path, MemoryMappedDataset._load_storage(path))

    def __getitem__(self, index: int):
        sequence_attribute = self.storage[self.index_matrix[index, 0]]

        image_size_matrix_index = self.index_matrix[index, 1]
        image_size_matrix = self.storage[image_size_matrix_index] if image_size_matrix_index != -1 else None

        bounding_box_matrix_index = self.index_matrix[index, 2]
        bounding_box_matrix = self.storage[bounding_box_matrix_index] if bounding_box_matrix_index != -1 else None

        bounding_box_validity_flag_vector_index = self.index_matrix[index, 3]
        bounding_box_validity_flag_vector = self.storage[
            bounding_box_validity_flag_vector_index] if bounding_box_validity_flag_vector_index != -1 else None

        sequence_additional_attributes_index = self.index_matrix[index, 4]

        if sequence_additional_attributes_index != -1:
            sequence_additional_attributes = LazyAttributesLoader(self.storage,
                                                                         sequence_additional_attributes_index)
        else:
            sequence_additional_attributes = DummyAttributesLoader()

        return SingleObjectTrackingDatasetSequence_MemoryMapped(self.root_path,
                                                                sequence_attribute, image_size_matrix,
                                                                bounding_box_matrix,
                                                                bounding_box_validity_flag_vector,
                                                                sequence_additional_attributes)
    
class SingleObjectTrackingDatasetFrame_MemoryMapped_MM:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 frame_index: int, image_size_v, image_size_i, 
                 bounding_box_v: Optional[np.ndarray],  bounding_box_validity_flag_v: Optional[np.ndarray], 
                 bounding_box_i: Optional[np.ndarray], bounding_box_validity_flag_i: Optional[np.ndarray],
                 sequence_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.frame_attributes = sequence_attributes['frames'][frame_index]
        self.frame_index = frame_index
        self.image_size_v = image_size_v
        self.image_size_i = image_size_i
        self.bounding_box_v = bounding_box_v
        self.bounding_box_i = bounding_box_i
        self.bounding_box_validity_flag_v = bounding_box_validity_flag_v
        self.bounding_box_validity_flag_i = bounding_box_validity_flag_i
        self.sequence_additional_attributes_loader = sequence_additional_attributes_loader

    def get_frame_index(self):
        return self.frame_index

    def get_bounding_box_v(self):
        return self.bounding_box_v
    
    def get_bounding_box_i(self):
        return self.bounding_box_i
    
    def get_bounding_box_validity_flag_v(self):
        return self.bounding_box_validity_flag_v
    
    def get_bounding_box_validity_flag_i(self):
        return self.bounding_box_validity_flag_i
    
    def get_image_size_v(self):
        return self.image_size_v
    
    def get_image_size_i(self):
        return self.image_size_i
    
    def get_image_path(self):
        return join_mmot_paths(self.root_path, self.sequence_attributes['path'], self.frame_attributes['path'])

    def has_bounding_box(self):
        return self.bounding_box_v is not None and self.bounding_box_i is not None 

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_v is not None and self.bounding_box_validity_flag_i is not None

    def get_all_frame_attribute_names(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(
            ('frames', self.frame_index))

    def get_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(('frames', self.frame_index, name))

    def has_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(('frames', self.frame_index, name))

    def get_all_object_attribute_names(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(
            ('frames', self.frame_index, 'object'))

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(
            ('frames', self.frame_index, name, 'object'))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(
            ('frames', self.frame_index, name, 'object'))
    
class SingleObjectTrackingDatasetSequence_MemoryMapped_MM:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 image_size_matrix_v: np.ndarray, image_size_matrix_i: np.ndarray,
                 bounding_box_matrix_v: np.ndarray, bounding_box_validity_flag_matrix_v: np.ndarray,
                 bounding_box_matrix_i: np.ndarray, bounding_box_validity_flag_matrix_i: np.ndarray,
                 sequence_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.image_size_matrix_v = image_size_matrix_v
        self.bounding_box_matrix_v = bounding_box_matrix_v
        self.bounding_box_validity_flag_vector_v = bounding_box_validity_flag_matrix_v
        self.image_size_matrix_i = image_size_matrix_i
        self.bounding_box_matrix_i = bounding_box_matrix_i
        self.bounding_box_validity_flag_vector_i = bounding_box_validity_flag_matrix_i
        self.sequence_additional_attributes = sequence_additional_attributes_loader

    def get_name(self):
        return self.sequence_attributes['name']

    def has_fps(self):
        return 'fps' in self.sequence_attributes

    def get_fps(self):
        return self.sequence_attributes['fps']

    def has_bounding_box(self):
        return self.bounding_box_matrix_v is not None and self.bounding_box_matrix_i is not None

    def has_category_id(self):
        return 'category_id' in self.sequence_attributes

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector_v is not None and self.bounding_box_validity_flag_vector_i is not None

    def has_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute(name)

    def get_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute(name)

    def get_all_frame_sizes_v(self):
        if self.image_size_matrix_v is None:
            return self.sequence_attributes['frame_size_v'][None, :].repeat(len(self), axis=0)
        return self.image_size_matrix_v
    
    def get_all_frame_sizes_v(self):
        if self.image_size_matrix_i is None:
            return self.sequence_attributes['frame_size_i'][None, :].repeat(len(self), axis=0)
        return self.image_size_matrix_i

    
    def get_all_sequence_attribute_names(self):
        return self.sequence_additional_attributes.get_all_attribute_name()

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(('object', name))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(('object', name))

    def get_all_object_attribute_names(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(('object',))

    def get_all_bounding_boxes_v(self):
        return self.bounding_box_matrix_v
    
    def get_all_bounding_boxes_i(self):
        return self.bounding_box_matrix_i
    
    def get_all_bounding_box_validity_flags_v(self):
        return self.bounding_box_validity_flag_vector_v
    
    def get_all_bounding_box_validity_flags_i(self):
        return self.bounding_box_validity_flag_vector_i

    def get_category_id(self):
        return self.sequence_attributes['category_id']

    def __getitem__(self, index: int):
        if type(index) is not int:
            index = int(index)
        if index >= len(self):
            raise IndexError
        
        if self.image_size_matrix_v is not None:
            image_size_v = self.image_size_matrix_v[index, :]
        else:
            image_size_v = self.sequence_attributes['frame_siz_v']

        if self.image_size_matrix_i is not None:
            image_size_i = self.image_size_matrix_i[index, :]
        else:
            image_size_i = self.sequence_attributes['frame_size_i']

        bounding_box_v = self.bounding_box_matrix_v[index, :] if self.bounding_box_matrix_v is not None else None
        bounding_box_i = self.bounding_box_matrix_i[index, :] if self.bounding_box_matrix_i is not None else None

        bounding_box_validity_flag_v = self.bounding_box_validity_flag_vector_v[index] if self.bounding_box_validity_flag_vector_v is not None else None
        bounding_box_validity_flag_i = self.bounding_box_validity_flag_vector_i[index] if self.bounding_box_validity_flag_vector_i is not None else None
        
        return SingleObjectTrackingDatasetFrame_MemoryMapped_MM(self.root_path,
                                                             self.sequence_attributes, index,
                                                             image_size_v, image_size_i,
                                                             bounding_box_v, bounding_box_validity_flag_v,
                                                             bounding_box_i, bounding_box_validity_flag_i,
                                                             self.sequence_additional_attributes)

    def __len__(self):
        return len(self.sequence_attributes['frames'])


class SingleObjectTrackingDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(SingleObjectTrackingDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'SingleObjectTracking')

    @staticmethod
    def load(root_path: str, path: str):
        return SingleObjectTrackingDataset_MemoryMapped(root_path, MemoryMappedDataset._load_storage(path))

    def __getitem__(self, index: int):
        sequence_attribute = self.storage[self.index_matrix[index, 0]]

        image_size_matrix_index = self.index_matrix[index, 1]
        image_size_matrix = self.storage[image_size_matrix_index] if image_size_matrix_index != -1 else None

        bounding_box_matrix_index = self.index_matrix[index, 2]
        bounding_box_matrix = self.storage[bounding_box_matrix_index] if bounding_box_matrix_index != -1 else None

        bounding_box_validity_flag_vector_index = self.index_matrix[index, 3]
        bounding_box_validity_flag_vector = self.storage[
            bounding_box_validity_flag_vector_index] if bounding_box_validity_flag_vector_index != -1 else None

        sequence_additional_attributes_index = self.index_matrix[index, 4]

        if sequence_additional_attributes_index != -1:
            sequence_additional_attributes = LazyAttributesLoader(self.storage,
                                                                         sequence_additional_attributes_index)
        else:
            sequence_additional_attributes = DummyAttributesLoader()

        return SingleObjectTrackingDatasetSequence_MemoryMapped(self.root_path,
                                                                sequence_attribute, image_size_matrix,
                                                                bounding_box_matrix,
                                                                bounding_box_validity_flag_vector,
                                                                sequence_additional_attributes)


class SingleObjectTrackingDataset_MemoryMapped_MM(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(SingleObjectTrackingDataset_MemoryMapped_MM, self).__init__(
            root_path, storage, __version__, 'SingleObjectTracking'
        )

    @staticmethod
    def load(root_path: str, path: str):
        return SingleObjectTrackingDataset_MemoryMapped_MM(
            root_path,
            MemoryMappedDataset._load_storage(path)
        )

    def __getitem__(self, index: int):
        # 第 0 列：序列属性
        sequence_attribute = self.storage[self.index_matrix[index, 0]]

        # 一个小工具函数，避免重复写三元表达式
        def _maybe_load(idx: int):
            return self.storage[idx] if idx != -1 else None

        # 第 1,2 列：RGB / TIR 所有帧尺寸矩阵
        image_size_matrix_v = _maybe_load(self.index_matrix[index, 1])
        image_size_matrix_i = _maybe_load(self.index_matrix[index, 2])

        # 第 3,4 列：RGB / TIR bbox 矩阵
        bounding_box_matrix_v = _maybe_load(self.index_matrix[index, 3])
        bounding_box_matrix_i = _maybe_load(self.index_matrix[index, 4])

        # 第 5,6 列：RGB / TIR bbox 有效标记向量
        bounding_box_validity_flag_vector_v = _maybe_load(self.index_matrix[index, 5])
        bounding_box_validity_flag_vector_i = _maybe_load(self.index_matrix[index, 6])

        # 第 7 列：额外属性
        sequence_additional_attributes_index = self.index_matrix[index, 7]
        if sequence_additional_attributes_index != -1:
            sequence_additional_attributes = LazyAttributesLoader(
                self.storage, sequence_additional_attributes_index
            )
        else:
            sequence_additional_attributes = DummyAttributesLoader()

        # 返回一个「序列级」的 MM 对象
        return SingleObjectTrackingDatasetSequence_MemoryMapped_MM(
            self.root_path,
            sequence_attribute,
            image_size_matrix_v, image_size_matrix_i,
            bounding_box_matrix_v, bounding_box_validity_flag_vector_v,
            bounding_box_matrix_i, bounding_box_validity_flag_vector_i,
            sequence_additional_attributes
        )
