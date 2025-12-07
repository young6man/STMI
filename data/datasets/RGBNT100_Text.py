# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os.path as osp
from .bases import BaseImageDataset
import json


class RGBNT100_Text(BaseImageDataset):

    dataset_dir = 'RGBNT100/rgbir'

    def __init__(self, root='', verbose=True, cfg=None, **kwargs):
        super(RGBNT100_Text, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        self.prefix = cfg.MODEL.PREFIX
        if self.prefix:
            print('~~~~~~~【We use modality prefix Here!】~~~~~~~')
        else:
            print('~~~~~~~【We do not use modality prefix Here!】~~~~~~~')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.train_text_dir = osp.join(self.dataset_dir, 'text')
        self.query_text_dir = osp.join(self.dataset_dir, 'text')
        self.gallery_text_dir = osp.join(self.dataset_dir, 'text')
        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)
        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        # pdb.set_trace()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def find_annotation(self,annotation_list, image_name):
        """从json列表中查找对应的文本标注"""
        for item in annotation_list:
            if item['item'] == image_name:  # 使用 'item' 作为图像文件名的键
                return item.get('description', "")  # 'description' 存储文本标注
        return ""
    def _process_dir(self, dir_path, text_dir_path, relabel=False):
        prefix = 'train' if 'train' in dir_path else 'test'
        # 加载RGB、NI、TI的文本标注json文件
        json_file_RGB = osp.join(text_dir_path, prefix + '.json')
        json_file_NI = osp.join(text_dir_path, prefix + '.json')
        json_file_TI = osp.join(text_dir_path, prefix + '.json')

        with open(json_file_RGB, 'r') as f_rgb:
            text_annotations_RGB = json.load(f_rgb)  # 假设是列表格式
        with open(json_file_NI, 'r') as f_ni:
            text_annotations_NI = json.load(f_ni)  # 假设是列表格式
        with open(json_file_TI, 'r') as f_ti:
            text_annotations_TI = json.load(f_ti)  # 假设是列表格式

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # pdb.set_trace()
            # if pid == -1: continue  # junk images are just ignored
            assert 1 <= pid <= 600  # pid == 0 means background
            assert 1 <= camid <= 8
            trackid = -1
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            jpg_name = img_path.split('/')[-1]
            if self.prefix:
                text_annotation_RGB = 'An image of a ' + self.prompt + 'vehicle in the visible spectrum: ' + self.find_annotation(
                    text_annotations_RGB, jpg_name)
                text_annotation_NI = 'An image of a ' + self.prompt + 'vehicle in the near infrared spectrum: ' + self.find_annotation(
                    text_annotations_NI, jpg_name)
                text_annotation_TI = 'An image of a ' + self.prompt + 'vehicle in the thermal infrared spectrum: ' + self.find_annotation(
                    text_annotations_TI, jpg_name)
                # text_annotation_RGB = 'An image of a ' + self.prompt + 'vehicle in the visible spectrum, capturing natural colors and fine details: ' + self.find_annotation(
                #     text_annotations_RGB, jpg_name)
                # text_annotation_NI = 'An image of a ' + self.prompt + 'vehicle in the near infrared spectrum, capturing contrasts and surface reflectance: ' + self.find_annotation(
                #     text_annotations_NI, jpg_name)
                # text_annotation_TI = 'An image of a ' + self.prompt + 'vehicle in the thermal infrared spectrum, capturing heat emissions as temperature gradients: ' + self.find_annotation(
                #     text_annotations_TI, jpg_name)
            else:
                text_annotation_RGB = self.find_annotation(text_annotations_RGB, jpg_name)
                text_annotation_NI = self.find_annotation(text_annotations_NI, jpg_name)
                text_annotation_TI = self.find_annotation(text_annotations_TI, jpg_name)
            dataset.append((img_path, pid, camid, trackid, text_annotation_RGB, text_annotation_NI, text_annotation_TI))
        return dataset
