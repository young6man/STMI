# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os
import os.path as osp
from .bases import BaseImageDataset
import json


class MSVR310_Text(BaseImageDataset):

    dataset_dir = 'MSVR310'

    def __init__(self, root='', verbose=True, cfg=None, **kwargs):
        super(MSVR310_Text, self).__init__()
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        self.prefix = cfg.MODEL.PREFIX
        if self.prefix:
            print('~~~~~~~【We use modality prefix Here!】~~~~~~~')
        else:
            print('~~~~~~~【We do not use modality prefix Here!】~~~~~~~')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.train_text_dir = osp.join(self.dataset_dir, 'text')
        self.query_text_dir = osp.join(self.dataset_dir, 'text')
        self.gallery_text_dir = osp.join(self.dataset_dir, 'text')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)
        # pdb.set_trace()
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

    def find_annotation(self, annotation_list, image_name):
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

        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
                _mask = vid_path.replace('/MSVR310/', '/MSVR310/mask/')
                mask_path = osp.join(_mask, img).replace('jpg', 'npy')
                vid = int(img[0:4])
                camid = int(img[11])
                sceneid = int(img[6:9])  # scene id
                assert 0 <= camid <= 7
                if relabel:
                    vid = vid2label[vid]
                jpg_name = img
                if self.prefix:
                    # text_annotation_RGB = 'An image of a ' + self.prompt + 'vehicle in the visible spectrum: ' + self.find_annotation(
                    #     text_annotations_RGB, jpg_name)
                    # text_annotation_NI = 'An image of a ' + self.prompt + 'vehicle in the near infrared spectrum: ' + self.find_annotation(
                    #     text_annotations_NI, jpg_name)
                    # text_annotation_TI = 'An image of a ' + self.prompt + 'vehicle in the thermal infrared spectrum: ' + self.find_annotation(
                    #     text_annotations_TI, jpg_name)
                    text_annotation_RGB = 'An image of a ' + self.prompt + 'vehicle in the visible spectrum, capturing natural colors and fine details: ' + self.find_annotation(
                        text_annotations_RGB, jpg_name)
                    text_annotation_NI = 'An image of a ' + self.prompt + 'vehicle in the near infrared spectrum, capturing contrasts and surface reflectance: ' + self.find_annotation(
                        text_annotations_NI, jpg_name)
                    text_annotation_TI = 'An image of a ' + self.prompt + 'vehicle in the thermal infrared spectrum, capturing heat emissions as temperature gradients: ' + self.find_annotation(
                        text_annotations_TI, jpg_name)
                else:
                    text_annotation_RGB = self.find_annotation(text_annotations_RGB, jpg_name)
                    text_annotation_NI = self.find_annotation(text_annotations_NI, jpg_name)
                    text_annotation_TI = self.find_annotation(text_annotations_TI, jpg_name)

                dataset.append(((r_img_path, n_img_path, t_img_path, mask_path), vid, camid, sceneid, text_annotation_RGB,
                                text_annotation_NI, text_annotation_TI))
        return dataset
