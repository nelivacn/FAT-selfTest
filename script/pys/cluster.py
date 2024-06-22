import os
import sys
import time
import datetime
import traceback
import subprocess
import threading
from typing import List
from queue import Queue
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from abcdict import AbcDict

taskid = None


def msg_info(info_str):
    print(f'{taskid}INFO:-{info_str}', flush=True)


def msg_error(error_str):
    print(f'{taskid}ERROR:-{error_str}', file=sys.stderr, flush=True)


def file2q(file_name: str, q: Queue, index: int, p_num: int):
    try:
        line_index = 0
        with open(file_name, 'r') as r_file:
            while line := r_file.readline():
                if line_index % p_num == index:
                    item = line.strip().split()
                    q.put(item)
                line_index += 1
        with open(file_name, 'r') as r_file:
            while line := r_file.readline():
                if line_index % p_num == index:
                    item = line.strip().split()
                    q.put(item)
                line_index += 1
        q.put(None)
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(1)


def q2q_list(in_None_num: int, all_item_num: int, in_q: Queue, out_q_list: List[Queue]):
    try:
        index, len_out, none_count = 0, len(out_q_list), 0
        while True:
            item = in_q.get()
            if item is None:
                none_count += 1
                if none_count == in_None_num:
                    if index == all_item_num:
                        for outp in out_q_list:
                            outp.put(None)
                    else:
                        raise RuntimeError('q2q_list None num error')
                    break
            else:
                q_index = index % len_out
                if out_q_list[q_index].full():
                    q_size_list = [i.qsize() for i in out_q_list]
                    out_q_list[q_size_list.index(min(q_size_list))].put(item)
                else:
                    out_q_list[index % len_out].put(item)
                index += 1
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(2)


def get_feature_tester(fat, test_item_q, feat_q, gfbn):
    try:
        def get_feature_batch(fat, feat_q, inner_item_list):
            _img_data_list = []
            for _i in inner_item_list:
                _img_data = cv2.imread(_i[0], cv2.IMREAD_COLOR)
                _img_data_list.append(_img_data)
            stime = datetime.datetime.now()
            _isS, _feat = fat.get_feature(_img_data_list)
            assert isinstance(_isS, list) and isinstance(_feat, list), 'fat.get_feature 接口返回不对'
            gftime = (datetime.datetime.now() - stime).total_seconds()
            for index in range(len(_img_data_list)):
                _isSi = _isS[index]
                _feati = _feat[index]
                assert isinstance(_isSi, bool) and isinstance(_feati, np.ndarray), 'fat.get_feature 接口返回不对'
                _test_itemi = inner_item_list[index]
                reitem = [_isSi, _feati, gftime, _test_itemi]
                feat_q.put(reitem)
        _item_list = []
        while True:
            test_item = test_item_q.get()
            if test_item is None:
                if len(_item_list) > 0:
                    get_feature_batch(fat, feat_q, _item_list)
                feat_q.put(None)
                break
            else:
                _item_list.append(test_item)
                if len(_item_list) == gfbn:
                    get_feature_batch(fat, feat_q, _item_list)
                    _item_list = []
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(3)


def main(pyfat_file, cfg):
    try:
        TASK_ID = cfg.task_id
        LD_LIBRARY_PATH = cfg.ld_library_path
        fat_dir = pyfat_file.parent
        assets_dir = str(fat_dir / 'assets')
        sys.path.append(str(fat_dir))
        msg_info(f'TASK_ID: {TASK_ID}')
        msg_info(f'LD_LIBRARY_PATH: {LD_LIBRARY_PATH}')
        msg_info(f'fat_dir: {fat_dir}')
        DEVICE = [0, 1]
        gallery_file = cfg.gallery_file
        gallery_count = cfg.gallery_count * 2

        from pyfat_implement import PyFAT

        msg_info('fat init start')
        fat = PyFAT(gallery_count)
        msg_info('fat.load start')
        msg_info(f'assets_dir: {assets_dir}, device: {DEVICE}')
        fat.load(assets_dir, DEVICE)
        msg_info('fat.get_feature_parallel_num start')
        gfpn, gfbn = fat.get_feature_parallel_num()

        assert isinstance(gfpn, int), 'fat.get_feature_parallel_num 接口返回格式不对'
        assert isinstance(gfbn, int), 'fat.get_feature_parallel_num 接口返回格式不对'

        msg_info('fat.get_feature_len start')
        get_feature_len_res = fat.get_feature_len()

        assert isinstance(get_feature_len_res, int), 'fat.get_feature_len 接口返回格式不对'

        load_sample_item_num = cfg.load_test_item_num
        sample_item_q = Queue(1024)
        feat_q = Queue(1024)
        file2test_item_p_list = []
        for i in range(load_sample_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        gallery_file, sample_item_q, i, load_sample_item_num
                    )
                )
            )
        for i in file2test_item_p_list:
            i.start()

        # gallery_feat_q = Queue(1024)
        q_list = [Queue(1024) for _ in range(gfpn)]

        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_sample_item_num, gallery_count, sample_item_q, q_list
            )
        )
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(
                threading.Thread(
                    target=get_feature_tester, args=(fat, q_list[i], feat_q, gfbn)
                )
            )
        for pp in p_list:
            msg_info('in get_feature_tester')
            pp.start()

        insert_count, feat_none_count, = 0, 0
        insert_time_list, progress_time_list, get_feature_time_list = [], [], []
        progress_time = datetime.datetime.now()
        isFirst = True
        imgid_idx_map = {}
        feat_cls, feat_shape, feat_len, feat_type = '', None, '', ''
        while True:
            item = feat_q.get()
            if item is None:
                feat_none_count += 1
                if feat_none_count == gfpn:
                    if insert_count == gallery_count:
                        break
                    else:
                        sys.exit(1)
            else:
                _isSi, _feati, gftime, _test_itemi = item
                _, _imgid = _test_itemi
                if isFirst and _isSi:
                    msg_info('check feat info')
                    try:
                        feat_type = type(_feati).__name__
                        if isinstance(_feati, np.ndarray):
                            feat_cls = _feati.dtype.__str__()
                            feat_type += f'.{feat_cls}'
                            feat_shape = np.shape(_feati)
                        _feat_len = len(_feati)
                        if feat_shape:
                            feat_len = f'len: {_feat_len}, shape: {feat_shape}'
                        else:
                            feat_len = f'len: {_feat_len}'
                    except Exception:
                        msg_info('get feat type len error')
                    msg_info(f'feat type is {feat_type}, feat len is {feat_len}')
                    isFirst = False

                imgid_idx_map[_imgid] = insert_count
                insert_start_time = datetime.datetime.now()

                # fat.insert_gallery(item[1], int(item[3][3]), int(item[3][3]), item[0])
                fat.insert_gallery(_feati, insert_count, 0, _isSi)

                insert_end_time = datetime.datetime.now()
                insert_time_list.append((insert_end_time - insert_start_time).total_seconds())

                progress_time_now = datetime.datetime.now()
                progress_time_item = (progress_time_now - progress_time).total_seconds()

                progress_time = progress_time_now
                insert_count += 1
                get_feature_time_list.append(gftime)
                progress_time_list.append(progress_time_item)
                if insert_count % 200 == 0:
                    mean_gf_item_time = np.array(get_feature_time_list).mean()
                    tps_gf = 1. / mean_gf_item_time * gfpn * gfbn
                    mean_progress_time = np.array(progress_time_list).mean()
                    mean_insert_time = np.array(insert_time_list).mean()
                    insert_time_list, progress_time_list, get_feature_time_list = [], [], []

                    msg_info(f'提特征每秒数量: {tps_gf}, 接口平均响应时间: {mean_gf_item_time}')
                    msg_info(f'fat.insert_gallery 平均响应时间: { mean_progress_time}')
                    msg_info(f'建库全流程平均耗时: {mean_insert_time}')

        fat.unload_feature()
        cluster_start_time = datetime.datetime.now()
        start_cluster = fat.start_cluster()
        assert isinstance(start_cluster, bool), 'fat.start_cluster 接口返回格式不对'

        while True:
            progress_cluster = fat.query_progress_cluster()
            assert isinstance(progress_cluster, int), 'fat.query_progress_cluster 接口返回格式不对'
            save_cluster_time = (datetime.datetime.now() - cluster_start_time).total_seconds()
            # assert save_cluster_time <=21600,'fat.query_progress_cluster 接口返回函数调用平均响应时间限制超出范围'
            msg_info(f'fat.query_progress_cluster接口返回函数调用平均响应时间:{save_cluster_time}')
            if progress_cluster >= 100:
                break
            time.sleep(6)
        qcr_time_list, qcr_count = [], 0
        save_qcr_time_list = []

        _idx = imgid_idx_map.values()

        for idx_ in _idx:
            qcr_start_time = datetime.datetime.now()
            qcr_cluster_id = fat.query_cluster_res(idx_)
            assert isinstance(qcr_cluster_id, int), 'query_cluster_res 接口返回格式不对'
            qcr_time_list.append((datetime.datetime.now() - qcr_start_time).total_seconds())
            qcr_count += 1
            if qcr_count % 200 == 0:
                mean_qcr_time = np.array(qcr_time_list).mean()
                if qcr_count % 2000 == 0:
                    save_qcr_time_list.append(mean_qcr_time)
        mean_save_qcr_time = np.mean(np.array(save_qcr_time_list))  # 函数调用平均响应时间限制
        # assert mean_save_qcr_time <= 0.002,'query_cluster_res 接口返回函数调用平均响应时间限制超出范围'
        msg_info(f'query_cluster_res接口返回函数调用平均响应时间:{mean_save_qcr_time}')
        qaoc_time_list, qaoc_count, save_qaoc_time_list = [], 0, []

        cluster_idxs = fat.get_all_clusters()
        clusters_num = fat.get_clusters_num()

        assert isinstance(cluster_idxs, list), 'fat.get_all_clusters 接口返回格式不对'
        assert isinstance(clusters_num, int), 'fat.get_clusters_num 接口返回格式不对'
        fat.unload_cluster()
        for idxi in cluster_idxs:
            qaoc_start_time = datetime.datetime.now()
            this_cluster_idx = fat.query_all_of_cluster(idxi)
            assert isinstance(this_cluster_idx, list), 'fat.query_all_of_cluster 接口返回格式不对'
            qaoc_time_list.append((datetime.datetime.now() - qaoc_start_time).total_seconds())
            cluster_size = fat.query_num_of_cluster(idxi)
            assert isinstance(cluster_size, int), 'fat.query_num_of_cluster 接口返回格式不对'
            main_id = fat.query_cover_idx(idxi)
            assert isinstance(main_id, int), 'fat.query_cover_idx 接口返回格式不对'
            qaoc_count += 1
            if qaoc_count % 200 == 0:
                mean_qaoc_time = np.array(qaoc_time_list).mean()
                if qaoc_count % 2000 == 0:
                    save_qaoc_time_list.append(mean_qaoc_time)
        mean_save_qaoc_time = np.mean(np.array(qaoc_time_list))
        # assert mean_save_qaoc_time <= 0.002, 'fat.query_all_of_cluster 接口返回函数调用平均响应时间限制超出范围'
        msg_info(f'fat.query_all_of_cluster接口返回函数调用平均响应时间:{mean_save_qaoc_time}')

    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(3)


if __name__ == '__main__':
    cluster_file = Path(sys.argv[0])
    yaml_file = cluster_file.with_suffix('.yaml')
    cfg = AbcDict(yaml_file)
    taskid = cfg.task_id
    test_count = int(sys.argv[-1])
    pyfat_file = Path(sys.argv[1])
    msg_info(f'聚类稳定性测试第{test_count}轮')
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    [msg_info(i) for i in result.stdout.split('\n')]
    if result.stderr:
        [msg_error(i) for i in result.stderr.split('\n')]
    time.sleep(.5)
    main(pyfat_file, cfg)
