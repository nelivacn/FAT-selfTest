import os
import sys
import time
import datetime
import threading
import traceback
import subprocess
from typing import List
from queue import Queue
from pathlib import Path

import cv2
import numpy as np
from abcdict import AbcDict

taskid = None


def msg_info(info_str):
    print(f'{taskid}INFO:-{info_str}', flush=True)


def msg_error(error_str):
    print(f'{taskid}ERROR:-{error_str}', file=sys.stderr, flush=True)


def file2q(file_name: Path, q: Queue, index: int, p_num: int):
    try:
        line_index = 0
        with file_name.open('r') as r_file:
            while True:
                line = r_file.readline()
                if not line:
                    break
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
        os._exit(1)


def get_topk_tester(fat, test_item_q, res_q, gtkbn):
    try:
        def get_topk_batch(fat, _item_list_inner, res_q):
            _feat_data_list, _usable_list = [], []
            for _i in _item_list_inner:
                _probe_feat = _i[1]
                _usable = _i[0]
                _feat_data_list.append(_probe_feat)
                _usable_list.append(_usable)
            stime = datetime.datetime.now()
            idxs, sims = fat.get_topk(_feat_data_list, _usable_list)

            assert isinstance(idxs, list) and isinstance(sims, list), 'fat.get_topk 接口返回格式不对'
            # assert 0.0 <= sims <= 1.0
            gtktime = (datetime.datetime.now() - stime).total_seconds()
            # print(_feat_data_list)
            # print(idxs, sims)
            for index in range(len(_feat_data_list)):
                _item_inner = _item_list_inner[index]
                _isSi = _item_inner[0]
                _probe_pid = _item_inner[1]
                _probe_imgid = _item_inner[3][1]
                _probe_url = _item_inner[3][0]
                # _probe_cls = _item_inner[3][3]

                _idx = idxs[index][0]
                _sim = sims[index][0]
                # print(_idx,_sim)
                # assert isinstance(_idx, np.int32) and isinstance(_sim, np.float32), 'fat.get_topk 接口返回格式不对'
                res_item = [_isSi, f'{_probe_pid}_{_probe_imgid}', _probe_url, gtktime, _idx, _sim]
                res_q.put(res_item)

        _item_list = []
        while True:
            test_item = test_item_q.get()
            if test_item is None:
                if len(_item_list) > 0:
                    get_topk_batch(fat, _item_list, res_q)
                res_q.put(None)
                break
            else:
                _item_list.append(test_item)
                if len(_item_list) == gtkbn:
                    get_topk_batch(fat, _item_list, res_q)
                    _item_list = []
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(1)


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
        msg_info('test get feature and get topk')
        DEVICE = [0, 1]
        gallery_file = Path(cfg.gallery_file)
        gallery_count = cfg.gallery_count
        probe_file = Path(cfg.probe_file)
        probe_count = cfg.probe_count

        from pyfat_implement import PyFAT

        msg_info('fat init start')
        fat = PyFAT(cfg.gallery_count, 1)
        msg_info('fat.load start')
        msg_info(f'assets_dir: {assets_dir}, device: {DEVICE}')
        load_code = fat.load(assets_dir, DEVICE)
        assert load_code == 0, 'fat.load 返回非0'
        msg_info('fat.get_feature_parallel_num start')
        gfpn, gfbn = fat.get_feature_parallel_num()
        assert isinstance(gfpn, int), 'fat.get_feature_parallel_num() 接口返回格式不对'
        assert isinstance(gfbn, int), 'fat.get_feature_parallel_num() 接口返回格式不对'
        msg_info('fat.get_topk_parallel_num start')
        gtkpn, gtkbn = fat.get_topk_parallel_num()
        assert isinstance(gtkpn, int), 'fat.get_topk_parallel_num() 接口返回格式不对'
        assert isinstance(gtkbn, int), 'fat.get_topk_parallel_num() 接口返回格式不对'

        msg_info('fat.get_feature_len start')
        get_feature_len_res = fat.get_feature_len()
        msg_info(f'fat.get_feature_len 接口返回: {get_feature_len_res}')
        assert isinstance(get_feature_len_res, int), 'get_feature_len_res 接口返回格式不对'

        load_test_item_num = cfg.load_test_item_num
        gallery_test_item_q = Queue(1024)
        probe_test_item_q = Queue(1024)
        probe_feat_q = Queue(1024)
        file2test_item_p_list = []
        for i in range(load_test_item_num):
            file2test_item_p_list.append(
                threading.Thread(target=file2q, args=(
                    gallery_file, gallery_test_item_q, i, load_test_item_num
                ))
            )
        for i in file2test_item_p_list:
            i.start()
        gallery_feat_q = Queue(1024)
        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(target=q2q_list, args=(
            load_test_item_num, gallery_count, gallery_test_item_q, q_list
        ))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(threading.Thread(target=get_feature_tester, args=(
                fat, q_list[i], gallery_feat_q, gfbn
            )))
        for pp in p_list:
            pp.start()

        insert_count = 0
        gallery_feat_none_count = 0
        insert_item_time_list, insert_gallery_item_time_list = [], []
        get_feature_item_time_list = []
        insert_item_time = datetime.datetime.now()
        feature2str_feature_list = []
        feature2str_feature_count = 0

        while True:
            item = gallery_feat_q.get()
            if item is None:
                gallery_feat_none_count += 1
                if gallery_feat_none_count == gfpn:
                    if insert_count == gallery_count:
                        break
                    else:
                        sys.exit(1)
            else:
                if feature2str_feature_count < 5000 and item[0]:
                    feature2str_feature_list.append(item[1])
                    feature2str_feature_count += 1
                get_feature_item_time_list.append(item[2])
                _insert_item_time = datetime.datetime.now()
                fat.insert_gallery(item[1], int(item[3][1]), 0, item[0])
                insert_gallery_item_time_list.append((datetime.datetime.now() - _insert_item_time).total_seconds())
                insert_item_time_list.append((_insert_item_time - insert_item_time).total_seconds())
                insert_item_time = _insert_item_time
                insert_count += 1
                if insert_count % 200 == 0:
                    mean_gf_item_time = np.array(get_feature_item_time_list).mean()
                    tps_gf = 1. / mean_gf_item_time * gfpn * gfbn
                    mean_insert_item_time = np.array(insert_item_time_list).mean()
                    mean_insert_gallery_item_time = np.array(insert_gallery_item_time_list).mean()
                    get_feature_item_time_list, insert_item_time_list = [], []
                    insert_gallery_item_time_list = []
                    msg_info(f'提特征每秒数量: {tps_gf:.6f}, 接口平均响应时间: {mean_gf_item_time:.6f}s')
                    msg_info(f'fat.insert_gallery 平均响应时间: {mean_insert_gallery_item_time:.6f}s')
                    msg_info(f'建库全流程平均耗时: {mean_insert_item_time:.6f}s')

        final_start = datetime.datetime.now()
        fat.finalize()
        finalize_time = (datetime.datetime.now() - final_start).total_seconds()
        msg_info(f'finalize time: {finalize_time}')

        file2test_item_p_list = []
        for i in range(load_test_item_num):
            file2test_item_p_list.append(
                threading.Thread(target=file2q, args=(
                    probe_file, probe_test_item_q, i, load_test_item_num
                ))
            )
        for i in file2test_item_p_list:
            i.start()

        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(target=q2q_list, args=(
            load_test_item_num, probe_count, probe_test_item_q, q_list
        ))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(threading.Thread(target=get_feature_tester, args=(
                fat, q_list[i], probe_feat_q, gfbn
            )))
        for pp in p_list:
            pp.start()

        qq_list = [Queue(1024) for _ in range(gtkpn)]
        qq2q_list_p = threading.Thread(target=q2q_list, args=(
            gfpn, probe_count, probe_feat_q, qq_list
        ))
        qq2q_list_p.start()

        res_q = Queue(1024)
        get_topk_p = []
        for i in range(gtkpn):
            get_topk_p.append(threading.Thread(target=get_topk_tester, args=(
                fat, qq_list[i], res_q, gtkbn
            )))
        for ppp in get_topk_p:
            ppp.start()

        det_gettopk_none_count, retrieval = 0, 0
        get_topk_time_time_list = []
        while True:
            item = res_q.get()
            if item is None:
                det_gettopk_none_count += 1
                if det_gettopk_none_count == gtkpn:
                    if retrieval == probe_count:
                        break
                    else:
                        sys.exit(1)
            else:
                get_topk_time_time_list.append(item[3])

                retrieval += 1
                if retrieval % 200 == 0:
                    mean_gtk_time_time = np.array(get_topk_time_time_list).mean()
                    tps_gtk = 1. / mean_gtk_time_time * gtkpn * gtkbn
                    msg_info(f'检索每秒数量: {tps_gtk:.6f}, 接口平均响应时间: {mean_gtk_time_time:.6f}s')

        f2s_time_list, gs_time_list = [], []
        for f2s_feat in feature2str_feature_list:
            f2st = datetime.datetime.now()
            feature_str = fat.feature_to_str([f2s_feat])
            assert isinstance(feature_str, list), 'fat.feature_to_str 接口返回格式不对'

            f2stime = (datetime.datetime.now() - f2st).total_seconds()
            f2s_time_list.append(f2stime)
            # feat_str_ = feature_str[0]['feature']
            # quality_str_ = feature_str[0]['quality']

            gst = datetime.datetime.now()
            sim = fat.get_sim(f2s_feat, feature2str_feature_list[0])
            gstime = (datetime.datetime.now() - gst).total_seconds()
            gs_time_list.append(gstime)
            assert isinstance(sim, float), 'fat.get_sim 接口返回格式不对'
            assert 0.0 <= sim <= 1.0, 'fat.get_sim 接口返回数值超出范围'

        f2s_TPS = 1. / np.mean(np.array(f2s_time_list))
        msg_info(f'fat.feature_to_str TPS: {f2s_TPS:.6f}')
        gs_TPS = 1. / np.mean(np.array(gs_time_list))
        msg_info(f'fat.get_sim TPS is: {gs_TPS:.6f}')
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]


def get_feature(pyfat_file, cfg):
    try:
        TASK_ID = cfg.task_id
        LD_LIBRARY_PATH = cfg.ld_library_path
        fat_dir = pyfat_file.parent
        assets_dir = str(fat_dir / 'assets')
        sys.path.append(str(fat_dir))
        msg_info(f'TASK_ID: {TASK_ID}')
        msg_info(f'LD_LIBRARY_PATH: {LD_LIBRARY_PATH}')
        msg_info(f'fat_dir: {fat_dir}')
        msg_info('test get feature and get topk')
        DEVICE = [0]
        gallery_file = Path(cfg.gallery_file)
        gallery_count = cfg.gallery_count

        from pyfat_implement import PyFAT

        msg_info('fat init start')
        fat = PyFAT(1, 1)
        msg_info('fat.load start')
        msg_info(f'assets_dir: {assets_dir}, device: {DEVICE}')
        load_code = fat.load(assets_dir, DEVICE)
        assert load_code == 0, 'fat.load 返回非0'
        msg_info('fat.get_feature_parallel_num start')
        gfpn, gfbn = fat.get_feature_parallel_num()
        assert isinstance(gfpn, int), 'fat.get_feature_parallel_num() 接口返回格式不对'
        assert isinstance(gfbn, int), 'fat.get_feature_parallel_num() 接口返回格式不对'

        msg_info('fat.get_feature_len start')
        get_feature_len_res = fat.get_feature_len()
        msg_info(f'fat.get_feature_len 接口返回: {get_feature_len_res}')
        assert isinstance(get_feature_len_res, int), 'get_feature_len_res 接口返回格式不对'

        load_test_item_num = cfg.load_test_item_num
        gallery_test_item_q = Queue(1024)
        file2test_item_p_list = []
        for i in range(load_test_item_num):
            file2test_item_p_list.append(
                threading.Thread(target=file2q, args=(
                    gallery_file, gallery_test_item_q, i, load_test_item_num
                ))
            )
        for i in file2test_item_p_list:
            i.start()

        gallery_feat_q = Queue(1024)
        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(target=q2q_list, args=(
            load_test_item_num, gallery_count, gallery_test_item_q, q_list
        ))
        q2q_list_p.start()

        p_list = []
        for i in range(gfpn):
            p_list.append(threading.Thread(target=get_feature_tester, args=(
                fat, q_list[i], gallery_feat_q, gfbn
            )))
        for pp in p_list:
            pp.start()

        insert_count = 0
        gallery_feat_none_count = 0
        get_feature_item_time_list = []

        while True:
            item = gallery_feat_q.get()
            if item is None:
                gallery_feat_none_count += 1
                if gallery_feat_none_count == gfpn:
                    if insert_count == gallery_count:
                        break
                    else:
                        sys.exit(1)
            else:
                get_feature_item_time_list.append(item[2])
                insert_count += 1
                if insert_count % 200 == 0:
                    mean_gf_item_time = np.array(get_feature_item_time_list).mean()
                    tps_gf = 1. / mean_gf_item_time * gfpn * gfbn
                    get_feature_item_time_list = []
                    msg_info(f'提特征每秒数量: {tps_gf}, 接口平均响应时间: {mean_gf_item_time}')
    except Exception:
        [msg_error(i) for i in traceback.format_exc().split('\n')]


if __name__ == '__main__':
    face1n1 = Path(sys.argv[0])
    yaml_file = face1n1.with_suffix('.yaml')
    cfg = AbcDict(yaml_file)
    taskid = cfg.task_id
    test_count = int(sys.argv[-1])
    CUDA_VISIBLE_DEVICES = cfg.cuda_visible_devices
    msg_info(f'CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}')
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    [msg_info(i) for i in result.stdout.split('\n')]
    if result.stderr:
        [msg_error(i) for i in result.stderr.split('\n')]
    time.sleep(.5)
    pyfat_file = Path(sys.argv[1])
    if test_count % 2:
        msg_info(f'检索稳定性测试第{test_count}轮 全流程')
        main(pyfat_file, cfg)
    else:
        msg_info(f'检索稳定性测试第{test_count}轮 提特征')
        get_feature(pyfat_file, cfg)
