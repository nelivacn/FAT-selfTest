import os
import sys
import datetime
import traceback
import subprocess
import threading
from typing import List
from queue import Queue
from pathlib import Path

import cv2
import numpy as np
from abcdict import AbcDict
sys.path.append(str(Path(__file__).resolve().parent))

taskid = None
def msg_info(info_str):
	print(f'{taskid}INFO:-{info_str}', flush=True)

def msg_error(error_str):
	print(f'{taskid}ERROR:-{error_str}', file=sys.stderr, flush=True)

def file2q(file_name: Path, q: Queue, index: int, p_num: int):
    try:
        line_index = 0
        with open(file_name,'r') as r_file:
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

def q2q_list(in_None_num: int, all_item_num: int, in_q: Queue,out_q_list:List[Queue]):
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
    except Exception as e:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(2)
def get_info_tester(fat,test_item_q, res_q, gfbn):
    try:
        def get_info_batch(fat, res_q, inner_item_list):

            _img_data_list = []
            _pts1_list, _pts2_list = [], []
            for _i in inner_item_list:
                _img_data = cv2.imread(_i[0], cv2.IMREAD_COLOR)
                poin_list = _i[0].split('/')[-1].split('.')[0].split('_')
                x1,y1,x2,y2 =  poin_list[1], poin_list[2],poin_list[3],poin_list[4]
                _img_data =_img_data[int(y1):int(y2),int(x1):int(x2)]
                _img_data_list.append(_img_data)
                _pts1_list.append((int(x1), int(y1)))
                _pts2_list.append((int(x2), int(y2)))
            stime = datetime.datetime.now()
            # _bbox = fat.get_vehicle_bbox(_img_data_list, _pts1_list, _pts2_list)

            _info = fat.get_vehicle_info(_img_data_list, _pts1_list, _pts2_list)
            assert isinstance(_info,list) ,'fat.get_vehicle_info 接口返回格式不对'
            gftime = (datetime.datetime.now() - stime).total_seconds()
            for index in range(len(_img_data_list)):
                _infoi = _info[index]
                _test_itemi = inner_item_list[index]
                reitem = [_infoi, gftime, _test_itemi[0], _test_itemi[-1]]
                res_q.put(reitem)
        _item_list = []
        while True:
            test_item = test_item_q.get()
            if test_item is None:
                if len(_item_list) > 0:
                    get_info_batch(fat, res_q, _item_list)
                res_q.put(None)
                break
            else:
                _item_list.append(test_item)
                if len(_item_list) == gfbn:
                    get_info_batch(fat, res_q, _item_list)
                    _item_list = []
    except Exception as e:
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

        sample_file = cfg.sample_file
        sample_count = 0
        DEVICE = [0, 1]
        with open(sample_file,'r') as r_file:
            while True:
                line = r_file.readline()
                if not line:
                    break
                sample_count += 1
        msg_info('fat init start')
        from pyfat_implement import PyFAT
        msg_info('fat init start')
        fat = PyFAT(cfg.gallery_count, 1)
        msg_info('fat.load start')
        msg_info(f'assets_dir: {assets_dir}, device: {DEVICE}')
        fat.load(assets_dir, DEVICE)
        msg_info('fat.get_feature_parallel_num start')
        gfpn, gfbn = fat.get_info_parallel_num()
        assert isinstance(gfpn,int),'fat.get_info_parallel_num 接口返回格式不对'
        assert isinstance(gfbn,int),'fat.get_info_parallel_num 接口返回格式不对'
        gdpn, gdbn=fat.get_detect_parallel_num()
        assert isinstance(gdpn,int),'fat.get_detect_parallel_num 接口返回格式不对'
        assert isinstance(gdbn,int),'fat.get_detect_parallel_num 接口返回格式不对'

        load_sample_item_num = cfg.load_sample_item_num
        sample_item_q = Queue(1024)
        res_q = Queue(1024)
        msg_info('load sample')
        file2test_item_p_list = []
        for i in range(load_sample_item_num):
            file2test_item_p_list.append(
                threading.Thread(
                    target=file2q, args=(
                        sample_file, sample_item_q, i, load_sample_item_num
                    )
                )
            )
        for i in file2test_item_p_list:
            i.start()
        msg_info('sample get_info_item')
        q_list = [Queue(1024) for _ in range(gfpn)]
        q2q_list_p = threading.Thread(
            target=q2q_list, args=(
                load_sample_item_num, sample_count, sample_item_q, q_list
            )
        )
        q2q_list_p.start()
        p_list = []
        for i in range(gfpn):
            p_list.append(
                threading.Thread(
                    target=get_info_tester, args=(fat, q_list[i], res_q, gfbn)
                )
            )
        for pp in p_list:
            msg_info('in get_info_tester')
            pp.start()
        msg_info('save res begin')
        res_count, info_none_count = 0, 0
        progress_time_list, get_info_time_list = [], []
        progress_time = datetime.datetime.now()
        while True:
            item = res_q.get()
            if item is None:
                info_none_count += 1
                if info_none_count == gfpn:
                    if res_count == sample_count:
                        break
                    else:
                        sys.exit(1)
            else:
                _infoi, gftime, imgid, imgurl = item
                progress_time_now = datetime.datetime.now()
                progress_time_item = (progress_time_now - progress_time).total_seconds()
                progress_time = progress_time_now
                res_count += 1
                get_info_time_list.append(gftime)
                progress_time_list.append(progress_time_item)
                if res_count % 200 == 0:
                    mean_gf_item_time = np.array(get_info_time_list).mean()
                    tps_gf = 1. / mean_gf_item_time * gfpn * gfbn
                    assert mean_gf_item_time <= 0.5,'函数平均响应时间限制超出系统设置时间'
                    assert tps_gf >=20,'每秒处理图片数低于系统设置数量'
                    mean_progress_time = np.array(progress_time_list).mean()
                    progress_time_list, get_info_time_list = [], []
                    msg_info(f'提特征每秒数量: {tps_gf:.6f}, 接口平均响应时间: {mean_gf_item_time:.6f}s')
    except Exception as e:
        [msg_error(i) for i in traceback.format_exc().split('\n')]
        os._exit(4)
if __name__ == '__main__':
    print(sys.argv)
    vechile = Path(sys.argv[0])
    yaml_file = vechile.with_suffix('.yaml')
    cfg = AbcDict(yaml_file)
    taskid=cfg.task_id
    test_count = int(sys.argv[-1])
    msg_info(f'车辆稳定性测试第{test_count}轮')
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    [msg_info(i) for i in result.stdout.split('\n')]
    if result.stderr:
        [msg_error(i) for i in result.stderr.split('\n')]
    pyfat_file = Path(sys.argv[1])
    main(pyfat_file, cfg)


