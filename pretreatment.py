import os
import scipy.misc
import cv2
import warnings
import time
import argparse
import multiprocessing as mp
from terminaltables import AsciiTable
import numpy as np
import logging

opt = None

def crop_silhouette(im, position, im_file):
    if opt.use_log:
        logging.info('Size of image %s is (%d ,%d)', im_file, im.shape[0], im.shape[1])
    if (im > 0).sum() < 100:
        if opt.use_log:
            logging.warning('Picture %s contains no more than 100 pixels, which has %d', im_file, (im > 0).sum())
        return None
    
    top = (im.sum(axis = 1) != 0).argmax()
    bottom = (im.sum(axis = 1) != 0).cumsum().argmax()

    im = im[top:bottom + 1, :]
    if opt.use_log:
        logging.info('Image %s vertical clip is based on top %d, bottom %d', im_file, top, bottom)

    w_h_ratio = im.shape[1] / im.shape[0]
    if opt.use_log:
        logging.info('Image %s w/h ratio is %f', im_file, w_h_ratio)
    new_width = int(opt.height * w_h_ratio)
    if opt.use_log:
        logging.info('Image %s new width based on w/h ratio is %d', im_file, new_width)
    im = cv2.resize(im, (new_width, opt.height), interpolation=cv2.INTER_CUBIC)
    
    _ = np.zeros((im.shape[0], int(opt.width / 2)))
    im = np.concatenate([_, im, _], axis=1)
    
    pixel_sum = im.sum()
    horizontal_pixel_sum = im.sum(axis=0).cumsum()

    horizontal_center = None
    for i in range(len(horizontal_pixel_sum)):
        if horizontal_pixel_sum[i] > pixel_sum / 2:
            horizontal_center = i
            break
    if opt.use_log:
        logging.info(
            "Image %s ' pixel sum is %d, horizontal_center is %d", 
            im_file, 
            pixel_sum, 
            horizontal_center)

    if horizontal_center is None:
        if opt.use_log:
            logging.critical('Impossible situation, input image %s has no center', im_file)
        raise ValueError('Impossible situation, input im has no center')
    left_border = horizontal_center - int(opt.width / 2)
    right_border = horizontal_center + int(opt.width / 2)

    if opt.use_log:
        logging.info("Image %s ' left_border is %d, right_border is %d", im_file, left_border, right_border)
    
    im = im[:, left_border:right_border]
    return im.astype('uint8')


def process(position):
    folder = os.path.join(opt.input_path, position)
    target_dir = os.path.join(opt.output_path, position)

    seqs = os.listdir(folder)
    seqs.sort()

    if opt.use_log:
        logging.info('Counted %d pictures in folder %s', len(seqs), folder)

    frame_count = 0
    for im_file in seqs:
        im_dir = os.path.join(folder, im_file)
        im = cv2.imread(im_dir, cv2.IMREAD_GRAYSCALE)
        im = crop_silhouette(im, position, im_file)

        if im is not None:
            target_im_dir = os.path.join(opt.output_path, position, im_file)
            scipy.misc.imsave(target_im_dir, im)
            frame_count += 1
        else:
            if opt.use_log:
                logging.warning('Image %s is None after crop', im_file)


def main():
    start = time.time()
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='./pretreatment.log')
    parser.add_argument('--use_log', action='store_true', default=False)
    parser.add_argument('--worker_num', default=4, type=int)
    parser.add_argument('--height', default=64, type=int)
    parser.add_argument('--width', default=64, type=int)
    opt = parser.parse_args()
    opt.worker_num = min(opt.worker_num, mp.cpu_count())

    if opt.use_log:
        logging.basicConfig(format='%(asctime)s::%(levelname)s::%(message)s', level=logging.DEBUG, filename=opt.log_dir)
        logging.info('logging system configured!')

    settings = []
    settings.append(['param', 'value'])
    settings.append(['input_path', opt.input_path])
    settings.append(['output_path', opt.output_path])
    settings.append(['log_dir', opt.log_dir])
    settings.append(['use_log', opt.use_log])
    settings.append(['worker_num', opt.worker_num])
    settings.append(['height', opt.height])
    settings.append(['width', opt.width])

    table = AsciiTable(settings)
    print(table.table)

    if opt.use_log:
        for i in range(1, len(settings)):
            logging.info('Param %s was set to %s', settings[i][0], settings[i][1])

    subjects = os.listdir(opt.input_path)
    subjects.sort()

    pool = mp.Pool(opt.worker_num)
    results = list()

    for _subject in subjects:
        _subject_dir = os.path.join(opt.input_path, _subject)
        status = os.listdir(_subject_dir)
        status.sort()

        for _status in status:
            _status_dir = os.path.join(_subject_dir, _status)
            views = os.listdir(_status_dir)
            views.sort()
            for _view in views:
                _view_dir = os.path.join(_status_dir, _view)
                processed_view_dir = os.path.join(opt.output_path, _subject, _status, _view)
                os.makedirs(processed_view_dir)
                if opt.use_log:
                    logging.info('%s folder is created.', processed_view_dir)
                results.append(
                    pool.apply_async(process, args=(os.path.join(_subject, _status, _view)))
                )
                # process(os.path.join(_subject, _status, _view))
                # import sys
                # sys.exit(0)

    pool.close()
    finished = False
    while not finished:
        finished = True
        for res in results:
            try:
                res.get(timeout=0.1)
            except mp.TimeoutError:
                finished = False
                continue
            except Exception as e:
                raise e 
    
    pool.join()
    end = time.time()
    if opt.use_log:
        logging.info('Pictures preprocessing finished, took %d mins, %d secs', int(end - start) // 60, int(end - start) % 60)

if __name__ == "__main__":
    main()
