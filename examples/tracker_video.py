import os
import time

import cv2
import numpy as np

from examples.otbdataset_config import OTBDatasetConfig
from lib.utils import get_img_list, get_ground_truthes, APCE, PSR, plot_success, plot_precision
from cftracker.mosse import MOSSE
from cftracker.csk import CSK
from cftracker.kcf import KCF
from cftracker.cn import CN
from cftracker.dsst import DSST
from cftracker.staple import Staple
from cftracker.dat import DAT
from cftracker.eco import ECO
from cftracker.bacf import BACF
from cftracker.csrdcf import CSRDCF
from cftracker.samf import SAMF
from cftracker.ldes import LDES
from cftracker.mkcfup import MKCFup
from cftracker.strcf import STRCF
from cftracker.mccth_staple import MCCTHStaple
from lib.eco.config import otb_deep_config,otb_hc_config
from cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config
class PyTracker:
    def __init__(self,img_dir,tracker_type,dataset_config):
        self.img_dir=img_dir
        self.tracker_type=tracker_type

        if self.tracker_type == 'MOSSE':
            self.tracker=MOSSE()
        elif self.tracker_type=='CSK':
            self.tracker=CSK()
        elif self.tracker_type=='CN':
            self.tracker=CN()
        elif self.tracker_type=='DSST':
            self.tracker=DSST(dsst_config.DSSTConfig())
        elif self.tracker_type=='Staple':
            self.tracker=Staple(config=staple_config.StapleConfig())
        elif self.tracker_type=='Staple-CA':
            self.tracker=Staple(config=staple_config.StapleCAConfig())
        elif self.tracker_type=='KCF_CN':
            self.tracker=KCF(features='cn',kernel='gaussian')
        elif self.tracker_type=='KCF_GRAY':
            self.tracker=KCF(features='gray',kernel='gaussian')
        elif self.tracker_type=='KCF_HOG':
            self.tracker=KCF(features='hog',kernel='gaussian')
        elif self.tracker_type=='DCF_GRAY':
            self.tracker=KCF(features='gray',kernel='linear')
        elif self.tracker_type=='DCF_HOG':
            self.tracker=KCF(features='hog',kernel='linear')
        elif self.tracker_type=='DAT':
            self.tracker=DAT()
        elif self.tracker_type=='ECO-HC':
            self.tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif self.tracker_type=='ECO':
            self.tracker=ECO(config=otb_deep_config.OTBDeepConfig())
        elif self.tracker_type=='BACF':
            self.tracker=BACF()
        elif self.tracker_type=='CSRDCF':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif self.tracker_type=='CSRDCF-LP':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFLPConfig())
        elif self.tracker_type=='SAMF':
            self.tracker=SAMF()
        elif self.tracker_type=='LDES':
            self.tracker=LDES(ldes_config.LDESDemoLinearConfig())
        elif self.tracker_type=='DSST-LP':
            self.tracker=DSST(dsst_config.DSSTLPConfig())
        elif self.tracker_type=='MKCFup':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupConfig())
        elif self.tracker_type=='MKCFup-LP':
            self.tracker=MKCFup(config=mkcf_up_config.MKCFupLPConfig())
        elif self.tracker_type=='STRCF':
            self.tracker=STRCF()
        elif self.tracker_type=='MCCTH-Staple':
            self.tracker=MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())
        else:
            raise NotImplementedError

    def tracking(self,verbose=True,video_path=None):
        poses = []

        cap_1 = cv2.VideoCapture(self.img_dir)
        ret,init_frame = cap_1.read()
        init_frame=cv2.resize(init_frame,(int(704*0.8),int(576*0.8)))
        #print(init_frame.shape)
        # init_gt = np.array(self.init_gt)
        # init_gt=int(533*0.8),int(43*0.8),int(31*0.8),int(57*0.8)
        init_gt=int(523*0.8),int(30*0.8),int(51*0.8),int(87*0.8)
        x1, y1, w, h =init_gt
        init_gt=tuple(init_gt)
        self.tracker.init(init_frame,init_gt)
        writer=None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))

        while True:
            ret, current_frame = cap_1.read()
            current_frame = cv2.resize(current_frame, (int(704 * 0.8), int(576 * 0.8)))
            # current_frame=cv2.imread(self.frame_list[idx])
            height,width=current_frame.shape[:2]
            start=time.time()
            bbox=self.tracker.update(current_frame,vis=verbose)
            print('track time',time.time()-start,current_frame.shape)
            x1,y1,w,h=bbox
            if verbose is True:
                if len(current_frame.shape)==2:
                    current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
                score = self.tracker.score
                apce = APCE(score)
                psr = PSR(score)
                F_max = np.max(score)
                size=self.tracker.crop_size
                score = cv2.resize(score, size)
                score -= score.min()
                score =score/ score.max()
                score = (score * 255).astype(np.uint8)
                # score = 255 - score
                score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                center = (int(x1+w/2),int(y1+h/2))
                x0,y0=center
                x0=np.clip(x0,0,width-1)
                y0=np.clip(y0,0,height-1)
                center=(x0,y0)
                xmin = int(center[0]) - size[0] // 2
                xmax = int(center[0]) + size[0] // 2 + size[0] % 2
                ymin = int(center[1]) - size[1] // 2
                ymax = int(center[1]) + size[1] // 2 + size[1] % 2
                left = abs(xmin) if xmin < 0 else 0
                xmin = 0 if xmin < 0 else xmin
                right = width - xmax
                xmax = width if right < 0 else xmax
                right = size[0] + right if right < 0 else size[0]
                top = abs(ymin) if ymin < 0 else 0
                ymin = 0 if ymin < 0 else ymin
                down = height - ymax
                ymax = height if down < 0 else ymax
                down = size[1] + down if down < 0 else size[1]
                score = score[top:down, left:right]
                crop_img = current_frame[ymin:ymax, xmin:xmax]
                score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
                current_frame[ymin:ymax, xmin:xmax] = score_map
                show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),1)
                """
                cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (0, 250), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (0, 0, 255), 5)
                cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (0, 300), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (255, 0, 0), 5)
                cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (0, 350), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (255, 0, 0), 5)
                """

                cv2.imshow('demo', show_frame)
                # if writer is not None:
                #     writer.write(show_frame)
                cv2.waitKey()

            poses.append(np.array([int(x1), int(y1), int(w), int(h)]))
        return np.array(poses)

if __name__ == '__main__':

    data_dir = r'D:\data\track\tianzong.mp4'

    dataset_config = OTBDatasetConfig()
        # gts = get_ground_truthes(data_path)

    tracker = PyTracker(data_dir, tracker_type='Staple', dataset_config=dataset_config)
    poses = tracker.tracking(verbose=True, video_path=os.path.join('../results/CF',  '_vis.avi'))
    # plot_success(gts, poses, o   s.path.join('../results/CF', data_name + '_success.jpg'))
    # plot_precision(gts, poses, os.path.join('../results/CF', data_name + '_precision.jpg'))