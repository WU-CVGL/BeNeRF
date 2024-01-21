import cv2 as cv
import numpy as np
from typing import Union


class UndistortFisheyeCamera:

    class KannalaBrandt:
        def __init__(self, image_calibdata, event_calibdata) -> None:
            self.img_calib = image_calibdata
            self.evt_calib = event_calibdata

            # calibration parameters of image camera
            self.img_K = np.zeros((3, 3))
            self.img_K[0, 0] = self.img_calib["fx"]
            self.img_K[0, 2] = self.img_calib["cx"]
            self.img_K[1, 1] = self.img_calib["fy"]
            self.img_K[1, 2] = self.img_calib["cy"]
            self.img_K[2, 2] = 1
            self.img_D = np.array(
                [
                    self.img_calib["k1"],
                    self.img_calib["k2"],
                    self.img_calib["k3"],
                    self.img_calib["k4"],
                ]
            )

            # calibration parameters of event camera
            self.evt_K = np.zeros((3, 3))
            self.evt_K[0, 0] = self.evt_calib["fx"]
            self.evt_K[0, 2] = self.evt_calib["cx"]
            self.evt_K[1, 1] = self.evt_calib["fy"]
            self.evt_K[1, 2] = self.evt_calib["cy"]
            self.evt_K[2, 2] = 1
            self.evt_D = np.array(
                [
                    self.evt_calib["k1"],
                    self.evt_calib["k2"],
                    self.evt_calib["k3"],
                    self.evt_calib["k4"],
                ]
            )

        def GetNewIntrinsicMatrix(
                self, raw_img_res, raw_evt_res, new_img_res, new_evt_res
            ) -> Union[np.ndarray, np.ndarray]:
            img_K_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K = self.img_K,
                D = self.img_D,
                image_size = (raw_img_res[1], raw_img_res[0]),
                R = np.identity(3),
                new_size = (new_img_res[1], new_img_res[0])
            )

            evt_K_new = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K = self.evt_K,
                D = self.evt_D,
                image_size = (raw_evt_res[1], raw_evt_res[0]),
                R = np.identity(3),
                new_size = (new_evt_res[1], new_evt_res[0])
            )
            return img_K_new, evt_K_new
        
        def UndistortImage(self, img_dist, img_new_K, new_img_res):
            img_undist = cv.fisheye.undistortImage(
                distorted = img_dist, 
                K = self.img_K, 
                D = self.img_D, 
                Knew = img_new_K, 
                new_size = (new_img_res[1], new_img_res[0])
            )
            return img_undist
        
        def UndistortAccumulatedEvents(
                self, evt_acc_dist, evt_new_K, new_evt_res
            ) -> np.ndarray:
            # map [-1,1] to [-255,0] and [0,255]
            evt_acc_dist = np.asarray(evt_acc_dist * 255, dtype = np.int16)
            array_neg_to_zero = np.where(evt_acc_dist <= 0, evt_acc_dist, 0)
            array_zero_to_pos = np.where(evt_acc_dist >= 0, evt_acc_dist, 0)

            # undistort as image
            evt_neg_acc_undist = cv.fisheye.undistortImage(
                distorted = (array_neg_to_zero * -1), 
                K = self.evt_K, 
                D = self.evt_D, 
                Knew = evt_new_K, 
                new_size = (new_evt_res[1], new_evt_res[0])
            ) * -1
            
            evt_pos_acc_undist = cv.fisheye.undistortImage(
                distorted = array_zero_to_pos, 
                K = self.evt_K, 
                D = self.evt_D, 
                Knew = evt_new_K, 
                new_size = (new_evt_res[1], new_evt_res[0])
            )

            # convert intensity to polarity
            evt_acc_undist = np.asarray((evt_pos_acc_undist + evt_neg_acc_undist), dtype = np.float16)
            mapping_rule = {
                (-255, -128): -1,
                (-127, 127): 0,
                (128, 255): 1
            }
            evt_acc_undist = np.piecewise(
                evt_acc_undist, 
                [np.logical_and(evt_acc_undist >= start, evt_acc_undist <= end) for start, end in mapping_rule.keys()],
                [mapping_rule[start, end] for start, end in mapping_rule.keys()]
            )
            return evt_acc_undist.astype(np.int16)
            
        def UndistortStreamEvents(self):
            pass        

    class Unified:
        def __init__(self) -> None:
            pass


    class ExtendedUnified:
        def __init__(self) -> None:
            pass


    class FOV:
        def __init__(self) -> None:
            pass


    class DoubleSphere:
        def __init__(self) -> None:
            pass
