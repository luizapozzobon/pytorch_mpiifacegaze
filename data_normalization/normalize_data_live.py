# -*- coding: utf-8 -*-
"""
Code rearranged from the original (license and paper below) to support full face normalization and live testing.

######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Any publications arising from the use of this software, including but
not limited to academic journal and conference publications, technical
reports and manuals, must cite at least one of the following works:

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
######################################################################################################################################
"""

import os
import cv2
import numpy as np
import csv
import scipy.io as sio
import dlib

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/luiza/Qiron/demographic_survey/scripts/shape_predictor_68_face_landmarks.dat')

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    # SolvePnP = Finds an object pose from 3D-2D point correspondences.
    # The function estimates the object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients.
    # cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) → retval, rvec, tvec
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    # rvec – Output rotation vector (see Rodrigues() ) that, together with tvec , brings points from the model coordinate system to the camera coordinate system.
    # tvec – Output translation vector.

    return rvec, tvec

def normalizeData(img, face, hr, ht, cam, gc=None):
    # Rodrigues -> Converts a rotation matrix to a rotation vector or vice versa.

    ## normalized camera parameters
    focal_norm = 1600 # focal length of normalized camera
    distance_norm = 1000 # normalized distance between eye and camera
    roiSize = (448, 448) # size of cropped eye image

    img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## compute estimated 3D positions of the landmarks
    #ht = ht.reshape((3,1)) #
    ht = np.repeat(ht, 6, axis=1)
    if gc: gc = gc.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # converts rotation vector to rotation matrix
    Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks
    face_center = np.sum(Fc, axis=1, dtype=np.float32)/6.0

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center) # actual distance face center and original camera

    z_scale = distance_norm/distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0]/2],
        [0, focal_norm, roiSize[1]/2],
        [0, 0, 1.0],
    ])
    S = np.array([ # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:,0]
    forward = (face_center/distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix

    img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
    img_warped = cv2.equalizeHist(img_warped)

    ## ---------- normalize rotation ----------
    hR_norm = np.dot(R, hR) # rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors

    ## ---------- normalize gaze vector ----------
    if gc:
        gc_normalized = gc - et # gaze vector
    # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
    # For original data normalization, here should be:
    # "M = np.dot(S,R)
    # gc_normalized = np.dot(R, gc_normalized)"
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)

    if gc:
        data = [img_warped, hr_norm, gc_normalized]
    else:
        data = [img_warped, hr_norm]

    return data

def load_camera_specs(camera_calib_path):
    """Load all camera calibration files an matrixes"""
    ## load calibration data, these paramters can be obtained by camera calibration functions in OpenCV
    cameraCalib = sio.loadmat('./data/calibration/cameraCalib.mat')
    camera_matrix = cameraCalib['cameraMatrix']
    camera_distortion = cameraCalib['distCoeffs']
    return camera_matrix, camera_distortion

def get_landmarks(frame):
    """Get 6 facial landmarks (corner of eyes and mouth)"""
    #output = numpy float32 to delete that conversion line in main
    landmarks_ids = [37, 40, 43, 46, 49, 55] # reye, leye, mouth
    face_locations = face_detector(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = None
    if face_locations:
        for i, d in enumerate(face_locations):
            shape = predictor(gray, d)
            landmarks = [[shape.part(i).x, shape.part(i).y] for i in landmarks_ids]
        #for i in landmarks_ids:
        #    curr_shape = shape.part(i)
        #    x = int(curr_shape.x)
        #    y = int(curr_shape.y)
        #    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        #cv2.imshow('Landmarks', frame)
        #cv2.waitKey(0)
        return np.array(landmarks), True
    else:
        return False, False

def main(path='./data/faceModelGeneric.mat'):

    camera_matrix, camera_distortion = load_camera_specs(path)
    video_capture = cv2.VideoCapture(0)

    process_this_frame = False
    while True:
        ret, original_frame = video_capture.read()
        if process_this_frame:
            #small_frame = cv2.resize(original_frame, (0, 0), fx=0.25, fy=0.25)

            # undistort image based on camera specs
            frame = cv2.undistort(original_frame, camera_matrix, camera_distortion)

            success = None
            # get landmarks for both eyes and mouth using dlib
            landmarks, success = get_landmarks(frame)

            if success:
                # estimate head pose
                # load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
                face = sio.loadmat('./data/faceModelGeneric.mat')['model']
                num_pts = face.shape[1]
                facePts = face.T.reshape(num_pts, 1, 3)
                # reshape landmarks to paper format
                landmarks = landmarks.astype(np.float32)
                landmarks = landmarks.reshape(num_pts, 1, 2)
                hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)
                # hr = rotation vector
                # ht = translation vector

                # get gc from data, if available
                gc = None

                data = normalizeData(frame, face, hr, ht, camera_matrix, gc=gc)
                # data[0] = img
                # data[1] = rotation matrix
                # data[2] = gc, if sent
                img_normalized = data[0]
                print(data[0].shape)
                cv2.imshow('Normalized', img_normalized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        process_this_frame = not process_this_frame

if __name__ == '__main__':
    main()
    video_capture.release()
    cv2.destroyAllWindows()
