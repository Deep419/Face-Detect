from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys
from os import listdir
from os.path import isfile, join, exists
import time
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import math

NETS = {'vgg16': ('VGG16',
          '/apps/pkg/frcn-1.0/face-py-faster-rcnn/output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_80000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
            help='Use CPU mode (overrides --gpu)',
            action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            choices=NETS.keys(), default='vgg16')

  args = parser.parse_args()

  return args


#Function used to get the rotation matrix
def yaw2rotmat(yaw,pitch,roll):
    x = roll #0.0
    y = pitch #0.0
    z = yaw
    ch = np.cos(z)
    sh = np.sin(z)
    ca = np.cos(y)
    sa = np.sin(y)
    cb = np.cos(x)
    sb = np.sin(x)
    rot = np.zeros((3,3), 'float32')
    rot[0][0] = ch * ca
    rot[0][1] = sh*sb - ch*sa*cb
    rot[0][2] = ch*sa*sb + sh*cb
    rot[1][0] = sa
    rot[1][1] = ca * cb
    rot[1][2] = -ca * sb
    rot[2][0] = -sh * ca
    rot[2][1] = sh*sa*cb + ch*sb
    rot[2][2] = -sh*sa*sb + ch*cb
    return rot

def drawGrid(img, det):
	cam_w = det[2] - det[0]  #image.shape[1]
	cam_h = det[3] - det[1]  #image.shape[0]

	cam_w_half = cam_w/2
	cam_h_half = cam_h/2;

	c_x = (cam_w / 2) + det[0]
	c_y = (cam_h / 2) + det[1]
	
	
	cv2.line(im, (int(c_x) , int(c_y-cam_h_half) ), (int(c_x) , int(c_y+cam_h_half) ), (255,255,255), 3) #RED
	cv2.line(im, (int(c_x-cam_w_half) , int(c_y) ), (int(c_x+cam_w_half) , int(c_y) ), (255,255,255), 3)
	
	
	cv2.circle(im, (int(c_x), int(c_y)), int(min(cam_w_half,cam_h_half)), (255,255,255), 3)
	
	
def drawYaw(img, yaw, det, maxVal ):
	cam_w = det[2] - det[0]  #image.shape[1]
	cam_h = det[3] - det[1]  #image.shape[0]

	cam_w_half = cam_w/2
	cam_h_half = cam_h/2;

	c_x = (cam_w / 2) + det[0]
	c_y = (cam_h / 2) + det[1]
	
	posVec = (-1, 0)

	yawNorm = yaw / maxVal
	if yawNorm > 1:
		yawNorm = 1.0
	if yawNorm < -1:
		yaNorm = -1.0

	xpt = c_x + (yawNorm * posVec[0] * cam_w_half)

	cv2.circle(im, (int(xpt), int(c_y)), 10, (0,255,255), -1)
        
def drawPitch(img, pitch, det, maxVal ):
	
	cam_w = det[2] - det[0]  #image.shape[1]
	cam_h = det[3] - det[1]  #image.shape[0]

	cam_w_half = cam_w/2
	cam_h_half = cam_h/2;

	c_x = (cam_w / 2) + det[0]
	c_y = (cam_h / 2) + det[1]
	
	posVec = (0, -1)

	pitchNorm = pitch / maxVal

	if pitchNorm > 1:
		pitchNorm = 1.0
	if pitchNorm < -1:
		pitchNorm = -1.0

	ypt = c_y + (pitchNorm * posVec[1] * cam_h_half)

	cv2.circle(im, (int(c_x), int(ypt)), 10, (226,43,138), -1)        

def drawRoll(img, roll, det, maxVal):
	cam_w = det[2] - det[0]  #image.shape[1]
	cam_h = det[3] - det[1]  #image.shape[0]

	cam_w_half = cam_w/2
	cam_h_half = cam_h/2;

	circRad = int(min(cam_w_half,cam_h_half))

	c_x = (cam_w / 2) + det[0]
	c_y = (cam_h / 2) + det[1]
	
	rollNorm = roll / maxVal
	
	cir_x = c_x + math.cos(math.radians(rollNorm*90+90)) * circRad
	cir_y = c_y + math.sin(math.radians(rollNorm*90+90)) * circRad
	cv2.circle(im, (int(cir_x), int(cir_y)  ), 10, (0,0,255), -1)



if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  # cfg.TEST.BBOX_REG = False

  args = parse_args()

  prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
              'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
  caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                NETS[args.demo_net][1])

  prototxt = '/apps/pkg/frcn-1.0/face-py-faster-rcnn/models/face/VGG16/faster_rcnn_end2end/test.prototxt'
  caffemodel = NETS[args.demo_net][1]

  print caffemodel

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

  if args.cpu_mode:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)

  print '\n\nLoaded network {:s}'.format(caffemodel)

  #config = tf.ConfigProto(device_count = {'CPU': 0})
  sess = tf.Session() #Launch the graph in a session.
  my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
  # Load the weights from the configuration folders
  

  # my_head_pose_estimator.load_roll_variables(os.path.realpath("/home/steve/code/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
  # my_head_pose_estimator.load_pitch_variables(os.path.realpath("/home/steve/code/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
  # my_head_pose_estimator.load_yaw_variables(os.path.realpath("/home/steve/code/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))



  # data_dir = '~/research/face/data/'
  # out_dir = '~/research/face/output/'

  # inputFile = '/home/steve/code/faceDetYOLO/practice-data/360vids/20180306_094911.mp4'
  # outputFile = '/home/steve/code/face-py-faster-rcnn/output/360vids/20180306_094911_output.avi'
  my_head_pose_estimator.load_roll_variables(os.path.realpath("/users/dghaghar/research/face/src/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
  my_head_pose_estimator.load_pitch_variables(os.path.realpath("/users/dghaghar/research/face/src/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
  my_head_pose_estimator.load_yaw_variables(os.path.realpath("/users/dghaghar/research/face/src/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))



  data_dir = '/users/dghaghar/research/face/data/'
  out_dir = '/users/dghaghar/research/face/output/'

  inputFile = '/users/dghaghar/research/face/data/test.mp4'
  outputFile = '/users/dghaghar/research/face/output/test.avi'

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  CONF_THRESH = 0.97 #0.65
  NMS_THRESH = 0.15  #0.15

  #imdb = get_imdb_fddb(data_dir)

  print '\n\n11AFTER SESSION\n\n'
  # Warmup on a dummy image
  #im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  #for i in xrange(2):
  #  _, _= im_detect(net, im)
  print '\n\n22AFTER SESSION\n\n'
  #nfold = len(imdb)

  #image_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]


  #startTime = time.time()

  #Defining the video capture object
  #video_capture = cv2.VideoCapture(0)
  print '\n\nBefore Video Capture ====================== \n'

  video_capture = cv2.VideoCapture(inputFile)
  if(video_capture.isOpened() == True): print("ex_pnp_head_pose_estimation: the video source has been opened correctly...")
      # Define the codec and create VideoWriter object
      #fourcc = cv2.VideoWriter_fourcc(*'XVID'
  fourcc = cv2.cv.CV_FOURCC(*'XVID')
  out = cv2.VideoWriter(outputFile, fourcc, 20.0, (3240,2160))

  '''
  if(video_capture.isOpened() == False):
       print("Error: the resource is busy or unvailable")
  else:
       print("The video source has been opened correctly...")
  '''
  #Create the main window and move it
  cv2.namedWindow('Video')
  cv2.moveWindow('Video', 20, 20)

  #Obtaining the CAM dimension
  cam_dim_w = int(video_capture.get(3))
  cam_dim_h = int(video_capture.get(4))


  while(True): #for im_name in image_files:
      # timer = Timer()
      # timer.tic()

      
      
      #full_image_file = os.path.join(data_dir, im_name)
      #print('Processing', full_image_file)
      
      # im_path = im_name + '.jpg'
      #im = cv2.imread(full_image_file)
      ret, im = video_capture.read()

      # # Detect all object classes and regress object bounds
      # timer = Timer()
      # timer.tic()
      scores, boxes = im_detect(net, im)
      # timer.toc()
      # print ('Detection took {:.3f}s for '
      #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

      cls_ind = 1
      cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
      cls_scores = scores[:, cls_ind]
      dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
      keep = nms(dets, NMS_THRESH)
      dets = dets[keep, :]

      keep = np.where(dets[:, 4] > CONF_THRESH)
      dets = dets[keep]

      for j in xrange(dets.shape[0]):
        #fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))
        '''
        dets[j,0] = dets[j,0] - 50;
        dets[j,1] = dets[j,1] - 50;
        dets[j,2] = dets[j,2] + 50;
        dets[j,3] = dets[j,3] + 50;
        '''
        
        cv2.rectangle(im, (dets[j,0],dets[j,1]), (dets[j,2], dets[j,3]), (0,255,0), 3)
        
        newDet = np.copy(dets[j, :]);
        
        if newDet[2] - newDet[0] < newDet[3] - newDet[1]:
			diff = int(newDet[3]) - int(newDet[1]) - (int(newDet[2]) - int(newDet[0]) )
			half = int(diff/2)
			mo = diff % 2;
			newDet[0] = int(newDet[0]) - half
			newDet[2] = int(newDet[2]) + half + mo
        else:
			diff = int(newDet[2]) - int(newDet[0]) - (int(newDet[3]) - int(newDet[1]))
			half = int(diff/2)
			mo = diff % 2;
			newDet[1] = int(newDet[1]) - half 
			newDet[3] = int(newDet[3]) + half + mo
        
        
        
        #Do gaze estimation
        cam_w = dets[j,2] - dets[j,0]  #image.shape[1]
        cam_h = dets[j,3] - dets[j,1]  #image.shape[0]
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60/2 * np.pi / 180)
        f_y = f_x
        camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y], 
                                [0.0, 0.0, 1.0] ])
        #print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")
        
        #Distortion coefficients
        camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        #Defining the axes
        axis = np.float32([[0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.0], 
                       [0.0, 0.0, 0.5]])

        imCrop = im[ int(newDet[1]):int(newDet[3]), int(newDet[0]):int(newDet[2]) ]
        
        #print imCrop.shape 
        if imCrop.shape[1] < 64 or imCrop.shape[0] < 64:
            imCrop = im[ int(dets[j,1]):int(dets[j,3]), int(dets[j,0]):int(dets[j,2]) ]
            print(dets[j,:])
            print(imCrop.shape)
            #scaleFactor = max(float(64)/float(imCrop.shape[1]), float(64)/float(imCrop.shape[0]))
            maxdim = max(imCrop.shape[0],imCrop.shape[1])
            if maxdim < 64:
                 maxdim = 64
            imCrop = cv2.resize(imCrop, (maxdim, maxdim) ) #fx=scaleFactor, fy=scaleFactor)
        
        if imCrop.shape[1] != imCrop.shape[0]: 
           maxdim = max(imCrop.shape[0],imCrop.shape[1])
           if maxdim < 64:
                 maxdim = 64
           imCrop = cv2.resize(imCrop, (maxdim, maxdim) )

        roll_degree = my_head_pose_estimator.return_roll(imCrop, radians=False)  # Evaluate the roll angle using a CNN
        pitch_degree = my_head_pose_estimator.return_pitch(imCrop, radians=False)  # Evaluate the pitch angle using a CNN
        yaw_degree = my_head_pose_estimator.return_yaw(imCrop, radians=False)  # Evaluate the yaw angle using a CNN
        print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0,0,0]) + "," + str(pitch_degree[0,0,0]) + "," + str(yaw_degree[0,0,0])  + "]")
        
        drawGrid(im, dets[j,:])
        drawYaw(im, yaw_degree, dets[j,:], 90 )
        drawPitch(im, pitch_degree, dets[j,:], 45 )
        drawRoll(im, -roll_degree, dets[j,:], 15)
        '''
        roll = my_head_pose_estimator.return_roll(imCrop, radians=True)  # Evaluate the roll angle using a CNN
        pitch = my_head_pose_estimator.return_pitch(imCrop, radians=True)  # Evaluate the pitch angle using a CNN
        yaw = my_head_pose_estimator.return_yaw(imCrop, radians=True)  # Evaluate the yaw angle using a CNN
        print("Estimated [roll, pitch, yaw] (radians) ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")
        #Getting rotation and translation vector
        rot_matrix = yaw2rotmat(-yaw[0,0,0], pitch[0,0,0], roll[0,0,0] ) #Deepgaze use different convention for the Yaw, we have to use the minus sign

        #Attention: OpenCV uses a right-handed coordinates system:
        #Looking along optical axis of the camera, X goes right, Y goes downward and Z goes forward.
        rvec, jacobian = cv2.Rodrigues(rot_matrix)
        tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector
        print rvec

        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
        p_start = (int(c_x + dets[j,0])  , int(c_y + dets[j,1]) )
        p_stop = (int(imgpts[2][0][0] + dets[j,0]) , int(imgpts[2][0][1] + dets[j,1]))
        #print("point start: " + str(p_start))
        #print("point stop: " + str(p_stop))
        #print("")

        cv2.line(im, p_start, p_stop, (0,0,255), 3) #RED
        cv2.circle(im, p_start, 1, (0,255,0), 3)
        '''
        
        
        

      #cv2.imwrite(("%s/%s.jpg" % (out_dir, im_name) ),  im);

      # vis_detections(im, 'face', dets, CONF_THRESH)

      #dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
      #dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

      # timer.toc()
      # print ('Detection took {:.3f}s for '
      #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

      #fid.write(im_name + '\n')
      #fid.write(str(dets.shape[0]) + '\n')
      #for j in xrange(dets.shape[0]):
      #  fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))


      #if ((idx + 1) % 10) == 0:
      #  sys.stdout.write('%.3f ' % ((idx + 1) / len(image_names) * 100))
      # sys.stdout.flush()
      cv2.imshow('Video', im)
      out.write(im)
      if cv2.waitKey(1) & 0xFF == ord('q'): break
      
      print ''


  video_capture.release()
  print("Bye...")
  #endTime = time.time()
  #print('This FPS is %f  \n' % ( float(len(image_files) )/ float(endTime-startTime) ) )

    

  # os.system('cp ./fddb_res/*.txt ~/Code/FDDB/results')
