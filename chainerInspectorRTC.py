#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python -*-

"""
 @file chainerInspectorRTC.py
 @brief inspect image by chainer
 @date $Date$


"""
import sys
import time
sys.path.append(".")

# Import RTM module
import RTC
import OpenRTM_aist


# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
# </rtc-template>

import argparse
import json
import random
import sys

import numpy as np
from PIL import Image

import cv2

import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import nin

# This module's spesification
# <rtc-template block="module_spec">
chainerinspectorrtc_spec = ["implementation_id", "chainerInspectorRTC", 
		 "type_name",         "chainerInspectorRTC", 
		 "description",       "inspect image by chainer", 
		 "version",           "1.0.0", 
		 "vendor",            "takahasi", 
		 "category",          "Category", 
		 "activity_type",     "STATIC", 
		 "max_instance",      "1", 
		 "language",          "Python", 
		 "lang_type",         "SCRIPT",
		 "conf.default.model", "model.model",
		 "conf.default.labels", "labels.txt",
		 "conf.default.mean", "mean.npy",
		 "conf.default.display_num", "10",

		 "conf.__widget__.model", "text",
		 "conf.__widget__.labels", "text",
		 "conf.__widget__.mean", "text",
		 "conf.__widget__.display_num", "text",

         "conf.__type__.model", "string",
         "conf.__type__.labels", "string",
         "conf.__type__.mean", "string",
         "conf.__type__.display_num", "int",

		 ""]
# </rtc-template>

##
# @class chainerInspectorRTC
# @brief inspect image by chainer
# 
# 
class chainerInspectorRTC(OpenRTM_aist.DataFlowComponentBase):
	
	##
	# @brief constructor
	# @param manager Maneger Object
	# 
	def __init__(self, manager):
		OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)

		in_img_arg = [None] * ((len(RTC._d_CameraImage) - 4) / 2)
		self._d_in_img = RTC.CameraImage(*in_img_arg)
		"""
		"""
		self._in_imageIn = OpenRTM_aist.InPort("in_image", self._d_in_img)
		out_img_arg = [None] * ((len(RTC._d_CameraImage) - 4) / 2)
		self._d_out_img = RTC.CameraImage(*out_img_arg)
		"""
		"""
		self._out_imageOut = OpenRTM_aist.OutPort("out_image", self._d_out_img)


		


		# initialize of configuration-data.
		# <rtc-template block="init_conf_param">
		"""
		
		 - Name:  model
		 - DefaultValue: model.model
		"""
		self._model = ['model.model']
		"""
		
		 - Name:  labels
		 - DefaultValue: labels.txt
		"""
		self._labels = ['labels.txt']
		"""
		
		 - Name:  mean
		 - DefaultValue: mean.npy
		"""
		self._mean = ['mean.npy']
		"""
		
		 - Name:  display_num
		 - DefaultValue: 10
		"""
		self._display_num = [10]
		
		# </rtc-template>

		self._cropwidth = 0
		self._nmodel = None
		self._mean_image= None

		 
	##
	#
	# The initialize action (on CREATED->ALIVE transition)
	# formaer rtc_init_entry() 
	# 
	# @return RTC::ReturnCode_t
	# 
	#
	def onInitialize(self):
		# Bind variables and configuration variable
		self.bindParameter("model", self._model, "model.model")
		self.bindParameter("labels", self._labels, "labels.txt")
		self.bindParameter("mean", self._mean, "mean.npy")
		self.bindParameter("display_num", self._display_num, "10")
		
		# Set InPort buffers
		self.addInPort("in_image",self._in_imageIn)
		
		# Set OutPort buffers
		self.addOutPort("out_image",self._out_imageOut)
		
		# Set service provider to Ports
		
		# Set service consumers to Ports
		
		# Set CORBA Service Ports
		
		return RTC.RTC_OK
	
	#	##
	#	# 
	#	# The finalize action (on ALIVE->END transition)
	#	# formaer rtc_exiting_entry()
	#	# 
	#	# @return RTC::ReturnCode_t
	#
	#	# 
	#def onFinalize(self):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The startup action when ExecutionContext startup
	#	# former rtc_starting_entry()
	#	# 
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onStartup(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The shutdown action when ExecutionContext stop
	#	# former rtc_stopping_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onShutdown(self, ec_id):
	#
	#	return RTC.RTC_OK
	
		##
		#
		# The activated action (Active state entry action)
		# former rtc_active_entry()
		#
		# @param ec_id target ExecutionContext Id
		# 
		# @return RTC::ReturnCode_t
		#
		#
	def onActivated(self, ec_id):
		parser = argparse.ArgumentParser(description='Image inspection using chainer')
		parser.add_argument('--model','-m',default='mlp.model', help='Path to model file')
		parser.add_argument('--mean', default='mean.npy', help='Path to the mean file (computed by compute_mean.py)')
		args = parser.parse_args()

		self._mean_image = np.load(args.mean)
		self._nmodel = nin.NIN()
		serializers.load_hdf5("mlp.model", self._nmodel)
		self._cropwidth = 256 - self._nmodel.insize
		self._nmodel.to_cpu()
		
	
		return RTC.RTC_OK
	
		##
		#
		# The deactivated action (Active state exit action)
		# former rtc_active_exit()
		#
		# @param ec_id target ExecutionContext Id
		#
		# @return RTC::ReturnCode_t
		#
		#
	def onDeactivated(self, ec_id):
	
		return RTC.RTC_OK
	
		##
		#
		# The execution action that is invoked periodically
		# former rtc_active_do()
		#
		# @param ec_id target ExecutionContext Id
		#
		# @return RTC::ReturnCode_t
		#
		#
	def onExecute(self, ec_id):
		
		if not self._in_imageIn.isNew():
			return RTC.RTC_OK

		data = self._in_imageIn.read()

		height = data.height
		width = data.width
		depth = data.bpp
		pil_img = Image.fromstring('RGB', (width, height), data.pixels)
		img = np.asarray(pil_img)
		output_side_length = 256
		new_height = output_side_length
		new_width = output_side_length

		if height > width:
			new_height = output_side_length * height / width
		else:
			new_width = output_side_length * width / height

		resized_img = cv2.resize(img, (new_width, new_height))
		height_offset = (new_height - output_side_length) / 2
		width_offset = (new_width - output_side_length) / 2
		cropped_img = resized_img[height_offset:height_offset + output_side_length,
		width_offset:width_offset + output_side_length]

		image = np.asarray(cropped_img).transpose(2, 0, 1)
		top = random.randint(0, self._cropwidth - 1)
		left = random.randint(0, self._cropwidth - 1)

		bottom = self._nmodel.insize + top
		right = self._nmodel.insize + left
		image = image[:, top:bottom, left:right].astype(np.float32)
		image -= self._mean_image[:, top:bottom, left:right]
		image /= 255

		x = np.ndarray((1, 3, self._nmodel.insize, self._nmodel.insize), dtype=np.float32)
		x[0] = image
		x = chainer.Variable(np.asarray(x), volatile='on')

		h = F.max_pooling_2d(F.relu(self._nmodel.mlpconv1(x)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self._nmodel.mlpconv2(h)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self._nmodel.mlpconv3(h)), 3, stride=2)
		h = self._nmodel.mlpconv4(F.dropout(h, train=self._nmodel.train))
		h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
		score = F.softmax(h)

		categories = np.loadtxt("labels.txt", str, delimiter="\t")

		top_k = 10
		prediction = zip(score.data[0].tolist(), categories)
		prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
		
		for rank, (score, name) in enumerate(prediction[:top_k], start=1):
			print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
            #txt = str(rank) + ':' + str(name) + ' ' + str(score * 100)
			#cv2.putText(img, txt, (0, rank * 12), cv2.FONT_HERSHEY_PLAIN, 10, (255,255,0))

		cv2.startWindowThread()
		cv2.namedWindow('preview')
		#preview_img = Image.fromarray(resized_img.astype('uint8'), 'RGB')
		preview_img = img
		cv2.imshow('preview', preview_img)

		#self._d_out_img.height = data.height
		#self._d_out_img.width = data.width
		#self._d_out_img.bpp = data.bpp
		#self._d_out_img.pixels = data.pixels
		#self._d_out_img.pixels.length(data.pixels.length)
		#self._out_imageOut.write()

		return RTC.RTC_OK
	
	#	##
	#	#
	#	# The aborting action when main logic error occurred.
	#	# former rtc_aborting_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onAborting(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The error action in ERROR state
	#	# former rtc_error_do()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onError(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The reset action that is invoked resetting
	#	# This is same but different the former rtc_init_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onReset(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The state update action that is invoked after onExecute() action
	#	# no corresponding operation exists in OpenRTm-aist-0.2.0
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#

	#	#
	#def onStateUpdate(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The action that is invoked when execution context's rate is changed
	#	# no corresponding operation exists in OpenRTm-aist-0.2.0
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onRateChanged(self, ec_id):
	#
	#	return RTC.RTC_OK
	



def chainerInspectorRTCInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=chainerinspectorrtc_spec)
    manager.registerFactory(profile,
                            chainerInspectorRTC,
                            OpenRTM_aist.Delete)

def MyModuleInit(manager):
    chainerInspectorRTCInit(manager)

    # Create a component
    comp = manager.createComponent("chainerInspectorRTC")

def main():
	mgr = OpenRTM_aist.Manager.init(sys.argv)
	mgr.setModuleInitProc(MyModuleInit)
	mgr.activateManager()
	mgr.runManager()

if __name__ == "__main__":
	main()

