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

import numpy as np
import chainer.functions as F
from chainer import initializers, optimizers, serializers, cuda, Variable
import chainer.links as L
from googlenet import GoogLeNet
import cv2


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
         "conf.default.model", "googlenet.model",
         "conf.default.labels", "labels.txt",
         "conf.default.mean", "mean.npy",
         "conf.default.display_num", "10",

         "conf.__widget__.model", "text",
         "conf.__widget__.labels", "text",
         "conf.__widget__.mean", "text",
         "conf.__widget__.display_num", "text",

         "conf.__type__.model", "string",
         "conf.__type__.labels", "string",
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
         - DefaultValue: googlenet.model
        """
        self._model = ['googlenet.model']
        """
        
         - Name:  labels
         - DefaultValue: labels.txt
        """
        self._labels = ['labels.txt']
        """
        
         - Name:  display_num
         - DefaultValue: 10
        """
        self._display_num = [10]
        
        # </rtc-template>

        self._net_model = GoogLeNet()

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
        self.bindParameter("model", self._model, "googlenet.model")
        self.bindParameter("labels", self._labels, "labels.txt")
        self.bindParameter("display_num", self._display_num, "10")
        
        # Set InPort buffers
        self.addInPort("in_image",self._in_imageIn)
        
        # Set OutPort buffers
        self.addOutPort("out_image",self._out_imageOut)
        
        # Set service provider to Ports
        
        # Set service consumers to Ports
        
        # Set CORBA Service Ports
        
        return RTC.RTC_OK
    
    #   ##
    #   # 
    #   # The finalize action (on ALIVE->END transition)
    #   # formaer rtc_exiting_entry()
    #   # 
    #   # @return RTC::ReturnCode_t
    #
    #   # 
    #def onFinalize(self):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The startup action when ExecutionContext startup
    #   # former rtc_starting_entry()
    #   # 
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onStartup(self, ec_id):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The shutdown action when ExecutionContext stop
    #   # former rtc_stopping_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onShutdown(self, ec_id):
    #
    #   return RTC.RTC_OK
    
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

        serializers.load_npz(self._model[0], self._net_model)

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

        # convert image data
        data = self._in_imageIn.read()
        image = np.frombuffer(data.pixels, dtype=np.uint8)
        image = image.reshape(data.height, data.width, 3)

        # load image
        img = cv2.resize(image, (224, 224)).astype(np.float32)
        img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224)

        # forward
        x = Variable(img)
        self._net_model.train = False
        y = self._net_model(x)

        # show prediction
        prediction = F.softmax(y)
        categories = np.loadtxt(self._labels[0], delimiter="\n", dtype=str)
        result = zip(prediction.data.reshape((prediction.data.size,)), categories)
        result = sorted(result, reverse=True)
        for i, (score, label) in enumerate(result[:self._display_num[0]]):
            print('{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label))

        return RTC.RTC_OK

    #   ##
    #   #
    #   # The aborting action when main logic error occurred.
    #   # former rtc_aborting_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onAborting(self, ec_id):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The error action in ERROR state
    #   # former rtc_error_do()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onError(self, ec_id):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The reset action that is invoked resetting
    #   # This is same but different the former rtc_init_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onReset(self, ec_id):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The state update action that is invoked after onExecute() action
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #

    #   #
    #def onStateUpdate(self, ec_id):
    #
    #   return RTC.RTC_OK
    
    #   ##
    #   #
    #   # The action that is invoked when execution context's rate is changed
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onRateChanged(self, ec_id):
    #
    #   return RTC.RTC_OK
    



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
