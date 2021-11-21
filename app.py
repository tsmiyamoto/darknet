# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tkinter
import tkinter.ttk as ttk

import time
from PIL import Image, ImageTk
import cv2

# for AWS S3
import boto3
import argparse

# for darknet
from ctypes import *
from pathlib import Path
global DARKNET_FORCE_CPU
DARKNET_FORCE_CPU = True
import cv2
import time
import darknet

# for multithread
import threading

import unicodedata
import random

global time_flag
time_flag = 0


class App:
    def __init__(self, window, window_title):
        #Camera initialization
        self.vcap = cv2.VideoCapture(0)

        self.window = window
        self.window.title(window_title)

        #initial window size
        self.window_width = 1700
        self.window_height = 800

        #window size
        self.window.geometry(f"{self.window_width}x{self.window_height}")

        #Close button
        self.close_btn = tkinter.Button(window, text='Close')
        self.close_btn.pack(side=tkinter.BOTTOM)
        self.close_btn.configure(command=self.destructor)

        #set camera's aspect
        self.CAMERA_WIDTH = 720
        self.CAMERA_HEIGHT = 480

        #set config_file, data_file, weights
        self.config_file = "yolov4.cfg"
        self.data_file = "classes.txt"
        self.weights = "yolov4_best.weights"

        #initialize darknet
        self.network, self.class_names, self.class_colors = darknet.load_network(config_file=self.config_file, data_file=self.data_file, weights=self.weights)

        self.img_width = darknet.network_width(self.network)
        self.img_height = darknet.network_height(self.network)

        self.darknet_img = darknet.make_image(self.img_width, self.img_height, 3)
        # 3 is color channels count https://github.com/AlexeyAB/darknet/blob/master/include/yolo_v2_class.hpp

        #揺れ補正用
        self.ng_count = 0
        self.corr_count = 0
        self.threshold = 15
        self.fsize = 50

        #Create widgets
        self.create_widgets()

        #set icon
        self.window.iconphoto(False, tkinter.PhotoImage(file='resource/tsp_icon.png'))

        #refresh update 15ms
        self.delay = 15
        self.update()

        #refresh update2 15ms
        self.delay2 = 15
        self.update2()

        self.parser()

        random.seed(3)

        self.window.mainloop()

    def parser(self):
        parser = argparse.ArgumentParser(description="S3 upload")
        parser.add_argument("--bucket", default="tsp-userfile62307-prod")
        parser.add_argument("--key")
        return parser.parse_args()

    def create_widgets(self):
        #Create logo image
        self.img = Image.open(open('resource/tsp_blue.png', 'rb'))
        self.img.thumbnail((200, 200), Image.ANTIALIAS)
        self.logo = ImageTk.PhotoImage(self.img)
        self.logo_canvas = tkinter.Canvas(self.window, width=300, height=100)
        self.logo_canvas.place(x=25, y=5)
        self.logo_canvas.create_image(100, 50, image=self.logo)

        #Create frame for camera live stream
        self.live_cam = tkinter.LabelFrame(text='Camera')
        self.live_cam.place(x=10, y=111)
        # self.live_cam.configure(width=self.CAMERA_WIDTH+20, height=self.CAMERA_HEIGHT+30)
        self.live_cam.configure(width=340, height=220)

        #Create canvas for live_cam
        self.live_canvas = tkinter.Canvas(self.live_cam)
        self.live_canvas.configure(width=self.CAMERA_WIDTH, height=self.CAMERA_HEIGHT)
        self.live_canvas.grid(column=0, row=0, padx=10, pady=10)

        #Create frame for result
        self.err_res = tkinter.LabelFrame(text='Result')
        self.err_res.place(x=800, y=10)
        self.err_res.configure(width=340, height=220)

        #Create notification text
        self.notice = tkinter.Label(self.err_res)
        self.notice.place(x=100, y=5)

        #Create canvas for NG image
        self.ng_canvas = tkinter.Canvas(self.err_res)
        self.ng_canvas.configure(width=self.CAMERA_WIDTH, height=self.CAMERA_HEIGHT)
        self.ng_canvas.grid(column=0, row=0, padx=10, pady=60)

        #Create model name label
        self.model_name = tkinter.Label(self.window, text=self.get_model_name())
        self.model_name.place(x=30, y=60)

        #Create device name label
        self.device_name = tkinter.Label(self.window, text=self.get_device_name())
        self.device_name.place(x=30, y=80)

    def update(self):
        _, frame = self.vcap.read()

        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))

        #### write prediction processing here!! ###
        resized_img = cv2.resize(frame, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_img, resized_img.tobytes())
        self.detections = darknet.detect_image(self.network, self.class_names, self.darknet_img, thresh=0.9)
        self.prev_img = darknet.draw_boxes(self.detections, resized_img, self.class_colors)


        self.prev_img = cv2.resize(self.prev_img, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))

        self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.prev_img))
        self.live_canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        ####

        s3_flag = 0
        self.corr_count = self.corr_count + 1

        #if ng_flag = 1, send to correction processing.
        if(self.check_ng()):
            s3_flag = self.correction()

        #揺れ補正の結果、異常と判断されれば別スレッドでS3に投げる。
        if(s3_flag):
            sub = threading.Thread(target=self.send_to_S3)
            sub.start()

        #refresh
        self.window.after(self.delay, self.update)

    def send_to_S3(self):
        print('send_to_S3... ', end='')

        timestamp = time.time()
        img_path = f'detection/{timestamp}.png'
        tmp_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, tmp_img)
        s3 = boto3.resource('s3')
        args = self.parser()
        bucket = s3.Bucket(str(args.bucket))
        key = str(args.key)
        bucket.upload_file(img_path, f'{key}/{timestamp}.png')

        print('done.')

    def check_ng(self):
        if (len(self.detections) != 0):
            for i in self.detections:
                if (unicodedata.normalize("NFKC", i[0]).upper().endswith('NG')):
                    return 1
        else:
            return 0

    def correction(self):
        #XXフレームためて、YY以上異常ならS3に通達
        self.ng_count = self.ng_count + 1
        if(self.corr_count >= self.fsize):
            self.corr_count = 0
            self.ng_count = 0
        if (self.ng_count >= self.threshold):
            self.corr_count = 0
            self.ng_count = 0
            return 1
        else:
            return 0

    def update2(self):
        global time_flag
        ng_flag = self.check_ng()

        if (ng_flag == 1):
            self.ng_message()
            time_flag = time.time()
            self.capture_frame = self.prev_img
            self.capture_frame = ImageTk.PhotoImage(image =Image.fromarray(self.capture_frame))
            self.ng_canvas.create_image(0, 0, image = self.capture_frame, anchor = tkinter.NW)
        elif (ng_flag == 0 and time.time()-time_flag <= 5):
            self.ng_message()
        elif (ng_flag == 0 and time.time()-time_flag > 5):
            self.ok_message()
            self.ng_canvas.delete()

        self.window.after(self.delay2, self.update2)

    def get_model_name(self):
        return ''

    def get_device_name(self):
        return ''

    def destructor(self):
        self.window.destroy()
        #Release camera
        self.vcap.release()

    def ok_message(self):
        message = ' 異常なし '
        self.notice['text'] = message
        self.notice['font'] = ('Arial', 30, 'bold')
        self.notice['fg'] = '#000000'
        self.notice['bg'] = '#cccccc'

    def ng_message(self):
        message = ' 異常あり '
        self.notice['text'] = message
        self.notice['font'] = ('Arial', 30, 'bold')
        self.notice['fg'] = '#ffffff'
        self.notice['bg'] = '#ff0000'


App(tkinter.Tk(), 'TSP - local')