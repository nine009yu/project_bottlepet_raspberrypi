# -*- coding: utf-8 -*-

#gui
from threading import Thread
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.progressbar import ProgressBar
from kivy.animation import Animation
from kivy.metrics import dp

from kivy.resources import resource_add_path
from kivy.uix.popup import Popup
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.core.text import LabelBase
LabelBase.register(name='THSarabunNew', fn_regular='/home/nine/Desktop/bottlepet/Roboto-VariableFont_wdth,wght.ttf')


# Google Sheet
import gspread
from google.oauth2.service_account import Credentials
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
client = gspread.authorize(creds)
sheet_id = "1xmGjjdWdavRIb1rkevZKAigCFGyYOF1B8RM0bwXQQn0"
workbook = client.open_by_key(sheet_id)

#send message
from datetime import datetime
from linebot.v3.messaging import MessagingApi, ApiClient, Configuration
from linebot.v3.messaging.models import TextMessage, PushMessageRequest, FlexMessage
channel_access_token = 'jNvR7g1S3S77nNKORcW1XDK6NdoqMXPEIC5UmJTEyUHjcsLojubyyEmgFFOFzV2vVHoHKr5Lk41Z42FDlDWzrys0FcvibgvzzcUWixXE00vHKV9uSyuuGxGG2YhXZg0lpJljdpLTS0YSHJ/Jr66Y+wdB04t89/1O/w1cDnyilFU='
configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

#sensor
import RPi.GPIO as GPIO
sensor_pin = 17
servo_pin = 18
servo_pin2 = 14
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(servo_pin2, GPIO.OUT)
pwm1 = GPIO.PWM(servo_pin, 50)
pwm2 = GPIO.PWM(servo_pin2, 50)
pwm1.start(0)
pwm2.start(0)


"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm1.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    pwm1.ChangeDutyCycle(0)

def move_servo():
    set_angle(90)
    time.sleep(0.5)
    set_angle(0)
def set_angle2(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin2, True)
    pwm2.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin2, False)
    pwm2.ChangeDutyCycle(0)

def move_servo2():
    set_angle2(90)
    
def move_servo3():
    set_angle2(0)

def alert(message):
    popup = Popup(
        title="Warning",
        title_font='/home/nine/Desktop/bottlepet/Roboto-VariableFont_wdth,wght.ttf',
        title_size='30sp',
        content=Label(
            text=message,
            font_name='THSarabunNew',
            font_size='30sp',  
            halign='center',
            valign='middle'
        ),
        size_hint=(None, None),
        size=(800, 300)  
    )
    popup.open()



def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  start_time2 = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.7)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  num=0
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)
    
    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    
    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    detections = detection_result.detections

    for detection in detections:
        # print(detection)
        label = detection.categories[0].category_name
        height = detection.bounding_box.height
        # height = bounding_box.height
        print(height)
        # heights = []
        # heights.append(height)
        # for i, height in enumerate(heights):
        #   print(f"Detection {i + 1} height: {height}")
        print(f"Detected object: {label}")
        if label == "Bottle-pet" :
          num+=1
          print(num)
    
    if num==5:
        if height < 200 :
           global num1
           num1+=1
        else :
           global num2
           num2+=1

        move_servo()
        num = 0 
        break  

    elapsed_time = time.time() - start_time2
    if elapsed_time > 10:
        alert("An error occurred. Please reinsert or replace the bottle")
        break
        
    #print(elapsed_time)
    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)
    # print(detection_result)
    
    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    # cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()

def main():

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      #default='efficientdet_lite0.tflite')
      default='pet.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))

# Custom Loading Screen
class LoadingScreen(Popup):
    def __init__(self, **kwargs):
        super(LoadingScreen, self).__init__(**kwargs)
        self.title = 'Processing data'
        self.title_size = '30sp'
        self.title_font = 'THSarabunNew'
        self.size_hint = (None, None)
        self.size = (800, 300)
        self.auto_dismiss = False
        
        content = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        label = Label(
            text='please wait a moment...',
            font_name='THSarabunNew',
            font_size='30sp'
        )
        
        self.progress = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(20))
        
        content.add_widget(label)
        content.add_widget(self.progress)
        
        self.content = content
        self._start_progress()
        
    def _start_progress(self):
        anim = Animation(value=100, duration=2)
        anim.bind(on_complete=self._progress_complete)
        anim.start(self.progress)
        
    def _progress_complete(self, *args):
        # Reset and repeat animation for continuous effect
        self.progress.value = 0
        self._start_progress()


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        layout = FloatLayout()
        with layout.canvas.before:
            Color(0.88, 1, 0.88, 1)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)

        layout.bind(size=self._update_rect, pos=self._update_rect)
        img = Image(source='QR Code.png', size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.6})
        layout.add_widget(img)
        label_scan = Label(text="Please scan the QR code to sign in.",font_name='THSarabunNew', font_size='80sp', 
                                size_hint=(None, None), pos_hint={'center_x': 0.5, 'y': 0.2}, color=(0, 0, 0, 1))
        layout.add_widget(label_scan)
        self.add_widget(layout)
        self.event = None
    
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos
    
    def on_pre_enter(self):
        if not self.event: 
            self.event = Clock.schedule_interval(self.check_condition, 5)
        
    def check_condition(self, dt):
        global row1,row2
        row2 = len(workbook.sheet1.get_all_records())
        print("wait data")
        print(row1)
        print(row2)
        if row2 > row1:
            row1 = row2
            if self.event:
                self.event.cancel()
                self.event = None
            self.manager.current = 'second'    

class SecondScreen(Screen):
    def on_pre_enter(self):
        global num1,num2,row2
        layout = FloatLayout()
        with layout.canvas.before:
            Color(0.88, 1, 0.88, 1) 
            self.rect = Rectangle(size=Window.size, pos=layout.pos)
        print(row2)
        self.sensor_event = None
        usname = workbook.sheet1.cell(row2+1, 2).value

        if not self.sensor_event:
            self.sensor_event = Clock.schedule_interval(self.check_sensor, 3)
        

        label_top = Label(
            text="Welcome, " + str(usname),
            font_name='THSarabunNew', font_size='80sp',
            size_hint=(None, None), pos_hint={'center_x': 0.5, 'top': 0.95},
            color=(0, 0, 0, 1)
        )
        layout.add_widget(label_top)

        # Small bottle image
        img_small = Image(
            source='bottle.png',  
            size_hint=(0.2, 0.2),       
            pos_hint={'x': 0.18, 'y': 0.5}
        )
        layout.add_widget(img_small)

        # Big bottle image
        img_big = Image(
            source='bottle.png',
            size_hint=(0.25, 0.25),     
            pos_hint={'x': 0.59, 'y': 0.5}
        )
        layout.add_widget(img_big)

        # smallbottle
        label_left = Label(
            text='Small bottle',
            font_name='THSarabunNew', font_size='70sp',
            size_hint=(None, None), pos_hint={'x': 0.25, 'y': 0.35},
            color=(0, 0, 0, 1)
        )
        layout.add_widget(label_left)

        # bigbottle
        label_right = Label(
            text='Large bottle',
            font_name='THSarabunNew', font_size='70sp',
            size_hint=(None, None), pos_hint={'x': 0.69, 'y': 0.35},
            color=(0, 0, 0, 1)
        )
        layout.add_widget(label_right)

        # ===== Numbers under the text =====

        # num1
        self.label_num1 = Label(
            text=str(num1), font_size='70sp',
            size_hint=(None, None), pos_hint={'x': 0.25, 'y': 0.25},
            color=(0, 0, 0, 1)
        )
        layout.add_widget(self.label_num1)

        # num2
        self.label_num2 = Label(
            text=str(num2), font_size='70sp',
            size_hint=(None, None), pos_hint={'x': 0.69, 'y': 0.25},
            color=(0, 0, 0, 1)
        )
        layout.add_widget(self.label_num2)


        # #addnum1
        # button_addnum1 = Button(text='add', size_hint=(0.1, 0.1), pos_hint={'center_x': 0.27, 'y': 0.5},background_color=(1, 0, 0))
        # button_addnum1.bind(on_press=self.add_num1)
        # layout.add_widget(button_addnum1)
        
        # #addnum2
        # button_addnum2 = Button(text='add', size_hint=(0.1, 0.1), pos_hint={'center_x': 0.77, 'y': 0.5},background_color=(1, 0, 0))
        # button_addnum2.bind(on_press=self.add_num2)
        # layout.add_widget(button_addnum2)

        #submit
        button_submit = Button(text='Confirm Points',font_name='THSarabunNew', font_size='60sp', size_hint=(0.4, 0.1), pos_hint={'center_x': 0.5, 'y': 0.1},background_color=(0, 0.6, 0.4, 1))
        button_submit.bind(on_press=self.start_process)
        layout.add_widget(button_submit)

        self.add_widget(layout)
        # self.sensor_event = None
    
    def show_loading(self):
        self.popup = LoadingScreen()
        self.popup.open()

    def close_loading(self):
        if hasattr(self, 'popup') and self.popup:
            self.popup.dismiss()

    def start_process(self, instance):
        if num1>0 or num2>0:
            # Show loading popup
            self.show_loading()
            # Run process in separate thread
            Thread(target=self.submit).start()
        else:
            alert("Please put in a bottle.")
    
    
        

    def check_sensor(self, dt) :
        print("wait sensor")
        if GPIO.input(sensor_pin)==0:
            move_servo2()
            main()
            global num1,num2
            self.label_num1.text = str(num1)
            self.label_num2.text = str(num2)
            move_servo3()
            # Clock.schedule_once(lambda dt: self.update_label_num1(num1))

    def add_num1(self, instance):
        global num1
        num1 += 1
        self.label_num1.text = str(num1)

    def add_num2(self, instance):
        global num2
        num2 += 1
        self.label_num2.text = str(num2)

    def submit(self):
        global num1,num2
        sum=(num1*10)+(num2*20)
        userid=workbook.sheet1.cell(row2+1, 1).value
        name=workbook.sheet1.cell(row2+1, 2).value
        user_id = userid

        #database
        sheet=workbook.worksheet("sheet2")
        row3 = len(sheet.get_all_records())
        cell = sheet.find(userid) 
        if cell == None:
            sheet.update_cell(row3+2, 1,userid)
            sheet.update_cell(row3+2, 2,name)
            sheet.update_cell(row3+2, 3,sum)
            total=int(sheet.cell(row3+2, 3).value)
        else:
            row4=cell.row
            val=0
            val = int(sheet.cell(row4, 3).value)
            val=sum+val
            sheet.update_cell(row4, 3,val)
            total=int(sheet.cell(row4, 3).value)

        #send message
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        msg_contents = {
            "type": "flex",
            "altText": "เนเธเน€เธชเธฃเนเธเธชเธฐเธชเธกเธเธฐเนเธเธ!",
            "contents": {
                "type": "bubble",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingBottom": "none",
                    "contents": [
                        {
                            "type": "text",
                            "text": "เนเธเน€เธชเธฃเนเธเธชเธฐเธชเธกเธเธฐเนเธเธ",
                            "weight": "bold",
                            "color": "#1DB446",
                            "size": "lg",
                            "align": "center"
                        },
                        {
                            "type": "separator",
                            "margin": "xl"
                        }
                    ]
                },
                "body": {
                    "type": "box",
                    "margin": "xl",
                    "layout": "vertical",
                    # "paddingTop": "none", 
                    "contents": []
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingTop": "none", 
                    "contents": [
                        {
                            "type": "text",
                            "text": f"{now}",
                            "size": "xs",
                            "color": "#aaaaaa",
                            "align": "end"
                        }
                    ]
                }
            }
        }

        if num1 >= 1:
            msg_contents["contents"]["body"]["contents"].append({
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": "เธเธงเธ”เน€เธฅเนเธ",
                        "size": "sm",
                        "color": "#555555",
                        "flex": 0
                    },
                    {
                        "type": "text",
                        "text": f"{num1} เธเธงเธ” {num1 * 10} เธเธฐเนเธเธ",
                        "size": "sm",
                        "color": "#111111",
                        "align": "end"
                    }
                ]
            })

        if num2 > 0:
            msg_contents["contents"]["body"]["contents"].append({
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": "เธเธงเธ”เนเธซเธเน",
                        "size": "sm",
                        "color": "#555555",
                        "flex": 0
                    },
                    {
                        "type": "text",
                        "text": f"{num2} เธเธงเธ” {num2 * 20} เธเธฐเนเธเธ",
                        "size": "sm",
                        "color": "#111111",
                        "align": "end"
                    }
                ]
            })

        msg_contents["contents"]["body"]["contents"].append({
            "type": "separator",
            "margin": "xl"
        })

        msg_contents["contents"]["body"]["contents"].append({
            "type": "box",
            "layout": "horizontal",
            "margin": "xl",
            "contents": [
                {
                    "type": "text",
                    "text": "เธเธฐเนเธเธเธฃเธงเธก",
                    "size": "sm",
                    "color": "#555555",
                    "weight": "bold",
                    "flex": 0
                },
                {
                    "type": "text",
                    "text": f"{sum} เธเธฐเนเธเธ",
                    "size": "sm",
                    "color": "#111111",
                    "align": "end",
                    "weight": "bold"
                }
            ]
        })

        msg_contents["contents"]["body"]["contents"].append({
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {
                    "type": "text",
                    "text": "เธเธฐเนเธเธเธ—เธฑเนเธเธซเธกเธ”",
                    "size": "sm",
                    "color": "#555555",
                    "weight": "bold",
                    "flex": 0
                },
                {
                    "type": "text",
                    "text": f"{total} เธเธฐเนเธเธ",
                    "size": "sm",
                    "color": "#1DB446",
                    "align": "end",
                    "weight": "bold"
                }
            ]
        })

        msg_contents["contents"]["body"]["contents"].append({
            "type": "separator",
            "margin": "xl"
        })

        msg = FlexMessage.from_dict(msg_contents)

        push_message_request = PushMessageRequest(
            to=user_id,
            messages=[msg]
        )

        line_bot_api.push_message(push_message_request)

        num1 = 0
        num2 = 0
        self.label_num1.text = str(num1)
        self.label_num2.text = str(num2)
        sum=0
        total=0
        if self.sensor_event:
            self.sensor_event.cancel()
            self.sensor_event = None
        print(self.sensor_event)
        Clock.schedule_once(self.back_to_main_screen)
        

    def back_to_main_screen(self, dt):
        self.manager.current = 'main'
        self.close_loading()

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(SecondScreen(name='second'))
        return sm

row1 = len(workbook.sheet1.get_all_records())
row2 = 0
num1 = 0
num2 = 0
if __name__ == '__main__':
    Window.fullscreen = True
    MyApp().run()
    GPIO.cleanup()
    
