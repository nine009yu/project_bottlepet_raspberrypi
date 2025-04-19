from threading import Thread
import time
import queue
#send message
from linebot.v3.messaging import MessagingApi, ApiClient, Configuration
from linebot.v3.messaging.models import TextMessage, PushMessageRequest
channel_access_token = 'jNvR7g1S3S77nNKORcW1XDK6NdoqMXPEIC5UmJTEyUHjcsLojubyyEmgFFOFzV2vVHoHKr5Lk41Z42FDlDWzrys0FcvibgvzzcUWixXE00vHKV9uSyuuGxGG2YhXZg0lpJljdpLTS0YSHJ/Jr66Y+wdB04t89/1O/w1cDnyilFU='
configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
#gui
from tkinter import *
#image
from PIL import Image, ImageTk
#google sheet
import gspread
from google.oauth2.service_account import Credentials
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
client = gspread.authorize(creds)
sheet_id="1xmGjjdWdavRIb1rkevZKAigCFGyYOF1B8RM0bwXQQn0"
workbook = client.open_by_key(sheet_id)
row1 = len(workbook.sheet1.get_all_records())
# print(len(row))
#sensor
import RPi.GPIO as GPIO
sensor_pin = 17
servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)
# data_queue = queue.Queue()
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


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
    def set_angle(angle):
        duty = angle / 18 + 2
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)

    def move_servo():
        set_angle(90)  
        time.sleep(1)
        set_angle(0)  

    if num==10:
        if height < 200 :
           global num1
           num1+=1
           result1.config(text=num1)     
        else :
           global num2
           num2+=1
           result2.config(text=num2) 
        move_servo()
        break  

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
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
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

# if __name__ == '__main__':
  # while True:
  #   if GPIO.input(17) == 0:
  #     servo.angle = 90
  #     main()
def login():     
        global row1
        row2 = len(workbook.sheet1.get_all_records())
        name = workbook.sheet1.cell(row2+1, 2).value
        userid=workbook.sheet1.cell(row2+1, 1).value
        if row2 > row1 :
            row1=row2
            root2 = Tk()
            root2.title("Window2")
            root2.geometry("500x300+500+250")
            def add1():
                global num1
                num1+=1
                result1.config(text=num1) 
                
                # print(result1)       
            def add2():
                global num2
                num2+=1
                result2.config(text=num2)
            def submit():
                global num1
                global num2
                sum=(num1*10)+(num2*20)
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
                message = TextMessage(text='you small bottle'+' '+str(num1)+' '+'\nyou small bottle'+' '+str(num2)+' '+'\nsum'+' '+str(sum)+' '+'\ntotal'+' '+str(total))
                push_message_request = PushMessageRequest(to=user_id, messages=[message])
                # Label(root2,text=num1,fg="black",font=('arial',15,'bold')).place(x=150,y=130)
                # Label(root2,text=num2,fg="black",font=('arial',15,'bold')).place(x=360,y=130)
                line_bot_api.push_message(push_message_request)
                num1=0
                num2=0
                sum=0
                total=0
                global check
                check=False
                check_sensor()
                root2.destroy()  

            def check_sensor():
                global check
                global after_id_2
                if check :
                    # print(check)
                    if GPIO.input(sensor_pin)==0:
                      main()
                      
                else:
                    root2.after_cancel(after_id_2)
                    check=True
                    return
                after_id_2 = root2.after(500, check_sensor)
                # print(after_id_2)
            global result1
            global result2
            Label(root2,text=str(name),fg="white",font=('arial',20,'bold'),bg="black").pack(pady=25)
            bottle1 = Label(root2,text="small bottle",fg="white",font=('arial',20,'bold'),bg="green").place(x=80,y=80)
            bottle2 = Label(root2,text="big bottle",fg="white",font=('arial',20,'bold'),bg="green").place(x=300,y=80)
            result1 = Label(root2,text=num1,fg="black",font=('arial',15,'bold'))
            result1.place(x=150, y=130)
            result2 = Label(root2,text=num2,fg="black",font=('arial',15,'bold'))
            result2.place(x=360,y=130)
            btn1=Button(root2,text="add",fg="white",bg="blue",command=add1,font=('arial',15,'bold')).place(x=130,y=180)
            btn2=Button(root2,text="add",fg="white",bg="blue",command=add2,font=('arial',15,'bold')).place(x=340,y=180)
            btn3=Button(root2,text="submit",fg="white",bg="red",command=submit,font=('arial',15,'bold')).place(x=220,y=230)
            check_sensor()
            root2.mainloop()
            pwm.stop()
            GPIO.cleanup()
result1=None
result2=None
after_id_2 = None            
check=True
num1=0
num2=0
sum=0 
total=0
root1 = Tk()
root1.title("Window1")
root1.geometry("500x300+500+250")
image = Image.open("QR Code.png").resize((170,170))
photo = ImageTk.PhotoImage(image)
label = Label(root1, image=photo).place(x=170,y=25)
btn=Button(root1,text="Login",fg="white",bg="red",command=login,font=('arial',15,'bold')).place(x=220,y=220)
root1.mainloop() 
