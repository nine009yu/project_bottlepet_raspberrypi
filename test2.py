#sensor
import time
import RPi.GPIO as GPIO
sensor_pin = 17
servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)
GPIO.setup(servo_pin, GPIO.OUT)
pwm = GPIO.PWM(servo_pin, 50)
def set_angle(angle):
        duty = angle / 18 + 2
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(servo_pin, False)
        pwm.ChangeDutyCycle(0)
def move_servo():
        pwm.start(0)
        set_angle(90)  
        time.sleep(1)
        set_angle(0)  
        pwm.stop()
move_servo()