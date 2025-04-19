import gspread
from google.oauth2.service_account import Credentials

import customtkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
client = gspread.authorize(creds)
sheet_id="1xmGjjdWdavRIb1rkevZKAigCFGyYOF1B8RM0bwXQQn0"
workbook = client.open_by_key(sheet_id)

def show_frame(frame):
    frame.tkraise()
    

def check_data_and_open_next_frame():
    global row1
    global row2
    global should_check
    
    if should_check :
        row2 = len(workbook.sheet1.get_all_records())
        
        if row2 > row1:
            print(row1)
            print(row2)
            row1=row2
            should_check = False
            show_frame(frame2)
        else:
            print("er")
        root.after(1000, check_data_and_open_next_frame)
    return row2
def go_back_to_frame1():
    global should_check
    show_frame(frame1)
    should_check = True
    root.after(1000, check_data_and_open_next_frame)

def check():
    print(should_check)
    
should_check = True
row1 = len(workbook.sheet1.get_all_records())
root = tk.CTk()
root.title("window1")
root.geometry("500x300+500+250")

container = tk.Frame(root)
container.pack(fill="both", expand=True)


frame1 = tk.Frame(container)
frame2 = tk.Frame(container)

for frame in (frame1, frame2):
    frame.grid(row=0, column=0, sticky='nsew')

#frame1
image = Image.open("QR Code.png").resize((170,170))
photo = ImageTk.PhotoImage(image)
label1 = tk.Label(frame1, image=photo)
label1.pack(pady=50,padx=170)
button1 = tk.Button(frame1, text="check", command=lambda:check())
button1.pack(pady=5)
#frame2
c=check_data_and_open_next_frame()
name = workbook.sheet1.cell(c, 2).value
label2 = tk.Label(frame2, text=str(name),fg="white",font=('arial',20,'bold'),bg="black")
label2.pack(pady=25)
button2 = tk.Button(frame2, text="1", command=lambda: go_back_to_frame1())
button2.pack(pady=10)

show_frame(frame1)
root.after(1000, check_data_and_open_next_frame)

root.mainloop()
