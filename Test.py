#este es el prototipo funcionando todos los botones test 2


import tkinter as tk
import requests
from PIL import ImageTk,Image
from tkinter import filedialog

HEIGHT = 600
WIDTH = 700

def test_function(entry):
	print("This is the entry:", entry)

# api.openweathermap.org/data/2.5/forecast?q={city name},{country code}
# a4aa5e3d83ffefaba8c00284de6ef7c3

def format_response(weather):
	try:
		name = weather['name']
		desc = weather['weather'][0]['description']
		temp = weather['main']['temp']

		final_str = 'City: %s \nConditions: %s \nTemperature (°F): %s' % (name, desc, temp)
	except:
		final_str = 'There was a problem retrieving that information'

	return final_str

def get_weather(city):
	weather_key = 'a4aa5e3d83ffefaba8c00284de6ef7c3'
	url = 'https://api.openweathermap.org/data/2.5/weather'
	params = {'APPID': weather_key, 'q': city, 'units': 'imperial'}
	response = requests.get(url, params=params)
	weather = response.json()

	label['text'] = format_response(weather)

# def open():
# 	#global my_cvs
# 	root.filename = filedialog.askopenfilename
# 	#root.filename = FileDialog.askopenfilename(initialdir="Users\Arturo A", title="Selecciona el CVC", filetypes=((cvc files), ("*.cvc"), ("all files",("*.*"))
# 	my_label = Label(root, text=root.filename).pack()


root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

background_image = tk.PhotoImage(file='landscape.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)
##frame 0 top part how to
frame0 = tk.Frame(root, bg='red', bd=5)
frame0.place(relx=0.5, rely=0.1, relwidth=.90, relheight=0.1, anchor='n')

#my_label = tk.Label(frame0, text = root.filename)
#my_label.place(relwidth=0.65, relheight=1)

##button of cvc with specifications
button = tk.Button(frame0, text="1: Seleccionar CVC", font=40) #, command=open).pack()
button.place(relx=0.7, relheight=1, relwidth=0.3)

##frame 1 top part
frame1 = tk.Frame(root, bg='#80c1ff', bd=5)
frame1.place(relx=0.5, rely=0.3, relwidth=.85, relheight=0.1, anchor='n')

entry = tk.Entry(frame1, font=40)
entry.place(relwidth=0.65, relheight=1)

##button of weather with specifications
button = tk.Button(frame1, text="2: Resultados", font=40, command=lambda: get_weather(entry.get()))
button.place(relx=0.7, relheight=1, relwidth=0.3)

##frame 2 lower part
lower_frame = tk.Frame(root, bg='#80c1ff', bd=10)
lower_frame.place(relx=0.5, rely=0.45, relwidth=0.75, relheight=0.5, anchor='n')

label = tk.Label(lower_frame)
label.place(relwidth=1, relheight=1)

root.mainloop()