import PySimpleGUI as sg

layout = [[sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=85)],
          [sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=25)],
          [sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=50)],
          [sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=75)]]

layout2 = [[sg.Button('Button1')],
           [sg.Button('Button2')],
           [sg.Button('Button3')],
           [sg.Button('Button4')]]

layout3 = [[sg.Column(layout2)],
           [sg.Column(layout)]]

window = sg.Window('Window Title', layout3)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

window.close()