import PySimpleGUI as sg
import yarp
from yarp import IFrameGrabberControls

EXPOSURE=1
GAIN=8
HUE=4

props = yarp.Property()
props.put('device', 'RGBDSensorClient')
props.put('localImagePort', '/RealsenseGUI/rgbImage:i')
props.put('localDepthPort', '/RealsenseGUI/depthImage:i')
props.put('localRpcPort', '/RealsenseGUI/rpc:i')

props.put('remoteImagePort', '/depthCamera/rgbImage:o')
props.put('remoteDepthPort', '/depthCamera/depthImage:o')
props.put('remoteRpcPort', '/depthCamera/rpc:i')

driver = yarp.PolyDriver(props)

yarp.Network.disconnect('/depthCamera/rgbImage:o', '/RealsenseGUI/rgbImage:i')
yarp.Network.disconnect('/depthCamera/depthImage:o', '/RealsenseGUI/depthImage:i')

iface = driver.viewIFrameGrabberControls()

# Define the custom theme
sg.theme('DarkAmber')
sg.set_options(font=('Helvetica', 12))

# Define the layout of the interface
layout = [
    [sg.Text('Exposure', size=(10, 1)),
     sg.Slider(range=(0, 4), resolution=0.01, default_value=50, orientation='h', size=(20, 15), key='exposure_sl',
               enable_events=True,
               border_width=0), sg.Spin(values=[i for i in range(1000)], initial_value=50, size=(8, 4),
                                        enable_events=True, key="exposure_tb")],
    [sg.Text('Gain', size=(10, 1)),
     sg.Slider(range=(0, 100), default_value=50, orientation='h', size=(20, 15), key='gain_sl', enable_events=True,
               border_width=0), sg.InputText(size=(5, 1), key='gain_tb')],
    [sg.Text('Hue', size=(10, 1)),
     sg.Slider(range=(0, 100), default_value=50, orientation='h', size=(20, 15), key='hue_sl', enable_events=True,
               border_width=0), sg.InputText(size=(5, 1), key='hue_tb')]
]

# Default Camera Parameters tested in roboarena

iface.setFeature(EXPOSURE, 0.75 / 100.)
iface.setFeature(GAIN, 35 / 100.)
iface.setFeature(HUE, 55 / 100.)


# Create the window
window = sg.Window('Realsense Parameters', layout)

# Event loop
while True:
    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Cancel'):
        break
    if event[:-3] == 'exposure':
        value = values[event]
        window.Element("exposure_tb").Update(value)
        window.Element("exposure_sl").Update(value)
        iface.setFeature(EXPOSURE, value / 100.)

    elif event == 'gain_sl':
        value = values['gain_sl']
        iface.setFeature(GAIN, value / 100.)

    elif event == 'hue_sl':
        value = values['hue_sl']
        iface.setFeature(HUE, value / 100.)

    # # Update the slider values based on their text inputs
    # for key in ('exposure', 'gain', 'hue'):
    #     if values[key + '_tb']:
    #         slider = window[key]
    #         slider.update(int(values[key + '_tb']))

    # Real-time value display
    for key, value in values.items():
        window[key[:-2] + 'tb'].update(value)

# Close the window
window.close()
