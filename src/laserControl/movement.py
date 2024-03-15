# pip install pyserial
import serial.tools.list_ports
from pynput.mouse import Controller
import time

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
mouse = Controller()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = 4

for i in range(len(portsList)):
    if portsList[i].startswith("COM" + str(com)):
        use = "COM" + str(com)
        print(use)

serialInst.baudrate = 9600
serialInst.port = use
serialInst.open()

while True:
    # time.sleep(0)
    y,x = mouse.position

    x1Angle = round((180/1919 )* x)
    # x2Angle = round((180/1919 )* x)
    x2Angle=x1Angle
    x1Angle=x2Angle-180
    x1Angle*=-1
    yAngle = round((180/1079)*y)
    # command = input("Arduino Command (ON/OFF/exit): ")
    xAngle = f"{x1Angle},{x2Angle},{yAngle}\n"
    # print(xAngle)
    # command=command + '\n'
    # print(xAngle)
    # print(yAngle)
    serialInst.write(xAngle.encode('utf-8'))
    if serialInst.in_waiting > 0:
        received_data = serialInst.readline().decode('utf-8').strip()
        if received_data == 'exit':
            break

# while True:
#     time.sleep(1)
#     x, y = mouse.position
#     normalizedX = round(x - 1919/2)
#     xAngle = round((180/1919) * x)
#     normalizedY = round(-(y - 1079/2))
    
#     # Ensure normalizedX and normalizedY are within the valid range
#     normalizedX = max(-959, min(normalizedX, 959))
#     normalizedY = max(-539, min(normalizedY, 539))
    
#     command = f"{normalizedX}\n"
#     print(command)
#     serialInst.write(command.encode('utf-8'))
    
#     # Exit loop if 'exit' command is sent from the Arduino
#     if serialInst.in_waiting > 0:
#         received_data = serialInst.readline().decode('utf-8').strip()
#         if received_data == 'exit':
#             break

serialInst.close()
