import dht
import machine
import time
 
sensor = dht.DHT11(machine.Pin('PA1', machine.Pin.IN))

uart = machine.UART(2, baudrate=115200)

while True:
    try:
        sensor.measure()
        temperature = sensor.temperature()
        humidity = sensor.humidity()
 
        print("Temperature: {}°C, Humidity: {}%".format(temperature, humidity))
 
        uart.write("Temp: {}°C, Humidity: {}%\n".format(temperature, humidity))
 
    except Exception as e:
        print("Error reading sensor:", e)
 
    time.sleep(10)