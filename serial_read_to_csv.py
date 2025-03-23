import serial
import csv

SERIAL_PORT = "COM6"
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

with open("sensor_data.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    
    csv_writer.writerow(["timestamp", "temperature", "humidity"])
    
    print("Listening for data... Press Ctrl+C to stop.")
    
    try:
        while True:
            line = ser.readline().decode("utf-8").strip()
            
            if line:
                print("Received:", line)
                
                parts = line.split(",")
                
                if len(parts) == 3:
                    csv_writer.writerow(parts)
                    csvfile.flush()
    except KeyboardInterrupt:
        print("\nStopped listening.")

ser.close()