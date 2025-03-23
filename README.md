# Temp.-Data-Analysis
Temperature/Environmental Data Acquisition and Quick Analysis

### **Overview**
This project demonstrates how to collect, transfer, analyze, and visualize environmental data using a MicroPython-enabled board (such as an STM32 or ESP32) and a DHT11 sensor. The data is transferred to a PC where it is preprocessed, analyzed, and visualized using Python.

## **Project Steps**

I. Hardware Setup & Data Collection

1. Integrate a DHT11 sensor with the development board.
2. Use MicroPython code to periodically read temperature and humidity data (e.g., every 10 seconds).
3. Store or stream the collected readings.

II. Data Transfer & Preprocessing

1. Transfer the recorded data from the board to a PC via serial communication.
2. Use Python (with libraries such as Pandas and NumPy) to clean the data, remove outliers, and compute basic statistics.

III. Further Analysis

1. Apply simple optimization techniques such as least-squares regression or polynomial fitting to model temperature trends.
2. Demonstrate a recursive or iterative solution to a small optimization problem.

IV. Visualization & Reporting

1. Create time-series plots or other graphs using Matplotlib to visualize the trends in the data.
2. Compile findings into a concise report or presentation that summarizes the data collection, analysis process, and key insights.

### **Components**

- MicroPython-Enabled Board (STM32/ESP32)
- DHT11 Sensor
- USB-to-Serial Adapter (for transferring data to a PC)
- PC with Python (using packages such as Pandas, NumPy, Matplotlib, and pySerial)
- Connecting Wires and Breadboard

#### **Usage**

	On the Microcontroller:
	Flash the board with MicroPython firmware and upload the provided data collection script.

	On the PC:
	Run the data transfer and analysis scripts to capture, preprocess, analyze, and visualize the sensor data.


## **Notes**

+ This project was developed as part of a final project for a Python lecture at SRH University Heidelberg (March 12, 2025).
+ Pin configurations and other settings should be adjusted according to your specific hardware setup.
+ The project demonstrates integration of sensor data acquisition with further data analysis and visualization techniques.
