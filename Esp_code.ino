//Libraries
#include <Arduino.h>
#include <FirebaseESP32.h>
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <MPU6050.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "time.h"
#include <EloquentTinyML.h>
#include "ecg_arrhythmia.h"
#include "x_test.h"
#include "time.h"

#define DEBUG 1

//Real time
const char* ntpServer = "pool.ntp.org";
double epochTime;

//Machine Learning parameter
#define NUMBER_OF_INPUTS  187
#define NUMBER_OF_OUTPUTS 5
#define TENSOR_ARENA_SIZE 25*1024
Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;
float y_pred[5] = {0};
int arrhythmia;

//Core
TaskHandle_t Task1;
TaskHandle_t Task2;
TaskHandle_t Task3;

//Wifi credentials
const char* ssid       = "SSID";
const char* password   = "password";

//Firebase variables
#define FIREBASE_HOST ""      //Enter firebase host name
#define FIREBASE_AUTH ""      //Enter firebase auth code
#define API_KEY ""            //Enter firebase api key
#define USER_EMAIL "dhruv@gmail.com"
#define USER_PASSWORD "dhruv@gmail.com"
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
String path = "";
String upath = "";
bool verify = true;
#define EEPROM_SIZE 2

//Max30102 heartbeat variables
MAX30105 particleSensor;
const byte RATE_SIZE = 10;
byte rates[RATE_SIZE]; //Array of heart rates

//Bmp280 variables
#define BMP_SDA 21
#define BMP_SCL 22
Adafruit_BMP280 bmp280;

//Max30102 spo2 variables
float spo = 0;
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
double ESpO2 = 95.0;//initial value of estimated SpO2
double FSpO2 = 0.7; //filter factor for estimated SpO2
double frate = 0.95; //low pass filter for IR/red LED value to eliminate AC component
int Num = 100;
double avered = 0;
double aveir = 0;
double sumirrms = 0;
double sumredrms = 0;
float temperature;
int i = 0;
uint32_t ir, red;
double fred, fir;
double SpO2 = 0;
#define TIMETOBOOT 3000
#define SCALE 88.0 //adjust to display heart beat and SpO2 in the same scale
#define SAMPLING 5 //if you want to see heart beat more precisely , set SAMPLING to 1
#define FINGER_ON 30000 // if red signal is lower than this , it indicates your finger is not on the sensor
#define MINIMUM_SPO2 80.0

//MPU6050 variables
MPU6050 mp;

//Multicore variables
SemaphoreHandle_t xMutex;
float alt;
float stemp;
float beatsPerMinute;
int beatAvg;
int battery;
float batt;
FirebaseJson arr;

//...................................................................................................................................................SETUP
void setup()
{
  //WiFi Connect
  Serial.begin(115200);
  Serial.printf("Connecting to %s ", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());

  //Firebase initialization
  config.host = FIREBASE_HOST;
  config.api_key = API_KEY;
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;
  String base_path = "/Userbodydata/";
  Firebase.begin(&config, &auth);
  path = base_path + auth.token.uid.c_str();
  while (!(Firebase.set(fbdo, path + "/verify", verify)))
  {
    Serial.println("FAILED Check user's email and password");
    delay(500);
  }
  Firebase.setFloatDigits(2);
#if DEBUG
  Serial.println("PASSED");
  Serial.println(path);
#endif
  Firebase.getBool(fbdo, path + "/Reset");
  if (fbdo.boolData())
  {
    Firebase.deleteNode(fbdo, path);
  }
  Firebase.setBool(fbdo, path + "/Reset", false);
  Firebase.setInt(fbdo, path + "/tinyML", 0);
  pinMode(25, OUTPUT);
  pinMode(39, INPUT);
  pinMode(35, INPUT);

  //Realtime clock initialization
  configTime(0, 0, ntpServer);

  //MAX30102 initialization
  particleSensor.begin(Wire, I2C_SPEED_FAST);
  particleSensor.setup();

  //BMP280 initialization
  bmp280.begin(0x76);

  //MPU6050 Settings
  mp.begin(MPU6050_SCALE_2000DPS, MPU6050_RANGE_16G);
  mp.setAccelPowerOnDelay(MPU6050_DELAY_3MS);
  mp.setIntFreeFallEnabled(true);
  mp.setIntZeroMotionEnabled(false);
  mp.setIntMotionEnabled(false);
  mp.setDHPFMode(MPU6050_DHPF_5HZ);
  mp.setFreeFallDetectionThreshold(17);
  mp.setFreeFallDetectionDuration(2);
  attachInterrupt(35, doInt, RISING);

  //Machine learning model load
  ml.begin(ecg_arrhythmia);
#if DEBUG
  Serial.print("Size of machine learning model is:");
  Serial.println(sizeof(ecg_arrhythmia));
#endif

  //MUTEX
  xMutex = xSemaphoreCreateMutex();
  xTaskCreatePinnedToCore(Task1code, "Task1", 50000, NULL, 2, &Task1, 0);
  delay(500);
  xTaskCreatePinnedToCore(Task2code, "Task2", 25000, NULL, 2, &Task2, 1);
  delay(500);
}

//...................................................................................................................................................INTERRUPT
void doInt()
{
  //if(beatAvg != 0){}
  xTaskCreatePinnedToCore(vibrate, "Task3", 4000, NULL, 1, &Task3, 0);
  Serial.println("Free Fall Detected");
}

//...................................................................................................................................................TASK 1
void Task1code( void * pvParameters )
{
#if DEBUG
  Serial.print("Task1 running on core ");
  Serial.println(xPortGetCoreID());
#endif
  bool state = 0;
  while (1)
  {
    unsigned long currentTime = millis();
    unsigned long previousTime_1 = currentTime;
    //................................................................................................................................................HEARTBEAT MEASURE
    while ( previousTime_1 - currentTime  <= 7000)
    {
      long irValue = particleSensor.getIR();
      if (checkForBeat(irValue) == true)
      {
        long delta = millis() - lastBeat;
        lastBeat = millis();
        beatsPerMinute = 60 / (delta / 1000.0);

        if (beatsPerMinute < 255 && beatsPerMinute > 20)
        {
          rates[rateSpot++] = (byte)beatsPerMinute;
          rateSpot %= RATE_SIZE;

          for (byte x = 0 ; x < RATE_SIZE ; x++)
            beatAvg += rates[x];
          beatAvg /= RATE_SIZE;
        }
      }
      state = 0;
      if (irValue < 50000)
      {
        state = 1;
      }
      previousTime_1 = millis();
    }
    if (state == 0 && beatAvg > 0)
    {
#if DEBUG
      Serial.print("Avg BPM=");
      Serial.println(beatAvg);
#endif
    }
    else
    {
      beatAvg = 0;
#if DEBUG
      Serial.print("Avg BPM=");
      Serial.println(beatAvg);
#endif
    }
    //................................................................................................................................................SPO2 MEASURE
    bool s = 1;
    currentTime = millis();
    previousTime_1 = currentTime;
    while ( previousTime_1 - currentTime  <= 7000)
    {
      if (s == 1)
      {
        byte ledBrightness = 0x7F; //Options: 0=Off to 255=50mA
        byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
        byte ledMode = 2; //Options: 1 = IR only, 2 = Red + IR
        int sampleRate = 200; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
        int pulseWidth = 411; //Options: 69, 118, 215, 411
        int adcRange = 16384; //Options: 2048, 4096, 8192, 16384

        particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
        s = 0;
      }
      particleSensor.check(); //Check the sensor, read up to 3 samples
      while (1) //do we have new data
      {
        red = particleSensor.getFIFOIR();
        ir = particleSensor.getFIFORed();
        i++;
        fred = (double)red;
        fir = (double)ir;
        avered = avered * frate + (double)red * (1.0 - frate);//average red level by low pass filter
        aveir = aveir * frate + (double)ir * (1.0 - frate); //average IR level by low pass filter
        sumredrms += (fred - avered) * (fred - avered); //square sum of alternate component of red level
        sumirrms += (fir - aveir) * (fir - aveir);//square sum of alternate component of IR level
        if ((i % SAMPLING) == 0) //For slowdown
        {
          if (ir < FINGER_ON)
          {
            ESpO2 = MINIMUM_SPO2;
            state = 1;
          }
          else
          {
            temperature = particleSensor.readTemperatureF();
            state = 0;
          }
        }
        if ((i % Num) == 0)
        {
          double R = (sqrt(sumredrms) / avered) / (sqrt(sumirrms) / aveir);
          SpO2 = -23.3 * (R - 0.4) + 100;
          ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;//low pass filter
          sumredrms = 0.0; sumirrms = 0.0; i = 0;
          break;
        }
        particleSensor.nextSample(); //We're finished with this sample so move to next sample
      }
      previousTime_1 = millis();
    }
    if (state == 0)
    {
      spo = (float)ESpO2;
    }
    else
    {
      spo = 0;
      temperature = 0;
    }
#if DEBUG
    Serial.print(" Body Temperature = ");
    Serial.println(temperature);
    Serial.print(" Oxygen % = ");
    Serial.println(spo);
#endif
    //................................................................................................................................................ALTITUDE AND TEMP
    //stemp = bmp280.readTemperature();
    alt = bmp280.readAltitude(1012.15);
    stemp = mp.readTemperature();
#if DEBUG
    Serial.print("Altitude:");
    Serial.println(alt);
    Serial.print("Surrounding Temperature:");
    Serial.println(stemp);
#endif

    time_t now;
    time(&now);
    epochTime = now;
    upath = path + "/Data/";
    arr.clear();
    arr.add("0", beatAvg);
    arr.add("1", temperature);
    arr.add("2", spo);
    arr.add("3", alt);
    arr.add("4", stemp);
    arr.add("Timestamp", epochTime);
    batt = analogRead(39);
    batt = (1.6113 * batt) / 1000;
    if (batt <= 3.6)
    {
      batt = 3.6;
    }
    battery = int((batt - 3.6) / 0.006);
#if DEBUG
    Serial.print("Battery percentage is:");
    Serial.println(battery);
#endif
    //.........................................................................MUTEX
    xSemaphoreTake( xMutex, portMAX_DELAY );
    Firebase.push(fbdo, upath, arr);
    Firebase.setInt(fbdo, path + "/Battery", battery);
    xSemaphoreGive( xMutex );
    //.........................................................................MUTEX RELEASE
    //Serial.print("Free heap (bytes)1: ");
    //Serial.println(xPortGetFreeHeapSize());
    //Serial.print("Free stack (bytes)1: ");
    //Serial.println(uxTaskGetStackHighWaterMark(NULL));
  }
  vTaskDelete(NULL);
}

//...................................................................................................................................................TASK 2
void Task2code( void * pvParameters )
{
#if DEBUG
  Serial.print("Task2 running on core ");
  Serial.println(xPortGetCoreID());
#endif
  while (1)
  {
    //.........................................................................MUTEX
    xSemaphoreTake( xMutex, portMAX_DELAY );
    Firebase.getInt(fbdo, path + "/tinyML");
    xSemaphoreGive( xMutex );
    //.........................................................................MUTEX RELEASE
    uint32_t start = millis();
    switch (fbdo.intData())
    {
      case 0:
        ml.predict(x_test_dat0, y_pred);
        break;
      case 1:
        ml.predict(x_test_dat1, y_pred);
        break;
      case 2:
        ml.predict(x_test_dat2, y_pred);
        break;
      case 3:
        ml.predict(x_test_dat3, y_pred);
        break;
      case 4:
        ml.predict(x_test_dat4, y_pred);
        break;
      default:
        break;
    }
    uint32_t timeit = millis() - start;
#if DEBUG
    Serial.print("It took ");
    Serial.print(timeit);
    Serial.println(" millis to run inference.");
    Serial.print("It can process ");
    Serial.print(60000 / timeit);
    Serial.println(" BPM.");
#endif
    for (int i = 0; i < 5; i++)
    {
#if DEBUG
      Serial.print(y_pred[i]);
      Serial.print(i == 4 ? '\n' : ',');
#endif
      y_pred[i] = y_pred[i] + 0.5;
      y_pred[i] = (int)y_pred[i];
      if (y_pred[i] == 1)
      {
        arrhythmia = i;
      }
    }
    if (arrhythmia != 0)
    {
#if DEBUG
      Serial.println(arrhythmia);
#endif
      //.........................................................................MUTEX
      xSemaphoreTake( xMutex, portMAX_DELAY );
      Firebase.setTimestamp(fbdo, path + "/tinyMLans" + arrhythmia);
      Firebase.setBool(fbdo, path + "/Emergency", true);
      xSemaphoreGive( xMutex );
      //.........................................................................MUTEX RELEASE
    }
    delay(100);
  }
  vTaskDelete(NULL);
}

void vibrate( void * pvParameters )
{
  //.........................................................................MUTEX
  xSemaphoreTake( xMutex, portMAX_DELAY );
  Firebase.setTimestamp(fbdo, path + "/Freefall");
  Firebase.setBool(fbdo, path + "/Emergency", true);
  xSemaphoreGive( xMutex );
  //.........................................................................MUTEX RELEASE
  digitalWrite(25, HIGH);
  vTaskDelay(500 / portTICK_PERIOD_MS);
  digitalWrite(25, LOW);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  digitalWrite(25, HIGH);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
  digitalWrite(25, LOW);
  vTaskDelete(NULL);
}
void loop() {}
