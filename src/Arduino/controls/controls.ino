#include <Servo.h>

Servo Xservo;  // Create a servo object
Servo Yservo;
Servo rotateservo;
int XservoPin = 9;  // Servo signal pin
int YservoPin = 10;
int rotatePin =11;
int laserPin =7;


void setup() {
  Xservo.attach(XservoPin);  // Attach the servo to the pin
  Yservo.attach(YservoPin);
  rotateservo.attach(rotatePin);
  pinMode(laserPin,OUTPUT);
  Serial.begin(9600);  // Start serial communication
}

void loop() {
   digitalWrite(laserPin,HIGH);
  if (Serial.available() > 0) {

    String data = Serial.readStringUntil('\n');
    int commaIndex = data.indexOf(',');

    if (commaIndex != -1) {
      // Extract the x and y angles
      int commaIndex1 = data.indexOf(',');
      int commaIndex2 = data.indexOf(',', commaIndex1 + 1);
      int xAngle = data.substring(0, commaIndex1).toInt();
      int yAngle = data.substring(commaIndex1 + 1, commaIndex2).toInt();
      int zAngle = data.substring(commaIndex2 + 1).toInt();
      // int xAngle = data.substring(0, commaIndex).toInt();
      // int yAngle = data.substring(commaIndex + 1).toInt();
      // int commaIndex1 = data.indexOf(',');
      // int commaIndex2 = data.indexOf(',', commaIndex1 + 1);
      // int zAngle = data.substring(commaIndex2 + 2).toInt();
      digitalWrite(laserPin,HIGH);
      rotateservo.write(zAngle);

      // int normalizedX = Serial.parseInt();  // Read the normalized x-coordinate
      // int angle = map(normalizedX, -959, 959, 0, 180);  // Map to servo angle (0-180 degrees)
      // Set servo angle
      yAngle=xAngle-180;
      yAngle=yAngle*(-1);
      Xservo.write(xAngle);
      Yservo.write(yAngle);
    }
  }
}
  // #include <Servo.h>

  // int servoPin = 3;
  // Servo servo;
  // void setup() {
  //   Serial.begin(9600);
  //      servo.attach(servoPin);
  // }

  // void loop() {
  //   // servo.write(180);
  //   if (Serial.available() > 0){
  //     // delay(2000);
  //     String msg = Serial.readString();
  //     int angle = msg.toInt();
  //   servo.write(angle);
  //   }
  // }
