
int solenoid = 5;

int solenoidState = LOW;  // ledState used to set the solenoid

unsigned long previousMillis = 0;  // will store last time solenoid was updated

const long interval = 1000/60;  // interval at which to blink (milliseconds)

void setup() {
  // put your setup code here, to run once:
  pinMode(solenoid, OUTPUT);
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    if (solenoidState == LOW) {
      solenoidState = HIGH;
    } else {
      solenoidState = LOW;
    }

    // set the soilenoid with the solenoidState of the variable:
    digitalWrite(solenoid, solenoidState);
  }
}