// librairies
#include <SoftwareSerial.h>
#include <Arduino.h>
#include <LiquidCrystal.h>

#define BUZZER 9
#define DOORLOCKA 7
#define DOORLOCKB 8

// Initialisation des pin pour le LCD  http://www.arduino.cc/en/Tutorial/LiquidCrystalDisplay

const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);



  int MessageRPI = 0;                            // initialisation des données
  int MessageRead = 0;
  int Loop1 = 0;
  int Loop2 = 0;
  

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
  Serial.print("Start");    
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);  

   
/**************************************************************************************************************************************************
 * ************************************************ POUR ARDUINO MEGA*****************************************************************************

       // initialisation du timer5
  noInterrupts();                                 // désactivation des interruption pendant l'initialisation
  TCCR5A = 0;
  TCCR5B = 0;
  TCNT5  = 0;

  OCR5A = 62500;                                  // Registre de comparaison 62500 pour 1hz
  TCCR5B |= (1 << WGM52);                         // CTC mode
  TCCR5B |= (1 << CS52);                          // 256 prescaler 
  TIMSK5 |= (1 << OCIE5A);                        // active l'interruption par comparaison
  interrupts();                                   // fin de l'initialisation, réactivation de toutes les interruptions
  Serial.begin(9600);
  Serial.print("Start");                          // debug


****************************************************************************************************************************************************************************/
   // initialisation de l'écran LCD

  lcd.begin(16, 2);
  lcd.print("Bonjour");

  
  // Déclaration des entrée - sorties
pinMode(BUZZER, OUTPUT); 
//pinMode(DOORLOCKA, OUTPUT); 
//pinMode(DOORLOCKB, OUTPUT); 
}

/**************************************INTERRUPTION ARDUINO MEGA**********************************************************
ISR(TIMER5_COMPA_vect)                            // Fonction activée à chaque interruption du timer5 quand le timer atteint la valeur OCR5A
{ 
}
/***************************************************************************************************************************/


void loop() {
  // put your main code here, to run repeatedly:


 if (Serial.available() > 0) {                   // lecture du message serial venant du RPI
    MessageRead = 0;
    MessageRPI = Serial.read();
    Serial.print(MessageRPI);                  //debug

    if (MessageRPI == '1')
    {
      
      lcd.print("Le chien est devant la porte");    // activation de la porte pendant 10s
      Serial.print("Le chien est devant la porte");
      digitalWrite(LED_BUILTIN, HIGH);
      digitalWrite(DOORLOCKA, HIGH);
      digitalWrite(DOORLOCKB, HIGH);
      delay(10000);
       
      lcd.print("Le chien est rentré");             // Desactivation
      Serial.print("Le chien est rentré");
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(DOORLOCKA, LOW);
      digitalWrite(DOORLOCKB, LOW);
    }

      
    if (MessageRPI == '2')
    {
      Serial.print("Intrus Detecte");
      lcd.print("Intrus Detecte");
      digitalWrite(LED_BUILTIN, HIGH);
      tone(BUZZER, 3000);                           //Buzz pendant 5s à 3Khz
      delay(5000);
 
      Serial.print("Intrus Partis");                // désactivation
      lcd.print("Intrus partis");
      noTone(BUZZER);
      }
    }
}
