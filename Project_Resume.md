# SI - Group4 - Dog Flap
This file is set to resume the project.

## Presentation
### General idea of the project
<p align="justify">
The project consists in making a dog flap. The goal is to detect the presence of a specific dog so that the hatch unlocks and the dog can enter into the house.The detection of other animals would leave the hatch closed and an audible signal could be emitted. For example, if a foreign dog or a cat shows up at the door, a bark will be issued.
The part concerning artificial intelligence will be managed by the raspberry and the sound elements and displays will be managed by an Arduino.

### Detailed description of the project

## Programmation
### First step
<p align="justify">
Based on artificial intelligence courses and lessons learned on image and video processing, we have created a classic python program including :
  
-Creation of a python code which detect the face of cats on images

-Function to extract cat's face and result display added
<p align="justify">
-Imwrite added to save the image present in "face" in a new folder. This method is used to save an image to any storage device. It returns true if image is saved successfully and false If "face" is empty.

-Creation of a code which detect, extract and save the face of a cat via the webcam

(-Creation of the dataset, generation of cat faces photos due to the "save)
DELETION OF THE CATS DATASET (It will be replaced by a dogs dataset)

### Second step
-We developt with the professor on the proposed subject to approve and improve the project.
<p align="justify">
-->Changes made: To meet a need that does not yet exist on the market, we will make a dog flap instead of a cat flap. The hatch will therefore be larger and the advantage of blocking the hatch will therefore be present to avoid the entry of small animals.
<p align="justify">
-Implementation of a second branch concerning the project functionalities (speakers, display, ...) which will be managed by an Arduino. At the end of the project, it will be enough to merge this branch with the master branch to bring everything together on the same branch. The merge will go smoothly since no similar file concerning the functionalities will be present in the main branch.
<p align="justify">
-Implementation of a third branch concerning the recognition programs which will be managed by an Raspberry. The interest of creating this branch parallel to the master branch will reside during the merger. Indeed, the main branch is also dedicated to AI programs (detection, ...) and it is therefore possible that some files are common to both branches. Thus, during the merge, it will be easy to see the differences between the files in question and to choose the correct file to keep.

### Third step
<p align="justify">
In order to optimize our working time, we have divided the section on artificial intelligence in two. So, for a security question, we thought of realizing two algorithms which use different methods. This will allow us to choose one or the other depending on the performance of each algorithm or even to implement the two programs into one if necessary. Thus, Arnaud looked into the creation of a CNN and Florian focused on the various SVMs that exist. This work will therefore be done on the different branches of the Github as mentioned above. During each step, whatever the code in question, the codes will be shared and commented on through Teams meetings. Thus, everyone can follow the progress as they go along and help if necessary.
Meanwhile, the programming of the Arduino by Christophe can already take its course because it can work independently of the part on artificial intelligence.

#### Creation of the CNN code
<p align="justify">
-We have created a CNN to recognize dog breeds from scratch. We use a dataset of 10,000 dog images (with 120 dog breeds) which we have converted into tensors usable by CNN. We have reached an accuracy of 3.5% which is not acceptable at all. So, we looked for other solutions.
<p align="justify">
-We created a dog detector using ResNet50. It would help the algorithm to know if there is a dog in front of the camera. ResNet50 is a powerful CNN trained on more than a million images from the ImageNet database.
<p align="justify">
-We heard about transfer learning which allows us to achieve an efficient CNN despite a limited dataset. Thanks to this method we were able to achieve an accuracy of 76% on the dataset of which we spoke previously.
<p align="justify">
-Always using transfer learning, we have created a code to recognize the Cavalier King Charles (Florian's dog). We achieved 100% accuracy on the dataset with only 50 photos of Cavalier King Charles and 50 photos of random dog breeds.

#### Creation of the SVM code
<p align="justify">
-Creation of a SVM code which recognize dog breeds. For this, we get the same dataset that we used for the CNN code and it is easily found on the "Kaggle" site. The entire pre-processing step with a few adaptations has been taken from the CNN code in order to work with the same data. Thus, the images have been processed at the level of their size and their color which is grayscale. We have reached again an accuracy of +-4% which is not acceptable at all.
<p align="justify">
-So we made some slight changes to this code in terms of image processing so that they were in color. However, only a few percent were earned. Multiple kernels (linear, poly, rbf, ...) have also been tested to see if the algorithm is getting better but this has not been the case. We can therefore hypothesize that the dataset was not suitable for our codes.
<p align="justify">
-The previous code not being conclusive, we started on a different code with simple and precise bases to effectively predict if we are in the presence of a dog or a cat. For that, another database was taken over "Kaggle". After running the code, we had a + 
<p align="justify">
-70% reliable prediction. This is therefore more satisfactory. Thus , the program being functional, we made some modifications to improve these predictions such as adding images in the dataset, modifying the size of these, ... This basic program can thus be used to respond to the next step of our objectives namely the detection of dog breeds.
<p align="justify">
-For the rest, as we said earlier, we modified the code so that it can predict different breeds of dog. To do this, it was necessary to modify the dataset and thus allow it to contain several classes, several dog breeds. After discussion with the professor, it is interesting to add the fact that if a race is not recognized, the algorithm tells us that the species is "Unknown" since it is not easy to teach it all existing dog breeds. This was therefore directly implemented in the CNN code which turned out to be very conclusive. Thus, it is enough to add a database containing images  of all kinds and which is therefore associated with the "unknown" output.

#### Creation of the Arduino code
<p align="justify">
-For the development, an Arduino mega was used with the Arduino IDE, no external library is needed for the current project. First, the use of the relay controlling the door lock were implemented to unlock the door flap when a good dog is in front of it, the door lock is activated during 10 second to let the time to the dog to enter. Then the buzzer was added to repulse the cat in front of the door during 5s using the tone function. The pins needed to command the relays are seven and eight. The buzzer needs a PWM pin the pin 9 was chosen.
<p align="justify">
-Implementation of the LCD screen 2x16 using the LiquidCrystal library making the project a bit more user friendly. The LCD needs the pin 2,3,4,5,11,12 of the Arduino to work.
<p align="justify">
-Implementation of a timer interruption to remove the delays in the code, to not have the Arduino freezing the main loop because of the delays function. The interruption timer frequency is set every second.
<p align="justify">
-Implementation of an SPI communication to communicate between the Arduino and the computer before trying to make the raspberry PI and the Arduino communicate together.

### Fourth step
<p align="justify">
This last step consisted in implementing our codes on the Raspberry. Since the CNN code with transfer learning is promising, the SVM part has been put on hold and kept in reserve in the event that CNN encounters problems. This was precisely the case, but the programs were not at fault. We had a lot of errors when importing libraries. After testing several different environments, using PyCharm, Ubuntu, ... we reset the Raspberry and put Raspbian back there. Subsequently, we imported directly and one by one the different useful libraries. These were finally imported correctly and the following steps could be taken.
<p align="justify">
-We imported the CNN training code on the Raspberry, and we managed to train the model on the raspberry. Some lines have been added in the training code to display where we are (code progression).
<p align="justify">
-We then create a code that loads the weights of the previously trained model. To do this, we create an empty model (identical to the one used for training) and we load the weights of the neural network into it. This code makes it possible to check that the loaded model recognizes the requested breed.
<p align="justify">
-We then put all the codes together to make the final code. In this final code, we also integrated the Pi Camera instead of asking for the prediction of an image via the path of it (with a test code first and then in the final program). So, predictions will be made on the images captured by the camera. We also established the serial communication between the Raspberry and the Arduino. This makes it possible to detect a cat (because we also integrated the Haarcascade step for the cat detection carried out at the beginning of the project) or a dog, and in the second case, to say if it is a Cavalier King Charles or an unknown. The information is then transferred to the Arduino via a serial link (USB). The different actuators act in consequence according to the Arduino program.

So as we have just said, the Arduino code had to be adjusted to respond to the messages received and act accordingly.
<p align="justify">
-First we tried the communication between the raspberry pi and the arduino through the SPI, but there were problems using the USB port. The problems was resolved by enabling the SPI communication in setting from the Raspberry PI.
<p align="justify">
-Then, we change the type of data receiving in the arduino from string to integer and using the Serial Library from de raspberry PI to send data as utf-8 through the USB port.
<p align="justify">
-We change back to delay in the code, because the arduino UNO don’t have the same timer setup as the Arduino Mega, and the interruption is already use by the SPI connection.
<p align="justify">
-Some actuators have also been temporarily replaced because we preferred to avoid buying elements with the current situation. Thus, the locks were simply replaced by servo motors fitted with propellers which rotate 180 ° to block or not the door.
<p align="justify">
-The display of information on the LCD has also been updated. (Positioning of the text and erasing of it as things progress).
<p align="justify">
Thus the main objectives have been achieved and a video presenting the operation of the project can be found at this address.( https://henallux-my.sharepoint.com/:v:/g/personal/etu30602_henallux_be/ERCnO1R38XRGn95g62237gsBmnrNxXemxKM-laoPfos4Xg?e=497sv5 )
Improvements have already been thought of such as the addition of a second database to recognize a second breed of dogs or even the procedure to be implemented so that the Raspberry can learn for itself a new breed by assimilating it to a name. These points will be addressed in the final written report.
