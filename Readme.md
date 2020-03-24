# SI - Group4 - Dog Flap
This file is set to resume the project.

## Presentation
### General idea of the project
The project consists in making a cat flap. The goal is to detect the presence of a specific cat so that the hatch unlocks and the cat can enter into the house.The detection of other animals would leave the hatch closed and an audible signal could be emitted. For example, if a foreign cat shows up at the door, a bark will be issued.
The part concerning artificial intelligence will be managed by the raspberry and the sound elements and displays will be managed by an Arduino

### Detailed description of the project

## Programmation
### First week (16/03/2020-22/03/2020)
Based on artificial intelligence courses and lessons learned on image and video processing, we have created a classic python program including :
-Creation of a python code which detect the face of cats on images
-Function to extract cat's face and result display added
-Imwrite added to save the image present in "face" in a new folder. This method is used to save an image to any storage device. It returns true if image is saved successfully and false If "face" is empty.
-Creation of a code which detect, extract and save the face of a cat via the webcam
(-Creation of the dataset, generation of cat faces photos due to the "save)
DELETION OF THE CATS DATASET (It will be replaced by a dogs dataset)

### Second week (23/03/2020-29/03/2020)
-We developt with the professor on the proposed subject to approve and improve the project.
-->Changes made: To meet a need that does not yet exist on the market, we will make a dog flap instead of a cat flap. The hatch will therefore be larger and the advantage of blocking the hatch will therefore be present to avoid the entry of small animals.
-Implementation of a second branch concerning the project functionalities (speakers, display, ...) which will be managed by an Arduino.