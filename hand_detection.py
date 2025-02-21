# ********************************GROUP MEMBERS
# *******************ROLL NO.      NAMES
# *******************BIT-21027  ASJAD BABAR
# *******************BIT-21017  ALI HASSAN
# *******************BIT-21032  SHEHARYAR SHAFIQUE
# *******************BIT-21020  USMAN HUNJRA
#***************
# Importing The Required Libraries
import cv2
import mediapipe as mp
# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict

# Initializing the  mpHands Model 
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# captures video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read Captured video frame by frame
    success, img = cap.read()
    
    # Check if the frame was successfully read
    if not success:
        print("Failed to grab frame")
        break
    
    # Flip the image(frame)
    img = cv2.flip(img, 1)
    
    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image
    results = hands.process(imgRGB)
    
    # If hands are present in image(frame)
    if results.multi_hand_landmarks:
        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)
        
        # If any hand present
        else:
            for i in results.multi_handedness:
                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']
                
                if label == 'Left':
                    # Display 'Left Hand' on left side of window
                    cv2.putText(img, label + ' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 100, 205), 2)
                
                if label == 'Right':
                    # Display 'Right Hand' on right side of window
                    cv2.putText(img, label + ' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
    
    # Display the video feed
    cv2.imshow('Hand Detection', img)
    
    # Save the last frame when 's' is pressed, or exit with 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('s'):  # Press 's' to save the image
        cv2.imwrite('output.jpg', img)
        print("Image saved as output.jpg")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
