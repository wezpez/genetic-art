# 3 Dimensional Art
## How to Create Digital Art Using Two Reference Images

**IMPORTANT:** The "results" folder generated during the execution of the program will overwrite any previous results still stored in the project folder. To save the results, rename the folder or move it to a different directory.

1. Navigate to the genetic-art/src/3D directory

2. Select two of the included reference images provided in the project media files that you would like to use and rename them to "reference1.png" and "reference2.png". If you would like to use your own reference images, add your desired reference image to the project folder and rename them to "reference1.png" and "reference2.png". *The reference images you use must be of type PNG and be equally sized.*

3. At this point the python file is ready to be executed using the default Control Variables. If you would like to use the default Control Variable skip to Step 6.

4. To change the default Control Variables open the python file **"genArt3D.py"**.

5. Change the values under the Control Variables to your desired settings and save your changes.

6. Using your terminal, navigate to the project folder.

7. Run the command "*python3 genArt3D.py*" in your terminal.

8. A browser window will pop up and display the 3D Scene.

9. You may need to slightly zoom out the camera in order to view the whole digital sculpture. The instructions next to the 3D scene explain how to manipulate the camera.

10. The terminal will also display the current generation and the current accuracies to the reference images. To view the saved results of the digital anamorphic sculptures seen from each of the perspectives, open the newly created "results" folder in the project folder. The images will be saved as *P[x]\_[Current Generation].png* where P indicates which perspective the image is viewing the sculpture from. The log file *log.txt* in the "results" folder contains comma separated data of the best Organism each generation.

11. The program will pause once it has reached the maximum generations. To unpause use the "run/pause" button provided. To stop the program completely, simply close the browser window or press *Control + C* on your keyboard in the terminal window.


## Project Requirements:

### Python Version Required:
- Python 3.8.1 (or higher)

### Libraries Required:
- PIL (Python Image Library)
- Vpython
- numpy
- multiprocessing
- random
- os
- datetime
- copy

Use pip3 to install the required libraries.
