# 2 Dimensional Art
## How to Create Digital Art Using a Single Reference Image

**IMPORTANT:** The "results" folder generated during the execution of the program will overwrite any previous results still stored in the project folder. To save the results, rename the folder or move it to a different directory.

1. Navigate to the genetic-art/src/2D directory

2. Select one of the included reference images provided in the project file that you would like to use and rename it to "reference1.png". If you would like to use your own reference image, add your desired reference image to the project folder and rename it to "reference1.png". *The reference image you use must be of type PNG.*

3. At this point the python file is ready to be executed using the default Control Variables. If you would like to use the default Control Variable skip to Step 6.

4. To change the default Control Variables open the python file **"genArt.py"**.

5. Change the values under the Control Variables to your desired settings and save your changes.

6. Using your terminal, navigate to the project folder.

7. Run the command "*python3 genArt.py*" in your terminal.

8. The terminal will display the current generation and the current accuracy to the reference image. To view the results, open the newly created "results" folder in the project folder. The images will be saved as *[Current Generation].png*. The log file *log.txt* in the "results" folder contains comma separated data of the best Organism each generation.

9. The program will stop executing once it has reached the maximum generations. To stop the program before press *Control + C* on your keyboard in the terminal window.


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
