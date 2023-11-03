# IR Spectra Functional Group Predictor using CNN's

Can predict functional groups from IR spectra images using CNN. This project includes all the necessary code for image processing, such as image manipulation, pixel mapping and interpolation. Additionally, it features a functional GUI for ease of use. It was trained on around 40,000 IR images.

**Note:** The training data and models are not included in this repository due to data and licensing restrictions.

## GUI


<div style="display: flex;">
  <img src="Figure/GUI black.png" alt="Image 1" width="320" />
  <img src="Figure/GUI red.png" alt="Image 2" width="320" />
</div>
The GUI is built using Tkinter and PIL. It features a simple interface for users to upload their IR spectra images and predict the functional groups. The GUI also allows users to select particular functional groups and gives a table of probabilities for each group.


## Installation
```bash
# Clone the repository
git clone https://github.com/lewmas1/IR-Image-to-FuncGroups.git

# Change directory to the project folder
cd ir-spectra-predictor

# Install dependencies (if applicable)
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Authors
lewmas1
