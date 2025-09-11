# Student's-t Distribution for Word Segmentation
Python implementation of the 2016 [paper](https://users.iit.demokritos.gr/~bgat/DAS2016_sfikas.pdf) by Louloudis et. al. with an expected words feature to solve the transcript mapping problem efficiently.

<img width="1682" height="153" alt="image" src="https://github.com/user-attachments/assets/879a9c68-3ba9-4e80-b7b4-f522a0e507f3" />
<img width="4470" height="1631" alt="image" src="https://github.com/user-attachments/assets/59e0f4fe-de7e-4929-9cc7-ffca212df0c5" />

# Usage
After installing the required dependencies, the code can be executed in the command line as follows:
```bash
python segment_words.py --img_path "/path/to/image.png" --expected_words INT
```
where:
- `--img_path`         | The path to a text line you want to segment.
- `--expected_words`   | The number of words you expect in the given line.  

# Structure
- `segment_words.py`   | Main word segmentation pipeline.            
- `preprocessing.py`   | Image cleaning and de-slanting.           
-  `distances.py`      | Component gap measurement.                  
-  `student_t.py`      | Mixture model for gap classification.       
