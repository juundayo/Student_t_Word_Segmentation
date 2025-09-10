# Student's-t Distribution for Word Segmentation
Python implementation of the 2016 [paper](https://users.iit.demokritos.gr/~bgat/DAS2016_sfikas.pdf) by Louloudis et. al. with an expected words feature to solve the transcript mapping problem efficiently.

<img width="1682" height="153" alt="image" src="https://github.com/user-attachments/assets/879a9c68-3ba9-4e80-b7b4-f522a0e507f3" />
<img width="4470" height="1631" alt="image" src="https://github.com/user-attachments/assets/a1e6b1bb-94c0-4edc-9282-d5630ea53658" />


# Structure
- `segment_words.py`   | Main word segmentation pipeline.            
- `preprocessing.py`   | Image cleaning and de-slanting.           
-  `distances.py`      | Component gap measurement.                  
-  `student_t.py`      | Mixture model for gap classification.       
