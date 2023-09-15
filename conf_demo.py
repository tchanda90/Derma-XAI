import random
import os
import numpy as np
import pandas as pd
import torch


char_labels_full = ['thick reticular or branched lines', 'black dots or globules in the periphery of the lesion', 
                    'white lines or white structureless area', 'eccentrically located structureless area', 'grey patterns',
                    'polymorphous vessels', 'pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference',
                    'asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria', 'melanoma simulator', 
                    'only one pattern and only one colour']

translations = {'thick reticular or branched lines': 'Dicke retikuläre oder verzweigte Linien', 
                'black dots or globules in the periphery of the lesion': 'Schwarze Punkte oder Schollen in der Läsionsperipherie', 
                'white lines or white structureless area': 'Weiße Linien oder weißes strukturloses Areal', 
                'eccentrically located structureless area': 'Exzentrisch gelegenes, strukturloses Areal jeglicher Farbe, außer hautfarben, weiß und grau', 
                'grey patterns': 'Graue Muster',
                'polymorphous vessels': 'Polymorphe Gefäße', 
                'pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference':
                    'Pseudopodien oder radiale Linien am Läsionsrand, die nicht den gesamten Läsionsumfang einnehmen',
                'asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria':
                    'Asymmetrische Kombination mehrerer Muster oder Farben ohne weitere Melanomkriterien', 
                'melanoma simulator': 'Melanomsimulator', 
                'only one pattern and only one colour': 'Nur ein Muster und nur eine Farbe'}

thresholds = torch.tensor([ 0.2581, -0.4436, -0.1006,  0.3087,  0.8408, -0.3522, -0.9660, -0.4659,
        -0.2477, -2.1751])
thresholds_sigmoid = torch.tensor([0.5642, 0.3909, 0.4749, 0.5766, 0.6986, 0.4129, 0.2757, 0.3856, 0.4384,
        0.1020])

temperature_dict = {'thick reticular or branched lines': 1.3761733770370483,
 'black dots or globules in the periphery of the lesion': 1.3037667274475098,
 'white lines or white structureless area': 1.2326982021331787,
 'eccentrically located structureless area': 1.1476147174835205,
 'grey patterns': 1.4328160285949707,
 'polymorphous vessels': 1.2088375091552734,
 'pseudopods or radial lines at the lesion margin that do not occupy the entire lesional circumference': 1.4442716836929321,
 'asymmetric combination of multiple patterns or colours in the absence of other melanoma criteria': 1.647501826286316,
 'melanoma simulator': 1.4534757137298584,
 'only one pattern and only one colour': 1.4301159381866455}

sigmoid_thresholds_dict = {}
for i in range(len(thresholds_sigmoid)):
    sigmoid_thresholds_dict[char_labels_full[i]] = thresholds_sigmoid[i].item()
    
    