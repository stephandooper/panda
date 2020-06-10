# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:42:48 2020

@author: Stephan

https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.primary_site%22%2C%22value%22%3A%5B%22prostate%20gland%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.access%22%2C%22value%22%3A%5B%22open%22%5D%7D%7D%5D%7D
only choose bcr xml files
Only choose diagnostic slides, not tissue slides
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm



def to_isup(gl_score):
    
    if gl_score == '3+3':
        isup = 1
    elif gl_score == '3+4':
        isup = 2
    elif gl_score == '4+3':
        isup = 3
    elif gl_score == '4+4':
        isup = 4
    elif gl_score == '3+5':
        isup = 4
    elif gl_score == '5+3':
        isup = 4
    elif gl_score == '4+5':
        isup = 5
    elif gl_score == '5+4':
        isup = 5
    elif gl_score == '5+5':
        isup = 5
    else:
        isup = 'None'
    return isup

def parse_file(path):
    tree= ET.parse(path)
    root = tree.getroot()
    
    name = root.findall('.//{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_barcode')[0].text
    
    try:
        #gleason_agg = root.findall('.//{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}gleason_score')[0].text
        gleason_primary = root.findall('.//{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}primary_pattern')[0].text
        gleason_secondary = root.findall('.//{http://tcga.nci/bcr/xml/clinical/shared/stage/2.7}secondary_pattern')[0].text
        gleason_score = gleason_primary + '+' + gleason_secondary
    except IndexError:
        # Gleason scores are not available
        #gleason_agg = 'None'
        gleason_primary = 'None'
        gleason_secondary = 'None'
        gleason_score = 'None'
    
    isup = to_isup(gleason_score)
    return name, gleason_score, isup



if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Path to TCGA xml files for diagnostic slides, should contain directories with xml files')

    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/images/dataset/",
                        help='Directory to xml files')
    parser.add_argument('--data_provider', required=False,
                        metavar="Data provider",
                        help='Data provider for the dataset xml files',
                        default='None')

    args = parser.parse_args()
    path = Path(args.dataset)
    
    
    ids = []
    gleason_scores = []
    isup = []     
    
    
    for p in tqdm(path.rglob('*.xml')):
        results = parse_file(p)
        ids.append(results[0])
        gleason_scores.append(results[1])
        isup.append(results[2])
    
    results = pd.DataFrame({'image_id': ids,
                            'gleason_score':gleason_scores,
                            'isup': isup})
    
    # 499 isup grades out of 1000, rest None
    results['data_provider'] = args.data_provider
    
    results.to_csv('external_data.csv', index=False)
    
    


