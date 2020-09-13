import os
import pandas as pd
import numpy as np
import sys
import csv
from time import strptime
import json
from datetime import datetime as dt
import re

csv.field_size_limit(sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_columns = None
pd.options.display.max_rows = None

def grab_field_from_array_of_objects(array, field):
    return [item[field] for item in array]

def sorted_string_from_array(array):
    return ','.join(list(np.sort(np.array(array))))

def parse_field_from_str_json_array(df, col_name, field):
    list_of_items = [read_as_json(c) for c in df[col_name]]
    list_of_fields = [grab_field_from_array_of_objects(item, field) for item in list_of_items]
    return [sorted_string_from_array(item) for item in list_of_fields]

def read_as_json2(string):
#   get rid of not ascii characters
    s = string.encode("ascii", errors="ignore").decode()
#   just to get rid of those character which are malformed in many cases
    s = re.sub(r'(character.*?)(credit_id)', r'\2', s)
#   replace tabs, new lines, trailing commas, and None, lowercase
    s = s.replace('\t','').replace('\\t', '').replace('\n','').replace(',}','}').replace(',]',']').replace('None', '\'\'').lower()
    allowed_symbols = '[a-z0-9 ,_\/\.\:\(\)\&-\[\]\!\?\#\$]'
#   to deal with enclosed double quotes
    t = f'(?:(?!\',|\':|, \'|: \'|\'\}}){allowed_symbols})*'
    s = re.sub(fr'\'({t})"({t})"({t})\'', r'"\1\2\3"', s).replace("\'", "'")
#   replace all the string wrapped in '' which don't contain "'," or "':" or "'}" (basically end of string in json)
    s = re.sub(fr'( |\{{)\'({t})\'(,|:|\}})', r'\1"\2"\3', s)
#   wrap digits into "" as well
    try:
        return json.loads(s)
    except:
        pass
                        
                        
def grab_fields_from_array_of_objects(array, fields):
    new_array = []
    for item in array:
        new_item = {}
        for field in fields:
            new_item[field] = item[field]
        new_array.append(new_item)
    return new_array

def parse_fields_from_str_json_array(df, col_name, fields):
    list_of_items = [read_as_json2(c) for c in df[col_name]]
    return [grab_fields_from_array_of_objects(item, fields) for item in list_of_items]