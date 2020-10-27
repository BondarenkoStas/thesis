import requests
import re
from pprint import pprint

def _get_only_digits(string):
    return re.sub('[^0-9]', '', string)

def get_budget(content):
    text = re.search(r'<h4 class="inline">Budget:</h4>(.*?)<', content)
    return  _get_only_digits(text.group(1)) if text else None

def get_gross(content):
    text = re.search(r'<h4 class="inline">Cumulative Worldwide Gross:</h4>(.*?)<', content)
    return _get_only_digits(text.group(1)) if text else None

def increment_id(id_str):
    return str(int(id_str) + 1).zfill(len(id_str))

movie_list = []
movie_id = '0007286456'
for i in range(100):
    response = requests.get(f'https://www.imdb.com/title/tt{movie_id}/')
    if response.ok:
        content = response.text.replace('\n', '').replace('\t', '')
        budget = get_budget(content)
        gross = get_gross(content)
        if budget or gross:
            movie_list.append({
                'id': movie_id,
                'budget': budget,
                'gross': gross
            })
        movie_id = increment_id(movie_id)

pprint(movie_list)