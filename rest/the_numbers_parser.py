# import requests
# import re
# from pprint import pprint

# def _get_only_digits(string):
#     return re.sub('[^0-9]', '', string)

# def get_table(content):
#     text = re.search(r'<table>(.*?)</table>', content)
#     return text.group(1) if text else None

# def get_table_data_rows(table):
#     return re.findall(r'<tr>(.*?)</tr>', table)

# def increment_id(id_str, increment):
#     return str(int(id_str) + increment).zfill(len(id_str))


# url = 'https://www.the-numbers.com/movie/budgets/all'
# response = requests.get(url)
# content = response.text.replace('\n', '').replace('\t', '')
# print(get_table_data_rows(get_table(content)))

# # movies_pages = []
# # page_id = '001'
# # last_page = 5901
# # for while str(page_id) < last_page:
# #     response = requests.get(f'https://www.imdb.com/title/tt{movie_id}/')
# #     if response.ok:
# #         content = response.text.replace('\n', '').replace('\t', '')
# #         budget = get_budget(content)
# #         gross = get_gross(content)
# #         if budget or gross:
# #             movie_list.append({
# #                 'id': movie_id,
# #                 'budget': budget,
# #                 'gross': gross
# #             })
# #         movie_id = increment_id(movie_id)

# # pprint(movie_list)

import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://www.the-numbers.com/movie/budgets/all'
html = requests.get(url).text

soup = BeautifulSoup(html, 'lxml')
table_html = soup.find_all('table')[0]

new_table = pd.DataFrame(columns=range(0,2), index = [0]) # I know the size 

row_marker = 0
for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
        new_table.iat[row_marker,column_marker] = column.get_text()
        column_marker += 1

new_table
