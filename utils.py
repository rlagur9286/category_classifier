import requests
import os
import re

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from itertools import count

GS_SHOP_PRODUCT_URL = 'http://www.gsshop.com/shop/sect/cateBestList.gs'
GS_SHOP_URL = 'http://www.gsshop.com/index.gs'

hangul = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z]+')


def do_crawl():
    with open('cate_id_list.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
    for idx, cate in enumerate(lines):
        sectid = cate.split('\t')[1].strip()
        print(sectid)
        get_popular_data(sectid=sectid)


def get_cate_id():
    html = requests.get(GS_SHOP_URL).text
    soup = BeautifulSoup(html, 'html.parser')
    cate_id_cnt = 0

    cate_group = soup.select('.category-group li .items a')
    with open('cate_id_list.txt', 'w', encoding='utf8') as f:
        for cate in cate_group:
            cate_href = cate.get('href')
            cate_name = cate.text
            matched = re.search(r"sectid=(\d+)&", cate_href)
            try:
                f.write(cate_name + '\t' + matched.group(1) + '\n')
                cate_id_cnt += 1
            except AttributeError:
                print(cate_href)
    print(cate_id_cnt)
    return cate_id_cnt


def get_popular_data(sectid):
    ua = UserAgent()
    headers = {
        'User-Agent': ua.ie,
        'Referer': 'http://www.gsshop.com/shop/sect/sectL.gs?sectid=1378773'
    }
    file_path = os.path.join(os.getcwd(), 'gs_data', str(sectid) + '.txt')
    if os.path.exists(file_path):
        return
    file = open(file_path, 'w', encoding='utf8')

    prev = None

    for page in count(1):
        params = {
            'sectid': sectid,
            'msectid': '',
            'orderBy': 'popular',
            'pageIndex': page,
            'lseq': 403227,
        }
        html = requests.post(GS_SHOP_PRODUCT_URL, headers=headers, params=params).text
        soup = BeautifulSoup(html, 'html.parser')
        data_list = soup.select('ul li')
        if len(data_list) == 0:
            print(prev, params)
            break

        for idx, data in enumerate(data_list):
            prd_img = data.select('.prd-img img')[0]
            img_src = prd_img.get('src')

            prd_info = data.select('.prd-info .prd-name')[0].text
            prd_info = " ".join(prd_info.split())
            file.write(img_src + '\t' + prd_info + '\n')
        prev = data_list[0]
        if (page % 50) == 0:
            print(params)
    file.close()

if __name__ == '__main__':
    get_cate_id()
    # get_popular_data(1378773)
    # do_crawl()