import requests
import os
import re
import time
import shutil
import tqdm

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from itertools import count
from utils.database import ProductManager
from urllib import request

GS_SHOP_PRODUCT_URL = 'http://www.gsshop.com/shop/sect/cateBestList.gs'
GS_SHOP_URL = 'http://www.gsshop.com/index.gs'

hangul = re.compile('[^ ㄱ-ㅣ가-힣a-zA-Z]+')


def do_crawl():
    product_db = ProductManager()
    cate_list = product_db.retrieve_cate_all()
    inserted_cate_list = [cate.get('product_cate') for cate in product_db.retrieve_inserted_cate_list()]

    for cate in cate_list:
        sectid = cate.get('cate_id')
        if str(sectid) not in inserted_cate_list:
            get_category_data(sectid=sectid)


def get_cate_id():
    product_db = ProductManager()
    html = requests.get(GS_SHOP_URL).text
    soup = BeautifulSoup(html, 'html.parser')
    cate_id_cnt = 0

    # DB 카테고리 테이블 생성
    res = product_db.create_cate_table()
    if res != 0 and res != 1050:
        raise Exception

    res = product_db.create_product_table()
    if res != 0 and res != 1050:
        raise Exception

    cate_group = soup.select('.category-group li .items a')
    for cate in cate_group:
        cate_href = cate.get('href')
        cate_name = cate.text
        matched = re.search(r"sectid=(\d+)&", cate_href)
        try:
            # 카테고리 id, 카테고리 이름 db에 저장
            res = product_db.insert_category(cate_id=matched.group(1), cate_name=cate_name)
            if not res:
                cate_id_cnt += 1
        except AttributeError:
            print(cate_href)
    print(cate_id_cnt, "개의 카테고리가 추가되었습니다.")
    return cate_id_cnt


def get_category_data(sectid):
    product_db = ProductManager()
    ua = UserAgent()
    headers = {
        'User-Agent': ua.ie,
        'Referer': 'http://www.gsshop.com/shop/sect/sectL.gs?sectid=1378773'
    }

    prev = None
    prd_cnt = 0
    for page in count(1):
        if prd_cnt > 10000:
            break
        params = {
            'sectid': sectid,
            'msectid': '',
            'orderBy': 'popular',
            'pageIndex': page,
            'lseq': 403227,
        }
        try:
            html = requests.post(GS_SHOP_PRODUCT_URL, headers=headers, params=params).text
        except Exception as e:
            print(e)
            time.sleep(10)
            html = requests.post(GS_SHOP_PRODUCT_URL, headers=headers, params=params).text
        soup = BeautifulSoup(html, 'html.parser')
        data_list = soup.select('ul li')
        if len(data_list) == 0:
            print(prev, params)
            break

        for idx, data in enumerate(data_list):
            prd_img = data.select('.prd-img img')[0]
            img_src = prd_img.get('src')
            matched = re.search(r"/(\d+)_O1", img_src)
            if matched:
                prd_info = data.select('.prd-info .prd-name')[0].text
                product_name = " ".join(prd_info.split())
                product_db.insert_product(product_id=matched.group(1), product_name=product_name, product_cate=sectid, product_img=img_src)
                prd_cnt += 1
        prev = data_list[0]
        if (page % 50) == 0:
            print(params)


def maybe_download(download_path):
    product_db = ProductManager()
    cate_list = [cate.get('product_cate') for cate in product_db.retrieve_inserted_cate_list()]
    total_size = len(cate_list)
    total_cnt = 0
    err_cnt = 0
    for i, cate in enumerate(cate_list):
        product_list = [(prod.get('product_id'), prod.get('product_img')) for prod in product_db.retrieve_products_by_cate(cate)]

        for prod in product_list:
            try:
                file_name = str(prod[0]) + '.jpg'
                url = 'http:' + str(prod[1])
                request.urlretrieve(url, file_name)
                if not os.path.exists(download_path + str(cate)):
                    os.mkdir(download_path + str(cate))
                shutil.move(os.getcwd() + '/' + file_name, download_path + str(cate) + '/' + file_name)
                total_cnt += 1
            except Exception as exp:
                print(exp)
                err_cnt += 1
        print('{}% 완료'.format((i/total_size) * 100))
if __name__ == '__main__':
    # get_cate_id()
    # get_category_data(1378773)
    # do_crawl()
    maybe_download(download_path='../data/')