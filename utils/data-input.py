import os
import shutil
import urllib

from urllib.parse import urljoin
from urllib import request
from utils.database import ProductManager
from utils.database import CategoryManager


IMAGE_BASE_URL = 'http://210.89.179.51:8080/files/data/images/'
# IMAGE_BASE_URL = 'http://210.89.179.51:8080/view/data/images/'


def read_data2db():
    with open('data_product', 'r', encoding='utf-8') as f:
        producat_dataset = f.readlines()

    with open('data_category', 'r', encoding='utf-8') as f:
        category_dataset = f.readlines()

    assert len(producat_dataset) > 0 and len(category_dataset) > 0

    category_db = CategoryManager()
    product_db = ProductManager()

    res = category_db.create_cate_table()
    if res != 0 and res != 1050:
        raise Exception

    res = product_db.create_product_table()
    if res != 0 and res != 1050:
        raise Exception

    for cate in category_dataset:
        category_seq = cate.split()[0]
        category_name = cate.split()[1]
        category_db.insert_category(category_seq=category_seq, category_name=category_name)

    for data in producat_dataset:
        data = data.strip()
        product_seq = data.split('\t')[0]
        category_seq = data.split('\t')[1]
        mall_seq = data.split('\t')[2]
        product_name = data.split('\t')[3]
        price = data.split('\t')[4]
        brand_name = data.split('\t')[5]
        maker_name = data.split('\t')[6]
        mall_category_name = data.split('\t')[7]
        img_url = ''
        for s in list(product_seq):
            img_url = img_url + s + '/'
        img = IMAGE_BASE_URL + img_url + 'img.jpg'
        product_db.insert_product(product_seq=product_seq, category_seq=category_seq, mall_seq=mall_seq, 
                                  product_name=product_name, price=price, brand_name=brand_name, maker_name=maker_name,
                                  mall_category_name=mall_category_name, img=img)
        

def maybe_download(train_download_path, test_download_path):
    with open('../data_/train.txt', 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    with open('../data_/test.txt', 'r', encoding='utf-8') as f:
        test_data = f.readlines()

    train_data_cnt = 0
    train_err_cnt = 0
    total_size = len(train_data)
    for idx, data in enumerate(train_data):
        product_seq = data.split(',')[2].strip()
        category_seq = data.split(',')[1].strip()

        img_ = ''
        for s in list(product_seq):
            img_ = img_ + s + '/'
        try:
            img_url = urljoin(IMAGE_BASE_URL, img_ + 'img.jpg')
            file_name = str(product_seq) + '.jpg'
            request.urlretrieve(img_url, file_name)
            if not os.path.exists(train_download_path + str(category_seq)):
                os.mkdir(train_download_path + str(category_seq))
            shutil.move(os.getcwd() + '/' + file_name, train_download_path + str(category_seq) + '/' + file_name)
            train_data_cnt += 1
        except Exception as exp:
            print(exp)
            train_err_cnt += 1
        print('{}% 완료'.format((idx/total_size) * 100))

    test_data_cnt = 0
    test_err_cnt = 0
    total_size = len(test_data)
    for idx, data in enumerate(test_data):
        product_seq = data.split(',')[2].strip()
        category_seq = data.split(',')[1].strip()

        img_ = ''
        for s in list(product_seq):
            img_ = img_ + s + '/'
        try:
            img_url = urljoin(IMAGE_BASE_URL, img_ + 'img.jpg')
            file_name = str(product_seq) + '.jpg'
            request.urlretrieve(img_url, file_name)
            if not os.path.exists(test_download_path + str(category_seq)):
                os.mkdir(test_download_path + str(category_seq))
            shutil.move(os.getcwd() + '/' + file_name, test_download_path + str(category_seq) + '/' + file_name)
            test_data_cnt += 1
        except Exception as exp:
            print(exp)
            test_err_cnt += 1
        print('{}% 완료'.format((idx/total_size) * 100))

if __name__ == '__main__':
    # read_data2db()
    maybe_download(train_download_path='../img/train/', test_download_path='../img/test/')