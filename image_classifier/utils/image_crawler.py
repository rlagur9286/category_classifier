import os
import csv
import shutil
# import cv2
import urllib.request
from project.engine.utils.database import ImageManager

ALLOWED_EXTEND = ['.jpg', '.JPG', '.png', '.PNG', '.gif', '.GIF', '.jpeg']

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def txt2list(file_path=None):
    url_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            url = f.readline()
            if not url:
                break
            url_list.append(url)

    return url_list


def download_image_and_save(file_path=None, download_path=None):
    image_db = ImageManager()
    total_cnt = 0
    err_cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            product_info = f.readline()
            if not product_info:
                break
            PRODUCT_CD = product_info.split('\t')[0].strip()
            GOODS_NAME = product_info.split('\t')[2].strip()
            GOODS_IMAGE_URL = product_info.split('\t')[3].strip()
            BRAND = product_info.split('\t')[4].strip()
            MODEL = product_info.split('\t')[5].strip()
            CATEGORY = product_info.split('\t')[8].strip()
            PRICE = product_info.split('\t')[14].strip()
            result = image_db.insert_product2db(PRODUCT_CD, GOODS_NAME, GOODS_IMAGE_URL, BRAND, MODEL, CATEGORY,
                                                float(PRICE.strip()))

            if result is True:
                try:
                    file_name = PRODUCT_CD + '.jpg'
                    url = str(GOODS_IMAGE_URL)
                    urllib.request.urlretrieve(url, file_name)
                    if not os.path.exists(download_path + CATEGORY):
                        os.mkdir(download_path + CATEGORY)
                    shutil.move(os.getcwd() + '/' + file_name, download_path + CATEGORY + '/' + file_name)
                    total_cnt += 1
                except Exception as exp:
                    print(exp)
                    err_cnt += 1
    print('Total download image cnt is ', total_cnt, 'with error cnt ', err_cnt)


def retrieve_image_from_db(file_path=None):
    image_db = ImageManager()
    results = image_db.retrieve_info_all()
    url_list = []
    if results is not []:
        for result in results:
            url_list.append(result.get('PRODUCT_CD'))

    if len(url_list) != 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            for url in url_list:
                f.write(url + '\n')
    print(len(url_list), ' Product retrieved')


def insert_info2db(file_path=None):
    image_db = ImageManager()

    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            product_info = f.readline()
            if not product_info:
                break
            PRODUCT_CD = product_info.split('\t')[0].strip()
            GOODS_NAME = product_info.split('\t')[2].strip()
            GOODS_IMAGE_URL = product_info.split('\t')[3].strip()
            BRAND = product_info.split('\t')[4].strip()
            MODEL = product_info.split('\t')[5].strip()
            PRICE = product_info.split('\t')[14].strip()
            CATEGORY = product_info.split('\t')[8].strip()
            result = image_db.insert_product2db(PRODUCT_CD, GOODS_NAME, GOODS_IMAGE_URL, BRAND, MODEL, CATEGORY,
                                                float(PRICE.strip()))
            if result is True:
                cnt += 1
        print(cnt, ' Product Info Inserted')


def delete_info_from_db(file_path=None):
    image_db = ImageManager()

    cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            product_info = f.readline()
            if not product_info:
                break
            result = image_db.delete_info_from_db(PRODUCT_CD=product_info.strip())
            if result is True:
                cnt += 1
        print(cnt, ' Product are Deleted')


def retrieve_image_from_csv(csv_path=None, file_path=None):
    url_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            url_list.append(str(row[0]))

    if len(url_list) != 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            for url in url_list:
                f.write(url + '\n')


def crop_face(path):
    for (path, dir, files) in os.walk(path):
        #if dir:
        #    continue
        if '_cut' in path:
            continue
        if not os.path.exists('../test_images/' + path + '_cut'):
            os.makedirs('../test_images/' + path + '_cut')

        print(path, dir, files)

        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in ALLOWED_EXTEND:
                img = cv2.imread(path + '/' + filename)

                print(img)

                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                except:
                    continue
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.5,
                    minNeighbors=3,
                    minSize=(30, 30)
                )

                # print('found {0} faces'.format(len(faces)))
                if len(faces) == 0:
                    continue

                for idx, face in enumerate(faces):
                    roi_gray = img[face[1] - 30:face[1] + face[3] + 30,
                               face[0] - 30:face[0] + face[2] + 30]
                    cv2.imwrite('../test_images/' + path + '_cut/' + str(idx) + '_' + filename , roi_gray)


if __name__ == '__main__':
    """ ####################################################################

    retrieve_image_from_csv(csv_path='woman_product.csv', file_path='woman_url.txt')

    get_file_list(dir_path='../download_images/', output='file_list.txt')   # Get download_images list from folder

    retrieve_image_from_db(file_path='woman_db_list.txt')   # Get DB item list

    delete_info_from_db(file_path='woman_db_list.txt')  # Delete product form DB
    #####################################################################"""

    download_image_and_save(file_path='part.txt', download_path='../image_all/')  # Download and save img to db
    # crop_face('../test_images')
