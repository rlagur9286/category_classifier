import logging
import pymysql
import timeit

from .config import *

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class DBManager(object):
    def __init__(self):
        try:
            self.conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, passwd=DB_PWD,
                                        db=DB_NAME, charset='utf8', use_unicode=True)
            self.connected = True
        except pymysql.Error:
            self.connected = False


class CategoryManager(DBManager):
    def __init__(self):
        DBManager.__init__(self)

    def create_cate_table(self):
        assert self.connected  # Connection Check Flag
        query_for_create = "CREATE TABLE data_category (" \
                                 "category_seq INTEGER(11) PRIMARY KEY, " \
                                 "category_name VARCHAR (255) NOT NULL" \
                                 ")"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                res = cur.execute(query_for_create)
                return res

        except Exception as exp:
            num, error_msg = exp.args
            if num != 1050:
                logger.error(">>>MYSQL WARNING<<<")
                logger.error("At create_data_category_table()")

                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
            return num

    def insert_category(self, category_seq, category_name):  # for retrieving Account
        assert self.connected  # Connection Check Flag
        query_insert_category = "INSERT INTO data_category " \
                                "(category_seq, category_name) " \
                                "VALUES (%s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_insert_category, (category_seq, category_name))
                return self.conn.commit()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At insert_category()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def retrieve_cate_list(self):  # for retrieving Product
        assert self.connected  # Connection Check Flag
        query_for_retrieve = "SELECT category_seq FROM data_category"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve)
                return cur.fetchall()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_cate_list()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def retrieve_cate_name_by_cate_id(self, category_seq=None):  # for retrieving Product
        assert self.connected  # Connection Check Flag
        query_for_retrieve = "SELECT category_name FROM data_category WHERE category_seq=%s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve, category_seq)
                return cur.fetchall()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_cate_name_by_cate_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg


class ProductManager(DBManager):
    def __init__(self):
        DBManager.__init__(self)

    def create_product_table(self):
        assert self.connected  # Connection Check Flag
        query_for_create = "CREATE TABLE data_product (" \
                                 "product_seq INTEGER(11) PRIMARY KEY, " \
                                 "category_seq INTEGER (11) NOT NULL," \
                                 "mall_seq INTEGER (11) NOT NULL," \
                                 "product_name VARCHAR (255) NOT NULL," \
                                 "price INTEGER (11) NOT NULL," \
                                 "brand_name VARCHAR (255)," \
                                 "maker_name VARCHAR (255)," \
                                 "mall_category_name VARCHAR (255)," \
                                 "img VARCHAR (255)" \
                                 ")"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                res = cur.execute(query_for_create)
                return res

        except Exception as exp:
            num, error_msg = exp.args
            if num != 1050:
                logger.error(">>>MYSQL WARNING<<<")
                logger.error("At create_data_product_table()")

                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
            return num

    def insert_product(self, product_seq, category_seq, mall_seq, product_name, price, brand_name, maker_name,
                       mall_category_name, img):  # for retrieving Account
        assert self.connected  # Connection Check Flag
        query_insert_product = "INSERT INTO data_product " \
                                "(product_seq, category_seq, mall_seq, product_name, price, brand_name, maker_name, mall_category_name, img) " \
                                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_insert_product, (product_seq, category_seq, mall_seq, product_name, price, brand_name, maker_name, mall_category_name, img))
                return self.conn.commit()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At insert_product()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def retrieve_products_by_cate(self, cate_id=None, limit=None):  # for retrieving Product
        assert self.connected  # Connection Check Flag
        if not cate_id:
            query_for_retrieve = "SELECT product_seq, product_name, category_seq FROM data_product ORDER BY rand() LIMIT 100000"
        else:
            query_for_retrieve = "SELECT product_seq, product_name, category_seq FROM data_product WHERE category_seq=(%s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve, cate_id)
                return cur.fetchall()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_products_by_cate()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

