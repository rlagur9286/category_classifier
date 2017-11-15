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


class ProductManager(DBManager):
    def __init__(self):
        DBManager.__init__(self)

    def create_cate_table(self):
        assert self.connected  # Connection Check Flag
        query_for_create = "CREATE TABLE cate_tb (" \
                                 "cate_id INTEGER(11) PRIMARY KEY, " \
                                 "cate_name VARCHAR (255) NOT NULL" \
                                 ")"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                res = cur.execute(query_for_create)
                return res

        except Exception as exp:
            num, error_msg = exp.args
            if num != 1050:
                logger.error(">>>MYSQL WARNING<<<")
                logger.error("At create_cate_table()")

                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
            return num

    def create_product_table(self):
        assert self.connected  # Connection Check Flag
        query_for_create = "CREATE TABLE product_tb (" \
                                 "product_id INTEGER(11) PRIMARY KEY, " \
                                 "product_name VARCHAR (255) NOT NULL," \
                                 "product_cate VARCHAR (255) NOT NULL," \
                                 "product_img VARCHAR (255) NOT NULL" \
                                 ")"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                res = cur.execute(query_for_create)
                return res

        except Exception as exp:
            num, error_msg = exp.args
            if num != 1050:
                logger.error(">>>MYSQL WARNING<<<")
                logger.error("At create_product_table()")

                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
            return num

    def insert_category(self, cate_id, cate_name):  # for retrieving Account
        assert self.connected  # Connection Check Flag
        query_insert_category = "INSERT INTO cate_tb " \
                                "(cate_id, cate_name) " \
                                "VALUES (%s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_insert_category, (cate_id, cate_name))
                return self.conn.commit()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At insert_category()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def insert_product(self, product_id, product_name, product_cate, product_img):  # for retrieving Account
        assert self.connected  # Connection Check Flag
        query_insert_category = "INSERT INTO product_tb " \
                                "(product_id, product_name,product_cate, product_img) " \
                                "VALUES (%s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_insert_category, (product_id, product_name,product_cate, product_img))
                return self.conn.commit()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At insert_product()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def retrieve_cate_all(self):  # for retrieving Account
        assert self.connected  # Connection Check Flag
        query_for_retrieve_url = "SELECT * FROM cate_tb"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_url)
                return cur.fetchall()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_cate_all()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg
