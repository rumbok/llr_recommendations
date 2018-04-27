import logging
from operator import add, itemgetter

from .llr_math import llr_sqrt
from pyspark import RDD
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLR:
    """
    Система на основе Co-/cross- occurence матриц используя LLR (Lig-Likelihood ratio) метрику
    """
    @staticmethod
    def calc_llr(ss: SparkSession, from_rdd: RDD, to_rdd: RDD) -> RDD:
        """
        Расчет матрицы co-/cross- occurence LLR для списка списков объектов для рекомендаций A -> B
        Вначале вычисляется произведение матриц left_rdd.Transpose * right_rdd.
        Затем LogLikelihoodRatio для каждой ячейки.
        :param ss: Spark Session объект
        :param from_rdd: rdd с разреженной матрицей со статистикой по объектам A. Формат значений - (ID списка, ID объекта A)
        :param to_rdd: rdd с разреженной матрицей со статистикой по объектам B. Формат значений - (ID списка, ID объекта B)
        :return: Разреженная матрица co-/cross- occurence LLR между треками
        """
        logger.info("Calculating co-/cross- occurence LLR matrix...")

        sc = ss.sparkContext

        lists_count = from_rdd.map(itemgetter(0)).distinct().count()
        bc_lists_count = sc.broadcast(lists_count)

        from_items_counts = from_rdd.map(itemgetter(1)).countByValue()
        bc_from_items_counts = sc.broadcast(from_items_counts)

        to_items_counts = to_rdd.map(itemgetter(1)).countByValue()
        bc_to_items_counts = sc.broadcast(to_items_counts)

        def llr_cell(x):
            i, j, cooc_count = x
            i_count = bc_to_items_counts.value[i]
            j_count = bc_from_items_counts.value[j]
            res_llr = llr_sqrt(cooc_count,
                               j_count - cooc_count,
                               i_count - cooc_count,
                               bc_lists_count.value - i_count - j_count + cooc_count)
            return i, j, res_llr

        llr_rdd = to_rdd \
            .join(from_rdd) \
            .map(lambda x: (x[1], 1)) \
            .reduceByKey(add) \
            .map(lambda x: (x[0][0], x[0][1], x[1])) \
            .map(llr_cell)

        logger.info("Co-/cross- occurence LLR matrix calculated")

        return llr_rdd

    @staticmethod
    def save_to_storage(matrix_saver, matrix_name: str, llr_rdd: RDD, index_rows: list):
        """
        Сохранение матрицы LLR в key-value хранилище по столбцам.
        Также сохраняет map объектов на индексы с названиями matrix_name+'_row'
        :param matrix_saver: Класс для сохранения матриц
        :param matrix_name: Название матрицы
        :param llr_rdd: Разреженная матрица co-/cross- occurence LLR между объектами
        :param index_rows: Массив с объектами строк, индексом является индекс в массиве
        """
        logger.info("Saving LLR matrix {0}...".format(matrix_name))

        llr_dict = llr_rdd \
            .map(lambda x: (x[1], {x[0]: x[2]})) \
            .reduceByKey(lambda a, b: {**a, **b}) \
            .collectAsMap()
        matrix_saver.save_matrix(matrix_name, llr_dict)

        matrix_saver.save_map(matrix_name, 'rows', index_rows)

        logger.info("LLR matrix {0} saved to storage".format(matrix_name))
