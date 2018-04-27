from .llr import LLR
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession, DataFrame, Row


class LLRTarget:
    """
    Система таргетирования пользователей на основе Co-/cross- occurence матриц используя LLR (Lig-Likelihood ratio) метрику
    """
    @staticmethod
    def train_artists_users(ss: SparkSession, events_df: DataFrame, tracks: dict, matrix_saver, matrix_name):
        """
        Вычисление и сохранение cross- occurence LLR матрицы: исполнитель->пользователь
        :param ss: Spark Session объект
        :param events_df: DataFrame со строками user, track
        :param tracks: Словарь с метаинформацией о треках
        :param matrix_saver: Класс для сохранения матриц
        """
        events_index_df = StringIndexer(inputCol='user', outputCol='user_index') \
            .fit(events_df) \
            .transform(events_df) \
            .drop('user_id')
        index_user = [f.metadata for f in events_index_df.schema.fields if f.name == "user_index"][0]["ml_attr"]["vals"]

        tracks_users_rdd = events_index_df\
            .rdd\
            .map(lambda x: (x.track, int(x.user_index)))

        # Список артистов трека приходит в виде '{uuid,uuid, ...}'
        tracks_artists = {id: tracks[id].artists[1:-2].split(',') for id in tracks}
        bc_tracks_artists = ss.sparkContext.broadcast(tracks_artists)

        def row_track_artist(track) -> ():
            if track in bc_tracks_artists.value:
                return [(track, artist) for artist in bc_tracks_artists.value[track]]
            else:
                return []

        tracks_artists_rdd = tracks_users_rdd \
            .map(lambda x: x[0])\
            .distinct()\
            .flatMap(row_track_artist)

        llr_rdd = LLR.calc_llr(ss, tracks_artists_rdd, tracks_users_rdd)

        LLR.save_to_storage(matrix_saver, matrix_name, llr_rdd, index_user)
