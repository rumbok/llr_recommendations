from .llr import LLR
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row, SparkSession, DataFrame


class LLRRec:
    """
    Рекомендательная система на основе Co-/cross- occurence матриц используя LLR (Lig-Likelihood ratio) метрику
    """
    @staticmethod
    def train_tracks_tracks(ss: SparkSession, events_df: DataFrame, matrix_saver, matrix_name):
        """Вычисление и сохранение co-occurence LLR матрицы"""

        user_track_row_df = StringIndexer(inputCol='track', outputCol='track_index')\
            .fit(events_df)\
            .transform(events_df)

        index_track = [f.metadata for f in user_track_row_df.schema.fields if f.name == "track_index"][0]["ml_attr"]["vals"]

        user_tracks_row_rdd = user_track_row_df\
            .rdd\
            .map(lambda x: (x.user, int(x.track_index)))

        user_track_col_rdd = events_df\
            .rdd\
            .map(lambda x: (x.user, x.track))

        llr_rdd = LLR.calc_llr(ss, user_track_col_rdd, user_tracks_row_rdd)
        LLR.save_to_storage(matrix_saver, matrix_name, llr_rdd, index_track)

    @staticmethod
    def train_tracks_artists(ss: SparkSession, events_df: DataFrame, tracks: dict, matrix_saver, matrix_name):
        """Вычисление и сохранение cross- occurence LLR матрицы: трек->исполнитель"""

        # Список артистов трека приходит в виде '{uuid,uuid, ...}'
        tracks_artists = {id: tracks[id].artists[1:-2].split(',') for id in tracks}
        bc_tracks_artists = ss.sparkContext.broadcast(tracks_artists)

        def row_user_artist(x) -> Row:
            user, track = x
            if track in bc_tracks_artists.value:
                return [Row(user=user, artist=artist) for artist in bc_tracks_artists.value[track]]
            else:
                return []

        user_track_rdd = events_df\
            .rdd\
            .map(lambda x: (x.user, x.track))

        user_artist_df = user_track_rdd \
            .flatMap(row_user_artist)\
            .toDF()
        user_artist_df = StringIndexer(inputCol='artist', outputCol='artist_index')\
            .fit(user_artist_df)\
            .transform(user_artist_df)\
            .drop('artist')
        user_artist_rdd = user_artist_df\
            .rdd\
            .map(lambda x: (x.user, int(x.artist_index)))

        index_artist = [f.metadata for f in user_artist_df.schema.fields if f.name == "artist_index"][0]["ml_attr"]["vals"]

        llr_rdd = LLR.calc_llr(ss, user_track_rdd, user_artist_rdd)
        LLR.save_to_storage(matrix_saver, matrix_name, llr_rdd, index_artist)
