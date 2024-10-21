import re
import string
from nltk.stem import PorterStemmer
from textblob import TextBlob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

#inisiasi spark
spark = SparkSession.builder.appName("dicoding_submission_sentiment_analytics").master("local[*]").getOrCreate()

#membaca file CSV ke Spark Dataframe
comments_df = spark.read.csv("thick_of_it_comments.csv", header=True, inferSchema=True)
comments_df.show()
comments_df.printSchema()
comments_df.summary().show()

# menghapus baris dengan nilai NaN
comments_df = comments_df.na.drop()

# menghapus duplikat
comments_df = comments_df.dropDuplicates()

# menampilkan dataframe setelah pembersihan
comments_df.show()
comments_df.printSchema()
comments_df.summary().show()

text_df = comments_df.select("text")
text_df.show()

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', '', text) # remove numbers


    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

# membuat UDF untuk preprocessing
cleaning_udf = udf(cleaningText, StringType())
casefolding_udf = udf(casefoldingText, StringType())
to_sentence_udf = udf(toSentence, StringType())

# menerapkan cleaning dan case folding
text_df = text_df.withColumn("cleaned_text", cleaning_udf(col("text")))
text_df = text_df.withColumn("final_text", casefolding_udf(col("cleaned_text")))

# tokenisasi
tokenizer = Tokenizer(inputCol="final_text", outputCol="tokens")
text_df = tokenizer.transform(text_df)

# penghapusan stopwords
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
text_df = remover.transform(text_df)

# inisalisasi fungsi untuk stemming
ps = PorterStemmer()

def stemming(tokens):
    return [ps.stem(token) for token in tokens]

# buat UDF untuk stemming
stemming_udf = udf(stemming, ArrayType(StringType()))

# terapkan stemming
df_stemmed = text_df.withColumn("stemmed_tokens", stemming_udf(col("filtered_tokens")))

# terapkan toSentence setelah mendapatkan stemmed_tokens
df_final = df_stemmed.withColumn("final_sentence", to_sentence_udf(col("stemmed_tokens")))

# tampilkan hasil akhir
df_final.select("text", "final_sentence").show(truncate=False)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"
    
polarity_udf = udf(get_sentiment, StringType())

df_labeled = df_final.withColumn("polarity", polarity_udf(col("final_sentence")))

# tampilkan hasil akhir
df_labeled.select("text", "final_sentence", "polarity").show(truncate=False)

# menghitung jumlah setiap kategori sentimen
polarity_counts = df_labeled.groupBy("polarity").count()

# tampilkan hasil
polarity_counts.show()

# Memilih kolom yang ingin disimpan
df_to_save = df_labeled.select("final_sentence", "polarity")

# Menggabungkan semua partisi menjadi satu
df_to_save.coalesce(1).write.csv("output_polarity_analysis", header=True, mode="overwrite")

spark.stop()