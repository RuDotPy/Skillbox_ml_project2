import os
import boto3
import pandas as pd
from io import StringIO

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5050'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Инициализация клиента
s3 = boto3.client('s3',
                  endpoint_url='http://localhost:9000',
                  aws_access_key_id='minio',
                  aws_secret_access_key='minio123')

# Считывание данных
obj = s3.get_object(Bucket='datasets', Key='kinopoisk_train.csv')
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(data))


# разбиение данных на train и test
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)

# Векторизация
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Предсказание
y_pred = clf.predict(X_test_vec)
print('Линейная регрессия обучилась, точность на проверочной выборке:', accuracy_score(y_test, y_pred))


# Создание баккита "mlflow"
try:
    s3.create_bucket(Bucket='mlflow')
    print('Баккит "mlflow" создан')
except s3.exceptions.BucketAlreadyOwnedByYou:
    print('Баккит "mlflow" уже существует')


# Настройка клиента boto3
boto3.setup_default_session(
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-west-1'  # или другой регион, если это применимо
)

# Логирование в MLflow
with mlflow.start_run() as run:
    # Логирование параметров и метрик
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Логирование модели
    mlflow.sklearn.log_model(clf, "model", registered_model_name="MyFirstModel")


