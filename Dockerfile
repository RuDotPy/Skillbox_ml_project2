
# Используйте официальный образ Python
FROM python:3.10-slim-buster

# Установите рабочую директорию в /app
WORKDIR /app

# Копируйте текущий каталог в /app
COPY . /app

# Установите все необходимые пакеты из файла requirements.txt
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# Запустите train.py при запуске контейнера
CMD ["python", "train.py"]
