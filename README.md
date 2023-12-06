# hw2_mlops
1. Добавлено хранилище S3 Minio (см. папку db, в ней docker-compose для поднятия контейнера и db.py для проверки подключения к Minio)
   - **cd db**
   - **docker-compose up**
3. В папке hw1:
     - папка app с кодом main.py (объединенные models.py и api.py)
     - requirements.txt - зависимости (до dvc)
     - Dockerfile
     - docker-compose для поднятия контейнера
     - **cd hw1**
     - **docker-compose up --build**
4. Добавлено версионирование с помощью DVC (см. папку datasets + https://github.com/ArinaPavlova/datasets)
   - DVC fixed.txt - файл, в котором описана последовательность выполненных команд для DVC.
   - config, config.local - конфиги для DVC
   - train.csv - датасет
   - train.csv.dvc - датасет после dvc add
