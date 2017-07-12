Все скритпы, лежащие здесь решают мелкие задачи, необходимые для обучения модели и получения результатов. Для любого из них можно получить помощь, запустив `python SCRIPT_NAME.py -h`.

Общее соглашение:
- Если аргумент является путём до файла, то он называется `ARGNAME_path` и его сокращение записывается как `SHORTNAMEp`.
- Если аргумент является путём до папки, то он называется `ARGNAME_dir` и его сокращение записывается как `SHORTNAMEd`.
- Все скрипты получают на вход путь до конфига.
- Предсказания хранятся вместе с идентификаторами объектов и хранят вероятности.
- Файлы с идентификаторами хранят один идентификатор на строку.

Скрипты:

| Имя | Описание| Аргументы |
| --- | ------- | --------- |
| train_model | Обучает модель.| Путь до идентификаторов обучающей и валидационной выборки, путь до места, куда сохранить модель и путь до места, куда писать логи tensorflow |
| predict | Использует обученную модель для предсказания на объектах. | Путь до идентификаторов, путь до обученной модели и путь, куда поместить предсказания. |
| find_threshold | Находит оптимальный порог для бинаризации, максимизируя dice score готовых предсказаний на объектах. | Путь до прогнозов, путь, куда поместить найденные пороги.
| binarize | Бинаризует предсказания по порогу. | Путь до предсказаний, путь до файла с порогами. |
| compute_dices | Считает dice scores на прогнозах. | Путь до прогнозов, путь, куда поместить dice scores |