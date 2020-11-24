Обучение модели:
Для обучения используется скрипт script_for_learning из mel_learning.py.

script_for_learning(path = 'E:\\data\\test\\goznak',
                        save_model_to_file='noise_classify.pkl',
                        epochs=30,
                        limit=None)

Его параметры:
    path - Путь, по которому лежит набор спектрограмм для обучения и валидации
    save_model_to_file - Полное имя файла, в который надо сохранить модель
    epochs - Количество эпох для обучения
    limit - Параметр для ограничения количества спикеров,
            записи от которых попадают в обучающую и валидационную выборки


Использование модели:
Для классификации записи из файла используется скрипт classify_one из mel_running.py.

classify_one(filename, model=None, load_model_from_file='noise_classify.pkl')

Его параметры:
    filename - Имя файла с аудиозаписью
    model - заранее загруженная модель (если таковой нет, т.е. параметр =None, то модель загрузится из файла модели)
    load_model_from_file - Файл модели