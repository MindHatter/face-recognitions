# Алгоритмы распознавания лиц

##### Тестовый стенд:

1. Ubuntu 16.04
2. Python 3.5
3. База тренировочных лиц [Georgia Tech face database](http://www.anefian.com/research/face_reco.htm) 

##### Подготовка рабочего пространства:
1. Устанавливаем менеджер пакетов Python  
*sudo apt install python3-pip*
2. Устанавливаем модуль virtualenv для создания контейнера  
*sudo pip3 install virtualenv*
3. Создаем контейнер  
*virtualenv fr*
4. Активируем контейнер (все зависимые модули устанавливаются в контейнер)  
*. ./fr/bin/activate*
5. Создаем папку проекта:  
*mkdir who_is_it && cd who_is_it*

##### Структура проекта:
./who_is_it
-- train_data - папка с базой тренировочных лиц  
-- test_data - папка с фотографиями тестируемых лиц  
-- depends - папка с необходимыми зависимостями для каждого из способов  
-- fr_opencv.py - программа распознавания лица с использованием OpenCV  
-- fr_fr.py - программа распознавания лица с использованием face_recognition  

### Способ 1. Каскад Хаара и алгоритм локальных бинарных шаблонов
Первый способ базируется на библиотеке OpenCV, в которой уже реализованы алгоритмы детектирования и распознавания лиц.

##### Источники:
[https://www.superdatascience.com/opencv-face-detection/](https://www.superdatascience.com/opencv-face-detection/)
[https://www.superdatascience.com/opencv-face-recognition/](https://www.superdatascience.com/opencv-face-recognition/)

##### Установка зависимостей:
*pip install opencv-python  
pip install opencv-contrib-python  
pip install Pillow  
pip install numpy*

##### Использование:
Находясь в папке who_is_it, выполнить команду  
*python fr_opencv.py <test_photo_name.jpg>*

### Способ 2. Сверточная нейронная сеть 
Программа основана на использовании библиотек dlib для детектирования и face_recognition для распознавания лиц
 
##### Источники:
[https://towardsdatascience.com/facial-recognition-using-deep-learning-a74e9059a150](https://towardsdatascience.com/facial-recognition-using-deep-learning-a74e9059a150)
[https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)

##### Установка зависимостей:
*pip install numpy  
pip install scipy  
pip install dlib  
pip install face_recognition*

##### Использование:
Находясь в папке who_is_it, выполнить команду  
*python fr_fr.py <test_photo_name.jpg>*
