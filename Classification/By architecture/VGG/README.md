# VGG
## Информация

VGG - архитектура свёрточной нейронной сети, предложенная Karen Simonyan и Andrew Zisserman из Оксфордского университета в 2014 году. Данная архитектура практически выиграла
соревнование [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/2014/index.php) с результатом top-5 Error Rate – 7.3% (у
победителя - сети Inception V1 - результат составил 6.67%).

Данная архитектура имеет несколько различных интерпретаций:
<p align="center">
  <img src="https://github.com/Mark-Sherman-SE/machine-learning-projects/blob/master/docs/images/VGG%20configurations.png">
</p>

На соревновании ILSVRC использовалась интерпретация D или VGG16.
<p align="center">
  <img src="https://github.com/Mark-Sherman-SE/machine-learning-projects/blob/master/docs/images/VGG16.png">
</p>

## Реализация

Был разработан класс VGG на основе [туториала](https://jaketae.github.io/study/pytorch-vgg/) с небольшими доработками. Пользователь может выбрать одну из четырёх архитектур: VGG11, VGG13, 
VGG16, VGG19. В дополнении к оригинальной архитектуре была добавлена нормализация по батчам для стабилизации обучения и улучшения эффективности. Модель также способна обрабатывать 
прямоугольные изображения размера, кратного 32 (по высоте и ширине), а не только изображения 224x224 пикселя.

Для проверки работоспособности модель была протестирована на датасете CIFAR10.
