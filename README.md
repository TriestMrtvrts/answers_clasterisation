<h1 align="center">Хакатон-проект: кейс - Улучшение представлений результатов в сервисе "Мой Голос", Цифровой прорыв.Всеросс.2023, Команда ALT+F4 <div align="center"><a href="https://hacks-ai.ru/hackathons.html?eventId=969091&caseEl=1001711&tab=1"><img src="https://img.shields.io/badge/hackathon--project-d513eb"></a></div></h1>

## Описание проекта:

###  🧮 Описание задачи:
<p>
  <i>В Росатоме разработан и применяется на рынке сервис для проведения онлайн-опросов Мой Голос. Одним из способов получения данных является возможность ввода произвольных ответов на открытые вопросы. Для более качественной обработки и наглядной визуализации ответов сервис объединяет в группы пользовательские ответы, семантически близкие друг к другу. Участникам хакатона предстоит с помощью алгоритмов машинного обучения создать модель для объединения в группы ответов, схожих по смыслу с учетом тематики опроса, а также визуализировать результат объединения и предложить способы более наглядного представления результатов объединения для пользователей сервиса.</i>
</p>

<br />

###  🧾 Наше решение: Пайплайн + Технологии:
<details> 
<summary><b>Нормализация текста</b></summary>
   
##### <a href="https://github.com/TriestMrtvrts/answers_clasterisation/tree/main/answers_clustering-main-3/notebooks">участок кода, где нормализуется текст</a>

| Задача  | Технология для её решения |
| ------------- | ------------- |
| <i>исправление опечаток, дополнение слов</i>  | <i>модель FamSpell</i>  |
| <i>исправление опечаток связанных с неправильной раскладкой</i> | <i><a href="https://github.com/TriestMrtvrts/answers_clasterisation/blob/main/answers_clustering-main-3/module/process_dset.py#L98C1-L99C1">кастомная технология</a></i> |
| <i>фильтрация нецензурной лексики</i> | <i>исключение нецензурных слов, находящихся в <a href="https://github.com/TriestMrtvrts/answers_clasterisation/blob/main/answers_clustering-main-3/module/restricted.txt">словаре</a></i> |

</details>
<details>
   <summary><b>Кластеризация</b></summary>
   <br />
   <ul>
      <li><i>извлекаем скрытое состояние, используя модель FastText`а</i></li>
      <li><i>Уменьшаем размерность эмбедингов до 2 с помощью UMap</i></li>
      <li><i>Проводим Агломеративную Кластеризацию</i></li>
   </ul>

</details>

## Инструкция по запуску (Windows):

### 💼 1.  <b>Склонировать репозиторий:</b>
   
   ```bash
   git clone https://github.com/TriestMrtvrts/answers_clasterisation.git
   ```
   ```bash
   cd answers_clasterisation
   ```
### 💻 2.  <b>В папке склонированного репозитория выполнить команды:</b>

   ```bash
   cd module
   ```
   ```bash
   pip install -r requirements.txt
   ```
### 📂 3. <b>В папке <a href=https://github.com/TriestMrtvrts/answers_clasterisation/tree/main/answers_clustering-main-3/module>`module`</a>:</b>
  - Положить туда файл со своим <a href="https://github.com/TriestMrtvrts/answers_clasterisation/blob/main/answers_clustering-main-3/module/cropped.csv">`датасетом`</a><i>(по ссылке пример датасета)</i>
  - _создать новый файлик_
  - _прописать в нём_:
    
    ```python
    from process_dset import process_dset
    
    process_dset("./название датасета.csv")
    # --> pd.DataFrame
    # --> columns - answers, sentiment, cluster
    ```
    - пример файла <a href="https://github.com/TriestMrtvrts/answers_clasterisation/blob/main/answers_clustering-main-3/module/test.py#L3">здесь</a>
 ### ✅ 4. <b>Запустить программу:</b>
 
  ```bash
  python имя_созданного_файла.py
  ```
### Пример результата <a href="https://github.com/TriestMrtvrts/answers_clasterisation/blob/main/answers_clustering-main-3/module/result.csv">тут</a>
