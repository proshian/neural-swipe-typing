# Yandex Cup 2023 ML. NeuroSwipe task

## Размышления:
* Кажется, из многих обучающих примеров для default клавиатуры можно "деформировав пространство" сделать обучающие примеры для extra клавиатуры.
    * Интересно, нужно ли будет также деформировать время

Есть предположение, что опытный пользователь вводит слова очень быстро и может быть не глядя. Возможно, траектории опытного и  начинающего пользователя свайп клавиатры будут отличаться


Производные берутся так:
1. cur_x_derivative = (next_x - cur_x) - (cur_x - prev_x); cur_y_derivative = (next_y - cur_y) - (cur_y - prev_y)
2. для первого элемента последовательности производная равна 0 или 2 * (next_x - cur_x)
3. для последнего производные совпадают с предпоследним или 2 * (cur_y - prev_y)


## Идеи из других источников:

Исследователи из google описали метод генерации синтетических данных для данной задачи в статье (Modeling Gesture-Typing Movements)[https://www.tandfonline.com/doi/permissions/10.1080/07370024.2016.1215922]
1. Ставим опорные точки в центрах клавиш, соответствуующих буквам слова
2. Добавляем шум из нормального распределения
3. Строим тракеторию, минимизирующую рывок (третью производную координаты по времени)


В статье (Phrase-Gesture Typing on Smartphones)[https://cs.dartmouth.edu/~zheer/files/PhraseSwipe.pdf] предлагается исполоьзовать предобученный Bert в качестве Encoder'а. Координаты траектории свайпа транслируются в послежовательность ближайших клавиш (то есть последовательность букв). Последовательность букв токенизируется специальными токенами Bert'а

В статье [Joint Transformer/RNN Architecture for Gesture Typing in Indic Languages](https://aclanthology.org/2020.coling-main.87.pdf) в качестве энкодера multihead self attention -> BiLSTM. На входе последовательность векторов [x_coord, y_coord, dx/dt, dy/dt, one_hot_letter_of_nearest_keyboard_key]

## Вопросы
* Как токенизировать выходную последовательность? Пока планирую побуквенно.
* Если входную последовательность использовать как [x_coord, y_coord, dx/dt, dy/dt, one_hot_letter_of_nearest_keyboard_key], не нужно ли заменить one_hot на embedding?

## TODO
Low urgency:
* Synthetic data generation
    * analyse the percenta
    * WordGesture-GAN
    * Modeling Gesture-Typing Movements (minimal jerk)
* Read and understand visualization tool
* Read and understand baseline


## Synthetic data generation:
### Minimal-jerk
1. Берем центры букв
2. Сдвигаем с помощью двумерного гаусовского распределения
3. Берем готовую реализацию minimal-jerk trajectory

Реализации minimal-jerk:
* https://github.com/Amos-Chen98/optimal_trajectory_generator

Вопросы:
1. Есть ли у minimal-jerk параметры?
2. Должны ли minimal-jerk trajectory алгоритм знать сколько времени прошло между via-points?
3. Как от кривой-результата алгоритма получить последовательность точек с временным интервалом t?
4. Вычитать как получаются Гауссианы в статье (Modeling Gesture-Typing Movements)[https://www.tandfonline.com/doi/permissions/10.1080/07370024.2016.1215922]

Ошибку попадания в клавишу можно моделировать гаусовким распределением ([Predicting Finger-Touch Accuracy Based on the Dual Gaussian Distribution Model](https://dl.acm.org/doi/pdf/10.1145/2984511.2984546))

$$
\begin{align}
σ^2_x = 0.0075W^2_x + 1.68 \\
σ^2_y = 0.0108W^2_y + 1.33\\
µ_x = µ_y = 0
\end{align}
$$

Где $W_x$ - размер клавиши по $x$, $W_y$ - размер клавиши по $y$.

## Subtasks
* baseline creation
* finding papers
* reading papers
* finding datasets
* creating dataset


## Baseline

```shell
python ./src/keyboard_start/lib/main.py --train-path data/data/train.jsonl --test-path data/data/test.jsonl --voc-path data/data/voc.txt --num-workers 4 --output-path ./result.csv

python ./src/keyboard_start/tools/viz.py --path data/data/train.jsonl --limit 10 --ideal
```

## DVC commands

```shell
dvc init
dvc remote add -d myremote gdrive://1OvqjaZKpSib_m6gCs1QvkfLILXKPiEs3
dvc remote modify myremote gdrive_acknowledge_abuse true

dvc add data\data_preprocessed\gridname_to_grid.json 
dvc add data\data_preprocessed\valid__default_only.jsonl 
dvc add data\data_preprocessed\valid__extra_only.jsonl 
dvc add data\data_preprocessed\test.jsonl 
dvc add data\data_preprocessed\train__default_only_no_errors__2023_10_31__03_26_16.jsonl
dvc add data\data_preprocessed\train__extra_only_no_errors__2023_11_01__19_49_14.jsonl
dvc add data\data_preprocessed\voc.txt

cd data\data_preprocessed
git add gridname_to_grid.json.dvc valid__default_only.jsonl.dvc valid__extra_only.jsonl.dvc test.jsonl.dvc train__default_only_no_errors__2023_10_31__03_26_16.jsonl.dvc train__extra_only_no_errors__2023_11_01__19_49_14.jsonl.dvc voc.txt.dvc
cd ..\..


```

# Configs:
* Для get_individual_models_predictions.py config должен содержать список моделей, от которых требуется получить предсказания и путь до датасета
* Для aggregate_predictions.py config должен хранить:
    * тип аггрегирования
    * список импользуемых файлов с предсказаниями (будет конвертирован в словарь [имя_файла -> предсказания для датасета])
        * если тип аггрегирования = список должен быть в правильном порядке
    * если тип аггрегирования = weighted, должен храниться путь до файла, хранязего соответствие [имя_файла -> вес]

Может быть для любого типа аггрегирования сделать словарь init_kwargs. Для weighted = {'weights_path'}, для appendage = empty_dict

При заполнении config'а для get_individual_models_predictions нужно помнить, что max_steps_n = max_word_len + 1

Набросок для aggregation_params:
```json
"aggregation_params": 
    {
        "aggregation_type": "appendage",
        "preds":
        {
            "default": 
            [
                "m1_bigger__m1_bigger_v2__2023_11_12__14_51_49__0.13115__greed_acc_0.86034__default_l2_0_ls0_switch_2.pt.pkl",
                "m1_bigger__m1_bigger_v2__2023_11_12__12_30_29__0.13121__greed_acc_0.86098__default_l2_0_ls0_switch_2.pt.pkl",
                "m1_bigger__m1_bigger_v2__2023_11_11__22_18_35__0.13542_default_l2_0_ls0_switch_1.pt.pkl",
                "m1_v2__m1_v2__2023_11_09__10_36_02__0.14229_default_switch_0.pt.pkl",
                "m1_bigger__m1_bigger_v2__2023_11_12__00_39_33__0.13297_default_l2_0_ls0_switch_1.pt.pkl",
                "m1_bigger__m1_bigger_v2__2023_11_11__14_29_37__0.13679_default_l2_0_ls0_switch_0.pt.pkl"
            ],
            "extra": 
            [
                "m1_v2__m1_v2__2023_11_09__17_47_40__0.14301_extra_l2_1e-05_switch_0.pt.pkl",
                "m1_bigger__m1_bigger_v2__2023_11_12__02_27_14__0.13413_extra_l2_0_ls0_switch_1.pt.pkl"
            ]
        },
        "grid_name_to_aggregator_init_kwargs":
        {
            "default":
            {

            },
            "extra":
            {

            }
        },
        "output_path": "./data/submissions/id3_with_baseline_without_old_preds.csv"
    },
```