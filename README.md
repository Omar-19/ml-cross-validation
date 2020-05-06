## Кросс-валидация
Кросс-валидация заключается в разделении выборки на m непересекающихся блоков примерно одинакового размера, 
после чего выполняется m шагов. 
На i-м шаге i-й блок выступает в качестве тестовой выборки, 
объединение всех остальных блоков — в качестве обучающей выборки. 
Соответственно, на каждом шаге алгоритм обучается на некоторой обучающей выборке, 
после чего вычисляется его качество на тестовой выборке. 
После выполнения m шагов мы получаем m показателей качества, усреднение которых и дает оценку кросс-валидации.