# 模板与多态
原本想要把这两个做成两部分，但是多态要依赖于模板部分，所以就放在一起了

## 模板
你需要把model类和矩阵类改成模板类，接收一个类型模板，代表了矩阵的元素类型，原本是float，现在改成模板传入的类型，model类里面的四个矩阵元素类型和model类的模板参数一致


## `if (你完成了part2)`
现在你可以用 `model<float>` 或 `model<double>` 来创建不同类型的model

而mnist-fc-plus文件夹里的矩阵就是double类型的，这一点通过json中的type字段说明了，同时plus版里的矩阵大小更大一些，所以你如果实现了json解析的话，你可以读取mnist-fc-plus文件夹中的矩阵参数验证一下

## 多态
你需要创建一个model基类，模板类model继承自这个model基类，这样外部在使用的时候就可以直接用model基类指针

## `if (你完成了part2)`
如果程序运行时传入的参数有plus，则从mnist-fc-plus文件夹中读取，你应该有一个构造model的函数，传入读取文件夹名，返回model基类指针