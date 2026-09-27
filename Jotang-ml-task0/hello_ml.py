students={"小明":95,"小张":90,"小李":80,"小许":85}
def average_score(scores):
    return sum(scores.values())/len(scores)
def highest_score(scores):
    best_student=""
    best_score=0
    for name in scores:
        if scores[name]>best_score:
            best_score=scores[name]
            best_student=name
    return best_score, best_student

average=average_score(students)
best_score, best_student=highest_score(students)

print(f"平均数为：{average:.2f}")
print(f"最高分为：{best_score},学生为：{best_student}")
import numpy as np
a=np.array([[1,2,3],
            [4,5,6]])
b=np.array([[7,8],
            [9,10],
            [11,12]])
c=a@b
print(f"a的形状为:{a.shape}")
print(f"b的形状为:{b.shape}")
print(f"c=a@b的结果为:\n{c}")
print(f"c的形状为:{c.shape}")
