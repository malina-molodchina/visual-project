

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#  head_length - длина острия стрелки:
ax.lines([0,1], [2, 0],

         width = 0.02,color="black")


#  Установим диапазон значений:
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
plt.axis("off")

plt.plot(-0.1,1,"o",color="black")


plt.show()